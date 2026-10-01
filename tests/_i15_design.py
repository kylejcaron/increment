"""Frozen seven-method inference-calibration design and independent references.

It imports no production code and runs no Monte Carlo. Regenerate the static
JSON with ``python -m tests._i15_design``; this only solves analytic population
equations and prospective binomial bounds.
Runtime evidence goes to pytest's per-test tmp_path, one i15-<case>.json.
"""

from __future__ import annotations

import json
import math
from functools import cache
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma, gammainc, gammaincc, roots_genlaguerre
from scipy.stats import beta, binom, norm

from tests.mc import binomial_error_upper_bound, family_eta, scientific_delta

PRODUCTION_INFERENCE_CANDIDATE = {
    "winsor_method": "positive-log-kernel-bootstrap-t-v1",
    "budget": 1999,
    "pilot": "Independent arm lognormal-kernel mixtures; h=1.06*sample_sd(log(Y))*n^(-1/5)",
    "resampling": "Fixed all-arm counts; type-7 cutoff, clipped means, log-kernel density and "
    "full centered influence studentization recomputed in every replicate",
    "centering": "Pilot population allocation-pooled quantile and analytic winsor means",
    "rng": "PCG64DXSM-SeedSequence-v1; seed=1729, stream=outer replication ordinal, "
    "spawn_key=(stream,canonical_arm,0 for centers or 1 for noise); fixed chunks of 64; "
    "exhaustive 70-label production check holds stream=0 for all labels",
    "inversion": "B=1999; tail rank floor((B+1)*alpha/2); opposite order statistics; unresolved "
    "tails unbounded, any failed root unavailable; never retry/drop",
    "ordinary_means": "independent-means-welch-v2; sum_g a_g^2*SS_g/[n_g*(n_g-1)]; "
    "df=V^2/sum_g(v_g^2/(n_g-1))",
    "scope": "Pointwise asymptotic candidate; finite-grid calibration unexecuted. All historical "
    "reference_derivation and known issue records below describe the preserved "
    "baseline.",
}

FAMILY_ALPHA = 0.01
# Case enumeration axes. The slot census below counts from these same axes, so
# adding a winsor cell cannot leave eta behind, and dropping one cannot leave a
# reserved slot behind either.
WINSOR_DISTRIBUTIONS = ("ln-s0.5", "ln-s1.6", "ln-s2.0", "contamination", "gamma-k2")
WINSOR_AA_AXES = ((50, 200, 500, 2000), ((1, 1), (1, 4), (4, 1)), (0.95, 0.99))
WINSOR_ALTERNATIVE_AXES = ((50, 2000), ((1, 1), (1, 4), (4, 1)), (0.95, 0.99))
WINSOR_AA_CASES = len(WINSOR_DISTRIBUTIONS) * math.prod(len(axis) for axis in WINSOR_AA_AXES)
WINSOR_ALTERNATIVE_CASES = math.prod(len(axis) for axis in WINSOR_ALTERNATIVE_AXES)
WINSOR_EXACT_CASES = 1
COUNTS = {
    "normal_posterior": 1,
    "lift_interval": 1,
    "ate_interval": 3,
    "binomial_end_to_end": 1,
    "pooled_winsor": WINSOR_AA_CASES + WINSOR_ALTERNATIVE_CASES + WINSOR_EXACT_CASES,
    "one_sided_lift": 1,
    "one_sided_ate": 1,
}
TOTAL_CASES = sum(COUNTS.values())

# prose: allow-long Bonferroni slot census derives each *_BOUNDS constant and eta
# Bonferroni slots count each distinct bound a gate reads at level eta, never
# each gate or case. Gates follow _CaseAccumulator.report
# (tests.estimation.test_inference_calibration):
#
#   * A coverage verdict's error upper bound, coverage lower bound and MC margin
#     read one beta quantile (tests.mc.coverage_lower_bound is an identity), so
#     the unconditional and conditional verdicts are two slots.
#   * finite_width, useful_width and availability read one bound each.
#   * conditional_precision_design and additive_precision_design compare
#     integer counts and read no bound.
#   * The additive sidecar, power and independent binary reference count only
#     on cases that declare them.
#
# Cases sharing an outer draw stream still gate separately, so shared draws
# never merge slots. family_eta spends alpha/(2m) per slot, so the m bounds
# union to alpha/2; that factor-of-two slack is not available for spending.
COVERAGE_BOUNDS = 2
WIDTH_BOUNDS = 2
AVAILABILITY_BOUNDS = 1
BASE_BOUNDS = COVERAGE_BOUNDS + WIDTH_BOUNDS + AVAILABILITY_BOUNDS
ADDITIVE_BOUNDS = 4
POWER_BOUNDS = 1
INDEPENDENT_REFERENCE_BOUNDS = 1
CENSUS_CASES = {
    "winsor_aa": WINSOR_AA_CASES,
    "winsor_alternative": WINSOR_ALTERNATIVE_CASES,
    "other_random": TOTAL_CASES - COUNTS["pooled_winsor"],
    # Extra slot on top of this case's base gates, which "other_random" counts.
    "binomial_independent_reference": COUNTS["binomial_end_to_end"],
    # Exhaustive C(8,4) enumeration: every label is visited, so no gate on it
    # reads a Monte-Carlo bound and it reserves nothing.
    "winsor_exact_labels": WINSOR_EXACT_CASES,
}
BOUNDS_PER_CASE = {
    "winsor_aa": BASE_BOUNDS + ADDITIVE_BOUNDS,
    "winsor_alternative": BASE_BOUNDS + ADDITIVE_BOUNDS + POWER_BOUNDS,
    "other_random": BASE_BOUNDS,
    "binomial_independent_reference": INDEPENDENT_REFERENCE_BOUNDS,
    "winsor_exact_labels": 0,
}
BOUND_CENSUS = {group: count * BOUNDS_PER_CASE[group] for group, count in CENSUS_CASES.items()}
GATED_BOUNDS = sum(BOUND_CENSUS.values())
ETA = family_eta(FAMILY_ALPHA, GATED_BOUNDS)


def _error_upper_bounds(errors, counts):
    """Vectorized :func:`tests.mc.binomial_error_upper_bound`, both branches.

    ``beta.isf(eta, k+1, n-k)`` elementwise, and exactly ``1.0`` where
    ``k == n``, because no finite beta quantile exists at that domain edge.
    This exists only so the smallest certifying count can be found by
    exhaustive scan; :func:`repetition_design` re-reads the count it returns
    through the scalar helper and refuses to freeze a design if the two
    readings ever disagree about the three conditions.
    """
    bounds = np.ones(counts.shape, dtype=float)
    interior = errors < counts
    bounds[interior] = beta.isf(ETA, errors[interior] + 1, counts[interior] - errors[interior])
    return bounds


def _certifying_mask(counts, q: float, delta: float, margin_limit: float):
    """Which candidate *counts* satisfy all three frozen conditions."""
    k = np.ceil(counts * (q + delta / 2)).astype(np.int64)
    worst = np.ceil(counts * (q + delta)).astype(np.int64)
    return (
        (_error_upper_bounds(worst, counts) - worst / counts <= margin_limit)
        & (_error_upper_bounds(k, counts) <= q + delta)
        & (binom.sf(k, counts, q) <= ETA)
    )


def _certification(n: int, q: float, delta: float) -> dict:
    """The three frozen quantities at count *n*, through the scalar helper."""
    k = math.ceil(n * (q + delta / 2))
    worst = math.ceil(n * (q + delta))
    return {
        "nominal_acceptance_count": k,
        "prospective_mc_margin": binomial_error_upper_bound(worst, n, ETA) - worst / n,
        "error_upper": binomial_error_upper_bound(k, n, ETA),
        "nominal_false_failure_bound": float(binom.sf(k, n, q)),
    }


def _certifies(values: dict, q: float, delta: float, margin_limit: float) -> bool:
    return (
        values["prospective_mc_margin"] <= margin_limit
        and values["error_upper"] <= q + delta
        and values["nominal_false_failure_bound"] <= ETA
    )


def _smallest_certifying_count(q: float, delta: float, margin_limit: float, block: int = 8192):
    """Smallest n >= 1 satisfying the three conditions.

    Both ceilings make the conditions sawtoothed in n, so no bisection is
    admissible: every count below the answer is evaluated and rejected. The
    scan walks contiguous blocks from n=1 and stops inside the first block
    that contains a satisfying count, so the result is minimal by
    construction rather than by an assumed monotonicity.
    """
    first = 1
    while True:
        counts = np.arange(first, first + block, dtype=np.int64)
        certifying = _certifying_mask(counts, q, delta, margin_limit)
        if certifying.any():
            return int(counts[int(np.argmax(certifying))])
        first += block


@cache
def repetition_design(q: float) -> dict:
    """Choose n without data; certify precision and nominal acceptance power.

    k*=ceil(n*(q+delta/2)). Require CP_U(k*) <= q+delta and
    P_q(K>k*) <= eta. Also require CP margin <= delta/2 at the
    largest potentially passing error rate q+delta. n is the smallest count
    meeting all three, with no outcome-dependent stopping or reseeding: a
    geometric ladder would freeze the first rung past the requirement and
    charge the campaign for the overshoot.
    """
    delta = scientific_delta(q)
    margin_limit = min(delta / 2, 0.00125) if q == 0.025 else delta / 2
    n = _smallest_certifying_count(q, delta, margin_limit)
    values = _certification(n, q, delta)
    if not _certifies(values, q, delta, margin_limit):
        raise ValueError(f"count {n} fails the frozen conditions at q={q}: {values}")
    if n > 1 and _certifies(_certification(n - 1, q, delta), q, delta, margin_limit):
        raise ValueError(f"count {n} is not minimal at q={q}: {n - 1} also satisfies them")
    return {
        "repetitions": n,
        "nominal_error": q,
        "scientific_delta": delta,
        "error_upper_max": q + delta,
        "coverage_lower_min": 1 - q - delta,
        "mc_margin_max": margin_limit,
        "prospective_mc_margin": values["prospective_mc_margin"],
        "margin_at_nominal": binomial_error_upper_bound(math.ceil(n * q), n, ETA)
        - math.ceil(n * q) / n,
        "nominal_acceptance_count": values["nominal_acceptance_count"],
        "nominal_false_failure_bound": values["nominal_false_failure_bound"],
        "eta_per_direction": ETA,
        "formula": (
            "smallest n>=1 with: "
            "k=ceil(n*(q+delta/2)); CP_U(k,n,eta)<=q+delta; "
            "Binom.sf(k,n,q)<=eta; "
            "CP_U(ceil(n*(q+delta)),n,eta)-ceil(n*(q+delta))/n<=delta/2"
        ),
        "nominal_band": None,
        "nominal_band_reason": "Scientific tolerance is separate from MC uncertainty.",
    }


def lognormal(mu: float, sigma: float) -> dict:
    return {"family": "lognormal", "mu": mu, "sigma": sigma}


def raw_moment(distribution: dict, order: int) -> float:
    """Completing the Gaussian square gives exp(k*mu+k^2*sigma^2/2).

    Gamma(shape=2,scale=b) has E[X^k]=b^k Gamma(2+k)/Gamma(2).
    Mixture moments are the weighted component moments; all are finite.
    """
    if distribution["family"] == "lognormal":
        return math.exp(order * distribution["mu"] + order**2 * distribution["sigma"] ** 2 / 2)
    if distribution["family"] == "gamma":
        return distribution["scale"] ** order * math.gamma(2 + order)
    return sum(w * raw_moment(d, order) for w, d in distribution["components"])


def describe(distribution: dict) -> dict:
    mean = raw_moment(distribution, 1)
    second = raw_moment(distribution, 2)
    return {**distribution, "mean": mean, "second_moment": second, "variance": second - mean**2}


def population_parts(distribution: dict, cutoff: float) -> tuple[float, float, float, float]:
    """Return F(c), f(c), E[min(X,c)], E[min(X,c)^2] independently.

    LN truncated moment: exp(k*mu+k^2*s^2/2)*Phi((log(c)-mu-k*s^2)/s).
    Gamma truncated moment: b^k Gamma(2+k)*P(2+k,c/b).
    Add c^k*S(c) for winsorization; use survival tails directly.
    """
    kind = distribution["family"]
    if kind == "mixture":
        parts = [(w, population_parts(d, cutoff)) for w, d in distribution["components"]]
        return tuple(sum(w * p[j] for w, p in parts) for j in range(4))
    if kind == "lognormal":
        mu, sigma = distribution["mu"], distribution["sigma"]
        z = (math.log(cutoff) - mu) / sigma
        cdf, sf = float(norm.cdf(z)), float(norm.sf(z))
        density = float(norm.pdf(z)) / (cutoff * sigma)
        moments = [
            raw_moment(distribution, k) * float(norm.cdf(z - k * sigma)) + cutoff**k * sf
            for k in (1, 2)
        ]
    else:
        scale = distribution["scale"]
        x = cutoff / scale
        cdf, sf = float(gammainc(2, x)), float(gammaincc(2, x))
        density = x * math.exp(-x) / scale
        moments = [
            raw_moment(distribution, k) * float(gammainc(2 + k, x)) + cutoff**k * sf for k in (1, 2)
        ]
    return cdf, density, moments[0], moments[1]


def pooled_reference(control: dict, treatment: dict, nc: int, nt: int, quantile: float) -> dict:
    """Independent population target, including cutoff estimation covariance.

    w_g=n_g/N, H=sum(w_g F_g), H(c)=p, f=H'(c), W=min(X,c).
    For theta=sum(a_g*m_g) locally, A=sum(a_g*S_g(c)), each arm's
    influence is a_g*(W-m_g)+A*w_g*(F_g(c)-1{X<=c})/f.
    Cov(W,F_g-1{X<=c})=S_g(c)*(c-m_g), Var(indicator)=F_g*S_g.
    Sum each influence variance / n_g, retaining both cross terms.
    Difference: a=(-1,1); log ratio: a=(-1/m_c,1/m_t).
    The A/A cancellation (A=0) does not apply to heterogeneous arms.

    The CDF and truncated moments are analytic; only inversion of H is
    numerical (Brent relative tolerance 1e-13, absolute tolerance 1e-12).
    First-order variances do not remove finite-N cutoff bias; actual
    production coverage is always tested against the population target.
    """
    weights = [nc / (nc + nt), nt / (nc + nt)]
    distributions = [control, treatment]

    def objective(c):
        return (
            sum(w * population_parts(d, c)[0] for w, d in zip(weights, distributions, strict=True))
            - quantile
        )

    high = max(raw_moment(d, 1) for d in distributions)
    while objective(high) < 0:
        high *= 2
    cutoff = float(brentq(objective, 1e-12, high, xtol=1e-12, rtol=1e-13))
    parts = [population_parts(d, cutoff) for d in distributions]
    means = [p[2] for p in parts]
    density = sum(w * p[1] for w, p in zip(weights, parts, strict=True))
    out = {
        "population_cutoff": cutoff,
        "mixture_cdf_residual": objective(cutoff),
        "pool_weights": weights,
        "winsorized_means": means,
        "difference": means[1] - means[0],
        "relative_lift": means[1] / means[0] - 1,
        "raw_mean_difference_not_target": raw_moment(treatment, 1) - raw_moment(control, 1),
        "finite_sample_bias": "Not removed or treated as MC noise; unconditional coverage gates it.",
    }
    for name, coefficients in (
        ("difference", [-1, 1]),
        ("log_ratio", [-1 / means[0], 1 / means[1]]),
    ):
        derivative = sum(a * (1 - p[0]) for a, p in zip(coefficients, parts, strict=True))
        fixed = cutoff_term = covariance = 0.0
        for a, w, p, n in zip(coefficients, weights, parts, (nc, nt), strict=True):
            cdf, _, mean, second = p
            sf = 1 - cdf
            b = derivative * w / density
            fixed += a * a * (second - mean * mean) / n
            cutoff_term += b * b * cdf * sf / n
            covariance += 2 * a * b * sf * (cutoff - mean) / n
        out[name + "_variance"] = {
            "fixed_cutoff": fixed,
            "cutoff_indicator": cutoff_term,
            "cross_covariance": covariance,
            "full": fixed + cutoff_term + covariance,
            "cutoff_derivative": derivative,
            "full_to_fixed_ratio": (fixed + cutoff_term + covariance) / fixed,
            "normal_cutoff_limiting_noncoverage": float(
                2 * norm.sf(norm.isf(0.025) * math.sqrt(fixed / (fixed + cutoff_term + covariance)))
            ),
        }
    return out


def sample(rng, distribution: dict, n: int):
    kind = distribution["family"]
    if kind == "lognormal":
        return rng.lognormal(distribution["mu"], distribution["sigma"], n)
    if kind == "gamma":
        return rng.gamma(2, distribution["scale"], n)
    # Fixed mixture, independent of labels and outcomes; no infinite-moment tails.
    weights, components = zip(*distribution["components"], strict=True)
    labels = rng.choice(len(weights), size=n, p=weights)
    values = np.empty(n)
    for j, component in enumerate(components):
        mask = labels == j
        values[mask] = sample(rng, component, int(mask.sum()))
    return values


def exact_label_reference(raw, nc: int, quantile: float):
    """All C(N,nc) labels on one pooled, linearly winsorized sample.

    Equal raw distributions imply exchangeable labels conditional on pooled
    values; the pooled cutoff is label-invariant. Equal transformed means
    under different distributions is only a weak null, not this sharp null.
    Ties use the conservative >= tail; no artificial rejection lower bound.
    This oracle calls neither the production estimator nor its p-values.
    """
    raw = np.asarray(raw, dtype=float)
    cutoff = float(np.quantile(raw, quantile, method="linear"))
    values = np.minimum(raw, cutoff)
    assignments = tuple(combinations(range(len(values)), nc))
    statistics = []
    for indices in assignments:
        mask = np.zeros(len(values), dtype=bool)
        mask[list(indices)] = True
        statistics.append(abs(float(values[~mask].mean() - values[mask].mean())))
    # A deterministic rounding allowance includes, rather than excludes, ties.
    tolerance = 32 * np.finfo(float).eps * float(np.max(np.abs(values)))
    pvalues = [sum(t >= s - tolerance for t in statistics) / len(statistics) for s in statistics]
    return assignments, pvalues, cutoff


def unequal_normal_reference(nc: int, nt: int) -> dict:
    """Integrate the normal-mean / independent chi-square variance pivot.

    The actual score SE squares to Vhat=U_c/n_c^2+4*U_t/n_t^2,
    U_g~chi2(n_g-1), while Var(point)=1/n_c+4/n_t. Conditional
    noncoverage is 2*Phi(-z*sqrt(Vhat/Var(point))). Integrate using
    generalized Gauss-Laguerre at 64 and 128 nodes, without sampling.
    """
    values = []
    for nodes in (64, 128):
        xc, wc = roots_genlaguerre(nodes, (nc - 1) / 2 - 1)
        xt, wt = roots_genlaguerre(nodes, (nt - 1) / 2 - 1)
        vhat = 2 * xc[:, None] / nc**2 + 8 * xt[None, :] / nt**2
        tails = 2 * norm.sf(norm.isf(0.025) * np.sqrt(vhat / (1 / nc + 4 / nt)))
        weights = (wc / gamma((nc - 1) / 2))[:, None] * (wt / gamma((nt - 1) / 2))[None, :]
        values.append(float(np.sum(weights * tails)))
    return {
        "truth": 0.05,
        "point_variance": 1 / nc + 4 / nt,
        "analytic_noncoverage": values[-1],
        "quadrature_64_128_difference": abs(values[1] - values[0]),
        "formula": "E[2*norm.sf(z_.025*sqrt((chi2(nc-1)/nc^2+4*chi2(nt-1)/nt^2)/(1/nc+4/nt)))]; independent chi-squares",
    }


def build_manifest() -> dict:
    records = []

    def add(identifier, family, seed, dgp, q=0.05, truth=None, reference=None, **extra):
        design = dict(repetition_design(q))
        # Conditional coverage needs enough available draws even at the allowed
        # 1% unavailability boundary; no denominator-dependent extension.
        design["conditional_required_repetitions"] = design["repetitions"]
        design["repetitions"] = math.ceil(design["repetitions"] / 0.99)
        design["formula"] += "; total reps=ceil(required_conditional_reps/.99)"
        design["conditional_nominal_acceptance_count"] = design["nominal_acceptance_count"]
        design["nominal_acceptance_count"] = math.ceil(
            design["repetitions"] * (q + scientific_delta(q) / 2)
        )
        design["nominal_false_failure_bound"] = float(
            binom.sf(
                design["nominal_acceptance_count"],
                design["repetitions"],
                q,
            )
        )
        design["margin_at_nominal"] = (
            binomial_error_upper_bound(
                math.ceil(design["repetitions"] * q),
                design["repetitions"],
                ETA,
            )
            - math.ceil(design["repetitions"] * q) / design["repetitions"]
        )
        design["prospective_mc_margin"] = (
            binomial_error_upper_bound(
                math.ceil(design["repetitions"] * (q + scientific_delta(q))),
                design["repetitions"],
                ETA,
            )
            - math.ceil(design["repetitions"] * (q + scientific_delta(q))) / design["repetitions"]
        )
        records.append(
            {
                "case_id": identifier,
                "family": family,
                "seed": seed,
                "dgp": dgp,
                "design": design,
                "truth": truth,
                "reference": reference,
                "availability": {
                    "applicable": True,
                    "minimum_probability": 0.99,
                    "observed": None,
                    "refusal_reasons": None,
                    "policy": "No draw filtering; production refusals are unconditional misses and recorded by reason.",
                },
                "nonvacuity": {
                    "finite_width_min": 0.99,
                    "useful_width_min": 0.95,
                    "width_cap_rule": "4 times independent nominal full width",
                    "power_min": None,
                },
                "results": {
                    "coverage": None,
                    "conditional_coverage": None,
                    "error_rate": None,
                    "width": None,
                    "power": None,
                    "mc_margin": None,
                },
                **extra,
            }
        )

    add(
        "normal-seed101",
        "normal_posterior",
        101,
        {
            "distribution": "normal_estimate",
            "truth": 0.1,
            "se": 0.05,
            "n_c": None,
            "n_t": None,
            "allocation": None,
            "prior_mu": 0,
            "prior_sigma": 1e6,
        },
        truth=0.1,
        reference="Independent precision-addition conjugate formula; near-flat fixed-truth coverage.",
    )
    log_dgp = {
        "distribution": "independent_normal_logs",
        "log_c": math.log(2),
        "log_rr": 0.1,
        "se_t": 0.04,
        "se_c": 0.03,
        "n_c": None,
        "n_t": None,
        "allocation": None,
    }
    score_dgp = {
        "distribution": "normal_scores",
        "mean": 0.05,
        "sigma": 1.0,
        "n": 200,
        "n_c": None,
        "n_t": None,
        "allocation": None,
    }
    add(
        "lift-seed202",
        "lift_interval",
        202,
        log_dgp,
        truth=math.expm1(0.1),
        reference="Independent normal log ratio: variance .04^2+.03^2; monotone expm1.",
    )
    add(
        "ate-seed303",
        "ate_interval",
        303,
        score_dgp,
        truth=0.05,
        reference="sqrt(200/199)*t_199 pivot; normal cutoff retained, no replacement t interval.",
    )
    for seed, nc, nt in ((910, 50, 200), (911, 200, 50)):
        add(
            f"ate-unequal-{nc}-{nt}",
            "ate_interval",
            seed,
            {
                "distribution": "two_normal_arms",
                "n_c": nc,
                "n_t": nt,
                "allocation": [nc // 50, nt // 50],
                "mu_c": 2.0,
                "mu_t": 2.05,
                "sigma_c": 1.0,
                "sigma_t": 2.0,
            },
            truth=0.05,
            reference=unequal_normal_reference(nc, nt),
        )
        if (nc, nt) == (200, 50):
            records[-1]["acceptance_status"] = (
                "NOT complete: normal score interval noncoverage .057319 exceeds .055."
            )
    add(
        "binomial-seed707",
        "binomial_end_to_end",
        707,
        {
            "distribution": "bernoulli",
            "p_c": 0.15,
            "p_t": 1.1 * 0.15,
            "n_c": 1000,
            "n_t": 1000,
            "allocation": [1, 1],
            "variance_c": 0.15 * 0.85,
            "variance_t": 0.165 * 0.835,
        },
        truth=0.1,
        reference="I12 tests.estimation.test_rare_event_calibration.bonferroni_lift_interval; same counts, separate rare-event grid.",
    )
    add(
        "one-lift-seed404",
        "one_sided_lift",
        404,
        log_dgp,
        q=0.025,
        truth=math.expm1(0.1),
        reference="Normal log ratio; greater alpha=.025 displays two-sided alpha=.05.",
    )
    add(
        "one-ate-seed505",
        "one_sided_ate",
        505,
        score_dgp,
        q=0.025,
        truth=0.05,
        reference="sqrt(200/199)*t_199 pivot; alpha doubling retained.",
    )

    contamination = {
        "family": "mixture",
        "components": [
            [0.98, lognormal(1.0, 0.5)],
            [0.02, lognormal(4.0, 1.0)],
        ],
    }
    library = {f"ln-s{sigma}": lognormal(1.0, sigma) for sigma in (0.5, 1.6, 2.0)}
    library["contamination"] = contamination
    library["gamma-k2"] = {"family": "gamma", "shape": 2, "scale": 2.0}
    # Indexed by the censused label tuple, so a distribution the census does
    # not know about cannot enter the grid, and one it does know about cannot
    # be dropped from it.
    distributions = [(label, library[label]) for label in WINSOR_DISTRIBUTIONS]
    for index, (n, allocation, p, (label, d)) in enumerate(product(*WINSOR_AA_AXES, distributions)):
        nc, nt = n * allocation[0], n * allocation[1]
        original = (n, allocation, p, label) == (500, (1, 1), 0.99, "ln-s1.6")
        identifier = (
            "winsor-original-seed808"
            if original
            else f"winsor-aa-{label}-n{n}-{allocation[0]}x{allocation[1]}-p{p}"
        )
        target = pooled_reference(d, d, nc, nt, p)
        add(
            identifier,
            "pooled_winsor",
            808 if original else 15000 + index,
            {
                "n_c": nc,
                "n_t": nt,
                "baseline_n": n,
                "allocation": allocation,
                "quantile": p,
                "quantile_method": "linear",
                "control": describe(d),
                "treatment": describe(d),
            },
            truth=0.0,
            reference=target,
            kind="aa",
        )
    for index, (n, allocation, p) in enumerate(product(*WINSOR_ALTERNATIVE_AXES)):
        nc, nt = n * allocation[0], n * allocation[1]
        control, treatment = lognormal(1.0, 1.6), lognormal(1.5, 2.0)
        target = pooled_reference(control, treatment, nc, nt, p)
        add(
            f"winsor-alt-n{n}-{allocation[0]}x{allocation[1]}-p{p}",
            "pooled_winsor",
            16000 + index,
            {
                "n_c": nc,
                "n_t": nt,
                "baseline_n": n,
                "allocation": allocation,
                "quantile": p,
                "quantile_method": "linear",
                "control": describe(control),
                "treatment": describe(treatment),
            },
            truth=target["relative_lift"],
            reference=target,
            kind="alternative",
            acceptance_status="Unresolved production cutoff covariance omission",
        )
        records[-1]["nonvacuity"]["power_min"] = 0.5 if n == 2000 else 0.05
    add(
        "winsor-exact-labels",
        "pooled_winsor",
        None,
        {
            "distribution": "conditional_exchangeable_labels",
            "raw_pooled_values": list(range(1, 9)),
            "n_c": 4,
            "n_t": 4,
            "allocation": [1, 1],
            "quantile": 0.99,
            "quantile_method": "linear",
        },
        truth=0.0,
        reference="All 70 label subsets; independent absolute mean-difference permutation p-values.",
        kind="exact",
    )
    records[-1]["design"] = {
        "repetitions": 70,
        "formula": "C(8,4), exhaustive, no MC",
        "nominal_error": 0.05,
        "scientific_delta": 0.005,
        "error_upper_max": 0.055,
        "coverage_lower_min": 0.945,
        "mc_margin_max": 0.0,
        "prospective_mc_margin": 0.0,
    }
    records[-1]["nonvacuity"] = {
        "finite_width_min": 1.0,
        "useful_width_min": 1.0,
        "width_cap_rule": "All log widths <= 8*z_.025*sqrt(s_pooled^2*(1/4+1/4)/mean_pooled^2); exact oracle rejects 2/70 labels.",
        "power_min": None,
    }
    # The slot census is only honest if it counts the cases actually built and
    # the gates they actually declare.
    alternative = [c["case_id"] for c in records if c.get("kind") == "alternative"]
    powered = [c["case_id"] for c in records if c["nonvacuity"]["power_min"] is not None]
    if powered != alternative:
        raise ValueError(f"power gates {powered} are not the censused alternative cases")
    built = {
        "winsor_aa": sum(1 for c in records if c.get("kind") == "aa"),
        "winsor_alternative": len(alternative),
        "other_random": sum(1 for c in records if c["family"] != "pooled_winsor"),
        "binomial_independent_reference": sum(
            1 for c in records if c["family"] == "binomial_end_to_end"
        ),
        "winsor_exact_labels": sum(1 for c in records if c.get("kind") == "exact"),
    }
    if built != CENSUS_CASES:
        raise ValueError(f"case enumeration {built} disagrees with the slot census {CENSUS_CASES}")
    return {
        "task": "I15",
        "executed": False,
        "production_inference_candidate": dict(PRODUCTION_INFERENCE_CANDIDATE),
        "execution_contract": {
            "version": "campaign-cli-v2",
            "random_outer_repetitions": sum(
                c["design"]["repetitions"] for c in records if c.get("kind") != "exact"
            ),
            "winsor_inner_replicates": 1999
            * sum(
                c["design"]["repetitions"]
                for c in records
                if c.get("kind") in ("aa", "alternative")
            ),
            "release_gate": "make test-all plus a complete run_i15_campaign --selection full evidence bundle",
            "acceptance": "Every frozen gate and repetition must pass; deterministic parity and exhaustive labels are necessary but insufficient.",
            "sampling": "Winsor draws sample control then treatment from the original case DGP; each draw invokes the captured-array estimator with its outer ordinal as bootstrap stream, without batching.",
        },
        "counts": COUNTS,
        "total_cases": TOTAL_CASES,
        "family_alpha": FAMILY_ALPHA,
        "gated_bounds": GATED_BOUNDS,
        "bound_census": BOUND_CENSUS,
        "bounds_per_case": BOUNDS_PER_CASE,
        "eta": ETA,
        "runtime_artifacts": "run_i15_campaign output bundle with immutable manifest, shard summaries, and completion status",
        "reference_derivation": {
            "raw_moments": "LN: E[X^k]=exp(k*mu+k^2*sigma^2/2); Gamma(2,b): b^k*Gamma(2+k); mixtures: weighted sums.",
            "cutoff": "H(c)=w_c*F_c(c)+w_t*F_t(c)=p; w_g=n_g/(n_c+n_t). Analytic CDF, Brent xtol=1e-12, rtol=1e-13.",
            "winsor_moments": "E[min(X,c)^k]=E[X^k*1{X<=c}]+c^k*S(c). LN truncated moment=exp(k*mu+k^2*sigma^2/2)*Phi((log(c)-mu-k*sigma^2)/sigma); gamma=b^k*Gamma(2+k)*P(2+k,c/b).",
            "targets": "m_t(c)-m_c(c) for additive sidecar; m_t(c)/m_c(c)-1 for relative interval.",
            "influence": "a_g*(W-m_g)+b_g*(F_g(c)-1{X<=c}); b_g=A*w_g/f; A=sum(a_g*S_g(c)); a=(-1,1) or (-1/m_c,1/m_t).",
            "variance": "sum_g [a_g^2*Var(W)+b_g^2*F_g*S_g+2*a_g*b_g*S_g*(c-m_g)]/n_g. A/A: A=0; heterogeneous arms: generally A!=0.",
            "bias": "Finite-N estimated-cutoff bias is not corrected or excused. The full IF sets width benchmarks; actual unchanged production intervals must cover independent population truth.",
            "one_sided": "greater alpha=.025 displays two-sided alpha=.05. Bernoulli coverage indicators use exact CP; no fractional observations are coerced into binomial counts.",
        },
        "known_production_issue": {
            "cases": "All 12 winsor-alt-* rows; acceptance NOT complete.",
            "locations": [
                "increment/estimation/engine.py:1903",
                "increment/estimation/engine.py:1416",
                "increment/estimation/inference.py:603",
            ],
            "defect": "Clipped per-arm variances combined as independent omit cutoff-indicator variance and cross covariance. Full/fixed ratios and implied limiting errors are recorded per row.",
        },
        "known_ate_issue": {
            "case": "ate-unequal-200-50; acceptance NOT complete.",
            "locations": [
                "increment/estimation/armstats.py:1657",
                "increment/estimation/inference.py:819",
            ],
            "defect": "Uncorrected score SE plus normal critical value undercovers this finite-N unequal-variance DGP; independent normal/chi-square integration yields .057319 noncoverage > .055.",
        },
        "smoke_seeds": {
            "winsor_ordering": 809,
            "normal": 11,
            "lift": 22,
            "ate": 33,
            "binomial": 66,
            "one_lift": 44,
            "one_ate": 55,
            "doubling": 606,
        },
        "cases": records,
    }


MANIFEST_PATH = Path(__file__).with_name("_i15_manifest.json")


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text())


if __name__ == "__main__":
    MANIFEST_PATH.write_text(json.dumps(build_manifest(), indent=2, allow_nan=False) + "\n")
