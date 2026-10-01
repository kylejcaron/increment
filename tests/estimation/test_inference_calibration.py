"""Bounded parity checks and helpers for the frozen inference calibration campaign.

The full 141-case design runs only through scripts.run_i15_campaign and retains
its original replication counts. Pytest checks representative parity and exact
arithmetic; it does not certify the full campaign. CLI evidence is stored in an
immutable output bundle, including failed gates and refusal counts.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, TypedDict

import numpy as np
import pytest
from scipy.stats import norm

from increment.estimation.armstats import IndependentMeanComponent, IndependentMeanReference
from increment.estimation.inference import infer_ate, infer_lift, normal_posterior
from tests._i15_design import ETA, exact_label_reference, load_manifest, sample
from tests.mc import binomial_error_upper_bound, coverage_lower_bound, scientific_delta

TRUE_LOG_RR = 0.10
TRUE_LIFT = math.exp(TRUE_LOG_RR) - 1.0
SE_T = 0.04
SE_C = 0.03
TRUE_ATE = 0.05
SIGMA_SCORE = 1.0
N_UNITS = 200
P_CONTROL = 0.15
TRUE_BINOMIAL_LIFT = 0.10
N_PER_ARM = 1000

MANIFEST = load_manifest()
CASES = {case["case_id"]: case for case in MANIFEST["cases"]}
WINSOR_CASES = [c for c in CASES.values() if c.get("kind") in ("aa", "alternative")]


class _Observation(TypedDict, total=False):
    interval: tuple[float | None, float | None]
    oracle: tuple[float, float | None]
    additive_interval: tuple[float | None, float | None]
    reported_variance: float
    reason: str | None
    method: str
    point: float | None
    additive_point: float | None
    cutoff: float
    quantile: float
    interval_status: tuple[str, str]
    additive_interval_status: tuple[str, str]
    reference_kind: str
    inference_status: str
    qualification: str
    counts: list[int]


def _bounds(estimate):
    if estimate.confidence_set is not None:
        region = estimate.confidence_set.relative
        return (
            region.lower.value if region.lower.value is not None else -math.inf,
            region.upper.value if region.upper.value is not None else math.inf,
        )
    lift = estimate.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return lift.lb, lift.ub


def _normal_posterior_interval(rng, alpha):
    sigma = 0.05
    mu_hat = rng.normal(TRUE_LOG_RR, sigma)
    posterior = normal_posterior(mu_hat, sigma)
    # Independent conjugate calculation, with the actual default N(0,1e6).
    variance = 1 / (1 / sigma**2 + 1 / 1e12)
    expected_mu = variance * mu_hat / sigma**2
    assert posterior.mu == pytest.approx(expected_mu, rel=1e-12, abs=1e-14)
    assert posterior.sigma == pytest.approx(math.sqrt(variance), rel=1e-12)
    # This near-flat prior has fixed-truth frequentist coverage ~.95 here.
    # No such assertion is made for arbitrary informative Bayes intervals.
    z = float(norm.isf(alpha / 2))
    return posterior.mu - z * posterior.sigma, posterior.mu + z * posterior.sigma


def _infer_lift_interval(rng, alpha, alternative="two-sided"):
    log_c = math.log(2.0)
    log_t = log_c + TRUE_LOG_RR
    return _bounds(
        infer_lift(
            metric="m",
            group_id="T",
            method="mean",
            method_role="decision",
            log_rr=rng.normal(log_t, SE_T) - rng.normal(log_c, SE_C),
            se_t=SE_T,
            se_c=SE_C,
            alpha=alpha,
            alternative=alternative,
        )
    )


def _infer_ate_interval(rng, alpha, alternative="two-sided"):
    x = rng.normal(TRUE_ATE, SIGMA_SCORE, size=N_UNITS)
    scores = IndependentMeanReference(
        components=(IndependentMeanComponent.from_values("T", x.tolist(), coefficient=1),)
    )
    return _bounds(
        infer_ate(
            metric="m",
            group_id="T",
            method="independent_mean",
            method_role="decision",
            point=scores.point,
            scores=scores,
            alpha=alpha,
            alternative=alternative,
            value_scale="absolute",
        )
    )


def _unequal_ate_interval(rng, case):
    d = case["dgp"]
    nc, nt = d["n_c"], d["n_t"]
    control = rng.normal(d["mu_c"], d["sigma_c"], nc)
    treatment = rng.normal(d["mu_t"], d["sigma_t"], nt)
    scores = IndependentMeanReference(
        components=(
            IndependentMeanComponent.from_values("C", control.tolist(), coefficient=-1),
            IndependentMeanComponent.from_values("T", treatment.tolist(), coefficient=1),
        )
    )
    return _bounds(
        infer_ate(
            metric="m",
            group_id="T",
            method="independent_mean",
            method_role="decision",
            point=scores.point,
            scores=scores,
            alpha=0.05,
            value_scale="absolute",
        )
    )


def _binomial_draw(rng, alpha):
    """Preserve seed order and the original ArmStats -> variance -> lift seam."""
    from increment.estimation.armstats import ArmStats
    from increment.estimation.variance import MeanVarianceModel

    model = MeanVarianceModel()
    counts, arms = [], []
    for group, probability in (("control", P_CONTROL), ("T", (1 + TRUE_BINOMIAL_LIFT) * P_CONTROL)):
        k = int(rng.binomial(N_PER_ARM, probability))
        counts.append(k)
        arms.append(
            model.log_mean_se(
                ArmStats.from_raw_sums(
                    study_id="e",
                    metric="conv",
                    group_id=group,
                    n=N_PER_ARM,
                    sum_y=float(k),
                    sum_y2=float(k),
                )
            )
        )
    (log_c, se_c), (log_t, se_t) = arms
    interval = _bounds(
        infer_lift(
            metric="conv",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=log_t - log_c,
            se_t=se_t,
            se_c=se_c,
            alpha=alpha,
        )
    )
    return interval, counts


def _binomial_lift_interval(rng, alpha):
    return _binomial_draw(rng, alpha)[0]


def _winsor_metric(quantile, *, stream=0):
    from increment.semantics.models import MeanMetric, Winsorization
    from increment.winsor import WinsorInferenceSpec, WinsorSupport

    return MeanMetric(
        name="revenue",
        entity="unit_id",
        fact="revenue",
        winsorization=Winsorization(
            upper_percentile=quantile,
            inference=WinsorInferenceSpec(stream=stream),
            support=WinsorSupport(
                lower=0, provenance="Frozen I15 DGPs have nonnegative support; no upper tail bound."
            ),
        ),
    )


def _winsor_production(con, control, treatment, quantile, *, stream=0):
    """Real pooled quantile -> real centered group summary -> public engine.

    Overwrite only this connection's in-memory input table each replicate.
    The production quantile is evaluated anew; numpy is only its independent
    linear-interpolation check, never a substitute for the actual transform.
    """
    import pyarrow as pa

    from increment.estimation.engine import estimate_lift
    from increment.query.builders import group_summary, winsorize_unit_totals
    from increment.winsor import RawArm, WinsorRawState

    values = np.concatenate((control, treatment))
    nc, nt = len(control), len(treatment)
    table = pa.table(
        {
            "unit_id": np.arange(nc + nt),
            "experiment_id": ["aa"] * (nc + nt),
            "metric": ["revenue"] * (nc + nt),
            "group_id": ["control"] * nc + ["treatment"] * nt,
            "y": values,
            "x": np.zeros(nc + nt),
            "y_den": np.zeros(nc + nt),
            "d": np.zeros(nc + nt),
        }
    )
    totals = con.create_table("i15_units", obj=table, overwrite=True)
    metric = _winsor_metric(quantile, stream=stream)
    transformed = winsorize_unit_totals(totals, metric)
    rows = con.to_pyarrow(group_summary(transformed)).to_pylist()
    raw_rows = con.to_pyarrow(transformed.select("group_id", "y_raw")).to_pylist()
    raw = WinsorRawState(
        metric="revenue",
        study_id="aa",
        missingness="error",
        quantile=quantile,
        support=metric.winsorization.support,
        inference=metric.winsorization.inference,
        arms=tuple(
            RawArm(group_id=g, values=tuple(r["y_raw"] for r in raw_rows if r["group_id"] == g))
            for g in ("control", "treatment")
        ),
    )
    cutoff = float(np.quantile(values, quantile, method="linear"))
    for row in rows:
        assert row["winsor_upper_bound"] == pytest.approx(cutoff, rel=1e-12, abs=1e-12)
    computation = estimate_lift(
        metrics=[metric],
        summary=rows,
        control_group="control",
        alpha=0.05,
        raw_outcomes={"revenue": raw},
    )
    reason = ";".join(f.code for f in computation.failures.values()) or None
    if not computation.results:
        return None, reason
    (result,) = computation.results
    return result, reason


def _winsor_raw_state(control, treatment, quantile, *, stream=0):
    """Captured-array raw state for the offline campaign; no SQL or engine calls."""
    from increment.winsor import RawArm, WinsorInferenceSpec, WinsorRawState, WinsorSupport

    return WinsorRawState(
        metric="revenue",
        study_id="aa",
        missingness="error",
        quantile=quantile,
        support=WinsorSupport(
            lower=0, provenance="Frozen I15 DGPs have nonnegative support; no upper tail bound."
        ),
        inference=WinsorInferenceSpec(stream=stream),
        arms=(
            RawArm(group_id="control", values=tuple(np.asarray(control).tolist())),
            RawArm(group_id="treatment", values=tuple(np.asarray(treatment).tolist())),
        ),
    )


@dataclass(slots=True)
class _SharedQuantileEntry:
    """One quantile's accumulating roots while its twin shares the same draw."""

    position: int
    raw: Any
    cutoff: float
    observed: Any
    point: float
    delta: float
    pilot_cutoff: float
    center_log: float
    center_delta: float
    log_roots: list[float | None] = field(default_factory=list)
    additive_roots: list[float | None] = field(default_factory=list)
    failures: list[int] = field(default_factory=list)


def _winsor_shared_quantile_regions(control, treatment, quantiles, *, stream=0):
    """Evaluate one captured sample at several clipping quantiles from one draw.

    The frozen v1 replicate draw is a function of the per-arm log centers, the
    bandwidth and the two seeded streams only; ``raw.quantile`` first enters
    when a replicate is reduced to statistics. Sibling cells that differ only
    in the clipping quantile therefore share both the replicate draw and its
    pooled order statistic -- ``np.quantile`` resolves every requested quantile
    from one partition -- while everything downstream stays per-quantile.

    This mirrors ``full_procedure_bootstrap_reference`` replicate for replicate.
    ``test_shared_quantile_draw_matches_the_production_kernel`` and the public
    engine parity check are what hold the two together.

    Returns one entry per requested quantile, in order: its confidence set, or
    the coded refusal that quantile raised. A refusal from the shared prologue
    belongs to every quantile and propagates instead of being returned.
    """
    from increment.errors import CodedError
    from increment.estimation._winsor_bootstrap import (
        _CHUNK,
        bootstrap_confidence_set,
        fit_positive_log_pilot,
        full_procedure_statistics,
        pilot_population_target,
    )
    from increment.estimation.winsor import _linear_cutoff, _mean_fraction
    from increment.winsor import BootstrapReference, BootstrapRoots, winsor_refuse

    states = tuple(_winsor_raw_state(control, treatment, q, stream=stream) for q in quantiles)
    base = states[0]
    pilots = fit_positive_log_pilot(base)
    groups = [p.group_id for p in pilots]
    ci, ti = groups.index("control"), groups.index("treatment")
    samples = tuple(np.asarray(arm.values)[None, :] for arm in base.arms)
    spec = base.inference
    results = [None] * len(states)
    ready = []
    for position, raw in enumerate(states):
        try:
            cutoff = _linear_cutoff(raw)
            observed = full_procedure_statistics(
                samples, raw.quantile, ci, ti, cutoff=np.asarray([cutoff])
            )
            if not observed.valid()[0]:
                reason = (
                    "density_unresolved"
                    if not math.isfinite(float(observed.density_scaled[0]))
                    or observed.density_scaled[0] <= 0
                    else "studentization_degenerate"
                )
                winsor_refuse(reason, "Observed full-procedure studentization is unavailable.")
            # Keep the exact type-7 clipped-arm statistic, independent of score scaling.
            mc = _mean_fraction(np.minimum(raw.arm("control").values, cutoff))
            mt = _mean_fraction(np.minimum(raw.arm("treatment").values, cutoff))
            try:
                relative_point = float((mt - mc) / mc)
            except OverflowError:
                relative_point = math.inf
            ell = (
                math.log1p(relative_point)
                if math.isfinite(relative_point) and relative_point > -1
                else math.log(float(mt)) - math.log(float(mc))
            )
            pilot_cutoff, center_log, center_delta = pilot_population_target(
                pilots, raw.quantile, "control", "treatment"
            )
        except CodedError as refusal:
            results[position] = refusal
            continue
        ready.append(
            _SharedQuantileEntry(
                position=position,
                raw=raw,
                cutoff=cutoff,
                observed=observed,
                point=ell,
                delta=float(mt - mc),
                pilot_cutoff=pilot_cutoff,
                center_log=center_log,
                center_delta=center_delta,
            )
        )
    if not ready:
        return tuple(results)
    # Separate center/noise streams per canonical arm. Chunk size is part of v1.
    generators = tuple(
        (
            np.random.Generator(
                np.random.PCG64DXSM(
                    np.random.SeedSequence(spec.seed, spawn_key=(spec.stream, g, 0))
                )
            ),
            np.random.Generator(
                np.random.PCG64DXSM(
                    np.random.SeedSequence(spec.seed, spawn_key=(spec.stream, g, 1))
                )
            ),
        )
        for g in range(len(pilots))
    )
    centers = tuple(np.asarray(pilot.log_centers) for pilot in pilots)
    wanted = [entry.raw.quantile for entry in ready]
    for start in range(0, spec.replicates, _CHUNK):
        size = min(_CHUNK, spec.replicates - start)
        draws = []
        with np.errstate(all="ignore"):
            for z, pilot, (index_stream, noise) in zip(centers, pilots, generators, strict=True):
                indices = index_stream.integers(len(z), size=(size, len(z)))
                draws.append(
                    np.exp(z[indices] + pilot.bandwidth * noise.standard_normal((size, len(z))))
                )
            draws = tuple(draws)
            pooled = np.quantile(np.concatenate(draws, axis=1), wanted, axis=1, method="linear")
            for order, entry in enumerate(ready):
                statistics = full_procedure_statistics(
                    draws, wanted[order], ci, ti, cutoff=pooled[order]
                )
                lr = (statistics.log_relative - entry.center_log) / statistics.log_se
                ar = (statistics.additive - entry.center_delta) / statistics.additive_se
                valid = statistics.valid() & np.isfinite(lr) & np.isfinite(ar)
                entry.log_roots.extend(
                    x if ok else None for x, ok in zip(lr.tolist(), valid.tolist(), strict=True)
                )
                entry.additive_roots.extend(
                    x if ok else None for x, ok in zip(ar.tolist(), valid.tolist(), strict=True)
                )
                entry.failures.extend(start + i for i, ok in enumerate(valid.tolist()) if not ok)
    for entry in ready:
        try:
            reference = BootstrapReference(
                raw=entry.raw,
                spec=spec,
                control="control",
                treatment="treatment",
                pilots=pilots,
                pilot_cutoff=entry.pilot_cutoff,
                observed_cutoff=entry.cutoff,
                log_relative=BootstrapRoots(
                    point=entry.point,
                    pilot_target=entry.center_log,
                    se=float(entry.observed.log_se[0]),
                    roots=tuple(entry.log_roots),
                ),
                additive=BootstrapRoots(
                    point=entry.delta,
                    pilot_target=entry.center_delta,
                    se=float(entry.observed.additive_se[0]),
                    roots=tuple(entry.additive_roots),
                ),
                failure_indices=tuple(entry.failures),
            )
            results[entry.position] = bootstrap_confidence_set(reference)
        except CodedError as refusal:
            results[entry.position] = refusal
    return tuple(results)


def _winsor_array_region(control, treatment, quantile, *, stream=0):
    """Single-quantile captured-array region; refusals propagate to the caller."""
    from increment.errors import CodedError

    (region,) = _winsor_shared_quantile_regions(control, treatment, (quantile,), stream=stream)
    if isinstance(region, CodedError):
        raise region
    return region


def _winsor_region_observation(region) -> _Observation:
    """Retain undefined endpoints separately from valid unbounded endpoints."""

    def endpoints(interval):
        return tuple(
            endpoint.value
            if endpoint.status == "finite"
            else direction * math.inf
            if endpoint.status == "unbounded"
            else None
            for endpoint, direction in ((interval.lower, -1), (interval.upper, 1))
        )

    reasons = tuple(
        endpoint.reason
        for endpoint in (region.relative.lower, region.relative.upper)
        if endpoint.reason is not None
    )
    return {
        "quantile": region.reference.raw.quantile,
        "cutoff": region.reference.observed_cutoff,
        "point": region.point,
        "additive_point": region.additive_point,
        "interval": endpoints(region.relative),
        "interval_status": (region.relative.lower.status, region.relative.upper.status),
        "reason": ";".join(reasons) if reasons else None,
        "additive_interval": endpoints(region.additive),
        "additive_interval_status": (region.additive.lower.status, region.additive.upper.status),
        "reported_variance": region.reference.log_relative.se**2,
        "method": region.method,
        "reference_kind": type(region.reference).__name__,
        "inference_status": region.reference.raw.inference.status,
        "qualification": region.qualification,
    }


def _winsor_unit_observations(cases, rng):
    """One outer sample per draw feeds every clipping quantile in the unit.

    Cells in a unit differ only in their clipping quantile, so they share the
    sample, the pilot fit and the replicate draw. Each cell still receives its
    own observation, journal and gates. Each sample is generated only after its
    attempt has been checkpointed.
    """
    from increment.errors import CodedError

    dgp = cases[0]["dgp"]
    identifiers = tuple(case["case_id"] for case in cases)
    quantiles = tuple(case["dgp"]["quantile"] for case in cases)
    for stream in range(cases[0]["design"]["repetitions"]):
        control = sample(rng, dgp["control"], dgp["n_c"])
        treatment = sample(rng, dgp["treatment"], dgp["n_t"])
        try:
            regions = _winsor_shared_quantile_regions(control, treatment, quantiles, stream=stream)
        except CodedError as exc:
            yield {case_id: {"reason": exc.code} for case_id in identifiers}
            continue
        yield {
            case_id: {"reason": region.code}
            if isinstance(region, CodedError)
            else _winsor_region_observation(region)
            for case_id, region in zip(identifiers, regions, strict=True)
        }


def _campaign_observations(case, rng):
    """Yield the original non-winsor calibration helpers without test gates."""
    family = case["family"]
    repetitions = case["design"]["repetitions"]
    for _ in range(repetitions):
        if family == "normal_posterior":
            interval = _normal_posterior_interval(rng, 0.05)
            yield {"interval": interval}
        elif family == "lift_interval":
            interval = _infer_lift_interval(rng, 0.05)
            yield {"interval": interval}
        elif family == "ate_interval":
            if case["case_id"] == "ate-seed303":
                interval = _infer_ate_interval(rng, 0.05)
            else:
                interval = _unequal_ate_interval(rng, case)
            yield {"interval": interval}
        elif family == "binomial_end_to_end":
            from tests.estimation.test_rare_event_calibration import bonferroni_lift_interval

            interval, counts = _binomial_draw(rng, 0.05)
            oracle = bonferroni_lift_interval(counts[0], N_PER_ARM, counts[1], N_PER_ARM, 0.05)
            yield {"interval": interval, "counts": counts, "oracle": oracle}
        elif family == "one_sided_lift":
            interval = _infer_lift_interval(rng, 0.025, "greater")
            yield {"interval": interval}
        elif family == "one_sided_ate":
            interval = _infer_ate_interval(rng, 0.025, "greater")
            yield {"interval": interval}
        else:
            raise ValueError(f"unsupported campaign family: {family}")


def _reference_width(case):
    """Independent nominal full width, on log scale for relative intervals."""
    family, d = case["family"], case["dgp"]
    if family == "pooled_winsor":
        se = math.sqrt(case["reference"]["log_ratio_variance"]["full"])
    elif family == "binomial_end_to_end":
        se = math.sqrt(
            (1 - d["p_c"]) / (d["n_c"] * d["p_c"]) + (1 - d["p_t"]) / (d["n_t"] * d["p_t"])
        )
    elif family in ("lift_interval", "one_sided_lift"):
        se = math.hypot(SE_T, SE_C)
    elif d["distribution"] == "two_normal_arms":
        se = math.sqrt(d["sigma_c"] ** 2 / d["n_c"] + d["sigma_t"] ** 2 / d["n_t"])
    elif family == "normal_posterior":
        se = 0.05
    else:
        se = SIGMA_SCORE / math.sqrt(N_UNITS)
    return 2 * float(norm.isf(0.025)) * se


def _error_evidence(errors, reps, q):
    if reps == 0:
        return {
            "errors": errors,
            "reps": reps,
            "rate": None,
            "upper": 1.0,
            "coverage_lower": 0.0,
            "margin": None,
            "passed": False,
        }
    upper = binomial_error_upper_bound(errors, reps, ETA)
    delta = scientific_delta(q)
    margin_limit = min(delta / 2, 0.00125) if q == 0.025 else delta / 2
    margin = upper - errors / reps
    lower = coverage_lower_bound(reps - errors, reps, ETA)
    return {
        "errors": errors,
        "reps": reps,
        "rate": errors / reps,
        "upper": upper,
        "coverage_lower": lower,
        "margin": margin,
        "passed": upper <= q + delta and lower >= 1 - q - delta and margin <= margin_limit,
    }


class _CaseAccumulator:
    """Every assigned replication stays in the unconditional denominator.

    Hits/misses are Bernoulli, including for binary-outcome DGPs. No fractional
    FCP/FDP is cast to a binomial count; such future statistics need the exact-KL
    bounded-variable helper ``tests.mc.kl_chernoff_upper_bound``. One contrast
    retains its alpha-doubling/FCR level.
    """

    def __init__(self, case, repetitions=None):
        self.case = case
        self.repetitions = case["design"]["repetitions"] if repetitions is None else repetitions
        self.q = case["design"]["nominal_error"]
        self.truth = case["truth"]
        self.one_sided = case["family"].startswith("one_sided")
        self.log_scale = case["family"] in (
            "pooled_winsor",
            "binomial_end_to_end",
            "lift_interval",
            "one_sided_lift",
        )
        self.available = self.hits = self.rejected = self.useful = self.finite = 0
        self.oracle_hits = self.oracle_count = 0
        self.widths, self.log_widths, self.reported_variances = [], [], []
        self.reasons = Counter()
        self.inference_methods = Counter()
        self.oracle_widths = []
        self.cap = 4 * _reference_width(case)
        self.additive_hits = self.additive_available = self.additive_useful = 0
        self.additive_widths = []
        self.additive_cap = None
        if case["family"] == "pooled_winsor":
            self.additive_cap = (
                8
                * float(norm.isf(0.025))
                * math.sqrt(case["reference"]["difference_variance"]["full"])
            )

    def add(self, observation: _Observation) -> None:
        case = self.case
        interval = observation.get("interval")
        if "method" in observation:
            self.inference_methods[observation["method"]] += 1
        # Additive availability is independent of relative endpoint availability.
        if self.additive_cap is not None:
            additive = observation.get("additive_interval")
            if additive is not None:
                alo, ahi = additive
                if (
                    alo is not None
                    and ahi is not None
                    and math.isfinite(alo)
                    and math.isfinite(ahi)
                    and alo < ahi
                ):
                    self.additive_available += 1
                    self.additive_hits += alo <= case["reference"]["difference"] <= ahi
                    self.additive_widths.append(ahi - alo)
                    self.additive_useful += ahi - alo <= self.additive_cap
        oracle = observation.get("oracle")
        if oracle is not None:
            lo, hi = oracle
            self.oracle_hits += lo <= self.truth and (hi is None or self.truth <= hi)
            self.oracle_count += 1
            if hi is not None:
                self.oracle_widths.append(hi - lo)
        if interval is None:
            self.reasons[observation.get("reason", "missing interval")] += 1
            return
        lb, ub = interval
        if lb is None or ub is None or not (math.isfinite(lb) and math.isfinite(ub) and lb < ub):
            self.reasons[observation.get("reason") or "nonfinite_or_collapsed_interval"] += 1
            return
        self.available += 1
        self.hits += bool(lb <= self.truth and (self.one_sided or self.truth <= ub))
        self.rejected += bool(lb > 0 or (not self.one_sided and ub < 0))
        width = ub - lb
        assessment_width = (
            math.log1p(ub) - math.log1p(lb)
            if self.log_scale and lb > -1
            else (math.inf if self.log_scale else width)
        )
        self.widths.append(width)
        if math.isfinite(assessment_width):
            self.finite += 1
            self.log_widths.append(assessment_width)
            self.useful += assessment_width <= self.cap
        if "reported_variance" in observation:
            self.reported_variances.append(observation["reported_variance"])

    def report(self) -> dict:
        case, reps, q = self.case, self.repetitions, self.q
        hits, available = self.hits, self.available
        # A refusal is an unconditional miss, never a covered infinite interval.
        unconditional = _error_evidence(reps - hits, reps, q)
        conditional = _error_evidence(available - hits, available, q)
        unavailable_upper = binomial_error_upper_bound(reps - available, reps, ETA)
        finite_lower = coverage_lower_bound(self.finite, reps, ETA)
        useful_lower = coverage_lower_bound(self.useful, reps, ETA)
        power_lower = coverage_lower_bound(self.rejected, reps, ETA)
        gates = {
            "unconditional_coverage": unconditional["passed"],
            "conditional_coverage": conditional["passed"],
            "conditional_precision_design": available
            >= case["design"]["conditional_required_repetitions"],
            "availability": unavailable_upper <= 0.01,
            "finite_width": finite_lower >= 0.99,
            "useful_width": useful_lower >= 0.95,
        }
        power_min = case["nonvacuity"]["power_min"]
        if power_min is not None:
            gates["power"] = power_lower >= power_min
        oracle_evidence = None
        if case["family"] == "binomial_end_to_end":
            oracle_evidence = _error_evidence(
                self.oracle_count - self.oracle_hits, self.oracle_count, q
            )
            gates["independent_binary_reference"] = (
                self.oracle_count == reps and oracle_evidence["passed"]
            )
        additive_evidence = None
        if self.additive_cap is not None:
            absolute_unconditional = _error_evidence(reps - self.additive_hits, reps, q)
            absolute_conditional = _error_evidence(
                self.additive_available - self.additive_hits,
                self.additive_available,
                q,
            )
            additive_evidence = {
                "unconditional": absolute_unconditional,
                "conditional": absolute_conditional,
                "available": self.additive_available,
                "width_cap": self.additive_cap,
                "mean_width": float(np.mean(self.additive_widths))
                if self.additive_widths
                else None,
            }
            gates.update(
                {
                    "additive_unconditional_coverage": absolute_unconditional["passed"],
                    "additive_conditional_coverage": absolute_conditional["passed"],
                    "additive_precision_design": self.additive_available
                    >= case["design"]["conditional_required_repetitions"],
                    "additive_availability": binomial_error_upper_bound(
                        reps - self.additive_available, reps, ETA
                    )
                    <= 0.01,
                    "additive_useful_width": coverage_lower_bound(self.additive_useful, reps, ETA)
                    >= 0.95,
                }
            )
        return {
            **case,
            "executed": True,
            "executed_repetitions": reps,
            "historical_repetitions_completed": reps == case["design"]["repetitions"],
            "production_inference": MANIFEST["production_inference_candidate"],
            "observed_inference_methods": dict(self.inference_methods),
            "availability": {
                **case["availability"],
                "observed": available / reps,
                "refusal_reasons": dict(self.reasons),
                "unavailable_upper": unavailable_upper,
            },
            "results": {
                "coverage": hits / reps,
                "conditional_coverage": hits / available if available else None,
                "error_rate": (reps - hits) / reps,
                "rejection_rate": self.rejected / reps,
                "width": float(np.mean(self.widths)) if self.widths else None,
                "assessment_width": float(np.mean(self.log_widths)) if self.log_widths else None,
                "width_cap": self.cap,
                "finite_width_lower": finite_lower,
                "useful_width_lower": useful_lower,
                "power": self.rejected / reps if self.truth != 0 else None,
                "power_lower": power_lower,
                "mc_margin": unconditional["margin"],
                "unconditional": unconditional,
                "conditional": conditional,
                "oracle": oracle_evidence,
                "additive": additive_evidence,
                "oracle_width": float(np.mean(self.oracle_widths)) if self.oracle_widths else None,
                "mean_reported_log_variance": float(np.mean(self.reported_variances))
                if self.reported_variances
                else None,
                "gates": gates,
            },
        }


def _run_shared_cases(cases, build, *, seed, repetitions=None, observers=None):
    """Drive one outer draw stream across every case that shares it.

    ``build`` returns one observation per case id per draw. Cases sharing a
    stream share the sample and the replicate draw, so their per-draw outcomes
    are dependent; each case keeps its own exact one-sided binomial bound, and
    the family-wise union bound is insensitive to dependence across cases.
    """
    from increment.errors import CodedError
    from increment.estimation.inference import LiftGuardError

    accumulators = {case["case_id"]: _CaseAccumulator(case, repetitions) for case in cases}
    counts = {accumulator.repetitions for accumulator in accumulators.values()}
    if len(counts) != 1:
        raise ValueError("cases sharing one draw stream must share their repetition count")
    (reps,) = counts
    observers = {} if observers is None else observers
    rng = np.random.default_rng(seed)
    for index in range(reps):
        for observer in observers.values():
            observer("started", index, None)
        try:
            observations = build(rng)
        except LiftGuardError as exc:
            observations = {case_id: _Observation(reason=exc.reason) for case_id in accumulators}
        except CodedError as exc:
            observations = {case_id: _Observation(reason=exc.code) for case_id in accumulators}
        except Exception as exc:
            detail = {"error_type": type(exc).__name__, "error": str(exc)}
            for observer in observers.values():
                observer("failed", index, detail)
            raise
        for case_id, accumulator in accumulators.items():
            observation = observations[case_id]
            observer = observers.get(case_id)
            if observer is not None:
                observer(
                    "refused" if set(observation) == {"reason"} else "completed", index, observation
                )
            accumulator.add(observation)
    return {case_id: accumulator.report() for case_id, accumulator in accumulators.items()}


@pytest.fixture
def winsor_connection():
    import ibis

    con = ibis.duckdb.connect()
    try:
        yield con
    finally:
        con.disconnect()


@pytest.mark.slow
class TestIntervalCalibration:
    """Representative parity checks; full historical R runs use the CLI."""

    @pytest.mark.parametrize(
        "case",
        [
            c
            for c in WINSOR_CASES
            if c["case_id"]
            in {
                "winsor-original-seed808",
                "winsor-aa-contamination-n50-1x1-p0.99",
                "winsor-aa-ln-s2.0-n50-1x4-p0.99",
                "winsor-alt-n2000-4x1-p0.99",
            }
        ],
        ids=lambda c: c["case_id"],
    )
    def test_pooled_winsor_representative(self, case, winsor_connection):
        stream = _winsor_unit_observations([case], np.random.default_rng(case["seed"]))
        observed = next(stream)[case["case_id"]]
        rng = np.random.default_rng(case["seed"])
        dgp = case["dgp"]
        control = sample(rng, dgp["control"], dgp["n_c"])
        treatment = sample(rng, dgp["treatment"], dgp["n_t"])
        public, reason = _winsor_production(
            winsor_connection, control, treatment, dgp["quantile"], stream=0
        )
        if public is None:
            assert observed["reason"] == reason
        else:
            assert reason == "evidence.experimental_reference"
            assert public.confidence_set is not None
            assert observed == _winsor_region_observation(public.confidence_set)

    def test_pooled_winsor_exact_label_reference_representative(self):
        case = CASES["winsor-exact-labels"]
        raw = np.asarray(case["dgp"]["raw_pooled_values"], dtype=float)
        assignments, pvalues, cutoff = exact_label_reference(raw, 4, 0.99)
        assert len(assignments) == 70
        assert sum(p <= 0.05 for p in pvalues) == 2
        assert cutoff == pytest.approx(np.quantile(raw, 0.99))


@pytest.mark.slow
class TestOneSidedCalibration:
    def test_one_sided_interval_matches_doubled_alpha_two_sided(self):
        _assert_doubled_alpha()


def _assert_doubled_alpha():
    for build in (_infer_lift_interval, _infer_ate_interval):
        one = build(np.random.default_rng(606), 0.025, "greater")
        two = build(np.random.default_rng(606), 0.05, "two-sided")
        assert one == pytest.approx(two)


class TestCalibrationSmoke:
    """Fixed arithmetic checks; no small-repetition statistical pass mode."""

    def test_manifest_precision_and_original_cases(self):
        assert len(CASES) == len(MANIFEST["cases"]) == 141
        assert dict(Counter(c["family"] for c in CASES.values())) == MANIFEST["counts"]
        assert MANIFEST["eta"] == ETA
        original = CASES["winsor-original-seed808"]
        assert original["seed"] == 808
        assert original["dgp"]["n_c"] == original["dgp"]["n_t"] == 500
        assert original["dgp"]["control"]["mu"] == 1.0
        assert original["dgp"]["control"]["sigma"] == 1.6
        assert original["dgp"]["quantile"] == 0.99
        assert len([c for c in WINSOR_CASES if c["kind"] == "aa"]) == 120
        assert any(
            c["dgp"]["baseline_n"] == 50
            and c["dgp"]["allocation"] == [1, 4]
            and c["dgp"]["control"].get("sigma") == 2.0
            for c in WINSOR_CASES
        )
        for identifier, seed in (("one-lift-seed404", 404), ("one-ate-seed505", 505)):
            case = CASES[identifier]
            design = case["design"]
            assert case["seed"] == seed
            assert design["nominal_error"] == 0.025
            assert design["scientific_delta"] == pytest.approx(0.0025)
            assert design["coverage_lower_min"] >= 0.9725
            for reps in (design["repetitions"], design["conditional_required_repetitions"]):
                errors = math.ceil(reps * 0.0275)
                margin = binomial_error_upper_bound(errors, reps, ETA) - errors / reps
                assert margin <= 0.00125

    def test_winsor_qualification_rejects_contradictory_reference(self):
        from increment.errors import InvalidRequestError
        from increment.winsor import WinsorInferenceSpec

        with pytest.raises(InvalidRequestError) as caught:
            WinsorInferenceSpec(
                method="joint-rank-projection-v1",
                qualification="pointwise_asymptotic_model_conditioned_v1",
            )
        assert caught.value.code == "estimation.winsor.invalid_state"

    def test_pooled_winsor_smoke(self):
        import pyarrow as pa

        from increment.estimation.engine import estimate_lift
        from increment.estimation.winsor import raw_state_from_source
        from increment.frame import from_unit_summary

        # Pooled [1,1,2,2,3,3,5,9], p=.75 -> c=3.5. Both arm means
        # become (1+2+3+3.5)/4=2.375, sample variance=59/48.
        groups = ["control"] * 4 + ["treatment"] * 4
        raw = [1.0, 2.0, 3.0, 9.0, 1.0, 2.0, 3.0, 5.0]
        order = np.random.default_rng(809).permutation(8)
        table = pa.table(
            {"unit": order, "group": [groups[i] for i in order], "revenue": [raw[i] for i in order]}
        )
        source = from_unit_summary(
            table,
            unit="unit",
            group="group",
            control="control",
            metrics=[
                {
                    "name": "revenue",
                    "winsorization": {
                        "upper_percentile": 0.75,
                        "support": {"lower": 0, "provenance": "Fixed arithmetic fixture"},
                    },
                }
            ],
        )
        for row in source.raw_moments:
            assert row["winsor_upper_bound"] == 3.5
            assert row["ref_y"] == 2.375
            assert row["cy1"] == 0
            assert row["cy2"] / (row["n"] - 1) == pytest.approx(59 / 48)
        [estimate] = estimate_lift(
            metrics=source.context.metrics,
            summary=source.raw_moments,
            control_group="control",
            raw_outcomes={"revenue": raw_state_from_source(source, source.context.metrics[0])},
        ).results
        assert estimate.require_lift().value == pytest.approx(0, abs=1e-14)
        assert estimate.reference_kind == "confidence_set"
        assert estimate.confidence_set is not None
        assert estimate.confidence_set.raw.arm("control").values == (1, 2, 3, 9)
        assert estimate.require_lift().log_se is None

    def test_two_sided_coverage_smoke(self):
        # Preserve the former smoke seeds, now checking identities on one draw.
        for build, seed in (
            (_normal_posterior_interval, 11),
            (_infer_lift_interval, 22),
            (_infer_ate_interval, 33),
            (_binomial_lift_interval, 66),
        ):
            lb, ub = build(np.random.default_rng(seed), 0.05)
            assert math.isfinite(lb) and math.isfinite(ub) and lb < ub
            narrow = build(np.random.default_rng(seed), 0.10)
            assert lb < narrow[0] < narrow[1] < ub

    def test_one_sided_coverage_smoke(self):
        for build, seed in ((_infer_lift_interval, 44), (_infer_ate_interval, 55)):
            one = build(np.random.default_rng(seed), 0.025, "greater")
            two = build(np.random.default_rng(seed), 0.05)
            assert one == pytest.approx(two)
        _assert_doubled_alpha()


#: Design fields read straight from library beta/binomial tails. Their last
#: digits are not reproducible across the supported SciPy releases (measured
#: 1e-12 relative), so regeneration compares them within a numerical
#: reproducibility budget; every other field must regenerate exactly.
_SPECIAL_FUNCTION_DESIGN_FIELDS = frozenset(
    {"prospective_mc_margin", "margin_at_nominal", "nominal_false_failure_bound"}
)
_SPECIAL_FUNCTION_REPRODUCIBILITY = 1e-9


def _assert_regenerates(regenerated: Any, committed: Any, path: str) -> None:
    if isinstance(committed, dict):
        assert isinstance(regenerated, dict), path
        assert list(regenerated) == list(committed), path
        for key, value in committed.items():
            child = f"{path}.{key}"
            if key in _SPECIAL_FUNCTION_DESIGN_FIELDS and path.endswith(".design"):
                assert regenerated[key] == pytest.approx(
                    value, rel=_SPECIAL_FUNCTION_REPRODUCIBILITY, abs=0
                ), child
            else:
                _assert_regenerates(regenerated[key], value, child)
    elif isinstance(committed, list):
        assert isinstance(regenerated, list) and len(regenerated) == len(committed), path
        for index, (left, right) in enumerate(zip(regenerated, committed, strict=True)):
            _assert_regenerates(left, right, f"{path}[{index}]")
    else:
        assert regenerated == committed, path


@pytest.mark.slow
def test_manifest_regeneration_is_complete_and_deterministic():
    """Regenerate decisions exactly and special-function readings within their
    numerical reproducibility budget, retaining every design condition."""
    from increment.winsor import WinsorInferenceSpec
    from tests._i15_design import MANIFEST_PATH, build_manifest

    committed_bytes = MANIFEST_PATH.read_bytes()
    committed = json.loads(committed_bytes)
    regenerated = json.loads(json.dumps(build_manifest(), allow_nan=False))
    _assert_regenerates(regenerated, committed, "manifest")
    for case in regenerated["cases"]:
        design = case["design"]
        assert 0.0 <= design["prospective_mc_margin"] <= design["mc_margin_max"]
        if case.get("kind") != "exact":
            assert 0.0 <= design["margin_at_nominal"] <= design["prospective_mc_margin"]
            assert 0.0 < design["nominal_false_failure_bound"] <= design["eta_per_direction"]
    candidate = regenerated["production_inference_candidate"]
    spec = WinsorInferenceSpec(method=candidate["winsor_method"], replicates=candidate["budget"])
    assert spec.method == "positive-log-kernel-bootstrap-t-v1"
    assert spec.qualification == "pointwise_asymptotic_model_conditioned_v1"
    assert regenerated["execution_contract"]["random_outer_repetitions"] == 24_536_444
    assert (
        sum(c["design"]["repetitions"] for c in regenerated["cases"] if c.get("kind") == "exact")
        == 70
    )
    assert regenerated["gated_bounds"] == 1241
    assert regenerated["family_alpha"] == 0.01
    assert all(c["results"]["coverage"] is None for c in regenerated["cases"])


@pytest.mark.slow
@pytest.mark.parametrize(
    "control,treatment,q,stream",
    [
        ((1.0, 2.0, 3.0, 9.0), (1.0, 2.0, 3.0, 5.0), 0.75, 0),
        ((1.0, 2.0, 3.0, 4.0), (2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0), 0.99, 17),
        ((2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0), (1.0, 2.0, 3.0, 4.0), 0.95, 23),
        ((1.0, 1.0, 2.0, 2.0), (1.0, 2.0, 2.0, 3.0), 0.99, 1),
    ],
)
def test_captured_array_kernel_matches_complete_public_output(
    winsor_connection, control, treatment, q, stream
):
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.results import LiftEstimate

    region = _winsor_array_region(control, treatment, q, stream=stream)
    public, reason = _winsor_production(winsor_connection, control, treatment, q, stream=stream)
    assert reason == "evidence.experimental_reference"
    assert public.confidence_set == region  # includes every root, pilot, point, status and reason
    expected = estimate_winsor_lift(region.raw, "control", "treatment", reference=region.reference)
    assert public == expected
    restored = LiftEstimate.model_validate_json(public.model_dump_json())
    assert region.qualification == "pointwise_asymptotic_model_conditioned_v1"
    assert restored == public
    assert restored.reintervalize(0.1) == expected.reintervalize(0.1)


def _campaign_sample(case, *, draw=0):
    """One outer draw of a frozen winsor cell, control arm first."""
    dgp = case["dgp"]
    rng = np.random.default_rng(case["seed"] + draw)
    return (
        sample(rng, dgp["control"], dgp["n_c"]),
        sample(rng, dgp["treatment"], dgp["n_t"]),
    )


def _negation_ulps(left, right):
    """How far ``right`` is from ``-left``, in ulps of the larger magnitude."""
    scale = max(abs(left), abs(right))
    return abs(left + right) / (math.ulp(scale) if scale else math.ulp(1.0))


def _endpoint(bound) -> float:
    """An A/A endpoint must be present; absence breaks the collapse premise."""
    assert bound.value is not None
    return float(bound.value)


@pytest.mark.parametrize(
    "case_id",
    (
        "winsor-aa-ln-s2.0-n50-1x4-p0.99",
        "winsor-aa-contamination-n50-1x4-p0.95",
        "winsor-aa-gamma-k2-n50-1x4-p0.95",
    ),
)
def test_aa_allocation_swap_is_pathwise_equivariant(case_id):
    """One fixed A/A sample, arm roles swapped, replicate stream held fixed.

    This is the identity the campaign's allocation-swap collapse stands on: an
    A/A cell and its arm-size-swapped twin are one simulation read two ways.
    Everything a gate consumes is either bitwise identical -- studentization,
    cutoffs, pilots, failure set -- or an exact negation: both centering
    targets, both root series, the additive point and the additive endpoints.
    Only the lift-scale relative endpoints carry rounding, and only from the
    expm1/log1p round trip; a broken identity sits ~1e13 ulps away, not eight.
    """
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import BootstrapReference

    case = CASES[case_id]
    control, treatment = _campaign_sample(case)
    raw = _winsor_raw_state(control, treatment, case["dgp"]["quantile"], stream=7)
    region = estimate_winsor_lift(raw, "control", "treatment").confidence_set
    mirrored = estimate_winsor_lift(raw, "treatment", "control").confidence_set
    assert region is not None and mirrored is not None
    direct, swapped = region.reference, mirrored.reference
    assert isinstance(direct, BootstrapReference)
    assert isinstance(swapped, BootstrapReference)

    assert swapped.pilots == direct.pilots
    assert swapped.observed_cutoff == direct.observed_cutoff
    assert swapped.pilot_cutoff == direct.pilot_cutoff
    assert swapped.failure_indices == direct.failure_indices
    assert swapped.log_relative.se == direct.log_relative.se
    assert swapped.additive.se == direct.additive.se
    assert swapped.log_relative.pilot_target == -direct.log_relative.pilot_target
    assert swapped.additive.pilot_target == -direct.additive.pilot_target
    assert swapped.additive.point == -direct.additive.point
    assert _negation_ulps(direct.log_relative.point, swapped.log_relative.point) <= 1
    assert len(direct.log_relative.roots) == 1999
    for series in ("log_relative", "additive"):
        mine = getattr(direct, series).roots
        mirror = getattr(swapped, series).roots
        assert all(a + b == 0.0 for a, b in zip(mine, mirror, strict=True))

    add_lo, add_hi = _endpoint(region.additive.lower), _endpoint(region.additive.upper)
    m_add_lo, m_add_hi = _endpoint(mirrored.additive.lower), _endpoint(mirrored.additive.upper)
    assert m_add_lo == -add_hi
    assert m_add_hi == -add_lo
    lo, hi = _endpoint(region.relative.lower), _endpoint(region.relative.upper)
    mirror_lo = _endpoint(mirrored.relative.lower)
    mirror_hi = _endpoint(mirrored.relative.upper)
    assert _negation_ulps(math.log1p(lo), math.log1p(mirror_hi)) <= 8
    assert _negation_ulps(math.log1p(hi), math.log1p(mirror_lo)) <= 8
    width = math.log1p(hi) - math.log1p(lo)
    mirror_width = math.log1p(mirror_hi) - math.log1p(mirror_lo)
    assert abs(width - mirror_width) <= 4 * math.ulp(width)
    # The twin's gate indicators are the source's: A/A truth is zero, and the
    # log-scale interval is negated, so coverage and both widths carry over.
    assert (lo <= 0 <= hi) == (mirror_lo <= 0 <= mirror_hi)
    assert (add_lo <= 0 <= add_hi) == (m_add_lo <= 0 <= m_add_hi)


@pytest.mark.parametrize(
    "case_id,quantiles",
    (
        ("winsor-aa-ln-s1.6-n50-1x4-p0.95", (0.95, 0.99)),
        ("winsor-aa-contamination-n50-1x1-p0.99", (0.99, 0.95)),
        ("winsor-alt-n50-4x1-p0.95", (0.95, 0.99)),
        ("winsor-aa-gamma-k2-n50-1x1-p0.95", (0.95,)),
    ),
)
def test_shared_quantile_draw_matches_the_production_kernel(case_id, quantiles):
    """Sharing the replicate draw across quantiles changes nothing numerically.

    Model equality covers every root, pilot, centering target, endpoint status
    and refusal reason, so this pins the shared-draw evaluator to the frozen
    kernel rather than to a summary of it.
    """
    from increment.estimation.winsor import estimate_winsor_lift

    control, treatment = _campaign_sample(CASES[case_id], draw=2)
    shared = _winsor_shared_quantile_regions(control, treatment, quantiles, stream=11)
    assert len(shared) == len(quantiles)
    for region, quantile in zip(shared, quantiles, strict=True):
        raw = _winsor_raw_state(control, treatment, quantile, stream=11)
        expected = estimate_winsor_lift(raw, "control", "treatment").confidence_set
        assert expected is not None
        assert region == expected


def test_a_refusing_quantile_leaves_its_draw_mates_untouched():
    """A quantile that refuses must not take its siblings down with it.

    Sharing a replicate draw joins the cells' compute, not their outcomes: a
    refusal belongs to the quantile that produced it, and the sibling's region
    stays exactly what it would have been on its own.
    """
    from increment.errors import CodedError
    from increment.estimation._winsor_bootstrap import (
        bootstrap_confidence_set,
        full_procedure_bootstrap_reference,
    )

    # Clipping at 0.3 collapses every retained value onto the tied cutoff, so
    # the observed studentization is unavailable there and only there.
    control = (1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0, 5.0)
    treatment = (1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 7.0)
    refused, kept = _winsor_shared_quantile_regions(control, treatment, (0.3, 0.99), stream=0)
    assert isinstance(refused, CodedError)
    assert refused.code == "estimation.winsor.studentization_degenerate"
    expected = bootstrap_confidence_set(
        full_procedure_bootstrap_reference(
            _winsor_raw_state(control, treatment, 0.99, stream=0), "control", "treatment"
        )
    )
    assert kept == expected
    with pytest.raises(CodedError) as caught:
        _winsor_array_region(control, treatment, 0.3, stream=0)
    assert caught.value.code == refused.code


def test_allocation_swap_collapse_certifies_every_declared_twin():
    """Every A/A swap twin is certified by a sibling's simulation, not re-run.

    The declared design keeps both (n_c, n_t) and (n_t, n_c); the campaign runs
    one of them. This pins the accounting so a twin can never be silently
    dropped, nor silently reported as executed.
    """
    from scripts.run_i15_campaign import _is_swap_image, _selection

    class _Args:
        case_ids = None
        selection = "full"
        diagnostic_override = False
        max_draws = None

    plan = _selection(_Args(), MANIFEST)
    rows = {row["case_id"]: row for row in plan["cases"]}
    assert set(rows) == set(CASES)
    derived = {
        case_id: row["certified_by"]["case_id"]
        for case_id, row in rows.items()
        if row["execution"] == "certified_by_equivariance"
    }
    swapped_cells = {
        case["case_id"]
        for case in WINSOR_CASES
        if case["kind"] == "aa" and case["dgp"]["n_c"] > case["dgp"]["n_t"]
    }
    assert set(derived) == swapped_cells
    assert len(derived) == 40
    for twin, source in derived.items():
        assert _is_swap_image(CASES[source], CASES[twin])
        assert rows[source]["execution"] == "executed"
        assert rows[twin]["directory"] is None
        assert rows[twin]["certified_by"]["argument"] == "aa_allocation_swap_equivariance"
        assert rows[twin]["requested"] == rows[source]["requested"]
    executed = [row for row in plan["cases"] if row["execution"] == "executed"]
    assert len(executed) == len(CASES) - 40
    # Every winsor cell pairs with its clipping-quantile sibling on one stream.
    units = {unit["unit"]: unit for unit in plan["units"]}
    assert len(units) == 55
    for unit in units.values():
        members = [CASES[case_id] for case_id in unit["case_ids"]]
        assert unit["seed"] == members[0]["seed"]
        if len(members) == 1:
            assert members[0].get("kind") not in ("aa", "alternative")
            continue
        assert {case["dgp"]["quantile"] for case in members} == {0.95, 0.99}
        assert len({(case["dgp"]["n_c"], case["dgp"]["n_t"]) for case in members}) == 1
    assert sum(1 for unit in units.values() if unit["shares_bootstrap_draw"]) == 46


def test_allocation_swap_collapse_refuses_a_broken_premise():
    """A candidate twin that is not an exact swap image must fail loudly."""
    import copy

    from scripts.run_i15_campaign import _allocation_swap_partners

    source = copy.deepcopy(CASES["winsor-aa-ln-s0.5-n50-1x4-p0.95"])
    twin = copy.deepcopy(CASES["winsor-aa-ln-s0.5-n50-4x1-p0.95"])
    assert _allocation_swap_partners([source, twin]) == {twin["case_id"]: source["case_id"]}
    twin["reference"]["log_ratio_variance"]["full"] *= 1.5
    with pytest.raises(ValueError):
        _allocation_swap_partners([source, twin])


def test_swap_exactness_tracks_lift_scale_representability():
    """The collapse holds a draw only when the twin's endpoints are representable.

    Log-scale negation is exact, but ``expm1(b)`` can be a finite lift while
    ``expm1(-b)`` collapses onto -1. Such a draw cannot carry the twin's
    availability count, so the campaign must not count it as exact.
    """
    from scripts.run_i15_campaign import _observation, _swap_exact

    case = CASES["winsor-aa-ln-s0.5-n50-1x4-p0.95"]

    def detail(interval, statuses):
        return _observation(case, {"interval": interval, "interval_status": statuses, "point": 0.0})

    assert _swap_exact("completed", detail((-0.3, 0.4), ("finite", "finite")))
    assert _swap_exact("refused", {})
    assert not _swap_exact("failed", {})
    assert math.expm1(-40.0) == -1.0
    assert not _swap_exact("completed", detail((-0.5, math.expm1(40.0)), ("finite", "finite")))
    assert not _swap_exact("completed", detail((None, 0.4), ("undefined", "finite")))
    assert not _swap_exact("completed", detail((-math.inf, 0.4), ("unbounded", "finite")))


def test_a_twin_is_only_certified_by_a_source_that_can_carry_it():
    """Evidence must withhold the twin whenever its source falls short.

    The collapse is only as good as the simulation behind it, so every way a
    source can fail to stand in for its twin has to surface as an uncertified
    row rather than a silently inherited pass.
    """
    from scripts.run_i15_campaign import _equivariance_record

    item = {"case_id": "twin", "requested": 100, "execution": "certified_by_equivariance"}
    source = {
        "case_id": "source",
        "status": "diagnostic_completed",
        "certifies_by_equivariance": "twin",
        "requested": 100,
        "attempted": 100,
        "swap_exact": 100,
        "historical_report": "unit-0000/source/calibration.json",
    }
    accepted = _equivariance_record(item, source)
    assert accepted["status"] == "certified_by_equivariance"
    assert accepted["executed"] is False
    assert accepted["uncertified_reason"] is None
    assert "attempted" not in accepted and "completed" not in accepted
    assert accepted["historical_report"] == source["historical_report"]

    for damage, expected in (
        ({"status": "incomplete"}, "certifying_case_incomplete"),
        ({"certifies_by_equivariance": None}, "certifying_case_did_not_track_this_twin"),
        ({"requested": 99}, "certifying_case_ran_a_different_repetition_count"),
        (
            {"swap_exact": 99},
            "certifying_case_has_draws_whose_swap_image_is_unrepresentable",
        ),
    ):
        record = _equivariance_record(item, {**source, **damage})
        assert record["status"] == "incomplete"
        assert record["uncertified_reason"] == expected
    assert _equivariance_record(item, None)["uncertified_reason"] == "certifying_case_incomplete"


@pytest.mark.slow
def test_exact_case_worker_records_its_draws_in_the_journal(tmp_path):
    """The exact path takes a journal, not a directory.

    It once received the output Path where a DrawJournal was due, so the first
    draw raised AttributeError and every exact case lost its accounting.
    """
    from calibration.journal import verify
    from scripts.run_i15_campaign import _worker

    case_id = "winsor-exact-labels"
    (tmp_path / "unit.json").write_text(
        json.dumps({"unit": "unit-0000", "case_ids": [case_id], "requested": 3})
    )
    assert _worker(tmp_path) == 0

    totals = verify(tmp_path / case_id, case_id=case_id)
    assert (totals.started, totals.completed) == (3, 3)
    assert totals.counters.get("completed") == 3
    assert not any(path.name.startswith("draw-") for path in (tmp_path / case_id).iterdir())
