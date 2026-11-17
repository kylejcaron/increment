"""Replay the two frozen raw-outcome studies with production relative Normal laws.

The protocols beside this module are unchanged historical registrations, not a
new registration of the production evaluator. Working-model calibration is not
part of this command. Raw discrepancy flags are findings, not execution failures.

For positive finite arm estimands T,C, x=T/C-1 and
s²=Vt/C²+T²*Vc/C⁴=(T/C)²*(Vt/T²+Vc/C²). Ratio arms use
Var(N-R*D)/(n*Dbar²), checked against the centered covariance expansion.
Shared-reference adjustment instead retains both independent raw-arm influence
scores and checks their variance against direct relative-gradient arithmetic.
These identities have a scale-relative floating-point tolerance of 1e-10;
they do not bound error in the raw-data Normal likelihood approximation.

The analysis law is unrestricted: r~Normal(0,.1²), x|r,s~Normal(r,s²).
Its update and CDF/survival queries come only from production code. No posterior
probability error enclosure is supplied by that API. Admission uses infer_lift
with prior=None and is not a public-ingress certification. Fixed-truth posterior
inclusion has no nominal-coverage gate. Proxy contrasts combine likelihood,
estimated-SE dependence and admission effects, rather than isolating one cause.
Prior-predictive PIT diagnostics concern the admitted population only.

Run from the repository root, with single-core numerical libraries::

    uv run --no-sync python -m scripts.test_slots -- python -m calibration.relative_prior_validation --phase all --output /tmp/relative-prior-raw.json
    uv run --no-sync python -m scripts.test_slots -- python -m calibration.relative_prior_validation --phase fixed_truth --scenario fixed_truth:shared_reference_cuped:200:0.2 --output /tmp/relative-prior-fixed.json

Repeat --scenario for a subset of original coordinates. Selection never changes
N, seeds, draw order, or stream indices. JSON is saved after every completed
scenario; an incomplete run never reports execution_checks_passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.special import ndtr, ndtri
from scipy.stats import binomtest

from increment.estimation import _relative_conjugate, inference
from increment.estimation._relative_conjugate import RelativeNormalPosterior
from increment.estimation.inference import LiftGuardError, infer_lift, normal_posterior
from increment.estimation.priors import Normal

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = Path(__file__).resolve().parent / "protocols"
FIXED_SEED = 2026111503
PREDICTIVE_SEED = 2026111504
FIXED_N = 4096
PREDICTIVE_N = 2048
BATCH = 128
IDENTITY_TOLERANCE = 1e-10
PRIOR = Normal(mu=0, sigma=0.1)
FAMILIES = (
    "gamma",
    "lognormal",
    "dense_binary",
    "sparse_binary",
    "ratio",
    "fixed_reference_cuped",
    "shared_reference_cuped",
)
FIXED_SCENARIOS = tuple(
    (family, n, effect) for family in FAMILIES for n in (200, 2000) for effect in (0.0, 0.2)
)
PREDICTIVE_SCENARIOS = tuple((family, n) for family in FAMILIES for n in (200, 2000))
HISTORICAL = {
    "fixed_truth": {
        "protocol": "relative_prior_primary.md",
        "protocol_sha256": "998850b9e6f7cc7911ad5e57b7b9532e98992e82491310f0a4a6bc4e9aadb3c8",
        "source_sha256": "db4428e34be0e8bece889a0360ecf897b2c4c13c0bfc321e4e4792184454f1b1",
        "output_sha256": "fcbde6cfbb43f9aa39fb4e508642fdd41e74c6ce91d44830dedcff16663ebd2d",
    },
    "prior_predictive": {
        "protocol": "relative_prior_secondary.md",
        "protocol_sha256": "21f2dbea044404a311efd28aa1139ff40dfbaf45ced650a1d1a35db77fa4fa2a",
        "source_sha256": "9a9350ac5cd07809a28d400c83a6371b1c422fab3b37b1339c61159e1746a1c5",
        "output_sha256": "9331f415f1e5c064b6a6be6a1eca6056750eb32ed2aeac8955604d088dbe5573",
    },
}


def _save(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _variance(x: np.ndarray) -> np.ndarray:
    centered = x - np.mean(x, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", centered, centered) / (x.shape[1] - 1)


def _covariance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - np.mean(x, axis=1, keepdims=True)
    yc = y - np.mean(y, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", xc, yc) / (x.shape[1] - 1)


def _relative_discrepancy(left: Any, right: Any) -> Any:
    return np.abs(left - right) / np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)


def _shared_adjustment(
    xc: np.ndarray, xt: np.ndarray, ec: np.ndarray, et: np.ndarray, n: int
) -> tuple[np.ndarray, ...]:
    beta = 0.8
    pooled = (np.mean(xc, axis=1) + np.mean(xt, axis=1)) / 2
    c, t = np.mean(ec, axis=1) + beta * pooled, np.mean(et, axis=1) + beta * pooled
    ok = (c > 0) & (t > 0) & np.isfinite(c) & np.isfinite(t)
    logc, logt, direct = (np.full(len(c), np.nan) for _ in range(3))
    g = beta / 2 * (1 / t[ok] - 1 / c[ok])
    # Centering in _variance retains the original independent-arm scores.
    score_c = g[:, None] * xc[ok] - ec[ok] / c[ok, None]
    score_t = g[:, None] * xt[ok] + et[ok] / t[ok, None]
    logc[ok], logt[ok] = _variance(score_c) / n, _variance(score_t) / n
    gr = beta / 2 * (1 / c[ok] - t[ok] / c[ok] ** 2)
    gc, gt = -t[ok] / c[ok] ** 2, 1 / c[ok]
    direct[ok] = (
        gr**2 * (_variance(xc[ok]) + _variance(xt[ok]))
        + gc**2 * _variance(ec[ok])
        + gt**2 * _variance(et[ok])
        + 2 * gr * gc * _covariance(xc[ok], ec[ok])
        + 2 * gr * gt * _covariance(xt[ok], et[ok])
    ) / n
    return c, t, logc, logt, direct


def _ratio_arms(
    means: tuple[float, float], n: int, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    points, variances, errors = [], [], []
    for mean in means:
        denominator = rng.gamma(2.0, 1.0, size=(count, n))
        numerator = denominator * rng.exponential(mean, size=(count, n))
        dbar = np.mean(denominator, axis=1)
        ratio = np.mean(numerator, axis=1) / dbar
        vcov = (
            _variance(numerator)
            + ratio**2 * _variance(denominator)
            - 2 * ratio * _covariance(numerator, denominator)
        ) / (n * dbar**2)
        vscore = _variance(numerator - ratio[:, None] * denominator) / (n * dbar**2)
        points.append(ratio)
        variances.append(vcov)
        errors.append(float(np.max(_relative_discrepancy(vcov, vscore))))
    return points[0], points[1], variances[0], variances[1], max(errors)


def _generated_batch(
    family: str, n: int, effect: float, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Original raw DGP and draw order, including centered variance checks."""
    means = (1.0, 1.0 + effect)
    ratio_error = 0.0
    if family.endswith("cuped"):
        beta = 0.8
        xc, xt = (rng.normal(size=(count, n)) for _ in range(2))
        yc = means[0] + beta * xc + rng.normal(scale=0.6, size=(count, n))
        yt = means[1] + beta * xt + rng.normal(scale=0.6, size=(count, n))
        ec, et = yc - beta * xc, yt - beta * xt
        if family == "shared_reference_cuped":
            c, t, logc, logt, direct = _shared_adjustment(xc, xt, ec, et, n)
            return c, t, logc, logt, direct, ratio_error
        c, t = np.mean(ec, axis=1), np.mean(et, axis=1)
        vc, vt = _variance(ec) / n, _variance(et) / n
    elif family == "ratio":
        c, t, vc, vt, ratio_error = _ratio_arms(means, n, count, rng)
    else:
        arrays = []
        for mean in means:
            if family == "gamma":
                values = rng.gamma(1.0, mean, size=(count, n))
            elif family == "lognormal":
                values = rng.lognormal(
                    math.log(mean) - math.log(10) / 2, math.sqrt(math.log(10)), size=(count, n)
                )
            else:
                p = (0.2 if family == "dense_binary" else 0.01) * mean
                values = rng.binomial(1, p, size=(count, n)).astype(float)
            arrays.append(values)
        c, t = (np.mean(a, axis=1) for a in arrays)
        vc, vt = (_variance(a) / n for a in arrays)
    ok = (c > 0) & (t > 0) & np.isfinite(c) & np.isfinite(t)
    logc, logt, direct = (np.full(count, np.nan) for _ in range(3))
    logc[ok] = vc[ok] / c[ok] ** 2
    logt[ok] = vt[ok] / t[ok] ** 2
    direct[ok] = vt[ok] / c[ok] ** 2 + t[ok] ** 2 * vc[ok] / c[ok] ** 4
    return c, t, logc, logt, direct, ratio_error


def _estimand_exclusion(c: float, t: float) -> str | None:
    if not math.isfinite(c) or not math.isfinite(t):
        return "nonfinite_estimand"
    if c <= 0 and t <= 0:
        return "nonpositive_both"
    if c <= 0:
        return "nonpositive_control"
    if t <= 0:
        return "nonpositive_treatment"
    return None


def _admit(
    c: float, t: float, logc: float, logt: float, direct: float
) -> tuple[float, float, float, float]:
    se_c, se_t = math.sqrt(logc), math.sqrt(logt)
    row = infer_lift(
        "validation",
        "treatment",
        "unadjusted",
        math.log(t / c),
        se_t,
        se_c,
        prior=None,
        method_role="decision",
    )
    ratio = t / c
    x, s = ratio - 1, ratio * math.hypot(se_c, se_t)
    discrepancy = float(_relative_discrepancy(s * s, direct))
    if not math.isfinite(discrepancy) or discrepancy > IDENTITY_TOLERANCE:
        raise ArithmeticError(f"Variance identities disagree: {discrepancy}")
    return x, s, discrepancy, abs(row.require_lift().value - x)


def _posterior_metrics(x: float, s: float, truth: float) -> dict[str, float]:
    law = RelativeNormalPosterior(normal_posterior(x, s, prior=PRIOR))
    pit, forecast, below = law.cdf(truth), law.survival(0.0), law.cdf(-1.0)
    if not all(math.isfinite(p) and 0 <= p <= 1 for p in (pit, forecast, below)):
        raise ArithmeticError(f"Invalid posterior probabilities: {pit}, {forecast}, {below}")
    return {
        "pit": pit,
        "likelihood_coverage": float(abs(x - truth) <= -float(ndtri(0.025)) * s),
        "posterior_coverage": float(0.025 <= pit <= 0.975),
        "probability_favorable": forecast,
        "decision": float(forecast >= 0.95),
        "below_minus_one": below,
    }


def _failure(index: int, stage: str, exc: Exception, **values: float) -> dict[str, Any]:
    # Invalid inputs remain explicit without emitting nonstandard JSON NaN/Infinity.
    context = {
        key: value if math.isfinite(value) else {"nonfinite": str(value)}
        for key, value in values.items()
    }
    return {
        "index": index,
        "stage": stage,
        **context,
        "exception": type(exc).__name__,
        "detail": str(exc),
    }


def _check_accounting(result: dict[str, Any]) -> None:
    accounted = (
        result["admitted"]
        + sum(result.get("generating_exclusions", {}).values())
        + sum(result["estimand_exclusions"].values())
        + sum(result["estimator_refusals"].values())
        + len(result["unexpected_failures"])
    )
    if accounted != result["attempted"]:
        raise ArithmeticError(
            f"Attempted accounting mismatch: {accounted} != {result['attempted']}"
        )
    result["accounting_checks_passed"] = True


def _fixed_diagnostics(
    x: np.ndarray,
    s: np.ndarray,
    effect: float,
    raw: list[dict[str, float]],
    proxy: list[dict[str, float]],
) -> dict[str, Any]:
    differences, flags = {}, []
    for name in ("likelihood_coverage", "posterior_coverage", "decision"):
        a = np.asarray([row[name] for row in raw])
        b = np.asarray([row[name] for row in proxy])
        delta = a - b
        mean = float(np.mean(delta))
        se = float(np.std(delta, ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else None
        flag = se is not None and abs(mean) > max(0.03, 4 * se)
        differences[name] = {
            "raw": float(np.mean(a)),
            "proxy": float(np.mean(b)),
            "difference": mean,
            "monte_carlo_se": se,
            "flag_for_investigation": flag,
        }
        if flag:
            flags.append(name)
    below = np.asarray([row["below_minus_one"] for row in raw])
    return {
        "lift_bias": float(np.mean(x - effect)),
        "lift_bias_monte_carlo_se": float(np.std(x, ddof=1) / math.sqrt(len(x)))
        if len(x) > 1
        else None,
        "standardized_error_mean": float(np.mean((x - effect) / s)),
        "standardized_error_quantiles": np.quantile((x - effect) / s, [0.025, 0.5, 0.975]).tolist(),
        "mean_probability_favorable": float(np.mean([r["probability_favorable"] for r in raw])),
        "proxy_mean_probability_favorable": float(
            np.mean([r["probability_favorable"] for r in proxy])
        ),
        "mean_posterior_mass_below_minus_one": float(np.mean(below)),
        "max_posterior_mass_below_minus_one": float(np.max(below)),
        "contrasts": differences,
        "flags": flags,
    }


def _fixed_posteriors(
    result: dict[str, Any],
    admitted: list[tuple[int, float, float]],
    effect: float,
    proxy_rng: np.random.Generator,
) -> None:
    result["numerical_seam_admitted"] = len(admitted)
    x = np.asarray([row[1] for row in admitted])
    s = np.asarray([row[2] for row in admitted])
    proxies = effect + s * proxy_rng.standard_normal(len(x))
    raw_metrics, proxy_metrics = [], []
    for (index, observed, error), proxy in zip(admitted, proxies, strict=True):
        try:
            raw = _posterior_metrics(observed, error, effect)
            comparison = _posterior_metrics(float(proxy), error, effect)
            raw_metrics.append(raw)
            proxy_metrics.append(comparison)
        except Exception as exc:
            result["unexpected_failures"].append(
                _failure(
                    index,
                    "production_posterior",
                    exc,
                    observed=observed,
                    s=error,
                    proxy=float(proxy),
                )
            )
    result["admitted"] = len(raw_metrics)
    result["posterior_evaluation_failures"] = len(admitted) - len(raw_metrics)
    result["diagnostics"] = None
    if result["posterior_evaluation_failures"]:
        result["diagnostics_status"] = "unavailable_unexpected_failures"
    elif not admitted:
        result["diagnostics_status"] = "unavailable_no_admitted_experiments"
    else:
        result["diagnostics_status"] = "measured"
        result["diagnostics"] = _fixed_diagnostics(x, s, effect, raw_metrics, proxy_metrics)


def _fixed_scenario(
    family: str, n: int, effect: float, rng: np.random.Generator, proxy_rng: np.random.Generator
) -> dict[str, Any]:
    excluded: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    unexpected = []
    admitted = []
    maximum_identity_error = ratio_identity_error = sampling_point_error = 0.0
    for offset in range(0, FIXED_N, BATCH):
        count = min(BATCH, FIXED_N - offset)
        c, t, logc, logt, direct, ratio_error = _generated_batch(family, n, effect, count, rng)
        ratio_identity_error = max(ratio_identity_error, ratio_error)
        for i in range(count):
            control, treatment, lc, lt, variance = (float(v[i]) for v in (c, t, logc, logt, direct))
            exclusion = _estimand_exclusion(control, treatment)
            if exclusion is not None:
                excluded[exclusion] += 1
                continue
            try:
                x, s, discrepancy, point_error = _admit(control, treatment, lc, lt, variance)
                maximum_identity_error = max(maximum_identity_error, discrepancy)
                sampling_point_error = max(sampling_point_error, point_error)
                admitted.append((offset + i, x, s))
            except LiftGuardError as exc:
                refused[exc.reason] += 1
            except Exception as exc:
                unexpected.append(
                    _failure(
                        offset + i,
                        "admission_and_variance",
                        exc,
                        control=control,
                        treatment=treatment,
                        log_variance_control=lc,
                        log_variance_treatment=lt,
                    )
                )
    result = {
        "family": family,
        "n_per_arm": n,
        "true_lift": effect,
        "attempted": FIXED_N,
        "estimand_exclusions": dict(excluded),
        "estimator_refusals": dict(refused),
        "unexpected_failures": unexpected,
        "variance_identity_max_relative_error": maximum_identity_error,
        "ratio_variance_identity_max_relative_error": ratio_identity_error,
        "sampling_point_max_absolute_difference": sampling_point_error,
        "identity_checks_passed": maximum_identity_error <= IDENTITY_TOLERANCE
        and ratio_identity_error <= IDENTITY_TOLERANCE,
    }
    _fixed_posteriors(result, admitted, effect, proxy_rng)
    _check_accounting(result)
    return result


def _generating_exclusion(family: str, truth: float) -> str | None:
    if not family.endswith("cuped") and truth <= -1:
        return "nonpositive_true_positive_outcome_mean"
    if "binary" in family:
        baseline = 0.2 if family == "dense_binary" else 0.01
        if baseline * (1 + truth) >= 1:
            return "true_binary_probability_at_least_one"
    return None


def _reliability_bins(p: np.ndarray, positive: np.ndarray) -> list[dict[str, Any]]:
    bins = []
    for j in range(10):
        lower, upper = j / 10, (j + 1) / 10
        selected = (p >= lower) & ((p < upper) if j < 9 else (p <= upper))
        count = int(np.count_nonzero(selected))
        bins.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_forecast": float(np.mean(p[selected])) if count else None,
                "fraction_positive": float(np.mean(positive[selected])) if count else None,
                "sparse": count < 20,
            }
        )
    return bins


def _predictive_diagnostics(
    result: dict[str, Any], metrics: list[dict[str, float]], signs: list[bool]
) -> None:
    m = len(metrics)
    if result["posterior_evaluation_failures"] or not m:
        status = (
            "unavailable_unexpected_failures"
            if result["posterior_evaluation_failures"]
            else "unavailable_no_admitted_experiments"
        )
        result.update(
            calibration_status=status,
            pit_distance=None,
            diagnostic_bound=None,
            discrepancy_flag=None,
            reliability_bins=[],
        )
        return
    u = np.asarray([row["pit"] for row in metrics])
    p = np.asarray([row["probability_favorable"] for row in metrics])
    positive = np.asarray(signs)
    ordered = np.sort(u)
    d = float(max(np.max(np.arange(1, m + 1) / m - ordered), np.max(ordered - np.arange(m) / m)))
    bound = math.sqrt(math.log(2 / (0.01 / 14)) / (2 * m))
    covered = int(np.count_nonzero((u >= 0.025) & (u <= 0.975)))
    test = binomtest(covered, m, p=0.95)
    ci = test.proportion_ci(confidence_level=0.99, method="exact")
    result.update(
        calibration_status="measured",
        pit_distance=d,
        diagnostic_bound=bound,
        discrepancy_flag=d > bound,
        pit_mean=float(np.mean(u)),
        pit_quantiles=np.quantile(u, [0.025, 0.5, 0.975]).tolist(),
        central_covered=covered,
        central_coverage=covered / m,
        central_coverage_interval=[ci.low, ci.high],
        exact_coverage_pvalue=test.pvalue,
        mean_probability_favorable=float(np.mean(p)),
        actual_positive_fraction=float(np.mean(positive)),
        reliability_bins=_reliability_bins(p, positive),
        mean_posterior_mass_below_minus_one=float(np.mean([r["below_minus_one"] for r in metrics])),
        max_posterior_mass_below_minus_one=max(r["below_minus_one"] for r in metrics),
    )


def _predictive_scenario(family: str, n: int, rng: np.random.Generator) -> dict[str, Any]:
    generated: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    refused: Counter[str] = Counter()
    failures, metrics, signs = [], [], []
    maximum_identity_error = ratio_identity_error = sampling_point_error = 0.0
    numerical_seam_admitted = posterior_failures = 0
    for index in range(PREDICTIVE_N):
        truth = float(rng.normal(0.0, 0.1))
        exclusion = _generating_exclusion(family, truth)
        if exclusion is not None:
            generated[exclusion] += 1
            continue
        c, t, logc, logt, direct, ratio_error = _generated_batch(family, n, truth, 1, rng)
        ratio_identity_error = max(ratio_identity_error, ratio_error)
        c, t, logc, logt, direct = (float(v[0]) for v in (c, t, logc, logt, direct))
        exclusion = _estimand_exclusion(c, t)
        if exclusion is not None:
            excluded[exclusion] += 1
            continue
        try:
            x, s, discrepancy, point_error = _admit(c, t, logc, logt, direct)
            maximum_identity_error = max(maximum_identity_error, discrepancy)
            sampling_point_error = max(sampling_point_error, point_error)
        except LiftGuardError as exc:
            refused[exc.reason] += 1
            continue
        except Exception as exc:
            failures.append(
                _failure(index, "admission_and_variance", exc, truth=truth, control=c, treatment=t)
            )
            continue
        numerical_seam_admitted += 1
        try:
            metrics.append(_posterior_metrics(x, s, truth))
            signs.append(truth > 0)
        except Exception as exc:
            posterior_failures += 1
            failures.append(
                _failure(index, "production_posterior", exc, truth=truth, observed=x, s=s)
            )
    result = {
        "family": family,
        "n_per_arm": n,
        "attempted": PREDICTIVE_N,
        "generating_exclusions": dict(generated),
        "estimand_exclusions": dict(excluded),
        "estimator_refusals": dict(refused),
        "unexpected_failures": failures,
        "admitted": len(metrics),
        "numerical_seam_admitted": numerical_seam_admitted,
        "posterior_evaluation_failures": posterior_failures,
        "variance_identity_max_relative_error": maximum_identity_error,
        "ratio_variance_identity_max_relative_error": ratio_identity_error,
        "sampling_point_max_absolute_difference": sampling_point_error,
        "identity_checks_passed": maximum_identity_error <= IDENTITY_TOLERANCE
        and ratio_identity_error <= IDENTITY_TOLERANCE,
    }
    _check_accounting(result)
    _predictive_diagnostics(result, metrics, signs)
    return result


def _scenario_ids() -> tuple[list[str], list[str]]:
    fixed = [f"fixed_truth:{family}:{n}:{effect}" for family, n, effect in FIXED_SCENARIOS]
    predictive = [f"prior_predictive:{family}:{n}" for family, n in PREDICTIVE_SCENARIOS]
    return fixed, predictive


def _metadata(phase: str, selected: list[str]) -> dict[str, Any]:
    protocols = {}
    for name, historical in HISTORICAL.items():
        path = PROTOCOLS / historical["protocol"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != historical["protocol_sha256"]:
            raise RuntimeError(f"Historical protocol hash changed: {path}")
        protocols[name] = {**historical, "protocol_path": str(path)}
    revision = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    sources = {
        "runner": Path(__file__).resolve(),
        "normal_posterior_and_admission": Path(inference.__file__).resolve(),
        "relative_normal_queries": Path(_relative_conjugate.__file__).resolve(),
    }
    fixed_ids, predictive_ids = _scenario_ids()
    return {
        "scope": "raw-only historical DGP replay with production relative Normal posterior evaluation",
        "registration_status": "unchanged historical protocols; production evaluator replacement is not a new registration",
        "historical_protocols": protocols,
        "historical_inference_baseline": "5ebc8e9f0791f2c0cadf9f1894ec82a817fba768",
        "git_revision": revision,
        "sources": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in sources.items()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "analysis_prior": {
            "family": "normal",
            "mean": 0,
            "sd": 0.1,
            "support": "unrestricted real line",
        },
        "probability_error_estimate": None,
        "probability_error_note": "production CDF/survival API supplies no error estimate or rigorous enclosure",
        "admission": "infer_lift(prior=None, method_role='decision'); numerical seam, not public ingress",
        "admitted_definition": "completed posterior evaluations; numerical_seam_admitted also retains evaluator failures",
        "raw_approximation_is_diagnostic_not_pass_gate": True,
        "phase": phase,
        "selected_scenarios": selected,
        "selection_scope": "complete_raw_roster"
        if selected == fixed_ids + predictive_ids
        else "selected_original_coordinates",
        "roster": {"fixed_truth": fixed_ids, "prior_predictive": predictive_ids},
        "fixed_truth": {
            "seed": FIXED_SEED,
            "spawn_count": 60,
            "reserved_working_model_streams": [0, 1, 2, 3],
            "attempted_per_scenario": FIXED_N,
            "batch": BATCH,
            "contrast_trigger": "abs(difference) > max(0.03, 4*MCSE)",
            "scenarios": [],
        },
        "prior_predictive": {
            "seed": PREDICTIVE_SEED,
            "spawn_count": 14,
            "attempted_per_scenario": PREDICTIVE_N,
            "generating_lower_invalid_mass_positive_families": float(ndtr(-10)),
            "generating_restrictions": "positive means and binary probabilities only; no bound for signed adjustment families; analysis prior unchanged",
            "diagnostic_alpha_per_cell": 0.01 / 14,
            "scenarios": [],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--phase", choices=("fixed_truth", "prior_predictive", "all"), default="all"
    )
    parser.add_argument("--output", type=Path, required=True)
    fixed_ids, predictive_ids = _scenario_ids()
    parser.add_argument(
        "--scenario",
        action="append",
        choices=fixed_ids + predictive_ids,
        help="repeat to select original coordinates without renumbering streams",
    )
    args = parser.parse_args(argv)
    available = (
        fixed_ids
        if args.phase == "fixed_truth"
        else predictive_ids
        if args.phase == "prior_predictive"
        else fixed_ids + predictive_ids
    )
    if args.scenario and any(name not in available for name in args.scenario):
        parser.error("every --scenario must belong to --phase")
    selected = [name for name in available if not args.scenario or name in args.scenario]
    result = _metadata(args.phase, selected)
    started = time.perf_counter()
    # Spawn the entire original roster even when evaluating a selected subset.
    fixed_streams = np.random.SeedSequence(FIXED_SEED).spawn(4 + 2 * len(FIXED_SCENARIOS))
    predictive_streams = np.random.SeedSequence(PREDICTIVE_SEED).spawn(len(PREDICTIVE_SCENARIOS))
    _save(args.output, result)
    for index, (family, n, effect) in enumerate(FIXED_SCENARIOS):
        if fixed_ids[index] not in selected:
            continue
        data, proxy = fixed_streams[4 + 2 * index], fixed_streams[5 + 2 * index]
        record = _fixed_scenario(
            family, n, effect, np.random.default_rng(data), np.random.default_rng(proxy)
        )
        record.update(
            scenario_id=fixed_ids[index],
            original_scenario_index=index,
            data_seed_spawn_key=list(data.spawn_key),
            proxy_seed_spawn_key=list(proxy.spawn_key),
        )
        result["fixed_truth"]["scenarios"].append(record)
        _save(args.output, result)
        print(json.dumps(record, allow_nan=False), flush=True)
    for index, (family, n) in enumerate(PREDICTIVE_SCENARIOS):
        if predictive_ids[index] not in selected:
            continue
        stream = predictive_streams[index]
        record = _predictive_scenario(family, n, np.random.default_rng(stream))
        record.update(
            scenario_id=predictive_ids[index],
            original_scenario_index=index,
            seed_spawn_key=list(stream.spawn_key),
        )
        result["prior_predictive"]["scenarios"].append(record)
        _save(args.output, result)
        print(json.dumps(record, allow_nan=False), flush=True)
    rows = result["fixed_truth"]["scenarios"] + result["prior_predictive"]["scenarios"]
    passed = len(rows) == len(selected) and all(
        not row["unexpected_failures"]
        and row["identity_checks_passed"]
        and row["accounting_checks_passed"]
        for row in rows
    )
    result.update(elapsed_seconds=time.perf_counter() - started, execution_checks_passed=passed)
    _save(args.output, result)
    print(
        f"COMPLETED raw_only phase={args.phase} scenarios={len(rows)}/{len(selected)} execution_checks_passed={passed}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
