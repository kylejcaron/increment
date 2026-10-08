"""Small matched CUPED/regression efficiency comparison through Analysis.run."""

from __future__ import annotations

import json
import math
import time

import numpy as np
from scipy.stats import t as student_t


def _ols_relative_interval(y: np.ndarray, x: np.ndarray, n: int) -> tuple[float, float]:
    """Matched OLS lift interval with numerator/denominator delta covariance."""
    treatment = np.repeat(np.array([0.0, 1.0]), n)
    design = np.column_stack((np.ones(2 * n), treatment, x))
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coef
    df = 2 * n - design.shape[1]
    sigma2 = float(residual @ residual / df)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    control_mean = float(y[:n].mean())
    control_weights = np.concatenate((np.full(n, 1.0 / n), np.zeros(n)))
    delta_covariance = sigma2 * np.linalg.solve(design.T @ design, design.T @ control_weights)
    covariance_delta_mean = float(delta_covariance[1])
    mean_variance = float(y[:n].var(ddof=1) / n)
    relative_estimate = float(coef[1] / control_mean)
    relative_variance = (
        covariance[1, 1] / control_mean**2
        + coef[1] ** 2 * mean_variance / control_mean**4
        - 2 * coef[1] * covariance_delta_mean / control_mean**3
    )
    se = math.sqrt(max(0.0, float(relative_variance)))
    critical = float(student_t.isf(0.025, df))
    return relative_estimate - critical * se, relative_estimate + critical * se


def run(*, reps: int = 50, seed: int = 20261008) -> dict[str, object]:
    import polars as pl

    from increment import Analysis, Method, MetricSpec
    from tests.analysis_factory import lift_rows

    rng = np.random.default_rng(seed)
    n = 100
    rho = 0.7
    true_lift = 0.0015
    candidate_widths: list[float] = []
    regression_widths: list[float] = []
    candidate_power = regression_power = 0
    candidate_cover = regression_cover = 0
    refusal_codes: dict[str, int] = {}
    elapsed_candidate = elapsed_regression = 0.0
    for _ in range(reps):
        x = rng.normal(size=2 * n)
        y = np.repeat(np.array([100.0, 100.0 * (1.0 + true_lift)]), n)
        y = y + rho * x + math.sqrt(1.0 - rho**2) * rng.normal(size=2 * n)
        frame = pl.DataFrame(
            {
                "unit_id": [f"u{i}" for i in range(2 * n)],
                "variant": np.repeat(np.array(["control", "treatment"]), n),
                "y": y,
                "x": x,
            }
        )
        metric = MetricSpec(name="y", type="mean", covariate="x")
        analysis = Analysis.from_unit_summary(
            frame, unit="unit_id", group="variant", control="control", metrics=[metric]
        )
        started = time.perf_counter()
        try:
            rows = lift_rows(
                analysis.run(
                    decision_method=Method(name="unadjusted"),
                    sensitivity_methods=(Method(name="cuped", variance_reduction="cuped"),),
                )
            )
        finally:
            analysis.close()
        elapsed_candidate += time.perf_counter() - started
        candidate = {row.method: row for row in rows}["cuped"]
        estimate = candidate.lift
        if estimate is None or estimate.lb is None or estimate.ub is None:
            code = candidate.failure_code or "estimate_unavailable"
            refusal_codes[code] = refusal_codes.get(code, 0) + 1
            continue
        candidate_widths.append(estimate.ub - estimate.lb)
        candidate_power += candidate.stat_sig()
        candidate_cover += estimate.lb <= true_lift <= estimate.ub

        started = time.perf_counter()
        lower, upper = _ols_relative_interval(y, x, n)
        elapsed_regression += time.perf_counter() - started
        regression_widths.append(upper - lower)
        regression_power += lower > 0 or upper < 0
        regression_cover += lower <= true_lift <= upper
    usable = len(candidate_widths)
    cuped_rate = candidate_power / usable if usable else None
    ols_rate = regression_power / usable if usable else None
    return {
        "regime": "randomized independent equal-allocation continuous outcome with pre-assignment covariate",
        "estimand": "relative mean lift, treatment minus control",
        "assignment": "independent 1:1; n=100 per arm; Gaussian outcome/covariate, rho=0.7",
        "guarantee": "both fitted CUPED and matched OLS are asymptotic Welch/t-reference comparisons",
        "reference": "OLS treatment coefficient adjusted for pre-assignment covariate",
        "seed": seed,
        "reps": reps,
        "available": usable,
        "refusals_by_code": refusal_codes,
        "mean_cuped_width": float(np.mean(candidate_widths)) if usable else None,
        "mean_ols_width": float(np.mean(regression_widths)) if regression_widths else None,
        "cuped_width_mcse": float(np.std(candidate_widths, ddof=1) / math.sqrt(usable))
        if usable > 1
        else None,
        "ols_width_mcse": float(np.std(regression_widths, ddof=1) / math.sqrt(usable))
        if usable > 1
        else None,
        "relative_width_loss": float(np.mean(candidate_widths) / np.mean(regression_widths) - 1)
        if usable and regression_widths
        else None,
        "cuped_power": cuped_rate,
        "ols_power": ols_rate,
        "cuped_coverage": candidate_cover / usable if usable else None,
        "ols_coverage": regression_cover / usable if usable else None,
        "power_mcse_cuped": math.sqrt(cuped_rate * (1 - cuped_rate) / usable)
        if usable and cuped_rate is not None
        else None,
        "power_mcse_ols": math.sqrt(ols_rate * (1 - ols_rate) / usable)
        if usable and ols_rate is not None
        else None,
        "elapsed_candidate_s": elapsed_candidate,
        "elapsed_reference_s": elapsed_regression,
    }


def main() -> int:
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
