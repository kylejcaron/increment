"""Prespecified, bounded comparisons for inference efficiency evidence.

Run with ``python -m calibration.comparative_efficiency``. Dense ordinary
comparisons use fixed 10% width and 5 percentage-point power budgets. Valid
sparse finite-sample methods retain their separate guarantee and are not judged
against asymptotic width or null-rejection targets. This evidence harness does
not replace calibration campaigns or select production routes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import t as student_t

WIDTH_TOLERANCE = 0.10
POWER_TOLERANCE = 0.05
COMPARATORS = {
    "mean": "Welch matched mean contrast",
    "risk_ratio": "Katz asymptotic log risk ratio",
    "difference": "Newcombe matched risk-difference interval",
    "adjustment": "OLS regression with treatment and pre-assignment covariate; CUPED contrast",
}


@dataclass(frozen=True, slots=True)
class Comparison:
    """One matched candidate/reference regime and separately measured outcomes."""

    regime: str
    candidate_width: float | None
    reference_width: float | None
    candidate_power: float | None
    reference_power: float | None
    candidate_available: bool
    reference_available: bool
    candidate_elapsed_s: float
    reference_elapsed_s: float
    guarantee: Literal["finite-sample", "asymptotic", "heuristic"]
    reference: str
    efficiency_comparable: bool = True
    candidate_n: int | None = None
    reference_n: int | None = None
    numerical_tolerance_loss: float = 0.0
    refusal_code: str | None = None


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Prespecified check with numerical, availability and runtime outcomes separated."""

    regime: str
    passed: bool
    width_loss: float | None
    power_loss: float | None
    required_n_inflation: float | None
    availability_loss: bool
    elapsed_ratio: float | None
    numerical_tolerance_loss: float
    refusal_code: str | None
    guarantee: str
    reference: str
    failure_dimensions: tuple[str, ...]


def evaluate_comparison(comparison: Comparison) -> ComparisonResult:
    """Apply the declared dense budgets; do not penalize valid discrete procedures."""
    width_loss = power_loss = n_inflation = None
    failures: list[str] = []
    if comparison.efficiency_comparable:
        if comparison.candidate_width is not None and comparison.reference_width is not None:
            if comparison.reference_width > 0:
                width_loss = comparison.candidate_width / comparison.reference_width - 1.0
                if comparison.candidate_width > comparison.reference_width * (
                    1.0 + WIDTH_TOLERANCE
                ):
                    failures.append("width")
        if comparison.candidate_power is not None and comparison.reference_power is not None:
            power_loss = comparison.reference_power - comparison.candidate_power
            if comparison.candidate_power < comparison.reference_power - POWER_TOLERANCE:
                failures.append("power")
        if comparison.candidate_n is not None and comparison.reference_n is not None:
            if comparison.reference_n > 0:
                n_inflation = comparison.candidate_n / comparison.reference_n - 1.0
                if comparison.candidate_n > comparison.reference_n * (1.0 + WIDTH_TOLERANCE):
                    failures.append("required_n")
    availability_loss = comparison.reference_available and not comparison.candidate_available
    if availability_loss:
        failures.append("availability")
    elapsed_ratio = (
        comparison.candidate_elapsed_s / comparison.reference_elapsed_s
        if comparison.reference_elapsed_s > 0
        else None
    )
    return ComparisonResult(
        comparison.regime,
        not failures,
        width_loss,
        power_loss,
        n_inflation,
        availability_loss,
        elapsed_ratio,
        comparison.numerical_tolerance_loss,
        comparison.refusal_code,
        comparison.guarantee,
        comparison.reference,
        tuple(failures),
    )


def _mean_case(reps: int, seed: int) -> dict[str, object]:
    """Compare production absolute-axis intervals with matched raw-array Welch intervals."""
    import polars as pl

    from increment import Analysis, MetricSpec
    from tests.analysis_factory import lift_rows

    rng = np.random.default_rng(seed)
    n = 240
    true_difference = 0.2
    candidate_widths: list[float] = []
    reference_widths: list[float] = []
    candidate_coverage = reference_coverage = 0
    candidate_power = reference_power = 0
    candidate_elapsed = reference_elapsed = 0.0
    unavailable = 0
    for rep in range(reps):
        control = rng.normal(100.0, 1.0, n)
        treatment = rng.normal(100.0 + true_difference, 1.0, n)
        frame = pl.DataFrame(
            {
                "unit_id": [f"r{rep}-u{i}" for i in range(2 * n)],
                "variant": ["control"] * n + ["treatment"] * n,
                "y": np.concatenate((control, treatment)),
            }
        )
        analysis = Analysis.from_unit_summary(
            frame,
            unit="unit_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="y", type="mean")],
        )
        started = time.perf_counter()
        try:
            (row,) = lift_rows(analysis.run())
        finally:
            analysis.close()
        candidate_elapsed += time.perf_counter() - started
        if row.abs_lb is None or row.abs_ub is None:
            unavailable += 1
            continue
        candidate_widths.append(row.abs_ub - row.abs_lb)
        candidate_coverage += row.abs_lb <= true_difference <= row.abs_ub
        candidate_power += row.abs_lb > 0 or row.abs_ub < 0

        started = time.perf_counter()
        mc, mt = float(control.mean()), float(treatment.mean())
        vc, vt = float(control.var(ddof=1)), float(treatment.var(ddof=1))
        se = math.sqrt(vc / n + vt / n)
        df = (vc / n + vt / n) ** 2 / ((vc / n) ** 2 / (n - 1) + (vt / n) ** 2 / (n - 1))
        radius = float(student_t.isf(0.025, df)) * se
        lower, upper = mt - mc - radius, mt - mc + radius
        reference_elapsed += time.perf_counter() - started
        reference_widths.append(upper - lower)
        reference_coverage += lower <= true_difference <= upper
        reference_power += lower > 0 or upper < 0
    usable = reps - unavailable
    cand_cov = candidate_coverage / usable if usable else None
    ref_cov = reference_coverage / usable if usable else None
    cand_pow = candidate_power / usable if usable else None
    ref_pow = reference_power / usable if usable else None
    return {
        "estimand": "independent mean difference, treatment minus control",
        "assignment": "independent equal allocation; iid Normal(100,1) arms shifted by 0.2",
        "guarantee": "production and matched reference use Welch-Satterthwaite t intervals",
        "reps": reps,
        "seed": seed,
        "available": usable,
        "unavailable": unavailable,
        "production_mean_width": float(np.mean(candidate_widths)) if candidate_widths else None,
        "welch_reference_mean_width": float(np.mean(reference_widths))
        if reference_widths
        else None,
        "relative_width_loss": float(np.mean(candidate_widths) / np.mean(reference_widths) - 1)
        if candidate_widths and reference_widths
        else None,
        "production_coverage": cand_cov,
        "welch_reference_coverage": ref_cov,
        "production_coverage_mcse": math.sqrt(cand_cov * (1 - cand_cov) / usable)
        if usable and cand_cov is not None
        else None,
        "welch_coverage_mcse": math.sqrt(ref_cov * (1 - ref_cov) / usable)
        if usable and ref_cov is not None
        else None,
        "production_power": cand_pow,
        "welch_reference_power": ref_pow,
        "production_power_mcse": math.sqrt(cand_pow * (1 - cand_pow) / usable)
        if usable and cand_pow is not None
        else None,
        "welch_power_mcse": math.sqrt(ref_pow * (1 - ref_pow) / usable)
        if usable and ref_pow is not None
        else None,
        "production_elapsed_s": candidate_elapsed,
        "reference_elapsed_s": reference_elapsed,
        "reference": COMPARATORS["mean"],
    }


def _benchmark_readout(
    reps: int, seed: int, *, metrics: int, arms: int, breakouts: int
) -> dict[str, object]:
    """Time all breakout slices for a metrics x arms x segments readout workload."""
    import pandas as pd

    from increment import readouts
    from increment.estimation.engine import Method
    from increment.frame import FrameTotalsSource, MetricSpec
    from increment.semantics.design import Randomized

    rng = np.random.default_rng(seed)
    groups = ["control", *(f"treatment_{i}" for i in range(arms - 1))]
    sources = []
    for breakout in range(breakouts):
        rows = []
        for group_index, group in enumerate(groups):
            values = rng.normal(100.0 + 2.0 * group_index, 1.0, (40, metrics))
            for unit, values_by_metric in enumerate(values):
                rows.append(
                    {
                        "unit_id": f"{group}-{unit}",
                        "variant": group,
                        **{f"m{i}": float(value) for i, value in enumerate(values_by_metric)},
                    }
                )
        sources.append(
            FrameTotalsSource.from_frame(
                pd.DataFrame(rows),
                unit="unit_id",
                group="variant",
                control="control",
                metrics=[MetricSpec(name=f"m{i}", type="mean") for i in range(metrics)],
                design=Randomized(
                    control_group="control", allocation=dict.fromkeys(groups, 1 / arms)
                ),
                experiment_id=f"comparative-efficiency-{breakout}",
            )
        )

    def run_once():
        return [
            readouts.run(source, decision_method=Method(name="unadjusted")) for source in sources
        ]

    started = time.perf_counter()
    result = run_once()
    cold = time.perf_counter() - started
    repeated = []
    for _ in range(reps):
        started = time.perf_counter()
        result = run_once()
        repeated.append(time.perf_counter() - started)
    return {
        "workload": {"metrics": metrics, "arms": arms, "breakouts": breakouts},
        "cold_elapsed_s": cold,
        "warm_elapsed_s_median": statistics.median(repeated),
        "warm_elapsed_s_samples": repeated,
        "readout_rows": sum(map(len, result)),
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "platform": platform.platform(),
    }


def campaign(*, reps: int = 200, seed: int = 20261008) -> dict[str, object]:
    """Run a production-vs-Welch mean study and representative cold/warm readout benchmark."""
    started = time.perf_counter()
    readout = _benchmark_readout(reps=5, seed=seed, metrics=3, arms=3, breakouts=4)
    return {
        "campaign": "B2 bounded comparative efficiency",
        "seed": seed,
        "comparison_tolerances_prespecified": {
            "dense_relative_width_loss_max": WIDTH_TOLERANCE,
            "power_loss_max": POWER_TOLERANCE,
            "sparse_discrete_width_gate": "not compared to asymptotic width; guarantee labels remain distinct",
        },
        "comparators": COMPARATORS,
        "mean_comparison": _mean_case(reps=reps, seed=seed),
        "readout_benchmark": readout,
        "campaign_elapsed_s": time.perf_counter() - started,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    output = json.dumps(campaign(reps=args.reps, seed=args.seed), indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(output + "\n")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
