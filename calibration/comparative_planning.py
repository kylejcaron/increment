"""Replay planned conversion inference through public automatic and exact routes."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np


def _frame(n: int, seed: int):
    import polars as pl

    p_control, lift = 0.30, 0.20
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.array(["control", "treatment"]), n)
    outcomes = np.concatenate(
        (rng.binomial(1, p_control, n), rng.binomial(1, p_control * (1.0 + lift), n))
    )
    return pl.DataFrame(
        {
            "unit_id": [f"u{i}" for i in range(2 * n)],
            "variant": groups,
            "converted": outcomes,
        }
    )


def _analysis_run(frame, route: str):
    from increment import Analysis, Method, MetricSpec
    from tests.analysis_factory import lift_rows

    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )
    try:
        if route == "auto":
            return lift_rows(analysis.run())
        return lift_rows(
            analysis.run(
                decision_method=Method(name="unadjusted", conversion_inference="finite_sample")
            )
        )
    finally:
        analysis.close()


def _route_timing_worker(route: str, *, warm: bool, n: int, seed: int) -> float:
    """Measure one route in an isolated worker process; warm mode primes that route first."""
    from increment import Analysis, Method, MetricSpec

    frame = _frame(n, seed)
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="converted", type="conversion")],
    )

    def run_route():
        if route == "auto":
            return analysis.run()
        return analysis.run(
            decision_method=Method(name="unadjusted", conversion_inference="finite_sample")
        )

    try:
        if warm:
            run_route()
        started = time.perf_counter()
        run_route()
        return time.perf_counter() - started
    finally:
        analysis.close()


def _measure_in_fresh_process(route: str, *, warm: bool, n: int, seed: int) -> float:
    command = [
        sys.executable,
        "-m",
        "calibration.comparative_planning",
        "--timing-worker",
        route,
        "--n",
        str(n),
        "--seed",
        str(seed),
    ]
    if warm:
        command.append("--warm")
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return float(json.loads(completed.stdout)["run_elapsed_s"])


def _matched_route_timings(n: int, seed: int) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for route in ("auto", "exact"):
        result[route] = {
            "cold_fresh_process_run_s": _measure_in_fresh_process(
                route, warm=False, n=n, seed=seed
            ),
            "warm_same_process_after_route_warmup_run_s": _measure_in_fresh_process(
                route, warm=True, n=n, seed=seed
            ),
        }
    return result


def _summary(reps: int, seed: int) -> tuple[dict[str, object], dict[str, object]]:
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.power import Baseline, PowerDesign, required_sample_size

    alpha, p_control, lift = 0.05, 0.30, 0.20
    target_power = 0.80
    baseline = Baseline.from_proportion(p_control)
    procedure = ArmPlanningProcedure.standard(
        "conversion", alpha=alpha, conversion_inference="auto"
    )
    design = PowerDesign(power=target_power, allocation=0.5)
    plan = required_sample_size(lift, baseline, procedure, design)
    n = plan.n_per_arm
    rejected = unavailable = exact_available = 0
    routes: dict[str, int] = {}
    for rep in range(reps):
        frame = _frame(n, seed + rep)
        rows = _analysis_run(frame, "auto")
        exact_rows = _analysis_run(frame, "exact")
        if exact_rows and exact_rows[0].lift is not None:
            exact_available += 1
        if not rows or rows[0].lift is None:
            unavailable += 1
            continue
        row = rows[0]
        routes[row.reference_kind] = routes.get(row.reference_kind, 0) + 1
        rejected += row.stat_sig()
    usable = reps - unavailable
    rate = rejected / usable if usable else None
    katz_n = _katz_required_n(p_control, lift, alpha, target_power)
    timings = _matched_route_timings(n, seed)
    shared: dict[str, object] = {
        "estimand": "risk ratio minus one",
        "assignment": "independent randomized two-arm fixed horizon; treatment allocation 0.5",
        "baseline_rate": p_control,
        "true_lift": lift,
        "alpha": alpha,
        "design_power_target": target_power,
        "planned_n_per_arm": n,
        "planner_power": plan.power,
        "planner_power_basis": plan.power_basis,
        "Katz_asymptotic_required_n_per_arm": katz_n,
        "reps": reps,
        "seed": seed,
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
    }
    return (
        {
            **shared,
            "runtime_procedure": "Analysis.from_unit_summary(...).run() automatic conversion inference",
            "runtime_rejection_rate": rate,
            "runtime_power_mcse": math.sqrt(rate * (1 - rate) / usable)
            if rate is not None
            else None,
            "runtime_routes": routes,
            "available": usable,
            "unavailable": unavailable,
            "runtime_timings": {
                "cache_state": "separate fresh subprocess per route and state",
                **timings["auto"],
            },
            "required_n_inflation_vs_Katz": n / katz_n - 1.0,
        },
        {
            **shared,
            "runtime_procedure": "Analysis.run(decision_method=Method(conversion_inference='finite_sample'))",
            "available": exact_available,
            "unavailable": reps - exact_available,
            "runtime_timings": {
                "cache_state": "separate fresh subprocess per route and state",
                **timings["exact"],
            },
            "runtime_route": "finite-sample binomial explicitly requested",
        },
    )


def _katz_required_n(p_control: float, lift: float, alpha: float, power: float) -> int:
    from scipy.stats import norm

    p_treatment = p_control * (1.0 + lift)
    log_rr = math.log1p(lift)
    critical = float(norm.isf(alpha / 2))
    z_power = float(norm.isf(1.0 - power))
    variance_numerator = (1 - p_control) / p_control + (1 - p_treatment) / p_treatment
    return math.ceil(variance_numerator * (critical + z_power) ** 2 / log_rr**2)


def run(*, reps: int = 30, seed: int = 20261008) -> dict[str, object]:
    automatic, explicit_exact = _summary(reps, seed)
    return {
        "campaign": "runtime-matched conversion planning",
        "automatic": automatic,
        "explicit_exact": explicit_exact,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-worker", choices=("auto", "exact"))
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--n", type=int, default=1023)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--reps", type=int, default=30)
    args = parser.parse_args(argv)
    if args.timing_worker:
        elapsed = _route_timing_worker(args.timing_worker, warm=args.warm, n=args.n, seed=args.seed)
        print(json.dumps({"run_elapsed_s": elapsed}))
        return 0
    print(json.dumps(run(reps=args.reps, seed=args.seed), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
