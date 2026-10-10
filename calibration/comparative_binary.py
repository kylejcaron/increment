"""Bounded matched interval/power comparisons for binary outcomes.

References are computed only where estimands and assumptions match. Run
outside pytest with ``python -m calibration.comparative_binary``.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

from increment.errors import CodedError
from increment.estimation.conversion_route import dense_min_count, route_for_counts
from increment.estimation.engine import estimate_lift
from increment.estimation.results import LiftEstimate
from tests.estimation._conversion_counts import CONVERSION_METRIC, count_summary, lift_row


@dataclass(frozen=True, slots=True)
class Cell:
    name: str
    n_control: int
    n_treatment: int
    p_control: float
    p_treatment: float
    alpha: float
    dense_comparable: bool


CELLS = (
    Cell("sparse_boundary", 1000, 1000, 0.01, 0.012, 0.05, False),
    Cell("dense_boundary", 12000, 12000, 0.30, 0.36, 0.05, True),
    Cell("unequal_allocation", 12000, 24000, 0.30, 0.36, 0.05, True),
    Cell("allocated_tail_below_0005", 120000, 240000, 0.30, 0.36, 0.0002, True),
)


def _wilson(x: int, n: int, z: float) -> tuple[float, float]:
    p = x / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den
    return center - half, center + half


def katz_interval(xc: int, nc: int, xt: int, nt: int, alpha: float) -> tuple[float, float] | None:
    """Katz large-sample interval for RR-1, only with four nonzero counts."""
    if min(xc, nc - xc, xt, nt - xt) == 0:
        return None
    log_rr = math.log((xt / nt) / (xc / nc))
    se = math.sqrt(1 / xt - 1 / nt + 1 / xc - 1 / nc)
    z = float(norm.isf(alpha / 2))
    return math.expm1(log_rr - z * se), math.expm1(log_rr + z * se)


def newcombe_interval(xc: int, nc: int, xt: int, nt: int, alpha: float) -> tuple[float, float]:
    """Newcombe hybrid-score interval for the matched risk-difference estimand."""
    pc, pt = xc / nc, xt / nt
    z = float(norm.isf(alpha / 2))
    lc, uc = _wilson(xc, nc, z)
    lt, ut = _wilson(xt, nt, z)
    difference = pt - pc
    return (
        difference - math.hypot(pt - lt, uc - pc),
        difference + math.hypot(ut - pt, pc - lc),
    )


def _selector_boundaries() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for alpha in (0.05, 0.0002):
        tail = alpha / 2
        threshold = dense_min_count(tail)
        edges: list[dict[str, object]] = []
        for count in (threshold - 1, threshold, threshold + 1):
            n = 2 * count
            route = route_for_counts(count, n, count, n, tail_alpha=tail, mode="auto")
            edges.append({"minimum_arm_count": count, "route": route})
        result.append(
            {"alpha": alpha, "tail_alpha": tail, "dense_min_count": threshold, "edges": edges}
        )
    return result


def _production_outcome(row: LiftEstimate) -> tuple[float | None, bool]:
    """Return finite interval width, if available, and the row's test decision."""
    lift = row.lift
    width = (
        lift.ub - lift.lb
        if lift is not None and lift.lb is not None and lift.ub is not None
        else None
    )
    return width, row.stat_sig()


def run(*, reps: int = 200, seed: int = 20261008) -> dict[str, object]:
    """Compare the production auto route with matched interval references."""
    seed_sequences = np.random.SeedSequence(seed).spawn(len(CELLS))
    out: list[dict[str, object]] = []
    for cell, seed_sequence in zip(CELLS, seed_sequences, strict=True):
        rng = np.random.default_rng(seed_sequence)
        widths_auto: list[float] = []
        widths_katz: list[float] = []
        widths_newcombe: list[float] = []
        power_auto = power_katz = 0
        available = 0
        unbounded_intervals = 0
        refusals: Counter[str] = Counter()
        elapsed_auto = elapsed_katz = 0.0
        abs_width_excess = 0.0
        routes: Counter[str] = Counter()
        for _ in range(reps):
            xc = int(rng.binomial(cell.n_control, cell.p_control))
            xt = int(rng.binomial(cell.n_treatment, cell.p_treatment))
            started = time.perf_counter()
            try:
                row = lift_row((xc, cell.n_control, xt, cell.n_treatment), alpha=cell.alpha)
            except CodedError as exc:
                refusals[exc.code] += 1
                continue
            elapsed_auto += time.perf_counter() - started
            available += 1
            routes[row.reference_kind] += 1
            interval_width, rejected = _production_outcome(row)
            power_auto += int(rejected)
            if interval_width is None:
                unbounded_intervals += 1
            else:
                widths_auto.append(interval_width)
            started = time.perf_counter()
            katz = katz_interval(xc, cell.n_control, xt, cell.n_treatment, cell.alpha)
            if katz is not None:
                widths_katz.append(katz[1] - katz[0])
                power_katz += katz[0] > 0 or katz[1] < 0
            elapsed_katz += time.perf_counter() - started
            difference = newcombe_interval(xc, cell.n_control, xt, cell.n_treatment, cell.alpha)
            widths_newcombe.append(difference[1] - difference[0])
            absolute = estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(xc, cell.n_control, xt, cell.n_treatment),
                control_group="control",
                alpha=cell.alpha,
                null_abs=0.0,
            )
            if (
                absolute.results
                and absolute.results[0].abs_lb is not None
                and absolute.results[0].abs_ub is not None
            ):
                abs_width_excess += max(
                    0.0,
                    absolute.results[0].abs_ub
                    - absolute.results[0].abs_lb
                    - difference[1]
                    + difference[0],
                )
        power_rate = power_auto / available if available else None
        out.append(
            {
                **asdict(cell),
                "seed_sequence": {
                    "entropy": seed_sequence.entropy,
                    "spawn_key": list(seed_sequence.spawn_key),
                },
                "reps": reps,
                "production_auto": {
                    "guarantee": "finite-sample on binomial route; asymptotic on dense route",
                    "routes": dict(routes),
                    "available": available,
                    "unbounded_interval_count": unbounded_intervals,
                    "finite_interval_count": len(widths_auto),
                    "rejection_power": power_rate,
                    "power_mcse": math.sqrt(power_rate * (1 - power_rate) / available)
                    if available and power_rate is not None
                    else None,
                    "elapsed_s": elapsed_auto,
                },
                "katz_asymptotic_rr": {
                    "availability": len(widths_katz),
                    "mean_rr_interval_width": float(np.mean(widths_katz)) if widths_katz else None,
                    "rejection_power": power_katz / len(widths_katz) if widths_katz else None,
                    "elapsed_s": elapsed_katz,
                },
                "newcombe_matched_risk_difference": {
                    "availability": len(widths_newcombe),
                    "mean_difference_interval_width": float(np.mean(widths_newcombe)),
                    "production_absolute_mean_excess_width": abs_width_excess / max(available, 1),
                },
                "efficiency_width_loss_vs_katz": (
                    float(np.mean(widths_auto) / np.mean(widths_katz) - 1)
                    if cell.dense_comparable and widths_auto and widths_katz
                    else None
                ),
            }
        )
    return {
        "campaign": "bounded production conversion versus matched references",
        "tolerances_prespecified": {"dense_width_loss_max": 0.10, "power_loss_max": 0.05},
        "seed": seed,
        "selector_boundaries": _selector_boundaries(),
        "cells": out,
        "not_executed": ["numerical tolerance loss as a separate arithmetic perturbation"],
    }


def main() -> int:
    print(json.dumps(run(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
