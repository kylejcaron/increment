"""Calibrate the ratio denominator-precision advisory threshold.

The delta-method ratio interval undercovers when the denominator mean is
poorly resolved: a heavy right tail at small n makes ``1 / d_bar`` skewed
in a way the first-order variance does not see. The carried moments do
reveal how precisely the denominator mean is resolved, as its relative
standard error ``sqrt(var_d / n) / d_bar``. This script measures, on a
grid of sample sizes and denominator laws, the production interval's
coverage under a null lift together with that statistic, then chooses the
advisory threshold by a fixed rule:

* a cell is *flagged* at threshold ``T`` when the statistic exceeds ``T``
  in at least half of its replications;
* the chosen ``T`` is the smallest value on a 0.01 grid such that every
  cell with coverage below 0.93 is flagged and no cell with coverage at or
  above 0.945 is flagged;
* if no single ``T`` separates the two sets, the script reports the
  largest ``T`` that still flags every cell below 0.93 and lists the
  well-covered cells it flags as false positives.

The numerator is drawn independently of the denominator (``Normal(2, 1)``),
the case in which the denominator's skew propagates undamped into the
ratio; a numerator proportional to the denominator largely cancels it and
covers nominally even at heavy skew, so it would understate the hazard.
Coverage counts only draws the engine admits; draws refused by the
log-scale admission guard are reported separately.

Run outside pytest, with a hard deadline::

    uv run python scripts/probe_ratio_denominator_precision.py
    uv run python scripts/probe_ratio_denominator_precision.py --pilot
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from increment.estimation.armstats import ArmStats
from increment.estimation.engine import estimate_lift
from increment.semantics.models import Measure, RatioMetric

SAMPLE_SIZES = (20, 50, 100, 200, 500, 2000)
DENOMINATORS = ("lognormal:0.5", "lognormal:1.0", "lognormal:1.5", "exponential", "gamma:0.5")
UNDERCOVERED = 0.93
WELL_COVERED = 0.945
THRESHOLD_GRID = [round(0.01 * k, 2) for k in range(1, 101)]

_METRIC = RatioMetric(
    name="ratio", entity="user", numerator=Measure(fact="num"), denominator=Measure(fact="den")
)


@dataclass(frozen=True)
class Cell:
    n: int
    kind: str
    coverage: float
    evaluated: int
    refused: int
    stats: np.ndarray

    def flagged(self, threshold: float) -> bool:
        return float(np.mean(self.stats > threshold)) >= 0.5


def draw_denominator(rng: np.random.Generator, kind: str, n: int) -> np.ndarray:
    family, _, parameter = kind.partition(":")
    if family == "lognormal":
        return rng.lognormal(mean=0.0, sigma=float(parameter), size=n)
    if family == "exponential":
        return rng.exponential(1.0, size=n)
    if family == "gamma":
        return rng.gamma(float(parameter), 1.0, size=n)
    raise SystemExit(f"unknown denominator law {kind!r}")


def arm(group_id: str, y: np.ndarray, d: np.ndarray) -> ArmStats:
    return ArmStats.from_raw_sums(
        study_id="probe",
        metric="ratio",
        group_id=group_id,
        n=len(y),
        sum_y=float(y.sum()),
        sum_y2=float((y * y).sum()),
        sum_den=float(d.sum()),
        sum_den2=float((d * d).sum()),
        sum_yden=float((y * d).sum()),
    )


def summary_row(a: ArmStats) -> dict[str, object]:
    return {
        "experiment_id": a.study_id,
        "metric": a.metric,
        "group_id": a.group_id,
        "n": float(a.n),
        "ref_y": a.ref_y,
        "cy1": a.cy1,
        "cy2": a.cy2,
        "ref_den": a.ref_den,
        "cden1": a.cden1,
        "cden2": a.cden2,
        "cyden": a.cyden,
    }


def denominator_precision(a: ArmStats) -> float:
    return math.sqrt(a.var_den() / a.n) / a.mean_den()


def run_cell(n: int, kind: str, reps: int, seed: int) -> Cell:
    rng = np.random.default_rng(seed)
    covered = evaluated = refused = 0
    stats = np.empty(reps)
    for rep in range(reps):
        arms = []
        for group_id in ("control", "treatment"):
            d = draw_denominator(rng, kind, n)
            y = rng.normal(2.0, 1.0, size=n)
            arms.append(arm(group_id, y, d))
        control, treatment = arms
        stats[rep] = max(denominator_precision(control), denominator_precision(treatment))
        computation = estimate_lift(
            metrics=[_METRIC],
            summary=pd.DataFrame([summary_row(control), summary_row(treatment)]),
            control_group="control",
        )
        if not computation.results:
            refused += 1
            continue
        lift = computation.results[0].require_lift()
        assert lift.lb is not None and lift.ub is not None
        evaluated += 1
        covered += lift.lb <= 0.0 <= lift.ub
    coverage = covered / evaluated if evaluated else math.nan
    return Cell(n, kind, coverage, evaluated, refused, stats)


def choose_threshold(
    cells: list[Cell],
) -> tuple[float | None, list[Cell], list[Cell]]:
    """Return (threshold, false_positive_cells, unflaggable_cells).

    `threshold` is `None` when no grid value flags every undercovered cell --
    i.e. some undercovered cell's precision statistic is never elevated, which
    is a genuine finding this probe exists to surface, not a bug. In that case
    `unflaggable_cells` names the offending cells and `false_positive_cells`
    is empty.
    """
    under = [c for c in cells if c.coverage < UNDERCOVERED]
    well = [c for c in cells if c.coverage >= WELL_COVERED]
    for threshold in THRESHOLD_GRID:
        if all(c.flagged(threshold) for c in under) and not any(c.flagged(threshold) for c in well):
            return threshold, [], []
    candidates = [t for t in THRESHOLD_GRID if all(c.flagged(t) for c in under)]
    if not candidates:
        unflaggable = [c for c in under if not any(c.flagged(t) for t in THRESHOLD_GRID)]
        return None, [], unflaggable
    fallback = max(candidates)
    return fallback, [c for c in well if c.flagged(fallback)], []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--deadline-seconds", type=float, default=900.0)
    parser.add_argument(
        "--pilot", action="store_true", help="one cell (n=50, lognormal sigma=1.5, 200 reps)"
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    if args.pilot:
        grid = [(50, "lognormal:1.5")]
        reps = 200
    else:
        grid = [(n, kind) for kind in DENOMINATORS for n in SAMPLE_SIZES]
        reps = args.reps
    cells: list[Cell] = []
    for index, (n, kind) in enumerate(grid):
        if time.monotonic() >= deadline:
            print(f"deadline reached after {len(cells)} of {len(grid)} cells", file=sys.stderr)
            return 124
        cell = run_cell(n, kind, reps, seed=args.seed + index)
        cells.append(cell)
        q05, q50, q95 = np.quantile(cell.stats, [0.05, 0.5, 0.95])
        print(
            f"{kind:14s} n={n:5d} coverage={cell.coverage:.4f} evaluated={cell.evaluated} "
            f"refused={cell.refused} stat q05/q50/q95={q05:.4f}/{q50:.4f}/{q95:.4f} "
            f"min={cell.stats.min():.4f} max={cell.stats.max():.4f} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    if args.pilot:
        return 0
    threshold, false_positives, unflaggable = choose_threshold(cells)
    if threshold is None:
        print("no threshold flags every undercovered cell; precision statistic never elevated for:")
        for cell in unflaggable:
            print(f"  {cell.kind} n={cell.n} coverage={cell.coverage:.4f}")
        return 1
    print(f"threshold={threshold:.2f}")
    if false_positives:
        print("no single threshold separates the sets; false-positive cells at this threshold:")
        for cell in false_positives:
            print(f"  {cell.kind} n={cell.n} coverage={cell.coverage:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
