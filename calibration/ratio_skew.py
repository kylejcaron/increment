"""Calibrate the ratio denominator-skewness coverage warning.

The delta-method ratio interval undercovers when the denominator mean's
sampling law is skewed: a heavy right tail at small n pushes ``1 / d_bar``
away from the first-order line. The third centered moment the moments
wire now carries exposes that skew directly. The natural statistic is the
standardized skewness of the denominator MEAN, ``g1 / sqrt(n)`` with
``g1`` the arm's sample skewness ``m3 / m2**1.5``: the leading Edgeworth
term of a mean's departure from normality. This script measures, on a
grid of sample sizes and lognormal denominators, the production
interval's coverage under a null lift together with the larger of the two
arms' statistics, then chooses the warning cutoff by a fixed rule on the
pooled curve of coverage against the observed statistic:

* admitted replications of every cell are pooled and binned by the
  statistic in steps of 0.05;
* the chosen cutoff is the lower edge of the first bin at which that bin
  AND the next both cover below 0.935, so one noisy bin cannot set it.

Pooling across cells is deliberate: a per-cell rule overfits the Monte
Carlo noise of cells whose coverage sits near 0.935 (two adjacent cells
of one denominator law can straddle it), while the pooled curve is the
claim an analyst actually reads off the warning -- the coverage of rows
whose observed statistic is at least the cutoff. For the chosen cutoff
the script reports each cell's flag rate and the pooled coverage of the
rows the warning would and would not have flagged. The samples that miss
a heavy tail look benign and undercover most, so the flagged/unflagged
contrast is bounded; the cutoff marks where observed skew starts to
predict undercoverage, not a line that separates safe rows from unsafe.

The numerator is drawn independently of the denominator (``Normal(2, 1)``),
the case in which the denominator's skew propagates undamped into the
ratio. Coverage counts only draws the engine admits; draws refused by the
log-scale admission guard are reported separately.

Run outside pytest, with a hard deadline::

    uv run python -m calibration.ratio_skew
    uv run python -m calibration.ratio_skew --pilot
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

SAMPLE_SIZES = (50, 100, 200, 400)
SIGMAS = (0.5, 1.0, 1.5, 2.0)
UNDERCOVERED = 0.935
BIN_WIDTH = 0.05

_METRIC = RatioMetric(
    name="ratio", entity="user", numerator=Measure(fact="num"), denominator=Measure(fact="den")
)


@dataclass(frozen=True)
class Cell:
    n: int
    sigma: float
    coverage: float
    evaluated: int
    refused: int
    stats: np.ndarray  # admitted replications only
    covered: np.ndarray  # admitted replications only

    def flag_rate(self, cutoff: float) -> float:
        return float(np.mean(self.stats > cutoff))


def arm(group_id: str, y: np.ndarray, d: np.ndarray) -> ArmStats:
    """Centered moments of one arm, third denominator moment included."""
    ref_y = float(y.mean())
    ref_d = float(d.mean())
    ry = y - ref_y
    rd = d - ref_d
    return ArmStats(
        study_id="probe",
        metric="ratio",
        group_id=group_id,
        n=len(y),
        ref_y=ref_y,
        cy1=float(ry.sum()),
        cy2=float((ry * ry).sum()),
        ref_den=ref_d,
        cden1=float(rd.sum()),
        cden2=float((rd * rd).sum()),
        cden3=float((rd * rd * rd).sum()),
        cyden=float((ry * rd).sum()),
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
        "cden3": a.cden3,
        "cyden": a.cyden,
    }


def mean_skewness(a: ArmStats) -> float:
    """Standardized skewness of the arm's denominator mean, ``g1 / sqrt(n)``."""
    return a.skew_den() / math.sqrt(a.n)


def run_cell(n: int, sigma: float, reps: int, seed: int, deadline: float) -> Cell | None:
    rng = np.random.default_rng(seed)
    refused = 0
    stats: list[float] = []
    covered: list[bool] = []
    for _ in range(reps):
        if time.monotonic() >= deadline:
            return None
        arms = []
        for group_id in ("control", "treatment"):
            d = rng.lognormal(mean=0.0, sigma=sigma, size=n)
            y = rng.normal(2.0, 1.0, size=n)
            arms.append(arm(group_id, y, d))
        control, treatment = arms
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
        stats.append(max(mean_skewness(control), mean_skewness(treatment)))
        covered.append(lift.lb <= 0.0 <= lift.ub)
    evaluated = len(covered)
    coverage = float(np.mean(covered)) if evaluated else math.nan
    return Cell(n, sigma, coverage, evaluated, refused, np.asarray(stats), np.asarray(covered))


def choose_cutoff(cells: list[Cell]) -> float | None:
    """Lower edge of the first statistic bin at which it and the next bin both
    cover below :data:`UNDERCOVERED`; ``None`` when no two consecutive bins do."""
    curve = binned_coverage(cells, BIN_WIDTH)
    for (lo, coverage, _), (_, next_coverage, _) in zip(curve, curve[1:], strict=False):
        if coverage < UNDERCOVERED and next_coverage < UNDERCOVERED:
            return lo
    return None


def binned_coverage(cells: list[Cell], width: float = BIN_WIDTH) -> list[tuple[float, float, int]]:
    """Pooled coverage of the admitted rows per statistic bin ``[lo, lo + width)``."""
    stats = np.concatenate([c.stats for c in cells])
    covered = np.concatenate([c.covered for c in cells])
    edges = np.arange(0.0, float(stats.max()) + width, width)
    out: list[tuple[float, float, int]] = []
    for lo in edges:
        inside = (stats >= lo) & (stats < lo + width)
        if inside.any():
            out.append((float(lo), float(covered[inside].mean()), int(inside.sum())))
    return out


def conditional_coverage(cells: list[Cell], cutoff: float) -> tuple[float, int, float, int]:
    """Pooled (coverage, count) of the admitted rows flagged and not flagged at *cutoff*."""
    stats = np.concatenate([c.stats for c in cells])
    covered = np.concatenate([c.covered for c in cells])
    flagged = stats > cutoff
    flagged_cov = float(covered[flagged].mean()) if flagged.any() else math.nan
    clear_cov = float(covered[~flagged].mean()) if (~flagged).any() else math.nan
    return flagged_cov, int(flagged.sum()), clear_cov, int((~flagged).sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--deadline-seconds", type=float, default=900.0)
    parser.add_argument(
        "--pilot", action="store_true", help="one cell (n=50, lognormal sigma=1.5, 200 reps)"
    )
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + args.deadline_seconds
    if args.pilot:
        grid = [(50, 1.5)]
        reps = 200
    else:
        grid = [(n, sigma) for sigma in SIGMAS for n in SAMPLE_SIZES]
        reps = args.reps
    cells: list[Cell] = []
    for index, (n, sigma) in enumerate(grid):
        if time.monotonic() >= deadline:
            print(f"deadline reached after {len(cells)} of {len(grid)} cells", file=sys.stderr)
            return 124
        cell = run_cell(n, sigma, reps, seed=args.seed + index, deadline=deadline)
        if cell is None:
            print(
                f"deadline reached after {len(cells)} of {len(grid)} completed cells",
                file=sys.stderr,
            )
            return 124
        cells.append(cell)
        q05, q50, q95 = np.quantile(cell.stats, [0.05, 0.5, 0.95])
        print(
            f"lognormal:{sigma:<4} n={n:5d} coverage={cell.coverage:.4f} "
            f"evaluated={cell.evaluated} refused={cell.refused} "
            f"stat q05/q50/q95={q05:.4f}/{q50:.4f}/{q95:.4f} "
            f"elapsed={time.monotonic() - started:.1f}s",
            flush=True,
        )
    if args.pilot:
        return 0
    print("pooled coverage by statistic bin (admitted rows):")
    for lo, coverage, count in binned_coverage(cells):
        print(f"  [{lo:.2f}, {lo + BIN_WIDTH:.2f}) coverage={coverage:.4f} rows={count}")
    print("cutoff sweep (pooled coverage of flagged / not flagged admitted rows):")
    for sweep in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        flagged_cov, flagged_n, clear_cov, clear_n = conditional_coverage(cells, sweep)
        rates = " ".join(f"{cell.flag_rate(sweep):.2f}" for cell in cells)
        print(
            f"  cutoff={sweep:.2f} flagged={flagged_cov:.4f} ({flagged_n}) "
            f"clear={clear_cov:.4f} ({clear_n}) cell flag rates: {rates}"
        )
    cutoff = choose_cutoff(cells)
    if cutoff is None:
        print("no two consecutive statistic bins cover below the undercoverage line")
        return 1
    print(f"cutoff={cutoff:.2f}")
    for cell in cells:
        print(
            f"  lognormal:{cell.sigma:<4} n={cell.n:5d} coverage={cell.coverage:.4f} "
            f"flag_rate={cell.flag_rate(cutoff):.3f}"
        )
    flagged_cov, flagged_n, clear_cov, clear_n = conditional_coverage(cells, cutoff)
    print(
        f"pooled coverage: flagged={flagged_cov:.4f} ({flagged_n} rows) "
        f"not flagged={clear_cov:.4f} ({clear_n} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
