"""Small-arm coverage and width of the default conversion and mean relative intervals.

Both measurements read production rows from ``estimate_lift`` at two-sided ``ALPHA``:

``conversion``
    The default (``conversion_inference="auto"``) conversion row against the Katz
    (log-scale Wald) risk-ratio interval, at arm sizes whose counts sit below
    ``dense_min_count(ALPHA / 2)``, where the row takes the finite-sample route. Coverage and
    expected log width are integrated over the joint binomial law of the counts, not drawn, so
    they carry no sampling error; each arm's law omits at most ``2 * OMITTED_MASS``, and the
    omitted mass is reported. The row's set reaches a risk ratio of zero (a -100% lift) at
    small arms even when treatment has events; the probability that it does is reported, and
    the log-width ratio (the row's expected log width over Katz's) is taken over the other
    count pairs. The command also reports the dense-count floor and the arm size at which a
    rate's expected events reach it.

``mean``
    A mean row's relative interval (log-ratio delta method) and the same row's additive Welch
    interval, on gamma outcomes with a true ratio of ``MEAN_RATIO``, by Monte Carlo with its
    standard error reported. Coverage is over the draws whose row has a relative interval; the
    others (the log-scale admission guard refuses them) are counted and reported.

Run outside pytest::

    uv run python -m calibration.small_arm_intervals conversion --workers 8
    uv run python -m calibration.small_arm_intervals mean
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from increment.estimation.conversion_route import dense_min_count
from increment.estimation.engine import estimate_lift
from increment.semantics.models import MeanMetric
from tests.estimation._conversion_counts import lift_row

ALPHA = 0.05
#: ``(units per arm, control rate, true risk ratio)`` of each conversion lattice.
CONVERSION_CELLS = ((200, 0.10, 1.0), (1000, 0.10, 1.08))
#: Rates at which the arm size reaching the dense-count floor is reported.
FLOOR_RATES = ("0.10", "0.20")
#: Probability omitted from each tail of an arm's count law.
OMITTED_MASS = 1e-9
#: ``(gamma shape, units per arm)`` of each mean cell. Outcomes have gamma scale 10 in control
#: and ``10 * MEAN_RATIO`` in treatment.
MEAN_CELLS = ((0.5, 30), (0.5, 100))
MEAN_RATIO = 1.1
MEAN_REPS = 20_000
MEAN_SEED = 77

_MEAN_METRIC = MeanMetric(name="y", entity="user", fact="y")


@dataclass(frozen=True, slots=True)
class LatticeCell:
    """Exact coverage and expected log width of the default row and of Katz on one lattice."""

    n: int
    rate: float
    ratio: float
    coverage: float
    katz_coverage: float
    log_width_ratio: float
    finite_sample_mass: float
    zero_lower_mass: float
    omitted_mass: float


@dataclass(frozen=True, slots=True)
class MeanCell:
    """Monte Carlo coverage of a mean row's relative and additive intervals, over the draws
    whose row has a relative interval."""

    shape: float
    n: int
    reps: int
    relative_coverage: float
    additive_coverage: float
    without_relative: int

    def standard_error(self, coverage: float) -> float:
        return math.sqrt(coverage * (1.0 - coverage) / (self.reps - self.without_relative))


def _window(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    lo = int(binom.ppf(OMITTED_MASS, n, p))
    hi = int(binom.isf(OMITTED_MASS, n, p))
    counts = np.arange(lo, hi + 1)
    return counts, binom.pmf(counts, n, p)


def _katz_log_bounds(x_c: int, n_c: int, x_t: int, n_t: int) -> tuple[float, float]:
    log_rr = math.log(x_t / n_t) - math.log(x_c / n_c)
    radius = float(norm.isf(ALPHA / 2)) * math.sqrt(1 / x_t - 1 / n_t + 1 / x_c - 1 / n_c)
    return log_rr - radius, log_rr + radius


def _log_bounds(lower: float, upper: float | None) -> tuple[float, float]:
    """Lift-scale endpoints on the log risk-ratio scale; a lower endpoint of -1 is ``-inf``."""
    assert upper is not None, "a zero control count lies outside every measured window"
    return (-math.inf if lower <= -1.0 else math.log1p(lower)), math.log1p(upper)


def _lattice_row(
    args: tuple[int, int, tuple[int, ...], tuple[float, ...], float],
) -> tuple[float, float, float, float, float, float]:
    """One control count's sums over the treatment window, each weighted by the treatment
    probability: row coverage, Katz coverage, row and Katz log widths where the row's lower
    endpoint is above zero, finite-sample mass and the mass of a zero lower endpoint."""
    x_c, n, treatment_counts, weights, ratio = args
    log_ratio = math.log(ratio)
    covered = katz_covered = width = katz_width = finite_sample = zero_lower = 0.0
    for x_t, weight in zip(treatment_counts, weights, strict=True):
        row = lift_row((x_c, n, x_t, n), alpha=ALPHA)
        if row.binomial_set is None:
            assert row.lift is not None and row.lift.lb is not None
            lower, upper = _log_bounds(row.lift.lb, row.lift.ub)
        else:
            lower, upper = _log_bounds(row.binomial_set.lower, row.binomial_set.upper)
            finite_sample += weight
        katz_lower, katz_upper = _katz_log_bounds(x_c, n, x_t, n)
        covered += weight * (lower <= log_ratio <= upper)
        katz_covered += weight * (katz_lower <= log_ratio <= katz_upper)
        if lower == -math.inf:
            zero_lower += weight
        else:
            width += weight * (upper - lower)
            katz_width += weight * (katz_upper - katz_lower)
    return covered, katz_covered, width, katz_width, finite_sample, zero_lower


def lattice_cell(n: int, rate: float, ratio: float, *, workers: int) -> LatticeCell:
    """Integrate the default row and Katz over both arms' count windows."""
    control_counts, control_weights = _window(n, rate)
    treatment_counts, treatment_weights = _window(n, rate * ratio)
    jobs = [
        (int(x_c), n, tuple(int(x) for x in treatment_counts), tuple(treatment_weights), ratio)
        for x_c in control_counts
    ]
    with multiprocessing.Pool(workers) as pool:
        rows = pool.map(_lattice_row, jobs)
    sums = np.asarray(rows).T @ control_weights
    covered, katz_covered, width, katz_width, finite_sample, zero_lower = (float(s) for s in sums)
    return LatticeCell(
        n=n,
        rate=rate,
        ratio=ratio,
        coverage=covered,
        katz_coverage=katz_covered,
        log_width_ratio=width / katz_width,
        finite_sample_mass=finite_sample,
        zero_lower_mass=zero_lower,
        omitted_mass=1.0 - float(control_weights.sum()) * float(treatment_weights.sum()),
    )


def _mean_summary(group_id: str, y: np.ndarray) -> dict[str, object]:
    ref = float(y.mean())
    residual = y - ref
    return {
        "experiment_id": "e",
        "metric": "y",
        "group_id": group_id,
        "n": len(y),
        "ref_y": ref,
        "cy1": float(residual.sum()),
        "cy2": float((residual * residual).sum()),
    }


def mean_cell(shape: float, n: int, *, reps: int, seed: int) -> MeanCell:
    """Coverage of the true lift and the true difference over ``reps`` seeded draws."""
    rng = np.random.default_rng(seed)
    control = rng.gamma(shape, 10.0, (reps, n))
    treatment = rng.gamma(shape, 10.0 * MEAN_RATIO, (reps, n))
    true_lift = MEAN_RATIO - 1.0
    true_difference = shape * 10.0 * (MEAN_RATIO - 1.0)
    relative = additive = without_relative = 0
    for y_c, y_t in zip(control, treatment, strict=True):
        computation = estimate_lift(
            metrics=[_MEAN_METRIC],
            summary=pd.DataFrame([_mean_summary("control", y_c), _mean_summary("treatment", y_t)]),
            control_group="control",
            alpha=ALPHA,
        )
        row = computation.results[0] if computation.results else None
        if row is None or row.lift is None:
            without_relative += 1
            continue
        assert row.lift.lb is not None and row.lift.ub is not None
        assert row.abs_lb is not None and row.abs_ub is not None
        relative += row.lift.lb <= true_lift <= row.lift.ub
        additive += row.abs_lb <= true_difference <= row.abs_ub
    evaluated = reps - without_relative
    return MeanCell(shape, n, reps, relative / evaluated, additive / evaluated, without_relative)


def _conversion(workers: int) -> None:
    floor = dense_min_count(ALPHA / 2)
    print(f"dense_min_count({ALPHA / 2}) = {floor} events and non-events per arm")
    for rate in FLOOR_RATES:
        size = math.ceil(Fraction(floor) / Fraction(rate))
        print(f"  a {rate} rate expects {floor} events at {size:,} units per arm")
    for n, rate, ratio in CONVERSION_CELLS:
        cell = lattice_cell(n, rate, ratio, workers=workers)
        print(
            f"n={n}/arm rate={rate} ratio={ratio}: coverage {cell.coverage:.4f} "
            f"(Katz {cell.katz_coverage:.4f}), log width {cell.log_width_ratio:.4f}x Katz, "
            f"finite-sample mass {cell.finite_sample_mass:.6f}, "
            f"zero-lower mass {cell.zero_lower_mass:.4g}, omitted mass {cell.omitted_mass:.1e}",
            flush=True,
        )


def _mean(reps: int, seed: int) -> None:
    for shape, n in MEAN_CELLS:
        cell = mean_cell(shape, n, reps=reps, seed=seed)
        print(
            f"gamma({shape}) n={n}/arm ratio={MEAN_RATIO}: relative coverage "
            f"{cell.relative_coverage:.4f}, additive Welch coverage {cell.additive_coverage:.4f} "
            f"(MC SE {cell.standard_error(cell.relative_coverage):.4f} / "
            f"{cell.standard_error(cell.additive_coverage):.4f}; reps {reps}, seed {seed}, "
            f"without a relative interval {cell.without_relative})",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    conversion = commands.add_parser("conversion", help="exact-lattice conversion cells")
    conversion.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    mean = commands.add_parser("mean", help="Monte Carlo mean cells")
    mean.add_argument("--reps", type=int, default=MEAN_REPS)
    mean.add_argument("--seed", type=int, default=MEAN_SEED)
    args = parser.parse_args(argv)
    if args.command == "conversion":
        _conversion(args.workers)
    else:
        _mean(args.reps, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
