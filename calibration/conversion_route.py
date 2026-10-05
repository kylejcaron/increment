"""Boundary coverage of the delta-method conversion route and its dense-count threshold.

``Method.conversion_inference="auto"`` sends an unadjusted conversion or retention contrast
to the delta-method route (``reference_kind="t"``) only when all four per-arm success and
failure counts reach ``dense_min_count(tail_alpha)`` (``increment.estimation.conversion_route``).
This module measures how far that route's one-sided noncoverage sits from its tail near the
threshold, and sets the threshold from the measurement.

A boundary cell has its sparsest expected count at ``m``: a rare event (the sparsest
success count is ``m``), a failure-limited rate (the sparsest failure count is ``m``), or a
central rate sized so that its sparsest success count is ``m``, at control sizes
``n_c in {1e3 .. 1e7}``, allocations 1:1, 1:4 and 4:1, and risk ratios
``{0.5, 1, 1.25, 2}``. For each tail the route's noncoverage is the probability, under the
cell's true rates, that the interval misses the true ratio on that side. It is computed by
integrating the interval over the joint binomial law of the counts, not drawn:
the interval is a function of the four counts alone, so integration adds no sampling error and
omits at most ``_OMITTED_MASS`` of probability per arm, which is reported.

Two noncoverages are tracked per tail. ``wald`` applies the delta-method interval to every
draw, whatever its counts: the route's own calibration at that cell. ``routed`` applies it
only to draws the rule routes asymptotic; the rest are the finite-sample route's, which is
valid at every count. The excess of either over the tail is read against
``tests.mc.scientific_delta(tail)``, the repository's tolerance (a tenth of the tail below
0.05, 0.005 at and above it). ``required_count(tail)`` is the smallest ``m`` whose ``wald``
excess is within tolerance at ``m`` and at every larger ladder step through ``1.25 * m``; the
shipped law is checked against it by ``verify``. ``wald`` is the stricter of the two and the
one the threshold follows: a draw the rule sends to the finite-sample route cannot make the
combined interval worse than the finite-sample route's own, which is valid, but the combined
noncoverage is only known from the finite-sample route's behaviour in the routed-away region,
which ``hybrid`` measures through the production route.

The vectorised delta-method interval here is the production formula (arm moments, Welch-
Satterthwaite ``t`` reference); ``conformance`` checks it against ``estimate_lift`` on sampled
count pairs before any table is trusted.

Planning is validated against the same production route. ``bound`` compares the planned power
of dense, sparse and borderline designs with the production pipeline's exact rejection
probability, summed over the count lattice; ``mirror`` compares it with the simulated rejection
rate of ``estimate_lift`` (``tests.estimation._conversion_counts.runtime_rejection_rate``).

Every lattice sum streams over blocks of control counts (``_BLOCK_CELLS`` cells each), so a
worker's footprint stays near a gibibyte whatever the arm sizes: the largest boundary cell has
a count lattice of tens of millions of cells.

    python -m calibration.conversion_route select --out /tmp/route-scan.jsonl
    python -m calibration.conversion_route verify
    python -m calibration.conversion_route conformance
    python -m calibration.conversion_route hybrid --workers 4 --tails 0.025 0.01
    python -m calibration.conversion_route bound --workers 4 --out /tmp/route-bound.jsonl
    python -m calibration.conversion_route mirror --reps 3000 --workers 4
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import multiprocessing
import multiprocessing.pool
import sys
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t

from increment.estimation.conversion_route import dense_min_count
from tests.estimation._conversion_counts import lift_row, runtime_rejection_rate
from tests.mc import (
    scientific_delta,
)

if TYPE_CHECKING:
    from increment.power._binomial import BinomialDecision, _Window

#: One-sided tails the production alphas produce: two-sided ``alpha / 2`` for
#: ``alpha in {.001, .01, .05, .1}`` and directional ``alpha in {.001, .01, .05, .1}``.
TAILS = (0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
RISK_RATIOS = (0.5, 1.0, 1.25, 2.0)
ALLOCATIONS = ((1, 1), (1, 4), (4, 1))
#: Control sizes at every 1, 2 and 5 of a decade: a boundary cell exists only while its sparsest
#: expected count fits in its arms, so a sparser grid drops the worst cell at the sizes just
#: beyond one it can hold and the measured excess falls by steps that carry no information.
CONTROL_SIZES = tuple(
    int(mantissa * 10**decade) for decade in range(3, 7) for mantissa in (1, 2, 5)
) + (10_000_000,)
#: Candidate thresholds of the original ladder, then geometric steps of 1.15 above it.
LADDER_STEP = 1.15
MARGIN = 1.25
#: Probability per arm the integration omits at each end of the count lattice.
_OMITTED_MASS = 1e-13
Family = Literal["rare", "failure", "central"]


@dataclass(frozen=True, slots=True)
class Cell:
    """True rates and sizes of one boundary cell; ``m`` is its sparsest expected count."""

    family: Family
    n_c: int
    n_t: int
    p_c: float
    p_t: float
    risk_ratio: float
    m: float

    @property
    def truth(self) -> float:
        return math.log(self.p_t / self.p_c)


def cells(m: float) -> tuple[Cell, ...]:
    """The boundary cells whose sparsest expected count is ``m``."""
    out: dict[tuple[int, int, float, float], Cell] = {}

    def add(family: Family, n_c: int, n_t: int, p_c: float, p_t: float, rr: float) -> None:
        if not (0.0 < p_c < 1.0 and 0.0 < p_t < 1.0) or n_t < 2 or n_c < 2:
            return
        expected = min(n_c * p_c, n_c * (1 - p_c), n_t * p_t, n_t * (1 - p_t))
        if expected < 0.999 * m:
            return
        out.setdefault(
            (n_c, n_t, round(p_c, 12), round(p_t, 12)), Cell(family, n_c, n_t, p_c, p_t, rr, m)
        )

    for a, b in ALLOCATIONS:
        for rr in RISK_RATIOS:
            for n_c in CONTROL_SIZES:
                n_t = n_c * b // a
                # rare: the lower of the two expected success counts is m
                scale = min(1.0, rr * n_t / n_c)
                p_c = m / (n_c * scale)
                if p_c <= 0.5 and rr * p_c <= 0.5:
                    add("rare", n_c, n_t, p_c, rr * p_c, rr)
                # failure-limited: the higher-rate arm has m expected failures
                if rr >= 1.0:
                    p_t = 1.0 - m / n_t
                    add("failure", n_c, n_t, p_t / rr, p_t, rr)
                else:
                    p_c = 1.0 - m / n_c
                    add("failure", n_c, n_t, p_c, p_c * rr, rr)
            # central: a 0.3 control rate at the size that makes the sparsest success count m
            n_c = math.ceil(m / 0.3 / min(1.0, rr * b / a))
            n_t = max(2, n_c * b // a)
            add("central", n_c, n_t, 0.3, min(0.999, rr * 0.3), rr)
    return tuple(out.values())


def _lattice(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    lo = max(0, int(_binom.ppf(_OMITTED_MASS, n, p)))
    hi = min(n, int(_binom.isf(_OMITTED_MASS, n, p)))
    counts = np.arange(lo, hi + 1)
    return counts, _binom.pmf(counts, n, p)


#: Cells of the joint count lattice held at once. Every sum over a lattice streams over blocks of
#: control counts of at most this many cells, so a worker's footprint is set by the block and not
#: by the arms' windows: a cell of ten million units per arm at a central rate has a window of
#: tens of thousands of counts, and its whole lattice is tens of millions of cells.
_BLOCK_CELLS = 1 << 21


def _row_blocks(rows: int, width: int) -> Iterator[slice]:
    """Slices of ``rows`` control counts holding at most ``_BLOCK_CELLS`` cells at ``width``
    treatment counts each, and at least one row."""
    step = max(1, _BLOCK_CELLS // max(1, width))
    for start in range(0, rows, step):
        yield slice(start, min(rows, start + step))


def _statistic(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(log_rr, se, df)`` of the delta-method route for count arrays with ``0 < x < n``: each
    arm's centered moments give ``mean = x / n`` and ``var = x (n - x) / (n (n - 1))``, its
    log standard error is ``sqrt(var / (n mean ** 2))``, the combined one their hypotenuse,
    and the reference's degrees of freedom are Welch-Satterthwaite."""
    with np.errstate(all="ignore"):
        mean_c, mean_t = x_c / n_c, x_t / n_t
        var_c = x_c * (n_c - x_c) / (n_c * (n_c - 1.0))
        var_t = x_t * (n_t - x_t) / (n_t * (n_t - 1.0))
        se_c = np.sqrt(var_c / (n_c * mean_c**2))
        se_t = np.sqrt(var_t / (n_t * mean_t**2))
        log_rr = np.log(mean_t) - np.log(mean_c)
        se = np.hypot(se_c, se_t)
        scale = np.maximum(se_c, se_t)
        a, b = se_t / scale, se_c / scale
        df = (a * a + b * b) ** 2 / (a**4 / (n_t - 1) + b**4 / (n_c - 1))
    return log_rr, se, df


#: Degree-of-freedom floor and node count of the Chebyshev interpolation in ``1 / df`` that
#: stands in for a per-point ``t`` quantile; above the floor the quantile is analytic in
#: ``1 / df`` over a lattice's narrow range and the interpolant agrees with ``t.isf`` to
#: ``_CRITICAL_AGREEMENT`` (checked by ``conformance``).
_INTERPOLATION_DF_FLOOR = 30.0
_INTERPOLATION_NODES = 24
_CRITICAL_AGREEMENT = 1e-12


def _critical(df: np.ndarray, tail: float) -> np.ndarray:
    """Student ``t`` upper-tail critical values at ``df``."""
    finite = df[np.isfinite(df)]
    if finite.size < 4096 or finite.min() < _INTERPOLATION_DF_FLOOR:
        return _student_t.isf(tail, df)
    lo, hi = 1.0 / finite.max(), 1.0 / finite.min()
    nodes = np.cos(np.pi * (np.arange(_INTERPOLATION_NODES) + 0.5) / _INTERPOLATION_NODES)
    inverse = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
    coefficients = np.polynomial.chebyshev.chebfit(
        nodes, _student_t.isf(tail, 1.0 / inverse), _INTERPOLATION_NODES - 1
    )
    with np.errstate(all="ignore"):
        position = (1.0 / df - 0.5 * (lo + hi)) * (2.0 / (hi - lo)) if hi > lo else 0.0 * df
    return np.polynomial.chebyshev.chebval(position, coefficients)


def delta_method_bounds(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int, tail: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log risk-ratio interval bounds the production delta-method route reports at
    one-sided ``tail``, for count arrays with ``0 < x < n``."""
    log_rr, se, df = _statistic(x_c, n_c, x_t, n_t)
    crit = _critical(df, tail)
    return log_rr - crit * se, log_rr + crit * se


def boundary_noncoverage(
    cell: Cell, tails: Sequence[float], threshold: dict[float, int] | int
) -> dict[float, tuple[float, float, float, float]]:
    """Per tail: ``(wald_lower, wald_upper, routed_lower, routed_upper)`` noncoverage of the
    delta-method interval at ``cell``, applied to every draw with positive counts (``wald``) and
    to the draws whose four counts reach ``threshold`` (``routed``). ``threshold`` is one count,
    or a count per tail."""
    x_c, w_c = _lattice(cell.n_c, cell.p_c)
    x_t, w_t = _lattice(cell.n_t, cell.p_t)
    totals = {tail: np.zeros(4) for tail in tails}
    for rows in _row_blocks(x_c.size, x_t.size):
        grid_c = x_c[rows, None]
        weight = np.outer(w_c[rows], w_t)
        smallest = np.minimum(
            np.minimum(grid_c, cell.n_c - grid_c), np.minimum(x_t, cell.n_t - x_t)[None, :]
        )
        defined = smallest >= 1
        log_rr, se, df = _statistic(grid_c, cell.n_c, x_t[None, :], cell.n_t)
        for tail in tails:
            crit = _critical(df, tail)
            floor = threshold[tail] if isinstance(threshold, dict) else threshold
            routed = defined & (smallest >= floor)
            low_miss = defined & (log_rr - crit * se > cell.truth)
            up_miss = defined & (log_rr + crit * se < cell.truth)
            totals[tail] += (
                weight[low_miss].sum(),
                weight[up_miss].sum(),
                weight[low_miss & routed].sum(),
                weight[up_miss & routed].sum(),
            )
    return {tail: (float(a), float(b), float(c), float(d)) for tail, (a, b, c, d) in totals.items()}


@dataclass(frozen=True, slots=True)
class Excess:
    """Worst boundary cell of one ``(tail, m)``: excess over the tail in tolerance units."""

    tail: float
    m: float
    wald: float
    routed: float
    cell: Cell | None


_POOLS: dict[int, multiprocessing.pool.Pool] = {}


def _pool(workers: int) -> multiprocessing.pool.Pool:
    """One process pool per size for the whole run: a worker imports the package once."""
    if workers not in _POOLS:
        _POOLS[workers] = multiprocessing.Pool(workers)
    return _POOLS[workers]


@atexit.register
def _close_pools() -> None:
    for pool in _POOLS.values():
        pool.terminate()


Noncoverage = dict[float, tuple[float, float, float, float]]


def _excess_job(args: tuple[int, Cell, tuple[float, ...]]) -> tuple[Cell, Noncoverage]:
    threshold, cell, tails = args
    return cell, boundary_noncoverage(cell, tails, threshold)


def noncoverage_table(
    m: float,
    tails: Sequence[float] = TAILS,
    *,
    threshold: int | None = None,
    workers: int = 1,
) -> list[tuple[Cell, Noncoverage]]:
    """Every boundary cell of design count ``m`` with its noncoverage at each tail, the rule
    taking the routed side at ``threshold`` (``round(m)`` unless given)."""
    floor = round(m) if threshold is None else threshold
    jobs = [(floor, cell, tuple(tails)) for cell in cells(m)]
    if workers > 1:
        return _pool(workers).map(_excess_job, jobs)
    return [_excess_job(job) for job in jobs]


def worst_excess(
    m: float, tails: Sequence[float] = TAILS, *, workers: int = 1
) -> dict[float, Excess]:
    """Worst ``wald`` and ``routed`` excess over every boundary cell at design count ``m``,
    per tail, in units of ``scientific_delta(tail)`` (at most one passes)."""
    worst: dict[float, Excess] = {}
    for cell, table in noncoverage_table(m, tails, workers=workers):
        for tail, (w_lo, w_up, r_lo, r_up) in table.items():
            delta = scientific_delta(tail)
            wald = (max(w_lo, w_up) - tail) / delta
            routed = (max(r_lo, r_up) - tail) / delta
            current = worst.get(tail)
            if current is None or wald > current.wald:
                worst[tail] = Excess(
                    tail, m, wald, max(routed, current.routed) if current else routed, cell
                )
            elif routed > current.routed:
                worst[tail] = Excess(tail, m, current.wald, routed, current.cell)
    return worst


def ladder(start: float = 10.0, stop: float = 40_000.0) -> list[int]:
    steps: list[int] = []
    m = start
    while m < stop:
        steps.append(round(m))
        m *= LADDER_STEP
    return steps


#: Ladder steps are rounded to integers, so a measured step may sit this far above one ladder
#: step over its predecessor.
_LADDER_ROUNDING = 2


def _covers(steps: Sequence[int], limit: float) -> bool:
    """Whether ``steps`` (ascending, the first being a candidate) measure every ladder step
    through ``limit``: no gap wider than one ladder step, and a last step the next ladder step
    above which exceeds ``limit``."""
    within = [step for step in steps if step <= limit]
    gaps_closed = all(
        b <= LADDER_STEP * a + _LADDER_ROUNDING
        for a, b in zip(steps, steps[1:], strict=False)
        if b <= limit
    )
    return gaps_closed and LADDER_STEP * within[-1] + _LADDER_ROUNDING > limit


def required_count(
    rows: dict[int, dict[float, Excess]], tail: float, *, key: str = "wald"
) -> int | None:
    """Smallest measured ``m`` whose excess is within tolerance at ``m`` and at every larger
    measured step, with every ladder step through ``MARGIN * m`` measured at that tail;
    ``None`` when no candidate has that coverage."""
    ms = sorted(m for m in rows if tail in rows[m])
    passing = [getattr(rows[m][tail], key) <= 1.0 for m in ms]
    for index, m in enumerate(ms):
        if all(passing[index:]) and _covers(ms[index:], MARGIN * m):
            return m
    return None


def select(
    out: Path | None, *, workers: int, start: float, stop: float, tails: Sequence[float] = TAILS
) -> int:
    """Measure every ladder step and report the required count per tail and the shipped law.
    Each step is appended to ``out`` as it completes, so an interrupted run keeps its steps."""
    rows: dict[int, dict[float, Excess]] = {}
    if out is not None:
        out.write_text("")
    for m in ladder(start, stop):
        rows[m] = worst_excess(m, tails, workers=workers)
        lines = [
            {
                "m": m,
                "tail": tail,
                "delta": scientific_delta(tail),
                "wald_excess": excess.wald,
                "routed_excess": excess.routed,
                "cell": asdict(excess.cell) if excess.cell else None,
            }
            for tail, excess in rows[m].items()
        ]
        if out is not None:
            with out.open("a") as handle:
                handle.write("".join(json.dumps(line) + "\n" for line in lines))
        print(
            f"m={m:>6}  "
            + "  ".join(
                f"{tail:g}:{rows[m][tail].wald:6.2f}/{rows[m][tail].routed:6.2f}" for tail in tails
            ),
            flush=True,
        )
    return required(rows, tails)


def required(rows: dict[int, dict[float, Excess]], tails: Sequence[float] = TAILS) -> int:
    """Print the required count per tail beside the shipped law."""
    for tail in tails:
        print(
            json.dumps(
                {
                    "tail": tail,
                    "z": float(_norm.isf(tail)),
                    "required_wald": required_count(rows, tail),
                    "required_routed": required_count(rows, tail, key="routed"),
                    "shipped": dense_min_count(tail),
                }
            )
        )
    return 0


def read_rows(paths: Sequence[Path]) -> dict[int, dict[float, Excess]]:
    """The ladder rows ``select`` wrote to ``paths``, merged (a later file replaces an earlier
    one at the same ``m`` and tail)."""
    rows: dict[int, dict[float, Excess]] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "m" not in record:
                continue
            cell = Cell(**record["cell"]) if record["cell"] else None
            rows.setdefault(record["m"], {})[record["tail"]] = Excess(
                record["tail"], record["m"], record["wald_excess"], record["routed_excess"], cell
            )
    return rows


def verify(*, workers: int, tails: Sequence[float] = TAILS) -> int:
    """The shipped threshold at each tail passes at its own count and at ``MARGIN`` times it."""
    failed = 0
    for tail in tails:
        shipped = dense_min_count(tail)
        for m in (shipped, math.ceil(MARGIN * shipped)):
            excess = worst_excess(m, (tail,), workers=workers)[tail]
            ok = excess.wald <= 1.0
            failed += not ok
            print(
                f"tail={tail:<7g} m={m:>6} wald_excess={excess.wald:6.3f} "
                f"routed_excess={excess.routed:6.3f} {'ok' if ok else 'FAIL'}"
            )
    return 1 if failed else 0


# --- Production route -------------------------------------------------------------------


def row_misses(row, truth_ratio: float) -> tuple[bool, bool]:
    """``(lower, upper)``: whether the row's interval lies wholly above / below the true
    relative lift ``truth_ratio - 1``."""
    if row.binomial_set is not None:
        lower, upper = row.binomial_set.lower, row.binomial_set.upper
    else:
        lower, upper = row.lift.lb, row.lift.ub
    lift = truth_ratio - 1.0
    return (lower is not None and lower > lift), (upper is not None and upper < lift)


def conformance(samples: int = 200, seed: int = 20261004) -> tuple[float, int, float]:
    """``(gap, compared, interpolation_gap)``: the largest relative gap between
    ``delta_method_bounds`` and the delta-method row ``estimate_lift`` returns over ``compared``
    sampled count pairs the rule routes asymptotic at their tail, and the largest relative gap
    between the interpolated and the direct ``t`` quantile over sampled degrees of freedom."""
    rng = np.random.default_rng(seed)
    worst, compared = 0.0, 0
    for _ in range(samples):
        tail = float(rng.choice(TAILS))
        m = dense_min_count(tail)
        n_c = int(rng.integers(2 * m, 40 * m))
        n_t = int(rng.integers(2 * m, 40 * m))
        x_c = int(rng.integers(m, n_c - m + 1))
        x_t = int(rng.integers(m, n_t - m + 1))
        row = lift_row((x_c, n_c, x_t, n_t), alpha=2.0 * tail)
        assert row.reference_kind == "t", (x_c, n_c, x_t, n_t, tail)
        lower, upper = delta_method_bounds(np.array([x_c]), n_c, np.array([x_t]), n_t, tail)
        compared += 1
        lift = row.require_lift()
        assert lift.lb is not None and lift.ub is not None
        worst = max(
            worst,
            abs(math.expm1(float(lower[0])) - lift.lb) / max(1e-12, abs(lift.lb)),
            abs(math.expm1(float(upper[0])) - lift.ub) / max(1e-12, abs(lift.ub)),
        )
    interpolation = 0.0
    for tail in TAILS:
        for df_lo, df_hi in ((30.0, 60.0), (800.0, 1600.0), (2e4, 8e4), (5e6, 2e7)):
            df = rng.uniform(df_lo, df_hi, size=5000)
            interpolation = max(
                interpolation,
                float(np.max(np.abs(_critical(df, tail) / _student_t.isf(tail, df) - 1.0))),
            )
    return worst, compared, interpolation


@dataclass(frozen=True, slots=True)
class HybridResult:
    cell: Cell
    alpha: float
    alternative: str
    reps: int
    lower_misses: int
    upper_misses: int
    asymptotic_share: float


def simulate_hybrid(
    cell: Cell,
    *,
    alpha: float,
    alternative: str,
    reps: int,
    seed: int,
    mode: Literal["auto", "finite_sample"] = "auto",
) -> HybridResult:
    """Noncoverage of the production pipeline in ``cell``: ``reps`` seeded binomial draws of
    the four counts, each routed and inferred by ``estimate_lift``. Every replication stays
    in the denominator; production runs once per distinct count pair."""
    rng = np.random.default_rng(seed)
    x_c = rng.binomial(cell.n_c, cell.p_c, size=reps)
    x_t = rng.binomial(cell.n_t, cell.p_t, size=reps)
    pairs, multiplicity = np.unique(np.stack([x_c, x_t]), axis=1, return_counts=True)
    ratio = cell.p_t / cell.p_c
    lower = upper = asymptotic = 0
    for (c, t), times in zip(pairs.T, multiplicity, strict=True):
        row = lift_row(
            (int(c), cell.n_c, int(t), cell.n_t), alpha=alpha, alternative=alternative, mode=mode
        )
        miss_lower, miss_upper = row_misses(row, ratio)
        lower += int(times) * miss_lower
        upper += int(times) * miss_upper
        asymptotic += int(times) * (row.reference_kind == "t")
    return HybridResult(cell, alpha, alternative, reps, lower, upper, asymptotic / reps)


def replicates(tail: float) -> int:
    """Seeded draws that resolve noncoverage at ``tail`` to its tolerance: the repository's
    ``16 tail (1 - tail) / delta ** 2``, a Monte Carlo standard error of ``delta / 4``."""
    return math.ceil(16.0 * tail * (1.0 - tail) / scientific_delta(tail) ** 2)


@dataclass(frozen=True, slots=True)
class HybridNoncoverage:
    """Exact per-side noncoverage of the production pipeline at one cell."""

    lower: float
    upper: float
    asymptotic_share: float
    omitted: float


def _runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    """Maximal runs of equal values in ``flags``: ``(start, stop, value)``."""
    edges = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = [0, *edges.tolist()]
    stops = [*edges.tolist(), flags.size]
    return [(a, b, bool(flags[a])) for a, b in zip(starts, stops, strict=True)]


def _finite_blocks(
    key: BinomialDecision, window_c: _Window, window_t: _Window, floor: int | None = None
) -> Iterator[tuple[slice, np.ndarray, np.ndarray]]:
    """``(rows, plus, minus)`` per block of control counts: where the runtime's finite-sample
    decision rejects, ``rows`` indexing ``window_c``. The planner replays that decision from
    ``RejectionGeometry`` on its ``exact`` route whatever the cell count (the runtime's own
    search, not the surrogate the planner substitutes above its budget). Each count pair has its
    own replay, so a region is classified by a geometry of its own and decides as the whole
    window would, while no more than ``_BLOCK_CELLS`` cells are stored.

    With ``floor``, only the pairs with a success or failure count below ``floor`` are
    classified, the pairs the finite-sample route decides when ``floor`` is the delta-method
    route's count threshold; every other cell is ``False``, and the caller must read it
    through its routing mask."""
    from increment.power._binomial import PLANNING_CELL_CEILING, RejectionGeometry

    x_t = np.arange(window_t.lo, window_t.hi + 1)
    column_spans = [
        (a, b)
        for a, b, undecided in _runs(
            np.ones(x_t.size, bool) if floor is None else np.minimum(x_t, key.n_t - x_t) < floor
        )
        if undecided
    ]
    for rows in _row_blocks(window_c.size, x_t.size):
        x_c = np.arange(window_c.lo + rows.start, window_c.lo + rows.stop)
        plus = np.zeros((x_c.size, x_t.size), bool)
        minus = np.zeros((x_c.size, x_t.size), bool)
        undecided_rows = (
            np.ones(x_c.size, bool) if floor is None else np.minimum(x_c, key.n_c - x_c) < floor
        )
        for first, stop, every_column in _runs(undecided_rows):
            for a, b in [(0, x_t.size)] if every_column else column_spans:
                region_plus, region_minus = RejectionGeometry(
                    key, "exact", PLANNING_CELL_CEILING
                ).cells(int(x_c[first]), int(x_c[stop - 1]), window_t.lo + a, window_t.lo + b - 1)
                plus[first:stop, a:b] = region_plus
                minus[first:stop, a:b] = region_minus
        yield rows, plus, minus


def _decision_key(
    cell: Cell, *, alpha: float, alternative: Literal["two-sided", "greater", "less"]
) -> BinomialDecision:
    """The runtime's finite-sample decision of ``cell``'s contrast at its true ratio."""
    from increment.estimation.binomial_rr import nuisance_beta
    from increment.power._binomial import BinomialDecision

    tail = alpha / 2.0 if alternative == "two-sided" else alpha
    return BinomialDecision(
        cell.n_c, cell.n_t, cell.p_t / cell.p_c, nuisance_beta(alpha), tail, alternative
    )


def finite_sample_misses(
    cell: Cell, *, alpha: float, alternative: Literal["two-sided", "greater", "less"]
) -> tuple[np.ndarray, np.ndarray, _Window, _Window]:
    """``(plus, minus, window_c, window_t)``: over the count lattice of ``cell``, where the
    finite-sample set misses the true ratio. Its miss is the rejection of the test at that
    ratio (``_finite_blocks``): a ``plus`` rejection puts the set wholly above the truth, a
    ``minus`` one wholly below. The whole lattice is held, so this is for cells whose windows
    are small."""
    from increment.power._binomial import _window

    window_c, window_t = _window(cell.n_c, cell.p_c), _window(cell.n_t, cell.p_t)
    blocks = list(
        _finite_blocks(
            _decision_key(cell, alpha=alpha, alternative=alternative), window_c, window_t
        )
    )
    return (
        np.concatenate([plus for _, plus, _ in blocks]),
        np.concatenate([minus for _, _, minus in blocks]),
        window_c,
        window_t,
    )


def hybrid_noncoverage(
    cell: Cell,
    *,
    alpha: float,
    threshold: int,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> HybridNoncoverage:
    """Exact noncoverage of the pipeline at ``cell``, summed over the joint binomial law of its
    counts: a pair whose four counts reach ``threshold`` is covered by the delta-method
    interval (the vectorised production interval, ``conformance``), every other pair by the
    finite-sample set (``_finite_blocks``, checked against ``estimate_lift`` by
    ``conformance``). Counts are not drawn, so there is no sampling error; the lattice omits at
    most the windows' ``omitted`` mass. The sums stream over blocks of control counts, so the
    footprint does not grow with the windows."""
    from increment.power._binomial import _window

    tail = alpha / 2.0 if alternative == "two-sided" else alpha
    window_c, window_t = _window(cell.n_c, cell.p_c), _window(cell.n_t, cell.p_t)
    key = _decision_key(cell, alpha=alpha, alternative=alternative)
    x_t = np.arange(window_t.lo, window_t.hi + 1)[None, :]
    lower = upper = share = 0.0
    for rows, plus, minus in _finite_blocks(key, window_c, window_t, threshold):
        x_c = np.arange(window_c.lo + rows.start, window_c.lo + rows.stop)[:, None]
        weight = np.outer(window_c.weights[rows], window_t.weights)
        smallest = np.minimum(np.minimum(x_c, cell.n_c - x_c), np.minimum(x_t, cell.n_t - x_t))
        log_rr, se, df = _statistic(
            np.clip(x_c, 1, cell.n_c - 1), cell.n_c, np.clip(x_t, 1, cell.n_t - 1), cell.n_t
        )
        crit = _critical(df, tail)
        routed = smallest >= threshold
        lower += float(weight[np.where(routed, log_rr - crit * se > cell.truth, plus)].sum())
        upper += float(weight[np.where(routed, log_rr + crit * se < cell.truth, minus)].sum())
        share += float(weight[routed].sum())
    return HybridNoncoverage(lower, upper, share, window_c.omitted + window_t.omitted)


#: Distance from the true lift within which a production endpoint and the replayed test at
#: that lift may disagree: the endpoints come from a root search, the replay from the test.
_ENDPOINT_TOLERANCE = 1e-3


def finite_conformance(
    per_tail: int = 6, seed: int = 20261004, tails: Sequence[float] = TAILS
) -> tuple[int, int, int]:
    """``(compared, edge, hard)``: the production finite-sample set against
    ``finite_sample_misses`` where they could differ, at the decision's edge. At each tail's
    central boundary cell, ``per_tail`` control counts are drawn from its law; in each row
    the pair either side of each miss region's edge (the first ``plus`` miss and its
    predecessor, the last ``minus`` miss and its successor) is run through ``estimate_lift``
    forced to the finite-sample route, and its reported miss of the true ratio is compared
    with the replay's. ``edge`` counts the disagreements whose nearer endpoint lies within
    ``_ENDPOINT_TOLERANCE`` of the true lift (the endpoint search's own resolution);
    ``hard`` counts every other."""
    rng = np.random.default_rng(seed)
    compared = edge = hard = 0
    for tail in tails:
        cell = next(c for c in cells(dense_min_count(tail)) if c.family == "central")
        alpha = 2.0 * tail
        plus, minus, window_c, window_t = finite_sample_misses(
            cell, alpha=alpha, alternative="two-sided"
        )
        width = window_t.hi - window_t.lo + 1
        lift = cell.p_t / cell.p_c - 1.0
        rows = rng.choice(
            window_c.weights.size, size=per_tail, p=window_c.weights / window_c.weights.sum()
        )
        for row in rows:
            probes: set[int] = set()
            if plus[row].any():
                first = int(np.argmax(plus[row]))
                probes |= {first - 1, first}
            if minus[row].any():
                last = width - 1 - int(np.argmax(minus[row][::-1]))
                probes |= {last, last + 1}
            for column in sorted(probe for probe in probes if 0 <= probe < width):
                x_c, x_t = window_c.lo + int(row), window_t.lo + column
                if not (0 < x_c < cell.n_c and 0 < x_t < cell.n_t):
                    continue
                produced = lift_row(
                    (x_c, cell.n_c, x_t, cell.n_t), alpha=alpha, mode="finite_sample"
                )
                compared += 1
                if row_misses(produced, cell.p_t / cell.p_c) == (
                    bool(plus[row, column]),
                    bool(minus[row, column]),
                ):
                    continue
                found = produced.binomial_set
                assert found is not None
                assert found.lower is not None and found.upper is not None
                gap = min(abs(found.lower - lift), abs(found.upper - lift))
                edge += gap <= _ENDPOINT_TOLERANCE
                hard += gap > _ENDPOINT_TOLERANCE
    return compared, edge, hard


def hybrid_cells(tail: float, offset: int, *, count: int, workers: int = 1) -> list[Cell]:
    """The ``count`` boundary cells at design count ``dense_min_count(tail) + offset`` whose
    routed (delta-method) side has the largest per-tail noncoverage: the cells where the
    finite-sample side has the least room."""
    shipped = dense_min_count(tail)
    table = noncoverage_table(shipped + offset, (tail,), threshold=shipped, workers=workers)
    ranked = sorted(table, key=lambda item: -max(item[1][tail][2], item[1][tail][3]))
    return [cell for cell, _ in ranked[:count]]


def _hybrid_job(args: tuple[Cell, float, int, Literal["two-sided", "greater", "less"]]):
    cell, alpha, threshold, alternative = args
    return hybrid_noncoverage(cell, alpha=alpha, threshold=threshold, alternative=alternative)


OFFSETS = (-2, -1, 0, 1, 2)


def hybrid(
    tails: Sequence[float], *, workers: int, count: int = 3, replicate_tails: Sequence[float] = ()
) -> int:
    """Production-pipeline noncoverage at cells straddling each tail's shipped threshold
    (design counts ``m - 2 .. m + 2``): per-tail and unconditional two-sided noncoverage
    within the repository's tolerance of its nominal level, two-sided at ``alpha = 2 tail`` and
    directional at ``alpha = tail``. Then ``replicate_tails``: the pipeline run through
    ``estimate_lift`` on ``replicates(tail)`` seeded draws at the worst cell, which must agree
    with the exact value within four Monte Carlo standard errors."""
    failed = 0
    for tail in tails:
        shipped = dense_min_count(tail)
        jobs: list[tuple[Cell, float, int, Literal["two-sided", "greater", "less"]]] = []
        for offset in OFFSETS:
            for rank, cell in enumerate(hybrid_cells(tail, offset, count=count, workers=workers)):
                jobs.append((cell, 2.0 * tail, shipped, "two-sided"))
                if rank == 0:
                    jobs.append((cell, tail, shipped, "greater"))
        results = _pool(workers).imap(_hybrid_job, jobs) if workers > 1 else map(_hybrid_job, jobs)
        for (cell, alpha, _, alternative), result in zip(jobs, results, strict=True):
            delta, level = scientific_delta(tail), 2.0 * tail
            ok = max(result.lower, result.upper) <= tail + delta + result.omitted
            if alternative == "two-sided":
                ok &= (
                    result.lower + result.upper <= level + scientific_delta(level) + result.omitted
                )
            failed += not ok
            print(
                f"tail={tail:<7g} m={cell.m:<7g} {alternative:<9} alpha={alpha:<6g} "
                f"cell={cell.family}/{cell.n_c}/{cell.n_t}/rr{cell.risk_ratio:g} "
                f"asym_share={result.asymptotic_share:.3f} lower={result.lower:.6f} "
                f"upper={result.upper:.6f} "
                f"excess={(max(result.lower, result.upper) - tail) / delta:+.3f} "
                f"{'ok' if ok else 'FAIL'}",
                flush=True,
            )
        if tail in replicate_tails:
            failed += not replicate_check(tail, workers=workers)
    return 1 if failed else 0


def replicate_check(tail: float, *, workers: int, seed: int = 20261004) -> bool:
    """The production pipeline on ``replicates(tail)`` seeded draws at the worst boundary cell
    of the shipped threshold, against the exact noncoverage of that cell."""
    shipped = dense_min_count(tail)
    (cell,) = hybrid_cells(tail, 0, count=1, workers=workers)
    exact = hybrid_noncoverage(cell, alpha=2.0 * tail, threshold=shipped)
    reps = replicates(tail)
    result = simulate_hybrid(cell, alpha=2.0 * tail, alternative="two-sided", reps=reps, seed=seed)
    ok = True
    for label, hits, truth in (
        ("lower", result.lower_misses, exact.lower),
        ("upper", result.upper_misses, exact.upper),
    ):
        se = math.sqrt(max(truth * (1.0 - truth), 1.0 / reps) / reps)
        agree = abs(hits / reps - truth) <= 4.0 * se + exact.omitted
        ok &= agree
        print(
            f"replicates tail={tail:g} {label} cell={cell.family}/{cell.n_c}/{cell.n_t} "
            f"reps={reps} simulated={hits / reps:.6f} exact={truth:.6f} se={se:.6f} "
            f"asym_share={result.asymptotic_share:.3f} {'ok' if agree else 'FAIL'}",
            flush=True,
        )
    return ok


@dataclass(frozen=True, slots=True)
class MirrorCell:
    """A planned design: ``n`` units per arm at control rate ``p_c`` and relative ``lift``."""

    label: str
    n: int
    p_c: float
    lift: float
    alpha: float
    alternative: Literal["two-sided", "greater", "less"]


def mirror_cells() -> tuple[MirrorCell, ...]:
    """Designs whose planning route is dense, sparse and borderline at the production tails."""
    cells: list[MirrorCell] = []
    for alpha, alternative in ((0.05, "two-sided"), (0.1, "greater"), (0.2, "two-sided")):
        tail = alpha / 2.0 if alternative == "two-sided" else alpha
        m = dense_min_count(tail)
        for p_c, lift in ((0.3, 0.05), (0.15, 0.1), (0.6, 0.04)):
            for label, factor in (("dense", 6.0), ("borderline", 1.04), ("sparse", 0.25)):
                n = math.ceil(factor * m / min(p_c, 1.0 - p_c))
                cells.append(MirrorCell(label, n, p_c, lift, alpha, alternative))
    return tuple(cells)


def _mirror_job(args: tuple[MirrorCell, int, int]) -> tuple[str, bool]:
    """One mirror design's report line and whether it passes."""
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.estimation.conversion_route import planning_route
    from increment.power import Baseline, achieved_power

    cell, reps, seed = args
    tail = cell.alpha / 2.0 if cell.alternative == "two-sided" else cell.alpha
    route = planning_route(
        cell.n, cell.n, cell.p_c, cell.p_c * (1.0 + cell.lift), tail_alpha=tail, mode="auto"
    )
    procedure = ArmPlanningProcedure.standard(
        "conversion", alpha=cell.alpha, alternative=cell.alternative
    )
    planned = achieved_power(cell.n, cell.lift, Baseline.from_proportion(cell.p_c), procedure)
    rate, se, share = runtime_rejection_rate(
        cell.n,
        cell.n,
        cell.p_c,
        cell.p_c * (1.0 + cell.lift),
        alpha=cell.alpha,
        alternative=cell.alternative,
        reps=reps,
        seed=seed,
    )
    gap = planned.power - rate
    ok = (
        abs(gap) <= 4 * se + 0.003
        if route == "dense"
        else abs(gap) <= 4 * se
        if route == "sparse"
        else gap <= 4 * se
    )
    return (
        f"{cell.label:<10} route={route:<10} n={cell.n:>8} p_c={cell.p_c:<5} "
        f"lift={cell.lift:<5} alpha={cell.alpha:<5} {cell.alternative:<9} "
        f"planned={planned.power:.4f} ({planned.power_basis}) simulated={rate:.4f} "
        f"se={se:.4f} asym_share={share:.2f} {'ok' if ok else 'FAIL'}",
        ok,
    )


def mirror(reps: int, seed: int, *, workers: int = 1) -> int:
    """Planned power against the simulated production rejection rate, per design: a dense plan
    within ``4 * se + 0.003``; a sparse plan within ``4 * se`` (its replay is exact up to the
    window mass); a borderline plan no larger than the simulated rate plus ``4 * se``."""
    jobs = [(cell, reps, seed) for cell in mirror_cells()]
    results = _pool(workers).imap(_mirror_job, jobs) if workers > 1 else map(_mirror_job, jobs)
    failed = 0
    for line, ok in results:
        failed += not ok
        print(line, flush=True)
    return 1 if failed else 0


@dataclass(frozen=True, slots=True)
class EnumeratedPower:
    """A planned design's power against its exact rejection probabilities, summed over the
    joint binomial law of the counts. ``asymptotic`` is the delta-method test applied to every
    draw; the production pipeline decides each draw by the route its counts select, so its
    rejection probability ``hybrid`` is the part the delta method decides (``asymptotic_part``,
    the routed draws it rejects) plus the part the finite-sample test decides
    (``finite_part``, the draws the rule keeps on it)."""

    route: str
    planned: float
    basis: str
    asymptotic_share: float
    asymptotic: float
    asymptotic_part: float
    finite_part: float
    omitted: float

    @property
    def hybrid(self) -> float:
        """The production pipeline's rejection probability."""
        return self.asymptotic_part + self.finite_part

    @property
    def margin(self) -> float:
        """How far the planned power sits below the hybrid rejection probability."""
        return self.hybrid - self.planned


def enumerated_power(cell: MirrorCell) -> EnumeratedPower:
    """Exact rejection probabilities of ``cell``'s design by enumerating its count lattice.

    The finite-sample decision of each count pair is the runtime's decision replayed on the
    ``exact`` rejection geometry whatever the cell count, never the surrogate the planner
    substitutes above its budget, so the planned side alone carries the planner's
    approximation; the delta-method decision is the vectorised production interval
    (``conformance``). A pair is decided by the delta method iff its four counts reach
    ``dense_min_count``, so the sum over the pairs is the pipeline's rejection probability,
    not a bound on it. Pairs with a zero count are always finite-sample. The sums stream over
    blocks of control counts, and only the pairs the finite-sample route decides are replayed
    (``_finite_blocks``).
    """
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.estimation.conversion_route import planning_route
    from increment.power import Baseline, achieved_power
    from increment.power._binomial import _window
    from increment.power.core import _binomial_key

    procedure = ArmPlanningProcedure.standard(
        "conversion", alpha=cell.alpha, alternative=cell.alternative
    )
    tail = procedure.compiled_tail_alpha
    p_t = cell.p_c * (1.0 + cell.lift)
    planned = achieved_power(cell.n, cell.lift, Baseline.from_proportion(cell.p_c), procedure)
    key = _binomial_key(procedure, cell.n, cell.n)
    window_c, window_t = _window(cell.n, cell.p_c), _window(cell.n, p_t)
    x_t = np.arange(window_t.lo, window_t.hi + 1)[None, :]
    floor = dense_min_count(tail)
    share = asymptotic = asymptotic_part = finite_part = 0.0
    for rows, plus, minus in _finite_blocks(key, window_c, window_t, floor):
        x_c = np.arange(window_c.lo + rows.start, window_c.lo + rows.stop)[:, None]
        weight = np.outer(window_c.weights[rows], window_t.weights)
        smallest = np.minimum(np.minimum(x_c, cell.n - x_c), np.minimum(x_t, cell.n - x_t))
        log_rr, se, df = _statistic(
            np.clip(x_c, 1, cell.n - 1), cell.n, np.clip(x_t, 1, cell.n - 1), cell.n
        )
        crit = _critical(df, tail)
        lower_clear, upper_clear = log_rr - crit * se > 0.0, log_rr + crit * se < 0.0
        delta_rejects = (
            lower_clear | upper_clear
            if cell.alternative == "two-sided"
            else lower_clear
            if cell.alternative == "greater"
            else upper_clear
        ) & (smallest >= 1)
        routed = smallest >= floor
        share += float(weight[routed].sum())
        asymptotic += float(weight[delta_rejects].sum())
        asymptotic_part += float(weight[routed & delta_rejects].sum())
        finite_part += float(weight[~routed & (plus | minus)].sum())
    return EnumeratedPower(
        planning_route(cell.n, cell.n, cell.p_c, p_t, tail_alpha=tail, mode="auto"),
        planned.power,
        planned.power_basis,
        share,
        asymptotic,
        asymptotic_part,
        finite_part,
        window_c.omitted + window_t.omitted,
    )


#: Largest gap between a dense plan's closed-form power and the pipeline's exact rejection
#: probability: the repository's absolute tolerance for a normal-approximation power
#: (``tests.mc.scientific_delta`` at its ceiling).
DENSE_AGREEMENT = 0.005

#: Sparsest expected count of a bound cell's design, in units of ``dense_min_count``: below
#: the boundary, across it, and above it.
BOUND_FACTORS = (0.5, 0.8, 0.9, 1.0, 1.04, 1.1, 1.25, 1.5, 2.0, 3.0)


def bound_cells() -> tuple[MirrorCell, ...]:
    """Designs spanning the route boundary at each production tail of the mirror."""
    out: list[MirrorCell] = []
    for alpha, alternative in ((0.05, "two-sided"), (0.1, "greater"), (0.2, "two-sided")):
        tail = alpha / 2.0 if alternative == "two-sided" else alpha
        m = dense_min_count(tail)
        for p_c, lift in (
            (0.3, 0.05),
            (0.15, 0.1),
            (0.6, 0.04),
            (0.3, 0.1),
            (0.15, 0.3),
            (0.01, 0.3),
            (0.002, 0.5),
            (0.5, 0.06),
        ):
            for factor in BOUND_FACTORS:
                n = math.ceil(factor * m / min(p_c, 1.0 - p_c))
                out.append(MirrorCell("bound", n, p_c, lift, alpha, alternative))
    return tuple(out)


def bound(*, workers: int, out: Path | None = None) -> int:
    """Planned power against the pipeline's exact rejection probability at every design of
    ``bound_cells``: a sparse plan equal to it up to the replay's omitted mass, a borderline
    plan no larger than it (the planning bound), and a dense plan, whose closed form is an
    approximation rather than a bound, within ``DENSE_AGREEMENT`` of it. Each design is appended
    to ``out`` as it completes, and designs already in ``out`` are not recomputed, so an
    interrupted run resumes."""
    designs = bound_cells()
    results: dict[int, EnumeratedPower] = {}
    if out is not None and out.exists():
        for line in out.read_text().splitlines():
            record = json.loads(line)
            results[record["design"]] = EnumeratedPower(**record["power"])
    pending = [index for index in range(len(designs)) if index not in results]
    jobs = [designs[index] for index in pending]
    computed = (
        _pool(workers).imap(enumerated_power, jobs) if workers > 1 else map(enumerated_power, jobs)
    )
    for index, power in zip(pending, computed, strict=True):
        results[index] = power
        if out is not None:
            with out.open("a") as handle:
                handle.write(json.dumps({"design": index, "power": asdict(power)}) + "\n")
    failed = 0
    margins: dict[str, list[float]] = {}
    for index, cell in enumerate(designs):
        power = results[index]
        slack = 1e-9 + power.omitted
        ok = (
            abs(power.margin) <= DENSE_AGREEMENT + slack
            if power.route == "dense"
            else abs(power.margin) <= slack
            if power.route == "sparse"
            else power.margin >= -slack
        )
        failed += not ok
        margins.setdefault(power.route, []).append(power.margin)
        print(
            f"{power.route:<10} n={cell.n:>8} p_c={cell.p_c:<5} lift={cell.lift:<5} "
            f"alpha={cell.alpha:<5} {cell.alternative:<9} share={power.asymptotic_share:.4f} "
            f"planned={power.planned:.6f} ({power.basis}) asym={power.asymptotic:.6f} "
            f"asym_part={power.asymptotic_part:.6f} finite_part={power.finite_part:.6f} "
            f"hybrid={power.hybrid:.6f} margin={power.margin:+.6f} "
            f"{'ok' if ok else 'FAIL'}",
            flush=True,
        )
    for route, values in sorted(margins.items()):
        print(
            f"{route:<10} designs={len(values):>4} least margin={min(values):+.6f} "
            f"greatest margin={max(values):+.6f}"
        )
    return 1 if failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    select_parser = sub.add_parser("select", help="measure the ladder and report required counts")
    select_parser.add_argument("--out", type=Path, default=None)
    select_parser.add_argument("--workers", type=int, default=1)
    select_parser.add_argument("--start", type=float, default=10.0)
    select_parser.add_argument("--stop", type=float, default=40_000.0)
    select_parser.add_argument("--tails", type=float, nargs="+", default=list(TAILS))
    required_parser = sub.add_parser("required", help="required counts from select's JSONL files")
    required_parser.add_argument("paths", type=Path, nargs="+")
    verify_parser = sub.add_parser("verify", help="check the shipped threshold at every tail")
    verify_parser.add_argument("--workers", type=int, default=1)
    verify_parser.add_argument("--tails", type=float, nargs="+", default=list(TAILS))
    sub.add_parser("conformance", help="compare the vectorised decisions with estimate_lift")
    hybrid_parser = sub.add_parser("hybrid", help="pipeline noncoverage at the boundary")
    hybrid_parser.add_argument("--workers", type=int, default=1)
    hybrid_parser.add_argument("--tails", type=float, nargs="+", default=list(TAILS))
    hybrid_parser.add_argument("--replicate-tails", type=float, nargs="*", default=[0.1, 0.05])
    hybrid_parser.add_argument("--count", type=int, default=3)
    bound_parser = sub.add_parser("bound", help="planned power against the exact pipeline power")
    bound_parser.add_argument("--workers", type=int, default=1)
    bound_parser.add_argument("--out", type=Path, default=None)
    mirror_parser = sub.add_parser("mirror", help="planned power against the production route")
    mirror_parser.add_argument("--reps", type=int, default=20_000)
    mirror_parser.add_argument("--workers", type=int, default=1)
    mirror_parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    if args.command == "select":
        return select(
            args.out, workers=args.workers, start=args.start, stop=args.stop, tails=args.tails
        )
    if args.command == "required":
        return required(read_rows(args.paths))
    if args.command == "verify":
        return verify(workers=args.workers, tails=args.tails)
    if args.command == "conformance":
        gap, compared, interpolation = conformance()
        print(
            f"largest relative gap to the production interval over {compared} count pairs: "
            f"{gap:.3g}; interpolated critical value: {interpolation:.3g}"
        )
        sets, edge, hard = finite_conformance()
        print(
            f"finite-sample misses at {sets} count pairs on the decision's edge: {edge} differ "
            f"from the production set within its endpoint resolution, {hard} beyond it"
        )
        return 0 if gap < 1e-9 and interpolation < _CRITICAL_AGREEMENT and not hard else 1
    if args.command == "mirror":
        return mirror(args.reps, args.seed, workers=args.workers)
    if args.command == "bound":
        return bound(workers=args.workers, out=args.out)
    return hybrid(
        args.tails, workers=args.workers, count=args.count, replicate_tails=args.replicate_tails
    )


if __name__ == "__main__":
    sys.exit(main())
