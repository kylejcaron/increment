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
excess is within tolerance at ``m`` and at every larger measured step, with every step of
``m``'s own ladder through ``1.25 * m`` measured; the shipped law is checked against it by
``verify``. ``wald`` is the stricter of the two and the
one the threshold follows: a draw the rule sends to the finite-sample route cannot make the
combined interval worse than the finite-sample route's own, which is valid, but the combined
noncoverage is only known from the finite-sample route's behaviour in the routed-away region,
which ``hybrid`` measures through the production route.

A scan measures the steps of ``ladder(start, stop)`` and writes that ladder beside each step,
so the steps a window needs are exactly the rungs of its own ladder: a row of another start's
ladder stands in for none of them, while a rung any scan measured counts. A file written
before steps named their ladder is placed on the ladder its own steps determine (every start
whose ladder opens with them) and refused when no ladder does; a rung past its last step
counts only where every such start agrees on it.

The vectorised delta-method interval (``delta_statistic``, ``critical_values`` and
``delta_log_bounds``) measures the asymptotic formula over a whole lattice: the ``wald`` and
``routed`` noncoverage tables, and the ranking of the cells ``hybrid`` examines.
``conformance`` checks it against ``estimate_lift`` on sampled count pairs before any table is
trusted. It is a measurement and decides no pair of an exact sum: ``hybrid_noncoverage`` and
``bound`` ask the runtime's own row (``conversion_delta.production_decision``), one routed pair
at a time, ``hybrid_noncoverage`` against the cell's true lift.

Planning is validated against the same production route. ``bound`` compares the plan of
dense, sparse and borderline designs with the production pipeline's exact rejection
probability, summed over the count lattice, at the enumeration's own numerical error: a
dense plan's closed form to within ``DENSE_AGREEMENT``, any other plan's enclosure meeting the
pipeline's interval with no further allowance. A checkpoint records each design's
enumeration beside the runtime it was summed under, and its plan beside the planner model, so
a resumed run keeps an enumeration of this runtime, plans again under a retired model, and
refuses what names neither (a ``CheckpointError``: its code names the reason, its context the
line and the next action). ``mirror`` compares the plan with the simulated rejection rate of
``estimate_lift`` (``tests.estimation._conversion_counts.runtime_rejection_rate``).

Every lattice sum streams over blocks of control counts (``_BLOCK_CELLS`` cells each), so a
worker's footprint stays near a gibibyte whatever the arm sizes: the largest boundary cell has
a count lattice of tens of millions of cells.

    python -m calibration.conversion_route select --out /tmp/route-scan.jsonl
    python -m calibration.conversion_route required /tmp/route-scan.jsonl
    python -m calibration.conversion_route verify
    python -m calibration.conversion_route conformance
    python -m calibration.conversion_route hybrid --workers 4 --tails 0.025 0.01
    python -m calibration.conversion_route bound --workers 4 --out /tmp/route-bound.jsonl
    python -m calibration.conversion_route mirror --reps 3000 --workers 4
"""

from __future__ import annotations

import argparse
import atexit
import itertools
import json
import math
import multiprocessing
import multiprocessing.pool
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NoReturn, cast, get_args, get_type_hints

import numpy as np
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t

from increment.errors import CodedError, raiser, refusals
from increment.estimation.conversion_route import dense_min_count, routed_share
from tests.estimation._conversion_counts import lift_row, runtime_rejection_rate
from tests.mc import (
    scientific_delta,
)

if TYPE_CHECKING:
    from increment.power._binomial import BinomialDecision, _Window

#: Significance levels of the production requests: each is read two-sided, at one-sided tail
#: ``alpha / 2``, and directionally (``greater`` or ``less``), at tail ``alpha``.
PRODUCTION_ALPHAS = (0.001, 0.01, 0.05, 0.1)
#: The one-sided tails those requests produce.
TAILS = (0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
Alternative = Literal["two-sided", "greater", "less"]


def _tail(alpha: float, alternative: str) -> float:
    """The one-sided level a request at ``alpha`` tests at."""
    return alpha / 2.0 if alternative == "two-sided" else alpha


def production_requests(tail: float) -> tuple[tuple[float, Alternative, int], ...]:
    """``(alpha, alternative, sign)`` of the production requests whose one-sided tail is ``tail``,
    each with the sign of the lift it is tested against: a two-sided request at ``2 * tail``
    against a rise, and a directional request at ``tail`` in both directions (``greater``
    against a rise, ``less`` against a fall). A tail two requests share has both forms."""
    requests: list[tuple[float, Alternative, int]] = []
    if 2.0 * tail in PRODUCTION_ALPHAS:
        requests.append((2.0 * tail, "two-sided", 1))
    if tail in PRODUCTION_ALPHAS:
        requests.extend([(tail, "greater", 1), (tail, "less", -1)])
    return tuple(requests)


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

    @property
    def lift(self) -> float:
        """The true relative lift."""
        return self.p_t / self.p_c - 1.0


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


#: Degree-of-freedom floor and node count of the Chebyshev interpolation in ``1 / df`` that
#: stands in for a per-point ``t`` quantile in the vectorised measurements below (they decide
#: no rejection); above the floor the interpolant agrees with ``t.isf`` to
#: ``CRITICAL_AGREEMENT`` over a lattice's narrow range.
INTERPOLATION_DF_FLOOR = 30.0
INTERPOLATION_NODES = 24
CRITICAL_AGREEMENT = 1e-12
_INTERPOLATION_MIN_POINTS = 4096


def delta_statistic(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(log_rr, se, df)`` of the delta-method route for count arrays with ``0 < x < n``."""
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


def critical_values(df: np.ndarray, tail: float) -> np.ndarray:
    """Student ``t`` upper-tail critical values at ``df``."""
    finite = df[np.isfinite(df)]
    if finite.size < _INTERPOLATION_MIN_POINTS or finite.min() < INTERPOLATION_DF_FLOOR:
        return _student_t.isf(tail, df)
    lo, hi = 1.0 / finite.max(), 1.0 / finite.min()
    nodes = np.cos(np.pi * (np.arange(INTERPOLATION_NODES) + 0.5) / INTERPOLATION_NODES)
    inverse = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
    coefficients = np.polynomial.chebyshev.chebfit(
        nodes, _student_t.isf(tail, 1.0 / inverse), INTERPOLATION_NODES - 1
    )
    with np.errstate(all="ignore"):
        position = (1.0 / df - 0.5 * (lo + hi)) * (2.0 / (hi - lo)) if hi > lo else 0.0 * df
    return np.polynomial.chebyshev.chebval(position, coefficients)


def delta_log_bounds(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int, tail: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log risk-ratio interval bounds the production delta-method route reports at one-sided
    ``tail``, for count arrays with ``0 < x < n``."""
    log_rr, se, df = delta_statistic(x_c, n_c, x_t, n_t)
    crit = critical_values(df, tail)
    return log_rr - crit * se, log_rr + crit * se


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


def boundary_noncoverage(
    cell: Cell, tails: Sequence[float], threshold: dict[float, int] | int
) -> dict[float, tuple[float, float, float, float]]:
    """Per tail: ``(wald_lower, wald_upper, routed_lower, routed_upper)`` noncoverage of the
    vectorised delta-method interval at ``cell`` (a measurement of the formula, which
    ``conformance`` checks, not the runtime's decision of any pair), applied to every draw with
    positive counts (``wald``) and to the draws whose four counts reach ``threshold``
    (``routed``). ``threshold`` is one count, or a count per tail."""
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
        log_rr, se, df = delta_statistic(grid_c, cell.n_c, x_t[None, :], cell.n_t)
        for tail in tails:
            crit = critical_values(df, tail)
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


def _rungs(start: float) -> Iterator[tuple[float, int]]:
    """The ladder from ``start`` without end: each step's geometric value, built by repeated
    multiplication, and that value rounded."""
    value = start
    while True:
        yield value, round(value)
        value *= LADDER_STEP


def ladder(start: float = 10.0, stop: float = 40_000.0) -> list[int]:
    """The steps from ``start`` whose geometric value is below ``stop``."""
    return [step for _, step in itertools.takewhile(lambda rung: rung[0] < stop, _rungs(start))]


@dataclass(frozen=True, slots=True)
class Anchor:
    """The starts a scan's ladder may have had: its steps are those of ``ladder(start, stop)``
    for every ``start`` in ``[low, high]``. A scan records its one start (``low == high``). Steps
    read from records that name no start have every start whose ladder opens with them
    (``anchor_of``, with ``stop`` unknown and therefore infinite); the later steps of such a
    ladder are the ones every one of those starts gives."""

    low: float
    high: float
    stop: float

    def __post_init__(self) -> None:
        if not (0.0 < self.low <= self.high < math.inf and self.low < self.stop):
            raise ValueError(
                f"a ladder needs 0 < start < stop; got starts [{self.low}, {self.high}], "
                f"stop {self.stop}"
            )

    @classmethod
    def recorded(cls, start: float, stop: float) -> Anchor:
        return cls(start, start, stop)

    def window(self, step: int, limit: float) -> tuple[int, ...] | None:
        """The steps of the ladder from ``step`` through ``limit``, or ``None`` when ``step`` is
        not one of its steps or its starts give different steps within that range. Each rung is
        nondecreasing in the start, so the ladders of the lowest and the highest start bound
        those of every start between."""
        steps: list[int] = []
        for (_, low), (_, high) in zip(_rungs(self.low), _rungs(self.high), strict=False):
            if low > limit:
                break
            if high < step:
                continue
            if low != high:
                return None
            steps.append(low)
        return tuple(steps) if steps and steps[0] == step else None


#: Relative margin around the starts the rounding of each step allows, within which the starts
#: whose ladder opens with given steps are searched for.
_ANCHOR_SLACK = 1e-9


def anchor_of(steps: Sequence[int]) -> Anchor | None:
    """The starts whose ladder opens with exactly ``steps`` (the first ``len(steps)`` rungs, in
    order), or ``None`` when no start's does: the steps skip a rung, repeat one, or come from
    scans of different starts. A rung within half a unit of ``start * LADDER_STEP ** k`` bounds
    the start, and the set of starts that reproduce every step is an interval because each
    rung is nondecreasing in the start, so its ends are found by bisection on the ladder itself."""
    wanted = list(steps)
    if not wanted:
        return None

    def opens(start: float) -> bool:
        return [step for _, step in itertools.islice(_rungs(start), len(wanted))] == wanted

    lowest = max((step - 0.5) / LADDER_STEP**k for k, step in enumerate(wanted))
    highest = min((step + 0.5) / LADDER_STEP**k for k, step in enumerate(wanted))
    outer_low, outer_high = lowest * (1.0 - _ANCHOR_SLACK), highest * (1.0 + _ANCHOR_SLACK)
    inside = next(
        (
            start
            for start in (0.5 * (lowest + highest), lowest, highest)
            if outer_low <= start <= outer_high and opens(start)
        ),
        None,
    )
    if inside is None or outer_low <= 0.0:
        return None

    def edge(outside: float) -> float:
        near = inside
        while not opens(outside):
            middle = 0.5 * (outside + near)
            if middle in (outside, near):
                return near
            outside, near = (outside, middle) if opens(middle) else (middle, near)
        return outside

    return Anchor(edge(outer_low), edge(outer_high), math.inf)


@dataclass(frozen=True, slots=True)
class Measured:
    """A tail's worst excess at one ladder step, and the ladder the scan took that step from."""

    excess: Excess
    anchor: Anchor


Rows = dict[int, dict[float, Measured]]


def _unmeasured(rows: Rows, tail: float, step: int) -> tuple[int, ...] | None:
    """The steps of ``step``'s own ladder through ``MARGIN * step`` that ``rows`` lacks at
    ``tail``: a step counts only as the rung of the ladder it was scheduled on, so a row of
    another ladder's rung stands in for none. ``None`` when that ladder is not determined
    across the range."""
    window = rows[step][tail].anchor.window(step, MARGIN * step)
    return None if window is None else tuple(s for s in window if tail not in rows.get(s, {}))


def _passing_from(rows: Rows, tail: float, key: str) -> list[int]:
    """Measured steps at ``tail`` that are within tolerance and below only steps that are."""
    ms = sorted(m for m in rows if tail in rows[m])
    passing = [getattr(rows[m][tail].excess, key) <= 1.0 for m in ms]
    return [m for index, m in enumerate(ms) if all(passing[index:])]


def required_count(rows: Rows, tail: float, *, key: str = "wald") -> int | None:
    """Smallest measured ``m`` whose excess is within tolerance at ``m`` and at every larger
    measured step, with every rung of ``m``'s own ladder through ``MARGIN * m`` measured at that
    tail; ``None`` when no candidate has that coverage."""
    return next(
        (m for m in _passing_from(rows, tail, key) if _unmeasured(rows, tail, m) == ()), None
    )


def unmet(rows: Rows, tail: float, *, key: str = "wald") -> str:
    """Why ``required_count`` finds no requirement at ``tail``."""
    if not any(tail in by_tail for by_tail in rows.values()):
        return "no step was measured at this tail"
    candidates = _passing_from(rows, tail, key)
    if not candidates:
        return "no measured step is within tolerance with every larger measured step"
    gaps = _unmeasured(rows, tail, candidates[0])
    if gaps is None:
        return (
            f"step {candidates[0]} passes, but the steps recorded do not determine the rungs "
            f"of its ladder through {MARGIN * candidates[0]:g}"
        )
    return (
        f"step {candidates[0]} passes, but rungs {list(gaps)} of its ladder through "
        f"{MARGIN * candidates[0]:g} were not measured"
    )


def select(
    out: Path | None, *, workers: int, start: float, stop: float, tails: Sequence[float] = TAILS
) -> int:
    """Measure every ladder step and report the required count per tail and the shipped law.
    Each step is appended to ``out`` as it completes, with the ladder it belongs to, so an
    interrupted run keeps its steps and their ladder."""
    anchor = Anchor.recorded(start, stop)
    rows: Rows = {}
    if out is not None:
        out.write_text("")
    for m in ladder(start, stop):
        measured = worst_excess(m, tails, workers=workers)
        rows[m] = {tail: Measured(excess, anchor) for tail, excess in measured.items()}
        lines = [
            {
                "m": m,
                "tail": tail,
                "delta": scientific_delta(tail),
                "wald_excess": excess.wald,
                "routed_excess": excess.routed,
                "cell": asdict(excess.cell) if excess.cell else None,
                "ladder": {"start": start, "stop": stop},
            }
            for tail, excess in measured.items()
        ]
        if out is not None:
            with out.open("a") as handle:
                handle.write("".join(json.dumps(line) + "\n" for line in lines))
        print(
            f"m={m:>6}  "
            + "  ".join(
                f"{tail:g}:{measured[tail].wald:6.2f}/{measured[tail].routed:6.2f}"
                for tail in tails
            ),
            flush=True,
        )
    return required(rows, tails)


def required(rows: Rows, tails: Sequence[float] = TAILS) -> int:
    """Print the required count per tail beside the shipped law, with the reason for each
    that is not established."""
    for tail in tails:
        found = {key: required_count(rows, tail, key=key) for key in ("wald", "routed")}
        reasons = {key: unmet(rows, tail, key=key) for key, count in found.items() if count is None}
        report: dict[str, object] = {
            "tail": tail,
            "z": float(_norm.isf(tail)),
            "required_wald": found["wald"],
            "required_routed": found["routed"],
            "shipped": dense_min_count(tail),
        }
        print(json.dumps(report | ({"unmet": reasons} if reasons else {})))
    return 0


class ScanError(ValueError):
    """A ``select`` file that cannot be read as the steps of ladders."""


def _scan_records(path: Path) -> list[tuple[int, dict]]:
    """``(line number, record)`` of the step records in ``path``."""
    records: list[tuple[int, dict]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if "m" not in record:
                continue
            int(record["m"]), float(record["tail"]), float(record["wald_excess"])
            float(record["routed_excess"]), record["cell"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ScanError(f"{path}:{number}: not a scan record") from error
        records.append((number, record))
    return records


def _recorded_anchor(where: str, record: dict) -> Anchor:
    """The ladder a record names, which must contain the record's own step."""
    try:
        anchor = Anchor.recorded(float(record["ladder"]["start"]), float(record["ladder"]["stop"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ScanError(
            f"{where}: the ladder {record['ladder']!r} is not a start and a stop"
        ) from error
    if record["m"] not in ladder(anchor.low, anchor.stop):
        raise ScanError(
            f"{where}: step {record['m']} is not a step of the ladder it names, "
            f"ladder({anchor.low:g}, {anchor.stop:g})"
        )
    return anchor


def _derived_anchor(path: Path, records: list[tuple[int, dict]]) -> Anchor | None:
    """The ladder of the records that name none, from the steps they measured in file order:
    one scan writes each step's tails together, so a step recurring after another is a second
    scan. ``None`` when there are no such records; a refusal when no single ladder opens with
    the steps."""
    steps: list[int] = []
    for number, record in records:
        if not steps or steps[-1] != record["m"]:
            if record["m"] in steps:
                raise ScanError(
                    f"{path}:{number}: step {record['m']} recurs after other steps, so the file "
                    "is more than one scan and names no ladder for either"
                )
            steps.append(record["m"])
    if not steps:
        return None
    anchor = anchor_of(steps)
    if anchor is None:
        prefix = max(k for k in range(len(steps) + 1) if k == 0 or anchor_of(steps[:k]))
        raise ScanError(
            f"{path}: no single ladder opens with its {len(steps)} steps ({steps[0]} .. "
            f"{steps[-1]}); the first {prefix} do, and step {steps[prefix]} is not the next "
            "rung of any start's ladder. The file names no ladder, so its steps cannot be "
            "placed on one; split it into one file per scan, or rescan with the ladder recorded"
        )
    return anchor


def read_rows(paths: Sequence[Path]) -> Rows:
    """The ladder rows ``select`` wrote to ``paths``, merged (a later file replaces an earlier
    one at the same ``m`` and tail). A record names the ladder it came from; a file whose
    records name none is placed on the ladder its own steps determine (``anchor_of``) or
    refused."""
    rows: Rows = {}
    for path in paths:
        records = _scan_records(path)
        derived = _derived_anchor(path, [(n, r) for n, r in records if "ladder" not in r])
        for number, record in records:
            anchor = _recorded_anchor(f"{path}:{number}", record) if "ladder" in record else derived
            assert anchor is not None
            cell = Cell(**record["cell"]) if record["cell"] else None
            rows.setdefault(record["m"], {})[record["tail"]] = Measured(
                Excess(
                    record["tail"],
                    record["m"],
                    record["wald_excess"],
                    record["routed_excess"],
                    cell,
                ),
                anchor,
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
    ``delta_log_bounds`` and the delta-method row ``estimate_lift`` returns over ``compared``
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
        lower, upper = delta_log_bounds(np.array([x_c]), n_c, np.array([x_t]), n_t, tail)
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
                float(np.max(np.abs(critical_values(df, tail) / _student_t.isf(tail, df) - 1.0))),
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

    tail = _tail(alpha, alternative)
    return BinomialDecision(
        cell.n_c, cell.n_t, cell.p_t / cell.p_c, nuisance_beta(alpha), tail, alternative
    )


def _count_windows(cell: Cell) -> tuple[_Window, _Window]:
    """The count windows of ``cell``'s two arms: the lattice every pipeline sum runs over."""
    from increment.power._binomial import _window

    return _window(cell.n_c, cell.p_c), _window(cell.n_t, cell.p_t)


def finite_sample_misses(
    cell: Cell, *, alpha: float, alternative: Literal["two-sided", "greater", "less"]
) -> tuple[np.ndarray, np.ndarray, _Window, _Window]:
    """``(plus, minus, window_c, window_t)``: over the count lattice of ``cell``, where the
    finite-sample set misses the true ratio. Its miss is the rejection of the test at that
    ratio (``_finite_blocks``): a ``plus`` rejection puts the set wholly above the truth, a
    ``minus`` one wholly below. The whole lattice is held, so this is for cells whose windows
    are small."""
    window_c, window_t = _count_windows(cell)
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


def _delta_decisions(
    x_c: np.ndarray,
    n_c: int,
    x_t: np.ndarray,
    n_t: int,
    routed: np.ndarray,
    *,
    tail: float,
    alternative: Alternative,
    null_lift: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(plus, minus)``: where the runtime's own row (`production_decision`, one call per pair)
    puts the interval of a routed pair wholly above or below ``null_lift``; every other pair is
    ``False``. ``x_c`` is a column and ``x_t`` a row of counts, ``routed`` their grid. Slow by
    design: nothing restates the runtime's calculation."""
    from increment.estimation.conversion_delta import production_decision

    plus = np.zeros(routed.shape, bool)
    minus = np.zeros(routed.shape, bool)
    for i, j in np.argwhere(routed):
        plus[i, j], minus[i, j] = production_decision(
            int(x_c[i, 0]),
            n_c,
            int(x_t[0, j]),
            n_t,
            tail=tail,
            alternative=alternative,
            null_lift=null_lift,
        )
    return plus, minus


def hybrid_noncoverage(
    cell: Cell,
    *,
    alpha: float,
    threshold: int,
    alternative: Alternative = "two-sided",
) -> HybridNoncoverage:
    """Exact noncoverage of the pipeline at ``cell``, summed over the joint binomial law of its
    counts: a pair whose four counts reach ``threshold`` is decided by the runtime's own
    delta-method row against the cell's true lift (``_delta_decisions``), every other pair by
    the finite-sample set (``_finite_blocks``, checked against ``estimate_lift`` by
    ``finite_conformance``). A side the alternative does not read never misses. Counts are not
    drawn, so there is no sampling error; the lattice omits at most the windows' ``omitted``
    mass. The sums stream over blocks of control counts, so the footprint does not grow with
    the windows."""
    tail = _tail(alpha, alternative)
    window_c, window_t = _count_windows(cell)
    key = _decision_key(cell, alpha=alpha, alternative=alternative)
    x_t = np.arange(window_t.lo, window_t.hi + 1)[None, :]
    lower = upper = share = 0.0
    for rows, plus, minus in _finite_blocks(key, window_c, window_t, threshold):
        x_c = np.arange(window_c.lo + rows.start, window_c.lo + rows.stop)[:, None]
        weight = np.outer(window_c.weights[rows], window_t.weights)
        smallest = np.minimum(np.minimum(x_c, cell.n_c - x_c), np.minimum(x_t, cell.n_t - x_t))
        routed = smallest >= threshold
        delta_plus, delta_minus = _delta_decisions(
            x_c,
            cell.n_c,
            x_t,
            cell.n_t,
            routed,
            tail=tail,
            alternative=alternative,
            null_lift=cell.lift,
        )
        lower += float(weight[np.where(routed, delta_plus, plus)].sum())
        upper += float(weight[np.where(routed, delta_minus, minus)].sum())
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
        lift = cell.lift
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
    finite-sample side has the least room. The vectorised measurement ranks the cells and
    decides none of them: ``hybrid_noncoverage`` decides every pair of the chosen cells through
    the runtime."""
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
    directional at ``alpha = tail`` (``greater`` and, at a tail a production alpha reaches
    directionally, ``less``). Then ``replicate_tails``: the pipeline run through
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
                    if tail in PRODUCTION_ALPHAS:
                        jobs.append((cell, tail, shipped, "less"))
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
    cell, reps, seed = args
    plan = plan_design(cell)
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
    if plan.closed_form:
        ok = abs(plan.planned - rate) <= 4 * se + 0.003
    else:
        ok = plan.certified and plan.lower <= rate + 4 * se and rate - 4 * se <= plan.upper
    return (
        f"{cell.label:<10} route={plan.route:<10} n={cell.n:>8} p_c={cell.p_c:<5} "
        f"lift={cell.lift:<5} alpha={cell.alpha:<5} {cell.alternative:<9} "
        f"planned={plan.planned:.4f} ({plan.basis}) simulated={rate:.4f} "
        f"se={se:.4f} asym_share={share:.2f} {'ok' if ok else 'FAIL'}",
        ok,
    )


def mirror(reps: int, seed: int, *, workers: int = 1) -> int:
    """Planned power against the simulated production rejection rate, per design: a dense plan
    within ``4 * se + 0.003``; any other plan's enclosure, widened by ``4 * se``, holds the
    simulated rate."""
    jobs = [(cell, reps, seed) for cell in mirror_cells()]
    results = _pool(workers).imap(_mirror_job, jobs) if workers > 1 else map(_mirror_job, jobs)
    failed = 0
    for line, ok in results:
        failed += not ok
        print(line, flush=True)
    return 1 if failed else 0


def _next_up(x: float) -> float:
    return math.nextafter(x, math.inf)


def _next_down(x: float) -> float:
    return math.nextafter(x, -math.inf)


@dataclass(frozen=True, slots=True)
class Enumeration:
    """A design's exact rejection probabilities, summed over the joint binomial law of the
    counts, with the runtime they were summed under. The production pipeline decides each draw
    by the route its counts select, so its rejection probability ``hybrid`` is the part the
    delta method decides (``asymptotic_part``, the routed draws the runtime's own row rejects)
    plus the part the finite-sample test decides (``finite_part``, the draws the rule keeps on
    it). ``construction`` is the finite-sample construction and ``floor`` the routing
    threshold of that runtime. ``omitted`` bounds the mass outside the windows and
    ``inflation`` (at least one) the numerical error of the sums, so the pipeline's rejection
    probability lies in ``[lower, upper]``."""

    construction: str
    floor: int
    asymptotic_share: float
    asymptotic_part: float
    finite_part: float
    omitted: float
    inflation: float

    @property
    def hybrid(self) -> float:
        """The pipeline's rejection probability over the retained cells, as summed."""
        return self.asymptotic_part + self.finite_part

    @property
    def lower(self) -> float:
        """No larger than the exact mass of the retained cells the pipeline rejects."""
        return max(0.0, _next_down(self.hybrid / self.inflation))

    @property
    def upper(self) -> float:
        """No smaller than the pipeline's rejection probability: the retained mass at the most
        the summation allows, plus the omitted mass."""
        return min(1.0, _next_up(_next_up(self.hybrid * self.inflation) + self.omitted))


@dataclass(frozen=True, slots=True)
class Plan:
    """What the planner reports for a design, under the model that produced it. ``planned`` is
    the plan's power, the certified lower figure of a hybrid plan; ``lower`` and ``upper``
    enclose the pipeline's rejection probability and ``ambiguous`` is the mass of the counts the
    plan could not decide. A ``closed_form`` plan is the delta-method model and encloses only
    itself; a plan that is not ``certified`` is a heuristic with no claim on the pipeline."""

    model: str
    route: str
    basis: str
    planned: float
    lower: float
    upper: float
    ambiguous: float
    closed_form: bool
    certified: bool


@dataclass(frozen=True, slots=True)
class EnumeratedPower:
    """A design's plan against the pipeline's exact rejection probability."""

    plan: Plan
    enumeration: Enumeration

    @property
    def margin(self) -> float:
        """How far the planned power sits below the pipeline's rejection probability."""
        return self.enumeration.hybrid - self.plan.planned


#: Planner models a checkpoint may still name. Resuming keeps their designs' enumerations and
#: plans each design again under the current model. ``borderline_minimum`` is the smaller of the
#: replay and the closed form that preceded ``increment.power.core.BINOMIAL_PLANNING_MODEL``.
RETIRED_PLANNER_MODELS = frozenset({"borderline_minimum"})


def plan_design(cell: MirrorCell) -> Plan:
    """The planner's report for ``cell``'s design under the current model: the plan's power,
    basis and enclosure of the pipeline's rejection probability, and its route."""
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.estimation.conversion_route import planning_route
    from increment.power import Baseline
    from increment.power.core import BINOMIAL_PLANNING_MODEL, planned_enclosure

    procedure = ArmPlanningProcedure.standard(
        "conversion", alpha=cell.alpha, alternative=cell.alternative
    )
    route = planning_route(
        cell.n,
        cell.n,
        cell.p_c,
        cell.p_c * (1.0 + cell.lift),
        tail_alpha=procedure.compiled_tail_alpha,
        mode="auto",
    )
    enclosure = planned_enclosure(cell.n, cell.lift, Baseline.from_proportion(cell.p_c), procedure)
    return Plan(
        BINOMIAL_PLANNING_MODEL,
        route,
        enclosure.basis,
        enclosure.reported,
        enclosure.lower,
        enclosure.upper,
        enclosure.ambiguous,
        enclosure.closed_form,
        enclosure.certified,
    )


@dataclass(frozen=True, slots=True)
class _DesignLattice:
    """The runtime's decision of a design and the count windows it is summed over."""

    tail: float
    floor: int
    key: BinomialDecision
    window_c: _Window
    window_t: _Window
    x_t: np.ndarray


def _design_lattice(cell: MirrorCell) -> _DesignLattice:
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.power._binomial import _window
    from increment.power.core import _binomial_key

    procedure = ArmPlanningProcedure.standard(
        "conversion", alpha=cell.alpha, alternative=cell.alternative
    )
    tail = procedure.compiled_tail_alpha
    p_t = cell.p_c * (1.0 + cell.lift)
    key = _binomial_key(procedure, cell.n, cell.n)
    window_c, window_t = _window(cell.n, cell.p_c), _window(cell.n, p_t)
    x_t = np.arange(window_t.lo, window_t.hi + 1)[None, :]
    floor = dense_min_count(tail)
    return _DesignLattice(tail, floor, key, window_c, window_t, x_t)


def _delta_rejects(
    cell: MirrorCell, lattice: _DesignLattice, x_c: np.ndarray, routed: np.ndarray
) -> np.ndarray:
    """Where the runtime's own row rejects a routed pair at the null (``null_lift = 0``); every
    other pair is ``False``."""
    plus, minus = _delta_decisions(
        x_c,
        cell.n,
        lattice.x_t,
        cell.n,
        routed,
        tail=lattice.tail,
        alternative=cell.alternative,
        null_lift=0.0,
    )
    return plus | minus


@dataclass(frozen=True, slots=True)
class _Block:
    """One block of control counts of a lattice: each pair's joint weight, whether the rule
    routes it to the delta method, where the runtime's row rejects the routed pairs, and the
    finite-sample replay's ``plus`` and ``minus`` rejections of the pairs the route keeps."""

    weight: np.ndarray
    routed: np.ndarray
    delta_routed: np.ndarray
    plus: np.ndarray
    minus: np.ndarray


def _decided_blocks(cell: MirrorCell, lattice: _DesignLattice) -> Iterator[_Block]:
    """The blocks of ``lattice``, streaming over control counts. The finite-sample decision is
    the runtime's decision replayed on the ``exact`` rejection geometry whatever the cell
    count, never the surrogate the planner substitutes above its budget; only the pairs the
    finite-sample route decides are replayed (``_finite_blocks``), so ``plus`` and ``minus``
    are read where ``routed`` is false."""
    floor, key = lattice.floor, lattice.key
    window_c, window_t, x_t = lattice.window_c, lattice.window_t, lattice.x_t
    for rows, plus, minus in _finite_blocks(key, window_c, window_t, floor):
        x_c = np.arange(window_c.lo + rows.start, window_c.lo + rows.stop)[:, None]
        weight = np.outer(window_c.weights[rows], window_t.weights)
        smallest = np.minimum(np.minimum(x_c, cell.n - x_c), np.minimum(x_t, cell.n - x_t))
        routed = smallest >= floor
        yield _Block(weight, routed, _delta_rejects(cell, lattice, x_c, routed), plus, minus)


def enumerate_design(cell: MirrorCell) -> Enumeration:
    """Exact rejection probabilities of ``cell``'s design by enumerating its count lattice.

    A pair is decided by the delta method iff its four counts reach ``dense_min_count``, so
    the sum over the pairs (``_decided_blocks``) is the pipeline's rejection probability, not a
    bound on it. Pairs with a zero count are always finite-sample. The sums stream over blocks
    of control counts.

    Every pmf weight is within its relative allowance, the products round once, and the
    nonnegative cell terms are summed in some order, whose relative error is at most
    ``gamma_k`` for ``k`` additions (the cells, one per block, and the two parts' sum); these
    compose into ``inflation`` as the planner's own enclosure does.
    """
    from increment.estimation.results import BINOMIAL_METHOD
    from increment.power._binomial import _UNIT_ROUNDOFF, _compounded, _inflation

    lattice = _design_lattice(cell)
    share = asymptotic_part = finite_part = 0.0
    blocks = 0
    for block in _decided_blocks(cell, lattice):
        weight, routed, plus, minus = block.weight, block.routed, block.plus, block.minus
        blocks += 1
        share += float(weight[routed].sum())
        asymptotic_part += float(weight[block.delta_routed].sum())
        finite_part += float(weight[~routed & (plus | minus)].sum())
    return Enumeration(
        BINOMIAL_METHOD,
        lattice.floor,
        share,
        asymptotic_part,
        finite_part,
        _next_up(lattice.window_c.omitted + lattice.window_t.omitted),
        _inflation(
            lattice.window_c.error,
            lattice.window_t.error,
            _UNIT_ROUNDOFF,
            _compounded(lattice.window_c.size * lattice.window_t.size + blocks + 2),
        ),
    )


def enumerated_power(cell: MirrorCell) -> EnumeratedPower:
    """``cell``'s plan against the pipeline's exact rejection probability."""
    return EnumeratedPower(plan_design(cell), enumerate_design(cell))


#: Largest gap between a dense plan's closed-form power and the pipeline's exact rejection
#: probability: the repository's ceiling for a normal approximation
#: (``tests.mc.scientific_delta``).
DENSE_AGREEMENT = 0.005

#: Sparsest expected count of a bound cell's design, in units of ``dense_min_count``: below
#: the boundary, across it, and above it.
BOUND_FACTORS = (0.5, 0.8, 0.9, 1.0, 1.04, 1.1, 1.25, 1.5, 2.0, 3.0)

#: ``(alpha, alternative)`` of the positive-lift designs: two-sided and directional levels, whose
#: one-sided tails are 0.025, 0.1 and 0.1.
_BOUND_LEVELS = ((0.05, "two-sided"), (0.1, "greater"), (0.2, "two-sided"))
#: The same levels read against a negative lift, whose directional test is ``less``.
_NEGATIVE_LEVELS = ((0.05, "two-sided"), (0.1, "less"), (0.2, "two-sided"))
#: ``(control rate, relative lift)`` of the original bound grid. Its ``(0.15, 0.3)``,
#: ``(0.01, 0.3)`` and ``(0.002, 0.5)`` shapes are saturated at the boundary, which makes their
#: margin vacuous.
_BOUND_SHAPES = (
    (0.3, 0.05),
    (0.15, 0.1),
    (0.6, 0.04),
    (0.3, 0.1),
    (0.15, 0.3),
    (0.01, 0.3),
    (0.002, 0.5),
    (0.5, 0.06),
)
#: Shapes whose power stays below saturation across the boundary: rare events, where the counts
#: are most skewed, and negative lifts, which exercise the other tail of the interval.
_EXTENDED_SHAPES = ((0.15, 0.06), (0.01, 0.06), (0.002, 0.06))
_EXTENDED_NEGATIVE_SHAPES = ((0.3, -0.05), (0.01, -0.06), (0.6, -0.04))


BoundGrid = Literal["original", "extended", "tails", "window"]

#: One-sided production tails the other grids do not reach at their production requests, read
#: at every ``(alpha, alternative)`` request that produces them (``production_requests``).
_OTHER_TAILS = (0.0005, 0.001, 0.005, 0.01, 0.05)
_OTHER_TAIL_FACTORS = (0.9, 1.0, 1.04, 1.1, 1.25)
#: Control rates of the ``tails`` and ``window`` grids: central, and rare where counts are most
#: skewed.
_TAIL_RATES = (0.3, 0.01)

#: Probabilities that all four counts reach the routing threshold, swept by the ``window`` grid:
#: from just inside the borderline class (``planning_route`` calls a plan sparse at or below
#: ``1e-6`` and dense at or above ``1 - 1e-6``), where the pipeline is almost the finite-sample
#: route, to just short of dense, where it is almost the delta method.
_WINDOW_SHARES = (1e-5, 1e-3, 0.05, 0.3, 0.7, 0.95, 0.999, 1.0 - 1e-5)


def _window_size(p_c: float, p_t: float, tail: float, share: float) -> int:
    """Least arm size at which ``routed_share`` reaches ``share`` at one-sided ``tail``: the
    share rises with the arm size, from nothing at 0.8 times the size that puts
    ``dense_min_count(tail)`` expected counts in the sparser arm to one at 1.4 times it."""
    m = dense_min_count(tail)
    q = min(p_c, p_t, 1.0 - p_c, 1.0 - p_t)
    lo, hi = math.ceil(0.8 * m / q), math.ceil(1.4 * m / q)
    while lo < hi:
        mid = (lo + hi) // 2
        if routed_share(mid, mid, p_c, p_t, tail_alpha=tail) >= share:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _window_cells() -> tuple[MirrorCell, ...]:
    out: list[MirrorCell] = []
    for tail in TAILS:
        m = dense_min_count(tail)
        z = float(_norm.isf(tail))
        for p_c in _TAIL_RATES:
            lift = round(z * math.sqrt(2.0 * (1.0 - p_c) / m), 3)
            for alpha, alternative, sign in production_requests(tail):
                signed = sign * lift
                sizes = sorted(
                    {
                        _window_size(p_c, p_c * (1.0 + signed), tail, share)
                        for share in _WINDOW_SHARES
                    }
                )
                out.extend(MirrorCell("bound", n, p_c, signed, alpha, alternative) for n in sizes)
    return tuple(out)


def bound_cells(grid: BoundGrid = "original") -> tuple[MirrorCell, ...]:
    """Designs spanning the route boundary. ``original`` is the grid of ``_BOUND_SHAPES`` at the
    production tails of the mirror (240 designs); ``extended`` adds the shapes whose power is
    not saturated at the boundary, at positive and negative lifts; ``tails`` covers the other
    production tails with a central and a rare-event shape whose lift puts the sparsest count's
    power near one half (``z * sqrt(2 (1 - p) / m)`` on the log scale, for tail quantile ``z``),
    at five distances across the boundary, at every production request that produces the tail
    (two-sided against a rise, directional in both directions); ``window`` sweeps every
    production tail through the whole borderline class at those shapes and requests, at the arm
    sizes where the probability of routing all four counts to the delta method is each of
    ``_WINDOW_SHARES``."""
    out: list[MirrorCell] = []
    if grid == "window":
        return _window_cells()
    if grid == "tails":
        for tail in _OTHER_TAILS:
            m = dense_min_count(tail)
            z = float(_norm.isf(tail))
            for p_c in _TAIL_RATES:
                lift = round(z * math.sqrt(2.0 * (1.0 - p_c) / m), 3)
                for factor in _OTHER_TAIL_FACTORS:
                    n = math.ceil(factor * m / min(p_c, 1.0 - p_c))
                    for alpha, alternative, sign in production_requests(tail):
                        out.append(MirrorCell("bound", n, p_c, sign * lift, alpha, alternative))
        return tuple(out)
    groups = (
        ((_BOUND_LEVELS, _EXTENDED_SHAPES), (_NEGATIVE_LEVELS, _EXTENDED_NEGATIVE_SHAPES))
        if grid == "extended"
        else ((_BOUND_LEVELS, _BOUND_SHAPES),)
    )
    for levels, shapes in groups:
        for alpha, alternative in levels:
            tail = alpha / 2.0 if alternative == "two-sided" else alpha
            m = dense_min_count(tail)
            for p_c, lift in shapes:
                for factor in BOUND_FACTORS:
                    n = math.ceil(factor * m / min(p_c, 1.0 - p_c))
                    out.append(MirrorCell("bound", n, p_c, lift, alpha, alternative))
    return tuple(out)


def bound_holds(power: EnumeratedPower) -> bool:
    """Whether a design meets its plan's claim, judged at the numerical error of the
    enumeration (the pipeline's rejection probability lies in ``[lower, upper]``) and no
    looser. A closed-form plan is within ``DENSE_AGREEMENT`` of it. Any other plan encloses it:
    the plan's certified lower end does not exceed the pipeline's rejection probability, and the
    pipeline's does not exceed the plan's upper end. A plan that is not certified makes no claim
    to hold."""
    plan, runtime = power.plan, power.enumeration
    if plan.closed_form:
        return runtime.lower - DENSE_AGREEMENT <= plan.planned <= runtime.upper + DENSE_AGREEMENT
    return plan.certified and plan.lower <= runtime.upper and runtime.lower <= plan.upper


def _design_key(cell: MirrorCell) -> list[object]:
    return [cell.n, cell.p_c, cell.lift, cell.alpha, cell.alternative]


class CheckpointError(CodedError):
    """A ``bound`` checkpoint record that cannot be resumed from. ``code`` is the stable reason;
    ``context`` holds the record's ``path`` and ``line``, what the reason names, and the
    ``next_action`` that resolves it: ``recompute_to_new_out`` (enumerate to another ``--out``)
    or ``remove_line`` (drop a line that is no record and resume from the others)."""


_RECOMPUTE = "recompute_to_new_out"
_REMOVE_LINE = "remove_line"

_REFUSALS = refusals(
    CheckpointError,
    {
        "calibration.conversion_route.checkpoint_unreadable": (
            "{path}:{line}: not a bound checkpoint record; remove the line to resume from the "
            "other records",
            ("next_action",),
        ),
        "calibration.conversion_route.checkpoint_design": (
            "{path}:{line}: design {design!r} is not a design key; remove the line to resume "
            "from the other records",
            ("next_action",),
        ),
        "calibration.conversion_route.checkpoint_layout": (
            "{path}:{line}: the {section} is not exactly {expected!r} with the types of this "
            "checkpoint layout (it holds {found!r}), so nothing in the record is reused, "
            "relabelled or enumerated again unnoticed; enumerate to a new --out",
            ("next_action",),
        ),
        "calibration.conversion_route.checkpoint_runtime": (
            "{path}:{line}: the enumeration was summed under {construction} with routing floor "
            "{floor}, and this runtime is {current_construction} with floor {current_floor} at "
            "this design's tail; enumerate to a new --out",
            ("next_action",),
        ),
        "calibration.conversion_route.checkpoint_planner_model": (
            "{path}:{line}: planner model {model!r} is neither {current!r} nor a retired model "
            "this checkpoint can be re-planned from ({retired!r}); enumerate to a new --out",
            ("next_action",),
        ),
    },
)
_raise = raiser(_REFUSALS)


@dataclass(frozen=True, slots=True)
class _Line:
    """The checkpoint line a refusal is about."""

    path: str
    number: int

    def refuse(self, code: str, /, **context: object) -> NoReturn:
        _raise(code, path=self.path, line=self.number, **context)


def _conforms(hint: object, value: object) -> bool:
    """Whether a record's JSON ``value`` is of the annotated type ``hint`` (a number is an int
    or a float but never a bool, and a bool only where a bool is annotated)."""
    options = get_args(hint) or (hint,)
    if value is None:
        return type(None) in options
    return any(
        isinstance(value, (int, float) if option is float else option)
        and (option is bool or not isinstance(value, bool))
        for option in options
        if isinstance(option, type) and option is not type(None)
    )


def _section[T](cls: type[T], value: object, where: _Line, name: str) -> T:
    """``cls`` read from a record's ``name`` section, which must hold exactly its fields, typed."""
    hints = get_type_hints(cls)
    found: tuple[str, ...] | None = None
    if isinstance(value, dict):
        section = {str(key): item for key, item in value.items()}
        if section.keys() == hints.keys() and all(
            _conforms(hint, section[field]) for field, hint in hints.items()
        ):
            return cls(**section)
        found = tuple(sorted(section))
    where.refuse(
        "calibration.conversion_route.checkpoint_layout",
        section=name,
        found=found,
        expected=tuple(sorted(hints)),
        next_action=_RECOMPUTE,
    )


def _checkpoint_record(
    where: _Line, record: Mapping[str, object]
) -> tuple[list[object], EnumeratedPower]:
    """The design and the enumerated power one checkpoint record holds, refused unless the
    enumeration was summed under this runtime and the plan's model is one it can be resumed
    from."""
    from increment.estimation.results import BINOMIAL_METHOD
    from increment.power.core import BINOMIAL_PLANNING_MODEL

    layout = ("design", "enumeration", "planner")
    if record.keys() != set(layout):
        where.refuse(
            "calibration.conversion_route.checkpoint_layout",
            section="record",
            found=tuple(sorted(record)),
            expected=layout,
            next_action=_RECOMPUTE,
        )
    design = record["design"]
    if not (
        isinstance(design, list)
        and len(design) == 5
        and _conforms(int, design[0])
        and all(_conforms(float, value) for value in design[1:4])
        and design[4] in ("two-sided", "greater", "less")
    ):
        where.refuse(
            "calibration.conversion_route.checkpoint_design",
            design=design,
            next_action=_REMOVE_LINE,
        )
    enumeration = _section(Enumeration, record["enumeration"], where, "enumeration")
    plan = _section(Plan, record["planner"], where, "planner")
    floor = dense_min_count(_tail(cast("float", design[3]), cast("str", design[4])))
    if enumeration.construction != BINOMIAL_METHOD or enumeration.floor != floor:
        where.refuse(
            "calibration.conversion_route.checkpoint_runtime",
            construction=enumeration.construction,
            floor=enumeration.floor,
            current_construction=BINOMIAL_METHOD,
            current_floor=floor,
            next_action=_RECOMPUTE,
        )
    if plan.model != BINOMIAL_PLANNING_MODEL and plan.model not in RETIRED_PLANNER_MODELS:
        where.refuse(
            "calibration.conversion_route.checkpoint_planner_model",
            model=plan.model,
            current=BINOMIAL_PLANNING_MODEL,
            retired=tuple(sorted(RETIRED_PLANNER_MODELS)),
            next_action=_RECOMPUTE,
        )
    return list(design), EnumeratedPower(plan, enumeration)


def read_checkpoint(path: Path) -> dict[str, EnumeratedPower]:
    """The designs a ``bound`` checkpoint holds, keyed by ``_design_key`` as JSON (a design
    recorded again is its last record). A record holds the design, its enumeration with the
    runtime it was summed under, and its plan with the planner model that made it. A record
    that names neither (one undated `power` section, as every earlier layout wrote), or an
    enumeration of another runtime, or a model that is neither current nor retired, is refused
    with a `CheckpointError` whose code names the reason, and never reused or relabelled; a
    retired model's plan is replaced when the run resumes (``bound``) and its enumeration
    kept."""
    done: dict[str, EnumeratedPower] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        where = _Line(str(path), number)
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        if not isinstance(record, dict):
            where.refuse(
                "calibration.conversion_route.checkpoint_unreadable", next_action=_REMOVE_LINE
            )
        design, power = _checkpoint_record(where, record)
        done[json.dumps(design)] = power
    return done


def _append(handle, cell: MirrorCell, power: EnumeratedPower) -> None:
    record = {
        "design": _design_key(cell),
        "enumeration": asdict(power.enumeration),
        "planner": asdict(power.plan),
    }
    handle.write(json.dumps(record) + "\n")


def bound(*, workers: int, out: Path | None = None, grid: BoundGrid = "original") -> int:
    """Planned power against the pipeline's exact rejection probability at every design of
    ``bound_cells``, each judged by ``bound_holds``. Each design is appended to ``out`` as it
    completes, and designs already in ``out`` are not enumerated again, so an interrupted run
    resumes. A recorded design planned under a retired model keeps its enumeration and is
    planned again under the current one, appended as that design's newest record."""
    from increment.power.core import BINOMIAL_PLANNING_MODEL

    def run(function, cells):
        return _pool(workers).imap(function, cells) if workers > 1 else map(function, cells)

    def key(cell: MirrorCell) -> str:
        return json.dumps(_design_key(cell))

    def record(cell: MirrorCell, power: EnumeratedPower) -> None:
        done[key(cell)] = power
        if out is not None:
            with out.open("a") as handle:
                _append(handle, cell, power)

    designs = bound_cells(grid)
    done = read_checkpoint(out) if out is not None and out.exists() else {}
    replanned = [
        cell
        for cell in designs
        if key(cell) in done and done[key(cell)].plan.model != BINOMIAL_PLANNING_MODEL
    ]
    pending = [cell for cell in designs if key(cell) not in done]
    for cell, plan in zip(replanned, run(plan_design, replanned), strict=True):
        record(cell, EnumeratedPower(plan, done[key(cell)].enumeration))
    for cell, power in zip(pending, run(enumerated_power, pending), strict=True):
        record(cell, power)
    failed = 0
    margins: dict[str, list[float]] = {}
    for cell in designs:
        power = done[key(cell)]
        plan, runtime = power.plan, power.enumeration
        ok = bound_holds(power)
        failed += not ok
        margins.setdefault(plan.route, []).append(power.margin)
        enclosure = f"[{plan.lower:.6f}, {plan.upper:.6f}]"
        print(
            f"{plan.route:<10} n={cell.n:>8} p_c={cell.p_c:<5} lift={cell.lift:<5} "
            f"alpha={cell.alpha:<5} {cell.alternative:<9} share={runtime.asymptotic_share:.4f} "
            f"planned={plan.planned:.6f} ({plan.basis}) enclosure={enclosure} "
            f"asym_part={runtime.asymptotic_part:.6f} "
            f"finite_part={runtime.finite_part:.6f} hybrid={runtime.hybrid:.6f} "
            f"margin={power.margin:+.6f} {'ok' if ok else 'FAIL'}",
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
    sub.add_parser("conformance", help="compare the vectorised intervals with estimate_lift")
    hybrid_parser = sub.add_parser("hybrid", help="pipeline noncoverage at the boundary")
    hybrid_parser.add_argument("--workers", type=int, default=1)
    hybrid_parser.add_argument("--tails", type=float, nargs="+", default=list(TAILS))
    hybrid_parser.add_argument("--replicate-tails", type=float, nargs="*", default=[0.1, 0.05])
    hybrid_parser.add_argument("--count", type=int, default=3)
    bound_parser = sub.add_parser("bound", help="planned power against the exact pipeline power")
    bound_parser.add_argument("--workers", type=int, default=1)
    bound_parser.add_argument("--out", type=Path, default=None)
    bound_parser.add_argument(
        "--grid", choices=("original", "extended", "tails", "window"), default="original"
    )
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
        try:
            return required(read_rows(args.paths))
        except ScanError as refusal:
            print(refusal, file=sys.stderr)
            return 2
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
        return 0 if gap < 1e-9 and interpolation < CRITICAL_AGREEMENT and not hard else 1
    if args.command == "mirror":
        return mirror(args.reps, args.seed, workers=args.workers)
    if args.command == "bound":
        try:
            return bound(workers=args.workers, out=args.out, grid=args.grid)
        except CheckpointError as refusal:
            print(f"{refusal.code}: {refusal}", file=sys.stderr)
            return 2
    return hybrid(
        args.tails, workers=args.workers, count=args.count, replicate_tails=args.replicate_tails
    )


if __name__ == "__main__":
    sys.exit(main())
