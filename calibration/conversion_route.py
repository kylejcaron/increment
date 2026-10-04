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

Planning is validated against the same production route: ``mirror`` compares the planned
power of dense, sparse and borderline designs with the simulated rejection rate of
``estimate_lift`` (``tests.estimation._conversion_counts.runtime_rejection_rate``).

    python -m calibration.conversion_route select --out /tmp/rsf0-route.jsonl
    python -m calibration.conversion_route verify
    python -m calibration.conversion_route conformance
    python -m calibration.conversion_route hybrid --reps 20000
    python -m calibration.conversion_route mirror --reps 20000
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.stats import binom as _binom
from scipy.stats import norm as _norm
from scipy.stats import t as _student_t

from increment.estimation.conversion_route import dense_min_count
from tests.estimation._conversion_counts import lift_row, runtime_rejection_rate
from tests.mc import (
    binomial_error_upper_bound,
    family_eta,
    scientific_delta,
)

#: One-sided tails the production alphas produce: two-sided ``alpha / 2`` for
#: ``alpha in {.001, .01, .05, .1}`` and directional ``alpha in {.001, .01, .05, .1}``.
TAILS = (0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
RISK_RATIOS = (0.5, 1.0, 1.25, 2.0)
ALLOCATIONS = ((1, 1), (1, 4), (4, 1))
CONTROL_SIZES = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
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
    grid_c, grid_t = np.meshgrid(x_c, x_t, indexing="ij")
    weight = np.outer(w_c, w_t)
    smallest = np.minimum.reduce([grid_c, cell.n_c - grid_c, grid_t, cell.n_t - grid_t])
    defined = smallest >= 1
    log_rr, se, df = _statistic(grid_c, cell.n_c, grid_t, cell.n_t)
    out: dict[float, tuple[float, float, float, float]] = {}
    for tail in tails:
        crit = _critical(df, tail)
        floor = threshold[tail] if isinstance(threshold, dict) else threshold
        routed = defined & (smallest >= floor)
        low_miss = defined & (log_rr - crit * se > cell.truth)
        up_miss = defined & (log_rr + crit * se < cell.truth)
        out[tail] = (
            float((weight * low_miss).sum()),
            float((weight * up_miss).sum()),
            float((weight * (low_miss & routed)).sum()),
            float((weight * (up_miss & routed)).sum()),
        )
    return out


@dataclass(frozen=True, slots=True)
class Excess:
    """Worst boundary cell of one ``(tail, m)``: excess over the tail in tolerance units."""

    tail: float
    m: float
    wald: float
    routed: float
    cell: Cell | None


def _excess_job(args: tuple[float, Cell, tuple[float, ...]]) -> tuple[Cell, dict]:
    m, cell, tails = args
    return cell, boundary_noncoverage(cell, tails, int(round(m)))


def worst_excess(
    m: float, tails: Sequence[float] = TAILS, *, workers: int = 1
) -> dict[float, Excess]:
    """Worst ``wald`` and ``routed`` excess over every boundary cell at design count ``m``,
    per tail, in units of ``scientific_delta(tail)`` (at most one passes)."""
    jobs = [(m, cell, tuple(tails)) for cell in cells(m)]
    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            results = pool.map(_excess_job, jobs)
    else:
        results = [_excess_job(job) for job in jobs]
    worst: dict[float, Excess] = {}
    for cell, table in results:
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


def required_count(
    rows: dict[int, dict[float, Excess]], tail: float, *, key: str = "wald"
) -> int | None:
    """Smallest ladder ``m`` whose excess is within tolerance at ``m`` and at every larger
    ladder step through ``MARGIN * m``; ``None`` when the ladder ends before one does."""
    ms = sorted(rows)
    passing = [getattr(rows[m][tail], key) <= 1.0 for m in ms]
    for index, m in enumerate(ms):
        window = [i for i, other in enumerate(ms) if i >= index and other <= MARGIN * m]
        if window and all(passing[i] for i in window) and all(passing[index:]):
            return m
    return None


def select(out: Path | None, *, workers: int, start: float, stop: float) -> int:
    """Measure every ladder step and report the required count per tail and the shipped law."""
    rows: dict[int, dict[float, Excess]] = {}
    lines: list[dict] = []
    for m in ladder(start, stop):
        rows[m] = worst_excess(m, workers=workers)
        for tail, excess in rows[m].items():
            lines.append(
                {
                    "m": m,
                    "tail": tail,
                    "delta": scientific_delta(tail),
                    "wald_excess": excess.wald,
                    "routed_excess": excess.routed,
                    "cell": asdict(excess.cell) if excess.cell else None,
                }
            )
        print(
            f"m={m:>6}  "
            + "  ".join(
                f"{tail:g}:{rows[m][tail].wald:6.2f}/{rows[m][tail].routed:6.2f}" for tail in TAILS
            ),
            flush=True,
        )
    summary = []
    for tail in TAILS:
        record = {
            "tail": tail,
            "z": float(_norm.isf(tail)),
            "required_wald": required_count(rows, tail),
            "required_routed": required_count(rows, tail, key="routed"),
            "shipped": dense_min_count(tail),
        }
        summary.append(record)
        print(json.dumps(record))
    if out is not None:
        out.write_text("\n".join(json.dumps(line) for line in [*lines, *summary]) + "\n")
    return 0


def verify(*, workers: int) -> int:
    """The shipped threshold at each tail passes at its own count and at ``MARGIN`` times it."""
    failed = 0
    for tail in TAILS:
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


def hybrid_upper_bounds(result: HybridResult, *, family_size: int) -> tuple[float, float]:
    """Exact one-sided upper confidence bounds on the per-tail noncoverage of ``result``."""
    eta = family_eta(0.01, family_size)
    return (
        binomial_error_upper_bound(result.lower_misses, result.reps, eta),
        binomial_error_upper_bound(result.upper_misses, result.reps, eta),
    )


def hybrid(reps: int, seed: int, tails: Iterable[float]) -> int:
    """Production-route noncoverage at cells straddling each shipped threshold."""
    failed = 0
    targets = [(tail, dense_min_count(tail)) for tail in tails]
    for tail, shipped in targets:
        for offset in (-2, -1, 0, 1, 2):
            m = shipped + offset
            for cell in cells(m)[:3]:
                result = simulate_hybrid(
                    cell, alpha=2.0 * tail, alternative="two-sided", reps=reps, seed=seed
                )
                delta = scientific_delta(tail)
                lo, up = hybrid_upper_bounds(result, family_size=len(targets) * 5 * 3)
                ok = max(lo, up) <= tail + delta
                failed += not ok
                print(
                    f"tail={tail:g} m={m} cell={cell.family}/{cell.n_c}/{cell.n_t} "
                    f"asym_share={result.asymptotic_share:.2f} bounds=({lo:.5f}, {up:.5f}) "
                    f"limit={tail + delta:.5f} {'ok' if ok else 'FAIL'}",
                    flush=True,
                )
    return 1 if failed else 0


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


def mirror(reps: int, seed: int) -> int:
    """Planned power against the simulated production rejection rate, per design: a dense plan
    within ``4 * se + 0.003``; a sparse plan within ``4 * se`` (its replay is exact up to the
    window mass); a borderline plan no larger than the simulated rate plus ``4 * se``."""
    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.estimation.conversion_route import planning_route
    from increment.power import Baseline, achieved_power

    failed = 0
    for cell in mirror_cells():
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
        failed += not ok
        print(
            f"{cell.label:<10} route={route:<10} n={cell.n:>8} p_c={cell.p_c:<5} "
            f"lift={cell.lift:<5} alpha={cell.alpha:<5} {cell.alternative:<9} "
            f"planned={planned.power:.4f} ({planned.power_basis}) simulated={rate:.4f} "
            f"se={se:.4f} asym_share={share:.2f} {'ok' if ok else 'FAIL'}",
            flush=True,
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
    verify_parser = sub.add_parser("verify", help="check the shipped threshold at every tail")
    verify_parser.add_argument("--workers", type=int, default=1)
    sub.add_parser("conformance", help="compare the vectorised interval with estimate_lift")
    hybrid_parser = sub.add_parser("hybrid", help="production-route noncoverage at the boundary")
    hybrid_parser.add_argument("--reps", type=int, default=20_000)
    hybrid_parser.add_argument("--seed", type=int, default=20261004)
    mirror_parser = sub.add_parser("mirror", help="planned power against the production route")
    mirror_parser.add_argument("--reps", type=int, default=20_000)
    mirror_parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args(argv)
    if args.command == "select":
        return select(args.out, workers=args.workers, start=args.start, stop=args.stop)
    if args.command == "verify":
        return verify(workers=args.workers)
    if args.command == "conformance":
        gap, compared, interpolation = conformance()
        print(
            f"largest relative gap to the production interval over {compared} count pairs: "
            f"{gap:.3g}; interpolated critical value: {interpolation:.3g}"
        )
        return 0 if gap < 1e-9 and interpolation < _CRITICAL_AGREEMENT else 1
    if args.command == "mirror":
        return mirror(args.reps, args.seed)
    return hybrid(args.reps, args.seed, (0.025, 0.05, 0.005))


if __name__ == "__main__":
    sys.exit(main())
