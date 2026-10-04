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

    python -m calibration.conversion_route select --out /tmp/rsf0-route.jsonl
    python -m calibration.conversion_route verify
    python -m calibration.conversion_route hybrid --reps 20000
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
from tests.estimation._conversion_counts import lift_row
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


def delta_method_bounds(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int, tail: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log risk-ratio interval bounds the production delta-method route reports at
    one-sided ``tail``, for count arrays with ``0 < x < n``: each arm's centered moments
    give ``mean = x / n`` and ``var = x (n - x) / (n (n - 1))``, its log standard error is
    ``sqrt(var / (n mean ** 2))``, and the critical value is the Welch-Satterthwaite ``t``."""
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
        crit = _student_t.isf(tail, df)
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
    out: dict[float, tuple[float, float, float, float]] = {}
    for tail in tails:
        lower, upper = delta_method_bounds(grid_c, cell.n_c, grid_t, cell.n_t, tail)
        floor = threshold[tail] if isinstance(threshold, dict) else threshold
        routed = defined & (smallest >= floor)
        low_miss = defined & (lower > cell.truth)
        up_miss = defined & (upper < cell.truth)
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


def conformance(samples: int = 200, seed: int = 20261004) -> float:
    """Largest relative gap between ``delta_method_bounds`` and the production delta-method
    row over sampled count pairs routed asymptotic at every tail."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        n_c = int(rng.integers(500, 200_000))
        n_t = int(rng.integers(500, 200_000))
        x_c = int(rng.integers(50, max(51, n_c // 2)))
        x_t = int(rng.integers(50, max(51, n_t // 2)))
        tail = float(rng.choice(TAILS))
        row = lift_row((x_c, n_c, x_t, n_t), alpha=2.0 * tail)
        if row.reference_kind != "t":
            continue
        lower, upper = delta_method_bounds(np.array([x_c]), n_c, np.array([x_t]), n_t, tail)
        worst = max(
            worst,
            abs(math.expm1(float(lower[0])) - row.lift.lb) / max(1e-12, abs(row.lift.lb)),
            abs(math.expm1(float(upper[0])) - row.lift.ub) / max(1e-12, abs(row.lift.ub)),
        )
    return worst


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
    args = parser.parse_args(argv)
    if args.command == "select":
        return select(args.out, workers=args.workers, start=args.start, stop=args.stop)
    if args.command == "verify":
        return verify(workers=args.workers)
    if args.command == "conformance":
        gap = conformance()
        print(f"largest relative gap to the production interval: {gap:.3g}")
        return 0 if gap < 1e-9 else 1
    return hybrid(args.reps, args.seed, (0.025, 0.05, 0.005))


if __name__ == "__main__":
    sys.exit(main())
