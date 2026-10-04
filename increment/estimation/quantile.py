"""Distribution-free quantile point + SE for the log-RR transport.

Quantiles are not additive across units, so this module consumes raw per-unit
values through ``MomentSource.unit_frame`` rather than moments. The SE uses the
Woodruff (1952) inversion of a deterministic, distribution-free binomial
order-statistic interval; its recovery simulation is documented in the tests.

Tied order statistics. The bracket ``[Y_(a), Y_(b)]`` itself covers the
population quantile with probability at least ``1 - alpha`` for every
distribution, discrete or not (Thompson 1936; Scheffé & Tukey 1945).
Woodruff's symmetric interval around ``Q̂`` relies on ``Q̂`` sitting at the
bracket's log midpoint. Continuous data keeps it there, but ties quantize
the bracket: ``Q̂`` sticks to one recorded value while the bracket reaches a
whole recording cell further on the other side, and on a grid with a few
off-grid values, or prices ending in both .99 and .00, no fixed step says
how far. So when a bracket holds a tie and its arm is not resolved (below),
the reported half-width is the smallest symmetric one around ``Q̂`` that
contains the bracket, which inherits the bracket's own coverage whatever
the recording grid. Each bracket end is first pushed out by a quarter of a
recording cell, a continuity allowance that keeps the two-arm lift at or
below its nominal size when both quantiles move in whole cells (a quarter
cell was the smallest allowance measured to do so; with none, whole-rounded
medians at n=30000 rejected a true null about 6% of the time at
alpha=0.05), and the half-width is floored by the log gap from ``Q̂`` to
its nearest other recorded value, so a bracket collapsed onto one value
keeps a positive width (`quantile_half_width`, `QuantileArm`).

An arm is resolved when its 95% bracket spans at least
``RESOLVED_REPEATED_VALUES`` distinct repeated values: the classical
formula covers there, so a resolved arm, like any bracket without a tie, is
reported by the classical Woodruff formula, bit-identically. Only values
that repeat count, so an off-grid value recorded once cannot make a coarse
grid look resolved.

Measured against exact population quantiles (seeded grids of cents-,
seconds- and millisecond-rounded lognormal outcomes, Poisson counts, those
grids with 0.1-50% of values left off-grid, .99/.00 price endings,
coincidental ties and continuous data; up to 2,000 draws per cell), no cell
covers below its nominal level by more than Monte Carlo error, per arm or
for the two-arm lift. The median SE relative to the classical one at
alpha=0.05 is 1.00 on continuous data and resolved arms, 1.00-1.05 with
coincidental ties, and 1.1-3 on coarse rounding and counts, the most where
the bracket spans only two or three recorded values. It has no upper bound,
because a bracket collapsed onto one value has zero classical width; a
widened row states its own ratio in its note.

Every piece is fixed per arm or grows as the bracket widens, so a smaller
alpha never reports a narrower interval. The p-value (`_quantile_p_value`,
the smallest alpha whose joint interval excludes the null) is therefore
dual to the reported interval, and since it never reads the caller's alpha,
a multiplicity allocation cannot move it; the reported interval tracks
the alpha it is given.

This fixed-horizon standard error does not establish a time-uniform
quantile process. Registered sequential mean/ratio likelihoods refuse this
route before accessing unit rows. Use fixed-horizon quantile inference until
a matching quantile-specific process and inversion are available.

Binomial inversion certifies every candidate rank itself: the mass at the
rank through certified logarithms, the lower terms as a float64 ratio sum
under Higham's rounding bound, and the truncated remainder by a geometric
series, so no library CDF accuracy is assumed. An undecided small count is
resolved exactly; an undecided large count rounds outward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, overload

from increment.estimation._certified import Interval, log_interval, log_rising
from increment.estimation._readout_refusals import refuse_quantile_moments
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.meta import ESTIMATION_META_ALPHA_TOO_SMALL

if TYPE_CHECKING:
    from increment.decision import DecisionComputation
    from increment.estimation.results import LiftEstimate

import numpy as np
from scipy.stats import binom, norm

from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation.variance import stable_log_ratio

# Quantile Welch references remain disabled until independently calibrated.
_QUANTILE_SUPPORTS_WELCH_REFERENCE = False


def _wider_interval_note(se_c: float, se_t: float, classical_c: float, classical_t: float) -> str:
    """Note for a row reported wider than classical Woodruff, stating its own widening."""
    classical = math.hypot(classical_t, classical_c)
    if classical > 0.0:
        widening = (
            f"is {math.hypot(se_t, se_c) / classical:.2f} times as wide as the classical "
            "Woodruff interval"
        )
    else:
        widening = "has positive width where the classical Woodruff interval has none"
    return (
        f"this row's interval {widening}: tied values leave the sample quantile "
        "off-centre in its order-statistic bracket, and the interval is widened to "
        "contain that bracket so it keeps its coverage."
    )


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.quantile.log_quantile_se": "q must be in (0, 1), got {q}",
        "estimation.quantile.values_one_dimensional": "values must be one-dimensional, got shape {shape}",
        "estimation.quantile.values_contain_non": RefusalSpec(
            "estimation.quantile.values_contain_non",
            InvalidRequestError,
            lambda *, n_bad: (
                f"values contain {n_bad} non-finite entr{'y' if n_bad == 1 else 'ies'} "
                "(NaN or +/-inf) -- drop or impute them before quantile estimation"
            ),
        ),
        "estimation.quantile.too_small_bound": "n={n} is too small to bound the q={q} quantile at level {level:.2f} -- the distribution-free order-statistic bound needs n >= {n_min} per arm; collect more units or target a less extreme quantile",
        "estimation.quantile.quantile_positive_log": "q={q} quantile must be positive for the log-scale lift transport (point={point:.6g}, lower order stat={lo:.6g}) -- collect more units, target a higher quantile, or narrow the enrolled population until this quantile clears zero",
        "estimation.quantile.degenerate_spread_order": "degenerate spread at q={q}: order-statistic bounds coincide (all values in the bracket identical)",
        "estimation.quantile.control_group_found": "control_group {control_group!r} not found; groups: {arms}",
        "estimation.quantile.estimate_quantile.metric_arm": "quantile metric {metric!r}, arm {arm!r}: {error}",
    },
)

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA

_REFUSALS["estimation.meta.alpha_too_small"] = ESTIMATION_META_ALPHA_TOO_SMALL
_raise = raiser(_REFUSALS)


@lru_cache(maxsize=256)
def _binomial_tail_bounds(
    n: int, q: float, *, upper: bool, k: int, precision: int
) -> tuple[Decimal, Decimal]:
    """Reuse tail-independent enclosures across p-value bisection comparisons."""
    rate, one = Decimal.from_float(q), Decimal(1)
    down = Context(prec=precision, rounding=ROUND_FLOOR)
    up = Context(prec=precision, rounding=ROUND_CEILING)
    if upper:
        p_lo, p_hi = down.subtract(one, rate), up.subtract(one, rate)
        failure_lo = failure_hi = rate
    else:
        p_lo = p_hi = rate
        failure_lo, failure_hi = down.subtract(one, rate), up.subtract(one, rate)
    term_lo = down.power(failure_lo, n)
    term_hi = up.power(failure_hi, n)
    lower, higher = term_lo, term_hi
    for j in range(1, k + 1):
        term_lo = down.divide(
            down.multiply(down.multiply(term_lo, n - j + 1), p_lo),
            up.multiply(j, failure_hi),
        )
        term_hi = up.divide(
            up.multiply(up.multiply(term_hi, n - j + 1), p_hi),
            down.multiply(j, failure_lo),
        )
        lower, higher = down.add(lower, term_lo), up.add(higher, term_hi)
    return lower, higher


#: Integers below it convert to float64 exactly, so the ratio sum's rounding
#: model holds; larger counts keep the exact directed recurrence.
_FLOAT_COUNT_LIMIT = 2**53
#: An undecided count up to here is resolved by the exact recurrence.
_EXACT_RANK_LIMIT = 4096
_UNIT_ROUNDOFF = Fraction(1, 2**53)
#: Each ratio ``j f / ((n - j + 1) p)`` costs four correctly rounded operations
#: (the complement rate, two products, one quotient).
_RATIO_ROUNDINGS = 4
#: Terms are summed until the remainder bound is this share of the sum.
_REMAINDER_TOLERANCE = Fraction(1, 2**40)
_CHUNK_LIMIT = 2**22


def _rounding_bound(count: int) -> Fraction:
    """Higham's ``gamma``: the relative bound after ``count`` rounded operations."""
    product = count * _UNIT_ROUNDOFF
    return product / (1 - product)


def _log_binomial_coefficient(n: int, k: int) -> Interval:
    small = min(k, n - k)
    return log_rising(Fraction(n - small + 1), small) - log_rising(Fraction(1), small)


@lru_cache(maxsize=1024)
def _log_tail_enclosure(n: int, q: float, *, upper: bool, k: int) -> Interval:
    """Enclose ``log P(X <= k)``, ``X`` the count at rate ``q`` or ``n`` minus it.

    The mass at ``k`` is enclosed through certified logarithms. Each lower
    term is that mass times a product of ratios accumulated in float64: a
    ratio costs ``_RATIO_ROUNDINGS`` operations, its product and its addition
    one each, so ``gamma_(8m + 8)`` bounds the relative error of a sum over
    ``m`` ratios however the additions associate. Ratios decrease away from
    the mode, so the terms below the last one summed total at most a
    geometric series in that term's own ratio.
    Gradual underflow adds absolute errors below this slack: the sum starts
    at one, and ratios below the mode do not amplify those errors.
    """
    p = 1 - Fraction(q) if upper else Fraction(q)
    p_float, f_float = (1.0 - q, q) if upper else (q, 1.0 - q)
    log_mass = _log_binomial_coefficient(n, k) + k * log_interval(p) + (n - k) * log_interval(1 - p)
    chunk = min(_CHUNK_LIMIT, max(2048, math.ceil(2.0 * math.sqrt(n * p_float * f_float))))
    j, scale, total, summed, remainder = k, 1.0, 1.0, 0, Fraction(0)
    while j > 0:
        count = min(chunk, j)
        index = np.arange(j, j - count, -1, dtype=np.float64)
        terms = scale * np.cumprod((index * f_float) / ((n + 1.0 - index) * p_float))
        total += float(terms.sum())
        if not math.isfinite(total):
            raise ArithmeticError("binomial tail ratio sum did not stay finite")
        scale, j, summed = float(terms[-1]), j - count, summed + count
        if j == 0:
            remainder = Fraction(0)
            break
        ratio = Fraction(j * f_float / ((n + 1.0 - j) * p_float)) / (
            1 - _rounding_bound(_RATIO_ROUNDINGS)
        )
        if ratio >= 1:
            continue
        remainder = Fraction(scale) / (1 - _rounding_bound(8 * summed + 8)) * ratio / (1 - ratio)
        if remainder <= _REMAINDER_TOLERANCE * Fraction(total):
            break
    bound = _rounding_bound(8 * summed + 8)
    ratio_sum = Fraction(total)
    return log_mass + log_interval(
        Interval(ratio_sum * (1 - bound), ratio_sum / (1 - bound) + remainder)
    )


def _binomial_tail_within(n: int, q: float, tail: float, *, upper: bool, k: int = 0) -> bool:
    """Certify ``P(X <= k) <= tail``, ``X`` the count at rate ``q`` or ``n`` minus it.

    The certified enclosure decides first. An undecided count up to
    ``_EXACT_RANK_LIMIT`` is resolved by the exact directed recurrence; an
    undecided larger count is reported as not within, so callers round outward.
    """
    if n < _FLOAT_COUNT_LIMIT:
        enclosure = _log_tail_enclosure(n, q, upper=upper, k=k)
        limit = log_interval(Fraction(tail))
        if enclosure.hi <= limit.lo:
            return True
        if enclosure.lo > limit.hi:
            return False
        if k > _EXACT_RANK_LIMIT:
            return False
    limit = Decimal.from_float(tail)
    precision = 32
    while True:
        lower, higher = _binomial_tail_bounds(n, q, upper=upper, k=k, precision=precision)
        if higher <= limit:
            return True
        if lower > limit:
            return False
        precision *= 2


def _quantile_n_min(q: float, alpha: float) -> int:
    """Smallest arm size with admissible finite-support bracket endpoints.

    Verify the logarithmic seed with the same directed comparison as runtime.
    """
    tail = alpha / 2.0
    if tail == 0.0:
        _raise("estimation.meta.alpha_too_small")
    rate = Decimal.from_float(q)
    precision = max(64, -int(rate.as_tuple().exponent) + 32)
    context = Context(prec=precision)
    upper = q >= 0.5
    base = rate if upper else context.subtract(Decimal(1), rate)
    seed = context.divide(context.ln(Decimal.from_float(tail)), context.ln(base))
    minimum = math.ceil(seed)
    while not _binomial_tail_within(minimum, q, tail, upper=upper):
        minimum += 1
    while minimum > 1 and _binomial_tail_within(minimum - 1, q, tail, upper=upper):
        minimum -= 1
    return minimum


#: Arms whose 95% order-statistic bracket spans at least this many distinct
#: repeated values are resolved: the classical formula covers there.
RESOLVED_REPEATED_VALUES = 12
_RESOLUTION_ALPHA = 0.05


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)
    if alpha / 2.0 == 0.0:
        _raise("estimation.meta.alpha_too_small")


def _binomial_bracket_rank(n: int, q: float, tail: float, *, upper: bool = False) -> int:
    """Largest certified rank, seeded by the inverse CDF without trusting it."""
    rank = int(binom.ppf(tail, n, 1 - q if upper else q)) + 1
    while rank > 1 and not _binomial_tail_within(n, q, tail, upper=upper, k=rank - 1):
        rank -= 1
    if rank == 1 and not _binomial_tail_within(n, q, tail, upper=upper):
        return 0
    while rank < n and _binomial_tail_within(n, q, tail, upper=upper, k=rank):
        rank += 1
    return rank


def _bracket_ranks(n: int, q: float, alpha: float) -> tuple[int, int] | None:
    """Return 1-based bracket ranks, or None when unclamped bounds do not exist.

    Invert both ends in lower tails; upper-tail inversion can lose tiny alpha.
    """
    if n == 0:
        return None
    tail = alpha / 2.0
    if tail == 0.0:
        _raise("estimation.meta.alpha_too_small")
    a = _binomial_bracket_rank(n, q, tail)
    k = _binomial_bracket_rank(n, q, tail, upper=True)
    b = n - k + 1
    if a < 1 or b > n or b <= a:
        return None
    return a, b


def _nearest_gap(y: np.ndarray, point: float) -> float:
    """Log gap from ``point`` to the nearest other positive value in sorted ``y``."""
    if not point > 0.0:
        return 0.0
    below = int(np.searchsorted(y, point, side="left"))
    above = int(np.searchsorted(y, point, side="right"))
    log_point = math.log(point)
    gaps = []
    if below > 0 and y[below - 1] > 0.0:
        gaps.append(log_point - math.log(y[below - 1]))
    if above < y.shape[0]:
        gaps.append(math.log(y[above]) - log_point)
    return min(gaps) if gaps else 0.0


def _repeated_step(repeated: np.ndarray, point: float) -> float:
    """Smallest spacing between the repeated values adjacent to ``point``
    (sorted ``repeated``); zero when fewer than two values repeat."""
    if repeated.shape[0] < 2:
        return 0.0
    lo = int(np.searchsorted(repeated, point, side="right")) - 1  # largest <= point
    hi = int(np.searchsorted(repeated, point, side="left"))  # smallest >= point
    gaps = []
    if 0 <= lo < hi < repeated.shape[0]:
        gaps.append(repeated[hi] - repeated[lo])
    if lo >= 1:
        gaps.append(repeated[lo] - repeated[lo - 1])
    if 0 <= hi < repeated.shape[0] - 1:
        gaps.append(repeated[hi + 1] - repeated[hi])
    return float(min(gaps))


@dataclass(frozen=True, slots=True)
class QuantileArm:
    """One arm's sample quantile plus the alpha-free facts about how finely
    its outcome is recorded, read by `quantile_half_width`.

    ``floor`` is the log gap from the point to its nearest other positive
    recorded value. ``cell`` is the recording step at the point, measured
    between values that repeat (an off-grid value recorded once cannot
    shrink it) and scaled by the share of observations whose value repeats
    (a continuous sample with a few coincidental ties keeps a negligible
    cell). ``repeated`` counts the distinct repeated values the arm's 95%
    order-statistic bracket spans (zero where that bracket does not exist).
    """

    q: float
    point: float
    floor: float
    cell: float
    repeated: int

    @classmethod
    def of(cls, sorted_values: np.ndarray, q: float) -> QuantileArm:
        y = sorted_values
        n = int(y.shape[0])
        if n == 0:
            return cls(q=q, point=math.nan, floor=0.0, cell=0.0, repeated=0)
        point = float(np.quantile(y, q))
        starts = np.flatnonzero(np.concatenate(([True], y[1:] != y[:-1])))
        counts = np.diff(np.append(starts, n))
        is_repeated = counts >= 2
        repeated = y[starts[is_repeated]]
        share = float(counts[is_repeated].sum()) / n
        ranks = _bracket_ranks(n, q, _RESOLUTION_ALPHA)
        spanned = 0
        if ranks is not None:
            spanned = int(
                np.searchsorted(repeated, y[ranks[1] - 1], side="right")
                - np.searchsorted(repeated, y[ranks[0] - 1], side="left")
            )
        return cls(
            q=q,
            point=point,
            floor=_nearest_gap(y, point),
            cell=share * _repeated_step(repeated, point),
            repeated=spanned,
        )

    @property
    def resolved(self) -> bool:
        return self.repeated >= RESOLVED_REPEATED_VALUES


def _bracket_half_width(arm: QuantileArm, lower: float, upper: float, *, tied: bool) -> float:
    """`quantile_half_width` before its positivity check: zero where the
    bracket has collapsed onto one value."""
    if not tied or arm.resolved:
        return (math.log(upper) - math.log(lower)) / 2.0
    # Without this allowance a lift between two lattice-valued quantiles was
    # measured rejecting a true null about 6% of the time at alpha=0.05
    # (whole-rounded medians at n=30000; tests/estimation/test_quantile_recovery.py).
    allowance = min(arm.cell, upper - lower) / 4.0
    low = lower - allowance
    if low <= 0.0:
        _raise("estimation.quantile.quantile_positive_log", q=arm.q, point=arm.point, lo=low)
    log_point = math.log(arm.point)
    return max(log_point - math.log(low), math.log(upper + allowance) - log_point, arm.floor)


def quantile_half_width(arm: QuantileArm, lower: float, upper: float, *, tied: bool) -> float:
    """Log half-width of the reported interval around ``arm.point`` for the
    order-statistic bracket ``[lower, upper]``.

    An untied bracket, or any bracket of a resolved arm, gets the classical
    Woodruff half-width ``(log upper - log lower) / 2``. Otherwise the
    half-width is the smallest symmetric one around the point that contains
    the bracket after each end is pushed out by a quarter of a recording
    cell (at most a quarter of the bracket's own width), floored at
    ``arm.floor``. Both grow as the bracket widens, so a smaller alpha never
    reports a narrower interval. Refuses where no positive log-scale
    interval exists.
    """
    half = _bracket_half_width(arm, lower, upper, tied=tied)
    if half <= 0.0:
        # Numerical backstop: distinct doubles can round to the same log.
        _raise("estimation.quantile.degenerate_spread_order", q=arm.q)
    return half


class _SortedArm:
    """One arm's values, validated and sorted once, with its alpha-free facts."""

    __slots__ = ("arm", "values", "_ties")

    def __init__(self, values: np.ndarray, q: float) -> None:
        if not 0.0 < q < 1.0:
            _raise("estimation.quantile.log_quantile_se", q=q)
        raw = np.asarray(values, dtype=float)
        if raw.ndim != 1:
            _raise("estimation.quantile.values_one_dimensional", shape=raw.shape)
        if raw.size and not np.isfinite(raw).all():
            # Checking only the sorted maximum misses a -inf, which sorts
            # first and silently shifts n/ranks; check every entry.
            n_bad = int((~np.isfinite(raw)).sum())
            _raise("estimation.quantile.values_contain_non", n_bad=n_bad)
        y = np.sort(raw)
        self.values = y
        # _ties[i]: equal adjacent pairs among y[: i + 1].
        self._ties = np.concatenate(([0], np.cumsum(y[1:] == y[:-1])))
        self.arm = QuantileArm.of(y, q)

    def bracket(self, alpha: float) -> tuple[float, float, bool]:
        """``(lower, upper, tied)``: the order-statistic bracket at ``alpha``.
        Refuses where none exists or its lower end is not positive."""
        y, arm = self.values, self.arm
        n = y.shape[0]
        ranks = _bracket_ranks(n, arm.q, alpha)
        if ranks is None:
            _raise(
                "estimation.quantile.too_small_bound",
                n=n,
                q=arm.q,
                level=math.fsum((1.0, -alpha)),
                n_min=_quantile_n_min(arm.q, alpha),
            )
        a, b = ranks
        lower, upper = float(y[a - 1]), float(y[b - 1])
        if arm.point <= 0.0 or lower <= 0.0:
            _raise("estimation.quantile.quantile_positive_log", q=arm.q, point=arm.point, lo=lower)
        return lower, upper, bool(self._ties[b - 1] > self._ties[a - 1])

    def half_width(self, alpha: float) -> float:
        """The log half-width at ``alpha``, zero where the bracket has
        collapsed onto one value (`se` refuses there)."""
        lower, upper, tied = self.bracket(alpha)
        return _bracket_half_width(self.arm, lower, upper, tied=tied)

    def half_widths(self, alpha: float) -> tuple[float, float]:
        """``(reported, classical)`` log half-widths at ``alpha``."""
        lower, upper, tied = self.bracket(alpha)
        half = quantile_half_width(self.arm, lower, upper, tied=tied)
        return half, (math.log(upper) - math.log(lower)) / 2.0

    def se(self, alpha: float) -> tuple[float, float, float]:
        """``(Q̂(q), se, classical_se)`` at ``alpha``."""
        half, classical = self.half_widths(alpha)
        z = norm.isf(alpha / 2)
        return self.arm.point, half / z, classical / z


def _log_quantile_se_impl(
    values: np.ndarray, q: float, alpha: float = 0.05
) -> tuple[float, float, float]:
    """``(Q̂(q), se, classical_se)``: the reported SE (see ``log_quantile_se``)
    and the classical Woodruff SE from the same sorted values and bracket."""
    _check_alpha(alpha)
    return _SortedArm(values, q).se(alpha)


def log_quantile_se(values: np.ndarray, q: float, alpha: float = 0.05) -> tuple[float, float]:
    """``(Q̂(q), se)``: the raw sample quantile and the SE of ``log Q̂(q)``
    (the Woodruff SE, or the tie-aware half-width of `quantile_half_width`
    over ``z``; see the module docstring).

    The point is returned untransformed so the caller can form the joint
    log ratio of two arms from their raw quantiles (``stable_log_ratio``)
    rather than from two separately rounded logs. The SE is on the log
    scale, ready for ``infer_lift``.
    """
    point, se, _ = _log_quantile_se_impl(values, q, alpha)
    return point, se


def _quantile_p_value(control: np.ndarray, treatment: np.ndarray, q: float, null: float) -> float:
    """Smallest two-sided alpha at which the reported joint interval
    excludes ``null`` -- a pure function of the two arms' values that never
    reads a caller-supplied alpha, so a multiplicity allocation cannot move
    it (the REPORTED interval, by contrast, tracks the caller's alpha).
    """
    return _inverted_p_value(_SortedArm(control, q), _SortedArm(treatment, q), null)


def _inverted_p_value(control: _SortedArm, treatment: _SortedArm, null: float) -> float:
    """`_quantile_p_value` on arms already sorted once.

    Each arm's half-width is non-increasing in alpha by construction (see
    `quantile_half_width`), and the critical value is resolved by the same
    reference-selection helper ``infer_lift`` uses, so ``p <= alpha`` exactly
    when the interval reported at ``alpha`` excludes ``null``, up to the
    bisection's float resolution (60 halvings of a 690-nat log-alpha span).

    An arm whose bracket does not exist at ``alpha``, or reaches a
    non-positive value, has no finite log-scale interval there, so it cannot
    exclude; that happens only below some alpha. One whose bracket has
    collapsed onto a single value, which happens only above some alpha, is
    a zero-width interval, which excludes every other value.
    """
    from increment.estimation.inference import _resolve_fixed_horizon

    if not (control.arm.point > 0.0 and treatment.arm.point > 0.0):
        return 1.0
    log_rr = stable_log_ratio(control.arm.point, treatment.arm.point)
    arm_ns = (
        (int(treatment.values.shape[0]), int(control.values.shape[0]))
        if _QUANTILE_SUPPORTS_WELCH_REFERENCE
        else None
    )

    def excludes_at(alpha: float) -> bool:
        try:
            half_c = control.half_width(alpha)
            half_t = treatment.half_width(alpha)
        except InvalidRequestError:
            return False
        z = norm.isf(alpha / 2)
        se_c, se_t = half_c / z, half_t / z
        crit = _resolve_fixed_horizon(
            alpha,
            "two-sided",
            dof=None,
            arm_ns=arm_ns,
            se_t=se_t,
            se_c=se_c,
            prior=None,
        ).crit
        assert crit is not None
        return abs(log_rr - null) > crit * math.hypot(se_t, se_c)

    lo, hi = 1e-300, math.nextafter(1.0, 0.0)
    if excludes_at(lo):
        # Excludes even at the widest interval tested: report that floor.
        return lo
    if not excludes_at(hi):
        # Non-exclusion must remain unselected for every admissible family threshold.
        return 1.0
    for _ in range(60):
        mid = math.sqrt(lo * hi)  # log-spaced midpoint
        if excludes_at(mid):
            hi = mid
        else:
            lo = mid
    return hi


@overload
def estimate_quantile_lift(
    *args: Any,
    _return_computation: Literal[False] = False,
    **kwargs: Any,
) -> list[LiftEstimate]: ...


@overload
def estimate_quantile_lift(
    *args: Any,
    _return_computation: Literal[True],
    **kwargs: Any,
) -> DecisionComputation[LiftEstimate]: ...
def estimate_quantile_lift(
    src: Any,
    metric: Any,
    control_group: str,
    *,
    methods: list[Any] | None = None,
    prior: Any = None,
    alpha: float = 0.05,
    inference: Any = None,
    method_roles: Any = None,
    unit_rows: Any = None,
    _return_computation: bool = False,
) -> list[LiftEstimate] | DecisionComputation[LiftEstimate]:
    """Relative lift of metric.quantile, one estimate per (method x arm).

    Consumes ``src.unit_frame(metric)`` (unit_id, group_id, y rows) -- or
    ``unit_rows``, the same native frame when the caller already loaded it -- and
    feeds the arms' joint ``log(Q̂_t / Q̂_c)`` (formed from the raw
    quantiles) with each arm's Woodruff se into ``infer_lift``, so priors
    and the closed-form interval behave identically to every other metric
    type. Arms are independent under randomization, matching infer_lift's
    ``sqrt(se_t^2 + se_c^2)`` combination.

    Sequential quantiles require a quantile-specific process. The registered
    mean and ratio likelihoods cannot use an order-statistic standard error.
    An observational source refuses: the Woodruff interval assumes independent
    randomized arms, so it would report a confounded contrast as a causal lift.
    """
    design = getattr(getattr(src, "context", None), "design", None)
    if getattr(design, "mechanism", None) == "observational":
        refuse_quantile_moments(metric, design)
    if inference is not None:
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "mean/ratio likelihoods do not certify quantiles; use fixed-horizon quantile "
            "inference (valid for one planned analysis, not repeated looks)",
        )
    import narwhals as nw

    from increment.estimation.engine import Method, _validate_methods
    from increment.estimation.inference import infer_lift

    methods = [Method(name="unadjusted")] if methods is None else methods

    def _frame() -> Any:
        # Read (or adopt) the unit rows only after every request-shape refusal.
        return nw.from_native(
            src.unit_frame(metric) if unit_rows is None else unit_rows, eager_only=True
        )

    if not methods:
        if not _return_computation:
            return []
        from increment.estimation.decision_types import (
            ArmHypothesisKey,
            DecisionComputation,
            DecisionFailure,
        )

        groups = sorted({str(group) for group in _frame()["group_id"].to_list()})
        failures: dict[Any, Any] = {}
        for group_id in groups:
            if group_id == control_group:
                continue
            hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
            failures[hypothesis] = DecisionFailure(
                hypothesis,
                "estimation.quantile.no_decision_method",
                {"metric": metric.name, "group_id": group_id},
            )
        return DecisionComputation(results=(), evidence={}, failures=failures)
    _validate_methods(methods)
    resolved_method_roles: dict[str, Literal["decision", "sensitivity"]] = dict(method_roles or {})
    if method_roles is None:
        from increment.estimation.engine import resolve_method_roles

        resolved_method_roles = resolve_method_roles(methods)
    for m in methods:
        if m.variance_reduction != "none":
            from increment.compatibility import Unsupported, refuse_unsupported

            refuse_unsupported(Unsupported("arm.metric.quantile_cuped"), metric=metric.name)

    frame = _frame()
    # Normalize group_id to str, matching the moments path's convention:
    # unit_frame does not cast it, so an int column would never match a str control_group.
    groups = [str(g) for g in frame["group_id"].to_list()]
    y = np.asarray(frame["y"].to_list(), dtype=float)
    arms = {g: y[np.asarray([gv == g for gv in groups])] for g in set(groups)}
    if control_group not in arms:
        _raise(
            "estimation.quantile.control_group_found",
            control_group=control_group,
            arms=sorted(arms),
        )

    def _for_arm(values: np.ndarray, arm: str) -> tuple[_SortedArm, float, float]:
        # Sort each arm once: the row's SE and the p-value inversion share it.
        try:
            _check_alpha(alpha)
            sorted_arm = _SortedArm(values, metric.quantile)
            _, se, classical_se = sorted_arm.se(alpha)
        except ValueError as e:
            _raise(
                "estimation.quantile.estimate_quantile.metric_arm",
                metric=metric.name,
                arm=arm,
                error=str(e),
            )
        return sorted_arm, se, classical_se

    control, se_c, classical_se_c = _for_arm(arms[control_group], control_group)
    q_c = control.arm.point
    out = []
    for g, values in sorted((g, v) for g, v in arms.items() if g != control_group):
        treatment, se_t, classical_se_t = _for_arm(values, g)
        q_t = treatment.arm.point
        widened = se_c != classical_se_c or se_t != classical_se_t
        log_rr = stable_log_ratio(q_c, q_t)
        n_comparison = int(values.shape[0] + arms[control_group].shape[0])
        arm_ns = (
            (values.shape[0], arms[control_group].shape[0])
            if _QUANTILE_SUPPORTS_WELCH_REFERENCE
            else None
        )
        # Inverting the bracket construction gives an alpha-free p-value, so multiplicity
        # reallocation cannot move it. The inversion ignores priors; with a prior, leave it
        # unset so p_value() uses the prior-aware Normal-posterior branch.
        p_value = _inverted_p_value(control, treatment, null=0.0) if prior is None else None
        for m in methods:
            est = infer_lift(
                metric=metric.name,
                group_id=str(g),
                method_role=resolved_method_roles.get(m.name, "decision"),
                method=m.name,
                log_rr=log_rr,
                se_t=se_t,
                se_c=se_c,
                prior=prior,
                alpha=alpha,
                inference_spec=inference,
                n_comparison=n_comparison,
                arm_ns=arm_ns,
                preferred_direction=metric.declared_preferred_direction,
            )
            updates: dict[str, Any] = {"quantile_p_value": p_value}
            if widened:
                updates["note"] = _wider_interval_note(se_c, se_t, classical_se_c, classical_se_t)
            out.append(est.model_copy(update=updates))
    if _return_computation:
        from increment.estimation.engine import _lift_decision_bundle

        return _lift_decision_bundle(
            out,
            inference=inference,
        )
    return out


def estimate_quantile_lift_computation(
    src: Any, metric: Any, control_group: str, **kwargs: Any
) -> DecisionComputation[LiftEstimate]:
    """Return quantile presentation rows together with typed decision evidence."""
    return estimate_quantile_lift(
        src,
        metric,
        control_group,
        _return_computation=True,
        **kwargs,
    )
