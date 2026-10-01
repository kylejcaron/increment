"""Tail-sensitive simultaneous uniform-rank bands with certified calibration.

The Steck ordered-simplex determinant is evaluated by its Hessenberg
recurrence, using directed Decimal intervals to enclose rounding error.
Calibration concerns the actual binary boundaries returned by Beta inversion.
Recurrence: Wang and Miecznikowski (2022), equation (5),
https://pmc.ncbi.nlm.nih.gov/articles/PMC9042032/.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING

from increment._winsor_errors import winsor_refuse

# Decimal calibration is quadratic per bisection step; bound its public cost.
MAX_RANK_ARM_SIZE = 64


def validate_rank_size(n: int) -> None:
    if n > MAX_RANK_ARM_SIZE:
        winsor_refuse(
            "rank_size_unsupported",
            f"Certified rank calibration supports at most {MAX_RANK_ARM_SIZE} units per arm.",
        )


@dataclass(frozen=True)
class RankBand:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    error: float
    calibration: str

    def __post_init__(self):
        object.__setattr__(self, "lower", tuple(map(float, self.lower)))
        object.__setattr__(self, "upper", tuple(map(float, self.upper)))
        if not self.lower or len(self.lower) != len(self.upper) or not 0 <= self.error < 1:
            winsor_refuse(
                "invalid_state", "Rank-band state requires an arm size and valid error budget."
            )


def crossing_probability(
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    *,
    precision: int = 80,
) -> tuple[Decimal, Decimal]:
    """Enclose P(a_i <= U_(i) <= b_i, all i), without iid-pool assumptions.

    P_k=sum_{j=1}^k (-1)^(j-1) binom(k,j)
        (b_{k-j+1}-a_k)_+^j P_{k-j}; P_0=1.
    This is n! times the ordered-simplex volume. Monotone boundaries
    and a_i <= b_i are required. Decimal arithmetic encloses every term.
    """
    n = len(lower)
    validate_rank_size(n)
    if (
        n != len(upper)
        or any(not 0 <= a <= b <= 1 for a, b in zip(lower, upper, strict=True))
        or tuple(sorted(lower)) != lower
        or tuple(sorted(upper)) != upper
    ):
        winsor_refuse("invalid_state", "Rank boundaries must be ordered probabilities.")
    down = Context(prec=precision, rounding=ROUND_FLOOR)
    up = Context(prec=precision, rounding=ROUND_CEILING)
    zero, one = Decimal(0), Decimal(1)
    a = tuple(Decimal.from_float(x) for x in lower)
    b = tuple(Decimal.from_float(x) for x in upper)
    bounds = [(one, one)]
    for k in range(1, n + 1):
        lo = hi = zero
        binomial = 1
        for j in range(1, k + 1):
            left = max(zero, down.subtract(b[k - j], a[k - 1]))
            right = max(zero, up.subtract(b[k - j], a[k - 1]))
            if right == zero:
                break
            binomial = binomial * (k - j + 1) // j
            coefficient = Decimal(binomial)
            term_lo = down.multiply(coefficient, down.power(left, j))
            term_hi = up.multiply(coefficient, up.power(right, j))
            term_lo = down.multiply(term_lo, bounds[k - j][0])
            term_hi = up.multiply(term_hi, bounds[k - j][1])
            if j % 2:
                lo, hi = down.add(lo, term_lo), up.add(hi, term_hi)
            else:
                lo, hi = down.subtract(lo, term_hi), up.subtract(hi, term_lo)
        # Each leading minor is itself an ordered-simplex probability.
        bounds.append((max(zero, lo), min(one, hi)))
    return bounds[-1]


def _boundaries(n: int, eta: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    import numpy as np
    from scipy.stats import beta

    if eta == 0:
        return (0.0,) * n, (1.0,) * n
    ranks = np.arange(1, n + 1)
    a = beta.ppf(eta, ranks, n + 1 - ranks)
    b = beta.isf(eta, ranks, n + 1 - ranks)
    # Widen to protect the inverse-CDF boundary rounding, then calibrate
    # these particular boundaries, not their idealized Beta quantiles.
    a = np.maximum(0, np.nextafter(a, -np.inf))
    b = np.minimum(1, np.nextafter(b, np.inf))
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return (0.0,) * n, (1.0,) * n
    return tuple(map(float, a)), tuple(map(float, b))


@lru_cache(maxsize=128)
def simultaneous_rank_band(n: int, error: float) -> RankBand:
    """Largest certified tail parameter on a fixed 24-step bisection grid.

    No Monte Carlo calibration, success-dependent selection, or recovery
    of small alpha from a rounded confidence level is involved.
    """
    validate_rank_size(n)
    if n < 1 or not 0 <= error < 1:
        winsor_refuse("invalid_state", "Invalid arm size or rank-band error allocation.")
    if error == 0:
        return RankBand((0.0,) * n, (1.0,) * n, error, "exact-rank-v1:full")
    # Round the coverage target upward, including subnormal allocations.
    exact_error = Decimal.from_float(error)
    context = Context(prec=max(80, -exact_error.adjusted() + 20), rounding=ROUND_CEILING)
    target = context.subtract(Decimal(1), exact_error)
    safe = ((0.0,) * n, (1.0,) * n)
    lo, hi = 0.0, min(error, 0.25)
    for _ in range(24):
        eta = (lo + hi) / 2
        candidate = _boundaries(n, eta)
        # Cancellation can require many digits at large n. An inconclusive
        # enclosure is never accepted; increase precision before deciding.
        precision = max(80, math.isqrt(n) * 8)
        for _ in range(4):
            probability = crossing_probability(*candidate, precision=precision)
            if probability[0] >= target or probability[1] < target:
                break
            precision *= 2
        if probability[0] >= target:
            lo, safe = eta, candidate
        else:
            hi = eta
    return RankBand(*safe, error, f"exact-rank-v1:eta={float(lo).hex()}")


def bonferroni_rank_band(n: int, error: float) -> RankBand:
    """Independent conservative baseline; production uses simultaneous calibration."""
    if n < 1 or not 0 <= error < 1:
        winsor_refuse("invalid_state", "Invalid arm size or rank-band error allocation.")
    eta = math.nextafter(error / (2 * n), 0.0)
    return RankBand(*_boundaries(n, eta), error, "bonferroni-rank-v1")


if TYPE_CHECKING:
    from increment.winsor import WinsorRawState


@dataclass(frozen=True)
class _NumericInterval:
    lower: float
    upper: float
    reason: str


def _interval(lo: float, hi: float, reason: str) -> _NumericInterval:
    return _NumericInterval(lo, hi, reason)


def _split_down(alpha: float, count: int) -> float:
    return max(0.0, math.nextafter(alpha / count, 0.0))


def _mean(values) -> float:
    return float(_mean_fraction(values))


def _mean_fraction(values) -> Fraction:
    return sum((Fraction(float(x)) for x in values), start=Fraction()) / len(values)


def _finite_point(value: Fraction) -> float | None:
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _linear_cutoff(raw: WinsorRawState) -> float:
    """Type-7 interpolation, without overflowing the neighboring-value gap."""
    values = sorted(x for arm in raw.arms for x in arm.values)
    rank = (len(values) - 1) * raw.quantile
    left, right = math.floor(rank), math.ceil(rank)
    weight = Fraction(rank - left)
    return float((1 - weight) * Fraction(values[left]) + weight * Fraction(values[right]))


def _up(value: float) -> float:
    return math.nextafter(float(value), math.inf)


def pooled_max_error(quantile: float, counts: Sequence[int]) -> float:
    """Sharp allocation-weighted maximum bound q**N, rounded upward."""
    if not 0 < quantile < 1 or not counts or any(n < 1 for n in counts):
        winsor_refuse(
            "invalid_state", "A maximum bound requires a quantile and positive arm sizes."
        )
    return _outward(Fraction(quantile) ** sum(counts), upper=True)


@lru_cache(maxsize=128)
def _maximum_admissible(quantile: float, total: int, error: float) -> bool:
    return Fraction(quantile) ** total <= Fraction(error)


@lru_cache(maxsize=128)
def _rank_construction(raw: WinsorRawState, control: str, treatment: str, alpha: float):
    """Reconstruct all observable fields from immutable raw state and alpha."""
    if not 0 < alpha < 1:
        winsor_refuse("invalid_state", "alpha must lie strictly between zero and one.")
    raw.arm(control)
    raw.arm(treatment)
    if raw.support is None or raw.inference.method != "joint-rank-projection-v1":
        winsor_refuse("support_required", "Select rank inference with declared support.")
    if control == treatment:
        winsor_refuse("pool_mismatch", "Contrast arms must differ.")
    for arm in raw.arms:
        validate_rank_size(len(arm.values))
    cutoff_alpha = _split_down(alpha, 2)
    arm_alpha = _split_down(alpha, 2 * len(raw.arms))
    bands = tuple(simultaneous_rank_band(len(a.values), arm_alpha) for a in raw.arms)
    upper = raw.support.upper if raw.support.upper is not None else math.inf
    if raw.support.quantile_upper is not None:
        upper = min(upper, raw.support.quantile_upper)
    total = sum(len(a.values) for a in raw.arms)
    if _maximum_admissible(raw.quantile, total, cutoff_alpha):
        upper = min(upper, max(a.values[-1] for a in raw.arms))
    relative, additive, cutoff = project_bands(raw, control, treatment, bands, upper)
    observed_cutoff = _linear_cutoff(raw)
    mc = _mean_fraction(tuple(min(y, observed_cutoff) for y in raw.arm(control).values))
    mt = _mean_fraction(tuple(min(y, observed_cutoff) for y in raw.arm(treatment).values))
    point = _finite_point(mt / mc - 1) if mc else None
    reference = (
        _up(math.fsum([arm_alpha] * len(raw.arms))) if arm_alpha else 0.0,
        cutoff_alpha,
        tuple(b.calibration for b in bands),
    )
    return reference, relative, additive, cutoff, point, _finite_point(mt - mc)


def _outward(value: Fraction | float, *, upper: bool) -> float:
    """Convert an exact endpoint once, toward the outside of the set."""
    if isinstance(value, float):
        return value
    try:
        result = float(value)
    except OverflowError:
        result = math.inf if value > 0 else -math.inf
    if math.isinf(result):
        if (result > 0) == upper:
            return result
        return math.nextafter(result, 0.0)
    if (upper and Fraction(result) < value) or (not upper and Fraction(result) > value):
        return math.nextafter(result, math.inf if upper else -math.inf)
    return result


def _exact_interval(lo: Fraction | float, hi: Fraction | float, reason: str) -> _NumericInterval:
    return _interval(_outward(lo, upper=False), _outward(hi, upper=True), reason)


class _JointProjection:
    """Piecewise-linear mean envelopes with exact binary-rational arithmetic."""

    def __init__(self, raw, control, treatment, bands, cutoff_upper):
        from bisect import bisect_left, bisect_right

        groups = [a.group_id for a in raw.arms]
        raw.arm(control)
        raw.arm(treatment)
        if control == treatment or len(bands) != len(groups):
            winsor_refuse(
                "pool_mismatch", "Each distinct contrast needs the complete arm-band pool."
            )
        self.raw = raw
        self.ci, self.ti = groups.index(control), groups.index(treatment)
        total = sum(len(a.values) for a in raw.arms)
        self.weights = tuple(Fraction(len(a.values), total) for a in raw.arms)
        self.q, self.lower = Fraction(raw.quantile), Fraction(raw.support.lower)
        self.upper = Fraction(cutoff_upper) if math.isfinite(cutoff_upper) else math.inf
        knots = {self.lower}
        for arm in raw.arms:
            knots.update(Fraction(x) for x in arm.values if x <= cutoff_upper)
        if math.isfinite(cutoff_upper):
            knots.add(Fraction(cutoff_upper))
        self.grid = tuple(sorted(knots))
        self.lows, self.ups, self.left_lows, self.left_ups = [], [], [], []
        for arm, band in zip(raw.arms, bands, strict=True):
            n = len(arm.values)
            if (
                len(band.lower) != n
                or len(band.upper) != n
                or any(not 0 <= a <= b <= 1 for a, b in zip(band.lower, band.upper, strict=True))
                or tuple(sorted(band.lower)) != band.lower
                or tuple(sorted(band.upper)) != band.upper
            ):
                winsor_refuse(
                    "invalid_state", "Rank bands must be ordered probabilities matching arm sizes."
                )
            a, b = (
                (Fraction(0), *(Fraction(value) for value in band.lower)),
                (*(Fraction(value) for value in band.upper), Fraction(1)),
            )
            right = [bisect_right(arm.values, x) for x in self.grid]
            left = [bisect_left(arm.values, x) for x in self.grid]
            hard_upper = raw.support.upper
            self.lows.append(
                tuple(
                    Fraction(1) if hard_upper is not None and x >= hard_upper else a[k]
                    for x, k in zip(self.grid, right, strict=True)
                )
            )
            self.ups.append(
                tuple(
                    Fraction(1) if hard_upper is not None and x >= hard_upper else b[k]
                    for x, k in zip(self.grid, right, strict=True)
                )
            )
            self.left_lows.append(tuple(a[k] for k in left))
            self.left_ups.append(tuple(b[k] for k in left))
        self.integral_l = self._integrals(self.lows)
        self.integral_u = self._integrals(self.ups)
        self.extrema = [math.inf, -math.inf, math.inf, -math.inf]
        self.cutoffs = [math.inf, -math.inf]
        self.denominator_unseparated = False

    def _integrals(self, cdfs):
        result = []
        for cdf in cdfs:
            prefix = [Fraction(0)]
            for i in range(1, len(self.grid)):
                prefix.append(prefix[-1] + (self.grid[i] - self.grid[i - 1]) * (1 - cdf[i - 1]))
            result.append(tuple(prefix))
        return result

    def _pool(self, values):
        return sum(w * v for w, v in zip(self.weights, values, strict=True))

    def _means(self, index, c, left_lo, left_up):
        from bisect import bisect_right

        pool_lower = self._pool(left_lo)
        low, high = [], []
        for g, weight in enumerate(self.weights):
            cap = min(left_up[g], (self.q - pool_lower + weight * left_lo[g]) / weight)
            split = bisect_right(self.ups[g], cap, 0, index)
            base = (
                self.lower
                + self.integral_u[g][split]
                + (self.grid[index] - self.grid[split]) * (1 - cap)
            )
            low_slope = 1 - min(self.ups[g][index], cap)
            high_base = self.lower + self.integral_l[g][index]
            high_slope = 1 - self.lows[g][index]
            if c == math.inf:
                low.append((base, low_slope))
                high.append((high_base, high_slope))
            else:
                distance = c - self.grid[index]
                low.append((base + distance * low_slope, Fraction(0)))
                high.append((high_base + distance * high_slope, Fraction(0)))
        return low, high

    @staticmethod
    def _ratio(num, den):
        if den[1] > 0:
            return num[1] / den[1] - 1
        if den[0] <= 0 or num[1] > 0:
            return math.inf
        return num[0] / den[0] - 1

    @staticmethod
    def _difference(x, y):
        slope = x[1] - y[1]
        if slope:
            return math.inf if slope > 0 else -math.inf
        return x[0] - y[0]

    def _consider(self, index, c, left_lo, left_up):
        low, high = self._means(index, c, left_lo, left_up)
        floor = self.raw.support.control_mean_lower
        if floor is not None:
            floor = Fraction(floor)
            if high[self.ci][1] == 0 and high[self.ci][0] < floor:
                return
            if low[self.ci][1] == 0:
                low[self.ci] = (max(low[self.ci][0], floor), Fraction(0))
        self.cutoffs[0] = min(self.cutoffs[0], c)
        self.cutoffs[1] = max(self.cutoffs[1], c)
        if self.lower < 0:
            # Signed support may contain a zero control mean.
            rlo, rhi = -math.inf, math.inf
        elif low[self.ci] == (0, 0):
            self.denominator_unseparated = True
            rlo, rhi = Fraction(-1), math.inf
        else:
            rlo = max(Fraction(-1), self._ratio(low[self.ti], high[self.ci]))
            rhi = self._ratio(high[self.ti], low[self.ci])
        alo = self._difference(low[self.ti], high[self.ci])
        ahi = self._difference(high[self.ti], low[self.ci])
        self.extrema = [
            min(self.extrema[0], rlo),
            max(self.extrema[1], rhi),
            min(self.extrema[2], alo),
            max(self.extrema[3], ahi),
        ]

    def _gap(self, index, right, lo, up):
        x = self.grid[index]
        self._consider(index, x, lo, up)
        self._consider(index, right, lo, up)
        floor = self.raw.support.control_mean_lower
        if floor is not None:
            low, high = self._means(index, math.inf, lo, up)
            for base, slope in (low[self.ci], high[self.ci]):
                if slope > 0:
                    crossing = x + (Fraction(floor) - base) / slope
                    if x < crossing < right:
                        self._consider(index, crossing, lo, up)

    def project(self):
        for i, x in enumerate(self.grid):
            lo, up = tuple(row[i] for row in self.lows), tuple(row[i] for row in self.ups)
            left_lo = tuple(row[i] for row in self.left_lows)
            left_up = tuple(row[i] for row in self.left_ups)
            if self._pool(left_lo) <= self.q <= self._pool(up):
                self._consider(i, x, left_lo, left_up)
            right = self.grid[i + 1] if i + 1 < len(self.grid) else self.upper
            if right > x and self._pool(lo) <= self.q <= self._pool(up):
                self._gap(i, right, lo, up)
        if self.extrema[0] == math.inf:
            # Never discard samples on which the confidence region is empty.
            return (
                _interval(-math.inf, math.inf, "empty_region"),
                _interval(-math.inf, math.inf, "empty_region"),
                _exact_interval(self.lower, self.upper, "empty_region"),
            )
        reason = (
            "denominator_nonseparation"
            if self.denominator_unseparated
            else "unidentified_cutoff_tail"
        )
        if self.lower < 0:
            reason = "signed_support_relative_projection"
        if self.raw.support.upper == self.raw.support.lower == 0:
            self.extrema[:2] = [math.nan, math.nan]
            reason = "control_mean_identically_zero"
        return (
            _exact_interval(self.extrema[0], self.extrema[1], reason),
            _exact_interval(self.extrema[2], self.extrema[3], "unidentified_cutoff_tail"),
            _exact_interval(self.cutoffs[0], self.cutoffs[1], "unidentified_cutoff_upper"),
        )


def project_bands(
    raw: WinsorRawState,
    control: str,
    treatment: str,
    bands: Sequence[RankBand],
    cutoff_upper: float = math.inf,
) -> tuple[_NumericInterval, _NumericInterval, _NumericInterval]:
    """Allocation-aware outer projection, including atoms and both gap limits.

    For each g, p*_g=min(U_g(c-),(q-sum_{h!=g} w_h L_h(c-))/w_g).
    Integrate 1-min(U_g,p*_g) for the lower mean and 1-L_g for the upper.
    Exact rational gap arithmetic precedes outward endpoint conversion.
    """
    if raw.support is None:
        winsor_refuse("support_required", "Rank projection requires declared support.")
    if math.isnan(cutoff_upper) or cutoff_upper < raw.support.lower:
        winsor_refuse("invalid_state", "Cutoff upper bound lies outside declared support.")
    if raw.support.upper is not None:
        cutoff_upper = min(cutoff_upper, raw.support.upper)
    if raw.support.quantile_upper is not None:
        cutoff_upper = min(cutoff_upper, raw.support.quantile_upper)
    return _JointProjection(raw, control, treatment, bands, cutoff_upper).project()
