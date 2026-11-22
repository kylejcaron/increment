"""Metric inference from one law-weighted row per independent unit or schedule block.

A pooled ratio and a regression with an estimated slope are generally not
finite-sample design-unbiased. The delta and delete-one standard errors below
are approximations across independent replicates, not randomization intervals.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from increment.estimation._student_t import student_t_isf

_Reason = Literal[
    "zero_denominator",
    "weak_denominator",
    "insufficient_replicates",
    "rank_deficient",
    "high_leverage",
    "zero_spread",
    "numeric_range",
    "invalid_input",
]
_Arm = Literal["treatment", "control"]
type _Vector = Sequence[int | float | Fraction]
type _Rows = Sequence[_Vector]


@dataclass(frozen=True, slots=True)
class SwitchbackMetricFit:
    """A descriptive point, with a reference only when uncertainty is estimable."""

    point: float | None = None
    se: float | None = None
    dof: int | None = None
    reason: _Reason | None = None
    arm: _Arm | None = None
    center: Fraction | None = None


def _integer_column(column: np.ndarray) -> tuple[np.ndarray, Fraction] | _Reason:
    """Factor out a common rational unit without rounding any HT average."""
    ratios: list[tuple[int, int]] = []
    for value in column:
        if isinstance(value, (int, np.integer)):
            ratios.append((int(value), 1))
        elif isinstance(value, Fraction):
            ratios.append(value.as_integer_ratio())
        elif isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return "numeric_range"
            ratios.append(value.as_integer_ratio())
        else:
            return "invalid_input"
    denominator = math.lcm(*(d for _, d in ratios))
    integers = [a * (denominator // d) for a, d in ratios]
    common = math.gcd(*integers) or 1
    return np.array([a // common for a in integers], dtype=object), Fraction(common, denominator)


def _point(value: Fraction) -> float | None:
    try:
        rounded = float(value)
    except OverflowError:
        return None
    return rounded if rounded or not value else None


def _with_variance(
    point: float, center: Fraction, variance: Fraction, dof: int
) -> SwitchbackMetricFit:
    if not variance:
        return SwitchbackMetricFit(point, reason="zero_spread")
    # Materialize only the square root in output units, not its square.
    shift = (variance.numerator.bit_length() - variance.denominator.bit_length()) // 2
    scale = 1 << abs(2 * shift)
    mantissa = variance / scale if shift >= 0 else variance * scale
    try:
        se = math.ldexp(math.sqrt(float(mantissa)), shift)
    except OverflowError:
        return SwitchbackMetricFit(point, reason="numeric_range")
    if not se:
        return SwitchbackMetricFit(point, reason="numeric_range")
    return SwitchbackMetricFit(point, se, dof, center=center)


@dataclass(frozen=True, slots=True)
class _RatioArm:
    y: np.ndarray
    d: np.ndarray
    sy: int
    sd: int
    rate_unit: Fraction

    @classmethod
    def from_columns(cls, numerator: np.ndarray, denominator: np.ndarray) -> _RatioArm | _Reason:
        y_column = _integer_column(numerator)
        if isinstance(y_column, str):
            return y_column
        d_column = _integer_column(denominator)
        if isinstance(d_column, str):
            return d_column
        y, y_unit = y_column
        d, d_unit = d_column
        return cls(y, d, int(y.sum()), int(d.sum()), y_unit / d_unit)

    def rate(self) -> Fraction:
        return self.rate_unit * self.sy / self.sd

    def influence(self) -> np.ndarray:
        # (Y_i - rate * D_i) / mean(D). Its sample mean is exactly zero.
        return (self.sd * self.y - self.sy * self.d) * (len(self.d) * self.rate_unit / self.sd**2)

    def relative_variance(self) -> Fraction:
        n = len(self.d)
        centered = n * self.d - self.sd
        return Fraction(int(centered @ centered), n * (n - 1) * self.sd**2)


def _denominator_guard(
    point: float, arms: tuple[tuple[_Arm, _RatioArm], ...], n: int, alpha: float
) -> SwitchbackMetricFit | None:
    critical_squared = None
    for name, arm in arms:
        rel_var = arm.relative_variance()
        if not rel_var:
            continue
        if critical_squared is None:
            try:
                critical = student_t_isf(alpha / 2, n - 1)
            except ArithmeticError:
                return SwitchbackMetricFit(point, reason="numeric_range", arm=name)
            if not math.isfinite(critical):
                return SwitchbackMetricFit(point, reason="numeric_range", arm=name)
            critical_squared = Fraction(critical) ** 2
        if critical_squared * rel_var >= 1:
            return SwitchbackMetricFit(point, reason="weak_denominator", arm=name)
    return None


def switchback_ratio(components: _Rows | np.ndarray, *, alpha: float) -> SwitchbackMetricFit:
    """Difference of arm ratios of totals for rows (YT, DT, YC, DC).

    Integer and Fraction components retain their input precision through the
    paired subtraction; finite floating inputs denote the supplied values.
    Each nonzero denominator mean must exclude zero under its own two-sided t
    reference at alpha. This is a stability check, not a joint confidence set.
    The delta variance is the sample variance of the *paired difference* of
    the arm influences divided by n, equivalently grad' Cov(mean row) grad.
    """
    rows = np.asarray(components, dtype=object)
    if not 0 < alpha < 1 or rows.ndim != 2 or rows.shape[1] != 4:
        return SwitchbackMetricFit(reason="invalid_input")
    n = len(rows)
    if not n:
        return SwitchbackMetricFit(reason="insufficient_replicates")
    t = _RatioArm.from_columns(rows[:, 0], rows[:, 1])
    if isinstance(t, str):
        return SwitchbackMetricFit(reason=t)
    c = _RatioArm.from_columns(rows[:, 2], rows[:, 3])
    if isinstance(c, str):
        return SwitchbackMetricFit(reason=c)
    arms: tuple[tuple[_Arm, _RatioArm], ...] = (("treatment", t), ("control", c))
    for name, arm in arms:
        if not arm.sd:
            return SwitchbackMetricFit(reason="zero_denominator", arm=name)
    center = t.rate() - c.rate()
    point = _point(center)
    if point is None:
        return SwitchbackMetricFit(reason="numeric_range")
    if n == 1:
        return SwitchbackMetricFit(point, reason="insufficient_replicates")
    unavailable = _denominator_guard(point, arms, n, alpha)
    if unavailable is not None:
        return unavailable
    paired = t.influence() - c.influence()
    return _with_variance(point, center, (paired @ paired) / (n * (n - 1)), n - 1)


def _jackknife_variance(deleted: list[Fraction], unit: Fraction) -> Fraction:
    """(n-1)/n times the centered sum of squares, in outcome units."""
    n = len(deleted)
    low = min(deleted)
    span = max(deleted) - low
    if not span:
        return Fraction(0)
    # Leave-one-out fits have unrelated denominators: center and rescale before
    # rounding, rather than growing their common denominator with sample size.
    scaled = np.fromiter((float((v - low) / span) for v in deleted), dtype=float, count=n)
    scaled -= math.fsum(scaled) / n
    return Fraction(float(scaled @ scaled)) * Fraction(n - 1, n) * (span * unit) ** 2


def _adjusted_fit(x: np.ndarray, y: np.ndarray, unit: Fraction) -> SwitchbackMetricFit:
    n = len(x)
    sx, sy = int(x.sum()), int(y.sum())
    sxx, sxy = int(x @ x), int(x @ y)
    # Integer coordinates make these centered products exact, including when
    # the predictor variation is small relative to its observed mean.
    det = n * sxx - sx**2
    if det == 0:
        return SwitchbackMetricFit(reason="rank_deficient")
    intercept = sxx * sy - sx * sxy
    center = unit * intercept / det
    point = _point(center)
    if point is None:
        return SwitchbackMetricFit(reason="numeric_range")
    if n == 2:
        return SwitchbackMetricFit(point, reason="insufficient_replicates")

    # Refit at zero using the remaining sums; common units do not change on
    # deletion. The determinant is zero exactly when that row has leverage 1.
    deleted: list[Fraction] = []
    for xi, yi in zip(x, y, strict=True):
        sx_i, sy_i = sx - xi, sy - yi
        sxx_i, sxy_i = sxx - xi * xi, sxy - xi * yi
        det_i = (n - 1) * sxx_i - sx_i**2
        if det_i == 0:
            return SwitchbackMetricFit(point, reason="high_leverage")
        deleted.append(Fraction(sxx_i * sy_i - sx_i * sxy_i, det_i))
    return _with_variance(point, center, _jackknife_variance(deleted, unit), n - 2)


def switchback_cuped(
    outcomes: _Vector | np.ndarray, predictors: _Vector | _Rows | np.ndarray
) -> SwitchbackMetricFit:
    """Fit signed HT outcomes on signed, pre-assignment HT forecast contrasts.

    Inputs are n outcomes and one numeric forecast contrast per replicate. The
    predictor must have *design* expectation zero. The caller must establish
    this from the assignment and forecast provenance; neither the observed
    covariate mean nor outcomes establish it.

    Report the intercept at predictor zero, with delete-one-replicate jackknife
    uncertainty and a t approximation on n-2 residual degrees of freedom.
    A static forecast canceling within every replicate has no estimable slope;
    as for any rank deficiency we do not substitute an unadjusted analysis.
    """
    y = np.asarray(outcomes, dtype=object)
    x = np.asarray(predictors, dtype=object)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if y.ndim != 1 or x.ndim != 1 or len(x) != len(y):
        return SwitchbackMetricFit(reason="invalid_input")
    if len(y) < 2:
        return SwitchbackMetricFit(reason="insufficient_replicates")
    x_column = _integer_column(x)
    if isinstance(x_column, str):
        return SwitchbackMetricFit(reason=x_column)
    y_column = _integer_column(y)
    if isinstance(y_column, str):
        return SwitchbackMetricFit(reason=y_column)
    # Scaling the forecast through zero preserves the target intercept.
    return _adjusted_fit(x_column[0], *y_column)
