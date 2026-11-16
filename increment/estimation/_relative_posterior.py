"""Continuous latent-precision Student-t posterior under a relative Normal likelihood.

For v=log(lambda*s²/a²), the conditional effect is Normal with mean
x*sigmoid(-v) and deviation s*sqrt(sigmoid(-v)). Its mixing log density is
nu*v/2-softplus(-v)/2-Q*exp(v)/2-t²*sigmoid(v)/2. Positive-kernel integration
retains the real-line target. Quadrature errors are estimates, not certificates.
"""

from __future__ import annotations

import math
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import erfcx, log_ndtr, ndtri_exp

from increment.errors import CodedError, CodedModel, InvalidRequestError, RefusalSpec, refuse
from increment.estimation.priors import StudentTPrior

_LOG2 = math.log(2)
_LOG2PI = math.log(2 * math.pi)
_LOGMAX = math.log(sys.float_info.max)
_EPS = sys.float_info.epsilon
_Pair = tuple[float, float]
_RationalPair = tuple[Fraction, Fraction]


class _PosteriorNumericalError(CodedError):
    """A posterior operation cannot meet its numerical accuracy requirements."""


_NUMERICAL = RefusalSpec(
    "estimation.posterior.numerical_unavailable",
    _PosteriorNumericalError,
    template="Posterior {operation} is numerically unavailable: {reason}",
)
_INVALID = RefusalSpec(
    "estimation.posterior.invalid_request",
    InvalidRequestError,
    template="Invalid posterior {operation}: {reason}",
)


def _unavailable(operation: str, reason: str, **parameters: object) -> NoReturn:
    refuse(_NUMERICAL, operation=operation, reason=reason, **parameters)


def _invalid(operation: str, reason: str, **parameters: object) -> NoReturn:
    refuse(_INVALID, operation=operation, reason=reason, **parameters)


class RelativeStudentTState(CodedModel, BaseModel):
    """Versioned defining density; integration work is never serialized."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False, extra="forbid")

    version: Literal[1]
    likelihood_point: float
    likelihood_se: float = Field(gt=0)
    prior: StudentTPrior

    @model_validator(mode="after")
    def _finite_prior(self) -> RelativeStudentTState:
        if not (math.isfinite(self.prior.nu) and math.isfinite(self.prior.scale)):
            _invalid("state", "prior parameters must be finite")
        return self


def _softplus(value: float) -> float:
    return max(value, 0) + math.log1p(math.exp(-abs(value)))


def _exp(value: float) -> float:
    return math.exp(value) if value <= _LOGMAX else math.inf


def _logadd(first: float, second: float) -> float:
    high, low = max(first, second), min(first, second)
    return high if low == -math.inf else high + math.log1p(math.exp(low - high))


def _polynomial(coefficients: tuple[Fraction, ...], value: Fraction) -> Fraction:
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = result * value + coefficient
    return result


def _remainder(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    result = first.copy()
    while len(result) >= len(second) and any(result):
        factor = result[-1] / second[-1]
        offset = len(result) - len(second)
        for index, coefficient in enumerate(second):
            result[offset + index] -= factor * coefficient
        while len(result) > 1 and result[-1] == 0:
            result.pop()
    return result


def _repeated_roots(coefficients: tuple[Fraction, ...]) -> list[Fraction] | None:
    first = list(coefficients)
    second = [index * coefficients[index] for index in range(1, len(coefficients))]
    while any(second):
        first, second = second, _remainder(first, second)
    if len(first) == 1:
        return None
    if len(first) == 2:
        double = -first[0] / first[1]
        remaining = -coefficients[2] / coefficients[3] - 2 * double
        return sorted({root for root in (double, remaining) if root > 0})
    triple = -coefficients[2] / (3 * coefficients[3])
    return [triple] if triple > 0 else []


def _interval_polynomial(
    coefficients: tuple[Fraction, ...], bounds: _RationalPair
) -> _RationalPair:
    low = high = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        products = (low * bounds[0], low * bounds[1], high * bounds[0], high * bounds[1])
        low, high = min(products) + coefficient, max(products) + coefficient
    return low, high


def _turning_intervals(
    coefficients: tuple[Fraction, ...], digits: int
) -> list[_RationalPair] | None:
    _, c1, c2, negative_q = coefficients
    q = -negative_q
    discriminant = c2 * c2 + 3 * q * c1
    if not (c1 < 0 and c2 > 0 and discriminant > 0):
        return []
    scale = 10**digits
    root = math.isqrt(discriminant.numerator * scale * scale // discriminant.denominator)
    low, high = Fraction(root, scale), Fraction(root + 1, scale)
    intervals = [
        (-c1 / (c2 + high), -c1 / (c2 + low)),
        ((c2 + low) / (3 * q), (c2 + high) / (3 * q)),
    ]
    if intervals[0][1] >= intervals[1][0]:
        return None
    for bounds in intervals:
        lower, upper = _interval_polynomial(coefficients, bounds)
        if lower <= 0 <= upper:
            return None
    return intervals


def _log_enclosure(bounds: _RationalPair, digits: int) -> _Pair:
    endpoints = []
    with localcontext() as context:
        context.prec = digits
        for value, rounding, direction in zip(
            bounds, (ROUND_FLOOR, ROUND_CEILING), (-math.inf, math.inf), strict=True
        ):
            context.rounding = rounding
            decimal = Decimal(value.numerator) / Decimal(value.denominator)
            logarithm = decimal.ln()
            logarithm = logarithm.next_minus() if direction < 0 else logarithm.next_plus()
            endpoints.append(math.nextafter(float(logarithm), direction))
    return endpoints[0], endpoints[1]


def _bisect_root(
    coefficients: tuple[Fraction, ...], low: Fraction, high: Fraction
) -> _RationalPair:
    f_low = _polynomial(coefficients, low)
    for _ in range(8192):
        if low > 0 and high - low <= low / (1 << 75):
            return low, high
        # Geometric localization prevents extreme scale ratios taking thousands
        # of arithmetic bisections before a small root becomes visible.
        if low > 0 and high > 4 * low:
            exponent = (
                (low.numerator.bit_length() - low.denominator.bit_length())
                + (high.numerator.bit_length() - high.denominator.bit_length())
            ) // 2
            middle = Fraction(2) ** exponent
            if not low < middle < high:
                middle = (low + high) / 2
        elif low == 0 and high > 1:
            middle = Fraction(1)
        else:
            middle = (low + high) / 2
        value = _polynomial(coefficients, middle)
        if value == 0:
            return middle, middle
        if f_low * value < 0:
            high = middle
        else:
            low, f_low = middle, value
    _unavailable("construction", "stationary_partition_unresolved")


def _stationary_partition(coefficients: tuple[Fraction, ...]) -> tuple[list[_Pair], list[_Pair]]:
    """Isolate every positive cubic root using exact or enclosed signs."""
    repeated = _repeated_roots(coefficients)
    if repeated is not None:
        _, _, c2, negative_q = coefficients
        discriminant = c2 * c2 - 3 * negative_q * coefficients[1]
        square_root = Fraction(
            math.isqrt(discriminant.numerator), math.isqrt(discriminant.denominator)
        )
        if square_root * square_root != discriminant:
            _unavailable("construction", "stationary_partition_unresolved")
        turns = sorted(
            {(c2 - square_root) / (-3 * negative_q), (c2 + square_root) / (-3 * negative_q)}
        )
        return (
            [_log_enclosure((root, root), 80) for root in repeated],
            [_log_enclosure((root, root), 80) for root in turns if root > 0],
        )
    for digits in (80, 160, 320, 640):
        turns = _turning_intervals(coefficients, digits)
        if turns is None:
            continue
        upper = 1 + max(abs(coefficient / coefficients[-1]) for coefficient in coefficients[:-1])
        boundaries = [Fraction(0), *((low + high) / 2 for low, high in turns), upper]
        roots = []
        for low, high in zip(boundaries, boundaries[1:], strict=False):
            if _polynomial(coefficients, low) * _polynomial(coefficients, high) < 0:
                roots.append(_log_enclosure(_bisect_root(coefficients, low, high), digits))
        if roots and all(
            first[1] < second[0] for first, second in zip(roots, roots[1:], strict=False)
        ):
            turning_logs = [_log_enclosure(bounds, digits) for bounds in turns]
            return roots, turning_logs
    _unavailable("construction", "stationary_partition_unresolved")


@dataclass(frozen=True)
class _Integral:
    value: float
    error: float


@dataclass(frozen=True)
class _Estimate:
    value: float
    lower: float
    upper: float

    @property
    def error(self) -> float:
        return max(self.value - self.lower, self.upper - self.value)


def _normal_loss_log(mean: float, log_deviation: float, standardized_mean: float) -> float:
    """Log positive Normal part, with an alternating Mills remainder."""
    z = standardized_mean
    log_density = -0.5 * z * z - 0.5 * _LOG2PI
    if z >= 0:
        log_mean = -math.inf if mean == 0 else math.log(mean)
        return _logadd(log_mean + float(log_ndtr(z)), log_deviation + log_density)
    z = -z
    if z < 10:
        mills = math.sqrt(math.pi / 2) * float(erfcx(z / math.sqrt(2)))
        g = -math.expm1(math.log(z * mills))
    else:
        inverse = (1 / z) ** 2
        term = inverse
        terms = [term]
        for index in range(1, 200):
            following = -term * (2 * index + 1) * inverse
            if abs(following) >= abs(term):
                _unavailable("loss", "accuracy_unresolved")
            if abs(following) <= 4 * _EPS * abs(math.fsum(terms)):
                break
            terms.append(following)
            term = following
        else:
            _unavailable("loss", "accuracy_unresolved")
        g = math.fsum(terms)
    if g <= 0:
        _unavailable("loss", "accuracy_unresolved")
    return log_deviation + log_density + math.log(g)


def _signed_sum(first: float, first_log: float, second: float, second_log: float) -> float:
    if first == 0:
        return math.copysign(_exp(second_log), second) if second else 0
    if second == 0:
        return math.copysign(_exp(first_log), first)
    if first_log < second_log:
        first, second, first_log, second_log = second, first, second_log, first_log
    if math.copysign(1, first) == math.copysign(1, second):
        logarithm = _logadd(first_log, second_log)
    elif first_log == second_log:
        return 0
    else:
        logarithm = first_log + math.log(-math.expm1(second_log - first_log))
    return math.copysign(_exp(logarithm), first)


class _Workspace:
    def __init__(self, state: RelativeStudentTState):
        self.x = state.likelihood_point
        self.s = state.likelihood_se
        self.a = state.prior.scale
        self.nu = state.prior.nu
        self.t = self.x / self.s
        if not math.isfinite(self.t):
            _unavailable(
                "construction",
                "standardized_likelihood_unrepresentable",
                **state.model_dump(exclude={"prior"}),
            )
        self.log_t = -math.inf if self.x == 0 else math.log(abs(self.x)) - math.log(self.s)
        self.log_a = 2 * self.log_t - _LOG2
        self.log_b = math.log(self.nu) + 2 * (math.log(self.a) - math.log(self.s)) - _LOG2
        n = Fraction(self.nu)
        q = n * (Fraction(self.a) / Fraction(self.s)) ** 2
        t2 = (Fraction(self.x) / Fraction(self.s)) ** 2
        roots, turns = _stationary_partition((n + 1, 2 * n + 1 - q - t2, n - 2 * q, -q))
        self.boundaries = tuple(sorted({value for bounds in (*roots, *turns) for value in bounds}))
        centers = [(low + high) / 2 for low, high in roots]
        self.reference = centers[0]
        for value in centers[1:]:
            if self.difference(value, self.reference) > 0:
                self.reference = value
        curvature = 2 * _exp(self.log_b + self.reference) + 2 * _exp(
            self.log_a - _softplus(self.reference) - _softplus(-self.reference)
        )
        delta = 1 / math.sqrt(max(1, curvature))
        bound = 0.125 + math.exp(delta) * curvature / 2
        self.log_zstar = (
            math.log(2 * delta)
            - abs(self.derivative(self.reference)) * delta
            - bound * delta**2 / 2
        )
        self.zstar = _exp(self.log_zstar)
        if not self.zstar > 0 or not math.isfinite(self.zstar):
            _unavailable("construction", "accuracy_unresolved")
        logsum = _logadd(0, _logadd(self.log_a + _LOG2, self.log_b + _LOG2))
        left = min(0, math.log(self.nu + 1) - math.log(4) - logsum)
        right = math.log(self.nu + 2) - self.log_b - _LOG2
        self.left, self.right = self.limits(
            left, right, math.log(1e-24) + self.log_zstar, loss=None
        )
        self.log_tail = _logadd(*self.tail_logs(self.left, self.right, loss=None))
        self.normalizer = self.integrate(
            lambda _: 0, self.left, self.right, 0, math.log(1e-14) + self.log_zstar, ()
        )
        if self.normalizer.value <= self.normalizer.error or self.normalizer.value < self.zstar:
            _unavailable("construction", "accuracy_unresolved")

    def derivative(self, value: float) -> float:
        return (
            self.nu / 2
            + math.exp(-_softplus(value)) / 2
            - _exp(self.log_b + value)
            - _exp(self.log_a - _softplus(value) - _softplus(-value))
        )

    def difference(self, value: float, center: float) -> float:
        delta = value - center
        if delta == 0:
            return 0
        logarithm = math.log(-math.expm1(-abs(delta)))
        if delta > 0:
            exponential = _exp(self.log_b + value + logarithm)
            sigmoid = _exp(self.log_a - _softplus(-value) - _softplus(center) + logarithm)
        else:
            exponential = -_exp(self.log_b + center + logarithm)
            sigmoid = -_exp(self.log_a - _softplus(-center) - _softplus(value) + logarithm)
        return math.fsum(
            (
                self.nu * delta / 2,
                -(_softplus(-value) - _softplus(-center)) / 2,
                -exponential,
                -sigmoid,
            )
        )

    def tail_logs(self, left: float, right: float, *, loss: bool | None) -> _Pair:
        dl = (
            (self.nu + 1) / 2
            - math.exp(-_softplus(-left)) / 2
            - _exp(self.log_b + left)
            - _exp(self.log_a - _softplus(-left))
        )
        dr = _exp(self.log_b + right) - self.nu / 2 - math.exp(-_softplus(right)) / 2
        ll = (
            math.inf
            if left > 0 or dl <= 0
            else self.difference(left, self.reference) - math.log(dl)
        )
        lr = math.inf if dr <= 0 else self.difference(right, self.reference) - math.log(dr)
        if loss is None:
            return ll, lr
        signed = self.t if loss else -self.t
        log_mean = math.log(signed) if signed > 0 else -math.inf
        ll += _logadd(log_mean, -0.5 * _LOG2PI)
        rho = math.exp(-_softplus(-right))
        gamma_log = -_softplus(right)
        lr = (
            self.difference(right, self.reference)
            + _logadd(
                log_mean + gamma_log - math.log(dr + rho),
                gamma_log / 2 - 0.5 * _LOG2PI - math.log(dr + rho / 2),
            )
            if dr > 0
            else math.inf
        )
        return ll, lr

    def limits(self, left: float, right: float, goal: float, *, loss: bool | None) -> _Pair:
        for _ in range(2048):
            ll, lr = self.tail_logs(left, right, loss=loss)
            if _logadd(ll, lr) <= goal:
                return left, right
            if ll > goal - _LOG2:
                left -= 4
            if lr > goal - _LOG2:
                right += 2
        _unavailable("integration", "accuracy_unresolved")

    def knots(self, left: float, right: float, extra: tuple[float, ...]) -> list[float]:
        count = math.ceil(right - left)
        if count > 16384 or count < 1:
            _unavailable("integration", "accuracy_unresolved")
        return sorted(
            {
                left,
                right,
                *self.boundaries,
                *extra,
                *(left + (right - left) * index / count for index in range(1, count)),
            }
        )

    def integrate(
        self,
        kernel: Callable[[float], float],
        left: float,
        right: float,
        log_scale: float,
        log_goal: float,
        extra: tuple[float, ...],
    ) -> _Integral:
        knots = self.knots(left, right, extra)
        absolute = _exp(min(0, log_goal - log_scale)) / len(knots)
        values, errors = [], []

        def integrand(value: float) -> float:
            return _exp(self.difference(value, self.reference) + kernel(value) - log_scale)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                for low, high in zip(knots, knots[1:], strict=False):
                    width = high - low
                    value, error = quad(
                        lambda u, width=width, low=low: width * integrand(low + width * u),
                        0,
                        1,
                        epsabs=absolute,
                        epsrel=3e-12,
                        limit=100,
                    )
                    if not (
                        math.isfinite(value) and math.isfinite(error) and value >= 0 and error >= 0
                    ):
                        _unavailable("integration", "quadrature_nonconvergence")
                    values.append(value)
                    errors.append(error)
        except (IntegrationWarning, OverflowError, ValueError) as error:
            if isinstance(error, CodedError):
                raise
            _unavailable("integration", "quadrature_nonconvergence")
        value = math.fsum(values)
        return _Integral(value, math.fsum(errors) + 32 * _EPS * value)

    def threshold(self, threshold: float, value: float) -> float:
        half = _softplus(value) / 2
        if threshold == 0:
            return -self.t * math.exp(-half)
        difference = threshold - self.x
        if math.isfinite(difference) and abs(difference) < 0.5 * abs(self.x):
            first, second = difference, self.x
            first_log = -math.inf if first == 0 else math.log(abs(first)) - math.log(self.s) + half
            second_log = self.log_t + value - half
        else:
            first, second = threshold, -self.x
            first_log = math.log(abs(first)) - math.log(self.s) + half
            second_log = self.log_t - half
        return _signed_sum(first, first_log, second, second_log)

    def crossings(self, thresholds: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(
            math.log(abs(self.x - h)) - math.log(abs(h))
            for h in thresholds
            if min(0, self.x) < h < max(0, self.x)
        )

    def loss_kernel(self, value: float, *, positive: bool) -> float:
        log_d = -_softplus(value) / 2
        signed = self.t if positive else -self.t
        mean = signed * math.exp(-_softplus(value))
        return _normal_loss_log(mean, log_d, signed * math.exp(log_d))

    def estimate(
        self,
        kernel: Callable[[float], float],
        *,
        loss: bool | None = None,
        thresholds: tuple[float, ...] = (),
        relative: bool = False,
    ) -> _Estimate:
        unit = 0 if loss is None else math.log(self.s)
        low_z = max(self.zstar, self.normalizer.value - self.normalizer.error)
        high_z = self.normalizer.value + self.normalizer.error + _exp(self.log_tail)
        left, right = self.left, self.right
        extra = self.crossings(thresholds)
        tolerance_log = math.log(1e-10)
        for _ in range(16):
            knots = self.knots(left, right, extra)
            samples = [self.difference(v, self.reference) + kernel(v) for v in knots]
            log_scale = max(samples)
            if not math.isfinite(log_scale):
                _unavailable("integration", "accuracy_unresolved")
            # A positive pilot supplies the loss-relative budget before extending
            # a potentially enormous original-unit numerator's integration range.
            goal = tolerance_log - unit + self.log_zstar - math.log(32)
            numerator = self.integrate(kernel, left, right, log_scale, goal, extra)

            def rescale(value: float, log_scale: float = log_scale) -> float:
                return 0 if value == 0 else _exp(math.log(value) + log_scale + unit)

            lower = rescale(max(0, numerator.value - numerator.error) / high_z)
            if loss is not None:
                requested_log = _logadd(
                    math.log(1e-10), math.log(lower) + math.log(1e-6) if lower else -math.inf
                )
            elif lower and (relative or lower <= 1e-4):
                requested_log = min(math.log(1e-10), math.log(lower) + math.log(1e-6))
            else:
                requested_log = math.log(1e-10) if not relative else tolerance_log - _LOG2
            goal = requested_log - unit + self.log_zstar - math.log(32)
            new_left, new_right = self.limits(left, right, goal, loss=loss)
            if (new_left, new_right) != (left, right) or requested_log < tolerance_log - _LOG2:
                left, right, tolerance_log = new_left, new_right, requested_log
                continue
            omitted_log = _logadd(*self.tail_logs(left, right, loss=loss))
            point = rescale(numerator.value / self.normalizer.value)
            omitted = _exp(omitted_log + unit - math.log(low_z))
            upper = rescale((numerator.value + numerator.error) / low_z) + omitted
            if not all(math.isfinite(v) for v in (point, lower, upper)):
                _unavailable("integration", "result_unrepresentable")
            contributions = (
                rescale(numerator.error / low_z),
                omitted,
                point * (self.normalizer.error / low_z),
                _exp(math.log(point) + self.log_tail - math.log(low_z)) if point else 0,
            )
            requested = _exp(requested_log)
            # Reserve for log kernels, local interval quadrature and summation.
            rounding = 64 * _EPS * point
            lower = max(0, lower - rounding)
            upper += rounding
            if (
                max(contributions) <= requested / 8
                and max(point - lower, upper - point) <= requested
            ):
                if loss is None:
                    return _Estimate(min(1, point), min(1, lower), min(1, upper))
                return _Estimate(point, lower, upper)
            tolerance_log = requested_log - _LOG2
        _unavailable("integration", "accuracy_unresolved")


def _interval_log(workspace: _Workspace, lower: float, upper: float, value: float) -> float:
    if lower == -math.inf:
        return float(log_ndtr(workspace.threshold(upper, value)))
    if upper == math.inf:
        return float(log_ndtr(-workspace.threshold(lower, value)))
    zl, zu = workspace.threshold(lower, value), workspace.threshold(upper, value)
    width = _exp(math.log(upper - lower) - math.log(workspace.s) + _softplus(value) / 2)
    midpoint = zl + width / 2
    if math.isfinite(midpoint) and width * max(1, abs(midpoint)) <= 1e-3:
        delta = math.sqrt(3 / 5) * width / 2
        correction = 4 / 9 + 5 / 9 * math.exp(-delta * delta / 2) * math.cosh(midpoint * delta)
        return (
            math.log(width) - midpoint * midpoint / 2 - _LOG2PI / 2 + math.log(correction)
            if width
            else -math.inf
        )
    if zl >= 0:
        big, small = float(log_ndtr(-zl)), float(log_ndtr(-zu))
    elif zu <= 0:
        big, small = float(log_ndtr(zu)), float(log_ndtr(zl))
    else:
        return math.log((math.erf(zu / math.sqrt(2)) - math.erf(zl / math.sqrt(2))) / 2)
    return big + math.log(-math.expm1(small - big)) if big > small else -math.inf


def _finite_float(value: Fraction) -> float:
    try:
        return float(value)
    except OverflowError:
        return sys.float_info.max if value > 0 else -sys.float_info.max


@dataclass(frozen=True)
class RelativeStudentTPosterior:
    """Value-only posterior with a runtime-only normalization workspace."""

    state: RelativeStudentTState
    _workspace: _Workspace = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_workspace", _Workspace(self.state))

    def _query(
        self, operation: str, parameters: dict[str, object], function: Callable[[], float]
    ) -> float:
        try:
            return function()
        except _PosteriorNumericalError as error:
            _unavailable(
                operation,
                str(error.context["reason"]),
                **parameters,
                likelihood_point=self.state.likelihood_point,
                likelihood_se=self.state.likelihood_se,
                prior_nu=self.state.prior.nu,
                prior_scale=self.state.prior.scale,
            )

    def _probability(self, value: float, *, survival: bool, relative: bool = False) -> _Estimate:
        if value == -math.inf:
            result = 1.0 if survival else 0.0
            return _Estimate(result, result, result)
        if value == math.inf:
            result = 0.0 if survival else 1.0
            return _Estimate(result, result, result)
        sign = -1 if survival else 1
        return self._workspace.estimate(
            lambda v: float(log_ndtr(sign * self._workspace.threshold(value, v))),
            thresholds=(value,),
            relative=relative,
        )

    def cdf(self, value: float) -> float:
        """Probability at or below a relative effect threshold."""
        if math.isnan(value):
            _invalid("cdf", "threshold must not be NaN", threshold=value)
        return self._query(
            "cdf", {"threshold": value}, lambda: self._probability(value, survival=False).value
        )

    def survival(self, value: float) -> float:
        """Probability above a relative effect threshold, evaluated directly."""
        if math.isnan(value):
            _invalid("survival", "threshold must not be NaN", threshold=value)
        return self._query(
            "survival", {"threshold": value}, lambda: self._probability(value, survival=True).value
        )

    def probability_between(self, lower: float, upper: float) -> float:
        """Integrate conditional interval masses without final-CDF subtraction."""
        if math.isnan(lower) or math.isnan(upper) or lower > upper:
            _invalid(
                "probability_between",
                "bounds must be ordered and not NaN",
                lower=lower,
                upper=upper,
            )
        if lower == upper:
            return 0
        if lower == -math.inf and upper == math.inf:
            return 1
        return self._query(
            "probability_between",
            {"lower": lower, "upper": upper},
            lambda: (
                self._workspace.estimate(
                    lambda v: _interval_log(self._workspace, lower, upper, v),
                    thresholds=(lower, upper),
                ).value
            ),
        )

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float:
        """Expected positive fractional effect, without exponentiation."""
        return self._loss(positive=True, scale=scale)

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float:
        """Expected negative fractional-effect magnitude, without a unit cap."""
        return self._loss(positive=False, scale=scale)

    def _loss(self, *, positive: bool, scale: str) -> float:
        operation = "expected_positive_part" if positive else "expected_negative_part"
        if scale != "linear":
            _invalid(operation, "relative losses require linear scale", scale=scale)
        return self._query(
            operation,
            {"scale": scale},
            lambda: (
                self._workspace.estimate(
                    lambda v: self._workspace.loss_kernel(v, positive=positive), loss=positive
                ).value
            ),
        )

    def quantile(self, probability: float) -> float:
        """Invert the smaller probability tail with a value-error bracket."""
        if not 0 < probability < 1:
            _invalid("quantile", "probability must be in (0, 1)", probability=probability)
        return self._query(
            "quantile",
            {"probability": probability},
            lambda: self._invert(probability, survival=False),
        )

    def isf(self, tail_probability: float) -> float:
        """Invert an explicit survival probability without complement rounding."""
        if not 0 < tail_probability < 1:
            _invalid("isf", "tail probability must be in (0, 1)", tail_probability=tail_probability)
        return self._query(
            "isf",
            {"tail_probability": tail_probability},
            lambda: self._invert(tail_probability, survival=True),
        )

    def _coordinates(self, alpha: float) -> tuple[float, float, Callable[[float], float]]:
        work = self._workspace
        radius = -float(ndtri_exp(math.log(alpha) - _LOG2))
        low = _finite_float(Fraction(min(0, work.x)) - Fraction(work.s) * Fraction(radius))
        high = _finite_float(Fraction(max(0, work.x)) + Fraction(work.s) * Fraction(radius))
        if work.a >= work.s:

            def coordinate(r: float) -> float:
                return _finite_float((Fraction(r) - Fraction(work.x)) / Fraction(work.s))

            def effect(z: float) -> float:
                product = work.s * z
                if math.isfinite(product) and math.isfinite(work.x + product):
                    return work.x + product
                return _finite_float(Fraction(work.x) + Fraction(work.s) * Fraction(z))
        else:

            def coordinate(r: float) -> float:
                if r == 0:
                    return 0
                logarithm = math.log(abs(r)) - math.log(work.a)
                return math.copysign(
                    logarithm + _LOG2 if logarithm > 350 else math.asinh(math.exp(logarithm)), r
                )

            def effect(z: float) -> float:
                if z == 0:
                    return 0
                absolute = abs(z)
                log_sinh = absolute - _LOG2 + math.log(-math.expm1(-2 * absolute))
                return math.copysign(min(sys.float_info.max, _exp(math.log(work.a) + log_sinh)), z)

        return coordinate(low), coordinate(high), effect

    def _invert(self, alpha: float, *, survival: bool) -> float:
        if alpha > 0.5:
            alpha, survival = 1 - alpha, not survival
        low, high, effect = self._coordinates(alpha)

        def evaluate(z: float) -> _Estimate:
            return self._probability(effect(z), survival=survival, relative=True)

        def bracketed(first: _Estimate, second: _Estimate) -> bool:
            return (
                first.lower >= alpha >= second.upper
                if survival
                else first.upper <= alpha <= second.lower
            )

        # First try a local likelihood bracket, retaining the full sufficient
        # bracket if the prior has shifted the requested quantile outside it.
        if self._workspace.a >= self._workspace.s:
            radius = -float(ndtri_exp(math.log(alpha) - _LOG2))
            local_low, local_high = max(low, -radius), min(high, radius)
            local_left, local_right = evaluate(local_low), evaluate(local_high)
            if bracketed(local_left, local_right):
                low, high = local_low, local_high
        left, right = evaluate(low), evaluate(high)
        if not bracketed(left, right):
            wrong = (
                (left.upper < alpha or right.lower > alpha)
                if survival
                else (left.lower > alpha or right.upper < alpha)
            )
            _unavailable("inversion", "result_unrepresentable" if wrong else "accuracy_unresolved")
        try:
            root = brentq(
                lambda z: evaluate(z).value - alpha,
                low,
                high,
                xtol=math.ulp(0.0),
                rtol=8 * _EPS,
                maxiter=256,
            )
        except (ValueError, RuntimeError):
            _unavailable("inversion", "accuracy_unresolved")
        result = effect(root)
        if not math.isfinite(result):
            _unavailable("inversion", "result_unrepresentable")
        pair = self._probability(result, survival=survival, relative=True)
        tolerance = 1e-9 + 1e-6 * abs(result)
        qlo = math.nextafter(result - tolerance / 4, -math.inf)
        qhi = math.nextafter(result + tolerance / 4, math.inf)
        first = self._probability(qlo, survival=survival, relative=True)
        second = self._probability(qhi, survival=survival, relative=True)
        if not bracketed(first, second):
            _unavailable("inversion", "accuracy_unresolved")
        minimum = 0 if qlo <= 0 <= qhi else min(abs(qlo), abs(qhi))
        if max(result - qlo, qhi - result) > 1e-9 + 1e-6 * minimum:
            _unavailable("inversion", "accuracy_unresolved")
        if abs(pair.value - alpha) + pair.error <= min(1e-10, 1e-6 * alpha):
            return result
        # A probability residual can be impossible at every representable effect.
        # Establish an actual adjacent-float enclosure instead of relaxing it.
        previous, following = math.nextafter(result, -math.inf), math.nextafter(result, math.inf)
        before = self._probability(previous, survival=survival, relative=True)
        after = self._probability(following, survival=survival, relative=True)
        adjacent = bracketed(before, pair) or bracketed(pair, after)
        if adjacent and max(result - previous, following - result) <= 1e-9 + 1e-6 * min(
            abs(previous), abs(following)
        ):
            return result
        _unavailable("inversion", "accuracy_unresolved")
