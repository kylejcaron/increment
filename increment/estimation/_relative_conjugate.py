"""Queries of already-updated, unrestricted relative Normal laws.

For z=(r-m)/s, F(r)=Phi(z) and E[r+]=m*Phi(m/s)+s*phi(m/s).
Mixture probabilities and partial moments are weight-linear. Direct log tails,
local interval integrals and signed tail differences avoid cancellation. Inverse
queries require a value bracket as well as a tail residual or adjacent-float
bracket; floating-point error allowances are estimates, not certificates.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, NoReturn

from scipy.special import log_ndtr, ndtri_exp

from increment.estimation._relative_posterior import (
    _finite_endpoint,
    _invalid,
    _normal_loss_log,
    _PosteriorNumericalError,
    _unavailable,
)
from increment.estimation.priors import MixturePosterior, Normal

_Component = tuple[float, float, float]
_LOG2 = math.log(2)
_LOG2PI = math.log(2 * math.pi)
_LOGMAX = math.log(sys.float_info.max)
_LOGMIN = math.log(math.ulp(0.0))
_EPS = sys.float_info.epsilon
_MAX = Fraction(sys.float_info.max)


def _standardized(value: float, mean: float, sigma: float) -> float:
    difference = value - mean
    if math.isfinite(difference) or not math.isfinite(value):
        return difference / sigma
    # An overflowing difference has opposite-signed terms, so division first
    # cannot introduce cancellation or lose a finite standardized difference.
    return value / sigma - mean / sigma


def _logsum(values: list[float]) -> float:
    high = max(values, default=-math.inf)
    if high == -math.inf:
        return high
    return high + math.log(math.fsum(math.exp(value - high) for value in values))


def _midpoint(lower: float, upper: float) -> float:
    width = upper - lower
    return lower + width / 2 if math.isfinite(width) else lower / 2 + upper / 2


def _value_tolerance(lower: float, upper: float) -> float:
    minimum = 0.0 if lower <= 0 <= upper else min(abs(lower), abs(upper))
    return 1e-9 + 1e-6 * minimum


def _interval_log(lower: float, upper: float, mean: float, sigma: float) -> float:
    zl, zu = _standardized(lower, mean, sigma), _standardized(upper, mean, sigma)
    if lower == -math.inf:
        return float(log_ndtr(zu))
    if upper == math.inf:
        return float(log_ndtr(-zl))
    width = _standardized(upper, lower, sigma)
    midpoint = zl + width / 2
    if width == 0:
        return -math.inf
    if math.isfinite(midpoint) and width * max(1, abs(midpoint)) <= 1e-3:
        # Three-point Gaussian integration; the sixth-order remainder at this
        # width is below the probability error allowance. Preserve physical
        # width even when the two standardized endpoints round identically.
        delta = math.sqrt(3 / 5) * width / 2
        correction = 4 / 9 + 5 / 9 * math.exp(-delta * delta / 2) * math.cosh(midpoint * delta)
        return math.log(width) - midpoint * midpoint / 2 - _LOG2PI / 2 + math.log(correction)
    if zl >= 0:
        big, small = float(log_ndtr(-zl)), float(log_ndtr(-zu))
    elif zu <= 0:
        big, small = float(log_ndtr(zu)), float(log_ndtr(zl))
    else:
        return math.log((math.erf(zu / math.sqrt(2)) - math.erf(zl / math.sqrt(2))) / 2)
    return big + math.log(-math.expm1(small - big)) if big > small else -math.inf


class _ConjugateQueries(ABC):
    @abstractmethod
    def _components(self) -> Iterator[_Component]: ...

    @abstractmethod
    def _parameters(self) -> dict[str, object]: ...

    def _fail(self, operation: str, reason: str, **query: object) -> NoReturn:
        _unavailable(operation, reason, **query, **self._parameters())

    def _tail(self, value: float, *, survival: bool) -> float:
        sign = -1 if survival else 1
        return math.exp(
            _logsum(
                [
                    math.log(weight) + float(log_ndtr(sign * _standardized(value, mean, sigma)))
                    for weight, mean, sigma in self._components()
                ]
            )
        )

    def cdf(self, value: float) -> float:
        """Probability at or below an unrestricted relative threshold."""
        if math.isnan(value):
            _invalid("cdf", "threshold must not be NaN", threshold=value)
        return self._tail(value, survival=False)

    def survival(self, value: float) -> float:
        """Probability above a threshold, retaining the direct small tail."""
        if math.isnan(value):
            _invalid("survival", "threshold must not be NaN", threshold=value)
        return self._tail(value, survival=True)

    def probability_between(self, lower: float, upper: float) -> float:
        """Interval mass from per-component, cancellation-safe integrals."""
        if math.isnan(lower) or math.isnan(upper) or lower > upper:
            _invalid(
                "probability_between",
                "bounds must be ordered and not NaN",
                lower=lower,
                upper=upper,
            )
        if lower == upper:
            return 0.0
        return math.exp(
            _logsum(
                [
                    math.log(weight) + _interval_log(lower, upper, mean, sigma)
                    for weight, mean, sigma in self._components()
                ]
            )
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
        terms = []
        try:
            for weight, mean, sigma in self._components():
                directed = mean if positive else -mean
                z = directed / sigma
                log_density_scale = math.log(sigma) - 0.5 * z * z - _LOG2PI / 2
                # For a negative mean the loss is bounded by sigma*phi(z).
                # Omit only a term that rounds to zero even before weighting.
                if z < 0 and log_density_scale < _LOGMIN - _LOG2:
                    continue
                # The central kernel uses only the nonnegative mean summand;
                # keep it nonnegative when a negative ratio rounds to -0.0.
                terms.append(
                    math.log(weight) + _normal_loss_log(max(0.0, directed), math.log(sigma), z)
                )
        except _PosteriorNumericalError as error:
            self._fail(operation, str(error.context["reason"]), scale=scale)
        logarithm = _logsum(terms)
        if logarithm > _LOGMAX:
            self._fail(operation, "result_unrepresentable", scale=scale)
        return math.exp(logarithm)

    def quantile(self, probability: float) -> float:
        """Invert a cumulative probability with value and tail checks."""
        if not 0 < probability < 1:
            _invalid("quantile", "probability must be in (0, 1)", probability=probability)
        return self._invert(probability, survival=False)

    def isf(self, tail_probability: float) -> float:
        """Invert an explicit upper-tail probability without complement rounding."""
        if not 0 < tail_probability < 1:
            _invalid("isf", "tail probability must be in (0, 1)", tail_probability=tail_probability)
        return self._invert(tail_probability, survival=True)

    def _comparison(self, value: float, alpha: float, *, survival: bool) -> int:
        """Sign of the increasing root function, without subtracting final CDFs."""
        baseline = [-alpha]
        positive, negative = [], []
        for weight, mean, sigma in self._components():
            z = _standardized(value, mean, sigma) * (-1 if survival else 1)
            tail = math.log(weight) + float(log_ndtr(-abs(z)))
            if z >= 0:
                baseline.append(weight)
                negative.append(tail)
            else:
                positive.append(tail)
        constant = math.fsum(baseline)
        if constant > 0:
            positive.append(math.log(constant))
        elif constant < 0:
            negative.append(math.log(-constant))
        first, second = _logsum(positive), _logsum(negative)
        sign = int(first > second) - int(first < second)
        return -sign if survival else sign

    def _accepted(self, value: float, alpha: float, *, survival: bool) -> bool:
        if abs(self._tail(value, survival=survival) - alpha) <= min(1e-10, 1e-6 * alpha):
            return True
        lower, upper = math.nextafter(value, -math.inf), math.nextafter(value, math.inf)
        if upper - lower > _value_tolerance(lower, upper):
            return False
        left = self._comparison(lower, alpha, survival=survival)
        right = self._comparison(upper, alpha, survival=survival)
        return left <= 0 <= right and left < right

    def _invert(self, alpha: float, *, survival: bool) -> float:
        operation = "isf" if survival else "quantile"
        query = {"tail_probability" if survival else "probability": alpha}
        if alpha > 0.5:
            alpha, survival = 1 - alpha, not survival
        radius = Fraction(-float(ndtri_exp(math.log(alpha) - _LOG2)))
        lower = min(
            _finite_endpoint(Fraction(mean) - Fraction(sigma) * radius, upper=False)
            for _, mean, sigma in self._components()
        )
        upper = max(
            _finite_endpoint(Fraction(mean) + Fraction(sigma) * radius, upper=True)
            for _, mean, sigma in self._components()
        )
        if (
            self._comparison(lower, alpha, survival=survival) > 0
            or self._comparison(upper, alpha, survival=survival) < 0
        ):
            self._fail(operation, "result_unrepresentable", **query)
        # Physical bisection avoids an overflowing span or a coordinate scaling
        # that erases small effects. 2200 steps cover the binary64 exponent range.
        for _ in range(2200):
            result = _midpoint(lower, upper)
            tolerance = _value_tolerance(lower, upper)
            if max(result - lower, upper - result) <= tolerance and self._accepted(
                result, alpha, survival=survival
            ):
                return result
            comparison = self._comparison(result, alpha, survival=survival)
            if comparison == 0:
                # A rounded plateau is not a value bracket. Local signs must
                # independently resolve before a zero residual can be accepted.
                distance = _value_tolerance(result, result) / 4
                local_lower = max(lower, math.nextafter(result - distance, -math.inf))
                local_upper = min(upper, math.nextafter(result + distance, math.inf))
                if (
                    self._comparison(local_lower, alpha, survival=survival)
                    < 0
                    < self._comparison(local_upper, alpha, survival=survival)
                ):
                    lower, upper = local_lower, local_upper
                    continue
                self._fail(operation, "accuracy_unresolved", **query)
            if result == lower or result == upper:
                self._fail(operation, "accuracy_unresolved", **query)
            if comparison < 0:
                lower = result
            else:
                upper = result
        self._fail(operation, "accuracy_unresolved", **query)


@dataclass(frozen=True)
class RelativeNormalPosterior(_ConjugateQueries):
    """Relative-effect queries retaining an already-updated Normal law."""

    normal: Normal

    def _components(self) -> Iterator[_Component]:
        yield 1.0, self.normal.mu, self.normal.sigma

    def _parameters(self) -> dict[str, object]:
        return {"mean": self.normal.mu, "sigma": self.normal.sigma}

    def _invert(self, alpha: float, *, survival: bool) -> float:
        operation = "isf" if survival else "quantile"
        query = {"tail_probability" if survival else "probability": alpha}
        if alpha > 0.5:
            alpha, survival = 1 - alpha, not survival
        z = float(ndtri_exp(math.log(alpha))) * (-1 if survival else 1)
        exact = Fraction(self.normal.mu) + Fraction(self.normal.sigma) * Fraction(z)
        if abs(exact) > _MAX:
            self._fail(operation, "result_unrepresentable", **query)
        result = float(exact)
        # Inverse-Normal rounding and affine rounding must fit the value budget,
        # including near cancellation where a small residual alone proves little.
        error = self.normal.sigma * (8 * _EPS * abs(z)) + math.ulp(result)
        if error > _value_tolerance(result, result) or not self._accepted(
            result, alpha, survival=survival
        ):
            self._fail(operation, "accuracy_unresolved", **query)
        return result


@dataclass(frozen=True)
class RelativeMixturePosterior(_ConjugateQueries):
    """Relative-effect queries retaining the updated mixture arrays by reference."""

    mixture: MixturePosterior

    def _components(self) -> Iterator[_Component]:
        for weight, mean, sigma in zip(
            self.mixture.weights, self.mixture.means, self.mixture.sigmas, strict=True
        ):
            if weight > 0:
                yield float(weight), float(mean), float(sigma)

    def _parameters(self) -> dict[str, object]:
        return {
            "weights": tuple(float(value) for value in self.mixture.weights),
            "means": tuple(float(value) for value in self.mixture.means),
            "sigmas": tuple(float(value) for value in self.mixture.sigmas),
        }
