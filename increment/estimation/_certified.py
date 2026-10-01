"""Exact rational intervals, outward logarithms, and certified real-root isolation.

Inputs must be integers or Fractions; floats require an explicit Fraction(float)
conversion by the caller. Algebra never rounds. Transcendental error is retained
in interval endpoints, using private Decimal contexts and signed Stirling bounds.
Inputs outside Decimal's exponent/precision capacity receive a coded refusal;
there is no floating-point fallback. Root isolation uses exact square-free Sturm counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import (
    MAX_EMAX,
    MAX_PREC,
    MIN_EMIN,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction
from functools import lru_cache
from math import comb, gcd, lcm

from increment.errors import InvalidRequestError, RefusalSpec, refuse

_INEXACT_VALUE = RefusalSpec(
    "estimation.certified.inexact_value",
    InvalidRequestError,
    template="expected an exact Fraction or integer, got {value_type}",
)
_INVALID_PRECISION = RefusalSpec(
    "estimation.certified.invalid_precision",
    InvalidRequestError,
    template="precision must be a positive integer",
    keys=frozenset({"precision"}),
)
_REVERSED_INTERVAL = RefusalSpec(
    "estimation.certified.reversed_interval",
    InvalidRequestError,
    template="interval lower bound exceeds upper bound",
    keys=frozenset({"lower", "upper"}),
)
_ZERO_DIVISOR_INTERVAL = RefusalSpec(
    "estimation.certified.zero_divisor_interval",
    InvalidRequestError,
    template="division by an interval containing zero",
    keys=frozenset({"lower", "upper"}),
)
_DECIMAL_INPUT_EXPONENT_CAPACITY = RefusalSpec(
    "estimation.certified.decimal_input_exponent_capacity",
    InvalidRequestError,
    template="rational input {value!r} exceeds Decimal's exponent range at precision {precision} (working precision {working_precision})",
)
_DECIMAL_LOG_EXPONENT_CAPACITY = RefusalSpec(
    "estimation.certified.decimal_log_exponent_capacity",
    InvalidRequestError,
    template="logarithm of {value!r} exceeds Decimal's exponent range at precision {precision} (working precision {working_precision})",
)
_DECIMAL_ARITHMETIC_CAPACITY = RefusalSpec(
    "estimation.certified.decimal_arithmetic_capacity",
    InvalidRequestError,
    template="logarithm of {value!r} exceeds Decimal's arithmetic capacity at precision {precision} (working precision {working_precision}, signal {signal})",
)
_NONPOSITIVE_LOG_DOMAIN = RefusalSpec(
    "estimation.certified.nonpositive_log_domain",
    InvalidRequestError,
    template="logarithm requires strictly positive bounds",
    keys=frozenset({"lower", "upper"}),
)
_NONPOSITIVE_RISING_VALUE = RefusalSpec(
    "estimation.certified.nonpositive_rising_value",
    InvalidRequestError,
    template="rising factorial requires a positive value",
    keys=frozenset({"value"}),
)
_INVALID_COUNT = RefusalSpec(
    "estimation.certified.invalid_count",
    InvalidRequestError,
    template="count must be a nonnegative integer",
    keys=frozenset({"count"}),
)
_NONPOSITIVE_GAMMA_VALUE = RefusalSpec(
    "estimation.certified.nonpositive_gamma_value",
    InvalidRequestError,
    template="gamma ratio requires a positive value",
    keys=frozenset({"value"}),
)
_NONPOSITIVE_ROOT_WIDTH = RefusalSpec(
    "estimation.certified.nonpositive_root_width",
    InvalidRequestError,
    template="max_width must be positive",
    keys=frozenset({"max_width"}),
)
_REVERSED_ROOT_DOMAIN = RefusalSpec(
    "estimation.certified.reversed_root_domain",
    InvalidRequestError,
    template="lower bound exceeds upper bound",
    keys=frozenset({"lower", "upper"}),
)
_ZERO_POLYNOMIAL = RefusalSpec(
    "estimation.certified.zero_polynomial",
    InvalidRequestError,
    template="the identically zero polynomial has no isolated roots",
    keys=frozenset({"coefficients"}),
)

_DECIMAL_PRECISION_CAPACITY = RefusalSpec(
    "estimation.certified.decimal_precision_capacity",
    InvalidRequestError,
    template="logarithm exceeds Decimal's precision capacity: requested precision {precision} needs working precision {working_precision}, which exceeds the {max_precision} limit",
)


def _rational(value: Fraction | int) -> Fraction:
    if type(value) is Fraction:
        return value
    if not isinstance(value, (Fraction, int)):
        refuse(_INEXACT_VALUE, value_type=type(value).__name__)
    return Fraction(value)


def _precision(precision: int) -> None:
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 1:
        refuse(_INVALID_PRECISION, precision=precision)


@dataclass(frozen=True, slots=True, init=False)
class Interval:
    """A closed, finite rational interval; implicit float ingestion is refused."""

    lo: Fraction
    hi: Fraction

    def __init__(self, lo: Fraction | int, hi: Fraction | int) -> None:
        lo, hi = _rational(lo), _rational(hi)
        if lo > hi:
            refuse(_REVERSED_INTERVAL, lower=lo, upper=hi)
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)

    @classmethod
    def exact(cls, value: Fraction | int) -> Interval:
        """Construct a point interval without rounding its input."""
        return cls(value, value)

    @property
    def width(self) -> Fraction:
        return self.hi - self.lo

    @property
    def midpoint(self) -> Fraction:
        return (self.lo + self.hi) / 2

    def __add__(self, other: Interval | Fraction | int) -> Interval:
        if isinstance(other, Interval):
            return _ordered(self.lo + other.lo, self.hi + other.hi)
        rhs = _rational(other)
        return _ordered(self.lo + rhs, self.hi + rhs)

    def __radd__(self, other: Interval | Fraction | int) -> Interval:
        return self + other

    def __neg__(self) -> Interval:
        return _ordered(-self.hi, -self.lo)

    def __sub__(self, other: Interval | Fraction | int) -> Interval:
        if isinstance(other, Interval):
            return _ordered(self.lo - other.hi, self.hi - other.lo)
        rhs = _rational(other)
        return _ordered(self.lo - rhs, self.hi - rhs)

    def __rsub__(self, other: Fraction | int) -> Interval:
        lhs = _rational(other)
        return _ordered(lhs - self.hi, lhs - self.lo)

    def __mul__(self, other: Interval | Fraction | int) -> Interval:
        if isinstance(other, Interval):
            products = (
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            )
            return _ordered(min(products), max(products))
        rhs = _rational(other)
        products = (self.lo * rhs, self.hi * rhs)
        return _ordered(min(products), max(products))

    def __rmul__(self, other: Interval | Fraction | int) -> Interval:
        return self * other

    def __truediv__(self, other: Interval | Fraction | int) -> Interval:
        rhs = _interval(other)
        if rhs.lo <= 0 <= rhs.hi:
            refuse(_ZERO_DIVISOR_INTERVAL, lower=rhs.lo, upper=rhs.hi)
        return self * Interval(1 / rhs.hi, 1 / rhs.lo)

    def __rtruediv__(self, other: Interval | Fraction | int) -> Interval:
        return _interval(other) / self


def _interval(value: Interval | Fraction | int) -> Interval:
    return value if isinstance(value, Interval) else Interval.exact(value)


def _ordered(lo: Fraction, hi: Fraction) -> Interval:
    """Wrap exact endpoints that arithmetic on valid intervals has already ordered."""
    interval = object.__new__(Interval)
    object.__setattr__(interval, "lo", lo)
    object.__setattr__(interval, "hi", hi)
    return interval


def _log_point(value: Fraction, precision: int) -> Interval:
    if value == 1:
        return Interval.exact(0)
    # Preserve relative accuracy near one, including arguments beyond binary64.
    gap = abs(value.numerator - value.denominator)
    lost_bits = max(0, value.denominator.bit_length() - gap.bit_length())
    digits = precision + 15 + (lost_bits + 2) // 3
    if digits > MAX_PREC:
        refuse(
            _DECIMAL_PRECISION_CAPACITY,
            precision=precision,
            working_precision=digits,
            max_precision=MAX_PREC,
        )
    return _log_point_at(value, digits, precision, MIN_EMIN, MAX_EMAX)


# The same argument at the same working precision recurs constantly: prior
# normalizers, family thresholds and the alpha threshold are re-enclosed at
# every probe of an inversion. The exponent limits are part of the key so a
# changed limit never serves an enclosure certified under another one.
@lru_cache(maxsize=4096)
def _log_point_at(value: Fraction, digits: int, precision: int, emin: int, emax: int) -> Interval:
    try:
        context = Context(
            prec=digits,
            rounding=ROUND_HALF_EVEN,
            Emin=emin,
            Emax=emax,
            capitals=1,
            clamp=0,
            flags=[],
            traps=[InvalidOperation, DivisionByZero, Overflow],
        )
        with localcontext(context) as ctx:
            numerator, denominator = Decimal(value.numerator), Decimal(value.denominator)
            ctx.rounding = ROUND_FLOOR
            lower = ctx.divide(numerator, denominator)
            ctx.rounding = ROUND_CEILING
            upper = ctx.divide(numerator, denominator)
            if lower <= 0 or not upper.is_finite():
                refuse(
                    _DECIMAL_INPUT_EXPONENT_CAPACITY,
                    value=value,
                    precision=precision,
                    working_precision=digits,
                )
            ctx.rounding = ROUND_HALF_EVEN
            # ln is nearest-rounded regardless of ctx.rounding; widen its result.
            lower_log = Decimal(0) if lower == 1 else ctx.ln(lower)
            lo = Decimal(0) if lower == 1 else ctx.next_minus(lower_log)
            if upper == 1:
                hi = Decimal(0)
            elif upper == lower:
                # An exactly representable input needs one logarithm, not two.
                hi = ctx.next_plus(lower_log)
            else:
                hi = ctx.next_plus(ctx.ln(upper))
            if not lo.is_finite() or not hi.is_finite():
                refuse(
                    _DECIMAL_LOG_EXPONENT_CAPACITY,
                    value=value,
                    precision=precision,
                    working_precision=digits,
                )
            return Interval(Fraction(lo), Fraction(hi))
    except DecimalException as exc:
        refuse(
            _DECIMAL_ARITHMETIC_CAPACITY,
            value=value,
            precision=precision,
            working_precision=digits,
            signal=type(exc).__name__,
        )


def log_interval(value: Fraction | Interval, *, precision: int = 60) -> Interval:
    """Enclose ln(value), with directed input division and outward ln ULPs.

    Decimal.ln is correctly rounded to nearest. One outward neighbor on each
    side encloses that rounding error; monotonicity accounts for the separately
    directed rational-to-Decimal conversion. At least 15 guard digits are used.
    """
    _precision(precision)
    interval = _interval(value)
    if interval.lo <= 0:
        refuse(_NONPOSITIVE_LOG_DOMAIN, lower=interval.lo, upper=interval.hi)
    lower = _log_point(interval.lo, precision)
    if interval.lo == interval.hi:
        return lower
    return Interval(lower.lo, _log_point(interval.hi, precision).hi)


def log_rising(value: Fraction, count: int, *, precision: int = 60) -> Interval:
    """Enclose log((value)_count), with work independent of large counts."""
    _precision(precision)
    value = _rational(value)
    if value <= 0:
        refuse(_NONPOSITIVE_RISING_VALUE, value=value)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        refuse(_INVALID_COUNT, count=count)
    if count > 32:
        return _log_gamma_increment(value, Fraction(count), precision)
    product = Fraction(1)
    for i in range(count):
        product *= value + i
    return log_interval(product, precision=precision)


@lru_cache(maxsize=16)
def _stirling_coefficients(order: int) -> tuple[Fraction, ...]:
    """Generate B_(2k)/(2k(2k-1)) via sum C(m+1,j) B_j = 0."""
    bernoulli = [Fraction(1), Fraction(-1, 2)]
    for m in range(2, 2 * order + 1):
        if m % 2:
            bernoulli.append(Fraction(0))
        else:
            total = sum(
                (comb(m + 1, j) * b for j, b in enumerate(bernoulli) if b),
                Fraction(0),
            )
            bernoulli.append(-total / (m + 1))
    return tuple(bernoulli[2 * k] / (2 * k * (2 * k - 1)) for k in range(1, order + 1))


def _stirling_tail(
    value: Fraction, coefficients: tuple[Fraction, ...], tolerance: Fraction
) -> Interval | None:
    total = Fraction(0)
    power = 1 / value
    step = power * power
    for coefficient in coefficients:
        term = coefficient * power
        if abs(term) <= tolerance:
            return Interval(total + min(0, term), total + max(0, term))
        total += term
        power *= step
    return None


def log_gamma_half_step(value: Fraction, *, precision: int = 60) -> Interval:
    """Enclose logGamma(value+1/2)-logGamma(value), width <= 10**(-precision)."""
    _precision(precision)
    value = _rational(value)
    if value <= 0:
        refuse(_NONPOSITIVE_GAMMA_VALUE, value=value)
    return _log_gamma_increment(value, Fraction(1, 2), precision)


@lru_cache(maxsize=128)
def _log_gamma_increment(value: Fraction, increment: Fraction, precision: int) -> Interval:
    """Shift both arguments, then subtract certified Stirling expansions.

    The shared log(2*pi)/2 cancels. Positive-argument remainders lie between
    zero and their signed first omitted term: https://dlmf.nist.gov/5.11#ii.
    Recurrence undoes the common shift: https://dlmf.nist.gov/5.5, as one
    exact rational product enclosed by a single logarithm.
    At fixed order the remainder decreases as x**(-(2*order-1)); doubling
    the threshold terminates without work proportional to the increment.
    """
    target = Fraction(1, 10**precision)
    order = precision + 10
    coefficients = _stirling_coefficients(order)
    threshold = order
    while True:
        difference = threshold - value
        shift = max(0, -(-difference.numerator // difference.denominator))
        x = value + shift
        first = _stirling_tail(x, coefficients, target / 8)
        second = _stirling_tail(x + increment, coefficients, target / 8)
        if first is not None and second is not None:
            correction = second - first
            break
        threshold *= 2
    shifted = Fraction(1)
    for i in range(shift):
        shifted *= (value + i) / (value + i + increment)

    work_precision = precision + 10
    while True:
        result = (
            increment * log_interval(x, precision=work_precision)
            + (x + increment - Fraction(1, 2))
            * log_interval(1 + increment / x, precision=work_precision)
            - increment
            + correction
        )
        if shift:
            result += log_interval(shifted, precision=work_precision)
        if result.width <= target:
            return result
        work_precision *= 2


def _trim(polynomial: Sequence[Fraction]) -> tuple[Fraction, ...]:
    result = list(polynomial)
    while result and result[-1] == 0:
        result.pop()
    return tuple(result)


def _derivative(polynomial: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    return tuple(i * polynomial[i] for i in range(1, len(polynomial)))


def _divmod_polynomial(
    dividend: tuple[Fraction, ...], divisor: tuple[Fraction, ...]
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    remainder = list(dividend)
    quotient = [Fraction(0)] * max(0, len(dividend) - len(divisor) + 1)
    while remainder and len(remainder) >= len(divisor):
        offset = len(remainder) - len(divisor)
        factor = remainder[-1] / divisor[-1]
        quotient[offset] = factor
        for i, coefficient in enumerate(divisor):
            remainder[offset + i] -= factor * coefficient
        while remainder and remainder[-1] == 0:
            remainder.pop()
    return _trim(quotient), tuple(remainder)


def _primitive(polynomial: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    """Remove positive rational content, preserving every evaluation's sign."""
    denominator = lcm(*(c.denominator for c in polynomial))
    integers = [c.numerator * (denominator // c.denominator) for c in polynomial]
    content = gcd(*integers)
    return tuple(Fraction(c // content) for c in integers)


def _square_free(polynomial: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
    a, b = polynomial, _derivative(polynomial)
    while b:
        remainder = _divmod_polynomial(a, b)[1]
        a, b = b, _primitive(remainder) if remainder else ()
    return _primitive(_divmod_polynomial(polynomial, a)[0])


def _integers(polynomial: tuple[Fraction, ...]) -> tuple[int, ...]:
    """Integer coefficients of a primitive polynomial; the sign tests need no division."""
    assert all(c.denominator == 1 for c in polynomial)
    return tuple(c.numerator for c in polynomial)


def _sign_at(integers: tuple[int, ...], p: int, q: int) -> int:
    """Sign of the polynomial at p/q (q > 0) from q**d * P(p/q) in integer Horner form.

    The ratio need not be reduced: scaling both terms leaves the sign unchanged.
    """
    result = integers[-1]
    scale = 1
    for coefficient in reversed(integers[:-1]):
        scale *= q
        result = result * p + coefficient * scale
    return (result > 0) - (result < 0)


def _sturm(polynomial: tuple[Fraction, ...]) -> tuple[tuple[int, ...], ...]:
    sequence = [polynomial, _primitive(_derivative(polynomial))]
    while True:
        remainder = _divmod_polynomial(sequence[-2], sequence[-1])[1]
        if not remainder:
            return tuple(_integers(member) for member in sequence)
        sequence.append(_primitive(tuple(-c for c in remainder)))


def _variations(sequence: tuple[tuple[int, ...], ...], p: int, q: int) -> int:
    previous = 0
    count = 0
    for integers in sequence:
        sign = _sign_at(integers, p, q)
        if sign:
            count += int(previous != 0 and sign != previous)
            previous = sign
    return count


def _isolate_single(
    polynomial: tuple[int, ...],
    sequence: tuple[tuple[int, ...], ...],
    lo: Fraction,
    hi: Fraction,
    left_variations: int,
    width: Fraction,
    denominator_bound: int,
) -> Interval:
    # Every midpoint is (lo + hi) / 2, so the whole bracket lives over one
    # denominator that doubles per step; only the returned endpoints are reduced.
    denominator = lcm(lo.denominator, hi.denominator)
    a = lo.numerator * (denominator // lo.denominator)
    b = hi.numerator * (denominator // hi.denominator)
    original_a, original_b = a, b
    while True:
        midpoint = a + b
        a, b, denominator = 2 * a, 2 * b, 2 * denominator
        original_a, original_b = 2 * original_a, 2 * original_b
        if _sign_at(polynomial, midpoint, denominator) == 0:
            return Interval.exact(Fraction(midpoint, denominator))
        # Strictly interior brackets cannot share endpoints with other roots.
        if (
            (b - a) * width.denominator <= width.numerator * denominator
            and a > original_a
            and b < original_b
        ):
            candidate = Fraction(midpoint, denominator).limit_denominator(denominator_bound)
            if (
                a * candidate.denominator
                <= candidate.numerator * denominator
                <= b * candidate.denominator
                and _sign_at(polynomial, candidate.numerator, candidate.denominator) == 0
            ):
                return Interval.exact(candidate)
            return Interval(Fraction(a, denominator), Fraction(b, denominator))
        middle_variations = _variations(sequence, midpoint, denominator)
        if left_variations - middle_variations:
            b = midpoint
        else:
            a, left_variations = midpoint, middle_variations


def isolate_roots(
    coefficients: Sequence[Fraction],
    *,
    lower: Fraction | None = None,
    upper: Fraction | None = None,
    max_width: Fraction = Fraction(1, 2**80),
) -> tuple[Interval, ...]:
    """Isolate every distinct real root in a closed domain, in ascending order.

    Ascending coefficients are reduced by gcd(P,P'); exact Sturm counts certify
    every subdivision. Omitted bounds use 2+max|a_i/a_d|, strictly outside all
    roots. Endpoint roots are recorded separately, including singleton domains.

    Verified rational roots may return point intervals; other roots stop at
    the requested width once their brackets are disjoint. Coefficient height
    does not force additional precision for rational-root reconstruction.
    """
    polynomial = _trim(tuple(_rational(c) for c in coefficients))
    width = _rational(max_width)
    lo = None if lower is None else _rational(lower)
    hi = None if upper is None else _rational(upper)
    if width <= 0:
        refuse(_NONPOSITIVE_ROOT_WIDTH, max_width=width)
    if lo is not None and hi is not None and lo > hi:
        refuse(_REVERSED_ROOT_DOMAIN, lower=lo, upper=hi)
    if not polynomial:
        refuse(_ZERO_POLYNOMIAL, coefficients=tuple(coefficients))
    if len(polynomial) == 1:
        return ()
    polynomial = _square_free(polynomial)
    if len(polynomial) == 2:
        root = -polynomial[0] / polynomial[1]
        if (lo is None or lo <= root) and (hi is None or root <= hi):
            return (Interval.exact(root),)
        return ()

    bound = 2 + max(abs(c / polynomial[-1]) for c in polynomial[:-1])
    lo = -bound if lo is None else max(lo, -bound)
    hi = bound if hi is None else min(hi, bound)
    if lo > hi:
        return ()
    integers = _integers(polynomial)
    if lo == hi:
        return (Interval.exact(lo),) if _sign_at(integers, *lo.as_integer_ratio()) == 0 else ()
    roots = []
    if _sign_at(integers, *lo.as_integer_ratio()) == 0:
        roots.append(Interval.exact(lo))
    right_is_root = _sign_at(integers, *hi.as_integer_ratio()) == 0
    if right_is_root:
        roots.append(Interval.exact(hi))

    sequence = _sturm(polynomial)
    left_variations = _variations(sequence, *lo.as_integer_ratio())
    right_variations = _variations(sequence, *hi.as_integer_ratio())
    # V(a)-V(b) counts (a,b]; subtract a root at b to count the open interval.
    count = left_variations - right_variations - int(right_is_root)
    pending = [(lo, hi, left_variations, right_variations, count)]
    denominator_bound = abs(integers[-1])
    while pending:
        lo, hi, left_variations, right_variations, count = pending.pop()
        if count == 0:
            continue
        if count == 1:
            roots.append(
                _isolate_single(
                    integers, sequence, lo, hi, left_variations, width, denominator_bound
                )
            )
            continue
        midpoint = (lo + hi) / 2
        middle_is_root = _sign_at(integers, *midpoint.as_integer_ratio()) == 0
        middle_variations = _variations(sequence, *midpoint.as_integer_ratio())
        left_count = left_variations - middle_variations - int(middle_is_root)
        if middle_is_root:
            roots.append(Interval.exact(midpoint))
        pending.append((lo, midpoint, left_variations, middle_variations, left_count))
        pending.append(
            (
                midpoint,
                hi,
                middle_variations,
                right_variations,
                count - left_count - int(middle_is_root),
            )
        )
    return tuple(sorted(roots, key=lambda interval: interval.lo))
