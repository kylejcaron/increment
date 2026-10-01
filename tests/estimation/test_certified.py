"""Mathematical behavior of the freestanding certified arithmetic layer."""

from __future__ import annotations

import math
import operator
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from decimal import ROUND_DOWN, ROUND_UP, Inexact, localcontext
from fractions import Fraction
from functools import lru_cache
from typing import cast

import pytest

from increment.errors import InvalidRequestError
from increment.estimation._certified import (
    Interval,
    isolate_roots,
    log_gamma_half_step,
    log_interval,
    log_rising,
)


@contextmanager
def _refuses(code: str, **context: object) -> Iterator[None]:
    with pytest.raises(InvalidRequestError) as raised:
        yield
    assert raised.value.code == "estimation.certified." + code
    assert raised.value.context == context


def _contains(outer: Interval, inner: Interval) -> None:
    assert outer.lo <= inner.lo <= inner.hi <= outer.hi


def _overlaps(first: Interval, second: Interval) -> None:
    assert max(first.lo, second.lo) <= min(first.hi, second.hi)


@lru_cache(maxsize=64)
def _reference_log(value: Fraction, digits: int = 100) -> Interval:
    """Independent convergent atanh series, with a geometric rational tail."""
    if value == 1:
        return Interval.exact(0)
    exponent = 0
    scaled = value
    if not Fraction(1, 2) <= value <= 2:
        exponent = value.numerator.bit_length() - value.denominator.bit_length()
        scaled = value / Fraction(2) ** exponent
    t = (scaled - 1) / (scaled + 1)
    power = t
    total = Fraction(0)
    n = 0
    tolerance = Fraction(1, 10**digits) * min(1, abs(value - 1)) ** 3
    while True:
        total += 2 * power / (2 * n + 1)
        n += 1
        power *= t * t
        remainder = 2 * abs(power) / ((2 * n + 1) * (1 - t * t))
        if remainder <= tolerance:
            result = Interval(total - remainder, total + remainder)
            break
    if exponent:
        result += exponent * _reference_log(Fraction(2), digits=digits)
    return result


@lru_cache(maxsize=1)
def _pi_bounds() -> Interval:
    """Machin's identity, bounded by alternating arctangent remainders."""
    bounds = []
    for denominator in (5, 239):
        z = Fraction(1, denominator)
        total = sum(
            ((-1) ** k * z ** (2 * k + 1) / (2 * k + 1) for k in range(90)),
            Fraction(0),
        )
        next_term = z**181 / 181
        bounds.append(Interval(total, total + next_term))
    pi = 16 * bounds[0] - 4 * bounds[1]
    # Shorter rational endpoints keep subsequent reference-series work small.
    scale = 10**100
    return Interval(
        Fraction((pi.lo * scale).__floor__(), scale),
        Fraction((pi.hi * scale).__ceil__(), scale),
    )


@lru_cache(maxsize=1)
def _log_pi_bounds() -> Interval:
    pi = _pi_bounds()
    return Interval(_reference_log(pi.lo).lo, _reference_log(pi.hi).hi)


def _polynomial_at(coefficients: list[Fraction], value: Fraction) -> Fraction:
    return sum((c * value**i for i, c in enumerate(coefficients)), Fraction(0))


def _multiply(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    product = [Fraction(0)] * (len(first) + len(second) - 1)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            product[i + j] += a * b
    return product


def _from_roots(roots: list[Fraction]) -> list[Fraction]:
    result = [Fraction(1)]
    for root in roots:
        result = _multiply(result, [-root, Fraction(1)])
    return result


def test_interval_copies_exact_values_and_is_frozen() -> None:
    interval = Interval(-2, Fraction(5, 3))
    assert isinstance(interval.lo, Fraction)
    assert isinstance(interval.hi, Fraction)
    assert interval.width == Fraction(11, 3)
    assert interval.midpoint == Fraction(-1, 6)
    assert Interval.exact(4) == Interval(Fraction(4), Fraction(4))
    assert hash(interval) == hash(Interval(-2, Fraction(5, 3)))
    with pytest.raises(FrozenInstanceError):
        interval.__setattr__("lo", Fraction(0))
    with _refuses("reversed_interval", lower=Fraction(2), upper=Fraction(1)):
        Interval(2, 1)


@pytest.mark.parametrize("value", [0.5, float("nan"), float("inf"), "1"])
def test_interval_refuses_implicit_inexact_inputs(value) -> None:
    with _refuses("inexact_value", value_type=type(value).__name__):
        Interval(value, 1)
    with _refuses("inexact_value", value_type=type(value).__name__):
        Interval(0, value)
    with _refuses("inexact_value", value_type=type(value).__name__):
        Interval.exact(value)


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("rhs", [Interval(2, 5), Interval(-5, -2)])
def test_interval_arithmetic_contains_endpoint_and_interior_values(
    operation, rhs: Interval
) -> None:
    lhs = Interval(Fraction(-7, 3), Fraction(11, 4))
    result = operation(lhs, rhs)
    for a in (lhs.lo, lhs.midpoint, (lhs.lo * 2 + lhs.hi) / 3, lhs.hi):
        for b in (rhs.lo, rhs.midpoint, rhs.hi):
            assert result.lo <= operation(a, b) <= result.hi


@pytest.mark.parametrize("scalar", [-3, Fraction(2, 7)])
def test_scalar_operations_and_reflections(scalar: Fraction | int) -> None:
    interval = Interval(2, 5)
    exact = Interval.exact(scalar)
    for operation in (operator.add, operator.sub, operator.mul, operator.truediv):
        assert operation(interval, scalar) == operation(interval, exact)
        assert operation(scalar, interval) == operation(exact, interval)
    assert -interval == Interval(-5, -2)


@pytest.mark.parametrize(
    "divisor", [Interval(-1, 1), Interval(0, 2), Interval(-2, 0), Interval.exact(0)]
)
def test_division_refuses_zero_including_endpoint_zeros(divisor: Interval) -> None:
    with _refuses("zero_divisor_interval", lower=divisor.lo, upper=divisor.hi):
        Interval(1, 2) / divisor
    with _refuses("zero_divisor_interval", lower=divisor.lo, upper=divisor.hi):
        1 / divisor
    with _refuses("zero_divisor_interval", lower=divisor.lo, upper=divisor.hi):
        Fraction(1, 3) / divisor
    with _refuses("zero_divisor_interval", lower=Fraction(0), upper=Fraction(0)):
        Interval(1, 2) / 0


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
def test_arithmetic_refuses_float_scalars_in_both_positions(operation) -> None:
    with _refuses("inexact_value", value_type="float"):
        operation(Interval(1, 2), 0.5)
    with _refuses("inexact_value", value_type="float"):
        operation(0.5, Interval(1, 2))


@pytest.mark.parametrize(
    "value",
    [
        Fraction(1),
        Fraction(2),
        Fraction(1, 3),
        Fraction(7, 4),
        Fraction(1, 2**2000),
        Fraction(2**2000),
    ],
)
def test_log_encloses_independent_rational_series(value: Fraction) -> None:
    result = log_interval(value)
    _contains(result, _reference_log(value))
    assert result.width < Fraction(1, 10**55)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("bits", [60, 2000])
def test_log_preserves_nonzero_values_next_to_one(direction: int, bits: int) -> None:
    value = 1 + direction * Fraction(1, 2**bits)
    result = log_interval(value, precision=25)
    _contains(result, _reference_log(value))
    assert result.lo > 0 if direction > 0 else result.hi < 0
    assert result.width < abs(value - 1) / 10**25


def test_log_of_interval_encloses_both_monotone_endpoints() -> None:
    value = Interval(Fraction(2, 3), Fraction(7, 3))
    result = log_interval(value, precision=35)
    _contains(result, _reference_log(value.lo))
    _contains(result, _reference_log(value.hi))
    assert result.lo < 0 < result.hi
    assert result.width < 2


@pytest.mark.parametrize("value", [Fraction(0), Fraction(-1), Interval(-1, 2), Interval(0, 1)])
def test_log_refuses_nonpositive_domain(value) -> None:
    interval = value if isinstance(value, Interval) else Interval.exact(value)
    with _refuses("nonpositive_log_domain", lower=interval.lo, upper=interval.hi):
        log_interval(value)


def test_log_precision_refines_an_outward_enclosure() -> None:
    value = Fraction(17, 19)
    lower = log_interval(value, precision=12)
    higher = log_interval(value, precision=55)
    _contains(lower, higher)
    _contains(higher, _reference_log(value))
    assert higher.width < lower.width


def test_decimal_context_is_private_even_with_hostile_thread_settings() -> None:
    expected = log_interval(Fraction(7, 3), precision=25)

    def calculate(rounding: str) -> Interval:
        with localcontext() as context:
            context.prec = 3
            context.rounding = rounding
            context.Emax = 9
            context.Emin = -9
            context.traps[Inexact] = True
            context.clear_flags()
            result = log_interval(Fraction(7, 3), precision=25)
            assert context.prec == 3
            assert context.rounding == rounding
            assert context.Emax == 9
            assert context.Emin == -9
            assert context.traps[Inexact]
            assert not any(context.flags.values())
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(calculate, (ROUND_DOWN, ROUND_UP))) == [expected, expected]


@pytest.mark.parametrize(
    ("value", "count", "product"),
    [
        (Fraction(2), 0, Fraction(1)),
        (Fraction(1, 2), 3, Fraction(15, 8)),
        (Fraction(1, 3), 3, Fraction(28, 27)),
        (Fraction(2), 4, Fraction(120)),
        (Fraction(1) + Fraction(1, 2**200), 1, Fraction(1) + Fraction(1, 2**200)),
    ],
)
def test_log_rising_matches_exact_small_products(
    value: Fraction, count: int, product: Fraction
) -> None:
    result = log_rising(value, count)
    _contains(result, _reference_log(product))
    if count == 0:
        assert result == Interval.exact(0)


@pytest.mark.slow
def test_log_rising_at_large_offset_preserves_neighboring_rationals() -> None:
    value = Fraction(2**200)
    result = log_rising(value, 3, precision=30)
    reference = sum((_reference_log(value + i) for i in range(3)), Interval.exact(0))
    _contains(result, reference)


@pytest.mark.slow
def test_log_rising_tens_of_thousands_preserves_partition_identity() -> None:
    value = Fraction(3, 2)
    whole = log_rising(value, 20_000, precision=20)
    partitioned = log_rising(value, 10_000, precision=20) + log_rising(
        value + 10_000, 10_000, precision=20
    )
    _overlaps(whole, partitioned)
    approximate = math.lgamma(float(value) + 20_000) - math.lgamma(float(value))
    assert float(whole.midpoint) == pytest.approx(approximate, rel=1e-14)
    assert whole.width < Fraction(1, 10**20)


@pytest.mark.slow
def test_log_rising_trillion_factorial_encloses_independent_stirling_bound() -> None:
    n = 10**12
    reference = (
        (n + Fraction(1, 2)) * _reference_log(Fraction(n))
        - n
        + (_reference_log(Fraction(2)) + _log_pi_bounds()) / 2
        + Fraction(1, 12 * n)
        - Fraction(1, 360 * n**3)
        + Fraction(1, 1260 * n**5)
        + Interval(Fraction(-1, 1680 * n**7), 0)
    )
    result = log_rising(Fraction(1), n, precision=40)
    _contains(result, reference)
    assert result.width <= Fraction(1, 10**40)


@pytest.mark.parametrize("count", [-1, 1.5, Fraction(3, 2), True])
def test_log_rising_refuses_invalid_counts(count) -> None:
    with _refuses("invalid_count", count=count):
        log_rising(Fraction(1), count)


@pytest.mark.parametrize("value", [Fraction(0), Fraction(-1)])
def test_rising_and_gamma_refuse_nonpositive_values_even_for_empty_product(
    value: Fraction,
) -> None:
    with _refuses("nonpositive_rising_value", value=value):
        log_rising(value, 0)
    with _refuses("nonpositive_gamma_value", value=value):
        log_gamma_half_step(value)


@pytest.mark.parametrize("precision", [0, -1, 1.5, True])
def test_log_functions_refuse_invalid_precision(precision) -> None:
    with _refuses("invalid_precision", precision=precision):
        log_interval(Fraction(1), precision=precision)
    with _refuses("invalid_precision", precision=precision):
        log_rising(Fraction(1), 0, precision=precision)
    with _refuses("invalid_precision", precision=precision):
        log_gamma_half_step(Fraction(1), precision=precision)


@pytest.mark.parametrize(
    "value", [Fraction(1, 8), Fraction(1, 3), Fraction(5, 4), Fraction(10), Fraction(100)]
)
def test_gamma_half_step_matches_approximate_float_oracle(value: Fraction) -> None:
    result = log_gamma_half_step(value, precision=30)
    approximate = math.lgamma(float(value) + 0.5) - math.lgamma(float(value))
    # A binary64 oracle is too coarse to require containment in a 30-digit interval.
    assert float(result.midpoint) == pytest.approx(approximate, abs=2e-13)
    assert result.width <= Fraction(1, 10**30)


@pytest.mark.slow
@pytest.mark.parametrize("twice_value", [1, 2, 3, 4, 20, 200])
def test_gamma_half_step_encloses_independent_factorial_and_pi_reference(twice_value: int) -> None:
    value = Fraction(twice_value, 2)
    if twice_value % 2 == 0:
        n = twice_value // 2
        rational = Fraction(math.factorial(2 * n), 4**n * math.factorial(n) * math.factorial(n - 1))
        reference = _reference_log(rational) + _log_pi_bounds() / 2
    else:
        n = (twice_value - 1) // 2
        rational = Fraction(4**n * math.factorial(n) ** 2, math.factorial(2 * n))
        reference = _reference_log(rational) - _log_pi_bounds() / 2
    coarse = log_gamma_half_step(value, precision=15)
    fine = log_gamma_half_step(value, precision=45)
    _contains(coarse, reference)
    _contains(fine, reference)
    _overlaps(coarse, fine)
    assert fine.width < coarse.width


@pytest.mark.parametrize("value", [Fraction(1, 7), Fraction(19, 5), Fraction(10**30)])
def test_gamma_half_step_obeys_two_half_steps_and_integer_recurrence(value: Fraction) -> None:
    base = log_gamma_half_step(value, precision=30)
    next_half = log_gamma_half_step(value + Fraction(1, 2), precision=30)
    next_integer = log_gamma_half_step(value + 1, precision=30)
    _overlaps(base + next_half, log_interval(value, precision=50))
    correction = log_interval((value + Fraction(1, 2)) / value, precision=50)
    _overlaps(next_integer - base, correction)


@pytest.mark.slow
def test_gamma_precision_can_grow_beyond_a_fixed_stirling_table() -> None:
    first = log_gamma_half_step(Fraction(1, 2), precision=110)
    second = log_gamma_half_step(Fraction(1), precision=110)
    _contains(first + second, _reference_log(Fraction(1, 2), digits=160))
    assert first.width <= Fraction(1, 10**110)
    assert second.width <= Fraction(1, 10**110)


@pytest.mark.slow
@pytest.mark.parametrize("value", [Fraction(1, 2**2000), Fraction(10**400)])
def test_gamma_extreme_scales_remain_finite_and_refinable(value: Fraction) -> None:
    coarse = log_gamma_half_step(value, precision=20)
    fine = log_gamma_half_step(value, precision=40)
    assert fine.width <= Fraction(1, 10**40)
    assert fine.width < coarse.width
    _overlaps(coarse, fine)
    _overlaps(fine + log_gamma_half_step(value + Fraction(1, 2), precision=40), log_interval(value))


@pytest.mark.slow
def test_gamma_huge_argument_encloses_independent_log_convexity_bounds() -> None:
    value = Fraction(10**100)
    # Log-convexity: log(x)-log(x+1/2)/2 <= C(x) <= log(x)/2.
    log_x = _reference_log(value, digits=130)
    log_next = _reference_log(value + Fraction(1, 2), digits=130)
    reference = Interval(log_x.lo - log_next.hi / 2, log_x.hi / 2)
    result = log_gamma_half_step(value, precision=35)
    _contains(result, reference)
    assert result.width <= Fraction(1, 10**35)


@pytest.mark.parametrize(
    "roots",
    [
        [Fraction(1), Fraction(1), Fraction(3)],
        [Fraction(-2), Fraction(-1), Fraction(0), Fraction(1), Fraction(2), Fraction(3)],
        [Fraction(-2, 7), Fraction(1, 3), Fraction(5, 11)],
        [Fraction(1, 3)] * 6,
    ],
)
def test_root_isolation_contains_all_distinct_rational_roots(roots: list[Fraction]) -> None:
    coefficients = _from_roots(roots)
    before = coefficients.copy()
    result = isolate_roots(coefficients, max_width=Fraction(1, 10))
    for interval, root in zip(result, sorted(set(roots)), strict=True):
        assert interval.lo <= root <= interval.hi
        assert interval.width <= Fraction(1, 10)
    assert all(a.hi < b.lo for a, b in zip(result, result[1:], strict=False))
    assert coefficients == before


def test_linear_root_and_trailing_zero_coefficients() -> None:
    expected = (Interval.exact(Fraction(2, 7)),)
    assert isolate_roots([Fraction(-2), Fraction(7), Fraction(0)]) == expected
    assert isolate_roots([Fraction(-2), Fraction(7)], upper=Fraction(1, 4)) == ()
    assert isolate_roots([Fraction(-2), Fraction(7)], lower=Fraction(2, 7)) == expected


def test_six_irrational_roots_are_complete_disjoint_and_sign_bracketed() -> None:
    # (x^2-2)(x^2-3)(x^2-5) has exactly six distinct real roots.
    coefficients = [Fraction(c) for c in (-30, 0, 31, 0, -10, 0, 1)]
    width = Fraction(1, 2**80)
    result = isolate_roots(coefficients, max_width=width)
    assert len(result) == 6
    for interval, square in zip(result, (5, 3, 2, 2, 3, 5), strict=True):
        assert 0 < interval.width <= width
        assert (
            _polynomial_at(coefficients, interval.lo) * _polynomial_at(coefficients, interval.hi)
            < 0
        )
        squared = interval * interval
        assert squared.lo <= square <= squared.hi
    assert all(a.hi < b.lo for a, b in zip(result, result[1:], strict=False))


def test_even_multiplicity_irrational_roots_survive_square_free_reduction() -> None:
    coefficients = [Fraction(4), Fraction(0), Fraction(-4), Fraction(0), Fraction(1)]
    result = isolate_roots(coefficients)
    assert len(result) == 2
    for interval in result:
        assert _polynomial_at(coefficients, interval.lo) > 0
        assert _polynomial_at(coefficients, interval.hi) > 0
        assert (interval.lo**2 - 2) * (interval.hi**2 - 2) < 0
        assert interval.width <= Fraction(1, 2**80)


def test_complex_roots_are_excluded_after_square_free_reduction() -> None:
    polynomial = _multiply(
        [Fraction(c) for c in (4, 0, -4, 0, 1)],
        [Fraction(c) for c in (1, 0, 1)],
    )
    result = isolate_roots(polynomial)
    assert len(result) == 2
    for interval in result:
        assert (interval.lo**2 - 2) * (interval.hi**2 - 2) < 0


@pytest.mark.parametrize(
    ("lower", "upper", "expected"),
    [
        (Fraction(1), Fraction(3), (1, 2, 3)),
        (Fraction(1), Fraction(1), (1,)),
        (Fraction(3, 2), Fraction(3, 2), ()),
        (None, Fraction(2), (1, 2)),
        (Fraction(2), None, (2, 3)),
        (None, Fraction(-100), ()),
        (Fraction(100), None, ()),
    ],
)
def test_closed_domains_include_boundary_roots_once(lower, upper, expected) -> None:
    polynomial = _from_roots([Fraction(r) for r in (1, 1, 2, 3, 3)])
    assert isolate_roots(polynomial, lower=lower, upper=upper) == tuple(
        Interval.exact(r) for r in expected
    )


def test_exact_root_and_irrational_brackets_do_not_share_endpoints() -> None:
    coefficients = [Fraction(0), Fraction(-2), Fraction(0), Fraction(1)]
    result = isolate_roots(
        coefficients, lower=Fraction(0), upper=Fraction(2), max_width=Fraction(100)
    )
    assert len(result) == 2
    assert result[0] == Interval.exact(0)
    assert 0 < result[1].lo < result[1].hi < 2
    assert result[1].lo ** 2 < 2 < result[1].hi ** 2


@pytest.mark.slow
def test_nearby_roots_and_large_offsets_are_not_merged() -> None:
    offset = Fraction(2**100)
    roots = [offset, offset + Fraction(1, 2**90), offset + Fraction(1, 3**60)]
    result = isolate_roots(_from_roots(roots))
    for interval, root in zip(result, sorted(roots), strict=True):
        assert interval.lo <= root <= interval.hi
        assert interval.width <= Fraction(1, 2**80)
    assert all(a.hi < b.lo for a, b in zip(result, result[1:], strict=False))


@pytest.mark.parametrize("scale", [Fraction(-1), Fraction(1, 2**300), Fraction(10**100)])
def test_root_isolation_is_invariant_to_nonzero_polynomial_scaling(scale: Fraction) -> None:
    polynomial = [Fraction(-2), Fraction(0), Fraction(1)]
    expected = isolate_roots(polynomial)
    assert isolate_roots([scale * c for c in polynomial]) == expected


@pytest.mark.parametrize(
    "coefficients", [[Fraction(7)], [Fraction(-7), Fraction(0)], [Fraction(c) for c in (1, 0, 1)]]
)
def test_nonzero_constants_and_polynomials_without_real_roots(coefficients: list[Fraction]) -> None:
    assert isolate_roots(coefficients) == ()


@pytest.mark.parametrize("coefficients", [[], [Fraction(0)], [Fraction(0), Fraction(0)]])
def test_identically_zero_polynomial_is_refused(coefficients: list[Fraction]) -> None:
    with _refuses("zero_polynomial", coefficients=tuple(coefficients)):
        isolate_roots(coefficients)


def test_root_isolation_refuses_invalid_domains_and_inexact_coefficients() -> None:
    with _refuses("reversed_root_domain", lower=Fraction(2), upper=Fraction(1)):
        isolate_roots([Fraction(1), Fraction(1)], lower=Fraction(2), upper=Fraction(1))
    for width in (Fraction(0), Fraction(-1)):
        with _refuses("nonpositive_root_width", max_width=width):
            isolate_roots([Fraction(1), Fraction(1)], max_width=width)
    with _refuses("inexact_value", value_type="float"):
        isolate_roots(cast(list[Fraction], [0.5, 1]))


@pytest.mark.slow
def test_large_coefficient_height_does_not_force_unrequested_root_precision() -> None:
    import subprocess
    import sys

    program = """
from fractions import Fraction
from increment.estimation._certified import isolate_roots
a = 2**20000
width = Fraction(1, 2**40)
roots = isolate_roots(
    [Fraction(-a - 1), Fraction(0), Fraction(a)],
    lower=Fraction(0), upper=Fraction(2), max_width=width,
)
assert len(roots) == 1
root = roots[0]
assert 0 < root.width <= width
# Refinement stops within one bisection of the request, not at the coefficient height.
assert root.width * 2 > width
assert a * root.lo**2 <= a + 1 <= a * root.hi**2
"""
    # The timeout only catches a hang; the precision assertions above carry the contract.
    subprocess.run(
        [sys.executable, "-c", program], check=True, capture_output=True, text=True, timeout=120
    )


def test_non_dyadic_domain_endpoint_is_preserved_exactly() -> None:
    lower = Fraction(1, 3)
    roots = isolate_roots(
        [Fraction(-2, 3), Fraction(5, 3), Fraction(1)],
        lower=lower,
        upper=Fraction(1),
        max_width=Fraction(1, 10),
    )
    assert roots == (Interval.exact(lower),)


def test_decimal_precision_capacity_is_coded_before_context_allocation() -> None:
    from decimal import MAX_PREC

    with _refuses(
        "decimal_precision_capacity",
        precision=MAX_PREC,
        working_precision=MAX_PREC + 15,
        max_precision=MAX_PREC,
    ):
        log_interval(Fraction(2), precision=MAX_PREC)


def test_decimal_arithmetic_capacity_uses_real_overflow_with_small_exponent_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from increment.estimation import _certified

    monkeypatch.setattr(_certified, "MAX_EMAX", 1)
    with _refuses(
        "decimal_arithmetic_capacity",
        value=Fraction(100),
        precision=10,
        working_precision=25,
        signal="Overflow",
    ):
        log_interval(Fraction(100), precision=10)


def test_decimal_input_capacity_uses_real_underflow_with_small_exponent_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from increment.estimation import _certified

    monkeypatch.setattr(_certified, "MIN_EMIN", -1)
    with _refuses(
        "decimal_input_exponent_capacity",
        value=Fraction(1, 10**100),
        precision=10,
        working_precision=25,
    ):
        log_interval(Fraction(1, 10**100), precision=10)


def test_refusal_context_snapshots_zero_polynomial_coefficients() -> None:
    coefficients = [Fraction(0), Fraction(0)]
    with pytest.raises(InvalidRequestError) as raised:
        isolate_roots(coefficients)
    coefficients[0] = Fraction(1)
    assert raised.value.code == "estimation.certified.zero_polynomial"
    assert raised.value.context == {"coefficients": (Fraction(0), Fraction(0))}
    with pytest.raises(TypeError):
        cast(dict[str, object], raised.value.context)["coefficients"] = ()
