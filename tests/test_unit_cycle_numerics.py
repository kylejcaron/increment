"""Independent arithmetic checks for the shared runtime/planning primitives."""

import math
from fractions import Fraction
from typing import cast

import pytest

from increment._unit_cycle import unit_cycle_admission_rule, unit_cycle_envelope_cutoff
from increment.errors import InvalidRequestError
from tests.estimation.test_unit_cycle_envelope import envelope


@pytest.mark.slow
@pytest.mark.parametrize("n,p", [(4, 0.9), (40, 0.5), (120, 0.1), (120, 0.9), (100, 0.5)])
def test_admission_matches_every_actual_binomtest_count_and_bounds_exact_mass(n, p):
    from scipy.stats import binomtest

    rule = unit_cycle_admission_rule(n, p)
    probability = Fraction(p)
    mass = Fraction(0)
    for k in range(n + 1):
        refused = binomtest(k, n, p).pvalue < 1e-6
        assert refused == (k < rule.minimum_ct or k > rule.maximum_ct)
        if refused:
            mass += math.comb(n, k) * probability**k * (1 - probability) ** (n - k)
    assert mass <= Fraction(rule.refusal_probability_upper)
    assert rule.refusal_probability_upper < 0.001


def test_all_same_orders_are_legitimate_when_the_actual_test_accepts_them():
    rule = unit_cycle_admission_rule(4, 0.9)
    assert rule.minimum_ct == 0 and rule.maximum_ct == 4
    assert rule.refusal_probability_upper == 0


@pytest.mark.slow
def test_admission_is_cached_and_does_not_allocate_per_declared_draw(monkeypatch):
    import increment._unit_cycle as numerical

    original = numerical.binomtest
    calls = []

    def record(k, n, p):
        calls.append(k)
        return original(k, n, p)

    monkeypatch.setattr(numerical, "binomtest", record)
    numerical.unit_cycle_admission_rule.cache_clear()
    rule = numerical.unit_cycle_admission_rule(10**9, 0.73)
    count = len(calls)
    assert count < 2 * (10**9).bit_length() + 3
    assert rule is numerical.unit_cycle_admission_rule(10**9, 0.73)
    assert len(calls) == count


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
@pytest.mark.parametrize(
    "n,p,variance,alpha",
    [
        (4, 0.5, 1.0, 0.05),
        (40, 0.5, 1.0, 0.05),
        (4, 0.9, 1e-300, 1e-300),
        (4, 0.5, 1e308, 0.05),
        (4, 0.5, math.ulp(0.0), 0.05),
    ],
)
def test_cutoff_and_error_allocation_are_outward_in_exact_binary_rational_arithmetic(
    alternative, n, p, variance, alpha
):
    cutoff = unit_cycle_envelope_cutoff(
        envelope(p=p, variance=variance), n=n, alpha=alpha, alternative=alternative
    )
    effective = Fraction(cutoff.effective_alpha)
    assert effective + Fraction(cutoff.refusal_probability_upper) <= Fraction(alpha)
    required_squared = Fraction(variance) / n / effective
    if alternative != "two-sided":
        required_squared *= 1 - effective
    assert Fraction(cutoff.value) ** 2 >= required_squared
    assert Fraction(math.nextafter(cutoff.value, 0)) ** 2 < required_squared


def test_zero_envelope_remains_exactly_zero():
    cutoff = unit_cycle_envelope_cutoff(
        envelope(variance=0), n=4, alpha=0.05, alternative="two-sided"
    )
    assert cutoff.value == 0


def test_admission_error_budget_exhaustion_has_its_own_refusal():
    rule = unit_cycle_admission_rule(100, 0.5)
    assert rule.refusal_probability_upper > 0
    with pytest.raises(InvalidRequestError) as caught:
        unit_cycle_envelope_cutoff(
            envelope(p=0.5), n=100, alpha=rule.refusal_probability_upper, alternative="two-sided"
        )
    assert caught.value.code == "unit_cycle.error_budget_exhausted"
    with pytest.raises(TypeError):
        cast("dict[str, object]", caught.value.context)["alpha"] = 0.5


def test_unrepresentable_cutoff_is_numerical_not_nonidentification():
    with pytest.raises(InvalidRequestError) as caught:
        unit_cycle_envelope_cutoff(
            envelope(p=0.5, variance=1e308), n=4, alpha=math.ulp(0.0), alternative="two-sided"
        )
    assert caught.value.code == "unit_cycle.numerical"
    assert caught.value.context["reason"] == "cutoff_unrepresentable"


@pytest.mark.parametrize("n", [True, 0, -1, 1.5])
def test_n_must_be_a_positive_nonboolean_integer(n):
    with pytest.raises(InvalidRequestError) as caught:
        unit_cycle_admission_rule(n, 0.5)
    assert caught.value.code == "unit_cycle.invalid_design"


def test_numerical_contract_does_not_depend_on_callers_decimal_context():
    from decimal import ROUND_FLOOR, localcontext

    expected = unit_cycle_envelope_cutoff(
        envelope(p=0.5), n=40, alpha=0.05, alternative="two-sided"
    )
    unit_cycle_admission_rule.cache_clear()
    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_FLOOR
        context.Emax = 3
        actual = unit_cycle_envelope_cutoff(
            envelope(p=0.5), n=40, alpha=0.05, alternative="two-sided"
        )
    assert actual == expected
