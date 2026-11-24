"""Numerical contracts at the independent unit or shared schedule block grain."""

from fractions import Fraction
from math import ldexp, sqrt

import pytest

from increment.estimation.switchback_metrics import switchback_cuped, switchback_ratio

# Rows are already weighted for arm/position exposure within each replicate.
_ROWS = [(8, 3, 4, 4), (17, 5, 10, 4), (28, 6, 15, 5), (20, 4, 12, 4), (11, 3, 8, 3)]


@pytest.mark.parametrize("exponent", [0, 1019, -1060])
def test_ratio_of_totals_with_full_replicate_covariance(exponent):
    # Exact rational arithmetic: grad' Cov(sample mean vector) grad = 1727741/28224000.
    # Setting the cross-arm blocks to zero instead gives 8741741/28224000;
    # averaging the individual rate differences instead gives 217/150.
    rows = [[ldexp(v, exponent) for v in r] for r in reversed(_ROWS)]
    fit = switchback_ratio(rows, alpha=0.05)
    assert fit.reason is None and fit.arm is None and fit.dof == 4
    assert fit.point == pytest.approx(31 / 20)
    assert fit.se == pytest.approx(sqrt(1727741 / 28224000), rel=1e-13)


@pytest.mark.parametrize("exponent", [-800, 0, 800])
def test_ratio_preserves_small_contrast_at_large_common_rate(exponent):
    # The treatment and control rates round to the same float separately.
    common = 2**52
    rows = [(ldexp(common + d, exponent), 1, ldexp(common, exponent), 1) for d in (0, 0, 0, 1)]
    fit = switchback_ratio(rows, alpha=0.05)
    assert fit.point == ldexp(0.25, exponent)
    assert fit.se == pytest.approx(ldexp(0.25, exponent), abs=0.0, rel=1e-14)


@pytest.mark.parametrize("common", [2**55, 2**2048])
def test_integer_and_rational_ht_totals_preserve_one_unit_rate_difference(common):
    deviations = [0, Fraction(1, 3), Fraction(-1, 3), Fraction(1, 5), Fraction(-1, 5)]
    rows = [(common + 1 + d, 1, common, 1) for d in deviations]
    fit = switchback_ratio(rows, alpha=0.05)
    assert fit.point == 1
    assert fit.se == pytest.approx(sqrt((2 / 9 + 2 / 25) / 20))
    assert fit.dof == 4


def test_cross_arm_cancellation_is_unavailable_not_a_zero_width_decision():
    rows = [(y + 3 * d, d, y, d) for y, d in [(3, 4), (8, 5), (1, 4), (9, 5)]]
    fit = switchback_ratio(rows, alpha=0.05)
    assert fit.point == 3 and fit.reason == "zero_spread"
    assert fit.se is None and fit.dof is None


@pytest.mark.parametrize("exchange", [False, True])
def test_denominator_guard_uses_replicate_t_precision_at_requested_level(exchange):
    # Denominator |mean|/SE = sqrt(10), between t_5(.975) and t_5(.99).
    rows = [(i, d, 1, 1) for i, d in enumerate((0, 0, 4, 4, 4, 4))]
    if exchange:
        rows = [(c, dc, t, dt) for t, dt, c, dc in rows]
    admitted = switchback_ratio(rows, alpha=0.05)
    weak = switchback_ratio(rows, alpha=0.02)
    assert admitted.reason is None and admitted.se is not None
    assert weak.point == admitted.point and weak.se is None
    assert weak.reason == "weak_denominator"
    assert weak.arm == ("control" if exchange else "treatment")


def test_zero_total_denominator_precedes_other_weak_denominator():
    fit = switchback_ratio([(1, d, 2, 0) for d in (0, 0, 0, 1)], alpha=0.05)
    assert fit.point is None and fit.se is None
    assert fit.reason == "zero_denominator" and fit.arm == "control"


def test_small_tail_does_not_reject_resolved_denominator():
    fit = switchback_ratio([(i, 1 + (i % 2) * 2.0**-40, 1, 2) for i in range(12)], alpha=1e-20)
    assert fit.reason is None and fit.se is not None


def test_unrepresentable_denominator_critical_keeps_point_and_affected_arm():
    # With one degree of freedom the Cauchy critical is cot(pi * alpha / 2),
    # larger than the biggest float here, while both rates remain finite.
    fit = switchback_ratio([(5, 1, 3, 1), (5, 1, 3, 2)], alpha=1e-310)
    assert fit.point == 3 and fit.se is None and fit.dof is None
    assert fit.reason == "numeric_range" and fit.arm == "control"


def test_nonrepresentable_ratio_uncertainty_preserves_finite_point():
    fit = switchback_ratio([(1e308, 0.25, 0, 1), (-1e308, 0.25, 0, 1)], alpha=0.05)
    assert fit.point == 0 and fit.se is None and fit.reason == "numeric_range"


def test_one_ratio_replicate_is_descriptive_only():
    fit = switchback_ratio([(18, 3, 8, 2)], alpha=0.05)
    assert fit.point == 2 and fit.se is None and fit.reason == "insufficient_replicates"


@pytest.mark.parametrize("exponent", [-800, 0, 800])
def test_cuped_refits_slope_on_each_deletion_even_at_large_offset(exponent):
    # Intercepts from separately solving the five-observation normal equations.
    ratios = [(368, 93), (1037, 165), (892, 183), (998, 183), (62, 15), (505, 93)]
    deleted = [Fraction(a, b) for a, b in ratios]
    mean = sum(deleted) / 6
    expected_se = sqrt(Fraction(5, 6) * sum((v - mean) ** 2 for v in deleted))
    offset = 2**51
    y = [ldexp(offset + v, exponent) for v in [3, 0, 7, 5, 15, 9]]
    x = [ldexp(v, -exponent) for v in [-4, -1, 1, 2, 4, 7]]
    fit = switchback_cuped(y, x)
    assert fit.reason is None and fit.dof == 4
    assert fit.point == ldexp(float(offset + Fraction(249, 49)), exponent)
    assert fit.se == pytest.approx(ldexp(expected_se, exponent), rel=1e-12, abs=0.0)


def test_cuped_uncertainty_does_not_depend_on_replicate_order():
    x = [0, 1, 0, 1, 0, 1]
    y = [-4, -5, 5, -1, -1, -8]
    # Dropping an x=0 row gives intercepts 2, -5/2 and 1/2; the other three give 0.
    forward = switchback_cuped(y, x)
    backward = switchback_cuped(y[::-1], x[::-1])
    assert forward.se == pytest.approx(sqrt(35 / 4))
    assert backward.se == forward.se


def test_exact_ht_inputs_retain_adjusted_effect_after_removing_large_slope():
    # Rescaling x by 1/3 and adding any multiple of x to y leaves the
    # intercept unchanged in both the full data and every separate refit.
    x = [Fraction(v, 3) for v in [-4, -1, 1, 2, 4, 7]]
    y = [2**55 * v + r for v, r in zip(x, [3, 0, 7, 5, 15, 9], strict=True)]
    fit = switchback_cuped(y, x)
    assert fit.point == pytest.approx(249 / 49)
    assert fit.se == pytest.approx(sqrt(63171971042 / 19470672045))
    assert fit.dof == 4


def test_multiple_forecasts_are_outside_scalar_adjustment():
    fit = switchback_cuped([1, 2, 3, 4], [[0, 1], [2, 3], [4, -1], [0, 0]])
    assert fit.reason == "invalid_input" and fit.point is None


def test_forecast_contrast_is_design_centered_not_sample_centered():
    # P(CT)=1/4. Positions differ by 3 in the forecast and 5 in the baseline;
    # the additive effect is 2. HT forecast is +3/(2p) or -3/(2(1-p)).
    # The design weights, not this particular sample's frequencies, center x.
    y = [14, -2] * 3
    x = [6, -2] * 3
    assert 0.25 * x[0] + 0.75 * x[1] == 0
    fit = switchback_cuped(y, x)
    assert fit.point == 2 and fit.point != sum(y) / len(y)
    assert fit.reason == "zero_spread" and fit.se is None


@pytest.mark.parametrize("x", [[0] * 5, [3] * 5])
def test_uninformative_forecast_is_not_an_unadjusted_fit(x):
    fit = switchback_cuped([1, 4, 5, 2, 3], x)
    assert fit.reason == "rank_deficient" and fit.point is None and fit.se is None


def test_single_leveraged_replicate_prevents_jackknife_but_not_full_fit():
    fit = switchback_cuped([1, 3, 5, 7, 20], [0, 0, 0, 0, 1])
    assert fit.point == 4 and fit.se is None and fit.reason == "high_leverage"


def test_saturated_fit_has_no_residual_degrees_of_freedom():
    fit = switchback_cuped([2, 8], [-1, 1])
    assert fit.point == 5 and fit.se is None and fit.reason == "insufficient_replicates"
