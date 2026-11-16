"""Finite-output and independent analytic references for relative conjugate laws."""

from __future__ import annotations

import math
import sys
from typing import cast

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from increment.errors import InvalidRequestError
from increment.estimation._relative_conjugate import (
    RelativeMixturePosterior,
    RelativeNormalPosterior,
)
from increment.estimation._relative_posterior import _PosteriorNumericalError
from increment.estimation.priors import MixturePosterior, Normal


def normal(mu, sigma):
    return RelativeNormalPosterior(Normal(mu=mu, sigma=sigma))


def mixture(weights, means, sigmas):
    return RelativeMixturePosterior(
        MixturePosterior(np.array(weights), np.array(means), np.array(sigmas))
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_finite_standardized_difference_survives_overflowing_subtraction(mixed):
    law = mixture((1.0,), (-1e308,), (1e308,)) if mixed else normal(-1e308, 1e308)
    assert law.cdf(1e308) == pytest.approx(norm.cdf(2), abs=1e-10)
    assert law.survival(1e308) == pytest.approx(norm.sf(2), rel=1e-6, abs=0)
    assert law.probability_between(-1e308, 1e308) == pytest.approx(norm.cdf(2) - 0.5, abs=1e-10)


@pytest.mark.parametrize("mixed", [False, True])
def test_direct_small_tail_and_adjacent_float_interval_keep_probability(mixed):
    law = mixture((0.3, 0.7), (-0.4, 0.6), (0.3, 0.7)) if mixed else normal(0.6, 0.7)
    lower = 5.6
    upper = math.nextafter(lower, math.inf)
    weight = 0.7 if mixed else 1.0
    reference_tail = weight * norm.sf((lower - 0.6) / 0.7)
    reference_mass = weight * (upper - lower) / 0.7 * norm.pdf((lower - 0.6) / 0.7)
    assert law.survival(lower) == pytest.approx(reference_tail, rel=1e-6, abs=0)
    assert law.probability_between(lower, upper) == pytest.approx(reference_mass, rel=1e-6, abs=0)


def test_normal_quantile_keeps_finite_affine_result_when_product_overflows():
    law = normal(-1e308, 1e308)
    reference = -1 + norm.isf(0.01)
    for result in (law.quantile(0.99), law.isf(0.01)):
        assert math.isfinite(result)
        assert result / 1e308 == pytest.approx(reference, abs=1e-9, rel=1e-6)
        assert law.survival(result) == pytest.approx(0.01, rel=1e-6, abs=0)


@pytest.mark.parametrize("upper_tail,probability", [(False, 0.2), (True, 0.025)])
def test_multicomponent_quantile_is_finite_despite_overflowing_search_hints(
    upper_tail, probability
):
    law = mixture((0.25, 0.75), (1.2e308, 1.3e308), (0.2e308, 0.15e308))

    def reference_tail(x):
        function = norm.sf if upper_tail else norm.cdf
        return 0.25 * function((x - 1.2) / 0.2) + 0.75 * function((x - 1.3) / 0.15)

    reference = brentq(lambda x: reference_tail(x) - probability, 0, 1.79, xtol=1e-14)
    result = law.isf(probability) if upper_tail else law.quantile(probability)
    assert math.isfinite(result)
    assert result / 1e308 == pytest.approx(reference, abs=1e-9, rel=1e-6)
    actual = law.survival(result) if upper_tail else law.cdf(result)
    assert actual == pytest.approx(probability, abs=min(1e-10, 1e-6 * probability), rel=0)


def test_mixture_isf_preserves_explicit_tail_below_complement_precision():
    law = mixture((0.25, 0.75), (-0.4, 0.6), (0.3, 0.7))
    alpha = 1e-20
    reference = brentq(
        lambda x: 0.25 * norm.sf((x + 0.4) / 0.3) + 0.75 * norm.sf((x - 0.6) / 0.7) - alpha,
        0,
        10,
        xtol=1e-14,
    )
    result = law.isf(alpha)
    assert result == pytest.approx(reference, abs=1e-9, rel=1e-6)
    assert law.survival(result) == pytest.approx(alpha, rel=1e-6, abs=0)


@pytest.mark.parametrize("mixed", [False, True])
def test_sub_ulp_quantile_accepts_adjacent_float_sign_bracket(mixed):
    following = math.nextafter(1.0, math.inf)
    law = mixture((0.5, 0.5), (1.0, following), (1e-17, 1e-17)) if mixed else normal(1, 1e-17)
    probability = 0.5 if mixed else 0.9
    result = law.quantile(probability)
    assert result in (1.0, following)
    assert abs(law.cdf(result) - probability) > 1e-10
    lower, upper = math.nextafter(result, -math.inf), math.nextafter(result, math.inf)
    assert law.cdf(lower) <= probability <= law.cdf(upper)
    assert upper - lower < 1e-9 + 1e-6 * abs(result)


def test_flat_mixture_cdf_quantile_has_independent_value_bracket():
    law = mixture((0.5, 0.5), (-1.0, 10.0), (0.1, 0.2))
    # Equal weighted tails imply (q+1)/.1=(10-q)/.2, even where F rounds to .5.
    reference = (0.2 * -1 + 0.1 * 10) / (0.1 + 0.2)
    assert law.quantile(0.5) == pytest.approx(reference, abs=1e-9, rel=1e-6)


def test_flat_computed_mixture_cdf_does_not_certify_quantile_value():
    law = mixture((0.5, 0.5), (-1e308, 1e308), (1.0, 1.0))
    with pytest.raises(_PosteriorNumericalError) as caught:
        law.quantile(0.5)
    assert caught.value.code == "estimation.posterior.numerical_unavailable"
    assert caught.value.context["reason"] == "accuracy_unresolved"
    assert caught.value.context["probability"] == 0.5


def test_weighted_losses_stay_finite_when_unweighted_component_loss_overflows():
    maximum = sys.float_info.max
    law = mixture((0.25, 0.75), (maximum, -maximum), (maximum, maximum))
    large = norm.pdf(1) + norm.cdf(1)
    small = norm.pdf(1) - norm.sf(1)
    assert large > 1
    assert law.expected_positive_part(scale="linear") / maximum == pytest.approx(
        0.25 * large + 0.75 * small, abs=1e-10, rel=1e-6
    )
    assert law.expected_negative_part(scale="linear") / maximum == pytest.approx(
        0.25 * small + 0.75 * large, abs=1e-10, rel=1e-6
    )


def test_relative_downside_is_uncapped_and_never_exponentiated():
    law = mixture((0.3, 0.7), (-4.0, 0.2), (0.5, 0.8))
    reference = math.fsum(
        weight
        * quad(
            lambda z, mean=mean, sd=sd: max(-mean - sd * z, 0) * norm.pdf(z),
            -15,
            15,
            points=[-mean / sd],
            epsabs=1e-13,
        )[0]
        for weight, mean, sd in ((0.3, -4.0, 0.5), (0.7, 0.2, 0.8))
    )
    assert reference > 1
    assert law.expected_negative_part(scale="linear") == pytest.approx(
        reference, abs=1e-10, rel=1e-6
    )


def test_tiny_normal_loss_retains_original_large_units():
    law = normal(1e308, 1e307)
    reference_unit = (
        norm.pdf(10)
        * quad(lambda u: u * math.exp(-10 * u - u * u / 2), 0, math.inf, epsabs=1e-14)[0]
    )
    assert law.expected_negative_part(scale="linear") / 1e307 == pytest.approx(
        reference_unit, rel=1e-6, abs=0
    )


@pytest.mark.parametrize("mixed", [False, True])
def test_unrepresentable_quantile_has_designated_error_and_original_parameters(mixed):
    maximum = sys.float_info.max
    law = mixture((1.0,), (maximum,), (1e306,)) if mixed else normal(maximum, 1e306)
    with pytest.raises(_PosteriorNumericalError) as caught:
        law.isf(0.025)
    error = caught.value
    assert error.code == "estimation.posterior.numerical_unavailable"
    assert error.context["reason"] == "result_unrepresentable"
    assert error.context["operation"] == "isf"
    assert error.context["tail_probability"] == 0.025
    if mixed:
        assert error.context["means"] == (maximum,)
        assert error.context["sigmas"] == (1e306,)
    else:
        assert error.context["mean"] == maximum
        assert error.context["sigma"] == 1e306
    with pytest.raises(TypeError):
        cast(dict[str, object], error.context)["reason"] = "changed"
    assert law.cdf(maximum) == pytest.approx(0.5, abs=1e-10)


def test_unrepresentable_loss_has_designated_error_not_infinity():
    law = normal(sys.float_info.max, sys.float_info.max)
    with pytest.raises(_PosteriorNumericalError) as caught:
        law.expected_positive_part(scale="linear")
    assert caught.value.context["reason"] == "result_unrepresentable"
    assert caught.value.context["operation"] == "expected_positive_part"
    assert caught.value.context["scale"] == "linear"


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("cdf", (math.nan,)),
        ("survival", (math.nan,)),
        ("quantile", (0.0,)),
        ("isf", (math.nan,)),
        ("probability_between", (1.0, -1.0)),
    ],
)
def test_invalid_queries_are_distinct_from_numerical_inability(method, arguments):
    with pytest.raises(InvalidRequestError) as caught:
        getattr(normal(0, 1), method)(*arguments)
    assert caught.value.code == "estimation.posterior.invalid_request"


def test_relative_loss_rejects_log_scale_without_changing_density():
    law = mixture((0.25, 0.75), (-0.4, 0.6), (0.3, 0.7))
    with pytest.raises(InvalidRequestError) as caught:
        law.expected_negative_part(scale="log")
    assert caught.value.code == "estimation.posterior.invalid_request"
    assert law.cdf(0.6) == pytest.approx(0.25 * norm.cdf(1 / 0.3) + 0.375, abs=1e-10)
