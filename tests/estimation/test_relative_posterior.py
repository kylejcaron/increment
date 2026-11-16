"""Independent integral and finite-precision regressions for relative posteriors.

Constants come from the 50/70-digit latent and 60/90-digit direct-density
calculations retained in relative_student_t_reference.json and
relative_student_t_direct_reference.json, with extreme/adversarial references
in relative_student_t_extreme_reference.json and relative_student_t_adversarial.json.
The oracles do not import this module's integration implementation.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import sys
from fractions import Fraction
from typing import cast

import pytest
from pydantic import ValidationError
from scipy.integrate import quad

from increment.errors import InvalidRequestError
from increment.estimation._relative_posterior import (
    RelativeStudentTPosterior,
    RelativeStudentTState,
    _PosteriorNumericalError,
    _stationary_partition,
    _turning_intervals,
)
from increment.estimation.priors import StudentTPrior


def posterior(x=0.2, s=0.1, nu=4.0, a=0.1):
    return RelativeStudentTPosterior(
        RelativeStudentTState(
            version=1, likelihood_point=x, likelihood_se=s, prior=StudentTPrior(nu=nu, scale=a)
        )
    )


@pytest.mark.parametrize(
    "x,s,nu,a,cdf,sf,positive,negative,mean,median",
    [
        (
            0.2,
            0.1,
            4,
            0.1,
            0.07220452532489161,
            0.5482432118345988,
            0.11645322009682455,
            0.0022799424866667902,
            0.11417327761015776,
            0.1097574451070751,
        ),
        (
            1,
            0.2,
            1,
            1e-6,
            0.4763388952961524,
            0.04713990418382563,
            0.04269047166558311,
            0.0000030331596225136573,
            0.0426874385059606,
            7.817077052581818e-8,
        ),
        (
            0,
            0.1,
            0.5,
            0.1,
            0.5,
            0.07631039656813454,
            0.02738013194190527,
            0.02738013194190527,
            0,
            0,
        ),
        (
            0.2,
            0.1,
            200,
            0.1,
            0.07845658497227087,
            0.5013997604308205,
            0.10287940243800838,
            0.00250598541034699,
            0.10037341702766139,
            0.10024895122913607,
        ),
    ],
)
def test_student_t_relative_posterior_matches_independent_integrals(
    x, s, nu, a, cdf, sf, positive, negative, mean, median
):
    distribution = posterior(x, s, nu, a)
    assert distribution.cdf(0) == pytest.approx(cdf, abs=1e-10)
    assert distribution.survival(0.1) == pytest.approx(sf, abs=1e-10)
    p = distribution.expected_positive_part(scale="linear")
    n = distribution.expected_negative_part(scale="linear")
    assert p == pytest.approx(positive, abs=1e-10, rel=1e-6)
    assert n == pytest.approx(negative, abs=1e-10, rel=1e-6)
    assert p - n == pytest.approx(mean, abs=1e-10, rel=1e-6)
    assert distribution.quantile(0.5) == pytest.approx(median, abs=1e-9, rel=1e-6)


def test_student_t_conflict_preserves_every_posterior_mode():
    distribution = posterior(1, 0.2, 1, 0.01)

    # Direct effect-coordinate integration, independent of latent precision.
    def density(r):
        return math.exp(-0.5 * ((r - 1) / 0.2) ** 2) / (1 + (r / 0.01) ** 2)

    minimum = 0.08640928069242707
    z = math.fsum(
        quad(density, lo, hi, epsabs=1e-16, epsrel=1e-12)[0]
        for lo, hi in [(-3, 0), (0, minimum), (minimum, 1), (1, 5)]
    )
    left = quad(density, -3, minimum, points=[0], epsabs=1e-16, epsrel=1e-12)[0] / z
    assert 0.001 < left < 0.01
    assert distribution.cdf(minimum) == pytest.approx(left, abs=1e-10)
    assert distribution.survival(minimum) == pytest.approx(1 - left, abs=1e-10)


def test_student_t_nu_two_broad_prior_and_narrow_interval():
    distribution = posterior(-0.3, 0.07, 2, 3)

    def kernel(z):
        r = -0.3 + 0.07 * z
        return math.exp(-z * z / 2) * (1 + r * r / 18) ** -1.5

    z = quad(kernel, -15, 15, epsabs=1e-13)[0]
    reference = (
        kernel(((-0.2 - 0.199999999999) / 2 + 0.3) / 0.07) * (-0.199999999999 + 0.2) / (0.07 * z)
    )
    actual = distribution.probability_between(-0.2, -0.199999999999)
    assert actual == pytest.approx(reference, rel=1e-6, abs=1e-20)


@pytest.mark.parametrize(
    "lower,upper,reference",
    [
        (-0.05, 0.05, 0.20535657153278085055),
        (0.05, 0.050000000001, 3.9829515711049156438e-12),
        (0.8, 0.801, 1.7658425032206112744e-13),
    ],
)
def test_student_t_conditional_interval_matches_independent_reference(lower, upper, reference):
    assert posterior().probability_between(lower, upper) == pytest.approx(
        reference, rel=1e-6, abs=1e-20
    )


def test_student_t_small_tail_uses_direct_survival():
    distribution = posterior()
    q = distribution.isf(1e-12)
    assert q == pytest.approx(0.8146953329581297, abs=1e-9, rel=1e-6)
    assert distribution.survival(q) == pytest.approx(1e-12, rel=1e-6, abs=0)
    assert distribution.quantile(0.025) == pytest.approx(-0.03542144354231888, abs=1e-9)
    assert distribution.quantile(0.975) == pytest.approx(0.28742976346985185, abs=1e-9)


@pytest.mark.parametrize(
    "tail,reference",
    [(1e-20, 8.476863201364703), (1e-40, 12.625587395611868), (1e-100, 20.7336702371218)],
)
def test_student_t_survival_quantile_keeps_extreme_tail(tail, reference):
    distribution = posterior(0, 1, 4, 1)
    q = distribution.isf(tail)
    assert q == pytest.approx(reference, abs=1e-9, rel=1e-6)
    assert distribution.survival(q) == pytest.approx(tail, rel=1e-6, abs=0)


def test_student_t_loss_budget_uses_original_units():
    # Independent Tricomi-U identity, evaluated at 60 and 90 digits.
    distribution = posterior(0, 1e200, 0.5, 1e100)
    reference = 3.4568366951814667278e149
    assert distribution.expected_positive_part(scale="linear") == pytest.approx(reference, rel=1e-6)
    assert distribution.expected_negative_part(scale="linear") == pytest.approx(reference, rel=1e-6)


@pytest.mark.slow
def test_student_t_tiny_prior_preserves_distant_mass_and_loss():
    distribution = posterior(7, 0.2, 1, 1e-260)
    assert distribution.cdf(0) == pytest.approx(0.0001512264838030503, abs=1e-10)
    assert distribution.expected_negative_part(scale="linear") == pytest.approx(
        5.708357881496948e-262, rel=1e-6, abs=0
    )
    assert distribution.expected_positive_part(scale="linear") == pytest.approx(
        6.986429573345983, rel=1e-6
    )
    narrow = posterior(0.2, 0.1, 1e-8, 1e-100)
    assert narrow.cdf(0) == pytest.approx(0.48777570951027486, abs=1e-10)
    assert narrow.survival(0.2) == pytest.approx(0.007115131303116894, abs=1e-10)
    assert posterior(0, 1, 1, 1e-310).quantile(0.75) == pytest.approx(1e-310, rel=1e-6, abs=1e-316)


@pytest.mark.parametrize(
    "x,count",
    [
        (float.fromhex("0x1.2548eb9151e84p+1"), 1),
        (float.fromhex("0x1.2548eb9151e85p+1"), 1),
        (float.fromhex("0x1.2548eb9151e86p+1"), 3),
    ],
)
def test_stationary_partition_resolves_near_double_root(x, count):
    n, q, t2 = Fraction(1), Fraction(0.2886751345948129) ** 2, Fraction(x) ** 2
    coefficients = (n + 1, 2 * n + 1 - q - t2, n - 2 * q, -q)
    assert _turning_intervals(coefficients, 8) is None
    assert _turning_intervals(coefficients, 16) is not None
    roots, _ = _stationary_partition(coefficients)
    assert len(roots) == count


def test_stationary_partition_retains_repeated_and_near_triple_roots():
    for coefficients, expected in [
        ((Fraction(2), Fraction(-7, 3), Fraction(5, 6), Fraction(-1, 12)), [2, 6]),
        ((Fraction(27, 16), Fraction(-27, 16), Fraction(9, 16), Fraction(-1, 16)), [3]),
    ]:
        roots, turns = _stationary_partition(coefficients)
        assert [math.exp(sum(root) / 2) for root in roots] == pytest.approx(expected, rel=1e-13)
        expected_turns = [2, 14 / 3] if len(expected) == 2 else [3]
        assert [math.exp(sum(bounds) / 2) for bounds in turns] == pytest.approx(
            expected_turns, rel=1e-13
        )
    n = Fraction(11, 16)
    q = n * Fraction(0.30151134457776363) ** 2
    roots, _ = _stationary_partition((n + 1, 2 * n + 1 - q - 4, n - 2 * q, -q))
    assert len(roots) == 1
    assert sum(roots[0]) / 2 == pytest.approx(1.0986080330883388056, abs=1e-13)
    assert posterior(2, 1, float(n), 0.30151134457776363).cdf(0) == pytest.approx(
        0.16089945855373531335, abs=1e-10
    )


def test_student_t_finite_endpoint_checks_are_query_local():
    distribution = posterior(0, 1e308, 4, 1e308)
    before = (
        distribution.cdf(0),
        distribution.expected_positive_part(scale="linear"),
        distribution.isf(0.01),
    )
    assert before[0] == pytest.approx(0.5, abs=1e-10)
    assert before[1] == pytest.approx(2.841370736311733e307, rel=1e-6)
    assert before[2] == pytest.approx(1.7229959640880685e308, rel=1e-6)
    with pytest.raises(_PosteriorNumericalError) as caught:
        distribution.isf(1e-12)
    assert caught.value.context["reason"] == "result_unrepresentable"
    assert caught.value.context["operation"] == "isf"
    assert caught.value.context["tail_probability"] == 1e-12
    assert before == (
        distribution.cdf(0),
        distribution.expected_positive_part(scale="linear"),
        distribution.isf(0.01),
    )
    with pytest.raises(_PosteriorNumericalError) as caught:
        distribution.quantile(0.5)
    assert caught.value.context["reason"] == "accuracy_unresolved"


def test_student_t_flat_cdf_refuses_unresolved_value_bracket():
    distribution = posterior(16, 1, 1, 1e-53)

    def available():
        return (
            distribution.cdf(0),
            distribution.expected_positive_part(scale="linear"),
            distribution.expected_negative_part(scale="linear"),
            distribution.quantile(0.5),
            distribution.quantile(0.025),
            distribution.isf(0.025),
        )

    before = available()
    defining_state = distribution.state.model_dump_json()
    with pytest.raises(_PosteriorNumericalError) as caught:
        distribution.quantile(0.44919976139382767)
    assert caught.value.code == "estimation.posterior.numerical_unavailable"
    assert caught.value.context["reason"] == "accuracy_unresolved"
    assert caught.value.context["probability"] == 0.44919976139382767
    assert available() == before
    assert distribution.state.model_dump_json() == defining_state


@pytest.mark.parametrize(
    "prior_scale,cdf_lower,cdf_upper",
    [
        (0.1, 0.499999997034172067085, 0.500000001463321119067),
        # Direct density in likelihood coordinates, refined at 60/85 digits.
        (1e-9, 0.499999997801368760165, 0.500000002230517812146),
    ],
)
def test_student_t_adjacent_float_median_accepts_value_enclosure(prior_scale, cdf_lower, cdf_upper):
    distribution = posterior(1, 1e-8, 4, prior_scale)
    q = distribution.quantile(0.5)
    assert q in (0.9999999999999994, 0.9999999999999996)
    assert abs(distribution.cdf(q) - 0.5) > 1e-10
    lower = distribution.cdf(0.9999999999999994)
    upper = distribution.cdf(0.9999999999999996)
    assert lower == pytest.approx(cdf_lower, abs=1e-10, rel=0)
    assert upper == pytest.approx(cdf_upper, abs=1e-10, rel=0)
    assert lower < 0.5 < upper


@pytest.mark.parametrize(
    "method,probability,cdf_target,lower,upper,cdf_lower,cdf_upper",
    [
        (
            "quantile",
            0.025,
            0.025,
            0.999999999999998,
            0.9999999999999981,
            0.022836577591835591508,
            0.029554674266138361913,
        ),
        (
            "quantile",
            0.5,
            0.5,
            0.9999999999999999,
            1.0,
            0.455799330626007186036,
            0.500000000000001994711,
        ),
        (
            "isf",
            0.025,
            0.975,
            1.0000000000000018,
            1.000000000000002,
            0.962162939890559314326,
            0.977163422408164950130,
        ),
    ],
)
def test_student_t_collapsed_asinh_bracket_preserves_physical_quantile(
    method, probability, cdf_target, lower, upper, cdf_lower, cdf_upper
):
    # Independent direct density in likelihood coordinates, 60/85 digits.
    distribution = posterior(1, 1e-15, 4, 1e-20)
    result = getattr(distribution, method)(probability)
    assert result in (lower, upper)
    actual_lower, actual_upper = distribution.cdf(lower), distribution.cdf(upper)
    assert actual_lower == pytest.approx(cdf_lower, abs=1e-10, rel=0)
    assert actual_upper == pytest.approx(cdf_upper, abs=1e-10, rel=0)
    assert actual_lower < cdf_target < actual_upper


def test_student_t_sub_ulp_likelihood_brackets_round_outward():
    # Independent direct-density tails, separately scaled at 60/85 digits.
    distribution = posterior(1, 1e-17, 4, 1e-20)
    lower, upper = math.nextafter(1, -math.inf), math.nextafter(1, math.inf)
    assert distribution.quantile(0.5) == 1.0
    assert distribution.isf(0.025) in (1.0, upper)
    assert distribution.cdf(1) == pytest.approx(0.5, abs=1e-10, rel=0)
    assert distribution.cdf(lower) == pytest.approx(6.117602605641599e-29, rel=1e-6, abs=0)
    assert distribution.survival(upper) == pytest.approx(1.5550263204184895e-109, rel=1e-6, abs=0)


def test_student_t_construction_failure_context_survives_transport():
    with pytest.raises(_PosteriorNumericalError) as caught:
        posterior(sys.float_info.max, math.ulp(0.0), 4, 1)
    error = caught.value
    assert error.code == "estimation.posterior.numerical_unavailable"
    assert error.context["reason"] == "standardized_likelihood_unrepresentable"
    assert error.context["operation"] == "construction"
    assert error.context["likelihood_point"] == sys.float_info.max
    for restored in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
        assert restored.code == error.code
        assert restored.context == error.context
        with pytest.raises(TypeError):
            cast("dict[str, object]", restored.context)["reason"] = "changed"


def test_student_t_defining_state_excludes_runtime_cache_and_replays_once():
    distribution = posterior()
    state = distribution.state
    before = state.model_dump_json()
    assert set(state.model_dump()) == {"version", "likelihood_point", "likelihood_se", "prior"}
    assert distribution.cdf(0.1) == pytest.approx(1 - 0.5482432118345988, abs=1e-10)
    distribution.isf(1e-12)
    assert state.model_dump_json() == before
    restored = RelativeStudentTPosterior(RelativeStudentTState.model_validate_json(before))
    assert restored.cdf(0.12) == distribution.cdf(0.12)
    with pytest.raises(ValidationError):
        state.likelihood_point = 3
    for changes in (
        {"version": 2},
        {"likelihood_se": 0},
        {"likelihood_point": math.inf},
        {"prior": StudentTPrior(nu=4, scale=math.inf)},
    ):
        with pytest.raises((InvalidRequestError, ValidationError)):
            RelativeStudentTState.model_validate({**state.model_dump(), **changes})


@pytest.mark.parametrize("extra", [{"mu": 0.2}, {"type": "normal"}])
def test_student_t_state_rejects_extra_prior_fields_without_changing_legacy(extra):
    original = {"nu": 4.0, "scale": 0.1, "k": 80}
    prior = {**original, **extra}
    assert StudentTPrior.model_validate(prior).model_dump() == original
    payload = {
        "version": 1,
        "likelihood_point": 0.2,
        "likelihood_se": 0.1,
        "prior": prior,
    }
    with pytest.raises(InvalidRequestError) as caught:
        RelativeStudentTState.model_validate_json(json.dumps(payload))
    assert caught.value.code == "estimation.posterior.invalid_request"
    assert caught.value.context["operation"] == "state"
    assert caught.value.context["unknown_fields"] == tuple(extra)


@pytest.mark.parametrize("method", ["expected_positive_part", "expected_negative_part"])
def test_student_t_relative_loss_rejects_log_scale(method):
    with pytest.raises(InvalidRequestError):
        getattr(posterior(), method)(scale="log")


def test_student_t_query_domain_and_endpoints():
    distribution = posterior()
    assert distribution.cdf(-math.inf) == 0
    assert distribution.survival(math.inf) == 0
    assert distribution.probability_between(-math.inf, math.inf) == 1
    assert distribution.probability_between(0.1, 0.1) == 0
    for call in (
        lambda: distribution.quantile(0),
        lambda: distribution.isf(1),
        lambda: distribution.quantile(math.nan),
        lambda: distribution.cdf(math.nan),
        lambda: distribution.probability_between(1, 0),
    ):
        with pytest.raises(InvalidRequestError):
            call()
