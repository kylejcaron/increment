"""Certified Bernoulli noncrossing screens for the bounded sequential evaluator.

These tests establish a one-sided elimination rule for Bernoulli prefixes. They
do not certify Gaussian rounding, public source plumbing, or the full manifest.
"""

from fractions import Fraction
from itertools import product

import pytest
from hypothesis import given
from hypothesis import strategies as st

from increment.estimation._sequential_likelihood import (
    BernoulliState,
    BetaPrior,
    bernoulli_evidence,
)
from increment.estimation.family import e_bh_select
from tests.estimation._sequential_events import bernoulli_noncrossing_screen
from tests.estimation._sequential_proof_oracle import (
    Alternative,
    Arm,
    evidence,
    log_bounds,
)

F = Fraction
ALTERNATIVES: tuple[Alternative, ...] = ("two-sided", "greater", "less")


@st.composite
def _bernoulli_states(draw):
    n = draw(st.integers(min_value=0, max_value=4))
    return Arm(n, draw(st.integers(min_value=0, max_value=n)))


def _assert_screen_encloses_exact_evidence(control, treatment, alternative):
    screen = bernoulli_noncrossing_screen(
        BernoulliState(control.n, control.successes),
        BernoulliState(treatment.n, treatment.successes),
        BetaPrior(1, 1),
        BetaPrior(1, 1),
        ratio=F(1),
        alternative=alternative,
        feasible_control=F(1, 2),
        feasible_treatment=F(1, 2),
        alpha=F(1, 10),
    )
    exact_log = log_bounds(evidence(control, treatment, alternative))[1]
    assert exact_log <= screen.log_e_upper


@pytest.mark.parametrize(
    ("control", "treatment", "alternative"),
    (
        pytest.param(Arm(0, 0), Arm(0, 0), "two-sided", id="control0-treatment0-two-sided"),
        pytest.param(Arm(0, 0), Arm(0, 0), "greater", id="control0-treatment0-greater"),
        pytest.param(Arm(0, 0), Arm(0, 0), "less", id="control0-treatment0-less"),
        pytest.param(Arm(1, 0), Arm(1, 0), "two-sided", id="control1-treatment1-two-sided"),
        pytest.param(Arm(1, 0), Arm(1, 0), "greater", id="control1-treatment1-greater"),
        pytest.param(Arm(1, 0), Arm(1, 0), "less", id="control1-treatment1-less"),
        pytest.param(Arm(1, 1), Arm(1, 1), "two-sided", id="control2-treatment2-two-sided"),
        pytest.param(Arm(1, 1), Arm(1, 1), "greater", id="control2-treatment2-greater"),
        pytest.param(Arm(1, 1), Arm(1, 1), "less", id="control2-treatment2-less"),
        pytest.param(Arm(2, 1), Arm(2, 1), "two-sided", id="control4-treatment4-two-sided"),
        pytest.param(Arm(2, 1), Arm(2, 1), "greater", id="control4-treatment4-greater"),
        pytest.param(Arm(2, 1), Arm(2, 1), "less", id="control4-treatment4-less"),
    ),
)
def test_screen_upper_encloses_independent_exact_evidence(control, treatment, alternative):
    _assert_screen_encloses_exact_evidence(control, treatment, alternative)


@given(
    control=_bernoulli_states(),
    treatment=_bernoulli_states(),
    alternative=st.sampled_from(ALTERNATIVES),
)
def test_screen_upper_encloses_exact_evidence_for_arbitrary_small_prefixes(
    control, treatment, alternative
):
    _assert_screen_encloses_exact_evidence(control, treatment, alternative)


@pytest.mark.parametrize("alternative", ALTERNATIVES)
@pytest.mark.parametrize("alpha", (F(1, 3), F(1, 10)))
def test_screened_prefix_cannot_cross_the_deployed_lower_certificate(alpha, alternative):
    prior = BetaPrior(1, 1)
    screened = 0
    for n in range(5):
        for control_successes, treatment_successes in product(range(n + 1), repeat=2):
            control = BernoulliState(n, control_successes)
            treatment = BernoulliState(n, treatment_successes)
            screen = bernoulli_noncrossing_screen(
                control,
                treatment,
                prior,
                prior,
                ratio=F(1),
                alternative=alternative,
                feasible_control=F(1, 2),
                feasible_treatment=F(1, 2),
                alpha=alpha,
            )
            if not screen.proves_noncrossing:
                continue
            screened += 1
            certificate = bernoulli_evidence(
                control,
                treatment,
                prior,
                prior,
                ratio=F(1),
                alternative=alternative,
            )
            assert certificate.log_e is not None
            assert e_bh_select((certificate.log_e.lo,), alpha) == []
    assert screened > 0


def test_screen_requires_a_feasible_positive_likelihood_null_point():
    control = treatment = BernoulliState(2, 1)
    prior_control = prior_treatment = BetaPrior(1, 1)
    with pytest.raises(ValueError, match="null constraint"):
        bernoulli_noncrossing_screen(
            control,
            treatment,
            prior_control,
            prior_treatment,
            ratio=F(1),
            alternative="two-sided",
            feasible_control=F(1, 2),
            feasible_treatment=F(1, 3),
            alpha=F(1, 10),
        )
    with pytest.raises(ValueError, match="positive likelihood"):
        bernoulli_noncrossing_screen(
            control,
            treatment,
            prior_control,
            prior_treatment,
            ratio=F(1),
            alternative="greater",
            feasible_control=F(0),
            feasible_treatment=F(0),
            alpha=F(1, 10),
        )


def test_empty_prefix_is_safely_screened():
    screen = bernoulli_noncrossing_screen(
        BernoulliState(0, 0),
        BernoulliState(0, 0),
        BetaPrior(1, 1),
        BetaPrior(1, 1),
        ratio=F(1),
        alternative="two-sided",
        feasible_control=F(1, 2),
        feasible_treatment=F(1, 2),
        alpha=F(1, 2),
    )
    assert screen.proves_noncrossing
    assert screen.log_e_upper == 0
    assert screen.log_threshold_lower > 0


def test_large_count_screen_uses_count_summaries_without_expanding_observations():
    screen = bernoulli_noncrossing_screen(
        BernoulliState(10**12, 10**9),
        BernoulliState(4 * 10**12, 4 * 10**9),
        BetaPrior(1, 1),
        BetaPrior(1, 1),
        ratio=F(1),
        alternative="two-sided",
        feasible_control=F(1, 1000),
        feasible_treatment=F(1, 1000),
        alpha=F(1, 20),
    )
    assert screen.proves_noncrossing
