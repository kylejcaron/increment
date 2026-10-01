"""Multiple-testing family procedures - BH, e-BH, FCR selected-interval
alpha, and registered raw-likelihood evidence."""

import math
from fractions import Fraction
from typing import Any, cast

import pytest
from scipy.stats import norm

from increment.decision import (
    ArmHypothesisKey,
    DecisionComputation,
    DecisionFailure,
    EValueEvidence,
    HypothesisKey,
    PValueEvidence,
    SegmentHypothesisKey,
)
from increment.errors import CapabilityError, DefinitionError, InvalidRequestError
from increment.estimation.family import (
    FamilyOutcome,
    bh_select,
    e_bh_select,
    family_discovery,
    select_family,
    select_sequential_family,
)
from increment.estimation.results import Estimate, LiftEstimate
from increment.estimation.sequential import AlwaysValid


def _lift_estimate(
    metric: str,
    group_id: str,
    mu: float,
    sigma: float,
    *,
    alternative: str = "two-sided",
    null_lift: float = 0.0,
    level: float = 0.95,
) -> LiftEstimate:
    """A log-scale LiftEstimate whose posterior is exactly Normal(mu, sigma)."""
    z = norm.ppf((1 + level) / 2)
    return LiftEstimate(
        metric=metric,
        group_id=group_id,
        method="unadjusted",
        method_role="decision",
        alternative=alternative,
        null_lift=null_lift,
        lift=Estimate(
            value=math.expm1(mu),
            lb=math.expm1(mu - z * sigma),
            ub=math.expm1(mu + z * sigma),
            level=level,
        ),
    )


def _typed_family_cells(
    cells: list[tuple[tuple[str, str], LiftEstimate]],
) -> tuple[list[tuple[ArmHypothesisKey, LiftEstimate]], DecisionComputation[Any]]:
    typed = []
    evidence: dict[HypothesisKey, PValueEvidence] = {}
    for _raw_key, row in cells:
        key = ArmHypothesisKey(row.metric, row.group_id, row.estimand)
        typed.append((key, row))
        if key not in evidence or row.method == "unadjusted":
            evidence[key] = PValueEvidence(key, row.method, row.p_value(), "normal")
    return typed, DecisionComputation(results=(), evidence=evidence, failures={})


def _raw_family(*, null=0, alternative="two-sided", c=None, t=None):
    from increment import SequentialCell, estimate_sequential
    from tests.sequential_cases import capture, records, registration

    reg = registration(
        cells=(
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                family=True,
                alternative=alternative,
                null_lift=null,
            ),
        )
    )
    policy = AlwaysValid(registration=reg)
    bundle = estimate_sequential(
        capture(
            reg,
            records(
                [0, 0, 0, 1] * 24 if c is None else c,
                [0, 1, 1, 1] * 24 if t is None else t,
            ),
        ),
        policy,
    )
    return policy, bundle


def test_bh_matches_hand_stepup():
    p = [0.001, 0.008, 0.039, 0.041, 0.60]
    sel, t = bh_select(p, q=0.10)
    assert sel == [0, 1, 2, 3] and t == pytest.approx(4 / 5 * 0.10)


def test_bh_empty_and_ties():
    assert bh_select([0.9, 0.8], 0.10) == ([], 0.0)
    sel, _ = bh_select([0.04, 0.04], 0.10)  # tie at the boundary: both in
    assert sel == [0, 1]


@pytest.mark.parametrize("selector", [bh_select, e_bh_select])
def test_family_select_q_refusal_uses_canonical_code(selector):
    with pytest.raises(InvalidRequestError) as exc_info:
        selector([0.1, 0.2] if selector is bh_select else [0.0, 0.0], 0.0)
    assert exc_info.value.code == "estimation.family.bh_select_q_finite"


def test_bh_refuses_out_of_range_p():
    with pytest.raises(ValueError):
        bh_select([0.1, 1.2], 0.10)
    with pytest.raises(ValueError):
        bh_select([0.1, -0.01], 0.10)
    with pytest.raises(ValueError):
        bh_select([0.1, math.nan], 0.10)
    with pytest.raises(ValueError):
        bh_select([0.1, math.inf], 0.10)


def test_bh_refuses_bad_q():
    with pytest.raises(ValueError):
        bh_select([0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        bh_select([0.1, 0.2], 1.5)
    with pytest.raises(ValueError):
        bh_select([0.1, 0.2], math.nan)


def test_bh_selects_everything_at_q_one():
    p = [0.9, 0.05, 0.3]
    sel, t = bh_select(p, q=1.0)
    assert sel == [0, 1, 2] and t == pytest.approx(1.0)


def test_bh_empty_input():
    assert bh_select([], 0.10) == ([], 0.0)


def test_select_family_propagates_direct_conservative_fcr_alpha():
    sigma = 0.02
    cells = [
        (
            (f"m_{i}", "treatment"),
            _lift_estimate(
                f"m_{i}",
                "treatment",
                0.20 if i == 0 else 0.0,
                sigma,
            ),
        )
        for i in range(7)
    ]

    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )

    assert outcome.selected == {ArmHypothesisKey("m_0", "treatment", "itt")}
    # R=1 of m=7 at q=0.10: both reported levels are the conservative q/m,
    # the greatest double not exceeding the exact rational 1/70.
    alpha = outcome.fcr_alpha
    assert alpha is not None
    assert Fraction(alpha) <= Fraction(1, 70) < Fraction(math.nextafter(alpha, math.inf))
    assert outcome.realized_threshold == alpha


def test_selected_alpha_is_the_greatest_double_not_exceeding_r_q_over_m():
    """R=3 of m=10 at q=0.10: the selected-interval alpha is R*q/m = 3/100,
    rounded downward to the greatest double that does not exceed it, and it
    is the same number the outcome reports as BH's realized threshold."""
    sigma = 0.02
    cells = [
        (
            (f"m_{i}", "treatment"),
            _lift_estimate(f"m_{i}", "treatment", 0.20 if i < 3 else 0.0, sigma),
        )
        for i in range(10)
    ]
    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )
    assert len(outcome.selected) == 3
    alpha = outcome.fcr_alpha
    assert alpha is not None
    assert Fraction(alpha) <= Fraction(3, 100) < Fraction(math.nextafter(alpha, math.inf))
    assert alpha == outcome.realized_threshold
    assert outcome.capped is False


def test_selected_alpha_at_extreme_q_is_finite_positive_and_conservative():
    """q=1e-300 with three cells whose p-values sit below 3e-301: the alpha
    is a finite positive double enclosing 3e-301 from below, not zero and
    not a level rounded to 1."""
    sigma = 0.02
    cells = [
        (
            (f"m_{i}", "treatment"),
            _lift_estimate(f"m_{i}", "treatment", 0.75 if i < 3 else 0.0, sigma),
        )
        for i in range(10)
    ]
    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=1e-300, inference=None, nominal_alpha=0.05, computation=computation
    )
    assert len(outcome.selected) == 3
    alpha = outcome.fcr_alpha
    assert alpha is not None and math.isfinite(alpha) and alpha > 0.0
    exact = Fraction(1e-300) * 3 / 10
    assert Fraction(alpha) <= exact < Fraction(math.nextafter(alpha, math.inf))


@pytest.mark.slow
def test_actual_bernoulli_evidence_encloses_independent_rational_likelihood():
    from fractions import Fraction as F

    from increment.estimation._certified import log_interval

    _, bundle = _raw_family(c=[0, 0], t=[1, 1])
    value = next(iter(bundle.evidence.values()))
    assert isinstance(value, EValueEvidence)
    actual = value.certificate.log_e
    assert actual is not None
    # Independent calculation: each Beta(1,1) predictive is 1/3;
    # the equality-null likelihood maximum is (1/2)^4.
    expected = log_interval(F(16, 9))
    assert actual.lo <= expected.hi and expected.lo <= actual.hi
    assert not bundle.results[0].stat_sig()


@pytest.mark.slow
def test_extreme_actual_evidence_keeps_log_selection_when_exp_overflows():
    policy, bundle = _raw_family(c=[0] * 600, t=[1] * 600)
    value = next(iter(bundle.evidence.values()))
    assert isinstance(value, EValueEvidence)
    assert value.e_value is None
    assert value.log_e > 710
    selected = select_sequential_family(
        [(value.hypothesis, bundle.results[0])],
        policy.registration.q,
        policy,
        0.05,
        computation=bundle,
    )
    assert selected.selected == {value.hypothesis}


def test_e_bh_reduces_to_bh_on_reciprocals():
    e = [200.0, 40.0, 25.0, 1.2]
    assert e_bh_select([math.log(v) for v in e], 0.10) == bh_select([1 / v for v in e], 0.10)[0]


def test_e_bh_empty_and_all_weak():
    assert e_bh_select([], 0.10) == []
    assert e_bh_select([0.0, math.log(2.0)], 0.10) == []


def test_e_bh_preserves_negative_logs_and_refuses_nan():
    assert e_bh_select([0.0, -2.0], 0.10) == []
    with pytest.raises(ValueError):
        e_bh_select([0.0, math.nan], 0.10)
    assert e_bh_select([0.0, math.inf], 0.10) == [1]


def test_e_bh_allows_zero_e_value_but_never_selects_it():
    # A zero e-value is the weakest possible evidence (1/e = inf); it
    # should never be selected, but is not itself a refusal.
    sel = e_bh_select([math.log(200.0), -math.inf], 0.10)
    assert sel == [0]


def test_e_bh_step_never_admits_evidence_below_the_bound_at_its_level():
    """Step k of m admits nothing below ``(-log_interval(q*k/m)).hi``, the bound
    a count boundary at level q*k/m is built on and a ratio's denominator cap is
    certified against; evidence just above it is admitted, ties across the
    roster included, and abstentions still count in m."""
    from increment.estimation._certified import log_interval

    q, m = Fraction(1, 20), 3
    bound = {k: (-log_interval(q * k / m)).hi for k in (1, 2)}
    below, above = Fraction(1, 10**200), Fraction(1, 10**9)
    abstain = float("-inf")
    assert e_bh_select([bound[1] - below, abstain, abstain], q) == []
    assert e_bh_select([bound[1] + above, abstain, abstain], q) == [0]
    assert e_bh_select([bound[2] + above, bound[2] - below, abstain], q) == []
    assert e_bh_select([bound[2] + above, bound[2] + above, abstain], q) == [0, 1]


def test_select_family_fixed_horizon_honors_one_sided_alternative():
    """4 secondaries under a declared "greater" plan: m_a's z=2.05 gives a
    one-sided p of ~0.0202, which clears the q=0.10/m=4 BH rank-1
    threshold (0.025) -- but the naive always-two-sided-against-zero
    p-value (~0.0404) does not. select_family's fixed-horizon branch must
    select m_a using the row's own one-sided alternative, not the naive
    two-sided figure (which would select nothing)."""
    sigma = 0.05
    mu_signal = 2.05 * sigma  # one-sided p ~ 0.0202; two-sided ~ 0.0404
    cells = [
        (
            ("m_a", "treatment"),
            _lift_estimate("m_a", "treatment", mu_signal, sigma, alternative="greater"),
        ),
        (
            ("m_b", "treatment"),
            _lift_estimate("m_b", "treatment", 0.0, sigma, alternative="greater"),
        ),
        (
            ("m_c", "treatment"),
            _lift_estimate("m_c", "treatment", 0.0, sigma, alternative="greater"),
        ),
        (
            ("m_d", "treatment"),
            _lift_estimate("m_d", "treatment", 0.0, sigma, alternative="greater"),
        ),
    ]

    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )

    assert outcome.selected == {ArmHypothesisKey("m_a", "treatment", "itt")}
    # R=1, m=4, q=0.10 -> BH cutoff 0.025, below the nominal 0.05, so the
    # cap does not bind and the FCR alpha is BH's own cutoff.
    assert outcome.fcr_alpha == pytest.approx(0.025, rel=1e-12)
    assert outcome.capped is False

    # The naive always-two-sided-against-zero computation would have
    # selected nothing at this same q/m -- proves the assertion above
    # isn't a coincidental pass.
    naive_p = [
        2.0 * min(norm.cdf(-mu / sigma), norm.sf(-mu / sigma)) for mu in (mu_signal, 0.0, 0.0, 0.0)
    ]
    assert bh_select(naive_p, 0.10) == ([], 0.0)


def test_select_family_fixed_horizon_honors_declared_margin():
    """A secondary with a nonzero declared null_lift=0.05 (margin): its
    observed 6% lift is far from 0 (two-sided-against-zero p ~ 0.0036,
    which the naive computation would select) but not significantly past
    the declared 5% margin (p ~ 0.318 against null_lift=0.05) --
    select_family's fixed-horizon branch must test against the row's own
    shifted null and correctly select nothing."""
    sigma = 0.02
    mu_margin = math.log1p(0.06)
    cells = [
        (
            ("m_a", "treatment"),
            _lift_estimate(
                "m_a", "treatment", mu_margin, sigma, alternative="greater", null_lift=0.05
            ),
        ),
        (
            ("m_b", "treatment"),
            _lift_estimate("m_b", "treatment", 0.0, sigma, alternative="greater"),
        ),
        (
            ("m_c", "treatment"),
            _lift_estimate("m_c", "treatment", 0.0, sigma, alternative="greater"),
        ),
        (
            ("m_d", "treatment"),
            _lift_estimate("m_d", "treatment", 0.0, sigma, alternative="greater"),
        ),
    ]

    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )

    assert outcome.selected == set()
    assert outcome.fcr_alpha is None

    # The naive always-two-sided-against-zero computation would have
    # wrongly selected m_a (a false discovery relative to its own
    # declared margin) -- proves the assertion above isn't a
    # coincidental pass.
    naive_p = [
        2.0 * min(norm.cdf(-mu / sigma), norm.sf(-mu / sigma)) for mu in (mu_margin, 0.0, 0.0, 0.0)
    ]
    naive_selected, _ = bh_select(naive_p, 0.10)
    assert naive_selected == [0]


def test_select_family_aborts_on_cluster_robust_absolute_margin_row():
    """A fixed-horizon clustered absolute-margin secondary is constructible
    (only sequential null_abs is refused at estimation) and reaches BH
    selection, where its p_value() is undefined -- the additive interval is
    a t-quantile pair, not a Normal tail. select_family must fail loud with
    a family-scoped message naming the offending cell, not surface a bare
    row-scoped ValueError from inside the list comprehension, and must not
    silently drop the row (which would let the rest of the family select as
    if it never existed)."""
    good = _lift_estimate("m_a", "treatment", 0.1, 0.02, alternative="greater")
    bad = object()
    good_key = ArmHypothesisKey("m_a", "treatment", "itt")
    bad_key = ArmHypothesisKey("m_b", "treatment", "itt")
    cells = [(good_key, good), (bad_key, bad)]
    computation = DecisionComputation(
        results=(),
        evidence={good_key: PValueEvidence(good_key, "unadjusted", 0.001, "normal")},
        failures={
            bad_key: DecisionFailure(
                bad_key,
                "evidence.p_value.unavailable",
                {"metric": "m_b", "group_id": "treatment", "reason": "absolute_margin"},
            )
        },
    )

    with pytest.raises(CapabilityError) as raised:
        select_family(
            cells,
            q=0.10,
            inference=None,
            nominal_alpha=0.05,
            computation=computation,
        )

    error = raised.value
    assert error.code == "family.evidence.incomplete"
    assert error.context["failed"] == (bad_key,)
    assert error.context["missing"] == ()


def test_select_family_keeps_a_zero_variance_cell_as_a_conservative_nonrejection():
    """A cell whose only failure is the zero-variance guard
    (estimation.engine.lift_guard) stays IN the family as a guaranteed
    non-rejection instead of aborting selection for the rest of the
    family. m (n_family) is unchanged; the degenerate cell can never be
    selected. (Non-positive mean no longer reaches this path -- see
    the row-based additive-only fix in engine.py; this test guards the
    REMAINING hazard that still funnels through DecisionFailure.)"""
    good = _lift_estimate("m_a", "treatment", 0.1, 0.02, alternative="greater")
    good_key = ArmHypothesisKey("m_a", "treatment", "itt")
    degenerate_key = ArmHypothesisKey("m_b", "treatment", "itt")
    cells = [(good_key, good), (degenerate_key, object())]
    computation = DecisionComputation(
        results=(),
        evidence={good_key: PValueEvidence(good_key, "unadjusted", 0.001, "normal")},
        failures={
            degenerate_key: DecisionFailure(
                degenerate_key,
                "estimation.engine.lift_guard",
                {
                    "metric": "m_b",
                    "group_id": "treatment",
                    "method": "unadjusted",
                    "reason": "Degenerate data: an arm has zero variance",
                },
            )
        },
    )
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )
    assert outcome.n_family == 2
    assert degenerate_key not in outcome.selected
    assert good_key in outcome.selected


@pytest.mark.parametrize("reason", ["nonpositive_mean", "zero_variance", "extreme_ratio"])
def test_select_family_keeps_a_breakout_outcome_exclusion_as_a_conservative_nonrejection(reason):
    """Breakout keys its OUTCOME-based unavailable cells with its own
    codes (breakout.nonpositive_mean from its raw pre-screen;
    breakout.zero_variance / breakout.extreme_ratio from estimate_lift's
    lift guards) rather than estimation.engine.lift_guard. run() and
    run_breakout() must not disagree about whether such a cell aborts
    the rest of its family -- each gets the non-rejection treatment."""
    good = _lift_estimate("m_a", "treatment", 0.1, 0.02, alternative="greater")
    good_key = SegmentHypothesisKey("m_a", "treatment", "itt", "country", "US")
    degenerate_key = SegmentHypothesisKey("m_b", "treatment", "itt", "country", "GB")
    cells = [(good_key, good), (degenerate_key, None)]
    computation = DecisionComputation(
        results=(),
        evidence={good_key: PValueEvidence(good_key, "unadjusted", 0.02, "normal")},
        failures={
            degenerate_key: DecisionFailure(
                degenerate_key,
                f"breakout.{reason}",
                {
                    "metric": "m_b",
                    "group_id": "treatment",
                    "dimension": "country",
                    "dimension_value": "GB",
                    "reason": reason,
                },
            )
        },
    )
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )
    assert outcome.n_family == 2
    assert degenerate_key not in outcome.selected
    assert good_key in outcome.selected


@pytest.mark.parametrize("reason", ["few_units", "no_control_arm"])
def test_select_family_still_aborts_on_a_design_based_breakout_exclusion(reason):
    """The design-based breakout reasons (arm counts, ancillary to the
    outcome) are NOT part of the non-rejection set; a family containing
    one still refuses as before."""
    good = _lift_estimate("m_a", "treatment", 0.1, 0.02, alternative="greater")
    good_key = SegmentHypothesisKey("m_a", "treatment", "itt", "country", "US")
    bad_key = SegmentHypothesisKey("m_b", "treatment", "itt", "country", "GB")
    computation = DecisionComputation(
        results=(),
        evidence={good_key: PValueEvidence(good_key, "unadjusted", 0.02, "normal")},
        failures={
            bad_key: DecisionFailure(
                bad_key,
                f"breakout.{reason}",
                {"metric": "m_b", "group_id": "treatment", "reason": reason},
            )
        },
    )
    with pytest.raises(CapabilityError) as raised:
        select_family(
            [(good_key, good), (bad_key, None)],
            q=0.10,
            inference=None,
            nominal_alpha=0.05,
            computation=computation,
        )
    assert raised.value.code == "family.evidence.incomplete"


def test_select_family_still_aborts_on_a_non_lift_guard_failure():
    """The conservative-nonrejection treatment is scoped to the shared
    lift-guard/breakout-prescreen codes; a different failure code must
    still abort the whole family -- this is the EXISTING
    test_select_family_aborts_on_cluster_robust_absolute_margin_row,
    cited here as the regression that proves the scoping stayed narrow."""
    good = _lift_estimate("m_a", "treatment", 0.1, 0.02, alternative="greater")
    good_key = ArmHypothesisKey("m_a", "treatment", "itt")
    bad_key = ArmHypothesisKey("m_b", "treatment", "itt")
    cells = [(good_key, good), (bad_key, object())]
    computation = DecisionComputation(
        results=(),
        evidence={good_key: PValueEvidence(good_key, "unadjusted", 0.001, "normal")},
        failures={
            bad_key: DecisionFailure(
                bad_key,
                "evidence.p_value.unavailable",
                {"metric": "m_b", "group_id": "treatment", "reason": "absolute_margin"},
            )
        },
    )
    with pytest.raises(CapabilityError) as raised:
        select_family(cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation)
    assert raised.value.code == "family.evidence.incomplete"


@pytest.mark.slow
def test_select_family_always_valid_honors_declared_null_lift():
    shifted, at_null = _raw_family(null=2)
    zero, beyond_null = _raw_family()
    key = ArmHypothesisKey("outcome", "treatment", "itt")
    accepted = select_sequential_family(
        [(key, at_null.results[0])], shifted.registration.q, shifted, 0.05, computation=at_null
    )
    rejected = select_sequential_family(
        [(key, beyond_null.results[0])], zero.registration.q, zero, 0.05, computation=beyond_null
    )
    assert not accepted.selected
    assert rejected.selected == {key}
    assert rejected.fcr_alpha is not None


def test_select_family_exposes_canonical_method_per_key():
    """`FamilyOutcome.canonical_method` records which method's row
    drove each key's decision -- "unadjusted" wins when present, else the
    first-encountered method for that key, mirroring the dedup rule
    itself."""
    sigma = 0.02
    mu = math.log1p(0.10)
    cells = [
        (("m_a", "treatment"), _lift_estimate("m_a", "treatment", mu, sigma)),
        (
            ("m_a", "treatment"),
            _lift_estimate("m_a", "treatment", mu, sigma).model_copy(update={"method": "cuped"}),
        ),
        (
            ("m_b", "treatment"),
            _lift_estimate("m_b", "treatment", 0.0, sigma).model_copy(update={"method": "cuped"}),
        ),
    ]
    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )
    assert outcome.canonical_method == {
        ArmHypothesisKey("m_a", "treatment", "itt"): "unadjusted",
        ArmHypothesisKey("m_b", "treatment", "itt"): "cuped",
    }


def test_fcr_alpha_is_capped_at_the_nominal_alpha():
    """BH's FCR cutoff is R*q/m, bounded by q and NOT by the nominal alpha.
    Once more than alpha/q of the family is selected it exceeds the nominal
    level, and a wider alpha is a NARROWER interval - so uncapped, a
    "corrected" row would be EASIER to call significant than an uncorrected
    one. The cap holds the interval at the nominal level while still
    reporting BH's own cutoff for disclosure.
    """
    sigma = 0.02
    mu = math.log1p(0.10)  # every cell a clear, identical signal -> all selected
    cells = [
        ((f"m_{i}", "treatment"), _lift_estimate(f"m_{i}", "treatment", mu, sigma))
        for i in range(4)
    ]

    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.50, inference=None, nominal_alpha=0.05, computation=computation
    )

    # R=4 of m=4 at q=0.50 -> BH's cutoff is 0.50, ten times the nominal.
    assert len(outcome.selected) == 4
    assert outcome.realized_threshold == pytest.approx(0.50)
    assert outcome.capped is True
    assert outcome.fcr_alpha == pytest.approx(0.05)
    # The property that matters: the level a selected interval is cut at is
    # never looser than the uncorrected read's.
    assert outcome.fcr_alpha is not None and outcome.fcr_alpha <= 0.05


def test_fcr_alpha_is_left_alone_when_the_cap_does_not_bind():
    """The cap must not silently tighten a family that never exceeded the
    nominal alpha - a sparse selection keeps BH's own cutoff exactly."""
    sigma = 0.02
    mu = math.log1p(0.10)
    cells = [((("m_0"), "treatment"), _lift_estimate("m_0", "treatment", mu, sigma))]
    cells += [
        ((f"m_{i}", "treatment"), _lift_estimate(f"m_{i}", "treatment", 0.0, sigma))
        for i in range(1, 10)
    ]

    cells, computation = _typed_family_cells(cells)
    outcome = select_family(
        cells, q=0.10, inference=None, nominal_alpha=0.05, computation=computation
    )

    # R=1 of m=10 at q=0.10 -> cutoff 0.01, well under the nominal 0.05.
    assert outcome.realized_threshold == pytest.approx(0.01)
    assert outcome.capped is False
    assert outcome.fcr_alpha == pytest.approx(0.01)


def test_family_discovery_policy_caps_a_nonsignificant_canonical_row():
    outcome = FamilyOutcome[tuple[str, str]](
        selected=frozenset({("m_a", "treatment")}),
        fcr_alpha=0.05,
        q=0.50,
        n_family=1,
        realized_threshold=0.50,
        capped=True,
        canonical_method={("m_a", "treatment"): "unadjusted"},
    )

    assert family_discovery(outcome, ("m_a", "treatment")) is True
    assert family_discovery(outcome, ("missing", "treatment")) is False


def test_a_metric_is_never_both_primary_and_secondary():
    """The other half of clause 2: a metric resolves to exactly one role, so
    the nested-Bonferroni primary path and the BH secondary family can never
    both claim the same metric's cells. Enforced at plan construction, so a
    double-corrected cell is unconstructable rather than merely unlikely."""
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, MeanMetric

    metrics = [
        MeanMetric(name=name, entity="user_id", fact=name, aggregation="avg_event")
        for name in ("a", "b")
    ]
    plan = compile_decision_plan(
        AnalysisPlan(primary=["a"], secondaries=["b"]),
        metrics,
        path="frame",
    )
    assert plan.procedures["a"].role == "primary"
    assert plan.procedures["b"].role == "secondary"

    # Declaring one metric under two roles is refused outright.
    with pytest.raises(DefinitionError) as raised:
        AnalysisPlan(primary=["a"], secondaries=["a"])
    assert raised.value.code == "definition.analysis.appears_both_metric"


def _typed_family_computation(
    p_values: dict[ArmHypothesisKey, float] | None = None,
    failures: dict[ArmHypothesisKey, DecisionFailure] | None = None,
) -> DecisionComputation[Any]:
    evidence: dict[ArmHypothesisKey, PValueEvidence | EValueEvidence] = {}
    for hypothesis, value in (p_values or {}).items():
        evidence[hypothesis] = PValueEvidence(hypothesis, "unadjusted", value, "normal")
    return DecisionComputation(
        results=(),
        evidence=cast("dict[HypothesisKey, PValueEvidence | EValueEvidence]", evidence),
        failures=cast("dict[HypothesisKey, DecisionFailure]", failures or {}),
    )


def test_select_family_aborts_when_any_compiled_hypothesis_failed_or_missing():
    good = ArmHypothesisKey("good", "treatment", "itt")
    failed = ArmHypothesisKey("failed", "treatment", "itt")
    missing = ArmHypothesisKey("missing", "treatment", "itt")
    failure = DecisionFailure(failed, "evidence.p_value.unavailable", {"reason": "degenerate"})
    computation = _typed_family_computation(p_values={good: 0.001}, failures={failed: failure})

    with pytest.raises(CapabilityError) as raised:
        select_family(
            [(good, object()), (failed, object()), (missing, object())],
            q=0.10,
            inference=None,
            nominal_alpha=0.05,
            computation=computation,
        )

    error = raised.value
    assert error.code == "family.evidence.incomplete"
    assert error.context["failed"] == (failed,)
    assert error.context["missing"] == (missing,)
    assert "failed" in str(error) and "missing" in str(error)


def test_select_family_family_size_is_compiled_hypotheses_not_method_rows():
    hypothesis = ArmHypothesisKey("metric", "treatment", "itt")
    computation = _typed_family_computation(p_values={hypothesis: 0.001})

    outcome = select_family(
        [(hypothesis, object()), (hypothesis, object())],
        q=0.10,
        inference=None,
        nominal_alpha=0.05,
        computation=computation,
    )

    assert outcome.n_family == 1
    assert outcome.selected == {hypothesis}


@pytest.mark.slow
def test_select_family_requires_evidence_matching_the_family_procedure():
    policy, e_computation = _raw_family()
    hypothesis = ArmHypothesisKey("outcome", "treatment", "itt")
    p_computation = _typed_family_computation(p_values={hypothesis: 0.001})
    with pytest.raises(CapabilityError) as e_exc:
        select_family(
            [(hypothesis, object())],
            q=0.10,
            inference=None,
            nominal_alpha=0.05,
            computation=e_computation,
        )
    assert e_exc.value.code == "family.evidence.incomplete"
    with pytest.raises(CapabilityError) as p_exc:
        select_sequential_family(
            [(hypothesis, object())],
            q=0.10,
            inference=policy,
            nominal_alpha=0.05,
            computation=p_computation,
        )
    assert p_exc.value.code == "sequential.source.invalid"


def test_family_absolute_margin_uses_absolute_axis():
    from increment.estimation.engine import _lift_decision_bundle

    key = ArmHypothesisKey("metric", "treatment", "itt")
    row = LiftEstimate(
        metric="metric",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        alternative="greater",
        lift=Estimate(value=0.1, log_mean=0.095, log_se=0.1),
        abs_diff=0.02,
        abs_se=0.005,
        null_abs=0.01,
    )
    computation = _lift_decision_bundle([row], inference=None)
    evidence = computation.evidence[key]
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == pytest.approx(norm.cdf((0.01 - 0.02) / 0.005))

    clustered = row.model_copy(update={"dof": 5.0, "reference_kind": "t", "reference_df": 5.0})
    clustered_computation = _lift_decision_bundle([clustered], inference=None)
    assert clustered_computation.failures[key].code == "evidence.p_value.unavailable"


def test_zero_relative_variance_keeps_the_family_roster_and_usable_discoveries():
    import pyarrow as pa

    from increment import Analysis, MetricSpec
    from increment.semantics.models import AnalysisPlan

    records = [
        {
            "unit": f"{arm}:{cluster}:{member}",
            "arm": arm,
            "cluster": f"{arm}:{cluster}",
            "zero": 1.0 + cluster % 3 if arm == "C" else 0.0,
            "borderline": 10.0 + 0.1 * cluster + (0.42 if arm == "T" else 0.0),
            "strong": 10.0 + 0.1 * cluster + (2.0 if arm == "T" else 0.0),
        }
        for arm in ("C", "T")
        for cluster in range(20)
        for member in range(3)
    ]
    names = ["zero", "borderline", "strong"]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(records),
        unit="unit",
        group="arm",
        control="C",
        cluster="cluster",
        metrics=[MetricSpec(name=name, type="mean") for name in names],
        plan=AnalysisPlan(secondaries=names, q=0.05),
    )
    rows: dict[str, LiftEstimate] = {}
    for row in analysis.run():
        assert isinstance(row, LiftEstimate)
        rows[row.metric] = row
    assert rows["zero"].relative_unavailable_reason == "zero_relative_variance"
    assert rows["zero"].discovery is False
    assert rows["strong"].discovery is True
    assert 0.05 * 2 / 3 < rows["borderline"].p_value() < 0.05
    assert rows["borderline"].discovery is False
