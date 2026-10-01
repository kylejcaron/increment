"""Independent oracles for effect-dependent switchback planning."""

import math
from collections import UserDict
from fractions import Fraction
from types import MappingProxyType

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chi, norm, t

from increment.errors import InvalidRequestError
from increment.estimation.decision_types import ContrastDecisionProcedure
from increment.power import (
    SwitchbackBaseline,
    SwitchbackPowerResult,
)
from increment.power import (
    switchback_achieved_power as achieved,
)
from increment.power import (
    switchback_minimum_detectable_effect as mde,
)
from increment.power import (
    switchback_required_blocks_or_units as required,
)
from increment.power.switchback import _coefficients, _quadratic_roots, _variance
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)


def procedure(**changes):
    return ContrastDecisionProcedure.model_validate(
        dict(
            {
                "metric": "orders",
                "role": "primary",
                "alternative": "two-sided",
                "null_abs": 0,
                "alpha": 0.05,
            },
            **changes,
        )
    )


def baseline(**changes):
    return SwitchbackBaseline.model_validate(
        dict(
            {
                "assignment": SwitchbackAssignment(
                    sequence=IndependentBernoulliOrder(probability_ct=0.75),
                    window=SwitchbackWindow(washout_steps=0, observation_steps=1),
                ),
                "metric": "orders",
                "control_group": "control",
                "treatment_group": "treatment",
                "aggregation": "sum",
                "estimand": "retained_window_total_difference",
                "cycles_per_unit": 1,
                "delta_ref": 0.0,
                "sd_a": 1.0,
                "sd_g": 0.0,
                "rho": 0.0,
            },
            **changes,
        )
    )


def branch_baseline(p, u, v, var_ct=0.0, var_tc=0.0):
    # Independently enumerate reference A and G, including conditional noise.
    weights = (1 / (2 * p), 1 / (2 * (1 - p)))
    values = (u * weights[0], v * weights[1])
    ref = (u + v) / 2
    va = p * (values[0] - ref) ** 2 + (1 - p) * (values[1] - ref) ** 2
    va += var_ct / (4 * p) + var_tc / (4 * (1 - p))
    vg = p * (weights[0] - 1) ** 2 + (1 - p) * (weights[1] - 1) ** 2
    cov = p * (values[0] - ref) * (weights[0] - 1) + (1 - p) * (values[1] - ref) * (weights[1] - 1)
    sa, sg = math.sqrt(va), math.sqrt(vg)
    rho = cov / (sa * sg) if sa * sg else 0.0
    # Exact singular branches have correlation +/-1; avoid rounding above one.
    if var_ct == var_tc == 0 and sa * sg:
        rho = math.copysign(1.0, cov)
    return baseline(
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=p),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
        delta_ref=ref,
        sd_a=sa,
        sd_g=sg,
        rho=rho,
    )


def direct_power(nc, n=4, alpha=0.05):
    """Two-sided noncentral-t power by direct integration over the chi factor:
    ``T = (Z + nc) / W`` with ``W = sqrt(chi2_df / df)``, so each tail is
    ``E[Phi(sign * nc - critical * W)]``; independent of any library
    noncentral-t routine."""
    df = n - 1
    critical = t.isf(alpha / 2, df)
    scale = math.sqrt(df)

    def tail(sign):
        value, _ = quad(
            lambda w: norm.cdf(sign * nc - critical * w) * scale * chi.pdf(w * scale, df),
            0.0,
            np.inf,
            epsabs=1e-15,
            epsrel=1e-12,
            limit=200,
        )
        return value

    return tail(1.0) + tail(-1.0)


def assert_boundary(row, b, p, target):
    assert row.mde_abs is not None, row
    result = achieved(row.n, row.mde_abs, b, p)
    assert result.power is not None
    assert result.power >= target
    direction = math.inf if p.alternative == "less" else -math.inf
    prev = math.nextafter(row.mde_abs, direction)
    prior = achieved(row.n, prev, b, p)
    assert prior.power is None or prior.power < target


def test_power_and_required_n_replay_the_recorded_effect_at_a_large_null():
    b = baseline()
    p = procedure(null_abs=float(2**53))
    effect = 2**53 + 1
    result = achieved(20, effect, b, p)
    assert result.delta is not None
    replay = achieved(20, result.delta, result.baseline, result.procedure)
    assert result.power == replay.power
    assert result.power == pytest.approx(p.alpha)

    sizing = required(effect, b, p, target_power=0.8)
    assert sizing.delta is not None
    replay_sizing = required(sizing.delta, sizing.baseline, sizing.procedure, target_power=0.8)
    assert sizing.n is None
    assert sizing.n_unavailable_reason == "non_favorable_effect"
    assert sizing.n_unavailable_reason == replay_sizing.n_unavailable_reason


def test_independent_branch_oracle_and_two_analytic_roots():
    b = branch_baseline(0.75, 2.0, -2.0, 0.25, 0.25)
    p = procedure()
    assert b.sd_a**2 == pytest.approx(17 / 3)
    assert b.sd_g**2 == pytest.approx(1 / 3)
    assert b.rho * b.sd_a * b.sd_g == pytest.approx(-4 / 3)
    for delta in (0.0, 1.0, 4.0, 17 / 4, 20.0):
        row = achieved(4, delta, b, p)
        variance = ((delta - 4) ** 2 + 1) / 3
        assert row.standard_error is not None
        assert row.standard_error**2 * 4 == pytest.approx(variance)
        assert row.power == pytest.approx(direct_power(delta * math.sqrt(4 / variance)))
    target = direct_power(math.sqrt(18))
    assert target == pytest.approx(0.7978078634333717)
    a, cross, c = _coefficients(b, 0.0, 1)
    roots = _quadratic_roots(4, math.sqrt(18), a, cross, c)
    assert tuple(map(float, roots)) == pytest.approx((12 - math.sqrt(93), 12 + math.sqrt(93)))
    assert float(-c / cross) == pytest.approx(17 / 4)
    for root in roots:
        assert achieved(4, float(root), b, p).power == pytest.approx(target)
    row = mde(4, b, p, target_power=target)
    assert row.df == 3
    assert row.mde_abs == pytest.approx(12 - math.sqrt(93))
    assert_boundary(row, b, p, target)
    # Noncentral-t power is a planning approximation, not an exact finite-branch law.
    assert direct_power(math.sqrt(12)) == pytest.approx(0.6435432102)
    assert achieved(4, 17 / 4, b, p).power > target


@pytest.mark.parametrize("p,expected", [(0.5, 9), (0.75, 4 / 3)])
def test_same_general_summary_path_unbalanced_can_win(p, expected):
    b = branch_baseline(p, 3.0, -3.0)
    result = achieved(4, 4.0, b, procedure())
    assert result.standard_error is not None
    assert result.standard_error**2 * 4 == pytest.approx(expected)


def test_equal_branch_noise_unbalanced_is_worse():
    balanced = branch_baseline(0.5, 0.0, 0.0, 0.25, 0.25)
    unbalanced = branch_baseline(0.75, 0.0, 0.0, 0.25, 0.25)
    balanced_se = achieved(4, 4.0, balanced, procedure()).standard_error
    unbalanced_se = achieved(4, 4.0, unbalanced, procedure()).standard_error
    assert balanced_se is not None and unbalanced_se is not None
    assert unbalanced_se > balanced_se


def test_exact_double_root_and_tangent_power():
    roots = _quadratic_roots(3, 2.0, Fraction(1), Fraction(-1), Fraction(4))
    assert tuple(map(float, roots)) == (4.0, 4.0)
    b, p = baseline(sd_a=2.0, sd_g=1.0, rho=-0.5), procedure()
    target = achieved(3, 4.0, b, p).power
    assert target is not None
    result = mde(3, b, p, target_power=target)
    assert result.mde_abs == pytest.approx(4.0, rel=1e-6)
    assert_boundary(result, b, p, target)


@pytest.mark.parametrize("rho", [-1.0, 1.0, math.nextafter(-1.0, 0.0), math.nextafter(1.0, 0.0)])
def test_psd_and_near_psd_edges(rho):
    b, p = baseline(sd_a=2.0, sd_g=1.0, rho=rho), procedure()
    result = achieved(5, -2 * math.copysign(1.0, rho), b, p)
    if abs(rho) == 1:
        assert result.power is None
        assert result.power_unavailable_reason == "degenerate_zero_variance"
    else:
        assert result.standard_error is not None
        assert result.standard_error > 0
        assert result.power is not None


def test_singular_peak_excluded_but_ascending_root_available():
    b, p = baseline(sd_a=2.0, sd_g=1.0, rho=-1.0), procedure()
    result = mde(4, b, p, target_power=0.8)
    assert result.mde_abs is not None
    assert result.mde_abs < 2.0
    assert_boundary(result, b, p, 0.8)
    assert achieved(4, 2.0, b, p).power_unavailable_reason == "degenerate_zero_variance"
    assert required(2.0, b, p, target_power=0.8).n_unavailable_reason == "degenerate_zero_variance"


def test_constant_half_line_returns_first_float_not_zero():
    b, p = baseline(sd_a=0.0, sd_g=1.0), procedure()
    result = mde(4, b, p, target_power=0.1)
    assert result.mde_abs == math.ulp(0.0)
    assert result.standard_error is None
    assert result.standard_error_unavailable_reason == "unrepresentable"
    assert result.power is not None
    assert_boundary(result, b, p, 0.1)
    assert mde(4, b, p, target_power=0.99).mde_unavailable_reason == "unattained"


def test_fully_degenerate_and_valid_null_target():
    p = procedure()
    assert (
        mde(4, baseline(sd_a=0.0), p, target_power=0.01).mde_unavailable_reason
        == "degenerate_zero_variance"
    )
    assert mde(4, baseline(), p, target_power=0.01).mde_abs == 0.0


@pytest.mark.parametrize("at_asymptote", [True, False])
def test_asymptote_equal_or_below_target_is_unattained(at_asymptote):
    from increment.power.switchback import _favorable_power, _nc

    p = procedure()
    # sd_g = 1 bounds the noncentrality by sqrt(n) = 2; equality needs the
    # kernel's own value at that limit, not a second implementation's rounding.
    asymptote = _favorable_power(_nc(Fraction(1), Fraction(1), 4), 4, p)
    assert asymptote is not None
    assert asymptote == pytest.approx(direct_power(2.0))
    target = asymptote if at_asymptote else 0.99
    result = mde(4, baseline(sd_g=1.0), p, target_power=target)
    assert result.mde_abs is None
    assert result.mde_unavailable_reason == "unattained"


@pytest.mark.parametrize(
    "alternative,delta", [("two-sided", 0.5), ("greater", 0.5), ("less", -0.5)]
)
def test_required_integer_predecessor(alternative, delta):
    b, p = baseline(sd_g=0.1), procedure(alternative=alternative)
    result = required(delta, b, p, target_power=0.8)
    assert result.n is not None and result.power is not None
    assert result.n >= 2
    assert result.df == result.n - 1
    assert result.power >= 0.8
    predecessor = achieved(result.n - 1, delta, b, p)
    assert predecessor.df == result.n - 2
    assert predecessor.power is not None
    assert predecessor.power < 0.8
    assert result.mde_abs is None
    assert result.mde_unavailable_reason == "not_requested"


def test_required_two_units_with_one_cycle():
    result = required(1e4, baseline(), procedure(), target_power=0.8)
    assert result.n == 2
    assert result.df == 1
    assert result.power is not None and result.power >= 0.8


@pytest.mark.parametrize(
    "alternative,delta", [("greater", 0.0), ("greater", -1.0), ("less", 1.0), ("two-sided", 0.0)]
)
def test_required_non_favorable_refusal(alternative, delta):
    result = required(delta, baseline(), procedure(alternative=alternative), target_power=0.8)
    assert result.n_unavailable_reason == "non_favorable_effect"


@pytest.mark.parametrize("alternative,sign", [("greater", 1), ("less", -1)])
def test_large_offset_adjacent_public_float(alternative, sign):
    b = baseline(delta_ref=sign * 1e16, sd_a=0.1)
    p = procedure(null_abs=sign * 1e16, alternative=alternative)
    result = mde(4, b, p, target_power=0.8)
    assert result.mde_abs == math.nextafter(p.null_abs, sign * math.inf)
    assert_boundary(result, b, p, 0.8)


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_intermediate_squares_underflow_or_overflow_but_answer_finite(scale):
    b = baseline(sd_a=scale, sd_g=0.0)
    row = achieved(4, scale, b, procedure())
    assert row.standard_error == scale / 2
    assert row.power == pytest.approx(direct_power(2.0))
    result = mde(4, b, procedure(), target_power=0.8)
    assert_boundary(result, b, procedure(), 0.8)


def test_delta_minus_reference_overflow_and_cancellation_are_safe():
    b = baseline(delta_ref=-1e308, sd_a=1e308, sd_g=0.5, rho=-1.0)
    delta = math.nextafter(1e308, 0.0)
    result = achieved(4, delta, b, procedure(null_abs=delta))
    exact_sd = abs(Fraction(1e308) - (Fraction(delta) + Fraction(1e308)) / 2)
    assert result.standard_error == float(exact_sd / 2)
    assert result.power == 0.05
    zero = achieved(4, 1e308, b, procedure())
    assert zero.power_unavailable_reason == "degenerate_zero_variance"


def test_unrepresentable_effect_differs_from_unattained():
    b, p = baseline(sd_a=1e308), procedure(null_abs=1e308)
    result = mde(2, b, p, target_power=0.99)
    assert result.mde_unavailable_reason == "unrepresentable"


@pytest.mark.parametrize("alpha", [1e-50, 1e-250, math.nextafter(1.0, 0.0)])
def test_extreme_alpha(alpha):
    row = achieved(4, 0.5, baseline(), procedure(alpha=alpha))
    assert row.power is not None
    assert 0 <= row.power <= 1


def test_unrepresentable_two_sided_tail_is_unavailable_not_zero():
    p = procedure(alpha=math.ulp(0.0))
    for result in (
        achieved(4, 1.0, baseline(), p),
        required(1.0, baseline(), p, target_power=0.8),
    ):
        assert result.power is None
        assert result.power_unavailable_reason == "numerical_resolution"
    assert achieved(4, 1.0, baseline(), p).standard_error == 0.5
    assert mde(4, baseline(), p, target_power=0.8).mde_unavailable_reason == "numerical_resolution"


@pytest.mark.parametrize(("alternative", "null_abs"), [("two-sided", 0.0), ("greater", 0.25)])
def test_two_period_power_matches_the_one_dof_oracle_and_replays(alternative, null_abs):
    """n = 2 is one degree of freedom. At alpha 4e-10 the critical value is
    about 1.6e9 and sd_a = 1e-9 puts the noncentrality near 1e9, where each
    direction of the answer is ``erf(nc / (sqrt(2) hypot(c, 1)))``: the odd
    extension of the chi distribution misses it by at most ``Phi(-nc)``, as
    does the opposite tail. The minimum detectable effect inverts that closed
    form, and sizing crosses to two degrees of freedom at n = 3."""
    from scipy.special import erfinv

    from increment.estimation._tails import student_t_isf

    alpha, sd = 4e-10, 1e-9
    b, p = baseline(sd_a=sd), procedure(alpha=alpha, alternative=alternative, null_abs=null_abs)
    crit = student_t_isf(alpha / 2 if alternative == "two-sided" else alpha, 1.0)
    scale = math.sqrt(2.0) * math.hypot(crit, 1.0)

    effect = 1.0
    nc = (effect - null_abs) * math.sqrt(2.0) / sd
    result = achieved(2, effect, b, p)
    assert result.power == pytest.approx(math.erf(nc / scale), rel=1e-12, abs=0.0)

    row = mde(2, b, p, target_power=0.5)
    assert_boundary(row, b, p, 0.5)
    expected_mde = null_abs + float(erfinv(0.5)) * scale * sd / math.sqrt(2.0)
    assert row.mde_abs == pytest.approx(expected_mde, rel=1e-9, abs=0.0)

    sized = required(effect, b, p, target_power=0.9)
    assert sized.n == 3
    assert sized.power == achieved(3, effect, b, p).power
    assert result.power is not None and result.power < 0.9


def test_metadata_refusal_and_no_companion_mde():
    row = achieved(4, 0.5, baseline(), procedure(metric="different"))
    assert row.power_unavailable_reason == "metadata_mismatch"
    assert row.standard_error_unavailable_reason == "metadata_mismatch"
    assert row.mde_unavailable_reason == "not_requested"
    assert row.model_dump(mode="json")["power"] is None


def test_conversion_bounds_do_not_authorize_additive_shift_variance():
    with pytest.raises(InvalidRequestError) as error:
        baseline(
            aggregation="any",
            estimand="retained_window_conversion_difference",
            sd_a=0.1,
            effect_bounds=(-0.2, 0.8),
        )
    assert error.value.code == "power.switchback.bounded_model_required"
    assert error.value.context["field"] == "aggregation"
    assert error.value.context["value"] == "any"
    assert error.value.context["constraint"] == "constant-additive model requires aggregation='sum'"
    assert error.value.context["route"]


def _admission_probability(n, p_ct):
    return 1.0 - p_ct**n - (1.0 - p_ct) ** n


def test_independent_order_baseline_admission_probability_is_always_one():
    b, p = baseline(), procedure()
    row = achieved(4, 2.0, b, p)
    assert row.admission_probability == 1.0
    assert row.admission_probability_unavailable_reason is None
    sizing = required(0.5, b, p, target_power=0.8)
    assert sizing.admission_probability == 1.0


def _shared_baseline(probability_ct, **changes):
    return baseline(
        assignment=SwitchbackAssignment(
            sequence=SharedScheduleOrder(probability_ct=probability_ct),
            window=SwitchbackWindow(washout_steps=1, observation_steps=2),
        ),
        shared_roster=("u0", "u1", "u2", "u3", "u4"),
        cycles_per_unit=None,
        **changes,
    )


def test_shared_schedule_achieved_power_is_scaled_by_admission_probability():
    b = _shared_baseline(0.8, sd_a=10.0, sd_g=0.0, rho=0.0)
    p = procedure()
    n, delta = 8, 20.0
    row = achieved(n, delta, b, p)
    expected_admission = _admission_probability(n, 0.8)
    assert row.admission_probability == pytest.approx(expected_admission)
    # power_given_admissible recovered by dividing the reported (already
    # scaled) power back out; must be a valid probability on its own.
    power_given_admissible = row.power / expected_admission
    assert 0.0 <= power_given_admissible <= 1.0
    assert row.power == pytest.approx(expected_admission * power_given_admissible)


def test_shared_schedule_required_blocks_rises_under_a_skewed_probability_ct():
    """A design sized without accounting for admission risk under-supplies
    power. Measured on this exact fixture (5-unit shared roster,
    probability_ct=0.8, sd_a=10.0, delta=20.0, target_power=0.8): the
    admission-unaware quantity (power_given_admissible alone) first clears
    0.8 at n=5, but P(admissible) at n=5 is only 0.8**5+0.2**5's
    complement (0.6723), so available power there is well under target.
    The corrected planner must return a larger n."""
    b = _shared_baseline(0.8, sd_a=10.0, sd_g=0.0, rho=0.0)
    p = procedure()
    delta = 20.0
    sizing = required(delta, b, p, target_power=0.8)
    assert sizing.n is not None
    assert sizing.n == 8
    assert sizing.admission_probability == pytest.approx(_admission_probability(8, 0.8))
    # The naive (admission-unaware) n=5 must now fail to reach the target:
    # available power at n=5 is admission_probability(5)*power_given_admissible(5).
    naive = achieved(5, delta, b, p)
    assert naive.power is not None
    assert naive.power < 0.8


def test_shared_schedule_mde_is_unattained_when_admission_probability_at_or_below_target():
    """At n=5, probability_ct=0.8: admission_probability(5, 0.8) = 1 -
    0.8**5 - 0.2**5 = 0.6723, at or below target_power=0.8. No finite
    delta can supply power_given_admissible >= target_power /
    admission_probability (> 1.0 here) -- 'unattained' is correct and is
    the answer that tells the user what to do (declare more blocks, or a
    probability_ct closer to 0.5), unlike 'numerical_resolution', which
    the unmodified solver returns instead: with sd_g=0 (this fixture),
    a=0, so the `a > 0` asymptote-check guard never runs, and
    _required_nc(5, procedure(), target_power/admission_probability) --
    a target above 1.0 -- returns None."""
    b = _shared_baseline(0.8, sd_a=10.0, sd_g=0.0, rho=0.0)
    p = procedure()
    row = mde(5, b, p, target_power=0.8)
    assert row.mde_abs is None
    assert row.mde_unavailable_reason == "unattained"


def test_shared_schedule_mde_and_required_blocks_round_trip_under_admission_scaling():
    b = _shared_baseline(0.8, sd_a=10.0, sd_g=0.0, rho=0.0)
    p = procedure()
    row = mde(8, b, p, target_power=0.8)
    assert_boundary(row, b, p, 0.8)
    assert row.mde_abs is not None
    count = required(row.mde_abs, b, p, target_power=0.8)
    assert count.n == 8


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_shared_schedule_power_available_matches_bounded_simulation():
    """The analytic power_available = admission_probability *
    power_given_admissible must match a direct Monte Carlo simulation of
    the same two-stage process (admission draw, then the noncentral-t
    decision), not just be internally self-consistent."""
    from scipy.stats import nct as sp_nct
    from scipy.stats import t as sp_t

    from increment.power.switchback import _nc, _variance

    b = _shared_baseline(0.8, sd_a=10.0, sd_g=0.0, rho=0.0)
    p = procedure()
    n, delta, p_ct = 8, 20.0, 0.8
    row = achieved(n, delta, b, p)
    assert row.power is not None

    variance = _variance(Fraction(delta), b)
    nc = _nc(Fraction(delta) - Fraction(p.null_abs), variance, n)
    dof = n - 1
    crit = sp_t.isf(p.alpha / 2.0, dof)

    rng = np.random.default_rng(7)
    reps = 200_000
    draws = rng.binomial(n, p_ct, size=reps)
    degenerate = (draws == 0) | (draws == n)
    t_samples = sp_nct.rvs(dof, nc, size=reps, random_state=rng)
    rejected = np.abs(t_samples) > crit
    mc_power_available = float((~degenerate & rejected).mean())
    mcse = math.sqrt(mc_power_available * (1 - mc_power_available) / reps)
    assert abs(mc_power_available - row.power) <= 4 * mcse


def test_shared_roster_counts_blocks_and_copies_input():
    roster = ["a", "b", "c"]
    b = baseline(
        assignment=SwitchbackAssignment(
            sequence=SharedScheduleOrder(probability_ct=0.75),
            window=SwitchbackWindow(washout_steps=2, observation_steps=4, carryover_order=1),
        ),
        cycles_per_unit=None,
        shared_roster=roster,
    )
    roster.append("d")
    assert b.shared_roster == ("a", "b", "c")
    row = achieved(4, 0.5, b, procedure())
    assert row.n == 4 and row.df == 3 and row.planning_grain == "shared_block"
    # The noncentral-t reference itself is identical between the shared and
    # independent laws at the same n (sd_a/sd_g/rho/delta_ref are shared,
    # unmodified defaults); only the admission-adjusted reported power
    # differs, by exactly this design's P(both cycle orders occur).
    independent_row = achieved(4, 0.5, baseline(), procedure())
    assert row.admission_probability == pytest.approx(1.0 - 0.75**4 - 0.25**4)
    assert independent_row.admission_probability == 1.0
    assert row.power is not None and independent_row.power is not None
    assert row.admission_probability is not None
    assert row.power == pytest.approx(row.admission_probability * independent_row.power)
    # Sizing a shared design under a skewed probability_ct now needs at
    # least as many blocks as the admission-unaware independent design.
    shared_required = required(0.5, b, procedure(), target_power=0.8)
    independent_required = required(0.5, baseline(), procedure(), target_power=0.8)
    assert shared_required.n is not None and independent_required.n is not None
    assert shared_required.n >= independent_required.n
    for bad in (("a", "a"), ("",), ()):
        with pytest.raises(InvalidRequestError) as exc:
            SwitchbackBaseline.model_validate(dict(b.model_dump(), shared_roster=bad))
        assert exc.value.code == "power.switchback.roster"


@pytest.mark.parametrize(
    "changes",
    [
        {"sd_a": 0.0, "rho": 0.5},
        {"control_group": "treatment"},
        {"shared_roster": ("a",)},
        {"cycles_per_unit": None},
    ],
)
def test_invalid_baseline_combinations_are_coded(changes):
    with pytest.raises(InvalidRequestError):
        baseline(**changes)


@pytest.fixture(params=["achieved", "mde", "required"])
def solver(request):
    def solve(b, p):
        if request.param == "achieved":
            return achieved(20, 0.3, b, p)
        if request.param == "mde":
            return mde(20, b, p, target_power=0.8)
        return required(0.3, b, p, target_power=0.8)

    return solve


def unchecked_change(model, path, value, mode):
    field, *rest = path.split(".")
    if rest:
        value = unchecked_change(getattr(model, field), ".".join(rest), value, mode)
    if mode == "construct":
        return type(model).model_construct(**(dict(model) | {field: value}))
    object.__setattr__(model, field, value)
    return model


@pytest.mark.parametrize("mode", ["construct", "mutate"])
@pytest.mark.parametrize(
    "input_name,path,value,code",
    [
        ("baseline", "sd_a", -1.0, "model.field.range"),
        ("baseline", "sd_g", math.nan, "model.field.nonfinite"),
        ("baseline", "rho", 2.0, "model.field.range"),
        ("baseline", "cycles_per_unit", None, "power.switchback.baseline"),
        ("baseline", "effect_bounds", (1.0,), "model.field.missing"),
        ("baseline", "aggregation", "any", "power.switchback.bounded_model_required"),
        ("baseline", "assignment.periods_per_cycle", 3, "power.switchback.baseline"),
        ("baseline", "assignment.sequence.probability_ct", 0.0, "power.switchback.baseline"),
        ("baseline", "assignment.window.washout_steps", -1, "power.switchback.baseline"),
        (
            "baseline",
            "assignment.window.carryover_order",
            1,
            "definition.assignment.carryover_order_range",
        ),
        ("procedure", "alpha", 0.0, "model.field.range"),
        ("procedure", "alpha", 0.5, "decision.arm_decision.one_sided_alpha"),
        ("procedure", "null_abs", math.inf, "model.field.nonfinite"),
        ("procedure", "alternative", "invalid", "model.field.literal"),
        ("procedure", "family.kind", "bonferroni", "model.field.literal"),
        ("procedure", "inference.kind", "sequential", "model.field.literal"),
    ],
)
def test_solvers_revalidate_before_calculation(solver, mode, input_name, path, value, code):
    inputs = {
        "baseline": baseline(),
        "procedure": procedure(alternative="greater").model_copy(deep=True),
    }
    inputs[input_name] = unchecked_change(inputs[input_name], path, value, mode)
    with pytest.raises(InvalidRequestError) as exc:
        solver(inputs["baseline"], inputs["procedure"])
    assert exc.value.code == code
    with pytest.raises(TypeError):
        exc.value.context["changed"] = True  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("mapping", [MappingProxyType, UserDict])
@pytest.mark.parametrize(
    "nested,field,value", [("sequence", "probability_ct", 0.0), ("window", "washout_steps", -1)]
)
def test_solvers_revalidate_nested_models_in_mapping(solver, mapping, nested, field, value):
    b = baseline()
    assignment = dict(b.assignment)
    assignment[nested] = assignment[nested].model_copy(update={field: value})
    forged = b.model_copy(update={"assignment": mapping(assignment)})

    with pytest.raises(InvalidRequestError) as exc:
        solver(forged, procedure())
    assert exc.value.code == "power.switchback.baseline"
    with pytest.raises(TypeError):
        exc.value.context["changed"] = True  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("mapping", [MappingProxyType, UserDict])
def test_solvers_accept_valid_nested_models_in_mapping(solver, mapping):
    b, p = baseline(), procedure()
    assignment = mapping(dict(b.assignment))
    forged = b.model_copy(update={"assignment": assignment})

    result = solver(forged, p)

    assert result == solver(b, p)
    assert result.baseline.assignment.sequence is not assignment["sequence"]
    assert result.baseline.assignment.window is not assignment["window"]
    assert forged.assignment is assignment


@pytest.mark.parametrize("input_name", ["baseline", "procedure"])
@pytest.mark.parametrize("value", [None, {}, "invalid"])
def test_solvers_reject_wrong_input_types(solver, input_name, value):
    inputs = {"baseline": baseline(), "procedure": procedure()}
    inputs[input_name] = value
    with pytest.raises(InvalidRequestError) as exc:
        solver(inputs["baseline"], inputs["procedure"])
    assert exc.value.code == f"power.switchback.{input_name}"


def test_solvers_consume_validated_copies(solver):
    b, p = baseline(), procedure()
    expected = solver(b, p)
    object.__setattr__(b, "sd_a", "1.0")
    object.__setattr__(b.assignment.sequence, "probability_ct", "0.75")
    object.__setattr__(p, "null_abs", "0.0")
    object.__setattr__(p, "alpha", "0.05")
    result = solver(b, p)
    assert result == expected
    assert result.baseline is not b
    assert result.baseline.assignment is not b.assignment
    assert result.baseline.assignment.sequence is not b.assignment.sequence
    assert result.procedure is not p
    assert b.sd_a == "1.0"
    assert p.alpha == "0.05"


def test_models_frozen_serializable_and_public_exports():
    import increment

    b = baseline()
    row = achieved(4, 1.0, b, procedure())
    assert row.power_kind == "moment_t_approximation"
    for obj, field, value in ((b, "sd_a", 2.0), (row, "power", 0.1)):
        with pytest.raises((TypeError, ValueError)):
            setattr(obj, field, value)
    assert SwitchbackPowerResult.model_validate_json(row.model_dump_json()) == row
    assert increment.SwitchbackBaseline is SwitchbackBaseline
    assert increment.switchback_achieved_power is achieved
    with pytest.raises(InvalidRequestError):
        SwitchbackPowerResult.model_validate(dict(row.model_dump(), mde_unavailable_reason=None))


def test_first_feasible_float_can_be_on_descending_side_of_excluded_peak():
    b, p = baseline(sd_a=math.ulp(0.0), sd_g=1.0, rho=-1.0), procedure()
    result = mde(4, b, p, target_power=0.1)
    assert result.mde_abs == 2 * math.ulp(0.0)
    assert_boundary(result, b, p, 0.1)


def test_bounded_domain_can_make_finite_target_unattained():
    b = baseline(effect_bounds=(-0.1, 0.1))
    assert mde(4, b, procedure(), target_power=0.99).mde_unavailable_reason == "unattained"


def test_two_sided_decrease_mde_searches_the_favorable_domain():
    baseline_decrease = baseline(effect_bounds=(-2.0, 0.0))
    decrease = procedure(preferred_direction="decrease")
    result = mde(100, baseline_decrease, decrease, target_power=0.8)
    assert result.mde_abs is not None
    assert result.mde_abs < 0
    attained = achieved(100, result.mde_abs, baseline_decrease, decrease)
    assert attained.power is not None and attained.power >= 0.8
    smaller_magnitude = achieved(
        100, math.nextafter(result.mde_abs, math.inf), baseline_decrease, decrease
    )
    assert smaller_magnitude.power is not None and smaller_magnitude.power < 0.8
    assert achieved(100, -0.5, baseline_decrease, decrease).power == pytest.approx(
        0.9986097259857082
    )

    baseline_increase = baseline(effect_bounds=(0.0, 2.0))
    increase = procedure(preferred_direction="increase")
    mirrored = mde(100, baseline_increase, increase, target_power=0.8)
    assert mirrored.mde_abs is not None and mirrored.mde_abs > 0
    mirrored_power = achieved(100, mirrored.mde_abs, baseline_increase, increase).power
    assert mirrored_power is not None and mirrored_power >= 0.8
    previous_power = achieved(
        100, math.nextafter(mirrored.mde_abs, -math.inf), baseline_increase, increase
    ).power
    assert previous_power is not None and previous_power < 0.8


def test_two_sided_decrease_mde_reports_genuine_domain_exhaustion():
    result = mde(
        100,
        baseline(effect_bounds=(-0.01, 0.0)),
        procedure(preferred_direction="decrease"),
        target_power=0.8,
    )
    assert result.mde_abs is None
    assert result.mde_unavailable_reason == "unattained"


def test_null_outside_domain_selects_first_included_favorable_endpoint():
    b, p = baseline(effect_bounds=(1.0, 2.0), delta_ref=1.0, sd_a=0.01), procedure()
    result = mde(4, b, p, target_power=0.8)
    assert result.mde_abs == 1.0
    assert result.power is not None
    assert result.power >= 0.8


def test_singular_exact_zero_does_not_confuse_positive_underflow():
    b, p = baseline(sd_a=math.ulp(0.0), sd_g=1.0, rho=math.nextafter(-1.0, 0.0)), procedure()
    result = achieved(4, math.ulp(0.0), b, p)
    assert result.power_unavailable_reason != "degenerate_zero_variance"
    assert result.standard_error_unavailable_reason == "unrepresentable"
    assert _variance(Fraction(math.ulp(0.0)), b) > 0


def test_near_singular_peak_backend_failure_does_not_hide_ascending_root():
    b, p = baseline(sd_a=2.0, sd_g=1.0, rho=math.nextafter(-1.0, 0.0)), procedure()
    result = mde(4, b, p, target_power=0.8)
    assert result.mde_abs is not None
    assert result.mde_abs < 2.0
    assert_boundary(result, b, p, 0.8)


@pytest.mark.parametrize("alpha", [1e-50, math.ulp(0.0), math.nextafter(1.0, 0.0)])
def test_extreme_alpha_inverse_queries_have_explicit_availability(alpha):
    b, p = baseline(), procedure(alpha=alpha)
    for result in (mde(4, b, p, target_power=0.8), required(0.5, b, p, target_power=0.8)):
        if result.power is None:
            assert result.power_unavailable_reason == "numerical_resolution"
        else:
            assert result.power >= 0.8


def test_domain_start_at_excluded_singular_peak_can_use_descending_side():
    step = math.ulp(0.0)
    b = baseline(sd_a=0.0, sd_g=1.0, delta_ref=step, effect_bounds=(step, 4 * step))
    p = procedure()
    result = mde(4, b, p, target_power=0.1)
    assert result.mde_abs == 2 * step
    assert_boundary(result, b, p, 0.1)


def test_bounded_asymptote_equality_is_not_rounded_into_attainment():
    from increment.power.switchback import _favorable_power, _nc

    epsilon = 2.0**-52
    b = baseline(sd_g=1.0 - epsilon, rho=1.0, effect_bounds=(0.0, 2.0))
    p = procedure(null_abs=-(1.0 + epsilon))
    asymptote = _favorable_power(_nc(Fraction(1), Fraction(b.sd_g) ** 2, 4), 4, p)
    assert asymptote is not None
    assert achieved(4, 0.0, b, p).power == asymptote
    assert mde(4, b, p, target_power=asymptote).mde_unavailable_reason == "unattained"


def _pilot_source(*, shared, p, cycles, observation_steps, carryover_order, shift=0.0):
    import numpy as np
    import polars as pl

    from increment.frame import from_switchback_panel
    from increment.semantics.design import Randomized

    rng = np.random.default_rng(77411)
    n_units = 12
    orders = rng.random((1 if shared else n_units, cycles)) < p
    retained = observation_steps - carryover_order
    rows = []
    for unit in range(n_units):
        unit_trend = rng.normal()
        for cycle in range(cycles):
            ct = orders[0 if shared else unit, cycle]
            for period in range(2):
                treated = (period == 1) == ct
                for step in range(observation_steps + 1):
                    value = 40 + period * (0.7 + unit_trend) + rng.normal()
                    if step >= 1 + carryover_order and treated:
                        value += (1.5 + shift) / retained
                    rows.append(
                        {
                            "unit": str(unit),
                            "cycle": cycle,
                            "period": period,
                            "step": step,
                            "group": "treatment" if treated else "control",
                            "orders": value,
                        }
                    )
    sequence = SharedScheduleOrder if shared else IndependentBernoulliOrder
    return from_switchback_panel(
        pl.DataFrame(rows),
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="group",
        metrics={"orders": "mean"},
        identification=Randomized(
            control_group="control",
            allocation={"control": 0.5, "treatment": 0.5},
        ),
        assignment=SwitchbackAssignment(
            sequence=sequence(probability_ct=p),
            window=SwitchbackWindow(
                washout_steps=1,
                observation_steps=observation_steps,
                carryover_order=carryover_order,
            ),
        ),
    )


@pytest.mark.parametrize(
    "case",
    [
        {"shared": True, "p": 0.75, "cycles": 20, "observation_steps": 4, "carryover_order": 2},
        {"shared": False, "p": 0.75, "cycles": 5, "observation_steps": 4, "carryover_order": 1},
        {"shared": False, "p": 0.5, "cycles": 20, "observation_steps": 1, "carryover_order": 0},
        {"shared": False, "p": 0.75, "cycles": 1, "observation_steps": 1, "carryover_order": 0},
    ],
)
def test_pilot_covariance_predicts_actual_shifted_panel_uncertainty(case):
    from increment.estimation.contrast import estimate_contrast

    source = _pilot_source(**case)
    decision = source.context.procedures["orders"]
    metric = source.metrics[0]
    b = source.planning_baseline(metric)
    stats = source.contrast_stats(metric)
    n = stats.n_blocks if case["shared"] else stats.n_units
    reference = estimate_contrast(stats, decision).results[0]
    assert b.delta_ref == pytest.approx(reference.estimate.value)
    assert achieved(n, b.delta_ref, b, decision).standard_error == pytest.approx(
        reference.standard_error,
        rel=1e-13,
    )
    for shift in (-2.0, 3.0):
        shifted = _pilot_source(**case, shift=shift)
        actual = estimate_contrast(
            shifted.contrast_stats(shifted.metrics[0]),
            decision,
        ).results[0]
        predicted = achieved(n, b.delta_ref + shift, b, decision)
        assert predicted.standard_error == pytest.approx(actual.standard_error, rel=1e-12)
    sizing = mde(n, b, decision, target_power=0.8)
    assert sizing.mde_abs is not None
    assert_boundary(sizing, b, decision, 0.8)
    count = required(sizing.mde_abs, b, decision, target_power=0.8)
    assert count.n == n


def test_single_shared_block_now_refuses_at_construction_not_only_at_planning():
    """A single shared block can never realize both cycle orders (one draw,
    one order), so the positivity fix now refuses it at construction --
    earlier and more precisely than the old ``planning_requires_replicates``
    refusal, which only fired later, inside ``planning_baseline``, and only
    for the subset of degenerate single-block panels that happened to reach
    it uncaught."""
    with pytest.raises(InvalidRequestError) as error:
        _pilot_source(
            shared=True,
            p=0.5,
            cycles=1,
            observation_steps=1,
            carryover_order=0,
        )
    assert error.value.code == "source.frame.switchback.schedule"
    assert error.value.context["reason"] == "shared_schedule_missing_cycle_order"
    assert error.value.context["n_blocks"] == 1
