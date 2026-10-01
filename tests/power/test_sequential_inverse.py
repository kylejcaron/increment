"""The certified sequential minimum-detectable-effect inverse.

Every answer is a public float whose own certified power enclosure reaches
the target, with the alternative's own variance behind both the drift and
(for GaussianScoreMixture) the boundary. Ordinary inversions are pinned to the
values the independent Simpson reference validated; refusals carry the
unresolved interval and enclosure the caller needs to act on them.
"""

from __future__ import annotations

import math

import pytest

from increment.errors import InvalidRequestError
from increment.estimation.sequential import GaussianScoreMixture
from increment.power import (
    Baseline,
    PowerDesign,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)
from increment.power import sequential as sequential_module
from increment.power._search import _se_from_log
from increment.power.sequential import (
    MAX_EVALUATIONS,
    RESOLUTION_MULTIPLE,
    _CrossingEnclosure,
    planning_bounds,
)

from ._procedures import make_procedure
from ._references import simpson_walk
from ._results import available_mde
from ._sequential import assert_sequential_solved, sequential_search, sequential_solve

BASE = Baseline(mean=1.0, var=2.0)
UNIT = Baseline(mean=1.0, var=1.0)
SHIFTED = Baseline(mean=1.0, var=1.0, compliance=0.1)
N_FIXED = 12576  # required_sample_size(make_procedure(), 0.05, BASE).n_per_arm
NUMERICAL_RESOLUTION_CONTEXT = {
    "target_power",
    "direction",
    "unresolved_interval",
    "power_enclosure",
    "stopping_reason",
}


def _reference_power(n_per_arm, baseline, procedure, design, looks, m, points=4001) -> float:
    """Simpson-rule power at candidate ``m`` with its own variance behind
    the drift and the boundary."""
    search = sequential_search(n_per_arm, baseline, procedure, design, looks)
    point = search.search.candidate(m)
    assert point is not None
    distance, theta = point
    se_full = _se_from_log(search.search.plan.log_se_sq(theta))
    fractions = [k / looks for k in range(1, looks + 1)]
    bounds = planning_bounds(
        search.inference, fractions, se_full, search.alpha_seq, search.exit_side
    )
    walk = simpson_walk(bounds, fractions, abs(distance) / se_full, points)
    return walk.upper if search.exit_side == "upper" else walk.both


def _assert_certified_answer(solved, target):
    enclosure = solved.enclosure
    assert enclosure.resolved
    assert enclosure.converged
    assert enclosure.expected_converged
    assert enclosure.lower >= target
    assert enclosure.estimate == solved.power
    assert enclosure.estimate - target <= RESOLUTION_MULTIPLE * enclosure.half_width
    # Refined until the quadrature term no longer dominates: the declared
    # resolution is the rounding budget, proportional to the estimate.
    assert enclosure.quadrature <= enclosure.rounding + enclosure.sf_rel
    assert enclosure.half_width <= 2.0 * (enclosure.rounding + enclosure.sf_rel)
    assert enclosure.half_width <= 1e-10 * enclosure.estimate
    a_excl, b = solved.bracket
    assert b == solved.mde_relative
    assert abs(a_excl) <= abs(b)
    # A documented gap lies inside the bracket, straddles the target, and is
    # resolved to the evaluator's own accuracy.
    for gap in solved.straddle_gaps:
        assert abs(a_excl) <= abs(gap.start) <= abs(gap.end) <= abs(b)
        assert gap.lower <= target <= gap.upper
        assert gap.upper - gap.lower <= RESOLUTION_MULTIPLE * gap.half


def test_fixed_horizon_anchor_is_unchanged():
    assert (
        required_sample_size(
            procedure=make_procedure(), relative_lift=0.05, baseline=BASE
        ).n_per_arm
        == N_FIXED
    )


class TestOrdinaryInversions:
    """Two-sided inversions at the fixed-horizon n for a 5% lift, pinned to
    the candidates whose Simpson reference power was verified inside the
    certified enclosure."""

    CASES = [
        pytest.param("av05", 5, 0.06759378861419829, id="av05-k5"),
        pytest.param("av05", 14, 0.06634631727956194, id="av05-k14"),
    ]
    SPECS = {
        "av05": GaussianScoreMixture(),
    }

    @pytest.mark.slow
    @pytest.mark.parametrize(("label", "looks", "pinned"), CASES)
    def test_certified_answer_matches_the_reference(self, label, looks, pinned):
        procedure = make_procedure(inference=self.SPECS[label], population="assigned")
        design = PowerDesign()
        public = minimum_detectable_effect(
            procedure=procedure,
            n_per_arm=N_FIXED,
            baseline=BASE,
            design=design,
            planned_looks=looks,
        )
        assert available_mde(public) == pytest.approx(pinned, rel=1e-9)
        solved = assert_sequential_solved(sequential_solve(N_FIXED, BASE, procedure, design, looks))
        assert solved.mde_relative == available_mde(public)
        assert solved.power == public.power
        _assert_certified_answer(solved, 0.8)
        a_excl, b = solved.bracket
        assert 0.0 < (b - a_excl) / b < 1e-8
        reference = _reference_power(N_FIXED, BASE, procedure, design, looks, solved.mde_relative)
        assert solved.enclosure.lower <= reference <= solved.enclosure.upper
        assert reference >= 0.8


@pytest.mark.slow
class TestDirectionalSigns:
    def test_greater_and_less_have_the_sign_of_their_direction(self):
        greater = minimum_detectable_effect(
            procedure=make_procedure(
                inference=GaussianScoreMixture(), alternative="greater", population="assigned"
            ),
            n_per_arm=N_FIXED,
            baseline=BASE,
            planned_looks=5,
        )
        less = minimum_detectable_effect(
            procedure=make_procedure(
                inference=GaussianScoreMixture(), alternative="less", population="assigned"
            ),
            n_per_arm=N_FIXED,
            baseline=BASE,
            planned_looks=5,
        )
        assert available_mde(greater) == pytest.approx(0.062419610296496536, rel=1e-9)
        assert available_mde(less) == pytest.approx(-0.062428507282561434, rel=1e-9)
        assert greater.power >= 0.8 and less.power >= 0.8


class TestShiftedNullEndpoint:
    """``Baseline(compliance=0.1)`` against ``null_lift=-0.5``: the increasing
    direction opens at the complier lift ``8.0`` (effective lift ``-0.2``)
    and the decreasing direction is empty."""

    def _procedure(self, spec, alternative="greater"):
        return make_procedure(
            inference=spec, alternative=alternative, null_lift=-0.5, population="assigned"
        )

    @pytest.mark.slow
    def test_always_valid_solves_inside_the_direction(self):
        # n=50 keeps this a genuine interior solve.
        design = PowerDesign()
        solved = assert_sequential_solved(
            sequential_solve(50, SHIFTED, self._procedure(GaussianScoreMixture()), design, 5)
        )
        assert solved.mde_relative == pytest.approx(10.090618739050115, rel=1e-9)
        assert 10.09061873902101 <= solved.mde_relative <= 10.090618739050115
        _assert_certified_answer(solved, 0.8)
        reference = _reference_power(
            50,
            SHIFTED,
            self._procedure(GaussianScoreMixture()),
            design,
            5,
            solved.mde_relative,
        )
        assert solved.enclosure.lower <= reference <= solved.enclosure.upper

    @pytest.mark.parametrize("spec", [GaussianScoreMixture()])
    def test_high_target_solves(self, spec):
        result = minimum_detectable_effect(
            procedure=self._procedure(spec),
            n_per_arm=100,
            baseline=SHIFTED,
            design=PowerDesign(power=0.999),
            planned_looks=5,
        )
        assert available_mde(result) > 8.0
        assert result.power >= 0.999

    def test_empty_decreasing_direction_is_unattainable_not_unrepresentable(self):
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                procedure=self._procedure(GaussianScoreMixture(), alternative="less"),
                n_per_arm=100,
                baseline=SHIFTED,
                planned_looks=5,
            )
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "empty_admissible_direction"
        assert refusal.value.context["maximum_power"] is None


@pytest.mark.slow
class TestDecreasesNearThePeak:
    """A decrease's noncentrality peaks; targets below the sequential maximum
    solve, targets above it are certified unattainable, and the band of
    targets the enclosure cannot separate from the maximum is refused."""

    AV_LESS = make_procedure(
        inference=GaussianScoreMixture(), alternative="less", population="assigned"
    )

    def test_always_valid_solves_below_and_refuses_above_the_maximum(self):
        solved = assert_sequential_solved(
            sequential_solve(50, UNIT, self.AV_LESS, PowerDesign(power=0.30549514412021417), 5)
        )
        assert solved.mde_relative == pytest.approx(-0.4678246512821715, rel=1e-9)
        _assert_certified_answer(solved, 0.30549514412021417)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                n_per_arm=50,
                baseline=UNIT,
                procedure=self.AV_LESS,
                design=PowerDesign(power=0.44),
                planned_looks=5,
            )
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "sequential_exclusion"
        maximum_power = refusal.value.context["maximum_power"]
        assert isinstance(maximum_power, float)
        assert 0.43 < maximum_power < 0.44


@pytest.mark.slow
class TestTinyAlpha:
    def test_always_valid_at_alpha_1e_300(self):
        procedure = make_procedure(
            inference=GaussianScoreMixture(), alpha=1e-300, population="assigned"
        )
        solved = assert_sequential_solved(sequential_solve(5000, BASE, procedure, PowerDesign(), 5))
        assert solved.mde_relative == pytest.approx(1.2968871301369844, rel=1e-9)
        _assert_certified_answer(solved, 0.8)
        coarse = _reference_power(
            5000, BASE, procedure, PowerDesign(), 5, solved.mde_relative, 4001
        )
        reference = _reference_power(
            5000, BASE, procedure, PowerDesign(), 5, solved.mde_relative, 8001
        )
        # This extreme-tail Simpson reference converges non-monotonically at
        # float64 precision. Keep its observed refinement error separate from
        # the production enclosure rather than treating it as exact.
        reference_error = 2.0 * abs(reference - coarse)
        assert (
            solved.enclosure.lower - reference_error
            <= reference
            <= solved.enclosure.upper + reference_error
        )


class TestBoundedMetric:
    CONVERSION_AV = make_procedure(
        inference=GaussianScoreMixture(),
        metric_type="conversion",
        alternative="greater",
        population="assigned",
    )

    def test_high_baseline_rate_is_unattainable_under_the_rate_ceiling(self):
        baseline = Baseline(mean=0.98, var=0.0196)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                procedure=self.CONVERSION_AV, n_per_arm=200, baseline=baseline, planned_looks=5
            )
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "sequential_exclusion"
        companion = achieved_power(200, 0.01, baseline, self.CONVERSION_AV, planned_looks=5)
        assert companion.mde_relative is None
        assert companion.mde_unavailable_reason == "unattainable"

    @pytest.mark.slow
    def test_high_baseline_rate_solves_when_reachable(self):
        baseline = Baseline(mean=0.9, var=0.09)
        result = minimum_detectable_effect(
            procedure=self.CONVERSION_AV, n_per_arm=2000, baseline=baseline, planned_looks=5
        )
        assert available_mde(result) == pytest.approx(0.03423847596711005, rel=1e-9)
        assert 0.9 * (1.0 + available_mde(result)) <= 1.0


@pytest.mark.slow
class TestNullTargets:
    """Targets at or near the boundary's own null-crossing probability."""

    PROCEDURE = make_procedure(inference=GaussianScoreMixture(), population="assigned")

    def _null(self):
        search = sequential_search(N_FIXED, BASE, self.PROCEDURE, PowerDesign(), 14)
        se_full = search.se_full(search.search.theta0)
        return sequential_module.sequential_power_enclosure(
            search.inference,
            0.0,
            se_full,
            search.alpha_seq,
            14,
            search.exit_side,
        )

    def test_below_the_null_power_raises_the_null_crossing_error(self):
        null = self._null()
        with pytest.raises(InvalidRequestError) as raised:
            sequential_solve(
                N_FIXED, BASE, self.PROCEDURE, PowerDesign(power=0.5 * null.estimate), 14
            )
        assert raised.value.code == "power.sequential_mde.design_search_minimum"
        assert raised.value.context["estimate"] == pytest.approx(null.estimate)

    def test_target_inside_the_null_enclosure_is_unresolved(self):
        null = self._null()
        target = 0.5 * (null.lower + null.upper)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                n_per_arm=N_FIXED,
                baseline=BASE,
                procedure=self.PROCEDURE,
                design=PowerDesign(power=target),
                planned_looks=14,
            )
        assert refusal.value.code == "power.minimum_detectable_effect.numerical_resolution"
        assert refusal.value.context["unresolved_interval"] == (0.0, 0.0)
        assert refusal.value.context["power_enclosure"] == pytest.approx(
            (null.lower, null.upper), abs=1e-15
        )

    def test_just_above_the_null_power_solves(self):
        null = self._null()
        target = null.estimate + 1e-9
        solved = assert_sequential_solved(
            sequential_solve(N_FIXED, BASE, self.PROCEDURE, PowerDesign(power=target), 14)
        )
        assert 0.0 < solved.mde_relative < 1e-5
        assert solved.mde_relative == pytest.approx(3.485707566141801e-06, rel=1e-3)
        assert solved.enclosure.lower >= target


class TestExpectedInformationAtTheSolvedEffect:
    @pytest.mark.slow
    def test_expected_n_total_is_evaluated_at_the_answer(self):
        procedure = make_procedure(inference=GaussianScoreMixture(), population="assigned")
        result = minimum_detectable_effect(
            procedure=procedure, n_per_arm=N_FIXED, baseline=BASE, planned_looks=5
        )
        # The null-variance solve reported 17978 here; the answer's own
        # variance stops earlier on average.
        assert result.expected_n_total != 17978
        at_answer = achieved_power(N_FIXED, available_mde(result), BASE, procedure, planned_looks=5)
        assert at_answer.expected_n_total == result.expected_n_total
        assert at_answer.power == pytest.approx(result.power, abs=1e-12)


def _controller_enclosure(
    estimate: float,
    *,
    lower: float,
    upper: float,
    half_width: float,
    expected_converged: bool,
    nodes: int = 16,
) -> _CrossingEnclosure:
    return _CrossingEnclosure(
        estimate=estimate,
        lower=lower,
        upper=upper,
        quadrature=half_width,
        rounding=0.0,
        sf_rel=0.0,
        slope=0.0,
        slope_half=0.0,
        nodes=nodes,
        resolved=True,
        converged=True,
        expected_converged=expected_converged,
        expected_fraction=0.75,
        expected_half=0.0 if expected_converged else 1.0,
        levels=(estimate, estimate, estimate),
    )


class TestExpectedInformationRefinement:
    def test_preserves_stronger_power_bounds(self, monkeypatch):
        search = sequential_search(
            N_FIXED,
            BASE,
            make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            PowerDesign(),
            5,
        )
        original = _controller_enclosure(
            0.805,
            lower=0.8,
            upper=0.81,
            half_width=0.005,
            expected_converged=False,
        )
        refined = _controller_enclosure(
            0.795,
            lower=0.79,
            upper=0.8,
            half_width=0.005,
            expected_converged=True,
            nodes=32,
        )
        monkeypatch.setattr(type(search), "point", lambda self, m, nodes=None: refined)

        certified = search.certify_expected(0.1, original, (0.0, 0.1))

        # A refinement that refused instead of resolving carries a reason.
        assert not hasattr(certified, "reason")

        assert certified.lower == 0.8
        assert certified.upper == 0.8
        # The finer estimate (0.795) lay outside the older, tighter bound; the
        # certified power must sit inside the enclosure it is reported with.
        assert certified.lower <= certified.estimate <= certified.upper
        assert certified.estimate == 0.8

    def test_rechecks_terminal_resolution_after_expected_refinement(self, monkeypatch):
        search = sequential_search(
            N_FIXED,
            BASE,
            make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            PowerDesign(),
            5,
        )
        initial = _controller_enclosure(
            0.800000000003,
            lower=0.800000000002,
            upper=0.800000000004,
            half_width=1e-12,
            expected_converged=False,
        )
        refined = _controller_enclosure(
            0.8000000000039,
            lower=0.8000000000038,
            upper=0.800000000004,
            half_width=1e-13,
            expected_converged=True,
            nodes=32,
        )
        monkeypatch.setattr(type(search), "certify", lambda self, m, enclosure, interval: initial)
        monkeypatch.setattr(
            type(search), "certify_expected", lambda self, m, enclosure, interval: refined
        )

        outcome = search.assess_candidate(
            a_excl=0.0,
            b=0.1,
            enclosure=initial,
            adjacent=False,
            gaps=[],
        )

        assert outcome is True


@pytest.mark.slow
class TestEvaluationBudget:
    """The most expensive default fourteen-look solves stay inside the
    explicit certified-evaluation ceiling."""

    def test_shifted_null_high_target(self):
        search = sequential_search(
            100,
            SHIFTED,
            make_procedure(
                inference=GaussianScoreMixture(),
                alternative="greater",
                null_lift=-0.5,
                population="assigned",
            ),
            PowerDesign(power=0.999),
            14,
        )
        solved = assert_sequential_solved(search.solve())
        assert solved.power >= 0.999
        assert search.evaluations <= MAX_EVALUATIONS

    def test_always_valid_decrease_near_the_peak(self):
        search = sequential_search(
            50,
            UNIT,
            make_procedure(
                inference=GaussianScoreMixture(), alternative="less", population="assigned"
            ),
            PowerDesign(power=0.30549514412021417),
            14,
        )
        solved = assert_sequential_solved(search.solve())
        assert solved.power >= 0.30549514412021417
        assert search.evaluations <= MAX_EVALUATIONS


class TestNodeCeiling:
    def test_an_unresolved_boundary_is_refused_never_consumed(self, monkeypatch):
        monkeypatch.setattr(sequential_module, "NODES_MAX", 64)
        procedure = make_procedure(
            inference=GaussianScoreMixture(), alpha=1e-300, population="assigned"
        )
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                procedure=procedure, n_per_arm=5000, baseline=BASE, planned_looks=14
            )
        assert refusal.value.code == "power.minimum_detectable_effect.numerical_resolution"
        assert set(refusal.value.context) == NUMERICAL_RESOLUTION_CONTEXT
        assert refusal.value.context["stopping_reason"] == (
            "node floor exceeds the quadrature ceiling"
        )
        se_search = sequential_search(5000, BASE, procedure, PowerDesign(), 14)
        se_full = se_search.se_full(0.4)
        with pytest.raises(InvalidRequestError) as node_floor:
            sequential_module.sequential_power(GaussianScoreMixture(), 0.4, se_full, 1e-300, 14)
        assert node_floor.value.code == "power.boundary_crossing_quadrature"
        assert node_floor.value.context["reason"] == "node floor exceeds the quadrature ceiling"


class TestExtremeAlternativeVariance:
    def test_underflowed_standard_error_does_not_raise_a_raw_numeric_error(self):
        result = minimum_detectable_effect(
            procedure=make_procedure(
                inference=GaussianScoreMixture(), alternative="greater", population="assigned"
            ),
            n_per_arm=100,
            baseline=Baseline(mean=1e300, var=1e-300),
            planned_looks=5,
        )
        assert result.mde_relative is not None
        assert math.isfinite(result.mde_relative)
        assert result.power == 1.0

    @pytest.mark.parametrize("inference", [GaussianScoreMixture()])
    def test_underflowed_standard_error_stays_log_scale_in_other_solvers(self, inference):
        procedure = make_procedure(
            inference=inference, alternative="greater", population="assigned"
        )
        baseline = Baseline(mean=1e300, var=1e-300)

        achieved = achieved_power(100, 0.1, baseline, procedure, planned_looks=5)
        sized = required_sample_size(0.1, baseline, procedure, planned_looks=5)

        assert achieved.power == 1.0
        assert achieved.expected_n_total == 40
        assert sized.n_per_arm == 2
        assert sized.power == 1.0
        assert sized.expected_n_total == 1
