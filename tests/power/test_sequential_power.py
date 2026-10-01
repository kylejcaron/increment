"""Sequential-aware power: planning n for GaussianScoreMixture."""

from collections.abc import MutableMapping
from operator import setitem
from typing import Any, cast

import pytest

from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.sequential import GaussianScoreMixture
from increment.power import (
    Baseline,
    PowerDesign,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
    required_sample_size,
)
from increment.power import sequential as sequential_module
from increment.power.sequential import (
    planning_bounds,
    sequential_expected_information_fraction,
    sequential_power,
    sequential_power_enclosure,
)

from ._procedures import make_procedure
from ._results import available_mde

BASE = Baseline(mean=1.0, var=2.0)


@pytest.mark.parametrize("rho", [0.3, -0.3])
@pytest.mark.parametrize("solver", ["required", "achieved", "mde"])
def test_sequential_power_supports_nonzero_cuped_rho(rho, solver):
    """CUPED now composes with sequential planning: arm_planning_support
    translates planning's ambiguous MethodSpec('cuped') into whichever
    flavour the asymptotic route always admits (retained_cuped), so a
    declared CUPED decision method under GaussianScoreMixture succeeds and
    a nonzero cuped_rho baseline sizes tighter than cuped_rho=0."""
    baseline = Baseline(mean=1.0, var=2.0, cuped_rho=rho)
    plain = Baseline(mean=1.0, var=2.0)
    cuped_procedure = make_procedure(
        inference=GaussianScoreMixture(), population="assigned", decision_method="cuped"
    )
    plain_procedure = make_procedure(
        inference=GaussianScoreMixture(), population="assigned", decision_method="unadjusted"
    )
    if solver == "required":
        with_cuped = required_sample_size(0.05, baseline, procedure=cuped_procedure)
        without_cuped = required_sample_size(0.05, plain, procedure=plain_procedure)
        assert with_cuped.n_per_arm < without_cuped.n_per_arm
    elif solver == "achieved":
        with_cuped = achieved_power(100, 0.05, baseline, procedure=cuped_procedure)
        without_cuped = achieved_power(100, 0.05, plain, procedure=plain_procedure)
        assert with_cuped.power > without_cuped.power
    else:
        with_cuped = minimum_detectable_effect(100, baseline, procedure=cuped_procedure)
        without_cuped = minimum_detectable_effect(100, plain, procedure=plain_procedure)
        assert available_mde(with_cuped) < available_mde(without_cuped)


def test_sequential_power_default_zero_cuped_rho_is_supported():
    result = achieved_power(
        100,
        0.05,
        BASE,
        procedure=make_procedure(
            inference=GaussianScoreMixture(), population="assigned", decision_method="unadjusted"
        ),
    )
    assert 0.0 <= result.power <= 1.0


@pytest.mark.parametrize(
    ("planned_looks", "code"),
    [
        pytest.param(True, "power.planned_looks_positive", id="bool"),
        pytest.param(2.5, "power.planned_looks_positive", id="float"),
        pytest.param(0, "power.planned_looks", id="zero"),
        pytest.param(-1, "power.planned_looks", id="negative"),
    ],
)
def test_sequential_power_rejects_invalid_planned_looks(
    planned_looks: Any,
    code: str,
):
    from increment.power.sequential import sequential_power

    with pytest.raises(InvalidRequestError) as raised:
        sequential_power(
            GaussianScoreMixture(),
            delta=0.05,
            se_full=0.1,
            alpha=0.05,
            planned_looks=planned_looks,
        )
    assert raised.value.code == code
    assert raised.value.context["planned_looks"] == planned_looks


class TestFixedPathUnchanged:
    def test_inference_none_matches_existing_closed_form(self):
        a = required_sample_size(procedure=make_procedure(), relative_lift=0.05, baseline=BASE)
        b = required_sample_size(
            procedure=make_procedure(inference=None), relative_lift=0.05, baseline=BASE
        )
        assert a == b

    def test_achieved_power_inference_none_matches_existing_closed_form(self):
        a = achieved_power(
            procedure=make_procedure(), n_per_arm=1000, relative_lift=0.05, baseline=BASE
        )
        b = achieved_power(
            procedure=make_procedure(inference=None),
            n_per_arm=1000,
            relative_lift=0.05,
            baseline=BASE,
        )
        assert a == b

    def test_mde_inference_none_matches_existing_closed_form(self):
        a = minimum_detectable_effect(procedure=make_procedure(), n_per_arm=1000, baseline=BASE)
        b = minimum_detectable_effect(
            procedure=make_procedure(inference=None), n_per_arm=1000, baseline=BASE
        )
        assert a == b

    def test_sequential_boundary_smoke_stays_fast_covered(self):
        """The one fast-suite execution of the sequential boundary code
        (planning_bounds plus the certified crossing evaluator) through a
        public solver: a two-look GaussianScoreMixture achieved-power call, plus
        its companion effect. Every other sequential-power test is
        slow-marked, so without this a regression in the boundary math
        would pass the fast suite silently."""
        from increment.estimation.sequential import GaussianScoreMixture

        result = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=5000,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=2,
        )
        # Sequential power at any look is strictly below the fixed-horizon
        # closed form at the same n - the boundary is wider by construction.
        fixed = achieved_power(
            procedure=make_procedure(), n_per_arm=5000, relative_lift=0.05, baseline=BASE
        )
        assert 0.0 < result.power < fixed.power


class TestExpectedNTotal:
    """expected_n_total: the early-stopping counterweight to n_max (wwzp)."""

    def test_none_for_fixed_horizon(self):
        result = required_sample_size(procedure=make_procedure(), relative_lift=0.05, baseline=BASE)
        assert result.expected_n_total is None

    def test_below_n_max_under_meaningful_early_stopping(self):
        from increment.estimation.sequential import GaussianScoreMixture

        result = required_sample_size(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=8,
        )
        assert result.expected_n_total is not None
        assert 0 < result.expected_n_total < result.n_total

    def test_achieved_power_and_mde_also_report_expected_n_total(self):
        from increment.estimation.sequential import GaussianScoreMixture

        spec = GaussianScoreMixture()
        ap = achieved_power(
            procedure=make_procedure(inference=spec, population="assigned"),
            n_per_arm=5000,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=8,
        )
        mde = minimum_detectable_effect(
            procedure=make_procedure(inference=spec, population="assigned"),
            n_per_arm=5000,
            baseline=BASE,
            planned_looks=8,
        )
        for result in (ap, mde):
            assert result.expected_n_total is not None
            assert 0 < result.expected_n_total <= result.n_total

    def test_single_look_expected_n_total_equals_n_total(self):
        # One look means the trial either exits there or never - always at
        # information fraction 1.0, so E[N] == n_total exactly.
        from increment.estimation.sequential import GaussianScoreMixture

        result = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=5000,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=1,
        )
        assert result.expected_n_total == result.n_total


@pytest.mark.slow
class TestSequentialPower:
    def test_always_valid_costs_two_to_three_x(self):
        from increment.estimation.sequential import GaussianScoreMixture

        n_fixed = required_sample_size(
            procedure=make_procedure(), relative_lift=0.05, baseline=BASE
        ).n_per_arm
        p_at_fixed = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=n_fixed,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        ).power
        p_at_3x = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=3 * n_fixed,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        ).power
        assert p_at_fixed < 0.65  # far below target at the fixed-horizon n
        assert p_at_3x >= 0.80


@pytest.mark.slow
class TestRoundTrip:
    def test_required_n_delivers_target_power(self):
        from increment.estimation.sequential import GaussianScoreMixture

        spec = GaussianScoreMixture()
        res = required_sample_size(
            procedure=make_procedure(inference=spec, population="assigned"),
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        )
        p = achieved_power(
            procedure=make_procedure(inference=spec, population="assigned"),
            n_per_arm=res.n_per_arm,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        ).power
        assert p >= 0.80 - 1e-6

    def test_unit_baseline_keeps_all_solver_entries_valid(self):
        from increment.estimation.sequential import GaussianScoreMixture

        inference = GaussianScoreMixture()
        required = required_sample_size(
            procedure=make_procedure(inference=inference, population="assigned"),
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        )
        achieved = achieved_power(
            required.n_per_arm,
            0.05,
            BASE,
            procedure=make_procedure(inference=inference, population="assigned"),
            planned_looks=5,
        )
        mde = minimum_detectable_effect(
            required.n_per_arm,
            BASE,
            procedure=make_procedure(inference=inference, population="assigned"),
            planned_looks=5,
        )
        assert achieved.power >= 0.80 - 1e-6
        assert available_mde(mde) > 0

    def test_mde_relative_is_internally_consistent_with_power(self):
        """A PowerResult's mde_relative and power must describe the SAME
        design: under sequential inference, achieved_power's mde_relative
        must be the sequential MDE at that n, not the fixed-horizon one;
        the fixed-horizon MDE is detectable at a HIGHER power than the
        sequential boundary reports at that same se, so the two fields
        would silently contradict each other otherwise (achieved_power's
        own power, evaluated AT its own reported mde_relative, must equal
        design.power).
        """
        from increment.estimation.sequential import GaussianScoreMixture

        n_fixed = required_sample_size(
            procedure=make_procedure(), relative_lift=0.05, baseline=BASE
        ).n_per_arm
        result = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=n_fixed,
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        )
        # mde_relative must be the sequential root-find, not the fixed-horizon
        # closed form (detectable at a different, higher power) - both must reach 0.80.
        check = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=n_fixed,
            relative_lift=available_mde(result),
            baseline=BASE,
            planned_looks=5,
        )
        assert check.power == pytest.approx(0.80, abs=1e-6)

        # required_sample_size's mde_relative must be consistent the same way.
        req = required_sample_size(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            relative_lift=0.05,
            baseline=BASE,
            planned_looks=5,
        )
        check2 = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=req.n_per_arm,
            relative_lift=available_mde(req),
            baseline=BASE,
            planned_looks=5,
        )
        assert check2.power == pytest.approx(0.80, abs=1e-6)

    def test_mde_round_trip_delivers_target_power(self):
        from increment.estimation.sequential import GaussianScoreMixture

        n_fixed = required_sample_size(
            procedure=make_procedure(), relative_lift=0.05, baseline=BASE
        ).n_per_arm
        mde_fixed = available_mde(
            minimum_detectable_effect(procedure=make_procedure(), n_per_arm=n_fixed, baseline=BASE)
        )
        result = minimum_detectable_effect(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=n_fixed,
            baseline=BASE,
            planned_looks=5,
        )
        # Sequential MDE must exceed the fixed-horizon MDE at the same n - it
        # needs a larger true effect to hit the same power (required-n inflation, mirrored).
        assert available_mde(result) > mde_fixed
        achieved = achieved_power(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            n_per_arm=n_fixed,
            relative_lift=available_mde(result),
            baseline=BASE,
            planned_looks=5,
        ).power
        assert achieved >= 0.80 - 1e-6

    def test_required_n_grows_bracket_then_converges(self):
        # A low target power (near the boundary's own alpha spend) pushes the
        # sequential/fixed-horizon ratio past the initial 8x bracket, forcing
        # the `hi *= 2` growth loop before bisection. GaussianScoreMixture's
        # se-independent tuning keeps that ratio modest at ordinary powers.
        from increment.estimation.sequential import GaussianScoreMixture

        design = PowerDesign(power=0.08)
        n_fixed = required_sample_size(
            procedure=make_procedure(), relative_lift=0.05, baseline=BASE, design=design
        ).n_per_arm
        n = required_sample_size(
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            relative_lift=0.05,
            baseline=BASE,
            design=design,
            planned_looks=5,
        ).n_per_arm
        assert n > 8 * n_fixed, "test must actually exercise the growth loop"
        assert n <= 1024 * n_fixed

    def test_sequential_cuped_preflight_precedes_zero_lift_sizing(self):
        # relative_lift=0.0 must raise the purpose-built null-boundary error
        # (zero distance from the null has no finite n on any path) even
        # though CUPED now composes with sequential planning and no longer
        # refuses first.
        from increment.estimation.sequential import GaussianScoreMixture

        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(
                procedure=make_procedure(
                    inference=GaussianScoreMixture(), population="assigned", decision_method="cuped"
                ),
                relative_lift=0.0,
                baseline=BASE,
                planned_looks=5,
            )
        assert raised.value.code == "power.size_design_relative"


@pytest.mark.slow
class TestOneSidedSequentialPower:
    def test_less_returns_negative_mde(self):
        from increment.estimation.sequential import GaussianScoreMixture

        r = minimum_detectable_effect(
            procedure=make_procedure(
                alternative="less", inference=GaussianScoreMixture(), population="assigned"
            ),
            n_per_arm=5000,
            baseline=BASE,
            design=PowerDesign(),
            planned_looks=5,
        )
        assert available_mde(r) < 0

    def test_zero_lift_sequential_sizing_refused_specifically(self):
        from increment.estimation.sequential import GaussianScoreMixture

        spec = GaussianScoreMixture()
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                procedure=make_procedure(inference=spec, population="assigned"),
                relative_lift=0.0,
                baseline=BASE,
                planned_looks=5,
            )
        assert exc_info.value.code == "power.size_design_relative"

    def test_zero_lift_fixed_horizon_refused_consistently(self):
        # Zero-distance sizing refuses on the fixed-horizon path too - same
        # message as the sequential paths, replacing the old n_per_arm=2 sentinel.
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(procedure=make_procedure(), relative_lift=0.0, baseline=BASE)
        assert exc_info.value.code == "power.size_design_relative"


class TestHighPowerTargets:
    """High targets drive almost all of the path's mass out at early looks;
    the answers still reach their targets with certified enclosures."""

    def test_minimum_detectable_effect_always_valid_reaches_the_target(self):
        result = minimum_detectable_effect(
            5000,
            Baseline(mean=1.0, var=2.0),
            make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            PowerDesign(power=0.95),
        )
        assert 0.95 <= result.power <= 1.0
        assert result.power == pytest.approx(0.95, abs=1e-9)
        assert available_mde(result) > 0.0


class TestPlanningBoundsFractionValidation:
    """``planning_bounds`` validates the fraction schedule once, before any
    numerical work runs, for every accepted planning spec."""

    @pytest.mark.parametrize(
        ("fractions", "code"),
        [
            pytest.param([0.5, 1.1], "power.information_fractions_lie", id="above-one"),
            pytest.param([0.5, 0.4], "power.information_fractions_strictly", id="not-increasing"),
            pytest.param([0.0, 1.0], "power.information_fractions_lie", id="zero-fraction"),
            pytest.param([], "power.fractions_nonempty", id="empty"),
        ],
    )
    def test_always_valid_rejects_invalid_fractions(self, fractions, code):
        from increment.power.sequential import planning_bounds

        with pytest.raises(InvalidRequestError) as raised:
            planning_bounds(GaussianScoreMixture(), fractions, se_full=0.1, alpha=0.05)
        assert raised.value.code == code

    def test_valid_fractions_still_produce_bounds(self):
        from increment.power.sequential import planning_bounds

        bounds = planning_bounds(GaussianScoreMixture(), [0.5, 1.0], se_full=0.1, alpha=0.05)
        assert len(bounds) == 2
        assert all(b > 0 for b in bounds)


class TestSeFullValidation:
    @pytest.mark.parametrize("se_full", [0.0, -1.0, float("inf"), float("nan")])
    def test_planning_bounds_rejects_invalid_se_full(self, se_full):
        from increment.power.sequential import planning_bounds

        with pytest.raises(InvalidRequestError) as raised:
            planning_bounds(GaussianScoreMixture(), [0.5, 1.0], se_full=se_full, alpha=0.05)
        assert raised.value.code == "power.se_full_finite"
        if not (isinstance(se_full, float) and se_full != se_full):
            assert raised.value.context["se_full"] == se_full

    @pytest.mark.parametrize("se_full", [0.0, -1.0, float("inf"), float("nan")])
    def test_sequential_power_enclosure_rejects_invalid_se_full(self, se_full):
        from increment.power.sequential import sequential_power_enclosure

        with pytest.raises(InvalidRequestError) as raised:
            sequential_power_enclosure(GaussianScoreMixture(), 0.05, se_full, 0.05, 14)
        assert raised.value.code == "power.se_full_finite"
        if not (isinstance(se_full, float) and se_full != se_full):
            assert raised.value.context["se_full"] == se_full

    def test_se_full_validation_matches_across_entry_points(self):
        from increment.power.sequential import planning_bounds, sequential_power_enclosure

        with pytest.raises(InvalidRequestError) as via_planning:
            planning_bounds(GaussianScoreMixture(), [0.5, 1.0], se_full=0.0, alpha=0.05)
        with pytest.raises(InvalidRequestError) as via_enclosure:
            sequential_power_enclosure(GaussianScoreMixture(), 0.05, 0.0, 0.05, 14)

        assert via_planning.value.code == via_enclosure.value.code == "power.se_full_finite"


def _registered_runtime():
    from increment import AlwaysValid
    from tests.sequential_cases import registration

    return {"anytime": AlwaysValid(registration=registration("gaussian"))}


STATEFUL = _registered_runtime()


def _forbid_numerical_work(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("numerical work ran before the schedule was validated")

    monkeypatch.setattr(sequential_module, "_cheap_bounds", forbidden)
    monkeypatch.setattr(sequential_module, "_gl_walk", forbidden)


def _solve(solver: str, procedure: Any, **kwargs: Any) -> Any:
    if solver == "required":
        return required_sample_size(0.05, BASE, procedure, **kwargs)
    if solver == "achieved":
        return achieved_power(2_000, 0.05, BASE, procedure, **kwargs)
    if solver == "mde":
        return minimum_detectable_effect(2_000, BASE, procedure, **kwargs)
    return power_curve(
        n_per_arm=2_000, relative_lift=0.05, baseline=BASE, procedure=procedure, **kwargs
    )


SOLVERS = ["required", "achieved", "mde", "curve"]


class TestRuntimeStateRefusedInPlanning:
    """A registered runtime policy carrying observed or absolute runtime
    state is not a prospective planning spec; it fails coded before any
    boundary is built."""

    @pytest.mark.parametrize("solver", SOLVERS)
    @pytest.mark.parametrize("fields", list(STATEFUL))
    def test_refused_before_numerical_work(self, solver, fields, monkeypatch):
        _forbid_numerical_work(monkeypatch)
        spec = STATEFUL[fields]
        with pytest.raises(CapabilityError) as raised:
            _solve(solver, make_procedure(inference=spec, population="assigned"))
        assert raised.value.code == "sequential.route.unsupported"
        # Deliberately violate the read-only type to test runtime immutability.
        with pytest.raises(TypeError):
            setitem(cast(MutableMapping[str, object], raised.value.context), "reason", "changed")

    def test_accepted_finding_probe_no_longer_returns_equal_power(self):
        baseline = Baseline(mean=10.0, var=25.0)
        plain = achieved_power(
            n_per_arm=500,
            relative_lift=0.1,
            baseline=baseline,
            procedure=make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            planned_looks=3,
        )
        assert plain.power == pytest.approx(0.5734914730354411, abs=1e-9)
        with pytest.raises(CapabilityError) as raised:
            achieved_power(
                n_per_arm=500,
                relative_lift=0.1,
                baseline=baseline,
                procedure=make_procedure(inference=STATEFUL["anytime"]),
                planned_looks=3,
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_low_level_entry_points_refuse_runtime_state(self, monkeypatch):
        _forbid_numerical_work(monkeypatch)
        spec = STATEFUL["anytime"]
        for call in (
            lambda: sequential_power(spec, 0.05, 0.02, 0.05, 3),
            lambda: sequential_power_enclosure(spec, 0.05, 0.02, 0.05),
            lambda: sequential_expected_information_fraction(spec, 0.05, 0.02, 0.05, 3),
            lambda: planning_bounds(spec, (0.5, 1.0), 1.0, 0.05),
        ):
            with pytest.raises(CapabilityError) as raised:
                call()
            assert raised.value.code == "sequential.route.unsupported"


class TestLookCountResolution:
    """``planned_looks`` omitted defaults to fourteen equal looks."""

    @pytest.mark.parametrize("solver", ["required", "achieved", "mde"])
    def test_fixed_horizon_ignores_an_omitted_count(self, solver):
        omitted = _solve(solver, make_procedure())
        explicit = _solve(solver, make_procedure(), planned_looks=14)
        assert omitted == explicit

    def test_omitted_count_without_a_schedule_is_fourteen_equal_looks(self):
        spec = GaussianScoreMixture()
        enclosure = sequential_power_enclosure(spec, 0.05, 0.02, 0.05)
        fourteen = sequential_power_enclosure(spec, 0.05, 0.02, 0.05, 14)
        assert enclosure == fourteen
        assert sequential_power(spec, 0.05, 0.02, 0.05) == fourteen.estimate
        assert sequential_expected_information_fraction(spec, 0.05, 0.02, 0.05) == (
            fourteen.expected_fraction
        )
        assert sequential_power(spec, 0.05, 0.02, 0.05, 13) != fourteen.estimate

    @pytest.mark.slow
    @pytest.mark.parametrize("solver", ["required", "achieved", "mde"])
    def test_public_solvers_keep_the_fourteen_look_default(self, solver):
        spec = GaussianScoreMixture()
        omitted = _solve(solver, make_procedure(inference=spec, population="assigned"))
        explicit = _solve(
            solver, make_procedure(inference=spec, population="assigned"), planned_looks=14
        )
        assert omitted == explicit
