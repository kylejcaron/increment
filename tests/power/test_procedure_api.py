import copy
import math
import pickle
from collections.abc import MutableMapping
from typing import cast

import pytest
from scipy.stats import norm

from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation._tails import student_t_isf
from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.estimation.sequential import GaussianScoreMixture
from increment.power import segment_pairwise_achieved_power
from increment.power.core import (
    Baseline,
    PowerDesign,
    _assigned_minimum_per_arm,
    _compute_arms,
    _power_at,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)
from increment.power.curve import power_curve
from increment.semantics.models import InferenceSpec
from tests.power._procedures import make_procedure


@pytest.mark.parametrize("field", ["alpha", "alternative", "null_lift", "n_variants", "correction"])
def test_power_design_contains_numeric_controls_only(field: str) -> None:
    with pytest.raises(InvalidRequestError) as raised:
        PowerDesign(
            **{field: 0.05 if field == "alpha" else "two-sided" if field == "alternative" else 0}
        )
    assert raised.value.code == "model.field.unknown"
    assert raised.value.context["field"] == field


def test_scalar_requires_a_validated_relative_procedure() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        required_sample_size(
            0.1,
            Baseline(mean=1.0, var=1.0),
            procedure=cast("ArmPlanningProcedure", object()),
        )
    assert exc_info.value.code == "power.procedure_armplanningprocedure"


def test_baseline_refusal_precedes_solver_numerics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """identification='observational' is an explicit choice, not
    standard()'s undeclared default -- _derive_axes_from_baseline leaves it
    alone, so a compliance-bearing baseline still refuses instead of being
    silently upgraded to 'encouragement'."""
    procedure = make_procedure(
        identification="observational",
        population="assigned",
        variance_adjustment="none",
        decision_method="unadjusted",
    )
    baseline = Baseline(mean=1.0, var=1.0, compliance=0.8)
    monkeypatch.setattr(
        "increment.power.core._power_at", lambda *args, **kwargs: pytest.fail("solver ran")
    )

    with pytest.raises(CapabilityError) as raised:
        achieved_power(100, 0.1, baseline, procedure=procedure)
    assert raised.value.code == "arm.baseline.compliance_undeclared"


def test_triggered_population_refuses_under_sequential_inference() -> None:
    """A triggered design has no runtime registration under asymptotic_mean
    (sequential_source.py refuses population != "assigned" unconditionally)
    -- refused, not silently sized. (Deferred from Task P1's inference_to_declare
    fix: this is Task P3's arm_planning_support refusal, exercised directly
    on an explicitly triggered procedure.)"""
    procedure = make_procedure(
        population="triggered",
        identification="randomized",
        variance_adjustment="none",
        inference=GaussianScoreMixture(),
    )
    baseline = Baseline(mean=20.0, var=400.0, trigger_rate=0.4)
    with pytest.raises(CapabilityError) as raised:
        required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
    assert raised.value.code == "sequential.route.unsupported"


def test_quantile_procedure_floor_is_enforced() -> None:
    procedure = make_procedure(metric_type="quantile")
    with pytest.raises(InvalidRequestError) as exc_info:
        achieved_power(19, 0.1, Baseline(mean=1.0, var=1.0), procedure=procedure)
    assert exc_info.value.code == "power.core.n_per_arm_min"
    assert exc_info.value.context["minimum"] == 20


def test_triggered_quantile_floor_is_on_assigned_units() -> None:
    procedure = make_procedure(metric_type="quantile")
    baseline = Baseline(mean=1.0, var=1.0, trigger_rate=0.1)
    with pytest.raises(InvalidRequestError) as exc_info:
        achieved_power(199, 0.1, baseline, procedure=procedure)
    assert exc_info.value.code == "power.core.n_per_arm_min"
    assert exc_info.value.context["minimum"] == 200
    result = required_sample_size(5.0, baseline, procedure)
    assert result.n_per_arm == 200


def test_imbalanced_quantile_floor_applies_to_derived_control_arm() -> None:
    procedure = make_procedure(metric_type="quantile")
    result = minimum_detectable_effect(
        20,
        Baseline(mean=1.0, var=1.0),
        procedure,
        PowerDesign(allocation=0.9),
    )
    assert result.n_per_arm == 20
    assert result.n_total == 40


def test_clustered_triggered_imbalanced_floor_uses_assigned_controls() -> None:
    # The public arm contract refuses quantile + cluster, so no end-to-end
    # call can carry the 20-unit quantile floor into a clustered search.
    # This direct-seam construction intentionally bypasses that refusal
    # only to exercise the assigned-unit conversion of the 20-unit floor.
    procedure = make_procedure(dependence="cluster", decision_method="unadjusted")
    baseline = Baseline(
        mean=1.0, var=1.0, avg_cluster_size=20.0, trigger_rate=0.1, cluster_participation=1.0
    )
    assigned_floor = _assigned_minimum_per_arm(
        procedure.model_copy(
            update={"metric": procedure.metric.model_copy(update={"metric_type": "quantile"})}
        ),
        baseline,
    )
    treatment, control = _compute_arms(
        23,
        PowerDesign(allocation=0.9),
        minimum_per_arm=assigned_floor,
    )
    assert treatment == 200
    assert control == 200
    assert control * baseline.trigger_rate >= 20


def test_clustered_triggered_public_floor_preserves_analyzed_control_units() -> None:
    procedure = make_procedure(dependence="cluster", decision_method="unadjusted")
    baseline = Baseline(
        mean=1.0, var=1.0, avg_cluster_size=20.0, trigger_rate=0.1, cluster_participation=1.0
    )
    result = required_sample_size(
        1_000.0,
        baseline,
        procedure,
        PowerDesign(allocation=0.99),
    )
    analyzed_control = (result.n_total - result.n_per_arm) * baseline.trigger_rate
    assert result.n_per_arm >= 20
    assert analyzed_control >= 2


def test_quantile_pairwise_contrast_refuses_as_a_quantile_breakout() -> None:
    procedure = make_procedure(metric_type="quantile")
    with pytest.raises(CapabilityError) as exc_info:
        segment_pairwise_achieved_power(
            39,
            0.2,
            0.05,
            0.5,
            0.5,
            Baseline(mean=1.0, var=1.0),
            procedure=procedure,
        )
    assert exc_info.value.code == "readout.metric.quantile_breakout"
    assert exc_info.value.context["metric"] is None
    assert exc_info.value.context["solver"] == "segment_pairwise_achieved_power"


def test_curve_refuses_baseline_before_worker_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    procedure = make_procedure(
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        decision_method="unadjusted",
    )
    baseline = Baseline(mean=1.0, var=1.0, avg_cluster_size=5.0)
    monkeypatch.setattr(
        "increment.power.curve.ThreadPoolExecutor", lambda **kwargs: pytest.fail("workers ran")
    )

    with pytest.raises(CapabilityError) as raised:
        power_curve(
            n_per_arm=[100, 200, 300, 400],
            relative_lift=0.1,
            baseline=baseline,
            procedure=procedure,
            max_workers=4,
        )
    assert raised.value.code == "arm.baseline.iid_cluster_knobs"


@pytest.mark.parametrize("family_size", [1, 3])
def test_tiny_compiled_alpha_has_finite_scalar_boundaries(family_size: int) -> None:
    nominal = math.ldexp(1.0, -1022)
    procedure = make_procedure(alpha=nominal, family_size=family_size)
    baseline = Baseline(mean=1.0, var=1.0)
    design = PowerDesign(power=0.8)

    sized = required_sample_size(0.5, baseline, procedure, design)
    achieved = achieved_power(sized.n_per_arm, 0.5, baseline, procedure, design)
    mde = minimum_detectable_effect(sized.n_per_arm, baseline, procedure, design)

    assert math.isfinite(procedure.compiled_alpha)
    assert procedure.compiled_alpha > 0.0
    assert math.isfinite(float(norm.isf(procedure.compiled_tail_alpha)))
    assert math.isfinite(achieved.power)
    assert mde.mde_relative is not None
    assert math.isfinite(mde.mde_relative)


def test_curve_records_decision_alpha_without_second_correction() -> None:
    procedure = make_procedure(alpha=0.05, family_size=3)
    point = power_curve(
        n_per_arm=5_000,
        relative_lift=0.1,
        baseline=Baseline(mean=1.0, var=1.0),
        procedure=procedure,
    )[0]

    assert point.alpha == procedure.compiled_alpha
    assert point.alpha != procedure.compiled_tail_alpha
    assert (
        point.power
        == achieved_power(
            5_000,
            0.1,
            Baseline(mean=1.0, var=1.0),
            procedure=procedure,
        ).power
    )


def test_tiny_compiled_alpha_matches_high_precision_normal_and_t_fixtures() -> None:
    nominal = math.ldexp(1.0, -1022)
    iid = make_procedure(alpha=nominal)
    cluster = make_procedure(
        alpha=nominal,
        dependence="cluster",
        decision_method="unadjusted",
    )
    baseline = Baseline(mean=1.0, var=1.0)

    assert float(norm.isf(iid.compiled_tail_alpha)) == pytest.approx(
        37.537836095576054, rel=0.0, abs=1e-13
    )
    # The oracle is the correctly-rounded binary64 quantile (independent
    # high-precision computation); SciPy's own t.isf at this subnormal tail
    # underflows to -inf, which is exactly the failure this shared primitive
    # exists to avoid. Assert against the shared primitive, not SciPy's.
    assert student_t_isf(cluster.compiled_tail_alpha, 98.0) == pytest.approx(
        13297.025372284563, rel=1e-9, abs=0.0
    )
    scalar = achieved_power(100, 0.1, baseline, procedure=iid)
    clustered_result = achieved_power(
        1_000,
        0.1,
        Baseline(mean=1.0, var=1.0, avg_cluster_size=20.0),
        procedure=cluster,
    )
    t_power = _power_at(0.1, 0.02, PowerDesign(), cluster, dof=98.0)
    # Own-arm noncentrality: log1p(.1) / sqrt(1/(n*1.1^2) + 1/n) at n=100 and n=1000.
    assert scalar.power == pytest.approx(2.777584655323694e-297, rel=1e-12, abs=0.0)
    # 50 clusters per arm, so the clustered reference is the analyzer's own
    # conservative min(K_t - 1, K_c - 1) = 49, not a pooled K - 2. An absolute
    # tolerance at this magnitude would let zero pass, so pin the independent
    # high-precision two-sided noncentral-t oracle relatively.
    assert clustered_result.power == pytest.approx(2.220916306147259e-302, rel=1e-9, abs=0.0)
    # At t_49 the critical value is 1.27e7, so even the largest representable
    # relative lift (exp(709.78) - 1) reaches only 4.3e-126 power: no float64
    # companion effect attains the 0.8 target, and the reason must say so.
    assert clustered_result.mde_relative is None
    assert clustered_result.mde_unavailable_reason == "unrepresentable"
    # Absolute tolerance would swamp a target this tiny; the independent
    # noncentral-power oracle (chi-square mixture, no nct.cdf/sf) agrees to
    # within a few parts in 1e13 regardless of which correctly-rounded
    # critical value feeds it, so a relative-only tolerance is the honest test.
    assert t_power == pytest.approx(1.0973882289776205e-305, rel=1e-9, abs=0.0)

    pairwise = segment_pairwise_achieved_power(
        1_000,
        0.2,
        0.05,
        0.2,
        0.2,
        baseline,
        procedure=iid,
    )
    curve = power_curve(
        n_per_arm=1_000,
        relative_lift=0.1,
        baseline=baseline,
        procedure=iid,
    )[0]
    assert pairwise.power == pytest.approx(1.8060131097476627e-293, rel=1e-12, abs=0.0)
    assert curve.power == pytest.approx(2.238624723817254e-273, rel=1e-12, abs=0.0)
    assert curve.alpha == iid.compiled_alpha


class TestStandardProcedure:
    """`ArmPlanningProcedure.standard` is the documented way in, so the
    choices it makes must be the ones a caller would otherwise write out."""

    @staticmethod
    def _standard(**kwargs):
        from increment.power import ArmPlanningProcedure

        return ArmPlanningProcedure.standard(**kwargs)

    def test_comparisons_splits_alpha_and_widens_the_sample_size(self):
        from increment.power import Baseline, required_sample_size

        one = self._standard()
        three = self._standard(comparisons=3)
        assert one.decision.family.kind == "none"
        assert three.decision.family.kind == "bonferroni"
        assert three.decision.family.axes == ("arm",)
        # Bonferroni over three arms is the conservative exact split, and a
        # smaller alpha must cost sample size.
        assert three.compiled_alpha == pytest.approx(0.05 / 3)
        assert one.compiled_alpha == pytest.approx(0.05)
        baseline = Baseline(mean=20.0, var=400.0)
        wide = required_sample_size(relative_lift=0.02, baseline=baseline, procedure=three)
        narrow = required_sample_size(relative_lift=0.02, baseline=baseline, procedure=one)
        assert wide.n_per_arm > narrow.n_per_arm

    def test_segmented_inference_is_refused_without_silent_loss(self):
        inference = InferenceSpec(
            kind="asymptotic_mean",
            segments={"segment": ("x", "y")},
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            self._standard(inference=inference)

        error = exc_info.value
        assert error.code == "arm_planning.inference_spec_segments_unplannable"
        assert error.context["rejected_segments"] == {"segment": ("x", "y")}
        assert error.context["view"] == "total"
        assert error.context["segmented"] is False
        assert isinstance(error.context["route"], str)
        assert error.context["route"].strip()
        with pytest.raises(TypeError):
            cast("MutableMapping[str, object]", error.context)["rejected_segments"] = {}
        for cloned in (copy.deepcopy(error), pickle.loads(pickle.dumps(error))):
            assert cloned.code == error.code
            assert cloned.context == error.context

    def test_clustered_selects_the_cluster_dependence(self):
        assert self._standard().dependence == "iid"
        assert self._standard(clustered=True).dependence == "cluster"

    def test_absolute_scale_selects_an_absolute_decision_policy(self):
        from increment.estimation.arm_contract import (
            AbsoluteDecisionPolicy,
            RelativeDecisionPolicy,
        )

        assert isinstance(self._standard().decision, RelativeDecisionPolicy)
        absolute = self._standard(value_scale="absolute", null_lift=1.5)
        assert isinstance(absolute.decision, AbsoluteDecisionPolicy)
        assert absolute.decision.null_abs == pytest.approx(1.5)
        assert absolute.metric.value_scale == "absolute"

    def test_one_sided_alternative_reaches_the_compiled_tail(self):
        two_sided = self._standard()
        greater = self._standard(alternative="greater")
        assert greater.decision.alternative == "greater"
        # A one-sided test spends its whole alpha in one tail.
        assert greater.compiled_tail_alpha == pytest.approx(greater.compiled_alpha)
        assert two_sided.compiled_tail_alpha == pytest.approx(two_sided.compiled_alpha / 2)


class TestSolverDerivesAxesFromBaseline:
    """Every guide-documented Baseline field is derived from the SAME
    baseline the solver call already receives -- standard() itself takes
    no baseline-related keyword."""

    def test_cuped_is_derived_from_the_solver_baseline(self):
        procedure = ArmPlanningProcedure.standard("mean")
        baseline = Baseline(mean=20.0, var=400.0, cuped_rho=0.5)
        result = required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert result.n_per_arm > 0

    def test_compliance_is_derived_from_the_solver_baseline(self):
        procedure = ArmPlanningProcedure.standard("mean")
        baseline = Baseline(mean=20.0, var=400.0, compliance=0.6)
        result = required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert result.n_per_arm > 0

    def test_triggered_is_derived_from_the_solver_baseline(self):
        procedure = ArmPlanningProcedure.standard("mean")
        baseline = Baseline(mean=20.0, var=400.0, trigger_rate=0.4)
        result = required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert result.n_per_arm > 0

    def test_absorption_is_derived_from_the_solver_baseline(self):
        procedure = ArmPlanningProcedure.standard("mean")
        baseline = Baseline(mean=20.0, var=400.0, icc=0.1)
        result = required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert result.n_per_arm > 0

    def test_one_standard_procedure_composes_with_either_baseline(self):
        """No hidden per-procedure state from an earlier baseline= call to
        go stale -- the SAME standard() object works with a plain baseline
        and a CUPED one."""
        procedure = ArmPlanningProcedure.standard("mean")
        plain = required_sample_size(
            0.05, Baseline(mean=20.0, var=400.0), procedure, PowerDesign(power=0.80)
        )
        cuped = required_sample_size(
            0.05,
            Baseline(mean=20.0, var=400.0, cuped_rho=0.6),
            procedure,
            PowerDesign(power=0.80),
        )
        assert cuped.n_per_arm < plain.n_per_arm

    def test_sensitivity_only_cuped_does_not_subsidize_decision_power(self):
        """A CUPED sensitivity is reported separately from an unadjusted
        decision and cannot reduce the decision's required sample size."""
        from increment.semantics.models import MethodSpec

        base = ArmPlanningProcedure.standard("mean")
        procedure = base.model_copy(
            update={"sensitivity_methods": (MethodSpec(name="cuped", variance_reduction="cuped"),)}
        )
        baseline = Baseline(mean=1.0, var=1.0, cuped_rho=0.9)

        planned = required_sample_size(0.1, baseline, procedure)
        unadjusted = required_sample_size(0.1, Baseline(mean=1.0, var=1.0), procedure)
        actual = achieved_power(planned.n_per_arm, 0.1, Baseline(mean=1.0, var=1.0), procedure)

        assert procedure.decision_method.variance_reduction == "none"
        assert procedure.sensitivity_methods[0].variance_reduction == "cuped"
        assert baseline.cuped_rho == pytest.approx(0.9)
        assert planned.n_per_arm == unadjusted.n_per_arm == 1579
        assert planned.power == pytest.approx(actual.power)
        assert actual.power == pytest.approx(0.8, abs=0.002)


class TestExplicitConflictingAxesRefuse:
    """A procedure explicitly built with an axis that conflicts with the
    solver's baseline is refused, not silently overridden."""

    def test_explicit_identification_conflicts_with_compliance(self):
        procedure = make_procedure(
            identification="observational",
            population="assigned",
            variance_adjustment="none",
            decision_method="unadjusted",
        )
        baseline = Baseline(mean=20.0, var=400.0, compliance=0.6)
        with pytest.raises(CapabilityError) as raised:
            required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert raised.value.code == "arm.baseline.compliance_undeclared"
        assert raised.value.context["knob"] == "compliance"

    def test_cluster_knob_on_an_iid_procedure_names_the_offending_knob(self):
        procedure = ArmPlanningProcedure.standard("mean")
        baseline = Baseline(mean=20.0, var=400.0, avg_cluster_size=5.0)
        with pytest.raises(CapabilityError) as raised:
            required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert raised.value.code == "arm.baseline.iid_cluster_knobs"
        assert raised.value.context["knob"] == "avg_cluster_size"

    def test_missing_avg_cluster_size_on_a_clustered_procedure_names_it(self):
        procedure = ArmPlanningProcedure.standard("mean", clustered=True)
        baseline = Baseline(mean=20.0, var=400.0)
        with pytest.raises(CapabilityError) as raised:
            required_sample_size(0.05, baseline, procedure, PowerDesign(power=0.80))
        assert raised.value.code == "arm.baseline.cluster_size_required"
        assert raised.value.context["knob"] == "avg_cluster_size"


class TestCupedUnderSequentialPlanning:
    """CUPED + sequential planning is supported wherever the runtime
    supports it: derived automatically from the solver's baseline
    exactly as the fixed-horizon case is."""

    def test_cuped_composes_with_sequential_inference(self):
        from increment.semantics.models import InferenceSpec

        procedure = ArmPlanningProcedure.standard(
            "mean", inference=InferenceSpec(kind="asymptotic_mean")
        )
        baseline = Baseline(mean=20.0, var=400.0, cuped_rho=0.4)
        result = required_sample_size(
            0.05, baseline, procedure, PowerDesign(power=0.80), planned_looks=14
        )
        assert result.n_per_arm > 0
        assert result.inference_to_declare is not None


class TestSecondaryPlanning:
    """standard() sizes a fixed-horizon secondary at alpha=q/m and a
    sequential secondary at min(alpha, q/m), m composed exactly as the
    runtime keys its own secondary family."""

    def test_fixed_horizon_secondary_alpha_is_q_over_secondaries_times_comparisons(self):
        procedure = ArmPlanningProcedure.standard(
            "mean", role="secondary", secondaries=2, comparisons=2, q=0.10
        )
        assert procedure.compiled_alpha == pytest.approx(0.10 / 4)

    def test_planned_secondary_alpha_matches_runtime_worst_case_bh_threshold(self):
        """Reproduces the runtime's own worst-case realized threshold
        (family.py::bh_select at k=1) bit for bit -- AGENTS.md's
        'quantities computed twice' guard for a 2-secondary, 2-arm plan."""
        from increment.estimation.family import bh_select

        q = 0.10
        m = 4  # 2 secondaries x 2 arms
        procedure = ArmPlanningProcedure.standard(
            "mean", role="secondary", secondaries=2, comparisons=2, q=q
        )
        p_values = [q / m - 1e-9, 0.99, 0.99, 0.99]
        _, realized_threshold = bh_select(p_values, q)
        assert procedure.compiled_alpha == pytest.approx(realized_threshold, rel=1e-9)

    def test_sequential_secondary_alpha_ignores_comparisons(self):
        from increment.semantics.models import InferenceSpec

        procedure = ArmPlanningProcedure.standard(
            "mean",
            role="secondary",
            secondaries=3,
            q=0.10,
            inference=InferenceSpec(kind="asymptotic_mean"),
        )
        assert procedure.compiled_alpha == pytest.approx(0.10 / 3)

    def test_compliance_in_family_is_counted_without_the_caller(self):
        """The uptake cell joins the sequential secondary family when the
        declared compliance policy says so; the caller passes the policy,
        never a larger count."""
        from fractions import Fraction

        from increment.semantics.models import InferenceSpec
        from increment.semantics.sequential import SequentialCompliancePolicy

        policy = SequentialCompliancePolicy(alpha=Fraction(1, 20), family=True)
        without = ArmPlanningProcedure.standard(
            "mean",
            role="secondary",
            secondaries=2,
            q=0.10,
            inference=InferenceSpec(kind="asymptotic_mean"),
        )
        with_uptake = ArmPlanningProcedure.standard(
            "mean",
            role="secondary",
            secondaries=2,
            q=0.10,
            inference=InferenceSpec(kind="asymptotic_mean"),
            compliance=policy,
        )
        assert with_uptake.compiled_alpha == pytest.approx(0.10 / 3)
        assert without.compiled_alpha == pytest.approx(0.05)
        assert with_uptake.analysis.identification == "encouragement"

    def test_planned_secondary_alpha_matches_the_runtime_registration(self):
        """Planner and runtime compute the same per-cell alpha for the same
        declaration (two in-family secondaries plus an in-family uptake
        cell): build the runtime registration with auto_register_scalar_mean
        (via tests/test_mixed_family_auto.py's ``_bound`` helper) on a small
        encouragement frame and compare its secondary cells' registered
        alpha to standard()'s compiled_alpha exactly."""
        from fractions import Fraction

        from increment.frame import MetricSpec
        from increment.semantics.models import InferenceSpec
        from increment.semantics.sequential import SequentialCompliancePolicy
        from tests.test_mixed_family_auto import _bound

        specs = [
            MetricSpec(name="revenue", type="mean"),
            MetricSpec(name="orders", type="mean"),
            MetricSpec(name="aov", type="mean"),
        ]
        policy = SequentialCompliancePolicy(alpha=Fraction(1, 20), family=True)
        registration = _bound(
            specs=specs, secondaries=["orders", "aov"], compliance=policy, q=0.1
        ).inference.registration
        family_cells = [cell for cell in registration.roster if cell.family]
        assert {cell.metric for cell in family_cells} == {"orders", "aov", "uptake"}

        procedure = ArmPlanningProcedure.standard(
            "mean",
            role="secondary",
            secondaries=2,
            q=0.1,
            inference=InferenceSpec(kind="asymptotic_mean"),
            compliance=policy,
        )
        for cell in family_cells:
            assert float(cell.alpha) == pytest.approx(procedure.compiled_alpha, rel=1e-9)

    def test_compliance_without_sequential_inference_refuses_like_the_plan(self):
        from fractions import Fraction

        from increment.semantics.sequential import SequentialCompliancePolicy

        with pytest.raises(InvalidRequestError) as raised:
            ArmPlanningProcedure.standard(
                "mean", compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20), family=True)
            )
        assert raised.value.code == "sequential.registration.invalid"

    def test_sequential_secondary_alpha_is_clamped_to_the_declared_alpha(self):
        """Post-wave-0 regression (RulingConsistency): with secondaries=1
        the un-clamped q/m=0.10 exceeds the runtime's own registered
        per-cell alpha=0.05 (sequential_source.py's auto_register_scalar_mean/
        auto_register_bernoulli: min(procedure_alpha, q/family_n)) --
        sizing at the un-clamped value would undersize the plan."""
        from increment.semantics.models import InferenceSpec

        procedure = ArmPlanningProcedure.standard(
            "mean",
            role="secondary",
            secondaries=1,
            alpha=0.05,
            q=0.10,
            inference=InferenceSpec(kind="asymptotic_mean"),
        )
        assert procedure.compiled_alpha == pytest.approx(0.05)

    def test_sequential_secondary_refuses_multiple_arms(self):
        from increment.semantics.models import InferenceSpec

        with pytest.raises(InvalidRequestError) as raised:
            ArmPlanningProcedure.standard(
                "mean",
                role="secondary",
                secondaries=2,
                comparisons=2,
                q=0.10,
                inference=InferenceSpec(kind="asymptotic_mean"),
            )
        assert raised.value.code == "arm_planning.sequential_secondary_one_arm"

    def test_secondary_role_without_secondaries_count_refuses(self):
        with pytest.raises(InvalidRequestError) as raised:
            ArmPlanningProcedure.standard("mean", role="secondary")
        assert raised.value.code == "arm_planning.secondary_requires_count"


class TestGuardrailPlanning:
    """Guardrails are full alpha, one-sided, and never folded into a
    multiplicity-corrected family."""

    def test_preferred_direction_derives_the_alternative(self):
        procedure = ArmPlanningProcedure.standard(
            "mean", role="guardrail", preferred_direction="decrease"
        )
        assert procedure.decision.alternative == "less"
        assert procedure.decision.family.kind == "none"

    def test_comparisons_never_splits_a_guardrails_alpha(self):
        procedure = ArmPlanningProcedure.standard(
            "mean", role="guardrail", preferred_direction="increase", comparisons=4
        )
        assert procedure.compiled_alpha == pytest.approx(0.05)

    def test_contradicting_alternative_and_preferred_direction_refuses(self):
        with pytest.raises(InvalidRequestError) as raised:
            ArmPlanningProcedure.standard(
                "mean",
                role="guardrail",
                preferred_direction="increase",
                alternative="less",
            )
        assert raised.value.code == "arm_planning.guardrail_preferred_direction_conflict"

    def test_two_sided_guardrail_refuses(self):
        with pytest.raises(InvalidRequestError) as raised:
            ArmPlanningProcedure.standard("mean", role="guardrail")
        assert raised.value.code == "arm_planning.guardrail_requires_one_sided"

    def test_neutral_preferred_direction_refuses_with_its_own_code(self):
        """preferred_direction='neutral' already answers the 'did you pass
        preferred_direction=' question -- it needs its own explanation, not
        guardrail_requires_one_sided's generic 'pass preferred_direction='."""
        with pytest.raises(InvalidRequestError) as raised:
            ArmPlanningProcedure.standard("mean", role="guardrail", preferred_direction="neutral")
        assert raised.value.code == "arm_planning.guardrail_neutral_requires_direction"
        assert raised.value.context["preferred_direction"] == "neutral"
        assert raised.value.context["alternative"] == "two-sided"
