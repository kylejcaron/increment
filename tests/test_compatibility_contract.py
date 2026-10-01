from __future__ import annotations

from typing import Literal

import pytest

from increment.compatibility import (
    ARM_COMPATIBILITY_REFUSALS,
    PowerDesign,
    Supported,
    UnitFloor,
    Unsupported,
    refuse_unsupported,
)
from increment.decision import FixedInference
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.arm_contract import (
    AbsoluteDecisionPolicy,
    AnalysisAxes,
    ArmCompatibilityRequest,
    ArmPlanningProcedure,
    FamilyPolicy,
    MethodCapability,
    MetricCapabilities,
    PlanningFamilyExpansion,
    RelativeDecisionPolicy,
    arm_runtime_support,
)
from increment.estimation.sequential import AlwaysValid
from increment.semantics.assignment import ParallelAssignment
from increment.semantics.models import MethodSpec
from tests.sequential_cases import registration


def _axes() -> AnalysisAxes:
    return AnalysisAxes(
        identification="randomized",
        view="total",
        segmented=False,
        completed_windows_only=False,
        population="assigned",
        variance_adjustment="none",
    )


def _metric() -> MetricCapabilities:
    return MetricCapabilities(
        metric_type="mean",
        value_scale="relative",
        winsorization="none",
        outcome_window="bounded",
        uptake_window="not_applicable",
    )


def _decision() -> RelativeDecisionPolicy:
    return RelativeDecisionPolicy(
        alternative="two-sided",
        null_lift=0.0,
        family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
    )


def test_sampling_floors_are_frozen_and_typed() -> None:
    with pytest.raises((AttributeError, TypeError)):
        UnitFloor(minimum_per_arm=2).minimum_per_arm = 3  # type: ignore[misc]  # ty: ignore[invalid-assignment]


def test_policy_discriminator_and_one_sided_alpha_gate() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        RelativeDecisionPolicy(
            alternative="greater",
            null_lift=0.0,
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.5),
        )
    assert exc_info.value.code == "decision.relative_decision.one_sided_nominal"
    assert (
        AbsoluteDecisionPolicy(
            alternative="greater",
            null_abs=0.0,
            family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
        ).scale
        == "absolute"
    )


@pytest.mark.parametrize(
    ("kind", "axes", "code"),
    [
        ("none", ("metric",), "decision.family.uncorrected_policy_empty"),
        ("bh", (), "decision.family.policy_least_one"),
        ("bh", ("metric", "metric"), "decision.family.axes_contain_duplicates"),
        ("bh", ("zeta", "alpha"), "decision.family.axes_contain_unknown"),
        ("bh", ("arm", "metric"), "decision.family.axes_use_canonical"),
    ],
)
def test_family_policy_rejects_invalid_axis_contract(kind, axes, code):
    with pytest.raises(InvalidRequestError) as exc_info:
        FamilyPolicy(kind=kind, axes=axes, nominal_alpha=0.05)
    assert exc_info.value.code == code
    if code == "decision.family.axes_contain_unknown":
        assert exc_info.value.context["unknown"] == ("alpha", "zeta")


def test_arm_requests_are_typed_and_planning_compiles_alpha() -> None:
    request = ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=_axes(),
        dependence="iid",
        inference=FixedInference(),
        estimand="itt",
        metric=_metric(),
        decision=_decision(),
        methods=(
            MethodCapability(
                role="decision",
                estimator="unadjusted",
                variance_reduction="none",
            ),
        ),
        prior_present=False,
    )
    assert request.sequential_request().inference is None
    planning_values = request.model_dump(exclude={"methods"})
    procedure = ArmPlanningProcedure(
        **planning_values,
        family_expansion=PlanningFamilyExpansion(family_size=1),
        decision_method=MethodSpec(name="unadjusted"),
        sensitivity_methods=(),
    )
    assert procedure.compiled_alpha == 0.05
    assert procedure.compiled_tail_alpha == 0.025


def test_bh_is_refused_for_scalar_planning() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        ArmPlanningProcedure(
            assignment=ParallelAssignment(),
            analysis=_axes(),
            dependence="iid",
            inference=FixedInference(),
            estimand="itt",
            metric=_metric(),
            decision=RelativeDecisionPolicy(
                alternative="two-sided",
                null_lift=0.0,
                family=FamilyPolicy(kind="bh", axes=("metric",), nominal_alpha=0.05),
            ),
            family_expansion=PlanningFamilyExpansion(family_size=2),
            decision_method=MethodSpec(name="unadjusted"),
            sensitivity_methods=(),
            prior_present=False,
        )
    assert exc_info.value.code == "decision.arm_planning.scalar_power_unsupported_family"


def test_unsupported_uses_common_coded_error_contract() -> None:
    support = Unsupported(refusal_code="arm.metric.unsupported")
    assert "arm.metric.unsupported" in ARM_COMPATIBILITY_REFUSALS
    with pytest.raises(CapabilityError) as raised:
        refuse_unsupported(support, metric="m")
    assert raised.value.code == "arm.metric.unsupported"
    assert raised.value.context["metric"] == "m"


def test_arm_compatibility_refusals_stays_narrowed_to_the_compatibility_catalog() -> None:
    """ARM_COMPATIBILITY_REFUSALS shares one _REFUSALS dict with every other
    compatibility.py code now; it must still expose exactly the arm/contrast
    catalog, not allocation.alpha.underflow or this module's own baseline-
    conversion codes."""
    assert all(code.startswith(("arm.", "contrast.")) for code in ARM_COMPATIBILITY_REFUSALS)
    assert "compatibility.denominator" not in ARM_COMPATIBILITY_REFUSALS
    assert "allocation.alpha.underflow" not in ARM_COMPATIBILITY_REFUSALS


def test_power_design_is_numeric_only() -> None:
    with pytest.raises(InvalidRequestError) as raised:
        PowerDesign(power=1.0)
    assert raised.value.code == "model.field.range"


def test_supported_and_metric_capabilities_are_immutable() -> None:
    support = Supported(
        assumptions=("iid",),
        floor=UnitFloor(minimum_per_arm=2),
        reference="unit_t",
    )
    with pytest.raises(AttributeError):
        support.assumptions += ("x",)  # type: ignore[misc]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cluster_floor_matches_small_cluster_runtime_admission() -> None:
    from dataclasses import asdict

    import pyarrow as pa

    from increment import Analysis, MetricSpec
    from increment.estimation.arm_contract import ARM_EVIDENCE_CONTRACT
    from increment.estimation.results import LiftEstimate

    request = ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=_axes(),
        dependence="cluster",
        inference=FixedInference(),
        estimand="itt",
        metric=_metric(),
        decision=_decision(),
        methods=(
            MethodCapability(role="decision", estimator="unadjusted", variance_reduction="none"),
        ),
        prior_present=False,
    )
    runtime = arm_runtime_support(request)
    assert isinstance(runtime, Supported)
    assert asdict(runtime.floor) == {"kind": "clusters", "minimum_per_arm": 2}
    planning = ArmPlanningProcedure(
        **request.model_dump(exclude={"methods"}),
        family_expansion=PlanningFamilyExpansion(family_size=1),
        decision_method=MethodSpec(name="unadjusted"),
        sensitivity_methods=(),
    )
    assert ARM_EVIDENCE_CONTRACT.planning_support(planning) == runtime
    analysis = Analysis.from_unit_summary(
        pa.table(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["C", "C", "T", "T"],
                "cluster": ["c1", "c2", "t1", "t2"],
                "outcome": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        unit="unit",
        group="arm",
        control="C",
        cluster="cluster",
        metrics=[MetricSpec(name="outcome", type="mean")],
    )
    (row,) = analysis.run()
    assert isinstance(row, LiftEstimate)
    assert row.lift is not None
    assert row.lift.value == pytest.approx(4 / 3)
    assert row.relative_confidence_set is not None


def test_runtime_and_planning_support_share_typed_refusal_shape() -> None:
    from increment.estimation.arm_contract import (
        ARM_EVIDENCE_CONTRACT,
        arm_runtime_support,
    )

    request = ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=_axes(),
        dependence="cluster",
        inference=FixedInference(),
        estimand="itt",
        metric=_metric(),
        decision=_decision(),
        methods=(MethodCapability(role="decision", estimator="cuped", variance_reduction="cuped"),),
        prior_present=False,
    )
    support = arm_runtime_support(request)
    assert isinstance(support, Unsupported)
    assert support.refusal_code == "arm.adjustment.cluster_cuped"
    assert ARM_EVIDENCE_CONTRACT.runtime_support(request) == support


def test_observational_adjustment_refusal_uses_public_spec() -> None:
    import pyarrow as pa

    from increment.estimation.adjust import (
        ADJUSTMENT_COMPATIBILITY_REFUSALS,
        estimate_ate,
    )
    from increment.estimation.inference import Normal
    from increment.frame import from_unit_summary
    from increment.semantics.design import AdjustmentSet, Observational

    assert (
        ADJUSTMENT_COMPATIBILITY_REFUSALS["arm.adjustment.cluster_prior"].code
        == "arm.adjustment.cluster_prior"
    )
    table = pa.table(
        {
            "u": [f"u{i}" for i in range(8)],
            "g": ["control"] * 4 + ["treatment"] * 4,
            "account": [f"a{i}" for i in range(8)],
            "z": [0.1 * i for i in range(8)],
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )
    source = from_unit_summary(
        table,
        unit="u",
        group="g",
        control="control",
        metrics={"y": "mean"},
        cluster="account",
    )
    design = Observational(control_group="control", adjustment=AdjustmentSet(covariates=("z",)))
    with pytest.raises(CapabilityError) as raised:
        estimate_ate(source, design, prior=Normal(mu=0.0, sigma=0.1))
    assert raised.value.code == "arm.adjustment.cluster_prior"
    assert raised.value.context["cluster"] == "account"


def test_contrast_family_policy_is_unsupported() -> None:
    from increment.estimation.arm_contract import (
        AbsoluteDecisionPolicy,
        ContrastCompatibilityRequest,
        ContrastEvidenceContract,
    )
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )

    request = ContrastCompatibilityRequest(
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(),
            window=SwitchbackWindow(
                washout_steps=0,
                observation_steps=1,
            ),
        ),
        inference=FixedInference(),
        estimand="retained_window_total_difference",
        metric=MetricCapabilities(
            metric_type="mean",
            value_scale="absolute",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=AbsoluteDecisionPolicy(
            alternative="two-sided",
            null_abs=0.0,
            family=FamilyPolicy(kind="bonferroni", axes=("metric",), nominal_alpha=0.05),
        ),
    )
    support = ContrastEvidenceContract().runtime_support(request)
    assert isinstance(support, Unsupported)
    assert support.refusal_code == "contrast.decision"


def test_static_quantile_and_sequential_refusal_precedence() -> None:
    from increment.estimation.arm_contract import arm_runtime_support

    request = ArmCompatibilityRequest(
        assignment=ParallelAssignment(),
        analysis=_axes(),
        dependence="iid",
        inference=AlwaysValid(registration=registration()),
        estimand="itt",
        metric=MetricCapabilities(
            metric_type="quantile",
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=_decision(),
        methods=(MethodCapability(role="decision", estimator="cuped", variance_reduction="cuped"),),
        prior_present=True,
    )
    support = arm_runtime_support(request)
    assert isinstance(support, Unsupported)
    assert support.refusal_code == "arm.adjustment.sequential_cuped"


_MetricType = Literal["mean", "conversion", "retention", "ratio", "quantile"]
_MEAN_FAMILY: tuple[_MetricType, ...] = ("mean", "conversion")


def _sequential_inferences() -> list[AlwaysValid]:
    return [AlwaysValid(registration=registration())]


class TestSequentialDeclaresTheBoundaryReference:
    """A sequential procedure's critical value comes from its own boundary, not
    from the fixed-horizon reference, and the declared ``reference`` must say so
    for every metric type -- including the ones whose fixed-horizon estimator is
    itself asymptotic."""

    @staticmethod
    def _support(metric_type: _MetricType, inference: FixedInference | AlwaysValid) -> Supported:
        support = arm_runtime_support(
            ArmCompatibilityRequest(
                assignment=ParallelAssignment(),
                analysis=_axes(),
                dependence="iid",
                inference=inference,
                estimand="itt",
                metric=MetricCapabilities(
                    metric_type=metric_type,
                    value_scale="relative",
                    winsorization="none",
                    outcome_window="bounded",
                    uptake_window="not_applicable",
                ),
                decision=_decision(),
                methods=(
                    MethodCapability(
                        role="decision", estimator="unadjusted", variance_reduction="none"
                    ),
                ),
                prior_present=False,
            )
        )
        assert isinstance(support, Supported)
        return support

    @pytest.mark.parametrize("metric_type", _MEAN_FAMILY)
    @pytest.mark.parametrize("inference", _sequential_inferences(), ids=["always_valid"])
    def test_a_mean_family_metric_under_sequential_declares_the_boundary(
        self, metric_type, inference
    ):
        support = self._support(metric_type, inference)
        assert support.reference == "sequential_boundary"

    @pytest.mark.parametrize("metric_type", _MEAN_FAMILY)
    def test_the_same_metric_declares_the_fixed_horizon_reference(self, metric_type):
        """The contrast with the sequential case is the point: at a fixed horizon
        the critical value comes straight from the fixed-horizon reference, with
        no boundary construction in between."""
        support = self._support(metric_type, FixedInference())
        assert support.reference == "fixed_horizon"


def test_switchback_study_envelope_refuses_an_allocation_with_two_treatment_arms():
    from increment import SwitchbackStudyEnvelope
    from increment.errors import InvalidRequestError
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    assignment = SwitchbackAssignment(
        sequence=IndependentBernoulliOrder(),
        window=SwitchbackWindow(washout_steps=1, observation_steps=2),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        SwitchbackStudyEnvelope(
            identification=Randomized(
                control_group="control",
                allocation={"control": 0.5, "a": 0.25, "b": 0.25},
            ),
            assignment=assignment,
        )
    assert exc_info.value.code == "facade.study.switchback_study.identification_exactly_one"


def test_plan_family_decision_refuses_a_runtime_effect_that_hides_a_limitation():
    from increment._plan_compatibility import PlanFamilyDecision
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as exc_info:
        PlanFamilyDecision(
            runtime_effect="not_applicable",
            participation="excluded",
            warning=None,
        )
    assert exc_info.value.code == "facade.plan_compatibility.plan_family.runtime_effect_limited"
