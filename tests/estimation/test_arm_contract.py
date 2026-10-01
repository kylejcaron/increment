"""Coded-refusal regression tests for ``increment.estimation.arm_contract``."""

from __future__ import annotations

import pytest

from increment.decision import FixedInference
from increment.errors import InvalidRequestError
from increment.estimation.arm_contract import (
    AbsoluteDecisionPolicy,
    AnalysisAxes,
    ArmPlanningProcedure,
    FamilyPolicy,
    MetricCapabilities,
    PlanningFamilyExpansion,
    RelativeDecisionPolicy,
)
from increment.semantics.assignment import ParallelAssignment
from increment.semantics.models import MethodSpec


def _analysis_axes(**overrides: object) -> AnalysisAxes:
    fields: dict[str, object] = {
        "identification": "randomized",
        "view": "total",
        "segmented": False,
        "completed_windows_only": True,
        "population": "assigned",
        "variance_adjustment": "none",
    }
    fields.update(overrides)
    return AnalysisAxes(**fields)


def _metric_capabilities(**overrides: object) -> MetricCapabilities:
    fields: dict[str, object] = {
        "metric_type": "mean",
        "value_scale": "absolute",
        "winsorization": "none",
        "outcome_window": "bounded",
        "uptake_window": "not_applicable",
    }
    fields.update(overrides)
    return MetricCapabilities(**fields)


def _arm_planning_procedure(*, family: FamilyPolicy, family_size: int) -> ArmPlanningProcedure:
    return ArmPlanningProcedure(
        assignment=ParallelAssignment(),
        analysis=_analysis_axes(),
        dependence="iid",
        inference=FixedInference(),
        estimand="ate",
        metric=_metric_capabilities(),
        decision=RelativeDecisionPolicy(alternative="two-sided", null_lift=0.0, family=family),
        family_expansion=PlanningFamilyExpansion(family_size=family_size),
        decision_method=MethodSpec(name="unadjusted"),
        sensitivity_methods=(),
        prior_present=False,
    )


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "decision.family.axes_contain_unknown",
            lambda: FamilyPolicy(kind="bh", axes=("bogus",), nominal_alpha=0.05),
        ),
        (
            "decision.family.axes_contain_duplicates",
            lambda: FamilyPolicy(kind="bh", axes=("metric", "metric"), nominal_alpha=0.05),
        ),
        (
            "decision.family.axes_use_canonical",
            lambda: FamilyPolicy(kind="bh", axes=("arm", "metric"), nominal_alpha=0.05),
        ),
        (
            "decision.family.uncorrected_policy_empty",
            lambda: FamilyPolicy(kind="none", axes=("metric",), nominal_alpha=0.05),
        ),
        (
            "decision.family.policy_least_one",
            lambda: FamilyPolicy(kind="bh", axes=(), nominal_alpha=0.05),
        ),
        (
            "decision.relative_decision.one_sided_nominal",
            lambda: RelativeDecisionPolicy(
                alternative="greater",
                null_lift=0.0,
                family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.6),
            ),
        ),
        (
            "decision.relative_decision.one_sided_nominal",
            lambda: AbsoluteDecisionPolicy(
                alternative="greater",
                null_abs=0.0,
                family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.6),
            ),
        ),
        (
            "decision.arm_planning.scalar_power_unsupported_family",
            lambda: _arm_planning_procedure(
                family=FamilyPolicy(kind="bh", axes=("metric",), nominal_alpha=0.05),
                family_size=1,
            ),
        ),
        (
            "decision.arm_planning.uncorrected_family_size",
            lambda: _arm_planning_procedure(
                family=FamilyPolicy(kind="none", axes=(), nominal_alpha=0.05),
                family_size=2,
            ),
        ),
    ],
)
def test_arm_contract_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
