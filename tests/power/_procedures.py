from __future__ import annotations

from typing import Literal, cast

from increment.decision import FixedInference
from increment.estimation.arm_contract import (
    AnalysisAxes,
    ArmPlanningProcedure,
    FamilyPolicy,
    MetricCapabilities,
    PlanningFamilyExpansion,
    RelativeDecisionPolicy,
)
from increment.estimation.sequential import (
    AlwaysValid,
    GaussianScoreMixture,
)
from increment.semantics.assignment import ParallelAssignment
from increment.semantics.models import MethodSpec


def make_procedure(
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    null_lift: float = 0.0,
    family_size: int = 1,
    inference: FixedInference | AlwaysValid | GaussianScoreMixture | None = None,
    dependence: str = "iid",
    metric_type: str = "mean",
    segmented: bool = False,
    identification: str = "encouragement",
    population: str = "triggered",
    variance_adjustment: str = "factor_absorption",
    decision_method: str = "unadjusted",
) -> ArmPlanningProcedure:
    family = FamilyPolicy(
        kind="bonferroni" if family_size > 1 else "none",
        axes=("arm",) if family_size > 1 else (),
        nominal_alpha=alpha,
    )
    return ArmPlanningProcedure(
        assignment=ParallelAssignment(),
        analysis=AnalysisAxes(
            identification=cast(
                "Literal['randomized', 'encouragement', 'observational']", identification
            ),
            view="breakout" if segmented else "total",
            segmented=segmented,
            completed_windows_only=True,
            population=cast("Literal['assigned', 'triggered']", population),
            variance_adjustment=cast("Literal['none', 'factor_absorption']", variance_adjustment),
        ),
        dependence=cast("Literal['iid', 'cluster']", dependence),
        inference=FixedInference() if inference is None else inference,
        estimand="mean",
        metric=MetricCapabilities(
            metric_type=cast(
                "Literal['mean', 'conversion', 'retention', 'ratio', 'quantile']", metric_type
            ),
            value_scale="relative",
            winsorization="none",
            outcome_window="bounded",
            uptake_window="not_applicable",
        ),
        decision=RelativeDecisionPolicy(
            alternative=cast("Literal['two-sided', 'greater', 'less']", alternative),
            null_lift=null_lift,
            family=family,
        ),
        family_expansion=PlanningFamilyExpansion(family_size=family_size),
        decision_method=MethodSpec(
            name=decision_method,
            variance_reduction="cuped" if decision_method == "cuped" else "none",
        ),
        sensitivity_methods=(),
        prior_present=False,
    )
