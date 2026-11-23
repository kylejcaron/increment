"""Admission of a named conversion for a two-stage offer experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from increment._spillover_design import refuse_input

if TYPE_CHECKING:
    from increment.semantics.models import AnalysisPlan, ConversionMetric, Definitions, Experiment


def validate_spillover_plan(plan: AnalysisPlan, metric_name: str) -> None:
    """Require one primary conversion and one fixed-horizon error budget."""
    primaries = plan.primaries
    if len(primaries) != 1:
        refuse_input("spillover.design.invalid", "plan")
    entry = primaries[0]
    if isinstance(entry, str):
        name = entry
    else:
        name = entry.metric
        if (
            entry.decision_method is not None
            or entry.sensitivity_methods
            or entry.prior is not None
            or entry.margin is not None
            or entry.margin_abs is not None
        ):
            refuse_input("spillover.design.invalid", "plan")
    if (
        name != metric_name
        or plan.secondaries
        or plan.guardrails
        or plan.inference is not None
        or plan.compliance is not None
        or plan.view_multiplicity is not None
        or plan.alternative != "two-sided"
        or "q" in plan.model_fields_set
    ):
        refuse_input("spillover.design.invalid", "plan")


def selected_spillover_metric(definitions: Definitions, experiment: Experiment) -> ConversionMetric:
    """Resolve the registered conversion without substituting a one-stage analysis."""
    if experiment.design is None or experiment.design.mechanism != "spillover":
        refuse_input("spillover.design.required", "design")
    if (
        experiment.trigger is not None
        or experiment.cluster is not None
        or experiment.intervention_grain != "unit"
        or experiment.n_pre_periods
        or experiment.breakouts
        or experiment.factors
        or experiment.allocation is not None
        or experiment.allocation_scheme is not None
        or experiment.assignment_ledger is not None
    ):
        refuse_input("spillover.design.invalid", "experiment")
    names = experiment.metric_names
    if len(names) != 1:
        refuse_input("spillover.design.invalid", "plan")
    name = names[0]
    validate_spillover_plan(experiment.plan, name)
    metric = definitions.metric(name)
    if (
        metric is None
        or metric.type != "conversion"
        or metric.entity != experiment.unit
        or metric.window_days is None
        or metric.margin is not None
        or metric.margin_abs is not None
        or metric.rollout_cost is not None
    ):
        refuse_input("spillover.design.invalid", "metric")
    return metric
