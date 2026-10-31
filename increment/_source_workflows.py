"""Source-bound inspection, refresh, export, and summary operations."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyarrow as pa

    from increment.decision import CompiledDecisionPlan
    from increment.kernel.experiment import Experiment as KernelExperiment
    from increment.kernel.resolver import Resolver
    from increment.semantics.models import Experiment, Metric

from increment._binding import SourceBinding, binding_for
from increment._source_operations import ExportMomentsOperation, MaterializeOperation
from increment._source_types import declared_evidence
from increment.decision import DefinitionsArmAnalysisState
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, refuse
from increment.errors import refuse as _refuse
from increment.readouts._whole_window import _require_analysis_operation

_EXPLORATORY_SOURCE_LIMITED = RefusalSpec(
    "facade.analysis.exploratory_metrics_source_limited",
    CapabilityError,
    template="{method}: exploratory_metrics adds metrics from the saved definitions, which this source does not carry -- use Analysis.from_definitions.",
)
_PLAN_MISSING = RefusalSpec(
    "store.run.legacy_analysis_required",
    InvalidRequestError,
    template="Switchback results do not have a registered family decision plan; use {route}",
)


def _arm_binding(experiment: KernelExperiment, resolver: Resolver, method: str) -> SourceBinding:
    binding = binding_for(experiment, resolver)
    if binding.state.family != "arm_moments":
        from increment._evidence_refusals import CONTRAST_UNAVAILABLE

        _refuse(CONTRAST_UNAVAILABLE, method=method)
    return binding


def _materialize_binding(binding: SourceBinding) -> None:
    source = _require_analysis_operation(
        binding.source,
        "materialize",
        MaterializeOperation,
        message=(
            "materialize() needs a native Analysis.from_definitions instance -- "
            "a frame/warehouse/moments-backed analysis has nothing to materialize."
        ),
    )
    if binding.store != "none" and binding.metrics:
        source.materialize()


def materialize(experiment: KernelExperiment, resolver: Resolver) -> None:
    """Refresh the source-owned materialized tables from current warehouse data."""
    _materialize_binding(_arm_binding(experiment, resolver, "materialize"))


def _export_moments_binding(binding: SourceBinding, path: str | Path) -> None:
    source = _require_analysis_operation(
        binding.source,
        "export_moments",
        ExportMomentsOperation,
        message="export() requires a native, frame, or artifact source with the export_moments operation.",
    )
    source.export_moments(path)


def export_moments(experiment: KernelExperiment, resolver: Resolver, path: str | Path) -> None:
    """Write the source's portable per-metric moments to a parquet file."""
    _export_moments_binding(binding_for(experiment, resolver), path)


def _experiment_definition_binding(binding: SourceBinding) -> Experiment:
    return binding.experiment


def experiment_definition(experiment: KernelExperiment, resolver: Resolver) -> Experiment:
    """Return the semantic experiment declaration carried by the source."""
    return _experiment_definition_binding(binding_for(experiment, resolver))


def _metrics_binding(binding: SourceBinding) -> list[Metric]:
    return binding.metrics


def metrics(experiment: KernelExperiment, resolver: Resolver) -> list[Metric]:
    """Return the source's declared metrics and guardrails."""
    return _metrics_binding(binding_for(experiment, resolver))


def _compiled_decision_plan_binding(binding: SourceBinding) -> CompiledDecisionPlan:
    plan = binding.plan
    if plan is None:
        refuse(_PLAN_MISSING, route="Analysis.run()")
    return plan


def compiled_decision_plan(
    experiment: KernelExperiment, resolver: Resolver
) -> CompiledDecisionPlan:
    """Return the immutable compiled decision plan for this experiment."""
    return _compiled_decision_plan_binding(binding_for(experiment, resolver))


def _available_metrics_binding(binding: SourceBinding) -> list[Metric]:
    state = binding.state
    if not isinstance(state, DefinitionsArmAnalysisState):
        refuse(_EXPLORATORY_SOURCE_LIMITED, method="available_metrics")
    declared = {*state.experiment.metric_names, *(metric.name for metric in binding.metrics)}
    return [
        metric
        for metric in state.definitions.metrics
        if metric.name not in declared
        and metric.type not in ("total", "active")
        and getattr(metric, "entity", None) == state.experiment.unit
    ]


def available_metrics(experiment: KernelExperiment, resolver: Resolver) -> list[Metric]:
    """Return eligible saved metrics not declared by the experiment, in catalog order."""
    return _available_metrics_binding(binding_for(experiment, resolver))


def breakout_summaries(
    experiment: KernelExperiment,
    resolver: Resolver,
    *,
    metrics: Sequence[str | Metric] | None = None,
) -> dict[str, dict[str, pa.Table]]:
    """Return per-breakout moment tables for the selected declared metrics."""
    return _breakout_summaries_binding(
        _arm_binding(experiment, resolver, "breakout_summaries"), metrics
    )


def _breakout_summaries_binding(
    binding: SourceBinding, metrics: Sequence[str | Metric] | None
) -> dict[str, dict[str, pa.Table]]:
    from increment._analysis_config import select_metrics
    from increment.breakout.estimates import reject_quantile_metrics, reject_retention_metrics
    from increment.query.native_contract import NativeViewSource

    source = binding.source
    if not isinstance(source, NativeViewSource):
        from increment._evidence_refusals import refuse_breakout

        offered = tuple(sorted(declared_evidence(source).breakouts))
        source_experiment = getattr(binding.state, "experiment", None)
        if source_experiment is None:
            source_experiment = getattr(source, "artifact_experiment", None)
        requested = (
            tuple(item.property for item in source_experiment.breakouts)
            if source_experiment is not None
            else ()
        )
        refuse_breakout(
            source,
            requested,
            offered,
            route=(
                "breakout_summaries() needs a native Analysis.from_definitions or "
                "Analysis.from_unit_day_artifact instance with breakout evidence"
            ),
        )
    native = _require_analysis_operation(
        source,
        "breakout_summaries",
        NativeViewSource,
        message=(
            "breakout_summaries() needs a native Analysis.from_definitions or "
            "Analysis.from_unit_day_artifact instance with breakout evidence."
        ),
    )
    effective = select_metrics(
        binding.metrics,
        metrics,
        caller="breakout_summaries",
        require_declared_definitions=True,
    )
    semantic_experiment = binding.experiment
    if not semantic_experiment.breakouts or not binding.metrics:
        return {}
    reject_quantile_metrics(
        effective, "breakout_summaries()", reason="quantiles do not decompose over segment moments"
    )
    reject_retention_metrics(effective, "breakout_summaries", view="cohort")
    triggered = semantic_experiment.trigger is not None
    populations = ("assigned", "triggered") if triggered else ("assigned",)
    native.validate_populations(populations, operation="breakout_summaries")
    if triggered:
        import pyarrow as pa

    results: dict[str, dict[str, pa.Table]] = {}
    for population in populations:
        summaries = native.breakout_summaries(metrics=effective, population=population)
        for key, tables in summaries.items():
            if not triggered:
                results[key] = tables
                continue
            results[f"{key}:{population}"] = {
                name: table.append_column(
                    "analysis_population", pa.array([population] * table.num_rows)
                )
                for name, table in tables.items()
            }
    return results


def factor_summaries(experiment: KernelExperiment, resolver: Resolver) -> dict[str, pa.Table]:
    """Return per-factor-level and arm moment tables for declared factors."""
    return _factor_summaries_binding(_arm_binding(experiment, resolver, "factor_summaries"))


def _factor_summaries_binding(binding: SourceBinding) -> dict[str, pa.Table]:
    from increment.breakout.estimates import reject_quantile_metrics
    from increment.query.native_contract import NativeViewSource

    native = _require_analysis_operation(
        binding.source,
        "factor_summaries",
        NativeViewSource,
        message=(
            "factor_summaries() needs a native Analysis.from_definitions or "
            "Analysis.from_unit_day_artifact instance with factor evidence."
        ),
    )
    semantic_experiment = binding.experiment
    if not semantic_experiment.factors or not binding.metrics:
        return {}
    reject_quantile_metrics(
        binding.metrics,
        "factor_summaries()",
        reason="quantiles do not decompose over factor moments",
    )
    triggered = semantic_experiment.trigger is not None
    populations = ("assigned", "triggered") if triggered else ("assigned",)
    native.validate_populations(populations, operation="factor_summaries")
    if triggered:
        import pyarrow as pa

    results: dict[str, pa.Table] = {}
    for population in populations:
        summaries = native.factor_summaries(metrics=binding.metrics, population=population)
        for key, table in summaries.items():
            result_key = f"{key}:{population}" if triggered else key
            results[result_key] = (
                table.append_column("analysis_population", pa.array([population] * table.num_rows))
                if triggered
                else table
            )
    return results
