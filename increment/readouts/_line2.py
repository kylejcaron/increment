"""Shared-capture primitives: scope planning and owned contexts over one binding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import UNSET
from increment._evidence_types import BreakoutIdentity, EvidenceScope
from increment._exploratory import exploratory_sibling, select_exploratory
from increment._source_admission import _arm_experiment
from increment._source_types import declared_evidence, is_custom_source
from increment.kernel.context import ExecutionMode
from increment.semantics.models import Metric

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._binding import SourceBinding
    from increment._day_axis import DayAxisReadouts
    from increment._evidence_types import SequentialCaptureRequest
    from increment.readouts._run_acquisition import PreparedRunAcquisition


@dataclass(frozen=True, slots=True)
class ScopePlanner:
    """View-owned validation consulted while one shared capture is planned."""

    contrast: Callable[[str, Mapping[str, Any]], None]
    whole_window: Callable[
        [
            SourceBinding,
            Sequence[Metric],
            Sequence[Metric],
            Mapping[str, Any],
            SequentialCaptureRequest | None,
        ],
        Sequence[PreparedRunAcquisition],
    ]
    breakout: Callable[..., tuple[tuple[Metric, ...], tuple[ResolvedMetricConfig, ...]]]
    day_axis: Callable[..., DayAxisReadouts]


def _compliance_only(binding: SourceBinding, estimands, dimension=None):
    from increment.semantics.design import Encouragement

    return (
        isinstance(binding.design, Encouragement)
        and estimands is not None
        and set(estimands) == {"compliance"}
        and dimension is None
    )


def _validate_source_populations(source: Any, populations: Sequence[str], operation: str) -> None:
    from increment.readouts._evidence import validate_populations, validates_populations

    if validates_populations(source):
        validate_populations(source, populations, operation=operation)


def owned_execution_context(
    binding: SourceBinding,
    scope: EvidenceScope,
    *,
    coherence: ExecutionMode = "captured",
):
    from uuid import uuid4

    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._execution_context import execution_context_scope
    from increment.readouts._execution_context import (
        owned_execution_context as _owned_execution_context,
    )
    from increment.readouts._run_acquisition import RunAcquisitionOwner

    provider = RunAcquisitionOwner.provider_for(binding.source)

    return _owned_execution_context(
        lambda: EvidenceExecutionContext(
            provider,
            scope,
            run_id=uuid4().hex,
            coherence=coherence,
            shared_capture=True,
        ),
        execution_context_scope=execution_context_scope,
    )


def _selected_breakouts(binding: SourceBinding, dimension: str | None = None):
    _arm_experiment(binding)
    return tuple(
        BreakoutIdentity(item.property, item.source)
        for item in binding.declared_breakouts
        if dimension is None or item.property == dimension
    )


def _contrast_operation_scope(
    binding: SourceBinding, operation: str, options: Mapping[str, Any], planner: ScopePlanner
) -> EvidenceScope:
    from increment._analysis_config import select_metrics
    from increment._evidence_dispatch import ContrastHandler, ContrastReadoutRequest

    source = binding.source
    select_exploratory(binding, options.get("exploratory_metrics"), caller=operation)
    planner.contrast(operation, options)
    selected = tuple(
        select_metrics(source.context.metrics, options.get("metrics"), caller=operation)
    )
    request = ContrastReadoutRequest(
        metrics=selected,
        procedures={metric.name: source.context.procedures[metric.name] for metric in selected},
    )
    ContrastHandler().preflight(source, request)
    return EvidenceScope(
        metrics=selected,
        population="assigned",
        populations=("assigned",),
        grain="total",
        snapshot=True,
        execution=True,
    )


def _whole_window_operation_scope(
    binding: SourceBinding,
    selected: Sequence[Metric],
    added: Sequence[Metric],
    options: Mapping[str, Any],
    planner: ScopePlanner,
    capture: SequentialCaptureRequest | None,
) -> EvidenceScope:
    from increment.estimation.sequential import SEQUENTIAL_POLICIES
    from increment.readouts._run_acquisition import RunAcquisitionOwner

    prepared = planner.whole_window(binding, selected, added, options, capture)
    additional_scopes = ()
    if isinstance(binding.plan.inference, SEQUENTIAL_POLICIES) and (
        capture is not None or any(item.scope.metrics for item in prepared)
    ):
        from increment.readouts._line3 import sequential_scope

        additional_scopes = (sequential_scope(binding, include_breakouts=False),)
    return RunAcquisitionOwner.combined_scope(
        binding.source,
        prepared,
        additional_scopes=additional_scopes,
    )


def _day_axis_scope_dependencies(binding: SourceBinding, request):
    """Retain completion evidence separately from requested uptake moments."""
    if request.grain != "asof":
        return None, ()
    estimands = request.estimands if request.caller == "run_asof_lift" else ()
    uptake_facts = ()
    if request.completed_windows_only:
        uptake = getattr(binding.design, "uptake", None)
        if uptake is not None:
            uptake_facts = (uptake.fact,)
    return estimands, uptake_facts


def operation_scope(
    binding: SourceBinding,
    operation: str,
    options: Mapping[str, Any],
    *,
    planner: ScopePlanner,
    capture: SequentialCaptureRequest | None = None,
) -> EvidenceScope:
    """Validate a selected operation and declare only the evidence it may read."""
    from increment._analysis_config import overlay_configs, select_metrics
    from increment._evidence_scopes import capture_requirements
    from increment._evidence_types import DayAxisRequest
    from increment._exploratory import select_day_axis_metrics
    from increment.readouts._design_scope import _config_snapshot

    source = binding.source
    if capture is not None:
        from increment._evidence_refusals import require_evidence
        from increment.sequential_source import validate_capture_request

        require_evidence(source, request=capture)
        validate_capture_request(
            capture,
            dynamic=declared_evidence(source).sequential_capture == "asof",
        )
    from increment.decision import ContrastContext

    if isinstance(source.context, ContrastContext):
        return _contrast_operation_scope(binding, operation, options, planner)
    added = select_exploratory(binding, options.get("exploratory_metrics"), caller=operation)
    selected = (
        select_day_axis_metrics(binding, options.get("metrics"), caller=operation)
        if operation in {"run_daily", "run_daily_lift", "run_asof", "run_asof_lift"}
        else select_metrics(tuple(binding.metrics), options.get("metrics"), caller=operation)
    )
    if operation == "run":
        return _whole_window_operation_scope(binding, selected, added, options, planner, capture)
    candidates = [(binding, selected)]
    if added:
        candidates.append((exploratory_sibling(binding, added, caller=operation), added))
    requests = []
    population = options.get("population") or "assigned"
    experiment = _arm_experiment(binding)
    populations = (
        ("assigned", "triggered")
        if options.get("population") is None and getattr(experiment, "trigger", None) is not None
        else (population,)
    )
    for candidate, metrics in candidates:
        overrides = {
            key: options.get(key, UNSET)
            for key in ("decision_method", "sensitivity_methods", "prior")
        }
        configs, scope_estimands, scope_uptake_facts = (), None, ()
        if operation == "run_breakout" and metrics:
            configs = overlay_configs(
                metrics,
                candidate.source.context.configs,
                methods=None,
                prior=None,
                decision_method=overrides["decision_method"],
                sensitivity_methods=overrides["sensitivity_methods"],
                prior_override=overrides["prior"],
            )
        if operation == "run_breakout":
            metrics, configs = planner.breakout(
                candidate,
                metrics,
                population=population,
                populations=populations,
                overrides=overrides,
            )
            grain = "total"
            breakouts = _selected_breakouts(candidate)
        else:
            grain = "daily" if operation in {"run_daily", "run_daily_lift"} else "asof"
            req = DayAxisRequest(
                caller=cast(
                    Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"], operation
                ),
                grain=grain,
                metrics=tuple(metrics),
                dimension=options.get("dimension"),
                estimands=tuple(options["estimands"])
                if options.get("estimands") is not None
                else None,
                population=population,
                experiment=_arm_experiment(candidate),
                completed_windows_only=options.get("completed_windows_only", False),
            )
            scope_estimands, scope_uptake_facts = _day_axis_scope_dependencies(candidate, req)
            reader = planner.day_axis(candidate)
            _validate_source_populations(candidate.source, populations, operation)
            reader.validate_request(req, lift=operation.endswith("_lift"))
            if operation.endswith("_lift"):
                run_standalone_day_axis_lift(
                    candidate,
                    req,
                    day_axis=partial(planner.day_axis, candidate),
                    validate_only=True,
                    **overrides,
                    coherence=options.get("coherence", "captured"),
                )
                configs = (
                    ()
                    if _compliance_only(candidate, req.estimands, req.dimension)
                    else reader.resolve_lift_options(req.metrics, **overrides).configs
                )
            breakouts = _selected_breakouts(candidate, req.dimension) if req.dimension else ()
        scope_breakouts = (
            breakouts
            if metrics
            or capture is not None
            or (operation != "run_breakout" and options.get("dimension") is not None)
            else ()
        )
        for config in configs:
            _config_snapshot(config, candidate.design)
        compliance_only = operation != "run_breakout" and _compliance_only(
            candidate, options.get("estimands"), options.get("dimension")
        )
        requests.append(
            EvidenceScope(
                metrics=() if compliance_only else tuple(metrics),
                horizon_metrics=() if compliance_only else tuple(candidate.source.context.metrics),
                configs=tuple(configs),
                estimands=scope_estimands,
                uptake_facts=scope_uptake_facts,
                population=population,
                populations=populations,
                grain=(
                    "total"
                    if operation in {"run_asof", "run_asof_lift"}
                    and getattr(candidate.plan.inference, "registration", None) is not None
                    else grain
                ),
                breakouts=scope_breakouts,
                snapshot=True,
                execution=True,
                include_breakouts=bool(scope_breakouts),
            )
        )
    from increment.estimation.sequential import SEQUENTIAL_POLICIES

    if isinstance(binding.plan.inference, SEQUENTIAL_POLICIES) and (
        capture is not None or any(request.metrics for request in requests)
    ):
        from increment.readouts._line3 import sequential_scope

        requests.append(
            sequential_scope(binding, include_breakouts=any(item.breakouts for item in requests))
        )
    return capture_requirements(source, requests)


def _triggered_day_axis_capture_supported(binding: SourceBinding, req) -> bool:
    if req.population != "triggered" or getattr(_arm_experiment(binding), "trigger", None) is None:
        return False
    from increment.readouts._evidence import ARTIFACT_ROUTE, NATIVE_ROUTE, day_axis_route

    return day_axis_route(binding.source) in (ARTIFACT_ROUTE, NATIVE_ROUTE)


def run_standalone_day_axis_values(
    binding: SourceBinding,
    req,
    execution_context=None,
    *,
    day_axis: Callable[..., DayAxisReadouts],
    coherence: ExecutionMode = "captured",
):
    """Read day-axis values, owning one capture when no run context is active."""
    if not req.metrics and execution_context is None:
        return day_axis().values(req)

    if execution_context is not None:
        return day_axis(execution_context).values(req)
    source = binding.source
    triggered_capture = _triggered_day_axis_capture_supported(binding, req)
    if is_custom_source(source) or (
        not triggered_capture
        and (req.population == "triggered" or req.grain not in declared_evidence(source).grains)
    ):
        return day_axis().values(req)

    from increment._evidence_scopes import capture_requirements

    scope = capture_requirements(
        source,
        (
            EvidenceScope(
                metrics=req.metrics,
                configs=(),
                population=req.population,
                grain=req.grain,
                dimension=req.dimension,
                breakouts=(
                    _selected_breakouts(binding, req.dimension) if req.dimension is not None else ()
                ),
                snapshot=coherence == "captured",
                execution=True,
                include_breakouts=req.dimension is not None,
            ),
        ),
    )
    with owned_execution_context(binding, scope, coherence=coherence) as context:
        return day_axis(context).values(req)


def run_standalone_day_axis_lift(
    binding: SourceBinding,
    req,
    *,
    day_axis: Callable[..., DayAxisReadouts],
    decision_method,
    sensitivity_methods,
    prior,
    validate_only: bool = False,
    coherence: ExecutionMode = "captured",
):
    """Resolve lift options before opening the owned evidence context."""

    source = binding.source
    plan = binding.plan
    design = binding.design
    triggered_capture = _triggered_day_axis_capture_supported(binding, req)
    unsupported_capture = is_custom_source(source) or (
        not triggered_capture
        and (req.population == "triggered" or req.grain not in declared_evidence(source).grains)
    )
    if validate_only and unsupported_capture:
        return None
    if unsupported_capture:
        return day_axis().lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
    compliance_only = _compliance_only(binding, req.estimands, req.dimension)
    metrics = () if compliance_only else req.metrics
    if req.caller == "run_daily_lift":
        from increment.readouts._common import _sequential_inference
        from increment.readouts._daily import _validate_daily_inference

        _validate_daily_inference(
            _sequential_inference(plan),
            method=req.caller,
        )

    from increment._analysis_config import overlay_configs, resolve_configs, validate_scalar_route

    base = source.context.configs
    base_names = {config.metric.name for config in base}
    declared = tuple(metric for metric in metrics if metric.name in base_names)
    undeclared = tuple(metric for metric in metrics if metric.name not in base_names)
    by_name = {
        config.metric.name: config
        for config in overlay_configs(
            declared,
            base,
            methods=None,
            prior=None,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior_override=prior,
        )
    }
    by_name.update(
        {
            config.metric.name: config
            for config in resolve_configs(
                undeclared,
                None,
                None,
                methods=None,
                prior=None,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior_override=prior,
            )
        }
    )
    configs = tuple(by_name[metric.name] for metric in metrics)
    from increment._analysis_config import effective_methods
    from increment._evidence_refusals import check_method_covariates

    for config in configs:
        check_method_covariates(
            source,
            config.metric.name,
            effective_methods(config, design=design),
        )
    scalar_evidence = declared_evidence(source).scalar
    validate_scalar_route(
        configs,
        route=f"Analysis.{req.caller}",
        representation=(
            "registered"
            if getattr(plan.inference, "registration", None) is not None
            else "implicit"
        ),
        registered_plan=plan,
        design=design,
        covariate_moments_available=(
            scalar_evidence is None
            or scalar_evidence.covariate_grains is None
            or req.grain in scalar_evidence.covariate_grains
        ),
    )
    if req.caller == "run_daily_lift":
        from increment._readout_request import ReadoutRequest, validate_request
        from increment.semantics.design import Encouragement

        compliance_only = (
            isinstance(design, Encouragement)
            and tuple(req.estimands or ()) == ("compliance",)
            and req.dimension is None
        )
        validation_metrics = () if compliance_only else tuple(req.metrics)
        validation_configs = () if compliance_only else configs
        validate_request(
            ReadoutRequest.from_source(
                source,
                metrics=validation_metrics,
                configs=validation_configs,
                view="daily",
                grain="daily",
                by=() if req.dimension is None else (req.dimension,),
                dimension=req.dimension,
                estimands=req.estimands,
                population=req.population,
                daily_lift=True,
            )
        )

    if req.caller == "run_asof_lift":
        from increment._readout_request import ReadoutRequest
        from increment.estimation.encouragement import validate_readout_encouragement
        from increment.sequential_source import validate_sequential_request

        request = ReadoutRequest.from_source(
            source,
            metrics=list(metrics),
            configs=configs,
            view="asof",
            grain="asof",
            estimands=req.estimands,
            completion_policy=req.completed_windows_only,
        )
        validate_readout_encouragement(request)
        validate_sequential_request(request)
    if validate_only:
        return None
    from increment.estimation.encouragement import ESTIMANDS
    from increment.semantics.design import Encouragement

    is_encouragement = isinstance(design, Encouragement)
    requested_estimands = (
        tuple(req.estimands)
        if req.estimands is not None
        else ESTIMANDS
        if is_encouragement
        else ("itt",)
    )
    compliance_only = (
        is_encouragement and requested_estimands == ("compliance",) and req.dimension is None
    )
    compliance_rows_requested = (
        is_encouragement and req.dimension is None and "compliance" in requested_estimands
    )
    if not req.metrics and not compliance_rows_requested:
        return day_axis().lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
    scope_metrics = () if compliance_only else req.metrics
    from increment._evidence_scopes import capture_requirements

    _, scope_uptake_facts = _day_axis_scope_dependencies(binding, req)

    scope = capture_requirements(
        source,
        (
            EvidenceScope(
                metrics=scope_metrics,
                horizon_metrics=() if compliance_only else None,
                configs=configs,
                population=req.population,
                grain=req.grain,
                dimension=req.dimension,
                estimands=req.estimands,
                uptake_facts=scope_uptake_facts,
                breakouts=(
                    _selected_breakouts(binding, req.dimension) if req.dimension is not None else ()
                ),
                snapshot=coherence == "captured",
                execution=True,
                include_breakouts=req.dimension is not None,
            ),
        ),
    )
    with owned_execution_context(binding, scope, coherence=coherence) as context:
        return day_axis(context).lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
