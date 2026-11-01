"""Execute an experiment's declared views over one source binding."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Literal, cast

from increment import readouts
from increment._analysis_config import UNSET, _Unset, overlay_configs, select_metrics
from increment._binding import _run_identity_hashes
from increment._breakout_readouts import plan_breakout_request, run_breakout
from increment._day_axis import (
    _TRIGGER_UNSUPPORTED,
    day_axis_readouts,
    require_arm_state,
    run_asof,
    run_asof_lift,
    run_daily,
    run_daily_lift,
)
from increment._evidence_dispatch import ContrastEvidenceOperation, ContrastReadoutRequest
from increment._evidence_types import SequentialCaptureRequest
from increment._exploratory import exploratory_sibling, select_exploratory
from increment._runnability import (
    _SCOPE_OPERATIONS,
    operation_options,
    records_breakout_sources,
    require_breakout_sources,
    scope_operation,
)
from increment._source_admission import (
    _arm_experiment,
    _require_analysis_operation,
    _require_arm_binding,
)
from increment._source_types import is_custom_source
from increment.decision import ContrastAnalysisState
from increment.errors import refuse
from increment.estimation.engine import Method
from increment.estimation.readout_types import ScopedReadout
from increment.readouts._evidence import retained_sequential_capture
from increment.readouts._execution_context import (
    current_execution_context,
    execution_context_scope,
)
from increment.readouts._line2 import ScopePlanner, operation_scope, owned_execution_context
from increment.readouts._whole_window import (
    WholeWindowExploratoryRequest,
    WholeWindowFamilyRequest,
    WholeWindowReadouts,
    WholeWindowRequest,
    validate_contrast_request,
)

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._binding import SourceBinding
    from increment._literals import ValueScale
    from increment.breakout.estimates import LiftEstimates
    from increment.decision import ContrastContext, ContrastSource
    from increment.estimation.contrast_results import ContrastResults
    from increment.estimation.inference import Prior
    from increment.kernel.context import EvidenceExecutionContext
    from increment.kernel.contracts import Scope
    from increment.kernel.experiment import Experiment
    from increment.readouts._run_acquisition import PreparedRunAcquisition
    from increment.semantics.models import Metric
    from increment.sources import _BuiltinMomentSource


def _cuped_recommendation_admitted(
    source: _BuiltinMomentSource, config: ResolvedMetricConfig
) -> bool:
    """Check the proposed CUPED method against the shared arm-support contract."""
    from increment.estimation.arm_contract import (
        MetricCapabilities,
        Supported,
        arm_cuped_advisory_support,
    )
    from increment.estimation.engine import is_open_ended
    from increment.sequential_source import adjustment_kind

    context = source.context
    metric = config.metric
    inference = context.plan.inference
    registration = getattr(inference, "registration", None)
    adjustment = adjustment_kind(registration, metric.name) if registration is not None else None
    capabilities = MetricCapabilities(
        metric_type=metric.type if metric.type not in ("total", "active") else "mean",
        value_scale="relative",
        winsorization=(
            "percentile"
            if getattr(getattr(metric, "winsorization", None), "has_percentile", False)
            else "fixed"
            if getattr(metric, "winsorization", None) is not None
            else "none"
        ),
        outcome_window="unbounded" if is_open_ended(metric) else "bounded",
        uptake_window="not_applicable",
    )
    support = arm_cuped_advisory_support(
        dependence="cluster" if context.cluster is not None else "iid",
        metric=capabilities,
        inference=inference,
        adjustment=adjustment,
        prior_present=config.prior is not None,
    )
    return isinstance(support, Supported)


def readout_source(
    binding: SourceBinding,
    *,
    population: Literal["assigned", "triggered"] = "assigned",
) -> _BuiltinMomentSource:
    """Validate a requested population; the evidence request selects it."""
    source = binding.source
    if population == "triggered":
        if is_custom_source(source):
            from increment._evidence_refusals import require_population

            require_population(source, population)
        else:
            _require_analysis_operation(
                source,
                "triggered_source",
                message=(
                    "A triggered readout requires a source retaining the raw event stream "
                    "and its declared trigger."
                ),
            )
    return source


def _validate_trigger_fires_in_every_arm(
    binding: SourceBinding, source: object | None = None
) -> None:
    from increment.readouts._evidence import validate_trigger_fires_in_every_arm

    _require_arm_binding(binding, "experiment")
    validate_trigger_fires_in_every_arm(
        cast(Any, binding.source if source is None else source),
        binding.experiment.trigger,
    )


def whole_window_readouts(binding: SourceBinding) -> WholeWindowReadouts:
    """The whole-window reader over this binding's source, plan and design."""
    require_arm_state(binding, "run")
    return WholeWindowReadouts(
        src=binding.source,
        plan=binding.plan,
        design=binding.design,
        experiment=_arm_experiment(binding),
        readout_source=partial(readout_source, binding),
        validate_trigger=partial(_validate_trigger_fires_in_every_arm, binding),
        arm_moments=readouts.arm_moments,
        execution_context_scope=execution_context_scope,
        registered_run=readouts.run,
    )


def _run_contrast(
    binding: SourceBinding,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
    metrics: Sequence[str | Metric] | None,
    estimands: Sequence[str] | None,
    value_scale: Mapping[str, ValueScale] | None,
    *,
    execution_context: EvidenceExecutionContext | None = None,
) -> ContrastResults:
    validate_contrast_request(
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
        estimands=estimands,
        value_scale=value_scale,
    )
    context = cast("ContrastContext", binding.context)
    selected = select_metrics(cast("Sequence[Metric]", context.metrics), metrics, caller="run")
    request = ContrastReadoutRequest(
        metrics=tuple(selected),
        procedures={
            metric.name: context.procedures[metric.name]
            for metric in selected
            if metric.name in context.procedures
        },
    )
    return ContrastEvidenceOperation().run(
        cast("ContrastSource", binding.source),
        request,
        execution_context=execution_context,
    )


# Metrics the executing run has already advised about an unused CUPED covariate.
_ADVISED_COVARIATES: ContextVar[set[str] | None] = ContextVar("_ADVISED_COVARIATES", default=None)


def _warn_run_advisories(
    binding: SourceBinding,
    selected: Sequence[Metric],
    added: Sequence[Metric],
    *,
    estimands: Sequence[str] | None,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
) -> None:
    from increment._frame_validation import _warn

    state = binding.state
    assert not isinstance(state, ContrastAnalysisState)
    source = state.source

    # A compliance-only encouragement read uses no outcome configuration.
    if (
        estimands is not None
        and not {"itt", "late"}.intersection(estimands)
        and getattr(source.context.design, "mechanism", None) == "encouragement"
    ):
        return
    selected_names = {metric.name for metric in selected}
    declared_configs = {
        config.metric.name: config
        for config in source.context.configs
        if config.metric.name in selected_names
    }
    default_decision = (
        Method(name="iptw")
        if getattr(source.context.design, "mechanism", None) == "observational"
        else Method(name="unadjusted")
    )
    conflicts: dict[str, list[tuple[str, object]]] = {
        "decision_method": [],
        "sensitivity_methods": [],
        "prior": [],
    }
    for name, config in declared_configs.items():
        if (
            decision_method is not UNSET
            and not config.decision_defaulted
            and config.decision_method != default_decision
            and config.decision_method != decision_method
        ):
            conflicts["decision_method"].append((name, config.decision_method))
        if (
            sensitivity_methods is not UNSET
            and config.sensitivity_methods
            and tuple(config.sensitivity_methods)
            != tuple(cast(Sequence[Method], sensitivity_methods))
        ):
            conflicts["sensitivity_methods"].append((name, tuple(config.sensitivity_methods)))
        if (
            prior is not UNSET
            and config.prior is not None
            and not config.prior_is_global
            and config.prior != prior
        ):
            conflicts["prior"].append((name, config.prior))

    overrides = {
        "decision_method": decision_method,
        "sensitivity_methods": (
            tuple(cast(Sequence[Method], sensitivity_methods))
            if sensitivity_methods is not UNSET
            else sensitivity_methods
        ),
        "prior": prior,
    }
    for setting, declarations in conflicts.items():
        if declarations:
            _warn(
                "analysis.run.override_replaces_declared",
                setting=setting,
                metrics=tuple(name for name, _ in declarations),
                declared=tuple(declarations),
                override=overrides[setting],
            )

    config_sources: list[tuple[Any, _BuiltinMomentSource]] = []
    if selected:
        config_sources.extend(
            (config, state.source)
            for config in overlay_configs(
                selected,
                state.source.context.configs,
                methods=None,
                prior=None,
                prior_override=prior,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
            )
        )
    if added:
        exploratory_source = cast(
            "_BuiltinMomentSource", exploratory_sibling(binding, added, caller="run").source
        )
        config_sources.extend(
            (config, exploratory_source)
            for config in overlay_configs(
                added,
                exploratory_source.context.configs,
                methods=None,
                prior=None,
                prior_override=prior,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
            )
        )
    advised = _ADVISED_COVARIATES.get()
    unused: list[str] = []
    for config, config_source in config_sources:
        design = config_source.context.design
        if (
            not config.covariate_available
            or not config.decision_defaulted
            or config.methods_explicitly_empty
            or getattr(design, "mechanism", None) == "observational"
            or any(method.variance_reduction == "cuped" for method in config.sensitivity_methods)
            or not _cuped_recommendation_admitted(config_source, config)
            or (advised is not None and config.metric.name in advised)
        ):
            continue
        unused.append(config.metric.name)
    if advised is not None:
        advised.update(unused)
    if unused:
        _warn(
            "analysis.cuped.covariate_unused",
            metric_names=tuple(unused),
            route=(
                "Request Method(name='cuped', variance_reduction='cuped') "
                "as decision_method or a sensitivity method in run()."
            ),
        )


def run_whole_window(
    binding: SourceBinding,
    *,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    metrics: Sequence[str | Metric] | None = None,
    estimands: Sequence[str] | None = None,
    value_scale: Mapping[str, ValueScale] | None = None,
    exploratory_metrics: Sequence[str] | None = None,
    population: Literal["assigned", "triggered"] | None = None,
    coherence: Literal["captured", "live"] = "captured",
    store: Any | None = None,
    _execution_context: EvidenceExecutionContext | None = None,
) -> LiftEstimates | ContrastResults:
    """Run the whole-window readout: one estimate per metric, method and non-control arm."""
    if _execution_context is None:
        _execution_context = current_execution_context()
    from increment.breakout.estimates import LiftEstimates

    state = binding.state
    if isinstance(state, ContrastAnalysisState) and population == "triggered":
        refuse(
            _TRIGGER_UNSUPPORTED,
            method="run",
            experiment="<unknown>",
            trigger=None,
            route="switchback",
            supported_sources=("from_definitions", "from_unit_day_artifact"),
        )
    added = select_exploratory(binding, exploratory_metrics, caller="run")
    runtime_request: dict[str, object] = {}
    if prior is not UNSET:
        runtime_request["prior"] = prior
    if estimands is not None:
        runtime_request["estimands"] = tuple(estimands)
    if value_scale is not None:
        runtime_request["value_scale"] = value_scale
    if isinstance(state, ContrastAnalysisState):
        rows = _run_contrast(
            binding,
            decision_method,
            sensitivity_methods,
            prior,
            metrics,
            estimands,
            value_scale,
            execution_context=_execution_context,
        )
        rows._default_backend = binding.input_backend
        if store is not None:
            _persist(
                binding, store, rows, decision_method, sensitivity_methods, (), runtime_request
            )
        return rows
    selected = select_metrics(cast("Sequence[Metric]", binding.metrics), metrics, caller="run")

    def request(chosen: Sequence[Metric]) -> WholeWindowRequest:
        return WholeWindowRequest(
            metrics=tuple(chosen),
            estimands=tuple(estimands) if estimands is not None else None,
            value_scale=value_scale,
            population=population,
            coherence=coherence,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )

    reader = whole_window_readouts(binding)
    if added:
        exploratory = exploratory_sibling(binding, added, caller="run")
        rows = reader.run_family(
            WholeWindowFamilyRequest(
                declared=request(selected),
                exploratory=WholeWindowExploratoryRequest(
                    request=request(added),
                    reader=whole_window_readouts(exploratory),
                ),
            ),
            execution_context=_execution_context,
        )
    else:
        rows = reader.run(request(selected), execution_context=_execution_context)
    if not isinstance(rows, LiftEstimates):
        rows = LiftEstimates(rows)
    _warn_run_advisories(
        binding,
        selected,
        added,
        estimands=estimands,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
    )
    if not added and _execution_context is not None and rows.metadata is not None:
        metadata = rows.metadata
        rows._metadata = type(metadata).model_validate(
            {
                **metadata.__dict__,
                "coherence": (
                    None
                    if "provider" in _execution_context.effective_modes.values()
                    else _execution_context.coherence
                ),
                "effective_modes": dict(_execution_context.effective_modes),
            }
        )
    rows._default_backend = binding.input_backend
    if store is not None:
        _persist(binding, store, rows, decision_method, sensitivity_methods, added, runtime_request)
    return rows


def _persist(
    binding: SourceBinding,
    store: Any,
    rows: Any,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    added: Sequence[Metric],
    runtime_request: Mapping[str, object],
) -> None:
    from uuid import uuid4

    from increment._run import _persistence_identity

    store.persist_results(
        rows,
        binding.experiment_name,
        uuid4().hex,
        _persistence_identity(
            binding,
            rows,
            decision_method,
            sensitivity_methods,
            added,
            runtime_request=runtime_request,
        ),
    )


def _contrast_request(operation: str, options: Mapping[str, Any]) -> None:
    if options.get("population") == "triggered":
        refuse(
            _TRIGGER_UNSUPPORTED,
            method=operation,
            experiment="<unknown>",
            trigger=None,
            route="switchback",
            supported_sources=("from_definitions", "from_unit_day_artifact"),
        )
    validate_contrast_request(
        decision_method=options.get("decision_method", UNSET),
        sensitivity_methods=options.get("sensitivity_methods", UNSET),
        prior=options.get("prior", UNSET),
        estimands=options.get("estimands"),
        value_scale=options.get("value_scale"),
    )


def _whole_window_family(
    binding: SourceBinding,
    selected: Sequence[Metric],
    added: Sequence[Metric],
    options: Mapping[str, Any],
    capture: SequentialCaptureRequest | None,
) -> tuple[PreparedRunAcquisition, ...]:
    def request(metrics: Sequence[Metric]) -> WholeWindowRequest:
        return WholeWindowRequest(
            metrics=tuple(metrics),
            estimands=tuple(options["estimands"]) if options.get("estimands") is not None else None,
            population=options.get("population"),
            value_scale=options.get("value_scale"),
            coherence=options.get("coherence", "captured"),
            decision_method=options.get("decision_method", UNSET),
            sensitivity_methods=options.get("sensitivity_methods", UNSET),
            prior=options.get("prior", UNSET),
        )

    exploratory = None
    if added:
        exploratory = WholeWindowExploratoryRequest(
            request=request(added),
            reader=whole_window_readouts(exploratory_sibling(binding, added, caller="run")),
        )
    family = WholeWindowFamilyRequest(
        declared=request(selected),
        exploratory=exploratory,
        capture_requested=capture is not None,
    )
    return whole_window_readouts(binding).prepare_family(family)


_PLANNER = ScopePlanner(
    contrast=_contrast_request,
    whole_window=_whole_window_family,
    breakout=plan_breakout_request,
    day_axis=day_axis_readouts,
)

_VIEWS: Mapping[str, Callable[..., Any]] = {
    "run": run_whole_window,
    "run_daily_lift": run_daily_lift,
    "run_daily": run_daily,
    "run_asof_lift": run_asof_lift,
    "run_asof": run_asof,
    "run_breakout": run_breakout,
}


def operation_parameters() -> dict[str, Any]:
    """Record the keyword contracts of the readout views a binding executes."""
    declarations = {}
    for operation in dict.fromkeys(name for names in _SCOPE_OPERATIONS.values() for name in names):
        parameters = tuple(signature(_VIEWS[operation]).parameters.values())[1:]
        keywords = tuple(
            parameter
            for parameter in parameters
            if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        )
        declarations[operation] = {
            "accepted": (*tuple(parameter.name for parameter in keywords), "capture"),
            "required": tuple(
                parameter.name for parameter in keywords if parameter.default is Parameter.empty
            ),
            "variadic": any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters),
        }
    return declarations


def _borrowed(
    execution_context: EvidenceExecutionContext,
    source: object,
    requests: Sequence[Any],
) -> AbstractContextManager[object]:
    for request in requests:
        execution_context.validate_request(
            source, request.metrics, request.configs, population=request.population
        )
        for population in request.populations:
            execution_context.validate_population(population)
        if request.breakouts:
            execution_context.validate_breakouts(request.breakouts)
    return execution_context_scope(execution_context)


def selected_binding(binding: SourceBinding, experiment: Experiment, scope: Scope) -> SourceBinding:
    """Narrow a binding to the breakout identities its scope selects, before any read.

    Identities come from the same declaration-based expansion ``Experiment.compile()``
    reports. A recorded definition matches by ``(property, source)``. A source without
    recorded definitions matches an unqualified selection by property and refuses a
    source qualifier, which none of its breakouts names. An unknown property refuses
    with the source's declared dimensions; an undeclared source of a known property
    refuses as an undeclared breakout.
    """
    if scope.view not in {"breakout", "daily", "asof"} or (
        scope.view != "breakout" and not scope.segment
    ):
        return binding
    from increment._evidence_refusals import require_evidence
    from increment.kernel.contracts import declared_breakouts
    from increment.semantics.models import Breakout

    identities = declared_breakouts(experiment.source_refs, scope, experiment.analysis_context)
    require_breakout_sources(experiment, scope)
    by_source = records_breakout_sources(experiment)
    declared = binding.declared_breakouts
    if not {identity.property for identity in identities} <= {item.property for item in declared}:
        if scope.view != "breakout":
            # A day-axis view refuses an unavailable grain before an unavailable dimension.
            return binding
        require_evidence(binding.source, by=tuple(identity.property for identity in identities))
    selected = tuple(
        dict.fromkeys(
            item
            for identity in identities
            for item in (
                tuple(
                    candidate
                    for candidate in declared
                    if candidate.property == identity.property
                    and (not by_source or candidate.source == identity.source)
                )
                or (Breakout(property=identity.property, source=identity.source),)
            )
        )
    )
    scope_declared = (
        declared
        if scope.view == "breakout"
        else tuple(item for item in declared if item.property == scope.segment[0].property)
    )
    if set(selected) == set(scope_declared):
        return binding
    return binding.with_breakouts(selected)


def execute_views(
    binding: SourceBinding,
    experiment: Experiment,
    scopes: Sequence[Scope],
    *,
    options: Mapping[str, object],
    capture: SequentialCaptureRequest | None = None,
    execution_context: EvidenceExecutionContext | None = None,
) -> tuple[ScopedReadout, ...]:
    """Plan every selected scope before any read, then run each under one capture.

    Views run in declaration order. Without a supplied context, more than one
    view shares one owned capture; a supplied context is borrowed only after
    every selected request validates against it. A sequential capture is taken
    inside that context before any view reads. The run's ``coherence`` applies to
    every view: a view that takes none of its own reads under a shared capture
    opened with it.
    """
    coherence = cast(Literal["captured", "live"], options.get("coherence", "captured"))
    planned = [
        (
            scope,
            scope_operation(experiment, scope),
            operation_options(experiment, scope, options),
            selected_binding(binding, experiment, scope),
        )
        for scope in scopes
    ]
    if len(planned) == 1 and (coherence == "captured" or "coherence" in planned[0][2]):
        scope, operation, kwargs, scope_binding = planned[0]
        if capture is not None:
            # Validate the view's request before the capture reads and retains source state.
            operation_scope(scope_binding, operation, kwargs, planner=_PLANNER, capture=capture)
        ambient = (
            nullcontext()
            if execution_context is None
            else execution_context_scope(execution_context)
        )
        with ambient:
            if capture is not None:
                retained_sequential_capture(binding.source, capture, current_execution_context())
            return (ScopedReadout(_VIEWS[operation](scope_binding, **kwargs), scope, kwargs),)
    requests = [
        operation_scope(scope_binding, operation, kwargs, planner=_PLANNER, capture=capture)
        for _, operation, kwargs, scope_binding in planned
    ]
    if execution_context is not None and "coherence" in options:
        execution_context.validate_requested_coherence(coherence)
    from increment._evidence_scopes import capture_requirements

    scope_request = capture_requirements(binding.source, requests)
    needs_evidence = bool(
        capture is not None
        or scope_request.metrics
        or scope_request.configs
        or scope_request.uptake_facts
        or scope_request.assignment_requests
        or scope_request.breakouts
        or scope_request.include_breakouts
    )
    if execution_context is not None:
        acquisition = _borrowed(execution_context, binding.source, requests)
    elif needs_evidence:
        acquisition = owned_execution_context(binding, scope_request, coherence=coherence)
    else:
        acquisition = nullcontext()
    # One run reading several views advises each metric's unused covariate once.
    advised = _ADVISED_COVARIATES.set(set())
    try:
        with acquisition:
            if capture is not None:
                retained_sequential_capture(binding.source, capture, current_execution_context())
            scoped = []
            for scope, operation, kwargs, scope_binding in planned:
                single = replace(experiment, scopes=(scope,), operations={scope.hash(): operation})
                scope_hash, _ = _run_identity_hashes(single, kwargs)
                scoped.append(
                    ScopedReadout(
                        _VIEWS[operation](scope_binding, **kwargs), scope, dict(kwargs), scope_hash
                    )
                )
    finally:
        _ADVISED_COVARIATES.reset(advised)
    return tuple(scoped)
