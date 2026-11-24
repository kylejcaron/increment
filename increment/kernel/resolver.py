from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from increment._source_types import declared_day_axis_metric_objects_supported
from increment._spillover_design import SPILLOVER_ESTIMATOR
from increment.errors import CodedError
from increment.kernel.contracts import View, _refuse, declared_breakouts
from increment.kernel.experiment import Experiment, SourceReference

if TYPE_CHECKING:
    from increment._evidence_types import SequentialCaptureRequest
    from increment.estimation.readout_types import ScopedReadout
    from increment.kernel.context import EvidenceExecutionContext
    from increment.semantics.models import Metric


@dataclass(frozen=True, slots=True, eq=False)
class Runnable:
    static_refusals: tuple[CodedError, ...] = ()
    deferred_checks: tuple[str, ...] = ()
    unknown: tuple[str, ...] = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Runnable):
            return NotImplemented
        return (
            self.deferred_checks == other.deferred_checks
            and self.unknown == other.unknown
            and tuple((error.code, error.context) for error in self.static_refusals)
            == tuple((error.code, error.context) for error in other.static_refusals)
        )


class _ReaderExecution(Protocol):
    """The private surface of a builder's reader that the kernel validates, reads and runs."""

    def _validate_experiment(self, experiment: Experiment) -> None: ...

    def _resolve(
        self, source_refs: Sequence[SourceReference], ctx: object | None = None
    ) -> object: ...

    def _verify(
        self, source_refs: Sequence[SourceReference], provider: object
    ) -> list[CodedError]: ...

    def _run_experiment(
        self,
        experiment: Experiment,
        provider: object,
        *,
        options: Mapping[str, object] | None = None,
        capture: SequentialCaptureRequest | None = None,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> tuple[ScopedReadout, ...]: ...


def _resolve_verified(
    exp: Experiment, reader: _ReaderExecution, *, ctx: object | None = None
) -> object:
    """Validate declarations and verify the resolved provider before use."""
    reader._validate_experiment(exp)
    provider = reader._resolve(exp.source_refs, ctx)
    failures = reader._verify(exp.source_refs, provider)
    if failures:
        raise failures[0]
    return provider


def _scope_compatibility(
    exp: Experiment, scope: View, exploratory_metric: Metric | None
) -> tuple[CodedError, ...]:
    from increment._runnability import operation_options, scope_refusals

    refusals = scope_refusals(exp, scope)
    if refusals or exploratory_metric is None:
        return refusals
    try:
        operation_options(exp, scope, {"exploratory_metrics": [exploratory_metric.name]})
    except CodedError as refusal:
        return (refusal,)
    return ()


def resolve(estimator_id: str, exp: Experiment, metric: object) -> list[CodedError]:
    """Return source-declaration incompatibilities without acquiring evidence."""
    if exp.design.get("kind") == "spillover":
        return list(_spillover_runnable(exp, metric, estimator_id).static_refusals)
    return _resolve(estimator_id, exp, metric, eligibility_unknown=False)


def _resolve(
    estimator_id: str,
    exp: Experiment,
    metric: object,
    *,
    eligibility_unknown: bool,
    exploratory_metric: Metric | None = None,
) -> list[CodedError]:
    from increment._evidence_refusals import (
        require_any_evidence_route,
        require_evidence_declaration,
    )
    from increment._evidence_types import (
        EvidenceScope,
        SequentialCaptureRequest,
        UnitRowsRequest,
    )
    from increment.declarations import Method, estimator_refs, thaw
    from increment.estimators.registry import routes as evidence_routes

    metric_name = getattr(metric, "name", metric)
    metric_spec: Any = metric if not isinstance(metric, str) else exp.metrics.get(metric, metric)
    refusals: list[CodedError] = []

    def append_unique(refusal: CodedError) -> None:
        if not any(
            prior.code == refusal.code and prior.context == refusal.context for prior in refusals
        ):
            refusals.append(refusal)

    from increment._runnability import operation_options, scope_operation

    for scope in exp.views:
        for refusal in _scope_compatibility(exp, scope, exploratory_metric):
            append_unique(refusal)
    if refusals:
        return refusals
    estimator = estimator_refs().get(estimator_id)
    if exp.analysis_context.get("family") == "contrast":
        from pydantic import TypeAdapter

        from increment._analysis_config import select_metrics
        from increment._runnability import contrast_request_refusals, typed_request_options
        from increment.semantics.models import Metric

        options = typed_request_options(exp.request_options)
        declared_metrics = tuple(
            TypeAdapter(Metric).validate_python(metric) for metric in exp.metrics.values()
        )
        try:
            select_metrics(declared_metrics, options.get("metrics"), caller="run")
        except CodedError as refusal:
            append_unique(refusal)
        for refusal in contrast_request_refusals(exp.request_options):
            append_unique(refusal)
        if estimator_id != "switchback.contrast":
            append_unique(
                CodedError(
                    "Switchback contrast readouts do not accept arm role overrides",
                    code="readout.contrast.override",
                    context={"argument": "decision_method"},
                )
            )
        return refusals
    # Recorded declarations are read as recorded; a corrupt one fails at the read.
    recorded: Mapping[str, Any] = exp.analysis_context
    readout_context_known = bool(recorded) and (
        not eligibility_unknown
        or {"configs", "plan", "study_id", "cluster", "trigger_name", "source_name"}
        <= recorded.keys()
    )
    for source in exp.source_refs:
        capabilities = source.capabilities
        if capabilities is None:
            continue
        constructor = recorded.get("source_name") or source.name
        for scope in exp.views:
            inference = recorded.get("plan", {}).get("inference") or {}
            checkpoint_projection = (
                scope_operation(exp, scope) == "run_asof_lift"
                and inference.get("registration") is not None
            )
            # An empty breakout segment reads every declared breakout, so validation sees the
            # identities execution expands it to.
            breakouts = declared_breakouts(exp.source_refs, scope, exp.analysis_context)
            if readout_context_known:
                from increment._runnability import readout_refusals

                for refusal in readout_refusals(
                    estimator_id=estimator_id,
                    design=exp.design,
                    declaration=exp.analysis_context,
                    capabilities=capabilities,
                    scope=scope.model_copy(update={"segment": breakouts}),
                    operation=scope_operation(exp, scope),
                    metric_name=str(metric_name),
                    options=operation_options(exp, scope),
                    live=exp.request_options.get("coherence") == "live",
                    exploratory_metric=exploratory_metric,
                ):
                    append_unique(refusal)
            request = EvidenceScope(
                metrics=() if eligibility_unknown else (metric_spec,),
                population=scope.population,
                grain=None if checkpoint_projection else scope.grain,
                breakouts=breakouts,
            )
            try:
                require_evidence_declaration(capabilities, constructor=constructor, request=request)
            except CodedError as refusal:
                append_unique(refusal)
            if checkpoint_projection:
                for refusal in require_any_evidence_route(
                    capabilities,
                    ((SequentialCaptureRequest(finalized=True),),),
                    constructor=constructor,
                ):
                    append_unique(refusal)
                continue

            if estimator is None:
                continue
            methods = (
                method
                for config in (() if eligibility_unknown else recorded.get("configs", ()))
                if config["metric"]["name"] == metric_name
                for method in (config["decision_method"], *config["sensitivity_methods"])
            )
            method_payload = next(
                (
                    method
                    for method in methods
                    if (
                        method["variance_reduction"] == (estimator.selector or "none")
                        if estimator_id.startswith("randomized.")
                        else method["name"] == estimator.selector
                    )
                ),
                None,
            )
            method = None if method_payload is None else Method.model_validate(thaw(method_payload))
            routes = list(
                evidence_routes(
                    estimator,
                    {
                        "metric": metric_spec,
                        "method": method,
                        "control": exp.design.get("control_group", ""),
                        "arm_order": (),
                        "population": scope.population,
                        "design_covariates": cast(
                            Mapping[str, Any], exp.design.get("adjustment", {})
                        ).get("covariates", ()),
                    },
                )
            )

            winsorization = (
                metric_spec.get("winsorization")
                if isinstance(metric_spec, Mapping)
                else getattr(metric_spec, "winsorization", None)
            )
            percentile_bounds = (
                (
                    winsorization.get("lower_percentile"),
                    winsorization.get("upper_percentile"),
                )
                if isinstance(winsorization, Mapping)
                else (
                    getattr(winsorization, "lower_percentile", None),
                    getattr(winsorization, "upper_percentile", None),
                )
            )
            if any(value is not None for value in percentile_bounds):
                raw_outcomes = UnitRowsRequest(
                    metric=metric_spec, outcome_stage="raw", population=scope.population
                )
                routes = [route + (raw_outcomes,) for route in routes] or [(raw_outcomes,)]
            for refusal in require_any_evidence_route(
                capabilities, tuple(routes), constructor=constructor
            ):
                append_unique(refusal)
    return refusals


def _metric_roster(exp: Experiment) -> Mapping[str, object]:
    """The binding's metric declarations, independent of the saved selection."""
    recorded = exp.analysis_context
    configs = recorded.get("configs")
    context = recorded.get("context")
    source = recorded.get("source_declaration")
    if isinstance(configs, (tuple, list)):
        metrics = [
            config.get("metric") if isinstance(config, Mapping) else None for config in configs
        ]
    elif recorded.get("family") == "contrast" and isinstance(context, Mapping):
        metrics = context.get("metrics")
    elif isinstance(source, Mapping):
        metrics = source.get("metrics")
    else:
        metrics = None
    if not isinstance(metrics, (tuple, list)):
        return exp.metrics
    roster = {}
    for metric in metrics:
        if not isinstance(metric, Mapping) or not isinstance(name := metric.get("name"), str):
            return exp.metrics
        roster[name] = metric
    return roster


def _possible_exploratory_source(source: SourceReference) -> bool:
    """Whether source declarations leave room for adding catalog metrics."""
    capabilities = source.capabilities
    if capabilities is not None:
        return capabilities.kind == "definitions" or "exploratory_source" in capabilities.operations
    return source.uri.partition("://")[0] not in {
        "artifact",
        "breakout",
        "moments",
        "switchback",
        "unit_panel",
        "unit_summary",
    }


def _catalog_metric_eligibility(
    recorded: Mapping[str, Any], source: Mapping[str, Any], metric: Mapping[str, Any]
) -> Metric | Literal[False] | None:
    from pydantic import TypeAdapter, ValidationError

    from increment._analysis_config import is_exploratory_metric
    from increment.semantics.models import AnalysisPlan, Metric

    experiment = source.get("experiment") or recorded.get("experiment")
    if not isinstance(experiment, Mapping) or not isinstance(unit := experiment.get("unit"), str):
        return None
    try:
        plan = AnalysisPlan.model_validate(experiment.get("plan"))
        candidate = TypeAdapter(Metric).validate_python(metric)
    except (ValidationError, CodedError):
        return None
    return (
        candidate
        if is_exploratory_metric(
            candidate.name,
            candidate.type,
            getattr(candidate, "entity", None),
            declared_names=plan.role_names(),
            unit=unit,
        )
        else False
    )


def _exploratory_metric(exp: Experiment, name: str) -> Metric | Literal[False] | None:
    """Find an eligible metric; missing declarations do not prove absence."""
    recorded: Mapping[str, Any] = exp.analysis_context
    if recorded.get("family") == "contrast" or not any(
        _possible_exploratory_source(source) for source in exp.source_refs
    ):
        return False
    source = recorded.get("source_declaration")
    if not isinstance(source, Mapping) or not isinstance(
        definitions := source.get("definitions"), Mapping
    ):
        return None
    catalog = definitions.get("metrics")
    if not isinstance(catalog, (tuple, list)):
        return None
    match = None
    for metric in catalog:
        if not isinstance(metric, Mapping) or not isinstance(
            metric_name := metric.get("name"), str
        ):
            return None
        if metric_name == name:
            match = metric
    return False if match is None else _catalog_metric_eligibility(recorded, source, match)


def _runnable_metric(
    exp: Experiment, name: str
) -> tuple[object, Literal["known", "unknown", "add"]]:
    """Resolve a name, keeping catalog additions separate from declared metrics."""
    from pydantic import TypeAdapter

    from increment._analysis_config import refuse_unknown_metrics
    from increment.semantics.models import Metric

    if name in exp.metrics:
        return name, "known"
    roster = _metric_roster(exp)
    if name in roster:
        return TypeAdapter(Metric).validate_python(roster[name]), "known"
    saved = exp.request_options.get("metrics")
    if (
        isinstance(saved, (tuple, list))
        and any(scope.view in {"daily", "asof"} for scope in exp.views)
        and any(
            source.capabilities is not None
            and declared_day_axis_metric_objects_supported(source.capabilities)
            for source in exp.source_refs
        )
    ):
        for metric in saved:
            if isinstance(metric, Mapping) and metric.get("name") == name:
                return TypeAdapter(Metric).validate_python(metric), "known"
    exploratory = _exploratory_metric(exp, name)
    if exploratory is False:
        refuse_unknown_metrics((name,), roster.keys(), caller="runnable")
    if exploratory is None:
        return name, "unknown"
    from increment._runnability import require_runnable_exploratory_metric

    require_runnable_exploratory_metric(exp.analysis_context, name)
    return exploratory, "add"


def _spillover_runnable(exp: Experiment, metric: object, estimator_id: str) -> Runnable:
    from increment._runnability import require_runnable_spillover_metric

    try:
        require_runnable_spillover_metric(exp, metric)
        if estimator_id != SPILLOVER_ESTIMATOR:
            _refuse("kernel.request.invalid", field="use the registered two-stage spillover method")
    except CodedError as refusal:
        return Runnable(static_refusals=(refusal,))
    return Runnable(
        deferred_checks=("recorded assignments and offer counts", "complete conversion outcomes")
    )


def runnable(exp: Experiment, metric: object) -> dict[str, Runnable]:
    """Conservative offline feasibility: absent declarations remain unknown."""
    from increment.declarations import estimator_refs

    if not isinstance(exp, Experiment):
        _refuse("kernel.request.invalid", field="runnable requires an Experiment")
    if exp.design.get("kind") == "spillover":
        return {SPILLOVER_ESTIMATOR: _spillover_runnable(exp, metric, SPILLOVER_ESTIMATOR)}
    exploratory_metric = None
    eligibility_unknown = False
    if isinstance(metric, str):
        metric, selection = _runnable_metric(exp, metric)
        eligibility_unknown = selection == "unknown"
        if selection == "add":
            exploratory_metric = cast("Metric", metric)
    contrast = exp.analysis_context.get("family") == "contrast"
    references = estimator_refs()
    missing = tuple(
        f"{source.name}.capabilities"
        for source in exp.source_refs
        if source.capabilities is None and not contrast
    )
    if eligibility_unknown:
        missing += ("exploratory metric eligibility",)
    return {
        estimator_id: Runnable(
            static_refusals=tuple(
                _resolve(
                    estimator_id,
                    exp,
                    metric,
                    eligibility_unknown=eligibility_unknown,
                    exploratory_metric=exploratory_metric,
                )
            ),
            deferred_checks=(
                ("sample floors", "usable denominators", "data-dependent estimator checks")
                if not missing
                else ()
            ),
            unknown=missing,
        )
        for estimator_id in ((*references, "switchback.contrast") if contrast else references)
    }


def _select_views(exp: Experiment, views: Sequence[View] | None) -> Experiment:
    """Narrow to the selected declared views, keeping declaration order."""
    if views is None:
        return exp
    hashes = tuple(scope.hash() for scope in views)
    declared = {scope.hash() for scope in exp.views}
    if not hashes or len(set(hashes)) != len(hashes) or not set(hashes) <= declared:
        raise CodedError(
            "The bound analysis cannot execute these experiment declarations",
            code="kernel.run.execution_unavailable",
            context={"experiment_id": exp.experiment_id, "field": "views"},
        )
    return replace(
        exp,
        views=tuple(view for view in exp.views if view.hash() in hashes),
        operations={key: value for key, value in exp.operations.items() if key in hashes},
    )


def run(
    exp: Experiment,
    reader: _ReaderExecution,
    *,
    ctx: object | None = None,
    options: Mapping[str, object] | None = None,
    views: Sequence[View] | None = None,
    capture: SequentialCaptureRequest | None = None,
    execution_context: EvidenceExecutionContext | None = None,
) -> tuple[ScopedReadout, ...]:
    """Resolve and verify the experiment's source before the reader executes its views."""
    exp = _select_views(exp, views)
    provider = _resolve_verified(exp, reader, ctx=ctx)
    return reader._run_experiment(
        exp,
        provider,
        options=options,
        capture=capture,
        execution_context=execution_context,
    )
