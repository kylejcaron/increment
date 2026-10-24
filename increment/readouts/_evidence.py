from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, overload

from increment._evidence_refusals import require_evidence, require_population
from increment._evidence_types import (
    BreakoutIdentity,
    BreakoutMethodMomentsRequest,
    BreakoutRoute,
    BreakoutSlice,
    BreakoutSlices,
    BreakoutSlicesRequest,
    BreakoutUnitRows,
    BreakoutUnitRowsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    CountEvidence,
    CountsRequest,
    DayAxisRequest,
    DayAxisRoute,
    DayAxisRows,
    EvidenceScope,
    EvidenceScopeResult,
    MethodFitAnswer,
    MethodFitRequest,
    MomentRows,
    MomentsRequest,
    ScalarBreakoutRows,
    SequentialCapture,
    SequentialCaptureRequest,
    UnitRows,
    UnitRowsRequest,
)
from increment._source_types import (
    ComplianceSummary,
    ScalarBreakoutMomentSource,
    ScalarMethodMomentSource,
    compliance_summary_series,
    declared_evidence,
    evidence_is_declared,
    validate_compliance_source_match,
)
from increment.semantics.models import RetentionMetric
from increment.sources import (
    BreakoutMomentsSource,
    MomentSource,
    SourceContext,
    require_operation,
)


@dataclass(frozen=True, slots=True)
class DayAxisSlice:
    rows: list[dict[str, Any]]
    metrics: tuple[Any, ...]
    view: Any
    dimension: str | None
    source_name: str | None


ARTIFACT_ROUTE = "artifact"
NATIVE_ROUTE = "native"
MOMENTS_ROUTE = "moments"
PANEL_ROUTE = "panel"
UNSUPPORTED_ROUTE = "unsupported"
SourceRoute = Literal["artifact", "native", "moments", "panel"]


def source_route(src: MomentSource) -> SourceRoute:
    evidence = declared_evidence(src)
    if "artifact_experiment" in evidence.operations:
        return ARTIFACT_ROUTE
    if "moments_source" not in evidence.operations:
        return PANEL_ROUTE if evidence.shape == "unit_panel" else MOMENTS_ROUTE
    return NATIVE_ROUTE


def day_axis_route(src: MomentSource) -> DayAxisRoute:
    route = source_route(src)
    return MOMENTS_ROUTE if route == PANEL_ROUTE else route


def is_artifact_route(src: MomentSource) -> bool:
    return source_route(src) == ARTIFACT_ROUTE


def breakout_route(src: MomentSource) -> BreakoutRoute:
    route = source_route(src)
    return UNSUPPORTED_ROUTE if route == MOMENTS_ROUTE else route


def require_breakout_identity(src: MomentSource, identities: Sequence[BreakoutIdentity]) -> None:
    """Validate breakout identities against the already-selected source declaration."""
    if isinstance(src, BreakoutMomentsSource):
        requested = tuple(f"{item.source}:{item.property}" for item in identities)
        offered = tuple(f"{src.source_name}:{dimension}" for dimension in src.breakouts)
        if any(
            item.property not in src.breakouts
            or (item.source is not None and item.source != src.source_name)
            for item in identities
        ):
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                src,
                requested=requested,
                offered=offered,
                route="use the breakout source declared by this scoped source",
            )
        return

    validator = getattr(src, "_preflight_breakout", None)
    if callable(validator):
        from increment.semantics.models import Breakout

        for identity in identities:
            validator(
                Breakout(property=identity.property, source=identity.source),
                operation="breakout_evidence",
            )
        return

    evidence = declared_evidence(src)
    if any(identity.property not in evidence.breakouts for identity in identities):
        from increment._evidence_refusals import refuse_breakout

        refuse_breakout(
            src,
            requested=tuple(identity.property for identity in identities),
            offered=evidence.breakouts,
            route="use a breakout declared by this source",
        )


@contextmanager
def evidence_scope(src: MomentSource, scope: EvidenceScope) -> Iterator[MomentSource]:
    """Pin the requested source view and reduction lifetime for one readout."""
    from increment._source_operations import ReadoutExecutionOperation, ReadoutSnapshotOperation

    with ExitStack() as stack:
        scoped = src
        if scope.snapshot and isinstance(scoped, ReadoutSnapshotOperation):
            scoped = stack.enter_context(
                scoped.readout_snapshot(
                    metrics=scope.metrics,
                    population=scope.population,
                    include_breakouts=scope.include_breakouts or bool(scope.breakouts),
                    configs=scope.configs,
                    uptake_facts=scope.uptake_facts,
                )
            )
        if scope.execution and isinstance(scoped, ReadoutExecutionOperation):
            stack.enter_context(scoped.readout_execution())
        yield scoped


def source_for_population(src: MomentSource, population: str) -> MomentSource:
    """Select an existing population view or acquire the requested one."""
    declaration = declared_evidence(src)
    if population == "triggered" and not (
        "triggered" in declaration.populations and "assigned" not in declaration.populations
    ):
        from increment._source_operations import TriggeredPopulationOperation

        return require_operation(
            src, "triggered_source", TriggeredPopulationOperation
        ).triggered_source()
    return src


def require_request_evidence(
    src: MomentSource, request: Any, *, declaration_source: MomentSource | None = None
) -> None:
    """Validate against the effective population and parent declaration."""
    if (
        isinstance(request, (MomentsRequest, UnitRowsRequest))
        and getattr(src, "_population", None) == "triggered"
        and getattr(request, "population", None) == "assigned"
    ):
        request = replace(request, population="triggered")
    source = declaration_source or src
    if evidence_is_declared(source):
        require_evidence(source, request=request)


def load_moments(src: MomentSource, request: MomentsRequest) -> MomentRows:
    """Acquire aggregate moments through the source's declared method route."""
    require_request_evidence(src, request)
    method = request.method
    if method is not None:
        from increment._evidence_refusals import refuse_grain, refuse_operation

        if request.by:
            refuse_operation(
                src,
                {"method": method.name, "by": request.by},
                ("method-specific breakout rows with segment refusals",),
                route="use BreakoutMethodMomentsRequest for dimensioned method evidence",
            )
        if request.grain != "total":
            refuse_grain(
                src,
                request.grain,
                ("total",),
                route="use method-specific evidence at total grain",
            )
        if request.include_covariate or request.completed_windows_only:
            option = "include_covariate" if request.include_covariate else "completed_windows_only"
            refuse_operation(
                src,
                {"method": method.name, option: True},
                ("method-specific total rows by breakout",),
                route="use a source with the requested method evidence option",
            )

    source = source_for_population(src, request.population)
    if method is None:
        return tuple(
            source.moments(
                request.metric,
                grain=request.grain,
                by=request.by,
                include_covariate=request.include_covariate,
                completed_windows_only=request.completed_windows_only,
            )
        )

    if method.variance_reduction == "none":
        scalar = declared_evidence(source).scalar
        if (
            scalar is not None
            and scalar.method_evidence
            and isinstance(source, ScalarMethodMomentSource)
        ):
            return tuple(source.moments_for_method(request.metric, method, by=request.by))
        return tuple(source.moments(request.metric, grain="total", by=request.by))

    scalar = declared_evidence(source).scalar
    if (
        scalar is not None
        and scalar.method_evidence
        and isinstance(source, ScalarMethodMomentSource)
    ):
        return tuple(source.moments_for_method(request.metric, method, by=request.by))
    refuse_operation(
        source,
        {"method": method.name},
        ("method-specific total rows by breakout",),
        route="use a source that retains method-specific evidence",
    )


def load_unit_rows(src: MomentSource, request: UnitRowsRequest) -> UnitRows:
    """Acquire source unit rows for the requested outcome stage."""
    require_request_evidence(src, request)

    source = source_for_population(src, request.population)
    kwargs = {"covariates": request.covariates}
    if request.outcome_stage == "raw":
        kwargs["outcome_stage"] = "raw"
    return cast(Any, source).unit_frame(request.metric, **kwargs)


def load_counts(src: MomentSource, request: CountsRequest) -> CountEvidence:
    """Acquire count evidence without changing the requested count grain."""
    if request.kind == "cluster":
        evidence = declared_evidence(src)
        capability_source = getattr(src, "_source", src)
        if "cluster" not in evidence.count_grains and (
            evidence_is_declared(capability_source)
            or getattr(getattr(src, "context", None), "cluster", None) is None
        ):
            from increment._evidence_refusals import refuse_cluster

            refuse_cluster(
                src,
                offered=evidence.count_grains,
                route="use a source that retains cluster identities or cluster counts",
            )
        require_request_evidence(src, request)
        source = source_for_population(src, request.population)
        unit_counts = (
            source.unit_counts() if "unit" in declared_evidence(source).count_grains else {}
        )
        return CountEvidence(request.population, "cluster", source.cluster_counts(), unit_counts)
    require_request_evidence(src, request)
    if request.kind == "triggered" or (
        request.population == "triggered" and callable(getattr(src, "triggered_counts", None))
    ):
        if request.kind != "triggered":
            require_population(src, request.population)
        grain, counts, units = cast(Any, src).triggered_counts()
        if request.kind == "unit":
            return CountEvidence(
                request.population,
                "unit",
                counts if grain == "unit" else units,
                units,
            )
        if request.kind == "assignment":
            return CountEvidence(request.population, "assignment", counts)
        return CountEvidence("triggered", grain, counts, units)

    source = source_for_population(src, request.population)
    if request.kind == "unit":
        return CountEvidence(request.population, "unit", source.unit_counts())
    if request.kind == "assignment":
        counts_method = getattr(source, "assignment_counts", None)
        if callable(counts_method):
            counts = counts_method(population=request.population)
        elif "cluster" in declared_evidence(source).count_grains or getattr(
            getattr(source, "context", None), "cluster", None
        ):
            counts = cast(Any, source).cluster_counts()
        else:
            counts = source.unit_counts()
        return CountEvidence(request.population, "assignment", counts)
    grain, counts, units = cast(Any, source).triggered_counts()
    return CountEvidence("triggered", grain, counts, units)


def load_method_fit(src: MomentSource, request: MethodFitRequest) -> MethodFitAnswer:
    """Acquire retained or warehouse method-fit evidence, never fitting locally."""
    require_request_evidence(src, request, declaration_source=getattr(src, "_source", src))
    from increment.readouts._metric_rows import _acquire_method_fit

    return _acquire_method_fit(src, request)


def load_sequential_capture(
    src: MomentSource, request: SequentialCaptureRequest
) -> SequentialCapture:
    """Capture a source snapshot using Analysis.capture_sequential's dispatch."""
    require_request_evidence(src, request)
    from increment.sequential_source import adopt_source_snapshot, source_snapshot
    from increment.sequential_state import declare_sequential_freeze, sequential_refuse

    capture = getattr(src, "capture_sequential", None)
    if capture is None:
        if not request.finalized or request.as_of is not None:
            sequential_refuse(
                "source.invalid", "this source requires its construction-time finalized checkpoint"
            )
        snapshot = source_snapshot(src, previous=request.previous)
    else:
        if request.as_of is None:
            sequential_refuse("source.invalid", "capture requires an explicit as_of horizon")
        snapshot = capture(
            finalized=request.finalized,
            previous=request.previous,
            as_of=request.as_of,
        )
    if request.freeze:
        snapshot = adopt_source_snapshot(
            src, declare_sequential_freeze(snapshot, tuple(request.freeze))
        )
    return snapshot


@contextmanager
def _scope_source(src: MomentSource, request: EvidenceScope) -> Iterator[MomentSource]:
    with evidence_scope(src, request) as scoped:
        yield scoped


def load_evidence_scope(src: MomentSource, request: EvidenceScope) -> EvidenceScopeResult:
    """Return a context-managed scoped view, closing source resources on exit."""
    require_request_evidence(src, request)
    return EvidenceScopeResult(_scope_source(src, request), request)


@overload
def load_breakout_evidence(src: MomentSource, request: BreakoutSlicesRequest) -> BreakoutSlices: ...


@overload
def load_breakout_evidence(
    src: MomentSource, request: BreakoutUnitRowsRequest
) -> BreakoutUnitRows: ...


def load_breakout_evidence(
    src: MomentSource, request: BreakoutSlicesRequest | BreakoutUnitRowsRequest
) -> BreakoutSlices | BreakoutUnitRows:
    """Acquire one typed breakout evidence request."""
    require_request_evidence(src, request)
    if isinstance(request, BreakoutUnitRowsRequest):
        identities = (request.breakout,)
    else:
        identities = request.breakouts
    require_breakout_identity(src, identities)

    if isinstance(request, BreakoutUnitRowsRequest):
        if isinstance(src, BreakoutMomentsSource):
            frame = src.unit_frame(request.metric, covariates=request.covariates)
            dimension_in_frame = False
        else:
            covariates = tuple(dict.fromkeys((request.breakout.property, *request.covariates)))
            frame = src.unit_frame(request.metric, covariates=covariates)
            dimension_in_frame = True
        return BreakoutUnitRows(request.breakout, breakout_route(src), frame, dimension_in_frame)

    route = breakout_route(src)
    if route == ARTIFACT_ROUTE:
        definitions = request.breakout_definitions
        if not definitions:
            from increment.semantics.models import Breakout

            definitions = tuple(
                Breakout(property=item.property, source=item.source) for item in request.breakouts
            )
        source = cast(Any, src)
        sources = source.breakout_sources(
            definitions,
            metrics=list(request.metrics),
            population=request.population,
            configs=request.configs,
        )
        slices = tuple(
            BreakoutSlice(identity, scoped, None, scoped.source_name)
            for identity, scoped in zip(request.breakouts, sources, strict=True)
        )
    elif route == PANEL_ROUTE:
        slices = tuple(BreakoutSlice(identity, src, None, None) for identity in request.breakouts)
    elif route == NATIVE_ROUTE:
        if "breakout_sources" not in src.operations:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                src,
                tuple(item.property for item in request.breakouts),
                tuple(src.breakouts),
                route=(
                    "run_breakout() needs a native Analysis.from_definitions or "
                    "Analysis.from_unit_panel instance with declared breakouts"
                ),
            )
        from increment._source_operations import BreakoutSourcesOperation

        route_message = (
            "run_breakout() needs a native Analysis.from_definitions or "
            "Analysis.from_unit_panel instance with declared breakouts"
        )
        if not isinstance(getattr(src, "context", None), SourceContext):
            from increment._evidence_refusals import refuse_operation

            refuse_operation(
                src, "breakout_sources", tuple(sorted(src.operations)), route=route_message
            )
        native = require_operation(src, "breakout_sources", BreakoutSourcesOperation)
        definitions = request.breakout_definitions
        if not definitions:
            from increment.semantics.models import Breakout

            definitions = tuple(
                Breakout(property=item.property, source=item.source) for item in request.breakouts
            )
        sources = native.breakout_sources(
            definitions,
            metrics=list(request.metrics),
            population=request.population,
            configs=request.configs,
        )
        slices = tuple(
            BreakoutSlice(
                identity,
                scoped,
                scoped.source_name
                if sum(item.property == identity.property for item in request.breakouts) > 1
                else None,
                scoped.source_name,
            )
            for identity, scoped in zip(request.breakouts, sources, strict=True)
        )
    else:
        slices = ()
    return BreakoutSlices(route, slices)


def _groups(req: DayAxisRequest) -> list[tuple[Any, tuple[Any, ...]]]:
    if req.caller in ("run_daily", "run_asof"):
        return [
            (
                "asof"
                if req.grain == "asof"
                else ("cohort" if isinstance(metric, RetentionMetric) else "daily"),
                (metric,),
            )
            for metric in req.metrics
        ]
    if req.grain == "asof":
        return [("asof", tuple(req.metrics))] if req.metrics else []
    activity = tuple(m for m in req.metrics if not isinstance(m, RetentionMetric))
    retention = tuple(m for m in req.metrics if isinstance(m, RetentionMetric))
    groups = []
    if activity:
        groups.append(("daily", activity))
    if retention:
        groups.append(("cohort", retention))
    return groups


def _include_covariate(req: DayAxisRequest, metric: Any) -> bool:
    return metric.name in req.covariate_metrics


def _metric_kwargs(metric: Any, req: DayAxisRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"grain": req.grain}
    if req.grain == "asof":
        kwargs["completed_windows_only"] = req.completed_windows_only
    if _include_covariate(req, metric):
        kwargs["include_covariate"] = True
    return kwargs


def _rows(
    src: MomentSource,
    req: DayAxisRequest,
    *,
    validate_daily_readout: bool = False,
) -> Iterator[DayAxisSlice]:
    by = req.by or ((req.dimension,) if req.dimension is not None else ())
    for view, metrics in _groups(req):
        rows: list[dict[str, Any]] = []
        for metric in metrics:
            if validate_daily_readout:
                _validate_daily_readout(src, req, metric)
            rows.extend(
                cast(
                    "Iterable[dict[str, Any]]",
                    src.moments(metric, by=by, **_metric_kwargs(metric, req)),
                )
            )
        yield DayAxisSlice(rows, metrics, view, req.dimension, None)


def _validate_daily_readout(src: MomentSource, req: DayAxisRequest, metric: Any) -> None:
    if req.caller not in ("run_daily", "run_daily_lift") or req.grain != "daily":
        return
    from increment._readout_request import ReadoutRequest, validate_request

    configs = {config.metric.name: config for config in src.context.configs}
    readout_request = ReadoutRequest.from_source(
        src,
        metrics=(metric,),
        configs=(configs[metric.name],),
        view="daily",
        grain="daily",
        by=req.by,
    )
    validate_request(readout_request)


class _MomentsDayAxis:
    __slots__ = ("src", "validate_daily_readout")

    def __init__(self, src: MomentSource, *, validate_daily_readout: bool) -> None:
        self.src = src
        self.validate_daily_readout = validate_daily_readout

    def slices(self, req: DayAxisRequest) -> Iterator[DayAxisSlice]:
        yield from _rows(
            self.src,
            req,
            validate_daily_readout=self.validate_daily_readout,
        )


class _ArtifactDayAxis:
    __slots__ = ("src", "experiment")

    def __init__(self, src: MomentSource, experiment: Any | None) -> None:
        self.src = src
        self.experiment = experiment

    def slices(self, req: DayAxisRequest) -> Iterator[DayAxisSlice]:
        if req.dimension is None:
            yield from _rows(self.src, req, validate_daily_readout=True)
            return
        matching = [
            breakout
            for breakout in (self.experiment.breakouts if self.experiment else ())
            if breakout.property == req.dimension
        ]
        for breakout in matching:
            for metric in req.metrics:
                result = cast(Any, self.src).breakout_moments(
                    metric, breakout, **_metric_kwargs(metric, req)
                )
                view = (
                    "asof"
                    if req.grain == "asof"
                    else ("cohort" if isinstance(metric, RetentionMetric) else "daily")
                )
                yield DayAxisSlice(
                    list(result.rows), (metric,), view, req.dimension, result.source_name
                )


class _NativeDayAxis:
    __slots__ = ("src", "experiment")

    def __init__(self, src: MomentSource, experiment: Any | None) -> None:
        self.src = src
        self.experiment = experiment

    def slices(self, req: DayAxisRequest) -> Iterator[DayAxisSlice]:
        if req.dimension is not None:
            matching = [
                breakout
                for breakout in (self.experiment.breakouts if self.experiment else ())
                if breakout.property == req.dimension
            ]
            for breakout in matching:
                for view, metrics in _groups(req):
                    rows: list[dict[str, Any]] = []
                    source_name: str | None = None
                    for index, metric in enumerate(metrics):
                        result = cast(Any, self.src).breakout_moments(
                            metric, breakout, **_metric_kwargs(metric, req)
                        )
                        rows.extend(result.rows)
                        if index == 0:
                            source_name = result.source_name
                    yield DayAxisSlice(rows, metrics, view, req.dimension, source_name)
            return
        yield from _rows(self.src, req)


def load_day_axis_rows(src: MomentSource, request: DayAxisRequest) -> DayAxisRows:
    """Acquire daily/as-of rows using the evidence route declared by ``src``."""
    require_request_evidence(src, request)
    route = day_axis_route(src)
    if route == NATIVE_ROUTE and (
        request.experiment is not None
        or request.caller in ("run_daily_lift", "run_asof_lift")
        or request.dimension is not None
        or request.population == "triggered"
    ):
        from increment._source_operations import DaySourceOperation

        operation = require_operation(src, "day_source", DaySourceOperation)
        day_source = operation.day_source(
            metrics=list(request.metrics), population=request.population
        )
        evidence = _NativeDayAxis(day_source, request.experiment)
    elif route == ARTIFACT_ROUTE:
        source = source_for_population(src, request.population)
        evidence = _ArtifactDayAxis(source, request.experiment)
    else:
        source = source_for_population(src, request.population)
        evidence = _MomentsDayAxis(source, validate_daily_readout=route != NATIVE_ROUTE)
    return DayAxisRows(evidence, route, evidence.slices(request))


def load_breakout_method_moments(
    src: MomentSource, request: BreakoutMethodMomentsRequest
) -> ScalarBreakoutRows:
    """Acquire the method-specific breakout moments used by segment readouts."""
    from increment._evidence_refusals import refuse_operation

    require_request_evidence(src, request)
    require_breakout_identity(src, (request.breakout,))
    if request.completed_windows_only:
        refuse_operation(
            src,
            {"breakout_method_moments": True, "completed_windows_only": True},
            ("method-specific breakout moments without window filtering",),
            route="use a source that supports completed-window method evidence",
        )
    method_source = src
    if callable(getattr(src, "breakout_source", None)):
        from increment._analysis_config import overlay_configs
        from increment.semantics.models import Breakout

        configs = overlay_configs(
            (request.metric,), src.context.configs, methods=[request.method], prior=None
        )
        method_source = cast(Any, src).breakout_source(
            Breakout(property=request.breakout.property, source=request.breakout.source),
            metrics=(request.metric,),
            configs=configs,
            population=request.population,
        )
    if not isinstance(method_source, ScalarBreakoutMomentSource):
        refuse_operation(
            src,
            "breakout_method_moments",
            ("method-specific breakout moments",),
            route="use a source that retains method-specific breakout evidence",
        )
    return method_source.breakout_moments_for_method(
        request.metric, request.method, by=(request.breakout.property,)
    )


def load_compliance(src: MomentSource, request: ComplianceRequest) -> ComplianceSummary:
    require_request_evidence(src, request)
    summary = src.compliance_summary(
        request.design,
        as_of=request.as_of,
        completed_windows_only=request.completed_windows_only,
    )
    validate_compliance_source_match(summary, src.context, as_of=request.as_of)
    return summary


def load_compliance_series(
    src: MomentSource, request: ComplianceSeriesRequest
) -> tuple[tuple[object, ComplianceSummary], ...]:
    require_request_evidence(src, request)
    source = source_for_population(src, request.population)
    summaries = compliance_summary_series(
        source, request.design, completed_windows_only=request.completed_windows_only
    )
    return tuple(summaries.items())
