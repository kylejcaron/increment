from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast, overload

from increment._evidence_refusals import require_evidence
from increment._evidence_types import (
    BreakoutSlice,
    BreakoutSlices,
    BreakoutSlicesRequest,
    BreakoutUnitRows,
    BreakoutUnitRowsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    DayAxisRequest,
    DayAxisRows,
    EvidenceScope,
)
from increment._source_types import (
    ComplianceSummary,
    compliance_summary_series,
    declared_evidence,
    validate_compliance_source_match,
)
from increment.semantics.models import RetentionMetric
from increment.sources import MomentSource, SourceContext, require_operation


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
DayAxisRoute = Literal["artifact", "moments", "native"]
BreakoutRoute = Literal["artifact", "native", "panel", "unsupported"]


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
    if isinstance(request, BreakoutUnitRowsRequest):
        from increment.sources import BreakoutMomentsSource

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
    require_evidence(
        src,
        grain=request.grain,
        by=request.by or ((request.dimension,) if request.dimension is not None else ()),
    )
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
        source: Any = src
        if request.population == "triggered":
            from increment._source_operations import TriggeredPopulationOperation

            source = require_operation(
                src, "triggered_source", TriggeredPopulationOperation
            ).triggered_source()
        evidence = _ArtifactDayAxis(source, request.experiment)
    else:
        if request.population == "triggered":
            from increment._source_operations import TriggeredPopulationOperation

            require_operation(src, "triggered_source", TriggeredPopulationOperation)
        evidence = _MomentsDayAxis(src, validate_daily_readout=route != NATIVE_ROUTE)
    return DayAxisRows(evidence, route, evidence.slices(request))


def load_compliance(src: MomentSource, request: ComplianceRequest) -> ComplianceSummary:
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
    source = src
    if request.population == "triggered":
        from increment._source_operations import TriggeredPopulationOperation

        source = require_operation(
            src, "triggered_source", TriggeredPopulationOperation
        ).triggered_source()
    summaries = compliance_summary_series(
        source, request.design, completed_windows_only=request.completed_windows_only
    )
    return tuple(summaries.items())
