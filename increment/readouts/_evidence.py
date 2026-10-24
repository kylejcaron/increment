from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Literal, cast

from increment._evidence_refusals import require_evidence
from increment._evidence_types import (
    ComplianceRequest,
    ComplianceSeriesRequest,
    DayAxisRequest,
    DayAxisRows,
)
from increment._source_types import (
    ComplianceSummary,
    classify_source,
    compliance_summary_series,
    validate_compliance_source_match,
)
from increment.semantics.models import RetentionMetric
from increment.sources import MomentSource, require_operation


@dataclass(frozen=True, slots=True)
class DayAxisSlice:
    rows: list[dict[str, Any]]
    metrics: tuple[Any, ...]
    view: Any
    dimension: str | None
    source_name: str | None


def _route(src: MomentSource) -> Literal["artifact", "moments", "native"]:
    route = classify_source(src)
    return "moments" if route == "panel" else route


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
    route = _route(src)
    if route == "native" and (
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
    elif route == "artifact":
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
        evidence = _MomentsDayAxis(src, validate_daily_readout=route != "native")
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
