"""Typed evidence request helpers shared by evidence-provider migration tests."""

from __future__ import annotations

from typing import Any, cast

from increment.evidence import (
    ComplianceRequest,
    ComplianceSeriesRequest,
    CountsRequest,
    MomentsRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)


def _moments(source: Any, metric: Any, **options: Any) -> Any:
    if "by" in options:
        options["by"] = tuple(options["by"])
    if "grain" not in options and "total" not in source.evidence.grains:
        options["grain"] = "daily"
    if "population" not in options and source.evidence.populations == frozenset({"triggered"}):
        options["population"] = "triggered"
    return request_evidence(evidence_provider(source), MomentsRequest(metric, **options))


def _unit_rows(source: Any, metric: Any, **options: Any) -> Any:
    if "covariates" in options:
        options["covariates"] = tuple(options["covariates"])
    if "population" not in options and source.evidence.populations == frozenset({"triggered"}):
        options["population"] = "triggered"
    return request_evidence(evidence_provider(source), UnitRowsRequest(metric, **options))


def _compliance(source: Any, design: Any, **options: Any) -> Any:
    return request_evidence(evidence_provider(source), ComplianceRequest(design, **options))


def _compliance_dates(source: Any) -> tuple[Any, ...]:
    rows = cast(
        Any,
        request_evidence(evidence_provider(source), ComplianceSeriesRequest(source.context.design)),
    )
    return tuple(day for day, _summary in rows)


def _counts(source: Any, kind: Any, **options: Any) -> Any:
    if "population" not in options and source.evidence.populations == frozenset({"triggered"}):
        options["population"] = "triggered"
    answer = cast(
        Any,
        request_evidence(evidence_provider(source), CountsRequest(kind, **options)),
    )
    return answer.unit_counts if kind == "triggered" else answer.counts


def _unit_counts(source: Any) -> Any:
    return _counts(source, "unit")


def _cluster_counts(source: Any) -> Any:
    return _counts(source, "cluster")


def _assignment_counts(source: Any, **options: Any) -> Any:
    return _counts(source, "assignment", **options)


def _breakout_moments(source: Any, metric: Any, breakout: Any, **options: Any) -> Any:
    options.setdefault("grain", "daily")
    return _moments(source, metric, by=(breakout.property,), **options)
