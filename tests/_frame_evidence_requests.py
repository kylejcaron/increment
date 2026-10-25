from __future__ import annotations

from typing import Any, Literal, cast

from increment.evidence import (
    BreakoutIdentity,
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    CountEvidence,
    CountsRequest,
    MomentsRequest,
    UnitRowsRequest,
    evidence_provider,
    request_evidence,
)


def _moments(source: Any, metric: Any, **kwargs: Any) -> Any:
    request = MomentsRequest(metric, **kwargs)
    return request_evidence(evidence_provider(source), request)


def _unit_rows(source: Any, metric: Any, **kwargs: Any) -> Any:
    request = UnitRowsRequest(metric, **kwargs)
    return request_evidence(evidence_provider(source), request)


def _counts(source: Any, kind: Literal["unit", "cluster", "assignment", "triggered"]) -> Any:
    answer = cast(
        CountEvidence,
        request_evidence(evidence_provider(source), CountsRequest(kind=kind)),
    )
    return answer.counts


def _compliance(source: Any, design: Any) -> Any:
    return request_evidence(evidence_provider(source), ComplianceRequest(design))


def _breakout_moments(source: Any, metric: Any, method: Any, *, by: tuple[str, ...]) -> Any:
    breakout = BreakoutIdentity(property=by[0], source=None)
    request = BreakoutMethodMomentsRequest(metric, method, breakout)
    return request_evidence(evidence_provider(source), request)
