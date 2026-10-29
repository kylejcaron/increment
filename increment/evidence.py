"""Typed requests and answers for evidence supplied by data sources.

Custom providers implement :class:`EvidenceProvider` and declare what they can
serve with ``EvidenceCapabilities``. Call :func:`request_evidence` to validate
each request against that declaration before dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from increment._evidence_refusals import refuse_unsupported_request, require_evidence
from increment._evidence_types import (
    AssignmentEvidence,
    AssignmentRequest,
    BreakoutCell,
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
    ComplianceSummary,
    CountEvidence,
    CountsRequest,
    DayAxisRequest,
    DayAxisRoute,
    DayAxisRows,
    EvidenceScope,
    EvidenceScopeResult,
    EvidenceSource,
    MethodFitAnswer,
    MethodFitRequest,
    MetricEvidence,
    MetricEvidenceRequest,
    MomentRows,
    MomentsRequest,
    PreparedMulticovariateFit,
    ScalarBreakoutRows,
    SequentialCapture,
    SequentialCaptureRequest,
    SequentialSnapshotRequest,
    SourceContext,
    UnitRows,
    UnitRowsRequest,
    warehouse_cuped_fit_evidence,
)
from increment._source_types import (
    EvidenceCapabilities,
    EvidenceProvider,
    ScalarEvidenceDeclaration,
    is_builtin_source,
    refuse_provider_unavailable,
)
from increment.readouts._provider import _BuiltInEvidenceProvider

if TYPE_CHECKING:
    from increment._frame_ledger import FrameLedger
    from increment.semantics.assignment import AssignmentSpec


def _validate(provider: Any, request: object) -> None:
    """Apply the shared capability check before calling a loader method."""
    require_evidence(provider, request=request)


def evidence_provider(source: Any) -> EvidenceProvider:
    """Return the built-in adapter or structural custom provider."""
    if is_builtin_source(source):
        return _BuiltInEvidenceProvider(source)
    if isinstance(source, EvidenceProvider):
        return source
    if isinstance(getattr(source, "evidence", None), EvidenceCapabilities):
        return _BuiltInEvidenceProvider(source)

    refuse_provider_unavailable(source)


def _provider_source(provider: EvidenceProvider) -> Any:
    return provider.source if isinstance(provider, _BuiltInEvidenceProvider) else provider


def request_evidence(provider: EvidenceProvider, request: object) -> object:
    """Validate and answer primitive requests, delegating compound reads."""
    source = _provider_source(provider)
    is_custom = not isinstance(provider, _BuiltInEvidenceProvider) and isinstance(
        provider, EvidenceProvider
    )
    if isinstance(request, MetricEvidenceRequest):
        from increment.readouts._metric_rows import load_metric_evidence, metric_evidence_scope

        if is_custom:
            require_evidence(provider, request=request)
            scoped = cast(
                EvidenceScopeResult,
                request_evidence(provider, metric_evidence_scope(cast(Any, provider), request)),
            )
            with scoped.source as pinned:
                return load_metric_evidence(pinned, request, scope_active=True)
        return load_metric_evidence(source, request)
    if isinstance(request, (DayAxisRequest, BreakoutSlicesRequest, BreakoutUnitRowsRequest)):
        from increment.readouts._evidence import load_breakout_evidence, load_day_axis_rows

        if isinstance(request, DayAxisRequest):
            return load_day_axis_rows(source, request)
        return load_breakout_evidence(source, request)
    if isinstance(request, SequentialSnapshotRequest):
        from increment.readouts._evidence import current_source_snapshot, loader_snapshot

        require_evidence(source, request=request)
        if not is_custom:
            # The built-in adapter answers from its loader's own retained checkpoint.
            return loader_snapshot(source, previous=request.previous)
        return current_source_snapshot(provider, previous=request.previous)
    if isinstance(request, AssignmentRequest):
        require_evidence(source, request=request)
        method = getattr(provider, "assignment", None)
        if method is None:
            from increment._evidence_refusals import refuse_assignment

            refuse_assignment(source, offered="declared capability without provider method")
        return method(request)
    names = {
        BreakoutMethodMomentsRequest: "breakout_method_moments",
        MomentsRequest: "moments",
        UnitRowsRequest: "unit_rows",
        MethodFitRequest: "method_fit",
        ComplianceRequest: "compliance",
        ComplianceSeriesRequest: "compliance_series",
        CountsRequest: "counts",
        SequentialCaptureRequest: "capture_sequential",
        EvidenceScope: "scope",
    }
    method = names.get(type(request))
    if method is None:
        refuse_unsupported_request(
            provider,
            request,
            tuple(request_type.__name__ for request_type in names),
        )
    require_evidence(source, request=request)
    return getattr(provider, method)(request)


def replay(spec: AssignmentSpec, units: Iterable[str]) -> FrameLedger:
    """Replay a versioned assignment spec into a validated identity ledger."""
    import pyarrow as pa

    from increment._frame_ledger import FrameLedger

    rows = spec.replay(units)
    schema = pa.schema(
        [
            ("unit", pa.string()),
            ("experiment", pa.string()),
            ("arm", pa.string()),
            ("period", pa.string()),
            ("version", pa.int64()),
            ("layer", pa.string()),
            ("allocated_at", pa.timestamp("us")),
            ("exposed_at", pa.timestamp("us")),
            ("triggered_at", pa.timestamp("us")),
        ]
    )
    frame = pa.Table.from_pylist(rows, schema=schema)
    return FrameLedger(frame, roles={name: name for name in frame.column_names})


__all__ = [
    "BreakoutCell",
    "BreakoutRoute",
    "BreakoutIdentity",
    "BreakoutSlice",
    "BreakoutSlices",
    "AssignmentEvidence",
    "FrameLedger",
    "AssignmentRequest",
    "BreakoutSlicesRequest",
    "BreakoutUnitRows",
    "BreakoutUnitRowsRequest",
    "ComplianceRequest",
    "ComplianceSeriesRequest",
    "ComplianceSummary",
    "CountEvidence",
    "CountsRequest",
    "DayAxisRequest",
    "DayAxisRoute",
    "DayAxisRows",
    "ScalarEvidenceDeclaration",
    "EvidenceSource",
    "EvidenceCapabilities",
    "EvidenceProvider",
    "EvidenceScope",
    "EvidenceScopeResult",
    "MethodFitAnswer",
    "MethodFitRequest",
    "MetricEvidence",
    "MetricEvidenceRequest",
    "MomentRows",
    "BreakoutMethodMomentsRequest",
    "ScalarBreakoutRows",
    "MomentsRequest",
    "PreparedMulticovariateFit",
    "SequentialCapture",
    "SequentialSnapshotRequest",
    "SequentialCaptureRequest",
    "SourceContext",
    "UnitRows",
    "UnitRowsRequest",
    "replay",
    "evidence_provider",
    "request_evidence",
    "require_evidence",
    "warehouse_cuped_fit_evidence",
]


def __getattr__(name: str) -> Any:
    if name == "FrameLedger":
        from increment._frame_ledger import FrameLedger

        return FrameLedger
    raise AttributeError(name)
