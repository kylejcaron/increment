"""Typed requests for the statistical evidence available from data sources.

Custom sources can implement :class:`EvidenceProvider`; built-in sources can be
adapted with :func:`evidence_provider` while their existing methods remain available.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from increment._evidence_refusals import refuse_unsupported_request, require_evidence
from increment._evidence_types import (
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
    CountEvidence,
    CountsRequest,
    DayAxisRequest,
    DayAxisRoute,
    DayAxisRows,
    EvidenceScope,
    EvidenceScopeResult,
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
    UnitRows,
    UnitRowsRequest,
    warehouse_cuped_fit_evidence,
)
from increment._source_types import (
    ComplianceSummary,
    EvidenceCapabilities,
    ScalarEvidenceDeclaration,
    SourceContext,
    declared_evidence,
)


@runtime_checkable
class EvidenceProvider(Protocol):
    """A source of declared statistical evidence for an analysis."""

    evidence: EvidenceCapabilities
    context: SourceContext

    def breakout_evidence(
        self, request: BreakoutSlicesRequest | BreakoutUnitRowsRequest
    ) -> BreakoutSlices | BreakoutUnitRows: ...

    def day_axis(self, request: DayAxisRequest) -> DayAxisRows: ...

    def metric_evidence(self, request: MetricEvidenceRequest) -> MetricEvidence: ...

    def moments(self, request: MomentsRequest) -> MomentRows: ...
    def breakout_method_moments(
        self, request: BreakoutMethodMomentsRequest
    ) -> ScalarBreakoutRows: ...

    def unit_rows(self, request: UnitRowsRequest) -> UnitRows: ...

    def method_fit(self, request: MethodFitRequest) -> MethodFitAnswer: ...

    def compliance(self, request: ComplianceRequest) -> ComplianceSummary: ...

    def compliance_series(
        self, request: ComplianceSeriesRequest
    ) -> tuple[tuple[object, ComplianceSummary], ...]: ...

    def counts(self, request: CountsRequest) -> CountEvidence: ...

    def capture_sequential(self, request: SequentialCaptureRequest) -> SequentialCapture: ...

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult: ...


def _validate(provider: Any, request: object) -> None:
    """Apply the shared capability check before calling a loader method."""
    require_evidence(provider, request=request)


class _BuiltInEvidenceProvider:
    """Present the evidence methods of an existing loader as typed requests."""

    __slots__ = ("source", "evidence", "context")

    def __init__(self, source: Any) -> None:
        self.source = source
        self.evidence = declared_evidence(source)
        self.context = source.context

    def moments(self, request: MomentsRequest) -> MomentRows:
        _validate(self.source, request)
        from increment.readouts._evidence import load_moments

        return load_moments(self.source, request)

    def unit_rows(self, request: UnitRowsRequest) -> UnitRows:
        _validate(self.source, request)
        from increment.readouts._evidence import load_unit_rows

        return load_unit_rows(self.source, request)

    def method_fit(self, request: MethodFitRequest) -> MethodFitAnswer:
        _validate(self.source, request)
        from increment.readouts._evidence import load_method_fit

        return load_method_fit(self.source, request)

    def breakout_method_moments(self, request: BreakoutMethodMomentsRequest) -> ScalarBreakoutRows:
        _validate(self.source, request)
        from increment.readouts._evidence import load_breakout_method_moments

        return load_breakout_method_moments(self.source, request)

    def compliance(self, request: ComplianceRequest) -> ComplianceSummary:
        _validate(self.source, request)
        from increment.readouts._evidence import load_compliance

        return load_compliance(self.source, request)

    def compliance_series(
        self, request: ComplianceSeriesRequest
    ) -> tuple[tuple[object, ComplianceSummary], ...]:
        _validate(self.source, request)
        from increment.readouts._evidence import load_compliance_series

        return load_compliance_series(self.source, request)

    def counts(self, request: CountsRequest) -> CountEvidence:
        _validate(self.source, request)
        from increment.readouts._evidence import load_counts

        return load_counts(self.source, request)

    def capture_sequential(self, request: SequentialCaptureRequest) -> SequentialCapture:
        _validate(self.source, request)
        from increment.readouts._evidence import load_sequential_capture

        return load_sequential_capture(self.source, request)

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        _validate(self.source, request)
        from increment.readouts._evidence import load_evidence_scope

        return load_evidence_scope(self.source, request)

    def breakout_evidence(
        self, request: BreakoutSlicesRequest | BreakoutUnitRowsRequest
    ) -> BreakoutSlices | BreakoutUnitRows:
        _validate(self.source, request)
        from increment.readouts._evidence import load_breakout_evidence

        return load_breakout_evidence(self.source, request)

    def day_axis(self, request: DayAxisRequest) -> DayAxisRows:
        _validate(self.source, request)
        from increment.readouts._evidence import load_day_axis_rows

        return load_day_axis_rows(self.source, request)

    def metric_evidence(self, request: MetricEvidenceRequest) -> MetricEvidence:
        _validate(self.source, request)
        from increment.readouts._metric_rows import load_metric_evidence

        return load_metric_evidence(self.source, request)


def evidence_provider(source: Any) -> EvidenceProvider:
    """Adapt a built-in data source to typed evidence requests."""
    if isinstance(source, EvidenceProvider):
        return source
    return _BuiltInEvidenceProvider(source)


def request_evidence(provider: EvidenceProvider, request: object) -> object:
    """Validate and answer one typed evidence request before data is read."""
    names = {
        BreakoutSlicesRequest: "breakout_evidence",
        BreakoutUnitRowsRequest: "breakout_evidence",
        DayAxisRequest: "day_axis",
        MetricEvidenceRequest: "metric_evidence",
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
    require_evidence(provider, request=request)
    return getattr(provider, method)(request)


__all__ = [
    "BreakoutCell",
    "BreakoutRoute",
    "BreakoutIdentity",
    "BreakoutSlice",
    "BreakoutSlices",
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
    "SequentialCaptureRequest",
    "SourceContext",
    "UnitRows",
    "UnitRowsRequest",
    "evidence_provider",
    "request_evidence",
    "require_evidence",
    "warehouse_cuped_fit_evidence",
]
