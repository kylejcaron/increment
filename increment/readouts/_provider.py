from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from increment._evidence_refusals import require_evidence
from increment._evidence_types import (
    AssignmentEvidence,
    AssignmentRequest,
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    ComplianceSummary,
    CountEvidence,
    CountsRequest,
    EvidenceScope,
    EvidenceScopeResult,
    MethodFitAnswer,
    MethodFitRequest,
    MomentRows,
    MomentsRequest,
    ScalarBreakoutRows,
    SequentialCapture,
    SequentialCaptureRequest,
    SequentialSnapshotRequest,
    UnitRows,
    UnitRowsRequest,
)
from increment._source_types import (
    EvidenceProvider,
    _OwnedLoaderAdapter,
    declared_evidence,
)


def _validate(provider: Any, request: object) -> None:
    require_evidence(provider, request=request)


class _BuiltInEvidenceProvider(_OwnedLoaderAdapter):
    """Present the evidence methods of an existing loader as typed requests."""

    __slots__ = ("source", "evidence", "context")

    def __init__(self, source: Any) -> None:
        self.source = source
        self.evidence = declared_evidence(source)
        self.context = source.context

    def moments(self, request: MomentsRequest) -> MomentRows:
        from increment.readouts._evidence import (
            NATIVE_ROUTE,
            day_axis_route,
            load_builtin_day_moments,
            load_moments,
        )

        if (
            request.grain != "total"
            and request.method is None
            and (
                request.grain not in self.evidence.grains
                or day_axis_route(self.source) == NATIVE_ROUTE
            )
        ):
            return load_builtin_day_moments(self.source, request)
        _validate(self.source, request)
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

    def assignment(self, request: AssignmentRequest) -> AssignmentEvidence:
        """Validate and delegate assignment acquisition to the evidence boundary."""
        _validate(self.source, request)
        from increment.readouts._evidence import load_assignment

        return load_assignment(self.source, request)

    def capture_sequential(self, request: SequentialCaptureRequest) -> SequentialCapture:
        _validate(self.source, request)
        from increment.readouts._evidence import load_sequential_capture

        return load_sequential_capture(self.source, request)

    def sequential_snapshot(self, request: SequentialSnapshotRequest) -> SequentialCapture:
        _validate(self.source, request)
        from increment.readouts._evidence import loader_snapshot

        return loader_snapshot(self.source, previous=request.previous)

    def readout_view(self, source: object, *, metrics: Sequence[object]) -> EvidenceProvider:
        from increment._evidence_scopes import loader_source

        template = loader_source(cast(Any, source))
        view = getattr(self.source, "readout_view", None)
        if callable(view):
            return _BuiltInEvidenceProvider(view(template, metrics=metrics))
        return self

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        _validate(self.source, request)
        from increment._evidence_scopes import load_evidence_scope

        return load_evidence_scope(self.source, request, _BuiltInEvidenceProvider)


def native_scope_provider(source: Any) -> EvidenceProvider:
    """Provide a complete typed adapter for a run-owned built-in loader."""
    return _BuiltInEvidenceProvider(source)
