"""A minimal custom evidence provider using only Increment's public API."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from pydantic import TypeAdapter

from increment import Metric
from increment.estimators import CapabilityError
from increment.evidence import (
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    ComplianceSummary,
    CountEvidence,
    CountsRequest,
    EvidenceCapabilities,
    EvidenceProvider,
    EvidenceScope,
    EvidenceScopeResult,
    MethodFitAnswer,
    MethodFitRequest,
    MomentsRequest,
    ScalarBreakoutRows,
    SequentialCapture,
    SequentialCaptureRequest,
    SourceContext,
    UnitRows,
    UnitRowsRequest,
    request_evidence,
    require_evidence,
)


class RevenueProvider:
    """Serve total revenue moments and refuse evidence this store lacks."""

    evidence = EvidenceCapabilities(
        kind="moments",
        grains=frozenset({"total"}),
        breakouts=(),
        shape=None,
        operations=frozenset(),
        unit_rows=False,
        raw_outcomes=False,
        method_fit="none",
        compliance_axis="none",
        count_grains=frozenset(),
        populations=frozenset({"assigned"}),
        sequential_capture="none",
        scalar=None,
        available_covariates=None,
    )
    # This example exercises request dispatch only. A provider installed in an
    # Analysis supplies its real SourceContext from its construction metadata.
    context = cast(SourceContext, None)

    def moments(self, request: MomentsRequest) -> tuple[dict[str, object], ...]:
        if request.metric.name != "revenue" or request.grain != "total":
            raise ValueError("This demonstration stores only total revenue moments.")
        return (
            {
                "experiment_id": "custom-demo",
                "metric": "revenue",
                "group_id": "control",
                "n": 3,
                "ref_y": 4.0,
                "cy1": 0.0,
                "cy2": 2.0,
            },
            {
                "experiment_id": "custom-demo",
                "metric": "revenue",
                "group_id": "treatment",
                "n": 3,
                "ref_y": 6.0,
                "cy1": 0.0,
                "cy2": 2.0,
            },
        )

    def breakout_method_moments(self, request: BreakoutMethodMomentsRequest) -> ScalarBreakoutRows:
        raise AssertionError("Shared request validation rejects this before dispatch.")

    def unit_rows(self, request: UnitRowsRequest) -> UnitRows:
        raise AssertionError("Shared request validation rejects this before dispatch.")

    def method_fit(self, request: MethodFitRequest) -> MethodFitAnswer:
        raise AssertionError("Shared request validation rejects this before dispatch.")

    def compliance(self, request: ComplianceRequest) -> ComplianceSummary:
        raise AssertionError("Shared request validation rejects this before dispatch.")

    def counts(self, request: CountsRequest) -> CountEvidence:
        raise AssertionError("Shared request validation rejects this before dispatch.")

    def capture_sequential(self, request: SequentialCaptureRequest) -> SequentialCapture:
        raise AssertionError("Shared request validation rejects this before dispatch.")

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        if request.snapshot:
            require_evidence(self, operation="readout_snapshot")
        if request.execution:
            require_evidence(self, operation="readout_execution")
        if request.include_breakouts:
            require_evidence(self, operation="breakout_sources")
        return EvidenceScopeResult(_no_op_scope(self), request)

    def compliance_series(
        self, request: ComplianceSeriesRequest
    ) -> tuple[tuple[object, object], ...]:
        raise AssertionError("Shared request validation rejects this before dispatch.")


@contextmanager
def _no_op_scope(provider: RevenueProvider) -> Iterator[RevenueProvider]:
    """Keep an immutable provider available for the request's lifetime."""
    yield provider


def run_example() -> dict[str, object]:
    metric = TypeAdapter(Metric).validate_python(
        {"type": "mean", "name": "revenue", "fact": "revenue", "entity": "customer_id"}
    )
    provider: EvidenceProvider = RevenueProvider()
    rows = request_evidence(provider, MomentsRequest(metric))
    assert isinstance(rows, tuple)

    try:
        request_evidence(provider, MomentsRequest(metric, grain="daily"))
    except CapabilityError as error:
        unsupported_code = error.code
    else:
        raise AssertionError("Daily moments were not declared by this provider.")

    from increment import Encouragement, UptakeSpec

    design = Encouragement(control_group="control", uptake=UptakeSpec(fact="clicked"))
    try:
        request_evidence(provider, ComplianceSeriesRequest(design))
    except CapabilityError as error:
        compliance_series_code = error.code
    else:
        raise AssertionError("Compliance-series evidence was not declared by this provider.")

    return {
        "moment_rows": rows,
        "unsupported_code": unsupported_code,
        "compliance_series_code": compliance_series_code,
    }


if __name__ == "__main__":
    print(run_example())
