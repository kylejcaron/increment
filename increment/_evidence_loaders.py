from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from increment._evidence_refusals import refuse_grain, refuse_operation, require_evidence
from increment._evidence_types import (
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    ContrastStatsRequest,
    CountsRequest,
    EvidenceScope,
    EvidenceSource,
    MethodFitRequest,
    MomentRows,
    MomentsRequest,
    SequentialCaptureRequest,
    UnitRows,
    UnitRowsRequest,
)
from increment._source_types import (
    _BuiltinMomentSource,
    declared_evidence,
    is_builtin_source,
    is_custom_source,
)

if TYPE_CHECKING:
    from increment.kernel.context import EvidenceExecutionContext


def view_population(src: EvidenceSource) -> Literal["assigned", "triggered"]:
    """Return the population a source answers by default: triggered only for triggered-only views."""
    populations = declared_evidence(src).populations
    return (
        "triggered" if "triggered" in populations and "assigned" not in populations else "assigned"
    )


def require_request_evidence(
    src: EvidenceSource, request: Any, *, declaration_source: EvidenceSource | None = None
) -> None:
    """Validate against the effective population and parent declaration."""
    require_evidence(declaration_source or src, request=request)


def acquire_contrast_stats(source: Any, request: ContrastStatsRequest) -> Any:
    """Read one declared switchback answer from its source."""
    return source.contrast_stats(request.metric)


def load_contrast_stats(
    source: object,
    request: ContrastStatsRequest,
    *,
    execution_context: EvidenceExecutionContext,
) -> Any:
    """Acquire one switchback answer through the run's borrowed scope."""
    execution_context.validate_request(source, (request.metric,), ())
    with execution_context.borrow() as borrowed:
        return acquire_contrast_stats(borrowed, request)


def route_provider_request(src: object, request: object) -> tuple[bool, Any]:
    """Route a typed request to a custom provider, leaving built-ins on loader methods."""
    if is_builtin_source(src) or not is_custom_source(src):
        return False, None
    method_names = {
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
    method = method_names.get(type(request))
    if method is None:
        return False, None
    require_request_evidence(cast(EvidenceSource, src), request)
    return True, getattr(src, method)(request)


def source_for_population(src: _BuiltinMomentSource, population: str) -> _BuiltinMomentSource:
    """Select an existing built-in population view or acquire the requested one."""
    declaration = declared_evidence(src)
    if population == "triggered" and not (
        "triggered" in declaration.populations and "assigned" not in declaration.populations
    ):
        return cast(Any, src).triggered_source()
    return src


def load_moments(src: EvidenceSource, request: MomentsRequest) -> MomentRows:
    """Acquire aggregate moments through the source's declared method route."""
    require_request_evidence(src, request)
    routed, answer = route_provider_request(src, request)
    if routed:
        return answer
    method = request.method
    if method is not None:
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

    source = source_for_population(cast(_BuiltinMomentSource, src), request.population)
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
        if scalar is not None and scalar.method_evidence:
            return tuple(
                cast(Any, source).moments_for_method(request.metric, method, by=request.by)
            )
        return tuple(source.moments(request.metric, grain="total", by=request.by))

    scalar = declared_evidence(source).scalar
    if scalar is not None and scalar.method_evidence:
        return tuple(cast(Any, source).moments_for_method(request.metric, method, by=request.by))
    refuse_operation(
        source,
        {"method": method.name},
        ("method-specific total rows by breakout",),
        route="use a source that retains method-specific evidence",
    )


def load_unit_rows(src: EvidenceSource, request: UnitRowsRequest) -> UnitRows:
    """Acquire source unit rows for the requested outcome stage."""
    require_request_evidence(src, request)
    routed, answer = route_provider_request(src, request)
    if routed:
        return answer

    source = source_for_population(cast(_BuiltinMomentSource, src), request.population)
    kwargs = {"covariates": request.covariates}
    if request.outcome_stage == "raw":
        kwargs["outcome_stage"] = "raw"
    return cast(Any, source).unit_frame(request.metric, **kwargs)
