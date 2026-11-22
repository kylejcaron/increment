from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._canonical import declaration_value
from increment._evidence_refusals import refuse_grain, refuse_operation, require_evidence
from increment._evidence_types import (
    BreakoutMethodMomentsRequest,
    ComplianceRequest,
    ComplianceSeriesRequest,
    ContrastStatsRequest,
    CountsRequest,
    EvidenceScope,
    EvidenceSource,
    HealthObservationEdgeAnswer,
    MethodFitRequest,
    MomentRows,
    MomentsRequest,
    SequentialCaptureRequest,
    SourceContext,
    UnitRows,
    UnitRowsRequest,
)
from increment._source_operations import (
    DashboardGroupDataOperation,
    ExploratorySourceOperation,
    NativeCoreSource,
)
from increment._source_types import (
    MOMENTS_BATCH_OWNERS,
    EvidenceProvider,
    _BuiltinMomentSource,
    declared_evidence,
    is_builtin_source,
    is_custom_source,
    source_class_name,
)
from increment.decision import ContrastContext
from increment.errors import CapabilityError, _ArtifactContractErrorMarker

if TYPE_CHECKING:
    from increment.kernel.context import EvidenceExecutionContext
    from increment.semantics.interaction_scan import (
        CombinationCell,
        CombinationFeed,
        CombinationRegistration,
        FocalSourceFacts,
        InteractionCombination,
        InteractionScanPlan,
        StudyFeedFact,
    )
    from increment.semantics.overlap import OverlapEvidence


def load_moments_batch(
    src: EvidenceSource, requests: Sequence[MomentsRequest]
) -> tuple[MomentRows, ...]:
    """Acquire ordered full-identity requests, retaining per-request provider fallback."""
    for request in requests:
        _validate_moments_request(src, request)
    acquire = MOMENTS_BATCH_OWNERS.get(type(src))
    answers = acquire(src, requests) if acquire is not None else (None,) * len(requests)
    return tuple(
        load_moments(src, request) if answer is None else answer
        for request, answer in zip(requests, answers, strict=True)
    )


def view_population(src: EvidenceSource) -> Literal["assigned", "triggered"]:
    """Return the population a source answers by default: triggered only for triggered-only views."""
    populations = declared_evidence(src).populations
    return (
        "triggered" if "triggered" in populations and "assigned" not in populations else "assigned"
    )


def load_native_trigger_rates(src: EvidenceSource) -> dict[str, float]:
    """Acquire rates only from sources satisfying the legacy native core contract."""
    declaration = declared_evidence(src)
    offered = tuple(sorted(declaration.operations))
    route = (
        "trigger_rates() needs an experiment built by from_definitions with a warehouse connection"
    )
    if not isinstance(getattr(src, "context", None), SourceContext):
        refuse_operation(src, "triggered_counts", offered, route=route)
    if "triggered_counts" not in offered:
        refuse_operation(src, "triggered_counts", offered, route=route)
    if not isinstance(src, NativeCoreSource):
        refuse_operation(
            src,
            "triggered_counts",
            offered,
            route="use a source that implements the requested operation",
        )
    return src.trigger_rates()


def load_triggered_observation_edge(
    src: EvidenceSource, metric: Any
) -> HealthObservationEdgeAnswer:
    """Acquire one maturity edge, retaining only established unavailable failures."""
    try:
        observed, finalized = cast(Any, src).triggered_observation_edges(metric)
    except (CapabilityError, _ArtifactContractErrorMarker) as failure:
        return HealthObservationEdgeAnswer(failure=failure)
    return HealthObservationEdgeAnswer(observed, finalized)


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


def day_axis_grains(src: object) -> frozenset[str]:
    """Return declared day grains plus those the declared day-source operation can acquire."""
    evidence = declared_evidence(src)
    if "day_source" not in evidence.operations:
        return frozenset(evidence.grains)
    return frozenset((*evidence.grains, "daily", "asof"))


def _validate_moments_request(src: EvidenceSource, request: MomentsRequest) -> None:
    """Keep scalar and batch declaration/option refusals on the same boundary."""
    require_request_evidence(src, request)
    if is_custom_source(src):
        return
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


def load_moments(src: EvidenceSource, request: MomentsRequest) -> MomentRows:
    """Acquire aggregate moments through the source's declared method route."""
    _validate_moments_request(src, request)
    routed, answer = route_provider_request(src, request)
    if routed:
        return answer
    method = request.method
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


def load_exploratory_source(
    operation: ExploratorySourceOperation, metrics: Sequence[Any]
) -> EvidenceSource:
    """Acquire a sibling source for declared exploratory metrics."""
    return operation.exploratory_source(metrics=metrics)


def load_dashboard_group_data(
    operation: DashboardGroupDataOperation,
    *,
    metrics: Sequence[Any],
    checkpoints: Mapping[str, Any] | None,
    population: Literal["assigned", "triggered"],
) -> tuple[Any, ...]:
    """Acquire dashboard group evidence through its declared source operation."""
    return tuple(
        operation.dashboard_group_data(
            metrics=metrics, checkpoints=checkpoints, population=population
        )
    )


def load_interaction_focal_facts(
    source: EvidenceSource, plan: InteractionScanPlan
) -> FocalSourceFacts | None:
    """Verify the focal fact projections on the existing pinned definitions source."""
    if not is_builtin_source(source) or declared_evidence(source).kind != "definitions":
        return None
    return cast(Any, source).interaction_focal_facts(plan)


def load_interaction_pair_facts(
    source: EvidenceSource,
    evidence: OverlapEvidence,
    tables: Mapping[str, tuple[str, str, str]],
) -> tuple[StudyFeedFact, ...]:
    """Verify original warehouse evidence against its qualified physical feeds."""
    return cast(Any, source).interaction_pair_facts(evidence, tables)


def load_interaction_combination_cells(
    source: EvidenceSource,
    catalog: str,
    plan: InteractionScanPlan,
    combination: InteractionCombination,
    feed: CombinationFeed,
) -> tuple[CombinationCell, ...]:
    """Reduce qualified joint facts without returning per-unit rows."""
    return cast(Any, source).interaction_combination_cells(catalog, plan, combination, feed)


def load_interaction_combination_registration(
    source: EvidenceSource, catalog: str, registration: CombinationRegistration
) -> bool:
    """Verify the registered joint-comparison commitment on the sealed source."""
    return cast(Any, source).interaction_combination_registered(catalog, registration)


def build_triggered_srm_provider(
    source: EvidenceSource, provider_factory: Callable[[object], EvidenceProvider]
) -> EvidenceProvider:
    """Build the adapter consumed by triggered SRM after caller authorization."""
    return provider_factory(source)


def load_sitewide_evidence(
    operation: Any,
    metric: Any,
    *,
    include_ratio: bool,
    population: Literal["assigned", "triggered"],
) -> Any:
    """Acquire whole-site evidence after request validation."""
    return operation.sitewide_evidence(metric, include_ratio=include_ratio, population=population)


def load_triggered_sitewide_counts(
    source: EvidenceSource, provider_factory: Callable[[object], EvidenceProvider]
) -> Any:
    """Acquire typed triggered counts for whole-site allocation diagnostics."""
    return provider_factory(source).counts(CountsRequest(kind="triggered"))


def load_assignment_diagnostic(source: Any) -> Any:
    """Acquire source-owned switchback assignment-integrity diagnostics."""
    return source.diagnostics


def breakout_readout_source(source: EvidenceSource) -> Any:
    """Select the panel's restricted summary reducer for whole-window breakouts."""
    summary = getattr(source, "summary_breakout_source", None)
    return source if summary is None else summary


def source_context_payload(source: Any) -> dict[str, Any]:
    """Acquire source-owned declaration metadata for static request validation."""

    context = source.context
    if isinstance(context, ContrastContext):
        return {"family": "contrast", "context": context.model_dump(mode="json")}
    experiment = getattr(source, "_experiment", None) or getattr(
        source, "artifact_experiment", None
    )
    identity = getattr(source, "declaration_identity", None)
    return {
        "source_declaration": declaration_value(identity()) if callable(identity) else None,
        "study_id": context.study_id,
        "plan": context.plan,
        "configs": context.configs,
        "cluster": context.cluster,
        "trigger_name": context.trigger_name,
        "source_name": source_class_name(source),
        "experiment": None if experiment is None else experiment.model_dump(mode="json"),
    }
