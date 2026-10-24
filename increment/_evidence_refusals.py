"""Shared refusals for evidence requests a source cannot serve."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NoReturn

from increment.errors import CapabilityError, RefusalSpec, refuse

if TYPE_CHECKING:
    from increment._source_types import Grain


def _spec(code: str, hazard: str) -> RefusalSpec:
    return RefusalSpec(
        code,
        CapabilityError,
        template=(
            "{constructor} cannot supply requested "
            + hazard
            + " {requested!r}; offered: {offered!r}. {route}"
        ),
        keys=frozenset({"constructor", "requested", "offered", "route"}),
    )


GRAIN_UNAVAILABLE = _spec("evidence.grain_unavailable", "grain")
BREAKOUT_UNAVAILABLE = _spec("evidence.breakout_unavailable", "breakout")
UNIT_ROWS_UNAVAILABLE = _spec("evidence.unit_rows_unavailable", "unit rows")
RAW_OUTCOMES_UNAVAILABLE = _spec("evidence.raw_outcomes_unavailable", "raw outcomes")
CLUSTER_UNAVAILABLE = _spec("evidence.cluster_unavailable", "cluster evidence")
COVARIATE_UNAVAILABLE = _spec("evidence.covariate_unavailable", "covariates")
OPERATION_UNAVAILABLE = _spec("evidence.operation_unavailable", "operation")
UNSUPPORTED_REQUEST = RefusalSpec(
    "evidence.request_unsupported",
    CapabilityError,
    template=(
        "{constructor} does not support evidence request {requested!r}; supported request "
        "types: {offered!r}. {route}"
    ),
    keys=frozenset({"constructor", "requested", "offered", "route"}),
)
EVIDENCE_REFUSALS = {
    spec.code: spec
    for spec in (
        GRAIN_UNAVAILABLE,
        BREAKOUT_UNAVAILABLE,
        UNIT_ROWS_UNAVAILABLE,
        RAW_OUTCOMES_UNAVAILABLE,
        CLUSTER_UNAVAILABLE,
        COVARIATE_UNAVAILABLE,
        OPERATION_UNAVAILABLE,
        UNSUPPORTED_REQUEST,
    )
}


def refuse_grain(
    src: object, requested: object, offered: Iterable[object], *, route: str
) -> NoReturn:
    refuse(
        GRAIN_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=requested,
        offered=tuple(offered),
        route=route,
    )


def refuse_breakout(
    src: object, requested: Sequence[str], offered: Sequence[str], *, route: str
) -> NoReturn:
    refuse(
        BREAKOUT_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=tuple(requested),
        offered=tuple(offered),
        route=route,
    )


def refuse_operation(
    src: object, requested: object, offered: Sequence[str], *, route: str
) -> NoReturn:
    refuse(
        OPERATION_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=requested,
        offered=tuple(offered),
        route=route,
    )


def refuse_unsupported_request(src: object, request: object, offered: Sequence[str]) -> NoReturn:
    refuse(
        UNSUPPORTED_REQUEST,
        constructor=type(src).__name__,
        requested=type(request).__name__,
        offered=tuple(sorted(offered)),
        route="use one of the supported evidence request types",
    )


def _require_population(src: object, population: str, offered: Iterable[str]) -> None:
    if population not in offered:
        refuse_operation(
            src,
            {"population": population},
            tuple(sorted(offered)),
            route="choose a source for this population",
        )


def require_population(src: object, population: str) -> None:
    """Validate a population only when the source declares its supported populations."""
    from increment._source_types import declared_evidence, evidence_is_declared

    if evidence_is_declared(src):
        _require_population(src, population, declared_evidence(src).populations)


def refuse_unit_rows(src: object, *, offered: object, route: str) -> NoReturn:
    refuse(
        UNIT_ROWS_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=True,
        offered=offered,
        route=route,
    )


def refuse_cluster(src: object, *, offered: Iterable[object], route: str) -> NoReturn:
    refuse(
        CLUSTER_UNAVAILABLE,
        constructor=type(src).__name__,
        requested="cluster",
        offered=tuple(offered),
        route=route,
    )


def refuse_raw_outcomes(src: object, *, offered: object, route: str) -> NoReturn:
    refuse(
        RAW_OUTCOMES_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=True,
        offered=offered,
        route=route,
    )


def _has_count_method(source: object, kind: str) -> bool:
    declaration = getattr(source, "evidence", None)
    if declaration is not None and kind in declaration.count_grains:
        return True
    count_source = getattr(source, "source", source)
    count_method = {
        "unit": "unit_counts",
        "cluster": "cluster_counts",
        "assignment": "assignment_counts",
        "triggered": "triggered_counts",
    }[kind]
    if callable(getattr(count_source, count_method, None)):
        return True
    return kind == "assignment" and any(
        callable(getattr(count_source, name, None)) for name in ("unit_counts", "cluster_counts")
    )


def _request_requirements(
    src: object, request: object
) -> tuple[Grain | None, Sequence[str], bool, bool, Sequence[str], str | None]:
    from increment._evidence_types import (
        BreakoutMethodMomentsRequest,
        BreakoutSlicesRequest,
        BreakoutUnitRowsRequest,
        ComplianceRequest,
        ComplianceSeriesRequest,
        CountsRequest,
        DayAxisRequest,
        EvidenceScope,
        MethodFitRequest,
        MetricEvidenceRequest,
        MomentsRequest,
        SequentialCaptureRequest,
        UnitRowsRequest,
    )
    from increment._source_types import declared_evidence

    declaration = declared_evidence(src)
    grain = None
    by: Sequence[str] = ()
    unit_rows = raw_outcomes = False
    covariates: Sequence[str] = ()
    population = None
    if isinstance(request, MomentsRequest):
        population, grain, by = request.population, request.grain, request.by
    elif isinstance(request, BreakoutMethodMomentsRequest):
        population = request.population
        by = (request.breakout.property,)
        if declaration.scalar is None or not declaration.scalar.breakout_evidence:
            refuse_operation(
                src,
                "breakout_method_moments",
                ("method-specific breakout moments",),
                route="use a source that retains method-specific breakout evidence",
            )
    elif isinstance(request, UnitRowsRequest):
        population = request.population
        unit_rows = True
        raw_outcomes = request.outcome_stage == "raw"
        covariates = request.covariates
    elif isinstance(request, BreakoutSlicesRequest):
        population = request.population
        by = tuple(item.property for item in request.breakouts)
    elif isinstance(request, BreakoutUnitRowsRequest):
        by = (request.breakout.property,)
        unit_rows = True
        covariates = request.covariates
    elif isinstance(request, DayAxisRequest):
        population = request.population
        grain = request.grain
        by = tuple(request.by) or (() if request.dimension is None else (request.dimension,))
    elif isinstance(request, MetricEvidenceRequest):
        by = request.by
        grain = "total"
    elif isinstance(request, MethodFitRequest):
        if declaration.method_fit == "none":
            refuse_operation(
                src,
                "method_fit",
                ("warehouse", "local", "retained"),
                route="use a source that retains method-specific evidence",
            )
        grain = "total"
        population = request.population
        by = () if request.cell is None else (request.cell.breakout.property,)
    elif isinstance(request, ComplianceRequest):
        if declaration.compliance_axis == "none":
            refuse_operation(
                src,
                "compliance",
                ("total", "asof"),
                route="use a source with enrollment and uptake records",
            )
    elif isinstance(request, ComplianceSeriesRequest):
        population = request.population
        if declaration.compliance_axis == "none":
            refuse_operation(
                src,
                "compliance_series",
                (declaration.compliance_axis,),
                route="use a source with the requested compliance axis and population",
            )
    elif isinstance(request, CountsRequest):
        population = "triggered" if request.kind == "triggered" else request.population
        if not _has_count_method(src, request.kind):
            refuse_operation(
                src,
                "counts",
                tuple(sorted(declaration.count_grains)),
                route="use a source that retains the requested counts",
            )
    elif isinstance(request, SequentialCaptureRequest):
        if declaration.sequential_capture == "none":
            refuse_operation(
                src,
                "capture_sequential",
                ("construction", "asof", "replay"),
                route="use a source with sequential history",
            )
    elif isinstance(request, EvidenceScope):
        grain = request.grain
        by = tuple(item.property for item in request.breakouts)
        population = request.population
    else:
        refuse_unsupported_request(
            src,
            request,
            tuple(
                request_type.__name__
                for request_type in (
                    BreakoutMethodMomentsRequest,
                    BreakoutSlicesRequest,
                    BreakoutUnitRowsRequest,
                    ComplianceRequest,
                    ComplianceSeriesRequest,
                    CountsRequest,
                    DayAxisRequest,
                    EvidenceScope,
                    MethodFitRequest,
                    MetricEvidenceRequest,
                    MomentsRequest,
                    SequentialCaptureRequest,
                    UnitRowsRequest,
                )
            ),
        )
    return grain, by, unit_rows, raw_outcomes, covariates, population


def require_evidence(
    src: object,
    *,
    request: object | None = None,
    grain: Grain | None = None,
    by: Sequence[str] = (),
    unit_rows: bool = False,
    raw_outcomes: bool = False,
    cluster: bool = False,
    covariates: Sequence[str] = (),
    operation: str | None = None,
) -> None:
    population = None
    if request is not None:
        (
            grain,
            by,
            unit_rows,
            raw_outcomes,
            covariates,
            population,
        ) = _request_requirements(src, request)
    from increment._source_types import declared_evidence, evidence_is_declared

    evidence = declared_evidence(src)
    declared = evidence_is_declared(src)
    if declared and covariates and not evidence.unit_rows:
        refuse(
            COVARIATE_UNAVAILABLE,
            constructor=type(src).__name__,
            requested=tuple(covariates),
            offered=tuple(sorted(evidence.available_covariates or ())),
            route="use a source that retains the requested covariates",
        )
    if declared and raw_outcomes and not evidence.raw_outcomes:
        refuse_raw_outcomes(
            src,
            offered=evidence.raw_outcomes,
            route="use a source that retains pre-transform unit outcomes",
        )
    if declared and unit_rows and not evidence.unit_rows:
        refuse_unit_rows(
            src, offered=evidence.unit_rows, route="use a source that retains unit-level rows"
        )
    if declared and cluster and "cluster" not in evidence.count_grains:
        refuse_cluster(
            src,
            offered=sorted(evidence.count_grains),
            route="use a source that retains cluster identities or cluster counts",
        )
    if operation == "sql" and not evidence.sql_grains:
        refuse_operation(
            src,
            operation,
            sorted(evidence.operations),
            route="use a source that supports SQL rendering",
        )
    if grain is not None:
        requested = grain
        offered = evidence.sql_grains if operation == "sql" else evidence.grains
        if grain not in offered:
            refuse_grain(
                src,
                requested,
                sorted(offered),
                route="use a source that retains the requested evidence",
            )
    if by and not set(by).issubset(evidence.breakouts):
        refuse_breakout(
            src,
            by,
            evidence.breakouts,
            route="use a source with the requested breakout dimensions",
        )
    if operation is not None and operation != "sql" and operation not in evidence.operations:
        refuse_operation(
            src,
            operation,
            sorted(evidence.operations),
            route="use a source that supports this operation",
        )
    if request is not None and population is not None:
        _require_population(src, population, evidence.populations)
