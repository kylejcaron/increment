"""Shared refusals for evidence requests a source cannot serve."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal, NoReturn

from increment._assignment_refusals import (
    ASSIGNMENT_INVALID,
    ASSIGNMENT_REPLAY_MISMATCH,
    ASSIGNMENT_SELECTION_UNAVAILABLE,
    ASSIGNMENT_UNAVAILABLE,
)
from increment._assignment_refusals import (
    refuse_assignment_selection as _refuse_assignment_selection,
)
from increment._source_types import PROVIDER_UNAVAILABLE
from increment.errors import CapabilityError, CodedError, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    from increment._source_types import Grain
    from increment.semantics.models import Metric


def _spec(code: str, hazard: str) -> RefusalSpec:
    return RefusalSpec(
        code,
        CapabilityError,
        template=(
            "{constructor} cannot supply requested "
            + hazard
            + " {requested!r}; offered: {offered!r}. {route}"
        ),
    )


def _winsorization_threshold_context(config: object) -> dict[str, object]:
    thresholds: dict[str, dict[str, object]] = {}
    sources: set[str] = set()
    for side in ("lower", "upper"):
        percentile = getattr(config, f"{side}_percentile")
        value = getattr(config, f"{side}_value")
        if percentile is not None:
            source, threshold = "percentile", percentile
        elif value is not None:
            source, threshold = "fixed_value", value
        else:
            continue
        sources.add(source)
        thresholds[side] = {"source": source, "value": threshold}
    return {
        "threshold_source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "thresholds": thresholds,
    }


def _winsorization_request(metric: Metric) -> dict[str, object]:
    config = getattr(metric, "winsorization", None)
    assert config is not None
    return {"metric": metric.name, **_winsorization_threshold_context(config)}


def winsorization_request_payload(metrics: Sequence[Metric]) -> dict[str, object]:
    return {
        "metrics": tuple(
            _winsorization_request(metric)
            for metric in metrics
            if getattr(metric, "winsorization", None) is not None
        )
    }


def refuse_winsorized_day_axis(
    metrics: Sequence[Metric],
    *,
    constructor: str,
    grain: Literal["daily", "asof"] | None,
) -> None:
    affected = tuple(
        metric for metric in metrics if getattr(metric, "winsorization", None) is not None
    )
    if not affected:
        return
    request = winsorization_request_payload(affected)
    axis = grain if grain is not None else "daily/as-of"
    refuse(
        OPERATION_UNAVAILABLE,
        constructor=constructor,
        requested=tuple(metric.name for metric in affected),
        request=request,
        offered=(),
        route=f"use total-grain run() or run_breakout(), or remove winsorization for {axis} evidence",
    )


CONTRAST_UNAVAILABLE = RefusalSpec(
    "facade.analysis.contrast_unavailable",
    CapabilityError,
    template="{method}() is unavailable for switchback contrast evidence.",
)

INVALID_POPULATION = RefusalSpec(
    "facade.analysis.invalid_population",
    InvalidRequestError,
    template="population must be 'assigned' or 'triggered'; got {population!r}",
)


GRAIN_UNAVAILABLE = _spec("evidence.grain_unavailable", "grain")
BREAKOUT_UNAVAILABLE = _spec("evidence.breakout_unavailable", "breakout")
UNIT_ROWS_UNAVAILABLE = _spec("evidence.unit_rows_unavailable", "unit rows")
RAW_OUTCOMES_UNAVAILABLE = _spec("evidence.raw_outcomes_unavailable", "raw outcomes")
CLUSTER_UNAVAILABLE = _spec("evidence.cluster_unavailable", "cluster evidence")
COVARIATE_UNAVAILABLE = _spec("evidence.covariate_unavailable", "covariates")
ASSIGNMENT_IDENTITY_UNAVAILABLE = RefusalSpec(
    "evidence.assignment.identity_unavailable",
    CapabilityError,
    template=(
        "{constructor} cannot supply assignment identity for {operation!r}; "
        "the source retains reduced counts only. {route}"
    ),
    keys=frozenset({"constructor", "operation", "route"}),
)


def _render_operation_unavailable(
    *, constructor: str, requested: object, offered: object, route: str, request: object = None
) -> str:
    _ = request
    return (
        f"{constructor} cannot supply requested operation {requested!r}; "
        f"offered: {offered!r}. {route}"
    )


OPERATION_UNAVAILABLE = RefusalSpec(
    "evidence.operation_unavailable",
    CapabilityError,
    _render_operation_unavailable,
)
JOINT_GROUPING_UNAVAILABLE = RefusalSpec(
    "evidence.joint_grouping_unavailable",
    CapabilityError,
    template=(
        "{constructor} cannot supply joint grouping {requested!r} from separately "
        "declared sources {offered!r}; declare all requested properties on one source"
    ),
    keys=frozenset({"constructor", "requested", "offered"}),
)
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
        ASSIGNMENT_IDENTITY_UNAVAILABLE,
        OPERATION_UNAVAILABLE,
        PROVIDER_UNAVAILABLE,
        JOINT_GROUPING_UNAVAILABLE,
        UNSUPPORTED_REQUEST,
        ASSIGNMENT_UNAVAILABLE,
        ASSIGNMENT_SELECTION_UNAVAILABLE,
        ASSIGNMENT_INVALID,
        ASSIGNMENT_REPLAY_MISMATCH,
    )
}


def _constructor_name(src: object) -> str:
    return src.constructor if isinstance(src, _DeclaredEvidence) else type(src).__name__


def refuse_joint_grouping(src: object, requested: Sequence[str], offered: object) -> NoReturn:
    refuse(
        JOINT_GROUPING_UNAVAILABLE,
        constructor=_constructor_name(src),
        requested=tuple(requested),
        offered=offered,
    )


def refuse_grain(
    src: object, requested: object, offered: Iterable[object], *, route: str
) -> NoReturn:
    refuse(
        GRAIN_UNAVAILABLE,
        constructor=_constructor_name(src),
        requested=requested,
        offered=tuple(offered),
        route=route,
    )


def refuse_breakout(
    src: object, requested: Sequence[str], offered: Sequence[str], *, route: str
) -> NoReturn:
    refuse(
        BREAKOUT_UNAVAILABLE,
        constructor=_constructor_name(src),
        requested=tuple(requested),
        offered=tuple(offered),
        route=route,
    )


def refuse_operation(
    src: object,
    requested: object,
    offered: Sequence[str],
    *,
    route: str,
    request: object | None = None,
) -> NoReturn:
    context: dict[str, object] = {
        "constructor": _constructor_name(src),
        "requested": requested,
        "offered": tuple(offered),
        "route": route,
    }
    if request is not None:
        context["request"] = request
    refuse(OPERATION_UNAVAILABLE, **context)


def refuse_assignment_identity(src: object, *, operation: str) -> NoReturn:
    """Refuse identity-only assignment operations when the source holds counts only."""
    refuse(
        ASSIGNMENT_IDENTITY_UNAVAILABLE,
        constructor=_constructor_name(src),
        operation=operation,
        route=(
            "supply a row-level assignment ledger via Analysis.from_definitions "
            "with experiment.assignment_ledger, or via a FrameLedger"
        ),
    )


def refuse_unsupported_request(src: object, request: object, offered: Sequence[str]) -> NoReturn:
    refuse(
        UNSUPPORTED_REQUEST,
        constructor=_constructor_name(src),
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
    """Validate a request against the source's declared populations."""
    from increment._source_types import declared_evidence

    _require_population(src, population, declared_evidence(src).populations)


def refuse_unit_rows(src: object, *, offered: object, route: str) -> NoReturn:
    refuse(
        UNIT_ROWS_UNAVAILABLE,
        constructor=_constructor_name(src),
        requested=True,
        offered=offered,
        route=route,
    )


def refuse_cluster(src: object, *, offered: Iterable[object], route: str) -> NoReturn:
    refuse(
        CLUSTER_UNAVAILABLE,
        constructor=_constructor_name(src),
        requested="cluster",
        offered=tuple(offered),
        route=route,
    )


def refuse_raw_outcomes(src: object, *, offered: object, route: str) -> NoReturn:
    refuse(
        RAW_OUTCOMES_UNAVAILABLE,
        constructor=_constructor_name(src),
        requested=True,
        offered=offered,
        route=route,
    )


def refuse_assignment(src: object, *, offered: str) -> NoReturn:
    """Refuse an assignment evidence request before source access."""
    from increment._source_types import declared_evidence

    instructions = {
        "definitions": "use Analysis.from_definitions(...) with experiment.assignment_ledger and semantic roles",
        "artifact": "publish retained assignment_counts with from_unit_day_artifact or use from_definitions with an identity ledger",
        "unit_summary": "retain unit and group columns with Analysis.from_unit_summary",
        "unit_panel": "retain a unit-stable group column with Analysis.from_unit_panel",
        "moments": "retain exact assignment_counts when using Analysis.from_moments",
        "switchback": "retain cycle, period and group roles with Analysis.from_switchback_panel",
    }
    guidance = instructions.get(
        declared_evidence(src).kind,
        "declare assignment_relation and implement EvidenceProvider.assignment",
    )
    refuse(
        ASSIGNMENT_UNAVAILABLE,
        constructor=_constructor_name(src),
        offered=offered,
        route=guidance,
    )


def refuse_assignment_selection(src: object, *, requested: object, offered: object) -> NoReturn:
    """Refuse a scope that aggregate assignment evidence cannot select."""
    _refuse_assignment_selection(src, requested=requested, offered=offered)


def refuse_assignment_invalid(reason: str, *, details: object = None) -> NoReturn:
    from increment._assignment_refusals import refuse_assignment_invalid as refuse

    refuse(reason, details=details)


def refuse_assignment_replay_mismatch(*, details: object = None) -> NoReturn:
    from increment._assignment_refusals import refuse_assignment_replay_mismatch as refuse

    refuse(details=details)


def _has_count_method(source: object, kind: str) -> bool:
    from increment._source_types import declared_evidence

    return kind in declared_evidence(source).count_grains


def _require_assignment_capability(src: object, request: object) -> bool:
    from increment._evidence_types import AssignmentRequest, EvidenceScope
    from increment._source_types import declared_evidence

    if isinstance(request, EvidenceScope):
        for assignment_request in request.assignment_requests:
            _require_assignment_capability(src, assignment_request)
        return False
    if not isinstance(request, AssignmentRequest):
        return False
    declaration = declared_evidence(src)
    if declaration.assignment_relation == "none":
        refuse_assignment(src, offered=declaration.assignment_relation)
    if request.population not in declaration.populations:
        _require_population(src, request.population, declaration.populations)
    return True


def _request_requirements(  # noqa: PLR0915
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
        SequentialSnapshotRequest,
        UnitRowsRequest,
    )
    from increment._source_types import declared_evidence, is_builtin_source

    declaration = declared_evidence(src)
    grain = None
    by: Sequence[str] = ()
    unit_rows = raw_outcomes = False
    covariates: Sequence[str] = ()
    population = None
    if isinstance(request, MomentsRequest):
        population, grain, by = request.population, request.grain, request.by
        if (
            request.grain != "total"
            and declaration.kind == "artifact"
            and "cluster" in declaration.count_grains
        ):
            refuse_grain(
                src,
                request.grain,
                ("total",),
                route="use a source that retains the requested evidence",
            )
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
        population = request.population
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
        population = request.population
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
        population = request.population
        if declaration.compliance_axis == "none" or (
            request.as_of is not None and declaration.compliance_axis != "asof"
        ):
            refuse_operation(
                src,
                "compliance",
                ("asof",) if request.as_of is not None else ("total", "asof"),
                route="use a source with the requested compliance axis and population",
            )
    elif isinstance(request, ComplianceSeriesRequest):
        population = request.population
        if declaration.compliance_axis != "asof":
            refuse_operation(
                src,
                "compliance_series",
                ("asof",),
                route="use a source with the requested compliance axis and population",
            )
    elif isinstance(request, CountsRequest):
        population = "triggered" if request.kind == "triggered" else request.population
        if request.kind == "cluster" and "cluster" not in declaration.count_grains:
            refuse_cluster(
                src,
                offered=declaration.count_grains,
                route="use a source that retains cluster identities or cluster counts",
            )
        if not is_builtin_source(src) and request.kind not in declaration.count_grains:
            refuse_operation(
                src,
                "counts",
                tuple(sorted(declaration.count_grains)),
                route="use a source that declares the requested count grain",
            )
        assignment_count_delegates = (
            is_builtin_source(src)
            and request.kind == "assignment"
            and (
                "unit" in declaration.count_grains
                or "cluster" in declaration.count_grains
                or (
                    declaration.kind == "artifact"
                    and "artifact_experiment" in declaration.operations
                )
            )
        )
        builtin_moments_count_refusal = (
            is_builtin_source(src)
            and declaration.kind == "moments"
            and request.kind in ("unit", "assignment")
        )
        if (
            not _has_count_method(src, request.kind)
            and not assignment_count_delegates
            and not builtin_moments_count_refusal
        ):
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
    elif isinstance(request, SequentialSnapshotRequest):
        # The owning loader or provider refuses with its own continuation code.
        pass
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
                    SequentialSnapshotRequest,
                    UnitRowsRequest,
                )
            ),
        )
    return grain, by, unit_rows, raw_outcomes, covariates, population


def check_method_covariates(src: object, metric: str, methods: Sequence[object]) -> None:
    """Validate method covariates before the corresponding evidence read."""
    covariates = tuple(
        dict.fromkeys(
            covariate.name
            for method in methods
            for covariate in (getattr(method, "covariates", None) or ())
        )
    )
    if covariates:
        from increment._source_types import is_builtin_source

        if is_builtin_source(src):
            checker = getattr(src, "_check_method_covariates", None)
            if callable(checker):
                checker(metric, methods)


def _require_evidence(
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
    assignment_request = request is not None and _require_assignment_capability(src, request)
    population = None
    if request is not None and not assignment_request:
        (
            grain,
            by,
            unit_rows,
            raw_outcomes,
            covariates,
            population,
        ) = _request_requirements(src, request)
    from increment._source_types import declared_evidence, is_builtin_source

    evidence = declared_evidence(src)
    if covariates and not evidence.unit_rows:
        refuse(
            COVARIATE_UNAVAILABLE,
            constructor=_constructor_name(src),
            requested=tuple(covariates),
            offered=tuple(sorted(evidence.available_covariates or ())),
            route="use a source that retains the requested covariates",
        )
    if not is_builtin_source(src) and covariates and evidence.available_covariates is not None:
        offered_covariates = set(evidence.available_covariates) | set(evidence.breakouts)
        missing = tuple(name for name in covariates if name not in offered_covariates)
        if missing:
            refuse(
                COVARIATE_UNAVAILABLE,
                constructor=_constructor_name(src),
                requested=missing,
                offered=tuple(sorted(offered_covariates)),
                route="use a source that retains the requested covariates",
            )
    if raw_outcomes and not evidence.raw_outcomes:
        refuse_raw_outcomes(
            src,
            offered=evidence.raw_outcomes,
            route="use a source that retains pre-transform unit outcomes",
        )
    if unit_rows and not evidence.unit_rows:
        refuse_unit_rows(
            src, offered=evidence.unit_rows, route="use a source that retains unit-level rows"
        )
    if cluster and "cluster" not in evidence.count_grains:
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
        if operation != "sql" and "day_source" in evidence.operations:
            from increment._source_types import is_builtin_source

            if not is_builtin_source(src):
                # A provider forwarding to a loader with a day source answers daily and
                # as-of requests through it; built-in loaders keep their own refusals.
                offered = offered | {"daily", "asof"}
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
    if (
        request is not None
        and population is not None
        and not (
            is_builtin_source(src)
            and getattr(request, "kind", None) == "assignment"
            and population == "triggered"
            and evidence.kind == "artifact"
        )
    ):
        _require_population(src, population, evidence.populations)


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
    _require_evidence(
        src,
        request=request,
        grain=grain,
        by=by,
        unit_rows=unit_rows,
        raw_outcomes=raw_outcomes,
        cluster=cluster,
        covariates=covariates,
        operation=operation,
    )


def require_evidence_declaration(
    capabilities: object,
    *,
    constructor: str,
    request: object | None = None,
    grain: Grain | None = None,
    by: Sequence[str] = (),
    unit_rows: bool = False,
    raw_outcomes: bool = False,
    cluster: bool = False,
    covariates: Sequence[str] = (),
    operation: str | None = None,
) -> None:
    """Evaluate declared support without a source read or backend access.

    ``constructor`` names the caller's source in any refusal.
    """
    from increment._source_types import EvidenceCapabilities

    if not isinstance(capabilities, EvidenceCapabilities):
        refuse(PROVIDER_UNAVAILABLE, constructor=type(capabilities).__name__)
    _require_evidence(
        _DeclaredEvidence(capabilities, constructor),
        request=request,
        grain=grain,
        by=by,
        unit_rows=unit_rows,
        raw_outcomes=raw_outcomes,
        cluster=cluster,
        covariates=covariates,
        operation=operation,
    )


def require_any_evidence_route(
    capabilities: object,
    routes: Sequence[Sequence[object]],
    *,
    constructor: str,
) -> tuple[CodedError, ...]:
    """Return route refusals only when every declared alternative is unavailable."""
    failures: list[CodedError] = []
    for route in routes:
        route_failures: list[CodedError] = []
        for request in route:
            try:
                require_evidence_declaration(capabilities, constructor=constructor, request=request)
            except CodedError as refusal:
                route_failures.append(refusal)
        if not route_failures:
            return ()
        failures.extend(route_failures)

    unique: list[CodedError] = []
    for failure in failures:
        if not any(
            existing.code == failure.code and existing.context == failure.context
            for existing in unique
        ):
            unique.append(failure)
    return tuple(unique)


class _DeclaredEvidence:
    __slots__ = ("constructor", "evidence")

    def __init__(self, evidence: object, constructor: str) -> None:
        self.evidence = evidence
        self.constructor = constructor
