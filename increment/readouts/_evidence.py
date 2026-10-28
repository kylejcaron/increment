from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal, cast, overload

from increment._evidence_loaders import (
    load_moments,
    load_unit_rows,
    require_request_evidence,
    source_for_population,
)
from increment._evidence_loaders import route_provider_request as _route_provider_request
from increment._evidence_refusals import (
    refuse_joint_grouping,
    require_evidence,
    require_population,
)
from increment._evidence_types import (
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
    MethodFitAnswer,
    MethodFitRequest,
    MomentRows,
    MomentsRequest,
    ScalarBreakoutRows,
    SequentialCapture,
    SequentialCaptureRequest,
    SequentialSnapshotRequest,
    SourceContext,
    UnitRows,
    UnitRowsRequest,
    day_axis_source_partitions,
    day_axis_time_basis,
    validate_compliance_source_match,
)
from increment._source_types import (
    EvidenceCapabilities,
    EvidenceProvider,
    _BuiltinMomentSource,
    compliance_summary_series,
    declared_evidence,
    is_builtin_source,
    is_custom_source,
)
from increment.errors import CodedError
from increment.semantics.models import Breakout
from increment.sources import BreakoutMomentsSource, EvidenceSource, require_operation


def _declared_breakout_property(value: object) -> str | None:
    if isinstance(value, Breakout):
        return value.property
    if isinstance(value, Mapping):
        property_name = value.get("property")
        return property_name if isinstance(property_name, str) else None
    return None


def _declared_breakout_source(value: object) -> str | None:
    if isinstance(value, Breakout):
        return value.source
    if isinstance(value, Mapping):
        source = value.get("source")
        return source if isinstance(source, str) else None
    return None


@dataclass(frozen=True, slots=True)
class DayAxisSlice:
    rows: list[dict[str, Any]]
    metrics: tuple[Any, ...]
    view: Any
    dimension: str | None
    source_name: str | None
    edge_source: Any | None = None
    experiment: Any | None = None


ARTIFACT_ROUTE = "artifact"
NATIVE_ROUTE = "native"
MOMENTS_ROUTE = "moments"
PANEL_ROUTE = "panel"
UNSUPPORTED_ROUTE = "unsupported"
SourceRoute = Literal["artifact", "native", "moments", "panel"]


def is_custom_provider(src: object) -> bool:
    """Whether ``src`` is a custom evidence provider rather than a built-in loader."""
    return is_custom_source(src)


def fit_source(src: EvidenceSource) -> EvidenceSource:
    """Return the loader that owns mixed-covariate CUPED fits for this view."""
    return getattr(src, "_source", src)


def fits_cuped_in_warehouse(src: EvidenceSource) -> bool:
    """Whether the owning loader fits mixed-covariate CUPED models itself."""
    if is_custom_source(src):
        return declared_evidence(src).method_fit == "warehouse"
    return getattr(fit_source(src), "fit_cuped_multicovariate", None) is not None


def retains_cuped_relation(src: EvidenceSource) -> bool:
    """Whether the owning source keeps CUPED fits rather than unit rows."""
    if is_custom_source(src):
        return declared_evidence(src).method_fit in ("retained", "warehouse")
    return getattr(fit_source(src), "cuped_relation", None) is not None


def fits_uncovered_cells_locally(src: EvidenceSource) -> bool:
    """Whether cells without retained or warehouse fits can be fitted from unit rows."""
    return (
        is_custom_source(src) and declared_evidence(src).unit_rows and retains_cuped_relation(src)
    )


def validates_populations(src: EvidenceSource) -> bool:
    """Whether the source can validate requested analysis populations."""
    return is_custom_source(src) or callable(getattr(src, "validate_populations", None))


def validate_populations(
    src: EvidenceSource, populations: Sequence[str], *, operation: str
) -> None:
    """Validate requested populations using declarations or source-owned evidence."""
    if is_custom_source(src):
        for population in populations:
            require_population(src, population)
        return
    cast(Any, src).validate_populations(populations, operation=operation)


def design_allocation(src: EvidenceSource) -> Any:
    """Return the declared design allocation, if the source's design has one."""
    return getattr(getattr(getattr(src, "context", None), "design", None), "allocation", None)


def assigned_breakout_values(src: EvidenceSource) -> Any:
    """Return the source's assigned-breakout-value reader, or None when it has none."""
    reader = getattr(src, "assigned_breakout_values", None)
    return reader if callable(reader) else None


def metric_authenticator(src: EvidenceSource) -> Any:
    """Return the source's metric authenticator for observational quantiles, if any."""
    return getattr(src, "validated_metric", None)


def prior_exclusions(src: EvidenceSource) -> tuple[str, ...]:
    """Metrics whose global prior the source excludes for legacy replay."""
    return tuple(getattr(src, "_legacy_prior_exclusions", ()))


def source_route(src: EvidenceSource) -> SourceRoute:
    evidence = declared_evidence(src)
    if "artifact_experiment" in evidence.operations:
        return ARTIFACT_ROUTE
    if "moments_source" not in evidence.operations:
        return PANEL_ROUTE if evidence.shape == "unit_panel" else MOMENTS_ROUTE
    return NATIVE_ROUTE


def day_axis_route(src: EvidenceSource) -> DayAxisRoute:
    route = source_route(src)
    return MOMENTS_ROUTE if route == PANEL_ROUTE else route


def is_artifact_route(src: EvidenceSource) -> bool:
    return source_route(src) == ARTIFACT_ROUTE


def serves_custom_breakouts(src: EvidenceSource) -> bool:
    """Whether a custom provider serves its declared breakouts through scoped slices."""
    return is_custom_source(src) and bool(declared_evidence(src).breakouts)


def breakout_snapshot_population(
    src: EvidenceSource, population: Literal["assigned", "triggered"]
) -> Literal["assigned", "triggered"]:
    """Population a breakout snapshot pins for the selected source."""
    return population if serves_custom_breakouts(src) else "assigned"


def breakout_route(src: EvidenceSource) -> BreakoutRoute:
    if serves_custom_breakouts(src):
        return PANEL_ROUTE
    route = source_route(src)
    return UNSUPPORTED_ROUTE if route == MOMENTS_ROUTE else route


def require_breakout_identity(src: EvidenceSource, identities: Sequence[BreakoutIdentity]) -> None:
    """Validate breakout identities against the already-selected source declaration."""
    if isinstance(src, BreakoutMomentsSource):
        requested = tuple(f"{item.source}:{item.property}" for item in identities)
        offered = tuple(f"{src.source_name}:{dimension}" for dimension in src.breakouts)
        if any(
            item.property not in src.breakouts
            or (item.source is not None and item.source != src.source_name)
            for item in identities
        ):
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                src,
                requested=requested,
                offered=offered,
                route="use the breakout source declared by this scoped source",
            )
        return

    validator = getattr(src, "_preflight_breakout", None)
    if callable(validator):
        from increment.semantics.models import Breakout

        for identity in identities:
            validator(
                Breakout(property=identity.property, source=identity.source),
                operation="breakout_evidence",
            )
        return

    evidence = declared_evidence(src)
    if any(identity.property not in evidence.breakouts for identity in identities):
        from increment._evidence_refusals import refuse_breakout

        refuse_breakout(
            src,
            requested=tuple(identity.property for identity in identities),
            offered=evidence.breakouts,
            route="use a breakout declared by this source",
        )


@overload
def evidence_scope(
    src: _BuiltinMomentSource, scope: EvidenceScope
) -> AbstractContextManager[_BuiltinMomentSource]: ...


@overload
def evidence_scope(
    src: EvidenceProvider, scope: EvidenceScope
) -> AbstractContextManager[EvidenceProvider]: ...
def evidence_scope(
    src: EvidenceSource, scope: EvidenceScope
) -> AbstractContextManager[EvidenceSource]:
    """Pin the requested source view and reduction lifetime for one readout."""
    return _evidence_scope(src, scope)


@contextmanager
def _evidence_scope(src: EvidenceSource, scope: EvidenceScope) -> Iterator[EvidenceSource]:
    from increment._source_types import is_custom_source

    if is_custom_source(src):
        require_request_evidence(src, scope)
        scoped = src.scope(scope)
        with scoped.source as pinned:
            token = _SCOPE_OWNERS.set((*_SCOPE_OWNERS.get(), (id(pinned), src)))
            try:
                yield pinned
            finally:
                _SCOPE_OWNERS.reset(token)
        return

    with ExitStack() as stack:
        scoped = _builtin_source(src)
        if scope.snapshot and "readout_snapshot" in declared_evidence(scoped).operations:
            scoped = stack.enter_context(
                cast(Any, scoped).readout_snapshot(
                    metrics=scope.metrics,
                    population=scope.population,
                    include_breakouts=scope.include_breakouts or bool(scope.breakouts),
                    configs=scope.configs,
                    uptake_facts=scope.uptake_facts,
                )
            )
        if scope.execution and "readout_execution" in declared_evidence(scoped).operations:
            stack.enter_context(cast(Any, scoped).readout_execution())
        yield scoped


def _builtin_source(src: EvidenceSource) -> _BuiltinMomentSource:
    """Narrow a source that this legacy acquisition path reads structurally."""
    return cast(_BuiltinMomentSource, src)


def load_counts(src: EvidenceSource, request: CountsRequest) -> CountEvidence:
    """Acquire count evidence without changing the requested count grain."""
    if request.kind == "cluster":
        evidence = declared_evidence(src)
        if "cluster" not in evidence.count_grains:
            from increment._evidence_refusals import refuse_cluster

            refuse_cluster(
                src,
                offered=evidence.count_grains,
                route="use a source that retains cluster identities or cluster counts",
            )
        require_request_evidence(src, request)
        routed, answer = _route_provider_request(src, request)
        if routed:
            return answer
        source = source_for_population(_builtin_source(src), request.population)
        unit_counts = (
            source.unit_counts() if "unit" in declared_evidence(source).count_grains else {}
        )
        return CountEvidence(request.population, "cluster", source.cluster_counts(), unit_counts)
    require_request_evidence(src, request)
    routed, answer = _route_provider_request(src, request)
    if routed:
        return answer
    evidence = declared_evidence(src)
    if request.kind == "triggered" or (
        request.population == "triggered"
        and callable(getattr(src, "triggered_counts", None))
        and not (
            request.kind == "assignment"
            and "artifact_experiment" in declared_evidence(src).operations
        )
    ):
        if request.kind != "triggered":
            require_population(src, request.population)
        grain, counts, units = cast(Any, src).triggered_counts()
        if request.kind == "unit":
            return CountEvidence(
                request.population,
                "unit",
                counts if grain == "unit" else units,
                units,
            )
        if request.kind == "assignment":
            return CountEvidence(request.population, "assignment", counts)
        return CountEvidence("triggered", grain, counts, units)

    if request.kind == "assignment":
        counts_method = getattr(src, "assignment_counts", None)
        if callable(counts_method):
            counts = counts_method(population=request.population)
        else:
            source = source_for_population(_builtin_source(src), request.population)
            if "cluster" in declared_evidence(source).count_grains or getattr(
                getattr(source, "context", None), "cluster", None
            ):
                counts = cast(Any, source).cluster_counts()
            else:
                counts = source.unit_counts()
        return CountEvidence(request.population, "assignment", counts)
    source = source_for_population(_builtin_source(src), request.population)
    if request.kind == "unit":
        return CountEvidence(request.population, "unit", source.unit_counts())
    grain, counts, units = cast(Any, source).triggered_counts()
    return CountEvidence("triggered", grain, counts, units)


def load_method_fit(src: EvidenceSource, request: MethodFitRequest) -> MethodFitAnswer:
    """Acquire retained or warehouse method-fit evidence, never fitting locally."""
    owner = src if is_custom_source(src) else getattr(src, "_source", src)
    require_request_evidence(src, request, declaration_source=owner)
    routed, answer = _route_provider_request(src, request)
    if routed:
        return answer
    if request.cell is None and declared_evidence(src).method_fit == "retained":
        evidence = cast(Any, src).categorical_evidence(request.metric, request.method)
        if evidence is not None:
            return MethodFitAnswer(
                origin="retained_categorical",
                cells=evidence.cells,
                evidence_sha256=evidence.payload_sha256,
            )

    fit_source = (
        getattr(src, "_source", src)
        if request.cell is not None
        else src
        if getattr(src, "cuped_relation", None) is not None
        else getattr(src, "_source", None) or src
    )
    origin = (
        "retained_fitted" if getattr(fit_source, "cuped_relation", None) is None else "warehouse"
    )
    fit = getattr(fit_source, "fit_cuped_multicovariate", None)
    if fit is None:
        return MethodFitAnswer(origin="unavailable")

    request_identity: tuple[object, ...] = ()
    if request.cell is None:
        from increment._canonical import canonical_json_bytes
        from increment._source_identity import source_identity

        request_identity = (
            "warehouse-cuped-fit-v1",
            request.metric.name,
            request.population,
            source_identity(fit_source),
            canonical_json_bytes(request.method.model_dump(mode="json", exclude={"name"})).decode(),
            request.control,
            request.arm_order,
            None,
            None,
            None,
            False,
        )
    kwargs = {
        "control": request.control,
        "arm_order": request.arm_order,
        "population": request.population,
    }
    if request.cell is not None:
        kwargs.update(
            breakout_dimension=request.cell.breakout.property,
            breakout_source=request.cell.breakout.source,
            breakout_value=request.cell.value,
            breakout_is_null=request.cell.is_null,
        )
    try:
        identity, envelope = fit(request.metric, request.method, **kwargs)
    except CodedError as exc:
        return MethodFitAnswer(
            origin=origin,
            failure=exc,
            fit_identity=request_identity,
        )
    if isinstance(envelope, CodedError):
        return MethodFitAnswer(
            origin=origin,
            failure=envelope,
            fit_identity=tuple(identity),
        )
    return MethodFitAnswer(
        origin=origin,
        envelope=envelope,
        fit_identity=tuple(identity),
    )


def load_sequential_capture(
    src: EvidenceSource, request: SequentialCaptureRequest
) -> SequentialCapture:
    """Capture a source snapshot: built-in loaders here, custom providers by their own request."""
    if is_custom_source(src):
        # The provider owns its horizon preconditions, continuation proof and freezes.
        require_request_evidence(src, request)
        return cast(Any, src).capture_sequential(request)
    from increment.sequential_source import adopt_source_snapshot, source_snapshot
    from increment.sequential_state import declare_sequential_freeze, sequential_refuse

    capture = getattr(src, "capture_sequential", None)
    if capture is None:
        if not request.finalized or request.as_of is not None:
            sequential_refuse(
                "source.invalid", "this source requires its construction-time finalized checkpoint"
            )
        snapshot = source_snapshot(src, previous=request.previous)
    else:
        if request.as_of is None:
            sequential_refuse("source.invalid", "capture requires an explicit as_of horizon")
        snapshot = capture(
            finalized=request.finalized,
            previous=request.previous,
            as_of=request.as_of,
        )
    if request.freeze:
        snapshot = adopt_source_snapshot(
            src, declare_sequential_freeze(snapshot, tuple(request.freeze))
        )
    return snapshot


def current_source_snapshot(
    src: EvidenceSource, *, previous: SequentialCapture | None = None
) -> SequentialCapture:
    """Request the provider-owned current checkpoint, linked to its optional parent.

    A view yielded by an entered custom scope, built-in or not, defers to that provider.
    """
    owner = _scope_owner(src)
    if is_custom_source(owner):
        method = getattr(owner, "sequential_snapshot", None)
        if method is None:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "continuation.legacy",
                "implement sequential_snapshot(SequentialSnapshotRequest)",
            )
        return method(SequentialSnapshotRequest(previous=previous))
    return loader_snapshot(src, previous=previous)


def loader_snapshot(
    src: EvidenceSource, *, previous: SequentialCapture | None = None
) -> SequentialCapture:
    """Read a built-in loader's own retained checkpoint."""
    from increment.sequential_source import source_snapshot

    return source_snapshot(src, previous=previous)


# Entered custom scopes, innermost last: (id of the yielded view, provider that owns it).
# Keyed by identity so providers need be neither hashable nor weak-referenceable.
_SCOPE_OWNERS: ContextVar[tuple[tuple[int, Any], ...]] = ContextVar("_SCOPE_OWNERS", default=())


def _scope_owner(src: Any) -> Any:
    """Return the provider whose entered scope yielded ``src``, following nested scopes."""
    owner = src._snapshot_provider if isinstance(src, _BreakoutScopedProvider) else src
    for view_id, scope_owner in reversed(_SCOPE_OWNERS.get()):
        if view_id == id(owner) and scope_owner is not owner:
            owner = scope_owner
    return owner


@contextmanager
def _scope_source(src: EvidenceSource, request: EvidenceScope) -> Iterator[EvidenceSource]:
    with evidence_scope(src, request) as scoped:
        yield scoped


class _BreakoutScopedProvider:
    """Bind each custom breakout slice's primitive reads to its identity scope."""

    __slots__ = (
        "provider",
        "identity",
        "population",
        "configs",
        "_scoped",
        "_snapshot_provider",
    )

    def __init__(
        self,
        provider: EvidenceProvider,
        identity: BreakoutIdentity,
        population: Literal["assigned", "triggered"],
        configs: tuple[object, ...] | None = None,
        *,
        scoped: bool = False,
        snapshot_provider: Any | None = None,
    ) -> None:
        self.provider = provider
        self.identity = identity
        self.population = population
        self.configs = configs
        self._scoped = scoped
        self._snapshot_provider = provider if snapshot_provider is None else snapshot_provider

    @property
    def evidence(self) -> EvidenceCapabilities:
        return self.provider.evidence

    @property
    def context(self) -> SourceContext:
        return self.provider.context

    def _scope_request(self, request: object) -> EvidenceScope:
        metric = getattr(request, "metric", None)
        metrics = getattr(request, "metrics", ())
        if not metrics and metric is not None:
            metrics = (metric,)
        requested_breakouts = getattr(request, "breakouts", ())
        conflicting = tuple(
            identity
            for identity in requested_breakouts
            if identity.property == self.identity.property
            and identity.source is not None
            and self.identity.source is not None
            and identity.source != self.identity.source
        )
        if conflicting:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self.provider,
                tuple(f"{item.source}:{item.property}" for item in conflicting),
                (f"{self.identity.source}:{self.identity.property}",),
                route="use the breakout identity bound to this slice",
            )
        return EvidenceScope(
            metrics=metrics,
            population=cast(
                Literal["assigned", "triggered"],
                getattr(request, "population", self.population),
            ),
            grain=getattr(request, "grain", None),
            breakouts=(self.identity,),
            configs=getattr(request, "configs", None) or self.configs,
        )

    def _run(self, request: object, method: str) -> object:
        require_evidence(self.provider, request=request)

        def read(pinned: EvidenceSource) -> object:
            if is_builtin_source(pinned):
                # Built-in loaders validate against the loader that owns each operation.
                loaders: dict[str, Any] = {
                    "moments": load_moments,
                    "unit_rows": load_unit_rows,
                    "method_fit": load_method_fit,
                    "breakout_method_moments": load_breakout_method_moments,
                    "compliance": load_compliance,
                    "compliance_series": load_compliance_series,
                    "counts": load_counts,
                    "capture_sequential": load_sequential_capture,
                }
                return loaders[method](pinned, request)
            require_request_evidence(pinned, request)
            return getattr(pinned, method)(request)

        if self._scoped:
            return read(self.provider)
        scoped = self.provider.scope(self._scope_request(request))
        with scoped.source as pinned:
            return read(pinned)

    def moments(self, request: MomentsRequest) -> MomentRows:
        return cast(MomentRows, self._run(request, "moments"))

    def unit_rows(self, request: UnitRowsRequest) -> UnitRows:
        return cast(UnitRows, self._run(request, "unit_rows"))

    def method_fit(self, request: MethodFitRequest) -> MethodFitAnswer:
        return cast(MethodFitAnswer, self._run(request, "method_fit"))

    def breakout_method_moments(self, request: BreakoutMethodMomentsRequest) -> ScalarBreakoutRows:
        return cast(ScalarBreakoutRows, self._run(request, "breakout_method_moments"))

    def compliance(self, request: ComplianceRequest) -> ComplianceSummary:
        return cast(ComplianceSummary, self._run(request, "compliance"))

    def compliance_series(
        self, request: ComplianceSeriesRequest
    ) -> tuple[tuple[object, ComplianceSummary], ...]:
        return cast(
            tuple[tuple[object, ComplianceSummary], ...],
            self._run(request, "compliance_series"),
        )

    def counts(self, request: CountsRequest) -> CountEvidence:
        return cast(CountEvidence, self._run(request, "counts"))

    def capture_sequential(self, request: SequentialCaptureRequest) -> SequentialCapture:
        return cast(SequentialCapture, self._run(request, "capture_sequential"))

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        conflicting = tuple(
            identity
            for identity in request.breakouts
            if identity.property == self.identity.property
            and identity.source is not None
            and self.identity.source is not None
            and identity.source != self.identity.source
        )
        if conflicting:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                self.provider,
                tuple(f"{item.source}:{item.property}" for item in conflicting),
                (f"{self.identity.source}:{self.identity.property}",),
                route="use the breakout identity bound to this slice",
            )
        scope = EvidenceScope(
            metrics=request.metrics,
            population=request.population,
            grain=request.grain,
            breakouts=tuple(
                dict.fromkeys(
                    (
                        self.identity,
                        *(
                            identity
                            for identity in request.breakouts
                            if identity.property != self.identity.property
                        ),
                    )
                )
            ),
            configs=request.configs if request.configs is not None else self.configs,
            uptake_facts=request.uptake_facts,
            snapshot=request.snapshot,
            execution=request.execution,
            include_breakouts=request.include_breakouts,
        )
        scoped = self.provider.scope(scope)

        @contextmanager
        def source() -> Iterator[EvidenceSource]:
            with scoped.source as pinned:
                if is_builtin_source(pinned):
                    yield pinned
                else:
                    yield cast(
                        EvidenceSource,
                        _BreakoutScopedProvider(
                            cast(EvidenceProvider, pinned),
                            self.identity,
                            self.population,
                            self.configs,
                            scoped=True,
                            snapshot_provider=self._snapshot_provider,
                        ),
                    )

        return EvidenceScopeResult(source(), scope)


def load_evidence_scope(src: EvidenceSource, request: EvidenceScope) -> EvidenceScopeResult:
    """Return a context-managed scoped view, closing source resources on exit."""
    require_request_evidence(src, request)
    scope = request
    if (
        not is_custom_source(src)
        and request.snapshot
        and request.include_breakouts
        and request.population == "triggered"
    ):
        scope = EvidenceScope(
            metrics=request.metrics,
            population="assigned",
            grain=request.grain,
            breakouts=request.breakouts,
            configs=request.configs,
            uptake_facts=request.uptake_facts,
            snapshot=request.snapshot,
            execution=request.execution,
            include_breakouts=request.include_breakouts,
        )
    return EvidenceScopeResult(_scope_source(src, scope), scope)


@overload
def load_breakout_evidence(
    src: EvidenceSource, request: BreakoutSlicesRequest
) -> BreakoutSlices: ...


@overload
def load_breakout_evidence(
    src: EvidenceSource, request: BreakoutUnitRowsRequest
) -> BreakoutUnitRows: ...


def load_breakout_evidence(
    src: EvidenceSource, request: BreakoutSlicesRequest | BreakoutUnitRowsRequest
) -> BreakoutSlices | BreakoutUnitRows:
    """Acquire one typed breakout evidence request."""
    require_request_evidence(src, request)
    if isinstance(request, BreakoutUnitRowsRequest):
        identities = (request.breakout,)
    else:
        identities = request.breakouts
    require_breakout_identity(src, identities)
    if is_custom_source(src):
        if isinstance(request, BreakoutUnitRowsRequest):
            scope = EvidenceScope(
                metrics=(request.metric,),
                population=request.population,
                breakouts=(request.breakout,),
            )
            with src.scope(scope).source as pinned:
                require_request_evidence(pinned, request)
                if is_builtin_source(pinned):
                    return load_breakout_evidence(pinned, request)
                covariates = tuple(dict.fromkeys((request.breakout.property, *request.covariates)))
                frame = load_unit_rows(
                    pinned,
                    UnitRowsRequest(
                        metric=request.metric,
                        covariates=covariates,
                        population=request.population,
                    ),
                )
            return BreakoutUnitRows(request.breakout, PANEL_ROUTE, frame, True)
        definitions = request.breakout_definitions
        return BreakoutSlices(
            PANEL_ROUTE,
            tuple(
                BreakoutSlice(
                    identity,
                    _BreakoutScopedProvider(src, identity, request.population, request.configs),
                    identity.source,
                    identity.source
                    if identity.source is not None or index >= len(definitions)
                    else definitions[index].source,
                )
                for index, identity in enumerate(request.breakouts)
            ),
        )
    if isinstance(request, BreakoutUnitRowsRequest):
        source = source_for_population(_builtin_source(src), request.population)
        declaration = declared_evidence(source)
        if declaration.kind == "breakout" and declaration.shape is None:
            frame = source.unit_frame(request.metric, covariates=request.covariates)
            dimension_in_frame = False
        else:
            covariates = tuple(dict.fromkeys((request.breakout.property, *request.covariates)))
            frame = source.unit_frame(request.metric, covariates=covariates)
            dimension_in_frame = True
        return BreakoutUnitRows(request.breakout, breakout_route(src), frame, dimension_in_frame)

    route = breakout_route(src)
    if route == ARTIFACT_ROUTE:
        definitions = request.breakout_definitions
        if not definitions:
            from increment.semantics.models import Breakout

            definitions = tuple(
                Breakout(property=item.property, source=item.source) for item in request.breakouts
            )
        source = cast(Any, src)
        sources = source.breakout_sources(
            definitions,
            metrics=list(request.metrics),
            population=request.population,
            configs=request.configs,
        )
        slices = tuple(
            BreakoutSlice(identity, scoped, None, scoped.source_name)
            for identity, scoped in zip(request.breakouts, sources, strict=True)
        )
    elif route == PANEL_ROUTE:
        slices = tuple(BreakoutSlice(identity, src, None, None) for identity in request.breakouts)
    elif route == NATIVE_ROUTE:
        declaration = declared_evidence(src)
        if "breakout_sources" not in declaration.operations:
            from increment._evidence_refusals import refuse_breakout

            refuse_breakout(
                src,
                tuple(item.property for item in request.breakouts),
                declaration.breakouts,
                route=(
                    "run_breakout() needs a native Analysis.from_definitions or "
                    "Analysis.from_unit_panel instance with declared breakouts"
                ),
            )
        route_message = (
            "run_breakout() needs a native Analysis.from_definitions or "
            "Analysis.from_unit_panel instance with declared breakouts"
        )
        if not isinstance(src.context, SourceContext):
            from increment._evidence_refusals import refuse_operation

            refuse_operation(
                src, "breakout_sources", tuple(sorted(declaration.operations)), route=route_message
            )
        native = cast(Any, require_operation(src, "breakout_sources"))
        definitions = request.breakout_definitions
        if not definitions:
            from increment.semantics.models import Breakout

            definitions = tuple(
                Breakout(property=item.property, source=item.source) for item in request.breakouts
            )
        sources = native.breakout_sources(
            definitions,
            metrics=list(request.metrics),
            population=request.population,
            configs=request.configs,
        )
        slices = tuple(
            BreakoutSlice(
                identity,
                scoped,
                scoped.source_name
                if sum(item.property == identity.property for item in request.breakouts) > 1
                else None,
                scoped.source_name,
            )
            for identity, scoped in zip(request.breakouts, sources, strict=True)
        )
    else:
        slices = ()
    return BreakoutSlices(route, slices)


def _groups(req: DayAxisRequest) -> list[tuple[Any, tuple[Any, ...]]]:
    if req.caller in ("run_daily", "run_asof"):
        return [
            (
                "asof"
                if req.grain == "asof"
                else ("cohort" if day_axis_time_basis(metric, req.grain) == "cohort" else "daily"),
                (metric,),
            )
            for metric in req.metrics
        ]
    if req.grain == "asof":
        return [("asof", tuple(req.metrics))] if req.metrics else []
    activity = tuple(
        metric for metric in req.metrics if day_axis_time_basis(metric, req.grain) == "calendar"
    )
    retention = tuple(
        metric for metric in req.metrics if day_axis_time_basis(metric, req.grain) == "cohort"
    )
    groups = []
    if activity:
        groups.append(("daily", activity))
    if retention:
        groups.append(("cohort", retention))
    return groups


def _include_covariate(req: DayAxisRequest, metric: Any) -> bool:
    return metric.name in req.covariate_metrics


def _metric_kwargs(metric: Any, req: DayAxisRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"grain": req.grain}
    if req.grain == "asof":
        kwargs["completed_windows_only"] = req.completed_windows_only
    if _include_covariate(req, metric):
        kwargs["include_covariate"] = True
    return kwargs


def _rows(
    src: EvidenceSource,
    req: DayAxisRequest,
    *,
    validate_daily_readout: bool = False,
) -> Iterator[DayAxisSlice]:
    by = req.by or ((req.dimension,) if req.dimension is not None else ())
    for view, metrics in _groups(req):
        rows: list[dict[str, Any]] = []
        for metric in metrics:
            routed, answer = _route_provider_request(
                src,
                MomentsRequest(
                    metric=metric,
                    grain=req.grain,
                    by=by,
                    include_covariate=_include_covariate(req, metric),
                    completed_windows_only=req.completed_windows_only,
                    population=req.population,
                ),
            )
            if routed:
                rows.extend(cast(Iterable[dict[str, Any]], answer))
                continue
            if validate_daily_readout:
                _validate_daily_readout(src, req, metric)
            rows.extend(
                cast(
                    "Iterable[dict[str, Any]]",
                    _builtin_source(src).moments(metric, by=by, **_metric_kwargs(metric, req)),
                )
            )
        yield DayAxisSlice(rows, metrics, view, req.dimension, None)


def _validate_daily_readout(src: EvidenceSource, req: DayAxisRequest, metric: Any) -> None:
    if req.caller not in ("run_daily", "run_daily_lift") or req.grain != "daily":
        return
    from increment._readout_request import ReadoutRequest, validate_request

    configs = {config.metric.name: config for config in src.context.configs}
    readout_request = ReadoutRequest.from_source(
        src,
        metrics=(metric,),
        configs=(configs[metric.name],),
        view="daily",
        grain="daily",
        by=req.by,
    )
    validate_request(readout_request)


def _trigger_edge_source(src: Any) -> Any:
    owner = _scope_owner(src)
    for root in (src, owner):
        if root is None:
            continue
        if callable(getattr(root, "triggered_observation_edges", None)):
            return root
        adapter = getattr(root, "adapter", None)
        source = getattr(root, "source", None)
        for candidate in (source, getattr(adapter, "source", None)):
            if callable(getattr(candidate, "triggered_observation_edges", None)):
                return candidate
    return src


def _trigger_metadata_source(src: Any) -> Any | None:
    owner = _scope_owner(src)
    for root in (src, owner):
        if root is None:
            continue
        adapter = getattr(root, "adapter", None)
        for candidate in (root, getattr(root, "source", None), getattr(adapter, "source", None)):
            if candidate is not None and any(
                getattr(candidate, name, None) is not None
                for name in ("artifact_experiment", "experiment", "_experiment")
            ):
                return candidate
    return None


def _trigger_experiment(src: Any) -> Any | None:
    metadata_source = _trigger_metadata_source(src)
    if metadata_source is None:
        return None
    for name in ("artifact_experiment", "experiment", "_experiment"):
        experiment = getattr(metadata_source, name, None)
        if experiment is not None:
            return experiment
    return None


class _MomentsDayAxis:
    __slots__ = ("src", "validate_daily_readout")

    def __init__(self, src: EvidenceSource, *, validate_daily_readout: bool) -> None:
        self.src = src
        self.validate_daily_readout = validate_daily_readout

    def slices(self, req: DayAxisRequest) -> Iterator[DayAxisSlice]:
        yield from _rows(
            self.src,
            req,
            validate_daily_readout=self.validate_daily_readout,
        )


class _ArtifactDayAxis:
    __slots__ = ("src", "experiment")

    def __init__(self, src: EvidenceSource, experiment: Any | None) -> None:
        self.src = src
        self.experiment = experiment

    def slices(self, req: DayAxisRequest) -> Iterator[DayAxisSlice]:
        if req.dimension is None:
            yield from _rows(self.src, req, validate_daily_readout=True)
            return
        matching = [
            breakout
            for breakout in (self.experiment.breakouts if self.experiment else ())
            if breakout.property == req.dimension
        ]
        for breakout in matching:
            for metric in req.metrics:
                result = cast(Any, self.src).breakout_moments(
                    metric, breakout, **_metric_kwargs(metric, req)
                )
                view = (
                    "asof"
                    if req.grain == "asof"
                    else (
                        "cohort" if day_axis_time_basis(metric, req.grain) == "cohort" else "daily"
                    )
                )
                yield DayAxisSlice(
                    list(result.rows), (metric,), view, req.dimension, result.source_name
                )


class _NativeDayAxis:
    __slots__ = ("src", "experiment")

    def __init__(self, src: EvidenceSource, experiment: Any | None) -> None:
        self.src = src
        self.experiment = experiment

    def slices(self, req: DayAxisRequest) -> Iterator[DayAxisSlice]:
        if req.dimension is not None:
            matching = [
                breakout
                for breakout in (self.experiment.breakouts if self.experiment else ())
                if breakout.property == req.dimension
            ]
            for breakout in matching:
                for view, metrics in _groups(req):
                    rows: list[dict[str, Any]] = []
                    source_name: str | None = None
                    for index, metric in enumerate(metrics):
                        result = cast(Any, self.src).breakout_moments(
                            metric, breakout, **_metric_kwargs(metric, req)
                        )
                        rows.extend(result.rows)
                        if index == 0:
                            source_name = result.source_name
                    yield DayAxisSlice(rows, metrics, view, req.dimension, source_name)
            return
        yield from _rows(self.src, req)


def load_day_axis_rows(src: EvidenceSource, request: DayAxisRequest) -> DayAxisRows:
    """Acquire daily/as-of rows using the evidence route declared by ``src``."""
    require_request_evidence(src, request)
    if is_custom_source(src):
        for metric in request.metrics:
            _validate_daily_readout(src, request, metric)
        context_trigger = getattr(src.context, "trigger_name", None)
        dimensions = tuple(
            dict.fromkeys((*request.by, *((request.dimension,) if request.dimension else ())))
        )
        experiment_metadata = request.experiment or _trigger_experiment(src)
        partitions = day_axis_source_partitions(experiment_metadata, dimensions)
        requested_properties = set(dimensions)
        source_properties = tuple(
            (
                source_name,
                tuple(
                    sorted(
                        {
                            property_name
                            for breakout in breakouts
                            if (
                                (property_name := _declared_breakout_property(breakout)) is not None
                                and property_name in requested_properties
                            )
                        }
                    )
                ),
            )
            for source_name, _, breakouts in partitions
        )
        declared_properties = {prop for _, properties in source_properties for prop in properties}
        bound_identity = src.identity if isinstance(src, _BreakoutScopedProvider) else None
        if bound_identity is None:
            if len(declared_properties) > 1 and any(
                set(properties) != declared_properties for _, properties in source_properties
            ):
                refuse_joint_grouping(src, dimensions, source_properties)
        else:
            bound_properties = {
                prop
                for source_name, properties in source_properties
                if source_name == bound_identity.source
                for prop in properties
            }
            if len(declared_properties) > 1 and not declared_properties.issubset(bound_properties):
                refuse_joint_grouping(src, dimensions, source_properties)
            if bound_identity.source is not None:
                bound_partitions = tuple(
                    partition for partition in partitions if partition[0] == bound_identity.source
                )
                if bound_partitions:
                    partitions = bound_partitions
        scopes = tuple(
            (
                source_name,
                EvidenceScope(
                    metrics=request.metrics,
                    population=request.population,
                    grain=request.grain,
                    breakouts=identities,
                ),
            )
            for source_name, identities, _ in partitions
        )
        route = day_axis_route(src)
        for _source_name, scope in scopes:
            require_evidence(src, request=scope)

        def scoped_slices() -> Iterator[DayAxisSlice]:
            for (source_name, scope), (_, _, breakouts) in zip(scopes, partitions, strict=True):
                with evidence_scope(src, scope) as pinned:
                    experiment = request.experiment or _trigger_experiment(pinned)
                    if experiment is not None and bound_identity is not None:
                        breakouts = tuple(
                            breakout
                            for breakout in experiment.breakouts
                            if _declared_breakout_property(breakout) not in requested_properties
                            or bound_identity.source is None
                            or _declared_breakout_source(breakout) == bound_identity.source
                        )
                        experiment = experiment.model_copy(update={"breakouts": breakouts})
                    elif experiment is not None and breakouts:
                        # Preserve declarations discovered only after entering the scope.
                        experiment = experiment.model_copy(update={"breakouts": breakouts})
                    scoped_request = (
                        request
                        if experiment is request.experiment
                        else replace(request, experiment=experiment)
                    )
                    trigger_declared = (
                        experiment is not None and experiment.trigger is not None
                    ) or context_trigger is not None
                    if (
                        request.population == "triggered"
                        and context_trigger is not None
                        and experiment is None
                        and _trigger_metadata_source(pinned) is None
                    ):
                        from increment._evidence_refusals import refuse_operation

                        refuse_operation(
                            pinned,
                            "artifact_experiment",
                            tuple(sorted(declared_evidence(pinned).operations)),
                            route="triggered day-axis filling needs experiment reconstruction metadata",
                        )
                    edge_source = _trigger_edge_source(pinned)
                    if (
                        request.population == "triggered"
                        and trigger_declared
                        and not callable(getattr(edge_source, "triggered_observation_edges", None))
                    ):
                        from increment._evidence_refusals import refuse_operation

                        refuse_operation(
                            pinned,
                            "triggered_observation_edges",
                            tuple(sorted(declared_evidence(pinned).operations)),
                            route="triggered daily readouts need trigger observation edges; "
                            "use Analysis.from_definitions or Analysis.from_unit_day_artifact",
                        )
                    # Custom typed moments can preserve trigger-aware scoped semantics.
                    # Direct loading is reserved for a built-in loader yielded by scope.
                    if is_builtin_source(pinned):
                        delegated = _load_day_axis_rows(pinned, scoped_request)
                        edge_source = _trigger_edge_source(delegated.source.src)
                        for item in delegated.slices:
                            yield DayAxisSlice(
                                item.rows,
                                item.metrics,
                                item.view,
                                item.dimension,
                                item.source_name or source_name,
                                edge_source,
                                experiment,
                            )
                        continue
                    require_request_evidence(pinned, scoped_request)
                    for item in _MomentsDayAxis(
                        pinned, validate_daily_readout=route != NATIVE_ROUTE
                    ).slices(scoped_request):
                        yield DayAxisSlice(
                            item.rows,
                            item.metrics,
                            item.view,
                            item.dimension,
                            item.source_name or source_name,
                            edge_source,
                            experiment,
                        )

        evidence = _MomentsDayAxis(src, validate_daily_readout=route != NATIVE_ROUTE)
        return DayAxisRows(evidence, route, scoped_slices())
    return _load_day_axis_rows(src, request)


def load_builtin_day_moments(src: EvidenceSource, request: MomentsRequest) -> MomentRows:
    """Answer a typed daily or as-of moments request from a built-in loader's day source."""
    experiment = getattr(src, "_experiment", None) or getattr(src, "artifact_experiment", None)
    day_request = DayAxisRequest(
        caller="run_daily_lift" if request.grain == "daily" else "run_asof_lift",
        grain=cast(Literal["daily", "asof"], request.grain),
        metrics=(request.metric,),
        dimension=request.by[0] if len(request.by) == 1 else None,
        by=request.by,
        completed_windows_only=request.completed_windows_only,
        population=request.population,
        experiment=experiment,
        covariate_metrics=(
            frozenset((request.metric.name,)) if request.include_covariate else frozenset()
        ),
    )
    declared_sources = {
        item.source
        for item in (() if experiment is None else experiment.breakouts)
        if item.property in request.by
    }
    if len(declared_sources) > 1:
        from increment._evidence_refusals import refuse_breakout

        # A typed moments request carries no breakout source, so rows would mix sources.
        refuse_breakout(
            src,
            request.by,
            tuple(sorted(str(item) for item in declared_sources)),
            route="request each breakout source through its own breakout scope",
        )
    return tuple(row for item in _load_day_axis_rows(src, day_request).slices for row in item.rows)


def _load_day_axis_rows(src: EvidenceSource, request: DayAxisRequest) -> DayAxisRows:
    require_request_evidence(src, request)
    route = day_axis_route(src)
    if route == NATIVE_ROUTE and (
        request.experiment is not None
        or request.caller in ("run_daily_lift", "run_asof_lift")
        or request.dimension is not None
        or request.population == "triggered"
    ):
        day_source = cast(Any, require_operation(src, "day_source")).day_source(
            metrics=list(request.metrics), population=request.population
        )
        evidence = _NativeDayAxis(day_source, request.experiment)
    elif route == ARTIFACT_ROUTE:
        source = source_for_population(_builtin_source(src), request.population)
        evidence = _ArtifactDayAxis(source, request.experiment)
    else:
        source = (
            src
            if is_custom_source(src)
            else source_for_population(_builtin_source(src), request.population)
        )
        evidence = _MomentsDayAxis(source, validate_daily_readout=route != NATIVE_ROUTE)
    return DayAxisRows(evidence, route, evidence.slices(request))


def load_breakout_method_moments(
    src: EvidenceSource, request: BreakoutMethodMomentsRequest
) -> ScalarBreakoutRows:
    """Acquire the method-specific breakout moments used by segment readouts."""
    from increment._evidence_refusals import refuse_operation

    require_request_evidence(src, request)
    require_breakout_identity(src, (request.breakout,))
    routed, answer = _route_provider_request(src, request)
    if routed:
        return answer
    if request.completed_windows_only:
        refuse_operation(
            src,
            {"breakout_method_moments": True, "completed_windows_only": True},
            ("method-specific breakout moments without window filtering",),
            route="use a source that supports completed-window method evidence",
        )
    method_source = _builtin_source(src)
    if callable(getattr(src, "breakout_source", None)):
        from increment._analysis_config import overlay_configs
        from increment.semantics.models import Breakout

        configs = overlay_configs(
            (request.metric,), src.context.configs, methods=[request.method], prior=None
        )
        method_source = cast(Any, src).breakout_source(
            Breakout(property=request.breakout.property, source=request.breakout.source),
            metrics=(request.metric,),
            configs=configs,
            population=request.population,
        )
    declaration = declared_evidence(method_source).scalar
    if declaration is None or not declaration.breakout_evidence:
        refuse_operation(
            src,
            "breakout_method_moments",
            ("method-specific breakout moments",),
            route="use a source that retains method-specific breakout evidence",
        )
    return cast(Any, method_source).breakout_moments_for_method(
        request.metric, request.method, by=(request.breakout.property,)
    )


def load_compliance(src: EvidenceSource, request: ComplianceRequest) -> ComplianceSummary:
    require_request_evidence(src, request)
    routed, answer = _route_provider_request(src, request)
    if routed:
        return answer
    source = source_for_population(_builtin_source(src), request.population)
    summary = source.compliance_summary(
        request.design,
        as_of=request.as_of,
        completed_windows_only=request.completed_windows_only,
    )
    validate_compliance_source_match(summary, source.context, as_of=request.as_of)
    return summary


def load_compliance_series(
    src: EvidenceSource, request: ComplianceSeriesRequest
) -> tuple[tuple[object, ComplianceSummary], ...]:
    require_request_evidence(src, request)
    routed, answer = _route_provider_request(src, request)
    if routed:
        return answer
    source = source_for_population(_builtin_source(src), request.population)
    summaries = compliance_summary_series(
        source, request.design, completed_windows_only=request.completed_windows_only
    )
    return tuple(summaries.items())
