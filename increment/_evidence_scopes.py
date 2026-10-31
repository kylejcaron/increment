from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, cast

from increment._evidence_loaders import acquire_contrast_stats, require_request_evidence
from increment._evidence_types import (
    BreakoutIdentity,
    ContrastStatsRequest,
    EvidenceScope,
    EvidenceScopeResult,
    EvidenceSource,
    EvidenceStream,
    evidence_scope_stream_inventory,
    evidence_streams_for_scope,
)
from increment._source_types import (
    EvidenceProvider,
    _BuiltinMomentSource,
    _loader_owner,
    _OwnedLoaderAdapter,
    declared_evidence,
    is_builtin_source,
    is_custom_source,
)
from increment.errors import InvalidRequestError


def _scope_streams(source: EvidenceSource, scope: EvidenceScope) -> tuple[EvidenceStream, ...]:
    return evidence_streams_for_scope(source, scope)


def _scope_breakout_streams(
    source: EvidenceSource, scope: EvidenceScope
) -> Mapping[BreakoutIdentity, str]:
    resolver = getattr(loader_source(source), "evidence_scope_breakout_streams", None)
    if callable(resolver):
        return MappingProxyType(dict(resolver(scope)))
    return MappingProxyType(
        {
            breakout: f"breakout:{breakout.property}:{breakout.source}"
            for breakout in scope.breakouts
        }
    )


def _scope_covariate_streams(
    source: EvidenceSource, scope: EvidenceScope
) -> Mapping[str, tuple[str, ...]]:
    resolver = getattr(loader_source(source), "evidence_scope_covariate_streams", None)
    if callable(resolver):
        return MappingProxyType(dict(resolver(scope)))
    return MappingProxyType({})


def capture_requirements(src: EvidenceSource, requests: Sequence[EvidenceScope]) -> EvidenceScope:
    """Ask a built-in loader to derive its capture from selected readout requests."""
    metrics_by_name = {metric.name: metric for request in requests for metric in request.metrics}
    configs_list = []
    for request in requests:
        for config in request.configs or ():
            if config not in configs_list:
                configs_list.append(config)
    metrics = tuple(metrics_by_name.values())
    configs = tuple(configs_list)
    breakouts = [item for request in requests for item in request.breakouts]
    source = loader_source(src)
    horizon_metrics = (
        tuple(
            dict.fromkeys(
                metric
                for request in requests
                for metric in (request.horizon_metrics or tuple(source.context.metrics))
            )
        )
        if not is_custom_source(src)
        else tuple(
            dict.fromkeys(metric for request in requests for metric in request.horizon_metrics)
        )
    )
    experiment = getattr(source, "_experiment", None) or getattr(
        source, "artifact_experiment", None
    )
    for request in requests:
        if not request.include_breakouts or request.breakouts:
            continue
        definitions = tuple(experiment.breakouts) if experiment is not None else ()
        if definitions:
            breakouts.extend(
                BreakoutIdentity(item.property, item.source)
                for item in definitions
                if request.dimension is None or item.property == request.dimension
            )
        else:
            breakouts.extend(
                BreakoutIdentity(name, None)
                for name in declared_evidence(src).breakouts
                if request.dimension is None or name == request.dimension
            )
    breakouts = tuple(dict.fromkeys(breakouts))
    uptake_facts = list(
        dict.fromkeys(fact for request in requests for fact in request.uptake_facts)
    )
    if not is_custom_source(src):
        design = getattr(getattr(source, "context", None), "design", None)
        uptake = getattr(design, "uptake", None)
        if uptake is not None and any(
            request.grain in {"total", "daily"}
            or (
                request.grain == "asof"
                and (
                    request.estimands is None
                    or bool({"compliance", "late"}.intersection(request.estimands))
                )
            )
            for request in requests
        ):
            if uptake.fact not in uptake_facts:
                uptake_facts.append(uptake.fact)
    return EvidenceScope(
        metrics=metrics,
        horizon_metrics=horizon_metrics,
        population=requests[0].population if requests else "assigned",
        populations=tuple(
            sorted({population for request in requests for population in request.populations})
        ),
        grain=next(
            (
                candidate
                for candidate in ("asof", "daily", "total")
                if any(request.grain == candidate for request in requests)
            ),
            None,
        ),
        dimension=next((request.dimension for request in requests if request.dimension), None),
        estimands=next(
            (request.estimands for request in requests if request.estimands is not None), None
        ),
        breakouts=breakouts,
        configs=configs if any(request.configs is not None for request in requests) else None,
        uptake_facts=tuple(uptake_facts),
        snapshot=any(request.snapshot for request in requests),
        execution=any(request.execution for request in requests),
        include_breakouts=any(request.include_breakouts for request in requests),
        checkpoint_only=any(request.checkpoint_only for request in requests),
        assignment_requests=tuple(
            item for request in requests for item in request.assignment_requests
        ),
    )


# Keyed by identity so providers need be neither hashable nor weak-referenceable.
SCOPE_OWNERS: ContextVar[tuple[tuple[int, Any], ...]] = ContextVar("_SCOPE_OWNERS", default=())


@contextmanager
def scope_source(src: EvidenceSource, scope: EvidenceScope) -> Iterator[EvidenceSource]:
    if is_custom_source(src):
        require_request_evidence(src, scope)
        scoped = src.scope(scope)
        with scoped.source as pinned:
            token = SCOPE_OWNERS.set((*SCOPE_OWNERS.get(), (id(pinned), src)))
            try:
                yield pinned
            finally:
                SCOPE_OWNERS.reset(token)
        return

    with ExitStack() as stack:
        source = loader_source(src)
        if scope.snapshot and "readout_snapshot" in declared_evidence(source).operations:
            snapshot_options: dict[str, Any] = {
                "metrics": scope.metrics,
                "population": scope.population,
                "assignment_requests": scope.assignment_requests,
                "include_breakouts": scope.include_breakouts or bool(scope.breakouts),
                "breakouts": scope.breakouts,
                "configs": scope.configs,
                "uptake_facts": scope.uptake_facts,
            }
            if (
                scope.horizon_metrics
                and "exploratory_source" in declared_evidence(source).operations
            ):
                snapshot_options["horizon_metrics"] = scope.horizon_metrics
            if (
                scope.checkpoint_only
                and scope.execution
                and scope.grain is None
                and scope.dimension is None
                and not scope.breakouts
                and not scope.include_breakouts
                and getattr(source, "supports_retained_snapshot_integrity_reuse", False)
            ):
                snapshot_options["reuse_retained_snapshot_integrity"] = True
            source = stack.enter_context(cast(Any, source).readout_snapshot(**snapshot_options))
        if (
            scope.execution
            and not scope.checkpoint_only
            and "readout_execution" in declared_evidence(source).operations
        ):
            stack.enter_context(cast(Any, source).readout_execution())
        yield source


def load_evidence_scope(
    src: EvidenceSource,
    request: EvidenceScope,
    provider_factory: Callable[[EvidenceSource], object],
) -> EvidenceScopeResult:
    request = capture_requirements(src, (request,))
    require_request_evidence(src, request)
    scope = request
    if not is_custom_source(src) and request.snapshot and request.population == "triggered":
        scope = EvidenceScope(
            metrics=request.metrics,
            horizon_metrics=request.horizon_metrics,
            population="assigned",
            populations=request.populations,
            grain=request.grain,
            dimension=request.dimension,
            estimands=request.estimands,
            breakouts=request.breakouts,
            configs=request.configs,
            uptake_facts=request.uptake_facts,
            snapshot=request.snapshot,
            execution=request.execution,
            include_breakouts=request.include_breakouts,
            assignment_requests=request.assignment_requests,
        )

    evidence = declared_evidence(src)
    capture_status = (
        "captured"
        if evidence.kind == "artifact"
        or (
            request.snapshot
            and (
                evidence.capture == "immutable"
                or (evidence.capture == "snapshot" and "readout_snapshot" in evidence.operations)
            )
        )
        else "live"
    )

    @contextmanager
    def source() -> Iterator[EvidenceProvider]:
        with scope_source(src, scope) as pinned:
            yield cast(EvidenceProvider, provider_factory(pinned))

    return EvidenceScopeResult(
        source(),
        request,
        stream_resolver=lambda scoped: _scope_streams(
            cast(EvidenceSource, _loader_owner(scoped)), request
        ),
        capture_status=capture_status,
        breakout_streams=_scope_breakout_streams(src, scope),
        covariate_streams=_scope_covariate_streams(src, scope),
    )


def switchback_scope_provider(source: object) -> EvidenceProvider:
    """Adapt switchback contrast acquisition to the shared execution scope."""
    return cast(EvidenceProvider, _SwitchbackScopeProvider(_loader_owner(source)))


class _SwitchbackScopeProvider(_OwnedLoaderAdapter):
    __slots__ = ("source",)

    def __init__(self, source: object) -> None:
        self.source = source

    @property
    def provider_owned(self) -> bool:
        return not is_builtin_source(self.source)

    def scope(self, request: EvidenceScope) -> EvidenceScopeResult:
        captured = request.snapshot and is_builtin_source(self.source)
        return EvidenceScopeResult(
            _switchback_scope_source(self.source, request),
            request,
            streams=tuple(
                EvidenceStream(identity, physical)
                for identity, physical in evidence_scope_stream_inventory(
                    EvidenceScope(metrics=request.metrics)
                ).items()
            ),
            capture_status="captured" if captured else "live",
        )


@contextmanager
def _switchback_scope_source(source: object, scope: EvidenceScope) -> Iterator[Any]:
    _validate_switchback_scope(source, scope)
    pinned: dict[str, Any] | None = None
    if scope.snapshot and is_builtin_source(source):
        pinned = {
            metric.name: acquire_contrast_stats(source, ContrastStatsRequest(metric))
            for metric in scope.metrics
        }
    yield _SwitchbackScopedSource(source, scope, pinned)


class _SwitchbackScopedSource(_OwnedLoaderAdapter):
    __slots__ = ("source", "scope", "_pinned")

    def __init__(self, source: object, scope: EvidenceScope, pinned: dict[str, Any] | None) -> None:
        self.source = source
        self.scope = scope
        self._pinned = pinned

    def validate_requests(self, requests: Sequence[ContrastStatsRequest]) -> None:
        _validate_switchback_scope(self.source, self.scope)
        source_metrics = self.source.context.metrics
        if any(
            request.metric not in self.scope.metrics or request.metric not in source_metrics
            for request in requests
        ):
            _refuse_switchback_metric()

    def contrast_stats(self, metric: Any) -> Any:
        request = ContrastStatsRequest(metric)
        self.validate_requests((request,))
        if self._pinned is not None:
            return self._pinned[metric.name]
        return acquire_contrast_stats(self.source, request)


def validate_switchback_requests(
    source: object,
    requests: Sequence[ContrastStatsRequest],
    execution_context: Any,
) -> None:
    """Validate all switchback requests against a borrowed scope before any read."""
    execution_context.validate_source(source)
    execution_context.validate_request(
        source,
        tuple(request.metric for request in requests),
        (),
    )
    scope = execution_context.scope
    if not isinstance(scope, EvidenceScope):
        _refuse_switchback_field("switchback_scope")
    _validate_switchback_scope(source, scope)
    with execution_context.borrow() as borrowed:
        if not isinstance(borrowed, _SwitchbackScopedSource):
            _refuse_switchback_field("switchback_scope")
        borrowed.validate_requests(requests)


def _validate_switchback_scope(source: object, scope: EvidenceScope) -> None:
    if scope.population != "assigned":
        _refuse_switchback_field("switchback_population")
    if scope.grain != "total" or scope.breakouts or scope.configs or scope.uptake_facts:
        _refuse_switchback_field("switchback_scope")
    source_metrics = cast(Any, _loader_owner(source)).context.metrics
    if any(metric not in source_metrics for metric in scope.metrics):
        _refuse_switchback_metric()


def _refuse_switchback_field(field: str) -> None:
    raise InvalidRequestError(
        f"Invalid analysis request: {field}",
        code="kernel.request.invalid",
        context={"field": field},
    )


def _refuse_switchback_metric() -> None:
    raise InvalidRequestError(
        "Execution context does not declare stream 'metric'",
        code="kernel.context.undeclared_stream",
        context={"stream": "metric"},
    )


def loader_source(source: EvidenceSource) -> _BuiltinMomentSource:
    """Unwrap only an explicitly owned adapter to its loader-shaped source."""
    owned = _loader_owner(source)
    if owned is not source:
        return cast(_BuiltinMomentSource, owned)
    if is_builtin_source(source):
        return source
    return cast(_BuiltinMomentSource, source)
