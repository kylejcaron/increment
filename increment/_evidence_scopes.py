from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, cast

from increment._evidence_loaders import require_request_evidence
from increment._evidence_types import (
    BreakoutIdentity,
    EvidenceScope,
    EvidenceScopeResult,
    EvidenceSource,
    EvidenceStream,
)
from increment._source_types import (
    EvidenceProvider,
    _BuiltinMomentSource,
    _loader_owner,
    declared_evidence,
    is_builtin_source,
    is_custom_source,
)


def _artifact_streams(source: object) -> Mapping[str, str] | None:
    manifest = getattr(source, "_manifest", None)
    if manifest is None:
        return None
    artifact_id = getattr(manifest, "artifact_id", None)
    generation_id = getattr(manifest, "generation_id", None)
    if artifact_id is None or generation_id is None:
        return None
    identity = f"artifact:{artifact_id}:{generation_id}"
    return MappingProxyType({identity: identity})


def _scope_streams(source: EvidenceSource, scope: EvidenceScope) -> tuple[EvidenceStream, ...]:
    inventory = getattr(source, "evidence_scope_streams", None)
    streams = inventory(scope) if callable(inventory) else _artifact_streams(source)
    if streams is None:
        names = {str(getattr(metric, "name", metric)) for metric in scope.metrics}
        names.update(f"fact:{fact}" for fact in scope.uptake_facts)
        names.update(
            f"breakout:{breakout.property}:{breakout.source}" for breakout in scope.breakouts
        )
        streams = MappingProxyType(dict.fromkeys(sorted(names), "captured"))
    return tuple(
        EvidenceStream(identity, physical) for identity, physical in sorted(streams.items())
    )


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


def capture_requirements(src: EvidenceSource, requests: Sequence[EvidenceScope]) -> EvidenceScope:
    """Ask a built-in loader to derive its capture from selected readout requests."""
    operation = getattr(loader_source(src), "capture_requirements", None)
    if callable(operation):
        return operation(tuple(requests))
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
        population=requests[0].population if requests else "assigned",
        grain=next((request.grain for request in requests if request.grain is not None), None),
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
            source = stack.enter_context(
                cast(Any, source).readout_snapshot(
                    metrics=scope.metrics,
                    population=scope.population,
                    include_breakouts=scope.include_breakouts or bool(scope.breakouts),
                    breakouts=scope.breakouts,
                    configs=scope.configs,
                    uptake_facts=scope.uptake_facts,
                )
            )
        if scope.execution and "readout_execution" in declared_evidence(source).operations:
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
            population="assigned",
            grain=request.grain,
            dimension=request.dimension,
            estimands=request.estimands,
            breakouts=request.breakouts,
            configs=request.configs,
            uptake_facts=request.uptake_facts,
            snapshot=request.snapshot,
            execution=request.execution,
            include_breakouts=request.include_breakouts,
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
    )


def loader_source(source: EvidenceSource) -> _BuiltinMomentSource:
    """Unwrap only an explicitly owned adapter to its loader-shaped source."""
    owned = _loader_owner(source)
    if owned is not source:
        return cast(_BuiltinMomentSource, owned)
    if is_builtin_source(source):
        return source
    return cast(_BuiltinMomentSource, source)
