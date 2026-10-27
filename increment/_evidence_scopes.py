from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from types import MappingProxyType
from typing import Any, cast

from increment._evidence_loaders import require_request_evidence
from increment._evidence_types import (
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
        streams = MappingProxyType(dict.fromkeys(sorted(names), "captured"))
    return tuple(
        EvidenceStream(identity, physical) for identity, physical in sorted(streams.items())
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
        scope,
        stream_resolver=lambda scoped: _scope_streams(
            cast(EvidenceSource, _loader_owner(scoped)), scope
        ),
        capture_status=capture_status,
    )


def loader_source(source: EvidenceSource) -> _BuiltinMomentSource:
    """Unwrap only an explicitly owned adapter to its loader-shaped source."""
    owned = _loader_owner(source)
    if owned is not source:
        return cast(_BuiltinMomentSource, owned)
    if is_builtin_source(source):
        return source
    return cast(_BuiltinMomentSource, source)
