from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType, TracebackType
from typing import Literal
from uuid import uuid4

from increment._evidence_types import BreakoutIdentity, EvidenceScope, EvidenceStream
from increment._source_types import EvidenceProvider, _loader_owner, is_custom_source
from increment.kernel.contracts import _refuse

ExecutionMode = Literal["captured", "live"]
EffectiveMode = Literal["captured", "live", "provider"]


@dataclass(slots=True)
class EvidenceExecutionContext:
    """Own one declared evidence scope and lend its pinned provider to readers."""

    provider: EvidenceProvider
    scope: EvidenceScope
    run_id: str
    coherence: ExecutionMode | None = "captured"
    observation_horizon: date | None = None
    _state: str = field(init=False, default="unopened")
    _stack: ExitStack = field(init=False)
    _scoped_provider: EvidenceProvider | None = field(init=False, default=None)
    _source_identity: int = field(init=False)
    _requested_coherence: ExecutionMode = field(init=False)
    _snapshot_manifest: Mapping[str, str] = field(
        init=False, default_factory=lambda: MappingProxyType({})
    )
    _pinned_stream_inventory: Mapping[str, str] = field(
        init=False, default_factory=lambda: MappingProxyType({})
    )
    _stream_resolver: Callable[[EvidenceProvider], tuple[EvidenceStream, ...]] | None = field(
        init=False, default=None
    )
    _effective_modes: Mapping[str, EffectiveMode] = field(
        init=False, default_factory=lambda: MappingProxyType({})
    )
    _breakout_streams: Mapping[BreakoutIdentity, str] = field(
        init=False, default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        self._state = "unopened"
        self._stack = ExitStack()
        self._scoped_provider: EvidenceProvider | None = None
        self._snapshot_manifest: Mapping[str, str] = MappingProxyType({})
        self._breakout_streams = MappingProxyType({})
        self._effective_modes: Mapping[str, EffectiveMode] = MappingProxyType({})
        self._pinned_stream_inventory: Mapping[str, str] = MappingProxyType({})
        requested_coherence = self.coherence
        if requested_coherence not in ("captured", "live"):
            _refuse("kernel.request.invalid", field="coherence")
        self._requested_coherence = requested_coherence
        self._source_identity = id(_loader_owner(self.provider))

    def open(self) -> EvidenceProvider:
        if self._state == "open":
            return self._require_open()
        if self._state != "unopened":
            _refuse("kernel.context.closed")
        self._state = "opening"
        source = _loader_owner(self.provider)
        provider_owned = is_custom_source(source)

        requested = EvidenceScope(
            metrics=self.scope.metrics,
            population=self.scope.population,
            grain=self.scope.grain,
            breakouts=self.scope.breakouts,
            configs=self.scope.configs,
            uptake_facts=self.scope.uptake_facts,
            snapshot=(
                self.scope.snapshot if provider_owned else self._requested_coherence == "captured"
            ),
            execution=True,
            include_breakouts=self.scope.include_breakouts,
        )
        try:
            result = self.provider.scope(requested)
            scoped = self._stack.enter_context(result.source)
            self._stream_resolver = result.stream_resolver
            self._scoped_provider = scoped
            self._breakout_streams = result.breakout_streams
            streams = result.streams or (
                result.stream_resolver(scoped) if result.stream_resolver is not None else ()
            )
            if not streams:
                if provider_owned:
                    streams = (EvidenceStream("provider", "provider"),)
                else:
                    names = {str(getattr(metric, "name", metric)) for metric in requested.metrics}
                    names.update(f"fact:{fact}" for fact in requested.uptake_facts)
                    streams = tuple(EvidenceStream(name, name) for name in sorted(names))
            capture_status = result.capture_status or "live"
            effective_coherence: ExecutionMode | None = None if provider_owned else capture_status
            effective_mode: EffectiveMode = "provider" if provider_owned else capture_status
            self.coherence = effective_coherence
            self._effective_modes = MappingProxyType(
                dict.fromkeys((stream.identity for stream in streams), effective_mode)
            )
            self._pinned_stream_inventory = MappingProxyType(
                {stream.identity: stream.physical for stream in streams}
            )
            self._snapshot_manifest = MappingProxyType(
                {
                    stream.identity: (
                        stream.physical if stream.identity.startswith("artifact:") else uuid4().hex
                    )
                    for stream in streams
                    if effective_mode == "captured"
                }
            )
            self._state = "open"
            return scoped
        except BaseException as error:
            self._state = "closed"
            self._stack.__exit__(type(error), error, error.__traceback__)
            raise

    @property
    def snapshot_manifest(self) -> Mapping[str, str]:
        return self._snapshot_manifest

    @property
    def effective_modes(self) -> Mapping[str, EffectiveMode]:
        return self._effective_modes

    @property
    def requested_coherence(self) -> ExecutionMode:
        return self._requested_coherence

    def validate_source(self, source: object) -> None:
        if id(_loader_owner(source)) != self._source_identity:
            _refuse("kernel.context.undeclared_stream", stream="source_ref")

    def validate_request(
        self, source: object, metrics: Sequence[object], configs: Sequence[object]
    ) -> None:
        self.validate_source(source)
        declared_metrics = {str(getattr(metric, "name", metric)) for metric in self.scope.metrics}
        if any(str(getattr(metric, "name", metric)) not in declared_metrics for metric in metrics):
            _refuse("kernel.context.undeclared_stream", stream="metric")
        declared_configs = set(self.scope.configs or ())
        if any(config not in declared_configs for config in configs):
            _refuse("kernel.context.undeclared_stream", stream="metric_configuration")

    def validate_breakouts(self, requested: Sequence[object]) -> None:
        declared = set(self.scope.breakouts)
        streams = set(self._pinned_stream_inventory)
        artifact_capture = any(stream.startswith("artifact:") for stream in streams)
        provider_capture = "provider" in self._effective_modes.values()
        for identity in requested:
            if identity not in declared:
                _refuse(
                    "kernel.context.undeclared_stream",
                    stream=f"breakout:{getattr(identity, 'property', '')}:{getattr(identity, 'source', None)}",
                )
            if artifact_capture:
                continue
            if provider_capture:
                continue
            stream = self._breakout_streams.get(identity)
            if stream is None or stream not in streams:
                _refuse(
                    "kernel.context.undeclared_stream",
                    stream=f"breakout:{getattr(identity, 'property', '')}:{getattr(identity, 'source', None)}",
                )

    def _require_open(self) -> EvidenceProvider:
        if self._state != "open" or self._scoped_provider is None:
            _refuse("kernel.context.closed")
        return self._scoped_provider

    def close(
        self,
        exc_info: tuple[type[BaseException], BaseException, TracebackType | None] | None = None,
    ) -> None:
        if self._state == "closed":
            return
        self._state = "closed"
        if exc_info is None:
            self._stack.close()
        else:
            self._stack.__exit__(*exc_info)
        self._scoped_provider = None

    @contextmanager
    def borrow(self) -> Iterator[EvidenceProvider]:
        provider = self._require_open()
        if self._snapshot_manifest:
            if self._stream_resolver is None:
                current_inventory = self._pinned_stream_inventory
            else:
                streams = self._stream_resolver(provider)
                current_inventory = MappingProxyType(
                    {stream.identity: stream.physical for stream in streams}
                )
            if current_inventory != self._pinned_stream_inventory:
                stream = next(
                    (
                        name
                        for name in sorted(
                            set(current_inventory).union(self._pinned_stream_inventory)
                        )
                        if current_inventory.get(name) != self._pinned_stream_inventory.get(name)
                    ),
                    next(iter(self._snapshot_manifest)),
                )
                _refuse(
                    "kernel.snapshot.integrity",
                    stream=stream,
                    snapshot_id=self._snapshot_manifest.get(stream, ""),
                )
        yield provider

    @contextmanager
    def evidence_scope(self):
        self.open()
        try:
            with self.borrow() as provider:
                yield provider
        except BaseException as error:
            self.close((type(error), error, error.__traceback__))
            raise
        else:
            self.close()
