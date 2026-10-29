from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType, TracebackType
from typing import Any, Literal, cast
from uuid import uuid4

from increment._evidence_types import BreakoutIdentity, EvidenceScope, EvidenceStream
from increment._source_types import (
    EvidenceProvider,
    _loader_owner,
    is_builtin_source,
    is_custom_source,
)
from increment.kernel.contracts import _refuse

ExecutionMode = Literal["captured", "live"]
EffectiveMode = Literal["captured", "live", "provider"]


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


def _source_labels(definitions: Any) -> dict[str, tuple[str, ...]]:
    labels: dict[str, list[str]] = {}
    for fact_source in definitions.fact_sources:
        labels.setdefault(fact_source.sql, []).append(fact_source.name)
    for dimension in definitions.dim_sources:
        labels.setdefault(dimension.sql, []).append(f"dimension:{dimension.name}")
    for exposure in definitions.exposures:
        if exposure.sql is not None:
            labels.setdefault(exposure.sql, []).append(f"exposure:{exposure.name}")
    return {query: tuple(names) for query, names in labels.items()}


def _captured_streams(source: object, definitions: Any) -> Mapping[str, str] | None:
    tables = getattr(getattr(source, "_session", None), "_source_tables", None)
    if tables is None:
        return None
    labels = _source_labels(definitions)
    streams = {
        label: query for query in tables for label in labels.get(query, (f"source:{query}",))
    }
    return MappingProxyType(dict(sorted(streams.items())))


def _declared_fact_names(
    definitions: Any,
    metrics: Sequence[object],
    configs: Sequence[object],
    breakouts: Sequence[object],
    uptake_facts: Sequence[str],
) -> tuple[set[str], set[str]]:
    fact_by_name = {fact_source.name: fact_source for fact_source in definitions.fact_sources}
    dimension_names = {dimension.name for dimension in definitions.dim_sources}
    requested_dimensions: set[str] = set()
    fact_names = set(uptake_facts)
    for metric in metrics:
        numerator = getattr(metric, "numerator", None)
        denominator = getattr(metric, "denominator", None)
        if numerator is not None and denominator is not None:
            fact_names.update((numerator.fact, denominator.fact))
        else:
            fact = getattr(metric, "fact", None)
            if fact is not None:
                fact_names.add(fact)
    for config in configs:
        for covariate in getattr(config, "covariates", ()):
            measure = getattr(covariate, "measure", None)
            if measure is not None:
                fact_names.add(measure.fact)
            declared_source = getattr(covariate, "source", None)
            property_name = getattr(covariate, "property", None) or getattr(covariate, "name", None)
            _add_declared_source(
                definitions,
                declared_source,
                property_name,
                fact_by_name,
                dimension_names,
                requested_dimensions,
                fact_names,
            )
    for breakout in breakouts:
        _add_declared_source(
            definitions,
            getattr(breakout, "source", None),
            getattr(breakout, "property", None),
            fact_by_name,
            dimension_names,
            requested_dimensions,
            fact_names,
        )
    return fact_names, requested_dimensions


def _add_declared_source(
    definitions: Any,
    declared_source: str | None,
    property_name: str | None,
    fact_by_name: Mapping[str, Any],
    dimension_names: set[str],
    requested_dimensions: set[str],
    fact_names: set[str],
) -> None:
    if declared_source is not None and declared_source in fact_by_name:
        fact_names.add(declared_source)
    elif declared_source in dimension_names:
        requested_dimensions.add(declared_source)
    elif property_name is not None:
        fact_names.update(
            fact_source.name
            for fact_source in definitions.fact_sources
            if any(prop.name == property_name for prop in fact_source.properties)
        )


def _adjustment_fact_names(experiment: Any, definitions: Any) -> set[str]:
    design = getattr(experiment, "design", None)
    unit = getattr(experiment, "unit", None)
    names: set[str] = set()
    for covariate in getattr(design, "covariates", ()):
        property_name = getattr(covariate, "property", covariate)
        declared_source = getattr(covariate, "source", None)
        if not isinstance(property_name, str):
            continue
        candidates = [
            fact_source
            for fact_source in definitions.fact_sources
            if unit in fact_source.entities
            and (declared_source is None or fact_source.name == declared_source)
            and any(prop.name == property_name for prop in definitions.properties_of(fact_source))
        ]
        names.update(fact_source.name for fact_source in candidates)
    return names


def _declared_streams(
    source: object,
    definitions: Any,
    metrics: Sequence[object],
    configs: Sequence[object],
    breakouts: Sequence[object],
    uptake_facts: Sequence[str],
) -> Mapping[str, str]:
    fact_by_name = {fact_source.name: fact_source for fact_source in definitions.fact_sources}
    dimension_by_name = {dimension.name: dimension for dimension in definitions.dim_sources}
    fact_owner = {
        fact.name: fact_source
        for fact_source in definitions.fact_sources
        for fact in fact_source.facts
    }
    fact_names, requested_dimensions = _declared_fact_names(
        definitions, metrics, configs, breakouts, uptake_facts
    )
    experiment = getattr(source, "_experiment", None)
    fact_names.update(_adjustment_fact_names(experiment, definitions))
    streams: dict[str, str] = {}
    if experiment is not None:
        exposure_by_name = {exposure.name: exposure for exposure in definitions.exposures}
        for name in (experiment.exposure, experiment.trigger):
            if name is None or name not in exposure_by_name:
                continue
            exposure = exposure_by_name[name]
            if exposure.fact is None:
                streams[f"exposure:{name}"] = f"exposure:{name}"
            else:
                fact_names.add(exposure.fact)
    for name in fact_names:
        fact_source = fact_by_name.get(name, fact_owner.get(name))
        if fact_source is None:
            continue
        streams[fact_source.name] = fact_source.name
        requested_dimensions.update(fact_source.dims)
    for name in requested_dimensions:
        if name in dimension_by_name:
            streams[f"dimension:{name}"] = f"dimension:{name}"
    return MappingProxyType(dict(sorted(streams.items())))


def _pinned_streams(
    provider: EvidenceProvider,
    metrics: Sequence[object],
    configs: Sequence[object],
    breakouts: Sequence[object],
    uptake_facts: Sequence[str],
) -> Mapping[str, str]:
    source = _loader_owner(provider)
    if (
        is_builtin_source(source)
        and type(source).__module__ == "increment.query._native_triggered"
        and type(source).__name__ == "TriggeredPopulationSource"
    ):
        source = cast(Any, source)._source
    artifact_streams = _artifact_streams(source)
    if artifact_streams is not None:
        return artifact_streams
    definitions = getattr(source, "_defs", None)
    if definitions is None:
        stream_names = {str(getattr(metric, "name", metric)) for metric in metrics}
        stream_names.update(f"fact:{fact}" for fact in uptake_facts)
        return MappingProxyType(dict.fromkeys(sorted(stream_names), "captured"))
    captured = _captured_streams(source, definitions)
    if captured is not None:
        return captured
    return _declared_streams(source, definitions, metrics, configs, breakouts, uptake_facts)


@dataclass(slots=True)
class EvidenceExecutionContext:
    """Own one evidence scope and lend its pinned view to readout acquisition."""

    provider: object
    scope: EvidenceScope
    run_id: str
    coherence: ExecutionMode | None = "captured"
    observation_horizon: date | None = None
    _state: str = field(init=False, default="unopened")
    _stack: ExitStack = field(init=False)
    _scoped_provider: Any | None = field(init=False, default=None)
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
        self._source_identity = id(_loader_owner(self.provider))
        if requested_coherence not in ("captured", "live"):
            _refuse("kernel.request.invalid", field="coherence")
        self._requested_coherence = requested_coherence

    def open(self) -> Any:
        if self._state == "open":
            return self._require_open()
        if self._state != "unopened":
            _refuse("kernel.context.closed")
        self._state = "opening"
        source = _loader_owner(cast(EvidenceProvider, self.provider))
        provider_owned = is_custom_source(source) or bool(
            getattr(self.provider, "provider_owned", False)
        )
        effective_coherence: ExecutionMode = (
            "captured"
            if getattr(getattr(source, "evidence", None), "kind", None) == "artifact"
            else cast(ExecutionMode, self.coherence)
        )
        effective_mode: EffectiveMode = "provider" if provider_owned else effective_coherence
        scope = self.scope
        requested = EvidenceScope(
            metrics=scope.metrics,
            population=scope.population,
            grain=scope.grain,
            breakouts=scope.breakouts,
            configs=scope.configs,
            uptake_facts=scope.uptake_facts,
            snapshot=(
                scope.snapshot if provider_owned else self._requested_coherence == "captured"
            ),
            execution=True,
            include_breakouts=scope.include_breakouts,
        )
        try:
            result = cast(EvidenceProvider, self.provider).scope(requested)
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

    def validate_observational_streams(self, source: object, design: object) -> None:
        """Check bound adjustment sources before the readout issues any evidence request."""
        self._require_open()
        self.validate_source(source)
        if getattr(design, "mechanism", None) != "observational":
            return
        owner = _loader_owner(self.provider)
        definitions = getattr(owner, "_defs", None)
        experiment = getattr(owner, "_experiment", None)
        if definitions is None or experiment is None:
            return
        unit = getattr(experiment, "unit", None)
        declared = getattr(getattr(experiment, "design", None), "covariates", ())
        requested = getattr(getattr(design, "adjustment", None), "covariates", ())
        fact_sources = {fact_source.name: fact_source for fact_source in definitions.fact_sources}
        for request in requested:
            property_name = getattr(request, "property", request)
            bindings = [
                binding
                for binding in declared
                if getattr(binding, "property", binding) == property_name
            ]
            sources = {
                getattr(binding, "source", None)
                for binding in bindings
                if getattr(binding, "source", None) is not None
            }
            if not sources:
                sources = {
                    fact_source.name
                    for fact_source in definitions.fact_sources
                    if unit in fact_source.entities
                    and any(
                        prop.name == property_name
                        for prop in definitions.properties_of(fact_source)
                    )
                }
            if not sources:
                # No source advertises this property. Defer to the provider so
                # its established covariate-capability refusal remains distinct.
                continue
            if len(sources) != 1:
                _refuse(
                    "kernel.context.undeclared_stream",
                    stream=f"covariate:{property_name}",
                )
            stream = next(iter(sources))
            if stream not in fact_sources:
                _refuse("kernel.context.undeclared_stream", stream=f"fact:{stream}")
            mode = self._effective_modes.get(stream)
            if mode is None:
                _refuse("kernel.context.undeclared_stream", stream=stream)
            if (
                self.coherence == "captured"
                and mode != "provider"
                and (mode != "captured" or stream not in self._snapshot_manifest)
            ):
                snapshot_id = self._snapshot_manifest.get(stream)
                _refuse(
                    "kernel.snapshot.integrity",
                    stream=stream,
                    snapshot_id=snapshot_id,
                )

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
        declared_configs = set(getattr(self.scope, "configs", ()) or ())
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
