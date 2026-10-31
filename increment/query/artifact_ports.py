"""Typed, lazily discovered ports for the optional artifact query extension."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import distribution
from threading import RLock
from typing import Any, Protocol

from increment.errors import _BUILTIN_PROVIDER_UNAVAILABLE, refuse

# Persisted digest identities are wire declarations, not implementation details.
SQL_DIGEST_FORMAT = 4
SQL_DIGEST_FORMAT_V2 = 2
BUCKETED_DIGEST_FORMAT = 3
SQL_DIGEST_BACKENDS = frozenset({"duckdb", "postgres", "snowflake", "bigquery"})


class ArtifactReaderPort(Protocol):
    """Artifact-backed reader operations consumed by query core."""

    @property
    def ArtifactMomentSource(self) -> type: ...

    LAZY_DIGEST_VERIFICATION: Any

    @property
    def _TriggeredReductionInputs(self) -> type: ...

    def _restore_raw_dimension_values(self, *args: Any, **kwargs: Any) -> Any: ...
    def restrict_to_units(self, *args: Any, **kwargs: Any) -> Any: ...
    def _rows(self, *args: Any, **kwargs: Any) -> Any: ...
    def _declared_metric_definitions(self, *args: Any, **kwargs: Any) -> Any: ...
    def _metric_spec_from_binding(self, *args: Any, **kwargs: Any) -> Any: ...


class ArtifactPublisherPort(Protocol):
    """Artifact publisher and publication-context operations."""

    @property
    def ArtifactPublisher(self) -> type: ...

    def artifact_context(self, *args: Any, **kwargs: Any) -> Any: ...
    def _effective_snapshot_edge(self, *args: Any, **kwargs: Any) -> Any: ...


class ArtifactExtensionsPort(Protocol):
    """Artifact extension encoding, reading and verification operations."""

    def read_extension(self, *args: Any, **kwargs: Any) -> Any: ...
    def verify_extension_relation(self, *args: Any, **kwargs: Any) -> Any: ...
    def encode_extension(self, *args: Any, **kwargs: Any) -> Any: ...


class SequentialCapturePort(Protocol):
    """Relational capture operations used by query/session implementations."""

    def certify_capture_feed(self, *args: Any, **kwargs: Any) -> Any: ...
    def capture_relations(self, *args: Any, **kwargs: Any) -> Any: ...
    def record_batches(self, *args: Any, **kwargs: Any) -> Any: ...
    def validate_relational_capture(self, *args: Any, **kwargs: Any) -> Any: ...


class ArtifactDigestPort(Protocol):
    """Wire digest and canonical-relation operations implemented by artifacts."""

    REFUSALS: Any

    def digest_relation_bucketed(self, *args: Any, **kwargs: Any) -> Any: ...
    def digest_relation_sql_v2(self, *args: Any, **kwargs: Any) -> Any: ...
    def digest_relation_stream(self, *args: Any, **kwargs: Any) -> Any: ...
    def fetch_bounded_arrow(self, *args: Any, **kwargs: Any) -> Any: ...
    def stream_canonical_rows(self, *args: Any, **kwargs: Any) -> Any: ...
    def manifest_sha256(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ArtifactQueryPorts:
    """Typed artifact implementation boundary for query core."""

    reader: ArtifactReaderPort
    publisher: ArtifactPublisherPort
    extensions: ArtifactExtensionsPort
    sequential_capture: SequentialCapturePort
    digest: ArtifactDigestPort


_lock = RLock()
_ports: ArtifactQueryPorts | None = None


def artifact_query_ports() -> ArtifactQueryPorts:
    """Load the bundled artifact implementation on first use."""
    global _ports
    ports = _ports
    if ports is not None:
        return ports
    with _lock:
        ports = _ports
        if ports is None:
            providers = tuple(
                entry
                for entry in distribution("increment").entry_points
                if entry.group == "increment.artifact_ports" and entry.name == "builtin"
            )
            if len(providers) != 1:
                refuse(
                    _BUILTIN_PROVIDER_UNAVAILABLE,
                    feature="unit-day artifacts",
                    group="increment.artifact_ports",
                    provider="builtin",
                    reason="entry_point_count",
                    count=len(providers),
                )
            factory = providers[0].load()
            ports = factory()
            if not isinstance(ports, ArtifactQueryPorts):
                refuse(
                    _BUILTIN_PROVIDER_UNAVAILABLE,
                    feature="unit-day artifacts",
                    group="increment.artifact_ports",
                    provider="builtin",
                    reason="invalid_contract",
                    count=len(providers),
                )
            _ports = ports
        return ports
