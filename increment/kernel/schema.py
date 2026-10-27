from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from increment.kernel.contracts import _refuse

_VERSION = 1
_EMPTY: Mapping[str, Mapping[str, str]] = MappingProxyType({})


class CatalogSchemaCatalog:
    """Backend-free schema lookup over declarations and bound schema records."""

    def __init__(
        self,
        *,
        properties: Mapping[str, Mapping[str, Any]],
        persisted: Mapping[str, Mapping[str, str]] = _EMPTY,
        definitions_digest: str | None = None,
    ) -> None:
        self._properties = _freeze(properties)
        self._persisted = _freeze(persisted)
        self._definitions_digest = definitions_digest
        self._stale: dict[str, str] = {}

    def columns(self, source_ref: str) -> Mapping[str, str]:
        if source_ref in self._stale:
            return _EMPTY.get(source_ref, MappingProxyType({}))
        if source_ref in self._properties:
            return self._properties[source_ref]
        if source_ref in self._persisted:
            return self._persisted[source_ref]
        _refuse("kernel.request.invalid", field="source_ref")

    def stale_reason(self, source_ref: str) -> str | None:
        return self._stale.get(source_ref)

    def _payload(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "definitions_digest": self._definitions_digest,
            "sources": {
                source: {"source_ref": source, "columns": dict(sorted(columns.items()))}
                for source, columns in sorted(self._persisted.items())
            },
        }


def _freeze(values: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Mapping[str, str]]:
    frozen: dict[str, Mapping[str, str]] = {}
    for source, columns in sorted(values.items()):
        declared: dict[str, str] = {}
        for name, value in sorted(columns.items()):
            dtype = getattr(value, "dtype", value)
            if not isinstance(dtype, str):
                _refuse("kernel.request.invalid", field="schema_dtype")
            declared[name] = dtype
        frozen[source] = MappingProxyType(declared)
    return MappingProxyType(frozen)


def write_schema_catalog(path: Path, catalog: CatalogSchemaCatalog) -> None:
    """Atomically persist bound schemas in deterministic JSON order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(catalog._payload(), sort_keys=True, separators=(",", ":")) + "\n"
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_schema_catalog(
    path: Path,
    *,
    definitions_digest: str | None = None,
    source_refs: Mapping[str, str] | None = None,
) -> CatalogSchemaCatalog:
    """Load a bound catalog; changed definition/source identities become stale."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_sources = payload.get("sources")
    if payload.get("version") != _VERSION or not isinstance(raw_sources, dict):
        _refuse("kernel.request.invalid", field="catalog")
    sources: dict[str, Mapping[str, str]] = {}
    identities: dict[str, str] = {}
    for key, record in raw_sources.items():
        if (
            not isinstance(key, str)
            or not isinstance(record, dict)
            or record.get("source_ref") != key
            or not isinstance(record.get("columns"), dict)
        ):
            _refuse("kernel.request.invalid", field="catalog")
        sources[key] = record["columns"]
        identities[key] = record["source_ref"]
    catalog = CatalogSchemaCatalog(
        properties={},
        persisted=sources,
        definitions_digest=payload.get("definitions_digest"),
    )
    if definitions_digest is not None and definitions_digest != payload.get("definitions_digest"):
        for source in sources:
            catalog._stale[source] = "kernel.schema.stale"
    if source_refs is not None:
        for source, expected in source_refs.items():
            if identities.get(source) != expected:
                catalog._stale[source] = "kernel.schema.stale"
    for source in tuple(catalog._stale):
        catalog._persisted = MappingProxyType(
            {key: value for key, value in catalog._persisted.items() if key != source}
        )
    return catalog
