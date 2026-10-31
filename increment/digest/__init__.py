"""Stable public re-exports for digest APIs."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

_EXPORTS = {
    "ArtifactDigestError": ("increment.query.artifact_digest", "ArtifactDigestError"),
    "DigestType": ("increment.query.artifact_digest", "DigestType"),
    "FieldSpec": ("increment.query.artifact_digest", "FieldSpec"),
    "RelationDigests": ("increment.query.artifact_digest", "RelationDigests"),
    "canonical_schema_bytes": ("increment.query.artifact_digest", "canonical_schema_bytes"),
    "canonical_row_bytes": ("increment.query.artifact_digest", "canonical_row_bytes"),
    "canonical_rows_bytes": ("increment.query.artifact_digest", "canonical_rows_bytes"),
    "canonical_json_bytes": ("increment.query.artifact_digest", "canonical_json_bytes"),
    "schema_sha256": ("increment.query.artifact_digest", "schema_sha256"),
    "content_sha256": ("increment.query.artifact_digest", "content_sha256"),
    "digest_relation": ("increment.query.artifact_digest", "digest_relation"),
    "manifest_sha256": ("increment.query.artifact_digest", "manifest_sha256"),
    "context_sha256": ("increment.query.artifact_digest", "context_sha256"),
    "request_sha256": ("increment.query.artifact_digest", "request_sha256"),
    "extension_definition_sha256": (
        "increment.query.artifact_digest",
        "extension_definition_sha256",
    ),
    "extension_source_provenance_sha256": (
        "increment.query.artifact_digest",
        "extension_source_provenance_sha256",
    ),
}

if TYPE_CHECKING:
    from increment.query.artifact_digest import (
        ArtifactDigestError as ArtifactDigestError,
    )
    from increment.query.artifact_digest import (
        DigestType as DigestType,
    )
    from increment.query.artifact_digest import (
        FieldSpec as FieldSpec,
    )
    from increment.query.artifact_digest import (
        RelationDigests as RelationDigests,
    )
    from increment.query.artifact_digest import (
        canonical_json_bytes as canonical_json_bytes,
    )
    from increment.query.artifact_digest import (
        canonical_row_bytes as canonical_row_bytes,
    )
    from increment.query.artifact_digest import (
        canonical_rows_bytes as canonical_rows_bytes,
    )
    from increment.query.artifact_digest import (
        canonical_schema_bytes as canonical_schema_bytes,
    )
    from increment.query.artifact_digest import (
        content_sha256 as content_sha256,
    )
    from increment.query.artifact_digest import (
        context_sha256 as context_sha256,
    )
    from increment.query.artifact_digest import (
        digest_relation as digest_relation,
    )
    from increment.query.artifact_digest import (
        extension_definition_sha256 as extension_definition_sha256,
    )
    from increment.query.artifact_digest import (
        extension_source_provenance_sha256 as extension_source_provenance_sha256,
    )
    from increment.query.artifact_digest import (
        manifest_sha256 as manifest_sha256,
    )
    from increment.query.artifact_digest import (
        request_sha256 as request_sha256,
    )
    from increment.query.artifact_digest import (
        schema_sha256 as schema_sha256,
    )


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(name) from None
    module = importlib.import_module(module_name)
    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "ArtifactDigestError",
    "DigestType",
    "FieldSpec",
    "RelationDigests",
    "canonical_schema_bytes",
    "canonical_row_bytes",
    "canonical_rows_bytes",
    "canonical_json_bytes",
    "schema_sha256",
    "content_sha256",
    "digest_relation",
    "manifest_sha256",
    "context_sha256",
    "request_sha256",
    "extension_definition_sha256",
    "extension_source_provenance_sha256",
]
