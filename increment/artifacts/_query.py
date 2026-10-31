"""Composition adapter exposing artifact implementations through core ports."""

from __future__ import annotations

from functools import cache

from increment.query.artifact_ports import ArtifactQueryPorts


@cache
def artifact_query_ports() -> ArtifactQueryPorts:
    """Construct the installed artifact implementation on first query use."""
    from increment.query import (
        artifact_digest,
        artifact_extensions,
        artifact_publish,
        artifact_reader,
        sequential_capture,
    )

    return ArtifactQueryPorts(
        reader=artifact_reader,
        publisher=artifact_publish,
        extensions=artifact_extensions,
        sequential_capture=sequential_capture,
        digest=artifact_digest,
    )
