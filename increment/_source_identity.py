"""Source identity shared by query and readout layers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from increment._canonical import CanonicalJSONError, canonical_json_bytes


def source_identity_is_canonical(identity: Mapping[str, object]) -> bool:
    try:
        canonical_json_bytes(identity)
    except CanonicalJSONError:
        return False
    return True


def source_identities_equal(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    """Compare identity records by canonical JSON, not Python's bool/int equality."""
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def source_identity(source: Any) -> dict[str, Any]:
    """Return stable coordinates, preferring identity embedded in moments cubes."""
    leaf = source
    seen: set[int] = set()
    evidence = None
    source_identity_record = None
    manifests = []
    while id(leaf) not in seen:
        seen.add(id(leaf))
        session = getattr(leaf, "_session", None)
        if source_identity_record is None:
            source_identity_record = getattr(leaf, "_source_identity_record", None)
        evidence = evidence or getattr(leaf, "_source_snapshot_evidence", None)
        evidence = evidence or getattr(session, "source_snapshot_evidence", None)
        manifest = getattr(leaf, "_manifest", None)
        if manifest is not None:
            manifests.append(
                {
                    "artifact_id": str(manifest.artifact_id),
                    "generation_id": str(manifest.generation_id),
                    "manifest_sha256": manifest.manifest_sha256,
                    "context_sha256": manifest.context.sha256,
                    "experiment_id": manifest.experiment_id,
                }
            )
        wrapped = getattr(leaf, "_source", None)
        if wrapped is None:
            break
        leaf = wrapped
    if source_identity_record is not None:
        return dict(source_identity_record)
    context = getattr(leaf, "context", None)
    identity: dict[str, Any] = {
        "study_id": getattr(context, "study_id", None),
        "artifacts": manifests,
    }
    if evidence is not None:
        identity["source_snapshot_evidence"] = {
            "observation_cutoff_ts": evidence.observation_cutoff_ts.isoformat(),
            "complete_through_by_feed": {
                feed: None if watermark is None else watermark.isoformat()
                for feed, watermark in sorted(evidence.complete_through_by_feed.items())
            },
        }
    return identity
