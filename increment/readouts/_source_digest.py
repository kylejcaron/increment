"""Canonical component records and composite readout source identities."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from increment._canonical import canonical_json_bytes


def component(
    *,
    kind: str,
    metric: str | None,
    population: str,
    sha256: str,
    source: str | None = None,
    dimension: str | None = None,
) -> dict[str, Any]:
    """Create a portable evidence component with its source coordinates."""
    return {
        "kind": kind,
        "metric": metric,
        "population": population,
        "source": source,
        "dimension": dimension,
        "sha256": sha256,
    }


def component_order(
    item: dict[str, Any],
) -> tuple[str, tuple[bool, str], str, tuple[bool, str], tuple[bool, str]]:
    """Order components by their complete identity key; null coordinates sort first."""
    return (
        item["kind"],
        (item["metric"] is not None, item["metric"] or ""),
        item["population"],
        (item["source"] is not None, item["source"] or ""),
        (item["dimension"] is not None, item["dimension"] or ""),
    )


def composite_source(components: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a composite source digest without collapsing duplicate components."""
    ordered = sorted(components, key=component_order)
    return {
        "kind": "composite",
        "sha256": sha256(canonical_json_bytes(ordered)).hexdigest(),
        "components": ordered,
    }
