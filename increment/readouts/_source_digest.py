"""Canonical input evidence components and composite readout source identities."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from increment._canonical import canonical_json_bytes

_FLOAT_INPUT_FIELDS = {"dimension_value", "ds", "n_control", "n_treat", "segment"}


def input_evidence(value: Any, *, field: str | None = None) -> Any:
    """Keep discrete inputs and counts while excluding float aggregates."""
    if isinstance(value, float):
        return value if field in _FLOAT_INPUT_FIELDS else None
    if isinstance(value, int) and not -(2**53) < value < 2**53:
        return {"$type": "integer", "value": str(value)}
    if isinstance(value, dict):
        return {
            key: input_evidence(item, field=key)
            for key, item in value.items()
            if key
            not in {
                "exact_delta_total",
                "discovery",
                "multiplicity_status",
                "family_id",
                "family_size",
                "family_axes",
                "family_q",
                "family_threshold",
                "family_nominal_alpha",
                "decision_scope_complete",
                "decision_scope_reason_code",
                "decision_scope_reason_context",
                "sampling_available",
                "sampling_reason_code",
                "sampling_reason_context",
                "failure_code",
                "failure_context",
                "inference",
            }
        }
    if isinstance(value, (list, tuple)):
        return [input_evidence(item, field=field) for item in value]
    return value


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
