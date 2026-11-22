"""Named spillover trials declared against fixed warehouse relations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from increment._spillover_design import SpilloverComparison, SpilloverDesign, refuse_input
from increment.errors import CodedError, refuse
from increment.semantics.artifact import RelationLocator
from increment.semantics.loader import _DUPLICATES, _check_category_duplicates, _raw_definitions

_ROLES = {"roster", "offer_counts", "group_policy", "offered", "outcomes"}


def _fields(
    raw: object, field: str, required: set[str], optional: frozenset[str] = frozenset()
) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or not required <= raw.keys() <= required | optional:
        refuse_input("spillover.design.invalid", field)
    return cast(Mapping[str, Any], raw)


def _locator(raw: object) -> str:
    if isinstance(raw, str):
        parts = raw.split(".")
        if not 1 <= len(parts) <= 3:
            refuse_input("spillover.design.invalid", "relations")
        raw = dict(zip(("catalog", "schema", "name")[-len(parts) :], parts, strict=True))
    fields = _fields(raw, "relations", {"name"}, frozenset({"catalog", "schema"}))
    if fields.get("catalog") is not None and fields.get("schema") is None:
        refuse_input("spillover.design.invalid", "relations")
    try:
        locator = RelationLocator.model_validate(fields, strict=True)
    except (CodedError, ValidationError):
        pass
    else:
        return ".".join(
            p for p in (locator.catalog, locator.schema_name, locator.name) if p is not None
        )
    refuse_input("spillover.design.invalid", "relations")


def relation_names(raw: object) -> dict[str, str]:
    """Validate the five table roles; flatten equivalent qualified locators identically."""
    return {role: _locator(value) for role, value in _fields(raw, "relations", _ROLES).items()}


@dataclass(frozen=True, slots=True)
class _Trial:
    name: str
    design: SpilloverDesign
    relations: dict[str, str]


def _trial(raw: object) -> _Trial:
    fields = _fields(raw, "spillovers", {"name", "design", "relations"})
    name = fields["name"]
    if not isinstance(name, str) or not name.strip():
        refuse_input("spillover.design.invalid", "experiment_id")
    design = dict(
        _fields(
            fields["design"],
            "design",
            {"groups_per_policy", "comparisons", "outcome", "horizon"},
            frozenset({"alpha"}),
        )
    )
    comparisons = design.pop("comparisons")
    if not isinstance(comparisons, list):
        refuse_input("spillover.design.invalid", "comparisons")
    return _Trial(
        name=name,
        design=SpilloverDesign(
            **design,
            comparisons=tuple(
                SpilloverComparison(
                    **_fields(
                        c,
                        "comparisons",
                        {"policy", "reference", "meaningful_effect"},
                        frozenset({"baseline_rate"}),
                    )
                )
                for c in comparisons
            ),
        ),
        relations=relation_names(fields["relations"]),
    )


def load_trials(path: Path) -> dict[str, _Trial]:
    raw, provenance = _raw_definitions(path, advise=False, shared_metrics=False)
    entries = _fields(raw, "spillovers", {"spillovers"})["spillovers"]
    trials = tuple(_trial(item) for item in entries)
    duplicates: list[str] = []
    _check_category_duplicates(entries, "spillover", provenance.origins, duplicates)
    if duplicates:
        refuse(
            _DUPLICATES, message="; ".join(duplicates), path=str(path), sources=provenance.origins
        )
    return {t.name: t for t in trials}
