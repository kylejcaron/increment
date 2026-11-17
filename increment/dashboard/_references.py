"""Source-free lookup of the original records behind dashboard results."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from increment._interpretation import interpret_readout, readout_evidence_columns
from increment.estimation.readout_types import CellKey, CellRecord
from increment.tables import estimates_to_readout

if TYPE_CHECKING:
    from increment.dashboard._data import DashboardSnapshot


def view_key(parts: Sequence[Any]) -> str:
    """Encode all captured request coordinates without label-based aliases."""
    return json.dumps(parts, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class DiagnosticSummary:
    metric: str
    code: str
    result_ids: tuple[str, ...]
    view_keys: tuple[str, ...]


def diagnostic_summaries(catalog: ReferenceCatalog) -> tuple[DiagnosticSummary, ...]:
    """Group warnings without counting an overlapping code or view as another result."""
    memberships: dict[str, list[str]] = {}
    for view, identities in catalog.views.items():
        for identity in identities:
            memberships.setdefault(identity, []).append(view)
    groups: dict[tuple[str, str], list[str]] = {}
    for reference in catalog.results.values():
        for code in reference.diagnostic_codes:
            groups.setdefault((reference.cell.metric, code), []).append(reference.id)
    return tuple(
        DiagnosticSummary(
            metric,
            code,
            tuple(dict.fromkeys(identities)),
            tuple(
                dict.fromkeys(
                    view for identity in identities for view in memberships.get(identity, ())
                )
            ),
        )
        for (metric, code), identities in groups.items()
    )


@dataclass(frozen=True, slots=True)
class ResultReference:
    id: str
    source_snapshot_id: str | None
    cell: CellKey
    row: Mapping[str, Any] | None
    record: CellRecord | None
    original: Mapping[str, Any] | None
    group_ids: tuple[str, ...]

    @property
    def diagnostic_codes(self) -> tuple[str, ...]:
        """Interval cautions for a retained estimate, not an unshown metadata record."""
        if self.row is None or self.record is None:
            return ()
        return tuple(
            advisory.code
            for advisory in self.record.advisories
            if advisory.code
            in (
                "estimation.engine.ratio_denominator_precision",
                "estimation.engine.ratio_denominator_skew",
            )
        )


@dataclass(frozen=True, slots=True)
class ReferenceCatalog:
    results: Mapping[str, ResultReference]
    groups: Mapping[str, Mapping[str, Any]]
    views: Mapping[str, tuple[str, ...]]
    view_refusals: Mapping[str, Mapping[str, Any]]

    def matching(self, key: str, metric: str) -> tuple[str, ...]:
        return tuple(
            identity
            for identity in self.views.get(key, ())
            if self.results[identity].cell.metric == metric
        )

    def row_ids(self, key: str, rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
        """Match original coordinates and retained intervals before display relabeling."""
        candidates: dict[tuple[Any, ...], list[ResultReference]] = {}
        axes = ("metric", "ds", "group_id")
        for identity in self.views.get(key, ()):
            reference = self.results[identity]
            coordinates = tuple(getattr(reference.cell, name) for name in axes)
            candidates.setdefault(coordinates, []).append(reference)
        output = []
        for row in rows:
            matches = []
            for reference in candidates.get(tuple(row.get(name) for name in axes), ()):
                if reference.row is None:
                    continue
                coordinate = reference.cell.model_dump()
                expected = {**row, "method": row.get("_evidence_method", row.get("method"))}
                if expected.get("method") == "report" and coordinate["method"] is None:
                    expected["method"] = None
                    expected["method_role"] = coordinate["method_role"]
                    expected["estimand"] = coordinate["estimand"]
                if expected.get("dimension") is not None and "dimension_value" not in expected:
                    expected["dimension_value"] = expected.get("segment")
                names = (
                    "metric",
                    "group_id",
                    "method",
                    "method_role",
                    "analysis_population",
                    "estimand",
                    "dimension",
                    "dimension_value",
                    "source",
                    "ds",
                    "ds_basis",
                )
                if any(name in expected and expected[name] != coordinate[name] for name in names):
                    continue
                if any(
                    row.get(name) != reference.row.get(name)
                    for name in ("source_snapshot_id", "lower", "higher", "coverage_alpha")
                ):
                    continue
                matches.append(reference.id)
            output.append(tuple(matches))
        return output


def _reference_group_signature(values: dict[str, Any]) -> str:
    # View visibility does not create another estimate.
    capture = values.get("capture")
    if capture is not None:
        values = {
            **values,
            "capture": {
                key: value
                for key, value in capture.items()
                if key not in ("partial", "partial_reason")
            },
        }
    return json.dumps(values, sort_keys=True, default=str)


def build_reference_catalog(snapshot: DashboardSnapshot) -> ReferenceCatalog:
    """Index captured rows and hidden required cells, never acquiring new data."""
    results: dict[str, ResultReference] = {}
    groups: dict[str, Mapping[str, Any]] = {}
    views: dict[str, tuple[str, ...]] = {}
    refusals: dict[str, Mapping[str, Any]] = {}
    result_ids: dict[str, str] = {}
    group_ids: dict[str, str] = {}

    def shared_group(values: dict[str, Any], *, view: str | None = None) -> str:
        capture = values.get("capture")
        if (signature := _reference_group_signature(values)) not in group_ids:
            identity = f"g{len(groups)}"
            group_ids[signature] = identity
            groups[identity] = MappingProxyType(
                {key: value for key, value in values.items() if key != "capture"}
                if capture is not None
                else values
            )
        identity = group_ids[signature]
        if capture is not None:
            assert view is not None
            groups[identity] = MappingProxyType(
                {
                    **groups[identity],
                    "captures_by_view": {
                        **groups[identity].get("captures_by_view", {}),
                        view: (
                            *groups[identity].get("captures_by_view", {}).get(view, ()),
                            capture,
                        ),
                    },
                }
            )
        return identity

    def add(
        parts: Sequence[Any],
        collection: Sequence[Any],
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        key = view_key(parts)
        metadata = getattr(collection, "metadata", None)
        records = (
            {(record.source_snapshot_id, record.cell): record for record in metadata.cells}
            if metadata is not None
            else {}
        )
        source_groups: dict[str, tuple[str, ...]] = {}
        family_groups: dict[tuple[str, CellKey], list[str]] = {}
        if metadata is not None:
            capture = metadata.model_dump(mode="json", exclude={"scope", "cells"})
            for source, scope in metadata.scope.by_source.items():
                source_groups[source] = (
                    shared_group(
                        {"source": scope.model_dump(mode="json"), "capture": capture},
                        view=key,
                    ),
                )
            for family in metadata.scope.families:
                field = "captured_correction" if selection is not None else "correction"
                identity = shared_group({field: family.model_dump(mode="json")})
                for cell in family.members:
                    family_groups.setdefault((family.source_snapshot_id, cell), []).append(identity)
        members: list[str] = []
        seen: set[tuple[str | None, CellKey]] = set()
        selected_group = (shared_group(dict(selection)),) if selection is not None else ()

        def retain(source, cell, row, record, original) -> None:
            group = (
                *source_groups.get(source, ()),
                *family_groups.get((source, cell), ()),
                *selected_group,
            )
            # Unknown sources cannot establish equality across independently captured views.
            signature = json.dumps(
                (
                    source,
                    cell.model_dump(mode="json"),
                    original,
                    record.model_dump(mode="json") if record else None,
                    group,
                    key if source is None else None,
                ),
                sort_keys=True,
                default=str,
            )
            identity = result_ids.get(signature)
            if identity is None:
                identity = f"r{len(results)}"
                result_ids[signature] = identity
                results[identity] = ResultReference(
                    identity,
                    source,
                    cell,
                    MappingProxyType(row) if row is not None else None,
                    record,
                    MappingProxyType(original) if original is not None else None,
                    group,
                )
            members.append(identity)
            seen.add((source, cell))

        interpretation = interpret_readout(collection, metadata)
        for original, row in zip(collection, estimates_to_readout(collection), strict=True):
            cell = CellKey.from_row(original)
            source = getattr(original, "source_snapshot_id", None)
            row.update(readout_evidence_columns(interpretation, original))
            retain(source, cell, row, records.get((source, cell)), original.model_dump(mode="json"))
        if metadata is not None:
            for source, scope in metadata.scope.by_source.items():
                for cell in scope.decision_cells:
                    if (source, cell) not in seen:
                        retain(source, cell, None, records.get((source, cell)), None)
        views[key] = tuple(dict.fromkeys(members))

    add(
        ("readout",),
        snapshot.readout_collection
        if snapshot.readout_collection is not None
        else snapshot.estimates,
    )
    for key, capture in snapshot.explore.items():
        if capture.population != (
            "triggered" if "triggered" in snapshot.population_estimates else "assigned"
        ):
            continue
        parts = ("explore", *key)
        if capture.refusal is not None:
            refusals[view_key(parts)] = MappingProxyType(
                {
                    "code": capture.refusal.code,
                    "reason": str(capture.refusal),
                    "context": capture.refusal.context,
                }
            )
        else:
            add(parts, capture.load())
    _add_overview_views(snapshot, views, add)
    return ReferenceCatalog(
        MappingProxyType(results),
        MappingProxyType(groups),
        MappingProxyType(views),
        MappingProxyType(refusals),
    )


def _add_overview_views(snapshot, views, add) -> None:
    overview = snapshot.overview
    selection = None
    if overview is not None:
        excluded = {(item.source_snapshot_id, item.cell) for item in overview.exclusions}
        selection = {
            "correction": {
                "name": "Exploratory overview",
                "family_q": overview.family_q,
                "family_size": overview.family_size,
                "members": [
                    {
                        **CellKey.from_row(row).model_dump(mode="json"),
                        "source_snapshot_id": row.source_snapshot_id,
                    }
                    for row in (
                        *overview.exploratory,
                        *(row for rows in overview.segments.values() for row in rows),
                    )
                    if row.multiplicity_status == "exploratory_family"
                    and (row.source_snapshot_id, CellKey.from_row(row)) not in excluded
                ],
                "exclusions": [
                    {
                        **item.cell.model_dump(mode="json"),
                        "source_snapshot_id": item.source_snapshot_id,
                        "reason": item.reason,
                    }
                    for item in overview.exclusions
                ],
            }
        }
    for breakout in (None, *snapshot.breakouts):
        parts = ("overview", *(breakout if breakout is not None else (None, None)))
        members = list(views[view_key(("readout",))])
        if overview is not None:
            whole = _overview_collection(snapshot, overview.exploratory, "overview_whole", None)
            add(parts, whole, selection)
            members.extend(views[view_key(parts)])
            if breakout is not None:
                segments = _overview_collection(
                    snapshot,
                    overview.segments.get(breakout, ()),
                    "overview_segments",
                    breakout,
                )
                add((*parts, "segments"), segments, selection)
                members.extend(views.pop(view_key((*parts, "segments"))))
        views[view_key(parts)] = tuple(dict.fromkeys(members))


def _overview_collection(snapshot, rows, view, breakout):
    population = rows[0].analysis_population if rows else "assigned"
    capture = snapshot.explore.get((population, view, None, False, breakout))
    if capture is None or capture.refusal is not None or capture.collection is None:
        return rows
    return capture.collection(rows)
