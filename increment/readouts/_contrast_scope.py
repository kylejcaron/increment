"""Scope records for switchback contrast readouts, which remain all-or-nothing."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Any

from increment._canonical import canonical_digest_bytes, canonical_json_bytes
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.readout_types import (
    CellKey,
    CellRecord,
    PopulationRoster,
    ReadoutMetadata,
    ReadoutScope,
    SourceReadoutScope,
    cell_order,
)
from increment.readouts._source_digest import component, composite_source, input_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment.estimation.contrast import ContrastStats
    from increment.estimation.contrast_results import ContrastResult


def scope_contrast_results(
    results: Sequence[ContrastResult],
    *,
    stats: Mapping[str, ContrastStats],
    procedures: Mapping[str, Any],
    component_source: str | None = None,
    dimension: str | None = None,
    source_identity_record: Mapping[str, Any] | None = None,
) -> ContrastResults:
    """Attach scope to a complete set of contrast results.

    A contrast cannot partially fail: every selected metric either yields its result or the
    request refuses, so the single assigned population is decision complete.
    """
    components = [
        component(
            kind="unit_evidence_rows",
            metric=metric,
            population="assigned",
            sha256=sha256(
                canonical_digest_bytes(input_evidence(value.model_dump(mode="json")))
            ).hexdigest(),
            source=component_source,
            dimension=dimension,
        )
        for metric, value in stats.items()
    ]
    source = composite_source(components)
    construction_fields = (
        "aggregation",
        "probability_ct",
        "randomization_law",
        "independence_grain",
        "washout_steps",
        "carryover_order",
        "observation_steps",
        "retained_steps",
        "identifying_assumption",
        "control_group",
        "treatment_group",
        "n_units",
        "n_cycles",
        "n_blocks",
        "ct_cycles",
        "tc_cycles",
        "minimum_cycles_per_unit",
        "maximum_cycles_per_unit",
    )
    construction = {metric: value.model_dump(mode="json") for metric, value in stats.items()}
    request = {
        "kind": "contrast",
        "metrics": sorted(stats),
        "procedures": {
            metric: procedures[metric].model_dump(mode="json") for metric in sorted(stats)
        },
        "populations": ["assigned"],
        "construction": {
            metric: {key: construction[metric][key] for key in construction_fields}
            for metric in sorted(construction)
        },
    }
    snapshot_id = (
        "sha256:"
        + sha256(
            canonical_json_bytes(
                {
                    "kind": "increment.readout.snapshot",
                    "version": 2,
                    "collection": "ContrastResults",
                    "source_identity": source_identity_record,
                    "source": source,
                    "request": request,
                }
            )
        ).hexdigest()
    )

    stamped = [
        row.model_copy(update={"source_snapshot_id": snapshot_id, "decision_scope_complete": True})
        for row in results
    ]
    cells = {CellKey.from_row(row): row for row in stamped}
    ordered = tuple(sorted(cells, key=cell_order))
    arms = tuple(
        sorted({group for row in results for group in (row.control_group, row.treatment_group)})
    )
    integrity = assignment_integrity(None, None)
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=ordered,
        decision_cells=ordered,
        rosters=(
            PopulationRoster(
                analysis_population="assigned",
                arms=arms,
                source="observed_union",
                complete=True,
            ),
        ),
        decision_complete_by_population={"assigned": True},
        integrity=(integrity,),
    )
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=ordered,
        decision_cells=ordered,
        populations=("assigned",),
        families=(),
        by_source={snapshot_id: source_scope},
    )
    metadata = ReadoutMetadata(
        scope=scope,
        cells=tuple(CellRecord(cell=cell, source_snapshot_id=snapshot_id) for cell in ordered),
    )
    return ContrastResults(stamped, metadata=metadata, source=source)
