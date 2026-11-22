"""Source-scoped switchback estimates and reasons for unavailable decisions."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from increment._canonical import canonical_digest_bytes, canonical_json_bytes
from increment.estimation.assignment_integrity import assignment_integrity, fixed_count_integrity
from increment.estimation.contrast_results import ContrastResults
from increment.estimation.readout_types import (
    CellFailure,
    CellKey,
    CellRecord,
    PopulationRoster,
    ReadoutMetadata,
    ReadoutScope,
    SamplingInference,
    SourceReadoutScope,
    cell_order,
)
from increment.readouts._metric_type import declared_metric_type, declared_metric_types
from increment.readouts._source_digest import component, composite_source, input_evidence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment.estimation.contrast import (
        ContrastStats,
        SwitchbackMetricStats,
        UnavailableContrastStats,
    )
    from increment.estimation.contrast_results import ContrastResult
    from increment.semantics.models import Metric


def _cell_record(cell: CellKey, row: ContrastResult, snapshot_id: str) -> CellRecord:
    if row.failure_code is None:
        return CellRecord(cell=cell, source_snapshot_id=snapshot_id)
    assert row.failure_context is not None
    return CellRecord(
        cell=cell,
        source_snapshot_id=snapshot_id,
        failure=CellFailure(
            hypothesis=cell.hypothesis(), code=row.failure_code, context=row.failure_context
        ),
        sampling=SamplingInference(
            available=False, reason_code=row.failure_code, reason_context=row.failure_context
        ),
    )


def scope_contrast_results(
    results: Sequence[ContrastResult],
    *,
    metrics: Sequence[Metric],
    stats: Mapping[str, ContrastStats | SwitchbackMetricStats | UnavailableContrastStats],
    procedures: Mapping[str, Any],
    component_source: str | None = None,
    dimension: str | None = None,
    source_identity_record: Mapping[str, Any] | None = None,
    coherence: Literal["captured", "live"] | None = None,
    effective_modes: Mapping[str, Literal["captured", "live", "provider"]] | None = None,
) -> ContrastResults:
    """Stamp every selected contrast, including cells without a decision."""
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
    construction_fields = {
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
        "periods_per_cycle",
        "treated_periods",
        "treatment_probability",
        "forecast_column",
    }
    request = {
        "kind": "contrast",
        "metrics": sorted(stats),
        "procedures": {
            metric: procedures[metric].model_dump(mode="json") for metric in sorted(stats)
        },
        "populations": ["assigned"],
        "construction": {
            metric: stats[metric].model_dump(mode="json", include=construction_fields)
            for metric in sorted(stats)
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

    metric_types = declared_metric_types(metrics)
    unavailable = tuple(sorted(row.metric for row in results if row.failure_code is not None))
    complete = not unavailable
    scope_reason = {"unavailable_metrics": unavailable} if unavailable else None
    stamped = [
        row.model_copy(
            update={
                "source_snapshot_id": snapshot_id,
                "metric_type": declared_metric_type(row, metric_types),
                "decision_scope_complete": complete,
                "decision_scope_reason_code": (
                    None if complete else "readout.scope.decision_incomplete"
                ),
                "decision_scope_reason_context": scope_reason,
            }
        )
        for row in results
    ]
    cells = {CellKey.from_row(row): row for row in stamped}
    ordered = tuple(sorted(cells, key=cell_order))
    arms = tuple(
        sorted({group for row in results for group in (row.control_group, row.treatment_group)})
    )
    design = next(iter(stats.values()), None)
    if design is not None and design.randomization_law == "fixed_count":
        assert design.periods_per_cycle is not None and design.treated_periods is not None
        integrity = fixed_count_integrity(
            periods=design.periods_per_cycle,
            treated=design.treated_periods,
            grain=design.independence_grain,
            draws=design.n_blocks if design.n_blocks is not None else design.n_cycles,
        )
    else:
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
        decision_complete_by_population={"assigned": complete},
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
        cells=tuple(_cell_record(cell, cells[cell], snapshot_id) for cell in ordered),
        coherence=coherence,
        effective_modes=effective_modes or {},
    )
    return ContrastResults(stamped, metadata=metadata, source=source)
