"""Scope and evidence records for registered sequential readouts."""

from __future__ import annotations

from hashlib import sha256
from typing import TYPE_CHECKING, Any

from increment._canonical import canonical_digest_bytes, canonical_json_bytes
from increment._source_identity import source_identity
from increment.breakout.estimates import LiftEstimates
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.decision_types import (
    AsymptoticSequentialEvidence,
    EValueEvidence,
    sequential_hypothesis_key,
)
from increment.estimation.readout_types import (
    CellFailure,
    CellKey,
    CellRecord,
    Population,
    PopulationRoster,
    ReadoutMetadata,
    ReadoutScope,
    SamplingInference,
    SourceReadoutScope,
    cell_order,
)
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential_result import AsymptoticSequentialResult
from increment.readouts._common import _raise, _require_design
from increment.readouts._design_scope import ResolvedRoster, resolve_roster
from increment.readouts._multiplicity_scope import attach_multiplicity_scope
from increment.readouts._source_digest import component, composite_source

if TYPE_CHECKING:
    from collections.abc import Sequence

    from increment.sequential_state import SequentialSnapshot

UNSUPPORTED_CODE = "readout.cell.unsupported_request"
TRIGGERED_SEQUENTIAL = "triggered_sequential"


def _evidence(row: LiftEstimate, cell: CellKey):
    result = row.sequential_result
    if result is None:
        return None
    hypothesis = sequential_hypothesis_key(result.checkpoint.cell)
    if hypothesis != cell.hypothesis():
        return None
    if isinstance(result, AsymptoticSequentialResult):
        return AsymptoticSequentialEvidence(hypothesis=hypothesis, method=row.method, result=result)
    return EValueEvidence(
        hypothesis=hypothesis,
        method=row.method,
        log_e=result.log_e,
        process="raw_likelihood_v1",
        checkpoint=result.checkpoint,
        certificate=result.certificate,
    )


def _snapshot_identity(
    plan,
    snapshot,
    metrics,
    estimands,
    trigger_name,
    assignment_counts,
    *,
    source=None,
    component_source: str | None = None,
    dimension: str | None = None,
    source_identity_record: Any = None,
):
    if source is None:
        components = [
            component(
                kind="sequential_prefix",
                metric=None,
                population="assigned",
                sha256=sha256(
                    canonical_json_bytes(
                        {
                            "registration_id": snapshot.registration_id,
                            "prefix_id": snapshot.prefix_id,
                        }
                    )
                ).hexdigest(),
                source=component_source,
                dimension=dimension,
            )
        ]
        if assignment_counts is not None:
            components.append(
                component(
                    kind="assignment_counts",
                    metric=None,
                    population="assigned",
                    sha256=sha256(canonical_digest_bytes(assignment_counts)).hexdigest(),
                    source=component_source,
                    dimension=dimension,
                )
            )
        source = composite_source(components)
    request = {
        "view": "run",
        "population": "assigned",
        "metrics": None
        if metrics is None or (estimands is not None and set(estimands) == {"compliance"})
        else sorted(metrics),
        "estimands": None if estimands is None else sorted(estimands),
        "registration_id": snapshot.registration_id,
        "alpha": plan.alpha,
        "q": plan.q,
        "trigger_declared": trigger_name is not None,
        "source_identity": source_identity_record,
    }
    if trigger_name is not None:
        request["trigger_name"] = trigger_name
    preimage = {
        "kind": "increment.readout.snapshot",
        "version": 2,
        "collection": "LiftEstimates",
        "source": source,
        "request": request,
    }
    return "sha256:" + sha256(canonical_json_bytes(preimage)).hexdigest(), source


def _checkpoint_roster(src: Any, design: Any, rows, snapshot: SequentialSnapshot):
    observed_by_metric: dict[str, set[str]] = {}
    for row in rows:
        observed_by_metric.setdefault(row.metric, set()).add(row.group_id)
    counts = dict(snapshot.assignment_counts) if snapshot.assignment_counts is not None else None
    arms = tuple(
        sorted(
            {
                snapshot.registration.control_group,
                *(cell.group_id for cell in snapshot.registration.roster),
            }
        )
    )
    has_enrollment_counts = snapshot.assignment_counts is not None
    base_roster = ResolvedRoster(
        arms=arms,
        source="experiment_counts" if has_enrollment_counts else "unknown",
        complete=has_enrollment_counts,
        counts=counts,
        integrity_counts=snapshot.assignment_counts,
        arms_by_metric={},
        known_arms=frozenset(arms),
    )
    return resolve_roster(src, design, "assigned", observed_by_metric, base_roster=base_roster)


def scope_sequential_results(
    src: Any,
    rows: Sequence[LiftEstimate],
    snapshot: SequentialSnapshot,
    *,
    metrics: Sequence[str] | None,
    estimands: Sequence[str] | None,
    base_roster=None,
) -> LiftEstimates:
    """Attach scope to the assigned sequential rows, reporting unsupported triggered cells.

    A registered sequential construction exists only for the assigned population, so a
    declared trigger yields one explicit unsupported record (with a null row) for each
    assigned decision cell instead of silently omitting that population.
    """
    plan = src.context.plan
    design = _require_design(src, "run")
    registration = plan.inference.registration
    trigger_name = src.context.trigger_name

    existing_source_scope = None
    existing_integrity = None
    existing_source = None
    if isinstance(rows, LiftEstimates) and rows.metadata is not None:
        existing_source_scope = next(iter(rows.metadata.scope.by_source.values()), None)
        if existing_source_scope is not None:
            existing_integrity = existing_source_scope.integrity[0]
            existing_source = rows.source
    if existing_source_scope is None:
        roster = (
            base_roster
            if base_roster is not None
            else _checkpoint_roster(src, design, rows, snapshot)
        )
        integrity_counts = roster.integrity_counts
    else:
        integrity_counts = None
    snapshot_id, source = _snapshot_identity(
        plan,
        snapshot,
        metrics,
        estimands,
        trigger_name,
        integrity_counts,
        source=existing_source,
        source_identity_record=source_identity(src),
    )

    family_cells = tuple(sorted({CellKey.from_row(row) for row in rows}, key=cell_order))
    metric_names = None if metrics is None else set(metrics)
    estimand_names = None if estimands is None else set(estimands)
    visible_rows = [
        row
        for row in rows
        if (metric_names is None or row.metric in metric_names or row.estimand == "compliance")
        and (estimand_names is None or row.estimand in estimand_names)
    ]
    assigned_rows = {CellKey.from_row(row): row for row in visible_rows}
    decision_assigned = [cell for cell in assigned_rows if cell.method_role == "decision"]
    triggered_cells = {}
    if trigger_name is not None:
        for cell in decision_assigned:
            triggered_cells[cell.model_copy(update={"analysis_population": "triggered"})] = cell

    roster_groups = {design.control_group} | {cell.group_id for cell in registration.roster}
    roster_source = (
        "declared_allocation" if getattr(design, "allocation", None) else "observed_union"
    )

    output = []
    records: list[CellRecord] = []
    for cell in sorted(assigned_rows, key=cell_order):
        row = assigned_rows[cell].model_copy(
            update={
                "source_snapshot_id": snapshot_id,
                "sampling_available": True,
                "decision_scope_complete": True,
            }
        )
        output.append(row)
        records.append(
            CellRecord(
                cell=cell,
                source_snapshot_id=snapshot_id,
                sampling=SamplingInference(available=True, evidence=_evidence(row, cell)),
            )
        )

    context = {"reason": TRIGGERED_SEQUENTIAL, "analysis_population": "triggered"}
    for cell in sorted(triggered_cells, key=cell_order):
        failure = CellFailure(hypothesis=cell.hypothesis(), code=UNSUPPORTED_CODE, context=context)
        assigned_row = assigned_rows[triggered_cells[cell]]
        output.append(
            LiftEstimate(
                metric=cell.metric,
                group_id=cell.group_id,
                method=cell.method,
                method_role=cell.method_role,
                estimand=cell.estimand,
                analysis_population="triggered",
                value_scale=cell.value_scale,
                alternative=cell.alternative,
                reference_kind="sequential",
                scale="linear",
                inference=assigned_row.inference,
                role=assigned_row.role,
                lift=None,
                source_snapshot_id=snapshot_id,
                failure_code=failure.code,
                failure_context=failure.context,
                sampling_available=False,
                sampling_reason_code=failure.code,
                sampling_reason_context=failure.context,
                decision_scope_complete=False,
                decision_scope_reason_code="readout.scope.decision_incomplete",
                decision_scope_reason_context={"missing_cells": len(triggered_cells)},
            )
        )
        records.append(
            CellRecord(
                cell=cell,
                source_snapshot_id=snapshot_id,
                failure=failure,
                sampling=SamplingInference(
                    available=False, reason_code=failure.code, reason_context=failure.context
                ),
            )
        )

    all_cells = tuple(sorted({*assigned_rows, *triggered_cells}, key=cell_order))
    decision_cells = tuple(sorted({*decision_assigned, *triggered_cells}, key=cell_order))
    complete: dict[Population, bool] = {"assigned": bool(decision_assigned)}
    rosters = [
        PopulationRoster(
            analysis_population="assigned",
            arms=tuple(sorted(roster_groups)),
            source=roster_source,
            complete=bool(decision_assigned),
        )
    ]
    populations = ("assigned",)
    if trigger_name is not None:
        complete["triggered"] = False
        populations = ("assigned", "triggered")
        rosters.append(
            PopulationRoster(
                analysis_population="triggered",
                arms=tuple(sorted(roster_groups)),
                source=roster_source,
                complete=False,
            )
        )
    integrity = (
        existing_integrity
        if existing_integrity is not None
        else assignment_integrity(
            design,
            integrity_counts,
            randomization_grain="cluster" if getattr(src.context, "cluster", None) else "unit",
        )
    )
    output, families = attach_multiplicity_scope(
        output,
        tuple(dict.fromkeys((*family_cells, *all_cells))),
        plan,
        src.context.configs,
        snapshot_id,
        family_populations={"assigned"},
    )
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=all_cells,
        decision_cells=decision_cells,
        rosters=tuple(rosters),
        decision_complete_by_population=complete,
        integrity=(integrity,),
    )
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=all_cells,
        decision_cells=decision_cells,
        populations=populations,
        families=families,
        by_source={snapshot_id: source_scope},
    )

    def cell_record_order(record: Any):
        if not isinstance(record, CellRecord):
            _raise(
                "readout.scope.cell_incomplete",
                cell={"record_type": type(record).__name__, "value": repr(record)},
            )
        return record.source_snapshot_id, cell_order(record.cell)

    metadata = ReadoutMetadata(
        scope=scope,
        cells=tuple(sorted(records, key=cell_record_order)),
    )
    return LiftEstimates(output, metadata=metadata, source=source, sequential_snapshot=snapshot)
