"""Project declared multiplicity provenance onto scoped readout rows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from increment.decision import MultiplicityFamily
from increment.estimation.multiplicity import multiplicity_status
from increment.estimation.readout_types import CellKey, FamilyScope, cell_order, family_identity


def attach_multiplicity_scope(
    rows: Sequence[Any],
    cells: Sequence[CellKey],
    plan: Any,
    configs: Sequence[Any],
    snapshot_id: str,
    *,
    view: str = "run",
    dimension: str | None = None,
    source: str | None = None,
    family_populations: set[str] | None = None,
) -> tuple[list[Any], tuple[FamilyScope, ...]]:
    """Attach row provenance and the unchanged source family membership.

    Membership is resolved from the predeclared source cells and effective
    per-metric configuration, never from surviving or displayed rows.
    """
    configs_by_metric = {config.metric.name: config for config in configs}
    family_cells = {}
    cell_family = {}
    family_by_key = {}
    row_by_cell = {CellKey.from_row(row): row for row in rows}
    for cell in cells:
        procedure = plan.procedures.get(cell.metric)
        row = row_by_cell.get(cell)
        role = (
            procedure.role
            if plan.declared and procedure is not None
            else None
            if row is None
            else row.role
        )
        membership = None if procedure is None else procedure.family
        config = configs_by_metric.get(cell.metric)
        effective_member = (
            membership is not None
            and membership.member
            and config is not None
            and (family_populations is None or cell.analysis_population in family_populations)
        )
        if effective_member and cell.method_role != "decision":
            cell_family[cell] = None
            continue
        family = (
            membership.family
            if effective_member and isinstance(membership.family, MultiplicityFamily)
            else None
        )
        if role is None or (role == "secondary" and family is None):
            cell_family[cell] = None
            continue
        key = (
            cell.analysis_population,
            role,
            family.name if family is not None else role,
            family,
        )
        family_cells.setdefault(key, []).append(cell)
        family_by_key[key] = family
        cell_family[cell] = key

    family_scopes = []
    family_ids = {}
    for key, members in family_cells.items():
        cell_population, _, name, family = key
        family_id = family_identity(snapshot_id, cell_population, view, dimension, source, name)
        family_ids[key] = family_id
        family_scopes.append(
            FamilyScope(
                family_id=family_id,
                analysis_population=cell_population,
                source_snapshot_id=snapshot_id,
                view=view,
                dimension=dimension,
                source=source,
                name=name,
                family=family,
                members=tuple(sorted(set(members), key=cell_order)),
                complete=True,
            )
        )

    output = []
    for row in rows:
        cell = CellKey.from_row(row)
        key = cell_family[cell]
        procedure = plan.procedures.get(row.metric)
        role = procedure.role if plan.declared and procedure is not None else row.role
        family = None if key is None else family_by_key[key]
        output.append(
            row.model_copy(
                update={
                    "role": role,
                    "family_id": None if key is None else family_ids[key],
                    "multiplicity_status": multiplicity_status(role, family),
                }
            )
        )
    return output, tuple(sorted(family_scopes, key=lambda family: family.family_id))
