"""Project source checkpoints through registered sequential decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from increment.sequential_source import source_snapshot
from increment.sequential_state import require_public_laws, sequential_refuse

if TYPE_CHECKING:
    from increment.estimation.results import LiftEstimate


def sequential_readout(source, *, metrics=None, estimands=None, previous=None):
    """Run the registered decision and selection path on a source checkpoint."""
    from increment.estimation.sequential_runtime import selected_snapshot_results

    inference = source.context.plan.inference
    registration = getattr(inference, "registration", None)
    if registration is None:
        sequential_refuse("source.invalid", "public sequential readout requires registration")
    require_public_laws(registration.models, "sequential readout")
    snapshot = source_snapshot(source, previous=previous)
    rows = selected_snapshot_results(
        snapshot,
        inference,
        nominal_alpha=source.context.plan.alpha,
    )
    result = []
    names = {m.name for m in metrics} if metrics is not None else None
    for row in rows:
        cell = row.require_sequential_result().checkpoint.cell
        procedure = source.context.plan.procedures.get(cell.metric)
        row = row.model_copy(update={"role": procedure.role if procedure is not None else None})
        if (names is None or cell.metric in names or cell.estimand == "compliance") and (
            estimands is None or cell.estimand in estimands
        ):
            result.append(row)
    return result


def sequential_asof_readout(source, *, metrics=None, estimands=None) -> list[LiftEstimate]:
    snapshot = source_snapshot(source)
    if snapshot.reveal_cursor is None:
        sequential_refuse(
            "route.unsupported", "as-of reporting requires a labeled finalized checkpoint"
        )
    return [
        row.model_copy(update={"ds": snapshot.reveal_cursor})
        for row in sequential_readout(source, metrics=metrics, estimands=estimands)
    ]
