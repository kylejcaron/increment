"""Project source checkpoints through registered sequential decisions."""

from __future__ import annotations

from increment.sequential_source import population_policy, source_snapshot
from increment.sequential_state import require_public_laws, sequential_refuse


def sequential_readout(
    source,
    *,
    metrics=None,
    estimands=None,
    previous=None,
    _include_unrequested=False,
    population="assigned",
):
    """Run the registered decision and selection path on a source checkpoint."""
    from increment.estimation.sequential_runtime import selected_snapshot_results

    inference = population_policy(source.context.plan.inference, population)
    registration = getattr(inference, "registration", None)
    if registration is None:
        sequential_refuse("source.invalid", "public sequential readout requires registration")
    require_public_laws(registration.models, "sequential readout")
    snapshot = source_snapshot(source, previous=previous).chain(population)
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
        if _include_unrequested or (
            (names is None or cell.metric in names or cell.estimand == "compliance")
            and (estimands is None or cell.estimand in estimands)
        ):
            result.append(row)
    return result
