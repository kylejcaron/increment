"""Project source checkpoints through registered sequential decisions."""

from __future__ import annotations

from increment._analysis_config import effective_methods, validate_scalar_route
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
    configs=None,
):
    """Run the registered decision and selection path on a source checkpoint."""
    from increment.estimation.sequential_runtime import selected_snapshot_results

    inference = population_policy(source.context.plan.inference, population)
    registration = getattr(inference, "registration", None)
    if registration is None:
        sequential_refuse("source.invalid", "public sequential readout requires registration")
    configs = source.context.configs if configs is None else configs
    validate_scalar_route(
        configs,
        route="sequential_readout",
        representation="registered",
        design=source.context.design,
        retained_configs=source.context.configs,
    )
    by_metric = {config.metric.name: config for config in configs}
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
        if not (
            _include_unrequested
            or (
                (names is None or cell.metric in names or cell.estimand == "compliance")
                and (estimands is None or cell.estimand in estimands)
            )
        ):
            continue
        config = by_metric.get(cell.metric)
        if config is None:
            result.append(row)
            continue
        for index, method in enumerate(effective_methods(config, design=source.context.design)):
            result.append(
                row.model_copy(
                    update={
                        "method": method.name,
                        "method_role": "decision" if index == 0 else "sensitivity",
                    }
                )
            )
    return result
