"""Register the two-stage assignment and metric without reading either draw or its outcomes."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from increment._binding import ExperimentReader, SpilloverBinding
from increment._canonical import canonical_digest_bytes
from increment._evidence_loaders import freeze_spillover
from increment._spillover_design import refuse_input
from increment._spillover_registration import spillover_registration
from increment.kernel.contracts import View
from increment.kernel.experiment import Experiment, SourceReference

if TYPE_CHECKING:
    from increment._evidence_loaders import SpilloverSource
    from increment._literals import Backend
    from increment._spillover_design import Spillover
    from increment.semantics.models import ConversionMetric
    from increment.semantics.models import Experiment as ExperimentDefinition


def bind_spillover(
    source: SpilloverSource,
    *,
    design: Spillover,
    metric: ConversionMetric,
    alpha: float,
    experiment_id: str,
    recipe: object,
    source_kind: str,
    input_backend: Backend | None = None,
    definition: ExperimentDefinition | None = None,
    definitions_guard: Callable[[], None] | None = None,
) -> tuple[Experiment, ExperimentReader]:
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        refuse_input("spillover.design.invalid", "experiment_id")
    if metric.window_days is None:
        refuse_input("spillover.design.invalid", "window_days")
    law = design.law(metric.name, metric.window_days, alpha)
    population = freeze_spillover(source, law)
    declaration = {
        **law.declaration(),
        "kind": "spillover",
        "protocol": "spillover-fixed-count-v2",
        "method": "two-stage-fixed-bet-v3",
        "control_group": design.control_group,
        "offered_group": design.offered_group,
    }
    metrics = {metric.name: metric.model_dump(mode="json")}
    source_digest = hashlib.sha256(canonical_digest_bytes(recipe)).hexdigest()
    source_refs = (SourceReference("source", f"{source_kind}://{experiment_id}", source_digest),)
    # Power depends on group sizes/quotas, not their identifiers or either realized draw.
    context: Mapping[str, Any] = {
        "family": "spillover",
        "roster_digest": population.roster_digest,
        "planning_groups": tuple((g.stratum, g.size, g.offers) for g in population.groups),
    }
    experiment = Experiment(
        experiment_id=experiment_id,
        design=declaration,
        metrics=metrics,
        registration=spillover_registration(
            experiment_id, declaration, metrics, source_refs, context
        ),
        source_refs=source_refs,
        views=(View(view="spillover"),),
        analysis_context=context,
    )
    return experiment, ExperimentReader(
        SpilloverBinding(source, population, metric, definition, input_backend),
        experiment,
        definitions_guard=definitions_guard,
    )
