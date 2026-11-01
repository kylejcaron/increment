"""Exploratory metric selection and the sibling source that reads them."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from increment._analysis_config import select_exploratory_metrics, select_metrics
from increment._source_admission import _require_analysis_operation, _require_arm_binding
from increment._source_operations import ExploratorySourceOperation
from increment.decision import DefinitionsArmAnalysisState
from increment.errors import CapabilityError, RefusalSpec, UnsupportedRequestError, refuse
from increment.readouts._evidence import NATIVE_ROUTE, day_axis_route
from increment.readouts._execution_context import current_execution_context

if TYPE_CHECKING:
    from increment._binding import SourceBinding
    from increment._literals import Correction
    from increment.breakout.estimates import DailyLiftEstimates
    from increment.semantics.models import Metric

EXPLORATORY_SOURCE_LIMITED = RefusalSpec(
    "facade.analysis.exploratory_metrics_source_limited",
    CapabilityError,
    template="{method}: exploratory_metrics adds metrics from the saved definitions, which this source does not carry -- build the experiment with from_definitions.",
)
_EXPLORATORY_SEQUENTIAL = RefusalSpec(
    "facade.analysis.exploratory_metrics_sequential",
    UnsupportedRequestError,
    template="{method}: exploratory_metrics {names!r} are chosen after data are visible, so no registered sequential model covers them under {inference}; read the declared metrics alone, or use a fixed-horizon plan.",
)


def available_metrics(binding: SourceBinding) -> list[Metric]:
    """Saved metrics on the experiment's unit that the experiment does not declare."""
    state = binding.state
    if not isinstance(state, DefinitionsArmAnalysisState):
        refuse(EXPLORATORY_SOURCE_LIMITED, method="available_metrics")
    declared = {*state.experiment.metric_names, *(metric.name for metric in binding.metrics)}
    return [
        metric
        for metric in state.definitions.metrics
        if metric.name not in declared
        and metric.type not in ("total", "active")
        and getattr(metric, "entity", None) == state.experiment.unit
    ]


def select_exploratory(
    binding: SourceBinding, names: Sequence[str] | None, *, caller: str
) -> list[Metric]:
    """Resolve ``exploratory_metrics=`` before any query; ``[]`` when none were named."""
    if not names:
        return []
    if not isinstance(binding.state, DefinitionsArmAnalysisState):
        refuse(EXPLORATORY_SOURCE_LIMITED, method=caller)
    added = select_exploratory_metrics(
        available_metrics(binding),
        {*binding.experiment.metric_names, *(metric.name for metric in binding.metrics)},
        names,
        caller=caller,
    )
    inference = binding.plan.inference
    if getattr(inference, "registration", None) is not None:
        refuse(
            _EXPLORATORY_SEQUENTIAL,
            method=caller,
            names=[metric.name for metric in added],
            inference=type(inference).__name__,
        )
    return added


def exploratory_sibling(
    binding: SourceBinding, added: Sequence[Metric], *, caller: str
) -> SourceBinding:
    """The binding reading from a sibling source that also carries *added*."""
    _require_arm_binding(binding, caller)
    source = _require_analysis_operation(
        binding.source,
        "exploratory_source",
        ExploratorySourceOperation,
        message=f"{caller}(exploratory_metrics=...) needs an experiment built by from_definitions.",
    )
    derived = binding.with_source(source.exploratory_source(metrics=added))
    context = current_execution_context()
    if context is not None:
        context.declare_source_alias(derived.source, parent=binding.source)
    return derived


def select_day_axis_metrics(
    binding: SourceBinding,
    metrics: Sequence[str | Metric] | None,
    *,
    caller: str,
    added: Sequence[Metric] = (),
) -> list[Metric]:
    """Select day-axis metrics; only the native route accepts undeclared Metric objects."""
    native = day_axis_route(binding.source) == NATIVE_ROUTE
    selected = select_metrics(
        binding.metrics, metrics, caller=caller, allow_undeclared_objects=native
    )
    if not added:
        return selected
    # Added metrics follow the declared ones; a shared name is a duplicate.
    return select_metrics((), [*selected, *added], caller=caller, allow_undeclared_objects=True)


def stamp_exploratory(
    rows: DailyLiftEstimates,
    added: Sequence[Metric],
    *,
    correction: Correction | None = None,
) -> DailyLiftEstimates:
    """Mark added metrics exploratory under the effective view correction."""
    from increment.breakout.estimates import DailyLiftEstimates

    names = {metric.name for metric in added}
    if not names:
        return rows
    from increment.estimation.multiplicity import stamp_multiplicity_status

    return DailyLiftEstimates(
        stamp_multiplicity_status(
            [
                row.model_copy(update={"role": "exploratory"}) if row.metric in names else row
                for row in rows
            ],
            correction=correction,
        ),
        metadata=rows.metadata,
    )
