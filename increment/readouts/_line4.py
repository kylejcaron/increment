from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from increment._evidence_types import BreakoutIdentity, EvidenceScope
from increment._source_types import declared_evidence
from increment.kernel.context import EvidenceExecutionContext, ExecutionMode

if TYPE_CHECKING:
    from increment._analysis_execution import Analysis


def line4_context(
    analysis: Analysis,
    *,
    coherence: ExecutionMode = "captured",
    population: Literal["assigned", "triggered"] = "triggered",
    include_breakouts: bool = False,
) -> EvidenceExecutionContext:
    """Declare the evidence scope shared by encouragement and triggered readers."""
    source = analysis._src
    design = analysis._design
    experiment = getattr(analysis._state, "experiment", None)
    if experiment is None and getattr(analysis._state, "family", None) == "arm_moments":
        experiment = getattr(analysis._state.source, "artifact_experiment", None)
    uptake = getattr(design, "uptake", None)
    if not include_breakouts:
        breakouts = ()
    elif experiment is None:
        breakouts = tuple(
            BreakoutIdentity(name, None) for name in declared_evidence(source).breakouts
        )
    else:
        breakouts = tuple(
            BreakoutIdentity(item.property, item.source) for item in experiment.breakouts
        )
    from increment.readouts._run_acquisition import RunAcquisitionOwner

    provider = RunAcquisitionOwner.provider_for(source)
    populations = tuple(sorted(declared_evidence(source).populations))
    scope_population = population if population in populations else "assigned"
    return EvidenceExecutionContext(
        provider,
        EvidenceScope(
            metrics=tuple(analysis._metrics),
            population=scope_population,
            populations=populations,
            configs=tuple(source.context.configs),
            breakouts=breakouts,
            uptake_facts=() if uptake is None else (uptake.fact,),
            snapshot=True,
            execution=True,
            include_breakouts=include_breakouts,
        ),
        run_id=uuid4().hex,
        coherence=coherence,
    )


def run_scoped_line4(
    analysis: Analysis,
    operation: Callable[[], Any] | None = None,
    *,
    coherence: ExecutionMode = "captured",
    population: Literal["assigned", "triggered"] = "triggered",
    include_breakouts: bool = False,
    readouts: Sequence[Literal["run", "daily", "daily_lift", "asof", "asof_lift"]] = (
        "run",
        "daily",
        "daily_lift",
        "asof",
        "asof_lift",
    ),
    on_context: Callable[[EvidenceExecutionContext], None] | None = None,
) -> dict[str, Any] | Any:
    """Run related operations under one owning evidence context."""
    if operation is None:
        analysis._preflight_line4_views(population=population, readouts=readouts)
    context = line4_context(
        analysis,
        coherence=coherence,
        population=population,
        include_breakouts=include_breakouts,
    )
    operations = {
        "run": lambda: analysis.run(population=population),
        "daily": lambda: analysis.run_daily(population=population),
        "daily_lift": lambda: analysis.run_daily_lift(population=population),
        "asof": lambda: analysis.run_asof(population=population),
        "asof_lift": lambda: analysis.run_asof_lift(population=population),
    }
    with context.evidence_scope(), analysis._scoped_execution_context(context):
        if on_context is not None:
            on_context(context)
        if operation is not None:
            return operation()
        return {name: operations[name]() for name in readouts}
