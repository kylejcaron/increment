from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import uuid4

from increment._evidence_types import EvidenceScope
from increment._source_types import declared_evidence, is_custom_source
from increment.kernel.context import EvidenceExecutionContext, ExecutionMode

if TYPE_CHECKING:
    from increment._analysis_execution import Analysis


def sequential_scope(analysis: Analysis, *, include_breakouts: bool = False) -> EvidenceScope:
    """Declare the complete registered family and its covariate dependencies."""
    from dataclasses import replace

    from increment._analysis_config import union_covariates

    source = analysis._src
    design = source.context.design
    uptake = getattr(design, "uptake", None)
    declared_configs = list(source.context.configs)
    for config in source.context.configs:
        methods = config.source_methods or (
            config.decision_method,
            *config.sensitivity_methods,
        )
        covariates = union_covariates((config,), (methods,))
        if covariates and config.covariates != covariates:
            declared_configs.append(replace(config, covariates=covariates))
    experiment = getattr(analysis, "_experiment", None)
    populations = (
        ("assigned", "triggered")
        if getattr(experiment, "trigger", None) is not None
        else tuple(sorted(declared_evidence(source).populations))
        if is_custom_source(source)
        else ()
    )
    return EvidenceScope(
        metrics=tuple(source.context.metrics),
        configs=tuple(declared_configs),
        populations=populations,
        uptake_facts=() if uptake is None else (uptake.fact,),
        snapshot=True,
        execution=True,
        include_breakouts=include_breakouts,
    )


def run_scoped_line3[T](
    analysis: Analysis,
    operation: Callable[[], T],
    *,
    coherence: ExecutionMode = "captured",
    include_breakouts: bool = False,
    on_context: Callable[[EvidenceExecutionContext], None] | None = None,
) -> T:
    """Run sequential captures and readouts under one family-wide evidence owner."""
    from increment.readouts._provider import native_scope_provider

    source = analysis._src
    provider = source if is_custom_source(source) else native_scope_provider(source)
    context = EvidenceExecutionContext(
        provider,
        sequential_scope(analysis, include_breakouts=include_breakouts),
        run_id=uuid4().hex,
        coherence=coherence,
    )
    with context.evidence_scope(), analysis._scoped_execution_context(context):
        result = operation()
        if on_context is not None:
            on_context(context)
        return result
