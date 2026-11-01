"""The registered sequential family's shared evidence scope over one binding."""

from __future__ import annotations

from typing import TYPE_CHECKING

from increment._evidence_types import EvidenceScope
from increment._source_admission import _arm_experiment
from increment._source_types import declared_evidence, is_custom_source

if TYPE_CHECKING:
    from increment._binding import SourceBinding


def sequential_scope(binding: SourceBinding, *, include_breakouts: bool = False) -> EvidenceScope:
    """Declare the complete registered family and its covariate dependencies."""
    from dataclasses import replace

    from increment._analysis_config import union_covariates

    source = binding.source
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
            expanded = replace(config, covariates=covariates)
            if expanded not in declared_configs:
                declared_configs.append(expanded)
    experiment = _arm_experiment(binding)
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
