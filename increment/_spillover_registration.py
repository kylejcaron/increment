"""Commit the two-stage design and all inputs used to plan it offline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from increment.kernel.contracts import Registration, _refuse, contract_hash

if TYPE_CHECKING:
    from increment.kernel.experiment import Experiment, SourceReference


def spillover_registration(
    experiment_id: str,
    design: Mapping[str, object],
    metrics: Mapping[str, object],
    source_refs: Sequence[SourceReference],
    context: Mapping[str, object],
) -> Registration:
    return Registration(
        registration_hash=contract_hash(
            "registration",
            {
                "experiment_id": experiment_id,
                "design": design,
                "metrics": metrics,
                "sources": [ref.canonical_payload() for ref in source_refs],
                "planning": context,
            },
        )
    )


def require_registered_spillover(experiment: Experiment) -> None:
    if (
        experiment.design.get("protocol") != "spillover-fixed-count-v2"
        or experiment.design.get("method") != "two-stage-fixed-bet-v3"
    ):
        _refuse("kernel.request.invalid", field="the saved spillover method is not supported")
    if experiment.registration != spillover_registration(
        experiment.experiment_id,
        experiment.design,
        experiment.metrics,
        experiment.source_refs,
        experiment.analysis_context,
    ):
        _refuse(
            "kernel.request.invalid",
            field="the saved spillover inputs no longer match their registration",
        )
