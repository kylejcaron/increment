"""Source-agnostic orchestration for experiment health checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from increment._evidence_types import EvidenceScope
from increment.estimation.health import ExperimentHealth
from increment.evidence import evidence_provider


def check_health(
    source: Any,
    *,
    metrics: Sequence[Any],
    collect: Callable[[Any], ExperimentHealth],
) -> ExperimentHealth:
    """Run health collection against a source-owned snapshot when available."""
    provider = evidence_provider(source)
    population = (
        "triggered"
        if "triggered" in provider.evidence.populations
        and "assigned" not in provider.evidence.populations
        else "assigned"
    )
    scoped = provider.scope(
        EvidenceScope(
            metrics=tuple(metrics),
            population=population,
            snapshot=True,
            include_breakouts=True,
        )
    )
    with scoped.source as pinned:
        return collect(pinned)


__all__ = ["check_health"]
