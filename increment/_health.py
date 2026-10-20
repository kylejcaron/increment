"""Source-agnostic orchestration for experiment health checks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from increment._source_operations import ReadoutSnapshotOperation
from increment.estimation.health import ExperimentHealth


def check_health(
    source: Any,
    *,
    metrics: Sequence[Any],
    collect: Callable[[Any], ExperimentHealth],
) -> ExperimentHealth:
    """Run health collection against a source-owned snapshot when available."""
    if not isinstance(source, ReadoutSnapshotOperation):
        return collect(source)
    with source.readout_snapshot(
        metrics=metrics,
        population="assigned",
        include_breakouts=True,
    ) as pinned:
        return collect(pinned)


__all__ = ["check_health"]
