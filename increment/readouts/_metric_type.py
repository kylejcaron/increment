"""Stamp scoped rows with the declared type of their metric.

The value is the user's declared ``Metric.type``, read from the metric
definitions in scope; it never passes through the estimation path, which
folds report-only ``total``/``active`` metrics onto the ``mean`` axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from increment._literals import MetricType
    from increment.semantics.models import Metric


def declared_metric_types(metrics: Iterable[Metric]) -> dict[str, MetricType]:
    """The declared type of every metric in scope, keyed by metric name."""
    return {metric.name: metric.type for metric in metrics}


def declared_metric_type(row: Any, metric_types: Mapping[str, MetricType]) -> MetricType | None:
    """The declared type behind ``row.metric``, or None when the row has none.

    The design-level compliance row estimates uptake rather than a declared
    metric, so it stays unknown even when a declared metric shares its
    ``"uptake"`` name. ``row`` may be a result row or a ``CellKey``.
    """
    if getattr(row, "estimand", None) == "compliance":
        return None
    return metric_types.get(row.metric)


__all__ = ["declared_metric_type", "declared_metric_types"]
