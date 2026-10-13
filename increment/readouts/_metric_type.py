"""Stamp scoped rows with the declared type of their metric.

The value is the user's declared ``Metric.type``, read from the metric
definitions in scope; it never passes through the estimation path, which
folds report-only ``total``/``active`` metrics onto the ``mean`` axis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from increment.readouts._common import _raise

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from increment._literals import MetricType
    from increment.semantics.models import Metric


def declared_metric_types(metrics: Iterable[Metric]) -> dict[str, MetricType]:
    """The declared type of every metric in scope, keyed by metric name."""
    return {metric.name: metric.type for metric in metrics}


def declared_metric_type(row: Any, metric_types: Mapping[str, MetricType]) -> MetricType | None:
    """The declared type behind ``row.metric``; ``row`` may be a result row or a ``CellKey``.

    The design-level compliance row estimates uptake rather than a declared
    metric, so it stays unknown even when a declared metric shares its
    ``"uptake"`` name. Any other row names a metric the scope must declare;
    a row outside the declared metrics is a projection defect, not an
    unknown type.
    """
    if getattr(row, "estimand", None) == "compliance":
        return None
    try:
        return metric_types[row.metric]
    except KeyError:
        _raise(
            "readout.scope.metric_undeclared",
            metric=row.metric,
            declared=tuple(sorted(metric_types)),
        )


__all__ = ["declared_metric_type", "declared_metric_types"]
