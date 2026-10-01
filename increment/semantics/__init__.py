"""Semantic layer — typed, validated A/B test definitions.

This package defines the structural vocabulary (fact sources, metrics,
exposures, experiments) as a set of pydantic models that map 1:1 onto
the metric types documented in ``docs/guides/metric-types.md``.

It is a **pure data layer**: it defines the schema, validates internal
consistency, and loads YAML definitions. It MUST NOT import ibis, build
queries, or touch a database.
"""

from increment.errors import DefinitionError
from increment.semantics.loader import load
from increment.semantics.models import (
    ActiveMetric,
    ConversionMetric,
    Definitions,
    Experiment,
    Exposure,
    Fact,
    FactRef,
    FactSource,
    Filter,
    MeanMetric,
    Measure,
    Metric,
    MetricBase,
    Property,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
    SourceColumn,
    TotalMetric,
    Winsorization,
)

__all__ = [
    "ActiveMetric",
    "ConversionMetric",
    "DefinitionError",
    "Definitions",
    "Experiment",
    "Exposure",
    "Fact",
    "FactRef",
    "FactSource",
    "Filter",
    "MeanMetric",
    "Measure",
    "Metric",
    "MetricBase",
    "Property",
    "QuantileMetric",
    "RatioMetric",
    "RetentionMetric",
    "SourceColumn",
    "TotalMetric",
    "Winsorization",
    "load",
]
