"""Version 1 declarations and selection of their dependencies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from graphlib import CycleError, TopologicalSorter
from keyword import iskeyword
from types import MappingProxyType
from typing import Annotated, Literal, LiteralString, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    FiniteFloat,
    PlainSerializer,
    StringConstraints,
    Tag,
    ValidationError,
    model_validator,
)
from pydantic_core import InitErrorDetails, PydanticCustomError

from .formula import Formula, parse_formula, references

_PATTERN = r"[A-Za-z_][A-Za-z_0-9]*"
type Identifier = Annotated[str, StringConstraints(pattern=rf"^{_PATTERN}$")]
type Relation = Annotated[str, StringConstraints(pattern=rf"^{_PATTERN}(\.{_PATTERN})*$")]
type DimensionRef = Annotated[str, StringConstraints(pattern=rf"^{_PATTERN}(\.{_PATTERN})?$")]
type FrozenMap[K, V] = Annotated[
    Mapping[K, V], AfterValidator(MappingProxyType), PlainSerializer(dict)
]
type ValueType = Literal["string", "boolean", "integer", "number", "date", "timestamp"]
type Scalar = str | bool | int | FiniteFloat
type FilterValue = Scalar | tuple[Scalar, ...]
type Location = tuple[str | int, ...]


def _tuple(value: object) -> object:
    return tuple(value) if isinstance(value, list) else value


def _version(value: object) -> object:
    if type(value) is not int:
        raise ValueError("Use integer schema version 1.")
    return value


class _Declaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, validate_default=True)


class Dimension(_Declaration):
    column: Identifier
    type: ValueType


class Join(_Declaration):
    source: Identifier
    foreign_key: Identifier


class Source(_Declaration):
    table: Relation
    row_key: Identifier
    entities: FrozenMap[Identifier, Identifier] = Field(default_factory=dict)
    time: Identifier | None = None
    dimensions: FrozenMap[Identifier, Dimension] = Field(default_factory=dict)
    joins: FrozenMap[Identifier, Join] = Field(default_factory=dict)


class Population(_Declaration):
    source: Identifier
    entity: Identifier


class Condition(_Declaration):
    dimension: DimensionRef
    op: Literal["eq", "ne", "lt", "le", "gt", "ge", "in"]
    value: Annotated[FilterValue, BeforeValidator(_tuple)]


class Filter(_Declaration):
    source: Identifier
    conditions: Annotated[tuple[Condition, ...], Field(min_length=1), BeforeValidator(_tuple)]


class Measure(_Declaration):
    source: Identifier
    sum: Identifier | None = None
    count_rows: bool | None = None
    filter: Identifier | None = None

    @model_validator(mode="after")
    def _one_aggregation(self) -> Self:
        if self.count_rows is False or (self.sum is None) == (self.count_rows is None):
            raise ValueError("Declare either a sum column or count_rows: true.")
        return self


class Mean(_Declaration):
    measure: Identifier
    population: Identifier


class Conversion(_Declaration):
    event_count: Identifier
    population: Identifier


class FormulaMetric(_Declaration):
    formula: Annotated[Formula, BeforeValidator(parse_formula, json_schema_input_type=str)]


class MeanMetric(_Declaration):
    per_unit_mean: Mean


class ConversionMetric(_Declaration):
    conversion: Conversion


_METRIC_KINDS = ("formula", "per_unit_mean", "conversion")


def _metric_kind(value: object) -> str | None:
    if isinstance(value, dict):
        kinds = [kind for kind in _METRIC_KINDS if kind in value]
        return kinds[0] if len(kinds) == 1 else None
    if isinstance(value, FormulaMetric):
        return "formula"
    if isinstance(value, MeanMetric):
        return "per_unit_mean"
    if isinstance(value, ConversionMetric):
        return "conversion"
    return None


type Metric = Annotated[
    Annotated[FormulaMetric, Tag("formula")]
    | Annotated[MeanMetric, Tag("per_unit_mean")]
    | Annotated[ConversionMetric, Tag("conversion")],
    Discriminator(_metric_kind),
]


class Catalog(_Declaration):
    """Portable declarations, without a data connection or execution settings."""

    version: Annotated[Literal[1], BeforeValidator(_version)]
    sources: FrozenMap[Identifier, Source]
    populations: FrozenMap[Identifier, Population] = Field(default_factory=dict)
    filters: FrozenMap[Identifier, Filter] = Field(default_factory=dict)
    measures: FrozenMap[Identifier, Measure]
    metrics: FrozenMap[Identifier, Metric]

    @model_validator(mode="after")
    def _validate_dependencies(self) -> Self:
        _validate_sources(self)
        _validate_measures(self)
        _validate_metrics(self)
        return self

    def select(self, names: Sequence[str]) -> Catalog:
        """Keep the named metrics and their dependencies, without mutating the catalog."""
        if isinstance(names, str):
            raise TypeError("Pass a sequence of metric names, not a string.")
        metrics: set[str] = set()
        measures: set[str] = set()
        populations: set[str] = set()
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in metrics:
                continue
            metric = self.metrics[name]
            metrics.add(name)
            match metric:
                case FormulaMetric(formula=formula):
                    for ref in references(formula):
                        if ref in self.measures:
                            measures.add(ref)
                        else:
                            pending.append(ref)
                case MeanMetric(per_unit_mean=rule):
                    measures.add(rule.measure)
                    populations.add(rule.population)
                case ConversionMetric(conversion=rule):
                    measures.add(rule.event_count)
                    populations.add(rule.population)

        sources = {self.measures[name].source for name in measures}
        sources.update(self.populations[name].source for name in populations)
        filters = {f for name in measures if (f := self.measures[name].filter) is not None}
        pending = list(sources)
        while pending:
            for join in self.sources[pending.pop()].joins.values():
                if join.source not in sources:
                    sources.add(join.source)
                    pending.append(join.source)
        return self.model_copy(
            update={
                "sources": _subset(self.sources, sources),
                "populations": _subset(self.populations, populations),
                "filters": _subset(self.filters, filters),
                "measures": _subset(self.measures, measures),
                "metrics": _subset(self.metrics, metrics),
            }
        )


def _subset[V](mapping: Mapping[str, V], names: set[str]) -> Mapping[str, V]:
    return MappingProxyType({name: value for name, value in mapping.items() if name in names})


def _bad(code: LiteralString, location: Location, detail: str) -> ValidationError:
    error: InitErrorDetails = {
        "type": PydanticCustomError(code, "{detail}", {"detail": detail}),
        "loc": location,
        "input": None,
    }
    return ValidationError.from_exception_data("Catalog", [error])


def _lookup[V](mapping: Mapping[str, V], name: str, location: Location) -> V:
    try:
        return mapping[name]
    except KeyError:
        raise _bad("unknown_reference", location, f"No declaration named {name!r}.") from None


def _validate_sources(catalog: Catalog) -> None:
    for name, source in catalog.sources.items():
        for alias, join in source.joins.items():
            _lookup(catalog.sources, join.source, ("sources", name, "joins", alias, "source"))
    for name, population in catalog.populations.items():
        location = ("populations", name)
        source = _lookup(catalog.sources, population.source, (*location, "source"))
        key = _lookup(source.entities, population.entity, (*location, "entity"))
        if key != source.row_key:
            raise _bad(
                "invalid_population",
                (*location, "entity"),
                f"Roster {source.table!r} must be keyed by {key!r} for entity {population.entity!r}.",
            )


def _validate_measures(catalog: Catalog) -> None:
    for name, filt in catalog.filters.items():
        location = ("filters", name)
        source = _lookup(catalog.sources, filt.source, (*location, "source"))
        for index, condition in enumerate(filt.conditions):
            path = (*location, "conditions", index)
            alias, dot, dim = condition.dimension.partition(".")
            if dot:
                join = _lookup(source.joins, alias, (*path, "dimension"))
                dimensions = catalog.sources[join.source].dimensions
            else:
                dimensions = source.dimensions
                dim = alias
            dtype = _lookup(dimensions, dim, (*path, "dimension")).type
            _validate_condition(condition, dtype, path)
    for name, measure in catalog.measures.items():
        location = ("measures", name)
        _lookup(catalog.sources, measure.source, (*location, "source"))
        if measure.filter is not None:
            filt = _lookup(catalog.filters, measure.filter, (*location, "filter"))
            if filt.source != measure.source:
                raise _bad(
                    "wrong_filter_source",
                    (*location, "filter"),
                    "Use a filter on the measure's source.",
                )


def _validate_condition(condition: Condition, dtype: ValueType, path: Location) -> None:
    value = condition.value
    if isinstance(value, tuple):
        valid = condition.op == "in" and bool(value) and all(_matches(v, dtype) for v in value)
    else:
        valid = condition.op != "in" and _matches(value, dtype)
    if not valid:
        raise _bad(
            "invalid_filter_value",
            (*path, "value"),
            f"Use a {dtype} value (a nonempty list for 'in').",
        )
    if dtype == "boolean" and condition.op not in ("eq", "ne", "in"):
        raise _bad(
            "invalid_filter_operator", (*path, "op"), "Use eq, ne or in for a boolean dimension."
        )


def _matches(value: Scalar, dtype: ValueType) -> bool:
    match dtype:
        case "boolean":
            return type(value) is bool
        case "integer":
            return type(value) is int
        case "number":
            return type(value) in (float, int)
        case "string":
            return isinstance(value, str)
        case "date" | "timestamp":
            if not isinstance(value, str):
                return False
            try:
                if dtype == "date":
                    date.fromisoformat(value)
                    return len(value) == 10 and value[4] == value[7] == "-"
                datetime.fromisoformat(value)
                return len(value) > 10 and value[4] == value[7] == "-" and value[10] in "T "
            except ValueError:
                return False


def _validate_metrics(catalog: Catalog) -> None:
    for collection, entries in (("measures", catalog.measures), ("metrics", catalog.metrics)):
        for name in entries:
            if iskeyword(name):
                raise _bad(
                    "invalid_reference_name",
                    (collection, name),
                    f"{name!r} is reserved in formulas.",
                )
    graph: dict[str, set[str]] = {}
    for name, metric in catalog.metrics.items():
        location = ("metrics", name)
        if name in catalog.measures:
            raise _bad("duplicate_name", location, f"{name!r} is already a measure.")
        match metric:
            case FormulaMetric(formula=formula):
                path = (*location, "formula")
                graph[name] = set()
                for ref in references(formula):
                    if ref not in catalog.measures:
                        _lookup(catalog.metrics, ref, path)
                        graph[name].add(ref)
            case MeanMetric(per_unit_mean=rule):
                _validate_population_metric(catalog, rule, (*location, "per_unit_mean"))
            case ConversionMetric(conversion=rule):
                _validate_population_metric(catalog, rule, (*location, "conversion"))
    try:
        TopologicalSorter(graph).prepare()
    except CycleError as exc:
        cycle = exc.args[1]
        raise _bad("formula_cycle", ("metrics", cycle[0], "formula"), " -> ".join(cycle)) from None


def _validate_population_metric(catalog: Catalog, rule: Mean | Conversion, path: Location) -> None:
    population = _lookup(catalog.populations, rule.population, (*path, "population"))
    if isinstance(rule, Conversion):
        measure = _lookup(catalog.measures, rule.event_count, (*path, "event_count"))
        if not measure.count_rows:
            raise _bad(
                "invalid_conversion",
                (*path, "event_count"),
                "Use a qualifying-row count for conversion.",
            )
    else:
        measure = _lookup(catalog.measures, rule.measure, (*path, "measure"))
    if population.entity not in catalog.sources[measure.source].entities:
        raise _bad(
            "wrong_population_entity",
            (*path, "population"),
            "The measure must identify the roster's entity.",
        )
