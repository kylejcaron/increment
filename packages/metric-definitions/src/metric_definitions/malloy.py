"""Compile shared metrics into native Malloy sources without opening a connection."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from graphlib import TopologicalSorter
from math import isfinite
from types import MappingProxyType
from typing import NamedTuple

from .formula import Binary, Formula, Literal, Negation, Reference, references
from .models import (
    Catalog,
    Condition,
    ConversionMetric,
    FormulaMetric,
    MeanMetric,
    Scalar,
    ValueType,
)

__all__ = ["MalloyExportError", "export_malloy"]

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_INTERNAL = "_md_"
_OPERATORS = {"eq": "=", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


class MalloyExportError(ValueError):
    """A selected definition cannot be compiled without changing its meaning."""

    def __init__(self, code: str, metric: str | None, requirement: str) -> None:
        super().__init__(code, metric, requirement)

    @property
    def code(self) -> str:
        """Return the cause of the refusal."""
        return self.args[0]

    @property
    def context(self) -> Mapping[str, str | None]:
        """Return the metric, requirement, and target of the refusal."""
        return MappingProxyType(
            {"metric": self.args[1], "requirement": self.args[2], "target": "malloy"}
        )

    def __str__(self) -> str:
        _, metric, requirement = self.args
        subject = f"metric {metric!r}" if metric is not None else "the selection"
        return f"Malloy cannot export {subject}: {requirement}"


def _name(value: str) -> str:
    # Catalog identifiers are validated before export; connection names are checked below.
    return f"`{value}`"


def _column(value: str) -> str:
    return _name(f"{_INTERNAL}col_{value}")


def _literal(value: Scalar, dtype: ValueType, metric: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        if isinstance(value, float) and not isfinite(value):
            raise MalloyExportError("malloy.unsupported_literal", metric, "Use a finite number.")
        return repr(value)
    if "\0" in value:
        raise MalloyExportError(
            "malloy.unsupported_literal", metric, "Use strings without NUL characters."
        )
    if dtype == "date":
        return f"{json.dumps(value)}::date"
    if dtype == "timestamp":
        fractions = re.findall(r"[.,]([0-9]+)", value)
        if any(fraction[6:].strip("0") for fraction in fractions):
            raise MalloyExportError(
                "malloy.unsupported_literal", metric, "Use timestamps with microsecond precision."
            )
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is not None:
            try:
                timestamp = timestamp.astimezone(UTC)
            except OverflowError as exc:
                raise MalloyExportError(
                    "malloy.unsupported_literal",
                    metric,
                    "The timestamp is outside the UTC date range.",
                ) from exc
            return f"{json.dumps(timestamp.isoformat(' '))}::timestamptz"
        return f"{json.dumps(timestamp.isoformat(' '))}::timestamp"
    return json.dumps(value, ensure_ascii=False)


def _condition(condition: Condition, dtype: ValueType, metric: str) -> str:
    field = ".".join(map(_name, condition.dimension.split(".")))
    value = condition.value
    if isinstance(value, tuple):
        comparison = " or ".join(f"{field} = {_literal(item, dtype, metric)}" for item in value)
    else:
        comparison = f"{field} {_OPERATORS[condition.op]} {_literal(value, dtype, metric)}"
    return f"({field} is not null and ({comparison}))"


def _arithmetic(formula: Formula, metric: str) -> str:
    match formula:
        case Literal(value):
            return _literal(value, "number", metric)
        case Reference(name):
            return _name(name)
        case Negation(operand):
            return f"(-({_arithmetic(operand, metric)}))"
        case Binary(operator, left, right):
            lhs, rhs = _arithmetic(left, metric), _arithmetic(right, metric)
            if operator == "/":
                rhs = f"nullif({rhs}, 0)"
            return f"({lhs} {operator} {rhs})"


def _declaration(name: str, base: str, fields: Iterable[str]) -> str:
    return f"source: {_name(name)} is {base} extend {{\n" + "\n".join(fields) + "\n}"


class _Roster(NamedTuple):
    name: str


# Literal-only formulas can be used on either grain without introducing a row source.
type _Grain = str | _Roster | None


class _Compiler:
    def __init__(self, catalog: Catalog, connection: str) -> None:
        self.catalog = catalog
        self.connection = _name(connection)
        self.blocks: list[str] = []
        self.sources: set[str] = set()
        self.dimensions: dict[str, dict[str, str]] = {}
        self.measures: dict[str, list[str]] = defaultdict(list)
        for name, measure in catalog.measures.items():
            self.measures[measure.source].append(name)
        self.grains: dict[str, _Grain] = {name: m.source for name, m in catalog.measures.items()}
        self.dependencies: dict[str, tuple[str, ...]] = {}
        self.metrics: dict[_Grain, list[str]] = defaultdict(list)

    def compile(self, requested: Sequence[str]) -> str:
        for name, metric in self.catalog.metrics.items():
            self.dependencies[name] = (
                tuple(dict.fromkeys(references(metric.formula)))
                if isinstance(metric, FormulaMetric)
                else ()
            )
        graph = {
            name: [ref for ref in refs if ref in self.dependencies]
            for name, refs in self.dependencies.items()
        }
        for name in TopologicalSorter(graph).static_order():
            grain = self._grain(name)
            self.grains[name] = grain
            self.metrics[grain].append(name)

        constants = [name for name in requested if self.grains[name] is None]
        public_sources = set(self.measures)
        for grain, names in self.metrics.items():
            if isinstance(grain, str) or (grain is None and not constants):
                continue
            public = _public_source(grain)
            if public in public_sources:
                raise MalloyExportError(
                    "malloy.name_collision",
                    names[0],
                    f"The generated source name {public!r} is already in use.",
                )
            public_sources.add(public)

        facts = self._fact_owners()
        for source, owner in facts.items():
            self._physical(source, owner, ())
            names = self.metrics.get(source, [])
            fields = [(name, self._measure(name, owner)) for name in self.measures[source]]
            fields.extend(self._formulas(names))
            _check_fields(
                owner, source, (*self._field_names(source), *(name for name, _ in fields))
            )
            self.blocks.append(_declaration(source, _base(source), _measure_lines(fields)))

        for grain, names in self.metrics.items():
            if isinstance(grain, _Roster):
                self._population(grain, names)
        if constants:
            fields = self._formulas(constants)
            _check_fields(constants[0], "constants", (name for name, _ in fields))
            self.blocks.append(
                _declaration(
                    "constants", f'{self.connection}.sql("SELECT 1")', _measure_lines(fields)
                )
            )
        return "\n\n".join(self.blocks) + "\n"

    def _grain(self, name: str) -> _Grain:
        metric = self.catalog.metrics[name]
        match metric:
            case MeanMetric(per_unit_mean=rule) | ConversionMetric(conversion=rule):
                return _Roster(rule.population)
            case FormulaMetric():
                grains = {self.grains[ref] for ref in self.dependencies[name]} - {None}
                if len(grains) > 1:
                    scopes = ", ".join(sorted(_public_source(grain) for grain in grains))
                    raise MalloyExportError(
                        "malloy.incompatible_grain",
                        name,
                        f"All operands must share a fact source or eligible population; found {scopes}.",
                    )
                return next(iter(grains), None)

    def _fact_owners(self) -> dict[str, str]:
        owners = {}
        for name, metric in self.catalog.metrics.items():
            if isinstance(metric, MeanMetric):
                owners.setdefault(self.catalog.measures[metric.per_unit_mean.measure].source, name)
            elif isinstance(metric, ConversionMetric):
                owners.setdefault(self.catalog.measures[metric.conversion.event_count].source, name)
            elif isinstance(grain := self.grains[name], str):
                owners.setdefault(grain, name)
        return owners

    def _field_names(self, source: str) -> Iterable[str]:
        return (*self.dimensions[source], *self.catalog.sources[source].joins)

    def _dimensions(self, source: str, owner: str) -> dict[str, str]:
        binding = self.catalog.sources[source]
        columns = {name: dimension.column for name, dimension in binding.dimensions.items()}
        other = dict(binding.entities)
        if binding.time:
            _add_dimension(other, binding.time, binding.time, owner, source)
        for name, column in other.items():
            _add_dimension(columns, name, column, owner, source)
        return columns

    def _physical(self, source: str, owner: str, path: tuple[str, ...]) -> None:
        if source in self.sources:
            return
        if source in path:
            cycle = " -> ".join((*path, source))
            raise MalloyExportError(
                "malloy.join_cycle", owner, f"Remove the cyclic source joins: {cycle}."
            )
        binding = self.catalog.sources[source]
        dimensions = self._dimensions(source, owner)
        self.dimensions[source] = dimensions
        _check_fields(owner, source, self._field_names(source))
        _check_fields(owner, "source names", (source,))
        keys_and_columns = {binding.row_key, *dimensions.values()}
        keys_and_columns.update(join.foreign_key for join in binding.joins.values())
        keys_and_columns.update(
            column
            for name in self.measures.get(source, ())
            if (column := self.catalog.measures[name].sum) is not None
        )
        relation = json.dumps(".".join(f'"{part}"' for part in binding.table.split(".")))
        projection = "\n".join(
            f"    {_column(col)} is {_name(col)}" for col in sorted(keys_and_columns)
        )
        base = f"({self.connection}.table({relation}) -> {{\n  select:\n{projection}\n}})"
        fields = [f"  primary_key: {_column(binding.row_key)}"]
        for name, column in dimensions.items():
            fields.append(f"  dimension: {_name(name)} is {_column(column)}")
        for alias, join in binding.joins.items():
            self._physical(join.source, owner, (*path, source))
            fields.append(
                f"  join_one: {_name(alias)} is {_base(join.source)} with {_column(join.foreign_key)}"
            )
        self.blocks.append(_declaration(f"{_INTERNAL}base_{source}", base, fields))
        self.sources.add(source)

    def _measure(self, name: str, owner: str) -> str:
        measure = self.catalog.measures[name]
        source = self.catalog.sources[measure.source]
        expression = (
            f"sum({_column(measure.sum)})"
            if measure.sum is not None
            else f"count({_column(source.row_key)})"
        )
        if measure.filter is None:
            return expression
        conditions = []
        for condition in self.catalog.filters[measure.filter].conditions:
            head, dot, tail = condition.dimension.partition(".")
            dimension = (
                self.catalog.sources[source.joins[head].source].dimensions[tail]
                if dot
                else source.dimensions[head]
            )
            conditions.append(_condition(condition, dimension.type, owner))
        return f"{expression} {{ where: " + " and ".join(conditions) + " }"

    def _formulas(self, names: Sequence[str]) -> list[tuple[str, str]]:
        required = set(names)
        pending = list(names)
        while pending:
            for ref in self.dependencies[pending.pop()]:
                if ref in self.dependencies and self.grains[ref] is None and ref not in required:
                    required.add(ref)
                    pending.append(ref)
        fields = []
        for name in self.grains:
            if name not in required or not isinstance(
                metric := self.catalog.metrics[name], FormulaMetric
            ):
                continue
            expression = _arithmetic(metric.formula, name)
            if self.grains[name] is None:
                # Keep a constant a measure even when its physical source has zero rows.
                expression = f"({expression} + 0 * count())"
            fields.append((name, expression))
        return fields

    def _population(self, population: _Roster, names: list[str]) -> None:
        binding = self.catalog.populations[population.name]
        self._physical(binding.source, names[0], ())
        entity = _name(binding.entity)
        values: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for name in names:
            match self.catalog.metrics[name]:
                case MeanMetric(per_unit_mean=rule):
                    values[self.catalog.measures[rule.measure].source].append(
                        (name, _name(rule.measure))
                    )
                case ConversionMetric(conversion=rule):
                    presence = f"pick 1 when {_name(rule.event_count)} > 0 else 0"
                    values[self.catalog.measures[rule.event_count].source].append((name, presence))
        joins = []
        fields = []
        for fact, metrics in values.items():
            alias = _name(f"{_INTERNAL}units_{fact}")
            output = []
            for name, expression in metrics:
                unit = _name(f"{_INTERNAL}unit_{name}")
                output.append(f"      {unit} is {expression}")
                fields.append((name, f"source.avg({alias}.{unit} ?? 0)"))
            per_unit = "\n".join(output)
            joins.append(
                f"  join_one: {alias} is ({_name(fact)} -> {{\n"
                f"    group_by: {entity}\n    aggregate:\n{per_unit}\n"
                f"  }} extend {{ primary_key: {entity} }}) with {entity}"
            )
        fields.extend(self._formulas(names))
        public = _public_source(population)
        _check_fields(
            names[0], public, (*self._field_names(binding.source), *(name for name, _ in fields))
        )
        self.blocks.append(
            _declaration(public, _base(binding.source), [*joins, *_measure_lines(fields)])
        )


def _base(source: str) -> str:
    return _name(f"{_INTERNAL}base_{source}")


def _public_source(grain: _Grain) -> str:
    if grain is None:
        return "constants"
    return grain if isinstance(grain, str) else f"population_{grain.name}"


def _add_dimension(
    columns: dict[str, str], name: str, column: str, metric: str, source: str
) -> None:
    if columns.setdefault(name, column) != column:
        raise MalloyExportError(
            "malloy.name_collision",
            metric,
            f"The dimension {name!r} refers to different columns in {source!r}.",
        )


def _check_fields(metric: str, source: str, names: Iterable[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if name.startswith(_INTERNAL) or name in seen:
            raise MalloyExportError(
                "malloy.name_collision",
                metric,
                f"The name {name!r} conflicts with another field in {source!r}.",
            )
        seen.add(name)


def _measure_lines(fields: Iterable[tuple[str, str]]) -> Iterable[str]:
    return (f"  measure: {_name(name)} is {expression}" for name, expression in fields)


def export_malloy(catalog: Catalog, *, metrics: Sequence[str], connection: str = "duckdb") -> str:
    """Return importable Malloy for the selected metric definitions.

    Query a fact source by its catalog name, or an eligible roster as
    ``population_<name>``. Literal-only metrics are available on ``constants``.
    Each source exposes only its own grouping dimensions. Before querying, the
    data owner must establish complete coverage and non-null, unique physical
    keys for the same snapshot; Malloy's ``primary_key`` does not enforce them.
    """
    if isinstance(metrics, str) or not metrics:
        raise MalloyExportError(
            "malloy.invalid_selection", None, "Pass a nonempty sequence of metric names."
        )
    for name in metrics:
        if name not in catalog.metrics:
            raise MalloyExportError(
                "malloy.unknown_metric", name, "Select a metric declared in the catalog."
            )
    if not _IDENTIFIER.fullmatch(connection):
        raise MalloyExportError(
            "malloy.invalid_connection", metrics[0], "Use a Malloy connection identifier."
        )
    return _Compiler(catalog.select(metrics), connection).compile(metrics)
