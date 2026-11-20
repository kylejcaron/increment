"""Bind selected portable metric definitions to native experiment facts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from increment._literals import PreferredDirection
from increment.errors import DefinitionError, RefusalSpec, refuse

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import metric_definitions as md

    from increment.semantics.models import FactSource

_INCLUDE = RefusalSpec(
    "definition.shared.include",
    DefinitionError,
    template="Shared metric declaration in {path}: {reason}; {route}",
)
_CATALOG = RefusalSpec(
    "definition.shared.catalog",
    DefinitionError,
    template="Shared metric document {path}:{line}:{column}: {reason}; {route}",
    keys=frozenset({"catalog_code"}),
)
_UNSUPPORTED = RefusalSpec(
    "definition.shared.metric_unsupported",
    DefinitionError,
    template="Shared metric {metric!r} cannot be used in {target}: {reason}; {route}",
)


def refuse_shared_metric(
    metric: str, reason: str, *, route: str, target: str = "Increment experiments"
) -> NoReturn:
    """Explain why the requested target cannot represent a shared metric."""
    refuse(_UNSUPPORTED, metric=metric, target=target, reason=reason, route=route)


def _refuse_uncaptured_shared_metric(metric: str) -> NoReturn:
    refuse_shared_metric(
        metric,
        "the shared inputs must be checked in the same snapshot as the readout",
        route="use captured coherence for this metric",
    )


def _refuse_shared_trigger(metric: str, experiment: str) -> NoReturn:
    refuse_shared_metric(
        metric,
        f"experiment {experiment!r} includes a separate triggered population",
        route="use a fixed assignment population with the declared eligible roster",
    )


def _unsupported(metric: str, reason: str) -> NoReturn:
    refuse_shared_metric(
        metric,
        reason,
        route=(
            "use a descriptive report in Terrine or Malloy, or bind a supported experiment metric"
        ),
    )


class _MetricOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str
    window_days: StrictInt | None = Field(ge=1)
    description: str = ""
    preferred_direction: PreferredDirection = "increase"
    margin: float | None = None
    margin_abs: float | None = None
    rollout_cost: float | None = None


class _Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    use: tuple[str, ...] = Field(min_length=1)
    sources: dict[str, str] = Field(default_factory=dict)
    measures: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, _MetricOptions] = Field(default_factory=dict)


def _manifest(raw: object, path: Path) -> _Manifest:
    try:
        manifest = _Manifest.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors(include_url=False)[0]
        field = ".".join(map(str, first["loc"])) or "metric_definitions"
        _refuse_include(path, f"{field}: {first['msg']}")
    if len(set(manifest.use)) != len(manifest.use):
        _refuse_include(path, "use lists a metric more than once")
    return manifest


def _refuse_include(path: Path, reason: str) -> NoReturn:
    refuse(
        _INCLUDE, path=str(path), reason=reason, route="repair the metric_definitions declaration"
    )


def shared_path(raw: object, path: Path, root: Path) -> Path:
    """Resolve the one included document under the definitions root."""
    from pathlib import Path

    include = Path(_manifest(raw, path).file)
    resolved = (path.parent / include).resolve()
    if include.is_absolute() or not resolved.is_relative_to(root.resolve()):
        _refuse_include(path, "file must be relative and stay within the definitions directory")
    if resolved == path.resolve():
        _refuse_include(path, "file includes its own native definitions")
    return resolved


def refuse_recursive_include(path: Path) -> NoReturn:
    """Refuse a shared include that declares another include."""
    _refuse_include(path, "a shared document cannot include native or shared definitions")


@dataclass(frozen=True, slots=True)
class SharedFact:
    """A selected event source mapped to a native event discriminator."""

    source: str
    fact: str
    row_key: str


@dataclass(frozen=True, slots=True)
class SharedRoster:
    """The declared eligible relation and unit key for one metric."""

    table: str
    key: str


@dataclass(frozen=True, slots=True)
class BoundSharedMetric:
    name: str
    unit: str
    facts: tuple[SharedFact, ...]
    roster: SharedRoster | None


@dataclass(frozen=True, slots=True)
class SharedMetrics:
    selection: str
    metrics: tuple[BoundSharedMetric, ...]


@dataclass(frozen=True, slots=True)
class _Operand:
    entity: str
    measure: dict[str, Any]
    fact: SharedFact


_FILTER_OPS = {
    "eq": "equals",
    "ne": "not_equals",
    "lt": "lt",
    "le": "lte",
    "gt": "gt",
    "ge": "gte",
    "in": "in",
}
_DTYPES = {"string": "string", "boolean": "bool", "integer": "int", "number": "float"}


def _filters(
    catalog: md.Catalog, measure: md.Measure, source: FactSource, metric: str
) -> list[dict]:
    if measure.filter is None:
        return []
    portable = catalog.sources[measure.source]
    filters = []
    for condition in catalog.filters[measure.filter].conditions:
        dimension = portable.dimensions.get(condition.dimension)
        if dimension is None:
            _unsupported(metric, f"filter dimension {condition.dimension!r} needs a portable join")
        dtype = _DTYPES.get(dimension.type)
        properties = [
            p.name for p in source.properties if p.column == dimension.column and p.dtype == dtype
        ]
        if len(properties) != 1:
            _unsupported(
                metric,
                f"filter dimension {condition.dimension!r} needs one matching native property",
            )
        values = condition.value if isinstance(condition.value, tuple) else (condition.value,)
        filters.append(
            {"property": properties[0], "op": _FILTER_OPS[condition.op], "values": values}
        )
    return filters


@dataclass(frozen=True, slots=True)
class _Binding:
    catalog: md.Catalog
    manifest: _Manifest
    sources: dict[str, FactSource]

    def operand(self, metric: str, entity: str, measure_name: str, window: int | None) -> _Operand:
        measure = self.catalog.measures[measure_name]
        portable = self.catalog.sources[measure.source]
        source = self.sources.get(self.manifest.sources.get(measure.source, ""))
        if source is None:
            _unsupported(metric, f"source {measure.source!r} needs a native fact source binding")
        fact_name = self.manifest.measures.get(measure_name)
        fact = next((fact for fact in source.facts if fact.name == fact_name), None)
        if fact is None or any(s.name == fact_name and s != source for s in self.sources.values()):
            _unsupported(metric, f"measure {measure_name!r} needs a fact in source {source.name!r}")
        unit = portable.entities.get(entity)
        if unit is None or unit not in source.entities:
            _unsupported(
                metric, f"source {measure.source!r} must expose entity {entity!r} as a native unit"
            )
        if portable.time != source.timestamp_column:
            _unsupported(metric, f"source {measure.source!r} must expose the declared event time")
        if measure.sum is not None and (measure.sum != fact.column or fact.missing != "error"):
            _unsupported(
                metric,
                f"measure {measure_name!r} needs the declared sum column with missing: error",
            )
        if measure.count_rows and fact.missing == "skip":
            _unsupported(
                metric, f"count measure {measure_name!r} cannot skip rows with null values"
            )
        return _Operand(
            unit,
            {
                "fact": fact.name,
                "aggregation": "sum" if measure.sum is not None else "count",
                "window_days": window,
                "filters": _filters(self.catalog, measure, source, metric),
            },
            SharedFact(source.name, fact.name, portable.row_key),
        )

    def metric(self, name: str) -> tuple[dict[str, Any], BoundSharedMetric]:
        import metric_definitions as md

        declaration = self.catalog.metrics[name]
        if isinstance(declaration, md.FormulaMetric):
            operands = _ratio_leaves(declaration.formula, self.catalog, name)
        options = self.manifest.metrics.get(name)
        if options is None:
            _unsupported(name, "declare native metric options (entity and window_days)")
        native = {
            "name": name,
            **options.model_dump(exclude={"entity", "window_days"}, exclude_unset=True),
        }
        if isinstance(declaration, md.FormulaMetric):
            left, right = (
                self.operand(name, options.entity, leaf, options.window_days) for leaf in operands
            )
            if left.entity != right.entity:
                _unsupported(name, "ratio operands must bind the same native unit column")
            native.update(
                type="ratio", entity=left.entity, numerator=left.measure, denominator=right.measure
            )
            return native, BoundSharedMetric(
                name, left.entity, tuple(dict.fromkeys((left.fact, right.fact))), None
            )

        if isinstance(declaration, md.MeanMetric):
            rule = declaration.per_unit_mean
            leaf, population, kind = rule.measure, rule.population, "mean"
        else:
            rule = declaration.conversion
            leaf, population, kind = rule.event_count, rule.population, "conversion"
        operand = self.operand(name, options.entity, leaf, options.window_days)
        roster = self.catalog.populations[population]
        if roster.entity != options.entity:
            _unsupported(
                name, f"eligible population {population!r} must use entity {options.entity!r}"
            )
        source = self.catalog.sources[roster.source]
        native.update(type=kind, entity=operand.entity, **operand.measure)
        if kind == "conversion":
            del native["aggregation"]
        return native, BoundSharedMetric(
            name, operand.entity, (operand.fact,), SharedRoster(source.table, source.row_key)
        )


def _ratio_leaves(formula: md.Formula, catalog: md.Catalog, name: str) -> tuple[str, str]:
    import metric_definitions as md

    match formula:
        case md.Binary(operator="/", left=md.Reference(name=left), right=md.Reference(name=right)):
            if left in catalog.measures and right in catalog.measures:
                return left, right
    _unsupported(
        name, "experiment formula inference requires a ratio of two named additive measures"
    )


def bind_shared(
    raw_manifest: object, path: Path, root: Path, fact_sources: Sequence[FactSource]
) -> tuple[list[dict[str, Any]], SharedMetrics]:
    """Select, then translate only the requested experiment metrics."""
    manifest = _manifest(raw_manifest, path)
    included = shared_path(raw_manifest, path, root)
    try:
        import metric_definitions as md
    except ModuleNotFoundError as exc:
        if exc.name != "metric_definitions":
            raise
        _refuse_include(
            path, "portable definitions require the increment[metric-definitions] extra"
        )
    try:
        catalog = md.load(included)
    except md.CatalogError as exc:
        refuse(
            _CATALOG,
            path=str(exc.path),
            line=exc.line,
            column=exc.column,
            catalog_code=exc.code,
            reason=exc.detail,
            route="repair the shared metric document",
        )
    except OSError as exc:
        _refuse_include(path, f"cannot read {manifest.file!r}: {exc.strerror}")
    for name in manifest.use:
        if name not in catalog.metrics:
            _unsupported(name, "the shared document does not declare that metric")
    selected = catalog.select(manifest.use)
    sources = {source.name: source for source in fact_sources}
    binding = _Binding(selected, manifest, sources)
    metrics, checks = zip(*(binding.metric(name) for name in sorted(manifest.use)), strict=True)
    selection = json.dumps(selected.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return list(metrics), SharedMetrics(selection, checks)
