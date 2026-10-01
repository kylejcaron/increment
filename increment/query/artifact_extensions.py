"""Typed unit-day artifact evidence extensions.

This module is deliberately a definitions-side boundary.  It turns an exact,
caller-selected extension request into a role-bound relation written through
an explicit publication handle, and verifies the same relation through one
immutable snapshot handle when adopted.  Extension identities and wire
schemas come from the AYM7-1 declarations; this module does not define a
second schema vocabulary.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import partial
from typing import Any, cast

import ibis
import narwhals as nw

from increment._source_operations import (
    BreakoutSourceOperation,
    DaySourceOperation,
    SitewideEvidenceOperation,
    TriggeredCountsOperation,
    TriggeredPopulationOperation,
)
from increment.errors import refuse
from increment.query.artifact_contract import (
    REFUSALS as _ARTIFACT_REFUSALS,
)
from increment.query.artifact_contract import (
    ArtifactPublication,
    ArtifactSnapshot,
    _aggregate_sum_within_extrema,
    unit_day_artifact_extension_catalog,
)
from increment.query.artifact_digest import (
    canonical_json,
    content_sha256,
    extension_source_provenance_sha256,
    schema_sha256,
)
from increment.query.schemas import (
    ARTIFACT_RELATION_PRIMARY_KEYS,
    ARTIFACT_RELATION_SCHEMAS,
)
from increment.semantics.artifact import (
    ArtifactExtensionCatalogEntry,
    ArtifactExtensionRef,
    ArtifactExtensionRequest,
    ArtifactRelationRef,
    AssignmentCountsExtension,
    BreakoutDimensionExtension,
    ClusterIdentityExtension,
    CupedPreperiodExtension,
    EncouragementUptakeExtension,
    FactorDimensionExtension,
    Freshness,
    SiteVolumeExtension,
    TriggerPopulationExtension,
    UnitCovariateExtension,
    UnitCovariateLevelExtension,
)
from increment.semantics.models import Breakout


def _reject(code: str, message: str, **context: object) -> None:
    refuse(_ARTIFACT_REFUSALS[code], message=f"{code}: {message}", **context)


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        if "rows" in value:
            return _rows(value["rows"])
        return [value]
    if hasattr(value, "to_pylist"):
        return _rows(value.to_pylist())
    if hasattr(value, "to_dicts"):
        return _rows(value.to_dicts())
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _rows(value.to_dict("records"))
    if hasattr(value, "execute") and callable(value.execute):
        return _rows(value.execute())
    try:
        result = list(value)
    except TypeError as exc:
        _reject(
            "artifact.extension.invalid",
            "extension evidence is not row-like",
            value_type=type(value).__name__,
        )
        raise AssertionError from exc
    if not all(isinstance(row, Mapping) for row in result):
        _reject("artifact.extension.invalid", "extension evidence rows must be mappings")
    return result


def _require_operation(source: Any, kind: str) -> None:
    handler = _handler(kind)
    if handler.operation not in getattr(source, "operations", frozenset()) or not isinstance(
        source, handler.protocol
    ):
        _reject(
            "artifact.evidence.unavailable",
            f"artifact extension {kind!r} requires source operation {handler.operation!r}",
            operation=handler.operation,
            extension_kind=kind,
        )


def _resolve_metric(source: Any, metric_name: str) -> Any:
    candidates = getattr(source, "_metrics", getattr(source, "metrics", ()))
    for metric in candidates:
        if getattr(metric, "name", None) == metric_name:
            return metric
    return metric_name


class _EvidenceRows(list[Mapping[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], audit: tuple[int, int] = (0, 0)) -> None:
        super().__init__(rows)
        self.audit = audit


@dataclass(frozen=True, slots=True)
class _ExtensionHandler:
    role: str
    operation: str
    protocol: type[Any]
    source_rows: Callable[..., Any]
    model: Callable[..., ArtifactExtensionRef]
    identity_matches: Callable[..., bool]
    request_from_extension: Callable[[Any], dict[str, Any]]
    validate_row: Callable[[Mapping[str, Any]], None]
    prepare_evidence: Callable[[Any], tuple[Any, tuple[int, int]]]
    validate_audit: Callable[[Any], None]
    validate_result: Callable[[Any, Sequence[Mapping[str, Any]]], None]
    bind_definition: Callable[[Any, Mapping[str, Any]], None]
    validate_definition: Callable[[Mapping[str, Any], Any], None]
    validate_coverage: Callable[[Sequence[Mapping[str, Any]], Any], None]
    validate_exposure: Callable[[Sequence[Mapping[str, Any]], Iterable[tuple[Any, ...]]], None]
    read: Callable[..., list[Mapping[str, Any]]]
    read_unit_covariate: Callable[..., nw.Series[Any]] | None = None


def _decode_unit_covariate(
    extension: UnitCovariateExtension | UnitCovariateLevelExtension,
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    unit_ids: Sequence[str],
    implementation: Any,
) -> nw.Series[Any]:
    categorical = extension.kind == "unit_covariate_level"
    field = "level" if categorical else "value"
    values = {str(row["unit_id"]): row[field] for row in rows}
    return nw.new_series(
        name,
        [values.get(unit) for unit in unit_ids],
        nw.String() if categorical else nw.Float64(),
        backend=implementation,
    )


def _handler(kind: str) -> _ExtensionHandler:
    return _HANDLERS[kind]


def _validate_exact_exposure_coverage(
    result: Sequence[Mapping[str, Any]], exposure_keys: Iterable[tuple[Any, ...]]
) -> None:
    actual = {
        tuple(row[field] for field in ARTIFACT_RELATION_PRIMARY_KEYS["exposures"]) for row in result
    }
    if actual != set(exposure_keys):
        _reject("artifact.extension.invalid", "extension/exposure join-key coverage mismatch")


def _bind_noop(_request: Any, _definition: Mapping[str, Any]) -> None:
    return


@dataclass(frozen=True, slots=True)
class ExtensionKindSpec:
    """Declarative shape for an extension kind whose bind/validate-
    definition/identity/request/model bodies are pure field echoes between
    the extension model and the catalog request. `assignment_counts` and
    `site_volume` keep bespoke `_ExtensionHandler` entries instead (real
    cross-checks against `definition`, not a field list)."""

    kind: str
    role: str
    operation: str
    protocol: type[Any]
    model_cls: type[Any]
    # (extension model attribute, request/definition attribute) pairs echoed
    # across identity/request/model/validate_definition; breakout/factor rename the field.
    fields: tuple[tuple[str, str], ...]
    row_validator: Callable[[Mapping[str, Any]], None]
    source_rows: Callable[..., Any]
    bind_definition: Callable[[Any, Mapping[str, Any]], None] = _bind_noop
    validate_definition: Callable[[Mapping[str, Any], Any], None] | None = None
    identity_matches: Callable[..., bool] | None = None
    request_from_extension: Callable[[Any], dict[str, Any]] | None = None
    model: Callable[..., ArtifactExtensionRef] | None = None
    validate_exposure: Callable[..., None] = _validate_exact_exposure_coverage


def _generic_identity_matches(
    spec: ExtensionKindSpec,
    extension: Any,
    request: Mapping[str, Any],
    _definition: Mapping[str, Any] | None,
) -> bool:
    return all(
        getattr(extension, ext_attr) == request.get(req_attr) for ext_attr, req_attr in spec.fields
    )


def _generic_request_from_extension(spec: ExtensionKindSpec, extension: Any) -> dict[str, Any]:
    return {
        "kind": spec.kind,
        **{req_attr: getattr(extension, ext_attr) for ext_attr, req_attr in spec.fields},
    }


def _generic_validate_definition(
    spec: ExtensionKindSpec, value: Mapping[str, Any], request: Any
) -> None:
    for ext_attr, req_attr in spec.fields:
        if value.get(ext_attr) != getattr(request, req_attr):
            _reject("artifact.extension.invalid", "definition does not match selected request")
            return


def _generic_model(
    spec: ExtensionKindSpec,
    *,
    relation: ArtifactRelationRef,
    entry: ArtifactExtensionCatalogEntry,
    **_: Any,
) -> ArtifactExtensionRef:
    request: Any = entry.request
    base = _model_base(relation, entry)
    base.update({ext_attr: getattr(request, req_attr) for ext_attr, req_attr in spec.fields})
    return spec.model_cls.model_validate(base)


def _dimension_source_rows(source: Any, request: Any, _experiment_id: str | None) -> Any:
    return _source_dimension_rows(source, request)


def _day_source_rows(kind: str) -> Callable[[Any, Any, str | None], Any]:
    def rows(source: Any, request: Any, _experiment_id: str | None) -> Any:
        return _source_day_rows(source, request, kind)

    return rows


def _build_handler(spec: ExtensionKindSpec) -> _ExtensionHandler:
    return _ExtensionHandler(
        role=spec.role,
        operation=spec.operation,
        protocol=spec.protocol,
        source_rows=spec.source_rows,
        model=spec.model or (lambda **kw: _generic_model(spec, **kw)),
        identity_matches=spec.identity_matches or (lambda *a: _generic_identity_matches(spec, *a)),
        request_from_extension=spec.request_from_extension
        or (lambda ext: _generic_request_from_extension(spec, ext)),
        validate_row=spec.row_validator,
        bind_definition=spec.bind_definition,
        validate_definition=spec.validate_definition
        or (lambda v, r: _generic_validate_definition(spec, v, r)),
        prepare_evidence=_prepare_plain_evidence,
        validate_audit=_validate_noop_audit,
        validate_result=_validate_noop_result,
        validate_coverage=_validate_noop_coverage,
        validate_exposure=spec.validate_exposure,
        read=validate_extension,
        read_unit_covariate=(
            _decode_unit_covariate
            if spec.kind in {"unit_covariate", "unit_covariate_level"}
            else None
        ),
    )


def _source_dimension_rows(source: Any, request: Any) -> Any:
    metric = _resolve_metric(source, getattr(request, "property_name", ""))
    breakout = Breakout(property=request.property_name, source=request.source_name)
    result = source.breakout_source(breakout, metrics=(metric,))
    if hasattr(result, "moments") or hasattr(result, "breakout_moments"):
        _reject(
            "artifact.evidence.unavailable",
            "breakout source does not expose unit-grain dimension rows",
        )
    return result


def _source_assignment_rows(
    source: Any, request: Any, experiment_id: str | None
) -> Mapping[str, Any]:
    populations = tuple(request.populations)
    reserved_labels = ("(mixed assignment)", "(unassigned)")
    assigned_randomization = dict(source.assignment_counts(population="assigned"))
    if hasattr(source, "unit_counts"):
        assigned_units = dict(source.unit_counts())
    elif getattr(getattr(source, "_experiment", None), "cluster", None) is None:
        assigned_units = dict(assigned_randomization)
    else:
        _reject(
            "artifact.evidence.unavailable",
            "clustered assigned source lacks unit-grain counts",
        )
    mixed_units = max(
        int(assigned_randomization.pop(reserved_labels[0], 0)),
        int(assigned_units.pop(reserved_labels[0], 0)),
    )
    unassigned_units = max(
        int(assigned_randomization.pop(reserved_labels[1], 0)),
        int(assigned_units.pop(reserved_labels[1], 0)),
    )
    result: list[Mapping[str, Any]] = []
    if "assigned" in populations:
        result.extend(
            {
                "experiment_id": experiment_id or "",
                "population": "assigned",
                "group_id": group_id,
                "n_units": assigned_units.get(group_id, count),
                "n_randomization_units": count,
            }
            for group_id, count in assigned_randomization.items()
        )
    if "triggered" in populations:
        _grain, triggered, triggered_units = source.triggered_counts()
        triggered = {
            group_id: count
            for group_id, count in dict(triggered).items()
            if group_id not in reserved_labels
        }
        triggered_units = {
            group_id: count
            for group_id, count in dict(triggered_units).items()
            if group_id not in reserved_labels
        }
        result.extend(
            {
                "experiment_id": experiment_id or "",
                "population": "triggered",
                "group_id": group_id,
                "n_units": triggered_units.get(group_id, count),
                "n_randomization_units": count,
            }
            for group_id, count in triggered.items()
        )
        # Arms with no triggered units are real zero cells, not gaps;
        # emit them so population coverage holds for sparse triggering.
        result.extend(
            {
                "experiment_id": experiment_id or "",
                "population": "triggered",
                "group_id": group_id,
                "n_units": 0,
                "n_randomization_units": 0,
            }
            for group_id in assigned_randomization
            if group_id not in triggered
        )
    # Reserved accounting labels never become relation rows; carry them
    # as descriptor audit metadata alongside the group rows.
    return {"rows": result, "audit": (mixed_units, unassigned_units)}


def _source_trigger_rows(source: Any) -> Any:
    result = source.triggered_source()
    if hasattr(result, "moments") or (
        not isinstance(result, (Mapping, Sequence)) and not hasattr(result, "to_pylist")
    ):
        _reject("artifact.evidence.unavailable", "triggered source does not expose trigger rows")
    return result


def _source_site_volume_rows(source: Any, request: Any) -> Any:
    metric = _resolve_metric(source, request.metric_names[0])
    evidence = source.sitewide_evidence(metric, include_ratio=True)
    if hasattr(evidence, "site_total"):
        _reject(
            "artifact.evidence.unavailable",
            "sitewide evidence is whole-window; site-volume needs per-measure/date rows",
        )
    _reject("artifact.evidence.unavailable", "sitewide operation did not return site-volume rows")
    return evidence


def _source_day_rows(source: Any, request: Any, kind: str) -> Any:
    metric_name = getattr(request, "metric_name", getattr(request, "uptake_name", ""))
    metric = _resolve_metric(source, metric_name)
    result = source.day_source(metrics=(metric,))
    if hasattr(result, "moments"):
        _reject(
            "artifact.evidence.unavailable",
            f"{kind} day source does not expose required unit-grain rows",
        )
    return result


def _source_trigger_population_rows(source: Any, request: Any, _experiment_id: str | None) -> Any:
    return _source_trigger_rows(source)


def _source_assignment_count_rows(source: Any, request: Any, experiment_id: str | None) -> Any:
    result = _source_assignment_rows(source, request, experiment_id)
    return _EvidenceRows(_rows(result["rows"]), cast("tuple[int, int]", result["audit"]))


def _source_site_volume_extension_rows(
    source: Any, request: Any, _experiment_id: str | None
) -> Any:
    return _source_site_volume_rows(source, request)


def _unit_covariate_source_rows(kind: str) -> Callable[[Any, Any, str | None], Any]:
    def rows(source: Any, request: Any, _experiment_id: str | None) -> Any:
        _reject(
            "artifact.evidence.unavailable",
            "unit covariate rows are produced by the unit-day artifact publisher",
            extension_kind=kind,
            property_name=request.property_name,
        )

    return rows


def _source_rows(
    source: Any,
    request: Any,
    kind: str,
    *,
    experiment_id: str | None = None,
    definition: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]] | Mapping[str, Any]:
    del definition
    result = _handler(kind).source_rows(source, request, experiment_id)
    if isinstance(result, _EvidenceRows):
        return result
    return _rows(result)


def _wire_table(value: Any, kind: str) -> tuple[Any, list[Mapping[str, Any]]]:
    if hasattr(value, "schema") and hasattr(value, "execute"):
        try:
            return value, _rows(value.to_pyarrow().to_pylist())
        except (AttributeError, TypeError):
            return value, _rows(value.execute())
    rows = _rows(value)
    type_names = {
        "STRING": "string",
        "BOOLEAN": "boolean",
        "INT64": "int64",
        "FLOAT64": "float64",
        "DATE": "date",
        "TIMESTAMP_UTC_US": "timestamp('UTC')",
    }
    schema = ibis.schema(
        {field: type_names[tag] for field, tag, _nullable in ARTIFACT_RELATION_SCHEMAS[kind]}
    )
    if rows:
        return ibis.memtable(rows, schema=schema), rows
    # ibis<=10.x drops every declared column when a memtable is built from
    # an empty Python list -- `schema` is accepted at expression build time
    # but ignored once materialised (ibis-project/ibis#10940, fixed in
    # ibis 11). An already-typed empty pyarrow table keeps the columns.
    return ibis.memtable(schema.to_pyarrow().empty_table(), schema=schema), rows


def _catalog_entry(context: Any, request: Any) -> ArtifactExtensionCatalogEntry:
    if context is None:
        _reject("artifact.extension.missing", "extension context is required")
    entries = unit_day_artifact_extension_catalog(context)
    try:
        wanted = canonical_json(request if isinstance(request, Mapping) else _dump(request))
    except (TypeError, ValueError) as exc:
        _reject(
            "artifact.extension.invalid",
            "extension request is not a closed request",
            error=str(exc),
        )
        raise AssertionError from exc
    matches = [entry for entry in entries if canonical_json(_dump(entry.request)) == wanted]
    if len(matches) != 1:
        code = "artifact.extension.missing" if not matches else "artifact.extension.invalid"
        _reject(
            code,
            "extension request is not selected exactly once",
            request=_dump(wanted),
            matches=len(matches),
        )
    return matches[0]


def _request_value(request: Any, name: str) -> Any:
    return request.get(name) if isinstance(request, Mapping) else getattr(request, name, None)


def _bind_dimension(request: Any, definition: Mapping[str, Any]) -> None:
    prop = definition.get("property", definition)
    prop_name = (
        prop.get("property", prop.get("property_name")) if isinstance(prop, Mapping) else None
    )
    if prop_name != _request_value(request, "property_name"):
        _reject("artifact.extension.invalid", "definition property does not match request")


def _bind_assignment(request: Any, definition: Mapping[str, Any]) -> None:
    if tuple(definition.get("populations", ())) != tuple(
        _request_value(request, "populations") or ()
    ):
        _reject("artifact.extension.invalid", "definition populations do not match request")


def _bind_site_volume(request: Any, definition: Mapping[str, Any]) -> None:
    if tuple(definition.get("metric_names", ())) != tuple(
        _request_value(request, "metric_names") or ()
    ):
        _reject("artifact.extension.invalid", "definition metrics do not match request")


def _bind_cuped(_request: Any, definition: Mapping[str, Any]) -> None:
    if "metric_name" not in definition:
        _reject("artifact.extension.invalid", "CUPED definition lacks metric_name")


def _bind_request_definition(kind: str, request: Any, definition: Mapping[str, Any]) -> None:
    if definition.get("kind") != kind:
        _reject("artifact.extension.invalid", "definition kind does not match request")
    for field in ("cluster_name", "metric_name", "trigger_name", "uptake_name"):
        expected = _request_value(request, field)
        if expected is not None and definition.get(field) != expected:
            _reject("artifact.extension.invalid", f"definition {field} does not match request")
    _handler(kind).bind_definition(request, definition)


def _canonical_assignment_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: tuple(
            str(row[field]) for field in ARTIFACT_RELATION_PRIMARY_KEYS["assignment_counts"]
        ),
    )


def _split_assignment_audit(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int, int]:
    """Split reserved accounting labels out of assignment rows into counts."""
    reserved = {"(mixed assignment)", "(unassigned)"}
    clean = [row for row in rows if row.get("group_id") not in reserved]
    mixed = sum(int(row["n_units"]) for row in rows if row.get("group_id") == "(mixed assignment)")
    unassigned = sum(int(row["n_units"]) for row in rows if row.get("group_id") == "(unassigned)")
    return clean, mixed, unassigned


def _validate_definition_dimension(value: Mapping[str, Any], request: Any) -> None:
    prop = value.get("property", value)
    prop_name = (
        prop.get("property", prop.get("property_name"))
        if isinstance(prop, Mapping)
        else value.get("property_name")
    )
    if prop_name != request.property_name:
        _reject("artifact.extension.invalid", "definition property does not match selected request")


def _validate_definition_assignment(value: Mapping[str, Any], request: Any) -> None:
    if tuple(value.get("populations", ())) != tuple(request.populations):
        _reject(
            "artifact.extension.invalid", "definition populations do not match selected request"
        )


def _validate_definition_site_volume(value: Mapping[str, Any], request: Any) -> None:
    if tuple(value.get("metric_names", ())) != tuple(request.metric_names):
        _reject("artifact.extension.invalid", "definition metrics do not match selected request")


def _definition(
    entry: ArtifactExtensionCatalogEntry, definition: Mapping[str, Any] | None
) -> dict[str, Any]:
    if definition is not None:
        value = dict(definition)
    else:
        import json

        value = json.loads(entry.canonical_definition_json)
    if definition is not None:
        import json

        supplied = dict(value)
        trusted = json.loads(entry.canonical_definition_json)
        for field in ("mixed_unit_count", "unassigned_unit_count", "assignment_fingerprint_sha256"):
            supplied.pop(field, None)
            trusted.pop(field, None)
        if canonical_json(supplied) != canonical_json(trusted):
            _reject(
                "artifact.extension.invalid", "supplied definition differs from trusted catalog"
            )
    req = cast(Any, entry.request)
    handler = _handler(entry.request.kind)
    handler.validate_definition(value, req)
    _bind_request_definition(entry.request.kind, entry.request, value)
    return value


def _source_recipe(
    entry: ArtifactExtensionCatalogEntry, recipe: Mapping[str, Any] | None
) -> dict[str, Any]:
    if recipe is not None:
        value = dict(recipe)
    else:
        import json

        value = json.loads(entry.canonical_source_recipe_json)
    if (
        extension_source_provenance_sha256(entry.request.kind, canonical_json(value))
        != entry.source_provenance_sha256
    ):
        _reject(
            "artifact.extension.invalid", "extension source recipe does not match trusted catalog"
        )
    return value


def _validate_assignment_row(row: Mapping[str, Any]) -> None:
    if row.get("group_id") in ("(mixed assignment)", "(unassigned)"):
        _reject(
            "artifact.extension.invalid",
            "assignment accounting labels are descriptor metadata, not rows",
        )
    if row["population"] not in {"assigned", "triggered"}:
        _reject("artifact.extension.invalid", "assignment population is outside closed domain")
    for field in ("n_units", "n_randomization_units"):
        value = row[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            _reject("artifact.extension.invalid", f"assignment {field} is invalid")


def _validate_dimension_row(row: Mapping[str, Any], value_field: str) -> None:
    if not isinstance(row["value_is_missing"], bool) or not isinstance(row[value_field], str):
        _reject("artifact.extension.invalid", "dimension/factor domain is invalid")


def _validate_stats_row(
    row: Mapping[str, Any], minimum: int, *, exact_integer_limit: bool = False
) -> None:
    n = row["n_events"]
    if not isinstance(n, int) or isinstance(n, bool) or n < minimum:
        _reject("artifact.extension.invalid", "event count domain is invalid")
    if exact_integer_limit and n > 2**53:
        _reject("artifact.extension.invalid", "site-volume n_events exceeds exact integer range")
    for name in ("sum_value", "min_value", "max_value"):
        if not isinstance(row[name], (int, float)) or isinstance(row[name], bool):
            _reject("artifact.extension.invalid", "aggregate value domain is invalid")
    if n and not _aggregate_sum_within_extrema(
        row["sum_value"], n, row["min_value"], row["max_value"]
    ):
        _reject("artifact.extension.invalid", "aggregate bounds are invalid")


def _validate_cluster_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row["cluster_id"], str):
        _reject("artifact.extension.invalid", "cluster identity domain is invalid")


def _validate_cuped_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row["x"], (int, float)) or isinstance(row["x"], bool):
        _reject("artifact.extension.invalid", "CUPED value domain is invalid")


def _validate_trigger_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row["first_trigger_ts"], datetime):
        _reject("artifact.extension.invalid", "trigger timestamp domain is invalid")


def _validate_encouragement_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row["uptake"], bool):
        _reject("artifact.extension.invalid", "uptake domain is invalid")
    if row["uptake"] != (row.get("first_uptake_ts") is not None):
        _reject("artifact.extension.invalid", "uptake flag and first timestamp disagree")


def _validate_row_domain(kind: str, row: Mapping[str, Any]) -> None:
    _handler(kind).validate_row(row)


def _validate_rows(
    kind: str, rows: Sequence[Mapping[str, Any]], *, experiment_id: str | None = None
) -> None:
    fields = tuple(name for name, _tag, _nullable in ARTIFACT_RELATION_SCHEMAS[kind])
    expected = set(fields)
    primary_key = ARTIFACT_RELATION_PRIMARY_KEYS[kind]
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        names = set(row)
        if names != expected:
            _reject(
                "artifact.extension.invalid",
                "extension relation schema mismatch",
                kind=kind,
                fields=sorted(names),
            )
        for name, tag, nullable in ARTIFACT_RELATION_SCHEMAS[kind]:
            value = row[name]
            if value is None:
                if nullable:
                    continue
                _reject(
                    "artifact.extension.invalid",
                    "extension relation contains NULL in a non-nullable field",
                    kind=kind,
                    field=name,
                )
            valid = (
                (tag == "STRING" and isinstance(value, str))
                or (tag == "BOOLEAN" and isinstance(value, bool))
                or (tag == "INT64" and isinstance(value, int) and not isinstance(value, bool))
                or (
                    tag == "FLOAT64"
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                )
                or (tag == "DATE" and isinstance(value, date) and not isinstance(value, datetime))
                or (
                    tag == "TIMESTAMP_UTC_US"
                    and isinstance(value, datetime)
                    and value.tzinfo is not None
                    and value.utcoffset() is not None
                )
            )
            if not valid:
                _reject(
                    "artifact.extension.invalid",
                    "extension relation physical domain mismatch",
                    kind=kind,
                    field=name,
                )
        if experiment_id is not None and row["experiment_id"] != experiment_id:
            _reject(
                "artifact.extension.invalid",
                "extension foreign key experiment_id mismatch",
                kind=kind,
            )
        key = tuple(row[name] for name in primary_key)
        if key in seen:
            _reject(
                "artifact.extension.invalid",
                "extension relation contains duplicate primary key",
                kind=kind,
            )
        seen.add(key)
        _validate_row_domain(kind, row)


def _model_base(
    relation: ArtifactRelationRef, entry: ArtifactExtensionCatalogEntry
) -> dict[str, Any]:
    return {
        "relation": relation,
        "definition_sha256": entry.definition_sha256,
        "source_provenance_sha256": entry.source_provenance_sha256,
    }


def _model_cuped(
    *,
    relation: ArtifactRelationRef,
    entry: ArtifactExtensionCatalogEntry,
    definition: Mapping[str, Any],
    **_: Any,
) -> ArtifactExtensionRef:
    fields = ("metric_name", "measure_key", "aggregation", "window_start_days", "window_end_days")
    for field in fields:
        if field not in definition:
            _reject("artifact.extension.invalid", f"CUPED definition lacks {field}")
    base = _model_base(relation, entry)
    base.update({field: definition[field] for field in fields})
    return CupedPreperiodExtension.model_validate(base)


def _model_assignment(
    *,
    relation: ArtifactRelationRef,
    entry: ArtifactExtensionCatalogEntry,
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_counts: tuple[int, int] | None = None,
    **_: Any,
) -> ArtifactExtensionRef:
    request: Any = entry.request
    mixed, unassigned = audit_counts if audit_counts is not None else (0, 0)
    fingerprint_payload = {
        "rows": _canonical_assignment_rows(rows),
        "mixed_unit_count": mixed,
        "unassigned_unit_count": unassigned,
    }
    fingerprint = hashlib.sha256(canonical_json(fingerprint_payload).encode()).hexdigest()
    if definition.get("assignment_fingerprint_sha256") not in (None, fingerprint):
        _reject(
            "artifact.extension.invalid", "assignment fingerprint does not match published rows"
        )
    base = _model_base(relation, entry)
    base.update(
        populations=request.populations,
        mixed_unit_count=mixed,
        unassigned_unit_count=unassigned,
        assignment_fingerprint_sha256=fingerprint,
    )
    return AssignmentCountsExtension.model_validate(base)


def _model_encouragement(
    *,
    relation: ArtifactRelationRef,
    entry: ArtifactExtensionCatalogEntry,
    definition: Mapping[str, Any],
    **_: Any,
) -> ArtifactExtensionRef:
    request: Any = entry.request
    base = _model_base(relation, entry)
    base.update(
        extension_version=2,
        uptake_name=request.uptake_name,
        window_days=definition["window_days"],
        one_sided=definition["one_sided"],
    )
    return EncouragementUptakeExtension.model_validate(base)


def _model_site_volume(
    *,
    relation: ArtifactRelationRef,
    entry: ArtifactExtensionCatalogEntry,
    definition: Mapping[str, Any],
    **_: Any,
) -> ArtifactExtensionRef:
    request: Any = entry.request
    base = _model_base(relation, entry)
    base.update(
        measure_keys=tuple(definition.get("measure_keys", request.metric_names)),
        first_ds=definition["first_ds"],
        last_ds=definition["last_ds"],
        freshness=Freshness.model_validate(definition["freshness"]),
    )
    return SiteVolumeExtension.model_validate(base)


def _unit_covariate_model(model_cls: type[Any]) -> Callable[..., ArtifactExtensionRef]:
    def model(
        *,
        relation: ArtifactRelationRef,
        entry: ArtifactExtensionCatalogEntry,
        definition: Mapping[str, Any],
        **_: Any,
    ) -> ArtifactExtensionRef:
        request: Any = entry.request
        prop = definition.get("property_definition")
        if not isinstance(prop, Mapping) or "as_of" not in prop:
            _reject("artifact.extension.invalid", "covariate definition lacks property as_of")
        base = _model_base(relation, entry)
        base.update(
            covariate_name=request.property_name,
            source_name=request.source_name,
            as_of=cast("Mapping[str, Any]", prop)["as_of"],
        )
        return model_cls.model_validate(base)

    return model


def _extension_model(
    kind: str,
    *,
    relation: ArtifactRelationRef,
    entry: ArtifactExtensionCatalogEntry,
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    audit_counts: tuple[int, int] | None = None,
) -> ArtifactExtensionRef:
    return _handler(kind).model(
        relation=relation,
        entry=entry,
        definition=definition,
        rows=rows,
        audit_counts=audit_counts,
    )


def _prepare_plain_evidence(raw_evidence: Any) -> tuple[Any, tuple[int, int]]:
    return raw_evidence, (0, 0)


def _prepare_assignment_evidence(raw_evidence: Any) -> tuple[Any, tuple[int, int]]:
    audit_counts: tuple[int, int] = (0, 0)
    if hasattr(raw_evidence, "audit"):
        audit_counts = cast("tuple[int, int]", raw_evidence.audit)
    if isinstance(raw_evidence, Mapping) and "audit" in raw_evidence:
        payload = cast("Mapping[str, Any]", raw_evidence)
        audit_counts = cast("tuple[int, int]", payload["audit"])
        raw_evidence = cast("Sequence[Mapping[str, Any]]", payload["rows"])
    clean_rows, mixed_units, unassigned_units = _split_assignment_audit(
        list(cast("Sequence[Mapping[str, Any]]", raw_evidence))
    )
    return clean_rows, (
        max(audit_counts[0], mixed_units),
        max(audit_counts[1], unassigned_units),
    )


def _validate_assignment_coverage(evidence: Sequence[Mapping[str, Any]], request: Any) -> None:
    if {row["population"] for row in evidence} != set(request.populations):
        _reject("artifact.extension.invalid", "assignment population coverage mismatch")


def _validate_noop_coverage(_evidence: Sequence[Mapping[str, Any]], _request: Any) -> None:
    return


def _validate_noop_audit(_extension: Any) -> None:
    return


def _validate_noop_result(_extension: Any, _result: Sequence[Mapping[str, Any]]) -> None:
    return


def encode_extension(
    source: Any,
    request: ArtifactExtensionRequest | Mapping[str, Any],
    publication: ArtifactPublication,
    *,
    context: Any,
    rows: Sequence[Mapping[str, Any]] | None = None,
    definition: Mapping[str, Any] | None = None,
    source_recipe: Mapping[str, Any] | None = None,
    experiment_id: str | None = None,
) -> ArtifactExtensionRef:
    """Encode one exact catalog request and write it via ``publication``."""
    selected = cast(
        Mapping[str, Any],
        request if isinstance(request, Mapping) else _dump(request),
    )
    kind = str(selected.get("kind"))
    if kind not in _HANDLERS:
        _reject("artifact.extension.invalid", "unknown extension request kind", kind=kind)
    handler = _handler(kind)
    _require_operation(source, kind)
    entry = _catalog_entry(context, selected)
    trusted_definition = _definition(entry, definition)
    _source_recipe(entry, source_recipe)
    trusted_experiment_id = (
        experiment_id
        or getattr(source, "experiment_id", None)
        or getattr(getattr(source, "_experiment", None), "name", None)
    )
    if not isinstance(trusted_experiment_id, str) or not trusted_experiment_id:
        _reject(
            "artifact.extension.invalid", "extension producer requires trusted experiment identity"
        )
    raw_evidence = (
        rows
        if rows is not None
        else _source_rows(
            source,
            entry.request,
            kind,
            experiment_id=trusted_experiment_id,
            definition=trusted_definition,
        )
    )
    raw_evidence, audit_counts = handler.prepare_evidence(raw_evidence)
    table, evidence = _wire_table(raw_evidence, kind)
    _validate_rows(kind, evidence, experiment_id=trusted_experiment_id)
    handler.validate_coverage(evidence, cast(Any, entry.request))
    relation = publication.write_relation(cast(Any, handler.role), table)
    if not isinstance(relation, ArtifactRelationRef):
        relation = ArtifactRelationRef.model_validate(relation)
    if relation.role != handler.role:
        _reject("artifact.extension.invalid", "publication returned a relation with the wrong role")
    if (
        relation.artifact_id != publication.artifact_id
        or relation.generation_id != publication.generation_id
    ):
        _reject(
            "artifact.extension.invalid", "publication returned a relation for another generation"
        )
    extension = _extension_model(
        kind,
        relation=relation,
        entry=entry,
        definition=trusted_definition,
        rows=evidence,
        audit_counts=audit_counts,
    )
    return extension


def _validate_relation_ref(
    snapshot: ArtifactSnapshot, extension: ArtifactExtensionRef, *, expected_role: str
) -> Any:
    relation = extension.relation
    if (
        relation.artifact_id != snapshot.artifact_id
        or relation.generation_id != snapshot.generation_id
    ):
        _reject(
            "artifact.snapshot.mixed",
            "extension relation is bound to a different artifact generation",
        )
    if relation.role != expected_role:
        _reject(
            "artifact.extension.invalid",
            "extension relation role mismatch",
            expected=expected_role,
            actual=relation.role,
        )
    expected_pk = ARTIFACT_RELATION_PRIMARY_KEYS[expected_role]
    if relation.primary_key != expected_pk:
        _reject(
            "artifact.extension.invalid",
            "extension relation primary key mismatch",
            expected=expected_pk,
        )
    expected_schema = schema_sha256(expected_role, ARTIFACT_RELATION_SCHEMAS[expected_role])
    if relation.schema_sha256 != expected_schema:
        _reject(
            "artifact.extension.invalid",
            "extension relation schema digest mismatch",
            expected=expected_schema,
        )
    return snapshot.verify_relation(relation, expected_role=cast(Any, expected_role))


def _identity_cuped(
    extension: Any, request: Mapping[str, Any], definition: Mapping[str, Any] | None
) -> bool:
    return extension.metric_name == request.get("metric_name") and (
        definition is None
        or all(
            extension.model_dump(mode="json").get(field) == definition.get(field)
            for field in ("measure_key", "aggregation", "window_start_days", "window_end_days")
        )
    )


def _identity_assignment(
    extension: Any, request: Mapping[str, Any], definition: Mapping[str, Any] | None
) -> bool:
    if tuple(extension.populations) != tuple(request.get("populations", ())):
        return False
    if definition is None:
        return True
    dumped = extension.model_dump(mode="json")
    return all(
        field not in definition or dumped.get(field) == definition[field]
        for field in ("mixed_unit_count", "unassigned_unit_count", "assignment_fingerprint_sha256")
    )


def _identity_encouragement(
    extension: Any, request: Mapping[str, Any], definition: Mapping[str, Any] | None
) -> bool:
    return extension.uptake_name == request.get("uptake_name") and (
        definition is None
        or (
            extension.window_days == definition.get("window_days")
            and extension.one_sided == definition.get("one_sided")
        )
    )


def _identity_site_volume(
    extension: Any, _request: Mapping[str, Any], definition: Mapping[str, Any] | None
) -> bool:
    if definition is None:
        return True
    dumped = extension.model_dump(mode="json")
    return tuple(extension.measure_keys) == tuple(definition.get("measure_keys", ())) and all(
        field not in definition or dumped.get(field) == definition[field]
        for field in ("first_ds", "last_ds", "freshness")
    )


def _extension_identity_matches(
    extension: Any, request: Mapping[str, Any], definition: Mapping[str, Any] | None = None
) -> bool:
    handler = _HANDLERS.get(cast(str, request.get("kind")))
    return handler is not None and handler.identity_matches(extension, request, definition)


def _validate_assignment_audit(extension: Any) -> None:
    # Validate before serialization: model_copy bypasses Pydantic validation.
    for value in (extension.mixed_unit_count, extension.unassigned_unit_count):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _reject(
                "artifact.extension.invalid",
                "assignment audit counts must be nonnegative integers",
            )


def _validate_assignment_result(extension: Any, result: Sequence[Mapping[str, Any]]) -> None:
    expected_populations = set(extension.populations)
    actual_populations = {row["population"] for row in result}
    if actual_populations != expected_populations:
        _reject("artifact.extension.invalid", "assignment population coverage mismatch")
    fingerprint_payload = {
        "rows": _canonical_assignment_rows(result),
        "mixed_unit_count": extension.mixed_unit_count,
        "unassigned_unit_count": extension.unassigned_unit_count,
    }
    fingerprint = hashlib.sha256(canonical_json(fingerprint_payload).encode()).hexdigest()
    if extension.assignment_fingerprint_sha256 != fingerprint:
        _reject("artifact.extension.invalid", "assignment fingerprint does not match adopted rows")


def _validate_trigger_exposure_coverage(
    result: Sequence[Mapping[str, Any]], exposure_keys: Iterable[tuple[Any, ...]]
) -> None:
    actual = {
        tuple(row[field] for field in ARTIFACT_RELATION_PRIMARY_KEYS["exposures"]) for row in result
    }
    if not actual <= set(exposure_keys):
        _reject("artifact.extension.invalid", "extension/exposure join-key coverage mismatch")


def _validate_noop_exposure_coverage(
    _result: Sequence[Mapping[str, Any]], _exposure_keys: Iterable[tuple[Any, ...]]
) -> None:
    return


def validate_extension(
    snapshot: ArtifactSnapshot,
    extension: ArtifactExtensionRef | None,
    *,
    request: ArtifactExtensionRequest | Mapping[str, Any] | None = None,
    context: Any | None = None,
    experiment_id: str | None = None,
    exposure_keys: Iterable[tuple[Any, ...]] | None = None,
) -> list[Mapping[str, Any]]:
    """Verify one extension relation and return rows from the same snapshot."""
    if extension is None:
        _reject("artifact.extension.missing", "requested artifact extension is absent")
    assert extension is not None
    ext: Any = extension
    if ext.kind == "encouragement_uptake" and ext.extension_version == 1:
        from increment._source_types import raise_legacy_compliance_state

        raise_legacy_compliance_state(
            study_id=experiment_id or "artifact",
            reason="version-1 uptake evidence contains no first qualifying timestamp",
        )
    kind = ext.kind
    _reject_relabeled_kind(ext)
    handler = _HANDLERS.get(kind)
    if handler is None:
        _reject("artifact.extension.invalid", f"extension kind {kind!r} is not registered")
    handler = cast(_ExtensionHandler, handler)
    handler.validate_audit(ext)
    request_identity = _dump(request) if request is not None else _request_from_extension(ext)
    if _request_value(request_identity, "kind") != kind:
        _reject(
            "artifact.extension.invalid", "extension identity does not match selected catalog entry"
        )
    entry = None
    if context is not None:
        if request is None:
            candidates = [
                candidate
                for candidate in unit_day_artifact_extension_catalog(context)
                if ext.definition_sha256 == candidate.definition_sha256
                and ext.source_provenance_sha256 == candidate.source_provenance_sha256
            ]
            if len(candidates) != 1:
                _reject(
                    "artifact.extension.missing"
                    if not candidates
                    else "artifact.extension.invalid",
                    "extension provenance does not select one trusted catalog entry",
                )
            entry = candidates[0]
            request_identity = _dump(entry.request)
        else:
            entry = _catalog_entry(context, _dump(request_identity))
        if (
            ext.definition_sha256 != entry.definition_sha256
            or ext.source_provenance_sha256 != entry.source_provenance_sha256
        ):
            _reject(
                "artifact.extension.invalid", "extension provenance does not match trusted catalog"
            )
    definition_payload = None
    if entry is not None:
        import json

        definition_payload = json.loads(entry.canonical_definition_json)
    if entry is not None:
        _bind_request_definition(kind, cast(Any, request_identity), definition_payload or {})
    if not _extension_identity_matches(ext, cast(Any, request_identity), definition_payload):
        _reject(
            "artifact.extension.invalid", "extension identity does not match selected catalog entry"
        )
    raw = _validate_relation_ref(snapshot, ext, expected_role=handler.role)
    result = _rows(snapshot.execute(raw))
    _validate_rows(kind, result, experiment_id=experiment_id)
    relation = ext.relation
    handler.validate_result(ext, result)
    if relation.row_count != len(result):
        _reject("artifact.extension.invalid", "extension relation row_count mismatch")
    expected_content = content_sha256(
        kind,
        ARTIFACT_RELATION_SCHEMAS[kind],
        result,
        primary_key=ARTIFACT_RELATION_PRIMARY_KEYS[kind],
    )
    if relation.content_sha256 != expected_content:
        _reject("artifact.extension.invalid", "extension relation content digest mismatch")
    if exposure_keys is not None:
        handler.validate_exposure(result, exposure_keys)
    return result


def _request_cuped(extension: Any) -> dict[str, Any]:
    return {"kind": extension.kind, "metric_name": extension.metric_name}


def _request_assignment(extension: Any) -> dict[str, Any]:
    return {"kind": extension.kind, "populations": extension.populations}


def _request_encouragement(extension: Any) -> dict[str, Any]:
    return {"kind": extension.kind, "uptake_name": extension.uptake_name}


def _request_site_volume(extension: Any) -> dict[str, Any]:
    return {"kind": extension.kind, "metric_names": extension.measure_keys}


def _request_from_extension(extension: Any) -> dict[str, Any]:
    handler = _HANDLERS.get(extension.kind)
    return (
        {"kind": extension.kind} if handler is None else handler.request_from_extension(extension)
    )


def read_site_volume_extension(
    snapshot: ArtifactSnapshot,
    extension: SiteVolumeExtension | None,
    *,
    request: ArtifactExtensionRequest | Mapping[str, Any] | None = None,
    context: Any | None = None,
    experiment_id: str | None = None,
    exposure_keys: Iterable[tuple[Any, ...]] | None = None,
) -> list[Mapping[str, Any]]:
    """Read site volume, densifying a genuinely empty complete relation to zero."""
    if extension is not None and extension.freshness.loaded_through < extension.last_ds:
        _reject(
            "artifact.extension.invalid", "site-volume freshness does not cover declared last_ds"
        )
    rows = validate_extension(
        snapshot,
        extension,
        request=request,
        context=context,
        experiment_id=experiment_id,
        exposure_keys=exposure_keys,
    )
    assert extension is not None
    if rows:
        expected = {(row["ds"], row["measure_key"]) for row in rows}
        all_keys = {
            (day, measure)
            for day in _date_range(extension.first_ds, extension.last_ds)
            for measure in extension.measure_keys
        }
        if not expected <= all_keys:
            _reject(
                "artifact.extension.invalid",
                "site-volume relation has out-of-range date/measure coverage",
            )
        if expected != all_keys and not extension.freshness.declared_complete:
            _reject("artifact.extension.invalid", "site-volume relation has incomplete freshness")
        if expected == all_keys:
            return rows
        if experiment_id is None:
            _reject(
                "artifact.extension.invalid",
                "site-volume zero densification requires experiment_id",
            )
        return rows + [
            {
                "experiment_id": experiment_id,
                "ds": day,
                "measure_key": measure,
                "n_events": 0,
                "sum_value": 0.0,
                "min_value": 0.0,
                "max_value": 0.0,
            }
            for day, measure in sorted(all_keys - expected)
        ]
    if experiment_id is None:
        _reject(
            "artifact.extension.invalid", "site-volume zero densification requires experiment_id"
        )
    if not extension.freshness.declared_complete:
        _reject("artifact.extension.invalid", "empty site-volume relation is not declared complete")
    return [
        {
            "experiment_id": experiment_id,
            "ds": day,
            "measure_key": measure,
            "n_events": 0,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        }
        for day in _date_range(extension.first_ds, extension.last_ds)
        for measure in extension.measure_keys
    ]


_HANDLERS = {
    "breakout_dimension": _build_handler(
        ExtensionKindSpec(
            kind="breakout_dimension",
            role="breakout_dimension",
            operation="breakout_source",
            protocol=BreakoutSourceOperation,
            model_cls=BreakoutDimensionExtension,
            fields=(("dimension_name", "property_name"), ("source_name", "source_name")),
            row_validator=partial(_validate_dimension_row, value_field="dimension_value"),
            source_rows=_dimension_source_rows,
            bind_definition=_bind_dimension,
            validate_definition=_validate_definition_dimension,
        )
    ),
    "factor_dimension": _build_handler(
        ExtensionKindSpec(
            kind="factor_dimension",
            role="factor_dimension",
            operation="breakout_source",
            protocol=BreakoutSourceOperation,
            model_cls=FactorDimensionExtension,
            fields=(("factor_name", "property_name"), ("source_name", "source_name")),
            row_validator=partial(_validate_dimension_row, value_field="factor_value"),
            source_rows=_dimension_source_rows,
            bind_definition=_bind_dimension,
            validate_definition=_validate_definition_dimension,
        )
    ),
    "cluster_identity": _build_handler(
        ExtensionKindSpec(
            kind="cluster_identity",
            role="cluster_identity",
            operation="day_source",
            protocol=DaySourceOperation,
            model_cls=ClusterIdentityExtension,
            fields=(("cluster_name", "cluster_name"),),
            row_validator=_validate_cluster_row,
            source_rows=_day_source_rows("cluster_identity"),
        )
    ),
    "cuped_preperiod": _build_handler(
        ExtensionKindSpec(
            kind="cuped_preperiod",
            role="cuped_preperiod",
            operation="day_source",
            protocol=DaySourceOperation,
            model_cls=CupedPreperiodExtension,
            fields=(("metric_name", "metric_name"),),
            row_validator=_validate_cuped_row,
            source_rows=_day_source_rows("cuped_preperiod"),
            bind_definition=_bind_cuped,
            identity_matches=_identity_cuped,
            request_from_extension=_request_cuped,
            model=_model_cuped,
        )
    ),
    "assignment_counts": _ExtensionHandler(
        role="assignment_counts",
        operation="triggered_counts",
        protocol=TriggeredCountsOperation,
        source_rows=_source_assignment_count_rows,
        model=_model_assignment,
        identity_matches=_identity_assignment,
        request_from_extension=_request_assignment,
        validate_row=_validate_assignment_row,
        bind_definition=_bind_assignment,
        validate_definition=_validate_definition_assignment,
        prepare_evidence=_prepare_assignment_evidence,
        validate_audit=_validate_assignment_audit,
        validate_result=_validate_assignment_result,
        validate_coverage=_validate_assignment_coverage,
        validate_exposure=_validate_noop_exposure_coverage,
        read=validate_extension,
    ),
    "trigger_population": _build_handler(
        ExtensionKindSpec(
            kind="trigger_population",
            role="trigger_population",
            operation="triggered_source",
            protocol=TriggeredPopulationOperation,
            model_cls=TriggerPopulationExtension,
            fields=(("trigger_name", "trigger_name"),),
            row_validator=_validate_trigger_row,
            source_rows=_source_trigger_population_rows,
            validate_exposure=_validate_trigger_exposure_coverage,
        )
    ),
    "encouragement_uptake": _build_handler(
        ExtensionKindSpec(
            kind="encouragement_uptake",
            role="encouragement_uptake",
            operation="day_source",
            protocol=DaySourceOperation,
            model_cls=EncouragementUptakeExtension,
            fields=(("uptake_name", "uptake_name"),),
            row_validator=_validate_encouragement_row,
            source_rows=_day_source_rows("encouragement_uptake"),
            identity_matches=_identity_encouragement,
            request_from_extension=_request_encouragement,
            model=_model_encouragement,
        )
    ),
    "site_volume": _ExtensionHandler(
        role="site_volume",
        operation="sitewide_evidence",
        protocol=SitewideEvidenceOperation,
        source_rows=_source_site_volume_extension_rows,
        model=_model_site_volume,
        identity_matches=_identity_site_volume,
        request_from_extension=_request_site_volume,
        validate_row=partial(_validate_stats_row, minimum=1, exact_integer_limit=True),
        bind_definition=_bind_site_volume,
        validate_definition=_validate_definition_site_volume,
        prepare_evidence=_prepare_plain_evidence,
        validate_audit=_validate_noop_audit,
        validate_result=_validate_noop_result,
        validate_coverage=_validate_noop_coverage,
        validate_exposure=_validate_noop_exposure_coverage,
        read=read_site_volume_extension,
    ),
    "unit_covariate": _build_handler(
        ExtensionKindSpec(
            kind="unit_covariate",
            role="unit_covariate",
            operation="day_source",
            protocol=DaySourceOperation,
            model_cls=UnitCovariateExtension,
            fields=(("covariate_name", "property_name"), ("source_name", "source_name")),
            row_validator=lambda _row: None,
            source_rows=_unit_covariate_source_rows("unit_covariate"),
            bind_definition=_bind_dimension,
            validate_definition=_validate_definition_dimension,
            model=_unit_covariate_model(UnitCovariateExtension),
        )
    ),
    "unit_covariate_level": _build_handler(
        ExtensionKindSpec(
            kind="unit_covariate_level",
            role="unit_covariate_level",
            operation="day_source",
            protocol=DaySourceOperation,
            model_cls=UnitCovariateLevelExtension,
            fields=(("covariate_name", "property_name"), ("source_name", "source_name")),
            row_validator=lambda _row: None,
            source_rows=_unit_covariate_source_rows("unit_covariate_level"),
            bind_definition=_bind_dimension,
            validate_definition=_validate_definition_dimension,
            model=_unit_covariate_model(UnitCovariateLevelExtension),
        )
    ),
}


def _date_range(first: date, last: date) -> Iterable[date]:
    current = first
    while current <= last:
        yield current
        current = date.fromordinal(current.toordinal() + 1)


def _reject_relabeled_kind(extension: Any) -> None:
    """Refuse an extension whose stored kind disagrees with its concrete model."""
    declared = type(extension).model_fields["kind"].default
    if declared != extension.kind:
        _reject(
            "artifact.extension.invalid", "extension identity does not match selected catalog entry"
        )


def read_extension(
    snapshot: ArtifactSnapshot, extension: ArtifactExtensionRef | None, **kwargs: Any
) -> list[Mapping[str, Any]]:
    if extension is None:
        return validate_extension(snapshot, extension, **kwargs)
    _reject_relabeled_kind(extension)
    handler = _HANDLERS.get(extension.kind)
    reader = validate_extension if handler is None else handler.read
    return reader(snapshot, extension, **kwargs)


def read_unit_covariate(
    extension: ArtifactExtensionRef,
    rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    unit_ids: Sequence[str],
    implementation: Any,
) -> nw.Series[Any]:
    """Decode one verified unit-covariate relation into an aligned typed series."""
    decoder = _handler(extension.kind).read_unit_covariate
    if decoder is None:
        _reject(
            "artifact.extension.invalid",
            "extension is not a unit covariate",
            extension_kind=extension.kind,
            operation="unit_covariate",
            route="provide a unit_covariate or unit_covariate_level extension",
        )
    assert decoder is not None
    return decoder(
        extension,
        rows,
        name=name,
        unit_ids=unit_ids,
        implementation=implementation,
    )


def _unsupported_analysis(kind: str) -> None:
    _reject(
        "artifact.evidence.unavailable",
        f"unit-day artifact extensions do not support {kind} operations",
        operation=kind,
    )


def encode_observational_extension(*args: Any, **kwargs: Any) -> None:
    _unsupported_analysis("observational")


def read_observational_extension(*args: Any, **kwargs: Any) -> None:
    _unsupported_analysis("observational")


def validate_observational_extension(*args: Any, **kwargs: Any) -> None:
    _unsupported_analysis("observational")


def validate_contrast_extension(*args: Any, **kwargs: Any) -> None:
    _unsupported_analysis("contrast")


def encode_contrast_extension(*args: Any, **kwargs: Any) -> None:
    _unsupported_analysis("contrast")


def read_contrast_extension(*args: Any, **kwargs: Any) -> None:
    _unsupported_analysis("contrast")


__all__ = [
    "encode_extension",
    "validate_extension",
    "read_extension",
    "read_unit_covariate",
    "read_site_volume_extension",
    "encode_observational_extension",
    "read_observational_extension",
    "validate_observational_extension",
    "validate_contrast_extension",
    "encode_contrast_extension",
    "read_contrast_extension",
]
