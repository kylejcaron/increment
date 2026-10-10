"""Adoption of immutable format-1 unit/day artifact snapshots.

This module is deliberately store-facing only: it never knows how a relation
was produced and never invokes an upstream definitions source.  A reader pins
one ``ArtifactSnapshot`` for its entire lifetime and materialises only the
base relations needed by the first reduction.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Self, cast

import ibis
import ibis.expr.types as ir

from increment._frame_validation import FRAME_UNIT_FRAME_PANEL
from increment._metric_specs import MetricSpec, coerce_metrics, synthesise_metric
from increment._moment_plan import COMPLIANCE_ARM_FROM_CLUSTER_ROW
from increment._source_types import (
    ComplianceArm,
    ComplianceSummary,
    invalid_compliance_state,
    raise_legacy_compliance_state,
    validate_compliance_design_match,
)
from increment._window import NO_DATA_SIGNAL
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.estimation._readout_refusals import refuse_observational_quantile
from increment.query.artifact_contract import (
    REFUSALS,
    ArtifactSnapshot,
    ArtifactStore,
    _aggregate_sum_within_extrema_sql,
    open_trusted_manifest_snapshot,
)
from increment.query.artifact_publish import _effective_snapshot_edge
from increment.query.builders import (
    _censor_to_observable_window,
    _final_maturity_day,
    _local_date_at_offset,
    _utc_timestamp_literal,
    asof_group_summary,
    cohort_group_summary,
    compliance_event_horizon,
    daily_group_summary,
    day_boundary_offset,
    declared_binary_metrics,
    group_summary,
    join_breakout_dimension,
    join_pre_period_covariate,
    panel_spine,
    unit_totals,
    window_bound_stats,
    winsorize_unit_totals,
)
from increment.query.schemas import (
    UNIT_DAY_ARTIFACT_PRIMARY_KEYS,
    UNIT_DAY_ARTIFACT_RELATION_SCHEMAS,
)
from increment.semantics.artifact import (
    ArtifactContext,
    MeasureManifest,
    RatioMetricMeasure,
    SimpleMetricMeasure,
    UnitDayArtifactManifest,
    UnitDayArtifactRef,
)
from increment.semantics.design import Encouragement
from increment.semantics.models import (
    AnalysisPlan,
    Experiment,
    MeanMetric,
    Metric,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
)
from increment.sequential_source import SequentialSourceMixin
from increment.sources import Grain, SourceContext, SourceOperation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.typing import IntoDataFrame


ARTIFACT_OPERATIONS: frozenset[SourceOperation] = frozenset(
    {"moments_source", "day_source", "export_moments"}
)

_ARTIFACT_GRAIN = RefusalSpec(
    "artifact.operation.unsupported",
    CapabilityError,
    template=(
        "artifact source cannot perform {operation!r} for {request!r}; "
        "available capability is {offered!r}. {route}"
    ),
)
_ARTIFACT_METRIC = REFUSALS["artifact.metric.binding_mismatch"]
_ARTIFACT_RELATION_INVALID = RefusalSpec(
    "artifact.relation.invalid", CapabilityError, template="{relation}: {check} -- {detail}"
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "query.artifact_reader.snapshot_execution_returned": "snapshot execution returned unsupported relation {type_name}",
        "query.artifact_reader.artifact_moment.verification_lazy_digest": "only verification='lazy_digest' is supported",
        "query.artifact_reader.artifact_moment.manifest_no_metric": "artifact manifest has no metric bindings",
    },
)
_raise = raiser(_REFUSALS)
LAZY_DIGEST_VERIFICATION = _REFUSALS[
    "query.artifact_reader.artifact_moment.verification_lazy_digest"
]


def _relation_refuse(code: str, message: str) -> NoReturn:
    refuse(REFUSALS[code], message=message)


def _rows(value: object) -> list[dict[str, Any]]:
    """Convert one snapshot execution result to row mappings without I/O."""
    if hasattr(value, "to_pylist"):
        converter = cast("Callable[[], Any]", value.to_pylist)
        return [dict(cast("Mapping[str, Any]", row)) for row in converter()]
    if hasattr(value, "to_dicts"):
        converter = cast("Callable[[], Any]", value.to_dicts)
        return [dict(cast("Mapping[str, Any]", row)) for row in converter()]
    if isinstance(value, Mapping):
        mapping = cast("Mapping[str, Sequence[Any]]", value)
        names = list(mapping)
        if not names:
            return []
        return [
            dict(zip(names, row, strict=True))
            for row in zip(*(mapping[name] for name in names), strict=True)
        ]
    if hasattr(value, "to_dict"):
        converter = cast("Callable[..., Any]", value.to_dict)
        return [dict(cast("Mapping[str, Any]", row)) for row in converter("records")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [dict(cast("Mapping[str, Any]", row)) for row in value]
    _raise("query.artifact_reader.snapshot_execution_returned", type_name=type(value).__name__)


def restrict_to_units(table: ir.Table, units: frozenset[str]) -> ir.Table:
    """Keep the rows of *table* whose ``unit_id`` is in *units*.

    The population travels as a relation the warehouse joins against, not as
    one string literal per unit in the query text.
    """
    schema = ibis.schema({"unit_id": "string"})
    members = ibis.memtable({"unit_id": sorted(units)}, schema=schema)
    return table.filter(table.unit_id.isin(members.unit_id))


def _measure_edge(measure: MeasureManifest, *, prefer_persisted_horizon: bool) -> dt.date | None:
    """Resolve a measure's observed or maturity edge from its manifest."""
    if prefer_persisted_horizon and "event_horizon" in measure.model_fields_set:
        edge = (
            None
            if measure.event_horizon is None or measure.event_horizon == NO_DATA_SIGNAL
            else measure.event_horizon
        )
    else:
        edge = (
            None
            if measure.freshness.loaded_through == NO_DATA_SIGNAL
            else measure.freshness.loaded_through
        )
    if "certified_edge" in measure.model_fields_set and measure.certified_edge is not None:
        return measure.certified_edge
    if "observed_edge" in measure.model_fields_set and measure.observed_edge is not None:
        return measure.observed_edge
    return edge


def _scalar_date(value: object) -> dt.date | None:
    """Normalize scalar DATE results, including backend Timestamp/NaT values."""
    if value is None or value != value:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    return cast("dt.date", value)


def _day_edge_scalar(value: dt.date | ir.Scalar | None) -> ir.Scalar:
    """Convert a Python or expression day edge to a date scalar."""
    if value is None:
        return cast("ir.Scalar", ibis.null().cast("date"))
    if isinstance(value, dt.date):
        return ibis.literal(value, type="date")
    return cast("ir.Scalar", value.cast("date"))


def _outcome_edge(
    measures: Sequence[MeasureManifest],
    stats: ir.Table,
    measure_keys: frozenset[str],
    execute: Callable[[ir.Table], object],
    *,
    prefer_persisted_horizon: bool = False,
) -> dt.date | None:
    """Latest date these outcome measure(s) are known to cover -- never an
    enrollment/exposure date.

    Prefers each measure's own persisted edge (see `_measure_edge`);
    falls back to the latest observed `measure_stats` row only when every
    relevant measure_key has no usable edge at all -- a single bounded
    warehouse aggregate (`ds.max()`), not a Python `max()` over materialised
    rows. For maturity, explicit certified or cutoff bounds take precedence;
    legacy manifests without those fields continue to use `loaded_through`.
    Spine-sizing callers pass `prefer_persisted_horizon=True` to prefer
    `event_horizon` when no explicit snapshot bound exists.
    """
    edges = [
        edge
        for m in measures
        if m.measure_key in measure_keys
        for edge in (_measure_edge(m, prefer_persisted_horizon=prefer_persisted_horizon),)
        if edge is not None
    ]
    if edges:
        return min(edges)
    scoped = stats.filter(stats.measure_key.isin(list(measure_keys)))
    edge = _rows(execute(scoped.aggregate(edge=scoped.ds.max())))[0]
    return _scalar_date(edge["edge"])


def _union_scope_measure_keys(
    metric_measures: Sequence[SimpleMetricMeasure | RatioMetricMeasure],
    names: frozenset[str] | None,
) -> frozenset[str]:
    """Every measure key the shared spine edge must account for --
    mirrors native_source.py's `_union_event_horizon_for`, which unions
    each metric's primary event stream with, for every `RatioMetric`,
    its denominator stream too (`den_tbl` appended after the primary
    streams, `_union_event_horizon_for:1228-1239`). `names=None` scopes
    to every manifest binding; otherwise only bindings for those metric
    names.
    """
    keys: set[str] = set()
    for binding in metric_measures:
        if names is not None and binding.metric_name not in names:
            continue
        if isinstance(binding, RatioMetricMeasure):
            keys.add(binding.numerator_measure_key)
            keys.add(binding.denominator_measure_key)
        else:
            keys.add(binding.measure_key)
    return frozenset(keys)


def _union_outcome_edge(
    measures: Sequence[MeasureManifest],
    scope_keys: frozenset[str],
) -> dt.date | None:
    """Spine-wide day-axis edge, shared across every metric this reader
    session reduces -- mirrors native_source.py's `_union_event_horizon`
    (default scope: every metric the source was constructed with, unioned
    over each metric's primary stream, plus its denominator stream too
    for a `RatioMetric` -- see `_union_scope_measure_keys`).

    A spine's day count per unit is literally an `avg_calendar_day`
    divisor: sizing it from one metric's own (possibly lagging) measure
    would understate that divisor relative to the fused native path,
    which always sizes a shared spine from the freshest measure among
    every metric it materializes together. Each metric's own maturity
    still gates separately via `data_as_of` (scoped to that metric's own
    numerator AND denominator) -- this is only the shared spine extent.

    Always prefers each measure's persisted `event_horizon` (see
    `_measure_edge`) -- this is exclusively a spine-sizing edge, never a
    censoring cap. A measure with literally zero matching rows and no
    persisted horizon contributes no edge, so it can never win the union
    and inflate the spine to a far-future date. `None` when every
    candidate measure has no usable edge.
    """
    candidates = [
        edge
        for m in measures
        if m.measure_key in scope_keys
        for edge in (_measure_edge(m, prefer_persisted_horizon=True),)
        if edge is not None
    ]
    return max(candidates) if candidates else None


def _schema_names(table: object) -> tuple[str, ...] | None:
    schema = getattr(table, "schema", None)
    if schema is None:
        return None
    schema = schema() if callable(schema) else schema
    names = getattr(schema, "names", None)
    if names is not None:
        return tuple(names)
    if isinstance(schema, Mapping):
        return tuple(schema)
    try:
        return tuple(schema)
    except TypeError:
        return None


def _validate_exposures_sql(
    relation: ir.Table,
    manifest: UnitDayArtifactManifest,
    execute: Callable[[ir.Table], object],
) -> int:
    """SQL-pushed integrity check for one verified ``exposures`` relation.

    Mirrors the retired per-row ``_validate_exposures``: one bounded
    ``.aggregate(...)`` computes every named boolean flag in a single
    round trip. Nullability/type checks ``isinstance(...)`` used to
    perform on individual cells collapse to a null/empty check now that
    the relation's schema is already digest-verified by
    ``ArtifactSnapshot.verify_relation`` -- a verified ``STRING`` column
    cannot hold a non-string Python value.
    """
    pk_cols = list(UNIT_DAY_ARTIFACT_PRIMARY_KEYS["exposures"])
    offset = day_boundary_offset(manifest.day_boundary)
    expected_date = _local_date_at_offset(relation.first_exposure_ts, offset)
    checks = relation.aggregate(
        n_rows=relation.count(),
        bad_experiment_id=(relation.experiment_id != manifest.experiment_id).any(),
        bad_identity=(
            relation.unit_id.isnull()
            | (relation.unit_id == "")
            | relation.group_id.isnull()
            | (relation.group_id == "")
        ).any(),
        bad_timestamp=relation.first_exposure_ts.isnull().any(),
        bad_day_boundary=(relation.first_exposure_date != expected_date).any(),
    )
    distinct_pk_table = relation.select(*pk_cols).distinct()
    distinct_pk = distinct_pk_table.aggregate(n_distinct=distinct_pk_table.count())
    row = _rows(execute(checks.cross_join(distinct_pk)))[0]
    if row["n_rows"] != row["n_distinct"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="exposures",
            check="primary_key",
            detail="primary key is not unique",
        )
    if row["bad_experiment_id"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="exposures",
            check="experiment_id",
            detail="experiment_id does not match manifest",
        )
    if row["bad_identity"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="exposures",
            check="identity_type",
            detail="unit_id and group_id must be non-empty strings",
        )
    if row["bad_timestamp"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="exposures",
            check="timestamp_type",
            detail="first_exposure_ts must be a timestamp",
        )
    if row["bad_day_boundary"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="exposures",
            check="day_boundary",
            detail="first_exposure_date disagrees with day boundary",
        )

    return int(row["n_rows"])


def _validate_stats_sql(
    relation: ir.Table,
    exposures: ir.Table,
    manifest: UnitDayArtifactManifest,
    execute: Callable[[ir.Table], object],
) -> int:
    """SQL-pushed integrity check for one verified ``measure_stats`` relation.

    Mirrors the retired per-row ``_validate_stats``: joins the
    already warehouse-backed ``exposures`` relation (``self._ensure("exposures")``,
    itself an ``ir.Table``) in SQL rather than building a Python
    ``exposure_dates`` dict, then computes every named boolean flag in
    ONE bounded ``.aggregate(...)`` round trip. Finiteness
    (``isnan``/``isinf``) stays a real check -- a verified ``FLOAT64``
    column still admits NaN/Inf, which a schema check alone cannot rule out.
    """
    pk_cols = list(UNIT_DAY_ARTIFACT_PRIMARY_KEYS["measure_stats"])
    declared = [m.measure_key for m in manifest.measures]
    joined = relation.left_join(
        exposures.select("experiment_id", "unit_id", "first_exposure_date"),
        ["experiment_id", "unit_id"],
        rname="{name}_exposure",
    )
    lower_bound = ibis.greatest(ibis.literal(manifest.first_ds), joined.first_exposure_date)
    checks = joined.aggregate(
        n_rows=joined.count(),
        bad_experiment_id=(joined.experiment_id != manifest.experiment_id).any(),
        bad_exposure_match=joined.first_exposure_date.isnull().any(),
        bad_measure_key_type=(joined.measure_key.isnull() | (joined.measure_key == "")).any(),
        bad_unit_id_type=(joined.unit_id.isnull() | (joined.unit_id == "")).any(),
        bad_undeclared_measure=(~joined.measure_key.isin(declared)).any(),
        bad_ds_type=joined.ds.isnull().any(),
        bad_ds_domain=(
            joined.first_exposure_date.notnull()
            & ((joined.ds < lower_bound) | (joined.ds > manifest.last_ds))
        ).any(),
        bad_n_events_range=(joined.n_events.isnull() | ~joined.n_events.between(1, 2**53)).any(),
        bad_sum_value=(
            joined.sum_value.isnull() | joined.sum_value.isnan() | joined.sum_value.isinf()
        ).any(),
        bad_min_value=(
            joined.min_value.isnull() | joined.min_value.isnan() | joined.min_value.isinf()
        ).any(),
        bad_max_value=(
            joined.max_value.isnull() | joined.max_value.isnan() | joined.max_value.isinf()
        ).any(),
        bad_min_max_order=(joined.min_value > joined.max_value).any(),
        bad_sum_extrema=(
            ~_aggregate_sum_within_extrema_sql(
                joined.sum_value,
                joined.n_events,
                joined.min_value,
                joined.max_value,
            )
        ).any(),
        bad_single_event_identity=(
            (joined.n_events == 1)
            & ((joined.sum_value != joined.min_value) | (joined.sum_value != joined.max_value))
        ).any(),
    )
    distinct_pk_table = relation.select(*pk_cols).distinct()
    distinct_pk = distinct_pk_table.aggregate(n_distinct=distinct_pk_table.count())
    row = _rows(execute(checks.cross_join(distinct_pk)))[0]
    if row["n_rows"] != row["n_distinct"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="primary_key",
            detail="primary key is not unique",
        )
    if row["bad_experiment_id"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="experiment_id",
            detail="experiment_id does not match manifest",
        )
    if row["bad_exposure_match"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="exposure_match",
            detail="row has no matching exposure",
        )
    if row["bad_measure_key_type"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="measure_key_type",
            detail="measure_key must be a non-empty string",
        )
    if row["bad_unit_id_type"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="unit_id_type",
            detail="unit_id must be a non-empty string",
        )
    if row["bad_undeclared_measure"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="undeclared_measure",
            detail="references an undeclared measure",
        )
    if row["bad_ds_type"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="ds_type",
            detail="ds must be a date",
        )
    if row["bad_ds_domain"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="ds_domain",
            detail="ds lies outside the admitted date domain",
        )
    if row["bad_n_events_range"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="n_events_range",
            detail="n_events must satisfy 1 <= n_events <= 2**53",
        )
    if row["bad_sum_value"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="sum_value",
            detail="must be finite",
        )
    if row["bad_min_value"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="min_value",
            detail="must be finite",
        )
    if row["bad_max_value"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="max_value",
            detail="must be finite",
        )
    if row["bad_min_max_order"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="min_max_order",
            detail="min_value must be <= max_value",
        )
    if row["bad_sum_extrema"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="sum_extrema",
            detail="sum_value lies outside n_events * extrema",
        )
    if row["bad_single_event_identity"]:
        refuse(
            _ARTIFACT_RELATION_INVALID,
            relation="measure_stats",
            check="single_event_identity",
            detail="single-event measure_stats sum/min/max must be equal",
        )

    return int(row["n_rows"])


@dataclass(frozen=True, slots=True)
class _MetricResolution:
    """Trusted definition and manifest binding for one requested metric."""

    spec: MetricSpec
    binding: SimpleMetricMeasure | RatioMetricMeasure
    definition: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _TriggeredReductionInputs:
    """Trigger-local relations, event spine edge, and certified data edge."""

    exposures: ir.Table
    measure_stats: ir.Table
    observed_edge: dt.date
    finalized_as_of: dt.date | None
    spine_edge: ir.Scalar


def _validate_metric_identity(resolution: _MetricResolution) -> None:
    spec = resolution.spec
    binding = resolution.binding
    definition = resolution.definition
    if isinstance(binding, RatioMetricMeasure) != (spec.type == "ratio"):
        refuse(
            _ARTIFACT_METRIC,
            message=f"metric {spec.name!r} type does not match manifest binding",
        )
    if definition.get("type") and spec.type != definition["type"]:
        refuse(
            _ARTIFACT_METRIC,
            message=f"metric {spec.name!r} type differs from trusted context",
        )
    if isinstance(binding, RatioMetricMeasure):
        if (spec.numerator, spec.denominator) != (
            binding.numerator_measure_key,
            binding.denominator_measure_key,
        ):
            refuse(
                _ARTIFACT_METRIC,
                message=f"metric {spec.name!r} ratio binding differs from manifest",
            )
    elif spec.y_column != binding.measure_key:
        refuse(
            _ARTIFACT_METRIC,
            message=f"metric {spec.name!r} measure binding differs from manifest",
        )


def _validate_metric_semantics(resolution: _MetricResolution) -> None:
    spec = resolution.spec
    definition = resolution.definition
    for key in ("window_days", "threshold_days", "quantile"):
        actual = getattr(spec, key)
        trusted = definition.get(key)
        if key == "window_days" and trusted is None and definition.get("type") == "ratio":
            numerator = definition.get("numerator")
            if isinstance(numerator, Mapping):
                trusted = numerator.get("window_days")
        if trusted is None:
            if actual is not None:
                refuse(
                    _ARTIFACT_METRIC,
                    message=f"metric {spec.name!r} adds semantics absent from trusted context",
                )
        else:
            if key == "threshold_days" and isinstance(trusted, list):
                trusted = tuple(trusted)
            if actual != trusted:
                refuse(
                    _ARTIFACT_METRIC,
                    message=f"metric {spec.name!r} {key} differs from trusted context",
                )
    actual_winsorization = (
        spec.winsorization.model_dump(mode="json") if spec.winsorization else None
    )
    trusted_winsorization = definition.get("winsorization")
    if trusted_winsorization is None:
        if actual_winsorization is not None:
            refuse(
                _ARTIFACT_METRIC,
                message=f"metric {spec.name!r} adds semantics absent from trusted context",
            )
    elif actual_winsorization != trusted_winsorization:
        refuse(
            _ARTIFACT_METRIC,
            message=f"metric {spec.name!r} winsorization differs from trusted context",
        )


def _record_metric_resolution(source: Any, resolution: _MetricResolution) -> None:
    spec = resolution.spec
    definition = resolution.definition
    binding = resolution.binding
    if isinstance(binding, RatioMetricMeasure):
        numerator = definition.get("numerator", {})
        denominator = definition.get("denominator", {})
        if isinstance(numerator, Mapping) and isinstance(denominator, Mapping):
            source._ratio_semantics[spec.name] = (
                str(numerator.get("aggregation", "count")),
                numerator.get("window_days"),
                str(denominator.get("aggregation", "count")),
                denominator.get("window_days"),
            )
    source._aggregation_by_metric[spec.name] = str(definition.get("aggregation", "count"))


def _declared_metric_definitions(canonical_json: str) -> dict[str, Mapping[str, Any]]:
    """Typed, unique metric declarations keyed by name, exactly the experiment's roster.

    Mirrors the roster validation of ``query.source._artifact_source_context`` (typed
    declarations, unique names, declared set equal to ``Experiment.metric_names``) so the
    direct source path refuses the same contexts with the same code.
    """
    from pydantic import TypeAdapter

    from increment.semantics.design import Encouragement
    from increment.semantics.models import EncouragementDeclaration, Experiment

    invalid = False
    declared: dict[str, Mapping[str, Any]] = {}
    try:
        payload = json.loads(canonical_json)
        items = payload["definitions"]["metrics"]
        adapter: TypeAdapter[Metric] = TypeAdapter(Metric)
        for item in items:
            name = adapter.validate_python(item).name
            invalid = invalid or name in declared
            declared[name] = item
        experiment_payload = dict(payload["experiment"])
        if (experiment_payload.get("design") or {}).get("mechanism") == "encouragement":
            design = Encouragement.model_validate(experiment_payload["design"])
            experiment_payload["design"] = EncouragementDeclaration(
                uptake=design.uptake,
                one_sided=design.one_sided,
                exclusion_restriction=design.exclusion_restriction,
            )
        roster = Experiment.model_validate(experiment_payload).metric_names
        invalid = invalid or set(declared) != set(roster)
    except (KeyError, TypeError, ValueError, AttributeError):
        invalid = True
    if invalid:
        _relation_refuse(
            "artifact.context.mismatch",
            "artifact context metric declarations must be valid, unique and match the "
            "experiment's metric roster",
        )
    return declared


def _metric_spec_from_binding(
    binding: SimpleMetricMeasure | RatioMetricMeasure,
    definition: Mapping[str, Any],
) -> MetricSpec:
    kwargs: dict[str, Any] = {
        "name": binding.metric_name,
        "type": definition["type"],
    }
    if isinstance(binding, SimpleMetricMeasure):
        kwargs["value_column"] = binding.measure_key
    else:
        kwargs.update(
            {
                "numerator": binding.numerator_measure_key,
                "denominator": binding.denominator_measure_key,
            }
        )
    for key in (
        "window_days",
        "threshold_days",
        "quantile",
        "missing",
        "covariate_missing",
        "winsorization",
    ):
        if key in definition:
            kwargs[key] = definition[key]
    return MetricSpec(**kwargs)


def _record_binding_resolution(
    source: Any,
    binding: SimpleMetricMeasure | RatioMetricMeasure,
    definition: Mapping[str, Any],
) -> None:
    source._aggregation_by_metric[binding.metric_name] = str(definition.get("aggregation", "count"))
    if isinstance(binding, RatioMetricMeasure):
        numerator = definition.get("numerator", {})
        denominator = definition.get("denominator", {})
        if isinstance(numerator, Mapping) and isinstance(denominator, Mapping):
            source._ratio_semantics[binding.metric_name] = (
                str(numerator.get("aggregation", "count")),
                numerator.get("window_days"),
                str(denominator.get("aggregation", "count")),
                denominator.get("window_days"),
            )


class _SnapshotLifecycle(AbstractContextManager[None]):
    """Closed-state and release of one pinned snapshot, shared by every view of it.

    It is itself a context manager, so a view can be built from it exactly as from the
    snapshot context it wraps; ``ArtifactMomentSource`` adopts it instead of re-wrapping.
    """

    def __init__(self, context: AbstractContextManager[object]) -> None:
        self._context = context
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._context.__exit__(None, None, None)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ArtifactMomentSource(SequentialSourceMixin):
    """Moment source backed by one immutable, lazily verified artifact handle."""

    capabilities: frozenset[Grain] = frozenset({"total", "daily", "asof"})

    shape: Literal["unit_summary", "unit_panel"] | None = "unit_panel"
    breakouts: tuple[str, ...] = ()

    def __init__(
        self,
        store: ArtifactStore,
        snapshot_context: AbstractContextManager[object],
        snapshot: ArtifactSnapshot,
        manifest: UnitDayArtifactManifest,
        *,
        metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
    ) -> None:
        self._store = store
        self._lifecycle = (
            snapshot_context
            if isinstance(snapshot_context, _SnapshotLifecycle)
            else _SnapshotLifecycle(snapshot_context)
        )
        self._snapshot = snapshot
        self._manifest = manifest
        self._population_units: frozenset[str] | None = None
        self._trigger_anchors: dict[str, dt.datetime] = {}
        self._verified_tables: dict[str, ir.Table] = {}
        self._metrics_arg = metrics
        self._aggregation_by_metric: dict[str, str] = {}
        self._ratio_semantics: dict[str, tuple[str, int | None, str, int | None]] = {}
        self._source_context: SourceContext | None = None
        self.operations: frozenset[SourceOperation] = ARTIFACT_OPERATIONS

    @classmethod
    def open(
        cls,
        store: ArtifactStore,
        ref: UnitDayArtifactRef,
        *,
        expected_context: ArtifactContext,
        metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
        verification: Literal["lazy_digest"] = "lazy_digest",
    ) -> ArtifactMomentSource:
        if verification != "lazy_digest":
            _raise("query.artifact_reader.artifact_moment.verification_lazy_digest")
        context = open_trusted_manifest_snapshot(store, ref, expected_context=expected_context)
        snapshot, manifest = context.__enter__()
        return cls(store, context, snapshot, manifest, metrics=metrics)

    from_artifact = open

    @property
    def context(self) -> SourceContext:
        if self._source_context is None:
            if any(
                extension.kind == "encouragement_uptake" for extension in self._manifest.extensions
            ):
                from increment.query.source import _artifact_source_context

                self._source_context = _artifact_source_context(self._manifest.context)[1]
                return self._source_context
            specs = self._metric_specs()
            from increment.query.source import _artifact_source_context

            experiment, _ = _artifact_source_context(self._manifest.context)
            metrics = tuple(self._trusted_metric(spec) for spec in specs)
            from increment._analysis_config import resolve_configs
            from increment.plan import compile_decision_plan

            plan = compile_decision_plan(None, metrics, path="frame")
            self._source_context = SourceContext(
                study_id=self._manifest.experiment_id,
                design=None,
                plan=plan,
                metrics=metrics,
                configs=resolve_configs(
                    metrics,
                    bindings=None,
                    specs={spec.name: spec for spec in specs},
                    methods=None,
                    prior=None,
                ),
                cluster=None,
                trigger_name=experiment.trigger,
            )
        assert self._source_context is not None
        return self._source_context

    @property
    def manifest(self) -> UnitDayArtifactManifest:
        return self._manifest

    def _ensure(self, role: Literal["exposures", "measure_stats"]) -> ir.Table:
        """Return this role's snapshot-verified relation as a trusted ``ir.Table``.

        ``ArtifactSnapshot.verify_relation`` already re-derives the
        relation's content/schema digest and compares it against the
        pinned ``ArtifactRelationRef`` before returning -- this reader
        never repeats that digest computation over materialised rows; it
        consumes the verified table directly, validating schema column
        names and business-level integrity in SQL. ``_reduce`` filters/
        joins this table with ordinary ibis operations and only ever
        ``.execute()``s the final, already-small reduction output -- not
        this relation itself.
        """
        cached = self._verified_tables.get(role)
        if cached is not None:
            return cached
        ref = getattr(self._manifest.base, role)
        if (
            ref.artifact_id != self._manifest.artifact_id
            or ref.generation_id != self._manifest.generation_id
        ):
            _relation_refuse(
                "artifact.snapshot.mixed",
                f"{role} relation is not bound to manifest generation",
            )
        expected_names = tuple(field[0] for field in UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role])
        self._store.validate_locator(
            ref.relation,
            artifact_id=ref.artifact_id,
            generation_id=ref.generation_id,
            role=role,
        )
        relation = self._snapshot.verify_relation(ref, expected_role=role)
        names = _schema_names(relation)
        if names is not None and names != expected_names:
            _relation_refuse(
                "artifact.relation.schema_mismatch", f"{role} relation schema mismatch"
            )
        if role == "exposures":
            row_count = _validate_exposures_sql(relation, self._manifest, self._snapshot.execute)
        else:
            row_count = _validate_stats_sql(
                relation, self._ensure("exposures"), self._manifest, self._snapshot.execute
            )
        if row_count != ref.row_count:
            _relation_refuse(
                "artifact.relation.digest_mismatch", f"{role} relation row_count mismatch"
            )
        self._verified_tables[role] = relation
        return relation

    def _metric_specs(self) -> list[MetricSpec]:
        if self._metrics_arg is None and not self._manifest.metric_measures:
            _raise("query.artifact_reader.artifact_moment.manifest_no_metric")
        metric_defs = _declared_metric_definitions(self._manifest.context.canonical_json)
        if self._metrics_arg is not None:
            specs = coerce_metrics(self._metrics_arg)
            bindings = {binding.metric_name: binding for binding in self._manifest.metric_measures}
            for spec in specs:
                binding = bindings.get(spec.name)
                if binding is None:
                    refuse(
                        _ARTIFACT_METRIC, message=f"metric {spec.name!r} has no manifest binding"
                    )
                definition = metric_defs.get(spec.name)
                if definition is None:
                    _relation_refuse(
                        "artifact.context.mismatch",
                        f"context declares no metric {spec.name!r} bound by the manifest",
                    )
                resolution = _MetricResolution(
                    spec,
                    binding,
                    definition,
                )
                _validate_metric_identity(resolution)
                _validate_metric_semantics(resolution)
                _record_metric_resolution(self, resolution)
            return specs
        specs = []
        for binding in self._manifest.metric_measures:
            definition = metric_defs.get(binding.metric_name)
            if definition is None:
                _relation_refuse(
                    "artifact.context.mismatch",
                    f"context declares no metric {binding.metric_name!r} bound by the manifest",
                )
            specs.append(_metric_spec_from_binding(binding, definition))
            _record_binding_resolution(self, binding, definition)
        return specs

    @staticmethod
    def _covariate_table(rows: list[dict[str, Any]]) -> Any:
        if rows:
            return ibis.memtable(rows)
        schema = ibis.schema(
            {"experiment_id": "string", "unit_id": "string", "sum_value": "float64"}
        )
        return ibis.memtable(schema.to_pyarrow().empty_table(), schema=schema)

    def _cuped_pre_stats(self, metric: Metric) -> Any:
        """Resolved by the trusted facade (``query.source``), which owns extension access."""
        _relation_refuse(
            "artifact.extension.missing",
            f"artifact source has no CUPED evidence extension for {metric.name!r}",
        )

    def _session_metric_names(self) -> frozenset[str] | None:
        """Metric names this reader session was opened to serve, or `None`
        for every metric bound in the manifest -- mirrors native_source.py's
        `self._metrics` scope used by `_union_event_horizon`."""
        if self._metrics_arg is None:
            return None
        return frozenset(spec.name for spec in coerce_metrics(self._metrics_arg))

    def _declared_observation_end(self) -> dt.date | None:
        """The experiment's declared observation edge day, from the admitted context."""
        stored = json.loads(self._manifest.context.canonical_json)["window_days"]
        horizon = stored["observation_horizon"]
        return None if horizon is None else dt.date.fromisoformat(horizon)

    def _normalize_metric(self, metric: Metric) -> Metric:
        """Restore trusted aggregation and window semantics without mutating the metric."""
        if metric.name not in self._aggregation_by_metric:
            self._metric_specs()
        aggregation = self._aggregation_by_metric.get(metric.name, "count")
        if isinstance(metric, MeanMetric | QuantileMetric):
            metric = metric.model_copy(update={"aggregation": aggregation})
        elif isinstance(metric, RatioMetric):
            semantics = self._ratio_semantics.get(metric.name)
            if semantics is None:
                semantics = (
                    aggregation,
                    metric.numerator.window_days,
                    aggregation,
                    metric.denominator.window_days,
                )
            num_agg, num_window, den_agg, den_window = semantics
            metric = metric.model_copy(
                update={
                    "numerator": metric.numerator.model_copy(
                        update={"aggregation": num_agg, "window_days": num_window}
                    ),
                    "denominator": metric.denominator.model_copy(
                        update={"aggregation": den_agg, "window_days": den_window}
                    ),
                }
            )
        return metric

    def _trusted_metric(self, spec: MetricSpec) -> Metric:
        """Build the real ``Metric`` `_custom_moments`/`_reduce` dispatch on."""
        return self._normalize_metric(synthesise_metric(spec))

    def _reduce(
        self,
        metric: Metric,
        grain: Grain,
        *,
        by: str | None = None,
        properties_table: ir.Table | None = None,
        cluster: str | None = None,
        cluster_table: ir.Table | None = None,
        pre_stats: ir.Table | None = None,
        population_units: frozenset[str] | None = None,
        completed_windows_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = self._reduction_query(
            metric,
            grain,
            by=by,
            properties_table=properties_table,
            cluster=cluster,
            cluster_table=cluster_table,
            pre_stats=pre_stats,
            population_units=population_units,
            completed_windows_only=completed_windows_only,
        )
        return _rows(self._snapshot.execute(query))

    def reduce_triggered(
        self,
        metric: Metric,
        grain: Grain,
        *,
        inputs: _TriggeredReductionInputs,
        by: str | None = None,
        properties_table: ir.Table | None = None,
        cluster: str | None = None,
        cluster_table: ir.Table | None = None,
        pre_stats: ir.Table | None = None,
        population_units: frozenset[str] | None = None,
        completed_windows_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Reduce one metric with trigger-local relations without changing reader state."""
        query = self._reduction_query(
            metric,
            grain,
            by=by,
            properties_table=properties_table,
            cluster=cluster,
            cluster_table=cluster_table,
            pre_stats=pre_stats,
            population_units=population_units,
            completed_windows_only=completed_windows_only,
            trigger_inputs=inputs,
        )
        return _rows(self._snapshot.execute(query))

    def _unit_frame(
        self,
        metric: Metric,
        *,
        cluster: str | None,
        resolve_cluster: Callable[[], ir.Table | None],
        population_units: frozenset[str] | None = None,
        outcome_stage: Literal["transformed", "raw"] = "transformed",
        trigger_inputs: _TriggeredReductionInputs | None = None,
    ) -> IntoDataFrame:
        """Resolve, project, and execute a unit frame on the pinned snapshot."""
        if outcome_stage not in ("raw", "transformed"):
            from increment.winsor import winsor_refuse

            winsor_refuse("invalid_state", "Unknown unit outcome stage.")
        totals = self._reduction_query(
            metric,
            "unit",
            cluster=cluster,
            cluster_table=resolve_cluster(),
            population_units=population_units,
            trigger_inputs=trigger_inputs,
        )
        columns = {name: totals[name] for name in ("unit_id", "group_id", "y")}
        if (
            outcome_stage == "raw"
            and getattr(metric, "winsorization", None) is not None
            and "y_raw" not in totals.columns
        ):
            from increment.winsor import winsor_refuse

            winsor_refuse(
                "raw_state_required", "Artifact reduction did not retain pre-winsor outcomes."
            )
        if outcome_stage == "raw" and "y_raw" in totals.columns:
            columns["y"] = totals.y_raw
        if isinstance(metric, RatioMetric):
            columns["y_den"] = totals.y_den
        if cluster is not None:
            columns["cluster_id"] = totals[cluster].cast("string")
        return self._snapshot.execute(totals.select(**columns))

    def _observation_spine_edge(
        self,
        measure_keys: frozenset[str],
        stats: ir.Table,
        *,
        exposures: ir.Table,
        experiment: Experiment,
        uptake_events: ir.Table | None = None,
    ) -> tuple[dt.date | None, dt.date | None]:
        declared_end = self._declared_observation_end()
        scope_keys = _union_scope_measure_keys(
            self._manifest.metric_measures, self._session_metric_names()
        )
        union_edge = _union_outcome_edge(self._manifest.measures, scope_keys)
        fallback_edge = (
            union_edge
            if union_edge is not None
            else _outcome_edge(
                self._manifest.measures,
                stats,
                measure_keys,
                self._snapshot.execute,
                prefer_persisted_horizon=True,
            )
        )
        snapshot_edges = [
            measure.certified_edge
            if "certified_edge" in measure.model_fields_set
            else measure.observed_edge
            if "observed_edge" in measure.model_fields_set
            else None
            for measure in self._manifest.measures
            if measure.measure_key in measure_keys
        ]
        snapshot_edges = [edge for edge in snapshot_edges if edge is not None]
        metric_edge = min(snapshot_edges) if len(snapshot_edges) == len(measure_keys) else None
        if metric_edge is not None:
            edge = min(declared_end, metric_edge) if declared_end is not None else metric_edge
            return edge, edge
        if declared_end is not None:
            return declared_end, declared_end
        if uptake_events is None:
            return fallback_edge, fallback_edge

        extension = next(
            ext for ext in self._manifest.extensions if ext.kind == "encouragement_uptake"
        )
        compliance_edge: dt.date | None = None
        if extension.observation_edge is not None:
            compliance_edge = extension.observation_edge
        elif "observation_edge" not in extension.model_fields_set:
            # Older artifacts retain a conservative edge from recorded uptake.
            horizon = compliance_event_horizon(exposures, uptake_events, experiment)
            coverage = ibis.literal(1).name("_one").as_table().select(edge=horizon)
            edge = _rows(self._snapshot.execute(coverage))[0]
            compliance_edge = _scalar_date(edge["edge"])
        if fallback_edge is None:
            # Compliance coverage cannot establish outcome observation coverage.
            return None, None
        if compliance_edge is None:
            return fallback_edge, fallback_edge
        return fallback_edge, max(fallback_edge, compliance_edge)

    def _uptake_inputs(self) -> tuple[ir.Table | None, int | None, bool, dt.date | None]:
        design = self.context.design
        if not isinstance(design, Encouragement):
            return None, None, False, None
        relation = self._uptake_relation(design)
        events = relation.filter(relation.uptake).select("unit_id", ts=relation.first_uptake_ts)
        extension = next(
            ext for ext in self._manifest.extensions if ext.kind == "encouragement_uptake"
        )
        return events, design.uptake.window_days, True, extension.certified_edge

    def _reduction_query(  # noqa: PLR0915
        self,
        metric: Metric,
        grain: Grain | Literal["unit"],
        *,
        by: str | None = None,
        properties_table: ir.Table | None = None,
        cluster: str | None = None,
        cluster_table: ir.Table | None = None,
        pre_stats: ir.Table | None = None,
        population_units: frozenset[str] | None = None,
        completed_windows_only: bool = False,
        trigger_inputs: _TriggeredReductionInputs | None = None,
        finalized_as_of: dt.date | None = None,
    ) -> ir.Table:
        """Reduce every artifact source through the same observation spine."""
        if cluster is not None and grain not in ("total", "unit"):
            refuse(
                _ARTIFACT_GRAIN,
                operation="moments",
                request={"grain": grain, "cluster": cluster},
                offered=tuple(sorted(self.capabilities)),
                route="request total- or unit-grain moments",
            )
        metric = self._normalize_metric(metric)
        uptake_events, uptake_window_days, has_uptake, uptake_certified_edge = self._uptake_inputs()
        exposures = (
            self._ensure("exposures") if trigger_inputs is None else trigger_inputs.exposures
        )
        stats = (
            self._ensure("measure_stats")
            if trigger_inputs is None
            else trigger_inputs.measure_stats
        )
        if trigger_inputs is not None:
            finalized_as_of = trigger_inputs.finalized_as_of
        if population_units is not None:
            exposures = restrict_to_units(exposures, population_units)
            stats = restrict_to_units(stats, population_units)
        if cluster is not None and cluster_table is not None:
            exposures = exposures.left_join(cluster_table, "unit_id").select(
                *[exposures[column] for column in exposures.columns],
                **{cluster: cluster_table[cluster]},
            )
        binding = next(
            item for item in self._manifest.metric_measures if item.metric_name == metric.name
        )
        measure_keys = (
            frozenset({binding.numerator_measure_key, binding.denominator_measure_key})
            if isinstance(binding, RatioMetricMeasure)
            else frozenset({binding.measure_key})
        )
        control_row = _rows(
            self._snapshot.execute(exposures.aggregate(control=exposures.group_id.min()))
        )[0]
        control = str(control_row["control"])
        experiment = Experiment(
            name=self._manifest.experiment_id,
            exposure="adopted",
            unit="unit_id",
            start=dt.datetime.combine(self._manifest.first_ds, dt.time()),
            control_group=control,
            day_boundary=self._manifest.day_boundary,
            cluster=cluster,
            # The shared builder requires a positive count to attach pre-period stats.
            n_pre_periods=1 if pre_stats is not None else 0,
            plan=AnalysisPlan(),
        )
        declared_end = self._declared_observation_end()
        if trigger_inputs is not None:
            observed_end = trigger_inputs.observed_edge
            if declared_end is not None:
                observed_end = min(observed_end, declared_end)
            experiment = experiment.model_copy(
                update={"observation_end": dt.datetime.combine(observed_end, dt.time())}
            )
        # Use the declared end or union source horizon for the shared day axis.
        # Keep each metric's maturity watermark separate from that spine boundary.
        metric_edge = (
            trigger_inputs.finalized_as_of
            if trigger_inputs is not None
            else finalized_as_of
            if finalized_as_of is not None
            else _outcome_edge(self._manifest.measures, stats, measure_keys, self._snapshot.execute)
        )
        if trigger_inputs is not None:
            outcome_spine_edge = trigger_inputs.spine_edge
            spine_edge = trigger_inputs.spine_edge
        elif finalized_as_of is not None:
            outcome_spine_edge, spine_edge = finalized_as_of, finalized_as_of
        else:
            outcome_spine_edge, spine_edge = self._observation_spine_edge(
                measure_keys,
                stats,
                exposures=exposures,
                experiment=experiment,
                uptake_events=uptake_events if grain == "asof" else None,
            )
        end_date_expr = _day_edge_scalar(spine_edge)
        spine = panel_spine(exposures, experiment, end_date=end_date_expr)

        def panel_for(measure_key: str) -> Any:
            # A day axis never includes a late enrollee's null-date
            # placeholder row.
            day_spine = spine.filter(spine.ds.notnull())
            measure_stats = stats.filter(stats.measure_key == measure_key)
            joined = day_spine.left_join(
                measure_stats,
                [day_spine.unit_id == measure_stats.unit_id, day_spine.ds == measure_stats.ds],
                rname="{name}_stats",
            )
            return joined.select(
                day_spine.unit_id,
                day_spine.experiment_id,
                day_spine.group_id,
                day_spine.first_exposure_ts,
                day_spine.first_exposure_date,
                day_spine.ds,
                **{
                    column: day_spine[column]
                    for column in ("__uptake_first_exposure_date", "__uptake_first_exposure_ts")
                    if column in day_spine.columns
                },
                n_events=ibis.coalesce(measure_stats.n_events, 0),
                sum_value=ibis.coalesce(measure_stats.sum_value, 0.0),
                min_value=ibis.coalesce(measure_stats.min_value, 0.0),
                max_value=ibis.coalesce(measure_stats.max_value, 0.0),
                metric=ibis.literal(metric.name),
            )

        if isinstance(binding, RatioMetricMeasure):
            num_stats = stats.filter(stats.measure_key == binding.numerator_measure_key)
            den_stats = stats.filter(stats.measure_key == binding.denominator_measure_key)
            panel = panel_for(binding.numerator_measure_key)
            den_panel = panel_for(binding.denominator_measure_key)
        else:
            num_stats = stats.filter(stats.measure_key == binding.measure_key)
            den_stats = None
            panel, den_panel = panel_for(binding.measure_key), None

        uptake_spine = spine
        uptake_exposures = None
        if trigger_inputs is not None and uptake_events is not None:
            uptake_exposures = self._ensure("exposures")
            if cluster is not None and cluster_table is not None:
                uptake_exposures = uptake_exposures.left_join(cluster_table, "unit_id").select(
                    *[uptake_exposures[column] for column in uptake_exposures.columns],
                    **{cluster: cluster_table[cluster]},
                )
            uptake_exposures = uptake_exposures.semi_join(
                exposures.select("experiment_id", "unit_id").distinct(),
                ["experiment_id", "unit_id"],
            )
            uptake_spine = panel_spine(uptake_exposures, experiment, end_date=end_date_expr)
        uptake_panel = self._uptake_panel(uptake_spine, uptake_events)

        by_list = [by] if by else None
        if grain in ("total", "unit"):
            total_experiment = experiment
            if grain == "total" and spine_edge is not None and trigger_inputs is None:
                # Every observed spine ends at this edge; null-day units stay censored.
                # Reuse the bound instead of re-aggregating the dense relation.
                total_experiment = experiment.model_copy(
                    update={
                        "observation_end": dt.datetime.combine(cast(dt.date, spine_edge), dt.time())
                    }
                )
            totals = unit_totals(
                spine,
                num_stats,
                metric,
                total_experiment,
                pre_stats=pre_stats,
                den_stats=den_stats,
                uptake_events=uptake_events,
                uptake_window_days=uptake_window_days,
                by=by_list,
                properties_table=properties_table,
                data_as_of=metric_edge,
                warn_on_censoring=(
                    (lambda query: _rows(self._snapshot.execute(query)))
                    if grain == "unit" and finalized_as_of is None
                    else False
                ),
                uptake_exposures=uptake_exposures,
            )
            totals = winsorize_unit_totals(totals, metric)
            if grain == "unit":
                return totals
            return group_summary(
                totals,
                by=by_list,
                cluster=cluster,
                ratio_metrics=[metric.name] if isinstance(metric, RatioMetric) else None,
                uptake=has_uptake,
                binary_metrics=declared_binary_metrics([metric]),
            )
        if properties_table is not None:
            dimension = [by] if by else []
            panel = join_breakout_dimension(panel, properties_table, dimension)
            if den_panel is not None:
                den_panel = join_breakout_dimension(den_panel, properties_table, dimension)
        if pre_stats is not None:
            panel = join_pre_period_covariate(panel, pre_stats)
        if grain == "daily" and isinstance(metric, RetentionMetric):
            result = cohort_group_summary(
                spine,
                num_stats,
                metric,
                experiment,
                pre_stats=pre_stats,
                by=by_list,
                properties_table=properties_table,
                data_as_of=metric_edge,
            )
        elif grain == "daily":
            result = daily_group_summary(
                window_bound_stats(panel, metric),
                metric=metric,
                by=by_list,
                den_panel=(
                    window_bound_stats(den_panel, metric) if den_panel is not None else None
                ),
            )
        else:
            if (
                completed_windows_only
                and (isinstance(metric, RatioMetric) or uptake_panel is not None)
                and _final_maturity_day(metric) is not None
            ):
                panel = _censor_to_observable_window(
                    panel, metric, experiment, metric_edge, warn_on_censoring=False
                )
            result = asof_group_summary(
                panel,
                metric,
                by=by_list,
                den_panel=den_panel,
                completed_windows_only=completed_windows_only,
                uptake_panel=uptake_panel,
                uptake_window_days=uptake_window_days,
                _uptake_elapsed_windowed=uptake_panel is not None,
                _uptake_certified_edge=uptake_certified_edge,
                _uptake_day_boundary_offset=experiment.day_boundary_offset,
                _outcome_observation_end=(
                    _day_edge_scalar(outcome_spine_edge)
                    if uptake_panel is not None and outcome_spine_edge is not None
                    else None
                ),
            )
        return result

    def _uptake_panel(self, spine: ir.Table, events: ir.Table | None) -> ir.Table | None:
        if events is None:
            return None
        offset = day_boundary_offset(self._manifest.day_boundary)
        events_by_day = events.mutate(ds=_local_date_at_offset(events.ts, offset))
        day_spine = spine.filter(spine.ds.notnull())
        joined = day_spine.left_join(events_by_day, ["unit_id", "ds"], rname="{name}_uptake")
        return joined.select(
            *[day_spine[name] for name in day_spine.columns],
            sum_value=events_by_day.ts.notnull().cast("int32").cast("float64"),
            n_events=events_by_day.ts.notnull().cast("int32").cast("int64"),
        )

    def _builder_total(self, metric: Metric) -> list[dict[str, Any]]:
        return self._reduce(
            metric,
            "total",
            cluster=self.context.cluster,
            cluster_table=self._cluster_table(),
        )

    def _cluster_table(self) -> ir.Table | None:
        cluster = self.context.cluster
        if cluster is None:
            return None
        matches = [
            ext
            for ext in self._manifest.extensions
            if ext.kind == "cluster_identity" and ext.cluster_name == cluster
        ]
        if len(matches) != 1:
            invalid_compliance_state("clustered artifact requires cluster identity evidence")
        relation = self._snapshot.verify_relation(
            matches[0].relation, expected_role="cluster_identity"
        )
        return relation.select("unit_id", **{cluster: relation.cluster_id})

    def _builder_day(
        self, metric: Metric, grain: Grain, *, completed_windows_only: bool = False
    ) -> list[dict[str, Any]]:
        return self._reduce(metric, grain, completed_windows_only=completed_windows_only)

    def _custom_moments(
        self, metric: Metric, grain: Grain, *, completed_windows_only: bool = False
    ) -> list[dict[str, Any]]:
        if grain == "total":
            return self._builder_total(metric)
        return self._builder_day(metric, grain, completed_windows_only=completed_windows_only)

    def validated_metric(self, metric: Metric) -> Metric:
        """The manifest-bound metric for *metric*'s name, refusing a caller copy whose
        semantics differ. Reads no evidence, so estimators can authenticate before refusing."""
        trusted = self._metric_specs()
        trusted_spec = next((spec for spec in trusted if spec.name == metric.name), None)
        if trusted_spec is None:
            refuse(
                _ARTIFACT_METRIC,
                message=f"metric {metric.name!r} has no trusted manifest binding",
            )
        trusted_metric = self._trusted_metric(trusted_spec)
        semantic_fields = (
            "aggregation",
            "window_days",
            "threshold_days",
            "quantile",
            "winsorization",
        )
        mismatch = type(metric) is not type(trusted_metric) or any(
            getattr(metric, field, None) != getattr(trusted_metric, field, None)
            for field in semantic_fields
        )
        if isinstance(metric, RatioMetric) and isinstance(trusted_metric, RatioMetric):
            mismatch = mismatch or any(
                getattr(metric_part, field, None) != getattr(trusted_part, field, None)
                for metric_part, trusted_part in (
                    (metric.numerator, trusted_metric.numerator),
                    (metric.denominator, trusted_metric.denominator),
                )
                for field in ("aggregation", "window_days")
            )
        if mismatch:
            refuse(
                _ARTIFACT_METRIC,
                message=f"metric {metric.name!r} differs from trusted manifest semantics",
            )
        return trusted_metric

    def moments(
        self,
        metric: Metric,
        *,
        grain: Grain = "total",
        by: Sequence[str] = (),
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> list[dict[str, Any]]:
        if grain not in self.capabilities:
            refuse(
                _ARTIFACT_GRAIN,
                operation="moments",
                request={"grain": grain},
                offered=tuple(sorted(self.capabilities)),
                route="request one of the offered grains",
            )
        metric = self.validated_metric(metric)
        if by:
            _relation_refuse(
                "artifact.extension.missing", "artifact base has no breakout evidence extension"
            )
        if include_covariate:
            _relation_refuse(
                "artifact.extension.missing", "artifact base has no CUPED evidence extension"
            )
        if (
            grain == "asof"
            and completed_windows_only
            and isinstance(metric, RetentionMetric)
            and metric.band[1] is None
        ):
            _relation_refuse(
                "artifact.evidence.unavailable",
                "completed-window filtering is unavailable for unbounded retention",
            )
        if grain == "daily" and isinstance(metric, RetentionMetric) and metric.band[1] is None:
            _relation_refuse(
                "artifact.evidence.unavailable",
                "daily evidence is unavailable for unbounded retention",
            )
        return self._custom_moments(metric, grain, completed_windows_only=completed_windows_only)

    def _sequential_observation_mapping(self):
        from increment.query.artifact_contract import artifact_source_mapping

        return artifact_source_mapping(self._manifest.context)

    def _certify_sequential_extensions(self, registration, as_of: dt.date) -> None:
        """Require every triggered feed to be certified through the captured edge."""
        from increment.query.sequential_capture import certify_capture_feed
        from increment.query.source import _artifact_source_context
        from increment.semantics.artifact import (
            TriggerMeasureStatsExtension,
            TriggerPopulationExtension,
        )
        from increment.semantics.design import Encouragement
        from increment.sequential_state import sequential_refuse

        experiment, _ = _artifact_source_context(self._manifest.context)
        self._artifact_experiment = experiment

        trigger_name = self.context.trigger_name
        if trigger_name is None:
            sequential_refuse("source.invalid", "triggered capture requires a declared trigger")
        outcomes = {model.metric for model in registration.models if model.observable == "outcome"}
        extensions = cast("Sequence[Any]", self._manifest.extensions)
        requests: list[tuple[str, str | None]] = [("trigger_population", None)]
        requests.extend(("trigger_measure_stats", name) for name in sorted(outcomes))
        for kind, metric_name in requests:
            if kind == "trigger_population":
                extension = cast(
                    "TriggerPopulationExtension | TriggerMeasureStatsExtension | None",
                    next(
                        (
                            item
                            for item in extensions
                            if isinstance(item, TriggerPopulationExtension)
                            and item.trigger_name == trigger_name
                        ),
                        None,
                    ),
                )
            else:
                if metric_name is None:
                    sequential_refuse("source.invalid", "trigger measure feed needs a metric")
                extension = cast(
                    "TriggerPopulationExtension | TriggerMeasureStatsExtension | None",
                    next(
                        (
                            item
                            for item in extensions
                            if isinstance(item, TriggerMeasureStatsExtension)
                            and item.trigger_name == trigger_name
                            and metric_name in item.metric_names
                        ),
                        None,
                    ),
                )
            if extension is None:
                sequential_refuse(
                    "source.invalid",
                    f"artifact is missing the {kind} feed for triggered capture",
                    feed=kind if metric_name is None else f"{kind}:{metric_name}",
                )
            certify_capture_feed(
                feed=kind if metric_name is None else f"{kind}:{metric_name}",
                cutoff=extension.observation_cutoff_ts,
                complete_through=extension.complete_through_ts,
                experiment=self._artifact_experiment,
                as_of=as_of,
            )
        edges = {
            metric.name: self.triggered_observation_edges(metric)[1]
            for metric in self.context.metrics
            if metric.name in outcomes
        }
        if any(model.observable == "uptake" for model in registration.models):
            design = self.context.design
            if not isinstance(design, Encouragement):
                sequential_refuse("source.invalid", "uptake requires encouragement assignment")
            edges[f"encouragement_uptake:{design.uptake.fact}"] = self._uptake_inputs()[3]
        for feed, edge in edges.items():
            if edge is None or edge < as_of:
                sequential_refuse(
                    "source.invalid",
                    f"feed {feed!r} is not certified complete through {as_of}",
                    feed=feed,
                    certified_day=None if edge is None else edge.isoformat(),
                    as_of=as_of.isoformat(),
                )

    def capture_sequential(self, *, finalized: bool, as_of: dt.date, previous=None):
        """Capture a finalized common cohort from the pinned immutable generation."""
        from dataclasses import replace

        from increment.query.builders import _local_date
        from increment.query.sequential_capture import (
            capture_relations,
            validate_relational_capture,
        )
        from increment.sequential_state import model_adjustment, sequential_refuse

        previous = previous if previous is not None else getattr(self, "_sequential_snapshot", None)
        # A registered adjustment reads the metric's published cuped_preperiod
        # extension, the same per-unit total fixed-horizon CUPED reads here.
        declared = getattr(self.context.plan.inference, "registration", None)
        adjusted = {
            model.metric
            for model in (declared.models if declared is not None else ())
            if model.observable == "outcome" and model_adjustment(model) is not None
        }
        published = {
            extension.metric_name
            for extension in self._manifest.extensions
            if extension.kind == "cuped_preperiod"
        }
        covariate = bool(adjusted) and adjusted <= published
        registration, _ = validate_relational_capture(
            self, finalized=finalized, as_of=as_of, previous=previous, covariate=covariate
        )
        # Trigger extension access is owned by the trusted facade (``query.source``),
        # the only reader that can capture the triggered chain.
        facade = cast("Any", self)
        monitor_triggered = self.context.trigger_name is not None and (
            previous is None or previous.triggered is not None
        )
        if monitor_triggered:
            facade._certify_sequential_extensions(registration, as_of)
        exposures = self._ensure("exposures")
        experiment = Experiment(
            name=self.context.study_id,
            exposure="adopted",
            unit="unit_id",
            start=dt.datetime.combine(self._manifest.first_ds, dt.time()),
            control_group=registration.control_group,
            day_boundary=self._manifest.day_boundary,
            plan=AnalysisPlan(),
        )
        window = ibis.interval(days=max(0, registration.reveal.longest_window_days - 1))

        def final_day(table):
            return _local_date(table.first_exposure_ts, experiment) + window

        cohort = exposures.filter(final_day(exposures) <= ibis.literal(as_of))

        def uptake():
            design = self.context.design
            if not isinstance(design, Encouragement):
                sequential_refuse("source.invalid", "uptake requires encouragement assignment")
            relation = self._uptake_relation(design)
            joined = cohort.left_join(relation, "unit_id")
            return joined.select(
                cohort.unit_id,
                cohort.group_id,
                d=relation.uptake.fill_null(False).cast("int32").cast("int64"),
            )

        metrics = {metric.name: metric for metric in self.context.metrics}
        triggered = None
        triggered_registration = getattr(
            self.context.plan.inference, "triggered_registration", None
        )
        if monitor_triggered and triggered_registration is not None:
            trigger_source = facade.triggered_source()
            inputs = trigger_source._triggered_reduction_inputs(
                metrics[triggered_registration.models[0].metric]
            )
            trigger_cohort = inputs.exposures.filter(
                final_day(inputs.exposures) <= ibis.literal(as_of)
            )

            def trigger_outcome(model):
                metric = metrics[model.metric]
                metric_inputs = replace(
                    trigger_source._triggered_reduction_inputs(metric),
                    finalized_as_of=as_of,
                    spine_edge=ibis.literal(as_of, type="date"),
                    exposures=trigger_cohort,
                )
                return trigger_source._reduction_query(
                    metric,
                    "unit",
                    trigger_inputs=metric_inputs,
                    pre_stats=self._cuped_pre_stats(metric) if metric.name in adjusted else None,
                )

            triggered = capture_relations(
                self,
                trigger_outcome,
                trigger_cohort,
                self._snapshot.batches,
                recipe_id=self._manifest.context.sha256,
                finalized=finalized,
                as_of=as_of,
                previous=None if previous is None else previous.triggered,
                covariate=covariate,
                population="triggered",
                assignment_counts=facade.triggered_counts()[1],
                final_day=final_day,
            )

        def relation_for(model):
            if model.observable == "uptake":
                return uptake()
            metric = metrics[model.metric]
            return self._reduction_query(
                metric,
                "unit",
                pre_stats=self._cuped_pre_stats(metric) if metric.name in adjusted else None,
                finalized_as_of=as_of,
            )

        return capture_relations(
            self,
            relation_for,
            cohort,
            self._snapshot.batches,
            recipe_id=self._manifest.context.sha256,
            finalized=finalized,
            as_of=as_of,
            previous=previous,
            covariate=covariate,
            assignment_counts=self.assignment_counts(population="assigned"),
            triggered=triggered,
            final_day=final_day if monitor_triggered else None,
        )

    def unit_frame(
        self,
        metric: Metric,
        *,
        covariates: Sequence[str] = (),
        outcome_stage: Literal["transformed", "raw"] = "transformed",
    ) -> IntoDataFrame:
        """Per-unit rows (``unit_id``, ``group_id``, ``y`` and ``y_den`` for a
        ratio) from the same ``unit_totals`` reduction ``moments(grain="total")``
        runs, executed on the pinned snapshot -- the one deliberate unit-level
        materialization this reader performs. The transformed stage covers
        unwindowed mean/ratio/conversion/quantile metrics; a windowed or
        retention metric refuses by name (compute each unit's windowed value
        upstream and declare it as an unwindowed metric on ``from_unit_summary``;
        no frame source serves unit-grain estimators for a retention metric).
        This reader carries no covariate evidence, so
        any requested covariate refuses.
        """
        if outcome_stage not in ("raw", "transformed"):
            from increment.winsor import winsor_refuse

            winsor_refuse("invalid_state", "Unknown unit outcome stage.")
        if covariates:
            _relation_refuse(
                "artifact.extension.missing", "artifact base has no covariate evidence"
            )
        trusted = self.validated_metric(metric)
        if outcome_stage == "transformed":
            spec = next(spec for spec in self._metric_specs() if spec.name == trusted.name)
            if spec.window_days is not None or spec.type == "retention":
                refuse(
                    FRAME_UNIT_FRAME_PANEL,
                    metric=metric.name,
                    shape="an artifact of a windowed or retention metric",
                    route=(
                        "For a windowed metric, compute each unit's windowed value upstream "
                        "and declare it as an unwindowed metric on from_unit_summary; no "
                        "frame source serves unit-grain estimators for a retention metric."
                    ),
                )
        return self._unit_frame(
            trusted,
            cluster=None,
            resolve_cluster=lambda: None,
            outcome_stage=outcome_stage,
        )

    def unit_counts(self) -> dict[str, int]:
        """Per-group enrolled unit counts -- a bounded `group_by` aggregate
        over `exposures`, never a caller of the per-unit reduction
        `unit_frame()` executes."""
        exposures = self._ensure("exposures")
        counts = exposures.group_by("group_id").agg(n=exposures.count())
        return {
            str(row["group_id"]): int(row["n"])
            for row in _rows(self._snapshot.execute(counts))
            if row["group_id"] is not None
        }

    def cluster_counts(self) -> dict[str, int]:
        """This reader carries no cluster extension -- only the facade's
        `cluster_identity` extension can answer a cluster-grain count."""
        if isinstance(self.context.design, Encouragement) and self.context.cluster is not None:
            return {
                arm.group_id: arm.n_clusters
                for arm in self.compliance_summary(self.context.design).arms
                if arm.n_clusters is not None
            }
        refuse(
            _ARTIFACT_GRAIN,
            operation="cluster_counts",
            request={"population": "assigned"},
            offered=tuple(sorted(self.capabilities)),
            route="use a source built with a declared cluster column",
        )

    def _moments_for_metrics(
        # One adapter signature covers all source routes.
        self,
        methods=None,
        selected=None,
        prior=None,
        population="assigned",
        narrow_cuped=False,  # noqa: FBT002
    ):
        import pyarrow as pa

        selected = tuple(selected or self.context.metrics)
        rows = [row for metric in selected for row in self.moments(metric)]
        return pa.Table.from_pylist(rows) if rows else pa.table({})

    def day_source(
        self,
        *,
        metrics: Sequence[Metric],
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> ArtifactMomentSource:
        if population == "triggered" and getattr(self, "_population", "assigned") != "triggered":
            factory = getattr(self, "triggered_source", None)
            if factory is None:
                _relation_refuse(
                    "artifact.extension.missing", "artifact has no triggered population evidence"
                )
            return cast(
                "ArtifactMomentSource",
                factory().day_source(metrics=metrics, population="triggered"),
            )
        return self

    def triggered_observation_edges(self, metric: Metric) -> tuple[dt.date, dt.date | None]:
        """Return cutoff-derived trigger spine edge and optional certified edge."""
        experiment = getattr(self, "_artifact_experiment", None)
        if experiment is None or self.context.trigger_name is None:
            _relation_refuse(
                "artifact.extension.missing", "artifact has no triggered observation evidence"
            )
        extension = next(
            (
                item
                for item in self._manifest.extensions
                if item.kind == "trigger_measure_stats"
                and item.trigger_name == self.context.trigger_name
                and metric.name in item.metric_names
            ),
            None,
        )
        if extension is None:
            _relation_refuse(
                "artifact.extension.missing",
                f"artifact has no triggered observation evidence for {metric.name!r}",
            )
        binding = next(
            (item for item in self._manifest.metric_measures if item.metric_name == metric.name),
            None,
        )
        if binding is not None:
            measure_keys = (
                (binding.numerator_measure_key, binding.denominator_measure_key)
                if isinstance(binding, RatioMetricMeasure)
                else (binding.measure_key,)
            )
            by_key = {item.measure_key: item for item in self._manifest.measures}
            measures = [by_key[key] for key in measure_keys if key in by_key]
            if len(measures) == len(measure_keys) and all(
                {"certified_edge", "observed_edge"} & item.model_fields_set for item in measures
            ):
                edges = [_measure_edge(item, prefer_persisted_horizon=False) for item in measures]
                if all(edge is not None for edge in edges):
                    observed_edge = min(cast(dt.date, edge) for edge in edges)
                    certified_edge = (
                        observed_edge
                        if all(
                            "certified_edge" in item.model_fields_set
                            and item.certified_edge is not None
                            for item in measures
                        )
                        else None
                    )
                    horizon = experiment.observation_horizon_day
                    if horizon is not None:
                        observed_edge = min(observed_edge, horizon)
                        if certified_edge is not None:
                            certified_edge = min(certified_edge, horizon)
                    return observed_edge, certified_edge
        offset = day_boundary_offset(experiment.day_boundary)
        cutoff = extension.observation_cutoff_ts
        complete_through = extension.complete_through_ts
        if complete_through is None:
            observed_edge = (cutoff + offset).date()
            certified_edge = None
        else:
            certified_edge = (min(cutoff, complete_through) + offset).date() - dt.timedelta(days=1)
            observed_edge = certified_edge
        horizon = experiment.observation_horizon_day
        if horizon is not None:
            observed_edge = min(observed_edge, horizon)
            if certified_edge is not None:
                certified_edge = min(certified_edge, horizon)
        return observed_edge, certified_edge

    def breakout_moments(
        self,
        metric: Metric,
        breakout: Any,
        *,
        grain: Grain = "daily",
        completed_windows_only: bool = False,
        include_covariate: bool = False,
    ) -> Any:
        _relation_refuse(
            "artifact.extension.missing", "artifact has no breakout evidence extension"
        )

    def moments_source(
        self,
        *,
        metrics: Sequence[Metric],
        methods: list[Any] | None = None,
        prior: Any | None = None,
        population: Literal["assigned", "triggered"] = "assigned",
        narrow_cuped: bool = False,
    ) -> ArtifactMomentSource:
        if population != "assigned":
            _relation_refuse(
                "artifact.evidence.unavailable", "artifact base does not carry triggered evidence"
            )
        return self

    def assignment_counts(
        self, *, population: Literal["assigned", "triggered"] = "assigned"
    ) -> dict[str, int]:
        if population != "assigned":
            _relation_refuse(
                "artifact.evidence.unavailable",
                "artifact base does not carry triggered assignment counts",
            )
        return self.unit_counts()

    def _uptake_relation(self, design: Encouragement) -> ir.Table:
        """Verify current uptake state before any readout reduction."""
        extensions = [
            ext for ext in self._manifest.extensions if ext.kind == "encouragement_uptake"
        ]
        if not extensions or (len(extensions) == 1 and extensions[0].extension_version == 1):
            raise_legacy_compliance_state(
                study_id=self._manifest.experiment_id,
                reason="artifact lacks timestamp-bearing version-2 encouragement uptake state",
            )
        if len(extensions) != 1:
            invalid_compliance_state("artifact must select exactly one uptake extension")
        extension = extensions[0]
        from increment.query.artifact_contract import unit_day_artifact_extension_catalog

        entries = [
            entry
            for entry in unit_day_artifact_extension_catalog(self._manifest.context)
            if entry.request.kind == "encouragement_uptake"
        ]
        if (
            len(entries) != 1
            or extension.definition_sha256 != entries[0].definition_sha256
            or extension.source_provenance_sha256 != entries[0].source_provenance_sha256
        ):
            invalid_compliance_state("uptake evidence differs from the trusted catalog")
        identity = ComplianceSummary(
            study_id=self._manifest.experiment_id,
            cohort=extension.uptake_name,
            window_days=extension.window_days,
            one_sided=extension.one_sided,
            control_group=str(design.control_group),
            cluster=self.context.cluster,
            as_of=None,
            arms=(),
        )
        validate_compliance_design_match(identity, design)
        if self.context.design is not None and self.context.design != design:
            invalid_compliance_state("requested design differs from artifact design")
        cached = self._verified_tables.get("encouragement_uptake")
        if cached is not None:
            return cached
        from increment.query.schemas import ARTIFACT_RELATION_SCHEMAS

        relation = self._snapshot.verify_relation(
            extension.relation, expected_role="encouragement_uptake"
        )
        expected = tuple(field[0] for field in ARTIFACT_RELATION_SCHEMAS["encouragement_uptake"])
        if _schema_names(relation) != expected:
            invalid_compliance_state("uptake relation has an obsolete or invalid schema")
        exposures = self._ensure("exposures")
        joined = relation.left_join(
            exposures, ["experiment_id", "unit_id"], rname="{name}_exposure"
        )
        invalid_time = joined.first_uptake_ts < joined.first_exposure_ts
        if design.uptake.window_days is not None:
            invalid_time = invalid_time | (
                joined.first_uptake_ts
                >= joined.first_exposure_ts + ibis.interval(days=design.uptake.window_days)
            )
        checks = joined.aggregate(
            n=joined.count(),
            invalid=(
                joined.first_exposure_ts.isnull()
                | joined.uptake.isnull()
                | (joined.uptake != joined.first_uptake_ts.notnull())
                | invalid_time.fill_null(False)
            ).any(),
        )
        distinct = relation.select("experiment_id", "unit_id").distinct()
        counts = distinct.aggregate(unique=distinct.count()).cross_join(
            exposures.aggregate(enrolled=exposures.count())
        )
        row = _rows(self._snapshot.execute(checks.cross_join(counts)))[0]
        if (
            row["invalid"]
            or row["n"] != row["unique"]
            or row["n"] != row["enrolled"]
            or row["n"] != extension.relation.row_count
        ):
            invalid_compliance_state(
                "uptake relation must cover every enrolled unit exactly once with a valid timestamp/flag"
            )
        self._verified_tables["encouragement_uptake"] = relation
        return relation

    def compliance_dates(self) -> Sequence[object]:
        """Enrollment dates, or a dense trigger-entry history through uptake."""
        design = cast(Encouragement, self.context.design)
        exposures = self._ensure("exposures")
        uptake = self._uptake_relation(design)
        extension = next(
            ext for ext in self._manifest.extensions if ext.kind == "encouragement_uptake"
        )
        experiment = Experiment(
            name=self._manifest.experiment_id,
            exposure="adopted",
            unit="unit_id",
            start=dt.datetime.combine(self._manifest.first_ds, dt.time()),
            control_group=str(getattr(design, "control_group", "control")),
            day_boundary=self._manifest.day_boundary,
            plan=AnalysisPlan(),
        )
        if self._population_units is not None:
            offset = day_boundary_offset(self._manifest.day_boundary)
            trigger_days = sorted(
                {
                    (timestamp.astimezone(dt.UTC) + offset).date()
                    if timestamp.tzinfo is not None
                    else (timestamp + offset).date()
                    for timestamp in self._trigger_anchors.values()
                }
            )
            if not trigger_days:
                return []
            declared_edge = self._declared_observation_end()
            edge = extension.observation_edge
            if edge is None and "observation_edge" not in extension.model_fields_set:
                horizon = compliance_event_horizon(
                    exposures, uptake.select("unit_id", ts=uptake.first_uptake_ts), experiment
                )
                coverage = ibis.literal(1).name("_one").as_table().select(edge=horizon)
                edge = _scalar_date(_rows(self._snapshot.execute(coverage))[0]["edge"])
            trigger_extension = next(
                (
                    item
                    for item in self._manifest.extensions
                    if item.kind == "trigger_population"
                    and item.trigger_name == self.context.trigger_name
                ),
                None,
            )
            limits = [
                candidate
                for candidate in (declared_edge, extension.certified_edge)
                if candidate is not None
            ]
            if trigger_extension is not None:
                cutoff_edge, _ = _effective_snapshot_edge(
                    trigger_extension.observation_cutoff_ts,
                    None,
                    offset,
                    declared_edge,
                )
                limits.append(cutoff_edge)
            last = max(candidate for candidate in (edge, trigger_days[-1]) if candidate is not None)
            if limits:
                last = min(last, *limits)
            if last < trigger_days[0]:
                return []
            return [
                trigger_days[0] + dt.timedelta(days=offset)
                for offset in range((last - trigger_days[0]).days + 1)
            ]
        edge = self._declared_observation_end() or extension.observation_edge
        if edge is not None:
            end_date = ibis.literal(edge)
        elif "observation_edge" in extension.model_fields_set:
            end_date = ibis.null().cast("date")
        else:
            # Older timestamp-bearing artifacts retain a conservative coverage lower bound.
            end_date = compliance_event_horizon(
                exposures, uptake.select("unit_id", ts=uptake.first_uptake_ts), experiment
            )
        spine = panel_spine(exposures, experiment, end_date=cast("ir.Scalar", end_date))
        days = spine.filter(spine.ds.notnull()).select("ds").distinct().order_by("ds")
        return [row["ds"] for row in _rows(self._snapshot.execute(days))]

    def compliance_summary(
        self,
        design: Encouragement,
        *,
        as_of: object | None = None,
        completed_windows_only: bool = False,
    ) -> ComplianceSummary:
        if completed_windows_only:
            from increment._source_types import validate_compliance_completion

            validate_compliance_completion(design, as_of=as_of)
        uptake = self._uptake_relation(design)
        exposures = self._ensure("exposures")
        if self._population_units is not None:
            exposures = restrict_to_units(exposures, self._population_units)
        if as_of is not None:
            offset = day_boundary_offset(self._manifest.day_boundary)
            as_of_date = cast(dt.date, as_of)
            if self._population_units is not None:
                trigger_units = {
                    unit_id
                    for unit_id, timestamp in self._trigger_anchors.items()
                    if (
                        timestamp.astimezone(dt.UTC) + offset
                        if timestamp.tzinfo is not None
                        else timestamp + offset
                    ).date()
                    <= as_of_date
                }
                exposures = restrict_to_units(exposures, frozenset(trigger_units))
            else:
                exposures = exposures.filter(exposures.first_exposure_date <= ibis.literal(as_of))
            uptake = uptake.filter(
                _local_date_at_offset(uptake.first_uptake_ts, offset) <= ibis.literal(as_of)
            )
        if completed_windows_only:
            uptake_completion_ts = exposures.first_exposure_ts + ibis.interval(
                days=design.uptake.window_days
            )
            exposures = exposures.filter(
                exposures.first_exposure_date + ibis.interval(days=design.uptake.window_days)
                <= ibis.literal(as_of)
            )
            extension = next(
                ext for ext in self._manifest.extensions if ext.kind == "encouragement_uptake"
            )
            if extension.certified_edge is not None:
                certified_edge_exclusive = (
                    dt.datetime.combine(extension.certified_edge + dt.timedelta(days=1), dt.time())
                    - day_boundary_offset(self._manifest.day_boundary)
                ).replace(tzinfo=dt.UTC)
                exposures = exposures.filter(
                    uptake_completion_ts
                    <= _utc_timestamp_literal(exposures.first_exposure_ts, certified_edge_exclusive)
                )
        cluster = self.context.cluster
        if cluster is not None:
            from increment.query.schemas import ARTIFACT_RELATION_SCHEMAS

            matches = [
                ext
                for ext in self._manifest.extensions
                if ext.kind == "cluster_identity" and ext.cluster_name == cluster
            ]
            if len(matches) != 1:
                invalid_compliance_state("clustered compliance requires cluster identity evidence")
            identities = self._snapshot.verify_relation(
                matches[0].relation, expected_role="cluster_identity"
            )
            if _schema_names(identities) != tuple(
                field[0] for field in ARTIFACT_RELATION_SCHEMAS["cluster_identity"]
            ):
                invalid_compliance_state("invalid cluster identity relation")
            joined = exposures.left_join(identities, ["experiment_id", "unit_id"])
            exposures = joined.select(
                *[exposures[name] for name in exposures.columns], **{cluster: identities.cluster_id}
            )
            checks = exposures.aggregate(
                n=exposures.count(), missing=exposures[cluster].isnull().any()
            )
            distinct = exposures.select("unit_id").distinct()
            row = _rows(
                self._snapshot.execute(
                    checks.cross_join(distinct.aggregate(unique=distinct.count()))
                )
            )[0]
            purity = exposures.group_by(cluster).aggregate(arms=exposures.group_id.nunique())
            mixed = _rows(self._snapshot.execute(purity.aggregate(mixed=(purity.arms > 1).any())))[
                0
            ]["mixed"]
            if row["missing"] or row["n"] != row["unique"] or mixed:
                invalid_compliance_state("cluster identities must be complete, unique and arm-pure")
        joined = exposures.left_join(uptake, ["experiment_id", "unit_id"])
        totals = joined.select(
            *[exposures[name] for name in exposures.columns],
            d=uptake.uptake.fill_null(False).cast("int32").cast("float64"),
        )
        base = _rows(
            self._snapshot.execute(
                totals.group_by("group_id").aggregate(
                    n_units=totals.count(), uptake_total=totals.d.sum()
                )
            )
        )
        rows: dict[str, dict[str, Any]] = {
            str(row["group_id"]): dict(row, group_id=str(row["group_id"])) for row in base
        }
        if cluster is not None:
            totals = totals.mutate(y=totals.d, metric=ibis.literal("uptake"))
            clustered = _rows(
                self._snapshot.execute(group_summary(totals, cluster=cluster, uptake=True))
            )
            for row in clustered:
                rows[str(row["group_id"])].update(
                    {name: row[field] for name, field in COMPLIANCE_ARM_FROM_CLUSTER_ROW.items()}
                )
        return ComplianceSummary(
            study_id=self._manifest.experiment_id,
            cohort=design.uptake.fact,
            window_days=design.uptake.window_days,
            one_sided=design.one_sided,
            control_group=str(design.control_group),
            cluster=cluster,
            as_of=as_of,
            arms=tuple(ComplianceArm(**row) for row in rows.values()),
        )

    def _refuse_bound_observational_quantile(self, metric: Metric) -> NoReturn:
        """Authenticate the metric against the artifact binding, then refuse it observationally."""
        refuse_observational_quantile(metric, source=self)

    def export_moments(self, path: str | Path) -> None:
        if getattr(self.context.plan.inference, "registration", None) is not None:
            from increment.sources import export_source_moments

            return export_source_moments(
                self, path, observational_refusal=self._refuse_bound_observational_quantile
            )
        if self.context.cluster is not None and not isinstance(self.context.design, Encouragement):
            from increment.sources import refuse_cluster_grain_transport

            refuse_cluster_grain_transport(
                operation="export_moments",
                source="artifact",
                cluster=self.context.cluster,
                design=self.context.design,
                route_forward=(
                    "analyse the clustered source directly; direct clustered inference "
                    "preserves cluster-grain degrees of freedom and counts"
                ),
            )
        from increment.sources import refuse_quantile_moments_export

        refuse_quantile_moments_export(
            self.context.metrics,
            design=self.context.design,
            observational_refusal=self._refuse_bound_observational_quantile,
        )
        if not self._manifest.metric_measures:
            from increment.sources import export_source_moments

            return export_source_moments(
                self, path, observational_refusal=self._refuse_bound_observational_quantile
            )
        import pyarrow as pa
        import pyarrow.parquet as pq

        from increment._source_identity import source_identity
        from increment.decision_wire import compiled_plan_to_json
        from increment.sources import (
            DECISION_PLAN_FIELD,
            MOMENTS_FORMAT,
            SOURCE_IDENTITY_FIELD,
            _validate_moment_counts,
        )

        compliance = (
            self.compliance_summary(self.context.design)
            if isinstance(self.context.design, Encouragement)
            else None
        )
        table = self._moments_for_metrics()
        for row in table.to_pylist():
            _validate_moment_counts(row)
        if "successes" in table.column_names:
            index = table.schema.get_field_index("successes")
            table = table.set_column(index, "successes", table["successes"].cast(pa.int64()))
        table = table.append_column(
            "moments_format", pa.array([MOMENTS_FORMAT] * table.num_rows, type=pa.int64())
        )
        plan_payload = compiled_plan_to_json(self.context.plan)
        table = table.append_column(
            DECISION_PLAN_FIELD,
            pa.array([plan_payload] * table.num_rows, type=pa.string()),
        )
        identity_payload = json.dumps(source_identity(self), sort_keys=True, separators=(",", ":"))
        table = table.append_column(
            SOURCE_IDENTITY_FIELD,
            pa.array([identity_payload] * table.num_rows, type=pa.string()),
        )
        from increment.sources import ASSIGNMENT_COUNTS_FIELD, COMPLIANCE_SUMMARY_FIELD

        counts = None
        if any(ext.kind == "assignment_counts" for ext in self._manifest.extensions):
            counts = self.unit_counts()
        elif compliance is not None:
            counts = {arm.group_id: arm.n_units for arm in compliance.arms}
        if counts is not None:
            counts_payload = json.dumps(counts, sort_keys=True, separators=(",", ":"))
            table = table.append_column(
                ASSIGNMENT_COUNTS_FIELD,
                pa.array([counts_payload] * table.num_rows, type=pa.string()),
            )
        if compliance is not None:
            payload = json.dumps(compliance.to_wire(), sort_keys=True, separators=(",", ":"))
            table = table.append_column(
                COMPLIANCE_SUMMARY_FIELD, pa.array([payload] * table.num_rows, type=pa.string())
            )
        table = table.replace_schema_metadata(
            {b"increment.moments_format": str(MOMENTS_FORMAT).encode()}
        )
        pq.write_table(table, str(path))

    def sql(self, *, grain: Grain = "total") -> dict[str, str]:
        refuse(
            _ARTIFACT_GRAIN,
            operation="sql",
            request={"grain": grain},
            offered=tuple(sorted(self.capabilities)),
            route="use moments() for artifact-backed reductions",
        )

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` released the snapshot this source shares with its views."""
        return self._lifecycle.closed

    def close(self) -> None:
        self._lifecycle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def open_artifact(
    store: ArtifactStore,
    ref: UnitDayArtifactRef,
    *,
    expected_context: ArtifactContext,
    metrics: Sequence[MetricSpec] | Mapping[str, str] | None = None,
    verification: Literal["lazy_digest"] = "lazy_digest",
) -> ArtifactMomentSource:
    """Open through the trusted facade shared with ``query.source``."""
    from increment.query.source import open_artifact as _open_trusted_artifact

    return _open_trusted_artifact(
        store,
        ref,
        expected_context=expected_context,
        metrics=metrics,
        verification=verification,
    )


__all__ = ["ARTIFACT_OPERATIONS", "ArtifactMomentSource", "open_artifact"]
