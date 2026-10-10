"""Unit-day artifact publication: assembled and written from a
`DefinitionsMomentSource`'s bound accessors, kept out of that class."""

from __future__ import annotations

import datetime as dt
import json
import warnings
from contextlib import ExitStack, closing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import ibis

from increment.errors import refuse as _refuse
from increment.query.artifact_contract import (
    REFUSALS as _ARTIFACT_REFUSALS,
)
from increment.query.artifact_contract import (
    ArtifactContractError,
    _measure_recipe_key,
    _request_key,
    compile_unit_day_artifact_context,
    open_trusted_snapshot,
    unit_day_artifact_extension_catalog,
)
from increment.query.artifact_digest import _DIGEST_BATCH_ROWS, canonical_json, manifest_sha256
from increment.query.builders import (
    _local_date,
    _utc_timestamp_literal,
    canonical_dimension_value,
    compliance_event_horizon,
    metric_events,
    post_exposure_stats,
    pre_period_stats,
    resolved_measure_key,
    union_event_horizon,
    unit_day_stats,
    validate_unit_day_aggregate_rows,
)
from increment.query.fact_resolution import _find_fact_source, _resolve_value_column
from increment.query.schemas import ARTIFACT_RELATION_SCHEMAS
from increment.semantics.artifact import (
    BaseRelations,
    Freshness,
    MeasureManifest,
    RatioMetricMeasure,
    SimpleMetricMeasure,
    TriggerMeasureStatsExtension,
    TriggerPopulationExtension,
    UnitDayArtifactManifest,
)
from increment.semantics.models import Definitions, RatioMetric
from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import ibis.expr.types as ir

    from increment.query.artifact_contract import ArtifactContext, ArtifactStore
    from increment.semantics.artifact import UnitDayArtifactRef
    from increment.semantics.models import Experiment, Metric


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _relation_table(rows: list[dict[str, object]], role: str) -> ir.Table:
    if rows:
        return ibis.memtable(rows)
    type_names = {
        "STRING": "string",
        "INT64": "int64",
        "FLOAT64": "float64",
        "DATE": "date",
        "TIMESTAMP_UTC_US": "timestamp('UTC')",
        "BOOLEAN": "boolean",
    }
    schema = ibis.schema(
        {
            field: type_names[type_tag]
            for field, type_tag, _nullable in ARTIFACT_RELATION_SCHEMAS[role]
        }
    )
    # ibis<=10.x drops every declared column when a memtable is built from
    # an empty Python list -- `schema` is accepted at expression build time
    # but ignored once materialised (ibis-project/ibis#10940, fixed in
    # ibis 11). An already-typed empty pyarrow table keeps the columns.
    return ibis.memtable(schema.to_pyarrow().empty_table(), schema=schema)


@dataclass(frozen=True, slots=True)
class _PublicationExtensionSpec:
    entry: Any
    kind: str
    rows: list[dict[str, object]] | ir.Table
    fields: dict[str, object]


def _effective_snapshot_edge(
    cutoff: dt.datetime,
    complete_through: dt.datetime | None,
    offset: dt.timedelta,
    declared_horizon: dt.date | None,
) -> tuple[dt.date, dt.date | None]:
    if complete_through is None:
        edge = (cutoff + offset).date()
        certified_edge = None
    else:
        certified_edge = (min(cutoff, complete_through) + offset).date() - dt.timedelta(days=1)
        edge = certified_edge
    if declared_horizon is not None:
        edge = min(edge, declared_horizon)
        if certified_edge is not None:
            certified_edge = min(certified_edge, declared_horizon)
    return edge, certified_edge


def _record_snapshot_edges(
    evidence: Any,
    feed: str,
    offset: dt.timedelta,
    declared_horizon: dt.date | None,
    manifest_keys: Sequence[str],
    certified_edges: dict[str, dt.date],
    observed_edges: dict[str, dt.date],
) -> None:
    if evidence is None:
        return
    edge, certified_edge = _effective_snapshot_edge(
        evidence.observation_cutoff_ts,
        evidence.complete_through_by_feed.get(feed),
        offset,
        declared_horizon,
    )
    for key in manifest_keys:
        if certified_edge is None:
            observed_edges[key] = edge
        else:
            certified_edges[key] = certified_edge


@dataclass(frozen=True, slots=True)
class _PublicationAssembly:
    exposures: ir.Table
    exposure_rows: list[dict[str, object]]
    stats: ir.Table
    freshness: dict[str, dt.date]
    event_horizons: dict[str, dt.date | None]
    certified_edges: dict[str, dt.date]
    observed_edges: dict[str, dt.date]
    measure_specs: dict[str, tuple[Any, Metric, str | None, str, tuple[Any, ...]]]
    metric_measures: tuple[Any, ...]
    first_ds: dt.date
    last_ds: dt.date


def _assemble_publication_manifest(
    *,
    artifact_id: Any,
    generation_id: Any,
    experiment_id: str,
    day_boundary: Any,
    assembly: _PublicationAssembly,
    exposure_ref: Any,
    measure_ref: Any,
    context: Any,
    extension_refs: list[Any],
) -> UnitDayArtifactManifest:
    measures = tuple(
        MeasureManifest(
            measure_key=key,
            source_provenance_sha256=__import__("hashlib")
            .sha256(canonical_json(ref.model_dump(mode="python")).encode())
            .hexdigest(),
            freshness=Freshness(
                loaded_through=assembly.freshness.get(key, assembly.last_ds),
                declared_complete=False,
            ),
            event_horizon=assembly.event_horizons.get(key),
            **(
                {"certified_edge": assembly.certified_edges[key]}
                if key in assembly.certified_edges
                else {}
            ),
            **(
                {"observed_edge": assembly.observed_edges[key]}
                if key in assembly.observed_edges
                else {}
            ),
        )
        for key, (ref, _metric, _value_col, _part, _physical) in sorted(
            assembly.measure_specs.items(), key=lambda item: item[0].encode("utf-8")
        )
    )
    prototype = UnitDayArtifactManifest.model_construct(
        artifact_id=artifact_id,
        generation_id=generation_id,
        experiment_id=experiment_id,
        created_at=_utc_now(),
        day_boundary=day_boundary,
        first_ds=assembly.first_ds,
        last_ds=assembly.last_ds,
        base=BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
        measures=measures,
        metric_measures=assembly.metric_measures,
        context=context,
        extensions=tuple(extension_refs),
        manifest_sha256="",
    )
    manifest_body = prototype.model_dump(mode="python")
    manifest_body["manifest_sha256"] = manifest_sha256(prototype)
    return UnitDayArtifactManifest.model_validate(manifest_body)


def artifact_context(
    definitions: Definitions,
    experiment: Experiment,
    policy: Literal["error", "warn", "exclude"],
    *,
    metrics: Sequence[Metric] | None = None,
    encouragement_uptake: Any | None = None,
    store: ArtifactStore | None = None,
    refresh_of: UnitDayArtifactRef | None = None,
) -> ArtifactContext:
    """Unit-day artifact context: `DefinitionsMomentSource.publish_unit_day_artifact`'s
    and `Analysis.publish_unit_day_artifact`'s shared compilation, timestamp-
    normalization retry, and (when *store*/*refresh_of* are both given) the
    refresh-context check. Replaces `Analysis._artifact_context` and
    `DefinitionsMomentSource._publication_context`."""
    effective_metrics = (
        list(metrics)
        if metrics is not None
        else [
            metric
            for name in experiment.metric_names
            if (metric := definitions.metric(name)) is not None
        ]
    )
    # Site volume is read over [start, end], the window its rows are filtered to.
    site_last = experiment.end_day or experiment.start_day
    coverage = (
        {
            "first_ds": experiment.start_day,
            "last_ds": site_last,
            "freshness": {"loaded_through": site_last, "declared_complete": True},
        }
        if effective_metrics
        and experiment.end is not None
        and all(
            getattr(metric, "type", None) not in {"retention", "quantile"}
            for metric in effective_metrics
        )
        else None
    )
    context = compile_unit_day_artifact_context(
        experiment.name,
        definitions,
        on_mixed_assignment=policy,
        site_volume_coverage=coverage,
        encouragement_uptake=encouragement_uptake,
    )
    if store is not None and refresh_of is not None:
        with open_trusted_snapshot(store, refresh_of, expected_context=context) as snapshot:
            prior = snapshot.read_manifest(
                refresh_of.manifest, expected_sha256=refresh_of.manifest_sha256
            )
            if prior.experiment_id != experiment.name or prior.context != context:
                raise ArtifactContractError(
                    "artifact.refresh.context_mismatch",
                    "fixed prior artifact belongs to another experiment or context",
                    context={"experiment_id": experiment.name},
                )
    return context


class ArtifactPublisher:
    """Owns unit-day artifact assembly and publication; constructed from a
    `DefinitionsMomentSource`'s bound private accessors so every `self.*`
    the moved methods use resolves unchanged."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        experiment: Experiment,
        definitions: Definitions,
        connection: Any,
        metrics: Sequence[Metric],
        on_mixed_assignment: Literal["error", "warn", "exclude"],
        no_data_signal: Any,
        get_exposures: Callable[[], ir.Table],
        get_fact_table: Callable[[Any], ir.Table],
        build_breakout_properties_table: Callable[..., ir.Table],
        counts_for_population: Callable[[Any], tuple[Any, Any, Any]],
        get_trigger_population: Callable[[], ir.Table],
        validate_cluster_labels: Callable[[ir.Table, str], None],
        validate_mixed_assignments: Callable[[], None],
        data_as_of: Callable[[ir.Table, str], Any],
        source_snapshot_evidence: Any,
    ) -> None:
        self._experiment = experiment
        self._defs = definitions
        self._con = connection
        self._metrics = metrics
        self._on_mixed_assignment = on_mixed_assignment
        self._NO_DATA_SIGNAL = no_data_signal
        self._get_exposures = get_exposures
        self._get_fact_table = get_fact_table
        self._build_breakout_properties_table = build_breakout_properties_table
        self._counts_for_population = counts_for_population
        self._get_trigger_population = get_trigger_population
        self._validate_cluster_labels = validate_cluster_labels
        self._validate_mixed_assignments = validate_mixed_assignments
        self._data_as_of = data_as_of
        self._source_snapshot_evidence = source_snapshot_evidence
        # These expressions read only the source snapshot owned by publish.
        self._event_cache: dict[tuple[Any, ...], ir.Table] = {}
        self._uptake_cache: dict[str, ir.Table] = {}

    def publish(
        self,
        store: ArtifactStore,
        *,
        extensions: Sequence[Any] = (),
        source: Any,
        refresh_of: UnitDayArtifactRef | None = None,
    ) -> UnitDayArtifactRef:
        """Publish an immutable unit-day artifact; failures are not atomic.
        Verify visible manifests and backing relations or drop the generation before retrying.
        ``source`` must be the live DefinitionsMomentSource for extension protocol checks.
        """
        context = artifact_context(
            self._defs,
            self._experiment,
            self._on_mixed_assignment,
            metrics=self._metrics,
            encouragement_uptake=(
                source.context.design
                if getattr(source.context.design, "mechanism", None) == "encouragement"
                else None
            ),
            store=store,
            refresh_of=refresh_of,
        )
        catalog = {entry.request for entry in unit_day_artifact_extension_catalog(context)}
        for extension in extensions:
            request = getattr(extension, "request", extension)
            if request not in catalog:
                kind = str(getattr(request, "kind", "unknown"))
                reason = "request is absent from the trusted extension catalog"
                _refuse(
                    _ARTIFACT_REFUSALS["artifact.extension.invalid"],
                    kind=kind,
                    reason=reason,
                    message=f"artifact extension {kind!r} is invalid: {reason}",
                )
        requested = [getattr(extension, "request", extension) for extension in extensions]
        # Every observational covariate is fully determined by
        # `design.covariates`; the artifact can't serve `unit_frame`'s
        # covariate join later without it, so it is never optional.
        for entry in unit_day_artifact_extension_catalog(context):
            if (
                entry.request.kind in {"unit_covariate", "unit_covariate_level"}
                and entry.request not in requested
            ):
                requested.append(entry.request)
        if getattr(source.context.design, "mechanism", None) == "encouragement":
            for entry in unit_day_artifact_extension_catalog(context):
                if (
                    entry.request.kind
                    in {"encouragement_uptake", "assignment_counts", "cluster_identity"}
                    and entry.request not in requested
                ):
                    requested.append(entry.request)
        selected_trigger_names = {
            request.trigger_name for request in requested if request.kind == "trigger_measure_stats"
        }
        for entry in unit_day_artifact_extension_catalog(context):
            if (
                entry.request.kind == "trigger_population"
                and entry.request.trigger_name in selected_trigger_names
                and entry.request not in requested
            ):
                requested.append(entry.request)
        if any(
            request.kind in {"trigger_population", "trigger_measure_stats"}
            or (request.kind == "assignment_counts" and "triggered" in request.populations)
            for request in requested
        ):
            source._require_source_snapshot_evidence(operation="publish_unit_day_artifact")
        site_volume_metrics = frozenset(
            name
            for request in requested
            if request.kind == "site_volume"
            for name in request.metric_names
        )
        with (
            source._pinned_source_execution(
                metrics=self._metrics,
                extra_sources=tuple(
                    request.source_name
                    for request in requested
                    if getattr(request, "source_name", None)
                ),
                uptake_facts=tuple(
                    request.uptake_name
                    for request in requested
                    if request.kind == "encouragement_uptake"
                ),
                site_volume_metrics=site_volume_metrics,
            ) as pinned,
            ExitStack() as materializations,
        ):
            publisher = pinned._publisher
            assembly = publisher._assemble_publication(
                context,
                materializations,
                site_volume_metrics=site_volume_metrics,
                uptake_facts=[
                    request.uptake_name
                    for request in requested
                    if request.kind == "encouragement_uptake"
                ],
            )
            extension_specs = publisher._assemble_extension_specs(
                context, requested, assembly.exposures, assembly.exposure_rows
            )
            return publisher._publish_artifact(
                store, context, assembly, extension_specs, pinned, refresh_of=refresh_of
            )

    def _validate_stats_batches(self, stats: ir.Table) -> dt.date | None:
        last_ds = None
        arrow_schema = stats.schema().to_pyarrow()
        with (
            warnings.catch_warnings(),
            closing(
                self._con.to_pyarrow_batches(
                    stats.order_by(["unit_id", "ds"]), chunk_size=_DIGEST_BATCH_ROWS
                )
            ) as batches,
        ):
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_record_batch\(\) is deprecated.*",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            for batch in batches:
                for start in range(0, batch.num_rows, _DIGEST_BATCH_ROWS):
                    rows = batch.slice(start, _DIGEST_BATCH_ROWS).cast(arrow_schema).to_pylist()
                    validate_unit_day_aggregate_rows(rows)
                    for row in rows:
                        if last_ds is None or row["ds"] > last_ds:
                            last_ds = row["ds"]
                    del rows
        return last_ds

    def _relevant_events(
        self, events: ir.Table, exposures: ir.Table, *, site_volume: bool = False
    ) -> ir.Table:
        """Keep enrolled post/pre-period events and requested whole-site volume."""
        anchors = exposures.select("unit_id", "first_exposure_ts", "first_exposure_date")
        joined = events.left_join(anchors, "unit_id")
        enrolled = events.ts >= anchors.first_exposure_ts
        if self._experiment.n_pre_periods:
            pre_start = anchors.first_exposure_date - ibis.interval(
                days=self._experiment.n_pre_periods
            )
            enrolled |= _local_date(events.ts, self._experiment) >= pre_start
        relevant = anchors.unit_id.notnull() & enrolled
        if site_volume:
            local_day = _local_date(events.ts, self._experiment)
            site = local_day >= ibis.literal(self._experiment.start_day)
            if self._experiment.end is not None:
                site &= local_day <= ibis.literal(self._experiment.end_day)
            relevant |= site
        return joined.filter(relevant).select(events)

    def _measure_event_input(
        self,
        fact_table: ir.Table,
        ref: Any,
        metric: Metric,
        value_col: str | None,
        part: str,
        exposures: ir.Table,
        *,
        site_volume: bool,
    ) -> ir.Table:
        events = metric_events(
            fact_table,
            metric,
            value_column=value_col,
            part=cast(Literal["numerator", "denominator"], part),
        )
        evidence = self._source_snapshot_evidence
        if evidence is not None:
            events = events.filter(
                events.ts <= _utc_timestamp_literal(events.ts, evidence.observation_cutoff_ts)
            )
        relevant = self._relevant_events(events, exposures, site_volume=site_volume)
        # Coverage scalars share the event materialization's execution and snapshot.
        metadata = (
            ibis.literal(1)
            .name("_one")
            .as_table()
            .select(
                unit_id=ibis.null().cast(events.unit_id.type()),
                ts=ibis.null().cast(events.ts.type()),
                metric=ibis.null().cast("string"),
                value=ibis.null().cast("float64"),
                _metadata=ibis.literal(True),
                _watermark=self._data_as_of(fact_table, ref.fact),
                _horizon=union_event_horizon([events], self._experiment),
            )
        )
        return relevant.mutate(
            _metadata=ibis.literal(False),
            _watermark=ibis.null().cast("date"),
            _horizon=ibis.null().cast("date"),
        ).union(metadata)

    def _prepare_event_inputs(
        self,
        physical_specs: dict[tuple[Any, ...], tuple[Any, Metric, str | None, str]],
        exposures: ir.Table,
        *,
        uptake_facts: Sequence[str],
        site_volume_keys: set[tuple[Any, ...]],
    ) -> None:
        """Derive narrow event streams from the already captured source snapshot."""
        for physical_key, (ref, metric, value_col, part) in physical_specs.items():
            fs, _ = _find_fact_source(self._defs, ref.fact)
            self._event_cache[physical_key] = self._measure_event_input(
                self._get_fact_table(fs),
                ref,
                metric,
                value_col,
                part,
                exposures,
                site_volume=physical_key in site_volume_keys,
            )
        for fact in uptake_facts:
            fs, _ = _find_fact_source(self._defs, fact)
            table = self._get_fact_table(fs)
            events = table.filter(table.event == fact).select("unit_id", "ts")
            evidence = self._source_snapshot_evidence
            if evidence is not None:
                events = events.filter(
                    events.ts <= _utc_timestamp_literal(events.ts, evidence.observation_cutoff_ts)
                )
                edge, _ = _effective_snapshot_edge(
                    evidence.observation_cutoff_ts,
                    evidence.complete_through_by_feed.get(fs.name),
                    self._experiment.day_boundary_offset,
                    self._experiment.observation_horizon_day,
                )
                events = events.filter(
                    _local_date(events.ts, self._experiment) <= ibis.literal(edge)
                )
            self._uptake_cache[fact] = self._relevant_events(events, exposures)

    def _coverage_date(self, value: Any) -> dt.date:
        """Use the shared eventless sentinel for absent coverage, never legacy null."""
        if isinstance(value, dt.datetime):
            return value.date()
        return value if isinstance(value, dt.date) else self._NO_DATA_SIGNAL

    def _assemble_publication(
        self,
        context: Any,
        materializations: ExitStack,
        *,
        uptake_facts: Sequence[str] = (),
        site_volume_metrics: set[str] | None = None,
    ) -> _PublicationAssembly:
        self._validate_mixed_assignments()
        exposures = self._get_exposures().mutate(
            first_exposure_date=_local_date(
                self._get_exposures().first_exposure_ts, self._experiment
            )
        )
        if self._experiment.cluster is not None:
            self._validate_cluster_labels(exposures, self._experiment.cluster)
        exposures = materializations.enter_context(self._con._cached_table(exposures))
        if self._experiment.cluster is not None:
            self._validate_cluster_labels(exposures, self._experiment.cluster)
        exposure_rows = [
            {
                "experiment_id": str(row["experiment_id"]),
                "unit_id": str(row["unit_id"]),
                "group_id": str(row["group_id"]),
                "first_exposure_ts": (
                    row["first_exposure_ts"].replace(tzinfo=dt.UTC)
                    if isinstance(row["first_exposure_ts"], dt.datetime)
                    and row["first_exposure_ts"].tzinfo is None
                    else row["first_exposure_ts"]
                ),
                "first_exposure_date": row["first_exposure_date"],
            }
            for row in self._con.to_pyarrow(exposures).to_pylist()
        ]
        measure_specs: dict[str, tuple[Any, Metric, str | None, str, tuple[Any, ...]]] = {}
        physical_specs: dict[tuple[Any, ...], tuple[Any, Metric, str | None, str]] = {}
        site_volume_keys: set[tuple[Any, ...]] = set()

        def add_measure(ref: Any, metric: Metric, *, part: str = "numerator") -> None:
            fs, fact_def = _find_fact_source(self._defs, ref.fact)
            value_col = _resolve_value_column(fs, fact_def)
            key = _measure_recipe_key(ref)
            physical_key = resolved_measure_key(ref, value_column=value_col)
            if metric.name in (site_volume_metrics or set()):
                site_volume_keys.add(physical_key)
            measure_specs.setdefault(key, (ref, metric, value_col, part, physical_key))
            physical_specs.setdefault(physical_key, (ref, metric, value_col, part))

        for metric in self._metrics:
            if isinstance(metric, RatioMetric):
                add_measure(metric.numerator, metric)
                add_measure(metric.denominator, metric, part="denominator")
            else:
                add_measure(metric, metric)
        manifest_keys_by_physical: dict[tuple[Any, ...], list[str]] = {}
        for key, spec in measure_specs.items():
            manifest_keys_by_physical.setdefault(spec[4], []).append(key)
        self._prepare_event_inputs(
            physical_specs,
            exposures,
            uptake_facts=uptake_facts,
            site_volume_keys=site_volume_keys,
        )
        stats_tables: list[ir.Table] = []
        last_stat_ds: dt.date | None = None
        freshness: dict[str, dt.date] = {}
        event_horizons: dict[str, dt.date | None] = {}
        certified_edges: dict[str, dt.date] = {}
        observed_edges: dict[str, dt.date] = {}
        for physical_key in sorted(physical_specs, key=repr):
            ref, metric, value_col, part = physical_specs[physical_key]
            pinned = self._event_cache[physical_key]
            events = pinned.filter(~pinned._metadata).select("unit_id", "ts", "metric", "value")
            stats = post_exposure_stats(
                events,
                exposures,
                source_key=_measure_recipe_key(ref),
                experiment=self._experiment,
            )
            # Validation, coverage, and publication must read the same execution.
            stats = materializations.enter_context(self._con._cached_table(stats))
            stats_last_ds = self._validate_stats_batches(stats)
            if stats_last_ds is not None:
                last_stat_ds = max(last_stat_ds or stats_last_ds, stats_last_ds)
            coverage = self._con.to_pyarrow(
                pinned.filter(pinned._metadata).select("_watermark", "_horizon")
            ).to_pylist()[0]
            watermark = self._coverage_date(coverage["_watermark"])
            horizon = self._coverage_date(coverage["_horizon"])
            _record_snapshot_edges(
                self._source_snapshot_evidence,
                _find_fact_source(self._defs, ref.fact)[0].name,
                self._experiment.day_boundary_offset,
                self._experiment.observation_horizon_day,
                manifest_keys_by_physical[physical_key],
                certified_edges,
                observed_edges,
            )
            for manifest_key in manifest_keys_by_physical[physical_key]:
                freshness[manifest_key] = watermark
                event_horizons[manifest_key] = horizon
                stats_tables.append(
                    stats.select(
                        experiment_id=ibis.literal(self._experiment.name),
                        unit_id=stats.unit_id.cast("string"),
                        ds=stats.ds,
                        measure_key=ibis.literal(manifest_key),
                        n_events=stats.n_events.cast("int64"),
                        sum_value=stats.sum_value.cast("float64"),
                        min_value=stats.min_value.cast("float64"),
                        max_value=stats.max_value.cast("float64"),
                    )
                )
        first_ds = min((row["first_exposure_date"] for row in exposure_rows), default=dt.date.max)
        last_ds = max(
            (cast("dt.date", row["first_exposure_date"]) for row in exposure_rows),
            default=cast("dt.date", first_ds),
        )
        if last_stat_ds is not None:
            last_ds = max(last_ds, last_stat_ds)
        stats_table = (
            ibis.union(*stats_tables, distinct=False)
            if stats_tables
            else _relation_table([], "measure_stats")
        )
        if isinstance(first_ds, dt.datetime):
            first_ds = first_ds.date()
        if isinstance(last_ds, dt.datetime):
            last_ds = last_ds.date()
        first_ds = cast(dt.date, first_ds)
        metric_measures = tuple(
            (
                RatioMetricMeasure(
                    metric_name=metric.name,
                    numerator_measure_key=_measure_recipe_key(metric.numerator),
                    denominator_measure_key=_measure_recipe_key(metric.denominator),
                    allow_same_measure=False,
                )
                if isinstance(metric, RatioMetric)
                else SimpleMetricMeasure(
                    metric_name=metric.name,
                    measure_key=_measure_recipe_key(metric),
                )
            )
            for metric in sorted(self._metrics, key=lambda value: value.name.encode("utf-8"))
        )
        return _PublicationAssembly(
            exposures=exposures,
            exposure_rows=exposure_rows,
            stats=stats_table,
            freshness=freshness,
            event_horizons=event_horizons,
            certified_edges=certified_edges,
            observed_edges=observed_edges,
            measure_specs=measure_specs,
            metric_measures=metric_measures,
            first_ds=first_ds,
            last_ds=last_ds,
        )

    def _assemble_extension_specs(
        self,
        context: Any,
        extensions: Sequence[Any],
        exposures: ir.Table,
        exposure_rows: list[dict[str, object]],
    ) -> tuple[_PublicationExtensionSpec, ...]:
        catalog_entries = {
            entry.request: entry for entry in unit_day_artifact_extension_catalog(context)
        }
        specs: list[_PublicationExtensionSpec] = []
        for requested in extensions:
            request = getattr(requested, "request", requested)
            entry = catalog_entries.get(request)
            if entry is None:
                kind = str(getattr(request, "kind", "unknown"))
                reason = "request is absent from the trusted extension catalog"
                _refuse(
                    _ARTIFACT_REFUSALS["artifact.extension.invalid"],
                    kind=kind,
                    reason=reason,
                    message=f"artifact extension {kind!r} is invalid: {reason}",
                )
            specs.append(self._extension_spec(entry, request, exposures, exposure_rows))
        specs.sort(key=lambda spec: _request_key(spec.entry.request))
        return tuple(specs)

    def _extension_spec(
        self,
        entry: Any,
        request: Any,
        exposures: ir.Table,
        exposure_rows: list[dict[str, object]],
    ) -> _PublicationExtensionSpec:
        kind = cast(str, request.kind)
        match kind:
            case "breakout_dimension" | "factor_dimension":
                return self._dimension_extension_spec(entry, request, exposures, kind)
            case "cuped_preperiod":
                return self._cuped_extension_spec(entry, request, exposures, exposure_rows)
            case "assignment_counts":
                return self._assignment_extension_spec(entry, request)
            case "trigger_population":
                return self._trigger_extension_spec(entry, request)
            case "trigger_measure_stats":
                return self._trigger_measure_extension_spec(entry, request)
            case "cluster_identity":
                return self._cluster_extension_spec(entry, request, exposures)
            case "encouragement_uptake":
                return self._uptake_extension_spec(entry, request, exposures)
            case "site_volume":
                return self._site_volume_extension_spec(entry, request)
            case "unit_covariate" | "unit_covariate_level":
                return self._covariate_extension_spec(entry, request, exposures, exposure_rows)
            case _:
                raise ArtifactContractError(
                    "artifact.extension.invalid",
                    f"extension kind {kind!r} has no definitions producer",
                )

    def _dimension_extension_spec(
        self,
        entry: Any,
        request: Any,
        exposures: ir.Table,
        kind: str,
    ) -> _PublicationExtensionSpec:
        source = next(
            source for source in self._defs.fact_sources if source.name == request.source_name
        )
        properties = self._build_breakout_properties_table(source, request.property_name, exposures)
        joined = exposures.select("unit_id").left_join(
            properties, "unit_id", rname="{name}_dimension"
        )
        properties = joined.select(
            "unit_id",
            canonical_dimension_value(joined[request.property_name]).name(request.property_name),
        )
        value_field = "factor_value" if kind == "factor_dimension" else "dimension_value"
        rows: list[dict[str, object]] = [
            {
                "experiment_id": self._experiment.name,
                "unit_id": str(row["unit_id"]),
                "value_is_missing": row.get(request.property_name) is None,
                value_field: (
                    row[request.property_name] if row.get(request.property_name) is not None else ""
                ),
            }
            for row in self._con.to_pyarrow(properties).to_pylist()
        ]
        identity_field = "dimension_name" if kind == "breakout_dimension" else "factor_name"
        return _PublicationExtensionSpec(
            entry,
            kind,
            rows,
            {identity_field: request.property_name, "source_name": request.source_name},
        )

    def _cuped_extension_spec(
        self,
        entry: Any,
        request: Any,
        exposures: ir.Table,
        exposure_rows: list[dict[str, object]],
    ) -> _PublicationExtensionSpec:
        metric = next(metric for metric in self._metrics if metric.name == request.metric_name)
        # A ratio metric's covariate is its numerator's pre-period total, the
        # same one covariate the definitions path feeds both component slopes.
        measure = metric.numerator if isinstance(metric, RatioMetric) else metric
        fs, fact_def = _find_fact_source(self._defs, measure.fact)
        value_col = _resolve_value_column(fs, fact_def)
        pinned = self._event_cache[resolved_measure_key(measure, value_column=value_col)]
        pre_events = pinned.filter(~pinned._metadata).select("unit_id", "ts", "metric", "value")
        if not self._experiment.n_pre_periods:
            kind = request.kind
            reason = "metric has no pre-period evidence"
            _refuse(
                _ARTIFACT_REFUSALS["artifact.extension.invalid"],
                kind=kind,
                reason=reason,
                message=f"artifact extension {kind!r} is invalid: {reason}",
            )
        pre = pre_period_stats(
            pre_events, exposures, self._experiment, source_key=f"{metric.name}:pre"
        )
        values_by_unit: dict[str, float] = {}
        for row in self._con.to_pyarrow(pre).to_pylist():
            unit_id = str(row["unit_id"])
            values_by_unit[unit_id] = values_by_unit.get(unit_id, 0.0) + float(
                row.get("x", row.get("sum_value", 0.0))
            )
        for exposure in exposure_rows:
            values_by_unit.setdefault(str(exposure["unit_id"]), 0.0)
        rows: list[dict[str, object]] = [
            {"experiment_id": self._experiment.name, "unit_id": unit_id, "x": value}
            for unit_id, value in values_by_unit.items()
        ]
        return _PublicationExtensionSpec(entry, request.kind, rows, {})

    def _covariate_extension_spec(
        self,
        entry: Any,
        request: Any,
        exposures: ir.Table,
        exposure_rows: list[dict[str, object]],
    ) -> _PublicationExtensionSpec:
        """Persist one declared covariate for every enrolled unit, NULL where
        the unit has no pre-exposure value: a numeric property's `value` on
        the `unit_covariate` relation, a string property's `level` on the
        `unit_covariate_level` relation."""
        source = next(
            source for source in self._defs.fact_sources if source.name == request.source_name
        )
        properties = self._build_breakout_properties_table(source, request.property_name, exposures)
        values = {
            str(row["unit_id"]): row[request.property_name]
            for row in self._con.to_pyarrow(properties).to_pylist()
        }
        field, encode = (
            ("level", str) if request.kind == "unit_covariate_level" else ("value", float)
        )
        rows: list[dict[str, object]] = []
        for exposure in exposure_rows:
            value = values.get(str(exposure["unit_id"]))
            rows.append(
                {
                    "experiment_id": self._experiment.name,
                    "unit_id": str(exposure["unit_id"]),
                    field: None if value is None else encode(value),
                }
            )
        return _PublicationExtensionSpec(entry, request.kind, rows, {})

    def _assignment_extension_spec(self, entry: Any, request: Any) -> _PublicationExtensionSpec:
        rows: list[dict[str, object]] = []
        for population in request.populations:
            _grain, counts, units = self._counts_for_population(population)
            for group_id, count in sorted(counts.items()):
                if group_id in (MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL):
                    continue
                rows.append(
                    {
                        "experiment_id": self._experiment.name,
                        "population": population,
                        "group_id": str(group_id),
                        "n_units": int(units.get(group_id, count)),
                        "n_randomization_units": int(count),
                    }
                )
        return _PublicationExtensionSpec(entry, request.kind, rows, {})

    def _trigger_extension_spec(self, entry: Any, request: Any) -> _PublicationExtensionSpec:
        rows: list[dict[str, object]] = []
        population = self._get_trigger_population()
        for row in self._con.to_pyarrow(population).to_pylist():
            timestamp = row["first_trigger_ts"]
            timestamp = (
                timestamp.replace(tzinfo=dt.UTC)
                if timestamp.tzinfo is None
                else timestamp.astimezone(dt.UTC)
            )
            rows.append(
                {
                    "experiment_id": self._experiment.name,
                    "unit_id": str(row["unit_id"]),
                    "first_trigger_ts": timestamp,
                }
            )
        evidence = self._source_snapshot_evidence
        trigger = next(
            exposure for exposure in self._defs.exposures if exposure.name == request.trigger_name
        )
        if trigger.fact is None:
            complete_through_ts = None
        else:
            trigger_feed, _ = _find_fact_source(self._defs, trigger.fact)
            complete_through_ts = evidence.complete_through_by_feed.get(trigger_feed.name)
        return _PublicationExtensionSpec(
            entry,
            request.kind,
            rows,
            {
                "observation_cutoff_ts": evidence.observation_cutoff_ts,
                "complete_through_ts": complete_through_ts,
            },
        )

    def _trigger_measure_extension_spec(
        self, entry: Any, request: Any
    ) -> _PublicationExtensionSpec:
        metric = next(metric for metric in self._metrics if metric.name == request.metric_names[0])
        anchors = self._get_trigger_population().select("unit_id", "first_trigger_ts")
        parts = (
            (
                (metric.numerator, "numerator"),
                (metric.denominator, "denominator"),
            )
            if isinstance(metric, RatioMetric)
            else ((metric, "numerator"),)
        )
        rows: list[dict[str, object]] = []
        feed_names: list[str] = []
        for measure, part in parts:
            fact_source, fact = _find_fact_source(self._defs, measure.fact)
            fact_table = self._get_fact_table(fact_source)
            value_col = _resolve_value_column(fact_source, fact)
            events = metric_events(
                fact_table,
                metric,
                value_column=value_col,
                part=part,
            )
            joined = events.join(anchors, events.unit_id == anchors.unit_id).filter(
                events.ts > anchors.first_trigger_ts
            )
            local_day = _local_date(joined.ts, self._experiment)
            trigger_day = _local_date(anchors.first_trigger_ts, self._experiment)
            joined = joined.filter(local_day >= trigger_day)
            window_days = getattr(measure, "window_days", None)
            if window_days is not None:
                joined = joined.filter(local_day < trigger_day + ibis.interval(days=window_days))
            observation_horizon = self._experiment.observation_horizon_day
            if observation_horizon is not None:
                joined = joined.filter(local_day <= ibis.literal(observation_horizon))
            measure_key = _measure_recipe_key(measure)
            joined = joined.filter(
                events.ts
                <= _utc_timestamp_literal(
                    events.ts, self._source_snapshot_evidence.observation_cutoff_ts
                )
            )
            daily = joined.group_by(joined.unit_id, local_day.name("ds")).agg(
                n_events=joined.value.count(),
                sum_value=joined.value.sum(),
                min_value=joined.value.min(),
                max_value=joined.value.max(),
            )
            rows.extend(
                {
                    "experiment_id": self._experiment.name,
                    "unit_id": str(row["unit_id"]),
                    "ds": row["ds"],
                    "measure_key": measure_key,
                    "n_events": int(row["n_events"]),
                    "sum_value": float(row["sum_value"]),
                    "min_value": float(row["min_value"]),
                    "max_value": float(row["max_value"]),
                }
                for row in self._con.to_pyarrow(daily).to_pylist()
            )
            feed_names.append(fact_source.name)
        watermarks = [
            self._source_snapshot_evidence.complete_through_by_feed.get(name) for name in feed_names
        ]
        complete_through = (
            min(watermarks)
            if watermarks and all(value is not None for value in watermarks)
            else None
        )
        return _PublicationExtensionSpec(
            entry,
            request.kind,
            rows,
            {
                "observation_cutoff_ts": self._source_snapshot_evidence.observation_cutoff_ts,
                "complete_through_ts": complete_through,
            },
        )

    def _cluster_extension_spec(
        self, entry: Any, request: Any, exposures: ir.Table
    ) -> _PublicationExtensionSpec:
        cluster = self._experiment.cluster
        if cluster is None:
            raise ArtifactContractError(
                "artifact.extension.invalid",
                "cluster identity extension requires a declared cluster",
            )
        self._validate_cluster_labels(exposures, cluster)
        rows: list[dict[str, object]] = [
            {
                "experiment_id": self._experiment.name,
                "unit_id": str(row["unit_id"]),
                "cluster_id": str(row[cluster]),
            }
            for row in self._con.to_pyarrow(exposures).to_pylist()
            if row.get(cluster) is not None
        ]
        return _PublicationExtensionSpec(entry, request.kind, rows, {})

    def _uptake_extension_spec(
        self, entry: Any, request: Any, exposures: ir.Table
    ) -> _PublicationExtensionSpec:
        """Persist the first qualifying uptake timestamp for every enrolled unit."""
        definition = json.loads(entry.canonical_definition_json)
        events = self._uptake_cache[request.uptake_name]
        joined = exposures.inner_join(events, exposures.unit_id == events.unit_id)
        eligible = joined.filter(events.ts >= exposures.first_exposure_ts)
        window = definition["window_days"]
        if window is not None:
            eligible = eligible.filter(
                events.ts < exposures.first_exposure_ts + ibis.interval(days=window)
            )
        first = eligible.group_by(exposures.unit_id).aggregate(first_uptake_ts=events.ts.min())
        joined = exposures.left_join(first, "unit_id")
        query = joined.select(
            exposures.experiment_id,
            exposures.unit_id,
            uptake=first.first_uptake_ts.notnull(),
            first_uptake_ts=first.first_uptake_ts.cast("timestamp('UTC')"),
        )
        edge = compliance_event_horizon(exposures, events, self._experiment)
        coverage = ibis.literal(1).name("_one").as_table().select(edge=edge)
        observation_edge = self._con.to_pyarrow(coverage).to_pylist()[0]["edge"]
        certified_edge = None
        evidence = self._source_snapshot_evidence
        if evidence is not None:
            _, certified_edge = _effective_snapshot_edge(
                evidence.observation_cutoff_ts,
                evidence.complete_through_by_feed.get(
                    _find_fact_source(self._defs, request.uptake_name)[0].name
                ),
                self._experiment.day_boundary_offset,
                self._experiment.observation_horizon_day,
            )
        return _PublicationExtensionSpec(
            entry,
            request.kind,
            query,
            {"observation_edge": observation_edge, "certified_edge": certified_edge},
        )

    def _site_volume_extension_spec(self, entry: Any, request: Any) -> _PublicationExtensionSpec:
        rows: list[dict[str, object]] = []
        emitted_recipe_keys: set[str] = set()
        for metric_name in request.metric_names:
            metric = next(metric for metric in self._metrics if metric.name == metric_name)
            parts = (
                ((metric.numerator, "numerator"), (metric.denominator, "denominator"))
                if isinstance(metric, RatioMetric)
                else ((metric, "numerator"),)
            )
            for measure, _part in parts:
                measure_key = _measure_recipe_key(measure)
                if measure_key in emitted_recipe_keys:
                    continue
                emitted_recipe_keys.add(measure_key)
                fs, fact_def = _find_fact_source(self._defs, measure.fact)
                value_col = _resolve_value_column(fs, fact_def)
                pinned = self._event_cache[resolved_measure_key(measure, value_column=value_col)]
                events = pinned.filter(~pinned._metadata).select("unit_id", "ts", "metric", "value")
                local_day = _local_date(events.ts, self._experiment)
                events = events.filter(local_day >= ibis.literal(self._experiment.start_day))
                if self._experiment.end is not None:
                    events = events.filter(local_day <= ibis.literal(self._experiment.end_day))
                stats = unit_day_stats(
                    events,
                    source_key=measure_key,
                    experiment=self._experiment,
                )
                grouped: dict[Any, dict[str, Any]] = {}
                for row in self._con.to_pyarrow(stats).to_pylist():
                    key = row["ds"]
                    item = grouped.setdefault(
                        key,
                        {
                            "n_events": 0,
                            "sum_value": 0.0,
                            "min_value": float("inf"),
                            "max_value": float("-inf"),
                        },
                    )
                    item["n_events"] += int(row["n_events"])
                    item["sum_value"] += float(row["sum_value"])
                    item["min_value"] = min(item["min_value"], float(row["min_value"]))
                    item["max_value"] = max(item["max_value"], float(row["max_value"]))
                rows.extend(
                    {
                        "experiment_id": self._experiment.name,
                        "ds": day,
                        "measure_key": measure_key,
                        **item,
                    }
                    for day, item in grouped.items()
                )
        return _PublicationExtensionSpec(entry, request.kind, rows, {})

    def _encode_publication_extensions(
        self,
        publication: Any,
        context: Any,
        specs: tuple[_PublicationExtensionSpec, ...],
        source: Any,
    ) -> list[Any]:
        # Reads the source's live assignment-audit counters: they are only
        # refreshed by _assemble_publication, after this publisher is built.
        from increment.query.artifact_extensions import encode_extension

        extension_refs = []
        for spec in specs:
            if spec.kind == "trigger_population":
                relation = publication.write_relation(
                    "trigger_population", _relation_table(spec.rows, "trigger_population")
                )
                extension_refs.append(
                    TriggerPopulationExtension(
                        extension_version=3,
                        relation=relation,
                        definition_sha256=spec.entry.definition_sha256,
                        source_provenance_sha256=spec.entry.source_provenance_sha256,
                        trigger_name=spec.entry.request.trigger_name,
                        observation_cutoff_ts=cast(
                            dt.datetime, spec.fields["observation_cutoff_ts"]
                        ),
                        complete_through_ts=cast(
                            dt.datetime | None, spec.fields["complete_through_ts"]
                        ),
                    )
                )
                continue
            if spec.kind == "trigger_measure_stats":
                from increment.errors import refuse

                trigger_ref = next(
                    (
                        extension
                        for extension in extension_refs
                        if isinstance(extension, TriggerPopulationExtension)
                    ),
                    None,
                )
                if trigger_ref is None:
                    refuse(
                        _ARTIFACT_REFUSALS["artifact.evidence.unavailable"],
                        extension_kind="trigger_population",
                        operation="triggered_source",
                        route="republish with the matching trigger_population extension",
                    )
                relation = publication.write_relation(
                    "trigger_measure_stats",
                    _relation_table(spec.rows, "trigger_measure_stats"),
                )
                extension_refs.append(
                    TriggerMeasureStatsExtension(
                        relation=relation,
                        definition_sha256=spec.entry.definition_sha256,
                        source_provenance_sha256=spec.entry.source_provenance_sha256,
                        trigger_name=spec.entry.request.trigger_name,
                        metric_names=spec.entry.request.metric_names,
                        trigger_population_content_sha256=trigger_ref.relation.content_sha256,
                        observation_cutoff_ts=cast(
                            dt.datetime, spec.fields["observation_cutoff_ts"]
                        ),
                        complete_through_ts=cast(
                            dt.datetime | None, spec.fields["complete_through_ts"]
                        ),
                    )
                )
                continue
            if spec.kind == "encouragement_uptake":
                from increment.semantics.artifact import EncouragementUptakeExtension

                definition = json.loads(spec.entry.canonical_definition_json)
                relation = publication.write_relation("encouragement_uptake", spec.rows)
                extension_refs.append(
                    EncouragementUptakeExtension(
                        extension_version=2,
                        relation=relation,
                        definition_sha256=spec.entry.definition_sha256,
                        source_provenance_sha256=spec.entry.source_provenance_sha256,
                        uptake_name=spec.entry.request.uptake_name,
                        window_days=definition["window_days"],
                        one_sided=definition["one_sided"],
                        observation_edge=cast(dt.date | None, spec.fields["observation_edge"]),
                        **(
                            {"certified_edge": cast(dt.date, spec.fields["certified_edge"])}
                            if spec.fields["certified_edge"] is not None
                            else {}
                        ),
                    )
                )
                continue
            encoded_rows: Any = spec.rows
            if spec.kind == "assignment_counts":
                encoded_rows = {
                    "rows": spec.rows,
                    "audit": (
                        source._mixed_assignment_units or 0,
                        source._unassigned_assignment_units or 0,
                    ),
                }
            extension_refs.append(
                cast(Any, encode_extension)(
                    source,
                    spec.entry.request,
                    publication,
                    context=context,
                    rows=encoded_rows,
                    definition=json.loads(spec.entry.canonical_definition_json),
                    source_recipe=json.loads(spec.entry.canonical_source_recipe_json),
                    experiment_id=self._experiment.name,
                )
            )
        extension_refs.sort(
            key=lambda extension: (
                extension.kind,
                extension.extension_version,
                extension.relation.relation.name,
            )
        )
        return extension_refs

    def _publish_artifact(
        self,
        store: ArtifactStore,
        context: Any,
        assembly: _PublicationAssembly,
        extension_specs: tuple[_PublicationExtensionSpec, ...],
        source: Any,
        *,
        refresh_of: UnitDayArtifactRef | None,
    ) -> UnitDayArtifactRef:
        with store.begin_publication(
            expected_context=context, refresh_of=refresh_of
        ) as publication:
            exposure_ref = publication.write_relation(
                "exposures", _relation_table(assembly.exposure_rows, "exposures")
            )
            measure_ref = publication.write_relation("measure_stats", assembly.stats)
            extension_refs = self._encode_publication_extensions(
                publication, context, extension_specs, source
            )
            manifest = _assemble_publication_manifest(
                artifact_id=publication.artifact_id,
                generation_id=publication.generation_id,
                experiment_id=self._experiment.name,
                day_boundary=self._experiment.day_boundary,
                assembly=assembly,
                exposure_ref=exposure_ref,
                measure_ref=measure_ref,
                context=context,
                extension_refs=extension_refs,
            )
            return publication.publish_manifest(manifest)
