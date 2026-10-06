"""Ibis builders for the canonical unit-day panel data model.

Every function produces an ibis expression that executes on DuckDB (the
default backend) and compiles to Snowflake / BigQuery / Postgres —
verified by the cross-backend render tests in ``test_builders.py``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Literal, Never, assert_never, cast

import ibis
import ibis.expr.types as ir

from increment._frame_validation import CENSORING_DROPPED_UNITS
from increment._moment_plan import (
    CLUSTER_SIZE_GRAIN,
    CLUSTER_UPTAKE_GRAIN,
    DAY_GRAIN,
    UNIT_GRAIN,
    X_SLOT_ROLES,
    Moment,
    MomentPlan,
)
from increment._window import final_maturity_day as _final_maturity_day
from increment._window import maturity_days as _maturity_days  # noqa: F401
from increment._window import resolve_window_days as _resolve_window_days
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    raiser,
    refusals,
)
from increment.errors import warn as _warn_spec
from increment.query.artifact_contract import _aggregate_sum_within_extrema
from increment.semantics.models import (
    _DAY_BOUNDARY_RE,
    ActiveMetric,
    ConversionMetric,
    Experiment,
    FactRef,
    Filter,
    MeanMetric,
    Measure,
    Metric,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
)
from increment.sources import CENSOR_WARN_FRACTION

Renderer = Callable[..., str]


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    _warn_spec(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_WARNINGS["frame.censoring.dropped_units"] = CENSORING_DROPPED_UNITS


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "query.builders.cluster_column_missing": RefusalSpec(
            "query.builders.cluster_column_missing",
            CapabilityError,
            template="experiment '{experiment}' declares cluster '{cluster}', but the {table} "
            "table has no such column -- {remedy} Available columns: {columns}",
        ),
        "query.builders.cluster_undeclared": RefusalSpec(
            "query.builders.cluster_undeclared",
            CapabilityError,
            template="experiment '{experiment}' declares no cluster -- {operation} has no "
            "grain to count over. Count units off first_exposures directly instead.",
        ),
        "query.builders.cluster_size_imbalance": RefusalSpec(
            "query.builders.cluster_size_imbalance",
            CapabilityError,
            template="experiment '{experiment}': mean cluster size differs by {gap:.0%} "
            "across the contrasted arms (cluster '{cluster}': '{control_group}' "
            "{control_mean_size:.1f} units/cluster vs '{target_group}' "
            "{target_mean_size:.1f}), above the {threshold:.0%} whole-site refusal "
            "threshold. Cluster randomization balances clusters, not units, so under this "
            "imbalance the whole-site impact is not identified as the unit-weighted "
            "average treatment effect -- the differenced ratio-of-sums confounds the "
            "treatment effect with the cluster-size gap. Read the per-unit lift from "
            "run() instead.",
        ),
        "query.builders.site_volume_metric_type": RefusalSpec(
            "query.builders.site_volume_metric_type",
            CapabilityError,
            template="metric '{metric}': site volume is undefined for a {metric_type} "
            "metric -- {reason}",
        ),
        "query.builders.cluster_total_grain": RefusalSpec(
            "query.builders.cluster_total_grain",
            CapabilityError,
            template="{caller}: {grain} rows cannot combine with the declared cluster "
            "'{cluster}' -- cluster-robust inference is total-grain only. Read the "
            "total-grain result instead.",
        ),
        "query.builders.invalid_day_boundary": "invalid day_boundary {day_boundary!r}",
        "query.builders.check_cluster.experiment_arm_no": "experiment '{experiment_name}': arm '{group}' has no clusters in the exposure counts -- cannot check cross-arm cluster-size balance.",
        "query.builders.unknown_part_ratiometric": "Unknown part '{part}' for RatioMetric",
        "query.builders.part_ratiometric": "part='{part}' is only valid for RatioMetric",
        "query.builders.unknown_aggregation": "unknown aggregation {aggregation!r}",
        "query.builders.union_event_horizon": "union_event_horizon requires at least one event table",
        "query.builders.window_bound_stats": RefusalSpec(
            "query.builders.window_bound_stats",
            UnsupportedRequestError,
            template="window_bound_stats for {metric_type} not implemented",
        ),
        "query.builders.ratiometric_den_stats": "RatioMetric requires den_stats (denominator sufficient stats)",
        "query.builders.unit_totals_properties": "unit_totals: `properties_table` given without `by` -- nothing to group by",
        "query.builders.unit_totals_by": "unit_totals: `by` requires `properties_table`",
        "query.builders.group_summary_declared": "group_summary: declared cluster column '{cluster}' is not on the totals table -- build totals with unit_totals under the same cluster-declaring experiment. Available columns: {columns}",
        "query.builders.group_summary_ratio": "group_summary: ratio_metrics={ratio_names} names a metric whose denominator must come from a 'y_den' column, which is not on the totals table. Available columns: {columns}",
        "query.builders.group_summary_uptake": "group_summary: uptake=True declares an encouragement uptake fact in a 'd' column, which is not on the totals table -- build totals with unit_totals(uptake_events=...). Available columns: {columns}",
        "query.builders.asof_group_summary_metric_type_not_implemented": RefusalSpec(
            "query.builders.asof_group_summary_metric_type_not_implemented",
            UnsupportedRequestError,
            template="asof_group_summary for {metric_type} not implemented",
        ),
        "query.builders.asof_group_summary_den_panel_retention": "asof_group_summary: den_panel is meaningless for a RetentionMetric (retention has no denominator events -- its denominator is the gated population, which the band-open/maturity gate already produces)",
        "query.builders.cohort_group_summary": RefusalSpec(
            "query.builders.cohort_group_summary",
            UnsupportedRequestError,
            template="cohort_group_summary for {metric_type} not implemented -- the cohort axis exists for retention, whose outcome is not a property of a single activity day; use daily_group_summary for metric types that have an independent per-day reading",
        ),
        "query.builders.cohort_group_summary_unbounded_band": "cohort_group_summary: retention metric '{name}' declares an unbounded band (threshold_days={threshold_days}), so no cohort's outcome ever completes and no cohort could be reported. Declare threshold_days: [a, b] to bound the band.",
        "query.builders.asof_group_summary_completed_requires_uptake_window": "asof_group_summary: completed_windows_only=True under an encouragement design requires bounded outcome and uptake windows",
        "query.builders.asof_group_summary": "asof_group_summary: completed_windows_only=True is contradictory for retention metric '{name}' -- its band (threshold_days={threshold_days}) is open on the right, so no unit's window ever completes and the gate would admit nothing. Drop completed_windows_only to get the cumulative monitoring series, or declare threshold_days: [a, b] to bound the band.",
        "query.builders.daily_state_not_implemented": RefusalSpec(
            "query.builders.daily_state_not_implemented",
            UnsupportedRequestError,
            template="daily_group_summary state finalization for {metric_type} not implemented",
        ),
    },
)
_raise = raiser(_REFUSALS)
UNKNOWN_AGGREGATION = _REFUSALS["query.builders.unknown_aggregation"]


# ── Dense date spine is sized to the effective observation edge ────────


def _days_literal(n: int | ir.IntegerScalar) -> ir.IntegerScalar:
    """Cast n to IntegerScalar for .as_interval() - ibis's own type stub
    for literal() returns the generic Scalar base, missing that subtype."""
    return cast("ir.IntegerScalar", ibis.literal(n))


def _local_date_at_offset(ts: ir.TimestampColumn, offset: dt.timedelta) -> ir.DateColumn:
    """Event timestamp -> calendar date under a fixed UTC *offset* day
    boundary.

    Timezone-naive columns shift by the fixed offset before casting to a
    date (portable, backend-agnostic SQL). Timezone-aware columns instead
    convert through whole seconds since the UTC epoch, since a direct date
    cast would depend on the backend's session timezone. Not applied to
    strict timestamp-to-timestamp comparisons, where the offset cancels on
    both sides, or to dates already localized upstream.
    """
    if getattr(ts.type(), "timezone", None) is not None:
        epoch = ibis.literal(dt.datetime(1970, 1, 1, tzinfo=dt.UTC))
        seconds = ts.delta(epoch, unit="second") + int(offset.total_seconds())
        days = (seconds.cast("float64") / 86400.0).floor().cast("int32")
        local = ibis.literal(dt.date(1970, 1, 1)) + days.as_interval("D")
        return cast("ir.DateColumn", local.cast(dt.date))
    if offset == dt.timedelta(0):
        return cast("ir.DateColumn", ts.cast(dt.date))
    minutes = int(offset.total_seconds() // 60)
    return cast("ir.DateColumn", (ts + ibis.interval(minutes=minutes)).cast(dt.date))


def _local_date(ts: ir.TimestampColumn, experiment: Experiment | None) -> ir.DateColumn:
    """Event timestamp -> calendar date under *experiment*'s declared day
    boundary. See :func:`_local_date_at_offset` for the bucketing rule.
    """
    offset = dt.timedelta(0) if experiment is None else experiment.day_boundary_offset
    return _local_date_at_offset(ts, offset)


def day_boundary_offset(day_boundary: str) -> dt.timedelta:
    """Parse an already-validated ``day_boundary`` string (``Definitions``
    or ``Experiment``'s own grammar: ``'UTC'`` or a fixed offset like
    ``'UTC-05:00'``) into a signed UTC offset for
    :func:`_local_date_at_offset`.

    Mirrors ``Experiment.day_boundary_offset``; needed by callers on the
    calendar/report path, which carry only ``Definitions.day_boundary`` --
    a raw string, never an ``Experiment`` instance.
    """
    match = _DAY_BOUNDARY_RE.match(day_boundary)
    if match is None:
        _raise("query.builders.invalid_day_boundary", day_boundary=day_boundary)
    if match.group(2) is None:
        return dt.timedelta(0)
    sign = 1 if match.group(1) == "+" else -1
    return sign * dt.timedelta(hours=int(match.group(2)), minutes=int(match.group(3)))


def _reject_null_identity(table: ir.Table, *columns: str) -> ir.Table:
    """Drop rows whose entity-identity column(s) are NULL.

    A NULL identity value never equals another NULL in an equi-join, so an
    unfiltered NULL unit_id/ts silently vanishes from a downstream
    population or panel join while the raw row still exists for any
    consumer that counts straight off the source table -- an enrolled
    unit can silently disappear from one readout while still being
    counted by another. Filtering once at the boundary keeps every
    consumer's join semantics honest.
    """
    condition = None
    for col in columns:
        c = table[col].notnull()
        condition = c if condition is None else (condition & c)
    return table if condition is None else table.filter(condition)


def _scope_exposure_events(exposure_events: ir.Table, experiment: Experiment) -> ir.Table:
    """Admit raw exposure rows to the experiment's declared enrollment window.

    Enrollment runs [experiment.start, experiment.end] at whole-day grain
    (not raw timestamp): a pre-launch exposure otherwise anchors a unit's
    day 0 before the experiment existed, and the end day's own exposures
    must not be wrongly dropped. `experiment_id` may arrive all-NULL
    (synthesized upstream); NULL never equals NULL in SQL, which would
    break the joins downstream, so it is named to the experiment here.

    Shared by `first_exposures` (lazy, unexecuted) and the native
    integrity boundary, which executes over this same admission rule --
    do not introduce a second one.
    """
    exposure_events = exposure_events.filter(
        _local_date(exposure_events.ts, experiment) >= ibis.literal(experiment.start_day)
    )
    if experiment.end is not None:
        exposure_events = exposure_events.filter(
            _local_date(exposure_events.ts, experiment) <= ibis.literal(experiment.end_day)
        )
    return exposure_events.mutate(
        experiment_id=ibis.coalesce(exposure_events.experiment_id, experiment.name)
    )


def first_exposures(exposure_events: ir.Table, experiment: Experiment) -> ir.Table:
    """Dedup raw exposure events to the first exposure per unit.

    Units assigned to more than one real group, or with a NULL assignment,
    are dropped silently; call `mixed_assignment_units` separately to count
    and surface those drops. A NULL `unit_id`/`ts` row is dropped before
    grouping too -- an unresolvable identity, not an invalid assignment.
    """
    exposure_events = _reject_null_identity(exposure_events, "unit_id", "ts")
    exposure_events = _scope_exposure_events(exposure_events, experiment)

    cluster = experiment.cluster
    if cluster is not None and cluster not in exposure_events.columns:
        _raise(
            "query.builders.cluster_column_missing",
            experiment=experiment.name,
            cluster=cluster,
            table="exposure events",
            remedy="the cluster label must ride the exposure source.",
            columns=sorted(exposure_events.columns),
        )

    # First timestamp per (unit, experiment, group); the cluster label
    # carried is the one from the unit's FIRST exposure row (argmin on
    # ts), not the lexicographic minimum over all of its exposure rows.
    agg_kwargs: dict[str, ir.Scalar] = {"first_exposure_ts": exposure_events.ts.min()}
    if cluster is not None:
        agg_kwargs[cluster] = exposure_events[cluster].argmin(exposure_events.ts)
    per_group = exposure_events.group_by(["unit_id", "experiment_id", "group_id"]).agg(**agg_kwargs)

    # NULL is not an arm. A unit with only NULL assignments has no distinct
    # group, while a real-arm-plus-NULL unit has one; both are invalid.
    status = per_group.group_by(["unit_id", "experiment_id"]).agg(
        n_groups=per_group.group_id.nunique(),
        has_null=per_group.group_id.isnull().any(),
    )
    valid_units = status.filter((status.n_groups == 1) & ~status.has_null)
    valid = per_group.semi_join(valid_units, ["unit_id", "experiment_id"])

    keep = ["unit_id", "experiment_id", "group_id", "first_exposure_ts"]
    if cluster is not None:
        keep.append(cluster)
    return valid.select(*keep)


def triggered_population(exposures: ir.Table, triggers: ir.Table) -> ir.Table:
    """Narrow an enrollment population to the units that actually triggered.

    Units that never triggered could not have been affected, so including
    them dilutes the measured effect by the trigger rate. A semi-join keeps
    each enrolled row once regardless of how many times a unit triggered.
    """
    # The semi-join is what single-counts; distinct only makes that obvious.
    return exposures.semi_join(triggers.select("unit_id").distinct(), "unit_id")


def mixed_assignment_units(exposure_events: ir.Table, experiment: Experiment) -> ir.Table:
    """Count invalid assignment units in one bounded aggregate.

    The result carries separate counts for units observed in more than one
    real arm and units with a NULL assignment, plus a fingerprint of all
    invalid assignment rows. NULL is included in the fingerprint but never
    treated as a real arm.
    """
    # Same enrollment window as first_exposures, so a mixed/unassigned count
    # never flags a row that first_exposures would already exclude on dates.
    exposure_events = exposure_events.filter(
        _local_date(exposure_events.ts, experiment) >= ibis.literal(experiment.start_day)
    )
    if experiment.end is not None:
        exposure_events = exposure_events.filter(
            _local_date(exposure_events.ts, experiment) <= ibis.literal(experiment.end_day)
        )
    # Fact sources without an experiment_id get an all-NULL synthesized column.
    # Normalize it before the integrity join: SQL NULL never equals NULL, and
    # first_exposures uses the experiment name for this same identity.
    exposure_events = exposure_events.mutate(
        experiment_id=ibis.coalesce(exposure_events.experiment_id, experiment.name)
    )
    per_group = exposure_events.select("unit_id", "experiment_id", "group_id").distinct()
    status = per_group.group_by(["unit_id", "experiment_id"]).agg(
        n_groups=per_group.group_id.nunique(),
        has_null=per_group.group_id.isnull().any(),
    )
    invalid = status.filter((status.n_groups > 1) | status.has_null)
    invalid_assignments = per_group.inner_join(invalid, ["unit_id", "experiment_id"])

    def length_prefixed(value: ir.StringColumn) -> ir.StringColumn:
        return cast(
            "ir.StringColumn",
            value.length().cast("string").concat(":").concat(value),
        )

    def normalized(value) -> ir.StringColumn:
        # Include a null marker as a separate length-prefixed field, so NULL
        # cannot collide with a real label whose text matches a sentinel.
        marker = value.isnull().cast("string")
        text = cast(
            "ir.StringColumn",
            ibis.coalesce(value.cast("string"), ibis.literal("")),
        )
        return length_prefixed(marker).concat(length_prefixed(text))

    assignment_key = (
        normalized(invalid_assignments.experiment_id)
        .concat(normalized(invalid_assignments.unit_id))
        .concat(normalized(invalid_assignments.group_id))
    )
    assignment_hash = assignment_key.hash()
    salted_assignment_hash = assignment_key.concat("\x1d").hash()
    decimal_zero = ibis.literal(0).cast("decimal(38, 9)")
    hash_sum = ibis.coalesce(
        assignment_hash.cast("decimal(38, 9)").sum(),
        decimal_zero,
    )
    salted_hash_sum = ibis.coalesce(
        salted_assignment_hash.cast("decimal(38, 9)").sum(),
        decimal_zero,
    )
    assignment_count = assignment_hash.count()
    aggregate_key = (
        hash_sum.cast("string")
        .concat("\x1f")
        .concat(salted_hash_sum.cast("string"))
        .concat("\x1f")
        .concat(assignment_count.cast("string"))
    )
    # Hash raw assignments before aggregation. The aggregate string has fixed
    # width regardless of population size and preserves assignment boundaries.
    fingerprint = ibis.ifelse(
        assignment_count == 0,
        ibis.literal(0).cast("int64"),
        aggregate_key.hash(),
    )
    return invalid_assignments.aggregate(
        mixed_count=invalid_assignments.unit_id.nunique(where=invalid_assignments.n_groups > 1),
        unassigned_count=invalid_assignments.unit_id.nunique(where=invalid_assignments.has_null),
        fingerprint=fingerprint,
    )


def cluster_exposure_counts(exposures: ir.Table, experiment: Experiment) -> ir.Table:
    """Distinct CLUSTER and unit counts per arm, from `first_exposures`.

    Under cluster randomization the sample-ratio chi-square belongs at the
    cluster grain, since unequal cluster sizes move unit counts for
    reasons the randomizer never controlled. *experiment* must declare
    `cluster`; *exposures* must carry that cluster's label column.

    Returns
    -------
    ir.Table
        One row per arm: `experiment_id`, `group_id`, `n_clusters`,
        `n_units`.

    Raises
    ------
    CapabilityError
        *experiment* declares no cluster, or *exposures* lacks the column.
    """
    cluster = experiment.cluster
    if cluster is None:
        _raise(
            "query.builders.cluster_undeclared",
            experiment=experiment.name,
            operation="cluster_exposure_counts",
        )
    if cluster not in exposures.columns:
        _raise(
            "query.builders.cluster_column_missing",
            experiment=experiment.name,
            cluster=cluster,
            table="exposures",
            remedy="build it with first_exposures, which carries the declared cluster "
            "label through.",
            columns=sorted(exposures.columns),
        )
    # A label spanning arms counts once per arm; randomized and encouragement
    # designs refuse it separately, while observational dependence clusters may span arms.
    return exposures.group_by(["experiment_id", "group_id"]).agg(
        n_clusters=exposures[cluster].nunique(),
        n_units=exposures.unit_id.nunique(),
    )


# Diverging mean cluster sizes confound the sitewide number with size
# imbalance (can flip sign). Refuse past a 20% relative gap: |m_T-m_C|/min(m_T,m_C).
_MAX_CLUSTER_SIZE_IMBALANCE = 0.2


def check_cluster_size_balance(
    n_clusters: Mapping[str, int],
    n_units: Mapping[str, int],
    *,
    experiment_name: str,
    cluster: str,
    control_group: str,
    target_group: str,
) -> None:
    """Refuse a clustered sitewide contrast whose two arms carry materially
    different mean cluster sizes (see `_MAX_CLUSTER_SIZE_IMBALANCE`).

    Compares only the contrasted pair; a co-enrolled third arm's cluster
    size is not this contrast's confounder.

    Raises
    ------
    CapabilityError
        The relative gap in mean cluster size exceeds the threshold.
    """

    def _mean_size(group: str) -> float:
        k = n_clusters.get(group, 0)
        if k <= 0:
            _raise(
                "query.builders.check_cluster.experiment_arm_no",
                experiment_name=experiment_name,
                group=group,
            )
        return n_units[group] / k

    m_control = _mean_size(control_group)
    m_target = _mean_size(target_group)
    gap = abs(m_target - m_control) / min(m_control, m_target)
    if gap > _MAX_CLUSTER_SIZE_IMBALANCE:
        _raise(
            "query.builders.cluster_size_imbalance",
            experiment=experiment_name,
            cluster=cluster,
            control_group=control_group,
            target_group=target_group,
            control_mean_size=m_control,
            target_mean_size=m_target,
            gap=gap,
            threshold=_MAX_CLUSTER_SIZE_IMBALANCE,
        )


def daily_exposure_counts(exposures: ir.Table, experiment: Experiment | None = None) -> ir.Table:
    """Daily and cumulative enrollment counts per experiment arm.

    Densifies over observed exposure dates and groups only: a day/arm pair
    with zero enrollments still gets an explicit `n_daily = 0` row, so a
    cross-arm daily total never silently drops the quiet arm.

    Returns
    -------
    ir.Table
        Columns: `experiment_id`, `ds`, `group_id`, `n_daily`, `n_cumulative`.
    """
    dated = exposures.mutate(ds=_local_date(exposures.first_exposure_ts, experiment))

    # 1. Raw daily counts (sparse: an arm with zero enrollments on a
    #    given day simply has no row here).
    daily = dated.group_by(["experiment_id", "ds", "group_id"]).agg(n_daily=dated.unit_id.count())

    # 2. Densify: cross-join observed dates x groups (scoped per
    #    experiment_id), then fill missing days with 0.
    dates = dated.select("experiment_id", "ds").distinct()
    groups = dated.select("experiment_id", "group_id").distinct()
    grid = dates.inner_join(groups, "experiment_id").select("experiment_id", "ds", "group_id")

    densified = grid.left_join(
        daily,
        (grid.experiment_id == daily.experiment_id)
        & (grid.ds == daily.ds)
        & (grid.group_id == daily.group_id),
        rname="{name}_daily",
    ).select(
        grid.experiment_id,
        grid.ds,
        grid.group_id,
        n_daily=ibis.coalesce(daily.n_daily, 0),
    )

    # ── 3. Cumulative sum per (experiment_id, group_id), ordered by ds ───
    w = ibis.cumulative_window(group_by=["experiment_id", "group_id"], order_by="ds")
    result = densified.mutate(n_cumulative=densified.n_daily.sum().over(w))

    return result.select("experiment_id", "ds", "group_id", "n_daily", "n_cumulative")


def _is_occurrence_only(ref: FactRef) -> bool:
    """True when *ref*'s events carry no real value (occurrence = 1 per
    event); `metric_events` and `resolved_measure_key` must agree on this
    exactly to treat two refs as interchangeable.
    """
    return isinstance(ref, ConversionMetric | RetentionMetric | ActiveMetric) or (
        isinstance(ref, Measure) and ref.aggregation == "count"
    )


def resolved_measure_key(
    ref: FactRef, *, value_column: str | None
) -> tuple[str, tuple[tuple[str, str, tuple[str | int | float | bool, ...]], ...], str | None]:
    """Identity of the resolved measure *ref* reduces to, before aggregation.

    A resolved measure is a fact plus its filters plus a value column -
    everything `metric_events` needs to build one ref's events. Two refs
    that produce the same key here produce identical `metric_events`
    output in every load-bearing column, which is how `Analysis` dedupes
    declared metrics down to the distinct measures actually scanned.

    `value_column` folds to None for an occurrence-only ref (its events
    never touch the value column). Filter order is canonicalized away:
    filters apply conjunctively, so two refs differing only in filter
    order are the same measure.
    """
    filters_key = tuple(
        sorted(
            ((f.property, f.op, tuple(f.values)) for f in ref.filters),
            # repr-map the values so heterogeneously-typed value tuples
            # (str vs int) still admit a deterministic total order.
            key=lambda k: (k[0], k[1], tuple(map(repr, k[2]))),
        )
    )
    return (ref.fact, filters_key, None if _is_occurrence_only(ref) else value_column)


def metric_events(
    fact_table: ir.Table,
    metric: Metric,
    value_column: str | None = None,
    part: Literal["numerator", "denominator"] = "numerator",
    keep: Sequence[str] = (),
) -> ir.Table:
    """Filter *fact_table* to the rows relevant for *metric* (or a ratio part).

    `value_column` is the numeric column to use as the event value (None
    for occurrence-only metrics - value is implicitly 1 per event). `part`
    selects numerator vs denominator for a `RatioMetric`; ignored otherwise.

    Returns
    -------
    ir.Table
        Columns: `unit_id`, `ts`, `metric`, `value`. A value-based ref
        drops rows whose value column is NULL (complete-case); an
        occurrence-only ref never reads the value column, so its rows
        always count.
    """
    # ── Determine the reference Measure/FactRef ────────────────────────
    if isinstance(metric, RatioMetric):
        if part == "numerator":
            _ref = metric.numerator
        elif part == "denominator":
            _ref = metric.denominator
        else:
            _raise("query.builders.unknown_part_ratiometric", part=part)
    else:
        if part != "numerator":
            _raise("query.builders.part_ratiometric", part=part)
        _ref = metric

    # ── Filter to the metric's fact rows ───────────────────────────────
    filtered = fact_table.filter(fact_table.event == _ref.fact)

    # ── Apply metric-level filters ─────────────────────────────────────
    for flt in _ref.filters:
        filtered = _apply_filter(filtered, flt)

    # ── Build the value column ─────────────────────────────────────────
    if _is_occurrence_only(_ref):
        # Cast the literal explicitly: PostgreSQL otherwise returns NUMERIC, not a float.
        return filtered.mutate(
            metric=ibis.literal(metric.name),
            value=ibis.literal(1, type="int32").cast("float64"),
        ).select("unit_id", "ts", "metric", "value", *keep)

    # A NULL value is dropped entirely (complete-case): counting it would
    # inflate n_events while adding nothing to sum_value, biasing avg down.
    col_name = value_column or "value"
    filtered = filtered.filter(filtered[col_name].notnull())
    return filtered.mutate(
        metric=ibis.literal(metric.name),
        value=filtered[col_name].cast("float64"),
    ).select("unit_id", "ts", "metric", "value", *keep)


def _windowed_fact_sum(
    fact_table: ir.Table,
    metric: Metric,
    experiment: Experiment,
    *,
    value_column: str | None,
    part: Literal["numerator", "denominator"],
) -> ir.Table:
    """One ratio part's whole-site sum over events in
    `[experiment.start, experiment.end]` (date-inclusive, an unset `end`
    open on the right).

    Shared by `site_volume` (one fact table serving both ratio parts) and
    any caller resolving numerator/denominator against independently
    picked fact tables. A missing-rows sum is a typed 0.0, never SQL
    NULL, so a whole-site ratio and its complement share one zero
    convention.
    """
    events = metric_events(fact_table, metric, value_column, part=part)
    local_date = _local_date(events.ts, experiment)
    events = events.filter(local_date >= ibis.literal(experiment.start_day))
    if experiment.end is not None:
        events = events.filter(local_date <= ibis.literal(experiment.end_day))
    return events.aggregate(y=ibis.coalesce(events.value.sum(), 0.0))


def site_volume(
    fact_table: ir.Table,
    metric: Metric,
    experiment: Experiment,
    *,
    value_column: str | None = None,
) -> ir.Table:
    """Whole-site metric volume within the experiment's enrollment window
    (no exposure join - every unit's events count, exposed or not).

    Window is `[experiment.start, experiment.end]`, both date-inclusive
    under the experiment's day boundary; an unset `end` leaves it open on
    the right. `RetentionMetric` and `QuantileMetric` have no site-wide
    reading and raise `CapabilityError`.

    Returns
    -------
    ir.Table
        One row: `metric`, `y` (numerator sum), `y_den` (denominator sum
        for a `RatioMetric`, else NULL) - same convention as `unit_totals`.
    """
    validate_site_volume_metric(metric)
    num = _windowed_fact_sum(
        fact_table, metric, experiment, value_column=value_column, part="numerator"
    )
    if isinstance(metric, RatioMetric):
        den = _windowed_fact_sum(
            fact_table, metric, experiment, value_column=value_column, part="denominator"
        )
        totals = num.cross_join(den, rname="{name}_den")
    else:
        totals = num.mutate(y_den=ibis.null().cast("float64"))
    return totals.mutate(metric=ibis.literal(metric.name)).select("metric", "y", "y_den")


def validate_site_volume_metric(metric: Metric) -> None:
    """Refuse metrics that have no exposure-independent site-wide value."""
    if isinstance(metric, RetentionMetric):
        _raise(
            "query.builders.site_volume_metric_type",
            metric=metric.name,
            metric_type="retention",
            reason="its outcome is anchored to each unit's OWN exposure time (the "
            "observation band), not a raw event stream that sums independently of exposure.",
        )
    if isinstance(metric, QuantileMetric):
        _raise(
            "query.builders.site_volume_metric_type",
            metric=metric.name,
            metric_type="quantile",
            reason="a distributional statistic is not additive across events.",
        )


def unit_day_stats(
    events: ir.Table,
    *,
    source_key: str,
    experiment: Experiment | None = None,
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Sparse per-unit-per-day sufficient statistics for one resolved measure.

    Sparse by design: a row exists only for a unit-day with at least one
    observed event; zero-filling is a separate concern handled later by
    `panel_spine`. No aggregation is applied here, so every metric sharing
    this resolved measure can reuse one materialized copy.

    Day bucketing uses *experiment*'s declared day boundary when given
    (the experiment path); otherwise *day_boundary_offset* directly (the
    report/calendar path, which carries only ``Definitions.day_boundary``
    -- see :func:`day_boundary_offset`).

    Returns
    -------
    ir.Table
        Columns: `unit_id`, `ds`, `source_key`, `n_events`, `sum_value`,
        `min_value`, `max_value`.
    """
    offset = experiment.day_boundary_offset if experiment is not None else day_boundary_offset
    daily = events.mutate(ds=_local_date_at_offset(events.ts, offset))
    stats = daily.group_by(["unit_id", "ds"]).agg(
        n_events=daily.count(),
        sum_value=daily.value.sum(),
        min_value=daily.value.min(),
        max_value=daily.value.max(),
    )
    return stats.mutate(source_key=ibis.literal(source_key)).select(
        "unit_id", "ds", "source_key", "n_events", "sum_value", "min_value", "max_value"
    )


def validate_unit_day_aggregate_rows(rows: Sequence[Mapping[str, object]]) -> None:
    """Reject sufficient-stat rows that cannot describe real events."""
    import math

    from increment.errors import CapabilityError

    for row in rows:
        try:
            n_events = row["n_events"]
            total = row["sum_value"]
            minimum = row["min_value"]
            maximum = row["max_value"]
        except KeyError as exc:
            raise CapabilityError(
                "unit-day aggregate is missing a sufficient statistic",
                code="artifact.aggregate.impossible",
                context={"field": str(exc)},
            ) from exc
        try:
            invalid = (
                isinstance(n_events, bool)
                or not isinstance(n_events, int)
                or n_events < 1
                or n_events > 2**53
                or not isinstance(total, (int, float))
                or not isinstance(minimum, (int, float))
                or not isinstance(maximum, (int, float))
                or not math.isfinite(total)
                or not math.isfinite(minimum)
                or not math.isfinite(maximum)
                or minimum > maximum
                or not _aggregate_sum_within_extrema(
                    float(total), n_events, float(minimum), float(maximum)
                )
                or (n_events == 1 and (total != minimum or total != maximum))
            )
        except OverflowError:
            invalid = True
        if invalid:
            raise CapabilityError(
                "unit-day aggregate statistics are impossible",
                code="artifact.aggregate.impossible",
                context={"row": dict(row)},
            )


def aggregate_stats(stats: ir.Table, *, aggregation: str) -> ir.Table:
    """Collapse per-unit-day sufficient statistics to one `y` per unit.

    ``avg_event`` is the sparse event mean. Calendar-day averages require a
    zero-filled experiment spine and are handled by :func:`unit_totals`.
    """
    g = stats.group_by("unit_id")
    if aggregation == "sum":
        return g.agg(y=stats.sum_value.sum())
    if aggregation == "count":
        return g.agg(y=stats.n_events.sum().cast("float64"))
    if aggregation == "avg_event":
        return g.agg(y=stats.sum_value.sum() / stats.n_events.sum())
    if aggregation == "min":
        return g.agg(y=stats.min_value.min())
    if aggregation == "max":
        return g.agg(y=stats.max_value.max())
    if aggregation == "count_distinct":
        # Day-grain by construction: this counts distinct daily-SUMMED
        # values, not distinct raw event values.
        return g.agg(y=stats.sum_value.nunique().cast("float64"))
    _raise("query.builders.unknown_aggregation", aggregation=aggregation)


def _measure_totals(keys: ir.Table, dense: ir.Table, aggregation: str) -> ir.Table:
    """Reduce a zero-filled dense unit-day relation to one row per unit under
    *aggregation*, preserving ``keys``' full column set (``unit_id``,
    ``experiment_id``, ``group_id``, ``metric``) plus ``y``. Same
    per-aggregation unit-inclusion rule MeanMetric/QuantileMetric already
    use for their own declared ``aggregation`` (see the module docstring on
    ``unit_totals``), generalized so a RatioMetric's numerator and
    denominator can each apply it independently.
    """
    key_cols = keys.columns
    if aggregation == "avg_calendar_day":
        agg = dense.group_by("unit_id").agg(y=dense.sum_value.sum() / dense.count())
        return keys.join(agg, "unit_id").select(*key_cols, "y")
    if aggregation == "avg_event":
        observed = dense.filter(dense.n_events > 0)
        agg = aggregate_stats(observed, aggregation="avg_event")
        return keys.join(agg, "unit_id").select(*key_cols, "y")
    if aggregation in ("min", "max", "count_distinct"):
        observed = dense.filter(dense.n_events > 0)
        agg = aggregate_stats(observed, aggregation=aggregation)
        totals = keys.left_join(agg, "unit_id").select(*key_cols, "y")
        return totals.mutate(y=ibis.coalesce(totals.y, 0.0))
    agg = aggregate_stats(dense, aggregation=aggregation)
    return keys.join(agg, "unit_id").select(*key_cols, "y")


def _finalize_state_value(state: ir.Table, aggregation: str) -> ir.Table:
    """Finalize one unit-day (or unit-window) sufficient-state row to a
    single ``value`` under *aggregation* -- the day-grain analogue of
    ``_measure_totals``'s per-aggregation unit-inclusion rule, applied to a
    SINGLE row instead of collapsing across every day in a window.

    ``sum``/``count``/``avg_calendar_day`` are defined at zero events and
    always kept (``avg_calendar_day`` divides by one calendar day at this
    grain, so it equals ``sum_value`` here -- the day-count divisor only
    matters once multiple days combine, see ``_asof_masked_value``).
    ``avg_event`` is undefined at zero events (0/0): the row is DROPPED,
    not zero-filled. ``min``/``max``/``count_distinct`` are undefined at
    zero events too, but folded to the ``0.0``/``0`` identity KEPT --
    *state*'s ``min_value``/``max_value`` already carry that zero-fill
    (see ``_dense_unit_days``/``unit_day_panel``), so no unit is dropped.
    """
    if aggregation in ("sum", "avg_calendar_day"):
        return state.mutate(value=state.sum_value)
    if aggregation == "count":
        return state.mutate(value=state.n_events.cast("float64"))
    if aggregation == "avg_event":
        observed = state.filter(state.n_events > 0)
        return observed.mutate(value=observed.sum_value / observed.n_events)
    if aggregation == "min":
        return state.mutate(value=state.min_value)
    if aggregation == "max":
        return state.mutate(value=state.max_value)
    if aggregation == "count_distinct":
        return state.mutate(value=(state.n_events > 0).cast("int32").cast("float64"))
    _raise("query.builders.unknown_aggregation", aggregation=aggregation)


def _finalize_daily_state(
    panel: ir.Table,
    metric: Metric,
    *,
    part: Literal["numerator", "denominator"] = "numerator",
) -> ir.Table:
    """Dispatch one day-panel's sufficient state to ``_finalize_state_value``
    under *metric*'s declared aggregation (or *part*'s, for a RatioMetric).

    ``ConversionMetric``'s any-occurrence semantics have no declared
    ``Measure.aggregation`` -- a day counts as converted if any event
    occurred, never a click count, so it finalizes directly from
    ``n_events`` rather than through the shared aggregation dispatch.
    """
    if isinstance(metric, ConversionMetric):
        return panel.mutate(value=(panel.n_events > 0).cast("int32").cast("float64"))
    if isinstance(metric, RatioMetric):
        aggregation = (
            metric.denominator.aggregation
            if part == "denominator"
            else metric.numerator.aggregation
        )
    elif isinstance(metric, MeanMetric | QuantileMetric):
        aggregation = metric.aggregation
    else:
        _raise("query.builders.daily_state_not_implemented", metric_type=type(metric).__name__)
    return _finalize_state_value(panel, aggregation)


def winsorize_unit_totals(totals: ir.Table, metric: Metric) -> ir.Table:
    """Apply one pooled outcome transform to a metric's unit totals."""
    config = getattr(metric, "winsorization", None)
    if config is None:
        return totals.mutate(
            y_raw=totals.y,
            **{
                field: ibis.null().cast(
                    "float64" if "percentile" in field or "bound" in field else "int64"
                )
                for field in (
                    "winsor_lower_percentile",
                    "winsor_upper_percentile",
                    "winsor_lower_bound",
                    "winsor_upper_bound",
                    "winsor_n",
                    "winsor_n_lower",
                    "winsor_n_upper",
                )
            },
        )
    raw_y = totals.y
    lower = (
        ibis.literal(config.lower_value)
        if config.lower_value is not None
        else raw_y.quantile(config.lower_percentile)
        if config.lower_percentile is not None
        else None
    )
    upper = (
        ibis.literal(config.upper_value)
        if config.upper_value is not None
        else raw_y.quantile(config.upper_percentile)
        if config.upper_percentile is not None
        else None
    )
    clipped = raw_y.clip(lower, upper)
    lower_bound = lower if lower is not None else ibis.null().cast("float64")
    upper_bound = upper if upper is not None else ibis.null().cast("float64")
    lower_flag = (
        (raw_y < lower).cast("int64") if lower is not None else ibis.literal(0).cast("int64")
    )
    upper_flag = (
        (raw_y > upper).cast("int64") if upper is not None else ibis.literal(0).cast("int64")
    )
    return totals.mutate(
        y_raw=raw_y,
        y=clipped,
        winsor_lower_percentile=ibis.literal(config.lower_percentile, type="float64"),
        winsor_upper_percentile=ibis.literal(config.upper_percentile, type="float64"),
        winsor_lower_bound=lower_bound,
        winsor_upper_bound=upper_bound,
        winsor_n=ibis.literal(1).cast("int64"),
        winsor_n_lower=lower_flag,
        winsor_n_upper=upper_flag,
    )


def union_event_horizon(
    event_tables: Sequence[ir.Table], experiment: Experiment | None = None
) -> ir.Scalar:
    """Latest event date across ALL metrics' event streams.

    Bounded by the union of every metric's events (not one metric's max)
    so the spine's right edge stays metric-independent. NULL only when
    every source is eventless.
    """
    if not event_tables:
        _raise("query.builders.union_event_horizon")
    maxes = [_local_date(t.ts, experiment).max() for t in event_tables]
    out = maxes[0]
    for m in maxes[1:]:
        # coalesce guards NULL-poisoning `greatest` semantics: prefer the
        # pairwise greatest, fall back to whichever side is non-NULL.
        out = ibis.coalesce(ibis.greatest(out, m), out, m)
    # ibis.greatest() is typed as the generic Value base (same stub gap as
    # _days_literal), but two Scalar .max() inputs always yield a Scalar.
    return cast("ir.Scalar", out)


def compliance_event_horizon(
    exposures: ir.Table, uptake_events: ir.Table, experiment: Experiment
) -> ir.Scalar:
    """Latest enrollment or post-enrollment uptake date, before uptake-window gates."""
    if experiment.observation_horizon is not None:
        return ibis.literal(experiment.observation_horizon_day)
    joined = uptake_events.inner_join(exposures, "unit_id")
    observed = joined.filter(uptake_events.ts >= exposures.first_exposure_ts).select(
        ts=uptake_events.ts
    )
    return union_event_horizon(
        [exposures.select(ts=exposures.first_exposure_ts), observed], experiment
    )


def panel_spine(
    exposures: ir.Table,
    experiment: Experiment,
    *,
    end_date: ir.Scalar | None,
) -> ir.Table:
    """Dense unit x day spine; carries no event data, so all metrics share it.

    Columns: `unit_id`, `experiment_id`, `group_id`, `first_exposure_ts`,
    `first_exposure_date`, `day_offset`, `ds`.

    A unit enrolled after the effective right edge gets exactly one row,
    with `day_offset`/`ds` both NULL: enrollment identity is not itself an
    observed outcome, and there is no observed calendar day to date it by.
    Downstream day-axis builders must filter `ds.notnull()` before using
    this as a day axis; total-grain censoring depends on seeing these
    rows, so the spine returned here is deliberately not pre-filtered.

    Parameters
    ----------
    exposures : ir.Table
        Output of `first_exposures`.
    experiment : Experiment
        Provides the unit-of-analysis grain and time bounds.
    end_date : ir.Scalar | None
        Right-edge fallback used only when `experiment.observation_horizon`
        is unset; an explicit horizon always wins.
    """
    firsts = exposures.mutate(
        first_exposure_date=_local_date(exposures.first_exposure_ts, experiment),
    )

    # The OBSERVATION horizon, not the enrollment end: a unit enrolled just
    # before `end` needs spine rows past `end` for its window to close.
    if experiment.observation_horizon is not None:
        edge = ibis.literal(experiment.observation_horizon_day)
    else:
        # Without a declared horizon, use the caller's event edge (NULL when
        # eventless). Never fall back to the latest first exposure: that would
        # treat a late enrollee as matured with zero observed days.
        edge = end_date if end_date is not None else ibis.null().cast("date")

    spine_cols = [
        "unit_id",
        "experiment_id",
        "group_id",
        "first_exposure_ts",
        "first_exposure_date",
    ]
    if experiment.cluster is not None:
        if experiment.cluster not in exposures.columns:
            _raise(
                "query.builders.cluster_column_missing",
                experiment=experiment.name,
                cluster=experiment.cluster,
                table="exposures",
                remedy="build it with first_exposures, which carries the declared cluster "
                "label through.",
                columns=sorted(exposures.columns),
            )
        spine_cols.append(experiment.cluster)

    # Each unit gets inclusive offsets from first exposure through the edge. A
    # non-positive count, including a NULL edge, unnests to zero rows; the left
    # join below keeps the unit with NULL day_offset and ds.
    keys = firsts.select(*spine_cols)
    day_count = ibis.greatest(
        ibis.coalesce(
            cast("ir.DateValue", edge).delta(firsts.first_exposure_date, unit="day") + 1, 0
        ),
        0,
    )
    # Relational unnest avoids the `_u`/`_u_2` alias pair sqlglot emits for a
    # projection-embedded unnest, which collides on Snowflake.
    expanded = firsts.mutate(_day_offsets=ibis.range(0, day_count, 1)).unnest("_day_offsets")
    days = expanded.select(
        "unit_id",
        "experiment_id",
        "first_exposure_date",
        day_offset=expanded["_day_offsets"],
    )
    days = days.mutate(
        ds=(days.first_exposure_date + days.day_offset.as_interval("D")).cast("date")
    ).select("unit_id", "experiment_id", "day_offset", "ds")
    return keys.left_join(days, ["unit_id", "experiment_id"]).select(
        *spine_cols, "day_offset", "ds"
    )


def post_exposure_stats(
    events: ir.Table,
    exposures: ir.Table,
    *,
    source_key: str,
    experiment: Experiment | None = None,
) -> ir.Table:
    """Day-grain sufficient stats for *events*, scoped to strictly after
    each unit's first exposure.

    The post-exposure boundary must be applied here, on raw events, before
    `unit_day_stats` collapses them to day grain - once merged into one
    day's row, pre- and post-exposure activity can no longer be told apart.
    """
    fe = exposures.select("unit_id", "first_exposure_ts").distinct()
    joined = events.join(fe, events.unit_id == fe.unit_id, rname="{name}_fe")
    scoped = joined.filter(joined.ts > joined.first_exposure_ts)
    return unit_day_stats(
        scoped.select(events.columns), source_key=source_key, experiment=experiment
    )


def pre_period_stats(
    events: ir.Table, exposures: ir.Table, experiment: Experiment, *, source_key: str
) -> ir.Table:
    """Day-grain sufficient stats for the CUPED pre-period covariate.

    Scoped to `[first_exposure_date - n_pre_periods, first_exposure_ts)`;
    like `post_exposure_stats`, this filter must run on raw events before
    `unit_day_stats` collapses them to day grain.
    """
    fe = exposures.select(
        "unit_id",
        "first_exposure_ts",
        first_exposure_date=_local_date(exposures.first_exposure_ts, experiment),
    ).distinct()
    joined = events.join(fe, events.unit_id == fe.unit_id, rname="{name}_fe")
    pre_start = joined.first_exposure_date - _days_literal(experiment.n_pre_periods).as_interval(
        "D"
    )
    # Lower bound compares against the LOCALIZED day (pre_start derives
    # from first_exposure_date); the upper bound stays a raw ts comparison.
    scoped = joined.filter(
        (_local_date(joined.ts, experiment) >= pre_start) & (joined.ts < joined.first_exposure_ts)
    )
    return unit_day_stats(
        scoped.select(events.columns), source_key=source_key, experiment=experiment
    )


def join_pre_period_covariate(table: ir.Table, pre_stats: ir.Table) -> ir.Table:
    """Attach one zero-filled fixed pre-period total ``x`` per unit."""
    original = list(table.columns)
    pre_totals = pre_stats.group_by("unit_id").agg(x=pre_stats.sum_value.sum())
    joined = table.left_join(pre_totals, "unit_id")
    joined = joined.mutate(x=ibis.coalesce(joined.x, 0.0))
    return joined.select(*original, "x")


def unit_day_spine_stats(
    exposures: ir.Table,
    events: ir.Table,
    experiment: Experiment,
    metric_name: str,
    *,
    end_date: ir.Scalar | None = None,
) -> tuple[ir.Table, ir.Table]:
    """The `(spine, stats)` pair `unit_totals` consumes.

    Parameters
    ----------
    end_date : ir.Scalar | None
        Spine right-edge fallback (see `panel_spine`); defaults to this
        metric's own latest event date when omitted - correct only for a
        single-stream metric evaluated standalone. A `RatioMetric`, or any
        caller sharing one spine across multiple metrics, MUST pass a
        `union_event_horizon` that includes every stream (denominators
        included); otherwise events dated past the default fallback
        silently vanish from a running experiment's aggregate.
    """
    if end_date is None:
        end_date = _local_date(events.ts, experiment).max()
    spine = panel_spine(exposures, experiment, end_date=end_date)
    stats = post_exposure_stats(events, exposures, source_key=metric_name, experiment=experiment)
    return spine, stats


def unit_day_panel(
    exposures: ir.Table,
    events: ir.Table,
    experiment: Experiment,
    metric_name: str | None = None,
    *,
    end_date: ir.Scalar | None = None,
    include_exposure: bool = False,
) -> ir.Table:
    """Build a dense unit x day panel of mergeable per-unit-day sufficient
    state.

    Every enrolled unit with an observed day gets a row for every calendar
    day from its first exposure through the effective finite edge: the
    declared observation horizon or the supplied event horizon. A unit
    enrolled past that edge has no observed day at all and is excluded
    here -- it stays on `panel_spine`'s total-grain spine as a null-date
    row, but never becomes a day-axis observation.
    Post-exposure events are left-joined; days without events get every
    state field zero-filled. A caller that also built this metric's
    `(spine, stats)` pair via `unit_day_spine_stats` with an explicit
    `end_date` MUST pass the same value here, or the two readouts can
    diverge for one metric.

    Carries the same mergeable sufficient state `unit_day_stats` produces
    (`n_events`, `sum_value`, `min_value`, `max_value`) rather than a
    single pre-summed `value` -- a caller needing this metric's declared
    aggregation (`avg_event`, `min`, `max`, `count_distinct`, not just
    `sum`) must be able to recombine this state over any requested day or
    window; see `_finalize_daily_state`/`_asof_masked_value`.

    Parameters
    ----------
    exposures : ir.Table
        Enrolled units and their first exposure timestamps.
    events : ir.Table
        Metric events with unit, timestamp, value and metric columns.
    experiment : Experiment
        Experiment identity, observation horizon and local day boundary.
    metric_name : str | None
        Metric label; when omitted, infer it from the event relation.
    end_date : ir.Scalar | None
        Fallback edge only when ``experiment.observation_horizon`` is unset;
        defaults to the last event's local date. A declared horizon wins.
    include_exposure : bool
        Include events exactly at first exposure, as required for uptake.

    Returns
    -------
    ir.Table
        Columns: `unit_id`, `ds`, `experiment_id`, `group_id`, `metric`,
        `n_events` (0-filled), `sum_value`/`min_value`/`max_value`
        (0.0-filled), `first_exposure_ts`, `first_exposure_date`.
    """
    _metric_literal = ibis.literal(metric_name) if metric_name else None

    # events may be empty: .max() is then NULL, so panel_spine leaves
    # every unit with a null-date row, keeping them off the day axis.
    if end_date is None:
        end_date = _local_date(events.ts, experiment).max()
    spine = panel_spine(exposures, experiment, end_date=end_date)
    spine = spine.filter(spine.ds.notnull())

    # Ordinary outcomes start strictly after exposure; inclusive facts such
    # as uptake also admit the exact exposure timestamp.
    event_ds = _local_date(events.ts, experiment)
    joined = spine.left_join(
        events,
        (spine.unit_id == events.unit_id)
        & (spine.ds == event_ds)
        & (
            (events.ts >= spine.first_exposure_ts)
            if include_exposure
            else (events.ts > spine.first_exposure_ts)
        ),
        rname="{name}_event",
    )

    # Collapse to one row per unit x day, keeping the mergeable sufficient
    # state a caller may need to recombine under any declared aggregation.
    collapse = joined.group_by(
        [
            "unit_id",
            "ds",
            "experiment_id",
            "group_id",
            "first_exposure_ts",
            "first_exposure_date",
        ]
    ).agg(
        n_events=joined.value.count(),
        sum_value=joined.value.sum(),
        min_value=joined.value.min(),
        max_value=joined.value.max(),
    )

    # ── Fill nulls and stamp the metric name ────────────────────────
    if _metric_literal is not None:
        result = collapse.mutate(
            n_events=ibis.coalesce(collapse.n_events, 0),
            sum_value=ibis.coalesce(collapse.sum_value, 0.0),
            min_value=ibis.coalesce(collapse.min_value, 0.0),
            max_value=ibis.coalesce(collapse.max_value, 0.0),
            metric=_metric_literal,
        )
    else:
        # Infer metric name from the events table (metric_events stamps every
        # row with the metric's literal name, so .min() is deterministic).
        result = collapse.mutate(
            n_events=ibis.coalesce(collapse.n_events, 0),
            sum_value=ibis.coalesce(collapse.sum_value, 0.0),
            min_value=ibis.coalesce(collapse.min_value, 0.0),
            max_value=ibis.coalesce(collapse.max_value, 0.0),
            metric=events.metric.min(),
        )
    return result.select(
        "unit_id",
        "ds",
        "experiment_id",
        "group_id",
        "metric",
        "n_events",
        "sum_value",
        "min_value",
        "max_value",
        "first_exposure_ts",
        "first_exposure_date",
    )


def breakout_property_table(
    raw: ir.Table, property_name: str, exposures: ir.Table | None
) -> ir.Table:
    """Deduplicate a breakout/factor property to one value per unit.

    With *exposures* None, takes each unit's latest value over the whole
    source. With *exposures* given, takes the latest value strictly before
    `first_exposure_ts` - conditioning a segment on a value measured at or
    after exposure would bias the result, so this is the correctness
    guarantee behind `Property.as_of = 'pre_exposure'`.

    Deterministic tiebreak: at an equal timestamp the largest value wins.
    Units with no qualifying row are simply absent from the result; the
    caller's left join routes them into the `"__null__"` bin instead of
    dropping them.
    """
    props = raw.select("unit_id", "ts", property_name)

    if exposures is not None:
        # Strictly pre-exposure, mirroring unit_totals' CUPED window.
        props = props.join(
            exposures.select("unit_id", "first_exposure_ts"),
            props.unit_id == exposures.unit_id,
        )
        props = props.filter(props.ts < props.first_exposure_ts)
        props = props.select("unit_id", "ts", property_name)

    # Latest timestamp per unit (NULL ts rows never participate: the
    # self-join on equality excludes them).
    latest = props.group_by("unit_id").agg(latest_ts=props.ts.max())
    at_latest = props.join(latest, ["unit_id", props.ts == latest.latest_ts])
    at_latest = at_latest.select("unit_id", property_name)

    agg = at_latest[property_name].max()
    return at_latest.group_by("unit_id").agg(**{property_name: agg})


def canonical_dimension_value(value: ir.Value) -> ir.Value:
    """String identity of a dimension value; Booleans are lower-cased, NULL stays NULL."""
    casted = value.cast("string")
    return casted.lower() if value.type().is_boolean() else casted


def join_breakout_dimension(table: ir.Table, properties_table: ir.Table, by: list[str]) -> ir.Table:
    """Left-join a breakout dimension onto *table* and coalesce NULLs.

    *properties_table* must already carry one row per `unit_id` and must
    not be pre-filtered: unmatched *table* rows get NULL, coalesced to the
    `"__null__"` sentinel bin rather than being dropped or NULL-grouped.

    Each `by` column is cast to string before the coalesce; only
    boolean-typed columns are also lower-cased, so a genuinely string
    property's casing (e.g. a country code) is never altered. A real
    property value that is literally `"__null__"` collides with the
    sentinel - an accepted, vanishingly unlikely limitation.

    Uses an explicit `rname` because *table* may already carry a
    `unit_id_right` column from an earlier `unit_id`-keyed join.
    """
    joined = table.left_join(properties_table, "unit_id", rname="{name}_dim_right")

    def _normalized(col: str) -> ir.Value:
        return ibis.coalesce(canonical_dimension_value(joined[col]), "__null__")

    return joined.mutate(**{col: _normalized(col) for col in by})


def join_unit_covariates(
    table: ir.Table, properties_table: ir.Table, name: str, *, categorical: bool = False
) -> ir.Table:
    """Left-join one covariate onto *table* keeping NULL: a numeric
    covariate as ``float64``, a *categorical* (string property) one as
    ``string``.

    *properties_table* carries one row per `unit_id` (see
    `breakout_property_table`). Unlike `join_breakout_dimension`, nothing is
    coalesced: a missing pre-exposure value must stay NULL for
    `AdjustmentSet.missing` -- a null level is a missing covariate value,
    never a level of its own.
    """
    value = properties_table.select("unit_id", _covariate_value=properties_table[name])
    joined = table.left_join(value, "unit_id", rname="{name}_cov_right")
    target = "string" if categorical else "float64"
    return joined.select(
        *(table[column] for column in table.columns if column != name),
        **{name: value._covariate_value.cast(target)},
    )


def window_bound_stats(
    dense: ir.Table,
    metric: Metric,
) -> ir.Table:
    """Bound a dense unit-day table to each unit's own per-day analysis window.

    Matches `unit_totals`'s own per-day window-end bound, so a day-axis
    reduction (`daily_group_summary`) sees the same per-day scoping.
    Without it, a unit whose window has closed keeps contributing
    zero-valued rows for every later day, diluting the daily mean toward
    zero in the window's tail.

    Deliberately excludes `unit_totals`'s whole-unit late-enrollee
    censoring: correct for a whole-window aggregate, but wrong here, where
    a still mid-window unit must keep showing its partial data.

    `RetentionMetric` is bounded only by its band's right edge; an
    unbounded band, or a metric with no fixed `window_days`, is a no-op.

    Only bounds the day axis -- it never finalizes a `value` from the
    retained sufficient state (`n_events`/`sum_value`/`min_value`/
    `max_value`); day-grain callers do that afterwards via
    `_finalize_daily_state`, which needs to know which `RatioMetric` part
    it is finalizing.
    """
    if "first_exposure_date" in dense.columns:
        fe_date = dense.first_exposure_date
    else:
        fe_date = dense.first_exposure_ts.cast(dt.date)

    window_days = _resolve_window_days(metric)

    if isinstance(metric, ConversionMetric | MeanMetric | RatioMetric | QuantileMetric):
        if window_days is not None:
            window_end = fe_date + _days_literal(window_days).as_interval("D")
            dense = dense.filter(dense.ds < window_end)
    elif isinstance(metric, RetentionMetric):
        # Bounds only the band's right edge; the left edge (threshold) is
        # applied downstream so daily/as-of builders keep a unit in `n`.
        if window_days is not None:
            window_end = fe_date + _days_literal(window_days).as_interval("D")
            dense = dense.filter(dense.ds < window_end)
    else:
        _raise("query.builders.window_bound_stats", metric_type=type(metric).__name__)

    return dense


def _dense_unit_days(spine_tbl: ir.Table, stats_tbl: ir.Table) -> ir.Table:
    """Build the zero-filled unit-day relation used by metric aggregation."""
    joined = spine_tbl.left_join(
        stats_tbl,
        [spine_tbl.unit_id == stats_tbl.unit_id, spine_tbl.ds == stats_tbl.ds],
        rname="{name}_stats",
    )
    return joined.select(
        spine_tbl.unit_id,
        spine_tbl.experiment_id,
        spine_tbl.group_id,
        spine_tbl.first_exposure_ts,
        spine_tbl.first_exposure_date,
        spine_tbl.ds,
        n_events=ibis.coalesce(stats_tbl.n_events, 0),
        sum_value=ibis.coalesce(stats_tbl.sum_value, 0.0),
        min_value=ibis.coalesce(stats_tbl.min_value, 0.0),
        max_value=ibis.coalesce(stats_tbl.max_value, 0.0),
    )


def _observable_window_flags(
    dense: ir.Table,
    metric: Metric,
    experiment: Experiment,
    data_as_of: ir.Scalar | dt.date | dt.datetime | None,
) -> tuple[ir.BooleanValue, ir.BooleanValue, ir.Value]:
    """Return observation and maturity predicates plus their shared cutoff."""
    fe_date = dense.first_exposure_date
    final_maturity_day = _final_maturity_day(metric)
    maturity_bound = (
        fe_date
        if final_maturity_day is None
        else fe_date + _days_literal(final_maturity_day).as_interval("D")
    )
    if experiment.observation_horizon is not None:
        observable_end = ibis.literal(experiment.observation_horizon_day)
    else:
        observable_end = dense.ds.max().as_scalar()
    if data_as_of is not None:
        if isinstance(data_as_of, dt.datetime):
            as_of_expr = ibis.literal(data_as_of.date())
        elif isinstance(data_as_of, dt.date):
            as_of_expr = ibis.literal(data_as_of)
        else:
            as_of_expr = data_as_of.cast("date")
        observable_end = ibis.least(observable_end, as_of_expr)
    return (
        dense.ds.notnull(),
        cast("ir.BooleanValue", ibis.coalesce(maturity_bound <= observable_end, False)),
        observable_end,
    )


def _censor_to_observable_window(
    dense: ir.Table,
    metric: Metric,
    experiment: Experiment,
    data_as_of: ir.Scalar | dt.date | dt.datetime | None,
    *,
    warn_on_censoring: bool,
    execute: Callable[[ir.Table], list[dict[str, Any]]] | None = None,
) -> ir.Table:
    """Drop units whose metric window is not observable yet, and units
    with no observed day at all (a null-date spine row)."""
    has_observed_day, window_complete, _ = _observable_window_flags(
        dense, metric, experiment, data_as_of
    )
    dense_before_censor = dense
    kept = has_observed_day & window_complete
    dense = dense.filter(kept)
    if not warn_on_censoring:
        return dense

    flagged = dense_before_censor.mutate(_kept=kept)
    counts_query = flagged.aggregate(
        enrolled=flagged.unit_id.nunique(),
        kept=flagged.unit_id.nunique(where=flagged._kept),
    )
    if execute is None:

        def execute(query: ir.Table) -> list[dict[str, Any]]:
            return query.execute().to_dict("records")

    counts = execute(counts_query)[0]
    enrolled = int(counts["enrolled"])
    kept = int(counts["kept"])
    dropped = enrolled - kept
    if not enrolled or dropped / enrolled <= CENSOR_WARN_FRACTION:
        return dense

    horizon_date = experiment.observation_horizon_day
    if data_as_of is None:
        as_of_date = None
    elif isinstance(data_as_of, dt.datetime):
        as_of_date = data_as_of.date()
    elif isinstance(data_as_of, dt.date):
        as_of_date = data_as_of
    else:
        resolved = execute(ibis.memtable({"_": [0]}).mutate(_as_of=data_as_of))[0]["_as_of"]
        as_of_date = (
            None
            if resolved is None or resolved != resolved
            else (resolved.date() if hasattr(resolved, "date") else resolved)
        )

    if horizon_date is not None:
        if as_of_date is not None and as_of_date < horizon_date:
            cause = (
                f"the underlying data, only loaded through {as_of_date} "
                f"(data_as_of) -- this will resolve once more data "
                f"arrives, not by changing observation_end"
            )
        else:
            cause = (
                "the declared observation horizon -- if this "
                "intervention's effect persists once the experiment "
                "stops, set observation_end to extend data collection; "
                "if it reverts, this censoring is correct"
            )
    else:
        cause = (
            "the experiment still running -- no end is declared yet, "
            "so these units have not had time to mature; this will "
            "resolve as the experiment continues, not by changing "
            "any experiment field"
        )
        if isinstance(metric, RatioMetric):
            cause += (
                f"; note this bound reflects observed event dates, "
                f"not pipeline freshness -- a denominator fact "
                f"('{metric.denominator.fact}') lagging behind the "
                f"numerator ('{metric.numerator.fact}') is another "
                f"possible cause worth checking"
            )
    _warn(
        "frame.censoring.dropped_units",
        metric_name=metric.name,
        dropped=dropped,
        enrolled=enrolled,
        cause=cause,
        stacklevel=3,
    )
    return dense


def _aggregate_unit_outcome(
    dense: ir.Table,
    spine: ir.Table,
    metric: Metric,
    den_stats: ir.Table | None,
) -> ir.Table:
    """Aggregate a bounded dense relation according to its metric kind."""
    keys = dense.select("unit_id", "experiment_id", "group_id", "metric").distinct()
    match metric:
        case ConversionMetric():
            return dense.group_by(["unit_id", "experiment_id", "group_id", "metric"]).agg(
                y=(dense.n_events > 0).max().cast("int32").cast("float64"),
            )
        case RetentionMetric():
            band_start, _band_end = metric.band
            threshold_date = dense.first_exposure_date + _days_literal(band_start).as_interval("D")
            active = dense.filter(dense.ds >= threshold_date)
            active_agg = active.group_by(["unit_id", "experiment_id", "group_id", "metric"]).agg(
                y=(active.n_events > 0).max().cast("int32").cast("float64")
            )
            totals = keys.left_join(
                active_agg,
                ["unit_id", "experiment_id", "group_id", "metric"],
            ).select("unit_id", "experiment_id", "group_id", "metric", "y")
            return totals.mutate(y=ibis.coalesce(totals.y, 0.0))
        case MeanMetric() | QuantileMetric():
            return _measure_totals(keys, dense, metric.aggregation)
        case RatioMetric():
            if den_stats is None:
                _raise("query.builders.ratiometric_den_stats")
            num_totals = _measure_totals(keys, dense, metric.numerator.aggregation)
            # The numerator already consumed `spine`; a structurally equal
            # copy would collide on generated aliases (see panel_spine).
            den_spine = spine.mutate(_spine_variant=ibis.literal("ratio_denominator"))
            den_dense = _dense_unit_days(den_spine, den_stats)
            den_dense = window_bound_stats(den_dense, metric)
            den_keys = (
                den_dense.select("unit_id", "experiment_id", "group_id")
                .mutate(metric=ibis.literal(metric.name))
                .distinct()
            )
            den_totals = _measure_totals(den_keys, den_dense, metric.denominator.aggregation)
            den_totals = den_totals.select("unit_id", y_den=den_totals.y)
            totals = num_totals.join(den_totals, "unit_id")
            return totals.select("unit_id", "experiment_id", "group_id", "metric", "y", "y_den")
        case _:
            assert_never(cast(Never, metric))


def _attach_covariate(
    totals: ir.Table,
    pre_stats: ir.Table | None,
    experiment: Experiment,
) -> ir.Table:
    """Attach the CUPED covariate or its typed null placeholder."""
    if pre_stats is not None and experiment.n_pre_periods > 0:
        return join_pre_period_covariate(totals, pre_stats)
    return totals.mutate(x=ibis.null().cast("float64"))


def _uptake_events_in_elapsed_window(
    exposures: ir.Table,
    uptake_events: ir.Table,
    uptake_window_days: int | None,
) -> ir.Table:
    """Filter uptake by raw timestamp relative to each unit's exposure."""
    exposure_anchor = exposures.select("unit_id", "first_exposure_ts").distinct()
    joined = uptake_events.join(
        exposure_anchor,
        uptake_events.unit_id == exposure_anchor.unit_id,
        rname="{name}_exp",
    )
    joined = joined.filter(joined.ts >= joined.first_exposure_ts)
    if uptake_window_days is not None:
        end = joined.first_exposure_ts + _days_literal(uptake_window_days).as_interval("D")
        joined = joined.filter(joined.ts < end)
    return joined


def _attach_uptake_flag(
    totals: ir.Table,
    spine: ir.Table,
    uptake_events: ir.Table | None,
    uptake_window_days: int | None,
) -> ir.Table:
    """Attach the encouragement uptake flag to unit totals."""
    if uptake_events is None:
        return totals.mutate(d=ibis.null().cast("float64"))

    uptake_joined = _uptake_events_in_elapsed_window(spine, uptake_events, uptake_window_days)
    uptake_joined = uptake_joined.mutate(_uptake_flag=ibis.literal(1, type="int32"))
    uptake_totals = uptake_joined.group_by(["unit_id"]).agg(d=uptake_joined._uptake_flag.max())
    totals = totals.left_join(uptake_totals, "unit_id")
    return totals.mutate(d=ibis.coalesce(totals.d, 0).cast("float64"))


def _project_unit_totals(
    totals: ir.Table,
    spine: ir.Table,
    metric: Metric,
    by: list[str] | None,
    properties_table: ir.Table | None,
    cluster: str | None,
) -> ir.Table:
    """Normalize optional columns and project the public unit-total schema."""
    if not isinstance(metric, RatioMetric):
        totals = totals.mutate(y_den=ibis.null().cast("float64"))

    base_cols = ["unit_id", "experiment_id", "group_id", "metric", "y", "x", "y_den", "d"]
    if properties_table is not None:
        if not by:
            _raise("query.builders.unit_totals_properties")
        totals = join_breakout_dimension(totals, properties_table, by)
        return totals.select(*base_cols, *by)
    if by:
        _raise("query.builders.unit_totals_by")
    if cluster is not None:
        cluster_map = spine.select("unit_id", cluster).distinct()
        totals = totals.left_join(cluster_map, "unit_id")
        return totals.select(*base_cols, cluster)
    return totals.select(*base_cols)


def unit_totals(
    spine: ir.Table,
    stats: ir.Table,
    metric: Metric,
    experiment: Experiment,
    pre_stats: ir.Table | None = None,
    den_stats: ir.Table | None = None,
    uptake_events: ir.Table | None = None,
    uptake_window_days: int | None = None,
    by: list[str] | None = None,
    properties_table: ir.Table | None = None,
    data_as_of: ir.Scalar | dt.date | dt.datetime | None = None,
    # The public aggregate signature preserves distinct evidence inputs.
    warn_on_censoring: bool | Callable[[ir.Table], list[dict[str, Any]]] = True,  # noqa: FBT001, FBT002
) -> ir.Table:
    """Aggregate spine + sparse stats to one row per unit (window aggregate).

    Zero-fill comes from *spine*: every enrolled unit gets a row for every
    calendar day in its window. *stats* stays sparse. Windows, aggregation,
    retention thresholds, and censoring are all applied here.

    ``y`` is the metric-specific per-unit value across the analysis window:

    * conversion / retention - 1 if any event occurred in the window (or,
      for retention, on/after ``threshold_days``), else 0.
    * mean (sum/count/avg_event/avg_calendar_day/min/max/count_distinct) -
      sum/count delegate to ``aggregate_stats`` on the zero-filled window;
      min/max/count_distinct use observed event days only, so a zero-filled
      day never injects a phantom value; avg_event divides by event count,
      while avg_calendar_day divides by the inclusive admitted day count.
      Both averages are undefined for a zero-event window.
    * ratio - numerator and denominator each honor their OWN declared
      ``aggregation`` independently via ``_measure_totals`` (the same
      per-aggregation unit-inclusion rule as mean, applied per side), then
      joined by unit. A unit undefined on either side (e.g. an
      ``avg_event`` part with zero events) is excluded from the ratio
      entirely -- never fabricated as a ``0.0`` denominator/numerator.

    ``x`` is the CUPED pre-period covariate; ``d`` is the binary
    encouragement-uptake flag. Both are ``None`` when not materialized.

    Late-enrollee censoring: a unit is kept only once the last calendar day
    its outcome depends on is itself observable - within the declared
    observation horizon (or the spine's own extent for a running
    experiment), and before ``data_as_of`` if given. Units that fail this
    are silently dropped; a material drop share raises a warning.

    Parameters
    ----------
    den_stats : ir.Table | None
        Ratio-denominator stats; required for a ``RatioMetric``.
    data_as_of : ir.Scalar | date | datetime | None
        Latest timestamp this metric's data has actually loaded through;
        caps censoring at the earlier of this and the observation horizon.
    warn_on_censoring : bool or callable
        A callable executes bookkeeping tables and returns rows through the
        owning source. Otherwise, warn on a material drop (default True).
        Warning bookkeeping costs an extra execution round-trip.
    """
    cluster = experiment.cluster
    if cluster is not None:
        if by or properties_table is not None:
            _raise(
                "query.builders.cluster_total_grain",
                caller="unit_totals",
                grain="breakout",
                cluster=cluster,
            )
        if cluster not in spine.columns:
            _raise(
                "query.builders.cluster_column_missing",
                experiment=experiment.name,
                cluster=cluster,
                table="spine",
                remedy="build it with panel_spine over first_exposures, which both carry "
                "the declared label.",
                columns=sorted(spine.columns),
            )

    dense = _dense_unit_days(spine, stats).mutate(metric=ibis.literal(metric.name))
    dense = _censor_to_observable_window(
        dense,
        metric,
        experiment,
        data_as_of,
        warn_on_censoring=bool(warn_on_censoring),
        execute=warn_on_censoring if callable(warn_on_censoring) else None,
    )
    dense = window_bound_stats(dense, metric)
    totals = _aggregate_unit_outcome(dense, spine, metric, den_stats)

    totals = _attach_covariate(totals, pre_stats, experiment)
    totals = _attach_uptake_flag(totals, spine, uptake_events, uptake_window_days)
    return _project_unit_totals(totals, spine, metric, by, properties_table, cluster)


def emit_centered_moments(
    table: ir.Table,
    plan: MomentPlan,
    *,
    keys: Sequence[str],
    columns: Mapping[str, str],
) -> ir.Table:
    """One centered-moment row per *keys* from *table*, shaped by *plan*.

    *columns* maps each materialized plan variable and mask to its column
    in *table*; a variable or mask the plan declares but *columns* omits is
    emitted as typed NULLs, so the row shape never depends on the input.
    Two-phase aggregation: window means per group become the references,
    then one aggregate sums the residual products against them, in the
    plan's moment order (family by family, winsor passthrough, then the
    masked family). No variance reduction here; that lives on ``ArmStats``.

    A passthrough column is carried through unchanged: a ``max`` column by
    maximum, a ``sum`` column by sum, and an ``int64`` sum as an exact integer
    -- never routed through the float moments. ``successes`` is such a column:
    an exact 0/1 integer per row of a declared binary outcome, ``NULL``
    elsewhere, so a group's exact count is the integer sum of its rows.
    """
    group_cols = list(keys)
    window = ibis.window(group_by=group_cols)
    live = [variable for variable in plan.variables if variable in columns]
    masks = [mask for mask in plan.masked if mask in columns]
    ref_name = {variable: plan.name(("ref", (variable,), None)) for variable in live}
    centered = table.mutate(
        **{ref_name[variable]: table[columns[variable]].mean().over(window) for variable in live}
    )
    residual = {
        variable: centered[columns[variable]] - centered[ref_name[variable]] for variable in live
    }
    mask_value = {mask: centered[columns[mask]] for mask in masks}

    def null(dtype: str = "float64") -> ir.Value:
        return ibis.null().cast(dtype)

    def reduced(moment: Moment) -> ir.Value:
        kind, variables, mask = moment
        if any(v not in residual for v in variables) or (
            mask is not None and mask not in mask_value
        ):
            return null()
        if kind == "ref":
            return centered[ref_name[variables[0]]].max()
        if kind == "count":
            assert mask is not None  # a count is always some mask's count
            return mask_value[mask].sum()
        product: ir.Value | None = mask_value[mask] if mask is not None else None
        for variable in variables:
            term = residual[variable]
            product = term if product is None else product * term
        assert product is not None
        return product.sum()

    aggs: dict[str, ir.Value] = {plan.name(("n", (), None)): centered.count()}
    for moment in plan.unmasked_moments():
        aggs[plan.name(moment)] = reduced(moment)
    for passthrough in plan.passthrough:
        if passthrough.column in centered.columns:
            column = centered[passthrough.column]
            if passthrough.reduce == "max":
                aggs[passthrough.column] = column.max()
            elif passthrough.dtype == "int64":
                # An integer sum stays an exact integer (never rides a float, and a
                # backend's wider accumulator is narrowed back to the declared type).
                aggs[passthrough.column] = column.sum().cast("int64")
            else:
                aggs[passthrough.column] = column.sum()
        else:
            aggs[passthrough.column] = null(passthrough.dtype)
    for moment in plan.masked_moments():
        aggs[plan.name(moment)] = reduced(moment)
    summary = centered.group_by(group_cols).agg(**aggs)

    x_var = plan.x_variable()
    if x_var is None:
        return summary
    if x_var == "x":
        # A covariate is optional: the declaration mirrors its presence.
        ref_x = summary[plan.name(("ref", (x_var,), None))]
        return summary.mutate(
            x_role=ref_x.notnull().ifelse(ibis.literal(X_SLOT_ROLES[x_var]), null("string"))
        )
    return summary.mutate(x_role=ibis.literal(X_SLOT_ROLES[x_var]))


def declared_binary_metrics(metrics: Iterable[Metric]) -> list[str]:
    """Names of the declared conversion/retention metrics among *metrics*.

    Only these carry a unit-grain 0/1 outcome: a cluster sum, a transformed
    or any other metric is never binary, whatever its observed values.
    """
    return [m.name for m in metrics if isinstance(m, ConversionMetric | RetentionMetric)]


def group_summary(
    totals: ir.Table,
    by: list[str] | None = None,
    *,
    cluster: str | None = None,
    ratio_metrics: Collection[str] | None = None,
    uptake: bool = False,
    binary_metrics: Collection[str] | None = None,
) -> ir.Table:
    """Per-group centered moments from *unit_totals*.

    Emits the centered wire shape: per-group references ``ref_y``/``ref_x``/
    ``ref_den`` (group means), residual first moments ``c*1``, and second
    moments centered on those references (``c*2``, cross terms) - plus raw
    counts ``n``/``sum_d``. No variance reduction here; that lives on
    ``ArmStats``, which expands these fields analytically. Two-phase
    aggregation: window functions compute each group's reference means,
    then a second pass aggregates the residuals against them.

    Parameters
    ----------
    cluster : str | None
        The experiment's declared randomization-grain column. When set,
        the row is a two-stage collapse over cluster rows instead of unit
        rows: ``n`` is the cluster count, and the centered moments are
        taken over each cluster's per-cluster sum/size, so
        ``ArmStats.mean_y() / mean_den()`` is exactly the cluster-robust
        estimand. Total grain only - combining with *by* refuses. The
        ``x`` family carries cluster size by default; a metric named in
        *ratio_metrics* rebinds the ``den`` family to its own denominator
        total instead, and *uptake* (if set) claims the ``x`` family for
        each cluster's uptake total.
    by : list[str] | None
        Extra breakout-dimension column(s) to group by (default: none).
    ratio_metrics : Collection[str] | None
        Names of metrics in *totals* that carry their own denominator in
        ``y_den``; read only under *cluster* (see above).
    uptake : bool
        Whether *totals*' ``d`` column carries a real encouragement uptake
        fact; read only under *cluster* (see above).
    binary_metrics : Collection[str] | None
        Names of metrics in *totals* declared conversion/retention, whose unit
        ``y`` is an exact 0/1. Their rows carry the exact integer ``successes``
        count (the sum of those 0/1 values); every other metric's is NULL.
        Ignored under *cluster*: a cluster sum is not binary, so a clustered
        row's ``successes`` is always an explicit NULL.

    The row shape comes from ``_moment_plan.UNIT_GRAIN`` (or the cluster
    plans under *cluster*): ``sum_d``/``cyd``/``cy2d``/``cxd`` are always
    emitted (typed NULL when *totals* has no ``d`` column), centered on the
    overall group reference, never on the ``d = 1`` subgroup mean.
    """
    if cluster is not None:
        if by:
            _raise(
                "query.builders.cluster_total_grain",
                caller="group_summary",
                grain="breakout",
                cluster=cluster,
            )
        if cluster not in totals.columns:
            _raise(
                "query.builders.group_summary_declared",
                cluster=cluster,
                columns=sorted(totals.columns),
            )
        ratio_names = sorted(ratio_metrics or ())
        group_cols = ["experiment_id", "metric", "group_id"]
        has_d_col = "d" in totals.columns
        per_cluster_aggs: dict[str, ir.Value] = {
            "g": totals.y.sum(),
            "m": totals.count().cast("float64"),
        }
        if ratio_names:
            if "y_den" not in totals.columns:
                _raise(
                    "query.builders.group_summary_ratio",
                    ratio_names=ratio_names,
                    columns=sorted(totals.columns),
                )
            per_cluster_aggs["m_den"] = totals.y_den.sum()
        if uptake:
            if not has_d_col:
                _raise("query.builders.group_summary_uptake", columns=sorted(totals.columns))
            # Each cluster's uptake TOTAL (not a 0/1 flag) claims the x family.
            per_cluster_aggs["x"] = totals.d.sum()
        else:
            # Nothing else claims the x family, so cluster SIZE rides it - the
            # one slot that survives a ratio metric's den rebinding below.
            per_cluster_aggs["x"] = totals.count().cast("float64")
        if "winsor_n" in totals.columns:
            for field in (
                "winsor_lower_percentile",
                "winsor_upper_percentile",
                "winsor_lower_bound",
                "winsor_upper_bound",
            ):
                per_cluster_aggs[field] = totals[field].max()
            for field in ("winsor_n", "winsor_n_lower", "winsor_n_upper"):
                per_cluster_aggs[field] = totals[field].sum()
        per_cluster = totals.group_by([*group_cols, cluster]).agg(**per_cluster_aggs)
        if ratio_names:
            # Declared ratio metrics keep their OWN per-cluster denominator
            # total in the den family; every other metric keeps cluster size.
            per_cluster = per_cluster.mutate(
                m=per_cluster.metric.isin(ratio_names).ifelse(per_cluster.m_den, per_cluster.m)
            )
        plan = CLUSTER_UPTAKE_GRAIN if uptake else CLUSTER_SIZE_GRAIN
        x_var = plan.x_variable()
        assert x_var is not None
        summary = emit_centered_moments(
            per_cluster, plan, keys=group_cols, columns={"y": "g", "den": "m", x_var: "x"}
        )
        # A cluster total is never a binary outcome: keep the declared row shape with an
        # explicit NULL count rather than any sum of cluster outcomes.
        return summary.mutate(successes=ibis.null().cast("int64"))

    group_cols = ["experiment_id", "metric", "group_id", *(by or [])]
    columns = {"y": "y", "x": "x", "den": "y_den"}
    if "d" in totals.columns:
        columns["d"] = "d"
    binary = sorted(binary_metrics or ())
    if binary:
        # Unit-grain y of a declared conversion/retention metric is an exact 0/1, so
        # the integer cast loses nothing; every other metric's rows carry NULL.
        totals = totals.mutate(
            successes=totals.metric.isin(binary).ifelse(
                totals.y.cast("int64"), ibis.null().cast("int64")
            )
        )
    return emit_centered_moments(totals, UNIT_GRAIN, keys=group_cols, columns=columns)


def daily_group_summary(
    panel: ir.Table,
    *,
    metric: Metric,
    by: list[str] | None = None,
    den_panel: ir.Table | None = None,
) -> ir.Table:
    """Per-group **per-day** centered moments for monitoring.

    The outcome ``y`` is the metric's declared per-day aggregation over
    its retained sufficient state (`n_events`/`sum_value`/`min_value`/
    `max_value`) -- never a hardcoded sum; see `_finalize_daily_state`.
    An optional ``x`` is a fixed pre-exposure unit covariate. CUPED uses
    that centered x-family only when the source requested native
    pre-period moments. The day partition is otherwise unchanged.

    Same centered-moment shape as `group_summary`, but grouped by
    `(ds, experiment_id, metric, group_id)` - each day slice is its own
    partition, with its own references. `by` column(s) must already be
    present on *panel* - unlike `unit_totals`, this builder does not join
    a properties table. `metric` is required: it supplies a
    `MeanMetric`/`QuantileMetric`'s own `aggregation`, or a `RatioMetric`'s
    numerator (for *panel*) and denominator (for *den_panel*) aggregations
    applied independently, retaining the existing numerator-window
    alignment. `den_panel`, if given, is a ratio metric's denominator
    panel carrying the same sufficient state, inner-joined after finalization
    so only units defined on both sides contribute to the moments.

    The encouragement-uptake moments always stay null. When *panel*
    carries a fixed pre-period ``x`` column, the centered x-family is
    populated in the same day partition (and ``cxden``, its cross moment
    with a ratio's denominator); otherwise those fields are typed NULLs.
    Clustered and quantile day-axis CUPED refusals are enforced by the
    readout/facade callers.
    """
    group_cols = ["ds", "experiment_id", "metric", "group_id", *(by or [])]

    panel = _finalize_daily_state(panel, metric, part="numerator")
    if isinstance(metric, ConversionMetric):
        # The day's any-occurrence value is an exact 0/1 per unit: count it as an integer.
        panel = panel.mutate(successes=(panel.n_events > 0).cast("int64"))

    if den_panel is not None:
        den_final = _finalize_daily_state(den_panel, metric, part="denominator")
        den_final = den_final.select("unit_id", "experiment_id", "ds", value_den=den_final.value)
        agg_source = panel.inner_join(
            den_final,
            _panel_row_key(panel, den_final),
            rname="{name}_den",
        )
    else:
        agg_source = panel
    columns = {"y": "value"}
    if "x" in agg_source.columns:
        columns["x"] = "x"
    if den_panel is not None:
        columns["den"] = "value_den"
    return emit_centered_moments(agg_source, DAY_GRAIN, keys=group_cols, columns=columns)


def _panel_row_key(left: ir.Table, right: ir.Table) -> ir.BooleanValue:
    """Join predicate identifying one panel row on both sides.

    A panel row is identified by (experiment_id, unit_id, ds), not by
    unit_id alone: a unit_id recurring across two experiments would
    otherwise fan out, joining each experiment's row to the other's -
    the same scoping the running sums already partition by. Panels
    without the column (a bare projection) fall back to (unit_id, ds).

    The comparison is NULL-safe rather than NULL-as-wildcard.
    ``first_exposures`` coalesces the column to the experiment's own name, so
    a panel built from a spine always carries it; treating an unknown value as
    matching anything would let one such row duplicate against every named
    experiment on the other side, which is the fan-out this exists to prevent.
    """
    key = (left.unit_id == right.unit_id) & (left.ds == right.ds)
    if "experiment_id" in left.columns and "experiment_id" in right.columns:
        key = key & left.experiment_id.identical_to(right.experiment_id)
    return key


def asof_group_summary(
    panel: ir.Table,
    metric: Metric,
    by: list[str] | None = None,
    den_panel: ir.Table | None = None,
    uptake_panel: ir.Table | None = None,
    uptake_window_days: int | None = None,
    *,
    completed_windows_only: bool = False,
    _uptake_elapsed_windowed: bool = False,
    _outcome_observation_end: ir.Scalar | None = None,
) -> ir.Table:
    """Per-group **"as of day N"** centered moments.

    The outcome ``y`` is cumulative through each calendar day, combined
    from the retained per-day sufficient state
    (`n_events`/`sum_value`/`min_value`/`max_value`) under *metric*'s
    declared aggregation (or, for a ``RatioMetric``, its numerator's for
    *panel* and its denominator's for *den_panel*, independently) -- see
    `_asof_masked_value`. An optional ``x`` is a fixed pre-exposure unit
    covariate. CUPED uses that centered x-family only when the source
    requested native pre-period moments.

    Same centered-moment shape as ``daily_group_summary``, but each unit
    contributes its cumulative outcome through that day, not that single
    day's raw activity: conversion uses a cumulative any-occurrence binary
    value (MAX/OR), and retention follows its own band-gated path (see
    ``_asof_retention_value``). When *panel* carries a fixed pre-period
    ``x`` column, the centered x-family is populated after the outcome
    row-admission rules; otherwise it is NULL.

    A fixed ``window_days`` masks a unit's state to the aggregation's
    identity once its window closes (kept, not dropped, so ``n`` grows
    monotonically and never shrinks); an unwindowed metric keeps
    accumulating.

    Uptake already filtered by elapsed timestamps sets ``_uptake_elapsed_windowed``.
    Its declared window still gates completion, without a second calendar mask.
    Uptake may extend the snapshot axis beyond outcome coverage; the supplied
    outcome edge freezes outcome state and excludes still-unobserved cohorts.

    **Two series, two jobs.** This is the cumulative MONITORING series,
    which exists to catch regressions early while an experiment runs.
    ``run()``/``run_breakout()``'s whole-window estimate is the windowed
    DECISION estimand and stays the readout of record; the two diverge
    before full maturity and coincide only once every unit has matured.
    Unlike ``unit_totals``, this series does not censor a still-open unit
    away - it legitimately includes its partial contribution.

    A ``RetentionMetric`` value is provisional: absent before its band
    opens, 0 until its first in-band return, then 1, freezing once a
    bounded band closes. ``completed_windows_only=True`` restores a maturity
    gate so every admitted value is final; it raises for an unbounded band,
    which never completes.

    Parameters
    ----------
    panel : ir.Table
        Per-unit-day outcome state, with optional fixed pre-period covariates.
    metric : Metric
        Determines the per-unit window used to mask state, and the
        aggregation each part finalizes under.
    by : list[str] | None
        Additional segment columns retained in the grouped moments.
    den_panel : ir.Table | None
        Ratio denominator state joined on experiment, unit and date;
        ``None`` leaves denominator fields null.
    uptake_panel : ir.Table | None
        Uptake state joined on experiment, unit and date;
        ``None`` leaves uptake fields null.
    uptake_window_days : int | None
        Uptake-window length for masking and maturity; ``None`` is unbounded.
    completed_windows_only : bool
        See the retention paragraph above. For a non-retention metric, a
        bounded outcome window admits a unit only once ``ds >= first_exposure
        + window_days``, uptake or no uptake. An encouragement design also
        requires the uptake window bounded and gates on the later of the
        two windows. An unbounded outcome with no uptake has no completion
        gate at all.
    _uptake_elapsed_windowed : bool
        Uptake events already satisfy the elapsed-timestamp window; do not
        apply another calendar-day mask.
    _outcome_observation_end : ir.Scalar | None
        Outcome coverage edge when uptake extends the snapshot axis. Freeze
        outcome state there and exclude still-unobserved cohorts.

    Returns
    -------
    ir.Table
        Centered cumulative moments per date, experiment, metric, arm and segment.
    """
    if not isinstance(metric, ConversionMetric | MeanMetric | RatioMetric | RetentionMetric):
        _raise(
            "query.builders.asof_group_summary_metric_type_not_implemented",
            metric_type=type(metric).__name__,
        )

    group_cols = ["ds", "experiment_id", "metric", "group_id", *(by or [])]
    outcome_edge_column = None
    if _outcome_observation_end is not None:
        outcome_edge_column = "__outcome_observation_end"
        columns = panel.columns
        while outcome_edge_column in columns:
            outcome_edge_column = "_" + outcome_edge_column
        panel = panel.mutate(**{outcome_edge_column: _outcome_observation_end})
        first_observed_day = panel.first_exposure_date
        if isinstance(metric, RetentionMetric):
            first_observed_day += _days_literal(metric.band[0]).as_interval("D")
        panel = panel.filter(first_observed_day <= panel[outcome_edge_column])

    if isinstance(metric, RetentionMetric):
        if den_panel is not None:
            _raise("query.builders.asof_group_summary_den_panel_retention")
        agg_source = _asof_retention_value(
            panel, metric, completed_windows_only=completed_windows_only
        )
    elif isinstance(metric, ConversionMetric):
        agg_source = _asof_conversion_value(
            panel, _resolve_window_days(metric), observation_end_column=outcome_edge_column
        )
    else:
        numerator_aggregation = (
            metric.numerator.aggregation if isinstance(metric, RatioMetric) else metric.aggregation
        )
        agg_source = _asof_masked_value(
            panel,
            _resolve_window_days(metric),
            numerator_aggregation,
            observation_end_column=outcome_edge_column,
        )
        if den_panel is not None:
            assert isinstance(metric, RatioMetric)
            den_vals = den_panel.select(
                "unit_id",
                "experiment_id",
                "ds",
                den_n_events="n_events",
                den_sum_value="sum_value",
                den_min_value="min_value",
                den_max_value="max_value",
            )
            aligned = panel.left_join(den_vals, _panel_row_key(panel, den_vals), rname="{name}_dv")
            state_cols = [
                c
                for c in panel.columns
                if c not in ("n_events", "sum_value", "min_value", "max_value")
            ]
            aligned = aligned.select(
                *state_cols,
                n_events=ibis.coalesce(aligned.den_n_events, 0),
                sum_value=ibis.coalesce(aligned.den_sum_value, 0.0),
                min_value=ibis.coalesce(aligned.den_min_value, 0.0),
                max_value=ibis.coalesce(aligned.den_max_value, 0.0),
            )
            den_panel = _asof_masked_value(
                aligned,
                _resolve_window_days(metric),
                metric.denominator.aggregation,
                observation_end_column=outcome_edge_column,
            )
            agg_source = agg_source.inner_join(
                den_panel,
                _panel_row_key(agg_source, den_panel),
                rname="{name}_den",
            )

    # The uptake join runs after any den join and before the reductions
    # below, so every centered expression is rooted in one final agg_source.
    if uptake_panel is not None:
        masked_uptake = _asof_masked_value(
            uptake_panel, None if _uptake_elapsed_windowed else uptake_window_days, "sum"
        )
        masked_uptake = masked_uptake.mutate(
            d=(masked_uptake.asof_value > 0).cast("int32").cast("float64")
        )
        masked_uptake = masked_uptake.select("unit_id", "experiment_id", "ds", "d")
        agg_source = agg_source.left_join(
            masked_uptake,
            _panel_row_key(agg_source, masked_uptake),
        )
        agg_source = agg_source.mutate(d=ibis.coalesce(agg_source.d, 0.0))

    if completed_windows_only and not isinstance(metric, RetentionMetric):
        agg_source = _completed_asof_rows(
            agg_source,
            outcome_window_days=_resolve_window_days(metric),
            uptake_window_days=uptake_window_days,
            has_uptake=uptake_panel is not None,
        )

    if isinstance(metric, ConversionMetric | RetentionMetric):
        # The cumulative any-occurrence value is an exact 0/1 per unit: count it as an integer.
        agg_source = agg_source.mutate(successes=agg_source.asof_value.cast("int64"))
    columns = {"y": "asof_value"}
    if "x" in agg_source.columns:
        columns["x"] = "x"
    if den_panel is not None:
        columns["den"] = "asof_value_den"
    if uptake_panel is not None:
        columns["d"] = "d"
    return emit_centered_moments(agg_source, DAY_GRAIN, keys=group_cols, columns=columns)


def cohort_group_summary(
    spine: ir.Table,
    stats: ir.Table,
    metric: Metric,
    experiment: Experiment,
    pre_stats: ir.Table | None = None,
    by: list[str] | None = None,
    properties_table: ir.Table | None = None,
    data_as_of: ir.Scalar | dt.date | dt.datetime | None = None,
    # Cohort aggregation keeps its inputs and output shape aligned.
    warn_on_censoring: bool = True,  # noqa: FBT001, FBT002
) -> ir.Table:
    """Per-group centered moments keyed by the unit's own exposure date.

    The retention outcome ``y`` is indexed by exposure cohort date; the
    optional ``x`` is a fixed pre-exposure unit covariate. Native CUPED
    consumes the centered x-family only when pre-period moments are
    materialized in ``pre_stats``.

    The day axis a retention series is conventionally indexed on: a row
    keyed `2025-01-20` means "units exposed on 2025-01-20". Because the
    outcome is a lagged fill, the series ends `band_end` days before the
    data cutoff, rather than starting late (the opposite shape from
    `asof_group_summary`).

    Not a panel aggregation - delegates to `unit_totals` for the per-unit
    `y` (inheriting its band, censoring, and maturity rule unchanged) and
    regroups those rows by exposure date. Only matured cohorts survive.

    Restricted to `RetentionMetric`; every other metric type already has a
    well-defined per-day reading via `daily_group_summary`. Emits the
    day-grain row shape including ``x_role``.
    """
    if not isinstance(metric, RetentionMetric):
        _raise("query.builders.cohort_group_summary", metric_type=type(metric).__name__)
    if metric.band[1] is None:
        _raise(
            "query.builders.cohort_group_summary_unbounded_band",
            name=metric.name,
            threshold_days=metric.threshold_days,
        )

    if experiment.cluster is not None:
        _raise(
            "query.builders.cluster_total_grain",
            caller="cohort_group_summary",
            grain="cohort",
            cluster=experiment.cluster,
        )

    # first_exposure_date is dropped by unit_totals's select, so capture the
    # cohort key from the spine before delegating.
    cohort_keys = spine.select("unit_id", cohort_ds="first_exposure_date").distinct()

    totals = unit_totals(
        spine,
        stats,
        metric,
        experiment,
        pre_stats=pre_stats,
        by=by,
        properties_table=properties_table,
        data_as_of=data_as_of,
        warn_on_censoring=warn_on_censoring,
    )
    joined = totals.join(cohort_keys, "unit_id")
    totals = joined.mutate(ds=joined.cohort_ds)

    group_cols = ["ds", "experiment_id", "metric", "group_id", *(by or [])]
    # Unit totals of a retention metric are exact 0/1 values: count them as integers.
    totals = totals.mutate(successes=totals.y.cast("int64"))
    columns = {"y": "y"}
    if "x" in totals.columns:
        columns["x"] = "x"
    return emit_centered_moments(totals, DAY_GRAIN, keys=group_cols, columns=columns)


# ── Private helpers ────────────────────────────────────────────────────


def _apply_filter(table: ir.Table, flt: Filter) -> ir.Table:
    """Apply a semantic :class:`Filter` to an ibis table."""
    col = table[flt.property]
    op = flt.op
    values = flt.values

    if op == "equals":
        return table.filter(col == values[0])
    if op == "not_equals":
        return table.filter(col != values[0])
    if op == "in":
        return table.filter(col.isin(values))
    if op == "not_in":
        return table.filter(~col.isin(values))
    if op == "gt":
        return table.filter(col > values[0])
    if op == "gte":
        return table.filter(col >= values[0])
    if op == "lt":
        return table.filter(col < values[0])
    if op == "lte":
        return table.filter(col <= values[0])
    if op == "between":
        return table.filter((col >= values[0]) & (col <= values[1]))

    assert_never(op)


def _asof_masked_value(
    panel: ir.Table,
    window_days: int | None,
    aggregation: str,
    *,
    observation_end_column: str | None = None,
) -> ir.Table:
    """Combine each unit's admitted per-day sufficient state into a
    cumulative ``asof_value`` under *aggregation*, ordered by ``ds``.

    Days at/after the unit's window close (when *window_days* is given)
    are masked to the aggregation's identity -- their state never joins
    the running combination -- but the row itself is KEPT (kept, not
    dropped, so `n` grows monotonically and never shrinks: see
    ``asof_group_summary``). ``window_days=None`` means unmasked: every
    admitted day's state combines.

    Finalization matches ``_measure_totals``'s per-aggregation window-level
    rule, generalized to a running prefix: ``sum``/``count`` combine
    directly; ``avg_event`` divides the cumulative ``sum_value`` by the
    cumulative ``n_events`` and DROPS a row before any event is observed
    (undefined, 0/0 -- mirrors the window-level "unit dropped" rule);
    ``avg_calendar_day`` divides by the cumulative count of admitted
    calendar days (not events); ``min``/``max`` take the running extremum
    over observed (``n_events > 0``) days only, folded to ``0.0`` before
    any event (KEPT, mirrors the window-level convention); ``count_distinct``
    is an exact running union of observed days' ``sum_value`` (the same
    day-grain distinct target ``aggregate_stats`` uses at window grain).
    Never averages daily means, and never sums daily distinct counts
    across repeated keys.
    """
    if window_days is not None:
        if "first_exposure_date" in panel.columns:
            fe_date = panel.first_exposure_date
        else:
            fe_date = panel.first_exposure_ts.cast(dt.date)
        window_end = fe_date + _days_literal(window_days).as_interval("D")
        in_window = panel.ds < window_end
    else:
        in_window = ibis.literal(True)
    if observation_end_column is not None:
        in_window &= panel.ds <= panel[observation_end_column]

    # Partition on (experiment_id, unit_id): unit_id alone bleeds one
    # experiment's running combination into another for a unit sharing an
    # id across experiments.
    w = ibis.cumulative_window(group_by=["experiment_id", "unit_id"], order_by="ds")
    observed = in_window & (panel.n_events > 0)

    if aggregation == "sum":
        return panel.mutate(asof_value=panel.sum_value.sum(where=in_window).over(w))
    if aggregation == "count":
        return panel.mutate(asof_value=panel.n_events.sum(where=in_window).over(w).cast("float64"))
    if aggregation == "avg_calendar_day":
        cum_sum = panel.sum_value.sum(where=in_window).over(w)
        cum_days = panel.ds.count(where=in_window).over(w)
        return panel.mutate(asof_value=cum_sum / cum_days.cast("float64"))
    if aggregation == "avg_event":
        staged = panel.mutate(
            _cum_sum=panel.sum_value.sum(where=observed).over(w),
            _cum_n=panel.n_events.sum(where=in_window).over(w),
        )
        kept = staged.filter(staged._cum_n > 0)
        return kept.mutate(asof_value=kept._cum_sum / kept._cum_n.cast("float64")).drop(
            "_cum_sum", "_cum_n"
        )
    if aggregation == "min":
        return panel.mutate(
            asof_value=ibis.coalesce(panel.min_value.min(where=observed).over(w), 0.0)
        )
    if aggregation == "max":
        return panel.mutate(
            asof_value=ibis.coalesce(panel.max_value.max(where=observed).over(w), 0.0)
        )
    if aggregation == "count_distinct":
        candidate = observed.ifelse(panel.sum_value, ibis.null().cast("float64"))
        return panel.mutate(
            asof_value=ibis.coalesce(
                candidate.collect().over(w).unique().length().cast("float64"), 0.0
            )
        )
    _raise("query.builders.unknown_aggregation", aggregation=aggregation)


def _completed_asof_rows(
    table: ir.Table,
    *,
    outcome_window_days: int | None,
    uptake_window_days: int | None,
    has_uptake: bool,
) -> ir.Table:
    """Keep snapshots where every window this row depends on has closed.

    Without an uptake design, a bounded outcome window admits a unit once
    its own window has closed; an unbounded outcome has no fixed offset to
    wait out, so the table passes through unchanged. With uptake, both the
    outcome and uptake windows must be bounded, and the later of the two
    decides completion.
    """
    if has_uptake and (outcome_window_days is None or uptake_window_days is None):
        _raise("query.builders.asof_group_summary_completed_requires_uptake_window")
    if outcome_window_days is None and not has_uptake:
        return table
    assert outcome_window_days is not None
    if "first_exposure_date" in table.columns:
        first_exposure_date = table.first_exposure_date
    else:
        first_exposure_date = table.first_exposure_ts.cast(dt.date)
    completion_date = first_exposure_date + _days_literal(
        max(outcome_window_days, uptake_window_days or 0)
    ).as_interval("D")
    return table.filter(table.ds >= completion_date)


def _asof_conversion_value(
    panel: ir.Table, window_days: int | None, *, observation_end_column: str | None = None
) -> ir.Table:
    """Cumulative any-occurrence conversion state, never a click count."""
    cumulative = _asof_masked_value(
        panel, window_days, "sum", observation_end_column=observation_end_column
    )
    return cumulative.mutate(asof_value=(cumulative.asof_value > 0).cast("int32").cast("float64"))


def _asof_retention_value(
    panel: ir.Table, metric: RetentionMetric, *, completed_windows_only: bool = False
) -> ir.Table:
    """Add a binary `asof_value` for a retention metric, and gate the panel
    to units whose observation band has opened.

    Three steps: mask `value` to the half-open band `[fe + band_start,
    fe + band_end)` (or `[fe + band_start, inf)` if unbounded); running-sum
    the masked value per unit, then binarise it (retention is a MAX over
    the band, not a SUM); gate to `ds >= fe + band_start` by default, so a
    unit is absent (never zero-filled) before its band opens and
    provisional - 0 then 1, ratcheting upward, freezing at a bounded
    band's close - after.

    `completed_windows_only=True` moves the gate to `ds >= fe + band_end`:
    a unit enters only once its outcome is final. Raises `ValueError` if
    combined with an unbounded band, which never completes.
    """
    band_start_days, band_end_days = metric.band
    if completed_windows_only and band_end_days is None:
        _raise(
            "query.builders.asof_group_summary",
            name=metric.name,
            threshold_days=metric.threshold_days,
        )

    if "first_exposure_date" in panel.columns:
        fe_date = panel.first_exposure_date
    else:
        fe_date = panel.first_exposure_ts.cast(dt.date)

    band_start = fe_date + _days_literal(band_start_days).as_interval("D")
    if band_end_days is not None:
        band_end = fe_date + _days_literal(band_end_days).as_interval("D")
        in_band = (panel.ds >= band_start) & (panel.ds < band_end)
        gate = band_end if completed_windows_only else band_start
    else:
        in_band = panel.ds >= band_start
        gate = band_start

    panel = panel.mutate(masked_value=in_band.ifelse(panel.sum_value, ibis.literal(0.0)))
    # Partition on (experiment_id, unit_id): see _asof_masked_value.
    w = ibis.cumulative_window(group_by=["experiment_id", "unit_id"], order_by="ds")
    panel = panel.mutate(
        asof_value=(panel.masked_value.sum().over(w) > 0).cast("int32").cast("float64")
    )
    return panel.filter(panel.ds >= gate)
