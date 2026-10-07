"""Ibis builders for calendar-period metric reports.

The experiment builders in :mod:`increment.query.builders` anchor every
window on ``first_exposure_ts``; a report has no exposure. These builders
anchor on the calendar instead: a period spine, per-period aggregation,
and as-of-period-end dimensions. Event resolution (``metric_events``) and
per-unit-day sufficient stats (``unit_day_stats``) are shared with the
experiment path.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import Literal, cast

import ibis
import ibis.expr.datatypes as dtypes
import ibis.expr.types as ir

from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.query.builders import (
    UNKNOWN_AGGREGATION,
    _days_literal,
    _local_date_at_offset,
)

PeriodGrain = Literal["day", "week", "month"]
WeekStart = Literal["monday", "sunday"]

_ONE_DAY = _days_literal(1).as_interval("D")

# Explicit ranges above the fixed calendar capacity refuse; scalar edges are
# unchecked here and must be admitted by Report before they reach this builder.
_MAX_REPORT_DAYS = 3653  # 10 years


def _offset_spine(n: int | ir.IntegerValue, *, tag: str) -> ir.Table:
    """A one-column table of ``day_offset`` values ``0..n-1``.

    Relational ``Table.unnest`` avoids the ``_u``/``_u_2`` alias pair
    sqlglot emits for a projection-embedded unnest, which collides on
    Snowflake. ``tag`` names the call site so two same-``n`` spines stay
    structurally distinct; keep it a stable string, not a generated one.
    """
    base = ibis.range(0, n).name("_day_offsets").as_table().mutate(_spine_tag=ibis.literal(tag))
    spine = base.unnest("_day_offsets")
    return spine.select(day_offset=spine["_day_offsets"])


_REPORT_RANGE = RefusalSpec(
    "query.calendar.range",
    CapabilityError,
    lambda *, start, end_edge, span: (
        f"report range {start}..{end_edge} spans {span} days, beyond "
        f"the {_MAX_REPORT_DAYS}-day spine capacity -- split the "
        f"report or use a coarser date range"
    ),
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "query.calendar.period_start_unknown_grain": "unknown grain {grain!r}",
        "query.calendar.population_unit_id": "population must have a 'unit_id' column; got {columns}",
        "query.calendar.period_moments_by": "period_moments 'by' table must carry exactly one dimension column besides unit_id/period; got {dim_cols}",
    },
)
_raise = raiser(_REFUSALS)


def validate_report_population(population: ir.Table | None) -> None:
    """Admit the population key before a report performs any data query."""
    if population is not None and "unit_id" not in population.columns:
        _raise("query.calendar.population_unit_id", columns=list(population.columns))


def report_horizon_relation(
    facts: Sequence[tuple[ir.Table, str]],
    *,
    day_boundary_offset: dt.timedelta,
    no_data_sentinel: dt.date,
) -> ir.Table:
    """Build a one-row bound aggregate query for selected fact horizons.

    Each fact's localized maximum is reduced independently, then the tiny
    aggregate relations are cross-joined. This supports ratio metrics whose
    components come from different warehouse relations without registering
    client-side rows or fetching fact data.
    """
    horizons = []
    for index, (fact_table, fact_name) in enumerate(facts):
        scoped = fact_table.filter(fact_table.event == fact_name)
        horizon = _local_date_at_offset(scoped.ts, day_boundary_offset).max()
        horizons.append(
            scoped.aggregate(
                **{f"_horizon_{index}": ibis.coalesce(horizon, ibis.literal(no_data_sentinel))}
            )
        )
    relation = horizons[0]
    for next_horizon in horizons[1:]:
        relation = relation.cross_join(next_horizon)
    names = [f"_horizon_{index}" for index in range(len(horizons))]
    if len(names) == 1:
        return relation.select(end_edge=relation[names[0]])
    return relation.select(end_edge=ibis.least(*(relation[name] for name in names)))


def period_start_expr(ds: ir.DateValue, grain: PeriodGrain, week_start: WeekStart) -> ir.DateValue:
    """Map a date column to the first day of its calendar period."""
    if grain == "day":
        return ds
    if grain == "week":
        if week_start == "monday":
            return cast("ir.DateValue", ds.truncate("W").cast("date"))
        # Sunday-start: shift into the Monday-based week, truncate, shift back.
        return cast("ir.DateValue", ((ds + _ONE_DAY).truncate("W") - _ONE_DAY).cast("date"))
    if grain == "month":
        return cast("ir.DateValue", ds.truncate("M").cast("date"))
    _raise("query.calendar.period_start_unknown_grain", grain=grain)


def _period_end_expr(period: ir.DateValue, grain: PeriodGrain) -> ir.DateValue:
    """Calendar end (inclusive last day) of the period starting at *period*."""
    if grain == "day":
        return period
    if grain == "week":
        return cast("ir.DateValue", (period + _days_literal(6).as_interval("D")).cast("date"))
    if grain == "month":
        return cast("ir.DateValue", (period + ibis.interval(months=1) - _ONE_DAY).cast("date"))
    _raise("query.calendar.period_start_unknown_grain", grain=grain)


def _admitted_day_counts(
    *,
    grain: PeriodGrain,
    week_start: WeekStart,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
) -> ir.Table:
    """Count requested inclusive calendar days in each reporting period."""
    offsets = _offset_spine(_MAX_REPORT_DAYS, tag="admitted_day_counts")
    edge = ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge
    days = offsets.mutate(
        ds=(ibis.literal(start) + offsets.day_offset.as_interval("D")).cast("date")
    )
    days = days.filter(days.ds <= edge)
    days = days.mutate(period=period_start_expr(days.ds, grain, week_start))
    return days.group_by("period").agg(n_days=days.count())


def calendar_periods(
    start: dt.date,
    *,
    end_edge: ir.Scalar | dt.date,
    data_horizon: ir.Scalar,
    grain: PeriodGrain,
    week_start: WeekStart,
) -> ir.Table:
    """Dense period spine covering ``[start, end_edge]``.

    Columns: ``period`` (start date), ``period_end`` (calendar end,
    inclusive), ``period_complete``. A period is complete only when the
    requested range covers it entirely (``period >= start`` and
    ``period_end <= end_edge``) AND the warehouse has data through its
    end (``period_end <= data_horizon``) - a trailing period during
    warehouse lag is incomplete even if the calendar has moved on.

    Raises :class:`CapabilityError` when the inclusive ``[start, end_edge]``
    span exceeds ``_MAX_REPORT_DAYS``. Scalar edges are bounded by the same
    fixed spine without range admission; the Report facade resolves its
    default scalar horizon to a date and validates it before query construction.
    """
    if isinstance(end_edge, dt.date):
        span = (end_edge - start).days + 1
        if span > _MAX_REPORT_DAYS:
            refuse(_REPORT_RANGE, start=start, end_edge=end_edge, span=span)
    start_lit = ibis.literal(start)
    edge = ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge
    offsets = _offset_spine(_MAX_REPORT_DAYS, tag="calendar_periods")
    days = offsets.mutate(ds=(start_lit + offsets.day_offset.as_interval("D")).cast("date"))
    days = days.filter(days.ds <= edge)
    periods = days.select(period=period_start_expr(days.ds, grain, week_start)).distinct()
    periods = periods.mutate(period_end=_period_end_expr(periods.period, grain))
    return periods.mutate(
        period_complete=(
            (periods.period >= start_lit)
            & (periods.period_end <= edge)
            & (periods.period_end <= data_horizon)
        )
    )


def period_unit_values(
    stats: ir.Table,
    *,
    aggregation: str,
    grain: PeriodGrain,
    week_start: WeekStart,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
) -> ir.Table:
    """Collapse ``unit_day_stats`` output to one ``y`` per (unit, period).

    ``avg_event`` means per-event (``sum_value.sum() / n_events.sum()``)
    over observed events only. ``avg_calendar_day`` means each unit's
    qualifying sum divided by the inclusive admitted calendar-day count,
    including zero-event days. ``min``/``max``/``count_distinct`` range over
    observed days only (this table is sparse - no zero-filled days exist to
    pollute them).
    """
    start_lit = ibis.literal(start)
    edge = ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge
    scoped = stats.filter((stats.ds >= start_lit) & (stats.ds <= edge))
    scoped = scoped.mutate(period=period_start_expr(scoped.ds, grain, week_start))
    g = scoped.group_by(["unit_id", "period"])
    if aggregation == "conversion":
        return g.agg(y=(scoped.n_events.sum() > 0).cast("int32").cast("float64"))
    if aggregation == "sum":
        return g.agg(y=scoped.sum_value.sum())
    if aggregation == "count":
        return g.agg(y=scoped.n_events.sum().cast("float64"))
    if aggregation == "avg_event":
        return g.agg(y=scoped.sum_value.sum() / scoped.n_events.sum())
    if aggregation == "avg_calendar_day":
        sums = g.agg(y=scoped.sum_value.sum())
        days = _admitted_day_counts(
            grain=grain, week_start=week_start, start=start, end_edge=end_edge
        )
        joined = sums.join(days, ["period"])
        return joined.select("unit_id", "period", y=joined.y / joined.n_days)
    if aggregation == "min":
        return g.agg(y=scoped.min_value.min())
    if aggregation == "max":
        return g.agg(y=scoped.max_value.max())
    if aggregation == "count_distinct":
        return g.agg(y=scoped.sum_value.nunique().cast("float64"))
    refuse(UNKNOWN_AGGREGATION, aggregation=aggregation)


def period_population(
    fact_table: ir.Table,
    *,
    grain: PeriodGrain,
    week_start: WeekStart,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    periods: ir.Table,
    population: ir.Table | None = None,
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """One row per (unit, period): the metric's denominator population.

    Default: distinct units with >=1 event of ANY fact on *fact_table*
    in the period, bucketed under *day_boundary_offset* (see
    :func:`increment.query.builders._local_date_at_offset`).
    ``population`` overrides - ``(unit_id)`` is static (every period),
    ``(unit_id, ds)`` is time-varying (member during the period
    containing ``ds``; ``ds`` is assumed already localized upstream, so
    it is used as-is). A NULL ``unit_id`` is excluded from every branch:
    unfiltered, it would join no metric value and be zero-filled by
    :func:`period_moments`, inflating the population denominator for a
    row that can never resolve.
    """
    if population is not None:
        validate_report_population(population)
        cols = population.columns
        population = population.filter(population.unit_id.notnull())
        if "ds" in cols:
            scoped = population.filter(
                (population.ds >= ibis.literal(start))
                & (
                    population.ds
                    <= (ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge)
                )
            )
            return scoped.select(
                "unit_id",
                period=period_start_expr(scoped.ds.cast(dtypes.date), grain, week_start),
            ).distinct()
        return population.select("unit_id").distinct().cross_join(periods.select("period"))
    start_lit = ibis.literal(start)
    edge = ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge
    days = fact_table.filter(fact_table.unit_id.notnull())
    days = days.mutate(ds=_local_date_at_offset(days.ts, day_boundary_offset))
    days = days.filter((days.ds >= start_lit) & (days.ds <= edge))
    return days.select("unit_id", period=period_start_expr(days.ds, grain, week_start)).distinct()


def period_moments(
    values: ir.Table,
    population: ir.Table,
    periods: ir.Table,
    *,
    zero_fill: bool,
    by: ir.Table | None = None,
) -> ir.Table:
    """Per-period CENTERED moments ``(n, ref_y, cy1, cy2)`` over the population.

    Same two-phase shape as every ``*group_summary`` (``builders.py``):
    phase 1 computes each period's reference mean ``ref_y`` with a window
    function, phase 2 aggregates the residuals against it - the second
    moment is centered while the raw values are still alive, so no
    ``n * mean**2`` term is ever formed for a reduction to cancel back
    off. ``cy1 = sum(y - ref_y)`` keeps the first moment exactly
    recoverable (``sum(y) == n * ref_y + cy1``); re-graining (week
    moments to month moments) goes through the combination identities on
    :class:`~increment.estimation.armstats.ArmStats`, never naive
    summation. The facade derives mean/var/CI in its final select via
    the analytic expansions.

    ``zero_fill=True`` (sum/count/count_distinct/conversion/avg_calendar_day):
    every population unit contributes, missing ``y`` counted as 0.
    ``zero_fill=False`` (avg_event/min/max): only units with events,
    intersected with the population. Dense over periods: an empty period
    keeps its row with ``n=0`` and NULL centered fields. ``by`` (optional,
    from :func:`asof_property_table`) has columns ``unit_id``, ``period``,
    and one dimension column; NULLs are surfaced as ``'<missing>'``.
    """
    if zero_fill:
        joined = population.left_join(values, ["unit_id", "period"]).select(
            "unit_id", "period", y=ibis.coalesce(values.y, 0.0)
        )
    else:
        joined = values.semi_join(population, ["unit_id", "period"]).select(
            "unit_id", "period", "y"
        )
    keys = ["period"]
    if by is not None:
        dim_cols = [c for c in by.columns if c not in ("unit_id", "period")]
        if len(dim_cols) != 1:
            _raise("query.calendar.period_moments_by", dim_cols=dim_cols)
        dim = dim_cols[0]
        joined = joined.left_join(by, ["unit_id", "period"]).select(
            "unit_id",
            "period",
            "y",
            **{dim: ibis.coalesce(by[dim].cast("string"), "<missing>")},
        )
        keys = ["period", dim]
    centered = joined.mutate(ref_y=joined.y.mean().over(ibis.window(group_by=keys)))
    dy = centered.y - centered.ref_y
    moments = centered.group_by(keys).agg(
        n=centered.count(),
        ref_y=centered.ref_y.max(),
        cy1=dy.sum(),
        cy2=(dy * dy).sum(),
    )
    out = periods.select("period", "period_complete").left_join(moments, ["period"])
    return out.select(
        "period",
        "period_complete",
        *[k for k in keys if k != "period"],
        n=ibis.coalesce(moments.n, 0),
        ref_y=moments.ref_y,
        cy1=moments.cy1,
        cy2=moments.cy2,
    )


def ratio_period_values(
    num_values: ir.Table,
    den_values: ir.Table,
    population: ir.Table,
    periods: ir.Table,
) -> ir.Table:
    """Per-period ratio point estimate: sum(num y) / sum(den y) over the
    population. No interval - v1 reports ratios point-only."""
    num = population.left_join(num_values, ["unit_id", "period"]).select(
        "unit_id", "period", num_y=ibis.coalesce(num_values.y, 0.0)
    )
    den = population.left_join(den_values, ["unit_id", "period"]).select(
        "unit_id", "period", den_y=ibis.coalesce(den_values.y, 0.0)
    )
    both = num.join(den, ["unit_id", "period"])
    agg = both.group_by("period").agg(
        n=both.count(),
        value=both.num_y.sum() / both.den_y.sum().nullif(0),
    )
    out = periods.select("period", "period_complete").left_join(agg, ["period"])
    return out.select(
        "period",
        "period_complete",
        n=ibis.coalesce(agg.n, 0),
        value=agg.value,
    )


def _scope_events_to_periods(
    events: ir.Table,
    *,
    grain: PeriodGrain,
    week_start: WeekStart,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Filter events to [start, end_edge] and stamp their period, bucketed
    under *day_boundary_offset*."""
    edge = ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge
    scoped = events.mutate(ds=_local_date_at_offset(events.ts, day_boundary_offset))
    scoped = scoped.filter((scoped.ds >= ibis.literal(start)) & (scoped.ds <= edge))
    return scoped.mutate(period=period_start_expr(scoped.ds, grain, week_start))


def total_period_values(
    events: ir.Table,
    periods: ir.Table,
    *,
    aggregation: str,
    grain: PeriodGrain,
    week_start: WeekStart,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    by: Sequence[str] = (),
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Per-period bare aggregate of event values - no unit layer.

    Dims are event-level: rows group by the event's own property value
    ("revenue by country" = where the purchase happened).

    Undimensioned output is dense over periods: zero-event periods emit 0
    for sum/count/count_distinct/avg_calendar_day and NULL for
    avg_event/min/max (an average over no events is undefined, except that
    calendar-day averages include an all-zero day interval).
    """
    scoped = _scope_events_to_periods(
        events,
        grain=grain,
        week_start=week_start,
        start=start,
        end_edge=end_edge,
        day_boundary_offset=day_boundary_offset,
    )
    keys = ["period", *by]
    g = scoped.group_by(keys)
    if aggregation == "sum":
        agg = g.agg(value=scoped.value.sum())
    elif aggregation == "count":
        agg = g.agg(value=scoped.count().cast("float64"))
    elif aggregation == "count_distinct":
        # Distinct raw event values - events are pre-daily-summing here,
        # so the aggregate_stats daily-sum caveat does not apply.
        agg = g.agg(value=scoped.value.nunique().cast("float64"))
    elif aggregation == "avg_event":
        agg = g.agg(value=scoped.value.mean())
    elif aggregation == "avg_calendar_day":
        agg = g.agg(value=scoped.value.sum())
        days = _admitted_day_counts(
            grain=grain, week_start=week_start, start=start, end_edge=end_edge
        )
        agg = agg.join(days, ["period"]).mutate(value=agg.value / days.n_days)
    elif aggregation == "min":
        agg = g.agg(value=scoped.value.min())
    elif aggregation == "max":
        agg = g.agg(value=scoped.value.max())
    else:
        refuse(UNKNOWN_AGGREGATION, aggregation=aggregation)

    if by:
        # Sparse: inner-join periods only to attach period_complete.
        out = agg.join(periods.select("period", "period_complete"), ["period"])
        return out.select("period", "period_complete", *by, "value")
    out = periods.select("period", "period_complete").left_join(agg, ["period"])
    zero_fills = aggregation in ("sum", "count", "count_distinct", "avg_calendar_day")
    value = ibis.coalesce(agg.value, 0.0) if zero_fills else agg.value
    return out.select("period", "period_complete", value=value)


def active_period_values(
    events: ir.Table,
    periods: ir.Table,
    *,
    grain: PeriodGrain,
    week_start: WeekStart,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    by: Sequence[str] = (),
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Distinct entities with >=1 qualifying event per period (+dims).

    Exact entity dedup on ``unit_id`` - immune to the distinct
    daily-summed-values caveat on ``aggregate_stats``'s count_distinct.
    Undimensioned: dense, empty periods emit 0. Dimensioned: sparse, like
    :func:`total_period_values`.
    """
    scoped = _scope_events_to_periods(
        events,
        grain=grain,
        week_start=week_start,
        start=start,
        end_edge=end_edge,
        day_boundary_offset=day_boundary_offset,
    )
    agg = scoped.group_by(["period", *by]).agg(value=scoped.unit_id.nunique())
    if by:
        out = agg.join(periods.select("period", "period_complete"), ["period"])
        return out.select("period", "period_complete", *by, "value")
    out = periods.select("period", "period_complete").left_join(agg, ["period"])
    return out.select("period", "period_complete", value=ibis.coalesce(agg.value, 0))


def asof_property_table(
    prop_events: ir.Table,
    periods: ir.Table,
    *,
    dim: str,
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Per-(unit, period) dimension value AS OF the period's end.

    The experiment-breakout rule (latest over the whole source, D11 on
    ``unit_totals``) reads future observations - rerunning a report
    after new data arrives would relabel completed periods. Bounding the
    lookup at ``period_end`` keeps completed periods reproducible.

    Scale: raw events are collapsed to one row per (unit, day) first -
    the join against the period spine sees |units x observation-days|
    rows, never |raw events x periods|. The join itself is an inequality
    join (obs day <= period_end) followed by a latest-day pick.

    Deterministic tiebreak: two observations at the identical timestamp
    resolve to the LARGEST value (taken after a max-``ts`` window, same
    rule as :func:`increment.query.builders.breakout_property_table`).
    ``order_by`` feeding ``distinct(keep=...)`` is NOT a SQL contract;
    physical row order must never pick the label.
    """
    # 1. Latest observation per (unit, day): collapses raw event volume.
    #    Max-ts window first, then max-value tiebreak at an equal ts.
    obs = prop_events.mutate(obs_ds=_local_date_at_offset(prop_events.ts, day_boundary_offset))
    obs = obs.select("unit_id", "obs_ds", "ts", dim)
    latest_ts = obs.group_by(["unit_id", "obs_ds"]).agg(latest_ts=obs.ts.max())
    at_latest = obs.join(latest_ts, ["unit_id", "obs_ds", obs.ts == latest_ts.latest_ts])
    daily = at_latest.group_by(["unit_id", "obs_ds"]).agg(**{dim: at_latest[dim].max()})
    # 2. As-of join: keep the latest observation day at or before the period's
    #    end. `daily` has one row per (unit_id, obs_ds), so exactly one row
    #    survives per (unit, period).
    p = periods.select("period", "period_end")
    joined = daily.cross_join(p)
    joined = joined.filter(joined.obs_ds <= joined.period_end)
    last_day = joined.group_by(["unit_id", "period"]).agg(last_ds=joined.obs_ds.max())
    latest = joined.join(last_day, ["unit_id", "period", joined.obs_ds == last_day.last_ds])
    return latest.select("unit_id", "period", dim)


def _rolling_spine(periods: ir.Table, window: int) -> ir.Table:
    """Periods with each row's trailing-window left edge attached."""
    return periods.mutate(
        window_start=(periods.period_end - _days_literal(window - 1).as_interval("D")).cast(
            dtypes.date
        )
    )


def _rolling_scope(
    events: ir.Table,
    *,
    window: int,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Day-stamped events for the rolling lookback: [start-(window-1), edge]."""
    edge = ibis.literal(end_edge) if isinstance(end_edge, dt.date) else end_edge
    lookback = ibis.literal(start - dt.timedelta(days=window - 1))
    scoped = events.mutate(ds=_local_date_at_offset(events.ts, day_boundary_offset))
    return scoped.filter((scoped.ds >= lookback) & (scoped.ds <= edge))


def rolling_active_values(
    events: ir.Table,
    periods: ir.Table,
    *,
    window: int,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    by: Sequence[str] = (),
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Distinct entities in the trailing ``window`` days ending at each
    row's ``period_end`` - rolling WAU/MAU. Windows overlap: one event
    contributes to up to ``window`` output rows, so values are NOT
    additive across rows (that non-additivity is exactly why this cannot
    be derived from calendar buckets).

    Scale discipline (same as :func:`asof_property_table`): events
    collapse to one row per (unit, day[, dims]) BEFORE the range join.
    The reporting spine's grain/week-start shape is already baked into
    *periods*; the window here is always in days.
    """
    scoped = _rolling_scope(
        events,
        window=window,
        start=start,
        end_edge=end_edge,
        day_boundary_offset=day_boundary_offset,
    )
    daily = scoped.select("unit_id", "ds", *by).distinct()
    p = _rolling_spine(periods, window)
    j = daily.cross_join(p.select("period", "period_complete", "window_start", "period_end"))
    j = j.filter((j.ds >= j.window_start) & (j.ds <= j.period_end))
    agg = j.group_by(["period", *by]).agg(value=j.unit_id.nunique())
    if by:
        out = agg.join(periods.select("period", "period_complete"), ["period"])
        return out.select("period", "period_complete", *by, "value")
    out = periods.select("period", "period_complete").left_join(agg, ["period"])
    return out.select("period", "period_complete", value=ibis.coalesce(agg.value, 0))


def rolling_total_values(
    events: ir.Table,
    periods: ir.Table,
    *,
    window: int,
    aggregation: str,
    start: dt.date,
    end_edge: ir.Scalar | dt.date,
    by: Sequence[str] = (),
    day_boundary_offset: dt.timedelta = dt.timedelta(0),
) -> ir.Table:
    """Bare aggregate over the trailing ``window`` days ending at each
    row's ``period_end``. Same overlap/non-additivity caveat as
    :func:`rolling_active_values`.

    Pre-aggregation before the range join: sum/count/avg_event/
    avg_calendar_day/min/max collapse to per-(day, dims) partials;
    count_distinct dedupes to distinct (day, value, dims) rows (duplicates
    within a day collapse safely; distinctness over the window is preserved).
    """
    scoped = _rolling_scope(
        events,
        window=window,
        start=start,
        end_edge=end_edge,
        day_boundary_offset=day_boundary_offset,
    )
    keys = ["ds", *by]
    if aggregation == "count_distinct":
        daily = scoped.select(*keys, "value").distinct()
    else:
        daily = scoped.group_by(keys).agg(
            n=scoped.count(),
            s=scoped.value.sum(),
            mn=scoped.value.min(),
            mx=scoped.value.max(),
        )
    p = _rolling_spine(periods, window)
    j = daily.cross_join(p.select("period", "period_complete", "window_start", "period_end"))
    j = j.filter((j.ds >= j.window_start) & (j.ds <= j.period_end))
    g = j.group_by(["period", *by])
    if aggregation == "sum":
        agg = g.agg(value=j.s.sum())
    elif aggregation == "count":
        agg = g.agg(value=j.n.sum().cast("float64"))
    elif aggregation == "count_distinct":
        agg = g.agg(value=j.value.nunique().cast("float64"))
    elif aggregation == "avg_event":
        agg = g.agg(value=j.s.sum() / j.n.sum())
    elif aggregation == "avg_calendar_day":
        agg = g.agg(value=j.s.sum())
        agg = agg.mutate(value=agg.value / ibis.literal(window))
    elif aggregation == "min":
        agg = g.agg(value=j.mn.min())
    elif aggregation == "max":
        agg = g.agg(value=j.mx.max())
    else:
        refuse(UNKNOWN_AGGREGATION, aggregation=aggregation)
    if by:
        out = agg.join(periods.select("period", "period_complete"), ["period"])
        return out.select("period", "period_complete", *by, "value")
    out = periods.select("period", "period_complete").left_join(agg, ["period"])
    zero_fills = aggregation in ("sum", "count", "count_distinct", "avg_calendar_day")
    value = ibis.coalesce(agg.value, 0.0) if zero_fills else agg.value
    return out.select("period", "period_complete", value=value)
