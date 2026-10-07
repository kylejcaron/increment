"""Calendar-period builders for reports. DuckDB in-memory only."""

import datetime as dt
from typing import Literal, cast

import ibis
import ibis.expr.types as ir
import pytest

from increment.errors import InvalidRequestError
from increment.query.builders import (
    aggregate_stats,
    day_boundary_offset,
    metric_events,
    unit_day_stats,
)
from increment.query.calendar import (
    PeriodGrain,
    WeekStart,
    _period_end_expr,
    active_period_values,
    asof_property_table,
    calendar_periods,
    period_moments,
    period_population,
    period_start_expr,
    period_unit_values,
    ratio_period_values,
    rolling_active_values,
    rolling_total_values,
    total_period_values,
)
from increment.semantics.models import ActiveMetric, ConversionMetric, MeanMetric, TotalMetric

# Every test here gets its own fresh, isolated `con` (see the fixture
# below) - tables can never leak between tests. Exempt the whole file
# rather than decorating the many fixtures that build one.
pytestmark = pytest.mark.creates_tables


@pytest.fixture
def con():
    return ibis.duckdb.connect()


def _periods(con, **kw):
    defaults = {
        "end_edge": dt.date(2026, 1, 31),
        "data_horizon": ibis.literal(dt.date(2026, 1, 31)),
        "grain": "week",
        "week_start": "monday",
    }
    defaults.update(kw)
    expr = calendar_periods(dt.date(2026, 1, 1), **defaults)
    # DuckDB's pandas conversion round-trips dates as pandas Timestamp, not datetime.date; use the pyarrow round-trip for real date objects.
    return con.to_pyarrow(expr.order_by("period")).to_pylist()


def test_week_grain_monday_start(con):
    rows = _periods(con)
    # 2026-01-01 is a Thursday; its ISO week starts Monday 2025-12-29.
    assert rows[0]["period"] == dt.date(2025, 12, 29)
    assert rows[0]["period_end"] == dt.date(2026, 1, 4)
    # Leading clipped period is incomplete.
    assert not rows[0]["period_complete"]
    # A fully interior week is complete.
    full = [r for r in rows if r["period"] == dt.date(2026, 1, 5)][0]
    assert full["period_complete"]
    assert full["period_end"] == dt.date(2026, 1, 11)


def test_week_grain_sunday_start(con):
    rows = _periods(con, week_start="sunday")
    # Week containing Thu 2026-01-01 starts Sunday 2025-12-28.
    assert rows[0]["period"] == dt.date(2025, 12, 28)
    assert rows[0]["period_end"] == dt.date(2026, 1, 3)
    # Leading clipped period is incomplete.
    assert not rows[0]["period_complete"]
    # A fully interior week is complete.
    full = [r for r in rows if r["period"] == dt.date(2026, 1, 4)][0]
    assert full["period_complete"]
    assert full["period_end"] == dt.date(2026, 1, 10)


def test_end_edge_as_ibis_scalar_horizon_bound(con):
    # Production usage passes end_edge as a computed ibis scalar (e.g. a source's max event date), not a python date; exercises the isinstance(end_edge, dt.date)==False branch without a real table.
    base_date = cast("ir.DateScalar", ibis.literal(dt.date(2026, 1, 20)))
    scalar_edge = cast("ir.DateScalar", base_date + ibis.interval(days=0))
    rows = _periods(
        con,
        end_edge=scalar_edge,
        data_horizon=scalar_edge,
        grain="week",
        week_start="monday",
    )
    # Same Jan 1 start as every other test: leading week starts Dec 29.
    assert rows[0]["period"] == dt.date(2025, 12, 29)
    assert not rows[0]["period_complete"]
    # Interior week (Jan 5-11) is fully inside [start, scalar horizon Jan 20].
    full = [r for r in rows if r["period"] == dt.date(2026, 1, 5)][0]
    assert full["period_complete"]
    # Trailing week (Jan 19-25) is clipped by the scalar horizon (Jan 20).
    trailing = [r for r in rows if r["period"] == dt.date(2026, 1, 19)][0]
    assert not trailing["period_complete"]
    assert rows[-1]["period"] == dt.date(2026, 1, 19)  # spine stops at the horizon


def test_month_grain(con):
    rows = _periods(
        con,
        grain="month",
        end_edge=dt.date(2026, 3, 15),
        data_horizon=ibis.literal(dt.date(2026, 3, 15)),
    )
    assert [r["period"] for r in rows] == [
        dt.date(2026, 1, 1),
        dt.date(2026, 2, 1),
        dt.date(2026, 3, 1),
    ]
    # March is clipped by end_edge -> incomplete.
    assert [r["period_complete"] for r in rows] == [True, True, False]


def test_day_grain_completeness_tracks_horizon(con):
    rows = _periods(
        con,
        grain="day",
        end_edge=dt.date(2026, 1, 5),
        data_horizon=ibis.literal(dt.date(2026, 1, 3)),  # warehouse 2 days behind
    )
    assert len(rows) == 5
    assert [r["period_complete"] for r in rows] == [True, True, True, False, False]


def test_range_beyond_capacity_refused(con):
    from increment.errors import CapabilityError

    with pytest.raises(CapabilityError) as exc_info:
        _periods(con, end_edge=dt.date(2036, 2, 1))  # > 10 years after start
    assert exc_info.value.code == "query.calendar.range"
    assert dict(exc_info.value.context) == {
        "start": dt.date(2026, 1, 1),
        "end_edge": dt.date(2036, 2, 1),
        "span": (dt.date(2036, 2, 1) - dt.date(2026, 1, 1)).days + 1,
    }
    with pytest.raises(TypeError):
        exc_info.value.context["span"] = 1


def test_period_start_expr_refuses_unknown_grain():
    ds = ibis.literal(dt.date(2026, 1, 5))
    with pytest.raises(InvalidRequestError) as exc_info:
        period_start_expr(ds, "fortnight", "monday")  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "query.calendar.period_start_unknown_grain"
    assert exc_info.value.context["grain"] == "fortnight"


def test_period_end_expr_refuses_unknown_grain():
    period = ibis.literal(dt.date(2026, 1, 5))
    with pytest.raises(InvalidRequestError) as exc_info:
        _period_end_expr(period, "fortnight")  # ty: ignore[invalid-argument-type]
    assert exc_info.value.code == "query.calendar.period_start_unknown_grain"
    assert exc_info.value.context["grain"] == "fortnight"


def test_calendar_periods_refuses_unknown_grain(con):
    with pytest.raises(InvalidRequestError) as exc_info:
        _periods(con, grain="fortnight")
    assert exc_info.value.code == "query.calendar.period_start_unknown_grain"
    assert exc_info.value.context["grain"] == "fortnight"


def test_period_unit_values_refuses_unknown_aggregation():
    stats = ibis.memtable(
        {
            "unit_id": ["u1"],
            "ds": [dt.date(2026, 1, 5)],
            "sum_value": [10.0],
            "n_events": [1],
            "min_value": [10.0],
            "max_value": [10.0],
        }
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        period_unit_values(
            stats,
            aggregation="bogus",
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 1),
            end_edge=dt.date(2026, 1, 31),
        )
    assert exc_info.value.code == "query.builders.unknown_aggregation"
    assert exc_info.value.context["aggregation"] == "bogus"


def test_unknown_aggregation_entry_points_share_canonical_code(con, fact_table, dimmed_fact_table):
    """The four aggregation dispatchers reject one hazard with one code."""
    stats = ibis.memtable(
        {
            "unit_id": ["u1"],
            "ds": [dt.date(2026, 1, 5)],
            "sum_value": [10.0],
            "n_events": [1],
            "min_value": [10.0],
            "max_value": [10.0],
        }
    )
    with pytest.raises(InvalidRequestError) as via_builder:
        aggregate_stats(stats, aggregation="bogus")
    with pytest.raises(InvalidRequestError) as via_period:
        period_unit_values(
            stats,
            aggregation="bogus",
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 1),
            end_edge=dt.date(2026, 1, 31),
        )
    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    events = metric_events(dimmed_fact_table, metric, value_column="revenue")
    with pytest.raises(InvalidRequestError) as via_total:
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="bogus",
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        )
    events = metric_events(fact_table, metric, value_column="revenue")
    with pytest.raises(InvalidRequestError) as via_rolling:
        rolling_total_values(
            events,
            _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 14)),
            window=7,
            aggregation="bogus",
            start=dt.date(2026, 1, 12),
            end_edge=dt.date(2026, 1, 14),
        )
    assert (
        via_builder.value.code
        == via_period.value.code
        == via_total.value.code
        == via_rolling.value.code
    )


def test_period_population_refuses_population_without_unit_id():
    population = ibis.memtable({"other_id": ["a"]})
    fact_table = ibis.memtable({"unit_id": ["u1"], "ts": [dt.datetime(2026, 1, 1)]})
    with pytest.raises(InvalidRequestError) as exc_info:
        period_population(
            fact_table,
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 1),
            end_edge=dt.date(2026, 1, 31),
            periods=None,  # ty: ignore[invalid-argument-type]
            population=population,
        )
    assert exc_info.value.code == "query.calendar.population_unit_id"
    assert exc_info.value.context["columns"] == ("other_id",)


EVENTS = [
    # unit, ts, event, revenue
    ("u1", dt.datetime(2026, 1, 5, 10), "purchase", 10.0),
    ("u1", dt.datetime(2026, 1, 6, 10), "purchase", 20.0),
    ("u2", dt.datetime(2026, 1, 7, 10), "purchase", 5.0),
    ("u3", dt.datetime(2026, 1, 8, 10), "page_view", None),
    # week 2 (starts 2026-01-12)
    ("u1", dt.datetime(2026, 1, 13, 10), "page_view", None),
    ("u2", dt.datetime(2026, 1, 14, 10), "purchase", 7.0),
]


@pytest.fixture
def fact_table(con):
    return con.create_table(
        "events",
        ibis.memtable(
            [{"unit_id": u, "ts": ts, "event": e, "revenue": v} for (u, ts, e, v) in EVENTS]
        ),
    )


def _wk_periods(con):
    return calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        data_horizon=ibis.literal(dt.date(2026, 1, 18)),
        grain="week",
        week_start="monday",
    )


def test_mean_metric_period_moments_zero_filled(con, fact_table):
    metric = MeanMetric(name="rev", entity="user_id", fact="purchase", aggregation="sum")
    events = metric_events(fact_table, metric, value_column="revenue")
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="sum",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
    )
    periods = _wk_periods(con)
    pop = period_population(
        fact_table,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        periods=periods,
    )
    out = con.execute(period_moments(values, pop, periods, zero_fill=True).order_by("period"))
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    # Week 1 active units: u1 (30), u2 (5), u3 (0, page_view only -> zero-filled).
    ref = 35.0 / 3
    assert wk1.n == 3
    assert wk1.ref_y == pytest.approx(ref)
    # First moment stays exactly recoverable: sum(y) == n*ref_y + cy1.
    assert wk1.n * wk1.ref_y + wk1.cy1 == pytest.approx(35.0)
    assert wk1.cy2 == pytest.approx((30.0 - ref) ** 2 + (5.0 - ref) ** 2 + (0.0 - ref) ** 2)
    wk2 = out[out.period.dt.date == dt.date(2026, 1, 12)].iloc[0]
    # Week 2 active: u1 (0 purchases, zero-filled), u2 (7).
    assert wk2.n == 2
    assert wk2.n * wk2.ref_y + wk2.cy1 == pytest.approx(7.0)


def test_conversion_values_are_binary(con, fact_table):
    metric = ConversionMetric(name="cvr", entity="user_id", fact="purchase")
    events = metric_events(fact_table, metric)
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="conversion",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
    )
    rows = con.execute(values)
    assert set(rows.y) == {1.0}
    # u1 purchased twice in week 1 -> still one row, y == 1.
    assert len(rows[(rows.unit_id == "u1") & (rows.period.dt.date == dt.date(2026, 1, 5))]) == 1


def test_static_population_zero_fills_absent_units(con, fact_table):
    metric = ConversionMetric(name="cvr", entity="user_id", fact="purchase")
    events = metric_events(fact_table, metric)
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="conversion",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
    )
    periods = _wk_periods(con)
    registered = ibis.memtable([{"unit_id": u} for u in ["u1", "u2", "u3", "u4"]])
    pop = period_population(
        fact_table,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        population=registered,
        periods=periods,
    )
    out = con.execute(period_moments(values, pop, periods, zero_fill=True))
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    # 4 registered units, 2 converted -> the reference mean IS the rate.
    assert wk1.n == 4
    assert wk1.ref_y == pytest.approx(0.5)
    assert wk1.n * wk1.ref_y + wk1.cy1 == pytest.approx(2.0)


def test_no_zero_fill_excludes_inactive_units(con, fact_table):
    metric = MeanMetric(name="avg_rev", entity="user_id", fact="purchase", aggregation="avg_event")
    events = metric_events(fact_table, metric, value_column="revenue")
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="avg_event",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
    )
    periods = _wk_periods(con)
    pop = period_population(
        fact_table,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        periods=periods,
    )
    out = con.execute(period_moments(values, pop, periods, zero_fill=False))
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    # Only purchasers count: u1 avg 15, u2 avg 5 -> n=2, mean 10.
    assert wk1.n == 2
    assert wk1.ref_y == pytest.approx(10.0)
    assert wk1.n * wk1.ref_y + wk1.cy1 == pytest.approx(20.0)
    assert wk1.cy2 == pytest.approx((15.0 - 10.0) ** 2 + (5.0 - 10.0) ** 2)


def test_avg_calendar_day_uses_inclusive_period_days(con, fact_table):
    metric = MeanMetric(
        name="avg_rev",
        entity="user_id",
        fact="purchase",
        aggregation="avg_calendar_day",
    )
    events = metric_events(fact_table, metric, value_column="revenue")
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="avg_calendar_day",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
    )
    periods = _wk_periods(con)
    population = ibis.memtable({"unit_id": ["u1", "u2"]})
    pop = period_population(
        fact_table,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        periods=periods,
        population=population,
    )
    out = con.execute(period_moments(values, pop, periods, zero_fill=True))
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    # u1+u2 qualifying sums are 35 across seven admitted calendar days.
    assert wk1.n == 2
    assert wk1.ref_y == pytest.approx(17.5 / 7.0)


def test_empty_period_emitted_with_n_zero(con, fact_table):
    metric = MeanMetric(name="rev", entity="user_id", fact="purchase", aggregation="sum")
    events = metric_events(fact_table, metric, value_column="revenue")
    stats = unit_day_stats(events, source_key="events")
    end = dt.date(2026, 1, 25)  # week of Jan 19 has no events at all
    values = period_unit_values(
        stats,
        aggregation="sum",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=end,
    )
    periods = calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=end,
        data_horizon=ibis.literal(end),
        grain="week",
        week_start="monday",
    )
    pop = period_population(
        fact_table,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=end,
        periods=periods,
    )
    out = con.execute(period_moments(values, pop, periods, zero_fill=True))
    wk3 = out[out.period.dt.date == dt.date(2026, 1, 19)]
    assert len(wk3) == 1
    assert wk3.iloc[0].n == 0


def test_ratio_period_values_point_estimates(con, fact_table):
    from increment.semantics.models import RatioMetric

    metric = RatioMetric(
        name="rev_per_pv",
        entity="user_id",
        numerator={"fact": "purchase", "aggregation": "sum"},
        denominator={"fact": "page_view", "aggregation": "count"},
    )
    num_events = metric_events(fact_table, metric, value_column="revenue", part="numerator")
    den_events = metric_events(fact_table, metric, part="denominator")
    grain, week_start = "week", "monday"
    start, end_edge = dt.date(2026, 1, 5), dt.date(2026, 1, 18)
    num = period_unit_values(
        unit_day_stats(num_events, source_key="e"),
        aggregation="sum",
        grain=grain,
        week_start=week_start,
        start=start,
        end_edge=end_edge,
    )
    den = period_unit_values(
        unit_day_stats(den_events, source_key="e"),
        aggregation="count",
        grain=grain,
        week_start=week_start,
        start=start,
        end_edge=end_edge,
    )
    periods = _wk_periods(con)
    pop = period_population(
        fact_table,
        grain=grain,
        week_start=week_start,
        start=start,
        end_edge=end_edge,
        periods=periods,
    )
    out = con.execute(ratio_period_values(num, den, pop, periods).order_by("period"))
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    # Week 1: revenue 35, page views 1 -> 35.0
    assert wk1.value == pytest.approx(35.0)


def test_period_unit_values_min_max_count_distinct(con, fact_table):
    # Per-unit-day sums in week 1: u1 day5=10, u1 day6=20, u2 day7=5.
    metric = MeanMetric(name="rev", entity="user_id", fact="purchase", aggregation="min")
    events = metric_events(fact_table, metric, value_column="revenue")
    stats = unit_day_stats(events, source_key="events")
    kw = {
        "grain": "week",
        "week_start": "monday",
        "start": dt.date(2026, 1, 5),
        "end_edge": dt.date(2026, 1, 18),
    }

    min_vals = con.execute(period_unit_values(stats, aggregation="min", **kw))
    u1_wk1 = min_vals[
        (min_vals.unit_id == "u1") & (min_vals.period.dt.date == dt.date(2026, 1, 5))
    ].iloc[0]
    assert u1_wk1.y == pytest.approx(10.0)  # min(10, 20)

    max_vals = con.execute(period_unit_values(stats, aggregation="max", **kw))
    u1_wk1 = max_vals[
        (max_vals.unit_id == "u1") & (max_vals.period.dt.date == dt.date(2026, 1, 5))
    ].iloc[0]
    assert u1_wk1.y == pytest.approx(20.0)  # max(10, 20)

    distinct_vals = con.execute(period_unit_values(stats, aggregation="count_distinct", **kw))
    u1_wk1 = distinct_vals[
        (distinct_vals.unit_id == "u1") & (distinct_vals.period.dt.date == dt.date(2026, 1, 5))
    ].iloc[0]
    assert u1_wk1.y == pytest.approx(2.0)  # distinct daily sums {10, 20}
    u2_wk1 = distinct_vals[
        (distinct_vals.unit_id == "u2") & (distinct_vals.period.dt.date == dt.date(2026, 1, 5))
    ].iloc[0]
    assert u2_wk1.y == pytest.approx(1.0)  # distinct daily sums {5}


def test_period_population_time_varying(con, fact_table):
    # u1 registers Jan 6 (week 1); u2 registers Jan 14 (week 2). Membership
    # is per-period (the period containing ds), not cumulative onward.
    registered = ibis.memtable(
        [
            {"unit_id": "u1", "ds": dt.date(2026, 1, 6)},
            {"unit_id": "u2", "ds": dt.date(2026, 1, 14)},
        ]
    )
    periods = _wk_periods(con)
    pop = con.execute(
        period_population(
            fact_table,
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 18),
            population=registered,
            periods=periods,
        )
    )
    wk1_units = set(pop[pop.period.dt.date == dt.date(2026, 1, 5)].unit_id)
    wk2_units = set(pop[pop.period.dt.date == dt.date(2026, 1, 12)].unit_id)
    assert wk1_units == {"u1"}
    assert wk2_units == {"u2"}


def test_period_population_excludes_null_unit_id(con):
    """Regression: a NULL unit_id row in the default population
    construction must not become a phantom population member. SQL NULL
    never equals NULL, so period_moments' left-join zero-fills it,
    inflating n and diluting the mean -- one real conversion for u1 plus
    a NULL-unit row must read n=1, ref_y=1.0, not n=2, ref_y=0.5.
    """
    fact = con.create_table(
        "null_unit_conversion_events",
        ibis.memtable(
            [
                {"unit_id": "u1", "ts": dt.datetime(2026, 1, 5, 10), "event": "convert"},
                {"unit_id": None, "ts": dt.datetime(2026, 1, 6, 10), "event": "convert"},
            ]
        ),
    )
    metric = ConversionMetric(name="cvr", entity="user_id", fact="convert")
    events = metric_events(fact, metric)
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="conversion",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 11),
    )
    periods = calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 11),
        data_horizon=ibis.literal(dt.date(2026, 1, 11)),
        grain="week",
        week_start="monday",
    )
    pop = period_population(
        fact,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 11),
        periods=periods,
    )
    row = con.execute(period_moments(values, pop, periods, zero_fill=True).order_by("period")).iloc[
        0
    ]
    assert row.n == 1
    assert row.ref_y == pytest.approx(1.0)


def test_period_moments_by_table_requires_exactly_one_dimension_column(con, fact_table):
    """Regression: `by[0]` silently dropped a second dimension column and
    IndexError'd on zero -- both must now be an explicit ValueError."""
    metric = MeanMetric(name="rev", entity="user_id", fact="purchase", aggregation="sum")
    events = metric_events(fact_table, metric, value_column="revenue")
    stats = unit_day_stats(events, source_key="events")
    values = period_unit_values(
        stats,
        aggregation="sum",
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
    )
    periods = _wk_periods(con)
    pop = period_population(
        fact_table,
        grain="week",
        week_start="monday",
        start=dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        periods=periods,
    )
    zero_dim_by = ibis.memtable([{"unit_id": "u1", "period": dt.date(2026, 1, 5)}])
    with pytest.raises(InvalidRequestError) as exc_info:
        period_moments(values, pop, periods, zero_fill=True, by=zero_dim_by)
    assert exc_info.value.code == "query.calendar.period_moments_by"
    assert exc_info.value.context["dim_cols"] == ()
    two_dim_by = ibis.memtable(
        [{"unit_id": "u1", "period": dt.date(2026, 1, 5), "country": "US", "device": "ios"}]
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        period_moments(values, pop, periods, zero_fill=True, by=two_dim_by)
    assert exc_info.value.code == "query.calendar.period_moments_by"
    assert exc_info.value.context["dim_cols"] == ("country", "device")


DIMMED_EVENTS = [
    ("u1", dt.datetime(2026, 1, 5, 10), "purchase", 10.0, "US"),
    ("u1", dt.datetime(2026, 1, 6, 10), "purchase", 20.0, "DE"),
    ("u2", dt.datetime(2026, 1, 7, 10), "purchase", 5.0, "US"),
    ("u2", dt.datetime(2026, 1, 14, 10), "purchase", 7.0, "US"),
]


@pytest.fixture
def dimmed_fact_table(con):
    return con.create_table(
        "dim_events",
        ibis.memtable(
            [
                {"unit_id": u, "ts": ts, "event": e, "revenue": v, "country_code": c}
                for (u, ts, e, v, c) in DIMMED_EVENTS
            ]
        ),
    )


def _periods_3wk(con):
    return calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 25),
        data_horizon=ibis.literal(dt.date(2026, 1, 25)),
        grain="week",
        week_start="monday",
    )


def test_total_sum_by_event_level_dim(con, dimmed_fact_table):
    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    events = metric_events(
        dimmed_fact_table, metric, value_column="revenue", keep=["country_code"]
    ).rename(country="country_code")
    out = con.execute(
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="sum",
            by=["country"],
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        ).order_by(["period", "country"])
    )
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)]
    # Event-level dims: u1's purchases split across US and DE.
    assert dict(zip(wk1.country, wk1.value, strict=True)) == {"DE": 20.0, "US": 15.0}


def test_total_avg_by_event_level_dim_omits_sparse_period_dim_rows(con, dimmed_fact_table):
    """Mirrors test_total_sum_by_event_level_dim for aggregation="avg_event".

    DIMMED_EVENTS spans two dimension values unevenly across periods:
    week 1 has both US and DE purchases, week 2 has a US purchase only
    (no DE), and week 3 has none. A dimensioned result must reflect that
    sparsity exactly - (period, dim) combinations with no underlying
    events are OMITTED (no week-2 DE row, no week-3 row at all), unlike
    the undimensioned total, which zero/null-fills every period."""
    metric = TotalMetric(name="rev", fact="purchase", aggregation="avg_event")
    events = metric_events(
        dimmed_fact_table, metric, value_column="revenue", keep=["country_code"]
    ).rename(country="country_code")
    out = con.execute(
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="avg_event",
            by=["country"],
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        ).order_by(["period", "country"])
    )
    assert len(out) == 3, "no zero/null-filled rows for the sparse (period, dim) gaps"
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)]
    # US: mean(10.0, 5.0); DE: mean(20.0) - a single event.
    assert dict(zip(wk1.country, wk1.value, strict=True)) == {"DE": 20.0, "US": 7.5}
    wk2 = out[out.period.dt.date == dt.date(2026, 1, 12)]
    # Week 2 has only a US event - DE must be absent, not zero-filled.
    assert dict(zip(wk2.country, wk2.value, strict=True)) == {"US": 7.0}
    assert (out.period.dt.date == dt.date(2026, 1, 19)).sum() == 0, (
        "week 3 has no events for either country -- no row at all"
    )


def test_total_min_by_event_level_dim_omits_sparse_period_dim_rows(con, dimmed_fact_table):
    """Mirrors test_total_sum_by_event_level_dim for aggregation="min"
    (see test_total_avg_..._sparse_period_dim_rows for the sparsity
    pattern this and test_total_max_... also exercise)."""
    metric = TotalMetric(name="rev", fact="purchase", aggregation="min")
    events = metric_events(
        dimmed_fact_table, metric, value_column="revenue", keep=["country_code"]
    ).rename(country="country_code")
    out = con.execute(
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="min",
            by=["country"],
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        ).order_by(["period", "country"])
    )
    assert len(out) == 3, "no zero/null-filled rows for the sparse (period, dim) gaps"
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)]
    # US: min(10.0, 5.0); DE: min(20.0) - a single event.
    assert dict(zip(wk1.country, wk1.value, strict=True)) == {"DE": 20.0, "US": 5.0}
    wk2 = out[out.period.dt.date == dt.date(2026, 1, 12)]
    # Week 2 has only a US event - DE must be absent, not zero-filled.
    assert dict(zip(wk2.country, wk2.value, strict=True)) == {"US": 7.0}
    assert (out.period.dt.date == dt.date(2026, 1, 19)).sum() == 0, (
        "week 3 has no events for either country -- no row at all"
    )


def test_total_max_by_event_level_dim_omits_sparse_period_dim_rows(con, dimmed_fact_table):
    """Mirrors test_total_sum_by_event_level_dim for aggregation="max"
    (see test_total_avg_..._sparse_period_dim_rows for the sparsity
    pattern this and test_total_min_... also exercise)."""
    metric = TotalMetric(name="rev", fact="purchase", aggregation="max")
    events = metric_events(
        dimmed_fact_table, metric, value_column="revenue", keep=["country_code"]
    ).rename(country="country_code")
    out = con.execute(
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="max",
            by=["country"],
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        ).order_by(["period", "country"])
    )
    assert len(out) == 3, "no zero/null-filled rows for the sparse (period, dim) gaps"
    wk1 = out[out.period.dt.date == dt.date(2026, 1, 5)]
    # US: max(10.0, 5.0); DE: max(20.0) - a single event.
    assert dict(zip(wk1.country, wk1.value, strict=True)) == {"DE": 20.0, "US": 10.0}
    wk2 = out[out.period.dt.date == dt.date(2026, 1, 12)]
    # Week 2 has only a US event - DE must be absent, not zero-filled.
    assert dict(zip(wk2.country, wk2.value, strict=True)) == {"US": 7.0}
    assert (out.period.dt.date == dt.date(2026, 1, 19)).sum() == 0, (
        "week 3 has no events for either country -- no row at all"
    )


def test_total_sum_zero_fills_empty_period(con, dimmed_fact_table):
    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    events = metric_events(dimmed_fact_table, metric, value_column="revenue")
    out = con.execute(
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="sum",
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        ).order_by("period")
    )
    wk3 = out[out.period.dt.date == dt.date(2026, 1, 19)].iloc[0]
    assert wk3.value == 0.0


def test_total_avg_empty_period_is_null_not_zero(con, dimmed_fact_table):
    metric = TotalMetric(name="rev", fact="purchase", aggregation="avg_event")
    events = metric_events(dimmed_fact_table, metric, value_column="revenue")
    out = con.execute(
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="avg_event",
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        )
    )
    wk3 = out[out.period.dt.date == dt.date(2026, 1, 19)].iloc[0]
    assert wk3.value != wk3.value or wk3.value is None  # NaN or None, never 0


def test_total_period_values_refuses_unknown_aggregation(con, dimmed_fact_table):
    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    events = metric_events(dimmed_fact_table, metric, value_column="revenue")
    with pytest.raises(InvalidRequestError) as exc_info:
        total_period_values(
            events,
            _periods_3wk(con),
            aggregation="bogus",
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        )
    assert exc_info.value.code == "query.builders.unknown_aggregation"
    assert exc_info.value.context["aggregation"] == "bogus"


def test_active_counts_distinct_entities_exactly(con, dimmed_fact_table):
    metric = ActiveMetric(name="wap", entity="user_id", fact="purchase")
    events = metric_events(dimmed_fact_table, metric)
    out = con.execute(
        active_period_values(
            events,
            _periods_3wk(con),
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 25),
        ).order_by("period")
    )
    # Week 1: u1, u2 -> 2. Week 2: u2 -> 1. Week 3: none -> 0.
    assert list(out.value) == [2, 1, 0]


def test_total_count_count_distinct_min_max(con, dimmed_fact_table):
    # Week 1 revenue values: u1=10 (Jan5), u1=20 (Jan6), u2=5 (Jan7) -> 3
    # events, 3 distinct values, min=5, max=20. Week 2: u2=7 -> 1 event.
    grain: PeriodGrain = "week"
    week_start: WeekStart = "monday"
    start = dt.date(2026, 1, 5)
    end_edge = dt.date(2026, 1, 25)

    def _run(
        aggregation: Literal[
            "sum", "count", "count_distinct", "avg_event", "avg_calendar_day", "min", "max"
        ],
    ):
        metric = TotalMetric(name="rev", fact="purchase", aggregation=aggregation)
        events = metric_events(dimmed_fact_table, metric, value_column="revenue")
        return con.execute(
            total_period_values(
                events,
                _periods_3wk(con),
                aggregation=aggregation,
                grain=grain,
                week_start=week_start,
                start=start,
                end_edge=end_edge,
            ).order_by("period")
        )

    counts = _run("count")
    assert list(counts.value) == [3.0, 1.0, 0.0]  # zero-filled empty week 3

    distinct = _run("count_distinct")
    assert list(distinct.value) == [3.0, 1.0, 0.0]  # {10,20,5}, {7}, {} -> zero-filled

    mins = _run("min")
    wk1_min = mins[mins.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    assert wk1_min.value == pytest.approx(5.0)
    wk3_min = mins[mins.period.dt.date == dt.date(2026, 1, 19)].iloc[0]
    assert wk3_min.value != wk3_min.value or wk3_min.value is None  # NULL, not 0

    maxs = _run("max")
    wk1_max = maxs[maxs.period.dt.date == dt.date(2026, 1, 5)].iloc[0]
    assert wk1_max.value == pytest.approx(20.0)


def test_asof_dimension_no_lookahead(con):
    # u1 US until Jan10, then DE. u2 no observation until week 2. u3 migrates DURING week 2 - the defining as-of case: one unit resolving to two different non-null values across consecutive periods.
    props = ibis.memtable(
        [
            {"unit_id": "u1", "ts": dt.datetime(2026, 1, 3, 9), "country": "US"},
            {"unit_id": "u1", "ts": dt.datetime(2026, 1, 10, 9), "country": "DE"},
            {"unit_id": "u2", "ts": dt.datetime(2026, 1, 14, 9), "country": "FR"},
            {"unit_id": "u3", "ts": dt.datetime(2026, 1, 3, 9), "country": "US"},
            {"unit_id": "u3", "ts": dt.datetime(2026, 1, 13, 9), "country": "DE"},
        ]
    )
    periods = calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        data_horizon=ibis.literal(dt.date(2026, 1, 18)),
        grain="week",
        week_start="monday",
    )
    out = con.execute(
        asof_property_table(props, periods, dim="country").order_by(["unit_id", "period"])
    )
    u1 = out[out.unit_id == "u1"]
    # Week 1 ends Jan 11: latest u1 observation <= Jan 11 is DE (Jan 10).
    assert dict(zip(u1.period.dt.date, u1.country, strict=True)) == {
        dt.date(2026, 1, 5): "DE",
        dt.date(2026, 1, 12): "DE",
    }
    u2 = out[out.unit_id == "u2"]
    # u2 absent in week 1 (no observation yet), FR in week 2 - no lookahead.
    assert dict(zip(u2.period.dt.date, u2.country, strict=True)) == {dt.date(2026, 1, 12): "FR"}
    u3 = out[out.unit_id == "u3"]
    # u3's Jan 13 observation falls AFTER week 1's end (Jan 11) but before
    # week 2's end (Jan 18): week 1 stays US, week 2 correctly picks up DE.
    assert dict(zip(u3.period.dt.date, u3.country, strict=True)) == {
        dt.date(2026, 1, 5): "US",
        dt.date(2026, 1, 12): "DE",
    }


def test_asof_dimension_takes_latest_within_bound(con):
    props = ibis.memtable(
        [
            {"unit_id": "u1", "ts": dt.datetime(2026, 1, 5, 9), "country": "US"},
            {"unit_id": "u1", "ts": dt.datetime(2026, 1, 5, 18), "country": "BR"},
        ]
    )
    periods = calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 11),
        data_horizon=ibis.literal(dt.date(2026, 1, 11)),
        grain="week",
        week_start="monday",
    )
    out = con.execute(asof_property_table(props, periods, dim="country"))
    assert list(out.country) == ["BR"]  # same-day later timestamp wins


def test_asof_dimension_tie_is_deterministic_regardless_of_row_order(con):
    """Two observations at the IDENTICAL timestamp must resolve to the same
    label whatever the physical row order - ``order_by`` feeding
    ``distinct(keep=...)`` is not a SQL contract, so the tiebreak is
    explicit: largest value wins (same rule as
    ``builders.breakout_property_table``)."""
    ts = dt.datetime(2026, 1, 5, 9)
    rows = [
        {"unit_id": "u1", "ts": ts, "country": "US"},
        {"unit_id": "u1", "ts": ts, "country": "CA"},
    ]
    periods = calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 11),
        data_horizon=ibis.literal(dt.date(2026, 1, 11)),
        grain="week",
        week_start="monday",
    )
    forward = con.execute(asof_property_table(ibis.memtable(rows), periods, dim="country"))
    reverse = con.execute(asof_property_table(ibis.memtable(rows[::-1]), periods, dim="country"))
    assert list(forward.country) == ["US"]  # max value, not insertion order
    assert list(reverse.country) == ["US"]


def _daily_periods(con, start, end):
    return calendar_periods(
        start,
        end_edge=end,
        data_horizon=ibis.literal(end),
        grain="day",
        week_start="monday",
    )


def test_rolling_active_counts_trailing_window(con, fact_table):
    metric = ActiveMetric(name="wap7", entity="user_id", fact="purchase")
    events = metric_events(fact_table, metric)
    periods = _daily_periods(con, dt.date(2026, 1, 10), dt.date(2026, 1, 16))
    out = con.execute(
        rolling_active_values(
            events,
            periods,
            window=7,
            start=dt.date(2026, 1, 10),
            end_edge=dt.date(2026, 1, 16),
        ).order_by("period")
    )
    # 7-day trailing windows: Jan10-12 see u1(Jan5,6)+u2(Jan7)=2; Jan13+ only u2's
    # purchases remain in-window (2->1). Events before `start` still feed the lookback.
    assert list(out.value) == [2, 2, 2, 1, 1, 1, 1]


def test_rolling_total_sum_trailing_window(con, fact_table):
    metric = TotalMetric(name="rev7", fact="purchase", aggregation="sum")
    events = metric_events(fact_table, metric, value_column="revenue")
    periods = _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 14))
    out = con.execute(
        rolling_total_values(
            events,
            periods,
            window=7,
            aggregation="sum",
            start=dt.date(2026, 1, 12),
            end_edge=dt.date(2026, 1, 14),
        ).order_by("period")
    )
    # [Jan 6..12]: 20+5=25. [Jan 7..13]: 5. [Jan 8..14]: 7.
    assert list(out.value) == [25.0, 5.0, 7.0]


def test_rolling_total_values_refuses_unknown_aggregation(con, fact_table):
    metric = TotalMetric(name="rev7", fact="purchase", aggregation="sum")
    events = metric_events(fact_table, metric, value_column="revenue")
    periods = _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 14))
    with pytest.raises(InvalidRequestError) as exc_info:
        rolling_total_values(
            events,
            periods,
            window=7,
            aggregation="bogus",
            start=dt.date(2026, 1, 12),
            end_edge=dt.date(2026, 1, 14),
        )
    assert exc_info.value.code == "query.builders.unknown_aggregation"
    assert exc_info.value.context["aggregation"] == "bogus"


def test_rolling_window_overlap_is_not_additive(con, fact_table):
    # The same u2 purchase (Jan 7) appears in multiple windows - summing
    # rolling values across days would double-count; assert the overlap.
    metric = ActiveMetric(name="wap", entity="user_id", fact="purchase")
    events = metric_events(fact_table, metric)
    periods = _daily_periods(con, dt.date(2026, 1, 8), dt.date(2026, 1, 10))
    out = con.execute(
        rolling_active_values(
            events,
            periods,
            window=7,
            start=dt.date(2026, 1, 8),
            end_edge=dt.date(2026, 1, 10),
        )
    )
    assert (out.value == 2).all()  # u1 and u2 present in every window


def test_rolling_active_differs_from_calendar_bucket_on_same_data(con, fact_table):
    # Calendar-bucket weekly active partitions events into non-overlapping buckets; rolling daily active looks back `window` days from every row - genuinely different questions on identical data, not just different shapes of the same number.
    metric = ActiveMetric(name="wap", entity="user_id", fact="purchase")
    events = metric_events(fact_table, metric)
    weekly_periods = calendar_periods(
        dt.date(2026, 1, 5),
        end_edge=dt.date(2026, 1, 18),
        data_horizon=ibis.literal(dt.date(2026, 1, 18)),
        grain="week",
        week_start="monday",
    )
    calendar_out = con.execute(
        active_period_values(
            events,
            weekly_periods,
            grain="week",
            week_start="monday",
            start=dt.date(2026, 1, 5),
            end_edge=dt.date(2026, 1, 18),
        ).order_by("period")
    )
    # Week 2 (Jan 12-18, second row after ordering) calendar bucket only
    # sees u2's Jan 14 purchase.
    week2_value = calendar_out.value.iloc[1]
    assert week2_value == 1

    daily_periods = _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 12))
    rolling_out = con.execute(
        rolling_active_values(
            events,
            daily_periods,
            window=7,
            start=dt.date(2026, 1, 12),
            end_edge=dt.date(2026, 1, 12),
        )
    )
    # The trailing-7d window ending Jan12 reaches back to Jan6, so it also sees u1's Jan5/6 purchases which the calendar week-2 bucket (starting Jan12) cannot.
    assert rolling_out.value.iloc[0] == 2
    assert rolling_out.value.iloc[0] != week2_value


def test_rolling_total_remaining_aggregations(con, fact_table):
    # Purchases: u1 Jan5=10, u1 Jan6=20, u2 Jan7=5, u2 Jan14=7. 7-day window ending Jan12 sees u1's Jan6=20 and u2's Jan7=5.
    metric = TotalMetric(name="rev7", fact="purchase", aggregation="sum")
    events = metric_events(fact_table, metric, value_column="revenue")
    periods = _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 12))

    def _run(aggregation: str):
        return con.execute(
            rolling_total_values(
                events,
                periods,
                window=7,
                aggregation=aggregation,
                start=dt.date(2026, 1, 12),
                end_edge=dt.date(2026, 1, 12),
            )
        )

    counts = _run("count")
    assert counts.value.iloc[0] == pytest.approx(2.0)  # 2 events (Jan6, Jan7)

    avgs = _run("avg_event")
    assert avgs.value.iloc[0] == pytest.approx(12.5)  # (20+5)/2

    mins = _run("min")
    assert mins.value.iloc[0] == pytest.approx(5.0)

    maxs = _run("max")
    assert maxs.value.iloc[0] == pytest.approx(20.0)

    distinct = _run("count_distinct")
    assert distinct.value.iloc[0] == pytest.approx(2.0)  # distinct values {20, 5}


def test_rolling_total_avg_empty_window_is_null_not_zero(con, fact_table):
    metric = TotalMetric(name="rev7", fact="purchase", aggregation="avg_event")
    events = metric_events(fact_table, metric, value_column="revenue")
    # Window ending Jan 25: 7-day lookback [Jan19..25] has zero purchases.
    periods = _daily_periods(con, dt.date(2026, 1, 25), dt.date(2026, 1, 25))
    out = con.execute(
        rolling_total_values(
            events,
            periods,
            window=7,
            aggregation="avg_event",
            start=dt.date(2026, 1, 25),
            end_edge=dt.date(2026, 1, 25),
        )
    )
    v = out.value.iloc[0]
    assert v != v or v is None  # NaN or None, never 0.0


def test_rolling_total_dimensioned_sparse(con, dimmed_fact_table):
    # u1 Jan5 US=10, u1 Jan6 DE=20, u2 Jan7 US=5, u2 Jan14 US=7. 7-day
    # window ending Jan 12: [Jan6..12] sees u1's DE=20 and u2's US=5.
    metric = TotalMetric(name="rev7", fact="purchase", aggregation="sum")
    events = metric_events(
        dimmed_fact_table, metric, value_column="revenue", keep=["country_code"]
    ).rename(country="country_code")
    periods = _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 12))
    out = con.execute(
        rolling_total_values(
            events,
            periods,
            window=7,
            aggregation="sum",
            by=["country"],
            start=dt.date(2026, 1, 12),
            end_edge=dt.date(2026, 1, 12),
        )
    )
    assert dict(zip(out.country, out.value, strict=True)) == {"DE": 20.0, "US": 5.0}


def test_rolling_active_dimensioned_sparse(con, dimmed_fact_table):
    metric = ActiveMetric(name="wap7", entity="user_id", fact="purchase")
    events = metric_events(dimmed_fact_table, metric, keep=["country_code"]).rename(
        country="country_code"
    )
    periods = _daily_periods(con, dt.date(2026, 1, 12), dt.date(2026, 1, 12))
    out = con.execute(
        rolling_active_values(
            events,
            periods,
            window=7,
            by=["country"],
            start=dt.date(2026, 1, 12),
            end_edge=dt.date(2026, 1, 12),
        )
    )
    # Window [Jan6..12]: u1 in DE (1 distinct unit), u2 in US (1 distinct unit).
    assert dict(zip(out.country, out.value, strict=True)) == {"DE": 1, "US": 1}


# Declared day boundary (Definitions.day_boundary) threaded through the
# report/calendar path -- mirrors increment.query.builders.TestDayBoundary
# for the experiment path.


def test_total_period_values_tz_aware_is_session_timezone_independent():
    """A TIMESTAMPTZ event must bucket to the same day regardless of the
    backend's session timezone - the declared day boundary decides the
    day, never `SET TimeZone`. Mirrors
    test_local_date_timestamptz_is_session_timezone_independent in
    test_builders.py for the report/calendar path."""
    results = {}
    for session_tz in ("UTC", "America/Los_Angeles"):
        con = ibis.duckdb.connect()
        con.raw_sql(f"SET TimeZone='{session_tz}'")
        con.raw_sql(
            "CREATE TABLE tz_events (unit_id VARCHAR, ts TIMESTAMPTZ, "
            "event VARCHAR, revenue DOUBLE)"
        )
        con.raw_sql(
            "INSERT INTO tz_events VALUES ('u1', '2025-01-02 00:30:00+00', 'purchase', 10.0)"
        )
        metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
        events = metric_events(con.table("tz_events"), metric, value_column="revenue")
        periods = calendar_periods(
            dt.date(2025, 1, 1),
            end_edge=dt.date(2025, 1, 3),
            data_horizon=ibis.literal(dt.date(2025, 1, 3)),
            grain="day",
            week_start="monday",
        )
        out = con.execute(
            total_period_values(
                events,
                periods,
                aggregation="sum",
                grain="day",
                week_start="monday",
                start=dt.date(2025, 1, 1),
                end_edge=dt.date(2025, 1, 3),
            )
        )
        results[session_tz] = dict(zip(out.period.dt.date, out.value, strict=True))
    assert results["UTC"] == results["America/Los_Angeles"]
    assert results["UTC"][dt.date(2025, 1, 2)] == 10.0


def test_total_period_values_respects_declared_day_boundary_offset(con):
    """A naive-timestamp event at 2025-01-02T03:00 under an offset of
    UTC-05:00 buckets to the PRIOR calendar day (local 2025-01-01
    22:00), not the raw UTC day."""
    events_tbl = con.create_table(
        "offset_events",
        ibis.memtable(
            [
                {
                    "unit_id": "u1",
                    "ts": dt.datetime(2025, 1, 2, 3, 0, 0),
                    "event": "purchase",
                    "revenue": 10.0,
                }
            ]
        ),
    )
    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    events = metric_events(events_tbl, metric, value_column="revenue")
    periods = calendar_periods(
        dt.date(2025, 1, 1),
        end_edge=dt.date(2025, 1, 3),
        data_horizon=ibis.literal(dt.date(2025, 1, 3)),
        grain="day",
        week_start="monday",
    )
    out = con.execute(
        total_period_values(
            events,
            periods,
            aggregation="sum",
            grain="day",
            week_start="monday",
            start=dt.date(2025, 1, 1),
            end_edge=dt.date(2025, 1, 3),
            day_boundary_offset=day_boundary_offset("UTC-05:00"),
        )
    )
    by_day = dict(zip(out.period.dt.date, out.value, strict=True))
    assert by_day[dt.date(2025, 1, 1)] == 10.0
    assert by_day[dt.date(2025, 1, 2)] == 0.0
