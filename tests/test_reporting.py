"""Report facade: calendar metric trends from definitions. DuckDB in-memory."""

import datetime as dt
import math
from fractions import Fraction
from typing import Any, cast

import ibis
import pytest
import yaml
from pydantic import ValidationError

from increment import CapabilityError, Report
from increment.errors import InvalidRequestError
from increment.semantics.models import Definitions

DEFS = {
    "fact_sources": [
        {
            "name": "events",
            "sql": "SELECT * FROM raw_events",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [
                {"name": "page_view", "column": None},
                {"name": "purchase", "column": "revenue"},
                {"name": "refund", "column": "revenue"},  # never occurs in ROWS
            ],
            "properties": [{"name": "country", "column": "country_code", "dtype": "string"}],
        },
        {
            "name": "orders_only",
            "sql": "SELECT * FROM raw_orders",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "order", "column": "amount"}],
            "properties": [{"name": "channel", "column": "channel", "dtype": "string"}],
        },
    ],
    "metrics": [
        {"type": "conversion", "name": "purchase_rate", "entity": "user_id", "fact": "purchase"},
        {
            "type": "mean",
            "name": "revenue",
            "entity": "user_id",
            "fact": "purchase",
            "aggregation": "sum",
        },
        {"type": "total", "name": "total_revenue", "fact": "purchase", "aggregation": "sum"},
        {"type": "total", "name": "refund_total", "fact": "refund", "aggregation": "sum"},
        {"type": "active", "name": "wau", "entity": "user_id", "fact": "page_view"},
        {
            "type": "retention",
            "name": "d7",
            "entity": "user_id",
            "fact": "page_view",
            "threshold_days": [7, 14],
        },
        {"type": "conversion", "name": "order_rate", "entity": "user_id", "fact": "order"},
        {
            "type": "conversion",
            "name": "web_order_rate",
            "entity": "user_id",
            "fact": "order",
            "filters": [{"property": "channel", "op": "equals", "values": ["web"]}],
        },
        {
            "type": "ratio",
            "name": "rev_per_order",
            "entity": "user_id",
            "numerator": {"fact": "purchase", "aggregation": "sum"},
            "denominator": {"fact": "order", "aggregation": "count"},
        },
        {
            "type": "ratio",
            "name": "refund_ratio",
            "entity": "user_id",
            "numerator": {"fact": "refund", "aggregation": "sum"},
            "denominator": {"fact": "purchase", "aggregation": "count"},
        },
    ],
}

ROWS = [
    {
        "user_id": "u1",
        "ts": dt.datetime(2026, 1, 5, 10),
        "event": "purchase",
        "revenue": 10.0,
        "country_code": "US",
    },
    {
        "user_id": "u1",
        "ts": dt.datetime(2026, 1, 6, 11),
        "event": "page_view",
        "revenue": None,
        "country_code": "US",
    },
    {
        "user_id": "u2",
        "ts": dt.datetime(2026, 1, 7, 9),
        "event": "page_view",
        "revenue": None,
        "country_code": "DE",
    },
    {
        "user_id": "u2",
        "ts": dt.datetime(2026, 1, 13, 9),
        "event": "purchase",
        "revenue": 4.0,
        "country_code": "DE",
    },
]

# Orders lag: latest order event is Jan 11 - one week behind the events
# source. u3 orders but never purchases (cross-source population case).
ORDER_ROWS = [
    {
        "user_id": "u1",
        "ts": dt.datetime(2026, 1, 5, 10),
        "event": "order",
        "amount": 10.0,
        "channel": "web",
    },
    {
        "user_id": "u3",
        "ts": dt.datetime(2026, 1, 11, 9),
        "event": "order",
        "amount": 3.0,
        "channel": "store",
    },
]


@pytest.fixture
def report():
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    return Report.from_definitions(Definitions.model_validate(DEFS), con)


@pytest.mark.parametrize("source_order", [("profile", "events"), ("events", "profile")])
def test_report_fact_resolution_ignores_property_source_order(source_order):
    sources = {
        "profile": {
            "name": "profile",
            "sql": "SELECT * FROM raw_profile",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [{"name": "profile_event", "column": None}],
            "properties": [{"name": "purchase", "column": "purchase", "as_of": "static"}],
        },
        "events": {
            "name": "events",
            "sql": "SELECT * FROM raw_purchase_events",
            "timestamp_column": "ts",
            "entities": ["user_id"],
            "facts": [
                {"name": "page_view", "column": None},
                {"name": "purchase", "column": None},
            ],
        },
    }
    definitions = Definitions.model_validate(
        {
            "fact_sources": [sources[name] for name in source_order],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "purchase_rate",
                    "entity": "user_id",
                    "fact": "purchase",
                }
            ],
        }
    )
    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "raw_profile",
            ibis.memtable(
                [
                    {"user_id": "u1", "ts": dt.datetime(2026, 1, 5), "purchase": "profile"},
                ]
            ),
        )
        con.create_table(
            "raw_purchase_events",
            ibis.memtable(
                [
                    {"user_id": "u1", "ts": dt.datetime(2026, 1, 5), "event": "purchase"},
                    {"user_id": "u2", "ts": dt.datetime(2026, 1, 6), "event": "page_view"},
                ]
            ),
        )
        frame = Report.from_definitions(definitions, con).metric("purchase_rate", **KW).to_frame()
        week = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
        assert week.n == 2
        assert week.value == pytest.approx(0.5)
    finally:
        con.disconnect()


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO t VALUES (1)",
        "WITH changed AS (INSERT INTO t VALUES (1) RETURNING *) SELECT * FROM changed",
    ],
)
def test_programmatic_definitions_reject_writes_before_query(sql: str):
    con = _SpyConnection()
    defs = Definitions.model_validate(
        {
            **DEFS,
            "fact_sources": [{**DEFS["fact_sources"][0], "sql": sql}, *DEFS["fact_sources"][1:]],
        }
    )

    with pytest.raises(InvalidRequestError) as raised:
        Report.from_definitions(defs, con)  # ty: ignore[invalid-argument-type]
    assert raised.value.code == "definition.sql.not_read_only"
    assert con.sql_calls == []


def test_direct_report_rejects_write_fact_source_before_query():
    con = _SpyConnection()
    defs = Definitions.model_validate(
        {
            **DEFS,
            "fact_sources": [
                {**DEFS["fact_sources"][0], "sql": "INSERT INTO t VALUES (1)"},
                *DEFS["fact_sources"][1:],
            ],
        }
    )

    with pytest.raises(InvalidRequestError) as raised:
        Report(defs, con)  # ty: ignore[invalid-argument-type]
    assert raised.value.code == "definition.sql.not_read_only"
    assert con.sql_calls == []


class _RecordingConnection:
    name = "duckdb"

    def __init__(self, con):
        self._con = con
        self.sql_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def sql(self, *args: object, **kwargs: object):
        self.sql_calls.append((args, kwargs))
        return self._con.sql(*args, **kwargs)


def test_report_uses_definitions_snapshot_after_caller_mutation():
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    recording_con = _RecordingConnection(con)
    defs = Definitions.model_validate(DEFS)
    report = Report(defs, recording_con)  # ty: ignore[invalid-argument-type]

    # The declaration layer is frozen, so a caller cannot rewrite the SQL it
    # already handed over -- a stronger guarantee than snapshotting a mutable
    # object. The report must still be unaffected either way.
    with pytest.raises(ValidationError):
        defs.fact_sources[0].sql = "INSERT INTO raw_events VALUES (1)"  # ty: ignore[invalid-assignment]

    trend = report.metric("purchase_rate", **KW)
    assert len(trend.to_frame()) == 2
    assert defs.fact_sources[0].sql == DEFS["fact_sources"][0]["sql"]
    assert all("INSERT" not in str(args[0]).upper() for args, _ in recording_con.sql_calls)


def test_programmatic_select_definitions_execute():
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))

    trend = Report.from_definitions(Definitions.model_validate(DEFS), con).metric(
        "purchase_rate", **KW
    )
    assert len(trend.to_frame()) == 2


def test_report_snapshots_immutable_allocation_definitions():
    from copy import deepcopy

    raw: dict[str, Any] = deepcopy(DEFS)
    raw["exposures"] = [{"name": "assignment", "sql": "SELECT 1"}]
    raw["experiments"] = [
        {
            "name": "pilot",
            "exposure": "assignment",
            "unit": "user_id",
            "start": "2026-01-01",
            "control_group": "C",
            "allocation": {"C": 1, "T": 1},
            "plan": {},
        }
    ]
    definitions = Definitions.model_validate(raw)
    con = ibis.duckdb.connect()
    try:
        con.create_table("raw_events", ibis.memtable(ROWS))
        con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
        report = Report.from_definitions(definitions, con)
        expected = report.metric("total_revenue", **KW).to_frame().sort_values("period")

        raw["experiments"][0]["allocation"]["T"] = 9
        raw["fact_sources"][0]["sql"] = "SELECT * FROM nonexistent_events"
        raw["metrics"][2]["aggregation"] = "count"
        actual = report.metric("total_revenue", **KW).to_frame().sort_values("period")
        assert actual["value"].tolist() == expected["value"].tolist()
        assert actual["value"].tolist() == pytest.approx([10.0, 4.0])
        allocation = definitions.experiments[0].allocation
        assert allocation is not None
        with pytest.raises(TypeError):
            allocation["T"] = 9  # ty: ignore[invalid-assignment]
    finally:
        con.disconnect()


@pytest.mark.filterwarnings(
    r"ignore:metric .* has window_days=None \(variable per-unit window\); specify window_days if a fixed window is intended:UserWarning"
)
def test_file_definitions_still_load_and_execute(tmp_path):
    path = tmp_path / "definitions.yaml"
    path.write_text(yaml.safe_dump(DEFS))
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))

    trend = Report.from_definitions(path, con).metric("purchase_rate", **KW)
    assert len(trend.to_frame()) == 2


class _SpyConnection:
    name = "snowflake"

    def __init__(self):
        self.sql_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def sql(self, *args: object, **kwargs: object):
        self.sql_calls.append((args, kwargs))
        return object()


@pytest.mark.parametrize(
    ("population", "expected_code"),
    [
        ("SELECT 1; DROP TABLE victim", "definition.sql.statement_count"),
        ("DELETE FROM victim", "definition.sql.not_read_only"),
        ("INSERT INTO victim VALUES (1)", "definition.sql.not_read_only"),
        ("UPDATE victim SET value = 1", "definition.sql.not_read_only"),
        ("CREATE TABLE victim (value INTEGER)", "definition.sql.not_read_only"),
        ("ALTER TABLE victim ADD COLUMN value INTEGER", "definition.sql.not_read_only"),
        ("DROP TABLE victim", "definition.sql.not_read_only"),
        ("SELECT * INTO victim FROM source", "definition.sql.not_read_only"),
        ("SELECT * FROM source FOR UPDATE", "definition.sql.not_read_only"),
        ("SELECT * FROM source WITH (UPDLOCK)", "definition.sql.not_read_only"),
    ],
)
def test_report_rejects_population_before_con_sql(population: str, expected_code: str):
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    recording = _RecordingConnection(con)
    report = Report(Definitions.model_validate(DEFS), recording)  # ty: ignore[invalid-argument-type]

    with pytest.raises(InvalidRequestError) as raised:
        report.metric("purchase_rate", population=population, **KW)
    assert raised.value.code == expected_code
    # The refused population never reached the warehouse at all.
    assert all(str(args[0]) != population for args, _ in recording.sql_calls)


class _DialectStandInConnection:
    """A snowflake-named connection that stands in for the population query."""

    name = "snowflake"

    def __init__(self, con, population: str) -> None:
        self.inner = con
        self._population = population
        self.sql_calls: list[str] = []

    def sql(self, query: str, *args: object, **kwargs: object):
        self.sql_calls.append(query)
        if query == self._population:
            return ibis.memtable({"unit_id": ["u1", "u2"]})
        return self.inner.sql(query, *args, **kwargs)


def test_report_population_validation_uses_backend_dialect():
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    population = "SELECT payload:user.id::string AS uid FROM raw_events"

    defs = Definitions.model_validate(DEFS)
    snowflake = _DialectStandInConnection(con, population)
    scoped = Report(defs, snowflake).metric(  # ty: ignore[invalid-argument-type]
        "purchase_rate", population=population, **KW
    )
    assert scoped.denominator == "population"
    assert population in snowflake.sql_calls

    # The same population is refused where no declared dialect can parse it.
    elsewhere = Report(defs, _RecordingConnection(con))  # ty: ignore[invalid-argument-type]
    with pytest.raises(InvalidRequestError) as raised:
        elsewhere.metric("purchase_rate", population=population, **KW)
    assert raised.value.code == "definition.sql.parse"


def test_completeness_horizon_scoped_to_metrics_own_fact():
    # A fresher sibling fact on the same source (page_view, Jan 20) must not
    # mask the metric's own fact (purchase, last event Jan 13) still being incomplete.
    lagging_rows = [
        *ROWS,
        {
            "user_id": "u1",
            "ts": dt.datetime(2026, 1, 20, 9),
            "event": "page_view",
            "revenue": None,
            "country_code": "US",
        },
    ]
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(lagging_rows))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    report = Report.from_definitions(Definitions.model_validate(DEFS), con)
    frame = report.metric(
        "revenue", grain="week", start=dt.date(2026, 1, 5), end=dt.date(2026, 1, 25)
    ).to_frame()
    # Week 1 (Jan 5-11): fully within purchase's horizon (Jan 13) - complete.
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    assert bool(wk1.period_complete)
    # Week 2 (Jan 12-18): purchase's own horizon (Jan 13) doesn't cover the
    # full week - must be incomplete, regardless of the fresher page_view.
    wk2 = frame[frame.period == dt.date(2026, 1, 12)].iloc[0]
    assert not bool(wk2.period_complete)


def test_report_bucketing_is_session_timezone_independent():
    """A TIMESTAMPTZ event must bucket to the same calendar day and the
    same completeness horizon regardless of the warehouse session
    timezone - the declared day boundary decides the day, never `SET
    TimeZone`. Mirrors test_total_period_values_tz_aware_is_session_
    timezone_independent (tests/query/test_report_builders.py) but
    exercises the full Report facade, including Report._build's own
    ts.max() horizon computation - the report/calendar path's other
    session-dependent cast site."""
    defs = {
        "fact_sources": [
            {
                "name": "tz_source",
                "sql": "SELECT * FROM tz_events",
                "timestamp_column": "ts",
                "entities": ["unit_id"],
                "facts": [{"name": "purchase", "column": "revenue"}],
            },
        ],
        "metrics": [
            {"type": "total", "name": "rev", "fact": "purchase", "aggregation": "sum"},
        ],
    }
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
        tz_report = Report.from_definitions(Definitions.model_validate(defs), con)
        # No explicit end=: the completeness horizon falls back to
        # tbl.ts.max() - the exact bug site. A session-dependent bucketing
        # there changes which periods the dense spine even covers.
        frame = tz_report.metric("rev", grain="day", start=dt.date(2025, 1, 1)).to_frame()
        results[session_tz] = dict(zip(frame.period, frame.value, strict=True))
    assert results["UTC"] == results["America/Los_Angeles"]
    assert results["UTC"][dt.date(2025, 1, 2)] == 10.0


def test_report_buckets_by_the_declared_day_boundary_not_utc():
    """A definitions-level ``day_boundary`` must decide the calendar day on the
    report/calendar path. The sibling session-timezone test runs at the default
    UTC boundary, so every ``day_boundary_offset`` argument could be deleted and
    it would still pass; this one fails if the offset stops being threaded.

    The event is 2025-01-02 00:30 UTC, which is 2025-01-01 19:30 at UTC-05:00 --
    a different calendar day, so the two boundaries cannot agree."""
    defs = {
        "day_boundary": "UTC-05:00",
        "fact_sources": [
            {
                "name": "tz_source",
                "sql": "SELECT * FROM tz_events",
                "timestamp_column": "ts",
                "entities": ["unit_id"],
                "facts": [{"name": "purchase", "column": "revenue"}],
            },
        ],
        "metrics": [
            {"type": "total", "name": "rev", "fact": "purchase", "aggregation": "sum"},
        ],
    }
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE tz_events (unit_id VARCHAR, ts TIMESTAMPTZ, event VARCHAR, revenue DOUBLE)"
    )
    con.raw_sql("INSERT INTO tz_events VALUES ('u1', '2025-01-02 00:30:00+00', 'purchase', 10.0)")
    report = Report.from_definitions(Definitions.model_validate(defs), con)
    frame = report.metric("rev", grain="day", start=dt.date(2025, 1, 1)).to_frame()
    values = dict(zip(frame.period, frame.value, strict=True))
    assert values[dt.date(2025, 1, 1)] == 10.0
    assert values.get(dt.date(2025, 1, 2), 0.0) == 0.0


def test_zero_row_fact_horizon_with_explicit_end_never_complete(report):
    # 'refund' has zero rows anywhere; its fact-scoped max(ts) is SQL NULL.
    # With an explicit end=, every period must be incomplete, never NULL/crash.
    frame = report.metric(
        "refund_total", grain="week", start=dt.date(2026, 1, 5), end=dt.date(2026, 1, 18)
    ).to_frame()
    assert len(frame) > 0
    assert (frame.period_complete == False).all()  # noqa: E712 - not NULL/NaN
    assert (frame.value == 0.0).all()  # zero-fill: no refund events anywhere


def test_zero_row_fact_horizon_with_default_end_empty_frame(report):
    # No end= given: edge falls back to horizon. A zero-row fact's sentinel
    # horizon (start - 1 day) precedes the spine entirely -> clean empty frame.
    frame = report.metric("refund_total", grain="week", start=dt.date(2026, 1, 5)).to_frame()
    assert len(frame) == 0


@pytest.mark.parametrize("grain", ["day", "week", "month"])
@pytest.mark.parametrize("span", [3652, 3653, 3654])
def test_default_report_horizon_capacity(grain, span):
    start = dt.date(2000, 1, 1)
    last = start + dt.timedelta(days=span - 1)
    definitions = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM horizon_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [{"name": "purchase", "column": "amount"}],
                }
            ],
            "metrics": [
                {"type": "total", "name": "revenue", "fact": "purchase", "aggregation": "sum"},
                {
                    "type": "mean",
                    "name": "calendar_average",
                    "entity": "unit_id",
                    "fact": "purchase",
                    "aggregation": "avg_calendar_day",
                },
            ],
        }
    )
    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "horizon_events",
            ibis.memtable(
                [
                    {
                        "unit_id": "u1",
                        "ts": dt.datetime.combine(start, dt.time(12)),
                        "event": "purchase",
                        "amount": 2.0,
                    },
                    {
                        "unit_id": "u2",
                        "ts": dt.datetime.combine(last, dt.time(12)),
                        "event": "purchase",
                        "amount": 7.0,
                    },
                ]
            ),
        )
        report = Report.from_definitions(definitions, con)
        if span > 3653:
            with pytest.raises(CapabilityError) as raised:
                report.metric("revenue", grain=grain, start=start).to_table().execute()
            assert raised.value.code == "query.calendar.range"
            assert dict(raised.value.context) == {
                "start": start,
                "end_edge": last,
                "span": span,
            }
            with pytest.raises(TypeError):
                cast(Any, raised.value.context)["span"] = 1
        else:
            default_trend = report.metric("revenue", grain=grain, start=start)
            assert default_trend.end is None
            assert default_trend.sql()
            default = default_trend.to_frame()
            assert len(default_trend.to_table().execute()) == len(default)
            assert len(default_trend.to_frame("polars")) == len(default)
            assert default_trend.to_frame("pyarrow").num_rows == len(default)
            explicit = report.metric("revenue", grain=grain, start=start, end=last).to_frame()
            columns = ["period", "period_complete", "n", "value", "ci_lb", "ci_ub"]
            from pandas.testing import assert_frame_equal

            assert_frame_equal(
                default.sort_values("period")[columns].reset_index(drop=True),
                explicit.sort_values("period")[columns].reset_index(drop=True),
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
            assert default["value"].sum() == pytest.approx(9.0)
            con.create_table("report_units", ibis.memtable({"unit_id": ["u1", "u2"]}))
            population = con.table("report_units")
            average = report.metric(
                "calendar_average", grain=grain, start=start, population=population
            ).to_frame()
            for row in average.itertuples():
                period = row.period.date() if isinstance(row.period, dt.datetime) else row.period
                if grain == "day":
                    period_end = period
                elif grain == "week":
                    period_end = period + dt.timedelta(days=6)
                else:
                    next_month = (period.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
                    period_end = next_month - dt.timedelta(days=1)
                low, high = max(period, start), min(period_end, last)
                days = (high - low).days + 1
                amount = 2.0 if low <= start <= high else 0.0
                amount += 7.0 if low <= last <= high else 0.0
                assert row.n == 2
                assert row.value == pytest.approx(amount / (2 * days), rel=1e-12, abs=1e-12)
    finally:
        con.disconnect()


def test_default_horizons_are_selected_per_metric_and_frozen():
    start = dt.date(2000, 1, 1)
    definitions = Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM scoped_horizon_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": name, "column": "amount"}
                        for name in ("purchase", "numerator", "denominator", "unrelated")
                    ],
                }
            ],
            "metrics": [
                {"type": "total", "name": "fresh", "fact": "purchase", "aggregation": "sum"},
                {
                    "type": "ratio",
                    "name": "ratio",
                    "entity": "unit_id",
                    "numerator": {"fact": "numerator", "aggregation": "sum"},
                    "denominator": {"fact": "denominator", "aggregation": "sum"},
                },
            ],
        }
    )
    con = ibis.duckdb.connect()
    try:
        con.create_table(
            "scoped_horizon_events",
            ibis.memtable(
                [
                    {
                        "unit_id": "u1",
                        "ts": dt.datetime(2000, 1, 10, 12),
                        "event": "purchase",
                        "amount": 5.0,
                    },
                    {
                        "unit_id": "u1",
                        "ts": dt.datetime(2000, 1, 15, 12),
                        "event": "numerator",
                        "amount": 8.0,
                    },
                    {
                        "unit_id": "u1",
                        "ts": dt.datetime(2000, 2, 1, 12),
                        "event": "numerator",
                        "amount": 8.0,
                    },
                    {
                        "unit_id": "u1",
                        "ts": dt.datetime(2000, 1, 15, 12),
                        "event": "denominator",
                        "amount": 4.0,
                    },
                    {
                        "unit_id": "u1",
                        "ts": dt.datetime(2020, 1, 1, 12),
                        "event": "unrelated",
                        "amount": 100.0,
                    },
                ]
            ),
        )
        report = Report.from_definitions(definitions, con)
        trends = report.metrics(["fresh", "ratio"], grain="day", start=start)
        assert trends["fresh"].end is None
        assert trends["ratio"].end is None
        fresh = trends["fresh"].to_frame()
        ratio = trends["ratio"].to_frame()
        assert max(fresh["period"]) == dt.date(2000, 1, 10)
        assert max(ratio["period"]) == dt.date(2000, 1, 15)
        assert ratio.loc[ratio.period == dt.date(2000, 1, 15), "value"].iloc[0] == pytest.approx(
            2.0
        )

        con.raw_sql(
            "INSERT INTO scoped_horizon_events VALUES "
            "('u1', TIMESTAMP '2020-01-02 12:00:00', 'purchase', 9.0)"
        )
        assert max(trends["fresh"].to_frame()["period"]) == dt.date(2000, 1, 10)
    finally:
        con.disconnect()


def test_omitted_end_executes_bound_horizon_aggregate(report, monkeypatch):
    executed = []
    original_execute = report._con.execute

    def record_execute(expression, *args, **kwargs):
        executed.append(ibis.to_sql(expression))
        return original_execute(expression, *args, **kwargs)

    monkeypatch.setattr(report._con, "execute", record_execute)
    trend = report.metric("revenue", grain="day", start=dt.date(2026, 1, 5))
    assert trend.end is None
    assert len(executed) == 1
    assert "raw_events" in executed[0]
    assert "VALUES" not in executed[0].upper()


def test_omitted_end_ratio_aggregates_bound_component_relations(report, monkeypatch):
    executed = []
    original_execute = report._con.execute

    def record_execute(expression, *args, **kwargs):
        executed.append(ibis.to_sql(expression))
        return original_execute(expression, *args, **kwargs)

    monkeypatch.setattr(report._con, "execute", record_execute)
    trend = report.metric("rev_per_order", grain="day", start=dt.date(2026, 1, 5))
    assert trend.end is None
    assert len(executed) == 1
    assert "raw_events" in executed[0] and "raw_orders" in executed[0]
    assert "VALUES" not in executed[0].upper()


def test_omitted_end_population_admission_precedes_data_execution(report, monkeypatch):
    executions = []
    original_execute = report._con.execute

    def record_execute(expression, *args, **kwargs):
        executions.append(expression)
        return original_execute(expression, *args, **kwargs)

    monkeypatch.setattr(report._con, "execute", record_execute)
    for invalid_sql in ("SELEC FROM raw_events", "DELETE FROM raw_events"):
        with pytest.raises(InvalidRequestError):
            report.metric(
                "revenue",
                grain="day",
                start=dt.date(2026, 1, 5),
                population=invalid_sql,
            )
        assert executions == []

    with pytest.raises(InvalidRequestError) as raised:
        report.metric(
            "revenue",
            grain="day",
            start=dt.date(2026, 1, 5),
            population=ibis.memtable({"not_unit_id": ["u1"]}),
        )
    assert raised.value.code == "query.calendar.population_unit_id"
    assert executions == []


def test_range_beyond_capacity_refused_through_facade(report):
    with pytest.raises(CapabilityError) as raised:
        report.metric(
            "total_revenue", grain="week", start=dt.date(2026, 1, 1), end=dt.date(2036, 2, 1)
        )
    assert raised.value.code == "query.calendar.range"
    assert dict(raised.value.context) == {
        "start": dt.date(2026, 1, 1),
        "end_edge": dt.date(2036, 2, 1),
        "span": (dt.date(2036, 2, 1) - dt.date(2026, 1, 1)).days + 1,
    }


def test_zero_row_fact_ratio_numerator_horizon_never_complete(report):
    # Ratio branch's ibis.least(num_horizon, den_horizon) ignores NULLs in
    # DuckDB, so an unguarded zero-row numerator would let the populated denominator alone drive completeness; the sentinel coalesce fixes this.
    frame = report.metric(
        "refund_ratio", grain="week", start=dt.date(2026, 1, 5), end=dt.date(2026, 1, 18)
    ).to_frame()
    assert len(frame) > 0
    assert (frame.period_complete == False).all()  # noqa: E712 - not NULL/NaN


def test_zero_row_fact_ratio_numerator_default_end_empty_frame(report):
    frame = report.metric("refund_ratio", grain="week", start=dt.date(2026, 1, 5)).to_frame()
    assert len(frame) == 0


KW: dict[str, Any] = {"grain": "week", "start": dt.date(2026, 1, 5), "end": dt.date(2026, 1, 18)}


def test_conversion_trend_shape_and_wilson_ci(report):
    frame = report.metric("purchase_rate", **KW).to_frame()
    assert set(frame.columns) == {
        "metric",
        "grain",
        "window",
        "period",
        "period_complete",
        "n",
        "value",
        "ci_lb",
        "ci_ub",
    }
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    # Active-in-source: u1 (purchased), u2 (page_view only) -> 1/2.
    assert wk1.n == 2
    assert wk1.value == pytest.approx(0.5)
    assert wk1.metric == "purchase_rate"
    assert wk1.grain == "week"
    # Wilson: bounded to [0, 1] and non-degenerate even at tiny n
    # (Wald at n=2, p=0.5 would give [-0.19, 1.19]).
    assert 0.0 <= wk1.ci_lb < wk1.value < wk1.ci_ub <= 1.0


def test_wilson_ci_finite_at_tiny_alpha(report):
    """norm.ppf(1 - alpha/2) rounds the complement to exactly 1.0 once
    alpha is small enough, returning z=inf and poisoning both CI bounds
    to NaN. The survival-form critical value (isf(alpha/2)) stays finite
    at this alpha."""
    frame = report.metric("purchase_rate", **KW, alpha=1e-16).to_frame()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    assert math.isfinite(wk1.ci_lb)
    assert math.isfinite(wk1.ci_ub)


def test_wilson_ci_not_zero_width_at_extreme_rate(report):
    pop = ibis.memtable([{"unit_id": "u1"}])
    frame = report.metric("order_rate", population=pop, **KW).to_frame()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    assert wk1.value == pytest.approx(1.0)  # p-hat == 1
    assert wk1.ci_lb < 1.0  # Wald would collapse to zero width here
    assert wk1.ci_ub <= 1.0


def test_mean_trend_normal_ci_and_n_equals_one_edge_case(report):
    frame = report.metric("revenue", **KW).to_frame()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    # Week 1 active-in-source: u1(purchase=10), u2(page_view->0). n=2, mean=5.0,
    # var=(100-10**2/2)/(2-1)=50.0, se=sqrt(50/2)=5.0.
    assert wk1.n == 2
    assert wk1.value == pytest.approx(5.0)
    z = 1.9599639845400545  # norm.ppf(0.975), alpha=0.05 default
    se = (50.0 / 2) ** 0.5
    assert wk1.ci_lb == pytest.approx(5.0 - z * se)
    assert wk1.ci_ub == pytest.approx(5.0 + z * se)

    wk2 = frame[frame.period == dt.date(2026, 1, 12)].iloc[0]
    # Week 2 active-in-source: only u2(purchase=4), n=1. Sample variance's
    # (n-1) denominator is guarded (nullif(0)): CI must be NULL, not a crash.
    assert wk2.n == 1
    assert wk2.value == pytest.approx(4.0)
    assert wk2.ci_lb != wk2.ci_lb or wk2.ci_lb is None  # NaN or None
    assert wk2.ci_ub != wk2.ci_ub or wk2.ci_ub is None


def test_mean_trend_variance_survives_extreme_mean():
    """The trend CI comes from the producer-centered moments, so its SE
    stays exact-grade when the mean dwarfs the spread. The raw-sum
    reduction this path used to run - (sum_y2 - sum_y**2/n)/(n-1) -
    loses the ENTIRE variance signal in float64 by mu ~ 1e9 (mirrors
    tests/estimation/test_moment_conditioning.py; the exact-rational
    reference is over the float64 values actually stored). The looser SE
    tolerance at mu=1e9 is the ulp of the reported CI ENDPOINT
    (mean-scale), not of the variance path under test.
    """
    z = 1.9599639845400545  # norm.ppf(0.975), alpha=0.05 default
    for mu, se_rel, raw_floor in ((1.0, 1e-12, None), (1e9, 1e-6, 1e-2)):
        y = [mu - 1.0, mu, mu + 1.0]  # every value exact in float64 at both scales
        rows = [
            {
                "user_id": f"u{i}",
                "ts": dt.datetime(2026, 1, 5 + i, 10),
                "event": "purchase",
                "revenue": v,
                "country_code": "US",
            }
            for i, v in enumerate(y)
        ]
        con = ibis.duckdb.connect()
        con.create_table("raw_events", ibis.memtable(rows))
        con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
        report = Report.from_definitions(Definitions.model_validate(DEFS), con)
        frame = report.metric(
            "revenue", grain="week", start=dt.date(2026, 1, 5), end=dt.date(2026, 1, 11)
        ).to_frame()
        wk = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
        assert wk.n == 3

        f = [Fraction(v) for v in y]
        m = sum(f, Fraction(0)) / 3
        exact_var = sum(((t - m) ** 2 for t in f), Fraction(0)) / 2
        exact_se = math.sqrt(float(exact_var) / 3)
        assert wk.value == pytest.approx(float(m), rel=1e-15)
        assert (wk.ci_ub - wk.value) / z == pytest.approx(exact_se, rel=se_rel)

        if raw_floor is not None:
            # The retired formula, on these exact same floats: measurably wrong.
            sum_y, sum_y2 = math.fsum(y), math.fsum(v * v for v in y)
            raw_var = (sum_y2 - sum_y * sum_y / 3) / 2
            assert abs(raw_var - float(exact_var)) / float(exact_var) > raw_floor


def test_mean_trend_constant_period_has_zero_variance_not_nan():
    """Every unit at the non-representable 0.1: the centered sum of
    squares is ~1e-17 noise of unpredictable sign, and an unclamped
    ddof=1 form can go negative -> NaN SE poisoning the period's CI.
    The clamp (mirroring ArmStats._ddof1_var) pins var to exactly 0."""
    rows = [
        {
            "user_id": f"u{i}",
            "ts": dt.datetime(2026, 1, 5 + i, 10),
            "event": "purchase",
            "revenue": 0.1,
            "country_code": "US",
        }
        for i in range(3)
    ]
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(rows))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    report = Report.from_definitions(Definitions.model_validate(DEFS), con)
    frame = report.metric(
        "revenue", grain="week", start=dt.date(2026, 1, 5), end=dt.date(2026, 1, 11)
    ).to_frame()
    wk = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    assert wk.n == 3
    assert wk.value == pytest.approx(0.1, rel=1e-15)
    assert wk.ci_lb == wk.ci_lb and wk.ci_ub == wk.ci_ub  # not NaN
    assert wk.ci_lb == pytest.approx(0.1, rel=1e-12)
    assert wk.ci_ub == pytest.approx(0.1, rel=1e-12)


def test_total_and_active_have_null_n_and_ci(report):
    frame = report.metric("wau", **KW).to_frame()
    assert frame.n.isna().all()
    assert frame.ci_lb.isna().all()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    assert wk1.value == 2


def test_dimensioned_trend_carries_dim_column(report):
    frame = report.metric("total_revenue", by=["country"], **KW).to_frame()
    assert "country" in frame.columns
    wk1 = frame[(frame.period == dt.date(2026, 1, 5))]
    # Dimensioned totals are sparse: exactly the segments with events.
    assert dict(zip(wk1.country, wk1.value, strict=True)) == {"US": 10.0}


def test_sql_returns_string(report):
    sql = report.metric("purchase_rate", **KW).sql()
    assert isinstance(sql, str) and "SELECT" in sql.upper()


def test_retention_refused(report):
    with pytest.raises(CapabilityError) as raised:
        report.metric("d7", **KW)
    assert raised.value.code == "report.metric.unsupported"


def test_ratio_with_dims_refused(report):
    with pytest.raises(CapabilityError) as raised:
        report.metric("rev_per_order", by=["country"], **KW)
    assert raised.value.code == "report.metric.unsupported"


def test_degenerate_default_denominator_refused(report):
    # 'order' is the only fact on orders_only and order_rate is filterless:
    # active default == converters.
    with pytest.raises(CapabilityError) as raised:
        report.metric("order_rate", **KW)
    assert raised.value.code == "report.metric.unsupported"


def test_filtered_conversion_on_single_fact_source_allowed(report):
    # Filtered conversion is NOT degenerate: denominator = all active units,
    # numerator = filtered subset.
    frame = report.metric("web_order_rate", **KW).to_frame()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    # Active on orders_only in wk1: u1 (web), u3 (store) -> 1/2 web.
    assert wk1.n == 2
    assert wk1.value == pytest.approx(0.5)


def test_degenerate_denominator_ok_with_population(report):
    pop = ibis.memtable([{"unit_id": u} for u in ["u1", "u2", "u3"]])
    frame = report.metric("order_rate", population=pop, **KW).to_frame()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    assert wk1.n == 3
    assert wk1.value == pytest.approx(2 / 3)  # u1 and u3 ordered in wk1


def test_cross_source_ratio_population_and_horizon(report):
    frame = report.metric("rev_per_order", **KW).to_frame()
    wk1 = frame[frame.period == dt.date(2026, 1, 5)].iloc[0]
    # Population = active on either source (u1,u2,u3); revenue wk1=10 (u1),
    # orders wk1=2 (u1,u3) -> 5.0. Dropping denominator-only u3 would wrongly give 10.0.
    assert wk1.value == pytest.approx(5.0)
    assert frame.ci_lb.isna().all()  # ratio is point-only in v1
    # Orders source horizon is Jan 11 < events' Jan 13: week 1 (ends Jan 11)
    # is complete, week 2 is NOT, even though end=Jan 18 was requested.
    assert bool(wk1.period_complete)
    wk2 = frame[frame.period == dt.date(2026, 1, 12)].iloc[0]
    assert not bool(wk2.period_complete)


def test_dimensioned_conversion_end_to_end_by_country(report):
    # Entity-scoped dims are as-of-period-end, unit-level: u1 (US) purchased
    # -> converted; u2 (DE, page_view only) -> not converted.
    frame = report.metric("purchase_rate", by=["country"], **KW).to_frame()
    assert "country" in frame.columns
    wk1 = frame[frame.period == dt.date(2026, 1, 5)]
    by_country = {row.country: (row.n, row.value) for row in wk1.itertuples()}
    assert by_country["US"] == (1, pytest.approx(1.0))
    assert by_country["DE"] == (1, pytest.approx(0.0))


def test_metrics_batch_validates_upfront(report):
    """One batch call reports every offender it was handed, once each."""
    with pytest.raises(CapabilityError) as raised:
        report.metrics(["purchase_rate", "d7", "nope", "purchase_rate"], **KW)

    error = raised.value
    assert error.code == "report.metric.unsupported"
    errors = cast("tuple[str, ...]", error.context["errors"])
    assert len(errors) == 3
    for offender in ("d7", "nope", "purchase_rate"):
        assert sum(offender in message for message in errors) == 1


def test_metrics_batch_returns_dict(report):
    out = report.metrics(["purchase_rate", "wau"], **KW)
    assert set(out) == {"purchase_rate", "wau"}


@pytest.mark.parametrize("alpha", [math.nan, math.inf, -math.inf, 0.0, 1.0, -0.1, 1.1])
@pytest.mark.parametrize("batch", [False, True])
def test_invalid_alpha_rejected_before_build(report, alpha, batch):
    method = report.metrics if batch else report.metric
    names = ["purchase_rate"] if batch else "purchase_rate"

    with pytest.raises(InvalidRequestError) as exc_info:
        method(names, alpha=alpha, **KW)
    assert exc_info.value.code == "reporting.alpha_finite"


@pytest.mark.parametrize("week_start", ["tuesday", ""])
@pytest.mark.parametrize("batch", [False, True])
def test_invalid_week_start_rejected_before_build(report, week_start, batch):
    method = report.metrics if batch else report.metric
    names = ["purchase_rate"] if batch else "purchase_rate"

    with pytest.raises(InvalidRequestError) as exc_info:
        method(names, week_start=week_start, **KW)
    assert exc_info.value.code == "reporting.week_start_monday"
    assert exc_info.value.context["week_start"] == week_start


def test_to_frame_refuses_an_unknown_backend(report):
    from increment.errors import InvalidRequestError

    trend = report.metric("purchase_rate", **KW)
    with pytest.raises(InvalidRequestError) as exc_info:
        trend.to_frame(backend="spark")
    assert exc_info.value.code == "reporting.metric_trend.unknown_backend"
    assert exc_info.value.context["backend"] == "spark"


def test_unknown_dim_refused_before_query(report):
    with pytest.raises(CapabilityError) as raised:
        report.metric("purchase_rate", by=["no_such_dim"], **KW)
    assert raised.value.code == "report.metric.unsupported"


def test_metadata(report):
    t = report.metric("purchase_rate", by=["country"], **KW)
    assert (t.metric, t.grain, t.week_start, t.denominator) == (
        "purchase_rate",
        "week",
        "monday",
        "active",
    )
    assert (t.start, t.end, t.by, t.alpha) == (
        dt.date(2026, 1, 5),
        dt.date(2026, 1, 18),
        ("country",),
        0.05,
    )


def test_week_start_defaults_from_definitions():
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(ROWS))
    con.create_table("raw_orders", ibis.memtable(ORDER_ROWS))
    defs = Definitions.model_validate({**DEFS, "week_start": "sunday"})
    t = Report.from_definitions(defs, con).metric("purchase_rate", **KW)
    assert t.week_start == "sunday"  # governed default, no per-call arg


def test_rolling_mau(report):
    t = report.metric(
        "wau", grain="day", window=7, start=dt.date(2026, 1, 10), end=dt.date(2026, 1, 16)
    )
    frame = t.to_frame()
    # page_views: u1 Jan6, u2 Jan7. Trailing-7d actives: Jan10-12 see both
    # (leading events count); Jan13 sees only u2; Jan14-16 see neither -> 0.
    assert list(frame.sort_values("period").value) == [2, 2, 2, 1, 0, 0, 0]
    assert (frame.window == 7).all()
    assert t.window == 7


def test_rolling_total_revenue(report):
    # Confirms the facade routes TotalMetric (not just ActiveMetric) to
    # rolling_total_values when window= is set.
    t = report.metric(
        "total_revenue", grain="day", window=7, start=dt.date(2026, 1, 10), end=dt.date(2026, 1, 16)
    )
    frame = t.to_frame()
    # purchase revenue: u1 Jan5=10.0, u2 Jan13=4.0. Trailing-7d sum sees
    # only Jan5 on Jan10-11, neither on Jan12, only Jan13 on Jan13-16.
    assert list(frame.sort_values("period").value) == [10.0, 10.0, 0.0, 4.0, 4.0, 4.0, 4.0]
    assert (frame.window == 7).all()


def test_window_null_for_calendar_buckets(report):
    frame = report.metric("wau", **KW).to_frame()
    assert frame.window.isna().all()


def test_rolling_refused_on_entity_scoped_metrics(report):
    with pytest.raises(CapabilityError) as raised:
        report.metric("purchase_rate", window=28, **KW)
    assert raised.value.code == "report.metric.unsupported"


def test_nonpositive_window_refused(report):
    with pytest.raises(CapabilityError) as raised:
        report.metric("wau", window=0, **KW)
    assert raised.value.code == "report.metric.unsupported"


def test_report_population_is_not_silently_dropped_for_total_metrics():
    """A total metric must honor population= or refuse it, and the read-only
    SQL admission must run on the argument before anything else."""
    import datetime as dt

    import ibis
    import pytest

    from increment import CapabilityError, Report
    from increment.semantics.models import Definitions

    defs = {
        "fact_sources": [
            {
                "name": "events",
                "sql": "SELECT * FROM raw_events",
                "timestamp_column": "ts",
                "entities": ["user_id"],
                "facts": [{"name": "purchase", "column": "revenue"}],
            }
        ],
        "metrics": [
            {"type": "total", "name": "total_revenue", "fact": "purchase", "aggregation": "sum"}
        ],
    }
    rows = [
        {"user_id": "u1", "ts": dt.datetime(2026, 1, 5, 10), "event": "purchase", "revenue": 10.0},
        {"user_id": "u2", "ts": dt.datetime(2026, 1, 13, 9), "event": "purchase", "revenue": 4.0},
    ]
    con = ibis.duckdb.connect()
    con.create_table("raw_events", ibis.memtable(rows))
    report = Report.from_definitions(Definitions.model_validate(defs), con)
    kwargs = {"grain": "week", "start": dt.date(2026, 1, 1)}
    baseline = sorted(
        report.metric("total_revenue", **kwargs)  # ty: ignore[invalid-argument-type]
        .to_frame()["value"]
        .tolist()
    )

    with pytest.raises((ValueError, CapabilityError)):
        report.metric(
            "total_revenue",
            population="DELETE FROM raw_events",
            **kwargs,  # ty: ignore[invalid-argument-type]
        )

    try:
        scoped = report.metric(
            "total_revenue",
            population="SELECT user_id FROM raw_events WHERE 1 = 0",
            **kwargs,  # ty: ignore[invalid-argument-type]
        ).to_frame()
    except (ValueError, CapabilityError):
        return  # refusing the unsupported selector outright is also correct
    assert sorted(scoped["value"].tolist()) != baseline
