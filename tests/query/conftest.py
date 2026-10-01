"""In-memory DuckDB fixtures for the query-layer builder tests.

Every fixture comes with hand-computed expected values below; the dataset is tiny
(3-4 units, 4-6 days) so results are trivially verifiable. Referenced throughout
test_builders.py as "the conftest module docstring's hand-computed values".

Exposure events (raw log rows): u1 exp_test/treatment 08-01 09:00 (dup at 08-02
09:00); u2 exp_test/control 08-01 10:00; u3 exp_test/treatment 08-02 11:00; u4
exposed both treatment (08-03 09:00) and control (08-03 10:00) -> mixed, dropped.
first_exposures: u1 08-01 09:00 treatment, u2 08-01 10:00 control, u3 08-02 11:00
treatment (u4 dropped).

Purchase events (event='purchase', value column amount): u1 08-01 09:05 $49.99,
u1 08-01 09:10 $12.00, u1 08-03 14:00 $7.50, u2 08-01 09:30 $25.00 (PRE-exposure,
ts < 10:00), u2 08-02 08:00 $30.00, u3 08-03 14:40 $19.99.

Page-view events (occurrence-only): u1 08-01 09:06, u1 08-03 15:00, u2 08-02 08:01,
u3 08-03 14:41.

Experiment "exp_test": unit=unit_id, start=08-01, end=08-06 (6-day window),
control_group=control.

Conversion metric "converted" (window_days=1): any purchase in
[first_exposure, first_exposure+1d) -> 1 else 0. u1 exposed 08-01 09:00, window
[08-01,08-02), purchase 08-01 -> y=1. u2 exposed 08-01 10:00, window [08-01,08-02),
its 09:30 purchase excluded (pre-exposure), no other -> y=0. u3 exposed 08-02 11:00,
window [08-02,08-03), purchase on 08-03 is outside -> y=0.

Mean metric "revenue" (aggregation=sum, window_days=3): window = first_exposure_date
 + 3 days. u1 window [08-01,08-04): 08-01=61.99, 08-02=0, 08-03=7.50 -> y=69.49.
u2 window [08-01,08-04): 08-01=0 (pre-exclusion), 08-02=30.00, 08-03=0 -> y=30.00.
u3 window [08-02,08-05): 08-02=0, 08-03=19.99, 08-04=0 -> y=19.99.

Retention metric "retained" (page_view, threshold_days=2): active on/after
first_exposure_date+2. u1 threshold 08-03, page view 08-03 -> y=1. u2 threshold
08-03, page view 08-02 only -> y=0. u3 threshold 08-04, page view 08-03 only -> y=0.

Bounded retention metric "retained_bounded" (page_view, threshold_days=[2,4]):
active in half-open band [first_exposure_date+2, +4). u1 band [08-03,08-05), page
view 08-03 -> y=1. u2 band [08-03,08-05), page view 08-02 only -> y=0. u3 band
[08-04,08-06), page view 08-03 only -> y=0.

Censored experiment exp_censor (end=2025-08-03): u5 exposed 08-03 12:00 (treatment).
Censoring keeps a unit once the last day its outcome depends on is observable:
conversion window_days=1 -> last window day 08-03 <= 08-03, KEPT; conversion
window_days=2 -> last window day 08-04 > 08-03, DROPPED; retention threshold_days=2
-> threshold day 08-05 > 08-03, DROPPED.
"""

from __future__ import annotations

import datetime as dt

import ibis
import pytest
from duckdb import FatalException

from increment.semantics.models import (
    AnalysisPlan,
    ConversionMetric,
    Experiment,
    MeanMetric,
    RetentionMetric,
)

# ── Shared DuckDB connection (all tables live in one database) ─────────

# Rows of every table `_table` created on `con`, replayed when a fatal DuckDB
# error forces `_no_stray_tables` to reopen the connection.
_SEED_ROWS: dict[str, list[dict]] = {}


@pytest.fixture(scope="session")
def con():
    """Single in-memory DuckDB connection shared by all fixtures.

    This avoids ``CatalogException`` when an ibis expression that
    references tables from multiple fixtures gets executed — all
    tables are in the same database.
    """
    return ibis.duckdb.connect()


def _base_tables(con) -> set[str]:
    """Names of real tables on *con*, excluding views.

    Executing an ``ibis.memtable(...)`` makes ibis push it into the
    backend as a VIEW, while ``create_table`` produces a BASE TABLE. That
    is a property of what the object IS, so the guard below discriminates
    on it rather than on ibis's generated names, which are an internal
    detail that changes without notice.
    """
    rows = con.raw_sql(
        "select table_name from information_schema.tables where table_type = 'BASE TABLE'"
    ).fetchall()
    return {name for (name,) in rows}


def _base_tables_or_reopen(con) -> set[str]:
    """`_base_tables`, or fail this test once and reopen `con` for later ones.

    A fatal DuckDB error (an internal assertion, say) invalidates the whole
    in-memory database, so every later statement on the connection fails
    until it is reopened. Reconnecting the same backend object keeps the
    session fixtures' cached table expressions valid; replaying the seed
    rows restores the tables they name.
    """
    try:
        return _base_tables(con)
    except FatalException as exc:
        cause = exc
    con.reconnect()
    for name, rows in _SEED_ROWS.items():
        con.create_table(name, obj=rows)
    pytest.fail(
        "the shared `con` connection was invalidated by a fatal DuckDB error; "
        f"it has been reopened for later tests. {cause}"
    )


@pytest.fixture(autouse=True)
def _no_stray_tables(request, con):
    """Fail a test that leaves new tables on the shared ``con`` connection.

    This guards leak hygiene: state bleeding from one test into a later
    one on the same worker. Mark a test ``@pytest.mark.creates_tables``
    (per-test or as a module-level ``pytestmark``) if it deliberately
    leaves tables behind; otherwise drop what you create before the test
    ends.

    A marked module is NOT an unguarded module. The companion failure --
    a test READING a table some earlier test left behind -- is caught by
    running the suite the way it actually runs, ``-n auto``: xdist
    scatters the pair onto separate workers with separate connections, so
    the reader raises TableNotFound. Verified by planting such a pair in
    the most heavily marked module and watching it fail in parallel while
    passing in serial file order. That is why the original instance of
    this bug went unnoticed for as long as the suite ran serially.

    Only base tables are checked, so a test that deliberately left a VIEW
    behind would slip through. Nothing in the suite creates one; if that
    changes, compare views too and exclude ibis's pushed memtables some
    other way.
    """
    before = _base_tables_or_reopen(con)
    yield
    after = _base_tables_or_reopen(con)
    if request.node.get_closest_marker("creates_tables") is not None:
        return
    leaked = after - before
    if leaked:
        pytest.fail(
            f"left new table(s) on the shared `con` connection: {sorted(leaked)}. "
            "Drop them before the test ends, or mark the test "
            "@pytest.mark.creates_tables if leaving them is intentional."
        )


def _table(con, rows: list[dict], name: str):
    """Create or reuse a table on the shared connection, remembering its rows."""
    if name in con.list_tables():
        return con.table(name)
    _SEED_ROWS[name] = rows
    return con.create_table(name, obj=rows)


# ── Raw data fixtures ──────────────────────────────────────────────────


@pytest.fixture(scope="session")
def exposure_events(con):
    return _table(
        con,
        [
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },  # dup
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "control",
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 8, 2, 11, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "u4",
                "ts": dt.datetime(2025, 8, 3, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "u4",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "control",
            },  # mixed
        ],
        "exposure_events",
    )


@pytest.fixture(scope="session")
def purchase_events(con):
    """Raw purchase events with 'amount' value column."""
    return _table(
        con,
        [
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 5, 0),
                "event": "purchase",
                "amount": 49.99,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 10, 0),
                "event": "purchase",
                "amount": 12.00,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 3, 14, 0, 0),
                "event": "purchase",
                "amount": 7.50,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 9, 30, 0),
                "event": "purchase",
                "amount": 25.00,
            },  # PRE-exposure
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 2, 8, 0, 0),
                "event": "purchase",
                "amount": 30.00,
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 8, 3, 14, 40, 0),
                "event": "purchase",
                "amount": 19.99,
            },
        ],
        "purchase_events",
    )


@pytest.fixture(scope="session")
def page_view_events(con):
    """Raw page-view events (occurrence-only, no value column)."""
    return _table(
        con,
        [
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 1, 9, 6, 0), "event": "page_view"},
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 3, 15, 0, 0), "event": "page_view"},
            {"unit_id": "u2", "ts": dt.datetime(2025, 8, 2, 8, 1, 0), "event": "page_view"},
            {"unit_id": "u3", "ts": dt.datetime(2025, 8, 3, 14, 41, 0), "event": "page_view"},
        ],
        "page_view_events",
    )


@pytest.fixture(scope="session")
def censored_exposure_events(con):
    """Exposure events for the ``exp_censor`` experiment (end = Aug 3)."""
    return _table(
        con,
        [
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_censor",
                "group_id": "treatment",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_censor",
                "group_id": "control",
            },
            {
                "unit_id": "u5",
                "ts": dt.datetime(2025, 8, 3, 12, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_censor",
                "group_id": "treatment",
            },  # late -> censored
        ],
        "censored_exposure_events",
    )


@pytest.fixture(scope="session")
def censored_purchase_events(con):
    """Minimal purchase events for censoring tests."""
    return _table(
        con,
        [
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 5, 0),
                "event": "purchase",
                "amount": 10.00,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 30, 0),
                "event": "purchase",
                "amount": 20.00,
            },
            {
                "unit_id": "u5",
                "ts": dt.datetime(2025, 8, 3, 13, 0, 0),
                "event": "purchase",
                "amount": 5.00,
            },
        ],
        "censored_purchase_events",
    )


@pytest.fixture(scope="session")
def censored_page_view_events(con):
    """Minimal page-view events for censoring tests."""
    return _table(
        con,
        [
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 3, 9, 0, 0), "event": "page_view"},
            {"unit_id": "u5", "ts": dt.datetime(2025, 8, 3, 13, 0, 0), "event": "page_view"},
        ],
        "censored_page_view_events",
    )


# ── Model fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def experiment():
    return Experiment(
        name="exp_test",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 6),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )


@pytest.fixture
def conversion_metric():
    return ConversionMetric(
        name="converted",
        entity="unit_id",
        fact="purchase",
        window_days=1,
    )


@pytest.fixture
def mean_metric():
    return MeanMetric(
        name="revenue",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        window_days=3,
    )


@pytest.fixture
def retention_metric():
    return RetentionMetric(
        name="retained",
        entity="unit_id",
        fact="page_view",
        threshold_days=2,
    )


@pytest.fixture
def bounded_retention_metric():
    """Retention with a real observation band: [fe+2, fe+4).

    Against the canonical page-view fixture:
      u1: exposed Aug 1, band [Aug 3, Aug 5).  Page view Aug 3  -> y = 1
      u2: exposed Aug 1, band [Aug 3, Aug 5).  Page view Aug 2 only -> y = 0
      u3: exposed Aug 2, band [Aug 4, Aug 6).  Page view Aug 3 only -> y = 0
    Maturity: u1/u2 mature Aug 5, u3 matures Aug 6.
    """
    return RetentionMetric(
        name="retained_bounded",
        entity="unit_id",
        fact="page_view",
        threshold_days=(2, 4),
    )


@pytest.fixture
def censored_experiment():
    """Experiment with a short window - some units will be censored."""
    return Experiment(
        name="exp_censor",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 3),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )


# ── Convenience: pre-built derived tables ──────────────────────────────


@pytest.fixture
def exposures(experiment, exposure_events):
    """Pre-computed first_exposures for the main experiment."""
    from increment.query.builders import first_exposures

    return first_exposures(exposure_events, experiment)


@pytest.fixture
def purchase_metric_events(mean_metric, purchase_events):
    """Pre-computed metric_events for purchase/revenue."""
    from increment.query.builders import metric_events

    return metric_events(purchase_events, mean_metric, value_column="amount")


@pytest.fixture
def panel(exposures, purchase_metric_events, experiment, mean_metric):
    """Pre-computed unit_day_panel for the mean metric."""
    from increment.query.builders import unit_day_panel

    return unit_day_panel(
        exposures, purchase_metric_events, experiment, metric_name=mean_metric.name
    )


@pytest.fixture
def totals(exposures, purchase_metric_events, mean_metric, experiment):
    """Pre-computed unit_totals for the mean metric."""
    from increment.query.builders import unit_day_spine_stats, unit_totals

    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    return unit_totals(spine, stats, mean_metric, experiment)
