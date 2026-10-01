"""End-to-end Analysis over declarative dim sources (in-memory DuckDB).

Includes the collider guard: a unit whose versioned value changes AFTER
exposure must resolve to the pre-exposure value in breakouts.

`Analysis` loads definitions from a path (`Analysis(experiment_name,
definitions_path, con)`), so each variant writes its YAML to tmp_path;
the warehouse tables live on the shared in-memory DuckDB connection.
"""

import datetime as dt
import math
import textwrap
import warnings

import ibis
import pytest

from increment import Analysis, Report
from increment.semantics.models import AnalysisPlan, MultiplicitySpec
from tests.analysis_factory import make_analysis_like
from tests.warning_codes import warning_codes

START = dt.datetime(2025, 1, 10)
UPGRADE = dt.datetime(2025, 1, 20)  # after every exposure below
FOREVER = dt.datetime(9999, 12, 31)

SHARED_YAML = textwrap.dedent(
    """
    dialect: duckdb
    exposures:
      - name: exposed
        fact: exposure
    metrics:
      - name: conversion
        type: conversion
        entity: user_id
        fact: order
        window_days: 14
    experiments:
      - name: exp
        unit: user_id
        exposure: exposed
        control_group: control
        start: "2025-01-01"
        plan:
          secondaries:
            - conversion
        breakouts:
          - property: plan
            source: orders
          - property: country
            source: orders
    """
)

# The exposure rides its own fact source, mirroring the realistic_demo
# shape. Deliberately NOT dedent()'d to column 0: appended after an already-opened `fact_sources:` block, so it must keep its 2-space indent to parse as a sibling list item, not a stray top-level key.
EXPOSURE_SOURCE_YAML = (
    "\n"
    "  - name: exposure_src\n"
    "    sql: SELECT 'exposure' AS event, user_id, experiment_id, group_id, exposed_at FROM exposures\n"
    "    timestamp_column: exposed_at\n"
    "    entities: [user_id]\n"
    "    facts:\n"
    "      - name: exposure\n"
    "        column: null\n"
)

DECLARATIVE_YAML = (
    SHARED_YAML
    + textwrap.dedent(
        """
    dim_sources:
      - name: users
        sql: SELECT user_id, country FROM dim_user
        entity: user_id
        properties:
          - name: country
            column: country
            as_of: static
      - name: user_plan
        sql: SELECT user_id, plan, valid_from, valid_to FROM snap_user_plan
        entity: user_id
        validity:
          valid_from: valid_from
          valid_to: valid_to
        properties:
          - name: plan
            column: plan
            as_of: pre_exposure
    fact_sources:
      - name: orders
        sql: SELECT 'order' AS event, user_id, ordered_at, amount FROM fact_orders
        timestamp_column: ordered_at
        entities: [user_id]
        dims: [users, user_plan]
        facts:
          - name: order
            column: amount
    """
    )
    + EXPOSURE_SOURCE_YAML
)

INLINE_YAML = (
    SHARED_YAML
    + textwrap.dedent(
        """
    fact_sources:
      - name: orders
        sql: |
          SELECT 'order' AS event, o.user_id, o.ordered_at, o.amount,
                 u.country AS country, p.plan AS plan
          FROM fact_orders o
          JOIN dim_user u ON o.user_id = u.user_id
          JOIN snap_user_plan p
            ON o.user_id = p.user_id
           AND o.ordered_at >= p.valid_from
           AND o.ordered_at <  p.valid_to
        timestamp_column: ordered_at
        entities: [user_id]
        properties:
          - name: country
            column: country
            as_of: static
          - name: plan
            column: plan
            as_of: pre_exposure
        facts:
          - name: order
            column: amount
    """
    )
    + EXPOSURE_SOURCE_YAML
)

N = 40  # small: this file must stay well under the fast-suite budget


@pytest.fixture
def con():
    con = ibis.duckdb.connect()
    users = [f"u{i:03d}" for i in range(N)]
    con.create_table(
        "dim_user",
        # Deliberately NOT keyed on i % 2 (the group_id split): a breakout
        # property correlated with the arm split degenerates that segment to a single arm (no comparison possible).
        ibis.memtable(
            {"user_id": users, "country": ["DE" if i % 5 == 0 else "US" for i in range(N)]}
        ),
    )
    # Everyone starts 'free'; every 4th user upgrades AFTER exposure.
    plan_rows = {"user_id": [], "plan": [], "valid_from": [], "valid_to": []}
    for i, u in enumerate(users):
        upgrades = i % 4 == 0
        plan_rows["user_id"].append(u)
        plan_rows["plan"].append("free")
        plan_rows["valid_from"].append(dt.datetime(2025, 1, 1))
        plan_rows["valid_to"].append(UPGRADE if upgrades else FOREVER)
        if upgrades:
            plan_rows["user_id"].append(u)
            plan_rows["plan"].append("pro")
            plan_rows["valid_from"].append(UPGRADE)
            plan_rows["valid_to"].append(FOREVER)
    con.create_table("snap_user_plan", ibis.memtable(plan_rows))
    con.create_table(
        "exposures",
        ibis.memtable(
            {
                "user_id": users,
                "experiment_id": ["exp"] * N,
                "group_id": ["treatment" if i % 2 else "control" for i in range(N)],
                "exposed_at": [START] * N,
            }
        ),
    )
    # A pre-exposure order establishes 'free' as everyone's pre-exposure plan
    # (else the pre-exposure breakout resolves to "__null__"); a post-upgrade order for a third of users (NOT keyed on i%2, else one arm's conversion is degenerate) makes plan-at-order-time non-degenerate in both arms.
    buyers = [u for i, u in enumerate(users) if i % 3 == 0]
    con.create_table(
        "fact_orders",
        ibis.memtable(
            {
                # Last row: a late, otherwise-inert order for an
                # already-converted buyer, pushing data_as_of past every unit's 14-day window close - without it every unit is censored and every arm is empty.
                "user_id": users + buyers + [buyers[0]],
                "ordered_at": [START - dt.timedelta(days=1)] * N
                + [UPGRADE + dt.timedelta(days=1)] * len(buyers)
                + [dt.datetime(2025, 2, 15)],
                "amount": [10.0] * (N + len(buyers) + 1),
            }
        ),
    )
    return con


def _analysis(tmp_path, con, yaml_text, subdir):
    d = tmp_path / subdir
    d.mkdir()
    (d / "definitions.yaml").write_text(yaml_text)
    return Analysis("exp", d, con)


def test_declarative_dims_match_inline_sql(tmp_path, con):
    inline = _analysis(tmp_path, con, INLINE_YAML, "inline").run()
    declarative = _analysis(tmp_path, con, DECLARATIVE_YAML, "declarative").run()
    by_metric_inline = {e.metric: e for e in inline}
    by_metric_decl = {e.metric: e for e in declarative}
    assert by_metric_inline.keys() == by_metric_decl.keys()
    for name, e_inline in by_metric_inline.items():
        e_decl = by_metric_decl[name]
        assert math.isfinite(e_decl.require_lift().value)
        assert e_decl.require_lift().value == pytest.approx(
            e_inline.require_lift().value, rel=1e-12
        )


def test_pre_exposure_breakout_never_sees_post_exposure_value(tmp_path, con):
    a = _analysis(tmp_path, con, DECLARATIVE_YAML, "declarative")
    a = make_analysis_like(
        a,
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
    )
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            estimates = a.run_breakout()
    assert all(code.startswith("breakout.estimates.") for code in warning_codes(record))
    plan_values = {e.dimension_value for e in estimates if e.dimension == "plan"}
    assert "pro" not in plan_values  # every upgrade happened after exposure
    assert "free" in plan_values


def test_summary_sql_contains_generated_joins(tmp_path, con):
    a = _analysis(tmp_path, con, DECLARATIVE_YAML, "declarative")
    sqls = a.summary_sql()  # dict keyed per metric
    joined = " ".join(sqls.values()).lower()
    assert "dim_user" in joined
    assert "snap_user_plan" in joined


def test_report_trend_over_dim_backed_dimension(tmp_path, con):
    # The calendar/analytics path: Report must see dim-joined columns too.
    d = tmp_path / "report"
    d.mkdir()
    (d / "definitions.yaml").write_text(DECLARATIVE_YAML)
    report = Report.from_definitions(d, con)
    trend = report.metric(
        "conversion",
        by=["country"],
        start=dt.date(2025, 1, 1),
        population="SELECT user_id AS unit_id FROM dim_user",
    )
    assert "dim_user" in trend.sql().lower()
    assert {"DE", "US"} <= set(trend.to_frame()["country"])


def test_two_fact_sources_sharing_one_dim_join_their_own_rows():
    """Two fact sources declaring the same dim each join their own rows to
    it: the second source must not reuse the first's resolved relation.
    """
    from increment.query.session import WarehouseSession
    from increment.semantics.models import Definitions

    con = ibis.duckdb.connect()
    con.create_table("dim_user", ibis.memtable({"user_id": ["u1", "u2"], "country": ["DE", "US"]}))
    con.create_table(
        "fact_a",
        ibis.memtable({"user_id": ["u1"], "ts": [dt.datetime(2025, 1, 1)], "amount": [1.0]}),
    )
    con.create_table(
        "fact_b", ibis.memtable({"user_id": ["u2"], "ts": [dt.datetime(2025, 1, 2)], "clicks": [5]})
    )

    defs = Definitions.model_validate(
        {
            "dim_sources": [
                {
                    "name": "users",
                    "sql": "SELECT user_id, country FROM dim_user",
                    "entity": "user_id",
                    "properties": [{"name": "country", "column": "country", "as_of": "static"}],
                }
            ],
            "fact_sources": [
                {
                    "name": "a",
                    "sql": "SELECT 'x' AS event, user_id, ts, amount FROM fact_a",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "dims": ["users"],
                    "facts": [{"name": "x", "column": "amount"}],
                },
                {
                    "name": "b",
                    "sql": "SELECT 'y' AS event, user_id, ts, clicks FROM fact_b",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "dims": ["users"],
                    "facts": [{"name": "y", "column": "clicks"}],
                },
            ],
        }
    )

    session = WarehouseSession(con, defs)
    tbl_a = session.fact_table(defs.fact_sources[0], "user_id")
    tbl_b = session.fact_table(defs.fact_sources[1], "user_id")

    assert tbl_a.execute()["country"].tolist() == ["DE"]
    assert tbl_b.execute()["country"].tolist() == ["US"]
