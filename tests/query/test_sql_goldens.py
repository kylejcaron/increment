from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import ibis
import pytest

from increment import from_definitions
from increment.query import panel_sql, summary_sql
from tests._unit_boundary import backend_fixture

pytestmark = pytest.mark.integration


@backend_fixture()
def sql_analysis(tmp_path):
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "unit_id": "u1",
                "user_id": "u1",
                "ts": datetime(2024, 1, 1, 9),
                "event": "exposure",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "user_id": "u2",
                "ts": datetime(2024, 1, 31, 9, 5),
                "event": "purchase",
                "group_id": "treatment",
            },
        ],
    )
    definitions = tmp_path / "definitions.yaml"
    definitions.write_text(
        """dialect: duckdb
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: exposure
        column: event
      - name: purchase
        column: event
exposures:
  - name: assigned
    sql: "SELECT user_id AS unit_id, ts, group_id FROM events WHERE event = 'exposure'"
metrics:
  - type: conversion
    name: purchase_rate
    entity: user_id
    fact: purchase
    window_days: 7
experiments:
  - name: sql_golden
    exposure: assigned
    unit: user_id
    start: '2024-01-01'
    end: '2024-01-31'
    control_group: control
    plan:
      secondaries: [purchase_rate]
""",
        encoding="utf-8",
    )
    experiment, resolver = from_definitions(
        definitions, con, experiment_name="sql_golden", store="none"
    )
    try:
        yield (experiment, resolver), con
    finally:
        cast(Any, resolver).close()
        con.disconnect()


def test_exported_panel_sql_keeps_zero_outcomes_and_excludes_unassigned_units(sql_analysis):
    binding, con = sql_analysis
    statements = panel_sql(*binding)
    rows = con.sql(statements["purchase_rate"]).to_pyarrow()
    assert rows.select(["unit_id", "day_offset", "n_events", "sum_value"]).sort_by(
        [("day_offset", "ascending")]
    ).to_pylist() == [
        {"unit_id": "u1", "day_offset": day, "n_events": 0, "sum_value": 0.0} for day in range(31)
    ]


def test_exported_summary_sql_counts_mature_assigned_units_without_purchases(sql_analysis):
    binding, con = sql_analysis
    statements = summary_sql(*binding)
    rows = con.sql(statements["purchase_rate"]).to_pyarrow()
    assert rows.select(["group_id", "n", "successes"]).to_pylist() == [
        {"group_id": "control", "n": 1, "successes": 0}
    ]
