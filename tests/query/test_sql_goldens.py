from __future__ import annotations

import itertools
import json
import uuid
from pathlib import Path
from typing import Any, cast

import ibis
import pytest

from increment import from_definitions
from increment.query import panel_sql, summary_sql
from tests._unit_boundary import backend_fixture

pytestmark = pytest.mark.integration

GOLDENS = Path(__file__).parents[1] / "goldens" / "sql"


def render_sql_goldens(binding: tuple[Any, Any]) -> dict[str, dict[str, str]]:
    return {
        "panel": {key: str(value) for key, value in sorted(panel_sql(*binding).items())},
        "summary": {key: str(value) for key, value in sorted(summary_sql(*binding).items())},
    }


@backend_fixture()
def sql_analysis(tmp_path, monkeypatch):
    generated_names = itertools.count(1)
    monkeypatch.setattr(
        uuid,
        "uuid4",
        lambda: uuid.UUID(int=next(generated_names)),
    )
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "unit_id": "u1",
                "user_id": "u1",
                "ts": "2024-01-01 09:00:00",
                "event": "exposure",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "user_id": "u2",
                "ts": "2024-01-01 09:05:00",
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
        yield experiment, resolver
    finally:
        cast(Any, resolver).close()
        con.disconnect()


def _golden(name: str) -> dict[str, object]:
    return json.loads((GOLDENS / f"{name}.sql.json").read_text(encoding="utf-8"))


def test_panel_sql_matches_golden_snapshot(sql_analysis):
    assert {
        "schema_version": 1,
        "dialect": "duckdb",
        "statements": render_sql_goldens(sql_analysis)["panel"],
    } == _golden("panel")


def test_summary_sql_matches_golden_snapshot(sql_analysis):
    assert {
        "schema_version": 1,
        "dialect": "duckdb",
        "statements": render_sql_goldens(sql_analysis)["summary"],
    } == _golden("summary")


def test_sql_golden_keys_are_stable_and_nonempty(sql_analysis):
    rendered = render_sql_goldens(sql_analysis)
    assert set(rendered) == {"panel", "summary"}
    for statements in rendered.values():
        assert list(statements) == sorted(statements)
        assert statements
        assert all(isinstance(sql, str) and sql.strip() for sql in statements.values())
