from __future__ import annotations

import pytest

from tests.query.statement_census import JOURNEYS, count_statements

pytestmark = pytest.mark.integration

# First-observed bounds; production changes may only tighten them.
BOUND = {
    "native_scalar": {
        "total": 15,
        "select": 6,
        "create_temp": 1,
        "drop": 1,
        "schema": 6,
        "other": 1,
    },
    "native_breakout": {
        "total": 14,
        "select": 5,
        "create_temp": 1,
        "drop": 1,
        "schema": 6,
        "other": 1,
    },
    "artifact_cuped_breakout": {
        "total": 2794,
        "select": 1321,
        "create_temp": 255,
        "drop": 244,
        "schema": 719,
        "other": 255,
    },
    "dashboard": {
        "total": 82,
        "select": 73,
        "create_temp": 1,
        "drop": 1,
        "schema": 6,
        "other": 1,
    },
}


def test_census_counts_each_execution_path_exactly():
    import ibis
    import pyarrow as pa

    backend = ibis.duckdb.connect()
    ibis.set_backend(backend)
    backend.create_table("census_source", obj=pa.table({"x": [1]}))
    try:
        counts = {}

        def observed(path, action):
            with count_statements(backend) as census:
                action()
            counts[path] = (census.total, census.by_kind.copy())

        observed("raw_sql", lambda: backend.raw_sql("SELECT 1"))
        observed("chained_execute", lambda: backend.raw_sql("SELECT 2").execute("SELECT 3"))
        observed("expression_execute", lambda: ibis.literal(4).execute())
        observed("to_pyarrow", lambda: ibis.literal(5).to_pyarrow())
        observed("get_schema", lambda: backend.get_schema("census_source"))
        observed("backend_sql", lambda: backend.sql("SELECT 6"))
        observed(
            "create_table",
            lambda: backend.create_table("census_created", obj=pa.table({"x": [2]})),
        )
        assert counts == {
            "raw_sql": (
                1,
                {"select": 1, "create_temp": 0, "drop": 0, "schema": 0, "other": 0},
            ),
            "chained_execute": (
                2,
                {"select": 2, "create_temp": 0, "drop": 0, "schema": 0, "other": 0},
            ),
            "expression_execute": (
                1,
                {"select": 1, "create_temp": 0, "drop": 0, "schema": 0, "other": 0},
            ),
            "to_pyarrow": (
                1,
                {"select": 1, "create_temp": 0, "drop": 0, "schema": 0, "other": 0},
            ),
            "get_schema": (
                1,
                {"select": 0, "create_temp": 0, "drop": 0, "schema": 1, "other": 0},
            ),
            "backend_sql": (
                1,
                {"select": 0, "create_temp": 0, "drop": 0, "schema": 1, "other": 0},
            ),
            "create_table": (
                6,
                {"select": 2, "create_temp": 1, "drop": 1, "schema": 1, "other": 1},
            ),
        }
    finally:
        backend.disconnect()


@pytest.mark.parametrize(
    "journey",
    [
        "native_scalar",
        "native_breakout",
        pytest.param("artifact_cuped_breakout", marks=pytest.mark.slow),
        "dashboard",
    ],
    ids=tuple(JOURNEYS),
)
def test_statement_counts_stay_within_bounds(journey, seeded_con):
    census = JOURNEYS[journey](seeded_con)
    assert census.total <= BOUND[journey]["total"]
