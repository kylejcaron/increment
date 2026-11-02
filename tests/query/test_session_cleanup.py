from __future__ import annotations

import ibis
import pytest

from increment import from_definitions, run
from increment.estimation.engine import Method
from tests.analysis_factory import _make_event_log_table

pytestmark = pytest.mark.integration


def test_temporary_relations_are_dropped_on_newest_sqlglot(example_definitions):
    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2"
    )
    source = con.table("event_log", database="analytics")
    source_rows = source.to_pyarrow().to_pylist()
    existing_relations = set(con.list_tables(database=("temp", "main")))

    try:
        result = run(experiment, resolver, decision_method=Method(name="unadjusted"))

        assert result.cells
        assert con.table("event_log", database="analytics").to_pyarrow().to_pylist() == source_rows
        assert set(con.list_tables(database=("temp", "main"))) == existing_relations
    finally:
        close = getattr(resolver, "close", None)
        if callable(close):
            close()
        con.disconnect()
