from __future__ import annotations

import pytest

from increment import from_definitions
from increment.errors import CodedError
from increment.query import panel_sql, summary_sql

pytestmark = pytest.mark.integration

TRANSLATES = ("duckdb", "postgres", "snowflake", "bigquery", "clickhouse", "trino", "databricks")
REFUSES = ("mysql", "sqlite", "mssql", "oracle", "datafusion")


@pytest.mark.parametrize("dialect", [*TRANSLATES, *REFUSES])
def test_analysis_sql_translates_or_refuses_per_ibis_dialect(con, example_definitions, dialect):
    """The sets are facts about the installed Ibis compilers for the panel and
    summary SQL, not a list Increment maintains. After an Ibis upgrade, run this
    test: a dialect that fails has moved, and belongs in the other set.

    Translating is not running: ``run()`` also needs Ibis ``hash()`` and
    warehouse TEMP tables (see the README), so a dialect in ``TRANSLATES`` is
    not a promise that ``run()`` completes on it.
    """
    experiment, resolver = from_definitions(
        example_definitions,
        con,
        experiment_name="new_onboarding_v2",
        backend=dialect,
        store="none",
    )
    with resolver:
        if dialect in TRANSLATES:
            panel = panel_sql(experiment, resolver)
            summary = summary_sql(experiment, resolver)
            expected = {"purchase_rate", "avg_session_duration", "d7_retention"}
            assert set(panel) == expected
            assert set(summary) == expected
            assert all(sql.strip() for sql in [*panel.values(), *summary.values()])
        else:
            for preview in (panel_sql, summary_sql):
                with pytest.raises(CodedError) as refused:
                    preview(experiment, resolver)
                assert refused.value.code == "query.backend.operation_unsupported"
                assert refused.value.context["backend"] == dialect
