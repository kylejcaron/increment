from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import ibis
import pytest

from increment import from_definitions
from increment.errors import CodedError
from increment.query import panel_sql, summary_sql
from increment.query.source import SqlPanelSource
from increment.semantics.design import Randomized
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import _make_event_log_table
from tests.source_conformance import unit_summary_table

pytestmark = pytest.mark.integration

VERIFIED = ("duckdb", "postgres", "snowflake", "bigquery")
TRANSLATES = (*VERIFIED, "clickhouse", "trino", "databricks")
REFUSES = ("mysql", "singlestoredb", "sqlite", "mssql", "oracle", "datafusion")


@backend_fixture(scope="module")
def aware_con():
    """The example event log with time-zone-aware event times."""
    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    con.raw_sql(
        "ALTER TABLE analytics.event_log ALTER event_at TYPE TIMESTAMPTZ "
        "USING timezone('UTC', event_at)"
    )
    yield con
    con.disconnect()


def _assert_previews(con, definitions, dialect, *, translates):
    # Checked without importing the compiler, so the preview has to load it itself.
    if importlib.util.find_spec(f"ibis.backends.sql.compilers.{dialect}") is None:
        pytest.skip(f"the installed Ibis has no {dialect} compiler")
    experiment, resolver = from_definitions(
        definitions,
        con,
        experiment_name="new_onboarding_v2",
        backend=dialect,
        store="none",
    )
    with resolver:
        if translates:
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
                detail = str(refused.value.context["detail"])
                assert re.search(r"(?<![\w.])_[A-Za-z]", detail) is None


@pytest.mark.grid("each Ibis SQL dialect's compiler is a separate translation fact")
@pytest.mark.parametrize("dialect", [*TRANSLATES, *REFUSES])
def test_analysis_sql_translates_or_refuses_per_ibis_dialect(con, example_definitions, dialect):
    """The sets are facts about the installed Ibis compilers for the panel and
    summary SQL, not a list Increment maintains. After an Ibis upgrade, run this
    test: a dialect that fails has moved, and belongs in the other set.

    Translating is not running: ``run()`` also needs Ibis ``hash()`` and
    warehouse TEMP tables (see the README), so a dialect in ``TRANSLATES`` is
    not a promise that ``run()`` completes on it.
    """
    _assert_previews(con, example_definitions, dialect, translates=dialect in TRANSLATES)


@pytest.mark.parametrize("dialect", TRANSLATES)
def test_time_zone_aware_timestamps_translate_only_where_increment_converts_them(
    aware_con, example_definitions, dialect
):
    """Time-zone-aware event times are read as UTC through a conversion
    Increment registers for the verified dialects only, so the other dialects
    that translate the analysis SQL refuse it for such data."""
    _assert_previews(aware_con, example_definitions, dialect, translates=dialect in VERIFIED)


_SINGLESTOREDB_PREVIEWS = """
import json
import sys

import ibis

from increment import from_definitions
from increment.errors import CodedError
from increment.query import panel_sql, summary_sql
from tests.analysis_factory import _make_event_log_table

con = ibis.duckdb.connect()
_make_event_log_table(con)
experiment, resolver = from_definitions(
    sys.argv[1], con, experiment_name="new_onboarding_v2", backend="singlestoredb", store="none"
)
assert "ibis.backends.sql.compilers.singlestoredb" not in sys.modules
outcomes = []
with resolver:
    for preview in (panel_sql, summary_sql):
        try:
            preview(experiment, resolver)
        except CodedError as exc:
            outcomes.append([exc.code, exc.context["backend"]])
        except Exception as exc:
            outcomes.append([type(exc).__name__, str(exc)])
        else:
            outcomes.append(["compiled", None])
print(json.dumps(outcomes))
"""


def test_singlestoredb_preview_compiles_in_a_fresh_process(example_definitions):
    """A SQL preview for a dialect loads that dialect's Ibis compiler, as
    connecting to the backend would, in a process that never connected to it."""
    if importlib.util.find_spec("ibis.backends.sql.compilers.singlestoredb") is None:
        expected = "query.backend.compiler_unavailable"
    else:
        expected = "query.backend.operation_unsupported"
    child = subprocess.run(
        [sys.executable, "-c", _SINGLESTOREDB_PREVIEWS, str(example_definitions)],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout.splitlines()[-1]) == [[expected, "singlestoredb"]] * 2


def test_preview_for_a_dialect_without_a_compiler_refuses(con, example_definitions):
    """An invented name, a module beside Ibis's compilers and a dotted name each
    name no compiler."""
    for dialect in ("notadialect", "base", "notadialect.extra"):
        experiment, resolver = from_definitions(
            example_definitions,
            con,
            experiment_name="new_onboarding_v2",
            backend=dialect,
            store="none",
        )
        with resolver:
            for preview in (panel_sql, summary_sql):
                with pytest.raises(CodedError) as refused:
                    preview(experiment, resolver)
                assert refused.value.code == "query.backend.compiler_unavailable"
                assert refused.value.context["backend"] == dialect


def test_sql_computed_summary_preview_refuses_a_dialect_without_a_translation(monkeypatch):
    """A summary computed in SQL from a loaded dataframe compiles in Ibis's
    default backend; SQLite has no translation of the NaN test it uses."""
    con = ibis.sqlite.connect()
    monkeypatch.setattr(ibis.options, "default_backend", con)
    try:
        source = SqlPanelSource.from_frame_via_memtable(
            con,
            unit_summary_table(),
            unit="user_id",
            group="variant",
            metrics={"revenue": "mean"},
            design=Randomized(control_group="control"),
        )
        with pytest.raises(CodedError) as refused:
            source.summary_sql()
    finally:
        con.disconnect()
    assert refused.value.code == "query.backend.operation_unsupported"
    assert refused.value.context["backend"] == "sqlite"
