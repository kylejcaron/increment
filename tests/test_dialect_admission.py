"""SQL admission is re-validated against the connection's own dialect when
`dialect:` is left undeclared, so admission (increment/semantics/loader.py)
and execution (increment/query/session.py's source_sql) agree by
construction wherever the connection's dialect is known to sqlglot; a
backend sqlglot has never heard of falls back to today's generic admission
rather than refusing, so nothing that runs today stops running.

Confirmed live: stock sqlglot ships no dialect under the name `datafusion`,
`flink`, `impala`, or `polars`, but importing an ibis SQL backend (e.g.
`ibis.duckdb.connect()`) registers ibis's own `sqlglot.Dialect` subclass for
every backend it compiles SQL for, including those four, so
`resolve_execution_dialect` resolves them directly rather than falling back
to `None`. `singlestoredb` is the one installed backend name that stays
unregistered under its own name either way (ibis registers it as
`singlestore`), which is why `_IBIS_TO_SQLGLOT_DIALECT` still maps it."""

from __future__ import annotations

import ibis
import pytest

from increment.semantics.loader import (
    resolve_execution_dialect,
    verify_sql_admission_matches_execution,
)
from increment.semantics.models import Definitions, Fact, FactSource


def _defs_with_fact_source(*, dialect: str | None, sql: str) -> Definitions:
    return Definitions(
        dialect=dialect,
        fact_sources=(
            FactSource(
                name="events",
                sql=sql,
                timestamp_column="ts",
                entities=("user_id",),
                facts=(Fact(name="occurred", column=None),),
            ),
        ),
    )


def test_resolve_execution_dialect_returns_the_declared_dialect_unchanged():
    con = ibis.duckdb.connect()
    assert resolve_execution_dialect(Definitions(dialect="snowflake"), con) == "snowflake"


@pytest.mark.parametrize(
    "backend_name,expected",
    [("mssql", "tsql"), ("pyspark", "spark"), ("singlestoredb", "singlestore")],
)
def test_resolve_execution_dialect_maps_the_confirmed_ibis_sqlglot_name_mismatches(
    backend_name, expected
):
    con = ibis.duckdb.connect()
    con.name = backend_name  # confirmed live: assignable on the DuckDB backend instance
    assert resolve_execution_dialect(Definitions(dialect=None), con) == expected


@pytest.mark.parametrize("backend_name", ["datafusion", "flink", "impala", "polars"])
def test_resolve_execution_dialect_resolves_ibis_registered_dialects_directly(
    backend_name,
):
    """Stock sqlglot ships no dialect under these four names, but ibis
    registers its own sqlglot `Dialect` subclass for every SQL backend it
    compiles for as a side effect of importing a connected backend (e.g.
    `ibis.duckdb.connect()`), so these resolve directly by the time
    `resolve_execution_dialect` runs -- confirmed live during
    implementation, contradicting the plan's original "no sqlglot dialect
    at all" premise, which was checked before any ibis backend was
    imported."""
    con = ibis.duckdb.connect()
    con.name = backend_name
    assert resolve_execution_dialect(Definitions(dialect=None), con) == backend_name


def test_resolve_execution_dialect_falls_back_to_none_for_a_truly_unknown_backend():
    """None -- not a refusal -- for a backend name neither stock sqlglot nor
    any imported ibis backend module has ever registered, so
    verify_sql_admission_matches_execution falls back to the same generic
    admission load() already ran."""
    con = ibis.duckdb.connect()
    con.name = "totallymadeupdb"
    assert resolve_execution_dialect(Definitions(dialect=None), con) is None


def test_verify_sql_admission_revalidates_ordinary_sql_against_duckdb_silently():
    con = ibis.duckdb.connect()
    defs = _defs_with_fact_source(dialect=None, sql="SELECT * FROM events")
    verify_sql_admission_matches_execution(defs, con)  # must not raise


def test_verify_sql_admission_falls_back_to_generic_admission_for_an_unknown_backend():
    con = ibis.duckdb.connect()
    con.name = "totallymadeupdb"
    defs = _defs_with_fact_source(dialect=None, sql="SELECT * FROM events")
    verify_sql_admission_matches_execution(defs, con)  # must not raise -- falls back, not refuses


def test_verify_sql_admission_is_a_no_op_when_dialect_is_declared():
    con = ibis.duckdb.connect()
    con.name = "totallymadeupdb"  # would matter if resolve_execution_dialect ran; it never does
    defs = _defs_with_fact_source(dialect="duckdb", sql="SELECT * FROM events")
    verify_sql_admission_matches_execution(defs, con)  # already validated at load(); no-op


def test_analysis_still_constructs_when_the_backend_has_no_known_dialect(tmp_path):
    """End-to-end regression: a backend name with no sqlglot dialect at all
    (neither stock sqlglot nor any imported ibis compiler module) must keep
    constructing exactly as it does today -- Analysis.__init__ calling
    verify_sql_admission_matches_execution must never turn a working
    definitions load into a construction-time refusal."""
    from increment import Analysis

    definitions_path = tmp_path / "definitions"
    definitions_path.mkdir()
    (definitions_path / "defs.yaml").write_text(
        """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: occurred
        column: null
exposures:
  - name: assignment
    fact: occurred
metrics: []
experiments:
  - name: exp
    exposure: assignment
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan:
      primary: []
"""
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=ibis.memtable({"user_id": ["u1"], "ts": ["2025-01-01"]}))
    con.name = "totallymadeupdb"
    analysis = Analysis.from_definitions("exp", definitions_path, con)
    assert analysis is not None
