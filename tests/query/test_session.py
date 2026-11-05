"""WarehouseSession: fact-table memoization and temp-table mechanics."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
import time
import warnings
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import ibis
import ibis.expr.types as ir
import pytest

from increment.errors import CapabilityError, IncrementWarning, InvalidRequestError
from increment.query.artifact_contract import ARTIFACT_DOMAIN_ROOT, ArtifactContractError
from increment.query.artifact_digest import manifest_sha256
from increment.query.session import (
    SourceScope,
    SourceSnapshotEvidence,
    WarehouseArtifactStore,
    WarehouseSession,
)
from increment.semantics.artifact import (
    ArtifactContext,
    BaseRelations,
    MeasureManifest,
    UnitDayArtifactManifest,
)
from tests.warning_codes import warning_codes

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_query_boundary_filters_only_ibis_duckdb_arrow_deprecation(tmp_path):
    import shutil

    definitions = tmp_path / "definitions"
    shutil.copytree("examples/definitions", definitions)
    script = textwrap.dedent(
        """
        import sys

        import ibis
        from increment import from_definitions
        from increment.workflows import health_check
        from tests.analysis_factory import _make_event_log_table, view_readout

        con = ibis.duckdb.connect()
        _make_event_log_table(con)
        experiment, resolver = from_definitions(
            sys.argv[1], con, experiment_name="new_onboarding_v2"
        )
        view_readout(experiment, resolver, metrics=["purchase_rate"])
        health_check(experiment, resolver)
        """
    )
    analysis = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", script, str(definitions)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert analysis.returncode == 0, analysis.stderr
    assert "fetch_arrow_table" not in analysis.stderr

    other_deprecation = subprocess.run(
        [
            sys.executable,
            "-W",
            "error::DeprecationWarning",
            "-c",
            "import warnings, increment.query; "
            'warnings.warn("probe is deprecated", DeprecationWarning)',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert other_deprecation.returncode != 0
    assert "DeprecationWarning: probe is deprecated" in other_deprecation.stderr


def _session():
    con = ibis.duckdb.connect()
    con.create_table("t", ibis.memtable({"x": [1, 2, 3]}))
    # This minimal dialect supports source_sql; fact_table tests use real Definitions.
    return con, WarehouseSession(
        con,
        defs=SimpleNamespace(dialect="duckdb"),  # type: ignore[ty:invalid-argument-type]
    )


def _postgres_sql_backend(
    *,
    description=None,
    type_rows=((20, "bigint"),),
    sql_result="live relation",
    probe_error=None,
):
    class Cursor:
        def __init__(self, *, description=None, rows=()):
            self.description = description
            self._rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def fetchall(self):
            return self._rows

    class TypeMapper:
        def from_string(self, name, *, nullable):
            assert nullable
            name = {
                "bigint": "int64",
                "timestamp without time zone": "timestamp",
            }.get(name, name)
            if name.startswith("numeric("):
                name = name.replace("numeric(", "decimal(", 1)
            if name.endswith("[]"):
                name = f"array<{name[:-2]}>"
            if name.startswith("character("):
                return ibis.dtype("string")
            return ibis.dtype(name)

    class Postgres:
        name = "postgres"
        compiler = SimpleNamespace(type_mapper=TypeMapper())

        def __init__(self):
            self.probe = None
            self.schema = None

        def _transpile_sql(self, query, *, dialect):
            self.input_dialect = dialect
            return query

        def raw_sql(self, statement):
            if "LIMIT 0" in statement:
                self.probe = statement
                if probe_error is not None:
                    raise probe_error
                columns = description or (SimpleNamespace(name="value", type_code=20),)
                return Cursor(description=columns)
            assert "pg_catalog.pg_type" in statement
            return Cursor(rows=type_rows)

        def sql(self, statement, *, dialect, schema):
            self.statement = statement
            self.dialect = dialect
            self.schema = schema
            return sql_result

    return Postgres()


def _postgres_catalog_backend(row, expected, relation):
    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def fetchone(self):
            return row

    class Postgres:
        name = "postgres"

        def raw_sql(self, statement):
            assert expected in statement
            return Cursor()

        def table(self, name, *, database):
            self.table_name = name
            self.database = database
            return relation

    return Postgres()


@pytest.mark.parametrize("sql", ["SELECT 1", "SELECT 1 -- source", "SELECT 1; -- source"])
def test_live_postgres_source_sql_does_not_require_temp_privilege(sql):
    backend = _postgres_sql_backend()
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql(sql) == "live relation"
    assert "LIMIT 0" in backend.probe
    assert "-- source" not in backend.probe
    assert backend.schema["value"] == ibis.dtype("int64")


def test_snapshot_schema_declaration_refuses_before_temp_capability_check():
    from increment.errors import DefinitionError
    from increment.semantics.models import FactSource

    backend = _postgres_sql_backend(
        description=(
            SimpleNamespace(name="user_id", type_code=20),
            SimpleNamespace(name="event_at", type_code=1114),
        ),
        type_rows=((20, "bigint"), (1114, "timestamp without time zone")),
        sql_result=SimpleNamespace(columns=("user_id", "event_at")),
    )
    source = FactSource.model_construct(
        name="purchase_rows",
        sql="SELECT user_id, event_at FROM analytics.event_log",
        timestamp_column="event_at",
        entities=("user_id",),
        facts=(),
    )
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    with pytest.raises(DefinitionError) as error:
        session.preflight_fact_source_schema(source, require_event=True)

    assert error.value.code == "definition.fact_source.missing_columns"
    assert error.value.context["missing"] == ("event",)


def test_live_postgres_plain_table_source_resolves_through_catalog_without_temp_introspection():
    relation = object()
    backend = _postgres_catalog_backend(
        ("public", "events"), 'to_regclass(\'"public"."events"\')', relation
    )
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql('SELECT * FROM "public"."events"') is relation
    assert backend.table_name == "events"
    assert backend.database == ("public",)


def test_live_postgres_unqualified_table_uses_search_path_and_folds_identifiers():
    relation = object()
    backend = _postgres_catalog_backend(("public", "events"), "to_regclass('EVENTS')", relation)
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql("SELECT * FROM EVENTS") is relation
    assert backend.table_name == "events"
    assert backend.database == ("public",)


def test_live_postgres_catalog_preserves_non_ascii_unquoted_identifier():
    relation = object()
    backend = _postgres_catalog_backend(("public", "Évents"), "to_regclass('Évents')", relation)
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql("SELECT * FROM Évents") is relation
    assert backend.table_name == "Évents"
    assert backend.database == ("public",)


def test_live_postgres_plain_table_catalog_lookup_ignores_identifier_comments():
    relation = object()
    backend = _postgres_catalog_backend(("public", "events"), "to_regclass('events')", relation)
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql("SELECT * FROM events /* source */") is relation
    assert backend.table_name == "events"


def test_live_postgres_table_lookup_preserves_quoted_namespaces():
    relation = object()
    backend = _postgres_catalog_backend(
        ("schema.with.dot", "Mixed Events"),
        'to_regclass(\'"schema.with.dot"."Mixed Events"\')',
        relation,
    )
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql('SELECT * FROM "schema.with.dot"."Mixed Events"') is relation
    assert backend.table_name == "Mixed Events"
    assert backend.database == ("schema.with.dot",)


def test_live_postgres_pg_temp_alias_resolves_physical_schema():
    relation = object()
    backend = _postgres_catalog_backend(
        ("pg_temp_42", "events"), "to_regclass('pg_temp.events')", relation
    )
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql("SELECT * FROM pg_temp.events") is relation
    assert backend.table_name == "events"
    assert backend.database == ("pg_temp_42",)


@pytest.mark.parametrize(
    ("sql", "sql_dialect"),
    [
        ("SELECT * FROM ONLY events", "postgres"),
        ("SELECT * FROM events TABLESAMPLE SYSTEM (0)", "postgres"),
        ("SELECT * FROM events AS source", "postgres"),
        ("SELECT * REPLACE (0 AS value) FROM events", "duckdb"),
        ("SELECT * EXCLUDE (value) FROM events", "duckdb"),
        ("SELECT * RENAME (value AS amount) FROM events", "duckdb"),
    ],
)
def test_live_postgres_table_lookup_preserves_table_modifiers(sql, sql_dialect):
    relation = object()
    backend = _postgres_sql_backend(sql_result=relation)

    def unexpected_table(*_args, **_kwargs):
        pytest.fail("modified relation SQL must not be simplified to a table")

    backend.table = unexpected_table
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect=sql_dialect),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql(sql) is relation
    assert backend.statement == sql
    assert backend.dialect == sql_dialect


def test_captured_postgres_sources_require_temp_privilege():
    from increment.errors import CapabilityError

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def fetchone(self):
            return (False,)

    class Postgres:
        name = "postgres"

        def raw_sql(self, statement):
            assert "has_database_privilege" in statement
            return Cursor()

        def sql(self, *_args, **_kwargs):
            pytest.fail("capture must refuse before reading without TEMP privilege")

    session = WarehouseSession(
        Postgres(),  # type: ignore[ty:invalid-argument-type]
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    with pytest.raises(CapabilityError) as error:
        session.pin_sources(("SELECT 1",))

    assert error.value.code == "query.session.snapshot.materialization_failed"


def test_live_sql_privilege_error_is_not_mapped_to_snapshot_refusal():
    class PrivilegeError(Exception):
        sqlstate = "42501"

    backend = _postgres_sql_backend(probe_error=PrivilegeError("permission denied"))
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    with pytest.raises(PrivilegeError):
        session.source_sql("SELECT user_id FROM analytics.event_log")
    assert "has_database_privilege" not in backend.probe


def test_postgres_sql_schema_preserves_numeric_precision():
    backend = _postgres_sql_backend(
        description=(
            SimpleNamespace(name="unit_id", type_code=1700, precision=38, scale=0),
            SimpleNamespace(
                name="unit_ids",
                type_code=1231,
                precision=38,
                scale=0,
                type_display="numeric(38,0)[]",
            ),
            SimpleNamespace(name="event", type_code=1042, type_display="bpchar(16)"),
        ),
        type_rows=((1231, "numeric[]"), (1700, "numeric"), (1042, "character(16)")),
    )
    session = WarehouseSession(
        backend,
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    session.source_sql("SELECT unit_id FROM events")

    assert backend.schema["unit_id"] == ibis.dtype("decimal(38, 0)")
    assert backend.schema["unit_ids"] == ibis.dtype("array<decimal(38, 0)>")
    assert backend.schema["event"] == ibis.dtype("string")


def _readout_source(snapshots):
    from contextlib import contextmanager

    class ReadoutSource:
        def __init__(self):
            self.snapshots = iter(snapshots)

        @contextmanager
        def readout_snapshot(self, **_kwargs):
            snapshot = next(self.snapshots)
            try:
                yield snapshot
            finally:
                snapshot.drop_materialized()

    return ReadoutSource()


@pytest.mark.integration
def test_snapshot_arrow_cache_reuses_only_captured_relation_results(monkeypatch):
    con, session = _session()
    sql = "select x from t"
    snapshot = session.pin_sources((sql,))
    expression = snapshot.source_sql(sql).filter(lambda row: row.x > 1)
    original_to_pyarrow = con.to_pyarrow
    executions = 0

    def counted_to_pyarrow(expr, *args, **kwargs):
        nonlocal executions
        executions += 1
        return original_to_pyarrow(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", counted_to_pyarrow)
    try:
        first = snapshot.fetch_arrow(expression)
        second = snapshot.fetch_arrow(expression)
        assert first.equals(second)
        assert first.to_pylist() == [{"x": 2}, {"x": 3}]
        assert executions == 1
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_snapshot_arrow_cache_does_not_cache_mutable_live_relations():
    con, session = _session()
    snapshot = session.pin_sources(("select x from t",))
    live = con.table("t")
    try:
        assert snapshot.fetch_arrow(live).to_pylist() == [{"x": 1}, {"x": 2}, {"x": 3}]
        con.insert("t", ibis.memtable({"x": [4]}))
        assert snapshot.fetch_arrow(live).to_pylist() == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
            {"x": 4},
        ]
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_snapshot_arrow_cache_skips_raw_sql_views_over_captured_relations():
    con, session = _session()
    sql = "select x from t"
    snapshot = session.pin_sources((sql,))
    captured = snapshot.source_sql(sql)
    opaque_query = captured.alias("captured").sql("select x from t")
    try:
        assert snapshot.fetch_arrow(opaque_query).to_pylist() == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]
        con.insert("t", ibis.memtable({"x": [4]}))
        assert snapshot.fetch_arrow(opaque_query).to_pylist() == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
            {"x": 4},
        ]
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.slow
def test_snapshot_arrow_cache_skips_results_over_per_result_budget(monkeypatch):
    con, session = _session()
    snapshot = session.pin_sources(("select x from t",))
    captured = snapshot.source_sql("select x from t")
    expression = captured.filter(captured.x == 1).select(
        payload=cast("ir.StringScalar", ibis.literal("a")).repeat(16 * 1024 * 1024)
    )
    original_to_pyarrow = con.to_pyarrow
    executions = 0

    def counted_to_pyarrow(expr, *args, **kwargs):
        nonlocal executions
        executions += 1
        return original_to_pyarrow(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", counted_to_pyarrow)
    try:
        first = snapshot.fetch_arrow(expression)
        second = snapshot.fetch_arrow(expression)
        assert first.nbytes > 16 * 1024 * 1024
        assert first.column("payload")[0].as_py() == "a" * (16 * 1024 * 1024)
        assert second.equals(first)
        assert executions == 2
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.slow
def test_snapshot_arrow_cache_stops_at_total_byte_budget(monkeypatch):
    con, session = _session()
    snapshot = session.pin_sources(("select x from t",))
    captured = snapshot.source_sql("select x from t")
    chunk_bytes = 16 * 1024 * 1024
    total_bytes = 128 * 1024 * 1024
    original_to_pyarrow = con.to_pyarrow

    def payload_expression(index, length):
        return captured.filter(captured.x == 1).select(
            payload=cast("ir.StringScalar", ibis.literal(str(index))).repeat(length)
        )

    overhead = original_to_pyarrow(payload_expression(0, 1)).nbytes - 1
    payload_chars = chunk_bytes - overhead
    expressions = [payload_expression(index, payload_chars) for index in range(9)]
    executions = 0

    def counted_to_pyarrow(expr, *args, **kwargs):
        nonlocal executions
        executions += 1
        return original_to_pyarrow(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", counted_to_pyarrow)
    try:
        admitted_bytes = 0
        for index, expression in enumerate(expressions[:8]):
            result = snapshot.fetch_arrow(expression)
            assert result.nbytes == chunk_bytes
            admitted_bytes += result.nbytes
            assert result.column("payload")[0].as_py() == str(index) * payload_chars
            assert snapshot.fetch_arrow(expression).equals(result)
        assert admitted_bytes == total_bytes
        assert executions == 8
        for index, expression in enumerate(expressions[:8]):
            assert snapshot.fetch_arrow(expression).column("payload")[0].as_py() == (
                str(index) * payload_chars
            )
        assert executions == 8
        uncached = snapshot.fetch_arrow(expressions[8])
        assert uncached.nbytes == chunk_bytes
        assert snapshot.fetch_arrow(expressions[8]).equals(uncached)
        assert executions == 10
        for index, expression in enumerate(expressions[:8]):
            assert snapshot.fetch_arrow(expression).column("payload")[0].as_py() == (
                str(index) * payload_chars
            )
        assert executions == 10
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_overlapping_non_lifo_readouts_leave_no_connection_cache():
    con, session = _session()
    first_snapshot = session.pin_sources(("select x from t",))
    second_snapshot = session.pin_sources(("select x from t",))
    source = _readout_source((first_snapshot, second_snapshot))
    first_readout = source.readout_snapshot(metrics=(), population="assigned")
    first = first_readout.__enter__()
    second_readout = source.readout_snapshot(metrics=(), population="assigned")
    second = second_readout.__enter__()
    first_expression = first.source_sql("select x from t")
    second_expression = second.source_sql("select x from t")
    try:
        assert first.fetch_arrow(first_expression).to_pylist() == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]
        assert second.fetch_arrow(second_expression).to_pylist() == [
            {"x": 1},
            {"x": 2},
            {"x": 3},
        ]
        first_readout.__exit__(None, None, None)
        con.insert("t", ibis.memtable({"x": [4]}))
        assert second.fetch_arrow(con.table("t")).to_pylist()[-1] == {"x": 4}
        second_readout.__exit__(None, None, None)
        con.insert("t", ibis.memtable({"x": [5]}))
        assert session.fetch_arrow(con.table("t")).to_pylist()[-1] == {"x": 5}
    finally:
        first_snapshot.drop_materialized()
        second_snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_dropped_generation_refusal_stays_visible_during_pinned_snapshot():
    con = ibis.duckdb.connect()
    con.create_table("t", ibis.memtable({"x": [1]}))
    store = WarehouseArtifactStore(con)
    ref, _measure = _publish_in(store, unit_id="tombstone-during-snapshot")
    session = WarehouseSession(
        con,
        defs=SimpleNamespace(dialect="duckdb"),  # type: ignore[ty:invalid-argument-type]
    )
    snapshot = session.pin_sources(("select x from t",))
    dropped = con.table("ud_manifest_dropped").filter(
        lambda row: (
            (row.artifact_id == str(ref.artifact_id))
            & (row.generation_id == str(ref.generation_id))
        )
    )
    source = _readout_source((snapshot,))
    try:
        with source.readout_snapshot(metrics=(), population="assigned") as readout:
            assert readout.fetch_arrow(dropped).to_pylist() == []
            store.abandon_generation(ref.artifact_id, ref.generation_id)
            with pytest.raises(ArtifactContractError) as refusal:
                with store.open_snapshot(ref):
                    pass
            assert refusal.value.code == "artifact.generation.dropped"
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_snapshot_exception_leaves_no_arrow_cache_state():
    con, session = _session()
    snapshot = session.pin_sources(("select x from t",))
    source = _readout_source((snapshot,))
    expression = snapshot.source_sql("select x from t")
    try:
        with pytest.raises(RuntimeError, match="readout failed"):
            with source.readout_snapshot(metrics=(), population="assigned") as readout:
                readout.fetch_arrow(expression)
                raise RuntimeError("readout failed")
        con.insert("t", ibis.memtable({"x": [4]}))
        assert session.fetch_arrow(con.table("t")).to_pylist()[-1] == {"x": 4}
    finally:
        snapshot.drop_materialized()
        con.disconnect()


def test_pin_sources_interprets_naive_boundary_cutoff_without_host_timezone(monkeypatch):
    con = ibis.duckdb.connect()
    try:
        sql = "select ts from events"
        event_ts = datetime(2025, 1, 12, 11, tzinfo=UTC)
        con.create_table("events", ibis.memtable({"ts": [event_ts]}))
        enrolled = con.sql("select 1 as unit_id")
        session = WarehouseSession(con, defs=SimpleNamespace(dialect="duckdb"))  # type: ignore[ty:invalid-argument-type]
        start = datetime(2025, 1, 15)
        n_pre_periods = 1
        # The snapshot margin starts two days before the experiment; 2025-01-13
        # at the UTC+14 boundary is 2025-01-12 10:00 UTC.
        window_start = start - timedelta(days=n_pre_periods + 1) - timedelta(hours=14)
        with monkeypatch.context() as tz:
            tz.setenv("TZ", "Etc/GMT+12")
            time.tzset()
            pinned = session.pin_sources(
                (sql,),
                column_hints={sql: (None, "ts")},
                scope=SourceScope(enrolled_units=enrolled, window_start=window_start),
            )
            assert pinned.source_sql(sql).execute()["ts"].tolist() == [event_ts]
    finally:
        time.tzset()
        con.disconnect()


def test_source_snapshot_evidence_keeps_unknown_watermarks_unknown():
    evidence = SourceSnapshotEvidence(
        datetime(2025, 1, 20, 12, tzinfo=UTC),
        {"trigger": None, "outcomes": datetime(2025, 1, 19, 23, tzinfo=UTC)},
    )
    assert evidence.observation_cutoff_ts == datetime(2025, 1, 20, 12, tzinfo=UTC)
    assert evidence.complete_through_by_feed["trigger"] is None
    assert evidence.complete_through_by_feed["outcomes"] == datetime(2025, 1, 19, 23, tzinfo=UTC)


def test_source_snapshot_evidence_rejects_naive_instants():
    with pytest.raises(InvalidRequestError) as cutoff:
        SourceSnapshotEvidence(datetime(2025, 1, 20, 12))
    assert cutoff.value.code == "query.session.snapshot.cutoff_not_aware"
    assert cutoff.value.context["field"] == "observation_cutoff_ts"

    with pytest.raises(InvalidRequestError) as feed:
        SourceSnapshotEvidence(datetime(2025, 1, 20, 12, tzinfo=UTC), {"": None})
    assert feed.value.code == "query.session.snapshot.feed_name_invalid"
    assert feed.value.context["value"] == ""

    with pytest.raises(InvalidRequestError) as watermark:
        SourceSnapshotEvidence(
            datetime(2025, 1, 20, 12, tzinfo=UTC),
            {"events": datetime(2025, 1, 20, 12)},
        )
    assert watermark.value.code == "query.session.snapshot.watermark_not_aware"
    assert watermark.value.context["feed"] == "events"
    assert watermark.value.context["field"] == "complete_through_by_feed"


def test_session_does_not_fabricate_source_snapshot_evidence():
    _con, session = _session()
    assert session.source_snapshot_evidence is None
    assert session.pin_sources(()).source_snapshot_evidence is None


def test_temp_name_caps_identifier_length():
    _, s = _session()
    name = s.temp_name("a" * 100, "b" * 100)
    assert len(name) <= 63  # portable identifier cap, matching current behavior


def test_temp_names_unique_across_sessions():
    con = ibis.duckdb.connect()
    a = WarehouseSession(con, defs=None)  # type: ignore[ty:invalid-argument-type]
    b = WarehouseSession(con, defs=None)  # type: ignore[ty:invalid-argument-type]
    assert a.temp_name("stats") != b.temp_name("stats")


def test_materialize_table_creates_temp_and_tracks_name():
    con, s = _session()
    out = s.materialize_table(s.temp_name("stats"), con.table("t"))
    assert out.execute()["x"].tolist() == [1, 2, 3]
    assert len(s.materialized_names) == 1
    s.drop_materialized()
    assert s.materialized_names == []


def test_materialized_cleanup_uses_connection_sqlglot_dialect(monkeypatch):
    con, session = _session()
    name = session.temp_name("dialect_cleanup")
    session.materialize_table(name, con.table("t"), required=True)
    assert name in con.list_tables(database=("temp", "main"))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(con, "name", "singlestoredb")
            session.drop_materialized()
        assert name not in con.list_tables(database=("temp", "main"))
        assert session.materialized_names == []
    finally:
        session.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_failed_insert_during_temp_materialization_is_cleaned_up():
    con, session = _session()
    name = session.temp_name("bad_materialization")
    source_rows = con.table("t").to_pyarrow().to_pylist()
    expression = ibis.literal("not-an-int").cast("int64").as_table()
    try:
        with pytest.raises(CapabilityError) as raised:
            session.materialize_table(name, expression, required=True)
        assert raised.value.code == "query.session.snapshot.materialization_failed"
        assert name not in con.list_tables(database=("temp", "main"))
        assert con.table("t").to_pyarrow().to_pylist() == source_rows
        assert session.materialized_names == []
    finally:
        session.drop_materialized()
        con.disconnect()


@pytest.mark.parametrize("owner_kind", ["session", "artifact_qr"])
def test_temp_catalog_probe_failure_retains_locator_for_retry(monkeypatch, owner_kind):
    from contextlib import nullcontext

    con, session = _session()
    store = WarehouseArtifactStore(con)
    expression = ibis.literal("not-an-int").cast("int64").as_table()
    create = con.create_table
    list_tables = con.list_tables
    created_names = []
    failed_probe = False

    def failed_create(name, *args, **kwargs):
        try:
            return create(name, *args, **kwargs)
        except Exception:
            created_names.append(name)
            raise

    def unavailable_catalog(*args, **kwargs):
        nonlocal failed_probe
        if created_names and not failed_probe and kwargs.get("database") == ("temp", "main"):
            failed_probe = True
            raise RuntimeError("temporary catalog unavailable")
        return list_tables(*args, **kwargs)

    monkeypatch.setattr(con, "create_table", failed_create)
    monkeypatch.setattr(con, "list_tables", unavailable_catalog)
    scope = store.qr_scratch() if owner_kind == "artifact_qr" else nullcontext(session)
    try:
        with scope as owner:
            with pytest.raises(CapabilityError) as raised:
                if owner_kind == "artifact_qr":
                    owner.materialize_input(expression)
                else:
                    owner.materialize_table(
                        session.temp_name("probe_retry"), expression, required=True
                    )
            assert raised.value.code == "query.session.snapshot.materialization_failed"
            assert failed_probe
            assert len(created_names) == 1
            name = created_names[0]
            assert name in con.list_tables(database=("temp", "main"))
            if owner_kind == "artifact_qr":
                assert owner.leftovers == (("temp", "main", name),)
                owner.close()
                assert owner.leftovers == ()
            else:
                assert owner.materialized_names == [name]
                owner.drop_materialized()
                assert owner.materialized_names == []
            assert name not in con.list_tables(database=("temp", "main"))
    finally:
        session.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_existing_temp_name_is_not_dropped_after_materialization_refusal():
    con, session = _session()
    name = session.temp_name("reserved_temp")
    con.raw_sql(f'CREATE TEMP TABLE "{name}" (x INTEGER)')
    con.raw_sql(f'INSERT INTO "{name}" VALUES (99)')
    source_rows = con.table("t").to_pyarrow().to_pylist()
    expression = ibis.literal("not-an-int").cast("int64").as_table()
    try:
        with pytest.raises(CapabilityError) as raised:
            session.materialize_table(name, expression, required=True)
        assert raised.value.code == "query.session.snapshot.materialization_failed"
        assert con.table(name).to_pyarrow().to_pylist() == [{"x": 99}]
        assert con.table("t").to_pyarrow().to_pylist() == source_rows
        assert session.materialized_names == []
    finally:
        session.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_drop_materialized_retries_after_storage_boundary_failure(monkeypatch):
    con, session = _session()
    name = session.temp_name("retry_cleanup")
    source_rows = con.table("t").to_pyarrow().to_pylist()
    session.materialize_table(name, con.table("t"), required=True)
    raw_sql = con.raw_sql
    failed = False

    def fail_first_drop(statement, *args, **kwargs):
        nonlocal failed
        if not failed and str(statement).startswith("DROP TABLE IF EXISTS") and name in statement:
            failed = True
            raise RuntimeError("transient storage cleanup failure")
        return raw_sql(statement, *args, **kwargs)

    monkeypatch.setattr(con, "raw_sql", fail_first_drop)
    try:
        session.drop_materialized()
        assert failed
        assert name in con.list_tables(database=("temp", "main"))
        assert session.materialized_names == [name]
        session.drop_materialized()
        assert name not in con.list_tables(database=("temp", "main"))
        assert con.table("t").to_pyarrow().to_pylist() == source_rows
    finally:
        session.drop_materialized()
        con.disconnect()


class _ScopedPostgresConnection:
    def __init__(self, *, fail_drop: bool = False):
        self.tables = {"owned"}
        self.committed_tables = set(self.tables)
        self.unrelated_changes = 0
        self.committed_unrelated_changes = 0
        self.in_transaction = False
        self.fail_drop = fail_drop

    def cursor(self):
        return _ScopedPostgresCursor(self)

    @contextmanager
    def transaction(self):
        outermost = not self.in_transaction
        previous = (set(self.tables), self.unrelated_changes)
        if outermost:
            self.in_transaction = True
        try:
            yield
        except BaseException:
            self.tables, self.unrelated_changes = previous
            if outermost:
                self.in_transaction = False
            raise
        else:
            if outermost:
                self.committed_tables = set(self.tables)
                self.committed_unrelated_changes = self.unrelated_changes
                self.in_transaction = False

    def rollback(self):
        if self.in_transaction:
            self.tables = set(self.committed_tables)
            self.unrelated_changes = self.committed_unrelated_changes
            self.in_transaction = False


class _ScopedPostgresCursor:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return None

    def execute(self, statement):
        if not self.connection.in_transaction:
            self.connection.in_transaction = True
        if statement.startswith("DROP TABLE"):
            if self.connection.fail_drop:
                raise RuntimeError("injected PostgreSQL DROP failure")
            self.connection.tables.discard("owned")
        elif statement == "INSERT unrelated":
            self.connection.unrelated_changes += 1

    def close(self):
        pass


class _ScopedPostgresBackend:
    name = "postgres"
    dialect = "postgres"

    def __init__(self, connection):
        self.con = connection

    @contextmanager
    def _safe_raw_sql(self, statement):
        with self.con.cursor() as cursor, self.con.transaction():
            yield cursor.execute(statement)

    def raw_sql(self, statement):
        cursor = self.con.cursor()
        try:
            cursor.execute(statement)
        except Exception:
            cursor.close()
            self.con.rollback()
            raise
        return cursor


@pytest.mark.integration
@pytest.mark.parametrize("fail_drop", [False, True])
def test_postgres_owned_drop_uses_scoped_transaction(fail_drop):
    from increment.query import session as module

    connection = _ScopedPostgresConnection(fail_drop=fail_drop)
    owner = WarehouseSession(cast(Any, _ScopedPostgresBackend(connection)), defs=cast(Any, None))
    owner._materialized.append(module._MaterializedRelation("owned", "owned", None, "pg_temp"))

    if fail_drop:
        with connection.transaction():
            cursor = connection.cursor()
            cursor.execute("INSERT unrelated")
            cursor.close()
            owner.drop_materialized()
            assert connection.unrelated_changes == 1
            assert owner.materialized_names == ["owned"]

        assert connection.committed_unrelated_changes == 1
        assert "owned" in connection.committed_tables
        connection.fail_drop = False
        owner.drop_materialized()
        assert "owned" not in connection.tables
        return

    owner.drop_materialized()

    assert owner.materialized_names == []
    assert not connection.in_transaction
    assert "owned" not in connection.committed_tables
    connection.rollback()
    assert "owned" not in connection.tables


@pytest.mark.integration
def test_snowflake_batch_owned_drop_closes_cursors():
    from contextlib import closing, contextmanager

    from increment.query import session as module

    class Cursor:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Backend:
        name = "snowflake"
        dialect = "snowflake"

        def __init__(self):
            self.tables = {"owned_a", "owned_b"}
            self.cursors = []

        def raw_sql(self, statement):
            for name in tuple(self.tables):
                if f'"{name}"' in statement:
                    self.tables.remove(name)
            cursor = Cursor()
            self.cursors.append(cursor)
            return cursor

        @contextmanager
        def _safe_raw_sql(self, statement):
            with closing(self.raw_sql(statement)) as cursor:
                yield cursor

    backend = Backend()
    relations = tuple(
        module._MaterializedRelation(name, name, "catalog", "schema")
        for name in ("owned_a", "owned_b")
    )

    module._drop_materialized_relations(cast(Any, backend), relations)

    assert backend.tables == set()
    assert len(backend.cursors) == 2
    assert all(cursor.closed for cursor in backend.cursors)


@pytest.mark.integration
def test_duckdb_batch_owned_drop_keeps_borrowed_connection_open():
    from increment.query import session as module

    con = ibis.duckdb.connect()
    names = ("owned_batch_a", "owned_batch_b")
    try:
        for name in names:
            con.raw_sql(f'CREATE TEMP TABLE "{name}" (x INTEGER)')
        relations = tuple(
            module._MaterializedRelation(name, name, "temp", "main") for name in names
        )

        module._drop_materialized_relations(con, relations)

        assert set(names).isdisjoint(con.list_tables(database=("temp", "main")))
        assert con.to_pyarrow(con.sql("SELECT 42 AS value")).to_pylist() == [{"value": 42}]
    finally:
        con.disconnect()


@pytest.mark.parametrize(
    ("physical_name", "remove_before_cleanup"),
    [("owned", False), ("renamed", False), ("owned", True)],
)
def test_cleanup_preserves_other_namespace_data(monkeypatch, physical_name, remove_before_cleanup):
    con, session = _session()
    con.create_database("materialized")
    sentinel = [{"sentinel": 314159}]
    con.create_table(physical_name, ibis.memtable(sentinel))
    create_table = con.create_table

    def create_in_physical_namespace(name, expr, *, temp):
        return create_table(physical_name, expr, database="materialized")

    monkeypatch.setattr(con, "create_table", create_in_physical_namespace)
    try:
        relation = session.materialize_table("owned", con.table("t"))
        assert relation.to_pyarrow().to_pylist() == [{"x": 1}, {"x": 2}, {"x": 3}]
        if remove_before_cleanup:
            import sqlglot

            schema_name = sqlglot.exp.to_identifier("materialized", quoted=True).sql("duckdb")
            table_name = sqlglot.exp.to_identifier(physical_name, quoted=True).sql("duckdb")
            con.raw_sql(f"DROP TABLE IF EXISTS {schema_name}.{table_name}")
        session.drop_materialized()
        assert physical_name not in con.list_tables(database="materialized")
        assert con.table(physical_name, database="main").to_pyarrow().to_pylist() == sentinel
        session.drop_materialized()
        assert con.table(physical_name, database="main").to_pyarrow().to_pylist() == sentinel
    finally:
        con.disconnect()


@pytest.mark.parametrize("resolution_raises", [False, True])
@pytest.mark.parametrize("cleanup_fails_once", [False, True])
def test_unresolved_materialization_is_cleaned_up(
    monkeypatch, resolution_raises, cleanup_fails_once
):
    con, session = _session()
    sentinel = [{"sentinel": 314159}]
    con.create_table("owned", ibis.memtable(sentinel))
    expr = con.table("t")

    def unresolved(*args):
        if resolution_raises:
            raise RuntimeError("namespace metadata unavailable")
        return None

    raw_sql = con.raw_sql

    def drop_once(statement, *args, **kwargs):
        nonlocal cleanup_fails_once
        if (
            cleanup_fails_once
            and str(statement).startswith("DROP TABLE IF EXISTS")
            and "owned" in str(statement)
        ):
            cleanup_fails_once = False
            raise RuntimeError("temporary cleanup failure")
        return raw_sql(statement, *args, **kwargs)

    monkeypatch.setattr(session, "_materialized_relation", unresolved)
    monkeypatch.setattr(con, "raw_sql", drop_once)
    try:
        for _ in range(2):
            deferred_cleanup = cleanup_fails_once
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert session.materialize_table("owned", expr) is expr
            assert any("namespace could not be resolved" in str(w.message) for w in caught)
            assert ("owned" in con.list_tables(database=("temp", "main"))) == deferred_cleanup
            session.drop_materialized()
            assert "owned" not in con.list_tables(database=("temp", "main"))
            assert con.table("owned", database="main").to_pyarrow().to_pylist() == sentinel
    finally:
        con.disconnect()


@pytest.mark.parametrize("backend", ["postgres", "snowflake"])
@pytest.mark.parametrize("resolution_raises", [False, True])
def test_unresolved_backend_namespace_cleanup_survives_namespace_switch(
    monkeypatch, backend, resolution_raises
):
    from contextlib import nullcontext

    con, session = _session()
    con.create_database("materialized")
    con.create_database("other")
    sentinel = [{"sentinel": 314159}]
    con.create_table("renamed", ibis.memtable(sentinel), database="other")
    expr = con.table("t")
    physical_database = ("temp", "main") if backend == "postgres" else ("memory", "materialized")

    def create_table(name, obj, *, temp, database=None):
        if backend == "snowflake":
            assert database == '"memory"."materialized"'
            database = physical_database
        # Model Snowflake's ordinary schema for TEMP tables in in-memory DuckDB.
        created = con.create_table(
            "renamed", obj, temp=temp and backend == "postgres", database=database
        )
        # These backends can return an expression without namespace metadata.
        return created.op().copy(namespace=type(created.op().namespace)()).to_expr()

    def list_tables(*, database=None):
        if backend == "postgres" and database == "pg_temp":
            database = physical_database
        elif backend == "snowflake" and database == ("memory", "materialized"):
            database = physical_database
        return con.list_tables(database=database)

    def raw_sql(statement):
        import sqlglot

        drop = sqlglot.parse_one(statement, read=backend)
        targets = drop.args.get("tables") or [drop.this]
        expected = "pg_temp" if backend == "postgres" else "materialized"
        assert len(targets) == 1
        assert targets[0].name == "renamed"
        assert targets[0].db == expected
        assert targets[0].catalog == ("memory" if backend == "snowflake" else "")
        con.drop_table("renamed", database=physical_database, force=True)

    def safe_raw_sql(statement):
        return nullcontext(raw_sql(statement))

    adapter = SimpleNamespace(
        name=backend,
        dialect=backend,
        current_catalog="memory",
        current_database="materialized",
        create_table=create_table,
        list_tables=list_tables,
        raw_sql=raw_sql,
        _safe_raw_sql=safe_raw_sql,
    )

    def unresolved(*args):
        con.raw_sql("SET schema = 'other'")
        adapter.current_database = "other"
        if resolution_raises:
            raise RuntimeError("namespace metadata unavailable")
        return None

    monkeypatch.setattr(session, "_con", adapter)
    monkeypatch.setattr(session, "_materialized_relation", unresolved)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert session.materialize_table("owned", expr) is expr
        assert any("namespace could not be resolved" in str(w.message) for w in caught)
        assert "renamed" not in con.list_tables(database=physical_database)
        assert con.table("renamed", database="other").to_pyarrow().to_pylist() == sentinel
        session.drop_materialized()
        assert con.table("renamed", database="other").to_pyarrow().to_pylist() == sentinel
    finally:
        con.disconnect()


@pytest.mark.parametrize("cleanup", ["invalidation", "close"])
def test_snowflake_live_store_cleanup_targets_the_creation_schema_after_a_schema_change(
    tmp_path, monkeypatch, cleanup
):
    """Live ``store="always"`` tables drop from the schema they were created in, leaving
    same-named tables in a later default schema intact."""
    from increment import from_definitions, run

    definitions = tmp_path / "defs.yaml"
    definitions.write_text(
        """
dialect: snowflake
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - {name: enrolled, column: null}
      - {name: purchase, column: value}
exposures:
  - {name: enrollment, fact: enrolled}
metrics:
  - {name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 2}
experiments:
  - name: exp
    exposure: enrollment
    unit: user_id
    start: 2025-01-01T00:00:00
    end: 2025-01-03T00:00:00
    control_group: control
    plan: {secondaries: [revenue]}
"""
    )
    rows = [
        row
        for arm in ("control", "treatment")
        for index in range(4)
        for row in (
            {
                "user_id": f"{arm}{index}",
                "ts": datetime(2025, 1, 1, 8),
                "event": "enrolled",
                "experiment_id": "exp",
                "group_id": arm,
                "value": None,
            },
            {
                "user_id": f"{arm}{index}",
                "ts": datetime(2025, 1, 2, 9),
                "event": "purchase",
                "experiment_id": None,
                "group_id": None,
                "value": 10.0 + index,
            },
        )
    ]
    con = ibis.duckdb.connect()
    con.create_database("other")
    for schema in ("main", "other"):
        con.create_table("events", ibis.memtable(rows), database=schema)
    create_table = con.create_table
    created: list[tuple[str, tuple[str, str]]] = []

    def create_temp_in_current_schema(name, obj=None, *, temp=False, database=None, **kwargs):
        if not temp:
            return create_table(name, obj, database=database, **kwargs)
        # Model Snowflake's ordinary schema for TEMP tables in in-memory DuckDB.
        location = ("memory", con.current_database)
        assert database == f'"memory"."{location[1]}"'
        created.append((name, location))
        return create_table(name, obj, database=location, **kwargs)

    monkeypatch.setattr(con, "name", "snowflake")
    monkeypatch.setattr(con, "create_table", create_temp_in_current_schema)
    sentinel = [{"sentinel": 314159}]

    def live_readout(experiment, resolver):
        start = len(created)
        run(experiment, resolver, coherence="live")
        batch = tuple(created[start:])
        assert sorted(name.rsplit("_", 1)[0] for name, _ in batch) == [
            "exp_spine",
            "exp_stats_revenue",
        ]
        assert all(name in con.list_tables(database=location) for name, location in batch)
        return batch

    def shadow_in_other_schema(batch):
        for name, _ in batch:
            con.create_table(name, ibis.memtable(sentinel), database="other")
        con.raw_sql('USE "memory"."other"')

    try:
        experiment, resolver = from_definitions(
            definitions, con, experiment_name="exp", store="always"
        )
        first = live_readout(experiment, resolver)
        if cleanup == "invalidation":
            shadow_in_other_schema(first)
        second = live_readout(experiment, resolver)
        assert not any(name in con.list_tables(database=location) for name, location in first)
        if cleanup == "close":
            shadow_in_other_schema(second)
        cast(Any, resolver).close()
        assert not any(name in con.list_tables(database=location) for name, location in second)
        for name, _ in first if cleanup == "invalidation" else second:
            assert con.table(name, database="other").to_pyarrow().to_pylist() == sentinel
    finally:
        con.disconnect()


def test_snowflake_unknown_creation_namespace_does_not_create(monkeypatch):
    con, session = _session()
    expr = con.table("t")
    before = con.list_tables(database=("temp", "main"))
    adapter = SimpleNamespace(
        name="snowflake", dialect="snowflake", current_catalog="memory", current_database=None
    )

    def create_table(*args, **kwargs):
        pytest.fail("cannot safely create a TEMP table without a known namespace")

    adapter.create_table = create_table
    monkeypatch.setattr(session, "_con", adapter)
    try:
        with pytest.warns(IncrementWarning) as rec:
            assert session.materialize_table("owned", expr) is expr
        assert "query.session.materialization_degraded" in warning_codes(rec)
        assert con.list_tables(database=("temp", "main")) == before
    finally:
        con.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("backend", ["postgres", "snowflake", "bigquery"])
def test_non_duckdb_partial_temp_create_is_cleaned_in_creation_namespace(backend):
    from contextlib import nullcontext

    con, session = _session()
    con.create_database("materialized")
    con.create_database("other")
    name = session.temp_name("partial_backend")
    sentinel = [{"sentinel": 314159}]
    con.create_table(name, ibis.memtable(sentinel), database="other")
    physical_database = ("memory", "materialized") if backend == "snowflake" else ("temp", "main")

    def create_table(relation_name, obj, *, temp, database=None):
        if backend == "snowflake":
            assert database == '"memory"."materialized"'
            database = physical_database
        con.create_table(
            relation_name,
            obj,
            temp=temp and backend in {"postgres", "bigquery"},
            database=database,
        )
        raise RuntimeError("create response lost after commit")

    def list_tables(*, database=None):
        if backend == "postgres" and database == "pg_temp":
            database = physical_database
        elif backend == "snowflake" and database == ("memory", "materialized"):
            database = physical_database
        return con.list_tables(database=database)

    def table(relation_name, *, database=None):
        assert database == ("memory", "session_dataset")
        return con.table(relation_name, database=physical_database)

    def raw_sql(statement):
        import sqlglot

        drop = sqlglot.parse_one(statement, read=backend)
        targets = drop.args.get("tables") or [drop.this]

        expected_schema = {
            "postgres": "pg_temp",
            "snowflake": "materialized",
            "bigquery": "session_dataset",
        }[backend]
        expected_catalog = "memory" if backend != "postgres" else ""
        assert len(targets) == 1
        assert targets[0].name == name
        assert targets[0].db == expected_schema
        assert targets[0].catalog == expected_catalog
        con.drop_table(name, database=physical_database, force=True)

    def safe_raw_sql(statement):
        return nullcontext(raw_sql(statement))

    adapter = SimpleNamespace(
        name=backend,
        dialect=backend,
        current_catalog="memory",
        current_database="materialized",
        _session_dataset=SimpleNamespace(project="memory", dataset_id="session_dataset"),
        create_table=create_table,
        list_tables=list_tables,
        raw_sql=raw_sql,
        _safe_raw_sql=safe_raw_sql,
    )
    if backend == "bigquery":
        # BigQuery looks one table up rather than listing its session dataset.
        adapter.table = table
        adapter.client = SimpleNamespace(get_dataset=lambda dataset_ref: dataset_ref)
    session = WarehouseSession(adapter, defs=None)  # type: ignore[ty:invalid-argument-type]
    expression = con.table("t")
    try:
        with pytest.raises(CapabilityError) as raised:
            session.materialize_table(name, expression, required=True)
        assert raised.value.code == "query.session.snapshot.materialization_failed"
        assert name not in con.list_tables(database=physical_database)
        assert con.table(name, database="other").to_pyarrow().to_pylist() == sentinel
    finally:
        session.drop_materialized()
        con.disconnect()


def test_materialize_table_degrades_with_warning(monkeypatch):
    con, s = _session()

    def boom(*a, **k):
        raise RuntimeError("no CREATE TEMP privilege")

    monkeypatch.setattr(con, "create_table", boom)
    expr = con.table("t")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = s.materialize_table(s.temp_name("stats"), expr)
    assert out is expr  # degrades to the unmaterialized expression
    assert "query.session.materialization_degraded" in warning_codes(w)
    assert s.materialized_names == []


def test_fact_table_memoized_per_source_and_unit():
    # Build real Definitions + event_log the way tests/test_analysis.py does.
    from tests.analysis_factory import _defs_and_con_for_session_tests

    defs, con = _defs_and_con_for_session_tests()
    s = WarehouseSession(con, defs)
    fs = defs.fact_sources[0]
    unit = fs.entities[0]
    t1 = s.fact_table(fs, unit)
    t2 = s.fact_table(fs, unit)
    assert t1 is t2  # memoized: same ibis expression object, no rebuild


def _artifact_context() -> ArtifactContext:
    payload = (
        '{"context_format":2,"experiment_name":"demo",'
        '"window_days":{"end":null,"observation_horizon":null,"start":"2025-01-01"}}'
    )
    return ArtifactContext(
        canonical_json=payload,
        sha256=hashlib.sha256(ARTIFACT_DOMAIN_ROOT + b"context\x00" + payload.encode()).hexdigest(),
    )


def _publish_in(store: WarehouseArtifactStore, *, unit_id: str) -> tuple[Any, Any]:
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation(
            "exposures",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": unit_id,
                        "group_id": "control",
                        "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                        "first_exposure_date": date(2024, 1, 1),
                    }
                ]
            ),
        )
        measure_ref = publication.write_relation(
            "measure_stats",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": unit_id,
                        "ds": date(2024, 1, 1),
                        "measure_key": "measure_a",
                        "n_events": 1,
                        "sum_value": 2.0,
                        "min_value": 2.0,
                        "max_value": 2.0,
                    }
                ]
            ),
        )
        body: dict[str, Any] = {
            "artifact_format": 1,
            "artifact_id": publication.artifact_id,
            "generation_id": publication.generation_id,
            "experiment_id": "demo",
            "created_at": datetime(2024, 1, 1, tzinfo=UTC),
            "day_boundary": "UTC",
            "first_ds": date(2024, 1, 1),
            "last_ds": date(2024, 1, 1),
            "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref),
            "measures": (
                MeasureManifest(
                    measure_key="measure_a",
                    source_provenance_sha256="0" * 64,
                    freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
                ),
            ),
            "metric_measures": (),
            "context": context,
        }
        prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
        body["manifest_sha256"] = manifest_sha256(prototype)
        ref = publication.publish_manifest(UnitDayArtifactManifest(**body))
    return ref, measure_ref


def _manifest_body(
    context: ArtifactContext,
    exposure_ref,
    measure_ref,
    *,
    artifact_id,
    generation_id,
) -> dict[str, Any]:
    return {
        "artifact_format": 1,
        "artifact_id": artifact_id,
        "generation_id": generation_id,
        "experiment_id": "demo",
        "created_at": datetime(2024, 1, 1, tzinfo=UTC),
        "day_boundary": "UTC",
        "first_ds": date(2024, 1, 1),
        "last_ds": date(2024, 1, 1),
        "base": BaseRelations(exposures=exposure_ref, measure_stats=measure_ref)
        if measure_ref.role == "measure_stats"
        else BaseRelations.model_construct(exposures=exposure_ref, measure_stats=measure_ref),
        "measures": (
            MeasureManifest(
                measure_key="measure_a",
                source_provenance_sha256="0" * 64,
                freshness={"loaded_through": date(2024, 1, 1), "declared_complete": True},
            ),
        ),
        "metric_measures": (),
        "context": context,
    }


def _sealed_manifest(body: dict[str, Any]) -> UnitDayArtifactManifest:
    prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
    body = dict(body, manifest_sha256=manifest_sha256(prototype))
    return UnitDayArtifactManifest(**body)


def test_ensure_open_refuses_once_the_publication_block_closes() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    with store.begin_publication(expected_context=_artifact_context()) as publication:
        pass
    with pytest.raises(InvalidRequestError) as exc_info:
        publication.write_relation("exposures", ibis.memtable({"x": [1]}))
    assert exc_info.value.code == "query.session.artifact_publication.handle_closed"


def test_write_relation_rejects_an_undeclared_role() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    with store.begin_publication(expected_context=_artifact_context()) as publication:
        with pytest.raises(InvalidRequestError) as exc_info:
            publication.write_relation("bogus_role", ibis.memtable({"x": [1]}))
        assert exc_info.value.code == "query.session.warehouse_artifact.unknown_relation_role"
        assert exc_info.value.context["role"] == "bogus_role"


def test_write_relation_and_publish_manifest_refuse_once_sealed() -> None:
    """Both sealed-publication entry points use the canonical refusal code."""
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref = publication.write_relation(
            "exposures",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "group_id": "control",
                        "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                        "first_exposure_date": date(2024, 1, 1),
                    }
                ]
            ),
        )
        measure_ref = publication.write_relation(
            "measure_stats",
            ibis.memtable(
                [
                    {
                        "experiment_id": "demo",
                        "unit_id": "u1",
                        "ds": date(2024, 1, 1),
                        "measure_key": "measure_a",
                        "n_events": 1,
                        "sum_value": 2.0,
                        "min_value": 2.0,
                        "max_value": 2.0,
                    }
                ]
            ),
        )
        body = _manifest_body(
            context,
            exposure_ref,
            measure_ref,
            artifact_id=publication.artifact_id,
            generation_id=publication.generation_id,
        )
        manifest = _sealed_manifest(body)
        publication.publish_manifest(manifest)

        with pytest.raises(InvalidRequestError) as exc_info:
            publication.write_relation("exposures", ibis.memtable({"x": [1]}))
        assert exc_info.value.code == "query.session.warehouse_artifact.publication_handle_sealed"

        with pytest.raises(InvalidRequestError) as exc_info:
            publication.publish_manifest(manifest)
        assert exc_info.value.code == "query.session.warehouse_artifact.publication_handle_sealed"


def _open_exposure_and_measure(publication):
    exposure_ref = publication.write_relation(
        "exposures",
        ibis.memtable(
            [
                {
                    "experiment_id": "demo",
                    "unit_id": "u1",
                    "group_id": "control",
                    "first_exposure_ts": datetime(2024, 1, 1, tzinfo=UTC),
                    "first_exposure_date": date(2024, 1, 1),
                }
            ]
        ),
    )
    measure_ref = publication.write_relation(
        "measure_stats",
        ibis.memtable(
            [
                {
                    "experiment_id": "demo",
                    "unit_id": "u1",
                    "ds": date(2024, 1, 1),
                    "measure_key": "measure_a",
                    "n_events": 1,
                    "sum_value": 2.0,
                    "min_value": 2.0,
                    "max_value": 2.0,
                }
            ]
        ),
    )
    return exposure_ref, measure_ref


def test_publish_manifest_rejects_ids_not_bound_to_this_publication() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref, measure_ref = _open_exposure_and_measure(publication)
        body = _manifest_body(
            context, exposure_ref, measure_ref, artifact_id=uuid4(), generation_id=uuid4()
        )
        # model_construct skips the manifest's own artifact/generation validator,
        # so publication's binding check against the store's generation fires.
        unbound = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="0" * 64)
        with pytest.raises(InvalidRequestError) as exc_info:
            publication.publish_manifest(unbound)
        assert exc_info.value.code == "query.session.warehouse_artifact.manifest_bound_publication"


def test_publish_manifest_rejects_a_context_mismatch() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref, measure_ref = _open_exposure_and_measure(publication)
        body = _manifest_body(
            context,
            exposure_ref,
            measure_ref,
            artifact_id=publication.artifact_id,
            generation_id=publication.generation_id,
        )
        other_payload = '{"context_format":2,"experiment_name":"other"}'
        body["context"] = ArtifactContext(
            canonical_json=other_payload,
            sha256=hashlib.sha256(
                ARTIFACT_DOMAIN_ROOT + b"context\x00" + other_payload.encode()
            ).hexdigest(),
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            publication.publish_manifest(_sealed_manifest(body))
        assert exc_info.value.code == "query.session.warehouse_artifact.manifest_context_differs"


def test_publish_manifest_rejects_a_digest_that_does_not_match_the_body() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref, measure_ref = _open_exposure_and_measure(publication)
        body = _manifest_body(
            context,
            exposure_ref,
            measure_ref,
            artifact_id=publication.artifact_id,
            generation_id=publication.generation_id,
        )
        # A wrong manifest_sha256 is refused by UnitDayArtifactManifest's own
        # validator first -- model_construct skips that so the publication's
        # own digest recomputation is what fires.
        tampered = UnitDayArtifactManifest.model_construct(**dict(body, manifest_sha256="0" * 64))  # ty: ignore[invalid-argument-type]
        with pytest.raises(InvalidRequestError) as exc_info:
            publication.publish_manifest(tampered)
        assert exc_info.value.code == "query.session.warehouse_artifact.manifest_digest_does"


def test_publish_manifest_rejects_duplicate_relation_locators() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref, _measure_ref = _open_exposure_and_measure(publication)
        # BaseRelations itself refuses a measure_stats slot holding a
        # non-measure_stats-role ref -- _manifest_body falls back to
        # model_construct for that slot, so the publication's own
        # locator-uniqueness check is what fires.
        body = _manifest_body(
            context,
            exposure_ref,
            exposure_ref,
            artifact_id=publication.artifact_id,
            generation_id=publication.generation_id,
        )
        prototype = UnitDayArtifactManifest.model_construct(**body, manifest_sha256="")
        manifest = UnitDayArtifactManifest.model_construct(
            **body, manifest_sha256=manifest_sha256(prototype)
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            publication.publish_manifest(manifest)
        assert exc_info.value.code == "query.session.warehouse_artifact.manifest_relation_locators"


def test_publish_manifest_rejects_a_relation_the_publication_never_wrote() -> None:
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    context = _artifact_context()
    with store.begin_publication(expected_context=context) as publication:
        exposure_ref, measure_ref = _open_exposure_and_measure(publication)
        drifted_measure_ref = measure_ref.model_copy(
            update={"row_count": measure_ref.row_count + 1}
        )
        body = _manifest_body(
            context,
            exposure_ref,
            drifted_measure_ref,
            artifact_id=publication.artifact_id,
            generation_id=publication.generation_id,
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            publication.publish_manifest(_sealed_manifest(body))
        assert (
            exc_info.value.code == "query.session.warehouse_artifact.manifest_references_unwritten"
        )


@pytest.mark.integration
def test_drop_generation_is_idempotent_in_qualified_namespace() -> None:
    con = ibis.duckdb.connect()
    catalog = con.current_catalog

    # ibis 12.0.0's DuckDB insert()/drop_table() fail for schemas named by reserved
    # words (select, table, order, where, from): sqlglot force-quotes the name and
    # ibis quotes it again. With no raw_sql escape hatch, only construction and
    # reads run against "select".
    reserved_store = WarehouseArtifactStore(con, schema_name="select")
    assert reserved_store.visible_manifests == ()

    # Exercise the full publish/drop lifecycle against a genuinely qualified
    # two-part catalog+schema namespace, which round-trips correctly.
    con.create_database("ud_reporting", force=True)
    store = WarehouseArtifactStore(con, catalog=catalog, schema_name="ud_reporting")

    ref_a, _measure_a = _publish_in(store, unit_id="ua")
    ref_b, measure_b = _publish_in(store, unit_id="ub")

    # Simulate an old deployment: the tombstone table predates this release.
    con.con.execute(f'drop table if exists "{catalog}"."ud_reporting".ud_manifest_dropped')
    reopened = WarehouseArtifactStore(con, catalog=catalog, schema_name="ud_reporting")
    assert set(reopened.visible_manifests) == {ref_a, ref_b}

    reopened.drop_generation(ref_a.artifact_id, ref_a.generation_id)
    reopened.drop_generation(ref_a.artifact_id, ref_a.generation_id)  # idempotent repeat
    reopened.drop_generation(uuid4(), uuid4())  # unknown pair is a no-op

    assert reopened.visible_manifests == (ref_b,)
    with pytest.raises(ArtifactContractError) as refusal:
        with reopened.open_snapshot(ref_a):
            pass
    assert refusal.value.code == "artifact.generation.dropped"

    with reopened.open_snapshot(ref_b) as snapshot:
        rows = (
            snapshot.verify_relation(measure_b, expected_role="measure_stats")
            .execute()
            .to_dict("records")
        )
    assert rows[0]["unit_id"] == "ub"
    assert rows[0]["sum_value"] == 2.0


@pytest.mark.slow
@pytest.mark.parametrize("reopen", [False, True])
def test_streaming_store_hashes_persisted_execution_and_cleans_snapshot(monkeypatch, reopen):
    from duckdb import CatalogException

    from increment.query.artifact_digest import digest_relation_sql_v2
    from increment.query.schemas import ARTIFACT_RELATION_PRIMARY_KEYS, ARTIFACT_RELATION_SCHEMAS

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    create = con.create_table
    arrow = con.to_pyarrow
    observations = []

    def create_changed(name, obj=None, **kwargs):
        if name.endswith("_measure_stats"):
            assert obj is not None
            obj = obj.mutate(sum_value=7.0, min_value=7.0, max_value=7.0)
        result = create(name, obj, **kwargs)
        if name.startswith("ud_snap_"):
            assert kwargs["temp"] is True
            observations.append(result)
        return result

    def no_base_fetch(expr, **kwargs):
        # Scalar aggregates (the digest's duplicate-key/cell-validity checks)
        # have no `.columns`; only a full-row fetch can read a base relation.
        columns = getattr(expr, "columns", None)
        assert columns is None or "unit_id" not in columns, "base relations must use Arrow batches"
        return arrow(expr, **kwargs)

    monkeypatch.setattr(con, "create_table", create_changed)
    monkeypatch.setattr(con, "to_pyarrow", no_base_fetch)
    try:
        ref, measure = _publish_in(store, unit_id="u")
        if reopen:
            store = WarehouseArtifactStore(con, schema_name="artifacts")
        with store.open_snapshot(ref) as snapshot:
            table = snapshot.verify_relation(measure, expected_role="measure_stats")
            rows = arrow(table).to_pylist()
            assert measure.digest_format == 4
            expected = digest_relation_sql_v2(
                con,
                table,
                "measure_stats",
                ARTIFACT_RELATION_SCHEMAS["measure_stats"],
                primary_key=ARTIFACT_RELATION_PRIMARY_KEYS["measure_stats"],
            )
            assert rows[0]["sum_value"] == 7.0
            assert expected.content_sha256 == measure.content_sha256
        assert observations
        for private in observations:
            with pytest.raises(CatalogException):
                arrow(private)
        assert con.table(measure.relation.name, database="artifacts") is not None
    finally:
        con.disconnect()


@pytest.mark.slow
def test_streaming_snapshot_isolates_post_verification_external_mutation():
    from duckdb import CatalogException

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    try:
        ref, measure = _publish_in(store, unit_id="u")
        with store.open_snapshot(ref) as snapshot:
            verified = snapshot.verify_relation(measure, expected_role="measure_stats")
            assert snapshot.execute(verified).column("sum_value").to_pylist() == [2.0]

            con.raw_sql(f'UPDATE artifacts."{measure.relation.name}" SET sum_value = 99')
            published = con.table(measure.relation.name, database="artifacts")
            assert con.to_pyarrow(published).column("sum_value").to_pylist() == [99.0]
            assert snapshot.execute(verified).column("sum_value").to_pylist() == [2.0]

        with pytest.raises(CatalogException):
            con.to_pyarrow(verified)
        with store.open_snapshot(ref) as fresh:
            with pytest.raises(ArtifactContractError) as raised:
                fresh.verify_relation(measure, expected_role="measure_stats")
            assert raised.value.code == "artifact.snapshot.mixed"
    finally:
        con.disconnect()


@pytest.mark.slow
def test_streaming_store_failed_write_drops_unregistered_relation(monkeypatch):
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    before = set(con.list_tables(database="artifacts"))
    create = con.create_table
    fetch = con.to_pyarrow
    written = []
    failed = False

    def track_write(name, *args, **kwargs):
        result = create(name, *args, **kwargs)
        if str(name).endswith("_exposures"):
            written.append(name)
        return result

    def broken(expression, *args, **kwargs):
        nonlocal failed
        if written and not failed:
            failed = True
            raise RuntimeError("stream failure")
        return fetch(expression, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", track_write)
    monkeypatch.setattr(con, "to_pyarrow", broken)
    try:
        with pytest.raises(RuntimeError, match="stream failure"):
            _publish_in(store, unit_id="u")
        assert failed and written
        assert set(con.list_tables(database="artifacts")) == before
        assert store.visible_manifests == ()
    finally:
        con.disconnect()


@pytest.mark.integration
def test_publication_collision_does_not_drop_existing_relation():
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    try:
        with store.begin_publication(expected_context=_artifact_context()) as publication:
            name = (
                f"ud_{publication.artifact_id.hex[:12]}_"
                f"{publication.generation_id.hex[:12]}_exposures"
            )
            con.raw_sql(f'CREATE TABLE artifacts."{name}" (sentinel INTEGER)')
            con.raw_sql(f'INSERT INTO artifacts."{name}" VALUES (314159)')
            source = ibis.table({"sentinel": "int64"}, name="not_materialized")

            with pytest.raises(CapabilityError) as raised:
                publication.write_relation("exposures", source)
            assert raised.value.code == "query.relation.name_occupied"
            assert set(raised.value.context) == {"backend", "catalog", "schema", "name", "route"}
            assert {
                key: value for key, value in raised.value.context.items() if key != "route"
            } == {
                "backend": "duckdb",
                "catalog": None,
                "schema": "artifacts",
                "name": name,
            }
            assert isinstance(raised.value.context["route"], str)
            with pytest.raises(TypeError):
                raised.value.context["name"] = "different"  # type: ignore[ty:invalid-assignment]

            assert name in con.list_tables(database="artifacts")
            assert con.table(name, database="artifacts").to_pyarrow().to_pylist() == [
                {"sentinel": 314159}
            ]
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("reopen", [False, True])
@pytest.mark.parametrize("column", ["ds", "n_events", "sum_value"])
def test_empty_snapshot_refuses_incorrect_column_types(monkeypatch, reopen, column):
    from increment.query.artifact_digest import ArtifactDigestError

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    create = con.create_table

    def empty_measure(name, obj=None, **kwargs):
        if name.endswith("_measure_stats"):
            assert obj is not None
            obj = obj.limit(0)
        return create(name, obj, **kwargs)

    monkeypatch.setattr(con, "create_table", empty_measure)
    try:
        ref, measure = _publish_in(store, unit_id="u")
        assert measure.row_count == 0
        con.raw_sql(
            f'ALTER TABLE artifacts."{measure.relation.name}" ALTER COLUMN {column} TYPE VARCHAR'
        )
        if reopen:
            store = WarehouseArtifactStore(con, schema_name="artifacts")
        with store.open_snapshot(ref) as snapshot:
            with pytest.raises(ArtifactDigestError) as raised:
                snapshot.verify_relation(measure, expected_role="measure_stats")
            assert raised.value.code == "artifact.digest.type"
        assert not any(name.startswith("ud_snap_") for name in con.list_tables())
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("failure", ["digest", "mismatch", "body"])
def test_streaming_snapshot_failure_cleans_private_tables(monkeypatch, failure):
    from duckdb import CatalogException

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref, measure = _publish_in(store, unit_id="u")
    private_tables = []
    create = con.create_table
    fetch = con.to_pyarrow
    failed_digest = False

    def tracked(*args, **kwargs):
        result = create(*args, **kwargs)
        if kwargs.get("temp"):
            private_tables.append(result)
        return result

    def broken(expression, *args, **kwargs):
        nonlocal failed_digest
        if private_tables and not failed_digest:
            failed_digest = True
            raise RuntimeError("stream failure")
        return fetch(expression, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", tracked)
    if failure == "digest":
        monkeypatch.setattr(con, "to_pyarrow", broken)
    if failure == "mismatch":
        con.raw_sql(f'UPDATE artifacts."{measure.relation.name}" SET sum_value = 99')
    expected_private_tables = 1 if failure == "digest" else 2
    try:
        with pytest.raises((RuntimeError, ArtifactContractError)):
            with store.open_snapshot(ref) as snapshot:
                snapshot.verify_relation(measure, expected_role="measure_stats")
                if failure == "body":
                    raise RuntimeError("body failure")
        assert failed_digest is (failure == "digest")
        assert len(private_tables) == expected_private_tables
        for private in private_tables:
            with pytest.raises(CatalogException):
                con.to_pyarrow(private)
    finally:
        con.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("failure", ("drop", "catalog_probe"))
def test_snapshot_partial_create_cleanup_failure_is_retried(monkeypatch, failure):
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref, measure = _publish_in(store, unit_id=f"snapshot-retry-{failure}")
    before_temp = set(con.list_tables(database=("temp", "main")))
    create = con.create_table
    raw_sql = con.raw_sql
    list_tables = con.list_tables
    snapshot_names = []
    failed_drop = False
    failed_probe = False

    def create_then_raise(name, obj=None, **kwargs):
        result = create(name, obj, **kwargs)
        if str(name).startswith("ud_snap_"):
            snapshot_names.append(str(name))
            raise RuntimeError("lost snapshot create response")
        return result

    def fail_first_snapshot_drop(statement, *args, **kwargs):
        nonlocal failed_drop
        if snapshot_names and snapshot_names[-1] in str(statement) and not failed_drop:
            failed_drop = True
            raise RuntimeError("transient snapshot cleanup failure")
        return raw_sql(statement, *args, **kwargs)

    def fail_snapshot_catalog_probe(*args, **kwargs):
        nonlocal failed_probe
        if snapshot_names and not failed_probe and kwargs.get("database") == ("temp", "main"):
            failed_probe = True
            raise RuntimeError("transient snapshot catalog probe failure")
        return list_tables(*args, **kwargs)

    monkeypatch.setattr(con, "create_table", create_then_raise)
    if failure == "drop":
        monkeypatch.setattr(con, "raw_sql", fail_first_snapshot_drop)
    else:
        monkeypatch.setattr(con, "list_tables", fail_snapshot_catalog_probe)
    try:
        with store.open_snapshot(ref) as snapshot:
            with pytest.raises(RuntimeError, match="lost snapshot create response"):
                snapshot.verify_relation(measure, expected_role="measure_stats")
        assert failed_drop is (failure == "drop")
        assert failed_probe is (failure == "catalog_probe")
        assert snapshot_names
        assert snapshot_names[-1] in con.list_tables(database=("temp", "main"))
        assert set(con.list_tables(database=("temp", "main"))) == before_temp | set(snapshot_names)

        with store.open_snapshot(ref):
            pass
        assert snapshot_names[-1] not in con.list_tables(database=("temp", "main"))
        assert con.table(measure.relation.name, database="artifacts") is not None
    finally:
        con.disconnect()


@pytest.mark.integration
def test_digest_cleanup_failure_is_retained_for_retry(monkeypatch):
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref, measure = _publish_in(store, unit_id="digest-retry")
    before_temp = set(con.list_tables(database=("temp", "main")))
    create = con.create_table
    created_hashes = []
    raw_sql = con.raw_sql
    failed_drop = False

    def track_hash_table(name, *args, **kwargs):
        result = create(name, *args, **kwargs)
        if str(name).startswith("incr_digest_"):
            created_hashes.append(str(name))
        return result

    def fail_first_digest_drop(statement, *args, **kwargs):
        nonlocal failed_drop
        if (
            not failed_drop
            and str(statement).startswith("DROP TABLE IF EXISTS")
            and "incr_digest_" in str(statement)
        ):
            failed_drop = True
            raise RuntimeError("transient digest cleanup failure")
        return raw_sql(statement, *args, **kwargs)

    monkeypatch.setattr(con, "raw_sql", fail_first_digest_drop)
    monkeypatch.setattr(con, "create_table", track_hash_table)
    try:
        with store.open_snapshot(ref) as snapshot:
            with pytest.raises(RuntimeError, match="transient digest cleanup failure"):
                snapshot.verify_relation(measure, expected_role="measure_stats")
        assert failed_drop
        assert len(created_hashes) == 1
        name = created_hashes[0]
        assert set(con.list_tables(database=("temp", "main"))) == before_temp | {name}

        with store.qr_scratch():
            pass
        assert name not in con.list_tables(database=("temp", "main"))
    finally:
        with store.qr_scratch():
            pass
        con.disconnect()


@pytest.mark.integration
def test_digest_body_error_remains_primary_when_cleanup_also_fails(monkeypatch):
    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref, measure = _publish_in(store, unit_id="digest-primary-error")
    before_temp = set(con.list_tables(database=("temp", "main")))
    raw_sql = con.raw_sql
    fetch = con.to_pyarrow
    create_table = con.create_table
    created_hashes = []
    failed_drop = False
    failed_body = False

    def track_hash_table(name, obj=None, **kwargs):
        result = create_table(name, obj, **kwargs)
        if str(name).startswith("incr_digest_"):
            created_hashes.append(str(name))
        return result

    def fail_digest_body(expression, *args, **kwargs):
        nonlocal failed_body
        if created_hashes and not failed_body:
            failed_body = True
            raise RuntimeError("primary digest computation failure")
        return fetch(expression, *args, **kwargs)

    def fail_digest_cleanup(statement, *args, **kwargs):
        nonlocal failed_drop
        if (
            not failed_drop
            and str(statement).startswith("DROP TABLE IF EXISTS")
            and "incr_digest_" in str(statement)
        ):
            failed_drop = True
            raise RuntimeError("digest cleanup failure")
        return raw_sql(statement, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", track_hash_table)
    monkeypatch.setattr(con, "raw_sql", fail_digest_cleanup)
    monkeypatch.setattr(con, "to_pyarrow", fail_digest_body)
    try:
        with store.open_snapshot(ref) as snapshot:
            with pytest.raises(RuntimeError, match="primary digest computation failure") as raised:
                snapshot.verify_relation(measure, expected_role="measure_stats")
        assert failed_drop
        assert isinstance(raised.value.__cause__, RuntimeError)
        assert str(raised.value.__cause__) == "digest cleanup failure"
        assert created_hashes
        assert set(con.list_tables(database=("temp", "main"))) == before_temp | set(created_hashes)

        with store.qr_scratch():
            pass
        assert created_hashes[0] not in con.list_tables(database=("temp", "main"))
        assert con.table(measure.relation.name, database="artifacts") is not None
    finally:
        with store.qr_scratch():
            pass
        con.disconnect()


_HOSTED_NAMES = [
    ("My_Cat", "Sch"),
    ("analytics", "prod_artifacts"),
    ("Mixed_Case", "lower"),
    ("Cat$1", "ds_2"),
    ("analytics-prod", "ud_hosted"),
]


@pytest.mark.parametrize("backend_name", ["snowflake", "bigquery"])
@pytest.mark.parametrize(("catalog", "schema"), _HOSTED_NAMES)
def test_hosted_metadata_creation_targets_the_explicit_catalog(
    monkeypatch: pytest.MonkeyPatch, backend_name: str, catalog: str, schema: str
) -> None:
    """Execute real hosted create_table SQL in DuckDB; no cloud authentication is exercised."""
    sqlglot = pytest.importorskip("sqlglot")
    module = pytest.importorskip(f"ibis.backends.{backend_name}")
    backend = module.Backend()
    if backend_name == "bigquery":
        backend.billing_project = "billing-project"
        backend.data_project = "default-project"
        backend.dataset = "default_dataset"
    real = ibis.duckdb.connect()
    try:
        quoted_catalog = sqlglot.exp.to_identifier(catalog, quoted=True).sql("duckdb")
        quoted_schema = sqlglot.exp.to_identifier(schema, quoted=True).sql("duckdb")
        real.raw_sql(f"ATTACH ':memory:' AS {quoted_catalog}")
        real.raw_sql(f"CREATE SCHEMA {quoted_catalog}.{quoted_schema}")
        sentinel = {
            "artifact_id": "default-sentinel",
            "generation_id": "default-sentinel",
            "dropped_at": "2025-01-01T00:00:00+00:00",
        }
        real.create_table("ud_manifest_dropped", ibis.memtable([sentinel]))

        def execute_hosted_sql(query: str):
            statement = sqlglot.parse_one(query, read=backend_name)
            cursor = real.con.cursor()
            try:
                cursor.execute(statement.sql("duckdb"))
            except BaseException:
                cursor.close()
                raise
            if backend_name == "bigquery":
                cursor.close()
                return None
            return cursor

        monkeypatch.setattr(backend, "raw_sql", execute_hosted_sql)
        for operation in (
            "table",
            "list_tables",
            "insert",
            "drop_table",
            "execute",
            "to_pyarrow",
        ):
            monkeypatch.setattr(backend, operation, getattr(real, operation))
        store = WarehouseArtifactStore(backend, catalog=catalog, schema_name=schema)
        artifact_id, generation_id = uuid4(), uuid4()
        store.abandon_generation(artifact_id, generation_id)
        rows = real.table("ud_manifest_dropped", database=(catalog, schema)).execute()
        assert rows[["artifact_id", "generation_id"]].to_dict("records") == [
            {"artifact_id": str(artifact_id), "generation_id": str(generation_id)}
        ]
        assert real.table("ud_manifest_dropped", database=("memory", "main")).execute().to_dict(
            "records"
        ) == [sentinel]
    finally:
        real.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("catalog", "schema", "project", "dataset"),
    [
        (None, None, "data-project", "default_dataset"),
        ("catalog-only", None, "data-project", "default_dataset"),
        (None, "explicit_dataset", "billing-project", "explicit_dataset"),
        ("quoted-project", "explicit_dataset", "quoted-project", "explicit_dataset"),
    ],
)
def test_bigquery_store_creation_uses_effective_hosted_sql_namespace(
    monkeypatch: pytest.MonkeyPatch,
    catalog: str | None,
    schema: str | None,
    project: str,
    dataset: str,
) -> None:
    sqlglot = pytest.importorskip("sqlglot")
    module = pytest.importorskip("ibis.backends.bigquery")
    backend = module.Backend()
    backend.billing_project = "billing-project"
    backend.data_project = "data-project"
    backend.dataset = "default_dataset"
    real = ibis.duckdb.connect()
    try:
        quoted_project = sqlglot.exp.to_identifier(project, quoted=True).sql("duckdb")
        quoted_dataset = sqlglot.exp.to_identifier(dataset, quoted=True).sql("duckdb")
        real.raw_sql(f"ATTACH ':memory:' AS {quoted_project}")
        real.raw_sql(f"CREATE SCHEMA {quoted_project}.{quoted_dataset}")

        hosted_targets: list[tuple[str | None, str | None]] = []

        def execute_hosted_sql(query: str):
            statement = sqlglot.parse_one(query, read="bigquery")
            target = statement.this.this
            hosted_targets.append((target.catalog, target.db))
            cursor = real.con.cursor()
            try:
                cursor.execute(statement.sql("duckdb"))
            except BaseException:
                cursor.close()
                raise
            cursor.close()
            return None

        monkeypatch.setattr(backend, "raw_sql", execute_hosted_sql)
        for operation in ("table", "list_tables", "insert", "drop_table", "execute", "to_pyarrow"):
            monkeypatch.setattr(backend, operation, getattr(real, operation))
        WarehouseArtifactStore(backend, catalog=catalog, schema_name=schema)
        assert hosted_targets == [(project, dataset), (project, dataset)]
        assert "ud_manifest_index" in real.list_tables(database=(project, dataset))
        assert "ud_manifest_dropped" in real.list_tables(database=(project, dataset))
    finally:
        real.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("catalog", "schema", "expected_project", "expected_dataset"),
    [
        (None, None, "data-project", "default_dataset"),
        ("catalog-only", None, "data-project", "default_dataset"),
        (None, "explicit_dataset", "billing-project", "explicit_dataset"),
        ("quoted-project", "explicit_dataset", "quoted-project", "explicit_dataset"),
    ],
)
def test_bigquery_publication_abort_drops_in_effective_dataset(
    monkeypatch: pytest.MonkeyPatch,
    catalog: str | None,
    schema: str | None,
    expected_project: str,
    expected_dataset: str,
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    try:
        boundary.backend.create_table(
            "source",
            ibis.memtable([{"value": "preserve"}]),
            database=(expected_project, expected_dataset),
        )
        store = WarehouseArtifactStore(boundary.backend, catalog=catalog, schema_name=schema)
        names: tuple[str, ...] = ()
        with pytest.raises(RuntimeError, match="abort publication"):
            with store.begin_publication(expected_context=_artifact_context()) as publication:
                exposure, measure = _open_exposure_and_measure(publication)
                names = (exposure.relation.name, measure.relation.name)
                raise RuntimeError("abort publication")

        assert names
        assert all(
            name not in boundary.backend.list_tables(database=(expected_project, expected_dataset))
            for name in names
        )
        assert boundary.backend.to_pyarrow(
            boundary.backend.table("source", database=(expected_project, expected_dataset))
        ).to_pylist() == [{"value": "preserve"}]
        assert boundary.drop_targets[-len(names) :] == [
            (expected_project, expected_dataset, name) for name in names
        ]
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("catalog", "schema", "project", "dataset"),
    [
        (None, None, "data-project", "default_dataset"),
        ("catalog-only", None, "data-project", "default_dataset"),
        (None, "explicit_dataset", "billing-project", "explicit_dataset"),
        ("quoted-project", "explicit_dataset", "quoted-project", "explicit_dataset"),
    ],
)
def test_bigquery_pinned_namespace_publishes_persists_locators_and_deletes(
    monkeypatch: pytest.MonkeyPatch,
    catalog: str | None,
    schema: str | None,
    project: str,
    dataset: str,
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    switched = ("billing-project", "switched_dataset")
    try:
        store = WarehouseArtifactStore(boundary.backend, catalog=catalog, schema_name=schema)
        boundary.backend.set_database(switched[1])
        ref, _ = _publish_in(store, unit_id="namespace-owner")
        assert (ref.manifest.catalog, ref.manifest.schema_name) == (project, dataset)
        physical_database = boundary._ensure_database(project, dataset)
        row = (
            boundary.real.table("ud_manifest_index", database=physical_database)
            .execute()
            .to_dict("records")[0]
        )
        manifest = UnitDayArtifactManifest.model_validate_json(row["manifest_json"])
        relation_names = (
            manifest.base.exposures.relation.name,
            manifest.base.measure_stats.relation.name,
        )
        assert all(
            (relation.catalog, relation.schema_name) == (project, dataset)
            for relation in (
                manifest.base.exposures.relation,
                manifest.base.measure_stats.relation,
            )
        )
        boundary.backend.create_table(
            relation_names[0],
            ibis.memtable([{"value": "unrelated"}]),
            database=switched,
        )

        reopened = WarehouseArtifactStore(boundary.backend, catalog=project, schema_name=dataset)
        assert reopened.visible_manifests == (ref,)
        reopened.drop_generation(ref.artifact_id, ref.generation_id)
        reopened.drop_generation(ref.artifact_id, ref.generation_id)
        assert reopened.visible_manifests == ()
        assert not set(relation_names) & set(
            boundary.backend.list_tables(database=(project, dataset))
        )
        assert boundary.backend.table(relation_names[0], database=switched).execute().to_dict(
            "records"
        ) == [{"value": "unrelated"}]
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("catalog", "schema", "project", "dataset"),
    [
        (None, None, "data-project", "default_dataset"),
        ("catalog-only", None, "data-project", "default_dataset"),
        (None, "explicit_dataset", "billing-project", "explicit_dataset"),
        ("quoted-project", "explicit_dataset", "quoted-project", "explicit_dataset"),
    ],
)
def test_bigquery_post_manifest_abort_invalidates_and_erases(
    monkeypatch: pytest.MonkeyPatch,
    catalog: str | None,
    schema: str | None,
    project: str,
    dataset: str,
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    try:
        store = WarehouseArtifactStore(boundary.backend, catalog=catalog, schema_name=schema)
        names: tuple[str, ...] = ()
        with pytest.raises(RuntimeError, match="abort after manifest"):
            with store.begin_publication(expected_context=_artifact_context()) as publication:
                exposure, measure = _open_exposure_and_measure(publication)
                names = (exposure.relation.name, measure.relation.name)
                body = _manifest_body(
                    _artifact_context(),
                    exposure,
                    measure,
                    artifact_id=publication.artifact_id,
                    generation_id=publication.generation_id,
                )
                publication.publish_manifest(_sealed_manifest(body))
                raise RuntimeError("abort after manifest")
        assert store.visible_manifests == ()
        assert names
        assert not set(names) & set(boundary.backend.list_tables(database=(project, dataset)))
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
def test_bigquery_legacy_ambiguous_receipt_refuses_before_tombstone_or_drop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    original = ("data-project", "default_dataset")
    current = ("billing-project", "switched_dataset")
    try:
        original_store = WarehouseArtifactStore(boundary.backend)
        ref, _ = _publish_in(original_store, unit_id="legacy-owner")
        row = (
            boundary.real.table("ud_manifest_index", database=boundary._ensure_database(*original))
            .execute()
            .to_dict("records")[0]
        )
        manifest = UnitDayArtifactManifest.model_validate_json(row["manifest_json"])
        relation_names = (
            manifest.base.exposures.relation.name,
            manifest.base.measure_stats.relation.name,
        )
        base = manifest.base.model_copy(
            update={
                role: getattr(manifest.base, role).model_copy(
                    update={
                        "relation": getattr(manifest.base, role).relation.model_copy(
                            update={"catalog": None, "schema_name": None}
                        )
                    }
                )
                for role in ("exposures", "measure_stats")
            }
        )
        legacy_unsealed = manifest.model_copy(update={"base": base, "manifest_sha256": ""})
        legacy_manifest = legacy_unsealed.model_copy(
            update={"manifest_sha256": manifest_sha256(legacy_unsealed)}
        )
        row["manifest_json"] = legacy_manifest.model_dump_json()
        row["manifest_sha256"] = legacy_manifest.manifest_sha256

        boundary.backend.set_database(current[1])
        current_store = WarehouseArtifactStore(boundary.backend)
        boundary.backend.create_table(
            relation_names[0],
            ibis.memtable([{"value": "unrelated"}]),
            database=current,
        )
        boundary.backend.insert("ud_manifest_index", ibis.memtable([row]), database=current)

        assert current_store.visible_manifests
        with pytest.raises(ArtifactContractError) as refused:
            current_store.drop_generation(ref.artifact_id, ref.generation_id)
        assert refused.value.code == "artifact.identifier.unsafe"
        assert current_store.visible_manifests
        assert boundary.backend.table(relation_names[0], database=current).execute().to_dict(
            "records"
        ) == [{"value": "unrelated"}]
        assert set(relation_names) <= set(boundary.backend.list_tables(database=original))
        tombstones = boundary.backend.to_pyarrow(
            boundary.backend.table("ud_manifest_dropped", database=current)
        ).to_pylist()
        assert not any(
            entry["artifact_id"] == str(ref.artifact_id)
            and entry["generation_id"] == str(ref.generation_id)
            for entry in tombstones
        )
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("missing_field", ["catalog", "schema_name"])
def test_bigquery_partial_data_locator_refuses_before_snapshot_data_access(
    monkeypatch: pytest.MonkeyPatch, missing_field: str
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    namespace = ("data-project", "default_dataset")
    try:
        publisher = WarehouseArtifactStore(boundary.backend)
        ref, _ = _publish_in(publisher, unit_id="partial-data-locator")
        metadata = boundary.backend.table("ud_manifest_index", database=namespace)
        row = boundary.backend.to_pyarrow(metadata).to_pylist()[0]
        manifest = UnitDayArtifactManifest.model_validate_json(row["manifest_json"])
        exposure = manifest.base.exposures.model_copy(
            update={
                "relation": manifest.base.exposures.relation.model_copy(
                    update={missing_field: None}
                )
            }
        )
        unsealed = manifest.model_copy(
            update={
                "base": manifest.base.model_copy(update={"exposures": exposure}),
                "manifest_sha256": "",
            }
        )
        receipt = unsealed.model_copy(update={"manifest_sha256": manifest_sha256(unsealed)})
        assert receipt.base.exposures.digest_format == receipt.base.measure_stats.digest_format == 4
        assert receipt.base.measure_stats.relation.catalog == namespace[0]
        assert receipt.base.measure_stats.relation.schema_name == namespace[1]
        assert (ref.manifest.catalog, ref.manifest.schema_name) == namespace
        row["manifest_json"] = receipt.model_dump_json()
        row["manifest_sha256"] = receipt.manifest_sha256
        physical = boundary._ensure_database(*namespace)
        boundary.real.raw_sql(f'DELETE FROM "{physical}"."ud_manifest_index"')
        boundary.backend.insert("ud_manifest_index", ibis.memtable([row]), database=namespace)
        trusted = ref.model_copy(update={"manifest_sha256": receipt.manifest_sha256})
        reader = WarehouseArtifactStore(boundary.backend)
        table = boundary.backend.table
        create_table = boundary.backend.create_table
        data_names = {exposure.relation.name, receipt.base.measure_stats.relation.name}
        data_reads = []
        temporary_creates = []
        before_drops = tuple(boundary.drop_targets)

        def record_table(name, *, database=None):
            if name in data_names:
                data_reads.append((name, database))
            return table(name, database=database)

        def record_create(name, obj=None, **kwargs):
            if kwargs.get("temp"):
                temporary_creates.append(name)
            return create_table(name, obj, **kwargs)

        monkeypatch.setattr(boundary.backend, "table", record_table)
        monkeypatch.setattr(boundary.backend, "create_table", record_create)
        with pytest.raises(ArtifactContractError) as refused:
            with reader.open_snapshot(trusted) as snapshot:
                loaded = snapshot.read_manifest(
                    trusted.manifest, expected_sha256=trusted.manifest_sha256
                )
                assert loaded == receipt
                snapshot.verify_relation(loaded.base.exposures, expected_role="exposures")
        assert refused.value.code == "artifact.identifier.unsafe"
        assert dict(refused.value.context) == {
            "namespace": namespace[1],
            "expected_catalog": namespace[0],
            "rejected_catalog": exposure.relation.catalog,
            "rejected_schema": exposure.relation.schema_name,
        }
        assert data_reads == []
        assert temporary_creates == []
        assert tuple(boundary.drop_targets) == before_drops
        assert reader.visible_manifests == (trusted,)
        assert data_names <= set(boundary.backend.list_tables(database=namespace))
        tombstones = boundary.backend.to_pyarrow(
            boundary.backend.table("ud_manifest_dropped", database=namespace)
        ).to_pylist()
        assert tombstones == []
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
def test_bigquery_publication_abort_uses_creation_namespace_after_default_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    original = ("data-project", "default_dataset")
    switched = ("billing-project", "switched_dataset")
    try:
        store = WarehouseArtifactStore(boundary.backend)
        names: tuple[str, ...] = ()
        with pytest.raises(RuntimeError, match="abort publication"):
            with store.begin_publication(expected_context=_artifact_context()) as publication:
                exposure, measure = _open_exposure_and_measure(publication)
                names = (exposure.relation.name, measure.relation.name)
                boundary.backend.set_database(switched[1])
                boundary.backend.create_table(
                    names[0],
                    ibis.memtable([{"value": "unrelated"}]),
                    database=switched,
                )
                raise RuntimeError("abort publication")

        assert names
        assert all(name not in boundary.backend.list_tables(database=original) for name in names)
        assert names[0] in boundary.backend.list_tables(database=switched)
        assert boundary.backend.to_pyarrow(
            boundary.backend.table(names[0], database=switched)
        ).to_pylist() == [{"value": "unrelated"}]
        assert boundary.drop_targets[-len(names) :] == [(*original, name) for name in names]
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
def test_bigquery_missing_default_dataset_refuses_before_metadata_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    boundary.backend.dataset = ""
    try:
        with pytest.raises(CapabilityError) as refused:
            WarehouseArtifactStore(boundary.backend)

        assert refused.value.code == "artifact.store.namespace"
        assert {key: value for key, value in refused.value.context.items() if key != "message"} == {
            "catalog": None,
            "schema": None,
            "table": "ud_manifest_index",
        }
        assert boundary.executed_ddl == []
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
def test_bigquery_publish_and_adopt_never_list_a_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Listing reads every table in a dataset, and the session dataset holds every query result.
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    try:
        store = WarehouseArtifactStore(boundary.backend)
        ref, measure = _publish_in(store, unit_id="looked-up")
        with store.open_snapshot(ref) as snapshot:
            manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
            snapshot.verify_relation(manifest.base.exposures, expected_role="exposures")
            verified = snapshot.verify_relation(measure, expected_role="measure_stats")
            assert snapshot.execute(verified).column("sum_value").to_pylist() == [2.0]
        assert boundary.listed == []
    finally:
        boundary.real.disconnect()


@pytest.mark.integration
@pytest.mark.parametrize("state", ["occupied", "missing", "forbidden"])
def test_bigquery_relation_existence_keeps_the_listing_outcomes(
    monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    boundary = _BigQueryNetworkBoundary(monkeypatch)
    namespace = ("data-project", "default_dataset")
    errors = {
        "missing": boundary.api_errors.NotFound("Not found: Dataset data-project:default_dataset"),
        "forbidden": boundary.api_errors.Forbidden(
            "Access Denied: Dataset data-project:default_dataset"
        ),
    }
    sentinel = [{"sentinel": 314159}]
    try:
        store = WarehouseArtifactStore(boundary.backend)
        with store.begin_publication(expected_context=_artifact_context()) as publication:
            name = (
                f"ud_{publication.artifact_id.hex[:12]}_"
                f"{publication.generation_id.hex[:12]}_exposures"
            )
            source = ibis.table({"sentinel": "int64"}, name="not_materialized")
            if state == "occupied":
                boundary.backend.create_table(name, ibis.memtable(sentinel), database=namespace)
                with pytest.raises(CapabilityError) as raised:
                    publication.write_relation("exposures", source)
                assert raised.value.code == "query.relation.name_occupied"
                assert {
                    key: value for key, value in raised.value.context.items() if key != "route"
                } == {
                    "backend": "bigquery",
                    "catalog": "data-project",
                    "schema": "default_dataset",
                    "name": name,
                }
            else:
                boundary.unavailable[namespace] = errors[state]
                statements = len(boundary.executed_ddl)
                with pytest.raises(type(errors[state])):
                    publication.write_relation("exposures", source)
                assert boundary.executed_ddl[statements:] == []
                boundary.unavailable.clear()
        if state == "occupied":
            stored = boundary.backend.table(name, database=namespace)
            assert boundary.backend.to_pyarrow(stored).to_pylist() == sentinel
    finally:
        boundary.real.disconnect()


class _DuckDBBigQueryResult:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    def to_arrow(self, **_kwargs):
        return self._cursor.to_arrow_table()

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _BigQueryNetworkBoundary:
    """Apply BigQuery SQL to an isolated DuckDB-backed dataset service.

    ``listed`` records every dataset listing. A dataset in ``unavailable`` fails
    every metadata call with its error, as a missing or forbidden dataset does.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sqlglot = pytest.importorskip("sqlglot")
        module = pytest.importorskip("ibis.backends.bigquery")
        self.bigquery = pytest.importorskip("google.cloud.bigquery")
        self.api_errors = pytest.importorskip("google.api_core.exceptions")
        self.backend = module.Backend()
        self.backend.billing_project = "billing-project"
        self.backend.data_project = "data-project"
        self.backend.dataset = "default_dataset"
        # A connected backend's anonymous dataset, which also holds its query results.
        monkeypatch.setattr(
            self.backend,
            "_Backend__session_dataset",
            self.bigquery.DatasetReference("billing-project", "_session_results"),
        )
        monkeypatch.setattr(self.backend, "client", _BigQueryDatasetClient(self), raising=False)
        self.listed: list[tuple[str, str]] = []
        self.unavailable: dict[tuple[str, str], Exception] = {}
        self.real = ibis.duckdb.connect()
        duckdb = pytest.importorskip("duckdb")
        self.real.con.create_function(
            "IEEE_DIVIDE",
            lambda numerator, denominator: numerator / denominator if denominator else float("nan"),
            [duckdb.sqltypes.DOUBLE, duckdb.sqltypes.DOUBLE],
            duckdb.sqltypes.DOUBLE,
        )
        self.initialized: set[str] = set()
        self.executed_ddl: list[str] = []
        self.drop_targets: list[tuple[str, str, str]] = []
        for name, method in {
            "create_database": self.create_database,
            "raw_sql": self.raw_sql,
            "create_table": self.create_table,
            "table": self.table,
            "list_tables": self.list_tables,
            "insert": self.insert,
            "drop_table": self.drop_table,
            "execute": self.real.execute,
            "to_pyarrow": self.to_pyarrow,
        }.items():
            monkeypatch.setattr(self.backend, name, method)

    def to_pyarrow(self, expr):
        return self.raw_sql(self.backend.compile(expr)).to_arrow()

    def create_table(
        self, name, obj=None, *, schema=None, database=None, overwrite=False, **kwargs
    ):
        self.executed_ddl.append("CREATE TABLE")
        if kwargs.get("temp") and obj is not None:
            compiled = self.backend.compile(obj)
            parsed = self.sqlglot.parse_one(compiled, read="bigquery")
            for padding in parsed.find_all(self.sqlglot.exp.Pad):
                padding.set(
                    "expression",
                    self.sqlglot.exp.Cast(
                        this=padding.args["expression"],
                        to=self.sqlglot.exp.DataType.build("INT"),
                    ),
                )
            query = parsed.sql("duckdb")
            self.real.con.execute(f'CREATE TEMP TABLE "{name}" AS {query}')
            return self.real.table(name)
        table = self.sqlglot.parse_one(name, into=self.sqlglot.exp.Table, read="bigquery")
        project, dataset = self._resolve_location(database)
        return self.real.create_table(
            table.name,
            obj,
            schema=schema,
            database=self._ensure_database(table.catalog or project, table.db or dataset),
            overwrite=overwrite,
        )

    def _physical_database(self, project: str, dataset: str) -> str:
        return f"bq_{project.replace('-', '_')}__{dataset}"

    def _resolve_location(self, database: Any) -> tuple[str, str]:
        if isinstance(database, tuple):
            return database
        return self.backend._parse_project_and_dataset(database)

    def _ensure_database(self, project: str, dataset: str) -> str:
        physical = self._physical_database(project, dataset)
        if physical not in self.initialized:
            self.real.create_database(physical)
            self.initialized.add(physical)
        return physical

    def create_database(self, name, *, catalog=None, **kwargs):
        self.executed_ddl.append("CREATE DATABASE")
        return self._ensure_database(catalog or self.backend.billing_project, name)

    def require_available(self, project: str, dataset: str) -> None:
        error = self.unavailable.get((project, dataset))
        if error is not None:
            raise error

    def table(self, name, *, database=None):
        from ibis.common.exceptions import TableNotFound

        project, dataset = self._resolve_location(database)
        try:
            self.require_available(project, dataset)
        except self.api_errors.NotFound as missing:
            # Ibis reports a missing dataset as a missing table.
            raise TableNotFound(name) from missing
        return self.real.table(name, database=self._ensure_database(project, dataset))

    def list_tables(self, *, database=None, **kwargs):
        project, dataset = self._resolve_location(database)
        self.listed.append((project, dataset))
        self.require_available(project, dataset)
        return self.real.list_tables(database=self._ensure_database(project, dataset), **kwargs)

    def insert(self, name, obj, *, database=None, **kwargs):
        project, dataset = self._resolve_location(database)
        return self.real.insert(
            name, obj, database=self._ensure_database(project, dataset), **kwargs
        )

    def drop_table(self, name, *, database=None, force=False):
        project, dataset = self._resolve_location(database)
        return self.real.drop_table(
            name,
            database=self._ensure_database(project, dataset),
            force=force,
        )

    def raw_sql(self, statement):
        query = self.sqlglot.parse_one(statement, read="bigquery")
        if not isinstance(query, (self.sqlglot.exp.Create, self.sqlglot.exp.Drop)):
            self.real.con.execute(query.sql("duckdb"))
            return _DuckDBBigQueryResult(self.real.con)
        target = (
            query.this.this
            if isinstance(query, self.sqlglot.exp.Create)
            else (query.args.get("tables") or [query.this])[0]
        )
        if not target.db:
            if isinstance(query, self.sqlglot.exp.Drop):
                self.real.raw_sql(query.sql("duckdb"))
                return None
            raise RuntimeError("BigQuery query has no default dataset")
        project = target.catalog or self.backend.billing_project
        dataset = target.db
        physical = self._ensure_database(project, dataset)
        if isinstance(query, self.sqlglot.exp.Drop):
            self.drop_targets.append((project, dataset, target.name))
        target.set("catalog", self.sqlglot.exp.to_identifier(physical, quoted=True))
        target.set("db", None)
        cursor = self.real.con.cursor()
        try:
            cursor.execute(query.sql("duckdb"))
        except BaseException:
            cursor.close()
            raise
        if isinstance(query, self.sqlglot.exp.Drop):
            cursor.close()
            return None
        return cursor


class _BigQueryDatasetClient:
    """The ``google.cloud.bigquery.Client`` dataset lookup, served by the boundary."""

    def __init__(self, boundary: _BigQueryNetworkBoundary) -> None:
        self._boundary = boundary

    def get_dataset(self, dataset_ref: str):
        reference = self._boundary.bigquery.DatasetReference.from_string(dataset_ref)
        self._boundary.require_available(reference.project, reference.dataset_id)
        self._boundary._ensure_database(reference.project, reference.dataset_id)
        return reference


def test_locator_refusals_carry_the_rejected_value_in_their_context() -> None:
    import uuid

    import ibis

    from increment.query.artifact_contract import ArtifactContractError
    from increment.semantics.artifact import RelationLocator

    store = WarehouseArtifactStore(ibis.duckdb.connect(), schema_name="artifact")
    artifact_id, generation_id = uuid.uuid4(), uuid.uuid4()
    prefix = f"ud_{artifact_id.hex[:12]}_{generation_id.hex[:12]}_"

    def refusal(locator: RelationLocator, role: str) -> ArtifactContractError:
        with pytest.raises(ArtifactContractError) as raised:
            store.validate_locator(
                locator, artifact_id=artifact_id, generation_id=generation_id, role=role
            )
        return raised.value

    outside = refusal(RelationLocator(schema="elsewhere", name=f"{prefix}exposures"), "exposures")
    assert outside.code == "artifact.identifier.unsafe"
    assert dict(outside.context) == {
        "namespace": "artifact",
        "expected_catalog": None,
        "rejected_catalog": None,
        "rejected_schema": "elsewhere",
    }

    foreign = refusal(RelationLocator(schema="artifact", name="ud_other_exposures"), "exposures")
    assert foreign.code == "artifact.refresh.invalid_ref"
    assert dict(foreign.context) == {"name": "ud_other_exposures"}

    wrong_role = refusal(
        RelationLocator(schema="artifact", name=f"{prefix}exposures"), "measure_stats"
    )
    assert wrong_role.code == "artifact.refresh.invalid_ref"
    assert dict(wrong_role.context) == {"role": "measure_stats"}


def test_bigquery_bounded_fetch_avoids_storage_api(monkeypatch: pytest.MonkeyPatch) -> None:
    import pyarrow as pa

    con, session = _session()
    monkeypatch.setattr(con, "name", "bigquery")
    query_options: dict[str, Any] = {}

    class QueryJob:
        def to_arrow(self, **kwargs: Any) -> pa.Table:
            query_options.update(kwargs)
            return pa.table({"total": [6]})

    monkeypatch.setattr(con, "raw_sql", lambda _sql: QueryJob())

    def unexpected_to_pyarrow(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("BigQuery bounded fetch used the Storage API-backed backend fetch")

    monkeypatch.setattr(con, "to_pyarrow", unexpected_to_pyarrow)
    table = con.table("t")
    result = session.fetch_arrow(table.aggregate(total=table.x.sum()))

    assert result.to_pylist() == [{"total": 6}]
    assert query_options == {"create_bqstorage_client": False, "progress_bar_type": None}
    con.disconnect()
