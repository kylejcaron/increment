"""WarehouseSession: fact-table memoization and temp-table mechanics."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import textwrap
import time
import warnings
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import ibis
import pytest

from increment.errors import IncrementWarning, InvalidRequestError
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
def test_query_boundary_filters_only_ibis_duckdb_arrow_deprecation():
    script = textwrap.dedent(
        """
        import ibis
        from increment import Analysis
        from tests.analysis_factory import _make_event_log_table

        con = ibis.duckdb.connect()
        _make_event_log_table(con)
        analysis = Analysis.from_definitions(
            "new_onboarding_v2", "examples/definitions", con
        )
        analysis.run(metrics=["purchase_rate"])
        analysis.health_check()
        """
    )
    analysis = subprocess.run(
        [sys.executable, "-W", "error::DeprecationWarning", "-c", script],
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


def test_live_postgres_source_sql_does_not_require_temp_privilege():
    class LivePostgres:
        name = "postgres"

        def sql(self, statement, *, dialect):
            assert statement == "SELECT 1"
            assert dialect == "postgres"
            return "live relation"

        def raw_sql(self, _statement):
            pytest.fail("live source access must not check TEMP privilege")

    session = WarehouseSession(
        LivePostgres(),  # type: ignore[ty:invalid-argument-type]
        defs=SimpleNamespace(dialect="postgres"),  # type: ignore[ty:invalid-argument-type]
    )

    assert session.source_sql("SELECT 1") == "live relation"


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


@pytest.mark.integration
def test_snapshot_arrow_cache_skips_results_over_per_result_budget(monkeypatch):
    from increment.query import session as session_module

    monkeypatch.setattr(session_module, "_MAX_ARROW_RESULT_BYTES", 8, raising=False)
    monkeypatch.setattr(session_module, "_MAX_ARROW_CACHE_BYTES", 128, raising=False)
    con, session = _session()
    snapshot = session.pin_sources(("select x from t",))
    expression = snapshot.source_sql("select x from t")
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
        assert first.nbytes > session_module._MAX_ARROW_RESULT_BYTES
        assert second.equals(first)
        assert executions == 2
    finally:
        snapshot.drop_materialized()
        con.disconnect()


@pytest.mark.integration
def test_snapshot_arrow_cache_stops_at_total_byte_budget(monkeypatch):
    from increment.query import session as session_module

    monkeypatch.setattr(session_module, "_MAX_ARROW_RESULT_BYTES", 128, raising=False)
    monkeypatch.setattr(session_module, "_MAX_ARROW_CACHE_BYTES", 9, raising=False)
    con, session = _session()
    snapshot = session.pin_sources(("select x from t",))
    captured = snapshot.source_sql("select x from t")
    small = captured.filter(captured.x == 1).select("x")
    original_to_pyarrow = con.to_pyarrow
    executions = 0

    def counted_to_pyarrow(expr, *args, **kwargs):
        nonlocal executions
        executions += 1
        return original_to_pyarrow(expr, *args, **kwargs)

    monkeypatch.setattr(con, "to_pyarrow", counted_to_pyarrow)
    try:
        first_small = snapshot.fetch_arrow(small)
        assert first_small.nbytes == session_module._MAX_ARROW_CACHE_BYTES
        assert snapshot.fetch_arrow(small).equals(first_small)
        large = snapshot.fetch_arrow(captured)
        assert large.nbytes > session_module._MAX_ARROW_CACHE_BYTES
        assert snapshot.fetch_arrow(captured).equals(large)
        assert executions == 3
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
            con.drop_table(physical_name, database="materialized")
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
    drop_table = con.drop_table

    def unresolved(*args):
        if resolution_raises:
            raise RuntimeError("namespace metadata unavailable")
        return None

    def drop_once(*args, **kwargs):
        nonlocal cleanup_fails_once
        if cleanup_fails_once:
            cleanup_fails_once = False
            raise RuntimeError("temporary cleanup failure")
        return drop_table(*args, **kwargs)

    monkeypatch.setattr(session, "_materialized_relation", unresolved)
    monkeypatch.setattr(con, "drop_table", drop_once)
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

    def drop_table(name, *, database, force):
        expected = "pg_temp" if backend == "postgres" else physical_database
        assert database == expected
        con.drop_table(name, database=physical_database, force=force)

    adapter = SimpleNamespace(
        name=backend,
        current_catalog="memory",
        current_database="materialized",
        create_table=create_table,
        drop_table=drop_table,
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


def test_snowflake_unknown_creation_namespace_does_not_create(monkeypatch):
    con, session = _session()
    expr = con.table("t")
    before = con.list_tables(database=("temp", "main"))
    adapter = SimpleNamespace(name="snowflake", current_catalog="memory", current_database=None)

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
    """Format-4 SQL digesting is one function call (no separate fetch-and-hash
    stages to fail independently), so a single injection point
    stands in for the old fetch/digest split: any exception raised while
    computing the digest must still drop the newly created table and leave
    no visible manifest."""
    from increment.query import session as module

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    before = set(con.list_tables(database="artifacts"))

    def broken(*args, **kwargs):
        raise RuntimeError("stream failure")

    monkeypatch.setattr(module, "digest_relation_sql_v2", broken)
    try:
        with pytest.raises(RuntimeError, match="stream failure"):
            _publish_in(store, unit_id="u")
        assert set(con.list_tables(database="artifacts")) == before
        assert store.visible_manifests == ()
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
    """Format-4 SQL digesting is one function call, so "fetch" and "digest"
    collapse into one injection point (patching digest_relation_sql_v2)."""
    from duckdb import CatalogException

    from increment.query import session as module

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref, measure = _publish_in(store, unit_id="u")
    private_tables = []
    create = con.create_table

    def tracked(*args, **kwargs):
        result = create(*args, **kwargs)
        if kwargs.get("temp"):
            private_tables.append(result)
        return result

    def broken(*args, **kwargs):
        raise RuntimeError("stream failure")

    monkeypatch.setattr(con, "create_table", tracked)
    if failure == "digest":
        monkeypatch.setattr(module, "digest_relation_sql_v2", broken)
    if failure == "mismatch":
        con.raw_sql(f'UPDATE artifacts."{measure.relation.name}" SET sum_value = 99')
    # "digest" replaces digest_relation_sql_v2 outright, so its own scratch
    # temp table (incr_digest_*) is never created; every other case runs the
    # real digest, which creates and drops that scratch table itself, on top
    # of the outer snapshot copy.
    expected_private_tables = 1 if failure == "digest" else 2
    try:
        with pytest.raises((RuntimeError, ArtifactContractError)):
            with store.open_snapshot(ref) as snapshot:
                snapshot.verify_relation(measure, expected_role="measure_stats")
                if failure == "body":
                    raise RuntimeError("body failure")
        assert len(private_tables) == expected_private_tables
        for private in private_tables:
            with pytest.raises(CatalogException):
                con.to_pyarrow(private)
    finally:
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
