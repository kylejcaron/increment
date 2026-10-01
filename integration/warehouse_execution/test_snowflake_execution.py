# integration/warehouse_execution/test_snowflake_execution.py
"""Real Snowflake execution through the actual public Analysis entry
points: a ratio metric, materialized-cache correction, and artifact
publish/adopt parity -- not a hand-assembled query-builder pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from increment import Analysis
from integration.warehouse_execution._suite import (
    _assert_materializations_absent,
    _cache_correction_sql,
    _create_index_table,
    _create_scratch_namespace,
    _drop_scratch_namespace,
    _materialized_cache_fixture,
    _physical_temp_relation,
    _required_materializations,
    _scratch_catalog,
    run_artifact_abort_probe,
    run_artifact_confidentiality_probe,
    run_artifact_explicit_catalog_probe,
    run_artifact_namespace_probe,
    run_artifact_publish_adopt_probe,
    run_artifact_sequential_identity_probe,
    run_materialized_cache_correction_probe,
    run_ratio_metric_probe,
)

pytestmark = pytest.mark.warehouse_snowflake


def test_ratio_metric_lift(snowflake_con):
    run_ratio_metric_probe(snowflake_con, dialect="snowflake")


def test_materialized_cache_reflects_true_correction(snowflake_con, monkeypatch):
    run_materialized_cache_correction_probe(
        snowflake_con, dialect="snowflake", monkeypatch=monkeypatch
    )


def test_artifact_publish_adopt_parity(snowflake_con):
    run_artifact_publish_adopt_probe(snowflake_con, dialect="snowflake")


def test_artifact_output_confidentiality(snowflake_con):
    run_artifact_confidentiality_probe(snowflake_con, dialect="snowflake")


def test_artifact_sequential_identity_matches_native(snowflake_con):
    run_artifact_sequential_identity_probe(snowflake_con, dialect="snowflake")


def test_artifact_manifest_index_namespace_contract(snowflake_con):
    run_artifact_namespace_probe(snowflake_con, dialect="snowflake")


def test_scratch_schema_helpers_qualify_with_the_database():
    class Connection:
        current_catalog = "RELEASE_DB"
        current_database = "RELEASE_SCHEMA"

        def __init__(self):
            self.calls = []

        def create_database(self, name, **kwargs):
            self.calls.append(("create_database", name, kwargs))

        def list_tables(self, **kwargs):
            self.calls.append(("list_tables", kwargs))
            return ["ud_manifest_index"]

        def drop_table(self, name, **kwargs):
            self.calls.append(("drop_table", name, kwargs))

        def drop_database(self, name, **kwargs):
            self.calls.append(("drop_database", name, kwargs))

    con = Connection()
    assert _scratch_catalog(con, "snowflake") == "RELEASE_DB"
    _create_scratch_namespace(con, "snowflake", "ud_probe_x")
    _drop_scratch_namespace(con, "ud_probe_x", catalog="RELEASE_DB")

    scratch = "ud_probe_x"
    assert con.calls == [
        ("create_database", "ud_probe_x", {"catalog": "RELEASE_DB"}),
        ("list_tables", {"database": scratch}),
        ("drop_table", "ud_manifest_index", {"database": scratch, "force": True}),
        ("drop_database", "ud_probe_x", {"catalog": "RELEASE_DB", "force": True}),
    ]


def _drop_created_materializations(drop_table, created_relations) -> None:
    for _, relation in created_relations:
        drop_table(
            relation.op().name,
            database=relation.op().namespace.database,
            force=True,
        )


def _plant_sentinels(create_table, schema, relations) -> set[str]:
    names = {relation.op().name for relation in relations}
    for name in names:
        create_table(
            name,
            obj=[{"sentinel": 314159}],
            database=schema,
            overwrite=False,
        )
    return names


def test_scratch_index_table_uses_a_string_database():
    """Hosted ``create_table`` rejects a (catalog, schema) tuple database."""

    class Connection:
        def __init__(self):
            self.databases = []

        def create_table(self, name, **kwargs):
            self.databases.append(kwargs["database"])

    con = Connection()
    _create_index_table(con, "ud_probe_x", {"artifact_id": "string"})
    assert con.databases == ["ud_probe_x"]


def _run_changed_default_namespace_probe(
    con,
    monkeypatch,
    sentinel_schema: str,
    *,
    cleanup: str,
    force_bare_name_cleanup: bool,
) -> set[str]:
    """Keep permanent sentinels separate from the owned TEMP namespace."""
    original_catalog = con.current_catalog
    original_schema = con.current_database
    con.create_database(sentinel_schema, catalog=original_catalog)
    assert con.current_database == original_schema

    yaml, rows = _materialized_cache_fixture(con, "snowflake")
    con.create_table("events_cache_smoke", obj=rows, overwrite=True)

    created_relations: list[tuple[str, object]] = []
    sentinel_names: set[str] = set()
    create_table = con.create_table
    drop_table = con.drop_table

    def recording_create_table(name, *args, **kwargs):
        relation = create_table(name, *args, **kwargs)
        if kwargs.get("temp") is True:
            created_relations.append((name, _physical_temp_relation(con, relation, "snowflake")))
        return relation

    def bare_name_drop(name, *args, **kwargs):
        if name.startswith(("exp_spine_", "exp_stats_revenue_")):
            return drop_table(name, force=True)
        return drop_table(name, *args, **kwargs)

    monkeypatch.setattr(con, "create_table", recording_create_table)
    if force_bare_name_cleanup:
        monkeypatch.setattr(con, "drop_table", bare_name_drop)

    analysis = None
    try:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "defs.yaml"
            path.write_text(yaml)
            analysis = Analysis.from_definitions("exp", path, con, store="always")
            before = analysis.run()[0].lift.value
            first_materializations = _required_materializations(con, created_relations)
            first_creation_count = len(created_relations)

            con.raw_sql(_cache_correction_sql(con, "snowflake"))
            if cleanup == "invalidation":
                sentinel_names.update(
                    _plant_sentinels(create_table, sentinel_schema, first_materializations)
                )
                fixture_rows = con.table("events_cache_smoke").to_pyarrow().to_pylist()
                create_table(
                    "events_cache_smoke",
                    obj=fixture_rows,
                    database=sentinel_schema,
                    overwrite=True,
                )
                with con.raw_sql(f'USE SCHEMA "{original_catalog}"."{sentinel_schema}"'):
                    pass

            cached = analysis.run()[0].lift.value
            second_materializations = _required_materializations(
                con, created_relations[first_creation_count:]
            )
            _assert_materializations_absent(con, first_materializations)

            with con.raw_sql(f'USE SCHEMA "{original_catalog}"."{original_schema}"'):
                pass
            current = Analysis.from_definitions("exp", path, con, store="none").run()[0].lift.value

            if cleanup == "close":
                sentinel_names.update(
                    _plant_sentinels(create_table, sentinel_schema, second_materializations)
                )
                with con.raw_sql(f'USE SCHEMA "{original_catalog}"."{sentinel_schema}"'):
                    pass

            assert abs(before - (2 / 11)) < 1e-9
            assert abs(cached - (15 / 11)) < 1e-9
            assert abs(current - (15 / 11)) < 1e-9

            analysis.close()
            _assert_materializations_absent(con, second_materializations)
    finally:
        if analysis is not None:
            analysis.close()
        with con.raw_sql(f'USE SCHEMA "{original_catalog}"."{original_schema}"'):
            pass
        _drop_created_materializations(drop_table, created_relations)
        drop_table("events_cache_smoke", force=True)

    return sentinel_names


@pytest.mark.parametrize("cleanup", ["invalidation", "close"])
def test_changed_default_namespace_sentinel_survives_cleanup(snowflake_con, monkeypatch, cleanup):
    con = snowflake_con
    sentinel_schema = f"cleanup_probe_{uuid4().hex}"
    try:
        sentinel_names = _run_changed_default_namespace_probe(
            con,
            monkeypatch,
            sentinel_schema,
            cleanup=cleanup,
            force_bare_name_cleanup=False,
        )
        assert sentinel_names
        for name in sentinel_names:
            rows = con.table(name, database=sentinel_schema).to_pyarrow().to_pylist()
            assert rows == [{"sentinel": 314159}]
    finally:
        con.drop_database(sentinel_schema, catalog=con.current_catalog, force=True)


@pytest.mark.parametrize("cleanup", ["invalidation", "close"])
def test_changed_default_namespace_bare_name_cleanup_mutation(snowflake_con, monkeypatch, cleanup):
    con = snowflake_con
    sentinel_schema = f"cleanup_probe_{uuid4().hex}"
    try:
        with pytest.raises(AssertionError, match="materialization leaked"):
            _run_changed_default_namespace_probe(
                con,
                monkeypatch,
                sentinel_schema,
                cleanup=cleanup,
                force_bare_name_cleanup=True,
            )
    finally:
        con.drop_database(sentinel_schema, catalog=con.current_catalog, force=True)


def test_artifact_abort_after_manifest_leaves_nothing_visible(snowflake_con):
    run_artifact_abort_probe(snowflake_con, dialect="snowflake")


def test_artifact_explicit_catalog_path(snowflake_con):
    """The hosted explicit-catalog path is unverified live; it needs Snowflake credentials."""
    run_artifact_explicit_catalog_probe(snowflake_con, dialect="snowflake")
