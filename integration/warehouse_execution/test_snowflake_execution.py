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


def test_snowflake_format4_digest_matches_duckdb(snowflake_con, monkeypatch):
    import ibis
    import pyarrow as pa

    from increment.query import artifact_digest
    from increment.query.artifact_digest import (
        _BUCKET_HEX_LEN,
        DigestType,
        _fields,
        _row_token_expr,
        _sql_binary_hex_expr,
        _sql_digest_sha256_binary_expr,
        _sql_digest_sha256_expr,
        digest_relation_sql_v2,
    )
    from tests.test_unit_day_artifact_digest_v2 import _PORTABLE_SCHEMA, _portable_digest_rows

    schema = _PORTABLE_SCHEMA
    rows = _portable_digest_rows()
    arrow = pa.Table.from_arrays(
        [
            pa.array([row[index] for row in rows], type=field_type)
            for index, field_type in enumerate(
                (
                    pa.int64(),
                    pa.string(),
                    pa.int64(),
                    pa.float64(),
                    pa.date32(),
                    pa.timestamp("us", tz="UTC"),
                    pa.bool_(),
                )
            )
        ],
        names=[name for name, _type, _nullable in schema],
    )
    duckdb = ibis.duckdb.connect()
    pg_name = f"digest_spec_{uuid4().hex[:10]}"
    try:
        duck_table = duckdb.create_table("digest_spec", arrow)
        expected = digest_relation_sql_v2(
            duckdb,
            duck_table,
            "portable_spec",
            schema,
            primary_key=("id",),
        )
        snowflake_con.create_table(pg_name, arrow)
        table = snowflake_con.sql(
            f'SELECT CAST("id" AS BIGINT) AS "id", "text", '
            f'CAST("integer" AS BIGINT) AS "integer", "float", "date", '
            f'"timestamp", "boolean" FROM "{pg_name}"'
        )
        fetched_integers = snowflake_con.to_pyarrow(
            table.select(id=table.id, integer=table.integer)
        ).to_pylist()
        fetched_integers.sort(key=lambda row: row["id"])
        assert [(row["id"], row["integer"]) for row in fetched_integers] == [
            (row[0], row[2]) for row in rows
        ]
        monkeypatch.setattr(
            artifact_digest, "_SQL_DIGEST_BACKENDS", frozenset({"duckdb", "snowflake"})
        )
        actual = digest_relation_sql_v2(
            snowflake_con, table, "portable_spec", schema, primary_key=("id",)
        )
        assert (
            expected.content_sha256
            == "28b8c5135e6da8dcddeca6854a0bb9cc86a205eee4f6b3e26795589124e669e5"
        )
        kinds = {
            DigestType.DATE: "date",
            DigestType.FLOAT64: "float",
            DigestType.TIMESTAMP_UTC_US: "timestamp",
        }
        tagged_fields = [
            (field.name, int(field.type_tag), kinds.get(field.type_tag, "other"))
            for field in _fields(schema)
        ]

        def field_tokens(con, relation, dialect):
            tokens = {
                f"token_{name}": _row_token_expr(relation, ((name, tag, kind),), dialect=dialect)
                for name, tag, kind in tagged_fields
            }
            return sorted(
                con.to_pyarrow(relation.select(id=relation.id, **tokens)).to_pylist(),
                key=lambda row: row["id"],
            )

        def digest_stages(con, relation, dialect):
            tokens = relation.select(
                id=relation.id,
                token=_row_token_expr(relation, tagged_fields, dialect=dialect),
            )
            hashes = tokens.select(
                id=tokens.id,
                token=tokens.token,
                row_hash=_sql_binary_hex_expr(
                    _sql_digest_sha256_binary_expr(tokens.token, dialect=dialect),
                    dialect=dialect,
                ),
            )
            hashes = hashes.select(
                id=hashes.id,
                token=hashes.token,
                row_hash=hashes.row_hash,
                bucket=hashes.row_hash.substr(0, _BUCKET_HEX_LEN),
            )
            row_stages = con.to_pyarrow(hashes.order_by("id")).to_pylist()
            bucket_input = hashes.select(bucket=hashes.bucket, h=hashes.row_hash)
            bucket_rows = (
                bucket_input.group_by("bucket")
                .aggregate(
                    bucket_digest=_sql_digest_sha256_expr(
                        bucket_input.h.group_concat(
                            sep="",
                            order_by=ibis.asc(bucket_input.h, nulls_first=dialect == "bigquery"),
                        ),
                        dialect=dialect,
                    ),
                    n=bucket_input.count(),
                )
                .order_by("bucket")
            )
            return {
                "rows": row_stages,
                "buckets": con.to_pyarrow(bucket_rows).to_pylist(),
            }

        duckdb_tokens = field_tokens(duckdb, duck_table, "duckdb")
        snowflake_tokens = field_tokens(snowflake_con, table, "snowflake")
        duckdb_stages = digest_stages(duckdb, duck_table, "duckdb")
        snowflake_stages = digest_stages(snowflake_con, table, "snowflake")
        assert duckdb_tokens == snowflake_tokens, (
            "format-4 field-token mismatch: "
            f"DuckDB={duckdb_tokens!r}; Snowflake={snowflake_tokens!r}"
        )
        assert actual.content_sha256 == expected.content_sha256, (
            "format-4 digest mismatch after stage diagnostics; "
            f"DuckDB={expected.content_sha256}, Snowflake={actual.content_sha256}; "
            f"DuckDB stages={duckdb_stages!r}; Snowflake stages={snowflake_stages!r}"
        )
    finally:
        duckdb.disconnect()
        snowflake_con.drop_table(pg_name, force=True)


def test_snowflake_float64_token_matches_pinned_python_binary64(snowflake_con):
    from integration.warehouse_execution._float_tokens import run_float_token_probe

    run_float_token_probe(snowflake_con, dialect="snowflake")


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
    """Explicit catalog/schema publication, fresh adoption and cleanup work on Snowflake."""
    run_artifact_explicit_catalog_probe(snowflake_con, dialect="snowflake")
