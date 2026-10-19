# integration/warehouse_execution/test_bigquery_execution.py
"""Real BigQuery execution through the actual public Analysis entry
points: a ratio metric, materialized-cache correction, and artifact
publish/adopt parity -- not a hand-assembled query-builder pipeline."""

from __future__ import annotations

import pytest

from integration.warehouse_execution._suite import (
    _create_index_table,
    _create_scratch_namespace,
    _drop_scratch_namespace,
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

pytestmark = pytest.mark.warehouse_bigquery


def test_bigquery_format4_digest_matches_duckdb(bigquery_con, monkeypatch):
    import math
    from io import BytesIO
    from uuid import uuid4

    import ibis
    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import bigquery

    from increment.query import artifact_digest
    from increment.query.artifact_digest import digest_relation_sql_v2
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
    table_name = f"digest_spec_{uuid4().hex[:10]}"
    try:
        expected = digest_relation_sql_v2(
            duckdb,
            duckdb.create_table("digest_spec", arrow),
            "portable_spec",
            schema,
            primary_key=("id",),
        )
        parquet = BytesIO()
        pq.write_table(arrow, parquet)
        parquet.seek(0)
        destination = f"{bigquery_con.current_catalog}.{bigquery_con.current_database}.{table_name}"
        job = bigquery_con.client.load_table_from_file(
            parquet,
            destination,
            size=parquet.getbuffer().nbytes,
            job_config=bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET),
        )
        job.result()
        table = bigquery_con.sql(
            f"SELECT CAST(`id` AS INT64) AS `id`, `text`, "
            f"CAST(`integer` AS INT64) AS `integer`, `float`, `date`, "
            f"`timestamp`, `boolean` "
            f"FROM `{destination}`"
        )
        fetched_rows = bigquery_con.to_pyarrow(
            table.select(id=table.id, integer=table.integer, float=table.float)
        ).to_pylist()
        fetched_rows.sort(key=lambda row: row["id"])
        assert [(row["id"], row["integer"]) for row in fetched_rows] == [
            (row[0], row[2]) for row in rows
        ]
        assert fetched_rows[6]["float"] is not None
        assert math.isnan(fetched_rows[6]["float"])
        assert fetched_rows[9]["float"] is None
        monkeypatch.setattr(
            artifact_digest, "_SQL_DIGEST_BACKENDS", frozenset({"duckdb", "bigquery"})
        )
        actual = digest_relation_sql_v2(
            bigquery_con, table, "portable_spec", schema, primary_key=("id",)
        )
        assert (
            expected.content_sha256
            == "28b8c5135e6da8dcddeca6854a0bb9cc86a205eee4f6b3e26795589124e669e5"
        )
        assert actual.content_sha256 == expected.content_sha256
    finally:
        duckdb.disconnect()
        bigquery_con.drop_table(table_name, force=True)


def test_bigquery_float64_token_matches_pinned_python_binary64(bigquery_con):
    from integration.warehouse_execution._float_tokens import run_float_token_probe

    run_float_token_probe(bigquery_con, dialect="bigquery")


def test_artifact_cleanup_qualifies_bigquery_metadata_tables():
    class Connection:
        current_catalog = "release-project"
        current_database = "release_dataset"

        def __init__(self):
            self.statements = []

        def raw_sql(self, statement):
            self.statements.append(statement)

    class Store:
        def drop_generation(self, artifact_id, generation_id):
            assert (artifact_id, generation_id) == ("artifact", "generation")

    ref = type("Ref", (), {"artifact_id": "artifact", "generation_id": "generation"})()
    con = Connection()

    from integration.warehouse_execution._suite import _drop_probe_generation

    _drop_probe_generation(con, Store(), ref, dialect="bigquery")

    assert [statement.split(" WHERE ", 1)[0] for statement in con.statements] == [
        "DELETE FROM `release-project`.`release_dataset`.`ud_manifest_index`",
        "DELETE FROM `release-project`.`release_dataset`.`ud_manifest_dropped`",
    ]


def test_scratch_dataset_helpers_qualify_with_the_project():
    class Connection:
        current_catalog = "release-project"
        current_database = "release_dataset"

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
    assert _scratch_catalog(con, "bigquery") == "release-project"
    _create_scratch_namespace(con, "bigquery", "ud_probe_x")
    _drop_scratch_namespace(con, "ud_probe_x", catalog="release-project")

    scratch = "ud_probe_x"
    assert con.calls == [
        ("create_database", "ud_probe_x", {"catalog": "release-project"}),
        ("list_tables", {"database": scratch}),
        ("drop_table", "ud_manifest_index", {"database": scratch, "force": True}),
        ("drop_database", "ud_probe_x", {"catalog": "release-project", "force": True}),
    ]


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


def test_ratio_metric_lift(bigquery_con):
    run_ratio_metric_probe(bigquery_con, dialect="bigquery")


def test_materialized_cache_reflects_true_correction(bigquery_con, monkeypatch):
    run_materialized_cache_correction_probe(
        bigquery_con, dialect="bigquery", monkeypatch=monkeypatch
    )


def test_artifact_publish_adopt_parity(bigquery_con):
    run_artifact_publish_adopt_probe(bigquery_con, dialect="bigquery")


def test_artifact_output_confidentiality(bigquery_con):
    run_artifact_confidentiality_probe(bigquery_con, dialect="bigquery")


def test_artifact_sequential_identity_matches_native(bigquery_con):
    run_artifact_sequential_identity_probe(bigquery_con, dialect="bigquery")


def test_artifact_manifest_index_namespace_contract(bigquery_con):
    run_artifact_namespace_probe(bigquery_con, dialect="bigquery")


def test_artifact_abort_after_manifest_leaves_nothing_visible(bigquery_con):
    run_artifact_abort_probe(bigquery_con, dialect="bigquery")


def test_artifact_explicit_catalog_path(bigquery_con):
    """Explicit project/dataset publication, fresh adoption and cleanup work on BigQuery."""
    run_artifact_explicit_catalog_probe(bigquery_con, dialect="bigquery")
