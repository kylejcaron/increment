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
    table_name = f"digest_spec_{uuid4().hex[:10]}"
    try:
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
            f"CAST(`timestamp` AS TIMESTAMP) AS `timestamp`, `boolean`, "
            f"IEEE_DIVIDE(1.0, `float`) AS `float_reciprocal` "
            f"FROM `{destination}`"
        )
        fetched_arrow = bigquery_con.raw_sql(bigquery_con.compile(table)).to_arrow(
            create_bqstorage_client=False, progress_bar_type=None
        )
        fetched_rows = fetched_arrow.select(
            ["id", "integer", "float", "float_reciprocal"]
        ).to_pylist()
        fetched_rows.sort(key=lambda row: row["id"])
        assert [(row["id"], row["integer"]) for row in fetched_rows] == [
            (row[0], row[2]) for row in rows
        ]
        assert fetched_rows[0]["float"] == 0.0
        assert fetched_rows[0]["float_reciprocal"] == float("inf")
        assert fetched_rows[6]["float"] is not None
        assert math.isnan(fetched_rows[6]["float"])
        stored_arrow = fetched_arrow.select([name for name, _type, _nullable in schema])
        duck_table = duckdb.create_table("digest_spec_stored", stored_arrow)
        expected = digest_relation_sql_v2(
            duckdb,
            duck_table,
            "portable_spec",
            schema,
            primary_key=("id",),
        )
        monkeypatch.setattr(
            artifact_digest, "_SQL_DIGEST_BACKENDS", frozenset({"duckdb", "bigquery"})
        )
        actual = digest_relation_sql_v2(
            bigquery_con, table, "portable_spec", schema, primary_key=("id",)
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

        def diagnostic_rows(con, expr, dialect):
            if dialect == "bigquery":
                return (
                    con.raw_sql(con.compile(expr))
                    .to_arrow(create_bqstorage_client=False, progress_bar_type=None)
                    .to_pylist()
                )
            return con.to_pyarrow(expr).to_pylist()

        def field_tokens(con, relation, dialect):
            tokens = {
                f"token_{name}": _row_token_expr(relation, ((name, tag, kind),), dialect=dialect)
                for name, tag, kind in tagged_fields
            }
            return sorted(
                diagnostic_rows(con, relation.select(id=relation.id, **tokens), dialect),
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
            row_stages = diagnostic_rows(con, hashes.order_by("id"), dialect)
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
                "buckets": diagnostic_rows(con, bucket_rows, dialect),
            }

        duckdb_tokens = field_tokens(duckdb, duck_table, "duckdb")
        bigquery_tokens = field_tokens(bigquery_con, table, "bigquery")
        duckdb_stages = digest_stages(duckdb, duck_table, "duckdb")
        bigquery_stages = digest_stages(bigquery_con, table, "bigquery")
        assert duckdb_tokens == bigquery_tokens, (
            f"format-4 field-token mismatch: DuckDB={duckdb_tokens!r}; BigQuery={bigquery_tokens!r}"
        )
        assert actual.content_sha256 == expected.content_sha256, (
            "format-4 digest mismatch after stage diagnostics; "
            f"DuckDB={expected.content_sha256}, BigQuery={actual.content_sha256}; "
            f"DuckDB stages={duckdb_stages!r}; BigQuery stages={bigquery_stages!r}"
        )
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


@pytest.mark.timeout(600, method="signal")
def test_artifact_sequential_identity_matches_native(bigquery_con):
    # The three sequential publish/adopt/readout journeys each issue many BigQuery jobs.
    run_artifact_sequential_identity_probe(bigquery_con, dialect="bigquery")


def test_artifact_manifest_index_namespace_contract(bigquery_con):
    run_artifact_namespace_probe(bigquery_con, dialect="bigquery")


def test_artifact_abort_after_manifest_leaves_nothing_visible(bigquery_con):
    run_artifact_abort_probe(bigquery_con, dialect="bigquery")


def test_artifact_explicit_catalog_path(bigquery_con):
    """Explicit project/dataset publication, fresh adoption and cleanup work on BigQuery."""
    run_artifact_explicit_catalog_probe(bigquery_con, dialect="bigquery")


@pytest.fixture(scope="module")
def _bigquery_mixed_cuped_probe(bigquery_con, tmp_path_factory):
    from integration.warehouse_execution.test_snowflake_execution import (
        _run_cloud_mixed_cuped_probe,
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        return _run_cloud_mixed_cuped_probe(
            bigquery_con,
            "bigquery",
            tmp_path_factory.mktemp("bigquery-mixed-cuped"),
            monkeypatch,
        )


# Prior BigQuery readouts took up to 501s per case; allow 1800s for this
# module-scoped fixture's shared native, adopted, and refusal readouts.
@pytest.mark.timeout(1800, method="signal")
def test_bigquery_mixed_cuped_matches_local_with_native_and_artifact(
    _bigquery_mixed_cuped_probe,
):
    assert _bigquery_mixed_cuped_probe is None
