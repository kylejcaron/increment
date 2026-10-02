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
