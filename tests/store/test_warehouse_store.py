from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import ibis
import pandas as pd
import pyarrow as pa
import pytest

from increment import from_frame, run
from increment.errors import CapabilityError, CodedError, InvalidRequestError
from increment.kernel.contracts import View
from increment.query.session import WarehouseArtifactStore
from increment.store import HistorySelector, Run, WarehouseResultsStore
from increment.store.types import run_record_key
from tests.analysis_factory import view_readout


def _warehouse_run(
    rows,
    *,
    run_id="one",
    experiment_id="exp",
    analysis_spec_hash="spec",
    committed_day=2,
):
    scope = View(view="run", grain="total")
    cells = tuple(
        replace(
            cell,
            identity=cell.identity.model_copy(
                update={
                    "experiment_id": experiment_id,
                    "analysis_spec_hash": analysis_spec_hash,
                    "metric_hash": "metric",
                    "estimator_id": "t",
                    "method_config_hash": "config",
                    "method_instance_id": "instance",
                    "scope": scope,
                }
            ),
            provenance=replace(
                cell.provenance,
                run_id=run_id,
                execution_identity_hash="exec",
            ),
        )
        for cell in rows.as_cells(created_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    return Run(
        experiment_id,
        run_id,
        analysis_spec_hash,
        "exec",
        None,
        tuple(cell.identity for cell in cells),
        cells,
        rows.metadata,
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, committed_day, tzinfo=UTC),
        scope_analysis_hashes=(analysis_spec_hash,),
        cell_scope_hashes=tuple(
            (cell.identity.hash(), cell.identity.analysis_spec_hash) for cell in cells
        ),
    )


def _committed_name(experiment_id, analysis_spec_hash, run_id):
    """The 63-byte table name of a committed run: the first 49 hex digits of its key."""
    return "increment_run_" + run_record_key(experiment_id, analysis_spec_hash, run_id)[:49]


@pytest.mark.integration
def test_analysis_run_persists_public_cells_without_identity_rewrites():
    backend = ibis.duckdb.connect()
    experiment, resolver = from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "value": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        experiment_id="exp",
    )
    try:
        backend.create_database("analytics")
        store = WarehouseResultsStore(backend, schema="analytics")
        result = run(experiment, resolver, results_store=store)
        assert result.metadata is not None
        assert result.run.analysis_spec_hash is not None
        assert result.run.run_id is not None
        handle = store.open_run("exp", result.run.analysis_spec_hash, result.run.run_id)
        cells = handle.envelope().cells
        assert cells
        assert all(cell.identity.source_snapshot_id for cell in cells)
        assert all(cell.identity.source_content_digest for cell in cells)
        assert all(
            cell.identity.source_content_digest == cell.provenance.source_content_digest
            for cell in cells
        )
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_results_store_creates_its_schema():
    backend = ibis.duckdb.connect()
    experiment, reader = from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "value": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        experiment_id="exp",
    )
    try:
        result = run(
            experiment, reader, results_store=WarehouseResultsStore(backend, schema="results")
        )
        handle = WarehouseResultsStore(backend, schema="results").open_run(
            "exp", result.run.analysis_spec_hash, result.run.run_id
        )
        saved = sorted(cell.identity.hash() for cell in result.run.cells)
        assert saved
        assert sorted(cell.identity.hash() for cell in handle.envelope().cells) == saved
    finally:
        reader.close()
        backend.disconnect()


@pytest.mark.parametrize(
    ("backend", "schema", "limit"),
    [
        ("postgres", "r" * 63, None),
        ("postgres", "r" * 64, 63),
        ("postgres", "é" * 32, 63),
        ("postgres", "analytics." + "r" * 64, 63),
        ("snowflake", "é" * 255, None),
        ("snowflake", "r" * 256, 255),
        ("snowflake", "d" * 255 + "." + "s" * 255, None),
        ("bigquery", "r" * 1024, None),
        ("bigquery", "r" * 1025, 1024),
        ("duckdb", "r" * 10_000, None),
    ],
    ids=[
        "postgres-63-bytes",
        "postgres-64-bytes",
        "postgres-64-bytes-32-characters",
        "postgres-catalog-schema",
        "snowflake-255-characters",
        "snowflake-256-characters",
        "snowflake-catalog-and-schema-each-255",
        "bigquery-1024-bytes",
        "bigquery-1025-bytes",
        "duckdb-unlimited",
    ],
)
def test_store_refuses_a_schema_name_over_the_backend_limit(backend, schema, limit):
    connection = SimpleNamespace(name=backend)
    if limit is None:
        assert WarehouseResultsStore(connection, schema=schema).schema == schema
        return
    with pytest.raises(InvalidRequestError) as refused:
        WarehouseResultsStore(connection, schema=schema)
    assert refused.value.code == "store.schema_name_too_long"
    assert refused.value.context == {
        "store": "WarehouseResultsStore",
        "schema": schema,
        "backend": backend,
        "limit": limit,
    }


def _no_table_operations(*args, **kwargs):
    raise AssertionError("a store whose schema does not exist has no tables to read or write")


def _artifact_backend(name):
    """A backend where no schema exists, so a store within the limit only looks its schema up."""
    connection: Any = SimpleNamespace(
        name=name,
        list_databases=lambda catalog=None: [],
        **dict.fromkeys(
            ("to_pyarrow", "execute", "create_table", "table", "drop_table", "insert"),
            _no_table_operations,
        ),
    )
    return connection


@pytest.mark.parametrize(
    ("backend", "catalog", "schema", "limit"),
    [
        ("postgres", None, "r" * 63, None),
        ("postgres", None, "r" * 64, 63),
        ("postgres", None, "é" * 32, 63),
        ("postgres", "c" * 64, "artifact", 63),
        ("snowflake", None, "r" * 255, None),
        ("snowflake", None, "r" * 256, 255),
        ("snowflake", "d" * 255, "s" * 255, None),
        ("bigquery", "project", "r" * 1025, 1024),
    ],
    ids=[
        "postgres-63-bytes",
        "postgres-64-bytes",
        "postgres-64-bytes-32-characters",
        "postgres-catalog-64-bytes",
        "snowflake-255-characters",
        "snowflake-256-characters",
        "snowflake-catalog-and-schema-each-255",
        "bigquery-1025-bytes",
    ],
)
def test_artifact_store_refuses_a_schema_name_over_the_backend_limit(
    backend, catalog, schema, limit
):
    connection = _artifact_backend(backend)
    if limit is None:
        WarehouseArtifactStore(connection, catalog=catalog, schema_name=schema)
        return
    with pytest.raises(InvalidRequestError) as refused:
        WarehouseArtifactStore(connection, catalog=catalog, schema_name=schema)
    assert refused.value.code == "store.schema_name_too_long"
    assert refused.value.context == {
        "store": "WarehouseArtifactStore",
        "schema": schema if catalog is None else f"{catalog}.{schema}",
        "backend": backend,
        "limit": limit,
    }


def test_artifact_store_refuses_a_name_over_255_characters_as_an_invalid_namespace():
    # Within BigQuery's 1,024 bytes, but past the 255 characters an artifact identifier allows.
    with pytest.raises(CapabilityError) as refused:
        WarehouseArtifactStore(
            _artifact_backend("bigquery"), catalog="project", schema_name="r" * 1024
        )
    assert refused.value.code == "artifact.store.namespace"


@pytest.mark.integration
def test_run_table_names_fit_63_bytes():
    backend = ibis.duckdb.connect()
    experiment, resolver = _analysis()
    try:
        store = WarehouseResultsStore(backend, schema="results")
        store.put(_warehouse_run(view_readout(experiment, resolver)))
        assert backend.list_tables(database="results") == [_committed_name("exp", "spec", "one")]
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_results_store_refuses_a_schema_it_can_neither_find_nor_create(monkeypatch):
    backend = ibis.duckdb.connect()
    backend.raw_sql("ATTACH ':memory:' AS other")
    backend.raw_sql("CREATE SCHEMA other.results")

    def deny_create(*args, **kwargs):
        raise RuntimeError("permission denied to create schema")

    monkeypatch.setattr(backend, "create_database", deny_create)
    experiment, resolver = _analysis()
    try:
        store = WarehouseResultsStore(backend, schema="results")
        with pytest.raises(CodedError) as refused:
            store.put(_warehouse_run(view_readout(experiment, resolver)))
        assert refused.value.code == "store.run.schema_unavailable"
        assert refused.value.context["schema"] == "results"
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_results_store_saves_into_a_schema_of_another_catalog():
    backend = ibis.duckdb.connect()
    backend.raw_sql("ATTACH ':memory:' AS other")
    backend.raw_sql("CREATE SCHEMA other.results")
    experiment, resolver = _analysis()
    try:
        saved = _warehouse_run(view_readout(experiment, resolver))
        store = WarehouseResultsStore(backend, schema="other.results")
        store.put(saved)
        assert store.open_run("exp", "spec", "one").run.manifest_hash == saved.manifest_hash
        assert "other.results" not in backend.list_databases()
    finally:
        resolver.close()
        backend.disconnect()


class _PostgresSchemaNames:
    """DuckDB tables behind PostgreSQL's schema naming.

    Schema creation builds the statement Ibis's PostgreSQL backend builds, then
    folds its unquoted identifiers to lower case, as PostgreSQL does.
    """

    name = "postgres"

    def __init__(self, real):
        self._real = real

    def __getattr__(self, attribute):
        return getattr(self._real, attribute)

    def create_database(self, name, /, *, catalog=None, force=False):
        from sqlglot import expressions as sge

        statement = sge.Create(kind="SCHEMA", this=sge.table_(name, catalog=catalog), exists=force)
        for identifier in statement.find_all(sge.Identifier):
            if not identifier.quoted:
                identifier.set("this", identifier.this.lower())
        self._real.raw_sql(statement.sql("duckdb"))


@pytest.mark.integration
def test_results_store_reopens_a_mixed_case_schema_it_created():
    real = ibis.duckdb.connect()
    backend = _PostgresSchemaNames(real)
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        WarehouseResultsStore(backend, schema="Results").put(_warehouse_run(rows))
        later = WarehouseResultsStore(backend, schema="Results")
        later.put(_warehouse_run(rows, run_id="two", committed_day=3))
        assert later.open_run("exp", "spec", "one").run.run_id == "one"
    finally:
        resolver.close()
        real.disconnect()


@pytest.mark.integration
def test_results_store_reads_a_missing_schema_without_creating_it():
    backend = ibis.duckdb.connect()
    try:
        store = WarehouseResultsStore(backend, schema="typo")
        with pytest.raises(CodedError) as missing:
            store.open_run("exp", "spec", "one")
        assert missing.value.code == "store.run.not_found"
        selector = HistorySelector(
            metric_hash="metric",
            estimator_id="t",
            method_config_hash="config",
            method_instance_id="instance",
            scope_hash=View(view="run", grain="total").hash(),
        )
        assert store.history("exp", selector).num_rows == 0
        assert "typo" not in backend.list_databases()
    finally:
        backend.disconnect()


class _SchemaLookupError(RuntimeError):
    pass


def _listing_backend(error: Exception, *, lists_schemas: bool) -> Any:
    """A Snowflake-shaped backend whose table listing in ``absent_schema`` raises ``error``."""

    def list_tables(database=None):
        assert database == "absent_schema"
        raise error

    backend: Any = SimpleNamespace(
        name="snowflake",
        list_tables=list_tables,
        create_database=_no_table_operations,
        **dict.fromkeys(("table", "create_table"), _no_table_operations),
    )
    if lists_schemas:
        backend.list_databases = lambda catalog=None: ["analytics"]
    return backend


def _history_selector() -> HistorySelector:
    return HistorySelector(
        metric_hash="metric",
        estimator_id="t",
        method_config_hash="config",
        method_instance_id="instance",
        scope_hash=View(view="run", grain="total").hash(),
    )


@pytest.mark.parametrize("lists_schemas", [True, False], ids=["schema-listing", "no-listing"])
def test_missing_schema_run_lookup_is_not_found_and_history_is_empty(lists_schemas):
    backend = _listing_backend(
        _SchemaLookupError("Schema ABSENT_SCHEMA does not exist"), lists_schemas=lists_schemas
    )
    store = WarehouseResultsStore(backend, schema="absent_schema")

    for run_id in ("run-id", None):
        with pytest.raises(CodedError) as missing:
            store.open_run("exp", "spec-hash", run_id)
        assert missing.value.code == "store.run.not_found"
    assert store.history("exp", _history_selector()).num_rows == 0


@pytest.mark.parametrize(
    "message",
    [
        "Insufficient privileges to operate on schema 'ABSENT_SCHEMA'",
        "Schema 'ABSENT_SCHEMA' does not exist or not authorized.",
        "Table INFORMATION_SCHEMA.TABLES does not exist",
    ],
    ids=["privileges", "not-authorized", "other-object"],
)
# A role's schema listing omits a schema it may not read, so the listing cannot prove absence.
@pytest.mark.parametrize("lists_schemas", [True, False], ids=["schema-listing", "no-listing"])
def test_unrelated_table_listing_errors_propagate(message, lists_schemas):
    error = _SchemaLookupError(message)
    store = WarehouseResultsStore(
        _listing_backend(error, lists_schemas=lists_schemas), schema="absent_schema"
    )

    with pytest.raises(_SchemaLookupError) as raised:
        store.open_run("exp", "spec-hash", "run-id")
    assert raised.value is error
    with pytest.raises(_SchemaLookupError) as raised:
        store.history("exp", _history_selector())
    assert raised.value is error


_BIGQUERY_DATASET = "data-project.default_dataset"


def _bigquery_backend(error: Exception) -> Any:
    """A BigQuery-shaped backend whose dataset listing omits the store's dataset.

    Its client reports an error as ``"<HTTP status> <message>"``.
    """

    def list_tables(database=None):
        assert database == _BIGQUERY_DATASET
        raise error

    return SimpleNamespace(
        name="bigquery",
        list_tables=list_tables,
        list_databases=lambda catalog=None: [],
        create_database=_no_table_operations,
        **dict.fromkeys(("table", "create_table"), _no_table_operations),
    )


def test_missing_bigquery_dataset_run_lookup_is_not_found_and_history_is_empty():
    store = WarehouseResultsStore(
        _bigquery_backend(
            _SchemaLookupError("404 Not found: Dataset data-project:default_dataset")
        ),
        schema=_BIGQUERY_DATASET,
    )

    for run_id in ("run-id", None):
        with pytest.raises(CodedError) as missing:
            store.open_run("exp", "spec-hash", run_id)
        assert missing.value.code == "store.run.not_found"
    assert store.history("exp", _history_selector()).num_rows == 0


@pytest.mark.parametrize(
    "message",
    [
        "403 Access Denied: Dataset data-project:default_dataset",
        "404 Not found: Table data-project:default_dataset.increment_run_0",
    ],
    ids=["access-denied", "other-object"],
)
def test_bigquery_dataset_access_and_other_object_errors_propagate(message):
    error = _SchemaLookupError(message)
    store = WarehouseResultsStore(_bigquery_backend(error), schema=_BIGQUERY_DATASET)

    with pytest.raises(_SchemaLookupError) as raised:
        store.open_run("exp", "spec-hash", "run-id")
    assert raised.value is error
    with pytest.raises(_SchemaLookupError) as raised:
        store.history("exp", _history_selector())
    assert raised.value is error


@pytest.mark.integration
def test_results_store_saves_into_an_existing_sqlite_schema():
    backend = ibis.sqlite.connect()
    experiment, resolver = _analysis()
    try:
        saved = _warehouse_run(view_readout(experiment, resolver))
        store = WarehouseResultsStore(backend, schema="main")
        store.put(saved)
        assert store.open_run("exp", "spec", "one").run.manifest_hash == saved.manifest_hash
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_store_publishes_and_pins_complete_run():
    backend = ibis.duckdb.connect()
    try:
        backend.create_database("analytics")
        experiment, resolver = from_frame(
            pd.DataFrame(
                {
                    "unit": ["c1", "c2", "t1", "t2"],
                    "arm": ["control", "control", "treatment", "treatment"],
                    "value": [1.0, 2.0, 2.0, 3.0],
                }
            ),
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
            experiment_id="exp",
        )
        rows = view_readout(experiment, resolver)
        scope = View(view="run", grain="total")
        cells = tuple(
            replace(
                cell,
                identity=cell.identity.model_copy(
                    update={
                        "experiment_id": "exp",
                        "analysis_spec_hash": "spec",
                        "metric_hash": "metric",
                        "estimator_id": "t",
                        "method_config_hash": "config",
                        "method_instance_id": "instance",
                        "scope": scope,
                    }
                ),
                provenance=replace(cell.provenance, run_id="one", execution_identity_hash="exec"),
            )
            for cell in rows.as_cells(created_at=datetime(2026, 1, 1, tzinfo=UTC))
        )
        run = Run(
            "exp",
            "one",
            "spec",
            "exec",
            None,
            tuple(c.identity for c in cells),
            cells,
            rows.metadata,
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 2, tzinfo=UTC),
            scope_analysis_hashes=("spec",),
            cell_scope_hashes=tuple((c.identity.hash(), "spec") for c in cells),
        )
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(run)
        backend.create_table(
            "increment_run_unrelated",
            obj=pa.table({"payload": pa.array([b"corrupt"], type=pa.binary())}),
            database="analytics",
        )
        handle = store.open_run("exp", "spec", "one")
        assert handle.run.manifest_hash == run.manifest_hash
        history = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=View(view="run", grain="total").hash(),
                contrast="control:treatment",
            ),
        )
        assert history.num_rows == 1
        assert handle.envelope().cells == cells
        with pytest.raises(CodedError) as duplicate:
            store.put(run)
        assert duplicate.value.code == "store.run.duplicate"
    finally:
        backend.disconnect()


def _analysis(*, control="control", treatment="treatment"):
    return from_frame(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": [control, control, treatment, treatment],
                "value": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        shape="summary",
        unit="unit",
        group="arm",
        control=control,
        metrics={"value": "mean"},
        experiment_id="exp",
    )


def _payloads_by_run_id(backend):
    import json

    import pyarrow.parquet as pq

    result = {}
    for name in backend.list_tables(database="analytics"):
        if not name.startswith("increment_run_"):
            continue
        payload = backend.table(name, database="analytics").to_pyarrow()["payload"][0].as_py()
        manifest = json.loads(
            pq.read_schema(pa.BufferReader(payload)).metadata[b"increment.run_manifest.v2"]
        )
        result[manifest["run_id"]] = (name, payload)
    return result


@pytest.mark.integration
@pytest.mark.parametrize("reopen", ["by id", "latest", "history"])
def test_reopen_rejects_a_foreign_run_behind_a_short_name(reopen):
    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    experiment, resolver = _analysis()
    try:
        foreign = _warehouse_run(
            view_readout(experiment, resolver),
            run_id="foreign",
            experiment_id="foreign-exp",
            analysis_spec_hash="foreign-spec",
        )
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(foreign)
        _, foreign_payload = _payloads_by_run_id(backend)["foreign"]
        backend.create_table(
            _committed_name("exp", "spec", "one"),
            obj=pa.table({"payload": pa.array([foreign_payload], type=pa.binary())}),
            database="analytics",
        )
        operation = {
            "by id": lambda: store.open_run("exp", "spec", "one"),
            "latest": lambda: store.open_run("exp", "spec"),
            "history": lambda: store.history(
                "exp",
                HistorySelector(
                    metric_hash="metric",
                    estimator_id="t",
                    method_config_hash="config",
                    method_instance_id="instance",
                    scope_hash=View(view="run", grain="total").hash(),
                ),
            ),
        }[reopen]

        with pytest.raises(CodedError) as error:
            operation()
        assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_discovery_ignores_corrupt_payloads_for_other_experiments():
    import pyarrow.parquet as pq

    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        healthy = _warehouse_run(rows, experiment_id="exp-a", analysis_spec_hash="spec-a")
        unrelated = _warehouse_run(
            rows,
            run_id="other",
            experiment_id="exp-b",
            analysis_spec_hash="spec-b",
        )
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(healthy)
        store.put(unrelated)
        table_name, payload = _payloads_by_run_id(backend)["other"]
        table = pq.read_table(pa.BufferReader(payload))
        index = table.schema.get_field_index("identity")
        damaged = table.set_column(
            index,
            table.schema.field(index),
            pa.array(["not json"] * table.num_rows),
        )
        sink = pa.BufferOutputStream()
        pq.write_table(damaged, sink)
        backend.drop_table(table_name, database="analytics")
        backend.create_table(
            table_name,
            obj=pa.table({"payload": pa.array([sink.getvalue().to_pybytes()], type=pa.binary())}),
            database="analytics",
        )

        assert store.open_run("exp-a", "spec-a").run.run_id == "one"
        assert healthy.cells[0].identity.scope is not None
        history = store.history(
            "exp-a",
            HistorySelector(
                "metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=healthy.cells[0].identity.scope.hash(),
            ),
        )
        assert history.num_rows == 1
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_latest_and_history_reject_unreadable_committed_records():
    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        older = _warehouse_run(rows, run_id="older", committed_day=2)
        newer = _warehouse_run(rows, run_id="newer", committed_day=3)
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(older)
        store.put(newer)
        table_name, _ = _payloads_by_run_id(backend)["newer"]
        backend.drop_table(table_name, database="analytics")
        backend.create_table(
            table_name,
            obj=pa.table({"payload": pa.array([b"unreadable"], type=pa.binary())}),
            database="analytics",
        )

        for operation in (
            lambda: store.open_run("exp", "spec"),
            lambda: store.history(
                "exp",
                HistorySelector(
                    metric_hash="metric",
                    estimator_id="t",
                    method_config_hash="config",
                    method_instance_id="instance",
                    scope_hash=older.expected_roster[0].scope.hash(),
                ),
            ),
        ):
            with pytest.raises(CodedError) as error:
                operation()
            assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_contrast_matches_control_labels_containing_colons():
    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    experiment, resolver = _analysis(control="control:v1")
    try:
        rows = view_readout(experiment, resolver)
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(_warehouse_run(rows))

        history = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=View(view="run", grain="total").hash(),
                contrast="control:v1:treatment",
            ),
        )
        assert history.num_rows == 1
    finally:
        resolver.close()
        backend.disconnect()
