from dataclasses import replace
from datetime import UTC, datetime

import ibis
import pandas as pd
import pyarrow as pa
import pytest

from increment import Analysis
from increment.errors import CodedError
from increment.kernel.contracts import Scope
from increment.store import HistorySelector, Run, WarehouseResultsStore


def _warehouse_run(
    rows,
    *,
    run_id="one",
    experiment_id="exp",
    analysis_spec_hash="spec",
    committed_day=2,
):
    scope = Scope(view="run", grain="total")
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


@pytest.mark.integration
def test_analysis_run_persists_public_cells_without_identity_rewrites():
    backend = ibis.duckdb.connect()
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": ["control", "control", "treatment", "treatment"],
                "value": [1.0, 2.0, 2.0, 3.0],
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics={"value": "mean"},
        experiment_id="exp",
    )
    try:
        backend.create_database("analytics")
        store = WarehouseResultsStore(backend, schema="analytics")
        rows = analysis.run(store=store)
        metadata = rows.metadata
        assert metadata is not None
        assert metadata.analysis_spec_hash is not None
        assert metadata.run_id is not None
        handle = store.open_run("exp", metadata.analysis_spec_hash, metadata.run_id)
        cells = handle.envelope().cells
        assert cells
        assert all(cell.identity.source_snapshot_id for cell in cells)
        assert all(cell.identity.source_content_digest for cell in cells)
        assert all(
            cell.identity.source_content_digest == cell.provenance.source_content_digest
            for cell in cells
        )
    finally:
        analysis.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_store_publishes_and_pins_complete_run():
    backend = ibis.duckdb.connect()
    try:
        backend.create_database("analytics")
        analysis = Analysis.from_unit_summary(
            pd.DataFrame(
                {
                    "unit": ["c1", "c2", "t1", "t2"],
                    "arm": ["control", "control", "treatment", "treatment"],
                    "value": [1.0, 2.0, 2.0, 3.0],
                }
            ),
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
            experiment_id="exp",
        )
        rows = analysis.run()
        scope = Scope(view="run", grain="total")
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
                scope_hash=Scope(view="run", grain="total").hash(),
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
    return Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": ["c1", "c2", "t1", "t2"],
                "arm": [control, control, treatment, treatment],
                "value": [1.0, 2.0, 2.0, 3.0],
            }
        ),
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
        if not name.startswith("increment_run_") or len(name.removeprefix("increment_run_")) != 64:
            continue
        payload = backend.table(name, database="analytics").to_pyarrow()["payload"][0].as_py()
        manifest = json.loads(
            pq.read_schema(pa.BufferReader(payload)).metadata[b"increment.run_manifest.v2"]
        )
        result[manifest["run_id"]] = (name, payload)
    return result


@pytest.mark.integration
def test_warehouse_explicit_open_rejects_a_different_valid_run():
    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    analysis = _analysis()
    try:
        rows = analysis.run()
        requested = _warehouse_run(rows)
        foreign = _warehouse_run(
            rows,
            run_id="foreign",
            experiment_id="foreign-exp",
            analysis_spec_hash="foreign-spec",
        )
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(requested)
        store.put(foreign)
        payloads = _payloads_by_run_id(backend)
        target_name, _ = payloads["one"]
        _, foreign_payload = payloads["foreign"]
        backend.drop_table(target_name, database="analytics")
        backend.create_table(
            target_name,
            obj=pa.table({"payload": pa.array([foreign_payload], type=pa.binary())}),
            database="analytics",
        )

        with pytest.raises(CodedError) as error:
            store.open_run("exp", "spec", "one")
        assert error.value.code == "store.run.corrupt"
    finally:
        analysis.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_discovery_ignores_corrupt_payloads_for_other_experiments():
    import pyarrow.parquet as pq

    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    analysis = _analysis()
    try:
        rows = analysis.run()
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
        analysis.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_latest_and_history_reject_unreadable_committed_records():
    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    analysis = _analysis()
    try:
        rows = analysis.run()
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
        analysis.close()
        backend.disconnect()


@pytest.mark.integration
def test_warehouse_contrast_matches_control_labels_containing_colons():
    backend = ibis.duckdb.connect()
    backend.create_database("analytics")
    analysis = _analysis(control="control:v1")
    try:
        rows = analysis.run()
        store = WarehouseResultsStore(backend, schema="analytics")
        store.put(_warehouse_run(rows))

        history = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=Scope(view="run", grain="total").hash(),
                contrast="control:v1:treatment",
            ),
        )
        assert history.num_rows == 1
    finally:
        analysis.close()
        backend.disconnect()
