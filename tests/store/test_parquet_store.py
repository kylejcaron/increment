import copy
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from increment import Analysis, from_frame
from increment import run as run_views
from increment.errors import CodedError
from increment.estimation.readout_types import ReadoutResults
from increment.kernel.contracts import Scope
from increment.result_envelope import ResultCell
from increment.store import HistorySelector, ParquetResultsStore, Run
from tests.analysis_factory import view_readout

_FRAME = {
    "unit": ["c1", "c2", "c3", "c4", "t1", "t2", "t3", "t4"],
    "value": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0],
}
# Readout JSON with persisted cells, written by the removed ``Analysis.run(store=)``.
_PERSISTED_READOUT = (
    Path(__file__).parents[1] / "fixtures" / "readout_persisted_cells_reference.json"
)


def _analysis(*, control="control", treatment="treatment", metrics=None):
    return from_frame(
        pd.DataFrame({**_FRAME, "arm": [control] * 4 + [treatment] * 4}),
        shape="summary",
        unit="unit",
        group="arm",
        control=control,
        metrics={"value": "mean"} if metrics is None else metrics,
        experiment_id="exp",
    )


def test_latest_run_identity_includes_the_effective_plan(tmp_path):
    from increment.semantics.models import AnalysisPlan

    store = ParquetResultsStore(tmp_path)
    hashes = []
    for alpha in (0.05, 0.1):
        experiment, resolver = from_frame(
            pd.DataFrame(
                {
                    "unit": ["c1", "c2", "t1", "t2"],
                    "arm": ["control", "control", "treatment", "treatment"],
                    "value": [1.0, 2.0, 2.0, 4.0],
                }
            ),
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics={"value": "mean"},
            experiment_id="exp",
            plan=AnalysisPlan(primary="value", alpha=alpha),
        )
        try:
            saved = run_views(experiment, resolver, results_store=store).run
            hashes.append(saved.analysis_spec_hash)
            opened = store.open_run("exp", saved.analysis_spec_hash)
            assert opened.run.analysis_spec_hash == saved.analysis_spec_hash
        finally:
            resolver.close()
    assert hashes[0] != hashes[1]


def test_source_capture_changes_keep_spec_and_latest_run_identity_stable(tmp_path):
    store = ParquetResultsStore(tmp_path)
    content_digests = []
    run_ids = []
    spec_hashes = []
    for offset in (0.0, 1.0):
        experiment, resolver = from_frame(
            pd.DataFrame(
                {
                    "unit": ["c1", "c2", "t1", "t2"],
                    "arm": ["control", "control", "treatment", "treatment"],
                    "value": [1.0 + offset, 2.0 + offset, 2.0 + offset, 4.0 + offset],
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
            saved = run_views(experiment, resolver, results_store=store).run
            content_digests.append(saved.cells[0].identity.source_content_digest)
            run_ids.append(saved.run_id)
            spec_hashes.append(saved.analysis_spec_hash)
        finally:
            resolver.close()

    assert content_digests[0] != content_digests[1]
    latest = store.open_run("exp", spec_hashes[1])
    assert latest.run.run_id == run_ids[1]


def test_latest_run_identity_includes_effective_cuped_covariates(tmp_path):
    from increment.estimation.engine import Method
    from increment.frame import MetricSpec

    data = pd.DataFrame(
        {
            "unit": ["c1", "c2", "t1", "t2"],
            "arm": ["control", "control", "treatment", "treatment"],
            "value": [1.0, 2.0, 2.0, 4.0],
            "pre_a": [2.0, 3.0, 4.0, 6.0],
            "pre_b": [5.0, 2.0, 1.0, 3.0],
        }
    )
    store = ParquetResultsStore(tmp_path)
    hashes = []
    for covariate in ("pre_a", "pre_b"):
        experiment, resolver = from_frame(
            data,
            shape="summary",
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="value", type="mean", decision_method=Method.cuped(covariates=covariate)
                )
            ],
            experiment_id="exp",
        )
        try:
            saved = run_views(experiment, resolver, results_store=store).run
            hashes.append(saved.analysis_spec_hash)
            assert store.open_run("exp", saved.analysis_spec_hash).run.run_id
        finally:
            resolver.close()
    assert hashes[0] != hashes[1]


def _cells(
    rows,
    *,
    run_id="run-1",
    execution_hash="execution",
    identity_fields=True,
    experiment_id="exp",
    analysis_spec_hash="spec",
):
    scope = Scope(view="run", grain="total")
    cells = []
    for cell in rows.as_cells(created_at=datetime(2026, 1, 1, tzinfo=UTC)):
        update = {
            "experiment_id": experiment_id,
            "analysis_spec_hash": analysis_spec_hash,
            "scope": scope,
        }
        if identity_fields:
            update.update(
                {
                    "metric_hash": "metric",
                    "estimator_id": "t",
                    "method_config_hash": "config",
                    "method_instance_id": "instance",
                }
            )
        identity = cell.identity.model_copy(update=update)
        provenance = replace(cell.provenance, run_id=run_id, execution_identity_hash=execution_hash)
        cells.append(replace(cell, identity=identity, provenance=provenance))
    return tuple(cells), scope


def _run(
    rows,
    cells,
    scope,
    *,
    run_id="run-1",
    execution_hash="execution",
    committed_day=2,
    experiment_id="exp",
    analysis_spec_hash="spec",
):
    return Run(
        experiment_id,
        run_id,
        analysis_spec_hash,
        execution_hash,
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


def test_saved_run_cells_keep_their_identity_and_provenance(tmp_path):
    experiment, resolver = _analysis()
    store = ParquetResultsStore(tmp_path)
    try:
        result = run_views(experiment, resolver, results_store=store)
        handle = store.open_run("exp", result.run.analysis_spec_hash, result.run.run_id)
        run = handle.run
        cells = run.cells
        assert run.metadata is not None
        changed = replace(run, metadata=run.metadata.model_copy(update={"coherence": "live"}))
        assert changed.manifest_hash != run.manifest_hash
        assert cells
        assert result.cells == cells
        assert all(cell.identity.source_snapshot_id for cell in cells)
        assert all(cell.identity.source_content_digest for cell in cells)
        assert all(
            cell.identity.source_content_digest == cell.provenance.source_content_digest
            for cell in cells
        )
        assert handle.envelope().cells == cells
        assert handle.get(result.cells[0].identity)
    finally:
        resolver.close()


def test_saved_readout_json_keeps_its_persisted_cell_identity():
    saved = json.loads(_PERSISTED_READOUT.read_text())["readout"]
    persisted = ResultCell.from_payload(saved["persisted_cells"][0]).identity
    rows = ReadoutResults.model_validate_json(json.dumps(saved))
    assert rows.as_cells()[0].identity == persisted
    assert rows[:].as_cells()[0].identity == persisted
    assert rows.filter(lambda _row: True).as_cells()[0].identity == persisted
    assert copy.deepcopy(rows).as_cells()[0].identity == persisted
    assert pickle.loads(pickle.dumps(rows)).as_cells()[0].identity == persisted
    restored = ReadoutResults.model_validate_json(rows.model_dump_json())
    assert restored.as_cells()[0].identity == persisted
    tampered = copy.deepcopy(saved)
    tampered["persisted_cells"] = []
    with pytest.raises(CodedError) as invalid_cache:
        ReadoutResults.model_validate_json(json.dumps(tampered))
    assert invalid_cache.value.code == "readout.serialization.identity_mismatch"
    foreign = copy.deepcopy(saved)
    foreign["metadata"]["run_id"] = "another-run"
    with pytest.raises(CodedError) as foreign_cells:
        ReadoutResults.model_validate_json(json.dumps(foreign))
    assert foreign_cells.value.code == "readout.serialization.identity_mismatch"


def test_percentile_winsorized_persisted_results_round_trip_json(tmp_path):
    import numpy as np

    from increment.semantics.models import AnalysisPlan

    rng = np.random.default_rng(3)
    n = 20
    analysis = Analysis.from_unit_summary(
        pd.DataFrame(
            {
                "unit": list(range(2 * n)),
                "arm": ["control"] * n + ["treatment"] * n,
                "value": np.concatenate([rng.lognormal(1.0, 1.0, n), rng.lognormal(1.3, 1.0, n)]),
            }
        ),
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            {
                "name": "value",
                "winsorization": {
                    "upper_percentile": 0.75,
                    "support": {"lower": 0, "upper": 1000, "provenance": "fixture"},
                },
            }
        ],
        experiment_id="exp",
        plan=AnalysisPlan(primary="value"),
    )
    try:
        from increment.estimation.results import LiftEstimate

        rows = analysis.run(store=ParquetResultsStore(tmp_path))

        from increment.estimation.readout_types import ReadoutResults

        original = rows[0]
        assert isinstance(original, LiftEstimate)
        assert original.confidence_set is not None

        restored = ReadoutResults.model_validate_json(rows.model_dump_json())
        restored_row = restored[0]
        assert isinstance(restored_row, LiftEstimate)
        assert restored_row.confidence_set == original.confidence_set
        assert restored.as_cells()[0].identity == rows.as_cells()[0].identity
    finally:
        analysis.close()


def test_declared_prior_changes_persisted_spec_identity(tmp_path):
    from increment.estimation.inference import Normal
    from increment.frame import MetricSpec

    store = ParquetResultsStore(tmp_path)
    hashes = []
    for mu in (0.0, 1.0):
        experiment, resolver = _analysis(
            metrics=[MetricSpec(name="value", type="mean", prior=Normal(mu=mu, sigma=0.1))]
        )
        try:
            hashes.append(
                run_views(experiment, resolver, results_store=store).run.analysis_spec_hash
            )
        finally:
            resolver.close()
    assert hashes[0] != hashes[1]


def test_history_keeps_additive_se_and_df_for_clustered_large_offset(tmp_path):
    import math

    origin = float(2**48)
    step = math.ulp(origin)
    values = {
        "unit": ["c0", "c1", "c2", "t0", "t1", "t2"],
        "group": ["control", "control", "control", "treatment", "treatment", "treatment"],
        "store": ["c0", "c1", "c2", "t0", "t1", "t2"],
        "y": [
            origin,
            origin + step,
            origin,
            origin + step,
            origin + 2 * step,
            origin + 2 * step,
        ],
    }
    experiment, resolver = from_frame(
        pd.DataFrame(values),
        shape="summary",
        unit="unit",
        group="group",
        control="control",
        metrics={"y": "mean"},
        cluster="store",
        experiment_id="offset",
    )
    store = ParquetResultsStore(tmp_path)
    try:
        from increment.errors import IncrementRuntimeWarning

        with pytest.warns(IncrementRuntimeWarning):
            saved = run_views(experiment, resolver, results_store=store).run
        run = store.open_run("offset", saved.analysis_spec_hash, saved.run_id).run
        cell = saved.cells[0]
        scope = cell.identity.scope
        assert cell.identity.metric_hash is not None
        assert scope is not None
        assert cell.identity.estimator_id is not None
        assert cell.identity.method_config_hash is not None
        assert cell.identity.method_instance_id is not None
        history = store.history(
            "offset",
            HistorySelector(
                metric_hash=cell.identity.metric_hash,
                estimator_id=cell.identity.estimator_id,
                method_config_hash=cell.identity.method_config_hash,
                method_instance_id=cell.identity.method_instance_id,
                scope_hash=scope.hash(),
                role=cell.identity.key.method_role,
            ),
        )
        retained = cell.retained_row()
        assert retained.abs_se == pytest.approx(0.02946278254943948)
        assert retained.abs_reference_df == pytest.approx(4.0)
        assert history["se"].to_pylist()[0] == pytest.approx(0.02946278254943948)
        assert history["df"].to_pylist()[0] == pytest.approx(4.0)
        assert run.run_id == saved.run_id
    finally:
        resolver.close()


def test_public_analysis_cells_round_trip_through_parquet_store(tmp_path):
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        store = ParquetResultsStore(tmp_path)
        store.put(run)
        handle = store.open_run("exp", "spec", "run-1")
        assert handle.run.source_content_digest == run.source_content_digest
        history = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=scope.hash(),
                role="decision",
            ),
        )
        assert history.schema.field("point").type == pa.float64()
        assert history.num_rows == 1

        assert history["se"].to_pylist()[0] == pytest.approx(
            rows[0].lift.log_se if rows[0].lift is not None else None
        )
        contrast = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=scope.hash(),
                contrast="control:treatment",
                role="decision",
            ),
        )
        assert contrast.num_rows == 1
        arrow = handle.envelope().to_arrow()
        for name in ("point", "lower", "upper", "se", "df"):
            assert arrow.schema.field(name).type == pa.float64()
        expected_se = rows[0].lift.log_se if rows[0].lift is not None else None
        mismatched_method = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="different-config",
                method_instance_id="instance",
                scope_hash=scope.hash(),
                role="decision",
            ),
        )
        assert mismatched_method.num_rows == 0
        assert arrow["se"].to_pylist()[0] == pytest.approx(expected_se)

        conflicting_cells = tuple(
            replace(cell, provenance=replace(cell.provenance, execution_identity_hash="other"))
            for cell in cells
        )
        with pytest.raises(CodedError) as duplicate:
            store.put(replace(run, execution_identity_hash="other", cells=conflicting_cells))
        assert duplicate.value.code == "store.run.duplicate"

        newer_cells = tuple(
            replace(cell, provenance=replace(cell.provenance, run_id="run-2")) for cell in cells
        )
        store.put(_run(rows, newer_cells, scope, run_id="run-2", committed_day=3))
        assert handle.run.run_id == "run-1"
        assert handle.envelope().cells == cells
    finally:
        resolver.close()


def test_run_manifest_keeps_artifact_provenance_separate_from_content_digest(tmp_path):
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        manifest_digest = "sha256:" + "a" * 64
        relation_digests = (("exposures", "sha256:" + "b" * 64),)
        cells = tuple(
            replace(
                cell,
                provenance=replace(
                    cell.provenance,
                    artifact_manifest_sha256=manifest_digest,
                    artifact_relation_digests=relation_digests,
                ),
            )
            for cell in cells
        )
        run = _run(rows, cells, scope)
        assert run.artifact_manifest_sha256 == manifest_digest
        assert run.artifact_relation_digests == relation_digests
        assert run.source_content_digest != manifest_digest

        store = ParquetResultsStore(tmp_path)
        store.put(run)
        restored = store.open_run("exp", "spec", "run-1").run
        assert restored.artifact_manifest_sha256 == manifest_digest
        assert restored.artifact_relation_digests == relation_digests
        assert restored.source_content_digest == run.source_content_digest
    finally:
        resolver.close()


def test_run_requires_every_cell_in_scope_mapping():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        with pytest.raises(CodedError) as error:
            replace(run, cell_scope_hashes=())
        assert error.value.code == "store.run.scope_mapping_invalid"
    finally:
        resolver.close()


def test_run_rejects_cell_provenance_from_another_execution():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        bad_cells = (
            replace(
                cells[0],
                provenance=replace(
                    cells[0].provenance,
                    run_id="other-run",
                    execution_identity_hash="other-implementation-pin",
                ),
            ),
            *cells[1:],
        )
        with pytest.raises(CodedError) as error:
            _run(rows, bad_cells, scope)
        assert error.value.code == "store.run.provenance_mismatch"
    finally:
        resolver.close()


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("experiment_id", "other-experiment", "store.run.identity_mismatch"),
        ("analysis_spec_hash", "other-spec", "store.run.scope_mapping_invalid"),
    ],
)
def test_run_rejects_cell_spec_or_experiment_outside_manifest(field, value, code):
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        identity = cells[0].identity.model_copy(update={field: value})
        bad_cell = replace(cells[0], identity=identity)
        with pytest.raises(CodedError) as error:
            _run(rows, (bad_cell, *cells[1:]), scope)
        assert error.value.code == code
    finally:
        resolver.close()


def test_result_cell_rejects_snapshot_that_differs_from_identity():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, _scope = _cells(rows)
        with pytest.raises(CodedError) as error:
            replace(cells[0], source_snapshot_id="other-snapshot")
        assert error.value.code == "results.cell.schema_unsupported"
    finally:
        resolver.close()


def test_run_rejects_population_scope_that_differs_from_cell():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        identity = cells[0].identity.model_copy(
            update={"scope": Scope(view="run", grain="total", population="triggered")}
        )
        bad_cell = replace(cells[0], identity=identity)
        with pytest.raises(CodedError) as error:
            _run(rows, (bad_cell, *cells[1:]), scope)
        assert error.value.code == "store.run.identity_mismatch"
    finally:
        resolver.close()


def test_run_rejects_mismatched_segment_scope_descriptor():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        key = cells[0].identity.key.model_copy(
            update={"kind": "segment", "dimension": "country", "dimension_value": "US"}
        )
        identity = cells[0].identity.model_copy(
            update={
                "key": key,
                "scope": Scope(
                    view="breakout",
                    grain="total",
                    segment=({"property": "other_dimension", "source": None},),
                ),
            }
        )
        bad_cell = replace(cells[0], identity=identity)
        with pytest.raises(CodedError) as error:
            _run(rows, (bad_cell, *cells[1:]), scope)
        assert error.value.code == "store.run.identity_mismatch"
    finally:
        resolver.close()


def test_null_numeric_fields_keep_their_declared_arrow_type():
    experiment, resolver = _analysis()
    try:
        cells = view_readout(experiment, resolver).as_cells()
        nullable_cells = (replace(cells[0], summary=None),)
        table = ResultCell.to_arrow(nullable_cells)
        assert table.num_rows
        assert table.schema.field("point").type == pa.float64()
        assert table["point"].to_pylist() == [None]
    finally:
        resolver.close()


def test_manifest_commits_retained_cell_payload_bytes():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        changed_summary = replace(cells[0].summary, point=cells[0].summary.point + 1)
        changed_cell = replace(cells[0], summary=changed_summary)
        changed_run = replace(run, cells=(changed_cell, *cells[1:]))
        assert changed_run.manifest_hash != run.manifest_hash
    finally:
        resolver.close()


def test_open_run_ignores_corrupt_records_for_other_experiments(tmp_path):
    import json

    import pyarrow.parquet as pq

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells_a, scope_a = _cells(
            rows,
            run_id="run-a",
            experiment_id="exp-a",
            analysis_spec_hash="spec-a",
        )
        cells_b, scope_b = _cells(
            rows,
            run_id="run-b",
            experiment_id="exp-b",
            analysis_spec_hash="spec-b",
        )
        store = ParquetResultsStore(tmp_path)
        store.put(
            _run(
                rows,
                cells_a,
                scope_a,
                run_id="run-a",
                experiment_id="exp-a",
                analysis_spec_hash="spec-a",
            )
        )
        store.put(
            _run(
                rows,
                cells_b,
                scope_b,
                run_id="run-b",
                experiment_id="exp-b",
                analysis_spec_hash="spec-b",
            )
        )
        corrupt = next(
            path
            for path in tmp_path.glob("*.parquet")
            if json.loads(pq.read_schema(path).metadata[b"increment.run_manifest.v2"])["run_id"]
            == "run-b"
        )
        table = pq.read_table(corrupt)
        identity_index = table.schema.get_field_index("identity")
        table = table.set_column(
            identity_index,
            table.schema.field(identity_index),
            pa.array(["not json"] * table.num_rows),
        )
        pq.write_table(table, corrupt)

        assert store.open_run("exp-a", "spec-a").run.run_id == "run-a"
        history = store.history(
            "exp-a",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=scope_a.hash(),
                role="decision",
            ),
        )
        assert history.num_rows == 1
    finally:
        resolver.close()


def test_manifest_rejects_tampered_scope_analysis_hashes(tmp_path):
    import json

    import pyarrow.parquet as pq

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        store = ParquetResultsStore(tmp_path)
        store.put(run)
        path = next(tmp_path.glob("*.parquet"))
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        manifest = json.loads(metadata[b"increment.run_manifest.v2"])
        manifest["scope_analysis_hashes"].append("sha256:unused")
        metadata[b"increment.run_manifest.v2"] = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        pq.write_table(table.replace_schema_metadata(metadata), path)

        with pytest.raises(CodedError) as error:
            store.open_run("exp", "spec", "run-1")
        assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()


def test_discovery_rejects_manifest_routing_coordinates_that_do_not_match_file(tmp_path):
    import json

    import pyarrow.parquet as pq

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        store = ParquetResultsStore(tmp_path)
        store.put(_run(rows, cells, scope))
        path = next(tmp_path.glob("*.parquet"))
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        manifest = json.loads(metadata[b"increment.run_manifest.v2"])
        manifest["experiment_id"] = "other-experiment"
        metadata[b"increment.run_manifest.v2"] = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        pq.write_table(table.replace_schema_metadata(metadata), path)

        with pytest.raises(CodedError) as error:
            store.open_run("exp", "spec")
        assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()


@pytest.mark.parametrize("manifest_payload", [b"{}", b"null"])
def test_malformed_run_manifest_is_reported_during_discovery(tmp_path, manifest_payload):
    import pyarrow.parquet as pq

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        store = ParquetResultsStore(tmp_path)
        store.put(_run(rows, cells, scope))
        path = next(tmp_path.glob("*.parquet"))
        table = pq.read_table(path)
        metadata = dict(table.schema.metadata or {})
        metadata[b"increment.run_manifest.v2"] = manifest_payload
        pq.write_table(table.replace_schema_metadata(metadata), path)

        with pytest.raises(CodedError) as error:
            store.open_run("exp", "spec")
        assert error.value.code == "store.run.corrupt"

        with pytest.raises(CodedError) as error:
            store.history(
                "exp",
                HistorySelector(
                    "metric",
                    estimator_id="t",
                    method_config_hash="config",
                    method_instance_id="instance",
                    scope_hash="scope",
                ),
            )
        assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()


def test_run_commit_time_cannot_precede_capture_time():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        with pytest.raises(CodedError) as error:
            replace(run, captured_at=datetime(2026, 1, 3, tzinfo=UTC))
        assert error.value.code == "store.run.timestamp_invalid"
    finally:
        resolver.close()


def test_run_rejects_missing_requested_metadata():
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        with pytest.raises(CodedError) as error:
            replace(run, metadata=None)
        assert error.value.code == "store.run.metadata_mismatch"
    finally:
        resolver.close()


def test_concurrent_parquet_put_has_one_immutable_winner(tmp_path):
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        run = _run(rows, cells, scope)
        store = ParquetResultsStore(tmp_path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(store.put, run) for _ in range(2)]
            outcomes = []
            for future in futures:
                try:
                    future.result()
                except CodedError as error:
                    outcomes.append(error.code)
                else:
                    outcomes.append("committed")
        assert outcomes.count("committed") == 1
        assert outcomes.count("store.run.duplicate") == 1
    finally:
        resolver.close()


def test_explicit_run_open_ignores_corrupt_unrelated_parquet(tmp_path):
    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        healthy = _run(rows, cells, scope)
        other_cells = tuple(
            replace(cell, provenance=replace(cell.provenance, run_id="other")) for cell in cells
        )
        other = _run(rows, other_cells, scope, run_id="other")
        store = ParquetResultsStore(tmp_path)
        store.put(healthy)
        store.put(other)
        import json

        import pyarrow.parquet as pq

        unrelated = next(
            path
            for path in tmp_path.glob("*.parquet")
            if json.loads(pq.read_metadata(path).metadata[b"increment.run_manifest.v2"])["run_id"]
            == "other"
        )
        unrelated.write_bytes(b"corrupt unrelated run")

        assert store.open_run("exp", "spec", "run-1").run.run_id == "run-1"
    finally:
        resolver.close()


def test_explicit_open_rejects_a_different_valid_run_at_requested_path(tmp_path):
    import json

    import pyarrow.parquet as pq

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        requested_cells, scope = _cells(rows)
        requested = _run(rows, requested_cells, scope)
        foreign_cells, foreign_scope = _cells(
            rows,
            run_id="foreign-run",
            experiment_id="foreign-exp",
            analysis_spec_hash="foreign-spec",
        )
        foreign = _run(
            rows,
            foreign_cells,
            foreign_scope,
            run_id="foreign-run",
            experiment_id="foreign-exp",
            analysis_spec_hash="foreign-spec",
        )
        store = ParquetResultsStore(tmp_path)
        store.put(requested)
        store.put(foreign)
        records = {
            json.loads(pq.read_schema(path).metadata[b"increment.run_manifest.v2"])["run_id"]: path
            for path in tmp_path.glob("*.parquet")
        }
        records["run-1"].write_bytes(records["foreign-run"].read_bytes())

        with pytest.raises(CodedError) as error:
            store.open_run("exp", "spec", "run-1")
        assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()


def test_latest_and_history_reject_unreadable_committed_parquet_records(tmp_path):
    import json

    import pyarrow.parquet as pq

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        older_cells = tuple(
            replace(cell, provenance=replace(cell.provenance, run_id="older")) for cell in cells
        )
        older = _run(rows, older_cells, scope, run_id="older", committed_day=2)
        newer_cells = tuple(
            replace(cell, provenance=replace(cell.provenance, run_id="newer")) for cell in cells
        )
        newer = _run(rows, newer_cells, scope, run_id="newer", committed_day=3)
        store = ParquetResultsStore(tmp_path)
        store.put(older)
        store.put(newer)
        records = {
            json.loads(pq.read_schema(path).metadata[b"increment.run_manifest.v2"])["run_id"]: path
            for path in tmp_path.glob("*.parquet")
        }
        records["newer"].write_bytes(b"unreadable committed record")

        for operation in (
            lambda: store.open_run("exp", "spec"),
            lambda: store.history(
                "exp",
                HistorySelector(
                    metric_hash="metric",
                    estimator_id="t",
                    method_config_hash="config",
                    method_instance_id="instance",
                    scope_hash=scope.hash(),
                ),
            ),
        ):
            with pytest.raises(CodedError) as error:
                operation()
            assert error.value.code == "store.run.corrupt"
    finally:
        resolver.close()


def test_contrast_selector_matches_control_labels_containing_colons(tmp_path):
    experiment, resolver = _analysis(control="control:v1")
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        store = ParquetResultsStore(tmp_path)
        store.put(_run(rows, cells, scope))

        history = store.history(
            "exp",
            HistorySelector(
                metric_hash="metric",
                estimator_id="t",
                method_config_hash="config",
                method_instance_id="instance",
                scope_hash=scope.hash(),
                contrast="control:v1:treatment",
            ),
        )
        assert history.num_rows == 1
    finally:
        resolver.close()


def test_compiled_plan_binding_round_trips_with_run_manifest(tmp_path):
    from increment import AnalysisPlan
    from increment.frame import MetricSpec, synthesise_metric
    from increment.kernel.contracts import CompiledPlanBinding, Registration
    from increment.plan import compile_decision_plan

    experiment, resolver = _analysis()
    try:
        rows = view_readout(experiment, resolver)
        cells, scope = _cells(rows)
        compiled = compile_decision_plan(
            AnalysisPlan(primary="revenue"),
            [synthesise_metric(MetricSpec(name="revenue", type="mean"))],
            path="frame",
        )
        registration = Registration.from_plan(compiled)
        run = replace(
            _run(rows, cells, scope),
            registration_hash=registration.registration_hash,
            decision_plan=CompiledPlanBinding.from_plan(registration, compiled),
        )
        store = ParquetResultsStore(tmp_path)
        store.put(run)

        restored = store.open_run("exp", "spec", "run-1").run

        assert restored.manifest_version == 3
        assert restored.decision_plan == run.decision_plan
        assert restored.manifest_hash == run.manifest_hash
    finally:
        resolver.close()
