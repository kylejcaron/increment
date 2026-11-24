"""Retained-window fits keep their effect or their diagnosis through storage."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from shutil import copytree

import polars as pl
import pytest

from increment import AnalysisPlan, ExperimentMetric, from_switchback_frame, run
from increment.estimation.engine import Method
from increment.frame import MetricSpec
from increment.result_envelope import render
from increment.results import ReadoutResults
from increment.semantics.assignment import (
    FixedCountOrder,
    IndependentBernoulliOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.store import HistorySelector, ParquetResultsStore
from increment.tables import estimates_to_readout
from tests.capability_consumers_scalar import RATIO_DENOMINATOR_ADVISORY


def _experiment(metrics):
    rows = []
    for unit in range(6):
        for period in range(2):
            treated = period == unit % 2
            rows.append(
                {
                    "u": str(unit),
                    "c": 0,
                    "p": period,
                    "s": 0,
                    "arm": "B" if treated else "A",
                    "y": 2 * (unit + 1) if treated else 0,
                    "two": 2,
                    "zero": int(not treated),
                    "rare": int(not treated or unit == 0),
                    "level": 2**55 + ((3 if unit % 2 else -1) if treated else 0),
                    "one": 1,
                }
            )
    return _paired_experiment(rows, metrics)


def _paired_experiment(rows, metrics, plan=None, *, with_forecasts=True):
    forecasts = pl.DataFrame(
        {
            "u": row["u"],
            "c": row["c"],
            "p": row["p"],
            "prior_position": row["p"],
            "same_position": row["p"],
            "constant": 1,
            "forecast_at": date(2025, 1, 1),
            "assigned_at": date(2025, 1, 2),
        }
        for row in rows
    )
    return from_switchback_frame(
        pl.DataFrame(rows),
        unit="u",
        cycle="c",
        period="p",
        step="s",
        group="arm",
        metrics=metrics,
        plan=plan,
        experiment_id="retained",
        pre_assignment_forecasts=forecasts if with_forecasts else None,
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.5),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
        identification=Randomized(control_group="A", allocation={"A": 0.5, "B": 0.5}),
    )


@pytest.mark.integration
def test_unavailable_ratios_keep_their_point_status_and_valid_siblings_in_storage(tmp_path):
    metrics = [
        MetricSpec(name="orders", type="mean", value_column="y"),
        MetricSpec(
            name="adjusted",
            type="mean",
            value_column="y",
            decision_method=Method.cuped(covariates="prior_position"),
        ),
        MetricSpec(
            name="unadjustable",
            type="mean",
            value_column="y",
            decision_method=Method.cuped(covariates="constant"),
        ),
        MetricSpec(name="level_difference", type="ratio", numerator="level", denominator="one"),
        *(
            MetricSpec(name=d, type="ratio", numerator="y", denominator=d)
            for d in ("two", "rare", "zero")
        ),
    ]
    experiment, reader = _experiment(metrics)
    store = ParquetResultsStore(tmp_path)
    with reader:
        saved = run(experiment, reader, results_store=store)
        (alone,) = run(experiment, reader, metrics=["orders"]).decisions.cells
    frame = saved.to_frame(backend="polars")
    assert isinstance(frame, pl.DataFrame)
    columns = ["metric", "forecast_column", "p_value"]
    selected = saved.to_frame(backend="polars", columns=columns)
    assert isinstance(selected, pl.DataFrame)
    assert selected.equals(
        frame.select("scope_view", "scope_values", "scope_key", *columns, "view_partial")
    )
    rows = {r["metric"]: r for r in frame.to_dicts()}
    assert {metric: r["estimate"] for metric, r in rows.items()} == {
        "orders": 7,
        "adjusted": pytest.approx(7),
        "two": 3.5,
        "rare": 42,
        "zero": None,
        "level_difference": 1,
        "unadjustable": None,
    }
    assert (rows["adjusted"]["method"], rows["adjusted"]["dof"]) == (
        "switchback_cuped_jackknife_t",
        4,
    )
    assert rows["adjusted"]["forecast_column"] == "prior_position"
    assert (rows["two"]["reference"], rows["two"]["dof"]) == ("switchback_ratio_delta_t", 5)
    assert rows["orders"]["sampling_available"] and rows["orders"]["standard_error"] > 0
    assert not rows["orders"]["decision_scope_complete"]
    expected = RATIO_DENOMINATOR_ADVISORY.outcome("switchback-weak-denominator").expected
    assert (rows["rare"]["failure_code"],) == expected.codes

    handle = store.open_run("retained", saved.run.analysis_spec_hash, saved.run.run_id)
    envelope = handle.envelope()
    stored = {cell.identity.key.metric: cell for cell in envelope.cells}
    verdicts = {v.identity_hash: v for v in saved.decisions.cells}
    assert verdicts[stored["orders"].identity.hash()].status == alone.status == "reject"
    projected = {
        json.loads(r["identity"])["key"]["metric"]: r for r in envelope.to_arrow().to_pylist()
    }
    report = render(envelope, audience="full")
    rank = stored["unadjustable"]
    assert rank.summary is not None and rank.summary.point is None
    assert verdicts[rank.identity.hash()].reason_codes == ("evidence.p_value.unavailable",)
    assert rank.components["sampling"][0].context == {
        "reason": "rank_deficient",
        "metric": "unadjustable",
    }
    for metric, reason in (("rare", "weak_denominator"), ("zero", "zero_denominator")):
        row, cell = rows[metric], stored[metric]
        assert not row["sampling_available"] and row["point_available"] == (metric == "rare")
        context = {"metric": metric, "reason": reason, "arm": "B"}
        assert json.loads(row["failure_context"]) == context
        status, _ = cell.components["sampling"]
        assert status.state == "unavailable" and status.context == context
        verdict = verdicts[cell.identity.hash()]
        assert (verdict.status, verdict.reason_codes) == ("not_evaluated", (status.code,))
        key = cell.identity.key
        assert (key.aggregation, key.estimand) == ("ratio", "retained_window_ratio_difference")
        identity = cell.identity
        assert identity.metric_hash and identity.estimator_id and identity.method_config_hash
        assert identity.method_instance_id and identity.scope
        (history,) = store.history(
            "retained",
            HistorySelector(
                identity.metric_hash,
                identity.estimator_id,
                identity.method_config_hash,
                identity.method_instance_id,
                identity.scope.hash(),
            ),
        ).to_pylist()
        for view in (projected[metric], history):
            assert view["point"] == row["estimate"]
            assert view["reason_code"] == row["failure_code"]
            assert all(view[name] is None for name in ("lower", "upper", "se", "df"))
        assert row["p_value"] is None
        assert reason in report


@pytest.mark.integration
def test_constant_fixed_count_effect_keeps_point_but_not_a_test_in_saved_reports(tmp_path):
    panel = pl.DataFrame(
        {
            "u": "store",
            "c": block,
            "p": slot,
            "s": 0,
            "arm": "B" if slot == block % 3 else "A",
            "constant": 2 if slot == block % 3 else 0,
            "varying": 10 + block if slot == block % 3 else 0,
        }
        for block in range(9)
        for slot in range(3)
    )
    experiment, reader = from_switchback_frame(
        panel,
        unit="u",
        cycle="c",
        period="p",
        step="s",
        group="arm",
        metrics={"constant": "mean", "varying": "mean"},
        experiment_id="blocks",
        assignment=SwitchbackAssignment(
            periods_per_cycle=3,
            sequence=FixedCountOrder(treated_periods=1, independence_unit="shared_block"),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
        identification=Randomized(control_group="A", allocation={"A": 2 / 3, "B": 1 / 3}),
    )
    store = ParquetResultsStore(tmp_path)
    with reader:
        saved = run(experiment, reader, results_store=store)
    frame = saved.to_frame(backend="polars")
    assert isinstance(frame, pl.DataFrame)
    flat, varied = frame.sort("metric").to_dicts()
    assert flat["estimate"] == 2 and flat["point_available"]
    assert not flat["sampling_available"]
    assert flat["lb"] is flat["ub"] is flat["p_value"] is flat["stat_sig"] is None
    assert varied["sampling_available"] and varied["stat_sig"] and varied["standard_error"] > 0
    code = "evidence.p_value.unavailable"
    context = {"metric": "constant", "reason": "zero_standard_error"}
    assert (flat["failure_code"], json.loads(flat["failure_context"])) == (code, context)
    assert not flat["decision_scope_complete"] and not varied["decision_scope_complete"]
    assert (
        flat["available_cell_count"],
        flat["unavailable_cell_count"],
        flat["required_cell_count"],
    ) == (1, 1, 2)
    assert {"periods_per_cycle", "treated_periods", "treatment_probability"} <= set(frame.columns)
    selected = saved.to_frame(backend="polars", columns=frame.columns)
    assert isinstance(selected, pl.DataFrame) and selected.equals(frame)

    report = ReadoutResults.model_validate_json(saved.readout().model_dump_json())
    row = next(r for r in estimates_to_readout(report) if r["metric"] == "constant")
    assert row["lift"] == 2
    assert row["lower"] is row["higher"] is row["stat_sig"] is row["p_value"] is None
    assert row["available_cell_count"] == 1 and row["decision_scope_complete"] is False
    assert isinstance(row["failure_context"], str) and json.loads(row["failure_context"]) == context
    envelope = store.open_run("blocks", saved.run.analysis_spec_hash, saved.run.run_id).envelope()
    assert envelope.metadata is not None
    (source,) = envelope.metadata.scope.by_source.values()
    assert not source.decision_complete_by_population["assigned"]
    assert len(source.decision_cells) == 2
    cells = {cell.identity.key.metric: cell for cell in envelope.cells}
    cell = cells["constant"]
    status, _ = cell.components["sampling"]
    assert (status.state, status.code, status.context) == ("unavailable", code, context)
    assert cell.summary is not None and cell.summary.point == 2
    assert cell.summary.lower is cell.summary.upper is None
    verdicts = {v.identity_hash: v for v in saved.decisions.cells}
    verdict = verdicts[cell.identity.hash()]
    assert (verdict.status, verdict.reason_codes) == ("not_evaluated", (code,))
    assert verdicts[cells["varying"].identity.hash()].status == "reject"
    assert report.metadata is not None
    records = {r.cell.metric: r for r in report.metadata.cells}
    (advisory,) = (
        a for a in records["varying"].advisories if a.code == "estimation.contrast.approximate_t"
    )
    assert "approximat" in advisory.message.lower()
    assert "false positive" in advisory.message.lower()
    assert advisory.message in render(envelope, audience="full")
    assert advisory.code not in {a.code for a in records["constant"].advisories}
    from increment.tables import readout_table

    assert advisory.message in readout_table(report).gt().as_raw_html()


@pytest.mark.integration
def test_legacy_zero_se_report_preserves_identity_without_claiming_sampling(tmp_path):
    from increment.estimation.contrast_results import ContrastResult, ContrastResults
    from increment.estimation.results import contrast_p_value, contrast_stat_sig
    from increment.result_envelope import AnalysisEnvelope
    from increment.tables import readout_table

    # Public paired CT/TC run at 77ab667b227b2005e18bcc3d0f56c23b6aaec651:
    # Two units, one cycle, control 0, treatment 2. JSON adds only a final newline.
    fixture = Path(__file__).parent / "fixtures" / "switchback_legacy"
    saved_json = (fixture / "base-readout.json").read_text()
    original = json.loads(saved_json)
    row_json = json.dumps(original["rows"][0], separators=(",", ":"))
    row = ContrastResult.model_validate_json(row_json)
    report = ReadoutResults.model_validate_json(saved_json)
    code = "evidence.p_value.unavailable"
    context = {"metric": "y", "reason": "zero_standard_error"}
    assert row.estimate is not None and row.estimate.value == 2
    assert row.estimate.lb is row.estimate.ub is None
    assert contrast_p_value(row) is contrast_stat_sig(row) is None
    assert not row.sampling_available
    assert (row.sampling_reason_code, row.sampling_reason_context) == (code, context)
    assert row.model_dump_json() == row_json
    assert json.loads(report.model_dump_json()) == original

    (frame,) = report.to_frame(backend="polars").to_dicts()
    (table,) = estimates_to_readout(report)
    for projection in (frame, table):
        assert projection["point_available"] and not projection["sampling_available"]
        assert projection["decision_scope_complete"] is False
        assert projection["failure_code"] == code
        failure_context = projection["failure_context"]
        assert isinstance(failure_context, str)
        assert json.loads(failure_context) == context
        assert (
            projection["required_cell_count"],
            projection["available_cell_count"],
            projection["unavailable_cell_count"],
        ) == (1, 0, 1)
    assert frame["estimate"] == table["lift"] == 2
    assert table["lower"] is table["higher"] is table["level"] is table["stat_sig"] is None
    bare_frame = ContrastResults([row]).to_frame(backend="polars")
    assert isinstance(bare_frame, pl.DataFrame)
    (bare,) = bare_frame.to_dicts()
    assert bare["decision_scope_complete"] is False
    assert bare["point_available"] and not bare["sampling_available"]

    store_path = copytree(fixture / "store", tmp_path / "store")
    (stored_path,) = store_path.glob("*.parquet")
    original_bytes = stored_path.read_bytes()
    import pyarrow.parquet as pq

    manifest = json.loads(pq.read_schema(stored_path).metadata[b"increment.run_manifest.v2"])
    store = ParquetResultsStore(store_path)
    handle = store.open_run(
        manifest["experiment_id"], manifest["analysis_spec_hash"], manifest["run_id"]
    )
    assert (
        handle.run.manifest_hash,
        handle.run.registration_hash,
        handle.run.execution_identity_hash,
    ) == (
        manifest["manifest_hash"],
        manifest["registration_hash"],
        manifest["execution_identity_hash"],
    )
    envelope = handle.envelope()
    (stored_cell,) = envelope.cells
    assert stored_cell.identity.hash() == manifest["cell_scope_hashes"][0][0]
    assert stored_cell.row_payload is not None
    assert stored_cell.row_payload["data"] == row.model_dump(mode="json")
    (cell,) = ContrastResults([row]).as_cells()
    for restored in (cell, stored_cell, AnalysisEnvelope.from_arrow(envelope.to_arrow()).cells[0]):
        status, _ = restored.components["sampling"]
        assert (status.state, status.code, status.context) == ("unavailable", code, context)
        assert restored.summary is not None and restored.summary.point == 2
        assert restored.summary.lower is restored.summary.upper is restored.summary.alpha is None
        assert restored.summary.reason == "zero_standard_error"
    (query,) = store.query(
        manifest["experiment_id"], manifest["analysis_spec_hash"], manifest["run_id"]
    ).to_pylist()
    assert query["point"] == 2 and query["reason_code"] == code
    assert query["lower"] is query["upper"] is query["alpha"] is None
    for rendered in (
        str(report),
        render(envelope, audience="brief"),
        render(envelope, audience="full"),
        readout_table(envelope).gt().as_raw_html(),
    ):
        assert code in rendered or "zero_standard_error" in rendered
        assert "95%" not in rendered and "95.0%" not in rendered
    assert stored_path.read_bytes() == original_bytes


@pytest.mark.integration
def test_legacy_envelope_scope_preserves_payload_across_portable_views():
    import pyarrow.parquet as pq

    from increment.estimation.readout_types import ReadoutMetadata
    from increment.result_envelope import AnalysisEnvelope

    fixture = Path(__file__).parent / "fixtures" / "switchback_legacy"
    (path,) = (fixture / "store").glob("*.parquet")
    original_bytes = path.read_bytes()
    table = pq.read_table(path)
    manifest = json.loads(table.schema.metadata[b"increment.run_manifest.v2"])
    saved_metadata = json.loads(table.schema.metadata[b"increment.readout_metadata"])
    handle = ParquetResultsStore(fixture / "store").open_run(
        manifest["experiment_id"], manifest["analysis_spec_hash"], manifest["run_id"]
    )
    envelope = handle.envelope()
    borrowed = ReadoutMetadata.from_saved_payload(saved_metadata)
    borrowed_json = borrowed.model_dump_json()
    payload = envelope.to_payload(include_capture_metadata=True)
    restored_payload = AnalysisEnvelope.from_payload(payload)
    report = ReadoutResults.model_validate_json((fixture / "base-readout.json").read_text())
    for metadata in (
        report.metadata,
        handle.run.metadata,
        envelope.metadata,
        restored_payload.metadata,
        AnalysisEnvelope.from_arrow(table).metadata,
        AnalysisEnvelope.from_arrow(envelope.to_arrow()).metadata,
        AnalysisEnvelope(envelope.cells, borrowed).metadata,
    ):
        assert metadata is not None
        assert metadata.scope.decision_complete("assigned") is False
    assert borrowed.scope.decision_complete("assigned") is True
    assert borrowed.model_dump_json() == borrowed_json
    assert payload["metadata"] == saved_metadata
    assert restored_payload.to_payload(include_capture_metadata=True) == payload
    assert handle.run.manifest_hash == manifest["manifest_hash"]
    assert path.read_bytes() == original_bytes


@pytest.mark.integration
@pytest.mark.parametrize("destination", ["parquet", "warehouse"])
def test_copying_legacy_run_preserves_payload_and_current_sampling(tmp_path, destination):
    import ibis
    import pyarrow.parquet as pq

    from increment.store import WarehouseResultsStore

    fixture = Path(__file__).parent / "fixtures" / "switchback_legacy" / "store"
    (path,) = fixture.glob("*.parquet")
    manifest = json.loads(pq.read_schema(path).metadata[b"increment.run_manifest.v2"])
    key = (manifest["experiment_id"], manifest["analysis_spec_hash"], manifest["run_id"])
    original = ParquetResultsStore(fixture).open_run(*key)
    backend = ibis.duckdb.connect() if destination == "warehouse" else None
    try:
        store = (
            WarehouseResultsStore(backend)
            if backend is not None
            else ParquetResultsStore(tmp_path / "copied")
        )
        store.put(original.run)
        copied = store.open_run(*key)
        assert copied.run.manifest_hash == original.run.manifest_hash
        assert copied.run.registration_hash == original.run.registration_hash
        assert copied.run.expected_roster == original.run.expected_roster
        envelope = copied.envelope()
        assert copied.run.metadata is not None and envelope.metadata is not None
        assert not copied.run.metadata.scope.decision_complete("assigned")
        assert not envelope.metadata.scope.decision_complete("assigned")
        (cell,) = envelope.cells
        assert cell.to_payload() == original.envelope().cells[0].to_payload()
        assert cell.components["sampling"][0].state == "unavailable"
        assert cell.summary is not None and cell.summary.point == 2
        assert cell.summary.alpha is None and cell.summary.reason == "zero_standard_error"
        (query,) = store.query(*key).to_pylist()
        assert query["point"] == 2 and query["alpha"] is None
        assert query["sampling"]["status"] == "unavailable"
        assert query["reason_code"] == "evidence.p_value.unavailable"
    finally:
        if backend is not None:
            backend.disconnect()


def _assert_selected_legacy_scope(selected):
    assert selected.metadata is not None
    assert selected.metadata.scope.decision_complete("assigned") is False
    (frame,) = selected.to_frame(backend="polars").to_dicts()
    (table,) = estimates_to_readout(selected)
    assert frame["estimate"] == table["lift"] == 1.5
    for projection in (frame, table):
        assert projection["sampling_available"] and projection["point_available"]
        standard_error = projection["standard_error"]
        assert isinstance(standard_error, (int, float)) and standard_error > 0
        assert projection["failure_code"] is None
        assert projection["decision_scope_complete"] is False
        assert (
            projection["required_cell_count"],
            projection["available_cell_count"],
            projection["unavailable_cell_count"],
            projection["hidden_cell_count"],
        ) == (2, 1, 1, 1)
        hidden_cells = projection["hidden_cells"]
        assert isinstance(hidden_cells, str)
        (unavailable,) = json.loads(hidden_cells)
        assert unavailable["cell"]["metric"] == "y"
        assert unavailable["code"] == "evidence.p_value.unavailable"
        assert unavailable["context"] == {"metric": "y", "reason": "zero_standard_error"}


def _assert_rejoined_legacy_scope(combined):
    assert combined.metadata is not None
    assert combined.metadata.scope.decision_complete("assigned") is False
    rows = {row["metric"]: row for row in estimates_to_readout(combined)}
    assert rows["healthy"]["sampling_available"] and not rows["y"]["sampling_available"]
    assert rows["healthy"]["decision_scope_complete"] is False
    assert rows["y"]["decision_scope_complete"] is False


def _assert_contrast_concat_collision(left, right):
    from increment.errors import InvalidRequestError

    for first, second in ((left, right), (right, left)):
        with pytest.raises(InvalidRequestError) as caught:
            first.concat(second)
        assert caught.value.code == "readout.collection.concat_cell_collision"


def _assert_unrelated_source_keeps_sampling(report, unrelated):
    foreign_source = unrelated[0].source_snapshot_id
    for combined in (report.concat(unrelated), unrelated.concat(report)):
        (foreign,) = [
            row
            for row in estimates_to_readout(combined)
            if row["source_snapshot_id"] == foreign_source
        ]
        assert foreign["sampling_available"] and foreign["decision_scope_complete"] is True
        hidden = foreign["hidden_cells"]
        assert isinstance(hidden, str)
        assert json.loads(hidden)[0]["code"] is None


def _assert_borrowed_scope_is_unchanged(report, saved):
    borrowed = type(report.metadata).from_saved_payload(json.loads(saved)["metadata"])
    borrowed_json = borrowed.model_dump_json()
    rebuilt = type(report)(report, metadata=borrowed, source=report.source)
    assert rebuilt.metadata.scope == report.metadata.scope
    assert borrowed.scope.decision_complete("assigned") is True
    assert borrowed.model_dump_json() == borrowed_json


def test_filtering_legacy_unavailable_primary_keeps_healthy_sampling_and_incomplete_scope():
    fixture = Path(__file__).parent / "fixtures" / "switchback_legacy"
    saved = (fixture / "base-mixed-readout.json").read_text()
    report = ReadoutResults.model_validate_json(saved)
    original = report.model_dump_json()
    healthy_index = next(i for i, row in enumerate(report) if row.metric == "healthy")
    filtered = report.filter(lambda row: row.metric == "healthy")
    sliced = report[healthy_index : healthy_index + 1]
    restored = ReadoutResults.model_validate_json(filtered.model_dump_json())
    assert report.metadata is not None and filtered.metadata is not None
    original_scope = report.metadata.scope
    assert original_scope.decision_complete("assigned") is False
    reset = filtered.metadata.model_copy(update={"cells": report.metadata.cells})
    assert reset.scope.decision_complete("assigned") is True
    for selected in (filtered, sliced, restored):
        assert selected.metadata.scope == original_scope
        _assert_selected_legacy_scope(selected)
    _assert_borrowed_scope_is_unchanged(report, saved)
    assert report.metadata.scope is original_scope
    assert report.model_dump_json() == original
    assert json.loads(original) == json.loads(saved)


def test_legacy_sampling_knowledge_merges_across_full_and_hidden_views():
    # Public baseline partials retain no sampling marker for hidden cells. The source named
    # "independent" has treatment y=(1, 2), healthy=(3, 4), so both primary tests are valid.
    fixture = Path(__file__).parent / "fixtures" / "switchback_legacy"
    full = ReadoutResults.model_validate_json((fixture / "base-mixed-readout.json").read_text())
    partial_json = (fixture / "base-partial-readout.json").read_text()
    old_partial = ReadoutResults.model_validate_json(partial_json)
    new_partial = full.filter(lambda row: row.metric == "healthy")
    restored_partial = ReadoutResults.model_validate_json(new_partial.model_dump_json())
    unrelated_json = (fixture / "base-unrelated-partial-readout.json").read_text()
    unrelated = ReadoutResults.model_validate_json(unrelated_json)
    views = (full, old_partial, new_partial, restored_partial)
    originals = tuple(view.model_dump_json() for view in (*views, unrelated))
    for left in views:
        _assert_unrelated_source_keeps_sampling(left, unrelated)
        for right in views:
            combined = left.concat(right)
            if left is full or right is full:
                _assert_rejoined_legacy_scope(combined)
            elif left is old_partial and right is old_partial:
                assert combined.metadata.scope.decision_complete("assigned") is True
            else:
                _assert_selected_legacy_scope(combined)
    for sampling in (
        {
            "available": False,
            "reason_code": "evidence.p_value.unavailable",
            "reason_context": {"metric": "y", "reason": "insufficient_replicates"},
        },
        {"available": True},
    ):
        conflicting = json.loads(new_partial.model_dump_json())
        for record in conflicting["metadata"]["cells"]:
            if record["cell"]["metric"] == "y":
                record["sampling"] = sampling
        other = ReadoutResults.model_validate_json(json.dumps(conflicting))
        for known in (full, new_partial, restored_partial):
            _assert_contrast_concat_collision(known, other)
    conflicting_rows = json.loads(old_partial.model_dump_json())
    conflicting_rows["rows"][0]["estimate"]["value"] += 1
    _assert_contrast_concat_collision(
        full, ReadoutResults.model_validate_json(json.dumps(conflicting_rows))
    )
    assert tuple(view.model_dump_json() for view in (*views, unrelated)) == originals
    assert json.loads(old_partial.model_dump_json()) == json.loads(partial_json)
    assert json.loads(unrelated.model_dump_json()) == json.loads(unrelated_json)


@pytest.mark.integration
def test_all_rank_deficient_cuped_readout_roundtrips_without_an_estimate(tmp_path):
    experiment, reader = _experiment(
        [
            MetricSpec(
                name="adjusted",
                type="mean",
                value_column="y",
                decision_method=Method.cuped(covariates="constant"),
            ),
        ]
    )
    store = ParquetResultsStore(tmp_path)
    with reader:
        saved = run(experiment, reader, results_store=store)
    envelope = store.open_run("retained", saved.run.analysis_spec_hash, saved.run.run_id).envelope()
    (cell,) = envelope.cells
    assert cell.summary is not None and cell.summary.point is None
    assert cell.summary.reason == "rank_deficient"
    restored = ReadoutResults.model_validate_json(saved.readout().model_dump_json())
    (row,) = estimates_to_readout(restored)
    assert row["method"] == "switchback_cuped_jackknife_t"
    assert row["lift"] is row["standard_error"] is row["dof"] is row["stat_sig"] is None
    context = row["failure_context"]
    assert isinstance(context, str) and json.loads(context)["reason"] == "rank_deficient"
    assert row["available_cell_count"] == 0 and row["required_cell_count"] == 1


def test_changing_frozen_forecast_changes_method_identity_even_for_equal_values():
    identities = []
    for column in ("prior_position", "same_position"):
        experiment, reader = _experiment(
            [
                MetricSpec(
                    name="adjusted",
                    type="mean",
                    value_column="y",
                    decision_method=Method.cuped(covariates=column),
                ),
            ]
        )
        with reader:
            (cell,) = run(experiment, reader).cells
        assert cell.summary is not None and cell.summary.point == pytest.approx(7)
        identities.append(cell.identity.method_config_hash)
    assert all(identities) and identities[0] != identities[1]


@pytest.mark.parametrize(
    ("kind", "n", "steps", "sign", "variance", "df"),
    [
        ("ratio", 8, 3, 1, 27 / 112, 7),
        ("adjusted", 12, 4, -1, 11 / 45, 10),
    ],
)
def test_switchback_guardrail_tests_unrounded_effect(kind, n, steps, sign, variance, df):
    from scipy.stats import t

    margin = 0.01
    unit = math.ulp(margin)
    rows = [
        {
            "u": str(i),
            "c": 0,
            "p": p,
            "s": 0,
            "arm": "B" if p == i % 2 else "A",
            "one": 1,
            "y": sign * (margin - (steps * unit if i >= n - 2 else 0)) if p == i % 2 else 0.0,
        }
        for i in range(n)
        for p in range(2)
    ]
    metric = (
        MetricSpec(
            name=kind,
            type="ratio",
            numerator="y",
            denominator="one",
            preferred_direction="decrease",
        )
        if kind == "ratio"
        else MetricSpec(
            name=kind,
            value_column="y",
            decision_method=Method.cuped(covariates="prior_position"),
            preferred_direction="increase",
        )
    )
    experiment, reader = _paired_experiment(
        rows,
        [metric],
        AnalysisPlan(guardrails=[ExperimentMetric(metric=kind, margin_abs=margin)]),
        with_forecasts=kind == "adjusted",
    )
    with reader:
        analyzed = run(experiment, reader)
    # A unit denominator gives the raw mean. With forecasts +/-1 the intercept
    # averages the two position means. One exceptional unit in each position
    # moves either estimate by 2*steps/n ULPs, not the single ULP displayed.
    expected_p = t.sf((2 * steps / n) / math.sqrt(variance), df)
    restored = ReadoutResults.model_validate_json(analyzed.readout().model_dump_json())
    (row,) = estimates_to_readout(restored)
    assert row["lift"] == sign * (margin - unit) and row["null_abs"] == sign * margin
    assert row["standard_error"] == pytest.approx(unit * math.sqrt(variance), abs=0.0)
    assert row["p_value"] == pytest.approx(expected_p)
    assert row["stat_sig"] is False and analyzed.decisions.cells[0].status == "retain"


@pytest.mark.parametrize(
    ("numerators", "point", "variance", "alpha"),
    [
        pytest.param((0, 2, 2, 4), 2, 2 / 3, 0.1, id="null-difference-overflows"),
        pytest.param((0, 0, 2, 2), 1, 1 / 3, 0.0025, id="half-width-overflows"),
    ],
)
def test_directional_ratio_tail_recovers_a_representable_standardized_margin(
    numerators, point, variance, alpha
):
    from fractions import Fraction
    from math import sqrt

    from scipy.stats import t

    from increment.decision import ContrastDecisionProcedure
    from increment.estimation.contrast import SwitchbackMetricStats, estimate_contrast
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.results import contrast_stat_sig

    # Express the independent reference in units of 2**1022; both endpoints
    # of the comparison are finite, but the intermediate arithmetic need not be.
    unit = 2**1022
    stats = SwitchbackMetricStats(
        metric="rate",
        aggregation="ratio",
        probability_ct=0.5,
        randomization_law="independent_bernoulli_order",
        independence_grain="unit_cycle",
        n_units=4,
        n_cycles=4,
        control_group="A",
        treatment_group="B",
        observation_steps=1,
        carryover_order=0,
        retained_steps=1,
        ratio_components=tuple(
            (Fraction(v * unit), Fraction(1), Fraction(0), Fraction(1)) for v in numerators
        ),
    )
    procedure = ContrastDecisionProcedure(
        metric="rate",
        role="primary",
        alpha=alpha,
        alternative="greater",
        null_abs=(point - 4.5) * unit,
    )
    computed = estimate_contrast(stats, procedure)
    (row,) = computed.results
    (evidence,) = computed.evidence.values()
    assert isinstance(evidence, PValueEvidence)
    assert evidence.p_value == pytest.approx(t.sf(4.5 / sqrt(variance), 3))
    assert contrast_stat_sig(row) is True
    assert row.estimate is not None and row.estimate.lb is not None and row.estimate.ub is None
    assert row.estimate.lb / unit == pytest.approx(point - t.isf(alpha, 3) * sqrt(variance))


def test_zero_denominator_estimator_returns_only_the_keyed_failure():
    from increment.decision import ContrastDecisionProcedure
    from increment.declarations import ContrastHypothesisKey
    from increment.estimation.contrast import SwitchbackMetricStats, estimate_contrast

    stats = SwitchbackMetricStats.model_validate(
        {
            "metric": "sales",
            "aggregation": "ratio",
            "probability_ct": 0.5,
            "randomization_law": "independent_bernoulli_order",
            "independence_grain": "unit_cycle",
            "n_units": 2,
            "n_cycles": 2,
            "control_group": "old",
            "treatment_group": "new",
            "observation_steps": 1,
            "carryover_order": 0,
            "retained_steps": 1,
            "ratio_components": [(1, 1, 3, 0), (3, 1, 3, 0)],
        }
    )
    computed = estimate_contrast(
        stats,
        ContrastDecisionProcedure(
            metric="sales",
            role="primary",
            alpha=0.05,
            alternative="two-sided",
            null_abs=0,
        ),
    )
    assert not computed.results and not computed.evidence
    (key,) = computed.failures
    assert key == ContrastHypothesisKey("sales", "old", "new")
    assert computed.failures[key].context == {
        "metric": "sales",
        "reason": "zero_denominator",
        "arm": "old",
    }


@pytest.mark.parametrize("context", [None, {"metric": "sales", "reason": "unobserved_response"}])
def test_absent_measurements_require_the_recorded_missing_step(context):
    from increment.errors import InvalidRequestError
    from increment.results import ContrastResult

    with pytest.raises(InvalidRequestError) as caught:
        ContrastResult.model_validate(
            {
                "metric": "sales",
                "aggregation": "ratio",
                "probability_ct": 0.5,
                "method": "switchback_ratio_delta_t",
                "reference": "switchback_ratio_delta_t",
                "estimand": "retained_window_ratio_difference",
                "role": "primary",
                "alternative": "two-sided",
                "alpha": 0.05,
                "null_abs": 0,
                "randomization_law": "independent_bernoulli_order",
                "independence_grain": "unit_cycle",
                "n_units": 2,
                "n_cycles": 2,
                "control_group": "old",
                "treatment_group": "new",
                "observation_steps": 1,
                "carryover_order": 0,
                "retained_steps": 1,
                "estimate": None,
                "standard_error": None,
                "dof": None,
                "failure_code": "source.frame.switchback.response_unobserved" if context else None,
                "failure_context": context,
            }
        )
    assert caught.value.code == "estimation.contrast.reference_metadata"
