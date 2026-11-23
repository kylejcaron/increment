import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pytest

from increment import (
    AnalysisPlan,
    Experiment,
    MetricSpec,
    RunResult,
    Spillover,
    SpilloverComparison,
    from_frame,
    run,
    runnable,
)
from increment.errors import CodedError
from increment.estimation.readout_types import SamplingInference
from increment.kernel.contracts import Registration
from increment.results import ReadoutResults, ResultCell, SpilloverEffect, SpilloverEffects
from increment.store import HistorySelector, ParquetResultsStore, Run, WarehouseResultsStore
from increment.store.types import pinned_run
from tests._unit_boundary import backend_fixture
from tests.spillover_trial import frame_trial, legacy_spillover_file, paired_trial

_PAIRS = (("a:b", "c"), ("a", "b:c"))


def _bind(*, purchased=True):
    quotas = {"a:b": 2, "c": 1, "a": 1, "b:c": 1}
    policies = list(quotas) * 2
    people = []
    for group, policy in enumerate(policies):
        for person in range(4):
            offered = person < quotas[policy]
            bought = purchased and policy == "a:b" and not offered
            people.append((group * 4 + person, group, "yes" if offered else "no", bought))
    frame = pd.DataFrame(
        people,
        columns=["person", "store", "offer", "bought"],
    )
    counts = pd.DataFrame(
        [(g, p, count) for g in range(len(policies)) for p, count in quotas.items()],
        columns=["store", "policy", "offers"],
    )
    return from_frame(
        frame,
        shape="summary",
        unit="person",
        group="offer",
        cluster="store",
        design=Spillover(
            control_group="no",
            offered_group="yes",
            groups_per_policy={"all": dict.fromkeys(policies, 2)},
            comparisons=tuple(SpilloverComparison(p, r, 0.1) for p, r in _PAIRS),
        ),
        metrics=[MetricSpec(name="bought", type="conversion", window_days=7)],
        plan=AnalysisPlan(primary="bought", alpha=0.05),
        experiment_id="coupon",
        offer_counts=counts,
        group_policy=pd.DataFrame({"store": range(8), "policy": policies}),
    )


@backend_fixture()
def warehouse_store():
    import ibis

    con = ibis.duckdb.connect()
    try:
        yield WarehouseResultsStore(con)
    finally:
        con.disconnect()


def test_reopened_policy_pairs_keep_direction_and_separate_histories(tmp_path):
    _check_reopened_policy_pairs(ParquetResultsStore(tmp_path))


def _run_upgraded_trial(saved, store_path):
    registered = Experiment.from_json(saved)
    current, reader = frame_trial(*paired_trial())
    assert current.design["method"] == "two-stage-fixed-bet-v3"
    assert current.registration != registered.registration
    store = ParquetResultsStore(store_path)
    with reader:
        replay = run(registered, reader, run_id="after-upgrade", results_store=store)
        fresh = run(current, reader)
    assert registered.to_json() == saved
    fresh_row, replay_row = fresh.readout()[0], replay.readout()[0]
    assert isinstance(fresh_row, SpilloverEffect) and isinstance(replay_row, SpilloverEffect)
    assert fresh_row.lower is not None and replay_row.lower is not None
    assert fresh_row.lower > replay_row.lower
    return (
        replay.readout().model_dump_json(),
        replay.cells[0].identity.hash(),
        fresh.cells[0].identity.method_config_hash,
    )


@pytest.mark.integration
def test_saved_v2_prelaunch_runs_its_original_interval_and_reopens_unchanged(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    registered = Experiment.from_json(legacy_spillover_file("paired-prelaunch.json").read_text())
    completed = ReadoutResults.model_validate_json(
        legacy_spillover_file("paired-readout.json").read_text()
    )
    assert isinstance(completed, SpilloverEffects)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as workers:
        saved, identity, fresh_method = workers.submit(
            _run_upgraded_trial, registered.to_json(), tmp_path
        ).result()
    replay = ReadoutResults.model_validate_json(saved)
    # Completed numbers and their method remain unchanged, even when the
    # registered trial is first measured by a fresh process after an upgrade.
    assert tuple(replay) == tuple(completed)
    assert replay[0].method == "two-stage-fixed-bet-v2"
    store = ParquetResultsStore(tmp_path)
    handle = store.open_run("pilot", registered.analysis_spec_hash, "after-upgrade")
    restored = RunResult.from_handle(handle, registration=registered.registration)
    assert tuple(restored.readout()) == tuple(completed)
    assert restored.cells[0].identity.hash() == identity
    assert restored.cells[0].identity.method_config_hash != fresh_method


@pytest.mark.integration
def test_warehouse_policy_pairs_keep_direction_and_separate_histories(warehouse_store):
    _check_reopened_policy_pairs(warehouse_store)


def _check_reopened_policy_pairs(store):
    experiment, reader = _bind()
    with reader:
        (estimator_id,) = runnable(experiment, "bought")
        live = run(experiment, reader, run_id="measured", results_store=store)
    handle = store.open_run("coupon", experiment.analysis_spec_hash, "measured")
    restored = RunResult.from_handle(handle, registration=experiment.registration)
    assert tuple(restored.readout()) == tuple(live.readout())
    frame = restored.to_frame(backend="pyarrow")
    assert isinstance(frame, pa.Table)
    assert (
        tuple(zip(frame["policy"].to_pylist(), frame["reference"].to_pylist(), strict=True))
        == _PAIRS
    )
    assert frame["own_assignment"].to_pylist() == ["unoffered"] * 2
    assert frame["analysis_population"].to_pylist() == ["assigned"] * 2
    assert frame["estimate"].to_pylist() == pytest.approx([1.0, 0.0])
    assert frame["range_basis"][1].as_py() == "identical_offer_counts"
    assert {"lift", "p_value", "mean_treat", "mean_control"}.isdisjoint(frame.column_names)
    assert restored.decisions.program.status == "not_evaluated"
    for cell, (policy, reference) in zip(restored.cells, _PAIRS, strict=True):
        ident = cell.identity
        assert ident.estimator_id == estimator_id == "spillover.two_stage"
        assert ident.metric_hash and ident.scope
        assert ident.method_config_hash and ident.method_instance_id
        selector = HistorySelector(
            metric_hash=ident.metric_hash,
            estimator_id=ident.estimator_id,
            method_config_hash=ident.method_config_hash,
            method_instance_id=ident.method_instance_id,
            scope_hash=ident.scope.hash(),
            policy=policy,
            reference=reference,
        )
        hist = store.history("coupon", selector).to_pylist()
        assert len(hist) == 1
        key = json.loads(hist[0]["cell"])["identity"]["key"]
        assert (key["policy"], key["reference"], key["own_assignment"]) == (
            policy,
            reference,
            "unoffered",
        )
        assert hist[0]["se"] is hist[0]["df"] is None
    with pytest.raises(CodedError) as wrong:
        RunResult.from_handle(handle, registration=Registration(registration_hash="different"))
    assert wrong.value.code == "store.decision.reference_invalid"


def test_capture_identifies_reduced_outcomes_not_the_registration_or_run_name():
    experiment, reader = _bind()
    with reader:
        first = run(experiment, reader, run_id="first")
        repeat = run(experiment, reader, run_id="repeat")
    changed, reader = _bind(purchased=False)
    assert changed.registration == experiment.registration
    with reader:
        later = run(experiment, reader, run_id="first")
    assert first.metadata and repeat.metadata and later.metadata
    assert first.metadata.scope.snapshot_id == repeat.metadata.scope.snapshot_id
    assert first.metadata.scope.snapshot_id != later.metadata.scope.snapshot_id
    assert first.metadata.source_content_digest is None
    for other in (repeat, later):
        with pytest.raises(CodedError) as mismatch:
            first.readout().concat(other.readout())
        assert mismatch.value.code == "readout.collection.concat_metadata_mismatch"


def test_partial_readouts_round_trip_and_rejoin_in_registered_order():
    experiment, reader = _bind()
    with reader:
        readout = run(experiment, reader, run_id="joined").readout()
    assert isinstance(readout, SpilloverEffects)
    first = readout.filter(lambda row: row.policy == _PAIRS[0][0])
    second = readout[1:]
    assert first.metadata.partial_reason == ("filter",)
    assert second.metadata.partial_reason == ("slice",)
    assert first[0].family_size == second[0].family_size == 2
    frame = second.to_frame(backend="pyarrow")
    assert isinstance(frame, pa.Table)
    assert frame["view_partial"].to_pylist() == [True]
    portable = ReadoutResults.model_validate_json(second.model_dump_json())
    assert isinstance(portable, SpilloverEffects)
    assert portable.rows == second.rows and portable.metadata == second.metadata
    combined = portable.concat(first)
    assert combined.rows == readout.rows
    assert combined.metadata == readout.metadata
    empty = readout[:0]
    restored = ReadoutResults.model_validate_json(empty.model_dump_json())
    assert isinstance(restored, SpilloverEffects)
    assert not restored and restored.metadata.partial
    assert restored.metadata.registration_hash == experiment.registration.registration_hash
    for collection in (readout, first):
        for field, value in (("alpha", 1e-9), ("window_days", 365)):
            payload = json.loads(collection.model_dump_json())
            payload["rows"][0][field] = value
            with pytest.raises(CodedError) as mismatch:
                ReadoutResults.model_validate_json(json.dumps(payload))
            assert mismatch.value.code == "readout.serialization.identity_mismatch"


def test_reopen_refuses_family_order_and_metadata_from_another_run(tmp_path):
    store = ParquetResultsStore(tmp_path)
    experiment, reader = _bind()
    with reader:
        run(experiment, reader, run_id="measured", results_store=store)
    record = store.open_run("coupon", experiment.analysis_spec_hash, "measured").run
    assert record.metadata is not None
    for changed, code in (
        (
            replace(record, expected_roster=record.expected_roster[::-1]),
            "store.run.roster_mismatch",
        ),
        (
            replace(record, metadata=record.metadata.model_copy(update={"run_id": "different"})),
            "store.run.metadata_mismatch",
        ),
    ):
        with pytest.raises(CodedError) as mismatch:
            RunResult.from_handle(pinned_run(changed), registration=experiment.registration)
        assert mismatch.value.code == code


def test_stored_unavailable_interval_reopens_without_losing_the_observed_point(tmp_path):
    experiment, reader = _bind()
    with reader:
        result = run(experiment, reader, run_id="failed-inversion")
    readout = result.readout()
    assert isinstance(readout, SpilloverEffects)
    failed = SpilloverEffect.model_validate(
        {**readout[0].model_dump(), "lower": None, "upper": None}
    )
    record = result.run
    assert isinstance(record, Run)
    first, second = record.cells
    failed_cell = ResultCell.from_report_payload(
        row_kind="spillover",
        identity=first.identity,
        label=first.label,
        row=failed,
        family=first.family,
    )
    sampling = SamplingInference(available=False, reason_code="nonfinite_interval")
    records = tuple(
        item.model_copy(update={"sampling": sampling}) if item.cell == first.identity.key else item
        for item in readout.metadata.cells
    )
    store = ParquetResultsStore(tmp_path)
    store.put(
        replace(
            record,
            cells=(replace(failed_cell, provenance=first.provenance), second),
            metadata=readout.metadata.model_copy(update={"cells": records}),
        )
    )
    handle = store.open_run("coupon", experiment.analysis_spec_hash, "failed-inversion")
    restored = RunResult.from_handle(handle, registration=experiment.registration)
    frame = restored.to_frame(backend="pyarrow")
    assert isinstance(frame, pa.Table)
    unavailable, other = frame.to_pylist()
    assert (unavailable["policy"], unavailable["reference"]) == _PAIRS[0]
    # Every unoffered customer under a:b bought; none under c did.
    assert unavailable["estimate"] == pytest.approx(1)
    assert unavailable["lower"] is unavailable["upper"] is None
    assert unavailable["direction"] == unavailable["range_basis"] == "unavailable"
    assert unavailable["unavailable_reason"] == "nonfinite_interval"
    assert (other["policy"], other["reference"]) == _PAIRS[1]
    assert (other["estimate"], other["lower"], other["upper"]) == pytest.approx((0, 0, 0))
    assert other["direction"] == "inconclusive"
    assert other["range_basis"] == "identical_offer_counts"
    assert other["unavailable_reason"] is None
    assert handle.envelope().to_arrow()["reason_code"].to_pylist() == ["nonfinite_interval", None]
