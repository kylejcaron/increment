import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pytest

from increment import (
    AnalysisPlan,
    MetricSpec,
    RunResult,
    Spillover,
    SpilloverComparison,
    from_frame,
    run,
    runnable,
)
from increment.errors import CodedError
from increment.kernel.contracts import Registration
from increment.results import ReadoutResults, SpilloverEffects
from increment.store import HistorySelector, ParquetResultsStore, WarehouseResultsStore
from increment.store.types import pinned_run

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


@pytest.fixture(params=["parquet", pytest.param("warehouse", marks=pytest.mark.integration)])
def store(request, tmp_path):
    if request.param == "parquet":
        yield ParquetResultsStore(tmp_path)
    else:
        import ibis

        con = ibis.duckdb.connect()
        try:
            yield WarehouseResultsStore(con)
        finally:
            con.disconnect()


def test_reopened_policy_pairs_keep_direction_and_separate_histories(store):
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


def test_failed_inversion_reopens_with_its_observed_point_and_null_bounds(tmp_path, monkeypatch):
    from increment.estimation.spillover import SpilloverInterval

    observed = SpilloverInterval.observed

    def one_failed_interval(self, comparison, cells):
        point, lower, upper, basis = observed(self, comparison, cells)
        return point, lower, float("nan") if comparison.policy == "a:b" else upper, basis

    monkeypatch.setattr(SpilloverInterval, "observed", one_failed_interval)
    experiment, reader = _bind()
    store = ParquetResultsStore(tmp_path)
    with reader:
        run(experiment, reader, results_store=store, run_id="failed-inversion")
    handle = store.open_run("coupon", experiment.analysis_spec_hash, "failed-inversion")
    restored = RunResult.from_handle(handle, registration=experiment.registration)
    frame = restored.to_frame(backend="pyarrow")
    assert isinstance(frame, pa.Table)
    rows = frame.to_pylist()
    assert rows[0]["estimate"] == pytest.approx(1)
    assert rows[0]["lower"] is rows[0]["upper"] is None
    assert rows[0]["direction"] == rows[0]["range_basis"] == "unavailable"
    assert rows[0]["unavailable_reason"] == "nonfinite_interval"
    assert handle.envelope().to_arrow()["reason_code"].to_pylist() == ["nonfinite_interval", None]
