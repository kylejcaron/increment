"""Behavioral contracts for portable, source-scoped readout results."""

import pickle
from copy import deepcopy

from increment.estimation.decision_types import EValueEvidence
from increment.estimation.readout_types import CellKey, IntegrityResult, StreamingDigest


def test_streaming_digest_is_order_independent_but_counts_duplicates():
    rows = [{"id": "a", "value": 1.0}, {"id": "b", "value": None}]
    forward = StreamingDigest()
    reverse = StreamingDigest()
    for row in rows:
        forward.update(row)
    for row in reversed(rows):
        reverse.update(row)
    assert forward.hexdigest() == reverse.hexdigest()
    reverse.update(rows[0])
    assert forward.hexdigest() != reverse.hexdigest()


def test_metadata_context_is_deeply_immutable_and_portable():
    result = IntegrityResult(
        status="not_applicable",
        analysis_population="assigned",
        construction="none",
        alpha=None,
        observed=None,
        expected=None,
        randomization_grain=None,
        code=None,
        context={"nested": {"values": [1, 2]}},
    )
    assert result.context["nested"]["values"] == (1, 2)
    assert deepcopy(result) == result
    assert pickle.loads(pickle.dumps(result)) == result


def test_cell_identity_retains_sampling_axes():
    base = {
        "kind": "arm",
        "metric": "rev",
        "method": "unadjusted",
        "method_role": "decision",
        "estimand": "itt",
        "analysis_population": "assigned",
        "group_id": "treatment",
    }
    assert CellKey.model_validate({**base, "value_scale": "relative"}) != CellKey.model_validate(
        {**base, "value_scale": "absolute"}
    )
    assert CellKey.model_validate({**base, "inference": "fixed"}) != CellKey.model_validate(
        {**base, "inference": "always_valid"}
    )


def test_sampling_inference_accepts_zero_likelihood_evidence():
    from math import inf

    from increment.estimation.readout_types import SamplingInference
    from tests.estimation.test_sequential_public_proof_acceptance import _public_family

    _, _, bundle, _ = _public_family(0, 0, 0, 0, "two-sided")
    missing_row = next(row for row in bundle.results if row.group_id == "never-enrolled")
    evidence = next(
        item
        for item in bundle.evidence.values()
        if item.hypothesis.group_id == missing_row.group_id
    )
    assert isinstance(evidence, EValueEvidence)

    sampling = SamplingInference(available=True, evidence=evidence)
    restored = SamplingInference.model_validate_json(sampling.model_dump_json())

    assert restored.available is True
    assert restored.evidence is not None
    assert isinstance(restored.evidence, EValueEvidence)
    assert restored.evidence.log_e == -inf


def test_triggered_sequential_scope_uses_explicit_unsupported_placeholder(tmp_path):
    from datetime import date

    import pyarrow.parquet as pq

    from increment import Analysis
    from increment.estimation.readout_types import ReadoutResults
    from increment.estimation.results import LiftEstimate
    from increment.frame import MetricSpec
    from increment.semantics.design import Randomized
    from tests.test_sequential_public_sources import _native_fixture

    _connection, definitions, analysis = _native_fixture("bernoulli", triggered=True)
    try:
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        results = analysis.run()
        assert results.metadata is not None
        by_population = {}
        for row in results:
            assert isinstance(row, LiftEstimate)
            by_population[row.analysis_population] = row
        assert set(by_population) == {"assigned", "triggered"}
        assigned = by_population["assigned"]
        placeholder = by_population["triggered"]
        assert placeholder.failure_code == "readout.cell.unsupported_request"
        assert placeholder.failure_context["reason"] == "triggered_sequential"
        assert placeholder.inference == "always_valid"
        assert placeholder.reference_kind == "sequential"
        assert placeholder.sequential_result is None
        assert placeholder.sampling_available is False
        assert (
            placeholder.estimand,
            placeholder.value_scale,
            placeholder.alternative,
        ) == (assigned.estimand, assigned.value_scale, assigned.alternative)
        restored = ReadoutResults.model_validate_json(results.model_dump_json())
        assert restored.metadata == results.metadata

        path = tmp_path / "sequential-moments.parquet"
        analysis.export(path)
        wire_rows = pq.read_table(path).to_pylist()
        assert wire_rows[0]["trigger_name"] == "triggered"
        replay = Analysis.from_moments(
            wire_rows,
            metrics=[MetricSpec(name="outcome", type="conversion", window_days=2)],
            design=Randomized(control_group="control"),
            plan=definitions.experiments[0].plan,
        )
        replayed = replay.run()
        replayed_by_population = {}
        for row in replayed:
            assert isinstance(row, LiftEstimate)
            replayed_by_population[row.analysis_population] = row
        assert set(replayed_by_population) == {"assigned", "triggered"}
        assert replayed.metadata is not None
        assert replayed_by_population["triggered"].failure_context == placeholder.failure_context
        assert replayed.metadata.scope.snapshot_id == results.metadata.scope.snapshot_id
    finally:
        analysis.close()
