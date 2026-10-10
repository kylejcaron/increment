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


def test_batched_digest_equals_row_wise_updates_on_every_scalar_shape():
    """The direct scalar encoder and the generic encoder must agree byte for byte."""
    import datetime as dt

    import numpy as np
    import pytest

    from increment._canonical import CanonicalJSONError

    rows = [
        {"unit_id": str(index), "group_id": "control" if index % 2 else "treatment", "y": value}
        for index, value in enumerate(np.random.default_rng(3).lognormal(size=500).tolist())
    ]
    rows += [
        {"unit_id": f"s{index}", "group_id": "t", "y": value}
        for index, value in enumerate(
            (0.0, -0.0, 1.0, 2.5, 1e21, 9.99e20, 1e-6, 9.9e-7, 123456789.123456789, -3.25)
        )
    ]
    rows += [
        {"unit_id": "tiny", "group_id": "t", "y": 5e-324},
        {"unit_id": "huge", "group_id": "t", "y": 1.7e308},
        {
            "b": True,
            "n": None,
            "i": 2**70,
            "neg": -17,
            "s": 'ü " \\ \n \u2028 \U0001f600',
            "f": 7.0,
        },
        {"z": [1, 2, {"a": 1}], "d": dt.datetime(2024, 1, 2, tzinfo=dt.UTC), "k": 3},
        {"unit_id": "1", "group_id": "control", "y": 0.0, "extra": {"nested": [1.5, "x"]}},
        {"x": np.float64(2.5), "i": 4},
        {"b_key": 1, "a_key": 2, "ä": 3, "Z": 4, "z": 5},
        {"only_other_order": 1, "a": 2},
    ]
    row_wise = StreamingDigest()
    for row in rows:
        row_wise.update(row)
    batched = StreamingDigest()
    batched.update_rows(iter(rows))
    assert (batched.accumulator, batched.count) == (row_wise.accumulator, row_wise.count)
    assert batched.hexdigest() == row_wise.hexdigest()
    with pytest.raises(CanonicalJSONError) as refusal:
        StreamingDigest().update_rows([{"y": float("nan")}])
    assert refusal.value.code == "artifact.digest.nonfinite"
    for surrogate_row in ({"s": "\ud800"}, {"s": "ok\udfff"}, {"\udc00": 1}):
        with pytest.raises(CanonicalJSONError) as generic:
            StreamingDigest().update(surrogate_row)
        with pytest.raises(CanonicalJSONError) as direct:
            StreamingDigest().update_rows([surrogate_row])
        assert direct.value.code == generic.value.code == "artifact.digest.json"


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


def test_triggered_sequential_scope_reports_and_replays_both_chains(tmp_path):
    from datetime import date

    import pyarrow.parquet as pq
    import pytest

    from increment import Analysis
    from increment.errors import CapabilityError
    from increment.estimation.readout_types import ReadoutResults
    from increment.estimation.results import LiftEstimate
    from increment.frame import MetricSpec
    from increment.semantics.design import Randomized
    from tests.test_sequential_public_sources import _native_fixture

    _connection, definitions, analysis = _native_fixture("bernoulli", triggered=True)
    try:
        snapshot = analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        results = analysis.run()
        assert results.metadata is not None
        by_population = {}
        for row in results:
            assert isinstance(row, LiftEstimate)
            by_population[row.analysis_population] = row
        assert set(by_population) == {"assigned", "triggered"}
        assigned = by_population["assigned"]
        triggered = by_population["triggered"]
        assert triggered.failure_code is None
        assert triggered.inference == "always_valid"
        assert triggered.reference_kind == "sequential"
        assert triggered.require_sequential_result().checkpoint.population == "triggered"
        assert triggered.sampling_available is True
        assert (
            triggered.estimand,
            triggered.value_scale,
            triggered.alternative,
        ) == (assigned.estimand, assigned.value_scale, assigned.alternative)
        restored = ReadoutResults.model_validate_json(results.model_dump_json())
        assert restored.metadata == results.metadata

        path = tmp_path / "sequential-moments.parquet"
        analysis.export(path)
        wire_rows = pq.read_table(path).to_pylist()
        assert wire_rows[0]["trigger_name"] == "triggered"
        specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
        design = Randomized(control_group="control")
        # The stored plan carries the triggered commitment; a caller-supplied plan
        # without it cannot adopt a checkpoint that holds the triggered chain.
        with pytest.raises(CapabilityError) as uncommitted:
            Analysis.from_moments(
                wire_rows, metrics=specs, design=design, plan=definitions.experiments[0].plan
            )
        assert uncommitted.value.code == "sequential.source.invalid"
        replay = Analysis.from_moments(wire_rows, metrics=specs, design=design)
        try:
            assert replay.sequential_snapshot() == snapshot
            replayed = replay.run()
            replayed_by_population = {}
            for row in replayed:
                assert isinstance(row, LiftEstimate)
                replayed_by_population[row.analysis_population] = row
            assert set(replayed_by_population) == {"assigned", "triggered"}
            assert replayed.metadata is not None
            for population, row in by_population.items():
                assert (
                    replayed_by_population[population].require_sequential_result()
                    == row.require_sequential_result()
                )
            assert {row.analysis_population for row in replay.run(population="triggered")} == {
                "triggered"
            }
        finally:
            replay.close()
    finally:
        analysis.close()
