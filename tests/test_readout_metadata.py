"""Behavioral contracts for portable, source-scoped readout results."""

import pickle
from copy import deepcopy

import pytest

from increment import from_frame, from_moments
from increment.estimation.decision_types import EValueEvidence
from increment.estimation.readout_types import CellKey, IntegrityResult, StreamingDigest
from increment.workflows import capture_sequential, export_moments, sequential_snapshot
from tests.analysis_factory import view_readout


def test_readout_metadata_repr_shows_snapshot_and_record_count():
    from types import SimpleNamespace

    from increment.estimation.readout_types import ReadoutMetadata

    metadata = ReadoutMetadata.model_construct(
        scope=SimpleNamespace(snapshot_id="snapshot-1"),
        cells=(),
        partial=False,
        partial_reason=(),
    )
    display = repr(metadata)
    assert "\n" in display and "ReadoutMetadata(" in display
    assert "snapshot-1" in display and "cells=0" in display
    assert not any(token in display for token in ("0x", "sha256:", "mappingproxy", "_v1"))


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


def test_frame_digest_matches_canonical_streaming_encoder_for_edge_values():
    import math

    import polars as pl

    from increment.readouts._metric_rows import _frame_digest_and_arms

    frames = [
        pl.DataFrame(
            {
                "group_id": ["a", "a", "b", "b", "b"],
                "value": [None, float("nan"), float("inf"), float("-inf"), 2.0],
            }
        ),
        pl.DataFrame({"group_id": ["a", "b"], "value": [0.0, -0.0]}),
        pl.DataFrame({"group_id": ["a", "b"], "value": ["café", "雪"]}),
        pl.DataFrame(
            {
                "group_id": ["a"],
                "bool": [True],
                "integer": [1],
                "float": [1.0],
            }
        ),
        pl.DataFrame({"group_id": [0.0, -0.0]}),
        pl.DataFrame({"group_id": [[1, 2], [3]]}),
        pl.DataFrame({"group_id": ["a", "a", "a"], "value": [3.0, 3.0, 3.0]}),
        pl.DataFrame({"group_id": ["a", "b"], "z": [2, 4], "a": [1, 3]}),
    ]

    def canonical_reference(frame):
        digest = StreamingDigest()
        for row in frame.iter_rows(named=True):
            normalized = {}
            for name, value in row.items():
                if isinstance(value, float) and math.isnan(value):
                    value = None
                elif isinstance(value, float) and math.isinf(value):
                    value = {"__float__": "+inf" if value > 0 else "-inf"}
                normalized[name] = value
            digest.update(normalized)
        return digest.hexdigest(), digest.count

    for frame in frames:
        arms, digest, count = _frame_digest_and_arms(frame)
        assert (digest, count) == canonical_reference(frame)
        assert arms == frozenset(
            str(value) for value in frame["group_id"].to_list() if value is not None
        )

    # Canonical map encoding makes physical column order irrelevant.
    reordered = pl.DataFrame({"group_id": ["a", "b"], "a": [1, 3], "z": [2, 4]})
    for frame in (frames[-1], reordered):
        assert _frame_digest_and_arms(frame)[1:] == canonical_reference(frame)
    assert _frame_digest_and_arms(frames[-1])[1] == _frame_digest_and_arms(reordered)[1]

    # The digest is a multiset: duplicates count, but row permutation does not.
    forward = frames[-1]
    reverse = forward.reverse()
    assert _frame_digest_and_arms(forward)[1:] == canonical_reference(forward)
    assert _frame_digest_and_arms(reverse)[1:] == canonical_reference(reverse)
    assert _frame_digest_and_arms(forward)[1] == _frame_digest_and_arms(reverse)[1]


def test_raw_moment_row_digest_is_pinned_and_encodes_nonfinite_moments():
    from increment.readouts._metric_rows import _rows_raw_digest_and_arms

    finite = (
        {
            "metric": "m",
            "group_id": "c",
            "n": 40,
            "ref_y": 1.5,
            "cy1": -0.0,
            "cy2": 2.25e-300,
            "cden3": None,
            "x_role": None,
        },
        {
            "metric": "m",
            "group_id": "t",
            "n": 2**60,
            "ref_y": 1e300,
            "cy1": 0.1,
            "cy2": 1.7976931348623157e308,
            "cden3": 5e-324,
            "x_role": "covariate",
        },
    )
    assert _rows_raw_digest_and_arms(finite) == (
        frozenset({"c", "t"}),
        "e7e31b58fa1e058a78dd2fa1360c08edbcab18ebc5e7819f3743efc1bb3d4d6f",
        2,
    )

    def reference(rows):
        digest = StreamingDigest()
        for row in rows:
            digest.update(
                {
                    key: None
                    if value != value
                    else {"__float__": "+inf" if value > 0 else "-inf"}
                    if isinstance(value, float) and value in (float("inf"), float("-inf"))
                    else value
                    for key, value in row.items()
                }
            )
        return digest.hexdigest(), digest.count

    overflowed = (
        {**finite[0], "cy2": float("inf"), "cden3": float("-inf")},
        {**finite[1], "cden3": float("nan")},
    )
    arms, digest, count = _rows_raw_digest_and_arms(overflowed)
    assert (arms, (digest, count)) == (frozenset({"c", "t"}), reference(overflowed))
    assert digest != _rows_raw_digest_and_arms(({**finite[0], "cy2": float("-inf")}, finite[1]))[1]
    assert (
        _rows_raw_digest_and_arms((finite[0], {**finite[1], "cden3": float("nan")}))[1]
        == (_rows_raw_digest_and_arms((finite[0], {**finite[1], "cden3": None}))[1])
    )


def test_categorical_missing_digest_is_stable_for_unit_frame_cuped_runs():
    import datetime as dt

    import pandas as pd

    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.semantics.design import Randomized

    rows = [
        {
            "unit": f"{arm}-{index}",
            "group": arm,
            "date": dt.date(2025, 1, 1),
            "outcome": float(index + (arm == "treatment")),
            "category": None if index % 5 == 0 else ("a" if index % 2 else "b"),
        }
        for arm in ("control", "treatment")
        for index in range(30)
    ]
    method = Method.cuped_multicovariate(
        covariates=(
            CupedCovariate(
                name="category",
                kind="categorical",
                missing="impute-indicator",
            ),
        )
    )
    metrics = (MetricSpec(name="outcome", decision_method=method),)
    design = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})

    summary_experiment, summary_resolver = from_frame(
        pd.DataFrame(rows),
        shape="summary",
        unit="unit",
        group="group",
        metrics=metrics,
        design=design,
    )
    panel_experiment, panel_resolver = from_frame(
        pd.DataFrame(rows),
        shape="panel",
        unit="unit",
        group="group",
        date="date",
        metrics=metrics,
        design=design,
    )

    summary_result = view_readout(summary_experiment, summary_resolver)
    panel_result = view_readout(panel_experiment, panel_resolver)
    assert summary_result == panel_result
    assert summary_result

    from increment.estimation.cuped.encoding import MISSING_CATEGORY
    from increment.readouts._metric_rows import _rows_raw_digest_and_arms

    fit_identity = {"layout": {"levels": (MISSING_CATEGORY,)}}
    row = {"group_id": "control", "fit_identity": fit_identity}
    arms, digest, count = _rows_raw_digest_and_arms((row,))
    expected = StreamingDigest()
    expected.update(
        {
            "group_id": "control",
            "fit_identity": {
                "layout": {"levels": ({"$categorical_missing": "typed missing category"},)}
            },
        }
    )
    assert arms == frozenset({"control"})
    assert (digest, count) == (expected.hexdigest(), expected.count)


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


@pytest.mark.integration
def test_triggered_sequential_scope_reports_and_replays_both_chains(tmp_path):
    from datetime import date

    import pyarrow.parquet as pq
    import pytest

    from increment.errors import CapabilityError
    from increment.estimation.readout_types import ReadoutResults
    from increment.estimation.results import LiftEstimate
    from increment.frame import MetricSpec
    from increment.semantics.design import Randomized
    from tests.test_analysis_trigger import _both_populations
    from tests.test_sequential_public_sources import _native_fixture

    _connection, definitions, (analysis_experiment, analysis_resolver) = _native_fixture(
        "bernoulli", triggered=True
    )
    try:
        snapshot = capture_sequential(
            analysis_experiment, analysis_resolver, finalized=True, as_of=date(2025, 1, 16)
        )
        results = _both_populations(analysis_experiment, analysis_resolver)
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
        export_moments(analysis_experiment, analysis_resolver, path)
        wire_rows = pq.read_table(path).to_pylist()
        assert wire_rows[0]["trigger_name"] == "triggered"
        specs = [MetricSpec(name="outcome", type="conversion", window_days=2)]
        design = Randomized(control_group="control")
        # The stored plan carries the triggered commitment; a caller-supplied plan
        # without it cannot adopt a checkpoint that holds the triggered chain.
        with pytest.raises(CapabilityError) as uncommitted:
            from_moments(
                wire_rows, metrics=specs, design=design, plan=definitions.experiments[0].plan
            )
        assert uncommitted.value.code == "sequential.source.invalid"
        replay_experiment, replay_resolver = from_moments(wire_rows, metrics=specs, design=design)
        try:
            assert sequential_snapshot(replay_experiment, replay_resolver) == snapshot
            replayed = _both_populations(replay_experiment, replay_resolver)
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
            assert {
                row.analysis_population
                for row in view_readout(replay_experiment, replay_resolver, population="triggered")
            } == {"triggered"}
        finally:
            replay_resolver.close()
    finally:
        analysis_resolver.close()
