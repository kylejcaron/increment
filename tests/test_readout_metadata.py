"""Behavioral contracts for portable, source-scoped readout results."""

import pickle
from copy import deepcopy

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
    assert CellKey(**base, value_scale="relative") != CellKey(**base, value_scale="absolute")
    assert CellKey(**base, inference="fixed") != CellKey(**base, inference="always_valid")


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
    assert evidence.log_e == -inf

    sampling = SamplingInference(available=True, evidence=evidence)
    restored = SamplingInference.model_validate_json(sampling.model_dump_json())

    assert restored.available is True
    assert restored.evidence is not None
    assert restored.evidence.log_e == -inf


def test_triggered_sequential_scope_uses_explicit_unsupported_placeholder():
    from types import SimpleNamespace

    from increment.readouts._sequential_scope import scope_sequential_results
    from tests.estimation.test_sequential_public_proof_acceptance import _public_family

    registration, policy, bundle, _ = _public_family(0, 0, 0, 0, "two-sided")
    result = bundle.results[0].require_sequential_result()
    plan = SimpleNamespace(
        alpha=float(registration.roster[0].alpha),
        q=float(registration.q),
        inference=policy,
    )
    source = SimpleNamespace(
        context=SimpleNamespace(
            plan=plan,
            design=SimpleNamespace(control_group="control"),
            trigger_name="registered_trigger",
        )
    )
    from increment.sequential_state import registration_id

    snapshot = SimpleNamespace(registration_id=registration_id(registration), prefix_id="prefix")

    scoped = scope_sequential_results(
        source,
        bundle.results,
        snapshot,
        metrics=("outcome",),
        estimands=("itt",),
    )

    placeholder = next(
        row
        for row in scoped
        if row.analysis_population == "triggered" and row.group_id == "first"
    )
    assert placeholder.failure_code == "readout.cell.unsupported_request"
    assert placeholder.failure_context["reason"] == "triggered_sequential"
    assert placeholder.inference == "always_valid"
    assert placeholder.reference_kind == "sequential"
    assert placeholder.sequential_result is None
    assert placeholder.sampling_available is False

    other_source = SimpleNamespace(
        context=SimpleNamespace(
            plan=plan,
            design=SimpleNamespace(control_group="control"),
            trigger_name="another_trigger",
        )
    )
    other = scope_sequential_results(
        other_source,
        bundle.results,
        snapshot,
        metrics=("outcome",),
        estimands=("itt",),
    )
    assert other.metadata.scope.snapshot_id != scoped.metadata.scope.snapshot_id
