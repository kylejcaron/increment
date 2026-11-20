"""The declared population, decision family and bounded replay contract."""

import copy
import json
import pickle
from datetime import UTC, timedelta
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError

from increment.errors import CodedError
from increment.semantics.overlap import OverlapCapture, OverlapEvidence, OverlapFactor, OverlapPlan


@pytest.fixture(scope="module")
def contract():
    path = Path(__file__).parent / "fixtures" / "overlap" / "contract_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_same_layer_is_structurally_exclusive_before_reads(contract):
    spec = copy.deepcopy(contract["evidence"]["plan"])
    for factor in ("a", "b"):
        spec[factor].update(layer="shared", target="shared")
    with pytest.raises(CodedError) as caught:
        OverlapPlan.model_validate(spec)
    assert caught.value.code == "overlap.law.unsupported"
    assert caught.value.context == {"reason": "same_layer", "factors": ("A", "B")}
    for restored in (copy.deepcopy(caught.value), pickle.loads(pickle.dumps(caught.value))):
        assert restored.code == caught.value.code
        assert restored.context == caught.value.context
        with pytest.raises(TypeError):
            restored.context["reason"] = "changed"  # ty: ignore[invalid-assignment]


def test_all_eligible_cannot_have_partial_admission(contract):
    spec = copy.deepcopy(contract["evidence"]["plan"])
    spec.update(target="jointly_eligible")
    spec["law"]["admission"] = "joint_gate_required"
    with pytest.raises(CodedError) as caught:
        OverlapPlan.model_validate(spec)
    assert caught.value.code == "overlap.law.unsupported"
    assert caught.value.context == {"reason": "admission_not_guaranteed", "factors": ("A", "B")}
    spec.update(target="jointly_eligible_and_enrolled")
    assert OverlapPlan.model_validate(spec).law.admission == "joint_gate_required"


@pytest.mark.parametrize(
    ("change", "field", "reason"),
    [
        ("omit_guardrail", "claims", "incomplete_joint_launch"),
        ("wrong_joint_direction", "claims", "joint_primary_requires_benefit"),
        ("wrong_guardrail_direction", "claims", "alternative_incompatible_with_role"),
        ("duplicate_claim", "claims", "duplicate_claim"),
        ("zero_guardrail_margin", "margin_abs", "guardrail_requires_positive_margin"),
    ],
)
def test_claims_lock_direction_margin_and_family(contract, change, field, reason):
    spec = copy.deepcopy(contract["evidence"]["plan"])
    if change == "omit_guardrail":
        spec["claims"].pop()
    elif change == "wrong_joint_direction":
        spec["claims"][1]["alternative"] = "difference"
    elif change == "wrong_guardrail_direction":
        spec["claims"][-1]["alternative"] = "benefit"
    elif change == "duplicate_claim":
        spec["claims"].append(spec["claims"][0])
    else:
        spec["metrics"][-1]["margin_abs"] = 0
    with pytest.raises(CodedError) as caught:
        OverlapPlan.model_validate(spec)
    assert caught.value.code == "overlap.plan.invalid"
    assert caught.value.context == {"field": field, "reason": reason}


def test_exact_weights_do_not_derive_the_rare_arm_by_subtraction(contract):
    spec = copy.deepcopy(contract["evidence"]["plan"])
    spec["a"].update(off_weight=1, on_weight=10**20)
    plan = OverlapPlan.model_validate(spec)
    assert plan.a.arm_probabilities == (Fraction(1, 1 + 10**20), Fraction(10**20, 1 + 10**20))
    assert plan.cell_probabilities[0] == Fraction(3, 4 * (1 + 10**20))
    assert sum(plan.cell_probabilities) == 1
    spec["a"]["on_weight"] = 10**350
    with pytest.raises(CodedError) as caught:
        OverlapPlan.model_validate(spec)
    assert caught.value.code == "overlap.law.unsupported"
    assert caught.value.context["reason"] == "joint_probability_unrepresentable"


def test_boolean_weight_is_not_an_integer(contract):
    raw = {**contract["evidence"]["plan"]["a"], "off_weight": True}
    with pytest.raises(CodedError) as caught:
        OverlapFactor.model_validate(raw)
    assert caught.value.context == {"field": "off_weight", "reason": "invalid_value"}


@pytest.mark.parametrize("start", ["2026-01-01T00:00:03", "2026-01-01T00:00:03.0000001Z"])
def test_plan_rejects_ambiguous_timestamp_order(contract, start):
    raw = {**contract["evidence"]["plan"], "start": start}
    with pytest.raises(CodedError) as caught:
        OverlapPlan.model_validate(raw)
    assert caught.value.code == "overlap.plan.invalid"
    assert caught.value.context == {"field": "start", "reason": "invalid_value"}


def test_offset_instants_and_claim_order_have_one_registered_identity(contract):
    raw = copy.deepcopy(contract["evidence"]["plan"])
    raw["start"] = "2025-12-31T19:00:03.000001-05:00"
    raw["claims"].reverse()
    plan = OverlapPlan.model_validate(raw)
    assert plan.start.tzinfo is UTC
    assert plan.start.microsecond == 1
    assert plan.claims[0].contrast == "b_given_a_on"
    assert plan.plan_sha256 == contract["plan_sha256"]
    assert plan.family_sha256 == contract["family_sha256"]


def test_capture_archive_crossing_refuses_even_with_matching_law_hash(contract):
    raw = copy.deepcopy(contract["evidence"])
    raw["capture"]["archive_sha256"].reverse()
    plan = OverlapPlan.model_validate(raw["plan"])
    capture = OverlapCapture.model_validate(raw["capture"])
    with pytest.raises(CodedError) as caught:
        capture.validate_for(plan)
    assert caught.value.code == "overlap.capture.invalid"
    assert caught.value.context == {"role": "decisions", "reason": "archive_mismatch"}


def test_capture_requires_delivery_coverage_on_the_final_receipt(contract):
    raw = copy.deepcopy(contract["evidence"])
    plan = OverlapPlan.model_validate(raw["plan"])
    raw["capture"]["feeds"][2]["complete_through"] = plan.end - timedelta(microseconds=1)
    capture = OverlapCapture.model_validate(raw["capture"])
    with pytest.raises(CodedError) as caught:
        capture.validate_for(plan)
    assert caught.value.context == {"role": "deliveries", "reason": "incomplete_coverage"}


def _replay(contract, wire):
    return OverlapEvidence.from_json(wire, expected_sha256=contract["evidence"]["evidence_sha256"])


def test_golden_wire_replays_exact_bounded_centers(contract):
    evidence = _replay(contract, contract["wire"])
    assert evidence.to_json() == contract["wire"]
    assert evidence.evidence_sha256 == contract["evidence"]["evidence_sha256"]
    assert evidence.eligibility_counts == (0, 0, 1, 2)
    assert evidence.gate_counts[0] == 2
    assert evidence.replay_status == "not_certified"
    assert tuple(
        (cell.cell, cell.anchor, cell.mean_delta, cell.m2) for cell in evidence.cells[:4]
    ) == (
        ("00", None, None, None),
        ("10", 14.0, 0.0, 0.0),
        ("01", 13.0, 0.0, 0.0),
        ("11", None, None, None),
    )


@pytest.mark.parametrize(
    ("damage", "field", "reason"),
    [
        ("repeat_cell", "cells", "cell_order"),
        ("future_version", "format", "unsupported_version"),
        ("repeat_key", "payload", "duplicate_key"),
    ],
)
def test_ordered_wire_rejects_repeated_cell_and_future_version(contract, damage, field, reason):
    raw = copy.deepcopy(contract["evidence"])
    if damage == "repeat_cell":
        raw["cells"][1] = raw["cells"][0]
    elif damage == "future_version":
        raw["format"] = 2
    wire = json.dumps(raw, separators=(",", ":"), sort_keys=True)
    if damage == "repeat_key":
        wire = '{"format":1,' + wire[1:]
    with pytest.raises(CodedError) as caught:
        _replay(contract, wire)
    assert caught.value.code == "overlap.format.invalid"
    assert caught.value.context == {"field": field, "reason": reason}


def test_replay_rejects_payload_changes_and_an_unpinned_digest(contract):
    wire = contract["wire"].replace('"anchor":14', '"anchor":15')
    with pytest.raises(CodedError) as caught:
        _replay(contract, wire)
    assert caught.value.context == {"field": "evidence_sha256", "reason": "digest_mismatch"}
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.from_json(contract["wire"], expected_sha256="0" * 64)
    assert caught.value.context == {"field": "evidence_sha256", "reason": "digest_mismatch"}


def test_incomplete_cell_must_erase_the_survivors_center(contract):
    raw = copy.deepcopy(contract["evidence"])
    raw.pop("evidence_sha256")
    raw["cells"][1].update(n_observed=0, n_missing=1)
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_validate(raw)
    assert caught.value.code == "overlap.evidence.invalid"
    assert caught.value.context == {"role": "cells", "reason": "unusable_moments", "count": 1}


def test_oversized_count_refuses_instead_of_overflowing_binary_center(contract):
    raw = copy.deepcopy(contract["evidence"])
    raw.pop("evidence_sha256")
    count = 10**600
    raw["eligibility_counts"][3] = raw["gate_counts"][0] = count + 1
    raw["cells"][1].update(n_assigned=count, n_observed=count)
    raw["cells"][5].update(n_assigned=count, n_observed=count, n_success=count // 2)
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_validate(raw)
    assert caught.value.code == "overlap.evidence.invalid"
    assert caught.value.context["reason"] == "invalid_value"


def test_model_construction_cannot_bypass_cell_integrity(contract):
    raw = copy.deepcopy(contract["evidence"])
    raw.pop("evidence_sha256")
    raw["cells"] = []
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_construct(**raw)
    assert caught.value.context == {"role": "cells", "reason": "cell_order", "count": 1}


@pytest.mark.parametrize("entry", ["model", "wire"])
def test_nested_evidence_refusal_keeps_shape_not_submitted_values(contract, entry):
    raw = copy.deepcopy(contract["evidence"])
    sentinel = "synthetic-private-record"
    raw["cells"][1]["n_observed"] = {"unit_id": sentinel}
    if entry == "model":
        raw.pop("evidence_sha256")
    with pytest.raises(CodedError) as caught:
        if entry == "model":
            OverlapEvidence.model_validate(raw)
        else:
            _replay(contract, json.dumps(raw))

    if entry == "model":
        assert caught.value.code == "overlap.evidence.invalid"
        assert caught.value.context == {"role": "cells", "reason": "invalid_value", "count": 1}
    else:
        assert caught.value.code == "overlap.format.invalid"
        assert caught.value.context == {"field": "cells", "reason": "invalid_value"}
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    detail = cause.errors()[0]
    assert (detail["loc"], detail["type"], detail.get("input")) == (
        ("cells", 1, "n_observed"),
        "int_type",
        None,
    )
    assert sentinel not in cause.json()
    assert caught.value.__context__ in (None, cause)
    assert cause.__context__ is None
    assert cause.__cause__ is None


@pytest.mark.parametrize("entry", ["model_json", "wire"])
def test_truncated_overlap_json_never_exposes_the_document(contract, entry):
    payload = '{"cells":[{"unit_id":"synthetic-private-record"}'
    with pytest.raises(CodedError) as caught:
        if entry == "model_json":
            OverlapEvidence.model_validate_json(payload)
        else:
            _replay(contract, payload)
    assert caught.value.code == "overlap.format.invalid"
    assert caught.value.context == {"field": "payload", "reason": "invalid_json"}
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_direct_model_json_rejects_duplicate_keys(contract):
    raw = copy.deepcopy(contract["evidence"])
    raw.pop("evidence_sha256")
    payload = '{"format":2,' + json.dumps(raw)[1:]
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_validate_json(payload)
    assert caught.value.context == {"field": "payload", "reason": "duplicate_key"}
