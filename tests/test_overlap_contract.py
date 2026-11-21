"""The declared population, decision family and bounded replay contract."""

import copy
import json
import pickle
from datetime import UTC, timedelta
from fractions import Fraction
from math import nextafter
from pathlib import Path

import pytest
from pydantic import ValidationError

from increment.errors import CodedError
from increment.semantics.overlap import (
    OverlapCapture,
    OverlapCell,
    OverlapEvidence,
    OverlapFactor,
    OverlapPlan,
)


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


def _evidence_with_target_count(contract, n_target):
    raw = copy.deepcopy(contract["evidence"])
    raw.pop("evidence_sha256")
    raw["eligibility_counts"][3] = raw["gate_counts"][0] = n_target
    for index in (1, 5):  # Cell 10 for each metric; cell 01 retains one unit.
        raw["cells"][index].update(n_assigned=n_target - 1, n_observed=n_target - 1)
    for row in raw["delivery_counts"][2:4]:
        row["n_without_delivery"] = n_target - 1
    return raw


def test_maximum_exact_count_and_weight_round_trip(contract):
    limit = contract["numeric_domain"]["max_exact_integer"]
    raw = _evidence_with_target_count(contract, limit)
    raw["plan"]["a"].update(off_weight=1, on_weight=limit)
    plan = OverlapPlan.model_validate(raw["plan"])
    raw["capture"]["law_sha256"] = plan.law_sha256
    evidence = OverlapEvidence.model_validate(raw)
    replay = OverlapEvidence.from_json(evidence.to_json(), expected_sha256=evidence.evidence_sha256)
    assert replay.target_count == limit
    assert replay.plan.a.arm_probabilities == (Fraction(1, 1 + limit), Fraction(limit, 1 + limit))
    assert replay.plan.cell_probabilities[0] == Fraction(3, 4 * (1 + limit))
    assert sum(replay.plan.cell_probabilities) == 1


def test_weight_above_portable_integer_limit_is_refused(contract):
    raw = copy.deepcopy(contract["evidence"]["plan"])
    raw["a"]["on_weight"] = contract["numeric_domain"]["max_exact_integer"] + 1
    with pytest.raises(CodedError) as caught:
        OverlapPlan.model_validate(raw)
    assert caught.value.code == "overlap.plan.invalid"
    assert caught.value.context == {"field": "on_weight", "reason": "invalid_value"}


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


def test_large_integral_float_center_round_trips_without_accepting_alternate_bytes(contract):
    raw = _evidence_with_target_count(contract, 3)
    center = dict(contract["numeric_domain"]["large_center"])
    anchor_token = center.pop("anchor_token")
    raw["cells"][1].update(center)
    evidence = OverlapEvidence.model_validate(raw)
    wire = evidence.to_json()
    assert f'"anchor":{anchor_token},' in wire
    replay = OverlapEvidence.from_json(wire, expected_sha256=evidence.evidence_sha256)
    assert {key: getattr(replay.cells[1], key) for key in center} == center

    alternate = wire.replace(f'"anchor":{anchor_token},', '"anchor":1e16,')
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.from_json(alternate, expected_sha256=evidence.evidence_sha256)
    assert caught.value.context == {"field": "payload", "reason": "noncanonical_encoding"}


def test_count_above_portable_integer_limit_is_refused(contract):
    count = contract["numeric_domain"]["max_exact_integer"] + 1
    raw = _evidence_with_target_count(contract, count)
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_validate(raw)
    assert caught.value.code == "overlap.evidence.invalid"
    assert caught.value.context == {
        "role": "eligibility_counts",
        "reason": "invalid_value",
        "count": 1,
    }


@pytest.mark.parametrize("m2", [0.25, 871.0])
def test_minimum_anchored_center_refuses_impossible_centered_ss(contract, m2):
    raw = dict(
        contract["evidence"]["cells"][1], n_assigned=30, n_observed=30, mean_delta=1.0, m2=m2
    )
    with pytest.raises(CodedError) as caught:
        OverlapCell.model_validate(raw)
    assert caught.value.context == {"role": "cells", "reason": "inconsistent_moments", "count": 1}


@pytest.mark.parametrize(
    "offsets",
    [(0, *(1,) * 29), (*(0,) * 29, 1), (0, 1.25 * 2**-537), (0, 1e154)],
    ids=["minimum-ss", "maximum-ss", "subnormal-ss", "large-ss"],
)
def test_minimum_anchored_center_accepts_valid_endpoints(
    contract, offsets: tuple[int | float, ...]
):
    n = len(offsets)
    exact = tuple(Fraction(offset) for offset in offsets)
    delta = sum(exact) / n
    m2 = sum((offset - delta) ** 2 for offset in exact)
    raw = dict(
        contract["evidence"]["cells"][1],
        n_assigned=n,
        n_observed=n,
        anchor=0,
        mean_delta=float(delta),
        m2=float(m2),
    )
    cell = OverlapCell.model_validate(raw)
    assert cell.m2 == float(m2) > 0


@pytest.mark.parametrize("quantity", ["mean_delta", "m2"])
def test_binary_moments_must_equal_the_exact_count_projection(contract, quantity):
    n = 10**14
    success = n - 10
    raw = _evidence_with_target_count(contract, n + 1)
    cell = raw["cells"][5]
    cell.update(
        n_success=success,
        mean_delta=float(Fraction(success, n)),
        m2=float(Fraction(success * 10, n)),
    )
    assert OverlapEvidence.model_validate(raw).cells[5].n_success == success

    cell[quantity] = nextafter(1.0 if quantity == "mean_delta" else cell["m2"], float("inf"))
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_validate(raw)
    assert caught.value.context == {
        "role": "cells",
        "reason": "binary_moments_mismatch",
        "count": 1,
    }


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


def test_nested_nonstring_key_does_not_leak_through_schema_location(contract):
    raw = _evidence_with_target_count(contract, 2)
    private_key = 987_654_321
    raw["cells"][1][private_key] = "synthetic-private-record"
    with pytest.raises(CodedError) as caught:
        OverlapEvidence.model_validate(raw)
    assert caught.value.code == "overlap.evidence.invalid"
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    detail = cause.errors()[0]
    assert (detail["loc"], detail["type"]) == (("cells", 1, "<unknown field>"), "invalid_key")
    assert str(private_key) not in cause.json()


@pytest.mark.parametrize("entry", ["model", "wire"])
def test_numeric_range_does_not_expose_a_metric_from_unverified_input(contract, entry):
    raw = _evidence_with_target_count(contract, 2)
    sentinel = "synthetic-unverified-metric"
    for cell in raw["cells"][:4]:
        cell["metric"] = sentinel
    raw["plan"]["metrics"][0]["name"] = sentinel
    for claim in raw["plan"]["claims"][:3]:
        claim["metric"] = sentinel
    raw["cells"][1].update(anchor=1e308, mean_delta=1e308, m2=1)

    with pytest.raises(CodedError) as caught:
        if entry == "model":
            OverlapEvidence.model_validate(raw)
        else:
            _replay(contract, json.dumps(raw))
    if entry == "model":
        assert caught.value.code == "overlap.numeric.range"
        assert caught.value.context == {"metric": "unverified", "quantity": "mean"}
    else:
        assert caught.value.code == "overlap.format.invalid"
        assert caught.value.context["reason"] == "invalid_value"
    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    diagnostic = cause.errors()[0]["ctx"]["error"]
    assert diagnostic.code == "overlap.numeric.range"
    assert diagnostic.context["quantity"] == "mean"
    assert sentinel not in cause.json()
