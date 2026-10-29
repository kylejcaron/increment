from __future__ import annotations

import json

import pytest

from increment.errors import CapabilityError
from increment.semantics.assignment import AssignmentSpec


def test_assignment_spec_round_trips_with_schema_version() -> None:
    spec = AssignmentSpec(
        salt="salt-1",
        bucket_count=100,
        allocation=(
            ("control", tuple(range(50))),
            ("treatment", tuple(range(50, 100))),
        ),
        experiment="exp",
    )
    payload = spec.to_json()
    assert json.loads(payload)["schema_version"] == 1
    assert AssignmentSpec.from_json(payload) == spec


def test_assignment_spec_rejects_unknown_schema_version() -> None:
    with pytest.raises(CapabilityError) as refused:
        AssignmentSpec.from_json('{"schema_version": 999}')
    assert refused.value.code == "evidence.assignment.invalid"


def test_replay_detects_changed_salt_and_duplicate_units() -> None:
    allocation = (("a", tuple(range(128))), ("b", tuple(range(128, 256))))
    spec = AssignmentSpec(salt="one", bucket_count=256, allocation=allocation)
    with pytest.raises(CapabilityError) as refused:
        spec.replay(["u1", "u1"])
    assert refused.value.code == "evidence.assignment.invalid"
    changed = AssignmentSpec(salt="two", bucket_count=256, allocation=allocation)
    assert any(
        spec.replay([unit])[0]["arm"] != changed.replay([unit])[0]["arm"]
        for unit in (f"u-{index}" for index in range(100))
    )


def test_sampled_verification_allows_a_subset_of_the_eligible_roster() -> None:
    allocation = (("a", tuple(range(128))), ("b", tuple(range(128, 256))))
    spec = AssignmentSpec(
        salt="salt",
        bucket_count=256,
        allocation=allocation,
        eligible_units=("u1", "u2", "u3"),
    )
    eligible = spec.eligible_units
    assert eligible is not None
    ledger = spec.replay(eligible)
    assert spec.verify(ledger, sample=("u1",)) == "sampled_check"


@pytest.mark.parametrize("units", [("u1", "u2"), ("u1", "u2", "u3", "u4")])
def test_full_verification_classifies_eligibility_mismatch_as_replay_mismatch(
    units: tuple[str, ...],
) -> None:
    spec = AssignmentSpec(
        salt="salt",
        bucket_count=256,
        allocation=(("a", tuple(range(128))), ("b", tuple(range(128, 256)))),
        eligible_units=("u1", "u2", "u3"),
    )
    if len(units) > len(spec.eligible_units or ()):
        eligible = spec.eligible_units
        assert eligible is not None
        ledger = (*spec.replay(eligible), {"unit": "u4", "arm": "a"})
    else:
        ledger = spec.replay(units, allow_eligible_subset=True)
    with pytest.raises(CapabilityError) as refused:
        spec.verify(ledger)
    assert refused.value.code == "evidence.assignment.replay_mismatch"


@pytest.mark.parametrize(
    ("coordinate", "replacement"),
    [("experiment", "other"), ("version", 2), ("period", "p2")],
)
def test_verification_refuses_changed_replay_identity(coordinate, replacement) -> None:
    spec = AssignmentSpec(
        salt="salt",
        bucket_count=256,
        allocation=(("a", tuple(range(128))), ("b", tuple(range(128, 256)))),
        experiment="exp",
    )
    ledger = [dict(row) for row in spec.replay(("u1", "u2"))]
    for row in ledger:
        row[coordinate] = replacement
    with pytest.raises(CapabilityError) as refused:
        spec.verify(ledger)
    assert refused.value.code == "evidence.assignment.replay_mismatch"


def test_invalid_replay_document_refuses_with_registered_code() -> None:
    with pytest.raises(CapabilityError) as refused:
        AssignmentSpec.from_json('{"schema_version":1}')
    assert refused.value.code == "evidence.assignment.invalid"


def test_public_replay_returns_an_identity_ledger(monkeypatch) -> None:
    import sys

    from increment.evidence import FrameLedger, replay

    monkeypatch.delitem(sys.modules, "pandas", raising=False)

    spec = AssignmentSpec(
        salt="salt",
        bucket_count=256,
        allocation=(("a", tuple(range(128))), ("b", tuple(range(128, 256)))),
        experiment="exp",
    )
    ledger = replay(spec, ("u1", "u2"))
    assert isinstance(ledger, FrameLedger)
    evidence = ledger.reduce(experiment="exp", population="assigned")
    assert sum(count for _, _, _, count in evidence.arm_counts) == 2
    assert {row.unit for row in evidence.rows} == {"u1", "u2"}
    assert isinstance(replay(spec, ()), FrameLedger)


def test_assignment_spec_rejects_design_owned_integrity_declarations() -> None:
    with pytest.raises(CapabilityError) as refused:
        AssignmentSpec.from_json('{"schema_version":1,"randomization_law":"shared_schedule"}')
    assert refused.value.code == "evidence.assignment.invalid"
    assert refused.value.context["reason"] == "design_declaration_conflict"
