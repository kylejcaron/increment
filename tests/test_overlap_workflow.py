"""Integration tests for the two-experiment workflow's standalone public API."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import ibis
import pyarrow as pa
import pytest

from increment._interaction_capture import capture_interaction_scan
from increment.assignment import compile_allocation
from increment.errors import CodedError
from increment.overlap import (
    OverlapCapture,
    OverlapEvidence,
    OverlapPlan,
    OverlapResult,
    OverlapSdkArchive,
    analyze_overlap,
    capture_overlap_frame,
    capture_overlap_warehouse,
    sdk_overlap_decision,
    sdk_overlap_plan,
    simulate_overlap,
)
from increment.query.overlap import bind_overlap_warehouse
from increment.semantics.interaction_scan import CapturedPair
from tests.test_interaction_capture import observed_at_look, scan_for_pair

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend

_DELIVERIES = pa.schema(
    {
        **dict.fromkeys(
            (
                "study_id",
                "event_id",
                "unit_id",
                "target",
                "experiment",
                "revision",
                "delivered_arm",
            ),
            pa.string(),
        ),
        "delivered_at": pa.timestamp("us", tz="UTC"),
    }
)
_TIMESTAMPS = {
    "entry_at",
    "eligibility_at_a",
    "eligibility_at_b",
    "gate_at_a",
    "gate_at_b",
    "decided_at",
    "delivered_at",
    "window_start",
    "window_end",
    "finalized_at",
}
_GATE_COLUMNS = {"gate_a", "gate_b", "gate_at_a", "gate_at_b"}


@pytest.fixture
def external_contract():
    file = Path(__file__).parent / "fixtures" / "overlap" / "contract_v1.json"
    return json.loads(file.read_text(encoding="utf-8"))


def _tables(rows, *, sdk=False):
    def converted(item):
        key, value = item
        if key in _TIMESTAMPS and isinstance(value, str):
            return datetime.fromisoformat(value)
        if key == "value" and value is not None:
            return float(value)
        return value

    tables = {}
    for role in ("roster", "decisions", "deliveries", "outcomes"):
        selected = [
            {
                key: converted((key, value))
                for key, value in row.items()
                if not (sdk and role == "roster" and key in _GATE_COLUMNS)
            }
            for row in rows[role]
        ]
        tables[role] = pa.Table.from_pylist(
            selected, schema=_DELIVERIES if role == "deliveries" else None
        )
    return tables


def _raw_sdk_records(tmp_path, archive_a, archive_b, plan, roster):
    evaluators = tuple(compile_allocation(json.loads(raw)) for raw in (archive_a, archive_b))
    file = tmp_path / "actual_sdk_decisions.json"
    file.write_text(
        json.dumps(
            [
                {
                    "unit_id": entry["unit_id"],
                    "decided_at": f"2026-01-01T00:00:0{i + 1}Z",
                    "decision": evaluators[i].decide(factor.target, entry["unit_id"]),
                }
                for entry in roster
                for i, factor in enumerate((plan.a, plan.b))
                if entry[("eligible_a", "eligible_b")[i]]
            ]
        ),
        encoding="utf-8",
    )
    return json.loads(file.read_text(encoding="utf-8"))


def _publish_tables(con: Backend, tables):
    for role, table in tables.items():
        con.create_table(role, table, overwrite=True)
    return {role: con.table(role, database=("overlap", "main")) for role in tables}


def _scientific_projection(evidence):
    result = analyze_overlap(evidence)
    return (
        evidence.eligibility_counts,
        evidence.gate_counts,
        evidence.delivery_counts,
        evidence.cells,
        result.rows,
        result.covariance,
        result.joint_decision,
    )


def _round_trip(evidence: OverlapEvidence) -> OverlapEvidence:
    restored = OverlapEvidence.from_json(
        evidence.to_json(), expected_sha256=evidence.evidence_sha256
    )
    result = analyze_overlap(restored)
    replay = OverlapResult.from_json(result.to_json(), expected_sha256=result.result_sha256)
    assert replay.to_json() == result.to_json()
    return restored


def _scan_pair_receipt(evidence: OverlapEvidence) -> OverlapEvidence:
    plan = scan_for_pair(evidence)
    capture = capture_interaction_scan(
        plan,
        observed_inventory=observed_at_look(plan),
        evidence_by_study={evidence.plan.study_id: evidence},
    )
    assert capture.coverage == "complete"
    (pair,) = capture.pairs
    assert isinstance(pair.value, CapturedPair)
    assert pair.value.focal_side == "A"
    assert not pair.value.metric_issues
    assert pair.value.evidence.evidence_sha256 == evidence.evidence_sha256
    return pair.value.evidence


def _sdk_plan(contract, a_config, b_config, *, target="jointly_eligible"):
    kwargs = dict(contract["evidence"]["plan"])
    law = kwargs.pop("law")
    for name in ("a", "b"):
        kwargs.pop(name)
    kwargs["target"] = target
    return sdk_overlap_plan(
        a_config=a_config,
        a_target="A",
        a_experiment="A",
        a_off="off",
        a_on="on",
        b_config=b_config,
        b_target="B",
        b_experiment="B",
        b_off="off",
        b_on="on",
        serving_barrier_id=law["serving_barrier_id"],
        **kwargs,
    )


def _capture_for(plan, external_contract):
    payload = dict(external_contract["evidence"]["capture"])
    payload.update(
        origin=plan.law.origin, law_sha256=plan.law_sha256, archive_sha256=plan.law.archive_sha256
    )
    return OverlapCapture.model_validate(payload)


def _sdk_tables(plan, records, config_a, config_b, external):
    decisions = [
        sdk_overlap_decision(
            study_id=plan.study_id,
            unit_id=entry["unit_id"],
            decided_at=datetime.fromisoformat(entry["decided_at"]),
            decision=entry["decision"],
            config=config_a if entry["decision"]["target"] == plan.a.target else config_b,
        )
        for entry in records
    ]
    return _tables({**external, "decisions": decisions}, sdk=True)


@pytest.mark.integration
def test_public_sdk_external_frame_warehouse_and_wire(tmp_path, external_contract):
    contract = external_contract
    external = contract["external"]
    compact = contract["sdk"]["archives"]["full"]["utf8"].encode("utf-8")
    readable = json.dumps(json.loads(compact), indent=2).encode("utf-8")
    plans = (
        _sdk_plan(contract, compact, readable),
        _sdk_plan(contract, readable, compact),
        OverlapPlan.model_validate(contract["evidence"]["plan"]),
    )
    raw_decisions = _raw_sdk_records(tmp_path, compact, readable, plans[0], external["roster"])
    assert [(r["unit_id"], r["decision"]["arm"]) for r in raw_decisions] == [
        ("001", "on"),
        ("001", "off"),
        ("1", "off"),
        ("1", "on"),
        ("x", "off"),
    ]
    a = OverlapSdkArchive.from_bytes(compact)
    b = OverlapSdkArchive.from_bytes(readable)
    relations = (
        _sdk_tables(plans[0], raw_decisions, a, b, external),
        _sdk_tables(plans[1], raw_decisions, b, a, external),
        _tables(external),
    )
    hashes = tuple(hashlib.sha256(raw).hexdigest() for raw in (compact, readable))
    assert plans[0].law.archive_sha256 == hashes
    assert plans[1].law.archive_sha256 == hashes[::-1]
    assert plans[0].a.arm_probabilities == plans[1].a.arm_probabilities
    assert plans[0].b.arm_probabilities == plans[1].b.arm_probabilities

    con = ibis.duckdb.connect(tmp_path / "overlap.duckdb")
    projections = []
    try:
        for plan, inputs in zip(plans, relations, strict=True):
            capture = _capture_for(plan, contract)
            frame = capture_overlap_frame(plan, capture=capture, **inputs)
            warehouse_tables = _publish_tables(con, inputs)
            bound = bind_overlap_warehouse(plan, capture=capture, **warehouse_tables)
            warehouse = capture_overlap_warehouse(plan, capture=bound, **warehouse_tables)
            assert frame.capture.generation_id != warehouse.capture.generation_id
            _assert_tiny_science(frame)
            for result in (frame, warehouse):
                assert result.capture.archive_sha256 == plan.law.archive_sha256
                restored = _round_trip(result)
                for bundle in (result, restored):
                    projections.append(_scientific_projection(_scan_pair_receipt(bundle)))

        plan = plans[0]
        inputs = relations[0]
        capture = _capture_for(plan, contract)
        alternative_capture = _capture_for(plans[1], contract)
        with pytest.raises(CodedError) as inconsistent_header:
            capture_overlap_frame(plan, capture=alternative_capture, **inputs)
        with pytest.raises(CodedError) as warehouse_header:
            capture_overlap_warehouse(
                plan, capture=alternative_capture, **_publish_tables(con, inputs)
            )
        assert (
            inconsistent_header.value.code
            == warehouse_header.value.code
            == ("overlap.capture.invalid")
        )
        final = len(inputs["decisions"]) - 1
        crossed = pa.concat_tables(
            [inputs["decisions"].slice(0, final), relations[1]["decisions"].slice(final)]
        )
        inputs = {**inputs, "decisions": crossed}
        with pytest.raises(CodedError) as frame_rows:
            capture_overlap_frame(plan, capture=capture, **inputs)
        tables = _publish_tables(con, inputs)
        bound = bind_overlap_warehouse(plan, capture=capture, **tables)
        with pytest.raises(CodedError) as warehouse_rows:
            capture_overlap_warehouse(plan, capture=bound, **tables)
        assert frame_rows.value.code == warehouse_rows.value.code == "overlap.evidence.invalid"
        assert (
            frame_rows.value.context
            == warehouse_rows.value.context
            == {
                "role": "decisions",
                "reason": "archive_mismatch",
                "count": 1,
            }
        )
    finally:
        con.disconnect()
    assert all(projection == projections[0] for projection in projections[1:])


def _assert_tiny_science(evidence):
    assert evidence.eligibility_counts == (0, 0, 1, 2)
    assert evidence.gate_counts == (2,) + (0,) * 24
    assert [(c.n_assigned, c.anchor) for c in evidence.cells if c.metric == "orders"] == [
        (0, None),
        (1, 14.0),
        (1, 13.0),
        (0, None),
    ]
    assert [(c.n_success, c.anchor) for c in evidence.cells if c.metric == "complaint"] == [
        (0, None),
        (0, 0.0),
        (1, 1.0),
        (0, None),
    ]
    result = analyze_overlap(evidence)
    assert {(row.metric, row.contrast) for row in result.rows if row.estimate is not None} == set()
    assert result.joint_decision == "insufficient_evidence"


@pytest.mark.integration
def test_sdk_declined_enrollment_stays_outside_both_randomized_population(
    tmp_path, external_contract
):
    contract = external_contract
    archives = contract["sdk"]["archives"]
    raw_a, raw_b = (archives[k]["utf8"].encode("utf-8") for k in ("partial", "full"))
    receipt_a, receipt_b = (OverlapSdkArchive.from_bytes(raw) for raw in (raw_a, raw_b))
    plan = _sdk_plan(contract, receipt_a, receipt_b, target="jointly_eligible_and_enrolled")
    external = contract["external"]
    roster = [*external["roster"], {**external["roster"][0], "unit_id": "case-1"}]
    events = _raw_sdk_records(tmp_path, raw_a, raw_b, plan, roster)
    assert [r["decision"]["status"] for r in events if r["unit_id"] == "case-1"] == [
        "not_enrolled",
        "assigned",
    ]
    inputs = _sdk_tables(plan, events, receipt_a, receipt_b, {**external, "roster": roster})
    capture = _capture_for(plan, contract)
    frame = capture_overlap_frame(plan, capture=capture, **inputs)
    assert frame.eligibility_counts == (0, 0, 1, 3)
    assert frame.target_count == 2
    assert tuple(row.n_assigned for row in frame.cells[:4]) == (0, 1, 1, 0)
    con = ibis.duckdb.connect(tmp_path / "overlap.duckdb")
    try:
        tables = _publish_tables(con, inputs)
        bound = bind_overlap_warehouse(plan, capture=capture, **tables)
        captured = capture_overlap_warehouse(plan, capture=bound, **tables)
        assert _scientific_projection(captured) == _scientific_projection(frame)
    finally:
        con.disconnect()


@pytest.mark.integration
def test_external_missing_or_late_decision_is_not_control(tmp_path, external_contract):
    contract = external_contract
    plan = OverlapPlan.model_validate(contract["evidence"]["plan"])
    capture = _capture_for(plan, contract)
    external = contract["external"]
    con = ibis.duckdb.connect(tmp_path / "overlap.duckdb")
    try:
        for damage in external["damage"].values():
            rows = dict(external)
            if "delete_decision" in damage:
                key = damage["delete_decision"]
                rows["decisions"] = [
                    row for row in rows["decisions"] if {name: row[name] for name in key} != key
                ]
            else:
                fix = damage["update_roster"]
                rows["roster"] = [
                    {**row, **fix} if row["unit_id"] == fix["unit_id"] else row
                    for row in rows["roster"]
                ]
            inputs = _tables(rows)
            with pytest.raises(CodedError) as frame:
                capture_overlap_frame(plan, capture=capture, **inputs)
            tables = _publish_tables(con, inputs)
            bound = bind_overlap_warehouse(plan, capture=capture, **tables)
            with pytest.raises(CodedError) as warehouse:
                capture_overlap_warehouse(plan, capture=bound, **tables)
            expected = dict(damage["context"])
            if "details" in expected:
                expected["details"] = tuple(expected["details"])
            assert frame.value.code == warehouse.value.code == damage["code"]
            assert frame.value.context == warehouse.value.context == expected
    finally:
        con.disconnect()


def test_planning_runtime_main_and_guardrail_decisions_agree(external_contract):
    contract = external_contract
    raw_plan = contract["evidence"]["plan"]
    factors = {side: {**raw_plan[side], "off_weight": 1, "on_weight": 1} for side in ("a", "b")}
    orders, complaint = raw_plan["metrics"]
    plan = OverlapPlan.model_validate(
        {
            **raw_plan,
            **factors,
            "metrics": [orders, {**complaint, "margin_abs": 0.25}],
        }
    )
    cells: tuple[Literal["00", "10", "01", "11"], ...] = ("00", "10", "01", "11")
    cell_outcomes = {
        cell: pa.table(
            {
                "orders": [mean] * 31 + [mean + 0.5],
                "complaint": [1.0] * (16 if cell == "11" else 8)
                + [0.0] * (16 if cell == "11" else 24),
            }
        )
        for cell, mean in zip(cells, (10.0, 14.0, 13.0, 12.0), strict=True)
    }
    simulation = simulate_overlap(
        plan, n_target=125, cell_outcomes=cell_outcomes, replications=96, seed=321
    )
    benefit = simulation.per_claim["orders", "joint_launch"]
    assert benefit.truth == 2.0
    assert 0 < benefit.n_reject == benefit.n_decision_ready < benefit.n_fit < 96
    assert benefit.power == benefit.n_reject / 96
    guardrail = simulation.per_claim["complaint", "joint_launch"]
    assert guardrail.truth == guardrail.threshold == 0.25
    assert guardrail.rate_kind == "type_i_error"
    assert guardrail.power is None
    assert guardrail.rejection_rate == guardrail.n_reject / 96
    assert simulation.joint_criteria_met_count <= guardrail.n_reject
    neg = simulation.per_claim["orders", "b_given_a_on"]
    assert neg.truth == -2.0
    assert neg.rate_kind == "type_i_error" and neg.power is None

    empty = simulate_overlap(
        plan, n_target=0, cell_outcomes=cell_outcomes, replications=23, seed=321
    )
    assert empty.replications == 23 and empty.joint_criteria_met_count == 0
    for claim in empty.claims:
        assert (claim.n_point, claim.n_fit, claim.n_reject) == (0, 0, 0)
        assert claim.rejection_rate == 0.0
        assert claim.mc_interval[1] > 0
