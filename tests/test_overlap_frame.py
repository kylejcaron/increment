"""Frame capture and archived SDK decision replay contracts."""

import json
from datetime import datetime, timedelta
from fractions import Fraction
from hashlib import sha256
from math import inf, nextafter
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from increment.assignment import Decision, compile_allocation
from increment.errors import CodedError
from increment.overlap import (
    OverlapCapture,
    OverlapEvidence,
    OverlapPlan,
    OverlapSdkArchive,
    capture_overlap_frame,
    sdk_overlap_decision,
    sdk_overlap_plan,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "overlap"
_SCHEMAS = {
    "roster": pa.schema(
        {
            "study_id": pa.string(),
            "unit_id": pa.string(),
            "entry_at": pa.timestamp("us", "UTC"),
            "eligible_a": pa.bool_(),
            "eligible_b": pa.bool_(),
            "eligibility_at_a": pa.timestamp("us", "UTC"),
            "eligibility_at_b": pa.timestamp("us", "UTC"),
            "gate_a": pa.bool_(),
            "gate_b": pa.bool_(),
            "gate_at_a": pa.timestamp("us", "UTC"),
            "gate_at_b": pa.timestamp("us", "UTC"),
        }
    ),
    "decisions": pa.schema(
        {
            "study_id": pa.string(),
            "unit_id": pa.string(),
            "target": pa.string(),
            "decided_at": pa.timestamp("us", "UTC"),
            "revision": pa.string(),
            "archive_sha256": pa.string(),
            "selected_experiment": pa.string(),
            "status": pa.string(),
            "arm": pa.string(),
            "assigned_weight": pa.int64(),
        }
    ),
    "deliveries": pa.schema(
        {
            "study_id": pa.string(),
            "event_id": pa.string(),
            "unit_id": pa.string(),
            "target": pa.string(),
            "experiment": pa.string(),
            "revision": pa.string(),
            "delivered_arm": pa.string(),
            "delivered_at": pa.timestamp("us", "UTC"),
        }
    ),
    "outcomes": pa.schema(
        {
            "study_id": pa.string(),
            "unit_id": pa.string(),
            "metric": pa.string(),
            "definition_sha256": pa.string(),
            "window_start": pa.timestamp("us", "UTC"),
            "window_end": pa.timestamp("us", "UTC"),
            "value": pa.float64(),
            "finalized_at": pa.timestamp("us", "UTC"),
        }
    ),
}


@pytest.fixture
def external():
    data = json.loads((_FIXTURES / "contract_v1.json").read_bytes())
    plan = OverlapPlan.model_validate(data["evidence"]["plan"])
    capture = OverlapCapture.model_validate(data["evidence"]["capture"])
    logs = data["external"]
    for role, schema in _SCHEMAS.items():
        for row in logs[role]:
            for field in schema:
                if pa.types.is_timestamp(field.type) and row.get(field.name) is not None:
                    row[field.name] = datetime.fromisoformat(row[field.name])
    return plan, capture, logs, data


def _frames(logs):
    return {role: pa.Table.from_pylist(logs[role], schema) for role, schema in _SCHEMAS.items()}


def _capture(plan, capture, logs) -> OverlapEvidence:
    return capture_overlap_frame(plan, capture=capture, **_frames(logs))


def _expect(error, code, **context):
    assert error.value.code == code
    assert error.value.context == context


def _meta(plan, capture):
    return capture.model_copy(
        update={
            "origin": plan.law.origin,
            "archive_sha256": plan.law.archive_sha256,
            "law_sha256": plan.law_sha256,
        }
    )


def _sdk_plan(
    plan: OverlapPlan,
    a_config: bytes | OverlapSdkArchive,
    b_config: bytes | OverlapSdkArchive | None = None,
    *,
    a_target: str = "A",
    b_target: str = "B",
    b_experiment: str = "B",
    a_off: str = "off",
    a_on: str = "on",
    target: Literal["jointly_eligible", "jointly_eligible_and_enrolled"] | None = None,
) -> OverlapPlan:
    return sdk_overlap_plan(
        a_config=a_config,
        a_target=a_target,
        a_experiment="A",
        a_off=a_off,
        a_on=a_on,
        b_config=a_config if b_config is None else b_config,
        b_target=b_target,
        b_experiment=b_experiment,
        b_off="off",
        b_on="on",
        study_id=plan.study_id,
        unit_namespace=plan.unit_namespace,
        population_id=plan.population_id,
        eligibility_rule_sha256=plan.eligibility_rule_sha256,
        enrollment_rule_sha256=plan.enrollment_rule_sha256,
        registered_at=plan.registered_at,
        entry_open=plan.entry_open,
        start=plan.start,
        end=plan.end,
        target=plan.target if target is None else target,
        metrics=plan.metrics,
        claims=plan.claims,
        alpha=plan.alpha,
        serving_barrier_id=plan.law.serving_barrier_id,
    )


def test_sdk_two_target_spine_includes_never_delivered(external):
    plan, capture, logs, _ = external
    fixture = json.loads((_FIXTURES / "sdk_config.json").read_bytes())
    raws = (
        json.dumps(fixture, separators=(",", ":")).encode(),
        json.dumps(fixture, indent=2).encode(),
    )
    evalu = compile_allocation(fixture)
    project = sdk_overlap_decision
    results = []
    for raw in raws:
        receipt = OverlapSdkArchive.from_bytes(raw)
        sdk_plan = _sdk_plan(plan, raw, receipt)
        assert sdk_plan.a.arm_probabilities[1] == Fraction(3, 4)
        assert sdk_plan.b.arm_probabilities[1] == Fraction(1, 4)
        normalized = []
        for row in logs["decisions"]:
            target = "A" if row["target"] == "target-a" else "B"
            decision = evalu.decide(target, row["unit_id"])
            normalized.append(
                project(
                    study_id=plan.study_id,
                    unit_id=row["unit_id"],
                    decided_at=row["decided_at"],
                    decision=decision,
                    config=receipt,
                )
            )
        assert {row["archive_sha256"] for row in normalized} == {sha256(raw).hexdigest()}
        result = _capture(sdk_plan, _meta(sdk_plan, capture), {**logs, "decisions": normalized})
        assert result.target_count == 2
        assert result.eligibility_counts == (0, 0, 1, 2)
        assert sum(row.n_without_delivery for row in result.delivery_counts) == 4
        results.append((sdk_plan, result, normalized))
    first_plan, first, first_rows = results[0]
    _, second, second_rows = results[1]
    assert first.cells == second.cells
    assert first.evidence_sha256 != second.evidence_sha256
    with pytest.raises(CodedError) as error:
        _capture(
            first_plan, first.capture, {**logs, "decisions": [second_rows[0], *first_rows[1:]]}
        )
    _expect(error, "overlap.evidence.invalid", role="decisions", reason="archive_mismatch", count=1)
    unreadable = cast(pa.Table, None)
    with pytest.raises(CodedError) as error:
        capture_overlap_frame(
            first_plan,
            capture=second.capture,
            roster=unreadable,
            decisions=unreadable,
            deliveries=unreadable,
            outcomes=unreadable,
        )
    _expect(error, "overlap.capture.invalid", role="decisions", reason="archive_mismatch")


def test_sdk_standalone_decision_cannot_be_unallocated_or_select_a_sibling(external):
    plan, capture, logs, data = external
    raw = data["sdk"]["archives"]["full"]["utf8"].encode()
    receipt = OverlapSdkArchive.from_bytes(raw)
    plan = _sdk_plan(plan, receipt)
    evaluator = compile_allocation(json.loads(raw))
    rows = [
        sdk_overlap_decision(
            study_id=plan.study_id,
            unit_id=row["unit_id"],
            decided_at=row["decided_at"],
            decision=evaluator.decide("A" if row["target"] == "target-a" else "B", row["unit_id"]),
            config=receipt,
        )
        for row in logs["decisions"]
    ]
    rows[0].update(status="unallocated", selected_experiment=None, arm=None, assigned_weight=None)
    rows[2]["selected_experiment"] = "sibling-experiment"
    with pytest.raises(CodedError) as error:
        _capture(plan, _meta(plan, capture), {**logs, "decisions": rows})
    _expect(
        error,
        "overlap.evidence.invalid",
        role="decisions",
        reason="invalid_disposition",
        count=2,
    )


def test_sdk_plan_derives_admission_and_draw_domains(external):
    plan, _, _, data = external
    raw = {label: rec["utf8"].encode() for label, rec in data["sdk"]["archives"].items()}
    with pytest.raises(CodedError) as error:
        _sdk_plan(plan, raw["partial"])
    _expect(error, "overlap.law.unsupported", reason="admission_not_guaranteed", factors=("A", "B"))
    admitted = _sdk_plan(plan, raw["partial"], target="jointly_eligible_and_enrolled")
    assert admitted.law.admission == "joint_gate_required"
    with pytest.raises(CodedError) as error:
        _sdk_plan(plan, raw["same_layer"], a_target="shared", b_target="shared")
    _expect(error, "overlap.law.unsupported", reason="same_layer", factors=("A", "B"))
    new_revision = raw["full"].replace(b"archive-v1", b"archive-v2")
    with pytest.raises(CodedError) as error:
        _sdk_plan(plan, raw["full"], new_revision, b_target="A", b_experiment="A")
    _expect(error, "overlap.law.unsupported", reason="duplicate_experiment", factors=("A", "B"))
    baseline = _sdk_plan(plan, raw["full"])
    revised = _sdk_plan(plan, new_revision)
    assert baseline.a.arm_draw_id == revised.a.arm_draw_id != revised.b.arm_draw_id
    domain = b"increment.alloc.v1\0\0\0\0\x03arm\0\0\0\x01A"
    assert baseline.a.arm_draw_id == sha256(domain).hexdigest()
    swapped = _sdk_plan(plan, raw["full"], a_off="on", a_on="off")
    assert swapped.a.arm_probabilities[1] == Fraction(1, 4)
    with pytest.raises(CodedError) as error:
        _sdk_plan(plan, raw["full"], a_off="absent")
    _expect(error, "overlap.plan.invalid", field="a_off", reason="unknown_arm")


def test_sdk_archive_rejects_decoded_and_ambiguous_input(external):
    plan, _, _, data = external
    archive_cls = OverlapSdkArchive
    raw = data["sdk"]["archives"]["full"]["utf8"].encode()
    for invalid, reason in (
        (cast(bytes, json.loads(raw)), "archive_requires_bytes"),
        (b"\xff", "invalid_json"),
        (raw.replace(b'"protocol":1', b'"protocol":1,"protocol":1'), "duplicate_key"),
    ):
        with pytest.raises(CodedError) as error:
            archive_cls.from_bytes(invalid)
        _expect(error, "overlap.plan.invalid", field="config", reason=reason)
        assert error.value.__context__ is None
    receipt = archive_cls.from_bytes(raw)
    revised = raw.replace(b'"weight_bps":2500', b'"weight_bps":5000', 1)
    with pytest.raises(CodedError):
        _sdk_plan(plan, revised)
    assert _sdk_plan(plan, receipt).a.arm_probabilities[1] == Fraction(3, 4)


def test_sdk_replay_rejects_forged_sibling_stage_weights(external):
    plan, _, _, data = external
    raw = data["sdk"]["archives"]["same_layer"]["utf8"].encode()
    config = json.loads(raw)
    config["holdouts"] = [{"key": "upstream", "held_out_bps": 500, "targets": ["shared"]}]
    raw = json.dumps(config).encode()
    evalu = compile_allocation(config)
    receipt = OverlapSdkArchive.from_bytes(raw)
    project = sdk_overlap_decision
    for i in range(100):
        unit = str(i)
        decision = evalu.decide("shared", unit)
        if decision["status"] == "assigned" and decision["experiment"] == "B":
            break
    else:
        pytest.fail("no archived sibling draw")
    fields = {
        "study_id": plan.study_id,
        "unit_id": unit,
        "decided_at": plan.entry_open,
        "config": receipt,
    }
    result = project(**fields, decision=decision)
    assert result["selected_experiment"] == "B"
    membership = decision["membership"]
    assert membership is not None
    assert result["assigned_weight"] in (2500, 7500)
    for bad in (
        {**decision, "arm_bps": True},
        {**decision, "arm_bps": 5000},
        {**decision, "membership": {**membership, "group_bps": 1}},
        {**decision, "status": "held_out"},
    ):
        with pytest.raises(CodedError) as error:
            project(**fields, decision=cast(Decision, bad))
        _expect(error, "evidence.assignment.replay_mismatch", details="decision")


def test_external_negative_gate_is_not_a_missing_decision(external):
    plan, capture, logs, _ = external
    law = plan.law.model_copy(update={"admission": "joint_gate_required"})
    plan = plan.model_copy(update={"target": "jointly_eligible_and_enrolled", "law": law})
    logs["roster"][0]["gate_b"] = False
    logs["decisions"][1].update(status="not_enrolled", arm=None, assigned_weight=None)
    result = _capture(plan, _meta(plan, capture), logs)
    assert result.eligibility_counts == (0, 0, 1, 2)
    assert result.target_count == 1
    assert result.gate_counts[3] == 1
    assert [row.n_assigned for row in result.cells[:4]] == [0, 0, 1, 0]
    logs["roster"][1]["gate_a"] = False
    logs["decisions"][2]["selected_experiment"] = "sibling-experiment"
    sibling = _capture(plan, _meta(plan, capture), logs)
    assert sibling.target_count == 0 and sibling.eligibility_counts[3] == 2
    assert sibling.gate_counts[20] == 1  # other_selected A, assigned B
    assert all(row.n_assigned == 0 for row in sibling.cells)
    logs["roster"] = logs["roster"][2:]
    logs["decisions"] = logs["decisions"][4:]
    no_joint = _capture(plan, _meta(plan, capture), logs)
    assert no_joint.target_count == no_joint.eligibility_counts[3] == 0


def test_duplicate_or_missing_unit_metric_cannot_inflate_precision(external):
    plan, capture, logs, _ = external
    duplicate = logs["outcomes"][0].copy()
    logs["outcomes"].extend([duplicate, duplicate.copy()])
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(error, "overlap.evidence.invalid", role="outcomes", reason="duplicate_key", count=1)
    del logs["outcomes"][-2:]
    logs["outcomes"].pop(0)
    result = _capture(plan, capture, logs)
    assert result.cells[1].n_assigned == result.cells[1].n_missing == 1
    assert result.cells[1].anchor is None
    assert (result.cells[5].n_observed, result.cells[5].n_success, result.cells[5].anchor) == (
        1,
        0,
        0,
    )


def test_deliveries_change_only_diagnostics_and_cannot_predate_start(external):
    plan, capture, logs, _ = external
    baseline = _capture(plan, capture, logs)
    event = {
        "study_id": plan.study_id,
        "event_id": "one",
        "unit_id": "001",
        "target": "target-a",
        "experiment": "experiment-a",
        "revision": "rev-a",
        "delivered_arm": "on",
        "delivered_at": plan.start,
    }
    logs["deliveries"].extend(
        [event, {**event, "event_id": "two"}, {**event, "event_id": "wrong", "revision": "old"}]
    )
    result = _capture(plan, capture, logs)
    assert baseline.cells == result.cells
    row = result.delivery_counts[2]
    assert (row.n_without_delivery, row.n_matching, row.n_inconsistent) == (0, 2, 1)
    logs["deliveries"][-1]["delivered_at"] -= timedelta(microseconds=1)
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(
        error,
        "overlap.evidence.invalid",
        role="deliveries",
        reason="delivery_before_start",
        count=1,
    )


def test_delivery_diagnostics_cover_only_the_half_open_outcome_window(external):
    plan, capture, logs, _ = external
    event = {
        "study_id": plan.study_id,
        "unit_id": "001",
        "target": "target-a",
        "experiment": "experiment-a",
        "revision": "rev-a",
        "delivered_arm": "on",
    }
    logs["deliveries"].extend(
        [
            {**event, "event_id": "last", "delivered_at": plan.end - timedelta(microseconds=1)},
            {**event, "event_id": "at-end", "delivered_at": plan.end},
            {
                **event,
                "event_id": "after-end",
                "unit_id": "x",
                "delivered_at": plan.end + timedelta(microseconds=1),
            },
        ]
    )
    result = _capture(plan, capture, logs)
    assert sum(row.n_matching for row in result.delivery_counts) == 1
    assert not any(row.n_inconsistent for row in result.delivery_counts)
    logs["deliveries"][-1]["delivered_at"] = capture.cutoff + timedelta(microseconds=1)
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(
        error,
        "overlap.evidence.invalid",
        role="deliveries",
        reason="delivery_after_cutoff",
        count=1,
    )


def test_both_ineligible_roster_timestamps_must_precede_focal_delivery(external):
    plan, capture, logs, _ = external
    logs["roster"][2].update(eligible_b=False, eligibility_at_a=plan.start)
    logs["decisions"].pop()
    assert _capture(plan, capture, logs).eligibility_counts == (1, 0, 0, 2)
    event = {
        "study_id": plan.study_id,
        "unit_id": "x",
        "target": plan.a.target,
        "experiment": plan.a.experiment,
        "revision": plan.a.revision,
        "delivered_arm": plan.a.off,
        "delivered_at": plan.start,
    }
    logs["deliveries"].extend({**event, "event_id": name} for name in ("first", "second"))
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(
        error,
        "overlap.evidence.invalid",
        role="deliveries",
        reason="eligibility_not_before_delivery",
        count=2,
    )


def test_pinned_cutoff_and_incomplete_outcomes_do_not_shrink_target(external):
    plan, capture, logs, _ = external
    outcomes = capture.feed("outcomes").model_copy(update={"complete_through": None})
    partial = capture.model_copy(update={"feeds": (*capture.feeds[:3], outcomes)})
    receipt = _capture(plan, partial, logs)
    assert [(cell.n_assigned, cell.n_pending, cell.n_observed) for cell in receipt.cells[:4]] == [
        (0, 0, 0),
        (1, 1, 0),
        (1, 1, 0),
        (0, 0, 0),
    ]
    logs["outcomes"][0]["finalized_at"] = None
    assert _capture(plan, capture, logs).cells[1].n_pending == 1
    logs["outcomes"][0]["finalized_at"] = capture.cutoff + timedelta(microseconds=1)
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(
        error, "overlap.evidence.invalid", role="outcomes", reason="finalized_after_cutoff", count=1
    )


def test_frame_schema_and_scoped_values_fail_with_no_raw_exit(external):
    plan, capture, logs, _ = external
    frames = _frames(logs)
    frames["roster"] = frames["roster"].set_column(1, "unit_id", pa.array([1, 1, 2]))
    with pytest.raises(CodedError) as error:
        capture_overlap_frame(plan, capture=capture, **frames)
    _expect(error, "overlap.evidence.invalid", role="roster", reason="invalid_column_type", count=1)
    logs["outcomes"][0]["value"] = inf
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(error, "overlap.evidence.invalid", role="outcomes", reason="invalid_value", count=1)
    logs["outcomes"][0]["study_id"] = "foreign"
    assert _capture(plan, capture, logs).cells[1].n_missing == 1
    logs["decisions"][0]["revision"] = "old"
    with pytest.raises(CodedError) as error:
        _capture(plan, capture, logs)
    _expect(
        error, "overlap.evidence.invalid", role="decisions", reason="revision_mismatch", count=1
    )


def test_centered_offset_matches_exact_rationals_under_order_and_backend(external):
    plan, capture, logs, _ = external
    base = 1e16
    second = nextafter(base, inf)
    values = (base, second, nextafter(second, inf), second)
    logs["outcomes"][0]["value"] = values[0]
    for index, value in enumerate(values[1:]):
        unit = f"copy-{index}"
        logs["roster"].append({**logs["roster"][0], "unit_id": unit})
        logs["decisions"].extend({**row, "unit_id": unit} for row in logs["decisions"][:2])
        logs["outcomes"].append({**logs["outcomes"][0], "unit_id": unit, "value": value})
        logs["outcomes"].append(
            {**logs["outcomes"][3], "unit_id": unit, "value": float(index != 0)}
        )
    arrow = _capture(plan, capture, logs)
    scrambled = {
        role: cast(
            pl.DataFrame, pl.from_arrow(pa.concat_tables([frame.slice(2), frame.slice(0, 2)]))
        )
        for role, frame in _frames(logs).items()
    }
    assert capture_overlap_frame(plan, capture=capture, **scrambled).cells == arrow.cells
    deltas = [Fraction(x) - Fraction(base) for x in values]
    center = sum(deltas) / len(deltas)
    cell = arrow.cells[1]
    assert (cell.n_assigned, cell.anchor, cell.mean_delta, cell.m2) == (
        4,
        base,
        float(center),
        float(sum((d - center) ** 2 for d in deltas)),
    )
    binary = arrow.cells[5]
    assert (binary.n_success, binary.anchor, binary.mean_delta, binary.m2) == (2, 0, 0.5, 1)


def test_nanosecond_frame_cannot_round_ambiguous_chronology(external):
    plan, capture, logs, _ = external
    frames = _frames(logs)
    frames["roster"] = cast(pl.DataFrame, pl.from_arrow(frames["roster"])).with_columns(
        pl.col("eligibility_at_b").dt.cast_time_unit("ns") + pl.duration(nanoseconds=1)
    )
    with pytest.raises(CodedError) as error:
        capture_overlap_frame(plan, capture=capture, **frames)
    _expect(error, "overlap.evidence.invalid", role="roster", reason="invalid_value", count=3)


def test_nullable_pandas_projection_preserves_measurement_null_and_nan(external):
    plan, capture, logs, _ = external
    logs["outcomes"][0]["value"] = None
    frames = {
        role: frame.to_pandas(types_mapper=pd.ArrowDtype) for role, frame in _frames(logs).items()
    }
    assert capture_overlap_frame(plan, capture=capture, **frames).cells[1].n_missing == 1
    logs["outcomes"][0]["value"] = float("nan")
    frames["outcomes"] = _frames(logs)["outcomes"].to_pandas(types_mapper=pd.ArrowDtype)
    with pytest.raises(CodedError) as error:
        capture_overlap_frame(plan, capture=capture, **frames)
    _expect(error, "overlap.evidence.invalid", role="outcomes", reason="invalid_value", count=1)


def test_pandas_index_does_not_supply_any_physical_columns(external):
    plan, capture, logs, _ = external
    frames = {
        role: frame.to_pandas(types_mapper=pd.ArrowDtype) for role, frame in _frames(logs).items()
    }
    frames["roster"].index = pd.Index([1, 2, 3], name="__overlap_submicrosecond")
    frames["decisions"].index = pd.Index([object()] * len(frames["decisions"]))
    result = capture_overlap_frame(plan, capture=capture, **frames)
    assert result == _capture(plan, capture, logs)
