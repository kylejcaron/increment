"""Check that research evidence cannot substitute for a matched decision."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from calibration import harm_campaign as campaign
from calibration import harm_case as case
from calibration import harm_verdict as verdict
from calibration.harm_bounds import StudyError
from calibration.harm_case import _FieldDecision, _OwnerDecision

if TYPE_CHECKING:
    from pathlib import Path


_PROTOCOL = "1" * 64
_CASE_SHA = "2" * 64
_CARD = "3" * 64
_SOURCE = "4" * 64
_EMPTY_CSV = b"scope,name,tau,outer_lower,outer_upper,decision,reason,threshold_origin\r\n"
_PIPELINES = ("P0", "P1", "P2", "P3")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _action(key: str, kind: str) -> dict[str, str]:
    return {"key": key, "kind": kind, "cost_bearer": "sponsor"}


def _owner(*, best_gain: bool = True, learned: bool = False) -> _OwnerDecision:
    clear = _action("action_if_clear", "clear")
    rule_out = _action("action_if_rule_out", "rule_out")
    abstain = _action("action_if_inconclusive", "abstain")
    actionable = {
        "action": clear,
        "decision": "clear",
        "changes_action": True,
        "gain_exceeds_minimum": best_gain,
        "best_gain_exceeds_minimum": best_gain,
        "g_min": 0.15,
        "g_max": 0.15,
        "ambiguity": None,
        "guess_vs_wait_worst_excess": None,
    }
    unresolved = {
        "action": abstain,
        "decision": "inconclusive",
        "changes_action": False,
        "gain_exceeds_minimum": False,
        "best_gain_exceeds_minimum": False,
        "g_min": -0.1,
        "g_max": -0.1,
        "ambiguity": "sampling_or_unresolved_identification",
        "guess_vs_wait_worst_excess": {"clear": 0.75, "rule_out": 0.75},
    }
    return _OwnerDecision.model_validate_json(
        json.dumps(
            {
                "tau": 0.25,
                "minimum_gain": 0.15,
                "pipelines": {
                    p: unresolved if learned and p != "P3" else actionable for p in _PIPELINES
                },
                "smallest_useful_simple_pipeline": "P0" if best_gain and not learned else None,
                "h_adds_decision": learned and best_gain,
                "irreducible_at_features": False,
                "current_action": _action("current_action", "abstain"),
                "incremental_workflow_cost": 0.1,
                "cost_false_clear": 1.0,
                "exact_costs": {
                    "minimum_gain": "0.15",
                    "incremental_workflow_cost": "0.1",
                    "cost_false_clear": "1.0",
                    "cost_false_rule_out": "1.0",
                    "cost_abstain": "0.250000000000000001" if best_gain else "0.25",
                },
                "cost_false_rule_out": 1.0,
                "cost_abstain": 0.25,
                "max_false_action_risk": 0.05,
                "alternatives": [clear, rule_out, abstain],
                "confirmation": None,
            }
        )
    )


def _field(
    owner: _OwnerDecision | None = None, *, learned: bool = False, protocol: str = _PROTOCOL
) -> _FieldDecision:
    completed = owner is not None
    return _FieldDecision.model_validate_json(
        json.dumps(
            {
                "schema_version": 2,
                "source": "owner_trial" if completed else "hillstrom",
                "status": "completed" if completed else "unavailable",
                "reason": None if completed else "missing_case_manifest",
                "protocol_sha256": protocol,
                "manifest_sha256": _SOURCE if completed else None,
                "card_sha256": _CARD if completed else None,
                "permission_verified": completed,
                "design_verified": completed,
                "bounds": (
                    {p: [0.125, 0.30 if learned and p != "P3" else 0.20] for p in _PIPELINES}
                    if completed
                    else None
                ),
                "owner_decision": owner.model_dump(mode="json") if owner else None,
                "diagnostics": []
                if completed
                else [
                    {
                        "reason": "missing_case_manifest",
                        "context": [["stage", "preflight"], ["coordinate", "manifest"]],
                    }
                ],
                "receipts": {"inputs": _SOURCE, "bounds": _SOURCE, "thresholds": _SOURCE},
                "reproduction": {
                    "source_revision": "a" * 40,
                    "protocol_base_revision": "77ab667b227b2005e18bcc3d0f56c23b6aaec651",
                    "harm_case_sha256": _SOURCE,
                    "harm_bounds_sha256": _SOURCE,
                    "python": "3.15.2",
                    "numpy": "2.5.4",
                    "scipy": "1.18.1",
                    "pydantic": "2.14.2",
                    "argv": ["python", "-m", "calibration.harm_case"],
                    "exact_arguments_sha256": _SOURCE,
                    "rng": "PCG64",
                    "seed": 20261119083,
                    "started_at": "2026-11-20T04:39:43Z",
                    "ended_at": "2026-11-20T04:39:44Z",
                },
            }
        )
    )


def _ack(
    *,
    decision_sha: str = _CASE_SHA,
    card_sha: str = _CARD,
    pipeline: str = "P0",
    action_key: str = "action_if_clear",
    signed_at: str = "2026-11-20T05:00:00Z",
    comparison_adequate: bool = False,
) -> verdict._Acknowledgment:
    return verdict._Acknowledgment.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "decision_sha256": decision_sha,
                "card_sha256": card_sha,
                "acknowledged_at": signed_at,
                "intended_action": {
                    "pipeline": pipeline,
                    "action_key": action_key,
                    "feasible_and_intended": True,
                },
                "all_admissible_baseline_features": False,
                "comparison_adequate": comparison_adequate,
            }
        )
    )


def test_green_campaign_without_owner_defers() -> None:
    result = verdict._recommend("green", _field(), None)
    assert result.recommendation == "defer"
    assert result.method is None
    assert "missing_case_manifest" in result.reasons
    assert "missing_eligible_owner_card" in result.reasons


@pytest.mark.parametrize("status", ("red", "inconclusive"))
def test_costed_owner_action_cannot_override_campaign(status: verdict._RunStatus) -> None:
    result = verdict._recommend(status, _field(_owner()), _ack())
    assert result.recommendation == "defer"
    assert result.method is None
    assert f"campaign_{status}" in result.reasons


@pytest.mark.parametrize(("learned", "expected"), [(False, "P0"), (True, "P3")])
def test_exact_gain_flags_qualify_despite_identical_display_values(
    learned: bool, expected: str
) -> None:
    owner = _owner(learned=learned)
    field = _field(owner, learned=learned)
    verdict._validate_owner(field, None)
    result = verdict._recommend("green", field, _ack(pipeline=expected))
    assert (result.recommendation, result.method) == ("implement", expected)
    assert owner.pipelines.P3.g_min == owner.minimum_gain


def test_best_compatible_gain_rejects_only_with_adequate_comparison() -> None:
    field = _field(_owner(best_gain=False))
    assert verdict._recommend("green", field, _ack()).recommendation == "defer"
    assert (
        verdict._recommend("green", field, _ack(comparison_adequate=True)).recommendation
        == "reject"
    )


def test_no_change_with_nonpositive_best_gain_can_reject() -> None:
    owner = _owner(best_gain=False)
    unchanged = owner.pipelines.P0.model_copy(
        update={"changes_action": False, "g_min": -0.1, "g_max": -0.1}
    )
    owner = owner.model_copy(
        update={
            "current_action": owner.current_action.model_copy(update={"kind": "clear"}),
            "pipelines": owner.pipelines.model_copy(update=dict.fromkeys(_PIPELINES, unchanged)),
        }
    )
    field = _field(owner)
    verdict._validate_owner(field, None)
    assert verdict._recommend("green", field, _ack()).recommendation == "defer"
    assert (
        verdict._recommend("green", field, _ack(comparison_adequate=True)).recommendation
        == "reject"
    )


def test_rounded_gain_and_consistent_flags_cannot_mask_forged_gain() -> None:
    owner = _owner(best_gain=False)
    forged = owner.pipelines.P0.model_copy(
        update={"gain_exceeds_minimum": True, "best_gain_exceeds_minimum": True}
    )
    owner = owner.model_copy(
        update={
            "smallest_useful_simple_pipeline": "P0",
            "pipelines": owner.pipelines.model_copy(update=dict.fromkeys(_PIPELINES, forged)),
        }
    )
    with pytest.raises(StudyError) as exc:
        verdict._validate_owner(_field(owner), None)
    assert exc.value.reason == "inconsistent_owner_evidence"


def test_full_feature_inner_span_requires_the_exact_q_map() -> None:
    field = _field(_owner())
    detail = case._Method.model_construct(
        L_ci=(0.125, 0.15),
        U_ci=(0.18, 0.20),
        outer=(0.125, 0.20),
        inner_span=(0.15, 0.18),
        leaf_map=(0, 0, 1, 1),
    )
    methods = case._Methods.model_construct(
        U0=detail.model_copy(update={"leaf_map": (0, 0, 0, 0)}),
        F=detail,
        Q=detail,
        H=detail,
        M=detail,
    )
    pipes = {
        name: case._Pipeline.model_construct(
            methods=members,
            outer=(0.125, 0.20),
            reason=None,
        )
        for name, members in zip(
            _PIPELINES, (("U0",), ("U0", "F"), ("U0", "F", "Q"), ("U0", "F", "Q", "H")), strict=True
        )
    }
    bounds = case._Bounds.model_construct(
        historical_benchmark=False,
        methods=methods,
        pipelines=case._Pipelines.model_construct(
            P0=pipes["P0"],
            P1=pipes["P1"],
            P2=pipes["P2"],
            P3=pipes["P3"],
        ),
    )
    with pytest.raises(StudyError) as exc:
        verdict._check_bounds(field, bounds)
    assert exc.value.reason == "invalid_field_leaf_map"


def test_missing_acknowledgment_never_promotes() -> None:
    result = verdict._recommend("green", _field(_owner()), None)
    assert (result.recommendation, result.method, result.reasons) == (
        "defer",
        None,
        ("missing_post_readout_acknowledgment",),
    )


def test_acknowledged_action_must_match_selected_pipeline() -> None:
    result = verdict._recommend("green", _field(_owner()), _ack(pipeline="P3"))
    assert result.recommendation == "defer"
    assert result.reasons == ("acknowledgment_action_mismatch",)


@pytest.mark.parametrize(
    ("decision_sha", "card_sha", "signed_at"),
    [
        ("a" * 64, _CARD, "2026-11-20T05:00:00Z"),
        (_CASE_SHA, "b" * 64, "2026-11-20T05:00:00Z"),
        (_CASE_SHA, _CARD, "2026-11-20T04:39:43Z"),
    ],
)
def test_acknowledgment_cannot_be_replayed_or_predate_readout(
    decision_sha: str, card_sha: str, signed_at: str
) -> None:
    ack = _ack(decision_sha=decision_sha, card_sha=card_sha, signed_at=signed_at)
    field = _field(_owner())
    with pytest.raises(StudyError) as exc:
        verdict._validate_ack(ack, field, _CASE_SHA, datetime(2026, 11, 21, tzinfo=UTC))
    assert exc.value.reason == "acknowledgment_binding_mismatch"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("protocol", "protocol_digest_mismatch"),
        ("inconsistent_provenance", "inconsistent_field_packet"),
        ("modified_artifact", "artifact_digest_mismatch"),
    ],
)
def test_field_provenance_fails_closed(tmp_path: Path, fault: str, reason: str) -> None:
    field = _field(protocol="0" * 64 if fault == "protocol" else _PROTOCOL)
    inputs = json.dumps(
        {
            "schema_version": 2,
            "source": "hillstrom",
            "manifest_sha256": None,
            "design_sha256": None,
            "card_sha256": None,
            "csv_sha256": None,
            "expected_csv_sha256": None,
            "permission_verified": fault == "inconsistent_provenance",
            "design_verified": False,
            "digest_verified": False,
            "assignment": None,
            "treatment_probability": None,
        }
    ).encode()
    bounds = json.dumps(
        {
            "schema_version": 2,
            "protocol_sha256": _PROTOCOL,
            "status": "unavailable",
            "reason": "missing_case_manifest",
            "bounds": None,
        }
    ).encode()
    artifacts = {"input_manifest.json": inputs, "bounds.json": bounds, "thresholds.csv": _EMPTY_CSV}
    document = field.model_dump(mode="json", by_alias=True)
    document["receipts"] = {name: _sha(raw) for name, raw in artifacts.items()}
    (tmp_path / "decision.json").write_text(json.dumps(document))
    for name, raw in artifacts.items():
        (tmp_path / name).write_bytes(raw + b" " if fault == "modified_artifact" else raw)

    with pytest.raises(StudyError) as exc:
        verdict._read_field(tmp_path, _PROTOCOL, _SOURCE, _SOURCE)
    assert exc.value.reason == reason


@pytest.mark.integration
@pytest.mark.parametrize("has_exact_costs", (False, True))
def test_archived_source_cannot_authorize_owner_action(
    tmp_path: Path, has_exact_costs: bool
) -> None:
    document = _field(_owner()).model_dump(mode="json", by_alias=True)
    if not has_exact_costs:
        del document["owner_decision"]["exact_costs"]
    (tmp_path / "decision.json").write_text(json.dumps(document))
    with pytest.raises(StudyError) as exc:
        verdict._read_field(tmp_path, _PROTOCOL, _SOURCE, "5" * 64, _SOURCE)
    assert exc.value.reason == (
        "inconsistent_field_provenance" if has_exact_costs else "invalid_evidence_schema"
    )


@pytest.mark.parametrize("counter", ("D3_P0_clear_50", "simultaneous_noncoverage"))
def test_campaign_gates_cannot_contradict_aggregate_rows(counter: str) -> None:
    coord = campaign._Coordinate(dgp=3, n_index=1, e_index=0, n=4096, e=0.5)
    n = 20_000
    rows = tuple(
        campaign._DecisionRow.model_construct(
            coordinate=coord,
            mode="estimated",
            target=target,
            tau=0.5,
            attempted=n,
            evaluated=n,
            clear=15_000,
            rule_out=0,
            inconclusive=5_000,
            false_clear=0,
            false_rule_out=0,
            false_id_clear=0,
            false_id_rule_out=0,
            id_abstain=0,
            sampling_abstain=5_000,
            diagnostic_abstain=0,
            diagnostic_abstain_reason=None,
        )
        for target in ("U0", "P0")
    )
    power = 20_000 if counter == "D3_P0_clear_50" else 15_000
    union = 0 if counter == "simultaneous_noncoverage" else 1
    counters = {
        "simultaneous_noncoverage": union,
        "any_false_clearance": 0,
        "any_false_rule_out": 0,
        "D3_P0_clear_50": power,
    }
    with pytest.raises(StudyError) as exc:
        verdict._reconcile_cell(n, 0.0, (1,), rows, counters)
    assert exc.value.reason == "inconsistent_campaign_counter_evidence"


@pytest.mark.integration
def test_failed_campaign_does_not_emit_unverified_oracle_or_diagnostics(tmp_path: Path) -> None:
    folder = tmp_path / "campaign"
    folder.mkdir()
    artifacts = {
        "cells.jsonl": b"",
        "decisions.jsonl": b"",
        "gates.json": json.dumps(
            {
                "schema_version": 1,
                "status": "execution_failure",
                "certification_eligible": False,
                "counters": [],
            }
        ).encode(),
    }
    for name, raw in artifacts.items():
        (folder / name).write_bytes(raw)
    manifest = campaign._Manifest(
        mode="full",
        certification_eligible=False,
        status="execution_failure",
        exit_code=1,
        source_revision=None,
        protocol_sha256=campaign._PROTOCOL_DIGEST,
        oracle_sha256="private-oracle-path",
        sources=(),
        inputs=(),
        artifacts=tuple(
            campaign._Receipt(path=name, sha256=_sha(raw)) for name, raw in artifacts.items()
        ),
        invocation=("private-executable",),
        python="unknown",
        numpy="unknown",
        scipy="unknown",
        pydantic="unknown",
        thread_environment=(),
        started_at=datetime(2026, 11, 20, tzinfo=UTC),
        ended_at=datetime(2026, 11, 20, tzinfo=UTC),
        trials_per_cell=20_000,
        completed_cells=0,
        failures=(
            campaign._Diagnostic(
                reason="private-failure-data", context=(("coordinate", "private-row-data"),)
            ),
        ),
    )
    (folder / "manifest.json").write_text(manifest.model_dump_json())
    out = tmp_path / "out"
    assert (
        verdict.main(
            ["--campaign", str(folder), "--case", str(tmp_path / "missing"), "--out", str(out)]
        )
        == 1
    )
    raw = (out / "verdict.json").read_text() + (out / "verdict.md").read_text()
    assert "private-" not in raw
    assert json.loads((out / "verdict.json").read_text())["oracle_sha256"] is None
