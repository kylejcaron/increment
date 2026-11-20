import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from calibration.harm_bounds import StudyError, decide
from calibration.harm_case import DecisionCard, _hillstrom_rows, _usefulness, main


def _card_fields() -> dict[str, object]:
    return {
        "owner": "Synthetic sponsor",
        "decision_id": "synthetic-decision",
        "signed_at": "2000-01-01T00:00:00Z",
        "unblind_at": "2000-01-02T00:00:00Z",
        "target": "Synthetic target",
        "arms": ["control", "treatment"],
        "outcome": "purchase",
        "horizon": "14 days",
        "positive_direction": "one_is_beneficial",
        "tau": 0.25,
        "max_false_action_risk": 0.05,
        "cost_false_clear": 10.0,
        "cost_false_rule_out": 1.0,
        "cost_abstain": 0.25,
        "incremental_workflow_cost": 0.10,
        "minimum_gain": 0.10,
        "currency_or_loss_unit": "synthetic loss units",
        "action_if_clear": {"kind": "clear", "description": "Send offer", "cost_bearer": "sponsor"},
        "action_if_rule_out": {
            "kind": "rule_out",
            "description": "Do not send offer",
            "cost_bearer": "sponsor",
        },
        "action_if_inconclusive": {
            "kind": "abstain",
            "description": "Wait for evidence",
            "cost_bearer": "sponsor",
        },
        "current_action": {
            "kind": "abstain",
            "description": "Wait for evidence",
            "cost_bearer": "sponsor",
        },
        "representativeness_evidence": "synthetic-population-document",
        "permission_ref": "synthetic-permission-document",
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def fixture_trial(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    data = tmp_path / "synthetic-private-data.csv"
    rows = ["unit_id,arm,y,s,b"]
    for unit in reversed(range(400)):
        arm, s, b = unit % 2, unit // 4 % 2, unit // 2 % 2
        rows.append(f"fixture-id-{unit:03},{arm},{arm},{s},{b}")
    data.write_text("\n".join(rows) + "\n", encoding="utf-8")
    design = _write_json(
        tmp_path / "synthetic-design.json",
        {
            "arms": ["control", "treatment"],
            "outcome": "purchase",
            "horizon": "14 days",
            "positive_direction": "one_is_beneficial",
            "assignment": "bernoulli",
            "treatment_probability": 0.5,
            "assumptions_ref": "synthetic-itt-complete-followup-sampling-evidence",
        },
    )
    return data, {
        "csv_path": str(data),
        "expected_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "permission_ref": "synthetic-permission-document",
        "design_ref": str(design),
        "followup_ref": "synthetic-complete-followup",
        "unit_ref": "synthetic-one-unit-one-row",
        "sampling_target": "Synthetic target",
        "analysis_permitted": True,
        "aggregates_permitted": True,
    }


def _run(
    tmp_path: Path, manifest: Path, *args: str, source: str = "owner_trial"
) -> tuple[int, Path]:
    out = tmp_path / "packet"
    code = main(["--source", source, "--case-manifest", str(manifest), "--out", str(out), *args])
    return code, out


@pytest.mark.integration
def test_missing_case_manifest_leaves_verifiable_null_packet(tmp_path: Path) -> None:
    code, out = _run(tmp_path, tmp_path / "absent.json", source="hillstrom")
    assert code == 2
    decision = _read_json(out / "decision.json")
    assert decision["status"] == "unavailable"
    assert decision["reason"] == "missing_case_manifest"
    assert decision["bounds"] is decision["owner_decision"] is None
    assert _read_json(out / "bounds.json")["bounds"] is None
    assert _read_json(out / "input_manifest.json")["csv_sha256"] is None
    assert set(decision["receipts"]) == {"input_manifest.json", "bounds.json", "thresholds.csv"}
    for name, sha in decision["receipts"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha


@pytest.mark.integration
def test_missing_owner_never_implies_clearance(
    tmp_path: Path, fixture_trial: tuple[Path, dict[str, object]]
) -> None:
    _, manifest = fixture_trial
    code, out = _run(tmp_path, _write_json(tmp_path / "manifest.json", manifest))
    assert code == 2
    decision = _read_json(out / "decision.json")
    assert (decision["status"], decision["reason"]) == ("research_only", "missing_decision_card")
    assert decision["owner_decision"] is None
    bounds = _read_json(out / "bounds.json")["bounds"]
    assert bounds["observed"] == {"n0": 200, "y0": 0, "n1": 200, "y1": 200, "mean_difference": 1.0}
    assert bounds["split"] == {"training": 133, "estimation": 267}
    assert bounds["methods"]["H"]["leaf_counts"] == [
        {"n0": 69, "y0": 0, "n1": 65, "y1": 65},
        {"n0": 64, "y0": 0, "n1": 69, "y1": 69},
    ]
    assert bounds["methods"]["U0"]["plugin"] == [0, 0]
    outer = bounds["methods"]["U0"]["outer"]
    assert outer == pytest.approx([0, 1 - (1 / 600) ** (1 / 200)], abs=1e-12)
    rows = tuple(csv.DictReader(io.StringIO((out / "thresholds.csv").read_text(encoding="utf-8"))))
    row = next(
        row
        for row in rows
        if row["scope"] == "pipeline" and row["name"] == "P0" and row["tau"] == "0.25"
    )
    assert row["decision"] == "clear"
    assert len(rows) == 9 * 13


def test_break_even_and_ties() -> None:
    card = DecisionCard.model_validate_json(json.dumps(_card_fields()))
    band = (0.125, 0.20)
    assert _usefulness(band, "clear", card) == pytest.approx((0.15, 0.15), abs=1e-12)
    assert decide(band, 0.2) == "inconclusive"
    at_tie = DecisionCard.model_validate_json(json.dumps({**_card_fields(), "tau": 0.20}))
    assert _usefulness(band, "clear", at_tie) == pytest.approx((-9.85, 0.15), abs=1e-12)
    assert _usefulness((0.2, 0.2), "rule_out", at_tie) == pytest.approx((-0.85, -0.85), abs=1e-12)


@pytest.mark.parametrize(
    ("manifest_change", "reason", "expected_code"),
    [
        ({"aggregates_permitted": False}, "use_permission_unconfirmed", 2),
        ({"design_ref": ""}, "unverified_trial_design", 2),
        ({"expected_sha256": "0" * 64}, "digest_mismatch", 2),
        ({"analysis_permitted": "true"}, "invalid_case_manifest", 1),
    ],
)
@pytest.mark.integration
def test_access_checks_precede_outcome_parsing(
    tmp_path: Path,
    fixture_trial: tuple[Path, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    manifest_change: dict[str, object],
    reason: str,
    expected_code: int,
) -> None:
    data, manifest = fixture_trial
    data.write_text("unit_id,arm,y,s,b\nsynthetic,0,not-binary,0,0\n", encoding="utf-8")
    original_open = Path.open
    if reason != "digest_mismatch":

        def refuse_data_open(path: Path, *args, **kwargs):
            if path == data:
                pytest.fail("data opened before source permission and design checks")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", refuse_data_open)
    code, out = _run(
        tmp_path, _write_json(tmp_path / "manifest.json", {**manifest, **manifest_change})
    )
    packet = _read_json(out / "decision.json")
    assert code == expected_code
    assert packet["reason"] == reason
    assert packet["bounds"] is packet["owner_decision"] is None


@pytest.mark.parametrize(
    ("extra_row", "reason"),
    [
        ("fixture-id-000,1,1,1,1", "duplicate_unit"),
        ("fixture-id-000,0,0,0,0", "duplicate_unit"),
        ("synthetic-private-id,0,2,1,1", "invalid_binary_outcome"),
        ("synthetic-private-id,2,1,1,1", "invalid_assignment"),
        ("synthetic-private-id,0,1,,1", "invalid_baseline_flag"),
    ],
)
@pytest.mark.integration
def test_invalid_owner_record_refuses_whole_case(
    tmp_path: Path, fixture_trial: tuple[Path, dict[str, object]], extra_row: str, reason: str
) -> None:
    data, manifest = fixture_trial
    with data.open("a", encoding="utf-8") as f:
        f.write(extra_row + "\n")
    manifest["expected_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    code, out = _run(tmp_path, _write_json(tmp_path / "manifest.json", manifest))
    packet = _read_json(out / "decision.json")
    assert code == 1
    assert packet["reason"] == reason
    assert packet["bounds"] is None
    assert extra_row.split(",")[0] not in (out / "decision.json").read_text(encoding="utf-8")


def test_hillstrom_uses_assignment_and_only_baseline_flags() -> None:
    rows = _hillstrom_rows(
        io.StringIO(
            "segment,conversion,newbie,recency,visit,spend\n"
            "No E-Mail,1,1,7,0,0\n"
            "Womens E-Mail,1,0,1,1,23\n"
            "Mens E-Mail,0,0,6,0,0\n"
            "No E-Mail,1,1,7,0,0\n"
        )
    )
    assert rows == [(0, 1, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0)]
    with pytest.raises(StudyError) as err:
        _hillstrom_rows(io.StringIO("segment,conversion,newbie,recency\nseen,1,0,1\n"))
    assert err.value.reason == "invalid_assignment"


@pytest.mark.parametrize(
    ("changes", "diagnostic"),
    [
        ({"signed_at": "2000-01-02T00:00:00Z"}, "nonprospective_decision_card"),
        ({"signed_at": "2000-01-01T00:00:00"}, "invalid_decision_card"),
        ({"tau": "0.25"}, "invalid_decision_card"),
        ({"max_false_action_risk": 0.049}, "insufficient_risk_ceiling"),
        ({"cost_false_clear": 0}, "invalid_decision_card"),
        ({"horizon": "28 days"}, "decision_card_mismatch"),
        ({"permission_ref": "other-authorization"}, "decision_card_mismatch"),
        ({"representativeness_evidence": "  "}, "invalid_decision_card"),
        ({"surprise": True}, "invalid_decision_card"),
    ],
)
@pytest.mark.integration
def test_ineligible_cards_cannot_supply_actions(
    tmp_path: Path,
    fixture_trial: tuple[Path, dict[str, object]],
    changes: dict[str, object],
    diagnostic: str,
) -> None:
    _, manifest = fixture_trial
    card = _write_json(tmp_path / "card.json", {**_card_fields(), **changes})
    code, out = _run(
        tmp_path, _write_json(tmp_path / "manifest.json", manifest), "--card", str(card)
    )
    packet = _read_json(out / "decision.json")
    assert code == 2
    assert packet["reason"] == "missing_decision_card"
    assert packet["owner_decision"] is None
    assert diagnostic in {issue["reason"] for issue in packet["diagnostics"]}


@pytest.mark.integration
def test_eligible_card_selects_simple_costed_action_without_confirmation(
    tmp_path: Path, fixture_trial: tuple[Path, dict[str, object]]
) -> None:
    _, manifest = fixture_trial
    card = _write_json(tmp_path / "card.json", {**_card_fields(), "tau": 0.21})
    code, out = _run(
        tmp_path, _write_json(tmp_path / "manifest.json", manifest), "--card", str(card)
    )
    packet = _read_json(out / "decision.json")
    assert code == 0
    assert packet["card_sha256"] == hashlib.sha256(card.read_bytes()).hexdigest()
    owner = packet["owner_decision"]
    assert owner["confirmation"] is None
    assert owner["smallest_useful_simple_pipeline"] == "P0"
    assert not owner["h_adds_decision"]
    assert owner["pipelines"]["P0"]["g_min"] == pytest.approx(0.15, abs=1e-12)
    assert owner["pipelines"]["P0"]["action"]["description"] == "Send offer"
    with (out / "thresholds.csv").open(encoding="utf-8") as f:
        assert sum(row["tau"] == "0.21" for row in csv.DictReader(f)) == 9
    for path in out.iterdir():
        wire = path.read_text(encoding="utf-8")
        assert str(tmp_path) not in wire
        assert "Synthetic sponsor" not in wire
        assert "fixture-id-" not in wire
