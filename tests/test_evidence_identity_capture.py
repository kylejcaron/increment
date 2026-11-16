from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts import capture_evidence_identity as capture_script
from scripts.capture_evidence_identity import iter_capture_keys, population_digest
from tests.capability_contracts import CONSTRUCTORS
from tests.parity_harness.cases import PARITY_CASES
from tests.parity_harness.refusal_cases import REFUSAL_SCENARIOS


def test_population_identity_keeps_same_arm_separate_by_population() -> None:
    from scripts.capture_evidence_identity import population_identity_from_ids

    identities = population_identity_from_ids(
        {"assigned": {"control": ["u1"]}, "triggered": {"control": ["u2"]}}
    )
    assert set(identities) == {"assigned", "triggered"}
    assert identities["assigned"]["control"]["unit_count"] == 1
    assert identities["triggered"]["control"]["unit_count"] == 1
    assert (
        identities["assigned"]["control"]["digest"] != identities["triggered"]["control"]["digest"]
    )


def test_capture_covers_the_scenario_inventory() -> None:
    expected = {(case.id, ingress) for case in PARITY_CASES for ingress in CONSTRUCTORS} | {
        (code, "refusal") for code in REFUSAL_SCENARIOS
    }
    assert set(iter_capture_keys()) == expected
    assert len(expected) == len(PARITY_CASES) * len(CONSTRUCTORS) + len(REFUSAL_SCENARIOS)


def test_population_identity_digest_is_order_independent() -> None:
    assert population_digest(["z", "a"], population="assigned", arm="control") == population_digest(
        ["a", "z"], population="assigned", arm="control"
    )


def test_population_digest_is_length_safe_for_delimiter_ids() -> None:
    left = population_digest(["a", "b\nc"], population="assigned", arm="control")
    right = population_digest(["a\nb", "c"], population="assigned", arm="control")
    assert left != right
    assert left[1] == right[1] == 2


@pytest.mark.integration
def test_cutover_capture_does_not_overwrite_root_fixture(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "identity"
    root.mkdir()
    originals = {name: f"preserve:{name}".encode() for name in ("readouts.json", "checkpoint.json")}
    for name, data in originals.items():
        (root / name).write_bytes(data)
    monkeypatch.setattr(capture_script, "iter_capture_keys", lambda: ())

    capture_script.capture(root, cutover="pre-kernel")

    assert {name: (root / name).read_bytes() for name in originals} == originals
    operations = root / "pre-kernel" / "operations.json.gz"
    first_bytes = operations.read_bytes()
    assert first_bytes[:3] == b"\x1f\x8b\x08"
    assert first_bytes[3] & 0x08 == 0
    assert int.from_bytes(first_bytes[4:8], byteorder="little") == 0
    capture_script.capture(root, cutover="pre-kernel")
    assert operations.read_bytes() == first_bytes
    record = json.loads(gzip.decompress(operations.read_bytes()))
    assert set(record) == {"schema_version", "cutover", "scenarios", "payloads"}
    assert record == {
        "schema_version": 1,
        "cutover": "pre-kernel",
        "scenarios": {},
        "payloads": {},
    }


@pytest.mark.integration
def test_cutover_capture_normalizes_operation_float_noise(tmp_path: Path, monkeypatch) -> None:
    values = iter((0.020833333333333346, 0.020833333333333367))
    monkeypatch.setattr(
        capture_script,
        "_capture_operations",
        lambda cutover, **kwargs: {"schema_version": 1, "cutover": cutover, "value": next(values)},
    )

    capture_script.capture(tmp_path, cutover="pre-kernel")
    first = (tmp_path / "pre-kernel" / "operations.json.gz").read_bytes()
    capture_script.capture(tmp_path, cutover="pre-kernel")

    assert (tmp_path / "pre-kernel" / "operations.json.gz").read_bytes() == first


@pytest.mark.integration
def test_aggregate_moment_payload_is_not_rounded_on_disk(tmp_path: Path, monkeypatch) -> None:
    value = 0.123456789012345
    monkeypatch.setattr(
        capture_script,
        "_capture_operations",
        lambda cutover, **kwargs: {
            "schema_version": 1,
            "cutover": cutover,
            "scenarios": {},
            "payloads": {"exact": {"cy1": value}},
            "_exact_payload_hashes": ["exact"],
        },
    )

    capture_script.capture(tmp_path, cutover="pre-kernel")
    record = json.loads(
        gzip.decompress((tmp_path / "pre-kernel" / "operations.json.gz").read_bytes())
    )

    assert record["payloads"]["exact"]["cy1"] == value


def test_nonaggregate_capture_records_population_digest() -> None:
    from scripts.capture_evidence_identity import load_operations_capture

    record = load_operations_capture(
        Path(__file__).parent
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    digest, count = population_digest(
        [f"u{i}" for i in range(20)], population="assigned", arm="control"
    )
    identity = record["scenarios"]["bare_metric_name_summary_panel"]["from_unit_summary"][
        "population_identity"
    ]
    assert identity["assigned"]["control"] == {"digest": digest, "unit_count": count}
    triggered = record["scenarios"]["triggered-sitewide-population-results"]["from_definitions"][
        "population_identity"
    ]
    assert (
        triggered["triggered"]["control"]["unit_count"]
        < triggered["assigned"]["control"]["unit_count"]
    )
    assert triggered["assigned"]["control"]["digest"] != triggered["triggered"]["control"]["digest"]


def test_sequential_capture_initializes_finalized_checkpoint_before_reading() -> None:
    from scripts.capture_evidence_identity import load_operations_capture

    record = load_operations_capture(
        Path(__file__).parent
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    outcomes = record["scenarios"]["sequential_asymptotic_mean_revenue"]
    for ingress in ("from_definitions", "from_unit_day_artifact", "from_unit_panel"):
        assert outcomes[ingress]["kind"] == "cells"
        assert outcomes[ingress]["cells"]


def test_every_captured_cell_restores_from_its_serialized_payload() -> None:
    from increment.result_envelope import ResultCell
    from scripts.capture_evidence_identity import load_operations_capture

    record = load_operations_capture(
        Path(__file__).parent
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    for scenario in record["scenarios"].values():
        for outcome in scenario.values():
            if outcome["kind"] == "cells":
                for captured in outcome["cells"].values():
                    ResultCell.from_payload(captured["result_cell"])


def test_sequential_and_quantile_cells_roundtrip_supported_payload_operations() -> None:
    from increment.result_envelope import ResultCell
    from scripts.capture_evidence_identity import load_operations_capture

    record = load_operations_capture(
        Path(__file__).parent
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    selected = (
        ("sequential_asymptotic_mean_revenue", "from_definitions"),
        ("audit-quantile-family", "from_unit_summary"),
    )
    for scenario_id, ingress in selected:
        outcome = record["scenarios"][scenario_id][ingress]
        assert outcome["kind"] == "cells"
        for captured in outcome["cells"].values():
            cell = ResultCell.from_payload(captured["result_cell"])
            canonical = json.loads(json.dumps(cell.to_payload()))
            canonical_reopened = json.loads(
                json.dumps(ResultCell.from_payload(canonical).to_payload())
            )
            assert canonical_reopened == canonical


def test_capture_float_normalization_preserves_tiny_nonzero_values() -> None:
    assert capture_script._stable_capture_value(5e-16) == 5e-16


def test_aggregate_only_capture_preserves_moments_evidence_and_digest() -> None:
    from increment.sources import source_sha256
    from scripts.capture_evidence_identity import load_operations_capture

    record = load_operations_capture(
        Path(__file__).parent
        / "fixtures"
        / "evidence_identity"
        / "pre-kernel"
        / "operations.json.gz"
    )
    outcome = record["scenarios"]["categorical-cuped-moments-format-13"]["from_moments"]
    envelope = outcome["aggregate_evidence"]
    rows = envelope["moments"]
    assert envelope["status"] == "available"
    assert envelope["moments_format"] == 13
    assert rows and all(row["moments_format"] == 13 for row in rows)
    assert all(row["source_identity"] for row in rows)
    evidence = json.loads(
        next(row["cuped_evidence"] for row in rows if row.get("record_kind") == "cuped_evidence")
    )
    identity = json.loads(rows[0]["source_identity"])
    assert evidence["source_sha256"] == source_sha256(identity)


def test_aggregate_capture_rejects_invalid_upstream_source_digest() -> None:
    with pytest.raises(ValueError, match="source digest"):
        capture_script._aggregate_evidence(
            {},
            [
                {
                    "moments_format": 13,
                    "source_identity": '{"population":"assigned"}',
                    "record_kind": "cuped_evidence",
                    "cuped_evidence": json.dumps({"source_sha256": "invalid"}),
                }
            ],
        )


@pytest.mark.integration
def test_targeted_operation_cli_keeps_selected_cases_and_constructor_dispositions(
    tmp_path: Path,
) -> None:
    selected = (
        "bare_metric_name_summary_panel",
        "edge-empty-arm",
        "compatibility.support_unsupported",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.capture_evidence_identity",
            "--out",
            str(tmp_path),
            "--cutover",
            "relative-prior-v1",
            *(argument for scenario in selected for argument in ("--scenario", scenario)),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    record = capture_script.load_operations_capture(
        tmp_path / "relative-prior-v1" / "operations.json.gz"
    )
    assert set(record["scenarios"]) == set(selected)
    frames = record["scenarios"]["bare_metric_name_summary_panel"]
    assert set(frames) == set(CONSTRUCTORS)
    assert {ingress: outcome["kind"] for ingress, outcome in frames.items()} == {
        "from_definitions": "absent",
        "from_unit_day_artifact": "absent",
        "from_unit_summary": "cells",
        "from_unit_panel": "cells",
        "from_switchback_panel": "absent",
        "from_moments": "absent",
    }
    assert all(frames[ingress]["cells"] for ingress in ("from_unit_summary", "from_unit_panel"))
    edge = record["scenarios"]["edge-empty-arm"]
    assert set(edge) == set(CONSTRUCTORS)
    assert edge["from_unit_summary"]["kind"] == "refusal"
    assert all(
        outcome["kind"] == "absent"
        for ingress, outcome in edge.items()
        if ingress != "from_unit_summary"
    )
    assert record["scenarios"]["compatibility.support_unsupported"] == {
        "refusal": {
            "kind": "refusal",
            "code": "compatibility.support_unsupported",
            "stage": "run",
            "context": {"support_type": "str"},
        }
    }


@pytest.mark.integration
def test_targeted_operation_capture_validates_all_ids_before_sources_or_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_source(*args, **kwargs):
        pytest.fail("invalid selection reached a source")

    monkeypatch.setattr(capture_script.builders, "from_frame", unexpected_source)
    root = tmp_path / "not-created"
    with pytest.raises(ValueError):
        capture_script.capture(
            root,
            cutover="relative-prior-v1",
            scenarios=("bare_metric_name_summary_panel", "unknown-scenario", ""),
        )
    assert not root.exists()


@pytest.mark.integration
def test_targeted_operation_capture_refuses_existing_output_before_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def unexpected_source(*args, **kwargs):
        pytest.fail("existing capture reached a source")

    monkeypatch.setattr(capture_script.builders, "from_frame", unexpected_source)
    target = tmp_path / "relative-prior-v1" / "operations.json.gz"
    target.parent.mkdir()
    original = b"historical operation inventory"
    target.write_bytes(original)

    with pytest.raises(FileExistsError):
        capture_script.capture(
            tmp_path,
            cutover="relative-prior-v1",
            scenarios=("bare_metric_name_summary_panel",),
        )
    assert target.read_bytes() == original


@pytest.mark.integration
def test_operation_scenario_selection_requires_cutover(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "not-created"
    with pytest.raises(ValueError):
        capture_script.capture(root, scenarios=("bare_metric_name_summary_panel",))
    monkeypatch.setattr(
        sys,
        "argv",
        ["capture_evidence_identity", "--out", str(root), "--scenario", "edge-empty-arm"],
    )
    with pytest.raises(SystemExit) as error:
        capture_script.main()
    assert error.value.code == 2
    assert not root.exists()


def _write_operation_inventory(
    root: Path,
    cutover: str,
    scenarios: dict[str, Any],
    payloads: dict[str, Any] | None = None,
) -> Path:
    target = root / cutover / "operations.json.gz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        gzip.compress(
            json.dumps(
                {
                    "schema_version": 1,
                    "cutover": cutover,
                    "scenarios": scenarios,
                    "payloads": payloads or {},
                }
            ).encode()
        )
    )
    return target


@pytest.mark.integration
def test_frozen_operations_merge_expands_payloads_and_keeps_disjoint_coordinates(
    tmp_path: Path,
) -> None:
    digest = hashlib.sha256(b'{"value":0.5}').hexdigest()
    _write_operation_inventory(
        tmp_path,
        "pre-kernel",
        {"shared": {"from_unit_summary": {"payload_hash": digest, "operations": {}}}},
        {digest: {"value": 0.5}},
    )
    refusal = {"kind": "refusal", "code": "unavailable", "context": {"metric": "revenue"}}
    _write_operation_inventory(
        tmp_path, "pre-kernel-asof", {"shared": {"from_unit_panel": refusal}}
    )
    absence = {"kind": "absent", "reason": "constructor is not declared"}
    _write_operation_inventory(tmp_path, "relative-prior-v1", {"new": {"from_moments": absence}})

    record = capture_script.load_frozen_operations_capture(tmp_path)

    assert record["scenarios"] == {
        "shared": {
            "from_unit_summary": {"result_cell": {"value": 0.5}, "operations": {}},
            "from_unit_panel": refusal,
        },
        "new": {"from_moments": absence},
    }
    assert record["cutovers"] == {
        cutover: {"schema_version": 1, "cutover": cutover}
        for cutover in ("pre-kernel", "pre-kernel-asof", "relative-prior-v1")
    }


@pytest.mark.integration
def test_frozen_operations_reject_duplicate_scenario_ingress(tmp_path: Path) -> None:
    for cutover, reason in (
        ("pre-kernel", "first"),
        ("pre-kernel-asof", "second"),
        ("relative-prior-v1", "collision"),
    ):
        scenario = "shared" if cutover != "pre-kernel-asof" else "distinct"
        _write_operation_inventory(
            tmp_path,
            cutover,
            {scenario: {"from_unit_summary": {"kind": "absent", "reason": reason}}},
        )

    with pytest.raises(ValueError, match="shared/from_unit_summary"):
        capture_script.load_frozen_operations_capture(tmp_path)


@pytest.mark.integration
def test_frozen_operations_require_every_cutover(tmp_path: Path) -> None:
    for cutover in ("pre-kernel", "pre-kernel-asof"):
        _write_operation_inventory(tmp_path, cutover, {})

    with pytest.raises(FileNotFoundError):
        capture_script.load_frozen_operations_capture(tmp_path)
