"""Exercise real bounded campaign processes and retained evidence."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[2]


def run_campaign(output, *options, timeout=40):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "calibration.unit_cycle",
            "--output",
            str(output),
            *options,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_smoke_exercises_only_declared_cases_and_preserves_existing_evidence(tmp_path):
    from tests._unit_cycle_design import CELLS
    from tests.simulate.test_unit_cycle_calibration import RELEASE_CELLS

    output = tmp_path / "smoke"
    # Generous budget: this bounds runtime, not the property under test,
    # and contention can push a normally ~5s smoke run past 30s.
    result = run_campaign(output, "--smoke", "--budget-seconds", "120", timeout=180)
    assert result.returncode == 2, result.stderr
    ending = json.loads((output / "campaign_end.json").read_text())
    assert ending["status"] == "inconclusive" and not ending["full_matrix_certified"]
    selection = json.loads((output / "selection.json").read_text())
    checkpoints = []
    for index in selection["design_indices"]:
        records = [
            json.loads(line)
            for line in (output / f"design-{index:04d}.jsonl").read_text().splitlines()
        ]
        checkpoints.extend(record for record in records if record["kind"] == "checkpoint")
    assert {CELLS[record["cell_index"]] for record in checkpoints} == set(RELEASE_CELLS)
    for record in checkpoints:
        assert record["status"] == "inconclusive"
        for report in record["reports"].values():
            assert report["attempted"] == 8 and report["excluded"] == 0
            assert report["point_estimable"] + report["failed"] == report["attempted"]
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    repeated = run_campaign(output, "--smoke", "--max-repetitions", "1")
    assert repeated.returncode == 2
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_deadline_retains_configuration_and_never_certifies_an_incomplete_campaign(tmp_path):
    output = tmp_path / "timeout"
    result = run_campaign(output, "--budget-seconds", "1", "--stop-design", "1")
    assert result.returncode == 124, result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    ending = json.loads((output / "campaign_end.json").read_text())
    assert manifest["configuration"]["stop_design"] == 1
    assert manifest["original_manifest"]["cell_count"] == 4320
    # The deadline fired before any design could complete: the run is reported
    # as a timeout and never certifies the incomplete matrix.
    assert not (output / "controller_end.json").exists()
    assert ending["status"] == "timeout" and not ending["full_matrix_certified"]
