from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_fast_test_evidence import verify_fast_test_evidence


def _run(root: Path, events: list[dict[str, object]], *, exit_code: int) -> None:
    run = root / "run-one"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "status": "finished",
                "exit_code": exit_code,
                "outcome": "passed" if exit_code == 0 else "failed",
                "collected_nodeids": ["test_sample.py::test_first", "test_sample.py::test_second"],
                "collected_tests": 2,
                "executed_tests": 2,
            }
        )
    )
    (run / "reports.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")


def _passed_events() -> list[dict[str, object]]:
    return [
        {"event": "report", "nodeid": nodeid, "when": "call", "outcome": "passed"}
        for nodeid in ("test_sample.py::test_first", "test_sample.py::test_second")
    ]


def test_fast_evidence_accepts_all_test_reports_after_optional_plugin_exit(tmp_path: Path):
    _run(tmp_path, _passed_events(), exit_code=3)

    passed, message = verify_fast_test_evidence(tmp_path)

    assert passed
    assert "terminal, non-failing" in message


def test_fast_evidence_rejects_test_failure(tmp_path: Path):
    events = _passed_events()
    events[-1]["outcome"] = "failed"
    _run(tmp_path, events, exit_code=3)

    passed, message = verify_fast_test_evidence(tmp_path)

    assert not passed
    assert "pytest reported failed tests" in message


def test_fast_evidence_rejects_incomplete_execution(tmp_path: Path):
    _run(tmp_path, _passed_events()[:1], exit_code=3)

    passed, message = verify_fast_test_evidence(tmp_path)

    assert not passed
    assert "execution is incomplete" in message


def test_fast_evidence_reads_full_runner_collection_inventory(tmp_path: Path):
    import os
    import subprocess
    import sys

    repository = Path(__file__).resolve().parents[1]
    project = tmp_path / "sample"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_sample.py").write_text(
        "def test_first():\n    assert True\n\ndef test_second():\n    assert True\n"
    )
    evidence = tmp_path / "evidence"
    environment = {**os.environ, "PYTHONPATH": str(repository)}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_test_tier",
            "all",
            "-q",
            "--evidence-root",
            str(evidence),
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    passed, message = verify_fast_test_evidence(evidence)
    assert passed, message
