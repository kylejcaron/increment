"""Runtime warnings aggregate phases instead of considering call time alone."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def test_report_flags_expensive_setup_even_when_call_is_fast(tmp_path: Path):
    run = tmp_path / "evidence" / "run-one"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "finished",
                "exit_code": 0,
                "python": "3.12.0",
                "platform": "test-host",
                "command": [],
                "resolved_worker_count": 0,
                "dist": "no",
                "thread_limits": {},
                "executed_tests": 1,
                "selected_nodeids": [],
            }
        )
    )
    (run / "reports.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "event": "report",
                    "nodeid": "tests/test_consumer.py::test_consumer",
                    "canonical_nodeid": "tests/test_consumer.py::test_consumer",
                    "markers": [],
                    "when": phase,
                    "duration": duration,
                    "outcome": "passed",
                }
            )
            for phase, duration in [("setup", 4.7), ("call", 0.6), ("teardown", 0.1)]
        )
    )
    output = tmp_path / "report.json"
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.report_test_runtime",
            str(run.parent),
            "--json",
            str(output),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    case = report["cases"][0]
    assert case["total_seconds"] == pytest.approx(5.4)
    assert case["phases"]["setup"] == pytest.approx(4.7)
    assert case["suggestion"] == "slow"


def _write_run(
    root: Path,
    *,
    run_id: str = "run-one",
    metadata: dict | None = None,
    events: list[dict] | None = None,
) -> Path:
    run = root / run_id
    run.mkdir(parents=True)
    report_events = events or []
    executed_nodeids = {
        event["nodeid"]
        for event in report_events
        if event.get("event") == "report"
        and isinstance(event.get("nodeid"), str)
        and (
            event.get("when") == "call"
            or event.get("when") == "setup"
            and event.get("outcome") in ("failed", "skipped")
        )
    }
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "finished",
                "exit_code": 0,
                "python": "3.12.0",
                "platform": "test-host",
                "command": [],
                "resolved_worker_count": 0,
                "dist": "no",
                "thread_limits": {},
                "executed_tests": len(executed_nodeids),
                "selected_nodeids": [],
                **(metadata or {}),
            }
        )
    )
    (run / "reports.jsonl").write_text("".join(json.dumps(event) + "\n" for event in report_events))
    return run


def _event(when: str, duration: object, **changes: object) -> dict:
    return {
        "event": "report",
        "nodeid": "tests/test_consumer.py::test_consumer[param@literal]",
        "canonical_nodeid": "tests/test_consumer.py::test_consumer[param@literal]",
        "markers": [],
        "when": when,
        "duration": duration,
        "outcome": "passed",
        **changes,
    }


def _run_report(
    root: Path,
    *args: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    project = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    return subprocess.run(
        [sys.executable, "-m", "scripts.report_test_runtime", str(root), *args],
        cwd=cwd or project,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_runtime_threshold_is_strict_and_priority_is_advisory(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_run(
        evidence,
        events=[
            _event("setup", 2.0),
            _event("call", 3.0),
            _event("teardown", 5.0),
        ],
    )
    output = tmp_path / "report.json"

    result = _run_report(
        evidence,
        "--threshold",
        "5",
        "--priority-threshold",
        "10",
        "--json",
        str(output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert "::warning::Runtime 10.000s exceeds advisory threshold" in result.stdout
    assert case["canonical_nodeid"].endswith("[param@literal]")
    assert case["new"] is None
    assert case["priority"] is False


def test_exact_slow_threshold_is_not_a_suggestion(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_run(
        evidence,
        events=[
            _event("setup", 2),
            _event("call", 2),
            _event("teardown", 1),
        ],
    )
    output = tmp_path / "report.json"

    result = _run_report(
        evidence,
        "--threshold",
        "5",
        "--priority-threshold",
        "10",
        "--json",
        str(output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert case["total_seconds"] == 5
    assert case["suggestion"] is None
    assert case["priority"] is False


def test_duplicate_phase_events_do_not_double_count(tmp_path: Path):
    evidence = tmp_path / "evidence"
    events = [_event("call", 2), _event("call", 2), _event("setup", 1), _event("teardown", 0)]
    _write_run(evidence, events=events)
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert case["phases"] == {"setup": 1, "call": 2, "teardown": 0}
    assert case["total_seconds"] == 3


@pytest.mark.parametrize(
    "duration", [-1, float("nan"), float("inf"), pytest.param(10**400, id="overflow"), "1.2", None]
)
def test_invalid_duration_is_reported_without_a_slow_suggestion(tmp_path: Path, duration: object):
    evidence = tmp_path / "evidence"
    _write_run(evidence, events=[_event("call", duration)])
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert case["complete"] is False
    assert case["suggestion"] is None
    assert case["warnings"]


def test_incomplete_and_legacy_receipts_are_explicit(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_run(
        evidence,
        run_id="run-legacy",
        metadata={"status": "interrupted", "resolved_worker_count": None},
        events=[_event("call", 20, canonical_nodeid=None, markers=None)],
    )
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert report["metadata"]["resolved_worker_count"] is None
    case = report["cases"][0]
    assert case["complete"] is False
    assert case["canonical_nodeid"] == _event("call", 20)["nodeid"]
    assert case["suggestion"] is None
    assert case["warnings"]


def test_existing_slow_marker_suppresses_source_suggestion(tmp_path: Path):
    evidence = tmp_path / "evidence"
    _write_run(
        evidence,
        events=[_event("call", 20, markers=["slow", "unit"])],
    )
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert case["markers"] == ["slow", "unit"]
    assert case["suggestion"] is None


def test_baseline_mismatch_is_incomparable(tmp_path: Path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    _write_run(current, events=[_event("call", 20)])
    _write_run(
        baseline,
        metadata={"platform": "other-host"},
        events=[_event("call", 2)],
    )
    output = tmp_path / "report.json"

    result = _run_report(
        current,
        "--baseline",
        str(baseline),
        "--json",
        str(output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["baseline"]["comparable"] is False
    assert "platform" in report["baseline"]["reason"]
    assert report["cases"][0]["regression"] is None


def test_new_case_claim_requires_exact_baseline_inventory(tmp_path: Path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    _write_run(baseline, metadata={"selected_nodeids": []}, events=[])
    _write_run(current, events=[_event("call", 20)])
    output = tmp_path / "report.json"

    result = _run_report(
        current,
        "--baseline",
        str(baseline),
        "--json",
        str(output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert case["new"] is None
    assert case["newly_observed"] is False
    assert case["warnings"]


def test_baseline_match_preserves_existing_slow_marker(tmp_path: Path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    events = [
        _event("setup", 1, markers=["slow"]),
        _event("call", 20, markers=["slow"]),
        _event("teardown", 1, markers=["slow"]),
    ]
    _write_run(current, events=events)
    _write_run(baseline, events=events)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    result = _run_report(
        current,
        "--baseline",
        str(baseline),
        "--json",
        str(json_output),
        "--markdown",
        str(markdown_output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(json_output.read_text())
    case = report["cases"][0]
    assert case["new"] is False
    assert case["regression"] is False
    assert case["suggestion"] is None
    assert case["priority"] is True


def test_base_revision_reports_ast_family_and_invalid_ref_without_traceback(tmp_path: Path):
    repo = tmp_path / "repo"
    tests_dir = repo / "tests"
    tests_dir.mkdir(parents=True)
    source = tests_dir / "test_family.py"
    source.write_text("def test_family():\n    pass\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "runtime-report@example.invalid"],
        ["git", "config", "user.name", "Runtime Report Test"],
        ["git", "add", "tests/test_family.py"],
        ["git", "commit", "-q", "-m", "baseline test family"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)

    evidence = tmp_path / "evidence"
    family = "tests/test_family.py::test_family"
    _write_run(
        evidence,
        events=[
            _event("setup", 1, nodeid=family, canonical_nodeid=family),
            _event("call", 1, nodeid=family, canonical_nodeid=family),
            _event("teardown", 0, nodeid=family, canonical_nodeid=family),
        ],
    )
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--base", "HEAD", "--json", str(output), cwd=repo)

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["base"]["revision"]
    assert report["base"]["family_comparison"] == [
        {
            "nodeid": family,
            "family": family,
            "present": True,
            "absence_meaning": None,
        }
    ]

    invalid = _run_report(
        evidence,
        "--base",
        "not-a-real-revision",
        "--json",
        str(output),
        cwd=repo,
    )
    assert invalid.returncode == 0
    assert "Traceback" not in invalid.stderr
    assert json.loads(output.read_text())["base"]["comparable"] is False


def test_start_only_and_malformed_journal_records_remain_visible(tmp_path: Path):
    evidence = tmp_path / "evidence"
    run = _write_run(
        evidence,
        metadata={"status": "interrupted", "exit_code": 130},
        events=[],
    )
    run_id = "tests/test_consumer.py::test_interrupted"
    (run / "reports.jsonl").write_text(
        json.dumps({"event": "start", "nodeid": run_id})
        + "\n"
        + "{malformed journal line\n"
        + json.dumps({"event": "future-event", "nodeid": run_id})
        + "\n"
    )
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert any(case["nodeid"] == run_id for case in report["cases"])
    assert any("malformed" in warning for warning in report["warnings"])
    assert any("unrecognized" in warning for warning in report["warnings"])


def test_baseline_subset_only_marks_case_newly_observed(tmp_path: Path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current_id = "tests/test_consumer.py::test_consumer[current]"
    baseline_id = "tests/test_consumer.py::test_consumer[baseline]"
    current_events = [
        _event(phase, 1, nodeid=current_id, canonical_nodeid=current_id)
        for phase in ("setup", "call", "teardown")
    ]
    baseline_events = [
        _event(phase, 1, nodeid=baseline_id, canonical_nodeid=baseline_id)
        for phase in ("setup", "call", "teardown")
    ]
    _write_run(current, events=current_events)
    _write_run(
        baseline,
        metadata={"selected_nodeids": [baseline_id]},
        events=baseline_events,
    )
    output = tmp_path / "report.json"

    result = _run_report(current, "--baseline", str(baseline), "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    case = json.loads(output.read_text())["cases"][0]
    assert case["new"] is None
    assert case["newly_observed"] is True
    assert case["regression"] is None


def test_markdown_and_annotations_are_bounded_without_filtering_json(tmp_path: Path):
    evidence = tmp_path / "evidence"
    events = []
    expected_nodeids = set()
    for index in range(130):
        nodeid = f"tests/test_many.py::test_case_{index}"
        expected_nodeids.add(nodeid)
        events.extend(
            _event(phase, 8 if phase == "call" else 0, nodeid=nodeid, canonical_nodeid=nodeid)
            for phase in ("setup", "call", "teardown")
        )
    _write_run(evidence, events=events)
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"

    result = _run_report(
        evidence,
        "--json",
        str(json_output),
        "--markdown",
        str(markdown_output),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report_cases = json.loads(json_output.read_text())["cases"]
    assert {case["canonical_nodeid"] for case in report_cases} == expected_nodeids
    assert all(case["total_seconds"] > 5 for case in report_cases)
    markdown = markdown_output.read_text()
    rendered_rows = [
        line for line in markdown.splitlines() if line.startswith("| `tests/test_many.py::")
    ]
    rendered_nodeids = {line.split("`")[1] for line in rendered_rows}
    assert len(rendered_rows) == 100
    assert rendered_nodeids <= expected_nodeids
    assert len(markdown.encode("utf-8")) < 30_000
    assert (
        len([line for line in result.stdout.splitlines() if line.startswith("::warning::")]) <= 26
    )


@pytest.mark.parametrize("event_kind", [[], {}])
def test_non_string_event_kinds_are_incomplete_evidence_not_cli_errors(
    tmp_path: Path, event_kind: object
):
    evidence = tmp_path / "evidence"
    events = [_event(phase, 1) for phase in ("setup", "call", "teardown")]
    events.append({"event": event_kind, "nodeid": "tests/test_consumer.py::test_consumer"})
    _write_run(evidence, events=events)
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert report["warnings"]
    assert len(report["cases"]) == 1


@pytest.mark.parametrize(
    "inventory",
    [
        {"executed_tests": 2},
        {
            "selected_nodeids": [
                "tests/test_consumer.py::test_consumer",
                "tests/test_consumer.py::test_missing",
            ]
        },
    ],
)
def test_whole_case_evidence_loss_cannot_be_reported_complete(tmp_path: Path, inventory: dict):
    evidence = tmp_path / "evidence"
    _write_run(
        evidence,
        metadata=inventory,
        events=[_event(phase, 1) for phase in ("setup", "call", "teardown")],
    )
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert report["warnings"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"executed_tests": -1},
        {"executed_tests": True},
        {"selected_nodeids": "not-an-id-list"},
        {"selected_nodeids": ["tests/test_consumer.py::test_consumer"] * 2},
    ],
)
def test_malformed_case_inventory_is_incomplete(tmp_path: Path, metadata: dict):
    evidence = tmp_path / "evidence"
    _write_run(
        evidence,
        metadata=metadata,
        events=[_event(phase, 1) for phase in ("setup", "call", "teardown")],
    )
    output = tmp_path / "report.json"

    result = _run_report(evidence, "--json", str(output))

    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert report["warnings"]
