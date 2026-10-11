"""Runtime receipts retain actual test identity and inherited category marks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def _child_environment() -> dict[str, str]:
    """The caller's environment without an outer affected run's selection controls."""
    return {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("INCREMENT_AFFECTED_")
    }


def test_worker_selection_route_requires_complete_consistent_evidence():
    from tests._evidence import _worker_selection_route

    assert _worker_selection_route(["import_graph", "import_graph"], 2) == "import_graph"
    assert _worker_selection_route(["testmon"], 2) is None
    assert _worker_selection_route(["import_graph", "testmon"], 2) is None
    assert _worker_selection_route(["import_graph", None], 2) is None


@pytest.mark.integration
def test_runtime_receipt_retains_inherited_markers_and_parameter_identity(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = slow: expensive\n integration: boundary\n module_mark: module mark\n class_mark: class mark\n param_mark: parameter mark\n"
    )
    (tmp_path / "test_consumer.py").write_text(
        "import pytest\npytestmark = pytest.mark.integration\n"
        "@pytest.mark.module_mark\nclass TestConsumer:\n"
        "    pytestmark = pytest.mark.class_mark\n"
        "    @pytest.mark.slow\n"
        "    @pytest.mark.parametrize('value', [pytest.param(1, marks=pytest.mark.param_mark)], ids=['slow@literal'])\n"
        "    def test_consumer(self, value):\n        assert value == 1\n"
    )
    environment = _child_environment()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = str(project)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        environment.pop(name, None)
    environment["POLARS_MAX_THREADS"] = "7"
    environment["OMP_NUM_THREADS"] = "3"
    evidence = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests._evidence",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--confcutdir",
            str(tmp_path),
            "--evidence-root",
            str(evidence),
            "-q",
            str(tmp_path / "test_consumer.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run = next(evidence.glob("run-*"))
    journal = run / "reports.jsonl"
    reports = [json.loads(line) for line in journal.read_text().splitlines()]
    call = next(
        event for event in reports if event.get("event") == "report" and event.get("when") == "call"
    )
    assert (
        call.get("canonical_nodeid")
        == "test_consumer.py::TestConsumer::test_consumer[slow@literal]"
    )
    assert call.get("markers") == [
        "class_mark",
        "integration",
        "module_mark",
        "param_mark",
        "parametrize",
        "slow",
    ]
    test_reports = [event for event in reports if event.get("event") == "report"]
    assert {event["when"] for event in test_reports} == {"setup", "call", "teardown"}
    assert all(
        event["canonical_nodeid"] == "test_consumer.py::TestConsumer::test_consumer[slow@literal]"
        and event["markers"]
        == [
            "class_mark",
            "integration",
            "module_mark",
            "param_mark",
            "parametrize",
            "slow",
        ]
        for event in test_reports
    )
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["status"] == "finished"
    assert manifest["exit_code"] == 0
    assert manifest["executed_tests"] == 1
    assert manifest["resolved_worker_count"] == 0
    assert manifest["dist"] == "no"
    assert manifest["thread_limits"] == {"OMP_NUM_THREADS": "3", "POLARS_MAX_THREADS": "7"}


@pytest.mark.integration
def test_loadgroup_receipt_preserves_properties_for_failures_and_skips(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\nmarkers = slow: expensive\n integration: boundary\n"
    )
    (tmp_path / "test_consumer.py").write_text(
        "import pytest\npytestmark = pytest.mark.integration\n"
        "@pytest.mark.slow\n@pytest.mark.xdist_group('runtime-consumer')\n"
        "@pytest.mark.parametrize('value', [1, 2], ids=['literal@id', 'skip'])\n"
        "def test_consumer(value):\n"
        "    if value == 1: assert False, 'retained failure'\n"
        "    pytest.skip('retained skip')\n"
    )
    environment = _child_environment()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = str(project)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        environment.pop(name, None)
    environment["POLARS_MAX_THREADS"] = "5"
    environment["MKL_NUM_THREADS"] = "2"
    evidence = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "tests._evidence",
            "-p",
            "xdist.plugin",
            "-n",
            "2",
            "--dist=loadgroup",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--confcutdir",
            str(tmp_path),
            "--evidence-root",
            str(evidence),
            "-q",
            str(tmp_path / "test_consumer.py"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    run = next(evidence.glob("run-*"))
    reports = [
        json.loads(line) for line in (run / "reports.jsonl").read_text().splitlines() if line
    ]
    call_reports = [
        event for event in reports if event.get("event") == "report" and event.get("when") == "call"
    ]
    assert {event["canonical_nodeid"] for event in call_reports} == {
        "test_consumer.py::test_consumer[literal@id]",
        "test_consumer.py::test_consumer[skip]",
    }
    assert all(
        event["markers"] == ["integration", "parametrize", "slow", "xdist_group"]
        for event in call_reports
    )
    assert all("duration" in event and "nodeid" in event for event in call_reports)
    assert {
        (event["canonical_nodeid"], event["when"])
        for event in reports
        if event.get("event") == "report"
    } == {
        (nodeid, when)
        for nodeid in (
            "test_consumer.py::test_consumer[literal@id]",
            "test_consumer.py::test_consumer[skip]",
        )
        for when in ("setup", "call", "teardown")
    }
    assert all(
        event["markers"] == ["integration", "parametrize", "slow", "xdist_group"]
        for event in reports
        if event.get("event") == "report"
    )
    failed = next(event for event in call_reports if event["outcome"] == "failed")
    skipped = next(event for event in call_reports if event["outcome"] == "skipped")
    assert "retained failure" in failed["longrepr"]
    assert skipped["outcome"] == "skipped"
    manifest = json.loads((run / "run.json").read_text())
    assert manifest["status"] == "finished"
    assert manifest["exit_code"] == 1
    assert manifest["outcome"] == "failed"
    assert manifest["executed_tests"] == 2
    assert manifest["resolved_worker_count"] == 2
    assert manifest["dist"] == "loadgroup"
    assert manifest["thread_limits"] == {"MKL_NUM_THREADS": "2", "POLARS_MAX_THREADS": "5"}
