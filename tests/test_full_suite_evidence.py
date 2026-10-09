"""Full-suite artifacts survive failed runs and repeated parallel invocations."""

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("workers", [0, 2])
def test_failed_parallel_run_retains_evidence_without_overwriting_previous_run(tmp_path, workers):
    project = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nfilterwarnings = ["error"]\ntestpaths = ["cases"]\n'
    )
    (sample / "scripts").mkdir()
    (sample / "scripts" / "__init__.py").touch()
    shutil.copyfile(
        project / "scripts" / "run_test_tier.py", sample / "scripts" / "run_test_tier.py"
    )
    shutil.copyfile(
        project / "scripts" / "_test_tier_policy.py",
        sample / "scripts" / "_test_tier_policy.py",
    )
    shutil.copyfile(project / "scripts" / "_test_impact.py", sample / "scripts" / "_test_impact.py")
    shutil.copyfile(
        project / "scripts" / "run_test_tier_plugin.py",
        sample / "scripts" / "run_test_tier_plugin.py",
    )
    (sample / ".gitignore").write_text("__pycache__/\n.pytest_cache/\n")
    subprocess.run(["git", "init", "-q", str(sample)], check=True)
    (sample / "cases").mkdir()
    (sample / "test_outside_selection.py").write_text(
        'raise AssertionError("The project testpaths must remain authoritative.")\n'
    )
    (sample / "conftest.py").write_text('pytest_plugins = ("tests._evidence",)\n')
    (sample / "cases" / "test_sample.py").write_text(
        "import json, os\n"
        "import pytest\n"
        "@pytest.mark.parametrize('case', ['first', 'second'])\n"
        "def test_case(case, tmp_path, record_property):\n"
        "    path = tmp_path / 'counts.json'\n"
        "    path.write_text(json.dumps({'case': case, 'attempts': 7}))\n"
        "    record_property('evidence_path', str(path))\n"
        "    assert os.environ['CASE_PASS'] == 'yes'\n"
    )
    evidence = tmp_path / "evidence"
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(project),
    }
    command = [
        sys.executable,
        "-m",
        "scripts.run_test_tier",
        "all",
        "-p",
        "xdist.plugin",
        "-n",
        str(workers),
        "-q",
        "--evidence-root",
        str(evidence),
    ]
    for passing, expected in (("yes", 0), ("no", 1)):
        outcome = subprocess.run(
            command,
            cwd=sample,
            env={**environment, "CASE_PASS": passing},
            capture_output=True,
            text=True,
            timeout=45,
        )
        assert outcome.returncode == expected, outcome.stdout + outcome.stderr
    runs = sorted(evidence.iterdir())
    assert len(runs) == 2
    downloaded = tmp_path / "downloaded"
    shutil.copytree(evidence, downloaded)
    statuses = []
    for run in runs:
        summary = json.loads((run / "run.json").read_text())
        statuses.append(summary["exit_code"])
        assert summary["status"] == "finished"
        journal = [json.loads(line) for line in (run / "reports.jsonl").read_text().splitlines()]
        failed_nodes = sorted(
            event["nodeid"].rsplit("::", 1)[-1]
            for event in journal
            if event.get("outcome") == "failed"
        )
        assert failed_nodes == (
            ["test_case[first]", "test_case[second]"] if summary["exit_code"] else []
        )
        reports = [json.loads(path.read_text()) for path in (run / "tmp").rglob("counts.json")]
        assert sorted(reports, key=lambda report: report["case"]) == [
            {"case": "first", "attempts": 7},
            {"case": "second", "attempts": 7},
        ]
        suite = ET.parse(run / "junit.xml").getroot().find("testsuite")
        assert suite is not None
        assert int(suite.attrib["tests"]) == 2
        assert int(suite.attrib["failures"]) == (2 if summary["exit_code"] else 0)
        referenced_cases = []
        for property_node in suite.findall("./testcase/properties/property"):
            if property_node.attrib["name"] == "evidence_path":
                reference = Path(property_node.attrib["value"])
                assert not reference.is_absolute()
                referenced_cases.append(json.loads((downloaded / reference).read_text()))
        assert sorted(referenced_cases, key=lambda report: report["case"]) == [
            {"case": "first", "attempts": 7},
            {"case": "second", "attempts": 7},
        ]
    assert sorted(statuses) == [0, 1]


def _diagnostic_files(run: Path) -> tuple[Path, Path, Path]:
    manifest = next(run.glob("tmp/**/runtime-diagnostics.json"))
    events = manifest.with_name("runtime-events.jsonl")
    resources = manifest.with_name("runtime-resources.jsonl")
    return manifest, events, resources


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_runtime_diagnostics_retain_parallel_failure_and_relative_reference(tmp_path):
    project = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\nfilterwarnings = ['error']\ntestpaths = ['cases']\n"
    )
    (sample / "cases").mkdir()
    (sample / "conftest.py").write_text('pytest_plugins = ("tests._evidence",)\n')
    (sample / "cases" / "test_diagnostics.py").write_text(
        "import time\n"
        "import pytest\n"
        "\n"
        "@pytest.mark.parametrize('case', ['ok', 'bad'])\n"
        "def test_case(case, runtime_diagnostics):\n"
        "    with runtime_diagnostics.stage('calibration-' + case):\n"
        "        runtime_diagnostics.progress(completed_batches=2, completed_draws=8,\n"
        "            planned_draws=16, batch_size=4)\n"
        "        time.sleep(1.2)\n"
        "        if case == 'bad':\n"
        "            raise RuntimeError('calibration failed')\n"
    )
    evidence = tmp_path / "evidence"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(sample / "pyproject.toml"),
        "-p",
        "tests._evidence",
        "-p",
        "xdist.plugin",
        "-n",
        "2",
        "-q",
        "--evidence-root",
        str(evidence),
        "--runtime-diagnostics",
    ]
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(project),
    }
    outcome = subprocess.run(
        command, cwd=sample, env=environment, capture_output=True, text=True, timeout=30
    )
    assert outcome.returncode == 1, outcome.stdout + outcome.stderr
    run = next(path for path in evidence.iterdir() if path.is_dir())
    suite = ET.parse(run / "junit.xml").getroot().find("testsuite")
    assert suite is not None
    references = [
        Path(node.attrib["value"])
        for node in suite.findall("./testcase/properties/property")
        if node.attrib["name"] == "evidence_path"
    ]
    assert len(references) == 2
    downloaded = tmp_path / "downloaded"
    shutil.copytree(evidence, downloaded)
    locations = []
    for reference in references:
        manifest = downloaded / reference
        assert not reference.is_absolute()
        locations.append((reference.parent, json.loads(manifest.read_text())))
    manifests = [manifest for _, manifest in locations]
    assert {manifest["nodeid"].rsplit("::", 1)[-1] for manifest in manifests} == {
        "test_case[ok]",
        "test_case[bad]",
    }
    import psutil

    for location, manifest in locations:
        events = _read_jsonl(downloaded / location / "runtime-events.jsonl")
        assert any(
            event["event"] == "progress" and event["completed_draws"] == 8 for event in events
        )
        assert any(event["event"] == "stage_start" for event in events)
        samples = _read_jsonl(downloaded / location / "runtime-resources.jsonl")
        assert samples
        assert all(sample["process"]["rss_bytes"] > 0 for sample in samples)
        assert all(sample["process"]["num_threads"] >= 1 for sample in samples)
        assert manifest["sampler_returncode"] == 0
        assert not psutil.pid_exists(manifest["sampler_pid"])
    failed_location, failed = next(
        item for item in locations if item[1]["nodeid"].endswith("[bad]")
    )
    failed_events = _read_jsonl(downloaded / failed_location / "runtime-events.jsonl")
    assert any(
        event["event"] == "stage_end"
        and event["stage"] == "calibration-bad"
        and event["outcome"] == "error"
        and event["exception_type"] == "RuntimeError"
        for event in failed_events
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_diagnostics_is_opt_in(tmp_path, enabled):
    project = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['cases']\n")
    (sample / "cases").mkdir()
    (sample / "conftest.py").write_text('pytest_plugins = ("tests._evidence",)\n')
    (sample / "cases" / "test_success.py").write_text(
        "def test_success(runtime_diagnostics):\n"
        "    with runtime_diagnostics.stage('success'):\n"
        "        runtime_diagnostics.progress(completed_batches=1, completed_draws=2,\n"
        "            planned_draws=2, batch_size=2)\n"
    )
    evidence = tmp_path / "evidence"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(sample / "pyproject.toml"),
        "-p",
        "tests._evidence",
        "-q",
        "--evidence-root",
        str(evidence),
    ]
    if enabled:
        command.append("--runtime-diagnostics")
    outcome = subprocess.run(
        command,
        cwd=sample,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(project)},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    run = next(path for path in evidence.iterdir() if path.is_dir())
    diagnostics = list((run / "tmp").rglob("runtime-diagnostics.json"))
    assert bool(diagnostics) is enabled


@pytest.mark.parametrize(
    "timeout_method",
    [
        "thread",
        pytest.param(
            "signal",
            marks=pytest.mark.skipif(
                not hasattr(signal, "SIGALRM"), reason="Signal timeouts require SIGALRM"
            ),
        ),
    ],
)
def test_runtime_diagnostics_retain_pytest_timeout_progress_and_samples(tmp_path, timeout_method):
    project = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['cases']\n")
    (sample / "cases").mkdir()
    (sample / "conftest.py").write_text('pytest_plugins = ("tests._evidence",)\n')
    (sample / "cases" / "test_timeout.py").write_text(
        "import time\n"
        "def test_timeout(runtime_diagnostics):\n"
        "    with runtime_diagnostics.stage('sleeping-calibration'):\n"
        "        runtime_diagnostics.progress(completed_batches=3, completed_draws=12,\n"
        "            planned_draws=20, batch_size=4)\n"
        "        time.sleep(5)\n"
    )
    evidence = tmp_path / "evidence"
    outcome = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(sample / "pyproject.toml"),
            "-p",
            "tests._evidence",
            "-p",
            "pytest_timeout",
            "-q",
            "--timeout=1",
            f"--timeout-method={timeout_method}",
            "-o",
            "timeout_func_only=true",
            "--evidence-root",
            str(evidence),
            "--runtime-diagnostics",
        ],
        cwd=sample,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(project)},
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert outcome.returncode == 1, outcome.stdout + outcome.stderr
    run = next(path for path in evidence.iterdir() if path.is_dir())
    manifest, events_path, resources_path = _diagnostic_files(run)
    events = _read_jsonl(events_path)
    assert any(event["event"] == "progress" and event["completed_draws"] == 12 for event in events)
    assert any(
        event["event"] == "stage_start" and event["stage"] == "sleeping-calibration"
        for event in events
    )
    samples = _read_jsonl(resources_path)
    assert samples
    assert any(sample["process"]["rss_bytes"] > 0 for sample in samples)
    details = json.loads(manifest.read_text())
    if timeout_method == "signal":
        assert any(
            event["event"] == "stage_end" and event["outcome"] == "error" for event in events
        )
        assert details["sampler_returncode"] == 0
    import psutil

    try:
        sampler = psutil.Process(details["sampler_pid"])
    except psutil.NoSuchProcess:
        pass
    else:
        sampler.wait(timeout=5)


def test_outer_deadline_retains_failure_and_interrupted_test(tmp_path):
    project = Path(__file__).resolve().parents[1]
    sample = tmp_path / "sample"
    sample.mkdir()
    config = sample / "pytest.ini"
    config.write_text("[pytest]\n")
    (sample / "conftest.py").write_text('pytest_plugins = ("tests._evidence",)\n')
    (sample / "test_sample.py").write_text(
        "import os, time\n"
        "from pathlib import Path\n"
        "def test_failure():\n"
        "    assert False, 'intentional retained failure'\n"
        "def test_waits_after_failure(runtime_diagnostics):\n"
        "    with runtime_diagnostics.stage('interrupted-calibration'):\n"
        "        runtime_diagnostics.progress(completed_batches=1, completed_draws=4,\n"
        "            planned_draws=16, batch_size=4)\n"
        "        Path(os.environ['READY']).touch()\n"
        "        time.sleep(60)\n"
    )

    evidence = tmp_path / "evidence"
    unrelated = evidence / "run-unrelated"
    unrelated.mkdir(parents=True)
    (unrelated / "run.json").write_text(
        json.dumps({"status": "running", "exit_code": None, "owner_token": "unrelated"}) + "\n"
    )
    command = [
        sys.executable,
        "-c",
        "import sys; from scripts.run_test_tier import TIERS, build_pytest_command, run_with_budget; "
        "raise SystemExit(run_with_budget(build_pytest_command(TIERS['all'], sys.argv[1:]), "
        "tier='evidence-interruption', budget_seconds=60))",
        "-c",
        str(config),
        "-q",
        "--evidence-root",
        str(evidence),
        "--runtime-diagnostics",
    ]
    ready = tmp_path / "ready"
    creationflags = vars(subprocess)["CREATE_NEW_PROCESS_GROUP"] if os.name == "nt" else 0
    interrupt_signal = vars(signal)["CTRL_BREAK_EVENT"] if os.name == "nt" else signal.SIGTERM
    expected_status = 128 + (vars(signal)["SIGBREAK"] if os.name == "nt" else signal.SIGTERM)
    process = subprocess.Popen(
        command,
        cwd=sample,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONPATH": str(project),
            "READY": str(ready),
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    try:
        startup_deadline = time.monotonic() + 10
        while not ready.exists() and process.poll() is None and time.monotonic() < startup_deadline:
            time.sleep(0.01)
        assert ready.exists(), "pytest did not reach the readiness handshake"
        time.sleep(1.2)
        process.send_signal(interrupt_signal)
        stdout, stderr = process.communicate(timeout=20)
    finally:
        if process.poll() is None:
            process.send_signal(interrupt_signal)
            try:
                process.communicate(timeout=6)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
    outcome = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    assert outcome.returncode == expected_status, outcome.stdout + outcome.stderr
    assert json.loads((unrelated / "run.json").read_text())["status"] == "running"
    run = next(path for path in evidence.iterdir() if path != unrelated)
    summary = json.loads((run / "run.json").read_text())
    assert summary["status"] == "finished"
    assert summary["exit_code"] == expected_status
    assert summary["termination"] == "interrupted"
    journal = [json.loads(line) for line in (run / "reports.jsonl").read_text().splitlines()]
    failures = [event for event in journal if event.get("outcome") == "failed"]
    assert len(failures) == 1
    assert failures[0]["nodeid"].endswith("test_sample.py::test_failure")
    assert "intentional retained failure" in failures[0]["longrepr"]
    assert any(
        event["event"] == "start"
        and event["nodeid"].endswith("test_sample.py::test_waits_after_failure")
        for event in journal
    )
    manifest, events_path, resources_path = _diagnostic_files(run)
    diagnostics = json.loads(manifest.read_text())
    events = _read_jsonl(events_path)
    assert any(
        event["event"] == "stage_start" and event["stage"] == "interrupted-calibration"
        for event in events
    )
    assert any(event["event"] == "progress" and event["completed_draws"] == 4 for event in events)
    samples = _read_jsonl(resources_path)
    assert samples
    assert all(sample["process"]["rss_bytes"] > 0 for sample in samples)
    import psutil

    sampler = (
        psutil.Process(diagnostics["sampler_pid"])
        if psutil.pid_exists(diagnostics["sampler_pid"])
        else None
    )
    if sampler is not None:
        sampler.wait(timeout=5)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("termination", ["budget_timeout", "process_exit"])
def test_runner_finalizes_owned_unfinished_evidence(tmp_path, wrapped, termination):
    project = Path(__file__).resolve().parents[1]
    evidence = tmp_path / "evidence"
    expected_status = 124 if termination == "budget_timeout" else 1
    finish = "time.sleep(60)" if termination == "budget_timeout" else "os._exit(1)"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib, time; "
            "root=pathlib.Path(os.environ['EVIDENCE']); run=root/'run-manual'; "
            "run.mkdir(parents=True); "
            "(run/'run.json').write_text(json.dumps({'status':'running','exit_code':None,"
            f"'owner_token':os.environ['INCREMENT_EVIDENCE_OWNER']}})); {finish}"
        ),
        "--evidence-root",
        str(evidence),
    ]
    if wrapped:
        command = [
            sys.executable,
            "-c",
            "import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[1:]))",
            *command,
        ]
    outcome = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.run_test_tier import run_with_budget; import sys; "
            "raise SystemExit(run_with_budget(sys.argv[1:], tier='manual-evidence', budget_seconds=5))",
            *command,
        ],
        cwd=project,
        env={**os.environ, "EVIDENCE": str(evidence)},
        capture_output=True,
        text=True,
        # The child must start and write its run file inside the budget, so the budget
        # leaves headroom for interpreter startup under a loaded suite; the outer
        # timeout is a hang guard only.
        timeout=60,
    )
    assert outcome.returncode == expected_status, outcome.stdout + outcome.stderr
    summary = json.loads((evidence / "run-manual" / "run.json").read_text())
    assert summary["status"] == "finished"
    assert summary["exit_code"] == expected_status
    assert summary["termination"] == termination


@pytest.mark.parametrize("budget_seconds", [None, 30])
@pytest.mark.parametrize("wrapped", [False, True])
def test_child_exit_124_is_not_a_supervisor_timeout(tmp_path, budget_seconds, wrapped):
    project = Path(__file__).resolve().parents[1]
    evidence = tmp_path / "evidence"
    command = [
        sys.executable,
        "-c",
        (
            "import json, os, pathlib; "
            "run=pathlib.Path(os.environ['EVIDENCE'])/'run-manual'; "
            "run.mkdir(parents=True); "
            "(run/'run.json').write_text(json.dumps({'status':'running','exit_code':None,"
            "'owner_token':os.environ['INCREMENT_EVIDENCE_OWNER']})); os._exit(124)"
        ),
        "--evidence-root",
        str(evidence),
    ]
    if wrapped:
        command = [
            sys.executable,
            "-c",
            "import subprocess,sys; raise SystemExit(subprocess.call(sys.argv[1:]))",
            *command,
        ]
    outcome = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts.run_test_tier import run_with_budget; import sys; "
            f"raise SystemExit(run_with_budget(sys.argv[1:], tier='manual-evidence', budget_seconds={budget_seconds!r}))",
            *command,
        ],
        cwd=project,
        env={**os.environ, "EVIDENCE": str(evidence)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert outcome.returncode == 124, outcome.stdout + outcome.stderr
    summary = json.loads((evidence / "run-manual" / "run.json").read_text())
    assert summary["status"] == "finished"
    assert summary["exit_code"] == 124
    assert summary["termination"] == "process_exit"


def test_affected_run_evidence_retains_selection_identity(tmp_path):
    from tests.test_impact_selection import _make_affected_fixture, _run_affected

    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    evidence = tmp_path / "selection-evidence"
    outcome = _run_affected(project, tmp_path, base, "fast", "--evidence-root", str(evidence))
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    manifest = json.loads(next(evidence.glob("run-*/run.json")).read_text())
    assert manifest["validation_kind"] == "affected"
    assert manifest["affected_base"] == base
    assert manifest["tier"] == "fast"
    assert manifest["selected_tests"] == 2
    assert manifest["deselected_tests"] == 1
    assert manifest["selection_status"] == "selected"
    assert manifest["outcome"] == "passed"
    assert manifest["dirty_inputs"]


@pytest.mark.slow
def test_affected_xdist_run_retains_controller_selection_and_worker_results(tmp_path):
    from tests.test_impact_selection import _make_affected_fixture, _run_affected

    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    evidence = tmp_path / "xdist-evidence"
    outcome = _run_affected(
        project,
        tmp_path,
        base,
        "fast",
        "-n",
        "2",
        "--dist",
        "loadgroup",
        "--evidence-root",
        str(evidence),
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    manifest_path = next(evidence.glob("run-*/run.json"))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["selected_tests"] == 2
    assert manifest["deselected_tests"] == 1
    assert manifest["worker_allowance"] == "2"
    assert manifest["resolved_worker_count"] == 2
    assert manifest["selection_route"] == "import_graph"
    assert manifest["outcome"] == "passed"
    reports = [
        json.loads(line)
        for line in manifest_path.with_name("reports.jsonl").read_text().splitlines()
    ]
    assert {
        report["nodeid"].rsplit("::", 1)[-1]
        for report in reports
        if report.get("event") == "report"
        and report["when"] == "call"
        and report["outcome"] == "passed"
    } == {
        "test_direct",
        "test_transitive",
    }
    assert set(manifest["selected_nodeids"]) == {
        report["nodeid"]
        for report in reports
        if report.get("event") == "report" and report["when"] == "call"
    }


def test_affected_evidence_fails_when_selected_tests_are_incompletely_reported(tmp_path):
    from types import SimpleNamespace

    from tests._evidence import _RUN, _SELECTION, pytest_sessionfinish

    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"status": "running", "exit_code": None}))
    counts = {"selected": 2, "deselected": 1, "executed": 0}
    (run / "reports.jsonl").write_text(
        json.dumps(
            {
                "event": "report",
                "nodeid": "tests/test_a.py::test_a",
                "when": "setup",
                "outcome": "passed",
            }
        )
        + "\n"
    )
    session = SimpleNamespace(
        config=SimpleNamespace(stash={_RUN: run, _SELECTION: counts}),
        exitstatus=0,
    )
    pytest_sessionfinish(cast(Any, session), 0)
    manifest = json.loads((run / "run.json").read_text())
    assert session.exitstatus != 0
    assert manifest["outcome"] == "incomplete"
    assert manifest["executed_tests"] == 0


def test_affected_evidence_counts_setup_skip_as_terminal_outcome(tmp_path):
    from types import SimpleNamespace

    from tests._evidence import _RUN, _SELECTION, pytest_sessionfinish

    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"status": "running", "exit_code": None}))
    counts = {"selected": 1, "deselected": 0}
    (run / "reports.jsonl").write_text(
        json.dumps(
            {
                "event": "report",
                "nodeid": "tests/test_a.py::test_a",
                "when": "setup",
                "outcome": "skipped",
            }
        )
        + "\n"
    )
    session = SimpleNamespace(
        config=SimpleNamespace(stash={_RUN: run, _SELECTION: counts}),
        exitstatus=0,
    )
    pytest_sessionfinish(cast(Any, session), 0)
    manifest = json.loads((run / "run.json").read_text())
    assert session.exitstatus == 0
    assert manifest["outcome"] == "passed"
    assert manifest["executed_tests"] == 1


def test_xdist_crash_without_worker_output_does_not_raise():
    from types import SimpleNamespace

    from tests._evidence import _XdistSelectionEvidence

    plugin = _XdistSelectionEvidence(cast(Any, SimpleNamespace()))
    plugin.pytest_testnodedown(cast(Any, SimpleNamespace()), RuntimeError("worker crashed"))


def _write_affected_evidence_case(
    root: Path,
    *,
    selected: list[str],
    reports: list[dict[str, object]] | None,
    status: str = "finished",
    owner: str = "owner",
) -> Path:
    run = root / "run-case"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps(
            {
                "validation_kind": "affected",
                "status": status,
                "owner_token": owner,
                "selected_nodeids": selected,
            }
        )
    )
    if reports is not None:
        (run / "reports.jsonl").write_text("".join(json.dumps(event) + "\n" for event in reports))
    return root


@pytest.mark.parametrize(
    ("selected", "terminal", "status", "owner", "pytest_status", "expected"),
    [
        (["a"], ["a"], "finished", "owner", 0, None),
        (["a"], [], "finished", "owner", 0, "incomplete"),
        (["a", "b"], ["a"], "finished", "owner", 0, "incomplete"),
        (["a"], ["a", "extra"], "finished", "owner", 0, "incomplete"),
        (["a"], ["different"], "finished", "owner", 0, "incomplete"),
        ([], [], "finished", "owner", 1, "pytest_failed"),
        (["a"], ["a"], "running", "owner", 0, "incomplete"),
        (["a"], ["a"], "finished", "other", 0, "incomplete"),
        (["a"], ["a"], "finished", "owner", 1, "pytest_failed"),
    ],
)
def test_affected_runner_accepts_only_complete_owned_nodeid_evidence(
    tmp_path: Path,
    selected: list[str],
    terminal: list[str],
    status: str,
    owner: str,
    pytest_status: int,
    expected: str | None,
) -> None:
    from scripts import run_test_tier

    outcome = run_test_tier._validate_affected_evidence(
        _write_affected_evidence_case(
            tmp_path,
            selected=selected,
            reports=[
                {
                    "event": "report",
                    "nodeid": nodeid,
                    "when": "call",
                    "outcome": "passed",
                }
                for nodeid in terminal
            ],
            status=status,
            owner=owner,
        ),
        owner_token="owner",
        pytest_status=pytest_status,
    )
    if expected is None:
        assert outcome is None
    else:
        assert outcome is not None
        assert expected in outcome


def test_affected_runner_requires_evidence_after_zero_exit(tmp_path: Path) -> None:
    from scripts import run_test_tier

    status = run_test_tier.run_with_budget(
        [sys.executable, "-c", "pass"],
        tier="fast",
        budget_seconds=10,
        affected_evidence_root=tmp_path,
    )
    assert status != 0


def test_affected_runner_uses_current_stderr_after_capture_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_test_tier

    stream = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stream)

    status = run_test_tier.run_with_budget(
        [sys.executable, "-c", "pass"],
        tier="fast",
        budget_seconds=10,
        affected_evidence_root=tmp_path,
    )

    assert status == 1
    assert "pytest: error: affected evidence rejected" in stream.getvalue()


def test_affected_runner_accepts_setup_skip_terminal_outcome(tmp_path: Path) -> None:
    from scripts import run_test_tier

    evidence_root = _write_affected_evidence_case(
        tmp_path,
        selected=["test.py::test_skip"],
        reports=[
            {
                "event": "report",
                "nodeid": "test.py::test_skip",
                "when": "setup",
                "outcome": "skipped",
            }
        ],
    )
    assert (
        run_test_tier._validate_affected_evidence(
            evidence_root, owner_token="owner", pytest_status=0
        )
        is None
    )
