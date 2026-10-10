"""Retain full-suite temporary outputs and JUnit without changing test selection."""

import json
import os
import platform
import tempfile
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from xdist.workermanage import WorkerController

_AFFECTED = pytest.StashKey[dict[str, object]]()
_SELECTION = pytest.StashKey[dict[str, Any]]()
_RUN = pytest.StashKey[Path]()
_RESOLVED_WORKER_COUNT = pytest.StashKey[int]()
_ROOT = pytest.StashKey[Path]()


class _ReportJournal:
    """Write controller events before an outer deadline can terminate pytest."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.started = time.monotonic()

    def _append(self, event: dict[str, object]) -> None:
        event["elapsed_seconds"] = time.monotonic() - self.started
        with self.destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def pytest_runtest_logstart(self, nodeid: str) -> None:
        self._append({"event": "start", "nodeid": nodeid})

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        event: dict[str, object] = {
            "event": "report",
            "nodeid": report.nodeid,
            "when": report.when,
            "outcome": report.outcome,
            "duration": report.duration,
        }
        if report.failed:
            event["longrepr"] = report.longreprtext
        self._append(event)


class _XdistSelectionEvidence:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(
        self, node: "WorkerController", ids: list[str]
    ) -> None:
        counts = self.config.stash.get(_SELECTION, None)
        if counts is None:
            return
        selected = sorted(set(ids))
        counts["selected_nodeids"] = selected
        counts["selected"] = len(selected)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: "WorkerController", error: object) -> None:
        workeroutput = getattr(node, "workeroutput", None)
        if workeroutput is None:
            return
        worker_counts = workeroutput.get("increment_affected_selection")
        counts = self.config.stash.get(_SELECTION, None)
        if counts is not None and worker_counts is not None:
            counts["selected"] = max(counts["selected"], worker_counts["selected"])
            counts["deselected"] = max(counts["deselected"], worker_counts["deselected"])


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(config: pytest.Config, specs: Sequence[object]) -> None:
    if config.stash.get(_AFFECTED, None) is not None:
        config.stash[_RESOLVED_WORKER_COUNT] = len(specs)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--evidence-root",
        metavar="DIRECTORY",
        help="Retain tmp_path outputs and JUnit in a new run directory, including failures.",
    )
    parser.addoption(
        "--runtime-diagnostics",
        action="store_true",
        help="Retain instrumented test phases and worker resources under --evidence-root.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    affected = os.environ.get("INCREMENT_AFFECTED_EVIDENCE")
    if affected:
        # Pytest has already loaded the explicit plugins; do not leak this
        # startup-only control into test code or nested subprocesses.
        os.environ.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
        config.stash[_AFFECTED] = json.loads(affected)
        config.pluginmanager.register(
            _XdistSelectionEvidence(config), f"increment-xdist-evidence-{id(config)}"
        )
    root = config.getoption("evidence_root")
    if root is None:
        if config.getoption("runtime_diagnostics"):
            raise pytest.UsageError("--runtime-diagnostics requires --evidence-root")
        return
    directory = Path(root).resolve()
    config.stash[_ROOT] = directory
    if hasattr(config, "workerinput"):
        return
    if config.option.basetemp or config.option.xmlpath:
        raise pytest.UsageError(
            "--evidence-root owns --basetemp and --junitxml; do not combine them"
        )
    directory.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="run-", dir=directory))
    config.stash[_RUN] = run
    config.pluginmanager.register(_ReportJournal(run / "reports.jsonl"), "increment-report-journal")
    config.option.basetemp = str(run / "tmp")
    config.option.xmlpath = str(run / "junit.xml")
    config.inicfg["junit_family"] = "legacy"
    # Failed-test payloads must survive pytest's normal temporary-directory pruning.
    config.inicfg["tmp_path_retention_policy"] = "all"
    metadata = {
        "schema_version": 1,
        "status": "running",
        "exit_code": None,
        "owner_token": os.environ.get("INCREMENT_EVIDENCE_OWNER"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": list(config.invocation_params.args),
    }
    (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    affected = os.environ.get("INCREMENT_AFFECTED_EVIDENCE")
    if affected:
        identity = json.loads(affected)
        config.stash[_AFFECTED] = identity
        config.stash[_SELECTION] = {
            "selected": 0,
            "deselected": 0,
            "selected_nodeids": [],
        }
        metadata.update(
            identity,
            selection_status="collecting",
            selected_tests=0,
            deselected_tests=0,
            started_at=time.time(),
        )
        (run / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")


def pytest_deselected(items: list[pytest.Item]) -> None:
    if items:
        counts = items[0].config.stash.get(_SELECTION, None)
        if counts is not None:
            counts["deselected"] += len(items)


def pytest_collection_finish(session: pytest.Session) -> None:
    counts = session.config.stash.get(_SELECTION, None)
    if counts is None:
        return
    selected_nodeids = sorted(item.nodeid for item in session.items)
    counts["selected_nodeids"] = selected_nodeids
    counts["selected"] = len(selected_nodeids)
    run = session.config.stash.get(_RUN, None)
    if run is not None:
        path = run / "run.json"
        metadata = json.loads(path.read_text())
        metadata.update(
            selected_tests=counts["selected"],
            selected_nodeids=selected_nodeids,
            deselected_tests=counts["deselected"],
            selection_status="selected" if counts["selected"] else "no_affected_tests",
        )
        path.write_text(json.dumps(metadata, indent=2) + "\n")


@pytest.fixture
def runtime_diagnostics(request: pytest.FixtureRequest):
    from tests._runtime_diagnostics import RuntimeDiagnostics

    directory = (
        request.getfixturevalue("tmp_path")
        if request.config.getoption("runtime_diagnostics")
        else None
    )
    diagnostics = RuntimeDiagnostics(directory, nodeid=request.node.nodeid)
    if directory is not None:
        request.node.user_properties.append(
            ("evidence_path", str(directory / "runtime-diagnostics.json"))
        )
    try:
        yield diagnostics
    finally:
        diagnostics.close()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item,
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    root = item.config.stash.get(_ROOT, None)
    if root is not None:
        for index, (name, value) in enumerate(report.user_properties):
            if name == "evidence_path" and isinstance(value, str):
                path = Path(value)
                if path.is_absolute() and path.is_relative_to(root):
                    report.user_properties[index] = (name, path.relative_to(root).as_posix())
    return report


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    run = config.stash.get(_RUN, None)
    counts = config.stash.get(_SELECTION, None)
    no_selection = counts is not None and counts["selected"] == 0
    if no_selection and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
    workeroutput = getattr(config, "workeroutput", None)
    if workeroutput is not None and counts is not None:
        workeroutput["increment_affected_selection"] = dict(counts)
    if run is None:
        return

    reports = run / "reports.jsonl"
    executed_nodeids = set()
    if reports.is_file():
        executed_nodeids = {
            event["nodeid"]
            for line in reports.read_text().splitlines()
            if line
            and (event := json.loads(line)).get("event") == "report"
            and (
                event.get("when") == "call"
                or (event.get("when") == "setup" and event.get("outcome") in {"failed", "skipped"})
            )
        }
    executed_tests = len(executed_nodeids)
    incomplete = counts is not None and counts["selected"] != executed_tests
    if incomplete and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
    destination = run / "run.json"
    metadata = json.loads(destination.read_text())
    final_status = (
        0
        if no_selection and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
        else int(session.exitstatus)
    )
    outcome = (
        "no_affected_tests"
        if no_selection
        else "incomplete"
        if incomplete
        else "passed"
        if final_status == 0
        else "failed"
    )
    metadata.update(
        status="finished",
        exit_code=final_status,
        executed_tests=executed_tests,
        selected_nodeids=counts.get("selected_nodeids", []) if counts is not None else [],
        outcome=outcome,
    )
    if "started_at" in metadata:
        metadata["elapsed_seconds"] = time.time() - metadata["started_at"]
    temporary = run / "run.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(destination)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    config = terminalreporter.config
    affected = config.stash.get(_AFFECTED, None)
    counts = config.stash.get(_SELECTION, None)
    run = config.stash.get(_RUN, None)
    if affected is not None and counts is not None and run is not None:
        reports = run / "reports.jsonl"
        if counts["selected"] == 0 and reports.is_file():
            nodeids = {
                event["nodeid"]
                for line in reports.read_text().splitlines()
                if line and (event := json.loads(line)).get("event") in {"start", "report"}
            }
            counts["selected"] = len(nodeids)
        counts["deselected"] = max(
            counts["deselected"], len(terminalreporter.stats.get("deselected", ()))
        )
        manifest = run / "run.json"
        metadata = json.loads(manifest.read_text())
        status = int(metadata["exit_code"])
        incomplete = metadata.get("outcome") == "incomplete"
        metadata.update(
            selected_tests=counts["selected"],
            deselected_tests=counts["deselected"],
            selection_status="selected" if counts["selected"] else "no_affected_tests",
            outcome=(
                "incomplete"
                if incomplete
                else "passed"
                if status == 0 and counts["selected"]
                else "failed"
                if status
                else "no_affected_tests"
            ),
        )
        resolved_worker_count = config.stash.get(_RESOLVED_WORKER_COUNT, None)
        if resolved_worker_count is not None:
            metadata["resolved_worker_count"] = resolved_worker_count
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    if affected is not None and counts is not None:
        if counts["selected"] == 0:
            terminalreporter.write_line("Affected-test selection: no affected tests")
        else:
            terminalreporter.write_line(
                "Affected-test selection: "
                f"{counts['selected']} selected, {counts['deselected']} deselected"
            )
        if run is not None:
            terminalreporter.write_line(f"Affected-test evidence: {run}")
        return
    if run is not None:
        terminalreporter.write_line(f"Full-suite evidence: {run}")
