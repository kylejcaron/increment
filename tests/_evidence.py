"""Retain full-suite temporary outputs and JUnit without changing test selection."""

import json
import os
import platform
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

_RUN = pytest.StashKey[Path]()
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
    run = session.config.stash.get(_RUN, None)
    if run is None:
        return
    destination = run / "run.json"
    metadata = json.loads(destination.read_text())
    metadata.update(status="finished", exit_code=int(exitstatus))
    temporary = run / "run.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(destination)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    run = terminalreporter.config.stash.get(_RUN, None)
    if run is not None:
        terminalreporter.write_line(f"Full-suite evidence: {run}")
