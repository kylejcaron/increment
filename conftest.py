"""Execute documentation examples and configure test scheduling.

Each document runs in one shared namespace inside a throwaway working
directory. Sybil items are grouped by markdown file so xdist executes their
blocks in order on one worker.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser

_FAST_TIMEOUT_SECONDS = 30
_SLOW_TIMEOUT_SECONDS = 180
_EXPENSIVE_TIMEOUT_SECONDS = 300
# Live warehouse probes cross a network to a managed backend, so their wall
# time is dominated by round trips and table creation rather than by the work
# under test. The local-backend deadlines do not describe them.
_WAREHOUSE_TIMEOUT_SECONDS = 300
_WAREHOUSE_MARKERS = ("warehouse_postgres", "warehouse_snowflake", "warehouse_bigquery")
_DURATION_SNAPSHOT: tuple[tuple[str, float], ...] | None = None


def _load_duration_snapshot(root: Path) -> tuple[tuple[str, float], ...]:
    try:
        durations = json.loads((root / ".test_durations").read_text())
    except FileNotFoundError:
        durations = {}
    return tuple(sorted((str(nodeid), float(value)) for nodeid, value in durations.items()))


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node) -> None:
    """Send workers the controller's single immutable duration-file snapshot."""
    global _DURATION_SNAPSHOT
    if _DURATION_SNAPSHOT is None:
        _DURATION_SNAPSHOT = _load_duration_snapshot(Path(node.config.rootpath))
    node.workerinput["increment_test_duration_snapshot"] = _DURATION_SNAPSHOT


def _duration_snapshot(config: pytest.Config) -> dict[str, float]:
    snapshot = getattr(config, "workerinput", {}).get("increment_test_duration_snapshot")
    if snapshot is None:
        global _DURATION_SNAPSHOT
        if _DURATION_SNAPSHOT is None:
            _DURATION_SNAPSHOT = _load_duration_snapshot(Path(config.rootpath))
        snapshot = _DURATION_SNAPSHOT
    return dict(snapshot)


_CALL_DURATIONS: dict[str, float] = {}

pytest_plugins = ("tests._evidence",)


def _setup(namespace: dict) -> None:
    tmp = tempfile.TemporaryDirectory()
    namespace["__docs_tmp"] = tmp
    namespace["__docs_cwd"] = os.getcwd()
    os.chdir(tmp.name)


def _teardown(namespace: dict) -> None:
    os.chdir(namespace.pop("__docs_cwd"))
    namespace.pop("__docs_tmp").cleanup()


# Every markdown file, so a new snippet-bearing page runs without anyone
# remembering to list it. Files with no python block collect nothing.
pytest_collect_file = Sybil(
    parsers=[PythonCodeBlockParser(), SkipParser()],
    patterns=["*.md"],
    setup=_setup,
    teardown=_teardown,
).pytest()


def _markdown_path(item: pytest.Item) -> str | None:
    """Return the source file for a Sybil item, if it is Markdown."""
    path = Path(str(item.location[0]))
    return path.as_posix() if path.suffix == ".md" else None


def _recorded_duration(item: pytest.Item, durations: dict[str, float]) -> float:
    duration = durations.get(item.nodeid, 0)
    for marker in item.iter_markers("xdist_group"):
        if marker.args:
            duration = max(duration, durations.get(f"{item.nodeid}@{marker.args[0]}", 0))
    return duration


def _order_by_recorded_duration(
    items: list[pytest.Item],
    durations_path: Path,
    durations: dict[str, float] | None = None,
    markdown_order: dict[int, int] | None = None,
) -> None:
    """Order ordinary tests longest-first and retain Markdown source order."""
    if durations is None:
        try:
            durations = json.loads(durations_path.read_text())
        except FileNotFoundError:
            durations = {}
    indexed = list(enumerate(items))
    items[:] = [
        item
        for _, item in sorted(
            indexed,
            key=lambda pair: (
                _markdown_path(pair[1]) is not None,
                (
                    markdown_order.get(id(pair[1]), pair[0])
                    if markdown_order is not None
                    else pair[0]
                )
                if _markdown_path(pair[1]) is not None
                else -_recorded_duration(pair[1], durations),
                pair[0],
            ),
        )
    ]


def _group_markdown_items(items: list[pytest.Item]) -> None:
    """Keep each Sybil file's snippets in one xdist worker."""
    for item in items:
        path = _markdown_path(item)
        if path is not None:
            item.add_marker(pytest.mark.xdist_group(f"docs:{path}"))


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Order tests and assign local performance deadlines."""
    _group_markdown_items(items)
    if os.environ.get("INCREMENT_DISABLE_DURATION_ORDER") != "1":
        _order_by_recorded_duration(
            items,
            Path(config.rootpath) / ".test_durations",
            _duration_snapshot(config),
        )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return
    for item in items:
        if item.get_closest_marker("timeout") is not None:
            continue
        if any(item.get_closest_marker(marker) for marker in _WAREHOUSE_MARKERS):
            seconds = _WAREHOUSE_TIMEOUT_SECONDS
        elif item.get_closest_marker("examples") or item.get_closest_marker("parameter_recovery"):
            seconds = _EXPENSIVE_TIMEOUT_SECONDS
        elif item.get_closest_marker("slow"):
            seconds = _SLOW_TIMEOUT_SECONDS
        else:
            seconds = _FAST_TIMEOUT_SECONDS
        item.add_marker(pytest.mark.timeout(seconds))


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        _CALL_DURATIONS[report.nodeid] = report.duration


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if (
        not _CALL_DURATIONS
        or hasattr(session.config, "workerinput")
        or (session.config.getoption("splits", default=1) or 1) > 1
    ):
        return

    path = Path(session.config.rootpath) / ".test_durations"
    try:
        durations = json.loads(path.read_text())
    except FileNotFoundError:
        durations = {}
    durations.update(_CALL_DURATIONS)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(durations, stream, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)
