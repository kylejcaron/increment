"""Execute documentation examples and configure test scheduling.

Each document runs in one shared namespace inside a throwaway working
directory. Sybil items are grouped by markdown file so xdist executes their
blocks in order on one worker.
"""

import contextlib
import difflib
import doctest
import io
import json
import os
import re
import tempfile
import time as _time
from functools import cache
from pathlib import Path

from tests._duckdb_resources import bound_duckdb_resources as _bound_duckdb_resources

for _thread_limit in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_thread_limit, "1")

from hypothesis import HealthCheck, settings  # noqa: E402

settings.register_profile("dev", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("ci", deadline=200, suppress_health_check=[])
settings.load_profile("ci" if "CI" in os.environ else "dev")


_bound_duckdb_resources()


import pytest  # noqa: E402

from tests._unit_boundary import (  # noqa: E402
    check_backend_fixtures,
    install_guards,
    set_current_unit,
)

install_guards()


_UNIT_CPU_BUDGET_SECONDS = 0.1
_unit_cpu_baseline: set[str] = set()
_unit_cpu_warmups: dict[str, float] = {}
_unit_cpu_measurements: dict[str, float] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--unit-cpu-budget",
        action="store_true",
        help="report per-test CPU time for unit-selected tests",
    )
    parser.addoption(
        "--unit-cpu-budget-enforce",
        action="store_true",
        help="fail unit-selected tests above the CPU-time budget",
    )


def pytest_configure(config: pytest.Config) -> None:
    if not (config.getoption("--unit-cpu-budget") or config.getoption("--unit-cpu-budget-enforce")):
        return
    baseline = Path(
        os.environ.get(
            "UNIT_CPU_BASELINE", str(Path(__file__).parent / "tests/unit_cpu_baseline.txt")
        )
    )
    if baseline.exists():
        _unit_cpu_baseline.update(
            line.strip()
            for line in baseline.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    config = terminalreporter.config
    if not (config.getoption("--unit-cpu-budget") or config.getoption("--unit-cpu-budget-enforce")):
        return
    if hasattr(config, "workerinput"):
        return
    for worker, seconds in sorted(_unit_cpu_warmups.items()):
        terminalreporter.write_line(f"unit CPU import warm-up ({worker}): {seconds * 1000:.1f} ms")
    unbaselined = sorted(
        (nodeid, seconds)
        for nodeid, seconds in _unit_cpu_measurements.items()
        if seconds > _UNIT_CPU_BUDGET_SECONDS and nodeid not in _unit_cpu_baseline
    )
    for nodeid, seconds in unbaselined:
        terminalreporter.write_line(f"unit CPU over budget: {seconds * 1000:.1f} ms ({nodeid})")
    stale = sorted(
        nodeid
        for nodeid in _unit_cpu_baseline
        if nodeid in _unit_cpu_measurements
        and _unit_cpu_measurements[nodeid] <= _UNIT_CPU_BUDGET_SECONDS
    )
    if stale:
        terminalreporter.write_sep("=", "stale unit CPU baseline entries (notes only)")
        for nodeid in stale:
            terminalreporter.write_line(nodeid)


from sybil import Document, Example, Region, Sybil  # noqa: E402
from sybil.evaluators.python import PythonEvaluator  # noqa: E402
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser  # noqa: E402
from sybil.parsers.markdown.lexers import (  # noqa: E402
    DirectiveInHTMLCommentLexer,
    FencedCodeBlockLexer,
)


class _CapturingPythonEvaluator:
    def __init__(self) -> None:
        self.evaluator = PythonEvaluator()

    def __call__(self, example: Example) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.evaluator(example)
        example.namespace["__last_output"] = output.getvalue()
        example.namespace["__last_executed_python_region"] = example.region.start


class PythonOutputBlockParser(PythonCodeBlockParser):
    def __init__(self) -> None:
        super().__init__()
        self.codeblock_parser._evaluator = _CapturingPythonEvaluator()


class OutputBlockParser:
    """Compare explicitly marked text fences with the preceding Python output."""

    _text_fences = FencedCodeBlockLexer(
        language=r"text", mapping={"language": "arguments", "source": "source"}
    )
    _python_fences = FencedCodeBlockLexer(language=r"python")
    _invisible_python_blocks = DirectiveInHTMLCommentLexer(
        directive=r"invisible-code-block", arguments=r"python"
    )
    _output_marker = re.compile(r"<!--\s*output\s*-->[ \t\r\n]*$")
    _checker = doctest.OutputChecker()
    _flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

    def __call__(self, document: Document):
        text = document.text
        python_blocks = sorted(
            (
                *self._python_fences(document),
                *self._invisible_python_blocks(document),
            ),
            key=lambda code: code.end,
        )
        for block in self._text_fences(document):
            prefix = text[: block.start]
            marker = self._output_marker.search(prefix)
            if marker is None:
                continue
            python = next(
                (code for code in reversed(python_blocks) if code.end <= marker.start()), None
            )
            if python is None:
                continue
            yield Region(
                block.start,
                block.end,
                parsed=(block.lexemes["source"], python.start),
                evaluator=self._evaluate_output,
            )

    @classmethod
    def _evaluate_output(cls, example: Example) -> None:
        expected_source, python_start = example.parsed
        if example.namespace.get("__last_executed_python_region") != python_start:
            return
        actual = example.namespace.get("__last_output", "")
        expected = str(expected_source)
        if cls._checker.check_output(expected, actual, cls._flags):
            return
        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                actual.splitlines(keepends=True),
                fromfile="documented output",
                tofile="captured output",
            )
        )
        raise AssertionError(f"documented output does not match captured output:\n{diff}")


_DURATION_SNAPSHOT: tuple[tuple[str, float], ...] | None = None
_PYTEST_ENVIRONMENT_SNAPSHOT = pytest.StashKey[dict[str, str]]()

_GIT_LOCAL_ENVIRONMENT = (
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG",
    "GIT_CONFIG_PARAMETERS",
    "GIT_CONFIG_COUNT",
    "GIT_OBJECT_DIRECTORY",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_GRAFT_FILE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_REPLACE_REF_BASE",
    "GIT_PREFIX",
    "GIT_INTERNAL_SUPER_PREFIX",
    "GIT_SHALLOW_FILE",
    "GIT_COMMON_DIR",
)


def _clear_git_repository_environment() -> None:
    for name in _GIT_LOCAL_ENVIRONMENT:
        os.environ.pop(name, None)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session: pytest.Session) -> None:
    """Keep test subprocesses from inheriting a hook's repository-local Git state."""
    _clear_git_repository_environment()


@pytest.hookimpl(specname="pytest_sessionstart", tryfirst=True)
def pytest_unit_cpu_sessionstart(session: pytest.Session) -> None:
    config = session.config
    if not (config.getoption("--unit-cpu-budget") or config.getoption("--unit-cpu-budget-enforce")):
        return
    is_worker = hasattr(config, "workerinput")
    numprocesses = config.getoption("numprocesses", default=0)
    if not is_worker and numprocesses not in (None, 0, "0"):
        return
    started = _time.process_time()
    import increment

    for name in increment.__all__:
        getattr(increment, name)
    seconds = _time.process_time() - started
    worker_id = config.workerinput.get("workerid", "local") if is_worker else "local"
    _unit_cpu_warmups[worker_id] = seconds
    if is_worker:
        config.workeroutput["increment_unit_cpu_warmup"] = seconds


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node, error) -> None:
    workeroutput = getattr(node, "workeroutput", None) or {}
    seconds = workeroutput.get("increment_unit_cpu_warmup")
    if seconds is not None:
        _unit_cpu_warmups[node.gateway.id] = float(seconds)


def _pytest_environment_controls() -> dict[str, str]:
    prefixes = ("COV_CORE_", "COVERAGE_")
    controls = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_PLUGINS", "PYTEST_ADDOPTS"}
    return {
        name: value
        for name, value in os.environ.items()
        if name in controls or name.startswith(prefixes)
    }


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Snapshot subprocess controls before any fixture of any item type runs."""
    item.stash[_PYTEST_ENVIRONMENT_SNAPSHOT] = _pytest_environment_controls()


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> None:
    """Fail a test that leaves a nested-run control changed after its teardown."""
    before = item.stash.get(_PYTEST_ENVIRONMENT_SNAPSHOT, None)
    if before is None:
        return
    after = _pytest_environment_controls()
    changed = sorted(
        name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
    )
    if changed:
        name = changed[0]
        pytest.fail(
            f"test leaked environment variable {name}={after.get(name)!r}",
            pytrace=False,
        )


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
    parsers=[PythonOutputBlockParser(), OutputBlockParser(), SkipParser()],
    patterns=["*.md"],
    setup=_setup,
    teardown=_teardown,
).pytest()


@cache
def _markdown_path(path: str) -> str | None:
    """Return the source file for a Sybil path, if it is Markdown."""
    markdown_path = Path(path)
    return markdown_path.as_posix() if markdown_path.suffix == ".md" else None


def _markdown_item_path(item: pytest.Item) -> str | None:
    return _markdown_path(str(item.location[0]))


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
                _markdown_item_path(pair[1]) is not None,
                (
                    markdown_order.get(id(pair[1]), pair[0])
                    if markdown_order is not None
                    else pair[0]
                )
                if _markdown_item_path(pair[1]) is not None
                else -_recorded_duration(pair[1], durations),
                pair[0],
            ),
        )
    ]


def _group_markdown_items(items: list[pytest.Item]) -> None:
    """Keep each Sybil file's snippets in one xdist worker."""
    for item in items:
        path = _markdown_item_path(item)
        if path is not None:
            item.add_marker(pytest.mark.xdist_group(f"docs:{path}"))


@pytest.hookimpl(hookwrapper=True, tryfirst=True, specname="pytest_runtest_setup")
def pytest_unit_boundary_runtest_setup(item: pytest.Item):
    set_current_unit(item)
    try:
        yield
    finally:
        set_current_unit(None)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item: pytest.Item):
    set_current_unit(item)
    budget_enabled = item.config.getoption("--unit-cpu-budget") or item.config.getoption(
        "--unit-cpu-budget-enforce"
    )
    from tests._unit_boundary import is_unit_selected

    should_measure = budget_enabled and is_unit_selected(item)
    start = _time.process_time() if should_measure else 0.0
    try:
        outcome = yield
    finally:
        if should_measure:
            elapsed = _time.process_time() - start
            item.user_properties.append(("unit_cpu_seconds", elapsed))
            _unit_cpu_measurements[item.nodeid] = elapsed
            if (
                item.config.getoption("--unit-cpu-budget-enforce")
                and elapsed > _UNIT_CPU_BUDGET_SECONDS
                and item.nodeid not in _unit_cpu_baseline
            ):
                outcome.force_exception(
                    pytest.fail.Exception(
                        f"unit CPU budget exceeded ({elapsed * 1000:.1f} ms > 100 ms); "
                        "mark slow/integration for a real boundary, or make the test cheaper"
                    )
                )
        set_current_unit(None)


@pytest.hookimpl(hookwrapper=True, tryfirst=True, specname="pytest_runtest_teardown")
def pytest_unit_boundary_runtest_teardown(item: pytest.Item):
    set_current_unit(item)
    try:
        yield
    finally:
        set_current_unit(None)


def _is_live_warehouse_item(item: pytest.Item, root: Path) -> bool:
    """Live and compile-only warehouse checks live under ``integration/``."""
    try:
        relative = Path(item.path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return relative.parts[:1] == ("integration",)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Validate categories and order tests."""
    root = Path(config.rootpath)
    for item in items:
        if _markdown_item_path(item) is not None or _is_live_warehouse_item(item, root):
            item.add_marker(pytest.mark.integration)

    check_backend_fixtures(items)
    for item in items:
        if item.get_closest_marker("slow") is not None:
            continue
        for category in (
            "parameter_recovery",
            "numerical_certification",
            "examples",
            "stress",
        ):
            if item.get_closest_marker(category) is not None:
                raise pytest.UsageError(f"{item.nodeid}: {category} requires the slow mark")
    _group_markdown_items(items)
    if os.environ.get("INCREMENT_DISABLE_DURATION_ORDER") != "1":
        _order_by_recorded_duration(
            items,
            Path(config.rootpath) / ".test_durations",
            _duration_snapshot(config),
        )


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        _CALL_DURATIONS[report.nodeid] = report.duration
        for name, value in report.user_properties:
            if name == "unit_cpu_seconds":
                _unit_cpu_measurements[report.nodeid] = float(value)


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
