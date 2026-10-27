"""Shared session-scoped DuckDB fixtures for the test suite."""

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests._selection import AffectedHandler, SelectionGraph, selection_graph
from tests._unit_boundary import backend_fixture

if TYPE_CHECKING:
    from tach.extension import TachPytestPluginHandler

_fixture_closures: dict[str, list[str]] = {}
_fixture_setup: dict[str, list[str]] = {}
_collection_imports: dict[str, list[str]] = {}
_collection_before: dict[str, set[str]] = {}
_current_item: pytest.Item | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--affected-report", default=None)
    parser.addoption("--affected-tier", default="fast")
    parser.addoption("--selection-fixtures", default=None)


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector: pytest.Collector) -> None:
    if not isinstance(collector, pytest.Module):
        return
    try:
        relative = collector.path.resolve().relative_to(Path(collector.config.rootpath).resolve())
    except ValueError:
        return
    name = relative.as_posix()
    if name.startswith("tests/"):
        _collection_before[name] = set(sys.modules)


@pytest.hookimpl(trylast=True)
def pytest_collectreport(report: pytest.CollectReport) -> None:
    name = report.nodeid
    if "::" in name or name not in _collection_before:
        return
    before = _collection_before.pop(name)
    _collection_imports[name] = sorted(
        module
        for module in set(sys.modules) - before
        if module == "increment" or module.startswith("increment.")
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if isinstance(item, pytest.Function) and item.path.is_relative_to(
            Path(item.config.rootpath) / "tests"
        ):
            _fixture_closures[item.nodeid] = sorted(set(item.fixturenames))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> Any:
    global _current_item
    previous = _current_item
    _current_item = item
    try:
        yield
    finally:
        _current_item = previous


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_fixture_setup(fixturedef: Any, request: pytest.FixtureRequest) -> Any:
    outcome = yield
    # Function-scoped fixtures run again for every consumer, whose own setup
    # context records them; only cached (wider-scope) setups need propagation.
    if outcome.excinfo is not None or getattr(request, "scope", fixturedef.scope) == "function":
        return
    node = getattr(request, "_pyfuncitem", None) or _current_item or request.node
    nodeid = getattr(node, "nodeid", None)
    if isinstance(nodeid, str) and nodeid.split("::", 1)[0].startswith("tests/"):
        producers = _fixture_setup.setdefault(fixturedef.argname, [])
        if nodeid not in producers:
            producers.append(nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    fixtures_dir = session.config.getoption("--selection-fixtures", default=None)
    if fixtures_dir:
        worker = getattr(session.config, "workerinput", {}).get("workerid", "master")
        output = Path(fixtures_dir) / f"fixtures.{worker}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "closures": _fixture_closures,
                    "setup": _fixture_setup,
                    "collected_imports": _collection_imports,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    from tach.pytest_plugin import tach_state_key

    state = session.config.stash.get(tach_state_key, None)
    if state is not None and isinstance(state.handler, AffectedHandlerWithReport):
        state.handler.pytest_sessionfinish(session, exitstatus)


class AffectedHandlerWithReport(AffectedHandler):
    def __init__(
        self, handler: "TachPytestPluginHandler", graph: SelectionGraph, report: Path | None
    ) -> None:
        super().__init__(handler, graph)
        self._report_path = report

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if self._report_path is not None:
            self.write_report(
                self._report_path,
                metadata={
                    "shard_id": os.environ.get("AFFECTED_SHARD_ID", ""),
                    "collection_complete": exitstatus == 0 and session.testscollected > 0,
                    "collected_tests": session.testscollected,
                },
            )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    from tach.pytest_plugin import tach_state_key

    state = config.stash.get(tach_state_key, None)
    if state is None or not state.skip_enabled:
        return
    base = config.getoption("--tach-base", default=None)
    if base is None:
        return
    graph = selection_graph(
        Path(config.rootpath),
        base,
        tier=config.getoption("--affected-tier", default="fast"),
    )
    report_option = config.getoption("--affected-report", default=None)
    report = Path(report_option) if report_option else None
    state.handler = cast(
        "TachPytestPluginHandler",
        AffectedHandlerWithReport(state.handler, graph, report),
    )


@backend_fixture(scope="session")
def con():
    """In-memory DuckDB with an ``analytics.event_log`` table."""
    import ibis

    from tests.analysis_factory import _make_event_log_table

    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    return con


@backend_fixture(scope="session")
def seeded_con():
    """A separate in-memory DuckDB, seeded via examples._seed.seed_event_log.

    Distinct from `con` above (which uses the small hand-written fixture):
    a 4-unit-per-arm population is too thin for `infer_lift`'s validity
    guards on any binary metric, so anything that estimates a lift
    end-to-end runs here. The baseline comparison below additionally needs
    the SAME population scripts/capture_parity_baseline.py used to capture
    the committed baseline, or the comparison is meaningless.
    """
    import ibis

    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con)
    return con


@backend_fixture(scope="session")
def seeded_pre_period_con():
    """`seeded_con`'s population plus the 14 days of pre-exposure activity
    CUPED's covariate is built from.

    Kept separate from `seeded_con`: the pinned baseline was
    captured without a pre-period, and adding one changes which units the
    pre-period lookback admits. The post-exposure draws are bit-identical
    either way (`seed_event_log` gives the pre-period its own RNG stream),
    so the two fixtures agree on every unadjusted number.
    """
    import ibis

    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, with_pre_period=True)
    return con


@pytest.fixture(scope="session")
def seeded_defs():
    """Definitions PATH for `seeded_con` - Analysis loads the YAML
    itself and does not accept a pre-loaded Definitions object.
    """
    return "examples/definitions"
