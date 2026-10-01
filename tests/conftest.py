"""Shared session-scoped DuckDB fixtures for the test suite."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

# Polars' thread pool defaults to one per core; under xdist that multiplies
# into far more threads than cores and dominates CPU on tiny per-test frames.
os.environ.setdefault("POLARS_MAX_THREADS", "1")

if TYPE_CHECKING:
    from tach.extension import TachPytestPluginHandler


class _FilesystemGuardHandler:
    """Preserve filesystem guards that an import graph cannot track."""

    def __init__(self, handler: "TachPytestPluginHandler") -> None:
        self._handler = handler
        self._guard_paths = {
            Path(__file__).with_name("test_refusal_gate.py").resolve(),
            Path(__file__).with_name("test_test_hygiene.py").resolve(),
            Path(__file__).with_name("test_impact_selection.py").resolve(),
            Path(__file__).with_name("test_full_suite_evidence.py").resolve(),
            Path(__file__).with_name("test_test_runner_cli.py").resolve(),
        }
        self._decisions: dict[Path, bool] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self._handler, name)

    def should_remove_items(self, file_path: Path) -> bool:
        # Asked once per test item; the answer depends only on the file.
        resolved = file_path.resolve()
        if resolved not in self._decisions:
            self._decisions[resolved] = (
                resolved not in self._guard_paths
                and self._handler.should_remove_items(file_path=file_path)
            )
        return self._decisions[resolved]


def _cap_duckdb_threads() -> None:
    """Default every ibis DuckDB connection to one thread.

    DuckDB's pool also defaults to one thread per core, so under xdist each
    test connection multiplies that by the worker count on tiny tables. An
    explicit ``threads=`` still wins, and ``threads=None`` asks for DuckDB's
    own default so one builder module keeps exercising its parallel operators.
    """
    from ibis.backends.duckdb import Backend

    connect = Backend.do_connect

    def capped_connect(self, *args, **kwargs):
        if kwargs.setdefault("threads", 1) is None:
            del kwargs["threads"]
        return connect(self, *args, **kwargs)

    Backend.do_connect = capped_connect


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    _cap_duckdb_threads()
    if config.getoption("--tach", default=None) is None:
        return

    from tach.pytest_plugin import tach_state_key

    state = config.stash.get(tach_state_key, None)
    if state is None:
        return
    if not state.skip_enabled:
        # Without --tach the plugin only prints a would-skip hint, yet scans
        # every item for it: seconds of collection on every xdist worker.
        del config.stash[tach_state_key]
        return
    state.handler = cast("TachPytestPluginHandler", _FilesystemGuardHandler(state.handler))


@pytest.fixture(scope="session")
def con():
    """In-memory DuckDB with an ``analytics.event_log`` table."""
    import ibis

    from tests.analysis_factory import _make_event_log_table

    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    return con


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
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
