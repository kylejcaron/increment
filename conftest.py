"""Execute the ```python blocks in README.md and the docs so prose can't rot.

Each document runs in one shared namespace inside a throwaway working
directory, so snippets may write files (e.g. definitions/) freely.

The chdir is process-global, so doc examples are serial-only: running them
under a parallel distributor would race the shared cwd.
"""

import os
import tempfile

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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Assign local performance deadlines; GitHub Actions uses job hang guards."""
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
