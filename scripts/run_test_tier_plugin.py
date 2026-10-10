"""Pytest-only hooks used by :mod:`scripts.run_test_tier`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import pytest


class _ChangedTestTachHandler:
    """Keep directly changed tests in Tach's selection before pytest applies filters."""

    def __init__(self, handler: Any, changed_paths: set[Path]) -> None:
        self._handler = handler
        self._changed_paths = changed_paths

    def __getattr__(self, name: str) -> object:
        return getattr(self._handler, name)

    def should_remove_items(self, file_path: Path) -> bool:
        if file_path.resolve() in self._changed_paths:
            return False
        return self._handler.should_remove_items(file_path)


_DYNAMIC_CONSUMER_SELECTION = pytest.StashKey[dict[str, Any]]()


@pytest.hookimpl(optionalhook=True, tryfirst=True)
def pytest_xdist_setupnodes(config: pytest.Config, specs: list[Any]) -> None:
    """Validate xdist's effective worker list before spawning any worker."""
    if not os.environ.get("INCREMENT_AFFECTED_EVIDENCE"):
        return
    expected = os.environ.get("INCREMENT_AFFECTED_WORKER_ALLOWANCE", "serial")
    count = len(specs)
    if expected == "serial" or not 1 <= count <= 8 or any(not spec.popen for spec in specs):
        raise pytest.UsageError(
            "effective affected xdist worker list must contain 1-8 local popen workers"
        )
    if count != int(expected):
        raise pytest.UsageError(
            f"effective affected xdist worker list has {count} workers; expected {expected}"
        )
    for spec in specs:
        spec.env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    os.environ["INCREMENT_AFFECTED_RESOLVED_WORKER_COUNT"] = str(count)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    # Affected inputs are validated by the parser-backed launcher before collection.
    raw_paths = json.loads(os.environ.get("INCREMENT_AFFECTED_TEST_PATHS", "[]"))
    changed_paths = {Path(path).resolve() for path in raw_paths}
    if not changed_paths:
        return
    from tach.pytest_plugin import tach_state_key

    state = config.stash.get(tach_state_key, None)
    if state is None:
        raise pytest.UsageError("Tach selection is unavailable for changed test files")
    cast(Any, state).handler = _ChangedTestTachHandler(state.handler, changed_paths)
    dynamic_paths = {
        Path(path).resolve()
        for path in json.loads(os.environ.get("INCREMENT_AFFECTED_DYNAMIC_TEST_PATHS", "[]"))
    }
    if dynamic_paths:
        config.stash[_DYNAMIC_CONSUMER_SELECTION] = {
            "paths": dynamic_paths,
            "collected": {},
            "controller_selected": {},
            "worker_collected": None,
            "worker_selected": None,
        }


_AFFECTED_COLLECTION = pytest.StashKey[dict[str, int]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--focused-file-defaults", action="store_true")
    parser.addoption("--focused-path-validation", action="store_true")


def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """Refuse a focused run naming any selector that is not an existing test file."""
    if not config.getoption("focused_path_validation"):
        return None
    base = config.invocation_params.dir
    rejected = [
        selector for selector in config.args if not (base / selector.split("::", 1)[0]).is_file()
    ]
    if rejected:
        names = ", ".join(repr(selector) for selector in rejected)
        print(
            f"pytest: error: focused selectors must be existing test files: {names}",
            file=sys.stderr,
        )
        return 4
    return None


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Apply focused defaults and record original and final selection."""
    if config.getoption("focused_file_defaults"):
        explicit: dict[Path, list[str]] = {}
        for selector in config.args:
            file, separator, node = selector.partition("::")
            if separator:
                explicit.setdefault(Path(file).resolve(), []).append(node)
        paths: dict[Path, Path] = {}
        kept, deselected = [], []
        for item in items:
            if not any(item.get_closest_marker(mark) for mark in ("slow", "parameter_recovery")):
                kept.append(item)
                continue
            if item.path not in paths:
                paths[item.path] = item.path.resolve()
            node = item.nodeid.partition("::")[2]
            selected = any(
                node == requested
                or node.startswith(requested + "[")
                or node.startswith(requested + "::")
                for requested in explicit.get(paths[item.path], ())
            )
            (kept if selected else deselected).append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
        items[:] = kept
    affected = bool(os.environ.get("INCREMENT_AFFECTED_EVIDENCE"))
    dynamic_paths = {
        Path(path).resolve()
        for path in json.loads(os.environ.get("INCREMENT_AFFECTED_DYNAMIC_TEST_PATHS", "[]"))
    }
    dynamic_nodes: dict[Path, set[str]] = {}
    original_nodeids = {id(item): item.nodeid for item in items}
    for item in items:
        path = item.path.resolve()
        if path in dynamic_paths:
            dynamic_nodes.setdefault(path, set()).add(original_nodeids[id(item)])
    collected = len(items)
    if affected:
        config.stash[_AFFECTED_COLLECTION] = {
            "selected": 0,
            "deselected": 0,
            "collected": collected,
        }
    if dynamic_paths:
        config.stash[_DYNAMIC_CONSUMER_SELECTION] = {
            "paths": dynamic_paths,
            "collected": dynamic_nodes,
            "controller_selected": {},
            "worker_collected": None,
            "worker_selected": None,
        }
    yield
    counts = config.stash.get(_AFFECTED_COLLECTION, None)
    if counts is not None:
        counts["selected"] = len(items)
        counts["deselected"] = collected - len(items)
    selection = config.stash.get(_DYNAMIC_CONSUMER_SELECTION, None)
    if selection is not None:
        selected_nodes: dict[Path, set[str]] = {}
        for item in items:
            path = item.path.resolve()
            if path in dynamic_paths:
                selected_nodes.setdefault(path, set()).add(original_nodeids[id(item)])
        selection["controller_selected"] = selected_nodes


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node: object, ids: list[str]) -> None:
    config = getattr(node, "config", None)
    if config is None:
        return
    counts = config.stash.get(_AFFECTED_COLLECTION, None)
    if counts is not None:
        counts["selected"] = len(ids)

    selection = config.stash.get(_DYNAMIC_CONSUMER_SELECTION, None)
    if selection is not None:
        selected = set(ids)
        previous = selection["worker_selected"]
        selection["worker_selected"] = (
            selected if previous is None else previous.intersection(selected)
        )


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: object, error: object) -> None:
    config = getattr(node, "config", None)
    workeroutput = getattr(node, "workeroutput", None)
    if config is None or not isinstance(workeroutput, dict):
        return
    incoming = workeroutput.get("increment_dynamic_selection")
    selection = config.stash.get(_DYNAMIC_CONSUMER_SELECTION, None)
    if not isinstance(incoming, dict) or selection is None:
        return
    collected = {
        Path(path): set(nodeids) for path, nodeids in incoming.get("collected", {}).items()
    }
    selected = {Path(path): set(nodeids) for path, nodeids in incoming.get("selected", {}).items()}
    previous_collected = selection["worker_collected"]
    if previous_collected is None:
        selection["worker_collected"] = collected
        selection["worker_selected"] = selected
        return
    for path, nodeids in collected.items():
        previous_collected.setdefault(path, set()).update(nodeids)
    previous_selected = selection["worker_selected"]
    for path in selection["paths"]:
        previous_selected[path] = previous_selected.get(path, set()).intersection(
            selected.get(path, set())
        )


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    selection = config.stash.get(_DYNAMIC_CONSUMER_SELECTION, None)
    if selection is None or hasattr(config, "workerinput"):
        return
    worker_collected = selection["worker_collected"]
    if worker_collected is not None:
        collected: dict[Path, set[str]] = worker_collected
        worker_selected: dict[Path, set[str]] = selection["worker_selected"]
        selected_nodes = {nodeid for nodeids in worker_selected.values() for nodeid in nodeids}
    else:
        selected_nodes = selection["worker_selected"]
        if selected_nodes is None:
            selected_nodes = {
                nodeid
                for nodeids in selection["controller_selected"].values()
                for nodeid in nodeids
            }
        collected = selection["collected"]
    paths: set[Path] = selection["paths"]
    incomplete: dict[Path, list[str]] = {}
    for path in paths:
        nodeids = collected.get(path, set())
        missing = sorted(nodeids.difference(selected_nodes))
        if not nodeids:
            incomplete[path] = ["<no tests collected>"]
        elif missing:
            incomplete[path] = missing
    tier = "affected"
    try:
        tier = str(json.loads(os.environ["INCREMENT_AFFECTED_EVIDENCE"]).get("tier", tier))
    except (KeyError, TypeError, ValueError):
        pass
    details = []
    for path, nodeids in sorted(incomplete.items()):
        try:
            display_path = path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            display_path = path.as_posix()
        preview = ", ".join(nodeids[:3])
        if len(nodeids) > 3:
            preview += f", … +{len(nodeids) - 3}"
        details.append(f"{display_path} [{preview}]")
    terminalreporter.write_sep(
        "=",
        f"Affected dynamic-import consumers: retained {len(paths)} files; "
        f"collected {len(collected)}; fully selected {len(paths) - len(incomplete)}; "
        f"{len(incomplete)} not fully run in the {tier} tier/pytest selection "
        f"(including -k/-m); omitted: {', '.join(details) if details else 'none'}. "
        "Run the relevant slow or broader validation for omitted consumers.",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    workeroutput = getattr(session.config, "workeroutput", None)
    counts = session.config.stash.get(_AFFECTED_COLLECTION, None)
    if workeroutput is not None and counts is not None:
        workeroutput["increment_affected_selection"] = {
            key: counts[key] for key in ("selected", "deselected")
        }
    selection = session.config.stash.get(_DYNAMIC_CONSUMER_SELECTION, None)
    if workeroutput is not None and selection is not None:
        workeroutput["increment_dynamic_selection"] = {
            "collected": {
                path.as_posix(): sorted(nodeids) for path, nodeids in selection["collected"].items()
            },
            "selected": {
                path.as_posix(): sorted(nodeids)
                for path, nodeids in selection["controller_selected"].items()
            },
        }
