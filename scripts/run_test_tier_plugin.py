"""Pytest-only hooks used by :mod:`scripts.run_test_tier`."""

from __future__ import annotations

import ast
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest


class _ChangedTestTachHandler:
    """Keep changed and import-graph consumer tests in Tach's initial selection."""

    def __init__(self, handler: Any, retained_paths: set[Path]) -> None:
        self._handler = handler
        self._retained_paths = retained_paths

    def __getattr__(self, name: str) -> object:
        return getattr(self._handler, name)

    def should_remove_items(self, file_path: Path) -> bool:
        if file_path.resolve() in self._retained_paths:
            return False
        return self._handler.should_remove_items(file_path)


def _testmon_protected_paths(
    root: Path,
    changed_paths: list[str],
    dynamic_paths: list[str],
    subprocess_paths: Sequence[str] = (),
) -> set[Path]:
    """Protect changed/dynamic tests and test-module import consumers."""
    changed = set(changed_paths)
    dynamic = set(dynamic_paths)
    subprocess = set(subprocess_paths)
    changed_test_modules = {
        path for path in changed if path.startswith("tests/") and path.endswith(".py")
    }
    if changed_test_modules:
        from scripts._test_impact import affected_test_paths

        test_import_consumers = set(affected_test_paths(root, sorted(changed_test_modules)))
    else:
        test_import_consumers = set()
    return {
        (root / path).resolve() for path in changed | dynamic | subprocess | test_import_consumers
    }


def _cache_backed_test_paths(root: Path) -> set[Path]:
    paths: set[Path] = set()
    for path in (root / "tests").rglob("test_*.py"):
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, UnicodeDecodeError):
            paths.add(path.resolve())
            continue
        if any(
            (isinstance(node, ast.ImportFrom) and node.module == "tests._shared_cache")
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "tests._shared_cache" for alias in node.names)
            )
            for node in ast.walk(tree)
        ):
            paths.add(path.resolve())
    return paths


_TESTMON_PRIMARY = pytest.StashKey[bool]()
_DYNAMIC_CONSUMER_SELECTION = pytest.StashKey[dict[str, Any]]()
_FULL_RUN_CERTIFICATION = pytest.StashKey[dict[str, Any]]()


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


def _activate_testmon(config: pytest.Config, protected_paths: set[Path]) -> bool:
    """Use Testmon only when the launcher certified the pre-open coverage map."""
    if not os.environ.get("INCREMENT_AFFECTED_USE_TESTMON"):
        return False
    data = getattr(config, "testmon_data", None)
    selector = config.pluginmanager.get_plugin("TestmonSelect")
    root = Path(config.rootpath)
    changed_paths = json.loads(os.environ.get("INCREMENT_AFFECTED_CHANGED_PATHS", "[]"))
    changed_modules = [
        path
        for path in changed_paths
        if isinstance(path, str) and path.endswith(".py") and not path.startswith("tests/")
    ]
    known_files = set()
    if data is not None:
        known_files = {
            (Path(filename) if Path(filename).is_absolute() else root / filename).resolve()
            for filename in getattr(data, "all_files", ())
        }
    primary = (
        os.environ.get("INCREMENT_AFFECTED_TESTMON_READY") == "1"
        and selector is not None
        and data is not None
        and not getattr(data, "new_db", True)
        and not getattr(data, "system_packages_change", None)
        and all((root / path).resolve() in known_files for path in changed_modules)
    )
    if not primary:
        if hasattr(config, "testmon_config"):
            config.testmon_config.select = False
        return False
    if any(isinstance(path, str) and path.endswith(".py") for path in changed_paths):
        protected_paths.update(
            root / "tests" / name
            for name in (
                "test_refusal_gate.py",
                "test_test_hygiene.py",
                "test_impact_selection.py",
                "test_full_suite_evidence.py",
                "test_test_runner_cli.py",
            )
        )
        protected_paths.update(_cache_backed_test_paths(root))
    protected = {path.resolve() for path in protected_paths}
    selector.deselected_files[:] = [
        filename
        for filename in selector.deselected_files
        if (Path(filename) if Path(filename).is_absolute() else root / filename).resolve()
        not in protected
    ]
    selector.deselected_tests[:] = [
        nodeid
        for nodeid in selector.deselected_tests
        if (
            Path(nodeid.split("::", 1)[0])
            if Path(nodeid.split("::", 1)[0]).is_absolute()
            else root / nodeid.split("::", 1)[0]
        ).resolve()
        not in protected
    ]
    return True


def _full_certification_is_safe(config: pytest.Config) -> bool:
    expected_config = os.environ.get("INCREMENT_TESTMON_EXPECTED_CONFIG")
    if (
        expected_config is None
        or Path(config.inipath or "").resolve() != Path(expected_config).resolve()
    ):
        return False
    option = config.option
    expected_mark = os.environ.get("INCREMENT_TESTMON_EXPECTED_MARK", "")
    actual_mark = getattr(option, "markexpr", "") or ""
    if actual_mark != expected_mark:
        return False
    selectors = (
        "keyword",
        "lf",
        "last_failed",
        "failedfirst",
        "stepwise",
        "collectonly",
        "deselect",
        "ignore",
        "ignore_glob",
        "splits",
        "group",
        "file_or_dir",
    )
    return not any(
        bool(value) for name in selectors if (value := getattr(option, name, None)) is not None
    )


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    report_path = os.environ.get("INCREMENT_TESTMON_CERTIFICATION_REPORT")
    if report_path:
        workers = config.getoption("numprocesses", default=0)
        dist = config.getoption("dist", default=None)
        if workers in (None, 0, "0"):
            dist = None
        config.stash[_FULL_RUN_CERTIFICATION] = {
            "path": report_path,
            "safe": _full_certification_is_safe(config),
            "collected": set(),
            "reported": set(),
            "dist": dist,
        }
        config.pluginmanager.register(
            _FullRunCertificationPlugin(config), "increment-testmon-full-certification"
        )
    # Affected inputs are validated by the parser-backed launcher before collection.
    root = Path(config.rootpath)
    changed_path_names = json.loads(os.environ.get("INCREMENT_AFFECTED_CHANGED_PATHS", "[]"))
    changed_test_paths = [
        path
        for path in changed_path_names
        if isinstance(path, str) and path.startswith("tests/") and path.endswith(".py")
    ]
    dynamic_path_names = json.loads(os.environ.get("INCREMENT_AFFECTED_DYNAMIC_TEST_PATHS", "[]"))
    import_path_names = json.loads(os.environ.get("INCREMENT_AFFECTED_IMPORT_TEST_PATHS", "[]"))
    subprocess_path_names = json.loads(
        os.environ.get("INCREMENT_AFFECTED_SUBPROCESS_TEST_PATHS", "[]")
    )
    protected_paths = _testmon_protected_paths(
        root,
        changed_test_paths,
        dynamic_path_names,
        subprocess_path_names,
    )
    dynamic_paths = {Path(path).resolve() for path in dynamic_path_names}
    import_paths = {Path(path).resolve() for path in import_path_names}
    os.environ.pop("INCREMENT_AFFECTED_SELECTION_ROUTE", None)
    testmon_requested = bool(os.environ.get("INCREMENT_AFFECTED_USE_TESTMON"))
    primary = _activate_testmon(config, protected_paths) if testmon_requested else False
    if os.environ.get("INCREMENT_AFFECTED_EVIDENCE"):
        os.environ["INCREMENT_AFFECTED_SELECTION_ROUTE"] = "testmon" if primary else "import_graph"
    if primary:
        retained_paths = {path.resolve() for path in (root / "tests").rglob("test_*.py")}
    else:
        retained_paths = protected_paths | import_paths
    config.stash[_TESTMON_PRIMARY] = primary
    if not retained_paths:
        return
    from tach.pytest_plugin import tach_state_key

    state = config.stash.get(tach_state_key, None)
    if state is None:
        raise pytest.UsageError("Tach selection is unavailable for affected test files")
    cast(Any, state).handler = _ChangedTestTachHandler(state.handler, retained_paths)
    if dynamic_paths:
        config.stash[_DYNAMIC_CONSUMER_SELECTION] = {
            "paths": dynamic_paths,
            "collected": {},
            "controller_selected": {},
            "worker_collected": None,
            "worker_selected": None,
        }


class _FullRunCertificationPlugin:
    def __init__(self, config: pytest.Config) -> None:
        self.config = config

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        state = self.config.stash.get(_FULL_RUN_CERTIFICATION, None)
        if state is not None and (
            report.when == "call"
            or (report.when == "setup" and report.outcome in {"failed", "skipped"})
        ):
            state["reported"].add(report.nodeid)

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node: Any, ids: list[str]) -> None:
        state = self.config.stash.get(_FULL_RUN_CERTIFICATION, None)
        if state is not None:
            state["collected"].update(ids)

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node: Any, error: object) -> None:
        state = self.config.stash.get(_FULL_RUN_CERTIFICATION, None)
        worker_evidence = getattr(node, "workeroutput", {}).get("increment_testmon_full_evidence")
        if state is None or worker_evidence is None:
            return
        state["collected"].update(worker_evidence["collected"])
        state["safe"] = state["safe"] and worker_evidence["safe"]


def pytest_collection_finish(session: pytest.Session) -> None:
    state = session.config.stash.get(_FULL_RUN_CERTIFICATION, None)
    if state is None:
        return
    workeroutput = getattr(session.config, "workeroutput", None)
    workers = session.config.getoption("numprocesses", default=0)
    if workeroutput is None and workers not in (None, 0, "0"):
        return
    nodeids = {item.nodeid for item in session.items}
    state["collected"].update(nodeids)
    if workeroutput is not None:
        workeroutput["increment_testmon_full_evidence"] = {
            "safe": state["safe"],
            "collected": sorted(state["collected"]),
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


def _testmon_is_registered(config: pytest.Config) -> bool:
    manager = config.pluginmanager
    if manager.get_plugin("TestmonSelect") is not None:
        return True
    return manager.hasplugin("pytest-testmon") or manager.hasplugin("testmon.pytest_testmon")


def _restore_duration_order_after_testmon(
    config: pytest.Config,
    items: list[pytest.Item],
    markdown_order: dict[str, int] | None = None,
) -> None:
    if os.environ.get("INCREMENT_DISABLE_DURATION_ORDER") == "1" or not _testmon_is_registered(
        config
    ):
        return
    import conftest

    conftest._order_by_recorded_duration(
        items,
        Path(config.rootpath) / ".test_durations",
        conftest._duration_snapshot(config),
        markdown_order,
    )


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
    markdown_order = {
        item.nodeid: index
        for index, item in enumerate(items)
        if Path(str(item.location[0])).suffix == ".md"
    }
    yield
    if affected and not config.stash.get(_TESTMON_PRIMARY, False):
        import_paths = {
            Path(path).resolve()
            for path in json.loads(os.environ.get("INCREMENT_AFFECTED_IMPORT_TEST_PATHS", "[]"))
        }
        retained = {
            Path(path).resolve()
            for path in json.loads(os.environ.get("INCREMENT_AFFECTED_TEST_PATHS", "[]"))
        } | import_paths
        changed_paths = json.loads(os.environ.get("INCREMENT_AFFECTED_CHANGED_PATHS", "[]"))
        if any(isinstance(path, str) and path.endswith(".py") for path in changed_paths):
            retained.update(
                Path(__file__).resolve().parents[1] / "tests" / name
                for name in (
                    "test_refusal_gate.py",
                    "test_test_hygiene.py",
                    "test_impact_selection.py",
                    "test_full_suite_evidence.py",
                    "test_test_runner_cli.py",
                )
            )
        deselected = [item for item in items if item.path.resolve() not in retained]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [item for item in items if item.path.resolve() in retained]
    counts = config.stash.get(_AFFECTED_COLLECTION, None)
    if counts is not None:
        counts["selected"] = len(items)
        counts["deselected"] = counts["collected"] - len(items)
    selection = config.stash.get(_DYNAMIC_CONSUMER_SELECTION, None)
    if selection is not None:
        selected_nodes: dict[Path, set[str]] = {}
        for item in items:
            path = item.path.resolve()
            if path in dynamic_paths:
                selected_nodes.setdefault(path, set()).add(original_nodeids[id(item)])
        selection["controller_selected"] = selected_nodes
    _restore_duration_order_after_testmon(config, items, markdown_order)


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
    certification = session.config.stash.get(_FULL_RUN_CERTIFICATION, None)
    if certification is not None:
        if workeroutput is not None:
            workeroutput["increment_testmon_full_evidence"] = {
                "safe": certification["safe"],
                "collected": sorted(certification["collected"]),
            }
        else:
            Path(certification["path"]).write_text(
                json.dumps(
                    {
                        "safe": certification["safe"],
                        "collected": sorted(certification["collected"]),
                        "reported": sorted(certification["reported"]),
                        "dist": certification["dist"],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
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
