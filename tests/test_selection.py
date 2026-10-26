from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.selection_map import SelectionMap, is_stale
from tests._selection import AffectedHandler, SelectionGraph

pytestmark = pytest.mark.integration


class _Tach:
    def __init__(self, removed: set[str] | None = None) -> None:
        self.removed = removed or set()
        self.calls: list[str] = []

    def should_remove_items(self, *, file_path: Path) -> bool:
        self.calls.append(file_path.as_posix())
        return file_path.as_posix() in self.removed


def _tree(root: Path, files: dict[str, str]) -> None:
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)


def _commit(root: Path) -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, check=True, text=True, capture_output=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "tests@example.com")
    git("config", "user.name", "Tests")
    git("add", ".")
    git("commit", "-qm", "base")
    return git("rev-parse", "HEAD")


@pytest.fixture
def scratch_worktree(tmp_path: Path) -> Iterator[Path]:
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "scratch-worktree"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(root), "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    try:
        yield root
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(root)],
            cwd=repo,
            check=True,
            text=True,
            capture_output=True,
        )


def _scratch_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _scratch_commit(root: Path, message: str) -> None:
    _scratch_git(root, "add", "-A")
    _scratch_git(
        root,
        "-c",
        "user.email=tests@example.com",
        "-c",
        "user.name=Tests",
        "commit",
        "--no-verify",
        "-qm",
        message,
    )


def _scratch_map(root: Path) -> str:
    base = _scratch_git(root, "rev-parse", "HEAD")
    _tree(
        root,
        {
            "tests/selection_map.json": json.dumps(
                {
                    "generated_at": datetime.now(UTC).isoformat(),
                    "base_sha": base,
                    "tier": "fast",
                    "modules": {},
                }
            )
        },
    )
    _scratch_commit(root, "install usable selection map")
    return base


def _scratch_change(root: Path, *paths: str) -> None:
    for name in paths:
        path = root / name
        path.write_text(path.read_text() + "\n# scratch selection probe\n")
    _scratch_commit(root, "change probe modules")


def _run_scratch_tach(root: Path, base: str, *test_paths: str) -> dict[str, object]:
    report = root / "affected-report.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "--tach-base",
            base,
            "--affected-report",
            str(report),
            *test_paths,
        ],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(report.read_text())


def _map(
    root: Path, modules: dict[str, list[str]], *, base: str, tier: str = "fast"
) -> SelectionMap:
    payload: SelectionMap = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_sha": base,
        "tier": tier,
        "modules": modules,
    }
    path = root / "tests/selection_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def test_map_helper_builds_a_fresh_map(tmp_path: Path) -> None:
    _tree(tmp_path, {"increment/x.py": "", "tests/test_a.py": ""})
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {}, base=base)

    assert not is_stale(mapping, root=tmp_path)


def test_handler_keeps_whatever_tach_keeps(tmp_path: Path) -> None:
    tach = _Tach()
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=None)
    handler = AffectedHandler(tach, graph)

    assert handler.should_remove_items(tmp_path / "tests/test_a.py") is False
    assert handler.should_remove_items(tmp_path / "tests/test_a.py") is False
    assert tach.calls == [(tmp_path / "tests/test_a.py").as_posix()]


def test_lazy_export_name_keeps_owner_consumer(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/estimation/__init__.py": "_LAZY_IMPORTS = {'Method': ('increment.engine', 'Method')}\n",
            "increment/engine.py": "class Method: pass\n",
            "tests/test_m.py": "from increment.estimation import Method\n",
        },
    )
    graph = SelectionGraph.from_tree(
        tmp_path, {"increment.engine"}, tier="fast", selection_map=None
    )

    assert "lazy_export:Method" in graph.keeps("tests/test_m.py")


def test_unresolvable_lazy_export_keeps_importing_test(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/__init__.py": "_LAZY_IMPORTS = {'Known': ('increment.impl', 'Known')}\n",
            "increment/impl.py": "class Known: pass\n",
            "tests/test_api.py": "from increment import Unknown\n",
        },
    )
    graph = SelectionGraph.from_tree(tmp_path, {"increment.impl"}, tier="fast", selection_map=None)

    assert "unresolved_lazy_name" in graph.keeps("tests/test_api.py")


def test_unusable_map_keeps_the_whole_tier_even_without_estimators(tmp_path: Path) -> None:
    _tree(tmp_path, {"tests/test_a.py": "", "tests/test_b.py": "", "increment/x.py": ""})
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=None)

    assert graph.keeps("tests/test_a.py") == ("map_unusable:missing",)
    assert graph.keeps("tests/test_b.py") == ("map_unusable:missing",)


def test_coverage_map_keeps_observed_consumer(tmp_path: Path) -> None:
    _tree(tmp_path, {"tests/test_dyn.py": "", "increment/leaf.py": ""})
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {"increment.leaf": ["tests/test_dyn.py"]}, base=base)
    graph = SelectionGraph.from_tree(
        tmp_path, {"increment.leaf"}, tier="fast", selection_map=mapping
    )

    assert graph.keeps("tests/test_dyn.py") == ("coverage_map",)


def test_missing_lazy_estimators_sources_are_ignored(tmp_path: Path) -> None:
    _tree(tmp_path, {"tests/test_a.py": "", "increment/x.py": ""})
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {"increment.x": ["tests/test_a.py"]}, base=base)
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=mapping)

    assert graph.keeps("tests/test_a.py") == ("coverage_map",)


def test_required_files_are_kept_for_package_change_with_valid_map(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/x.py": "VALUE = 1\n",
            "tests/test_a.py": "",
            "tests/selection_required.txt": "tests/test_a.py\n",
        },
    )
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {}, base=base)
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=mapping)

    assert graph.keeps("tests/test_a.py") == ("required",)


def test_test_added_after_map_is_kept(tmp_path: Path) -> None:
    _tree(tmp_path, {"increment/x.py": "", "tests/test_a.py": ""})
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {}, base=base)
    (tmp_path / "tests/test_new.py").write_text("def test_new(): pass\n")
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=mapping)

    assert graph.keeps("tests/test_new.py") == ("since_map",)


def test_map_for_another_tier_keeps_every_test(tmp_path: Path) -> None:
    _tree(tmp_path, {"increment/x.py": "", "tests/test_a.py": "", "tests/test_b.py": ""})
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {"increment.x": ["tests/test_a.py"]}, base=base, tier="slow")
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=mapping)

    assert graph.keeps("tests/test_a.py") == ("map_unusable:tier",)
    assert graph.keeps("tests/test_b.py") == ("map_unusable:tier",)


def test_handler_records_decision_reasons(tmp_path: Path) -> None:
    _tree(tmp_path, {"tests/test_a.py": "", "increment/x.py": ""})
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=None)
    handler = AffectedHandler(_Tach(), graph)

    handler.should_remove_items(tmp_path / "tests/test_a.py")
    assert handler.decisions[Path(tmp_path / "tests/test_a.py")] == (
        True,
        ("map_unusable:missing",),
    )
    report = tmp_path / "report.json"
    handler.write_report(report)
    assert json.loads(report.read_text())["tests/test_a.py"] == {
        "kept": True,
        "reasons": ["map_unusable:missing"],
        "tach_kept": True,
    }


def test_literal_dynamic_import_keeps_consumer(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/__init__.py": "",
            "increment/x.py": "VALUE = 1\n",
            "tests/test_dynamic.py": "import importlib\nmodule = importlib.import_module('increment.x')\n",
        },
    )
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=None)

    assert "dynamic_import" in graph.keeps("tests/test_dynamic.py")


def test_dynamic_import_in_transitive_helper_keeps_consumer(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/__init__.py": "",
            "increment/leaf.py": "VALUE = 1\n",
            "tests/helper.py": (
                "import importlib\nmodule = importlib.import_module('increment.leaf')\n"
            ),
            "tests/test_a.py": "from tests.helper import module\n",
        },
    )
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {}, base=base)
    (tmp_path / "increment/leaf.py").write_text("VALUE = 2\n")
    graph = SelectionGraph.from_tree(
        tmp_path, {"increment.leaf"}, tier="fast", selection_map=mapping
    )

    assert "dynamic_import" in graph.keeps("tests/test_a.py")


def test_package_lazy_import_in_transitive_closure_keeps_consumer(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/__init__.py": "",
            "increment/dispatch.py": ("_LAZY_IMPORTS = {'Thing': ('increment.leaf', 'Thing')}\n"),
            "increment/facade.py": "from increment.dispatch import Thing\n",
            "increment/leaf.py": "class Thing: pass\n",
            "tests/test_a.py": "import increment.facade\n",
        },
    )
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {}, base=base)
    (tmp_path / "increment/leaf.py").write_text("class Thing: pass\n# changed\n")
    graph = SelectionGraph.from_tree(
        tmp_path, {"increment.leaf"}, tier="fast", selection_map=mapping
    )

    assert "lazy_export:Thing" in graph.keeps("tests/test_a.py")


def test_package_walk_keeps_test_on_the_required_floor(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/__init__.py": "__path__ = []\n",
            "increment/x.py": "",
            "tests/test_discovery.py": "import pkgutil\nimport increment\npkgutil.walk_packages(increment.__path__)\n",
        },
    )
    graph = SelectionGraph.from_tree(tmp_path, {"increment.x"}, tier="fast", selection_map=None)

    assert "required" in graph.keeps("tests/test_discovery.py")


def test_manifest_module_change_keeps_required_selection_floor(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/estimators/manifest.json": '{"modules":["increment.impl"]}',
            "increment/impl.py": "",
            "tests/test_registry.py": "",
            "tests/selection_required.txt": "tests/test_registry.py\n",
        },
    )
    graph = SelectionGraph.from_tree(tmp_path, {"increment.impl"}, tier="fast", selection_map=None)

    assert "manifest" in graph.keeps("tests/test_registry.py")


@pytest.mark.slow
def test_handler_keeps_a_superset_of_tach_on_the_probe_modules(
    scratch_worktree: Path,
) -> None:
    probe_modules = (
        "increment/estimation/engine.py",
        "increment/estimation/_adjust/iptw.py",
        "increment/estimation/cuped/multicovariate.py",
        "increment/power/_binomial.py",
        "increment/query/_cuped_qr.py",
        "increment/_lift_options.py",
        "increment/semantics/models.py",
    )
    test_files = (
        "tests/estimation/test_engine.py",
        "tests/estimation/test_adjust.py",
        "tests/estimation/cuped/test_prepare.py",
        "tests/power/test_sequential_power.py",
        "tests/test_warehouse_cuped_qr_edges.py",
        "tests/test_imports.py",
        "tests/semantics/test_models.py",
    )
    base = _scratch_map(scratch_worktree)
    _scratch_change(scratch_worktree, *probe_modules)
    decisions = _run_scratch_tach(scratch_worktree, base, *test_files)

    tach_kept = {
        path
        for path, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("tach_kept") is True
    }
    kept = {
        path
        for path, decision in decisions.items()
        if isinstance(decision, dict) and decision.get("kept") is True
    }
    assert tach_kept
    assert tach_kept <= kept


def test_dispatch_table_change_invalidates_map_and_keeps_whole_tier(tmp_path: Path) -> None:
    _tree(
        tmp_path,
        {
            "increment/__init__.py": "_LAZY_IMPORTS = {'A': ('increment.a', 'A')}\n",
            "increment/a.py": "",
            "tests/test_a.py": "",
            "tests/test_b.py": "",
        },
    )
    base = _commit(tmp_path)
    mapping = _map(tmp_path, {}, base=base)
    (tmp_path / "increment/__init__.py").write_text("_LAZY_IMPORTS = {'A': ('increment.b', 'A')}\n")

    graph = SelectionGraph.from_tree(tmp_path, {"increment.a"}, tier="fast", selection_map=mapping)

    assert graph.keeps("tests/test_a.py") == ("map_unusable:stale",)
    assert graph.keeps("tests/test_b.py") == ("map_unusable:stale",)


def _real_tach_decision(root: Path, base: str, test_path: str) -> dict[str, object]:
    decisions = _run_scratch_tach(root, base, test_path)
    decision = decisions[test_path]
    assert isinstance(decision, dict)
    return {str(key): value for key, value in decision.items()}


@pytest.mark.slow
def test_real_tach_function_local_import_is_an_edge(scratch_worktree: Path) -> None:
    base = _scratch_map(scratch_worktree)
    _scratch_change(scratch_worktree, "increment/estimation/adjust.py")

    decision = _real_tach_decision(scratch_worktree, base, "tests/estimation/test_engine.py")

    assert decision["tach_kept"] is True
    assert decision["kept"] is True


@pytest.mark.slow
def test_real_tach_registration_consumer_is_kept(scratch_worktree: Path) -> None:
    base = _scratch_map(scratch_worktree)
    _scratch_change(scratch_worktree, "increment/estimation/_adjust/iptw.py")

    decision = _real_tach_decision(scratch_worktree, base, "tests/test_refusal_uniqueness.py")

    assert decision["tach_kept"] is True
    assert decision["kept"] is True


@pytest.mark.slow
def test_redirected_function_local_import_is_seen_on_next_selection(
    scratch_worktree: Path,
) -> None:
    _tree(
        scratch_worktree,
        {
            "tests/test_redirected_function_local.py": (
                "import importlib\n\n"
                "def test_redirected_import():\n"
                "    importlib.import_module("
                "'increment.estimation._adjust.iptw')\n"
            )
        },
    )
    _scratch_commit(scratch_worktree, "add redirected-import consumer")
    base = _scratch_map(scratch_worktree)
    _scratch_change(scratch_worktree, "increment/estimation/_adjust/iptw.py")

    decision = _real_tach_decision(
        scratch_worktree, base, "tests/test_redirected_function_local.py"
    )

    assert decision["kept"] is True
    reasons = decision["reasons"]
    assert isinstance(reasons, list)
    assert "dynamic_import" in reasons
