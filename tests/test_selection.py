from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.selection_map import SelectionMap
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


def _map(
    root: Path, modules: dict[str, list[str]], *, base: str, tier: str = "fast"
) -> SelectionMap:
    payload: SelectionMap = {
        "generated_at": "2026-10-25T00:00:00+00:00",
        "base_sha": base,
        "tier": tier,
        "modules": modules,
    }
    path = root / "tests/selection_map.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


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


def test_handler_keeps_a_superset_of_tach_on_the_probe_modules(tmp_path: Path) -> None:
    test_files = {f"tests/test_probe_{index}.py": "" for index in range(7)}
    _tree(tmp_path, {**test_files, **{f"increment/probe_{i}.py": "" for i in range(7)}})
    tach_kept = {"tests/test_probe_0.py", "tests/test_probe_3.py"}
    tach = _Tach({(tmp_path / name).as_posix() for name in set(test_files) - tach_kept})

    for index in range(7):
        graph = SelectionGraph.from_tree(
            tmp_path, {f"increment.probe_{index}"}, tier="fast", selection_map=None
        )
        handler = AffectedHandler(tach, graph)
        kept = {name for name in test_files if not handler.should_remove_items(tmp_path / name)}
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
