from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol

from scripts.selection_map import SelectionMap, is_stale, tests_changed_since


class TachHandler(Protocol):
    def should_remove_items(self, *, file_path: Path) -> bool: ...


def _module_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


_FACTS_FILE = Path(".pytest_cache") / "selection-facts.json"


def _extract_facts(source: str) -> dict[str, Any]:
    """What selection needs from one file, independent of every other file."""
    tree = ast.parse(source)
    imports: list[list[Any]] = []
    dynamic: list[str] = []
    walks_package = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.append(["import", None, 0, [alias.name for alias in node.names]])
        elif isinstance(node, ast.ImportFrom):
            imports.append(["from", node.module, node.level, [alias.name for alias in node.names]])
        elif isinstance(node, ast.Call):
            function = node.func
            if (
                (isinstance(function, ast.Attribute) and function.attr == "import_module")
                or (isinstance(function, ast.Name) and function.id == "import_module")
            ) and (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                dynamic.append(node.args[0].value)
            if isinstance(function, ast.Attribute) and function.attr in {
                "walk_packages",
                "__subclasses__",
            }:
                walks_package = True
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sys"
            and node.value.attr == "modules"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            dynamic.append(node.slice.value)
    lazy_table: dict[str, str] | None = None
    top_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            top_names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.Assign):
            top_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_names.add(node.target.id)
        is_lazy = (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_LAZY_IMPORTS" for t in node.targets)
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_LAZY_IMPORTS"
        )
        if is_lazy and lazy_table is None and node.value is not None:
            try:
                entries = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                entries = None
            lazy_table = (
                {
                    name: target[0]
                    for name, target in entries.items()
                    if isinstance(name, str)
                    and isinstance(target, tuple)
                    and len(target) == 2
                    and isinstance(target[0], str)
                }
                if isinstance(entries, dict)
                else {}
            )
    return {
        "imports": imports,
        "dynamic": dynamic,
        "walks_package": walks_package,
        "lazy_table": lazy_table,
        "top_names": sorted(top_names),
    }


class _Facts:
    """Per-file facts keyed by content digest, persisted between runs."""

    def __init__(self, root: Path) -> None:
        self._path = root / _FACTS_FILE
        try:
            loaded = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        self._entries: dict[str, Any] = loaded if isinstance(loaded, dict) else {}
        self._dirty = False

    def get(self, path: Path) -> dict[str, Any] | None:
        try:
            source = path.read_bytes()
        except OSError:
            return None
        digest = hashlib.sha256(source).hexdigest()
        cached = self._entries.get(digest)
        if cached is not None:
            return cached
        try:
            facts = _extract_facts(source.decode("utf-8"))
        except (SyntaxError, UnicodeError, ValueError):
            return None
        self._entries[digest] = facts
        self._dirty = True
        return facts

    def save(self) -> None:
        if not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(f"{self._path.name}.{os.getpid()}")
            temporary.write_text(json.dumps(self._entries), encoding="utf-8")
            os.replace(temporary, self._path)
        except OSError:
            return


def _lazy_tables(root: Path, facts: _Facts | None = None) -> dict[str, dict[str, str]]:
    facts = facts or _Facts(root)
    tables: dict[str, dict[str, str]] = {}
    package = root / "increment"
    if not package.is_dir():
        return tables
    for path in package.rglob("*.py"):
        found = facts.get(path)
        if found is not None and found["lazy_table"] is not None:
            tables[_module_name(path, root)] = found["lazy_table"]
    return tables


def _imports(
    facts: dict[str, Any] | None,
    path: Path,
    module: str,
    known: set[str],
    tables: dict[str, dict[str, str]],
    eager: dict[str, set[str]] | None = None,
) -> tuple[set[str], set[str], set[tuple[str, str]]]:
    imports: set[str] = set()
    supplemental: set[tuple[str, str]] = set()
    unresolved: set[str] = set()
    if facts is None:
        return imports, unresolved, supplemental
    for kind, name, level, aliases in facts["imports"]:
        if kind == "import":
            imports.update(alias for alias in aliases if alias in known)
            continue
        base = _resolve_from(module, path, name, level)
        if base in known:
            imports.add(base)
        imports.update(f"{base}.{alias}" for alias in aliases if f"{base}.{alias}" in known)
        if base in tables:
            for alias in aliases:
                target = tables[base].get(alias)
                if target is None:
                    # Names bound when the module runs, and its submodules,
                    # resolve through the import graph, not the lazy table.
                    if (
                        alias != "*"
                        and alias not in (eager or {}).get(base, set())
                        and f"{base}.{alias}" not in known
                    ):
                        unresolved.add(alias)
                else:
                    imports.add(target)
                    supplemental.add((target, f"lazy_export:{alias}"))
    for name in facts["dynamic"]:
        if name == "increment" or name.startswith("increment."):
            imports.add(name)
            supplemental.add((name, "dynamic_import"))
    return imports, unresolved, supplemental


def _resolve_from(module: str, path: Path, name: str | None, level: int) -> str:
    base = name or ""
    if level:
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        parts = package.split(".")
        if level > 1:
            parts = parts[: -(level - 1)]
        base = ".".join((*parts, *base.split("."))) if base else ".".join(parts)
    return base


def _resolve_import(module: str, path: Path, node: ast.ImportFrom) -> str:
    base = node.module or ""
    if node.level:
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        parts = package.split(".")
        if node.level > 1:
            parts = parts[: -(node.level - 1)]
        base = ".".join((*parts, *base.split("."))) if base else ".".join(parts)
    return base


def _reachable(start: str, adjacency: dict[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    pending = [start]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency.get(current, set()) - visited)
    return visited


class SelectionGraph:
    def __init__(
        self, root: Path, changed_modules: set[str], tier: str, reasons: dict[str, set[str]]
    ) -> None:
        self.root = root
        self.changed_modules = changed_modules
        self.tier = tier
        self._reasons = reasons

    @classmethod
    def from_tree(  # noqa: C901, PLR0915
        cls,
        root: Path,
        changed_modules: set[str],
        *,
        tier: str,
        selection_map: SelectionMap | None,
        changed_paths: set[str] | None = None,
    ) -> SelectionGraph:
        root = root.resolve()
        reasons: dict[str, set[str]] = {}
        tests = sorted((root / "tests").rglob("test_*.py")) if (root / "tests").is_dir() else []
        names = {path: _module_name(path, root) for path in tests}
        facts = _Facts(root)
        tables = _lazy_tables(root, facts)
        usable = selection_map is not None and selection_map["tier"] == tier
        why_unusable = "missing"
        if selection_map is not None and selection_map["tier"] == tier:
            try:
                if is_stale(selection_map, root=root):
                    usable = False
                    why_unusable = "stale"
                else:
                    usable = True
            except (KeyError, TypeError, ValueError):
                usable = False
                why_unusable = "invalid"
        if selection_map is not None and selection_map["tier"] != tier:
            usable = False
            why_unusable = "tier"
        if usable and selection_map is not None:
            for test in tests_changed_since(selection_map, root):
                reasons.setdefault(test, set()).add("since_map")
        elif not usable:
            for path in tests:
                reasons.setdefault(path.relative_to(root).as_posix(), set()).add(
                    f"map_unusable:{why_unusable}"
                )

        # A missing, malformed, or unreadable estimator manifest is an absent
        # source, not a reason to reject an otherwise usable map.
        manifest = root / "increment/estimators/manifest.json"
        if manifest.is_file():
            try:
                manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
                if isinstance(manifest_data, list) and all(
                    isinstance(record, dict)
                    and isinstance(record.get("module"), str)
                    and record["module"]
                    for record in manifest_data
                ):
                    listed = {record["module"] for record in manifest_data}
                else:
                    listed = set()
            except (OSError, ValueError, TypeError):
                listed = set()
            if changed_modules & listed:
                for test in (
                    (root / "tests/selection_required.txt").read_text().splitlines()
                    if (root / "tests/selection_required.txt").exists()
                    else []
                ):
                    reasons.setdefault(test, set()).add("manifest")

        if changed_paths if changed_paths is not None else changed_modules:
            required = root / "tests/selection_required.txt"
            if required.is_file():
                for test in required.read_text(encoding="utf-8").splitlines():
                    if test.strip():
                        reasons.setdefault(test.strip(), set()).add("required")

        if usable and selection_map is not None:
            for module in changed_modules:
                for test in selection_map["modules"].get(module, []):
                    reasons.setdefault(test, set()).add("coverage_map")

        package_root = root / "increment"
        package_paths = (
            {_module_name(path, root): path for path in package_root.rglob("*.py")}
            if package_root.is_dir()
            else {}
        )
        helper_paths = (
            {_module_name(path, root): path for path in (root / "tests").rglob("*.py")}
            if (root / "tests").is_dir()
            else {}
        )
        all_paths = {**package_paths, **helper_paths}
        known = set(all_paths)
        adjacency: dict[str, set[str]] = {}
        supplemental_edges: dict[str, set[tuple[str, str]]] = {}
        unresolved_by_module: dict[str, set[str]] = {}
        file_facts = {module: facts.get(path) for module, path in all_paths.items()}
        eager = {
            module: set(found["top_names"])
            for module in tables
            if (found := file_facts.get(module)) is not None
        }
        for module, path in all_paths.items():
            found = file_facts[module]
            imports, unresolved, supplemental = _imports(found, path, module, known, tables, eager)
            adjacency[module] = imports
            supplemental_edges[module] = supplemental
            unresolved_by_module[module] = unresolved
            if module in helper_paths and found is not None:
                relative = path.relative_to(root).as_posix()
                if any(
                    name == "tests.refusal_registry"
                    for kind, name, _level, aliases in found["imports"]
                    for name in ([name] if kind == "from" else aliases)
                ):
                    reasons.setdefault(relative, set()).add("required")
                if found["walks_package"] and changed_modules:
                    reasons.setdefault(relative, set()).add("required")
        facts.save()

        for test_path, module in names.items():
            visited = _reachable(module, adjacency)
            test_file = test_path.relative_to(root).as_posix()
            if any(unresolved_by_module.get(imported) for imported in visited):
                reasons.setdefault(test_file, set()).add("unresolved_lazy_name")
            for owner in visited:
                for target, reason in supplemental_edges.get(owner, ()):
                    if changed_modules.intersection(_reachable(target, adjacency)):
                        reasons.setdefault(test_file, set()).add(reason)

        return cls(root, changed_modules, tier, reasons)

    def keeps(self, file_path: str | Path) -> tuple[str, ...]:
        path = Path(file_path)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(self.root)
            except ValueError:
                return ()
        return tuple(sorted(self._reasons.get(path.as_posix(), ())))


class AffectedHandler:
    def __init__(self, tach_handler: TachHandler, graph: SelectionGraph) -> None:
        self._tach_handler = tach_handler
        self.graph = graph
        self._tach_kept: dict[Path, bool] = {}
        self.decisions: dict[Path, tuple[bool, tuple[str, ...]]] = {}

    def __getattr__(self, name: str) -> object:
        return getattr(self._tach_handler, name)

    def should_remove_items(self, file_path: Path) -> bool:
        resolved = file_path.resolve()
        if resolved not in self.decisions:
            tach_remove = self._tach_handler.should_remove_items(file_path=file_path)
            self._tach_kept[resolved] = not tach_remove
            try:
                relative = resolved.relative_to(self.graph.root)
            except ValueError:
                relative = file_path
            reasons = self.graph.keeps(relative)
            kept = not tach_remove or bool(reasons)
            self.decisions[resolved] = (kept, reasons)
        kept, _ = self.decisions[resolved]
        return not kept

    def write_report(self, path: Path, *, metadata: dict[str, object] | None = None) -> None:
        payload: dict[str, object] = {
            str(file.relative_to(self.graph.root)) if file.is_absolute() else file.as_posix(): {
                "kept": kept,
                "reasons": list(reasons),
                "tach_kept": self._tach_kept[file],
            }
            for file, (kept, reasons) in sorted(
                self.decisions.items(), key=lambda item: str(item[0])
            )
        }
        if metadata is not None:
            payload["__metadata__"] = {**metadata, "decision_files": sorted(payload)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selection_graph(root: Path, base: str, *, tier: str, head: str | None = None) -> SelectionGraph:
    """Build the graph for the paths changed since ``base``."""
    from scripts.affected import _changed_paths
    from scripts.selection_map import load

    changed_paths = set(_changed_paths(base, root, head=head))
    changed_modules = set()
    for name in changed_paths:
        path = Path(name)
        if name.startswith("increment/") and path.suffix == ".py":
            parts = path.with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            changed_modules.add(".".join(parts))
    return SelectionGraph.from_tree(
        root,
        changed_modules,
        tier=tier,
        selection_map=load(root / "tests/selection_map.json"),
        changed_paths=changed_paths,
    )


def kept_test_files(
    root: Path, base: str, *, tier: str, head: str | None = None
) -> tuple[str, ...] | None:
    """Decide every test file the way the pytest handler does, before collection.

    Returns ``None`` when tach has no project configuration, so the caller
    collects the whole test tree instead.
    """
    from tach.extension import TachPytestPluginHandler
    from tach.filesystem.git_ops import get_changed_files
    from tach.parsing import parse_project_config

    project_config = parse_project_config(root=root)
    if project_config is None:
        return None
    changed_files = get_changed_files(project_root=root, head=head, base=base)
    handler = AffectedHandler(
        TachPytestPluginHandler(
            project_root=root,
            project_config=project_config,
            changed_files=changed_files,
            all_affected_modules={path.resolve() for path in changed_files},
        ),
        selection_graph(root, base, tier=tier, head=head),
    )
    test_root = root / "tests"
    candidates = sorted({*test_root.rglob("test_*.py"), *test_root.rglob("*_test.py")})
    return tuple(
        path.relative_to(root).as_posix()
        for path in candidates
        if not handler.should_remove_items(path)
    )
