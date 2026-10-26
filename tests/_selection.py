from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.selection_map import is_stale, load, tests_changed_since

if TYPE_CHECKING:
    from collections.abc import Mapping


def _module_name(path: Path, root: Path) -> str:
    parts = path.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _lazy_tables(root: Path) -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    package = root / "increment"
    if not package.is_dir():
        return tables
    for path in package.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in tree.body:
            value = None
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS" for target in node.targets
            ):
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "_LAZY_IMPORTS":
                value = node.value
            if value is None:
                continue
            try:
                entries = ast.literal_eval(value)
            except (ValueError, TypeError):
                break
            if isinstance(entries, dict):
                tables[_module_name(path, root)] = {
                    name: target[0]
                    for name, target in entries.items()
                    if isinstance(name, str) and isinstance(target, tuple) and len(target) == 2 and isinstance(target[0], str)
                }
            break
    return tables


def _imports(
    path: Path, module: str, known: set[str], tables: dict[str, dict[str, str]]
) -> tuple[set[str], set[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeError):
        return set(), set()
    imports: set[str] = set()
    unresolved: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names if alias.name in known)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import(module, path, node)
            if base in known:
                imports.add(base)
            imports.update(f"{base}.{alias.name}" for alias in node.names if f"{base}.{alias.name}" in known)
            for alias in node.names:
                if base in tables:
                    target = tables[base].get(alias.name)
                    if target is None:
                        unresolved.add(alias.name)
                    else:
                        imports.add(target)
        elif isinstance(node, ast.Call):
            function = node.func
            if (isinstance(function, ast.Attribute) and function.attr == "import_module") or (
                isinstance(function, ast.Name) and function.id == "import_module"
            ):
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    name = node.args[0].value
                    if name == "increment" or name.startswith("increment."):
                        imports.add(name)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name) and node.value.value.id == "sys" and node.value.attr == "modules":
                if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                    name = node.slice.value
                    if name == "increment" or name.startswith("increment."):
                        imports.add(name)
    return imports, unresolved


def _resolve_import(module: str, path: Path, node: ast.ImportFrom) -> str:
    base = node.module or ""
    if node.level:
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        parts = package.split(".")
        if node.level > 1:
            parts = parts[: -(node.level - 1)]
        base = ".".join((*parts, *base.split("."))) if base else ".".join(parts)
    return base


class SelectionGraph:
    def __init__(self, root: Path, changed_modules: set[str], tier: str, reasons: dict[str, set[str]]) -> None:
        self.root = root
        self.changed_modules = changed_modules
        self.tier = tier
        self._reasons = reasons

    @classmethod
    def from_tree(
        cls,
        root: Path,
        changed_modules: set[str],
        *,
        tier: str,
        selection_map: Mapping[str, object] | None,
    ) -> SelectionGraph:
        root = root.resolve()
        reasons: dict[str, set[str]] = {}
        tests = sorted((root / "tests").rglob("test_*.py")) if (root / "tests").is_dir() else []
        names = {path: _module_name(path, root) for path in tests}
        tables = _lazy_tables(root)
        usable = selection_map is not None and selection_map.get("tier") == tier
        why_unusable = "missing"
        if usable:
            try:
                if is_stale(selection_map, root=root):
                    usable = False
                    why_unusable = "stale"
            except (KeyError, TypeError, ValueError):
                usable = False
                why_unusable = "invalid"
        if selection_map is not None and selection_map.get("tier") != tier:
            why_unusable = "tier"
        if usable and tests_changed_since(selection_map, root):
            # Newly introduced tests are covered by since_map below; modified
            # pre-map tests invalidate evidence because their behavior changed.
            reasons_for_changes = tests_changed_since(selection_map, root)
            for test in reasons_for_changes:
                reasons.setdefault(test, set()).add("since_map")
        elif not usable:
            for path in tests:
                reasons.setdefault(path.relative_to(root).as_posix(), set()).add(f"map_unusable:{why_unusable}")

        # A missing estimator package or manifest is an absent source, not a
        # reason to reject an otherwise usable map.
        manifest = root / "increment/estimators/manifest.json"
        if manifest.is_file():
            try:
                manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
                listed = set(manifest_data.get("modules", ())) if isinstance(manifest_data, dict) else set()
            except (OSError, ValueError, TypeError):
                listed = set()
            if changed_modules & listed:
                for test in (root / "tests/selection_required.txt").read_text().splitlines() if (root / "tests/selection_required.txt").exists() else []:
                    reasons.setdefault(test, set()).add("manifest")

        if changed_modules:
            required = root / "tests/selection_required.txt"
            if required.is_file():
                for test in required.read_text(encoding="utf-8").splitlines():
                    if test.strip():
                        reasons.setdefault(test.strip(), set()).add("required")

        if usable:
            modules = selection_map.get("modules", {})
            if isinstance(modules, dict):
                for module in changed_modules:
                    for test in modules.get(module, ()):
                        reasons.setdefault(str(test), set()).add("coverage_map")

        package_root = root / "increment"
        package_paths = {
            _module_name(path, root): path
            for path in package_root.rglob("*.py")
        } if package_root.is_dir() else {}
        helper_paths = {
            _module_name(path, root): path
            for path in (root / "tests").rglob("*.py")
        } if (root / "tests").is_dir() else {}
        all_paths = {**package_paths, **helper_paths}
        known = set(all_paths)
        adjacency: dict[str, set[str]] = {}
        lazy_edges: dict[str, list[tuple[str, str]]] = {}
        unresolved_by_module: dict[str, set[str]] = {}
        for module, path in all_paths.items():
            imports, unresolved = _imports(path, module, known, tables)
            adjacency[module] = imports
            unresolved_by_module[module] = unresolved
            if module in helper_paths:
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except (OSError, SyntaxError, UnicodeError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import) and any(alias.name == "tests.refusal_registry" for alias in node.names):
                        reasons.setdefault(path.relative_to(root).as_posix(), set()).add("required")
                    if isinstance(node, ast.ImportFrom) and node.module == "tests.refusal_registry":
                        reasons.setdefault(path.relative_to(root).as_posix(), set()).add("required")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "walk_packages":
                        if changed_modules:
                            reasons.setdefault(path.relative_to(root).as_posix(), set()).add("required")
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "__subclasses__":
                        if changed_modules:
                            reasons.setdefault(path.relative_to(root).as_posix(), set()).add("required")
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ImportFrom):
                        continue
                    base = _resolve_import(module, path, node)
                    if base in tables:
                        for alias in node.names:
                            target = tables[base].get(alias.name)
                            if target is not None:
                                lazy_edges.setdefault(module, []).append((alias.name, target))
                            else:
                                unresolved_by_module.setdefault(module, set()).add(alias.name)

        for test_path, module in names.items():
            visited: set[str] = set()
            pending = [module]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(adjacency.get(current, ()) - visited)
            if any(unresolved_by_module.get(imported) for imported in visited):
                reasons.setdefault(test_path.relative_to(root).as_posix(), set()).add("unresolved_lazy_name")
            for owner in visited:
                for symbol, target in lazy_edges.get(owner, ()):
                    target_seen = {target}
                    target_pending = [target]
                    while target_pending:
                        current = target_pending.pop()
                        for imported in adjacency.get(current, ()):
                            if imported not in target_seen:
                                target_seen.add(imported)
                                target_pending.append(imported)
                    if changed_modules.intersection(target_seen):
                        reasons.setdefault(test_path.relative_to(root).as_posix(), set()).add(f"lazy_export:{symbol}")
            for target in adjacency.get(module, ()):
                if target in changed_modules:
                    reasons.setdefault(test_path.relative_to(root).as_posix(), set()).add("dynamic_import")

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
    def __init__(self, tach_handler: object, graph: SelectionGraph) -> None:
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

    def write_report(self, path: Path) -> None:
        payload = {
            str(file.relative_to(self.graph.root)) if file.is_absolute() else file.as_posix(): {
                "kept": kept,
                "reasons": list(reasons),
                "tach_kept": self._tach_kept[file],
            }
            for file, (kept, reasons) in sorted(self.decisions.items(), key=lambda item: str(item[0]))
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
