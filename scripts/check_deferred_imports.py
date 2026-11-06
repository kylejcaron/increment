#!/usr/bin/env python3
"""Ratchet deferred imports and cycles; omit lazy facade and package re-export edges."""

import argparse
import ast
import importlib.util
from pathlib import Path


def _module(path: Path, root: Path) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((root.name, *parts))


def _module_exists(name: str, root: Path) -> bool:
    parts = name.split(".")[1:]
    base = root.joinpath(*parts)
    return base.with_suffix(".py").is_file() or (base / "__init__.py").is_file()


def _targets(
    node: ast.AST, module: str, package_context: str, package: str, root: Path
) -> set[str]:
    targets: set[str] = set()
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == package or alias.name.startswith(package + "."):
                targets.add(alias.name if _module_exists(alias.name, root) else alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            prefix = package_context.split(".")
            prefix = prefix[: len(prefix) - node.level + 1]
            base = ".".join(prefix)
            if node.module:
                base = f"{base}.{node.module}" if base else node.module
        else:
            base = node.module or ""
        if base == package or base.startswith(package + "."):
            targets.add(base)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                if _module_exists(candidate, root):
                    targets.add(candidate)
    return targets


class _BoundNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(alias.asname or alias.name for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return


class _Imports(ast.NodeVisitor):
    def __init__(self, module: str, package_context: str, package: str, root: Path) -> None:
        self.module = module
        self.package_context = package_context
        self.package = package
        self.root = root
        self.deferred = False
        self.type_checking = False
        self.alias_scopes: list[dict[str, str | None]] = [{}]
        self.edges: set[str] = set()
        self.deferred_edges: set[str] = set()

    def _bind(self, name: str, target: str | None) -> None:
        self.alias_scopes[-1][name] = target

    def _lookup(self, name: str) -> str | None:
        for scope in reversed(self.alias_scopes):
            if name in scope:
                return scope[name]
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name.split(".")[0]
            target = alias.name if alias.name in {"typing", "importlib"} else None
            self._bind(bound, target)
        self._record(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            bound = alias.asname or alias.name
            target = None
            if node.module == "typing" and alias.name == "TYPE_CHECKING":
                target = "typing.TYPE_CHECKING"
            elif node.module == "importlib" and alias.name == "import_module":
                target = "importlib.import_module"
            self._bind(bound, target)
        self._record(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self._bind(node.id, None)

    def _record(self, node: ast.AST) -> None:
        targets = _targets(node, self.module, self.package_context, self.package, self.root)
        self._record_targets(targets)

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        is_import_module = (
            isinstance(function, ast.Name)
            and self._lookup(function.id) == "importlib.import_module"
        ) or (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and self._lookup(function.value.id) == "importlib"
        )
        name_node = (
            node.args[0]
            if node.args
            else next((keyword.value for keyword in node.keywords if keyword.arg == "name"), None)
        )
        package_node = (
            node.args[1]
            if len(node.args) > 1
            else next(
                (keyword.value for keyword in node.keywords if keyword.arg == "package"), None
            )
        )
        if (
            is_import_module
            and isinstance(name_node, ast.Constant)
            and isinstance(name_node.value, str)
        ):
            name = name_node.value
            package_arg = (
                package_node.value
                if isinstance(package_node, ast.Constant) and isinstance(package_node.value, str)
                else None
            )
            try:
                target = (
                    importlib.util.resolve_name(name, package_arg) if name.startswith(".") else name
                )
            except (ImportError, ValueError):
                target = ""
            if target == self.package or target.startswith(self.package + "."):
                self._record_targets({target})
        self.generic_visit(node)

    def _record_targets(self, targets: set[str]) -> None:
        if self.type_checking:
            return
        expanded = set(targets)
        for target in targets:
            parts = target.split(".")
            while len(parts) > 1:
                parts.pop()
                parent = ".".join(parts)
                if parent == self.package:
                    break
                expanded.add(parent)
        self.edges.update(expanded)
        if self.deferred:
            self.deferred_edges.update(expanded)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for expression in node.decorator_list:
            self.visit(expression)
        for expression in (*node.args.defaults, *node.args.kw_defaults):
            if expression is not None:
                self.visit(expression)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            if argument.annotation:
                self.visit(argument.annotation)
        if node.args.vararg and node.args.vararg.annotation:
            self.visit(node.args.vararg.annotation)
        if node.args.kwarg and node.args.kwarg.annotation:
            self.visit(node.args.kwarg.annotation)
        if node.returns:
            self.visit(node.returns)

        names = _BoundNames()
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            names.names.add(argument.arg)
        if node.args.vararg:
            names.names.add(node.args.vararg.arg)
        if node.args.kwarg:
            names.names.add(node.args.kwarg.arg)
        for statement in node.body:
            names.visit(statement)
        inherited: dict[str, str | None] = {}
        for scope in self.alias_scopes:
            inherited.update(scope)
        for name in names.names:
            inherited[name] = None
        self.alias_scopes.append(inherited)
        previous = self.deferred
        self.deferred = True
        for item in node.body:
            self.visit(item)
        self.deferred = previous
        self.alias_scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for expression in (*node.decorator_list, *node.bases):
            self.visit(expression)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self.alias_scopes.append(dict(self.alias_scopes[-1]))
        for item in node.body:
            self.visit(item)
        self.alias_scopes.pop()

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        is_tc = (
            isinstance(test, ast.Name) and self._lookup(test.id) == "typing.TYPE_CHECKING"
        ) or (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and self._lookup(test.value.id) == "typing"
        )
        previous = self.type_checking
        if is_tc:
            self.type_checking = True
            for item in node.body:
                self.visit(item)
            self.type_checking = previous
            for item in node.orelse:
                self.visit(item)
            return
        self.generic_visit(node)


def _graphs(root: Path) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    root = root.resolve()
    package = root.name
    graph: dict[str, set[str]] = {}
    deferred: set[tuple[str, str]] = set()
    for path in sorted(root.rglob("*.py")):
        module = _module(path, root)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        package_context = module if path.name == "__init__.py" else module.rpartition(".")[0]
        visitor = _Imports(module, package_context, package, root)
        visitor.visit(tree)
        graph[module] = visitor.edges
        deferred.update((module, target) for target in visitor.deferred_edges)
    return graph, deferred


def runtime_graph(root: Path) -> dict[str, set[str]]:
    """Return internal module imports executed at runtime, including deferred imports."""
    return _graphs(Path(root))[0]


def deferred_edges(root: Path) -> set[tuple[str, str]]:
    """Return internal imports located inside functions or methods."""
    return _graphs(Path(root))[1]


def cycles(graph: dict[str, set[str]]) -> list[frozenset[str]]:
    """Return sorted strongly connected components containing runtime cycles."""
    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[frozenset[str]] = []

    def connect(node: str) -> None:
        nonlocal index
        indices[node] = lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, set()):
            if target.startswith(node + "."):
                continue
            if target not in indices:
                connect(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])
        if lowlinks[node] == indices[node]:
            component: set[str] = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            if len(component) > 1:
                found.append(frozenset(component))

    for node in sorted(graph):
        if node not in indices:
            connect(node)
    return sorted(found, key=lambda component: tuple(sorted(component)))


def _read_baseline(path: Path) -> tuple[set[tuple[str, str]], set[frozenset[str]]]:
    edges: set[tuple[str, str]] = set()
    components: list[set[str]] = []
    section = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            if section == "cycle" and components and components[-1]:
                components.append(set())
            continue
        if line.startswith("#"):
            section = line[1:].strip()
            continue
        if section == "deferred":
            source, target = (part.strip() for part in line.split("->", 1))
            edges.add((source, target))
        elif section == "cycle":
            if not components or not components[-1]:
                components.append(set())
            components[-1].add(line)
    return edges, {frozenset(component) for component in components if component}


def _baseline_text(edges: set[tuple[str, str]], components: list[frozenset[str]]) -> str:
    lines = [
        "# deferred",
        *(f"{source} -> {target}" for source, target in sorted(edges)),
        "",
        "# cycle",
    ]
    for component in components:
        lines.extend(sorted(component))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("increment"))
    parser.add_argument(
        "--baseline", type=Path, default=Path("tests/deferred_imports_baseline.txt")
    )
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    graph, edges = _graphs(args.root)
    components = cycles(graph)
    if args.update:
        args.baseline.write_text(_baseline_text(edges, components), encoding="utf-8")
        return 0
    known_edges, known_components = _read_baseline(args.baseline)
    violations = [
        f"deferred import: {source} -> {target}" for source, target in sorted(edges - known_edges)
    ]
    violations.extend(
        "runtime cycle: " + ", ".join(sorted(component))
        for component in components
        if component not in known_components
    )
    if violations:
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
