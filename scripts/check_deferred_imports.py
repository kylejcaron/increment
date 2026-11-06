#!/usr/bin/env python3
"""Ratchet deferred internal imports and runtime import cycles."""

import argparse
import ast
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
                    targets.discard(base)
                    targets.add(candidate)
    return targets


class _Imports(ast.NodeVisitor):
    def __init__(self, module: str, package_context: str, package: str, root: Path) -> None:
        self.module = module
        self.package_context = package_context
        self.package = package
        self.root = root
        self.deferred = False
        self.type_checking = False
        self.aliases: dict[str, str] = {}
        self.edges: set[str] = set()
        self.deferred_edges: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "typing":
                self.aliases[alias.asname or "typing"] = "typing"
        self._record(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "typing":
            for alias in node.names:
                if alias.name == "TYPE_CHECKING":
                    self.aliases[alias.asname or alias.name] = "typing.TYPE_CHECKING"
        self._record(node)

    def _record(self, node: ast.AST) -> None:
        targets = _targets(node, self.module, self.package_context, self.package, self.root)
        if not self.type_checking:
            self.edges.update(targets)
            if self.deferred:
                self.deferred_edges.update(targets)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        previous = self.deferred
        self.deferred = True
        for item in node.body:
            self.visit(item)
        self.deferred = previous

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        previous = self.deferred
        self.deferred = True
        for item in node.body:
            self.visit(item)
        self.deferred = previous

    def visit_If(self, node: ast.If) -> None:
        test = node.test
        is_tc = (
            isinstance(test, ast.Name) and self.aliases.get(test.id) == "typing.TYPE_CHECKING"
        ) or (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and self.aliases.get(test.value.id) == "typing"
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
