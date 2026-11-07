"""Decide whether a pull request needs the functional slow tier; uncertainty requires it.

The tier is required when a change touches the configuration that selects and runs it, a
slow test, or code or data a slow test reaches through imports or module invocations.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import tomllib
from collections import deque
from collections.abc import Collection, Iterable, Iterator, Mapping
from pathlib import Path

_GATE_FILES = frozenset(
    {
        "pyproject.toml",
        "uv.lock",
        "Makefile",
        "noxfile.py",
        ".github/workflows/ci.yml",
        "scripts/slow_required.py",
        "scripts/check_deferred_imports.py",
    }
)
_SUPPORT_TREES = (
    "integration/",
    "tests/fixtures/",
    "tests/parity_harness/",
    "tests/evidence_fixtures/",
    "tests/query/",
)
_PRODUCTION_PACKAGES = ("increment", "calibration")
# Slow tests also import and run test, script and example code.
_PACKAGES = (*_PRODUCTION_PACKAGES, "tests", "scripts", "examples")
_SLOW_MARK = "pytest.mark.slow"
# Marks the functional slow tier deselects; each belongs to its own gate.
_OTHER_TIERS = (
    "tooling",
    "exhaustive",
    "parameter_recovery",
    "numerical_certification",
    "examples",
    "compile_registry",
    "benchmark",
)
# A module an import statement names, and the names it imports from it.
type Import = tuple[str, tuple[str, ...]]
# Literal module runs: a "-m", "<module>" argument pair or run_module("<module>").
_INVOCATION = re.compile(r"""["']-m["']\s*,\s*["']([\w.]+)["']|run_module\(\s*["']([\w.]+)["']""")


def _reach(graph: Mapping[str, Collection[str]], roots: Iterable[str]) -> dict[str, str | None]:
    """Every module imported transitively from ``roots``, mapped to the module importing it."""
    importer: dict[str, str | None] = dict.fromkeys(sorted(roots))
    pending = deque(importer)
    while pending:
        module = pending.popleft()
        for target in sorted(graph.get(module, ())):
            if target not in importer:
                importer[target] = module
                pending.append(target)
    return importer


def _chain(importer: Mapping[str, str | None], module: str) -> str:
    chain = [module]
    while (parent := importer[chain[-1]]) is not None:
        chain.append(parent)
    return " -> ".join(reversed(chain))


def _module_of(path: str, known: Collection[str]) -> str:
    """The module a file defines or, for package data, the nearest enclosing package."""
    stem = path.removesuffix(".py") if path.endswith(".py") else path.rpartition("/")[0]
    module = stem.replace("/", ".").removesuffix(".__init__")
    while module not in known and "." in module:
        module = module.rpartition(".")[0]
    return module


def _local_modules(root: Path) -> dict[str, Path]:
    modules = {"conftest": root / "conftest.py"} if (root / "conftest.py").is_file() else {}
    for package in _PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            parts = path.relative_to(root).with_suffix("").parts
            modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return modules


def _is_type_checking(test: ast.expr) -> bool:
    name = test.attr if isinstance(test, ast.Attribute) else getattr(test, "id", None)
    return name == "TYPE_CHECKING"


def _statements(tree: ast.Module) -> Iterator[ast.stmt]:
    """Every statement, skipping the bodies of ``if TYPE_CHECKING:`` blocks."""
    pending: list[ast.stmt] = list(tree.body)
    while pending:
        node = pending.pop()
        yield node
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            pending.extend(node.orelse)
            continue
        for field in ("body", "orelse", "finalbody"):
            pending.extend(getattr(node, field, ()))
        for clause in (*getattr(node, "handlers", ()), *getattr(node, "cases", ())):
            pending.extend(clause.body)


def _text(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _lazy_exports(tree: ast.Module, modules: Collection[str]) -> dict[str, str]:
    """Names a module-level ``__getattr__`` loads from a literal ``{name: (module, ...)}`` table."""
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__" for node in tree.body
    ):
        return {}
    exports: dict[str, str] = {}
    for node in tree.body:
        table = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if not isinstance(table, ast.Dict):
            continue
        for key, value in zip(table.keys, table.values, strict=True):
            name = _text(key) if key is not None else None
            target = _text(value.elts[0]) if isinstance(value, ast.Tuple) and value.elts else None
            if name and target and target in modules:
                exports[name] = target
    return exports


def _import_records(module: str, tree: ast.Module, *, package: bool) -> list[Import]:
    """Each import as ``(module, names)``; ``names`` is empty for ``import module``."""
    context = module if package else module.rpartition(".")[0]
    records: list[Import] = []
    for node in _statements(tree):
        if isinstance(node, ast.Import):
            records.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = context.split(".")[: len(context.split(".")) - node.level + 1]
                base = ".".join([*parts, base] if base else parts)
            records.append((base, tuple(alias.name for alias in node.names)))
    return records


def _targets(
    records: Iterable[Import],
    invoked: Iterable[str],
    modules: Collection[str],
    exports: Mapping[str, Mapping[str, str]],
) -> set[str]:
    """Repository modules imported or run, with the packages holding them."""
    names = list(invoked)
    for base, imported in records:
        lazy = exports.get(base, {})
        names.append(base)
        if not imported:
            # Attribute access on an imported lazy package can load any of its exports.
            names.extend(lazy.values())
        for name in imported:
            names.extend([f"{base}.{name}", lazy.get(name, "")])
    found: set[str] = set()
    for name in names:
        while name:
            if name in modules:
                found.add(name)
            name = name.rpartition(".")[0]
    return found


def _in_slow_tier(tree: ast.Module, source: str) -> bool:
    """Whether a test module holds slow tests that no module-wide mark moves to another gate."""
    module_marks = " ".join(
        ast.unparse(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
        )
    )
    return _SLOW_MARK in source and not any(f"mark.{mark}" in module_marks for mark in _OTHER_TIERS)


def _mentions(path: str, sources: Iterable[str]) -> bool:
    """Whether a source names the data file or a directory holding it as a path segment."""
    segments = "|".join(re.escape(segment) for segment in path.split("/")[1:])
    pattern = re.compile(rf"(?<=[\"'/])(?:{segments})(?=[\"'/])")
    return any(pattern.search(source) for source in sources)


def _entry_point_modules(root: Path) -> list[str]:
    """Modules named by the project's entry points, which the package loads at run time."""
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return []
    groups = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
    return sorted(
        target.partition(":")[0].strip()
        for group in groups.get("entry-points", {}).values()
        for target in group.values()
    )


def _runs(source: str, plugins: list[str]) -> list[str]:
    """Modules a source runs other than by import: literal module runs and entry points."""
    invoked = [first or second for first, second in _INVOCATION.findall(source)]
    return invoked + plugins if "entry_points" in source else invoked


def _slow_reach(root: Path) -> tuple[dict[str, str | None], dict[str, str]]:
    """Modules the slow tier reaches from its test modules and conftest files, with sources."""
    # Imported here so that a missing or broken graph module fails closed in main().
    from scripts.check_deferred_imports import runtime_graph

    modules = _local_modules(root)
    plugins = _entry_point_modules(root)
    sources: dict[str, str] = {}
    records: dict[str, list[Import]] = {}
    exports: dict[str, dict[str, str]] = {}
    roots: list[str] = []
    for module, path in modules.items():
        sources[module] = source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, str(path))
        records[module] = _import_records(module, tree, package=path.name == "__init__.py")
        if lazy := _lazy_exports(tree, modules):
            exports[module] = lazy
        if path.name == "conftest.py" or (
            module.startswith("tests.") and _in_slow_tier(tree, source)
        ):
            roots.append(module)
    graph = {
        module: _targets(records[module], _runs(sources[module], plugins), modules, exports)
        for module in modules
    }
    for package in _PRODUCTION_PACKAGES:
        if (root / package).is_dir():
            for module, targets in runtime_graph(root / package).items():
                graph.setdefault(module, set()).update(targets)
    return _reach(graph, roots), sources


def _slow_reason(paths: list[str], root: Path) -> str | None:
    importer, sources = _slow_reach(root)
    reached = [sources[module] for module in importer if module in sources]
    for path in paths:
        if path.endswith(".py"):
            module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
        elif path.partition("/")[0] in _PRODUCTION_PACKAGES:
            module = _module_of(path, sources)
        elif _mentions(path, reached):
            return f"{path} is read by a slow test or a module it uses"
        else:
            continue
        if module in importer:
            return f"{path} is used by the slow tier ({_chain(importer, module)})"
    return None


def decide_paths(paths: list[str], root: Path) -> tuple[bool, str]:
    """Whether changed ``paths`` need the slow tier, judged against the tree at ``root``."""
    for path in paths:
        if path in _GATE_FILES or path.rpartition("/")[2] == "conftest.py":
            return True, f"{path} selects or configures the slow tier"
        if path.startswith(_SUPPORT_TREES):
            return True, f"{path} supports the slow tests"
    code_paths = [path for path in paths if path.partition("/")[0] in _PACKAGES]
    if code_paths and (reason := _slow_reason(code_paths, root)):
        return True, reason
    return False, "no changed path reaches the slow tier"


def _changed_paths(base: str, head: str, root: Path) -> list[str]:
    """Paths changed on ``head`` since it left ``base``; empty when they cannot be listed."""
    if not base or not head:
        return []
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}", "--"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode or not result.stdout.endswith(b"\0"):
        return []
    return [os.fsdecode(path) for path in result.stdout[:-1].split(b"\0")]


def decide(event: str, base: str, head: str, root: Path) -> tuple[bool, str]:
    """Whether a run needs the slow tier; only pull requests can skip it."""
    if event != "pull_request":
        return True, f"every {event} run keeps the slow tier"
    paths = _changed_paths(base, head, root)
    if not paths:
        return True, f"no changed files could be listed between {base or '(no base)'} and {head}"
    return decide_paths(paths, root)


def main(argv: list[str] | None = None) -> None:
    """Print the decision and its reason as GitHub Actions outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default="pull_request")
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args(argv)
    try:
        required, reason = decide(args.event, args.base, args.head, Path.cwd())
    except Exception as error:  # an undecidable change keeps the tier
        required, reason = True, f"the decision failed ({type(error).__name__}: {error})"
    print(f"required={str(required).lower()}")
    print(f"reason={' '.join(reason.split())}")


if __name__ == "__main__":
    main()
