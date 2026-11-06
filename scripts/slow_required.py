"""Decide whether a pull request needs the functional slow tier; uncertainty requires it.

The tier is required when a change can reach warehouse or cross-source behaviour, a slow
test or what it imports or reads, or the configuration that selects and runs the tier.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
from collections import deque
from collections.abc import Collection, Iterable, Mapping
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
# Every source runs through these modules or they query a warehouse themselves;
# estimation changes the answer on every source.
_WAREHOUSE_PACKAGES = (
    "increment.query",
    "increment.readouts",
    "increment.kernel",
    "increment.breakout",
    "increment.dashboard",
    "increment.estimation",
    "increment.artifacts",
    "increment.store",
    "increment.reporting",
    "increment.workflows",
)
_WAREHOUSE_MODULES = frozenset(
    {
        "increment._execute",
        "increment.builders",
        "increment.sources",
        "increment.evidence",
        "increment.result_envelope",
    }
)
# Repository packages that slow tests and conftest files import.
_LOCAL_PACKAGES = ("tests", "scripts", "examples", "calibration")
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


def _is_warehouse(module: str) -> bool:
    return module in _WAREHOUSE_MODULES or any(
        module == package or module.startswith(f"{package}.") for package in _WAREHOUSE_PACKAGES
    )


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
    """The module a file defines or, for a deleted file or data, the nearest enclosing one."""
    stem = path.removesuffix(".py") if path.endswith(".py") else path.rpartition("/")[0]
    module = stem.replace("/", ".").removesuffix(".__init__")
    while module not in known and "." in module:
        module = module.rpartition(".")[0]
    return module


def _warehouse_reason(paths: list[str], root: Path) -> str | None:
    # Imported here so that a missing or broken graph module fails closed in main().
    from scripts.check_deferred_imports import runtime_graph

    graph = runtime_graph(root / "increment")
    importer = _reach(graph, (module for module in graph if _is_warehouse(module)))
    for path in paths:
        module = _module_of(path, graph)
        if _is_warehouse(module):
            return f"{path} is in the warehouse path ({module})"
        if module in importer:
            return f"{path} is imported by the warehouse path ({_chain(importer, module)})"
    return None


def _local_modules(root: Path) -> dict[str, Path]:
    modules = {"conftest": root / "conftest.py"} if (root / "conftest.py").is_file() else {}
    for package in _LOCAL_PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            parts = path.relative_to(root).with_suffix("").parts
            modules[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return modules


def _imports(module: str, tree: ast.Module, *, package: bool, modules: Collection[str]) -> set[str]:
    """Repository modules an import in ``module`` executes, with the packages holding them."""
    context = module if package else module.rpartition(".")[0]
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = context.split(".")[: len(context.split(".")) - node.level + 1]
                base = ".".join([*parts, base] if base else parts)
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
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


def _slow_test_reason(paths: list[str], root: Path) -> str | None:
    modules = _local_modules(root)
    sources = {module: path.read_text(encoding="utf-8") for module, path in modules.items()}
    trees = {module: ast.parse(sources[module], str(path)) for module, path in modules.items()}
    roots = [
        module
        for module, path in modules.items()
        if path.name == "conftest.py"
        or (module.startswith("tests.") and _in_slow_tier(trees[module], sources[module]))
    ]
    graph = {
        module: _imports(module, trees[module], package=path.name == "__init__.py", modules=modules)
        for module, path in modules.items()
    }
    importer = _reach(graph, roots)
    reached = [sources[module] for module in importer]
    for path in paths:
        if path.endswith(".py"):
            module = path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
            if module in importer:
                return f"{path} is a slow test or imported by one ({_chain(importer, module)})"
        elif _mentions(path, reached):
            return f"{path} is read by a slow test or a module it imports"
    return None


def decide_paths(paths: list[str], root: Path) -> tuple[bool, str]:
    """Whether changed ``paths`` need the slow tier, judged against the tree at ``root``."""
    for path in paths:
        if path in _GATE_FILES or path.rpartition("/")[2] == "conftest.py":
            return True, f"{path} selects or configures the slow tier"
        if path.startswith(_SUPPORT_TREES):
            return True, f"{path} supports the slow tests"
    package_paths = [path for path in paths if path.startswith("increment/")]
    if package_paths and (reason := _warehouse_reason(package_paths, root)):
        return True, reason
    local_paths = [path for path in paths if path.partition("/")[0] in _LOCAL_PACKAGES]
    if local_paths and (reason := _slow_test_reason(local_paths, root)):
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
