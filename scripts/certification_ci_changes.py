"""Select the numerical-certification and exhaustive PR gates; uncertainty requires them to run."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path

_GATE_FILES = {
    "Makefile",
    "noxfile.py",
    "pyproject.toml",
    "uv.lock",
    "tach.toml",
    "scripts/certification_ci_changes.py",
    "increment/_frame_moments.py",
    "increment/query/builders.py",
    "increment/query/artifact_reader.py",
    "increment/sources.py",
    "tests/power/test_binomial_planning.py",
    "tests/estimation/test_conversion_route_calibration.py",
    "tests/calibration/test_cuped_linear.py",
    "tests/parity_harness/cases.py",
}
_CERTIFICATION_TEST_MODULES = (
    "tests/power/test_binomial_planning.py",
    "tests/estimation/test_conversion_route_calibration.py",
    "tests/calibration/test_cuped_linear.py",
    "tests/estimation/test_sequential_public_proof_acceptance.py",
)
_ROOT_DOCS = {"README.md", "CONTRIBUTING.md", "AGENTS.md", "CHANGELOG.md"}
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
# Loaders, query, the evidence layer (AGENTS.md places source declarations in it), the parity
# harness and evidence fixtures, and the files that define how the gate selects and runs.
_EXHAUSTIVE_FILES = {
    "Makefile",
    "noxfile.py",
    "pyproject.toml",
    "uv.lock",
    "scripts/certification_ci_changes.py",
    ".github/workflows/ci.yml",
    "increment/sources.py",
    "increment/frame.py",
    "increment/readouts/_evidence.py",
    "increment/_source_types.py",
}
_EXHAUSTIVE_TREES = ("increment/query/", "tests/parity_harness/", "tests/evidence_fixtures/")
_EXHAUSTIVE_MODULE_PREFIXES = ("increment/_frame_", "increment/_evidence_")
_EXHAUSTIVE_MARK = b"pytest.mark.exhaustive"


def _module_for_path(path: str, module_names: set[str]) -> str | None:
    if not path.startswith("increment/") or not path.endswith(".py"):
        return None
    module = path[:-3].replace("/", ".").removesuffix(".__init__")
    while module and module not in module_names:
        module = module.rpartition(".")[0]
    return module or None


def _module_for_import(imported: str, module_names: set[str]) -> str:
    module = imported
    while module and module not in module_names:
        module = module.rpartition(".")[0]
    if module == "increment" or not module:
        raise ValueError(f"Cannot map certification import {imported!r} to a tach module.")
    return module


def _package_exports(root: Path, package: str) -> dict[str, str]:
    init = root / package.replace(".", "/") / "__init__.py"
    if not init.is_file():
        return {}
    tree = ast.parse(init.read_text(), filename=str(init))
    exports: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.level == 0 and node.module.startswith("increment."):
            source = node.module
        elif node.level:
            source = package if not node.module else f"{package}.{node.module}"
        else:
            continue
        for alias in node.names:
            exports[alias.asname or alias.name] = source
    return exports


def _imported_modules(node: ast.AST, root: Path, module_names: set[str]) -> set[str]:
    if isinstance(node, ast.Import):
        return {
            _module_for_import(alias.name, module_names)
            for alias in node.names
            if alias.name.startswith("increment.")
        }
    if not isinstance(node, ast.ImportFrom) or node.level != 0 or not node.module:
        return set()
    if not node.module.startswith("increment."):
        return set()
    package = _module_for_import(node.module, module_names)
    exports = _package_exports(root, package)
    modules = set()
    for alias in node.names:
        if alias.name == "*":
            modules.add(package)
            continue
        child = f"{node.module}.{alias.name}"
        target = child if child in module_names else exports.get(alias.name, package)
        modules.add(_module_for_import(target, module_names))
    return modules


def _dependency_closure(root: Path, roots: set[str]) -> set[str]:
    graph = tomllib.loads((root / "tach.toml").read_text())
    dependencies = {item["path"]: set(item["depends_on"]) for item in graph["modules"]}
    closure = set(roots)
    if not closure <= dependencies.keys():
        raise ValueError("Dependency roots are missing from tach.toml.")
    pending = list(closure)
    while pending:
        for dependency in dependencies[pending.pop()]:
            if dependency not in closure:
                closure.add(dependency)
                pending.append(dependency)
    return closure


def _certified_modules(root: Path) -> set[str]:
    graph = tomllib.loads((root / "tach.toml").read_text())
    module_names = {item["path"] for item in graph["modules"]}
    certified: set[str] = set()
    for test_path in _CERTIFICATION_TEST_MODULES:
        tree = ast.parse((root / test_path).read_text(), filename=test_path)
        for node in ast.walk(tree):
            certified.update(_imported_modules(node, root, module_names))
    if not certified:
        raise ValueError("No certified modules were found in the test imports.")
    return _dependency_closure(root, certified)


def _changed_entries(base: str, head: str, root: Path, gate: str) -> list[tuple[bytes, str]] | None:
    """The PR's ``(status, path)`` changes, or ``None`` when they cannot be determined."""
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}", "--"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        print(f"Cannot determine PR changes; requiring {gate}.", file=sys.stderr)
        return None
    if not result.stdout or not result.stdout.endswith(b"\0"):
        return None
    entries = result.stdout[:-1].split(b"\0")
    if len(entries) % 2:
        return None
    return [
        (entries[index], os.fsdecode(entries[index + 1])) for index in range(0, len(entries), 2)
    ]


def _requires_certification(event: str, base: str, head: str, root: Path | None = None) -> bool:
    if event != "pull_request":
        return True
    if not base or not head:
        print("Missing PR base or head; requiring numerical certification.", file=sys.stderr)
        return True
    root = root or Path.cwd()
    entries = _changed_entries(base, head, root, "numerical certification")
    if entries is None:
        return True
    try:
        graph = tomllib.loads((root / "tach.toml").read_text())
        module_names = {item["path"] for item in graph["modules"]}
        certified_modules = _certified_modules(root)
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError, SyntaxError):
        print(
            "Cannot determine certified module graph; requiring numerical certification.",
            file=sys.stderr,
        )
        return True
    for status, path in entries:
        if status not in {b"A", b"M"}:
            return True
        if path in _GATE_FILES or path.startswith(".github/workflows/"):
            return True
        if path.startswith("calibration/"):
            return True
        module = _module_for_path(path, module_names)
        if module is not None:
            if module in certified_modules:
                return True
            continue
        if path.startswith("increment/"):
            return True
        if path in _ROOT_DOCS or (path.startswith("docs/") and path.endswith(_DOC_SUFFIXES)):
            continue
        return True
    return False


def _reaches_exhaustive_tests(path: str, head: str, root: Path) -> bool:
    if path in _EXHAUSTIVE_FILES or path.startswith(_EXHAUSTIVE_TREES):
        return True
    if (
        path.endswith(".py")
        and path.count("/") == 1
        and path.startswith(_EXHAUSTIVE_MODULE_PREFIXES)
    ):
        return True
    if not (path.startswith("tests/") and path.endswith(".py")):
        return False
    # A test module that holds exhaustive tests runs them when it changes.
    shown = subprocess.run(
        ["git", "show", f"{head}:{path}"], cwd=root, capture_output=True, check=False
    )
    return shown.returncode == 0 and _EXHAUSTIVE_MARK in shown.stdout


def _requires_exhaustive(event: str, base: str, head: str, root: Path | None = None) -> bool:
    if event != "pull_request":
        return True
    if not base or not head:
        print("Missing PR base or head; requiring the exhaustive gate.", file=sys.stderr)
        return True
    root = root or Path.cwd()
    entries = _changed_entries(base, head, root, "the exhaustive gate")
    if entries is None:
        return True
    return any(_reaches_exhaustive_tests(path, head, root) for _, path in entries)


_GATES = {"certification": _requires_certification, "exhaustive": _requires_exhaustive}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=tuple(_GATES), default="certification")
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    required = _GATES[args.gate](args.event, args.base, args.head)
    print(f"required={str(required).lower()}")


if __name__ == "__main__":
    main()
