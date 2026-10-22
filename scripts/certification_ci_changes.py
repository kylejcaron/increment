"""Select the numerical-certification PR gate; uncertainty requires it to run."""

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
    "tests/power/test_binomial_planning.py",
    "tests/estimation/test_conversion_route_calibration.py",
    "tests/calibration/test_cuped_linear.py",
    "tests/parity_harness/cases.py",
}
_CERTIFICATION_TEST_MODULES = (
    "tests/power/test_binomial_planning.py",
    "tests/estimation/test_conversion_route_calibration.py",
    "tests/calibration/test_cuped_linear.py",
)
_ROOT_DOCS = {"README.md", "CONTRIBUTING.md", "AGENTS.md", "CHANGELOG.md"}
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


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


def _requires_certification(event: str, base: str, head: str, root: Path | None = None) -> bool:
    if event != "pull_request":
        return True
    if not base or not head:
        print("Missing PR base or head; requiring numerical certification.", file=sys.stderr)
        return True
    root = root or Path.cwd()
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}", "--"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        print("Cannot determine PR changes; requiring numerical certification.", file=sys.stderr)
        return True
    if not result.stdout or not result.stdout.endswith(b"\0"):
        return True
    entries = result.stdout[:-1].split(b"\0")
    if len(entries) % 2:
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
    for index in range(0, len(entries), 2):
        status = entries[index]
        path = os.fsdecode(entries[index + 1])
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    required = _requires_certification(args.event, args.base, args.head)
    print(f"required={str(required).lower()}")


if __name__ == "__main__":
    main()
