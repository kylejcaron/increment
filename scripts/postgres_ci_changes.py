"""Select the PostgreSQL PR gate; only known docs-only changes may skip it."""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from certification_ci_changes import _dependency_closure, _module_for_path

_GATE_FILES = {"pyproject.toml", "uv.lock", "noxfile.py", "Makefile"}
_ROOT_DOCS = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md"}
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


_POSTGRES_SUITE = Path("integration/warehouse_execution/test_postgres_execution.py")


def _module_file(root: Path, module: str) -> Path | None:
    base = root.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate.relative_to(root)
    return None


def _suite_test_dependencies(root: Path) -> set[str]:
    """Repository test helpers the live PostgreSQL suite imports, transitively."""
    seen: set[Path] = set()
    pending = [_POSTGRES_SUITE]
    while pending:
        path = pending.pop()
        if path in seen or not (root / path).is_file():
            continue
        seen.add(path)
        for node in ast.walk(ast.parse((root / path).read_text(encoding="utf-8"))):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level:
                    package = list(path.with_suffix("").parts)
                    if path.name != "__init__.py":
                        package = package[:-1]
                    package = package[: len(package) - (node.level - 1)]
                    module = ".".join([*package, *([module] if module else [])])
                names = [module, *(f"{module}.{alias.name}" for alias in node.names)]
            for name in names:
                if name.split(".")[0] not in {"tests", "integration"}:
                    continue
                parts = name.split(".")
                for end in range(1, len(parts) + 1):
                    found = _module_file(root, ".".join(parts[:end]))
                    if found is not None:
                        pending.append(found)
    return {path.as_posix() for path in seen}


def _requires_postgres(event: str, base: str, head: str) -> bool:
    if event != "pull_request":
        return True
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}", "--"],
        capture_output=True,
        check=False,
    )
    if result.returncode or not result.stdout or not result.stdout.endswith(b"\0"):
        print("Cannot determine PR changes; requiring PostgreSQL.", file=sys.stderr)
        return True
    entries = result.stdout[:-1].split(b"\0")
    if len(entries) % 2:
        return True
    root = Path.cwd()
    try:
        graph = tomllib.loads((root / "tach.toml").read_text())
        module_names = {item["path"] for item in graph["modules"]}
        query_roots = {name for name in module_names if name.startswith("increment.query")}
        query_modules = _dependency_closure(root, query_roots)
        suite_helpers = _suite_test_dependencies(root)
    except (OSError, KeyError, SyntaxError, ValueError, tomllib.TOMLDecodeError):
        print("Cannot determine query dependency graph; requiring PostgreSQL.", file=sys.stderr)
        return True
    for index in range(0, len(entries), 2):
        status = entries[index]
        path = os.fsdecode(entries[index + 1])
        if status not in {b"A", b"M"}:
            return True
        if path in _GATE_FILES or path.startswith(".github/workflows/"):
            return True
        if path.startswith("integration/warehouse_execution/"):
            return True
        if path.startswith("tests/"):
            if "postgres" in path.lower() or path in suite_helpers:
                return True
            continue
        module = _module_for_path(path, module_names)
        if module is not None:
            if module in query_modules:
                return True
            continue
        if path in _ROOT_DOCS or (path.startswith("docs/") and path.endswith(_DOC_SUFFIXES)):
            continue
        if path.startswith("increment/"):
            return True
        if path.startswith("tests/"):
            continue
        return True
    return False


def main() -> None:
    """Emit a single GitHub Actions output for the required-check aggregator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    required = _requires_postgres(args.event, args.base, args.head)
    print(f"required={str(required).lower()}")


if __name__ == "__main__":
    main()
