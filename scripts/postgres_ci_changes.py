"""Select the PostgreSQL PR gate; only known docs-only changes may skip it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from certification_ci_changes import _dependency_closure, _module_for_path

_GATE_FILES = {"pyproject.toml", "uv.lock", "noxfile.py", "Makefile"}
_ROOT_DOCS = {"README.md", "CONTRIBUTING.md", "CHANGELOG.md"}
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


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
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError):
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
            if "postgres" in path.lower():
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
