"""Select the warehouse compile-registry PR gate; uncertainty requires it to run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ROOT_DOCS = {"README.md", "CONTRIBUTING.md", "AGENTS.md", "CHANGELOG.md"}
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")
_CONFIG_FILES = {"pyproject.toml", "uv.lock", "noxfile.py"}


def _requires_compile_registry(event: str, base: str, head: str, root: Path | None = None) -> bool:
    if event != "pull_request":
        return True
    if not base or not head:
        print("Missing PR base or head; requiring warehouse compile registry.", file=sys.stderr)
        return True
    root = root or Path.cwd()
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}", "--"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode or (result.stdout and not result.stdout.endswith(b"\0")):
        print("Cannot determine PR changes; requiring warehouse compile registry.", file=sys.stderr)
        return True
    try:
        dependencies = {
            line.strip()
            for line in (root / "integration/warehouse_execution/_compile_dependencies.txt")
            .read_text()
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError:
        print(
            "Cannot read compile dependency record; requiring warehouse compile registry.",
            file=sys.stderr,
        )
        return True

    for raw_path in result.stdout[:-1].split(b"\0") if result.stdout else ():
        path = os.fsdecode(raw_path)
        if path.startswith("integration/warehouse_execution/") or path in _CONFIG_FILES:
            return True
        if path in dependencies:
            return True
        if path.startswith("increment/") and path.endswith(".py"):
            module = path[:-3].replace("/", ".").removesuffix(".__init__")
            if module in dependencies:
                return True
        if path.endswith("conftest.py") or (
            path.startswith("tests/") and Path(path).name.startswith("_") and path.endswith(".py")
        ):
            return True
        if path in _ROOT_DOCS or (path.startswith("docs/") and path.endswith(_DOC_SUFFIXES)):
            continue
        return True
    return False


def requires_compile_registry(event: str, base: str, head: str, root: Path | None = None) -> bool:
    return _requires_compile_registry(event, base, head, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    required = requires_compile_registry(args.event, args.base, args.head)
    print(f"required={str(required).lower()}")


if __name__ == "__main__":
    main()
