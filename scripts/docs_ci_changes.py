"""Select documentation snippet and strict-site PR checks from changed paths."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_SNIPPET_ROOTS = {
    "README.md",
    "noxfile.py",
    "mkdocs.yml",
    "pyproject.toml",
    "uv.lock",
    ".github/workflows/ci.yml",
}
_STRICT_ROOTS = _SNIPPET_ROOTS | {"CONTRIBUTING.md"}
_STRICT_PACKAGE_FILES = {"increment/__init__.py", "increment/results.py"}
_REQUIRED_LABELS = {"breaking", "docs-required"}


def _changed_paths(base: str, head: str) -> tuple[set[str], bool]:
    if not base or not head:
        return set(), False
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}", "--"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        print(
            "Cannot determine changed files; requiring both documentation checks.", file=sys.stderr
        )
        return set(), False
    if result.stdout and not result.stdout.endswith(b"\0"):
        return set(), False
    entries = result.stdout[:-1].split(b"\0") if result.stdout else []
    if len(entries) % 2:
        return set(), False
    paths: set[str] = set()
    for index in range(0, len(entries), 2):
        status, raw_path = entries[index : index + 2]
        if status not in {b"A", b"M"}:
            return set(), False
        paths.add(os.fsdecode(raw_path))
    return paths, True


def _decisions(event: str, base: str, head: str, labels: str) -> tuple[bool, bool]:
    paths, known_diff = _changed_paths(base, head)
    if not known_diff:
        return True, True

    snippets = any(
        path in _SNIPPET_ROOTS
        or path.startswith("docs/")
        or (path.startswith("increment/") and path.endswith(".py"))
        for path in paths
    )
    strict_docs = any(
        path in _STRICT_ROOTS or path.startswith("docs/") or path in _STRICT_PACKAGE_FILES
        for path in paths
    )
    if event == "pull_request" and _REQUIRED_LABELS.intersection(labels.split(",")):
        strict_docs = True
    return snippets, strict_docs


def main() -> None:
    """Emit GitHub Actions outputs consumed by conditional docs jobs and ci-ok."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--labels", default="")
    args = parser.parse_args()
    snippets, strict_docs = _decisions(args.event, args.base, args.head, args.labels)
    print(f"snippets={str(snippets).lower()}")
    print(f"strict_docs={str(strict_docs).lower()}")


if __name__ == "__main__":
    main()
