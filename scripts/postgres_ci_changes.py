"""Select the PostgreSQL PR gate; only known docs-only changes may skip it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

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
    if result.returncode:
        print("Cannot determine PR changes; requiring PostgreSQL.", file=sys.stderr)
        return True
    if not result.stdout or not result.stdout.endswith(b"\0"):
        return True
    entries = result.stdout[:-1].split(b"\0")
    if len(entries) % 2:
        return True
    for index in range(0, len(entries), 2):
        status = entries[index]
        path = os.fsdecode(entries[index + 1])
        # --no-renames exposes both sides; deletions and type changes always run.
        if status not in {b"A", b"M"}:
            return True
        if path not in _ROOT_DOCS and not (
            path.startswith("docs/") and path.endswith(_DOC_SUFFIXES)
        ):
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
