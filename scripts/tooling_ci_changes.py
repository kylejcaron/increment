"""Select the repository-tooling CI gate; only increment-only PRs can skip it."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def requires_tooling(event: str, base: str, head: str, root: Path | None = None) -> bool:
    if event != "pull_request":
        return True
    if not base or not head:
        print("Missing PR base or head; requiring tooling tests.", file=sys.stderr)
        return True
    result = subprocess.run(
        ["git", "diff", "--no-renames", "--name-only", "-z", f"{base}...{head}", "--"],
        cwd=root or Path.cwd(),
        capture_output=True,
        check=False,
    )
    if result.returncode or (result.stdout and not result.stdout.endswith(b"\0")):
        print("Cannot determine PR changes; requiring tooling tests.", file=sys.stderr)
        return True
    if not result.stdout:
        print("Cannot determine PR changes; requiring tooling tests.", file=sys.stderr)
        return True
    paths = result.stdout[:-1].split(b"\0")
    if not paths or any(not path for path in paths):
        print("Cannot determine PR changes; requiring tooling tests.", file=sys.stderr)
        return True
    return any(not os.fsdecode(path).startswith("increment/") for path in paths)


def main() -> None:
    """Emit a single GitHub Actions output for the conditional tooling gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    required = requires_tooling(args.event, args.base, args.head)
    print(f"required={str(required).lower()}")


if __name__ == "__main__":
    main()
