"""Select the cross-platform wheel smoke gate for packaging changes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _requires_wheel_smoke(event: str, base: str, head: str) -> bool:
    if event != "pull_request":
        return True
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}", "--"],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        print("Cannot determine PR changes; requiring wheel smoke.", file=sys.stderr)
        return True
    if not result.stdout or not result.stdout.endswith(b"\0"):
        return True
    entries = result.stdout[:-1].split(b"\0")
    if len(entries) % 2:
        return True
    for index in range(0, len(entries), 2):
        status = entries[index]
        path = os.fsdecode(entries[index + 1])
        if status not in {b"A", b"M"}:
            return True
        if path in {"pyproject.toml", "uv.lock", "MANIFEST.in", "hatch_build.py"}:
            return True
        if path.startswith("increment/") and (
            path.endswith("/py.typed") or not path.endswith(".py")
        ):
            return True
    return False


def main() -> None:
    """Emit a single GitHub Actions output for the conditional wheel smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    required = _requires_wheel_smoke(args.event, args.base, args.head)
    print(f"required={str(required).lower()}")


if __name__ == "__main__":
    main()
