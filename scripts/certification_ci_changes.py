"""Select the numerical-certification PR gate; uncertainty requires it to run."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_GATE_FILES = {
    "Makefile",
    "noxfile.py",
    "pyproject.toml",
    "uv.lock",
    "scripts/_test_tier_policy.py",
    "scripts/certification_ci_changes.py",
    "scripts/run_test_tier.py",
    "scripts/run_test_tier_plugin.py",
    "tests/estimation/test_conversion_route_calibration.py",
    ".github/workflows/ci.yml",
    ".github/workflows/weekly.yml",
}
_ROOT_DOCS = {"README.md", "CONTRIBUTING.md", "AGENTS.md", "CHANGELOG.md"}
_DOC_SUFFIXES = (".md", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def _requires_certification(event: str, base: str, head: str) -> bool:
    if event != "pull_request":
        return True
    if not base or not head:
        print("Missing PR base or head; requiring numerical certification.", file=sys.stderr)
        return True
    result = subprocess.run(
        ["git", "diff", "--name-status", "--no-renames", "-z", f"{base}...{head}", "--"],
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
    for index in range(0, len(entries), 2):
        status = entries[index]
        path = os.fsdecode(entries[index + 1])
        if status not in {b"A", b"M"}:
            return True
        if path in _GATE_FILES:
            return True
        if path.startswith(("increment/", "calibration/")):
            return True
        if path.startswith("tests/estimation/") and (
            path == "tests/estimation/test_conversion_route_calibration.py"
            or "conversion_route" in path
        ):
            return True
        if path.startswith("scripts/") and (
            path.startswith("scripts/run_test_") or path == "scripts/_test_tier_policy.py"
        ):
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
