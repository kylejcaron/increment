"""Run unit tests affected by the current tree, reusing successful receipts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def receipt_path(root: Path, base: str) -> Path:
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    directory = Path(_git(root, "rev-parse", "--git-path", "prepush-receipts"))
    if not directory.is_absolute():
        directory = root / directory
    return directory / f"{tree}-{base}"


def _run(command: list[str], root: Path) -> int:
    return subprocess.run(command, cwd=root, check=False).returncode


def _changed_paths(root: Path, base: str) -> set[str]:
    paths = set(_git(root, "diff", "--name-only", f"{base}...HEAD").splitlines())
    paths.update(_git(root, "diff", "--name-only").splitlines())
    paths.update(_git(root, "diff", "--name-only", "--cached").splitlines())
    paths.update(_git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    return paths


def main() -> int:
    root = Path.cwd()
    base = _git(root, "merge-base", "origin/main", "HEAD")
    receipt = receipt_path(root, base)
    if receipt.is_file():
        print("pre-push: tests already passed for this tree and merge base")
        return 0

    status = 0
    if "tach.toml" in _changed_paths(root, base):
        status = _run(["uv", "run", "--no-sync", "tach", "check"], root)
    if status == 0:
        if (root / ".testmondata").exists():
            status = _run([sys.executable, "-m", "scripts.test_changed", "--tier", "unit"], root)
        else:
            # The first run in a worktree records coverage for every unit test.
            status = _run(
                [sys.executable, "-m", "pytest", "--testmon", "-q", "-m", _unit_marker()], root
            )

    if status == 0:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.touch()
    return status


def _unit_marker() -> str:
    from scripts.test_changed import _marker_expression

    return _marker_expression("unit")


if __name__ == "__main__":
    raise SystemExit(main())
