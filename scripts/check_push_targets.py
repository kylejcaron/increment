"""Validate Git pre-push updates against the checked-out candidate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PushUpdate:
    local_ref: str
    local_oid: str
    remote_ref: str
    remote_oid: str


def parse_push_updates(lines: Iterable[str]) -> tuple[PushUpdate, ...]:
    """Parse Git's four-field, line-oriented pre-push protocol."""
    updates: list[PushUpdate] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 4:
            raise ValueError(f"malformed pre-push record on line {line_number}: expected 4 fields")
        updates.append(PushUpdate(*fields))
    return tuple(updates)


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"git {' '.join(args)} failed in {repo}") from exc
    return result.stdout.strip()


def checked_push_commits(repo: Path, updates: Sequence[PushUpdate]) -> tuple[str, ...]:
    """Require every non-deletion update to name the clean checked-out HEAD."""
    repo = repo.expanduser().resolve()
    zero_oids: set[int] = set()
    commits: list[str] = []
    seen: set[str] = set()
    for update in updates:
        oid = update.local_oid
        if not oid or any(character not in "0123456789abcdefABCDEF" for character in oid):
            raise ValueError(f"invalid local object ID for {update.local_ref!r}")
        if len(oid) not in (40, 64):
            raise ValueError(f"invalid local object ID for {update.local_ref!r}")
        if set(oid) == {"0"}:
            zero_oids.add(len(oid))
            continue
        try:
            commit = _git(repo, "rev-parse", "--verify", f"{oid}^{{commit}}")
        except ValueError as exc:
            raise ValueError(
                f"local object {oid} for {update.local_ref!r} is not a commit"
            ) from exc
        head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
        if commit != head:
            raise ValueError(f"{update.local_ref!r} candidate does not match checked-out HEAD")
        status = _git(repo, "status", "--porcelain", "--untracked-files=all")
        if status:
            raise ValueError(f"worktree is not clean for candidate {update.local_ref!r}")
        if commit not in seen:
            commits.append(commit)
            seen.add(commit)
    if len(zero_oids) > 1:
        raise ValueError("deletion records use inconsistent object ID lengths")
    return tuple(commits)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="absolute path to the repository")
    args = parser.parse_args(argv)
    if not args.repo.is_absolute():
        parser.error("--repo must be an absolute path")
    try:
        updates = parse_push_updates(sys.stdin)
        commits = checked_push_commits(args.repo, updates)
    except ValueError as exc:
        print(f"check-push-targets: {exc}", file=sys.stderr)
        return 1
    for commit in commits:
        print(commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
