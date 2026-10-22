"""Require a PR-description reason for every tier whose collected item count rises.

Counts come from collecting the merge base and the pull-request head; lowering
a count needs nothing.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts.check_collect_count import TIERS, collect, collect_base, report

_REASON = re.compile(r"tests: \+(\d+) items — (.+)")


def _approval_lines(body: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line in body.splitlines():
        match = _REASON.fullmatch(line.strip())
        if match and match.group(2).strip():
            lines.append((int(match.group(1)), line.strip()))
    return lines


def check_increases(base: Mapping[str, int], head: Mapping[str, int], body: str) -> tuple[int, str]:
    increases = [(tier, head[tier] - base[tier]) for tier in TIERS if head[tier] > base[tier]]
    reasons = _approval_lines(body)
    matches = [
        (tier, delta, next((line for count, line in reasons if count >= delta), None))
        for tier, delta in increases
    ]
    failures = [
        f"{tier}: {delta:+d} items; add a PR-description line tests: +{delta} items — <reason>"
        for tier, delta, line in matches
        if line is None
    ]
    accepted = [
        f"{tier}: {delta:+d} items covered by {line}" for tier, delta, line in matches if line
    ]
    message = report(base, head) + "".join(f"{line}\n" for line in (*accepted, *failures))
    return int(bool(failures)), message


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base branch reference")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--tests", default="tests")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        base = collect_base(repo, args.base, args.tests)
        head = collect(repo, args.tests)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"could not compare collected items: {error}", file=sys.stderr)
        return 2
    code, message = check_increases(base.tiers, head.tiers, os.environ.get("GITHUB_PR_BODY", ""))
    print(message, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
