from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_REASON = re.compile(r"tests: \+(\d+) items — (.+)")


def _baseline(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[1].isdecimal() or parts[0] in values:
            raise ValueError(f"invalid collected-item baseline line: {line!r}")
        values[parts[0]] = int(parts[1])
    if not values:
        raise ValueError("collected-item baseline is empty")
    return values


def _approval_lines(body: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for line in body.splitlines():
        match = _REASON.fullmatch(line)
        if match and match.group(2).strip():
            lines.append((int(match.group(1)), line))
    return lines


def check_increases(base: Mapping[str, int], head: Mapping[str, int], body: str) -> tuple[int, str]:
    if base.keys() != head.keys():
        return 2, "collected-item baseline tiers differ between base and pull request\n"
    increases = [(tier, head[tier] - base[tier]) for tier in base if head[tier] > base[tier]]
    reasons = _approval_lines(body)
    matches = [
        (tier, delta, next((line for count, line in reasons if count >= delta), None))
        for tier, delta in increases
    ]
    failures = [
        f"{tier}: baseline increased by {delta}; add a PR-body line tests: +{delta} items — <reason>"
        for tier, delta, line in matches
        if line is None
    ]
    accepted = [
        f"{tier}: baseline increase +{delta} covered by {line}"
        for tier, delta, line in matches
        if line is not None
    ]

    message = "\n".join([*accepted, *failures]) + "\n"
    return int(bool(failures)), message


def _git(repo: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", repo, *args], text=True, stderr=subprocess.PIPE
    ).strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base branch reference")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--baseline", default=".github/collect-baseline")
    args = parser.parse_args(argv)
    try:
        base_text = _git(args.repo, "show", f"{args.base}:{args.baseline}")
        head_text = (Path(args.repo) / args.baseline).read_text(encoding="utf-8")
        code, message = check_increases(
            _baseline(base_text), _baseline(head_text), os.environ.get("GITHUB_PR_BODY", "")
        )
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"could not compare collected-item baselines: {error}", file=sys.stderr)
        return 2
    print(message, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
