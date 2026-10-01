"""Run complete official test entry points with local wall-clock budgets."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass

from scripts.run_test_tier import run_with_budget


@dataclass(frozen=True, slots=True)
class Entrypoint:
    command: tuple[str, ...]
    budget_seconds: int
    cleanup_grace_seconds: int


ENTRYPOINTS = {
    "examples": Entrypoint(
        command=("uv", "run", "--with", "nox", "nox", "-s", "examples"),
        budget_seconds=300,
        cleanup_grace_seconds=10,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", choices=tuple(ENTRYPOINTS))
    args = parser.parse_args(argv)
    entrypoint = ENTRYPOINTS[args.entrypoint]
    return run_with_budget(
        entrypoint.command,
        tier=f"{args.entrypoint}-entrypoint",
        budget_seconds=(
            None if os.environ.get("GITHUB_ACTIONS") == "true" else entrypoint.budget_seconds
        ),
        grace_seconds=entrypoint.cleanup_grace_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
