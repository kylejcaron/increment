"""Measure how many fast-tier test items a one-file change selects."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple

from scripts import affected
from scripts.affected import Selection, select


class ProbeRow(NamedTuple):
    module: str
    selected_items: int
    fast_items: int
    direct_files: int
    tach_items: int = 0
    wall_seconds: float = 0.0


DEFAULT_MODULES = (
    "increment/estimation/engine.py",
    "increment/estimation/_adjust/iptw.py",
    "increment/estimation/cuped/multicovariate.py",
    "increment/power/_binomial.py",
    "increment/query/_cuped_qr.py",
    "increment/_lift_options.py",
    "increment/semantics/models.py",
)


_COLLECTED = re.compile(r"(\d+)/(\d+) tests collected")


def _parse_collected(line: str) -> tuple[int, int]:
    match = _COLLECTED.search(line)
    if match is None:
        raise ValueError(f"could not parse pytest collection summary: {line!r}")
    return int(match.group(1)), int(match.group(2))


def _collect(root: Path, arguments: Sequence[str]) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *arguments, "--co", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if "::" in line]


def _component_counts(root: Path, selection: Selection, base: str) -> tuple[int, int, int, int]:
    marker = ["-m", affected._changed_markers("fast", root)]
    direct_files = len(selection.test_files)
    tach_items = 0
    node_ids: set[str] = set()
    for invocation in affected._pytest_invocations(selection, base, "fast", []):
        collected = _collect(root, invocation)
        node_ids.update(collected)
        if "--tach-base" in invocation:
            tach_items = len(set(collected))
    fast_items = len(set(_collect(root, marker)))
    return len(node_ids), fast_items, direct_files, tach_items


def _scratch_commit_command() -> list[str]:
    return [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "user.name=Selection Probe",
        "-c",
        "user.email=selection-probe@example.invalid",
        "commit",
        "--no-verify",
        "-m",
        "selection probe",
    ]


def _run_one(root: Path, module: str, *, timing: bool = True) -> ProbeRow:
    with tempfile.TemporaryDirectory(prefix="selection-probe-") as temporary:
        scratch = Path(temporary) / "checkout"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(scratch), "HEAD"], cwd=root, check=True
        )
        try:
            target = scratch / module
            with target.open("a", encoding="utf-8") as stream:
                stream.write("\n# selection probe\n")
            subprocess.run(["git", "add", module], cwd=scratch, check=True)
            subprocess.run(
                _scratch_commit_command(),
                cwd=scratch,
                check=True,
                capture_output=True,
                text=True,
            )
            selection = select("HEAD~1", scratch)
            selected, fast, direct, tach = _component_counts(scratch, selection, "HEAD~1")
            wall_seconds = 0.0
            if timing:
                started = time.perf_counter()
                subprocess.run(["make", "test-affected", "BASE=HEAD~1"], cwd=scratch, check=True)
                wall_seconds = time.perf_counter() - started
            return ProbeRow(module, selected, fast, direct, tach, wall_seconds)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(scratch)], cwd=root, check=True
            )


def probe(
    root: Path,
    modules: Sequence[str],
    *,
    runner: Callable[[Path, str], ProbeRow] = _run_one,
) -> list[ProbeRow]:
    return [runner(root, module) for module in modules]


def _format(rows: Sequence[ProbeRow]) -> str:
    header = "| Module | Selected items | Fast items | Direct files | Tach items | Wall seconds |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    body = [
        f"| `{row.module}` | {row.selected_items} | {row.fast_items} | {row.direct_files} | {row.tach_items} | {row.wall_seconds:.2f} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", action="store_true", help="append the table to GitHub step summary"
    )
    parser.add_argument("--json", type=Path, metavar="PATH", help="write rows as JSON")
    parser.add_argument(
        "--no-timing", action="store_true", help="skip affected-test execution timing"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent

    def runner(checkout: Path, module: str) -> ProbeRow:
        return _run_one(checkout, module, timing=not args.no_timing)

    rows = probe(root, DEFAULT_MODULES, runner=runner)
    table = _format(rows)
    print(table)
    if args.summary:
        summary = Path(__import__("os").environ["GITHUB_STEP_SUMMARY"])
        with summary.open("a", encoding="utf-8") as stream:
            stream.write(table + "\n")
    if args.json:
        args.json.write_text(
            json.dumps([row._asdict() for row in rows], indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
