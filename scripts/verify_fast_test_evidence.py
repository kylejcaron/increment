"""Check that every selected fast test has a terminal, non-failing report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast


def verify_fast_test_evidence(evidence_root: Path) -> tuple[bool, str]:
    """Distinguish test failures from optional pytest-cov finalization failures."""
    runs = sorted(evidence_root.glob("run-*/run.json"))
    if len(runs) != 1:
        return False, f"expected one completed fast evidence run, found {len(runs)}"
    run_file = runs[0]
    try:
        run = json.loads(run_file.read_text())
        selected = run["collected_nodeids"]
        if (
            run["status"] != "finished"
            or not isinstance(selected, list)
            or not selected
            or any(not isinstance(nodeid, str) for nodeid in selected)
            or len(set(selected)) != len(selected)
            or run.get("collected_tests") != len(selected)
        ):
            return False, "fast evidence has no complete nonempty collected-test inventory"
        events = [
            json.loads(line)
            for line in run_file.with_name("reports.jsonl").read_text().splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return False, "fast evidence is missing or malformed"
    collected = cast(list[str], selected)

    reports = [event for event in events if event.get("event") == "report"]
    failed = sorted(
        {event.get("nodeid", "?") for event in reports if event.get("outcome") == "failed"}
    )
    if failed:
        return False, f"pytest reported failed tests: {', '.join(failed)}"

    terminal: dict[str, bool] = {}
    for event in reports:
        nodeid = event.get("nodeid")
        if not isinstance(nodeid, str) or nodeid not in collected:
            continue
        if event.get("when") == "call" and event.get("outcome") in {"passed", "skipped"}:
            terminal[nodeid] = True
        elif event.get("when") == "setup" and event.get("outcome") == "skipped":
            terminal[nodeid] = True
    missing = sorted(set(collected) - terminal.keys())
    if missing or run.get("executed_tests") != len(collected):
        return (
            False,
            f"fast test execution is incomplete ({len(missing)} selected tests lack terminal reports)",
        )
    return True, "All selected fast tests have terminal, non-failing reports."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    passed, message = verify_fast_test_evidence(arguments.evidence_root)
    print(message)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
