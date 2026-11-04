"""Regenerate the preflight refusal hazards that every pull request keeps one cell for.

Runs every `test_runnable_preserves_preflight_refusals` cell, exhaustive ones included, and
records each distinct hazard -- a refusal code with the keys of its context, raised by one
ingress at one stage -- with the first cell in matrix order that raises it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "tests/test_parity_matrix.py::test_runnable_preserves_preflight_refusals"


def observed_hazards(junit: Path) -> dict[str, list[tuple[str, str, tuple[str, ...], str]]]:
    """Each cell's recorded ``refusal_hazards`` property, keyed by its parameter id."""
    observed = {}
    for case in ET.parse(junit).iter("testcase"):
        name = case.get("name", "")
        prefix = "test_runnable_preserves_preflight_refusals["
        if not name.startswith(prefix):
            continue
        for prop in case.iter("property"):
            if prop.get("name") == "refusal_hazards":
                observed[name.removeprefix(prefix).removesuffix("]")] = [
                    (ingress, code, tuple(context), stage)
                    for ingress, code, context, stage in json.loads(prop.get("value", "[]"))
                ]
    return observed


def hazard_record(
    order: list[str], observed: dict[str, list[tuple[str, str, tuple[str, ...], str]]]
) -> dict[str, list[dict[str, object]]]:
    """One entry per distinct hazard, represented by the first cell in *order* raising it."""
    first: dict[tuple[str, str, tuple[str, ...], str], str] = {}
    for cell in order:
        for hazard in sorted(observed[cell]):
            first.setdefault(hazard, cell)
    return {
        "hazards": [
            {
                "cell": cell,
                "ingress": ingress,
                "code": code,
                "context": list(context),
                "stage": stage,
            }
            for (ingress, code, context, stage), cell in first.items()
        ]
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="auto", help="pytest-xdist workers")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from tests.parity_harness import matrix

    order = [f"{cell.id}-{ingress}" for cell, ingress in matrix.preflight_refusal_groups()]
    with tempfile.TemporaryDirectory(prefix="preflight-hazards-") as scratch:
        junit = Path(scratch) / "junit.xml"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                TEST,
                "-m",
                "",
                "-q",
                "--tb=no",
                "-n",
                args.workers,
                "--dist",
                "loadgroup",
                f"--junitxml={junit}",
            ],
            cwd=ROOT,
            env={**os.environ, "INCREMENT_PARITY_MATRIX": "full"},
            check=False,
        )
        observed = observed_hazards(junit)
    missing = [cell for cell in order if cell not in observed]
    if missing:
        print(f"{len(missing)} cells recorded no hazards, first {missing[0]}", file=sys.stderr)
        return 1
    record = hazard_record(order, observed)
    matrix.PREFLIGHT_HAZARDS.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"{len(record['hazards'])} hazards from {len(order)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
