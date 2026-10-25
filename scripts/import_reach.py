from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import NamedTuple, Sequence


class ReachRow(NamedTuple):
    entry_point: str
    modules: int | None
    milliseconds: float | None
    loaded: tuple[str, ...]


DEFAULT_ENTRY_POINTS = (
    "increment",
    "increment.errors",
    "increment.semantics",
    "increment._evidence_types",
    "increment.evidence",
    "increment.estimation",
    "increment.estimation.engine",
    "increment.estimation.cuped",
    "increment.estimation.readout_types",
    "increment.power",
    "increment.analysis",
    "increment.declarations",
    "increment.estimators.registry",
)

_CHILD = r'''
import importlib
import json
import sys
import time

name = sys.argv[1]
started = time.perf_counter()
try:
    importlib.import_module(name)
except ModuleNotFoundError:
    print(json.dumps({"modules": None, "milliseconds": None, "loaded": []}))
else:
    elapsed = (time.perf_counter() - started) * 1000
    loaded = sorted(module for module in sys.modules if module.startswith("increment"))
    print(json.dumps({"modules": len(loaded), "milliseconds": elapsed, "loaded": loaded}))
'''


def measure(
    entry_points: Sequence[str], python: str = sys.executable
) -> list[ReachRow]:
    rows = []
    for entry_point in entry_points:
        result = subprocess.run(
            [python, "-c", _CHILD, entry_point],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"Import measurement for {entry_point!r} failed: {result.stderr.strip()}"
            )
        measured = json.loads(result.stdout)
        rows.append(
            ReachRow(
                entry_point,
                measured["modules"],
                measured["milliseconds"],
                tuple(measured["loaded"]),
            )
        )
    return rows


def _markdown(rows: Sequence[ReachRow]) -> str:
    lines = [
        "| Entry point | Modules | Import time (ms) |",
        "| --- | ---: | ---: |",
    ]
    for row in rows:
        if row.modules is None:
            lines.append(f"| `{row.entry_point}` | absent | — |")
        else:
            lines.append(f"| `{row.entry_point}` | {row.modules} | {row.milliseconds:.3f} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure Python import reach.")
    parser.add_argument("entry_points", nargs="*", default=DEFAULT_ENTRY_POINTS)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    rows = measure(args.entry_points)
    table = _markdown(rows)
    print(table)
    for row in rows:
        if row.modules is None:
            print(f"Note: {row.entry_point} is not available; skipped.", file=sys.stderr)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as output:
            json.dump([row._asdict() for row in rows], output, indent=2)
            output.write("\n")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY") if args.summary else None
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(table + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
