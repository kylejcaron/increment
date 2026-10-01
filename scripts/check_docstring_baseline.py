"""Fail unless pydoclint's checked-in baseline lists exactly today's mismatches.

pydoclint itself fails only on new mismatches, so a fixed one would stay in the
baseline and could silently return. Regenerating into a temporary file and
comparing catches both directions.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE = Path("pydoclint-baseline.txt")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        fresh = Path(directory) / BASELINE.name
        subprocess.run(
            ["pydoclint", "--generate-baseline=True", f"--baseline={fresh}", "increment"],
            check=True,
            capture_output=True,
        )
        expected = BASELINE.read_text().splitlines(keepends=True)
        actual = fresh.read_text().splitlines(keepends=True)
    if expected == actual:
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(expected, actual, str(BASELINE), "current mismatches")
    )
    print(
        f"\n{BASELINE} is out of date: fix each '+' mismatch in its docstring, "
        "and delete each '-' line whose mismatch is already fixed."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
