from __future__ import annotations

import json
import sys
from pathlib import Path

from tests.benchmarks.test_analysis_benchmark import _CASES


def main() -> int:
    path = Path(sys.argv[1])
    baseline = json.loads(path.read_text())
    names = {entry["name"] for entry in baseline["benchmarks"]}
    expected = {f"test_analysis_end_to_end_benchmark[{case[0]}]" for case in _CASES}
    expected.add("test_population_identity_hashing_benchmark")
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        raise SystemExit(f"benchmark baseline names differ; missing={missing}, extra={extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
