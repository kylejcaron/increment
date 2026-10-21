"""Build a unit CPU baseline from repeated per-test process CPU samples."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path

HALF_BUDGET_SECONDS = 0.05


def baseline_nodeids(samples: Iterable[Mapping[str, float]]) -> set[str]:
    """Select tests exceeding half the enforcement budget in any run."""
    return {
        nodeid
        for sample in samples
        for nodeid, seconds in sample.items()
        if seconds > HALF_BUDGET_SECONDS
    }


def _read_sample(path: Path) -> dict[str, float]:
    sample: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        nodeid, value = line.rsplit("\t", 1)
        sample[nodeid] = max(sample.get(nodeid, 0.0), float(value))
    return sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tests/unit_cpu_baseline.txt"))
    args = parser.parse_args(argv)
    if len(args.samples) < 3:
        parser.error("at least three full-run CPU samples are required")
    nodeids = baseline_nodeids(_read_sample(path) for path in args.samples)
    content = "# Tests above 50 ms in at least one of three or more full-run CPU samples.\n"
    content += "".join(f"{nodeid}\n" for nodeid in sorted(nodeids))
    args.output.write_text(content, encoding="utf-8")
    print(f"wrote {len(nodeids)} baseline node IDs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
