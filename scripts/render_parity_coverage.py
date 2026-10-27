from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from tests.parity_harness import matrix

COVERAGE = Path(__file__).resolve().parents[1] / "tests/parity_harness/COVERAGE.md"
_GENERATED_END = "## Unfinished cells and trackers\n"
_STATUSES = (
    "supported",
    "source_limited",
    "construction_limited",
    "not_expressible",
    "unfinished",
    "unsound",
)


def render_coverage(existing: str) -> str:
    if _GENERATED_END not in existing:
        raise ValueError(f"{COVERAGE} lacks the generated-section boundary")
    cells = tuple(matrix.iter_cells())
    dispositions = [(cell, matrix.classify(cell)) for cell in cells]
    status_counts = Counter(disposition.status for _, disposition in dispositions)
    legs = 0
    verdicts = 0
    verdict_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for _, disposition in dispositions:
        legs += len(disposition.legs)
        verdicts += len(disposition.all_verdicts)
        for ingress in matrix.INGRESSES:
            verdict_counts[ingress].update(
                verdict.status
                for leg in disposition.legs.values()
                for name, verdict in leg.items()
                if name == ingress
            )
    lines = [
        "# Parity matrix coverage",
        "",
        "Counts and trackers only. Every per-ingress outcome, reason and authority lives in",
        "`tests/parity_harness/matrix.py` (`RULES`, `_SPECS`); `tests/test_parity_matrix.py` runs",
        "the PR covering scope by default and all cells when `INCREMENT_PARITY_MATRIX=full`.",
        "",
        "Every count below is generated from `matrix.iter_cells()` and `matrix.classify()`.",
        "",
        "## Axes",
        "",
        "| Axis | Values |",
        "|---|---|",
        f"| metric | {len(matrix.METRICS)}: {', '.join(matrix.METRICS)} |",
        f"| view | {', '.join(matrix.VIEWS)} |",
        f"| option | {', '.join(matrix.OPTIONS)} |",
        f"| day_boundary | {', '.join(matrix.DAY_BOUNDARIES)} |",
        f"| missing | {', '.join(matrix.MISSING)} |",
        "",
        f"Cells: {len(cells):,}. Ingress-cells ({len(matrix.INGRESSES)} constructors each): {len(cells) * len(matrix.INGRESSES):,}.",
        "",
        "## Cells per disposition",
        "",
        "| Status | Cells |",
        "|---|---|",
    ]
    lines.extend(f"| {status} | {status_counts[status]:,} |" for status in _STATUSES)
    lines.extend(
        (
            f"| total | {len(cells):,} |",
            "",
            "## Ingress verdicts per status",
            "",
            f"Verdicts are per leg: {legs:,} legs carry {verdicts:,} ingress verdicts.",
            "",
            "| Ingress | " + " | ".join(_STATUSES) + " |",
            "|---|" + "|".join("---" for _ in _STATUSES) + "|",
        )
    )
    for ingress in matrix.INGRESSES:
        counts = verdict_counts[ingress]
        lines.append(
            "| " + ingress + " | " + " | ".join(f"{counts[s]:,}" for s in _STATUSES) + " |"
        )
    lines.extend(("", "## Cells where one ingress runs and another does not", ""))
    split_cells = sum(
        any(
            len({verdict.status == "supported" for verdict in leg.values()}) > 1
            for leg in disposition.legs.values()
        )
        for _, disposition in dispositions
    )
    lines.append(f"Cells with a mixed run/refusal verdict across ingresses: {split_cells:,}.")
    lines.extend(("", _GENERATED_END.rstrip(), ""))
    return "\n".join(lines) + existing.split(_GENERATED_END, 1)[1]


def check_coverage(path: Path = COVERAGE) -> bool:
    current = path.read_text()
    if _GENERATED_END not in current:
        return False
    return render_coverage(current) == current


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if not check_coverage():
            print("tests/parity_harness/COVERAGE.md is stale")
            return 1
        return 0
    COVERAGE.write_text(render_coverage(COVERAGE.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
