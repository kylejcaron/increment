"""Render capability indexes and records in three delimited reference regions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from tests.capability_entry_point_catalog import (
    ENTRY_ROWS,
    TABLE_COLUMNS,
    TABLE_HEADERS,
    CellDisplay,
    EntryRow,
    resolve_cell,
    validate_catalog,
)

PAGE = Path("docs/reference/capabilities-by-entry-point.md")
TABLES = (
    (
        "ingresses",
        "<!-- capability-table:ingresses:start -->",
        "<!-- capability-table:ingresses:end -->",
    ),
    (
        "triggered-auxiliary",
        "<!-- capability-table:triggered-auxiliary:start -->",
        "<!-- capability-table:triggered-auxiliary:end -->",
    ),
    (
        "stored-state",
        "<!-- capability-table:stored-state:start -->",
        "<!-- capability-table:stored-state:end -->",
    ),
)


def _record_anchor(row: EntryRow, ingress: str) -> str:
    return f"capability-{row.table}-{row.id}-{ingress}"


def _cell_text(row: EntryRow, ingress: str) -> str:
    clauses = resolve_cell(row.bindings[ingress])
    statuses = dict.fromkeys(display.status.replace("_", " ") for _, display in clauses)
    label = " / ".join(statuses)
    if len(clauses) > 1 or any(condition for condition, _ in clauses):
        label += " (conditional)"
    return f"[{label}](#{_record_anchor(row, ingress)})"


def _record_lines(clauses: Sequence[tuple[str, CellDisplay]]) -> list[str]:
    lines = []
    for condition, display in clauses:
        if condition:
            lines.extend((f"**Condition:** {condition}", ""))
        lines.extend((f"**Status:** {display.status.replace('_', ' ')}", ""))
        fields = (
            ("Classification", display.classification),
            ("Codes", ", ".join(f"`{code}`" for code in display.codes)),
            ("Reason", display.reason),
            ("Operation", f"`{display.operation}`" if display.operation else ""),
            ("Stage", display.stage),
            ("Tracked issue", f"`{display.tracker}`" if display.tracker else ""),
        )
        for label, value in fields:
            if value:
                lines.extend((f"**{label}:** {value}", ""))
    return lines


def _render_table(table: str, rows: Sequence[EntryRow]) -> str:
    columns = TABLE_COLUMNS[table]
    headers = TABLE_HEADERS[table]
    first_header = "Read" if table == "triggered-auxiliary" else "Capability"
    lines = [
        "Select a status for its conditions, codes and reason. "
        "A conditional cell is not a claim about every request.",
        "",
        "| " + " | ".join((first_header, *headers)) + " |",
        "|" + "|".join("---" for _ in range(len(headers) + 1)) + "|",
    ]
    for row in rows:
        cells = [row.label, *(_cell_text(row, column) for column in columns)]
        lines.append(
            "| "
            + " | ".join(value.replace("\n", " ").replace("|", "\\|") for value in cells)
            + " |"
        )
    for row in rows:
        lines.extend(
            ("", f'<span id="capability-{table}-{row.id}"></span>', "", f"### {row.label}", "")
        )
        for column, header in zip(columns, headers, strict=True):
            lines.extend(
                (
                    f'<span id="{_record_anchor(row, column)}"></span>',
                    "",
                    f"#### {header}",
                    "",
                    f"**Recorded path:** `{column}`",
                    "",
                    *_record_lines(resolve_cell(row.bindings[column])),
                )
            )
            if note := row.notes.get(column):
                lines.extend((f"**Note:** {note}", ""))
        if row.family_clauses:
            lines.extend(
                (
                    f'<span id="{_record_anchor(row, "family")}"></span>',
                    "",
                    f"#### {row.label}: family-owned operations",
                    "",
                    *_record_lines(resolve_cell(row.family_clauses)),
                )
            )
    return "\n".join(lines).rstrip()


def render_document(existing: str, rows: Sequence[EntryRow]) -> str:
    """Replace the three uniquely delimited generated regions, preserving prose."""
    updated = existing
    for table, start, end in TABLES:
        starts = updated.count(start)
        ends = updated.count(end)
        if starts != 1 or ends != 1:
            raise ValueError(
                f"{table}: expected one pair of generated markers; found {starts}/{ends}"
            )
        left = updated.index(start)
        right = updated.index(end)
        if right < left:
            raise ValueError(f"{table}: generated markers are reversed")
        selected = tuple(row for row in rows if row.table == table)
        content = "\n" + _render_table(table, selected) + "\n"
        updated = updated[: left + len(start)] + content + updated[right:]
    return updated


def check_document(path: Path = PAGE) -> bool:
    validate_catalog()
    existing = path.read_text(encoding="utf-8")
    return render_document(existing, ENTRY_ROWS) == existing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="return 1 when generated tables are stale"
    )
    parser.add_argument("path", nargs="?", type=Path, default=PAGE)
    args = parser.parse_args(argv)
    validate_catalog()
    current = args.path.read_text(encoding="utf-8")
    rendered = render_document(current, ENTRY_ROWS)
    if args.check:
        if rendered != current:
            print(f"{args.path}: capability tables are stale")
            return 1
        return 0
    args.path.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
