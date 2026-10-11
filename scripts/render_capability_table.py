"""Render the three evidence-backed capability tables in the reference page."""

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


def _display_text(display: CellDisplay) -> str:
    text = display.status.replace("_", " ")
    details = ([display.classification] if display.classification else []) + [
        f"`{code}`" for code in display.codes
    ]
    if details:
        text += " (" + ", ".join(details) + ")"
    if display.reason:
        text += " — " + display.reason
    if display.tracker:
        text += f" — tracked as `{display.tracker}`"
    return text


def _cell_text(row: EntryRow, ingress: str) -> str:
    clauses = resolve_cell(row.bindings[ingress])
    rendered = []
    for condition, display in clauses:
        value = _display_text(display)
        rendered.append(f"{condition}: {value}" if condition else value)
    note = row.notes.get(ingress)
    if note:
        rendered.append(note)
    return "; ".join(rendered).replace("|", "\\|")


def _render_table(table: str, rows: Sequence[EntryRow]) -> str:
    columns = TABLE_COLUMNS[table]
    headers = TABLE_HEADERS[table]
    first_header = "Read" if table == "triggered-auxiliary" else "Capability"
    lines = [
        "| " + " | ".join((first_header, *headers)) + " |",
        "|" + "|".join("---" for _ in range(len(headers) + 1)) + "|",
    ]
    for row in rows:
        cells = [row.label, *(_cell_text(row, column) for column in columns)]
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in cells) + " |")
    for row in rows:
        if row.family_clauses:
            lines.extend(("", f"**{row.label}: family-owned operations**"))
            for condition, display in resolve_cell(row.family_clauses):
                value = _display_text(display)
                lines.append(f"- {condition}: {value}" if condition else f"- {value}")
    return "\n".join(lines)


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
