from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts import render_capability_table as renderer
from tests.capability_entry_point_catalog import (
    INGRESSES,
    TABLE_COLUMNS,
    CaseBinding,
    CellClause,
    EntryRow,
    Table,
    UnmeasuredBinding,
)


def _document() -> str:
    return """lead\n<!-- capability-table:ingresses:start -->\nold\n<!-- capability-table:ingresses:end -->\naux\n<!-- capability-table:triggered-auxiliary:start -->\nold aux\n<!-- capability-table:triggered-auxiliary:end -->\nstored\n<!-- capability-table:stored-state:start -->\nold stored\n<!-- capability-table:stored-state:end -->\ntrail\n"""


def _row(row_id: str, table: Table, clauses: tuple[CellClause, ...]) -> EntryRow:
    return EntryRow(
        row_id,
        row_id,
        table,
        dict.fromkeys(TABLE_COLUMNS[table], clauses),
        {},
    )


def test_render_replaces_all_three_regions_and_preserves_authored_text() -> None:
    rows = (
        _row("second", "ingresses", (CellClause("", UnmeasuredBinding("explicit")),)),
        _row("first", "ingresses", (CellClause("", UnmeasuredBinding("explicit")),)),
        _row("auxiliary", "triggered-auxiliary", (CellClause("", UnmeasuredBinding("explicit")),)),
        _row("stored", "stored-state", (CellClause("", UnmeasuredBinding("explicit")),)),
    )
    rendered = renderer.render_document(_document(), rows)
    markers = (
        ("<!-- capability-table:ingresses:start -->", "<!-- capability-table:ingresses:end -->"),
        (
            "<!-- capability-table:triggered-auxiliary:start -->",
            "<!-- capability-table:triggered-auxiliary:end -->",
        ),
        (
            "<!-- capability-table:stored-state:start -->",
            "<!-- capability-table:stored-state:end -->",
        ),
    )
    outside = _document()
    for start, end in markers:
        before, remainder = outside.split(start)
        _, after = remainder.split(end)
        outside = before + start + end + after
    assert rendered.startswith("lead\n")
    preserved = rendered
    for start, end in markers:
        before, remainder = preserved.split(start)
        _, after = remainder.split(end)
        preserved = before + start + end + after
    assert preserved == outside
    assert rendered.index("second") < rendered.index("first")
    assert all(label in rendered for label in ("auxiliary", "stored"))
    assert "old aux" not in rendered and "old stored" not in rendered


def test_duplicate_or_missing_markers_are_errors() -> None:
    with pytest.raises(ValueError):
        renderer.render_document("no markers", ())
    duplicate = _document().replace(
        "<!-- capability-table:ingresses:start -->",
        "<!-- capability-table:ingresses:start --><!-- capability-table:ingresses:start -->",
    )
    with pytest.raises(ValueError):
        renderer.render_document(duplicate, ())


def test_changed_composite_constituent_changes_only_its_clause() -> None:
    baseline = (
        CellClause(
            "ITT present",
            CaseBinding("encouragement_margin_guardrail", INGRESSES[0]),
        ),
        CellClause(
            "ITT excluded",
            CaseBinding("encouragement_margin_guardrail_late_only", INGRESSES[0]),
        ),
    )
    changed = (
        baseline[0],
        CellClause(
            "ITT excluded",
            CaseBinding("encouragement_guardrail_late_only_request", INGRESSES[0]),
        ),
    )
    before = renderer._cell_text(_row("composite", "ingresses", baseline), INGRESSES[0])
    after = renderer._cell_text(_row("composite", "ingresses", changed), INGRESSES[0])
    before_parts = before.split("; ")
    after_parts = after.split("; ")
    assert before_parts[0] == after_parts[0]
    assert before_parts[1] != after_parts[1]
    assert "`readout.encouragement.margin`" in before_parts[1]
    assert "`readout.encouragement.margin`" not in after_parts[1]


def test_all_tables_have_their_declared_column_shapes() -> None:
    rows = tuple(
        EntryRow(
            table,
            table,
            table,
            {
                column: (CellClause("", UnmeasuredBinding(column)),)
                for column in TABLE_COLUMNS[table]
            },
            {},
        )
        for table in ("ingresses", "triggered-auxiliary", "stored-state")
    )
    rendered = renderer.render_document(_document(), rows)
    for table in ("ingresses", "triggered-auxiliary", "stored-state"):
        start, end = next((start, end) for name, start, end in renderer.TABLES if name == table)
        _, after_start = rendered.split(start)
        table_body, _ = after_start.split(end)
        table_lines = [line for line in table_body.splitlines() if line.startswith("|")]
        data_cells = [cell.strip() for cell in table_lines[2].strip("|").split("|")]
        assert data_cells[0] == table
        assert all(
            column in data_cells[index]
            for index, column in enumerate(TABLE_COLUMNS[table], start=1)
        )


@pytest.mark.integration
def test_renderer_import_does_not_load_scenarios_or_builders() -> None:
    code = """
import builtins

original_import = builtins.__import__
blocked = (
    "tests.parity_harness.cases",
    "tests.parity_harness.runner",
    "tests.parity_harness.scalar_capability_cases",
    "tests.parity_harness.auxiliary_capability_cases",
    "increment.builders",
)

def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in blocked):
        raise AssertionError(f"static renderer imported {name}")
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import scripts.render_capability_table
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_stale_generated_block_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "reference.md"
    path.write_text(_document(), encoding="utf-8")
    assert not renderer.check_document(path)
    assert renderer.main(["--check", str(path)]) == 1


def test_family_persistence_is_disclosed_without_claiming_analysis_support() -> None:
    row = next(row for row in renderer.ENTRY_ROWS if row.id == "tabular-policy-persistence")
    rendered = renderer.render_document(_document(), (row,))
    table_body, family_body = rendered.split("**", 1)
    data_line = next(
        line for line in table_body.splitlines() if line.startswith(f"| {row.label} |")
    )
    cells = [cell.strip() for cell in data_line.strip("|").split("|")]
    assert all("not comparable" in cell for cell in cells[1:])
    assert "logged_policy.policy.json_context_key" not in data_line
    assert "`logged_policy.policy.json_context_key`" in family_body
    assert "supported" in family_body and "refused" in family_body
