from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import replace
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
    resolve_cell,
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


def _records(document: str) -> dict[str, str]:
    return {
        match.group(1): section
        for section in re.split(r'(?=^<span id=")', document, flags=re.MULTILINE)
        if (match := re.match(r'<span id="([^"]+)"></span>', section))
    }


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
    before = _records(
        renderer.render_document(_document(), (_row("composite", "ingresses", baseline),))
    )
    after = _records(
        renderer.render_document(_document(), (_row("composite", "ingresses", changed),))
    )
    for anchor, record in before.items():
        if "ITT present" not in record:
            continue
        before_parts = record.split("ITT excluded")
        after_parts = after[anchor].split("ITT excluded")
        assert before_parts[0] == after_parts[0]
        assert before_parts[1] != after_parts[1]
        assert "`readout.encouragement.margin`" in before_parts[1]
        assert "`readout.encouragement.margin`" not in after_parts[1]
    assert any("ITT present" in record for record in before.values())


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


def test_linked_records_keep_distinct_paths_and_all_conditional_outcomes() -> None:
    rows = tuple(
        replace(
            _row(row_id, "ingresses", ()),
            label="Same capability name",
            bindings={
                ingress: (
                    CellClause(
                        "ITT excluded",
                        CaseBinding("encouragement_margin_guardrail_late_only", ingress),
                    ),
                    CellClause("Unmeasured request", UnmeasuredBinding(f"{row_id}: {ingress}")),
                )
                for ingress in INGRESSES
            },
        )
        for row_id in ("first-path", "second-path")
    )
    rendered = renderer.render_document(_document(), rows)
    records = _records(rendered)
    table_rows = [line for line in rendered.splitlines() if line.startswith("| Same capability")]
    targets = []
    for row, table_row in zip(rows, table_rows, strict=True):
        links = re.findall(r"\]\(#([^)]+)\)", table_row)
        assert len(links) == len(INGRESSES)
        targets.extend(links)
        for ingress, anchor in zip(INGRESSES, links, strict=True):
            record = records[anchor]
            assert ingress in record
            assert row.label in record
            for condition, display in resolve_cell(row.bindings[ingress]):
                assert condition in record
                assert display.status.replace("_", " ") in record
                assert display.classification in record
                assert display.reason in record
                assert all(f"`{code}`" in record for code in display.codes)
    assert len(targets) == len(set(targets)) == len(rows) * len(INGRESSES)


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
    family_body = next(
        record for record in _records(rendered).values() if "family-owned operations" in record
    )
    data_line = next(line for line in rendered.splitlines() if line.startswith(f"| {row.label} |"))
    cells = [cell.strip() for cell in data_line.strip("|").split("|")]
    assert all("not comparable" in cell for cell in cells[1:])
    assert "logged_policy.policy.json_context_key" not in data_line
    assert "`logged_policy.policy.json_context_key`" in family_body
    assert "supported" in family_body and "refused" in family_body
