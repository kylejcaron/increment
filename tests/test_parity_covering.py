from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.parity_harness import covering, matrix

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.integration

AXES = (
    matrix.METRICS,
    matrix.VIEWS,
    matrix.OPTIONS,
    matrix.DAY_BOUNDARIES,
    matrix.MISSING,
)


def _runs_on(cell: matrix.Cell, method: str) -> bool:
    return any(
        isinstance(outcome, matrix.Runs)
        for outcome in matrix.classify(cell).outcomes(method).values()
    )


def test_pr_subset_selects_the_cheapest_cell_per_capability_and_keeps_hazards() -> None:
    durations = json.loads((ROOT / ".github/slow-test-durations.json").read_text())
    cells = tuple(matrix.iter_cells())
    by_capability: dict[tuple[str, str, str], list[matrix.Cell]] = {}
    for cell in cells:
        for method in matrix.methods(cell):
            if _runs_on(cell, method):
                by_capability.setdefault((cell.metric, method, cell.option), []).append(cell)

    def duration(cell: matrix.Cell) -> float:
        return durations.get(f"tests/test_parity_matrix.py::test_cell[{cell.id}]", float("inf"))

    hazard_cell_ids = {hazard.cell.rsplit("-", 1)[0] for hazard in matrix.preflight_hazards()}
    expected = {min(candidates, key=duration) for candidates in by_capability.values()} | {
        cell for cell in cells if cell.id in hazard_cell_ids
    }
    assert set(matrix.pr_matrix_cells()) == expected


def test_pr_subset_covers_every_supported_matrix_capability_with_a_running_cell() -> None:
    supported_capabilities = {
        (cell.metric, method, cell.option)
        for cell in matrix.iter_cells()
        for method in matrix.methods(cell)
        if _runs_on(cell, method)
    }
    pr_running_capabilities = {
        (cell.metric, method, cell.option)
        for cell in matrix.pr_matrix_cells()
        for method in matrix.methods(cell)
        if _runs_on(cell, method)
    }
    assert supported_capabilities <= pr_running_capabilities

    ratio_cuped_pair = any(
        {"from_definitions", "from_unit_day_artifact"}
        <= {
            ingress
            for ingress, outcome in matrix.classify(cell).outcomes("lift").items()
            if isinstance(outcome, matrix.Runs)
        }
        for cell in matrix.pr_matrix_cells()
        if cell.metric == "ratio" and cell.option == "cuped" and "lift" in matrix.methods(cell)
    )
    assert ratio_cuped_pair


def test_pr_asymptotic_subset_selects_the_cheapest_cell_per_code_path() -> None:
    durations = json.loads((ROOT / ".github/slow-test-durations.json").read_text())
    variants = tuple(matrix.iter_asymptotic_retention_cells())
    by_code_path: dict[tuple[str, str, str], list[matrix.Cell]] = {}
    for cell in variants:
        for method in matrix.methods(cell):
            if _runs_on(cell, method):
                by_code_path.setdefault((cell.metric, method, cell.option), []).append(cell)

    def duration(cell: matrix.Cell) -> float:
        return durations.get(
            f"tests/test_parity_matrix.py::test_asymptotic_retention_cell[{cell.id}]",
            float("inf"),
        )

    expected = {min(candidates, key=duration) for candidates in by_code_path.values()}
    assert set(matrix.pr_asymptotic_retention_cells()) == expected
    assert len(by_code_path) == 3


def test_covering_cells_are_deterministic() -> None:
    first = covering.covering_cells(AXES, strength=3)
    assert covering.covering_cells(AXES, strength=3) == first

    script = (
        "from tests.parity_harness import covering, matrix; "
        "print(repr(covering.covering_cells((matrix.METRICS, matrix.VIEWS, "
        "matrix.OPTIONS, matrix.DAY_BOUNDARIES, matrix.MISSING), strength=3)))"
    )
    environment = {**os.environ, "PYTHONHASHSEED": "8675309"}
    other_process = subprocess.check_output(
        [sys.executable, "-c", script], text=True, env=environment
    )
    assert other_process == f"{first!r}\n"
