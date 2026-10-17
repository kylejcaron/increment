from __future__ import annotations

import itertools
import os
import subprocess
import sys

import pytest

from tests.parity_harness import covering, matrix

pytestmark = pytest.mark.integration

AXES = (
    matrix.METRICS,
    matrix.VIEWS,
    matrix.OPTIONS,
    matrix.DAY_BOUNDARIES,
    matrix.MISSING,
)


def _values(cell: matrix.Cell) -> tuple[str, ...]:
    return cell.metric, cell.view, cell.option, cell.day_boundary, cell.missing


def _full_cells() -> tuple[tuple[str, ...], ...]:
    return tuple(_values(cell) for cell in matrix.iter_cells())


def _signature(disposition: matrix.Disposition) -> tuple[object, ...]:
    verdicts = tuple(
        (method, ingress, verdict.status)
        for method, leg in disposition.legs.items()
        for ingress, verdict in leg.items()
    )
    refusal_codes = tuple(
        (method, ingress, verdict.outcome.code)
        for method, leg in disposition.legs.items()
        for ingress, verdict in leg.items()
        if isinstance(verdict.outcome, matrix.Refuses)
    )
    waived_ingresses = frozenset(
        ingress
        for leg in disposition.legs.values()
        for ingress, verdict in leg.items()
        if not isinstance(verdict.outcome, matrix.Runs)
    )
    return verdicts, refusal_codes, waived_ingresses


def test_covering_cells_cover_every_three_way_combination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INCREMENT_PARITY_MATRIX", "cover")
    full = _full_cells()
    triples = tuple(itertools.combinations(range(len(AXES)), 3))
    required = {
        (indices, tuple(cell[index] for index in indices)) for indices in triples for cell in full
    }
    covered = {
        (indices, tuple(cell[index] for index in indices))
        for cell in covering.covering_cells(AXES, strength=3)
        for indices in triples
    }
    assert required <= covered

    scoped = {_values(cell) for cell in matrix.parity_matrix_cells()}
    scope_coverage = {
        (indices, tuple(cell[index] for index in indices)) for cell in scoped for indices in triples
    }
    assert required <= scope_coverage


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


def test_pr_scope_keeps_every_disposition(monkeypatch: pytest.MonkeyPatch) -> None:
    full_by_signature: dict[tuple[object, ...], matrix.Cell] = {}
    for cell in matrix.iter_cells():
        full_by_signature.setdefault(_signature(matrix.classify(cell)), cell)

    monkeypatch.setenv("INCREMENT_PARITY_MATRIX", "cover")
    pr_cells = matrix.parity_matrix_cells()
    pr_signatures = {_signature(matrix.classify(cell)) for cell in pr_cells}
    assert set(full_by_signature) <= pr_signatures
