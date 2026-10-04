"""Every cell of the enumerated parity matrix, on every ingress.

``tests/parity_harness/matrix.py`` names, for each cell of metric x view x option x
day_boundary x missing, what each of the six ingress constructors must do: run (and agree
with every other running ingress within 1e-9 relative), refuse with a recorded code, or be
structurally unable to express the request. ``test_cell`` builds the cell on each ingress,
runs it, and asserts exactly that, so a stale refusal code, a refusal that became a number,
a number that became a refusal and a numeric disagreement all fail.

Cells in which a warehouse route builds and reads (publishes a unit-day artifact, compiles
the readout) are ``slow``; the remainder refuse before any data is read and run in the fast
tier. A warehouse route is memoised per process on its exact inputs, and the cells that
share those inputs share an ``xdist_group``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.parity_harness import matrix, matrix_cases
from tests.parity_harness.cases import ParityCase
from tests.parity_harness.runner import CaseResult, assert_parity, run_case

# Advisories (a dropped-row count, a small-K cluster note) are not part of a cell's contract.
pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")

_WAREHOUSE = ("from_definitions", "from_unit_day_artifact")
_MEMO: dict[tuple, CaseResult] = {}


def _reads_data(disposition: matrix.Disposition) -> bool:
    """A warehouse route that runs, or refuses only once a request is read."""
    return any(
        isinstance(v.outcome, matrix.Runs)
        or (isinstance(v.outcome, matrix.Refuses) and v.stage == "request")
        for verdicts in disposition.legs.values()
        for name, v in verdicts.items()
        if name in _WAREHOUSE
    )


def _params() -> list:
    params = []
    for cell in matrix.iter_cells():
        disposition = matrix.classify(cell)
        marks = [
            pytest.mark.xdist_group(
                f"parity-matrix-{cell.metric}-{cell.view}-{cell.option}-{cell.day_boundary}"
            )
        ]
        if _reads_data(disposition):
            marks.append(pytest.mark.slow)
        params.append(pytest.param(cell, id=cell.id, marks=marks))
    return params


def _run_one(cell: matrix.Cell, method: str, case: ParityCase, name: str) -> CaseResult:
    single = replace(
        case,
        build={name: case.build[name]},
        waive={n: why for n, why in case.waive.items() if n == name},
        waived_refusal_codes={n: c for n, c in case.waived_refusal_codes.items() if n == name},
        expected_absence={n: e for n, e in case.expected_absence.items() if n == name},
    )
    inputs = matrix_cases.memo_key(cell, method, name)
    if inputs is None:
        return run_case(single)
    key = (*inputs, case.waived_refusal_codes.get(name), case.expected_absence.get(name))
    if key not in _MEMO:
        _MEMO[key] = run_case(single)
    return _MEMO[key]


def _matched(cell: matrix.Cell, method: str, disposition: matrix.Disposition) -> None:
    case = matrix_cases.build_case(cell, method, disposition.outcomes(method))
    parts = [_run_one(cell, method, case, name) for name in case.build]
    merged = CaseResult(
        rows={n: r for part in parts for n, r in part.rows.items()},
        refusals={n: c for part in parts for n, c in part.refusals.items()},
        sequential_state={n: s for part in parts for n, s in part.sequential_state.items()},
        absences={n: e for part in parts for n, e in part.absences.items()},
    )
    assert_parity(case, merged)


def _switchback(cell: matrix.Cell, method: str, disposition: matrix.Disposition) -> None:
    case = matrix_cases.build_switchback_case(cell, method, disposition.outcomes(method))
    assert_parity(case, run_case(case))


def test_every_cell_is_dispositioned_and_its_split_explained():
    """No cell is unclassified, and every number-versus-refusal split or code divergence is
    named with a status, a reason, an authority and, where unfinished, a tracker."""
    count = 0
    for cell in matrix.iter_cells():
        disposition = matrix.classify(cell)
        matrix.check_disposition(cell, disposition)
        count += 1
    assert count == (
        len(matrix.METRICS)
        * len(matrix.VIEWS)
        * len(matrix.OPTIONS)
        * len(matrix.DAY_BOUNDARIES)
        * len(matrix.MISSING)
    )


@pytest.mark.parametrize("cell", _params())
def test_cell(cell: matrix.Cell) -> None:
    """Each leg of the cell (a day-axis view has a value leg and a lift leg) runs on its own,
    so one leg refusing never hides the other's rows."""
    disposition = matrix.classify(cell)
    for method in disposition.legs:
        _matched(cell, method, disposition)
        _switchback(cell, method, disposition)
