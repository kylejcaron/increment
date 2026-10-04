"""Every cell of the enumerated parity matrix, on every ingress.

``tests/parity_harness/matrix.py`` names, for each cell of metric x view x option x
day_boundary x missing, what each of the six ingress constructors must do: run, refuse with a
recorded code, or be structurally unable to express the request. The five matched-arm
ingresses that run are compared with each other within 1e-9 relative;
``from_switchback_panel`` (a different estimand) is run on its own, with its rows required to
exist but never compared. ``test_cell`` builds the cell on each ingress, runs it, and asserts
exactly that, so a stale refusal code, a refusal that became a number, a number that became a
refusal and a numeric disagreement all fail.

Cells in which any ingress builds and reads data (an ingress runs, or refuses only once a
request is read) are ``slow``. In the remaining cells no ingress runs or reaches the request
stage: each declaration either refuses with a code or is structurally absent (a schema or
keyword the ingress cannot express), so they run in the fast tier. A warehouse route is
memoised per process on its exact inputs, and the cells that share those inputs share an
``xdist_group``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.parity_harness import matrix, matrix_cases
from tests.parity_harness.cases import ParityCase
from tests.parity_harness.runner import CaseResult, assert_parity, run_case

# Advisories (a dropped-row count, a small-K cluster note) are not part of a cell's contract.
pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")

_MEMO: dict[tuple, CaseResult] = {}


def _reads_data(disposition: matrix.Disposition) -> bool:
    """Some ingress runs, or refuses only once a request is read."""
    return any(
        isinstance(v.outcome, matrix.Runs)
        or (isinstance(v.outcome, matrix.Refuses) and v.stage == "request")
        for v in disposition.all_verdicts
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


def _assert_runners_produced_rows(
    case: ParityCase, outcomes: dict[str, matrix.Outcome], result: CaseResult
) -> None:
    """`assert_parity` treats a constructor that returned nothing as live and self-agreeing,
    so a `Runs` verdict is held to its promise here: the ingress produced rows."""
    for name in case.build:
        if isinstance(outcomes[name], matrix.Runs):
            assert result.rows.get(name), (
                f"{case.id}: {name} is classified Runs but emitted no rows"
            )


def _matched(cell: matrix.Cell, method: str, disposition: matrix.Disposition) -> None:
    outcomes = disposition.outcomes(method)
    case = matrix_cases.build_case(cell, method, outcomes)
    parts = [_run_one(cell, method, case, name) for name in case.build]
    merged = CaseResult(
        rows={n: r for part in parts for n, r in part.rows.items()},
        refusals={n: c for part in parts for n, c in part.refusals.items()},
        sequential_state={n: s for part in parts for n, s in part.sequential_state.items()},
        absences={n: e for part in parts for n, e in part.absences.items()},
    )
    _assert_runners_produced_rows(case, outcomes, merged)
    assert_parity(case, merged)


def _switchback(cell: matrix.Cell, method: str, disposition: matrix.Disposition) -> None:
    outcomes = disposition.outcomes(method)
    case = matrix_cases.build_switchback_case(cell, method, outcomes)
    result = run_case(case)
    _assert_runners_produced_rows(case, outcomes, result)
    assert_parity(case, result)


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


def test_observational_quantile_is_refused_with_one_code_on_every_ingress_that_reads_it():
    """An observational design has no quantile estimator, so every matched-arm ingress that
    can declare the request refuses it with the shared readout-seam code, never a number and
    never a route-specific code. A windowed quantile reaches that seam on the warehouse
    routes only; the frame routes refuse its window declaration first."""
    seam = "source.frame.quantile_no_moments"
    warehouse = ("from_definitions", "from_unit_day_artifact")
    for cell in matrix.iter_cells():
        if not (
            cell.base == "quantile"
            and cell.option == "observational"
            and cell.view == "run"
            and cell.missing in ("error", "zero")
        ):
            continue
        verdicts = matrix.classify(cell).legs["rows"]
        reaching = warehouse if cell.windowed else matrix_cases.MATCHED
        for name in reaching:
            assert verdicts[name].outcome == matrix.Refuses(seam), (
                f"{cell.id}: {name} -> {verdicts[name].outcome}"
            )
            assert verdicts[name].tracker is None, f"{cell.id}: {name} is tracked as unfinished"


@pytest.mark.parametrize("cell", _params())
def test_cell(cell: matrix.Cell) -> None:
    """Each leg of the cell (a day-axis view has a value leg and a lift leg) runs on its own,
    so one leg refusing never hides the other's rows."""
    disposition = matrix.classify(cell)
    for method in disposition.legs:
        _matched(cell, method, disposition)
        _switchback(cell, method, disposition)
