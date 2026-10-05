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
keyword the ingress cannot express, with the declared field named by the error), so they run
in the fast tier. Every cell builds and runs each of its ingresses itself: no result is
shared between cells, so a cell never depends on test order or on a sibling having run.
"""

from __future__ import annotations

import pytest

from tests.parity_harness import matrix, matrix_cases
from tests.parity_harness.cases import ParityCase
from tests.parity_harness.runner import CaseResult, assert_parity, run_case

# Advisories (a dropped-row count, a small-K cluster note) are not part of a cell's contract.
pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")


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
        marks = [pytest.mark.slow] if _reads_data(disposition) else []
        params.append(pytest.param(cell, id=cell.id, marks=marks))
    return params


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
    result = run_case(case)
    _assert_runners_produced_rows(case, outcomes, result)
    assert_parity(case, result)


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
    seam = "readout.observational.quantile"
    warehouse = ("from_definitions", "from_unit_day_artifact")
    per_unit = (*warehouse, "from_unit_summary", "from_unit_panel")
    for cell in matrix.iter_cells():
        if not (
            cell.base == "quantile"
            and cell.option == "observational"
            and cell.view == "run"
            and cell.missing in ("error", "zero")
        ):
            continue
        verdicts = matrix.classify(cell).legs["rows"]
        reaching = warehouse if cell.windowed else per_unit
        for name in reaching:
            assert verdicts[name].outcome == matrix.Refuses(seam), (
                f"{cell.id}: {name} -> {verdicts[name].outcome}"
            )
            assert verdicts[name].tracker is None, f"{cell.id}: {name} is tracked as unfinished"
        if not cell.windowed:
            assert verdicts["from_moments"].outcome == matrix.Refuses(seam)
            assert verdicts["from_moments"].status == "construction_limited"


@pytest.mark.parametrize("cell", _params())
def test_cell(cell: matrix.Cell) -> None:
    """Each leg of the cell (a day-axis view has a value leg and a lift leg) runs on its own,
    so one leg refusing never hides the other's rows."""
    disposition = matrix.classify(cell)
    for method in disposition.legs:
        _matched(cell, method, disposition)
        _switchback(cell, method, disposition)
