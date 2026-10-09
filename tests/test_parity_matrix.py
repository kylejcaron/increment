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

import itertools
from collections.abc import Iterable
from dataclasses import replace

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


def _params(cells: Iterable[matrix.Cell]) -> list:
    params = []
    for cell in cells:
        disposition = matrix.classify(cell)
        marks = [pytest.mark.slow] if _reads_data(disposition) else []
        params.append(pytest.param(cell, id=cell.id, marks=marks))
    return params


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    cells = {
        "test_cell": matrix.iter_cells,
        "test_asymptotic_retention_cell": matrix.iter_asymptotic_retention_cells,
    }.get(metafunc.function.__name__)
    if cells is not None:
        metafunc.parametrize("cell", _params(cells()))


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


def _matched(cell: matrix.Cell, method: str, disposition: matrix.Disposition) -> CaseResult:
    outcomes = disposition.outcomes(method)
    case = matrix_cases.build_case(cell, method, outcomes)
    result = run_case(case)
    _assert_runners_produced_rows(case, outcomes, result)
    assert_parity(case, result)
    return result


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


def _quantile_cell(option: str, missing: str) -> matrix.Cell:
    return next(
        cell
        for cell in matrix.iter_cells()
        if (cell.base, cell.view, cell.option, cell.missing, cell.day_boundary)
        == ("quantile", "run", option, missing, "utc")
    )


@pytest.mark.parametrize("missing", ["error", "zero"])
def test_sequential_quantile_is_refused_by_from_moments_over_an_exported_checkpoint(
    missing: str,
) -> None:
    """The producer builds and exports its construction-time checkpoint; the refusal comes
    from `from_moments` declaring a quantile over that scalar-moments source."""
    import tempfile
    from pathlib import Path

    import pyarrow.parquet as pq

    from increment import Analysis
    from increment.errors import CodedError

    cell = _quantile_cell("sequential", missing)
    ingress = matrix_cases._Ingress(cell)
    metrics = matrix_cases._replay_metrics(cell)
    producer = ingress._scalar_moments_producer()
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moments.parquet"
            producer.export(path)
            rows = pq.read_table(path).to_pylist()
    finally:
        producer.close()
    assert rows[0]["record_kind"] == "sequential_checkpoint"
    with pytest.raises(CodedError) as refused:
        Analysis.from_moments(rows, control="control", metrics=metrics)
    assert refused.value.code == "sequential.route.unsupported"


def test_sequential_quantile_drop_and_impute_are_refused_before_the_portable_ingress() -> None:
    """`drop` is refused by the mean producer's own sequential registration and `impute` by
    the quantile declaration the replay builds first; neither reaches `from_moments`."""
    from increment.errors import CodedError

    drop = matrix_cases._Ingress(_quantile_cell("sequential", "drop"))
    with pytest.raises(CodedError) as producer:
        drop._scalar_moments_producer()
    assert producer.value.code == "sequential.route.unsupported"
    with pytest.raises(CodedError) as declaration:
        matrix_cases._replay_metrics(_quantile_cell("sequential", "impute"))
    assert declaration.value.code == "frame.metric.missing_impute"


@pytest.mark.slow
def test_percentile_winsor_three_arm_retained_state_matches_across_supported_ingresses() -> None:
    """A real three-arm pool retains both treatment contrasts and its common cutoff state."""
    from tests.parity_harness.cases import _winsor_three_arm_case

    case = _winsor_three_arm_case()
    result = run_case(case)
    assert_parity(case, result)
    assert set(result.rows) == {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
    }
    assert result.refusals == {
        "from_unit_panel": "estimation.winsor.raw_state_required",
        "from_moments": "estimation.winsor.raw_state_required",
    }


@pytest.mark.slow
@pytest.mark.parametrize("treatment_arms", [1, 2], ids=["two-arm", "three-arm"])
def test_percentile_winsor_zero_outcomes_match_across_supported_ingresses(
    treatment_arms: int,
) -> None:
    """Zero outcomes are ordinary pool members on every raw-outcome source; source-limited
    routes stay limited."""
    from tests.parity_harness.cases import _winsor_zero_inclusive_case

    case = _winsor_zero_inclusive_case(treatment_arms)
    result = run_case(case)
    assert_parity(case, result)
    assert set(result.rows) == {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
    }
    assert result.refusals == {
        "from_unit_panel": "estimation.winsor.raw_state_required",
        "from_moments": "estimation.winsor.raw_state_required",
    }


@pytest.mark.slow
def test_percentile_winsor_size_route_runs_the_analytic_interval_on_every_raw_source() -> None:
    """Past the pooled-size threshold the default route executes the influence interval;
    every raw-outcome source agrees and records the executed method."""
    from tests.parity_harness.cases import _winsor_size_routed_case

    case = _winsor_size_routed_case()
    result = run_case(case)
    assert_parity(case, result)
    assert set(result.rows) == {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
    }
    for rows in result.rows.values():
        (methods,) = rows.values()
        (payload,) = methods.values()
        confidence = payload["confidence_set"]
        assert confidence["reference"]["method"] == "influence-normal-v1"
        assert confidence["raw"]["inference"]["method"] == "pooled-size-route-v1"
        assert confidence["raw"]["allocation"] == (
            ("control", 10200, 20400),
            ("treatment", 10200, 20400),
        )


def test_percentile_winsor_unsupported_ingress_classifications_remain_source_specific() -> None:
    """Moments lack unit outcomes; switchback is a different design, not a parity route."""
    for missing in ("error", "zero", "drop"):
        cell = matrix.Cell("mean", "run", "winsor_percentile", "utc", missing)
        verdicts = matrix.classify(cell).legs["rows"]
        assert verdicts["from_moments"].status == "source_limited"
        assert verdicts["from_moments"].outcome == matrix.Refuses(
            "estimation.winsor.raw_state_required"
        )
        if missing in ("error", "zero"):
            assert verdicts["from_unit_panel"].status == "source_limited"
            assert verdicts["from_unit_panel"].outcome == matrix.Refuses(
                "estimation.winsor.raw_state_required"
            )
        else:
            assert verdicts["from_unit_panel"].outcome == matrix.Refuses(
                "frame.missing_policy.panel_drop"
            )
        assert verdicts["from_switchback_panel"].status == "source_limited"
        assert verdicts["from_switchback_panel"].outcome == matrix.Refuses(
            "source.frame.switchback.metric"
        )


def test_cell(cell: matrix.Cell) -> None:
    """Each leg of the cell (a day-axis view has a value leg and a lift leg) runs on its own,
    so one leg refusing never hides the other's rows."""
    disposition = matrix.classify(cell)
    for method in disposition.legs:
        _matched(cell, method, disposition)
        _switchback(cell, method, disposition)


def test_asymptotic_retention_variant_enumerates_every_retention_sequential_cell():
    """The variant is an enumerated set, not a one-off: every retention and windowed-retention
    `sequential` cell of the matrix, once, on the asymptotic route, with the same disposition
    as its default cell. The expected set is spelled out here from the axes' literal values, so
    a generator that returned another metric's cells (the conversion `sequential` cells have
    the same count and uniqueness) or dropped one retention cell fails."""
    expected = sorted(
        itertools.product(
            ("retention", "windowed_retention"),
            ("run", "breakout", "daily", "asof"),
            ("sequential",),
            ("utc", "fixed_offset"),
            ("error", "zero", "drop", "impute"),
            ("asymptotic_mean",),
        )
    )
    variants = list(matrix.iter_asymptotic_retention_cells())
    assert (
        sorted(
            (c.metric, c.view, c.option, c.day_boundary, c.missing, c.inference) for c in variants
        )
        == expected
    )
    for cell in variants:
        default = replace(cell, inference="default")
        assert matrix.classify(cell) == matrix.classify(default)
        asymptotic = matrix_cases.plan_for(cell, frame=True).inference
        exact = matrix_cases.plan_for(default, frame=True).inference
        assert asymptotic is not None and asymptotic.kind == "asymptotic_mean"
        assert exact is not None and exact.kind == "always_valid"


def test_asymptotic_retention_cell(cell: matrix.Cell) -> None:
    """Retention under the asymptotic scalar-mean route is held to the cell's recorded
    per-ingress outcomes, as the exact-Bernoulli route is: the matched ingresses that run
    agree, the dateless unit summary refuses at its constructor, and every other refusal keeps
    its recorded code. Agreement alone would also hold if every ingress fell back to exact
    Bernoulli monitoring, so each live ingress's retained state must name the scalar-mean
    law."""
    disposition = matrix.classify(cell)
    for method in disposition.legs:
        merged = _matched(cell, method, disposition)
        _switchback(cell, method, disposition)
        case = matrix_cases.build_case(cell, method, disposition.outcomes(method))
        for name in merged.sequential_state:
            analysis = case.build[name]()
            try:
                as_of = getattr(analysis, "_sequential_as_of", None)
                snapshot = analysis.capture_sequential(
                    finalized=True, **({"as_of": as_of} if as_of is not None else {})
                )
                assert {state.law for state in snapshot.states} == {"scalar_mean"}, name
            finally:
                analysis.close()
                for connection in getattr(analysis, "_parity_connections", ()):
                    connection.disconnect()
