"""Behavioral checks for parity matrix cells, on every ingress.

``tests/parity_harness/matrix.py`` names, for each cell of metric x view x option x
day_boundary x missing, what each of the six ingress constructors must do: run, refuse with a
recorded code, or be structurally unable to express the request. The five matched-arm
ingresses that run are compared with each other within 1e-9 relative;
``from_switchback_panel`` (a different estimand) is run on its own, with its rows required to
exist but never compared. ``test_cell`` builds the cell on each ingress, runs it, and asserts
exactly that, so a stale refusal code, a refusal that became a number, a number that became a
refusal and a numeric disagreement all fail.

The pull-request subset contains the cheapest cell that runs for each supported (metric
type, method, design feature) capability, plus one representative per recorded
preflight-refusal hazard. Costs come from ``.github/slow-test-durations.json``; missing
weights sort after measured cells and ties follow matrix order. Other primary cells remain in
the full matrix and are marked ``exhaustive``. The 64 asymptotic-retention variants have three
running metric/method code paths; two cheapest running cells cover all three on PRs, while the
other 62 stay in full scope as ``exhaustive``.

Cells in which any ingress builds and reads data (an ingress runs, or refuses only once a
request is read) are ``slow``. In the remaining cells no ingress runs or reaches the request
stage: each declaration either refuses with a code or is structurally absent (a schema or
keyword the ingress cannot express, with the declared field named by the error), so they run
in the fast tier. Every cell builds and runs each of its ingresses itself: no result is
shared between cells, so a cell never depends on test order or on a sibling having run. The
unit-day artifact a cell adopts may come from a publication its test process already made
from the same definitions, rows and extensions; adopting one is read-only and checked
(``parity_harness/publications.py``).
"""

from __future__ import annotations

import itertools
import json
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


def _params(cells: Iterable[matrix.Cell], pr_cells: frozenset[matrix.Cell]) -> list:
    params = []
    for cell in cells:
        disposition = matrix.classify(cell)
        if cell not in pr_cells:
            marks = [pytest.mark.slow, pytest.mark.exhaustive]
        elif _reads_data(disposition):
            marks = [pytest.mark.slow]
        else:
            marks = []
        params.append(pytest.param(cell, id=cell.id, marks=marks))
    return params


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if metafunc.function.__name__ == "test_cell":
        metafunc.parametrize(
            "cell", _params(matrix.parity_matrix_cells(), frozenset(matrix.pr_matrix_cells()))
        )
    elif metafunc.function.__name__ == "test_asymptotic_retention_cell":
        variants = tuple(matrix.iter_asymptotic_retention_cells())
        pr_variants = frozenset(matrix.pr_asymptotic_retention_cells())
        metafunc.parametrize("cell", _params(variants, pr_variants))


def _preflight_refusal_params() -> list:
    """One cell per recorded hazard runs on every pull request; the full scope also generates
    every other refusing cell, as exhaustive."""
    representatives = {hazard.cell for hazard in matrix.preflight_hazards()}
    full = matrix.parity_matrix_scope() == "full"
    params = []
    for (cell, ingress), requests in matrix.preflight_refusal_groups().items():
        param_id = f"{cell.id}-{ingress}"
        if param_id in representatives:
            params.append(pytest.param(cell, ingress, requests, id=param_id))
        elif full:
            params.append(
                pytest.param(cell, ingress, requests, id=param_id, marks=pytest.mark.exhaustive)
            )
    return params


def test_each_scope_generates_its_cells_and_marks_the_rest_exhaustive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default scope generates the pull-request cells and refusal representatives; the
    full scope generates every cell and refusing cell, marking those outside the
    pull-request set exhaustive."""
    pr_cells = frozenset(matrix.pr_matrix_cells())
    representatives = {hazard.cell for hazard in matrix.preflight_hazards()}
    refusing = {f"{cell.id}-{ingress}" for cell, ingress in matrix.preflight_refusal_groups()}
    for scope, cells, preflight in (
        ("cover", pr_cells, representatives),
        ("full", frozenset(matrix.iter_cells()), refusing),
    ):
        monkeypatch.setenv("INCREMENT_PARITY_MATRIX", scope)
        params = _params(matrix.parity_matrix_cells(), pr_cells)
        assert {param.values[0] for param in params} == cells, scope
        assert {
            param.values[0]
            for param in params
            if any(mark.name == "exhaustive" for mark in param.marks)
        } == cells - pr_cells, scope
        variants = tuple(matrix.iter_asymptotic_retention_cells())
        pr_variants = frozenset(matrix.pr_asymptotic_retention_cells())
        variant_params = _params(variants, pr_variants)
        assert {param.values[0] for param in variant_params} == set(variants), scope
        assert {
            param.values[0]
            for param in variant_params
            if any(mark.name == "exhaustive" for mark in param.marks)
        } == set(variants) - pr_variants, scope
        generated = _preflight_refusal_params()
        assert {param.id for param in generated} == preflight, scope
        assert {
            param.id
            for param in generated
            if not any(mark.name == "exhaustive" for mark in param.marks)
        } == representatives, scope


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.grid("Every request-stage refusal in the enumerated parity matrix")
@pytest.mark.parametrize(("cell", "ingress", "requests"), _preflight_refusal_params())
def test_runnable_preserves_preflight_refusals(
    cell, ingress, requests, monkeypatch, record_property
):
    from increment.errors import CodedError
    from tests.parity_harness.cases import _close_parity_binding

    build_case = (
        matrix_cases.build_switchback_case
        if ingress == "from_switchback_panel"
        else matrix_cases.build_case
    )
    disposition = matrix.classify(cell)
    cases = [
        (method, code, build_case(cell, method, disposition.outcomes(method)))
        for method, code in requests
    ]
    hazards: set[tuple[str, str, tuple[str, ...], str]] = set()
    try:
        bound = cases[0][2].build[ingress]()
    except CodedError as error:
        for method, code, _ in cases:
            assert error.code == code, method
            record_property(f"refusal_stage_{method}", "declaration")
        hazards.add((ingress, error.code, tuple(sorted(error.context)), "declaration"))
        _assert_recorded_hazards(f"{cell.id}-{ingress}", hazards, record_property)
        return
    try:
        for method, code, case in cases:
            try:
                with monkeypatch.context() as method_patch:
                    stage, context = _assert_runnable_refusal(
                        bound, cell, method, ingress, code, case, method_patch, record_property
                    )
            except BaseException as error:
                error.add_note(
                    f"Preflight refusal check: cell={cell.id}, source={ingress}, method={method}"
                )
                raise
            hazards.add((ingress, code, context, stage))
    finally:
        _close_parity_binding(bound)
    _assert_recorded_hazards(f"{cell.id}-{ingress}", hazards, record_property)


def _assert_recorded_hazards(cell_id, hazards, record_property):
    """Every refusal a cell raised is a recorded hazard, and a cell that runs a hazard on every
    pull request still raises it."""
    record_property("refusal_hazards", json.dumps(sorted(hazards)))
    recorded = {(h.ingress, h.code, h.context, h.stage): h.cell for h in matrix.preflight_hazards()}
    unrecorded = hazards - recorded.keys()
    assert not unrecorded, (
        f"{cell_id}: unrecorded refusal hazards {sorted(unrecorded)}; "
        "regenerate tests/parity_harness/preflight_hazards.json with "
        "scripts/record_preflight_hazards.py"
    )
    represented = {hazard for hazard, cell in recorded.items() if cell == cell_id}
    assert represented <= hazards, f"{cell_id} no longer raises {sorted(represented - hazards)}"


def _assert_runnable_refusal(
    bound, cell, method, ingress, code, case, monkeypatch, record_property
):
    from increment import Experiment, runnable, workflows
    from increment.errors import CodedError
    from increment.evidence import BreakoutIdentity
    from increment.kernel.contracts import View
    from tests.analysis_factory import held_source
    from tests.parity_harness.cases import _capture_kwargs
    from tests.parity_harness.runner import _read
    from tests.test_metric_evidence import _CountingSource

    experiment, resolver = bound
    scope = View(
        view=cell.view,
        grain=cell.view if cell.view in {"daily", "asof"} else "total",
        values=method == "values",
        segment=(BreakoutIdentity("store", None),) if cell.view == "breakout" else (),
    )
    experiment = replace(experiment, views=(scope,))
    counted = _CountingSource(held_source(resolver))
    counted.loader_reads.clear()
    for name in (
        "moments",
        "unit_frame",
        "unit_counts",
        "cluster_counts",
        "assignment_counts",
        "triggered_counts",
        "contrast_stats",
        "breakout_moments_for_method",
        "_ensure",
        "fit_cuped_multicovariate",
        "compliance_summary",
        "moments_for_method",
        "_bounded_day_panel",
        "_day_identity",
    ):
        owner = type(counted.source)
        original = getattr(owner, name, None)
        if original is not None:

            def track(instance, *args, _name=name, _original=original, **kwargs):
                counted.loader_reads.append(_name)
                return _original(instance, *args, **kwargs)

            monkeypatch.setattr(owner, name, track)
    with pytest.raises(CodedError) as executed:
        if case.sequential:
            workflows.capture_sequential(*bound, finalized=True, **_capture_kwargs(bound))
        _read(bound, ingress, case)
    assert executed.value.code == code, method
    executed_reads = tuple(counted.loader_reads)
    counted.loader_reads.clear()
    estimator = (
        "observational.iptw"
        if cell.option == "observational"
        else ("randomized.cuped" if cell.option == "cuped" else "randomized.unadjusted")
    )
    status = runnable(Experiment.from_json(experiment.to_json()), "m")[estimator]
    stage = "construction" if case.sequential else ("data" if executed_reads else "preflight")
    record_property(f"refusal_stage_{method}", stage)
    if stage == "preflight":
        _assert_preflight_refusal(status, executed.value, scope)
    elif stage == "data":
        assert status.deferred_checks
    assert counted.loader_reads == []
    return stage, tuple(sorted(executed.value.context))


def test_pr_preflight_keeps_one_cell_per_refusal_hazard():
    """Pull requests run one preflight cell per recorded refusal hazard: a code with the keys
    of its context, raised by one ingress at one stage. The record names every request-stage
    refusal the full matrix enumerates; the exhaustive cells hold every refusal to it."""
    requests = {
        f"{cell.id}-{ingress}": (ingress, {code for _, code in pairs})
        for (cell, ingress), pairs in matrix.preflight_refusal_groups().items()
    }
    hazards = matrix.preflight_hazards()
    keys = [(h.ingress, h.code, h.context, h.stage) for h in hazards]
    assert len(set(keys)) == len(keys)
    for hazard in hazards:
        ingress, codes = requests[hazard.cell]
        assert ingress == hazard.ingress and hazard.code in codes, hazard
    assert {(ingress, code) for ingress, codes in requests.values() for code in codes} == {
        (h.ingress, h.code) for h in hazards
    }
    pr_cells = {
        param.id
        for param in _preflight_refusal_params()
        if not any(mark.name == "exhaustive" for mark in param.marks)
    }
    assert pr_cells == {h.cell for h in hazards}


def _assert_preflight_refusal(status, expected, scope):
    refusal = next((item for item in status.static_refusals if item.code == expected.code), None)
    assert refusal is not None, tuple(item.code for item in status.static_refusals)
    if expected.code == "evidence.breakout_unavailable":
        # Execution reads the implicit run_breakout(); preflight names the authored selection.
        selection = tuple(identity.property for identity in scope.segment)
        assert dict(refusal.context) == {**expected.context, "requested": selection}


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


def test_qr_cuped_ingress_contract_is_enumerated_separately_from_scalar_cells():
    from tests.capability_contracts import CONSTRUCTORS

    assert matrix.QR_CUPED_INGRESSES == CONSTRUCTORS
    assert matrix.QR_CUPED_SOURCE_REFUSALS == {
        "from_switchback_panel": "source.frame.switchback.metric"
    }
    assert "mixed-cuped-multicovariate-fixed-horizon" in matrix.QR_CUPED_ROWS


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

    from increment import builders, workflows
    from increment.errors import CodedError

    cell = _quantile_cell("sequential", missing)
    ingress = matrix_cases._Ingress(cell)
    metrics = matrix_cases._replay_metrics(cell)
    producer = ingress._scalar_moments_producer()
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "moments.parquet"
            workflows.export_moments(*producer, path)
            rows = pq.read_table(path).to_pylist()
    finally:
        producer[1].close()
    assert rows[0]["record_kind"] == "sequential_checkpoint"
    with pytest.raises(CodedError) as refused:
        builders.from_moments(rows, control="control", metrics=metrics)
    assert refused.value.code == "sequential.route.unsupported"


def test_sequential_quantile_skip_and_impute_are_refused_before_the_portable_ingress() -> None:
    """`skip` is refused by the mean producer's own sequential registration and `impute` by
    the quantile declaration the replay builds first; neither reaches `from_moments`."""
    from increment.errors import CodedError

    skip = matrix_cases._Ingress(_quantile_cell("sequential", "skip"))
    with pytest.raises(CodedError) as producer:
        skip._scalar_moments_producer()
    assert producer.value.code == "sequential.route.unsupported"
    with pytest.raises(CodedError) as declaration:
        matrix_cases._replay_metrics(_quantile_cell("sequential", "impute"))
    assert declaration.value.code == "frame.metric.missing_impute"


@pytest.mark.parametrize(
    ("ingress", "code", "field", "value"),
    [
        ("from_unit_summary", "frame.metric.missing_drop_renamed", "route", "missing='skip'"),
        ("from_unit_panel", "frame.metric.missing_drop_renamed", "route", "missing='skip'"),
        ("from_switchback_panel", "frame.metric.missing_drop_renamed", "route", "missing='skip'"),
        ("from_moments", "frame.metric.missing_drop_renamed", "route", "missing='skip'"),
        ("from_definitions", "model.field.unknown", "field", "missing"),
    ],
)
def test_drop_missing_policy_is_refused_at_declaration_on_every_ingress(
    ingress: str, code: str, field: str, value: str
) -> None:
    """`drop`, the former spelling of `skip`, refuses with one code routing to `skip` on every
    ingress declaring a `MetricSpec`; the definitions schema has no missing-policy field."""
    from increment.errors import CodedError

    cell = matrix.Cell("mean", "run", "none", "utc", "drop")
    built = matrix_cases._Ingress(cell)
    build = {
        "from_unit_summary": built.summary,
        "from_unit_panel": built.panel,
        "from_switchback_panel": matrix_cases._switchback_builder(cell),
        "from_moments": lambda: matrix_cases._replay_metrics(cell),
        "from_definitions": built.definitions,
    }[ingress]
    with pytest.raises(CodedError) as refused:
        build()
    assert refused.value.code == code
    assert refused.value.context[field] == value


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
        "from_unit_panel": "evidence.raw_outcomes_unavailable",
        "from_moments": "evidence.raw_outcomes_unavailable",
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
        "from_unit_panel": "evidence.raw_outcomes_unavailable",
        "from_moments": "evidence.raw_outcomes_unavailable",
    }


@pytest.mark.slow
def test_percentile_winsor_size_route_runs_the_analytic_interval_on_every_raw_source() -> None:
    """Past the pooled-size threshold the default route executes the influence interval;
    every raw-outcome source agrees and records the executed method."""
    from tests.parity_harness.cases import _winsor_size_routed_case
    from tests.parity_harness.matrix_data import scaled_units_per_arm

    case = _winsor_size_routed_case()
    result = run_case(case)
    assert_parity(case, result)
    assert set(result.rows) == {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
    }
    # The case repeats each enrolled unit 170 times: (arm, arm units, pooled units).
    per_arm = 170 * scaled_units_per_arm()
    for rows in result.rows.values():
        (methods,) = rows.values()
        (payload,) = methods.values()
        confidence = payload["confidence_set"]
        assert confidence["reference"]["method"] == "influence-normal-v1"
        assert confidence["raw"]["inference"]["method"] == "pooled-size-route-v1"
        assert confidence["raw"]["allocation"] == (
            ("control", per_arm, 2 * per_arm),
            ("treatment", per_arm, 2 * per_arm),
        )


def test_percentile_winsor_unsupported_ingress_classifications_remain_source_specific() -> None:
    """Moments lack unit outcomes; switchback is a different design, not a parity route."""
    for missing in ("error", "zero", "skip"):
        cell = matrix.Cell("mean", "run", "winsor_percentile", "utc", missing)
        verdicts = matrix.classify(cell).legs["rows"]
        assert verdicts["from_moments"].status == "source_limited"
        assert verdicts["from_moments"].outcome == matrix.Refuses(
            "evidence.raw_outcomes_unavailable"
        )
        if missing in ("error", "zero"):
            assert verdicts["from_unit_panel"].status == "source_limited"
            assert verdicts["from_unit_panel"].outcome == matrix.Refuses(
                "evidence.raw_outcomes_unavailable"
            )
        else:
            assert verdicts["from_unit_panel"].outcome == matrix.Refuses(
                "frame.missing_policy.panel_skip"
            )
        assert verdicts["from_switchback_panel"].status == "source_limited"
        assert verdicts["from_switchback_panel"].outcome == matrix.Refuses(
            "source.frame.switchback.metric"
        )


@pytest.mark.integration
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
            ("error", "zero", "skip", "impute"),
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
        for name, capture in merged.sequential_state.items():
            assert {state[3] for state in capture.fingerprint[0]} == {"scalar_mean"}, name
