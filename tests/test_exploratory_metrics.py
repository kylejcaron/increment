"""Metrics added after the plan: ``Analysis.available_metrics`` and ``exploratory_metrics=``.

Real DuckDB fixtures throughout. An added metric must read exactly like the same
metric declared in a plan of its own, never move a declared row, and be refused
before any query when it cannot be served.
"""

from __future__ import annotations

import copy
import pickle
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import pytest

from increment import Analysis, SourceSnapshotEvidence
from increment.errors import CapabilityError, CodedError, IncrementWarning
from increment.estimation.engine import Method
from increment.plan import bind_automatic_sequential_plan
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, Definitions, MultiplicitySpec
from increment.sequential_source import native_observation_mapping
from tests.analysis_factory import _native_source, lift_rows, make_analysis
from tests.parity_harness import dataset as ds

ADDED = ("purchase_rate", "rps", "revenue_cuped")
PLAN_ALPHAS = (0.05, 0.01)
CUPED = Method(name="cuped", variance_reduction="cuped")
UNAVAILABLE = "facade.analysis_config.exploratory_metric_unavailable"


def _definitions(plan: AnalysisPlan, *, breakout: bool = False) -> Definitions:
    return Definitions.model_validate(ds.definitions_dict(plan=plan, breakout=breakout))


def _analysis(con: Any, plan: AnalysisPlan, *, breakout: bool = False) -> Analysis:
    return make_analysis(con, _definitions(plan, breakout=breakout), experiment="exp")


def _triggered_analysis(
    con: Any,
    plan: AnalysisPlan,
    *,
    include_constant_metric: bool = False,
    delayed_trigger: bool = False,
) -> Analysis:
    """Run the parity data through an explicitly declared, source-backed trigger."""
    definitions = ds.definitions_dict(plan=plan)
    trigger_fact = "trigger_signal" if delayed_trigger else "purchase"
    if delayed_trigger:
        definitions["fact_sources"][0]["facts"].append({"name": "trigger_signal", "column": None})
    definitions["exposures"].append({"name": "trigger", "fact": trigger_fact})
    definitions["experiments"][0]["trigger"] = "trigger"
    if delayed_trigger:
        definitions["experiments"][0]["observation_end"] = "2025-02-16"
    if include_constant_metric:
        definitions["metrics"].append(
            {
                "type": "mean",
                "name": "constant_session",
                "entity": "user_id",
                "fact": "session_end",
                "aggregation": "sum",
                "window_days": 1,
            }
        )
    defs = Definitions.model_validate(definitions)
    return make_analysis(
        con,
        defs,
        experiment="exp",
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=datetime(2025, 2, 16, tzinfo=UTC),
            complete_through_by_feed={"events": datetime(2025, 2, 16, tzinfo=UTC)},
        ),
    )


def _triggered_event_rows(*, include_triggered_sessions: bool = True) -> list[dict[str, Any]]:
    """Supply post-enrollment triggers and outcomes after the experiment end."""
    rows = ds.event_rows()
    assignments = [row for row in rows if row["event"] == "exposure"]
    for row in assignments:
        unit_index = int(row["user_id"][1:])
        events = [
            {
                **row,
                "event_at": datetime(2025, 1, 21, 10),
                "event": "trigger_signal",
                "revenue": None,
                "sess": None,
            },
            {
                **row,
                "event_at": datetime(2025, 1, 21, 11),
                "event": "purchase",
                "revenue": float(10 + unit_index % 7),
                "sess": None,
            },
        ]
        if include_triggered_sessions:
            events.append(
                {
                    **row,
                    "event_at": datetime(2025, 1, 21, 12),
                    "event": "session_end",
                    "revenue": None,
                    "sess": 1,
                }
            )
        rows.extend(events)
    return rows


def _width(row: Any) -> float:
    lift = row.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return lift.ub - lift.lb


def _rows(
    rows: Any,
    *,
    drop: tuple[str, ...] = ("role", "source_snapshot_id", "family_id", "multiplicity_status"),
) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.model_dump().items() if k not in drop} for row in rows]


def _nominal_alphas(rows: Any) -> list[float]:
    """The nominal level each estimated row's interval was built at."""
    return [row.lift.alpha for row in rows if row.lift is not None]


@pytest.fixture(scope="module")
def warehouse():
    con = ds.duckdb_connection()
    yield con
    con.disconnect()


@pytest.fixture
def triggered_warehouse():
    con = ds.duckdb_connection(_triggered_event_rows())
    yield con
    con.disconnect()


@pytest.fixture
def triggered_warehouse_without_followup_sessions():
    con = ds.duckdb_connection(_triggered_event_rows(include_triggered_sessions=False))
    yield con
    con.disconnect()


def test_available_metrics_are_the_undeclared_definitions_in_definitions_order(warehouse):
    analysis = _analysis(warehouse, AnalysisPlan(primary="purchase_rate", secondaries=["rps"]))
    assert [m.name for m in analysis.available_metrics] == ["revenue", "revenue_cuped"]


@pytest.mark.parametrize("alpha", PLAN_ALPHAS)
@pytest.mark.parametrize(
    ("name", "call_wide"),
    [(name, {}) for name in ADDED] + [("revenue_cuped", {"decision_method": CUPED})],
    ids=[*ADDED, "revenue_cuped-cuped"],
)
def test_added_whole_window_rows_equal_the_metric_declared_in_its_own_plan(
    warehouse, name, call_wide, alpha
):
    declared = lift_rows(
        _analysis(warehouse, AnalysisPlan(alpha=alpha, primary=name)).run(**call_wide)
    )
    added = lift_rows(
        _analysis(warehouse, AnalysisPlan(alpha=alpha, primary="revenue")).run(
            metrics=[], exploratory_metrics=[name], **call_wide
        )
    )
    assert declared and {row.role for row in declared} == {"primary"}
    assert {row.role for row in added} == {"exploratory"}
    assert {row.multiplicity_status for row in added} == {"exploratory_unadjusted"}
    assert all(row.family_id is not None for row in added)
    assert added.metadata is not None
    assert {family.name for family in added.metadata.scope.families} == {"exploratory"}
    assert {cell.metric for family in added.metadata.scope.families for cell in family.members} == {
        name
    }
    frame = added.to_frame(backend="pandas")
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["multiplicity_status"]) == {"exploratory_unadjusted"}
    from increment.estimation.readout_types import ReadoutResults
    from increment.tables import estimates_to_readout

    assert {row["multiplicity_status"] for row in estimates_to_readout(added)} == {
        "exploratory_unadjusted"
    }
    restored = lift_rows(ReadoutResults.model_validate_json(added.model_dump_json()))
    assert restored.metadata == added.metadata
    assert [row.family_id for row in restored] == [row.family_id for row in added]


def test_declared_whole_window_rows_are_identical_with_added_metrics(warehouse):
    analysis = _analysis(
        warehouse, AnalysisPlan(primary="revenue", secondaries=["purchase_rate"], q=0.1)
    )
    without = lift_rows(analysis.run())
    combined = lift_rows(analysis.run(exploratory_metrics=["rps", "revenue_cuped"]))
    assert [row.model_dump() for row in combined[: len(without)]] == [
        row.model_dump() for row in without
    ]
    added = combined[len(without) :]
    assert {row.metric for row in added} == {"rps", "revenue_cuped"}
    assert {row.role for row in added} == {"exploratory"}
    assert {row.multiplicity_status for row in added} == {"exploratory_unadjusted"}
    assert all(row.discovery is None and row.family_axes is None for row in added)
    narrowed = lift_rows(analysis.run(metrics=["revenue"], exploratory_metrics=["rps"]))
    assert {row.metric for row in narrowed} == {"revenue", "rps"}
    assert _rows([row for row in narrowed if row.metric == "revenue"]) == _rows(
        [row for row in without if row.metric == "revenue"]
    )
    assert (
        next(row for row in narrowed if row.metric == "rps").multiplicity_status
        == "exploratory_unadjusted"
    )


def test_triggered_added_metric_reports_both_populations_without_changing_declared_family(
    triggered_warehouse,
):
    analysis = _triggered_analysis(
        triggered_warehouse,
        AnalysisPlan(primary="revenue"),
        delayed_trigger=True,
    )
    without = lift_rows(analysis.run())
    added_names = set(ADDED)
    combined = lift_rows(analysis.run(exploratory_metrics=list(ADDED)))
    assert combined and {row.analysis_population for row in combined} == {
        "assigned",
        "triggered",
    }
    added_rows = [row for row in combined if row.metric in added_names]
    assert {(row.metric, row.analysis_population) for row in added_rows} == {
        (name, population) for name in ADDED for population in ("assigned", "triggered")
    }
    assert {row.role for row in added_rows} == {"exploratory"}
    assert {row.multiplicity_status for row in added_rows} == {"exploratory_unadjusted"}
    declared_without = [row.model_dump() for row in without]
    declared_with = [row.model_dump() for row in combined if row.metric not in added_names]
    assert declared_with == declared_without
    assert combined.metadata is not None and without.metadata is not None
    declared_families = [
        family for family in combined.metadata.scope.families if family.name != "exploratory"
    ]
    assert [family.model_dump() for family in declared_families] == [
        family.model_dump()
        for family in without.metadata.scope.families
        if family.name != "exploratory"
    ]


def test_triggered_added_cuped_metric_returns_both_populations(triggered_warehouse):
    analysis = _triggered_analysis(
        triggered_warehouse,
        AnalysisPlan(primary="revenue"),
        delayed_trigger=True,
    )
    rows = lift_rows(
        analysis.run(metrics=[], exploratory_metrics=["revenue_cuped"], decision_method=CUPED)
    )
    assert rows and {row.analysis_population for row in rows} == {"assigned", "triggered"}
    assert {row.metric for row in rows} == {"revenue_cuped"}
    assert {row.role for row in rows} == {"exploratory"}


@pytest.fixture
def constant_covariate_warehouse():
    rows = ds.event_rows()
    for row in rows:
        if row["event"] == "purchase" and row["event_at"] < datetime(2025, 1, 10, 9):
            row["revenue"] = 3.0
    con = ds.duckdb_connection(rows)
    yield con
    con.disconnect()


@pytest.fixture
def zero_denominator_warehouse():
    rows = [
        row
        for row in ds.event_rows()
        if not (row["event"] == "session_end" and row["event_at"] < datetime(2025, 1, 19, 9))
    ]
    con = ds.duckdb_connection(rows)
    yield con
    con.disconnect()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_all_failed_methods_still_refuse_the_readout(zero_denominator_warehouse):
    analysis = _analysis(zero_denominator_warehouse, AnalysisPlan(primary="revenue"))
    with pytest.warns(IncrementWarning):
        with pytest.raises(CodedError) as refused:
            analysis.run(
                metrics=[],
                exploratory_metrics=["rps"],
                sensitivity_methods=[CUPED],
            )
    assert refused.value.code == "readout.estimate_lift_every"


@pytest.mark.parametrize("source_fixture", ["warehouse", "triggered_warehouse"])
def test_capability_refusal_propagates_after_a_valid_sibling(request, monkeypatch, source_fixture):
    from increment.readouts import _passes

    con = request.getfixturevalue(source_fixture)
    plan = AnalysisPlan(primary="revenue")
    analysis = (
        _triggered_analysis(con, plan, delayed_trigger=True)
        if source_fixture == "triggered_warehouse"
        else _analysis(con, plan)
    )
    estimate_lift = _passes._estimate_lift
    attempted = []

    def refuse_conversion(*args, **kwargs):
        metric = kwargs["metrics"][0]
        attempted.append(metric.name)
        if metric.name == "purchase_rate":
            raise CapabilityError(
                "Exact counts are required.",
                code="estimation.binomial.exact_counts_required",
                context={"metric": "purchase_rate"},
            )
        return estimate_lift(*args, **kwargs)

    monkeypatch.setattr(_passes, "_estimate_lift", refuse_conversion)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            analysis.run(metrics=["revenue"], exploratory_metrics=["purchase_rate"])
        except CapabilityError as raised:
            assert raised.code == "estimation.binomial.exact_counts_required"
        else:
            pytest.fail(f"capability refusal converted to a partial result; calls={attempted!r}")
    assert not any(isinstance(warning.message, IncrementWarning) for warning in caught)
    assert "revenue" in attempted
    assert "purchase_rate" in attempted
    assert attempted.index("revenue") < attempted.index("purchase_rate")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_cuped_sensitivity_guard_keeps_the_decision_cell(constant_covariate_warehouse):
    analysis = _analysis(
        constant_covariate_warehouse,
        AnalysisPlan(primary="revenue_cuped", alternative="greater"),
    )
    with pytest.warns(IncrementWarning) as warnings:
        rows = lift_rows(analysis.run(sensitivity_methods=[CUPED]))

    assert {
        warning.message.code
        for warning in warnings
        if isinstance(warning.message, IncrementWarning)
    } == {"readouts.run.cell_refused"}
    decision = next(row for row in rows if row.method_role == "decision")
    sensitivity = next(row for row in rows if row.method_role == "sensitivity")
    assert decision.method == "unadjusted" and decision.lift is not None
    assert decision.alternative == sensitivity.alternative == "greater"
    assert sensitivity.method == "cuped" and sensitivity.lift is None
    assert sensitivity.failure_code == "estimation.cuped.covariate_zero_variance"
    assert sensitivity.failure_context == {
        "weighted_var_x": 0.0,
        "method": "cuped",
    }
    assert rows.metadata is not None
    cuped_cell = next(
        record
        for record in rows.metadata.cells
        if record.cell.method == "cuped" and record.cell.method_role == "sensitivity"
    )
    assert cuped_cell.failure is not None
    assert cuped_cell.failure.code == sensitivity.failure_code
    assert cuped_cell.failure.context == sensitivity.failure_context
    assert cuped_cell.cell.alternative == "greater"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_triggered_added_bad_metric_preserves_typed_cell_identity_and_reason(
    triggered_warehouse,
):
    analysis = _triggered_analysis(
        triggered_warehouse,
        AnalysisPlan(primary="revenue"),
        include_constant_metric=True,
        delayed_trigger=True,
    )
    with pytest.warns(IncrementWarning) as warnings:
        rows = lift_rows(analysis.run(metrics=[], exploratory_metrics=["rps", "constant_session"]))
    assert {
        warning.message.code
        for warning in warnings
        if isinstance(warning.message, IncrementWarning)
    } == {"readouts.run.cell_refused"}
    good = [row for row in rows if row.metric == "rps"]
    bad = [row for row in rows if row.metric == "constant_session"]
    assert {row.analysis_population for row in good} == {"assigned", "triggered"}
    assert {row.analysis_population for row in bad} == {"assigned", "triggered"}
    assert all(row.lift is not None for row in good)
    assert all(row.lift is None for row in bad)
    contexts = [row.failure_context for row in bad if row.failure_context is not None]
    assert len(contexts) == len(bad)
    assert all(context["metric"] == "constant_session" for context in contexts)
    assert {context["reason"] for context in contexts} == {"zero_variance"}


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_triggered_ratio_guard_keeps_its_failed_cell_and_good_metric_siblings(
    triggered_warehouse_without_followup_sessions,
):
    analysis = _triggered_analysis(
        triggered_warehouse_without_followup_sessions,
        AnalysisPlan(primary="revenue"),
        delayed_trigger=True,
    )
    with pytest.warns(IncrementWarning) as warnings:
        rows = lift_rows(analysis.run(metrics=[], exploratory_metrics=["rps", "revenue_cuped"]))
    assert {
        warning.message.code
        for warning in warnings
        if isinstance(warning.message, IncrementWarning)
    } == {"readouts.run.cell_refused"}
    ratio_rows = [row for row in rows if row.metric == "rps"]
    assert {row.role for row in rows} == {"exploratory"}
    cuped_rows = [row for row in rows if row.metric == "revenue_cuped"]
    assert {row.analysis_population for row in ratio_rows} == {"assigned", "triggered"}
    assert {row.analysis_population for row in cuped_rows} == {"assigned", "triggered"}
    assert next(row for row in ratio_rows if row.analysis_population == "assigned").lift
    failed = next(row for row in ratio_rows if row.analysis_population == "triggered")
    assert failed.lift is None
    assert failed.failure_code == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
    assert failed.sampling_reason_code == failed.failure_code
    assert failed.failure_context == {
        "group_id": "treatment",
        "metric": "rps",
        "d_bar": 0.0,
        "method": "unadjusted",
    }
    assert all(row.lift is not None for row in cuped_rows)


@pytest.mark.parametrize("alpha", PLAN_ALPHAS)
@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
@pytest.mark.parametrize("name", ADDED)
def test_added_breakout_rows_equal_the_metric_declared_in_its_own_plan(
    warehouse, name, correction, alpha
):
    view = MultiplicitySpec(correction=correction)
    declared_plan = AnalysisPlan(alpha=alpha, primary=name, view_multiplicity=view)
    added_plan = AnalysisPlan(alpha=alpha, primary="revenue", view_multiplicity=view)
    declared = _analysis(warehouse, declared_plan, breakout=True).run_breakout()
    added = _analysis(warehouse, added_plan, breakout=True).run_breakout(
        metrics=[], exploratory_metrics=[name]
    )
    assert declared and {row.role for row in added} == {"exploratory"}
    expected_status = (
        "exploratory_family" if correction in ("bh", "bonferroni") else "exploratory_unadjusted"
    )
    assert {row.multiplicity_status for row in added} == {expected_status}
    frame = added.to_frame(backend="pandas")
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["multiplicity_status"]) == {expected_status}
    assert _rows(added) == _rows(declared)


def test_uncorrected_secondary_breakout_has_no_family(warehouse):
    plan = AnalysisPlan(
        primary="revenue",
        secondaries=["purchase_rate"],
        view_multiplicity=MultiplicitySpec(correction="none"),
    )
    rows = _analysis(warehouse, plan, breakout=True).run_breakout(metrics=["purchase_rate"])
    assert rows and rows.metadata is not None
    assert {row.role for row in rows} == {"exploratory"}
    assert {row.multiplicity_status for row in rows} == {"exploratory_unadjusted"}
    assert not rows.metadata.scope.families
    assert all(row.family_id is None for row in rows)


def test_breakout_bh_uses_one_joint_family_for_primary_and_secondary(warehouse):
    from increment.estimation.readout_types import CellKey

    plan = AnalysisPlan(
        primary="revenue",
        secondaries=["purchase_rate"],
        q=0.1,
        view_multiplicity=MultiplicitySpec(correction="bh"),
    )
    rows = _analysis(warehouse, plan, breakout=True).run_breakout(
        metrics=["revenue", "purchase_rate"]
    )
    assert rows and rows.metadata is not None
    (family,) = rows.metadata.scope.families
    assert family.name == "breakout"
    assert family.family is not None
    assert family.family.correction == "bh"
    assert family.family.q == pytest.approx(0.1)
    assert family.family.axes == ("metric", "arm", "segment")
    assert {cell.metric for cell in family.members} == {"revenue", "purchase_rate"}
    decision_rows = [row for row in rows if row.method_role == "decision"]
    assert set(family.members) == {CellKey.from_row(row) for row in decision_rows}
    assert {row.family_id for row in decision_rows} == {family.family_id}


def test_added_segmented_asof_bonferroni_rows_are_disclosed_as_family(warehouse):
    plan = AnalysisPlan(
        primary="revenue",
        view_multiplicity=MultiplicitySpec(correction="bonferroni"),
    )
    analysis = _analysis(warehouse, plan, breakout=True)
    rows = analysis.run_asof_lift(
        metrics=[], dimension="store", exploratory_metrics=["purchase_rate"]
    )
    assert rows and {row.role for row in rows} == {"exploratory"}
    assert {row.multiplicity_status for row in rows} == {"exploratory_family"}
    frame = rows.to_frame(backend="pandas")
    assert isinstance(frame, pd.DataFrame)
    assert set(frame["multiplicity_status"]) == {"exploratory_family"}


def test_declared_breakout_rows_are_identical_with_added_metrics(warehouse):
    analysis = _analysis(
        warehouse,
        AnalysisPlan(
            primary="revenue",
            secondaries=["purchase_rate"],
            view_multiplicity=MultiplicitySpec(correction="bh"),
        ),
        breakout=True,
    )
    without = analysis.run_breakout()
    combined = analysis.run_breakout(exploratory_metrics=["rps"])
    assert [row.model_dump() for row in combined[: len(without)]] == [
        row.model_dump() for row in without
    ]
    assert {row.metric for row in combined[len(without) :]} == {"rps"}


@pytest.mark.parametrize("alpha", PLAN_ALPHAS)
@pytest.mark.parametrize("name", ADDED)
def test_added_day_axis_rows_equal_the_metric_declared_in_its_own_plan(warehouse, name, alpha):
    declared = _analysis(warehouse, AnalysisPlan(alpha=alpha, primary=name))
    added = _analysis(warehouse, AnalysisPlan(alpha=alpha, primary="revenue"))
    lift = added.run_asof_lift(metrics=[], exploratory_metrics=[name])
    assert lift and {row.role for row in lift} == {"exploratory"}
    assert {row.multiplicity_status for row in lift} == {"exploratory_unadjusted"}
    assert _rows(lift) == _rows(declared.run_asof_lift())
    for read in ("run_asof", "run_daily"):
        values = getattr(added, read)(metrics=[], exploratory_metrics=[name])
        assert values and _rows(values) == _rows(getattr(declared, read)())
    both = added.run_asof_lift(exploratory_metrics=[name])
    assert [row.role for row in both if row.metric == "revenue"] == [
        row.role for row in added.run_asof_lift() if row.metric == "revenue"
    ]


@pytest.mark.parametrize("name", ADDED)
@pytest.mark.parametrize("alpha", [0.01, 0.1])
def test_an_added_metric_keeps_the_plan_alpha_in_every_lift_view(warehouse, name, alpha):
    analysis = _analysis(warehouse, AnalysisPlan(alpha=alpha, primary="revenue"), breakout=True)
    reads = {
        "run": analysis.run(metrics=[], exploratory_metrics=[name]),
        "run_asof_lift": analysis.run_asof_lift(metrics=[], exploratory_metrics=[name]),
        "run_breakout": analysis.run_breakout(metrics=[], exploratory_metrics=[name]),
    }
    for view, rows in reads.items():
        nominal = _nominal_alphas(rows)
        assert nominal, view
        assert nominal == pytest.approx([alpha] * len(nominal), abs=1e-12), view


@pytest.mark.parametrize("alpha", [0.01, 0.1])
def test_a_day_source_over_an_added_metric_tests_it_at_the_plan_alpha(warehouse, alpha):
    analysis = _analysis(warehouse, AnalysisPlan(alpha=alpha, primary="revenue"))
    added = next(metric for metric in analysis.available_metrics if metric.name == "rps")
    day = _native_source(analysis).day_source(metrics=(*analysis.metrics, added))
    procedures = day.context.plan.procedures
    assert procedures["rps"].alpha == pytest.approx(alpha, abs=1e-12)
    assert procedures["revenue"].alpha == pytest.approx(alpha, abs=1e-12)


def test_dimensioned_daily_reads_keep_their_refusal_of_undeclared_metrics(warehouse):
    analysis = _analysis(warehouse, AnalysisPlan(primary="revenue"), breakout=True)
    with pytest.raises(CodedError) as refused:
        analysis.run_daily(metrics=[], exploratory_metrics=["purchase_rate"], dimension="store")
    assert refused.value.code == "facade.analysis.undeclared_metric_for_dimension"


def test_dimensioned_asof_lift_serves_added_metrics_as_exploratory_rows(warehouse):
    analysis = _analysis(warehouse, AnalysisPlan(primary="revenue"), breakout=True)
    rows = analysis.run_asof_lift(
        metrics=[], exploratory_metrics=["purchase_rate"], dimension="store"
    )
    assert rows and {row.metric for row in rows} == {"purchase_rate"}
    assert {row.role for row in rows} == {"exploratory"}
    assert {row.dimension for row in rows} == {"store"}


CALLS: dict[str, Callable[..., Any]] = {
    "run": lambda analysis, names: analysis.run(exploratory_metrics=names),
    "run_breakout": lambda analysis, names: analysis.run_breakout(exploratory_metrics=names),
    "run_asof_lift": lambda analysis, names: analysis.run_asof_lift(exploratory_metrics=names),
    "run_asof": lambda analysis, names: analysis.run_asof(exploratory_metrics=names),
    "run_daily": lambda analysis, names: analysis.run_daily(exploratory_metrics=names),
}


@pytest.mark.parametrize("method", CALLS)
@pytest.mark.parametrize(
    ("names", "unknown", "declared"),
    [
        (["nope"], ["nope"], []),
        (["revenue"], [], ["revenue"]),
        (["rps", "nope", "revenue"], ["nope"], ["revenue"]),
    ],
)
def test_unknown_or_declared_names_are_refused_before_any_query(method, names, unknown, declared):
    con = ds.duckdb_connection()
    analysis = _analysis(con, AnalysisPlan(primary="revenue"), breakout=True)
    con.disconnect()  # any query from here on fails, so a coded refusal precedes every read
    with pytest.raises(CodedError) as refused:
        CALLS[method](analysis, names)
    assert refused.value.code == UNAVAILABLE
    assert refused.value.context["unknown_names"] == tuple(unknown)
    assert refused.value.context["declared_names"] == tuple(declared)


def test_a_repeated_added_name_is_refused_with_the_duplicate_code(warehouse):
    analysis = _analysis(warehouse, AnalysisPlan(primary="revenue"))
    with pytest.raises(CodedError) as refused:
        analysis.run(exploratory_metrics=["rps", "rps"])
    assert refused.value.code == "facade.analysis_config.duplicate_metric_name"


def _sequential_analysis(con: Any) -> Analysis:
    plan = AnalysisPlan(
        primary="revenue",
        inference={"kind": "asymptotic_mean", "expected_decision_sample_size": 100},
    )
    allocation = {"control": 0.5, "treatment": 0.5}
    defs = Definitions.model_validate(
        ds.definitions_dict(plan=plan, allocation=allocation, breakout=True)
    )
    experiment = defs.experiment("exp")
    assert experiment is not None
    bound = bind_automatic_sequential_plan(
        experiment.plan,
        [m for m in defs.metrics if m.name in experiment.metric_names],
        design=Randomized(control_group="control", allocation=allocation),
        source_id=experiment.name,
        source_mapping=native_observation_mapping(defs, experiment, on_mixed_assignment="error"),
        pre_period_covariate=experiment.n_pre_periods > 0,
    )
    return make_analysis(con, defs, experiment="exp", plan=bound)


def test_sequential_plans_refuse_added_metrics_and_uncorrected_segments(warehouse):
    analysis = _sequential_analysis(warehouse)
    for call in CALLS.values():
        with pytest.raises(CodedError) as refused:
            call(analysis, ["purchase_rate"])
        assert refused.value.code == "facade.analysis.exploratory_metrics_sequential"
    (declared,) = analysis.experiment.breakouts
    with pytest.raises(CodedError) as refused:
        analysis.dashboard_breakout_reads(declared).uncorrected_segments(metrics=["revenue"])
    assert refused.value.code == "facade.analysis.uncorrected_segments_sequential"


@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
def test_uncorrected_segments_are_the_unadjusted_run_breakout_rows(warehouse, correction):
    names = ["revenue", "purchase_rate"]
    plan = AnalysisPlan(
        primary="revenue",
        secondaries=["purchase_rate"],
        view_multiplicity=MultiplicitySpec(correction=correction),
    )
    analysis = _analysis(warehouse, plan, breakout=True)
    (declared,) = analysis.experiment.breakouts
    reads = analysis.dashboard_breakout_reads(declared)
    uncorrected = reads.uncorrected_segments(metrics=names)
    standalone = _analysis(
        warehouse,
        plan.model_copy(update={"view_multiplicity": MultiplicitySpec(correction="none")}),
        breakout=True,
    ).run_breakout(metrics=names)
    assert uncorrected and _rows(uncorrected) == _rows(standalone)
    assert all(row.family_q is None and row.discovery is None for row in uncorrected)
    if correction == "none":
        assert _rows(uncorrected) == _rows(analysis.run_breakout(metrics=names))
        return
    corrected = {
        (row.metric, row.group_id, row.dimension_value): row
        for row in analysis.run_breakout(metrics=names)
    }
    for row in uncorrected:
        wide = corrected[row.metric, row.group_id, row.dimension_value]
        assert _width(row) <= _width(wide) + 1e-12


def test_uncorrected_segments_carry_added_metrics_as_exploratory_rows(warehouse):
    analysis = _analysis(warehouse, AnalysisPlan(primary="revenue"), breakout=True)
    (declared,) = analysis.experiment.breakouts
    rows = analysis.dashboard_breakout_reads(declared).uncorrected_segments(
        metrics=[], exploratory_metrics=["purchase_rate"]
    )
    assert rows and {row.metric for row in rows} == {"purchase_rate"}
    assert all(row.role == "exploratory" and row.family_q is None for row in rows)


def test_dashboard_snapshot_pins_and_serves_added_metrics():
    con: Any = ds.duckdb_connection()
    try:
        analysis = _analysis(con, AnalysisPlan(primary="revenue"), breakout=True)
        (declared,) = analysis.experiment.breakouts
        added = [metric for metric in analysis.available_metrics if metric.name == "purchase_rate"]
        names = [metric.name for metric in added]

        def read(source: Analysis) -> dict[str, list[dict[str, Any]]]:
            reads = source.dashboard_breakout_reads(declared)
            return {
                "whole": _rows(source.run(metrics=[], exploratory_metrics=names)),
                "segments": _rows(reads.run_breakout(metrics=[], exploratory_metrics=names)),
                "asof": _rows(reads.run_asof_lift(metrics=[], exploratory_metrics=names)),
            }

        baseline = read(analysis)
        assert all(baseline.values())

        def read_after_mutation(isolated: Analysis) -> Any:
            con.raw_sql(
                "DELETE FROM events WHERE event = 'purchase' AND user_id LIKE 't%' "
                "AND event_at >= timestamp '2025-01-10 09:00' "
                "AND event_at < timestamp '2025-01-11 09:00'"
            )
            return read(isolated)

        pinned = analysis.dashboard_snapshot(
            read_after_mutation,
            metrics=[*analysis.metrics, *added],
        )
        assert pinned == baseline
        assert read(analysis)["whole"] != baseline["whole"], "the mutation must be visible unpinned"
    finally:
        con.disconnect()


def _refusals_of_every_new_code(warehouse) -> list[CodedError]:
    refusals: list[CodedError] = []
    analysis = _analysis(warehouse, AnalysisPlan(primary="revenue"), breakout=True)
    with pytest.raises(CodedError) as refused:
        analysis.run(exploratory_metrics=["nope"])
    refusals.append(refused.value)
    sequential = _sequential_analysis(warehouse)
    with pytest.raises(CodedError) as refused:
        sequential.run(exploratory_metrics=["purchase_rate"])
    refusals.append(refused.value)
    (declared,) = sequential.experiment.breakouts
    with pytest.raises(CodedError) as refused:
        sequential.dashboard_breakout_reads(declared).uncorrected_segments(metrics=["revenue"])
    refusals.append(refused.value)
    seam = Analysis.from_unit_summary(
        _unit_summary(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[_revenue_spec()],
    )
    with pytest.raises(CodedError) as refused:
        seam.run(exploratory_metrics=["purchase_rate"])
    refusals.append(refused.value)
    with pytest.raises(CodedError) as refused:
        seam.available_metrics  # noqa: B018
    refusals.append(refused.value)
    return refusals


def _unit_summary():
    con = ds.duckdb_connection()
    try:
        return ds.unit_summary_frame(con)
    finally:
        con.disconnect()


def _revenue_spec():
    from increment.frame import MetricSpec

    return MetricSpec(name="revenue", type="mean", missing="zero", preferred_direction="increase")


def test_every_new_refusal_survives_pickle_and_deepcopy(warehouse):
    refusals = _refusals_of_every_new_code(warehouse)
    assert {r.code for r in refusals} == {
        UNAVAILABLE,
        "facade.analysis.exploratory_metrics_sequential",
        "facade.analysis.uncorrected_segments_sequential",
        "facade.analysis.exploratory_metrics_source_limited",
    }
    for refusal in refusals:
        for clone in (copy.deepcopy(refusal), pickle.loads(pickle.dumps(refusal))):
            assert type(clone) is type(refusal)
            assert clone.code == refusal.code
            assert dict(clone.context) == dict(refusal.context)
            assert str(clone) == str(refusal)
