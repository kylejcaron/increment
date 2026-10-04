"""Metrics added after the plan: ``Analysis.available_metrics`` and ``exploratory_metrics=``.

Real DuckDB fixtures throughout. An added metric must read exactly like the same
metric declared in a plan of its own, never move a declared row, and be refused
before any query when it cannot be served.
"""

from __future__ import annotations

import copy
import pickle
from collections.abc import Callable
from typing import Any

import pytest

from increment import Analysis
from increment.errors import CodedError
from increment.estimation.engine import Method
from increment.plan import bind_automatic_sequential_plan
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan, Definitions, MultiplicitySpec
from increment.sequential_source import native_observation_mapping
from tests.analysis_factory import lift_rows, make_analysis
from tests.parity_harness import dataset as ds

ADDED = ("purchase_rate", "rps", "revenue_cuped")
CUPED = Method(name="cuped", variance_reduction="cuped")
UNAVAILABLE = "facade.analysis_config.exploratory_metric_unavailable"


def _definitions(plan: AnalysisPlan, *, breakout: bool = False) -> Definitions:
    return Definitions.model_validate(ds.definitions_dict(plan=plan, breakout=breakout))


def _analysis(con: Any, plan: AnalysisPlan, *, breakout: bool = False) -> Analysis:
    return make_analysis(con, _definitions(plan, breakout=breakout), experiment="exp")


def _width(row: Any) -> float:
    lift = row.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return lift.ub - lift.lb


def _rows(rows: Any, *, drop: tuple[str, ...] = ("role",)) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.model_dump().items() if k not in drop} for row in rows]


@pytest.fixture(scope="module")
def warehouse():
    con = ds.duckdb_connection()
    yield con
    con.disconnect()


def test_available_metrics_are_the_undeclared_definitions_in_definitions_order(warehouse):
    analysis = _analysis(warehouse, AnalysisPlan(primary="purchase_rate", secondaries=["rps"]))
    assert [m.name for m in analysis.available_metrics] == ["revenue", "revenue_cuped"]


@pytest.mark.parametrize(
    ("name", "call_wide"),
    [(name, {}) for name in ADDED] + [("revenue_cuped", {"decision_method": CUPED})],
    ids=[*ADDED, "revenue_cuped-cuped"],
)
def test_added_whole_window_rows_equal_the_metric_declared_in_its_own_plan(
    warehouse, name, call_wide
):
    declared = _analysis(warehouse, AnalysisPlan(primary=name)).run(**call_wide)
    added = _analysis(warehouse, AnalysisPlan(primary="revenue")).run(
        metrics=[], exploratory_metrics=[name], **call_wide
    )
    assert declared and {row.role for row in declared} == {"primary"}
    assert {row.role for row in added} == {"exploratory"}
    assert _rows(added) == _rows(declared)


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
    assert all(row.discovery is None and row.family_axes is None for row in added)
    narrowed = analysis.run(metrics=["revenue"], exploratory_metrics=["rps"])
    assert {row.metric for row in narrowed} == {"revenue", "rps"}
    assert [row.model_dump() for row in narrowed if row.metric == "revenue"] == [
        row.model_dump() for row in without if row.metric == "revenue"
    ]


@pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
@pytest.mark.parametrize("name", ADDED)
def test_added_breakout_rows_equal_the_metric_declared_in_its_own_plan(warehouse, name, correction):
    view = MultiplicitySpec(correction=correction)
    declared = _analysis(
        warehouse, AnalysisPlan(primary=name, view_multiplicity=view), breakout=True
    ).run_breakout()
    added = _analysis(
        warehouse, AnalysisPlan(primary="revenue", view_multiplicity=view), breakout=True
    ).run_breakout(metrics=[], exploratory_metrics=[name])
    assert declared and {row.role for row in added} == {"exploratory"}
    assert _rows(added) == _rows(declared)


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


@pytest.mark.parametrize("name", ADDED)
def test_added_day_axis_rows_equal_the_metric_declared_in_its_own_plan(warehouse, name):
    declared = _analysis(warehouse, AnalysisPlan(primary=name))
    added = _analysis(warehouse, AnalysisPlan(primary="revenue"))
    lift = added.run_asof_lift(metrics=[], exploratory_metrics=[name])
    assert lift and {row.role for row in lift} == {"exploratory"}
    assert _rows(lift) == _rows(declared.run_asof_lift())
    for read in ("run_asof", "run_daily"):
        values = getattr(added, read)(metrics=[], exploratory_metrics=[name])
        assert values and _rows(values) == _rows(getattr(declared, read)())
    both = added.run_asof_lift(exploratory_metrics=[name])
    assert [row.role for row in both if row.metric == "revenue"] == [
        row.role for row in added.run_asof_lift() if row.metric == "revenue"
    ]


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
