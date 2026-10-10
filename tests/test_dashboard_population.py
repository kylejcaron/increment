from __future__ import annotations

import datetime as dt

import ibis
import pytest

from increment import Analysis, SourceSnapshotEvidence
from increment.dashboard import (
    DashboardConfig,
    load_explore,
    prepare_dashboard,
    render_header,
    render_health,
    render_metric_details,
)
from increment.dashboard._app import build_payload
from increment.dashboard._data import estimate_for_metric, row_for_metric
from increment.errors import InvalidRequestError
from tests.test_analysis_trigger import _events, _write_defs

pytestmark = pytest.mark.filterwarnings("ignore::increment.errors.IncrementWarning")


def _triggered_dashboard(tmp_path, *, sequential=False, extra_metric=False):
    definitions = _write_defs(tmp_path, allocation_scheme="independent")
    plan = "primary: revenue\n      alternative: two-sided"
    if sequential:
        plan += "\n      inference:\n        kind: asymptotic_mean"
    definitions_text = definitions.read_text().replace("secondaries: [revenue]", plan)
    if extra_metric:
        definitions_text = definitions_text.replace(
            "experiments:",
            "  - name: trigger_conversion\n"
            "    type: conversion\n"
            "    entity: user_id\n"
            "    fact: saw_surface\n"
            "    window_days: 7\n"
            "experiments:",
            1,
        )
    definitions.write_text(definitions_text)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=40, trigger_rate=0.5, effect=0.8))
    return Analysis.from_definitions(
        "exp",
        definitions,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            observation_cutoff_ts=dt.datetime(2024, 1, 16, tzinfo=dt.UTC),
            complete_through_by_feed={"events": dt.datetime(2024, 1, 16, tzinfo=dt.UTC)},
        ),
    )


def _assigned_dashboard(tmp_path):
    definitions = _write_defs(tmp_path, trigger=None, allocation_scheme="independent")
    definitions.write_text(
        definitions.read_text().replace(
            "secondaries: [revenue]", "primary: revenue\n      alternative: two-sided"
        )
    )
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_events(n_per_arm=40, trigger_rate=0.5, effect=0.8))
    return Analysis.from_definitions("exp", definitions, con)


def test_dashboard_snapshot_retains_population_specific_headline_and_group_evidence(tmp_path):
    analysis = _triggered_dashboard(tmp_path)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        assert {row.analysis_population for row in snapshot.estimates} == {"triggered"}
        assert set(snapshot.population_estimates) == {"assigned", "triggered"}
        assert {row.analysis_population for row in snapshot.group_data} == {"assigned", "triggered"}
        assert {
            (row.analysis_population, row.metric, row.group_id) for row in snapshot.group_data
        } == {
            (population, "revenue", arm)
            for population in ("assigned", "triggered")
            for arm in ("C", "T")
        }
        default = estimate_for_metric(snapshot, "revenue")
        assigned = estimate_for_metric(snapshot, "revenue", population="assigned")
        assert default is not None and default.analysis_population == "triggered"
        assert assigned is not None and assigned.analysis_population == "assigned"
        default_row = row_for_metric(snapshot, "revenue")
        assigned_row = row_for_metric(snapshot, "revenue", population="assigned")
        assert default_row is not None and default_row["analysis_population"] == "triggered"
        assert assigned_row is not None and assigned_row["analysis_population"] == "assigned"
        payload = build_payload(analysis, snapshot=snapshot)
        assert payload["defaultPopulation"] == "triggered"
        assert payload["populationOptions"] == ["assigned", "triggered"]
        assert payload["populationViews"]["triggered"]["population"] == "triggered"
        assert payload["populationViews"]["assigned"]["population"] == "assigned"
        assert (
            payload["populationViews"]["triggered"]["readoutCsv"]
            != payload["populationViews"]["assigned"]["readoutCsv"]
        )
        for population in ("assigned", "triggered"):
            trajectory = load_explore(
                analysis,
                snapshot=snapshot,
                metric="revenue",
                view="cumulative_lift",
                population=population,
            )
            assert trajectory
            assert {row.analysis_population for row in trajectory} == {population}
    finally:
        analysis.close()


def test_dashboard_without_trigger_is_assigned_only(tmp_path):
    analysis = _assigned_dashboard(tmp_path)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        payload = build_payload(analysis, snapshot=snapshot)
        assert {row.analysis_population for row in snapshot.estimates} == {"assigned"}
        assert set(snapshot.population_estimates) == {"assigned"}
        assert payload["defaultPopulation"] == "assigned"
        assert payload["populationOptions"] == ["assigned"]
        assert set(payload["populationViews"]) == {"assigned"}
    finally:
        analysis.close()


def test_triggered_explore_refusal_has_population_specific_route_forward(tmp_path):
    import dataclasses

    from increment._source_operations import DashboardExploreCapture
    from increment.errors import CodedError

    analysis = _triggered_dashboard(tmp_path)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        key = ("triggered", "daily_values", None, False, None)
        refusal = CodedError(
            "run_daily_lift does not support triggered compliance estimates; use population='assigned'.",
            code="facade.analysis.triggered_compliance_unsupported",
            context={"method": "run_daily_lift"},
        )
        metric_key = ("triggered", "daily_values", "revenue", False, None)
        snapshot = dataclasses.replace(
            snapshot,
            explore={
                **snapshot.explore,
                key: DashboardExploreCapture.refused(key, refusal),
                metric_key: DashboardExploreCapture.refused(metric_key, refusal),
            },
        )
        payload = build_payload(analysis, snapshot=snapshot)
        triggered = payload["populationViews"]["triggered"]["explore"]["overall"]["revenue"][
            "daily_values"
        ]
        assigned = payload["populationViews"]["assigned"]["explore"]["overall"]["revenue"][
            "daily_values"
        ]
        assert triggered["routeForward"]
        assert "facade.analysis.triggered_compliance_unsupported" in triggered["caption"]
        assert triggered["pointCount"] == 0
        assert assigned["pointCount"] > 0
    finally:
        analysis.close()


def test_overview_exploratory_rows_stay_in_their_population_family(tmp_path):
    analysis = _triggered_dashboard(tmp_path, extra_metric=True)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        assert any(metric.name == "trigger_conversion" for metric in snapshot.offered_metrics)
        for population in ("assigned", "triggered"):
            overview = snapshot.overviews[population]
            assert overview.exploratory
            assert {row.analysis_population for row in overview.exploratory} == {population}
    finally:
        analysis.close()


def test_standalone_population_sections_use_one_population(tmp_path):
    analysis = _triggered_dashboard(tmp_path)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        triggered_header = render_header(snapshot).text
        assigned_header = render_header(snapshot, population="assigned").text
        assert ">40</p>" in triggered_header
        assert ">80</p>" in assigned_header

        triggered_details = render_metric_details(snapshot, metric="revenue").text
        assigned_details = render_metric_details(
            snapshot, metric="revenue", population="assigned"
        ).text
        for details in (triggered_details, assigned_details):
            assert details.count("data-inc-literal>C</th>") == 1
            assert details.count("data-inc-literal>T</th>") == 1
        assert snapshot.allocation is not None
        assert snapshot.triggered_allocation is not None
        assert set(snapshot.triggered_allocation.observed.values()) == {18, 22}
        health = render_health(snapshot).text
        assert ">40</td>" in health
        assert ">18</td>" in health
        assert ">22</td>" in health
    finally:
        analysis.close()


def test_triggered_health_uses_triggered_balance_result(tmp_path):
    import dataclasses
    import re

    analysis = _triggered_dashboard(tmp_path)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        assert snapshot.allocation is not None and not snapshot.allocation.is_srm
        assert snapshot.triggered_allocation is not None
        triggered = snapshot.triggered_allocation.model_copy(update={"is_srm": True})
        history = tuple(
            {
                "experiment_id": snapshot.experiment_name,
                "ds": snapshot.start + dt.timedelta(days=day),
                "group_id": arm,
                "n_daily": count,
                "n_cumulative": count,
            }
            for day in (0, 1)
            for arm, count in (
                (snapshot.control_group, 0),
                (snapshot.treatment_group, 10 if day else 0),
            )
        )
        snapshot = dataclasses.replace(
            snapshot,
            triggered_allocation=triggered,
            triggered_allocation_history=history,
        )
        assert snapshot.triggered_allocation is not None
        html = render_health(snapshot).text
        assert "27.8%" in html
        assert "72.2%" in html
        statuses = re.findall(r"inc-dashboard-status--(bad|ok)", html)
        assert statuses[:2] == ["ok", "bad"]
        assert set(snapshot.triggered_allocation.observed.values()) == {18, 22}
        assert ">40</td>" in html
        assert ">18</td>" in html
        assert ">22</td>" in html
    finally:
        analysis.close()


def test_triggered_allocation_history_is_immutable_and_refusal_consistent(tmp_path):
    import dataclasses

    analysis = _triggered_dashboard(tmp_path)
    try:
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        assert snapshot.triggered_allocation_history
        with pytest.raises(TypeError):
            snapshot.triggered_allocation_history[0]["n_daily"] = 0  # ty: ignore[invalid-assignment]
        with pytest.raises(InvalidRequestError) as caught:
            dataclasses.replace(
                snapshot,
                triggered_allocation_history_refusal=(
                    "source.native.operation",
                    "unsupported",
                ),
            )
        assert caught.value.code == "dashboard.invalid_snapshot"
    finally:
        analysis.close()


def test_explicit_triggered_sequential_run_returns_unavailable_triggered_rows(tmp_path):
    from increment.estimation.readout_types import ReadoutResults

    analysis = _triggered_dashboard(tmp_path, sequential=True)
    try:
        analysis.capture_sequential(finalized=True, as_of=dt.date(2024, 1, 16))
        results = analysis.run(population="triggered")
        assert results
        assert {result.analysis_population for result in results} == {"triggered"}
        assert {result.failure_code for result in results} == {"readout.cell.unsupported_request"}
        assert {result.failure_context["reason"] for result in results} == {"triggered_sequential"}
        assert all(result.lift is None for result in results)
        assigned = analysis.run(population="assigned")
        assert assigned.metadata is not None
        assert assigned.sequential_snapshot is not None
        restored = ReadoutResults.model_validate_json(assigned.model_dump_json())
        assert restored.metadata is not None
        assert restored.sequential_snapshot is not None
        assert {row.analysis_population for row in restored} == {"assigned"}
        assert all(
            family.analysis_population == "assigned" for family in restored.metadata.scope.families
        )
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        triggered_row = row_for_metric(snapshot, "revenue", population="triggered")
        assert triggered_row is not None
        assert triggered_row["failure_code"] == "readout.cell.unsupported_request"
        assert triggered_row["failure_context"]["reason"] == "triggered_sequential"
        details = render_metric_details(snapshot, metric="revenue").text
        assert "readout.cell.unsupported_request" in details
        assert "population='assigned'" in details
    finally:
        analysis.close()


@pytest.mark.slow
def test_triggered_sequential_group_data_does_not_borrow_assigned_checkpoints(tmp_path):
    analysis = _triggered_dashboard(tmp_path, sequential=True)
    try:
        analysis.capture_sequential(finalized=True, as_of=dt.date(2024, 1, 16))
        snapshot = prepare_dashboard(
            analysis,
            config=DashboardConfig(expected_allocation={"C": 0.5, "T": 0.5}),
        )
        assert {row.analysis_population for row in snapshot.group_data} == {
            "assigned",
            "triggered",
        }
        assert {
            row.source_kind for row in snapshot.group_data if row.analysis_population == "assigned"
        } == {"retained_checkpoint"}
        assert {
            row.source_kind for row in snapshot.group_data if row.analysis_population == "triggered"
        } == {"pinned_warehouse"}
    finally:
        analysis.close()
