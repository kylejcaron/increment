"""Behavioral witnesses for immutable and transformed group evidence."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date

import pytest

pytestmark = pytest.mark.slow


@contextmanager
def _cuped_native(tmp_path, *, sequential=False, predeclared=False):
    import ibis
    import pyarrow as pa

    from increment import Analysis
    from tests.warehouse_cuped_cases import definitions_yaml, event_rows, unit_rows

    units = unit_rows(seed=19, n=12)
    for unit in units:
        if unit["variant"] == "treatment":
            unit["pre"] += 3.0
            unit["revenue"] += 6.0
    plan = (
        "      primary:\n"
        "        metric: revenue\n"
        "        decision_method: {name: cuped, variance_reduction: cuped}\n"
    )
    if sequential:
        plan += "      inference:\n        kind: asymptotic_mean\n"
        if predeclared:
            plan += "        adjustments: {revenue: {coefficient: 2, center: 5}}\n"
    path = tmp_path / "definitions.yaml"
    path.write_text(definitions_yaml("duckdb", "events", n_pre_periods=7, plan=plan))
    connection = ibis.duckdb.connect()
    connection.create_table("events", pa.Table.from_pylist(event_rows(units)))
    try:
        with Analysis.from_definitions("exp", path, connection, store="none") as analysis:
            yield analysis, units
    finally:
        connection.disconnect()


def test_fixed_snapshot_keeps_old_rows_headline_and_csv_after_live_correction():
    import pyarrow as pa

    from increment.dashboard import DashboardConfig, group_data_csv, prepare_dashboard
    from increment.dashboard._data import group_data_rows
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(metric_names=("mean",)) as (connection, analysis):
        config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        old = prepare_dashboard(analysis, config=config)
        old_rows = group_data_rows(old, metric="mean")
        old_csv = group_data_csv(old, metric="mean")
        old_headline = next(row for row in old.estimates if row.metric == "mean")
        events = connection.table("events").to_pyarrow()
        corrected = events.to_pylist()
        for row in corrected:
            if row["event"] == "purchase" and row["unit_id"].startswith("candidate-"):
                row["value"] *= 2
        connection.create_table(
            "events", pa.Table.from_pylist(corrected, schema=events.schema), overwrite=True
        )
        fresh = prepare_dashboard(analysis, config=config)
        fresh_headline = next(row for row in fresh.estimates if row.metric == "mean")

        assert group_data_rows(old, metric="mean") == old_rows
        assert group_data_csv(old, metric="mean") == old_csv
        assert old_headline.require_lift().value == pytest.approx(6.5 / 4.5 - 1)
        assert fresh_headline.require_lift().value == pytest.approx(13 / 4.5 - 1)
        assert [row["observed_value"] for row in old_rows] == pytest.approx([4.5, 6.5])
        assert [
            row["observed_value"] for row in group_data_rows(fresh, metric="mean")
        ] == pytest.approx([4.5, 13])


def test_native_cuped_means_are_raw_and_adjusted_headline_matches_frame(tmp_path):
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from tests.warehouse_cuped_cases import frame_fixed

    with _cuped_native(tmp_path) as (analysis, units):
        snapshot = prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"control": 1, "treatment": 1})
        )
        unadjusted, adjusted = frame_fixed(units, "revenue")
        expected = {
            arm: sum(unit["revenue"] for unit in units if unit["variant"] == arm) / 12
            for arm in ("control", "treatment")
        }
        for row in snapshot.group_data:
            assert row.observed_value == pytest.approx(expected[row.group_id])
        lift = snapshot.estimates[0].require_lift().value
        assert lift == pytest.approx(adjusted.require_lift().value)
        assert lift != pytest.approx(unadjusted.require_lift().value)


def test_retained_winsorized_scalar_exposes_input_not_raw_outcomes():
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(sequential=True, winsor=True) as (_, analysis):
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        snapshot = prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        )
        rows = {row.group_id: row for row in snapshot.group_data if row.metric == "mean"}
        for arm, expected in (("baseline", 3.0), ("candidate", 4.0)):
            assert rows[arm].analysis_input_value == pytest.approx(expected)
            assert rows[arm].observed_value is None
            assert rows[arm].sum_value is None
            assert {"observed_value", "sum_value"} <= rows[arm].unavailable.keys()


def test_retained_joint_cuped_preserves_raw_y_not_residual(tmp_path):
    from increment.dashboard import DashboardConfig, prepare_dashboard

    with _cuped_native(tmp_path, sequential=True) as (analysis, units):
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 14))
        snapshot = prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"control": 1, "treatment": 1})
        )
        assert (
            snapshot.estimates[0].require_sequential_result().checkpoint.model.law
            == "adjusted_mean"
        )
        for row in snapshot.group_data:
            values = [unit["revenue"] for unit in units if unit["variant"] == row.group_id]
            assert row.eligible_units == len(values)
            assert row.observed_value == pytest.approx(sum(values) / len(values))
            assert row.sum_value == pytest.approx(sum(values))
            assert row.analysis_input_value is None


def test_predeclared_scalar_adjustment_does_not_invent_unretained_raw_values(tmp_path):
    from increment.dashboard import DashboardConfig, prepare_dashboard

    with _cuped_native(tmp_path, sequential=True, predeclared=True) as (analysis, units):
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 14))
        snapshot = prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"control": 1, "treatment": 1})
        )
        for row in snapshot.group_data:
            values = [
                unit["revenue"] - 2 * (unit["pre"] - 5)
                for unit in units
                if unit["variant"] == row.group_id
            ]
            assert row.eligible_units == len(values)
            assert row.analysis_input_value == pytest.approx(sum(values) / len(values))
            assert row.observed_value is None
            assert row.sum_value is None
            assert {"observed_value", "sum_value"} <= row.unavailable.keys()


def test_empty_retained_checkpoint_is_unavailable_not_a_zero_mean():
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(sequential=True, late_treatment=True) as (_, analysis):
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 5))
        snapshot = prepare_dashboard(
            analysis, config=DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        )
        candidate = next(row for row in snapshot.group_data if row.group_id == "candidate")
        assert candidate.eligible_units == 0
        assert candidate.observed_value is None
        assert candidate.analysis_input_value is None
        assert {"observed_value", "analysis_input_value"} <= candidate.unavailable.keys()
