"""Independent group-value, eligibility, and retained-evidence witnesses."""

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

pytestmark = pytest.mark.slow


@contextmanager
def _native_groups(
    *,
    extra_exclusions=False,
    late_treatment=False,
    denominator_average=False,
    winsor=False,
    large=False,
    sequential=False,
    metric_names=None,
):
    import ibis
    import pyarrow as pa

    from increment import Analysis
    from increment.semantics.models import Definitions

    common = {"entity": "unit_id", "preferred_direction": "increase"}
    metrics = [
        {
            **common,
            "name": "conversion",
            "type": "conversion",
            "fact": "purchase",
            "window_days": 4,
        },
        {
            **common,
            "name": "mean",
            "type": "mean",
            "fact": "purchase",
            "aggregation": "sum",
            "window_days": 4,
            **({"winsorization": {"upper_value": 5}} if winsor else {}),
        },
        {
            **common,
            "name": "retention",
            "type": "retention",
            "fact": "return",
            "threshold_days": [2, 4],
        },
        {
            **common,
            "name": "ratio",
            "type": "ratio",
            "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 4},
            "denominator": {
                "fact": "denominator",
                "aggregation": "avg_event" if denominator_average else "sum",
                "window_days": 4,
            },
        },
        {
            **common,
            "name": "quantile",
            "type": "quantile",
            "fact": "purchase",
            "aggregation": "sum",
            "quantile": 0.25,
        },
    ]
    plan: dict[str, object] = {
        "primary": "conversion",
        "secondaries": ["mean", "retention", "ratio", "quantile"],
    }
    if sequential:
        metrics = metrics[:2]
        plan = {
            "primary": "conversion",
            "secondaries": ["mean"],
            "inference": {"kind": "asymptotic_mean"},
        }
    if metric_names is not None:
        metrics = [metric for metric in metrics if metric["name"] in metric_names]
        plan["primary"] = metrics[0]["name"]
        plan["secondaries"] = [metric["name"] for metric in metrics[1:]]
    definitions = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": name, "column": "value"}
                        for name in ("purchase", "return", "denominator")
                    ],
                }
            ],
            "exposures": [
                {"name": "assigned", "sql": "SELECT unit_id, ts, group_id FROM enrolled"}
            ],
            "metrics": metrics,
            "experiments": [
                {
                    "name": "groups",
                    "exposure": "assigned",
                    "unit": "unit_id",
                    "control_group": "baseline",
                    "start": "2025-01-01",
                    "end": None,
                    "n_pre_periods": 0,
                    "allocation": {"baseline": 0.5, "candidate": 0.5},
                    "plan": plan,
                }
            ],
        }
    )
    enrolled, events = [], []
    for group, shift in (("baseline", 0), ("candidate", 2)):
        for index in range(16):
            unit = f"{group}-{index}"
            start = datetime(
                2025, 1, 9 if late_treatment and group == "candidate" else 1, tzinfo=UTC
            )
            enrolled.append({"unit_id": unit, "group_id": group, "ts": start})
            value = 1e308 if large else (0, 2, 6, 10)[index % 4] + shift
            if value:
                events.append(
                    {
                        "unit_id": unit,
                        "event": "purchase",
                        "value": value,
                        "ts": start + timedelta(days=1),
                    }
                )
            if not denominator_average or index % 4:
                events.append(
                    {
                        "unit_id": unit,
                        "event": "denominator",
                        "value": float(index % 4 + 1),
                        "ts": start + timedelta(days=1),
                    }
                )
            if index % 2 == 0:
                events.append(
                    {
                        "unit_id": unit,
                        "event": "return",
                        "value": 1.0,
                        "ts": start + timedelta(days=2),
                    }
                )
    if extra_exclusions:
        for name, day in (("immature", 9), ("unobserved", 22)):
            enrolled.append(
                {"unit_id": name, "group_id": "baseline", "ts": datetime(2025, 1, day, tzinfo=UTC)}
            )
    for fact in ("purchase", "return", "denominator"):
        events.append(
            {
                "unit_id": "not-enrolled",
                "event": fact,
                "value": 99.0,
                "ts": datetime(2025, 1, 10, tzinfo=UTC),
            }
        )
    connection = ibis.duckdb.connect()
    connection.create_table("enrolled", pa.Table.from_pylist(enrolled))
    connection.create_table("events", pa.Table.from_pylist(events))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "definitions.yaml"
        path.write_text(definitions.model_dump_json())
        analysis = Analysis.from_definitions("groups", path, connection)
        try:
            yield connection, analysis
        finally:
            analysis.close()
            connection.disconnect()


def _capture(analysis, *names):
    metrics = [metric for metric in analysis.metrics if not names or metric.name in names]
    return {
        (row.metric, row.group_id): row for row in analysis.dashboard_group_data(metrics=metrics)
    }


def test_running_source_without_observations_explains_missing_cutoff():
    with _native_groups(metric_names=("mean",)) as (connection, analysis):
        connection.raw_sql("DELETE FROM events")
        rows = _capture(analysis, "mean")
    assert {
        (row.group_id, row.assigned_units, row.eligible_units, row.observation_end)
        for row in rows.values()
    } == {("baseline", 16, 0, None), ("candidate", 16, 0, None)}
    for row in rows.values():
        assert "observation_end" in row.unavailable


@pytest.mark.parametrize("dialect", ["duckdb", "postgres", "snowflake", "bigquery"])
def test_group_quantiles_preserve_linear_interpolation_through_backend_compilers(dialect):
    import ibis
    import pyarrow as pa
    import sqlglot

    from increment.query.native_source import _dashboard_quantiles

    samples = {
        "control": [0.75 + index / 2 for index in range(30)],
        "treatment": [1.375 + index / 4 for index in range(30)],
        "single": [8.0],
        "nullable": [None, 2.0, 10.0],
        "extreme": [-1e308, 1e308],
    }
    records = [
        {"group_id": group, "y": value} for group, values in samples.items() for value in values
    ]
    connection = ibis.duckdb.connect()
    try:
        table = connection.create_table("quantile_units", pa.Table.from_pylist(records[::-1]))
        query = _dashboard_quantiles(table, 0.25)
        compiled = ibis.to_sql(query, dialect=dialect)
        executable = sqlglot.transpile(str(compiled), read=dialect, write="duckdb")[0]
        rows = connection.sql(executable).to_pyarrow().to_pylist()
        assert {row["group_id"]: row["quantile_value"] for row in rows} == pytest.approx(
            {
                "control": 4.375,
                "treatment": 3.1875,
                "single": 8.0,
                "nullable": 4.0,
                "extreme": -5e307,
            },
            rel=1e-14,
        )
    finally:
        connection.disconnect()


def test_native_group_values_match_independent_five_kind_oracle():
    import numpy as np

    with _native_groups() as (_, analysis):
        rows = _capture(analysis)
    for group, shift in (("baseline", 0), ("candidate", 2)):
        values = np.tile([0, 2, 6, 10], 4) + shift
        denominators = np.tile([1, 2, 3, 4], 4)
        expected = {
            "conversion": float(np.mean(values > 0)),
            "mean": float(values.mean()),
            "retention": 0.5,
            "ratio": float(values.sum() / denominators.sum()),
            "quantile": float(np.quantile(values, 0.25, method="linear")),
        }
        for metric, value in expected.items():
            row = rows[metric, group]
            assert row.observed_value == pytest.approx(value)
            assert row.assigned_units == row.eligible_units == 16
            assert (
                row.excluded_no_observed_day == row.excluded_not_mature == row.excluded_other == 0
            )
            assert row.observation_end == date(2025, 1, 10)
        assert rows["mean", group].sum_value == pytest.approx(float(values.sum()))
        assert rows["ratio", group].numerator == pytest.approx(float(values.sum()))
        assert rows["ratio", group].denominator == 40
        assert rows["conversion", group].retained_units == int(np.count_nonzero(values))
        assert rows["conversion", group].event_count == int(np.count_nonzero(values))
        assert rows["retention", group].retained_units == rows["retention", group].event_count == 8
        assert (
            rows["retention", group].window_start_days,
            rows["retention", group].window_end_days,
        ) == (2, 4)
        assert (rows["mean", group].window_start_days, rows["mean", group].window_end_days) == (
            0,
            4,
        )


def test_maturity_and_absent_observation_exclusions_are_disjoint():
    with _native_groups(extra_exclusions=True) as (_, analysis):
        row = _capture(analysis, "retention")["retention", "baseline"]
    assert (row.assigned_units, row.eligible_units) == (18, 16)
    assert (row.excluded_not_mature, row.excluded_no_observed_day, row.excluded_other) == (1, 1, 0)
    assert row.retained_units == 8
    assert row.observed_value == 0.5


def test_fully_immature_arm_keeps_accounting_and_null_value():
    with _native_groups(late_treatment=True) as (_, analysis):
        row = _capture(analysis, "retention")["retention", "candidate"]
    assert row.assigned_units == row.excluded_not_mature == 16
    assert row.eligible_units == row.retained_units == row.event_count == 0
    assert row.excluded_no_observed_day == row.excluded_other == 0
    assert row.observed_value is None
    assert "observed_value" in row.unavailable


def test_ratio_events_follow_joint_denominator_eligibility():
    with _native_groups(denominator_average=True) as (_, analysis):
        row = _capture(analysis, "ratio")["ratio", "candidate"]
    assert (row.assigned_units, row.eligible_units, row.excluded_other) == (16, 12, 4)
    assert row.event_count == 12
    assert row.numerator == 96
    assert row.denominator == 36
    assert row.observed_value == pytest.approx(96 / 36)


def test_raw_group_values_precede_winsorization():
    with _native_groups(winsor=True) as (_, analysis):
        row = _capture(analysis, "mean")["mean", "candidate"]
    assert row.observed_value == 6.5
    assert row.sum_value == 104


def test_finite_group_mean_survives_unrepresentable_display_total():
    with _native_groups(large=True) as (_, analysis):
        row = _capture(analysis, "mean")["mean", "baseline"]
    assert row.observed_value == pytest.approx(1e308)
    assert row.sum_value is None
    assert "sum_value" in row.unavailable


@pytest.mark.parametrize("law", ["bernoulli", "scalar_mean"])
def test_retained_checkpoint_values_survive_changed_live_outcomes(law, tmp_path):
    import pyarrow as pa

    from increment import Analysis
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from tests.test_sequential_public_sources import _native_fixture

    connection, definitions, analysis = _native_fixture(law)
    analysis.close()
    try:
        path = tmp_path / "definitions.yaml"
        path.write_text(definitions.model_dump_json())
        analysis = Analysis.from_definitions("experiment", path, connection)
        connection.raw_sql(
            "INSERT INTO events (unit_id, event, value, ts) VALUES "
            "('freshness-only', 'outcome_event', 0, TIMESTAMPTZ '2025-01-20 00:00:00+00')"
        )
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        config = DashboardConfig(expected_allocation={"control": 1, "treatment": 1})
        snapshot = prepare_dashboard(analysis, config=config)
        checkpoints = {
            estimate.metric: estimate.require_sequential_result().checkpoint
            for estimate in snapshot.estimates
        }
        events = connection.table("events").to_pyarrow()
        ghost = [row for row in events.to_pylist() if row["unit_id"] == "freshness-only"]
        connection.create_table(
            "events", pa.Table.from_pylist(ghost, schema=events.schema), overwrite=True
        )
        rows = analysis.dashboard_group_data(metrics=analysis.metrics, checkpoints=checkpoints)
        assert rows == snapshot.group_data
        expected = (
            {"control": 0.25, "treatment": 0.75}
            if law == "bernoulli"
            else {"control": 2.5, "treatment": 12.0}
        )
        for row in rows:
            assert row.eligible_units == 96
            assert row.observed_value == pytest.approx(expected[row.group_id])
            assert row.source_kind == "retained_checkpoint"
            assert row.prefix_id == checkpoints[row.metric].prefix_id
            assert row.assigned_units is None
            assert row.event_count is None
            assert {"assigned_units", "event_count"} <= row.unavailable.keys()
    finally:
        analysis.close()
        connection.disconnect()


def test_frozen_secondary_keeps_its_earlier_displayed_prefix():
    import pyarrow as pa

    from increment.dashboard import DashboardConfig, prepare_dashboard

    with _native_groups(sequential=True) as (connection, analysis):
        early = analysis.capture_sequential(
            finalized=True, as_of=date(2025, 1, 10), freeze=["mean"]
        )
        config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        before = prepare_dashboard(analysis, config=config)
        enrollment = connection.table("enrolled").to_pyarrow()
        events = connection.table("events").to_pyarrow()
        new_enrollment, new_events = enrollment.to_pylist(), events.to_pylist()
        for arm, value in (("baseline", 100.0), ("candidate", 200.0)):
            unit = f"{arm}-new"
            new_enrollment.append(
                {"unit_id": unit, "group_id": arm, "ts": datetime(2025, 1, 11, tzinfo=UTC)}
            )
            new_events.append(
                {
                    "unit_id": unit,
                    "event": "purchase",
                    "value": value,
                    "ts": datetime(2025, 1, 12, tzinfo=UTC),
                }
            )
        new_events.append(
            {
                "unit_id": "freshness-next",
                "event": "purchase",
                "value": 0.0,
                "ts": datetime(2025, 1, 20, tzinfo=UTC),
            }
        )
        connection.create_table(
            "enrolled",
            pa.Table.from_pylist(new_enrollment, schema=enrollment.schema),
            overwrite=True,
        )
        connection.create_table(
            "events", pa.Table.from_pylist(new_events, schema=events.schema), overwrite=True
        )
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 20), previous=early)
        after = prepare_dashboard(analysis, config=config)
        old = {(row.metric, row.group_id): row for row in before.group_data}
        for row in after.group_data:
            if row.metric == "mean":
                assert row == old[row.metric, row.group_id]
                assert row.eligible_units == 16
            else:
                assert row.eligible_units == 17
                assert row.prefix_id != old[row.metric, row.group_id].prefix_id


def test_zero_ratio_denominator_preserves_other_available_evidence():
    with _native_groups() as (connection, analysis):
        connection.raw_sql("UPDATE events SET value = 0 WHERE event = 'denominator'")
        row = _capture(analysis, "ratio")["ratio", "baseline"]
    assert row.eligible_units == 16
    assert row.numerator == pytest.approx(72)
    assert row.denominator == 0
    assert row.observed_value is None
    assert "observed_value" in row.unavailable


def test_group_capture_materializes_only_bounded_aggregate_rows(monkeypatch):
    with _native_groups() as (connection, analysis):
        original = connection.to_pyarrow
        row_counts = []

        def capture(*args, **kwargs):
            table = original(*args, **kwargs)
            row_counts.append(table.num_rows)
            assert "unit_id" not in table.column_names
            return table

        monkeypatch.setattr(connection, "to_pyarrow", capture)
        rows = _capture(analysis)
    assert len(rows) == 10
    assert max(row_counts) <= 2


@pytest.mark.parametrize("kind", ["swapped_metric", "foreign_prefix"])
def test_group_capture_rejects_unbound_checkpoints_before_data_access(kind, monkeypatch):
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from increment.errors import CapabilityError

    config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
    with _native_groups(sequential=True) as (connection, analysis):
        analysis.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
        snapshot = prepare_dashboard(analysis, config=config)
        checkpoints = {
            row.metric: row.require_sequential_result().checkpoint for row in snapshot.estimates
        }
        if kind == "swapped_metric":
            checkpoints = {"mean": checkpoints["conversion"], "conversion": checkpoints["mean"]}
            code = "source.native.operation"
        else:
            with _native_groups(sequential=True, extra_exclusions=True) as (_, other):
                other.capture_sequential(finalized=True, as_of=date(2025, 1, 16))
                foreign = prepare_dashboard(other, config=config)
                checkpoints = {
                    row.metric: row.require_sequential_result().checkpoint
                    for row in foreign.estimates
                }
            code = "sequential.continuation.rewrite"

        def unexpected_read(*args, **kwargs):
            pytest.fail("checkpoint validation must precede warehouse reads")

        monkeypatch.setattr(connection, "to_pyarrow", unexpected_read)
        with pytest.raises(CapabilityError) as caught:
            analysis.dashboard_group_data(metrics=analysis.metrics, checkpoints=checkpoints)
        assert caught.value.code == code


@pytest.mark.parametrize(
    "kind, code",
    [
        ("unknown", "facade.analysis_config.unknown_metric_declared"),
        ("changed", "facade.analysis_config.metric_definition_mismatch"),
        ("duplicate", "facade.analysis_config.duplicate_metric_name"),
    ],
)
def test_group_metric_selection_is_validated_before_source_access(kind, code, monkeypatch):
    from increment.errors import InvalidRequestError

    with _native_groups(metric_names=("mean",)) as (connection, analysis):
        metric = analysis.metrics[0]
        requested = {
            "unknown": [metric.model_copy(update={"name": "undeclared"})],
            "changed": [metric.model_copy(update={"window_days": 1})],
            "duplicate": [metric, metric],
        }[kind]

        def unexpected_read(*args, **kwargs):
            pytest.fail("metric validation must precede warehouse reads")

        monkeypatch.setattr(connection, "to_pyarrow", unexpected_read)
        with pytest.raises(InvalidRequestError) as caught:
            analysis.dashboard_group_data(metrics=requested)
        assert caught.value.code == code
