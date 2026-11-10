"""Independent group-value, eligibility, and retained-evidence witnesses."""

from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pytest

from increment._dashboard_capture import dashboard_group_data
from increment.workflows import capture_sequential

pytestmark = pytest.mark.slow


def _group_event_values(
    enrolled, events, *, occurrence_labels, late_fact, null_values, filtered_domain, domain
):
    if late_fact:
        for fact, day in (("late", 15), ("later", 17)):
            events.append(
                {
                    "unit_id": "not-enrolled",
                    "event": fact,
                    "value": 1.0,
                    "ts": datetime(2025, 1, day, tzinfo=UTC),
                }
            )
    if null_values:
        events.append(
            {
                "unit_id": "baseline-0",
                "event": "purchase",
                "value": None,
                "ts": datetime(2025, 1, 2, tzinfo=UTC),
            }
        )
    if filtered_domain:
        events.append(
            {
                "unit_id": "not-enrolled",
                "event": "purchase",
                "value": -1.0,
                "ts": datetime(2025, 1, 15, tzinfo=UTC),
            }
        )
    if domain == "anchor_collision":
        for row in enrolled:
            row["ts"] += timedelta(hours=12)
        events.append(
            {
                "unit_id": "baseline-1",
                "event": "purchase",
                "value": 900.0,
                "ts": datetime(2025, 1, 1, 11, tzinfo=UTC),
            }
        )
        for row in events:
            row["first_exposure_ts"] = datetime(2024, 12, 1, tzinfo=UTC)
    for row in events:
        row["price_value"] = row["value"]
        row["declared_segment"] = row["unit_id"]
    if occurrence_labels:
        for index, row in enumerate(events):
            row["value"] = f"order-{index}"
    if domain == "value_collision":
        for row in events:
            row["ds"] = row.pop("value")


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
    fact_missing="error",
    null_values=False,
    occurrence_labels=False,
    unequal_freshness=False,
    domain=None,
):
    import ibis
    import pyarrow as pa

    from increment import from_definitions
    from increment.semantics.models import Definitions

    late_fact, filtered_domain = domain in ("temporal", "total"), domain == "total"

    common = {
        "entity": "unit_id",
        "preferred_direction": "increase",
        **(
            {"filters": [{"property": "price", "op": "gt", "values": [0]}]}
            if filtered_domain
            else {}
        ),
    }
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
    value_column = "ds" if domain == "value_collision" else "value"
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
                        {"name": name, "column": value_column, "missing": fact_missing}
                        for name in (
                            "purchase",
                            "return",
                            "denominator",
                            *(("late", "later") if late_fact else ()),
                        )
                    ],
                    "properties": [
                        {
                            "name": "price",
                            "column": "price_value",
                            "dtype": "float",
                            "as_of": "static",
                        },
                        {
                            "name": "declared_segment",
                            "column": "declared_segment",
                            "dtype": "string",
                            "as_of": "static",
                        },
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
                    "breakouts": {
                        "breakout": [{"property": "declared_segment", "source": "events"}]
                    }.get(domain, []),
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
    if filtered_domain:
        enrolled.append(
            {
                "unit_id": "beyond-original",
                "group_id": "baseline",
                "ts": datetime(2025, 1, 12, tzinfo=UTC),
            }
        )
    for fact in ("purchase", "return", "denominator"):
        events.append(
            {
                "unit_id": "not-enrolled",
                "event": fact,
                "value": 99.0,
                "ts": datetime(
                    2025, 1, 3 if unequal_freshness and fact != "purchase" else 10, tzinfo=UTC
                ),
            }
        )
    _group_event_values(
        enrolled,
        events,
        occurrence_labels=occurrence_labels,
        late_fact=late_fact,
        null_values=null_values,
        filtered_domain=filtered_domain,
        domain=domain,
    )
    connection = ibis.duckdb.connect()
    connection.create_table("enrolled", pa.Table.from_pylist(enrolled))
    connection.create_table("events", pa.Table.from_pylist(events))
    with TemporaryDirectory() as directory:
        path = Path(directory) / "definitions.yaml"
        path.write_text(definitions.model_dump_json())
        experiment, resolver = from_definitions(path, connection, experiment_name="groups")
        try:
            yield connection, experiment, resolver
        finally:
            cast(Any, resolver).close()
            connection.disconnect()


def _capture(resolver, *names):
    binding = resolver._binding
    metrics = [metric for metric in binding.metrics if not names or metric.name in names]
    return {
        (row.metric, row.group_id): row for row in dashboard_group_data(binding, metrics=metrics)
    }


def _snowflake_no_match_fetch(connection, query):
    """Execute conditional counts with Snowflake's NULL-on-no-match semantics."""
    import ibis
    import pyarrow as pa
    import sqlglot
    from sqlglot import exp

    statement = sqlglot.parse_one(ibis.to_sql(query, dialect="duckdb"), read="duckdb")
    converted = 0
    for conditional in tuple(statement.find_all(exp.Filter)):
        if isinstance(conditional.this, exp.Count) and isinstance(conditional.this.this, exp.Star):
            converted += 1
            conditional.replace(
                exp.Nullif(this=conditional.copy(), expression=exp.Literal.number(0))
            )
    assert "COUNT_IF(" not in ibis.to_sql(query, dialect="snowflake") or converted
    cursor = connection.raw_sql(statement.sql("duckdb"))
    columns = [column[0] for column in cursor.description]
    rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
    return pa.Table.from_pylist(rows, schema=query.schema().to_pyarrow())


def _assert_display_parity(source, metrics):
    from dataclasses import fields

    independent = [
        row for metric in metrics for row in source.dashboard_group_data(metrics=(metric,))
    ]
    batched = source.dashboard_group_data(metrics=metrics)

    def order(row):
        return row.metric, row.group_id, row.eligible_units

    for got, reference in zip(
        sorted(batched, key=order), sorted(independent, key=order), strict=True
    ):
        for field in fields(reference):
            value = getattr(reference, field.name)
            assert getattr(got, field.name) == (
                pytest.approx(value) if isinstance(value, float) else value
            )


def _assert_temporal_parity(day_source, metrics, grain, completed):
    from increment.evidence import DayAxisRequest, evidence_provider, request_evidence

    def acquire(selected):
        answer = request_evidence(
            evidence_provider(day_source),
            DayAxisRequest(
                caller="run_daily" if grain == "daily" else "run_asof",
                grain=grain,
                metrics=tuple(selected),
                dimension=None,
                completed_windows_only=completed,
            ),
        )
        return [row for part in cast(Any, answer).slices for row in part.rows]

    individual = tuple(acquire((metric,)) for metric in metrics)
    combined = acquire(metrics)
    batched = tuple([row for row in combined if row["metric"] == metric.name] for metric in metrics)

    def identity(row):
        return row["ds"], row["group_id"], row["metric"]

    for got, independent in zip(batched, individual, strict=True):
        got, independent = sorted(got, key=identity), sorted(independent, key=identity)
        assert len(got) == len(independent)
        for row, reference in zip(got, independent, strict=True):
            assert row.keys() == reference.keys()
            for key, value in reference.items():
                assert row[key] == (
                    pytest.approx(value, rel=1e-10, abs=1e-10)
                    if isinstance(value, float)
                    else value
                )


def _assert_total_domain_parity(source, metrics):
    from increment.evidence import MomentsRequest, evidence_provider, request_evidence

    independent = tuple(
        cast(Any, request_evidence(evidence_provider(source), MomentsRequest(metric)))
        for metric in metrics
    )
    assert [
        next(row["n"] for row in rows if row["group_id"] == "baseline") for rows in independent
    ] == [17, 18]
    _assert_display_parity(source, metrics)


def _assert_total_readout_uncertainty(source, metrics):
    from math import sqrt

    from increment import readouts
    from increment.evidence import MetricEvidenceRequest, evidence_provider, request_evidence

    def arm_variance(row):
        n = row["n"]
        var_num = (row["cy2"] - row["cy1"] ** 2 / n) / (n - 1)
        if row["ref_den"] is None:
            return var_num / n
        mean_num = row["ref_y"] + row["cy1"] / n
        mean_den = row["ref_den"] + row["cden1"] / n
        ratio = mean_num / mean_den
        var_den = (row["cden2"] - row["cden1"] ** 2 / n) / (n - 1)
        cov = (row["cyden"] - row["cy1"] * row["cden1"] / n) / (n - 1)
        return (var_num - 2 * ratio * cov + ratio**2 * var_den) / (n * mean_den**2)

    independent = {}
    for metric in metrics:
        answer = cast(
            Any,
            request_evidence(evidence_provider(source), MetricEvidenceRequest(metric, "baseline")),
        )
        independent[metric.name] = {row["group_id"]: row for row in answer.rows}
    batched = readouts.run(source, metrics=[metric.name for metric in metrics])
    assert {
        (
            row.metric,
            row.group_id,
            row.method,
            row.method_role,
            row.estimand,
            row.analysis_population,
            row.role,
        )
        for row in batched
    } == {
        (
            metric.name,
            "candidate",
            "unadjusted",
            "decision",
            "itt",
            "assigned",
            source.context.plan.procedures[metric.name].role,
        )
        for metric in metrics
    }
    for row in batched:
        control, treatment = (
            independent[row.metric]["baseline"],
            independent[row.metric]["candidate"],
        )
        assert row.failure_code is None
        assert (row.n_control, row.n_treat) == (control["n"], treatment["n"])
        assert row.abs_se == pytest.approx(
            sqrt(arm_variance(control) + arm_variance(treatment)), rel=1e-10, abs=1e-10
        )


@pytest.mark.integration
def test_run_refuses_nonempty_by_instead_of_dropping_segments():
    from increment import readouts
    from increment.errors import UnsupportedRequestError

    with _native_groups(domain="breakout", metric_names=("conversion", "mean")) as (_, _, resolver):
        with pytest.raises(UnsupportedRequestError) as raised:
            readouts.run(resolver._binding.source, by=("declared_segment",))

        assert raised.value.code == "readout.run.segment_unsupported"
        assert raised.value.context == {"metric": "conversion"}


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:.*censoring dropped:increment.errors.IncrementWarning")
@pytest.mark.parametrize(
    "missing,snowflake,domain",
    [
        ("zero", False, None),
        ("skip", False, None),
        ("error", False, None),
        ("zero", True, None),
        ("zero", False, "temporal"),
        ("zero", False, "total"),
        ("zero", False, "anchor_collision"),
    ],
)
def test_shared_order_components_keep_request_identity_maturity_and_ratio_covariance(
    missing, monkeypatch, snowflake, domain
):
    from increment import readouts
    from increment.errors import CodedError
    from increment.evidence import MetricEvidenceRequest, evidence_provider, request_evidence
    from increment.semantics.models import Filter, RatioMetric

    with _native_groups(
        extra_exclusions=True,
        metric_names=("conversion", "mean", "retention"),
        domain=domain,
        fact_missing=missing,
        null_values=True,
        unequal_freshness=True,
    ) as (connection, _, resolver):
        source = resolver._binding.source
        conversion, revenue, disjoint = resolver._binding.metrics
        conversion = conversion.model_copy(update={"window_days": 7})
        revenue = revenue.model_copy(update={"window_days": 1 if domain == "total" else 7})
        if domain == "total":
            _assert_total_domain_parity(
                source, (revenue, revenue.model_copy(update={"fact": "late"}))
            )
            return
        if domain == "temporal":
            late_metrics = tuple(
                metric.model_copy(update={"name": f"late_{metric.name}", "fact": "late"})
                for metric in (conversion, revenue)
            )
            later = revenue.model_copy(update={"name": "later_mean", "fact": "later"})
            day_source = source.day_source(metrics=(*late_metrics, later, revenue, disjoint))
            for grain, completed in (("daily", False), ("asof", False), ("asof", True)):
                _assert_temporal_parity(
                    day_source, (*late_metrics, later, revenue), grain, completed
                )
            return
        long_revenue = revenue.model_copy(update={"name": "long_mean", "window_days": 30})
        ratio = RatioMetric(
            name="revenue_per_order",
            entity="unit_id",
            preferred_direction="increase",
            numerator={"fact": "purchase", "aggregation": "sum", "window_days": 7},
            denominator={"fact": "purchase", "aggregation": "count", "window_days": 7},
        )
        filtered = revenue.model_copy(
            update={
                "name": "expensive",
                "filters": (Filter(property="price", op="gt", values=(5,)),),
            }
        )
        other_ratio = ratio.model_copy(
            update={
                "name": "revenue_per_denominator",
                "denominator": ratio.denominator.model_copy(
                    update={"fact": "denominator", "aggregation": "sum"}
                ),
            }
        )
        unbounded = other_ratio.model_copy(
            update={
                "name": "unbounded",
                "numerator": other_ratio.numerator.model_copy(update={"window_days": None}),
                "denominator": other_ratio.denominator.model_copy(update={"window_days": None}),
            }
        )
        source = source.exploratory_source(
            metrics=(ratio, filtered, other_ratio, unbounded, long_revenue)
        )
        if snowflake:
            monkeypatch.setattr(
                connection, "to_pyarrow", partial(_snowflake_no_match_fetch, connection)
            )
        metrics = (
            conversion,
            revenue,
            ratio,
            long_revenue,
            disjoint,
            filtered,
            other_ratio,
            unbounded,
        )
        requests = (
            *(MetricEvidenceRequest(metric, "baseline") for metric in metrics),
            MetricEvidenceRequest(conversion, "candidate"),
        )
        if missing == "error":
            with pytest.raises(CodedError) as single:
                request_evidence(evidence_provider(source), requests[0])
            with pytest.raises(CodedError) as batch:
                readouts.run(source, metrics=["conversion"])
            assert (batch.value.code, batch.value.context) == (
                single.value.code,
                single.value.context,
            )
            assert not any(
                name.startswith("component_values_") for name in connection.list_tables()
            )
            return
        actual = tuple(
            cast(Any, request_evidence(evidence_provider(source), request)) for request in requests
        )
        assert actual[0].control_group == "baseline"
        assert actual[-1].control_group == "candidate"
        baseline = next(row for row in actual[2].rows if row["group_id"] == "baseline")
        assert baseline["n"] == 16
        assert baseline["ref_y"] == pytest.approx(4.5, rel=1e-10, abs=1e-10)
        assert baseline["ref_den"] == pytest.approx(
            13 / 16 if missing == "zero" else 0.75, rel=1e-10, abs=1e-10
        )
        assert baseline["cy2"] == pytest.approx(236, rel=1e-10, abs=1e-10)
        assert baseline["cden2"] == pytest.approx(
            39 / 16 if missing == "zero" else 3, rel=1e-10, abs=1e-10
        )
        assert baseline["cyden"] == pytest.approx(
            13.5 if missing == "zero" else 18, rel=1e-10, abs=1e-10
        )
        assert actual[3].rows == ()
        _assert_total_readout_uncertainty(
            source,
            (
                ratio,
                next(metric for metric in source.context.metrics if metric.name == revenue.name),
            ),
        )
        _assert_display_parity(source, metrics[:4])
        for grain, completed in (("daily", False), ("asof", False), ("asof", True)):
            day_source = source.day_source(metrics=metrics)
            _assert_temporal_parity(day_source, metrics, grain, completed)


def test_running_source_without_observations_explains_missing_cutoff():
    with _native_groups(metric_names=("mean",)) as (connection, experiment, resolver):
        connection.raw_sql("DELETE FROM events")
        rows = _capture(resolver, "mean")
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

    from increment.query._native_dashboard import _dashboard_quantiles

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

    with _native_groups() as (_, experiment, resolver):
        rows = _capture(resolver)
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
    with _native_groups(extra_exclusions=True) as (_, experiment, resolver):
        row = _capture(resolver, "retention")["retention", "baseline"]
    assert (row.assigned_units, row.eligible_units) == (18, 16)
    assert (row.excluded_not_mature, row.excluded_no_observed_day, row.excluded_other) == (1, 1, 0)
    assert row.retained_units == 8
    assert row.observed_value == 0.5


def test_fully_immature_arm_keeps_accounting_and_null_value():
    with _native_groups(late_treatment=True) as (_, experiment, resolver):
        row = _capture(resolver, "retention")["retention", "candidate"]
    assert row.assigned_units == row.excluded_not_mature == 16
    assert row.eligible_units == row.retained_units == row.event_count == 0
    assert row.excluded_no_observed_day == row.excluded_other == 0
    assert row.observed_value is None
    assert "observed_value" in row.unavailable


def test_ratio_events_follow_joint_denominator_eligibility():
    with _native_groups(denominator_average=True) as (_, experiment, resolver):
        row = _capture(resolver, "ratio")["ratio", "candidate"]
    assert (row.assigned_units, row.eligible_units, row.excluded_other) == (16, 12, 4)
    assert row.event_count == 12
    assert row.numerator == 96
    assert row.denominator == 36
    assert row.observed_value == pytest.approx(96 / 36)


def test_raw_group_values_precede_winsorization():
    with _native_groups(winsor=True) as (_, experiment, resolver):
        row = _capture(resolver, "mean")["mean", "candidate"]
    assert row.observed_value == 6.5
    assert row.sum_value == 104


def test_finite_group_mean_survives_unrepresentable_display_total():
    with _native_groups(large=True) as (_, experiment, resolver):
        row = _capture(resolver, "mean")["mean", "baseline"]
    assert row.observed_value == pytest.approx(1e308)
    assert row.sum_value is None
    assert "sum_value" in row.unavailable


@pytest.mark.parametrize("law", ["bernoulli", "scalar_mean"])
def test_retained_checkpoint_values_survive_changed_live_outcomes(law, tmp_path):
    import pyarrow as pa

    from increment import from_definitions
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from tests.test_sequential_public_sources import _native_fixture

    connection, definitions, (_, fixture_resolver) = _native_fixture(law)
    fixture_resolver.close()
    try:
        path = tmp_path / "definitions.yaml"
        path.write_text(definitions.model_dump_json())
        experiment, resolver = from_definitions(path, connection, experiment_name="experiment")
        connection.raw_sql(
            "INSERT INTO events (unit_id, event, value, ts) VALUES "
            "('freshness-only', 'outcome_event', 0, TIMESTAMPTZ '2025-01-20 00:00:00+00')"
        )
        capture_sequential(experiment, resolver, finalized=True, as_of=date(2025, 1, 16))
        config = DashboardConfig(expected_allocation={"control": 1, "treatment": 1})
        snapshot = prepare_dashboard(experiment, resolver, config=config)
        checkpoints = {
            estimate.metric: estimate.require_sequential_result().checkpoint
            for estimate in snapshot.estimates
        }
        events = connection.table("events").to_pyarrow()
        ghost = [row for row in events.to_pylist() if row["unit_id"] == "freshness-only"]
        connection.create_table(
            "events", pa.Table.from_pylist(ghost, schema=events.schema), overwrite=True
        )
        binding = cast(Any, resolver)._binding
        rows = dashboard_group_data(binding, metrics=binding.metrics, checkpoints=checkpoints)
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
        cast(Any, resolver).close()
        connection.disconnect()


def test_frozen_secondary_keeps_its_earlier_displayed_prefix():
    import pyarrow as pa

    from increment.dashboard import DashboardConfig, prepare_dashboard

    with _native_groups(sequential=True) as (connection, experiment, resolver):
        early = capture_sequential(
            experiment, resolver, finalized=True, as_of=date(2025, 1, 10), freeze=["mean"]
        )
        config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
        before = prepare_dashboard(experiment, resolver, config=config)
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
        capture_sequential(
            experiment, resolver, finalized=True, as_of=date(2025, 1, 20), previous=early
        )
        after = prepare_dashboard(experiment, resolver, config=config)
        old = {(row.metric, row.group_id): row for row in before.group_data}
        for row in after.group_data:
            if row.metric == "mean":
                assert row == old[row.metric, row.group_id]
                assert row.eligible_units == 16
            else:
                assert row.eligible_units == 17
                assert row.prefix_id != old[row.metric, row.group_id].prefix_id


def test_zero_ratio_denominator_preserves_other_available_evidence():
    with _native_groups() as (connection, experiment, resolver):
        connection.raw_sql("UPDATE events SET value = 0 WHERE event = 'denominator'")
        row = _capture(resolver, "ratio")["ratio", "baseline"]
    assert row.eligible_units == 16
    assert row.numerator == pytest.approx(72)
    assert row.denominator == 0
    assert row.observed_value is None
    assert "observed_value" in row.unavailable


def test_group_capture_materializes_only_bounded_aggregate_rows(monkeypatch):
    with _native_groups() as (connection, experiment, resolver):
        original = connection.to_pyarrow
        row_counts = []

        def capture(*args, **kwargs):
            table = original(*args, **kwargs)
            row_counts.append(table.num_rows)
            assert "unit_id" not in table.column_names
            return table

        monkeypatch.setattr(connection, "to_pyarrow", capture)
        rows = _capture(resolver)
    assert set(rows) == {
        (metric, arm)
        for metric in ("conversion", "mean", "retention", "ratio", "quantile")
        for arm in ("baseline", "candidate")
    }
    assert max(row_counts) <= len(rows)


@pytest.mark.parametrize("kind", ["swapped_metric", "foreign_prefix"])
def test_group_capture_rejects_unbound_checkpoints_before_data_access(kind, monkeypatch):
    from increment.dashboard import DashboardConfig, prepare_dashboard
    from increment.errors import CapabilityError

    config = DashboardConfig(expected_allocation={"baseline": 1, "candidate": 1})
    with _native_groups(sequential=True) as (connection, experiment, resolver):
        capture_sequential(experiment, resolver, finalized=True, as_of=date(2025, 1, 16))
        snapshot = prepare_dashboard(experiment, resolver, config=config)
        checkpoints = {
            row.metric: row.require_sequential_result().checkpoint for row in snapshot.estimates
        }
        if kind == "swapped_metric":
            checkpoints = {"mean": checkpoints["conversion"], "conversion": checkpoints["mean"]}
            code = "evidence.operation_unavailable"
        else:
            with _native_groups(sequential=True, extra_exclusions=True) as (
                _,
                other,
                other_resolver,
            ):
                capture_sequential(other, other_resolver, finalized=True, as_of=date(2025, 1, 16))
                foreign = prepare_dashboard(other, other_resolver, config=config)
                checkpoints = {
                    row.metric: row.require_sequential_result().checkpoint
                    for row in foreign.estimates
                }
            code = "sequential.continuation.rewrite"

        def unexpected_read(*args, **kwargs):
            pytest.fail("checkpoint validation must precede warehouse reads")

        binding = resolver._binding
        monkeypatch.setattr(connection, "to_pyarrow", unexpected_read)
        with pytest.raises(CapabilityError) as caught:
            dashboard_group_data(binding, metrics=binding.metrics, checkpoints=checkpoints)
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

    with _native_groups(metric_names=("mean",)) as (connection, experiment, resolver):
        binding = resolver._binding
        metric = binding.metrics[0]
        requested = {
            "unknown": [metric.model_copy(update={"name": "undeclared"})],
            "changed": [metric.model_copy(update={"window_days": 1})],
            "duplicate": [metric, metric],
        }[kind]

        def unexpected_read(*args, **kwargs):
            pytest.fail("metric validation must precede warehouse reads")

        monkeypatch.setattr(connection, "to_pyarrow", unexpected_read)
        with pytest.raises(InvalidRequestError) as caught:
            dashboard_group_data(binding, metrics=requested)
        assert caught.value.code == code
