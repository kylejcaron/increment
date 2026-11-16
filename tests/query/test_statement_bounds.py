from __future__ import annotations

from typing import Any, cast

import pytest

from tests.query.statement_census import JOURNEYS, count_statements

pytestmark = pytest.mark.integration

# Observed counts; production changes may only tighten them.
BOUND = {
    "native_scalar": {
        "total": 12,
        "select": 6,
        "create_temp": 1,
        "drop": 1,
        "schema": 3,
        "other": 1,
    },
    "native_breakout": {
        "total": 11,
        "select": 5,
        "create_temp": 1,
        "drop": 1,
        "schema": 3,
        "other": 1,
    },
    "artifact_cuped_breakout": {
        "total": 1259,
        "select": 567,
        "create_temp": 139,
        "drop": 128,
        "schema": 286,
        "other": 139,
    },
    "dashboard": {
        "total": 79,
        "select": 73,
        "create_temp": 1,
        "drop": 1,
        "schema": 3,
        "other": 1,
    },
}


def test_census_counts_each_execution_path_exactly():
    import ibis
    import pyarrow as pa

    backend = ibis.duckdb.connect()
    backend.create_table("census_source", obj=pa.table({"x": [1]}))
    try:

        def observed(kind, action, *, total=1):
            with count_statements(backend) as census:
                result = action()
            assert census.total == total, census.statements
            assert census.by_kind == {name: total if name == kind else 0 for name in census.by_kind}
            return result

        assert observed("select", lambda: backend.raw_sql("SELECT 1")).fetchone() == (1,)
        assert observed(
            "select", lambda: backend.raw_sql("SELECT 2").execute("SELECT 3"), total=2
        ).fetchone() == (3,)
        assert observed("select", lambda: backend.execute(ibis.literal(4))) == 4
        assert observed("select", lambda: backend.to_pyarrow(ibis.literal(5))).as_py() == 5
        assert observed("schema", lambda: backend.get_schema("census_source")) == ibis.schema(
            {"x": "int64"}
        )
        assert list(observed("schema", lambda: backend.sql("SELECT 6 AS x")).columns) == ["x"]
        observed(
            "create_temp",
            lambda: backend.con.execute("CREATE TEMP TABLE census_created (x INTEGER)"),
        )
        observed("other", lambda: backend.con.execute("INSERT INTO census_created VALUES (7)"))
        assert observed(
            "select", lambda: backend.con.sql("SELECT x FROM census_created")
        ).fetchall() == [(7,)]
        observed("drop", lambda: backend.con.execute("DROP TABLE census_created"))
    finally:
        backend.disconnect()


@pytest.mark.parametrize(
    "journey",
    [
        "native_scalar",
        "native_breakout",
        pytest.param(
            "artifact_cuped_breakout",
            marks=pytest.mark.slow(
                reason="artifact publication, adoption and breakout read take about 11 s locally"
            ),
        ),
        pytest.param(
            "dashboard",
            marks=pytest.mark.slow(reason="dashboard preparation takes about 10 s locally"),
        ),
    ],
    ids=tuple(JOURNEYS),
)
def test_statement_counts_stay_within_bounds(journey, seeded_con):
    census = JOURNEYS[journey](seeded_con)
    bounds = BOUND[journey]
    assert census.total <= bounds["total"]
    assert all(census.by_kind[kind] <= bounds[kind] for kind in census.by_kind)


@pytest.mark.slow
@pytest.mark.parametrize(
    "size,occurrence_only,collision",
    [
        (5, False, False),
        (10, False, False),
        (20, False, False),
        (30, False, False),
        (5, True, False),
        (5, False, True),
    ],
)
def test_compatible_metrics_share_executed_reductions(size, occurrence_only, collision):
    from tests.test_dashboard_group_source import _native_groups

    with _native_groups(
        metric_names=("conversion", "mean"),
        occurrence_labels=occurrence_only,
        domain="value_collision" if collision else None,
    ) as (backend, _, resolver):
        source = resolver._binding.source
        base = [
            metric.model_copy(update={"aggregation": "count"})
            if occurrence_only and metric.type == "mean"
            else metric
            for metric in resolver._binding.metrics
        ]
        metrics = [
            base[index % len(base)].model_copy(update={"name": f"metric_{index}"})
            for index in range(size)
        ]
        source = source.exploratory_source(metrics=metrics)
        with count_statements(backend, explain_inserts=True) as census:
            rows = source._moments_for_metrics(
                selected=metrics, include_covariate=False
            ).to_pylist()
        assert {row["metric"] for row in rows} == {metric.name for metric in metrics}
        assert census.by_kind["select"] <= 6
        assert census.by_kind["create_temp"] <= 1
        assert census.by_kind["other"] <= 1
        assert not any(name.startswith("component_values_") for name in backend.list_tables())
        assert len(census.table_scan_counts) == 1
        assert census.table_scan_counts[0] <= 4
        from increment.evidence import DayAxisRequest, evidence_provider, request_evidence

        for grain in ("daily", "asof"):
            with count_statements(backend, explain_inserts=True) as census:
                acquired = request_evidence(
                    evidence_provider(source.day_source(metrics=metrics)),
                    DayAxisRequest(
                        caller="run_daily" if grain == "daily" else "run_asof",
                        grain=grain,
                        metrics=tuple(metrics),
                        dimension=None,
                    ),
                )
                rows = [row for part in cast(Any, acquired).slices for row in part.rows]
                answers = [
                    [row for row in rows if row["metric"] == metric.name] for metric in metrics
                ]
            assert all(
                {row["metric"] for row in answer} == {metric.name}
                for metric, answer in zip(metrics, answers, strict=True)
            )
            assert census.by_kind["select"] <= 5
            assert census.by_kind["create_temp"] <= 1
            assert census.by_kind["other"] <= 1
            assert not any(name.startswith("component_values_") for name in backend.list_tables())
            assert len(census.table_scan_counts) == 1
            assert census.table_scan_counts[0] <= 4
