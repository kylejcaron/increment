from __future__ import annotations

import ibis
import pytest
import sqlglot
from sqlglot import exp

from increment import from_definitions
from increment.query import summary_sql


@pytest.mark.integration
def test_native_summary_exposes_outcome_and_moment_stages(con, example_definitions):
    """An analyst can locate outcomes, reference windows and moment reduction."""
    experiment, reader = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2", store="none"
    )
    try:
        sql = summary_sql(experiment, reader)["purchase_rate"]
        tree = sqlglot.parse_one(sql, read="duckdb")
        stages = {cte.alias: cte.this for cte in tree.ctes}
        assert "inc_group_references" in stages
        assert stages["inc_group_references"].find(exp.Window) is not None
        assert "inc_group_moments" in stages
        assert stages["inc_group_moments"].args.get("group") is not None
        assert "inc_unit_outcomes" in stages or "inc_analysis_rows" in stages
        rows = con.sql(sql).to_pyarrow().to_pylist()
        assert {row["group_id"]: row["n"] for row in rows} == {
            "control": 2,
            "treatment": 2,
        }
    finally:
        reader.close()


@pytest.mark.integration
def test_staged_binary_moments_keep_exact_counts_and_null_families(con):
    from increment.query._summary_sql import SummarySqlStages
    from increment.query.builders import group_summary

    totals = con.sql(
        "SELECT 'study' AS experiment_id, 'converted' AS metric, arm AS group_id, "
        "value::DOUBLE AS y, NULL::DOUBLE AS x, NULL::DOUBLE AS y_den "
        "FROM (VALUES ('treatment', 1), ('treatment', 0), ('control', 0)) AS t(arm, value)"
    )
    stages = SummarySqlStages()
    summary = group_summary(totals, binary_metrics=["converted"], _sql_stages=stages)
    canonical = con.sql(ibis.to_sql(summary, dialect="duckdb")).to_pyarrow()
    rendered = con.sql(stages.to_sql(summary, dialect="duckdb")).to_pyarrow()
    assert rendered.schema == canonical.schema
    rows = {row["group_id"]: row for row in rendered.to_pylist()}
    assert rows == {row["group_id"]: row for row in canonical.to_pylist()}
    assert (rows["treatment"]["n"], rows["treatment"]["successes"]) == (2, 1)
    assert [rows["treatment"][key] for key in ("ref_y", "cy1", "cy2")] == [0.5, 0, 0.5]
    assert [rows["control"][key] for key in ("ref_y", "cy1", "cy2")] == [0, 0, 0]
    assert all(row[key] is None for row in rows.values() for key in ("ref_x", "cx2", "sum_d"))


@pytest.mark.integration
def test_stage_names_do_not_capture_physical_or_nested_source_names(con):
    from increment.query._summary_sql import SummarySqlStages

    con.create_table("inc_group_moments", ibis.memtable({"value": [2, 3]}))
    source = con.sql(
        'WITH "INC_GROUP_MOMENTS_2" AS (SELECT * FROM inc_group_moments) '
        'SELECT value FROM "INC_GROUP_MOMENTS_2" ORDER BY value DESC LIMIT 1'
    )
    stages = SummarySqlStages()
    maximum = source.aggregate(total=source.value.sum())
    stages.mark("group_moments", maximum)
    physical = con.table("inc_group_moments")
    result = maximum.union(physical.select(total=physical.value), distinct=False)
    sql = stages.to_sql(result, dialect="duckdb")
    assert sorted(row["total"] for row in con.sql(sql).to_pyarrow().to_pylist()) == [2, 3, 3]
    names = [cte.alias.casefold() for cte in sqlglot.parse_one(sql, read="duckdb").ctes]
    assert len(names) == len(set(names))


@pytest.mark.integration
def test_stages_preserve_independent_repeated_sources(con):
    from increment.query._summary_sql import SummarySqlStages

    raw = con.sql("SELECT i, random() AS value FROM range(3) AS r(i)")
    con.raw_sql("SELECT setseed(0.42)")
    repeated = raw.view()
    combined = repeated.union(repeated, distinct=False)
    result = combined.aggregate(n=combined.count(), distinct_values=combined.value.nunique())
    stages = SummarySqlStages()
    stages.mark("source", raw)
    stages.mark("repeated", repeated)
    stages.mark("unit_outcomes", combined)
    stages.mark("group_moments", result)
    sql = stages.to_sql(result, dialect="duckdb")
    assert con.sql(sql).to_pyarrow().to_pylist() == [{"n": 6, "distinct_values": 6}]
    assert "inc_group_moments" in {cte.alias for cte in sqlglot.parse_one(sql).ctes}


@pytest.mark.integration
def test_stages_preserve_sharing_after_filter_projection_fusion(con):
    from increment.query._summary_sql import SummarySqlStages

    raw = con.sql("SELECT i, random() AS value FROM range(3) AS r(i)")
    con.raw_sql("SELECT setseed(0.42)")
    first = raw.filter(raw.i >= 0)
    left = first.select("i", "value")
    second = raw.select("i", "value")
    right = second.filter(second.i >= 0)
    combined = left.union(right, distinct=False)
    result = combined.aggregate(n=combined.count(), distinct_values=combined.value.nunique())
    stages = SummarySqlStages()
    stages.mark("left_branch", first)
    stages.mark("right_branch", second)
    stages.mark("unit_outcomes", combined)
    stages.mark("group_moments", result)
    canonical = con.sql(ibis.to_sql(result, dialect="duckdb")).to_pyarrow().to_pylist()
    assert canonical[0]["n"] == 6
    assert con.sql(stages.to_sql(result, dialect="duckdb")).to_pyarrow().to_pylist() == canonical


def test_rendering_does_not_mutate_graph_or_accumulate_names():
    from increment.query._summary_sql import SummarySqlStages

    table = ibis.table({"value": "float64"}, name="observations")
    stages = SummarySqlStages()
    filtered = table.filter(table.value > 0)
    assert stages.mark("unit_outcomes", filtered) is filtered
    result = filtered.aggregate(total=filtered.value.sum())
    original = ibis.to_sql(result, dialect="duckdb")
    stages.mark("group_moments", result)
    first = stages.to_sql(result, dialect="duckdb")
    stages.to_sql(filtered, dialect="postgres")
    assert stages.to_sql(result, dialect="duckdb") == first
    assert ibis.to_sql(result, dialect="duckdb") == original


@pytest.mark.integration
def test_sql_computed_summary_keeps_metric_filter_and_named_reduction(con):
    import polars as pl

    from increment.frame import MetricSpec
    from increment.query.source import SqlPanelSource

    source = SqlPanelSource.from_frame_via_memtable(
        con,
        pl.DataFrame(
            {
                "unit": ["a", "b", "c"],
                "arm": ["C", "T", "T"],
                "revenue": [2.0, 3.0, 5.0],
                "other": [7.0, 11.0, 13.0],
            }
        ),
        unit="unit",
        group="arm",
        metrics=[MetricSpec(name="revenue"), MetricSpec(name="other")],
        study_id="study",
    )
    source.moments(source.context.metrics[0])
    sql = source.summary_sql()["revenue"]
    stages = {cte.alias for cte in sqlglot.parse_one(sql, read="duckdb").ctes}
    assert {"inc_group_references", "inc_group_moments"} <= stages
    rows = con.sql(sql).to_pyarrow().to_pylist()
    assert {row["metric"] for row in rows} == {"revenue"}
    assert {row["group_id"]: (row["n"], row["ref_y"], row["cy2"]) for row in rows} == {
        "C": (1, 2.0, 0.0),
        "T": (2, 4.0, 2.0),
    }


@pytest.mark.integration
def test_stages_keep_correlated_scalar_bound_to_its_outer_row(con):
    from increment.query._summary_sql import SummarySqlStages

    table = con.create_table("scope_values", ibis.memtable({"value": [1, 2, 3]}))
    other = table.view()
    inner = other.filter(other.value > table.value)
    next_value = inner.aggregate(next_value=inner.value.min()).as_scalar()
    result = table.select(value=table.value, next_value=next_value)
    stages = SummarySqlStages()
    stages.mark("correlated_input", inner)
    stages.mark("group_moments", result)
    rows = con.sql(stages.to_sql(result, dialect="duckdb")).to_pyarrow().to_pylist()
    assert sorted(rows, key=lambda row: row["value"]) == [
        {"value": 1, "next_value": 2},
        {"value": 2, "next_value": 3},
        {"value": 3, "next_value": None},
    ]


@pytest.mark.integration
def test_supplied_summary_scans_names_in_its_inferred_backend_dialect(con, monkeypatch):
    import ibis.expr.operations as ops

    from increment.query.source import SqlPanelSource
    from increment.semantics.models import MeanMetric

    monkeypatch.setattr(type(con), "name", "bigquery")
    supplied = ops.SQLQueryResult(
        query="SELECT `metric`, `value` FROM `inc_supplied_summary`",
        schema=ibis.schema({"metric": "string", "value": "float64"}),
        source=con,
    ).to_expr()
    source = SqlPanelSource(
        con,
        supplied,
        metrics=[MeanMetric(name="revenue", entity="user", fact="revenue", aggregation="sum")],
        study_id="study",
    )
    query = source.summary_sql()["revenue"]
    tree = sqlglot.parse_one(query, read="bigquery")
    assert "inc_supplied_summary_2" in {cte.alias for cte in tree.ctes}
    assert any(table.name == "inc_supplied_summary" for table in tree.find_all(exp.Table))
