import json
import os
import subprocess
from pathlib import Path

import pytest

from increment_metrics import Catalog, load
from increment_metrics.malloy import MalloyExportError, export_malloy

CORE = Path(__file__).resolve().parents[1]
SHOP = CORE / "examples" / "shop"
RUNNER = CORE / "conformance" / "malloy"
NODE = os.environ.get("MALLOY_NODE", "node")
pytestmark = pytest.mark.integration


def _run(database, report, catalog, metrics, queries):
    keys = sorted({(src.table, src.row_key) for src in catalog.select(metrics).sources.values()})
    return subprocess.run(
        [NODE, str(RUNNER / "run.mjs")],
        input=json.dumps(
            {
                "database": str(database),
                "model": str(report),
                "queries": queries,
                "keys": keys,
            }
        ),
        capture_output=True,
        text=True,
        check=False,
        cwd=RUNNER,
    )


def _write_database(database, sql):
    subprocess.run(
        [
            NODE,
            "--input-type=module",
            "--eval",
            """
        import {DuckDBConnection} from '@malloydata/db-duckdb';
        import {readFileSync} from 'node:fs';
        const connection = new DuckDBConnection('duckdb', process.argv[1]);
        try {
          await connection.runSQL(readFileSync(0, 'utf8'));
        } finally {
          await connection.close();
        }
        """,
            str(database),
        ],
        input=sql,
        text=True,
        check=True,
        cwd=RUNNER,
    )


def _export(tmp_path, catalog, metrics):
    (tmp_path / "generated.malloy").write_text(export_malloy(catalog, metrics=metrics))


@pytest.fixture
def shop():
    return load(SHOP / "metrics.yaml"), json.loads((SHOP / "expected.json").read_text())


def test_handwritten_report_imports_and_aggregates_shared_metrics(tmp_path, shop):
    catalog, expected = shop
    names = list(catalog.metrics)
    _export(tmp_path, catalog, names)
    report = tmp_path / "report.malloy"
    report.write_text((SHOP / "report.malloy").read_text())
    database = tmp_path / "shop.duckdb"
    _write_database(database, (SHOP / "data.sql").read_text())
    total_fields = {
        "fact_totals": (
            "revenue refunds order_count average_order_value net_revenue "
            "scaled_net_revenue web_revenue_per_order revenue_less_store_refunds"
        ).split(),
        "eligible_totals": ["revenue_per_customer", "purchase_conversion", "order_count"],
        "buyer_totals": ["revenue_per_buyer", "buyer_conversion"],
    }

    queries = [*total_fields, "by_customer", "by_channel", "only_c"]
    result = _run(database, report, catalog, names, queries)

    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    totals = expected["totals"]
    for query, fields in total_fields.items():
        assert rows[query] == [pytest.approx({name: totals[name] for name in fields})]
    assert {row.pop("customer"): row for row in rows["by_customer"]} == expected["per_customer"]
    assert {row["marketing_group"]: row["average_order_value"] for row in rows["by_channel"]} == {
        "digital": 40,
        "physical": 40,
    }
    assert rows["only_c"] == [
        {
            "customer": "C",
            "order_count": 0,
            "average_order_value": expected["scenarios"]["zero_orders"]["average_order_value"],
        }
    ]


@pytest.mark.parametrize("invalid", ["duplicate", "null"])
def test_certified_runner_refuses_invalid_keys_before_query(tmp_path, shop, invalid):
    catalog, expected = shop
    names = ["average_order_value"]
    _export(tmp_path, catalog, names)
    report = tmp_path / "report.malloy"
    report.write_text(
        "import 'generated.malloy'\nquery: totals is orders -> { aggregate: average_order_value }\n"
    )
    database = tmp_path / "corrupt.duckdb"
    setup = (SHOP / "data.sql").read_text()
    if invalid == "duplicate":
        setup += expected["scenarios"]["duplicate_dimension_key"]["sql"]
    else:
        setup += "ALTER TABLE channels ALTER COLUMN channel DROP NOT NULL; INSERT INTO channels VALUES (NULL, 'unknown');"
    _write_database(database, setup)

    result = _run(database, report, catalog, names, ["totals"])

    assert result.returncode != 0
    assert not result.stdout
    assert json.loads(result.stderr) == {
        "code": f"invalid_key.{invalid}",
        "relation": "channels",
        "key": "channel",
    }


@pytest.mark.parametrize(
    ("source", "dimension"),
    [
        ("orders", "not_declared"),
        ("population_eligible_customers", "channel_details.marketing_group"),
    ],
)
def test_native_report_refuses_unknown_or_mis_scoped_grouping(tmp_path, shop, source, dimension):
    catalog, _ = shop
    names = ["average_order_value", "revenue_per_customer"]
    _export(tmp_path, catalog, names)
    report = tmp_path / "report.malloy"
    report.write_text(
        f"import 'generated.malloy'\nquery: grouped is {source} -> {{ group_by: {dimension} }}\n"
    )
    database = tmp_path / "shop.duckdb"
    _write_database(database, (SHOP / "data.sql").read_text())

    result = _run(database, report, catalog, names, ["grouped"])

    assert result.returncode != 0
    assert not result.stdout
    refusal = json.loads(result.stderr)
    assert refusal["code"] == "malloy.query"
    assert [problem["code"] for problem in refusal["problems"]] == ["field-not-found"]


def test_formula_refuses_operands_from_different_populations(shop):
    catalog, _ = shop
    catalog = Catalog.model_validate(
        {
            **catalog.model_dump(),
            "metrics": {
                **catalog.metrics,
                "mixed": {"formula": "revenue_per_customer - revenue_per_buyer"},
            },
        }
    )

    with pytest.raises(MalloyExportError) as error:
        export_malloy(catalog, metrics=["average_order_value", "mixed"])

    assert error.value.code == "malloy.incompatible_grain"
    assert error.value.context["metric"] == "mixed"
    assert error.value.context["target"] == "malloy"


def test_selected_join_cycle_refuses_before_export():
    catalog = Catalog.model_validate(
        {
            "version": 1,
            "sources": {
                "orders": {
                    "table": "orders",
                    "row_key": "id",
                    "joins": {"related": {"source": "orders", "foreign_key": "related"}},
                }
            },
            "measures": {"order_count": {"source": "orders", "count_rows": True}},
            "metrics": {"orders_total": {"formula": "order_count"}},
        }
    )

    with pytest.raises(MalloyExportError) as error:
        export_malloy(catalog, metrics=["orders_total"])

    assert error.value.code == "malloy.join_cycle"
    assert error.value.context["metric"] == "orders_total"


def test_typed_predicates_and_quoted_identifiers_preserve_their_values(tmp_path):
    label = 'O\'Reilly "☕" \\'
    catalog = Catalog.model_validate(
        {
            "version": 1,
            "sources": {
                "where": {
                    "table": "main.select",
                    "row_key": "key",
                    "time": "stamp",
                    "dimensions": {
                        "stamp": {"column": "stamp", "type": "timestamp"},
                        "day": {"column": "day", "type": "date"},
                        "label": {"column": "label", "type": "string"},
                        "ready": {"column": "ready", "type": "boolean"},
                    },
                }
            },
            "filters": {
                "qualified": {
                    "source": "where",
                    "conditions": [
                        {"dimension": "label", "op": "in", "value": [label, "other"]},
                        {"dimension": "ready", "op": "eq", "value": True},
                        {"dimension": "day", "op": "eq", "value": "2026-01-01"},
                        {
                            "dimension": "stamp",
                            "op": "eq",
                            "value": "2026-01-01T02:00:00.000001+01:00",
                        },
                    ],
                }
            },
            "measures": {
                "select": {"source": "where", "sum": "value", "filter": "qualified"},
                "group_by": {"source": "where", "count_rows": True},
            },
            "metrics": {
                "offset": {"formula": "2"},
                "extend": {"formula": "-(select - offset) / group_by"},
            },
        }
    )
    names = ["extend", "offset"]
    _export(tmp_path, catalog, names)
    report = tmp_path / "report.malloy"
    report.write_text(
        "import 'generated.malloy'\n"
        "query: result is `where` -> { aggregate: `extend` }\n"
        "query: const is constants -> { aggregate: `offset` }\n"
    )
    database = tmp_path / "shop.duckdb"
    _write_database(
        database,
        r"""
        CREATE TABLE main."select" (key INTEGER, value INTEGER, stamp TIMESTAMPTZ, day DATE,
                                    label VARCHAR, ready BOOLEAN);
        INSERT INTO main."select" VALUES
            (1, 10, '2026-01-01 01:00:00.000001+00', '2026-01-01', 'O''Reilly "☕" \', true),
            (2, 20, '2026-01-01 01:00:00.000001+00', '2026-01-01', 'O''Reilly "☕" \', false),
            (3, 30, '2026-01-01 01:00:00.000002+00', '2026-01-01', 'O''Reilly "☕" \', true),
            (4, 40, '2026-01-01 01:00:00.000001+00', '2026-01-01', NULL, true);
        """,
    )

    result = _run(database, report, catalog, names, ["result", "const"])

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"result": [{"extend": -2}], "const": [{"offset": 2}]}
