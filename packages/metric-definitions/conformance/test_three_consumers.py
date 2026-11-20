"""Exercise all three consumers against the same shop inputs."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from increment_metrics import load
from increment_metrics.malloy import export_malloy

CONFORMANCE = Path(__file__).resolve().parent
CORE = CONFORMANCE.parent
SHOP = CORE / "examples" / "shop"
MALLOY = CONFORMANCE / "malloy"
pytestmark = pytest.mark.integration


def _run(command, *, input=None):
    result = subprocess.run(command, input=input, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(result.stdout)


def _native(consumer, database, scenario):
    python = os.environ[f"{consumer.upper()}_PYTHON"]
    return _run(
        [
            python,
            "-I",
            str(CONFORMANCE / f"_{consumer}_shop.py"),
            str(SHOP),
            str(database),
            scenario,
        ]
    )


def _malloy(tmp_path, db):
    catalog = load(SHOP / "metrics.yaml")
    names = list(catalog.metrics)
    (tmp_path / "generated.malloy").write_text(export_malloy(catalog, metrics=names))
    report = tmp_path / "report.malloy"
    report.write_bytes((SHOP / "report.malloy").read_bytes())
    request = {
        "database": str(db),
        "model": str(report),
        "queries": ["fact_totals", "eligible_totals", "buyer_totals", "by_customer", "only_c"],
        "keys": sorted({(source.table, source.row_key) for source in catalog.sources.values()}),
    }
    return _run([os.environ["MALLOY_NODE"], str(MALLOY / "run.mjs")], input=json.dumps(request))


def _ratio(parts):
    return sum(numerator for numerator, _ in parts) / sum(count for _, count in parts)


@pytest.fixture(scope="module")
def expected():
    return json.loads((SHOP / "expected.json").read_text())


def test_shared_shop_matches_each_known_answer_before_comparing(tmp_path, expected):
    database = tmp_path / "matched.duckdb"
    incremental = _native("increment", database, "matched")
    temporal = _native("terrine", database, "matched")
    descriptive = _malloy(tmp_path, database)
    per_customer = expected["per_customer"]
    totals = expected["totals"]
    eligible = expected["scope"]["eligible_customers"]
    parts = {unit: [row["revenue"], row["order_count"]] for unit, row in per_customer.items()}

    assert incremental["average_order_value"] == parts
    assert _ratio(incremental["web_revenue_per_order"].values()) == pytest.approx(
        totals["web_revenue_per_order"]
    )
    assert set(temporal) == set(eligible)
    assert {unit: row["net_revenue"] for unit, row in temporal.items()} == {
        unit: row["net_revenue"] for unit, row in per_customer.items()
    }
    assert all(
        row["net_revenue__complete"] and row["net_revenue__reason"] is None
        for row in temporal.values()
    )
    for unit, row in per_customer.items():
        assert temporal[unit]["captured_refunds"] == row["refunds"]
        if row["order_count"]:
            assert temporal[unit]["average_order_value"] == row["revenue"] / row["order_count"]
            assert temporal[unit]["average_order_value__complete"]
            assert temporal[unit]["average_order_value__reason"] is None
    assert temporal["C"]["average_order_value"] is None
    assert temporal["C"]["average_order_value__complete"]
    assert temporal["C"]["average_order_value__reason"] == "zero_denominator"
    assert sum(row["purchase_conversion"] for row in temporal.values()) / len(
        eligible
    ) == pytest.approx(totals["purchase_conversion"])
    assert {
        row["customer"]: {key: value for key, value in row.items() if key != "customer"}
        for row in descriptive["by_customer"]
    } == per_customer
    [fact_totals] = descriptive["fact_totals"]
    [eligible_totals] = descriptive["eligible_totals"]
    [buyer_totals] = descriptive["buyer_totals"]
    assert fact_totals["average_order_value"] == pytest.approx(totals["average_order_value"])
    assert fact_totals["net_revenue"] == pytest.approx(totals["net_revenue"])
    assert fact_totals["web_revenue_per_order"] == pytest.approx(totals["web_revenue_per_order"])
    assert eligible_totals["revenue_per_customer"] == pytest.approx(totals["revenue_per_customer"])
    assert eligible_totals["purchase_conversion"] == pytest.approx(totals["purchase_conversion"])
    assert buyer_totals["revenue_per_buyer"] == pytest.approx(totals["revenue_per_buyer"])
    assert buyer_totals["buyer_conversion"] == pytest.approx(totals["buyer_conversion"])
    assert descriptive["only_c"] == [
        {"customer": "C", "order_count": 0, "average_order_value": None}
    ]

    terrine_parts = {
        unit: [row["revenue_per_customer"], row["captured_order_count"]]
        for unit, row in temporal.items()
    }
    malloy_parts = {
        row["customer"]: [row["revenue"], row["order_count"]] for row in descriptive["by_customer"]
    }
    assert terrine_parts == malloy_parts == incremental["average_order_value"]
    assert _ratio(terrine_parts.values()) == fact_totals["average_order_value"]
    assert _ratio(
        [(row["captured_web_revenue"], row["captured_order_count"]) for row in temporal.values()]
    ) == pytest.approx(fact_totals["web_revenue_per_order"])


def test_noon_anchor_is_intentionally_not_a_matched_boundary(tmp_path, expected):
    database = tmp_path / "anchor.duckdb"
    artifact = _native("increment", database, "anchor")
    temporal = _native("terrine", database, "anchor")
    oracle = expected["scenarios"]["at_anchor"]
    assert artifact["average_order_value"]["A"][1] == oracle["increment_order_count_for_a"]
    assert temporal["A"]["captured_order_count"] == oracle["terrine_order_count_for_a"]


def test_incomplete_orders_retain_the_eligible_rows_with_nulls(tmp_path, expected):
    database = tmp_path / "incomplete.duckdb"
    _native("increment", database, "seed")
    temporal = _native("terrine", database, "incomplete")
    assert set(temporal) == set(expected["scope"]["eligible_customers"])
    for row in temporal.values():
        assert row["net_revenue"] is None
        assert not row["net_revenue__complete"]
        assert row["net_revenue__reason"] == "incomplete_coverage"


def test_increment_refuses_selected_general_formula_without_partial_results(tmp_path):
    rejected = _native("increment", tmp_path / "refused.duckdb", "unsupported")
    assert rejected["code"] == "definition.shared.metric_unsupported"
    assert rejected["context"]["metric"] == "net_revenue"
    assert rejected["context"]["target"] == "Increment experiments"
