from pathlib import Path
from typing import cast

import pytest

from metric_definitions import Binary, CatalogError, FormulaMetric, Literal, Reference, load

SHOP = Path(__file__).resolve().parents[1] / "examples" / "shop"
pytestmark = pytest.mark.integration


def test_population_and_order_grain_are_distinct():
    catalog = load(SHOP / "metrics.yaml").select(["revenue_per_customer"])
    orders = catalog.sources["orders"]
    roster = catalog.populations["eligible_customers"]
    assert orders.row_key == "order_id"
    assert orders.entities["customer"] == "customer_id"
    assert catalog.sources[roster.source].table == "customers"
    assert roster.entity == "customer"
    assert "buyers" not in catalog.sources
    assert "paying_customers" not in catalog.populations
    assert catalog.sources[orders.joins["channel_details"].source].row_key == "channel"


def test_formula_dependencies_and_precedence(tmp_path):
    path = tmp_path / "shop.yaml"
    text = (SHOP / "metrics.yaml").read_text()
    path.write_text(
        text + "  weighted: {formula: '2 * (revenue - refunds) + scaled_net_revenue'}\n"
    )
    catalog = load(path).select(["weighted"])
    metric = catalog.metrics["weighted"]
    assert isinstance(metric, FormulaMetric)
    assert metric.formula == Binary(
        "+",
        Binary("*", Literal(2), Binary("-", Reference("revenue"), Reference("refunds"))),
        Reference("scaled_net_revenue"),
    )
    assert set(catalog.metrics) == {"weighted", "scaled_net_revenue", "net_revenue"}
    assert set(catalog.measures) == {"revenue", "refunds"}
    assert catalog.populations == {}


def test_canonical_payload_ignores_formula_spelling(tmp_path):
    path = tmp_path / "shop.yaml"
    text = (SHOP / "metrics.yaml").read_text()
    before = load(SHOP / "metrics.yaml").select(["average_order_value"])
    path.write_text(
        text.replace(
            "formula: revenue / order_count", "formula: '(revenue)/order_count'", 1
        ).replace("2 * net_revenue", "3 * net_revenue")
    )
    after = load(path).select(["average_order_value"])
    assert after.model_dump(mode="json") == before.model_dump(mode="json")


def test_selected_filter_does_not_change_other_operands():
    selected = load(SHOP / "metrics.yaml").select(["web_revenue_per_order"])
    assert selected.measures["web_revenue"].filter == "web_only"
    assert selected.measures["order_count"].filter is None
    assert set(selected.filters) == {"web_only"}


def test_catalog_does_not_allow_in_place_entity_rebinding():
    entities = load(SHOP / "metrics.yaml").sources["orders"].entities
    with pytest.raises(TypeError):
        cast(dict[str, str], entities)["customer"] = "order_id"


@pytest.mark.parametrize(
    ("line_fragment", "replacement", "code"),
    [
        ("revenue / order_count", "missing / order_count", "unknown_reference"),
        ("revenue / order_count", "average_order_value", "formula_cycle"),
        ("revenue / order_count", "net_revenue ** 2", "invalid_formula"),
        ("revenue / order_count", "sum(revenue)", "invalid_formula"),
        ("revenue / order_count", "1e999", "invalid_formula"),
        ("revenue / order_count", "order_count if revenue else refunds", "invalid_formula"),
        (
            "formula: revenue / order_count",
            "formula: revenue, formula: order_count",
            "duplicate_key",
        ),
        ("  average_order_value:", "  net_revenue:", "duplicate_key"),
        ("table: orders", "table: 'orders; SELECT 1'", "invalid_definition"),
        ("value: web", "value: 3", "invalid_filter_value"),
        ("value: web", "value: null", "invalid_definition"),
        ("event_count: order_count", "event_count: revenue", "invalid_conversion"),
    ],
)
def test_invalid_declaration_names_its_source_line(tmp_path, line_fragment, replacement, code):
    path = tmp_path / "invalid.yaml"
    original = (SHOP / "metrics.yaml").read_text()
    text = original.replace(line_fragment, replacement, 1)
    lines = [i for i, line in enumerate(text.splitlines(), 1) if replacement in line]
    path.write_text(text)
    with pytest.raises(CatalogError) as exc:
        load(path)
    assert exc.value.code == code
    assert exc.value.path == path
    assert exc.value.line in lines
    assert exc.value.column >= 1
