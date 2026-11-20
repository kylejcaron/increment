"""Candidate subprocess for the Terrine side of the shop comparison."""

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import ibis
import terrine as tr
from terrine.data import BoundCoverage

from increment_metrics import Catalog, FormulaMetric, load

_COMPONENTS = {
    "captured_order_count": "order_count",
    "captured_refunds": "refunds",
    "captured_web_revenue": "web_revenue",
}
_METRICS = (
    "net_revenue",
    "average_order_value",
    "web_revenue_per_order",
    "revenue_per_customer",
    "purchase_conversion",
    *_COMPONENTS,
)


def run(fixture, database, scenario):
    expected = json.loads((fixture / "expected.json").read_text())
    scope = expected["scope"]
    start = datetime.fromisoformat(scope["start"]).replace(tzinfo=UTC)
    end_text = (
        expected["scenarios"]["incomplete_coverage"]["end"]
        if scenario == "incomplete"
        else scope["end"]
    )
    end = datetime.fromisoformat(end_text).replace(tzinfo=UTC)
    catalog = load(fixture / "metrics.yaml")
    catalog = Catalog(
        version=catalog.version,
        sources=catalog.sources,
        populations=catalog.populations,
        filters=catalog.filters,
        measures=catalog.measures,
        metrics={
            **catalog.metrics,
            **{
                name: FormulaMetric.model_validate({"formula": formula})
                for name, formula in _COMPONENTS.items()
            },
        },
    )
    connection = ibis.duckdb.connect(str(database), read_only=True)
    try:
        model = tr.Model.from_metric_definitions(
            catalog,
            con=connection,
            units={"customer": tr.UnitSource("customers", "customer_id")},
            readiness={"orders": BoundCoverage(start, end)},
            metrics=_METRICS,
        )
        design = tr.Design(
            "customer",
            tr.FixedDate(start),
            outcomes={name: tr.agg(name, window=("0s", "1d")) for name in _METRICS},
        )
        columns = [
            column for name in _METRICS for column in (name, f"{name}__complete", f"{name}__reason")
        ]
        table = design.build(model).to_pyarrow().select(["unit", *columns])
        return {row.pop("unit"): row for row in table.to_pylist()}
    finally:
        connection.disconnect()


def _json_number(value):
    if isinstance(value, Decimal):
        return float(value)
    raise TypeError(f"Unexpected result type: {type(value).__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("scenario", choices=("matched", "anchor", "incomplete"))
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, args.database, args.scenario), default=_json_number))
