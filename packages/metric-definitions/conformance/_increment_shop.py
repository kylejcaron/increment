"""Candidate subprocess for the Increment side of the shop comparison."""

import argparse
import json
import shutil
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import ibis
import yaml
from increment import from_definitions
from increment.artifacts import (
    RatioMetricMeasure,
    SourceSnapshotEvidence,
    WarehouseArtifactStore,
    publish_unit_day_artifact,
)
from increment.errors import DefinitionError

_METRICS = ("average_order_value", "web_revenue_per_order")


def _manifest(start, end, names):
    return {
        "dialect": "duckdb",
        "fact_sources": [
            {
                "name": "assignments",
                "sql": "SELECT *, 'shop' AS experiment_id, 'assigned' AS event FROM assignments",
                "timestamp_column": "assigned_at",
                "entities": ["customer_id"],
                "facts": [{"name": "assigned", "column": None}],
            },
            {
                "name": "orders",
                "sql": "SELECT *, 'ordered' AS event FROM orders",
                "timestamp_column": "placed_at",
                "entities": ["customer_id"],
                "facts": [
                    {"name": "ordered", "column": "gross_amount"},
                    {"name": "refunded", "column": "refunded_amount"},
                ],
                "properties": [
                    {
                        "name": "channel",
                        "column": "channel",
                        "dtype": "string",
                        "as_of": "event_time",
                    }
                ],
            },
        ],
        "exposures": [{"name": "assignment", "fact": "assigned"}],
        "experiments": [
            {
                "name": "shop",
                "exposure": "assignment",
                "unit": "customer_id",
                "start": start.date().isoformat(),
                "end": start.date().isoformat(),
                "observation_end": end.date().isoformat(),
                "control_group": "control",
                "allocation": {"control": 0.5, "treatment": 0.5},
                "plan": {"primary": names[0], "secondaries": names[1:]},
            }
        ],
        "metric_definitions": {
            "file": "shop.metrics.yaml",
            "use": names,
            "sources": {"orders": "orders"},
            "measures": (
                {"revenue": "ordered", "refunds": "refunded"}
                if names == ["net_revenue"]
                else {"revenue": "ordered", "web_revenue": "ordered", "order_count": "ordered"}
            ),
            "metrics": {
                name: {"entity": "customer", "window_days": 1, "preferred_direction": "increase"}
                for name in names
            },
        },
    }


def _seed(database, fixture, start, scenario):
    with duckdb.connect(str(database)) as raw:
        raw.execute((fixture / "data.sql").read_text())
        if scenario is not None:
            raw.execute(scenario["sql"])
        raw.execute(
            """CREATE TABLE assignments AS
            SELECT customer_id, CASE customer_id WHEN 'B' THEN 'treatment' ELSE 'control' END
                AS group_id, ?::TIMESTAMP AS assigned_at FROM customers""",
            [start.replace(tzinfo=None)],
        )


def _capture(con, experiment, reader, end):
    con.create_database("artifacts")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    ref = publish_unit_day_artifact(experiment, reader, store)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
        exposures = snapshot.verify_relation(manifest.base.exposures, expected_role="exposures")
        stats = snapshot.verify_relation(manifest.base.measure_stats, expected_role="measure_stats")
        units = sorted(exposures.unit_id.execute().tolist())
        cells = defaultdict(lambda: [0.0, 0])
        for row in con.to_pyarrow(stats).to_pylist():
            cell = cells[row["unit_id"], row["measure_key"]]
            cell[0] += row["sum_value"]
            cell[1] += row["n_events"]
    for measure in manifest.measures:
        assert measure.certified_edge is not None and measure.certified_edge >= end.date()
    result = {}
    for bound in manifest.metric_measures:
        assert isinstance(bound, RatioMetricMeasure)
        result[bound.metric_name] = {
            unit: [
                cells[unit, bound.numerator_measure_key][0],
                cells[unit, bound.denominator_measure_key][1],
            ]
            for unit in units
        }
    return result


def run(fixture, database, scenario):
    expected = json.loads((fixture / "expected.json").read_text())
    start = datetime.fromisoformat(expected["scope"]["start"]).replace(tzinfo=UTC)
    end = datetime.fromisoformat(expected["scope"]["end"]).replace(tzinfo=UTC)
    cutoff = datetime.combine(end.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    names = ["net_revenue"] if scenario == "unsupported" else list(_METRICS)
    if scenario != "unsupported":
        _seed(
            database,
            fixture,
            start,
            expected["scenarios"]["at_anchor"] if scenario == "anchor" else None,
        )
    if scenario == "seed":
        return {}
    con = ibis.duckdb.connect(str(database))
    try:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "native.yaml"
            shutil.copyfile(fixture / "metrics.yaml", path.with_name("shop.metrics.yaml"))
            path.write_text(yaml.safe_dump(_manifest(start, end, names), sort_keys=False))
            try:
                experiment, reader = from_definitions(
                    path,
                    con,
                    source_snapshot_evidence=SourceSnapshotEvidence(
                        cutoff, {"orders": cutoff, "assignments": cutoff}
                    ),
                )
            except DefinitionError as exc:
                if scenario != "unsupported":
                    raise
                return {"code": exc.code, "context": dict(exc.context)}
            try:
                return _capture(con, experiment, reader, end)
            finally:
                reader.close()
    finally:
        con.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("database", type=Path)
    parser.add_argument("scenario", choices=("matched", "anchor", "unsupported", "seed"))
    args = parser.parse_args()
    print(json.dumps(run(args.fixture, args.database, args.scenario)))
