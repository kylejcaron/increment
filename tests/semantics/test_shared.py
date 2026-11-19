"""Shared documents extend, but do not replace, native experiment YAML."""

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import yaml

from increment import AnalysisPlan, from_definitions
from increment.errors import CodedError, DefinitionError
from increment.reporting import MetricTrends
from increment.semantics.loader import load
from increment.semantics.models import ConversionMetric, MeanMetric, RatioMetric
from tests.parity_harness import dataset as ds

if TYPE_CHECKING:
    from ibis.backends.sql import SQLBackend


_PORTABLE = """\
version: 1
sources:
  orders:
    table: orders
    row_key: order_id
    entities: {user: user_id}
    time: event_at
    dimensions: {channel: {column: store_id, type: string}}
  roster:
    table: enrolled
    row_key: user_id
    entities: {user: user_id}
populations:
  eligible: {source: roster, entity: user}
filters:
  web_only:
    source: orders
    conditions: [{dimension: channel, op: eq, value: web}]
measures:
  revenue: {source: orders, sum: revenue}
  web_revenue: {source: orders, sum: revenue, filter: web_only}
  order_count: {source: orders, count_rows: true}
metrics:
  aov: {formula: web_revenue / order_count}
  net_revenue: {formula: revenue - 1}
  mean: {per_unit_mean: {measure: revenue, population: eligible}}
  conversion: {conversion: {event_count: order_count, population: eligible}}
"""


@pytest.fixture
def shared_files(tmp_path: Path):
    folder = tmp_path / "native"
    folder.mkdir()
    manifest = folder / "definitions.yaml"
    portable = folder / "01-shop.yaml"
    portable.write_text(_PORTABLE, encoding="utf-8")
    defs = ds.definitions_dict(plan=AnalysisPlan(primary="aov"))
    defs["fact_sources"][0]["sql"] = "SELECT *, row_number() OVER () AS order_id FROM events"
    defs["experiments"][0]["n_pre_periods"] = 0
    defs["shared_metrics"] = {
        "file": portable.name,
        "use": ["aov"],
        "sources": {"orders": "events"},
        "measures": dict.fromkeys(("revenue", "web_revenue", "order_count"), "purchase"),
        "metrics": {
            name: {"entity": "user", "window_days": 1} for name in ("aov", "mean", "conversion")
        },
    }
    manifest.write_text(yaml.safe_dump(defs), encoding="utf-8")
    return manifest, portable


def test_shared_filters_stay_on_the_selected_operand(shared_files):
    manifest, _ = shared_files
    defs = load(manifest.parent.parent)
    ratio = defs.metric("aov")
    assert isinstance(ratio, RatioMetric)
    assert ratio.numerator.aggregation == "sum"
    (condition,) = ratio.numerator.filters
    assert (condition.property, condition.op, condition.values) == ("store", "equals", ("web",))
    assert ratio.denominator.aggregation == "count"
    assert ratio.denominator.filters == ()
    assert ratio.numerator.window_days == ratio.denominator.window_days == 1
    assert defs.metric("net_revenue") is None


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"use": ["net_revenue"]}, "definition.shared.metric_unsupported"),
        ({"use": ["aov", "aov"]}, "definition.shared.include"),
        ({"measures": {"web_revenue": "session_end"}}, "definition.shared.metric_unsupported"),
        ({"file": "../../outside.yaml"}, "definition.shared.include"),
        ({"file": "definitions.yaml"}, "definition.shared.include"),
    ],
    ids=["general-formula", "duplicate-selection", "wrong-value", "path-escape", "self-include"],
)
def test_shared_definition_refusals(shared_files, change, code):
    manifest, _ = shared_files
    native = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    native["shared_metrics"].update(change)
    manifest.write_text(yaml.safe_dump(native), encoding="utf-8")
    with pytest.raises(DefinitionError) as exc:
        load(manifest)
    assert exc.value.code == code
    if change == {"use": ["net_revenue"]}:
        assert exc.value.context["metric"] == "net_revenue"
        assert exc.value.context["target"] == "Increment experiments"
        assert "descriptive report" in str(exc.value.context["route"])


@pytest.mark.integration
def test_offline_catalog_tracks_only_validated_selected_shared_semantics(shared_files):
    manifest, portable = shared_files
    con = cast("SQLBackend", ds.duckdb_connection())
    try:
        online, reader = from_definitions(manifest.parent, con)
        reader.close()
    finally:
        con.disconnect()
    equivalent = _PORTABLE.replace("web_revenue / order_count", "( web_revenue / order_count )")
    equivalent = equivalent.replace("revenue - 1", "revenue - 2") + "# Edited unrelated formula\n"
    portable.write_text(equivalent, encoding="utf-8")
    restored, _ = from_definitions(manifest.parent)
    assert restored.to_json() == online.to_json()
    portable.write_text(equivalent.replace("value: web", "value: store"), encoding="utf-8")
    with pytest.raises(CodedError) as exc:
        from_definitions(manifest.parent)
    assert exc.value.code == "kernel.definitions.catalog_stale"


def _select_roster_metrics(manifest: Path) -> None:
    native = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    native["shared_metrics"]["use"] = ["mean", "conversion"]
    native["experiments"][0]["plan"] = {"primary": "mean", "secondaries": ["conversion"]}
    manifest.write_text(yaml.safe_dump(native), encoding="utf-8")


@pytest.mark.integration
def test_shared_mean_and_conversion_require_the_assigned_roster(shared_files):
    manifest, _ = shared_files
    _select_roster_metrics(manifest)
    declarations = load(manifest)
    assert isinstance(declarations.metric("mean"), MeanMetric)
    assert isinstance(declarations.metric("conversion"), ConversionMetric)
    con = cast("SQLBackend", ds.duckdb_connection())
    try:
        events = con.table("events")
        assigned = events.filter(events.event == "exposure").select("user_id")
        con.create_table("enrolled", assigned)
        experiment, reader = from_definitions(manifest, con)
        reader.close()
        assert experiment.experiment_id == "exp"
        # Same number of eligible IDs, but not the same people.
        con.create_table(
            "enrolled", assigned.mutate(user_id=assigned.user_id + "-other"), overwrite=True
        )
        with pytest.raises(DefinitionError) as exc:
            from_definitions(manifest, con)
        assert exc.value.code == "definition.shared.metric_unsupported"
        assert exc.value.context["metric"] == "conversion"
    finally:
        con.disconnect()


@pytest.mark.integration
def test_trends_refuse_shared_roster_metric_before_resolving_facts(shared_files):
    import ibis

    manifest, _ = shared_files
    _select_roster_metrics(manifest)
    con = ibis.duckdb.connect()
    try:
        trends = MetricTrends.from_definitions(manifest, con)
        with pytest.raises(DefinitionError) as exc:
            trends.metric("mean", start=dt.date(2025, 1, 10), end=dt.date(2025, 1, 11))
        assert exc.value.code == "definition.shared.metric_unsupported"
        assert exc.value.context["metric"] == "mean"
        assert exc.value.context["target"] == "Increment metric trends"
    finally:
        con.disconnect()
