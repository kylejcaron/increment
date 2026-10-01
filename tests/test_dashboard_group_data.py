"""Snapshot consistency and consumer-visible group CSV contracts."""

import csv
import datetime as dt
import io
import json
from dataclasses import replace

import pytest

from increment._source_operations import DashboardGroupData
from increment.dashboard._data import (
    DashboardConfig,
    DashboardSnapshot,
    group_data_csv,
    group_data_rows,
)
from increment.errors import InvalidRequestError
from increment.semantics.models import ConversionMetric, MeanMetric


def _item(metric, group, value, *, unavailable=None):
    reasons = {
        "analysis_input_value": "not separately retained",
        "numerator": "not a ratio metric",
        "denominator": "not a ratio metric",
        "prefix_id": "warehouse evidence has no retained checkpoint",
        **(unavailable or {}),
    }
    if metric != "rate":
        reasons["retained_units"] = "not a binary outcome"
    eligible = 4 if value is not None else 0
    if value is None:
        reasons.setdefault("observed_value", "no observed units")
        reasons["sum_value"] = "no observed units"
    return DashboardGroupData(
        metric=metric,
        group_id=group,
        assigned_units=4,
        eligible_units=eligible,
        observed_value=value,
        analysis_input_value=None,
        sum_value=None if value is None else value * eligible,
        event_count=eligible,
        retained_units=None if metric != "rate" else round((value or 0) * eligible),
        numerator=None,
        denominator=None,
        excluded_not_mature=0,
        excluded_no_observed_day=4 - eligible,
        excluded_other=0,
        observation_end=dt.date(2025, 1, 4),
        window_start_days=0,
        window_end_days=3,
        source_kind="pinned_warehouse",
        prefix_id=None,
        unavailable=reasons,
    )


def _snapshot(*items: DashboardGroupData, units=None, treatment="treatment") -> DashboardSnapshot:
    supplied = {(item.metric, item.group_id): item for item in items}
    for metric in ("rate", "latency"):
        for group in ("control", treatment):
            supplied.setdefault((metric, group), _item(metric, group, 0.0))
    return DashboardSnapshot(
        experiment_name="demo",
        binding_fingerprint="binding",
        title="Demo",
        description=None,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=None,
        control_group="control",
        treatment_group=treatment,
        primary_metric="rate",
        metrics=(
            ConversionMetric(name="rate", entity="user", fact="event"),
            MeanMetric(name="latency", entity="user", fact="event", aggregation="sum"),
        ),
        breakouts=(),
        estimates=(),
        readout_rows=(),
        allocation=None,
        allocation_refusal=("dashboard.allocation_not_applicable", "fixture"),
        allocation_history=(),
        allocation_history_refusal=None,
        config=DashboardConfig(
            expected_allocation={"control": 1, treatment: 1}, metric_units=units or {}
        ),
        computed_at=dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
        group_data=tuple(supplied.values()),
    )


def test_group_rows_order_metrics_and_arms_without_changing_values_or_units():
    snapshot = _snapshot(
        _item("latency", "treatment", 2.5),
        _item("rate", "treatment", 0.25),
        _item("rate", "control", 0.5),
        _item("latency", "control", 1.5),
        units={"latency": "ms", "rate": "not-percent"},
    )
    rows = group_data_rows(snapshot)
    assert [(row["metric"], row["group_id"], row["observed_value"]) for row in rows] == [
        ("rate", "control", 0.5),
        ("rate", "treatment", 0.25),
        ("latency", "control", 1.5),
        ("latency", "treatment", 2.5),
    ]
    assert [row["unit"] for row in rows] == ["%", "%", "ms", "ms"]


def test_group_evidence_owns_unavailable_reasons_and_selection_is_validated():
    reason = {"observed_value": "zero eligible"}
    item = _item("rate", "control", None, unavailable=reason)
    snapshot = _snapshot(item)
    reason["observed_value"] = "changed"
    row = group_data_rows(snapshot, metric="rate")[0]
    assert row["unavailable"]["observed_value"] == "zero eligible"
    with pytest.raises(TypeError):
        row["unavailable"]["observed_value"] = "changed"
    with pytest.raises(InvalidRequestError) as raised:
        group_data_rows(snapshot, metric="missing")
    assert raised.value.code == "dashboard.invalid_view"


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "accounting", "infinite", "unexplained_null"]
)
def test_snapshot_refuses_incomplete_or_contradictory_group_evidence(mutation):
    snapshot = _snapshot()
    rows = list(snapshot.group_data)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "accounting":
        rows[0] = replace(rows[0], assigned_units=5)
    elif mutation == "infinite":
        rows[0] = replace(rows[0], observed_value=float("inf"))
    else:
        rows[0] = replace(rows[0], observed_value=None)
    with pytest.raises(InvalidRequestError) as raised:
        replace(snapshot, group_data=tuple(rows))
    assert raised.value.code == "dashboard.invalid_snapshot"


def test_group_csv_preserves_nulls_precision_provenance_and_safe_text():
    snapshot = _snapshot(
        _item("latency", "control", None),
        _item("latency", "=1+1", 0.12345678901234567),
        units={"latency": "@ms"},
        treatment="=1+1",
    )
    rows = list(csv.DictReader(io.StringIO(group_data_csv(snapshot, metric="latency").decode())))
    assert [row["metric"] for row in rows] == ["latency", "latency"]
    assert rows[0]["observed_value"] == ""
    assert json.loads(rows[0]["unavailable"])["observed_value"] == "no observed units"
    assert float(rows[1]["observed_value"]) == 0.12345678901234567
    assert rows[1]["group_id"] == "'=1+1"
    assert rows[1]["unit"] == "'@ms"
    assert rows[1]["experiment"] == "demo"
    assert rows[1]["computed_at"] == snapshot.computed_at.isoformat()
    assert rows[1]["binding_fingerprint"] == "binding"
    assert "unit_id" not in rows[1]
    with pytest.raises(InvalidRequestError) as raised:
        group_data_csv(snapshot, metric="missing")
    assert raised.value.code == "dashboard.invalid_view"


@pytest.mark.parametrize("field", ["event_count", "observation_end", "prefix_id"])
def test_snapshot_refuses_unexplained_nullable_evidence(field):
    item = _item("rate", "control", 0.5)
    reasons = {name: reason for name, reason in item.unavailable.items() if name != field}
    item = replace(item, **{field: None, "unavailable": reasons})
    with pytest.raises(InvalidRequestError) as caught:
        _snapshot(item)
    assert caught.value.code == "dashboard.invalid_snapshot"
