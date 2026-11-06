"""Exposure-building correctness on `DefinitionsMomentSource`:

- yxca: a fact-based exposure's declared `filters` must narrow enrollment.
- 43m8: a present-but-NULL `experiment_id` column must coalesce to the
  experiment name before filtering, matching the missing-column contract,
  on a fact-based and a SQL exposure; a key that matches no row refuses.
- 7319: a query-based (sql) exposure missing the canonical unit_id/ts/
  group_id columns must refuse cleanly, not crash deep inside
  `first_exposures`.
- jkvr (native-source side): a `RatioMetric`'s whole-site windowed sum
  must coalesce an empty window to 0.0, not surface SQL NULL and crash
  `sitewide_evidence`.
"""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Mapping
from statistics import fmean
from typing import Any, cast

import ibis
import pyarrow as pa
import pytest

from increment import (
    CupedCovariate,
    Method,
    MetricSpec,
    View,
    from_definitions,
    from_frame,
    from_unit_day_artifact,
    run,
)
from increment.artifacts import artifact_context, publish_unit_day_artifact
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimators import estimate_cate
from increment.results import LiftEstimate
from increment.workflows import allocation_history, metrics
from tests.analysis_factory import bound_source, held_source, view_readout
from tests.warning_codes import warning_codes, warning_context

pytestmark = pytest.mark.integration


def _write_defs(tmp_path, yaml: str):
    defs_path = tmp_path / "defs.yaml"
    defs_path.write_text(yaml)
    return defs_path


def _enrolled_by_arm(analysis: tuple[Any, Any]) -> dict[str, int]:
    """Per-arm enrolled unit counts from the public allocation history."""
    history = allocation_history(*analysis)
    return dict(
        zip(history["group_id"].to_pylist(), history["n_cumulative"].to_pylist(), strict=True)
    )


def test_fact_based_exposure_applies_declared_filters(tmp_path):
    """VERIFIED repro (yxca): an exposure with filters=[platform == 'web']
    over a web row and an ios row must enrol only the web unit; before the
    fix, Exposure.filters was dropped entirely and both units enrolled.
    """
    yaml = """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
    properties:
      - name: platform
        column: platform
        dtype: string
exposures:
  - name: enrolled_exposure
    fact: enrolled
    filters:
      - property: platform
        op: equals
        values: [web]
metrics:
  - name: converted
    type: conversion
    entity: user_id
    fact: enrolled
    window_days: 7
experiments:
  - name: exp
    exposure: enrolled_exposure
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {secondaries: [converted]}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "user_id": "u_web",
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "enrolled",
                "experiment_id": "exp",
                "group_id": "control",
                "platform": "web",
            },
            {
                "user_id": "u_ios",
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "enrolled",
                "experiment_id": "exp",
                "group_id": "control",
                "platform": "ios",
            },
        ],
    )
    defs_path = _write_defs(tmp_path, yaml)
    analysis = from_definitions(defs_path, con, experiment_name="exp")
    assert _enrolled_by_arm(analysis) == {"control": 1}


def test_fact_based_exposure_coalesces_null_experiment_id_before_filtering(tmp_path):
    """VERIFIED repro (43m8): a present `experiment_id` column that is
    NULL on every row must still enrol, exactly like the missing-column
    path already does via first_exposures's own coalesce; before the fix
    the early `experiment_id == name` filter dropped the NULL row first.
    """
    yaml = """
fact_sources:
  - name: events
    sql: >
      SELECT user_id, ts, event, group_id, CAST(NULL AS VARCHAR) AS experiment_id
      FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
exposures:
  - name: enrolled_exposure
    fact: enrolled
metrics:
  - name: converted
    type: conversion
    entity: user_id
    fact: enrolled
    window_days: 7
experiments:
  - name: exp
    exposure: enrolled_exposure
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {secondaries: [converted]}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "user_id": "u1",
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "enrolled",
                "group_id": "control",
            },
        ],
    )
    defs_path = _write_defs(tmp_path, yaml)
    analysis = from_definitions(defs_path, con, experiment_name="exp")
    assert _enrolled_by_arm(analysis) == {"control": 1}


def test_sql_exposure_missing_canonical_columns_refuses_cleanly(tmp_path):
    """VERIFIED repro (7319): the documented `(entity_id, first_exposure_ts)`
    query-exposure shape from docs/guides/data-model.md is missing `ts`
    and `group_id`; before the fix this crashed deep inside
    first_exposures with an opaque AttributeError instead of refusing.
    """
    yaml = """
fact_sources:
  - name: purchases
    sql: SELECT * FROM purchase_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: value
exposures:
  - name: enrollment_sql
    sql: SELECT user_id, first_exposure_ts FROM enrollment_q
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: enrollment_sql
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {secondaries: [revenue]}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "purchase_events",
        obj=[
            {"user_id": "u1", "ts": dt.datetime(2025, 1, 2, 9, 0, 0), "value": 10.0},
        ],
    )
    con.create_table(
        "enrollment_q",
        obj=[
            {"user_id": "u1", "first_exposure_ts": dt.datetime(2025, 1, 1, 9, 0, 0)},
        ],
    )
    defs_path = _write_defs(tmp_path, yaml)
    experiment, resolver = from_definitions(defs_path, con, experiment_name="exp")
    with pytest.raises(CapabilityError) as exc_info:
        allocation_history(experiment, resolver)
    assert exc_info.value.code == "evidence.operation_unavailable"


@pytest.mark.integration
def test_captured_exposure_and_dimension_sharing_sql_preserves_coordinates(tmp_path):
    yaml = """
fact_sources:
  - name: outcomes
    sql: SELECT * FROM outcome_events
    timestamp_column: ts
    entities: [unit_id]
    dims: [regions]
    facts:
      - name: purchase
        column: value
dim_sources:
  - name: regions
    sql: SELECT unit_id, ts, group_id, region FROM shared_stream
    entity: unit_id
    properties:
      - {name: region, column: region, dtype: string, as_of: static}
exposures:
  - name: enrollment
    sql: SELECT unit_id, ts, group_id, region FROM shared_stream
metrics:
  - name: revenue
    type: mean
    entity: unit_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: enrollment
    unit: unit_id
    start: 2025-01-01T00:00:00
    end: 2025-01-08T00:00:00
    control_group: control
    plan: {secondaries: [revenue]}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "shared_stream",
        obj=[
            {
                "unit_id": unit,
                "ts": dt.datetime(2025, 1, 1, 9, 0),
                "group_id": group,
                "region": "north" if group == "control" else "south",
            }
            for unit, group in (
                ("c1", "control"),
                ("c2", "control"),
                ("t1", "treatment"),
                ("t2", "treatment"),
            )
        ],
    )
    con.create_table(
        "outcome_events",
        obj=[
            {
                "unit_id": unit,
                "ts": dt.datetime(2025, 1, 2, 9, 0),
                "event": "purchase",
                "value": value,
            }
            for unit, value in (("c1", 9.0), ("c2", 11.0), ("t1", 24.0), ("t2", 28.0))
        ]
        + [
            {
                "unit_id": "c1",
                "ts": dt.datetime(2025, 1, 10, 9, 0),
                "event": "purchase",
                "value": 0.0,
            }
        ],
    )
    experiment, resolver = from_definitions(_write_defs(tmp_path, yaml), con, experiment_name="exp")
    try:
        captured = view_readout(experiment, resolver)
        live = view_readout(experiment, resolver, coherence="live")
        assert captured[0].require_lift().value == pytest.approx(1.6)
        assert live[0].require_lift().value == pytest.approx(1.6)
        assert captured.metadata is not None and captured.metadata.coherence == "captured"
        assert live.metadata is not None and live.metadata.coherence == "live"
    finally:
        cast(Any, resolver).close()
        con.disconnect()


@pytest.mark.integration
@pytest.mark.filterwarnings("ignore:fact 'impressions' has no loaded rows")
def test_sitewide_evidence_ratio_metric_coalesces_empty_window_to_zero(tmp_path):
    """VERIFIED repro pattern (jkvr, native-source side): a RatioMetric's
    denominator fact has zero rows in the experiment window, so the
    windowed sum is SQL NULL unless coalesced; before the fix
    `_windowed_metric_sum` (native_source.py's own duplicate of
    `site_volume`'s windowed-sum closure) had no coalesce and
    `sitewide_evidence`'s `float(row["y_den"])` crashed with a TypeError.
    """
    yaml = """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
      - name: clicks
        column: value
      - name: impressions
        column: value
exposures:
  - name: enrolled_exposure
    fact: enrolled
metrics:
  - name: ctr
    type: ratio
    entity: user_id
    numerator:
      fact: clicks
      aggregation: sum
      window_days: 2
    denominator:
      fact: impressions
      aggregation: sum
      window_days: 2
experiments:
  - name: exp
    exposure: enrolled_exposure
    unit: user_id
    start: 2025-01-01T00:00:00
    end: 2025-01-10T00:00:00
    control_group: control
    plan: {secondaries: [ctr]}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "user_id": "u1",
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "enrolled",
                "experiment_id": "exp",
                "group_id": "control",
                "value": None,
            },
            {
                "user_id": "u1",
                "ts": dt.datetime(2025, 1, 2, 9, 0, 0),
                "event": "clicks",
                "experiment_id": None,
                "group_id": None,
                "value": 3.0,
            },
            # No "impressions" rows at all -- the denominator's windowed
            # sum has nothing to aggregate.
        ],
    )
    defs_path = _write_defs(tmp_path, yaml)
    experiment, resolver = from_definitions(defs_path, con, experiment_name="exp")
    metric = metrics(experiment, resolver)[0]
    evidence = bound_source(resolver).sitewide_evidence(metric, include_ratio=True)
    assert evidence.site_total == pytest.approx(3.0)
    assert evidence.site_total_denominator == pytest.approx(0.0)


@pytest.mark.parametrize(
    "exposure_sql",
    [
        "SELECT unit_id, ts, group_id FROM enrollments",
        "SELECT unit_id, ts, group_id, CAST(NULL AS VARCHAR) AS experiment_id FROM enrollments",
    ],
    ids=["key_omitted", "key_all_null"],
)
def test_sql_exposure_all_null_key_enrolls_everyone(tmp_path, exposure_sql):
    """The guide calls `experiment_id` optional, so a query returning exactly
    `unit_id`, `ts` and `group_id` must enrol rather than crash: downstream
    reductions read `experiment_id` unconditionally, so it is synthesized from
    the experiment name when the query omits it. An all-NULL column carries no
    key either, so it enrols the same units, as on a fact-based exposure.
    """
    yaml = f"""
fact_sources:
  - name: purchases
    sql: SELECT * FROM purchase_events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: purchase
        column: value
exposures:
  - name: sql_enrolled
    sql: {exposure_sql}
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    window_days: 7
experiments:
  - name: exp
    exposure: sql_enrolled
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {{secondaries: [revenue]}}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "enrollments",
        obj=[
            {"unit_id": "u1", "ts": dt.datetime(2025, 1, 1, 9, 0, 0), "group_id": "control"},
            {"unit_id": "u2", "ts": dt.datetime(2025, 1, 1, 9, 0, 0), "group_id": "treatment"},
        ],
    )
    con.create_table(
        "purchase_events",
        obj=[
            {"user_id": "u1", "ts": dt.datetime(2025, 1, 2, 9, 0, 0), "value": 10.0},
            {"user_id": "u2", "ts": dt.datetime(2025, 1, 2, 9, 0, 0), "value": 12.0},
        ],
    )
    defs_path = _write_defs(tmp_path, yaml)
    analysis = from_definitions(defs_path, con, experiment_name="exp")

    assert _enrolled_by_arm(analysis) == {"control": 1, "treatment": 1}


def _feed_defs(
    tmp_path, *, exposure: str = "{name: enrolled, fact: page_view}", allocation: str = ""
):
    return _write_defs(
        tmp_path,
        f"""
fact_sources:
  - {{name: events, sql: SELECT * FROM events, timestamp_column: ts, entities: [user_id],
     facts: [{{name: page_view, column: null}}, {{name: purchase, column: revenue}}]}}
exposures: [{exposure}]
metrics:
  - {{name: converted, type: conversion, entity: user_id, fact: purchase, window_days: 3}}
experiments:
  - {{name: exp, exposure: enrolled, unit: user_id, start: 2025-01-01, end: 2025-01-01,
     n_pre_periods: 3, observation_end: 2025-01-06, control_group: control, {allocation}
     plan: {{primary: converted}}}}
""",
    )


def _feed_events(
    *, key: str = "exp", label: str = "purchase", bought: dt.datetime = dt.datetime(2025, 1, 2)
) -> pa.Table:
    """Two units per arm exposed at 2025-01-01 08:00 carrying *key*, each buying once."""
    units = {"c1": "control", "c2": "control", "t1": "treatment", "t2": "treatment"}
    exposed = [
        (unit, dt.datetime(2025, 1, 1, 8), "page_view", key, arm) for unit, arm in units.items()
    ]
    purchases = [(unit, bought, label, None, None) for unit in units]
    rows = exposed + purchases
    return pa.table(
        {
            "user_id": [row[0] for row in rows],
            "ts": [row[1] for row in rows],
            "event": [row[2] for row in rows],
            "revenue": [None] * len(exposed) + [5.0] * len(purchases),
            "experiment_id": [row[3] for row in rows],
            "group_id": [row[4] for row in rows],
        }
    )


def test_experiment_key_mismatch_refuses_before_estimation(tmp_path):
    """Exposure rows keyed `exp-42` enrol nobody in experiment `exp`: the run refuses
    with the keys the rows carry rather than reading outcomes for an empty roster."""
    con = ibis.duckdb.connect()
    con.create_table("events", _feed_events(key="exp-42"))
    experiment, reader = from_definitions(_feed_defs(tmp_path), con, experiment_name="exp")
    with pytest.raises(InvalidRequestError) as refused:
        run(experiment, reader)
    assert refused.value.code == "source.native.experiment_key_unmatched"
    assert dict(refused.value.context) == {"experiment": "exp", "observed_keys": ("exp-42",)}


@pytest.mark.parametrize(
    ("feed", "expected", "absent"),
    [
        ({"label": "Purchase"}, "source.native.fact_no_rows", "frame.censoring.dropped_units"),
        (
            {"bought": dt.datetime(2024, 12, 30)},
            "frame.censoring.dropped_units",
            "source.native.fact_no_rows",
        ),
    ],
    ids=["label_mismatch", "stale_feed"],
)
def test_fact_with_no_rows_warns(tmp_path, feed, expected, absent):
    """A fact whose label matches no loaded row warns with the labels the source does
    carry and still runs; a feed loaded only before launch is censored as before."""
    con = ibis.duckdb.connect()
    con.create_table("events", _feed_events(**feed))
    experiment, reader = from_definitions(_feed_defs(tmp_path), con, experiment_name="exp")
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        run(experiment, reader)
    codes = warning_codes(record)
    assert expected in codes
    assert absent not in codes
    if expected == "source.native.fact_no_rows":
        context = warning_context(record, expected)
        assert (context["fact"], context["fact_source"]) == ("purchase", "events")
        assert "Purchase" in cast("tuple[str, ...]", context["observed_events"])


def test_empty_warehouse_enrollment_is_none_enrolled(tmp_path):
    """A declared 50/50 allocation whose exposure selects no row has zero enrolled units
    by count, so the run names the empty enrollment, not a missing treatment arm."""
    con = ibis.duckdb.connect()
    con.create_table("events", _feed_events())
    defs = _feed_defs(
        tmp_path,
        exposure='{name: enrolled, sql: "SELECT user_id AS unit_id, ts, group_id '
        'FROM events WHERE FALSE"}',
        allocation="allocation: {control: 0.5, treatment: 0.5},",
    )
    experiment, reader = from_definitions(defs, con, experiment_name="exp")
    with pytest.raises(InvalidRequestError) as refused:
        run(experiment, reader)
    assert refused.value.code == "readout.arms.none_enrolled"
    assert dict(cast("Mapping[str, int]", refused.value.context["counts"])) == {
        "control": 0,
        "treatment": 0,
    }


@pytest.mark.parametrize("aware", ["fact_exposure", "sql_exposure", "covariate", "cate"])
def test_mixed_timestamp_kinds_are_session_independent(tmp_path, aware):
    """Every unit is exposed at 10:00Z, buys at 09:00 (1.0) and 10:30 (its outcome), and has
    a pre-exposure tenure recorded at 09:00 (its covariate) and 10:30 (a later value). One
    relation has a time zone: the exposure, read through a fact or SQL, or the tenure's,
    read as a declared CUPED covariate or as a CATE interaction named only at the call;
    naive timestamps are UTC. So whatever the session time zone, only the 10:30 purchase
    follows exposure, each arm's as-of mean is that arm's 5.0 average outcome on every day,
    and the result is the one computed from each unit's 10:30 purchase and 09:00 tenure.
    July puts Los Angeles at a UTC offset other than its offset at the epoch, which an epoch
    difference taken in session time would add to the exposure."""
    # Twenty units, so a one-interaction CATE fit has ten units per covariate column.
    units = {f"{arm[0]}{i}": arm for arm in ("control", "treatment") for i in range(10)}
    tenure = {unit: 1.0 + (3 * i) % 7 for i, unit in enumerate(units)}
    later_tenure = {unit: 100.0 + i for i, unit in enumerate(units)}
    # Outcomes rise with tenure, more steeply under treatment, alternate 0.25 either side of
    # that line, and average 5.0 in each arm.
    arm_tenure = {
        arm: fmean(tenure[unit] for unit in units if units[unit] == arm)
        for arm in ("control", "treatment")
    }
    outcome = {
        unit: 5.0
        + (1.5 if arm == "treatment" else 0.5) * (tenure[unit] - arm_tenure[arm])
        + 0.25 * (-1) ** i
        for i, (unit, arm) in enumerate(units.items())
    }

    def times(values: list[dt.datetime], *, zoned: bool) -> pa.Array:
        if zoned:
            return pa.array([at.replace(tzinfo=dt.UTC) for at in values], pa.timestamp("us", "UTC"))
        return pa.array(values, pa.timestamp("us"))

    exposed = times([dt.datetime(2025, 7, 1, 10)] * len(units), zoned=aware.endswith("exposure"))
    views = pa.table(
        {
            "user_id": list(units),
            "event_at": exposed,
            "event": ["page_view"] * len(units),
            "group_id": list(units.values()),
        }
    )
    enrollments = pa.table(
        {"unit_id": list(units), "ts": exposed, "group_id": list(units.values())}
    )
    # The unenrolled unit's later purchase shows the feed loaded through the window.
    purchases = [
        (unit, dt.datetime(2025, 7, 1, hour, minute), value)
        for unit in units
        for hour, minute, value in ((9, 0, 1.0), (10, 30, outcome[unit]))
    ]
    purchases.append(("late", dt.datetime(2025, 7, 22), 1.0))
    orders = pa.table(
        {
            "user_id": [unit for unit, _, _ in purchases],
            "event_at": times([at for _, at, _ in purchases], zoned=False),
            "event": ["purchase"] * len(purchases),
            "revenue": [value for _, _, value in purchases],
        }
    )
    recorded = [
        (unit, dt.datetime(2025, 7, 1, hour, minute), values[unit])
        for unit in units
        for hour, minute, values in ((9, 0, tenure), (10, 30, later_tenure))
    ]
    profiles = pa.table(
        {
            "user_id": [unit for unit, _, _ in recorded],
            "observed_at": times(
                [at for _, at, _ in recorded], zoned=not aware.endswith("exposure")
            ),
            "tenure": [value for _, _, value in recorded],
        }
    )
    enrolled = (
        '{name: enrolled, sql: "SELECT unit_id, ts, group_id FROM enrollments"}'
        if aware == "sql_exposure"
        else "{name: enrolled, fact: page_view}"
    )
    primary = (
        "revenue"
        if aware == "cate"
        else "{metric: revenue, decision_method: {name: scalar, variance_reduction: cuped, "
        "covariates: [{name: tenure, kind: continuous, property: tenure, source: profiles}]}}"
    )
    defs_path = _write_defs(
        tmp_path,
        f"""
fact_sources:
  - {{name: views, sql: SELECT * FROM views, timestamp_column: event_at, entities: [user_id],
     facts: [{{name: page_view, column: null}}]}}
  - {{name: orders, sql: SELECT * FROM orders, timestamp_column: event_at, entities: [user_id],
     facts: [{{name: purchase, column: revenue}}]}}
  - {{name: profiles, sql: SELECT * FROM profiles, timestamp_column: observed_at,
     entities: [user_id], facts: [],
     properties: [{{name: tenure, column: tenure, dtype: float, as_of: pre_exposure}}]}}
exposures: [{enrolled}]
metrics:
  - {{name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 7}}
experiments:
  - {{name: exp, exposure: enrolled, unit: user_id, start: 2025-07-01, end: 2025-07-02,
     observation_end: 2025-07-20, control_group: control, plan: {{primary: {primary}}}}}
""",
    )

    def readout(experiment: Any, reader: Any) -> tuple[float, ...]:
        if aware == "cate":
            fit = estimate_cate(experiment, reader, "revenue", interact=["tenure"])
            return fit.ate, fit.se, fit.interactions[0].coef
        estimate = view_readout(experiment, reader)[0]
        assert isinstance(estimate, LiftEstimate)
        return estimate.require_lift().value, estimate.require_lift().log_se

    method = Method.cuped(
        name="scalar", covariates=CupedCovariate(name="tenure", kind="continuous")
    )
    expected = readout(
        *from_frame(
            pa.table(
                {
                    "unit": list(units),
                    "group": list(units.values()),
                    "revenue": list(outcome.values()),
                    "tenure": list(tenure.values()),
                }
            ),
            shape="summary",
            unit="unit",
            group="group",
            control="control",
            metrics=[MetricSpec(name="revenue", decision_method=method)],
        )
    )
    tables = {"views": views, "enrollments": enrollments, "orders": orders, "profiles": profiles}
    means = {}
    results = {}
    for zone in ("UTC", "America/Los_Angeles", "Asia/Tokyo"):
        con = ibis.duckdb.connect()
        con.raw_sql(f"SET TimeZone = '{zone}'")
        for name, table in tables.items():
            con.create_table(name, table)
        experiment, reader = from_definitions(defs_path, con, experiment_name="exp")
        try:
            rows = view_readout(experiment, reader, View(view="asof", grain="asof", values=True))
            results[zone] = readout(experiment, reader)
        finally:
            reader.close()
            con.disconnect()
        means[zone] = {
            (row.group_id, row.ds): None if row.value is None else row.value.value for row in rows
        }

    for zone in ("America/Los_Angeles", "Asia/Tokyo"):
        assert means[zone].keys() == means["UTC"].keys()
        for key, mean in means[zone].items():
            assert mean == pytest.approx(5.0, abs=1e-12)
            assert mean == pytest.approx(means["UTC"][key], abs=1e-12)
    for observed in results.values():
        assert observed == pytest.approx(expected, rel=1e-9)


def _feed_defect_definitions(tmp_path):
    """Three experiments over one definitions set, each carrying one feed defect: views are
    time-zone-aware and orders naive, every unit is exposed at 10:00Z, ``nulls`` units buy
    once with a NULL revenue, ``unloaded`` reads a refund fact with no loaded row, and
    ``mixed`` units buy at naive 09:00 (1.0) and 10:30 (5.0)."""
    arms = {"c1": "control", "c2": "control", "t1": "treatment", "t2": "treatment"}
    names = ("nulls", "unloaded", "mixed")
    units = [(f"{name}-{unit}", name, arm) for name in names for unit, arm in arms.items()]
    exposed = dt.datetime(2025, 7, 1, 10, tzinfo=dt.UTC)
    views = pa.table(
        {
            "user_id": [unit for unit, _, _ in units],
            "event_at": pa.array([exposed] * len(units), pa.timestamp("us", "UTC")),
            "event": ["page_view"] * len(units),
            "experiment_id": [name for _, name, _ in units],
            "group_id": [arm for _, _, arm in units],
        }
    )
    purchases = [(f"nulls-{unit}", dt.datetime(2025, 7, 1, 10, 30), None) for unit in arms]
    purchases += [
        (f"mixed-{unit}", dt.datetime(2025, 7, 1, hour, minute), value)
        for unit in arms
        for hour, minute, value in ((9, 0, 1.0), (10, 30, 5.0))
    ]
    # The unenrolled unit's later purchase shows the feed loaded through the window.
    purchases.append(("late", dt.datetime(2025, 7, 22), 1.0))
    orders = pa.table(
        {
            "user_id": [unit for unit, _, _ in purchases],
            "event_at": pa.array([at for _, at, _ in purchases], pa.timestamp("us")),
            "event": ["purchase"] * len(purchases),
            "revenue": pa.array([value for _, _, value in purchases], pa.float64()),
        }
    )
    experiments = "\n".join(
        f"  - {{name: {name}, exposure: enrolled, unit: user_id, start: 2025-07-01, "
        f"end: 2025-07-02, observation_end: 2025-07-20, control_group: control, "
        f"plan: {{primary: {'refunded' if name == 'unloaded' else 'revenue'}}}}}"
        for name in names
    )
    defs_path = _write_defs(
        tmp_path,
        f"""
fact_sources:
  - {{name: views, sql: SELECT * FROM views, timestamp_column: event_at, entities: [user_id],
     facts: [{{name: page_view, column: null}}]}}
  - {{name: orders, sql: SELECT * FROM orders, timestamp_column: event_at, entities: [user_id],
     facts: [{{name: purchase, column: revenue}}, {{name: refund, column: null}}]}}
exposures: [{{name: enrolled, fact: page_view}}]
metrics:
  - {{name: revenue, type: mean, entity: user_id, fact: purchase, aggregation: sum, window_days: 7}}
  - {{name: refunded, type: conversion, entity: user_id, fact: refund, window_days: 7}}
experiments:
{experiments}
""",
    )
    return defs_path, {"views": views, "orders": orders}


@pytest.mark.parametrize("experiment_name", ["nulls", "unloaded", "mixed"])
def test_warehouse_feed_defects_report_distinct_causes(tmp_path, experiment_name):
    """Each experiment surfaces its own feed defect beside the others: the NULL revenue
    refuses under the default rule, the unloaded refund fact warns and still runs, and the
    mixed timestamp kinds read each arm's as-of mean as 5.0 in a UTC and a Los Angeles
    session alike."""
    defs_path, tables = _feed_defect_definitions(tmp_path)

    def open_reader(zone):
        con = ibis.duckdb.connect()
        con.raw_sql(f"SET TimeZone = '{zone}'")
        for name, table in tables.items():
            con.create_table(name, table)
        return from_definitions(defs_path, con, experiment_name=experiment_name)

    if experiment_name == "nulls":
        with pytest.raises(InvalidRequestError) as refused:
            run(*open_reader("UTC"))
        assert refused.value.code == "source.native.fact_null_values"
        assert dict(refused.value.context) == {
            "fact": "purchase",
            "column": "revenue",
            "null_rows": 4,
        }
    elif experiment_name == "unloaded":
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            run(*open_reader("UTC"))
        context = warning_context(record, "source.native.fact_no_rows")
        assert (context["fact"], context["fact_source"]) == ("refund", "orders")
    else:
        means = {}
        for zone in ("UTC", "America/Los_Angeles"):
            experiment, reader = open_reader(zone)
            try:
                rows = view_readout(
                    experiment, reader, View(view="asof", grain="asof", values=True)
                )
            finally:
                reader.close()
            means[zone] = {(row.group_id, row.ds): row.value.value for row in rows}
        assert means["America/Los_Angeles"].keys() == means["UTC"].keys()
        assert list(means["UTC"].values()) == pytest.approx([5.0] * len(means["UTC"]), abs=1e-12)
        assert means["America/Los_Angeles"] == pytest.approx(means["UTC"], abs=1e-12)


def test_definitions_assignment_ledger_uses_explicit_role_mapping(tmp_path):
    from increment.evidence import (
        AssignmentEvidence,
        AssignmentRequest,
        EvidenceScope,
        evidence_provider,
        request_evidence,
    )

    yaml = """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
      - name: purchase
        column: value
exposures:
  - name: enrolled
    fact: enrolled
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    window_days: 7
experiments:
  - name: exp
    exposure: enrolled
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {secondaries: [revenue]}
    assignment_ledger:
      sql: |
        SELECT * FROM events
        UNION ALL
        SELECT user_id, ts, event, group_id, value, subject, experiment_key, arm_name,
               4 AS allocation_version, 'campaign' AS allocation_period, allocation_time,
               first_exposure_time, first_trigger_time
        FROM events
      roles:
        unit: subject
        experiment: experiment_key
        arm: arm_name
        version: allocation_version
        period: allocation_period
        allocated_at: allocation_time
        exposed_at: first_exposure_time
        triggered_at: first_trigger_time
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "user_id": "u1",
                "ts": dt.datetime(2025, 1, 1, 9),
                "event": "enrolled",
                "group_id": "control",
                "value": 10.0,
                "subject": "u1",
                "experiment_key": "exp",
                "arm_name": "control",
                "allocation_version": 3,
                "allocation_period": "global",
                "allocation_time": dt.datetime(2024, 12, 31, 10),
                "first_exposure_time": dt.datetime(2025, 1, 1, 9),
                "first_trigger_time": None,
            },
            {
                "user_id": "u2",
                "ts": dt.datetime(2025, 1, 1, 9),
                "event": "enrolled",
                "group_id": "treatment",
                "value": 12.0,
                "subject": "u2",
                "experiment_key": "exp",
                "arm_name": "treatment",
                "allocation_version": 3,
                "allocation_period": "global",
                "allocation_time": dt.datetime(2024, 12, 31, 11),
                "first_exposure_time": dt.datetime(2025, 1, 1, 10),
                "first_trigger_time": dt.datetime(2025, 1, 1, 12),
            },
        ],
    )
    analysis_experiment, analysis_resolver = from_definitions(
        _write_defs(tmp_path, yaml), con, experiment_name="exp"
    )
    artifact = None
    try:
        from increment.query.session import WarehouseArtifactStore

        store = WarehouseArtifactStore(con, schema_name="assignment_identity")
        context = artifact_context(analysis_experiment, analysis_resolver)
        reference = publish_unit_day_artifact(analysis_experiment, analysis_resolver, store)
        artifact = from_unit_day_artifact(store, reference, expected_context=context)
        assert {
            (row["group_id"], row["n_cumulative"])
            for row in allocation_history(*artifact).to_pylist()
        } == {("control", 1), ("treatment", 1)}
        artifact_assignment = cast(
            AssignmentEvidence,
            request_evidence(
                evidence_provider(held_source(artifact[1])),
                AssignmentRequest(experiment="exp", version=3, period="global"),
            ),
        )
        assert artifact_assignment.arm_counts == (
            (3, "global", "control", 1),
            (3, "global", "treatment", 1),
        )

        with store.open_snapshot(reference) as snapshot:
            manifest = snapshot.read_manifest(
                reference.manifest, expected_sha256=reference.manifest_sha256
            )
        assignment_extension = next(
            extension for extension in manifest.extensions if extension.kind == "assignment_counts"
        )
        assert assignment_extension.extension_version == 5
        assert assignment_extension.identity_relation is not None
        identity_ref = assignment_extension.identity_relation
        con.raw_sql(
            f'UPDATE "{identity_ref.relation.schema_name}"."{identity_ref.relation.name}" '
            "SET arm = 'mutated' WHERE unit_id = 'u1'"
        )
        with pytest.raises(CapabilityError) as digest_mismatch:
            allocation_history(*artifact)
        assert digest_mismatch.value.code == "artifact.relation.digest_mismatch"
        request = AssignmentRequest(experiment="exp", version=3, period="global")
        provider = evidence_provider(bound_source(analysis_resolver))
        with pytest.raises(CapabilityError) as refused:
            request_evidence(
                provider,
                AssignmentRequest(experiment="other", version=3, period="global"),
            )
        assert refused.value.code == "evidence.assignment.selection_unavailable"
        scoped = provider.scope(
            EvidenceScope(
                metrics=provider.context.metrics,
                population="assigned",
                snapshot=True,
                assignment_requests=(request,),
            )
        )
        assert scoped.capture_status == "captured"
        with scoped.source as pinned:
            evidence = cast(
                AssignmentEvidence,
                request_evidence(evidence_provider(pinned), request),
            )
        assert evidence.population == "assigned"
        assert evidence.versions == (3,)
        assert evidence.periods == ("global",)
    finally:
        if artifact is not None:
            artifact[1].close()
        cast(Any, analysis_resolver).close()
        con.disconnect()


def test_artifact_without_assignment_rows_refuses_allocation_history(tmp_path):
    yaml = """
fact_sources:
  - name: events
    sql: SELECT * FROM events
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: enrolled
        column: null
      - name: purchase
        column: value
exposures:
  - name: enrolled
    fact: enrolled
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    window_days: 7
experiments:
  - name: exp
    exposure: enrolled
    unit: user_id
    start: 2025-01-01T00:00:00
    control_group: control
    plan: {secondaries: [revenue]}
"""
    con = ibis.duckdb.connect()
    con.create_table(
        "events",
        obj=[
            {
                "user_id": "u1",
                "ts": dt.datetime(2025, 1, 1, 9),
                "event": "enrolled",
                "group_id": "control",
                "value": 10.0,
            },
        ],
    )
    experiment, resolver = from_definitions(_write_defs(tmp_path, yaml), con, experiment_name="exp")
    artifact = None
    try:
        from increment.query.session import WarehouseArtifactStore

        store = WarehouseArtifactStore(con, schema_name="assignment_identity_absent")
        context = artifact_context(experiment, resolver)
        reference = publish_unit_day_artifact(experiment, resolver, store)
        artifact = from_unit_day_artifact(store, reference, expected_context=context)
        with pytest.raises(CapabilityError) as refused:
            allocation_history(*artifact)
        assert refused.value.code == "evidence.assignment.identity_unavailable"
    finally:
        if artifact is not None:
            artifact[1].close()
        cast(Any, resolver).close()
        con.disconnect()
