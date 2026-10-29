"""Exposure-building correctness on `DefinitionsMomentSource`:

- yxca: a fact-based exposure's declared `filters` must narrow enrollment.
- 43m8: a present-but-NULL `experiment_id` column must coalesce to the
  experiment name before filtering, matching the missing-column contract.
- 7319: a query-based (sql) exposure missing the canonical unit_id/ts/
  group_id columns must refuse cleanly, not crash deep inside
  `first_exposures`.
- jkvr (native-source side): a `RatioMetric`'s whole-site windowed sum
  must coalesce an empty window to 0.0, not surface SQL NULL and crash
  `sitewide_evidence`.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, cast

import ibis
import pytest

from increment import Analysis
from increment.errors import CapabilityError
from tests.analysis_factory import _native_source

pytestmark = pytest.mark.integration


def _write_defs(tmp_path, yaml: str):
    defs_path = tmp_path / "defs.yaml"
    defs_path.write_text(yaml)
    return defs_path


def _enrolled_by_arm(analysis) -> dict[str, int]:
    """Per-arm enrolled unit counts from the public allocation history."""
    history = analysis.allocation_history()
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
    analysis = Analysis.from_definitions("exp", defs_path, con)
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
    analysis = Analysis.from_definitions("exp", defs_path, con)
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
    analysis = Analysis.from_definitions("exp", defs_path, con)
    with pytest.raises(CapabilityError) as exc_info:
        analysis.allocation_history()
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
    analysis = Analysis.from_definitions("exp", _write_defs(tmp_path, yaml), con)
    try:
        captured = analysis.run()
        live = analysis.run(coherence="live")
        assert cast(Any, captured[0]).require_lift().value == pytest.approx(1.6)
        assert cast(Any, live[0]).require_lift().value == pytest.approx(1.6)
        assert captured.metadata is not None and captured.metadata.coherence == "captured"
        assert live.metadata is not None and live.metadata.coherence == "live"
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.integration
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
    analysis = Analysis.from_definitions("exp", defs_path, con)
    metric = analysis.metrics[0]
    evidence = _native_source(analysis).sitewide_evidence(metric, include_ratio=True)
    assert evidence.site_total == pytest.approx(3.0)
    assert evidence.site_total_denominator == pytest.approx(0.0)


def test_sql_exposure_with_only_the_documented_columns_enrols(tmp_path):
    """The guide calls `experiment_id` optional, so a query returning exactly
    `unit_id`, `ts` and `group_id` must enrol rather than crash: downstream
    reductions read `experiment_id` unconditionally, so it is synthesized from
    the experiment name when the query omits it.
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
  - name: sql_enrolled
    sql: SELECT unit_id, ts, group_id FROM enrollments
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
    plan: {secondaries: [revenue]}
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
    analysis = Analysis.from_definitions("exp", defs_path, con)

    assert _enrolled_by_arm(analysis) == {"control": 1, "treatment": 1}


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
      sql: SELECT * FROM events
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
    analysis = Analysis.from_definitions("exp", _write_defs(tmp_path, yaml), con)
    try:
        request = AssignmentRequest(experiment="exp", version=3, period="global")
        provider = evidence_provider(_native_source(analysis))
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
        assert evidence.arm_counts == ((3, "global", "control", 1), (3, "global", "treatment", 1))
    finally:
        analysis.close()
        con.disconnect()
