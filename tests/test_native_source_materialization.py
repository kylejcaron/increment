"""Valid warehouse changes invalidate a materialized native panel.

New or corrected exposure and metric rows must be visible, even when the
updated assignments remain valid.
"""

from __future__ import annotations

import datetime as dt
from typing import cast

import ibis
import pytest

from increment import from_definitions
from increment.evidence import (
    CountEvidence,
    CountsRequest,
    MomentRows,
    MomentsRequest,
    evidence_provider,
    request_evidence,
)
from increment.workflows import metrics
from tests.analysis_factory import bound_source


def _defs_and_con(tmp_path):
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
  - name: enrolled_exposure
    fact: enrolled
metrics:
  - name: revenue
    type: mean
    entity: user_id
    fact: purchase
    aggregation: sum
    window_days: 2
experiments:
  - name: exp
    exposure: enrolled_exposure
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
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "enrolled",
                "experiment_id": "exp",
                "group_id": "control",
                "value": None,
            },
            {
                "user_id": "u1",
                "ts": dt.datetime(2025, 1, 2, 9, 0, 0),
                "event": "purchase",
                "experiment_id": None,
                "group_id": None,
                "value": 10.0,
            },
        ],
    )
    defs_path = tmp_path / "defs.yaml"
    defs_path.write_text(yaml)
    return defs_path, con


@pytest.mark.integration
def test_materialized_panel_reflects_a_valid_exposure_and_fact_added_after_materialize(tmp_path):
    """VERIFIED repro (9wj2): materialize with store='always', then insert a
    new valid exposure plus a purchase; the fingerprint invalidation path
    only tracks mixed/unassigned rows, so before the fix the next reduction
    kept serving the pre-insert moments while unit_counts() (never
    materialized) already saw the new unit.
    """
    defs_path, con = _defs_and_con(tmp_path)
    experiment, resolver = from_definitions(defs_path, con, experiment_name="exp", store="always")
    src = bound_source(resolver)
    metric = metrics(experiment, resolver)[0]

    baseline = cast(MomentRows, request_evidence(evidence_provider(src), MomentsRequest(metric)))[0]
    assert baseline["n"] == 1
    assert baseline["ref_y"] == pytest.approx(10.0)
    assert (
        cast(
            CountEvidence, request_evidence(evidence_provider(src), CountsRequest(kind="unit"))
        ).counts["control"]
        == 1
    )

    con.insert(
        "events",
        obj=[
            {
                "user_id": "u2",
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "enrolled",
                "experiment_id": "exp",
                "group_id": "control",
                "value": None,
            },
            {
                "user_id": "u2",
                "ts": dt.datetime(2025, 1, 2, 9, 0, 0),
                "event": "purchase",
                "experiment_id": None,
                "group_id": None,
                "value": 5.0,
            },
        ],
    )

    assert (
        cast(
            CountEvidence, request_evidence(evidence_provider(src), CountsRequest(kind="unit"))
        ).counts["control"]
        == 2
    )
    updated = cast(MomentRows, request_evidence(evidence_provider(src), MomentsRequest(metric)))[0]
    assert updated["n"] == 2
    assert updated["ref_y"] == pytest.approx(7.5)
