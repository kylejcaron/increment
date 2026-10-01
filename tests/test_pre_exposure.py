"""Pre-exposure breakout scoping (issue speg): the collider hazard, fixed.

Reproduces the artefact documented in finding 78v6 through the REAL
pipeline (Analysis.run_breakout -> segment_contrast), then shows the fix
(Property.as_of='pre_exposure' scoping the property lookup to before
first exposure) removes it.

DGP: unobserved U drives BOTH the outcome Y and a post-exposure marker M;
treatment T additionally shifts M (e.g. 'opened the new feature'). Within
each M stratum, T and U are now correlated - conditioning on M opens a
collider path T -> M <- U -> Y - so segmenting on M measured AFTER
exposure produces a large, stable, entirely artefactual segment contrast.
M measured BEFORE exposure is independent of T by construction and the
same data shows no artefact.

We deliberately use TWO tables with IDENTICAL outcome data (same unit
events) that differ only in which marker value backs the property: the
pre-exposure marker (independent of T) at a pre-exposure timestamp, or
the post-exposure marker (driven by T and U) at a post-exposure
timestamp. The only moving part is Property.as_of.

This file also carries a fast, unmarked, deterministic 8-unit end-to-end
test that exercises the same as_of='pre_exposure' dispatch directly
through Analysis.run_breakout, without the Monte Carlo
artefact-reproduction machinery above.
"""

from __future__ import annotations

import datetime as dt

import ibis
import numpy as np
import pandas as pd
import pytest

from increment import Analysis


def _units(rng: np.random.Generator, n_arm: int):
    """Per-unit (U, T, Y, M_pre, M_post) - U never reaches the library.

    Calibrated so the unscoped post-exposure marker reproduces the issue's
    artefact scale (|z| ~ 6): U drives both Y (weight 1.2) and M (weight 1),
    T additionally shifts M (weight 0.9 - 'users who opened the feature'),
    so conditioning on M opens T -> M <- U -> Y. M_pre is independent of T
    and U by construction."""
    n = 2 * n_arm
    t = np.array([0] * n_arm + [1] * n_arm)
    u = rng.normal(size=n)
    y = (1.2 * u + rng.normal(size=n)) > 0
    m_pre = rng.uniform(size=n) < 0.3  # independent of T and U
    m_post = (0.9 * t + u + rng.normal(size=n)) > 0.55  # collider on U
    return t, y, m_pre, m_post


def _tables(con, t, y, marker, ts_marker, suffix: str):
    """Load exposure + conversion events and a property table; returns
    (events_tbl, convs_tbl, props_tbl) table names."""
    n = len(t)
    uid = [f"u{i}" for i in range(n)]
    exposure = pd.DataFrame(
        {
            "user_id": uid,
            "ts": [dt.datetime(2025, 6, 2, 10, 0, 0)] * n,
            "event": "page_view",
            "group_id": np.where(t == 1, "treatment", "control"),
            "experiment_id": "exp",
        }
    )
    conv_mask = y
    conv_groups = np.where(t[conv_mask] == 1, "treatment", "control")
    conv = pd.DataFrame(
        {
            "user_id": [u for u, c in zip(uid, y, strict=True) if c],
            "ts": [dt.datetime(2025, 6, 2, 11, 0, 0)] * int(y.sum()),
            "event": ["conversion"] * int(y.sum()),
            "group_id": conv_groups.astype(str),
            "experiment_id": "exp",
            "amount": [1.0] * int(y.sum()),
            "used_feature": marker[y].astype(int),
        }
    )
    # Keep the source's finite data horizon aligned with the seven-day metric
    # window without enrolling the sentinel in the experiment.
    conv = pd.concat(
        [
            conv,
            pd.DataFrame(
                {
                    "user_id": ["background"],
                    "ts": [dt.datetime(2025, 6, 9, 23, 59, 0)],
                    "event": ["conversion"],
                    "group_id": ["control"],
                    "experiment_id": ["exp"],
                    "amount": [0.0],
                    "used_feature": [0],
                }
            ),
        ],
        ignore_index=True,
    )
    props = pd.DataFrame(
        {
            "user_id": uid,
            "ts": [ts_marker] * n,
            "used_feature": marker.astype(int),
        }
    )
    con.create_table(f"events_{suffix}", obj=exposure)
    con.create_table(f"convs_{suffix}", obj=conv)
    con.create_table(f"props_{suffix}", obj=props)
    return f"events_{suffix}", f"convs_{suffix}", f"props_{suffix}"


def _definitions(events_tbl: str, convs_tbl: str, props_tbl: str, as_of: str) -> str:
    return f"""
dialect: duckdb
fact_sources:
  - name: events
    sql: 'SELECT * FROM {events_tbl}'
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: page_view
    properties:
      - name: used_feature
        column: used_feature
        dtype: int
  - name: convs
    sql: 'SELECT * FROM {convs_tbl}'
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: conversion
        column: amount
    properties:
      - name: used_feature
        column: used_feature
        dtype: int
  - name: user_props
    sql: 'SELECT * FROM {props_tbl}'
    timestamp_column: ts
    entities: [user_id]
    facts: []
    properties:
      - name: used_feature
        column: used_feature
        dtype: int
        as_of: {as_of}
exposures:
  - name: e
    fact: page_view
metrics:
  - type: mean
    name: conv_rate
    entity: user_id
    fact: conversion
    aggregation: sum
    window_days: 7
experiments:
  - name: exp
    exposure: e
    unit: user_id
    start: 2025-06-01
    end: 2025-06-02
    observation_end: 2025-06-09
    control_group: control
    plan: {{secondaries: [conv_rate]}}
    breakouts:
      - property: used_feature
        source: user_props
"""


def _contrast_z(estimates) -> float:
    """z of the segment contrast between used_feature=0 and 1 on the
    treatment arm - the artefact's signature in issue 78v6 (z = 6.4).
    Mirrors segment_contrast's own arithmetic: log-ratio contrast with
    variance = var_a + var_b (segments are independent)."""
    arm = [e for e in estimates if e.group_id == "treatment"]
    row_a = next(e for e in arm if e.dimension_value == "0")
    row_b = next(e for e in arm if e.dimension_value == "1")
    assert row_a.require_lift().log_mean is not None and row_a.require_lift().log_se is not None
    assert row_b.require_lift().log_mean is not None and row_b.require_lift().log_se is not None
    diff = row_a.require_lift().log_mean - row_b.require_lift().log_mean
    se = (row_a.require_lift().log_se ** 2 + row_b.require_lift().log_se ** 2) ** 0.5
    return diff / se


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_post_exposure_segment_artefact_killed_by_pre_exposure_scoping(tmp_path):
    """Reproduce the collider artefact (unscoped/static), then show the
    pre_exposure-scoped lookup on IDENTICAL outcome data is null. Uses
    ~120k units/arm, the same order as the issue's n=400k."""
    rng = np.random.default_rng(42)
    t, y, m_pre, m_post = _units(rng, n_arm=120_000)

    results = {}
    for label, as_of, marker, ts_marker in [
        ("pre_exposure", "pre_exposure", m_pre, dt.datetime(2025, 6, 1, 8, 0, 0)),
        ("static", "static", m_post, dt.datetime(2025, 6, 3, 8, 0, 0)),
    ]:
        con = ibis.duckdb.connect()
        events_tbl, convs_tbl, props_tbl = _tables(con, t, y, marker, ts_marker, label)

        path = tmp_path / f"{label}.yaml"
        path.write_text(_definitions(events_tbl, convs_tbl, props_tbl, as_of))
        analysis = Analysis.from_definitions("exp", path, con)
        results[label] = _contrast_z(analysis.run_breakout())
        con.disconnect()

    pre_z, static_z = results["pre_exposure"], results["static"]
    print(f"contrast z: pre_exposure={pre_z:.3f}, static(unscoped)={static_z:.3f}")

    # The fixture must actually be a collider: unscoped must NOT be null.
    assert abs(static_z) > 4.0, (
        f"unscoped post-exposure marker gave |z|={abs(static_z):.2f} -- "
        "fixture is not reproducing the collider artefact (need > 4)"
    )
    # The fix: identical outcomes, pre-exposure-scoped marker, is null.
    assert abs(pre_z) < 2.0, (
        f"pre_exposure-scoped marker still shows |z|={abs(pre_z):.2f} -- "
        "scoping did not remove the artefact"
    )


def test_run_breakout_uses_pre_exposure_value_and_nulls_units_with_no_pre_exposure_row(tmp_path):
    """Fast (unmarked, small-N, in-memory DuckDB) end-to-end companion to
    the ``parameter_recovery``-marked artefact reproduction above - that
    test is excluded from the fast suite, so without this one the
    ``as_of == "pre_exposure"`` dispatch in
    ``DefinitionsMomentSource._build_breakout_properties_table`` would be exercised by
    NOTHING the fast suite/pre-push checks/main CI job run.
    Two claims, both through the real ``Analysis.run_breakout`` pipeline
    (not the builder function directly - that's already covered by
    ``tests/query/test_builders.py``):

    1. A unit whose property value CHANGES after exposure must be
       segmented on its PRE-exposure value. c1/t1 are "loyal" before
       exposure and "churned" after - if the post-exposure value ever
       leaked into the breakout, a "churned" segment would appear and
       "loyal" would lose two units' worth of revenue.
    2. A unit with no qualifying (pre-exposure) property row lands in the
       visible ``"__null__"`` bin, not dropped: c3/t3 only have a
       POST-exposure row, c4/t4 have none at all, yet all four still
       contribute a real (non-excluded) estimate.
    """
    con = ibis.duckdb.connect()
    exposure_ts = dt.datetime(2025, 6, 2, 10, 0, 0)
    purchase_ts = dt.datetime(2025, 6, 2, 11, 0, 0)
    freshness_ts = dt.datetime(2025, 6, 10, 11, 0, 0)
    pre_ts = dt.datetime(2025, 6, 1, 8, 0, 0)
    post_ts = dt.datetime(2025, 6, 3, 8, 0, 0)

    # "loyal" segment: pre-exposure value "loyal" for all four; c1/t1
    # additionally get a POST-exposure "churned" row that must be ignored.
    revenue = {
        "c1": 10.0,
        "c2": 12.0,
        "t1": 20.0,
        "t2": 24.0,
        "c3": 5.0,
        "c4": 7.0,
        "t3": 9.0,
        "t4": 11.0,
    }
    group_id = {
        "c1": "control",
        "c2": "control",
        "c3": "control",
        "c4": "control",
        "t1": "treatment",
        "t2": "treatment",
        "t3": "treatment",
        "t4": "treatment",
    }
    exposure_rows = [
        {
            "unit_id": uid,
            "ts": exposure_ts,
            "event": "page_view",
            "group_id": gid,
            "revenue": None,
            "experiment_id": "pre_exp",
        }
        for uid, gid in group_id.items()
    ]
    purchase_rows = [
        {
            "unit_id": uid,
            "ts": ts,
            "event": "purchase",
            "group_id": None,
            "revenue": amount,
            "experiment_id": None,
        }
        for uid, amount in revenue.items()
        for ts in (purchase_ts, freshness_ts)
    ]
    con.create_table("pre_exp_run_events", obj=exposure_rows + purchase_rows)

    prop_rows = [
        {"unit_id": "c1", "ts": pre_ts, "segment": "loyal"},
        {"unit_id": "c1", "ts": post_ts, "segment": "churned"},  # must be ignored
        {"unit_id": "c2", "ts": pre_ts, "segment": "loyal"},
        {"unit_id": "t1", "ts": pre_ts, "segment": "loyal"},
        {"unit_id": "t1", "ts": post_ts, "segment": "churned"},  # must be ignored
        {"unit_id": "t2", "ts": pre_ts, "segment": "loyal"},
        {"unit_id": "c3", "ts": post_ts, "segment": "new"},  # no pre row -> __null__
        {"unit_id": "t3", "ts": post_ts, "segment": "new"},  # no pre row -> __null__
        # c4/t4: no property row at all -> __null__
    ]
    con.create_table("pre_exp_run_props", obj=prop_rows)

    path = tmp_path / "definitions.yaml"
    path.write_text(
        """
dialect: duckdb
fact_sources:
  - name: events
    sql: 'SELECT * FROM pre_exp_run_events'
    timestamp_column: ts
    entities: [unit_id]
    facts:
      - name: page_view
      - name: purchase
        column: revenue
  - name: props
    sql: 'SELECT * FROM pre_exp_run_props'
    timestamp_column: ts
    entities: [unit_id]
    facts: []
    properties:
      - name: segment
        column: segment
        dtype: string
        as_of: pre_exposure
exposures:
  - name: e
    fact: page_view
metrics:
  - type: mean
    name: revenue
    entity: unit_id
    fact: purchase
    aggregation: sum
    window_days: 7
experiments:
  - name: pre_exp
    exposure: e
    unit: unit_id
    start: 2025-06-01
    control_group: control
    plan: {secondaries: [revenue]}
    breakouts:
      - property: segment
        source: props
        skip_missing: true
"""
    )
    analysis = Analysis.from_definitions("pre_exp", path, con)
    results = analysis.run_breakout()
    con.disconnect()

    by_segment = {r.dimension_value for r in results}
    # Claim 1: post-exposure "churned"/"new" values never surface as a
    # segment - only the pre-exposure value and the null bin do.
    assert by_segment == {"loyal", "__null__"}, by_segment

    loyal = next(r for r in results if r.dimension_value == "loyal")
    null_bin = next(r for r in results if r.dimension_value == "__null__")
    # Claim 2: units with no pre-exposure row (c3/c4/t3/t4) still produced
    # a real estimate - not dropped, not excluded for lack of units.
    assert loyal.excluded is None, loyal.excluded
    assert null_bin.excluded is None, null_bin.excluded
    # Control/treatment means differ (11->22 vs 6->10) since c1/t1's
    # post-exposure "churned" rows didn't pull them into a shared bucket.
    loyal_lift = loyal.lift
    null_bin_lift = null_bin.lift
    assert loyal_lift is not None and null_bin_lift is not None
    assert loyal_lift.value == pytest.approx(1.0, abs=0.05)
    assert null_bin_lift.value == pytest.approx(4.0 / 6.0, abs=0.05)
