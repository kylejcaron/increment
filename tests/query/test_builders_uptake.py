"""Warehouse-path uptake moments: a per-unit binary uptake flag `d` materialized on `unit_totals`, rolled up into `sum_d`/`cyd`/`cy2d` on `group_summary`.

Canonical scenario: 6 units, all exposed 2025-08-01 09:00 (u1-u3 group "treat",
u4-u6 group "control"). Revenue (mean_metric fixture: fact "purchase",
aggregation=sum, window_days=3) - one purchase per unit inside the window, so y
equals the amount exactly: u1=10.0, u2=20.0, u3=30.0, u4=1.0, u5=2.0, u6=3.0.
Uptake (fact "click"): u1 clicks 08-02 09:00 (1 day post-exposure, inside a 7-day
window); u2 clicks 08-10 09:00 (9 days post-exposure, outside a 7-day window);
u3-u6 never click.
"""

from __future__ import annotations

import datetime as dt

import pytest

from increment.query.builders import (
    first_exposures,
    group_summary,
    metric_events,
    unit_day_spine_stats,
    unit_totals,
)
from increment.query.schemas import GROUP_SUMMARY, UNIT_TOTALS
from increment.semantics.models import AnalysisPlan, Experiment
from tests.query.conftest import _table

# Fixtures


def _cancel_tol(n, ref):
    """Absolute tolerance for a cancelling residual first moment.

    ``cx1`` is ~0 but carries the last bits of an ``n * ref`` sized
    cancellation, so it needs an absolute band scaled to that magnitude.
    """
    return 1e-9 * max(1.0, abs(n * ref))


@pytest.fixture
def uptake_experiment():
    return Experiment(
        name="exp_uptake",
        unit="unit_id",
        # Generous end: covers both the revenue metric's 3-day maturity window and the day-9 "ever" click, so no unit is administratively censored out.
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 20),
        control_group="control",
        exposure="uptake_exposure",
        plan=AnalysisPlan(),
    )


@pytest.fixture(scope="session")
def uptake_exposure_events(con):
    """6 units, all exposed 2025-08-01 09:00: u1-u3 treat, u4-u6 control."""
    return _table(
        con,
        [
            {
                "unit_id": unit_id,
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_uptake",
                "group_id": group_id,
            }
            for unit_id, group_id in [
                ("u1", "treat"),
                ("u2", "treat"),
                ("u3", "treat"),
                ("u4", "control"),
                ("u5", "control"),
                ("u6", "control"),
            ]
        ],
        "uptake_exposure_events",
    )


@pytest.fixture(scope="session")
def uptake_purchase_events(con):
    """One purchase per unit, same day as exposure - y == amount exactly."""
    return _table(
        con,
        [
            {
                "unit_id": unit_id,
                "ts": dt.datetime(2025, 8, 1, 10, 0, 0),
                "event": "purchase",
                "amount": amount,
            }
            for unit_id, amount in [
                ("u1", 10.0),
                ("u2", 20.0),
                ("u3", 30.0),
                ("u4", 1.0),
                ("u5", 2.0),
                ("u6", 3.0),
            ]
        ],
        "uptake_purchase_events",
    )


@pytest.fixture(scope="session")
def uptake_click_events(con):
    """u1 clicks 1 day post-exposure; u2 clicks 9 days post-exposure;
    u3-u6 never click."""
    return _table(
        con,
        [
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 2, 9, 0, 0), "event": "click"},
            {"unit_id": "u2", "ts": dt.datetime(2025, 8, 10, 9, 0, 0), "event": "click"},
        ],
        "uptake_click_events",
    )


@pytest.fixture
def uptake_pipeline_kwargs(
    uptake_experiment, uptake_exposure_events, uptake_purchase_events, mean_metric
):
    """``(spine, stats, metric, experiment)`` for the revenue metric -
    everything ``unit_totals`` needs besides the uptake-specific kwargs
    each test supplies itself.
    """
    exposures = first_exposures(uptake_exposure_events, uptake_experiment)
    events = metric_events(uptake_purchase_events, mean_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, uptake_experiment, mean_metric.name)
    return {
        "spine": spine,
        "stats": stats,
        "metric": mean_metric,
        "experiment": uptake_experiment,
    }


# Tests


def test_group_summary_uptake_moments_windowed(con, uptake_pipeline_kwargs, uptake_click_events):
    """``window_days=7`` - u1's day-1 click counts, u2's day-9 click doesn't."""
    click_events = uptake_click_events.filter(uptake_click_events.event == "click")
    totals = unit_totals(**uptake_pipeline_kwargs, uptake_events=click_events, uptake_window_days=7)
    assert set(totals.columns) == UNIT_TOTALS

    d_by_unit = {r["unit_id"]: r["d"] for r in con.to_pyarrow(totals).to_pylist()}
    assert d_by_unit == {"u1": 1.0, "u2": 0.0, "u3": 0.0, "u4": 0.0, "u5": 0.0, "u6": 0.0}

    summary = group_summary(totals)
    assert set(summary.columns) == GROUP_SUMMARY
    rows = con.to_pyarrow(summary).to_pylist()
    assert {r["group_id"] for r in rows} == {"treat", "control"}

    # The uptake moments are centered on the group's OVERALL ref_y, never
    # on the taken-up subgroup's mean: treat y = [10, 20, 30] -> ref_y=20.
    t = {r["group_id"]: r for r in rows}["treat"]
    assert t["ref_y"] == pytest.approx(20.0)
    assert abs(t["cy1"]) < _cancel_tol(3, 20.0)
    assert t["sum_d"] == 1  # only u1 inside the 7-day window
    assert t["cyd"] == pytest.approx(-10.0)  # 1 * (10 - 20)
    assert t["cy2d"] == pytest.approx(100.0)  # 1 * (10 - 20)**2

    c = {r["group_id"]: r for r in rows}["control"]
    assert c["ref_y"] == pytest.approx(2.0)
    assert c["sum_d"] == 0
    assert c["cyd"] == 0.0
    assert c["cy2d"] == 0.0


def test_group_summary_uptake_moments_ever(con, uptake_pipeline_kwargs, uptake_click_events):
    """``window_days=None`` ("ever took up") - u2's day-9 click now counts too."""
    click_events = uptake_click_events.filter(uptake_click_events.event == "click")
    totals = unit_totals(
        **uptake_pipeline_kwargs, uptake_events=click_events, uptake_window_days=None
    )
    rows = con.to_pyarrow(group_summary(totals)).to_pylist()

    t = {r["group_id"]: r for r in rows}["treat"]
    assert t["ref_y"] == pytest.approx(20.0)
    assert t["sum_d"] == 2  # u1 and u2 both took up, ever
    assert t["cyd"] == pytest.approx(-10.0)  # (10-20)*1 + (20-20)*1 + (30-20)*0
    assert t["cy2d"] == pytest.approx(100.0)  # (10-20)**2 + (20-20)**2

    c = {r["group_id"]: r for r in rows}["control"]
    assert c["sum_d"] == 0
    assert c["cyd"] == 0.0
    assert c["cy2d"] == 0.0


@pytest.mark.creates_tables
def test_group_summary_covariate_uptake_cross_moment(
    con, uptake_experiment, uptake_exposure_events, uptake_click_events, mean_metric
):
    """``cxd`` - the only moment the CUPED-adjusted LATE adds. It exists
    exactly when a pre-period covariate AND an uptake fact are both
    configured, which is the definitions path's own combination
    (``n_pre_periods > 0`` under an Encouragement design)."""
    from increment.query.builders import pre_period_stats

    experiment = uptake_experiment.model_copy(update={"n_pre_periods": 3})
    # Pre-exposure purchases (exposure is 2025-08-01 09:00) become x; the
    # post-exposure ones become y.
    purchases = _table(
        con,
        [
            {"unit_id": u, "ts": ts, "event": "purchase", "amount": amt}
            for u, ts, amt in [
                ("u1", dt.datetime(2025, 7, 31, 9, 0, 0), 7.0),
                ("u2", dt.datetime(2025, 7, 31, 9, 0, 0), 3.0),
                ("u4", dt.datetime(2025, 7, 31, 9, 0, 0), 5.0),
                ("u1", dt.datetime(2025, 8, 1, 10, 0, 0), 10.0),
                ("u2", dt.datetime(2025, 8, 1, 10, 0, 0), 20.0),
                ("u3", dt.datetime(2025, 8, 1, 10, 0, 0), 30.0),
                ("u4", dt.datetime(2025, 8, 1, 10, 0, 0), 1.0),
                ("u5", dt.datetime(2025, 8, 1, 10, 0, 0), 2.0),
                ("u6", dt.datetime(2025, 8, 1, 10, 0, 0), 3.0),
            ]
        ],
        "uptake_cuped_purchase_events",
    )
    exposures = first_exposures(uptake_exposure_events, experiment)
    events = metric_events(purchases, mean_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, mean_metric.name)
    pre_stats = pre_period_stats(events, exposures, experiment, source_key="pre")
    clicks = uptake_click_events.filter(uptake_click_events.event == "click")
    totals = unit_totals(
        spine,
        stats,
        mean_metric,
        experiment,
        pre_stats=pre_stats,
        uptake_events=clicks,
        uptake_window_days=7,
    )
    per_unit = {r["unit_id"]: r for r in con.to_pyarrow(totals).to_pylist()}
    assert per_unit["u1"]["x"] == 7.0 and per_unit["u1"]["d"] == 1.0
    # Units with no pre-period purchase zero-fill: treat x = [7, 3, 0],
    # control x = [5, 0, 0].
    assert [per_unit[u]["x"] for u in ("u2", "u3")] == [3.0, 0.0]
    assert [per_unit[u]["x"] for u in ("u4", "u5", "u6")] == [5.0, 0.0, 0.0]

    rows = {r["group_id"]: r for r in con.to_pyarrow(group_summary(totals)).to_pylist()}
    # Only u1 took up inside the window, and cxd centres its covariate on
    # the treat group's OWN ref_x = (7 + 3 + 0) / 3.
    treat_ref_x = 10.0 / 3.0
    assert rows["treat"]["ref_x"] == pytest.approx(treat_ref_x)
    assert abs(rows["treat"]["cx1"]) < _cancel_tol(3, treat_ref_x)
    assert rows["treat"]["cxd"] == pytest.approx(7.0 - treat_ref_x)
    # No control unit took up, so every d*(x - ref_x) term is zero - 0.0,
    # NOT null: the moment is materialised, it just has nothing in it.
    assert rows["control"]["ref_x"] == pytest.approx(5.0 / 3.0)
    assert rows["control"]["cxd"] == pytest.approx(0.0)


def test_no_uptake_fact_leaves_moments_null(con, uptake_pipeline_kwargs):
    """No ``uptake_events`` passed - ``d``/``sum_d``/``cyd``/``cy2d``
    are NULL (absent), not zero - distinguishing "no encouragement design
    configured" from "configured, and nobody took up".
    """
    totals = unit_totals(**uptake_pipeline_kwargs)
    assert set(totals.columns) == UNIT_TOTALS
    totals_rows = con.to_pyarrow(totals).to_pylist()
    assert len(totals_rows) == 6
    assert all(r["d"] is None for r in totals_rows)

    rows = con.to_pyarrow(group_summary(totals)).to_pylist()
    assert len(rows) == 2  # treat, control
    for field in ("sum_d", "cyd", "cy2d", "cxd"):
        assert all(r.get(field) is None for r in rows)


def test_ratio_metric_with_uptake_events_builds_and_scores(con, uptake_experiment):
    """RatioMetric + ``uptake_events`` must build and execute: the ratio
    branch's numerator/denominator join used to leave a dangling
    ``unit_id_right`` that collided with the uptake join
    (``IntegrityError: Name collisions``) - an ``Encouragement`` design
    carries no metric-type restriction, so a declared ratio metric
    reaches this path in production.
    """
    import ibis

    from increment.query.builders import post_exposure_stats
    from increment.semantics.models import Measure, RatioMetric

    metric = RatioMetric(
        name="rev_per_view",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum"),
        denominator=Measure(fact="page_view", aggregation="count"),
    )
    fe = dt.datetime(2025, 8, 1, 9, 0, 0)
    exposures_t = ibis.memtable(
        [
            {
                "unit_id": u,
                "experiment_id": "exp_uptake",
                "group_id": g,
                "first_exposure_ts": fe,
            }
            for u, g in [("u1", "treat"), ("u2", "control")]
        ]
    )
    num_t = ibis.memtable(
        [
            {
                "unit_id": "u1",
                "ts": fe + dt.timedelta(hours=3),
                "event": "purchase",
                "amount": 10.0,
            },
            {"unit_id": "u2", "ts": fe + dt.timedelta(hours=4), "event": "purchase", "amount": 5.0},
        ]
    )
    den_t = ibis.memtable(
        [
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=1, hours=1), "event": "page_view"},
            {"unit_id": "u2", "ts": fe + dt.timedelta(days=1, hours=2), "event": "page_view"},
        ]
    )
    uptake_t = ibis.memtable(
        [{"unit_id": "u1", "ts": fe + dt.timedelta(hours=1), "event": "click"}]
    )
    num_events = metric_events(num_t, metric, value_column="amount", part="numerator")
    den_events = metric_events(den_t, metric, part="denominator")
    spine, stats = unit_day_spine_stats(exposures_t, num_events, uptake_experiment, metric.name)
    den_stats = post_exposure_stats(den_events, exposures_t, source_key="den")

    totals = unit_totals(
        spine,
        stats,
        metric,
        uptake_experiment,
        den_stats=den_stats,
        uptake_events=uptake_t,
        uptake_window_days=7,
        warn_on_censoring=False,
    )
    assert set(totals.columns) == UNIT_TOTALS
    rows = {r["unit_id"]: r for r in con.to_pyarrow(totals).to_pylist()}
    assert rows["u1"]["d"] == 1.0  # clicked inside the window
    assert rows["u2"]["d"] == 0.0  # never clicked
    assert rows["u1"]["y"] == 10.0 and rows["u1"]["y_den"] == 1.0
    assert rows["u2"]["y"] == 5.0 and rows["u2"]["y_den"] == 1.0

    summary = group_summary(totals)
    srows = {r["group_id"]: r for r in con.to_pyarrow(summary).to_pylist()}
    assert srows["treat"]["sum_d"] == 1.0
    # One unit per arm, so the subgroup moment centres on that unit's own
    # value and vanishes; ref_y carries the 10.0 it used to encode.
    assert srows["treat"]["ref_y"] == 10.0
    assert srows["treat"]["cyd"] == 0.0
    assert srows["control"]["sum_d"] == 0.0


def test_group_summary_without_d_column_still_emits_full_schema(con):
    """A totals table with NO ``d`` column must still emit the full declared
    schema - the uptake family lands as typed NULLs, never dropped. A short
    row that silently omits declared columns is a contract break a consumer
    cannot safely reinterpret, so the wire shape always equals the schema
    regardless of whether an uptake fact was materialised upstream."""
    import ibis

    totals = ibis.memtable(
        [
            {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": g,
                "store_id": s,
                "y": y,
                "x": x,
                "y_den": 1.0,
            }
            for g, s, y, x in [
                ("control", "s1", 1.0, 0.5),
                ("control", "s1", 2.0, 1.5),
                ("control", "s2", 3.0, 0.5),
                ("treat", "s3", 4.0, 1.5),
                ("treat", "s3", 5.0, 0.5),
                ("treat", "s4", 6.0, 1.5),
            ]
        ]
    )
    assert "d" not in totals.columns

    summary = group_summary(totals)
    assert set(summary.columns) == GROUP_SUMMARY
    for row in con.to_pyarrow(summary).to_pylist():
        for slot in ("sum_d", "cyd", "cy2d", "cxd"):
            assert row[slot] is None, (row["group_id"], slot)

    clustered = group_summary(totals, cluster="store_id")
    assert set(clustered.columns) == GROUP_SUMMARY
    for row in con.to_pyarrow(clustered).to_pylist():
        for slot in ("sum_d", "cyd", "cy2d", "cxd"):
            assert row[slot] is None, (row["group_id"], slot)
