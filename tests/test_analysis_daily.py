"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pytest

from increment import Analysis
from increment.breakout.estimates import DailyLiftEstimate, DailyMetricValue
from increment.errors import IncrementWarning, InvalidRequestError, UnsupportedRequestError
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    MeanMetric,
    MultiplicitySpec,
    RetentionMetric,
)
from tests.analysis_factory import make_analysis, make_analysis_like
from tests.warning_codes import warning_codes


@pytest.fixture(scope="module")
def retention_cuped_con():
    """Seed enough first-cohort units for every retention lift slice to estimate."""
    import ibis

    from examples._seed import seed_event_log

    con = ibis.duckdb.connect()
    seed_event_log(con, n_units=50_000, with_pre_period=True)
    return con


def _analysis_with_country_breakout(con):
    """Build an ``Analysis`` with one MeanMetric and a single
    ``country`` breakout (US/CA, deliberately different control-arm means
    so a cross-segment mixup would flip a lift's sign) - bypasses
    ``load()``'s YAML-file requirement the same way
    ``tests/query/test_builders.py``'s ``_analysis_with_cross_source_breakouts``
    does (``__new__`` plus manual attribute assignment), since
    ``Analysis``'s only file-reading step is
    ``load(definitions_path)``.
    """
    if "breakout_run_events" not in con.list_tables():
        # Exposure: 2 control + 2 treatment units per country.
        exposure_groups = {
            "US": {"control": ["bu1", "bu2"], "treatment": ["bu3", "bu4"]},
            "CA": {"control": ["bu5", "bu6"], "treatment": ["bu7", "bu8"]},
        }
        country_by_unit = {
            uid: country
            for country, groups in exposure_groups.items()
            for units in groups.values()
            for uid in units
        }
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "country_code": country,
                "revenue": None,
                # first_exposures' dedup semi-joins on (unit_id, experiment_id): a NULL experiment_id never matches itself (SQL NULL = NULL is not true), so every row needs the real experiment name.
                "experiment_id": "breakout_run_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        # Revenue (mean-metric fact), one purchase per unit: US control mean=10, US treatment mean=15 (+50% lift); CA control mean=20, CA treatment mean=15 (-25% lift) - opposite-signed so cross-segment contamination flips a sign.
        revenue_by_unit = {
            "bu1": 9.0,
            "bu2": 11.0,
            "bu3": 14.0,
            "bu4": 16.0,
            "bu5": 19.0,
            "bu6": 21.0,
            "bu7": 14.0,
            "bu8": 16.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in revenue_by_unit.items()
        ]
        con.create_table("breakout_run_events", obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM breakout_run_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "breakout_run_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


@pytest.fixture(scope="module")
def country_daily_lift_by_segment(con):
    """Plain ``run_daily_lift(dimension="country")`` on the country fixture,
    computed once and shared the same way."""
    return _analysis_with_country_breakout(con).run_daily_lift(dimension="country")


def _analysis_with_daily_events(con):
    """Build an ``Analysis`` (no breakout) whose revenue events
    span 2 distinct calendar days with deliberately different per-day arm
    means (day 1: treatment > control; day 2: treatment < control) - the
    time-axis analogue of ``_analysis_with_country_breakout``'s
    opposite-signed-lift design, so a bug that mixed two days' moments
    into one ``estimate_lift`` call would show up as a wrong sign, not
    just a slightly-off number.
    """
    if "daily_run_events" not in con.list_tables():
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "revenue": None,
                "experiment_id": "daily_run_exp",
            }
            for group_id, units in {
                "control": ["du1", "du2"],
                "treatment": ["du3", "du4"],
            }.items()
            for uid in units
        ]
        # Day 1: control mean=10, treatment mean=14 (+lift). Day 2: control mean=20, treatment mean=14 (-lift) - opposite-signed lift per day, so cross-day contamination shows as a wrong sign.
        purchases = {
            (datetime(2025, 6, 1, 10, 0, 0), "du1"): 9.0,
            (datetime(2025, 6, 1, 10, 0, 0), "du2"): 11.0,
            (datetime(2025, 6, 1, 10, 0, 0), "du3"): 13.0,
            (datetime(2025, 6, 1, 10, 0, 0), "du4"): 15.0,
            (datetime(2025, 6, 2, 10, 0, 0), "du1"): 19.0,
            (datetime(2025, 6, 2, 10, 0, 0), "du2"): 21.0,
            (datetime(2025, 6, 2, 10, 0, 0), "du3"): 13.0,
            (datetime(2025, 6, 2, 10, 0, 0), "du4"): 15.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": ts,
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "experiment_id": None,
            }
            for (ts, uid), amount in purchases.items()
        ]
        con.create_table("daily_run_events", obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM daily_run_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "daily_run_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "end": "2025-06-02",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


@pytest.fixture(scope="module")
def daily_events_daily_values(con):
    """Plain ``run_daily()`` on the daily fixture, computed once - the
    tests below assert different facets of one deterministic readout."""
    return _analysis_with_daily_events(con).run_daily()


@pytest.fixture(scope="module")
def daily_events_daily_lifts(con):
    """Plain ``run_daily_lift()`` on the daily fixture, computed once and
    shared the same way."""
    return _analysis_with_daily_events(con).run_daily_lift()


def test_run_daily_returns_per_day_metric_values(daily_events_daily_values):
    """Analysis.run_daily() wires daily_group_summary (no
    breakout dimension) into run_daily and returns one DailyMetricValue
    per (day x arm), with per-day absolute means matching the day's
    actual purchase amounts."""
    results = daily_events_daily_values

    assert len(results) == 4  # 1 metric x 2 days x 2 arms
    for r in results:
        assert isinstance(r, DailyMetricValue)
        assert r.metric == "revenue"
        assert r.n == 2
        assert r.value.lb is not None and r.value.ub is not None
        assert r.value.lb < r.value.value < r.value.ub

    by_key = {(r.ds, r.group_id): r for r in results}
    assert set(by_key) == {
        (date(2025, 6, 1), "control"),
        (date(2025, 6, 1), "treatment"),
        (date(2025, 6, 2), "control"),
        (date(2025, 6, 2), "treatment"),
    }
    assert by_key[(date(2025, 6, 1), "control")].value.value == pytest.approx(10.0)
    assert by_key[(date(2025, 6, 1), "treatment")].value.value == pytest.approx(14.0)
    assert by_key[(date(2025, 6, 2), "control")].value.value == pytest.approx(20.0)
    assert by_key[(date(2025, 6, 2), "treatment")].value.value == pytest.approx(14.0)


def test_run_daily_lift_returns_per_day_lift_estimates(daily_events_daily_lifts):
    """Analysis.run_daily_lift() wires daily_group_summary
    into run_daily_lift and returns one DailyLiftEstimate per (day x
    method x treatment arm), correctly isolated per day (day 1: positive
    lift, day 2: negative lift - opposite signs, not cross-contaminated
    across days)."""
    results = daily_events_daily_lifts

    assert len(results) == 2  # 1 metric x 1 method x 1 treatment arm x 2 days
    for r in results:
        assert isinstance(r, DailyLiftEstimate)
        assert r.metric == "revenue"
        assert r.group_id == "treatment"

    by_day = {r.ds: r for r in results}
    assert set(by_day) == {date(2025, 6, 1), date(2025, 6, 2)}
    assert by_day[date(2025, 6, 1)].require_lift().value > 0, (
        "day 1 treatment (14) > control (10) -- expected positive lift"
    )
    assert by_day[date(2025, 6, 2)].require_lift().value < 0, (
        "day 2 treatment (14) < control (20) -- expected negative lift"
    )


def test_run_daily_lift_inherits_declared_plan_alpha(con):
    """Analysis.run_daily_lift() must use the declared plan's ``alpha``
    instead of a hardcoded 0.05: a plan declared with alpha=0.1 (a 90%
    interval) must produce visibly narrower confidence intervals than
    the same fixture's default (alpha=0.05, 95% interval) readout, and
    each estimate's own ``lift.alpha`` must reflect the declared value."""
    base = _analysis_with_daily_events(con)
    narrow_alpha_analysis = make_analysis_like(
        base, plan=AnalysisPlan(alpha=0.1, secondaries=["revenue"])
    )

    default_results = base.run_daily_lift()
    narrow_results = narrow_alpha_analysis.run_daily_lift()

    assert len(default_results) == len(narrow_results) == 2
    for r in default_results:
        assert r.lift is not None
        assert r.require_lift().alpha == pytest.approx(0.05)
    for r in narrow_results:
        assert r.lift is not None
        assert r.require_lift().alpha == pytest.approx(0.1)

    default_by_day = {r.ds: r for r in default_results}
    narrow_by_day = {r.ds: r for r in narrow_results}
    for ds in default_by_day:
        default_lift = default_by_day[ds].lift
        narrow_lift = narrow_by_day[ds].lift
        assert (
            default_lift is not None and default_lift.lb is not None and default_lift.ub is not None
        )
        assert narrow_lift is not None and narrow_lift.lb is not None and narrow_lift.ub is not None
        default_width = default_lift.ub - default_lift.lb
        narrow_width = narrow_lift.ub - narrow_lift.lb
        assert narrow_width < default_width, (
            f"alpha=0.1 (90% CI) must be narrower than alpha=0.05 (95% CI) on {ds}"
        )


def test_run_daily_and_run_daily_lift_empty_when_no_metrics(con):
    """Both un-broken-out daily methods return [] when the experiment
    declares no metrics - mirrors run_breakout's own empty-input
    contract (there: no breakouts; here: no metrics)."""
    analysis = _analysis_with_daily_events(con)
    analysis = make_analysis_like(analysis, [], plan=AnalysisPlan())

    assert analysis.run_daily() == []
    assert analysis.run_daily_lift() == []


def test_run_daily_raises_when_experiment_has_retention_metric(con):
    """Analysis.run_daily() raises ValueError naming a
    RetentionMetric present in the experiment's metrics/guardrails,
    checked up front, before any query runs (see run_daily's own
    ``Raises`` docstring entry)."""
    analysis = _analysis_with_daily_events(con)
    analysis = make_analysis_like(
        analysis,
        [
            *analysis.metrics,
            RetentionMetric(
                name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
            ),
        ],
    )

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily()
    assert raised.value.code == "breakout.retention.unbounded"
    assert raised.value.context["names"] == ("d7_retention",)


def test_run_daily_lift_raises_when_experiment_has_retention_metric(con):
    """Analysis.run_daily_lift() raises ValueError naming a
    RetentionMetric present in the experiment's metrics/guardrails -
    same up-front check as run_daily, never a partial-then-crash loop."""
    analysis = _analysis_with_daily_events(con)
    analysis = make_analysis_like(
        analysis,
        [
            *analysis.metrics,
            RetentionMetric(
                name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
            ),
        ],
    )

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift()
    assert raised.value.code == "breakout.retention.unbounded"
    assert raised.value.context["names"] == ("d7_retention",)


def test_run_daily_lift_raises_for_encouragement_design_before_any_query_runs(con, monkeypatch):
    """Analysis.run_daily_lift() refuses outright when this Analysis's
    design is an Encouragement design - per-day incremental LATE is
    statistically meaningless under a weak daily first stage, and this
    method has no `estimands` parameter to narrow away from "late" with
    (unlike `run_asof_lift`), so the whole call is refused rather than
    silently reporting itt-only. The refusal fires BEFORE
    `breakout_summaries`/any query runs, proven by making
    `breakout_summaries` blow up if called (mirrors
    `test_run_daily_lift_dimension_raises_before_breakout_summaries_runs`'s
    own proof technique for the RetentionMetric guard)."""
    from increment.analysis import Analysis as _Analysis

    def _boom(self, *args, **kwargs):
        raise AssertionError("breakout_summaries should not be called")

    monkeypatch.setattr(_Analysis, "breakout_summaries", _boom)

    analysis = _analysis_with_asof_encouragement_events(con)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift()
    assert raised.value.code == "facade.analysis.encouragement_daily_late"


def test_run_daily_and_run_daily_lift_unaffected_without_retention_metric(
    daily_events_daily_values, daily_events_daily_lifts
):
    """Neither method's new guard fires for an experiment whose declared
    metrics/guardrails contain no RetentionMetric - both still return
    their normal per-day results."""
    assert daily_events_daily_values != []
    assert daily_events_daily_lifts != []


def test_run_daily_metrics_override_excludes_retention_metric(con):
    """Passing `metrics=` lets a caller scope the call to the
    non-Retention subset even though the experiment's full declared list still
    contains a RetentionMetric.
    """
    analysis = _analysis_with_daily_events(con)
    retention = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, retention])
    non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]

    assert analysis.run_daily(metrics=non_retention) != []
    assert analysis.run_daily_lift(metrics=non_retention) != []


@pytest.fixture(scope="module")
def cohort_daily_values(con):
    """``run_daily`` over the mean metrics plus a bounded retention metric on
    new_onboarding_v2, computed once - shared by the cohort-series tests."""
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = next(m for m in analysis.metrics if m.name == "d7_retention").model_copy(
        update={"threshold_days": (7, 14)}
    )
    mean_metrics = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
    return analysis.run_daily(metrics=[*mean_metrics, bounded])


def test_run_daily_returns_cohort_indexed_retention_rows(cohort_daily_values):
    """Retention rows carry ds_basis='cohort'; other metrics stay 'calendar'.

    new_onboarding_v2's u1-u4 are all exposed 2025-01-20; d7_retention
    (band [7, 14)) matures for all four on 2025-02-03.
    """
    results = cohort_daily_values

    retention_rows = [r for r in results if r.metric == "d7_retention"]
    other_rows = [r for r in results if r.metric != "d7_retention"]
    assert retention_rows, "retention must produce a cohort series"
    assert other_rows, "other metrics must keep their activity series"
    assert all(r.ds_basis == "cohort" for r in retention_rows)
    assert all(r.ds_basis == "calendar" for r in other_rows)
    assert retention_rows[0].ds == date(2025, 1, 20), "cohort ds is the exposure date"
    # Regression pin: the band migration must not move the cohort series' numbers. One cohort (2025-01-20), band [7,14): control={u1 (day7->y=1), u2 (day6, before band->y=0)}->mean 0.5; treatment={u3 (day8), u4 (day9)}->mean 1.0.
    by_arm = {r.group_id: r for r in retention_rows}
    assert set(by_arm) == {"control", "treatment"}
    assert by_arm["control"].value.value == pytest.approx(0.5)
    assert by_arm["treatment"].value.value == pytest.approx(1.0)
    assert by_arm["control"].n == 2 and by_arm["treatment"].n == 2


def test_run_daily_lift_returns_cohort_indexed_retention_rows(con):
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = next(m for m in analysis.metrics if m.name == "d7_retention").model_copy(
        update={"threshold_days": (7, 14)}
    )
    estimates = analysis.run_daily_lift(metrics=[bounded])
    assert estimates
    assert all(e.ds_basis == "cohort" for e in estimates)


def test_run_daily_still_rejects_unbounded_retention_metric(con):
    """Both cohort-view entry points refuse an unbounded band, and the
    error routes the caller to the as-of view: on the cohort axis an
    unbounded band's slope would measure observation time, not retention
    (newer cohorts have simply been watched less, and elapsed observation
    time is perfectly collinear with the cohort axis)."""
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    unbounded = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(metrics=[unbounded])
    assert raised.value.code == "breakout.retention.unbounded"
    assert raised.value.context["names"] == ("d7_retention",)
    with pytest.raises(InvalidRequestError) as raised_lift:
        analysis.run_daily_lift(metrics=[unbounded])
    assert raised_lift.value.code == "breakout.retention.unbounded"


def test_run_daily_cohort_series_ends_before_the_activity_series(cohort_daily_values):
    """The lagged fill is visible: retention's last cohort precedes the cutoff."""
    results = cohort_daily_values
    last_retention = max(r.ds for r in results if r.metric == "d7_retention")
    last_other = max(r.ds for r in results if r.metric != "d7_retention")
    assert last_retention < last_other


def _analysis_with_stale_retention_data(con):
    """Build an ``Analysis`` (no breakout) with one bounded
    RetentionMetric where the fact source's real data stops well before
    the cohort's maturity date, but ``observation_end`` is declared far
    enough out that - WITHOUT the ``data_as_of`` cap threaded from the
    real fact table - the cohort would wrongly be treated as matured.

    4 units all exposed 2025-08-01 (2 per arm). c1/t1 return via an
    ``app_open`` event on 2025-08-02 (a real, already-loaded event); c2/t2
    have no ``app_open`` row at all - indistinguishable, in the raw
    event data, between "did not return" and "data for that unit hasn't
    loaded yet". The fact source's freshness bound comes from the metric's
    own fact (``app_open``, filtered via ``event == fact`` - see
    :meth:`Analysis._data_as_of`), whose last event is
    2025-08-02, long before the cohort's maturity date (exposure + window_days=10 =
    2025-08-11). ``observation_end=2025-12-01`` is declared far past that
    maturity date, so only the ``data_as_of`` cap can prevent the cohort
    from being (wrongly) admitted.
    """
    if "stale_retention_events" not in con.list_tables():
        exposure_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "experiment_id": "stale_retention_exp",
            }
            for group_id, units in {
                "control": ["c1", "c2"],
                "treatment": ["t1", "t2"],
            }.items()
            for uid in units
        ]
        return_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 2, 10, 0, 0),
                "event": "app_open",
                "group_id": None,
                "experiment_id": None,
            }
            for uid in ["c1", "t1"]
        ]
        con.create_table("stale_retention_events", obj=exposure_rows + return_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM stale_retention_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "app_open", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "retention",
                    "name": "returned",
                    "entity": "unit_id",
                    "fact": "app_open",
                    "threshold_days": [1, 10],
                }
            ],
            "experiments": [
                {
                    "name": "stale_retention_exp",
                    "exposure": "e",
                    "unit": "unit_id",
                    "start": "2025-08-01",
                    "end": "2025-08-01",
                    "observation_end": "2025-12-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["returned"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_daily_retention_cohort_censored_by_real_fact_table_data_as_of(con):
    """``_data_as_of`` must reach a real censoring decision through
    ``Analysis.run_daily``, not just through the builder
    functions directly (Step 3's unit test already covers the builder
    level in isolation).

    Without the fix, the sole 2025-08-01 cohort would be admitted (its
    maturity date 2025-08-11 is well within the declared
    observation_end=2025-12-01), scoring c2/t2 as "not returned" for a
    period the data pipeline has not even loaded through yet. With
    ``_data_as_of`` correctly threaded from the real ``events`` fact
    table (whose last row is 2025-08-02), the cohort is censored -
    dropped entirely, not scored - so run_daily returns no rows at all.
    """
    analysis = _analysis_with_stale_retention_data(con)
    with pytest.warns(IncrementWarning) as rec:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            results = analysis.run_daily()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert results == [], (
        "the only cohort (2025-08-01) matures 2025-08-11, well past the "
        "fact table's real data (2025-08-02) -- data_as_of must censor it, "
        "not the declared observation_end=2025-12-01 alone"
    )


def _analysis_with_boundary_sensitive_freshness(con, day_boundary):
    """Build an ``Analysis`` whose censoring decision hinges on WHICH
    calendar ``_data_as_of`` buckets the last loaded fact into.

    One cohort exposed 2025-08-01 09:00 UTC (04:00 local under UTC-05:00
    - Aug 1 in both calendars) with a day-1 retention band
    (``threshold_days: [1, 2]``), so the last day the cohort's outcome
    depends on is day 1, 2025-08-02. c1/t1 return on day 0 and day 1.
    The metric's fact (``app_open``) last loaded 2025-08-02 03:10 UTC -
    UTC day Aug 2, but local day Aug 1 (22:10) under
    ``day_boundary="UTC-05:00"``. Under UTC the cohort is exactly mature
    (final day Aug 2 <= data_as_of Aug 2) and gets scored; under the
    declared boundary the pipeline has NOT loaded through local Aug 2,
    so the same instants must censor the cohort. ``observation_end`` is
    declared far out so only the ``data_as_of`` cap can censor.
    """
    if "boundary_freshness_events" not in con.list_tables():
        exposure_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "experiment_id": "boundary_freshness_exp",
            }
            for group_id, units in {
                "control": ["c1", "c2"],
                "treatment": ["t1", "t2"],
            }.items()
            for uid in units
        ]
        return_rows = [
            {
                "unit_id": uid,
                "ts": ts,
                "event": "app_open",
                "group_id": None,
                "experiment_id": None,
            }
            for uid in ["c1", "t1"]
            # A day-0 return plus a day-1 return that doubles as the freshness row: the 2025-08-02 03:10 UTC max is what _data_as_of buckets (UTC day Aug 2, local day Aug 1).
            for ts in [datetime(2025, 8, 1, 12, 0, 0), datetime(2025, 8, 2, 3, 10, 0)]
        ]
        con.create_table("boundary_freshness_events", obj=exposure_rows + return_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM boundary_freshness_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "app_open", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "retention",
                    "name": "returned",
                    "entity": "unit_id",
                    "fact": "app_open",
                    "threshold_days": [1, 2],
                }
            ],
            "experiments": [
                {
                    "name": "boundary_freshness_exp",
                    "exposure": "e",
                    "unit": "unit_id",
                    "start": "2025-08-01",
                    "end": "2025-08-01",
                    "observation_end": "2025-12-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["returned"]},
                    "day_boundary": day_boundary,
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_daily_censors_on_the_local_data_as_of_day(con):
    """The censoring CONSEQUENCE of the localized freshness day: the same
    instants admit the exactly-mature Aug 1 cohort under UTC (data_as_of
    Aug 2) but censor it under ``UTC-05:00`` (data_as_of local Aug 1 -
    the pipeline has not loaded through the band's final day)."""
    utc = _analysis_with_boundary_sensitive_freshness(con, "UTC")
    assert utc.run_daily() != [], (
        "under UTC the cohort matures exactly on data_as_of (2025-08-02) and must be scored"
    )

    localized = _analysis_with_boundary_sensitive_freshness(con, "UTC-05:00")
    with pytest.warns(IncrementWarning) as rec:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            localized_results = localized.run_daily()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert localized_results == [], (
        "under UTC-05:00 the last app_open row is local 2025-08-01 -- the "
        "band's final day (2025-08-02) is past the pipeline's local "
        "freshness and must be censored"
    )


def _analysis_with_boundary_sensitive_horizon(con, day_boundary):
    """Build a RUNNING-experiment ``Analysis`` (no ``end``, no
    ``observation_end``) whose spine right edge can only come from the
    Analysis-provided union event horizon.

    With ``observation_horizon`` unset, ``panel_spine`` falls back to the
    ``end_date`` the facade threads in - ``union_event_horizon(...,
    self._experiment)``. One cohort exposed 2025-08-01 09:00 UTC (04:00
    local under UTC-05:00 - Aug 1 in both calendars) with a plain count
    metric (no ``window_days``). The latest ``app_open`` lands 2025-08-05
    02:00 UTC - UTC day Aug 5, but local day Aug 4 (21:00) under
    ``UTC-05:00`` - so the spine's right edge, and with it the last
    ``ds`` :meth:`Analysis.run_daily` reports, must be Aug 5 under UTC
    but Aug 4 under the declared boundary.
    """
    if "boundary_horizon_events" not in con.list_tables():
        exposure_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "experiment_id": "boundary_horizon_exp",
            }
            for group_id, units in {
                "control": ["c1", "c2"],
                "treatment": ["t1", "t2"],
            }.items()
            for uid in units
        ]
        return_rows = [
            {
                "unit_id": uid,
                "ts": ts,
                "event": "app_open",
                "group_id": None,
                "experiment_id": None,
            }
            for uid, ts in [
                ("c1", datetime(2025, 8, 1, 12, 0, 0)),
                ("t1", datetime(2025, 8, 1, 12, 0, 0)),
                # The horizon-defining row: 2025-08-05 02:00 UTC is UTC
                # day Aug 5, local day Aug 4 under UTC-05:00.
                ("c1", datetime(2025, 8, 5, 2, 0, 0)),
            ]
        ]
        con.create_table("boundary_horizon_events", obj=exposure_rows + return_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM boundary_horizon_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "app_open", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "opens",
                    "entity": "unit_id",
                    "fact": "app_open",
                    "aggregation": "count",
                }
            ],
            "experiments": [
                {
                    "name": "boundary_horizon_exp",
                    "exposure": "e",
                    "unit": "unit_id",
                    "start": "2025-08-01",
                    # Deliberately no end / observation_end: a running experiment, so panel_spine's right edge is the union_event_horizon fallback under test here.
                    "control_group": "control",
                    "plan": {"secondaries": ["opens"]},
                    "day_boundary": day_boundary,
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_running_experiment_daily_series_ends_on_the_localized_horizon(con):
    """A running experiment's spine right edge is the union event horizon
    bucketed into the experiment's own calendar: the same 2025-08-05
    02:00 UTC event closes the daily series on Aug 5 under UTC but on
    Aug 4 under ``UTC-05:00``. Guards ``Analysis._union_event_horizon``
    passing ``self._experiment`` into ``union_event_horizon`` - an
    un-localized horizon would leave a phantom local-Aug-5 day on the
    UTC-05:00 spine."""
    utc = _analysis_with_boundary_sensitive_horizon(con, "UTC")
    assert max(r.ds for r in utc.run_daily()) == date(2025, 8, 5)

    localized = _analysis_with_boundary_sensitive_horizon(con, "UTC-05:00")
    assert max(r.ds for r in localized.run_daily()) == date(2025, 8, 4), (
        "the last app_open (2025-08-05 02:00 UTC) is local 2025-08-04 under "
        "UTC-05:00 -- the spine must not extend past the localized horizon"
    )


def _analysis_with_boundary_sensitive_denominator(con, day_boundary):
    """Build an ``Analysis`` whose only metric is a :class:`RatioMetric`
    (``revenue_per_session`` = sum(purchase.revenue) / count(session_end),
    ``numerator.window_days=1`` - canonical for the denominator's
    scoping too) where ONE denominator event is inside the day-0-only
    window ``[Aug 1, Aug 2)`` only when bucketed into the local calendar.

    Exposure 2025-08-01 09:00 UTC (04:00 local under UTC-05:00 - Aug 1
    in both calendars). Day-0 purchases: control 14 + 16, treatment
    34 + 36 (unequal within each arm, but with a tight spread: identical
    units trip infer_lift's degenerate-arm guard, and a wide spread at
    n=2/arm trips its combined log-scale SE ceiling). Every unit
    has one day-0 session (Aug 1 13:00 UTC); t1 has
    an EXTRA session at 2025-08-02 03:10 UTC - UTC day Aug 2 (day 1,
    outside the window) but local day Aug 1 (22:10, day 0, inside) under
    ``UTC-05:00``. Hand-computed group ratios:

      UTC:       control 30/2 = 15, treatment 70/2 = 35
      UTC-05:00: control 30/2 = 15, treatment 70/3 = 23.33

    An unexposed ``anchor`` unit carries one purchase and one session at
    2025-08-03 12:00 UTC (Aug 3 in BOTH calendars), so the ratio's
    ``least(num, den)`` data_as_of is calendar-invariant (Aug 3) and the
    Aug-1 cohort (mature Aug 2) is never censored - the ONLY thing that
    may differ between the two fixtures is the denominator's day bucket.
    """
    if "boundary_denominator_events" not in con.list_tables():
        exposure_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 9, 0, 0),
                "event": "page_view",
                "revenue": None,
                "group_id": group_id,
                "experiment_id": "boundary_denominator_exp",
            }
            for group_id, units in {
                "control": ["c1", "c2"],
                "treatment": ["t1", "t2"],
            }.items()
            for uid in units
        ]
        purchase_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 12, 0, 0),
                "event": "purchase",
                "revenue": revenue,
                "group_id": None,
                "experiment_id": None,
            }
            for uid, revenue in [("c1", 14.0), ("c2", 16.0), ("t1", 34.0), ("t2", 36.0)]
        ]
        session_rows = [
            {
                "unit_id": uid,
                "ts": ts,
                "event": "session_end",
                "revenue": None,
                "group_id": None,
                "experiment_id": None,
            }
            for uid, ts in [
                ("c1", datetime(2025, 8, 1, 13, 0, 0)),
                ("c2", datetime(2025, 8, 1, 13, 0, 0)),
                ("t1", datetime(2025, 8, 1, 13, 0, 0)),
                ("t2", datetime(2025, 8, 1, 13, 0, 0)),
                # The boundary-straddling denominator event: UTC day
                # Aug 2 (outside window_days=1), local day Aug 1 (inside).
                ("t1", datetime(2025, 8, 2, 3, 10, 0)),
            ]
        ]
        anchor_rows = [
            {
                "unit_id": "anchor",
                "ts": datetime(2025, 8, 3, 12, 0, 0),
                "event": event,
                "revenue": 0.0 if event == "purchase" else None,
                "group_id": None,
                "experiment_id": None,
            }
            for event in ["purchase", "session_end"]
        ]
        con.create_table(
            "boundary_denominator_events",
            obj=exposure_rows + purchase_rows + session_rows + anchor_rows,
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM boundary_denominator_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "session_end", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "ratio",
                    "name": "revenue_per_session",
                    "entity": "unit_id",
                    "numerator": {"fact": "purchase", "aggregation": "sum", "window_days": 1},
                    "denominator": {"fact": "session_end", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "boundary_denominator_exp",
                    "exposure": "e",
                    "unit": "unit_id",
                    "start": "2025-08-01",
                    "end": "2025-08-01",
                    "observation_end": "2025-08-10",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue_per_session"]},
                    "day_boundary": day_boundary,
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_ratio_denominator_buckets_events_into_the_local_day(con):
    """The ratio denominator's window scoping follows the experiment's
    own calendar: t1's 2025-08-02 03:10 UTC session is day 1 (outside
    ``window_days=1``) under UTC but day 0 (inside) under ``UTC-05:00``,
    so the treatment ratio drops from 70/2=35 to 70/3=23.33 while control
    stays at 15. Guards the denominator's ``post_exposure_stats`` call
    receiving ``experiment=self._experiment`` on ``Analysis.run``'s path
    - a UTC-bucketed denominator would make the two fixtures agree."""
    utc = _analysis_with_boundary_sensitive_denominator(con, "UTC")
    utc_treatment = next(
        r for r in utc.run() if r.metric == "revenue_per_session" and r.group_id == "treatment"
    )
    # abs_diff is the raw arm-mean difference: 35 - 15.
    assert utc_treatment.abs_diff == pytest.approx(20.0)

    localized = _analysis_with_boundary_sensitive_denominator(con, "UTC-05:00")
    localized_treatment = next(
        r
        for r in localized.run()
        if r.metric == "revenue_per_session" and r.group_id == "treatment"
    )
    # Local bucketing admits t1's extra session into the day-0 window:
    # treatment 70/3 = 23.33, so the raw difference shrinks to 70/3 - 15.
    assert localized_treatment.abs_diff == pytest.approx(70.0 / 3.0 - 15.0), (
        "the 2025-08-02 03:10 UTC session is local 2025-08-01 (day 0) under "
        "UTC-05:00 and must be counted in the treatment denominator"
    )
    assert localized_treatment.require_lift().value < utc_treatment.require_lift().value


def _analysis_with_fact_scoped_freshness(con):
    """Build an ``Analysis`` (no breakout) with one shared fact
    source backing THREE facts with deliberately different real-world
    freshness: ``page_view`` (exposure), ``app_open`` (the metric's own
    fact - STALE, real data only through 2025-08-03), and ``heartbeat``
    (a SIBLING fact on the same source that is FRESH through
    2025-08-20, but that no declared metric depends on).

    Proves ``_data_as_of`` is scoped to the metric's own fact: taking the
    shared source's unfiltered max ts would read ``heartbeat``'s
    freshness (2025-08-20) and wrongly treat the ``app_open``-backed
    retention metric as observable all the way out there, when the data
    that actually matters to THIS metric stops 2025-08-03.
    """
    if "fact_scoped_freshness_events" not in con.list_tables():
        exposure_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "experiment_id": "fact_scoped_freshness_exp",
            }
            for group_id, units in {
                "control": ["fc1", "fc2"],
                "treatment": ["ft1", "ft2"],
            }.items()
            for uid in units
        ]
        # fact B (app_open, the metric's own fact): stale, stops 2025-08-03.
        stale_return_rows = [
            {
                "unit_id": uid,
                "ts": datetime(2025, 8, 3, 10, 0, 0),
                "event": "app_open",
                "group_id": None,
                "experiment_id": None,
            }
            for uid in ["fc1", "ft1"]
        ]
        # fact A (heartbeat, a sibling fact no metric depends on): fresh, runs through 2025-08-20 - must not leak into app_open's freshness bound.
        fresh_sibling_rows = [
            {
                "unit_id": "fc1",
                "ts": datetime(2025, 8, 20, 8, 0, 0),
                "event": "heartbeat",
                "group_id": None,
                "experiment_id": None,
            }
        ]
        con.create_table(
            "fact_scoped_freshness_events",
            obj=exposure_rows + stale_return_rows + fresh_sibling_rows,
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM fact_scoped_freshness_events",
                    "timestamp_column": "ts",
                    "entities": ["unit_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "app_open", "column": None},
                        {"name": "heartbeat", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "retention",
                    "name": "returned",
                    "entity": "unit_id",
                    "fact": "app_open",
                    "threshold_days": [1, 10],
                }
            ],
            "experiments": [
                {
                    "name": "fact_scoped_freshness_exp",
                    "exposure": "e",
                    "unit": "unit_id",
                    "start": "2025-08-01",
                    "end": "2025-08-01",
                    "observation_end": "2025-12-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["returned"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_data_as_of_scoped_to_metrics_own_fact_not_masked_by_fresher_sibling_fact(con):
    """``_data_as_of`` must read the freshness of the metric's OWN fact,
    not the unfiltered max of every fact sharing that fact source.

    ``app_open`` (the metric's fact) is stale, real data stopping
    2025-08-03; ``heartbeat`` (an unrelated sibling fact on the SAME
    source) is fresh through 2025-08-20. The cohort's maturity date is
    2025-08-11. If ``_data_as_of`` were scoped to the whole source (the
    bug), ``heartbeat``'s freshness would mask ``app_open``'s staleness
    and wrongly admit the cohort (2025-08-11 <= 2025-08-20). Scoped
    correctly to ``app_open`` alone, the cohort remains censored
    (2025-08-11 > 2025-08-03).
    """
    analysis = _analysis_with_fact_scoped_freshness(con)
    with pytest.warns(IncrementWarning) as rec:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"fetch_arrow_table\(\) is deprecated, use to_arrow_table\(\) instead\.",
                category=DeprecationWarning,
                module=r"ibis\.backends\.duckdb",
            )
            results = analysis.run_daily()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert results == [], (
        "the cohort must stay censored on app_open's own freshness "
        "(2025-08-03), not get admitted via the fresher sibling fact "
        "heartbeat (2025-08-20) that shares the same fact source"
    )


def test_run_daily_dimension_returns_per_day_per_segment_values(con):
    """Analysis.run_daily(dimension="country") reuses
    breakout_summaries's already-computed daily_group_summary table and
    returns one DailyMetricValue per (day x arm x country), with
    dimension/dimension_value/source populated and matching
    breakout_summaries's own segment values - the day-axis counterpart
    to test_run_breakout_returns_per_segment_estimates, on the exact
    same fixture."""
    analysis = _analysis_with_country_breakout(con)

    results = analysis.run_daily(dimension="country")

    assert len(results) == 8  # 1 metric x 2 days x 2 arms x 2 countries
    for r in results:
        assert isinstance(r, DailyMetricValue)
        assert r.metric == "revenue"
        assert r.dimension == "country"
        assert r.source == "events"  # the resolved FactSource name
        assert r.n == 2

    by_key = {(r.ds, r.dimension_value, r.group_id): r for r in results}
    assert set(by_key) == {
        (day, dimension_value, group_id)
        for day in (date(2025, 6, 1), date(2025, 6, 2))
        for dimension_value in ("US", "CA")
        for group_id in ("control", "treatment")
    }
    # Exposure day (2025-06-01) has no purchases yet - a NaN row, not a
    # dropped one.
    for dimension_value in ("US", "CA"):
        for group_id in ("control", "treatment"):
            assert by_key[(date(2025, 6, 1), dimension_value, group_id)].value is None
    # Purchase day (2025-06-02) matches breakout_summaries' own group_summary values exactly (same 8 units).
    assert by_key[(date(2025, 6, 2), "US", "control")].value.value == pytest.approx(10.0)
    assert by_key[(date(2025, 6, 2), "US", "treatment")].value.value == pytest.approx(15.0)
    assert by_key[(date(2025, 6, 2), "CA", "control")].value.value == pytest.approx(20.0)
    assert by_key[(date(2025, 6, 2), "CA", "treatment")].value.value == pytest.approx(15.0)


def test_run_daily_without_dimension_unaffected_by_declared_breakout(con):
    """analysis.run_daily() (no `dimension`) on an experiment that DOES
    declare a `country` breakout is completely unaffected by that
    breakout's existence - dimension/dimension_value/source stay None
    on every result, and the reported means collapse across country
    exactly like this method's pre-existing (pre-dimension-parameter)
    behavior did - a direct regression guard for the strict
    backward-compatibility requirement on the un-dimensioned call."""
    analysis = _analysis_with_country_breakout(con)

    results = analysis.run_daily()

    assert len(results) == 4  # 1 metric x 2 days x 2 arms, collapsed across country
    for r in results:
        assert r.dimension is None
        assert r.dimension_value is None
        assert r.source is None
        assert r.n == 4  # both countries pooled

    by_day_arm = {(r.ds, r.group_id): r for r in results}
    assert set(by_day_arm) == {
        (date(2025, 6, 1), "control"),
        (date(2025, 6, 1), "treatment"),
        (date(2025, 6, 2), "control"),
        (date(2025, 6, 2), "treatment"),
    }
    # Exposure day (2025-06-01) has no purchases yet - a NaN row, not a
    # dropped one.
    assert by_day_arm[(date(2025, 6, 1), "control")].value is None
    assert by_day_arm[(date(2025, 6, 1), "treatment")].value is None
    # Collapsed across both countries (2 units each): control mean =
    # mean(9, 11, 19, 21) = 15; treatment mean = mean(14, 16, 14, 16) = 15.
    assert by_day_arm[(date(2025, 6, 2), "control")].value.value == pytest.approx(15.0)
    assert by_day_arm[(date(2025, 6, 2), "treatment")].value.value == pytest.approx(15.0)


def test_run_daily_dimension_metrics_override_excludes_retention_metric(con):
    """Passing `metrics=` alongside `dimension=` scopes the per-segment
    call to the non-Retention subset even though the experiment's declared
    metrics still contain a RetentionMetric.
    """
    analysis = _analysis_with_country_breakout(con)
    retention = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, retention])
    non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]

    results = analysis.run_daily(dimension="country", metrics=non_retention)

    assert results != []
    assert all(r.metric == "revenue" for r in results)


def test_run_daily_dimension_raises_when_experiment_has_retention_metric(con):
    """Analysis.run_daily(dimension=...) raises ValueError
    naming a RetentionMetric present in the experiment's metrics/
    guardrails - same up-front check as the un-dimensioned path."""
    analysis = _analysis_with_country_breakout(con)
    analysis = make_analysis_like(
        analysis,
        [
            *analysis.metrics,
            RetentionMetric(
                name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
            ),
        ],
    )

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(dimension="country")
    assert raised.value.code == "breakout.retention.unbounded"
    assert raised.value.context["names"] == ("d7_retention",)


def test_run_daily_dimension_raises_before_breakout_summaries_runs(con, monkeypatch):
    """The RetentionMetric guard fires before `breakout_summaries` (and
    hence any query) runs at all - proven by making `breakout_summaries`
    itself blow up if called, then confirming the RetentionMetric
    ValueError is what actually propagates, not the patched failure.
    Regression guard for the exact bug shape the task brief warns about:
    breakout_summaries computes daily_group_summary for EVERY declared
    RetentionMetric among the declared metrics would otherwise be caught
    deep in the per-(breakout, metric) loop, after other metrics' queries
    already ran."""
    analysis = _analysis_with_country_breakout(con)
    analysis = make_analysis_like(
        analysis,
        [
            *analysis.metrics,
            RetentionMetric(
                name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
            ),
        ],
    )

    def _boom(self):
        raise AssertionError("breakout_summaries must not run before the RetentionMetric guard")

    monkeypatch.setattr(Analysis, "breakout_summaries", _boom)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(dimension="country")
    assert raised.value.code == "breakout.retention.unbounded"


def test_run_daily_unknown_dimension_raises_naming_declared_properties(con):
    """A `dimension` that doesn't match any declared breakout's `property`
    is treated as a caller error (most likely a typo), not a legitimate
    empty result - raises ValueError naming the experiment's actual
    declared breakout properties."""
    analysis = _analysis_with_country_breakout(con)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(dimension="platform")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["declared"] == ("country",)


def test_run_daily_dimension_raises_for_metric_not_declared_on_experiment(con):
    """Regression: `run_daily(con, dimension=..., metrics=[...])` reuses
    `breakout_summaries`'s precomputed tables are only built for declared
    metrics. A `metrics=` entry that is not among those must raise rather
    than silently return an empty result.
    guard, `run_daily(con, metrics=[undeclared])` (no dimension, builds
    a fresh panel) would return real rows for `undeclared` while
    `run_daily(con, dimension=..., metrics=[undeclared])` would silently
    return `[]` - a divergence between the two call shapes with no
    error at all."""
    analysis = _analysis_with_country_breakout(con)
    undeclared_metric = MeanMetric(
        name="not_declared_on_this_experiment",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
    )

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(dimension="country", metrics=[undeclared_metric])
    assert raised.value.code == "facade.analysis.undeclared_metric_for_dimension"
    assert raised.value.context["undeclared"] == ("not_declared_on_this_experiment",)


def test_run_daily_unknown_dimension_raises_when_no_breakouts_declared(con):
    """An experiment with NO declared breakouts at all still raises for
    any `dimension` value, naming an empty list of declared properties,
    rather than silently returning []."""
    analysis = _analysis_with_daily_events(con)
    assert analysis.experiment.breakouts == ()

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(dimension="country")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["declared"] == ()


def test_run_daily_dimension_accepts_bounded_retention_metric(con):
    """The per-segment daily path takes retention too, cohort-indexed.

    Named to match the declared d7_retention guardrail: run_daily's
    dimensioned path (unlike run_asof's) rejects any metrics= entry not
    already declared on the experiment, since it reuses
    breakout_summaries's precomputed per-declared-metric tables.
    """
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = RetentionMetric(
        name="d7_retention",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    results = analysis.run_daily(dimension="country", metrics=[bounded])
    assert results
    assert {r.dimension for r in results} == {"country"}
    assert all(r.ds_basis == "cohort" for r in results)


@pytest.mark.parametrize("has_breakouts", [True, False])
@pytest.mark.parametrize("request_kind", ["unknown_name", "unknown_object", "changed_definition"])
def test_breakout_summaries_rejects_invalid_subset(con, has_breakouts, request_kind):
    from increment.errors import InvalidRequestError

    analysis = _analysis_with_country_breakout(con)
    if not has_breakouts:
        analysis = make_analysis_like(
            analysis, experiment=analysis.experiment.model_copy(update={"breakouts": ()})
        )
    declared = analysis.metrics[0]
    if request_kind == "changed_definition":
        requested = [declared.model_copy(update={"window_days": 999})]
        code = "facade.analysis_config.metric_definition_mismatch"
        field, name = "mismatched_names", declared.name
    else:
        name = "not_a_declared_metric"
        requested = (
            [name]
            if request_kind == "unknown_name"
            else [declared.model_copy(update={"name": name})]
        )
        code = "facade.analysis_config.unknown_metric_declared"
        field = "unknown_names"
    with pytest.raises(InvalidRequestError) as raised:
        analysis.breakout_summaries(metrics=requested)
    assert raised.value.code == code
    assert raised.value.context[field] == (name,)
    assert raised.value.context["caller"] == "breakout_summaries"


@pytest.mark.parametrize("by_name", [False, True])
def test_breakout_summaries_accepts_declared_definition_values(con, by_name):
    analysis = _analysis_with_country_breakout(con)
    declared = analysis.metrics[0]
    expected = analysis.breakout_summaries(metrics=[declared])
    requested = declared.name if by_name else declared.model_copy()
    actual = analysis.breakout_summaries(metrics=[requested])
    assert set(actual) == {"revenue:country:events"}
    for key, tables in actual.items():
        for grain, table in tables.items():
            sort_keys = [(name, "ascending") for name in table.column_names]
            assert table.sort_by(sort_keys).equals(expected[key][grain].sort_by(sort_keys))


def test_breakout_summaries_metrics_filter_skips_excluded_panel(con):
    """The summaries reduction returns only the requested metric's panel."""
    analysis = _analysis_with_country_breakout(con)
    second_metric = MeanMetric(
        name="page_views", entity="user_id", fact="page_view", aggregation="count"
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, second_metric])

    results = analysis.breakout_summaries(metrics=[second_metric])

    assert set(results) == {"page_views:country:events"}


def test_run_daily_dimension_metrics_filter_returns_only_requested_metric(con):
    """The public daily interface returns only the requested metric."""
    analysis = _analysis_with_country_breakout(con)
    second_metric = MeanMetric(
        name="page_views", entity="user_id", fact="page_view", aggregation="count"
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, second_metric])

    results = analysis.run_daily(dimension="country", metrics=[second_metric])

    assert results
    assert {r.metric for r in results} == {"page_views"}


def test_run_daily_lift_dimension_returns_per_day_per_segment_estimates(
    country_daily_lift_by_segment,
):
    """Analysis.run_daily_lift(dimension="country") reuses
    breakout_summaries's already-computed daily_group_summary tables and
    returns one DailyLiftEstimate per (day x segment x metric x method x
    non-control arm), with dimension/dimension_value/source populated -
    the lift counterpart to
    test_run_daily_dimension_returns_per_day_per_segment_values, on the
    exact same fixture."""
    results = country_daily_lift_by_segment

    # 1 metric x 1 method x 1 treatment arm x 2 countries x 2 days: 2025-06-01 has no purchases, so every arm's row is dropped and each segment comes back as a NaN row that day; 2025-06-02 estimates normally for both.
    assert len(results) == 4
    for r in results:
        assert isinstance(r, DailyLiftEstimate)
        assert r.metric == "revenue"
        assert r.group_id == "treatment"
        assert r.dimension == "country"
        assert r.source == "events"  # the resolved FactSource name

    assert {r.dimension_value for r in results} == {"US", "CA"}
    by_key = {(r.ds, r.dimension_value): r for r in results}
    assert by_key[(date(2025, 6, 1), "US")].lift is None
    assert by_key[(date(2025, 6, 1), "CA")].lift is None
    assert by_key[(date(2025, 6, 2), "US")].lift is not None
    assert by_key[(date(2025, 6, 2), "CA")].lift is not None


def test_run_daily_lift_dimension_isolates_segments(country_daily_lift_by_segment):
    """Each (day x segment) lift is attributed to the right segment, not
    cross-contaminated: US treatment (15) beats US control (10) while CA
    treatment (15) trails CA control (20), so a bug that partitioned the
    combined table by `ds` ALONE - pooling both countries' control rows
    into one estimate_lift call for the day - would flip a sign here.
    The integration-level counterpart to
    ``TestRunDailyLiftSegmentIsolation`` in tests/breakout/test_daily.py,
    and the day-axis counterpart to
    test_run_breakout_returns_per_segment_estimates."""
    results = country_daily_lift_by_segment

    # 2025-06-01 has no purchases and comes back as a NaN row for both segments; scope to the one usable day so this test isolates per-segment attribution, not the NaN-row mechanism.
    by_segment = {r.dimension_value: r for r in results if r.ds == date(2025, 6, 2)}
    assert set(by_segment) == {"US", "CA"}
    assert by_segment["US"].require_lift().value > 0, (
        "US treatment (15) > control (10) -- expected positive lift"
    )
    assert by_segment["CA"].require_lift().value < 0, (
        "CA treatment (15) < control (20) -- expected negative lift"
    )
    # Per-segment relative lifts (exp(x)-1 scale): US 15/10-1=+0.5, CA 15/20-1=-0.25. A day-only partition would compare both countries' treatment arms against whichever control row it iterated last, yielding the wrong pairing.
    assert by_segment["US"].require_lift().value == pytest.approx(0.5, abs=0.01)
    assert by_segment["CA"].require_lift().value == pytest.approx(-0.25, abs=0.01)


def test_run_daily_lift_without_dimension_unaffected_by_declared_breakout(con):
    """analysis.run_daily_lift() (no `dimension`) on an experiment that
    DOES declare a `country` breakout is completely unaffected by that
    breakout's existence - dimension/dimension_value/source stay None on
    every result, and the estimate collapses across country exactly like
    this method's pre-existing (pre-dimension-parameter) behavior did:
    control mean(9, 11, 19, 21) = 15 vs treatment mean(14, 16, 14, 16) =
    15, the same collapsed values
    test_run_daily_without_dimension_unaffected_by_declared_breakout
    establishes for run_daily on this fixture - i.e. no lift at all,
    which is precisely what the dimensioned view above recovers."""
    analysis = _analysis_with_country_breakout(con)

    results = analysis.run_daily_lift()

    assert len(results) == 2  # 1 metric x 1 method x 1 treatment arm x 2 days
    for r in results:
        assert r.dimension is None
        assert r.dimension_value is None
        assert r.source is None
        assert r.metric == "revenue"
        assert r.group_id == "treatment"

    by_day = {r.ds: r for r in results}
    # 2025-06-01 (exposure day) has no purchases at all - a NaN row, not
    # a dropped one, matching run_daily's own convention on this fixture.
    assert by_day[date(2025, 6, 1)].lift is None
    # Both collapsed arm means are exactly 15, so the relative lift is
    # zero up to the delta method's small second-order correction.
    assert by_day[date(2025, 6, 2)].require_lift().value == pytest.approx(0.0, abs=0.01)


def test_run_daily_lift_dimension_metrics_override_excludes_retention_metric(con):
    """Passing `metrics=` alongside `dimension=` scopes the per-segment
    call to the non-Retention subset even though declared metrics still
    contain a RetentionMetric."""
    analysis = _analysis_with_country_breakout(con)
    retention = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, retention])
    non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]

    results = analysis.run_daily_lift(dimension="country", metrics=non_retention)

    assert results != []
    assert all(r.metric == "revenue" for r in results)


def test_run_daily_lift_dimension_raises_when_experiment_has_retention_metric(con):
    """Analysis.run_daily_lift(dimension=...) raises
    ValueError naming a RetentionMetric present in the experiment's
    metrics/guardrails - same up-front check as the un-dimensioned
    path."""
    analysis = _analysis_with_country_breakout(con)
    analysis = make_analysis_like(
        analysis,
        [
            *analysis.metrics,
            RetentionMetric(
                name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
            ),
        ],
    )

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(dimension="country")
    assert raised.value.code == "breakout.retention.unbounded"
    assert raised.value.context["names"] == ("d7_retention",)


def test_run_daily_lift_dimension_raises_before_breakout_summaries_runs(con, monkeypatch):
    """The RetentionMetric guard fires before `breakout_summaries` (and
    hence any query) runs at all - proven by making `breakout_summaries`
    itself blow up if called, then confirming the RetentionMetric
    ValueError is what actually propagates, not the patched failure.
    Same ordering guarantee run_daily's dimensioned path carries
    (test_run_daily_dimension_raises_before_breakout_summaries_runs):
    breakout_summaries computes daily_group_summary for EVERY declared
    metric unconditionally, with no guard of its own."""
    analysis = _analysis_with_country_breakout(con)
    analysis = make_analysis_like(
        analysis,
        [
            *analysis.metrics,
            RetentionMetric(
                name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
            ),
        ],
    )

    def _boom(self):
        raise AssertionError("breakout_summaries must not run before the RetentionMetric guard")

    monkeypatch.setattr(Analysis, "breakout_summaries", _boom)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(dimension="country")
    assert raised.value.code == "breakout.retention.unbounded"


def test_run_daily_lift_unknown_dimension_raises_naming_declared_properties(con):
    """A `dimension` that doesn't match any declared breakout's `property`
    is a caller error (most likely a typo), not a legitimate empty result
    - raises ValueError naming the experiment's actual declared breakout
    properties, exactly as run_daily does."""
    analysis = _analysis_with_country_breakout(con)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(dimension="platform")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["declared"] == ("country",)


def test_run_daily_lift_unknown_dimension_raises_when_no_breakouts_declared(con):
    """An experiment with NO declared breakouts at all still raises for
    any `dimension` value, naming an empty list of declared properties,
    rather than silently returning []."""
    analysis = _analysis_with_daily_events(con)
    assert analysis.experiment.breakouts == ()

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(dimension="country")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["declared"] == ()


def test_run_daily_lift_dimension_raises_for_metric_not_declared_on_experiment(con):
    """`run_daily_lift(con, dimension=..., metrics=[...])` reuses
    `breakout_summaries`'s precomputed tables are only built for declared
    metrics. A `metrics=` entry that is not among those can never have a
    matching key and must raise rather than silently drop that metric."""
    analysis = _analysis_with_country_breakout(con)
    undeclared_metric = MeanMetric(
        name="not_declared_on_this_experiment",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
    )

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(dimension="country", metrics=[undeclared_metric])
    assert raised.value.code == "facade.analysis.undeclared_metric_for_dimension"
    assert raised.value.context["undeclared"] == ("not_declared_on_this_experiment",)


def test_day_axis_lift_compiles_call_time_metric_as_unassigned(con):
    analysis = _analysis_with_daily_events(con)
    metric = MeanMetric(
        name="extra_revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
    )

    for results in (
        analysis.run_daily_lift(metrics=[metric]),
        analysis.run_asof_lift(metrics=[metric]),
    ):
        assert results
        assert {row.metric for row in results} == {"extra_revenue"}
        assert {row.role for row in results} == {"unassigned"}


def test_run_daily_lift_dimension_metrics_filter_skips_excluded_panel(con):
    """The public daily-lift reduction returns only the requested panel."""
    analysis = _analysis_with_country_breakout(con)
    second_metric = MeanMetric(
        name="page_views", entity="user_id", fact="page_view", aggregation="count"
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, second_metric])

    results = analysis.run_daily_lift(dimension="country", metrics=[second_metric])

    assert results
    assert {r.metric for r in results} == {"page_views"}


@pytest.mark.slow
def test_run_daily_lift_dimension_returns_cohort_indexed_retention_rows(con):
    """Same segment path for run_daily_lift, split from any activity metrics
    sharing the call - both metric kinds must survive in one result with
    their own ds_basis and shared dimension/dimension_value/source stamping.

    Named to match the declared d7_retention guardrail - see
    test_run_daily_dimension_accepts_bounded_retention_metric for why.
    """
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = RetentionMetric(
        name="d7_retention",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    activity_metrics = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]
    estimates = analysis.run_daily_lift(dimension="country", metrics=[*activity_metrics, bounded])
    retention_estimates = [e for e in estimates if e.ds_basis == "cohort"]
    other_estimates = [e for e in estimates if e.ds_basis != "cohort"]
    assert retention_estimates, "retention must produce a cohort-indexed segment series"
    assert other_estimates, "activity metrics must keep their calendar-indexed series"
    assert all(e.dimension == "country" for e in retention_estimates)
    assert all(e.ds_basis == "cohort" for e in retention_estimates)
    assert all(e.ds_basis == "calendar" for e in other_estimates)


def _analysis_with_asof_encouragement_events(con):
    """Build a native (DuckDB, ``Analysis.from_definitions``-shape)
    ``Analysis`` with a real :class:`Encouragement` design and uptake
    fact, exercised end to end through :meth:`Analysis.run_asof_lift` -
    mirrors ``_analysis_with_asof_events``'s own idempotent-table
    construction, and ``_guardrail_analysis``'s (tests/
    test_readouts_encouragement.py) ``Analysis.__new__`` + manual
    attribute assignment + ``analysis._design = Encouragement(...)``
    pattern for a native encouragement fixture.

    30 control units and 30 treatment units are exposed on day 0
    (2025-02-01). 24 of the 30 treatment units eventually click
    "clicked" (the uptake fact, one-sided: control never clicks) - 12
    on day 0, 12 more on day 1 - so the as-of first stage grows from
    12/30 (day 0) to 24/30 (day 1 onward, frozen thereafter). Every
    unit's revenue is realized in a single day-0 purchase (noisy but
    with a real, fixed per-unit gap between clickers and non-clickers),
    so the numerator of the Wald ratio (treatment mean - control mean)
    stays constant across days while the first stage grows - the LATE
    estimate is expected to roughly HALVE from day 0 to day 1 as the
    complier population identified so far grows, then freeze, the same
    "day-over-day trend, then freeze" shape :meth:`run_asof_lift`'s
    other tests exercise for ITT.
    """
    if "asof_late_events" not in con.list_tables():
        rng = np.random.default_rng(42)
        control_units = [f"c{i}" for i in range(1, 31)]
        early_clickers = [f"t{i}" for i in range(1, 13)]
        late_clickers = [f"t{i}" for i in range(13, 25)]
        never_clickers = [f"t{i}" for i in range(25, 31)]
        treat_units = early_clickers + late_clickers + never_clickers

        control_rev = {u: float(10.0 + rng.normal(0, 2.0)) for u in control_units}
        clicker_rev = {u: float(16.0 + rng.normal(0, 2.0)) for u in early_clickers + late_clickers}
        never_rev = {u: float(10.0 + rng.normal(0, 2.0)) for u in never_clickers}
        treat_rev = {**clicker_rev, **never_rev}

        exposure_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 2, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "control",
                "revenue": None,
                "experiment_id": "asof_late_exp",
            }
            for u in control_units
        ] + [
            {
                "user_id": u,
                "ts": datetime(2025, 2, 1, 9, 0, 0),
                "event": "exposed",
                "group_id": "treatment",
                "revenue": None,
                "experiment_id": "asof_late_exp",
            }
            for u in treat_units
        ]
        click_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 2, 1, 12, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
                "experiment_id": None,
            }
            for u in early_clickers
        ] + [
            {
                "user_id": u,
                "ts": datetime(2025, 2, 2, 12, 0, 0),
                "event": "clicked",
                "group_id": None,
                "revenue": None,
                "experiment_id": None,
            }
            for u in late_clickers
        ]
        purchase_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 2, 1, 13, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": control_rev[u],
                "experiment_id": None,
            }
            for u in control_units
        ] + [
            {
                "user_id": u,
                "ts": datetime(2025, 2, 1, 13, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": treat_rev[u],
                "experiment_id": None,
            }
            for u in treat_units
        ]
        con.create_table("asof_late_events", obj=exposure_rows + click_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM asof_late_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposed", "column": None},
                        {"name": "clicked", "column": None},
                        {"name": "purchase", "column": "revenue"},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "exposed"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "asof_late_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-02-01",
                    "end": "2025-02-03",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )

    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    analysis = make_analysis(
        con,
        defs,
        _design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="button gates revenue"
            ),
            one_sided=True,
        ),
    )
    return analysis


def test_native_daily_lift_supports_cuped(seeded_pre_period_con):
    from increment import Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    rows = analysis.run_daily_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["purchase_rate"],
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert any(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


def test_native_dimensioned_daily_lift_supports_cuped(seeded_pre_period_con):
    from increment import Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    rows = analysis.run_daily_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["purchase_rate"],
        dimension="country",
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert any(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


def test_native_daily_retention_lift_supports_cuped(retention_cuped_con):
    from increment import Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", retention_cuped_con)
    rows = analysis.run_daily_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["d7_retention"],
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert all(row.ds_basis == "cohort" for row in rows)
    assert all(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


@pytest.mark.slow
def test_native_dimensioned_daily_retention_lift_supports_cuped(
    retention_cuped_con,
):
    from increment import Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", retention_cuped_con)
    analysis = make_analysis_like(
        analysis,
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
    )
    rows = analysis.run_daily_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["d7_retention"],
        dimension="country",
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert {row.dimension for row in rows} == {"country"}
    assert all(row.ds_basis == "cohort" for row in rows)
    assert all(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


def test_native_daily_lift_refuses_sequential_before_moments(seeded_pre_period_con):
    """Daily/cohort slices refuse sequential plans through the public facade."""
    from tests.sequential_cases import registered_native

    original = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    metrics = [m for m in original.metrics if m.name == "purchase_rate"]
    analysis = registered_native(original, metrics=metrics)

    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run_daily_lift(metrics=["purchase_rate"])
    assert raised.value.code == "readout.inference.disjoint_slices"


@pytest.mark.parametrize(
    ("metric", "source_fixture"),
    [
        pytest.param("avg_session_duration", "seeded_pre_period_con", id="mean"),
        pytest.param("purchase_rate", "seeded_pre_period_con", id="conversion"),
        pytest.param(
            "d7_retention",
            "retention_cuped_con",
            id="retention",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_native_daily_lift_mixed_methods_preserves_matrix(request, metric, source_fixture):
    from increment import Method

    con = request.getfixturevalue(source_fixture)
    analysis = Analysis("new_onboarding_v2", "examples/definitions", con)
    methods = [Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")]
    rows = analysis.run_daily_lift(
        decision_method=methods[0], sensitivity_methods=tuple(methods[1:]), metrics=[metric]
    )
    assert rows

    methods_by_slice: dict[tuple[date, str], set[str]] = {}
    for row in rows:
        assert isinstance(row.ds, date)
        methods_by_slice.setdefault((row.ds, row.group_id), set()).add(row.method)
    assert methods_by_slice
    assert all(methods == {"unadjusted", "cuped"} for methods in methods_by_slice.values())


# Daily and as-of sweeps over a conversion metric, each inverting an exact
# binomial interval per day and arm.
@pytest.mark.slow
def test_native_daily_and_asof_lift_inherit_configured_method_for_metric_object(
    seeded_pre_period_con,
):
    """A caller-provided metric object keeps the source's configured decision method."""
    from increment.semantics.models import ExperimentMetric, MethodSpec

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    configured = ExperimentMetric(
        metric="purchase_rate",
        decision_method=MethodSpec(name="configured"),
    )
    experiment = analysis.experiment.model_copy(
        update={"plan": AnalysisPlan(secondaries=(configured,))}
    )
    analysis = make_analysis_like(analysis, experiment=experiment)
    metric = next(item for item in analysis.metrics if item.name == "purchase_rate").model_copy(
        update={"description": "caller-supplied metric object"}
    )

    for estimates in (
        analysis.run_daily_lift(metrics=[metric]),
        analysis.run_asof_lift(metrics=[metric]),
    ):
        assert estimates
        assert {row.method for row in estimates} == {"configured"}
        assert {row.method_role for row in estimates} == {"decision"}


# A dimensioned daily sweep over a conversion metric, inverting an exact
# binomial interval per day, arm and dimension value.
@pytest.mark.slow
def test_artifact_dimensioned_daily_lift_preserves_estimator_warning_contract(monkeypatch):
    """Artifact-backed daily lift keeps estimator warnings live and exact."""
    from tests.test_unit_day_artifact_facade import _extensions, _native

    _con, native, context, store = _native()
    ref = native.publish_unit_day_artifact(
        store,
        extensions=_extensions(context, "breakout_dimension"),
    )
    analysis = Analysis.from_unit_day_artifact(store, ref, expected_context=context)
    from tests.analysis_factory import _native_source

    source_type = cast(Any, type(_native_source(analysis)))
    original = source_type._dimensioned_moments

    def add_guarded_arm(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        augmented = list(rows)
        for row in rows:
            if (
                row["group_id"] != "treatment"
                or row["ref_y"] <= 0
                or row["ds"] != date(2025, 2, 2)
                or row["country"] != "US"
            ):
                continue
            bad = dict(row)
            bad["group_id"] = "treatment_bad"
            bad["cy2"] = float(row["cy2"]) + 1.0e9
            augmented.append(bad)
            break
        return augmented

    monkeypatch.setattr(source_type, "_dimensioned_moments", add_guarded_arm)

    from increment import analysis as analysis_module
    from increment.breakout import estimates as breakout_estimates

    for module in (analysis_module, breakout_estimates):
        getattr(module, "__warningregistry__", {}).clear()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimates = analysis.run_daily_lift(dimension="country", metrics=["purchase_rate"])

    assert estimates
    estimator_warnings = [
        warning
        for warning in caught
        if isinstance(warning.message, IncrementWarning)
        and warning.message.code == "breakout.estimates.daily_partial_guarded_arms"
    ]
    assert len(estimator_warnings) == 1, [
        (str(warning.message), warning.category, warning.filename, warning.lineno)
        for warning in caught
    ]
    assert estimator_warnings[0].category is IncrementWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="retained estimable arms"):
            analysis.run_daily_lift(dimension="country", metrics=["purchase_rate"])


def test_native_daily_lift_cuped_without_pre_period_refuses_without_fallback(seeded_con):
    from increment import Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_con)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift(
            decision_method=Method(name="cuped", variance_reduction="cuped"),
            metrics=["purchase_rate"],
        )
    assert raised.value.code == "estimation.cuped.covariate_zero_variance"


def test_native_daily_value_readout_does_not_build_pre_period_stats(
    seeded_pre_period_con, monkeypatch
):
    from increment.query import _native_day_source, native_source

    def fail(*args, **kwargs):
        raise AssertionError("value-only daily readout requested pre-period stats")

    monkeypatch.setattr(native_source, "pre_period_stats", fail)
    monkeypatch.setattr(_native_day_source, "pre_period_stats", fail)
    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    assert analysis.run_daily(metrics=["purchase_rate"])
