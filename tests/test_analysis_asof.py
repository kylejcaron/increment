"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta

import narwhals as nw
import numpy as np
import pytest

from increment import Analysis
from increment._frame_moments import _asof_unit_rows
from increment.breakout.estimates import DailyLiftEstimate, DailyMetricValue
from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.encouragement import estimate_encouragement
from increment.frame import MetricSpec
from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    MeanMetric,
    MultiplicitySpec,
    RetentionMetric,
)
from tests.analysis_factory import make_analysis, make_analysis_like
from tests.test_sequential_public_sources import gaussian_plan


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
                    "window_days": 7,
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


def _analysis_with_asof_events(con, *, sequential: bool = False):
    """Build the as-of fixture, optionally with admitted binary outcomes."""
    table_name = "asof_sequential_events" if sequential else "asof_run_events"
    if table_name not in con.list_tables():
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "revenue": None,
                "experiment_id": "asof_run_exp",
            }
            for group_id, units in {
                "control": ["cu1", "cu2"],
                "treatment": ["tu1", "tu2"],
            }.items()
            for uid in units
        ]
        purchases = {
            (datetime(2025, 6, 1, 10, 0, 0), "cu1"): 0.0 if sequential else 8.0,
            (datetime(2025, 6, 1, 10, 0, 0), "cu2"): 0.0 if sequential else 12.0,
            (datetime(2025, 6, 1, 10, 0, 0), "tu1"): 1.0 if sequential else 13.0,
            (datetime(2025, 6, 1, 10, 0, 0), "tu2"): 1.0 if sequential else 15.0,
            (datetime(2025, 6, 2, 10, 0, 0), "cu1"): 0.0 if sequential else 500.0,
            (datetime(2025, 6, 2, 10, 0, 0), "cu2"): 0.0 if sequential else 500.0,
            (datetime(2025, 6, 2, 10, 0, 0), "tu1"): 0.0 if sequential else 500.0,
            (datetime(2025, 6, 2, 10, 0, 0), "tu2"): 0.0 if sequential else 500.0,
            (datetime(2025, 6, 3, 10, 0, 0), "cu1"): 0.0 if sequential else 900.0,
            (datetime(2025, 6, 3, 10, 0, 0), "cu2"): 0.0 if sequential else 900.0,
            (datetime(2025, 6, 3, 10, 0, 0), "tu1"): 0.0 if sequential else 900.0,
            (datetime(2025, 6, 3, 10, 0, 0), "tu2"): 0.0 if sequential else 900.0,
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
        con.create_table(table_name, obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table_name}",
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
                    "type": "conversion" if sequential else "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    **({} if sequential else {"aggregation": "sum"}),
                    "window_days": 1,
                }
            ],
            "experiments": [
                {
                    "name": "asof_run_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "end": "2025-06-03",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )
    return make_analysis(con, defs)


@pytest.fixture(scope="module")
def asof_events_values(con):
    """Plain ``run_asof()`` on the as-of fixture, computed once - the tests
    below assert different facets of one deterministic readout."""
    return _analysis_with_asof_events(con).run_asof()


def test_run_asof_returns_running_total_frozen_after_window_close(asof_events_values):
    """Analysis.run_asof() wires asof_group_summary
    (no breakout dimension) into run_daily and returns one DailyMetricValue
    per (day x arm), whose per-arm mean FREEZES at day 0's value once each
    unit's 1-day window closes - proving the estimation-layer output
    reflects asof_group_summary's masking, not just its schema."""
    results = asof_events_values

    assert len(results) == 6  # 1 metric x 3 days x 2 arms
    for r in results:
        assert isinstance(r, DailyMetricValue)
        assert r.metric == "revenue"
        assert r.n == 2

    by_key = {(r.ds, r.group_id): r for r in results}
    assert set(by_key) == {
        (date(2025, 6, 1), "control"),
        (date(2025, 6, 1), "treatment"),
        (date(2025, 6, 2), "control"),
        (date(2025, 6, 2), "treatment"),
        (date(2025, 6, 3), "control"),
        (date(2025, 6, 3), "treatment"),
    }
    for ds in (date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)):
        assert by_key[(ds, "control")].value.value == pytest.approx(10.0), (
            f"control mean on {ds} should stay frozen at day 0's value (10.0), "
            "not include the huge post-window purchases"
        )
        assert by_key[(ds, "treatment")].value.value == pytest.approx(14.0), (
            f"treatment mean on {ds} should stay frozen at day 0's value "
            "(14.0), not include the huge post-window purchases"
        )


def test_run_asof_differs_from_run_daily_on_the_same_data(con, asof_events_values):
    """Regression: on the SAME underlying panel, run_daily (independent
    per-day snapshot, window_bound_panel applied) only reports the one
    in-window day, while run_asof (running total, its own internal
    masking) reports all three calendar days - the two views are not
    accidentally computing the same thing."""
    analysis = _analysis_with_asof_events(con)

    daily_results = analysis.run_daily()
    asof_results = asof_events_values

    assert {r.ds for r in daily_results} == {date(2025, 6, 1)}
    assert {r.ds for r in asof_results} == {
        date(2025, 6, 1),
        date(2025, 6, 2),
        date(2025, 6, 3),
    }


def _asof_encouragement_design(*, uptake_window_days: int | None = None) -> Encouragement:
    """The declared encouragement design every as-of late fixture runs under.

    ``uptake_window_days=None`` is the ever-took-up form; a value bounds the
    uptake window so a completed-windows gate has a closing edge.
    """
    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked", window_days=uptake_window_days),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="button gates revenue"
        ),
        one_sided=True,
    )


def _analysis_with_asof_encouragement_events(
    con, *, events_table="asof_late_events", sequential: bool = False
):
    """Build a native (DuckDB, ``Analysis.from_definitions``-shape)
    ``Analysis`` with a real :class:`Encouragement` design and uptake
    fact, exercised end to end through :meth:`Analysis.run_asof_lift` -
    mirrors ``_analysis_with_asof_events``'s own idempotent-table
    construction, and ``_guardrail_analysis``'s (tests/
    test_readouts_encouragement.py) declared-Encouragement fixture
    pattern for a native encouragement design.

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
    if events_table not in con.list_tables():
        rng = np.random.default_rng(42)
        control_units = [f"c{i}" for i in range(1, 31)]
        early_clickers = [f"t{i}" for i in range(1, 13)]
        late_clickers = [f"t{i}" for i in range(13, 25)]
        never_clickers = [f"t{i}" for i in range(25, 31)]
        treat_units = early_clickers + late_clickers + never_clickers

        control_rev = {
            u: 0.0 if sequential else float(10.0 + rng.normal(0, 2.0)) for u in control_units
        }
        clicker_rev = {
            u: 1.0 if sequential else float(16.0 + rng.normal(0, 2.0))
            for u in early_clickers + late_clickers
        }
        never_rev = {
            u: 0.0 if sequential else float(10.0 + rng.normal(0, 2.0)) for u in never_clickers
        }
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
        pre_purchase_rows = [
            {
                "user_id": u,
                "ts": datetime(2025, 1, 31, 13, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": (
                    1.0
                    if sequential and u in early_clickers + late_clickers
                    else 0.0
                    if sequential
                    else (10.0 if u in early_clickers + late_clickers else 0.0)
                ),
                "experiment_id": None,
            }
            for u in control_units + treat_units
        ]
        purchase_rows = (
            pre_purchase_rows
            + [
                {
                    "user_id": u,
                    "ts": datetime(2025, 2, 1, 13, 0, 0),
                    "event": "purchase",
                    "group_id": None,
                    "revenue": control_rev[u],
                    "experiment_id": None,
                }
                for u in control_units
            ]
            + [
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
        )
        con.create_table(events_table, obj=exposure_rows + click_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {events_table}",
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
                    "type": "conversion" if sequential else "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    **({} if sequential else {"aggregation": "sum"}),
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
                    "n_pre_periods": 2,
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )

    analysis = make_analysis(
        con,
        defs,
        _design=_asof_encouragement_design(),
    )
    return analysis


def test_analysis_run_native_encouragement_forwards_prior(con):
    from increment.estimation import Normal

    analysis = _analysis_with_asof_encouragement_events(con)
    flat = analysis.run(estimands=("itt",))[0]
    shrunk = analysis.run(estimands=("itt",), prior=Normal(mu=0.0, sigma=0.01))[0]

    assert flat.estimand == shrunk.estimand == "itt"
    assert flat.require_lift().value > 0.0
    assert abs(shrunk.require_lift().value) < abs(flat.require_lift().value)


def test_run_asof_lift_reports_late_trend_for_native_encouragement_design(con):
    """End-to-end native path: a real DuckDB Analysis.from_definitions
    (well, the Analysis.__new__ construction ``_guardrail_analysis``
    already established for this) with a declared Encouragement design
    and a real uptake fact, run through Analysis.run_asof_lift()
    proving the whole chain (_build_asof_panels resolving the uptake
    fact and building its unit_day_panel, asof_group_summary joining it
    to populate sum_d/sum_yd/sum_y2d, run_daily_lift's new design-aware
    branch, and Analysis.run_asof_lift's estimands filtering) is wired
    correctly, not just each piece in isolation.

    Both itt and late are present for every as-of day. late's value
    matches - to the same precision - calling estimate_encouragement
    directly on hand-aggregated moments for the same underlying data,
    proving this is genuinely the same estimator reached through the
    real query pipeline, not a re-derivation. The as-of first stage
    grows from 12/30 compliers (day 0) to 24/30 (day 1 onward, frozen
    thereafter), so late's point estimate roughly halves from day 0 to
    day 1, then freezes - a real as-of LATE *trend*, not a single
    number.
    """
    analysis = _analysis_with_asof_encouragement_events(con)

    results = analysis.run_asof_lift(estimands=("itt", "late"))

    by_day_estimand = defaultdict(list)
    for r in results:
        by_day_estimand[(r.ds, r.estimand)].append(r)

    days = {date(2025, 2, 1), date(2025, 2, 2), date(2025, 2, 3)}
    for ds in days:
        assert len(by_day_estimand[(ds, "itt")]) == 1
        assert by_day_estimand[(ds, "itt")][0].lift is not None

    late_absolute = {
        ds: [r for r in rows if r.value_scale == "absolute"][0]
        for ds in days
        for rows in [by_day_estimand[(ds, "late")]]
        if rows
    }
    assert set(late_absolute) == days, "late is estimable on every as-of day"
    for r in late_absolute.values():
        assert r.lift is not None
        assert r.estimand == "late"

    day0 = late_absolute[date(2025, 2, 1)].require_lift().value
    day1 = late_absolute[date(2025, 2, 2)].require_lift().value
    day2 = late_absolute[date(2025, 2, 3)].require_lift().value
    assert day0 > day1 > 0, "late shrinks as the identified complier population grows"
    assert day1 == pytest.approx(day2), "late freezes once uptake stops accruing (day 1 onward)"

    # Ground truth: the same estimator, called directly on hand-aggregated
    # moments for the identical underlying per-unit data.
    day0_clickers = {f"t{i}" for i in range(1, 13)}
    day1_clickers = day0_clickers | {f"t{i}" for i in range(13, 25)}
    control_units = [f"c{i}" for i in range(1, 31)]
    treat_units = [f"t{i}" for i in range(1, 31)]

    control_rev_frame = (
        con.table("asof_late_events")
        .filter(lambda t: t.event == "purchase")
        .filter(lambda t: t.ts >= datetime(2025, 2, 1))
        .filter(lambda t: t.user_id.isin(control_units))
        .execute()
        .set_index("user_id")["revenue"]
    )
    treat_rev_frame = (
        con.table("asof_late_events")
        .filter(lambda t: t.event == "purchase")
        .filter(lambda t: t.ts >= datetime(2025, 2, 1))
        .filter(lambda t: t.user_id.isin(treat_units))
        .execute()
        .set_index("user_id")["revenue"]
    )

    def moments(rev, group_id, clickers):
        y = rev.to_numpy()
        d = np.array([1.0 if u in clickers else 0.0 for u in rev.index])
        yd = y * d
        return centered_row_from_raw_sums(
            {
                "experiment_id": "s",
                "metric": "revenue",
                "group_id": group_id,
                "n": len(y),
                "sum_y": float(y.sum()),
                "sum_y2": float((y**2).sum()),
                "sum_d": float(d.sum()),
                "sum_yd": float(yd.sum()),
                "sum_y2d": float((y**2 * d).sum()),
            }
        )

    for ds, clickers in ((date(2025, 2, 1), day0_clickers), (date(2025, 2, 2), day1_clickers)):
        expected = [
            r
            for r in estimate_encouragement(
                [MeanMetric(name="revenue", entity="user_id", fact="purchase")],
                [
                    moments(control_rev_frame, "control", set()),
                    moments(treat_rev_frame, "treatment", clickers),
                ],
                _asof_encouragement_design(),
                estimands=("late",),
            ).results
            if r.value_scale == "absolute"
        ][0]
        assert late_absolute[ds].require_lift().value == pytest.approx(
            expected.require_lift().value
        )
        assert late_absolute[ds].require_lift().lb == pytest.approx(expected.require_lift().lb)


def _analysis_with_bounded_asof_encouragement_events(con, *, sequential: bool = False):
    analysis = _analysis_with_asof_encouragement_events(
        con,
        events_table="asof_bounded_late_sequential_events"
        if sequential
        else "asof_bounded_late_events",
        sequential=sequential,
    )
    analysis = make_analysis_like(
        analysis,
        [analysis.metrics[0].model_copy(update={"window_days": 1})],
        design=_asof_encouragement_design(uptake_window_days=1),
    )
    return analysis


def test_native_encouragement_cuped_asof_late_identity(con):
    """Public as-of CUPED rows preserve the LATE/ITT-over-compliance identity."""
    from increment import Method

    analysis = _analysis_with_bounded_asof_encouragement_events(con)

    totals = analysis.run(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["revenue"],
        estimands=("itt", "compliance", "late"),
    )
    total_itt = next(row for row in totals if row.estimand == "itt")
    total_compliance = next(row for row in totals if row.estimand == "compliance")
    total_late = next(
        row for row in totals if row.estimand == "late" and row.value_scale == "absolute"
    )
    assert total_itt.abs_diff is not None
    assert total_compliance.require_lift().value != 0
    assert total_late.require_lift().value == pytest.approx(
        total_itt.abs_diff / total_compliance.require_lift().value
    )

    asof = analysis.run_asof_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["revenue"],
        estimands=("itt", "compliance", "late"),
        completed_windows_only=True,
    )
    cuped_by_date = {ds: [row for row in asof if row.ds == ds] for ds in {r.ds for r in asof}}
    expected_dates = {date(2025, 2, 2), date(2025, 2, 3)}
    assert set(cuped_by_date) == expected_dates
    expected_late = total_itt.abs_diff / total_compliance.require_lift().value
    for rows in cuped_by_date.values():
        itt = next(row for row in rows if row.estimand == "itt" and row.value_scale == "relative")
        compliance = next(
            row for row in rows if row.estimand == "compliance" and row.value_scale == "absolute"
        )
        late = next(row for row in rows if row.estimand == "late" and row.value_scale == "absolute")
        assert itt.lift is not None and compliance.lift is not None and late.lift is not None
        assert itt.require_lift().value == pytest.approx(total_itt.require_lift().value)
        assert compliance.require_lift().value == pytest.approx(
            total_compliance.require_lift().value
        )
        assert late.require_lift().value == pytest.approx(expected_late)


def test_native_encouragement_always_valid_uses_finalized_windows(con):
    from increment.errors import CapabilityError, InvalidRequestError
    from tests.sequential_cases import registered_native

    analysis = registered_native(
        _analysis_with_bounded_asof_encouragement_events(con, sequential=True)
    )
    with pytest.raises(InvalidRequestError) as unfinished:
        analysis.run_asof_lift(estimands=("itt",))
    assert unfinished.value.code == "readout.encouragement.asof_completion"
    with pytest.raises(CapabilityError) as raised:
        analysis.run_asof_lift(completed_windows_only=True, estimands=("late",))
    assert raised.value.code == "sequential.route.unsupported"
    snapshot = analysis.capture_sequential(finalized=True, as_of=date(2025, 3, 1))
    results = analysis.run_asof_lift(completed_windows_only=True, estimands=("itt",))
    assert results and snapshot.records
    assert {row.estimand for row in results} == {"itt"}
    assert all(row.sequential_result is not None for row in results)


def test_run_asof_lift_returns_frozen_per_day_lift_estimates(con):
    """Analysis.run_asof_lift() wires
    asof_group_summary into run_daily_lift and returns one
    DailyLiftEstimate per (day x method x treatment arm), whose lift
    FREEZES at day 0's value once each unit's window closes - the same
    frozen-value property as run_asof, now visible through the
    relative-lift estimator."""
    analysis = _analysis_with_asof_events(con)

    results = analysis.run_asof_lift()

    assert len(results) == 3  # 1 metric x 1 method x 1 treatment arm x 3 days
    for r in results:
        assert isinstance(r, DailyLiftEstimate)
        assert r.metric == "revenue"
        assert r.group_id == "treatment"

    by_day = {r.ds: r for r in results}
    assert set(by_day) == {date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)}
    day0_lift = by_day[date(2025, 6, 1)].require_lift().value
    # Not hand-computed: "unadjusted" applies posterior shrinkage on top of the raw ratio. The property under test is the FREEZE (below), so just confirm direction (treatment > control).
    assert day0_lift > 0
    for ds in (date(2025, 6, 2), date(2025, 6, 3)):
        assert by_day[ds].require_lift().value == pytest.approx(day0_lift), (
            f"lift on {ds} should stay frozen at day 0's value, not drift "
            "from the huge post-window purchases"
        )


def test_run_asof_lift_returns_the_registered_checkpoint(con):
    from tests.sequential_cases import registered_native

    original = _analysis_with_asof_events(con, sequential=True)
    analysis = registered_native(original)
    checkpoint = analysis.capture_sequential(finalized=True, as_of=date(2025, 7, 1))
    results = analysis.run_asof_lift(completed_windows_only=True)
    assert len(results) == 1
    assert results[0].sequential_result is not None
    assert results[0].sequential_result.checkpoint.prefix_id == checkpoint.prefix_id
    assert results[0].ds == date(2025, 7, 1)
    assert {row.inference for row in original.run_asof_lift()} == {"fixed"}


def test_run_asof_and_run_asof_lift_empty_when_no_metrics(con):
    """Both un-broken-out as-of methods return [] when the experiment
    declares no metrics - mirrors run_daily/run_daily_lift's own
    empty-input contract."""
    analysis = _analysis_with_daily_events(con)
    analysis = make_analysis_like(analysis, [], plan=AnalysisPlan())

    assert analysis.run_asof() == []
    assert analysis.run_asof_lift() == []


def test_run_asof_completed_windows_only_raises_for_declared_unbounded_retention(con):
    """Analysis.run_asof(completed_windows_only=True) refuses, naming an
    unbounded RetentionMetric present in the experiment's
    metrics/guardrails - an unbounded band never completes, so the flag
    contradicts it. Checked up front, before any query runs, same guard
    shape as run_daily's own. (Without the flag the same metric is
    accepted - covered separately.)"""
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
        analysis.run_asof(completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    assert raised.value.context["names"] == ("d7_retention",)
    assert raised.value.context["fn_name"] == "run_asof"


def test_run_asof_lift_completed_windows_only_raises_for_declared_unbounded_retention(con):
    """Analysis.run_asof_lift(completed_windows_only=True) refuses, naming
    an unbounded RetentionMetric present in the experiment's
    metrics/guardrails - same up-front check as run_asof, never a
    partial-then-crash loop."""
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
        analysis.run_asof_lift(completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    assert raised.value.context["names"] == ("d7_retention",)
    assert raised.value.context["fn_name"] == "run_asof_lift"


def test_run_asof_and_run_asof_lift_unaffected_without_retention_metric(con):
    """Neither method's new guard fires for an experiment whose declared
    metrics/guardrails contain no RetentionMetric - both still return
    their normal as-of results."""
    analysis = _analysis_with_daily_events(con)

    assert analysis.run_asof() != []
    assert analysis.run_asof_lift() != []


def test_run_asof_metrics_override_excludes_retention_metric(con):
    """Passing `metrics=` scopes the call to non-Retention metrics even
    when the experiment's declared list contains a RetentionMetric."""
    analysis = _analysis_with_daily_events(con)
    retention = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    analysis = make_analysis_like(analysis, [*analysis.metrics, retention])
    non_retention = [m for m in analysis.metrics if not isinstance(m, RetentionMetric)]

    assert analysis.run_asof(metrics=non_retention) != []
    assert analysis.run_asof_lift(metrics=non_retention) != []


@pytest.fixture(scope="module")
def bounded_asof_values(con):
    """``run_asof`` over the bounded d7 retention metric on new_onboarding_v2,
    computed once - shared by the bounded-band as-of tests."""
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = RetentionMetric(
        name="d7_retention_bounded",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    return analysis.run_asof(metrics=[bounded])


def test_run_asof_accepts_bounded_retention_metric(bounded_asof_values):
    """A bounded band unlocks the as-of retention series.

    new_onboarding_v2's u1-u4 are all exposed 2025-01-20; d7_retention
    (band [7, 14)) matures for all four on
    2025-02-03, well before the experiment's 2025-02-15 end. u1/u3/u4
    return inside the band -> y=1; u2 returns on day 6, before the band
    opens -> y=0.
    """
    results = bounded_asof_values
    assert results, "bounded retention must produce an as-of series"
    assert {r.metric for r in results} == {"d7_retention_bounded"}
    assert all(r.ds_basis == "calendar" for r in results)
    # n grows monotonically as cohorts mature, and never shrinks
    for group_id in {r.group_id for r in results}:
        series = sorted((r for r in results if r.group_id == group_id), key=lambda r: r.ds)
        counts = [r.n for r in series]
        assert counts == sorted(counts), "matured population must not shrink"
    by_key = {(r.ds, r.group_id): r for r in results}
    maturity_day = date(2025, 2, 3)
    assert by_key[(maturity_day, "control")].n == 2
    assert by_key[(maturity_day, "treatment")].n == 2
    # control = {u1 (y=1), u2 (y=0)} -> mean 0.5; treatment = {u3, u4} (both y=1) -> mean 1.0
    assert by_key[(maturity_day, "control")].value.value == pytest.approx(0.5)
    assert by_key[(maturity_day, "treatment")].value.value == pytest.approx(1.0)


def test_run_asof_lift_accepts_bounded_retention_metric(con):
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = RetentionMetric(
        name="d7_retention_bounded",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    estimates = analysis.run_asof_lift(metrics=[bounded])
    assert estimates
    assert all(e.ds_basis == "calendar" for e in estimates)


def test_run_asof_accepts_unbounded_retention_metric(con):
    """The as-of view reports an unbounded band [N, inf): the series
    opens at first exposure + threshold and each unit's value ratchets
    on its first return at or past the threshold, never freezing.

    Every unit is exposed on 2025-01-20, so the band opens on 2025-01-27.
    u1/u3/u4 return inside the band -> y=1; u2 returns on day 6, before the
    band opens -> y=0. Final levels are control=0.5, treatment=1.0.
    """
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    unbounded = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )

    results = analysis.run_asof(metrics=[unbounded])
    assert results
    assert min(r.ds for r in results) == date(2025, 1, 27)
    last_day = max(r.ds for r in results)
    final = {r.group_id: r for r in results if r.ds == last_day}
    control_value = final["control"].value
    treatment_value = final["treatment"].value
    assert control_value is not None and treatment_value is not None
    assert control_value.value == pytest.approx(0.5)
    assert treatment_value.value == pytest.approx(1.0)

    estimates = analysis.run_asof_lift(metrics=[unbounded])
    assert estimates
    assert all(e.ds_basis == "calendar" for e in estimates)


def test_run_asof_completed_windows_only_rejects_unbounded_retention_metric(con):
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    unbounded = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof(metrics=[unbounded], completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    assert raised.value.context["names"] == ("d7_retention",)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof_lift(metrics=[unbounded], completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    assert raised.value.context["names"] == ("d7_retention",)


def test_run_asof_lift_unbounded_retention_refuses_sequential_capture(con):
    from increment.errors import CapabilityError
    from tests.sequential_cases import registered_native

    original = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    metric = RetentionMetric(
        name="d7_retention", entity="user_id", fact="page_view", threshold_days=7
    )
    analysis = registered_native(original, metrics=[metric])
    with pytest.raises(CapabilityError) as raised:
        analysis.capture_sequential(finalized=True, as_of=date(2025, 3, 1))
    assert raised.value.code == "sequential.route.unsupported"


def test_run_asof_lift_bounded_retention_has_raw_bernoulli_checkpoint(con):
    from tests.sequential_cases import registered_native

    original = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    metric = RetentionMetric(
        name="d7_retention_bounded", entity="user_id", fact="page_view", threshold_days=(7, 14)
    )
    analysis = registered_native(original, metrics=[metric])
    snapshot = analysis.capture_sequential(finalized=True, as_of=date(2025, 3, 1))
    results = analysis.run_asof_lift(completed_windows_only=True)
    assert results and snapshot.records
    assert all(
        row.sequential_result is not None
        and row.sequential_result.checkpoint.model.law == "bernoulli"
        for row in results
    )


def test_run_asof_completed_windows_only_starts_retention_series_at_band_close(
    con, bounded_asof_values
):
    """The flag threads end to end: on a bounded band [7, 14) with every
    unit exposed 2025-01-20, the default (band-open) gate opens the
    series 2025-01-27 while completed_windows_only=True opens it
    2025-02-03 - and both agree once every band has closed."""
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = RetentionMetric(
        name="d7_retention_bounded",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )

    monitoring = bounded_asof_values
    decision = analysis.run_asof(metrics=[bounded], completed_windows_only=True)

    assert min(r.ds for r in monitoring) == date(2025, 1, 27)
    assert min(r.ds for r in decision) == date(2025, 2, 3)
    last_day = max(r.ds for r in monitoring)
    assert max(r.ds for r in decision) == last_day
    final_m = {r.group_id: r for r in monitoring if r.ds == last_day}
    final_d = {r.group_id: r for r in decision if r.ds == last_day}
    for arm in ("control", "treatment"):
        assert final_m[arm].n == final_d[arm].n
        monitoring_value = final_m[arm].value
        decision_value = final_d[arm].value
        assert monitoring_value is not None and decision_value is not None
        assert monitoring_value.value == pytest.approx(decision_value.value, nan_ok=True)


def test_run_asof_dimension_accepts_bounded_retention_metric(con):
    """The per-segment as-of path takes retention too."""
    analysis = Analysis("new_onboarding_v2", definitions_path="examples/definitions/", con=con)
    bounded = RetentionMetric(
        name="d7_retention_bounded",
        entity="user_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    results = analysis.run_asof(dimension="country", metrics=[bounded])
    assert results
    assert {r.dimension for r in results} == {"country"}


def test_run_asof_dimension_returns_per_day_per_segment_values(con):
    """Analysis.run_asof(dimension="country") returns
    one DailyMetricValue per (day x arm x country) with
    dimension/dimension_value/source populated - the running-total
    counterpart to test_run_daily_dimension_returns_per_day_per_segment_values,
    on the exact same fixture. This metric declares no `window_days`, so
    nothing is ever masked and the as-of running total on the single
    purchase day equals the independent per-day value - the point here is
    the per-segment WIRING (correct segment attribution, correct
    stamping), not the freeze (covered separately below on a windowed
    fixture)."""
    analysis = _analysis_with_country_breakout(con)

    results = analysis.run_asof(dimension="country")

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
    # Exposure day (2025-06-01) has no purchases in its running total yet
    for dimension_value in ("US", "CA"):
        for group_id in ("control", "treatment"):
            assert by_key[(date(2025, 6, 1), dimension_value, group_id)].value is None
    # Same per-segment arm means: a cross-segment mixup would show up here as CA's 20.0 bleeding into US's 10.0.
    assert by_key[(date(2025, 6, 2), "US", "control")].value.value == pytest.approx(10.0)
    assert by_key[(date(2025, 6, 2), "US", "treatment")].value.value == pytest.approx(15.0)
    assert by_key[(date(2025, 6, 2), "CA", "control")].value.value == pytest.approx(20.0)
    assert by_key[(date(2025, 6, 2), "CA", "treatment")].value.value == pytest.approx(15.0)


def test_run_asof_lift_dimension_returns_per_segment_lift_estimates(con):
    """Analysis.run_asof_lift(dimension="country")
    returns one DailyLiftEstimate per (day x method x treatment arm x
    country), with each segment's lift attributed to that segment's OWN
    control arm - US positive (10 -> 15) and CA negative (20 -> 15) on
    this deliberately opposite-signed fixture, so any cross-segment
    contamination flips a sign rather than nudging a number."""
    analysis = _analysis_with_country_breakout(con)
    analysis = make_analysis_like(
        analysis,
        plan=AnalysisPlan(
            alpha=0.02,
            view_multiplicity=MultiplicitySpec(correction="bonferroni"),
        ),
    )
    results = analysis.run_asof_lift(dimension="country")

    # 1 metric x 2 days x 1 method x 1 arm x 2 countries: the exposure day has no purchases in its running total yet, so it comes back as a NaN row per segment instead of vanishing.
    assert len(results) == 4
    for r in results:
        assert isinstance(r, DailyLiftEstimate)
        assert r.metric == "revenue"
        assert r.group_id == "treatment"
        assert r.dimension == "country"
        assert r.source == "events"

    by_key = {(r.ds, r.dimension_value): r for r in results}
    assert by_key[(date(2025, 6, 1), "US")].lift is None
    assert by_key[(date(2025, 6, 1), "CA")].lift is None
    us_lift = by_key[(date(2025, 6, 2), "US")].lift
    ca_lift = by_key[(date(2025, 6, 2), "CA")].lift
    assert us_lift is not None and ca_lift is not None
    assert us_lift.value > 0, "US treatment (15) beats US control (10)"
    assert ca_lift.value < 0, "CA treatment (15) trails CA control (20)"
    live = [
        row for row in results if row.lift is not None and math.isfinite(row.require_lift().value)
    ]
    assert live
    for row in live:
        lift = row.lift
        assert lift is not None
        assert lift.level == pytest.approx(0.99)

    undimensioned = analysis.run_asof_lift()
    undimensioned_live = [
        row
        for row in undimensioned
        if row.lift is not None and math.isfinite(row.require_lift().value)
    ]
    assert undimensioned_live
    for row in undimensioned_live:
        lift = row.lift
        assert lift is not None
        assert lift.level == pytest.approx(0.98)


def test_run_asof_without_dimension_unaffected_by_declared_breakout(con):
    """Both un-dimensioned as-of calls on an experiment that DOES
    declare a `country` breakout are completely unaffected by that
    breakout's existence - dimension/dimension_value/source stay None and
    the reported values collapse across country, matching
    test_run_daily_without_dimension_unaffected_by_declared_breakout's
    numbers on the same fixture (this metric has no window_days, so the
    as-of and independent views coincide here). The
    backward-compatibility guard for the un-dimensioned call shape."""
    analysis = _analysis_with_country_breakout(con)

    values = analysis.run_asof()
    lifts = analysis.run_asof_lift()

    assert len(values) == 4  # 1 metric x 2 days x 2 arms, collapsed
    for r in values:
        assert r.dimension is None
        assert r.dimension_value is None
        assert r.source is None
        assert r.n == 4  # both countries pooled

    by_day_arm = {(r.ds, r.group_id): r for r in values}
    # Exposure day (2025-06-01) has no purchases in its running total yet
    # - a NaN row, not a dropped one.
    assert by_day_arm[(date(2025, 6, 1), "control")].value is None
    assert by_day_arm[(date(2025, 6, 1), "treatment")].value is None
    # control = mean(9, 11, 19, 21) = 15; treatment = mean(14, 16, 14, 16) = 15.
    assert by_day_arm[(date(2025, 6, 2), "control")].value.value == pytest.approx(15.0)
    assert by_day_arm[(date(2025, 6, 2), "treatment")].value.value == pytest.approx(15.0)

    # 1 metric x 2 days x 1 method x 1 arm: the exposure day is a NaN row.
    assert len(lifts) == 2
    for r in lifts:
        assert r.dimension is None
        assert r.dimension_value is None
        assert r.source is None
    by_day_lift = {r.ds: r for r in lifts}
    assert by_day_lift[date(2025, 6, 1)].lift is None
    assert by_day_lift[date(2025, 6, 2)].lift is not None


def _analysis_with_asof_country_breakout(con):
    """``_analysis_with_country_breakout``'s per-country design crossed with
    ``_analysis_with_asof_events``'s windowing: a ``country``
    breakout AND a ``window_days=1`` metric whose day-1/day-2 purchases
    are ~40-60x day 0's.

    Every unit is exposed on day 0 (2025-06-01) and the 1-day window
    closes at 2025-06-02 (``ds < window_end``), so day 0 is the only
    in-window day and the huge later purchases must be masked to 0 in the
    running sum. Day-0 arm means keep the opposite-signed-lift design:
    US control=10 / treatment=14 (positive lift), CA control=20 /
    treatment=14 (negative lift). Neither
    ``_analysis_with_country_breakout`` (no ``window_days``, so nothing
    ever freezes) nor ``_analysis_with_asof_events`` (no breakout)
    can express the per-segment freeze on its own.
    """
    if "asof_breakout_events" not in con.list_tables():
        exposure_groups = {
            "US": {"control": ["ku1", "ku2"], "treatment": ["ku3", "ku4"]},
            "CA": {"control": ["ku5", "ku6"], "treatment": ["ku7", "ku8"]},
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
                "experiment_id": "asof_breakout_exp",
            }
            for country, groups in exposure_groups.items()
            for group_id, units in groups.items()
            for uid in units
        ]
        # Day 0 (in window): US control 8/12 -> 10, US treatment 13/15 -> 14,
        # CA control 19/21 -> 20, CA treatment 13/15 -> 14.
        day0 = {
            "ku1": 8.0,
            "ku2": 12.0,
            "ku3": 13.0,
            "ku4": 15.0,
            "ku5": 19.0,
            "ku6": 21.0,
            "ku7": 13.0,
            "ku8": 15.0,
        }
        purchase_rows = [
            {
                "user_id": uid,
                "ts": ts,
                "event": "purchase",
                "group_id": None,
                "country_code": country_by_unit[uid],
                "revenue": amount,
                "experiment_id": None,
            }
            for ts, amounts in (
                (datetime(2025, 6, 1, 10, 0, 0), day0),
                # Post-window, deliberately huge: masked to 0 in the
                # running sum, per segment, or every assertion below moves.
                (datetime(2025, 6, 2, 10, 0, 0), dict.fromkeys(day0, 500.0)),
                (datetime(2025, 6, 3, 10, 0, 0), dict.fromkeys(day0, 900.0)),
            )
            for uid, amount in amounts.items()
        ]
        con.create_table("asof_breakout_events", obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM asof_breakout_events",
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
                    "window_days": 1,
                }
            ],
            "experiments": [
                {
                    "name": "asof_breakout_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-06-01",
                    "end": "2025-06-03",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue"]},
                    "breakouts": [{"property": "country", "source": "events"}],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_asof_dimension_freezes_each_segment_after_window_close(con):
    """The as-of freeze property holds PER SEGMENT: each segment's
    running total holds steady at its own last in-window value once its
    units' windows close, rather than collapsing to nothing or absorbing
    the huge post-window purchases - the per-segment counterpart of
    test_run_asof_returns_running_total_frozen_after_window_close.

    The segments also stay independent while frozen: US control holds at
    10 while CA control holds at 20, so a cross-segment mixup would show
    as both converging on the pooled 15.
    """
    analysis = _analysis_with_asof_country_breakout(con)

    results = analysis.run_asof(dimension="country")

    assert len(results) == 12  # 1 metric x 3 days x 2 arms x 2 countries
    by_key = {(r.ds, r.dimension_value, r.group_id): r for r in results}
    frozen = {
        ("US", "control"): 10.0,
        ("US", "treatment"): 14.0,
        ("CA", "control"): 20.0,
        ("CA", "treatment"): 14.0,
    }
    for ds in (date(2025, 6, 1), date(2025, 6, 2), date(2025, 6, 3)):
        for (country, arm), expected in frozen.items():
            row = by_key[(ds, country, arm)]
            assert row.n == 2, f"{country}/{arm} on {ds}"
            assert row.dimension == "country"
            assert row.source == "events"
            assert row.value.value == pytest.approx(expected), (
                f"{country}/{arm} on {ds} should stay frozen at day 0's value "
                f"({expected}), not include the huge post-window purchases"
            )


def test_run_asof_lift_dimension_freezes_each_segment_after_window_close(con):
    """The same per-segment freeze, seen through the relative-lift
    estimator: each segment's as-of lift holds at its day-0 value for
    every later day, and the two segments keep their opposite signs (US
    treatment 14 > control 10; CA treatment 14 < control 20)."""
    analysis = _analysis_with_asof_country_breakout(con)

    results = analysis.run_asof_lift(dimension="country")

    assert len(results) == 6  # 1 metric x 3 days x 1 method x 1 arm x 2 countries
    by_key = {(r.ds, r.dimension_value): r for r in results}
    for country, sign in (("US", 1.0), ("CA", -1.0)):
        day0 = by_key[(date(2025, 6, 1), country)].require_lift().value
        assert day0 * sign > 0, f"{country} lift should point {'up' if sign > 0 else 'down'}"
        for ds in (date(2025, 6, 2), date(2025, 6, 3)):
            row = by_key[(ds, country)]
            assert row.dimension == "country"
            assert row.source == "events"
            assert row.require_lift().value == pytest.approx(day0), (
                f"{country} lift on {ds} should stay frozen at day 0's value, "
                "not drift from the huge post-window purchases"
            )


def test_run_asof_dimension_completed_windows_only_raises_for_unbounded_retention(con):
    """Both dimensioned as-of calls reject completed_windows_only=True
    with an unbounded RetentionMetric in the effective metric list -
    same up-front contradiction guard as their un-dimensioned
    counterparts: an unbounded band never completes, per segment no more
    than pooled."""
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
        analysis.run_asof(dimension="country", completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    assert raised.value.context["names"] == ("d7_retention",)
    assert raised.value.context["fn_name"] == "run_asof"
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof_lift(dimension="country", completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    assert raised.value.context["names"] == ("d7_retention",)
    assert raised.value.context["fn_name"] == "run_asof_lift"


def test_run_asof_dimension_contradiction_is_publicly_refused(con):
    """The completed-window contradiction is exposed by both public methods."""
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
        analysis.run_asof(dimension="country", completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof_lift(dimension="country", completed_windows_only=True)
    assert raised.value.code == "breakout.retention.completion"


def test_run_asof_unknown_dimension_raises_naming_declared_properties(con):
    """A `dimension` matching no declared breakout is a caller error (most
    likely a typo), not a legitimate empty result - both methods raise
    naming the experiment's actual declared breakout properties, and do so
    before any query runs."""
    analysis = _analysis_with_country_breakout(con)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof(dimension="platform")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["dimension"] == "platform"
    assert raised.value.context["declared"] == ("country",)
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof_lift(dimension="platform")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["dimension"] == "platform"
    assert raised.value.context["declared"] == ("country",)


def test_run_asof_unknown_dimension_raises_when_no_breakouts_declared(con):
    """An experiment with NO declared breakouts still raises for any
    `dimension`, naming an empty list of declared properties, rather than
    silently returning []."""
    analysis = _analysis_with_asof_events(con)

    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof(dimension="country")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["dimension"] == "country"
    assert raised.value.context["declared"] == ()
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_asof_lift(dimension="country")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["dimension"] == "country"
    assert raised.value.context["declared"] == ()


def test_run_asof_dimension_allows_metric_not_declared_on_experiment(con):
    """Unlike the daily dimensioned path, as-of operations build a fresh
    panel for each requested metric, so undeclared metrics are accepted."""
    analysis = _analysis_with_country_breakout(con)
    undeclared = MeanMetric(
        name="revenue_undeclared",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
    )

    results = analysis.run_asof(dimension="country", metrics=[undeclared])

    assert len(results) == 8  # same shape as the declared metric's own call
    assert {r.metric for r in results} == {"revenue_undeclared"}
    assert {r.dimension_value for r in results} == {"US", "CA"}


def test_run_asof_returns_nan_row_for_unestimable_day(con):
    """run_asof inherits run_daily's dense contract: a day whose
    running total is still zero (nobody has converted yet) comes back as a
    NaN row rather than vanishing, so the series stays continuous.

    This fixture's first day (2025-06-01) has zero revenue in both arms;
    before this change it produced 2 dropped rows and 2 UserWarnings.
    """
    analysis = _analysis_with_country_breakout(con)

    # simplefilter("error") isn't used here because a con-backed call routes through ibis's DuckDB backend, which emits an unrelated DeprecationWarning on every query, and erroring on it crashes ibis internals.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = analysis.run_asof()
    assert not any(issubclass(w.category, UserWarning) for w in record)

    assert results, "expected a dense series, not an empty result"
    nan_rows = [r for r in results if r.value is None]
    assert nan_rows, "2025-06-01 has zero signal -- expected NaN rows for it"
    assert all(r.n == 4 for r in nan_rows), "n is preserved on a NaN row (D3)"


def test_run_asof_lift_returns_nan_row_for_unestimable_day(con):
    """Same dense contract through the relative-lift estimator. This
    fixture produced 3 UserWarnings on this path before the change."""
    analysis = _analysis_with_country_breakout(con)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = analysis.run_asof_lift()
    assert not any(issubclass(w.category, UserWarning) for w in record)

    assert results
    assert any(r.lift is None for r in results)


def test_run_asof_dimension_drops_no_rows_to_sparsity(con):
    """The dense contract holds per segment: a (day, segment, arm) with no
    signal gets a NaN row, so the result has exactly as many rows as the
    underlying panel - nothing is lost to sparsity.

    This fixture's panel is 2 countries x 2 days (2025-06-01, 2025-06-02)
    x 2 arms = 8 rows, and day 1 has zero revenue everywhere. Before this
    change the zero-revenue rows were dropped; now they come back NaN, so
    the expected count is the full 8.

    Note the equal-per-segment shape here is a property of THIS fixture
    (all 8 units enrol on the same day), not a general guarantee - a
    segment whose units enrol later legitimately has fewer panel days,
    since `unit_day_panel` starts each unit's rows at its own first
    exposure. The invariant under test is "no row dropped", not "all
    segments equal".
    """
    analysis = _analysis_with_country_breakout(con)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = analysis.run_asof(dimension="country")
    assert not any(issubclass(w.category, UserWarning) for w in record)

    assert len(results) == 8, f"expected the full dense panel, got {len(results)}"
    assert {r.dimension_value for r in results} == {"US", "CA"}
    assert any(r.value is None for r in results), (
        "day 1 has zero revenue -- expected NaN rows, not dropped ones"
    )


def _analysis_with_daily_ratio_metric(con):
    """Build an ``Analysis`` (no breakout) whose only metric is
    a :class:`RatioMetric` (``revenue_per_session`` = sum(purchase.revenue)
    / count(session_end)), with 1 day of events for 2 control + 2
    treatment units - the real-DuckDB end-to-end counterpart to
    ``tests/breakout/test_daily.py``'s ``TestRunDailyRatioMetric`` (which
    exercises ``run_daily`` directly on hand-built rows): this instead
    drives ``Analysis.run_daily()`` so the fix (computing a
    ratio metric's per-day value as ``sum_y / sum_den`` via
    ``VARIANCE_MODELS`` dispatch, not the old ``sum_y / n`` mean) is
    covered through the whole ``daily_group_summary`` pipeline, not just
    ``run_daily`` in isolation.

    Day 2025-07-01 revenue (numerator) and session_end (denominator)
    counts, by design distinct from a plain per-unit mean:
      control: du1 revenue=10 session_end=2; du2 revenue=20 session_end=3
        -> sum_y=30, sum_den=5, ratio=6.0 (per-unit revenue mean would be 15.0)
      treatment: du3 revenue=15 session_end=1; du4 revenue=45 session_end=3
        -> sum_y=60, sum_den=4, ratio=15.0 (per-unit revenue mean would be 30.0)
    """
    if "daily_ratio_events" not in con.list_tables():
        exposure_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 7, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": group_id,
                "revenue": None,
                "experiment_id": "daily_ratio_exp",
            }
            for group_id, units in {
                "control": ["ru1", "ru2"],
                "treatment": ["ru3", "ru4"],
            }.items()
            for uid in units
        ]
        purchase_amounts = {"ru1": 10.0, "ru2": 20.0, "ru3": 15.0, "ru4": 45.0}
        purchase_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 7, 1, 10, 0, 0),
                "event": "purchase",
                "group_id": None,
                "revenue": amount,
                "experiment_id": None,
            }
            for uid, amount in purchase_amounts.items()
        ]
        session_end_counts = {"ru1": 2, "ru2": 3, "ru3": 1, "ru4": 3}
        session_end_rows = [
            {
                "user_id": uid,
                "ts": datetime(2025, 7, 1, 10, i, 0),
                "event": "session_end",
                "group_id": None,
                "revenue": None,
                "experiment_id": None,
            }
            for uid, count in session_end_counts.items()
            for i in range(count)
        ]
        con.create_table("daily_ratio_events", obj=exposure_rows + purchase_rows + session_end_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM daily_ratio_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
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
                    "entity": "user_id",
                    "numerator": {"fact": "purchase", "aggregation": "sum"},
                    "denominator": {"fact": "session_end", "aggregation": "count"},
                }
            ],
            "experiments": [
                {
                    "name": "daily_ratio_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-07-01",
                    "end": "2025-07-01",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue_per_session"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_daily_ratio_metric_value_is_sum_ratio_not_per_unit_mean(con):
    """Analysis.run_daily() computes a RatioMetric's per-day
    value as sum(numerator) / sum(denominator), not the raw numerator's
    per-unit mean - end-to-end through the real DuckDB pipeline
    (daily_group_summary -> run_daily -> VARIANCE_MODELS dispatch)."""
    analysis = _analysis_with_daily_ratio_metric(con)

    results = analysis.run_daily()

    assert len(results) == 2  # 1 metric x 1 day x 2 arms
    by_group = {r.group_id: r for r in results}
    assert set(by_group) == {"control", "treatment"}

    for r in results:
        assert r.ds == date(2025, 7, 1)
        assert r.metric == "revenue_per_session"
        assert r.n == 2

    # control: sum_y=30, sum_den=5 -> ratio=6.0 (per-unit mean would be 15.0)
    assert by_group["control"].value.value == pytest.approx(6.0)
    # treatment: sum_y=60, sum_den=4 -> ratio=15.0 (per-unit mean would be 30.0)
    assert by_group["treatment"].value.value == pytest.approx(15.0)


def _analysis_with_missing_control_day(con):
    """Build an ``Analysis`` (no breakout) where day 1
    (2025-06-01) has ONLY treatment-arm units/events - the control-arm
    units aren't first exposed until day 2 (2025-06-02), so day 1's dense
    unit-day panel has zero control rows at all - to exercise
    ``run_daily_lift(con)``'s per-day "no control arm, skip the day"
    handling through the real ``Analysis`` pipeline against
    DuckDB (the day-axis analogue of
    ``_analysis_with_missing_control_segment`` above)."""
    if "daily_missing_control_events" not in con.list_tables():
        exposure_rows = [
            # Treatment: first exposed day 1 - panel covers day 1 + day 2.
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 1, 9, 0, 0),
                "event": "page_view",
                "group_id": "treatment",
                "revenue": None,
                "experiment_id": "daily_missing_control_exp",
            }
            for uid in ["mcu1", "mcu2"]
        ] + [
            # Control: not first exposed until day 2, so day 1 has zero control rows (not merely zero-valued ones).
            {
                "user_id": uid,
                "ts": datetime(2025, 6, 2, 9, 0, 0),
                "event": "page_view",
                "group_id": "control",
                "revenue": None,
                "experiment_id": "daily_missing_control_exp",
            }
            for uid in ["mcu3", "mcu4"]
        ]
        purchases = {
            # Day 1: treatment only (mean=12).
            (datetime(2025, 6, 1, 10, 0, 0), "mcu1"): 10.0,
            (datetime(2025, 6, 1, 10, 0, 0), "mcu2"): 14.0,
            # Day 2: both arms - treatment (mean=21) > control (mean=11).
            (datetime(2025, 6, 2, 10, 0, 0), "mcu1"): 20.0,
            (datetime(2025, 6, 2, 10, 0, 0), "mcu2"): 22.0,
            (datetime(2025, 6, 2, 10, 0, 0), "mcu3"): 10.0,
            (datetime(2025, 6, 2, 10, 0, 0), "mcu4"): 12.0,
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
        con.create_table("daily_missing_control_events", obj=exposure_rows + purchase_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM daily_missing_control_events",
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
                    "name": "daily_missing_control_exp",
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


def test_run_daily_lift_returns_nan_for_day_missing_control_arm(con):
    """A day with no control-arm units/events at all (day 1, treatment-only)
    must not abort the whole ``run_daily_lift(con)`` call - day 2's
    ``DailyLiftEstimate`` is still returned correctly, and day 1 comes back
    as a NaN-valued lift rather than raising or vanishing - the real-DuckDB
    integration counterpart to the unit-level
    ``TestRunDailyLiftNaNRowsForUnestimableSlices.test_day_missing_control_arm_returns_nan_rows``
    in ``tests/breakout/test_daily.py``."""
    analysis = _analysis_with_missing_control_day(con)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        results = analysis.run_daily_lift()
    assert not any(issubclass(w.category, UserWarning) for w in record)

    assert results != []
    by_day = {r.ds: r for r in results}
    assert set(by_day) == {date(2025, 6, 1), date(2025, 6, 2)}

    day1 = by_day[date(2025, 6, 1)]
    assert isinstance(day1, DailyLiftEstimate)
    assert day1.metric == "revenue"
    assert day1.group_id == "treatment"
    assert day1.lift is None

    day2 = by_day[date(2025, 6, 2)]
    assert isinstance(day2, DailyLiftEstimate)
    assert day2.metric == "revenue"
    assert day2.group_id == "treatment"
    assert day2.require_lift().value > 0, (
        "day 2 treatment (21) > control (11) -- expected positive lift"
    )


def _analysis_with_daily_ratio_metric_window_dilution(con):
    """Build an ``Analysis`` (no breakout) whose only metric is
    a :class:`RatioMetric` (``revenue_per_session`` = sum(purchase.revenue)
    / count(session_end)), sized and shaped to reproduce the real
    dilution investigation for a ratio metric specifically: 36 units
    enroll 6/day (3 control + 3 treatment) across 6 calendar day-offsets,
    ``window_days=3``.

    Every unit generates a constant $20.00 purchase (numerator) on EACH
    day of its OWN 3-day window and nothing else. Every unit ALSO
    generates exactly 1 session_end (denominator) EVERY calendar day
    from its own enrollment through ``experiment.end`` (2025-08-15) -
    ongoing product usage that outlives the short revenue-attribution
    window, unlike the numerator. The TRUE in-window ratio is a flat
    20.0 (= $20 / 1 session) on every calendar day; mixing in
    out-of-window units' ongoing sessions (inflating the denominator
    without inflating the numerator) dilutes the reported ratio toward 0
    - the RatioMetric analogue of
    ``test_window_bound_panel_removes_dilution_at_scale`` in
    ``tests/query/test_builders.py``.
    """
    n_enroll_days, per_day, window_days, end_offset = 6, 6, 3, 14
    start = datetime(2025, 8, 1)

    table_name = "daily_ratio_dilution_events"
    if table_name not in con.list_tables():
        exposure_rows = []
        purchase_rows = []
        session_rows = []
        for d in range(n_enroll_days):
            fe_ts = start + timedelta(days=d, hours=9)
            for i in range(per_day):
                uid = f"wu{d}_{i}"
                group_id = "control" if i < per_day // 2 else "treatment"
                exposure_rows.append(
                    {
                        "user_id": uid,
                        "ts": fe_ts,
                        "event": "page_view",
                        "group_id": group_id,
                        "revenue": None,
                        "experiment_id": "daily_ratio_dilution_exp",
                    }
                )
                for w in range(window_days):
                    purchase_rows.append(
                        {
                            "user_id": uid,
                            "ts": start + timedelta(days=d + w, hours=10),
                            "event": "purchase",
                            "group_id": None,
                            "revenue": 20.0,
                            "experiment_id": None,
                        }
                    )
                for off in range(0, end_offset - d + 1):
                    session_rows.append(
                        {
                            "user_id": uid,
                            "ts": start + timedelta(days=d + off, hours=11),
                            "event": "session_end",
                            "group_id": None,
                            "revenue": None,
                            "experiment_id": None,
                        }
                    )
        con.create_table(table_name, obj=exposure_rows + purchase_rows + session_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": f"SELECT * FROM {table_name}",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
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
                    "entity": "user_id",
                    "numerator": {
                        "fact": "purchase",
                        "aggregation": "sum",
                        "window_days": window_days,
                    },
                    "denominator": {
                        "fact": "session_end",
                        "aggregation": "count",
                        "window_days": window_days,
                    },
                }
            ],
            "experiments": [
                {
                    "name": "daily_ratio_dilution_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-08-01",
                    "end": "2025-08-15",
                    "control_group": "control",
                    "plan": {"secondaries": ["revenue_per_session"]},
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_run_daily_ratio_metric_window_bound_removes_dilution(con):
    """Regression (dilution bug), RatioMetric via the full facade:
    Analysis.run_daily() produces an undiluted per-day ratio
    once both the numerator panel and the denominator panel are
    window-bounded before reducing. This is an end-to-end check of the
    final output, not an isolation of the denominator bind's marginal
    effect - for this fixture (``experiment.end`` is set), both dense
    panels share an identical ``(unit_id, ds)`` key domain once the
    numerator panel is bound, so the explicit denominator bind is
    provably redundant here (see the matching comment at each
    ``window_bound_panel(den_panel, ...)`` call site in
    ``increment/analysis.py``); it's still applied everywhere per that
    seam's contract, and stops being redundant when ``experiment.end``
    is ``None``."""
    analysis = _analysis_with_daily_ratio_metric_window_dilution(con)

    results = analysis.run_daily()

    n_enroll_days, per_day, window_days = 6, 6, 3

    def in_window_units(day_offset: int) -> int:
        valid_days = [d for d in range(n_enroll_days) if d <= day_offset <= d + window_days - 1]
        return len(valid_days) * per_day

    mid_offset = 4
    expected_in_window = in_window_units(mid_offset)
    assert 0 < expected_in_window < n_enroll_days * per_day, "test fixture drifted"
    mid_date = date(2025, 8, 1) + timedelta(days=mid_offset)

    by_key = {(r.ds, r.group_id): r for r in results}
    for group_id in ("control", "treatment"):
        r = by_key[(mid_date, group_id)]
        assert r.metric == "revenue_per_session"
        assert r.n == expected_in_window // 2  # split evenly control/treatment
        # Undiluted: $20 revenue / 1 session per in-window unit, exactly -
        # not diluted toward ~12.0 by out-of-window units' ongoing sessions.
        assert r.value.value == pytest.approx(20.0)

    # Well past every unit's window close (enrollment day 5 + window 3 closes day 8): no in-window units remain, so the day has no reported row at all, not a diluted near-zero ratio.
    tail_offset = 11
    assert in_window_units(tail_offset) == 0
    tail_date = date(2025, 8, 1) + timedelta(days=tail_offset)
    assert tail_date not in {r.ds for r in results}


def test_native_asof_lift_supports_cuped(seeded_pre_period_con):
    from increment import Analysis, Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    rows = analysis.run_asof_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["purchase_rate"],
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert any(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


@pytest.mark.parametrize("correction", ["none", "bonferroni"])
def test_native_dimensioned_asof_lift_supports_cuped(seeded_pre_period_con, correction):
    from increment import Analysis, Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    analysis = make_analysis_like(
        analysis,
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction=correction)),
    )
    rows = analysis.run_asof_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["purchase_rate"],
        dimension="country",
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert {row.dimension for row in rows} == {"country"}
    assert {row.dimension_value for row in rows} == {"US", "GB", "DE", "CA"}
    assert any(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


def test_native_asof_retention_lift_supports_cuped_and_completed_windows(
    retention_cuped_con,
):
    from increment import Analysis, Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", retention_cuped_con)
    method = [Method(name="cuped", variance_reduction="cuped")]
    monitoring = analysis.run_asof_lift(decision_method=method[0], metrics=["d7_retention"])
    completed = analysis.run_asof_lift(
        decision_method=method[0],
        metrics=["d7_retention"],
        completed_windows_only=True,
    )
    assert monitoring and completed
    assert {row.method for row in monitoring} == {"cuped"}
    assert {row.method for row in completed} == {"cuped"}
    assert all(row.ds_basis == "calendar" for row in [*monitoring, *completed])
    assert all(
        row.lift is not None and math.isfinite(row.require_lift().value) for row in completed
    )
    assert all(
        row.lift is not None and math.isfinite(row.require_lift().value) for row in monitoring
    )
    completed_days = [row.ds for row in completed]
    monitoring_days = [row.ds for row in monitoring]
    assert all(isinstance(day, date) for day in [*completed_days, *monitoring_days])
    assert min(day for day in completed_days if isinstance(day, date)) >= min(
        day for day in monitoring_days if isinstance(day, date)
    )


@pytest.mark.slow
@pytest.mark.parametrize("correction", ["none", "bonferroni"])
def test_native_dimensioned_asof_retention_lift_supports_cuped(retention_cuped_con, correction):
    from increment import Analysis, Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", retention_cuped_con)
    analysis = make_analysis_like(
        analysis,
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction=correction)),
    )
    rows = analysis.run_asof_lift(
        decision_method=Method(name="cuped", variance_reduction="cuped"),
        metrics=["d7_retention"],
        dimension="country",
    )
    assert rows
    assert {row.method for row in rows} == {"cuped"}
    assert {row.dimension for row in rows} == {"country"}
    assert {row.dimension_value for row in rows} == {"US", "GB", "DE", "CA"}
    assert all(row.ds_basis == "calendar" for row in rows)
    assert all(row.lift is not None and math.isfinite(row.require_lift().value) for row in rows)


def test_native_asof_value_readout_does_not_build_pre_period_stats(
    seeded_pre_period_con, monkeypatch
):
    from increment.query import native_source

    def fail(*args, **kwargs):
        raise AssertionError("value-only as-of readout requested pre-period stats")

    monkeypatch.setattr(native_source, "pre_period_stats", fail)
    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    assert analysis.run_asof(metrics=["purchase_rate"])


@pytest.mark.parametrize(
    ("metric", "source_fixture"),
    [
        pytest.param("avg_session_duration", "seeded_pre_period_con", id="mean"),
        pytest.param(
            "purchase_rate",
            "seeded_pre_period_con",
            id="conversion",
            # An exact binomial interval is inverted per day and arm.
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "d7_retention",
            "retention_cuped_con",
            id="retention",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_native_asof_lift_mixed_methods_preserves_matrix(request, metric, source_fixture):
    from increment import Method

    con = request.getfixturevalue(source_fixture)
    analysis = Analysis("new_onboarding_v2", "examples/definitions", con)
    methods = [Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")]
    rows = analysis.run_asof_lift(
        decision_method=methods[0], sensitivity_methods=tuple(methods[1:]), metrics=[metric]
    )
    assert rows

    methods_by_slice: dict[tuple[date, str], set[str]] = {}
    for row in rows:
        assert isinstance(row.ds, date)
        methods_by_slice.setdefault((row.ds, row.group_id), set()).add(row.method)
    assert methods_by_slice
    assert all(methods == {"unadjusted", "cuped"} for methods in methods_by_slice.values())


def test_native_asof_lift_segmented_bh_refuses_before_moments(seeded_pre_period_con):
    from increment import Method

    analysis = Analysis("new_onboarding_v2", "examples/definitions", seeded_pre_period_con)
    analysis = make_analysis_like(
        analysis,
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="bh")),
    )

    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run_asof_lift(
            decision_method=Method(name="unadjusted"),
            metrics=["purchase_rate"],
            dimension="country",
        )
    assert raised.value.code == "facade.analysis.bh_segmented_asof"


def test_native_asof_lift_preserves_ratio_cuped_refusal(con):
    from increment import Method

    analysis = _analysis_with_daily_ratio_metric(con)
    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run_asof_lift(
            decision_method=Method(name="cuped", variance_reduction="cuped"),
        )
    assert raised.value.code == "estimation.engine.ratio.cuped"


def test_asof_unit_rows_emits_only_observed_rows_after_window_freezes():
    """A mid-series window freezes its state; a unit with no observed day emits nothing."""
    panel = nw.from_dict(
        {
            "ds": [
                datetime(2025, 5, 31),
                datetime(2025, 6, 1),
                datetime(2025, 6, 2),
                datetime(2025, 6, 3),
            ],
            "unit_id": ["u2", "u1", "u1", "u1"],
            "group_id": ["control", "control", "control", "control"],
            "revenue": [99.0, 1.0, 2.0, 100.0],
        },
        backend="polars",
    )
    exposure = nw.from_dict(
        {
            "unit_id": ["u1", "u2"],
            "__exposure__": [datetime(2025, 6, 1), datetime(2025, 6, 1)],
        },
        backend="polars",
    )
    spec = MetricSpec(name="revenue", type="mean", window_days=2)
    metric = MeanMetric(
        name="revenue",
        entity="unit_id",
        fact="revenue",
        aggregation="sum",
        window_days=2,
    )

    rows = _asof_unit_rows(
        panel,
        spec=spec,
        metric=metric,
        by=(),
        exposure=exposure,
        first_exposure=None,
        uptake=None,
        uptake_window_days=None,
        completed_windows_only=False,
    ).iter_rows(named=True)

    assert [(row["unit_id"], row["ds"], row["y"]) for row in rows] == [
        ("u1", datetime(2025, 6, 1), 1.0),
        ("u1", datetime(2025, 6, 2), 3.0),
        ("u1", datetime(2025, 6, 3), 3.0),
    ]


def test_asof_lift_refuses_a_prior_under_sequential_inference() -> None:
    """prior and sequential inference are mutually exclusive on every readout:
    run() refuses, so the as-of series refuses too."""
    import datetime as dt
    import warnings

    import pandas as pd
    import pytest

    from increment.analysis import Analysis
    from increment.errors import CodedError
    from increment.estimation.inference import Normal

    rows = []
    for i in range(60):
        group = "treatment" if i % 2 else "control"
        for day in range(3):
            rows.append(
                {
                    "user_id": f"u{i}",
                    "variant": group,
                    "ds": dt.date(2024, 1, 1) + dt.timedelta(days=day),
                    "revenue": int(group == "treatment"),
                }
            )
    analysis = Analysis.from_unit_panel(
        pd.DataFrame(rows),
        unit="user_id",
        group="variant",
        date="ds",
        control="control",
        metrics=[MetricSpec(name="revenue", type="conversion")],
        plan=gaussian_plan(
            [MetricSpec(name="revenue", type="conversion")],
            law="bernoulli",
            date="ds",
            exposure_date=None,
        ),
    )
    prior = Normal(mu=0.0, sigma=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(CodedError) as run_refusal:
            analysis.run(prior=prior)
        assert run_refusal.value.code == "sequential.route.unsupported"
        with pytest.raises(CodedError) as asof_refusal:
            analysis.run_asof_lift(prior=prior)
    assert asof_refusal.value.code == "sequential.route.unsupported"


def test_asof_lift_native_refuses_effect_prior_before_capture(con):
    from increment.errors import CapabilityError
    from increment.estimation.inference import Normal
    from tests.sequential_cases import registered_native

    analysis = registered_native(_analysis_with_asof_events(con, sequential=True))
    with pytest.raises(CapabilityError) as raised:
        analysis.run_asof_lift(prior=Normal(mu=0, sigma=0.1))
    assert raised.value.code == "sequential.route.unsupported"
