"""Step 2: Pre-period query test — pre-period events land in x, post-period in y.

Verifies the strict exposure boundary:
  - Events before ``first_exposure_ts`` → ``x`` (pre-period covariate)
  - Events strictly after ``first_exposure_ts`` → ``y`` (outcome)
  - An event at EXACTLY ``first_exposure_ts`` is excluded from both - it is
    the exposure row itself whenever its fact happens to match the metric's
    fact, and must never count as either a pre-period or post-period
    occurrence (``increment/query/builders.py``'s ``unit_day_panel`` join).
"""

from __future__ import annotations

import datetime as dt

import pytest

from increment.query.builders import (
    first_exposures,
    metric_events,
    pre_period_stats,
    unit_day_spine_stats,
    unit_totals,
)
from increment.query.schemas import UNIT_TOTALS
from increment.semantics.models import AnalysisPlan, Experiment
from tests.query.conftest import _table

# Fixtures


@pytest.fixture
def cuped_experiment():
    return Experiment(
        name="exp_cuped",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 10),
        control_group="control",
        exposure="test_exposure",
        n_pre_periods=3,  # 3-day pre-period lookback
        plan=AnalysisPlan(),
    )


@pytest.fixture(scope="session")
def exposure_events_cuped(con):
    """Exposure events for the CUPED experiment.

    u1: exposed Aug 3 10:00 — pre events Jul 31 - Aug 3 10:00
    u2: exposed Aug 4 10:00 — pre events Aug 1 - Aug 4 10:00
    u3: exposed Aug 5 10:00 — pre events Aug 2 - Aug 5 10:00
    """
    return _table(
        con,
        [
            {
                "unit_id": "u1",
                "experiment_id": "exp_cuped",
                "group_id": "control",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
            },
            {
                "unit_id": "u2",
                "experiment_id": "exp_cuped",
                "group_id": "control",
                "ts": dt.datetime(2025, 8, 4, 10, 0, 0),
            },
            {
                "unit_id": "u3",
                "experiment_id": "exp_cuped",
                "group_id": "treatment",
                "ts": dt.datetime(2025, 8, 5, 10, 0, 0),
            },
        ],
        "exposure_events_cuped",
    )


@pytest.fixture(scope="session")
def purchase_events_cuped(con):
    """Purchase events spanning pre and post periods.

    Jul 31: u1 pre-purchase
    Aug  1: u1 pre-purchase
    Aug  2: u1 pre-purchase
    Aug  3: u1 POST (11:00 > 10:00 exposure), u2 pre-purchase
    Aug  4: u2 POST (12:00 > 10:00 exposure)
    Aug  5: u2 POST (window), u3 POST (13:00 > 10:00 exposure)
    Aug  6: u3 POST (window)
    Aug  7: u3 POST (window)
    """
    return _table(
        con,
        [
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 7, 31, 12, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 12, 0, 0),
                "event": "purchase",
                "amount": 20.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 2, 12, 0, 0),
                "event": "purchase",
                "amount": 30.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 3, 11, 0, 0),
                "event": "purchase",
                "amount": 40.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 3, 9, 0, 0),
                "event": "purchase",
                "amount": 5.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 4, 12, 0, 0),
                "event": "purchase",
                "amount": 15.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 5, 12, 0, 0),
                "event": "purchase",
                "amount": 25.0,
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 8, 5, 13, 0, 0),
                "event": "purchase",
                "amount": 100.0,
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 8, 6, 12, 0, 0),
                "event": "purchase",
                "amount": 200.0,
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 8, 7, 12, 0, 0),
                "event": "purchase",
                "amount": 300.0,
            },
        ],
        "purchase_events_cuped",
    )


# Tests


class TestPrePeriodCovariate:
    """Pre-period events populate x; post-period populate y."""

    def test_pre_period_schema(
        self, con, cuped_experiment, exposure_events_cuped, purchase_events_cuped, mean_metric
    ):
        """unit_totals with pre_events follows the canonical schema."""
        exposures = first_exposures(exposure_events_cuped, cuped_experiment)
        events = metric_events(purchase_events_cuped, mean_metric, value_column="amount")
        spine, stats = unit_day_spine_stats(exposures, events, cuped_experiment, mean_metric.name)
        pre_stats = pre_period_stats(events, exposures, cuped_experiment, source_key="pre")
        totals = unit_totals(
            spine, stats, mean_metric, cuped_experiment, pre_stats=pre_stats
        ).execute()

        assert set(totals.columns) == UNIT_TOTALS
        assert totals["x"].notna().all(), "x should be populated when pre_events is provided"

    def test_pre_post_boundary_strict(
        self, con, cuped_experiment, exposure_events_cuped, purchase_events_cuped, mean_metric
    ):
        """Events at exactly first_exposure_ts go to y, not x."""
        exposures = first_exposures(exposure_events_cuped, cuped_experiment)
        events = metric_events(purchase_events_cuped, mean_metric, value_column="amount")
        spine, stats = unit_day_spine_stats(exposures, events, cuped_experiment, mean_metric.name)
        pre_stats = pre_period_stats(events, exposures, cuped_experiment, source_key="pre")
        totals = unit_totals(
            spine, stats, mean_metric, cuped_experiment, pre_stats=pre_stats
        ).execute()

        totals = totals.set_index("unit_id")

        # u1 pre-period [Jul 31, Aug 3 10:00): x = 10+20+30 = 60
        # post-exposure y: Aug 3 11:00 (40)
        assert abs(totals.loc["u1", "x"] - 60.0) < 1e-9, (
            f"u1 x expected 60.0, got {totals.loc['u1', 'x']}"
        )
        assert abs(totals.loc["u1", "y"] - 40.0) < 1e-9, (
            f"u1 y expected 40.0, got {totals.loc['u1', 'y']}"
        )

        # u2 pre-period [Aug 1, Aug 4 10:00): x = 5
        # post-exposure y: 15 + 25 = 40
        assert abs(totals.loc["u2", "x"] - 5.0) < 1e-9, (
            f"u2 x expected 5.0, got {totals.loc['u2', 'x']}"
        )
        assert abs(totals.loc["u2", "y"] - 40.0) < 1e-9, (
            f"u2 y expected 40.0, got {totals.loc['u2', 'y']}"
        )

    def test_pre_period_null_when_no_pre_events(
        self, con, cuped_experiment, exposure_events_cuped, purchase_events_cuped, mean_metric
    ):
        """x is null when pre_events is not provided."""
        exposures = first_exposures(exposure_events_cuped, cuped_experiment)
        events = metric_events(purchase_events_cuped, mean_metric, value_column="amount")
        spine, stats = unit_day_spine_stats(exposures, events, cuped_experiment, mean_metric.name)
        totals = unit_totals(spine, stats, mean_metric, cuped_experiment).execute()

        assert totals["x"].isna().all(), "x should be null when pre_events is not provided"

    def test_pre_period_with_n_pre_periods_zero(
        self, con, cuped_experiment, exposure_events_cuped, purchase_events_cuped, mean_metric
    ):
        """x is null when n_pre_periods=0 even if pre_events provided."""
        exp = cuped_experiment.model_copy(update={"n_pre_periods": 0})
        exposures = first_exposures(exposure_events_cuped, exp)
        events = metric_events(purchase_events_cuped, mean_metric, value_column="amount")
        spine, stats = unit_day_spine_stats(exposures, events, exp, mean_metric.name)
        pre_stats = pre_period_stats(events, exposures, exp, source_key="pre")
        totals = unit_totals(spine, stats, mean_metric, exp, pre_stats=pre_stats).execute()

        assert totals["x"].isna().all(), "x should be null when n_pre_periods=0"

    @pytest.mark.creates_tables
    def test_event_at_exact_exposure_ts_is_excluded_from_both_pre_and_post(
        self, con, cuped_experiment, exposure_events_cuped, mean_metric
    ):
        """An event landing at EXACTLY first_exposure_ts is excluded from
        BOTH y (post) and x (pre).

        The pre-period window is already strict on its upper bound
        ([first_exposure_ts - n_pre_periods days, first_exposure_ts)), so a
        boundary event was never eligible for x. It used to fall through to
        y instead, via unit_day_panel's inclusive `events.ts >=
        first_exposure_ts` join. That inclusive join is exactly what let a
        unit's own exposure-defining event get double-counted as a same-day
        metric occurrence whenever a metric's fact happened to match the
        exposure's underlying fact (the panel-join fix in
        increment/query/builders.py). Fixing that required tightening the
        join to strict `>` - and since the raw fact tables carry no unique
        per-event row id to key an explicit exclusion on instead, that
        necessarily also drops any event merely coincident with
        first_exposure_ts to the microsecond, not just the exposure row
        itself. In real, continuous-timestamp data that coincidence is
        unreachable outside the true self-counting case this fix targets,
        so a boundary event now correctly lands in neither x nor y. u1's
        exposure is exactly Aug 3 10:00:00; a purchase at that exact
        instant must be excluded from both.
        """
        boundary_events = _table(
            con,
            [
                {
                    "unit_id": "u1",
                    "ts": dt.datetime(2025, 8, 3, 10, 0, 0),  # exactly first_exposure_ts
                    "event": "purchase",
                    "amount": 999.0,
                },
            ],
            "boundary_purchase_events",
        )
        exposures = first_exposures(exposure_events_cuped, cuped_experiment)
        events = metric_events(boundary_events, mean_metric, value_column="amount")
        spine, stats = unit_day_spine_stats(exposures, events, cuped_experiment, mean_metric.name)
        pre_stats = pre_period_stats(events, exposures, cuped_experiment, source_key="pre")
        totals = unit_totals(
            spine, stats, mean_metric, cuped_experiment, pre_stats=pre_stats
        ).execute()
        totals = totals.set_index("unit_id")

        assert totals.loc["u1", "y"] == 0.0, (
            "event at exactly first_exposure_ts must be excluded from post (y), "
            f"got y={totals.loc['u1', 'y']}"
        )
        assert totals.loc["u1", "x"] == 0.0, (
            "event at exactly first_exposure_ts must NOT be included in the "
            f"pre-period covariate, got x={totals.loc['u1', 'x']}"
        )


def _pre_boundary_metric():
    from increment.semantics.models import MeanMetric

    return MeanMetric(
        name="pre_boundary_spend",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
    )


@pytest.mark.creates_tables
class TestPrePeriodDayBoundary:
    """The pre-period window's LOWER bound lives on the local day grid.

    Under ``day_boundary="UTC-05:00"`` with exposure 2025-08-05 09:00 UTC
    (local Aug 5) and ``n_pre_periods=2``, the window is local days
    [Aug 3, exposure). An event at 2025-08-03 02:00 UTC is LOCAL Aug 2,
    outside the window - and must be excluded; comparing raw UTC ts
    against the localized ``pre_start`` date (naive Aug 3 midnight) would
    wrongly admit its ~5-hour leak. An event at 2025-08-03 12:00 UTC is
    local Aug 3 and stays in-window either way (sanity anchor).
    """

    def _run(self, con, day_boundary: str) -> float:
        exp = Experiment(
            name="exp_pre_boundary",
            unit="unit_id",
            start=dt.datetime(2025, 8, 1),
            end=dt.datetime(2025, 8, 10),
            control_group="control",
            exposure="test_exposure",
            n_pre_periods=2,
            day_boundary=day_boundary,
            plan=AnalysisPlan(),
        )
        exposure_rows = _table(
            con,
            [
                {
                    "unit_id": "u1",
                    "experiment_id": "exp_pre_boundary",
                    "group_id": "control",
                    "ts": dt.datetime(2025, 8, 5, 9, 0, 0),
                }
            ],
            f"exp_pre_boundary_expo_{day_boundary[-5:].replace(':', '')}",
        )
        events = _table(
            con,
            [
                # local Aug 2 under UTC-05:00 (02:00 UTC): the leak candidate
                {
                    "unit_id": "u1",
                    "ts": dt.datetime(2025, 8, 3, 2, 0, 0),
                    "event": "purchase",
                    "amount": 100.0,
                },
                # local Aug 3 both ways: the in-window anchor
                {
                    "unit_id": "u1",
                    "ts": dt.datetime(2025, 8, 3, 12, 0, 0),
                    "event": "purchase",
                    "amount": 7.0,
                },
            ],
            f"exp_pre_boundary_ev_{day_boundary[-5:].replace(':', '')}",
        )
        exposures = first_exposures(exposure_rows, exp)
        m_events = metric_events(events, _pre_boundary_metric(), value_column="amount")
        pre = pre_period_stats(m_events, exposures, exp, source_key="pre").execute()
        return float(pre["sum_value"].sum())

    def test_lower_bound_excludes_events_from_before_the_local_window(self, con):
        assert self._run(con, "UTC-05:00") == 7.0, (
            "the 02:00 UTC event is local Aug 2 -- before the local [Aug 3, "
            "exposure) window -- and must not leak into the CUPED covariate"
        )

    def test_default_utc_window_admits_both_aug_3_events(self, con):
        assert self._run(con, "UTC") == 107.0

    def test_positive_offset_admits_event_from_the_earlier_utc_date(self, con):
        """The drop branch: under ``UTC+05:00`` an event at 2025-08-02
        21:00 UTC is LOCAL Aug 3 02:00 - inside the local [Aug 3,
        exposure) window - yet its UTC date (Aug 2) precedes
        ``pre_start``, so the old naive-midnight bound wrongly dropped
        it. The localized bound must admit it."""
        exp = Experiment(
            name="exp_pre_boundary_pos",
            unit="unit_id",
            start=dt.datetime(2025, 8, 1),
            end=dt.datetime(2025, 8, 10),
            control_group="control",
            exposure="test_exposure",
            n_pre_periods=2,
            day_boundary="UTC+05:00",
            plan=AnalysisPlan(),
        )
        exposure_rows = _table(
            con,
            [
                {
                    "unit_id": "u1",
                    "experiment_id": "exp_pre_boundary_pos",
                    "group_id": "control",
                    "ts": dt.datetime(2025, 8, 5, 9, 0, 0),
                }
            ],
            "exp_pre_boundary_pos_expo",
        )
        events = _table(
            con,
            [
                # UTC Aug 2 but local Aug 3: the previously-dropped event
                {
                    "unit_id": "u1",
                    "ts": dt.datetime(2025, 8, 2, 21, 0, 0),
                    "event": "purchase",
                    "amount": 50.0,
                },
                # local Aug 3 both ways: the in-window anchor
                {
                    "unit_id": "u1",
                    "ts": dt.datetime(2025, 8, 3, 12, 0, 0),
                    "event": "purchase",
                    "amount": 7.0,
                },
            ],
            "exp_pre_boundary_pos_ev",
        )
        exposures = first_exposures(exposure_rows, exp)
        m_events = metric_events(events, _pre_boundary_metric(), value_column="amount")
        pre = pre_period_stats(m_events, exposures, exp, source_key="pre").execute()
        assert float(pre["sum_value"].sum()) == 57.0
