"""End-to-end query test for RatioMetric.

Tests that a ratio metric flows through the full pipeline: metric_events
(numerator) -> unit_day_panel -> unit_totals (y, y_den, den_events from
denominator fact) -> group_summary (centered den moments). All expected values
are hand-computed from the conftest.py fixture data.

Design: metric_events for RatioMetric uses the numerator's Measure (the primary
stream; the caller builds denominator events separately). In unit_totals,
numerator.window_days is canonical for censoring/window filtering; the
denominator follows the same window and is aggregated as a per-unit sum.
"""

from __future__ import annotations

import ibis
import numpy as np
import pytest

from increment.errors import IncrementWarning
from increment.query.builders import (
    first_exposures,
    group_summary,
    metric_events,
    post_exposure_stats,
    unit_day_spine_stats,
    unit_totals,
)
from increment.query.schemas import GROUP_SUMMARY, UNIT_TOTALS
from increment.semantics.models import Measure, RatioMetric
from tests.warning_codes import warning_codes


def _total(row, ref, resid):
    """Recover a raw first moment from the centered wire fields."""
    return row["n"] * row[ref] + row[resid]


def _cancel_tol(n, ref):
    """Absolute tolerance for a cancelling residual first moment.

    ``cy1``/``cden1`` are ~0 but carry the last bits of an ``n * ref``
    sized cancellation, so they need an absolute band scaled to that
    magnitude, never a relative one against zero.
    """
    return 1e-9 * max(1.0, abs(n * ref))


# Fixtures


@pytest.fixture
def ratio_metric():
    """Ratio metric: sum(purchase amount) / count(page_view events).

    Hand-computed expected per-unit totals (window_days=3):
      u1: num=49.99+12.00+7.50=69.49, den=2 (Aug 1 + Aug 3 page views)
      u2: num=30.00 (Aug 2; Aug 1 is pre-exposure), den=1 (Aug 2)
      u3: num=19.99, den=1 (Aug 3)
    """
    return RatioMetric(
        name="revenue_per_pageview",
        entity="unit_id",
        numerator=Measure(
            fact="purchase",
            aggregation="sum",
            window_days=3,
        ),
        denominator=Measure(
            fact="page_view",
            aggregation="count",
            window_days=3,
        ),
    )


@pytest.fixture
def ratio_metric_variable_window():
    """Ratio metric with variable window (window_days=None).

    Both numerator and denominator have window_days=None.  The panel
    should not be window-filtered, only experiment-end-censored.
    """
    return RatioMetric(
        name="revenue_per_pageview_vw",
        entity="unit_id",
        numerator=Measure(
            fact="purchase",
            aggregation="sum",
            window_days=None,
        ),
        denominator=Measure(
            fact="page_view",
            aggregation="count",
            window_days=None,
        ),
    )


@pytest.fixture
def ratio_metric_diff_window():
    """Ratio metric where numerator and denominator have DIFFERENT window_days.

    This tests the convention that numerator.window_days is canonical
    for censoring and window filtering.
    """
    return RatioMetric(
        name="revenue_per_pageview_diff",
        entity="unit_id",
        numerator=Measure(
            fact="purchase",
            aggregation="sum",
            window_days=3,
        ),
        denominator=Measure(
            fact="page_view",
            aggregation="count",
            window_days=1,  # different from numerator
        ),
    )


# Tests: metric_events


class TestRatioMetricEvents:
    """metric_events handles RatioMetric by returning numerator events."""

    def test_metric_events_ratio_uses_numerator_fact(self, purchase_events, ratio_metric):
        """RatioMetric metric_events returns only numerator-fact rows."""
        result = metric_events(purchase_events, ratio_metric, value_column="amount")
        df = result.execute()
        # All rows should be purchase events
        assert len(df) > 0
        assert all(df["metric"] == "revenue_per_pageview")
        # Check value column is populated from the amount column
        assert df["value"].dtype in (np.float64, np.float32)

    def test_metric_events_ratio_schema(self, purchase_events, ratio_metric):
        """RatioMetric metric_events output matches canonical schema."""
        result = metric_events(purchase_events, ratio_metric, value_column="amount")
        assert set(result.columns) == {"unit_id", "ts", "metric", "value"}

    def test_metric_events_ratio_numerator_count_aggregation(self, page_view_events):
        """RatioMetric with a count-aggregated numerator stamps value=1 per
        event (same occurrence semantics as ConversionMetric/RetentionMetric),
        not a real column value - regression test for a bug where the
        occurrence-vs-value dispatch only checked isinstance(metric, MeanMetric)
        and never inspected a RatioMetric part's own aggregation, crashing on
        any count-aggregated ratio part (e.g. clicks_per_session)."""
        count_ratio_metric = RatioMetric(
            name="views_per_session",
            entity="unit_id",
            numerator=Measure(fact="page_view", aggregation="count"),
            # Distinct sessions, so the two sides are genuinely different
            # measures: an identical pair always resolves to a ratio of 1 and
            # is refused at the declaration boundary.
            denominator=Measure(fact="page_view", aggregation="count_distinct"),
        )
        result = metric_events(page_view_events, count_ratio_metric, part="numerator").execute()
        assert len(result) > 0
        assert (result["value"] == 1.0).all()

    def test_metric_events_unknown_part_refused(self, purchase_events, ratio_metric):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            metric_events(purchase_events, ratio_metric, part="bogus")  # ty: ignore[invalid-argument-type]
        assert exc_info.value.code == "query.builders.unknown_part_ratiometric"
        assert exc_info.value.context["part"] == "bogus"

    def test_metric_events_denominator_part_refused_for_non_ratio_metric(
        self, purchase_events, mean_metric
    ):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as exc_info:
            metric_events(purchase_events, mean_metric, part="denominator")
        assert exc_info.value.code == "query.builders.part_ratiometric"
        assert exc_info.value.context["part"] == "denominator"


# Tests: unit_totals


class TestRatioUnitTotals:
    """unit_totals processes RatioMetric with numerator/denominator aggregation."""

    def test_ratio_totals_y_and_y_den_populated(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric,
    ):
        """y = numerator sum, y_den = denominator sum, both populated."""
        # Build numerator events
        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")

        # Build denominator events directly (occurrence-counting fact)
        den_events = metric_events(page_view_events, ratio_metric, part="denominator")

        # Build spine+stats from numerator events
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)

        # Build totals with both numerator and denominator
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        totals = unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)

        df = totals.execute().set_index("unit_id")

        # Verify schema
        assert set(totals.columns) == UNIT_TOTALS

        # Hand-computed values (window_days=3 for both)
        # u1: num=49.99+12.00+7.50=69.49, den=2 (Aug 1 + Aug 3 page views)
        assert abs(df.loc["u1", "y"] - 69.49) < 1e-9, (
            f"u1 y expected 69.49, got {df.loc['u1', 'y']}"
        )
        assert df.loc["u1", "y_den"] == 2.0, f"u1 y_den expected 2, got {df.loc['u1', 'y_den']}"

        # u2: num=30.00 (Aug 2), den=1 (Aug 2)
        assert abs(df.loc["u2", "y"] - 30.00) < 1e-9, (
            f"u2 y expected 30.00, got {df.loc['u2', 'y']}"
        )
        assert df.loc["u2", "y_den"] == 1.0, f"u2 y_den expected 1, got {df.loc['u2', 'y_den']}"

        # u3: num=19.99, den=1 (Aug 3)
        assert abs(df.loc["u3", "y"] - 19.99) < 1e-9, (
            f"u3 y expected 19.99, got {df.loc['u3', 'y']}"
        )
        assert df.loc["u3", "y_den"] == 1.0, f"u3 y_den expected 1, got {df.loc['u3', 'y_den']}"

    def test_ratio_totals_schema(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric,
    ):
        """unit_totals output matches canonical schema."""
        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")
        den_events = metric_events(page_view_events, ratio_metric, part="denominator")
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        totals = unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)
        assert set(totals.columns) == UNIT_TOTALS

    def test_ratio_totals_x_is_null(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric,
    ):
        """x column is still None (CUPED not materialised)."""
        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")
        den_events = metric_events(page_view_events, ratio_metric, part="denominator")
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        totals = unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)
        df = totals.execute()
        assert df["x"].isna().all()

    def test_ratio_totals_requires_den_events(
        self, exposures, purchase_events, experiment, ratio_metric
    ):
        """RatioMetric without den_events raises."""
        from increment.errors import InvalidRequestError

        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
        with pytest.raises(InvalidRequestError) as exc_info:
            unit_totals(spine, stats, ratio_metric, experiment)
        assert exc_info.value.code == "query.builders.ratiometric_den_stats"

    @pytest.mark.parametrize(
        ("aggregation", "expected_units"),
        [("count", {"u1", "u2", "u3"}), ("avg_event", {"u2", "u3"})],
    )
    def test_ratio_totals_preserve_undefined_denominator_policy(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric,
        aggregation,
        expected_units,
    ):
        """An empty denominator is defined for count, undefined for event averages."""
        ratio_metric = RatioMetric(
            name=ratio_metric.name,
            entity=ratio_metric.entity,
            numerator=ratio_metric.numerator,
            denominator=Measure(fact="page_view", aggregation=aggregation, window_days=3),
        )
        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")
        den_events = metric_events(
            page_view_events.mutate(amount=ibis.literal(1.0)),
            ratio_metric,
            part="denominator",
            value_column="amount",
        )
        # Remove u1's denominator events — u1 has numerator events (y=69.49)
        den_events_filtered = den_events.filter(den_events.unit_id != ibis.literal("u1"))
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
        den_stats = post_exposure_stats(den_events_filtered, exposures, source_key="den")
        totals = unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)
        df = totals.execute().set_index("unit_id")

        assert set(df.index) == expected_units
        if aggregation == "count":
            assert df.loc["u1", "y_den"] == 0.0
            assert df.loc["u1", "y"] == pytest.approx(69.49)


# Tests: group_summary


class TestRatioGroupSummary:
    """group_summary produces populated denominator moments for ratio metrics."""

    def test_group_summary_den_moments_populated(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric,
    ):
        """ref_den/cden1/cden2/cyden are populated (not null) for ratio."""
        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")
        den_events = metric_events(page_view_events, ratio_metric, part="denominator")
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        totals = unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)
        summary = group_summary(totals)
        df = summary.execute()

        # Verify schema
        assert set(summary.columns) == GROUP_SUMMARY

        # Verify the whole denominator family is populated
        for _, row in df.iterrows():
            for field in ("ref_den", "cden1", "cden2", "cyden"):
                assert not (isinstance(row[field], float) and np.isnan(row[field])), (
                    f"{field} should be populated"
                )

    def test_group_summary_den_moments_hand_computed(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric,
    ):
        """Denominator moments match hand-computed values.

        Units and groups:
          u1: treatment, num=69.49, den=2
          u2: control, num=30.00, den=1
          u3: treatment, num=19.99, den=1

        Treatment (u1, u3): n=2, ref_y=(69.49+19.99)/2=44.74,
          ref_den=(2+1)/2=1.5, so the residuals are y-44.74 = +-24.75
          and den-1.5 = +-0.5::

            cy1=0, cy2=2*24.75**2=1225.125
            cden1=0, cden2=2*0.5**2=0.5
            cyden=24.75*0.5 + (-24.75)*(-0.5)=24.75

        Control (u2): n=1, ref_y=30.00, ref_den=1, every centered
          moment exactly 0.
        """
        num_events = metric_events(purchase_events, ratio_metric, value_column="amount")
        den_events = metric_events(page_view_events, ratio_metric, part="denominator")
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        totals = unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)
        summary = group_summary(totals)
        df = summary.execute().set_index(["group_id", "metric"])

        # Treatment - references recomputed from the per-unit values.
        trt_y = [69.49, 19.99]
        trt_den = [2.0, 1.0]
        ref_y = sum(trt_y) / 2
        ref_den = sum(trt_den) / 2
        dy = [v - ref_y for v in trt_y]
        dden = [v - ref_den for v in trt_den]

        trt = df.loc[("treatment", "revenue_per_pageview")]
        assert trt["n"] == 2
        assert abs(trt["ref_y"] - ref_y) < 1e-9
        assert abs(trt["cy1"] - sum(dy)) < _cancel_tol(2, ref_y)
        assert abs(trt["cy2"] - sum(v * v for v in dy)) < 1e-9
        assert abs(_total(trt, "ref_y", "cy1") - 89.48) < 1e-9
        assert trt["ref_den"] == ref_den
        assert abs(trt["cden1"] - sum(dden)) < _cancel_tol(2, ref_den)
        assert trt["cden2"] == sum(v * v for v in dden)
        assert _total(trt, "ref_den", "cden1") == 3.0
        assert abs(trt["cyden"] - sum(a * b for a, b in zip(dy, dden, strict=True))) < 1e-9

        # Control - a single unit, so every centered moment is exactly 0.
        ctrl = df.loc[("control", "revenue_per_pageview")]
        assert ctrl["n"] == 1
        assert abs(ctrl["ref_y"] - 30.00) < 1e-9
        assert ctrl["cy1"] == 0.0
        assert ctrl["cy2"] == 0.0
        assert ctrl["ref_den"] == 1.0
        assert ctrl["cden1"] == 0.0
        assert ctrl["cden2"] == 0.0
        assert ctrl["cyden"] == 0.0


# Tests: window alignment


class TestRatioWindowAlignment:
    """Window-alignment conventions for RatioMetric.

    Design decision: numerator.window_days is canonical for
    censoring and window filtering.  The denominator follows the
    same window (not its own).
    """

    def test_censoring_uses_numerator_window(
        self,
        censored_exposure_events,
        censored_page_view_events,
        censored_purchase_events,
        censored_experiment,
    ):
        """RatioMetric censoring uses numerator.window_days.

        Experiment ends Aug 3.  With numerator.window_days=2, u5
        (exposed Aug 3) is dropped: its last window day (Aug 4) is
        past the Aug 3 bound.
        """
        metric = RatioMetric(
            name="test_ratio",
            entity="unit_id",
            numerator=Measure(fact="purchase", aggregation="sum", window_days=2),
            denominator=Measure(fact="page_view", aggregation="count", window_days=2),
        )
        exp = censored_experiment
        exposures = first_exposures(censored_exposure_events, exp)

        num_events = metric_events(censored_purchase_events, metric, value_column="amount")
        den_events = metric_events(censored_page_view_events, metric, part="denominator")
        spine, stats = unit_day_spine_stats(exposures, num_events, exp, metric.name)
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        with pytest.warns(IncrementWarning) as rec:
            totals = unit_totals(spine, stats, metric, exp, den_stats=den_stats)
        assert "frame.censoring.dropped_units" in warning_codes(rec)
        df = totals.execute()
        uids = sorted(df["unit_id"].tolist())
        assert "u5" not in uids, "u5 should be censored"
        assert "u1" in uids
        assert "u2" in uids

    def test_variable_window_no_censoring(
        self,
        exposures,
        purchase_events,
        page_view_events,
        experiment,
        ratio_metric_variable_window,
    ):
        """Variable window (window_days=None) skips censoring."""
        metric = ratio_metric_variable_window
        num_events = metric_events(purchase_events, metric, value_column="amount")
        den_events = metric_events(page_view_events, metric, part="denominator")
        spine, stats = unit_day_spine_stats(exposures, num_events, experiment, metric.name)
        den_stats = post_exposure_stats(den_events, exposures, source_key="den")
        totals = unit_totals(spine, stats, metric, experiment, den_stats=den_stats)
        df = totals.execute()
        assert set(df["unit_id"].tolist()) == {"u1", "u2", "u3"}


# Tests: denominator event horizon (running experiments)


def test_running_experiment_denominator_events_past_numerator_horizon_count(tmp_path):
    """On a RUNNING experiment (no declared end), the spine's right edge
    falls back to observed event dates. That union must include a ratio's
    DENOMINATOR stream: a page_view dated after every purchase would
    otherwise be spine-truncated and silently vanish from the emitted
    denominator moments, biasing the ratio upward and diverging from the
    frame substrate (which sums denominator events unconditionally).

    Exercises the real production path: ``Analysis.from_definitions`` ->
    ``_union_event_horizon`` -> ``export`` (group_summary moments).
    """
    import datetime as dt

    import pandas as pd
    import pyarrow.parquet as pq

    from increment.analysis import Analysis

    con = ibis.duckdb.connect()
    fe = dt.datetime(2025, 6, 2, 10, 0, 0)
    rows = [
        # exposures
        {
            "user_id": "u_t",
            "ts": fe,
            "event": "exposure",
            "amount": None,
            "group_id": "treatment",
            "experiment_id": "exp",
        },
        {
            "user_id": "u_c",
            "ts": fe,
            "event": "exposure",
            "amount": None,
            "group_id": "control",
            "experiment_id": "exp",
        },
        # purchases (numerator); latest purchase anywhere is Jun 3
        {
            "user_id": "u_t",
            "ts": dt.datetime(2025, 6, 2, 12),
            "event": "purchase",
            "amount": 10.0,
            "group_id": "treatment",
            "experiment_id": "exp",
        },
        {
            "user_id": "u_c",
            "ts": dt.datetime(2025, 6, 3, 11),
            "event": "purchase",
            "amount": 4.0,
            "group_id": "control",
            "experiment_id": "exp",
        },
        # page views (denominator); u_t's last four land Jun 10, a week
        # PAST the numerator's own event horizon
        {
            "user_id": "u_t",
            "ts": dt.datetime(2025, 6, 3, 9),
            "event": "page_view",
            "amount": None,
            "group_id": "treatment",
            "experiment_id": "exp",
        },
        *[
            {
                "user_id": "u_t",
                "ts": dt.datetime(2025, 6, 10, 9 + h),
                "event": "page_view",
                "amount": None,
                "group_id": "treatment",
                "experiment_id": "exp",
            }
            for h in range(4)
        ],
        {
            "user_id": "u_c",
            "ts": dt.datetime(2025, 6, 3, 9),
            "event": "page_view",
            "amount": None,
            "group_id": "control",
            "experiment_id": "exp",
        },
        {
            "user_id": "u_c",
            "ts": dt.datetime(2025, 6, 3, 10),
            "event": "page_view",
            "amount": None,
            "group_id": "control",
            "experiment_id": "exp",
        },
    ]
    con.create_table("horizon_events", obj=pd.DataFrame(rows))

    defs_path = tmp_path / "defs.yaml"
    defs_path.write_text(
        """
dialect: duckdb
fact_sources:
  - name: events
    sql: 'SELECT * FROM horizon_events'
    timestamp_column: ts
    entities: [user_id]
    facts:
      - name: exposure
      - name: purchase
        column: amount
      - name: page_view
exposures:
  - name: e
    fact: exposure
metrics:
  - type: ratio
    name: rev_per_view
    entity: user_id
    numerator:
      fact: purchase
      aggregation: sum
    denominator:
      fact: page_view
      aggregation: count
experiments:
  - name: exp
    exposure: e
    unit: user_id
    start: 2025-06-01
    control_group: control
    plan: {secondaries: [rev_per_view]}
"""
    )
    a = Analysis.from_definitions("exp", defs_path, con)
    out = tmp_path / "moments.parquet"
    a.export(out)
    moments = {r["group_id"]: r for r in pq.read_table(out).to_pylist()}

    # 5 treatment page views (incl. 4 dated past the numerator's last purchase) and 2 control page views; one unit per arm, so ref_den/ref_y carry the whole totals.
    assert _total(moments["treatment"], "ref_den", "cden1") == 5.0
    assert _total(moments["control"], "ref_den", "cden1") == 2.0
    assert _total(moments["treatment"], "ref_y", "cy1") == 10.0
    assert _total(moments["control"], "ref_y", "cy1") == 4.0


def test_ratio_totals_honor_numerator_avg_event_and_denominator_count(con, experiment):
    """Unequal event counts must not turn the declared +25% lift into -37.5%."""
    import datetime as dt

    metric = RatioMetric(
        name="mean_purchase_per_view",
        entity="unit_id",
        numerator=Measure(fact="num", aggregation="avg_event", window_days=2),
        denominator=Measure(fact="den", aggregation="count", window_days=2),
    )
    exposure_rows = []
    numerator_rows = []
    denominator_rows = []
    event_ts = experiment.start + dt.timedelta(hours=1)
    for group in ("control", "treatment"):
        for i in range(30):
            unit = f"{group}-{i}"
            exposure_rows.append(
                {
                    "unit_id": unit,
                    "experiment_id": experiment.name,
                    "group_id": group,
                    "ts": experiment.start,
                }
            )
            values = [2 + i % 3, 4 + i % 3] if group == "control" else [4 + i % 3]
            numerator_rows.extend(
                {"unit_id": unit, "ts": event_ts, "event": "num", "amount": value}
                for value in values
            )
            denominator_rows.append({"unit_id": unit, "ts": event_ts, "event": "den"})
    exposures = first_exposures(ibis.memtable(exposure_rows), experiment)
    numerator = metric_events(ibis.memtable(numerator_rows), metric, value_column="amount")
    denominator = metric_events(ibis.memtable(denominator_rows), metric, part="denominator")
    spine, stats = unit_day_spine_stats(exposures, numerator, experiment, metric.name)
    den_stats = post_exposure_stats(denominator, exposures, source_key="den")
    moments = con.to_pyarrow(
        group_summary(unit_totals(spine, stats, metric, experiment, den_stats=den_stats))
    ).to_pylist()
    ratios = {
        row["group_id"]: _total(row, "ref_y", "cy1") / _total(row, "ref_den", "cden1")
        for row in moments
    }
    assert ratios == pytest.approx({"control": 4.0, "treatment": 5.0}, rel=0, abs=1e-9)
    assert ratios["treatment"] / ratios["control"] - 1 == pytest.approx(0.25, rel=0, abs=1e-9)
