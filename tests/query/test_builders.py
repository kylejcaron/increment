"""Tests for the ibis query-layer builders.

Every test group follows TDD: the expected values are hand-computed in
``conftest.py`` comments, and each test starts by verifying the **red**
failing assertion before turning green.
"""

from __future__ import annotations

import datetime as dt
import math
import uuid
from typing import Literal

import ibis
import numpy as np
import pyarrow as pa
import pytest
from pydantic import ValidationError

from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    UnsupportedRequestError,
)
from increment.query.builders import (
    asof_group_summary,
    breakout_property_table,
    canonical_dimension_value,
    cohort_group_summary,
    daily_exposure_counts,
    daily_group_summary,
    declared_binary_metrics,
    first_exposures,
    group_summary,
    join_breakout_dimension,
    metric_events,
    panel_spine,
    post_exposure_stats,
    site_volume,
    unit_day_panel,
    unit_day_spine_stats,
    unit_totals,
    window_bound_stats,
    winsorize_unit_totals,
)
from increment.query.schemas import (
    DAILY_EXPOSURE_COUNTS,
    DAILY_GROUP_SUMMARY,
    EXPOSURES,
    GROUP_SUMMARY,
    METRIC_EVENTS,
    SITE_VOLUME,
    UNIT_DAY_PANEL,
    UNIT_TOTALS,
)
from increment.semantics.models import (
    AnalysisPlan,
    ConversionMetric,
    Definitions,
    Experiment,
    Filter,
    MeanMetric,
    Measure,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
    Winsorization,
)
from tests.analysis_factory import make_analysis
from tests.warning_codes import warning_codes, warning_context

# Nearly every test here builds its own tiny scratch table directly on
# the shared `con` connection - that is this module's established style,
# not accidental cross-test state. Exempt the whole file from the
# no-stray-tables guard rather than decorating ~150 individual tests.
pytestmark = pytest.mark.creates_tables


def _total(row, ref="ref_y", resid="cy1"):
    """Recover a raw first moment from the centered wire fields.

    ``sum_v == n * ref_v + cv1`` - the identity the residual first
    moment exists to make exact.
    """
    return row["n"] * row[ref] + row[resid]


def _cancel_tol(n, ref):
    """Absolute tolerance for a cancelling residual first moment.

    ``cy1``/``cx1``/``cden1`` are ~0 but carry the last bits of an
    ``n * ref`` sized cancellation, so they need an absolute band scaled
    to that magnitude, never a relative one against zero.
    """
    return 1e-9 * max(1.0, abs(n * ref))


# Step 2 — first_exposures


def test_first_exposures_schema(exposure_events, experiment):
    """Output columns match the canonical schema."""
    result = first_exposures(exposure_events, experiment)
    assert set(result.columns) == EXPOSURES


def test_first_exposures_dedup_min(exposure_events, experiment):
    """Duplicate unit×experiment rows are deduped to MIN(ts)."""
    result = first_exposures(exposure_events, experiment).execute()
    u1 = result[result["unit_id"] == "u1"].iloc[0]
    assert u1["first_exposure_ts"] == dt.datetime(2025, 8, 1, 9, 0, 0)


def test_first_exposures_exact_rows(exposure_events, experiment):
    """Output contains exactly the expected 3 units (u1, u2, u3)."""
    result = first_exposures(exposure_events, experiment).execute()
    uids = sorted(result["unit_id"].tolist())
    assert uids == ["u1", "u2", "u3"]


def test_first_exposures_mixed_group_dropped(exposure_events, experiment):
    """u4 (treatment + control) is excluded."""
    result = first_exposures(exposure_events, experiment).execute()
    assert "u4" not in result["unit_id"].values


def test_mixed_assignment_units_counts_the_dropped(exposure_events, experiment):
    """The one mixed unit (u4: treatment + control) is surfaced as a count.

    first_exposures drops mixed units before any count is taken, so the
    SRM check's post-drop counts cannot see a symmetric contamination.
    ``mixed_assignment_units`` is the accounting entry that makes it visible.
    """
    from increment.query.builders import mixed_assignment_units

    out = mixed_assignment_units(exposure_events, experiment).execute()
    assert int(out["mixed_count"].iloc[0]) == 1
    assert int(out["unassigned_count"].iloc[0]) == 0


def test_first_exposures_separates_null_and_mixed_assignments(con, experiment):
    """NULL assignments are invalid but never become a mixed arm."""
    from increment.query.builders import mixed_assignment_units

    rows = con.create_table(
        "assignment_integrity_exposures",
        obj=pa.table(
            {
                "unit_id": [
                    "null_only",
                    "control_plus_null",
                    "control_plus_null",
                    "mixed",
                    "mixed",
                    "valid",
                ],
                "ts": [dt.datetime(2025, 8, 1, 9)] * 6,
                "event": ["exposure"] * 6,
                "experiment_id": [experiment.name] * 6,
                "group_id": pa.array(
                    [None, "control", None, "control", "treatment", "control"],
                    type=pa.string(),
                ),
            }
        ),
    )
    first = first_exposures(rows, experiment).execute()
    assert first["unit_id"].tolist() == ["valid"]

    integrity = mixed_assignment_units(rows, experiment).execute().iloc[0]
    assert int(integrity["mixed_count"]) == 1
    assert int(integrity["unassigned_count"]) == 2


def test_mixed_assignment_units_respects_enrollment_close(con):
    """A second-group exposure AFTER experiment.end's day is out of scope:
    the unit is not mixed within the enrollment window (same whole-day
    close first_exposures applies)."""
    from increment.query.builders import mixed_assignment_units

    exp = Experiment(
        name="exp_mixed_close",
        unit="unit_id",
        exposure="exposure",
        control_group="control",
        start=dt.datetime(2025, 3, 1),
        end=dt.datetime(2025, 3, 20),
        plan=AnalysisPlan(),
    )
    rows = con.create_table(
        "mixed_close_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 3, 10, 9),
                "event": "exposure",
                "experiment_id": "exp_mixed_close",
                "group_id": "treatment",
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 3, 21, 9),  # past end's day
                "event": "exposure",
                "experiment_id": "exp_mixed_close",
                "group_id": "control",
            },
        ],
    )
    out = mixed_assignment_units(rows, exp).execute()
    assert int(out["mixed_count"].iloc[0]) == 0
    assert int(out["unassigned_count"].iloc[0]) == 0


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery", "postgres"])
def test_mixed_assignment_units_renders_supported_backends(exposure_events, experiment, dialect):
    from increment.query.builders import mixed_assignment_units

    ibis.to_sql(mixed_assignment_units(exposure_events, experiment), dialect=dialect)


def test_first_exposures_end_time_of_day_ignored(con):
    """Enrollment close is whole-day inclusive: an exposure at 18:00 on an
    ``end`` declared at 12:00 the same day is still enrolled, and the next
    day's exposure is dropped (time-of-day on ``end`` is deliberately
    ignored - documented on Experiment.end)."""
    exp = Experiment(
        name="exp_end_tod",
        unit="unit_id",
        exposure="exposure",
        control_group="control",
        start=dt.datetime(2025, 3, 1),
        end=dt.datetime(2025, 3, 20, 12, 0),
        plan=AnalysisPlan(),
    )
    rows = con.create_table(
        "end_tod_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 3, 20, 18, 0),
                "event": "exposure",
                "experiment_id": "exp_end_tod",
                "group_id": "treatment",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 3, 21, 0, 1),
                "event": "exposure",
                "experiment_id": "exp_end_tod",
                "group_id": "control",
            },
        ],
    )
    out = first_exposures(rows, exp).execute()
    assert sorted(out["unit_id"].tolist()) == ["u1"]


def test_first_exposures_survives_a_source_without_an_experiment_id(con):
    """A fact source with no ``experiment_id`` column arrives with an
    all-NULL one (synthesized by ``_rename_to_builder_cols``), and every
    such row still belongs to the analysed experiment.

    Regression: the single-group semi-join keyed on ``[unit_id,
    experiment_id]`` never matched under ``NULL = NULL``, so enrollment
    came back EMPTY - silently reporting zero exposed units instead of
    raising. The declared experiment's name now fills the column.
    """
    exp = Experiment(
        name="exp_no_id",
        unit="unit_id",
        exposure="exposure",
        control_group="control",
        start=dt.datetime(2025, 3, 1),
        plan=AnalysisPlan(),
    )
    rows = con.create_table(
        "no_id_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 3, 2, 9, 0),
                "event": "exposure",
                "group_id": "treatment",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 3, 2, 10, 0),
                "event": "exposure",
                "group_id": "control",
            },
        ],
    ).mutate(experiment_id=ibis.null().cast("string"))
    out = first_exposures(rows, exp).execute()
    assert sorted(out["unit_id"].tolist()) == ["u1", "u2"]
    assert set(out["experiment_id"]) == {"exp_no_id"}


def test_first_exposures_drops_units_exposed_after_the_enrollment_end(con, experiment):
    """Enrollment closes at experiment.end; a later exposure is not in the experiment.

    The `experiment` fixture ends 2025-08-06.
    """
    exposure_rows = con.create_table(
        "late_enrol_exposures",
        obj=[
            {
                "unit_id": "early",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "late",
                "ts": dt.datetime(2025, 8, 9, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
        ],
    )
    got = first_exposures(exposure_rows, experiment).execute()
    assert sorted(got["unit_id"]) == ["early"]


def test_first_exposures_keeps_a_unit_exposed_later_the_same_day_as_end(con, experiment):
    """A whole-day-inclusive `end`: an exposure at 9am on the end day still enrolls.

    Regression for the midnight-comparison bug: comparing the raw ts against
    `experiment.end` (midnight) would wrongly drop this unit even though it
    enrolled on the last valid day.
    """
    exposure_rows = con.create_table(
        "end_day_exposures",
        obj=[
            {
                "unit_id": "end_day",
                "ts": dt.datetime(2025, 8, 6, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    got = first_exposures(exposure_rows, experiment).execute()
    assert sorted(got["unit_id"]) == ["end_day"]


def test_first_exposures_bounds_enrollment_at_experiment_start(con, experiment):
    """Exposures strictly before experiment.start must not enroll -- a
    pre-launch event otherwise anchors a unit's day 0 before the
    experiment existed, silently changing window eligibility.

    The `experiment` fixture starts 2025-08-01.
    """
    exposure_rows = con.create_table(
        "pre_start_exposures",
        obj=[
            {
                "unit_id": "u_pre",
                "ts": dt.datetime(2025, 7, 25, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "u_post",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
        ],
    )
    got = first_exposures(exposure_rows, experiment).execute()
    assert sorted(got["unit_id"]) == ["u_post"]


def test_first_exposures_rejects_null_ts_and_unit_id(con, experiment):
    """A NULL `ts`/`unit_id` exposure row is dropped before grouping.

    A unit exposed only via a NULL ts otherwise gets
    first_exposure_ts=NULL (SQL MIN skips NULLs only when every row is
    NULL); panel_spine then derives a NULL day range and emits zero rows
    for that unit, while a consumer counting straight off
    `exposures`/`first_exposures` output still counts it -- an enrolled
    unit silently vanishes from one readout while still being counted by
    another.
    """
    exposure_rows = con.create_table(
        "null_ts_exposures",
        obj=[
            {
                "unit_id": "u_ok",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "u_null_ts",
                "ts": None,
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
            {
                "unit_id": None,
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            },
        ],
    )
    got = first_exposures(exposure_rows, experiment).execute()
    assert sorted(got["unit_id"]) == ["u_ok"]
    assert got["first_exposure_ts"].notna().all()


# Step 2c — daily_exposure_counts


def test_daily_exposure_counts_schema(exposures):
    """Output columns match the canonical schema."""
    result = daily_exposure_counts(exposures)
    assert set(result.columns) == DAILY_EXPOSURE_COUNTS


def test_daily_exposure_counts_cumulative_monotonic(exposures):
    """n_cumulative never decreases within an (experiment_id, group_id)."""
    df = daily_exposure_counts(exposures).execute()
    for _, group in df.sort_values("ds").groupby(["experiment_id", "group_id"]):
        assert (group["n_cumulative"].diff().dropna() >= 0).all()


def test_daily_exposure_counts_ties_to_first_exposures_total(
    exposures, exposure_events, experiment
):
    """Final-day n_cumulative per group equals that group's row count in
    first_exposures - daily_exposure_counts must not lose or invent units."""
    df = daily_exposure_counts(exposures).execute()
    firsts = first_exposures(exposure_events, experiment)

    for group_id in ("treatment", "control"):
        expected = firsts.filter(firsts.group_id == group_id).count().execute()
        final_cumulative = df[df["group_id"] == group_id]["n_cumulative"].max()
        assert final_cumulative == expected


def test_daily_exposure_counts_gap_day_carries_forward(exposures):
    """A quiet arm (control) gets an explicit n_daily=0 row on a day another
    arm (treatment) enrolled, with n_cumulative carried forward unchanged.

    exposure_events: u2/control enrolls Aug 1 only; u1/treatment enrolls
    Aug 1, and u3/treatment enrolls Aug 2 - so Aug 2 is an observed date
    for the experiment, but control enrolled nobody that day.
    """
    df = daily_exposure_counts(exposures).execute()
    control_aug2 = df[(df["group_id"] == "control") & (df["ds"].dt.date == dt.date(2025, 8, 2))]
    control_aug1 = df[(df["group_id"] == "control") & (df["ds"].dt.date == dt.date(2025, 8, 1))]
    assert len(control_aug2) == 1
    assert control_aug2["n_daily"].iloc[0] == 0
    assert control_aug2["n_cumulative"].iloc[0] == control_aug1["n_cumulative"].iloc[0]


def test_daily_exposure_counts_compiles_to_snowflake(exposures):
    """The daily exposure query compiles for Snowflake."""
    ibis.to_sql(daily_exposure_counts(exposures), dialect="snowflake")


def test_daily_exposure_counts_does_not_cross_experiments(con):
    """Densification is scoped per experiment_id: a date observed only in one
    experiment must not spawn a row for a group that belongs to another
    experiment, and vice versa.

    The shared ``exposures`` fixture only carries a single ``experiment_id``,
    so it cannot exercise this - the cross-join collapses to the single-key
    case regardless of whether the join is scoped correctly. This builds a
    dedicated two-experiment table instead.
    """
    rows = [
        # exp_a: treatment/control, active 2025-08-01 and 2025-08-02
        {
            "unit_id": "a1",
            "experiment_id": "exp_a",
            "group_id": "treatment",
            "first_exposure_ts": dt.datetime(2025, 8, 1, 9, 0),
        },
        {
            "unit_id": "a2",
            "experiment_id": "exp_a",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 8, 2, 9, 0),
        },
        # exp_b: only a "variant" arm, active 2025-08-05 - a date exp_a
        # never observed, and a group_id exp_a never has.
        {
            "unit_id": "b1",
            "experiment_id": "exp_b",
            "group_id": "variant",
            "first_exposure_ts": dt.datetime(2025, 8, 5, 9, 0),
        },
    ]
    table_name = "two_exp_exposures"
    if table_name in con.list_tables():
        exposures_two = con.table(table_name)
    else:
        exposures_two = con.create_table(table_name, obj=rows)

    df = daily_exposure_counts(exposures_two).execute()

    # exp_a never has a "variant" row, and never has a row on 2025-08-05.
    assert not ((df["experiment_id"] == "exp_a") & (df["group_id"] == "variant")).any()
    assert not ((df["experiment_id"] == "exp_a") & (df["ds"].dt.date == dt.date(2025, 8, 5))).any()
    # exp_b never has "treatment"/"control" rows, or the exp_a dates.
    assert not ((df["experiment_id"] == "exp_b") & (df["group_id"] != "variant")).any()
    assert not (
        (df["experiment_id"] == "exp_b")
        & (df["ds"].dt.date.isin([dt.date(2025, 8, 1), dt.date(2025, 8, 2)]))
    ).any()
    # Each experiment's own final cumulative matches its own total units.
    exp_a_final = df[(df["experiment_id"] == "exp_a")].sort_values("ds")["n_daily"].sum()
    exp_b_final = df[(df["experiment_id"] == "exp_b")].sort_values("ds")["n_daily"].sum()
    assert exp_a_final == 2
    assert exp_b_final == 1


# Step 2b — metric_events


def test_metric_events_schema(purchase_events, mean_metric):
    """Output columns match the canonical schema."""
    result = metric_events(purchase_events, mean_metric, value_column="amount")
    assert set(result.columns) == METRIC_EVENTS


def test_metric_events_filters_fact(purchase_events, mean_metric):
    """Only rows matching the metric's fact are kept."""
    result = metric_events(purchase_events, mean_metric, value_column="amount").execute()
    # All rows should have the right fact
    events = purchase_events.execute()
    assert len(result) <= len(events)
    # All rows should be purchase events
    purchase_in = purchase_events.filter(purchase_events.event == "purchase").execute()
    assert len(result) == len(purchase_in)


def test_metric_events_occurrence_value(page_view_events, retention_metric):
    """Occurrence-only metrics (e.g., retention) get value = 1 per row."""
    result = metric_events(page_view_events, retention_metric).execute()
    assert (result["value"] == 1.0).all()


# Step 3 — unit_day_panel


def test_unit_day_panel_schema(exposures, purchase_metric_events, experiment, mean_metric):
    """Output columns match the canonical schema (superset)."""
    result = unit_day_panel(
        exposures, purchase_metric_events, experiment, metric_name=mean_metric.name
    )
    assert UNIT_DAY_PANEL.issubset(set(result.columns))


def test_unit_day_panel_dense(exposures, purchase_metric_events, experiment, mean_metric):
    """Every unit × day is present — zero rows included."""
    result = unit_day_panel(
        exposures, purchase_metric_events, experiment, metric_name=mean_metric.name
    ).execute()
    # u1/u2 span Aug 1-6 (6 rows each), u3 Aug 2-6 (5 rows) = 17 rows total
    assert len(result) == 17
    # Check a zero row exists (ds is datetime64 from DuckDB; compare as Timestamp)
    aug1 = result[result["ds"].dt.date == dt.date(2025, 8, 1)]
    zero = aug1[(aug1["unit_id"] == "u2")]
    assert len(zero) == 1
    assert zero["sum_value"].iloc[0] == 0.0


def test_unit_day_panel_excludes_pre_exposure(
    exposures, purchase_metric_events, experiment, mean_metric
):
    """Event before first_exposure_ts (u2/purchase at 09:30) excluded."""
    result = unit_day_panel(
        exposures, purchase_metric_events, experiment, metric_name=mean_metric.name
    ).execute()
    u2_aug1 = result[(result["unit_id"] == "u2") & (result["ds"].dt.date == dt.date(2025, 8, 1))]
    assert len(u2_aug1) == 1
    assert u2_aug1["sum_value"].iloc[0] == 0.0


def test_unit_day_panel_window_days(exposures, purchase_metric_events, experiment, mean_metric):
    """Panel extends beyond window_days but events past window are 0."""
    # u1: purchase on Aug 3 (7.50) — within window_days=3 (ds ∈ [Aug 1, Aug 4) → Aug 3 included)
    result = unit_day_panel(
        exposures, purchase_metric_events, experiment, metric_name=mean_metric.name
    ).execute()
    u1_aug3 = result[(result["unit_id"] == "u1") & (result["ds"].dt.date == dt.date(2025, 8, 3))]
    assert len(u1_aug3) == 1
    assert u1_aug3["sum_value"].iloc[0] == 7.50
    # u1 Aug 4 (past window_days=3) — should be 0
    u1_aug4 = result[(result["unit_id"] == "u1") & (result["ds"].dt.date == dt.date(2025, 8, 4))]
    assert len(u1_aug4) == 1
    assert u1_aug4["sum_value"].iloc[0] == 0.0


def test_unit_day_panel_inferred_metric(exposures, purchase_metric_events, experiment):
    """metric_name=None infers metric column from events table."""
    result = unit_day_panel(exposures, purchase_metric_events, experiment)
    assert UNIT_DAY_PANEL.issubset(set(result.columns))
    df = result.execute()
    # All rows should have the same metric name as the events table
    assert (df["metric"] == "revenue").all()


# Step 4 — unit_totals + group_summary


def test_unit_totals_conversion_01(exposures, purchase_events, experiment, conversion_metric):
    """Conversion metric: y ∈ {0, 1} per unit."""
    events = metric_events(purchase_events, conversion_metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, conversion_metric.name)
    totals = unit_totals(spine, stats, conversion_metric, experiment).execute()
    assert set(totals["y"].unique()) == {0.0, 1.0}
    # u1 has an in-window purchase; u2's purchase is pre-exposure and excluded;
    # u3 has none in window.
    assert totals[totals["unit_id"] == "u1"]["y"].iloc[0] == 1.0
    assert totals[totals["unit_id"] == "u2"]["y"].iloc[0] == 0.0
    assert totals[totals["unit_id"] == "u3"]["y"].iloc[0] == 0.0


def test_unit_totals_mean_sum(exposures, purchase_events, experiment, mean_metric):
    """Mean (sum) metric: y = window total."""
    events = metric_events(purchase_events, mean_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, mean_metric.name)
    totals = unit_totals(spine, stats, mean_metric, experiment).execute()
    # Per-unit sums across the window: u1=69.49, u2=30.00, u3=19.99
    assert abs(totals[totals["unit_id"] == "u1"]["y"].iloc[0] - 69.49) < 1e-9
    assert abs(totals[totals["unit_id"] == "u2"]["y"].iloc[0] - 30.00) < 1e-9
    assert abs(totals[totals["unit_id"] == "u3"]["y"].iloc[0] - 19.99) < 1e-9


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("avg_event", {"u1": 69.49 / 3, "u2": 30.0, "u3": 19.99}),
        ("avg_calendar_day", {"u1": 69.49 / 3, "u2": 30.0 / 3, "u3": 19.99 / 3}),
    ],
)
def test_unit_totals_explicit_average_basis(
    exposures, purchase_events, experiment, aggregation, expected
):
    metric = MeanMetric(
        name="revenue_avg",
        entity="unit_id",
        fact="purchase",
        aggregation=aggregation,
        window_days=3,
    )
    events = metric_events(purchase_events, metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    totals = unit_totals(spine, stats, metric, experiment).execute()
    got = totals.set_index("unit_id")["y"].to_dict()
    assert got == pytest.approx(expected)


def test_winsorize_unit_totals_pools_percentile_across_arms(con):
    totals = ibis.memtable(
        {
            "unit_id": ["c1", "c2", "t1", "t2"],
            "experiment_id": ["e"] * 4,
            "group_id": ["control", "control", "treatment", "treatment"],
            "metric": ["revenue"] * 4,
            "y": [1.0, 2.0, 100.0, 200.0],
            "x": [0.0] * 4,
            "y_den": [0.0] * 4,
            "d": [0.0] * 4,
        }
    )
    metric = MeanMetric(
        name="revenue",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        winsorization=Winsorization(upper_percentile=0.75),
    )

    clipped = winsorize_unit_totals(totals, metric)
    rows = {row["group_id"]: row for row in con.to_pyarrow(group_summary(clipped)).to_pylist()}

    assert rows["control"]["ref_y"] == pytest.approx(1.5)
    assert rows["treatment"]["ref_y"] == pytest.approx(112.5)
    assert rows["treatment"]["winsor_upper_bound"] == pytest.approx(125.0)
    assert rows["treatment"]["winsor_n_upper"] == 1


def test_winsorize_unit_totals_counts_all_eligible_units(con):
    totals = ibis.memtable(
        {
            "unit_id": ["c1", "c2", "t1"],
            "experiment_id": ["e"] * 3,
            "group_id": ["control", "control", "treatment"],
            "metric": ["revenue"] * 3,
            "y": [1.0, 2.0, 20.0],
            "x": [0.0] * 3,
            "y_den": [0.0] * 3,
            "d": [0.0] * 3,
        }
    )
    metric = MeanMetric(
        name="revenue",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        winsorization=Winsorization(upper_value=10.0),
    )

    rows = con.to_pyarrow(winsorize_unit_totals(totals, metric)).to_pylist()

    assert [row["winsor_n"] for row in rows] == [1, 1, 1]
    assert [row["winsor_n_lower"] for row in rows] == [0, 0, 0]
    assert [row["winsor_n_upper"] for row in rows] == [0, 0, 1]


def test_winsorize_unit_totals_percentile_columns_are_always_float64(con):
    """A fixed-value-winsorized metric and an unwinsorized one must carry
    the same winsor-percentile dtype, or unioning their outputs raises."""
    tbl = ibis.memtable({"unit_id": ["u1", "u2", "u3"], "y": [1.0, 2.0, 3.0]})
    winsorized = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        winsorization=Winsorization(upper_value=1000.0),
    )
    plain = MeanMetric(name="revenue_plain", entity="user_id", fact="purchase", aggregation="sum")
    t1 = winsorize_unit_totals(tbl, winsorized)
    t2 = winsorize_unit_totals(tbl, plain)
    assert t1.schema()["winsor_lower_percentile"] == t2.schema()["winsor_lower_percentile"]
    combined = con.to_pyarrow(t1.union(t2))
    assert len(combined) == 6


def test_unit_totals_mean_sum_filter(exposures, purchase_events, experiment):
    """Mean (sum) metric with Filter: only matching events included."""
    filtered_metric = MeanMetric(
        name="revenue_high",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        window_days=3,
        filters=[Filter(property="amount", op="gt", values=[10.0])],
    )
    events = metric_events(purchase_events, filtered_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, filtered_metric.name)
    totals = unit_totals(spine, stats, filtered_metric, experiment).execute()
    totals = totals.set_index("unit_id")
    # u1: Aug 1 amounts 49.99, 12.00 (> 10); Aug 3 amount 7.50 filtered out
    #     49.99 + 12.00 = 61.99
    assert abs(totals.loc["u1", "y"] - 61.99) < 1e-9
    # u2: 30.00 (Aug 2, > 10; Aug 1 pre-exposure filtered by unit_day_panel)
    assert abs(totals.loc["u2", "y"] - 30.00) < 1e-9
    # u3: 19.99 (Aug 3, > 10)
    assert abs(totals.loc["u3", "y"] - 19.99) < 1e-9


def test_unit_totals_mean_count_distinct(exposures, purchase_events, experiment):
    """Mean (count_distinct): nunique of daily sums over OBSERVED days only.

    A zero-filled day is bookkeeping, not a measurement - it must not
    inject a phantom 0.0 into the distinct-value count (before this
    contract, every unit with >=1 eventless in-window day was off by one).
    """
    cd_metric = MeanMetric(
        name="revenue_distinct",
        entity="unit_id",
        fact="purchase",
        aggregation="count_distinct",
        window_days=3,
    )
    events = metric_events(purchase_events, cd_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, cd_metric.name)
    totals = unit_totals(spine, stats, cd_metric, experiment).execute()
    totals = totals.set_index("unit_id")
    # u1: window Aug 1-3, observed daily sums [61.99, 7.50] → nunique = 2
    #     (Aug 2 has no events - the zero-fill is NOT a distinct value)
    assert totals.loc["u1", "y"] == 2.0
    # u2: window Aug 1-3, observed daily sums [30.00] → nunique = 1
    assert totals.loc["u2", "y"] == 1.0
    # u3: window Aug 2-4, observed daily sums [19.99] → nunique = 1
    assert totals.loc["u3", "y"] == 1.0


def test_unit_totals_mean_count_distinct_is_daily_grain(exposures, con, experiment):
    """count_distinct operates on the daily-collapsed panel, not raw events.

    u1 has two same-day events (Aug 1: 10.0, 10.0 -> daily sum 20.0) and one
    event on a different day (Aug 2: 10.0); Aug 3 has no events and does NOT
    contribute (observed days only - no phantom 0.0). Raw-event distinct
    values are {10.0} (nunique=1); daily-collapsed distinct sums are
    {20.0, 10.0} (nunique=2). Asserting y=2.0 locks in both halves of the
    documented semantic: it would fail as 1.0 if a future change computed
    nunique over raw per-event values, and as 3.0 if the dense zero-fill
    leaked back into the distinct set.
    """
    grain_events = con.create_table(
        "grain_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 1),  # not 9:00:00 - avoids colliding
                # with u1's first_exposure_ts (an equal ts is excluded, see unit_day_panel).
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 14, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            },
        ],
    )
    cd_metric = MeanMetric(
        name="revenue_distinct_grain",
        entity="unit_id",
        fact="purchase",
        aggregation="count_distinct",
        window_days=3,
    )
    events = metric_events(grain_events, cd_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, cd_metric.name)
    totals = unit_totals(spine, stats, cd_metric, experiment).execute()
    assert totals[totals["unit_id"] == "u1"]["y"].iloc[0] == 2.0


def test_unit_totals_mean_min_ignores_zero_fill(con):
    """min of an all-positive unit is its real minimum, never a phantom 0.

    The dense spine zero-fills eventless in-window days; letting that 0.0
    into the range made every all-positive unit report min=0. A unit with
    zero in-window events still gets an explicit unit-level 0.0.
    """
    metric = MeanMetric(
        name="rev_min", entity="unit_id", fact="orders", aggregation="min", window_days=3
    )
    spine, stats, exp = _spine_stats_fixture(
        con,
        unit_rows=[("u1", "treatment"), ("u2", "control")],
        exposure_first=dt.date(2025, 8, 1),
        days=10,
        # u1: events on 2 of its 3 window days, all positive. u2: none.
        stat_rows=[
            ("u1", dt.date(2025, 8, 1), 1, 7.0),
            ("u1", dt.date(2025, 8, 2), 1, 3.0),
        ],
    )
    got = unit_totals(spine, stats, metric, exp).execute().set_index("unit_id")
    assert got.loc["u1", "y"] == pytest.approx(3.0)  # not 0.0
    assert got.loc["u2", "y"] == pytest.approx(0.0)  # explicit unit-level zero


def test_unit_totals_mean_max_of_all_negative_unit_is_negative(con):
    """max of an all-negative fact (refunds) is the real (negative) max.

    Before the observed-days contract the zero-filled day won the max and
    reported 0.0 for a unit whose raw max is -3.0.
    """
    metric = MeanMetric(
        name="refund_max", entity="unit_id", fact="orders", aggregation="max", window_days=3
    )
    spine, stats, exp = _spine_stats_fixture(
        con,
        unit_rows=[("u1", "treatment"), ("u2", "control")],
        exposure_first=dt.date(2025, 8, 1),
        days=10,
        stat_rows=[
            ("u1", dt.date(2025, 8, 1), 1, -7.0),
            ("u1", dt.date(2025, 8, 2), 1, -3.0),
        ],
    )
    got = unit_totals(spine, stats, metric, exp).execute().set_index("unit_id")
    assert got.loc["u1", "y"] == pytest.approx(-3.0)  # not 0.0
    assert got.loc["u2", "y"] == pytest.approx(0.0)


def test_unit_totals_mean_count_events(exposures, page_view_events, experiment):
    """Mean (count) aggregation: y counts actual events, not panel days.

    Regression test: old bug used panel.value.count() which counted non-null
    panel rows (= days in window due to COALESCE(value, 0)) instead of actual
    event count.  Using page_view (occurrence fact) where events < panel days
    for every unit exposes the difference.
    """
    count_metric = MeanMetric(
        name="page_views",
        entity="unit_id",
        fact="page_view",
        aggregation="count",
        window_days=3,
    )
    events = metric_events(page_view_events, count_metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, count_metric.name)
    totals = unit_totals(spine, stats, count_metric, experiment).execute()
    totals = totals.set_index("unit_id")
    # u1: exposed Aug 1, page_views on Aug 1 (1) and Aug 3 (1) → 2 events
    #     Panel days in window (Aug 1-3): 3.  Old bug: count=3.  Correct: 2.
    assert totals.loc["u1", "y"] == 2.0
    # u2: exposed Aug 1, page_view on Aug 2 (1) → 1 event.  Panel days: 3.
    assert totals.loc["u2", "y"] == 1.0
    # u3: exposed Aug 2, page_view on Aug 3 (1) → 1 event.  Panel days: 3.
    assert totals.loc["u3", "y"] == 1.0


# unit_totals derives from spine + stats tables directly


def _spine_stats_fixture(con, *, unit_rows, exposure_first, days, stat_rows):
    """Build a (spine, stats) pair directly, bypassing raw events.

    unit_rows: list of (unit_id, group_id). exposure_first: dict unit_id ->
    first_exposure date (default `exposure_first` applied to every unit not
    listed). stat_rows: list of (unit_id, ds, n_events, sum_value).
    """
    from increment.query.builders import panel_spine

    exposures_rows = [
        {
            "unit_id": u,
            "experiment_id": "exp_test",
            "group_id": g,
            "first_exposure_ts": dt.datetime.combine(exposure_first, dt.time(9, 0)),
        }
        for u, g in unit_rows
    ]
    # `id()` is only unique among live objects: CPython reuses an address once
    # an earlier object is freed, which collided with a prior table here.
    exposures_tbl = con.create_table(f"t3_exposures_{uuid.uuid4().hex}", obj=exposures_rows)
    exp = Experiment(
        name="exp_test",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime.combine(exposure_first, dt.time(0, 0)),
        end=dt.datetime.combine(exposure_first, dt.time(0, 0)) + dt.timedelta(days=days + 30),
        plan=AnalysisPlan(),
    )
    end_date = ibis.literal(exposure_first + dt.timedelta(days=days))
    spine = panel_spine(exposures_tbl, exp, end_date=end_date)
    stat_dicts = [
        {
            "unit_id": u,
            "ds": ds,
            "source_key": "s",
            "n_events": n,
            "sum_value": v,
            "min_value": v,
            "max_value": v,
        }
        for u, ds, n, v in stat_rows
    ]
    stats_tbl = con.create_table(f"t3_stats_{uuid.uuid4().hex}", obj=stat_dicts)
    return spine, stats_tbl, exp


def test_unit_totals_zero_fills_from_spine_not_storage(con):
    """A unit with no events in the window gets y=0, not a missing row."""
    metric = MeanMetric(
        name="revenue", entity="unit_id", fact="orders", aggregation="sum", window_days=14
    )
    spine, stats, exp = _spine_stats_fixture(
        con,
        unit_rows=[("u1", "treatment"), ("u2", "control")],
        exposure_first=dt.date(2026, 1, 1),
        days=3,
        stat_rows=[("u1", dt.date(2026, 1, 1), 1, 25.0)],  # u2: no events at all
    )

    got = {
        r["unit_id"]: r for r in con.to_pyarrow(unit_totals(spine, stats, metric, exp)).to_pylist()
    }

    assert set(got) == {"u1", "u2"}, "u2 must survive with a zero, not vanish"
    assert got["u1"]["y"] == pytest.approx(25.0)
    assert got["u2"]["y"] == pytest.approx(0.0)


def test_conversion_and_retention_read_occurrence_from_n_events(con):
    """value=0 for a real $0 event must not read as 'no event'."""
    conv = ConversionMetric(name="converted", entity="unit_id", fact="orders", window_days=14)
    ret = RetentionMetric(name="d7", entity="unit_id", fact="orders", threshold_days=(7, 14))

    spine, stats, exp = _spine_stats_fixture(
        con,
        unit_rows=[("u1", "treatment"), ("u2", "control")],
        exposure_first=dt.date(2026, 1, 1),
        days=10,
        stat_rows=[
            ("u1", dt.date(2026, 1, 1), 1, 0.0),  # a real event worth 0.00
            ("u2", dt.date(2026, 1, 9), 1, 5.0),  # after the d7 threshold
        ],
    )

    c = {
        r["unit_id"]: r["y"]
        for r in con.to_pyarrow(unit_totals(spine, stats, conv, exp)).to_pylist()
    }
    assert c["u1"] == 1.0, "a $0 event still converted -- n_events, not sum_value"
    assert c["u2"] == 1.0

    r = {
        row["unit_id"]: row["y"]
        for row in con.to_pyarrow(unit_totals(spine, stats, ret, exp)).to_pylist()
    }
    assert r["u1"] == 0.0, "day 0 is before the d7 threshold"
    assert r["u2"] == 1.0, "day 8 is on/after the threshold"


def test_unit_totals_schema(exposures, purchase_events, experiment, conversion_metric):
    """Output columns match canonical schema."""
    events = metric_events(purchase_events, conversion_metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, conversion_metric.name)
    totals = unit_totals(spine, stats, conversion_metric, experiment)
    assert set(totals.columns) == UNIT_TOTALS


def test_group_summary_schema(totals):
    """Output columns match canonical schema."""
    summary = group_summary(totals)
    assert set(summary.columns) == GROUP_SUMMARY


def _arrow_rows(con, expr) -> tuple[pa.Table, list[dict]]:
    table = con.to_pyarrow(expr)
    return table, table.to_pylist()


def test_group_summary_successes_is_exact_integer_for_declared_binary(
    con, exposures, purchase_events, experiment, conversion_metric
):
    """A declared conversion metric carries its exact int64 success count."""
    events = metric_events(purchase_events, conversion_metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, conversion_metric.name)
    totals = unit_totals(spine, stats, conversion_metric, experiment)
    expected: dict[str, int] = {}
    for row in con.to_pyarrow(totals).to_pylist():
        expected[row["group_id"]] = expected.get(row["group_id"], 0) + int(row["y"])
    table, rows = _arrow_rows(
        con, group_summary(totals, binary_metrics=declared_binary_metrics([conversion_metric]))
    )
    assert table.schema.field("successes").type == pa.int64()
    assert {row["group_id"]: row["successes"] for row in rows} == expected
    assert all(isinstance(row["n"], int) for row in rows)


def test_group_summary_successes_null_without_binary_declaration(
    con, exposures, purchase_events, experiment, conversion_metric, totals, mean_metric
):
    """No declaration, a different declared name, or a nonbinary metric: NULL, never 0."""
    events = metric_events(purchase_events, conversion_metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, conversion_metric.name)
    conversion_totals = unit_totals(spine, stats, conversion_metric, experiment)
    undeclared = con.to_pyarrow(group_summary(conversion_totals))
    assert undeclared.schema.field("successes").type == pa.int64()
    assert undeclared.column("successes").null_count == undeclared.num_rows
    other = con.to_pyarrow(group_summary(conversion_totals, binary_metrics=["other"]))
    assert other.column("successes").null_count == other.num_rows
    mean = con.to_pyarrow(
        group_summary(totals, binary_metrics=declared_binary_metrics([mean_metric]))
    )
    assert mean.schema.field("successes").type == pa.int64()
    assert mean.column("successes").null_count == mean.num_rows


def test_group_summary_centered_moments(totals):
    """group_summary emits PURE centered moments — no variance.
    Cross-check: cy2 / (n−1) == np.var(ddof=1).
    """
    summary = group_summary(totals).execute()
    # treatment: y = [69.49, 19.99]
    # control: y = [30.00]
    trt = summary[summary["group_id"] == "treatment"].iloc[0]
    ctrl = summary[summary["group_id"] == "control"].iloc[0]
    trt_y = np.array([69.49, 19.99])
    trt_ref = trt_y.mean()
    assert trt["n"] == 2
    assert abs(trt["ref_y"] - trt_ref) < 1e-9
    assert abs(trt["cy1"] - (trt_y - trt_ref).sum()) < _cancel_tol(2, trt_ref)
    assert abs(trt["cy2"] - ((trt_y - trt_ref) ** 2).sum()) < 1e-6
    # The first moment stays exactly recoverable: n*ref_y + cy1.
    assert abs(_total(trt) - (69.49 + 19.99)) < 1e-9
    assert ctrl["n"] == 1
    assert abs(ctrl["ref_y"] - 30.00) < 1e-9
    assert abs(ctrl["cy1"]) < _cancel_tol(1, 30.00)
    assert abs(ctrl["cy2"]) < 1e-9

    # Cross-check variance: var = cy2 / (n−1)
    var_from_moments = trt["cy2"] / (trt["n"] - 1)
    np_var = np.var(trt_y, ddof=1)
    assert abs(var_from_moments - np_var) < 1e-9


def test_group_summary_x_moments_null_when_unmaterialized(totals):
    """x-den moments are None (not 0) when CUPED is not enabled."""
    summary = group_summary(totals).execute()
    trt = summary[summary["group_id"] == "treatment"].iloc[0]
    # the whole covariate family is null — check pandas NA
    for field in ("ref_x", "cx1", "cx2", "cxy"):
        value = trt[field]
        assert value is None or (isinstance(value, float) and np.isnan(value))


# Step 4b — type-aggregation dispatch + windowed censoring


def test_retention_metric_dispatch(exposures, page_view_events, experiment, retention_metric):
    """Retention metric: y ∈ {0,1}, active on/after threshold_days."""
    events = metric_events(page_view_events, retention_metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, retention_metric.name)
    totals = unit_totals(spine, stats, retention_metric, experiment).execute()
    assert set(totals.columns) == UNIT_TOTALS
    # u1's page_view clears its Aug1+2 threshold; u2/u3's land before their
    # own thresholds.
    assert totals[totals["unit_id"] == "u1"]["y"].iloc[0] == 1.0
    assert totals[totals["unit_id"] == "u2"]["y"].iloc[0] == 0.0
    assert totals[totals["unit_id"] == "u3"]["y"].iloc[0] == 0.0


def test_censoring_fence_keeps_unit_whose_last_window_day_is_observable(
    censored_exposure_events, censored_purchase_events, censored_experiment, conversion_metric
):
    """Conversion, window_days=1: a unit exposed ON the observable bound is
    KEPT - its whole window ``[fe, fe+1)`` is the exposure day itself,
    fully observed through ``end``. The fence compares the LAST window day
    (``fe + window_days - 1``) against the bound, never the exclusive
    right edge, which dropped fully-observed units one day early.
    """
    from increment.query.builders import metric_events, unit_day_spine_stats, unit_totals

    exposures = first_exposures(censored_exposure_events, censored_experiment)
    events = metric_events(censored_purchase_events, conversion_metric)
    spine, stats = unit_day_spine_stats(
        exposures, events, censored_experiment, conversion_metric.name
    )
    totals = unit_totals(spine, stats, conversion_metric, censored_experiment).execute()
    # u5's window [Aug 3, Aug 4) has its last day == experiment.end, so the fence
    # treats it as final (KEPT), and its 13:00 purchase scores y=1.
    assert sorted(totals["unit_id"].tolist()) == ["u1", "u2", "u5"]
    assert totals.set_index("unit_id").loc["u5", "y"] == 1.0


def test_censoring_drops_late_enrollee_conversion(
    censored_exposure_events, censored_purchase_events, censored_experiment
):
    """Conversion: late-enrollee whose window genuinely outruns ``end`` is
    dropped - window_days=2 puts u5's last window day (Aug 4) past the
    Aug 3 bound."""
    from increment.query.builders import metric_events, unit_day_spine_stats, unit_totals

    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=2)
    exposures = first_exposures(censored_exposure_events, censored_experiment)
    events = metric_events(censored_purchase_events, metric)
    spine, stats = unit_day_spine_stats(exposures, events, censored_experiment, metric.name)
    with pytest.warns(IncrementWarning) as rec:
        totals = unit_totals(spine, stats, metric, censored_experiment).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    # u5 enrolled Aug 3; last window day Aug 4 > experiment.end Aug 3 → DROPPED
    assert "u5" not in totals["unit_id"].values
    assert sorted(totals["unit_id"].tolist()) == ["u1", "u2"]


def test_censoring_drops_late_enrollee_retention(
    censored_exposure_events, censored_page_view_events, censored_experiment, retention_metric
):
    """Retention: late-enrollee (threshold past end) dropped."""
    exposures = first_exposures(censored_exposure_events, censored_experiment)
    events = metric_events(censored_page_view_events, retention_metric)
    spine, stats = unit_day_spine_stats(
        exposures, events, censored_experiment, retention_metric.name
    )
    with pytest.warns(IncrementWarning) as rec:
        totals = unit_totals(spine, stats, retention_metric, censored_experiment).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    # u5 enrolled on Aug 3, threshold_days=2 → window Aug 5 > experiment.end Aug 3 → DROPPED
    assert "u5" not in totals["unit_id"].values
    assert sorted(totals["unit_id"].tolist()) == ["u1", "u2"]


def test_unit_totals_never_admits_a_unit_whose_window_outruns_the_data(con):
    """A declared end past the available data must not score units on absent events.

    u1's window closes 2025-08-15, but the event log stops on 2025-08-05
    (passed explicitly as data_as_of - see this task's brief for why the
    bound is not inferred from event occurrence). Admitting u1 would score
    it y=0 for a return that simply has not been observed yet, which is
    indistinguishable from a real non-return.
    """
    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    experiment = Experiment(
        name="exp_test",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 31),  # well past the data
        plan=AnalysisPlan(),
    )
    exposure_rows = con.create_table(
        "avail_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    purchase_rows = con.create_table(
        "avail_purchases",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 5, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    with pytest.warns(IncrementWarning) as rec:
        totals = unit_totals(
            spine, stats, metric, experiment, data_as_of=dt.datetime(2025, 8, 5)
        ).execute()

    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert len(totals) == 0, (
        "u1's 14-day window closes 2025-08-15 but data is only loaded through "
        "2025-08-05; it must be censored, not scored on unobserved days"
    )


def test_unit_totals_data_as_of_none_applies_no_cap(con):
    """Without an explicit data_as_of, censoring uses only the declared horizon (today's default)."""
    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    experiment = Experiment(
        name="exp_test",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 31),
        observation_end=dt.datetime(2025, 8, 31),
        plan=AnalysisPlan(),
    )
    exposure_rows = con.create_table(
        "nodataasof_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    purchase_rows = con.create_table(
        "nodataasof_purchases",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 5, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    totals = unit_totals(spine, stats, metric, experiment).execute()  # data_as_of omitted

    assert len(totals) == 1
    assert totals["y"].iloc[0] == 1.0


def test_observation_end_admits_a_unit_whose_window_closes_after_the_experiment(con):
    """A late enrollee is scored when observation is declared to run past `end`.

    u_late enrols 2025-08-05, one day before enrollment closes; its 14-day
    window closes 2025-08-19. With observation_end unset it is censored
    (today's behaviour). With observation_end=2025-08-20 it is scored, and
    its purchase on 2025-08-12 - after the experiment ended - counts.
    data_as_of is intentionally omitted here: this test is about the
    declared horizon, not data freshness (that's Step 3's job).
    """
    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    exposure_rows = con.create_table(
        "obs_exposures",
        obj=[
            {
                "unit_id": "u_late",
                "ts": dt.datetime(2025, 8, 5, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    purchase_rows = con.create_table(
        "obs_purchases",
        obj=[
            {
                "unit_id": "u_late",
                "ts": dt.datetime(2025, 8, 12, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )

    def _totals(observation_end):
        experiment = Experiment(
            name="exp_test",
            unit="unit_id",
            exposure="test_exposure",
            control_group="control",
            start=dt.datetime(2025, 8, 1),
            end=dt.datetime(2025, 8, 6),
            observation_end=observation_end,
            plan=AnalysisPlan(),
        )
        exposures = first_exposures(exposure_rows, experiment)
        events = metric_events(purchase_rows, metric)
        spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
        return unit_totals(spine, stats, metric, experiment).execute()

    with pytest.warns(IncrementWarning) as rec:
        censored = _totals(None)
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert len(censored) == 0, "today's behaviour: window outruns `end`, unit is censored"

    admitted = _totals(dt.datetime(2025, 8, 20))
    assert len(admitted) == 1
    assert admitted["y"].iloc[0] == 1.0, "the post-experiment purchase must count"


def test_unit_totals_warns_when_censoring_drops_a_material_share(con):
    """Dropping most of the enrolled population must not be silent.

    No data_as_of given - the declared horizon (experiment.end, since
    observation_end is unset) is the only bound, so the message must point
    at observation_end.
    """
    import pytest

    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    experiment = Experiment(
        name="exp_test",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 20),
        plan=AnalysisPlan(),
    )
    # 1 unit matures by 2025-08-20; 9 do not.
    rows = [
        {
            "unit_id": "u_early",
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_test",
            "group_id": "treatment",
        }
    ]
    rows += [
        {
            "unit_id": f"u_late{i}",
            "ts": dt.datetime(2025, 8, 15, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_test",
            "group_id": "treatment",
        }
        for i in range(9)
    ]
    exposure_rows = con.create_table("warn_exposures", obj=rows)
    purchase_rows = con.create_table(
        "warn_purchases",
        obj=[
            {
                "unit_id": "u_early",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    with pytest.warns(IncrementWarning) as rec:
        unit_totals(spine, stats, metric, experiment).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    cause = str(warning_context(rec, "frame.censoring.dropped_units")["cause"])
    assert "observation_end" in cause


def test_unit_totals_warns_with_data_as_of_when_that_is_the_binding_bound(con):
    """When data freshness - not the declared horizon - is what's dropping units,
    the message must point at data_as_of, not observation_end (changing observation_end
    would not fix anything here - the data simply hasn't loaded yet)."""
    import pytest

    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    experiment = Experiment(
        name="exp_test",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 1),
        observation_end=dt.datetime(2025, 9, 1),  # generous declared horizon
        plan=AnalysisPlan(),
    )
    rows = [
        {
            "unit_id": "u_early",
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_test",
            "group_id": "treatment",
        }
    ]
    rows += [
        {
            "unit_id": f"u_late{i}",
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_test",
            "group_id": "treatment",
        }
        for i in range(9)
    ]
    exposure_rows = con.create_table("dataasof_warn_exposures", obj=rows)
    purchase_rows = con.create_table(
        "dataasof_warn_purchases",
        obj=[
            {
                "unit_id": "u_early",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    # Data loaded only through 2025-08-05, before the 14-day window closes for
    # the late-enrolled units - triggers despite observation_end allowing it.
    with pytest.warns(IncrementWarning) as rec:
        unit_totals(spine, stats, metric, experiment, data_as_of=dt.datetime(2025, 8, 5)).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    cause = str(warning_context(rec, "frame.censoring.dropped_units")["cause"])
    assert "data_as_of" in cause


def test_unit_totals_does_not_warn_when_nothing_is_censored(con, experiment):
    """No drop, no noise."""
    import warnings

    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=1)
    exposure_rows = con.create_table(
        "quiet_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    purchase_rows = con.create_table(
        "quiet_purchases",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        unit_totals(spine, stats, metric, experiment).execute()


def test_unit_totals_warns_running_experiment_does_not_suggest_observation_end(con):
    """Running experiment (no ``end`` declared at all): censoring falls
    back to the panel's own extent, not a declared horizon. The warning
    must not blame - or suggest changing - observation_end: there IS no
    declared horizon here, and "set observation_end" would newly admit
    units whose data hasn't arrived, reproducing the exact bug this
    mechanism exists to prevent.
    """
    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    experiment = Experiment(
        name="exp_running_warn",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=None,
        plan=AnalysisPlan(),
    )
    # end=None derives the panel extent from the single observed purchase
    # (2025-08-03); every unit's 14-day window closes well after that, so all are censored.
    rows = [
        {
            "unit_id": f"u{i}",
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_running_warn",
            "group_id": "treatment",
        }
        for i in range(10)
    ]
    exposure_rows = con.create_table("running_warn_exposures", obj=rows)
    purchase_rows = con.create_table(
        "running_warn_purchases",
        obj=[
            {
                "unit_id": "u0",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    # observation_horizon=None -> observable_end falls back to panel.ds.max(),
    # trimmed to the latest observed purchase; every unit's window closes well after that.
    import pytest

    with pytest.warns(IncrementWarning) as record:
        unit_totals(spine, stats, metric, experiment).execute()

    message = str(record[0].message)
    assert "observation_end" not in message, message
    assert "set" not in message, message
    assert "running" in message, message


def test_unit_totals_running_ratio_warning_names_denominator_lag(con):
    """Running experiment + RatioMetric: the censoring warning must flag
    a lagging denominator fact as another possible cause.

    A running experiment's observable bound is the spine's own extent of
    observed event dates, not pipeline freshness - and a denominator
    fact lagging behind the numerator can never shrink that bound, so a
    denominator data delay is invisible to it and the plain "still
    running" wording would tell the user not to investigate. The ratio
    wording must name both facts so the delay is checked, not waited out.
    """
    metric = RatioMetric(
        name="revenue_per_pageview",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=14),
        denominator=Measure(fact="page_view", aggregation="count", window_days=14),
    )
    experiment = Experiment(
        name="exp_running_ratio_warn",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=None,
        plan=AnalysisPlan(),
    )
    # Same shape as the non-ratio running-experiment case: numerator extent
    # ends at the single purchase (Aug 3), every 14-day window closes Aug 15 -> censored.
    rows = [
        {
            "unit_id": f"u{i}",
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_running_ratio_warn",
            "group_id": "treatment",
        }
        for i in range(10)
    ]
    exposure_rows = con.create_table("running_ratio_warn_exposures", obj=rows)
    purchase_rows = con.create_table(
        "running_ratio_warn_purchases",
        obj=[
            {
                "unit_id": "u0",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    # Denominator fact lags the numerator by a day - exactly the state
    # the numerator-only bound cannot see.
    page_view_rows = con.create_table(
        "running_ratio_warn_page_views",
        obj=[
            {
                "unit_id": "u0",
                "ts": dt.datetime(2025, 8, 2, 10, 0, 0),
                "event": "page_view",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    num_events = metric_events(purchase_rows, metric)
    den_events = metric_events(page_view_rows, metric, part="denominator")
    spine, stats = unit_day_spine_stats(exposures, num_events, experiment, metric.name)
    den_stats = post_exposure_stats(den_events, exposures, source_key="den")

    with pytest.warns(
        UserWarning,
        match=r"denominator fact \('page_view'\) lagging behind the numerator",
    ) as record:
        unit_totals(spine, stats, metric, experiment, den_stats=den_stats).execute()

    message = str(record[0].message)
    # The running-experiment attribution is still the headline cause...
    assert "the experiment still running" in message, message
    # ...and the bound's observed-event provenance is spelled out.
    assert "reflects observed event dates" in message, message
    assert "observation_end" not in message, message


def test_unit_totals_warns_tie_break_attributes_to_horizon_not_data_as_of(con):
    """At an exact tie (``as_of_date == horizon_date``), extending
    observation_end further would not move observable_end (still capped
    by data_as_of at the same date) - so the declared horizon, not data
    freshness, is the actionable cause. Regression for the strict
    less-than in the attribution branch.
    """
    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=14)
    experiment = Experiment(
        name="exp_tie_warn",
        unit="unit_id",
        exposure="test_exposure",
        control_group="control",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 20),
        plan=AnalysisPlan(),
    )
    # 1 unit matures by 2025-08-20; 9 do not.
    rows = [
        {
            "unit_id": "u_early",
            "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_tie_warn",
            "group_id": "treatment",
        }
    ]
    rows += [
        {
            "unit_id": f"u_late{i}",
            "ts": dt.datetime(2025, 8, 15, 9, 0, 0),
            "event": "test_exposure",
            "experiment_id": "exp_tie_warn",
            "group_id": "treatment",
        }
        for i in range(9)
    ]
    exposure_rows = con.create_table("tie_warn_exposures", obj=rows)
    purchase_rows = con.create_table(
        "tie_warn_purchases",
        obj=[
            {
                "unit_id": "u_early",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),
                "event": "purchase",
                "value": 1.0,
            }
        ],
    )
    exposures = first_exposures(exposure_rows, experiment)
    events = metric_events(purchase_rows, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)

    import pytest

    # data_as_of exactly equals the declared horizon (2025-08-20). Absence of
    # "data_as_of" in the cause is the discriminator against a regression to a non-strict "<=" comparison.
    with pytest.warns(IncrementWarning) as rec:
        unit_totals(spine, stats, metric, experiment, data_as_of=dt.datetime(2025, 8, 20)).execute()

    cause = str(warning_context(rec, "frame.censoring.dropped_units")["cause"])
    assert "declared observation horizon" in cause
    assert "data_as_of" not in cause


# Step 4c — timezone/date handling


def test_local_date_timestamptz_is_session_timezone_independent():
    """A timezone-AWARE exposure timestamp must bucket to the SAME exposure
    day under any backend session timezone - the declared ``day_boundary``
    decides the day, never ``SET TimeZone``.

    A bare ``CAST(ts AS DATE)`` on a TIMESTAMPTZ column localizes through
    the session zone: ``2025-03-06 02:00Z`` reads as March 5 under a
    ``America/Los_Angeles`` session and March 6 under a ``UTC`` one. The
    July event additionally straddles a DST transition relative to the
    epoch, catching any normalization that diffs in local wall-clock
    space instead of on instants.
    """
    con = ibis.duckdb.connect()
    con.raw_sql(
        "CREATE TABLE tz_events (unit_id VARCHAR, ts TIMESTAMPTZ, "
        "experiment_id VARCHAR, group_id VARCHAR)"
    )
    con.raw_sql(
        "INSERT INTO tz_events VALUES "
        "('a', '2025-03-06 02:00:00+00', 'tz_exp', 'treatment'), "
        "('b', '2025-07-06 23:30:00+00', 'tz_exp', 'treatment')"
    )
    events = con.table("tz_events")
    assert events.ts.type().timezone is not None  # the fixture really is tz-aware

    def enrollment(day_boundary: str) -> Experiment:
        return Experiment(
            name="tz_exp",
            unit="unit_id",
            start=dt.datetime(2025, 1, 1),
            end=dt.datetime(2025, 8, 1),
            control_group="control",
            exposure="test_exposure",
            day_boundary=day_boundary,
            plan=AnalysisPlan(),
        )

    def days_by_session(experiment: Experiment) -> dict[str, list[dt.date]]:
        """Exposure days per session timezone - they must not differ."""
        counts = daily_exposure_counts(first_exposures(events, experiment), experiment)
        by_session = {}
        for session_tz in ("UTC", "America/Los_Angeles"):
            con.raw_sql(f"SET TimeZone='{session_tz}'")
            by_session[session_tz] = sorted(counts.execute()["ds"].dt.date.tolist())
        con.raw_sql("SET TimeZone='UTC'")
        return by_session

    # Declared zero-offset boundary: both events keep their UTC calendar day.
    utc_days = days_by_session(enrollment("UTC"))
    assert utc_days["UTC"] == utc_days["America/Los_Angeles"]
    assert utc_days["UTC"] == [dt.date(2025, 3, 6), dt.date(2025, 7, 6)]

    # Declared UTC-05:00 boundary: 02:00Z is 21:00 the previous local day;
    # 23:30Z is 18:30 the same local day.
    offset_days = days_by_session(enrollment("UTC-05:00"))
    assert offset_days["UTC"] == offset_days["America/Los_Angeles"]
    assert offset_days["UTC"] == [dt.date(2025, 3, 5), dt.date(2025, 7, 6)]


def test_unit_totals_window_beyond_spine_capacity_is_complete(con):
    """A long window includes events past the historical 730-day spine bound."""
    from increment.query.builders import unit_day_spine_stats, unit_totals

    experiment = Experiment(
        name="exp_cap",
        unit="unit_id",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 10),
        observation_end=dt.datetime(2027, 6, 1),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(
        name="rev_800", entity="unit_id", fact="purchase", aggregation="sum", window_days=800
    )
    fe = dt.datetime(2025, 1, 1, 9)
    exposures_t = ibis.memtable(
        [
            {
                "unit_id": "u1",
                "experiment_id": "exp_cap",
                "group_id": "treatment",
                "first_exposure_ts": fe,
            }
        ]
    )
    events_t = ibis.memtable(
        [
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=3, hours=1), "value_col": 10.0},
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=750, hours=1), "value_col": 90.0},
        ]
    )
    events_t = events_t.mutate(event=ibis.literal("purchase"))
    events = metric_events(events_t, metric, value_column="value_col")
    spine, stats = unit_day_spine_stats(exposures_t, events, experiment, metric.name)
    result = unit_totals(spine, stats, metric, experiment, warn_on_censoring=False).execute()

    assert result["unit_id"].tolist() == ["u1"]
    assert result["y"].iloc[0] == 100.0


def test_unit_totals_unwindowed_horizon_includes_late_events(con):
    """An unwindowed metric includes all events through a declared horizon."""
    experiment = Experiment(
        name="exp_unwindowed_horizon",
        unit="unit_id",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 10),
        observation_end=dt.datetime(2027, 6, 1),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(
        name="rev_unwindowed",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
    )
    fe = dt.datetime(2025, 1, 1, 9)
    exposures_t = ibis.memtable(
        [
            {
                "unit_id": "u1",
                "experiment_id": "exp_unwindowed_horizon",
                "group_id": "treatment",
                "first_exposure_ts": fe,
            }
        ]
    )
    events_t = ibis.memtable(
        [
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=3, hours=1), "value_col": 10.0},
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=750, hours=1), "value_col": 90.0},
        ]
    )
    events_t = events_t.mutate(event=ibis.literal("purchase"))
    events = metric_events(events_t, metric, value_column="value_col")
    spine, stats = unit_day_spine_stats(exposures_t, events, experiment, metric.name)
    result = unit_totals(spine, stats, metric, experiment, warn_on_censoring=False).execute()

    assert result["unit_id"].tolist() == ["u1"]
    assert result["y"].iloc[0] == 100.0


def test_unit_totals_window_at_spine_capacity_is_served(con):
    """A 730-day window excludes day 750 by its window, not spine construction."""
    from increment.query.builders import unit_day_spine_stats, unit_totals

    experiment = Experiment(
        name="exp_cap_fit",
        unit="unit_id",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 10),
        observation_end=dt.datetime(2027, 6, 1),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(
        name="rev_730", entity="unit_id", fact="purchase", aggregation="sum", window_days=730
    )
    fe = dt.datetime(2025, 1, 1, 9)
    exposures_t = ibis.memtable(
        [
            {
                "unit_id": "u1",
                "experiment_id": "exp_cap_fit",
                "group_id": "treatment",
                "first_exposure_ts": fe,
            }
        ]
    )
    events_t = ibis.memtable(
        [
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=3, hours=1), "value_col": 10.0},
            {"unit_id": "u1", "ts": fe + dt.timedelta(days=750, hours=1), "value_col": 90.0},
        ]
    )
    events_t = events_t.mutate(event=ibis.literal("purchase"))
    events = metric_events(events_t, metric, value_column="value_col")
    spine, stats = unit_day_spine_stats(exposures_t, events, experiment, metric.name)
    df = unit_totals(spine, stats, metric, experiment, warn_on_censoring=False).execute()
    assert df["unit_id"].tolist() == ["u1"]
    assert df["y"].iloc[0] == 10.0


# daily_group_summary


def test_daily_group_summary_schema(panel, mean_metric):
    """Output columns match canonical schema."""
    dgs = daily_group_summary(panel, metric=mean_metric)
    assert set(dgs.columns) == DAILY_GROUP_SUMMARY


def test_daily_group_summary_monitoring_only(panel, mean_metric):
    """Daily group summary: per-ds means for trend charts."""
    dgs = daily_group_summary(panel, metric=mean_metric).execute()
    # Aug 1: treatment has u1 (61.99), control has u2 (0)
    aug1 = dgs[dgs["ds"].dt.date == dt.date(2025, 8, 1)]
    trt_d1 = aug1[aug1["group_id"] == "treatment"]
    ctrl_d1 = aug1[aug1["group_id"] == "control"]
    assert trt_d1["n"].iloc[0] == 1
    assert abs(trt_d1["ref_y"].iloc[0] - 61.99) < 1e-9
    assert trt_d1["cy1"].iloc[0] == 0.0  # n=1: the residual is exactly 0
    assert ctrl_d1["n"].iloc[0] == 1
    assert ctrl_d1["ref_y"].iloc[0] == 0.0
    assert ctrl_d1["cy1"].iloc[0] == 0.0


# Step 6 — cross-backend render test


def test_cross_backend_snowflake_sql(exposures, purchase_events, experiment, mean_metric):
    """The full summary chain compiles to Snowflake SQL."""
    events = metric_events(purchase_events, mean_metric, value_column="amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, mean_metric.name)
    totals = unit_totals(spine, stats, mean_metric, experiment)
    summary = group_summary(totals)
    ibis.to_sql(summary, dialect="snowflake")


def test_panel_renders_snowflake_sql(exposures, purchase_events, experiment, mean_metric):
    """The panel stage compiles to Snowflake SQL."""
    events = metric_events(purchase_events, mean_metric, value_column="amount")
    panel = unit_day_panel(exposures, events, experiment, metric_name=mean_metric.name)
    ibis.to_sql(panel, dialect="snowflake")


def test_panel_bounded_when_experiment_end_is_none(exposures, page_view_events, experiment):
    """A running panel trims each unit to the latest observed event date.

    ``end_date`` is computed from ``events.ts.max()`` and must bound the
    dynamically sized spine when ``experiment.end`` is None.
    """
    open_experiment = experiment.model_copy(update={"end": None})
    count_metric = MeanMetric(
        name="page_views",
        entity="unit_id",
        fact="page_view",
        aggregation="count",
    )
    events = metric_events(page_view_events, count_metric)
    panel = unit_day_panel(
        exposures, events, open_experiment, metric_name=count_metric.name
    ).execute()
    # page_view_events fixture spans early August 2025; the panel must not
    # extend anywhere near a 730-day spine from first exposure.
    span_days = (panel["ds"].max() - panel["ds"].min()).days
    assert span_days < 30


def test_panel_dense_when_events_empty_and_experiment_end_none(
    exposures, page_view_events, experiment
):
    """Zero-event metric with experiment.end=None produces an EMPTY panel:
    events.ts.max() is NULL on an empty table, so every enrolled unit gets
    a null-date spine row instead of a phantom populated day -- enrollment
    identity alone is not evidence any calendar day was observed (D1)."""
    open_experiment = experiment.model_copy(update={"end": None})
    no_event_metric = MeanMetric(
        name="never_fires",
        entity="unit_id",
        fact="page_view",
        aggregation="count",
        filters=[Filter(property="unit_id", op="equals", values=["nonexistent-unit"])],
    )
    empty_events = metric_events(page_view_events, no_event_metric)
    panel = unit_day_panel(
        exposures, empty_events, open_experiment, metric_name=no_event_metric.name
    ).execute()
    assert len(panel) == 0


# Breakout dimension (`by` / `properties_table` / `den_panel`)


def test_unit_totals_by_carries_dimension_and_null_becomes_sentinel(
    exposures, purchase_metric_events, mean_metric, experiment, con
):
    """unit_totals(by=[...], properties_table=...) carries the dimension
    column through to the output; a unit missing from properties_table
    gets the "__null__" sentinel bin instead of NULL or being
    dropped."""
    props = con.create_table(
        "breakout_props_unit_totals",
        obj=[
            {"unit_id": "u1", "country": "US"},
            {"unit_id": "u2", "country": "US"},
            # u3 intentionally absent -> NULL after the left join
        ],
    )
    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    totals = unit_totals(
        spine, stats, mean_metric, experiment, by=["country"], properties_table=props
    ).execute()
    assert "country" in totals.columns
    totals = totals.set_index("unit_id")
    assert totals.loc["u1", "country"] == "US"
    assert totals.loc["u2", "country"] == "US"
    assert totals.loc["u3", "country"] == "__null__"


def test_unit_totals_by_bool_dimension_lower_cased_string_dimension_preserved(
    exposures, purchase_metric_events, mean_metric, experiment, con
):
    """Only boolean-typed `by` columns are lower-cased for backend-consistent
    rendering; a genuinely string-typed property (e.g. a country code) keeps
    its original casing - lower-casing it would silently change its label."""
    bool_props = con.create_table(
        "breakout_props_bool",
        obj=[
            {"unit_id": "u1", "is_new": True},
            {"unit_id": "u2", "is_new": False},
        ],
    )
    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    totals = unit_totals(
        spine, stats, mean_metric, experiment, by=["is_new"], properties_table=bool_props
    ).execute()
    totals = totals.set_index("unit_id")
    assert totals.loc["u1", "is_new"] == "true"
    assert totals.loc["u2", "is_new"] == "false"


def test_unit_totals_properties_table_dedup_contract_prevents_duplicate_rows(
    exposures, purchase_metric_events, mean_metric, experiment, con
):
    """The properties_table fed to unit_totals must be deduplicated to one
    row per unit_id: applying the documented
    ``.order_by(ibis.desc(ts)).distinct(on="unit_id", keep="first")``
    idiom before the join keeps unit_totals' row count correct (one row
    per unit) even though the raw property source has 2 rows for u1;
    skipping the dedup step would silently duplicate u1's totals row
    through the join."""
    raw_props = con.create_table(
        "breakout_props_raw_dedup",
        obj=[
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 1, 9, 0, 0), "country": "US"},
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 2, 9, 0, 0), "country": "CA"},  # later
            {"unit_id": "u2", "ts": dt.datetime(2025, 8, 1, 10, 0, 0), "country": "US"},
        ],
    )
    deduped = raw_props.order_by(ibis.desc("ts")).distinct(on="unit_id", keep="first")
    properties_table = deduped.select("unit_id", "country")

    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    totals = unit_totals(
        spine, stats, mean_metric, experiment, by=["country"], properties_table=properties_table
    ).execute()
    # One row per unit - an un-deduped source with 2 rows for u1 would
    # inflate this to 4 total rows if the join weren't fed a deduped table.
    assert len(totals) == 3
    assert sorted(totals["unit_id"].tolist()) == ["u1", "u2", "u3"]
    # Latest value wins: u1's Aug-2 "CA" beats Aug-1's "US".
    assert totals[totals["unit_id"] == "u1"]["country"].iloc[0] == "CA"


def test_unit_totals_by_requires_properties_table(
    exposures, purchase_metric_events, mean_metric, experiment
):
    """`by` without `properties_table` is a caller error, not a silent
    no-op - there is no dimension data to carry through."""
    import pytest

    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        unit_totals(spine, stats, mean_metric, experiment, by=["country"])
    assert exc_info.value.code == "query.builders.unit_totals_by"


def test_unit_totals_properties_table_requires_by(
    exposures, purchase_metric_events, mean_metric, experiment, con
):
    """`properties_table` without `by` is a caller error, not a silent
    no-op - without `by` the join would run but add no columns, quietly
    discarding the dimension data (the mirror image of `by` without
    `properties_table`, above)."""
    import pytest

    props = con.create_table(
        "breakout_props_no_by",
        obj=[{"unit_id": "u1", "country": "US"}],
    )
    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        unit_totals(spine, stats, mean_metric, experiment, properties_table=props)
    assert exc_info.value.code == "query.builders.unit_totals_properties"


def test_group_summary_by_groups_by_dimension(
    exposures, purchase_metric_events, mean_metric, experiment, con
):
    """group_summary(by=[...]) groups additionally by the dimension
    column; each (group_id, dimension_value) pair gets its own row."""
    props = con.create_table(
        "breakout_props_group_summary",
        obj=[
            {"unit_id": "u1", "country": "US"},
            {"unit_id": "u2", "country": "US"},
            # u3 -> "__null__"
        ],
    )
    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    totals = unit_totals(
        spine, stats, mean_metric, experiment, by=["country"], properties_table=props
    )
    summary = group_summary(totals, by=["country"]).execute()
    assert "country" in summary.columns
    summary = summary.set_index(["group_id", "country"])
    # treatment/US=u1 only (69.49); treatment/__null__=u3 only (19.99); control/US=u2
    # only (30.00) - per-unit totals are hand-computed in conftest.py's module docstring.
    assert summary.loc[("treatment", "US"), "n"] == 1
    assert abs(_total(summary.loc[("treatment", "US")]) - 69.49) < 1e-9
    assert summary.loc[("treatment", "__null__"), "n"] == 1
    assert abs(_total(summary.loc[("treatment", "__null__")]) - 19.99) < 1e-9
    assert summary.loc[("control", "US"), "n"] == 1
    assert abs(_total(summary.loc[("control", "US")]) - 30.00) < 1e-9


def test_daily_group_summary_by_groups_by_dimension(
    exposures, purchase_metric_events, experiment, mean_metric, con
):
    """daily_group_summary(by=[...]) groups additionally by a dimension
    column that is already present on the input panel - joined
    upstream via join_breakout_dimension, mirroring how the daily
    per-segment view is wired (daily_group_summary itself takes no
    properties_table)."""
    daily_panel = unit_day_panel(
        exposures, purchase_metric_events, experiment, metric_name=mean_metric.name
    )
    props = con.create_table(
        "breakout_props_daily_group_summary",
        obj=[
            {"unit_id": "u1", "country": "US"},
            {"unit_id": "u2", "country": "US"},
            # u3 -> "__null__"
        ],
    )
    dim_panel = join_breakout_dimension(daily_panel, props, ["country"])
    dgs = daily_group_summary(dim_panel, metric=mean_metric, by=["country"]).execute()
    assert "country" in dgs.columns

    aug1 = dgs[dgs["ds"].dt.date == dt.date(2025, 8, 1)]
    trt_us_d1 = aug1[(aug1["group_id"] == "treatment") & (aug1["country"] == "US")]
    assert len(trt_us_d1) == 1
    assert abs(_total(trt_us_d1.iloc[0]) - 61.99) < 1e-9  # u1 Aug1 = 49.99+12.00

    aug3 = dgs[dgs["ds"].dt.date == dt.date(2025, 8, 3)]
    trt_null_d3 = aug3[(aug3["group_id"] == "treatment") & (aug3["country"] == "__null__")]
    assert len(trt_null_d3) == 1
    assert abs(_total(trt_null_d3.iloc[0]) - 19.99) < 1e-9  # u3 Aug3 purchase


def test_join_breakout_dimension_labels_booleans_lowercase_and_keeps_string_case(con):
    table = con.create_table("dim_units", obj=[{"unit_id": u} for u in ("u1", "u2", "u3", "u4")])
    props = con.create_table(
        "dim_props",
        schema=ibis.schema({"unit_id": "string", "flag": "boolean", "country": "string"}),
        obj=[
            {"unit_id": "u1", "flag": True, "country": "US"},
            {"unit_id": "u2", "flag": False, "country": "us"},
            {"unit_id": "u3", "flag": None, "country": None},
        ],
    )
    rows = join_breakout_dimension(table, props, ["flag", "country"]).execute()
    labels = {r.unit_id: (r.flag, r.country) for r in rows.itertuples()}
    assert labels == {
        "u1": ("true", "US"),
        "u2": ("false", "us"),
        "u3": ("__null__", "__null__"),
        "u4": ("__null__", "__null__"),
    }


def test_canonical_dimension_value_casts_non_boolean_types_like_sql_cast(con):
    t = con.create_table("canon_vals", obj=[{"n": 7, "s": "Ab"}])
    row = t.select(n=canonical_dimension_value(t.n), s=canonical_dimension_value(t.s)).execute()
    assert (row.n[0], row.s[0]) == ("7", "Ab")


def test_daily_group_summary_den_panel_computes_real_ratio_moments(exposures, experiment, con):
    """daily_group_summary(den_panel=...) computes a real
    ref_den/cden1/cden2/cyden family from the joined denominator panel
    instead of leaving it null (D4)."""
    ratio_metric = RatioMetric(
        name="rev_per_session",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=3),
        denominator=Measure(fact="session", aggregation="sum", window_days=3),
    )
    num_events_raw = con.create_table(
        "breakout_ratio_num_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 5, 0),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 5, 0),
                "event": "purchase",
                "amount": 20.0,
            },
        ],
    )
    den_events_raw = con.create_table(
        "breakout_ratio_den_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 4, 0),
                "event": "session",
                "n": 2.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 4, 0),
                "event": "session",
                "n": 4.0,
            },
        ],
    )
    num_events = metric_events(
        num_events_raw, ratio_metric, value_column="amount", part="numerator"
    )
    den_events = metric_events(den_events_raw, ratio_metric, value_column="n", part="denominator")

    num_panel = unit_day_panel(exposures, num_events, experiment, metric_name=ratio_metric.name)
    den_panel = unit_day_panel(exposures, den_events, experiment, metric_name=ratio_metric.name)

    dgs = daily_group_summary(num_panel, metric=ratio_metric, den_panel=den_panel).execute()
    aug1 = dgs[dgs["ds"].dt.date == dt.date(2025, 8, 1)]
    trt = aug1[aug1["group_id"] == "treatment"].iloc[0]  # u1
    ctrl = aug1[aug1["group_id"] == "control"].iloc[0]  # u2
    # Each arm has a single unit on Aug 1, so ref_den carries the whole
    # denominator and every centered moment is exactly 0.
    assert abs(trt["ref_den"] - 2.0) < 1e-9
    assert abs(_total(trt, "ref_den", "cden1") - 2.0) < 1e-9
    assert trt["cden1"] == 0.0
    assert trt["cden2"] == 0.0
    assert trt["cyden"] == 0.0
    assert abs(ctrl["ref_den"] - 4.0) < 1e-9
    assert abs(_total(ctrl, "ref_den", "cden1") - 4.0) < 1e-9
    assert ctrl["cden1"] == 0.0
    assert ctrl["cden2"] == 0.0
    assert ctrl["cyden"] == 0.0


def test_daily_group_summary_without_den_panel_stays_null(panel, mean_metric):
    """Anti-drift: daily_group_summary without den_panel keeps the ratio
    cross-moment columns null, unchanged from before this parameter
    existed."""
    dgs = daily_group_summary(panel, metric=mean_metric).execute()
    for field in ("ref_den", "cden1", "cden2", "cyden"):
        assert dgs[field].isna().all()


def test_daily_group_summary_avg_event_matches_per_day_event_mean_not_sum(con):
    """day0 events [2,4] (event mean 3), day1 event [12] (mean 12): the
    daily outcome is the per-day EVENT MEAN {3, 12} -- never the per-day
    SUM {6, 12}. This is the finding's own reproduction: the author ran
    this exact same-day [2,4] avg_event witness through the old
    hardcoded-sum reducer and observed 6, not 3."""
    metric = MeanMetric(
        name="avg_purchase_daily", entity="unit_id", fact="purchase", aggregation="avg_event"
    )
    rows = [
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 1),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 2,
            "sum_value": 6.0,
            "min_value": 2.0,
            "max_value": 4.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 2),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 1,
            "sum_value": 12.0,
            "min_value": 12.0,
            "max_value": 12.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
    ]
    panel = con.create_table("daily_avg_event_witness", obj=rows)
    dgs = con.to_pyarrow(daily_group_summary(panel, metric=metric)).to_pylist()
    by_day = {row["ds"]: _total(row, "ref_y", "cy1") for row in dgs}
    assert by_day == pytest.approx(
        {dt.date(2025, 8, 1): 3.0, dt.date(2025, 8, 2): 12.0}, rel=0, abs=1e-9
    )


def test_daily_group_summary_ratio_parts_honor_independent_aggregations(con):
    """A RatioMetric's day-grain numerator and denominator each honor
    their OWN declared aggregation independently -- numerator avg_event
    (event mean), denominator count (raw event count) -- never both
    silently forced to sum."""
    metric = RatioMetric(
        name="avg_purchase_per_view_daily",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="avg_event"),
        denominator=Measure(fact="page_view", aggregation="count"),
    )
    num_rows = [
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 1),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 2,
            "sum_value": 6.0,
            "min_value": 2.0,
            "max_value": 4.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 2),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 1,
            "sum_value": 12.0,
            "min_value": 12.0,
            "max_value": 12.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
    ]
    den_rows = [
        {**num_rows[0], "n_events": 5, "sum_value": 0.0, "min_value": 0.0, "max_value": 0.0},
        {**num_rows[1], "n_events": 2, "sum_value": 0.0, "min_value": 0.0, "max_value": 0.0},
    ]
    num_panel = con.create_table("daily_ratio_num_witness", obj=num_rows)
    den_panel = con.create_table("daily_ratio_den_witness", obj=den_rows)
    dgs = con.to_pyarrow(
        daily_group_summary(num_panel, metric=metric, den_panel=den_panel)
    ).to_pylist()
    by_day = {row["ds"]: row for row in dgs}
    assert _total(by_day[dt.date(2025, 8, 1)], "ref_y", "cy1") == pytest.approx(3.0)
    assert _total(by_day[dt.date(2025, 8, 1)], "ref_den", "cden1") == pytest.approx(5.0)
    assert _total(by_day[dt.date(2025, 8, 2)], "ref_y", "cy1") == pytest.approx(12.0)
    assert _total(by_day[dt.date(2025, 8, 2)], "ref_den", "cden1") == pytest.approx(2.0)


@pytest.mark.parametrize("grain", ["daily", "asof"])
@pytest.mark.parametrize("aggregation", ["avg_event", "sum", "min", "max"])
def test_ratio_day_moments_exclude_only_undefined_denominators(con, grain, aggregation):
    metric = RatioMetric(
        name="ratio",
        entity="unit_id",
        numerator=Measure(fact="num", aggregation="sum"),
        denominator=Measure(fact="den", aggregation=aggregation),
    )
    num_rows, den_rows = [], []
    for day in (1, 2, 3):
        for unit, value in enumerate((2.0, 6.0, 10.0)):
            row = {
                "unit_id": str(unit),
                "ds": dt.date(2025, 8, day),
                "experiment_id": "e",
                "group_id": "treatment",
                "metric": metric.name,
                "first_exposure_date": dt.date(2025, 8, 1),
                "n_events": 1,
                "sum_value": value,
                "min_value": value,
                "max_value": value,
            }
            num_rows.append(row)
            observed = (day == 1 and unit < 2) or (day == 2 and unit == 2)
            den_value = float(4 * unit) if observed else 0.0
            den_rows.append(
                {
                    **row,
                    "n_events": int(observed),
                    "sum_value": den_value,
                    "min_value": den_value,
                    "max_value": den_value,
                }
            )
    num_panel = con.create_table(f"undefined_den_num_{grain}_{aggregation}", obj=num_rows)
    den_panel = con.create_table(f"undefined_den_den_{grain}_{aggregation}", obj=den_rows)
    result = (
        daily_group_summary(num_panel, metric=metric, den_panel=den_panel)
        if grain == "daily"
        else asof_group_summary(num_panel, metric, den_panel=den_panel)
    )
    by_day = {row["ds"].day: row for row in con.to_pyarrow(result).to_pylist()}
    expected_units = (
        ({1: [0, 1], 2: [2]} if grain == "daily" else {1: [0, 1], 2: [0, 1, 2], 3: [0, 1, 2]})
        if aggregation == "avg_event"
        else {day: [0, 1, 2] for day in (1, 2, 3)}
    )
    assert set(by_day) == set(expected_units)
    for day, units in expected_units.items():
        y = np.array([2.0, 6.0, 10.0])[units] * (day if grain == "asof" else 1)
        den_values = (
            ([0.0, 4.0, 0.0] if day == 1 else [0.0, 4.0, 8.0])
            if grain == "asof"
            else {1: [0.0, 4.0, 0.0], 2: [0.0, 0.0, 8.0], 3: [0.0] * 3}[day]
        )
        den = np.array(den_values)[units]
        dy, dden = y - y.mean(), den - den.mean()
        expected = {
            "n": len(units),
            "ref_y": y.mean(),
            "ref_den": den.mean(),
            "cy1": dy.sum(),
            "cden1": dden.sum(),
            "cy2": (dy * dy).sum(),
            "cden2": (dden * dden).sum(),
            "cyden": (dy * dden).sum(),
        }
        assert {key: by_day[day][key] for key in expected} == pytest.approx(expected)


def test_daily_group_summary_min_max_count_distinct_kept_at_zero_event_day(con):
    """A zero-event day is DEFINED (folded to the aggregation's identity,
    row kept) for min/max/count/count_distinct, but UNDEFINED (row
    dropped) for avg_event -- the same per-aggregation zero-event policy
    already enforced at the unit_totals grain, now checked at the daily
    grain across all four supported non-sum aggregations."""
    days = [
        {
            "ds": dt.date(2025, 8, 1),
            "n_events": 2,
            "sum_value": 14.0,
            "min_value": 5.0,
            "max_value": 9.0,
        },
        {
            "ds": dt.date(2025, 8, 2),
            "n_events": 0,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        },
        {
            "ds": dt.date(2025, 8, 3),
            "n_events": 1,
            "sum_value": 3.0,
            "min_value": 3.0,
            "max_value": 3.0,
        },
    ]

    def _panel(name, metric_name):
        rows = [
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "treatment",
                "metric": metric_name,
                "first_exposure_ts": dt.datetime(2025, 8, 1),
                "first_exposure_date": dt.date(2025, 8, 1),
                **day,
            }
            for day in days
        ]
        return con.create_table(name, obj=rows)

    expectations: dict[Literal["min", "max", "count", "count_distinct"], dict[dt.date, float]] = {
        "min": {dt.date(2025, 8, 1): 5.0, dt.date(2025, 8, 2): 0.0, dt.date(2025, 8, 3): 3.0},
        "max": {dt.date(2025, 8, 1): 9.0, dt.date(2025, 8, 2): 0.0, dt.date(2025, 8, 3): 3.0},
        "count": {dt.date(2025, 8, 1): 2.0, dt.date(2025, 8, 2): 0.0, dt.date(2025, 8, 3): 1.0},
        "count_distinct": {
            dt.date(2025, 8, 1): 1.0,
            dt.date(2025, 8, 2): 0.0,
            dt.date(2025, 8, 3): 1.0,
        },
    }
    for aggregation, expected in expectations.items():
        metric = MeanMetric(
            name=f"m_daily_zero_{aggregation}",
            entity="unit_id",
            fact="purchase",
            aggregation=aggregation,
        )
        panel = _panel(f"daily_zero_event_{aggregation}", metric.name)
        dgs = con.to_pyarrow(daily_group_summary(panel, metric=metric)).to_pylist()
        by_day = {row["ds"]: _total(row, "ref_y", "cy1") for row in dgs}
        assert by_day == pytest.approx(expected, rel=0, abs=1e-9), aggregation

    avg_metric = MeanMetric(
        name="m_daily_zero_avg_event", entity="unit_id", fact="purchase", aggregation="avg_event"
    )
    avg_panel = _panel("daily_zero_event_avg_event", avg_metric.name)
    avg_dgs = con.to_pyarrow(daily_group_summary(avg_panel, metric=avg_metric)).to_pylist()
    avg_by_day = {row["ds"]: _total(row, "ref_y", "cy1") for row in avg_dgs}
    assert dt.date(2025, 8, 2) not in avg_by_day, (
        "a zero-event day is undefined (0/0) for avg_event and must be dropped, not folded to 0"
    )
    assert avg_by_day == pytest.approx(
        {dt.date(2025, 8, 1): 7.0, dt.date(2025, 8, 3): 3.0}, rel=0, abs=1e-9
    )


def test_breakout_dedup_and_dimensioned_query_renders_snowflake_sql(
    exposures, purchase_metric_events, mean_metric, experiment, con
):
    """The dimensioned summary chain compiles to Snowflake SQL."""
    raw_props = con.create_table(
        "breakout_props_snowflake_render",
        obj=[
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 1, 9, 0, 0), "country": "US"},
            {"unit_id": "u1", "ts": dt.datetime(2025, 8, 2, 9, 0, 0), "country": "CA"},
            {"unit_id": "u2", "ts": dt.datetime(2025, 8, 1, 10, 0, 0), "country": "US"},
        ],
    )
    deduped = raw_props.order_by(ibis.desc("ts")).distinct(on="unit_id", keep="first")
    properties_table = deduped.select("unit_id", "country")

    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    totals = unit_totals(
        spine, stats, mean_metric, experiment, by=["country"], properties_table=properties_table
    )
    summary = group_summary(totals, by=["country"])
    ibis.to_sql(summary, dialect="snowflake")


def test_unit_totals_by_with_ratio_metric_no_unit_id_collision(exposures, experiment, con):
    """Regression: unit_totals's ratio-metric branch already performs one
    "unit_id"-keyed left_join (numerator x denominator) before the
    breakout dimension join runs. The dimension join must use its own
    right-side alias rather than ibis's default "{name}_right" - reusing
    the default collides with the column that join already produced and
    raises ``IntegrityError: Name collisions``. The same collision class
    applies to CUPED's pre-period join, which also keys on "unit_id"."""
    ratio_metric = RatioMetric(
        name="rev_per_session",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=3),
        denominator=Measure(fact="session", aggregation="sum", window_days=3),
    )
    num_events_raw = con.create_table(
        "breakout_ratio_collision_num",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 5, 0),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 5, 0),
                "event": "purchase",
                "amount": 20.0,
            },
        ],
    )
    den_events_raw = con.create_table(
        "breakout_ratio_collision_den",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 4, 0),
                "event": "session",
                "n": 2.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 1, 10, 4, 0),
                "event": "session",
                "n": 4.0,
            },
        ],
    )
    num_events = metric_events(
        num_events_raw, ratio_metric, value_column="amount", part="numerator"
    )
    den_events = metric_events(den_events_raw, ratio_metric, value_column="n", part="denominator")
    spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)

    props = con.create_table(
        "breakout_props_ratio_collision",
        obj=[
            {"unit_id": "u1", "country": "US"},
            {"unit_id": "u2", "country": "CA"},
        ],
    )
    den_stats = post_exposure_stats(den_events, exposures, source_key=f"{ratio_metric.name}:den")
    totals = unit_totals(
        spine,
        stats,
        ratio_metric,
        experiment,
        den_stats=den_stats,
        by=["country"],
        properties_table=props,
    ).execute()
    totals = totals.set_index("unit_id")
    assert totals.loc["u1", "country"] == "US"
    assert totals.loc["u1", "y"] == 10.0
    assert totals.loc["u1", "y_den"] == 2.0
    assert totals.loc["u2", "country"] == "CA"
    assert totals.loc["u2", "y"] == 20.0
    assert totals.loc["u2", "y_den"] == 4.0


# Exposure-event self-counting: the panel join and ratio-metric denominator
# scope both require strict `ts > first_exposure_ts`, never `>=`.


def test_unit_day_panel_excludes_event_at_exact_exposure_timestamp(exposures, experiment, con):
    """Regression: unit_day_panel's event join used an inclusive
    ``events.ts >= spine.first_exposure_ts``. The join itself has no
    fact-awareness - it simply excludes any event whose timestamp exactly
    equals ``first_exposure_ts``, regardless of what fact/event type it's
    labeled. In production this matters whenever a metric's fact is the
    same event type that produced the exposure: `metric_events` filters the
    shared fact table to that event type BEFORE `unit_day_panel` runs, so
    the exposure row itself (timestamp == first_exposure_ts exactly) is
    already present in that metric's events stream and would otherwise
    phantom-count as a same-day metric occurrence. This test exercises the
    join directly with a synthetic events table (bypassing the fact-filter
    step) to isolate the join's own boundary behavior.

    u1's first_exposure_ts is 2025-08-01 09:00:00 (see conftest). This
    fixture places an event at that EXACT timestamp (the phantom) plus a
    genuine later event the next day, which must still register - proving
    the fix removes exactly the phantom row and nothing else.
    """
    collision_metric = MeanMetric(
        name="visits_collision",
        entity="unit_id",
        fact="visit",
        aggregation="sum",
        window_days=3,
    )
    collision_events_raw = con.create_table(
        "collision_mean_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),  # == u1's first_exposure_ts exactly
                "event": "visit",
                "amount": 100.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),  # genuine later occurrence
                "event": "visit",
                "amount": 7.0,
            },
        ],
    )
    events = metric_events(collision_events_raw, collision_metric, value_column="amount")
    panel = unit_day_panel(exposures, events, experiment, metric_name=collision_metric.name)
    panel_df = panel.execute()

    # Day 0 (the exposure day) must show NO signal from the exposure row itself.
    day0 = panel_df[(panel_df["unit_id"] == "u1") & (panel_df["ds"].dt.date == dt.date(2025, 8, 1))]
    assert day0["sum_value"].iloc[0] == 0.0
    # The later, genuine same-fact event still registers on its own day.
    day1 = panel_df[(panel_df["unit_id"] == "u1") & (panel_df["ds"].dt.date == dt.date(2025, 8, 2))]
    assert day1["sum_value"].iloc[0] == 7.0

    # unit_totals: the window sum reflects ONLY the genuine later event
    # not the phantom exposure-day amount (would be 107.0 pre-fix).
    spine, stats = unit_day_spine_stats(exposures, events, experiment, collision_metric.name)
    totals = unit_totals(spine, stats, collision_metric, experiment).execute().set_index("unit_id")
    assert totals.loc["u1", "y"] == 7.0


def test_unit_totals_ratio_denominator_excludes_event_at_exact_exposure_timestamp(
    exposures, experiment, con
):
    """Regression: unit_totals's RatioMetric branch scopes den_events to
    post-exposure via its OWN ``den_scoped.ts >= den_scoped.first_exposure_ts``
    filter - a code path separate from unit_day_panel's join (the numerator
    here uses an unrelated fact, so the panel-level fix alone does not
    exercise this). Like that join, this filter has no fact-awareness - it
    simply excludes any denominator event whose timestamp exactly equals
    ``first_exposure_ts``. In production this matters whenever a ratio
    metric's DENOMINATOR fact is the same event type that produced the
    exposure: the exposure row itself (timestamp == first_exposure_ts
    exactly) would otherwise be double-counted into ``y_den``.
    """
    ratio_metric = RatioMetric(
        name="purchases_per_session_collision",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=3),
        denominator=Measure(fact="session", aggregation="sum", window_days=3),
    )
    num_events_raw = con.create_table(
        "collision_ratio_num_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 10, 0),
                "event": "purchase",
                "amount": 10.0,
            },
        ],
    )
    den_events_raw = con.create_table(
        "collision_ratio_den_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),  # == u1's first_exposure_ts exactly
                "event": "session",
                "n": 2.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),  # genuine later session
                "event": "session",
                "n": 5.0,
            },
        ],
    )
    num_events = metric_events(
        num_events_raw, ratio_metric, value_column="amount", part="numerator"
    )
    den_events = metric_events(den_events_raw, ratio_metric, value_column="n", part="denominator")
    spine, stats = unit_day_spine_stats(exposures, num_events, experiment, ratio_metric.name)
    den_stats = post_exposure_stats(den_events, exposures, source_key=f"{ratio_metric.name}:den")
    totals = (
        unit_totals(spine, stats, ratio_metric, experiment, den_stats=den_stats)
        .execute()
        .set_index("unit_id")
    )
    # y_den reflects ONLY the genuine later session - not the phantom
    # exposure-day session (would be 7.0 pre-fix: 2.0 phantom + 5.0 genuine).
    assert totals.loc["u1", "y_den"] == 5.0


def test_unit_day_panel_retention_fact_collision_no_phantom_day_zero(exposures, experiment, con):
    """Regression, panel layer: proves unit_day_panel itself no longer
    contains a phantom day-0 event when a RetentionMetric's fact is the same
    event type that produced first_exposure_ts - independent of the fact
    that unit_totals's retention branch already excludes day 0 via its own
    ``active = panel.filter(panel.ds >= threshold_date)`` filter regardless
    of this bug. The query-layer fix must be correct at the panel-building
    layer itself, not just wherever a caller happens to mask the symptom.
    """
    retention_collision_metric = RetentionMetric(
        name="retained_collision",
        entity="unit_id",
        fact="visit",
        threshold_days=2,
    )
    collision_events_raw = con.create_table(
        "collision_retention_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),  # == u1's first_exposure_ts exactly
                "event": "visit",
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 3, 10, 0, 0),  # on/after threshold (Aug1+2=Aug3)
                "event": "visit",
            },
        ],
    )
    events = metric_events(collision_events_raw, retention_collision_metric)
    panel = unit_day_panel(
        exposures, events, experiment, metric_name=retention_collision_metric.name
    )
    panel_df = panel.execute()

    # Day 0 (the exposure day) must show no phantom occurrence from the exposure
    # row itself, inspected directly on the panel before any downstream threshold filtering.
    day0 = panel_df[(panel_df["unit_id"] == "u1") & (panel_df["ds"].dt.date == dt.date(2025, 8, 1))]
    assert day0["sum_value"].iloc[0] == 0.0
    # The genuine later on/after-threshold occurrence is unaffected.
    day2 = panel_df[(panel_df["unit_id"] == "u1") & (panel_df["ds"].dt.date == dt.date(2025, 8, 3))]
    assert day2["sum_value"].iloc[0] == 1.0

    # Sanity: unit_totals's y is unaffected for this fixture (threshold filter
    # already excludes day 0) - confirms the bug is invisible downstream but fixed at the panel layer.
    spine, stats = unit_day_spine_stats(
        exposures, events, experiment, retention_collision_metric.name
    )
    totals = unit_totals(spine, stats, retention_collision_metric, experiment).execute()
    assert totals[totals["unit_id"] == "u1"]["y"].iloc[0] == 1.0


# Breakout dict keys don't collide across fact sources


def _analysis_with_cross_source_breakouts(con):
    """Build an ``Analysis`` over two fact sources ('src_a',
    'src_b') that both declare a ``country`` property - the exact shape
    validated by ``test_breakout_same_property_different_sources_allowed``
    in ``tests/semantics/test_breakout_models.py`` - backed by real
    DuckDB tables. Bypasses ``load()``'s YAML-file requirement the same
    way ``tests/test_analysis.py``'s zero-metrics tests do (``__new__``
    plus manual attribute assignment), since ``Analysis``'s
    only file-reading step is ``load(definitions_path)``.
    """
    if "breakout_collision_src_a" not in con.list_tables():
        con.create_table(
            "breakout_collision_src_a",
            obj=[
                {
                    "user_id": "u1",
                    "ts": dt.datetime(2024, 1, 2, 9, 0, 0),
                    "event": "page_view",
                    "group_id": "C",
                    "country_code": "US",
                },
                {
                    "user_id": "u2",
                    "ts": dt.datetime(2024, 1, 2, 9, 5, 0),
                    "event": "page_view",
                    "group_id": "T",
                    "country_code": "CA",
                },
            ],
        )
    if "breakout_collision_src_b" not in con.list_tables():
        con.create_table(
            "breakout_collision_src_b",
            obj=[
                {
                    "user_id": "u1",
                    "ts": dt.datetime(2024, 1, 3, 10, 0, 0),
                    "event": "order",
                    "order_id": "o1",
                    "country_code": "MX",
                },
                {
                    "user_id": "u2",
                    "ts": dt.datetime(2024, 1, 3, 10, 5, 0),
                    "event": "order",
                    "order_id": "o2",
                    "country_code": "MX",
                },
            ],
        )

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "src_a",
                    "sql": "SELECT * FROM breakout_collision_src_a",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "page_view", "column": None}],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                },
                {
                    "name": "src_b",
                    "sql": "SELECT * FROM breakout_collision_src_b",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "order", "column": "order_id"}],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                },
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "conversion",
                    "name": "visit_rate",
                    "entity": "user_id",
                    "fact": "page_view",
                },
                {
                    "type": "conversion",
                    "name": "checkout",
                    "entity": "user_id",
                    "fact": "order",
                },
            ],
            "experiments": [
                {
                    "name": "collision_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2024-01-01",
                    "control_group": "C",
                    "plan": {"secondaries": ["visit_rate", "checkout"]},
                    "breakouts": [
                        {"property": "country", "source": "src_a"},
                        {"property": "country", "source": "src_b"},
                    ],
                }
            ],
        }
    )

    analysis = make_analysis(con, defs)
    return analysis


def test_panel_sql_keys_dont_collide_across_breakout_sources(con):
    """Two breakouts sharing a ``property`` name but resolving to
    different fact sources must each get their own ``panel_sql`` key.
    Pre-fix, both breakouts' dict keys were plain
    ``f"{metric.name}:{breakout.property}"`` with no source qualifier,
    so the second breakout's SQL silently clobbered the first's entry."""
    analysis = _analysis_with_cross_source_breakouts(con)
    breakouts = analysis.experiment.breakouts

    result = analysis.panel_sql(breakouts=breakouts)
    assert "visit_rate:country:src_a" in result
    assert "visit_rate:country:src_b" in result
    assert result["visit_rate:country:src_a"] != result["visit_rate:country:src_b"]


def test_summary_sql_keys_dont_collide_across_breakout_sources(con):
    """Same collision regression as above, for ``summary_sql``."""
    analysis = _analysis_with_cross_source_breakouts(con)
    breakouts = analysis.experiment.breakouts

    result = analysis.summary_sql(breakouts=breakouts)
    assert "checkout:country:src_a" in result
    assert "checkout:country:src_b" in result
    assert result["checkout:country:src_a"] != result["checkout:country:src_b"]


def test_breakout_summaries_keys_dont_collide_across_breakout_sources(con):
    """Same collision regression as above, for ``breakout_summaries``
    all 2 metrics x 2 breakout-sources combinations must be present, not
    collapsed down to 2 entries by the second source's breakout
    overwriting the first's."""
    analysis = _analysis_with_cross_source_breakouts(con)

    result = analysis.breakout_summaries()
    assert set(result) == {
        "visit_rate:country:src_a",
        "visit_rate:country:src_b",
        "checkout:country:src_a",
        "checkout:country:src_b",
    }
    for tables in result.values():
        assert set(tables) == {"group_summary", "daily_group_summary"}


# window_bound_stats: per-day window scoping


def test_window_bound_stats_removes_dilution_at_scale(con):
    """Regression (dilution bug): daily_group_summary's per-day `n` must
    reflect only units still WITHIN their own metric window that day,
    not the full calendar-alive population.

    Without window-bounding, unit_day_panel keeps a dense zero-valued row
    for every enrolled unit through experiment.end regardless of whether
    that unit's own analysis window has closed. Handing that panel
    straight to daily_group_summary inflates its per-day `n` with
    calendar-alive-but-out-of-window units: once enrollment completes,
    `n` stays FLAT for the rest of the experiment while the genuinely
    in-window population collapses toward 0 as more units age past their
    window - diluting the day's mean toward 0 in the window's tail even
    though the TRUE in-window daily mean never moves. This is the real
    shape of the investigated bug, reproduced here at a scale (140 units,
    14 enrollment days, multi-week span) the module's existing 2-3-unit
    fixtures are too small to expose.

    Fixture: 140 units enroll 10/day (5 treatment + 5 control) across 14
    calendar day-offsets (0-13). window_days=7. Every unit generates a
    constant $10.00 purchase on EACH day of its OWN 7-day window and
    nothing else - so the TRUE in-window daily mean is exactly $10.00
    on every calendar day, and any other value is dilution.
    """
    n_enroll_days = 14
    per_day = 10  # 5 treatment + 5 control
    window_days = 7
    start = dt.datetime(2025, 1, 1)

    experiment = Experiment(
        name="dilution_test",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=30),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(
        name="daily_revenue",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        window_days=window_days,
    )

    if "dilution_exposure_events" in con.list_tables():
        exposure_events = con.table("dilution_exposure_events")
        purchase_events = con.table("dilution_purchase_events")
    else:
        exposure_rows = []
        purchase_rows = []
        for day in range(n_enroll_days):
            fe_ts = start + dt.timedelta(days=day, hours=9)
            for i in range(per_day):
                uid = f"u{day}_{i}"
                group = "treatment" if i % 2 == 0 else "control"
                exposure_rows.append(
                    {
                        "unit_id": uid,
                        "ts": fe_ts,
                        "event": "exposure",
                        "experiment_id": "dilution_test",
                        "group_id": group,
                    }
                )
                for w in range(window_days):
                    purchase_rows.append(
                        {
                            "unit_id": uid,
                            "ts": fe_ts + dt.timedelta(days=w, hours=1),
                            "event": "purchase",
                            "amount": 10.0,
                        }
                    )
        exposure_events = con.create_table("dilution_exposure_events", obj=exposure_rows)
        purchase_events = con.create_table("dilution_purchase_events", obj=purchase_rows)

    exposures = first_exposures(exposure_events, experiment)
    events = metric_events(purchase_events, metric, value_column="amount")
    panel = unit_day_panel(exposures, events, experiment, metric_name=metric.name)

    total_units = n_enroll_days * per_day  # 140

    def in_window_units(day_offset: int) -> int:
        valid_days = [d for d in range(n_enroll_days) if d <= day_offset <= d + window_days - 1]
        return len(valid_days) * per_day

    # - Sanity check on the raw (unbounded) panel: the bug's exact shape
    unbounded = daily_group_summary(panel, metric=metric).execute()
    tail_offset = 25  # every unit's window has closed well before this
    assert in_window_units(tail_offset) == 0
    tail_date = (start + dt.timedelta(days=tail_offset)).date()
    tail_rows = unbounded[unbounded["ds"].dt.date == tail_date]
    assert tail_rows["n"].sum() == total_units, (
        "sanity check: the unbounded panel keeps every enrolled unit calendar-alive"
    )
    assert _total(tail_rows).sum() == 0.0, (
        "sanity check: no unit generates revenue once its own window has closed"
    )

    # - Fix: bound the panel to each unit's own window before aggregating
    bounded_panel = window_bound_stats(panel, metric)
    bounded = daily_group_summary(bounded_panel, metric=metric).execute()

    # Well past every unit's window close: no in-window units remain, so the
    # day has no row at all - not a diluted, near-zero row.
    assert tail_date not in set(bounded["ds"].dt.date), (
        "window-bounded daily_group_summary must have no row for a day past "
        "every unit's own window close"
    )

    # Mid-enrollment day: only units within the trailing `window_days` are
    # still in-window; the rest have aged out, but the unbounded panel still counts them.
    mid_offset = 15
    expected_in_window = in_window_units(mid_offset)
    assert 0 < expected_in_window < total_units, "test fixture drifted"
    mid_date = (start + dt.timedelta(days=mid_offset)).date()

    mid_unbounded = unbounded[unbounded["ds"].dt.date == mid_date]
    assert mid_unbounded["n"].sum() == total_units  # still the full flat population
    diluted_mean = _total(mid_unbounded).sum() / mid_unbounded["n"].sum()
    assert abs(diluted_mean - 10.0 * expected_in_window / total_units) < 1e-9
    assert diluted_mean < 10.0  # visibly diluted below the true in-window mean

    mid_bounded = bounded[bounded["ds"].dt.date == mid_date]
    assert mid_bounded["n"].sum() == expected_in_window
    fixed_mean = _total(mid_bounded).sum() / mid_bounded["n"].sum()
    assert abs(fixed_mean - 10.0) < 1e-9  # undiluted - exactly the true in-window mean


def test_resolve_window_days_retention_uses_band_right_edge(
    retention_metric, bounded_retention_metric
):
    """Retention's row bound is the band's right edge; None when open."""
    from increment._window import resolve_window_days

    assert resolve_window_days(bounded_retention_metric) == 4
    assert resolve_window_days(retention_metric) is None


def test_window_bound_stats_quantile_metric_is_a_noop(panel):
    """A quantile metric carries no window semantics of its own, so bounding
    is a no-op for it rather than an unsupported-metric refusal - it shares
    the conversion/mean dispatch branch, not the not-implemented one."""
    metric = QuantileMetric(name="p50_revenue", entity="unit_id", fact="purchase", quantile=0.5)

    unbounded = panel.execute()
    bounded = window_bound_stats(panel, metric).execute()

    def _keys(df):
        return sorted(zip(df["unit_id"], df["ds"].astype(str), strict=True))

    assert len(bounded) == len(unbounded)
    assert _keys(bounded) == _keys(unbounded)


def test_maturity_days_retention_falls_back_to_threshold(
    retention_metric, bounded_retention_metric, mean_metric, conversion_metric
):
    """Maturity is the band's right edge when bounded, the band open when not.

    The band-open fallback is what keeps late-enrollee censoring working
    for an unbounded retention metric, whose outcome is final - for
    censoring purposes - once the band opens.
    """
    from increment._window import maturity_days

    assert maturity_days(bounded_retention_metric) == 4
    assert maturity_days(retention_metric) == 2
    assert maturity_days(mean_metric) == 3
    assert maturity_days(conversion_metric) == 1


def test_retention_band_returns_band_start_and_end(retention_metric, bounded_retention_metric):
    assert bounded_retention_metric.band == (2, 4)
    assert retention_metric.band == (2, None)


def test_window_bound_stats_bounds_bounded_retention(
    exposures, page_view_events, experiment, bounded_retention_metric
):
    """A bounded band trims the retention panel's right edge."""
    events = metric_events(page_view_events, bounded_retention_metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=bounded_retention_metric.name)
    bounded = window_bound_stats(panel, bounded_retention_metric).execute()

    # u1 exposed Aug 1 -> rows Aug 1..Aug 4 (Aug 5 excluded, end-exclusive)
    u1_days = sorted(bounded[bounded["unit_id"] == "u1"]["ds"].dt.date.tolist())
    assert u1_days == [
        dt.date(2025, 8, 1),
        dt.date(2025, 8, 2),
        dt.date(2025, 8, 3),
        dt.date(2025, 8, 4),
    ]
    # u3 exposed Aug 2 -> rows Aug 2..Aug 5
    u3_days = sorted(bounded[bounded["unit_id"] == "u3"]["ds"].dt.date.tolist())
    assert u3_days == [
        dt.date(2025, 8, 2),
        dt.date(2025, 8, 3),
        dt.date(2025, 8, 4),
        dt.date(2025, 8, 5),
    ]


def test_window_bound_stats_unbounded_retention_metric_is_noop(
    exposures, page_view_events, experiment, retention_metric
):
    """An unbounded band has no right edge to bind.

    Bounded retention IS trimmed - see
    test_window_bound_stats_bounds_bounded_retention.
    """
    events = metric_events(page_view_events, retention_metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=retention_metric.name)

    unbounded = panel.execute()
    bounded = window_bound_stats(panel, retention_metric).execute()

    def _keys(df):
        return sorted(zip(df["unit_id"], df["ds"].astype(str), strict=True))

    assert len(bounded) == len(unbounded)
    assert _keys(bounded) == _keys(unbounded)


def test_window_bound_stats_refuses_unsupported_metric_type(panel):
    from increment.semantics.models import TotalMetric

    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    with pytest.raises(UnsupportedRequestError) as exc_info:
        window_bound_stats(panel, metric)
    assert exc_info.value.code == "query.builders.window_bound_stats"
    assert exc_info.value.context["metric_type"] == "TotalMetric"


def test_unit_totals_refuses_unsupported_metric_type(
    exposures, purchase_metric_events, experiment, mean_metric
):
    """A metric type with no unit-total aggregation is refused by name, with
    the same code the window-bound check raises for it."""
    from increment.semantics.models import TotalMetric

    metric = TotalMetric(name="rev", fact="purchase", aggregation="sum")
    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    with pytest.raises(UnsupportedRequestError) as exc_info:
        unit_totals(spine, stats, metric, experiment, warn_on_censoring=False)
    assert exc_info.value.code == "query.builders.window_bound_stats"
    assert exc_info.value.context["metric_type"] == "TotalMetric"


def test_bounded_retention_respects_band_right_edge(exposures, page_view_events, experiment):
    """An event outside the band does not count as a return."""
    # Band [fe+4, fe+5): u1 (exposed Aug 1) -> [Aug 5, Aug 6), matures Aug 6
    # (within the fixture's Aug 6 end). Its only post-exposure page view (Aug 3) is before the band starts.
    late_band = RetentionMetric(
        name="retained_late",
        entity="unit_id",
        fact="page_view",
        threshold_days=(4, 5),  # band opens Aug 5 for u1, closes Aug 6
    )
    events = metric_events(page_view_events, late_band)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, late_band.name)
    totals = unit_totals(spine, stats, late_band, experiment).execute()
    # u1's only post-Aug-1 page view is Aug 3, BEFORE the Aug 5 band start
    assert totals[totals["unit_id"] == "u1"]["y"].iloc[0] == 0.0


def test_bounded_retention_right_edge_is_exclusive_at_the_semantic_level(con, experiment):
    """An event landing exactly ON the maturity day is outside the band.

    Direct pin of the half-open, day-0-indexed choice, at the y=0/y=1
    level rather than only indirectly via which panel rows
    window_bound_stats keeps.
    """
    boundary = RetentionMetric(
        name="retained_boundary",
        entity="unit_id",
        fact="page_view",
        threshold_days=(2, 4),  # band [fe+2, fe+4); u1 exposed Aug 1 -> [Aug 3, Aug 5)
    )
    exposure_rows = con.create_table(
        "boundary_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    # u1's only post-exposure page view lands exactly on the band's right edge
    # (Aug 5, the maturity day) - one day past Aug 4. An inclusive edge would wrongly score y=1.
    view_rows = con.create_table(
        "boundary_page_views",
        obj=[{"unit_id": "u1", "ts": dt.datetime(2025, 8, 5, 10, 0, 0), "event": "page_view"}],
    )
    exposures_b = first_exposures(exposure_rows, experiment)
    events = metric_events(view_rows, boundary)
    spine, stats = unit_day_spine_stats(exposures_b, events, experiment, boundary.name)
    totals = unit_totals(spine, stats, boundary, experiment).execute()
    assert totals[totals["unit_id"] == "u1"]["y"].iloc[0] == 0.0


def test_retention_band_case_table_pins_every_boundary(con):
    """Six units, one band, both shapes - the per-unit ratchet table from
    the design doc (Decision 5), pinned directly.

    Same underlying events feed a bounded ``[7, 14)`` metric and an
    unbounded ``[7, inf)`` metric with the same left edge. Returns land
    at day 6 (before threshold), day 7 (on threshold), day 10
    (mid-band), day 14 (on the right edge - half-open, must differ
    between the two shapes), day 30 (after the bounded band has closed
    but still inside the unbounded one), and never.
    """
    band_experiment = Experiment(
        name="exp_band_table",
        unit="unit_id",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 3, 1),  # far past every return so nothing censors
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    bounded = RetentionMetric(
        name="retained_band_table_bounded",
        entity="unit_id",
        fact="page_view",
        threshold_days=(7, 14),
    )
    unbounded = RetentionMetric(
        name="retained_band_table_unbounded",
        entity="unit_id",
        fact="page_view",
        threshold_days=7,
    )
    exposure_rows = con.create_table(
        "band_table_exposures",
        obj=[
            {
                "unit_id": unit,
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_band_table",
                "group_id": "treatment",
            }
            for unit in ("day6", "day7", "day10", "day14", "day30", "never")
        ],
    )
    view_rows = con.create_table(
        "band_table_page_views",
        obj=[
            {"unit_id": "day6", "ts": dt.datetime(2025, 1, 7, 10, 0, 0), "event": "page_view"},
            {"unit_id": "day7", "ts": dt.datetime(2025, 1, 8, 10, 0, 0), "event": "page_view"},
            {"unit_id": "day10", "ts": dt.datetime(2025, 1, 11, 10, 0, 0), "event": "page_view"},
            {"unit_id": "day14", "ts": dt.datetime(2025, 1, 15, 10, 0, 0), "event": "page_view"},
            {"unit_id": "day30", "ts": dt.datetime(2025, 1, 31, 10, 0, 0), "event": "page_view"},
            # "never" gets no post-exposure event at all.
        ],
    )
    exposures_bt = first_exposures(exposure_rows, band_experiment)

    # (unit_id, y under the bounded [7, 14) band, y under the unbounded [7, inf) band)
    expected = {
        "day6": (0.0, 0.0),  # before threshold in both
        "day7": (1.0, 1.0),  # on threshold, in-band in both
        "day10": (1.0, 1.0),  # mid-band in both
        "day14": (0.0, 1.0),  # right edge: bounded excludes, unbounded ratchets
        "day30": (0.0, 1.0),  # bounded band long closed frozen at 0; unbounded ratchets
        "never": (0.0, 0.0),
    }

    results = {}
    for metric in (bounded, unbounded):
        events = metric_events(view_rows, metric)
        spine, stats = unit_day_spine_stats(exposures_bt, events, band_experiment, metric.name)
        totals = unit_totals(spine, stats, metric, band_experiment).execute()
        results[metric.name] = totals.set_index("unit_id")["y"]

    for unit_id, (y_bounded, y_unbounded) in expected.items():
        assert results[bounded.name].loc[unit_id] == y_bounded, unit_id
        assert results[unbounded.name].loc[unit_id] == y_unbounded, unit_id


def test_bounded_and_unbounded_retention_disagree_on_a_late_return(con, experiment):
    """Discrimination guard: the band's right edge must actually apply.

    One unit returns only AFTER its bounded band closes. The bounded
    metric must score it 0 while an unbounded metric with the same
    threshold scores it 1. If the two ever agree here, the band's right
    edge has been silently lost somewhere between the metric declaration
    and unit_totals - exactly the regression a missed band reader would
    cause (an unbounded y silently computed for a bounded metric).
    """
    bounded = RetentionMetric(
        name="retained_band", entity="unit_id", fact="page_view", threshold_days=(2, 4)
    )
    unbounded = RetentionMetric(
        name="retained_open", entity="unit_id", fact="page_view", threshold_days=2
    )
    exposure_rows = con.create_table(
        "late_return_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    # u1's only post-exposure page view lands on day 5 (Aug 6): after the bounded
    # band [Aug 3, Aug 5) has closed, but inside the unbounded band [fe+2, inf).
    view_rows = con.create_table(
        "late_return_page_views",
        obj=[{"unit_id": "u1", "ts": dt.datetime(2025, 8, 6, 10, 0, 0), "event": "page_view"}],
    )
    exposures_l = first_exposures(exposure_rows, experiment)

    ys = {}
    for metric in (bounded, unbounded):
        events = metric_events(view_rows, metric)
        spine, stats = unit_day_spine_stats(exposures_l, events, experiment, metric.name)
        totals = unit_totals(spine, stats, metric, experiment).execute()
        ys[metric.name] = totals[totals["unit_id"] == "u1"]["y"].iloc[0]

    assert ys["retained_band"] == 0.0
    assert ys["retained_open"] == 1.0


@pytest.fixture
def running_tables(con):
    """The two tables both running-experiment censoring tests read.

    A fixture, not a side effect of whichever test happens to run first:
    consuming a table another test created makes the second test pass only
    in file order, and fail when run alone or on another parallel worker.
    """
    if "running_exposures" not in con.list_tables():
        con.create_table(
            "running_exposures",
            obj=[
                {
                    "unit_id": "early",
                    "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                    "event": "exposure",
                    "experiment_id": "exp_running",
                    "group_id": "treatment",
                },
                {
                    "unit_id": "late",
                    "ts": dt.datetime(2025, 8, 4, 9, 0, 0),
                    "event": "exposure",
                    "experiment_id": "exp_running",
                    "group_id": "control",
                },
            ],
        )
    if "running_page_views" not in con.list_tables():
        con.create_table(
            "running_page_views",
            obj=[
                # early: exposed Aug 1, threshold Aug 3, returns Aug 3 -> y = 1
                {"unit_id": "early", "ts": dt.datetime(2025, 8, 3, 10, 0, 0), "event": "page_view"},
                # last observed event date is Aug 4; `late` was exposed Aug 4 so its
                # threshold (Aug 6) has not arrived -> not yet observable
                {"unit_id": "late", "ts": dt.datetime(2025, 8, 4, 10, 0, 0), "event": "page_view"},
            ],
        )
    return con.table("running_exposures"), con.table("running_page_views")


def test_retention_censors_immature_units_when_experiment_end_is_none(
    con, retention_metric, running_tables
):
    """A running experiment must not score a not-yet-observable unit as 0.

    Regression: with experiment.end unset the censoring filter used to be
    skipped entirely, so a unit exposed fewer than threshold_days before
    the last observed event date fell through the left join and got
    coalesced to y = 0 - counted as "did not return" rather than
    excluded as not yet measurable.
    """
    exposure_rows, view_rows = running_tables
    running = Experiment(
        name="exp_running",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=None,
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    exposures = first_exposures(exposure_rows, running)
    events = metric_events(view_rows, retention_metric)
    spine, stats = unit_day_spine_stats(exposures, events, running, retention_metric.name)
    with pytest.warns(IncrementWarning) as rec:
        totals = unit_totals(spine, stats, retention_metric, running).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)

    assert "late" not in totals["unit_id"].values, "immature unit must be absent, not y=0"
    assert totals[totals["unit_id"] == "early"]["y"].iloc[0] == 1.0


def test_retention_end_none_agrees_with_end_pinned_to_last_event(
    con, retention_metric, running_tables
):
    """The two censoring paths must produce identical group summaries.

    Same data, experiment.end=None vs experiment.end pinned to the last
    observed event date - any divergence means the end=None branch is
    using a different bound.
    """
    exposure_rows, view_rows = running_tables

    def summary_for(end):
        experiment = Experiment(
            name="exp_running",
            unit="unit_id",
            start=dt.datetime(2025, 8, 1),
            end=end,
            control_group="control",
            exposure="test_exposure",
            plan=AnalysisPlan(),
        )
        exposures = first_exposures(exposure_rows, experiment)
        events = metric_events(view_rows, retention_metric)
        spine, stats = unit_day_spine_stats(exposures, events, experiment, retention_metric.name)
        with pytest.warns(IncrementWarning) as rec:
            totals = unit_totals(spine, stats, retention_metric, experiment)
            assert "frame.censoring.dropped_units" in warning_codes(rec)
        return group_summary(totals).execute().sort_values("group_id").reset_index(drop=True)

    open_ended = summary_for(None)
    pinned = summary_for(dt.datetime(2025, 8, 4))
    assert open_ended["n"].tolist() == pinned["n"].tolist()
    for field in ("ref_y", "cy1", "cy2"):
        assert open_ended[field].tolist() == pinned[field].tolist()


def test_empty_events_panel_censors_every_unit_when_end_is_unset(con, retention_metric):
    """With NO events and end unset, neither unit has an observed
    calendar day: enrollment identity alone is not evidence any time has
    passed, so `early` (exposed Aug 1) and `late` (exposed Aug 4) both
    stay censored rather than one maturing with a phantom y = 0."""
    exposure_rows = con.create_table(
        "empty_events_running_exposures",
        obj=[
            {
                "unit_id": "early",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_running",
                "group_id": "treatment",
            },
            {
                "unit_id": "late",
                "ts": dt.datetime(2025, 8, 4, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_running",
                "group_id": "control",
            },
        ],
    )
    empty_views = con.create_table(
        "empty_page_views",
        obj=[{"unit_id": "nobody", "ts": dt.datetime(2025, 8, 1, 0, 0, 0), "event": "other"}],
    )
    running = Experiment(
        name="exp_running",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=None,
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    exposures = first_exposures(exposure_rows, running)
    events = metric_events(empty_views, retention_metric)
    spine, stats = unit_day_spine_stats(exposures, events, running, retention_metric.name)
    with pytest.warns(IncrementWarning) as rec:
        totals = unit_totals(spine, stats, retention_metric, running).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)

    assert len(totals) == 0, "no unit has an observed calendar day yet"


def test_empty_events_windowed_metric_censors_every_unit(con):
    """Zero events + no declared end must never admit windowed units as y = 0.

    Zero matching events NULLs `unit_day_spine_stats`' event-horizon
    fallback (`events.ts.max()`), so `panel_spine` gives every enrolled
    unit a null-date placeholder row instead of a dated one (D1):
    enrollment identity alone is not evidence any calendar day was
    observed. `_censor_to_observable_window`'s `ds.notnull()` gate then
    drops both units before their 7-day windows are ever evaluated --
    the honest reading when no time beyond enrollment was observed.
    """
    metric = ConversionMetric(name="converted", entity="unit_id", fact="purchase", window_days=7)
    exposure_rows = con.create_table(
        "noevents_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_noevents",
                "group_id": "treatment",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "test_exposure",
                "experiment_id": "exp_noevents",
                "group_id": "control",
            },
        ],
    )
    # No purchase rows at all - only a non-matching event so the table
    # has the right schema but metric_events yields zero rows.
    no_purchases = con.create_table(
        "noevents_purchases",
        obj=[
            {
                "unit_id": "nobody",
                "ts": dt.datetime(2025, 8, 1, 0, 0, 0),
                "event": "other",
                "value": 1.0,
            }
        ],
    )
    running = Experiment(
        name="exp_noevents",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=None,
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    exposures = first_exposures(exposure_rows, running)
    events = metric_events(no_purchases, metric)
    spine, stats = unit_day_spine_stats(exposures, events, running, metric.name)
    with pytest.warns(IncrementWarning) as rec:
        totals = unit_totals(spine, stats, metric, running).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)

    assert len(totals) == 0, "no unit's window closed within observed enrollment time"


# asof_group_summary (as-of "as of day N" view)


def test_asof_group_summary_avg_event_cumulative_matches_hand_reduced_not_sum_or_mean_of_means(
    con,
):
    """Same day0 [2,4] / day1 [12] witness, cumulative through each day:
    {3, 6} (combined sum_value / combined n_events) -- never {6, 18} (the
    hardcoded-sum bug) and never the mean-of-daily-means {3, 7.5}
    ((3+12)/2)."""
    metric = MeanMetric(
        name="avg_purchase_asof", entity="unit_id", fact="purchase", aggregation="avg_event"
    )
    rows = [
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 1),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 2,
            "sum_value": 6.0,
            "min_value": 2.0,
            "max_value": 4.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 2),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 1,
            "sum_value": 12.0,
            "min_value": 12.0,
            "max_value": 12.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
    ]
    panel = con.create_table("asof_avg_event_witness", obj=rows)
    ags = con.to_pyarrow(asof_group_summary(panel, metric)).to_pylist()
    by_day = {row["ds"]: _total(row, "ref_y", "cy1") for row in ags}
    assert by_day == pytest.approx(
        {dt.date(2025, 8, 1): 3.0, dt.date(2025, 8, 2): 6.0}, rel=0, abs=1e-9
    )


def test_asof_group_summary_ratio_parts_honor_independent_aggregations(con):
    """The cumulative as-of series honors each RatioMetric part's own
    declared aggregation independently -- numerator avg_event, denominator
    count -- while the denominator stays aligned to the numerator's own
    rows (the intentional numerator-window alignment), not a separately
    windowed series."""
    metric = RatioMetric(
        name="avg_purchase_per_view_asof",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="avg_event"),
        denominator=Measure(fact="page_view", aggregation="count"),
    )
    num_rows = [
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 1),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 2,
            "sum_value": 6.0,
            "min_value": 2.0,
            "max_value": 4.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
        {
            "unit_id": "u1",
            "ds": dt.date(2025, 8, 2),
            "experiment_id": "e",
            "group_id": "treatment",
            "metric": metric.name,
            "n_events": 1,
            "sum_value": 12.0,
            "min_value": 12.0,
            "max_value": 12.0,
            "first_exposure_ts": dt.datetime(2025, 8, 1),
            "first_exposure_date": dt.date(2025, 8, 1),
        },
    ]
    den_rows = [
        {**num_rows[0], "n_events": 5, "sum_value": 0.0, "min_value": 0.0, "max_value": 0.0},
        {**num_rows[1], "n_events": 2, "sum_value": 0.0, "min_value": 0.0, "max_value": 0.0},
    ]
    num_panel = con.create_table("asof_ratio_num_witness", obj=num_rows)
    den_panel = con.create_table("asof_ratio_den_witness", obj=den_rows)
    ags = con.to_pyarrow(asof_group_summary(num_panel, metric, den_panel=den_panel)).to_pylist()
    by_day = {row["ds"]: row for row in ags}
    assert _total(by_day[dt.date(2025, 8, 1)], "ref_y", "cy1") == pytest.approx(3.0)
    assert _total(by_day[dt.date(2025, 8, 1)], "ref_den", "cden1") == pytest.approx(5.0)
    assert _total(by_day[dt.date(2025, 8, 2)], "ref_y", "cy1") == pytest.approx(6.0)
    assert _total(by_day[dt.date(2025, 8, 2)], "ref_den", "cden1") == pytest.approx(7.0)


def test_asof_group_summary_min_max_count_distinct_kept_before_first_event_unlike_avg_event(
    con,
):
    """Before this unit's first observed event, min/max/count/
    count_distinct are all DEFINED (folded to the identity, row kept) --
    ``n`` must grow monotonically and never shrink -- but avg_event is
    UNDEFINED (0/0) and its row is dropped until an event actually
    occurs."""
    days = [
        {
            "ds": dt.date(2025, 8, 1),
            "n_events": 0,
            "sum_value": 0.0,
            "min_value": 0.0,
            "max_value": 0.0,
        },
        {
            "ds": dt.date(2025, 8, 2),
            "n_events": 1,
            "sum_value": 7.0,
            "min_value": 7.0,
            "max_value": 7.0,
        },
        {
            "ds": dt.date(2025, 8, 3),
            "n_events": 2,
            "sum_value": 6.0,
            "min_value": 3.0,
            "max_value": 3.0,
        },
    ]

    def _panel(name, metric_name):
        rows = [
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "treatment",
                "metric": metric_name,
                "first_exposure_ts": dt.datetime(2025, 8, 1),
                "first_exposure_date": dt.date(2025, 8, 1),
                **day,
            }
            for day in days
        ]
        return con.create_table(name, obj=rows)

    expectations: dict[Literal["min", "max", "count", "count_distinct"], dict[dt.date, float]] = {
        "min": {dt.date(2025, 8, 1): 0.0, dt.date(2025, 8, 2): 7.0, dt.date(2025, 8, 3): 3.0},
        "max": {dt.date(2025, 8, 1): 0.0, dt.date(2025, 8, 2): 7.0, dt.date(2025, 8, 3): 7.0},
        "count": {dt.date(2025, 8, 1): 0.0, dt.date(2025, 8, 2): 1.0, dt.date(2025, 8, 3): 3.0},
        "count_distinct": {
            dt.date(2025, 8, 1): 0.0,
            dt.date(2025, 8, 2): 1.0,
            dt.date(2025, 8, 3): 2.0,
        },
    }
    for aggregation, expected in expectations.items():
        metric = MeanMetric(
            name=f"m_asof_zero_{aggregation}",
            entity="unit_id",
            fact="purchase",
            aggregation=aggregation,
        )
        panel = _panel(f"asof_before_first_event_{aggregation}", metric.name)
        ags = con.to_pyarrow(asof_group_summary(panel, metric)).to_pylist()
        by_day = {row["ds"]: _total(row, "ref_y", "cy1") for row in ags}
        assert by_day == pytest.approx(expected, rel=0, abs=1e-9), aggregation

    avg_metric = MeanMetric(
        name="m_asof_zero_avg_event", entity="unit_id", fact="purchase", aggregation="avg_event"
    )
    avg_panel = _panel("asof_before_first_event_avg_event", avg_metric.name)
    avg_ags = con.to_pyarrow(asof_group_summary(avg_panel, avg_metric)).to_pylist()
    avg_by_day = {row["ds"]: _total(row, "ref_y", "cy1") for row in avg_ags}
    assert dt.date(2025, 8, 1) not in avg_by_day, (
        "before any observed event, avg_event is undefined (0/0) and its row must be dropped"
    )
    assert avg_by_day == pytest.approx(
        {dt.date(2025, 8, 2): 7.0, dt.date(2025, 8, 3): 13.0 / 3.0}, rel=0, abs=1e-9
    )


def test_asof_group_summary_schema(panel, mean_metric):
    """Output columns match daily_group_summary's centered-moment shape
    exactly - same (ds, experiment_id, metric, group_id) grain, same
    n/ref_y/cy1/cy2/ref_x/cx1/cx2/cxy/ref_den/cden1/cden2/cyden/
    sum_d/cyd/cy2d/cxd columns."""
    result = asof_group_summary(panel, mean_metric)
    assert set(result.columns) == DAILY_GROUP_SUMMARY


def test_asof_group_summary_partitions_by_experiment_not_just_unit(con):
    """Regression: the as-of running sum must partition on
    (experiment_id, unit_id), not unit_id alone -- a unit_id that
    recurs across two experiments would otherwise bleed one
    experiment's cumulative values into the other's."""
    panel = ibis.memtable(
        [
            {
                "unit_id": "u1",
                "experiment_id": "expA",
                "group_id": "treatment",
                "metric": "rev",
                "ds": dt.date(2025, 1, 1),
                "n_events": 1,
                "sum_value": 100.0,
                "min_value": 100.0,
                "max_value": 100.0,
                "first_exposure_date": dt.date(2025, 1, 1),
            },
            {
                "unit_id": "u1",
                "experiment_id": "expB",
                "group_id": "treatment",
                "metric": "rev",
                "ds": dt.date(2025, 1, 5),
                "n_events": 1,
                "sum_value": 1.0,
                "min_value": 1.0,
                "max_value": 1.0,
                "first_exposure_date": dt.date(2025, 1, 5),
            },
        ]
    )
    metric = MeanMetric(name="rev", entity="unit_id", fact="purchase", aggregation="sum")
    result = asof_group_summary(panel, metric).execute().set_index("experiment_id")
    assert result.loc["expA", "ref_y"] == 100.0
    assert result.loc["expB", "ref_y"] == 1.0


def test_asof_group_summary_by_groups_by_dimension(
    exposures, purchase_metric_events, experiment, mean_metric, panel, con
):
    """asof_group_summary(by=[...]) groups additionally by a
    dimension column that is already present on the input panel - joined
    upstream via join_breakout_dimension, the same contract
    daily_group_summary's own `by` has (neither builder takes a
    properties_table of its own).

    Each (ds, group_id, dimension_value) triple gets its OWN running
    total that freezes independently once that segment's units close
    their windows - and the per-segment running totals partition the
    un-dimensioned ones exactly (no double counting, no dropped unit).
    All values are the conftest module docstring's hand-computed
    per-unit-per-day revenue, accumulated: u1 (treatment/US, window
    [Aug 1, Aug 4)) 61.99 -> 61.99 -> 69.49 then frozen; u2
    (control/US, same window) 0 -> 30.00 -> 30.00 then frozen; u3
    (treatment/__null__, window [Aug 2, Aug 5), so it only enters the
    panel on Aug 2) 0 -> 19.99 then frozen.
    """
    import pytest

    props = con.create_table(
        "breakout_props_asof_group_summary",
        obj=[
            {"unit_id": "u1", "country": "US"},
            {"unit_id": "u2", "country": "US"},
            # u3 -> "__null__"
        ],
    )
    dim_panel = join_breakout_dimension(
        unit_day_panel(exposures, purchase_metric_events, experiment, metric_name=mean_metric.name),
        props,
        ["country"],
    )
    ags = asof_group_summary(dim_panel, mean_metric, by=["country"]).execute()
    assert "country" in ags.columns
    ags["ds"] = ags["ds"].dt.date
    ags = ags.set_index(["ds", "group_id", "country"])

    # - (a) every (ds, arm, segment) cell is exactly one unit here, so
    # its as-of ref_y is that unit's own running total
    expected = {
        "treatment/US": (dt.date(2025, 8, 1), "treatment", "US", 61.99),
        "control/US": (dt.date(2025, 8, 1), "control", "US", 0.0),
        # Aug 2: u1 had no purchase (running total holds at 61.99), u2's
        # 30.00 lands, u3 enters the panel with nothing yet.
        "treatment/US@2": (dt.date(2025, 8, 2), "treatment", "US", 61.99),
        "control/US@2": (dt.date(2025, 8, 2), "control", "US", 30.00),
        "treatment/null@2": (dt.date(2025, 8, 2), "treatment", "__null__", 0.0),
        # Aug 3: u1 +7.50 -> 69.49, u3 +19.99 - the two treatment
        # segments must NOT be pooled into one 89.48 row.
        "treatment/US@3": (dt.date(2025, 8, 3), "treatment", "US", 69.49),
        "control/US@3": (dt.date(2025, 8, 3), "control", "US", 30.00),
        "treatment/null@3": (dt.date(2025, 8, 3), "treatment", "__null__", 19.99),
    }
    for label, (ds, group_id, country, total_y) in expected.items():
        row = ags.loc[(ds, group_id, country)]
        assert row["n"] == 1, label
        assert row["ref_y"] == pytest.approx(total_y), label
        # n=1 centered on its own value: both residual moments vanish.
        assert row["cy1"] == 0.0, label
        assert row["cy2"] == 0.0, label

    # (b) each segment freezes at its own last in-window value through experiment.end
    # (Aug 6) rather than collapsing or accumulating further - u1 closes Aug 4, u3 Aug 5.
    for ds in (dt.date(2025, 8, 4), dt.date(2025, 8, 5), dt.date(2025, 8, 6)):
        assert ags.loc[(ds, "treatment", "US"), "ref_y"] == pytest.approx(69.49)
        assert ags.loc[(ds, "treatment", "__null__"), "ref_y"] == pytest.approx(19.99)
        assert ags.loc[(ds, "control", "US"), "ref_y"] == pytest.approx(30.00)

    # (c) segments partition the un-dimensioned view exactly: per (ds, arm), segment
    # n and totals sum back to the collapsed row's.
    collapsed = asof_group_summary(panel, mean_metric).execute()
    collapsed["ds"] = collapsed["ds"].dt.date
    regrouped = (
        ags.assign(total_y=_total(ags)).groupby(level=["ds", "group_id"])[["n", "total_y"]].sum()
    )
    for _, c in collapsed.iterrows():
        agg = regrouped.loc[(c["ds"], c["group_id"])]
        assert agg["n"] == c["n"]
        assert agg["total_y"] == pytest.approx(_total(c))


def test_asof_group_summary_at_scale_freezes_after_window_close(con):
    """On a fixture sized closer to the
    real investigation (hundreds of units, multi-week window, staggered
    enrollment so units reach full window-closure partway through the
    experiment), the as-of series has (a) `n` that grows
    monotonically toward the population size and never exceeds it, and
    (b) a per-unit-frozen value that holds steady once past that unit's
    window close rather than collapsing toward zero.

    Fixture: 200 units enroll 10/day (5 treatment + 5 control) across 20
    calendar day-offsets (0-19), matching the scale of the
    window_bound_stats dilution-at-scale fixture above. window_days=7.
    Every unit generates a constant $10.00 purchase on EACH day of its
    OWN 7-day window and nothing else, so its per-unit as-of total
    is exactly $70.00 once its window has closed - the last cohort
    (day offset 19) closes at calendar day 19+7=26 - and stays frozen
    at $70.00 for the remainder of the 40-day experiment.
    """
    n_enroll_days = 20
    per_day = 10  # 5 treatment + 5 control
    window_days = 7
    start = dt.datetime(2025, 3, 1)

    experiment = Experiment(
        name="asof_scale_test",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=40),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(
        name="asof_daily_revenue",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        window_days=window_days,
    )

    if "asof_exposure_events" in con.list_tables():
        exposure_events = con.table("asof_exposure_events")
        purchase_events = con.table("asof_purchase_events")
    else:
        exposure_rows = []
        purchase_rows = []
        for day in range(n_enroll_days):
            fe_ts = start + dt.timedelta(days=day, hours=9)
            for i in range(per_day):
                uid = f"cu{day}_{i}"
                group = "treatment" if i % 2 == 0 else "control"
                exposure_rows.append(
                    {
                        "unit_id": uid,
                        "ts": fe_ts,
                        "event": "exposure",
                        "experiment_id": "asof_scale_test",
                        "group_id": group,
                    }
                )
                for w in range(window_days):
                    purchase_rows.append(
                        {
                            "unit_id": uid,
                            "ts": fe_ts + dt.timedelta(days=w, hours=1),
                            "event": "purchase",
                            "amount": 10.0,
                        }
                    )
        exposure_events = con.create_table("asof_exposure_events", obj=exposure_rows)
        purchase_events = con.create_table("asof_purchase_events", obj=purchase_rows)

    exposures = first_exposures(exposure_events, experiment)
    events = metric_events(purchase_events, metric, value_column="amount")
    panel = unit_day_panel(exposures, events, experiment, metric_name=metric.name)

    total_units = n_enroll_days * per_day  # 200
    per_group = total_units // 2  # 100

    ags = asof_group_summary(panel, metric).execute()
    ags["ds"] = ags["ds"].dt.date

    # - (a) n grows monotonically toward the population, never exceeds it
    daily_n = ags.groupby("ds")["n"].sum().sort_index()
    assert (daily_n.diff().dropna() >= 0).all(), "daily n must never decrease"
    assert (daily_n <= total_units).all(), "daily n must never exceed the population"
    assert daily_n.max() == total_units

    full_enrollment_date = (start + dt.timedelta(days=n_enroll_days - 1)).date()
    assert daily_n.loc[full_enrollment_date] == total_units

    for gid, grp in ags.groupby("group_id"):
        grp_n = grp.set_index("ds")["n"].sort_index()
        assert (grp_n.diff().dropna() >= 0).all(), f"{gid}: n must never decrease"
        assert grp_n.max() == per_group

    # - (b) value holds/stabilizes once every unit's own window has
    # closed - does not collapse toward zero.
    full_close_date = (start + dt.timedelta(days=n_enroll_days - 1 + window_days)).date()
    tail_date = (start + dt.timedelta(days=35)).date()
    expected_frozen = per_group * window_days * 10.0  # 100 * 7 * 10 = 7000.0

    for gid in ("treatment", "control"):
        at_close = ags[(ags["ds"] == full_close_date) & (ags["group_id"] == gid)]
        at_tail = ags[(ags["ds"] == tail_date) & (ags["group_id"] == gid)]
        assert len(at_close) == 1 and len(at_tail) == 1
        assert abs(_total(at_close.iloc[0]) - expected_frozen) < 1e-9
        # holds steady from window-close through the tail - does not drop
        assert abs(_total(at_tail.iloc[0]) - _total(at_close.iloc[0])) < 1e-9
        assert at_tail["n"].iloc[0] == per_group

    # Contrast: the raw per-day bounded view vanishes in the tail (window_bound_stats
    # drops the row once a unit's window closes) - the collapse asof_group_summary avoids.
    bounded_tail = daily_group_summary(window_bound_stats(panel, metric), metric=metric).execute()
    bounded_tail["ds"] = bounded_tail["ds"].dt.date
    assert tail_date not in set(bounded_tail["ds"])


def test_asof_group_summary_includes_late_enrollee_unit_totals_censors(con):
    """unit_totals's late-enrollee
    censoring (any unit whose window would extend past experiment.end is
    dropped entirely) is intentionally NOT replicated by
    asof_group_summary - assert the documented divergence
    directly on a fixture engineered to trigger it.

    u_early (treatment) enrolls on day 0; its 3-day window closes well
    before experiment.end (day 9) - included everywhere.
    u_late (control) enrolls on day 8; its 3-day window would close on
    day 11, past experiment.end (day 9) - unit_totals censors it
    entirely, but it legitimately has two real panel days (day 8, day 9)
    before experiment.end cuts the panel off, and asof_group_summary
    must include its partial contribution on those two days.
    """
    start = dt.datetime(2025, 5, 1)
    experiment = Experiment(
        name="asof_censor_test",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=9),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(
        name="asof_censor_revenue",
        entity="unit_id",
        fact="purchase",
        aggregation="sum",
        window_days=3,
    )

    exposure_events = con.create_table(
        "asof_censor_exposure_events",
        obj=[
            {
                "unit_id": "u_early",
                "ts": start + dt.timedelta(hours=9),
                "event": "exposure",
                "experiment_id": "asof_censor_test",
                "group_id": "treatment",
            },
            {
                "unit_id": "u_late",
                "ts": start + dt.timedelta(days=8, hours=9),
                "event": "exposure",
                "experiment_id": "asof_censor_test",
                "group_id": "control",
            },
        ],
    )
    purchase_events = con.create_table(
        "asof_censor_purchase_events",
        obj=[
            {
                "unit_id": "u_early",
                "ts": start + dt.timedelta(hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u_early",
                "ts": start + dt.timedelta(days=1, hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u_early",
                "ts": start + dt.timedelta(days=2, hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u_late",
                "ts": start + dt.timedelta(days=8, hours=10),
                "event": "purchase",
                "amount": 7.0,
            },
            {
                "unit_id": "u_late",
                "ts": start + dt.timedelta(days=9, hours=10),
                "event": "purchase",
                "amount": 3.0,
            },
        ],
    )

    exposures = first_exposures(exposure_events, experiment)
    events = metric_events(purchase_events, metric, value_column="amount")
    panel = unit_day_panel(exposures, events, experiment, metric_name=metric.name)

    # - unit_totals censors u_late outright: its window (day8 + 3 =
    # day11) extends past experiment.end (day9)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    with pytest.warns(IncrementWarning) as rec:
        totals_expr = unit_totals(spine, stats, metric, experiment)
    totals = totals_expr.execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    assert sorted(totals["unit_id"].tolist()) == ["u_early"]
    summary = group_summary(totals_expr).execute()
    assert sorted(summary["group_id"].tolist()) == ["treatment"]  # control (u_late) has no row

    # asof_group_summary legitimately includes u_late's partial contribution on
    # the two days it had data before experiment.end cut the panel off.
    ags = asof_group_summary(panel, metric).execute()
    ags["ds"] = ags["ds"].dt.date
    day8 = (start + dt.timedelta(days=8)).date()
    day9 = (start + dt.timedelta(days=9)).date()

    ctrl_day8 = ags[(ags["ds"] == day8) & (ags["group_id"] == "control")]
    ctrl_day9 = ags[(ags["ds"] == day9) & (ags["group_id"] == "control")]
    assert len(ctrl_day8) == 1 and len(ctrl_day9) == 1
    assert ctrl_day8["n"].iloc[0] == 1
    assert ctrl_day9["n"].iloc[0] == 1
    assert abs(_total(ctrl_day8.iloc[0]) - 7.0) < 1e-9
    assert abs(_total(ctrl_day9.iloc[0]) - 10.0) < 1e-9  # 7 + 3, as-of - not censored away

    # treatment (u_early) window closes day0+3=day3 -> frozen at 30.0 by day9
    trt_day9 = ags[(ags["ds"] == day9) & (ags["group_id"] == "treatment")]
    assert abs(_total(trt_day9.iloc[0]) - 30.0) < 1e-9


def test_asof_group_summary_ratio_den_panel_is_asof_not_raw(con):
    """RatioMetric + den_panel: ref_den/cden1/cden2/cyden must reflect
    the AS-OF denominator (the per-unit running total through that day),
    not that day's raw per-day denominator value."""
    start = dt.datetime(2025, 6, 1)
    experiment = Experiment(
        name="asof_ratio_test",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=10),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    ratio_metric = RatioMetric(
        name="asof_rev_per_session",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=5),
        denominator=Measure(fact="session", aggregation="sum", window_days=5),
    )
    exposure_events = con.create_table(
        "asof_ratio_exposure_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=9),
                "event": "exposure",
                "experiment_id": "asof_ratio_test",
                "group_id": "treatment",
            },
        ],
    )
    num_events_raw = con.create_table(
        "asof_ratio_num_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(days=1, hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
        ],
    )
    den_events_raw = con.create_table(
        "asof_ratio_den_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=9, minutes=30),
                "event": "session",
                "n": 2.0,
            },
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(days=1, hours=9, minutes=30),
                "event": "session",
                "n": 3.0,
            },
        ],
    )

    exposures = first_exposures(exposure_events, experiment)
    num_events = metric_events(
        num_events_raw, ratio_metric, value_column="amount", part="numerator"
    )
    den_events = metric_events(den_events_raw, ratio_metric, value_column="n", part="denominator")
    num_panel = unit_day_panel(exposures, num_events, experiment, metric_name=ratio_metric.name)
    den_panel = unit_day_panel(exposures, den_events, experiment, metric_name=ratio_metric.name)

    ags = asof_group_summary(num_panel, ratio_metric, den_panel=den_panel).execute()
    ags["ds"] = ags["ds"].dt.date
    day0 = start.date()
    day1 = (start + dt.timedelta(days=1)).date()

    row0 = ags[ags["ds"] == day0].iloc[0]
    row1 = ags[ags["ds"] == day1].iloc[0]

    # Day 0: raw == as-of (first day) - den=2.0. One unit, so ref_den
    # carries it whole and every centered denominator moment is 0.
    assert abs(row0["ref_den"] - 2.0) < 1e-9
    assert abs(_total(row0, "ref_den", "cden1") - 2.0) < 1e-9
    assert row0["cden1"] == 0.0
    assert row0["cden2"] == 0.0
    assert row0["cyden"] == 0.0

    # Day 1: as-of den = 2.0 + 3.0 = 5.0 - NOT the raw day-1 value (3.0)
    assert abs(row1["ref_den"] - 5.0) < 1e-9
    assert abs(_total(row1, "ref_den", "cden1") - 5.0) < 1e-9
    assert row1["cden1"] == 0.0
    assert row1["cden2"] == 0.0
    assert row1["cyden"] == 0.0


def test_asof_conversion_repeated_days_remains_binary(con) -> None:
    start = dt.datetime(2026, 1, 1)
    experiment = Experiment(
        name="asof_conversion_binary",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=1),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = ConversionMetric(
        name="converted",
        entity="unit_id",
        fact="purchase",
        window_days=3,
    )
    exposure_events = con.create_table(
        "asof_conversion_binary_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": start,
                "event": "exposure",
                "experiment_id": experiment.name,
                "group_id": "treatment",
            }
        ],
    )
    purchase_events = con.create_table(
        "asof_conversion_binary_events",
        obj=[
            {"unit_id": "u1", "ts": start + dt.timedelta(hours=1), "event": "purchase"},
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(days=1, hours=1),
                "event": "purchase",
            },
        ],
    )
    exposures = first_exposures(exposure_events, experiment)
    events = metric_events(purchase_events, metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=metric.name)
    rows = asof_group_summary(panel, metric).execute().sort_values("ds")

    assert rows["n"].tolist() == [1, 1]
    assert rows["ref_y"].tolist() == [1.0, 1.0]
    assert rows["cy1"].tolist() == [0.0, 0.0]
    assert rows["successes"].tolist() == [1, 1]


def test_day_axis_conversion_successes_are_exact_integers(
    con, exposures, purchase_events, experiment, conversion_metric
):
    events = metric_events(purchase_events, conversion_metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=conversion_metric.name)
    daily = con.to_pyarrow(
        daily_group_summary(window_bound_stats(panel, conversion_metric), metric=conversion_metric)
    )
    assert daily.schema.field("successes").type == pa.int64()
    # Only u1's two purchases on its exposure day qualify; they count as one success.
    assert {(row["ds"].day, row["group_id"]): row["successes"] for row in daily.to_pylist()} == {
        (1, "control"): 0,
        (1, "treatment"): 1,
        (2, "treatment"): 0,
    }
    asof = con.to_pyarrow(asof_group_summary(panel, conversion_metric))
    assert asof.schema.field("successes").type == pa.int64()
    assert {(row["group_id"], row["successes"]) for row in asof.to_pylist()} == {
        ("control", 0),
        ("treatment", 1),
    }


def test_day_axis_retention_successes_are_exact_integers(
    con, exposures, page_view_events, experiment, bounded_retention_metric
):
    metric = bounded_retention_metric
    events = metric_events(page_view_events, metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=metric.name)
    asof = con.to_pyarrow(asof_group_summary(panel, metric))
    assert asof.schema.field("successes").type == pa.int64()
    # Only u1 returns in its [exposure+2, exposure+4) band.
    assert {(row["group_id"], row["successes"]) for row in asof.to_pylist()} == {
        ("control", 0),
        ("treatment", 1),
    }
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    cohorts = con.to_pyarrow(cohort_group_summary(spine, stats, metric, experiment))
    assert cohorts.schema.field("successes").type == pa.int64()
    assert {(row["ds"].day, row["group_id"]): row["successes"] for row in cohorts.to_pylist()} == {
        (1, "control"): 0,
        (1, "treatment"): 1,
        (2, "treatment"): 0,
    }


def test_day_axis_nonbinary_successes_stay_null(
    con, exposures, purchase_events, experiment, mean_metric
):
    events = metric_events(purchase_events, mean_metric, value_column="amount")
    panel = unit_day_panel(exposures, events, experiment, metric_name=mean_metric.name)
    for expr in (
        daily_group_summary(window_bound_stats(panel, mean_metric), metric=mean_metric),
        asof_group_summary(panel, mean_metric),
    ):
        table = con.to_pyarrow(expr)
        assert table.schema.field("successes").type == pa.int64()
        assert table.column("successes").null_count == table.num_rows


def test_asof_group_summary_ratio_plus_uptake_computes_both_correctly(con):
    """Regression: ratio's den reductions AND uptake's sum_d/cyd/cy2d
    must both be correct in the SAME asof_group_summary call.

    Exercises the join-ordering fix directly: den_panel is joined first,
    then the uptake join is layered on top of that same agg_source, then
    every moment expression (den AND uptake) is built from that one final
    table - never from a stale reference to an ancestor relation.
    """
    start = dt.datetime(2025, 7, 1)
    experiment = Experiment(
        name="asof_ratio_uptake_test",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=10),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    ratio_metric = RatioMetric(
        name="asof_rev_per_session_uptake",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=5),
        denominator=Measure(fact="session", aggregation="sum", window_days=5),
    )
    exposure_events = con.create_table(
        "asof_ratio_uptake_exposure_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=9),
                "event": "exposure",
                "experiment_id": "asof_ratio_uptake_test",
                "group_id": "treatment",
            },
        ],
    )
    num_events_raw = con.create_table(
        "asof_ratio_uptake_num_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(days=1, hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
        ],
    )
    den_events_raw = con.create_table(
        "asof_ratio_uptake_den_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=9, minutes=30),
                "event": "session",
                "n": 2.0,
            },
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(days=1, hours=9, minutes=30),
                "event": "session",
                "n": 3.0,
            },
        ],
    )
    click_events_raw = con.create_table(
        "asof_ratio_uptake_click_events",
        obj=[
            {"unit_id": "u1", "ts": start + dt.timedelta(days=1, hours=8), "event": "click"},
        ],
    )

    exposures = first_exposures(exposure_events, experiment)
    num_events = metric_events(
        num_events_raw, ratio_metric, value_column="amount", part="numerator"
    )
    den_events = metric_events(den_events_raw, ratio_metric, value_column="n", part="denominator")
    click_events = click_events_raw.mutate(value=ibis.literal(1.0))
    num_panel = unit_day_panel(exposures, num_events, experiment, metric_name=ratio_metric.name)
    den_panel = unit_day_panel(exposures, den_events, experiment, metric_name=ratio_metric.name)
    uptake_panel = unit_day_panel(
        exposures, click_events, experiment, metric_name=ratio_metric.name
    )

    ags = asof_group_summary(
        num_panel,
        ratio_metric,
        den_panel=den_panel,
        uptake_panel=uptake_panel,
        uptake_window_days=None,
    ).execute()
    ags["ds"] = ags["ds"].dt.date
    day0 = start.date()
    day1 = (start + dt.timedelta(days=1)).date()

    row0 = ags[ags["ds"] == day0].iloc[0]
    row1 = ags[ags["ds"] == day1].iloc[0]

    # Ratio's den readouts must match test_asof_group_summary_ratio_den_panel_is_asof_not_raw
    # exactly, proving the uptake join didn't perturb them.
    assert abs(row0["ref_den"] - 2.0) < 1e-9
    assert row0["cden2"] == 0.0
    assert row0["cyden"] == 0.0
    assert abs(row1["ref_den"] - 5.0) < 1e-9
    assert row1["cden2"] == 0.0
    assert row1["cyden"] == 0.0

    # Uptake moments: u1 clicks day 1, so day0's sum_d=0 and day1's sum_d=1. With n=1
    # the moments centre on the unit's own value and vanish; ref_y carries the as-of numerator (20.0).
    assert row0["sum_d"] == 0.0
    assert row0["cyd"] == 0.0
    assert row0["cy2d"] == 0.0
    assert row1["sum_d"] == 1.0
    assert abs(row1["ref_y"] - 20.0) < 1e-9
    assert row1["cyd"] == 0.0
    assert row1["cy2d"] == 0.0


def _bounded_encouragement_panels():
    start = dt.date(2026, 1, 1)
    units = (
        ("c1", "control", 1.0),
        ("c2", "control", 3.0),
        ("t1", "treatment", 2.0),
        ("t2", "treatment", 6.0),
    )
    outcome_rows = []
    uptake_rows = []
    for unit, group, value in units:
        for offset in range(5):
            ds = start + dt.timedelta(days=offset)
            common = {
                "unit_id": unit,
                "ds": ds,
                "experiment_id": "late_over_time",
                "group_id": group,
                "first_exposure_date": start,
            }
            outcome_value = value if offset < 2 else 1000.0
            outcome_rows.append(
                common
                | {
                    "metric": "revenue",
                    "n_events": 1,
                    "sum_value": outcome_value,
                    "min_value": outcome_value,
                    "max_value": outcome_value,
                    # Fixed pre-period covariate: larger values predict the
                    # units that take up in the treatment arm.
                    "x": {"c1": 1.0, "c2": 3.0, "t1": 2.0, "t2": 6.0}[unit],
                }
            )
            uptake_value = float(group == "treatment" and offset == (0 if unit == "t2" else 2))
            uptake_rows.append(
                common
                | {
                    "metric": "__uptake",
                    "n_events": 1,
                    "sum_value": uptake_value,
                    "min_value": uptake_value,
                    "max_value": uptake_value,
                }
            )
    return ibis.memtable(outcome_rows), ibis.memtable(uptake_rows), start


def test_asof_group_summary_encouragement_cxd_matches_centered_oracle():
    outcome, uptake, start = _bounded_encouragement_panels()
    metric = MeanMetric(
        name="revenue", entity="user_id", fact="purchase", aggregation="sum", window_days=2
    )

    rows = asof_group_summary(
        outcome,
        metric,
        uptake_panel=uptake,
        uptake_window_days=3,
    ).execute()
    rows["ds"] = rows["ds"].dt.date

    # Treatment x = [2, 6], ref_x = 4.  t2 takes up on day 0, while t1
    # takes up on day 2, so cxd is nonzero before both units are taken up.
    treatment = rows[rows["group_id"] == "treatment"].sort_values("ds")
    assert treatment["ds"].tolist() == [
        start,
        start + dt.timedelta(days=1),
        start + dt.timedelta(days=2),
        start + dt.timedelta(days=3),
        start + dt.timedelta(days=4),
    ]
    assert treatment["cxd"].tolist() == pytest.approx([2.0, 2.0, 0.0, 0.0, 0.0])

    # Control has no uptake: cxd is materialized as 0.0, not a typed NULL.
    control = rows[rows["group_id"] == "control"]
    assert control["cxd"].tolist() == pytest.approx([0.0] * len(control))

    no_uptake = asof_group_summary(outcome, metric).execute()
    assert no_uptake["cxd"].isna().all()

    no_x = outcome.select([name for name in outcome.columns if name != "x"])
    no_x_with_uptake = asof_group_summary(
        no_x,
        metric,
        uptake_panel=uptake,
        uptake_window_days=3,
    ).execute()
    assert no_x_with_uptake["cxd"].isna().all()


def test_asof_group_summary_completed_encouragement_gates_later_window_edge():
    outcome, uptake, start = _bounded_encouragement_panels()
    metric = MeanMetric(
        name="revenue", entity="user_id", fact="purchase", aggregation="sum", window_days=2
    )

    rows = asof_group_summary(
        outcome,
        metric,
        uptake_panel=uptake,
        uptake_window_days=3,
        completed_windows_only=True,
    ).execute()
    rows["ds"] = rows["ds"].dt.date

    assert rows["ds"].min() == start + dt.timedelta(days=3)
    treatment = rows.loc[rows["group_id"] == "treatment"].sort_values("ds")
    assert treatment["n"].tolist() == [2, 2]
    assert treatment["sum_d"].tolist() == [2.0, 2.0]
    assert treatment["ref_y"].tolist() == [8.0, 8.0]


def test_asof_group_summary_completed_encouragement_gates_later_window_edge_reverse_order():
    """Same gate, with the outcome window the LARGER of the two (uptake
    smaller) - regression against a max()/min() or argument-order
    mistake that only the other ordering would hide."""
    start = dt.date(2026, 1, 1)
    units = (
        ("c1", "control", 1.0),
        ("c2", "control", 3.0),
        ("t1", "treatment", 2.0),
        ("t2", "treatment", 6.0),
    )
    outcome_window_days = 3
    outcome_rows = []
    uptake_rows = []
    for unit, group, value in units:
        for offset in range(5):
            ds = start + dt.timedelta(days=offset)
            common = {
                "unit_id": unit,
                "ds": ds,
                "experiment_id": "late_over_time",
                "group_id": group,
                "first_exposure_date": start,
            }
            outcome_value = value if offset < outcome_window_days else 1000.0
            outcome_rows.append(
                common
                | {
                    "metric": "revenue",
                    "n_events": 1,
                    "sum_value": outcome_value,
                    "min_value": outcome_value,
                    "max_value": outcome_value,
                }
            )
            uptake_value = float(group == "treatment" and offset == (0 if unit == "t2" else 2))
            uptake_rows.append(
                common
                | {
                    "metric": "__uptake",
                    "n_events": 1,
                    "sum_value": uptake_value,
                    "min_value": uptake_value,
                    "max_value": uptake_value,
                }
            )
    outcome = ibis.memtable(outcome_rows)
    uptake = ibis.memtable(uptake_rows)
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        window_days=outcome_window_days,
    )

    rows = asof_group_summary(
        outcome,
        metric,
        uptake_panel=uptake,
        uptake_window_days=2,
        completed_windows_only=True,
    ).execute()
    rows["ds"] = rows["ds"].dt.date

    assert rows["ds"].min() == start + dt.timedelta(days=3)
    treatment = rows.loc[rows["group_id"] == "treatment"].sort_values("ds")
    assert treatment["n"].tolist() == [2, 2]
    assert treatment["sum_d"].tolist() == [1.0, 1.0]
    assert treatment["ref_y"].tolist() == [12.0, 12.0]


@pytest.mark.parametrize(
    ("outcome_window_days", "uptake_window_days"),
    [(None, 3), (2, None)],
)
def test_asof_group_summary_completed_encouragement_requires_two_bounded_windows(
    outcome_window_days, uptake_window_days
):
    outcome, uptake, _ = _bounded_encouragement_panels()
    metric = MeanMetric(
        name="revenue",
        entity="user_id",
        fact="purchase",
        aggregation="sum",
        window_days=outcome_window_days,
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        asof_group_summary(
            outcome,
            metric,
            uptake_panel=uptake,
            uptake_window_days=uptake_window_days,
            completed_windows_only=True,
        )
    assert (
        exc_info.value.code == "query.builders.asof_group_summary_completed_requires_uptake_window"
    )


def test_asof_group_summary_refuses_unsupported_metric_type(panel):
    metric = QuantileMetric(name="p50_revenue", entity="unit_id", fact="purchase", quantile=0.5)
    with pytest.raises(UnsupportedRequestError) as exc_info:
        asof_group_summary(panel, metric)
    assert exc_info.value.code == "query.builders.asof_group_summary_metric_type_not_implemented"
    assert exc_info.value.context["metric_type"] == "QuantileMetric"


def test_asof_group_summary_refuses_den_panel_for_retention_metric(panel, retention_metric):
    with pytest.raises(InvalidRequestError) as exc_info:
        asof_group_summary(panel, retention_metric, den_panel=panel)
    assert exc_info.value.code == "query.builders.asof_group_summary_den_panel_retention"


def test_asof_group_summary_ratio_uses_numerator_window_for_denominator(con):
    """Regression: den_panel's masking must use the RatioMetric's
    NUMERATOR window_days for both sides (the same convention
    unit_totals's RatioMetric branch already uses), not the
    denominator's own window_days - even when they differ.

    numerator.window_days=3, denominator.window_days=8 (deliberately
    much longer, so a bug that masks the denominator by its OWN window
    would NOT freeze it in time to be caught by day 3). One session
    event lands on each of day0/1/2/3 (n=1.0 each). The numerator's
    3-day window closes at day0+3=day3 (mask condition ds < window_end,
    same strict boundary as window_bound_stats), so day3's session event
    must be masked to 0 - the as-of denominator total must freeze at
    3.0 from day3 onward. A denominator-window-governed implementation
    would still be well within its own 8-day window on day3 and
    incorrectly include it, giving 4.0.
    """
    start = dt.datetime(2025, 7, 1)
    experiment = Experiment(
        name="asof_ratio_window_test",
        unit="unit_id",
        start=start,
        end=start + dt.timedelta(days=10),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    ratio_metric = RatioMetric(
        name="asof_rev_per_session_mismatched_window",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=3),
        denominator=Measure(fact="session", aggregation="sum", window_days=8),
    )
    exposure_events = con.create_table(
        "asof_ratio_window_exposure_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=9),
                "event": "exposure",
                "experiment_id": "asof_ratio_window_test",
                "group_id": "treatment",
            },
        ],
    )
    num_events_raw = con.create_table(
        "asof_ratio_window_num_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(hours=10),
                "event": "purchase",
                "amount": 10.0,
            },
        ],
    )
    den_events_raw = con.create_table(
        "asof_ratio_window_den_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": start + dt.timedelta(days=w, hours=9, minutes=30),
                "event": "session",
                "n": 1.0,
            }
            for w in range(4)  # day0, day1, day2, day3
        ],
    )

    exposures = first_exposures(exposure_events, experiment)
    num_events = metric_events(
        num_events_raw, ratio_metric, value_column="amount", part="numerator"
    )
    den_events = metric_events(den_events_raw, ratio_metric, value_column="n", part="denominator")
    num_panel = unit_day_panel(exposures, num_events, experiment, metric_name=ratio_metric.name)
    den_panel = unit_day_panel(exposures, den_events, experiment, metric_name=ratio_metric.name)

    ags = asof_group_summary(num_panel, ratio_metric, den_panel=den_panel).execute()
    ags["ds"] = ags["ds"].dt.date
    day2 = (start + dt.timedelta(days=2)).date()
    day3 = (start + dt.timedelta(days=3)).date()

    row2 = ags[ags["ds"] == day2].iloc[0]
    row3 = ags[ags["ds"] == day3].iloc[0]

    # Day 2: within the numerator's 3-day window (ds=day2 < window_end=day3)
    # all three session events (day0,1,2) count.
    assert abs(_total(row2, "ref_den", "cden1") - 3.0) < 1e-9

    # Day 3: at/past the numerator's window_end - masked to 0 despite being within the
    # denominator's own 8-day window. Frozen at 3.0, not 4.0.
    assert abs(_total(row3, "ref_den", "cden1") - 3.0) < 1e-9, (
        "denominator masking must use the numerator's canonical window_days, "
        "not the denominator's own -- got 4.0-shaped drift if wrong"
    )


def test_asof_retention_default_gate_admits_units_from_band_open(
    exposures, page_view_events, experiment, bounded_retention_metric
):
    """The default gate admits a unit the day its band OPENS, with a
    provisional value: 0 until its first in-band return, then 1. Absent
    before band open, never zero-filled.
    """
    events = metric_events(page_view_events, bounded_retention_metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=bounded_retention_metric.name)
    asof = asof_group_summary(panel, bounded_retention_metric).execute()

    # Band [fe+2, fe+4), gate at fe+2: u1 (Aug1 exposure) opens Aug3, y=1 from Aug3
    # (view Aug3); u2 opens Aug3, y=0 (view Aug2 pre-band); u3 opens Aug4, y=0 (view Aug3<Aug4).
    asof["ds"] = asof["ds"].dt.date
    assert asof[asof["ds"] < dt.date(2025, 8, 3)].empty, "absent before any band opens"

    treatment = asof[asof["group_id"] == "treatment"].set_index("ds")
    # Aug 3: only u1's band is open -> n = 1, total y = 1
    assert treatment.loc[dt.date(2025, 8, 3), "n"] == 1
    assert _total(treatment.loc[dt.date(2025, 8, 3)]) == 1.0
    # Aug 4: u3's band opens with a provisional 0 -> n = 2, total y still 1
    assert treatment.loc[dt.date(2025, 8, 4), "n"] == 2
    assert _total(treatment.loc[dt.date(2025, 8, 4)]) == 1.0

    control = asof[asof["group_id"] == "control"].set_index("ds")
    assert control.loc[dt.date(2025, 8, 3), "n"] == 1
    assert _total(control.loc[dt.date(2025, 8, 3)]) == 0.0


def test_asof_retention_default_gate_value_ratchets_from_provisional_zero(
    con, experiment, bounded_retention_metric
):
    """A unit whose in-band return lands AFTER its band opens is present
    with 0 from band open, flips to 1 the day the return lands, and (the
    band being bounded) stays 1 after the band closes."""
    exposure_rows = con.create_table(
        "ratchet_exposures",
        obj=[
            {
                "unit_id": "r1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    view_rows = con.create_table(
        "ratchet_page_views",
        obj=[
            # Band [Aug 3, Aug 5): the only return lands Aug 4, one day
            # after the band opens.
            {"unit_id": "r1", "ts": dt.datetime(2025, 8, 4, 10, 0, 0), "event": "page_view"},
        ],
    )
    exposures_r = first_exposures(exposure_rows, experiment)
    events = metric_events(view_rows, bounded_retention_metric)
    panel = unit_day_panel(
        exposures_r, events, experiment, metric_name=bounded_retention_metric.name
    )
    asof = asof_group_summary(panel, bounded_retention_metric).execute()

    asof["ds"] = asof["ds"].dt.date
    by_day = asof.set_index("ds")
    assert asof[asof["ds"] < dt.date(2025, 8, 3)].empty, "absent before band open"
    assert _total(by_day.loc[dt.date(2025, 8, 3)]) == 0.0, "provisional 0 at band open"
    assert _total(by_day.loc[dt.date(2025, 8, 4)]) == 1.0, "ratchets the day the return lands"
    assert _total(by_day.loc[dt.date(2025, 8, 5)]) == 1.0, "frozen after band close"
    assert (by_day["n"] == 1).all(), "present on every day from band open"


def test_asof_retention_completed_windows_only_reproduces_maturity_gate(
    exposures, page_view_events, experiment, bounded_retention_metric
):
    """completed_windows_only=True restores the band-close gate: each
    as-of day reports only matured units, with a final y - exactly the
    default behavior before the gate moved to band open.
    """
    events = metric_events(page_view_events, bounded_retention_metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=bounded_retention_metric.name)
    asof = asof_group_summary(
        panel, bounded_retention_metric, completed_windows_only=True
    ).execute()

    # Band [fe+2, fe+4), maturity fe+4: u1 matures Aug5 y=1 (view Aug3); u2 matures
    # Aug5 y=0 (view Aug2 only); u3 matures Aug6 y=0 (view Aug3<Aug4). None present before maturity.
    asof["ds"] = asof["ds"].dt.date
    assert asof[asof["ds"] < dt.date(2025, 8, 5)].empty

    treatment = asof[asof["group_id"] == "treatment"].set_index("ds")
    # Aug 5: only u1 matured -> n = 1, total y = 1
    assert treatment.loc[dt.date(2025, 8, 5), "n"] == 1
    assert _total(treatment.loc[dt.date(2025, 8, 5)]) == 1.0
    # Aug 6: u3 joins the matured cohort -> n = 2, total y still 1
    assert treatment.loc[dt.date(2025, 8, 6), "n"] == 2
    assert _total(treatment.loc[dt.date(2025, 8, 6)]) == 1.0


def test_asof_retention_unbounded_band_ratchets_and_never_freezes(con, experiment):
    """An unbounded band [N, inf) is accepted on the as-of view: the value
    counts any return at or past the threshold, and a return landing
    after an arbitrary later date (here: after a bounded [2, 4) band
    would already have closed) still flips it - the ratchet never
    freezes."""
    unbounded = RetentionMetric(
        name="retained_open_asof", entity="unit_id", fact="page_view", threshold_days=2
    )
    exposure_rows = con.create_table(
        "unbounded_asof_exposures",
        obj=[
            {
                "unit_id": "late1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    view_rows = con.create_table(
        "unbounded_asof_page_views",
        obj=[
            # Tenure 4: past a bounded [2, 4) band's close, in [2, inf).
            {"unit_id": "late1", "ts": dt.datetime(2025, 8, 5, 10, 0, 0), "event": "page_view"},
        ],
    )
    exposures_l = first_exposures(exposure_rows, experiment)
    events = metric_events(view_rows, unbounded)
    panel = unit_day_panel(exposures_l, events, experiment, metric_name=unbounded.name)
    asof = asof_group_summary(panel, unbounded).execute()

    asof["ds"] = asof["ds"].dt.date
    by_day = asof.set_index("ds")
    assert asof[asof["ds"] < dt.date(2025, 8, 3)].empty, "absent before the band opens (fe+2)"
    assert _total(by_day.loc[dt.date(2025, 8, 3)]) == 0.0
    assert _total(by_day.loc[dt.date(2025, 8, 4)]) == 0.0
    assert _total(by_day.loc[dt.date(2025, 8, 5)]) == 1.0, "the late return still flips it"
    assert _total(by_day.loc[dt.date(2025, 8, 6)]) == 1.0


def test_asof_retention_gate_comparison_band_start_starts_earlier_same_final_day(con):
    """Three cohorts on seed-shaped dates, band [7, 14): the default gate
    starts the series at first exposure + 7 (2025-01-22), the
    completed-windows gate at first exposure + 14 (2025-01-29), and once
    every band has closed the two series report identical final-day
    aggregates."""
    experiment = Experiment(
        name="exp_gate_cmp",
        unit="unit_id",
        start=dt.datetime(2025, 1, 15),
        end=dt.datetime(2025, 2, 5),
        control_group="control",
        exposure="gate_cmp_exposure",
        plan=AnalysisPlan(),
    )
    metric = RetentionMetric(
        name="d7_gate_cmp", entity="unit_id", fact="page_view", threshold_days=(7, 14)
    )
    exposure_rows = con.create_table(
        "gate_cmp_exposures",
        obj=[
            {
                "unit_id": unit,
                "ts": dt.datetime(2025, 1, day, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_gate_cmp",
                "group_id": group,
            }
            for unit, day, group in [
                ("c1", 15, "treatment"),  # band [Jan 22, Jan 29)
                ("c2", 16, "control"),  # band [Jan 23, Jan 30)
                ("c3", 17, "treatment"),  # band [Jan 24, Jan 31)
            ]
        ],
    )
    view_rows = con.create_table(
        "gate_cmp_page_views",
        obj=[
            # c1 returns at tenure 7 (in band); c2 never; c3 at tenure 13
            # (last in-band day).
            {"unit_id": "c1", "ts": dt.datetime(2025, 1, 22, 10, 0, 0), "event": "page_view"},
            {"unit_id": "c3", "ts": dt.datetime(2025, 1, 30, 10, 0, 0), "event": "page_view"},
        ],
    )
    exposures_c = first_exposures(exposure_rows, experiment)
    events = metric_events(view_rows, metric)
    panel = unit_day_panel(exposures_c, events, experiment, metric_name=metric.name)

    monitoring = asof_group_summary(panel, metric).execute()
    decision = asof_group_summary(panel, metric, completed_windows_only=True).execute()
    monitoring["ds"] = monitoring["ds"].dt.date
    decision["ds"] = decision["ds"].dt.date

    assert monitoring["ds"].min() == dt.date(2025, 1, 22), "series opens at fe + band_start"
    assert decision["ds"].min() == dt.date(2025, 1, 29), "old gate opened at fe + band_end"

    last = monitoring["ds"].max()
    assert decision["ds"].max() == last
    final_m = monitoring[monitoring["ds"] == last].set_index("group_id")
    final_d = decision[decision["ds"] == last].set_index("group_id")
    for group_id in ("treatment", "control"):
        assert final_m.loc[group_id, "n"] == final_d.loc[group_id, "n"]
        for field in ("ref_y", "cy1", "cy2"):
            assert final_m.loc[group_id, field] == final_d.loc[group_id, field]


def test_asof_cumulative_series_diverges_from_run_under_arm_dependent_return_timing(con):
    """Property test (Decision 6 / spec Blocker 1's failure mode (b)), not a
    bug test: the cumulative as-of series equals the whole-window ``run()``
    estimand only when the arms' return-TIMING distributions agree, even if
    their eventual return RATES agree. Here they deliberately don't.

    Band [7, 13): both arms have IDENTICAL eventual return rates (3 of 5
    units in each arm return somewhere in-band - zero true windowed
    lift). Control's returners are spread across the band (tenure 7, 9,
    12); treatment's are concentrated at its open (tenure 7, 8, 8). By
    day 8 - before the band has closed - treatment's cumulative series
    reads far ahead of control's purely because its returns landed
    earlier, not because more of it returned. This is exactly the
    variance-for-a-timing-assumption trade `completed_windows_only=False`
    accepts (see `asof_group_summary`'s docstring): the cumulative series
    is a monitoring artifact, not an unbiased estimate of the windowed
    lift at every date. `run()` - the windowed estimand computed here via
    `unit_totals` + `group_summary` - is unaffected by the timing
    difference and stays flat.
    """
    experiment = Experiment(
        name="exp_timing",
        unit="unit_id",
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 2, 1),  # past every band close, nothing censors
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    metric = RetentionMetric(
        name="d7_timing", entity="unit_id", fact="page_view", threshold_days=(7, 13)
    )
    exposure_rows = con.create_table(
        "timing_exposures",
        obj=[
            {
                "unit_id": unit,
                "ts": dt.datetime(2025, 1, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_timing",
                "group_id": group,
            }
            for unit, group in [
                ("c1", "control"),
                ("c2", "control"),
                ("c3", "control"),
                ("c4", "control"),
                ("c5", "control"),
                ("t1", "treatment"),
                ("t2", "treatment"),
                ("t3", "treatment"),
                ("t4", "treatment"),
                ("t5", "treatment"),
            ]
        ],
    )
    view_rows = con.create_table(
        "timing_page_views",
        obj=[
            # control: 3 of 5 return, spread across the band (tenure 7, 9, 12)
            {"unit_id": "c1", "ts": dt.datetime(2025, 1, 8, 10, 0, 0), "event": "page_view"},
            {"unit_id": "c2", "ts": dt.datetime(2025, 1, 10, 10, 0, 0), "event": "page_view"},
            {"unit_id": "c3", "ts": dt.datetime(2025, 1, 13, 10, 0, 0), "event": "page_view"},
            # treatment: 3 of 5 return, concentrated at the band's open
            # (tenure 7, 8, 8) - same COUNT as control, earlier TIMING
            {"unit_id": "t1", "ts": dt.datetime(2025, 1, 8, 10, 0, 0), "event": "page_view"},
            {"unit_id": "t2", "ts": dt.datetime(2025, 1, 9, 10, 0, 0), "event": "page_view"},
            {"unit_id": "t3", "ts": dt.datetime(2025, 1, 9, 11, 0, 0), "event": "page_view"},
            # c4, c5, t4, t5: never return
        ],
    )
    exposures_t = first_exposures(exposure_rows, experiment)
    events = metric_events(view_rows, metric)

    # - (a) run(): the windowed estimand, unaffected by timing
    spine, stats = unit_day_spine_stats(exposures_t, events, experiment, metric.name)
    totals = unit_totals(spine, stats, metric, experiment)
    whole = group_summary(totals).execute().set_index("group_id")
    rate = _total(whole) / whole["n"]
    assert rate["control"] == pytest.approx(0.6)
    assert rate["treatment"] == pytest.approx(0.6)
    assert rate["treatment"] - rate["control"] == pytest.approx(0.0), (
        "identical eventual return rates -> zero true windowed lift"
    )

    # - (b) the cumulative as-of series, mid-band (day 8, before the band
    # closes at day 13): treatment reads far ahead purely from timing
    panel = unit_day_panel(exposures_t, events, experiment, metric_name=metric.name)
    asof = asof_group_summary(panel, metric).execute()
    asof["ds"] = asof["ds"].dt.date
    mid_band = asof[asof["ds"] == dt.date(2025, 1, 9)].set_index("group_id")
    mid_rate = _total(mid_band) / mid_band["n"]
    assert mid_rate["control"] == pytest.approx(0.2), "only c1's tenure-7 return has landed"
    assert mid_rate["treatment"] == pytest.approx(0.6), "t1/t2/t3 have all landed by tenure 8"
    assert mid_rate["treatment"] - mid_rate["control"] == pytest.approx(0.4), (
        "large apparent mid-flight lift from timing alone, though (a) proves the true "
        "windowed lift is zero"
    )

    # once the band has fully closed the cumulative series converges back
    # to run()'s windowed rate for both arms
    last_day = asof["ds"].max()
    final = asof[asof["ds"] == last_day].set_index("group_id")
    final_rate = _total(final) / final["n"]
    assert final_rate["control"] == pytest.approx(0.6)
    assert final_rate["treatment"] == pytest.approx(0.6)


def test_asof_retention_y_is_binary_not_a_running_sum(con, experiment, bounded_retention_metric):
    """Retention is a MAX over the band, not a SUM - two returns is still 1."""
    exposure_rows = con.create_table(
        "twice_exposures",
        obj=[
            {
                "unit_id": "twice",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_test",
                "group_id": "treatment",
            }
        ],
    )
    view_rows = con.create_table(
        "twice_page_views",
        obj=[
            {"unit_id": "twice", "ts": dt.datetime(2025, 8, 3, 10, 0, 0), "event": "page_view"},
            {"unit_id": "twice", "ts": dt.datetime(2025, 8, 3, 11, 0, 0), "event": "page_view"},
            {"unit_id": "twice", "ts": dt.datetime(2025, 8, 4, 10, 0, 0), "event": "page_view"},
        ],
    )
    exposures_twice = first_exposures(exposure_rows, experiment)
    events = metric_events(view_rows, bounded_retention_metric)
    panel = unit_day_panel(
        exposures_twice, events, experiment, metric_name=bounded_retention_metric.name
    )
    asof = asof_group_summary(panel, bounded_retention_metric).execute()

    assert set(asof["n"].tolist()) == {1}
    assert set(asof["ref_y"].tolist()) == {1.0}, "three events in band must still be y = 1"
    assert set(asof["cy1"].tolist()) == {0.0}
    assert set(asof["cy2"].tolist()) == {0.0}


def test_asof_retention_converges_to_whole_window_estimate(
    exposures, page_view_events, experiment, bounded_retention_metric
):
    """The last as-of day must match unit_totals + group_summary exactly."""
    events = metric_events(page_view_events, bounded_retention_metric)
    panel = unit_day_panel(exposures, events, experiment, metric_name=bounded_retention_metric.name)
    asof = asof_group_summary(panel, bounded_retention_metric).execute()
    spine, stats = unit_day_spine_stats(
        exposures, events, experiment, bounded_retention_metric.name
    )
    totals = unit_totals(spine, stats, bounded_retention_metric, experiment)
    whole = group_summary(totals).execute()

    last_day = asof["ds"].max()
    final = asof[asof["ds"] == last_day].set_index("group_id")
    for group_id, row in whole.set_index("group_id").iterrows():
        assert final.loc[group_id, "n"] == row["n"]
        for field in ("ref_y", "cy1", "cy2"):
            assert final.loc[group_id, field] == row[field]


def test_asof_retention_completed_windows_only_with_unbounded_band_raises(panel, retention_metric):
    """An unbounded band never completes, so asking for completed windows
    only is a contradiction - raise rather than silently pick a gate."""
    import pytest

    with pytest.raises(InvalidRequestError) as exc_info:
        asof_group_summary(panel, retention_metric, completed_windows_only=True)
    assert exc_info.value.code == "query.builders.asof_group_summary"
    assert exc_info.value.context["name"] == "retained"
    assert exc_info.value.context["threshold_days"] == 2


def test_cohort_group_summary_keys_on_exposure_date(
    exposures, page_view_events, experiment, bounded_retention_metric
):
    """ds is the cohort's exposure date, not activity and not maturity."""
    events = metric_events(page_view_events, bounded_retention_metric)
    spine, stats = unit_day_spine_stats(
        exposures, events, experiment, bounded_retention_metric.name
    )
    cohorts = cohort_group_summary(spine, stats, bounded_retention_metric, experiment).execute()

    # u1 (treatment) and u2 (control) exposed Aug 1; u3 (treatment) Aug 2.
    cohorts["ds"] = cohorts["ds"].dt.date
    assert sorted(cohorts["ds"].unique().tolist()) == [
        dt.date(2025, 8, 1),
        dt.date(2025, 8, 2),
    ]
    aug1_treatment = cohorts[
        (cohorts["ds"] == dt.date(2025, 8, 1)) & (cohorts["group_id"] == "treatment")
    ]
    assert aug1_treatment["n"].iloc[0] == 1  # u1 only
    assert _total(aug1_treatment.iloc[0]) == 1.0  # u1 returned Aug 3, in band


def test_cohort_group_summary_omits_immature_cohorts(
    con, page_view_events, bounded_retention_metric
):
    """A cohort appears only once its band has closed within the data."""
    exposure_rows = con.create_table(
        "cohort_exposures",
        obj=[
            {
                "unit_id": "early",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_cohort",
                "group_id": "treatment",
            },
            {
                "unit_id": "late",
                "ts": dt.datetime(2025, 8, 3, 9, 0, 0),
                "event": "exposure",
                "experiment_id": "exp_cohort",
                "group_id": "treatment",
            },
        ],
    )
    # end = Aug 5. Band is 4 days, so the Aug-1 cohort matures Aug 5 (included)
    # and the Aug-3 cohort would mature Aug 7 (excluded).
    cohort_experiment = Experiment(
        name="exp_cohort",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 5),
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    exposures_c = first_exposures(exposure_rows, cohort_experiment)
    events = metric_events(page_view_events, bounded_retention_metric)
    spine, stats = unit_day_spine_stats(
        exposures_c, events, cohort_experiment, bounded_retention_metric.name
    )
    with pytest.warns(IncrementWarning) as rec:
        cohorts = cohort_group_summary(
            spine, stats, bounded_retention_metric, cohort_experiment
        ).execute()
    assert "frame.censoring.dropped_units" in warning_codes(rec)

    assert cohorts["ds"].dt.date.tolist() == [dt.date(2025, 8, 1)]


def test_cohort_group_summary_totals_match_whole_window(
    exposures, page_view_events, experiment, bounded_retention_metric
):
    """Summing the matured cohorts reproduces the whole-window moments."""
    events = metric_events(page_view_events, bounded_retention_metric)
    spine, stats = unit_day_spine_stats(
        exposures, events, experiment, bounded_retention_metric.name
    )
    cohorts = cohort_group_summary(spine, stats, bounded_retention_metric, experiment).execute()
    whole = group_summary(unit_totals(spine, stats, bounded_retention_metric, experiment)).execute()

    by_group = cohorts.assign(total_y=_total(cohorts)).groupby("group_id")[["n", "total_y"]].sum()
    for group_id, row in whole.set_index("group_id").iterrows():
        assert by_group.loc[group_id, "n"] == row["n"]
        assert by_group.loc[group_id, "total_y"] == _total(row)


def test_cohort_group_summary_centers_fixed_x_after_maturity_filter(con):
    """Cohort CUPED moments use only admitted units, not immature extreme x."""
    exposure_rows = [
        {
            "unit_id": unit_id,
            "ts": dt.datetime(2025, 8, day, 9, 0, 0),
            "event": "exposure",
            "experiment_id": "cohort_cuped",
            "group_id": "treatment",
        }
        for unit_id, day in (("early_a", 1), ("early_b", 1), ("late_extreme", 3))
    ]
    return_rows = [
        {
            "unit_id": "early_a",
            "ts": dt.datetime(2025, 8, 2, 10, 0, 0),
            "event": "return",
            "experiment_id": None,
            "group_id": None,
        },
        {
            "unit_id": "early_b",
            "ts": dt.datetime(2025, 8, 5, 10, 0, 0),
            "event": "return",
            "experiment_id": None,
            "group_id": None,
        },
        {
            "unit_id": "late_extreme",
            "ts": dt.datetime(2025, 8, 4, 10, 0, 0),
            "event": "return",
            "experiment_id": None,
            "group_id": None,
        },
    ]
    exposure_events = con.create_table(
        "cohort_cuped_events",
        obj=exposure_rows + return_rows,
        overwrite=True,
    )
    experiment = Experiment(
        name="cohort_cuped",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 5),
        control_group="control",
        exposure="exposure",
        n_pre_periods=14,
        plan=AnalysisPlan(),
    )
    metric = RetentionMetric(
        name="returned",
        entity="unit_id",
        fact="return",
        threshold_days=(1, 4),
    )
    exposures = first_exposures(
        exposure_events.filter(exposure_events.event == "exposure"), experiment
    )
    events = metric_events(exposure_events, metric)
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    pre_stats = ibis.memtable(
        {
            "unit_id": ["early_a", "early_b", "late_extreme"],
            "sum_value": [1.0, 3.0, 1000.0],
        }
    )

    cohorts = cohort_group_summary(
        spine,
        stats,
        metric,
        experiment,
        pre_stats=pre_stats,
        warn_on_censoring=False,
    ).execute()
    cohort_keys = cohorts.assign(ds=cohorts["ds"].dt.date)[
        ["ds", "experiment_id", "metric", "group_id"]
    ].to_dict("records")
    assert len(cohorts) == 1
    assert cohort_keys == [
        {
            "ds": dt.date(2025, 8, 1),
            "experiment_id": "cohort_cuped",
            "metric": "returned",
            "group_id": "treatment",
        }
    ], "only the mature Aug-1 cohort may be admitted"
    row = cohorts.iloc[0]
    expected = _centered_x_oracle([{"x": 1.0, "y": 1.0}, {"x": 3.0, "y": 0.0}])
    for field, value in expected.items():
        assert row[field] == pytest.approx(value, abs=1e-12)

    assert row["n"] == 2
    assert row["ref_x"] == pytest.approx(2.0)
    assert row["cx1"] == pytest.approx(0.0, abs=1e-12)
    assert row["cx2"] == pytest.approx(2.0)
    assert row["cxy"] == pytest.approx(-1.0)


def test_cohort_group_summary_rejects_unbounded_and_non_retention(
    exposures, purchase_metric_events, retention_metric, mean_metric, experiment
):
    import pytest

    spine, stats = unit_day_spine_stats(
        exposures, purchase_metric_events, experiment, mean_metric.name
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        cohort_group_summary(spine, stats, retention_metric, experiment)
    assert exc_info.value.code == "query.builders.cohort_group_summary_unbounded_band"
    assert exc_info.value.context["name"] == "retained"
    assert exc_info.value.context["threshold_days"] == 2
    with pytest.raises(UnsupportedRequestError) as exc_info:
        cohort_group_summary(spine, stats, mean_metric, experiment)
    assert exc_info.value.code == "query.builders.cohort_group_summary"
    assert exc_info.value.context["metric_type"] == "MeanMetric"


def test_spine_extent_is_union_over_metrics(con):
    """The spine's right edge must not depend on which metric you ask for."""
    from increment.query.builders import panel_spine, union_event_horizon

    exposures = con.create_table(
        "spine_union_exp",
        obj=pa.table(
            {
                "unit_id": ["u1"],
                "experiment_id": ["e1"],
                "group_id": ["t"],
                "first_exposure_ts": [dt.datetime(2026, 1, 1)],
            }
        ),
        overwrite=True,
    )
    exp = Experiment(
        name="e1",
        unit="unit_id",
        exposure="x",
        control_group="c",
        start=dt.datetime(2026, 1, 1),
        plan=AnalysisPlan(),
    )

    early = con.create_table(
        "spine_union_early",
        obj=pa.table(
            {"unit_id": ["u1"], "ts": [dt.datetime(2026, 1, 3)], "metric": ["a"], "value": [1.0]}
        ),
        overwrite=True,
    )
    late = con.create_table(
        "spine_union_late",
        obj=pa.table(
            {"unit_id": ["u1"], "ts": [dt.datetime(2026, 1, 9)], "metric": ["b"], "value": [1.0]}
        ),
        overwrite=True,
    )

    end = union_event_horizon([early, late])
    spine = panel_spine(exposures, exp, end_date=end)
    days = sorted(r["ds"] for r in con.to_pyarrow(spine).to_pylist())
    assert days[-1] == dt.date(2026, 1, 9), "spine must extend to the LATEST event across metrics"
    assert len(days) == 9, "2026-01-01 .. 2026-01-09 inclusive"


def test_union_event_horizon_refuses_empty_input():
    from increment.query.builders import union_event_horizon

    with pytest.raises(InvalidRequestError) as exc_info:
        union_event_horizon([])
    assert exc_info.value.code == "query.builders.union_event_horizon"


def test_union_event_horizon_ignores_eventless_table(con):
    from increment.query.builders import union_event_horizon

    full = con.create_table(
        "horizon_full",
        obj=pa.table({"ts": [dt.datetime(2026, 1, 5), dt.datetime(2026, 1, 9)]}),
        overwrite=True,
    )
    empty = con.create_table(
        "horizon_empty",
        obj=pa.table({"ts": pa.array([], type=pa.timestamp("us"))}),
        overwrite=True,
    )
    # ibis/duckdb can't promote a bare scalar spanning multiple base tables straight
    # to a table (`RelationError`); wrap in a single-row memtable to force materialization.
    horizon = union_event_horizon([full, empty])
    got = con.execute(ibis.memtable({"_": [1]}).mutate(horizon=horizon))["horizon"][0]
    assert got.date() == dt.date(2026, 1, 9), "an eventless source must not NULL the horizon"


# unit_day_stats / aggregate_stats: sparse per-unit-day stats, aggregation deferred to read time


def test_unit_day_stats_are_sparse_and_aggregation_free(con):
    """One row per observed (unit, day, source). No zero-fill, no aggregation applied."""
    from increment.query.builders import unit_day_stats

    events = con.create_table(
        "unit_day_stats_ev",
        obj=pa.table(
            {
                "unit_id": ["u1", "u1", "u1", "u3"],
                "ts": [
                    dt.datetime(2026, 1, 1, 10),
                    dt.datetime(2026, 1, 1, 18),
                    dt.datetime(2026, 1, 3, 9),
                    dt.datetime(2026, 1, 1, 12),
                ],
                "metric": ["m"] * 4,
                "value": [15.0, 25.0, 7.0, 18.0],
            }
        ),
        overwrite=True,
    )

    got = {
        (r["unit_id"], r["ds"]): r
        for r in con.to_pyarrow(unit_day_stats(events, source_key="orders:gmv")).to_pylist()
    }

    assert len(got) == 3, "sparse: u1 has no row for 2026-01-02"
    u1d1 = got[("u1", dt.date(2026, 1, 1))]
    assert u1d1["n_events"] == 2
    assert u1d1["sum_value"] == pytest.approx(40.0)
    assert u1d1["min_value"] == pytest.approx(15.0)
    assert u1d1["max_value"] == pytest.approx(25.0)
    assert u1d1["source_key"] == "orders:gmv"


@pytest.mark.parametrize(
    "aggregation,expected",
    [
        ("sum", 47.0),
        ("count", 3),
        ("avg_event", 47.0 / 3),
        ("min", 7.0),
        ("max", 25.0),
        ("count_distinct", 2),
    ],
)
def test_every_aggregation_recovers_from_unit_day_stats(con, aggregation, expected):
    """Each supported aggregation is derivable from the same sufficient statistics."""
    from increment.query.builders import aggregate_stats, unit_day_stats

    events = con.create_table(
        "agg_stats_ev",
        obj=pa.table(
            {
                "unit_id": ["u1"] * 3,
                "ts": [
                    dt.datetime(2026, 1, 1, 10),
                    dt.datetime(2026, 1, 1, 18),
                    dt.datetime(2026, 1, 3, 9),
                ],
                "metric": ["m"] * 3,
                "value": [15.0, 25.0, 7.0],
            }
        ),
        overwrite=True,
    )
    stats = unit_day_stats(events, source_key="s")
    got = con.to_pyarrow(aggregate_stats(stats, aggregation=aggregation)).to_pylist()
    assert got[0]["y"] == pytest.approx(expected)


def test_aggregate_stats_refuses_unknown_aggregation(con):
    from increment.query.builders import aggregate_stats, unit_day_stats

    events = con.create_table(
        "agg_stats_unknown_ev",
        obj=pa.table(
            {
                "unit_id": ["u1"],
                "ts": [dt.datetime(2026, 1, 1, 10)],
                "metric": ["m"],
                "value": [15.0],
            }
        ),
        overwrite=True,
    )
    stats = unit_day_stats(events, source_key="s")
    with pytest.raises(InvalidRequestError) as exc_info:
        aggregate_stats(stats, aggregation="bogus")
    assert exc_info.value.code == "query.builders.unknown_aggregation"
    assert exc_info.value.context["aggregation"] == "bogus"


def test_max_ranges_over_raw_events_not_daily_totals(con):
    """max = max(max_value),
    the raw-event extreme - NOT the max daily total (that would be 40)."""
    from increment.query.builders import aggregate_stats, unit_day_stats

    events = con.create_table(
        "minmax_raw_ev",
        obj=pa.table(
            {
                "unit_id": ["u1"] * 3,
                "ts": [
                    dt.datetime(2026, 1, 1, 10),
                    dt.datetime(2026, 1, 1, 18),
                    dt.datetime(2026, 1, 3, 9),
                ],
                "metric": ["m"] * 3,
                "value": [15.0, 25.0, 7.0],
            }
        ),
        overwrite=True,
    )
    stats = unit_day_stats(events, source_key="s")
    got = con.to_pyarrow(aggregate_stats(stats, aggregation="max")).to_pylist()
    assert got[0]["y"] == pytest.approx(25.0)


# breakout_property_table - pre-exposure scoping (Property.as_of)


@pytest.fixture
def prop_rows(con):
    """Two property rows per unit: one before exposure, one after."""
    table_name = "breakout_prop_rows_speg"
    if table_name in con.list_tables():
        return con.table(table_name)
    return con.create_table(
        table_name,
        obj=[
            {"unit_id": "u1", "ts": dt.datetime(2025, 6, 1, 8, 0, 0), "country": "US"},
            {"unit_id": "u1", "ts": dt.datetime(2025, 6, 3, 8, 0, 0), "country": "CA"},
            {"unit_id": "u2", "ts": dt.datetime(2025, 6, 1, 8, 0, 0), "country": "MX"},
            {"unit_id": "u2", "ts": dt.datetime(2025, 6, 3, 8, 0, 0), "country": "US"},
            {"unit_id": "u3", "ts": dt.datetime(2025, 6, 1, 8, 0, 0), "country": "FR"},
            {"unit_id": "u5", "ts": dt.datetime(2025, 6, 3, 8, 0, 0), "country": "JP"},
        ],
    )


def test_breakout_property_table_scopes_to_pre_exposure(con, prop_rows, experiment):
    """[speg] With exposures supplied, a unit's property value is the latest
    row STRICTLY before first_exposure_ts - a post-exposure value must never
    reach the breakout, and a unit with no pre-exposure row is absent (it
    lands in the caller's "__null__" bin)."""
    exposure_events = con.create_table(
        "breakout_prop_exposures_speg",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
            {
                "unit_id": "u5",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
        ],
    )
    # Exposures predate August: the shared `experiment` fixture's own start
    # (Aug 1) would wrongly refuse them, so scope enrollment to June instead.
    exp = experiment.model_copy(update={"start": dt.datetime(2025, 6, 1)})
    exposures = first_exposures(exposure_events, exp)
    result = (
        breakout_property_table(prop_rows, "country", exposures)
        .execute()
        .set_index("unit_id")
        .sort_index()
    )
    # u1/u2: pre-exposure value wins over the post-exposure one.
    assert result.loc["u1", "country"] == "US"
    assert result.loc["u2", "country"] == "MX"
    # u3: single pre-exposure row survives.
    assert result.loc["u3", "country"] == "FR"
    # u5: only a POST-exposure row exists -> absent from the scoped table.
    assert "u5" not in result.index


def test_breakout_property_table_unscoped_takes_latest(con, prop_rows):
    """[speg] With exposures None (as_of='static'), the historical behaviour:
    the latest value over the whole source, post-exposure rows included."""
    result = (
        breakout_property_table(prop_rows, "country", None)
        .execute()
        .set_index("unit_id")
        .sort_index()
    )
    assert result.loc["u1", "country"] == "CA"
    assert result.loc["u2", "country"] == "US"
    assert result.loc["u3", "country"] == "FR"
    assert result.loc["u5", "country"] == "JP"


def test_breakout_property_table_scoping_renders_across_backends(con, prop_rows, experiment):
    """The pre-exposure-scoped property dedup compiles across backends."""
    exposure_events = con.create_table(
        "breakout_prop_render_exposures",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
        ],
    )
    exposures = first_exposures(exposure_events, experiment)
    scoped = breakout_property_table(prop_rows, "country", exposures)
    for dialect in ("snowflake", "bigquery", "postgres"):
        sql = ibis.to_sql(scoped, dialect=dialect)
        assert "first_exposure_ts" in sql, dialect


def test_breakout_property_table_scoping_renders_across_backends_bool_property(con, experiment):
    """[speg] Same scoped dedup, but for a bool-typed property. ``bool`` is
    a first-class breakout dtype (``Definitions._check_breakouts``); this
    checks the dedup tiebreak (max-ts self-join + ``max()`` on the
    winning value) renders on other dialects and executes the correct
    pre-exposure-only winner on DuckDB."""
    bool_prop_rows = con.create_table(
        "breakout_prop_rows_bool_speg",
        obj=[
            {"unit_id": "u1", "ts": dt.datetime(2025, 6, 1, 8, 0, 0), "active": True},
            {"unit_id": "u1", "ts": dt.datetime(2025, 6, 3, 8, 0, 0), "active": False},
            {"unit_id": "u2", "ts": dt.datetime(2025, 6, 1, 8, 0, 0), "active": False},
        ],
    )
    exposure_events = con.create_table(
        "breakout_prop_render_exposures_bool",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 6, 2, 10, 0, 0),
                "event": "exposure",
                "experiment_id": "exp",
                "group_id": "control",
            },
        ],
    )
    # Exposures predate August: scope enrollment to June like the sibling test above.
    exp = experiment.model_copy(update={"start": dt.datetime(2025, 6, 1)})
    exposures = first_exposures(exposure_events, exp)
    scoped = breakout_property_table(bool_prop_rows, "active", exposures)
    # Boolean-or aggregate is dialect-specific: snowflake=BOOLOR_AGG, bigquery=LOGICAL_OR,
    # postgres=BOOL_OR - verified against each dialect's actual rendered SQL.
    dialect_bool_agg = {
        "snowflake": "BOOLOR_AGG",
        "bigquery": "LOGICAL_OR",
        "postgres": "BOOL_OR",
    }
    for dialect in ("snowflake", "bigquery", "postgres"):
        sql = ibis.to_sql(scoped, dialect=dialect)
        assert "first_exposure_ts" in sql, dialect
        # The dedup tiebreak's max() must compile to the dialect's boolean aggregate, not a
        # bare MAX(bool_col); DuckDB accepts that natively, so only cross-dialect rendering catches it.
        assert dialect_bool_agg[dialect] in sql.upper(), (
            f"{dialect}: expected {dialect_bool_agg[dialect]}() for the boolean "
            f"tiebreak aggregate, got:\n{sql}"
        )

    # And it still executes correctly on DuckDB: u1's only PRE-exposure
    # row (True) wins even though a later post-exposure row (False) exists.
    result = scoped.execute().set_index("unit_id").sort_index()
    assert bool(result.loc["u1", "active"]) is True
    assert bool(result.loc["u2", "active"]) is False


# Declared day boundary (Experiment.day_boundary)


def test_day_boundary_offset_refuses_unparseable_string():
    from increment.query.builders import day_boundary_offset

    with pytest.raises(InvalidRequestError) as exc_info:
        day_boundary_offset("not-a-boundary")
    assert exc_info.value.code == "query.builders.invalid_day_boundary"
    assert exc_info.value.context["day_boundary"] == "not-a-boundary"


class TestDayBoundary:
    """Day-bucketing honors ``Experiment.day_boundary``.

    Canonical fixture: one unit exposed at ``2025-08-02 03:10 UTC``.
    Under ``day_boundary="UTC-05:00"`` that instant is
    ``2025-08-01 22:10`` local, so the exposure day is **Aug 1** — one
    day earlier than the UTC cast.
    """

    @staticmethod
    def _experiment(
        end: dt.datetime = dt.datetime(2025, 8, 6), day_boundary: str = "UTC-05:00"
    ) -> Experiment:
        return Experiment(
            name="exp_test",
            unit="unit_id",
            start=dt.datetime(2025, 8, 1),
            end=end,
            control_group="control",
            exposure="test_exposure",
            day_boundary=day_boundary,
            plan=AnalysisPlan(),
        )

    @staticmethod
    def _exposure_rows(con, name):
        """One unit exposed 2025-08-02 03:10 UTC (2025-08-01 22:10 at UTC-05:00)."""
        return con.create_table(
            name,
            obj=[
                {
                    "unit_id": "u_shift",
                    "ts": dt.datetime(2025, 8, 2, 3, 10, 0),
                    "event": "test_exposure",
                    "experiment_id": "exp_test",
                    "group_id": "treatment",
                }
            ],
        )

    def test_exposure_day_shifts_under_negative_offset(self, con):
        """Exposure 2025-08-02 03:10 UTC -> local day 2025-08-01.

        Both localized exposure-day sites agree: panel_spine's
        ``first_exposure_date`` and daily_exposure_counts' ``ds``.
        """
        rows = self._exposure_rows(con, "day_boundary_shift_exposures")
        exp = self._experiment()
        exposures = first_exposures(rows, exp)

        spine = panel_spine(exposures, exp, end_date=None).execute()
        assert spine["first_exposure_date"].dt.date.unique().tolist() == [dt.date(2025, 8, 1)]

        counts = daily_exposure_counts(exposures, exp).execute()
        assert counts["ds"].dt.date.tolist() == [dt.date(2025, 8, 1)]

    def test_window_membership_follows_the_shifted_day(self, con):
        """window_days=3 counts from the SHIFTED exposure day.

        Exposure 2025-08-02 03:10 UTC -> local day Aug 1 -> window
        [Aug 1, Aug 4) local. Hand-computed:
          - event 2025-08-04 02:00 UTC = local Aug 3 -> IN   (value 5.0)
          - event 2025-08-04 06:00 UTC = local Aug 4 -> OUT  (value 7.0)
          - event 2025-08-10 12:00 UTC = local Aug 10 -> OUT (value 11.0,
            present only so the unit's 3-day window is matured, never
            censored)
        y = 5.0. The naive UTC bucketing would put BOTH Aug-4-UTC events
        on day Aug 4 inside window [Aug 2, Aug 5) -> y = 12.0.
        """
        rows = self._exposure_rows(con, "day_boundary_window_exposures")
        events = con.create_table(
            "day_boundary_window_events",
            obj=[
                {
                    "unit_id": "u_shift",
                    "ts": dt.datetime(2025, 8, 4, 2, 0, 0),
                    "metric": "m",
                    "value": 5.0,
                },
                {
                    "unit_id": "u_shift",
                    "ts": dt.datetime(2025, 8, 4, 6, 0, 0),
                    "metric": "m",
                    "value": 7.0,
                },
                {
                    "unit_id": "u_shift",
                    "ts": dt.datetime(2025, 8, 10, 12, 0, 0),
                    "metric": "m",
                    "value": 11.0,
                },
            ],
        )
        exp = self._experiment()
        metric = MeanMetric(
            name="m", entity="unit_id", fact="purchase", aggregation="sum", window_days=3
        )

        exposures = first_exposures(rows, exp)
        spine, stats = unit_day_spine_stats(exposures, events, exp, metric.name)
        totals = unit_totals(spine, stats, metric, exp, warn_on_censoring=False).execute()

        assert totals["unit_id"].tolist() == ["u_shift"]  # matured, not censored
        assert totals["y"].tolist() == [5.0]

    def test_end_day_enrollment_respects_local_day(self, con):
        """Mirror of the whole-day-inclusive `end` test above: an exposure at
        2025-08-02 03:10 UTC with end=Aug 1 and day_boundary="UTC-05:00"
        STILL ENROLLS (locally it is Aug 1, the end day itself) - while the
        same rows under plain UTC are dropped (Aug 2 > Aug 1).
        """
        rows = self._exposure_rows(con, "day_boundary_end_day_exposures")

        enrolled = first_exposures(rows, self._experiment(end=dt.datetime(2025, 8, 1))).execute()
        assert sorted(enrolled["unit_id"]) == ["u_shift"]

        utc_exp = self._experiment(end=dt.datetime(2025, 8, 1), day_boundary="UTC")
        assert first_exposures(rows, utc_exp).execute().empty

    def test_spine_extent_and_event_days_localize_together(self, con):
        """The default ``end_date`` fallback localizes WITH ``event_ds``.

        A RUNNING experiment (no ``end``, so no observation_horizon) -
        the only case where the fallback is live. Exposure 2025-08-02
        03:10 UTC (local Aug 1). The latest event, 2025-08-04 02:00 UTC,
        straddles the boundary: local Aug 3. Both the spine's right edge
        and that event's day bucket must land on Aug 3 - an unswapped
        UTC fallback would extend the spine to Aug 4, one day past any
        bucket the localized stats can ever populate. (A non-straddling
        max event could never tell the two calendars apart, so the
        straddle is the whole test.)
        """
        rows = self._exposure_rows(con, "day_boundary_extent_exposures")
        events = con.create_table(
            "day_boundary_extent_events",
            obj=[
                {
                    "unit_id": "u_shift",
                    "ts": dt.datetime(2025, 8, 4, 2, 0, 0),
                    "metric": "m",
                    "value": 5.0,
                }
            ],
        )
        exp = Experiment(
            name="exp_test",
            unit="unit_id",
            start=dt.datetime(2025, 8, 1),
            control_group="control",
            exposure="test_exposure",
            day_boundary="UTC-05:00",
            plan=AnalysisPlan(),
        )
        exposures = first_exposures(rows, exp)
        spine, stats = unit_day_spine_stats(exposures, events, exp, "m")

        spine_days = spine.execute()["ds"].dt.date
        stats_days = stats.execute()["ds"].dt.date
        assert spine_days.min() == dt.date(2025, 8, 1)  # localized exposure day
        assert spine_days.max() == dt.date(2025, 8, 3)  # localized event max
        assert stats_days.max() == spine_days.max()

    # -- declared window edges are days at the boundary, whatever their spelling --

    _EST = dt.timezone(dt.timedelta(hours=-5))

    @staticmethod
    def _windowed(boundary: str, start: dt.datetime, end: dt.datetime | None) -> Experiment:
        return Experiment(
            name="exp_test",
            unit="unit_id",
            start=start,
            end=end,
            control_group="control",
            exposure="test_exposure",
            day_boundary=boundary,
            plan=AnalysisPlan(),
        )

    @staticmethod
    def _stamped(con, name, stamps):
        return con.create_table(
            name,
            obj=[
                {
                    "unit_id": unit,
                    "ts": ts,
                    "event": "test_exposure",
                    "experiment_id": "exp_test",
                    "group_id": "treatment" if i % 2 else "control",
                }
                for i, (unit, ts) in enumerate(stamps)
            ],
        )

    _SPELLINGS = {
        "zulu": (
            dt.datetime(2025, 1, 10, 0, tzinfo=dt.UTC),
            dt.datetime(2025, 1, 11, 0, tzinfo=dt.UTC),
        ),
        "offset": (
            dt.datetime(2025, 1, 9, 19, tzinfo=_EST),
            dt.datetime(2025, 1, 10, 19, tzinfo=_EST),
        ),
    }

    @pytest.mark.parametrize(
        ("boundary", "expected"),
        [("UTC", {"b", "c", "e"}), ("UTC-05:00", {"a", "b", "c", "e"})],
    )
    def test_equal_instants_enroll_the_same_units_and_spine(self, con, boundary, expected):
        """Two aware spellings of one instant declare one window at either boundary."""
        stamps = [
            ("d", dt.datetime(2025, 1, 8, 12)),
            ("a", dt.datetime(2025, 1, 9, 12)),
            ("b", dt.datetime(2025, 1, 10, 2)),
            ("c", dt.datetime(2025, 1, 10, 12)),
            ("e", dt.datetime(2025, 1, 11, 3)),
            ("f", dt.datetime(2025, 1, 12, 12)),
        ]
        rows = self._stamped(con, f"spelling_rows_{boundary[-2:]}", stamps)
        spines = {}
        for spelling, (start, end) in self._SPELLINGS.items():
            exp = self._windowed(boundary, start, end)
            exposures = first_exposures(rows, exp)
            enrolled = exposures.execute()
            assert set(enrolled["unit_id"]) == expected, spelling
            spine = panel_spine(exposures, exp, end_date=None).execute()
            spines[spelling] = spine.sort_values(["unit_id", "ds"]).reset_index(drop=True)
        assert spines["zulu"].equals(spines["offset"])

    @pytest.mark.parametrize("boundary", ["UTC", "UTC-05:00"])
    @pytest.mark.parametrize("spelling", ["zulu", "offset"])
    def test_an_exposure_at_the_declared_start_instant_is_enrolled(self, con, boundary, spelling):
        start, end = self._SPELLINGS[spelling]
        rows = self._stamped(
            con,
            f"start_instant_{boundary[-2:]}_{spelling}",
            [("at_start", dt.datetime(2025, 1, 10, 0))],
        )
        exp = self._windowed(boundary, start, end)
        assert first_exposures(rows, exp).execute()["unit_id"].tolist() == ["at_start"]

    def test_a_bare_date_declaration_is_the_local_day_at_the_boundary(self, con):
        """Naive 2025-01-15..2025-01-20 at UTC-05:00 opens and closes on those local days."""
        rows = self._stamped(
            con,
            "naive_window_rows",
            [
                ("before", dt.datetime(2025, 1, 15, 4, 59)),
                ("first", dt.datetime(2025, 1, 15, 5, 0)),
                ("last", dt.datetime(2025, 1, 21, 4, 59)),
                ("after", dt.datetime(2025, 1, 21, 5, 0)),
            ],
        )
        exp = self._windowed("UTC-05:00", dt.datetime(2025, 1, 15), dt.datetime(2025, 1, 20))
        assert set(first_exposures(rows, exp).execute()["unit_id"]) == {"first", "last"}


# Censoring-warning bookkeeping cost - one combined round-trip


def test_unit_totals_censor_warning_costs_one_round_trip(con, monkeypatch):
    """warn_on_censoring's enrolled/kept counts come back in ONE combined
    execute - not two separate panel-shaped nunique() queries per
    unit_totals call (which doubled per metric per breakout segment)."""
    import ibis.expr.types.core as _ibis_core

    metric = MeanMetric(
        name="rev_rt", entity="unit_id", fact="orders", aggregation="sum", window_days=3
    )
    spine, stats, exp = _spine_stats_fixture(
        con,
        unit_rows=[("u1", "treatment"), ("u2", "control")],
        exposure_first=dt.date(2025, 8, 1),
        days=10,
        stat_rows=[("u1", dt.date(2025, 8, 1), 1, 5.0)],
    )
    executed = []
    orig = _ibis_core.Expr.execute

    def counting(self, *args, **kwargs):
        executed.append(self)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(_ibis_core.Expr, "execute", counting)
    # Building the expression fires the warning bookkeeping; the returned
    # expression itself is NOT executed here.
    unit_totals(spine, stats, metric, exp, warn_on_censoring=True)
    assert len(executed) == 1


# resolved_measure_key - filter order canonicalized


def test_resolved_measure_key_is_filter_order_insensitive():
    """Filters apply conjunctively, so two refs listing identical filters
    in a different order are the same measure and must share one key
    (a differing key silently misses the stats-table dedup)."""
    from increment.query.builders import resolved_measure_key

    f_country = Filter(property="country", op="in", values=["US", "CA"])
    f_plan = Filter(property="plan", op="equals", values=["pro"])
    a = Measure(fact="purchase", aggregation="sum", filters=[f_country, f_plan])
    b = Measure(fact="purchase", aggregation="sum", filters=[f_plan, f_country])
    assert resolved_measure_key(a, value_column="amount") == resolved_measure_key(
        b, value_column="amount"
    )
    # Different filters still produce different keys.
    c = Measure(fact="purchase", aggregation="sum", filters=[f_country])
    assert resolved_measure_key(c, value_column="amount") != resolved_measure_key(
        a, value_column="amount"
    )


def test_filter_refuses_unknown_op():
    """An unknown filter op is rejected when the filter is defined, so it can
    never reach the SQL builder as a silently-dropped or misapplied filter."""
    with pytest.raises(ValidationError) as exc_info:
        Filter(property="amount", op="bogus_op", values=[1])  # ty: ignore[invalid-argument-type]
    assert [(error["loc"], error["type"]) for error in exc_info.value.errors()] == [
        (("op",), "literal_error")
    ]


# NULL-valued event rows - complete-case on value-based refs


def test_metric_events_drops_null_valued_rows_for_value_refs(con):
    """A NULL in the value column is not a measurement: the row counts
    neither as an occurrence (n_events) nor as a value (sum_value);
    otherwise it inflates count-style reads and biases sum/n_events
    averages downward."""
    from increment.query.builders import unit_day_stats

    rows = con.create_table(
        "null_value_purchase_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 10),
                "event": "purchase",
                "amount": 5.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 11),
                "event": "purchase",
                "amount": None,
            },
        ],
    )
    metric = MeanMetric(
        name="rev_null", entity="unit_id", fact="purchase", aggregation="sum", window_days=3
    )
    events = metric_events(rows, metric, value_column="amount")
    stats = unit_day_stats(events, source_key="s").execute()
    assert len(stats) == 1
    assert int(stats["n_events"].iloc[0]) == 1  # NULL row is no occurrence
    assert stats["sum_value"].iloc[0] == pytest.approx(5.0)


def test_metric_events_occurrence_refs_keep_null_valued_rows(con):
    """Occurrence-only refs never read the value column, so a NULL there
    is irrelevant and the row still counts as an occurrence."""
    rows = con.create_table(
        "null_value_occurrence_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 10),
                "event": "purchase",
                "amount": 5.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 11),
                "event": "purchase",
                "amount": None,
            },
        ],
    )
    conversion = ConversionMetric(
        name="converted_null", entity="unit_id", fact="purchase", window_days=3
    )
    got = metric_events(rows, conversion).execute()
    assert len(got) == 2
    assert set(got["value"]) == {1.0}


# site_volume — window-scoped, exposure-join-free metric volume


def test_site_volume_sums_all_events_including_non_exposed_units(con, experiment):
    """No exposure join: a unit that never appears in any exposure table
    still counts, and the total matches a hand-computed sum over every
    event in the window."""
    rows = con.create_table(
        "site_volume_all_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 5, 0),
                "event": "purchase",
                "amount": 49.99,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 3, 14, 0, 0),
                "event": "purchase",
                "amount": 7.50,
            },
            {
                "unit_id": "never_exposed",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "purchase",
                "amount": 10.00,
            },
        ],
    )
    metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", aggregation="sum")
    result = site_volume(rows, metric, experiment, value_column="amount").execute()
    assert set(result.columns) == SITE_VOLUME
    assert result["metric"].iloc[0] == "revenue"
    assert result["y"].iloc[0] == pytest.approx(49.99 + 7.50 + 10.00)
    assert np.isnan(result["y_den"].iloc[0])


def test_site_volume_ratio_numerator_and_denominator(con, experiment):
    """RatioMetric: numerator and denominator sums come from their own
    fact/filters, independently of each other."""
    rows = con.create_table(
        "site_volume_ratio_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u2",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "purchase",
                "amount": 20.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "session",
                "amount": None,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 2, 9, 0, 0),
                "event": "session",
                "amount": None,
            },
            {
                "unit_id": "u3",
                "ts": dt.datetime(2025, 8, 3, 9, 0, 0),
                "event": "session",
                "amount": None,
            },
        ],
    )
    metric = RatioMetric(
        name="rev_per_session",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum"),
        denominator=Measure(fact="session", aggregation="count"),
    )
    result = site_volume(rows, metric, experiment, value_column="amount").execute()
    assert result["y"].iloc[0] == pytest.approx(30.0)
    assert result["y_den"].iloc[0] == pytest.approx(3.0)


def test_site_volume_window_edge_end_inclusive_day_after_excluded(con, experiment):
    """An event on experiment.end's calendar day counts (whole-day
    inclusive, same convention as first_exposures's enrollment-close
    bound); the next calendar day does not."""
    rows = con.create_table(
        "site_volume_window_edge_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 6, 23, 0, 0),
                "event": "purchase",
                "amount": 100.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 7, 0, 0, 1),
                "event": "purchase",
                "amount": 999.0,
            },
        ],
    )
    metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", aggregation="sum")
    result = site_volume(rows, metric, experiment, value_column="amount").execute()
    assert result["y"].iloc[0] == pytest.approx(100.0)


def test_site_volume_refuses_retention_metric(page_view_events, experiment, retention_metric):
    """Retention has no site-wide reading: its outcome is anchored to each
    unit's OWN exposure time, not summable as a raw event stream."""
    with pytest.raises(CapabilityError) as exc_info:
        site_volume(page_view_events, retention_metric, experiment)
    assert exc_info.value.code == "query.builders.site_volume_metric_type"


def test_site_volume_refuses_quantile_metric(con, experiment):
    """A quantile is a distributional statistic, not additive across
    events - no site-wide reading, same as retention."""
    rows = con.create_table(
        "site_volume_quantile_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            }
        ],
    )
    metric = QuantileMetric(name="p50_revenue", entity="unit_id", fact="purchase", quantile=0.5)
    with pytest.raises(CapabilityError) as exc_info:
        site_volume(rows, metric, experiment, value_column="amount")
    assert exc_info.value.code == "query.builders.site_volume_metric_type"


def test_site_volume_window_edge_start_inclusive_day_before_excluded(con, experiment):
    """An event before experiment.start does not contribute - the start
    bound is unique to site_volume (first_exposures only bounds the end);
    unlike the end bound this direction was previously untested."""
    rows = con.create_table(
        "site_volume_start_edge_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 7, 31, 23, 59, 0),
                "event": "purchase",
                "amount": 999.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 0, 0, 1),
                "event": "purchase",
                "amount": 100.0,
            },
        ],
    )
    metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", aggregation="sum")
    result = site_volume(rows, metric, experiment, value_column="amount").execute()
    assert result["y"].iloc[0] == pytest.approx(100.0)


def test_site_volume_zero_matching_events_returns_typed_zero_not_null(con, experiment):
    """An empty matching window must return a typed 0.0, not SQL NULL --
    SUM over zero rows is NULL, and sitewide_evidence's bare float()
    crashes on that NULL."""
    rows = con.create_table(
        "site_volume_no_matching_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            },
        ],
    )
    metric = MeanMetric(name="revenue", entity="unit_id", fact="checkout", aggregation="sum")
    result = site_volume(rows, metric, experiment, value_column="amount").execute()
    assert result["y"].iloc[0] == 0.0
    assert not np.isnan(result["y"].iloc[0])


def test_site_volume_ratio_zero_matching_events_both_parts_typed_zero(con, experiment):
    """Both ratio parts share the same zero convention when neither has
    matching events -- the numerator and denominator windowed sums must
    not diverge on their empty-set convention."""
    rows = con.create_table(
        "site_volume_ratio_no_matching_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "other",
                "amount": 1.0,
            },
        ],
    )
    metric = RatioMetric(
        name="rev_per_session",
        entity="unit_id",
        numerator=Measure(fact="missing_num", aggregation="sum"),
        denominator=Measure(fact="missing_den", aggregation="sum"),
    )
    result = site_volume(rows, metric, experiment, value_column="amount").execute()
    assert result["y"].iloc[0] == 0.0
    assert result["y_den"].iloc[0] == 0.0


def test_site_volume_open_window_when_experiment_end_unset(con):
    """experiment.end=None (a running experiment) leaves the window open
    on the right - a late event still counts, unlike the fixed-end
    experiment fixture's window-edge test."""
    running_experiment = Experiment(
        name="exp_running",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=None,
        control_group="control",
        exposure="test_exposure",
        plan=AnalysisPlan(),
    )
    rows = con.create_table(
        "site_volume_open_window_events",
        obj=[
            {
                "unit_id": "u1",
                "ts": dt.datetime(2025, 8, 1, 9, 0, 0),
                "event": "purchase",
                "amount": 10.0,
            },
            {
                "unit_id": "u1",
                "ts": dt.datetime(2026, 1, 1, 0, 0, 0),
                "event": "purchase",
                "amount": 500.0,
            },
        ],
    )
    metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", aggregation="sum")
    result = site_volume(rows, metric, running_experiment, value_column="amount").execute()
    assert result["y"].iloc[0] == pytest.approx(510.0)


def _fixed_x_day_panel(con, *, include_x: bool = True):
    rows = []
    x_by_unit = {}
    for arm, offset in (("control", 0.0), ("treatment", 10.0)):
        for i in range(4):
            unit_id = f"{arm[:1]}{i}"
            x = float(i + (0.5 if arm == "treatment" else 0.0))
            x_by_unit[unit_id] = x
            for ds, y in (
                (dt.date(2025, 1, 1), offset + float(i)),
                (dt.date(2025, 1, 2), offset + float(2 * i + 1)),
            ):
                row = {
                    "unit_id": unit_id,
                    "ds": ds,
                    "experiment_id": "fixed_x",
                    "metric": "revenue",
                    "group_id": arm,
                    "n_events": 1,
                    "sum_value": y,
                    "min_value": y,
                    "max_value": y,
                    "first_exposure_date": dt.date(2025, 1, 1),
                }
                if include_x:
                    row["x"] = x
                rows.append(row)
    suffix = len(con.list_tables())
    return con.create_table(
        f"fixed_x_panel_{'x' if include_x else 'no_x'}_{suffix}",
        obj=rows,
    ), x_by_unit


def _centered_x_oracle(rows):
    x = np.asarray([row["x"] for row in rows], dtype=float)
    y = np.asarray([row["y"] for row in rows], dtype=float)
    ref_x = float(x.mean())
    ref_y = float(y.mean())
    dx = x - ref_x
    dy = y - ref_y
    return {
        "ref_x": ref_x,
        "cx1": float(dx.sum()),
        "cx2": float((dx * dx).sum()),
        "cxy": float((dx * dy).sum()),
    }


@pytest.mark.parametrize("builder", [daily_group_summary, asof_group_summary])
def test_day_axis_group_summaries_populate_fixed_x_moments(con, builder):
    panel, x_by_unit = _fixed_x_day_panel(con)
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="sum")
    if builder is asof_group_summary:
        summary = builder(panel, metric=metric)
        # The builder's cumulative y is reconstructed independently for the
        # oracle, preserving the same fixed x per unit across dates.
        source = panel
        rows = source.execute().to_dict("records")
        grouped = {}
        for row in rows:
            key = (row["ds"], row["group_id"])
            grouped.setdefault(key, []).append(
                {
                    "x": x_by_unit[row["unit_id"]],
                    "y": sum(
                        r["sum_value"]
                        for r in rows
                        if r["unit_id"] == row["unit_id"] and r["ds"] <= row["ds"]
                    ),
                }
            )
    else:
        summary = builder(panel, metric=metric)
        rows = panel.execute().to_dict("records")
        grouped = {}
        for row in rows:
            grouped.setdefault((row["ds"], row["group_id"]), []).append(
                {"x": x_by_unit[row["unit_id"]], "y": row["sum_value"]}
            )

    actual = summary.execute().to_dict("records")
    for row in actual:
        key = (row["ds"], row["group_id"])
        expected = _centered_x_oracle(grouped[key])
        for field, value in expected.items():
            assert row[field] == pytest.approx(value, abs=1e-12), (builder.__name__, key, field)
        # A materialised covariate declares its role, as the unit-grain summary does.
        assert row["x_role"] == "covariate", (builder.__name__, key)


@pytest.mark.parametrize("builder", [daily_group_summary, asof_group_summary])
def test_day_axis_group_summaries_keep_x_family_null_without_covariate(con, builder):
    panel, _ = _fixed_x_day_panel(con, include_x=False)
    metric = MeanMetric(name="revenue", entity="user_id", fact="revenue", aggregation="sum")
    summary = builder(panel, metric=metric)
    rows = summary.execute().to_dict("records")
    assert rows
    # No covariate: the role is declared null, never inferred.
    assert all(row["x_role"] is None or row["x_role"] != row["x_role"] for row in rows)
    assert all(
        row[field] is None or np.isnan(row[field])
        for row in rows
        for field in ("ref_x", "cx1", "cx2", "cxy")
    )


def test_panel_joins_do_not_fan_out_across_experiments(con):
    """A panel row is (experiment_id, unit_id, ds), matching the scoping the
    running sums already partition by. Joining on (unit_id, ds) alone joined
    each experiment's row to the other's, doubling the rows and mixing one
    experiment's denominator into the other's outcome."""
    rows = [
        {
            "unit_id": "u1",
            "experiment_id": "expA",
            "group_id": "treatment",
            "metric": "rev_per_click",
            "ds": dt.date(2025, 1, 1),
            "n_events": 1,
            "sum_value": 10.0,
            "min_value": 10.0,
            "max_value": 10.0,
            "first_exposure_date": dt.date(2025, 1, 1),
        },
        {
            "unit_id": "u1",
            "experiment_id": "expB",
            "group_id": "treatment",
            "metric": "rev_per_click",
            "ds": dt.date(2025, 1, 1),
            "n_events": 1,
            "sum_value": 99.0,
            "min_value": 99.0,
            "max_value": 99.0,
            "first_exposure_date": dt.date(2025, 1, 1),
        },
    ]
    metric = RatioMetric(
        name="rev_per_click",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum"),
        denominator=Measure(fact="click", aggregation="sum"),
    )
    summary = asof_group_summary(
        ibis.memtable(rows), metric, den_panel=ibis.memtable(rows)
    ).execute()
    # One row per (experiment, day): a cross-experiment join would double each.
    assert sorted(summary["n"].tolist()) == [1, 1]
    out = {r["experiment_id"]: r for r in summary.to_dict("records")}
    assert out["expA"]["ref_y"] == 10.0
    assert out["expA"]["ref_den"] == 10.0
    assert out["expB"]["ref_y"] == 99.0
    assert out["expB"]["ref_den"] == 99.0


def test_panel_row_key_falls_back_when_a_side_lacks_experiment_id(con):
    """A bare projection without the column still joins on (unit_id, ds)."""
    from increment.query.builders import _panel_row_key

    left = ibis.memtable(
        [{"unit_id": "u1", "experiment_id": "expA", "ds": dt.date(2025, 1, 1), "value": 1.0}]
    )
    right = ibis.memtable([{"unit_id": "u1", "ds": dt.date(2025, 1, 1), "d": 1.0}])
    joined = left.left_join(right, _panel_row_key(left, right))
    assert len(joined.execute()) == 1


def test_panel_row_key_is_null_safe_but_not_a_wildcard(con):
    """first_exposures coalesces experiment_id to the experiment's own name, so
    a spine-built panel always carries it. Two unknown values still match each
    other, but an unknown must NOT match every named experiment: one such row
    would then duplicate against each of them, which is the fan-out the key
    exists to prevent."""
    from increment.query.builders import _asof_masked_value, _panel_row_key

    def _panel(experiment_id, value):
        return _asof_masked_value(
            ibis.memtable(
                [
                    {
                        "unit_id": "u1",
                        "experiment_id": experiment_id,
                        "ds": dt.date(2025, 1, 1),
                        "n_events": 1,
                        "sum_value": value,
                        "min_value": value,
                        "max_value": value,
                        "first_exposure_date": dt.date(2025, 1, 1),
                    }
                ]
            ),
            None,
            "sum",
        )

    # Unknown on both sides: one row, matched.
    num, den = _panel(None, 10.0), _panel(None, 2.0)
    matched = num.left_join(den, _panel_row_key(num, den), rname="{name}_den").execute()
    assert len(matched) == 1
    assert matched["asof_value_den"].tolist() == [2.0]

    # Unknown against a named experiment: no match, and crucially no duplication
    # when the named side carries several experiments.
    num = _panel(None, 10.0)
    den = _asof_masked_value(
        ibis.memtable(
            [
                {
                    "unit_id": "u1",
                    "experiment_id": name,
                    "ds": dt.date(2025, 1, 1),
                    "n_events": 1,
                    "sum_value": value,
                    "min_value": value,
                    "max_value": value,
                    "first_exposure_date": dt.date(2025, 1, 1),
                }
                for name, value in (("expA", 2.0), ("expB", 5.0))
            ]
        ),
        None,
        "sum",
    )
    unmatched = num.left_join(den, _panel_row_key(num, den), rname="{name}_den").execute()
    assert len(unmatched) == 1


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        ("sum", {"u1": 10.0, "u2": 20.0}),
        ("avg_calendar_day", {"u1": 5.0, "u2": 10.0}),
    ],
)
def test_panel_spine_keeps_units_enrolled_after_the_event_horizon(aggregation, expected):
    """A running experiment whose fact table lags enrollment must not drop the
    units enrolled since the last loaded event: an empty per-unit day range
    unnests to nothing and deletes the unit with no censoring warning."""
    con = ibis.duckdb.connect()
    exposures_tbl = con.create_table(
        "late_enrollee_exposures",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "c",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u3",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 5, 9),
            },
            {
                "unit_id": "u4",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 5, 10),
            },
            {
                "unit_id": "u5",
                "experiment_id": "e",
                "group_id": "c",
                "ts": dt.datetime(2025, 8, 5, 9),
            },
        ],
    )
    facts = con.create_table(
        "late_enrollee_facts",
        obj=[
            {
                "unit_id": "u1",
                "event": "purchase",
                "amount": 10.0,
                "ts": dt.datetime(2025, 8, 2, 10),
            },
            {
                "unit_id": "u2",
                "event": "purchase",
                "amount": 20.0,
                "ts": dt.datetime(2025, 8, 2, 10),
            },
        ],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        control_group="c",  # still running: no end, no observation horizon
        exposure="late_enrollee_exposures",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(name="rev", entity="unit_id", fact="purchase", aggregation=aggregation)

    exposures = first_exposures(exposures_tbl, experiment)
    events = metric_events(facts, metric, "amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    spine_rows = con.to_pyarrow(spine).to_pylist()
    assert {row["unit_id"] for row in spine_rows} == {"u1", "u2", "u3", "u4", "u5"}
    assert {row["ds"] for row in spine_rows if row["ds"] is not None} == {
        dt.date(2025, 8, 1),
        dt.date(2025, 8, 2),
    }
    assert {row["unit_id"] for row in spine_rows if row["ds"] is None} == {"u3", "u4", "u5"}
    with pytest.warns(UserWarning):
        totals = unit_totals(spine, stats, metric, experiment)
    observed = con.to_pyarrow(totals).to_pylist()
    assert {row["unit_id"]: row["y"] for row in observed} == pytest.approx(expected)
    per_arm = {
        row["group_id"]: row["n"] for row in con.to_pyarrow(group_summary(totals)).to_pylist()
    }
    assert per_arm == {"t": 1, "c": 1}


def test_panel_spine_no_declared_horizon_and_no_events_keeps_identities_undated():
    """With no events and no declared horizon, every enrolled identity
    survives as a null-date row -- there is no observed calendar day to
    date any of them by, so nothing is analyzable yet."""
    con = ibis.duckdb.connect()
    exposures_tbl = con.create_table(
        "no_event_exposures",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "c",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u3",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 5, 9),
            },
            {
                "unit_id": "u4",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 5, 10),
            },
            {
                "unit_id": "u5",
                "experiment_id": "e",
                "group_id": "c",
                "ts": dt.datetime(2025, 8, 5, 9),
            },
        ],
    )
    facts = con.create_table(
        "no_event_facts",
        # A non-matching row keeps the schema real while metric_events
        # still yields zero rows for "purchase" -- ibis/duckdb reject a
        # truly columnless empty table.
        obj=[{"unit_id": "nobody", "event": "other", "amount": 1.0, "ts": dt.datetime(2025, 8, 1)}],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        control_group="c",
        exposure="no_event_exposures",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(name="rev", entity="unit_id", fact="purchase", aggregation="sum")

    exposures = first_exposures(exposures_tbl, experiment)
    events = metric_events(facts, metric, "amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    spine_rows = con.to_pyarrow(spine).to_pylist()
    assert {row["unit_id"] for row in spine_rows} == {"u1", "u2", "u3", "u4", "u5"}
    assert all(row["ds"] is None for row in spine_rows)

    panel = unit_day_panel(exposures, events, experiment, metric.name)
    assert con.to_pyarrow(panel).to_pylist() == []

    totals = unit_totals(spine, stats, metric, experiment, warn_on_censoring=False)
    assert con.to_pyarrow(totals).to_pylist() == []


def test_panel_spine_declared_horizon_keeps_existing_zero_event_maturity():
    """A declared finite observation horizon is unaffected by the null-date
    spine change: with no events, every unit still matures with y=0 once
    its window closes before the horizon."""
    con = ibis.duckdb.connect()
    exposures_tbl = con.create_table(
        "declared_horizon_exposures",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
        ],
    )
    facts = con.create_table(
        "declared_horizon_facts",
        obj=[{"unit_id": "nobody", "event": "other", "amount": 1.0, "ts": dt.datetime(2025, 8, 1)}],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 10),
        control_group="c",
        exposure="declared_horizon_exposures",
        plan=AnalysisPlan(),
    )
    metric = ConversionMetric(name="conv", entity="unit_id", fact="purchase", window_days=3)

    exposures = first_exposures(exposures_tbl, experiment)
    events = metric_events(facts, metric, "amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    totals = con.to_pyarrow(unit_totals(spine, stats, metric, experiment)).to_pylist()
    assert {row["unit_id"]: row["y"] for row in totals} == {"u1": 0.0}


def _asof_panel(con, name, values):
    return con.create_table(
        name,
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "metric": "r",
                "ds": dt.date(2025, 8, day),
                "n_events": 1 if value else 0,
                "sum_value": float(value),
                "min_value": float(value),
                "max_value": float(value),
                "first_exposure_ts": dt.datetime(2025, 8, 1, 0, 0),
                "first_exposure_date": dt.date(2025, 8, 1),
            }
            for day, value in zip((1, 2, 3, 4), values, strict=True)
        ],
    )


def test_asof_ratio_den_survives_a_window_bounded_den_panel():
    """A row-truncated denominator panel must not NULL the den family: the
    as-of builder masks per unit itself, so the den is realigned onto the
    numerator's day axis and zero-filled before masking."""
    con = ibis.duckdb.connect()
    num = _asof_panel(con, "asof_ratio_num", (10, 10, 0, 0))
    den = _asof_panel(con, "asof_ratio_den", (2, 2, 0, 0))
    metric = RatioMetric(
        name="r",
        entity="unit_id",
        numerator=Measure(fact="purchase", aggregation="sum", window_days=2),
        denominator=Measure(fact="session", aggregation="sum", window_days=2),
    )

    frame = asof_group_summary(num, metric, den_panel=window_bound_stats(den, metric)).execute()
    frame["ds"] = frame["ds"].astype("datetime64[ns]").dt.date
    rows = {row["ds"]: row for row in frame.to_dict("records")}

    for day in (3, 4):  # window closed on Aug 3; den frozen at its total
        row = rows[dt.date(2025, 8, day)]
        assert row["n"] == 1
        assert not math.isnan(row["ref_den"]), f"den family is NULL on Aug {day}"
        sum_den = row["n"] * row["ref_den"] + row["cden1"]
        sum_y = row["n"] * row["ref_y"] + row["cy1"]
        assert sum_den == 4.0
        assert sum_y / sum_den == 5.0


def test_first_exposures_carries_the_first_exposures_cluster_label():
    """The carried cluster label is the one the unit was FIRST exposed in, not
    the lexicographic minimum over all of its exposure rows."""
    con = ibis.duckdb.connect()
    exposure_events = con.create_table(
        "cluster_label_conflict",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "treatment",
                "store_id": "s9",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "treatment",
                "store_id": "s1",
                "ts": dt.datetime(2025, 8, 1, 10),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "control",
                "store_id": "s2",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
        ],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 2),
        control_group="control",
        cluster="store_id",
        exposure="cluster_label_conflict",
        plan=AnalysisPlan(),
    )
    rows = {
        r["unit_id"]: r
        for r in first_exposures(exposure_events, experiment).execute().to_dict("records")
    }
    assert rows["u1"]["first_exposure_ts"] == dt.datetime(2025, 8, 1, 9)
    assert rows["u1"]["store_id"] == "s9"
    assert rows["u2"]["store_id"] == "s2"


def test_conversion_daily_series_is_binary_not_an_event_count():
    """A unit converting twice in one day contributes 1.0 to the daily
    series, the same any-occurrence estimand run() reports."""
    con = ibis.duckdb.connect()
    exposure_events = con.create_table(
        "conv_daily_exposures",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "treatment",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "control",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
        ],
    )
    facts = con.create_table(
        "conv_daily_facts",
        obj=[
            {
                "unit_id": "u1",
                "event": "purchase",
                "amount": 5.0,
                "ts": dt.datetime(2025, 8, 1, 10),
            },
            {
                "unit_id": "u1",
                "event": "purchase",
                "amount": 5.0,
                "ts": dt.datetime(2025, 8, 1, 11),
            },
            {
                "unit_id": "u2",
                "event": "purchase",
                "amount": 5.0,
                "ts": dt.datetime(2025, 8, 1, 10),
            },
        ],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        end=dt.datetime(2025, 8, 2),
        control_group="control",
        exposure="conv_daily_exposures",
        plan=AnalysisPlan(),
    )
    metric = ConversionMetric(name="conv", entity="unit_id", fact="purchase", window_days=1)

    exposures = first_exposures(exposure_events, experiment)
    events = metric_events(facts, metric, "amount")
    panel = unit_day_panel(exposures, events, experiment, metric.name)
    daily = daily_group_summary(window_bound_stats(panel, metric), metric=metric).execute()

    per_arm = {row["group_id"]: row for row in daily.to_dict("records")}
    assert per_arm["treatment"]["ref_y"] == 1.0
    assert per_arm["control"]["ref_y"] == 1.0
    assert per_arm["treatment"]["cy2"] == 0.0


def test_asof_completed_windows_only_gates_a_windowed_mean_without_uptake():
    """completed_windows_only=True admits a unit only once its own outcome
    window has closed, uptake or no uptake. c1/t1 (exposed Jan 1, 3-day
    window) finalize on Jan 4; c2/t2 (exposed Jan 3) never do, so the
    decision series is one date with one unit per arm and cumulative
    outcome 3 -- the same Jan-4 boundary Task 6 enforces on the frame."""
    con = ibis.duckdb.connect()
    group = {"c1": "control", "c2": "control", "t1": "treatment", "t2": "treatment"}
    exposure = {
        "c1": dt.date(2026, 1, 1),
        "c2": dt.date(2026, 1, 3),
        "t1": dt.date(2026, 1, 1),
        "t2": dt.date(2026, 1, 3),
    }
    days = [dt.date(2026, 1, d) for d in (1, 2, 3, 4)]
    rows = [
        {
            "unit_id": unit,
            "experiment_id": "e",
            "group_id": group[unit],
            "metric": "revenue",
            "ds": day,
            "n_events": 1,
            "sum_value": 1.0,
            "min_value": 1.0,
            "max_value": 1.0,
            "first_exposure_ts": dt.datetime.combine(exposure[unit], dt.time()),
            "first_exposure_date": exposure[unit],
        }
        for unit in group
        for day in days
        if day >= exposure[unit]
    ]
    panel = con.create_table("completion_panel", obj=rows)
    metric = MeanMetric(name="revenue", entity="unit_id", fact="purchase", window_days=3)

    provisional = con.to_pyarrow(asof_group_summary(panel, metric)).to_pylist()
    decision = con.to_pyarrow(
        asof_group_summary(panel, metric, completed_windows_only=True)
    ).to_pylist()

    assert {row["ds"] for row in provisional} == set(days)
    assert {row["ds"] for row in decision} == {dt.date(2026, 1, 4)}
    assert sorted(row["n"] for row in decision) == [1, 1]
    assert {row["group_id"]: row["ref_y"] for row in decision} == {"control": 3.0, "treatment": 3.0}


def test_panel_spine_bounded_window_still_cannot_mature_the_aug5_enrollment():
    """A fixed-window metric's maturity gate is unaffected by the spine
    change: the Aug 5 enrollment is still too recent to mature against an
    event horizon that only reaches Aug 2, while the Aug 1 enrollment's
    2-day window (open through Aug 2) captures its own purchase and
    matures with y = 1.0."""
    con = ibis.duckdb.connect()
    exposures_tbl = con.create_table(
        "bounded_late_enrollee_exposures",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "c",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u3",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 5, 9),
            },
        ],
    )
    facts = con.create_table(
        "bounded_late_enrollee_facts",
        obj=[
            {
                "unit_id": "u1",
                "event": "purchase",
                "amount": 10.0,
                "ts": dt.datetime(2025, 8, 2, 10),
            },
            {
                "unit_id": "u2",
                "event": "purchase",
                "amount": 20.0,
                "ts": dt.datetime(2025, 8, 2, 10),
            },
        ],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        control_group="c",
        exposure="bounded_late_enrollee_exposures",
        plan=AnalysisPlan(),
    )
    metric = ConversionMetric(name="conv", entity="unit_id", fact="purchase", window_days=2)

    exposures = first_exposures(exposures_tbl, experiment)
    events = metric_events(facts, metric, "amount")
    spine, stats = unit_day_spine_stats(exposures, events, experiment, metric.name)
    totals = con.to_pyarrow(unit_totals(spine, stats, metric, experiment, warn_on_censoring=False))
    assert {row["unit_id"]: row["y"] for row in totals.to_pylist()} == {"u1": 1.0, "u2": 1.0}


def test_unit_day_panel_dates_end_at_the_event_horizon_for_late_enrollees():
    """The day axis for a running, undeclared-horizon experiment ends at
    the last loaded event date -- late enrollees with no observed day
    contribute no day-axis rows at all, dated or otherwise."""
    con = ibis.duckdb.connect()
    exposures_tbl = con.create_table(
        "day_axis_late_enrollee_exposures",
        obj=[
            {
                "unit_id": "u1",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u2",
                "experiment_id": "e",
                "group_id": "c",
                "ts": dt.datetime(2025, 8, 1, 9),
            },
            {
                "unit_id": "u3",
                "experiment_id": "e",
                "group_id": "t",
                "ts": dt.datetime(2025, 8, 5, 9),
            },
        ],
    )
    facts = con.create_table(
        "day_axis_late_enrollee_facts",
        obj=[
            {
                "unit_id": "u1",
                "event": "purchase",
                "amount": 10.0,
                "ts": dt.datetime(2025, 8, 2, 10),
            },
            {
                "unit_id": "u2",
                "event": "purchase",
                "amount": 20.0,
                "ts": dt.datetime(2025, 8, 2, 10),
            },
        ],
    )
    experiment = Experiment(
        name="e",
        unit="unit_id",
        start=dt.datetime(2025, 8, 1),
        control_group="c",
        exposure="day_axis_late_enrollee_exposures",
        plan=AnalysisPlan(),
    )
    metric = MeanMetric(name="rev", entity="unit_id", fact="purchase", aggregation="sum")

    exposures = first_exposures(exposures_tbl, experiment)
    events = metric_events(facts, metric, "amount")
    panel_expr = unit_day_panel(exposures, events, experiment, metric.name)
    panel = con.to_pyarrow(panel_expr).to_pylist()
    assert {row["unit_id"] for row in panel} == {"u1", "u2"}
    assert max(row["ds"] for row in panel) == dt.date(2025, 8, 2)

    daily = con.to_pyarrow(daily_group_summary(panel_expr, metric=metric)).to_pylist()
    assert {row["ds"] for row in daily} == {dt.date(2025, 8, 1), dt.date(2025, 8, 2)}


def _role_memtable(role: str, rows: list[dict]):
    """Build a properly-typed `ir.Table` for an ``exposures``/``measure_stats``
    stub `verify_relation` -- `_ensure` now consumes the verified relation
    directly as a warehouse-backed table, never a cached Python row list."""
    from increment.query.schemas import UNIT_DAY_ARTIFACT_RELATION_SCHEMAS

    type_map = {
        "STRING": "string",
        "INT64": "int64",
        "FLOAT64": "float64",
        "DATE": "date",
        "TIMESTAMP_UTC_US": "timestamp('UTC')",
        "BOOLEAN": "boolean",
    }
    schema = {
        field: type_map[type_tag]
        for field, type_tag, _nullable in UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role]
    }
    if rows:
        return ibis.memtable(rows, schema=schema)
    return ibis.memtable(ibis.schema(schema).to_pyarrow().empty_table(), schema=schema)


def _artifact_reader_source(exposures, stats, *, last_ds, freshness_loaded_through):
    """Minimal ArtifactMomentSource over hand-built exposure/measure_stats
    rows, mirroring `test_artifact_reducer_uses_one_snapshot_handle_and_caches_relations`
    in `tests/test_unit_day_artifact_adoption.py`. `last_ds` stands in for
    whatever native_source.py's manifest assembly computed -- callers pass
    the REAL `max(outcome ds, exposure first_exposure_date)` value to
    reproduce a round-tripped artifact honestly. `freshness_loaded_through`
    overrides the single "orders" measure's persisted watermark, the
    genuine per-fact freshness signal native_source.py sets independently
    of `last_ds`."""
    from contextlib import contextmanager
    from typing import cast

    from increment.frame import MetricSpec
    from increment.query.artifact_contract import ArtifactStore
    from increment.query.artifact_digest import digest_relation, manifest_sha256
    from increment.query.artifact_reader import ArtifactMomentSource
    from increment.query.schemas import (
        UNIT_DAY_ARTIFACT_PRIMARY_KEYS,
        UNIT_DAY_ARTIFACT_RELATION_SCHEMAS,
    )
    from increment.semantics.artifact import RelationLocator, UnitDayArtifactRef
    from tests.semantics.test_unit_day_artifact import _manifest

    manifest = _manifest()
    refs = {}
    for role, rows in (("exposures", exposures), ("measure_stats", stats)):
        digest = digest_relation(
            role,
            UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role],
            rows,
            primary_key=UNIT_DAY_ARTIFACT_PRIMARY_KEYS[role],
        )
        refs[role] = getattr(manifest.base, role).model_copy(
            update={
                "schema_sha256": digest.schema_sha256,
                "content_sha256": digest.content_sha256,
                "row_count": len(rows),
            }
        )
    measures = tuple(
        measure.model_copy(
            update={
                "freshness": measure.freshness.model_copy(
                    update={"loaded_through": freshness_loaded_through}
                )
            }
        )
        for measure in manifest.measures
    )
    manifest = manifest.model_copy(
        update={
            "base": manifest.base.model_copy(update=refs),
            "last_ds": last_ds,
            "measures": measures,
        }
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest_sha256(manifest)})
    ref = UnitDayArtifactRef(
        artifact_id=manifest.artifact_id,
        generation_id=manifest.generation_id,
        manifest=RelationLocator(name="manifest"),
        manifest_sha256=manifest.manifest_sha256,
    )

    class Snapshot:
        artifact_id = manifest.artifact_id
        generation_id = manifest.generation_id

        def read_manifest(self, locator, *, expected_sha256):
            return manifest

        def verify_relation(self, relation, *, expected_role):
            rows = exposures if expected_role == "exposures" else stats
            return _role_memtable(expected_role, rows)

        def execute(self, expression):
            return expression.execute()

    snapshot = Snapshot()

    class Store:
        def validate_locator(self, locator, **kwargs):
            return None

        @contextmanager
        def open_snapshot(self, fixed_ref):
            yield snapshot

    return ArtifactMomentSource.open(
        cast(ArtifactStore, Store()),
        ref,
        expected_context=manifest.context,
        metrics=(MetricSpec(name="conversion", type="mean", value_column="orders"),),
    )


def test_artifact_round_trip_never_dates_a_late_enrollee_by_its_own_enrollment():
    """`manifest.last_ds` is `max(outcome ds, exposure first_exposure_date)`
    (native_source.py's manifest assembly): a late enrollee's own
    enrollment date, not any outcome evidence, can set that edge. The
    measure's own persisted freshness watermark (Jan 2, the true last
    outcome day) must win instead -- the late enrollee stays off the
    spine as a null-date, censored identity (D1), both at the total grain
    and on the day axis."""
    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u_late",
            "group_id": "treatment",
            "first_exposure_ts": dt.datetime(2025, 1, 10, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 10),
        },
    ]
    stats = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 2),
            "measure_key": "orders",
            "n_events": 1,
            "sum_value": 2.0,
            "min_value": 2.0,
            "max_value": 2.0,
        },
    ]
    # u_late's own enrollment (Jan 10) is what actually set manifest.last_ds
    # in a real round trip; the measure's own freshness watermark (Jan 2)
    # is the true last outcome day and must govern instead.
    source = _artifact_reader_source(
        exposures,
        stats,
        last_ds=dt.date(2025, 1, 10),
        freshness_loaded_through=dt.date(2025, 1, 2),
    )
    metric = source.context.metrics[0]

    totals = source.moments(metric)
    assert {row["group_id"]: row["n"] for row in totals} == {"control": 1}, (
        "u_late (treatment) must not surface as a matured row"
    )

    daily = source.moments(metric, grain="daily")
    daily_dates = {row["ds"].date() if hasattr(row["ds"], "date") else row["ds"] for row in daily}
    assert all(d <= dt.date(2025, 1, 2) for d in daily_dates), (
        "the day axis must never extend past the last real outcome date"
    )
    source.close()


def test_artifact_round_trip_with_a_stale_freshness_watermark_censors_everyone():
    """Zero measure_stats rows and a freshness watermark BEFORE either
    unit's own enrollment date: outcome data genuinely has not loaded far
    enough yet, decoupled from `manifest.last_ds` (which is polluted with
    both units' own enrollment dates and would otherwise, wrongly, make
    them look mature)."""
    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 2),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "group_id": "treatment",
            "first_exposure_ts": dt.datetime(2025, 1, 3, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 3),
        },
    ]
    source = _artifact_reader_source(
        exposures,
        [],
        last_ds=dt.date(2025, 1, 3),
        freshness_loaded_through=dt.date(2025, 1, 1),
    )
    metric = source.context.metrics[0]

    totals = source.moments(metric)
    assert totals == []

    daily = source.moments(metric, grain="daily")
    assert daily == []
    source.close()


def _typed_metric_definition(partial):
    """Complete a test's semantic overrides into a valid typed metric declaration.

    The reader refuses a context whose declarations do not validate as typed metrics, so the
    hand-built fixture supplies the required identity fields and lets the caller override
    only the semantics under test (aggregation, windows, thresholds)."""
    definition = {"entity": "unit_id", "type": "mean", **partial}
    if definition["type"] == "ratio":
        definition["numerator"] = {"fact": "numerator_fact", **partial.get("numerator", {})}
        definition["denominator"] = {"fact": "denominator_fact", **partial.get("denominator", {})}
    else:
        definition.setdefault("fact", "orders")
    return definition


def _custom_artifact_reader_source(
    exposures,
    stats,
    *,
    last_ds,
    measures,
    metric_bindings,
    metric_specs,
    definitions_metrics,
    declared_end=None,
):
    """General artifact-reader test fixture: like `_artifact_reader_source`
    but lets a caller declare its own metric binding(s)/measure(s) (for a
    RetentionMetric, or a metric sharing the artifact with a fresher
    sibling measure) and an explicit declared observation end
    (`context.experiment.end`)."""
    from contextlib import contextmanager
    from typing import cast

    from increment.query.artifact_contract import ArtifactStore
    from increment.query.artifact_digest import (
        canonical_json,
        context_sha256,
        digest_relation,
        manifest_sha256,
    )
    from increment.query.artifact_reader import ArtifactMomentSource
    from increment.query.schemas import (
        UNIT_DAY_ARTIFACT_PRIMARY_KEYS,
        UNIT_DAY_ARTIFACT_RELATION_SCHEMAS,
    )
    from increment.semantics.artifact import ArtifactContext, RelationLocator, UnitDayArtifactRef
    from tests.semantics.test_unit_day_artifact import _manifest

    manifest = _manifest()
    refs = {}
    for role, rows in (("exposures", exposures), ("measure_stats", stats)):
        digest = digest_relation(
            role,
            UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[role],
            rows,
            primary_key=UNIT_DAY_ARTIFACT_PRIMARY_KEYS[role],
        )
        refs[role] = getattr(manifest.base, role).model_copy(
            update={
                "schema_sha256": digest.schema_sha256,
                "content_sha256": digest.content_sha256,
                "row_count": len(rows),
            }
        )
    from increment.semantics.models import AnalysisPlan, Experiment

    typed_definitions = [_typed_metric_definition(item) for item in definitions_metrics]
    names = [item["name"] for item in typed_definitions]
    # A real artifact context always carries a typed experiment whose metric roster
    # equals the declared metric definitions.
    experiment = Experiment(
        name="exp",
        exposure="assigned",
        unit="unit_id",
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        control_group="control",
        plan=AnalysisPlan(primary=names[0], secondaries=names[1:] or None),
    )
    experiment_payload = experiment.model_dump(mode="json")
    if declared_end is not None:
        experiment_payload["end"] = dt.datetime.combine(declared_end, dt.time(), dt.UTC).isoformat()
    context_payload: dict[str, object] = {
        "context_format": 2,
        "definitions": {"metrics": typed_definitions},
        "experiment": experiment_payload,
        "experiment_name": "exp",
        "window_days": {
            "start": "2025-01-01",
            "end": None if declared_end is None else declared_end.isoformat(),
            "observation_horizon": None if declared_end is None else declared_end.isoformat(),
        },
    }
    context_json = canonical_json(context_payload)
    manifest = manifest.model_copy(
        update={
            "base": manifest.base.model_copy(update=refs),
            "last_ds": last_ds,
            "measures": tuple(measures),
            "metric_measures": tuple(metric_bindings),
            "context": ArtifactContext(
                canonical_json=context_json,
                sha256=context_sha256({"canonical_json": context_json}),
            ),
        }
    )
    manifest = manifest.model_copy(update={"manifest_sha256": manifest_sha256(manifest)})
    ref = UnitDayArtifactRef(
        artifact_id=manifest.artifact_id,
        generation_id=manifest.generation_id,
        manifest=RelationLocator(name="manifest"),
        manifest_sha256=manifest.manifest_sha256,
    )

    class Snapshot:
        artifact_id = manifest.artifact_id
        generation_id = manifest.generation_id

        def read_manifest(self, locator, *, expected_sha256):
            return manifest

        def verify_relation(self, relation, *, expected_role):
            rows = exposures if expected_role == "exposures" else stats
            return _role_memtable(expected_role, rows)

        def execute(self, expression):
            return expression.to_pyarrow()

    snapshot = Snapshot()

    class Store:
        def validate_locator(self, locator, **kwargs):
            return None

        @contextmanager
        def open_snapshot(self, fixed_ref):
            yield snapshot

    return ArtifactMomentSource.open(
        cast(ArtifactStore, Store()),
        ref,
        expected_context=manifest.context,
        metrics=tuple(metric_specs),
    )


def test_artifact_reader_retention_cohort_gates_on_its_own_watermark_not_a_declared_end():
    """A declared observation end (Jan 31) legitimately sizes the spine
    for a mid-flight artifact, but a retention cohort must still gate
    maturity on the metric's OWN outcome watermark (Jan 10) -- not the
    declared end -- or dates no outcome evidence covers yet (Jan 11-31)
    wrongly mature with a phantom zero (D1)."""
    from increment.frame import MetricSpec
    from increment.semantics.artifact import Freshness, MeasureManifest, SimpleMetricMeasure

    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "group_id": "treatment",
            "first_exposure_ts": dt.datetime(2025, 1, 20, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 20),
        },
    ]
    measures = (
        MeasureManifest(
            measure_key="return",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 10), declared_complete=False),
        ),
    )
    source = _custom_artifact_reader_source(
        exposures,
        [],
        last_ds=dt.date(2025, 1, 20),
        measures=measures,
        metric_bindings=(SimpleMetricMeasure(metric_name="retained", measure_key="return"),),
        metric_specs=(
            MetricSpec(
                name="retained", type="retention", value_column="return", threshold_days=(0, 5)
            ),
        ),
        definitions_metrics=[{"name": "retained", "type": "retention", "threshold_days": [0, 5]}],
        declared_end=dt.date(2025, 1, 31),
    )
    metric = source.context.metrics[0]

    # u1's band [Jan1, Jan6) closes well within the watermark (Jan10) and
    # matures with y=0 (no return observed); u2's band [Jan20, Jan25) is
    # entirely beyond the watermark and must not appear at all.
    with pytest.warns(IncrementWarning) as rec:
        cohorts = source.moments(metric, grain="daily")
    assert "frame.censoring.dropped_units" in warning_codes(rec)
    cohort_dates = {
        row["ds"].date() if hasattr(row["ds"], "date") else row["ds"] for row in cohorts
    }
    assert cohort_dates == {dt.date(2025, 1, 1)}
    source.close()


def test_artifact_total_grain_avg_calendar_day_uses_the_freshest_sibling_measures_spine():
    """The spine's day-axis edge is shared across every metric this reader
    session reduces (mirrors native_source.py's `_union_event_horizon`,
    whose default scope is every metric the source was constructed with),
    not just this metric's own lagging fact -- an `avg_calendar_day`
    divisor is literally the spine's day count per unit, so metric A,
    sharing a session with metric B, must get B's fresher day-axis length
    even though A never reads B's measure. The metric's own maturity/
    censor cap stays scoped to its own measure regardless."""
    from increment.frame import MetricSpec
    from increment.semantics.artifact import Freshness, MeasureManifest, SimpleMetricMeasure

    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
    ]
    stats = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 2),
            "measure_key": "a_orders",
            "n_events": 1,
            "sum_value": 10.0,
            "min_value": 10.0,
            "max_value": 10.0,
        },
    ]
    measures = (
        MeasureManifest(
            measure_key="a_orders",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 5), declared_complete=False),
        ),
        # Metric B shares this reader session; its fresher watermark must
        # still size the shared spine even though A never reads b_orders.
        MeasureManifest(
            measure_key="b_orders",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 10), declared_complete=False),
        ),
    )
    source = _custom_artifact_reader_source(
        exposures,
        stats,
        last_ds=dt.date(2025, 1, 11),
        measures=measures,
        metric_bindings=(
            SimpleMetricMeasure(metric_name="A", measure_key="a_orders"),
            SimpleMetricMeasure(metric_name="B", measure_key="b_orders"),
        ),
        metric_specs=(
            MetricSpec(name="A", type="mean", value_column="a_orders"),
            MetricSpec(name="B", type="mean", value_column="b_orders"),
        ),
        definitions_metrics=[
            {"name": "A", "aggregation": "avg_calendar_day"},
            {"name": "B", "aggregation": "sum"},
        ],
    )
    metric = next(m for m in source.context.metrics if m.name == "A")

    totals = source.moments(metric)
    row = next(r for r in totals if r["group_id"] == "control")
    # Spine day-axis is Jan1..Jan10 (10 days, sized off b_orders' fresher
    # watermark, not a_orders' own Jan5), with one event (10.0) on Jan2.
    assert row["ref_y"] == pytest.approx(10.0 / 10)
    source.close()


def test_artifact_reader_honours_fact_freshness_beyond_the_enrollment_edge():
    """A fact's own freshness watermark can legitimately exceed
    `manifest.last_ds` -- native scans the WHOLE fact table for this
    event type, including rows dated after the last enrollment, so a
    zero-event unit's window closing before that true watermark is a
    genuine observed zero (native-consistent), not a censoring drop. A
    unit enrolled AFTER even that watermark still has no observed day at
    all and stays a null placeholder."""
    from increment.frame import MetricSpec
    from increment.semantics.artifact import Freshness, MeasureManifest, SimpleMetricMeasure

    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "group_id": "treatment",
            "first_exposure_ts": dt.datetime(2025, 1, 25, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 25),
        },
    ]
    measures = (
        MeasureManifest(
            measure_key="converted",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 20), declared_complete=False),
        ),
    )
    source = _custom_artifact_reader_source(
        exposures,
        [],
        last_ds=dt.date(2025, 1, 5),
        measures=measures,
        metric_bindings=(SimpleMetricMeasure(metric_name="conv", measure_key="converted"),),
        metric_specs=(
            MetricSpec(name="conv", type="conversion", value_column="converted", window_days=10),
        ),
        definitions_metrics=[{"name": "conv", "type": "conversion", "window_days": 10}],
    )
    metric = source.context.metrics[0]

    # u1's window closes Jan 10, after last_ds (Jan 5) but inside the Jan 20
    # freshness watermark: an observed zero. u2 enrolls Jan 25, past the
    # watermark, and never surfaces. `_builder_total` uses
    # warn_on_censoring=False like every total-grain artifact reduction.
    totals = source.moments(metric)
    assert {row["group_id"]: row["ref_y"] for row in totals} == {"control": 0.0}
    source.close()


def test_artifact_reader_sentinel_freshness_falls_back_instead_of_exploding_the_spine():
    """A measure with literally zero matching rows reports its freshness
    as native's far-future 'no data' sentinel -- the shared spine edge
    must exclude it outright rather than extend the spine to a
    nonsensical date."""
    from increment.frame import MetricSpec
    from increment.query.native_source import NO_DATA_SIGNAL
    from increment.semantics.artifact import Freshness, MeasureManifest, SimpleMetricMeasure

    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
    ]
    measures = (
        MeasureManifest(
            measure_key="converted",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=NO_DATA_SIGNAL, declared_complete=True),
        ),
    )
    source = _custom_artifact_reader_source(
        exposures,
        [],
        last_ds=dt.date(2025, 1, 1),
        measures=measures,
        metric_bindings=(SimpleMetricMeasure(metric_name="conv", measure_key="converted"),),
        metric_specs=(MetricSpec(name="conv", type="conversion", value_column="converted"),),
        definitions_metrics=[{"name": "conv", "type": "conversion"}],
    )
    metric = source.context.metrics[0]

    totals = source.moments(metric)
    assert totals == []

    daily = source.moments(metric, grain="daily")
    assert daily == []
    source.close()


def test_artifact_reader_ratio_spine_uses_the_denominator_watermark_when_freshest():
    """native's `_union_event_horizon_for` appends a `RatioMetric`'s
    denominator events after its numerator's own, so the shared spine
    edge must account for BOTH -- a numerator-only scope would silently
    truncate the day axis to whichever part happens to be staler."""
    from increment.frame import MetricSpec
    from increment.semantics.artifact import Freshness, MeasureManifest, RatioMetricMeasure

    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
    ]
    stats = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 2),
            "measure_key": "num_orders",
            "n_events": 1,
            "sum_value": 10.0,
            "min_value": 10.0,
            "max_value": 10.0,
        },
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 2),
            "measure_key": "den_sessions",
            "n_events": 1,
            "sum_value": 2.0,
            "min_value": 2.0,
            "max_value": 2.0,
        },
    ]
    measures = (
        MeasureManifest(
            measure_key="num_orders",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 5), declared_complete=False),
        ),
        # The denominator's own watermark is the freshest of the two.
        MeasureManifest(
            measure_key="den_sessions",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 10), declared_complete=False),
        ),
    )
    source = _custom_artifact_reader_source(
        exposures,
        stats,
        last_ds=dt.date(2025, 1, 2),
        measures=measures,
        metric_bindings=(
            RatioMetricMeasure(
                metric_name="R",
                numerator_measure_key="num_orders",
                denominator_measure_key="den_sessions",
            ),
        ),
        metric_specs=(
            MetricSpec(name="R", type="ratio", numerator="num_orders", denominator="den_sessions"),
        ),
        definitions_metrics=[{"name": "R", "type": "ratio"}],
    )
    metric = source.context.metrics[0]

    daily = source.moments(metric, grain="daily")
    dates = {row["ds"].date() if hasattr(row["ds"], "date") else row["ds"] for row in daily}
    # The spine reaches Jan 10 (the denominator's fresher watermark), not
    # Jan 5 (the numerator's own).
    assert max(dates) == dt.date(2025, 1, 10)
    source.close()


def test_artifact_reader_daily_ratio_denominator_respects_the_numerator_window_bound():
    """Expired units cannot contaminate another unit's active daily denominator."""
    from increment.frame import MetricSpec
    from increment.semantics.artifact import Freshness, MeasureManifest, RatioMetricMeasure

    exposures = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 1),
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "group_id": "control",
            "first_exposure_ts": dt.datetime(2025, 1, 4, tzinfo=dt.UTC),
            "first_exposure_date": dt.date(2025, 1, 4),
        },
    ]
    stats = [
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 1),
            "measure_key": "num_orders",
            "n_events": 1,
            "sum_value": 5.0,
            "min_value": 5.0,
            "max_value": 5.0,
        },
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 1),
            "measure_key": "den_sessions",
            "n_events": 1,
            "sum_value": 2.0,
            "min_value": 2.0,
            "max_value": 2.0,
        },
        # Both fall AFTER the 2-day window (Jan 1-Jan 2 inclusive) closes.
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 5),
            "measure_key": "num_orders",
            "n_events": 1,
            "sum_value": 999.0,
            "min_value": 999.0,
            "max_value": 999.0,
        },
        {
            "experiment_id": "exp",
            "unit_id": "u1",
            "ds": dt.date(2025, 1, 5),
            "measure_key": "den_sessions",
            "n_events": 1,
            "sum_value": 999.0,
            "min_value": 999.0,
            "max_value": 999.0,
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "ds": dt.date(2025, 1, 5),
            "measure_key": "num_orders",
            "n_events": 1,
            "sum_value": 11.0,
            "min_value": 11.0,
            "max_value": 11.0,
        },
        {
            "experiment_id": "exp",
            "unit_id": "u2",
            "ds": dt.date(2025, 1, 5),
            "measure_key": "den_sessions",
            "n_events": 1,
            "sum_value": 4.0,
            "min_value": 4.0,
            "max_value": 4.0,
        },
    ]
    measures = (
        MeasureManifest(
            measure_key="num_orders",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 5), declared_complete=False),
        ),
        MeasureManifest(
            measure_key="den_sessions",
            source_provenance_sha256="0" * 64,
            freshness=Freshness(loaded_through=dt.date(2025, 1, 5), declared_complete=False),
        ),
    )
    source = _custom_artifact_reader_source(
        exposures,
        stats,
        last_ds=dt.date(2025, 1, 5),
        measures=measures,
        metric_bindings=(
            RatioMetricMeasure(
                metric_name="R",
                numerator_measure_key="num_orders",
                denominator_measure_key="den_sessions",
            ),
        ),
        metric_specs=(
            MetricSpec(
                name="R",
                type="ratio",
                numerator="num_orders",
                denominator="den_sessions",
                window_days=2,
            ),
        ),
        definitions_metrics=[
            {
                "name": "R",
                "type": "ratio",
                "numerator": {"aggregation": "sum", "window_days": 2},
                "denominator": {"aggregation": "sum", "window_days": 2},
            }
        ],
    )
    metric = source.context.metrics[0]

    daily = source.moments(metric, grain="daily")
    by_date = {row["ds"].date() if hasattr(row["ds"], "date") else row["ds"]: row for row in daily}
    assert set(by_date) == {dt.date(2025, 1, day) for day in (1, 2, 4, 5)}
    assert by_date[dt.date(2025, 1, 1)]["ref_den"] == pytest.approx(2.0)
    assert by_date[dt.date(2025, 1, 1)]["ref_y"] == pytest.approx(5.0)
    assert by_date[dt.date(2025, 1, 5)]["ref_den"] == pytest.approx(4.0)
    assert by_date[dt.date(2025, 1, 5)]["ref_y"] == pytest.approx(11.0)
    source.close()
