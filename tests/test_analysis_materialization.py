"""Live warehouse materialization and operation-scoped freshness."""

from __future__ import annotations

from datetime import datetime

import ibis
import pytest

from increment import Analysis
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    Filter,
    MeanMetric,
    RetentionMetric,
)
from tests.analysis_factory import (
    _native_source,
    _recovered_sum,
    lift_rows,
    make_analysis,
    make_analysis_like,
)


@pytest.mark.slow
def test_store_always_sees_an_in_place_metric_correction_on_the_next_run():
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    try:
        con.raw_sql(
            "UPDATE union_horizon_events SET value = value / 2 + 2 "
            "WHERE event = 'early_event' AND user_id LIKE 't%'"
        )
        with _union_horizon_analysis(con, defs, store="always", metrics=[metric]) as analysis:
            before = lift_rows(analysis.run())[0].require_lift().value
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            after = lift_rows(analysis.run())[0].require_lift().value
            with _union_horizon_analysis(con, defs, store="none", metrics=[metric]) as fresh:
                current = lift_rows(fresh.run())[0].require_lift().value
            assert before == pytest.approx(2 / 11)
            assert current == pytest.approx(15 / 11)
            assert after == pytest.approx(current)
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["auto", "always"])
def test_direct_unit_frame_observes_corrections_after_materialization(store):
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    try:
        with _union_horizon_analysis(con, defs, store=store, metrics=[metric]) as analysis:
            analysis.materialize()
            source = _native_source(analysis)
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            after = source.unit_frame(metric).to_pylist()
            assert {row["unit_id"]: row["y"] for row in after} == pytest.approx(
                {"c0": 5.0, "c1": 6.0, "t0": 20.0, "t1": 24.0}
            )
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["auto", "always"])
def test_lazy_summary_remains_live_after_another_readout_refreshes_materialization(store):
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    try:
        with _union_horizon_analysis(con, defs, store=store, metrics=[metric]) as analysis:
            analysis.run()
            analysis.run()
            summary = con.sql(analysis.summary_sql()[metric.name])
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            analysis.run()
            records = summary.to_pyarrow().to_pylist()
            assert {
                row["group_id"]: _recovered_sum(row["n"], row["ref_y"], row["cy1"])
                for row in records
            } == pytest.approx({"control": 11.0, "treatment": 44.0})
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["auto", "always"])
def test_saved_panel_query_remains_live_after_materialization_refresh(store):
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    try:
        with _union_horizon_analysis(con, defs, store=store, metrics=[metric]) as analysis:
            analysis.run()
            analysis.run()
            panel = con.sql(analysis.panel_sql()[metric.name])
            unit_values = panel.group_by("unit_id").aggregate(y=panel.sum_value.mean())
            before = unit_values.to_pyarrow().to_pylist()
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            analysis.run()
            after = unit_values.to_pyarrow().to_pylist()
            assert {row["unit_id"]: row["y"] for row in before} == pytest.approx(
                {"c0": 5.0, "c1": 6.0, "t0": 10.0, "t1": 12.0}
            )
            assert {row["unit_id"]: row["y"] for row in after} == pytest.approx(
                {"c0": 5.0, "c1": 6.0, "t0": 20.0, "t1": 24.0}
            )
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["auto", "always"])
def test_retention_cohort_observes_in_place_return_time_corrections(store):
    con, defs = _union_horizon_defs()
    metric = RetentionMetric(
        name="return_rate", entity="user_id", fact="late_event", threshold_days=(1, 3)
    )
    try:
        with _union_horizon_analysis(con, defs, store=store, metrics=[metric]) as analysis:
            analysis.run_daily()
            before = analysis.run_daily()
            assert {row.group_id: row.value.value for row in before if row.value} == {
                "control": 0.5,
                "treatment": 0.5,
            }
            con.raw_sql(
                "UPDATE union_horizon_events SET ts=TIMESTAMP '2025-08-03 10:00:00' "
                "WHERE event='late_event' AND user_id='t1'"
            )
            after = analysis.run_daily()
            assert {row.ds_basis for row in after} == {"cohort"}
            assert {row.ds for row in after} == {datetime(2025, 8, 1).date()}
            assert {row.group_id: row.value.value for row in after if row.value} == {
                "control": 0.5,
                "treatment": 1.0,
            }
    finally:
        con.disconnect()


@pytest.mark.slow
def test_repeated_materialize_refreshes_separate_cohorts_on_one_connection():
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    cohort_defs = defs.model_copy(
        update={
            "fact_sources": (
                defs.fact_sources[0].model_copy(
                    update={"sql": "SELECT * FROM union_horizon_events WHERE user_id <> 't1'"}
                ),
            )
        }
    )

    def totals(analysis):
        rows = con.sql(analysis.summary_sql()["early_avg"]).to_pyarrow().to_pylist()
        return {row["group_id"]: _recovered_sum(row["n"], row["ref_y"], row["cy1"]) for row in rows}

    try:
        with (
            _union_horizon_analysis(con, defs, store="always", metrics=[metric]) as full,
            _union_horizon_analysis(con, cohort_defs, store="always", metrics=[metric]) as cohort,
        ):
            for treatment_total, cohort_total in [(22.0, 10.0), (32.0, 20.0), (52.0, 40.0)]:
                full.materialize()
                cohort.materialize()
                assert totals(full) == pytest.approx(
                    {"control": 11.0, "treatment": treatment_total}
                )
                assert totals(cohort) == pytest.approx({"control": 11.0, "treatment": cohort_total})
                con.raw_sql(
                    "UPDATE union_horizon_events SET value=value*2 "
                    "WHERE event='early_event' AND user_id='t0'"
                )
    finally:
        con.disconnect()


@pytest.mark.slow
def test_metrics_sharing_a_measure_keep_their_numeric_answers(seeded_con, seeded_defs):
    """Different aggregations of a shared measure retain their own answers."""
    original = Analysis.from_definitions(
        "new_onboarding_v2", seeded_defs, seeded_con, store="always"
    )
    shared_measure_metric = MeanMetric(
        name="total_session_duration",
        entity="user_id",
        fact="session_end",
        aggregation="sum",
        filters=[Filter(property="platform", op="equals", values=["web"])],
    )
    replacement = make_analysis_like(original, [*original.metrics, shared_measure_metric])
    original.close()
    with replacement as a:
        shared_results = {
            (r.metric, r.group_id): r.require_lift().value
            for r in lift_rows(a.run())
            if r.metric == "total_session_duration"
        }

    unshared_original = Analysis.from_definitions(
        "new_onboarding_v2", seeded_defs, seeded_con, store="none"
    )
    unshared = make_analysis_like(
        unshared_original,
        [*unshared_original.metrics, shared_measure_metric],
    )
    unshared_original.close()
    with unshared:
        unshared_results = {
            (r.metric, r.group_id): r.require_lift().value
            for r in lift_rows(unshared.run())
            if r.metric == "total_session_duration"
        }
    assert shared_results, "the shared-measure metric must actually produce rows"
    assert shared_results == pytest.approx(unshared_results)


def test_materialize_on_zero_metrics_does_not_crash(con):
    """An empty metric selection stays empty after explicit materialization."""
    analysis = Analysis(
        "pricing_tier_test", definitions_path="examples/definitions/", con=con, store="auto"
    )
    analysis = make_analysis_like(analysis, [], plan=AnalysisPlan())

    assert lift_rows(analysis.materialize().run()) == []


def test_temp_name_caps_at_backend_identifier_limit(seeded_con, seeded_defs):
    """``_temp_name`` stays within backend identifier limits."""
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con) as analysis:
        source = _native_source(analysis)
        short = source._temp_name("exp", "spine", "abc123")
        assert len(short) <= 63, len(short)

        long_a = source._temp_name("a" * 40, "stats", "metric_one")
        long_b = source._temp_name("a" * 40, "stats", "metric_two")
        assert len(long_a) <= 63, len(long_a)
        assert len(long_b) <= 63, len(long_b)
        assert long_a != long_b


def test_temp_name_caps_by_byte_length_not_char_count(seeded_con, seeded_defs):
    """UTF-8 byte length, rather than character count, controls capping."""
    with Analysis.from_definitions("new_onboarding_v2", seeded_defs, seeded_con) as analysis:
        source = _native_source(analysis)
        under_byte_limit = "\u00e9" * 21
        assert len(under_byte_limit.encode()) == 42
        assert source._temp_name(under_byte_limit).startswith(under_byte_limit)

        over_byte_limit = "\u4e2d" * 22
        assert len(over_byte_limit) == 22
        assert len(over_byte_limit.encode()) == 66
        result = source._temp_name(over_byte_limit)
        assert result != over_byte_limit
        assert len(result.encode()) <= 63, len(result.encode())


def _union_horizon_defs():
    """One experiment declaring BOTH ``early_avg`` (an unbounded - no
    ``window_days`` - ``avg`` :class:`MeanMetric`, whose own last event
    is day 2) and ``late_conv`` (a :class:`ConversionMetric` whose own
    last event is day 14), still running (no ``end``/``observation_end``).

    ``early_avg``'s spine right edge is the union of every metric
    DECLARED ON self._metrics's own event horizon - day 14 when
    ``late_conv`` is also in play, day 2 when ``early_avg`` is analysed
    alone (see :func:`_union_horizon_analysis`'s ``metrics=`` override).
    An avg metric with no ``window_days`` divides by the spine's day
    count (``dense.sum_value.sum() / dense.count()``), so these two
    spines produce genuinely different numeric answers for the identical
    metric over the identical units - the two tests below both lean on
    that.
    """
    con = ibis.duckdb.connect()
    exposure_rows = [
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 1, 9, 0, 0),
            "event": "page_view",
            "group_id": g,
            "experiment_id": "union_horizon_exp",
            "value": None,
        }
        for g, uids in [("control", ["c0", "c1"]), ("treatment", ["t0", "t1"])]
        for uid in uids
    ]
    early_values = {"c0": 10.0, "c1": 12.0, "t0": 20.0, "t1": 24.0}
    early_rows = [
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 2, 10, 0, 0),
            "event": "early_event",
            "group_id": None,
            "experiment_id": None,
            "value": v,
        }
        for uid, v in early_values.items()
    ]
    # Only c0/t0 convert in-window (day 3, well inside window_days=7); c1/t1 stay unconverted, so each arm has nonzero variance. Censoring cares only about maturity, not whether an occurrence sits inside or outside the window.
    late_rows = [
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 3, 10, 0, 0),
            "event": "late_event",
            "group_id": None,
            "experiment_id": None,
            "value": None,
        }
        for uid in ("c0", "t0")
    ] + [
        # A second occurrence per unit, on day 14, outside window_days=7 (doesn't count toward conversion) but still part of the raw event stream union_event_horizon reads - pushes early_avg's shared spine to day 14 while late_conv's own maturity (day 8) still clears it, so no censoring.
        {
            "user_id": uid,
            "ts": datetime(2025, 8, 14, 10, 0, 0),
            "event": "late_event",
            "group_id": None,
            "experiment_id": None,
            "value": None,
        }
        for uid in early_values
    ]
    con.create_table("union_horizon_events", obj=exposure_rows + early_rows + late_rows)

    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM union_horizon_events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "page_view", "column": None},
                        {"name": "early_event", "column": "value"},
                        {"name": "late_event", "column": None},
                    ],
                }
            ],
            "exposures": [{"name": "e", "fact": "page_view"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "early_avg",
                    "entity": "user_id",
                    "fact": "early_event",
                    "aggregation": "avg_calendar_day",
                    # window_days omitted: unbounded per-unit window -
                    # the branch that divides by the spine's day count.
                },
                {
                    "type": "conversion",
                    "name": "late_conv",
                    "entity": "user_id",
                    "fact": "late_event",
                    "window_days": 7,
                },
            ],
            "experiments": [
                {
                    "name": "union_horizon_exp",
                    "exposure": "e",
                    "unit": "user_id",
                    "start": "2025-08-01",
                    # end/observation_end both omitted - still running
                    "control_group": "control",
                    "plan": {"secondaries": ["early_avg", "late_conv"]},
                },
            ],
        }
    )
    return con, defs


def _union_horizon_analysis(con, defs, *, store, metrics=None):
    """Build an ``Analysis`` for ``union_horizon_exp``.

    *metrics*, when given, is threaded through ``make_analysis``'s
    ``metrics=`` kwarg, which wires the selected metric list directly while
    building the instance. This scopes which metrics are declared for the
    instance without needing a second experiment or exposure population.
    """
    experiment = defs.experiment("union_horizon_exp")
    analysis = make_analysis(
        con,
        defs,
        experiment=experiment,
        store=store,
        metrics=metrics,
        plan=(
            AnalysisPlan(secondaries=tuple(m.name for m in metrics))
            if metrics is not None
            else None
        ),
    )
    return analysis


def test_store_never_changes_the_numeric_answer_for_an_unbounded_avg_metric():
    """Regression: an unbounded (no ``window_days``) ``avg`` MeanMetric on
    a still-running experiment must return the SAME value whether or not
    *store* triggers materialization.

    Before the union-event-horizon fix, the fused (single-readout) path
    bounded the panel spine's right edge by THIS metric's own event max,
    while the materialized path bounded it by the union across ALL
    declared metrics - for ``early_avg`` (``y = sum_value.sum() /
    dense.count()``, dividing by the spine's day count), a wider union
    horizon changes the denominator and so the numeric answer, purely as
    a side effect of *store*'s policy and how many readouts happened to
    run first. See :func:`increment.query.builders.union_event_horizon`.

    Asserts on the ABSOLUTE per-arm ``sum_y`` (executed via
    ``summary_sql()``), not the relative :attr:`LiftEstimate.lift.value`
    - a spine-day-count change is a uniform multiplicative factor
    applied identically to every unit, which cancels out of a RELATIVE
    lift by construction and would make that assertion pass vacuously
    even with the bug present.
    """
    con, defs = _union_horizon_defs()
    none_analysis = _union_horizon_analysis(con, defs, store="none")
    always_analysis = _union_horizon_analysis(con, defs, store="always")

    # summary_sql() never calls _note_reduction() (only stage="moments" does) - force materialization explicitly so this compares the fused path against the materialized one, not two identical fused computations.
    always_analysis.materialize()
    for name in ("early_avg", "late_conv"):
        none_sql = none_analysis.summary_sql()[name]
        always_sql = always_analysis.summary_sql()[name]
        none_df = con.sql(none_sql).execute().set_index("group_id").sort_index()
        always_df = con.sql(always_sql).execute().set_index("group_id").sort_index()
        assert none_df["n"].equals(always_df["n"]), (
            f"{name}: arm sizes differ between store='none' and store='always' -- "
            f"{none_df['n'].to_dict()} vs {always_df['n'].to_dict()}"
        )
        none_sum = _recovered_sum(none_df["n"], none_df["ref_y"], none_df["cy1"])
        always_sum = _recovered_sum(always_df["n"], always_df["ref_y"], always_df["cy1"])
        assert none_sum.to_numpy() == pytest.approx(always_sum.to_numpy()), (
            f"{name}: sum_y diverged between store='none' "
            f"({none_sum.to_dict()}) and store='always' "
            f"({always_sum.to_dict()})"
        )


def test_unbounded_avg_metric_spine_is_bounded_by_every_declared_metric_not_just_itself():
    """Non-vacuousness check for the test above: proves the union horizon
    is actually load-bearing (not a fixture that happens to agree either
    way) by comparing ``early_avg`` under two DIFFERENT declared metric
    sets on the SAME experiment/population.

    Sharing the spine with ``late_conv`` (own last event: day 14) vs.
    analysing ``early_avg`` alone (own last event: day 2) - both under
    ``store='none'`` - is the only thing that differs, so a real
    difference here is attributable to the union bound, not to *store*.

    Compares ``sum_y`` (not relative lift - see the sibling test's
    docstring for why a uniform per-unit rescale cancels out of a
    relative lift and would make this assertion vacuous).
    """
    con, defs = _union_horizon_defs()
    metric_lookup = {m.name: m for m in defs.metrics}
    shared_analysis = _union_horizon_analysis(con, defs, store="none")
    solo_analysis = _union_horizon_analysis(
        con, defs, store="none", metrics=[metric_lookup["early_avg"]]
    )

    def _arm_sums(analysis):
        df = (
            con.sql(analysis.summary_sql()["early_avg"])
            .execute()
            .set_index("group_id")
            .sort_index()
        )
        return _recovered_sum(df["n"], df["ref_y"], df["cy1"])

    shared_sum_y = _arm_sums(shared_analysis)
    solo_sum_y = _arm_sums(solo_analysis)

    assert shared_sum_y.to_numpy() != pytest.approx(solo_sum_y.to_numpy()), (
        "early_avg's sum_y must differ between a shared (day-14) and a solo "
        f"(day-2) spine bound -- got {shared_sum_y.to_dict()} vs "
        f"{solo_sum_y.to_dict()}, which would mean the union horizon has no effect"
    )
    # Concrete expected values, hand-derived: day-2 events {c0:10, c1:12, t0:20, t1:24} divided by the spine's day count. Shared spine = 14 days/unit (bounded by late_conv); solo spine = 2 days/unit (bounded only by early_avg).
    assert shared_sum_y["control"] == pytest.approx((10.0 + 12.0) / 14)
    assert shared_sum_y["treatment"] == pytest.approx((20.0 + 24.0) / 14)
    assert solo_sum_y["control"] == pytest.approx((10.0 + 12.0) / 2)
    assert solo_sum_y["treatment"] == pytest.approx((20.0 + 24.0) / 2)


def test_run_daily_and_panel_sql_share_the_same_union_bound_as_run():
    """The panel-path readouts (``run_daily``/``run_asof``/``panel_sql``,
    fed by :func:`~increment.query.builders.unit_day_panel`) must use the
    IDENTICAL spine right edge as the spine-path readouts (``run``,
    fed by :func:`~increment.query.builders.unit_day_spine_stats`);
    both derive from the SAME :meth:`Analysis._union_event_horizon`.

    Regression: threading ``end_date`` into ``unit_day_spine_stats``
    alone (without also threading it into ``unit_day_panel``) would
    make ``run()`` union-bounded while ``run_daily()``/``panel_sql()``
    stayed bounded by each metric's own event max - the SAME metric
    reporting two different values depending on which readout you
    called. Verified directly on ``panel_sql()``'s generated SQL (no
    execution needed for the row-count assertion): ``early_avg``'s own
    last event is day 2, but the union with ``late_conv`` (day 14)
    means every unit's panel spans 14 calendar days, not 2.
    """
    con, defs = _union_horizon_defs()
    a = _union_horizon_analysis(con, defs, store="none")

    panel_df = con.sql(a.panel_sql()["early_avg"]).execute()
    day_counts = panel_df.groupby("unit_id").size()
    assert (day_counts == 14).all(), (
        f"early_avg's panel must span the union bound (14 days/unit), got {day_counts.to_dict()}"
    )

    solo_a = _union_horizon_analysis(
        con, defs, store="none", metrics=[m for m in a.metrics if m.name == "early_avg"]
    )
    solo_panel_df = con.sql(solo_a.panel_sql()["early_avg"]).execute()
    solo_day_counts = solo_panel_df.groupby("unit_id").size()
    assert (solo_day_counts == 2).all(), (
        "early_avg analysed alone must span only its own bound (2 days/unit), got "
        f"{solo_day_counts.to_dict()}"
    )
