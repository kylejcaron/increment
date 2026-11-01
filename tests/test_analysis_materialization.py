"""Live warehouse materialization and operation-scoped freshness."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import ibis
import pytest

from increment import Scope, from_definitions, workflows
from increment.query import panel_sql, summary_sql
from increment.semantics.models import (
    AnalysisPlan,
    Definitions,
    Filter,
    MeanMetric,
    RetentionMetric,
)
from increment.workflows import materialize
from tests.analysis_factory import (
    _recovered_sum,
    bound_source,
    lift_rows,
    make_binding,
    make_binding_like,
    view_readout,
)

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_store_always_sees_an_in_place_metric_correction_on_the_next_run():
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    try:
        con.raw_sql(
            "UPDATE union_horizon_events SET value = value / 2 + 2 "
            "WHERE event = 'early_event' AND user_id LIKE 't%'"
        )
        analysis_experiment, analysis_resolver = _union_horizon_analysis(
            con, defs, store="always", metrics=[metric]
        )
        with analysis_resolver:
            before = (
                lift_rows(view_readout(analysis_experiment, analysis_resolver))[0]
                .require_lift()
                .value
            )
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            after = (
                lift_rows(view_readout(analysis_experiment, analysis_resolver))[0]
                .require_lift()
                .value
            )
            fresh_experiment, fresh_resolver = _union_horizon_analysis(
                con, defs, store="none", metrics=[metric]
            )
            with fresh_resolver:
                current = (
                    lift_rows(view_readout(fresh_experiment, fresh_resolver))[0]
                    .require_lift()
                    .value
                )
            assert before == pytest.approx(2 / 11)
            assert current == pytest.approx(15 / 11)
            assert after == pytest.approx(current)
    finally:
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["auto", "always"])
def test_readout_observes_corrections_after_materialization(store, tmp_path):
    import yaml

    from increment import run
    from increment.errors import IncrementWarning

    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    experiment_definition = defs.experiment("union_horizon_exp")
    assert experiment_definition is not None
    experiment_definition = experiment_definition.model_copy(
        update={"plan": AnalysisPlan(secondaries=(metric.name,))}
    )
    defs = defs.model_copy(
        update={
            "experiments": tuple(
                experiment_definition if item.name == experiment_definition.name else item
                for item in defs.experiments
            )
        }
    )
    definitions_path = tmp_path / "definitions.yaml"
    definitions_path.write_text(
        yaml.safe_dump(defs.model_dump(mode="json", exclude_unset=True)), encoding="utf-8"
    )
    with pytest.warns(IncrementWarning, match="early_avg"):
        experiment, resolver = from_definitions(
            definitions_path,
            con,
            experiment_name="union_horizon_exp",
            store=store,
        )
    try:
        materialize(experiment, resolver)
        con.raw_sql(
            "UPDATE union_horizon_events SET value = value * 2 "
            "WHERE event = 'early_event' AND user_id LIKE 't%'"
        )
        result = run(experiment, resolver)
        after = {
            (cell.identity.key.metric, cell.identity.key.group_id): effect.point
            for cell, effect in zip(result.cells, result.effects, strict=True)
            if effect is not None and effect.point is not None
        }
        assert after[("early_avg", "treatment")] == pytest.approx(3.0)
    finally:
        close = getattr(resolver, "close", None)
        if callable(close):
            close()
        con.disconnect()


@pytest.mark.slow
@pytest.mark.parametrize("store", ["auto", "always"])
def test_lazy_summary_remains_live_after_another_readout_refreshes_materialization(store):
    con, defs = _union_horizon_defs()
    metric = defs.metrics[0]
    try:
        experiment, resolver = _union_horizon_analysis(con, defs, store=store, metrics=[metric])
        with resolver:
            view_readout(experiment, resolver)
            view_readout(experiment, resolver)
            summary = con.sql(summary_sql(experiment, resolver)[metric.name])
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            view_readout(experiment, resolver)
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
        experiment, resolver = _union_horizon_analysis(con, defs, store=store, metrics=[metric])
        with resolver:
            view_readout(experiment, resolver)
            view_readout(experiment, resolver)
            panel = con.sql(panel_sql(experiment, resolver)[metric.name])
            unit_values = panel.group_by("unit_id").aggregate(y=panel.sum_value.mean())
            before = unit_values.to_pyarrow().to_pylist()
            con.raw_sql(
                "UPDATE union_horizon_events SET value = value * 2 "
                "WHERE event = 'early_event' AND user_id LIKE 't%'"
            )
            view_readout(experiment, resolver)
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
        experiment, resolver = _union_horizon_analysis(con, defs, store=store, metrics=[metric])
        with resolver:
            view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True))
            before = view_readout(
                experiment, resolver, Scope(view="daily", grain="daily", values=True)
            )
            assert {row.group_id: row.value.value for row in before if row.value} == {
                "control": 0.5,
                "treatment": 0.5,
            }
            con.raw_sql(
                "UPDATE union_horizon_events SET ts=TIMESTAMP '2025-08-03 10:00:00' "
                "WHERE event='late_event' AND user_id='t1'"
            )
            after = view_readout(
                experiment, resolver, Scope(view="daily", grain="daily", values=True)
            )
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

    def totals(analysis: tuple[Any, Any]):
        rows = con.sql(summary_sql(*analysis)["early_avg"]).to_pyarrow().to_pylist()
        return {row["group_id"]: _recovered_sum(row["n"], row["ref_y"], row["cy1"]) for row in rows}

    try:
        full = _union_horizon_analysis(con, defs, store="always", metrics=[metric])
        cohort = _union_horizon_analysis(con, cohort_defs, store="always", metrics=[metric])
        with full[1], cohort[1]:
            for treatment_total, cohort_total in [(22.0, 10.0), (32.0, 20.0), (52.0, 40.0)]:
                materialize(*full)
                materialize(*cohort)
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
    from increment.estimation.engine import Method

    original_experiment, original_resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="always"
    )
    shared_measure_metric = MeanMetric(
        name="total_session_duration",
        entity="user_id",
        fact="session_end",
        aggregation="sum",
        filters=[Filter(property="platform", op="equals", values=["web"])],
    )
    replacement = make_binding_like(
        original_resolver,
        [*workflows.metrics(original_experiment, original_resolver), shared_measure_metric],
    )
    cast(Any, original_resolver).close()
    a_experiment, a_resolver = replacement
    with a_resolver:
        shared_results = {
            (r.metric, r.group_id): r.require_lift().value
            for r in lift_rows(
                view_readout(a_experiment, a_resolver, decision_method=Method(name="unadjusted"))
            )
            if r.metric == "total_session_duration"
        }

    unshared_original_experiment, unshared_original_resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2", store="none"
    )
    unshared_experiment, unshared_resolver = make_binding_like(
        unshared_original_resolver,
        [
            *workflows.metrics(unshared_original_experiment, unshared_original_resolver),
            shared_measure_metric,
        ],
    )
    cast(Any, unshared_original_resolver).close()
    with unshared_resolver:
        unshared_results = {
            (r.metric, r.group_id): r.require_lift().value
            for r in lift_rows(
                view_readout(
                    unshared_experiment,
                    unshared_resolver,
                    decision_method=Method(name="unadjusted"),
                )
            )
            if r.metric == "total_session_duration"
        }
    assert shared_results, "the shared-measure metric must actually produce rows"
    assert shared_results == pytest.approx(unshared_results)


def test_materialize_on_zero_metrics_does_not_crash(seeded_con, seeded_defs, tmp_path):
    """An empty source metric selection remains empty after explicit materialization."""
    import yaml

    from increment import run
    from increment.semantics.loader import load
    from increment.workflows import metrics

    definitions = load(seeded_defs)
    experiment_definition = definitions.experiment("pricing_tier_test")
    assert experiment_definition is not None
    empty_experiment = experiment_definition.model_copy(update={"plan": AnalysisPlan()})
    definitions = definitions.model_copy(
        update={
            "experiments": tuple(
                empty_experiment if item.name == empty_experiment.name else item
                for item in definitions.experiments
            )
        }
    )
    definitions_path = tmp_path / "definitions.yaml"
    definitions_path.write_text(
        yaml.safe_dump(definitions.model_dump(mode="json", exclude_unset=True)),
        encoding="utf-8",
    )
    experiment, resolver = from_definitions(
        definitions_path, seeded_con, experiment_name="pricing_tier_test"
    )
    try:
        assert metrics(experiment, resolver) == []
        assert materialize(experiment, resolver) is None
        assert run(experiment, resolver).cells == ()
    finally:
        close = getattr(resolver, "close", None)
        if callable(close):
            close()


def test_temp_name_caps_at_backend_identifier_limit(seeded_con, seeded_defs):
    """``_temp_name`` stays within backend identifier limits."""
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    with cast(Any, resolver):
        source = bound_source(resolver)
        short = source._temp_name("exp", "spine", "abc123")
        assert len(short) <= 63, len(short)

        long_a = source._temp_name("a" * 40, "stats", "metric_one")
        long_b = source._temp_name("a" * 40, "stats", "metric_two")
        assert len(long_a) <= 63, len(long_a)
        assert len(long_b) <= 63, len(long_b)
        assert long_a != long_b


def test_temp_name_caps_by_byte_length_not_char_count(seeded_con, seeded_defs):
    """UTF-8 byte length, rather than character count, controls capping."""
    experiment, resolver = from_definitions(
        seeded_defs, seeded_con, experiment_name="new_onboarding_v2"
    )
    with cast(Any, resolver):
        source = bound_source(resolver)
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
    analysis = make_binding(
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


@pytest.mark.slow
@pytest.mark.integration
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
    none_experiment, none_resolver = _union_horizon_analysis(con, defs, store="none")
    always_experiment, always_resolver = _union_horizon_analysis(con, defs, store="always")

    # summary_sql() never calls _note_reduction() (only stage="moments" does) - force materialization explicitly so this compares the fused path against the materialized one, not two identical fused computations.
    materialize(always_experiment, always_resolver)
    for name in ("early_avg", "late_conv"):
        none_sql = summary_sql(none_experiment, none_resolver)[name]
        always_sql = summary_sql(always_experiment, always_resolver)[name]
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


@pytest.mark.integration
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

    def _arm_sums(analysis: tuple[Any, Any]):
        df = (
            con.sql(summary_sql(*analysis)["early_avg"])
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


@pytest.mark.integration
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
    a_experiment, a_resolver = _union_horizon_analysis(con, defs, store="none")

    panel_df = con.sql(panel_sql(a_experiment, a_resolver)["early_avg"]).execute()
    day_counts = panel_df.groupby("unit_id").size()
    assert (day_counts == 14).all(), (
        f"early_avg's panel must span the union bound (14 days/unit), got {day_counts.to_dict()}"
    )

    solo_a_experiment, solo_a_resolver = _union_horizon_analysis(
        con,
        defs,
        store="none",
        metrics=[m for m in workflows.metrics(a_experiment, a_resolver) if m.name == "early_avg"],
    )
    solo_panel_df = con.sql(panel_sql(solo_a_experiment, solo_a_resolver)["early_avg"]).execute()
    solo_day_counts = solo_panel_df.groupby("unit_id").size()
    assert (solo_day_counts == 2).all(), (
        "early_avg analysed alone must span only its own bound (2 days/unit), got "
        f"{solo_day_counts.to_dict()}"
    )
