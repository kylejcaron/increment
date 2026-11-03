from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from ibis.backends.sql import SQLBackend

from increment import run
from increment.builders import from_definitions
from increment.declarations import Method
from increment.errors import CodedError, IncrementWarning
from increment.evidence import BreakoutIdentity
from increment.kernel.contracts import Scope
from increment.query import panel_sql
from increment.semantics.models import AnalysisPlan, Definitions
from tests.analysis_factory import make_binding, view_readout
from tests.parity_harness import dataset as ds
from tests.parity_harness.comparison import nested_close

pytestmark = pytest.mark.integration


def _analysis(con, *, running=False):
    definitions = ds.definitions_dict(
        plan=AnalysisPlan(primary="revenue", secondaries=["purchase_rate"])
    )
    if running:
        definitions["experiments"][0].pop("end")
        definitions["fact_sources"][0]["facts"].append({"name": "late_signal", "column": None})
        definitions["metrics"][0].update(aggregation="avg_calendar_day", window_days=None)
        definitions["metrics"][1].update(fact="late_signal", window_days=None)
        definitions["metrics"][3].update(
            aggregation="avg_calendar_day",
            window_days=None,
        )
    return make_binding(con, Definitions.model_validate(definitions), experiment="exp")


def _builder(
    con,
    tmp_path,
    *,
    triggered=False,
    winsorized=False,
    trigger_evidence=True,
    separate_metric_source=False,
    running_revenue=False,
    include_breakouts=False,
):
    from increment.artifacts import SourceSnapshotEvidence

    definitions = ds.definitions_dict(
        plan=AnalysisPlan(primary="revenue", secondaries=["purchase_rate"]),
        breakout=not separate_metric_source or include_breakouts,
    )
    if running_revenue:
        next(metric for metric in definitions["metrics"] if metric["name"] == "revenue")[
            "aggregation"
        ] = "avg_calendar_day"
        next(metric for metric in definitions["metrics"] if metric["name"] == "revenue")[
            "window_days"
        ] = None
        definitions["experiments"][0].pop("end")
    kwargs: dict[str, Any] = {}
    if winsorized:
        definitions["metrics"][0]["winsorization"] = {"upper_percentile": 0.9}
    if separate_metric_source:
        event_source = definitions["fact_sources"][0]
        purchase_fact = next(fact for fact in event_source["facts"] if fact["name"] == "purchase")
        event_source["facts"] = [
            fact for fact in event_source["facts"] if fact["name"] != "purchase"
        ]
        definitions["fact_sources"].extend(
            (
                {
                    "name": "purchase_events",
                    "sql": "SELECT * FROM events WHERE event = 'purchase'",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [purchase_fact],
                    "properties": [
                        {
                            "name": "store",
                            "column": "store_id",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                },
                {
                    "name": "alternate_events",
                    "sql": "SELECT * FROM alternate_events WHERE event = 'purchase'",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [{"name": "alternate_purchase", "column": "revenue"}],
                    "properties": [
                        {
                            "name": "store",
                            "column": "store_id",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                },
            )
        )
        if running_revenue:
            definitions["fact_sources"][-1]["sql"] = (
                "SELECT * REPLACE ('alternate_purchase' AS event) "
                "FROM alternate_events WHERE event = 'purchase'"
            )
        if include_breakouts:
            definitions["experiments"][0]["breakouts"][0]["source"] = "events"
    if triggered:
        definitions["fact_sources"][0]["facts"].append({"name": "trigger_signal", "column": None})
        definitions["exposures"].append({"name": "trigger", "fact": "trigger_signal"})
        con.raw_sql(
            "INSERT INTO events SELECT * REPLACE (event_at + INTERVAL 1 HOUR AS event_at, "
            "'trigger_signal' AS event) FROM events WHERE event = 'exposure'"
        )
        definitions["experiments"][0]["trigger"] = "trigger"
        if trigger_evidence:
            kwargs["source_snapshot_evidence"] = SourceSnapshotEvidence(
                observation_cutoff_ts=datetime(2025, 2, 16, tzinfo=UTC),
                complete_through_by_feed={"events": datetime(2025, 2, 16, tzinfo=UTC)},
            )
    path = tmp_path / "definitions.yaml"
    path.write_text(json.dumps(definitions), encoding="utf-8")
    if running_revenue:
        with pytest.warns(IncrementWarning):
            return from_definitions(path, con, experiment_name="exp", store="none", **kwargs)
    return from_definitions(path, con, experiment_name="exp", store="none", **kwargs)


def _scope(cell) -> Scope:
    scope = cell.identity.scope
    assert scope is not None
    return scope


def _same_cells(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected, strict=True):
        assert left.identity.model_dump(
            exclude={"source_content_digest"}
        ) == right.identity.model_dump(exclude={"source_content_digest"})
        assert (left.identity.source_content_digest is None) == (
            right.identity.source_content_digest is None
        )
        assert nested_close(left.to_payload()["row_payload"], right.to_payload()["row_payload"])
        assert left.summary == right.summary or (
            left.summary is not None
            and right.summary is not None
            and left.summary.point == pytest.approx(right.summary.point, rel=1e-9)
            and left.summary.lower == pytest.approx(right.summary.lower, rel=1e-9)
            and left.summary.upper == pytest.approx(right.summary.upper, rel=1e-9)
        )


# Compare narrowed and complete warehouse families without dropping the late horizon.
@pytest.mark.slow
def test_narrowed_running_family_keeps_unselected_later_horizon():
    rows = ds.event_rows()
    for row in rows:
        if row["user_id"].startswith("t"):
            row["event_at"] = row["event_at"].replace(day=row["event_at"].day + 2)
    rows.append(
        {
            **rows[0],
            "event": "late_signal",
            "event_at": datetime(2025, 1, 25),
            "revenue": None,
            "sess": None,
        }
    )
    rows.append(
        {
            **rows[0],
            "event": "session_end",
            "event_at": datetime(2025, 1, 30),
            "revenue": None,
            "sess": 1,
        }
    )
    con = ds.duckdb_connection(rows)
    analysis_experiment, analysis_resolver = _analysis(con, running=True)
    try:
        declared = view_readout(
            analysis_experiment,
            analysis_resolver,
            metrics=["revenue"],
            decision_method=Method(name="unadjusted"),
        )
        all_declared = view_readout(
            analysis_experiment, analysis_resolver, decision_method=Method(name="unadjusted")
        )
        full_revenue = next(row for row in all_declared if row.metric == "revenue")
        narrowed_revenue = next(row for row in declared if row.metric == "revenue")
        assert nested_close(
            narrowed_revenue.lift.model_dump(),
            full_revenue.lift.model_dump(),
        )
        panel = (
            cast(Any, con)
            .sql(panel_sql(analysis_experiment, analysis_resolver)["revenue"])
            .execute()
        )
        observed_day_counts = panel.groupby("unit_id").size().to_dict()
        expected_day_counts = {
            row["user_id"]: (datetime(2025, 1, 25).date() - row["event_at"].date()).days + 1
            for row in rows
            if row["event"] == "exposure"
        }
        assert observed_day_counts == expected_day_counts
        live_experiment, live_resolver = _analysis(con, running=True)
        try:
            live_narrowed = view_readout(
                live_experiment,
                live_resolver,
                metrics=["revenue"],
                coherence="live",
                decision_method=Method(name="unadjusted"),
            )
            live_all = view_readout(
                live_experiment,
                live_resolver,
                coherence="live",
                decision_method=Method(name="unadjusted"),
            )
            live_narrowed_revenue = next(row for row in live_narrowed if row.metric == "revenue")
            live_all_revenue = next(row for row in live_all if row.metric == "revenue")
            assert nested_close(
                live_narrowed_revenue.lift.model_dump(),
                live_all_revenue.lift.model_dump(),
            )
        finally:
            live_resolver.close()
        extra = view_readout(
            analysis_experiment,
            analysis_resolver,
            metrics=[],
            exploratory_metrics=["rps", "revenue_cuped"],
            decision_method=Method(name="unadjusted"),
        )
        actual = view_readout(
            analysis_experiment,
            analysis_resolver,
            metrics=["revenue"],
            exploratory_metrics=["rps", "revenue_cuped"],
            decision_method=Method(name="unadjusted"),
        )
        from tests.capability_consumers_auxiliary import assert_support

        assert_support("shared_capture", "exploratory_from_definitions", actual)
        assert len(actual.metadata.captures) == 1
        assert {row.metric for row in actual} == {"revenue", "rps", "revenue_cuped"}
        assert nested_close(
            [row.model_dump(exclude={"source_snapshot_id"}) for row in actual],
            [row.model_dump(exclude={"source_snapshot_id"}) for row in (*declared, *extra)],
        )
    finally:
        analysis_resolver.close()
        con.disconnect()


# Compare separate and shared captures for all four views, then persist and replay.
@pytest.mark.slow
def test_registered_views_share_capture_preserve_cells_and_roundtrip(tmp_path):
    from increment.result_envelope import AnalysisEnvelope
    from increment.store import ParquetResultsStore

    rows = ds.event_rows()
    rows.append(
        {
            **next(row for row in rows if row["event"] == "purchase"),
            "event": "ad_hoc_late",
            "event_at": datetime(2025, 2, 20),
            "revenue": 100.0,
            "sess": None,
        }
    )
    con = ds.duckdb_connection(rows)
    experiment, resolver = _builder(con, tmp_path)
    try:
        scopes = (
            Scope(view="run", grain="total"),
            Scope(view="daily", grain="daily"),
            Scope(view="asof", grain="asof"),
            Scope(view="breakout", grain="total", segment=(BreakoutIdentity("store", None),)),
        )
        experiment = replace(experiment, scopes=scopes)
        separate = [
            run(
                replace(experiment, scopes=(scope,)),
                resolver,
                decision_method=Method(name="unadjusted"),
            )
            for scope in scopes
        ]
        actual = run(experiment, resolver, decision_method=Method(name="unadjusted"))
        for scope, expected in zip(scopes, separate, strict=True):
            cells = tuple(cell for cell in actual.cells if _scope(cell).view == scope.view)
            _same_cells(cells, expected.cells)
        assert nested_close(
            [asdict(effect) if effect is not None else None for effect in actual.effects],
            [
                asdict(effect) if effect is not None else None
                for result in separate
                for effect in result.effects
            ],
        )
        assert len({cell.provenance.run_id for cell in actual.cells}) == 1
        monitoring = {cell.identity.hash() for cell in actual.cells if _scope(cell).view != "run"}
        assert all(
            verdict.reason_codes == ("decision.not_evaluated.monitoring_view",)
            for verdict in actual.decisions.cells
            if verdict.identity_hash in monitoring
        )
        assert actual.decisions.program == separate[0].decisions.program
        assert all(family in actual.decisions.families for family in separate[0].decisions.families)
        assert len(actual.run.metadata.captures) == 1
        restored = AnalysisEnvelope.from_payload(actual.envelope.to_payload())
        assert restored.cells == actual.cells
        assert pickle.loads(pickle.dumps(actual)).cells == actual.cells
        store = ParquetResultsStore(tmp_path / "runs")
        store.put(actual.run)
        saved = store.open_run(
            actual.run.experiment_id, actual.run.analysis_spec_hash, actual.run.run_id
        ).run
        assert saved.cells == actual.cells
        assert saved.metadata == actual.run.metadata
        selected = run(
            experiment,
            resolver,
            views=(scopes[2], scopes[0]),
            decision_method=Method(name="unadjusted"),
        )
        assert tuple(dict.fromkeys(_scope(cell).view for cell in selected.cells)) == (
            "run",
            "asof",
        )
        assert len(selected.run.metadata.captures) == 1
        from increment.semantics.models import MeanMetric

        unregistered = MeanMetric(
            name="unregistered_revenue",
            entity="user_id",
            fact="purchase",
            aggregation="sum",
            window_days=1,
        )
        day_scopes = (
            Scope(view="daily", grain="daily"),
            Scope(view="asof", grain="asof"),
        )
        day_experiment = replace(experiment, scopes=day_scopes)
        day_values = run(
            day_experiment,
            resolver,
            metrics=[unregistered],
            decision_method=Method(name="unadjusted"),
        )
        assert {_scope(cell).view for cell in day_values.cells} == {"daily", "asof"}
        assert {cell.identity.key.metric for cell in day_values.cells} == {"unregistered_revenue"}
    finally:
        con.disconnect()


def test_selected_late_ad_hoc_metric_keeps_family_horizon_in_both_modes(tmp_path):
    from increment.semantics.models import MeanMetric

    rows = ds.event_rows()
    rows.append(
        {
            **next(row for row in rows if row["event"] == "purchase"),
            "event": "ad_hoc_late",
            "event_at": datetime(2025, 2, 20),
            "revenue": 100.0,
            "sess": None,
        }
    )
    definitions = ds.definitions_dict(
        plan=AnalysisPlan(primary="revenue", secondaries=["purchase_rate"]), breakout=True
    )
    definitions["experiments"][0].pop("end")
    definitions["fact_sources"][0]["facts"].append({"name": "ad_hoc_late", "column": "revenue"})
    definitions_path = tmp_path / "running-ad-hoc.yaml"
    definitions_path.write_text(json.dumps(definitions), encoding="utf-8")
    day_scopes = (Scope(view="daily", grain="daily"), Scope(view="asof", grain="asof"))
    late_metric = MeanMetric(
        name="late_ad_hoc",
        entity="user_id",
        fact="ad_hoc_late",
        aggregation="avg_calendar_day",
    )

    for coherence in ("captured", "live"):
        con = ds.duckdb_connection(rows)
        try:
            experiment, resolver = from_definitions(
                definitions_path, cast(SQLBackend, con), experiment_name="exp", store="none"
            )
            experiment = replace(experiment, scopes=day_scopes)
            cast(Any, resolver.resolve(experiment.source_refs)).panel_sql()
            values = run(
                experiment,
                resolver,
                metrics=[late_metric],
                decision_method=Method(name="unadjusted"),
                coherence=coherence,
            )
            for view in ("daily", "asof"):
                dates = [cell.identity.key.ds for cell in values.cells if _scope(cell).view == view]
                assert max(dates) == datetime(2025, 2, 20).date()
        finally:
            con.disconnect()


def _shared_panel_frame() -> Any:
    import pandas as pd

    return pd.DataFrame(
        {
            "unit": [f"u{index}" for index in range(40)],
            "arm": ["control", "treatment"] * 20,
            "day": [datetime(2025, 1, 1).date()] * 40,
            "country": ["US" if (index // 2) % 2 == 0 else "CA" for index in range(40)],
            "value": [float(index % 7) for index in range(40)],
        }
    )


def test_unit_panel_declared_breakout_runs_with_combined_total_view():
    from increment.builders import from_frame

    frame = _shared_panel_frame()
    experiment, resolver = from_frame(
        frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        control="control",
        metrics={"value": "mean"},
        breakouts=["country"],
    )
    scopes = (
        Scope(view="run", grain="total"),
        Scope(
            view="breakout",
            grain="total",
            segment=(BreakoutIdentity("country", None),),
        ),
    )
    experiment = replace(experiment, scopes=scopes)
    result = run(experiment, resolver)
    assert {_scope(cell).view for cell in result.cells} == {"run", "breakout"}
    assert {
        cell.identity.key.dimension_value
        for cell in result.cells
        if _scope(cell).view == "breakout"
    } == {"US", "CA"}


def _encouragement_panel_analysis(frame: Any) -> tuple[Any, Any]:
    from increment.builders import from_frame
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    encouraged_frame = frame.assign(
        help_click=[
            0 if arm == "control" else int(index % 3 == 0) for index, arm in enumerate(frame["arm"])
        ]
    )
    return from_frame(
        encouraged_frame,
        shape="panel",
        unit="unit",
        group="arm",
        date="day",
        design=Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="help_click"),
            exclusion_restriction=ExclusionRestriction(justification="fixture"),
        ),
        metrics={"value": "mean"},
        breakouts=["country"],
    )


def test_dimensioned_empty_encouragement_values_keep_permissions():
    experiment, resolver = _encouragement_panel_analysis(_shared_panel_frame())
    run_scope = Scope(view="run", grain="total")
    daily_scope = Scope(view="daily", grain="daily")
    asof_scope = Scope(view="asof", grain="asof")
    experiment = replace(
        experiment,
        scopes=(run_scope, daily_scope, asof_scope),
        operations={
            run_scope.hash(): "run",
            daily_scope.hash(): "run_daily",
            asof_scope.hash(): "run_asof",
        },
    )
    empty_values = run(experiment, resolver, metrics=[])
    assert empty_values.cells == ()

    dimensioned_scopes = (
        Scope(
            view="daily",
            grain="daily",
            segment=(BreakoutIdentity("country", None),),
        ),
        Scope(
            view="asof",
            grain="asof",
            segment=(BreakoutIdentity("country", None),),
        ),
    )
    dimensioned_experiment = replace(
        experiment,
        scopes=dimensioned_scopes,
        operations={
            dimensioned_scopes[0].hash(): "run_daily",
            dimensioned_scopes[1].hash(): "run_asof",
        },
    )
    empty_dimensioned = run(dimensioned_experiment, resolver, metrics=[])
    assert empty_dimensioned.cells == ()


def test_combined_encouragement_views_preserve_compliance_rows():
    experiment, resolver = _encouragement_panel_analysis(_shared_panel_frame())
    run_scope = Scope(view="run", grain="total")
    daily_scope = Scope(view="daily", grain="daily")
    asof_scope = Scope(view="asof", grain="asof")
    experiment = replace(
        experiment,
        scopes=(run_scope, daily_scope, asof_scope),
        operations={
            run_scope.hash(): "run",
            daily_scope.hash(): "run_daily",
            asof_scope.hash(): "run_asof",
        },
    )
    combined_values = run(experiment, resolver, metrics=["value"])
    assert {_scope(cell).view for cell in combined_values.cells} == {
        "run",
        "daily",
        "asof",
    }
    assert any(cell.identity.key.estimand == "compliance" for cell in combined_values.cells)


def test_invalid_view_selection_and_late_capability_refuse_without_capture(tmp_path):
    con = ds.duckdb_connection()
    experiment, resolver = _builder(con, tmp_path)
    scopes = (
        Scope(view="run", grain="total"),
        Scope(view="daily", grain="daily"),
        Scope(
            view="breakout",
            grain="total",
            segment=(BreakoutIdentity("store", None),),
        ),
    )
    experiment = replace(experiment, scopes=scopes)

    try:
        for views in ((), (scopes[0], scopes[0]), (Scope(view="asof", grain="asof"),)):
            with pytest.raises(CodedError) as error:
                run(experiment, resolver, views=views)
            assert error.value.code == "kernel.run.execution_unavailable"
        empty = run(experiment, resolver, metrics=[])
        assert empty.cells == ()
        invalid = replace(
            experiment, scopes=(*scopes, Scope(view="asof", grain="asof", through_day="2025-01-11"))
        )
        with pytest.raises(CodedError) as error:
            run(invalid, resolver)
        assert error.value.code == "kernel.run.execution_unavailable"
        with pytest.raises(CodedError) as error:
            run(experiment, resolver, population="triggered")
        assert error.value.code == "kernel.request.population_scope_only"
        with pytest.raises(CodedError) as error:
            run(replace(experiment, request_options={"unexpected": True}), resolver)
        assert error.value.code == "kernel.run.execution_unavailable"
        invalid_breakout = replace(
            experiment,
            scopes=(
                Scope(view="run", grain="total"),
                Scope(
                    view="breakout",
                    grain="total",
                    segment=(BreakoutIdentity("store", None),),
                    population="triggered",
                ),
            ),
        )
        with pytest.raises(CodedError) as error:
            run(invalid_breakout, resolver)
        assert error.value.code == "facade.analysis.trigger_unsupported"
        trigger_con = ds.duckdb_connection()
        trigger_experiment, trigger_resolver = _builder(
            trigger_con, tmp_path, triggered=True, trigger_evidence=False
        )
        trigger_experiment = replace(
            trigger_experiment,
            scopes=(
                Scope(
                    view="breakout",
                    grain="total",
                    segment=(BreakoutIdentity("store", None),),
                    population="assigned",
                ),
                Scope(
                    view="breakout",
                    grain="total",
                    segment=(BreakoutIdentity("store", None),),
                    population="triggered",
                ),
            ),
        )
        trigger_con.disconnect()
        with pytest.raises(CodedError) as error:
            run(trigger_experiment, trigger_resolver)
        assert error.value.code == "source.native.trigger_evidence_required"
    finally:
        con.disconnect()


# Raw confidence sets require both populations and a real persistence refusal.
@pytest.mark.slow
def test_live_confidence_rows_align_across_scopes_without_raw_persistence(tmp_path):
    from increment import render
    from increment.store import ParquetResultsStore
    from tests.test_kernel_review_regressions import _frame_read_calls

    con = ds.duckdb_connection(ds.event_rows(n_per_arm=40))
    experiment, resolver = _builder(con, tmp_path, triggered=True, winsorized=True)
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="run", grain="total", population="triggered"),
    )
    experiment = replace(experiment, scopes=scopes)
    result = run(experiment, resolver, decision_method=Method(name="unadjusted"))
    assert len(result.live_rows) == len(result.cells) == len(result.effects)
    assert any(row is None for row in result.live_rows)
    for cell, row, effect in zip(result.cells, result.live_rows, result.effects, strict=True):
        if cell.identity.key.metric == "revenue":
            assert row is not None and effect is not None
            assert row.confidence_set is not None
            assert effect.point == pytest.approx(row.confidence_set.point, rel=1e-9)
            assert effect.qualification == row.confidence_set.qualification
        else:
            assert row is None
    assert render(result, "brief")
    assert render(result, "full")
    with pytest.raises(CodedError) as error:
        pickle.dumps(result)
    assert error.value.code == "wire.result.raw_outcomes_unavailable"

    with _frame_read_calls() as reads, pytest.raises(CodedError) as error:
        run(experiment, resolver, results_store=ParquetResultsStore(tmp_path / "raw"))
    assert error.value.code == "wire.result.raw_outcomes_unavailable"
    assert reads == []
    assert not tuple((tmp_path / "raw").rglob("*.parquet"))
    con.disconnect()


# All three views must agree with independent runs across both populations.
@pytest.mark.slow
def test_assigned_and_triggered_views_share_one_population_union(tmp_path):
    con = ds.duckdb_connection()
    experiment, resolver = _builder(con, tmp_path, triggered=True)
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="run", grain="total", population="triggered"),
        Scope(view="asof", grain="asof", population="triggered"),
    )
    experiment = replace(experiment, scopes=scopes)
    method = Method(name="unadjusted")
    try:
        separate = [
            run(replace(experiment, scopes=(scope,)), resolver, decision_method=method)
            for scope in scopes
        ]
        actual = run(experiment, resolver, decision_method=method)
        assert len(actual.run.metadata.captures) == 1
        for scope, expected in zip(scopes, separate, strict=True):
            _same_cells(
                tuple(
                    cell
                    for cell in actual.cells
                    if _scope(cell).view == scope.view
                    and _scope(cell).population == scope.population
                ),
                expected.cells,
            )
        assert {cell.identity.key.analysis_population for cell in actual.cells} == {
            "assigned",
            "triggered",
        }
    finally:
        con.disconnect()


def test_two_population_run_advises_each_unused_covariate_once(tmp_path):
    """The unused-covariate advisory names a metric's configuration, not a population, so
    reading both populations in one run advises exactly what reading one does."""
    import warnings

    con = ds.duckdb_connection()
    experiment, resolver = _builder(con, tmp_path, triggered=True)
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="run", grain="total", population="triggered"),
    )

    def advisories(selected):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run(replace(experiment, scopes=selected), resolver)
        return [
            dict(item.message.context)
            for item in caught
            if isinstance(item.message, IncrementWarning)
            and item.message.code == "analysis.cuped.covariate_unused"
        ]

    try:
        single = advisories(scopes[:1])
        assert len(single) == 1
        assert advisories(scopes) == single
    finally:
        resolver.close()
        con.disconnect()


# Distinct metric sources must remain pinned through separate and shared runs.
@pytest.mark.slow
def test_assigned_triggered_shared_run_uses_declared_source_for_same_name_variant(tmp_path):
    from increment.semantics.models import MeanMetric

    con = ds.duckdb_connection()
    cast(Any, con).raw_sql("CREATE TABLE alternate_events AS SELECT * FROM events")
    cast(Any, con).raw_sql("UPDATE alternate_events SET revenue = revenue + 1000")
    experiment, resolver = _builder(con, tmp_path, triggered=True, separate_metric_source=True)
    declared = MeanMetric.model_validate(dict(experiment.metrics["revenue"]))
    variant = declared.model_copy(update={"fact": "alternate_purchase"})
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="run", grain="total", population="triggered"),
    )
    experiment = replace(experiment, scopes=scopes)
    method = Method(name="unadjusted")
    try:
        separate = [
            run(
                replace(experiment, scopes=(scope,)),
                resolver,
                metrics=["revenue"],
                decision_method=method,
            )
            for scope in scopes
        ]
        actual = run(experiment, resolver, metrics=[variant], decision_method=method)
        assert len(actual.run.metadata.captures) == 1
        for scope, expected in zip(scopes, separate, strict=True):
            actual_scope = tuple(cell for cell in actual.cells if cell.identity.scope == scope)
            assert tuple(cell.identity.key for cell in actual_scope) == tuple(
                cell.identity.key for cell in expected.cells
            )
            assert nested_close(
                [cell.to_payload()["row_payload"] for cell in actual_scope],
                [cell.to_payload()["row_payload"] for cell in expected.cells],
            )
    finally:
        con.disconnect()


def test_whole_window_and_daily_capture_same_name_metric_sources(tmp_path):
    from increment.semantics.models import MeanMetric

    con = ds.duckdb_connection()
    cast(Any, con).raw_sql("CREATE TABLE alternate_events AS SELECT * FROM events")
    cast(Any, con).raw_sql("UPDATE alternate_events SET revenue = revenue + 1000")
    cast(Any, con).raw_sql(
        "INSERT INTO alternate_events SELECT * REPLACE "
        "(TIMESTAMP '2025-02-20 00:00:00' AS event_at, revenue + 1000 AS revenue) "
        "FROM events WHERE event = 'purchase' LIMIT 1"
    )
    experiment, resolver = _builder(
        con,
        tmp_path,
        triggered=False,
        separate_metric_source=True,
        running_revenue=True,
    )
    declared = MeanMetric.model_validate(dict(experiment.metrics["revenue"]))
    variant = declared.model_copy(update={"fact": "alternate_purchase", "window_days": None})
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="daily", grain="daily", population="assigned"),
    )
    experiment = replace(experiment, scopes=scopes)
    method = Method(name="unadjusted")
    try:
        separate = (
            run(
                replace(experiment, scopes=(scopes[0],)),
                resolver,
                metrics=[variant],
                decision_method=method,
            ),
            run(
                replace(experiment, scopes=(scopes[1],)),
                resolver,
                metrics=[variant],
                decision_method=method,
            ),
        )
        assert (
            max(cell.identity.key.ds for cell in separate[1].cells) == datetime(2025, 2, 20).date()
        )
        actual = run(experiment, resolver, metrics=[variant], decision_method=method)
        assert len(actual.run.metadata.captures) == 1
        for scope, expected in zip(scopes, separate, strict=True):
            actual_scope = tuple(cell for cell in actual.cells if _scope(cell).view == scope.view)
            assert tuple(cell.identity.key for cell in actual_scope) == tuple(
                cell.identity.key for cell in expected.cells
            )
            assert nested_close(
                [cell.to_payload()["row_payload"] for cell in actual_scope],
                [cell.to_payload()["row_payload"] for cell in expected.cells],
            )
    finally:
        con.disconnect()


def test_breakout_capture_uses_normalized_metric_for_both_populations(tmp_path):
    from increment.semantics.models import MeanMetric

    con = ds.duckdb_connection()
    cast(Any, con).raw_sql("CREATE TABLE alternate_events AS SELECT * FROM events")
    cast(Any, con).raw_sql("UPDATE alternate_events SET revenue = revenue + 1000")
    experiment, resolver = _builder(
        con,
        tmp_path,
        triggered=True,
        separate_metric_source=True,
        include_breakouts=True,
    )
    variant = MeanMetric.model_validate(dict(experiment.metrics["revenue"])).model_copy(
        update={"fact": "alternate_purchase"}
    )
    breakout = BreakoutIdentity("store", None)
    scopes = (
        Scope(view="breakout", grain="total", population="assigned", segment=(breakout,)),
        Scope(view="breakout", grain="total", population="triggered", segment=(breakout,)),
    )
    experiment = replace(experiment, scopes=scopes)
    try:
        separate = tuple(
            run(replace(experiment, scopes=(scope,)), resolver, metrics=[variant])
            for scope in scopes
        )
        actual = run(experiment, resolver, metrics=[variant])
        assert len(actual.run.metadata.captures) == 1
        for scope, expected in zip(scopes, separate, strict=True):
            actual_scope = tuple(
                cell
                for cell in actual.cells
                if _scope(cell).view == "breakout"
                and cell.identity.key.analysis_population == scope.population
            )
            assert tuple(cell.identity.key for cell in actual_scope) == tuple(
                cell.identity.key for cell in expected.cells
            )
            assert nested_close(
                [cell.to_payload()["row_payload"] for cell in actual_scope],
                [cell.to_payload()["row_payload"] for cell in expected.cells],
            )
    finally:
        con.disconnect()


# All six ingress dispositions must retain shared-view results and refusal codes.
@pytest.mark.slow
@pytest.mark.filterwarnings(
    "ignore:metric 'm' has window_days=None:increment.errors.IncrementWarning"
)
def test_shared_capture_capabilities_follow_existing_six_ingress_dispositions():
    from tests.capability_consumers_auxiliary import assert_refusal, assert_support
    from tests.parity_harness import matrix, matrix_cases
    from tests.parity_harness.cases import _close_parity_binding

    baseline = matrix.Cell("mean", "run", "none", "utc", "error")
    temporal = replace(baseline, view="daily")
    dispositions = matrix.classify(temporal).outcomes("values")
    outcomes = matrix.classify(baseline).outcomes("rows")
    builders = {
        **matrix_cases.build_case(baseline, "rows", outcomes).build,
        **matrix_cases.build_switchback_case(baseline, "rows", outcomes).build,
    }
    for ingress, build in builders.items():
        bound = build()
        try:
            experiment, resolver = bound
            scopes = (Scope(view="run", grain="total"), Scope(view="daily", grain="daily"))
            experiment = replace(experiment, scopes=scopes)
            disposition = dispositions[ingress]

            if isinstance(disposition, matrix.Runs):
                separate = [run(replace(experiment, scopes=(scope,)), resolver) for scope in scopes]
                result = run(experiment, resolver)
                for scope, expected in zip(scopes, separate, strict=True):
                    _same_cells(
                        tuple(cell for cell in result.cells if _scope(cell).view == scope.view),
                        expected.cells,
                    )
                assert len(result.run.metadata.captures) == 1
                assert_support("shared_capture", f"multiview_{ingress}", result)
            else:
                assert isinstance(disposition, matrix.Refuses)
                with pytest.raises(CodedError) as error:
                    run(experiment, resolver)
                assert error.value.code == disposition.code
                assert_refusal("shared_capture", f"multiview_{ingress}", error.value, stage="read")

            if ingress != "from_definitions":
                with pytest.raises(CodedError) as error:
                    run(
                        experiment,
                        resolver,
                        views=(scopes[0],),
                        exploratory_metrics=["additional"],
                    )
                assert_refusal(
                    "shared_capture",
                    f"exploratory_{ingress}",
                    error.value,
                    stage="read",
                )
        finally:
            _close_parity_binding(bound)


def test_switchback_metric_selection_scopes_share_contrast_evidence():
    from tests.parity_harness import matrix, matrix_cases
    from tests.parity_harness.cases import _close_parity_binding

    cell = matrix.Cell("mean", "run", "none", "utc", "error")
    case = matrix_cases.build_switchback_case(cell, "rows", matrix.classify(cell).outcomes("rows"))
    bound = case.build["from_switchback_panel"]()
    try:
        experiment, resolver = bound
        scope = Scope(
            view="switchback",
            grain="total",
            metrics=tuple(dict(cast(Any, item)) for item in experiment.metrics.values()),
        )
        scopes = (*experiment.scopes, scope)
        experiment = replace(experiment, scopes=scopes)
        separate = [run(replace(experiment, scopes=(item,)), resolver) for item in scopes]
        result = run(experiment, resolver)
        assert len(result.run.metadata.captures) == 1
        for item, expected in zip(scopes, separate, strict=True):
            _same_cells(
                tuple(cell for cell in result.cells if cell.identity.scope == item),
                expected.cells,
            )
    finally:
        _close_parity_binding(bound)
