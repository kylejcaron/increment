from __future__ import annotations

import json
import pickle
from dataclasses import asdict, replace
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from increment import run
from increment.builders import from_definitions
from increment.declarations import Method
from increment.errors import CodedError
from increment.evidence import BreakoutIdentity
from increment.kernel.contracts import Scope
from increment.semantics.models import AnalysisPlan, Definitions
from tests.analysis_factory import make_analysis
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
    return make_analysis(con, Definitions.model_validate(definitions), experiment="exp")


def _builder(con, tmp_path, *, triggered=False, winsorized=False):
    from increment.artifacts import SourceSnapshotEvidence

    definitions = ds.definitions_dict(
        plan=AnalysisPlan(primary="revenue", secondaries=["purchase_rate"]), breakout=True
    )
    kwargs: dict[str, Any] = {}
    if winsorized:
        definitions["metrics"][0]["winsorization"] = {"upper_percentile": 0.9}
    if triggered:
        definitions["fact_sources"][0]["facts"].append({"name": "trigger_signal", "column": None})
        definitions["exposures"].append({"name": "trigger", "fact": "trigger_signal"})
        con.raw_sql(
            "INSERT INTO events SELECT * REPLACE (event_at + INTERVAL 1 HOUR AS event_at, "
            "'trigger_signal' AS event) FROM events WHERE event = 'exposure'"
        )
        definitions["experiments"][0]["trigger"] = "trigger"
        kwargs["source_snapshot_evidence"] = SourceSnapshotEvidence(
            observation_cutoff_ts=datetime(2025, 2, 16, tzinfo=UTC),
            complete_through_by_feed={"events": datetime(2025, 2, 16, tzinfo=UTC)},
        )
    path = tmp_path / "definitions.yaml"
    path.write_text(json.dumps(definitions), encoding="utf-8")
    return from_definitions(path, con, experiment_name="exp", store="none", **kwargs)


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
    analysis = _analysis(con, running=True)
    try:
        declared = analysis.run(metrics=["revenue"], decision_method=Method(name="unadjusted"))
        all_declared = analysis.run(decision_method=Method(name="unadjusted"))
        full_revenue = next(row for row in all_declared if row.metric == "revenue")
        narrowed_revenue = next(row for row in declared if row.metric == "revenue")
        assert nested_close(
            narrowed_revenue.lift.model_dump(),
            full_revenue.lift.model_dump(),
        )
        panel = cast(Any, con).sql(analysis.panel_sql()["revenue"]).execute()
        observed_day_counts = panel.groupby("unit_id").size().to_dict()
        expected_day_counts = {
            row["user_id"]: (datetime(2025, 1, 25).date() - row["event_at"].date()).days + 1
            for row in rows
            if row["event"] == "exposure"
        }
        assert observed_day_counts == expected_day_counts
        live_analysis = _analysis(con, running=True)
        try:
            live_narrowed = live_analysis.run(
                metrics=["revenue"],
                coherence="live",
                decision_method=Method(name="unadjusted"),
            )
            live_all = live_analysis.run(
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
            live_analysis.close()
        extra = analysis.run(
            metrics=[],
            exploratory_metrics=["rps", "revenue_cuped"],
            decision_method=Method(name="unadjusted"),
        )
        actual = analysis.run(
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
        analysis.close()
        con.disconnect()


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
                analysis_options={"decision_method": Method(name="unadjusted")},
            )
            for scope in scopes
        ]
        actual = run(
            experiment, resolver, analysis_options={"decision_method": Method(name="unadjusted")}
        )
        for scope, expected in zip(scopes, separate, strict=True):
            cells = tuple(cell for cell in actual.cells if cell.identity.scope.view == scope.view)
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
        monitoring = {
            cell.identity.hash() for cell in actual.cells if cell.identity.scope.view != "run"
        }
        assert all(
            verdict.reason_codes == ("decision.not_evaluated.monitoring_view",)
            for verdict in actual.decisions.cells
            if verdict.identity_hash in monitoring
        )
        assert actual.decisions.program == separate[0].decisions.program
        assert all(family in actual.decisions.families for family in separate[0].decisions.families)
        assert len(actual.run.metadata.captures) == 1
        envelope = AnalysisEnvelope(actual.cells, actual.run.metadata)
        restored = AnalysisEnvelope.from_payload(envelope.to_payload())
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
            analysis_options={"decision_method": Method(name="unadjusted")},
        )
        assert tuple(dict.fromkeys(cell.identity.scope.view for cell in selected.cells)) == (
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
            analysis_options={
                "metrics": [unregistered],
                "decision_method": Method(name="unadjusted"),
            },
        )
        assert {cell.identity.scope.view for cell in day_values.cells} == {"daily", "asof"}
        assert {cell.identity.key.metric for cell in day_values.cells} == {"unregistered_revenue"}
        running_definitions = ds.definitions_dict(
            plan=AnalysisPlan(primary="revenue", secondaries=["purchase_rate"]), breakout=True
        )
        running_definitions["experiments"][0].pop("end")
        running_definitions["fact_sources"][0]["facts"].append(
            {"name": "ad_hoc_late", "column": "revenue"}
        )
        running_path = tmp_path / "running-ad-hoc.yaml"
        running_path.write_text(json.dumps(running_definitions), encoding="utf-8")
        running_experiment, running_resolver = from_definitions(
            running_path, con, experiment_name="exp", store="none"
        )
        running_experiment = replace(running_experiment, scopes=day_scopes)
        late_metric = MeanMetric(
            name="late_ad_hoc",
            entity="user_id",
            fact="ad_hoc_late",
            aggregation="avg_calendar_day",
        )
        late_values = run(
            running_experiment,
            running_resolver,
            analysis_options={
                "metrics": [late_metric],
                "decision_method": Method(name="unadjusted"),
            },
        )
        for view in ("daily", "asof"):
            dates = [
                cell.identity.key.ds
                for cell in late_values.cells
                if cell.identity.scope.view == view
            ]
            assert max(dates) == datetime(2025, 2, 20).date()
    finally:
        con.disconnect()


def test_unit_panel_declared_breakout_runs_with_combined_total_view():
    import pandas as pd

    from increment.builders import from_frame

    frame = pd.DataFrame(
        {
            "unit": [f"u{index}" for index in range(40)],
            "arm": ["control", "treatment"] * 20,
            "day": [datetime(2025, 1, 1).date()] * 40,
            "country": ["US" if (index // 2) % 2 == 0 else "CA" for index in range(40)],
            "value": [float(index % 7) for index in range(40)],
        }
    )
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
    assert {cell.identity.scope.view for cell in result.cells} == {"run", "breakout"}
    assert {
        cell.identity.key.dimension_value
        for cell in result.cells
        if cell.identity.scope.view == "breakout"
    } == {"US", "CA"}
    from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

    encouraged_frame = frame.assign(
        help_click=[
            0 if arm == "control" else int(index % 3 == 0) for index, arm in enumerate(frame["arm"])
        ]
    )
    encouraged, encouraged_resolver = from_frame(
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
    run_scope = Scope(view="run", grain="total")
    daily_scope = Scope(view="daily", grain="daily")
    asof_scope = Scope(view="asof", grain="asof")
    encouraged = replace(
        encouraged,
        scopes=(run_scope, daily_scope, asof_scope),
        operations={
            run_scope.hash(): "run",
            daily_scope.hash(): "run_daily",
            asof_scope.hash(): "run_asof",
        },
    )
    empty_values = run(encouraged, encouraged_resolver, analysis_options={"metrics": []})
    assert empty_values.cells == ()
    combined_values = run(encouraged, encouraged_resolver, analysis_options={"metrics": ["value"]})
    assert {cell.identity.scope.view for cell in combined_values.cells} == {
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
        empty = run(experiment, resolver, analysis_options={"metrics": []})
        assert empty.cells == ()
        invalid = replace(
            experiment, scopes=(*scopes, Scope(view="asof", grain="asof", through_day="2025-01-11"))
        )
        with pytest.raises(CodedError) as error:
            run(invalid, resolver)
        assert error.value.code == "kernel.run.execution_unavailable"
        with pytest.raises(CodedError) as error:
            run(experiment, resolver, analysis_options={"population": "triggered"})
        assert error.value.code == "facade.analysis.trigger_unsupported"
        with pytest.raises(CodedError) as error:
            run(experiment, resolver, analysis_options={"unexpected": True})
        assert error.value.code == "kernel.run.execution_unavailable"
        with pytest.raises(CodedError) as error:
            run(experiment, resolver, analysis_options={"store": object()})
        assert error.value.code == "kernel.run.invalid_input"
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
    finally:
        con.disconnect()


def test_live_confidence_rows_align_across_scopes_without_raw_persistence(tmp_path):
    from increment import render
    from increment.store import ParquetResultsStore

    con = ds.duckdb_connection(ds.event_rows(n_per_arm=40))
    experiment, resolver = _builder(con, tmp_path, triggered=True, winsorized=True)
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="run", grain="total", population="triggered"),
    )
    experiment = replace(experiment, scopes=scopes)
    result = run(
        experiment, resolver, analysis_options={"decision_method": Method(name="unadjusted")}
    )
    assert len(result.live_rows) == len(result.cells) == len(result.effects)
    assert any(row is None for row in result.live_rows)
    for cell, row, effect in zip(result.cells, result.live_rows, result.effects, strict=True):
        if cell.identity.key.metric == "revenue":
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

    with pytest.raises(CodedError) as error:
        run(experiment, resolver, results_store=ParquetResultsStore(tmp_path / "raw"))
    assert error.value.code == "kernel.run.invalid_input"
    con.disconnect()


def test_assigned_and_triggered_views_share_one_population_union(tmp_path):
    con = ds.duckdb_connection()
    experiment, resolver = _builder(con, tmp_path, triggered=True)
    scopes = (
        Scope(view="run", grain="total", population="assigned"),
        Scope(view="run", grain="total", population="triggered"),
        Scope(view="asof", grain="asof", population="triggered"),
    )
    experiment = replace(experiment, scopes=scopes)
    options = {"decision_method": Method(name="unadjusted")}
    try:
        separate = [
            run(replace(experiment, scopes=(scope,)), resolver, analysis_options=options)
            for scope in scopes
        ]
        actual = run(experiment, resolver, analysis_options=options)
        assert len(actual.run.metadata.captures) == 1
        for scope, expected in zip(scopes, separate, strict=True):
            _same_cells(
                tuple(
                    cell
                    for cell in actual.cells
                    if cell.identity.scope.view == scope.view
                    and cell.identity.scope.population == scope.population
                ),
                expected.cells,
            )
        assert {cell.identity.key.analysis_population for cell in actual.cells} == {
            "assigned",
            "triggered",
        }
    finally:
        con.disconnect()


@pytest.mark.filterwarnings(
    "ignore:metric 'm' has window_days=None:increment.errors.IncrementWarning"
)
def test_shared_capture_capabilities_follow_existing_six_ingress_dispositions():
    from tests.analysis_factory import facade_binding
    from tests.capability_consumers_auxiliary import assert_refusal, assert_support
    from tests.parity_harness import matrix, matrix_cases
    from tests.parity_harness.cases import _close_parity_analysis

    baseline = matrix.Cell("mean", "run", "none", "utc", "error")
    temporal = replace(baseline, view="daily")
    dispositions = matrix.classify(temporal).outcomes("values")
    outcomes = matrix.classify(baseline).outcomes("rows")
    builders = {
        **matrix_cases.build_case(baseline, "rows", outcomes).build,
        **matrix_cases.build_switchback_case(baseline, "rows", outcomes).build,
    }
    for ingress, build in builders.items():
        analysis = build()
        try:
            experiment, resolver = facade_binding(analysis)
            scopes = (Scope(view="run", grain="total"), Scope(view="daily", grain="daily"))
            experiment = replace(experiment, scopes=scopes)
            disposition = dispositions[ingress]

            if isinstance(disposition, matrix.Runs):
                separate = [run(replace(experiment, scopes=(scope,)), resolver) for scope in scopes]
                result = run(experiment, resolver)
                for scope, expected in zip(scopes, separate, strict=True):
                    _same_cells(
                        tuple(
                            cell for cell in result.cells if cell.identity.scope.view == scope.view
                        ),
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
                        analysis_options={"exploratory_metrics": ["additional"]},
                    )
                assert_refusal(
                    "shared_capture",
                    f"exploratory_{ingress}",
                    error.value,
                    stage="read",
                )
        finally:
            _close_parity_analysis(analysis)


def test_switchback_metric_selection_scopes_share_contrast_evidence():
    from tests.analysis_factory import facade_binding
    from tests.parity_harness import matrix, matrix_cases
    from tests.parity_harness.cases import _close_parity_analysis

    cell = matrix.Cell("mean", "run", "none", "utc", "error")
    case = matrix_cases.build_switchback_case(cell, "rows", matrix.classify(cell).outcomes("rows"))
    analysis = case.build["from_switchback_panel"]()
    try:
        experiment, resolver = facade_binding(analysis)
        scope = Scope(
            view="switchback",
            grain="total",
            metrics=tuple(dict(item) for item in experiment.metrics.values()),
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
        _close_parity_analysis(analysis)
