from __future__ import annotations

from typing import Any, cast

import narwhals as nw

from increment.analysis import Analysis
from increment.estimation.results import LiftEstimate
from increment.semantics import load
from tests.analysis_factory import _moment_source
from tests.covariate_cases import categorical_defs_and_con


def _with_metric_covariates(path, binding: str, *, n_pre_periods: int = 0):
    text = path.read_text()
    text = text.replace(
        "    plan: {primary: revenue}",
        "    n_pre_periods: "
        + str(n_pre_periods)
        + "\n    plan: {primary: {metric: revenue, covariates: ["
        + binding
        + "]}}",
    )
    path.write_text(text)
    return path


def test_definition_rejects_covariate_kind_mismatch_before_acquisition(tmp_path):
    import pytest

    from increment.errors import DefinitionError

    defs_path, _connection, _units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        "{name: region_number, kind: continuous, property: region, source: events}",
    )

    with pytest.raises(DefinitionError) as raised:
        load(defs_path)

    assert raised.value.code == "definition.invalid"
    assert any(
        code == "definition.check_cuped.covariate_kind"
        for code, _message in cast(Any, raised.value.context["errors"])
    )


def test_native_definition_covariate_property_acquisition_preserves_category_and_null(tmp_path):
    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=("u3",))
    _with_metric_covariates(
        defs_path,
        "{name: pre_region, kind: categorical, property: region, source: events}",
    )
    source = _moment_source(Analysis.from_definitions("cat_test", defs_path, con))
    metric = source.context.metrics[0]
    table = nw.from_native(
        source.unit_frame(metric, covariates=["pre_region"]), eager_only=True
    ).to_arrow()
    rows = {row["unit_id"]: row["pre_region"] for row in table.to_pylist()}

    assert rows["u3"] is None
    assert rows["u0"] == units["u0"]["region"]
    assert set(rows) == set(units)


def test_native_measure_covariate_is_scoped_before_assignment(tmp_path):
    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    con.raw_sql(
        "INSERT INTO cat_events (user_id, ts, event, experiment_id, group_id, revenue, "
        "tenure, region) VALUES ('u0', TIMESTAMP '2025-07-20 10:00:00', 'purchase', "
        "NULL, NULL, 11.0, NULL, NULL)"
    )
    _with_metric_covariates(
        defs_path,
        "{name: pre_revenue, kind: continuous, measure: {fact: purchase, aggregation: sum}}",
        n_pre_periods=30,
    )
    source = _moment_source(Analysis.from_definitions("cat_test", defs_path, con))
    metric = source.context.metrics[0]
    table = nw.from_native(
        cast(Any, source).unit_frame(metric, covariates=["pre_revenue"], outcome_stage="raw"),
    ).to_arrow()
    rows = {row["unit_id"]: row["pre_revenue"] for row in table.to_pylist()}

    assert rows["u0"] == 11.0
    assert rows["u1"] is None


def test_raw_unit_frame_pins_separate_measure_fact_source(tmp_path):
    import datetime as dt

    import pyarrow as pa

    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    con.create_table(
        "history",
        obj=pa.Table.from_pylist(
            [
                {
                    "user_id": unit,
                    "ts": dt.datetime(2025, 7, 20, 10, 0),
                    "event": "history_value",
                    "value": 0.0 if i == 0 else 2.0,
                }
                for i, unit in enumerate(list(units)[:2])
            ]
        ),
    )
    text = defs_path.read_text().replace(
        "exposures:\n",
        "  - name: history\n"
        "    sql: SELECT * FROM history\n"
        "    timestamp_column: ts\n"
        "    entities: [user_id]\n"
        "    facts:\n"
        "      - {name: history_value, column: value}\n"
        "exposures:\n",
    )
    defs_path.write_text(text)
    _with_metric_covariates(
        defs_path,
        "{name: pre_value, kind: continuous, measure: {fact: history_value, aggregation: sum}}",
        n_pre_periods=30,
    )
    analysis = Analysis.from_definitions("cat_test", defs_path, con)
    source = _moment_source(analysis)
    metric = source.context.metrics[0]
    rows = (
        nw.from_native(
            cast(Any, source).unit_frame(metric, covariates=["pre_value"], outcome_stage="raw"),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )
    values = {row["unit_id"]: row["pre_value"] for row in rows}
    assert values["u0"] == 0.0
    assert values["u1"] == 2.0
    assert values["u2"] is None

    definitions = load(defs_path)
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = analysis.publish_unit_day_artifact(store)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    artifact = _moment_source(restored)
    artifact_values = {
        row["unit_id"]: row["pre_value"]
        for row in nw.from_native(
            artifact.unit_frame(metric, covariates=["pre_value"]), eager_only=True
        )
        .to_arrow()
        .to_pylist()
    }
    assert artifact_values == values


def test_occurrence_only_count_measure_acquires_native_and_artifact(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        "{name: pre_visits, kind: continuous, measure: {fact: exposure, aggregation: count}}",
        n_pre_periods=30,
    )
    definitions = load(defs_path)
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    analysis = Analysis.from_definitions("cat_test", defs_path, con)
    native = _moment_source(analysis)
    metric = native.context.metrics[0]
    native_values = {
        row["unit_id"]: row["pre_visits"]
        for row in nw.from_native(
            cast(Any, native).unit_frame(metric, covariates=["pre_visits"], outcome_stage="raw"),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    }
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = analysis.publish_unit_day_artifact(store)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    artifact = _moment_source(restored)
    artifact_values = {
        row["unit_id"]: row["pre_visits"]
        for row in nw.from_native(
            artifact.unit_frame(metric, covariates=["pre_visits"]), eager_only=True
        )
        .to_arrow()
        .to_pylist()
    }

    assert native_values == artifact_values == dict.fromkeys(units, 1.0)


def test_raw_unit_frame_pins_separate_property_source(tmp_path):
    defs_path, con, units = categorical_defs_and_con(tmp_path)
    text = defs_path.read_text().replace(
        "exposures:\n",
        "  - name: profiles\n"
        "    sql: SELECT * FROM cat_events\n"
        "    timestamp_column: ts\n"
        "    entities: [user_id]\n"
        "    facts:\n"
        "      - {name: profile_event, column: null}\n"
        "    properties:\n"
        "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
        "exposures:\n",
    )
    defs_path.write_text(text)
    _with_metric_covariates(
        defs_path,
        "{name: pre_region, kind: categorical, property: region, source: profiles}",
    )
    source = _moment_source(Analysis.from_definitions("cat_test", defs_path, con))
    metric = source.context.metrics[0]
    rows = (
        nw.from_native(
            cast(Any, source).unit_frame(metric, covariates=["pre_region"], outcome_stage="raw"),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )

    assert {row["unit_id"] for row in rows} == set(units)
    assert {row["unit_id"]: row["pre_region"] for row in rows} == {
        unit: truth["region"] for unit, truth in units.items()
    }


def test_multiple_primary_artifact_binding_resolves_property_alias(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    text = (
        defs_path.read_text()
        .replace(
            "metrics:\n",
            "metrics:\n"
            "  - name: revenue_other\n"
            "    type: mean\n"
            "    entity: user_id\n"
            "    fact: purchase\n"
            "    aggregation: sum\n"
            "    window_days: 7\n",
        )
        .replace(
            "    plan: {primary: revenue}",
            "    plan: {primary: [{metric: revenue, covariates: "
            "[{name: pre_region, kind: categorical, property: region, source: events}]}, "
            "revenue_other]}",
        )
    )
    defs_path.write_text(text)
    definitions = load(defs_path)
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    analysis = Analysis.from_definitions("cat_test", defs_path, con)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = analysis.publish_unit_day_artifact(store)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    source = _moment_source(restored)

    rows = (
        nw.from_native(
            source.unit_frame(source.context.metrics[0], covariates=["pre_region"]),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )

    assert rows


def test_artifact_deduplicates_omitted_and_explicit_resolved_source(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, *_ = covariate_defs_and_con(tmp_path, observational=True)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, covariates: "
            "[{name: pre_tenure, kind: continuous, property: tenure}]}}",
        )
    )
    definitions = load(defs_path)
    experiment = definitions.experiment("cov_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    analysis = Analysis.from_definitions("cov_test", defs_path, con)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = analysis.publish_unit_day_artifact(store)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    source = _moment_source(restored)
    rows = (
        nw.from_native(
            source.unit_frame(source.context.metrics[0], covariates=["pre_tenure"]),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )

    assert {row["unit_id"]: row["pre_tenure"] for row in rows} == tenure


def test_property_extensions_roundtrip_from_native_to_artifact(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=("u3",))
    _with_metric_covariates(
        defs_path,
        "{name: pre_region, kind: categorical, property: region, source: events}",
    )
    definitions = load(defs_path)
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    native_analysis = Analysis.from_definitions("cat_test", defs_path, con)
    reference = native_analysis.publish_unit_day_artifact(store)
    artifact_analysis = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    native = _moment_source(native_analysis)
    artifact = _moment_source(artifact_analysis)
    metric = native.context.metrics[0]

    native_rows = (
        nw.from_native(native.unit_frame(metric, covariates=["pre_region"]), eager_only=True)
        .to_arrow()
        .to_pylist()
    )
    artifact_rows = (
        nw.from_native(artifact.unit_frame(metric, covariates=["pre_region"]), eager_only=True)
        .to_arrow()
        .to_pylist()
    )

    assert artifact_rows == native_rows
    assert {row["unit_id"]: row["pre_region"] for row in artifact_rows} == {
        unit: truth["region"] for unit, truth in units.items()
    }


def test_artifact_keeps_same_property_from_distinct_sources(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, *_ = covariate_defs_and_con(tmp_path, observational=True)
    text = (
        defs_path.read_text()
        .replace(
            "exposures:\n",
            "  - name: events_b\n"
            "    sql: SELECT * FROM cov_events\n"
            "    timestamp_column: ts\n"
            "    entities: [user_id]\n"
            "    facts:\n"
            "      - {name: exposure_b, column: null}\n"
            "      - {name: purchase_b, column: revenue}\n"
            "    properties:\n"
            "      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}\n"
            "exposures:\n",
        )
        .replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, covariates: "
            "[{name: pre_tenure, kind: continuous, property: tenure, source: events_b}]}}",
        )
    )
    defs_path.write_text(text)
    definitions = load(defs_path)
    experiment = definitions.experiment("cov_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    analysis = Analysis.from_definitions("cov_test", defs_path, con)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = analysis.publish_unit_day_artifact(store)
    artifact_analysis = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    artifact = cast(Any, _moment_source(artifact_analysis))
    source_names = {
        extension.source_name
        for extension in artifact._manifest.extensions
        if extension.kind == "unit_covariate"
    }

    assert source_names == {"events", "events_b"}
    rows = (
        nw.from_native(
            artifact.unit_frame(artifact.context.metrics[0], covariates=["tenure", "pre_tenure"]),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )
    assert {row["unit_id"]: row["tenure"] for row in rows} == tenure
    assert {row["unit_id"]: row["pre_tenure"] for row in rows} == tenure


def test_measure_and_property_covariate_names_are_scoped_by_metric(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    con.raw_sql("ALTER TABLE cat_events ADD COLUMN prior DOUBLE")
    con.raw_sql("UPDATE cat_events SET prior = tenure")
    text = (
        defs_path.read_text()
        .replace(
            "metrics:\n",
            "metrics:\n"
            "  - name: revenue_other\n"
            "    type: mean\n"
            "    entity: user_id\n"
            "    fact: purchase\n"
            "    aggregation: sum\n"
            "    window_days: 7\n",
        )
        .replace(
            "      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}\n",
            "      - {name: tenure, column: tenure, dtype: float, as_of: pre_exposure}\n"
            "      - {name: prior, column: prior, dtype: float, as_of: pre_exposure}\n",
        )
        .replace(
            "    plan: {primary: revenue}",
            "    n_pre_periods: 30\n"
            "    plan: {primary: [{metric: revenue, covariates: "
            "[{name: prior, kind: continuous, measure: {fact: purchase, aggregation: sum}}]}, "
            "{metric: revenue_other, covariates: "
            "[{name: prior, kind: continuous, property: prior, source: events}]}]}",
        )
    )
    defs_path.write_text(text)
    definitions = load(defs_path)
    experiment = definitions.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    analysis = Analysis.from_definitions("cat_test", defs_path, con)
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = analysis.publish_unit_day_artifact(store)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    source = cast(Any, _moment_source(restored))

    measured = (
        nw.from_native(
            source.unit_frame(source.context.metrics[0], covariates=["prior"]),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )
    property_bound = (
        nw.from_native(
            source.unit_frame(source.context.metrics[1], covariates=["prior"]),
            eager_only=True,
        )
        .to_arrow()
        .to_pylist()
    )

    assert measured
    assert {row["unit_id"]: row["prior"] for row in property_bound} == {
        unit: truth["tenure"] for unit, truth in units.items()
    }


def test_cuped_linear_observational_route_refuses_before_acquisition(tmp_path):
    import pytest

    from increment.errors import UnsupportedRequestError

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        "{name: region, kind: categorical, property: region, source: events}",
    )
    text = defs_path.read_text().replace(
        "covariates: [",
        "decision_method: {name: linear, variance_reduction: cuped_linear}, covariates: [",
    )
    defs_path.write_text(text)
    analysis = Analysis.from_definitions("cat_test", defs_path, con)

    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run()

    assert raised.value.code == "estimation.engine.cuped_linear_observational"


def test_from_moments_refuses_cuped_linear_with_unit_source_route():
    import pytest

    from increment.errors import UnsupportedRequestError
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec

    spec = MetricSpec(
        name="revenue",
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
        covariates=(CupedCovariate(name="prior_spend", kind="continuous"),),
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        Analysis.from_moments([], metrics=[spec], control="control")

    assert raised.value.code == "facade.analysis_config.cuped_linear_moments"
    import copy
    import pickle

    for restored in (copy.deepcopy(raised.value), pickle.loads(pickle.dumps(raised.value))):
        assert restored.code == raised.value.code
        assert restored.context == raised.value.context


def test_switchback_refuses_cuped_linear_with_independent_unit_route():
    import pytest

    from increment.errors import CapabilityError
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.semantics.assignment import (
        IndependentBernoulliOrder,
        SwitchbackAssignment,
        SwitchbackWindow,
    )
    from increment.semantics.design import Randomized

    spec = MetricSpec(
        name="revenue",
        decision_method=Method(name="linear", variance_reduction="cuped_linear"),
        covariates=(CupedCovariate(name="prior_spend", kind="continuous"),),
    )
    with pytest.raises(CapabilityError) as raised:
        Analysis.from_switchback_panel(
            object(),  # ty: ignore[invalid-argument-type]
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics=[spec],
            identification=Randomized(
                control_group="control",
                allocation={"control": 0.5, "treatment": 0.5},
            ),
            assignment=SwitchbackAssignment(
                sequence=IndependentBernoulliOrder(probability_ct=0.5),
                window=SwitchbackWindow(washout_steps=0, observation_steps=1),
            ),
        )

    assert raised.value.code == "source.frame.switchback.metric"


def test_triggered_covariates_remain_anchored_to_assignment(tmp_path):
    import datetime as dt

    import ibis
    import pyarrow as pa

    from increment import SourceSnapshotEvidence
    from tests.test_analysis_trigger import _defs_yaml, _events, _with_preassignment_region

    defs_path = tmp_path / "triggered_covariates.yaml"
    defs = (
        _defs_yaml()
        .replace(
            "      - name: saw_surface\n        column: null",
            "      - name: saw_surface\n        column: null\n    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}",
        )
        .replace(
            "    plan:\n      secondaries: [revenue]",
            "    plan: {primary: {metric: revenue, covariates: "
            "[{name: region, kind: categorical, property: region, source: events}]}}",
        )
    )
    defs_path.write_text(defs)
    rows = _with_preassignment_region(_events(n_per_arm=20, trigger_rate=0.5)).to_pylist()
    for row in rows:
        if row["event"] == "saw_surface":
            row["ts"] = dt.datetime(2024, 1, 10)
    late_profiles = [
        {
            **row,
            "ts": dt.datetime(2024, 1, 8),
            "event": "profile",
            "region": "post_assignment",
        }
        for row in rows
        if row["event"] == "profile"
    ]
    rows.extend(late_profiles)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=pa.Table.from_pylist(rows))
    analysis = Analysis.from_definitions(
        "exp",
        defs_path,
        con,
        source_snapshot_evidence=SourceSnapshotEvidence(
            dt.datetime(2024, 1, 31, tzinfo=dt.UTC),
            {"events": dt.datetime(2024, 1, 31, tzinfo=dt.UTC)},
        ),
    )
    try:
        source = _moment_source(analysis)
        metric = source.context.metrics[0]
        rows_by_unit = {
            row["unit_id"]: row["region"]
            for row in nw.from_native(
                cast(Any, source).unit_frame(metric, covariates=["region"], population="triggered"),
                eager_only=True,
            )
            .to_arrow()
            .to_pylist()
        }
        assert rows_by_unit
        assert set(rows_by_unit.values()) <= {"north", "south"}
        assert all(
            region == ("north" if int(unit[1:]) % 2 else "south")
            for unit, region in rows_by_unit.items()
        )
    finally:
        analysis.close()
        con.disconnect()


def test_cuped_linear_definition_and_restored_artifact_match_frame_journey(tmp_path):
    import pyarrow as pa
    import pytest

    from increment import AnalysisPlan
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, groups, revenue = covariate_defs_and_con(tmp_path)
    text = defs_path.read_text().replace(
        "    plan: {secondaries: [revenue]}",
        "    plan: {primary: {metric: revenue, "
        "decision_method: {name: linear, variance_reduction: cuped_linear}, "
        "covariates: [{name: pre_tenure, kind: continuous, property: tenure, source: events}]}}",
    )
    defs_path.write_text(text)

    native = Analysis.from_definitions("cov_test", defs_path, con)
    definitions = load(defs_path)
    experiment = definitions.experiment("cov_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    reference = native.publish_unit_day_artifact(store)
    restored = Analysis.from_unit_day_artifact(store, reference, expected_context=context)
    assert native._src.context.configs[0].covariates[0].name == "pre_tenure"
    table = pa.Table.from_pylist(
        [
            {
                "unit": unit,
                "group": groups[unit],
                "revenue": revenue[unit],
                "pre_tenure": tenure[unit],
            }
            for unit in sorted(groups)
        ]
    )
    dataframe = Analysis.from_unit_summary(
        table,
        unit="unit",
        group="group",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method(name="linear", variance_reduction="cuped_linear"),
                covariates=(CupedCovariate(name="pre_tenure", kind="continuous"),),
            )
        ],
        plan=AnalysisPlan(primary="revenue"),
    )

    frame_result = cast(list[LiftEstimate], dataframe.run())
    native_result = cast(list[LiftEstimate], native.run())
    artifact_result = cast(list[LiftEstimate], restored.run())
    expected = next(row for row in frame_result if row.group_id == "treatment").require_lift()

    def source_signature(result):
        metadata = result.metadata
        assert metadata is not None
        scope = metadata.scope
        cell_signature = set()
        evidence_signature = {}
        for row in result:
            cell = next(
                key
                for key in scope.cells
                if key.metric == row.metric
                and key.group_id == row.group_id
                and key.method == row.method
                and key.method_role == row.method_role
                and key.analysis_population == row.analysis_population
            )
            assert row.source_snapshot_id in scope.by_source
            assert cell in scope.decision_cells
            source_scope = scope.by_source[row.source_snapshot_id]
            assert source_scope.source_snapshot_id == row.source_snapshot_id
            assert source_scope.decision_complete_by_population[row.analysis_population] is True
            record = next(
                item
                for item in metadata.cells
                if item.cell == cell and item.source_snapshot_id == row.source_snapshot_id
            )
            assert record.sampling is not None
            assert record.sampling.available == row.sampling_available
            assert record.sampling.evidence is not None
            cell_signature.add(
                (
                    cell.kind,
                    cell.metric,
                    cell.group_id,
                    cell.method,
                    cell.method_role,
                    cell.estimand,
                    cell.analysis_population,
                )
            )
            evidence = record.sampling.evidence
            evidence_signature[(cell.metric, cell.group_id, cell.method)] = (
                record.sampling.available,
                evidence.p_value,
                evidence.reference,
            )
        scope_signature = (
            frozenset(cell_signature),
            tuple(scope.populations),
            tuple(sorted((key, value) for key, value in scope.by_source.items())),
        )
        return scope_signature, evidence_signature

    expected_scope, expected_evidence = source_signature(frame_result)
    for result in (native_result, artifact_result):
        actual = next(row for row in result if row.group_id == "treatment").require_lift()
        assert actual.value == pytest.approx(expected.value, rel=1e-10, abs=1e-10)
        assert actual.lb == pytest.approx(expected.lb, rel=1e-10, abs=1e-10)
        assert actual.ub == pytest.approx(expected.ub, rel=1e-10, abs=1e-10)
        scope, evidence = source_signature(result)
        assert scope[:2] == expected_scope[:2]
        assert evidence == expected_evidence


def test_cuped_evidence_digest_accepts_infinities_without_rewriting_values():
    import polars as pl

    from increment.readouts._metric_rows import _frame_digest_and_arms

    frame = pl.DataFrame({"group_id": ["C", "T"], "covariate": [float("inf"), float("-inf")]})
    first = _frame_digest_and_arms(frame)
    second = _frame_digest_and_arms(frame)
    assert first == second
    assert frame.get_column("covariate").to_list() == [float("inf"), float("-inf")]


def test_linear_readout_keeps_moments_and_units_on_one_snapshot(tmp_path, monkeypatch):
    import pytest

    from increment import Method
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, *_ = covariate_defs_and_con(tmp_path)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: linear, variance_reduction: cuped_linear}, covariates: "
            "[{name: pre_tenure, kind: continuous, property: tenure, source: events}]}}",
        )
    )
    analysis = Analysis.from_definitions("cov_test", defs_path, con, store="none")
    try:
        baseline = next(
            row.require_lift()
            for row in cast(
                list[LiftEstimate],
                analysis.run(
                    decision_method=Method(name="plain"),
                    sensitivity_methods=[Method(name="linear", variance_reduction="cuped_linear")],
                ),
            )
            if row.group_id == "treatment" and row.method == "linear"
        )
        source = analysis._src
        original = type(source).unit_frame
        mutated = False

        def mutate_before_units(self, *args, **kwargs):
            nonlocal mutated
            if not mutated:
                mutated = True
                con.raw_sql(
                    "UPDATE cov_events SET revenue = revenue * 2 "
                    "WHERE event = 'purchase' AND user_id BETWEEN 'u10' AND 'u19'"
                )
            return original(self, *args, **kwargs)

        monkeypatch.setattr(type(source), "unit_frame", mutate_before_units)
        after = next(
            row.require_lift()
            for row in cast(
                list[LiftEstimate],
                analysis.run(
                    decision_method=Method(name="plain"),
                    sensitivity_methods=[Method(name="linear", variance_reduction="cuped_linear")],
                ),
            )
            if row.group_id == "treatment" and row.method == "linear"
        )
        assert mutated
        assert after.value == pytest.approx(baseline.value, rel=1e-10, abs=1e-10)
    finally:
        analysis.close()
        con.disconnect()


def test_infinite_cuped_covariate_is_localized_by_estimator(tmp_path):
    import pyarrow as pa

    from increment import Method, MetricSpec
    from increment.frame import CupedCovariate

    rows = [
        {
            "unit": f"{group}-{index}",
            "group": group,
            "revenue": float(index + (group == "treatment")),
            "pre": float("inf") if group == "treatment" and index == 0 else float(index),
        }
        for group in ("control", "treatment")
        for index in range(12)
    ]
    analysis = Analysis.from_unit_summary(
        pa.Table.from_pylist(rows),
        unit="unit",
        group="group",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method(name="linear", variance_reduction="cuped_linear"),
                covariates=(CupedCovariate(name="pre", kind="continuous"),),
            )
        ],
    )
    results = cast(
        list[LiftEstimate],
        list(
            analysis.run(
                decision_method=Method(name="plain"),
                sensitivity_methods=[Method(name="linear", variance_reduction="cuped_linear")],
            )
        ),
    )
    treatment = next(
        row for row in results if row.group_id == "treatment" and row.method == "linear"
    )
    assert treatment.lift is None
    assert treatment.failure_code == "estimation.cuped_linear.nonfinite"
