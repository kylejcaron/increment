from __future__ import annotations

from typing import Any, cast

import narwhals as nw
import pytest

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
        + "\n    plan: {primary: {metric: revenue, decision_method: {name: mixed, "
        "variance_reduction: cuped_multicovariate, covariates: [" + binding + "]}}}",
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


def test_scalar_cuped_uses_definition_property_binding_without_preperiods(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    text = defs_path.read_text()
    text = text.replace(
        "    design:\n"
        "      mechanism: observational\n"
        "      covariates:\n"
        "        - {property: tenure, source: events}\n"
        "        - {property: region, source: events}\n",
        "",
    )
    text = text.replace(
        "plan: {primary: revenue}",
        "plan: {primary: {metric: revenue, decision_method: {name: scalar, "
        "variance_reduction: cuped, covariates: [{name: pre_tenure, kind: continuous, "
        "property: tenure, source: events}]}}}",
    )
    defs_path.write_text(text)
    analysis = Analysis.from_definitions("cat_test", defs_path, con)
    exported = tmp_path / "scalar.parquet"
    analysis.export(exported)
    raw_moments = pq.read_table(exported).to_pylist()
    assert raw_moments[0]["ref_x"] is not None
    native = analysis.run()[0]
    oracle_frame = pa.table(
        {
            "unit": list(units),
            "group": [units[unit]["variant"] for unit in units],
            "revenue": [units[unit]["revenue"] for unit in units],
            "pre_tenure": [units[unit]["tenure"] for unit in units],
        }
    )
    oracle = Analysis.from_unit_summary(
        oracle_frame,
        unit="unit",
        group="group",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method.cuped(
                    name="scalar",
                    covariates=CupedCovariate(name="pre_tenure", kind="continuous"),
                ),
            )
        ],
    ).run()[0]

    assert isinstance(native, LiftEstimate) and isinstance(oracle, LiftEstimate)
    assert native.require_lift().value == pytest.approx(oracle.require_lift().value)
    assert native.require_lift().log_se == pytest.approx(oracle.require_lift().log_se)


@pytest.mark.parametrize("missing", ["impute", "zero", "refuse"])
def test_scalar_definition_covariate_applies_missing_policy(tmp_path, missing):
    import pyarrow as pa

    from increment.errors import IncrementWarning
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec

    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=("u3",))
    con.raw_sql(
        "UPDATE cat_events SET tenure = NULL WHERE user_id = 'u3' "
        "AND ts < TIMESTAMP '2025-08-01 00:00:00'"
    )
    text = defs_path.read_text().replace(
        "    design:\n"
        "      mechanism: observational\n"
        "      covariates:\n"
        "        - {property: tenure, source: events}\n"
        "        - {property: region, source: events}\n",
        "",
    )
    text = text.replace(
        "plan: {primary: revenue}",
        "plan: {primary: {metric: revenue, decision_method: {name: scalar, "
        "variance_reduction: cuped, covariates: [{name: pre_tenure, kind: continuous, "
        "property: tenure, source: events, missing: " + missing + "}]}}}",
    )
    defs_path.write_text(text)
    native = Analysis.from_definitions("cat_test", defs_path, con)
    oracle_frame = pa.table(
        {
            "unit": list(units),
            "group": [units[unit]["variant"] for unit in units],
            "revenue": [units[unit]["revenue"] for unit in units],
            "pre_tenure": [None if unit == "u3" else units[unit]["tenure"] for unit in units],
        }
    )
    if missing == "refuse":
        oracle = Analysis.from_unit_summary(
            oracle_frame,
            unit="unit",
            group="group",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped(
                        name="scalar",
                        covariates=CupedCovariate(
                            name="pre_tenure", kind="continuous", missing=missing
                        ),
                    ),
                )
            ],
        )
        for result in (oracle.run(), native.run()):
            (row,) = result
            assert isinstance(row, LiftEstimate)
            assert row.lift is None
            assert row.failure_code == "frame.validation.metric_covariate_column"
        return

    with pytest.warns() as imputation_warnings:
        oracle = Analysis.from_unit_summary(
            oracle_frame,
            unit="unit",
            group="group",
            control="control",
            metrics=[
                MetricSpec(
                    name="revenue",
                    decision_method=Method.cuped(
                        name="scalar",
                        covariates=CupedCovariate(
                            name="pre_tenure", kind="continuous", missing=missing
                        ),
                    ),
                )
            ],
        )

    warning = cast(IncrementWarning, imputation_warnings[0].message)
    assert warning.code == "frame.validation.metric_covariate_missing_imputed"
    assert warning.context["n_missing"] == 1

    native_row = native.run()[0]
    oracle_row = oracle.run()[0]
    assert isinstance(native_row, LiftEstimate)
    assert isinstance(oracle_row, LiftEstimate)
    assert native_row.require_lift().value == pytest.approx(
        oracle_row.require_lift().value, rel=1e-10
    )
    assert native_row.require_lift().log_se == pytest.approx(
        oracle_row.require_lift().log_se, rel=1e-10
    )


@pytest.mark.slow
def test_method_covariate_alias_conflicts_with_observational_binding(tmp_path):
    from increment.errors import InvalidRequestError

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        "{name: tenure, kind: categorical, property: region, source: events}",
    )
    source = _moment_source(Analysis.from_definitions("cat_test", defs_path, con))

    with pytest.raises(InvalidRequestError) as raised:
        source.unit_frame(source.context.metrics[0], covariates=["tenure"])
    assert raised.value.code == "facade.analysis_config.covariate_binding_conflict"


@pytest.mark.parametrize(
    "alias",
    [
        "unit_id",
        "group_id",
        "y",
        "y_den",
        "y_raw",
        "cluster_id",
        "experiment_id",
        "metric",
        "x",
        "d",
    ],
)
def test_property_binding_rejects_unit_frame_identity_alias(tmp_path, alias):
    import pytest

    from increment.errors import DefinitionError

    defs_path, _con, _units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        f"{{name: {alias}, kind: continuous, property: tenure, source: events}}",
    )

    with pytest.raises(DefinitionError) as raised:
        load(defs_path)

    assert raised.value.code == "definition.experiment.covariate_name_reserved"


def test_property_binding_rejects_experiment_cluster_alias(tmp_path):
    import pytest

    from increment.errors import DefinitionError

    defs_path, _con, _units = categorical_defs_and_con(tmp_path)
    defs_path.write_text(
        defs_path.read_text().replace(
            '    end: "2025-08-07"',
            '    end: "2025-08-07"\n    cluster: tenure',
        )
    )
    _with_metric_covariates(
        defs_path,
        "{name: tenure, kind: continuous, property: tenure, source: events}",
    )

    with pytest.raises(DefinitionError) as raised:
        load(defs_path)

    assert raised.value.code == "definition.experiment.covariate_name_reserved"


def test_measure_covariate_filter_is_validated_when_definitions_load(tmp_path):
    import pytest

    from increment.errors import DefinitionError

    defs_path, _con, _units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        "{name: prior_purchase, kind: continuous, measure: {fact: purchase, "
        "aggregation: sum, filters: [{property: absent, op: equals, values: [1]}]}}",
        n_pre_periods=30,
    )

    with pytest.raises(DefinitionError) as raised:
        load(defs_path)

    assert raised.value.code == "definition.invalid"
    assert any(
        code == "definition.validate_filters.filter_references_unknown"
        for code, *_rest in cast(Any, raised.value.context["errors"])
    ), raised.value.context["errors"]


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
            "    plan: {primary: [{metric: revenue, decision_method: {name: mixed, "
            "variance_reduction: cuped_multicovariate, covariates: "
            "[{name: pre_region, kind: categorical, property: region, source: events}]}}, "
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
            "    plan: {primary: {metric: revenue, decision_method: {name: mixed, "
            "variance_reduction: cuped_multicovariate, covariates: "
            "[{name: pre_tenure, kind: continuous, property: tenure}]}}}",
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
            "    plan: {primary: {metric: revenue, decision_method: {name: mixed, "
            "variance_reduction: cuped_multicovariate, covariates: "
            "[{name: pre_tenure, kind: continuous, property: tenure, source: events_b}]}}}",
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
            "    plan: {primary: [{metric: revenue, decision_method: {name: scalar, "
            "variance_reduction: cuped, covariates: [{name: prior, kind: continuous, "
            "measure: {fact: purchase, aggregation: sum}}]}}, "
            "{metric: revenue_other, decision_method: {name: mixed, "
            "variance_reduction: cuped_multicovariate, covariates: "
            "[{name: prior, kind: continuous, property: prior, source: events}]}}]}",
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


def test_multicovariate_observational_route_refuses_before_acquisition(tmp_path):
    import pytest

    from increment.errors import UnsupportedRequestError

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    _with_metric_covariates(
        defs_path,
        "{name: region, kind: categorical, property: region, source: events}",
    )
    analysis = Analysis.from_definitions("cat_test", defs_path, con)

    with pytest.raises(UnsupportedRequestError) as raised:
        analysis.run()

    assert raised.value.code == "estimation.engine.cuped_linear_observational"


def test_from_moments_refuses_multicovariate_with_unit_source_route():
    import pytest

    from increment.errors import UnsupportedRequestError
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec

    spec = MetricSpec(
        name="revenue",
        decision_method=Method.cuped_multicovariate(
            covariates=CupedCovariate(name="prior_spend", kind="continuous")
        ),
    )
    with pytest.raises(UnsupportedRequestError) as raised:
        Analysis.from_moments([], metrics=[spec], control="control")

    assert raised.value.code == "facade.analysis_config.cuped_linear_moments"
    import copy
    import pickle

    for restored in (copy.deepcopy(raised.value), pickle.loads(pickle.dumps(raised.value))):
        assert restored.code == raised.value.code
        assert restored.context == raised.value.context


def test_from_moments_named_scalar_accepts_legacy_x_and_refuses_known_mismatch(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from increment.errors import UnsupportedRequestError
    from increment.estimation.engine import Method
    from increment.frame import CupedCovariate, MetricSpec

    frame = pa.table(
        {
            "unit": range(8),
            "arm": ["control"] * 4 + ["treatment"] * 4,
            "revenue": [10.0, 12.0, 9.0, 11.0, 13.0, 14.0, 12.0, 15.0],
            "history_a": [1.0, 2.0, 4.0, 3.0, 1.0, 3.0, 2.0, 5.0],
        }
    )
    source = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="arm",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                decision_method=Method.cuped(
                    name="history_a",
                    covariates=CupedCovariate(name="history_a", kind="continuous"),
                ),
            )
        ],
    )
    path = tmp_path / "moments.parquet"
    source.export(path)
    stored = pq.read_table(path).to_pylist()
    named = MetricSpec(
        name="revenue",
        decision_method=Method.cuped(
            name="history_b",
            covariates=CupedCovariate(name="history_b", kind="continuous"),
        ),
    )

    (result,) = Analysis.from_moments(stored, metrics=[named], control="control").run()
    assert isinstance(result, LiftEstimate)
    assert result.lift is not None

    mismatched = [{**row, "covariate_name": "history_a"} for row in stored]
    with pytest.raises(UnsupportedRequestError) as raised:
        Analysis.from_moments(mismatched, metrics=[named], control="control")

    assert raised.value.code == "facade.analysis_config.cuped_scalar_moments"


def test_switchback_refuses_multicovariate_with_independent_unit_route():
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
        decision_method=Method.cuped_multicovariate(
            covariates=CupedCovariate(name="prior_spend", kind="continuous")
        ),
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
            "    plan: {primary: {metric: revenue, decision_method: {name: mixed, "
            "variance_reduction: cuped_multicovariate, covariates: "
            "[{name: region, kind: categorical, property: region, source: events}]}}}",
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
    from tests.analysis_factory import _native_source
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, tenure, groups, revenue = covariate_defs_and_con(tmp_path)
    text = defs_path.read_text().replace(
        "    plan: {secondaries: [revenue]}",
        "    plan: {primary: {metric: revenue, "
        "decision_method: {name: linear, variance_reduction: cuped_multicovariate, "
        "covariates: [{name: pre_tenure, kind: continuous, property: tenure, source: events}]}}}",
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
    covariates = cast(Any, _native_source(native).context.configs[0].covariates)
    assert covariates[0].name == "pre_tenure"
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
                decision_method=Method.cuped_multicovariate(
                    name="linear",
                    covariates=CupedCovariate(name="pre_tenure", kind="continuous"),
                ),
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
        assert evidence.keys() == expected_evidence.keys()
        for key, (available, p_value, reference) in evidence.items():
            expected_available, expected_p, expected_reference = expected_evidence[key]
            assert (available, reference) == (expected_available, expected_reference)
            assert p_value == pytest.approx(expected_p, rel=1e-10, abs=1e-12)


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
    from increment.frame import CupedCovariate
    from tests.analysis_factory import _native_source
    from tests.covariate_cases import covariate_defs_and_con

    defs_path, con, *_ = covariate_defs_and_con(tmp_path)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: pre_tenure, kind: continuous, property: tenure, source: events}]}}}",
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
                    sensitivity_methods=[
                        Method.cuped_multicovariate(
                            name="linear",
                            covariates=CupedCovariate(name="pre_tenure", kind="continuous"),
                        )
                    ],
                ),
            )
            if row.group_id == "treatment" and row.method == "linear"
        )
        source = _native_source(analysis)
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
                    sensitivity_methods=[
                        Method.cuped_multicovariate(
                            name="linear",
                            covariates=CupedCovariate(name="pre_tenure", kind="continuous"),
                        )
                    ],
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
                decision_method=Method.cuped_multicovariate(
                    name="linear", covariates=CupedCovariate(name="pre", kind="continuous")
                ),
            )
        ],
    )
    results = cast(
        list[LiftEstimate],
        list(
            analysis.run(
                decision_method=Method(name="plain"),
                sensitivity_methods=[
                    Method.cuped_multicovariate(
                        name="linear",
                        covariates=CupedCovariate(name="pre", kind="continuous"),
                    )
                ],
            )
        ),
    )
    treatment = next(
        row for row in results if row.group_id == "treatment" and row.method == "linear"
    )
    assert treatment.lift is None
    assert treatment.failure_code == "estimation.cuped_linear.nonfinite"


def _scalar_native_case(tmp_path, *, bare=False, mixed=False):
    import json

    from increment import Method
    from increment.semantics.models import CupedCovariate, CupedCovariateBinding
    from tests.covariate_cases import covariate_defs_and_con

    path, con, tenure, groups, revenue = covariate_defs_and_con(tmp_path)
    con.raw_sql("ALTER TABLE cov_events ADD COLUMN history_b DOUBLE")
    con.raw_sql("ALTER TABLE cov_events ADD COLUMN segment VARCHAR")
    con.raw_sql(
        "UPDATE cov_events SET history_b = (CAST(tenure AS INTEGER) * 7) % 11, "
        "segment = 'all' WHERE tenure IS NOT NULL"
    )
    first = (
        CupedCovariate(name="tenure", kind="continuous", missing="zero")
        if bare
        else CupedCovariateBinding(
            name="history_a", kind="continuous", missing="zero", property="tenure", source="events"
        )
    )
    second = (
        CupedCovariate(name="history_b", kind="continuous", missing="refuse")
        if bare
        else CupedCovariateBinding(
            name="history_b",
            kind="continuous",
            missing="refuse",
            property="history_b",
            source="events",
        )
    )
    methods = (
        Method.cuped(name="scalar_a", covariates=first),
        (Method.cuped_multicovariate if mixed else Method.cuped)(
            name="companion_b", covariates=second
        ),
    )
    plan = {
        "primary": {
            "metric": "revenue",
            "decision_method": methods[0].model_dump(mode="json", exclude_none=True),
            "sensitivity_methods": [methods[1].model_dump(mode="json", exclude_none=True)],
        }
    }
    path.write_text(
        path.read_text()
        .replace(
            "exposures:\n",
            "      - {name: history_b, column: history_b, dtype: float, as_of: pre_exposure}\n"
            "      - {name: segment, column: segment, dtype: string, as_of: pre_exposure}\n"
            "exposures:\n",
        )
        .replace("    plan: {secondaries: [revenue]}", "    plan: " + json.dumps(plan))
    )
    return path, con, methods, tenure, groups, revenue


@pytest.mark.parametrize("bare", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_native_method_families_match_standalone_and_frame(tmp_path, bare, mixed):
    import pyarrow as pa

    from increment import Method, MetricSpec
    from increment.frame import CupedCovariate

    path, con, methods, tenure, groups, revenue = _scalar_native_case(
        tmp_path, bare=bare, mixed=mixed
    )
    table = pa.Table.from_pylist(
        [
            {
                "unit": unit,
                "group": groups[unit],
                "revenue": revenue[unit],
                methods[0].covariates[0].name: tenure[unit],
                "history_b": (int(tenure[unit]) * 7) % 11,
            }
            for unit in groups
        ]
    )
    try:
        with Analysis.from_definitions("cov_test", path, con) as native:
            joint = {row.method: row for row in native.run() if isinstance(row, LiftEstimate)}
            for method in methods:
                (alone,) = native.run(decision_method=method, sensitivity_methods=())
                assert isinstance(alone, LiftEstimate)
                factory = (
                    Method.cuped
                    if method.variance_reduction == "cuped"
                    else Method.cuped_multicovariate
                )
                with Analysis.from_unit_summary(
                    table,
                    unit="unit",
                    group="group",
                    control="control",
                    metrics=[
                        MetricSpec(
                            name="revenue",
                            decision_method=factory(
                                name=method.name,
                                covariates=CupedCovariate(
                                    name=method.covariates[0].name,
                                    kind="continuous",
                                    missing=method.covariates[0].missing,
                                ),
                            ),
                        )
                    ],
                ) as frame:
                    (expected,) = frame.run()
                assert isinstance(expected, LiftEstimate)
                for actual in (joint[method.name], alone):
                    assert actual.require_lift().value == pytest.approx(
                        expected.require_lift().value, rel=1e-10
                    )
                    assert actual.require_lift().log_se == pytest.approx(
                        expected.require_lift().log_se, rel=1e-10
                    )
    finally:
        con.disconnect()


def test_native_breakout_uses_effective_scalar_family(tmp_path):
    path, con, methods, *_ = _scalar_native_case(tmp_path)
    path.write_text(
        path.read_text().replace(
            "    start:", "    breakouts: [{property: segment, source: events}]\n    start:"
        )
    )
    try:
        with Analysis.from_definitions("cov_test", path, con) as analysis:
            (expected,) = analysis.run(decision_method=methods[1], sensitivity_methods=())
            assert isinstance(expected, LiftEstimate)
            (actual,) = analysis.run_breakout(decision_method=methods[1], sensitivity_methods=())
            assert actual.dimension_value == "all"
            assert actual.require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert actual.require_lift().log_se == pytest.approx(
                expected.require_lift().log_se, rel=1e-10
            )
    finally:
        con.disconnect()


def test_direct_native_readout_does_not_acquire_removed_scalar_binding(tmp_path):
    from increment import Method, readouts
    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession

    path, con, *_ = _scalar_native_case(tmp_path)
    definitions = load(path)
    experiment = definitions.experiment("cov_test")
    assert experiment is not None
    design = experiment.resolved_design()
    source = DefinitionsMomentSource(
        WarehouseSession(con, definitions),
        experiment,
        definitions.metrics,
        store="none",
        on_mixed_assignment="error",
        design=design,
        plan=compile_decision_plan(
            experiment.plan, definitions.metrics, path="warehouse", design=design
        ),
    )
    try:
        con.raw_sql("ALTER TABLE cov_events DROP COLUMN tenure")
        con.raw_sql("ALTER TABLE cov_events DROP COLUMN history_b")
        (row,) = readouts.run(source, decision_method=Method.unadjusted(), sensitivity_methods=())
        assert row.method == "naive"
        assert row.require_lift().value is not None
    finally:
        con.disconnect()


def test_native_scalar_measure_binding_runs_end_to_end(tmp_path):
    import pyarrow as pa

    from increment import Method, MetricSpec
    from increment.frame import CupedCovariate
    from tests.covariate_cases import covariate_defs_and_con

    path, con, tenure, groups, revenue = covariate_defs_and_con(tmp_path)
    con.raw_sql(
        "INSERT INTO cov_events SELECT user_id, TIMESTAMP '2025-07-25', 'purchase', "
        "NULL, NULL, tenure, NULL FROM cov_events WHERE tenure IS NOT NULL"
    )
    path.write_text(
        path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    n_pre_periods: 30\n"
            "    plan: {primary: {metric: revenue, decision_method: {name: scalar, "
            "variance_reduction: cuped, covariates: [{name: pre_revenue, kind: continuous, "
            "missing: refuse, measure: {fact: purchase, aggregation: sum}}]}}}",
        )
    )
    table = pa.Table.from_pylist(
        [
            {
                "unit": unit,
                "group": groups[unit],
                "revenue": revenue[unit],
                "pre_revenue": tenure[unit],
            }
            for unit in groups
        ]
    )
    try:
        with (
            Analysis.from_definitions("cov_test", path, con) as native,
            Analysis.from_unit_summary(
                table,
                unit="unit",
                group="group",
                control="control",
                metrics=[
                    MetricSpec(
                        name="revenue",
                        decision_method=Method.cuped(
                            name="scalar",
                            covariates=CupedCovariate(
                                name="pre_revenue", kind="continuous", missing="refuse"
                            ),
                        ),
                    )
                ],
            ) as frame,
        ):
            (actual,) = native.run()
            (expected,) = frame.run()
            assert isinstance(actual, LiftEstimate)
            assert isinstance(expected, LiftEstimate)
            assert actual.require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert actual.require_lift().log_se == pytest.approx(
                expected.require_lift().log_se, rel=1e-10
            )
    finally:
        con.disconnect()
