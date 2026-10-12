from __future__ import annotations

import narwhals as nw

from increment.analysis import Analysis
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


def test_native_definition_covariate_property_acquisition_preserves_category_and_null(tmp_path):
    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=("u3",))
    _with_metric_covariates(
        defs_path,
        "{name: region, kind: categorical, property: region, source: events}",
    )
    source = _moment_source(Analysis.from_definitions("cat_test", defs_path, con))
    metric = source.context.metrics[0]
    table = nw.from_native(
        source.unit_frame(metric, covariates=["region"]), eager_only=True
    ).to_arrow()
    rows = {row["unit_id"]: row["region"] for row in table.to_pylist()}

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
        source.unit_frame(metric, covariates=["pre_revenue"]), eager_only=True
    ).to_arrow()
    rows = {row["unit_id"]: row["pre_revenue"] for row in table.to_pylist()}

    assert rows["u0"] == 11.0
    assert rows["u1"] == 0.0


def test_property_extensions_roundtrip_from_native_to_artifact(tmp_path):
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore

    defs_path, con, _units = categorical_defs_and_con(tmp_path, null_units=("u3",))
    _with_metric_covariates(
        defs_path,
        "{name: region, kind: categorical, property: region, source: events}",
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
        nw.from_native(native.unit_frame(metric, covariates=["region"]), eager_only=True)
        .to_arrow()
        .to_pylist()
    )
    artifact_rows = (
        nw.from_native(artifact.unit_frame(metric, covariates=["region"]), eager_only=True)
        .to_arrow()
        .to_pylist()
    )

    assert artifact_rows == native_rows


def test_cuped_linear_run_refuses_as_unfinished_before_acquisition(tmp_path):
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

    assert raised.value.code == "estimation.engine.cuped_linear_unfinished"
