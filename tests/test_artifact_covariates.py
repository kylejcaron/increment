"""Declared observational covariates round-trip through a unit-day artifact:
a numeric property on the frozen `unit_covariate` relation, a string
property on its own `unit_covariate_level` relation."""

from __future__ import annotations

import narwhals as nw
import pytest

from increment.analysis import Analysis
from increment.errors import CodedError
from increment.semantics.artifact import UnitDayArtifactManifest
from tests.analysis_factory import _moment_source
from tests.covariate_cases import categorical_defs_and_con, covariate_defs_and_con

pytestmark = pytest.mark.integration

_NUMERIC_COVARIATE_SCHEMA = (
    ("experiment_id", "STRING", False),
    ("unit_id", "STRING", False),
    ("value", "FLOAT64", True),
)
_LEGACY_COVARIATE_EXTENSION_FIELDS = frozenset(
    {
        "extension_version",
        "relation",
        "definition_sha256",
        "source_provenance_sha256",
        "kind",
        "covariate_name",
        "source_name",
        "as_of",
    }
)


def _publish_and_reopen(
    defs_path, con, *, request_covariate: bool, full_catalog: bool = False
) -> tuple[Analysis, UnitDayArtifactManifest]:
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics import load
    from increment.semantics.artifact import UnitCovariateRequest

    defs = load(defs_path)
    experiment = defs.experiment("cov_test")
    assert experiment is not None
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    native = Analysis.from_definitions("cov_test", defs_path, con)
    requests = (
        [UnitCovariateRequest(property_name="tenure", source_name="events")]
        if request_covariate
        else []
    )
    if full_catalog:
        requests = [entry.request for entry in unit_day_artifact_extension_catalog(context)]
    ref = native.publish_unit_day_artifact(store, extensions=requests)
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context), manifest


@pytest.mark.integration
def test_artifact_serves_declared_unit_covariate_and_keeps_missing_null(tmp_path):
    from increment.semantics.design import Observational

    defs_path, con, tenure, _group, _revenue = covariate_defs_and_con(tmp_path, observational=True)
    con.raw_sql("UPDATE cov_events SET tenure = NULL WHERE user_id = 'u3'")
    analysis, _ = _publish_and_reopen(defs_path, con, request_covariate=True)
    source = _moment_source(analysis)
    assert isinstance(source.context.design, Observational)
    assert source.context.design.adjustment.covariates == ("tenure",)
    table = nw.from_native(
        source.unit_frame(source.context.metrics[0], covariates=["tenure"]), eager_only=True
    ).to_arrow()
    rows = {row["unit_id"]: row["tenure"] for row in table.to_pylist()}
    assert rows.pop("u3") is None
    assert rows == pytest.approx({u: t for u, t in tenure.items() if u != "u3"})


@pytest.mark.integration
def test_artifact_auto_publishes_declared_covariate_without_explicit_request(tmp_path):
    """Regression: a covariate is fully determined by `design.covariates`, so
    publishing must include it even when the caller never builds a
    `UnitCovariateRequest` by hand."""
    defs_path, con, tenure, _group, _revenue = covariate_defs_and_con(tmp_path, observational=True)
    analysis, _ = _publish_and_reopen(defs_path, con, request_covariate=False)
    source = _moment_source(analysis)
    table = nw.from_native(
        source.unit_frame(source.context.metrics[0], covariates=["tenure"]), eager_only=True
    ).to_arrow()
    rows = {row["unit_id"]: row["tenure"] for row in table.to_pylist()}
    assert rows == pytest.approx(tenure)


@pytest.mark.integration
def test_metric_binding_does_not_shadow_observational_covariate_on_artifact(tmp_path):
    defs_path, con, units = categorical_defs_and_con(tmp_path)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {primary: revenue}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: tenure, kind: categorical, property: region, source: events}]}}}",
        )
    )
    analysis, _ = _publish_categorical(defs_path, con)
    source = _moment_source(analysis)
    table = nw.from_native(
        source.unit_frame(source.context.metrics[0], covariates=["tenure"]), eager_only=True
    ).to_arrow()
    rows = {row["unit_id"]: row["tenure"] for row in table.to_pylist()}

    assert rows["u0"] == units["u0"]["tenure"]


@pytest.mark.integration
def test_observational_property_wins_over_same_named_measure_binding_on_replay(tmp_path):
    defs_path, con, tenure, *_ = covariate_defs_and_con(tmp_path, observational=True)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    n_pre_periods: 30\n"
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: tenure, kind: continuous, measure: "
            "{fact: exposure, aggregation: count}}]}}}",
        )
    )
    analysis, _ = _publish_and_reopen(defs_path, con, request_covariate=False)
    artifact = _moment_source(analysis)
    artifact_rows = nw.from_native(
        artifact.unit_frame(artifact.context.metrics[0], covariates=["tenure"]),
        eager_only=True,
    ).to_arrow()
    native = _moment_source(Analysis.from_definitions("cov_test", defs_path, con))
    native_rows = nw.from_native(
        native.unit_frame(native.context.metrics[0], covariates=["tenure"]),
        eager_only=True,
    ).to_arrow()
    expected = tenure

    assert {row["unit_id"]: row["tenure"] for row in native_rows.to_pylist()} == expected
    assert {row["unit_id"]: row["tenure"] for row in artifact_rows.to_pylist()} == expected


def test_artifact_scalar_cuped_uses_named_property_covariate(tmp_path):
    defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=False)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: scalar, variance_reduction: cuped, covariates: "
            "[{name: tenure, kind: continuous, property: tenure, source: events}]}}}",
        )
    )
    artifact, _ = _publish_and_reopen(defs_path, con, request_covariate=False)
    native = Analysis.from_definitions("cov_test", defs_path, con)

    from increment.estimation.results import LiftEstimate

    artifact_row = next(row for row in artifact.run() if getattr(row, "method", None) == "scalar")
    native_row = next(row for row in native.run() if getattr(row, "method", None) == "scalar")
    assert isinstance(artifact_row, LiftEstimate)
    assert isinstance(native_row, LiftEstimate)
    assert artifact_row.require_lift().value == pytest.approx(native_row.require_lift().value)
    assert artifact_row.require_lift().log_se == pytest.approx(native_row.require_lift().log_se)


def test_reserved_binding_alias_has_same_native_and_artifact_refusal(tmp_path):
    defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=False)
    defs_path.write_text(
        defs_path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: linear, variance_reduction: cuped_multicovariate, covariates: "
            "[{name: metric, kind: continuous, property: tenure, source: events}]}}}",
        )
    )

    with pytest.raises(CodedError) as native_error:
        Analysis.from_definitions("cov_test", defs_path, con)
    with pytest.raises(CodedError) as artifact_error:
        _publish_and_reopen(defs_path, con, request_covariate=False)

    assert native_error.value.code == artifact_error.value.code
    assert native_error.value.code == "definition.experiment.covariate_name_reserved"


def test_undeclared_covariate_cannot_be_published(tmp_path):
    """The catalog is compiled from declared state: a randomized experiment
    offers no covariate extension, so a request for one is refused."""
    defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=False)
    with pytest.raises(CodedError) as raised:
        _publish_and_reopen(defs_path, con, request_covariate=True)
    assert raised.value.code == "artifact.extension.missing"


def _publish_categorical(
    defs_path, con, *, extensions=()
) -> tuple[Analysis, UnitDayArtifactManifest]:
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.semantics import load

    defs = load(defs_path)
    experiment = defs.experiment("cat_test")
    assert experiment is not None
    context = artifact_context(defs, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    native = Analysis.from_definitions("cat_test", defs_path, con)
    ref = native.publish_unit_day_artifact(store, extensions=list(extensions))
    with store.open_snapshot(ref) as snapshot:
        manifest = snapshot.read_manifest(ref.manifest, expected_sha256=ref.manifest_sha256)
    return Analysis.from_unit_day_artifact(store, ref, expected_context=context), manifest


@pytest.mark.integration
def test_artifact_serves_a_string_covariate_as_levels_and_keeps_missing_null(tmp_path):
    """A declared string property is published on its own
    `unit_covariate_level` relation and read back as a string column -- the
    label verbatim, a unit without a pre-exposure level NULL -- while the
    numeric covariate beside it still travels on `unit_covariate`."""
    import pyarrow as pa

    defs_path, con, units = categorical_defs_and_con(tmp_path, null_units=("u3",))
    analysis, manifest = _publish_categorical(defs_path, con)
    source = _moment_source(analysis)
    carriers = {
        extension.covariate_name: (extension.kind, extension.relation.role)
        for extension in manifest.extensions
        if extension.kind == "unit_covariate" or extension.kind == "unit_covariate_level"
    }
    assert carriers == {
        "tenure": ("unit_covariate", "unit_covariate"),
        "region": ("unit_covariate_level", "unit_covariate_level"),
    }
    table = nw.from_native(
        source.unit_frame(source.context.metrics[0], covariates=["tenure", "region"]),
        eager_only=True,
    ).to_arrow()
    assert table.schema.field("region").type == pa.string()
    assert table.schema.field("tenure").type == pa.float64()
    rows = {row["unit_id"]: row for row in table.to_pylist()}
    assert set(rows) == set(units)
    assert rows["u3"]["region"] is None
    assert {unit: row["region"] for unit, row in rows.items() if unit != "u3"} == {
        unit: truth["region"] for unit, truth in units.items() if unit != "u3"
    }
    assert {unit: row["tenure"] for unit, row in rows.items()} == pytest.approx(
        {unit: truth["tenure"] for unit, truth in units.items()}
    )


@pytest.mark.integration
def test_string_covariate_is_requestable_only_under_its_level_request(tmp_path):
    """The catalog types a string covariate as `unit_covariate_level`: the
    numeric request for it is refused as absent from the trusted catalog,
    the level request publishes it."""
    from increment.semantics.artifact import UnitCovariateLevelRequest, UnitCovariateRequest

    defs_path, con, _units = categorical_defs_and_con(tmp_path)
    with pytest.raises(CodedError) as raised:
        _publish_categorical(
            defs_path,
            con,
            extensions=[UnitCovariateRequest(property_name="region", source_name="events")],
        )
    assert raised.value.code == "artifact.extension.missing"

    defs_path, con, units = categorical_defs_and_con(tmp_path)
    analysis, _ = _publish_categorical(
        defs_path,
        con,
        extensions=[UnitCovariateLevelRequest(property_name="region", source_name="events")],
    )
    source = _moment_source(analysis)
    table = nw.from_native(
        source.unit_frame(source.context.metrics[0], covariates=["region"]), eager_only=True
    ).to_arrow()
    assert {row["unit_id"]: row["region"] for row in table.to_pylist()} == {
        unit: truth["region"] for unit, truth in units.items()
    }


@pytest.mark.integration
def test_numeric_covariate_wire_is_untouched_by_the_level_relation(tmp_path):
    """Legacy numeric artifacts keep verifying: the numeric covariate
    relation still carries its frozen three-column schema digest, and its
    extension serializes with exactly its legacy fields, so a manifest
    published before categorical covariates existed reproduces the same
    relation and manifest digests today."""
    from increment.query.artifact_digest import schema_sha256
    from increment.semantics.artifact import UnitCovariateExtension

    defs_path, con, *_ = covariate_defs_and_con(tmp_path, observational=True)
    _analysis, manifest = _publish_and_reopen(defs_path, con, request_covariate=False)
    (extension,) = [e for e in manifest.extensions if e.kind == "unit_covariate"]
    assert extension.relation.role == "unit_covariate"
    assert extension.relation.schema_sha256 == schema_sha256(
        "unit_covariate", _NUMERIC_COVARIATE_SCHEMA
    )
    payload = extension.model_dump(mode="json")
    assert set(payload) == _LEGACY_COVARIATE_EXTENSION_FIELDS
    assert UnitCovariateExtension.model_validate(payload).model_dump(mode="json") == payload


@pytest.mark.parametrize("missing", ["zero", "impute", "refuse"])
@pytest.mark.parametrize("null_value", ["null", "nan"])
def test_artifact_scalar_missing_policy_matches_native_and_frame(tmp_path, missing, null_value):
    import pyarrow as pa

    from increment import Method
    from increment.errors import IncrementWarning
    from increment.estimation.results import LiftEstimate
    from increment.frame import CupedCovariate, MetricSpec

    path, con, tenure, groups, revenue = covariate_defs_and_con(tmp_path)
    name = "pre_tenure"
    path.write_text(
        path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: scalar, variance_reduction: cuped, covariates: "
            f"[{{name: {name}, kind: continuous, missing: {missing}, "
            "property: tenure, source: events}]}}}",
        )
    )
    value = "NULL" if null_value == "null" else "'NaN'::DOUBLE"
    con.raw_sql(f"UPDATE cov_events SET tenure = {value} WHERE user_id IN ('u1', 'u12')")
    table = pa.Table.from_pylist(
        [
            {
                "unit": unit,
                "group": groups[unit],
                "revenue": revenue[unit],
                name: (
                    (None if null_value == "null" else float("nan"))
                    if unit in ("u1", "u12")
                    else tenure[unit]
                ),
            }
            for unit in groups
        ]
    )
    artifact, _ = _publish_and_reopen(path, con, request_covariate=False)
    try:
        native = Analysis.from_definitions("cov_test", path, con)

        def make_frame():
            return Analysis.from_unit_summary(
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
                                name=name, kind="continuous", missing=missing
                            ),
                        ),
                    )
                ],
            )

        if missing == "refuse":
            with native, make_frame() as frame:
                for result in (native.run(), frame.run(), artifact.run()):
                    (row,) = result
                    assert isinstance(row, LiftEstimate)
                    assert row.lift is None
                    assert row.failure_code == "frame.validation.metric_covariate_column"
            return

        with pytest.warns(IncrementWarning):
            frame = make_frame()
        with native, frame:
            (expected,) = frame.run()
            assert isinstance(expected, LiftEstimate)
            for result in (native.run(), artifact.run()):
                (actual,) = result
                assert isinstance(actual, LiftEstimate)
                assert actual.require_lift().value == pytest.approx(
                    expected.require_lift().value, rel=1e-10
                )
                assert actual.require_lift().log_se == pytest.approx(
                    expected.require_lift().log_se, rel=1e-10
                )
    finally:
        artifact.close()
        con.disconnect()


@pytest.mark.parametrize("mixed", [False, True])
def test_artifact_bare_property_method_catalog_publish_adopt_run(tmp_path, mixed):
    from increment.estimation.results import LiftEstimate
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, *_ = _scalar_native_case(tmp_path, bare=True, mixed=mixed)
    artifact, _ = _publish_and_reopen(path, con, request_covariate=False)
    try:
        with Analysis.from_definitions("cov_test", path, con) as native:
            expected = {row.method: row for row in native.run()}
            joint = {row.method: row for row in artifact.run()}
            for method in methods:
                (alone,) = artifact.run(decision_method=method, sensitivity_methods=())
                for actual in (joint[method.name], alone):
                    oracle = expected[method.name]
                    assert isinstance(actual, LiftEstimate)
                    assert isinstance(oracle, LiftEstimate)
                    assert actual.require_lift().value == pytest.approx(
                        oracle.require_lift().value, rel=1e-10
                    )
                    assert actual.require_lift().log_se == pytest.approx(
                        oracle.require_lift().log_se, rel=1e-10
                    )
    finally:
        artifact.close()
        con.disconnect()


def test_artifact_implicit_and_named_scalar_families_match_standalone(tmp_path):
    import json

    from increment import Method
    from increment.estimation.results import LiftEstimate
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, *_ = _scalar_native_case(tmp_path)
    implicit = Method.cuped(name="implicit")
    named = methods[1]
    plan = {
        "primary": {
            "metric": "revenue",
            "decision_method": implicit.model_dump(mode="json", exclude_none=True),
            "sensitivity_methods": [named.model_dump(mode="json", exclude_none=True)],
        }
    }
    text = path.read_text()
    path.write_text(
        text[: text.index("    plan:")] + "    n_pre_periods: 30\n    plan: " + json.dumps(plan)
    )
    con.raw_sql(
        "INSERT INTO cov_events SELECT user_id, TIMESTAMP '2025-07-25', 'purchase', "
        "NULL, NULL, tenure, tenure, history_b, segment FROM cov_events WHERE tenure IS NOT NULL"
    )
    artifact, _ = _publish_and_reopen(path, con, request_covariate=False, full_catalog=True)
    try:
        joint = {
            row.method: row
            for row in artifact.run(decision_method=implicit, sensitivity_methods=(named,))
        }
        for method in (implicit, named):
            (alone,) = artifact.run(decision_method=method, sensitivity_methods=())
            actual = joint[method.name]
            assert isinstance(actual, LiftEstimate)
            assert isinstance(alone, LiftEstimate)
            assert actual.require_lift().value == pytest.approx(
                alone.require_lift().value, rel=1e-10
            )
            assert actual.require_lift().log_se == pytest.approx(
                alone.require_lift().log_se, rel=1e-10
            )
    finally:
        artifact.close()
        con.disconnect()


def test_artifact_named_scalar_breakout_refuses_before_query(tmp_path, monkeypatch):
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, *_ = _scalar_native_case(tmp_path)
    path.write_text(
        path.read_text().replace(
            "    start:", "    breakouts: [{property: segment, source: events}]\n    start:"
        )
    )
    artifact, _ = _publish_and_reopen(path, con, request_covariate=False, full_catalog=True)
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("artifact scalar breakout must refuse before querying evidence")

        monkeypatch.setattr(con, "to_pyarrow", forbidden)
        with pytest.raises(CodedError) as raised:
            artifact.run_breakout(decision_method=methods[1], sensitivity_methods=())
        assert raised.value.code == "artifact.cuped.scalar_breakout"
        assert raised.value.context["route"] == "from_unit_day_artifact.run_breakout"
    finally:
        artifact.close()
        con.disconnect()


@pytest.mark.parametrize("missing", ["zero", "impute", "refuse"])
def test_artifact_scalar_override_applies_effective_missing_preparation(tmp_path, missing):
    from increment import Method
    from increment.estimation.results import LiftEstimate
    from increment.semantics.models import CupedCovariateBinding

    path, con, *_ = covariate_defs_and_con(tmp_path)
    path.write_text(
        path.read_text().replace(
            "    plan: {secondaries: [revenue]}",
            "    plan: {primary: {metric: revenue, decision_method: "
            "{name: scalar, variance_reduction: cuped, covariates: "
            "[{name: pre_tenure, kind: continuous, missing: impute, "
            "property: tenure, source: events}]}}}",
        )
    )
    con.raw_sql("UPDATE cov_events SET tenure = NULL WHERE user_id IN ('u1', 'u12')")
    artifact, _ = _publish_and_reopen(path, con, request_covariate=False)
    requested = Method.cuped(
        name="scalar",
        covariates=CupedCovariateBinding(
            name="pre_tenure",
            kind="continuous",
            missing=missing,
            property="tenure",
            source="events",
        ),
    )
    path.write_text(path.read_text().replace("missing: impute", f"missing: {missing}"))
    try:
        with Analysis.from_definitions("cov_test", path, con) as standalone:
            (expected,) = standalone.run()
            (actual,) = artifact.run(decision_method=requested, sensitivity_methods=())
        assert isinstance(expected, LiftEstimate)
        assert isinstance(actual, LiftEstimate)
        if missing == "refuse":
            assert actual.lift is expected.lift is None
            assert (
                actual.failure_code
                == expected.failure_code
                == "frame.validation.metric_covariate_column"
            )
        else:
            assert actual.require_lift().value == pytest.approx(
                expected.require_lift().value, rel=1e-10
            )
            assert actual.require_lift().log_se == pytest.approx(
                expected.require_lift().log_se, rel=1e-10
            )
            if missing == "zero":
                (published,) = artifact.run()
                assert isinstance(published, LiftEstimate)
                assert actual.require_lift().log_se != pytest.approx(
                    published.require_lift().log_se, rel=1e-10
                )
    finally:
        artifact.close()
        con.disconnect()
