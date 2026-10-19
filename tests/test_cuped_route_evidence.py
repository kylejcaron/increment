from __future__ import annotations

from fractions import Fraction
from typing import Any, Literal, cast

import pyarrow as pa
import pytest

from increment import Analysis, AnalysisPlan, Method, MetricSpec
from increment.errors import CodedError
from increment.frame import CupedCovariate, from_unit_summary
from increment.readouts import run


def _records() -> list[dict[str, Any]]:
    return [
        {
            "unit": f"{arm}-{i}",
            "arm": arm,
            "day": 1,
            "exposure": 1,
            "segment": "all",
            "outcome": 15.0 + i % 17 + 0.4 * (i % 7) + 3.0 * (arm == "treatment"),
            "history_a": float(i % 7 + 1),
            "history_b": float((i * 5) % 13 + 1),
        }
        for arm in ("control", "treatment")
        for i in range(48)
    ]


def _scalar(name: str, column: str, *, missing: Literal["zero", "impute", "refuse"] = "zero"):
    return Method.cuped(
        name=name,
        covariates=CupedCovariate(name=column, kind="continuous", missing=missing),
    )


def _frame(records, methods, *, panel: bool, encouragement: bool = False):
    kwargs: dict[str, Any] = {"date": "day"} if panel else {}
    if encouragement:
        from increment import Encouragement, ExclusionRestriction, UptakeSpec

        kwargs.update(
            design=Encouragement(
                control_group="control",
                uptake=UptakeSpec(fact="uptake"),
                exclusion_restriction=ExclusionRestriction(
                    acknowledged=True, justification="assignment affects outcomes through uptake"
                ),
            ),
            uptake="uptake",
        )
    else:
        kwargs["control"] = "control"
    return (Analysis.from_unit_panel if panel else Analysis.from_unit_summary)(
        pa.Table.from_pylist(records),
        unit="unit",
        group="arm",
        metrics=[
            MetricSpec(name="outcome", decision_method=methods[0], sensitivity_methods=methods[1:])
        ],
        breakouts=["segment"],
        **kwargs,
    )


@pytest.mark.parametrize("panel", [False, True])
def test_frame_breakout_acquires_effective_scalar_binding(panel):
    first = _scalar("declared", "history_a")
    requested = _scalar("requested", "history_b")
    analysis = _frame(_records(), (first,), panel=panel)
    actual = analysis.run_breakout(decision_method=requested, sensitivity_methods=())[0]
    expected = _frame(_records(), (requested,), panel=panel).run()[0]
    assert actual.require_lift().value == pytest.approx(expected.require_lift().value, rel=1e-10)
    assert actual.require_lift().log_se == pytest.approx(expected.require_lift().log_se, rel=1e-10)
    assert actual.method == "requested"


@pytest.mark.parametrize("panel", [False, True])
@pytest.mark.parametrize("role", ["decision", "sensitivity"])
@pytest.mark.parametrize("missing", ["refuse", "impute"])
def test_frame_breakout_scalar_failure_is_method_local(panel, role, missing):
    records = [
        {**row, "history_a": None if missing == "impute" or i == 0 else row["history_a"]}
        for i, row in enumerate(_records())
    ]
    failed = _scalar("unavailable", "history_a", missing=missing)
    naive = Method.unadjusted()
    methods = (failed, naive) if role == "decision" else (naive, failed)
    result = _frame(records, methods, panel=panel).run_breakout()
    indexed = {row.method: row for row in result}
    expected = _frame(records, (naive,), panel=panel).run_breakout()[0]
    assert indexed["naive"].require_lift().value == pytest.approx(
        expected.require_lift().value, rel=1e-10
    )
    assert indexed["naive"].require_lift().log_se == pytest.approx(
        expected.require_lift().log_se, rel=1e-10
    )
    unavailable = indexed["unavailable"]
    assert unavailable.method_role == role
    assert unavailable.lift is None
    assert unavailable.failure_code == (
        "frame.validation.metric_covariate_column"
        if missing == "refuse"
        else "frame.validation.metric_covariate_impute_all_null"
    )
    assert result.metadata is not None
    record = next(cell for cell in result.metadata.cells if cell.cell.method == "unavailable")
    assert record.failure is not None and record.failure.code == unavailable.failure_code


@pytest.mark.parametrize("construction", [False, True])
def test_anonymous_moments_refuse_multiple_scalar_inputs(construction):
    from tests.test_analysis_moments import _legacy_fixed10_rows

    rows = _legacy_fixed10_rows(ref_x=10.0, cx1=1.0, cx2=20.0, cxy=4.0, x_role="covariate")
    first, second = _scalar("first", "history_a"), _scalar("second", "history_b")
    with pytest.raises(CodedError) as raised:
        if construction:
            Analysis.from_moments(
                rows,
                metrics=[
                    MetricSpec(name="revenue", decision_method=first, sensitivity_methods=(second,))
                ],
                control="control",
            )
        else:
            Analysis.from_moments(rows, metrics={"revenue": "mean"}, control="control").run(
                decision_method=first, sensitivity_methods=(second,)
            )
    assert raised.value.code == "facade.analysis_config.cuped_scalar_moments"


def _native_source(path, con):
    from increment.plan import compile_decision_plan
    from increment.query.native_source import DefinitionsMomentSource
    from increment.query.session import WarehouseSession
    from increment.semantics import load

    definitions = load(path)
    experiment = definitions.experiment("cov_test")
    assert experiment is not None
    design = experiment.resolved_design()
    return DefinitionsMomentSource(
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


def _artifact_source(path, con):
    from increment.query.artifact_contract import unit_day_artifact_extension_catalog
    from increment.query.artifact_publish import artifact_context
    from increment.query.session import WarehouseArtifactStore
    from increment.query.source import open_artifact
    from increment.semantics import load

    definitions = load(path)
    experiment = definitions.experiment("cov_test")
    assert experiment is not None
    context = artifact_context(definitions, experiment, "error")
    store = WarehouseArtifactStore(con, schema_name="artifacts")
    producer = Analysis.from_definitions("cov_test", path, con)
    ref = producer.publish_unit_day_artifact(
        store, extensions=[entry.request for entry in unit_day_artifact_extension_catalog(context)]
    )
    return open_artifact(store, ref, expected_context=context)


@pytest.mark.parametrize("ingress", ["native", "artifact"])
@pytest.mark.parametrize("transport", ["export", "moments_source"])
def test_single_cube_transports_carry_each_scalar_family(tmp_path, monkeypatch, ingress, transport):
    import pyarrow.parquet as pq

    from increment.estimation.results import LiftEstimate
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, *_ = _scalar_native_case(tmp_path)
    source = (_native_source if ingress == "native" else _artifact_source)(path, con)
    try:
        expected = {row.method: row for row in run(source) if isinstance(row, LiftEstimate)}
        monkeypatch.setattr(
            type(source),
            "unit_frame",
            lambda *args, **kwargs: pytest.fail("scalar family evidence never reads unit frames"),
        )
        if transport == "export":
            destination = tmp_path / "families.parquet"
            source.export_moments(destination)
            rows = pq.read_table(destination).to_pylist()
            assert {row["moments_format"] for row in rows} == {13}
            assert sum(row.get("record_kind") == "cuped_evidence" for row in rows) == 2
            restored = Analysis.from_moments(
                rows,
                metrics=[
                    MetricSpec(
                        name="revenue", decision_method=methods[0], sensitivity_methods=methods[1:]
                    )
                ],
                design=source.context.design,
                experiment_id="cov_test",
            )
            try:
                actual = restored.run()
            finally:
                restored.close()
        else:
            actual = run(
                source.moments_source(metrics=source.context.metrics, methods=list(methods))
            )
        replayed = {row.method: row for row in actual if isinstance(row, LiftEstimate)}
        assert replayed.keys() == expected.keys() == {"scalar_a", "companion_b"}
        for name, row in expected.items():
            assert (replayed[name].role, replayed[name].method_role) == (row.role, row.method_role)
            lift, reference = replayed[name].require_lift(), row.require_lift()
            for field in ("value", "log_se", "lb", "ub"):
                assert getattr(lift, field) == pytest.approx(
                    getattr(reference, field), rel=1e-10, abs=1e-12
                )
    finally:
        if ingress == "artifact":
            source.close()
        con.disconnect()


@pytest.mark.parametrize("field", ["property", "source", "measure"])
def test_artifact_scalar_rebinding_refuses_before_reading(tmp_path, monkeypatch, field):
    from increment.semantics.models import CupedCovariateBinding
    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, methods, *_ = _scalar_native_case(tmp_path)
    source = _artifact_source(path, con)
    changes = {"property": "history_b"} if field == "property" else {"source": "different"}
    if field == "measure":
        from increment.semantics.models import Measure

        changes = {
            "property": None,
            "source": None,
            "measure": Measure(fact="purchase", aggregation="sum"),
        }
    covariate = CupedCovariateBinding.model_validate(
        {**methods[0].covariates[0].model_dump(), **changes}
    )
    requested = Method.cuped(name="rebound", covariates=covariate)
    monkeypatch.setattr(
        con,
        "to_pyarrow",
        lambda *args, **kwargs: pytest.fail("binding refusal must precede evidence access"),
    )
    try:
        with pytest.raises(CodedError) as raised:
            run(source, decision_method=requested, sensitivity_methods=())
        assert raised.value.code == "readout.scalar_method_evidence"
        route = raised.value.context["route"]
        assert isinstance(route, str)
        assert route.endswith(".run")
        assert raised.value.context["method"] == "rebound"
    finally:
        source.close()
        con.disconnect()


@pytest.mark.parametrize("binding", ["property", "measure"])
def test_native_sequential_explicit_scalar_refuses_before_capture(tmp_path, monkeypatch, binding):
    import yaml

    from tests.test_cuped_covariate_sources import _scalar_native_case

    path, con, *_ = _scalar_native_case(tmp_path)
    definition = yaml.safe_load(path.read_text())
    experiment = definition["experiments"][0]
    experiment["allocation"] = {"control": 0.5, "treatment": 0.5}
    experiment["n_pre_periods"] = 30
    experiment["plan"]["primary"]["decision_method"]["name"] = "cuped"
    experiment["plan"]["primary"]["sensitivity_methods"] = []
    experiment["plan"]["inference"] = {"kind": "asymptotic_mean"}
    if binding == "measure":
        covariate = experiment["plan"]["primary"]["decision_method"]["covariates"][0]
        covariate.pop("property", None)
        covariate.pop("source", None)
        covariate["measure"] = {"fact": "purchase", "aggregation": "sum"}
    path.write_text(yaml.safe_dump(definition))
    monkeypatch.setattr(
        con,
        "to_pyarrow",
        lambda *args, **kwargs: pytest.fail("unsupported binding must refuse before capture"),
    )
    try:
        with pytest.raises(CodedError) as raised:
            Analysis.from_definitions("cov_test", path, con)
        assert raised.value.code == "sequential.source.invalid", raised.value
    finally:
        con.disconnect()


def _sequential_source(method):
    from increment.semantics.design import Randomized
    from increment.semantics.models import InferenceSpec, PredeclaredAdjustment

    return from_unit_summary(
        pa.Table.from_pylist(_records()),
        unit="unit",
        group="arm",
        control="control",
        exposure_date="exposure",
        metrics=[MetricSpec(name="outcome", decision_method=method)],
        design=Randomized(
            control_group="control",
            allocation={"control": Fraction(1, 2), "treatment": Fraction(1, 2)},
        ),
        plan=AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(
                kind="asymptotic_mean",
                adjustments=(
                    {
                        "outcome": PredeclaredAdjustment(
                            coefficient=Fraction(1, 2), center=Fraction(4)
                        )
                    }
                    if method.variance_reduction == "cuped"
                    else {}
                ),
            ),
        ),
    )


class _CaptureGuard:
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    def sequential_snapshot(self, *, previous=None):
        pytest.fail("changed scalar inputs must refuse before checkpoint capture")


@pytest.mark.parametrize("change", ["column", "missing"])
def test_sequential_scalar_overrides_must_match_registered_input(change):
    source = _sequential_source(_scalar("cuped", "history_a"))
    requested = _scalar(
        "cuped", "history_b" if change == "column" else "history_a", missing="impute"
    )
    with pytest.raises(CodedError) as raised:
        run(cast(Any, _CaptureGuard(source)), decision_method=requested)
    assert raised.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("scalar", [False, True])
def test_sequential_dispatch_preserves_custom_method_labels(scalar):
    method = (
        _scalar("adjusted_display", "history_a")
        if scalar
        else Method.unadjusted(name="raw_display")
    )
    source = _sequential_source(method)
    result = run(source)
    assert result
    assert {row.method for row in result} == {method.name}
    assert all(row.require_lift() is not None for row in result)


@pytest.mark.parametrize("panel", [False, True])
def test_frame_breakout_scalar_missing_is_local_to_segment(panel):
    records = [
        {
            **row,
            "segment": "healthy" if i % 2 else "unavailable",
            "history_a": None if i == 0 else row["history_a"],
        }
        for i, row in enumerate(_records())
    ]
    scalar = _scalar("adjusted", "history_a", missing="refuse")
    methods = (Method.unadjusted(), scalar)
    rows = _frame(records, methods, panel=panel).run_breakout()
    indexed = {(row.dimension_value, row.method): row for row in rows}
    assert len(indexed) == 4
    assert indexed[("unavailable", "adjusted")].failure_code == (
        "frame.validation.metric_covariate_column"
    )
    healthy = indexed[("healthy", "adjusted")]
    expected = _frame(
        [row for row in records if row["segment"] == "healthy"], (scalar,), panel=panel
    ).run_breakout()[0]
    assert healthy.require_lift().value == pytest.approx(expected.require_lift().value, rel=1e-10)
    assert healthy.require_lift().log_se == pytest.approx(expected.require_lift().log_se, rel=1e-10)
    assert all(
        indexed[(segment, "naive")].lift is not None for segment in ("healthy", "unavailable")
    )


@pytest.mark.parametrize("scalar", [False, True])
def test_sequential_matching_recipe_can_be_relabelled_without_new_state(scalar):
    registered = (
        _scalar("registered", "history_a") if scalar else Method.unadjusted(name="registered")
    )
    source = _sequential_source(registered)
    renamed = registered.model_copy(update={"name": "renamed"})
    companion = registered.model_copy(update={"name": "companion"})
    standalone = run(source, decision_method=renamed, sensitivity_methods=())
    together = run(source, decision_method=renamed, sensitivity_methods=(companion,))
    assert {row.method: row.method_role for row in together} == {
        "renamed": "decision",
        "companion": "sensitivity",
    }
    for row in together:
        assert row.require_lift() == standalone[0].require_lift()
        assert row.require_sequential_result() == standalone[0].require_sequential_result()


@pytest.mark.parametrize("name", ["iptw", "aipw", "dml"])
def test_sequential_reserved_observational_methods_remain_refused(name):
    with pytest.raises(CodedError) as raised:
        _sequential_source(Method(name=name))
    assert raised.value.code == "sequential.route.unsupported"


@pytest.mark.parametrize("panel", [False, True])
def test_encouragement_breakout_acquires_effective_scalar_binding(panel):
    records = [
        {**row, "uptake": int(i % 4 < (3 if row["arm"] == "treatment" else 1))}
        for i, row in enumerate(_records())
    ]
    declared, requested = _scalar("declared", "history_a"), _scalar("requested", "history_b")
    result = _frame(records, (declared,), panel=panel, encouragement=True).run_breakout(
        decision_method=requested, sensitivity_methods=()
    )
    result = [row for row in result if row.estimand == "itt"]
    expected = _frame(records, (requested,), panel=panel, encouragement=True).run(
        estimands=("itt",)
    )
    assert len(result) == len(expected) == 1
    assert result[0].require_lift().value == pytest.approx(
        expected[0].require_lift().value, rel=1e-10
    )
    assert result[0].require_lift().log_se == pytest.approx(
        expected[0].require_lift().log_se, rel=1e-10
    )


@pytest.mark.parametrize("panel", [False, True])
def test_encouragement_breakout_scalar_failure_keeps_naive_sibling(panel):
    records = [
        {
            **row,
            "uptake": int(i % 4 < (3 if row["arm"] == "treatment" else 1)),
            "history_a": None if i == 0 else row["history_a"],
        }
        for i, row in enumerate(_records())
    ]
    scalar, naive = _scalar("unavailable", "history_a", missing="refuse"), Method.unadjusted()
    result = _frame(records, (naive, scalar), panel=panel, encouragement=True).run_breakout()
    expected = next(
        row
        for row in _frame(records, (naive,), panel=panel, encouragement=True).run_breakout()
        if row.estimand == "itt"
    )
    indexed = {row.method: row for row in result if row.estimand == "itt"}
    assert indexed["naive"].require_lift() == expected.require_lift()
    assert indexed["unavailable"].lift is None
    assert indexed["unavailable"].failure_code == "frame.validation.metric_covariate_column"
