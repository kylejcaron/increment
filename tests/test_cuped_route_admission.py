from __future__ import annotations

import inspect
import pickle
from fractions import Fraction
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from increment import Analysis, Method
from increment.errors import CodedError
from increment.estimation.results import LiftEstimate
from increment.frame import MetricSpec
from increment.readouts import run
from increment.semantics.models import (
    AnalysisPlan,
    ExperimentMetric,
    InferenceSpec,
    PredeclaredAdjustment,
)
from tests.test_cuped_covariate_sources import _scalar_native_case
from tests.test_cuped_route_evidence import (
    _artifact_source,
    _native_source,
    _records,
    _scalar,
    _sequential_source,
)

_NON_SCALAR_OPERATIONS = {
    "allocation_history",
    "assignment_diagnostic",
    "breakout_summaries",
    "build_panel_for_metric",
    "capture_sequential",
    "close",
    "dashboard_breakout_reads",
    "dashboard_group_data",
    "dashboard_snapshot",
    "estimate_cate",
    "factor_summaries",
    "from_definitions",
    "from_switchback_panel",
    "from_unit_day_artifact",
    "health_check",
    "materialize",
    "panel_sql",
    "planning_baseline",
    "publish_unit_day_artifact",
    "run_asof",
    "run_daily",
    "select_targeting_rule",
    "sequential_snapshot",
    "sitewide",
    "srm",
    "summary_sql",
    "targeting_rule",
    "trigger_rates",
    "unit_counts",
    "validate_cate",
}
_SCALAR_EXECUTION_CASES = {
    "run": "readout",
    "run_breakout": "readout",
    "run_asof_lift": "readout",
    "run_daily_lift": "readout",
    "export": "export",
    "from_unit_summary": "registration",
    "from_unit_panel": "registration",
    "from_moments": "checkpoint",
}


def _scalar_operation_cases(kind=None):
    public_operations = {
        name
        for name, method in inspect.getmembers(Analysis)
        if not name.startswith("_") and (inspect.isfunction(method) or inspect.ismethod(method))
    }
    operations = public_operations - _NON_SCALAR_OPERATIONS
    missing = operations - _SCALAR_EXECUTION_CASES.keys()
    assert not missing, f"Public operations without an execution case: {sorted(missing)}"
    assert _SCALAR_EXECUTION_CASES.keys() <= operations
    assert set(_SCALAR_EXECUTION_CASES.values()) <= {
        "readout",
        "export",
        "registration",
        "checkpoint",
    }
    return tuple(
        name for name in sorted(operations) if kind is None or _SCALAR_EXECUTION_CASES[name] == kind
    )


def _native_case(tmp_path, *, decision=None, sensitivity=(), breakout=False):
    path, con, methods, *_ = _scalar_native_case(tmp_path)
    definition = yaml.safe_load(path.read_text())
    experiment = definition["experiments"][0]
    experiment["n_pre_periods"] = 30
    experiment["plan"]["primary"] = {
        "metric": "revenue",
        "decision_method": (decision or methods[0]).model_dump(mode="json", exclude_none=True),
        "sensitivity_methods": [
            method.model_dump(mode="json", exclude_none=True) for method in sensitivity
        ],
    }
    if breakout:
        experiment["breakouts"] = [{"property": "segment", "source": "events"}]
    path.write_text(yaml.safe_dump(definition))
    return path, con, methods


def test_artifact_single_scalar_export_replays_its_bound_family(tmp_path):
    path, con, methods = _native_case(tmp_path)
    source = _artifact_source(path, con)
    try:
        (expected,) = run(source)
        destination = tmp_path / "single-scalar.parquet"
        source.export_moments(destination)
        restored = Analysis.from_moments(
            pq.read_table(destination).to_pylist(),
            metrics=[MetricSpec(name="revenue", decision_method=methods[0])],
            design=source.context.design,
            experiment_id="cov_test",
        )
        (actual,) = restored.run()
        assert isinstance(actual, LiftEstimate)
        assert actual.require_lift().value == pytest.approx(
            expected.require_lift().value, rel=1e-10
        )
        assert actual.require_lift().log_se == pytest.approx(
            expected.require_lift().log_se, rel=1e-10
        )
    finally:
        source.close()
        con.disconnect()


@pytest.mark.parametrize("operation", ["run_daily_lift", "run_asof_lift"])
@pytest.mark.parametrize("dimension", [None, "segment"])
def test_facade_day_scalar_binding_refuses_before_reading(
    tmp_path, monkeypatch, operation, dimension
):
    path, con, methods = _native_case(tmp_path, breakout=True)
    analysis = Analysis.from_definitions("cov_test", path, con)
    reads = []

    def forbidden(*args, **kwargs):
        reads.append("evidence")
        pytest.fail("unsupported scalar request accessed day evidence")

    monkeypatch.setattr(con, "to_pyarrow", forbidden)
    try:
        with pytest.raises(CodedError) as raised:
            getattr(analysis, operation)(
                decision_method=methods[0], sensitivity_methods=(), dimension=dimension
            )
        assert raised.value.code == "readout.scalar_method_evidence", raised.value.context
        assert reads == []
    finally:
        analysis.close()
        con.disconnect()


@pytest.mark.parametrize("operation", ["run_daily_lift", "run_asof_lift"])
@pytest.mark.parametrize("panel", [False, True])
def test_day_axis_source_capability_refuses_before_scalar_admission(monkeypatch, operation, panel):
    from increment.frame import FramePanelSource, FrameTotalsSource

    scalar = _scalar("cuped", "history_a")
    analysis = (Analysis.from_unit_panel if panel else Analysis.from_unit_summary)(
        pa.Table.from_pylist(_records()),
        unit="unit",
        group="arm",
        control="control",
        metrics=[MetricSpec(name="outcome", decision_method=scalar)],
        **({"date": "day"} if panel else {}),
    )
    reads = []

    def unread(*args, **kwargs):
        reads.append("evidence")
        pytest.fail("source capability refusal accessed evidence")

    for source_type in (FrameTotalsSource, FramePanelSource):
        monkeypatch.setattr(source_type, "moments", unread)
        monkeypatch.setattr(source_type, "unit_frame", unread)
    try:
        with pytest.raises(CodedError) as raised:
            getattr(analysis, operation)()
        assert raised.value.code == (
            "estimation.cuped.arm_no_covariate" if panel else "facade.analysis.no_definitions"
        )
        assert reads == []
    finally:
        analysis.close()


@pytest.mark.parametrize("change", ["column", "missing"])
def test_restored_sequential_checkpoint_refuses_recipe_mismatch(tmp_path, change):
    source = _sequential_source(_scalar("cuped", "history_a"))
    destination = tmp_path / "checkpoint.parquet"
    source.export_moments(destination)
    requested = _scalar(
        "cuped", "history_b" if change == "column" else "history_a", missing="impute"
    )
    with pytest.raises(CodedError) as raised:
        Analysis.from_moments(
            pq.read_table(destination).to_pylist(),
            metrics=[MetricSpec(name="outcome", decision_method=requested)],
            design=source.context.design,
            experiment_id="frame",
        )
    assert raised.value.code == "sequential.source.invalid"


@pytest.mark.parametrize("change", ["column", "missing", "same"])
def test_explicit_checkpoint_plan_preserves_saved_scalar_recipe(tmp_path, monkeypatch, change):
    from increment.semantics.models import MethodSpec
    from increment.sources import MomentsSource

    original = _scalar("cuped", "history_a")
    source = _sequential_source(original)
    destination = tmp_path / "checkpoint.parquet"
    source.export_moments(destination)
    checkpoint = source.sequential_snapshot()
    requested = _scalar(
        "cuped",
        "history_b" if change == "column" else "history_a",
        missing="impute" if change == "missing" else "zero",
    )
    registration = source.context.plan.inference.registration
    assert registration is not None
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="outcome",
            decision_method=MethodSpec.model_validate(
                requested.model_dump(mode="json", exclude_none=True)
            ),
        ),
        inference=InferenceSpec(kind="asymptotic_mean", registration=registration),
    )
    rows = pq.read_table(destination).to_pylist()
    reads = []

    def unread(*args, **kwargs):
        reads.append(True)
        pytest.fail("a replacement recipe reached checkpoint adoption")

    if change == "same":
        restored = Analysis.from_moments(
            rows,
            metrics=[MetricSpec(name="outcome", decision_method=requested)],
            design=source.context.design,
            plan=plan,
        )
        assert restored.sequential_snapshot() == checkpoint
        (actual,) = restored.run()
        (expected,) = run(source)
        assert isinstance(actual, LiftEstimate)
        assert actual.require_lift().value == pytest.approx(
            expected.require_lift().value, rel=1e-10
        )
        assert actual.require_lift().log_se == pytest.approx(
            expected.require_lift().log_se, rel=1e-10
        )
    else:
        monkeypatch.setattr(MomentsSource, "adopt_sequential_snapshot", unread)
        monkeypatch.setattr(MomentsSource, "moments", unread)
        monkeypatch.setattr(MomentsSource, "unit_frame", unread)
        monkeypatch.setattr(MomentsSource, "unit_counts", unread)
        with pytest.raises(CodedError) as raised:
            Analysis.from_moments(
                rows,
                metrics=[MetricSpec(name="outcome", decision_method=requested)],
                design=source.context.design,
                plan=plan,
            )
        assert raised.value.code == "sequential.source.invalid"
        assert raised.value.context["route"] == "sequential.checkpoint_adoption"
    assert reads == []


def test_summary_breakout_refusing_input_precedes_outcome_deletion():
    scalar = _scalar("adjusted", "history_a", missing="refuse")
    records = [
        {
            **row,
            "segment": "unavailable" if i % 2 == 0 else "healthy",
            "outcome": None if i == 0 else row["outcome"],
            "history_a": None if i == 0 else row["history_a"],
        }
        for i, row in enumerate(_records())
    ]
    from increment.errors import IncrementWarning

    with pytest.warns(IncrementWarning):
        analysis = Analysis.from_unit_summary(
            pa.Table.from_pylist(records),
            unit="unit",
            group="arm",
            control="control",
            metrics=[
                MetricSpec(
                    name="outcome",
                    missing="drop",
                    decision_method=Method.unadjusted(),
                    sensitivity_methods=(scalar,),
                )
            ],
            breakouts=("segment",),
        )
    rows = {(row.dimension_value, row.method): row for row in analysis.run_breakout()}
    assert (
        rows[("unavailable", "adjusted")].failure_code == "frame.validation.metric_covariate_column"
    )
    assert rows[("healthy", "adjusted")].require_lift() is not None
    assert all(
        rows[(segment, "naive")].require_lift() is not None
        for segment in ("healthy", "unavailable")
    )


@pytest.mark.parametrize("role", ["decision", "sensitivity"])
@pytest.mark.parametrize("missing", ["refuse", "impute"])
def test_native_breakout_scalar_missing_is_method_and_segment_local(tmp_path, role, missing):
    from increment.semantics.models import CupedCovariateBinding

    scalar = Method.cuped(
        name="adjusted",
        covariates=CupedCovariateBinding(
            name="history_b",
            kind="continuous",
            missing=missing,
            property="history_b",
            source="events",
        ),
    )
    path, con, _ = _native_case(
        tmp_path,
        decision=scalar if role == "decision" else Method.unadjusted(),
        sensitivity=(Method.unadjusted(),) if role == "decision" else (scalar,),
        breakout=True,
    )
    con.raw_sql(
        "UPDATE cov_events SET segment = CASE WHEN tenure IS NULL OR CAST(tenure AS INTEGER) % 2 = 0 THEN 'unavailable' ELSE 'healthy' END"
    )
    if missing == "impute":
        con.raw_sql("UPDATE cov_events SET history_b = NULL")
    else:
        con.raw_sql("UPDATE cov_events SET history_b = NULL WHERE segment = 'unavailable'")
    try:
        with Analysis.from_definitions("cov_test", path, con) as analysis:
            rows = {(row.dimension_value, row.method): row for row in analysis.run_breakout()}
        expected_code = (
            "frame.validation.metric_covariate_column"
            if missing == "refuse"
            else "frame.validation.metric_covariate_impute_all_null"
        )
        assert rows[("unavailable", "adjusted")].failure_code == expected_code
        if missing == "refuse":
            assert rows[("healthy", "adjusted")].require_lift() is not None
        else:
            assert rows[("healthy", "adjusted")].failure_code == expected_code
        assert all(
            rows[(segment, "naive")].require_lift() is not None
            for segment in ("healthy", "unavailable")
        )
    finally:
        con.disconnect()


def test_explicit_frame_registration_refuses_multiple_scalar_inputs_before_frame_access():
    from increment._metric_specs import synthesise_metric
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id

    first = _scalar("cuped", "history_a")
    second = _scalar("other", "history_b")
    source = _sequential_source(first)
    spec = MetricSpec(name="outcome", decision_method=first, sensitivity_methods=(second,))
    registration = source.context.plan.inference.registration
    assert registration is not None
    registration = registration.model_copy(
        update={
            "definitions_id": sequential_definition_id(
                [synthesise_metric(spec)],
                source.context.design,
                transformations=(spec,),
                source_mapping=frame_observation_mapping(
                    unit="unit", group="arm", exposure_date="exposure"
                ),
            )
        }
    )
    with pytest.raises(CodedError) as raised:
        Analysis.from_unit_summary(
            cast(Any, object()),
            unit="unit",
            group="arm",
            exposure_date="exposure",
            design=source.context.design,
            metrics=[spec],
            plan=AnalysisPlan(
                primary="outcome",
                inference=InferenceSpec(kind="asymptotic_mean", registration=registration),
            ),
        )
    assert raised.value.code == "sequential.source.invalid"
    assert raised.value.context["route"] == "sequential.registration"


@pytest.mark.parametrize("roundtrip", ["pickle", "decision_wire", "yaml"])
def test_explicit_raw_procedure_survives_serialized_roundtrip(tmp_path, roundtrip):
    from increment._analysis_config import method_from_spec
    from increment.decision import RelativeArmDecisionProcedure
    from increment.decision_wire import compiled_plan_from_json, compiled_plan_to_json
    from increment.semantics import load

    method = Method.unadjusted(name="companion")
    assert method.model_dump(mode="json")["procedure"] == "unadjusted"
    if roundtrip == "pickle":
        restored = pickle.loads(pickle.dumps(method))
    elif roundtrip == "decision_wire":
        plan = _sequential_source(method).context.plan
        procedure = compiled_plan_from_json(compiled_plan_to_json(plan)).procedures["outcome"]
        assert isinstance(procedure, RelativeArmDecisionProcedure)
        restored = procedure.decision_method
    else:
        path, con, _ = _native_case(tmp_path, decision=method)
        try:
            definitions = load(path)
            assert definitions.experiments[0].plan is not None
            declaration = definitions.experiments[0].plan.primary
            assert isinstance(declaration, ExperimentMetric)
            assert declaration.decision_method is not None
            restored = method_from_spec(declaration.decision_method)
        finally:
            con.disconnect()
    assert restored == method
    assert {row.method for row in run(_sequential_source(restored))} == {"companion"}


def test_analysis_method_operation_inventory_is_complete():
    operations = {
        name
        for name, method in inspect.getmembers(Analysis, predicate=inspect.isfunction)
        if not name.startswith("_")
        and {"decision_method", "sensitivity_methods"}
        <= inspect.signature(method).parameters.keys()
    }
    assert operations == {"run", "run_breakout", "run_asof_lift", "run_daily_lift"}
    assert operations <= set(_scalar_operation_cases("readout"))
    assert set(_scalar_operation_cases()) == _SCALAR_EXECUTION_CASES.keys()


def test_new_public_operation_requires_an_execution_case(monkeypatch):
    monkeypatch.setattr(Analysis, "future_evidence", Analysis.unit_counts, raising=False)
    with pytest.raises(AssertionError):
        _scalar_operation_cases()


@pytest.fixture
def scalar_probe(monkeypatch):
    from increment import _analysis_config

    reads = []
    admissions = []
    original_admission = _analysis_config.validate_scalar_route

    def admit(*args, **kwargs):
        admissions.append(kwargs["route"])
        return original_admission(*args, **kwargs)

    def unread(*args, **kwargs):
        reads.append(True)
        pytest.fail("evidence was read before scalar input validation")

    monkeypatch.setattr(_analysis_config, "validate_scalar_route", admit)
    return admissions, reads, unread


@pytest.mark.parametrize("operation", _scalar_operation_cases("readout"))
def test_public_scalar_operations_refuse_before_evidence(
    tmp_path, monkeypatch, scalar_probe, operation
):
    from increment.frame import from_unit_summary
    from increment.sources import MomentsSource

    admissions, reads, unread = scalar_probe
    first = _scalar("first", "history_a")
    second = _scalar("second", "history_b")
    if operation == "run":
        source = from_unit_summary(
            pa.Table.from_pylist(_records()),
            unit="unit",
            group="arm",
            control="control",
            metrics=[MetricSpec(name="outcome", decision_method=first)],
        )
        path = tmp_path / "legacy.parquet"
        source.export_moments(path)
        rows = pq.read_table(path).to_pylist()
        analysis = Analysis.from_moments(
            rows, control="control", metrics=[MetricSpec(name="outcome")]
        )
        monkeypatch.setattr(MomentsSource, "moments", unread)
        monkeypatch.setattr(MomentsSource, "unit_counts", unread)
        admissions.clear()
        with pytest.raises(CodedError) as raised:
            analysis.run(decision_method=first, sensitivity_methods=(second,))
        assert raised.value.code == "facade.analysis_config.cuped_scalar_moments"
    else:
        path, con, _ = _native_case(
            tmp_path, sensitivity=(second,), breakout=operation == "run_breakout"
        )
        try:
            analysis = Analysis.from_definitions("cov_test", path, con, store="none")
            invoke = getattr(analysis, operation)
            monkeypatch.setattr(con, "to_pyarrow", unread)
            admissions.clear()
            with pytest.raises(CodedError) as raised:
                invoke()
            expected = (
                "facade.analysis_config.multifamily_scalar_breakout"
                if operation == "run_breakout"
                else "readout.scalar_method_evidence"
            )
            assert raised.value.code == expected
        finally:
            con.disconnect()
    assert admissions
    assert reads == []


@pytest.mark.parametrize("operation", _scalar_operation_cases("export"))
def test_public_export_carries_each_scalar_family(tmp_path, operation):
    second = _scalar("second", "history_b")
    path, con, methods = _native_case(tmp_path, sensitivity=(second,))
    destination = tmp_path / "families.parquet"
    try:
        with Analysis.from_definitions("cov_test", path, con, store="none") as analysis:
            expected = {row.method: row for row in analysis.run() if isinstance(row, LiftEstimate)}
            getattr(analysis, operation)(destination)
        design = _native_source(path, con).context.design
        rows = pq.read_table(destination).to_pylist()
        assert {row["moments_format"] for row in rows} == {13}
        restored = Analysis.from_moments(
            rows,
            metrics=[
                MetricSpec(
                    name="revenue", decision_method=methods[0], sensitivity_methods=(second,)
                )
            ],
            design=design,
            experiment_id="cov_test",
        )
        try:
            replayed = {row.method: row for row in restored.run() if isinstance(row, LiftEstimate)}
        finally:
            restored.close()
        assert replayed.keys() == expected.keys() == {methods[0].name, "second"}
        for name, row in expected.items():
            assert replayed[name].require_lift().value == pytest.approx(
                row.require_lift().value, rel=1e-10
            )
            assert replayed[name].require_lift().log_se == pytest.approx(
                row.require_lift().log_se, rel=1e-10
            )
    finally:
        con.disconnect()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("operation", _scalar_operation_cases("registration"))
@pytest.mark.parametrize("ingress", ["facade", "source"])
def test_frame_registration_refuses_before_evidence(
    monkeypatch, scalar_probe, explicit, operation, ingress
):
    import narwhals as nw

    from increment._metric_specs import synthesise_metric
    from increment.frame import from_unit_panel, from_unit_summary
    from increment.sequential_source import frame_observation_mapping, sequential_definition_id

    panel = operation == "from_unit_panel"
    admissions, reads, unread = scalar_probe
    first = _scalar("first", "history_a")
    second = _scalar("second", "history_b")
    source = _sequential_source(first)
    spec = MetricSpec(name="outcome", decision_method=first, sensitivity_methods=(second,))
    if explicit:
        registration = source.context.plan.inference.registration
        assert registration is not None
        registration = registration.model_copy(
            update={
                "definitions_id": sequential_definition_id(
                    [synthesise_metric(spec)],
                    source.context.design,
                    transformations=(spec,),
                    source_mapping=frame_observation_mapping(
                        unit="unit",
                        group="arm",
                        exposure_date="exposure",
                        date="day" if panel else None,
                    ),
                )
            }
        )
        inference = InferenceSpec(kind="asymptotic_mean", registration=registration)
    else:
        inference = InferenceSpec(
            kind="asymptotic_mean",
            adjustments={
                "outcome": PredeclaredAdjustment(coefficient=Fraction(1, 2), center=Fraction(4))
            },
        )
    if ingress == "facade":
        constructor = getattr(Analysis, operation)
    else:
        constructor = from_unit_panel if panel else from_unit_summary
    admissions.clear()
    monkeypatch.setattr(nw, "from_native", unread)
    with pytest.raises(CodedError) as raised:
        constructor(
            cast(Any, object()),
            unit="unit",
            group="arm",
            exposure_date="exposure",
            design=source.context.design,
            metrics=[spec],
            plan=AnalysisPlan(primary="outcome", inference=inference),
            **({"date": "day"} if panel else {}),
            **({"control": "control"} if ingress == "source" else {}),
        )
    assert raised.value.code == "sequential.source.invalid"
    assert admissions
    assert reads == []


@pytest.mark.parametrize("operation", _scalar_operation_cases("checkpoint"))
def test_checkpoint_adoption_refuses_before_evidence(
    tmp_path, monkeypatch, scalar_probe, operation
):
    from increment.sources import MomentsSource

    admissions, reads, unread = scalar_probe
    first = _scalar("first", "history_a")
    second = _scalar("second", "history_b")
    source = _sequential_source(first)
    path = tmp_path / "checkpoint.parquet"
    source.export_moments(path)
    rows = pq.read_table(path).to_pylist()
    monkeypatch.setattr(MomentsSource, "adopt_sequential_snapshot", unread)
    admissions.clear()
    with pytest.raises(CodedError) as raised:
        getattr(Analysis, operation)(
            rows,
            metrics=[MetricSpec(name="outcome", decision_method=second)],
            design=source.context.design,
        )
    assert raised.value.code == "sequential.source.invalid"
    assert admissions
    assert reads == []
