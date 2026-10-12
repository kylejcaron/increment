"""Boundary tests for the substrate-free moment-source seam.

``increment.sources`` defines a protocol that data sources implement without
depending on any particular query engine or dataframe library - these tests
enforce that boundary and pin the shape of the errors it raises.
"""

from __future__ import annotations

import math
import pickle
import subprocess
import sys
from collections.abc import Mapping

import pytest
from pydantic import ValidationError


def test_sources_module_imports_no_engine():
    code = (
        "import sys; import increment.sources; "
        "assert 'ibis' not in sys.modules; "
        "assert not any(name == 'increment.query' or name.startswith('increment.query.') "
        "for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _source_for_refusal():
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")

    return MomentsSource(
        [
            _format3_row(
                winsor_upper_percentile=None,
                winsor_upper_bound=None,
                winsor_n_upper=0,
                winsor_n=0,
            )
        ],
        metrics=[metric],
        study_id="exp",
    ), metric


def test_capability_error_names_the_fix():
    from increment.errors import CapabilityError

    source, metric = _source_for_refusal()
    with pytest.raises(CapabilityError) as raised:
        source.moments(metric, grain="daily")

    assert raised.value.code == "source.moments.grain"
    assert raised.value.context["grain"] == "daily"
    assert raised.value.context["offered"] == frozenset({"total"})


def test_arm_source_context_construction_does_not_import_upward(monkeypatch):
    import builtins
    from dataclasses import replace

    source, _ = _source_for_refusal()
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(
            (
                "increment.decision",
                "increment.sources",
                "increment.semantics",
                "increment._analysis_config",
            )
        ):
            pytest.fail(f"SourceContext construction imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    context = replace(source.context, study_id="copy")
    assert context.study_id == "copy"
    assert context.plan is source.context.plan
    assert not hasattr(context, "contrast_references")


def test_source_refusal_context_and_message_are_local():
    from increment.errors import CapabilityError

    source, metric = _source_for_refusal()
    with pytest.raises(CapabilityError) as raised:
        source.unit_frame(metric)

    assert raised.value.code == "source.moments.unit_grain"
    assert raised.value.context["method"] == "unit_frame"


def test_moments_source_names_covariate_refusal_distinctly():
    from increment.errors import CapabilityError

    source, metric = _source_for_refusal()
    with pytest.raises(CapabilityError) as raised:
        source.unit_frame(metric, covariates=["tenure"])
    assert raised.value.code == "source.moments.covariate_unavailable"
    assert raised.value.context["covariates"] == ("tenure",)


@pytest.mark.parametrize(
    ("operation", "expected_code", "expected_context"),
    [
        (
            lambda source, metric: source.moments(metric, grain="daily"),
            "source.moments.grain",
            {"grain": "daily", "offered": frozenset({"total"})},
        ),
        (
            lambda source, metric: source.unit_frame(metric),
            "source.moments.unit_grain",
            {"method": "unit_frame"},
        ),
        (
            lambda source, metric: source.sql(),
            "source.moments.sql",
            {},
        ),
    ],
)
def test_moments_source_refusals_are_coded_and_actionable(
    operation, expected_code, expected_context
):
    from increment.errors import CapabilityError

    source, metric = _source_for_refusal()
    with pytest.raises(CapabilityError) as raised:
        operation(source, metric)

    assert raised.value.code == expected_code
    assert expected_context.items() <= raised.value.context.items()


def test_moved_errors_are_not_owner_module_exports():
    import increment.semantics.loader as loader
    import increment.sources as sources

    assert not hasattr(sources, "CapabilityError")
    assert not hasattr(loader, "DefinitionError")


def test_randomized_is_frozen_and_discriminated():
    from increment.semantics.design import Randomized

    d = Randomized(control_group="control")

    assert d.mechanism == "randomized"

    with pytest.raises(ValidationError):
        d.control_group = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


def _format3_row(**extra):
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    metric_name = str(extra.get("metric", "revenue"))
    wire_plan = compiled_plan_to_json(
        compile_decision_plan(
            None,
            [MeanMetric(name=metric_name, entity="user", fact=metric_name)],
        )
    )
    return {
        "experiment_id": "exp",
        "metric": "revenue",
        "group_id": "control",
        "n": 10,
        "successes": None,
        "ref_y": 5.0,
        "cy1": 0.0,
        "cy2": 1.0,
        **dict.fromkeys(
            (
                "ref_x",
                "cx1",
                "cx2",
                "cxy",
                "ref_den",
                "cden1",
                "cden2",
                "cyden",
                "sum_d",
                "cyd",
                "cy2d",
                "cxd",
                "x_role",
            ),
            None,
        ),
        "winsor_lower_percentile": None,
        "winsor_upper_percentile": 0.99,
        "winsor_lower_bound": None,
        "winsor_upper_bound": 100.0,
        "winsor_n": 10,
        "winsor_n_lower": 0,
        "winsor_n_upper": 1,
        "moments_format": 11,
        "decision_plan": wire_plan,
        **extra,
    }


def _plain_format_row(**extra):
    return _format3_row(
        winsor_upper_percentile=None,
        winsor_upper_bound=None,
        winsor_n=None,
        winsor_n_lower=None,
        winsor_n_upper=None,
        **extra,
    )


def test_moments_source_preserves_format3_winsorization_metadata():
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(
        name="revenue",
        entity="user",
        fact="revenue",
        winsorization={"upper_value": 100.0},
    )
    source = MomentsSource(
        [_format3_row(winsor_upper_percentile=None)],
        metrics=[metric],
        study_id="exp",
    )

    loaded = source.moments(metric)[0]
    assert loaded["winsor_upper_bound"] == 100.0
    assert loaded["winsor_n_upper"] == 1
    assert "moments_format" not in loaded


def test_moments_source_resolves_plan_on_the_frame_path_like_from_unit_summary():
    """MomentsSource (reached via Analysis.from_moments) resolves a
    declared plan the same way FrameTotalsSource (from_unit_summary) does:
    every declared metric not named primary/guardrail defaults to
    role="secondary" and is BH-eligible, not the warehouse path's
    role="unassigned"."""
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import MomentsSource

    metrics = [
        MeanMetric(name="revenue", entity="user", fact="revenue"),
        MeanMetric(name="orders", entity="user", fact="orders"),
    ]
    source = MomentsSource(
        [
            _format3_row(
                metric="revenue",
                winsor_upper_percentile=None,
                winsor_upper_bound=None,
                winsor_n_upper=0,
                winsor_n=0,
            ),
            _format3_row(
                metric="orders",
                winsor_upper_percentile=None,
                winsor_upper_bound=None,
                winsor_n_upper=0,
                winsor_n=0,
            ),
        ],
        metrics=metrics,
        study_id="exp",
        plan=AnalysisPlan(primary="revenue"),
    )

    assert source.context.plan.procedures["revenue"].role == "primary"
    assert source.context.plan.procedures["orders"].role == "secondary"
    assert source.context.plan.procedures["orders"].family.member is True  # ty: ignore[unresolved-attribute]


def test_moments_source_accepts_resolved_percentile_winsorization_metadata():
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(
        name="revenue",
        entity="user",
        fact="revenue",
        winsorization={"upper_percentile": 0.99},
    )
    source = MomentsSource([_format3_row()], metrics=[metric], study_id="exp")

    loaded = source.moments(metric)[0]
    assert loaded["winsor_upper_percentile"] == 0.99
    assert loaded["winsor_upper_bound"] == 100.0


@pytest.mark.parametrize("bound", [float("nan"), float("inf"), -float("inf")])
def test_moments_source_rejects_nonfinite_resolved_percentile_bound(bound):
    from increment.errors import CapabilityError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(
        name="revenue",
        entity="user",
        fact="revenue",
        winsorization={"upper_percentile": 0.99},
    )
    with pytest.raises(CapabilityError) as raised:
        MomentsSource(
            [_format3_row(winsor_upper_bound=bound)],
            metrics=[metric],
            study_id="exp",
        )
    assert raised.value.context["metric"] == "revenue"
    assert raised.value.context["side"] == "upper"
    if math.isnan(bound):
        actual = raised.value.context["actual_bound"]
        assert isinstance(actual, float)
        assert math.isnan(actual)
    else:
        assert raised.value.context["actual_bound"] == bound
    restored = pickle.loads(pickle.dumps(raised.value))
    assert restored.code == raised.value.code
    assert restored.context.keys() == raised.value.context.keys()


def test_format3_rejects_incomplete_winsorization_metadata():
    from increment.errors import WireFormatError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    row = _format3_row()
    del row["winsor_n_upper"]
    with pytest.raises(WireFormatError) as raised:
        MomentsSource([row], metrics=[metric], study_id="exp")
    assert raised.value.code == "moments.v7_rows_missing_winsorization_fields"


def test_reject_duplicate_keys_carries_code():
    from increment.errors import WireFormatError
    from increment.sources import _reject_duplicate_keys

    with pytest.raises(WireFormatError) as raised:
        _reject_duplicate_keys([("group", 1), ("group", 2)])
    assert raised.value.code == "moments.duplicate_key"
    assert raised.value.context["key"] == "group"


def test_format3_rejects_null_metadata_for_configured_winsorization():
    from increment.errors import CapabilityError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(
        name="revenue",
        entity="user",
        fact="revenue",
        winsorization={"upper_value": 100.0},
    )
    row = _format3_row(
        winsor_upper_percentile=None,
        winsor_upper_bound=None,
        winsor_n=None,
        winsor_n_upper=None,
    )
    with pytest.raises(CapabilityError) as raised:
        MomentsSource([row], metrics=[metric], study_id="exp")
    assert raised.value.context["metric"] == "revenue"
    assert raised.value.context["side"] == "upper"


def test_format3_rejects_metadata_for_undeclared_winsorization():
    from increment.errors import CapabilityError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    with pytest.raises(CapabilityError) as raised:
        MomentsSource([_format3_row()], metrics=[metric], study_id="exp")
    assert raised.value.context == {
        "metric": "revenue",
        "field": "winsor_upper_percentile",
        "value": 0.99,
    }


def test_source_operations_replace_warehouse_native_switch():
    """Native-only facade routes use a declared operation, not a marker flag."""
    from increment.query.native_source import DefinitionsMomentSource
    from increment.sources import MomentsSource

    assert not hasattr(DefinitionsMomentSource, "warehouse_native")
    assert not hasattr(MomentsSource, "warehouse_native")
    assert "moments_source" in DefinitionsMomentSource.operations
    assert "moments_source" not in MomentsSource.operations


def test_compiled_plan_wire_round_trips_guardrail_and_view_policy():
    from increment.decision_wire import compiled_plan_from_json, compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, MeanMetric, MultiplicitySpec

    metrics = [
        MeanMetric(name="revenue", entity="user", fact="revenue"),
        MeanMetric(
            name="latency",
            entity="user",
            fact="latency",
            preferred_direction="decrease",
        ),
    ]
    plan = AnalysisPlan(
        primary="revenue",
        guardrails=["latency"],
        view_multiplicity=MultiplicitySpec(correction="bonferroni"),
    )
    compiled = compile_decision_plan(plan, metrics, path="warehouse")
    restored = compiled_plan_from_json(compiled_plan_to_json(compiled))
    assert restored.procedures["latency"].role == "guardrail"
    assert restored.procedures["latency"].alternative == "less"
    assert restored.view_policies.randomized_breakout.correction == "bonferroni"


def test_runtime_method_cannot_cross_compiled_plan_wire():
    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import WireFormatError
    from increment.estimation.engine import Method
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    compiled = compile_decision_plan(
        None,
        [metric],
        methods=[Method(name="iptw", propensity_learner=lambda: None)],
    )
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_to_json(compiled)
    assert raised.value.code == "wire.procedure.runtime_method"


def test_runtime_prior_cannot_cross_compiled_plan_wire():
    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import WireFormatError
    from increment.estimation.priors import StudentTPrior
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    compiled = compile_decision_plan(None, [metric], prior=StudentTPrior(nu=5, scale=1))
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_to_json(compiled)
    assert raised.value.code == "wire.procedure.runtime_prior"


@pytest.mark.parametrize("received", [1, 2, 3, 4, 5, 6, 7, 8])
def test_legacy_moments_format_is_refused_with_stable_context(received):
    from increment.errors import WireFormatError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    with pytest.raises(WireFormatError) as raised:
        MomentsSource(
            [_format3_row(moments_format=received)],
            metrics=[MeanMetric(name="revenue", entity="user", fact="revenue")],
            study_id="exp",
        )
    assert raised.value.code == "moments.format.unsupported_legacy"
    assert raised.value.context["received"] == received
    assert raised.value.context["required"] == 12


def test_future_moments_format_is_refused():
    from increment.errors import WireFormatError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    with pytest.raises(WireFormatError) as raised:
        MomentsSource(
            [_format3_row(moments_format=13)],
            metrics=[MeanMetric(name="revenue", entity="user", fact="revenue")],
            study_id="exp",
        )
    assert raised.value.code == "moments.format.unsupported_future"


def test_v7_plan_payload_refuses_partial_conflicting_and_mixed_rows():
    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import WireFormatError
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    plan = compiled_plan_to_json(compile_decision_plan(None, [metric]))
    other = compiled_plan_to_json(
        compile_decision_plan(
            None,
            [metric],
            methods=[],
        )
    )
    first = _format3_row(decision_plan=plan)
    second = _format3_row(decision_plan=plan)
    second.pop("decision_plan")
    with pytest.raises(WireFormatError) as partial:
        MomentsSource([first, second], metrics=[metric], study_id="exp")
    assert partial.value.context["field"] == "decision_plan"
    details = partial.value.context["value"]
    assert isinstance(details, Mapping)
    assert ("unplanned_rows", 1) in details.items()

    with pytest.raises(WireFormatError) as conflict:
        MomentsSource(
            [_format3_row(decision_plan=plan), _format3_row(decision_plan=other)],
            metrics=[metric],
            study_id="exp",
        )
    assert conflict.value.context["field"] == "decision_plan"
    plans = conflict.value.context["value"]
    assert isinstance(plans, tuple)
    assert set(plans) == {plan, other}

    with pytest.raises(WireFormatError) as mixed:
        MomentsSource(
            [_format3_row(decision_plan=plan), _format3_row(decision_plan=plan, moments_format=6)],
            metrics=[metric],
            study_id="exp",
        )
    assert mixed.value.code == "moments.format.mixed"


def test_empty_methods_roundtrip_preserves_readout():
    from increment import readouts
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    plan = compile_decision_plan(None, [metric], methods=[])
    payload = compiled_plan_to_json(plan)
    common = {
        "metric": "revenue",
        "winsor_upper_percentile": None,
        "winsor_upper_bound": None,
        "winsor_n": None,
        "winsor_n_lower": None,
        "winsor_n_upper": None,
        "decision_plan": payload,
    }
    rows = [
        _format3_row(group_id="control", ref_y=10.0, **common),
        _format3_row(group_id="treatment", ref_y=11.0, **common),
    ]
    before = MomentsSource(
        rows,
        metrics=[metric],
        study_id="exp",
        design=Randomized(control_group="control"),
        plan=plan,
    )
    after = MomentsSource(
        rows,
        metrics=[metric],
        study_id="exp",
        design=Randomized(control_group="control"),
    )
    assert readouts.run(before) == readouts.run(after) == []


def test_per_metric_prior_scope_roundtrip():
    from dataclasses import replace

    from increment._analysis_config import effective_methods
    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import CodedError
    from increment.estimation.adjust import _validate_global_prior_method_scales
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, ExperimentMetric, MeanMetric
    from increment.sources import MomentsSource

    metrics = [
        MeanMetric(name="m1", entity="user", fact="m1"),
        MeanMetric(name="m2", entity="user", fact="m2"),
    ]
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="m1",
            decision_method={"name": "unadjusted"},
            prior={"mu": 0.0, "sigma": 0.5},
        ),
        secondaries=[
            ExperimentMetric(
                metric="m2",
                decision_method={"name": "iptw"},
                prior={"mu": 0.0, "sigma": 0.5},
            )
        ],
    )
    compiled = compile_decision_plan(plan, metrics, methods=None, prior=None)
    payload = compiled_plan_to_json(compiled)
    rows = [
        _format3_row(
            metric=name,
            decision_plan=payload,
            winsor_upper_percentile=None,
            winsor_upper_bound=None,
            winsor_n=None,
            winsor_n_lower=None,
            winsor_n_upper=None,
        )
        for name in ("m1", "m2")
    ]
    source = MomentsSource(rows, metrics=metrics, study_id="exp")
    configs = source.context.configs
    methods = tuple(effective_methods(config, design=source.context.design) for config in configs)
    _validate_global_prior_method_scales(configs, methods)
    with pytest.raises(CodedError) as raised:
        _validate_global_prior_method_scales(
            tuple(replace(config, prior_is_global=True) for config in configs), methods
        )
    assert raised.value.code == "estimation.adjust.prior.method_scale"


def test_embedded_plan_missing_metric_refuses():
    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import WireFormatError
    from increment.plan import compile_decision_plan
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    revenue = MeanMetric(name="revenue", entity="user", fact="revenue")
    orders = MeanMetric(name="orders", entity="user", fact="orders")
    payload = compiled_plan_to_json(compile_decision_plan(None, [revenue]))
    with pytest.raises(WireFormatError) as raised:
        MomentsSource(
            [
                _plain_format_row(metric="revenue", decision_plan=payload),
                _plain_format_row(metric="orders", decision_plan=payload),
            ],
            metrics=[revenue, orders],
            study_id="exp",
        )
    assert raised.value.context["field"] == "decision_plan"
    assert raised.value.context["value"] == ("orders",)


def test_all_planless_v7_refuses():
    from increment.errors import WireFormatError
    from increment.semantics.models import MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    rows = [_plain_format_row(), _plain_format_row(group_id="treatment")]
    for row in rows:
        row.pop("decision_plan")
    with pytest.raises(WireFormatError) as raised:
        MomentsSource(rows, metrics=[metric], study_id="exp")
    assert raised.value.code == "moments.plan.invalid"


def test_explicit_plan_overrides_embedded_plan():
    from increment.decision_wire import compiled_plan_to_json
    from increment.plan import compile_decision_plan
    from increment.semantics.models import AnalysisPlan, MeanMetric
    from increment.sources import MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    embedded = compiled_plan_to_json(compile_decision_plan(None, [metric]))
    source = MomentsSource(
        [_plain_format_row(decision_plan=embedded)],
        metrics=[metric],
        study_id="exp",
        plan=AnalysisPlan(primary="revenue"),
        path="warehouse",
    )
    assert source.context.plan.declared is True
    assert source.context.plan.procedures["revenue"].role == "primary"


def test_frame_plan_matches_configs_and_wire_refuses_runtime_learner():
    import pyarrow as pa

    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import WireFormatError
    from increment.estimation.engine import Method
    from increment.estimation.inference import Normal
    from increment.frame import from_unit_summary

    spec = {
        "name": "metric",
        "covariate": "x",
        "decision_method": Method(name="cuped", variance_reduction="cuped"),
        "prior": Normal(mu=0.0, sigma=0.5),
    }
    source = from_unit_summary(
        pa.table(
            {
                "unit": ["u0", "u1", "u2", "u3"],
                "variant": ["control", "control", "treatment", "treatment"],
                "metric": [1.0, 2.0, 2.0, 3.0],
                "x": [0.0, 1.0, 0.0, 1.0],
            }
        ),
        unit="unit",
        group="variant",
        control="control",
        metrics=[spec],
    )
    procedure = source.context.plan.procedures["metric"]
    assert procedure.decision_method.name == "cuped"  # ty: ignore[unresolved-attribute]
    assert procedure.prior is not None  # ty: ignore[unresolved-attribute]

    runtime_spec = dict(spec, decision_method=Method(name="iptw", propensity_learner=lambda: None))
    runtime_source = from_unit_summary(
        pa.table(
            {
                "unit": ["u0", "u1", "u2", "u3"],
                "variant": ["control", "control", "treatment", "treatment"],
                "metric": [1.0, 2.0, 2.0, 3.0],
                "x": [0.0, 1.0, 0.0, 1.0],
            }
        ),
        unit="unit",
        group="variant",
        control="control",
        metrics=[runtime_spec],
    )
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_to_json(runtime_source.context.plan)
    assert raised.value.code == "wire.procedure.runtime_method"


def test_assignment_counts_stable_under_row_order():
    import json

    from increment.semantics.models import MeanMetric
    from increment.sources import ASSIGNMENT_COUNTS_FIELD, MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    counts = json.dumps({"control": 10, "treatment": 12}, sort_keys=True)
    rows = [
        _plain_format_row(group_id="control", **{ASSIGNMENT_COUNTS_FIELD: counts}),
        _plain_format_row(group_id="treatment", **{ASSIGNMENT_COUNTS_FIELD: counts}),
    ]
    first = MomentsSource(rows, metrics=[metric], study_id="exp")
    second = MomentsSource(list(reversed(rows)), metrics=[metric], study_id="exp")
    assert first.unit_counts() == second.unit_counts() == {"control": 10, "treatment": 12}


def test_assignment_counts_rejects_forged_duplicate_keys():
    """A cube row's assignment_counts JSON with a duplicate object key is a
    forged/corrupted wire payload, not a value to silently decode last-wins:
    it must fail the same trustworthy-counts refusal as any other malformed
    payload, never resolve to the duplicated value."""
    from increment.errors import CodedError
    from increment.semantics.models import MeanMetric
    from increment.sources import ASSIGNMENT_COUNTS_FIELD, MomentsSource

    metric = MeanMetric(name="revenue", entity="user", fact="revenue")
    forged = '{"control":1,"control":999,"treatment":1}'
    rows = [_plain_format_row(group_id="control", **{ASSIGNMENT_COUNTS_FIELD: forged})]
    with pytest.raises(CodedError) as raised:
        MomentsSource(rows, metrics=[metric], study_id="exp")
    assert raised.value.code == "moments.duplicate_key"
