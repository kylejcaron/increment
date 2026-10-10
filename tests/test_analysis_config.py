"""Behavioral tests for the metric-selection / per-metric-configuration
seam every readout resolves metric names and per-metric ``methods``/``prior``
through.

Selection is asserted through ``Analysis`` results, and method/prior
resolution through the compiled decision plan (``compile_decision_plan``,
and the ``src.plan`` a source publishes). Three selection contracts have no
cheap public route -- accepting undeclared call-time ``Metric`` objects, the
order those append in, and the refusal precedence when several selection
hazards co-occur (that one is native-only, via ``breakout_summaries``) --
so those are exercised on ``select_metrics`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from increment import Analysis
from increment._analysis_config import select_metrics
from increment.decision import ArmDecisionProcedure
from increment.errors import InvalidRequestError
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.frame import MetricSpec, from_unit_summary
from increment.plan import compile_decision_plan
from increment.semantics.design import AdjustmentSet, Observational
from increment.semantics.models import (
    AnalysisPlan,
    ConversionMetric,
    ExperimentMetric,
    MeanMetric,
    MethodSpec,
    NormalPriorSpec,
)

if TYPE_CHECKING:
    from increment.decision import CompiledDecisionPlan

REVENUE = MeanMetric(name="revenue", entity="user_id", fact="purchase", aggregation="sum")
SIGNUP = ConversionMetric(name="signup", entity="user_id", fact="signup")
ORDERS = MeanMetric(name="orders", entity="user_id", fact="purchase", aggregation="count")
LATENCY = MeanMetric(name="latency", entity="user_id", fact="latency_fact", aggregation="avg_event")
DECLARED = [REVENUE, ORDERS, LATENCY]

UNADJUSTED = Method(name="unadjusted")
CUPED = Method(name="cuped", variance_reduction="cuped")

OBSERVATIONAL = Observational(
    control_group="control", adjustment=AdjustmentSet(covariates=("covariate",))
)

UNITS = 60


def _unit_frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(UNITS)],
            "variant": ["control" if i % 2 == 0 else "treatment" for i in range(UNITS)],
            "revenue": [float(i % 7) for i in range(UNITS)],
            "orders": [float(i % 3) for i in range(UNITS)],
            "latency": [float(i % 5) + 1.0 for i in range(UNITS)],
            "pre_revenue": [float(i % 4) for i in range(UNITS)],
            "covariate": [float(i % 5) for i in range(UNITS)],
        }
    )


def _procedure(compiled: CompiledDecisionPlan, name: str = "revenue") -> ArmDecisionProcedure:
    return cast("ArmDecisionProcedure", compiled.procedures[name])


@pytest.fixture(scope="module")
def analysis() -> Analysis:
    return Analysis.from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean", "orders": "mean", "latency": "mean"},
    )


# ── select_metrics: selection and order, through Analysis results ────────


def test_select_metrics_none_returns_declared_order(analysis):
    assert [row.metric for row in analysis.run()] == ["revenue", "orders", "latency"]


def test_select_metrics_by_name_resolves_to_declared_objects(analysis):
    assert [row.metric for row in analysis.run(metrics=["orders"])] == ["orders"]


def test_select_metrics_output_follows_declared_order_not_request_order(analysis):
    """Result order follows declaration order, never caller order (Global
    Constraint)."""
    rows = analysis.run(metrics=["latency", "revenue"])
    assert [row.metric for row in rows] == ["revenue", "latency"]


def test_select_metrics_duplicate_name_rejected(analysis):
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(metrics=["revenue", "revenue"])
    assert raised.value.code == "facade.analysis_config.duplicate_metric_name"


def test_select_metrics_duplicate_across_string_and_object_rejected(analysis):
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(metrics=["revenue", REVENUE])
    assert raised.value.code == "facade.analysis_config.duplicate_metric_name"


def test_select_metrics_accepts_declared_metric_objects(analysis):
    rows = analysis.run(metrics=[ORDERS, REVENUE])
    assert [row.metric for row in rows] == ["revenue", "orders"]


def test_select_metrics_preserves_caller_object_over_declared_when_names_coincide():
    """A caller-constructed Metric object sharing a declared metric's
    name is never silently substituted with the declared object - a
    modified variant (e.g. a different band/window) must survive
    unchanged, matching the pre-existing pass-through contract for
    Metric-object selection. No public route can tell preservation from
    substitution for a name it already declares, so the resolver is
    exercised directly."""
    shadow_revenue = MeanMetric(
        name="revenue", entity="user_id", fact="a_different_fact", aggregation="avg_event"
    )
    selected = select_metrics(DECLARED, [shadow_revenue], caller="run_daily")
    assert selected == [shadow_revenue]
    assert selected[0] is shadow_revenue
    assert selected[0].fact == "a_different_fact"


def test_select_metrics_undeclared_object_refused_by_default(analysis):
    undeclared = MeanMetric(name="undeclared", entity="user_id", fact="x", aggregation="sum")
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run(metrics=[undeclared])
    assert raised.value.code == "facade.analysis_config.unknown_metric_declared"


def test_select_metrics_undeclared_object_allowed_when_flagged():
    """Only the native day-axis route opts into undeclared call-time Metric
    objects, so the flag itself has no cheap public route."""
    undeclared = MeanMetric(name="undeclared", entity="user_id", fact="x", aggregation="sum")
    selected = select_metrics(
        DECLARED, [REVENUE, undeclared], caller="run_daily", allow_undeclared_objects=True
    )
    assert selected == [REVENUE, undeclared]


def test_select_metrics_undeclared_objects_appended_at_end_in_caller_order():
    """The append order for undeclared objects is set by this resolver and is
    not separately observable from a public route."""
    u1 = MeanMetric(name="u1", entity="user_id", fact="x", aggregation="sum")
    u2 = MeanMetric(name="u2", entity="user_id", fact="x", aggregation="sum")
    selected = select_metrics(
        DECLARED,
        [u2, LATENCY, u1, REVENUE],
        caller="run_daily",
        allow_undeclared_objects=True,
    )
    # declared-matched entries follow DECLARED order; undeclared entries
    # are appended at the end, in the order the caller supplied them.
    assert selected == [REVENUE, LATENCY, u2, u1]


@pytest.mark.parametrize(
    ("extra", "code", "field", "names"),
    [
        (["typo", "revenue"], "duplicate_metric_name", "duplicate_names", ("revenue",)),
        (["typo"], "unknown_metric_declared", "unknown_names", ("typo",)),
        ([], "metric_definition_mismatch", "mismatched_names", ("orders", "revenue")),
    ],
)
def test_strict_metric_selection_refusal_precedence(extra, code, field, names):
    """Which refusal wins when several selection hazards co-occur. The only
    public route that also requires field-equal declared definitions is the
    native breakout-summaries path, so the resolver is exercised directly."""
    requested = [
        REVENUE.model_copy(update={"window_days": 2}),
        ORDERS.model_copy(update={"window_days": 3}),
        *extra,
    ]
    with pytest.raises(InvalidRequestError) as raised:
        select_metrics(
            DECLARED, requested, caller="breakout_summaries", require_declared_definitions=True
        )
    assert raised.value.code == f"facade.analysis_config.{code}"
    assert raised.value.context[field] == names
    assert raised.value.context["caller"] == "breakout_summaries"


def test_select_metrics_empty_request_returns_empty(analysis):
    assert list(analysis.run(metrics=[])) == []


# ── resolve_configs: method-role precedence, through the compiled plan ───


def test_resolve_configs_no_global_no_binding_defaults_to_unadjusted_decision():
    procedure = _procedure(compile_decision_plan(None, [REVENUE]))
    assert procedure.decision_method == UNADJUSTED
    assert procedure.sensitivity_methods == ()


def test_resolve_configs_global_methods_win_for_every_metric():
    compiled = compile_decision_plan(None, [REVENUE, ORDERS], methods=[UNADJUSTED])
    for name in ("revenue", "orders"):
        procedure = _procedure(compiled, name)
        assert procedure.decision_method == UNADJUSTED
        assert procedure.sensitivity_methods == ()


@pytest.mark.parametrize("design", [None, OBSERVATIONAL])
def test_resolve_configs_explicit_empty_methods_emit_no_estimation_rows(design):
    from increment._analysis_config import UNSET, resolve_configs
    from increment._readout_request import ReadoutRequest, validate_request
    from increment.readouts._run import _run_prepared

    source = from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=design,
    )
    metrics = source.context.metrics
    configs = resolve_configs(metrics, None, None, methods=[], prior=None)
    request = ReadoutRequest.from_source(
        source, metrics=metrics, configs=configs, view="run", grain="total"
    )
    validate_request(request)
    assert (
        _run_prepared(
            source,
            metrics,
            configs,
            prior=UNSET,
            by=(),
            estimands=None,
            value_scale=None,
            population="assigned",
        )
        == []
    )


def test_lift_option_resolution_preserves_the_observational_decision():
    from increment import readouts
    from increment._analysis_config import UNSET
    from increment._lift_options import LiftOptions
    from increment.readouts._run import _run_prepared

    source = from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
        design=OBSERVATIONAL,
    )
    metrics = source.context.metrics
    options = LiftOptions.resolve(
        source,
        metrics,
        decision_method=UNSET,
        sensitivity_methods=UNSET,
        prior=UNSET,
        alpha=0.05,
    )
    (actual,) = _run_prepared(
        source,
        metrics,
        options.configs,
        prior=UNSET,
        by=(),
        estimands=None,
        value_scale=None,
        population="assigned",
    )
    (expected,) = readouts.run(source)
    assert (actual.method, actual.method_role) == ("iptw", "decision")
    actual_lift, expected_lift = actual.require_lift(), expected.require_lift()
    assert (actual_lift.value, actual_lift.lb, actual_lift.ub) == pytest.approx(
        (expected_lift.value, expected_lift.lb, expected_lift.ub)
    )


def test_resolve_configs_explicit_cuped_decision_is_not_rewritten():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="revenue",
            decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
        )
    )
    procedure = _procedure(compile_decision_plan(plan, [REVENUE]))
    assert procedure.decision_method == CUPED
    assert procedure.sensitivity_methods == ()


def test_resolve_configs_omitted_decision_keeps_unadjusted_then_sensitivity():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="revenue",
            sensitivity_methods=(MethodSpec(name="cuped", variance_reduction="cuped"),),
        )
    )
    procedure = _procedure(compile_decision_plan(plan, [REVENUE]))
    assert procedure.decision_method == UNADJUSTED
    assert procedure.sensitivity_methods == (CUPED,)


@pytest.mark.parametrize(
    "binding",
    [
        ExperimentMetric(metric="revenue", prior=NormalPriorSpec(mu=0.0, sigma=0.03)),
        ExperimentMetric(metric="revenue", margin_abs=0.01),
    ],
)
def test_resolve_configs_empty_binding_keeps_design_default_identity(binding):
    """A binding that declares no method roles (prior-only here; margin-only
    follows the identical branch) must keep the design's own default
    estimator rather than pinning one, and add no sensitivity rows."""
    directional = REVENUE.model_copy(update={"preferred_direction": "increase"})
    procedure = _procedure(
        compile_decision_plan(AnalysisPlan(primary=binding), [directional], design=OBSERVATIONAL)
    )
    # Observational's shipped default is IPTW; the resolver's own
    # unadjusted default would mean the binding pinned a method.
    assert procedure.decision_method == Method(name="iptw")
    assert procedure.sensitivity_methods == ()


def test_resolve_configs_rejects_duplicate_method_names_even_when_equal():
    with pytest.raises(InvalidRequestError) as raised:
        compile_decision_plan(None, [REVENUE], methods=[Method(name="same"), Method(name="same")])
    assert raised.value.code == "estimation.engine.method_names_unique"


def test_public_methods_roles_keep_stable_names_and_unadjusted_canonical():
    procedure = _procedure(compile_decision_plan(None, [REVENUE], methods=[CUPED, UNADJUSTED]))
    assert procedure.decision_method.name == "unadjusted"
    assert [method.name for method in procedure.sensitivity_methods] == ["cuped"]


def test_resolve_configs_registered_adjustment_takes_decision_over_unadjusted():
    """A registered adjustment name (aipw here) is only legal under
    observational identification, so its presence in methods= IS the
    observational signal: it takes the decision role over unadjusted."""
    procedure = _procedure(
        compile_decision_plan(None, [REVENUE], methods=[Method(name="aipw"), UNADJUSTED])
    )
    assert procedure.decision_method.name == "aipw"
    assert [m.name for m in procedure.sensitivity_methods] == ["unadjusted"]


def test_resolve_configs_binding_roles_used_when_no_global():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="revenue",
            decision_method=MethodSpec(name="unadjusted"),
            sensitivity_methods=(MethodSpec(name="cuped", variance_reduction="cuped"),),
        )
    )
    compiled = compile_decision_plan(plan, [REVENUE, ORDERS])
    revenue_procedure = _procedure(compiled, "revenue")
    orders_procedure = _procedure(compiled, "orders")
    assert revenue_procedure.decision_method == UNADJUSTED
    assert [m.name for m in revenue_procedure.sensitivity_methods] == ["cuped"]
    assert orders_procedure.decision_method == UNADJUSTED
    assert orders_procedure.sensitivity_methods == ()


def test_a_binding_carries_conversion_inference_into_the_compiled_methods():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="signup",
            decision_method=MethodSpec(name="unadjusted", conversion_inference="finite_sample"),
        ),
        secondaries=("revenue",),
    )
    compiled = compile_decision_plan(plan, [SIGNUP, REVENUE])
    assert _procedure(compiled, "signup").decision_method == Method(
        name="unadjusted", conversion_inference="finite_sample"
    )
    assert _procedure(compiled, "revenue").decision_method.conversion_inference == "auto"


def test_call_time_methods_replace_a_binding_conversion_inference():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="signup",
            decision_method=MethodSpec(name="unadjusted", conversion_inference="finite_sample"),
        )
    )
    procedure = _procedure(compile_decision_plan(plan, [SIGNUP], methods=[UNADJUSTED]), "signup")
    assert procedure.decision_method.conversion_inference == "auto"


def test_resolve_configs_global_methods_override_binding_roles():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="revenue",
            decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
        )
    )
    procedure = _procedure(compile_decision_plan(plan, [REVENUE], methods=[UNADJUSTED]))
    assert procedure.decision_method == UNADJUSTED
    assert procedure.sensitivity_methods == ()


def test_resolve_configs_spec_roles_used_when_no_global():
    src = from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(
                name="revenue",
                covariate="pre_revenue",
                decision_method=UNADJUSTED,
                sensitivity_methods=(CUPED,),
            )
        ],
    )
    procedure = _procedure(src.plan)
    assert procedure.decision_method == UNADJUSTED
    assert procedure.sensitivity_methods == (CUPED,)


def test_resolve_configs_pure_source_path_uses_unadjusted_default():
    """A source with no per-metric method declaration resolves an unadjusted
    decision."""
    src = from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue")],
    )
    procedure = _procedure(src.plan)
    assert procedure.decision_method == UNADJUSTED
    assert procedure.sensitivity_methods == ()


# ── resolve_configs: prior precedence ───────────────────────────────────


def test_prior_overrides_inherit_clear_replace_without_mutating_declaration():
    from tests.analysis_factory import lift_rows

    declared_prior = Normal(mu=0.0, sigma=0.2)
    replacement = Normal(mu=0.0, sigma=0.01)
    declared = Analysis.from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[MetricSpec(name="revenue", prior=declared_prior)],
    )
    unshrunk = Analysis.from_unit_summary(
        _unit_frame(),
        unit="user_id",
        group="variant",
        control="control",
        metrics={"revenue": "mean"},
    )

    def interval(report):
        [row] = lift_rows(report)
        lift = row.require_lift()
        return lift.value, lift.lb, lift.ub

    inherited = interval(declared.run())
    cleared = interval(declared.run(prior=None))
    replaced = interval(declared.run(prior=replacement))
    assert inherited == pytest.approx(cleared)
    assert inherited == pytest.approx(interval(unshrunk.run()))
    assert replaced == pytest.approx(inherited)
    [inherited_row] = lift_rows(declared.run())
    [replaced_row] = lift_rows(declared.run(prior=replacement))
    assert inherited_row.posterior_available
    assert replaced_row.posterior_available
    assert inherited_row.posterior_estimate != pytest.approx(inherited_row.require_lift().value)
    assert replaced_row.posterior_estimate != pytest.approx(replaced_row.require_lift().value)
    [cleared_row] = lift_rows(declared.run(prior=None))
    assert not cleared_row.posterior_available
    assert cleared_row.posterior_estimate is None
    assert interval(declared.run()) == pytest.approx(inherited)
