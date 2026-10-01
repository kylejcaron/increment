"""Compiler and decision-wire contract tests."""

import json
import math

import pytest

from increment._analysis_config import resolve_configs
from increment.compatibility import _conservative_divide, _conservative_ratio
from increment.decision import (
    AbsoluteArmDecisionProcedure,
    ContrastDecisionProcedure,
    FixedInference,
    RelativeArmDecisionProcedure,
)
from increment.decision_wire import compiled_plan_from_json, compiled_plan_to_json
from increment.errors import CodedError, InvalidRequestError, WireFormatError
from increment.estimation.engine import Method
from increment.estimation.inference import Normal
from increment.estimation.priors import StudentTPrior
from increment.estimation.sequential import AlwaysValid
from increment.plan import compile_decision_plan
from increment.semantics.design import Encouragement, ExclusionRestriction, Randomized, UptakeSpec
from increment.semantics.models import (
    AnalysisPlan,
    ConversionMetric,
    ExperimentMetric,
    InferenceSpec,
    MeanMetric,
    MethodSpec,
    MultiplicitySpec,
    NormalPriorSpec,
    QuantileMetric,
    TotalMetric,
)
from tests.sequential_cases import declared_plan, registered_spec


def _mean(name: str = "rev", **kwargs) -> MeanMetric:
    return MeanMetric(name=name, entity="user", fact=name, **kwargs)


def test_conservative_divide_rounds_toward_zero_against_exact_ratio():
    result = _conservative_divide(0.1, 7)
    numerator, denominator = (0.1).as_integer_ratio()
    result_numerator, result_denominator = result.as_integer_ratio()

    assert result_numerator * 7 * denominator <= numerator * result_denominator
    next_float = math.nextafter(result, math.inf)
    next_numerator, next_denominator = next_float.as_integer_ratio()
    assert next_numerator * 7 * denominator > numerator * next_denominator


def test_conservative_divide_reports_stable_underflow():
    with pytest.raises(CodedError) as raised:
        _conservative_divide(math.nextafter(0.0, 1.0), 2)

    assert raised.value.code == "allocation.alpha.underflow"


def test_conservative_ratio_keeps_exact_endpoint_after_composite_rounding():
    numerator = math.nextafter(math.nextafter(1.0, 0.0), 0.0)

    assert _conservative_ratio(numerator, 3, 3) == numerator


@pytest.mark.parametrize("numerator", [math.nan, math.inf, -math.inf])
def test_conservative_ratio_refuses_nonfinite_numerator_with_stable_code(numerator):
    with pytest.raises(CodedError) as raised:
        _conservative_ratio(numerator, 1, 1)

    assert raised.value.code == "allocation.alpha.underflow"


def test_compile_decision_plan_primary_share_is_conservatively_allocated():
    metrics = [_mean(f"m_{i}") for i in range(7)]
    plan = AnalysisPlan(alpha=0.1, primary=tuple(metric.name for metric in metrics))

    resolved = compile_decision_plan(plan, metrics, path="warehouse")

    expected = _conservative_divide(plan.alpha, len(metrics))
    assert {test.alpha for test in resolved.procedures.values()} == {expected}


def test_seven_primary_alpha_wire_roundtrip_rejects_one_ulp_tamper():
    metrics = [_mean(f"m_{i}") for i in range(7)]
    compiled = compile_decision_plan(
        AnalysisPlan(alpha=0.1, primary=tuple(metric.name for metric in metrics)),
        metrics,
        path="warehouse",
    )
    restored = compiled_plan_from_json(compiled_plan_to_json(compiled))
    assert restored == compiled

    raw = json.loads(compiled_plan_to_json(compiled))
    name = metrics[0].name
    raw["procedures"][name]["alpha"] = math.nextafter(raw["procedures"][name]["alpha"], math.inf)
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json(json.dumps(raw))
    assert raised.value.code == "wire.compiled_plan.procedure_alpha_mismatch"


def _conv(name: str = "conv", **kwargs) -> ConversionMetric:
    return ConversionMetric(name=name, entity="user", fact=name, **kwargs)


def _quantile(name: str = "lat", **kwargs) -> QuantileMetric:
    return QuantileMetric(name=name, entity="user", fact=name, quantile=0.5, **kwargs)


class TestPlanNoneDefault:
    """Rule: plan=None -> declared=False, everyone unassigned, alpha_share=0.05,
    two-sided, nulls from resolve_null_and_alternative with no overrides."""

    def test_plan_none_default(self):
        metrics = [_mean("rev"), _conv("conv")]
        resolved = compile_decision_plan(None, metrics, path="warehouse")

        assert resolved.declared is False
        assert resolved.alpha == 0.05
        assert resolved.q == 0.10
        assert isinstance(resolved.inference, FixedInference)
        assert set(resolved.procedures) == {"rev", "conv"}
        for test in resolved.procedures.values():
            assert test.role == "unassigned"
            assert test.alpha == 0.05
            assert test.alternative == "two-sided"
            assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
            assert getattr(test, "null_abs", None) is None
            assert test.family.member is False  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_plan_none_catalog_margin_still_applies(self):
        """Catalog margins still apply even with no declared plan."""
        metric = _mean("rev", margin=0.05, preferred_direction="increase")
        resolved = compile_decision_plan(None, [metric], path="warehouse")
        test = resolved.procedures["rev"]
        assert test.null_lift == pytest.approx(-0.05)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
        assert test.alternative == "greater"


class TestFrameSecondariesDerivation:
    """Rule: frame secondaries default = declared - primaries - guardrails;
    explicit secondaries must equal that set exactly (refusal lists both)."""

    def test_default_secondaries_are_declared_minus_primary_and_guardrail(self):
        metrics = [_mean("rev"), _mean("cost"), _conv("conv", preferred_direction="increase")]
        plan = AnalysisPlan(
            primary="rev",
            guardrails=[ExperimentMetric(metric="conv", margin_abs=0.01)],
        )
        resolved = compile_decision_plan(plan, metrics, path="frame")

        assert resolved.procedures["rev"].role == "primary"
        assert resolved.procedures["conv"].role == "guardrail"
        assert resolved.procedures["cost"].role == "secondary"

    def test_explicit_secondaries_matching_default_is_accepted(self):
        metrics = [_mean("rev"), _mean("cost")]
        plan = AnalysisPlan(primary="rev", secondaries=["cost"])
        resolved = compile_decision_plan(plan, metrics, path="frame")
        assert resolved.procedures["cost"].role == "secondary"

    def test_explicit_secondaries_not_matching_default_refuses(self):
        metrics = [_mean("rev"), _mean("cost"), _mean("extra")]
        plan = AnalysisPlan(primary="rev", secondaries=["cost"])
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="frame")

        error = raised.value
        assert error.code == "plan.frame.secondaries"
        assert error.context["explicit"] == {"cost"}
        assert error.context["derived"] == {"cost", "extra"}


class TestWarehouseMembership:
    """Rule: warehouse membership = union of roles; a plan name not in the
    catalog refuses rather than silently dropping a role."""

    def test_membership_is_union_of_roles_rest_unassigned(self):
        metrics = [_mean("rev"), _mean("cost", preferred_direction="decrease"), _mean("extra")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        resolved = compile_decision_plan(plan, metrics, path="warehouse")

        assert resolved.procedures["rev"].role == "primary"
        assert resolved.procedures["cost"].role == "guardrail"
        assert resolved.procedures["extra"].role == "unassigned"

    def test_unknown_plan_name_refuses(self):
        metrics = [_mean("rev")]
        plan = AnalysisPlan(primary="not_a_declared_metric")
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="warehouse")
        assert raised.value.code == "plan.metrics.unknown"


class TestGuardrailDirection:
    """Rule: a guardrail without an explicitly-set preferred_direction refuses,
    naming the metric."""

    def test_guardrail_without_explicit_direction_refuses(self):
        metrics = [_mean("rev"), _mean("cost")]  # cost: preferred_direction not set
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="warehouse")
        assert raised.value.code == "plan.guardrail.direction"

    def test_guardrail_without_explicit_direction_still_refuses_after_dump_reload(self):
        """Round-trip fidelity (yjc9): a default-direction metric used as
        a guardrail must refuse identically whether it was just
        constructed or dumped and reloaded first. `model_dump()` used to
        write out the DEFAULTED `preferred_direction`, and reloading
        that marked it as explicitly declared -- so the same guardrail
        that refuses before serialisation compiled successfully after."""
        metrics = [_mean("rev"), _mean("cost")]  # cost: preferred_direction not set
        reloaded_cost = MeanMetric.model_validate(metrics[1].model_dump(mode="json"))
        assert reloaded_cost.declared_preferred_direction is None
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, [metrics[0], reloaded_cost], path="warehouse")
        assert raised.value.code == "plan.guardrail.direction"

    def test_guardrail_with_explicit_direction_succeeds(self):
        metrics = [_mean("rev"), _mean("cost", preferred_direction="decrease")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        assert resolved.procedures["cost"].role == "guardrail"

    def test_margin_less_guardrail_decrease_is_one_sided_less(self):
        """A guardrail with no declared margin/margin_abs still tests
        one-sided against zero, on its declared adverse side -- it must
        never fall back to the plan's two-sided default."""
        metrics = [_mean("rev"), _mean("cost", preferred_direction="decrease")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "less"
        assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_margin_less_guardrail_increase_is_one_sided_greater(self):
        metrics = [_mean("rev"), _mean("cost", preferred_direction="increase")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "greater"
        assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_guardrail_neutral_direction_refuses(self):
        """A guardrail with an explicitly-set but neutral preferred_direction
        also refuses -- 'neutral' names no adverse side to test against, the
        same reasoning `resolve_margin` already applies to a margin-declared
        guardrail."""
        metrics = [_mean("rev"), _mean("cost", preferred_direction="neutral")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="warehouse")
        assert raised.value.code == "plan.guardrail.direction"

    def test_margin_less_guardrail_tail_ignores_plans_alternative_decrease(self):
        """A margin-less guardrail's tail is a pure function of its own
        preferred_direction, never of the plan's own `alternative` (scoped
        to primaries/secondaries only) -- here the plan's alternative
        ('greater') is the opposite tail from what the guardrail resolves
        to, so a leak would be caught."""
        metrics = [_mean("rev"), _mean("cost", preferred_direction="decrease")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"], alternative="greater")
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "less"
        assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_margin_less_guardrail_tail_ignores_plans_alternative_increase(self):
        metrics = [_mean("rev"), _mean("cost", preferred_direction="increase")]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"], alternative="less")
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "greater"
        assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_margin_declared_guardrail_tail_ignores_plans_alternative(self):
        """A margin-DECLARED guardrail's tail must also be a pure function
        of its own margin/preferred_direction, never of the plan's own
        `alternative` -- resolve_null_and_alternative forwards the plan's
        alternative through untouched whenever it is not "two-sided"
        (correct for primary/secondary, wrong here), so the guardrail
        branch must force alternative="two-sided" into that call rather
        than trusting its own default gate (which only fires when no
        margin resolved)."""
        metrics = [_mean("rev"), _mean("cost", preferred_direction="decrease")]
        plan = AnalysisPlan(
            primary="rev",
            guardrails=[ExperimentMetric(metric="cost", margin=0.02)],
            alternative="greater",
        )
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "less"
        assert test.null_lift == pytest.approx(0.02)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_margin_declared_guardrail_increase_tail_ignores_plans_alternative(self):
        """Same rule as above, mirrored on the increase/greater side."""
        metrics = [_mean("rev"), _mean("cost", preferred_direction="increase")]
        plan = AnalysisPlan(
            primary="rev",
            guardrails=[ExperimentMetric(metric="cost", margin=0.03)],
            alternative="less",
        )
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "greater"
        assert test.null_lift == pytest.approx(-0.03)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_margin_abs_declared_guardrail_tail_ignores_plans_alternative(self):
        """The absolute-axis branch has its own independent forward of the
        plan's `alternative` inside resolve_null_and_alternative -- a
        regression there is invisible to every other test in this class,
        which only exercises the relative axis against a non-default
        alternative."""
        metrics = [_mean("rev"), _mean("cost", preferred_direction="decrease")]
        plan = AnalysisPlan(
            primary="rev",
            guardrails=[ExperimentMetric(metric="cost", margin_abs=1.5)],
            alternative="greater",
        )
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        test = resolved.procedures["cost"]
        assert test.alternative == "less"
        assert test.null_abs == pytest.approx(1.5)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture


class TestNullMarginPrecedence:
    """Rule: plan-binding margin/margin_abs beats catalog MetricBase.margin,
    beats 0 -- delegated to resolve_null_and_alternative."""

    def test_binding_margin_beats_catalog_margin(self):
        metric = _mean("rev", margin=0.05, preferred_direction="increase")
        plan = AnalysisPlan(primary=ExperimentMetric(metric="rev", margin=0.10))
        resolved = compile_decision_plan(plan, [metric], path="warehouse")
        assert resolved.procedures["rev"].null_lift == pytest.approx(-0.10)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_catalog_margin_beats_zero(self):
        metric = _mean("rev", margin=0.05, preferred_direction="increase")
        plan = AnalysisPlan(primary="rev")
        resolved = compile_decision_plan(plan, [metric], path="warehouse")
        assert resolved.procedures["rev"].null_lift == pytest.approx(-0.05)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_no_margin_defaults_to_zero_two_sided(self):
        metric = _mean("rev")
        plan = AnalysisPlan(primary="rev")
        resolved = compile_decision_plan(plan, [metric], path="warehouse")
        test = resolved.procedures["rev"]
        assert test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
        assert test.alternative == "two-sided"

    def test_binding_margin_abs_beats_catalog_margin_abs(self):
        metric = _mean("rev", margin_abs=1.0, preferred_direction="increase")
        plan = AnalysisPlan(primary=ExperimentMetric(metric="rev", margin_abs=2.0))
        resolved = compile_decision_plan(plan, [metric], path="warehouse")
        assert resolved.procedures["rev"].null_abs == pytest.approx(-2.0)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture


class TestFrameMethodsAndPriorRefused:
    """Rule: frame plan entries with method roles/prior set refuse, pointing
    at MetricSpec."""

    def test_frame_entry_with_prior_refuses(self):
        metrics = [_mean("rev")]
        plan = AnalysisPlan(
            primary=ExperimentMetric(metric="rev", prior=NormalPriorSpec(mu=0, sigma=1))
        )
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="frame")
        assert raised.value.code == "plan.frame.override"
        assert raised.value.context["metric"] == "rev"
        assert raised.value.context["field"] == "prior"

    def test_frame_entry_with_methods_refuses(self):
        from increment.semantics.models import MethodSpec

        metrics = [_mean("rev")]
        plan = AnalysisPlan(
            primary=ExperimentMetric(
                metric="rev",
                decision_method=MethodSpec(name="ols"),
            )
        )
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="frame")
        assert raised.value.code == "plan.frame.override"
        assert raised.value.context["metric"] == "rev"
        assert raised.value.context["field"] == "decision_method"

    def test_frame_entry_sensitivity_only_refuses_with_field_context(self):
        from increment.semantics.models import MethodSpec

        metrics = [_mean("rev")]

        plan = AnalysisPlan(
            primary=ExperimentMetric(
                metric="rev",
                sensitivity_methods=(MethodSpec(name="sensitivity"),),
            )
        )
        with pytest.raises(InvalidRequestError) as raised:
            compile_decision_plan(plan, metrics, path="frame")

        assert raised.value.code == "plan.frame.override"
        assert raised.value.context["metric"] == "rev"
        assert raised.value.context["field"] == "sensitivity_methods"

    def test_warehouse_entry_with_prior_is_allowed(self):
        metrics = [_mean("rev")]
        plan = AnalysisPlan(
            primary=ExperimentMetric(metric="rev", prior=NormalPriorSpec(mu=0, sigma=1))
        )
        resolved = compile_decision_plan(plan, metrics, path="warehouse")
        assert resolved.procedures["rev"].role == "primary"


class TestInferenceConversion:
    def test_compiled_wire_preserves_actual_raw_decision(self):
        from increment import estimate_sequential
        from increment.semantics.models import ConversionMetric
        from tests.sequential_cases import (
            capture,
            records,
            registration,
        )

        reg = registration("bernoulli")
        rows = records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
        snapshot = capture(reg, rows)
        metrics = [ConversionMetric(name="outcome", entity="unit_id", fact="revenue")]
        spec = InferenceSpec(kind="always_valid", registration=reg)
        compiled = compile_decision_plan(
            AnalysisPlan(primary="outcome", inference=spec),
            metrics,
            path="warehouse",
            design=Randomized(control_group="control"),
        )
        restored = compiled_plan_from_json(compiled_plan_to_json(compiled))
        assert isinstance(compiled.inference, AlwaysValid)
        assert isinstance(restored.inference, AlwaysValid)
        first = estimate_sequential(snapshot, compiled.inference).results[0]
        replay = estimate_sequential(snapshot, restored.inference).results[0]
        assert first.require_sequential_result() == replay.require_sequential_result()
        assert first.require_lift().value > 0


class TestAlwaysValidQuantileWarning:
    """Quantile sequential plans refuse before claiming sequential evidence."""

    def test_quantile_refuses_before_plan_can_claim_sequential_evidence(self):
        metrics = [_quantile("lat"), _mean("rev")]
        design = Randomized(control_group="control")
        plan = declared_plan(metrics, source_id="experiment", design=design)
        with pytest.raises(CodedError) as raised:
            compile_decision_plan(plan, metrics, path="warehouse", design=design)
        assert raised.value.code == "sequential.route.unsupported"


class TestReportOnlyMetricNullResolution:
    """Rule: a report-only metric (MetricIdentity-based, not MetricBase --
    e.g. TotalMetric) never crashes resolution even though it carries no
    margin/direction concept to resolve a null/alternative from."""

    def test_total_metric_alongside_mean_metrics_resolves_without_raising(self):
        metrics = [
            _mean("rev"),
            _mean("cost", preferred_direction="decrease"),
            TotalMetric(name="gmv", fact="gmv"),
        ]
        plan = AnalysisPlan(primary="rev", guardrails=["cost"])
        resolved = compile_decision_plan(plan, metrics, path="warehouse")

        total_test = resolved.procedures["gmv"]
        assert total_test.role == "unassigned"
        assert total_test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
        assert getattr(total_test, "null_abs", None) is None
        assert total_test.family.member is False  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture

    def test_total_metric_with_no_plan_also_resolves(self):
        metrics = [_mean("rev"), TotalMetric(name="gmv", fact="gmv")]
        resolved = compile_decision_plan(None, metrics, path="warehouse")

        total_test = resolved.procedures["gmv"]
        assert total_test.null_lift == 0.0  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
        assert getattr(total_test, "null_abs", None) is None
        assert total_test.alternative == "two-sided"

    def test_total_metric_named_into_a_testing_role_refuses(self):
        """A report-only metric has no null it can be tested against, so
        naming it primary/secondary/guardrail must refuse rather than
        silently resolve a flat null_lift=0.0 -- and on the frame path it
        would otherwise join the BH/e-BH family with a null it can never
        be tested against."""
        metrics = [_mean("rev"), TotalMetric(name="gmv", fact="gmv")]
        cases = (
            (AnalysisPlan(primary="gmv"), "plan.metric.report_only"),
            (AnalysisPlan(primary="rev", secondaries=["gmv"]), "plan.metric.report_only"),
            # A guardrail's direction check runs before role resolution, so a
            # TotalMetric guardrail (no preferred_direction field at all) trips
            # the direction refusal first, not the report-only one.
            (AnalysisPlan(primary="rev", guardrails=["gmv"]), "plan.guardrail.direction"),
        )
        for plan, code in cases:
            with pytest.raises(InvalidRequestError) as raised:
                compile_decision_plan(plan, metrics, path="warehouse")
            assert raised.value.code == code


def test_compile_decision_plan_builds_immutable_relative_default():
    compiled = compile_decision_plan(None, [_mean("rev")])
    procedure = compiled.procedures["rev"]

    assert isinstance(procedure, RelativeArmDecisionProcedure)
    assert procedure.null_lift == 0.0
    assert procedure.decision_method == Method(name="unadjusted")
    assert type(compiled.procedures).__name__ == "mappingproxy"
    with pytest.raises(TypeError):
        compiled.procedures["new"] = procedure  # type: ignore[index]  # ty: ignore[invalid-assignment] -- frozen mapping contract test


def test_compile_decision_plan_carries_roles_methods_family_and_absolute_null():
    metric = _mean("rev", preferred_direction="increase", margin_abs=2.0)
    plan = AnalysisPlan(
        alpha=0.04,
        primary="rev",
        guardrails=(),
    )
    compiled = compile_decision_plan(plan, [metric])
    procedure = compiled.procedures["rev"]

    assert isinstance(procedure, AbsoluteArmDecisionProcedure)
    assert procedure.null_abs == -2.0
    assert procedure.alpha == 0.04
    assert procedure.family.member is False
    assert compiled.view_policies.randomized_breakout.correction == "bh"


def test_compile_decision_plan_uses_warehouse_method_roles_and_secondary_family():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="rev",
            decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
            sensitivity_methods=(MethodSpec(name="unadjusted"),),
        ),
        secondaries=("conv",),
    )
    compiled = compile_decision_plan(plan, [_mean("rev"), _conv("conv")])

    primary = compiled.procedures["rev"]
    secondary = compiled.procedures["conv"]
    assert primary.decision_method.name == "cuped"  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
    assert [method.name for method in primary.sensitivity_methods] == ["unadjusted"]  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
    assert secondary.family.member is True  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
    assert secondary.family.name == "secondary"  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture


def test_wire_round_trips_absolute_and_sequential_inference_with_normal_prior():
    from increment.estimation.inference import Normal

    metric = _mean("rev", margin_abs=2.0, preferred_direction="increase")
    compiled = compile_decision_plan(
        AnalysisPlan(primary="rev"),
        [metric],
        prior=Normal(mu=0.1, sigma=0.4),
    )
    restored = compiled_plan_from_json(compiled_plan_to_json(compiled))
    procedure = restored.procedures["rev"]
    assert isinstance(procedure, AbsoluteArmDecisionProcedure)
    assert isinstance(procedure.inference, FixedInference)
    assert procedure.prior is not None
    assert procedure.prior.mu == pytest.approx(0.1)  # ty: ignore[unresolved-attribute] -- plan variant is fixed by test fixture
    assert procedure.prior_is_global is True
    from increment import estimate_sequential
    from increment.semantics.models import ConversionMetric
    from tests.sequential_cases import capture, records, registration

    reg = registration("bernoulli")
    rows = records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    snapshot = capture(reg, rows)
    metrics = [ConversionMetric(name="outcome", entity="unit_id", fact="revenue")]
    policy = AlwaysValid(registration=reg)
    sequential = compile_decision_plan(
        AnalysisPlan(
            primary="outcome",
            inference=InferenceSpec(kind="always_valid", registration=reg),
        ),
        metrics,
        design=Randomized(control_group="control"),
    )
    seq_restored = compiled_plan_from_json(compiled_plan_to_json(sequential))
    assert isinstance(seq_restored.inference, AlwaysValid)
    expected = estimate_sequential(snapshot, policy).results[0]
    actual = estimate_sequential(snapshot, seq_restored.inference).results[0]
    assert actual.require_sequential_result() == expected.require_sequential_result()


def test_wire_round_trips_contrast_procedure():
    metric = _mean("rev", margin_abs=2.0, preferred_direction="increase")
    compiled = compile_decision_plan(AnalysisPlan(primary="rev"), [metric], path="frame/contrast")
    restored = compiled_plan_from_json(compiled_plan_to_json(compiled))
    procedure = restored.procedures["rev"]
    assert isinstance(procedure, ContrastDecisionProcedure)
    assert procedure.null_abs == pytest.approx(-2.0)
    assert procedure.alpha == pytest.approx(0.05)


def test_wire_rejects_duplicate_json_keys_and_invalid_boundaries():
    compiled = compile_decision_plan(None, [_mean("rev")])
    payload = compiled_plan_to_json(compiled)
    with pytest.raises(WireFormatError) as duplicate:
        compiled_plan_from_json(payload[:-1] + ',"alpha":0.2}')
    assert duplicate.value.code == "wire.payload.duplicate_key"

    raw = json.loads(payload)
    raw["procedures"]["rev"]["family"]["member"] = True
    with pytest.raises(WireFormatError) as invalid:
        compiled_plan_from_json(json.dumps(raw))
    assert invalid.value.code == "wire.payload.invalid"


def test_wire_rejects_tampered_nonprimary_alpha():
    compiled = compile_decision_plan(None, [_mean("rev")])
    raw = json.loads(compiled_plan_to_json(compiled))
    raw["procedures"]["rev"]["alpha"] = 0.2
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json(json.dumps(raw))
    assert raised.value.code == "wire.compiled_plan.procedure_alpha_mismatch"


def test_wire_rejects_nonfinite_normal_prior():
    """A nonfinite prior arriving via raw wire JSON is caught at the
    payload boundary (NormalPriorSpec's own field constraint, reused
    directly as the wire type) before decision_wire's own
    already-constructed-object check ever runs; that check stays live
    for a runtime Normal(mu=inf) built in-process."""
    compiled = compile_decision_plan(None, [_mean("rev")])
    raw = json.loads(compiled_plan_to_json(compiled))
    raw["procedures"]["rev"]["prior"] = {"mu": float("nan"), "sigma": 1.0}
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json(json.dumps(raw))
    assert raised.value.code == "wire.payload.invalid"


def test_duplicate_method_names_refuse():
    compiled = compile_decision_plan(None, [_mean("rev")])
    raw = json.loads(compiled_plan_to_json(compiled))
    raw["procedures"]["rev"]["sensitivity_methods"] = [raw["procedures"]["rev"]["decision_method"]]
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_json(json.dumps(raw))
    assert raised.value.code == "wire.procedure.invalid_method"


def test_runtime_folds_refuse_on_wire():
    from increment.decision_wire import compiled_plan_to_json
    from increment.errors import WireFormatError

    compiled = compile_decision_plan(
        None,
        [_mean("rev")],
        methods=[Method(name="dml", folds=2)],
    )
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_to_json(compiled)
    assert raised.value.code == "wire.procedure.runtime_method"


class TestEncouragementSeamRefusals:
    """Every plan seam refuses unsupported encouragement combinations with
    the registered coded contract: the compiler and each supplied-plan
    source constructor."""

    @staticmethod
    def _metric():
        from increment.semantics.models import MeanMetric

        return MeanMetric(
            name="revenue",
            entity="user",
            fact="orders",
            aggregation="sum",
            margin=0.01,
            preferred_direction="decrease",
        )

    @staticmethod
    def _design():
        from increment.semantics.design import (
            Encouragement,
            ExclusionRestriction,
            UptakeSpec,
        )

        return Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment moves revenue via uptake"
            ),
        )

    def _margin_plan(self):
        # A sequential margined plan, compiled without a design so the
        # encouragement validator cannot fire at compile time; supplying it
        # to an encouragement source must then refuse at the seam. Only
        # sequential encouragement lacks a shifted null.
        from increment.estimation.sequential import AlwaysValid
        from increment.plan import compile_decision_plan
        from tests.sequential_cases import registration

        fixed = compile_decision_plan(None, [self._metric()], design=None)
        return fixed.model_copy(
            update={"inference": AlwaysValid(registration=registration("gaussian"))}
        )

    def test_compiler_refuses_sequential_declared_margin(self):
        from increment.errors import CodedError
        from increment.plan import compile_decision_plan

        plan = AnalysisPlan(guardrails=["revenue"], inference=registered_spec())
        with pytest.raises(CodedError) as raised:
            compile_decision_plan(plan, [self._metric()], design=self._design())
        assert raised.value.code == "plan.encouragement.margin"

    def test_compiler_refuses_encouragement_sequential_cuped(self):
        from increment.errors import CodedError
        from increment.plan import compile_decision_plan
        from increment.semantics.models import (
            AnalysisPlan,
            ExperimentMetric,
            MethodSpec,
        )

        metric = self._metric().model_copy(update={"margin": None, "preferred_direction": None})
        plan = AnalysisPlan(
            primary=ExperimentMetric(
                metric="revenue",
                decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
            ),
            inference=registered_spec(),
        )
        with pytest.raises(CodedError) as raised:
            compile_decision_plan(plan, [metric], design=self._design())
        assert raised.value.code == "plan.encouragement.sequential_cuped"

    def test_moments_source_refuses_supplied_margin_plan(self):
        from increment.errors import CodedError
        from increment.sources import MomentsSource

        with pytest.raises(CodedError) as raised:
            MomentsSource(
                [],
                metrics=[self._metric()],
                study_id="exp",
                design=self._design(),
                plan=self._margin_plan(),
            )
        assert raised.value.code == "plan.encouragement.margin"

    def test_breakout_source_refuses_supplied_margin_plan(self):
        from increment.errors import CodedError
        from increment.sources import BreakoutMomentsSource

        with pytest.raises(CodedError) as raised:
            BreakoutMomentsSource(
                [],
                metrics=[self._metric()],
                dimension="region",
                study_id="exp",
                design=self._design(),
                plan=self._margin_plan(),
            )
        assert raised.value.code == "plan.encouragement.margin"

    def test_sql_panel_source_refuses_supplied_margin_plan(self):
        import ibis

        from increment.errors import CodedError
        from increment.query.source import SqlPanelSource

        con = ibis.duckdb.connect()
        summary = con.create_table(
            "summary",
            {
                "group_id": ["control"],
                "metric": ["revenue"],
                "n": [1],
                "mean": [0.0],
                "m2": [0.0],
            },
        )
        with pytest.raises(CodedError) as raised:
            SqlPanelSource(
                con,
                summary,
                metrics=[self._metric()],
                study_id="exp",
                design=self._design(),
                plan=self._margin_plan(),
            )
        assert raised.value.code == "plan.encouragement.margin"


def _encouragement_design() -> Encouragement:
    return Encouragement(
        control_group="control",
        uptake=UptakeSpec(fact="clicked"),
        exclusion_restriction=ExclusionRestriction(
            acknowledged=True, justification="assignment moves revenue via uptake"
        ),
    )


def _resolved_config(metric, *, methods=None, prior=None):
    """Resolve one metric's config the way a source resolves it before compiling."""
    (config,) = resolve_configs([metric], None, None, methods=methods, prior=prior)
    return config


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "plan.configs_keys_match",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="frame/contrast",
                configs={"wrong": _resolved_config(_mean("rev"))},
            ),
        ),
        (
            "plan.compile_configs_wrong_type",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="frame/contrast",
                configs=[None],  # ty: ignore[invalid-argument-type]
            ),
        ),
        (
            "plan.compile_configs_template_mismatch",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="frame/contrast",
                configs={
                    "rev": _resolved_config(MeanMetric(name="rev", entity="user", fact="alt"))
                },
            ),
        ),
        (
            "plan.metric_frame_contrast",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="frame/contrast",
                configs={"rev": _resolved_config(_mean("rev"), prior=Normal(mu=0.0, sigma=1.0))},
            ),
        ),
        (
            "plan.configs_contain_one",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="warehouse",
                configs=[_resolved_config(_mean("rev")), _resolved_config(_mean("rev"))],
            ),
        ),
        (
            "plan.compile_configs_wrong_type",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="warehouse",
                configs=[None],  # ty: ignore[invalid-argument-type]
            ),
        ),
        (
            "plan.compile_configs_template_mismatch",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="warehouse",
                configs={
                    "rev": _resolved_config(MeanMetric(name="rev", entity="user", fact="alt"))
                },
            ),
        ),
        (
            "plan.metric_informative_priors",
            lambda: compile_decision_plan(
                AnalysisPlan(primary="rev", inference=registered_spec()),
                [_mean("rev")],
                path="warehouse",
                prior=Normal(mu=0.0, sigma=0.2),
            ),
        ),
        (
            "plan.metric_priors_relative",
            lambda: compile_decision_plan(
                None,
                [_mean("rev", margin_abs=1.0, preferred_direction="increase")],
                path="warehouse",
                prior=StudentTPrior(nu=5.0, scale=0.1),
            ),
        ),
        (
            "plan.corrected_encouragement_breakout",
            lambda: compile_decision_plan(
                AnalysisPlan(primary="rev", view_multiplicity=MultiplicitySpec(correction="bh")),
                [_mean("rev")],
                design=_encouragement_design(),
            ),
        ),
        (
            "plan.unsupported_decision_plan",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="bogus",  # ty: ignore[invalid-argument-type]
            ),
        ),
        (
            "plan.duplicate_metric_name",
            lambda: compile_decision_plan(None, [_mean("rev"), _mean("rev")], path="warehouse"),
        ),
        (
            "plan.frame_contrast_does",
            lambda: compile_decision_plan(
                None,
                [_mean("rev")],
                path="frame/contrast",
                methods=[Method(name="unadjusted")],
            ),
        ),
        (
            "plan.metric_cuped_methods",
            lambda: compile_decision_plan(
                AnalysisPlan(
                    primary=ExperimentMetric(
                        metric="rev",
                        decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
                    ),
                    inference=registered_spec(),
                ),
                [_mean("rev")],
                path="warehouse",
                design=None,
            ),
        ),
    ],
)
def test_plan_refusal_carries_code(code, build):
    with pytest.raises(CodedError) as raised:
        build()
    assert raised.value.code == code
