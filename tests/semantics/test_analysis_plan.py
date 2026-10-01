"""Tests for AnalysisPlan / InferenceSpec — the declared decision rule — and
the per-experiment margin fields on ExperimentMetric bindings."""

import pytest
from pydantic import ValidationError

from increment.errors import DefinitionError, InvalidRequestError
from increment.semantics.models import AnalysisPlan, ExperimentMetric, InferenceSpec
from tests.sequential_cases import registered_spec


def test_compliance_requires_registered_sequential_inference():
    from fractions import Fraction

    from increment.semantics.sequential import SequentialCompliancePolicy

    with pytest.raises(InvalidRequestError) as exc_info:
        AnalysisPlan(
            primary="y",
            inference=None,
            compliance=SequentialCompliancePolicy(alpha=Fraction(1, 20)),
        )
    assert exc_info.value.code == "sequential.registration.invalid"


# ── AnalysisPlan: full construction from the spec's YAML example ───────


def _full_plan() -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "alpha": 0.05,
            "q": 0.10,
            "primary": "conversion_rate",
            "secondaries": [
                "revenue_per_user",
                {"metric": "add_to_cart", "prior": {"mu": 0.0, "sigma": 0.03}},
            ],
            "guardrails": [
                {"metric": "checkout_latency_ms", "margin": 0.01},
            ],
        }
    )


def test_full_plan_constructs_from_yaml_shaped_dict():
    plan = _full_plan()
    assert plan.alpha == 0.05
    assert plan.q == 0.10
    assert plan.primary == "conversion_rate"
    assert plan.inference is None


def test_full_plan_role_names():
    plan = _full_plan()
    assert plan.role_names() == {
        "conversion_rate": "primary",
        "revenue_per_user": "secondary",
        "add_to_cart": "secondary",
        "checkout_latency_ms": "guardrail",
    }


def test_full_plan_primaries_normalizes_scalar_to_list():
    plan = _full_plan()
    assert plan.primaries == ["conversion_rate"]


def test_primaries_is_empty_list_when_primary_is_none():
    plan = AnalysisPlan()
    assert plan.primaries == []


def test_primaries_normalizes_list_form():
    plan = AnalysisPlan(primary=["a", "b"])
    assert plan.primaries == ["a", "b"]


def test_entry_looks_up_binding_across_roles():
    plan = _full_plan()
    binding = plan.entry("add_to_cart")
    assert isinstance(binding, ExperimentMetric)
    assert binding.prior is not None
    assert binding.prior.mu == 0.0

    guardrail = plan.entry("checkout_latency_ms")
    assert isinstance(guardrail, ExperimentMetric)
    assert guardrail.margin == 0.01

    primary = AnalysisPlan(primary={"metric": "conversion_rate", "margin_abs": 2.0})
    found = primary.entry("conversion_rate")
    assert isinstance(found, ExperimentMetric)
    assert found.margin_abs == 2.0


def test_entry_returns_none_for_bare_string_entry():
    plan = _full_plan()
    assert plan.entry("revenue_per_user") is None


def test_entry_returns_none_for_unknown_name():
    plan = _full_plan()
    assert plan.entry("nope") is None


# ── round-trip through model_dump()/model_validate() ────────────────────


def test_analysis_plan_round_trips_through_model_dump():
    plan = _full_plan()
    assert AnalysisPlan.model_validate(plan.model_dump()) == plan


# ── Duplicate-role refusal ──────────────────────────────────────────────


def test_duplicate_role_refused():
    with pytest.raises(DefinitionError) as exc_info:
        AnalysisPlan(primary="x", guardrails=["x"])
    assert exc_info.value.code == "definition.analysis.appears_both_metric"


def test_same_role_duplicate_refused():
    with pytest.raises(DefinitionError) as exc_info:
        AnalysisPlan(secondaries=["x", "x"])
    assert exc_info.value.code == "definition.analysis.appears_more_once"


def test_same_role_duplicate_metric_object_and_bare_string_refused():
    with pytest.raises(DefinitionError) as exc_info:
        AnalysisPlan(guardrails=["x", {"metric": "x", "margin": 0.1}])
    assert exc_info.value.code == "definition.analysis.appears_more_once"


# ── always-valid role admission ─────────────────────────────────────────


def test_always_valid_admits_every_role():
    plan = AnalysisPlan(
        primary="conversion_rate",
        secondaries=["revenue_per_user", "add_to_cart"],
        guardrails=["checkout_latency_ms"],
        inference=registered_spec(),
    )
    assert plan.role_names() == {
        "conversion_rate": "primary",
        "revenue_per_user": "secondary",
        "add_to_cart": "secondary",
        "checkout_latency_ms": "guardrail",
    }


def test_always_valid_without_secondaries_allowed():
    plan = AnalysisPlan(
        primary="conversion_rate",
        guardrails=["checkout_latency_ms"],
        inference=registered_spec(),
    )
    assert plan.inference is not None
    assert plan.inference.kind == "always_valid"


# ── alpha / q bounds ─────────────────────────────────────────────────────


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_alpha_out_of_bounds_refused(alpha: float):
    with pytest.raises(ValidationError):
        AnalysisPlan(alpha=alpha)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1])
def test_q_out_of_bounds_refused(q: float):
    with pytest.raises(ValidationError):
        AnalysisPlan(q=q)


# ── extra=forbid ─────────────────────────────────────────────────────────


def test_analysis_plan_forbids_extra_fields():
    with pytest.raises(ValidationError) as exc_info:
        AnalysisPlan(bogus_field="x")  # ty: ignore[unknown-argument]  # proving extra=forbid rejects it
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
    assert exc_info.value.errors()[0]["loc"] == ("bogus_field",)


# ── InferenceSpec: field/kind mismatch ──────────────────────────────────


def test_registered_inference_round_trip_preserves_actual_decision():
    from increment import AlwaysValid, estimate_sequential
    from tests.sequential_cases import capture, records, registration

    spec = InferenceSpec(kind="always_valid", registration=registration("bernoulli"))
    assert spec.registration is not None
    restored = InferenceSpec.model_validate_json(spec.model_dump_json())
    assert restored.registration is not None
    policy = AlwaysValid(registration=restored.registration)
    rows = records([0, 0, 0, 1] * 24, [0, 1, 1, 1] * 24)
    result = estimate_sequential(capture(restored.registration, rows), policy).results[0]
    assert result.stat_sig()
    assert result.require_sequential_result().checkpoint.model == spec.registration.models[0]


@pytest.mark.parametrize(
    "field,value",
    [("effect_scale", 0.2), ("n_max", 1000), ("prior_looks", [120, 240]), ("first_analysis", True)],
)
def test_obsolete_inference_fields_refuse_before_any_source_iteration(field, value):
    from tests.sequential_cases import registered_spec

    payload = {**registered_spec().model_dump(), field: value}
    with pytest.raises(ValidationError) as exc_info:
        InferenceSpec.model_validate(payload)
    assert {error["type"] for error in exc_info.value.errors()} == {"extra_forbidden"}
    assert {error["loc"] for error in exc_info.value.errors()} == {(field,)}


# ── ExperimentMetric margins (per-experiment tolerance) ─────────────────


def test_experiment_metric_margin_and_margin_abs_mutually_exclusive():
    with pytest.raises(DefinitionError) as exc_info:
        ExperimentMetric(metric="m", margin=0.01, margin_abs=0.02)
    assert exc_info.value.code == "definition.experiment.binding_margin_margin"


def test_experiment_metric_margin_must_be_positive():
    with pytest.raises(ValidationError) as exc_info:
        ExperimentMetric(metric="m", margin=0.0)
    assert exc_info.value.errors()[0]["type"] == "greater_than"
    assert exc_info.value.errors()[0]["loc"] == ("margin",)


def test_experiment_metric_margin_abs_must_be_positive():
    with pytest.raises(ValidationError) as exc_info:
        ExperimentMetric(metric="m", margin_abs=-1.0)
    assert exc_info.value.errors()[0]["type"] == "greater_than"
    assert exc_info.value.errors()[0]["loc"] == ("margin_abs",)


def test_experiment_metric_margin_alone_allowed():
    binding = ExperimentMetric(metric="checkout_latency_ms", margin=0.01)
    assert binding.margin == 0.01
    assert binding.margin_abs is None


def test_experiment_metric_margin_abs_alone_allowed():
    binding = ExperimentMetric(metric="checkout_latency_ms", margin_abs=5.0)
    assert binding.margin_abs == 5.0
    assert binding.margin is None
