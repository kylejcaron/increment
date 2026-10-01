import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import pytest

from increment import SwitchbackStudyEnvelope
from increment.decision import (
    ArmAnalysisState,
    ContrastAnalysisState,
    ContrastContext,
    ContrastDecisionProcedure,
    FixedInference,
    NoFamily,
    compile_contrast_procedures,
)
from increment.errors import CapabilityError, InvalidRequestError
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized
from increment.semantics.models import (
    AnalysisPlan,
    ExperimentMetric,
    MeanMetric,
    MethodSpec,
    NormalPriorSpec,
)
from increment.semantics.unit_cycle import UnitCycleTApproximation
from tests.sequential_cases import registered_spec


def metric(name="revenue"):
    return MeanMetric(name=name, entity="user", fact="revenue", aggregation="sum")


def study():
    return SwitchbackStudyEnvelope(
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        ),
    )


def test_fixed_contrast_procedure_rejects_one_sided_alpha_at_half():
    with pytest.raises(InvalidRequestError) as raised:
        ContrastDecisionProcedure(
            metric="m", role="primary", alternative="greater", null_abs=0, alpha=0.5
        )
    assert raised.value.code == "decision.arm_decision.one_sided_alpha"
    p = ContrastDecisionProcedure(
        metric="m", role="primary", alternative="greater", null_abs=0, alpha=0.49
    )
    assert p.family == NoFamily()
    assert p.inference == FixedInference()


def test_compile_contrast_procedures_is_fixed_and_metric_keyed():
    plan = AnalysisPlan(alpha=0.04, alternative="greater", primary="revenue")
    procedures = compile_contrast_procedures(plan, (metric(),))
    assert procedures["revenue"].role == "primary"
    assert procedures["revenue"].alpha == 0.04
    assert procedures["revenue"].family == NoFamily()
    assert procedures["revenue"].inference == FixedInference()


def test_compile_binds_prospective_references_by_metric_without_mutable_aliases():
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(metric="revenue")
    references = {"revenue": reference}
    compiled = compile_contrast_procedures(None, (metric(),), contrast_references=references)
    references.clear()
    assert compiled["revenue"].reference == reference
    with pytest.raises(TypeError):
        cast("dict[str, ContrastDecisionProcedure]", compiled)["revenue"] = compiled["revenue"]


def test_procedure_rejects_a_reference_for_another_metric_at_construction():
    from tests.estimation.test_unit_cycle_envelope import envelope

    with pytest.raises(InvalidRequestError) as caught:
        ContrastDecisionProcedure(
            metric="other",
            role="primary",
            alternative="two-sided",
            null_abs=0,
            alpha=0.05,
            reference=envelope(metric="revenue"),
        )
    assert caught.value.code == "decision.contrast_decision.reference_metric"


def test_compile_splits_alpha_across_primaries():
    procedures = compile_contrast_procedures(
        AnalysisPlan(alpha=0.04, primary=["revenue", "orders"]),
        (metric(), metric("orders")),
    )
    assert procedures["revenue"].alpha == 0.02
    assert procedures["orders"].alpha == 0.02


def test_compile_rejects_secondary_family_and_sequential_inference():
    with pytest.raises(CapabilityError) as family:
        compile_contrast_procedures(
            AnalysisPlan(secondaries=["revenue"]),
            (metric(),),
        )
    assert family.value.code == "source.frame.switchback.plan"
    assert family.value.context["reason"] == "family_inference"
    with pytest.raises(CapabilityError) as inference:
        compile_contrast_procedures(
            AnalysisPlan(
                primary="revenue",
                inference=registered_spec(),
            ),
            (metric(),),
        )
    assert inference.value.code == "source.frame.switchback.plan"
    assert inference.value.context["reason"] == "unsupported_inference"


def test_compile_rejects_unknown_plan_metrics_with_contrast_path_label():
    with pytest.raises(InvalidRequestError) as raised:
        compile_contrast_procedures(AnalysisPlan(primary="missing"), (metric(),))
    assert raised.value.code == "plan.metrics.unknown"
    assert raised.value.context["path"] == "frame/contrast"


@pytest.mark.parametrize("role", ["primary", "secondaries", "guardrails"])
@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("decision_method", MethodSpec(name="unadjusted")),
        ("prior", NormalPriorSpec(mu=0.0, sigma=1.0)),
    ],
)
def test_compile_rejects_plan_bound_estimation_overrides_for_every_role(role, override, value):
    declared = metric().model_copy(update={"preferred_direction": "increase"})
    entry = ExperimentMetric(metric="revenue", **{override: value})
    if role == "primary":
        plan = AnalysisPlan(primary=entry)
    elif role == "secondaries":
        plan = AnalysisPlan(secondaries=[entry])
    else:
        plan = AnalysisPlan(guardrails=[entry])

    with pytest.raises(CapabilityError) as raised:
        compile_contrast_procedures(plan, (declared,))
    assert raised.value.code == "source.frame.switchback.plan"
    assert raised.value.context["reason"] == "unsupported_plan_override"
    assert raised.value.context["fields"] == (override,)


def test_compile_rejects_sensitivity_only_plan_entry_with_field_context():
    plan = AnalysisPlan(
        primary=ExperimentMetric(
            metric="revenue",
            sensitivity_methods=(MethodSpec(name="sensitivity"),),
        )
    )

    with pytest.raises(ValueError) as raised:
        compile_contrast_procedures(plan, (metric(),))

    assert raised.value.code == "source.frame.switchback.plan"  # ty: ignore[unresolved-attribute]
    assert raised.value.context["metric"] == "revenue"  # ty: ignore[unresolved-attribute]
    assert raised.value.context["fields"] == ("sensitivity_methods",)  # ty: ignore[unresolved-attribute]


def test_compile_uses_metric_direction_for_marginless_guardrail():
    guardrail = MeanMetric(
        name="revenue",
        entity="user",
        fact="revenue",
        aggregation="sum",
        preferred_direction="decrease",
    )
    procedure = compile_contrast_procedures(
        AnalysisPlan(guardrails=["revenue"]),
        (guardrail,),
    )["revenue"]
    assert procedure.alternative == "less"


def test_compile_guardrail_ignores_plan_wide_alternative():
    guardrail = metric().model_copy(update={"preferred_direction": "decrease", "margin_abs": 1.0})
    procedure = compile_contrast_procedures(
        AnalysisPlan(alternative="greater", guardrails=["revenue"]),
        (guardrail,),
    )["revenue"]
    assert procedure.alternative == "less"
    assert procedure.null_abs == 1.0


@pytest.mark.parametrize(
    ("declared", "primary"),
    [
        (metric().model_copy(update={"preferred_direction": "increase", "margin": 0.1}), "revenue"),
        (
            metric().model_copy(update={"preferred_direction": "increase"}),
            ExperimentMetric(metric="revenue", margin=0.1),
        ),
        (
            metric().model_copy(
                update={"preferred_direction": "increase", "margin": 0.1, "margin_abs": 1.0}
            ),
            "revenue",
        ),
        (
            metric().model_copy(update={"preferred_direction": "increase", "margin_abs": 1.0}),
            ExperimentMetric(metric="revenue", margin=0.1),
        ),
    ],
)
def test_compile_rejects_relative_and_mixed_margin_declarations(declared, primary):
    with pytest.raises(CapabilityError) as raised:
        compile_contrast_procedures(AnalysisPlan(primary=primary), (declared,))
    assert raised.value.code == "source.frame.switchback.plan"
    assert raised.value.context["reason"] == "relative_margin"


def test_compile_uses_margin_abs_tail_and_binding_precedence():
    declared = metric().model_copy(update={"preferred_direction": "increase", "margin_abs": 1.0})
    declared_procedure = compile_contrast_procedures(AnalysisPlan(primary="revenue"), (declared,))[
        "revenue"
    ]
    assert declared_procedure.null_abs == -1.0
    assert declared_procedure.alternative == "greater"

    explicit_plan = compile_contrast_procedures(
        AnalysisPlan(alternative="less", primary="revenue"), (declared,)
    )["revenue"]
    assert explicit_plan.alternative == "less"

    bound = compile_contrast_procedures(
        AnalysisPlan(primary=ExperimentMetric(metric="revenue", margin_abs=2.0)),
        (declared,),
    )["revenue"]
    assert bound.null_abs == -2.0
    assert bound.alternative == "greater"


def test_compiled_and_context_procedures_are_mapping_proxies():
    compiled = compile_contrast_procedures(AnalysisPlan(primary="revenue"), (metric(),))
    assert isinstance(compiled, MappingProxyType)
    with pytest.raises(TypeError):
        dict.__setitem__(compiled, "other", compiled["revenue"])  # type: ignore[arg-type]


def test_contrast_context_defensively_copies_procedures():
    procedures = {
        "revenue": ContrastDecisionProcedure(
            metric="revenue", role="primary", alternative="two-sided", null_abs=0, alpha=0.05
        )
    }
    context = ContrastContext(
        study_id="s", study=study(), metrics=(metric(),), procedures=procedures
    )
    procedures.clear()
    assert "revenue" in context.procedures
    assert isinstance(context.procedures, MappingProxyType)
    with pytest.raises(TypeError):
        dict.__setitem__(context.procedures, "other", context.procedures["revenue"])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        context.procedures["x"] = context.procedures["revenue"]  # type: ignore[index]


@pytest.mark.parametrize("names", [(), ("orders",), ("revenue", "orders")])
def test_contrast_context_requires_exact_procedure_metrics(names):
    procedures = compile_contrast_procedures(None, tuple(metric(name) for name in names))
    with pytest.raises(InvalidRequestError) as caught:
        ContrastContext(study_id="s", study=study(), metrics=(metric(),), procedures=procedures)
    assert caught.value.code == "decision.contrast_context.procedure_metrics"
    assert caught.value.context == {
        "missing": tuple(sorted({"revenue"} - set(names))),
        "undeclared": tuple(sorted(set(names) - {"revenue"})),
    }


@pytest.mark.parametrize("deserialize", [False, True])
@pytest.mark.parametrize(
    "names", [("revenue", "revenue"), ("revenue", "orders", "orders", "revenue")]
)
def test_contrast_context_rejects_duplicate_metric_names(deserialize, names):
    context = ContrastContext(
        study_id="s",
        study=study(),
        metrics=tuple(metric(name) for name in sorted(set(names))),
        procedures=compile_contrast_procedures(None, tuple(metric(name) for name in set(names))),
    )
    metrics = tuple(metric(name) for name in names)
    with pytest.raises(InvalidRequestError) as caught:
        if deserialize:
            payload = context.model_dump(mode="json")
            payload["metrics"] = [item.model_dump(mode="json") for item in metrics]
            ContrastContext.model_validate_json(json.dumps(payload))
        else:
            ContrastContext(
                study_id=context.study_id,
                study=context.study,
                metrics=metrics,
                procedures=context.procedures,
            )
    assert caught.value.code == "decision.contrast_context.duplicate_metrics"
    assert cast("tuple[str, ...]", caught.value.context["metrics"]) == tuple(sorted(set(names)))


def test_contrast_context_rejects_procedure_mapping_key_mismatch():
    procedure = compile_contrast_procedures(None, (metric("orders"),))["orders"]
    with pytest.raises(InvalidRequestError) as caught:
        ContrastContext(
            study_id="s", study=study(), metrics=(metric(),), procedures={"revenue": procedure}
        )
    assert caught.value.code == "decision.compiled_decision.procedure_mapping_key"
    assert caught.value.context["key"] == "revenue"
    assert caught.value.context["metric"] == "orders"


@pytest.mark.parametrize("mismatch", ["missing", "extra", "unbound", "kind", "variance"])
def test_contrast_context_rejects_reference_disagreement(mismatch):
    from tests.estimation.test_unit_cycle_envelope import envelope

    reference = envelope(p=0.5, metric="revenue")
    procedures = compile_contrast_procedures(
        None,
        (metric(),),
        contrast_references={} if mismatch == "unbound" else {"revenue": reference},
    )
    references = {"revenue": reference}
    if mismatch == "missing":
        references.clear()
    elif mismatch == "extra":
        references["orders"] = UnitCycleTApproximation()
    elif mismatch == "kind":
        references["revenue"] = UnitCycleTApproximation()
    elif mismatch == "variance":
        references["revenue"] = envelope(p=0.5, metric="revenue", variance=0.02)
    with pytest.raises(InvalidRequestError) as caught:
        ContrastContext(
            study_id="s",
            study=study(),
            metrics=(metric(),),
            procedures=procedures,
            contrast_references=references,
        )
    assert caught.value.code == "decision.contrast_context.reference_mismatch"
    assert caught.value.context["metrics"] == ("orders" if mismatch == "extra" else "revenue",)


def test_contrast_context_copies_and_roundtrips_matching_mixed_references():
    from tests.estimation.test_unit_cycle_envelope import envelope

    metrics = (metric(), metric("orders"), metric("unbound"))
    references = {
        "revenue": envelope(p=0.5, metric="revenue"),
        "orders": UnitCycleTApproximation(),
    }
    procedures = dict(compile_contrast_procedures(None, metrics, contrast_references=references))
    context = ContrastContext(
        study_id="s",
        study=study(),
        metrics=metrics,
        procedures=procedures,
        contrast_references=references,
    )
    references.clear()
    procedures.clear()
    assert set(context.contrast_references) == {"revenue", "orders"}
    for name, reference in context.contrast_references.items():
        assert reference == context.procedures[name].reference
    assert context.procedures["unbound"].reference is None
    with pytest.raises(TypeError):
        cast("dict[str, object]", context.contrast_references)["orders"] = UnitCycleTApproximation()
    assert ContrastContext.model_validate_json(context.model_dump_json()) == context


@dataclass
class Source:
    context: ContrastContext

    def contrast_stats(self, metric):
        raise NotImplementedError

    def planning_baseline(self, metric):
        raise NotImplementedError


def test_contrast_analysis_state_requires_contrast_context_and_switchback():
    source = Source(
        ContrastContext(
            study_id="s",
            study=study(),
            metrics=(metric(),),
            procedures=compile_contrast_procedures(None, (metric(),)),
        )
    )
    state = ContrastAnalysisState(source=source, experiment_name="exp")
    assert state.family == "contrast"
    assert isinstance(state, ContrastAnalysisState)
    assert ArmAnalysisState is not ContrastAnalysisState
    with pytest.raises(AssertionError):
        ContrastAnalysisState(
            source=object(),  # ty: ignore[invalid-argument-type] -- deliberately wrong type to trip an internal assertion
            experiment_name="exp",
        )
