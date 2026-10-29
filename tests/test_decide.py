from __future__ import annotations

from increment import AnalysisPlan
from increment.decide import Registration, register
from increment.frame import MetricSpec, synthesise_metric


def test_register_hashes_compiled_plan_without_opening_a_source():
    metric = synthesise_metric(MetricSpec(name="revenue", type="mean"))
    plan = AnalysisPlan(primary="revenue")

    registration = register(plan, [metric], path="frame")
    repeated = register(plan, [metric], path="frame")

    assert isinstance(registration, Registration)
    assert registration == repeated


def test_compiled_plan_binding_is_versioned_and_validates_registration():
    from increment.decision_wire import compiled_plan_from_dict, compiled_plan_to_dict
    from increment.kernel.contracts import CompiledPlanBinding
    from increment.plan import compile_decision_plan

    metric = synthesise_metric(MetricSpec(name="revenue", type="mean"))
    plan = AnalysisPlan(primary="revenue")
    compiled = compile_decision_plan(plan, [metric], path="frame")
    registration = Registration.from_plan(compiled)

    binding = CompiledPlanBinding.from_plan(registration, compiled)

    assert binding.version == 1
    restored = compiled_plan_from_dict(binding.plan_payload)
    assert registration.matches(restored)
    assert binding.plan_payload == compiled_plan_to_dict(compiled)


def test_compiled_plan_binding_rejects_changed_payload_with_same_hash():
    import pytest

    from increment.decision_wire import compiled_plan_to_dict
    from increment.errors import CodedError
    from increment.kernel.contracts import CompiledPlanBinding
    from increment.plan import compile_decision_plan

    metric = synthesise_metric(MetricSpec(name="revenue", type="mean"))
    compiled = compile_decision_plan(AnalysisPlan(primary="revenue"), [metric], path="frame")
    registration = Registration.from_plan(compiled)
    payload = compiled_plan_to_dict(compiled)
    payload["alpha"] = 0.2

    with pytest.raises(CodedError):
        CompiledPlanBinding(registration.registration_hash, 1, payload)
