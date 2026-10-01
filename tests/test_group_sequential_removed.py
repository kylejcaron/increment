"""The finite-look inference kind is gone; its payloads refuse by name."""

from __future__ import annotations

from fractions import Fraction as F
from typing import Any, cast

import pytest
from pydantic import ValidationError

from increment.errors import CapabilityError, WireFormatError
from increment.semantics.models import InferenceSpec


def test_inference_spec_rejects_the_group_sequential_kind():
    from tests.sequential_cases import registration

    with pytest.raises(ValidationError) as raised:
        InferenceSpec.model_validate({"kind": "group_sequential", "registration": registration()})
    assert raised.value.errors()[0]["loc"] == ("kind",)


def test_wire_group_sequential_payload_refuses_with_the_invalid_inference_code():
    from increment.decision_wire import compiled_plan_from_dict, compiled_plan_to_dict
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.semantics.models import AnalysisPlan, ConversionMetric
    from tests.sequential_cases import registration

    reg = registration("bernoulli")
    metric = ConversionMetric(name="outcome", entity="unit_id", fact="outcome")
    plan = compile_decision_plan(
        AnalysisPlan(
            primary="outcome", inference=InferenceSpec(kind="always_valid", registration=reg)
        ),
        [metric],
        path="frame",
        design=Randomized(control_group="control"),
    )
    payload = cast(dict[str, Any], compiled_plan_to_dict(plan))
    # A finite-look plan as an earlier release serialized it: the registration
    # carries the schedule and every procedure repeats the plan inference.
    finite = {
        "kind": "group_sequential",
        "registration": {
            **payload["inference"]["registration"],
            "finite_looks": [200, 1000],
            "cumulative_spending": ["1/10", "1"],
        },
        "looks": [200, 1000],
        "cumulative_spending": ["1/10", "1"],
        "policy_version": "likelihood_spending_v1",
    }
    payload["inference"] = finite
    for procedure in payload["procedures"].values():
        procedure["inference"] = finite
    with pytest.raises(WireFormatError) as raised:
        compiled_plan_from_dict(payload)
    assert raised.value.code == "wire.procedure.invalid_inference"
    assert raised.value.context["inference"] == "group_sequential"


def test_registration_payload_carrying_finite_looks_refuses_as_legacy():
    from increment import SequentialRegistration
    from tests.sequential_cases import registration

    payload = {**registration().model_dump(), "finite_looks": [200], "cumulative_spending": [F(1)]}
    with pytest.raises(CapabilityError) as raised:
        SequentialRegistration.model_validate(payload)
    assert raised.value.code == "sequential.continuation.legacy"
