"""Portable, immutable scientific declarations, independent of runtime data."""

from typing import Any, cast

import pytest
from pydantic import TypeAdapter, ValidationError

from increment.errors import InvalidRequestError
from increment.semantics.assignment import (
    IndependentBernoulliOrder,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.unit_cycle import (
    CenteredGammaInnovation,
    CenteredLognormalInnovation,
    NormalInnovation,
    ProspectiveAssumptionProvenance,
    UnitCycleCycleLaw,
    UnitCycleJointLaw,
    UnitCycleReference,
    UnitCycleTApproximation,
    UnitCycleTypeLaw,
    UnitCycleVarianceEnvelope,
)


def declaration():
    return {
        "assignment": SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(probability_ct=0.9),
            window=SwitchbackWindow(washout_steps=1, observation_steps=3, carryover_order=2),
        ),
        "metric": "outcome",
        "control_group": "control",
        "treatment_group": "treatment",
        "response_meaning": "retained_total",
        "provenance": ProspectiveAssumptionProvenance(
            assumption_id="population-law",
            assumption_version="1",
            justification="Independent prospective scientific argument.",
            declaration_id="example",
        ),
    }


def cycle():
    return UnitCycleCycleLaw(
        ct_mean=2,
        tc_mean=-2,
        ct_noise_load=1,
        tc_noise_load=-1,
        ct_innovation_count=2,
        tc_innovation_count=3,
    )


@pytest.mark.parametrize(
    "field", ["assumption_id", "assumption_version", "justification", "declaration_id"]
)
@pytest.mark.parametrize("value", ["", "   "])
def test_provenance_fields_must_be_nonempty(field, value):
    payload = declaration()["provenance"].model_dump()
    payload[field] = value
    with pytest.raises((InvalidRequestError, ValidationError)):
        ProspectiveAssumptionProvenance(**payload)


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
def test_cycle_counts_are_strict_positive_integers(value):
    expected = (
        "model.field.range"
        if isinstance(value, int) and not isinstance(value, bool)
        else "model.field.type"
    )
    with pytest.raises(InvalidRequestError) as raised:
        UnitCycleVarianceEnvelope(**declaration(), cycles_per_unit=value, residual_variance_upper=1)
    assert raised.value.code == expected
    with pytest.raises(InvalidRequestError) as raised:
        UnitCycleCycleLaw(**{**cycle().model_dump(), "ct_innovation_count": value})
    assert raised.value.code == expected


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_variance_upper_must_be_finite_nonnegative(value):
    expected = "model.field.range" if value == -1 else "model.field.nonfinite"
    with pytest.raises(InvalidRequestError) as raised:
        UnitCycleVarianceEnvelope(**declaration(), cycles_per_unit=1, residual_variance_upper=value)
    assert raised.value.code == expected


@pytest.mark.parametrize("innovation", [CenteredLognormalInnovation, CenteredGammaInnovation])
@pytest.mark.parametrize("shape", [0, -1, float("inf"), float("nan")])
def test_standardized_innovations_require_finite_positive_shape(innovation, shape):
    expected = "model.field.range" if shape in (0, -1) else "model.field.nonfinite"
    with pytest.raises(InvalidRequestError) as raised:
        innovation(shape=shape)
    assert raised.value.code == expected


@pytest.mark.parametrize(
    "reference",
    [
        UnitCycleTApproximation(),
        UnitCycleVarianceEnvelope(**declaration(), cycles_per_unit=1, residual_variance_upper=0),
    ],
)
def test_reference_discriminator_round_trip(reference):
    adapter = TypeAdapter(UnitCycleReference)
    restored = adapter.validate_json(adapter.dump_json(reference))
    assert type(restored) is type(reference)
    assert restored == reference


def test_additive_binary_conversion_is_a_coded_refusal():
    with pytest.raises(InvalidRequestError) as caught:
        UnitCycleVarianceEnvelope(
            **declaration(), cycles_per_unit=1, residual_variance_upper=1, aggregation="any"
        )
    assert caught.value.code == "definition.unit_cycle.aggregation"


def test_shared_assignment_refuses_and_nested_forged_assignment_is_revalidated():
    values = declaration()
    values["assignment"] = values["assignment"].model_copy(
        update={"sequence": SharedScheduleOrder(probability_ct=0.5)}
    )
    with pytest.raises(InvalidRequestError) as caught:
        UnitCycleVarianceEnvelope(**values, cycles_per_unit=1, residual_variance_upper=1)
    assert caught.value.code == "definition.unit_cycle.assignment"
    values = declaration()
    values["assignment"] = values["assignment"].model_copy(
        update={"sequence": IndependentBernoulliOrder().model_copy(update={"probability_ct": 0})}
    )
    with pytest.raises(ValidationError):
        UnitCycleVarianceEnvelope(**values, cycles_per_unit=1, residual_variance_upper=1)


def test_joint_law_copies_lists_and_relative_weights_need_no_probability_tolerance():
    cycles = [cycle()]
    types = [
        UnitCycleTypeLaw(weight=0.1, cycles=cycles),
        UnitCycleTypeLaw(weight=0.2, cycles=cycles),
    ]
    law = UnitCycleJointLaw(
        **declaration(), types=types, innovation=NormalInnovation(), reuse_probability=0.9
    )
    cycles.clear()
    types.clear()
    assert len(law.types) == 2 and len(law.types[0].cycles) == 1
    assert tuple(item.weight for item in law.types) == (0.1, 0.2)
    assert UnitCycleJointLaw.model_validate_json(law.model_dump_json()) == law
    with pytest.raises(ValidationError):
        cast(Any, law.types[0]).weight = 4


def test_every_type_must_have_the_same_nonempty_cycle_count():
    with pytest.raises(InvalidRequestError) as caught:
        UnitCycleTypeLaw(weight=1, cycles=[])
    assert caught.value.code == "model.field.length"
    with pytest.raises(InvalidRequestError) as caught:
        UnitCycleJointLaw(
            **declaration(),
            types=[
                UnitCycleTypeLaw(weight=1, cycles=[cycle()]),
                UnitCycleTypeLaw(weight=1, cycles=[cycle(), cycle()]),
            ],
            innovation=NormalInnovation(),
            reuse_probability=0,
        )
    assert caught.value.code == "definition.unit_cycle.cycles"


def test_discriminated_innovation_member_field_failure_is_coded():
    with pytest.raises(InvalidRequestError) as raised:
        UnitCycleJointLaw(
            **declaration(),
            types=[UnitCycleTypeLaw(weight=1, cycles=[cycle()])],
            innovation={"kind": "centered_gamma", "shape": 0},
            reuse_probability=1,
        )
    assert raised.value.code == "model.field.range"
    assert raised.value.context["model"] == "CenteredGammaInnovation"
    assert raised.value.context["field"] == "shape"


@pytest.mark.parametrize(
    "innovation",
    [
        NormalInnovation(),
        CenteredGammaInnovation(shape=0.01),
        CenteredLognormalInnovation(shape=40),
    ],
)
def test_innovation_choice_is_portable_without_sampling(innovation):
    law = UnitCycleJointLaw(
        **declaration(),
        types=[UnitCycleTypeLaw(weight=1, cycles=[cycle()])],
        innovation=innovation,
        reuse_probability=1,
    )
    restored = UnitCycleJointLaw.model_validate_json(law.model_dump_json())
    assert restored == law and type(restored.innovation) is type(innovation)
