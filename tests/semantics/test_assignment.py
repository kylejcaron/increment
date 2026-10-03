import pytest
from pydantic import TypeAdapter, ValidationError

from increment import ParallelStudyEnvelope, SwitchbackStudyEnvelope
from increment.errors import InvalidRequestError
from increment.semantics.assignment import (
    Assignment,
    IndependentBernoulliOrder,
    ParallelAssignment,
    SharedScheduleOrder,
    SwitchbackAssignment,
    SwitchbackWindow,
)
from increment.semantics.design import Randomized


def test_assignment_union_dispatches_orthogonal_models():
    parallel = TypeAdapter(Assignment).validate_python({"kind": "parallel"})
    switchback = TypeAdapter(Assignment).validate_python(
        {
            "kind": "switchback",
            "sequence": {"scheme": "independent_bernoulli_order"},
            "window": {"washout_steps": 1, "observation_steps": 2},
        }
    )
    assert isinstance(parallel, ParallelAssignment)
    assert isinstance(switchback, SwitchbackAssignment)
    assert isinstance(switchback.sequence, IndependentBernoulliOrder)
    assert switchback.sequence.probability_ct == 0.5


def test_switchback_sequence_dispatches_shared_schedule_order():
    """The sequence union's ``scheme`` discriminator resolves a declared
    shared schedule to ``SharedScheduleOrder``, not the independent scheme."""
    switchback = TypeAdapter(Assignment).validate_python(
        {
            "kind": "switchback",
            "sequence": {"scheme": "shared_schedule", "probability_ct": 0.6},
            "window": {"washout_steps": 1, "observation_steps": 2},
        }
    )
    assert isinstance(switchback.sequence, SharedScheduleOrder)
    assert switchback.sequence.probability_ct == 0.6
    assert switchback.sequence.independence_unit == "shared_block"


def test_switchback_assignment_rejects_sequence_and_window_violations():
    with pytest.raises(ValidationError):
        IndependentBernoulliOrder(probability_ct=0.0)
    with pytest.raises(ValidationError):
        IndependentBernoulliOrder(probability_ct=1.0)
    with pytest.raises(ValidationError):
        SharedScheduleOrder(probability_ct=0.0)
    with pytest.raises(ValidationError):
        SharedScheduleOrder(probability_ct=1.0)
    with pytest.raises(ValidationError):
        SwitchbackWindow(washout_steps=-1, observation_steps=1)
    with pytest.raises(ValidationError):
        SwitchbackWindow(washout_steps=0, observation_steps=0)
    with pytest.raises(ValidationError):
        SwitchbackAssignment(
            periods_per_cycle=3,  # ty: ignore[invalid-argument-type]
            sequence=IndependentBernoulliOrder(),
            window=SwitchbackWindow(washout_steps=0, observation_steps=1),
        )


def test_parallel_envelope_keeps_identification_separate_from_assignment():
    envelope = ParallelStudyEnvelope(identification=Randomized(control_group="control"))
    assert envelope.assignment == ParallelAssignment()
    assert envelope.identification.control_group == "control"
    with pytest.raises(ValidationError):
        envelope.assignment = ParallelAssignment()  # ty: ignore[invalid-assignment]
    with pytest.raises(ValidationError):
        ParallelStudyEnvelope(
            identification=Randomized(control_group="control"),
            typo=True,  # ty: ignore[unknown-argument]
        )


def test_switchback_envelope_requires_one_treatment_arm_and_randomized_identification():
    envelope = SwitchbackStudyEnvelope(
        identification=Randomized(
            control_group="control", allocation={"control": 0.5, "treatment": 0.5}
        ),
        assignment=SwitchbackAssignment(
            sequence=IndependentBernoulliOrder(),
            window=SwitchbackWindow(washout_steps=1, observation_steps=2),
        ),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        SwitchbackStudyEnvelope(
            identification=Randomized(control_group="control"),
            assignment=envelope.assignment,
        )
    assert exc_info.value.code == "facade.study.switchback_study.identification_allocation_exactly"
    with pytest.raises(ValidationError) as raised:
        SwitchbackStudyEnvelope.model_validate(
            {
                "kind": "switchback",
                "identification": {
                    "mechanism": "observational",
                    "control_group": "control",
                    "adjustment": {"covariates": ["x"]},
                },
                "assignment": envelope.assignment.model_dump(),
            }
        )
    assert {error["loc"] for error in raised.value.errors()} == {
        ("identification", "mechanism"),
        ("identification", "adjustment"),
    }
    with pytest.raises(InvalidRequestError) as exc_info:
        SwitchbackStudyEnvelope(
            identification=Randomized(
                control_group="control", allocation={"control": 0.5, "a": 0.25, "b": 0.25}
            ),
            assignment=envelope.assignment,
        )
    assert exc_info.value.code == "facade.study.switchback_study.identification_exactly_one"


@pytest.mark.parametrize(
    "allocation",
    [
        {"control": 0.9, "treatment": 0.1},
        {"control": 0.4, "treatment": 0.4},
        {"control": 0.5, "treatment": 0.5000001},
    ],
)
def test_switchback_allocation_refuses_identically_at_envelope_and_panel(allocation):
    """One hazard, one code and one context, whichever ingress meets it first."""
    from increment.switchback import from_switchback_panel

    assignment = SwitchbackAssignment(
        sequence=IndependentBernoulliOrder(),
        window=SwitchbackWindow(washout_steps=1, observation_steps=2),
    )
    identification = Randomized(control_group="control", allocation=allocation)
    with pytest.raises(InvalidRequestError) as envelope:
        SwitchbackStudyEnvelope(identification=identification, assignment=assignment)
    with pytest.raises(InvalidRequestError) as panel:
        from_switchback_panel(
            object(),  # ty: ignore[invalid-argument-type]
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="group",
            metrics={"outcome": "mean"},
            identification=identification,
            assignment=assignment,
        )
    assert envelope.value.code == panel.value.code == "source.frame.switchback.identification"
    assert envelope.value.context == panel.value.context
    assert envelope.value.context["allocation"] == allocation


def test_switchback_window_rejects_bool_steps():
    """bool is an int subclass in Python; washout/observation steps must
    never silently become 1/0 from True/False."""
    with pytest.raises(ValidationError) as exc_info:
        SwitchbackWindow(washout_steps=True, observation_steps=1)
    assert exc_info.value.errors()[0]["ctx"]["error"].code == "definition.assignment.reject_bool"
    with pytest.raises(ValidationError) as exc_info:
        SwitchbackWindow(washout_steps=0, observation_steps=True)
    assert exc_info.value.errors()[0]["ctx"]["error"].code == "definition.assignment.reject_bool"


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize(
    "sequence",
    [IndependentBernoulliOrder(probability_ct=0.5), SharedScheduleOrder(probability_ct=0.5)],
)
def test_switchback_window_accepts_carryover_orders_zero_one_two_under_both_laws(order, sequence):
    """Orders 0/1/2 construct under both the independent and shared laws
    whenever enough observation steps remain past the declared order."""
    window = SwitchbackWindow(washout_steps=1, observation_steps=order + 1, carryover_order=order)
    assignment = SwitchbackAssignment(sequence=sequence, window=window)
    assert assignment.window.carryover_order == order
    # Retained length is observation_steps - carryover_order, and must stay
    # nonempty for every order this construction accepts.
    assert assignment.window.observation_steps - assignment.window.carryover_order >= 1


def test_switchback_window_rejects_bool_carryover_order():
    with pytest.raises(ValidationError) as exc_info:
        SwitchbackWindow(washout_steps=0, observation_steps=2, carryover_order=True)
    assert exc_info.value.errors()[0]["ctx"]["error"].code == "definition.assignment.reject_bool"


def test_switchback_window_rejects_nonintegral_carryover_order():
    with pytest.raises(ValidationError) as exc_info:
        SwitchbackWindow(washout_steps=0, observation_steps=2, carryover_order=1.5)
    assert (
        exc_info.value.errors()[0]["ctx"]["error"].code
        == "definition.assignment.carryover_order_type"
    )


def test_switchback_window_rejects_negative_carryover_order():
    with pytest.raises(ValidationError) as exc_info:
        SwitchbackWindow(washout_steps=0, observation_steps=2, carryover_order=-1)
    assert (
        exc_info.value.errors()[0]["ctx"]["error"].code
        == "definition.assignment.carryover_order_range"
    )


def test_switchback_window_rejects_carryover_order_at_or_past_observation_steps():
    """An order at or past observation_steps would empty the retained
    window entirely; both boundaries are refused with the same coded range
    error naming the offending order and observation_steps."""
    with pytest.raises(ValidationError) as exc_info:
        SwitchbackWindow(washout_steps=0, observation_steps=2, carryover_order=2)
    error = exc_info.value.errors()[0]["ctx"]["error"]
    assert error.code == "definition.assignment.carryover_order_range"
    assert error.context["carryover_order"] == 2
    assert error.context["observation_steps"] == 2
    with pytest.raises(ValidationError):
        SwitchbackWindow(washout_steps=0, observation_steps=2, carryover_order=3)


def test_switchback_window_default_carryover_order_is_zero():
    window = SwitchbackWindow(washout_steps=1, observation_steps=2)
    assert window.carryover_order == 0
