"""Private normalized study envelopes used by each evidence family."""

from __future__ import annotations

import math
from numbers import Real
from typing import Annotated, Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.semantics.assignment import (
    ParallelAssignment,
    SwitchbackAssignment,
)
from increment.semantics.design import Design, Randomized

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.study.switchback_study.identification_allocation_exactly": "switchback identification requires allocation with exactly one treatment arm",
        "facade.study.switchback_study.allocation_include_its": "switchback allocation must include its declared control_group",
        "facade.study.switchback_study.identification_exactly_one": "switchback identification requires exactly one treatment arm",
    },
)
_raise = raiser(_REFUSALS)


def _render_message(*, message: str, **_: object) -> str:
    return message


#: Shared by ``SwitchbackStudyEnvelope`` (construction) and ``from_switchback_panel``:
#: one hazard, one code, whichever ingress meets it first.
SWITCHBACK_IDENTIFICATION = RefusalSpec(
    "source.frame.switchback.identification",
    InvalidRequestError,
    _render_message,
)

AllocationDefect = Literal["not_finite_nonnegative", "not_normalized", "unequal_allocation"]


def allocation_defect(
    control_weight: object, treatment_weight: object, assignment: SwitchbackAssignment
) -> AllocationDefect | None:
    """Check that the declared allocation agrees with the schedule's exposure."""
    if (
        not isinstance(control_weight, Real)
        or not isinstance(treatment_weight, Real)
        or not math.isfinite(float(control_weight))
        or not math.isfinite(float(treatment_weight))
        or float(control_weight) < 0.0
        or float(treatment_weight) < 0.0
    ):
        return "not_finite_nonnegative"
    control = float(control_weight)
    treatment = float(treatment_weight)
    if not math.isclose(control + treatment, 1.0, rel_tol=0.0, abs_tol=1e-12):
        return "not_normalized"
    expected_control, expected_treatment = assignment.exposure
    if assignment.sequence.scheme == "fixed_count":
        # Accept either m/K or its floating-point complement for each arm.
        if not (
            control > 0
            and treatment > 0
            and control in (expected_control, 1 - expected_treatment)
            and treatment in (expected_treatment, 1 - expected_control)
        ):
            return "unequal_allocation"
    elif control != expected_control or treatment != expected_treatment:
        return "unequal_allocation"
    return None


def refuse_allocation_defect(
    defect: AllocationDefect,
    *,
    allocation: object,
    control_group: str,
    treatment_group: str,
    assignment: SwitchbackAssignment,
) -> NoReturn:
    """Raise ``source.frame.switchback.identification`` for *defect*."""
    if defect == "not_finite_nonnegative":
        refuse(
            SWITCHBACK_IDENTIFICATION,
            message="switchback allocation weights must be finite and nonnegative",
            allocation=allocation,
            reason="invalid_allocation",
        )
    if defect == "not_normalized":
        refuse(
            SWITCHBACK_IDENTIFICATION,
            message="switchback allocation weights must sum to one",
            allocation=allocation,
            reason="invalid_allocation",
        )
    refuse(
        SWITCHBACK_IDENTIFICATION,
        message=(
            "switchback allocation must match the control and treatment shares of the cycle"
            if assignment.sequence.scheme == "fixed_count"
            else "switchback allocation requires exact 0.5/0.5 control/treatment weights"
        ),
        allocation=allocation,
        control_group=control_group,
        treatment_group=treatment_group,
        reason="unequal_allocation",
    )


class ParallelStudyEnvelope(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["parallel"] = "parallel"
    identification: Design
    assignment: ParallelAssignment = ParallelAssignment()


class SwitchbackObservationDesign(CodedModel, BaseModel):
    """Expected roster, observation horizon, and complete event feeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    units: tuple[str, ...]
    cycles: int = Field(ge=1, strict=True)
    complete_event_metrics: tuple[str, ...]


class SwitchbackStudyEnvelope(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["switchback"] = "switchback"
    identification: Randomized
    assignment: SwitchbackAssignment
    observation_design: SwitchbackObservationDesign | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def _exactly_one_treatment_arm(self) -> SwitchbackStudyEnvelope:
        allocation = self.identification.allocation
        if allocation is None:
            _raise("facade.study.switchback_study.identification_allocation_exactly")
        control = self.identification.control_group
        arms = tuple(allocation)
        if control not in allocation:
            _raise("facade.study.switchback_study.allocation_include_its")
        treatment_arms = [arm for arm in arms if arm != control]
        if len(treatment_arms) != 1:
            _raise("facade.study.switchback_study.identification_exactly_one")
        treatment = treatment_arms[0]
        defect = allocation_defect(allocation[control], allocation[treatment], self.assignment)
        if defect is not None:
            refuse_allocation_defect(
                defect,
                allocation=allocation,
                control_group=str(control),
                treatment_group=str(treatment),
                assignment=self.assignment,
            )
        return self


StudyEnvelope = Annotated[
    ParallelStudyEnvelope | SwitchbackStudyEnvelope,
    Field(discriminator="kind"),
]


__all__ = [
    "ParallelStudyEnvelope",
    "StudyEnvelope",
    "SwitchbackStudyEnvelope",
    "allocation_defect",
    "refuse_allocation_defect",
]
