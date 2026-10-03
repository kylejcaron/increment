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


def allocation_defect(control_weight: object, treatment_weight: object) -> AllocationDefect | None:
    """The one switchback allocation contract: finite nonnegative weights that sum
    to one (within ``1e-12``) and are exactly 0.5/0.5. ``None`` when satisfied."""
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
    if control != 0.5 or treatment != 0.5:
        return "unequal_allocation"
    return None


def refuse_allocation_defect(
    defect: AllocationDefect,
    *,
    allocation: object,
    control_group: str,
    treatment_group: str,
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
        message="switchback allocation requires exact 0.5/0.5 control/treatment weights",
        allocation=allocation,
        control_group=control_group,
        treatment_group=treatment_group,
        reason="unequal_allocation",
    )


class ParallelStudyEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["parallel"] = "parallel"
    identification: Design
    assignment: ParallelAssignment = ParallelAssignment()


class SwitchbackStudyEnvelope(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["switchback"] = "switchback"
    identification: Randomized
    assignment: SwitchbackAssignment

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
        defect = allocation_defect(allocation[control], allocation[treatment])
        if defect is not None:
            refuse_allocation_defect(
                defect,
                allocation=allocation,
                control_group=str(control),
                treatment_group=str(treatment),
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
