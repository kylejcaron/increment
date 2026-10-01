"""Private normalized study envelopes used by each evidence family."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
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
        arms = tuple(allocation)
        if self.identification.control_group not in allocation:
            _raise("facade.study.switchback_study.allocation_include_its")
        treatment_arms = [arm for arm in arms if arm != self.identification.control_group]
        if len(treatment_arms) != 1:
            _raise("facade.study.switchback_study.identification_exactly_one")
        return self


StudyEnvelope = Annotated[
    ParallelStudyEnvelope | SwitchbackStudyEnvelope,
    Field(discriminator="kind"),
]


__all__ = [
    "ParallelStudyEnvelope",
    "StudyEnvelope",
    "SwitchbackStudyEnvelope",
]
