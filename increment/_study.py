"""Private normalized study envelopes used by each evidence family."""

from __future__ import annotations

import math
from numbers import Real
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
        "facade.study.switchback_study.allocation_equal_split": "switchback allocation requires exact 0.5/0.5 control/treatment weights, got {allocation!r}",
    },
)
_raise = raiser(_REFUSALS)

AllocationDefect = Literal["not_finite_nonnegative", "not_normalized", "unequal_allocation"]


def allocation_defect(control_weight: object, treatment_weight: object) -> AllocationDefect | None:
    """The one switchback allocation contract: finite nonnegative weights that sum
    to one (within ``1e-12``) and are exactly 0.5/0.5. ``None`` when satisfied.

    Shared by ``SwitchbackStudyEnvelope`` (construction) and
    ``from_switchback_panel`` (which keeps its own message per defect).
    """
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
        if allocation_defect(
            allocation[self.identification.control_group], allocation[treatment_arms[0]]
        ):
            _raise(
                "facade.study.switchback_study.allocation_equal_split", allocation=dict(allocation)
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
]
