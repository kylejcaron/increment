"""Orthogonal assignment declarations for parallel and switchback studies."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from increment.errors import InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "definition.assignment.reject_bool": "expected an int, got a bool",
        "definition.assignment.carryover_order_type": "carryover_order must be a non-boolean int",
        "definition.assignment.carryover_order_range": "carryover_order={carryover_order} must satisfy 0 <= carryover_order < observation_steps={observation_steps}; an order at or past observation_steps would discard the entire retained window",
    },
)
_raise = raiser(_REFUSALS)
ASSIGNMENT_REJECT_BOOL = _REFUSALS["definition.assignment.reject_bool"]


def _reject_bool(v: object) -> object:
    """``bool`` is an ``int`` subclass in Python; a declared step count
    must never silently accept ``True``/``False`` as ``1``/``0``."""
    if isinstance(v, bool):
        _raise("definition.assignment.reject_bool")
    return v


class _AssignmentModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParallelAssignment(_AssignmentModel):
    kind: Literal["parallel"] = "parallel"


class IndependentBernoulliOrder(_AssignmentModel):
    """One independent Bernoulli CT/TC order draw per unit-cycle.

    Every unit draws its own order independently of every other unit, which
    is what makes the unit the independent replicate: cross-unit replication
    exists for a unit-level reference distribution.
    """

    scheme: Literal["independent_bernoulli_order"] = "independent_bernoulli_order"
    probability_ct: float = Field(default=0.5, gt=0.0, lt=1.0)
    independence_unit: Literal["unit_cycle"] = "unit_cycle"


class SharedScheduleOrder(_AssignmentModel):
    """One independent Bernoulli CT/TC order draw per two-period block,
    shared by every unit in a fixed, complete unit roster.

    Unlike :class:`IndependentBernoulliOrder`, no unit draws its own order:
    the whole roster follows one realized order per block, so the
    independent replicate is the block, not the unit. A common roster
    duplicated uniformly adds no independent blocks and therefore cannot
    change the mean-unit retained-window effect scale or inferential sample
    size. The switchback panel path reduces the roster within each block
    and uses ``switchback_block_t`` with a ``block_t`` reference. At least
    two independent blocks are required; a one-unit roster is supported.
    """

    scheme: Literal["shared_schedule"] = "shared_schedule"
    probability_ct: float = Field(gt=0.0, lt=1.0)
    independence_unit: Literal["shared_block"] = "shared_block"


AssignmentSequence = Annotated[
    IndependentBernoulliOrder | SharedScheduleOrder, Field(discriminator="scheme")
]


class SwitchbackWindow(_AssignmentModel):
    """A washout/observation window with a declared residual-carryover order.

    ``carryover_order`` names additional observation steps, immediately
    after the declared washout, still assumed contaminated by the prior
    period and therefore discarded before the retained window used for
    inference. Retained steps satisfy ``step >= washout_steps +
    carryover_order``; the retained window's length is ``observation_steps -
    carryover_order``. ``carryover_order=0`` is the plain
    no-residual-carryover-after-washout case. Declaring an order is an
    assumption, not something a balance diagnostic can prove true.
    """

    washout_steps: int = Field(ge=0)
    observation_steps: int = Field(ge=1)
    carryover_order: int = 0

    @field_validator("washout_steps", "observation_steps", mode="before")
    @classmethod
    def _validate_steps(cls, v: object) -> object:
        return _reject_bool(v)

    @field_validator("carryover_order", mode="before")
    @classmethod
    def _validate_carryover_order_type(cls, v: object) -> object:
        if isinstance(v, bool):
            _raise("definition.assignment.reject_bool")
        if not isinstance(v, int):
            _raise("definition.assignment.carryover_order_type")
        return v

    @model_validator(mode="after")
    def _validate_carryover_order_range(self) -> SwitchbackWindow:
        # This bound is load-bearing, not cosmetic: it is what makes the
        # retained window (``observation_steps - carryover_order``)
        # nonempty for every constructible window.
        if not (0 <= self.carryover_order < self.observation_steps):
            _raise(
                "definition.assignment.carryover_order_range",
                carryover_order=self.carryover_order,
                observation_steps=self.observation_steps,
            )
        return self


class SwitchbackAssignment(_AssignmentModel):
    kind: Literal["switchback"] = "switchback"
    periods_per_cycle: Literal[2] = 2
    sequence: AssignmentSequence
    window: SwitchbackWindow


Assignment = Annotated[ParallelAssignment | SwitchbackAssignment, Field(discriminator="kind")]


__all__ = [
    "Assignment",
    "IndependentBernoulliOrder",
    "ParallelAssignment",
    "SharedScheduleOrder",
    "SwitchbackAssignment",
    "SwitchbackWindow",
]
