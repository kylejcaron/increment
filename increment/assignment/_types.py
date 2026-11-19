"""JSON records returned by an allocation evaluator."""

from typing import Literal, TypedDict


class Membership(TypedDict):
    """A unit's holdout group and that group's probability in basis points."""

    protocol: Literal[1]
    revision: str
    holdout: str
    group: Literal["held_out", "eligible"]
    group_bps: int


class Decision(TypedDict):
    """An assignment result with the probabilities of the gates it reached."""

    protocol: Literal[1]
    revision: str
    target: str
    membership: Membership | None
    experiment: str | None
    arm: str | None
    status: Literal["held_out", "unallocated", "not_enrolled", "assigned"]
    layer_bps: int | None
    enrollment_bps: int | None
    arm_bps: int | None
