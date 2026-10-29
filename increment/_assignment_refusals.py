"""Coded refusals for assignment evidence and replay contracts."""

from __future__ import annotations

from typing import NoReturn

from increment.errors import CapabilityError, RefusalSpec, refuse

ASSIGNMENT_UNAVAILABLE = RefusalSpec(
    "evidence.assignment.unavailable",
    CapabilityError,
    template="Source {constructor} cannot provide assignment evidence; offered: {offered!r}. {route}",
)
ASSIGNMENT_SELECTION_UNAVAILABLE = RefusalSpec(
    "evidence.assignment.selection_unavailable",
    CapabilityError,
    template=(
        "Source {constructor} cannot select assignment evidence for {requested!r}; "
        "offered: {offered!r}"
    ),
)
ASSIGNMENT_INVALID = RefusalSpec(
    "evidence.assignment.invalid",
    CapabilityError,
    template="Invalid assignment evidence: {reason}; details: {details!r}",
)
ASSIGNMENT_REPLAY_MISMATCH = RefusalSpec(
    "evidence.assignment.replay_mismatch",
    CapabilityError,
    template="Assignment replay does not match the declared ledger: {details!r}",
)


def refuse_assignment(src: object, *, offered: object, route: str) -> NoReturn:
    """Refuse assignment evidence before source access."""
    refuse(
        ASSIGNMENT_UNAVAILABLE,
        constructor=type(src).__name__,
        offered=offered,
        route=route,
    )


def refuse_assignment_selection(src: object, *, requested: object, offered: object) -> NoReturn:
    """Refuse a scope aggregate evidence cannot satisfy."""
    refuse(
        ASSIGNMENT_SELECTION_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=requested,
        offered=offered,
    )


def refuse_assignment_invalid(reason: str, *, details: object = None) -> NoReturn:
    refuse(ASSIGNMENT_INVALID, reason=reason, details=details)


def refuse_assignment_replay_mismatch(*, details: object = None) -> NoReturn:
    refuse(ASSIGNMENT_REPLAY_MISMATCH, details=details)
