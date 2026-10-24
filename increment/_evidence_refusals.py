"""Shared refusals for evidence requests a source cannot serve."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, NoReturn

from increment.errors import CapabilityError, RefusalSpec, refuse

if TYPE_CHECKING:
    from increment._source_types import Grain


def _spec(code: str, hazard: str) -> RefusalSpec:
    return RefusalSpec(
        code,
        CapabilityError,
        template=(
            "{constructor} cannot supply requested "
            + hazard
            + " {requested!r}; offered: {offered!r}. {route}"
        ),
        keys=frozenset({"constructor", "requested", "offered", "route"}),
    )


GRAIN_UNAVAILABLE = _spec("evidence.grain_unavailable", "grain")
BREAKOUT_UNAVAILABLE = _spec("evidence.breakout_unavailable", "breakout")
OPERATION_UNAVAILABLE = _spec("evidence.operation_unavailable", "operation")
EVIDENCE_REFUSALS = {
    spec.code: spec for spec in (GRAIN_UNAVAILABLE, BREAKOUT_UNAVAILABLE, OPERATION_UNAVAILABLE)
}


def refuse_grain(
    src: object, requested: object, offered: Iterable[object], *, route: str
) -> NoReturn:
    refuse(
        GRAIN_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=requested,
        offered=tuple(offered),
        route=route,
    )


def refuse_breakout(
    src: object, requested: Sequence[str], offered: Sequence[str], *, route: str
) -> NoReturn:
    refuse(
        BREAKOUT_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=tuple(requested),
        offered=tuple(offered),
        route=route,
    )


def refuse_operation(
    src: object, requested: object, offered: Sequence[str], *, route: str
) -> NoReturn:
    refuse(
        OPERATION_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=requested,
        offered=tuple(offered),
        route=route,
    )


def require_evidence(
    src: object,
    *,
    grain: Grain | None = None,
    by: Sequence[str] = (),
    operation: str | None = None,
) -> None:
    """Refuse unsupported grain, breakout or operation requests pre-read."""
    from increment._source_types import declared_evidence

    evidence = declared_evidence(src)
    if operation == "sql" and not evidence.sql_grains:
        refuse_operation(
            src,
            operation,
            sorted(evidence.operations),
            route="use a source that supports SQL rendering",
        )
    if grain is not None:
        requested = grain
        offered = evidence.sql_grains if operation == "sql" else evidence.grains
        if grain not in offered:
            refuse_grain(
                src,
                requested,
                sorted(offered),
                route="use a source that retains the requested evidence",
            )
    if by and not set(by).issubset(evidence.breakouts):
        refuse_breakout(
            src,
            by,
            evidence.breakouts,
            route="use a source with the requested breakout dimensions",
        )
    if operation is not None and operation != "sql" and operation not in evidence.operations:
        refuse_operation(
            src,
            operation,
            sorted(evidence.operations),
            route="use a source that supports this operation",
        )
