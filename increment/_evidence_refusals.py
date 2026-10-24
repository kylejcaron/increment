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
UNIT_ROWS_UNAVAILABLE = _spec("evidence.unit_rows_unavailable", "unit rows")
RAW_OUTCOMES_UNAVAILABLE = _spec("evidence.raw_outcomes_unavailable", "raw outcomes")
CLUSTER_UNAVAILABLE = _spec("evidence.cluster_unavailable", "cluster evidence")
COVARIATE_UNAVAILABLE = _spec("evidence.covariate_unavailable", "covariates")
OPERATION_UNAVAILABLE = _spec("evidence.operation_unavailable", "operation")
EVIDENCE_REFUSALS = {
    spec.code: spec
    for spec in (
        GRAIN_UNAVAILABLE,
        BREAKOUT_UNAVAILABLE,
        UNIT_ROWS_UNAVAILABLE,
        RAW_OUTCOMES_UNAVAILABLE,
        CLUSTER_UNAVAILABLE,
        COVARIATE_UNAVAILABLE,
        OPERATION_UNAVAILABLE,
    )
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


def refuse_unit_rows(src: object, *, offered: object, route: str) -> NoReturn:
    refuse(
        UNIT_ROWS_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=True,
        offered=offered,
        route=route,
    )


def refuse_cluster(src: object, *, offered: Iterable[object], route: str) -> NoReturn:
    refuse(
        CLUSTER_UNAVAILABLE,
        constructor=type(src).__name__,
        requested="cluster",
        offered=tuple(offered),
        route=route,
    )


def refuse_raw_outcomes(src: object, *, offered: object, route: str) -> NoReturn:
    refuse(
        RAW_OUTCOMES_UNAVAILABLE,
        constructor=type(src).__name__,
        requested=True,
        offered=offered,
        route=route,
    )


def require_evidence(
    src: object,
    *,
    grain: Grain | None = None,
    by: Sequence[str] = (),
    unit_rows: bool = False,
    raw_outcomes: bool = False,
    cluster: bool = False,
    covariates: Sequence[str] = (),
    operation: str | None = None,
) -> None:
    """Refuse unsupported evidence requests before reads."""
    from increment._source_types import declared_evidence, evidence_is_declared

    evidence = declared_evidence(src)
    declared = evidence_is_declared(src)
    if declared and covariates and not evidence.unit_rows:
        refuse(
            COVARIATE_UNAVAILABLE,
            constructor=type(src).__name__,
            requested=tuple(covariates),
            offered=tuple(sorted(evidence.available_covariates or ())),
            route="use a source that retains the requested covariates",
        )
    if declared and raw_outcomes and not evidence.raw_outcomes:
        refuse_raw_outcomes(
            src,
            offered=evidence.raw_outcomes,
            route="use a source that retains pre-transform unit outcomes",
        )
    if declared and unit_rows and not evidence.unit_rows:
        refuse_unit_rows(
            src, offered=evidence.unit_rows, route="use a source that retains unit-level rows"
        )
    if declared and cluster and "cluster" not in evidence.count_grains:
        refuse_cluster(
            src,
            offered=sorted(evidence.count_grains),
            route="use a source that retains cluster identities or cluster counts",
        )
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
