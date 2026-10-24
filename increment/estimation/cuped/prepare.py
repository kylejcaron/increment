from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from increment._evidence_types import (
    BreakoutCell,
    MethodFitAnswer,
    MethodFitRequest,
    PreparedMulticovariateFit,
)


def prepare_multicovariate_fit(
    method: Any,
    answers: Iterable[tuple[MethodFitRequest, MethodFitAnswer]],
    *,
    cell: BreakoutCell | None = None,
) -> PreparedMulticovariateFit:
    """Select retained evidence for one method/cell, or mark it for local fitting."""
    answer = next(
        (
            answer
            for request, answer in answers
            if request.method.name == method.name and request.cell == cell
        ),
        None,
    )
    if answer is None or answer.origin == "unavailable":
        return PreparedMulticovariateFit(origin="local")
    return PreparedMulticovariateFit(
        origin=answer.origin,
        envelope=answer.envelope,
        cells=answer.cells,
        failure=answer.failure,
        fit_identity=answer.fit_identity,
    )
