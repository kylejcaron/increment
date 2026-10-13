"""Admitted per-cell moments: the only input the overlap estimator consumes."""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from increment._overlap.contrasts import Cell
from increment._overlap.design import ExperimentColumns, OverlapDesign
from increment._overlap.refusals import _raise
from increment.errors import CodedModel


class CellMoments(CodedModel, BaseModel):
    """Count, mean, and sample variance of one metric in one joint assignment cell.

    ``cell`` is ``(first_treated, second_treated)``. ``mean`` is null for an
    empty cell and ``variance`` is null below two units; a null is never
    replaced by zero.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    cell: Cell
    first_arm: str
    second_arm: str
    n: int = Field(ge=0)
    mean: float | None
    variance: float | None


class EligibilityAudit(CodedModel, BaseModel):
    """Unit counts by eligibility. Only ``jointly_eligible`` units enter the target."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    units: int
    jointly_eligible: int
    first_only_eligible: int
    second_only_eligible: int
    neither_eligible: int


@dataclass(frozen=True, slots=True)
class Admission:
    """Validated moments of the jointly eligible units, produced by one ingress."""

    design: OverlapDesign
    metrics: tuple[str, ...]
    moments: tuple[CellMoments, ...]
    audit: EligibilityAudit
    outcome_window: dt.timedelta


def cell_arms(design: OverlapDesign, cell: Cell) -> tuple[str, str]:
    first, second = design.first, design.second
    return (
        first.treatment if cell[0] else first.control,
        second.treatment if cell[1] else second.control,
    )


def check_bindings(
    design: OverlapDesign, columns: Mapping[str, ExperimentColumns], metrics: Mapping[str, str]
) -> tuple[ExperimentColumns, ExperimentColumns]:
    """Shared request checks for both ingresses, run before any data is read."""
    declared = (design.first.name, design.second.name)
    if set(columns) != set(declared) or len(columns) != 2:
        _raise("overlap.ingress.columns_unbound", bound=tuple(columns), declared=declared)
    if not metrics:
        _raise("overlap.ingress.no_metrics")
    return columns[design.first.name], columns[design.second.name]
