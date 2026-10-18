"""Warehouse CUPED preparation and compact estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from increment.estimation.cuped.cells import fit_categorical_cells
from increment.estimation.cuped.types import _freeze_payload
from increment.query._cuped_cells import (
    categorical_cells_from_relation,
    general_qr_required,
    known_outcome_domain,
)

if TYPE_CHECKING:
    from increment.estimation.cuped.types import CategoricalCells, MulticovariateCupedFit
    from increment.estimation.engine import Method


@dataclass(frozen=True, slots=True)
class _ResolvedCupedRequest:
    identity: tuple[object, ...]
    control: str
    arm_order: tuple[str, ...]
    method: Method
    population: Literal["assigned", "triggered"] = "assigned"

    def __post_init__(self):
        object.__setattr__(self, "identity", _freeze_payload(self.identity))
        object.__setattr__(self, "arm_order", tuple(self.arm_order))


def categorical_cells_source(source, metric, request: _ResolvedCupedRequest) -> CategoricalCells:
    """Capture categorical evidence on the source's pinned warehouse relation."""
    if any(item.kind != "categorical" for item in request.method.covariates or ()):
        general_qr_required(metric, "continuous_covariates")
    if known_outcome_domain(metric):
        general_qr_required(metric, "outcome_domain")
    relation_context = getattr(source, "cuped_relation", None)
    if relation_context is None:
        general_qr_required(metric, "source_relation_unavailable")
    with relation_context(metric, request) as (relation, execute):
        return categorical_cells_from_relation(relation, execute, metric=metric, request=request)


def fit_multicovariate_source(
    source, metric, request: _ResolvedCupedRequest
) -> MulticovariateCupedFit:
    """Fit categorical warehouse evidence without transferring individual observations."""
    return fit_categorical_cells(
        categorical_cells_source(source, metric, request), control=request.control
    )
