"""Warehouse CUPED preparation and compact estimation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from increment._cuped_evidence import categorical_identity, method_key
from increment.estimation.cuped.cells import fit_categorical_cells
from increment.estimation.cuped.types import _freeze_payload
from increment.query._cuped_cells import (
    _rows,
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


def _require_compact_request(source, metric, method: Method) -> None:
    """Declaration-level routes to general QR, checked before any warehouse query."""
    if any(item.kind != "categorical" for item in method.covariates or ()):
        general_qr_required(metric, "continuous_covariates")
    if known_outcome_domain(metric):
        general_qr_required(metric, "outcome_domain")
    if getattr(source, "cuped_relation", None) is None:
        general_qr_required(metric, "source_relation_unavailable")


def categorical_cells_source(source, metric, request: _ResolvedCupedRequest) -> CategoricalCells:
    """Capture categorical evidence on the source's pinned warehouse relation."""
    _require_compact_request(source, metric, request.method)
    with source.cuped_relation(metric, request) as (relation, execute, _scratch):
        return categorical_cells_from_relation(relation, execute, metric=metric, request=request)


def categorical_request_cells(
    source, metric, method: Method, *, population: Literal["assigned", "triggered"]
) -> CategoricalCells:
    """Portable joint cells for one request over every arm its warehouse relation carries.

    The source's own routes to general QR precede the arm-label query, which
    transfers only distinct ``group_id`` values; other arms follow the control
    in label order.
    """
    control = str(source.context.design.control_group)
    request = _ResolvedCupedRequest(
        identity=categorical_identity(metric.name, method_key(method), population),
        control=control,
        arm_order=(control,),
        method=method,
        population=population,
    )
    _require_compact_request(source, metric, method)
    with source.cuped_relation(metric, request) as (relation, execute, _scratch):
        labels = {row["group_id"] for row in _rows(execute(relation.select("group_id").distinct()))}
        arms = sorted(str(label) for label in labels if label is not None and str(label) != control)
        request = replace(request, arm_order=(control, *arms))
        return categorical_cells_from_relation(relation, execute, metric=metric, request=request)


def resolved_multicovariate_request(
    source,
    metric,
    method: Method,
    *,
    control: str,
    arm_order: tuple[str, ...],
    population: Literal["assigned", "triggered"] = "assigned",
) -> _ResolvedCupedRequest:
    """Bind one warehouse fit to its immutable source and normalized request."""
    from increment._source_identity import source_identity

    identity = (
        "warehouse-cuped-fit-v1",
        metric.name,
        population,
        source_identity(source),
        method_key(method),
        control,
        arm_order,
    )
    return _ResolvedCupedRequest(
        identity=identity,
        control=control,
        arm_order=arm_order,
        method=method,
        population=population,
    )


def fit_multicovariate_source(
    source, metric, request: _ResolvedCupedRequest
) -> MulticovariateCupedFit:
    """Fit categorical warehouse evidence without transferring individual observations."""
    return fit_categorical_cells(
        categorical_cells_source(source, metric, request), control=request.control
    )
