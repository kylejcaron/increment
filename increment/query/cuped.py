"""Warehouse CUPED preparation and compact estimation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from increment._cuped_evidence import categorical_identity, method_key
from increment.estimation.cuped.cells import fit_categorical_cells
from increment.estimation.cuped.types import _freeze_payload
from increment.query._cuped_cells import (
    _rows,
    categorical_cells_from_relation,
    general_qr_required,
    known_outcome_domain,
    known_qr_outcome_domain,
)

if TYPE_CHECKING:
    from increment.estimation.cuped.types import (
        CategoricalCells,
        MulticovariateCupedFit,
    )
    from increment.estimation.engine import Method


@dataclass(frozen=True, slots=True)
class _ResolvedCupedRequest:
    identity: tuple[object, ...]
    control: str
    arm_order: tuple[str, ...]
    method: Method
    population: Literal["assigned", "triggered"] = "assigned"
    breakout_dimension: str | None = None
    breakout_source: str | None = None
    breakout_value: str | None = None
    breakout_is_null: bool = False

    def __post_init__(self):
        object.__setattr__(self, "identity", _freeze_payload(self.identity))
        object.__setattr__(self, "arm_order", tuple(self.arm_order))


@dataclass(frozen=True, slots=True)
class _WarehouseCupedFitEvidence:
    fit: MulticovariateCupedFit | None
    arm_order: tuple[str, ...]
    withheld: tuple[tuple[str, object], ...] = ()


def _require_compact_request(source, metric, method: Method) -> None:
    """Declaration-level routes to general QR, checked before any warehouse query."""
    if known_outcome_domain(metric):
        general_qr_required(metric, "outcome_domain")
    if any(item.kind != "categorical" for item in method.covariates or ()):
        general_qr_required(metric, "continuous_covariates")
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
    breakout_dimension: str | None = None,
    breakout_source: str | None = None,
    breakout_value: str | None = None,
    breakout_is_null: bool = False,
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
        breakout_dimension,
        breakout_source,
        breakout_value,
        breakout_is_null,
    )
    return _ResolvedCupedRequest(
        identity=identity,
        control=control,
        arm_order=arm_order,
        method=method,
        population=population,
        breakout_dimension=breakout_dimension,
        breakout_source=breakout_source,
        breakout_value=breakout_value,
        breakout_is_null=breakout_is_null,
    )


def _known_cells_outcome_domain(source, metric, relation, outcomes) -> bool:
    """Check source-certified domains before considering categorical cells."""
    domain_reader = getattr(source, "cuped_outcome_domains", None)
    if callable(domain_reader):
        return any(domain != "continuous" for domain in domain_reader(metric))
    leaf = source
    seen: set[int] = set()
    manifest = None
    while id(leaf) not in seen:
        seen.add(id(leaf))
        manifest = getattr(leaf, "_manifest", None)
        if manifest is not None:
            break
        leaf = getattr(leaf, "_source", None)
        if leaf is None:
            break
    if manifest is not None:
        binding = next(
            (item for item in manifest.metric_measures if item.metric_name == metric.name),
            None,
        )
        if binding is None:
            return True
        measure_keys = (
            (binding.numerator_measure_key, binding.denominator_measure_key)
            if binding.kind == "ratio"
            else (binding.measure_key,)
        )
        domains = {item.measure_key: item.outcome_domain for item in manifest.measures}
        return any(domains.get(key, "unknown") != "continuous" for key in measure_keys)
    from increment.query._cuped_cells import _known_outcome_domain

    return _known_outcome_domain(metric, relation, outcomes)


def fit_multicovariate_source(
    source, metric, request: _ResolvedCupedRequest, *, return_components: bool = False
) -> MulticovariateCupedFit | _WarehouseCupedFitEvidence | tuple[Any, Any, Any]:
    """Fit from bounded categorical cells or stable warehouse QR evidence."""
    from increment.errors import CodedError
    from increment.query._cuped_cells import REFUSALS as CELL_REFUSALS
    from increment.query._cuped_qr import fit_warehouse_qr

    if known_qr_outcome_domain(metric):
        general_qr_required(metric, "outcome_domain")
    relation_factory = getattr(source, "cuped_relation", None)
    if not callable(relation_factory):
        general_qr_required(
            metric,
            "outcome_domain" if known_outcome_domain(metric) else "source_relation_unavailable",
        )
    with relation_factory(metric, request) as (relation, execute, scratch_factory):
        categorical_only = all(
            item.kind == "categorical" for item in request.method.covariates or ()
        )
        outcomes = ("y", "y_den") if metric.type == "ratio" else ("y",)
        from increment.estimation.variance import require_positive_denominator_mean
        from increment.query._cuped_qr import _stable_arm_means

        withheld = {}
        fitted_request = request
        if metric.type == "ratio":
            denominator_means = _stable_arm_means(
                relation, relation["y_den"].cast("float64"), execute=execute
            )
            for arm, mean in denominator_means.items():
                try:
                    require_positive_denominator_mean(metric=metric.name, group_id=arm, d_bar=mean)
                except CodedError as exc:
                    withheld[arm] = exc
            admitted = tuple(arm for arm in request.arm_order if arm not in withheld)
            if request.control in withheld or len(admitted) < 2:
                return _WarehouseCupedFitEvidence(None, request.arm_order, tuple(withheld.items()))
            relation = relation.filter(relation["group_id"].isin(admitted))
            fitted_request = replace(
                request,
                arm_order=admitted,
                identity=request.identity + ("admitted_arms", admitted),
            )
        if categorical_only and not _known_cells_outcome_domain(source, metric, relation, outcomes):
            try:
                fit = fit_categorical_cells(
                    categorical_cells_from_relation(
                        relation, execute, metric=metric, request=fitted_request
                    ),
                    control=fitted_request.control,
                )
                return (
                    _WarehouseCupedFitEvidence(fit, request.arm_order, tuple(withheld.items()))
                    if withheld
                    else fit
                )
            except CodedError as exc:
                if exc.code not in CELL_REFUSALS:
                    raise
        with scratch_factory() as scratch:
            fit = fit_warehouse_qr(
                relation,
                metric=metric,
                request=fitted_request,
                execute=execute,
                scratch=scratch,
                return_components=return_components,
            )
        return (
            _WarehouseCupedFitEvidence(fit, request.arm_order, tuple(withheld.items()))
            if withheld
            else fit
        )


def _portable_categorical_domain(source, metric, method, population) -> None:
    """Refuse non-cell categorical exports before any outcome aggregate is transferred."""
    from increment.query._cuped_cells import general_qr_required, known_outcome_domain

    if known_outcome_domain(metric):
        general_qr_required(metric, "outcome_domain")
    control = str(source.context.design.control_group)
    request = _ResolvedCupedRequest(
        identity=categorical_identity(metric.name, method_key(method), population),
        control=control,
        arm_order=(control,),
        method=method,
        population=population,
    )
    with source.cuped_relation(metric, request) as (relation, _execute, _scratch):
        outcomes = ("y", "y_den") if metric.type == "ratio" else ("y",)
        if _known_cells_outcome_domain(source, metric, relation, outcomes):
            general_qr_required(metric, "outcome_domain")


def portable_qr_requests(source, requests, *, population: Literal["assigned", "triggered"]):
    """Move categorical requests that exceed the cells contract to compact QR evidence."""
    from increment.errors import CodedError
    from increment.query._cuped_cells import known_qr_outcome_domain

    resolved = []
    for metric, scalar, categorical, fitted in requests:
        cell_methods = []
        fitted_methods = list(fitted)
        for method in categorical:
            try:
                _portable_categorical_domain(source, metric, method, population)
                _require_compact_request(source, metric, method)
            except CodedError as exc:
                if exc.code != "query.cuped.general_qr_required" or known_qr_outcome_domain(metric):
                    raise
                fitted_methods.append(method)
            else:
                cell_methods.append(method)
        resolved.append((metric, scalar, tuple(cell_methods), tuple(fitted_methods)))
    return tuple(resolved)


def fit_multicovariate_request(
    source,
    metric,
    method: Method,
    *,
    control: str,
    arm_order: tuple[str, ...],
    population: Literal["assigned", "triggered"] = "assigned",
    breakout_dimension: str | None = None,
    breakout_source: str | None = None,
    breakout_value: str | None = None,
    breakout_is_null: bool = False,
    return_components: bool = False,
):
    """Build request identity and fit through the source-owned query interface."""
    from increment.errors import CodedError

    request = resolved_multicovariate_request(
        source,
        metric,
        method,
        control=control,
        arm_order=arm_order,
        population=population,
        breakout_dimension=breakout_dimension,
        breakout_source=breakout_source,
        breakout_value=breakout_value,
        breakout_is_null=breakout_is_null,
    )
    try:
        fit = fit_multicovariate_source(
            source, metric, request, return_components=return_components
        )
    except CodedError as exc:
        fit = exc
    return request.identity, fit
