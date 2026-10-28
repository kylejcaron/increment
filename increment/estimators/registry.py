from __future__ import annotations

import importlib
from collections.abc import Iterator, Mapping
from typing import Any

from increment._evidence_types import (
    CountsRequest,
    MethodFitRequest,
    MomentsRequest,
    UnitRowsRequest,
)
from increment.declarations import (
    ADJUSTMENTS,
    BUILTIN_ESTIMATORS,
    VARIANCE_REDUCTION,
    CountsSpec,
    EstimatorRef,
    Method,
    MethodFitSpec,
    MomentsSpec,
    UnitRowsSpec,
    estimator_id_for_selector,
    estimator_refs,
    register,
    register_implementation,
)
from increment.errors import CodedError, InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.estimator.need.unsupported": "Estimator {estimator_id!r} declares unsupported need spec {need_type!r}.",
    },
)
_refuse = raiser(_REFUSALS)


class _EstimatorMapping(Mapping[str, EstimatorRef]):
    def __getitem__(self, key: str) -> EstimatorRef:
        return estimator_refs()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(estimator_refs())

    def __len__(self) -> int:
        return len(estimator_refs())


ESTIMATORS: Mapping[str, EstimatorRef] = _EstimatorMapping()

_BUILTIN_ESTIMATOR_IDS = frozenset(ref.estimator_id for ref in BUILTIN_ESTIMATORS)


def load(estimator_id: str) -> Any:
    ref = estimator_refs().get(estimator_id)
    if ref is None:
        raise KeyError(estimator_id)
    implementation = _implementation(estimator_id)
    if implementation is not None:
        return implementation
    if not ref.module or not ref.attribute:
        raise LookupError(f"no implementation registered for {estimator_id!r}")
    module = importlib.import_module(ref.module)
    implementation = getattr(module, ref.attribute)
    register_implementation(estimator_id, implementation)
    if ref.selector is not None and estimator_id.startswith("observational."):
        ADJUSTMENTS._entries[ref.selector] = implementation
    return implementation


def _implementation(estimator_id: str) -> Any | None:
    from increment.declarations import _IMPLEMENTATIONS

    return _IMPLEMENTATIONS.get(estimator_id)


def estimator_id(method: Method) -> str:
    return (
        observational_estimator_id(method)
        if method.is_observational
        else randomized_estimator_id(method)
    )


def observational_estimator_id(method: Method) -> str:
    estimator_id = estimator_id_for_selector(method.name)
    if estimator_id is None:
        ADJUSTMENTS.get(method.name)
        raise AssertionError("unreachable")
    return estimator_id


def randomized_estimator_id(method: Method) -> str:
    return VARIANCE_REDUCTION.get(method.variance_reduction)


def uses_builtin_adjustment_preflight(method: Method) -> bool:
    estimator_id = estimator_id_for_selector(method.name)
    if estimator_id is None:
        return False
    ref = estimator_refs()[estimator_id]
    return ref.module in {
        "increment.estimation._adjust.iptw",
        "increment.estimation._adjust.dml",
        "increment.estimation._adjust.aipw",
    }


def routes(ref: EstimatorRef, context: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Build concrete acquisition requests from each declared alternative route."""
    if (
        any(
            isinstance(spec, UnitRowsSpec) and spec.covariates == "design"
            for route in ref.needs
            for spec in route
        )
        and "design_covariates" not in context
    ):
        raise CodedError(
            "observational routes need design covariates in their context",
            code="estimation.engine.method_covariates",
            context={"method": ref.selector or ref.estimator_id},
        )
    built: list[tuple[Any, ...]] = []
    for route in ref.needs:
        requests: list[Any] = []
        for spec in route:
            if isinstance(spec, MomentsSpec):
                requests.append(
                    MomentsRequest(
                        metric=context["metric"],
                        grain=spec.grain,
                        method=context.get("method") if spec.method_specific else None,
                        include_covariate=spec.include_covariate,
                        population=context.get("population", "assigned"),
                    )
                )
            elif isinstance(spec, UnitRowsSpec):
                if spec.covariates == "design":
                    covariates = context.get("design_covariates", ())
                elif spec.covariates == "method":
                    method = context.get("method")
                    covariates = context.get("method_covariates")
                    if covariates is None and method is not None:
                        covariates = tuple(value.name for value in (method.covariates or ()))
                else:
                    covariates = ()
                requests.append(
                    UnitRowsRequest(
                        metric=context["metric"],
                        covariates=tuple(covariates or ()),
                        outcome_stage=spec.outcome_stage,
                        population=context.get("population", "assigned"),
                    )
                )
            elif isinstance(spec, MethodFitSpec):
                requests.append(
                    MethodFitRequest(
                        metric=context["metric"],
                        method=context["method"],
                        control=context["control"],
                        arm_order=tuple(context.get("arm_order", ())),
                        population=context.get("population", "assigned"),
                    )
                )
            elif isinstance(spec, CountsSpec):
                requests.append(
                    CountsRequest(kind=spec.kind, population=context.get("population", "assigned"))
                )
            else:
                _refuse(
                    "estimation.estimator.need.unsupported",
                    estimator_id=ref.estimator_id,
                    need_type=type(spec).__name__,
                )
        built.append(tuple(requests))
    return tuple(built)


def require_routes(source: object, ref: EstimatorRef, context: Mapping[str, Any]) -> None:
    """Require one complete declared route to be supported by the source."""
    from increment._evidence_refusals import require_evidence
    from increment.errors import CapabilityError

    failures: list[CapabilityError] = []
    for route in routes(ref, context):
        try:
            for request in route:
                require_evidence(source, request=request)
        except CapabilityError as error:
            failures.append(error)
        else:
            return
    if failures:
        raise failures[-1]


def validate_observational_routes(
    source: object,
    metrics: tuple[Any, ...],
    methods: tuple[Any, ...],
    design: Any,
    population: str = "assigned",
) -> None:
    """Validate each observational estimator's declared alternatives before acquisition."""
    design_covariates = tuple(getattr(getattr(design, "adjustment", None), "covariates", ()))
    for method in methods:
        if not method.is_observational:
            continue
        ref = ESTIMATORS[observational_estimator_id(method)]
        for metric in metrics:
            require_routes(
                source,
                ref,
                {
                    "metric": metric,
                    "method": method,
                    "control": design.control_group,
                    "arm_order": (),
                    "population": population,
                    "design_covariates": design_covariates,
                    "method_covariates": tuple(
                        value.name for value in (getattr(method, "covariates", None) or ())
                    ),
                },
            )


for _ref in BUILTIN_ESTIMATORS:
    register(_ref)
