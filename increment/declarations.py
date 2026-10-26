"""Lightweight shared declaration types and registries.

This module is intentionally independent of the estimation implementation so callers can build
method declarations without importing estimator machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from increment._canonical import canonical_digest_bytes
from increment._finite_sample_refusals import refuse_finite_sample_cuped
from increment._immutable import _FrozenMapping
from increment._literals import ConversionInference
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.semantics.models import CupedCovariate, CupedCovariateBinding


@dataclass(frozen=True, slots=True)
class ArmHypothesisKey:
    metric: str
    group_id: str
    estimand: str


@dataclass(frozen=True, slots=True)
class SegmentHypothesisKey:
    metric: str
    group_id: str
    estimand: str
    dimension: str
    dimension_value: str

    @property
    def segment(self):
        return ((self.dimension, self.dimension_value),)


@dataclass(frozen=True, slots=True)
class ContrastHypothesisKey:
    metric: str
    control_group: str
    treatment_group: str


HypothesisKey = ArmHypothesisKey | SegmentHypothesisKey | ContrastHypothesisKey
Population = Literal["assigned", "triggered"]


_CELL_KEY_INVALID_REFUSAL = RefusalSpec(
    "readout.cell.key_invalid",
    InvalidRequestError,
    template="Invalid {kind} readout cell fields: {fields}",
)
_VARIANCE_REGISTRY_REFUSAL = RefusalSpec(
    "estimation.variance.registry.no_registered_available",
    UnsupportedRequestError,
    template="No {registry_name} registered for '{key}'. Available keys: {available}",
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.engine.method.name_without_variance": "Method(name={self!r}) without its matching variance_reduction would label an unadjusted estimate as CUPED-adjusted.",
        "estimation.engine.method_covariates": "method {method!r} has an invalid CUPED covariate declaration for its variance-reduction procedure",
        "estimation.engine.method.name_iptw_does": "Method(name='iptw') does not accept outcome_learner/folds -- IPTW fits one propensity model with no cross-fitting; set propensity_learner (mapped to iptw_estimate's learner=) instead, or use Method(name='dml') / Method(name='aipw') for cross-fit outcome-model pluggability.",
        "estimation.variance.registry.no_registered_available": _VARIANCE_REGISTRY_REFUSAL,
        "readout.cell.key_invalid": _CELL_KEY_INVALID_REFUSAL,
    },
)
_refuse = raiser(_REFUSALS)


def freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenMapping({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [thaw(item) for item in value]
    return value


class _ReadoutModel(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    @model_validator(mode="after")
    def _freeze_mappings(self):
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, Mapping):
                canonical_digest_bytes(thaw(value))
                object.__setattr__(self, name, freeze(value))
        return self

    @field_serializer("*", when_used="json")
    def _serialize_values(self, value):
        return thaw(value)


class CellKey(_ReadoutModel):
    kind: Literal["arm", "segment", "contrast"]
    metric: str
    method: str | None
    method_role: Literal["decision", "sensitivity"] | None
    estimand: str | None
    aggregation: Literal["sum", "any"] | None = None
    analysis_population: Population
    value_scale: Literal["relative", "absolute"] | None = None
    inference: Literal["fixed", "always_valid", "asymptotic_mean"] | None = None
    alternative: Literal["two-sided", "greater", "less"] | None = None
    group_id: str | None = None
    control_group: str | None = None
    treatment_group: str | None = None
    dimension: str | None = None
    dimension_value: str | None = None
    source: str | None = None
    ds: date | datetime | str | int | float | None = None
    ds_basis: Literal["calendar", "cohort"] | None = None

    @classmethod
    def from_row(cls, row):
        contrast = hasattr(row, "control_group")
        segment = getattr(row, "dimension", None) is not None
        values: dict[str, Any] = {name: getattr(row, name, None) for name in cls.model_fields}
        values.update(
            kind="contrast" if contrast else "segment" if segment else "arm",
            analysis_population=getattr(row, "analysis_population", "assigned"),
        )
        return cls(**values)

    def __hash__(self):
        return hash(
            (
                self.kind,
                self.metric,
                self.method,
                self.method_role,
                self.estimand,
                self.aggregation,
                self.analysis_population,
                self.value_scale,
                self.inference,
                self.alternative,
                self.group_id,
                self.control_group,
                self.treatment_group,
                self.dimension,
                self.dimension_value,
                self.source,
                self.ds,
                self.ds_basis,
            )
        )

    @field_validator("ds", mode="before")
    @classmethod
    def _deserialize_day_axis(cls, value):
        if isinstance(value, Mapping) and set(value) == {"label"}:
            return value["label"]
        if isinstance(value, Mapping) and set(value) == {"datetime"}:
            return datetime.fromisoformat(value["datetime"])
        if isinstance(value, str):
            try:
                return date.fromisoformat(value)
            except ValueError:
                pass
        return value

    @model_validator(mode="after")
    def _identity_axes(self):
        valid = (
            self.kind == "arm"
            and self.group_id is not None
            and all(
                value is None
                for value in (
                    self.control_group,
                    self.treatment_group,
                    self.dimension,
                    self.dimension_value,
                    self.aggregation,
                )
            )
            or self.kind == "segment"
            and self.group_id is not None
            and self.dimension is not None
            and self.dimension_value is not None
            and all(
                value is None
                for value in (self.control_group, self.treatment_group, self.aggregation)
            )
            or self.kind == "contrast"
            and self.control_group is not None
            and self.treatment_group is not None
            and self.aggregation is not None
            and all(
                value is None
                for value in (
                    self.group_id,
                    self.dimension,
                    self.dimension_value,
                    self.source,
                    self.ds,
                    self.ds_basis,
                    self.value_scale,
                )
            )
        )
        if self.ds_basis is not None and self.ds is None:
            valid = False
        if self.method is None:
            valid = valid and self.method_role is None and self.estimand is None
        else:
            valid = valid and self.method_role is not None and self.estimand is not None
        if self.kind == "contrast" and self.analysis_population != "assigned":
            valid = False
        if (self.dimension is None) != (self.dimension_value is None):
            valid = False
        if not valid:
            _refuse(
                "readout.cell.key_invalid",
                kind=self.kind,
                fields=tuple(
                    name
                    for name in (
                        "group_id",
                        "control_group",
                        "treatment_group",
                        "dimension",
                        "dimension_value",
                        "aggregation",
                        "method",
                        "method_role",
                        "estimand",
                    )
                    if getattr(self, name) is None
                ),
            )
        return self

    @field_serializer("ds", when_used="json")
    def _serialize_ds(self, value):
        return serialize_day_axis(value)

    def hypothesis(self) -> HypothesisKey:
        if self.kind == "arm":
            assert self.group_id is not None and self.estimand is not None
            return ArmHypothesisKey(self.metric, self.group_id, self.estimand)
        if self.kind == "segment":
            assert (
                self.group_id is not None
                and self.estimand is not None
                and self.dimension is not None
                and self.dimension_value is not None
            )
            return SegmentHypothesisKey(
                self.metric, self.group_id, self.estimand, self.dimension, self.dimension_value
            )
        assert self.control_group is not None and self.treatment_group is not None
        return ContrastHypothesisKey(self.metric, self.control_group, self.treatment_group)


def serialize_day_axis(value):
    if isinstance(value, datetime):
        return {"datetime": value.isoformat()}
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return {"label": value}
    return value


class IntegrityStatus(StrEnum):
    FAILED = "failed"
    NOT_REJECTED = "not_rejected"
    NOT_CHECKED_MISSING_DECLARATION = "not_checked_missing_declaration"
    NOT_CHECKED_MISSING_COUNTS = "not_checked_missing_counts"
    UNSUPPORTED_ASSIGNMENT = "unsupported_assignment"
    NOT_APPLICABLE = "not_applicable"


class IntegrityResult(_ReadoutModel):
    status: IntegrityStatus
    analysis_population: Population
    construction: Literal["always_valid", "none"]
    alpha: float | None
    observed: Mapping[str, int] | None
    expected: Mapping[str, float] | None
    randomization_grain: str | None
    code: str | None
    context: Mapping[str, Any]


class PopulationRoster(_ReadoutModel):
    analysis_population: Population
    arms: tuple[str, ...]
    source: Literal["declared_allocation", "experiment_counts", "observed_union", "unknown"]
    complete: bool
    control_group: str | None = None


class Registry[T]:
    """Small registry for orthogonal dispatch axes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        self._entries[key] = value

    def get(self, key: str) -> T:
        if key not in self._entries:
            _refuse(
                "estimation.variance.registry.no_registered_available",
                key=key,
                registry_name=self._name,
                available=sorted(self._entries),
            )
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries


VARIANCE_REDUCTION: Registry[str] = Registry("variance reduction")
VARIANCE_REDUCTION.register("none", "randomized.unadjusted")
VARIANCE_REDUCTION.register("cuped", "randomized.cuped")
VARIANCE_REDUCTION.register("cuped_multicovariate", "randomized.cuped_multicovariate")

AdjustmentFn = Callable[..., list[Any]]
Route = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class MomentsSpec:
    grain: Literal["total", "daily", "asof"] = "total"
    method_specific: bool = False
    include_covariate: bool = False


@dataclass(frozen=True, slots=True)
class UnitRowsSpec:
    outcome_stage: Literal["transformed", "raw"] = "transformed"
    covariates: Literal["design", "method", "none"] = "none"


@dataclass(frozen=True, slots=True)
class CountsSpec:
    kind: Literal["unit", "cluster", "assignment", "triggered"] = "unit"


@dataclass(frozen=True, slots=True)
class MethodFitSpec:
    pass


@dataclass(frozen=True, slots=True)
class EstimatorRef:
    estimator_id: str
    module: str
    attribute: str
    contract: Literal["moments", "panel"]
    estimand: str
    selector: str | None
    needs: tuple[Route, ...] = ()


_TOTAL_MOMENTS = MomentsSpec("total")
_DESIGN_ROWS = UnitRowsSpec(covariates="design")
_METHOD_ROWS = UnitRowsSpec(covariates="method")
_FIT = MethodFitSpec()
BUILTIN_ESTIMATORS: tuple[EstimatorRef, ...] = (
    EstimatorRef("observational.iptw", "increment.estimation._adjust.iptw", "iptw_estimate", "panel", "ate", "iptw", ((_TOTAL_MOMENTS, _DESIGN_ROWS),)),
    EstimatorRef("observational.dml", "increment.estimation._adjust.dml", "dml_estimate", "panel", "plr_slope", "dml", ((_TOTAL_MOMENTS, _DESIGN_ROWS),)),
    EstimatorRef("observational.aipw", "increment.estimation._adjust.aipw", "aipw_estimate", "panel", "ate", "aipw", ((_TOTAL_MOMENTS, _DESIGN_ROWS),)),
    EstimatorRef("randomized.unadjusted", "increment.estimation.engine", "estimate_lift", "moments", "ate", None, ((_TOTAL_MOMENTS,),)),
    EstimatorRef("randomized.cuped", "increment.estimation.engine", "estimate_lift", "moments", "ate", "cuped", ((MomentsSpec("total", method_specific=True),),)),
    EstimatorRef("randomized.cuped_multicovariate", "increment.estimation.engine", "estimate_lift", "panel", "ate", "cuped_multicovariate", ((_FIT,), (_TOTAL_MOMENTS, _DESIGN_ROWS))),
)

_ESTIMATORS: dict[str, EstimatorRef] = {ref.estimator_id: ref for ref in BUILTIN_ESTIMATORS}
_IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {}


def register(ref: EstimatorRef) -> None:
    if not ref.estimator_id or "." not in ref.estimator_id:
        raise ValueError("estimator_id must be a non-empty qualified identifier")
    if ref.estimator_id.startswith("observational.") and not any(
        any(isinstance(spec, MomentsSpec) and spec.grain == "total" for spec in route)
        and any(isinstance(spec, UnitRowsSpec) and spec.covariates == "design" for spec in route)
        for route in ref.needs
    ):
        raise ValueError("observational estimators require a route with total moments and design rows")
    existing = _ESTIMATORS.get(ref.estimator_id)
    if existing is not None and existing != ref:
        raise ValueError(f"estimator {ref.estimator_id!r} is already registered")
    _ESTIMATORS[ref.estimator_id] = ref


def unregister_estimator(estimator_id: str) -> None:
    if estimator_id in {ref.estimator_id for ref in BUILTIN_ESTIMATORS}:
        return
    ref = _ESTIMATORS.pop(estimator_id, None)
    _IMPLEMENTATIONS.pop(estimator_id, None)
    if ref is not None and ref.selector is not None:
        ADJUSTMENTS._entries.pop(ref.selector, None)


def register_implementation(estimator_id: str, fn: Callable[..., Any]) -> None:
    if estimator_id not in _ESTIMATORS:
        raise ValueError(f"implementation has no declared estimator ref: {estimator_id!r}")
    _IMPLEMENTATIONS[estimator_id] = fn


def estimator_refs() -> Mapping[str, EstimatorRef]:
    return _ESTIMATORS


class AdjustmentRegistry(Registry[AdjustmentFn]):
    def __init__(self) -> None:
        super().__init__("adjustment")

    def register(self, key: str, value: AdjustmentFn) -> None:
        estimator_id = estimator_id_for_selector(key) or f"observational.{key}"
        if estimator_id not in _ESTIMATORS:
            register(EstimatorRef(estimator_id, "", "", "panel", "ate", key, ((_TOTAL_MOMENTS, _DESIGN_ROWS),)))
        register_implementation(estimator_id, value)
        self._entries[key] = value

    def get(self, key: str) -> AdjustmentFn:
        if key in self._entries:
            return self._entries[key]
        estimator_id = estimator_id_for_selector(key)
        if estimator_id is not None:
            from importlib import import_module

            ref = _ESTIMATORS[estimator_id]
            implementation = _IMPLEMENTATIONS.get(estimator_id)
            if implementation is None:
                implementation = getattr(import_module(ref.module), ref.attribute)
                register_implementation(estimator_id, implementation)
            self._entries[key] = implementation
            return cast(AdjustmentFn, implementation)
        available = sorted(
            ref.selector
            for ref in _ESTIMATORS.values()
            if ref.estimator_id.startswith("observational.") and ref.selector is not None
        )
        _refuse(
            "estimation.variance.registry.no_registered_available",
            key=key,
            registry_name=self._name,
            available=available,
        )

    def __contains__(self, key: str) -> bool:
        return estimator_id_for_selector(key) is not None

    def unregister(self, key: str) -> None:
        estimator_id = estimator_id_for_selector(key)
        self._entries.pop(key, None)
        if estimator_id is not None and estimator_id not in {
            ref.estimator_id for ref in BUILTIN_ESTIMATORS
        }:
            unregister_estimator(estimator_id)


def estimator_id_for_selector(name: str) -> str | None:
    return next(
        (
            ref.estimator_id
            for ref in _ESTIMATORS.values()
            if ref.estimator_id.startswith("observational.") and ref.selector == name
        ),
        None,
    )


def is_observational_name(name: str) -> bool:
    return estimator_id_for_selector(name) is not None


ADJUSTMENTS = AdjustmentRegistry()
OBSERVATIONAL_METHOD_NAMES = frozenset(
    ref.selector for ref in BUILTIN_ESTIMATORS if ref.estimator_id.startswith("observational.") and ref.selector is not None
)

CovariateArg = str | CupedCovariate | CupedCovariateBinding


def _as_covariate(value: CovariateArg) -> CupedCovariate | CupedCovariateBinding:
    """A bare column name is one numeric covariate; categorical ones are declared explicitly."""
    if isinstance(value, str):
        return CupedCovariate(name=value, kind="continuous")
    return value


def _normalize_method_covariates(
    covariates: CovariateArg | Sequence[CovariateArg] | None,
) -> tuple[CupedCovariate | CupedCovariateBinding, ...] | None:
    if covariates is None:
        return None
    if isinstance(covariates, (str, CupedCovariate)):
        return (_as_covariate(covariates),)
    return tuple(_as_covariate(covariate) for covariate in covariates)


def _normalize_method_covariates_input(value: object) -> object:
    if value is None or isinstance(value, (str, CupedCovariate)):
        return _normalize_method_covariates(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return tuple(
            _as_covariate(item) if isinstance(item, str) else item
            for item in cast(Sequence[object], value)
        )
    return value


class Method(CodedModel, BaseModel):
    """An estimation configuration: named method with optional adjustments."""

    model_config = ConfigDict(frozen=True)

    name: str
    variance_reduction: str = "none"
    conversion_inference: ConversionInference = "auto"
    covariates: tuple[CupedCovariateBinding | CupedCovariate, ...] | None = None
    procedure: Literal["unadjusted"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    propensity_learner: Callable[[], Any] | None = None
    outcome_learner: Callable[[], Any] | None = None
    folds: int | None = None

    @field_validator("covariates", mode="before")
    @classmethod
    def _normalize_covariates(cls, value: object) -> object:
        return _normalize_method_covariates_input(value)

    @model_validator(mode="after")
    def _check_label_matches_configuration(self) -> Method:
        VARIANCE_REDUCTION.get(self.variance_reduction)
        self._validate_method_configuration(self)
        if self.conversion_inference == "finite_sample" and self.variance_reduction in (
            "cuped",
            "cuped_multicovariate",
        ):
            refuse_finite_sample_cuped(self.name)
        if self.name == "cuped" and self.variance_reduction != "cuped":
            _refuse("estimation.engine.method.name_without_variance", self=self.name)
        if self.name == "iptw" and (self.outcome_learner is not None or self.folds is not None):
            _refuse("estimation.engine.method.name_iptw_does")
        return self

    @property
    def is_observational(self) -> bool:
        """Whether the method name selects an observational adjustment."""
        return is_observational_name(self.name)

    @classmethod
    def unadjusted(
        cls, *, name: str = "naive", conversion_inference: ConversionInference = "auto"
    ) -> Method:
        return cls(
            name=name,
            variance_reduction="none",
            conversion_inference=conversion_inference,
            procedure="unadjusted" if name not in {"naive", "unadjusted"} else None,
        )

    @classmethod
    def cuped(
        cls,
        *,
        covariates: CovariateArg | Sequence[CovariateArg] | None = None,
        name: str = "cuped",
    ) -> Method:
        """CUPED with one numeric covariate; a column name such as ``'pre'`` is enough."""
        values = _normalize_method_covariates(covariates)
        return cls(name=name, variance_reduction="cuped", covariates=values)

    @classmethod
    def cuped_multicovariate(
        cls,
        *,
        covariates: CovariateArg | Sequence[CovariateArg],
        name: str = "cuped_multicovariate",
    ) -> Method:
        """CUPED with several covariates; categorical ones are declared explicitly."""
        values = _normalize_method_covariates(covariates)
        return cls(name=name, variance_reduction="cuped_multicovariate", covariates=values)

    @classmethod
    def _validate_method_configuration(cls, method: Method) -> None:
        values = method.covariates
        procedure = method.variance_reduction
        if procedure == "none" and values is not None:
            _refuse("estimation.engine.method_covariates", method=method.name)
        if procedure == "cuped" and values is not None:
            if len(values) != 1 or values[0].kind != "continuous":
                _refuse("estimation.engine.method_covariates", method=method.name)
            if values[0].missing not in ("impute", "zero", "refuse"):
                _refuse("estimation.engine.method_covariates", method=method.name)
        if procedure == "cuped_multicovariate":
            if not values or len({item.name for item in values}) != len(values):
                _refuse("estimation.engine.method_covariates", method=method.name)
            if any(item.missing not in ("refuse", "impute-indicator") for item in values):
                _refuse("estimation.engine.method_covariates", method=method.name)
