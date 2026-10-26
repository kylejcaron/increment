"""Lightweight shared declaration types and registries.

This module is intentionally independent of the estimation implementation so callers can build
method declarations without importing estimator machinery.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from increment._finite_sample_refusals import refuse_finite_sample_cuped
from increment._literals import ConversionInference
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)
from increment.semantics.models import CupedCovariate, CupedCovariateBinding
from increment._finite_sample_refusals import refuse_finite_sample_cuped


_DECLARATION_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.engine.method.name_without_variance": "Method(name={self!r}) without its matching variance_reduction would label an unadjusted estimate as CUPED-adjusted.",
        "estimation.engine.method_covariates": "method {method!r} has an invalid CUPED covariate declaration for its variance-reduction procedure",
        "estimation.engine.method.name_iptw_does": "Method(name='iptw') does not accept outcome_learner/folds -- IPTW fits one propensity model with no cross-fitting; set propensity_learner (mapped to iptw_estimate's learner=) instead, or use Method(name='dml') / Method(name='aipw') for cross-fit outcome-model pluggability.",
        "estimation.variance.registry.no_registered_available": "{registry_name} {key!r} is not registered; available: {available}",
    },
)
_refuse_declaration = raiser(_DECLARATION_REFUSALS)


class Registry[T]:
    """Small registry for orthogonal dispatch axes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        self._entries[key] = value

    def get(self, key: str) -> T:
        if key not in self._entries:
            _refuse_declaration(
                "estimation.variance.registry.no_registered_available",
                key=key,
                registry_name=self._name,
                available=sorted(self._entries),
            )
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries


VARIANCE_REDUCTION: Registry[str] = Registry("variance reduction")
VARIANCE_REDUCTION.register("none", "none")
VARIANCE_REDUCTION.register("cuped", "cuped")
VARIANCE_REDUCTION.register("cuped_multicovariate", "cuped_multicovariate")

AdjustmentFn = Callable[..., list[Any]]
ADJUSTMENTS: Registry[AdjustmentFn] = Registry("adjustment")
OBSERVATIONAL_METHOD_NAMES = frozenset({"iptw", "dml", "aipw"})

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
            _refuse_declaration("estimation.engine.method.name_without_variance", self=self.name)
        if self.name == "iptw" and (self.outcome_learner is not None or self.folds is not None):
            _refuse_declaration("estimation.engine.method.name_iptw_does")
        return self

    @property
    def is_observational(self) -> bool:
        """Whether the method name selects an observational adjustment."""
        return self.name in OBSERVATIONAL_METHOD_NAMES or self.name in ADJUSTMENTS

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
            _refuse_declaration("estimation.engine.method_covariates", method=method.name)
        if procedure == "cuped" and values is not None:
            if len(values) != 1 or values[0].kind != "continuous":
                _refuse_declaration("estimation.engine.method_covariates", method=method.name)
            if values[0].missing not in ("impute", "zero", "refuse"):
                _refuse_declaration("estimation.engine.method_covariates", method=method.name)
        if procedure == "cuped_multicovariate":
            if not values or len({item.name for item in values}) != len(values):
                _refuse_declaration("estimation.engine.method_covariates", method=method.name)
            if any(item.missing not in ("refuse", "impute-indicator") for item in values):
                _refuse_declaration("estimation.engine.method_covariates", method=method.name)
