"""Serializable declarations for genuine fractional-relative priors."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from increment._prior_validation import validate_mixture_prior
from increment.errors import MODEL_FIELD_REFUSALS, CodedModel, refuse


class _PriorSpec(CodedModel, BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, revalidate_instances="always")


class RelativeNormalSpec(_PriorSpec):
    """Normal mean and standard deviation in fractional relative-lift units."""

    mu: float = Field(allow_inf_nan=False)
    sigma: float = Field(gt=0, allow_inf_nan=False)
    type: Literal["normal"] = "normal"


class RelativeStudentTSpec(_PriorSpec):
    """Zero-centered Student-t declaration in fractional relative-lift units."""

    type: Literal["student_t"] = "student_t"
    nu: float = Field(gt=0, le=200, allow_inf_nan=False)
    scale: float = Field(gt=0, allow_inf_nan=False)
    k: int = Field(default=80, ge=2)


class RelativeMixtureSpec(_PriorSpec):
    """Ordered Normal-mixture components in fractional relative-lift units."""

    type: Literal["normal_mixture"] = "normal_mixture"
    weights: tuple[float, ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]

    @model_validator(mode="after")
    def _consistent(self) -> RelativeMixtureSpec:
        validate_mixture_prior(self.weights, self.means, self.sigmas)
        return self


class RelativeLiftPriorSpec(_PriorSpec):
    """Explicit relative-scale prior, with unrestricted real support."""

    type: Literal["relative_lift"] = "relative_lift"
    distribution: Annotated[
        RelativeNormalSpec | RelativeStudentTSpec | RelativeMixtureSpec,
        Field(discriminator="type"),
    ]

    @field_validator("distribution", mode="before")
    @classmethod
    def _known_distribution(cls, value: object) -> object:
        if isinstance(value, (RelativeNormalSpec, RelativeStudentTSpec, RelativeMixtureSpec)):
            return value
        if isinstance(value, Mapping):
            tag = value.get("type")
            if isinstance(tag, str) and tag in ("normal", "student_t", "normal_mixture"):
                return value
            code, field, loc = "model.field.literal", "distribution.type", ("distribution", "type")
            invalid_value = tag if isinstance(tag, str) else repr(tag)
        else:
            code, field, loc = "model.field.type", "distribution", ("distribution",)
            invalid_value = type(value).__name__
        refuse(
            MODEL_FIELD_REFUSALS[code],
            model="RelativeLiftPrior",
            field=field,
            loc=loc,
            value=invalid_value,
            constraint=("normal", "student_t", "normal_mixture"),
        )
