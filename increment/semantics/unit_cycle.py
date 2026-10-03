"""Prospective unit-cycle assumptions and portable population laws.

Provenance records a scientific declaration; it does not prove that it is true.
Population type weights are relative probabilities, never analysis weights.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from increment.errors import CodedModel, InvalidRequestError, RefusalSpec, refuse, unwrap_coded
from increment.semantics.assignment import SwitchbackAssignment

_REFUSALS = {
    code: RefusalSpec(code, InvalidRequestError, lambda *, reason: reason)
    for code in (
        "definition.unit_cycle.assignment",
        "definition.unit_cycle.groups",
        "definition.unit_cycle.cycles",
        "definition.unit_cycle.nonempty",
        "definition.unit_cycle.aggregation",
        "definition.unit_cycle.references",
    )
}


class _UnitCycleModel(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")


class ProspectiveAssumptionProvenance(_UnitCycleModel):
    assumption_id: str = Field(min_length=1)
    assumption_version: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    declaration_id: str = Field(min_length=1)

    @field_validator("*")
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            refuse(_REFUSALS["definition.unit_cycle.nonempty"], reason="empty provenance")
        return value


def _copy_assignment(value: object) -> SwitchbackAssignment:
    # Assignment models predate instance revalidation; validate every nested
    # declaration as a plain mapping so forged child instances cannot bypass it.
    def _plain(item: object) -> object:
        if isinstance(item, BaseModel):
            item = dict(item)
        if isinstance(item, Mapping):
            return {key: _plain(nested) for key, nested in item.items()}
        return item

    assignment = SwitchbackAssignment.model_validate(_plain(value))
    if assignment.sequence.scheme != "independent_bernoulli_order":
        refuse(
            _REFUSALS["definition.unit_cycle.assignment"],
            reason="unit-cycle declarations require independent Bernoulli orders",
        )
    return assignment


class UnitCycleVarianceEnvelope(_UnitCycleModel):
    """Optional finite-sample research/reference envelope.

    This is not required by the ordinary pilot-fitted inference path.
    It asserts ``E[A-delta G]=0`` and
    ``Var(mean(A-delta G)) <= V0/N`` under the declared unit population.
    """

    kind: Literal["unit_cycle_residual_variance_v1"] = "unit_cycle_residual_variance_v1"
    assignment: SwitchbackAssignment
    metric: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    aggregation: Literal["sum"] = "sum"
    response_meaning: Literal["retained_total", "pre_normalized_retained_mean"]
    estimand: Literal["mean_unit_retained_window_difference"] = (
        "mean_unit_retained_window_difference"
    )
    cycles_per_unit: int = Field(ge=1, strict=True)
    target_population: Literal["independent_unit_population"] = "independent_unit_population"
    effect_model: Literal["additive_retained_aggregate_shift"] = "additive_retained_aggregate_shift"
    residual_variance_upper: float = Field(ge=0, allow_inf_nan=False)
    provenance: ProspectiveAssumptionProvenance

    _assignment = field_validator("assignment", mode="before")(_copy_assignment)

    @field_validator("aggregation", mode="before")
    @classmethod
    def _aggregation(cls, value: object) -> object:
        if value != "sum":
            refuse(
                _REFUSALS["definition.unit_cycle.aggregation"],
                reason="the additive residual envelope supports sum aggregation only",
            )
        return value

    @model_validator(mode="after")
    def _groups(self) -> UnitCycleVarianceEnvelope:
        if self.control_group == self.treatment_group:
            refuse(_REFUSALS["definition.unit_cycle.groups"], reason="groups must be distinct")
        return self


class UnitCycleTApproximation(_UnitCycleModel):
    """Qualified Student-t approximation on the sample variance of unit contributions.

    The interval uses ``dof = n_units - 1`` and has no finite-sample calibration.
    It is the default reference for independent unit-cycle orders when none is declared.
    """

    kind: Literal["unit_t_approximation"] = "unit_t_approximation"


UnitCycleReference = Annotated[
    UnitCycleVarianceEnvelope | UnitCycleTApproximation, Field(discriminator="kind")
]
_REFERENCE_ADAPTER = TypeAdapter(UnitCycleReference)


def _copy_references(value: object) -> Mapping[str, UnitCycleReference]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        refuse(_REFUSALS["definition.unit_cycle.references"], reason="references must be a mapping")
    copied = {}
    for key, reference in value.items():
        if not isinstance(key, str) or not key.strip():
            refuse(_REFUSALS["definition.unit_cycle.references"], reason="invalid metric key")
        try:
            parsed = _REFERENCE_ADAPTER.validate_python(reference)
        except (ValidationError, InvalidRequestError) as exc:
            if isinstance(exc, ValidationError):
                unwrap_coded(exc)
            raise
        if isinstance(parsed, UnitCycleVarianceEnvelope) and parsed.metric != key:
            refuse(
                _REFUSALS["definition.unit_cycle.references"], reason="reference metric mismatch"
            )
        copied[key] = parsed
    return MappingProxyType(copied)


class NormalInnovation(_UnitCycleModel):
    """Standard mean-zero, variance-one normal innovation."""

    kind: Literal["normal"] = "normal"


class CenteredLognormalInnovation(_UnitCycleModel):
    """Standardized exp(shape Z), with exact mean zero and variance one."""

    kind: Literal["centered_lognormal"] = "centered_lognormal"
    shape: float = Field(gt=0, allow_inf_nan=False)


class CenteredGammaInnovation(_UnitCycleModel):
    """(Gamma(shape, 1) - shape) / sqrt(shape)."""

    kind: Literal["centered_gamma"] = "centered_gamma"
    shape: float = Field(gt=0, allow_inf_nan=False)


UnitCycleInnovation = Annotated[
    NormalInnovation | CenteredLognormalInnovation | CenteredGammaInnovation,
    Field(discriminator="kind"),
]


class UnitCycleCycleLaw(_UnitCycleModel):
    """Signed treatment-minus-control contrasts at the reference population effect."""

    ct_mean: float = Field(allow_inf_nan=False)
    tc_mean: float = Field(allow_inf_nan=False)
    ct_noise_load: float = Field(allow_inf_nan=False)
    tc_noise_load: float = Field(allow_inf_nan=False)
    ct_innovation_count: int = Field(gt=0, strict=True)
    tc_innovation_count: int = Field(gt=0, strict=True)


class UnitCycleTypeLaw(_UnitCycleModel):
    weight: float = Field(gt=0, allow_inf_nan=False)
    cycles: tuple[UnitCycleCycleLaw, ...] = Field(min_length=1)


class UnitCycleJointLaw(_UnitCycleModel):
    """Complete-law research oracle for independent calibration.

    Ordinary inference and planning do not require a declared population law.
    This model is retained for simulation, exact-law comparisons and
    model-conditioned Monte Carlo validation.
    """

    kind: Literal["finite_type_reuse_unit_cycle_v1"] = "finite_type_reuse_unit_cycle_v1"
    assignment: SwitchbackAssignment
    metric: str = Field(min_length=1)
    control_group: str = Field(min_length=1)
    treatment_group: str = Field(min_length=1)
    response_meaning: Literal["retained_total", "pre_normalized_retained_mean"]
    types: tuple[UnitCycleTypeLaw, ...] = Field(min_length=1)
    innovation: UnitCycleInnovation
    reuse_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    provenance: ProspectiveAssumptionProvenance

    _assignment = field_validator("assignment", mode="before")(_copy_assignment)

    @model_validator(mode="after")
    def _consistent(self) -> UnitCycleJointLaw:
        if self.control_group == self.treatment_group:
            refuse(_REFUSALS["definition.unit_cycle.groups"], reason="groups must be distinct")
        if len({len(item.cycles) for item in self.types}) != 1:
            refuse(
                _REFUSALS["definition.unit_cycle.cycles"],
                reason="all population types must have the same number of cycles",
            )
        return self


__all__ = [
    "ProspectiveAssumptionProvenance",
    "UnitCycleVarianceEnvelope",
    "UnitCycleTApproximation",
    "UnitCycleReference",
    "NormalInnovation",
    "CenteredLognormalInnovation",
    "CenteredGammaInnovation",
    "UnitCycleInnovation",
    "UnitCycleCycleLaw",
    "UnitCycleTypeLaw",
    "UnitCycleJointLaw",
]
