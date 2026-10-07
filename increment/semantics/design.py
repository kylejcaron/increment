"""Identification declarations: what makes a causal comparison valid.

The discriminated union of identification mechanisms: ``Randomized``
(named control arm), ``Encouragement`` (randomized uptake, identifying
ITT and, under a declared exclusion restriction, LATE), and
``Observational`` (adjustment-set identification gated by an
overlap/positivity policy). ``Design`` dispatches on ``mechanism``.

Re-exported at the top level as ``increment.Design``: this is the
identification concept every readout's ``design=`` argument names. It is
an annotation-only alias -- construct one of its members
(``Randomized``/``Encouragement``/``Observational``), which are exported
alongside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from increment.errors import (
    CapabilityError,
    CodedValidationMixin,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.semantics.assignment import ASSIGNMENT_REJECT_BOOL

_ALLOCATION_INVALID = RefusalSpec(
    "design.allocation.invalid",
    InvalidRequestError,
    template="allocation {allocation!r} is not a valid assignment split for control group {control_group!r}: it must be non-empty, contain {control_group!r} as a key, and every weight must be finite and strictly positive",
)

ALLOCATION_SCHEME_INCOMPATIBLE = RefusalSpec(
    "design.allocation_scheme.incompatible",
    InvalidRequestError,
    template="allocation_scheme {allocation_scheme!r} is incompatible with {design!r}",
)

AllocationScheme = Literal["independent", "blocked", "adaptive", "quota", "fixed_counts"]


# Shared by the readout layer and the frame constructors, which refuse it earlier.
RETENTION_UNDER_ENCOURAGEMENT = RefusalSpec(
    "readout.encouragement.retention",
    CapabilityError,
    template="retention metric(s) {names!r} are not supported under an encouragement design; its maturity gate changes the first-stage population by outcome metric",
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "definition.adjustment_set.adjustmentset_covariates_contains": RefusalSpec(
            "definition.adjustment_set.adjustmentset_covariates_contains",
            InvalidRequestError,
            lambda *, dups: (
                f"AdjustmentSet.covariates contains duplicate column(s): "
                f"{', '.join(sorted(dups))} -- each covariate may be declared once"
            ),
        ),
        "design.allocation_scheme.incompatible": ALLOCATION_SCHEME_INCOMPATIBLE,
    },
)
_raise = raiser(_REFUSALS)


def _reject_bool(v: object) -> object:
    """``bool`` is an ``int`` subclass in Python; a declared count field
    must never silently accept ``True``/``False`` as ``1``/``0``."""
    if isinstance(v, bool):
        refuse(ASSIGNMENT_REJECT_BOOL)
    return v


def _freeze_allocation(v: Mapping[str, float] | None) -> MappingProxyType[str, float] | None:
    """Snapshot a caller-owned allocation mapping into an immutable view,
    so mutating the caller's dict after construction cannot silently
    change a validated SRM expectation."""
    return None if v is None else MappingProxyType(dict(v))


def _serialize_allocation(v: Mapping[str, float] | None, _info: object) -> dict[str, float] | None:
    return None if v is None else dict(v)


def _validate_allocation_semantics(
    control_group: str, allocation: Mapping[str, float] | None
) -> None:
    """Refuse an allocation that cannot describe a real assignment split.

    Weights need not sum to 1: ``sample_ratio_mismatch``
    (``increment/estimation/diagnostics.py``) evaluates the e-process
    "against the normalized expected allocation", normalizing expected
    shares before comparing them to observed counts.
    """
    if allocation is None:
        return
    if not allocation:
        refuse(_ALLOCATION_INVALID, control_group=control_group, allocation=dict(allocation))
    if control_group not in allocation:
        refuse(_ALLOCATION_INVALID, control_group=control_group, allocation=dict(allocation))
    if any(not math.isfinite(float(v)) or v <= 0 for v in allocation.values()):
        refuse(_ALLOCATION_INVALID, control_group=control_group, allocation=dict(allocation))


class Randomized(BaseModel):
    """A randomized assignment with a named control arm."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Literal["randomized"] = "randomized"
    control_group: str
    #: Target allocation weights; SRM checks against them. Required unless
    #: the caller passes ``expected``; fixed inference falls back to equal split.
    allocation: Mapping[str, float] | None = None
    #: Assignment law; proportions alone never declare independence.
    allocation_scheme: AllocationScheme | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @field_validator("allocation")
    @classmethod
    def _validate_allocation(
        cls, v: Mapping[str, float] | None
    ) -> MappingProxyType[str, float] | None:
        return _freeze_allocation(v)

    @field_serializer("allocation")
    def _dump_allocation(
        self, v: Mapping[str, float] | None, info: object
    ) -> dict[str, float] | None:
        return _serialize_allocation(v, info)

    @model_validator(mode="after")
    def _check_allocation_semantics(self) -> Randomized:
        _validate_allocation_semantics(self.control_group, self.allocation)
        return self

    def __hash__(self) -> int:
        items = tuple(sorted(self.allocation.items())) if self.allocation is not None else None
        return hash((self.mechanism, self.control_group, items, self.allocation_scheme))


class UptakeSpec(BaseModel):
    """The binary uptake fact of an encouragement design.

    ``fact`` names the uptake event (0/1 column on the dataframe path,
    overridable via ``uptake=``); ``window_days=None`` means ever-took-up,
    else uptake freezes to the first *window_days* days after exposure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fact: str
    window_days: int | None = Field(default=None, ge=1)

    @field_validator("window_days", mode="before")
    @classmethod
    def _validate_window_days(cls, v: object) -> object:
        return None if v is None else _reject_bool(v)


class ExclusionRestriction(BaseModel):
    """Explicit acknowledgment that assignment moves the outcome only
    through uptake. Untestable from data, so it must be declared, like
    ``Observational.adjustment``: assumptions stay visible, never implicit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    acknowledged: Literal[True]
    justification: str = Field(min_length=10)


class Encouragement(BaseModel):
    """A randomized encouragement: assignment is random, uptake is chosen.

    Identifies ITT and compliance without an exclusion declaration; LATE
    requires it. ``one_sided`` declares control cannot take up (the
    estimator hard-errors on any control uptake). ``min_first_stage_z``
    gates LATE emission - below it, only ITT and compliance are reported.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Literal["encouragement"] = "encouragement"
    control_group: str
    uptake: UptakeSpec
    exclusion_restriction: ExclusionRestriction | None = None
    one_sided: bool = False
    #: Target allocation weights - same conditional-assignment SRM contract
    #: as ``Randomized.allocation`` (encouragement assignment is randomized).
    allocation: Mapping[str, float] | None = None
    allocation_scheme: AllocationScheme | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    min_first_stage_z: float = Field(default=4.0, gt=0.0, allow_inf_nan=False)

    @field_validator("allocation")
    @classmethod
    def _validate_allocation(
        cls, v: Mapping[str, float] | None
    ) -> MappingProxyType[str, float] | None:
        return _freeze_allocation(v)

    @field_serializer("allocation")
    def _dump_allocation(
        self, v: Mapping[str, float] | None, info: object
    ) -> dict[str, float] | None:
        return _serialize_allocation(v, info)

    @model_validator(mode="after")
    def _check_allocation_semantics(self) -> Encouragement:
        _validate_allocation_semantics(self.control_group, self.allocation)
        return self

    def __hash__(self) -> int:
        items = tuple(sorted(self.allocation.items())) if self.allocation is not None else None
        return hash(
            (
                self.mechanism,
                self.control_group,
                self.uptake,
                self.exclusion_restriction,
                self.one_sided,
                items,
                self.allocation_scheme,
                self.min_first_stage_z,
            )
        )


class AdjustmentSet(BaseModel):
    """Covariates declared sufficient to identify treatment effect in an
    observational comparison; a declaration, not a validation - callers
    must argue elsewhere that it satisfies conditional ignorability.

    ``missing`` controls null/NaN handling: ``"refuse"`` (default)
    raises; ``"impute-indicator"`` pooled-mean imputes with a balanced
    indicator; ``"pattern"`` fits propensity per missingness pattern;
    ``"complete-case"`` keeps fully-observed units only; ``"allow"``
    passes NaN to explicit NaN-native learners.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    covariates: tuple[str, ...] = Field(min_length=1)
    missing: Literal["refuse", "impute-indicator", "pattern", "complete-case", "allow"] = "refuse"

    @field_validator("covariates")
    @classmethod
    def _no_duplicate_covariates(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        # A duplicated column enters the design matrix twice downstream
        # (rank deficiency or silently doubled weight, learner-dependent).
        seen: set[str] = set()
        dups: set[str] = set()
        for c in v:
            (dups if c in seen else seen).add(c)
        if dups:
            _raise("definition.adjustment_set.adjustmentset_covariates_contains", dups=dups)
        return v


class IdentificationGate(BaseModel):
    """Overlap/positivity policy for an observational comparison.

    Defaults refuse rather than silently proceed when propensities are
    extreme (``overlap="trim"`` opts in to dropping poor overlap
    instead). ``min_propensity`` bounds a symmetric band: a fitted
    propensity outside ``[min_propensity, 1 - min_propensity]`` trips.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    overlap: Literal["refuse", "trim"] = "refuse"
    min_propensity: float = Field(default=0.01, gt=0.0, lt=0.5)
    # None: SMDs are advisory (warn > 0.1); set: exceeding it refuses.
    # Must be >= 0 - a negative threshold would silently force refuse-always.
    max_smd: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)


class Observational(CodedValidationMixin, BaseModel):
    """A non-randomized comparison identified via an explicit adjustment set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mechanism: Literal["observational"] = "observational"
    control_group: str
    adjustment: AdjustmentSet
    gate: IdentificationGate = IdentificationGate()

    @model_validator(mode="before")
    @classmethod
    def _refuse_allocation_scheme(cls, value: object) -> object:
        if isinstance(value, Mapping) and value.get("allocation_scheme") is not None:
            refuse(
                ALLOCATION_SCHEME_INCOMPATIBLE,
                design="observational",
                allocation_scheme=value["allocation_scheme"],
            )
        return value


Design = Annotated[Randomized | Encouragement | Observational, Field(discriminator="mechanism")]


READOUT_ENCOURAGEMENT_RETENTION = RETENTION_UNDER_ENCOURAGEMENT
