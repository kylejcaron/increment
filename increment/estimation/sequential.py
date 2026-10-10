"""Registered raw-likelihood runtime policies and the always-valid Gaussian
planning boundary.

The public runtime separates exact Bernoulli likelihoods and scalar-mean AsympCSs.
``GaussianScoreMixture`` below is a Gaussian-model planning approximation only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
    refuse,
)
from increment.semantics.sequential import (
    ASYMPTOTIC_LAWS,
    MIXED_REQUIRES_ASYMPTOTIC_MEAN,
    SequentialRegistration,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "power.fractions_nonempty": "fractions must be nonempty",
        "power.information_fractions_lie": "information fractions must lie in (0, 1]; got {fr}",
        "power.information_fractions_strictly": "information fractions must be strictly increasing; got {fr}",
    },
)
_refuse = raiser(_REFUSALS)


def _validate_information_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    """Nonempty, strictly increasing information fractions in ``(0, 1]``.

    Shared by every sequential boundary implementation so a caller-supplied
    look schedule is rejected identically regardless of which spec ends up
    dispatching it. A partial schedule (not reaching 1.0) is allowed here,
    as runtime boundary callers supply the looks taken so far.
    """
    fr = tuple(float(t) for t in fractions)
    if not fr:
        _refuse("power.fractions_nonempty")
    if any(not math.isfinite(t) or not (0.0 < t <= 1.0) for t in fr):
        _refuse("power.information_fractions_lie", fr=fr)
    if any(b <= a for a, b in zip(fr, fr[1:], strict=False)):
        _refuse("power.information_fractions_strictly", fr=fr)
    return fr


def _log1pexp(value: float) -> float:
    """Stable ``log(1 + exp(value))`` for finite values."""
    if value > 0.0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


@dataclass(frozen=True, slots=True)
class SequentialSupportRequest:
    """Inputs needed to validate sequential estimator support."""

    inference: AsymptoticMean | AlwaysValid | MixedFamily | None
    uses_cuped: bool


def sequential_support_refusal(request: SequentialSupportRequest) -> str | None:
    """Return the stable refusal code for unsupported sequential adjustments."""

    if request.inference is not None and request.uses_cuped:
        return "arm.adjustment.sequential_cuped"
    return None


def _validate_population_pair(policy) -> None:
    """A runtime policy may carry only the derivation of its own assigned registration."""
    if policy.triggered_registration is not None:
        from increment.semantics.sequential import validate_triggered_registration

        validate_triggered_registration(policy.registration, policy.triggered_registration)


class AlwaysValid(CodedModel, BaseModel):
    """Raw likelihood evidence with a committed model, roster and reveal law."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    registration: SequentialRegistration
    triggered_registration: SequentialRegistration | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def label(self) -> str:
        return "always_valid"

    @model_validator(mode="after")
    def _policy(self):
        from increment.semantics.sequential import invalid_registration

        laws = {m.law for m in self.registration.models}
        asymptotic = laws & set(ASYMPTOTIC_LAWS)
        if asymptotic:
            if "bernoulli" in laws:
                refuse(
                    MIXED_REQUIRES_ASYMPTOTIC_MEAN,
                    metrics=sorted(
                        m.metric for m in self.registration.models if m.law in asymptotic
                    ),
                )
            invalid_registration("asymptotic laws require explicit AsymptoticMean inference")
        _validate_population_pair(self)
        return self

    def allocated_alpha(self, alpha: Fraction, n: int) -> Fraction:
        return alpha


class AsymptoticMean(CodedModel, BaseModel):
    """Registered count-clock scalar means; asymptotic, never exact e-values."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    registration: SequentialRegistration
    triggered_registration: SequentialRegistration | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def _policy(self):
        from increment.semantics.sequential import invalid_registration

        if any(m.law not in ASYMPTOTIC_LAWS for m in self.registration.models):
            invalid_registration("AsymptoticMean requires an exclusively asymptotic registration")
        _validate_population_pair(self)
        return self

    @property
    def label(self) -> str:
        return "asymptotic_mean"

    def allocated_alpha(self, alpha: Fraction, n: int) -> Fraction:
        return alpha


class MixedFamily(CodedModel, BaseModel):
    """Internal runtime policy composing an asymptotic scalar-mean ITT cell with
    an exact Bernoulli uptake compliance cell from one joint registration.

    Never exported and never a value of ``InferenceSpec.kind``: a user declares
    ``InferenceSpec(kind="asymptotic_mean")`` plus ``AnalysisPlan.compliance``
    and an encouragement design, and ``_build_inference`` composes this policy
    from the registration's own law content. Each cell is evaluated by its own
    law's evidence, and the in-family cells of the joint roster are selected
    together by e-BH (see ``select_sequential_family``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    registration: SequentialRegistration
    triggered_registration: SequentialRegistration | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def label(self) -> str:
        return "asymptotic_mean"

    @model_validator(mode="after")
    def _policy(self):
        from increment.semantics.sequential import invalid_registration

        laws = {m.law for m in self.registration.models}
        if "bernoulli" not in laws or not (laws - {"bernoulli"}):
            invalid_registration(
                "MixedFamily requires both an asymptotic law and a Bernoulli uptake model"
            )
        _validate_population_pair(self)
        return self

    def allocated_alpha(self, alpha: Fraction, n: int) -> Fraction:
        return alpha


SEQUENTIAL_POLICIES = (AsymptoticMean, AlwaysValid, MixedFamily)
ASYMPTOTIC_PROCEDURE_POLICIES = (AsymptoticMean, MixedFamily)
UPTAKE_COMPLETION_POLICIES = (AlwaysValid, MixedFamily)


def compose_asymptotic_or_mixed(
    registration: SequentialRegistration,
    *,
    triggered_registration: SequentialRegistration | None = None,
) -> AsymptoticMean | MixedFamily:
    """Compose the asymptotic-procedure runtime policy a registration's own
    law content calls for: ``MixedFamily`` when it mixes a Bernoulli uptake
    model into an asymptotic roster, ``AsymptoticMean`` otherwise."""
    laws = {m.law for m in registration.models}
    if "bernoulli" in laws and (laws - {"bernoulli"}):
        return MixedFamily(registration=registration, triggered_registration=triggered_registration)
    return AsymptoticMean(registration=registration, triggered_registration=triggered_registration)


class GaussianScoreMixture(CodedModel, BaseModel):
    """Internal: the always-valid asymptotic_mean boundary the count-clock
    runtime executes. Not part of the public surface -- construct planning
    requests via ArmPlanningProcedure.standard(inference=InferenceSpec(kind=
    "asymptotic_mean")), which builds this internally. c^2 = (1+1/r)(log1p(r)
    - 2 log alpha_directional) at r = information_fraction * mixture_r_star(
    alpha), the mixture's own se-independent optimal tuning at the RAW alpha
    -- see estimation.asymptotic_mean.mixture_r_star."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    calibration: Literal["gaussian_model_approximation"] = "gaussian_model_approximation"
