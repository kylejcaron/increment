"""Pure plan-family compatibility decision contract.

Frozen declaration-time contract. Effective-prior eligibility is resolved
per readout. The inspector consumes this module's output without being
imported here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from increment._literals import Role
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)

InferenceKind = Literal["fixed", "always_valid", "asymptotic_mean"]
PlanFamilyParticipation = Literal["participates", "excluded", "not_applicable"]
RuntimeEffect = Literal["limited", "not_applicable"]


class PlanFamilyCompatibilityRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    # Keep this discriminator open so it mirrors ``Metric.type`` without a
    # second enum that must change whenever a metric type is added.
    metric_type: str
    role: Role
    inference: InferenceKind


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.plan_compatibility.plan_family.runtime_effect_limited": "runtime_effect must be limited for warnings/exclusions and not_applicable otherwise",
    },
)
_raise = raiser(_REFUSALS)


class PlanFamilyDecision(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_effect: RuntimeEffect
    participation: PlanFamilyParticipation
    warning: str | None

    @model_validator(mode="after")
    def _runtime_matches_context(self) -> PlanFamilyDecision:
        expected: RuntimeEffect = (
            "limited"
            if self.warning is not None or self.participation == "excluded"
            else "not_applicable"
        )
        if self.runtime_effect != expected:
            _raise("facade.plan_compatibility.plan_family.runtime_effect_limited")
        return self


def plan_family_compatibility(
    request: PlanFamilyCompatibilityRequest,
) -> PlanFamilyDecision:
    quantile_under_always_valid = (
        request.metric_type == "quantile" and request.inference == "always_valid"
    )
    warning = (
        f"metric {request.metric!r}: quantile sequential inference is refused because "
        "the registered Beta/NIG/NIW likelihoods target means and ratios of means. "
        "Use fixed-horizon quantile inference (valid for one planned analysis, not "
        "repeated looks)."
        if quantile_under_always_valid
        else None
    )
    participation: PlanFamilyParticipation = (
        "not_applicable"
        if request.role != "secondary"
        else "excluded"
        if quantile_under_always_valid
        else "participates"
    )
    runtime_effect: RuntimeEffect = (
        "limited" if warning is not None or participation == "excluded" else "not_applicable"
    )
    return PlanFamilyDecision(
        runtime_effect=runtime_effect,
        participation=participation,
        warning=warning,
    )
