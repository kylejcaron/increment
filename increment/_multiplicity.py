from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._literals import MultiplicityCorrection
from increment.errors import CodedModel, InvalidRequestError, raiser, refusals

_MultiplicityGuarantee = Literal["none", "fwer", "fdr"]

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "decision.multiplicity.validate_policy": "{correction} multiplicity requires q",
        "decision.multiplicity.bh": "q is only valid for BH multiplicity, got {correction!r}",
        "decision.multiplicity.guarantee_mismatch": "multiplicity guarantee {guarantee!r} does not match correction {correction!r} -- the exact matrix is none->none, bonferroni->fwer, bh/e_bh->fdr",
    },
)
_raise = raiser(_REFUSALS)


def guarantee_for_correction(correction: MultiplicityCorrection) -> _MultiplicityGuarantee:
    if correction in ("bh", "e_bh"):
        return "fdr"
    if correction == "bonferroni":
        return "fwer"
    return "none"


class MultiplicityFamily(CodedModel, BaseModel):
    """A named multiplicity procedure and the axes it spans."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    correction: MultiplicityCorrection = "none"
    q: float | None = Field(default=None, gt=0.0, lt=1.0)
    axes: tuple[str, ...] = ()
    guarantee: _MultiplicityGuarantee = "none"
    validity_regime: Literal["finite_sample", "asymptotic_sequential"] = Field(
        default="finite_sample", exclude_if=lambda value: value == "finite_sample"
    )

    @model_validator(mode="after")
    def _validate_policy(self) -> MultiplicityFamily:
        if self.validity_regime == "asymptotic_sequential" and self.correction not in (
            "none",
            "bonferroni",
            "e_bh",
        ):
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported",
                "asymptotic families support fixed-roster Bonferroni or e-BH selection only",
            )
        uses_q = self.correction in ("bh", "e_bh") or (
            self.correction == "bonferroni" and self.validity_regime == "asymptotic_sequential"
        )
        if uses_q and self.q is None:
            _raise("decision.multiplicity.validate_policy", correction=self.correction)
        if not uses_q and self.q is not None:
            _raise("decision.multiplicity.bh", correction=self.correction)
        expected_guarantee = guarantee_for_correction(self.correction)
        if self.guarantee != expected_guarantee:
            _raise(
                "decision.multiplicity.guarantee_mismatch",
                correction=self.correction,
                guarantee=self.guarantee,
            )
        return self
