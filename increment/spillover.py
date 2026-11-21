"""Plan a two-stage offer experiment and estimate its effect on people not offered treatment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from increment._spillover_design import SpilloverDesign, SpilloverGroup
from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation.spillover import Z95, reason_text, summarize_counts
from increment.power._spillover import PlanningEvidence, evaluate_design

__all__ = ["SpilloverDesign", "SpilloverGroup", "plan_spillover"]

_REFUSALS = refusals(
    InvalidRequestError,
    {"spillover.design.required": "Provide the two-stage design frozen before assignment."},
)
_raise = raiser(_REFUSALS)


@dataclass(frozen=True, slots=True)
class _SpilloverResult:
    outcome: str
    horizon: str
    estimate: float
    lower: float | None
    upper: float | None
    observed_reason: str | None
    qualification_reason: str | None
    planning_status: str

    @property
    def unavailable_reason(self) -> str | None:
        """Why the approximate interval was withheld, if it was."""
        return self.observed_reason or self.qualification_reason


@dataclass(frozen=True, slots=True)
class _SpilloverPlan:
    design: SpilloverDesign = field(repr=False)
    meaningful_effect: float
    baseline_rate: float | None
    evidence: PlanningEvidence = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.design, SpilloverDesign):
            _raise("spillover.design.required")
        evidence = evaluate_design(self.design, self.baseline_rate, self.meaningful_effect)
        object.__setattr__(self, "evidence", evidence)

    @property
    def message(self) -> str:
        """A pre-launch interpretation for the signed change the analyst chose."""
        if self.evidence.calibration_passed:
            effect = next(s.true_effect for s in self.evidence.scenarios if s.true_effect != 0)
            change = f"{100 * effect:+.6g} percentage points"
            if self.evidence.status == "supported":
                return (
                    f"All six scenarios modeling a change of {change} met the 80% detection target."
                )
            if self.evidence.status == "underpowered":
                return (
                    f"At least one scenario modeling a change of {change} fell short of 80% detection. "
                    "Not detecting the change would not rule it out."
                )
            return f"The simulations could not establish 80% detection for a change of {change}."
        if self.evidence.reason == "missing_baseline_rate":
            return "Precision was not assessed. Supply a comparable pre-experiment event rate to check it."
        return (
            "This plan did not establish reliable uncertainty across the modeled scenarios. "
            "Report an estimate only; do not use detection rates to size the experiment."
        )

    def analyze(
        self,
        *,
        group_high: Mapping[str, int | bool],
        treated: Mapping[str, int | bool],
        outcomes: Mapping[str, int | bool],
    ) -> _SpilloverResult:
        """Read both assignment logs and the binary outcome at the declared horizon."""
        counts = self.design.logged_counts(group_high, treated, outcomes)
        stats = summarize_counts(self.design, *counts)
        estimate = float(stats.estimate[0])
        observed_reason = reason_text(int(stats.reason[0]))
        qualification_reason = None if self.evidence.calibration_passed else self.evidence.reason
        lower = upper = None
        if observed_reason is None and self.evidence.calibration_passed:
            margin = Z95 * float(stats.standard_error[0])
            lower, upper = estimate - margin, estimate + margin
        return _SpilloverResult(
            outcome=self.design.outcome,
            horizon=self.design.horizon,
            estimate=estimate,
            lower=lower,
            upper=upper,
            observed_reason=observed_reason,
            qualification_reason=qualification_reason,
            planning_status=self.evidence.status,
        )


def plan_spillover(
    design: SpilloverDesign, *, meaningful_effect: float, baseline_rate: float | None = None
) -> _SpilloverPlan:
    """Check detection before assignment; effects and rates are proportions, not percentages.

    The optional prior event rate drives fixed sensitivity scenarios, not adjustment of the
    measured effect. Qualification applies only to those scenarios, under no effects between
    groups and dependence within groups on how many peers received offers, not which peers.
    """
    return _SpilloverPlan(design, meaningful_effect, baseline_rate)
