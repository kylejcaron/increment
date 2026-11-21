"""Plan a two-stage offer experiment and estimate its effect on people not offered treatment."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Never, Self

from increment._canonical import canonical_digest_bytes
from increment._evidence_loaders import SpilloverSource, freeze_spillover, observe_spillover
from increment._spillover_design import (
    SpilloverComparison,
    SpilloverDesign,
    SpilloverPopulation,
    refuse_input,
)
from increment.errors import CodedError
from increment.estimation.spillover import critical_value, estimate_comparison
from increment.kernel.contracts import _refuse
from increment.power._spillover import PlanningEvidence, evaluate_design

if TYPE_CHECKING:
    from increment.frame import IntoDataFrame

__all__ = [
    "SpilloverComparison",
    "SpilloverDesign",
    "SpilloverExperiment",
    "SpilloverReader",
    "SpilloverResult",
    "from_frame",
]

_PROTOCOL = "spillover-fixed-count-v1"


@dataclass(frozen=True, slots=True)
class _ComparisonResult:
    policy: str
    reference: str
    family_size: int
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

    @property
    def multiplicity(self) -> str:
        """How to interpret the interval alongside the other declared comparisons."""
        return (
            f"Bonferroni adjustment for {self.family_size} declared comparison(s). Approximate "
            "95% family coverage requires a world consistent with all their modeled scenarios."
        )


@dataclass(frozen=True, slots=True)
class SpilloverResult:
    """Untreated policy differences in the order registered before assignment."""

    registration_id: str
    comparisons: tuple[_ComparisonResult, ...]


@dataclass(frozen=True, slots=True)
class _ComparisonPlan:
    policy: str
    reference: str
    evidence: PlanningEvidence = field(repr=False)

    @property
    def status(self) -> str:
        return self.evidence.status

    @property
    def reason(self) -> str | None:
        return self.evidence.reason

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
        if self.reason == "simulation_budget_exceeded":
            return (
                "This design exceeds the planning limit. Before assignment, select a smaller "
                "set of decision-relevant comparisons or report estimates without intervals."
            )
        return (
            "This plan did not establish reliable uncertainty across the modeled scenarios. "
            "Report an estimate only; do not use detection rates to size the experiment."
        )


@dataclass(frozen=True, slots=True)
class SpilloverExperiment:
    """The declared trial, its frozen population identity and preassignment planning evidence."""

    experiment_id: str
    population: SpilloverPopulation = field(repr=False)
    source_recipe: tuple[tuple[str, str], ...]
    expected_registration: InitVar[str | None] = None
    registration_id: str = field(init=False)
    planning: tuple[_ComparisonPlan, ...] = field(init=False)
    protocol_version: str = field(default=_PROTOCOL, init=False)

    def __post_init__(self, expected_registration: str | None) -> None:
        receipt = {
            "protocol": self.protocol_version,
            "experiment_id": self.experiment_id,
            "source": self.source_recipe,
            "design": self.design.declaration(),
            "roster_digest": self.population.roster_digest,
            "groups": tuple(
                (g.group_id, g.stratum, g.size, g.offers) for g in self.population.groups
            ),
        }
        registration = hashlib.sha256(canonical_digest_bytes(receipt)).hexdigest()
        if expected_registration is not None and expected_registration != registration:
            refuse_input("spillover.roster.changed", "roster")
        object.__setattr__(self, "registration_id", registration)
        plans = tuple(
            _ComparisonPlan(c.policy, c.reference, evaluate_design(self.population, c))
            for c in self.design.comparisons
        )
        object.__setattr__(self, "planning", plans)

    @property
    def design(self) -> SpilloverDesign:
        return self.population.design


class SpilloverReader:
    """Read a registered spillover trial without owning its dataframes or connection."""

    def __init__(self, experiment: SpilloverExperiment, source: SpilloverSource) -> None:
        self._experiment = experiment
        self._source: SpilloverSource | None = source

    @property
    def experiment(self) -> SpilloverExperiment:
        return self._experiment

    @property
    def closed(self) -> bool:
        return self._source is None

    def close(self) -> None:
        """Release the reader's reference to the inputs; borrowed connections stay open."""
        self._source = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __reduce__(self) -> Never:
        _refuse("kernel.run.reader_unportable", experiment_id=self.experiment.experiment_id)

    def _read(self, experiment: SpilloverExperiment) -> SpilloverResult:
        if self._source is None:
            _refuse("kernel.run.reader_closed", experiment_id=self.experiment.experiment_id)
        if experiment is not self.experiment and experiment != self.experiment:
            raise CodedError(
                "The reader belongs to a different spillover experiment; use the pair "
                "returned by the builder.",
                code="kernel.run.execution_unavailable",
                context={
                    "experiment_id": experiment.experiment_id,
                    "reader_experiment_id": self.experiment.experiment_id,
                    "field": "registration_id",
                },
            )
        cells = observe_spillover(self._source, experiment.population)
        design = experiment.design
        m = len(design.comparisons)
        z = critical_value(m)
        rows = []
        for comparison, plan in zip(design.comparisons, experiment.planning, strict=True):
            point, se, observed_reason = estimate_comparison(
                experiment.population, comparison, cells
            )
            qualified = plan.evidence.calibration_passed
            lower = upper = None
            if qualified and se is not None and observed_reason is None:
                lower, upper = point - z * se, point + z * se
            rows.append(
                _ComparisonResult(
                    policy=comparison.policy,
                    reference=comparison.reference,
                    family_size=m,
                    outcome=design.outcome,
                    horizon=design.horizon,
                    estimate=point,
                    lower=lower,
                    upper=upper,
                    observed_reason=observed_reason,
                    qualification_reason=None if qualified else plan.reason,
                    planning_status=plan.status,
                )
            )
        return SpilloverResult(experiment.registration_id, tuple(rows))


def _unsupported(options: Iterable[str]) -> Never:
    _refuse(
        "kernel.request.invalid", field=f"not supported for spillover: {', '.join(sorted(options))}"
    )


def _bind(
    source: SpilloverSource,
    design: SpilloverDesign,
    *,
    experiment_id: str,
    recipe: tuple[tuple[str, str], ...],
    expected_registration: str | None,
) -> tuple[SpilloverExperiment, SpilloverReader]:
    if not isinstance(design, SpilloverDesign):
        refuse_input("spillover.design.required", "design")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        refuse_input("spillover.design.invalid", "experiment_id")
    if expected_registration is not None and (
        not isinstance(expected_registration, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_registration)
    ):
        refuse_input("spillover.design.invalid", "expected_registration")
    experiment = SpilloverExperiment(
        experiment_id=experiment_id,
        population=freeze_spillover(source, design),
        source_recipe=recipe,
        expected_registration=expected_registration,
    )
    return experiment, SpilloverReader(experiment, source)


def from_frame(
    roster: IntoDataFrame,
    *,
    offer_counts: IntoDataFrame,
    design: SpilloverDesign,
    group_policy: IntoDataFrame | None = None,
    offered: IntoDataFrame | None = None,
    outcomes: IntoDataFrame | None = None,
    experiment_id: str = "frame",
    expected_registration: str | None = None,
    **kwargs: Never,
) -> tuple[SpilloverExperiment, SpilloverReader]:
    """Plan from the frozen roster and quotas; only run reads the assignments and outcomes."""
    from increment._frame_spillover import FrameSpilloverSource

    if kwargs:
        _unsupported(kwargs)
    source = FrameSpilloverSource(
        roster, offer_counts, group_policy=group_policy, offered=offered, outcomes=outcomes
    )
    return _bind(
        source,
        design,
        experiment_id=experiment_id,
        recipe=(("source", "frame"),),
        expected_registration=expected_registration,
    )
