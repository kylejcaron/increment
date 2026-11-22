"""Plan a two-stage offer experiment and estimate its effect on people not offered treatment."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from math import isfinite
from typing import TYPE_CHECKING, Literal, Never, Self

from increment._canonical import canonical_digest_bytes
from increment._evidence_loaders import SpilloverSource, freeze_spillover, observe_spillover
from increment._spillover_design import (
    SpilloverComparison,
    SpilloverDesign,
    SpilloverPopulation,
    refuse_input,
)
from increment.errors import CodedError
from increment.estimation.spillover import SpilloverInterval, estimate_comparison
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

_PROTOCOL = "spillover-fixed-count-v2"
_METHOD = "two-stage-fixed-bet-v1"


@dataclass(frozen=True, slots=True)
class _ComparisonResult:
    policy: str
    reference: str
    family_size: int
    policy_count: int
    alpha: float
    method: str
    outcome: str
    horizon: str
    estimate: float
    lower: float | None
    upper: float | None
    unavailable_reason: str | None = None
    direction: Literal["increase", "decrease", "inconclusive", "unavailable"] = field(init=False)

    def __post_init__(self) -> None:
        if self.lower is None or self.upper is None:
            direction = "unavailable"
        elif not (isfinite(self.lower) and isfinite(self.upper)):
            # Inversion can fail numerically for one comparison without
            # invalidating its estimate or the other registered comparisons.
            object.__setattr__(self, "lower", None)
            object.__setattr__(self, "upper", None)
            object.__setattr__(self, "unavailable_reason", "nonfinite_interval")
            direction = "unavailable"
        elif self.lower > 0:
            direction = "increase"
        elif self.upper < 0:
            direction = "decrease"
        else:
            direction = "inconclusive"
        object.__setattr__(self, "direction", direction)

    @property
    def multiplicity(self) -> str:
        """How to interpret the interval alongside the other declared comparisons."""
        return (
            f"Under the registered draws, the chance that any of the {self.family_size} ranges "
            f"excludes its true effect is at most {self.alpha} in exact arithmetic. The reported "
            f"endpoints approximate this rule, which splits that budget over both tails of "
            f"the {self.policy_count} compared policies."
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
        if self.status != "not_assessed":
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
        match self.reason:
            case "missing_baseline_rate":
                why = "Supply a comparable pre-experiment event rate to check power."
            case "simulation_budget_exceeded":
                why = "This design exceeds the planning simulation limit."
            case "group_draw_exceeds_planning_limit":
                why = "A group's simulated draw exceeds the planning limit."
            case "effect_rounds_to_zero":
                why = "On this roster the chosen change rounds to zero people changing outcome."
            case "effect_exceeds_capacity":
                why = (
                    "In a modeled scenario, too few outcomes can change in the requested "
                    "direction among groups where the two policies offer different counts."
                )
            case "degenerate_cancellation":
                why = (
                    "The simulation cannot place offsetting gains and losses in separate "
                    "groups where the two policies offer different counts."
                )
            case _:
                why = "The planning simulations did not establish detection probabilities."
        return f"Power was not assessed. {why} The experiment can still report an effect range."


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
    method: str = field(default=_METHOD, init=False)

    def __post_init__(self, expected_registration: str | None) -> None:
        receipt = {
            "protocol": self.protocol_version,
            "method": self.method,
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
        rule = SpilloverInterval(experiment.population)
        policy_count = len(design.compared_policies)
        rows = []
        for comparison in design.comparisons:
            point = estimate_comparison(experiment.population, comparison, cells)[0]
            lower, upper = rule.bounds(comparison, cells)
            rows.append(
                _ComparisonResult(
                    policy=comparison.policy,
                    reference=comparison.reference,
                    family_size=len(design.comparisons),
                    policy_count=policy_count,
                    alpha=design.alpha,
                    method=experiment.method,
                    outcome=design.outcome,
                    horizon=design.horizon,
                    estimate=point,
                    lower=float(lower[0]),
                    upper=float(upper[0]),
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
