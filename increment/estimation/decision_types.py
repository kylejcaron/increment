"""Decision-evidence value types with no `AnalysisPlan`/`MomentSource` dependency.

Leaf module under `increment.decision`: value types and hypothesis-key
construction have no import of `decision`, `plan`, `sources`, or
`_readout_request` at runtime. Cannot import
`decision.py`'s `_REFUSALS` registry (that would recreate the cycle
this module exists to break), so it mints its own local one below.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from increment._literals import Alternative, PreferredDirection, Role
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.semantics.unit_cycle import UnitCycleReference, UnitCycleVarianceEnvelope

if TYPE_CHECKING:
    from increment.decision import HypothesisKey
    from increment.estimation._sequential_likelihood import LikelihoodCertificate
    from increment.estimation.contrast_results import ContrastResult
    from increment.estimation.results import LiftEstimate
    from increment.estimation.sequential_result import (
        AsymptoticSequentialResult,
        SequentialCheckpoint,
    )
    from increment.semantics.sequential import SequentialCell
    from increment.sequential_state import SequentialSnapshot


def exact_fraction(value: Fraction | float) -> Fraction:
    """Bind a declared alpha/q to the decimal it was typed as.

    ``Fraction(0.05)`` is the float's raw binary value, a hair above the
    exact ``1/20`` -- using it as a selection ceiling is non-conservative.
    """
    return Fraction(str(value)) if isinstance(value, float) else Fraction(value)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "decision.arm_decision.one_sided_alpha": "one-sided alpha must be < 0.5 for alpha-doubled display level",
        "decision.contrast_decision.reference_metric": "procedure and prospective reference must name the same metric",
        "decision.p_value_evidence.finite_unit_interval": "p_value must be finite and in [0, 1], got {p_value!r}",
    },
)
_raise = raiser(_REFUSALS)


class FixedInference(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["fixed"] = "fixed"


class NoFamily(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    kind: Literal["none"] = "none"


class ContrastDecisionProcedure(CodedModel, BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", revalidate_instances="always")

    metric: str = Field(min_length=1)
    role: Role
    alternative: Alternative
    null_abs: float = Field(allow_inf_nan=False)
    alpha: float = Field(gt=0.0, lt=1.0)
    preferred_direction: PreferredDirection | None = None
    reference: UnitCycleReference | None = None
    family: NoFamily = NoFamily()
    inference: FixedInference = FixedInference()

    @model_validator(mode="after")
    def _one_sided_alpha(self) -> ContrastDecisionProcedure:
        if (
            isinstance(self.reference, UnitCycleVarianceEnvelope)
            and self.metric != self.reference.metric
        ):
            _raise("decision.contrast_decision.reference_metric")
        if (
            self.alternative != "two-sided"
            and self.alpha >= 0.5
            and not isinstance(self.reference, UnitCycleVarianceEnvelope)
        ):
            _raise("decision.arm_decision.one_sided_alpha")
        return self


@dataclass(frozen=True, slots=True)
class ArmHypothesisKey:
    metric: str
    group_id: str
    estimand: str


@dataclass(frozen=True, slots=True)
class SegmentHypothesisKey:
    metric: str
    group_id: str
    estimand: str
    dimension: str
    dimension_value: str

    @property
    def segment(self):
        return ((self.dimension, self.dimension_value),)


def sequential_hypothesis_key(cell: SequentialCell) -> ArmHypothesisKey | SegmentHypothesisKey:
    """Construct the decision key for a registered arm or segment cell."""
    from increment.sequential_state import sequential_refuse

    if cell.segment:
        if len(cell.segment) != 1:
            sequential_refuse(
                "route.unsupported", "public breakout keys support one declared dimension"
            )
        return SegmentHypothesisKey(cell.metric, cell.group_id, cell.estimand, *cell.segment[0])
    return ArmHypothesisKey(cell.metric, cell.group_id, cell.estimand)


@dataclass(frozen=True, slots=True)
class ContrastHypothesisKey:
    metric: str
    control_group: str
    treatment_group: str


@dataclass(frozen=True, slots=True)
class PValueEvidence:
    hypothesis: HypothesisKey
    method: str
    p_value: float
    reference: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.p_value) or not (0.0 <= self.p_value <= 1.0):
            _raise("decision.p_value_evidence.finite_unit_interval", p_value=self.p_value)


@dataclass(frozen=True, slots=True)
class EValueEvidence:
    hypothesis: ArmHypothesisKey | SegmentHypothesisKey
    method: str
    # Float first deliberately: the certified zero-likelihood boundary is
    # represented by -inf, which Fraction cannot parse. Finite exact inputs
    # remain Fractions under Pydantic's smart union matching.
    log_e: float | Fraction
    process: str
    checkpoint: SequentialCheckpoint
    certificate: LikelihoodCertificate

    def __post_init__(self) -> None:
        from increment.estimation._sequential_likelihood import LikelihoodCertificate
        from increment.estimation.sequential_result import SequentialCheckpoint
        from increment.sequential_state import sequential_refuse

        if not isinstance(self.checkpoint, SequentialCheckpoint) or not isinstance(
            self.certificate, LikelihoodCertificate
        ):
            sequential_refuse(
                "source.invalid", "evidence requires a typed likelihood checkpoint and certificate"
            )
        if self.checkpoint.model.law != "bernoulli":
            sequential_refuse(
                "route.unsupported",
                "public EValueEvidence is admitted only for Bernoulli observations",
            )
        from increment.estimation.sequential_result import checkpoint_certificate

        if self.certificate != checkpoint_certificate(self.checkpoint):
            sequential_refuse(
                "source.invalid", "likelihood certificate differs from its exact state"
            )
        cp, cert = self.checkpoint, self.certificate
        expected = (
            cert.log_e.lo
            if cert.log_e is not None
            else float("-inf")
            if cert.status == "zero"
            else float("inf")
        )
        if self.log_e != expected or self.process != "raw_likelihood_v1":
            sequential_refuse(
                "source.invalid", "evidence log does not match its numerical certificate"
            )
        if (self.hypothesis.metric, self.hypothesis.group_id, self.hypothesis.estimand) != (
            cp.cell.metric,
            cp.cell.group_id,
            cp.cell.estimand,
        ):
            sequential_refuse("source.invalid", "evidence hypothesis differs from its checkpoint")
        if getattr(self.hypothesis, "segment", ()) != cp.cell.segment:
            sequential_refuse("source.invalid", "evidence segment differs from its checkpoint")

    @property
    def e_value(self) -> float | None:
        """Display projection only; selection always compares certified log evidence."""
        try:
            value = math.exp(self.log_e)
        except OverflowError:
            return None
        return value if math.isfinite(value) else None

    def rejects(self) -> bool:
        """Equality-null evidence at or above 1/alpha, at this cell's own
        registered allocation -- the fixed-Bonferroni companion to
        ``SequentialInferenceResult.rejects()``, used where a roster mixes
        Bernoulli uptake cells with asymptotic scalar-mean cells and each
        cell is judged at its own committed alpha rather than a step-up rule."""
        from increment.estimation._certified import log_interval

        return self.log_e >= (-log_interval(self.checkpoint.cell.alpha)).hi


@dataclass(frozen=True, slots=True)
class AsymptoticSequentialEvidence:
    """Set-exclusion evidence with registered assumptions and actual arm clocks.

    ``rejects`` reads the result's set at its own decision alpha. Family
    selection instead reads ``result.log_e``, the stopped state's evidence,
    which for a ratio law is capped by denominator stability and so is not the
    exact dual of that set.
    """

    hypothesis: HypothesisKey
    method: str
    result: AsymptoticSequentialResult

    def __post_init__(self) -> None:
        from increment.estimation.sequential_result import AsymptoticSequentialResult
        from increment.sequential_state import sequential_refuse

        if not isinstance(self.result, AsymptoticSequentialResult):
            sequential_refuse("source.invalid", "asymptotic evidence requires a replayable result")
        object.__setattr__(self, "result", AsymptoticSequentialResult.model_validate(self.result))
        if self.hypothesis != sequential_hypothesis_key(self.result.checkpoint.cell):
            sequential_refuse("source.invalid", "asymptotic evidence hypothesis changed")

    @property
    def checkpoint(self) -> SequentialCheckpoint:
        return self.result.checkpoint

    def rejects(self) -> bool:
        return self.result.rejects()


TestEvidence = PValueEvidence | EValueEvidence | AsymptoticSequentialEvidence

ResultT = TypeVar("ResultT", bound="LiftEstimate | ContrastResult")


@dataclass(frozen=True, slots=True)
class DecisionFailure:
    hypothesis: HypothesisKey
    code: str
    context: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    def display(self) -> str:
        """Human-facing detail, falling back to the stable reason or code."""
        return str(self.context.get("display", self.context.get("reason", self.code)))


@dataclass(frozen=True, slots=True)
class DecisionComputation(Generic[ResultT]):  # noqa: UP046 -- preserve Generic API across supported Python versions
    results: Sequence[ResultT]
    evidence: Mapping[HypothesisKey, TestEvidence]
    failures: Mapping[HypothesisKey, DecisionFailure]
    sequential_snapshot: SequentialSnapshot | None = None

    def __post_init__(self) -> None:
        results = tuple(self.results)
        evidence = dict(self.evidence)
        for result in results:
            sequential_result = getattr(result, "sequential_result", None)
            checkpoint = getattr(sequential_result, "checkpoint", None)
            if checkpoint is not None:
                from increment.sequential_state import require_public_laws

                require_public_laws((checkpoint.model,), "decision computations")
        failures = dict(self.failures)
        for value in evidence.values():
            if isinstance(value, (EValueEvidence, AsymptoticSequentialEvidence)):
                from increment.sequential_state import sequential_refuse

                if self.sequential_snapshot is None:
                    sequential_refuse(
                        "source.invalid",
                        "sequential computation requires its verified source snapshot",
                    )
                value.checkpoint.verify_snapshot(self.sequential_snapshot)
        overlap = evidence.keys() & failures.keys()
        if overlap:
            raise AssertionError("decision evidence and failures must not overlap")
        for entries in (evidence, failures):
            for key, value in entries.items():
                if key != value.hypothesis:
                    raise AssertionError("decision entry key must match its hypothesis")
        object.__setattr__(self, "results", results)
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        object.__setattr__(self, "failures", MappingProxyType(failures))


DECISION_ARM_DECISION_ONE_SIDED_ALPHA = _REFUSALS["decision.arm_decision.one_sided_alpha"]
