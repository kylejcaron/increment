"""Orthogonal assignment declarations for parallel and switchback studies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from increment.errors import InvalidRequestError, raiser, refusals

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "definition.assignment.reject_bool": "expected an int, got a bool",
        "definition.assignment.carryover_order_type": "carryover_order must be a non-boolean int",
        "definition.assignment.carryover_order_range": "carryover_order={carryover_order} must satisfy 0 <= carryover_order < observation_steps={observation_steps}; an order at or past observation_steps would discard the entire retained window",
        "definition.assignment.sequence_law": "{message}",
    },
)
_raise = raiser(_REFUSALS)
ASSIGNMENT_REJECT_BOOL = _REFUSALS["definition.assignment.reject_bool"]


def _reject_bool(v: object) -> object:
    """``bool`` is an ``int`` subclass in Python; a declared step count
    must never silently accept ``True``/``False`` as ``1``/``0``."""
    if isinstance(v, bool):
        _raise("definition.assignment.reject_bool")
    return v


class _AssignmentModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ParallelAssignment(_AssignmentModel):
    kind: Literal["parallel"] = "parallel"


class IndependentBernoulliOrder(_AssignmentModel):
    """One independent Bernoulli CT/TC order draw per unit-cycle.

    Every unit draws its own order independently of every other unit, which
    is what makes the unit the independent replicate: cross-unit replication
    exists for a unit-level reference distribution.
    """

    scheme: Literal["independent_bernoulli_order"] = "independent_bernoulli_order"
    probability_ct: float = Field(default=0.5, gt=0.0, lt=1.0)
    independence_unit: Literal["unit_cycle"] = "unit_cycle"


class SharedScheduleOrder(_AssignmentModel):
    """One independent Bernoulli CT/TC order draw per two-period block,
    shared by every unit in a fixed, complete unit roster.

    Unlike :class:`IndependentBernoulliOrder`, no unit draws its own order:
    the whole roster follows one realized order per block, so the
    independent replicate is the block, not the unit. A common roster
    duplicated uniformly adds no independent blocks and therefore cannot
    change the mean-unit retained-window effect scale or inferential sample
    size. The switchback panel path reduces the roster within each block
    and uses ``switchback_block_t`` with a ``block_t`` reference. At least
    two independent blocks are required; a one-unit roster is supported.
    """

    scheme: Literal["shared_schedule"] = "shared_schedule"
    probability_ct: float = Field(gt=0.0, lt=1.0)
    independence_unit: Literal["shared_block"] = "shared_block"


_SEQUENCE_ARMS = frozenset("CT")
# Declared probabilities are floats; this absorbs only representation error.
_PROBABILITY_SUM_TOLERANCE = 1e-12


def _sequence_law_error(message: str, **context: object) -> None:
    _raise("definition.assignment.sequence_law", message=message, **context)


class DeclaredSequenceOrder(_AssignmentModel):
    """A predeclared finite law over K-period treatment sequences.

    ``sequences`` maps each possible cycle sequence to its declared draw
    probability. A sequence is a string over ``"C"`` (control) and ``"T"``
    (treatment), one character per period, so ``{"CTT": 0.5, "TTC": 0.5}``
    declares three-period cycles with two treated periods in every cycle and a
    randomized position for the control period. Every period must be able to
    receive both arms (``0 < P(period k treated) < 1``); a law without that
    positivity, including any single deterministic sequence such as fixed
    alternation, is not a randomized switchback and is refused.

    ``independence_unit`` names where draws are independent: ``"unit_cycle"``
    draws one sequence per unit-cycle, so units are the inferential replicates;
    ``"shared_block"`` draws one sequence per cycle for a fixed, complete roster,
    so blocks are the replicates. Unequal treatment exposure is expressed by the
    law itself; the analysis weights each period by its declared marginal
    assignment probability, never by realized counts.
    """

    scheme: Literal["declared_sequences"] = "declared_sequences"
    sequences: tuple[tuple[str, float], ...]
    independence_unit: Literal["unit_cycle", "shared_block"]

    @field_validator("sequences", mode="before")
    @classmethod
    def _canonical_sequences(cls, value: object) -> tuple[tuple[str, float], ...]:
        items: list[tuple[object, object]]
        if isinstance(value, Mapping):
            items = list(value.items())
        elif isinstance(value, (tuple, list)) and all(
            isinstance(item, (tuple, list)) and len(item) == 2 for item in value
        ):
            items = [(item[0], item[1]) for item in value if isinstance(item, (tuple, list))]
        else:
            _sequence_law_error(
                "sequences must map each 'C'/'T' sequence string to its probability",
                reason="sequences_mapping",
            )
        if len(items) < 2:
            _sequence_law_error(
                "a declared sequence law needs at least two sequences with positive "
                "probability; a single sequence is a deterministic schedule, not randomization",
                reason="deterministic_schedule",
                sequences=len(items),
            )
        canonical: dict[str, float] = {}
        for key, probability in items:
            if not isinstance(key, str) or not key or set(key) - _SEQUENCE_ARMS:
                _sequence_law_error(
                    f"sequence {key!r} must be a nonempty string of 'C' and 'T' characters",
                    reason="sequence_alphabet",
                )
            if key in canonical:
                _sequence_law_error(
                    f"sequence {key!r} is declared twice", reason="duplicate_sequence"
                )
            if not isinstance(key, str):
                raise AssertionError  # pragma: no cover - refused above
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                _sequence_law_error(
                    f"probability for sequence {key!r} must be a number",
                    reason="probability_type",
                )
                raise AssertionError  # pragma: no cover - refused above
            probability = float(probability)
            if not math.isfinite(probability) or not 0.0 < probability <= 1.0:
                _sequence_law_error(
                    f"probability for sequence {key!r} must satisfy 0 < p <= 1; omit "
                    "impossible sequences rather than declaring them with probability zero",
                    reason="probability_range",
                )
            canonical[key] = probability
        lengths = {len(key) for key in canonical}
        if len(lengths) != 1:
            _sequence_law_error(
                "every declared sequence must have the same number of periods",
                reason="sequence_lengths",
                lengths=sorted(lengths),
            )
        if lengths.pop() < 2:
            _sequence_law_error(
                "declared sequences need at least two periods per cycle",
                reason="sequence_lengths",
            )
        total = math.fsum(canonical.values())
        if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            _sequence_law_error(
                f"declared sequence probabilities must sum to 1; they sum to {total!r}",
                reason="probability_sum",
                total=total,
            )
        return tuple(sorted(canonical.items()))

    @model_validator(mode="after")
    def _period_positivity(self) -> DeclaredSequenceOrder:
        for period, probability in enumerate(self.exact_treatment_probabilities):
            if not 0 < probability < 1:
                _sequence_law_error(
                    f"period {period} is assigned "
                    f"{'treatment' if probability == 1 else 'control'} under every declared "
                    "sequence; each period must be able to receive both arms",
                    reason="period_positivity",
                    period=period,
                )
        return self

    @field_serializer("sequences")
    def _serialize_sequences(self, value: tuple[tuple[str, float], ...]) -> dict[str, float]:
        return dict(value)

    @property
    def periods(self) -> int:
        """Number of periods in every declared cycle sequence."""
        return len(self.sequences[0][0])

    @property
    def probabilities(self) -> Mapping[str, float]:
        """Declared draw probability per sequence, in sequence order."""
        return dict(self.sequences)

    @property
    def exact_sequence_probabilities(self) -> dict[str, Fraction]:
        """Declared probabilities as exact rationals renormalized to sum to one."""
        exact = {key: Fraction(probability) for key, probability in self.sequences}
        total = sum(exact.values(), Fraction(0))
        return {key: value / total for key, value in exact.items()}

    @property
    def exact_treatment_probabilities(self) -> tuple[Fraction, ...]:
        """Exact marginal probability that each period is treated."""
        exact = self.exact_sequence_probabilities
        return tuple(
            sum((q for key, q in exact.items() if key[period] == "T"), Fraction(0))
            for period in range(self.periods)
        )

    @property
    def treatment_probabilities(self) -> tuple[float, ...]:
        """Marginal probability that each period is treated."""
        return tuple(float(value) for value in self.exact_treatment_probabilities)

    @property
    def expected_treated_share(self) -> float:
        """Expected share of a cycle's periods assigned to treatment."""
        exact = self.exact_treatment_probabilities
        return float(sum(exact, Fraction(0)) / len(exact))


AssignmentSequence = Annotated[
    IndependentBernoulliOrder | SharedScheduleOrder | DeclaredSequenceOrder,
    Field(discriminator="scheme"),
]


class SwitchbackWindow(_AssignmentModel):
    """A washout/observation window with a declared residual-carryover order.

    ``carryover_order`` names additional observation steps, immediately
    after the declared washout, still assumed contaminated by the prior
    period and therefore discarded before the retained window used for
    inference. Retained steps satisfy ``step >= washout_steps +
    carryover_order``; the retained window's length is ``observation_steps -
    carryover_order``. ``carryover_order=0`` is the plain
    no-residual-carryover-after-washout case. Declaring an order is an
    assumption, not something a balance diagnostic can prove true.
    """

    washout_steps: int = Field(ge=0)
    observation_steps: int = Field(ge=1)
    carryover_order: int = 0

    @field_validator("washout_steps", "observation_steps", mode="before")
    @classmethod
    def _validate_steps(cls, v: object) -> object:
        return _reject_bool(v)

    @field_validator("carryover_order", mode="before")
    @classmethod
    def _validate_carryover_order_type(cls, v: object) -> object:
        if isinstance(v, bool):
            _raise("definition.assignment.reject_bool")
        if not isinstance(v, int):
            _raise("definition.assignment.carryover_order_type")
        return v

    @model_validator(mode="after")
    def _validate_carryover_order_range(self) -> SwitchbackWindow:
        # This bound is load-bearing, not cosmetic: it is what makes the
        # retained window (``observation_steps - carryover_order``)
        # nonempty for every constructible window.
        if not (0 <= self.carryover_order < self.observation_steps):
            _raise(
                "definition.assignment.carryover_order_range",
                carryover_order=self.carryover_order,
                observation_steps=self.observation_steps,
            )
        return self


class SwitchbackAssignment(_AssignmentModel):
    """A switchback declaration: cycle sequence law plus washout/retained window.

    Paired CT/TC laws (:class:`IndependentBernoulliOrder`,
    :class:`SharedScheduleOrder`) have exactly two periods per cycle. A
    :class:`DeclaredSequenceOrder` declares its own period count through its
    sequence length; ``periods_per_cycle`` must then match it.
    """

    kind: Literal["switchback"] = "switchback"
    periods_per_cycle: int = 2
    sequence: AssignmentSequence
    window: SwitchbackWindow

    @field_validator("periods_per_cycle", mode="before")
    @classmethod
    def _validate_periods_type(cls, v: object) -> object:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError("periods_per_cycle must be a non-boolean int")
        return v

    @model_validator(mode="after")
    def _consistent_periods(self) -> SwitchbackAssignment:
        expected = self.sequence.periods if isinstance(self.sequence, DeclaredSequenceOrder) else 2
        if self.periods_per_cycle != expected:
            # A plain validation error, as the former ``Literal[2]`` field raised:
            # consuming models (planning baselines, envelopes) keep owning the code.
            raise ValueError(
                f"periods_per_cycle={self.periods_per_cycle} does not match the "
                f"{expected}-period cycles of the declared {self.sequence.scheme!r} law"
            )
        return self

    @classmethod
    def declared(
        cls,
        sequences: Mapping[str, float],
        *,
        independence_unit: Literal["unit_cycle", "shared_block"],
        window: SwitchbackWindow,
    ) -> SwitchbackAssignment:
        """Build a declared multi-period sequence assignment with matching period count."""
        sequence = DeclaredSequenceOrder(
            sequences=sequences,  # type: ignore[arg-type]
            independence_unit=independence_unit,
        )
        return cls(periods_per_cycle=sequence.periods, sequence=sequence, window=window)


Assignment = Annotated[ParallelAssignment | SwitchbackAssignment, Field(discriminator="kind")]


def paired_sequence(
    assignment: SwitchbackAssignment,
) -> IndependentBernoulliOrder | SharedScheduleOrder:
    """The paired CT/TC law of ``assignment``, refusing a declared sequence law.

    Paired-only consumers (unit-cycle envelopes and power, paired planning)
    are guarded before they run; this narrows their type and keeps a coded
    refusal if a declared law ever reaches them.
    """
    sequence = assignment.sequence
    if isinstance(sequence, DeclaredSequenceOrder):
        _sequence_law_error(
            "this operation supports paired CT/TC switchback laws only",
            reason="paired_law_required",
        )
        raise AssertionError  # pragma: no cover - _sequence_law_error never returns
    return sequence


__all__ = [
    "Assignment",
    "DeclaredSequenceOrder",
    "paired_sequence",
    "IndependentBernoulliOrder",
    "ParallelAssignment",
    "SharedScheduleOrder",
    "SwitchbackAssignment",
    "SwitchbackWindow",
]
