from __future__ import annotations

from collections.abc import Hashable, Sequence
from fractions import Fraction

from increment.estimation.decision_types import DecisionComputation, FixedInference
from increment.estimation.family import FamilyOutcome, select_family
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily


def select_fixed_roster[Key: Hashable](
    cells: Sequence[tuple[Key, object]],
    q: float | Fraction,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | FixedInference | None,
    alpha: float,
    *,
    computation: DecisionComputation,
) -> FamilyOutcome[Key]:
    """Dispatch only the package's existing fixed-roster family procedures."""
    return select_family(cells, q, inference, alpha, computation=computation)
