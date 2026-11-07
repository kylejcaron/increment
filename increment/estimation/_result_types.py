"""Leaf values helpers return for the result classes to assemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from increment.winsor import WinsorConfidenceSet


@dataclass(frozen=True, slots=True)
class WinsorLiftParts:
    """One winsorized contrast's region, cutoff and clipped counts."""

    region: WinsorConfidenceSet
    validation_context: dict[object, object] | None
    cutoff: float
    control_n_upper: int
    treatment_n_upper: int
    control_n: int
    treatment_n: int
