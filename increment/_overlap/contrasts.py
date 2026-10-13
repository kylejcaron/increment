"""Stable estimand labels and their weights over the four joint assignment cells.

A cell is ``(first_treated, second_treated)``. Every contrast is a linear
combination of cell means whose weights sum to zero. Marginal effects weight
the other experiment's arms by its declared treatment probability, so they
average over the experiment's treatment mix rather than the observed mix.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Literal, get_args

Cell = tuple[int, int]
CELLS: tuple[Cell, ...] = ((0, 0), (1, 0), (0, 1), (1, 1))

OverlapContrast = Literal[
    "interaction",
    "joint_launch",
    "first_only_launch",
    "second_only_launch",
    "first_given_second_treatment",
    "second_given_first_treatment",
    "marginal_first",
    "marginal_second",
]
CONTRASTS: tuple[str, ...] = get_args(OverlapContrast)

_DESCRIPTIONS = MappingProxyType(
    {
        "interaction": "both minus each alone plus neither: (11 - 10) - (01 - 00)",
        "joint_launch": "launch both versus launch neither: 11 - 00",
        "first_only_launch": "first launched alone, second held at control: 10 - 00",
        "second_only_launch": "second launched alone, first held at control: 01 - 00",
        "first_given_second_treatment": "first's effect with second launched: 11 - 01",
        "second_given_first_treatment": "second's effect with first launched: 11 - 10",
        "marginal_first": "first's effect averaged over second's declared treatment mix",
        "marginal_second": "second's effect averaged over first's declared treatment mix",
    }
)
CONTRAST_DESCRIPTIONS = _DESCRIPTIONS


def contrast_weights(
    contrast: str, first_probability: float, second_probability: float
) -> dict[Cell, float]:
    """Weights on the cell means ``mu[(first_treated, second_treated)]``."""
    p1, p2 = first_probability, second_probability
    fixed: dict[str, dict[Cell, float]] = {
        "interaction": {(1, 1): 1.0, (1, 0): -1.0, (0, 1): -1.0, (0, 0): 1.0},
        "joint_launch": {(1, 1): 1.0, (0, 0): -1.0},
        "first_only_launch": {(1, 0): 1.0, (0, 0): -1.0},
        "second_only_launch": {(0, 1): 1.0, (0, 0): -1.0},
        "first_given_second_treatment": {(1, 1): 1.0, (0, 1): -1.0},
        "second_given_first_treatment": {(1, 1): 1.0, (1, 0): -1.0},
    }
    if contrast in fixed:
        return fixed[contrast]
    if contrast == "marginal_first":
        return {(1, 1): p2, (0, 1): -p2, (1, 0): 1.0 - p2, (0, 0): -(1.0 - p2)}
    if contrast == "marginal_second":
        return {(1, 1): p1, (1, 0): -p1, (0, 1): 1.0 - p1, (0, 0): -(1.0 - p1)}
    raise KeyError(contrast)


def cell_probabilities(first_probability: float, second_probability: float) -> dict[Cell, float]:
    """Joint assignment probability of each cell under independent randomization."""
    p1, p2 = first_probability, second_probability
    return {
        (0, 0): (1.0 - p1) * (1.0 - p2),
        (1, 0): p1 * (1.0 - p2),
        (0, 1): (1.0 - p1) * p2,
        (1, 1): p1 * p2,
    }
