"""Deterministic covering arrays for the parity matrix."""

from __future__ import annotations

import itertools
from collections.abc import Sequence


def covering_cells(axes: Sequence[Sequence[str]], strength: int = 3) -> tuple[tuple[str, ...], ...]:
    """Return a deterministic greedy covering array over the supplied axes."""
    if strength < 1 or strength > len(axes):
        raise ValueError("strength must be between one and the number of axes")
    if any(not axis for axis in axes):
        return ()
    candidates = itertools.product(*(tuple(sorted(axis)) for axis in axes))
    return _covering_subset(tuple(sorted(candidates)), strength)


def _covering_subset(
    candidates: Sequence[tuple[str, ...]], strength: int
) -> tuple[tuple[str, ...], ...]:
    """Cover interactions present in a possibly constrained full candidate grid."""
    candidates = tuple(sorted(candidates))
    if not candidates:
        return ()
    axis_count = len(candidates[0])
    if strength < 1 or strength > axis_count:
        raise ValueError("strength must be between one and the number of axes")
    interactions = tuple(itertools.combinations(range(axis_count), strength))
    candidate_interactions = {
        candidate: {
            (indices, tuple(candidate[index] for index in indices)) for indices in interactions
        }
        for candidate in candidates
    }
    uncovered = set().union(*candidate_interactions.values())

    selected: list[tuple[str, ...]] = []
    remaining = list(candidates)
    while uncovered:
        best: tuple[str, ...] | None = None
        best_gain = -1
        for candidate in remaining:
            gain = len(candidate_interactions[candidate] & uncovered)
            if gain > best_gain:
                best = candidate
                best_gain = gain
        if best is None or best_gain == 0:
            raise RuntimeError("the supplied candidates cannot cover all interactions")
        selected.append(best)
        uncovered.difference_update(candidate_interactions[best])
        remaining.remove(best)

    return tuple(selected)
