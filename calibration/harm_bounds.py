"""Fraction-harmed identification and simultaneous bands for randomized binary trials."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import fsum
from typing import Literal

from scipy.stats import beta

_ATOMS = ("00", "01", "10", "11")
_TOL = 1e-12

type Interval = tuple[float, float]
type LeafMap = tuple[int, int, int, int]
type Decision = Literal["clear", "rule_out", "inconclusive"]
type WeightMode = Literal["known", "estimated"]
type DiagnosticContext = tuple[tuple[str, str | int], ...]
type _TargetMass = float | Fraction
type AtomWeights = tuple[_TargetMass, _TargetMass, _TargetMass, _TargetMass]


class StudyError(ValueError):
    """A reason and immutable coordinate context for an unavailable result."""

    def __init__(self, reason: str, context: DiagnosticContext) -> None:
        super().__init__(reason, context)
        self._reason = reason
        self._context = context

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def context(self) -> DiagnosticContext:
        return self._context


def _error(reason: str, stage: str, coordinate: str, **details: str | int) -> StudyError:
    return StudyError(reason, (("stage", stage), ("coordinate", coordinate), *details.items()))


@dataclass(frozen=True, slots=True)
class Cell:
    """Arm sizes and factual success counts in one atom or pooled leaf."""

    n0: int
    y0: int
    n1: int
    y1: int


type Counts4 = tuple[Cell, Cell, Cell, Cell]


@dataclass(frozen=True, slots=True)
class Band:
    """Plug-in estimates, sampling ranges for both endpoints, and the enclosing band."""

    plugin: Interval | None
    L_ci: Interval
    U_ci: Interval
    outer: Interval
    reason: str | None
    leaf_map: LeafMap
    leaf_counts: tuple[Cell, ...]
    weight_mode: WeightMode

    @property
    def width(self) -> float:
        return self.outer[1] - self.outer[0]


def _shape(value: object) -> int | str:
    return len(value) if isinstance(value, tuple) else type(value).__name__


def _check_counts(counts: Counts4, stage: str) -> None:
    if not isinstance(counts, tuple) or len(counts) != 4:
        raise _error("invalid_counts", stage, "atoms", count=_shape(counts))
    for atom, cell in zip(_ATOMS, counts, strict=True):
        if not isinstance(cell, Cell):
            raise _error("invalid_counts", stage, atom, count=type(cell).__name__)
        for key, value in (("n0", cell.n0), ("y0", cell.y0), ("n1", cell.n1), ("y1", cell.y1)):
            if type(value) is not int or value < 0:
                raise _error("invalid_counts", stage, atom, field=key, count=str(value))
        for key, n, y in (("y0", cell.n0, cell.y0), ("y1", cell.n1, cell.y1)):
            if y > n:
                raise _error("invalid_counts", stage, atom, field=key, count=y, arm_count=n)


def _pool(cells: Iterable[Cell]) -> Cell:
    n0 = y0 = n1 = y1 = 0
    for cell in cells:
        n0 += cell.n0
        y0 += cell.y0
        n1 += cell.n1
        y1 += cell.y1
    return Cell(n0, y0, n1, y1)


def _signs(cell: Cell) -> tuple[bool, bool]:
    d0, d1 = cell.n0 + 2, cell.n1 + 2
    a, b = (cell.y0 + 1) * d1, (cell.y1 + 1) * d0
    return a > b, a + b > d0 * d1


def learn(train: Counts4, *, omit_b: bool) -> LeafMap:
    """Group atoms by S and strict branch signs fitted only on training counts."""
    _check_counts(train, "learn")
    if omit_b:
        a, b = _pool(train[:2]), _pool(train[2:])
        train = (a, a, b, b)
    tags: dict[tuple[int, bool, bool], int] = {}
    a0, a1, a2, a3 = (
        tags.setdefault((i // 2, *_signs(cell)), len(tags)) for i, cell in enumerate(train)
    )
    return a0, a1, a2, a3


def _leaf_count(leaf_map: LeafMap) -> int:
    if not isinstance(leaf_map, tuple) or len(leaf_map) != 4:
        raise _error("invalid_leaf_map", "summarize", "leaf_map", count=_shape(leaf_map))
    size = 0
    for atom, leaf in zip(_ATOMS, leaf_map, strict=True):
        if type(leaf) is not int or not 0 <= leaf <= size:
            raise _error("invalid_leaf_map", "summarize", atom, leaf=str(leaf), count=size)
        if leaf == size:
            size += 1
    return size


def _check_target(weights: AtomWeights) -> None:
    if not isinstance(weights, tuple) or len(weights) != 4:
        raise _error("invalid_target_mass", "summarize", "atom_weights", count=_shape(weights))
    for atom, weight in zip(_ATOMS, weights, strict=True):
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float, Fraction))
            or not 0 <= weight <= 1 + _TOL
        ):
            raise _error("invalid_target_mass", "summarize", atom, mass=str(weight))
    if all(isinstance(w, (int, Fraction)) for w in weights):
        total = sum(weights)
        valid = total == 1
    else:
        total = fsum(weights)
        valid = abs(total - 1) <= _TOL
    if not valid:
        raise _error("invalid_target_mass", "summarize", "atom_weights", sum=str(total))


@lru_cache(maxsize=16384)
def _cp_interval(k: int, n: int, leaves: int) -> Interval:
    delta = 0.05 / (5 * 6 * leaves)
    lower = 0.0 if k == 0 else float(beta.ppf(delta, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.isf(delta, k + 1, n - k))
    if not 0 <= lower <= upper <= 1:
        raise _error(
            "invalid_cp_interval", "sampling", "binomial", count=k, trials=n, leaves=leaves
        )
    return lower, upper


def _weighted(coefficients: tuple[float, ...], weights: Iterable[float]) -> float:
    # An accepted floating-point target can have a tiny positive simplex residual.
    return min(1.0, fsum(c * w for c, w in zip(coefficients, weights, strict=True)))


def _weight_extreme(
    coefficients: tuple[float, ...], limits: tuple[Interval, ...], *, maximize: bool
) -> float:
    weights = [lo for lo, _ in limits]
    left = 1.0 - fsum(weights)
    if left < -_TOL or fsum(hi for _, hi in limits) < 1.0 - _TOL:
        raise _error("infeasible_weight_box", "projection", "leaf_weights", count=len(weights))
    order = sorted(range(len(limits)), key=coefficients.__getitem__, reverse=maximize)
    for i in order:
        if left <= 0:
            break
        extra = min(left, limits[i][1] - weights[i])
        weights[i] += extra
        left -= extra
    return _weighted(coefficients, weights)


def _hinges(p: float, q: float) -> Interval:
    return max(0.0, fsum((p, q, -1.0))), min(p, q)


def _leaf_limits(cell: Cell, leaves: int) -> tuple[float, float, float, float]:
    p0, p1 = _cp_interval(cell.y0, cell.n0, leaves)
    q0, q1 = _cp_interval(cell.n1 - cell.y1, cell.n1, leaves)
    l0, u0 = _hinges(p0, q0)
    l1, u1 = _hinges(p1, q1)
    return l0, l1, u0, u1


def _plugin(
    counts: tuple[Cell, ...], weights: tuple[float, ...], *, known_mass: bool
) -> Interval | None:
    lower, upper = [], []
    for cell, weight in zip(counts, weights, strict=True):
        if not known_mass or weight > 0:
            if not cell.n0 or not cell.n1:
                return None
            # The sign of p + q - 1 is the sign of the exact success-rate difference.
            lo = max(0, cell.y0 * cell.n1 - cell.y1 * cell.n0) / (cell.n0 * cell.n1)
            hi = min(cell.y0 / cell.n0, (cell.n1 - cell.y1) / cell.n1)
            lower.append(weight * lo)
            upper.append(weight * hi)
    return min(1.0, fsum(lower)), min(1.0, fsum(upper))


def summarize(counts: Counts4, leaf_map: LeafMap, *, atom_weights: AtomWeights | None) -> Band:
    """Project simultaneous binomial ranges over a known target or a weight simplex."""
    _check_counts(counts, "summarize")
    leaves = _leaf_count(leaf_map)
    leaf_counts = tuple(
        _pool(cell for cell, g in zip(counts, leaf_map, strict=True) if g == leaf)
        for leaf in range(leaves)
    )
    membership = tuple(cell.n0 + cell.n1 for cell in leaf_counts)
    n = sum(membership)
    if n == 0:
        raise _error("empty_study", "summarize", "estimation_sample", count=n)

    if atom_weights is None:
        w = tuple(k / n for k in membership)
    else:
        _check_target(atom_weights)
        w = tuple(
            fsum(w_i for w_i, g in zip(atom_weights, leaf_map, strict=True) if g == leaf)
            for leaf in range(leaves)
        )
    if leaves == 1:
        w = (1.0,)

    l0, l1, u0, u1 = zip(*(_leaf_limits(cell, leaves) for cell in leaf_counts), strict=True)
    if atom_weights is None and leaves > 1:
        limits = tuple(_cp_interval(k, n, leaves) for k in membership)
        L_ci = (
            _weight_extreme(l0, limits, maximize=False),
            _weight_extreme(l1, limits, maximize=True),
        )
        U_ci = (
            _weight_extreme(u0, limits, maximize=False),
            _weight_extreme(u1, limits, maximize=True),
        )
    else:
        L_ci = (_weighted(l0, w), _weighted(l1, w))
        U_ci = (_weighted(u0, w), _weighted(u1, w))

    plugin = _plugin(leaf_counts, w, known_mass=atom_weights is not None)
    return Band(
        plugin=plugin,
        L_ci=L_ci,
        U_ci=U_ci,
        outer=(L_ci[0], U_ci[1]),
        reason="empty_arm_leaf" if plugin is None else None,
        leaf_map=leaf_map,
        leaf_counts=leaf_counts,
        weight_mode="estimated" if atom_weights is None else "known",
    )


def decide(outer: Interval, tau: float) -> Decision:
    """Classify a non-disjoint outer band without rounding or breaking ties."""
    if not isinstance(tau, (int, float)) or not 0 < tau < 1:
        raise _error("invalid_tolerance", "decide", "tau", value=str(tau))
    lo, hi = outer
    if tau > hi + _TOL:
        return "clear"
    if lo > tau + _TOL:
        return "rule_out"
    return "inconclusive"
