"""Person-weighted untreated contrasts for stratified two-stage randomization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri_exp

from increment._spillover_design import PolicyCell, SpilloverComparison, SpilloverPopulation
from increment.errors import InvalidRequestError, RefusalSpec, refuse

_AVAILABLE, _TOO_FEW, _NO_VARIATION, _ZERO_VARIANCE, _NONFINITE = range(5)
_REASON_TEXT = {
    _AVAILABLE: None,
    _TOO_FEW: "too_few_groups",
    _NO_VARIATION: "no_outcome_variation",
    _ZERO_VARIANCE: "zero_variance",
    _NONFINITE: "nonfinite_variance",
}
_INVALID_COUNTS = RefusalSpec(
    "spillover.counts.invalid",
    InvalidRequestError,
    template="Invalid spillover counts: {requirement}.",
)
# Bound working arrays independently of the number of assignments supplied.
_WORK_CELLS = 262_144


def critical_value(family_size: int) -> float:
    """Two-sided 95% Bonferroni critical value for the frozen family."""
    if isinstance(family_size, bool) or not isinstance(family_size, int) or family_size < 1:
        refuse(_INVALID_COUNTS, requirement="family size must be a positive integer")
    return float(-ndtri_exp(math.log(0.025) - math.log(family_size)))


def reason_text(code: int | np.integer) -> str | None:
    """Decode the availability of a single observed untreated contrast."""
    return _REASON_TEXT[int(code)]


@dataclass(frozen=True, slots=True)
class SpilloverBatch:
    """One estimate, nullable standard error and availability code per assignment."""

    estimate: NDArray[np.float64]
    standard_error: NDArray[np.float64]
    reason: NDArray[np.uint8]

    @property
    def available(self) -> NDArray[np.bool_]:
        return self.reason == _AVAILABLE


@dataclass(frozen=True, slots=True)
class BatchCell:
    """Simulated means, centered sums of squares and unoffered counts in one policy cell."""

    mean_total: NDArray[np.float64]
    ss: NDArray[np.float64]
    events: np.ndarray
    unoffered: np.ndarray


@dataclass(frozen=True, slots=True)
class _Stratum:
    name: str
    columns: slice
    n_policy: int
    n_reference: int
    weight: float


def _strata(
    population: SpilloverPopulation, comparison: SpilloverComparison
) -> tuple[_Stratum, ...]:
    people = sum(g.size for g in population.groups)
    start = 0
    strata = []
    # Population columns are in canonical (stratum, group) order.
    for name, quotas in sorted(population.design.groups_per_policy.items()):
        groups = sum(quotas.values())
        strata.append(
            _Stratum(
                name,
                slice(start, start + groups),
                quotas[comparison.policy],
                quotas[comparison.reference],
                groups / people,
            )
        )
        start += groups
    return tuple(strata)


@dataclass(frozen=True, slots=True)
class _Pair:
    difference: NDArray[np.float64]
    policy_ss: NDArray[np.float64]
    reference_ss: NDArray[np.float64]
    has_event: NDArray[np.bool_]
    has_nonevent: NDArray[np.bool_]


def _finish(
    strata: tuple[_Stratum, ...],
    pairs: Iterable[_Pair],
    estimate: NDArray[np.float64],
    se: NDArray[np.float64],
    reason: NDArray[np.uint8],
) -> None:
    # With T_j = n_j*y_j/u_j, each policy mean estimates a group total, not a rate.
    # The stratum expands it to all G_s groups, then divides by full roster size N.
    has_event = np.zeros(len(estimate), dtype=bool)
    has_nonevent = np.zeros_like(has_event)
    scarce = False
    for s, pair in zip(strata, pairs, strict=True):
        estimate += s.weight * pair.difference
        if min(s.n_policy, s.n_reference) < 2:
            scarce = True
            continue
        # SS/[g*(g-1)] is the sample variance of T divided by its cell count.
        se += s.weight**2 / (s.n_policy * (s.n_policy - 1)) * pair.policy_ss
        se += s.weight**2 / (s.n_reference * (s.n_reference - 1)) * pair.reference_ss
        has_event |= pair.has_event
        has_nonevent |= pair.has_nonevent
    if scarce:
        reason.fill(_TOO_FEW)
        se.fill(np.nan)
        return
    reason[~np.isfinite(se)] = _NONFINITE
    reason[se == 0] = _ZERO_VARIANCE
    reason[~(has_event & has_nonevent)] = _NO_VARIATION
    se[reason != _AVAILABLE] = np.nan
    np.sqrt(se, out=se)


def _empty_batch(rows: int) -> SpilloverBatch:
    return SpilloverBatch(np.zeros(rows), np.zeros(rows), np.zeros(rows, dtype=np.uint8))


def _cell_pairs(
    strata: tuple[_Stratum, ...],
    comparison: SpilloverComparison,
    cells: Mapping[tuple[str, str], BatchCell],
) -> Iterable[_Pair]:
    for s in strata:
        p = cells.get((s.name, comparison.policy))
        r = cells.get((s.name, comparison.reference))
        if p is None or r is None:
            refuse(
                _INVALID_COUNTS, requirement="supply one cell for each compared policy and stratum"
            )
        yield _Pair(
            p.mean_total - r.mean_total,
            p.ss,
            r.ss,
            (p.events > 0) | (r.events > 0),
            (p.events < p.unoffered) | (r.events < r.unoffered),
        )


def summarize_cells(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    cells: Mapping[tuple[str, str], BatchCell],
) -> SpilloverBatch:
    """Finish pre-reduced simulations; cell fields are vectors over the same assignments."""
    strata = _strata(population, comparison)
    first = cells.get((strata[0].name, comparison.policy))
    if first is None:
        refuse(_INVALID_COUNTS, requirement="supply one cell for each compared policy and stratum")
    result = _empty_batch(len(first.mean_total))
    pairs = _cell_pairs(strata, comparison, cells)
    _finish(strata, pairs, result.estimate, result.standard_error, result.reason)
    return result


def estimate_comparison(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    cells: Sequence[PolicyCell],
) -> tuple[float, float | None, str | None]:
    """Estimate policy minus reference; preserve the observed point when SE is unavailable."""
    batched = {}
    quotas = population.design.groups_per_policy
    for cell in cells:
        if cell.policy not in (comparison.policy, comparison.reference):
            continue
        key = (cell.stratum, cell.policy)
        if (
            key in batched
            or cell.stratum not in quotas
            or quotas[cell.stratum][cell.policy] != cell.groups
        ):
            refuse(
                _INVALID_COUNTS, requirement="each compared cell must match its frozen group count"
            )
        batched[key] = BatchCell(
            np.array([cell.mean_total]),
            np.array([cell.ss]),
            np.array([cell.events]),
            np.array([cell.unoffered]),
        )
    batch = summarize_cells(population, comparison, batched)
    reason = reason_text(batch.reason[0])
    se = float(batch.standard_error[0]) if reason is None else None
    return float(batch.estimate[0]), se, reason


def _moments(
    totals: NDArray[np.float64], mask: NDArray[np.bool_], count: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if np.any(np.count_nonzero(mask, axis=1) != count):
        refuse(_INVALID_COUNTS, requirement="each stratum must match its compared policy quotas")
    values = totals[mask].reshape(len(totals), count)
    # Center on an observed value before averaging. Repeated noninteger totals
    # must have exactly zero spread, not a rounding artifact of the mean.
    origin = values[:, 0].copy()
    values -= origin[:, None]
    offset = values.mean(axis=1)
    origin += offset
    values -= offset[:, None]
    return origin, np.einsum("ij,ij->i", values, values)


def _row_pair(
    s: _Stratum,
    assignment: np.ndarray,
    successes: np.ndarray,
    sizes: NDArray[np.int64],
    policy_untreated: NDArray[np.int64],
    reference_untreated: NDArray[np.int64],
) -> _Pair:
    policy = assignment == 1
    reference = assignment == 0
    other = assignment == 2
    if assignment.min() < 0 or assignment.max() > 2:
        refuse(_INVALID_COUNTS, requirement="use policy codes 0, 1 or 2")
    untreated = np.where(policy, policy_untreated, reference_untreated)
    untreated[other] = 0  # The marginal draw collects no outcomes under nuisance policies.
    # Compare uint64 events without converting adjacent large integers to the same float.
    limit = untreated.view(np.uint64) if successes.dtype.kind == "u" else untreated
    if np.any(successes < 0) or np.any(successes > limit):
        refuse(
            _INVALID_COUNTS,
            requirement="bound events by the unoffered count and supply zero for nuisance groups",
        )
    totals = np.multiply(successes, sizes, dtype=np.float64)
    np.divide(totals, untreated, out=totals, where=~other)
    p_mean, p_ss = _moments(totals, policy, s.n_policy)
    r_mean, r_ss = _moments(totals, reference, s.n_reference)
    return _Pair(
        p_mean - r_mean,
        p_ss,
        r_ss,
        successes.any(axis=1),
        (successes < limit).any(axis=1),
    )


def summarize_counts(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    assignment: np.ndarray,
    successes: np.ndarray,
) -> SpilloverBatch:
    """Estimate a directed contrast from simulated assignments and untreated event counts.

    Parameters
    ----------
    population : SpilloverPopulation
        Frozen group roster and assignment law.
    comparison : SpilloverComparison
        Named policy minus reference.
    assignment : np.ndarray
        Integer batch-by-group array in canonical population order: reference=0, policy=1, other=2.
    successes : np.ndarray
        Integer unoffered event counts in compared groups; zero in other columns.

    Returns
    -------
    SpilloverBatch
        Person-weighted contrasts and their within-policy, between-group uncertainty.

    Notes
    -----
    For group j, T_j = n_j * successes_j / untreated_j estimates its untreated outcome total.
    Stratum s contributes (G_s / N) times the difference of the compared policy means of T_j,
    with all randomized groups in G_s and all eligible people in N. Its variance contribution
    is (G_s / N)^2 times the sum of each cell's sample variance divided by its group count
    (Jiang, Imai & Malani, Biometrics 2023).
    Inference assumes outcomes depend on the number, not the identities, of treated peers.
    """
    g = len(population.groups)
    if (
        not isinstance(assignment, np.ndarray)
        or not isinstance(successes, np.ndarray)
        or assignment.ndim != 2
        or assignment.shape[1] != g
        or assignment.shape != successes.shape
    ):
        refuse(
            _INVALID_COUNTS, requirement="supply aligned assignment-by-group arrays in design order"
        )
    if assignment.dtype.kind not in "iu" or successes.dtype.kind not in "iu":
        refuse(_INVALID_COUNTS, requirement="supply integer policy codes and integer event counts")
    policy_idx = population.design.policies.index(comparison.policy)
    reference_idx = population.design.policies.index(comparison.reference)
    sizes = np.fromiter((group.size for group in population.groups), np.int64, count=g)
    p_untreated = sizes - np.fromiter(
        (group.offers[policy_idx] for group in population.groups), np.int64, count=g
    )
    r_untreated = sizes - np.fromiter(
        (group.offers[reference_idx] for group in population.groups), np.int64, count=g
    )
    strata = _strata(population, comparison)
    n = len(assignment)
    result = _empty_batch(n)
    batch_rows = max(1, _WORK_CELLS // g)
    for start in range(0, n, batch_rows):
        rows = slice(start, start + batch_rows)
        pairs = (
            _row_pair(
                s,
                assignment[rows, s.columns],
                successes[rows, s.columns],
                sizes[s.columns],
                p_untreated[s.columns],
                r_untreated[s.columns],
            )
            for s in strata
        )
        _finish(
            strata,
            pairs,
            result.estimate[rows],
            result.standard_error[rows],
            result.reason[rows],
        )
    return result
