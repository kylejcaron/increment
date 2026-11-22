"""Person-weighted untreated contrasts for stratified two-stage randomization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from increment._spillover_design import PolicyCell, SpilloverComparison, SpilloverPopulation
from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.estimation._spillover_mixture import PolicyMixture

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
    policy_mean: NDArray[np.float64]
    reference_mean: NDArray[np.float64]
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
        estimate += s.weight * (pair.policy_mean - pair.reference_mean)
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
            p.mean_total,
            r.mean_total,
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


def _observed_batch(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    cells: Sequence[PolicyCell],
) -> dict[tuple[str, str], BatchCell]:
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
    return batched


def estimate_comparison(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    cells: Sequence[PolicyCell],
) -> tuple[float, float | None, str | None]:
    """Estimate policy minus reference; preserve the observed point when SE is unavailable."""
    batched = _observed_batch(population, comparison, cells)
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
        p_mean,
        r_mean,
        p_ss,
        r_ss,
        successes.any(axis=1),
        (successes < limit).any(axis=1),
    )


@dataclass(frozen=True, slots=True)
class _CountDesign:
    strata: tuple[_Stratum, ...]
    sizes: NDArray[np.int64]
    policy_untreated: NDArray[np.int64]
    reference_untreated: NDArray[np.int64]

    @classmethod
    def prepare(
        cls, population: SpilloverPopulation, comparison: SpilloverComparison
    ) -> _CountDesign:
        strata = _strata(population, comparison)
        sizes = np.fromiter(
            (group.size for group in population.groups), np.int64, count=len(population.groups)
        )

        def untreated(policy: str) -> NDArray[np.int64]:
            idx = population.design.policies.index(policy)
            return sizes - np.fromiter(
                (g.offers[idx] for g in population.groups), dtype=np.int64, count=len(sizes)
            )

        return cls(strata, sizes, untreated(comparison.policy), untreated(comparison.reference))

    def rows(self, assignment: np.ndarray, successes: np.ndarray) -> int:
        if (
            not isinstance(assignment, np.ndarray)
            or not isinstance(successes, np.ndarray)
            or assignment.ndim != 2
            or assignment.shape[1] != len(self.sizes)
            or assignment.shape != successes.shape
        ):
            refuse(
                _INVALID_COUNTS,
                requirement="supply aligned assignment-by-group arrays in design order",
            )
        if assignment.dtype.kind not in "iu" or successes.dtype.kind not in "iu":
            refuse(
                _INVALID_COUNTS, requirement="supply integer policy codes and integer event counts"
            )
        return len(assignment)

    def batches(
        self, assignment: np.ndarray, successes: np.ndarray, row_limit: int
    ) -> Iterator[tuple[slice, tuple[_Pair, ...]]]:
        block = min(row_limit, max(1, _WORK_CELLS // len(self.sizes)))
        for start in range(0, len(assignment), block):
            rows = slice(start, start + block)
            yield (
                rows,
                tuple(
                    _row_pair(
                        s,
                        assignment[rows, s.columns],
                        successes[rows, s.columns],
                        self.sizes[s.columns],
                        self.policy_untreated[s.columns],
                        self.reference_untreated[s.columns],
                    )
                    for s in self.strata
                ),
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
    design = _CountDesign.prepare(population, comparison)
    result = _empty_batch(design.rows(assignment, successes))
    for rows, pairs in design.batches(assignment, successes, _WORK_CELLS):
        _finish(
            design.strata,
            pairs,
            result.estimate[rows],
            result.standard_error[rows],
            result.reason[rows],
        )
    return result


class SpilloverInterval:
    """Simultaneous person-weighted untreated contrasts at the frozen family alpha.

    In stratum h, normalize group totals T by its largest group B_h. A policy's
    person rate is C times the weighted mean of X=T/B_h, with weights proportional
    to G_h B_h and C=sum(G_h B_h)/N. For each fixed bet, uniform group sampling
    without replacement bounds the expected product by powers of the stratum mean
    factors (elementary symmetric means). Independent strata and 1+x <= exp(x)
    bound this by one at the true target: the linear terms cancel because n_h eta_h
    is proportional to w_h. The same holds for 1-X. Markov's inequality and a union
    bound over the registered policies and both tails give the family level.
    """

    def __init__(self, population: SpilloverPopulation) -> None:
        self._population = population
        design = population.design
        quotas = design.groups_per_policy
        self._strata = tuple(quotas)
        maxima: dict[str, int] = {}
        for group in population.groups:
            maxima[group.stratum] = max(maxima.get(group.stratum, 0), group.size)
        expanded = [sum(quotas[s].values()) * maxima[s] for s in self._strata]
        total = sum(expanded)
        self._maxima = np.array([maxima[s] for s in self._strata], dtype=np.float64)
        self._scale = total / sum(g.size for g in population.groups)
        self._log_scale = math.log(self._scale)
        self._cap = self._log_scale - math.log(math.ulp(0.0)) + 1
        weights = np.array([e / total for e in expanded])
        policies = design.compared_policies
        level = math.log(2 * len(policies)) - math.log(design.alpha)
        self._laws = {
            p: PolicyMixture(np.array([quotas[s][p] for s in self._strata]), weights, level)
            for p in policies
        }
        self._dense: dict[tuple[str, str], _CountDesign] = {}

    def _block(self, comparison: SpilloverComparison) -> int:
        terms = max(self._laws[p].terms for p in (comparison.policy, comparison.reference))
        return max(1, _WORK_CELLS // (len(self._strata) * terms))

    def _range(
        self,
        policy: str,
        means: Sequence[NDArray[np.float64]],
        squares: Sequence[NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Deduplicate joint stratum summaries, not individual group outcomes.
        # Both tails and all identical draws reuse the same centered moments.
        data = np.column_stack((*means, *squares))
        inverse = None
        if len(data) > 1:
            data, inverse = np.unique(data, axis=0, return_inverse=True)
        mean_total, ss = np.hsplit(data, 2)
        law = self._laws[policy]
        n, b = law.counts, self._maxima
        # Summing bounded totals in a backend may accumulate relative error
        # O(n*eps). Test support before clipping roundoff at the upper face.
        tol = 8 * np.finfo(float).eps * n
        if (
            not np.isfinite(data).all()
            or np.any(mean_total < 0)
            or np.any(mean_total > b * (1 + tol))
            or np.any(ss < 0)
            or np.any(((n == 1) | (mean_total == 0)) & (ss != 0))
        ):
            refuse(_INVALID_COUNTS, requirement="supply finite feasible bounded group moments")
        a = np.minimum(mean_total / b, 1)
        complement = np.maximum(b - mean_total, 0) / b
        variance = (ss / n / b) / b
        # At fixed sum, maximal variance puts floor(sum) observations at one,
        # one at the remainder and the rest at zero. Reflect about 1/2 to
        # retain precision near either face; accommodate rounding of the mean.
        d = np.minimum(a, complement)
        mass = n * d
        k = np.floor(mass)
        max_ss = k * (1 - d) ** 2 + (mass - k - d) ** 2 + (n - k - 1) * d**2
        if np.any(variance > max_ss / n + tol * a):
            refuse(_INVALID_COUNTS, requirement="centered sums must fit the bounded group totals")

        # Model mean and centered-SD errors as at most (n+8)*eps*max(X).
        # Bound max(X) by min(1, a + sqrt(n-1)*SD), allowing for both errors.
        # This is not a bit-exact certificate for every warehouse.
        sd = np.sqrt(variance)
        error = (n + 8) * np.finfo(float).eps
        radius = np.sqrt(n - 1)
        denom = 1 - error * (1 + radius)
        maximum = np.divide(a + radius * sd, denom, out=np.ones_like(a), where=denom > 0)
        delta = error * np.minimum(maximum, 1)
        outer_variance = (sd + delta) ** 2
        # The gain above log(1-eta) is a*g(b), g(b)=log1p(k*b)/b for k>0.
        # Since g'<=0 and g+b*g'>0, it decreases with variance and increases
        # with mean. Lower either tail's mean by the *forward* absolute error.
        low_y = law.negative_log_limit(np.maximum(a - delta, 0), outer_variance, self._cap)
        reflected = np.maximum(complement - delta, 0)
        high_y = law.negative_log_limit(reflected, outer_variance, self._cap)
        lower = np.exp(self._log_scale - low_y)
        upper = self._scale * -np.expm1(-high_y)
        lower = np.clip(np.nextafter(lower, -np.inf), 0, 1)
        upper = np.clip(np.nextafter(upper, np.inf), 0, 1)
        if inverse is None:
            return lower, upper
        return lower[inverse], upper[inverse]

    @staticmethod
    def _subtract(
        policy: tuple[NDArray[np.float64], NDArray[np.float64]],
        reference: tuple[NDArray[np.float64], NDArray[np.float64]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        lower = np.maximum(np.nextafter(policy[0] - reference[1], -np.inf), -1)
        upper = np.minimum(np.nextafter(policy[1] - reference[0], np.inf), 1)
        failed = np.isnan(lower) | np.isnan(upper)
        lower[failed] = np.nan
        upper[failed] = np.nan
        return lower, upper

    def bounds(
        self,
        comparison: SpilloverComparison,
        cells: Sequence[PolicyCell] | Mapping[tuple[str, str], BatchCell],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return directed lower and upper vectors from observed or simulated cells.

        Invalid summaries refuse. A numerical inversion failure gives two NaNs
        for that row, never an imputed physical interval.
        """
        source = (
            cast("Mapping[tuple[str, str], BatchCell]", cells)
            if isinstance(cells, Mapping)
            else _observed_batch(self._population, comparison, cells)
        )
        arms: list[list[BatchCell]] = []
        rows = None
        for policy in (comparison.policy, comparison.reference):
            arm: list[BatchCell] = []
            for s in self._strata:
                key = (s, policy)
                if key not in source:
                    refuse(_INVALID_COUNTS, requirement="supply each compared policy and stratum")
                cell = source[key]
                vectors = (cell.mean_total, cell.ss, cell.events, cell.unoffered)
                if not all(isinstance(v, np.ndarray) and v.ndim == 1 for v in vectors):
                    refuse(_INVALID_COUNTS, requirement="supply aligned policy summary vectors")
                if rows is None:
                    rows = len(cell.mean_total)
                if any(len(v) != rows for v in vectors):
                    refuse(_INVALID_COUNTS, requirement="supply aligned policy summary vectors")
                if (
                    cell.mean_total.dtype.kind not in "fiu"
                    or cell.ss.dtype.kind not in "fiu"
                    or cell.events.dtype.kind not in "iu"
                    or cell.unoffered.dtype.kind not in "iu"
                ):
                    refuse(_INVALID_COUNTS, requirement="supply feasible policy summary counts")
                arm.append(cell)
            arms.append(arm)
        assert rows is not None
        lower, upper = np.empty(rows), np.empty(rows)
        block = self._block(comparison)
        for start in range(0, rows, block):
            ix = slice(start, start + block)
            ranges = []
            for name, arm in zip((comparison.policy, comparison.reference), arms, strict=True):
                for cell, n, b in zip(arm, self._laws[name].counts, self._maxima, strict=True):
                    e, u, mu = cell.events[ix], cell.unoffered[ix], cell.mean_total[ix]
                    eps = 8 * np.finfo(float).eps * n
                    # y <= T=size*y/u <= B*y, and B-T >= u-y for each group.
                    if (
                        np.any(e < 0)
                        or np.any(e > u)
                        or np.any(u < n)
                        or np.any(u > n * b)
                        or np.any((e == 0) != (mu == 0))
                        or np.any(mu < (e / n) * (1 - eps))
                        or np.any(mu > (e / n) * b * (1 + eps))
                        or np.any(mu > b * (1 + eps) - (u - e) / n)
                    ):
                        refuse(_INVALID_COUNTS, requirement="supply feasible policy summary counts")
                ranges.append(
                    self._range(name, [c.mean_total[ix] for c in arm], [c.ss[ix] for c in arm])
                )
            p, r = ranges
            lower[ix], upper[ix] = self._subtract(p, r)
        return lower, upper

    def bounds_from_counts(
        self, comparison: SpilloverComparison, assignments: np.ndarray, outcome_counts: np.ndarray
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Use the same centered group reductions and column codes as summarize_counts."""
        key = (comparison.policy, comparison.reference)
        if key not in self._dense:
            self._dense[key] = _CountDesign.prepare(self._population, comparison)
        design = self._dense[key]
        n = design.rows(assignments, outcome_counts)
        lower, upper = np.empty(n), np.empty(n)
        for rows, pairs in design.batches(assignments, outcome_counts, self._block(comparison)):
            p = self._range(key[0], [x.policy_mean for x in pairs], [x.policy_ss for x in pairs])
            r = self._range(
                key[1], [x.reference_mean for x in pairs], [x.reference_ss for x in pairs]
            )
            lower[rows], upper[rows] = self._subtract(p, r)
        return lower, upper
