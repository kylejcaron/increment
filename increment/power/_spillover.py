"""Fixed prospective sensitivity checks for two-stage binary-outcome designs.

With N eligible people, model round-half-up(N * rate) reference events and
round-half-up(N * abs(effect)) changes. Only unequal policy quotas permit a
change. Condition on each stratum's exact group allocation, then draw untreated
events without replacement within each selected group. For m frozen contrasts
count misses against the Normal interval at two-sided error .05/m, including
draws with no interval. Probability bounds use one-sided error .05/(24*m)
across the declared scenarios and contrasts, not across successful fits.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from typing import Literal

import numpy as np
from scipy.special import betainccinv, betaincinv
from scipy.stats import binom

from increment._spillover_design import SpilloverComparison, SpilloverPopulation
from increment.estimation.spillover import (
    BatchCell,
    SpilloverBatch,
    critical_value,
    summarize_cells,
    summarize_counts,
)

_ATTEMPT_UNIT = 65_536
_PROFILES = ("spread", "largest_first", "smallest_first")
_SCHEDULES = ("unchanged", "cancelling", "spread", "concentrated")
_BLOCK_CELLS = 262_144
_NUMPY_URN_LIMIT = 1_000_000_000  # Exclusive for each color, not for total group size.
_MAX_DIRECT_DRAWS = 64

_Counts = tuple[int, ...]
_Status = Literal["supported", "underpowered", "inconclusive", "not_calibrated"]


@dataclass(frozen=True, slots=True)
class ScenarioEvidence:
    """Assignment counts and simultaneous probability bounds for one modeled schedule."""

    baseline_profile: str
    schedule: str
    true_effect: float
    cancelling_pairs: int | None
    attempts: int
    misses: int
    unavailable: int
    false_calls: int | None
    correct_detections: int | None
    wrong_direction_detections: int | None
    miss_upper: float
    detection_lower: float | None
    detection_upper: float | None

    @property
    def name(self) -> str:
        return f"{self.baseline_profile}/{self.schedule}"


@dataclass(frozen=True, slots=True)
class PlanningEvidence:
    """Model-conditional calibration and ability to detect the prespecified change."""

    scenarios: tuple[ScenarioEvidence, ...]
    calibration_passed: bool
    status: _Status
    reason: str | None


@dataclass(frozen=True, slots=True)
class _Scenario:
    baseline_profile: str
    schedule: str
    low: _Counts
    high: _Counts
    true_effect: float
    cancelling_pairs: int | None


def _nearest(rate: float, people: int) -> int:
    # Half-person ties in the supplied decimal rate round up, not toward even.
    p = Fraction(str(rate)) * people
    return (2 * p.numerator + p.denominator) // (2 * p.denominator)


def _spread(capacities: _Counts, count: int) -> _Counts:
    total = sum(capacities)
    if not count:
        return (0,) * len(capacities)
    shares, remainders = zip(*(divmod(c * count, total) for c in capacities), strict=True)
    result = list(shares)
    order = sorted(range(len(result)), key=remainders.__getitem__, reverse=True)
    for i in order[: count - sum(shares)]:
        result[i] += 1
    return tuple(result)


def _fill(capacities: _Counts, count: int, *, smallest_first: bool = False) -> _Counts:
    result = [0] * len(capacities)
    for i in sorted(range(len(result)), key=capacities.__getitem__, reverse=not smallest_first):
        if not count:
            break
        result[i] = min(capacities[i], count)
        count -= result[i]
    return tuple(result)


def _schedules(
    profile: str, sizes: _Counts, low: _Counts, delta: int, movable: tuple[bool, ...]
) -> tuple[_Scenario, ...]:
    groups = range(len(low))
    order = sorted((i for i in groups if movable[i]), key=lambda i: Fraction(low[i], sizes[i]))
    lower = set(order[: len(order) // 2])
    gain_cap = tuple(sizes[i] - low[i] if i in lower else 0 for i in groups)
    loss_cap = tuple(low[i] if movable[i] and i not in lower else 0 for i in groups)
    pairs = min(sum(gain_cap), sum(loss_cap))
    if not pairs:
        return ()
    gains, losses = _fill(gain_cap, pairs), _fill(loss_cap, pairs)
    cancelling = tuple(low[i] + gains[i] - losses[i] for i in groups)
    eligible = _capacity(sizes, low, movable, delta)
    sign = 1 if delta > 0 else -1
    result = [
        _Scenario(profile, "unchanged", low, low, 0.0, None),
        _Scenario(profile, "cancelling", low, cancelling, 0.0, pairs),
    ]
    truth = delta / sum(sizes)
    for name, allocate in (("spread", _spread), ("concentrated", _fill)):
        flips = allocate(eligible, abs(delta))
        high = tuple(low[i] + sign * flips[i] for i in groups)
        result.append(_Scenario(profile, name, low, high, truth, None))
    return tuple(result)


def _capacity(sizes: _Counts, low: _Counts, movable: tuple[bool, ...], delta: int) -> _Counts:
    return tuple(
        (size - events if delta > 0 else events) if allowed else 0
        for size, events, allowed in zip(sizes, low, movable, strict=True)
    )


def _model_grid(
    sizes: _Counts, rate: float, effect: float, movable: tuple[bool, ...]
) -> tuple[_Scenario, ...] | str:
    people = sum(sizes)
    events, change = _nearest(rate, people), _nearest(abs(effect), people)
    if not change:
        return "effect_rounds_to_zero"
    baselines = (
        _spread(sizes, events),
        _fill(sizes, events),
        _fill(sizes, events, smallest_first=True),
    )
    delta = change if effect > 0 else -change
    if any(change > sum(_capacity(sizes, low, movable, delta)) for low in baselines):
        return "effect_exceeds_capacity"
    cases = []
    for profile, low in zip(_PROFILES, baselines, strict=True):
        schedules = _schedules(profile, sizes, low, delta, movable)
        if not schedules:
            return "degenerate_cancellation"
        cases.extend(schedules)
    return tuple(cases)


def _large_urn_draw(
    rng: np.random.Generator, good: int, bad: int, draws: int, rows: int
) -> np.ndarray:
    # The intersection of a random subset and a fixed subset is symmetric in
    # their sizes. Draw the smaller minority subset, then undo the complements.
    size = good + bad
    colors, sample = min(good, bad), min(draws, size - draws)
    steps, targets = sorted((colors, sample))
    hits = np.zeros(rows, dtype=np.int64)
    for step in range(steps):
        hits += rng.integers(size - step, size=rows) < targets - hits
    in_sample = hits if good <= bad else sample - hits
    return in_sample if draws <= size - draws else good - in_sample


class _Population:
    """Separate fixed from stochastic columns of a policy's untreated event counts."""

    def __init__(self, sizes: np.ndarray, events: _Counts, draws: np.ndarray) -> None:
        good = np.asarray(events)
        bad = sizes - good
        self.fixed = np.where(draws == sizes, good, np.where(bad == 0, draws, 0))
        self.variable = np.flatnonzero((good > 0) & (bad > 0) & (draws < sizes))
        self.good, self.bad, self.draws = (a[self.variable] for a in (good, bad, draws))
        oversized = (self.good >= _NUMPY_URN_LIMIT) | (self.bad >= _NUMPY_URN_LIMIT)
        self.large = [
            (i, int(self.good[i]), int(self.bad[i]), int(self.draws[i]))
            for i in np.flatnonzero(oversized)
        ]
        self.ordinary = np.flatnonzero(~oversized) if self.large else None
        self.lookup: np.ndarray | None = None
        if not self.variable.size:
            return
        # For a repeated small urn, one uniform subset rank samples the exact
        # hypergeometric law. Never build a subset table for a large urn.
        if all(np.all(a == a[0]) for a in (self.good, self.bad, self.draws)):
            g, b, m = int(self.good[0]), int(self.bad[0]), int(self.draws[0])
            n = g + b
            if n <= 64 and math.comb(n, m) <= 4096:
                support = range(max(0, m - b), min(g, m) + 1)
                masses = [math.comb(g, k) * math.comb(b, m - k) for k in support]
                self.lookup = np.repeat(np.asarray(support), masses)

    def sample(self, rng: np.random.Generator, rows: int) -> np.ndarray:
        if not self.variable.size:
            return self.fixed  # broadcasts over assignments
        shape = (rows, len(self.variable))
        if self.large:
            random = np.empty(shape, dtype=np.int64)
            cols = self.ordinary
            if cols is not None and cols.size:
                random[:, cols] = rng.hypergeometric(
                    self.good[cols], self.bad[cols], self.draws[cols], size=(rows, len(cols))
                )
            for i, good, bad, draws in self.large:
                random[:, i] = _large_urn_draw(rng, good, bad, draws, rows)
        elif self.lookup is not None:
            random = self.lookup[rng.integers(len(self.lookup), size=shape)]
        else:
            random = rng.hypergeometric(self.good, self.bad, self.draws, size=shape)
        if len(self.variable) == len(self.fixed):
            return random
        result = np.broadcast_to(self.fixed, (rows, len(self.fixed))).copy()
        result[:, self.variable] = random
        return result


def _stage_one(
    population: SpilloverPopulation, comparison: SpilloverComparison
) -> list[tuple[slice, np.ndarray]]:
    strata = []
    start = 0
    for name, groups in groupby(population.groups, key=lambda g: g.stratum):
        n = sum(1 for _ in groups)
        quotas = population.design.groups_per_policy[name]
        r, p = quotas[comparison.reference], quotas[comparison.policy]
        labels = np.repeat(np.arange(3, dtype=np.uint8), (r, p, n - r - p))
        strata.append((slice(start, start + n), labels))
        start += n
    return strata


def _states(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    models: Sequence[_Scenario],
    attempts: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    r, p = _indices(population, comparison)
    sizes = np.array([g.size for g in population.groups])
    low_draws = sizes - np.array([g.offers[r] for g in population.groups])
    high_draws = sizes - np.array([g.offers[p] for g in population.groups])
    lows = {
        counts: _Population(sizes, counts, low_draws)
        for counts in dict.fromkeys(m.low for m in models)
    }
    highs = {
        counts: _Population(sizes, counts, high_draws)
        for counts in dict.fromkeys(m.high for m in models)
    }
    laws = [(lows[m.low], highs[m.high]) for m in models]
    uses = Counter(law for pair in laws for law in pair)
    strata = _stage_one(population, comparison)
    rng = np.random.Generator(np.random.PCG64(9501))
    block = min(4096, _BLOCK_CELLS // len(sizes))
    for start in range(0, attempts, block):
        rows = min(block, attempts - start)
        assignment = np.empty((rows, len(sizes)), dtype=np.uint8)
        for cols, allocation in strata:
            target = assignment[:, cols]
            rng.permuted(np.broadcast_to(allocation, target.shape), axis=1, out=target)
        reference, policy = assignment == 0, assignment == 1
        shared: dict[_Population, np.ndarray] = {}
        for i, (low_law, high_law) in enumerate(laws):
            for law in (low_law, high_law):
                if uses[law] > 1 and law not in shared:
                    shared[law] = law.sample(rng, rows)
            low = shared[low_law] if low_law in shared else low_law.sample(rng, rows)
            upper = shared[high_law] if high_law in shared else high_law.sample(rng, rows)
            yield i, assignment, np.where(reference, low, np.where(policy, upper, 0))


@dataclass(frozen=True, slots=True)
class _Urn:
    size: int
    draws: int
    events: int

    @property
    def support(self) -> range:
        return range(max(0, self.events + self.draws - self.size), min(self.events, self.draws) + 1)

    def bins(self) -> tuple[tuple[int, float], ...]:
        # For independent groups of this type, the histogram of untreated
        # events is multinomial with hypergeometric bin masses. Adjacent-mass
        # ratios avoid huge binomial coefficients when the support is narrow.
        support = self.support
        mass = Fraction(1)
        weights = [(support.start, mass)]
        for y in support[1:]:
            mass *= Fraction(
                (self.events - y + 1) * (self.draws - y + 1),
                y * (self.size - self.events - self.draws + y),
            )
            weights.append((y, mass))
        # Sequential binomial splits. Put the largest bin last so small tails
        # are sampled directly rather than by rounding a probability to one.
        weights.sort(key=lambda pair: pair[1])
        remaining = sum((w for _, w in weights), Fraction(0))
        bins = []
        for y, weight in weights:
            bins.append((y, float(weight / remaining)))
            remaining -= weight
        return tuple(bins)


_Classes = dict[str, Counter[tuple[_Urn, _Urn]]]


def _classes(
    population: SpilloverPopulation, comparison: SpilloverComparison, model: _Scenario
) -> _Classes:
    r, p = _indices(population, comparison)
    classes: _Classes = {}
    for g, low, high in zip(population.groups, model.low, model.high, strict=True):
        pair = (_Urn(g.size, g.size - g.offers[r], low), _Urn(g.size, g.size - g.offers[p], high))
        if g.stratum not in classes:
            classes[g.stratum] = Counter()
        classes[g.stratum][pair] += 1
    return classes


def _compact(classes: _Classes, groups: int) -> bool:
    # Either sampler draws from the same law. Tabulate a narrow support only
    # when the bins cost substantially less than processing individual groups.
    bins = 0
    for types in classes.values():
        for pair in types:
            for urn in pair:
                width = len(urn.support)
                if width > 64:
                    return False
                bins += width
    return bins * 4 <= groups


def _partition(
    rng: np.random.Generator,
    classes: Counter[tuple[_Urn, _Urn]],
    reference_quota: int,
    policy_quota: int,
    rows: int,
) -> Iterator[tuple[tuple[_Urn, _Urn], np.ndarray, np.ndarray]]:
    # A uniform fixed-count partition, marginalized over group identities
    # within each type and over the labels of all nuisance policies.
    remaining = classes.total()
    reference = np.full(rows, reference_quota)
    policy = np.full(rows, policy_quota)
    for pair, n in classes.items():
        if n == remaining:
            yield pair, reference, policy
            break
        r = rng.hypergeometric(reference, remaining - reference, n)
        p = rng.hypergeometric(policy, remaining - reference - policy, n - r)
        yield pair, r, p
        reference -= r
        policy -= p
        remaining -= n


class _WeightedTotals:
    """Merge identical group totals with their multiplicities, centering before squaring."""

    def __init__(self, rows: int) -> None:
        self.count = np.zeros(rows, dtype=np.int64)
        self.origin = np.zeros(rows)
        self.offset = np.zeros(rows)
        self.ss = np.zeros(rows)
        self.events = np.zeros(rows, dtype=np.int64)
        self.unoffered = np.zeros(rows, dtype=np.int64)

    def sample(
        self,
        rng: np.random.Generator,
        count: np.ndarray,
        urn: _Urn,
        bins: tuple[tuple[int, float], ...],
    ) -> None:
        self.unoffered += count * urn.draws
        left = count
        for y, probability in bins:
            if probability == 1.0:
                freq = left
            else:
                freq = rng.binomial(left, probability)
                left = left - freq
            self.events += freq * y
            total = self.count + freq
            fraction = freq / np.maximum(total, 1)
            value = float(urn.size) * y / urn.draws
            np.copyto(self.origin, value, where=(self.count == 0) & (freq != 0))
            difference = (value - self.origin) - self.offset
            self.ss += difference**2 * (self.count * fraction)
            self.offset += difference * fraction
            self.count = total

    def cell(self) -> BatchCell:
        return BatchCell(
            mean_total=self.origin + self.offset,
            ss=self.ss,
            events=self.events,
            unoffered=self.unoffered,
        )


def _frequency_batches(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    classes: _Classes,
    attempts: int,
) -> Iterator[SpilloverBatch]:
    rng = np.random.Generator(np.random.PCG64(9501))
    urns = {urn for types in classes.values() for pair in types for urn in pair}
    bins = {urn: urn.bins() for urn in urns}
    block = min(4096, _BLOCK_CELLS // (2 * len(classes)))
    for start in range(0, attempts, block):
        rows = min(block, attempts - start)
        cells = {}
        for s, types in classes.items():
            q = population.design.groups_per_policy[s]
            ref, pol = _WeightedTotals(rows), _WeightedTotals(rows)
            partition = _partition(rng, types, q[comparison.reference], q[comparison.policy], rows)
            for (ru, pu), r, p in partition:
                ref.sample(rng, r, ru, bins[ru])
                pol.sample(rng, p, pu, bins[pu])
            cells[s, comparison.reference] = ref.cell()
            cells[s, comparison.policy] = pol.cell()
        yield summarize_cells(population, comparison, cells)


def _lower(hits: int, attempts: int, tail: float) -> float:
    return float(betaincinv(hits, attempts - hits + 1, tail)) if hits else 0.0


def _upper(hits: int, attempts: int, tail: float) -> float:
    return float(betainccinv(hits + 1, attempts - hits, tail)) if hits < attempts else 1.0


@dataclass(slots=True)
class _Tally:
    attempts: int = 0
    misses: int = 0
    unavailable: int = 0
    positive: int = 0
    negative: int = 0

    def observe(
        self, point: np.ndarray, se: np.ndarray, available: np.ndarray, truth: float, z: float
    ) -> None:
        radius = z * se
        lo, hi = point - radius, point + radius
        self.attempts += len(point)
        self.unavailable += len(point) - int(np.count_nonzero(available))
        self.misses += len(point) - int(np.count_nonzero(available & (lo <= truth) & (truth <= hi)))
        self.positive += int(np.count_nonzero(available & (lo > 0)))
        self.negative += int(np.count_nonzero(available & (hi < 0)))

    def evidence(self, model: _Scenario, tail: float) -> ScenarioEvidence:
        truth = model.true_effect
        correct = self.positive if truth > 0 else self.negative
        wrong = self.negative if truth > 0 else self.positive
        return ScenarioEvidence(
            baseline_profile=model.baseline_profile,
            schedule=model.schedule,
            true_effect=truth,
            cancelling_pairs=model.cancelling_pairs,
            attempts=self.attempts,
            misses=self.misses,
            unavailable=self.unavailable,
            false_calls=self.positive + self.negative if not truth else None,
            correct_detections=correct if truth else None,
            wrong_direction_detections=wrong if truth else None,
            miss_upper=_upper(self.misses, self.attempts, tail),
            detection_lower=_lower(correct, self.attempts, tail) if truth else None,
            detection_upper=_upper(correct, self.attempts, tail) if truth else None,
        )


def _measure(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    models: tuple[_Scenario, ...],
    attempts: int,
) -> tuple[ScenarioEvidence, ...]:
    m = len(population.design.comparisons)
    tail, z = 0.05 / (24 * m), critical_value(m)
    # Equal potential schedules can share draws across labels. Dependence across
    # schedules does not affect the simultaneous Bonferroni probability bounds.
    unique = {(model.low, model.high): model for model in models}
    tallies = {key: _Tally() for key in unique}
    dense = []
    for key, model in unique.items():
        classes = _classes(population, comparison, model)
        if not _compact(classes, len(population.groups)):
            dense.append(model)
            continue
        for batch in _frequency_batches(population, comparison, classes, attempts):
            tallies[key].observe(
                batch.estimate, batch.standard_error, batch.available, model.true_effect, z
            )
    if dense:
        for i, assignment, counts in _states(population, comparison, dense, attempts):
            model = dense[i]
            batch = summarize_counts(population, comparison, assignment, counts)
            tallies[model.low, model.high].observe(
                batch.estimate, batch.standard_error, batch.available, model.true_effect, z
            )
    return tuple(tallies[m.low, m.high].evidence(m, tail) for m in models)


def _conclude(rows: tuple[ScenarioEvidence, ...], family_size: int) -> PlanningEvidence:
    expected = {(p, s) for p in _PROFILES for s in _SCHEDULES}
    if len(rows) != len(expected) or {(s.baseline_profile, s.schedule) for s in rows} != expected:
        return PlanningEvidence(rows, False, "not_calibrated", "incomplete_bounds")
    if any(not (0 <= s.miss_upper <= 0.055 / family_size) for s in rows):
        return PlanningEvidence(rows, False, "not_calibrated", "miss_bound_exceeded")
    adequate, underpowered = True, False
    for s in rows:
        if s.schedule in ("unchanged", "cancelling"):
            continue
        lo, hi = s.detection_lower, s.detection_upper
        if lo is None or hi is None or not 0 <= lo <= hi <= 1:
            return PlanningEvidence(rows, False, "not_calibrated", "incomplete_bounds")
        adequate &= lo >= 0.8
        underpowered |= hi < 0.8
    if adequate:
        return PlanningEvidence(rows, True, "supported", None)
    if underpowered:
        return PlanningEvidence(rows, True, "underpowered", "detection_below_target")
    return PlanningEvidence(rows, True, "inconclusive", "detection_bounds_overlap_target")


def _attempts(groups: int, family_size: int) -> int | None:
    if family_size == 1:
        return _ATTEMPT_UNIT if groups <= 2048 else None
    limit = min(2**29 // groups, 2**31 // (groups * family_size))
    tail, cap = 0.05 / (24 * family_size), 0.055 / family_size
    for b in range(_ATTEMPT_UNIT, limit + 1, _ATTEMPT_UNIT):
        # U_CP(k) <= cap iff Pr_cap[X <= k] <= tail. The discrete quantile
        # brackets k and the forward CDF resolves equality at its boundary.
        k = int(binom.ppf(tail, b, cap))
        k -= int(binom.cdf(k, b, cap) > tail)
        if binom.logsf(k, b, 0.05 / family_size) <= math.log(0.05 / (12 * family_size)):
            return b
    return None


def _indices(population: SpilloverPopulation, comparison: SpilloverComparison) -> tuple[int, int]:
    policies = population.design.policies
    return policies.index(comparison.reference), policies.index(comparison.policy)


def evaluate_design(
    population: SpilloverPopulation, comparison: SpilloverComparison
) -> PlanningEvidence:
    """Check calibration and directional detection before the trial is assigned."""
    if comparison.baseline_rate is None:
        return PlanningEvidence((), False, "not_calibrated", "missing_baseline_rate")
    family_size = len(population.design.comparisons)
    attempts = _attempts(len(population.groups), family_size)
    if attempts is None:
        return PlanningEvidence((), False, "not_calibrated", "simulation_budget_exceeded")
    if any(
        min(q[comparison.policy], q[comparison.reference]) < 2
        for q in population.design.groups_per_policy.values()
    ):
        return PlanningEvidence((), False, "not_calibrated", "too_few_groups")
    r, p = _indices(population, comparison)
    sizes = tuple(g.size for g in population.groups)
    movable = tuple(g.offers[r] != g.offers[p] for g in population.groups)
    grid = _model_grid(sizes, comparison.baseline_rate, comparison.meaningful_effect, movable)
    if isinstance(grid, str):
        return PlanningEvidence((), False, "not_calibrated", grid)
    for i, group in enumerate(population.groups):
        if group.size <= _NUMPY_URN_LIMIT:
            continue
        for policy, events in ((r, (s.low[i] for s in grid)), (p, (s.high[i] for s in grid))):
            quota = group.offers[policy]
            if min(quota, group.size - quota) > _MAX_DIRECT_DRAWS and any(
                min(k, group.size - k) > _MAX_DIRECT_DRAWS
                and max(k, group.size - k) >= _NUMPY_URN_LIMIT
                for k in events
            ):
                return PlanningEvidence(
                    (), False, "not_calibrated", "group_draw_exceeds_planning_limit"
                )
    return _conclude(_measure(population, comparison, grid, attempts), family_size)
