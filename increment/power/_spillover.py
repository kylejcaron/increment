"""Fixed prospective sensitivity checks for two-stage binary-outcome designs.

With N eligible people, model round-half-up(N * rate) reference events and
round-half-up(N * abs(effect)) changes. Only unequal policy quotas permit a
change. Condition on each stratum's exact group allocation, then draw untreated
events without replacement within each selected group. Count misses and signed
detections using the registered interval. Probability bounds use one-sided
error .05/(24*m) across all scenarios and the m registered comparisons.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import groupby
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.special import betainccinv, betaincinv

from increment._spillover_design import (
    GroupProfile,
    SpilloverComparison,
    SpilloverDesign,
    SpilloverPopulation,
    refuse_input,
)
from increment._spillover_registration import require_registered_spillover
from increment.estimation.spillover import BatchCell, SpilloverInterval

if TYPE_CHECKING:
    from increment._spillover_registration import SpilloverMethod
    from increment.kernel.experiment import Experiment

_ATTEMPTS = 4096
_MAX_GROUP_ATTEMPTS = 2**25
_PROFILES = ("spread", "largest_first", "smallest_first")
_SCHEDULES = ("unchanged", "cancelling", "spread", "concentrated")
_BLOCK_CELLS = 262_144
_NUMPY_URN_LIMIT = 1_000_000_000  # Exclusive for each color, not for total group size.
_MAX_DIRECT_DRAWS = 64

_Counts = tuple[int, ...]
_Status = Literal["supported", "underpowered", "inconclusive", "not_assessed"]


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
    """Model-conditional ability to detect the prespecified change."""

    scenarios: tuple[ScenarioEvidence, ...]
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
    """Sample both stages in the population's column order."""
    r, p = _indices(population, comparison)
    groups = population.groups
    sizes = np.array([g.size for g in groups])
    low_draws = sizes - np.array([g.offers[r] for g in groups])
    high_draws = sizes - np.array([g.offers[p] for g in groups])
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
            counts = np.where(reference, low, np.where(policy, upper, 0))
            yield i, assignment, counts


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


def _classes(groups: tuple[GroupProfile, ...], r: int, p: int, model: _Scenario) -> _Classes:
    classes: _Classes = {}
    for g, low, high in zip(groups, model.low, model.high, strict=True):
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


def _frequency_cells(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    classes: _Classes,
    attempts: int,
) -> Iterator[dict[tuple[str, str], BatchCell]]:
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
        yield cells


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

    def observe(self, lo: np.ndarray, hi: np.ndarray, truth: float) -> None:
        available = np.isfinite(lo) & np.isfinite(hi)
        self.attempts += len(lo)
        self.unavailable += len(lo) - int(np.count_nonzero(available))
        self.misses += len(lo) - int(np.count_nonzero(available & (lo <= truth) & (truth <= hi)))
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
    method: SpilloverMethod,
) -> tuple[ScenarioEvidence, ...]:
    rule = SpilloverInterval(population, method=method)
    tail = 0.05 / (24 * len(population.design.comparisons))
    r, p = _indices(population, comparison)
    # Equal potential schedules can share draws across labels. Dependence across
    # schedules does not affect the simultaneous Bonferroni probability bounds.
    unique = {(model.low, model.high): model for model in models}
    tallies = {key: _Tally() for key in unique}
    dense = []
    for key, model in unique.items():
        classes = _classes(population.groups, r, p, model)
        if not _compact(classes, len(population.groups)):
            dense.append(model)
            continue
        for cells in _frequency_cells(population, comparison, classes, _ATTEMPTS):
            lo, hi = rule.bounds(comparison, cells)
            tallies[key].observe(lo, hi, model.true_effect)
    if dense:
        for i, assignment, counts in _states(population, comparison, dense, _ATTEMPTS):
            model = dense[i]
            lo, hi = rule.bounds_from_counts(comparison, assignment, counts)
            tallies[model.low, model.high].observe(lo, hi, model.true_effect)
    return tuple(tallies[m.low, m.high].evidence(m, tail) for m in models)


def _conclude(rows: tuple[ScenarioEvidence, ...]) -> PlanningEvidence:
    expected = {(p, s) for p in _PROFILES for s in _SCHEDULES}
    if len(rows) != len(expected) or {(s.baseline_profile, s.schedule) for s in rows} != expected:
        return PlanningEvidence(rows, "not_assessed", "incomplete_bounds")
    adequate, underpowered = True, False
    for s in rows:
        if s.schedule in ("unchanged", "cancelling"):
            continue
        lo, hi = s.detection_lower, s.detection_upper
        if lo is None or hi is None or not 0 <= lo <= hi <= 1:
            return PlanningEvidence(rows, "not_assessed", "incomplete_bounds")
        adequate &= lo >= 0.8
        underpowered |= hi < 0.8
    if adequate:
        return PlanningEvidence(rows, "supported", None)
    if underpowered:
        return PlanningEvidence(rows, "underpowered", "detection_below_target")
    return PlanningEvidence(rows, "inconclusive", "detection_bounds_overlap_target")


def _indices(population: SpilloverPopulation, comparison: SpilloverComparison) -> tuple[int, int]:
    policies = population.design.policies
    return policies.index(comparison.reference), policies.index(comparison.policy)


def _planning_population(
    population: SpilloverPopulation, comparison: SpilloverComparison
) -> SpilloverPopulation:
    """Order allocation ties and random draws by the directed comparison's assignment law."""
    design = population.design
    r, p = _indices(population, comparison)

    def profile(group: GroupProfile) -> tuple[int, int, int]:
        return group.size, group.offers[r], group.offers[p]

    strata = []
    for name, members in groupby(population.groups, key=lambda g: g.stratum):
        ordered = sorted(members, key=profile)
        counts = design.groups_per_policy[name]
        reference, policy = counts[comparison.reference], counts[comparison.policy]
        # The assignment law pools all other policies for this comparison.
        allocation = (reference, policy, len(ordered) - reference - policy)
        strata.append((allocation, ordered, counts))
    # Smaller first-stage allocations precede larger ones. Break ties by the
    # ordered group profiles, larger first. Equal signatures are exchangeable.
    strata.sort(key=lambda block: [profile(g) for g in block[1]], reverse=True)
    strata.sort(key=lambda block: block[0])
    # Population columns sort by (stratum label, group ID); use ranks for both
    # so neither simulator nor interval reduction needs a second permutation.
    width = len(str(len(strata) - 1))
    quotas = {}
    groups = []
    for rank, (_, members, counts) in enumerate(strata):
        name = str(rank).zfill(width)
        quotas[name] = counts
        for group in members:
            groups.append(GroupProfile(len(groups), name, group.size, group.offers))
    return replace(
        population, design=replace(design, groups_per_policy=quotas), groups=tuple(groups)
    )


def evaluate_design(
    population: SpilloverPopulation,
    comparison: SpilloverComparison,
    *,
    method: SpilloverMethod = "two-stage-fixed-bet-v3",
) -> PlanningEvidence:
    """Assess directional detection before the trial is assigned."""
    if comparison.baseline_rate is None:
        return PlanningEvidence((), "not_assessed", "missing_baseline_rate")
    work = len(population.groups) * len(population.design.comparisons) * _ATTEMPTS
    if work > _MAX_GROUP_ATTEMPTS:
        return PlanningEvidence((), "not_assessed", "simulation_budget_exceeded")
    population = _planning_population(population, comparison)
    groups = population.groups
    r, p = _indices(population, comparison)
    sizes = tuple(g.size for g in groups)
    movable = tuple(g.offers[r] != g.offers[p] for g in groups)
    grid = _model_grid(sizes, comparison.baseline_rate, comparison.meaningful_effect, movable)
    if isinstance(grid, str):
        return PlanningEvidence((), "not_assessed", grid)
    for i, group in enumerate(groups):
        if group.size <= _NUMPY_URN_LIMIT:
            continue
        for policy, events in ((r, (s.low[i] for s in grid)), (p, (s.high[i] for s in grid))):
            quota = group.offers[policy]
            if min(quota, group.size - quota) > _MAX_DIRECT_DRAWS and any(
                min(k, group.size - k) > _MAX_DIRECT_DRAWS
                and max(k, group.size - k) >= _NUMPY_URN_LIMIT
                for k in events
            ):
                return PlanningEvidence((), "not_assessed", "group_draw_exceeds_planning_limit")
    return _conclude(_measure(population, comparison, grid, method))


@dataclass(frozen=True, slots=True)
class SpilloverPowerResult:
    """Ability to detect the registered change in one policy comparison before launch."""

    policy: str
    reference: str
    evidence: PlanningEvidence = field(repr=False)

    @property
    def status(self) -> str:
        return self.evidence.status

    @property
    def reason(self) -> str | None:
        return self.evidence.reason

    @property
    def message(self) -> str:
        """A pre-launch interpretation for the signed change the analyst chose."""
        if self.status != "not_assessed":
            effect = next(s.true_effect for s in self.evidence.scenarios if s.true_effect != 0)
            change = f"{100 * effect:+.6g} percentage points"
            if self.evidence.status == "supported":
                return (
                    f"All six scenarios modeling a change of {change} met the 80% detection target."
                )
            if self.evidence.status == "underpowered":
                return (
                    f"At least one scenario modeling a change of {change} fell short of 80% detection. "
                    "Not detecting the change would not rule it out."
                )
            return f"The simulations could not establish 80% detection for a change of {change}."
        match self.reason:
            case "missing_baseline_rate":
                why = "Supply a comparable pre-experiment event rate to check power."
            case "simulation_budget_exceeded":
                why = "This design exceeds the planning simulation limit."
            case "group_draw_exceeds_planning_limit":
                why = "A group's simulated draw exceeds the planning limit."
            case "effect_rounds_to_zero":
                why = "On this roster the chosen change rounds to zero people changing outcome."
            case "effect_exceeds_capacity":
                why = (
                    "In a modeled scenario, too few outcomes can change in the requested "
                    "direction among groups where the two policies offer different counts."
                )
            case "degenerate_cancellation":
                why = (
                    "The simulation cannot place offsetting gains and losses in separate "
                    "groups where the two policies offer different counts."
                )
            case _:
                why = "The planning simulations did not establish detection probabilities."
        return f"Power was not assessed. {why} The experiment can still report an effect range."


def spillover_power(experiment: Experiment) -> tuple[SpilloverPowerResult, ...]:
    """Assess the registered policy changes using only pre-launch group sizes and quotas."""
    if experiment.design.get("kind") != "spillover":
        refuse_input("spillover.design.required", "design")
    method = require_registered_spillover(experiment)
    declaration: Mapping[str, Any] = experiment.design
    law = SpilloverDesign(
        declaration["groups_per_policy"],
        tuple(SpilloverComparison(**item) for item in declaration["comparisons"]),
        declaration["outcome"],
        declaration["horizon"],
        alpha=declaration["alpha"],
    )
    context: Mapping[str, Any] = experiment.analysis_context
    population = SpilloverPopulation(
        law,
        tuple(
            GroupProfile(i, stratum, size, offers)
            for i, (stratum, size, offers) in enumerate(context["planning_groups"])
        ),
        context["roster_digest"],
    )
    return tuple(
        SpilloverPowerResult(c.policy, c.reference, evaluate_design(population, c, method=method))
        for c in law.comparisons
    )
