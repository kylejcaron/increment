"""Fixed prospective sensitivity checks for a two-stage binary-outcome design."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import groupby
from numbers import Real
from typing import Literal

import numpy as np
from scipy.special import betainccinv, betaincinv

from increment._spillover_design import SpilloverDesign
from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.estimation.spillover import Z95, summarize_counts

_ATTEMPTS = 65_536
_TAIL = 0.05 / 24
_PROFILES = ("spread", "largest_first", "smallest_first")
_SCHEDULES = ("unchanged", "cancelling", "spread", "concentrated")
# Per schedule: at most 2,048 groups at the fixed number of assignments. Limit
# actual array dimensions separately, regardless of how many schedules coincide.
_MAX_GROUP_ATTEMPTS = 134_217_728
_BLOCK_CELLS = 262_144

_INVALID_RATE = RefusalSpec(
    "spillover.planning.baseline_rate",
    InvalidRequestError,
    template="Use a pre-assignment event rate between 0 and 1, or omit it for a point estimate only.",
    keys=frozenset({"baseline_rate"}),
)
_INVALID_EFFECT = RefusalSpec(
    "spillover.planning.meaningful_effect",
    InvalidRequestError,
    template="Give a nonzero change between -1 and 1 in the probability of the outcome.",
    keys=frozenset({"meaningful_effect"}),
)

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


def _schedules(profile: str, sizes: _Counts, low: _Counts, delta: int) -> tuple[_Scenario, ...]:
    groups = range(len(low))
    order = sorted(groups, key=lambda i: Fraction(low[i], sizes[i]))
    lower = set(order[: len(low) // 2])
    gain_cap = tuple(sizes[i] - low[i] if i in lower else 0 for i in groups)
    loss_cap = tuple(0 if i in lower else low[i] for i in groups)
    pairs = min(sum(gain_cap), sum(loss_cap))
    if not pairs:
        return ()
    gains, losses = _fill(gain_cap, pairs), _fill(loss_cap, pairs)
    cancelling = tuple(low[i] + gains[i] - losses[i] for i in groups)
    eligible = tuple(sizes[i] - low[i] for i in groups) if delta > 0 else low
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


def _model_grid(sizes: _Counts, rate: float, effect: float) -> tuple[_Scenario, ...] | str:
    people = sum(sizes)
    events, change = _nearest(rate, people), _nearest(abs(effect), people)
    if not change:
        return "effect_rounds_to_zero"
    if change > (people - events if effect > 0 else events):
        return "effect_exceeds_capacity"
    baselines = (
        _spread(sizes, events),
        _fill(sizes, events),
        _fill(sizes, events, smallest_first=True),
    )
    cases = []
    for profile, low in zip(_PROFILES, baselines, strict=True):
        schedules = _schedules(profile, sizes, low, change if effect > 0 else -change)
        if not schedules:
            return "degenerate_cancellation"
        cases.extend(schedules)
    return tuple(cases)


class _Population:
    """Separate fixed from stochastic columns of a policy's untreated event counts."""

    def __init__(self, sizes: np.ndarray, events: _Counts, draws: np.ndarray) -> None:
        good = np.asarray(events)
        bad = sizes - good
        self.fixed = np.where(bad == 0, draws, 0)
        self.variable = np.flatnonzero((good > 0) & (bad > 0))
        self.good, self.bad, self.draws = (a[self.variable] for a in (good, bad, draws))
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
        if self.lookup is None:
            random = rng.hypergeometric(self.good, self.bad, self.draws, size=shape)
        else:
            random = self.lookup[rng.integers(len(self.lookup), size=shape)]
        if len(self.variable) == len(self.fixed):
            return random
        result = np.broadcast_to(self.fixed, (rows, len(self.fixed))).copy()
        result[:, self.variable] = random
        return result


def _stage_one(design: SpilloverDesign) -> list[tuple[slice, np.ndarray]]:
    strata = []
    start = 0
    for name, groups in groupby(design.groups, key=lambda g: g.stratum):
        n = sum(1 for _ in groups)
        strata.append((slice(start, start + n), np.arange(n) < design.high_groups[name]))
        start += n
    return strata


def _states(
    design: SpilloverDesign, models: Sequence[_Scenario], attempts: int
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    sizes = np.array([len(g.members) for g in design.groups])
    low_draws = sizes - np.array([g.low_treated for g in design.groups])
    high_draws = sizes - np.array([g.high_treated for g in design.groups])
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
    strata = _stage_one(design)
    rng = np.random.Generator(np.random.PCG64(9501))
    block = min(4096, _BLOCK_CELLS // len(sizes))
    for start in range(0, attempts, block):
        rows = min(block, attempts - start)
        high = np.empty((rows, len(sizes)), dtype=bool)
        for cols, allocation in strata:
            target = high[:, cols]
            rng.permuted(np.broadcast_to(allocation, target.shape), axis=1, out=target)
        shared: dict[_Population, np.ndarray] = {}
        for i, (low_law, high_law) in enumerate(laws):
            for law in (low_law, high_law):
                if uses[law] > 1 and law not in shared:
                    shared[law] = law.sample(rng, rows)
            low = shared[low_law] if low_law in shared else low_law.sample(rng, rows)
            upper = shared[high_law] if high_law in shared else high_law.sample(rng, rows)
            yield i, high, np.where(high, upper, low)


def _lower(hits: int, attempts: int) -> float:
    return float(betaincinv(hits, attempts - hits + 1, _TAIL)) if hits else 0.0


def _upper(hits: int, attempts: int) -> float:
    return float(betainccinv(hits + 1, attempts - hits, _TAIL)) if hits < attempts else 1.0


@dataclass(slots=True)
class _Tally:
    attempts: int = 0
    misses: int = 0
    unavailable: int = 0
    positive: int = 0
    negative: int = 0

    def observe(
        self, point: np.ndarray, se: np.ndarray, available: np.ndarray, truth: float
    ) -> None:
        radius = Z95 * se
        lo, hi = point - radius, point + radius
        self.attempts += len(point)
        self.unavailable += len(point) - int(np.count_nonzero(available))
        self.misses += len(point) - int(np.count_nonzero(available & (lo <= truth) & (truth <= hi)))
        self.positive += int(np.count_nonzero(available & (lo > 0)))
        self.negative += int(np.count_nonzero(available & (hi < 0)))

    def evidence(self, model: _Scenario) -> ScenarioEvidence:
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
            miss_upper=_upper(self.misses, self.attempts),
            detection_lower=_lower(correct, self.attempts) if truth else None,
            detection_upper=_upper(correct, self.attempts) if truth else None,
        )


def _measure(
    design: SpilloverDesign, models: tuple[_Scenario, ...], attempts: int
) -> tuple[ScenarioEvidence, ...]:
    # Equal potential count schedules can use identical assignments, even under
    # different labels. Each labeled row still represents the full fixed experiment.
    unique = {(m.low, m.high): m for m in models}
    cases = tuple(unique.values())
    tallies = {key: _Tally() for key in unique}
    ordered = tuple(tallies.values())
    for i, high, counts in _states(design, cases, attempts):
        batch = summarize_counts(design, high, counts)
        ordered[i].observe(
            batch.estimate, batch.standard_error, batch.available, cases[i].true_effect
        )
    return tuple(tallies[m.low, m.high].evidence(m) for m in models)


def _conclude(rows: tuple[ScenarioEvidence, ...]) -> PlanningEvidence:
    expected = {(p, s) for p in _PROFILES for s in _SCHEDULES}
    if len(rows) != len(expected) or {(s.baseline_profile, s.schedule) for s in rows} != expected:
        return PlanningEvidence(rows, False, "not_calibrated", "incomplete_bounds")
    if any(not (0 <= s.miss_upper <= 0.055) for s in rows):
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


def evaluate_design(
    design: SpilloverDesign, baseline_rate: float | None, meaningful_effect: float
) -> PlanningEvidence:
    """Check calibration and directional detection before the trial is assigned."""
    if (
        not isinstance(meaningful_effect, Real)
        or isinstance(meaningful_effect, (bool, np.bool_))
        or not -1 <= meaningful_effect <= 1
        or meaningful_effect == 0
    ):
        refuse(_INVALID_EFFECT, meaningful_effect=meaningful_effect)
    if baseline_rate is None:
        return PlanningEvidence((), False, "not_calibrated", "missing_baseline_rate")
    if (
        not isinstance(baseline_rate, Real)
        or isinstance(baseline_rate, (bool, np.bool_))
        or not 0 <= baseline_rate <= 1
    ):
        refuse(_INVALID_RATE, baseline_rate=baseline_rate)
    if len(design.groups) * _ATTEMPTS > _MAX_GROUP_ATTEMPTS:
        return PlanningEvidence((), False, "not_calibrated", "simulation_budget_exceeded")
    stratum_sizes = Counter(g.stratum for g in design.groups)
    if any(min(high, stratum_sizes[s] - high) < 2 for s, high in design.high_groups.items()):
        return PlanningEvidence((), False, "not_calibrated", "too_few_groups")
    sizes = tuple(len(g.members) for g in design.groups)
    grid = _model_grid(sizes, float(baseline_rate), float(meaningful_effect))
    if isinstance(grid, str):
        return PlanningEvidence((), False, "not_calibrated", grid)
    return _conclude(_measure(design, grid, _ATTEMPTS))
