"""Reproduce spillover plans or certify their randomization laws and decision boundaries."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from fractions import Fraction
from itertools import combinations, permutations, product
from math import comb
from time import monotonic

import numpy as np
import pyarrow as pa
from scipy.stats import binom

from increment import AnalysisPlan, MetricSpec, Spillover, SpilloverComparison, from_frame, run
from increment._spillover_design import GroupProfile, SpilloverPopulation
from increment.estimation.spillover import SpilloverInterval, summarize_cells, summarize_counts
from increment.estimation.spillover_results import SpilloverEffects
from increment.power import _spillover, spillover_power


def offer_trial(level_counts, quotas, changes):
    """Generate fixed-size groups with a recorded random policy and person-level offer draw."""
    labels = tuple(level_counts)
    size = 20
    count = sum(level_counts.values())
    rng = np.random.default_rng(501)
    groups = [f"g{i:04d}" for i in range(count)]
    assignments = rng.permutation(np.repeat(labels, list(level_counts.values()))).tolist()
    # Only even-numbered groups have an event under the second policy; the
    # potential outcomes are fixed before either draw.
    person_ids, group_ids, own_assignments, bought, buyers = [], [], [], [], []
    for j, (group, policy) in enumerate(zip(groups, assignments, strict=True)):
        selected = set(rng.choice(size, size=quotas[policy], replace=False))
        event = policy == labels[1] and j % 2 == 0
        for i in range(size):
            person = f"{group}/{i}"
            person_ids.append(person)
            group_ids.append(group)
            offered = i in selected
            own_assignments.append("offered" if offered else "no_offer")
            bought.append(None if offered else event)
            if not offered and event:
                buyers.append(person)
    roster = pa.table(
        {"group_id": group_ids, "person_id": person_ids, "stratum": ["all"] * len(person_ids)}
    )
    frames = {
        "roster": roster,
        "summary": roster.append_column("assignment", pa.array(own_assignments)).append_column(
            "bought", pa.array(bought)
        ),
        "offer_counts": pa.table(
            {
                "group_id": [g for g in groups for _ in labels],
                "policy": list(labels) * count,
                "offers": [quotas[label] for label in labels] * count,
            }
        ),
        "group_policy": pa.table({"group_id": groups, "policy": assignments}),
        "assignments": pa.table(
            {
                "person_id": person_ids,
                "assignment": own_assignments,
                "assigned_at": [datetime(2025, 1, 1, 12, tzinfo=UTC)] * len(person_ids),
                "experiment_id": ["pilot"] * len(person_ids),
            }
        ),
        "purchases": pa.table(
            {
                "person_id": buyers,
                "purchased_at": [datetime(2025, 1, 2, tzinfo=UTC)] * len(buyers),
                "channel": ["web"] * len(buyers),
            }
        ),
    }
    design = {
        "alpha": 0.05,
        "groups_per_policy": {"all": level_counts},
        "comparisons": [
            {
                "policy": p,
                "reference": labels[0],
                "meaningful_effect": effect,
                "baseline_rate": 0.25,
            }
            for p, effect in zip(labels[1:], changes, strict=True)
        ],
    }
    return design, frames


def three_levels():
    return offer_trial(
        {"base": 400, "middle": 300, "upper": 300},
        {"base": 0, "middle": 5, "upper": 10},
        (0.10, 0.10),
    )


def _report(fields, frames):
    design = Spillover(
        control_group="no_offer",
        offered_group="offered",
        groups_per_policy=fields["groups_per_policy"],
        comparisons=tuple(SpilloverComparison(**c) for c in fields["comparisons"]),
    )
    experiment, reader = from_frame(
        frames["summary"],
        shape="summary",
        unit="person_id",
        group="assignment",
        cluster="group_id",
        stratum="stratum",
        design=design,
        offer_counts=frames["offer_counts"],
        group_policy=frames["group_policy"],
        metrics=[MetricSpec(name="bought", type="conversion", window_days=7)],
        plan=AnalysisPlan(primary="bought", alpha=fields["alpha"]),
    )
    with reader:
        start = monotonic()
        plans = spillover_power(experiment)
        elapsed = monotonic() - start
        rows = run(experiment, reader).readout()
        assert isinstance(rows, SpilloverEffects)
    for planning, row in zip(plans, rows, strict=True):
        scenarios = planning.evidence.scenarios
        print(
            json.dumps(
                {
                    "groups": sum(design.groups_per_policy["all"].values()),
                    "comparison": [row.policy, row.reference],
                    "target": next(
                        c.meaningful_effect for c in design.comparisons if c.policy == row.policy
                    ),
                    "planning_seconds_for_family": elapsed,
                    "planning": planning.status,
                    "reason": planning.reason,
                    "attempts_per_scenario": None if not scenarios else scenarios[0].attempts,
                    "worst_detection_lower": min(
                        (s.detection_lower for s in scenarios if s.detection_lower is not None),
                        default=None,
                    ),
                    "worst_detection_upper": min(
                        (s.detection_upper for s in scenarios if s.detection_upper is not None),
                        default=None,
                    ),
                    "worst_miss_upper": max((s.miss_upper for s in scenarios), default=None),
                    "estimate": row.estimate,
                    "interval": [row.lower, row.upper],
                    "unavailable_reason": row.unavailable_reason,
                },
                allow_nan=False,
            ),
            flush=True,
        )


@dataclass(frozen=True)
class CalibrationCheck:
    """Maximum error against the named reference and its prespecified limit."""

    name: str
    error: float
    limit: float


def diagnostic_population(groups, quotas, comparisons, *, alpha=0.05):
    """Construct a finite population directly from known group profiles, without person logs."""
    design = Spillover("no_offer", "offered", quotas, comparisons)
    profiles = tuple(GroupProfile(*group) for group in groups)
    return SpilloverPopulation(
        design.law("event", 7, alpha), profiles, "synthetic finite population"
    )


def _population(sizes, strata, quotas, offers):
    # These are generated finite populations, specified without person identities.
    comparison = SpilloverComparison("upper", "base", 0.01, 0.25)
    groups = ((i, s, n, q) for i, (s, n, q) in enumerate(zip(strata, sizes, offers, strict=True)))
    return diagnostic_population(groups, quotas, (comparison,)), comparison


def _hypergeometric(n, events, draws):
    return {
        y: Fraction(comb(events, y) * comb(n - events, draws - y), comb(n, draws))
        for y in range(max(0, events + draws - n), min(events, draws) + 1)
    }


def _allocate_exact(capacity, count, *, proportional):
    # Exhaust the feasible integer simplex. Spreading minimizes squared distance
    # to the fractional shares; concentration places flips at the largest capacities.
    def objective(move):
        if proportional:
            return -sum(
                (sum(capacity) * x - count * c) ** 2 for c, x in zip(capacity, move, strict=True)
            ), move
        return sum(c * x for c, x in zip(capacity, move, strict=True)), move

    candidates = product(*(range(min(c, count) + 1) for c in capacity))
    return max((v for v in candidates if sum(v) == count), key=objective)


def _allocation_errors(sizes, low, movable, delta, schedules):
    groups = range(len(sizes))
    order = sorted((i for i in groups if movable[i]), key=lambda i: Fraction(low[i], sizes[i]))
    gain = tuple(sizes[i] - low[i] if i in order[: len(order) // 2] else 0 for i in groups)
    loss = tuple(low[i] if i in order[len(order) // 2 :] else 0 for i in groups)
    pairs = min(sum(gain), sum(loss))
    up = _allocate_exact(gain, pairs, proportional=False)
    down = _allocate_exact(loss, pairs, proportional=False)
    capacity = tuple(
        (sizes[i] - low[i] if delta > 0 else low[i]) if movable[i] else 0 for i in groups
    )
    expected = {
        "unchanged": (0,) * len(sizes),
        "cancelling": tuple(g - l for g, l in zip(up, down, strict=True)),
    }
    for label in ("spread", "concentrated"):
        flips = _allocate_exact(capacity, abs(delta), proportional=label == "spread")
        expected[label] = tuple(x if delta > 0 else -x for x in flips)
    labels = {s.schedule for s in schedules}
    error = int(len(schedules) != len(expected) or labels != expected.keys())
    for s in schedules:
        if s.schedule not in expected:
            continue
        error += s.low != low
        moves = tuple(h - l for h, l in zip(s.high, low, strict=True))
        error += sum(abs(a - b) for a, b in zip(moves, expected[s.schedule], strict=True))
        error += s.true_effect != sum(moves) / sum(sizes)
        if s.schedule == "cancelling":
            error += s.cancelling_pairs != pairs
    return error


def _schedule_error():
    error = 0
    for sizes, low, movable in [
        ((10,) * 4, (1, 2, 4, 5), (True,) * 4),
        ((10,) * 4, (0, 3, 7, 10), (False, True, True, False)),
        ((100, 10, 40, 20), (10, 3, 20, 4), (True,) * 4),
    ]:
        for delta in (-4, 4):
            schedules = _spillover._schedules("spread", sizes, low, delta, movable)
            error += _allocation_errors(sizes, low, movable, delta, schedules)

    # Count the schedules actually presented to the planner. A rate of .145 among
    # 100 people rounds to 15; equal offer quotas cannot move anyone's outcome.
    unequal = {
        "spread": (2, 3, 4, 6),
        "largest_first": (0, 0, 0, 15),
        "smallest_first": (10, 5, 0, 0),
    }
    for sizes, movable, rate, change, count, baselines in (
        ((10, 20, 30, 40), (True,) * 4, 0.145, 0.145, 15, unequal),
        ((10,) * 4, (False, True, True, False), 0.5, 0.1, 4, {"spread": (5,) * 4}),
    ):
        for sign in (-1, 1):
            cases = _spillover._model_grid(sizes, rate, sign * change, movable)
            if isinstance(cases, str):
                error += 1
                continue
            for profile, low in baselines.items():
                schedules = tuple(s for s in cases if s.baseline_profile == profile)
                error += _allocation_errors(sizes, low, movable, sign * count, schedules)
    return CalibrationCheck("schedule deviations from exact feasible integer allocations", error, 0)


def _urn_errors():
    # H(y)=C(E,y)C(N-E,d-y)/C(N,d). A tiny nonzero tail can disappear
    # completely without changing any practicable Monte Carlo histogram.
    errors = []
    for n, events, draws in ((11, 4, 5), (11, 9, 7), (2**54, 2, 2)):
        expected = _hypergeometric(n, events, draws)
        remaining = Fraction(1)
        observed = {}
        for y, split in _spillover._Urn(n, draws, events).bins():
            mass = remaining * Fraction(split)
            observed[y] = mass
            remaining -= mass
        errors.append(float(expected.keys() != observed.keys()))
        errors.extend(float(abs(observed.get(y, 0) / p - 1)) for y, p in expected.items())
    return CalibrationCheck("urn bin masses: maximum relative error", max(errors), 1e-14)


def _centered_errors():
    # T=N*y/u. For these one-person draws y=u=1, hence T=N exactly.
    # The middle value is the exact mean for both multiplicity vectors.
    rng = np.random.default_rng(0)
    mean_error = ss_error = 0.0
    center = 2**54 + 4
    for order in permutations((-4, 0, 4)):
        cell = _spillover._WeightedTotals(2)
        for offset in order:
            size = center + offset
            urn = _spillover._Urn(size, 1, size)
            cell.sample(rng, np.array([1, 7 if offset == 0 else 2]), urn, urn.bins())
        totals = cell.cell()
        mean_error = max(mean_error, float(np.max(abs(totals.mean_total - center))))
        ss_error = max(ss_error, float(np.max(abs(totals.ss / [32, 64] - 1))))
    return (
        CalibrationCheck("centered frequency mean: absolute error", mean_error, 0),
        CalibrationCheck("centered frequency SS: relative error", ss_error, 1e-14),
    )


def _contrast_error(population, comparison, assignment, counts):
    # For stratum s the two cell means of T differ by D_s. HT is sum G_s*D_s/N;
    # Var(HT) sums the squared weights times each cell's sample variance / cell count.
    sizes = np.array([g.size for g in population.groups])
    offers = np.array([g.offers for g in population.groups])
    draws = sizes - np.where(assignment == 1, offers[:, 1], offers[:, 0])
    totals = counts * sizes / draws
    estimate = np.zeros(len(totals))
    variance = np.zeros_like(estimate)
    start = 0
    for law in population.design.groups_per_policy.values():
        groups = sum(law.values())
        weight = groups / sum(sizes)
        section = slice(start, start + groups)
        for label, policy, sign in ((0, "base", -1), (1, "upper", 1)):
            values = totals[:, section][assignment[:, section] == label].reshape(-1, law[policy])
            estimate += sign * weight * values.mean(axis=1)
            # Pair differences keep identically nonintegral totals at exactly zero spread.
            n = law[policy]
            for i, j in combinations(range(n), 2):
                variance += (weight * (values[:, i] - values[:, j]) / n) ** 2 / (n - 1)
        start += groups
    batch = summarize_counts(population, comparison, assignment, counts)
    available = (variance > 0) & counts.any(axis=1) & (counts < draws).any(axis=1)
    if not np.array_equal(batch.available, available) or not np.all(
        np.isnan(batch.standard_error[~available])
    ):
        return 1.0
    return float(
        max(
            np.max(abs(estimate - batch.estimate)),
            np.max(abs(np.sqrt(variance[available]) - batch.standard_error[available]), initial=0),
        )
    )


def _stratified_errors():
    sizes = (6, 7, 8, 9, 6, 7, 8, 9, 10)
    pop, comparison = _population(
        sizes,
        ("a",) * 4 + ("b",) * 5,
        {"a": {"base": 2, "upper": 2}, "b": {"base": 2, "upper": 3}},
        [(1, n - 2) for n in sizes],
    )
    low, high = (3, 0, 8, 0, 0, 7, 0, 9, 0), (4,) * 9
    case = _spillover._Scenario(
        "spread", "spread", low, high, (sum(high) - sum(low)) / sum(sizes), None
    )
    hist = np.zeros((2, 4), dtype=int)
    quota_errors = contrast_error = 0.0
    for _, assignment, counts in _spillover._states(pop, comparison, (case,), 8192):
        quota_errors += np.count_nonzero(assignment[:, :4].sum(axis=1) != 2)
        quota_errors += np.count_nonzero(assignment[:, 4:].sum(axis=1) != 3)
        for arm in (0, 1):
            hist[arm] += np.bincount(counts[assignment[:, 0] == arm, 0], minlength=4)
        if not quota_errors:
            contrast_error = max(
                contrast_error, _contrast_error(pop, comparison, assignment, counts)
            )
    arm_error = float(np.max(abs(hist.sum(axis=1) / 8192 - 0.5)))
    mass_error = 0.0
    for arm, (events, draws) in enumerate(((3, 5), (4, 2))):
        expected = _hypergeometric(6, events, draws)
        for y, hits in enumerate(hist[arm]):
            mass_error = max(mass_error, abs(hits / hist[arm].sum() - float(expected.get(y, 0))))
    _, a, y = next(_spillover._states(pop, comparison, (case,), 1))
    _, a2, y2 = next(_spillover._states(pop, comparison, (case,), 1))
    replay_error = np.count_nonzero(a != a2) + np.count_nonzero(y != y2)
    return (
        CalibrationCheck("stratified allocations violating fixed quotas", float(quota_errors), 0),
        CalibrationCheck("stratified assignment marginal: absolute error", arm_error, 0.025),
        CalibrationCheck("conditional untreated event mass: absolute error", mass_error, 0.035),
        CalibrationCheck("stratified contrast and SE: absolute error", contrast_error, 1e-14),
        CalibrationCheck("changed assignments or counts on replay", float(replay_error), 0),
    )


def _marginal_errors(nuisance):
    # Of six groups only one can contribute. It is reference with probability
    # 2/6 (point -1/2), or policy AND its one event untreated with probability
    # (2/6)*(2/4) (point +1/4). All remaining trials have point zero and no SE.
    pop, comparison = _population(
        (4,) * 6,
        ("all",) * 6,
        {"all": {"base": 2, **nuisance, "upper": 2}},
        [(0, *(4,) * len(nuisance), 2)] * 6,
    )
    case = _spillover._Scenario(
        "spread", "spread", (4, 0, 0, 0, 0, 0), (1, 0, 0, 0, 0, 0), -3 / 24, None
    )
    attempts = 8192
    marginal_error = quota_errors = 0.0
    for _, assignment, counts in _spillover._states(pop, comparison, (case,), attempts):
        for k in (0, 1, 2):
            mask = assignment == k
            quota_errors += np.count_nonzero(mask.sum(axis=1) != 2)
            marginal_error = max(marginal_error, float(np.max(abs(mask.mean(axis=0) - 1 / 3))))
        quota_errors += np.count_nonzero(counts[assignment == 2])
    classes = _spillover._classes(pop.groups, 0, len(pop.design.policies) - 1, case)
    batches = tuple(
        summarize_cells(pop, comparison, cells)
        for cells in _spillover._frequency_cells(pop, comparison, classes, attempts)
    )
    points = np.concatenate([b.estimate for b in batches])
    errors = np.concatenate([b.standard_error for b in batches])
    law = {-0.5: 1 / 3, 0.0: 1 / 2, 0.25: 1 / 6}
    mass_error = max(
        abs(np.count_nonzero(points == value) / attempts - p) for value, p in law.items()
    )
    se_error = float(np.max(np.abs(errors[points != 0] - np.abs(points[points != 0])), initial=0))
    invalid = np.count_nonzero(np.isnan(errors) != (points == 0)) + abs(attempts - len(points))
    prefix = f"{len(nuisance)} nuisance policies"
    return (
        CalibrationCheck(f"{prefix}: quota or nuisance count errors", float(quota_errors), 0),
        CalibrationCheck(f"{prefix}: allocation marginal error", marginal_error, 0.035),
        CalibrationCheck(f"{prefix}: frequency sampler point mass error", mass_error, 0.025),
        CalibrationCheck(f"{prefix}: frequency sampler SE error", se_error, 1e-14),
        CalibrationCheck(f"{prefix}: incorrect availability or attempt count", float(invalid), 0),
    )


def _large_group_errors():
    n = 2_400_000_000
    sizes = (10, 10, 10, n)
    pop, comparison = _population(
        sizes, ("all",) * 4, {"all": {"base": 2, "upper": 2}}, ((0, 2),) * 4
    )
    events = (3, 3, 3, n // 2)
    case = _spillover._Scenario("spread", "unchanged", events, events, 0.0, None)
    offered = []
    ref_errors = 0
    for _, a, y in _spillover._states(pop, comparison, (case,), 16_384):
        ref_errors += np.count_nonzero(y[a[:, -1] == 0, -1] != events[-1])
        offered.append(events[-1] - y[a[:, -1] == 1, -1])
    seen = np.concatenate(offered)
    expected = _hypergeometric(n, events[-1], 2)
    mass_error = max(abs(np.mean(seen == k) - float(p)) for k, p in expected.items())
    wide = replace(pop, groups=(*pop.groups[:-1], replace(pop.groups[-1], offers=(0, n // 2))))
    plan = _spillover.evaluate_design(wide, comparison)
    correct = (
        plan.status == "not_assessed"
        and not plan.scenarios
        and plan.reason == "group_draw_exceeds_planning_limit"
    )
    return (
        CalibrationCheck("billion-member reference census errors", float(ref_errors), 0),
        CalibrationCheck("billion-member two-offer mass error", float(mass_error), 0.025),
        CalibrationCheck("wide-draw planning refusal errors", float(not correct), 0),
    )


def certify_sampling() -> tuple[CalibrationCheck, ...]:
    """Measure distributional error, deterministic numerical error and invalid assignments."""
    return (
        _schedule_error(),
        _urn_errors(),
        *_centered_errors(),
        *_stratified_errors(),
        *_marginal_errors({"middle": 2}),
        *_marginal_errors({"middle": 1, "spare": 1}),
        *_large_group_errors(),
    )


def _tail_error():
    # P_L[X >= k] = P_U[X <= k] = alpha away from the closed endpoints.
    errors = []
    for family, hits in product((1, 3), (0, 7, 37)):
        tail = 0.05 / (24 * family)
        tally = _spillover._Tally()
        tally.observe((np.arange(37) < hits).astype(float), np.ones(37), 1.0)
        row = tally.evidence(_spillover._Scenario("spread", "spread", (0,), (1,), 1.0, None), tail)
        lower, upper = row.detection_lower, row.detection_upper
        if lower is None or upper is None:
            return CalibrationCheck("missing directional simulation bound", 1, 0)
        errors.append(abs(binom.sf(hits - 1, 37, lower) / tail - 1) if hits else float(lower != 0))
        errors.append(
            abs(binom.cdf(hits, 37, upper) / tail - 1) if hits < 37 else float(upper != 1)
        )
    return CalibrationCheck("binomial endpoints: forward tail relative error", max(errors), 1e-12)


def _direction_error():
    # Correct sign, wrong sign, tiny null, missing, point interval, and a range
    # that covers the truth but touches zero. Reflect the whole set for decreases.
    lower = np.array([0.15, -0.25, -0.01, np.nan, 0.2, 0.0])
    upper = np.array([0.25, -0.15, 0.01, np.nan, 0.2, 0.2])
    mistakes = 0
    for truth in (-0.2, 0.2):
        lo, hi = (lower, upper) if truth > 0 else (-upper, -lower)
        tally = _spillover._Tally()
        tally.observe(lo, hi, truth)
        case = _spillover._Scenario("spread", "spread", (1,), (2 if truth > 0 else 0,), truth, None)
        row = tally.evidence(case, 0.05 / 24)
        mistakes += (row.attempts, row.misses, row.unavailable) != (6, 3, 1)
        mistakes += (row.correct_detections, row.wrong_direction_detections, row.false_calls) != (
            2,
            1,
            None,
        )
        mistakes += row.detection_upper is None or not np.isclose(
            binom.cdf(2, 6, row.detection_upper), 0.05 / 24, atol=0, rtol=1e-12
        )
    return CalibrationCheck("signed interval endpoint classification errors", float(mistakes), 0)


def _joint_decision_errors(family):
    # The ordinary scenario and stress tests have independent 80% decisions.
    attempts = 4096
    tail = 0.05 / (24 * family)
    rows = []
    for profile, schedule in product(
        ("spread", "largest_first", "smallest_first"),
        ("unchanged", "cancelling", "spread", "concentrated"),
    ):
        alt = schedule in ("spread", "concentrated")
        case = _spillover._Scenario(
            profile, schedule, (1,), (int(alt) + 1,), 0.1 if alt else 0.0, None
        )
        tally = _spillover._Tally(attempts=attempts, positive=attempts if alt else 0)
        rows.append(tally.evidence(case, tail))

    def witness(profile, lower, upper):
        return tuple(
            replace(s, detection_lower=lower, detection_upper=upper)
            if (s.baseline_profile, s.schedule) == (profile, "concentrated")
            else s
            for s in rows
        )

    below = float(np.nextafter(0.8, 0))
    witnesses = [
        (rows, "supported", "supported", None),
        (witness("spread", 0.8, 1.0), "supported", "supported", None),
        (
            witness("spread", below, 0.8),
            "inconclusive",
            "supported",
            "detection_bounds_overlap_target",
        ),
        (witness("spread", below, below), "underpowered", "supported", "detection_below_target"),
        (witness("largest_first", 0.8, 1.0), "supported", "supported", None),
        (witness("smallest_first", below, 0.8), "supported", "inconclusive", None),
        (witness("largest_first", below, below), "supported", "underpowered", None),
        (witness("smallest_first", None, None), "not_assessed", None, "incomplete_bounds"),
        (rows[:-1], "not_assessed", None, "incomplete_bounds"),
        (
            [*rows[:-1], replace(rows[-1], misses=attempts, miss_upper=1.0)],
            "supported",
            "supported",
            None,
        ),
    ]
    mistakes = 0
    for evidence, status, stress, reason in witnesses:
        result = _spillover._conclude(tuple(evidence))
        mistakes += (result.status, result.stress_status, result.reason) != (status, stress, reason)
    return CalibrationCheck(f"{family}-comparison power decision errors", float(mistakes), 0)


def certify_decisions() -> tuple[CalibrationCheck, ...]:
    """Challenge strict detection, simultaneous tails and the joint decision discontinuities."""
    return (
        _tail_error(),
        _direction_error(),
        _joint_decision_errors(1),
        _joint_decision_errors(2),
    )


def _width_example(sizes, strata):
    """Report one seeded two-stage draw, with nested 20% and 25% potential outcomes."""
    rng = np.random.default_rng(521)
    group_ids = np.arange(len(sizes))
    assignment = np.zeros(len(sizes), dtype=int)
    quotas = {}
    for region in sorted(set(strata)):
        members = group_ids[strata == region]
        half = len(members) // 2
        quotas[str(region)] = {"base": len(members) - half, "upper": half}
        assignment[rng.choice(members, half, replace=False)] = 1
    c = SpilloverComparison("upper", "base", 0.05, 0.20)
    groups = []
    counts = []
    for i, (size, stratum, high) in enumerate(zip(sizes, strata, assignment, strict=True)):
        n = int(size)
        offers = (0, n // 2)
        groups.append((i, str(stratum), n, offers))
        u = rng.random(n)
        successes = np.count_nonzero(u < (0.25 if high else 0.20))
        counts.append(rng.hypergeometric(successes, n - successes, n - offers[high]))
    pop = diagnostic_population(groups, quotas, (c,))
    lo, hi = SpilloverInterval(pop).bounds_from_counts(c, assignment[None, :], np.array([counts]))
    return {
        "stores": len(sizes),
        "size_range": [int(min(sizes)), int(max(sizes))],
        "strata": len(quotas),
        "width_percentage_points": float(100 * (hi[0] - lo[0])),
    }


def _widths():
    for count, size in ((200, 50), (1000, 50), (400, 30)):
        print(json.dumps(_width_example(np.full(count, size), np.zeros(count, dtype=int))))
    bands = np.repeat(np.arange(8), 50)
    sizes = 10 * (bands + 1)
    for strata in (np.zeros_like(bands), bands):
        print(json.dumps(_width_example(sizes, strata)))


def main() -> None:
    """Run a planning example or the numerical certification campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--two-policy", action="store_true", help="compare two-policy planning cases")
    mode.add_argument("--certify", action="store_true", help="measure sampler and decision errors")
    mode.add_argument("--widths", action="store_true", help="show single-draw interval widths")
    args = parser.parse_args()
    if args.certify:
        checks = (*certify_sampling(), *certify_decisions())
        passed = all(c.error <= c.limit for c in checks)
        print(
            json.dumps({"passed": passed, "checks": [asdict(c) for c in checks]}, allow_nan=False)
        )
        if not passed:
            raise SystemExit(1)
    elif args.widths:
        _widths()
    elif args.two_policy:
        for n in (240, 500, 1000):
            changes = (0.01, 0.03, 0.10, 0.25) if n == 1000 else (0.01, 0.10)
            for change in changes:
                _report(
                    *offer_trial(
                        {"base": n // 2, "upper": n // 2}, {"base": 5, "upper": 10}, (change,)
                    )
                )
    else:
        _report(*three_levels())


if __name__ == "__main__":
    main()
