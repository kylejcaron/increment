"""Operating characteristics for the prospective two-stage binary model."""

from dataclasses import FrozenInstanceError
from functools import cache
from itertools import combinations, product
from math import ceil, comb, log2, prod

import numpy as np
import pyarrow as pa
import pytest
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binom

from calibration.spillover import certify_decisions, certify_sampling, three_levels
from increment.errors import CodedError
from increment.power._spillover import _model_grid
from increment.spillover import SpilloverComparison, SpilloverDesign, from_frame


def _comparison(rate=0.4, effect=0.1, policy="upper"):
    return SpilloverComparison(
        policy=policy, reference="base", meaningful_effect=effect, baseline_rate=rate
    )


def _plan(sizes=(10,) * 4, *, comparisons=None, quotas=None, offers=None, alpha=0.05, groups=None):
    if groups is None:
        groups = range(len(sizes))
    quotas = quotas or {"base": len(sizes) - len(sizes) // 2, "upper": len(sizes) // 2}
    design = SpilloverDesign(
        groups_per_policy={"all": quotas},
        comparisons=comparisons or (_comparison(),),
        outcome="renewed",
        horizon="90 days",
        alpha=alpha,
    )
    if offers is None:
        offers = [(1, n - 2) for n in sizes]
    roster = pa.table(
        {
            "person_id": np.arange(sum(sizes)),
            "group_id": np.repeat(groups, sizes),
            "stratum": ["all"] * sum(sizes),
        }
    )
    counts = pa.table(
        {
            "group_id": np.repeat(groups, len(design.policies)),
            "policy": list(design.policies) * len(sizes),
            "offers": [q for row in offers for q in row],
        }
    )
    experiment, reader = from_frame(roster, offer_counts=counts, design=design)
    with reader:
        return tuple(plan.evidence for plan in experiment.planning)


@cache
def _lower_limit(n, mean, second, alpha=0.05):
    # Independent single-stratum inversion: the moment-matching law puts mass
    # mean²/second at second/mean, and its remaining mass at zero.
    if mean == 0:
        return 0.0
    p, value = mean**2 / second, second / mean
    bets = 0.95 * 0.5 ** np.arange(ceil(log2(n)) + 1)

    def objective(m):
        if m == 0:
            return np.inf
        at_zero = np.log1p(-bets)
        at_value = np.log1p(bets * (value / m - 1))
        return logsumexp(n * (at_zero + p * (at_value - at_zero))) - np.log(len(bets) * 4 / alpha)

    return brentq(objective, 0, 1)


def _exact_rates(sizes, offers, low, high):
    """Enumerate both fixed-count draws; compare public tallies, not simulated assignments."""
    groups = len(sizes)
    per_arm = groups // 2
    scale = groups * max(sizes) / sum(sizes)
    truth = (sum(high) - sum(low)) / sum(sizes)

    def limits(totals):
        n = len(totals)
        x = np.array(totals) / max(sizes)
        lo, comp = (_lower_limit(n, v.mean(), np.mean(v**2)) for v in (x, 1 - x))
        return min(1, scale * lo), min(1, scale * (1 - comp))

    rates = np.zeros(4)  # unavailable, missed truth, positive interval, negative interval
    for policy in combinations(range(groups), per_arm):
        draws = [
            n - q[int(j in policy)] for j, (n, q) in enumerate(zip(sizes, offers, strict=True))
        ]
        laws = []
        for j, (n, u) in enumerate(zip(sizes, draws, strict=True)):
            k = high[j] if j in policy else low[j]
            laws.append(
                [
                    (y, comb(k, y) * comb(n - k, u - y) / comb(n, u))
                    for y in range(max(0, k + u - n), min(k, u) + 1)
                ]
            )
        for counts in product(*laws):
            y, masses = zip(*counts, strict=True)
            totals = [n * c / u for n, c, u in zip(sizes, y, draws, strict=True)]
            p = [t for j, t in enumerate(totals) if j in policy]
            r = [t for j, t in enumerate(totals) if j not in policy]
            p_lo, p_hi = limits(p)
            r_lo, r_hi = limits(r)
            lower, upper = p_lo - r_hi, p_hi - r_lo
            rates += (
                prod(masses)
                / comb(groups, per_arm)
                * np.array(
                    [
                        False,
                        not lower <= truth <= upper,
                        lower > 0,
                        upper < 0,
                    ]
                )
            )
    return rates


def _check_rates(row, sizes, offers, low, high):
    truth = (sum(high) - sum(low)) / sum(sizes)
    assert row.true_effect == truth
    unavailable, missed, positive, negative = _exact_rates(sizes, offers, low, high)
    assert row.unavailable / row.attempts == pytest.approx(unavailable, abs=0.025)
    assert row.misses / row.attempts == pytest.approx(missed, abs=0.025)
    if truth:
        correct, wrong = (positive, negative) if truth > 0 else (negative, positive)
        assert row.correct_detections / row.attempts == pytest.approx(correct, abs=0.025)
        assert row.wrong_direction_detections / row.attempts == pytest.approx(wrong, abs=0.025)
        assert row.false_calls is None
    else:
        assert row.false_calls / row.attempts == pytest.approx(positive + negative, abs=0.025)
        assert row.misses == row.unavailable + row.false_calls
        assert row.correct_detections is row.wrong_direction_detections is None


def _check_bounds(row, family):
    # Invert the reported probabilities with the forward binomial tails.
    tail = 0.05 / (24 * family)
    for hits, upper in [
        (row.misses, row.miss_upper),
        (row.correct_detections, row.detection_upper),
    ]:
        if hits is None:
            assert upper is None
        elif hits == row.attempts:
            assert upper == 1
        else:
            assert binom.cdf(hits, row.attempts, upper) == pytest.approx(tail)
    if row.correct_detections is None:
        assert row.detection_lower is None
    elif row.correct_detections == 0:
        assert row.detection_lower == 0
    else:
        assert binom.sf(
            row.correct_detections - 1, row.attempts, row.detection_lower
        ) == pytest.approx(tail)


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("effect", [-0.145, 0.145])
def test_whole_roster_rounding_and_profile_schedules_predict_interval_frequencies(effect):
    # On this roster both the baseline and |change| round from 14.5 to 15 people.
    sizes, offers = (10, 20, 30, 40), ((0, 1), (0, 2), (0, 1), (0, 2))
    grid = _model_grid(sizes, 0.145, effect, (True,) * len(sizes))
    assert not isinstance(grid, str)
    policies = {(s.baseline_profile, s.schedule): (s.low, s.high) for s in grid}
    (plan,) = _plan(sizes, offers=offers, comparisons=(_comparison(0.145, effect),))
    # Baseline, cancellation, proportional gains, concentrated gains. Raising
    # the least affected groups uses event FRACTIONS, not raw event counts.
    profiles = {
        "spread": ((2, 3, 4, 6), (0, 3, 12, 0), (3, 6, 9, 12), (2, 3, 4, 21)),
        "largest_first": ((0, 0, 0, 15), (0, 15, 0, 0), (2, 4, 5, 19), (0, 0, 15, 15)),
        "smallest_first": ((10, 5, 0, 0), (0, 0, 0, 15), (10, 8, 5, 7), (10, 5, 0, 15)),
    }
    assert {(s.baseline_profile, s.schedule) for s in plan.scenarios} == set(
        product(profiles, ("unchanged", "cancelling", "spread", "concentrated"))
    )
    for row in plan.scenarios:
        low, cancel, spread, concentrated = profiles[row.baseline_profile]
        if effect < 0:
            spread = concentrated = (0,) * 4
        high = {
            "unchanged": low,
            "cancelling": cancel,
            "spread": spread,
            "concentrated": concentrated,
        }
        assert policies[row.baseline_profile, row.schedule] == (low, high[row.schedule])
        _check_rates(row, sizes, offers, low, high[row.schedule])
        if row.schedule == "cancelling":
            assert row.cancelling_pairs == sum(
                max(0, h - l) for l, h in zip(low, cancel, strict=True)
            )


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("effect", [-0.1, 0.1])
def test_signed_changes_and_cancellation_leave_equal_offer_groups_unchanged(effect):
    sizes = (10,) * 4
    offers = ((0, 0), (0, 1), (0, 2), (1, 1))
    grid = _model_grid(sizes, 0.5, effect, tuple(r != p for r, p in offers))
    assert not isinstance(grid, str)
    spread = {s.schedule: (s.low, s.high) for s in grid if s.baseline_profile == "spread"}
    (plan,) = _plan(sizes, offers=offers, comparisons=(_comparison(0.5, effect),))
    sign = 1 if effect > 0 else -1
    low = (5,) * 4
    expected = {
        "cancelling": (5, 10, 0, 5),
        "spread": (5, 5 + 2 * sign, 5 + 2 * sign, 5),
        "concentrated": (5, 5 + 4 * sign, 5, 5),
    }
    rows = {s.schedule: s for s in plan.scenarios if s.baseline_profile == "spread"}
    assert rows["cancelling"].cancelling_pairs == 5
    for schedule, high in expected.items():
        assert spread[schedule] == (low, high)
        _check_rates(rows[schedule], sizes, offers, low, high)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_rare_baseline_does_not_cap_a_feasible_meaningful_increase():
    (plan,) = _plan((10,) * 1000, comparisons=(_comparison(0.01, 0.05),))
    assert len(plan.scenarios) == 12
    for row in plan.scenarios:
        if row.schedule == "cancelling":
            assert row.true_effect == 0 and row.cancelling_pairs == 100
        elif row.schedule != "unchanged":
            assert row.true_effect == 500 / 10_000
    # Largest-first has ten all-event reference groups. A concentrated gain
    # makes fifty more all-event groups; stage two cannot change their rates.
    # Count the high-policy allocations by their hits in each of these two sets.
    miss = power = 0.0
    allocations = comb(1000, 500)
    for a, b in product(range(11), range(51)):
        mass = comb(10, a) * comb(50, b) * comb(940, 500 - a - b) / allocations
        p, r = (a + b) / 500, (10 - a) / 500
        # X is binary so its first and second raw moments agree, on either tail.
        lower = _lower_limit(500, p, p) + _lower_limit(500, 1 - r, 1 - r) - 1
        upper = 1 - _lower_limit(500, 1 - p, 1 - p) - _lower_limit(500, r, r)
        miss += mass * (not lower <= 0.05 <= upper)
        power += mass * (lower > 0)
    row = next(s for s in plan.scenarios if s.name == "largest_first/concentrated")
    assert row.misses / row.attempts == pytest.approx(miss, abs=0.025)
    assert row.correct_detections / row.attempts == pytest.approx(power, abs=0.025)


@pytest.mark.parametrize(
    ("rate", "effect", "field"),
    [
        (np.nan, 0.1, "baseline_rate"),
        (-0.1, 0.1, "baseline_rate"),
        (True, 0.1, "baseline_rate"),
        (None, 0, "meaningful_effect"),
        (0.1, -1.1, "meaningful_effect"),
        (None, np.inf, "meaningful_effect"),
    ],
)
def test_invalid_planning_inputs_refuse_even_without_history(rate, effect, field):
    with pytest.raises(CodedError) as exc:
        _comparison(rate=rate, effect=effect)
    assert exc.value.code == f"spillover.planning.{field}"
    assert set(exc.value.context) == {field}


@pytest.mark.parametrize(
    ("rate", "effect", "movable", "reason"),
    [
        (None, 0.1, (True,) * 4, "missing_baseline_rate"),
        (0.3, 0.001, (True,) * 4, "effect_rounds_to_zero"),
        (0.1, -0.3, (True,) * 4, "effect_exceeds_capacity"),
        (1, 0.1, (True,) * 4, "effect_exceeds_capacity"),
        (0, 0.05, (True,) * 4, "degenerate_cancellation"),
        (0.5, 0.2, (True, True, False, False), "degenerate_cancellation"),
        (0.5, -0.2, (False, False, True, True), "effect_exceeds_capacity"),
        (0.25, 0.025, (True, False, False, False), "degenerate_cancellation"),
    ],
)
def test_unassessable_power_keeps_its_specific_reason(rate, effect, movable, reason):
    (plan,) = _plan(
        comparisons=(_comparison(rate, effect),), offers=[(1, 1 + int(move)) for move in movable]
    )
    assert (plan.status, plan.reason, plan.scenarios) == ("not_assessed", reason, ())


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_sparse_design_keeps_all_labelled_schedules_and_all_attempts():
    (plan,) = _plan(comparisons=(_comparison(rate=0.025),))
    assert (plan.status, plan.reason) == ("underpowered", "detection_below_target")
    assert len({row.name for row in plan.scenarios}) == 12
    for row in plan.scenarios:
        assert row.attempts == 4096
        _check_bounds(row, 1)
        if row.true_effect == 0:
            assert row.misses == row.unavailable + row.false_calls
        else:
            assert row.false_calls is None
    for record, field in [(plan.scenarios[0], "misses"), (plan, "status")]:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("sign,dense", [(-1, False), (1, True)])
def test_registered_alpha_crosses_a_signed_detection_boundary(sign, dense):
    # With the policy arm all successes (all failures for a decrease), the
    # reference is a census mean of 1/2. For constant observations the lower
    # limit is c*x, so separation requires (1 + 1/2)*c > 1. Directly evaluate
    # the five registered bets at c=2/3.
    e = sum((1 + 0.5 * 0.95 / 2**j) ** 16 for j in range(5)) / 5
    assert 4 / 0.05 < e < 4 / 0.01
    sizes = (20,) * 32
    offers = tuple((0, (i % 16) + 1 if dense else 19) for i in range(32))
    effect = 0.5 if sign > 0 else -0.45
    for alpha in (0.05, 0.01):
        (plan,) = _plan(sizes, offers=offers, comparisons=(_comparison(0.5, effect),), alpha=alpha)
        row = next(s for s in plan.scenarios if s.name == "spread/spread")
        if sign > 0:
            # The policy arm is deterministically all successes.
            assert row.correct_detections == (4096 if alpha == 0.05 else 0)
            assert row.misses == 0
        else:
            # Each policy group has one event and one unoffered person among 20.
            # Enumerate all 17 possible successes across the 16 sampled groups.
            r_lo = _lower_limit(16, 0.5, 0.25, alpha)
            power = miss = 0.0
            for k in range(17):
                p = k / 16
                lo = _lower_limit(16, p, p, alpha) + r_lo - 1
                hi = 1 - _lower_limit(16, 1 - p, 1 - p, alpha) - r_lo
                mass = binom.pmf(k, 16, 0.05)
                power += mass * (hi < 0)
                miss += mass * (not lo <= effect <= hi)
            assert power == pytest.approx(0.95**16 if alpha == 0.05 else 0)
            assert row.correct_detections / row.attempts == pytest.approx(power, abs=0.025)
            assert row.misses / row.attempts == pytest.approx(miss, abs=0.025)
        assert row.unavailable == row.wrong_direction_detections == 0
        _check_bounds(row, 1)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_scarce_sibling_keeps_family_and_simulation_tails():
    # For n=16 and constant group means (1/2, 1), the separation boundary
    # is crossed at 2*2/.05 but not at 2*3/.05 for a three-policy family.
    e = sum((1 + 0.5 * 0.95 / 2**j) ** 16 for j in range(5)) / 5
    assert 4 / 0.05 < e < 6 / 0.05
    first, scarce = _plan(
        (10,) * 33,
        quotas={"base": 16, "middle": 16, "upper": 1},
        comparisons=(_comparison(0.5, 0.5, policy="middle"), _comparison(0.5, 0.5)),
        offers=((0, 1, 2),) * 33,
    )
    for plan in (first, scarce):
        assert len(plan.scenarios) == 12
        row = next(s for s in plan.scenarios if s.name == "spread/spread")
        assert row.correct_detections == row.unavailable == 0
        _check_bounds(row, 2)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_three_policy_plans_assess_detection_with_simultaneous_bounds():
    design, frames = three_levels()
    experiment, reader = from_frame(
        frames["roster"], offer_counts=frames["offer_counts"], design=design
    )
    # Forward binomial tests, independent of the reported inverse-beta bounds.
    with reader:
        assert len(experiment.planning) == 2
        for plan in experiment.planning:
            rows = plan.evidence.scenarios
            assert len(rows) == 12 and {s.attempts for s in rows} == {4096}
            for row in rows:
                _check_bounds(row, 2)
            alternatives = [s for s in rows if s.true_effect != 0]
            assert len(alternatives) == 6
            assert any(
                binom.cdf(s.correct_detections, s.attempts, 0.8) < 0.05 / 48 for s in alternatives
            )
            assert plan.status == "underpowered"


@pytest.mark.slow
@pytest.mark.parametrize("family", [1, 2, 3, 5])
def test_family_work_limit_is_monotone_through_2100_groups(family):
    labels = [f"p{i}" for i in range(family)]
    siblings = tuple(_comparison(None, policy=p) for p in labels[1:])

    def reason(groups):
        quotas = {"base": groups - family, **dict.fromkeys(labels, 1)}
        return _plan(
            (2,) * groups,
            quotas=quotas,
            comparisons=(_comparison(0.25, 1e-8, labels[0]), *siblings),
            offers=((0, *(1,) * family),) * groups,
        )[0].reason

    # One family-wide ceiling of 2**25 group-attempts, even when history is
    # missing for siblings. Rounding a tiny effect avoids running the simulation.
    maximum = 2**25 // (4096 * family)
    assert reason(maximum) == "effect_rounds_to_zero"
    assert reason(maximum + 1) == "simulation_budget_exceeded"
    assert reason(2100) == (
        "effect_rounds_to_zero" if 2100 <= maximum else "simulation_budget_exceeded"
    )


@pytest.mark.slow
def test_group_renaming_preserves_models_and_sampling_with_structural_ties():
    # The middle 20 groups permit changes. Under an ID-dependent model the
    # rename fills all of them with baseline events, leaving no capacity to gain.
    sizes = (10,) * 40
    offers = ((0, 0),) * 10 + ((0, 1),) * 20 + ((1, 1),) * 10
    rename = (*range(20, 30), *range(20), *range(30, 40))
    comparisons = (_comparison(0.5, 0.2),)
    (original,) = _plan(sizes, offers=offers, comparisons=comparisons)
    (renamed,) = _plan(sizes, offers=offers, comparisons=comparisons, groups=rename)
    assert len(original.scenarios) == 12
    assert renamed == original


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("campaign", [certify_sampling, certify_decisions])
def test_calibration_instruments_meet_their_analytic_error_limits(campaign):
    checks = campaign()
    assert checks
    for check in checks:
        assert check.error <= check.limit, check
