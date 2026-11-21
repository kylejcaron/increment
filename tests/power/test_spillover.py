"""Operating characteristics for the prospective two-stage binary model."""

from dataclasses import FrozenInstanceError
from itertools import combinations, product
from math import comb, prod, sqrt
from statistics import mean, variance

import numpy as np
import pyarrow as pa
import pytest
from scipy.stats import beta, binom, norm

from calibration.spillover import certify_decisions, certify_sampling, three_levels
from increment.errors import CodedError
from increment.spillover import SpilloverComparison, SpilloverDesign, from_frame


def _comparison(rate=0.4, effect=0.1, policy="upper"):
    return SpilloverComparison(
        policy=policy, reference="base", meaningful_effect=effect, baseline_rate=rate
    )


def _plan(sizes=(10,) * 4, *, comparisons=None, quotas=None, offers=None):
    groups = range(len(sizes))
    quotas = quotas or {"base": len(sizes) - len(sizes) // 2, "upper": len(sizes) // 2}
    design = SpilloverDesign(
        groups_per_policy={"all": quotas},
        comparisons=comparisons or (_comparison(),),
        outcome="renewed",
        horizon="90 days",
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


def _exact_rates(sizes, offers, low, high):
    """Enumerate both fixed-count draws; compare public tallies, not simulated assignments."""
    groups = len(sizes)
    per_arm = groups // 2
    weight = groups / sum(sizes)
    truth = (sum(high) - sum(low)) / sum(sizes)
    z = norm.isf(0.025)
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
            point = weight * (mean(p) - mean(r))
            se = weight * sqrt((variance(p) + variance(r)) / per_arm)
            available = se > 0 and 0 < sum(y) < sum(draws)
            lower, upper = point - z * se, point + z * se
            rates += (
                prod(masses)
                / comb(groups, per_arm)
                * np.array(
                    [
                        not available,
                        not (available and lower <= truth <= upper),
                        available and lower > 0,
                        available and upper < 0,
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
    offers = ((0, 0), (0, 1), (0, 2), (0, 0))
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
    allocations, z = comb(1000, 500), norm.isf(0.025)
    for a, b in product(range(11), range(51)):
        mass = comb(10, a) * comb(50, b) * comb(940, 500 - a - b) / allocations
        p, r = (a + b) / 500, (10 - a) / 500
        se = sqrt((p * (1 - p) + r * (1 - r)) / 499)
        lower, upper = p - r - z * se, p - r + z * se
        miss += mass * (se == 0 or not lower <= 0.05 <= upper)
        power += mass * (se > 0 and lower > 0)
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
        (0.5, 0.2, (True, True, False, False), "effect_exceeds_capacity"),
        (0.5, -0.2, (False, False, True, True), "effect_exceeds_capacity"),
        (0.25, -0.025, (True, False, False, False), "degenerate_cancellation"),
    ],
)
def test_unassessable_models_keep_the_point_only_decision(rate, effect, movable, reason):
    (plan,) = _plan(
        comparisons=(_comparison(rate, effect),), offers=[(1, 1 + int(move)) for move in movable]
    )
    assert not plan.calibration_passed
    assert (plan.status, plan.reason, plan.scenarios) == ("not_calibrated", reason, ())


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_sparse_design_keeps_all_labelled_schedules_and_all_attempts():
    (plan,) = _plan(comparisons=(_comparison(rate=0.025),))
    assert (plan.status, plan.reason) == ("not_calibrated", "miss_bound_exceeded")
    assert len({row.name for row in plan.scenarios}) == 12
    for row in plan.scenarios:
        assert row.attempts == 65_536
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
def test_unavailable_draws_count_as_misses_but_not_false_alarms():
    (plan,) = _plan((4,) * 5, comparisons=(_comparison(0.6),))
    row = next(s for s in plan.scenarios if s.name == "largest_first/unchanged")
    # Of ten allocations, one gives two all-zero groups high (zero SE), and
    # three give two all-success groups high (point = 2/3, SE = 1/3).
    assert row.unavailable / row.attempts == pytest.approx(1 / 10, abs=0.025)
    assert row.false_calls / row.attempts == pytest.approx(3 / 10, abs=0.025)
    assert row.misses == row.false_calls + row.unavailable

    (plan,) = _plan(comparisons=(_comparison(0.5, -0.5),), offers=((0, 1),) * 4)
    # Each reference group sees all five events, each policy group none:
    # the contrast is nonzero but neither cell has between-group variation.
    row = next(s for s in plan.scenarios if s.name == "spread/concentrated")
    assert row.misses == row.unavailable == row.attempts
    assert row.correct_detections == row.wrong_direction_detections == 0
    _check_bounds(row, 1)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_scarce_sibling_does_not_relax_nominal_or_simulation_tails():
    first, scarce = _plan(
        (6,) * 6,
        quotas={"base": 2, "middle": 3, "upper": 1},
        comparisons=(_comparison(0.5, policy="middle"), _comparison()),
        offers=((1, 0, 4),) * 6,
    )
    assert (scarce.reason, scarce.scenarios) == ("too_few_groups", ())
    # Three all-success and three all-zero groups. Available allocations
    # can reach |t|=2, enough for one comparison but not for two.
    border = next(s for s in first.scenarios if s.name == "largest_first/unchanged")
    assert border.false_calls == 0
    assert 0 < border.misses == border.unavailable < border.attempts
    for row in first.scenarios:
        assert row.attempts == 196_608
        _check_bounds(row, 2)


@pytest.mark.slow
@pytest.mark.numerical_certification
def test_three_policy_plans_clear_the_miss_limit_but_not_the_detection_target():
    design, frames = three_levels()
    experiment, reader = from_frame(
        frames["roster"], offer_counts=frames["offer_counts"], design=design
    )
    # Forward binomial tests at the two decision thresholds, independent of
    # the planner's reported inverse-beta bounds and statuses.
    with reader:
        assert len(experiment.planning) == 2
        for plan in experiment.planning:
            rows = plan.evidence.scenarios
            assert len(rows) == 12 and {s.attempts for s in rows} == {196_608}
            for s in rows:
                assert binom.cdf(s.misses, s.attempts, 0.055 / 2) <= 0.05 / 48
            alternatives = [s for s in rows if s.true_effect != 0]
            assert len(alternatives) == 6
            assert any(
                binom.cdf(s.correct_detections, s.attempts, 0.8) < 0.05 / 48 for s in alternatives
            )
            assert plan.evidence.calibration_passed and plan.status == "underpowered"


def _attempt_count(family):
    # At true miss probability .05/m, all 12m scenarios must clear their
    # CP upper miss limit except with family probability at most .05.
    for attempts in range(65_536, 2**31 + 1, 65_536):
        lo, hi = -1, attempts
        while hi - lo > 1:
            k = (hi + lo) // 2
            if beta.isf(0.05 / (24 * family), k + 1, attempts - k) <= 0.055 / family:
                lo = k
            else:
                hi = k
        if 12 * family * binom.sf(lo, attempts, 0.05 / family) <= 0.05:
            return attempts
    raise AssertionError("no feasible assignment count")


@pytest.mark.slow
@pytest.mark.parametrize("family", [1, 2, 3, 5])
def test_attempt_schedule_is_minimal_and_respects_both_work_bounds(family):
    draws = _attempt_count(family)
    labels = [f"p{i}" for i in range(family)]
    siblings = tuple(_comparison(None, policy=p) for p in labels[1:])

    def plan(groups, effect):
        quotas = {"base": groups - 2 * family, **dict.fromkeys(labels, 2)}
        return _plan(
            (4,) * groups,
            quotas=quotas,
            comparisons=(_comparison(0.25, effect, labels[0]), *siblings),
            offers=((0, *(1,) * family),) * groups,
        )[0]

    maximum = 2048 if family == 1 else min(2**29 // draws, 2**31 // (family * draws))
    observed = plan(min(1000, maximum), 0.1)
    assert {r.attempts for r in observed.scenarios} == {draws}
    for row in observed.scenarios:
        _check_bounds(row, family)
    # A sub-person change is rejected after budgeting, without doing a
    # half-billion-draw simulation just to see which input was out of bounds.
    assert plan(maximum, 1e-8).reason == "effect_rounds_to_zero"
    assert plan(maximum + 1, 1e-8).reason == "simulation_budget_exceeded"


@pytest.mark.slow
@pytest.mark.numerical_certification
@pytest.mark.parametrize("campaign", [certify_sampling, certify_decisions])
def test_calibration_instruments_meet_their_analytic_error_limits(campaign):
    checks = campaign()
    assert checks
    for check in checks:
        assert check.error <= check.limit, check
