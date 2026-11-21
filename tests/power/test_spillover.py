"""Operating characteristics for the prospective two-stage binary model."""

from dataclasses import FrozenInstanceError, replace
from itertools import combinations
from math import comb, sqrt

import numpy as np
import pytest
from scipy.stats import beta, binom, norm

from increment._spillover_design import (
    GroupProfile,
    SpilloverComparison,
    SpilloverDesign,
    SpilloverPopulation,
)
from increment.errors import CodedError
from increment.estimation.spillover import summarize_counts
from increment.power import _spillover as model


def _comparison(rate=0.4, effect=0.1, policy="upper"):
    return SpilloverComparison(
        policy=policy, reference="base", meaningful_effect=effect, baseline_rate=rate
    )


def _population(sizes=(10,) * 4, *, strata=None, quotas=None, comparisons=None, offers=None):
    strata = strata or ("all",) * len(sizes)
    quotas = quotas or {"all": {"base": len(sizes) - len(sizes) // 2, "upper": len(sizes) // 2}}
    design = SpilloverDesign(
        groups_per_policy=quotas,
        comparisons=comparisons or (_comparison(),),
        outcome="renewed",
        horizon="90 days",
    )
    if offers is None:
        offers = tuple(
            tuple({"base": 1, "middle": 0, "upper": n - 2}[p] for p in design.policies)
            for n in sizes
        )
    return SpilloverPopulation(
        design=design,
        groups=tuple(
            GroupProfile(group_id=f"g{i:04}", size=n, stratum=s, offers=o)
            for i, (n, s, o) in enumerate(zip(sizes, strata, offers, strict=True))
        ),
        roster_digest="f" * 64,
    )


def test_offsetting_and_signed_schedules_move_only_eligible_people():
    sizes, low = (10,) * 4, (1, 2, 4, 5)
    for delta, spread, concentrated in [
        (4, (2, 3, 5, 6), (5, 2, 4, 5)),
        (-4, (1, 1, 3, 3), (1, 2, 4, 1)),
    ]:
        cases = {s.schedule: s for s in model._schedules("spread", sizes, low, delta, (True,) * 4)}
        assert {key: s.high for key, s in cases.items()} == {
            "unchanged": low,
            "cancelling": (10, 2, 0, 0),
            "spread": spread,
            "concentrated": concentrated,
        }
        assert cases["cancelling"].cancelling_pairs == 9
        assert all(s.true_effect == (sum(s.high) - sum(low)) / 40 for s in cases.values())

    unequal = model._schedules("spread", (100, 10, 40, 20), (10, 3, 20, 4), 1, (True,) * 4)
    cancelling = next(s for s in unequal if s.schedule == "cancelling")
    assert cancelling.high == (33, 0, 0, 4)  # rank by event fraction, not count


def test_baseline_profiles_and_half_up_rounding_use_the_entire_roster():
    # 0.145 * 100 is a half-person tie, for both the baseline and the alternative.
    cases = model._model_grid((10, 20, 30, 40), 0.145, -0.145, (True,) * 4)
    assert not isinstance(cases, str)
    assert len({(s.baseline_profile, s.schedule) for s in cases}) == 12
    assert {s.baseline_profile: s.low for s in cases} == {
        "spread": (2, 3, 4, 6),
        "largest_first": (0, 0, 0, 15),
        "smallest_first": (10, 5, 0, 0),
    }
    assert all(s.true_effect == -0.15 for s in cases if s.schedule in {"spread", "concentrated"})


def test_rare_baseline_does_not_cap_a_feasible_meaningful_increase():
    cases = model._model_grid((10,) * 1000, 0.01, 0.05, (True,) * 1000)
    assert not isinstance(cases, str)
    assert len(cases) == 12
    for case in cases:
        assert sum(case.low) == 100
        assert sum(case.high) == (600 if case.true_effect else 100)
        if case.schedule == "cancelling":
            assert case.cancelling_pairs == 100
            assert sum(max(0, h - base) for h, base in zip(case.high, case.low, strict=True)) == 100
        elif case.schedule != "unchanged":
            assert case.true_effect == 0.05


@pytest.mark.parametrize("family", [1, 3])
@pytest.mark.parametrize("hits", [0, 7, 37])
def test_exact_bounds_invert_each_of_the_frozen_family_tails(hits, family):
    tail = 0.05 / (24 * family)
    lower, upper = model._lower(hits, 37, tail), model._upper(hits, 37, tail)
    if hits:
        assert binom.sf(hits - 1, 37, lower) == pytest.approx(tail)
    else:
        assert lower == 0
    if hits < 37:
        assert binom.cdf(hits, 37, upper) == pytest.approx(tail)
    else:
        assert upper == 1


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
    ("rate", "effect", "reason"),
    [
        (None, 0.1, "missing_baseline_rate"),
        (0.3, 0.001, "effect_rounds_to_zero"),
        (0.1, -0.3, "effect_exceeds_capacity"),
        (1, 0.1, "effect_exceeds_capacity"),
        (0, 0.05, "degenerate_cancellation"),
    ],
)
def test_unassessable_models_keep_the_point_only_decision(rate, effect, reason):
    comparison = _comparison(rate, effect)
    result = model.evaluate_design(_population(comparisons=(comparison,)), comparison)
    assert not result.calibration_passed
    assert (result.status, result.reason, result.scenarios) == ("not_calibrated", reason, ())


def test_scarcity_and_operational_limit_are_distinct_from_bad_calibration():
    for population, reason in [
        (_population((6,) * 5, quotas={"all": {"base": 4, "upper": 1}}), "too_few_groups"),
        (_population((4,) * 2049), "simulation_budget_exceeded"),
    ]:
        result = model.evaluate_design(population, population.design.comparisons[0])
        assert (result.reason, result.scenarios) == (reason, ())


def test_sampled_states_follow_both_quota_laws_and_reconcile_to_the_logged_contrast():
    sizes = (6, 7, 8, 9, 6, 7, 8, 9, 10)
    quotas = {"a": {"base": 2, "upper": 2}, "b": {"base": 2, "upper": 3}}
    population = _population(sizes, strata=("a",) * 4 + ("b",) * 5, quotas=quotas)
    comparison = population.design.comparisons[0]
    low, high_events = (3, 0, 8, 0, 0, 7, 0, 9, 0), (4,) * 9
    state = model._Scenario("spread", "spread", low, high_events, 9 / sum(sizes), None)
    histogram = np.zeros((2, 4), dtype=int)
    for _, high, counts in model._states(population, comparison, (state,), 8192):
        assert np.all(high[:, :4].sum(axis=1) == 2)
        assert np.all(high[:, 4:].sum(axis=1) == 3)
        for arm in (0, 1):
            values = counts[high[:, 0] == arm, 0]
            histogram[arm] += np.bincount(values, minlength=4)

    for arm, (events, samples) in enumerate([(3, 5), (4, 2)]):
        expected = [
            comb(events, y) * comb(6 - events, samples - y) / comb(6, samples)
            if y <= samples
            else 0
            for y in range(4)
        ]
        assert histogram[arm].sum() / 8192 == pytest.approx(0.5, abs=0.025)
        np.testing.assert_allclose(histogram[arm] / histogram[arm].sum(), expected, atol=0.035)

    _, high, counts = next(model._states(population, comparison, (state,), 1))
    _, repeated_high, repeated_counts = next(model._states(population, comparison, (state,), 1))
    np.testing.assert_array_equal(repeated_high, high)
    np.testing.assert_array_equal(repeated_counts, counts)
    totals = {(s, p): [] for s in quotas for p in quotas[s]}
    for g, flag, count in zip(population.groups, high[0], counts[0], strict=True):
        label = comparison.policy if flag else comparison.reference
        quota = g.offers[population.design.policies.index(label)]
        unoffered_outcomes = [int(j - quota < count) for j in range(quota, g.size)]
        totals[g.stratum, label].append(g.size * sum(unoffered_outcomes) / len(unoffered_outcomes))
    point = variance = 0.0
    for s, law in quotas.items():
        weight = sum(law.values()) / sum(sizes)
        p, r = totals[s, "upper"], totals[s, "base"]
        point += weight * (np.mean(p) - np.mean(r))
        variance += weight**2 * (np.var(p, ddof=1) / len(p) + np.var(r, ddof=1) / len(r))
    simulated = summarize_counts(population, comparison, high, counts)
    assert simulated.estimate[0] == pytest.approx(point)
    assert simulated.available[0] == (variance > 0)
    if variance:
        assert simulated.standard_error[0] == pytest.approx(sqrt(variance))
    else:
        assert np.isnan(simulated.standard_error[0])


def test_unavailable_draws_count_as_misses_but_not_false_alarms():
    # Of ten allocations, only assigning both all-zero groups high has zero SE.
    # Choosing two all-success groups gives estimate 2/3 and SE 1/3 (3 false alarms);
    # choosing one of each gives an interval containing zero (6 allocations).
    events = (0, 0, 4, 4, 4)
    null = model._Scenario("spread", "unchanged", events, events, 0.0, None)
    all_zero = replace(null, low=(0,) * 5, high=(0,) * 5)
    population = _population((4,) * 5)
    evidence, no_events = model._measure(
        population, population.design.comparisons[0], (null, all_zero), 4096
    )
    assert no_events.misses == no_events.unavailable == no_events.attempts == 4096
    assert no_events.false_calls == 0 and no_events.miss_upper == 1
    assert evidence.attempts == 4096
    assert evidence.unavailable / evidence.attempts == pytest.approx(1 / 10, abs=0.025)
    assert evidence.false_calls is not None
    assert evidence.false_calls / evidence.attempts == pytest.approx(3 / 10, abs=0.025)
    assert evidence.misses == evidence.unavailable + evidence.false_calls
    assert evidence.correct_detections is evidence.wrong_direction_detections is None
    assert evidence.detection_lower is evidence.detection_upper is None


def test_detection_is_strict_and_in_the_direction_of_the_modeled_effect():
    for effect in (0.2, -0.2):
        z = norm.isf(0.025)
        estimates = np.array([effect, -effect, 0.0, effect, np.sign(effect) * z * 0.1])
        tally = model._Tally()
        tally.observe(
            estimates,
            np.array([0.01, 0.01, 0.01, 0.0, 0.1]),
            np.array([True, True, True, False, True]),
            effect,
            z,
        )
        high = (2 if effect > 0 else 0,)
        case = model._Scenario("spread", "concentrated", (1,), high, effect, None)
        record = tally.evidence(case, 0.05 / 24)
        assert (record.attempts, record.misses, record.unavailable) == (5, 3, 1)
        assert (record.correct_detections, record.wrong_direction_detections) == (1, 1)
        assert record.false_calls is None
        assert record.detection_upper == pytest.approx(beta.isf(0.05 / 24, 2, 4))


def test_a_single_adverse_profile_controls_the_joint_decision():
    grid = model._model_grid((10,) * 4, 0.3, 0.1, (True,) * 4)
    assert not isinstance(grid, str)
    # At the simultaneous CP tail, 3,437 is the last passing miss count.
    good_null = model._Tally(attempts=65_536, misses=3437, positive=3437)
    good_alt = model._Tally(attempts=65_536, misses=3437, positive=60_000)
    tail = 0.05 / 24
    rows = tuple(
        (good_alt if case.true_effect else good_null).evidence(case, tail) for case in grid
    )
    result = model._conclude(rows, 1)
    assert result.calibration_passed and result.status == "supported" and result.reason is None
    for obj, field in [(rows[0], "misses"), (result, "status")]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)
    adverse = (
        (replace(rows[-1], detection_upper=0.79, detection_lower=0.78), "underpowered"),
        (replace(rows[-1], detection_lower=0.79, detection_upper=0.8), "inconclusive"),
        (replace(good_alt, misses=3438).evidence(grid[-1], tail), "not_calibrated"),
        (replace(rows[-1], detection_lower=None), "not_calibrated"),
    )
    for row, status in adverse:
        decision = model._conclude((*rows[:-1], row), 1)
        assert decision.status == status and decision.reason
        assert decision.calibration_passed == (status != "not_calibrated")
    assert not model._conclude(rows[:1], 1).calibration_passed


@pytest.mark.slow
def test_sparse_design_keeps_all_labelled_schedules_and_all_attempts():
    comparison = _comparison(rate=0.025)
    result = model.evaluate_design(_population(comparisons=(comparison,)), comparison)
    assert (result.status, result.reason) == ("not_calibrated", "miss_bound_exceeded")
    assert len({row.name for row in result.scenarios}) == 12
    for row in result.scenarios:
        assert row.attempts == 65_536
        if row.true_effect == 0:
            assert row.false_calls is not None
            assert row.misses == row.unavailable + row.false_calls
        else:
            assert row.false_calls is None


def test_equal_quota_groups_cannot_change_under_any_schedule():
    mask = (False, True, True, False)
    for delta in (-4, 4):
        low = (0, 3, 7, 10)
        cases = model._schedules("spread", (10,) * 4, low, delta, mask)
        assert len(cases) == 4
        for case in cases:
            assert (case.high[0], case.high[3]) == (0, 10)
            assert sum(case.high) - sum(low) == (delta if case.true_effect else 0)
        assert cases[1].cancelling_pairs == 7

    # The whole-roster increase is eight people. Ten of twenty non-events
    # are movable in the spread profile, but none in the largest-first profile.
    comparison = _comparison(rate=0.5, effect=0.2)
    population = _population(comparisons=(comparison,), offers=((1, 3), (1, 3), (2, 2), (2, 2)))
    result = model.evaluate_design(population, comparison)
    assert (result.reason, result.scenarios) == ("effect_exceeds_capacity", ())
    assert model._model_grid((10,) * 4, 0.5, -0.2, (False, False, True, True)) == (
        "effect_exceeds_capacity"
    )
    assert model._model_grid((10,) * 4, 0.25, -0.025, (True, False, False, False)) == (
        "degenerate_cancellation"
    )


def test_attempt_schedule_is_minimal_and_respects_both_work_bounds():
    for m, expected in [(1, 65_536), (2, 196_608), (3, 262_144)]:
        assert model._attempts(1000, m) == expected
        if m > 1:
            for b in range(65_536, expected + 1, 65_536):
                # Invert the exact CP bound directly, independent of the planner's selection.
                lo, hi = -1, b
                while hi - lo > 1:
                    k = (hi + lo) // 2
                    if beta.isf(0.05 / (24 * m), k + 1, b - k) <= 0.055 / m:
                        lo = k
                    else:
                        hi = k
                risk = 12 * m * binom.sf(lo, b, 0.05 / m)
                assert (risk <= 0.05) == (b == expected)
    assert model._attempts(2048, 1) == 65_536
    assert model._attempts(2049, 1) is None
    # For two comparisons the per-pair allowance binds, for five the total binds.
    for m in (2, 5):
        b = model._attempts(10, m)
        assert b is not None
        max_groups = min(2**29 // b, 2**31 // (m * b))
        assert model._attempts(max_groups, m) == b
        assert model._attempts(max_groups + 1, m) is None


def test_scarce_sibling_does_not_relax_tail_or_miss_cap():
    first, scarce = _comparison(policy="middle"), _comparison()
    pop = _population(
        (6,) * 6,
        quotas={"all": {"base": 2, "middle": 3, "upper": 1}},
        comparisons=(first, scarce),
    )
    assert model.evaluate_design(pop, scarce).reason == "too_few_groups"
    grid = model._model_grid((6,) * 6, 0.4, 0.1, (True,) * 6)
    assert not isinstance(grid, str)
    # With three all-event groups and three all-zero groups, nonzero-variance
    # assignments can reach |estimate/SE| = 2 but never the two-pair cutoff.
    events = (0, 0, 0, 6, 6, 6)
    null = model._Scenario("spread", "unchanged", events, events, 0.0, None)
    *rows, border = model._measure(pop, first, (*grid, null), 128)
    assert border.false_calls == 0
    assert 0 < border.misses == border.unavailable < 128
    for row in rows:
        upper = beta.isf(0.05 / 48, row.misses + 1, 128 - row.misses) if row.misses < 128 else 1
        assert row.miss_upper == pytest.approx(upper)
    optimistic = tuple(
        replace(r, miss_upper=0.03, detection_lower=0.9, detection_upper=1) for r in rows
    )
    assert model._conclude(optimistic, 2).reason == "miss_bound_exceeded"


@pytest.mark.parametrize("nuisance", [{"middle": 2}, {"middle": 1, "spare": 1}])
def test_three_label_marginals_and_hypergeometric_histograms_match_enumeration(nuisance):
    comparison = _comparison()
    pop = _population(
        (4,) * 6,
        quotas={"all": {"base": 2, **nuisance, "upper": 2}},
        offers=((0, *(4,) * len(nuisance), 2),) * 6,
    )
    # One exceptional group: reference sees its four events; policy sees
    # zero or one from two draws; all other groups have no events.
    case = model._Scenario("spread", "spread", (4,) + (0,) * 5, (1,) + (0,) * 5, -3 / 24, None)
    draws = 8192
    for _, assignment, counts in model._states(pop, comparison, (case,), draws):
        for label in (0, 1, 2):
            assert np.all(np.count_nonzero(assignment == label, axis=1) == 2)
        assert np.all(counts[assignment == 2] == 0)
        frequency = np.bincount(assignment[:, 0], minlength=3) / len(assignment)
        np.testing.assert_allclose(frequency, 1 / 3, atol=0.035)

    expected = []
    for policy in combinations(range(6), 2):
        for reference in combinations(set(range(6)).difference(policy), 2):
            for unoffered in combinations(range(4), 2):
                point = -0.5 if 0 in reference else 0.25 * (0 in policy and 0 in unoffered)
                expected.append(point)
    classes = model._classes(pop, comparison, case)
    batches = tuple(model._frequency_batches(pop, comparison, classes, draws))
    points = np.concatenate([b.estimate for b in batches])
    errors = np.concatenate([b.standard_error for b in batches])
    assert len(points) == draws
    np.testing.assert_equal(np.isnan(errors), points == 0)
    np.testing.assert_allclose(errors[points != 0], np.abs(points[points != 0]))
    for value in set(expected):
        probability = expected.count(value) / len(expected)
        assert np.mean(points == value) == pytest.approx(probability, abs=0.025)


def test_frequency_cells_preserve_small_spreads_on_large_group_totals():
    counts = model._WeightedTotals(1)
    rng = np.random.default_rng(0)
    origin = 2**54
    for offset in (0, 4, 8):
        size = origin + offset
        urn = model._Urn(size=size, draws=1, events=size)
        counts.sample(rng, np.ones(1, dtype=int), urn, urn.bins())
    cell = counts.cell()
    assert cell.mean_total[0] == origin + 4
    assert cell.ss[0] == pytest.approx(4**2 + 4**2)


def test_conditional_histogram_splits_give_hypergeometric_not_binomial_masses():
    for size, events, draws in [(11, 4, 5), (11, 9, 7), (2**54, 2, 2)]:
        urn = model._Urn(size=size, events=events, draws=draws)
        remaining = 1.0
        actual = {}
        for y, split in urn.bins():
            actual[y] = remaining * split
            remaining -= actual[y]
        expected = {
            y: comb(events, y) * comb(size - events, draws - y) / comb(size, draws)
            for y in range(max(0, events + draws - size), min(events, draws) + 1)
        }
        assert actual == pytest.approx(expected, rel=1e-14, abs=0)


def test_billion_member_group_with_two_offers_has_the_exact_untreated_count_law():
    # 0, 1 or 2 of the two offered people are events; all remaining events are untreated.
    n = 2_400_000_000
    events = n // 2
    pop = _population((10, 10, 10, n), offers=((0, 2),) * 4)
    case = model._Scenario("spread", "unchanged", (3, 3, 3, events), (3, 3, 3, events), 0.0, None)
    observed = np.zeros(3, dtype=int)
    for _, assignment, counts in model._states(pop, pop.design.comparisons[0], (case,), 16_384):
        assert np.all(counts[assignment[:, -1] == 0, -1] == events)
        offered_events = events - counts[assignment[:, -1] == 1, -1]
        observed += np.bincount(offered_events, minlength=3)
    expected = [comb(events, i) * comb(n - events, 2 - i) / comb(n, 2) for i in range(3)]
    np.testing.assert_allclose(observed / observed.sum(), expected, atol=0.025)


def test_huge_wide_offer_draw_retains_an_unqualified_plan_before_simulating():
    n = 2_400_000_000
    comparison = _comparison(rate=0.25, effect=0.01)
    pop = _population(
        (20, 20, 20, n), comparisons=(comparison,), offers=((0, 2),) * 3 + ((0, n // 2),)
    )
    plan = model.evaluate_design(pop, comparison)
    assert not plan.calibration_passed and plan.scenarios == ()
    assert plan.reason == "group_draw_exceeds_planning_limit"
