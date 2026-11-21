"""Operating characteristics for the prospective two-stage binary model."""

from dataclasses import FrozenInstanceError, replace
from math import comb

import numpy as np
import pytest
from scipy.stats import binom

from increment._spillover_design import SpilloverDesign, SpilloverGroup
from increment.errors import CodedError
from increment.estimation.spillover import summarize_counts
from increment.power import _spillover as model


def _design(sizes=(10, 10, 10, 10), *, strata=None, high=None):
    strata = strata or ("all",) * len(sizes)
    return SpilloverDesign(
        groups=tuple(
            SpilloverGroup(
                group_id=f"g{i:04}",
                members=tuple(f"{i}/{j}" for j in range(n)),
                low_treated=1,
                high_treated=n - 2,
                stratum=s,
            )
            for i, (n, s) in enumerate(zip(sizes, strata, strict=True))
        ),
        high_groups=high or {"all": len(sizes) // 2},
        outcome="renewed",
        horizon="90 days",
    )


def test_offsetting_and_signed_schedules_move_only_eligible_people():
    sizes, low = (10,) * 4, (1, 2, 4, 5)
    for delta, spread, concentrated in [
        (4, (2, 3, 5, 6), (5, 2, 4, 5)),
        (-4, (1, 1, 3, 3), (1, 2, 4, 1)),
    ]:
        cases = {s.schedule: s for s in model._schedules("spread", sizes, low, delta)}
        assert {key: s.high for key, s in cases.items()} == {
            "unchanged": low,
            "cancelling": (10, 2, 0, 0),
            "spread": spread,
            "concentrated": concentrated,
        }
        assert cases["cancelling"].cancelling_pairs == 9
        assert all(s.true_effect == (sum(s.high) - sum(low)) / 40 for s in cases.values())

    unequal = model._schedules("spread", (100, 10, 40, 20), (10, 3, 20, 4), 1)
    cancelling = next(s for s in unequal if s.schedule == "cancelling")
    assert cancelling.high == (33, 0, 0, 4)  # rank by event fraction, not count


def test_baseline_profiles_and_half_up_rounding_use_the_entire_roster():
    # 0.145 * 100 is a half-person tie, for both the baseline and the alternative.
    cases = model._model_grid((10, 20, 30, 40), 0.145, -0.145)
    assert not isinstance(cases, str)
    assert len({(s.baseline_profile, s.schedule) for s in cases}) == 12
    assert {s.baseline_profile: s.low for s in cases} == {
        "spread": (2, 3, 4, 6),
        "largest_first": (0, 0, 0, 15),
        "smallest_first": (10, 5, 0, 0),
    }
    assert all(s.true_effect == -0.15 for s in cases if s.schedule in {"spread", "concentrated"})


def test_rare_baseline_does_not_cap_a_feasible_meaningful_increase():
    cases = model._model_grid((10,) * 1000, 0.01, 0.05)
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


@pytest.mark.parametrize("hits", [0, 7, 37])
def test_exact_bounds_invert_each_of_the_24_one_sided_tails(hits):
    lower, upper = model._lower(hits, 37), model._upper(hits, 37)
    if hits:
        assert binom.sf(hits - 1, 37, lower) == pytest.approx(0.05 / 24)
    else:
        assert lower == 0
    if hits < 37:
        assert binom.cdf(hits, 37, upper) == pytest.approx(0.05 / 24)
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
        model.evaluate_design(_design(), rate, effect)
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
    result = model.evaluate_design(_design(), rate, effect)
    assert not result.calibration_passed
    assert (result.status, result.reason, result.scenarios) == ("not_calibrated", reason, ())


def test_scarcity_and_operational_limit_are_distinct_from_bad_calibration():
    for design, reason in [
        (_design((6,) * 5, high={"all": 1}), "too_few_groups"),
        (_design((4,) * 2049), "simulation_budget_exceeded"),
    ]:
        result = model.evaluate_design(design, 0.4, 0.1)
        assert (result.reason, result.scenarios) == (reason, ())


def test_sampled_states_follow_both_quota_laws_and_reconcile_to_the_logged_contrast():
    sizes = (6, 7, 8, 9, 6, 7, 8, 9, 10)
    design = _design(sizes, strata=("a",) * 4 + ("b",) * 5, high={"a": 2, "b": 3})
    low, high_events = (3, 0, 8, 0, 0, 7, 0, 9, 0), (4,) * 9
    state = model._Scenario("spread", "spread", low, high_events, 9 / sum(sizes), None)
    histogram = np.zeros((2, 4), dtype=int)
    for _, high, counts in model._states(design, (state,), 8192):
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

    _, high, counts = next(model._states(design, (state,), 1))
    _, repeated_high, repeated_counts = next(model._states(design, (state,), 1))
    np.testing.assert_array_equal(repeated_high, high)
    np.testing.assert_array_equal(repeated_counts, counts)
    flags, assignments, outcomes = {}, {}, {}
    for g, flag, count in zip(design.groups, high[0], counts[0], strict=True):
        flags[g.group_id] = bool(flag)
        quota = g.high_treated if flag else g.low_treated
        for j, person in enumerate(g.members):
            assignments[person] = j < quota
            if j >= quota:
                outcomes[person] = bool(j - quota < count)
    logged_high, logged_successes = design.logged_counts(
        group_high=flags, treated=assignments, outcomes=outcomes
    )
    simulated = summarize_counts(design, high, counts)
    logged = summarize_counts(design, logged_high.reshape(1, -1), logged_successes.reshape(1, -1))
    np.testing.assert_equal(simulated.estimate, logged.estimate)
    np.testing.assert_equal(simulated.standard_error, logged.standard_error)
    np.testing.assert_equal(simulated.reason, logged.reason)


def test_unavailable_draws_count_as_misses_but_not_false_alarms():
    # Of ten allocations, only assigning both all-zero groups high has zero SE.
    # Choosing two all-success groups gives estimate 2/3 and SE 1/3 (3 false alarms);
    # choosing one of each gives an interval containing zero (6 allocations).
    events = (0, 0, 4, 4, 4)
    null = model._Scenario("spread", "unchanged", events, events, 0.0, None)
    all_zero = replace(null, low=(0,) * 5, high=(0,) * 5)
    evidence, no_events = model._measure(_design((4,) * 5), (null, all_zero), 4096)
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
        estimates = np.array([effect, -effect, 0.0, effect, np.sign(effect) * model.Z95 * 0.1])
        tally = model._Tally()
        tally.observe(
            estimates,
            np.array([0.01, 0.01, 0.01, 0.0, 0.1]),
            np.array([True, True, True, False, True]),
            effect,
        )
        high = (2 if effect > 0 else 0,)
        case = model._Scenario("spread", "concentrated", (1,), high, effect, None)
        record = tally.evidence(case)
        assert (record.attempts, record.misses, record.unavailable) == (5, 3, 1)
        assert (record.correct_detections, record.wrong_direction_detections) == (1, 1)
        assert record.false_calls is None
        assert record.detection_upper == pytest.approx(model._upper(1, 5))


def test_a_single_adverse_profile_controls_the_joint_decision():
    grid = model._model_grid((10,) * 4, 0.3, 0.1)
    assert not isinstance(grid, str)
    # At the simultaneous CP tail, 3,437 is the last passing miss count.
    good_null = model._Tally(attempts=65_536, misses=3437, positive=3437)
    good_alt = model._Tally(attempts=65_536, misses=3437, positive=60_000)
    rows = tuple((good_alt if case.true_effect else good_null).evidence(case) for case in grid)
    result = model._conclude(rows)
    assert result.calibration_passed and result.status == "supported" and result.reason is None
    for obj, field in [(rows[0], "misses"), (result, "status")]:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)
    adverse = (
        (replace(rows[-1], detection_upper=0.79, detection_lower=0.78), "underpowered"),
        (replace(rows[-1], detection_lower=0.79, detection_upper=0.8), "inconclusive"),
        (replace(good_alt, misses=3438).evidence(grid[-1]), "not_calibrated"),
        (replace(rows[-1], detection_lower=None), "not_calibrated"),
    )
    for row, status in adverse:
        decision = model._conclude((*rows[:-1], row))
        assert decision.status == status and decision.reason
        assert decision.calibration_passed == (status != "not_calibrated")
    assert not model._conclude(rows[:1]).calibration_passed


@pytest.mark.slow
def test_sparse_design_keeps_all_labelled_schedules_and_all_attempts():
    result = model.evaluate_design(_design(), baseline_rate=0.025, meaningful_effect=0.1)
    assert (result.status, result.reason) == ("not_calibrated", "miss_bound_exceeded")
    assert len({row.name for row in result.scenarios}) == 12
    for row in result.scenarios:
        assert row.attempts == 65_536
        if row.true_effect == 0:
            assert row.false_calls is not None
            assert row.misses == row.unavailable + row.false_calls
        else:
            assert row.false_calls is None
