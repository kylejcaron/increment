"""Adaptive logger simulators: exact logging law, version monotonicity, reward closure, seeds.

The epsilon-greedy and Thompson oracles below recompute each version's
table from the trace's own closed rewards, so a propensity that drifted
from the registered law would fail here rather than be silently admitted.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from increment.errors import InvalidRequestError
from increment.logged_policy import (
    PROPENSITY_FLOOR,
    REFERENCE_POLICY_V1,
    TARGET_POLICY_V1,
    LoggedTrace,
    TabularPolicy,
)
from increment.simulate.bandit_loggers import (
    ACTIONS,
    CONTEXTS,
    PERIOD,
    REWARD_DELAY,
    START,
    LoggedRun,
    RewardModel,
    beta_exceedance,
    simulate_logged_run,
    simulate_logged_trace,
    true_policy_value,
)

LOGGERS = ("fixed_random", "epsilon_greedy", "thompson", "contextual_thompson")


def _by_time(trace: LoggedTrace):
    return sorted(trace.records, key=lambda r: (r.decision_time, r.unit_id))


def _closed_counts(records, before):
    counts = {(x, a): [0, 0] for x in CONTEXTS for a in ACTIONS}
    for r in records:
        if r.reward_observation_boundary < before:
            key = (r.pre_decision_context["x"], r.chosen_action)
            counts[key][0] += int(r.reward)
            counts[key][1] += 1
    return counts


@pytest.mark.parametrize("logger", LOGGERS)
def test_every_propensity_is_the_registered_table_probability(logger):
    run = simulate_logged_run(logger, n_units=30, horizon=3, seed=7, batch_size=20)
    trace = run.trace
    for k, record in enumerate(trace.records):
        policy = run.registry.get(record.logging_policy_id, record.logging_policy_version)
        assert policy is not None
        assert record.propensity == policy.probability(
            record.chosen_action, record.pre_decision_context
        )
        chosen = trace.candidate_actions.index(record.chosen_action)
        assert record.propensity == trace.logging_distributions[k][chosen]
        assert record.reward in (0.0, 1.0)
        assert record.reward_observation_boundary == record.decision_time + REWARD_DELAY
        assert record.decision_time == START + (record.decision_index - 1) * PERIOD


@pytest.mark.parametrize("logger", LOGGERS[1:])
def test_versions_are_monotone_in_time_and_batches_follow_versions(logger):
    trace = simulate_logged_trace(logger, n_units=30, horizon=3, seed=7, batch_size=20)
    versions = [int(r.logging_policy_version[1:]) for r in _by_time(trace)]
    assert versions == sorted(versions)
    # 90 decisions / batch 20 -> updates after 20, 40, 60, 80 -> v1..v5
    assert sorted(set(versions)) == [1, 2, 3, 4, 5]
    prefix = {"epsilon_greedy": "eg-b", "thompson": "ts-b", "contextual_thompson": "cts-b"}[logger]
    assert trace.update_batches == tuple(f"{prefix}{k:03d}" for k in range(1, 6))
    assert trace.logging_policy_versions == tuple(
        f"{trace.records[0].logging_policy_id}/v{k}" for k in range(1, 6)
    )


def test_fixed_random_logs_exact_halves_and_never_updates():
    trace = simulate_logged_trace("fixed_random", n_units=25, horizon=4, seed=1, batch_size=5)
    assert {r.propensity for r in trace.records} == {0.5}
    assert trace.logging_policy_versions == ("fixed-random/v1",)
    assert trace.update_batches == ("fixed-b000",)


def test_a_posterior_that_crosses_the_floor_refuses_instead_of_clipping():
    # A large effect lets the contextual posterior separate the arms within a few
    # batches; the first sub-floor draw refuses with the trace's own code.
    with pytest.raises(InvalidRequestError) as raised:
        simulate_logged_trace(
            "contextual_thompson", n_units=50, horizon=5, seed=11, reward_effect=0.3
        )
    assert raised.value.code == "logged_policy.trace.propensity_below_floor"
    assert cast("float", raised.value.context["propensity"]) < PROPENSITY_FLOOR


def test_epsilon_greedy_versions_are_fitted_only_from_rewards_closed_before_the_update():
    trace = simulate_logged_trace("epsilon_greedy", n_units=30, horizon=3, seed=7, batch_size=20)
    ordered = _by_time(trace)
    first_use = {}
    for r in ordered:
        first_use.setdefault(r.logging_policy_version, r.decision_time)
    for record in ordered:
        counts = _closed_counts(ordered, before=first_use[record.logging_policy_version])
        x = record.pre_decision_context["x"]
        means = {a: (counts[(x, a)][0] + 1) / (counts[(x, a)][1] + 2) for a in ACTIONS}
        greedy = "A" if means["A"] >= means["B"] else "B"
        assert record.propensity == (0.9 if record.chosen_action == greedy else 0.1)
    # An update inside period t sees no period-t rewards: v2 starts at decision 21 of
    # period 1 (30 units per period), so its table is the empty-data table.
    assert first_use["v2"] == START
    assert _closed_counts(ordered, before=START) == {k: [0, 0] for k in _closed_counts([], START)}


@pytest.mark.parametrize("contextual", [False, True])
def test_thompson_propensities_are_exact_beta_exceedances_of_closed_counts(contextual):
    logger = "contextual_thompson" if contextual else "thompson"
    trace = simulate_logged_trace(logger, n_units=30, horizon=3, seed=7, batch_size=20)
    ordered = _by_time(trace)
    first_use = {}
    for r in ordered:
        first_use.setdefault(r.logging_policy_version, r.decision_time)
    for record in ordered:
        counts = _closed_counts(ordered, before=first_use[record.logging_policy_version])
        pool = (record.pre_decision_context["x"],) if contextual else CONTEXTS
        s = {a: sum(counts[(c, a)][0] for c in pool) for a in ACTIONS}
        n = {a: sum(counts[(c, a)][1] for c in pool) for a in ACTIONS}
        p_b = beta_exceedance(1 + s["B"], 1 + n["B"] - s["B"], 1 + s["A"], 1 + n["A"] - s["A"])
        expected = p_b if record.chosen_action == "B" else 1.0 - p_b
        assert record.propensity == expected


def test_beta_exceedance_matches_exact_values_and_symmetry():
    cases = (
        ((5, 3, 2, 6), 0.9487179487179487),
        ((40, 60, 55, 45), 0.01626017451283473),
        ((1, 1, 1, 1), 0.5),
    )
    for params, expected in cases:
        ax, bx, ay, by = params
        assert beta_exceedance(*params) == pytest.approx(expected, abs=1e-13)
        assert beta_exceedance(ax, bx, ay, by) + beta_exceedance(ay, by, ax, bx) == pytest.approx(
            1.0, abs=1e-12
        )


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_beta_exceedance_matches_monte_carlo_and_symmetry():
    rng = np.random.default_rng(0)
    for params in ((5, 3, 2, 6), (40, 60, 55, 45), (3, 1, 1, 3)):
        ax, bx, ay, by = params
        mc = float(np.mean(rng.beta(ax, bx, 200_000) > rng.beta(ay, by, 200_000)))
        assert beta_exceedance(*params) == pytest.approx(mc, abs=0.004)
        assert beta_exceedance(ax, bx, ay, by) + beta_exceedance(ay, by, ax, bx) == pytest.approx(
            1.0
        )
    assert beta_exceedance(1, 1, 1, 1) == 0.5


def test_same_seed_reproduces_the_trace_and_a_different_seed_does_not():
    a = simulate_logged_trace("thompson", n_units=25, horizon=2, seed=3, batch_size=10)
    b = simulate_logged_trace("thompson", n_units=25, horizon=2, seed=3, batch_size=10)
    c = simulate_logged_trace("thompson", n_units=25, horizon=2, seed=4, batch_size=10)
    assert a == b
    assert a != c


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_context_dependence_makes_later_contexts_follow_the_previous_action():
    trace = simulate_logged_trace(
        "fixed_random", n_units=2000, horizon=2, seed=5, context_dependence=0.8
    )
    by_unit = {}
    for r in trace.records:
        by_unit.setdefault(r.unit_id, {})[r.decision_index] = r
    after_b = [
        u[2].pre_decision_context["x"] for u in by_unit.values() if u[1].chosen_action == "B"
    ]
    after_a = [
        u[2].pre_decision_context["x"] for u in by_unit.values() if u[1].chosen_action == "A"
    ]
    # P(x_2 = 1 | a_1 = B) = 0.9, P(x_2 = 1 | a_1 = A) = 0.5; binomial noise ~ 0.016 at n~1000
    assert np.mean(after_b) == pytest.approx(0.9, abs=0.05)
    assert np.mean(after_a) == pytest.approx(0.5, abs=0.05)
    assert trace.records[1].pre_decision_context["prior"] == trace.records[0].reward


def test_true_policy_value_closed_forms():
    rm = RewardModel(effect=0.1)
    assert true_policy_value(TARGET_POLICY_V1, rm) == pytest.approx(0.43)
    assert true_policy_value(REFERENCE_POLICY_V1, rm) == pytest.approx(0.40)
    for horizon in (1, 2, 5):
        assert true_policy_value(TARGET_POLICY_V1, rm, horizon=horizon) == pytest.approx(0.43)
    dynamic = RewardModel(effect=0.1, context_dependence=0.8)
    # Two-state Markov recursion: kernel rows target (0.82, 0.58), reference (0.70, 0.70).
    assert true_policy_value(TARGET_POLICY_V1, dynamic, horizon=1) == pytest.approx(0.43)
    assert true_policy_value(TARGET_POLICY_V1, dynamic, horizon=2) == pytest.approx(0.44)
    assert true_policy_value(TARGET_POLICY_V1, dynamic, horizon=3) == pytest.approx(0.4417333333333)
    assert true_policy_value(REFERENCE_POLICY_V1, dynamic, horizon=3) == pytest.approx(
        0.4133333333333
    )
    always_b = TabularPolicy(policy_id="always", version="B", default={"A": 0.0, "B": 1.0})
    assert true_policy_value(always_b, rm) == pytest.approx(0.5 * 0.4 + 0.5 * 0.4)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_true_policy_value_by_on_policy_rollout():
    rm = RewardModel(effect=0.1, context_dependence=0.8)
    rng = np.random.default_rng(1)
    n, horizon = 20_000, 3
    total = 0.0
    for _ in range(n):
        p_one = rm.context_probability
        for _t in range(horizon):
            x = int(rng.random() < p_one)
            action = "B" if rng.random() < TARGET_POLICY_V1.probability("B", {"x": x}) else "A"
            total += rm.mean(action, x)
            p_one = rm.next_context_probability(action)
    # Step means lie in [0.3, 0.5], so the MC standard error is below 0.001 at 60k steps.
    assert total / (n * horizon) == pytest.approx(
        true_policy_value(TARGET_POLICY_V1, rm, horizon=3), abs=0.008
    )


def test_invalid_arguments_are_coded_refusals():
    with pytest.raises(InvalidRequestError) as raised:
        simulate_logged_trace("fixed_random", n_units=0, horizon=1, seed=0)
    assert raised.value.code == "simulate.bandit_loggers.count_positive"
    assert raised.value.context["name"] == "n_units"
    assert raised.value.context["value"] == 0
    with pytest.raises(InvalidRequestError) as raised:
        simulate_logged_trace("epsilon_greedy", n_units=5, horizon=1, seed=0, epsilon=0.0)
    assert raised.value.code == "simulate.bandit_loggers.epsilon"
    with pytest.raises(InvalidRequestError) as raised:
        RewardModel(effect=0.8)
    assert raised.value.code == "simulate.bandit_loggers.reward_mean_out_of_range"
    assert raised.value.context["action"] == "B"
    assert raised.value.context["x"] == 0
    assert raised.value.context["mean"] == pytest.approx(1.1)
    assert isinstance(simulate_logged_run("fixed_random", n_units=3, horizon=1, seed=0), LoggedRun)
    assert math.isclose(RewardModel().mean("B", 1), 0.4)
