"""Monte-Carlo calibration of the trajectory Hajek contrast interval under a fixed logger.

Pre-registered before any outcome was inspected:

* Randomization: ``simulate_logged_trace`` (time-major decisions; context,
  action, reward drawn per decision from one ``numpy.random.default_rng``
  stream seeded per replication as ``seed0 + r``) under the fixed 0.5/0.5
  logger, the only logging law the estimator admits.
* Truth: ``Delta_T = true_policy_value(target) - true_policy_value(reference)``
  from the closed-form Markov recursion; ``0.03`` at ``context_dependence=0``
  and ``0.0284`` at ``context_dependence=0.8, T=3``.
* Units: ``n_units`` independent trajectories; no clusters beyond the unit.
* Statistic: the Bernoulli coverage indicator ``lb <= Delta_T <= ub`` of the
  two-sided 95% interval, and the replication mean of ``estimate``.
* Tolerance: coverage inside ``nominal_band(0.95, reps, k=3)`` (binomial
  MCSE ``sqrt(0.95 * 0.05 / reps)``); ``|mean - truth| <= 3 * sd / sqrt(reps)``.
* Missing intervals: every refusal is counted and the test fails if any
  replication refuses, so an always-refusing implementation cannot pass.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.mc import Coverage, nominal_band

if TYPE_CHECKING:
    from increment.simulate.bandit_loggers import Logger

TRUTH_STATIC = 0.03


def _study(
    logger: Logger,
    *,
    n_units: int,
    horizon: int,
    reps: int,
    seed0: int,
    context_dependence: float = 0.0,
):
    from increment.errors import CodedError
    from increment.logged_policy import (
        REFERENCE_POLICY_V1,
        TARGET_POLICY_V1,
        estimate_policy_contrast,
    )
    from increment.simulate.bandit_loggers import (
        RewardModel,
        simulate_logged_trace,
        true_policy_value,
    )

    reward_model = RewardModel(effect=0.1, context_dependence=context_dependence)
    truth = true_policy_value(TARGET_POLICY_V1, reward_model, horizon=horizon) - true_policy_value(
        REFERENCE_POLICY_V1, reward_model, horizon=horizon
    )
    coverage = Coverage()
    estimates, refusals = [], []
    for r in range(reps):
        trace = simulate_logged_trace(
            logger,
            n_units=n_units,
            horizon=horizon,
            seed=seed0 + r,
            context_dependence=context_dependence,
        )
        try:
            result = estimate_policy_contrast(trace, TARGET_POLICY_V1, REFERENCE_POLICY_V1)
        except CodedError as exc:
            refusals.append(exc.code)
            continue
        coverage.record(result.lb <= truth <= result.ub)
        estimates.append(result.estimate)
    assert not refusals, f"{len(refusals)} replications refused: {sorted(set(refusals))}"
    return truth, coverage.rate, np.array(estimates)


def _assert_calibrated(truth: float, rate: float, estimates: np.ndarray, reps: int) -> None:
    lo, hi = nominal_band(0.95, reps, k=3.0)
    assert lo <= rate <= hi, (rate, lo, hi)
    mcse_mean = estimates.std(ddof=1) / math.sqrt(reps)
    assert abs(estimates.mean() - truth) <= 3.0 * mcse_mean, (estimates.mean(), truth, mcse_mean)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_fixed_random_logger_interval_is_nominal_at_horizon_three():
    truth, rate, estimates = _study("fixed_random", n_units=200, horizon=3, reps=400, seed0=10_000)
    assert truth == pytest.approx(TRUTH_STATIC)
    _assert_calibrated(truth, rate, estimates, reps=400)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_history_dependent_contexts_are_recovered_by_the_cumulative_product():
    truth, rate, estimates = _study(
        "fixed_random", n_units=200, horizon=3, reps=400, seed0=40_000, context_dependence=0.8
    )
    assert truth == pytest.approx(0.0284)
    _assert_calibrated(truth, rate, estimates, reps=400)


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_fixed_random_logger_interval_is_nominal_smoke():
    """Small-N twin of the full studies under the same pre-registered rule;
    its thirty replications exceed the fast tier's per-test budget."""
    truth, rate, estimates = _study("fixed_random", n_units=60, horizon=2, reps=30, seed0=20_000)
    _assert_calibrated(truth, rate, estimates, reps=30)
