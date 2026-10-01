"""The planner must size n that achieves its nominal power under the
boundary the runtime actually executes -- the OBF K=14 example from the
review, corrected, plus a Monte Carlo that simulates the runtime's own
shifted linear contrast rather than a log-scale approximation."""

import math

import numpy as np
import pytest

from increment import PowerDesign
from increment.estimation.sequential import GaussianScoreMixture
from increment.power import Baseline, achieved_power, required_sample_size
from increment.semantics.models import InferenceSpec
from tests.power._procedures import make_procedure

ALPHA = 0.05
LOOKS = 14
BASELINE = Baseline(mean=20.0, var=400.0)
PROCEDURE = make_procedure(inference=GaussianScoreMixture(), alpha=ALPHA, population="assigned")


def test_required_n_reaches_nominal_power_under_the_same_boundary():
    """Fast analytic smoke: true by construction once planning and runtime
    share one boundary (P1), pinned so a regression cannot silently reopen
    the reported gap (n=6567 at ~0.50 power, not 0.80)."""
    sized = required_sample_size(
        relative_lift=0.05,
        baseline=BASELINE,
        procedure=PROCEDURE,
        design=PowerDesign(power=0.80),
        planned_looks=LOOKS,
    )
    assert sized.power >= 0.80
    assert sized.inference_to_declare == InferenceSpec(
        kind="asymptotic_mean", expected_decision_sample_size=sized.n_total
    )
    under_provisioned = achieved_power(
        n_per_arm=6567,
        relative_lift=0.05,
        baseline=BASELINE,
        procedure=PROCEDURE,
        planned_looks=LOOKS,
    )
    assert under_provisioned.power < 0.55
    assert sized.n_per_arm > 6567


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_planned_n_achieves_nominal_power_under_the_runtime_boundary():
    """Bounded, memory-chunked Monte Carlo: calls the runtime's own
    count_boundary at rho derived from the planner's declared N
    (inference_to_declare.expected_decision_sample_size), and tests the
    exact rejection rule asymptotic_mean_set's quadratic reduces to at
    the null (r=1, null_lift=0): reject iff (m_t - m_c)^2 > K * (v_c +
    v_t) -- algebraically identical to _component_for_quadratic(mc*mc -
    K*vc, -2*mt*mc, mt*mt - K*vt) excluding r=1
    (estimation/asymptotic_mean.py:279-292), not a reimplementation of
    the planner's own boundary. Reps are processed in memory-bounded
    batches, accumulating only running per-arm sums/sums-of-squares per
    look (never a full reps x n array) so memory stays O(batch_size)."""
    from fractions import Fraction

    from increment.estimation.asymptotic_mean import (
        count_boundary,
        directional_alpha,
        mixture_r_star,
    )

    rng = np.random.default_rng(20260922)
    total_reps = 4000
    batch_size = 500
    sized = required_sample_size(
        relative_lift=0.05,
        baseline=BASELINE,
        procedure=PROCEDURE,
        design=PowerDesign(power=0.80),
        planned_looks=LOOKS,
    )
    n_per_arm = sized.n_per_arm
    n_total = sized.n_total
    assert sized.inference_to_declare is not None
    declared_n = sized.inference_to_declare.expected_decision_sample_size
    assert declared_n is not None
    raw_alpha = Fraction(ALPHA)
    rho = Fraction(math.sqrt(float(mixture_r_star(raw_alpha) / declared_n)))
    boundary_alpha = directional_alpha(raw_alpha, "two-sided")
    true_mean_t = BASELINE.mean * 1.05
    sd_c = math.sqrt(BASELINE.var)
    sd_t = math.sqrt(BASELINE.var)  # planning's own equal-variance assumption

    fractions = np.array([k / LOOKS for k in range(1, LOOKS + 1)])
    n_c_targets = np.round(fractions * n_per_arm).astype(int)
    n_t_targets = np.round(fractions * n_per_arm).astype(int)
    n_c_targets[n_c_targets < 1] = 1
    n_t_targets[n_t_targets < 1] = 1
    counts = n_c_targets + n_t_targets  # the runtime's own "count" clock
    boundaries_sq = np.array([float(count_boundary(int(n), boundary_alpha, rho)) for n in counts])

    total_crossed = 0
    remaining = total_reps
    while remaining > 0:
        batch = min(batch_size, remaining)
        remaining -= batch
        sum_c = np.zeros(batch)
        sumsq_c = np.zeros(batch)
        sum_t = np.zeros(batch)
        sumsq_t = np.zeros(batch)
        n_c_so_far = 0
        n_t_so_far = 0
        crossed = np.zeros(batch, dtype=bool)
        for _look, (n_c_target, n_t_target, k_bound) in enumerate(
            zip(n_c_targets, n_t_targets, boundaries_sq, strict=True)
        ):
            inc_c = n_c_target - n_c_so_far
            inc_t = n_t_target - n_t_so_far
            if inc_c > 0:
                new_c = rng.normal(BASELINE.mean, sd_c, size=(batch, inc_c))
                sum_c += new_c.sum(axis=1)
                sumsq_c += (new_c**2).sum(axis=1)
                n_c_so_far = n_c_target
            if inc_t > 0:
                new_t = rng.normal(true_mean_t, sd_t, size=(batch, inc_t))
                sum_t += new_t.sum(axis=1)
                sumsq_t += (new_t**2).sum(axis=1)
                n_t_so_far = n_t_target
            mc = sum_c / n_c_so_far
            mt = sum_t / n_t_so_far
            vc = (sumsq_c / n_c_so_far - mc**2) * n_c_so_far / (n_c_so_far - 1) / n_c_so_far
            vt = (sumsq_t / n_t_so_far - mt**2) * n_t_so_far / (n_t_so_far - 1) / n_t_so_far
            crossed |= (mt - mc) ** 2 > k_bound * (vc + vt)
        total_crossed += int(crossed.sum())

    empirical_power = total_crossed / total_reps
    mcse = math.sqrt(empirical_power * (1 - empirical_power) / total_reps)
    assert empirical_power >= 0.80 - 3 * mcse
    assert n_total > 6567 * 2
