"""Calibration for one-way factor absorption.

Every ``@pytest.mark.parameter_recovery`` test here is simulation-based and
excluded from the fast suite. Thresholds are floors with margin against
measurements taken when this suite was written, not exact reproductions;
a change that moves them is a real regression. ``test_simulate_fixture_shape``
below is the one exception: an unmarked, small-N smoke test so a signature
break in the simulation helpers is caught by the fast suite too (AGENTS.md).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from increment.estimation.absorption import absorb_one_way
from tests.mc import Coverage

TAU = 0.20


def _simulate(rng, n_levels, per_level, alpha_sd, p_treat, het=1.0):
    """Cell moments for a one-way design with optional arm heteroskedasticity."""
    n_c = np.zeros(n_levels)
    s_c = np.zeros(n_levels)
    q_c = np.zeros(n_levels)
    n_t = np.zeros(n_levels)
    s_t = np.zeros(n_levels)
    q_t = np.zeros(n_levels)
    alpha = rng.normal(0.0, alpha_sd, n_levels)
    for k in range(n_levels):
        d = rng.random(per_level) < p_treat
        noise = rng.normal(0.0, 1.0, per_level) * np.where(d, het, 1.0)
        y = alpha[k] + TAU * d + noise
        yc, yt = y[~d], y[d]
        n_c[k], s_c[k], q_c[k] = yc.size, yc.sum(), (yc**2).sum()
        n_t[k], s_t[k], q_t[k] = yt.size, yt.sum(), (yt**2).sum()
    return n_c, s_c, q_c, n_t, s_t, q_t


def _simulate_bernoulli(
    rng, n_levels, per_level, alpha_sd_logit, tau, p_base, p_treat: float = 0.5
):
    """Bernoulli outcome with a logit-normal per-level factor effect on the
    CONTROL probability. An additive-on-[0,1] normal saturates at the
    boundary when p_base is small and alpha_sd is non-trivial (verified
    degenerate at p_base=0.10, alpha_sd=1.0: coverage collapses to ~55%
    from boundary clipping alone) - the logit scale avoids this."""
    n_c = np.zeros(n_levels)
    s_c = np.zeros(n_levels)
    q_c = np.zeros(n_levels)
    n_t = np.zeros(n_levels)
    s_t = np.zeros(n_levels)
    q_t = np.zeros(n_levels)
    logit0 = np.log(p_base / (1 - p_base))
    logit_k = logit0 + rng.normal(0.0, alpha_sd_logit, n_levels)
    p0_k = 1.0 / (1.0 + np.exp(-logit_k))
    for k in range(n_levels):
        d = rng.random(per_level) < p_treat
        p0 = p0_k[k]
        p1 = p0 + tau
        assert np.all((p1 > 0) & (p1 < 1)), (
            "tau pushed p1 out of (0,1) -- fixture is silently wrong, "
            "adjust p_base/tau/alpha_sd_logit"
        )
        y = np.where(d, rng.random(per_level) < p1, rng.random(per_level) < p0).astype(float)
        yc, yt = y[~d], y[d]
        n_c[k], s_c[k], q_c[k] = yc.size, yc.sum(), (yc**2).sum()
        n_t[k], s_t[k], q_t[k] = yt.size, yt.sum(), (yt**2).sum()
    return n_c, s_c, q_c, n_t, s_t, q_t


def _coverage(reps, seed, sim_fn=_simulate, **kw):
    rng = np.random.default_rng(seed)
    cov = Coverage()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"treatment share varies across factor levels",
            category=UserWarning,
        )
        for _ in range(reps):
            try:
                res = absorb_one_way(*sim_fn(rng, **kw))
            except ValueError:
                continue
            cov.record(res.lb <= TAU <= res.ub)
    return cov.rate, cov.reps


def test_simulate_fixture_shape():
    rng = np.random.default_rng(0)
    n_c, s_c, q_c, n_t, s_t, q_t = _simulate(
        rng, n_levels=6, per_level=8, alpha_sd=0.5, p_treat=0.5
    )
    assert n_c.shape == (6,)
    assert np.all(n_c + n_t == 8)

    n_c, s_c, q_c, n_t, s_t, q_t = _simulate_bernoulli(
        rng, n_levels=4, per_level=8, alpha_sd_logit=0.5, tau=0.05, p_base=0.10, p_treat=0.5
    )
    assert n_c.shape == (4,)
    assert np.all(n_c + n_t == 8)
    assert np.all(s_c <= n_c)
    assert np.all(s_t <= n_t)


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestCoverage:
    def test_balanced_homoskedastic(self):
        cov, used = _coverage(800, 11, n_levels=50, per_level=80, alpha_sd=1.0, p_treat=0.5)
        assert used == 800, f"used {used}"
        assert 0.92 <= cov <= 0.98

    @pytest.mark.parametrize("p_treat", [0.5, 0.3, 0.1])
    def test_skewed_allocation_under_heteroskedasticity(self, p_treat):
        """The coverage trap. A homoskedastic variance gives 63.8% at 90/10.

        This is the single highest-risk regression in the feature: if the
        sandwich is ever replaced by a pooled sigma^2, this test is what
        catches it.
        """
        cov, used = _coverage(
            800, 22, n_levels=50, per_level=80, alpha_sd=1.0, p_treat=p_treat, het=3.0
        )
        assert used == 800, f"used {used}"
        assert cov >= 0.92, f"coverage {cov:.3f} at p_treat={p_treat}"

    def test_small_number_of_levels_still_covers(self):
        """t_{K-2} rather than z is what keeps this nominal at K=20."""
        cov, used = _coverage(
            800, 33, n_levels=20, per_level=80, alpha_sd=1.0, p_treat=0.1, het=3.0
        )
        assert used == 800, f"used {used}"
        assert cov >= 0.92

    @pytest.mark.parametrize("p_treat", [0.5, 0.1])
    def test_bernoulli_outcome_at_low_base_rate_covers(self, p_treat):
        """p_ctrl ~= 0.10 with real per-level heterogeneity on the logit
        scale (alpha_sd_logit=1.0, comparable to the Gaussian cases'
        alpha_sd=1.0). Parametrized over allocation to exercise the skewed
        regime too, since a Bernoulli outcome's Var = p(1-p) differs by
        construction between arms once p_ctrl != p_treat. Verified (seed 9,
        reps=1500): p_treat=0.5 gives coverage 0.9460, p_treat=0.1 gives
        0.9480 (used=1500 both)."""
        rng = np.random.default_rng(9)
        cov = Coverage()
        for _ in range(1500):
            cells = _simulate_bernoulli(
                rng,
                n_levels=50,
                per_level=80,
                alpha_sd_logit=1.0,
                tau=0.05,
                p_base=0.10,
                p_treat=p_treat,
            )
            try:
                res = absorb_one_way(*cells)
            except ValueError:
                continue
            cov.record(res.lb <= 0.05 <= res.ub)
        assert cov.reps == 1500
        assert cov.rate >= 0.92

    @pytest.mark.parametrize("n_levels,cov_hi", [(3, 0.9999), (5, 0.985), (10, 0.98), (20, 0.98)])
    def test_few_levels_still_covers(self, n_levels, cov_hi):
        """The corrected k/(k-1) cluster correction has its largest effect
        at small K (K=5: SE -13.4% vs the old k/(k-2) form), and
        _MIN_LEVELS=3 is the smallest the estimator accepts - verify coverage
        actually holds at the floor rather than assuming it from the K>=20
        cases.

        Measured (seed 13, reps=1500): K=3 0.9960, K=5 0.9687, K=10 0.9533,
        K=20 0.9413, increasingly conservative as K shrinks. The upper
        bound matters as much as the floor: reverting to the old k/(k-2)
        form only widens intervals, which only INCREASES coverage - a
        one-sided floor can't detect that regression. K=3's cap (0.9999) is
        a shape check only, rejecting the literal all-1500-covered case."""
        rng = np.random.default_rng(13)
        cov = Coverage()
        for _ in range(1500):
            cells = _simulate(rng, n_levels=n_levels, per_level=80, alpha_sd=1.0, p_treat=0.5)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"treatment share varies across factor levels",
                        category=UserWarning,
                    )
                    res = absorb_one_way(*cells)
            except ValueError:
                continue
            cov.record(res.lb <= TAU <= res.ub)
        assert cov.reps == 1500
        assert 0.92 <= cov.rate <= cov_hi


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPayoff:
    def test_se_reduction_tracks_the_factor_signal(self):
        """Measured SE cut (seed 44, 1000 reps per point - the ICC=0.038/0.80
        legs need this many to separate from the ICC=0 sandwich-vs-naive
        artifact): ~0.8% at ICC 0, ~2.0% at 0.038, ~9.9% at 0.20, ~28.3% at
        0.50, ~54.5% at 0.80. An independently measured reference table
        reads +0.9/+2.4/+10.6/+29.0/+54.8% for the same five points - agrees
        to within ~1pp everywhere, consistent Monte Carlo noise."""
        rng = np.random.default_rng(44)
        cuts = {}
        for alpha_sd in (0.0, 0.2, 0.5, 1.0, 2.0):
            vals = []
            for _ in range(1000):
                cells = _simulate(rng, n_levels=50, per_level=80, alpha_sd=alpha_sd, p_treat=0.5)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=r"treatment share varies across factor levels",
                        category=UserWarning,
                    )
                    res = absorb_one_way(*cells)
                vals.append(res.se_reduction)
            cuts[alpha_sd] = float(np.mean(vals))
        assert cuts[0.0] < 0.03, "a worthless factor must not buy precision"
        assert cuts[0.2] > 0.012, "must clear the ICC=0 sandwich-vs-naive artifact with margin"
        assert cuts[0.5] > 0.04
        assert cuts[1.0] > 0.20
        assert cuts[2.0] > 0.45, "the ICC=0.80 row must show its large payoff"
        assert cuts[0.0] < cuts[0.2] < cuts[0.5] < cuts[1.0] < cuts[2.0]

    def test_absorption_is_unbiased(self):
        rng = np.random.default_rng(55)
        est = [
            absorb_one_way(
                *_simulate(rng, n_levels=50, per_level=80, alpha_sd=1.0, p_treat=0.5)
            ).effect
            for _ in range(400)
        ]
        assert abs(float(np.mean(est)) - TAU) < 0.01


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPoolingTracksTheBetterEndpoint:
    """Partial pooling is not a guardrail against catastrophic failure:
    under the >=2-per-arm floor, hard absorption of a weak factor costs at
    most ~1-2% against no adjustment, never a runaway loss. Measured (n=50
    levels, per_level=8, p_treat=0.2, TAU=0.20, 1000 reps, seed 11):
    alpha_sd=0.3 (weak factor) hard/none SE ratio ~1.005-1.006; alpha_sd=1.0
    (strong factor) ratio ~0.878. Hard absorption is never meaningfully
    worse than no adjustment on a weak factor, and substantially better
    once the factor carries real signal.

    The weak-factor leg's three properties (hard-vs-none, partial-vs-
    endpoints, partial-unbiasedness) are asserted from ONE shared
    simulation run rather than three independently-seeded ones, since with
    identical parameters separate seeds add no independent evidence."""

    def test_hard_vs_none_and_partial_pooling_on_a_weak_thin_config(self):
        rng = np.random.default_rng(11)
        hard, part, none = [], [], []
        for _ in range(1000):
            cells = _simulate(rng, n_levels=50, per_level=8, alpha_sd=0.3, p_treat=0.2)
            try:
                hard.append(absorb_one_way(*cells, pooling="hard").effect)
                part.append(absorb_one_way(*cells, pooling="partial").effect)
                none.append(absorb_one_way(*cells, pooling="none").effect)
            except ValueError:
                continue
        sd_hard, sd_part, sd_none = np.std(hard), np.std(part), np.std(none)
        assert sd_hard <= sd_none * 1.05
        assert sd_part <= min(sd_hard, sd_none) * 1.02
        assert abs(float(np.mean(part)) - TAU) < 0.03

    def test_hard_absorption_wins_big_on_a_strong_factor(self):
        rng = np.random.default_rng(11)
        hard, none = [], []
        for _ in range(1000):
            cells = _simulate(rng, n_levels=50, per_level=8, alpha_sd=1.0, p_treat=0.2)
            try:
                hard.append(absorb_one_way(*cells, pooling="hard").effect)
                none.append(absorb_one_way(*cells, pooling="none").effect)
            except ValueError:
                continue
        sd_hard, sd_none = np.std(hard), np.std(none)
        assert sd_hard <= sd_none * 0.95
