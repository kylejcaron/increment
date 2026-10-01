"""Wald-interval coverage of :func:`sitewide_impact`/:func:`sitewide_impact_ratio`
at cluster grain, via :meth:`SitewideContrast.from_clusters`/
:meth:`SitewideRatioContrast.from_clusters`. ``test_cluster_coverage.py`` covers
``estimate_lift``'s cluster path but never touches sitewide; this is the analogue.

DGP (sum metric): ``y_ij = mu (+ effect if target) + b_j + e_ij``, ``b_j ~
N(0, sigma_b**2)`` per cluster, ``e_ij ~ N(0, sigma_e**2)`` per unit - same
ICC-based DGP as ``test_cluster_coverage.py`` but with a KNOWN nonzero constant
per-unit effect (0.4) instead of a null lift, and Poisson(20)-sized (floored at
2) clusters, so the ratio-of-cluster-sums estimator does genuine work instead
of degenerating to a plain per-unit mean difference.

Because the DGP adds a CONSTANT effect to every target unit, the counterfactual
"what this arm would have summed to with nobody treated" is exact:
``total_y(T) - effect * n_treatment``. So the true ship-to-all target is::

    TRUE_IMPACT = effect * (n_control + n_treatment)

i.e. ``sitewide_impact``'s ``absolute_impact = delta * n_exp`` formula at the
population parameter. The ratio-metric analogue nets out both a numerator and
denominator DGP the same way (see ``_ratio_coverage``'s docstring).

Measured coverage with the ``t`` reference: near-nominal (~95%) at K=360
clusters, comfortably above a "does not collapse" floor (~93-95% sum,
~91-93% ratio) at K=12 clusters - equal-K, matched-per-arm-variance makes
the pairwise-dof interval close to the exact equal-n Student-t case, so it
runs a bit higher than a pre-implementation design estimate would suggest.
Swapping in a Normal reference at K=12 measurably under-covers relative to
``t_{dof}`` (~2-3pp gap), confirming the t reference is required at low
cluster counts. Bounds asserted below come from these measurements, not
a priori estimates.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.stats import norm as _norm

from increment.estimation.armstats import ArmStats
from increment.estimation.sitewide import (
    SitewideContrast,
    SitewideRatioContrast,
    _combination_var,
    sitewide_impact,
    sitewide_impact_ratio,
)

# Reused from test_sitewide.py: builds an ArmStats from exact per-cluster
# mean/var/cov moments (the x-family wire contract for clustered rows).
from tests.estimation.test_sitewide import _cluster_arm, _cluster_ratio_arm
from tests.mc import CoverageSet

# Sum-metric DGP: y_ij = mu (+ effect if target) + b_j + e_ij, ICC = 0.5
# (same as test_cluster_coverage.py). EFFECT is the known per-unit lift checked.
MU = 5.0
SIGMA_B = 0.5
SIGMA_E = 0.5
EFFECT = 0.4
M_MEAN = 20  # mean cluster size; Poisson-distributed, not fixed.


def _sum_arm(
    rng: np.random.Generator, k: int, effect: float, group_id: str
) -> tuple[ArmStats, float, int]:
    """One arm's clustered sum-metric draw: K clusters, Poisson(M_MEAN)
    sizes (floored at 2), ICC-correlated unit outcomes. Returns the
    normalized ArmStats plus the REALIZED total outcome and unit count
    (both DATA the coverage check needs, not moments)."""
    sizes = np.maximum(rng.poisson(lam=M_MEAN, size=k), 2)
    b = rng.normal(0.0, SIGMA_B, size=k)
    n_total = int(sizes.sum())
    e = rng.normal(0.0, SIGMA_E, size=n_total)
    y = MU + effect + np.repeat(b, sizes) + e
    cluster_start = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    g = np.add.reduceat(y, cluster_start)  # per-cluster outcome totals
    arm = _cluster_arm(
        n=k,
        g_mean=float(g.mean()),
        g_var=float(g.var(ddof=1)),
        m_mean=float(sizes.mean()),
        m_var=float(sizes.var(ddof=1)),
        cov_gm=float(np.cov(g, sizes, ddof=1)[0, 1]),
        group_id=group_id,
    )
    return arm, float(g.sum()), n_total


def _sum_coverage(reps: int, k: int, seed: int) -> dict[str, float]:
    """Wald-interval coverage of the TRUE ship-to-all sum-metric impact
    (module docstring) over *reps* draws of K clusters/arm, both with the
    contrast's own ``t_{dof}`` reference and with a Normal reference
    substituted in - same simulated intervals, only the critical value
    swapped, so the two coverage numbers are a real paired comparison."""
    rng = np.random.default_rng(seed)
    covset = CoverageSet()
    z = float(_norm.ppf(0.975))
    with warnings.catch_warnings():
        # Below the 40-cluster warn floor by design; expected noise, not signal.
        warnings.filterwarnings(
            "ignore", message=r".*total clusters.*is below", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=r"sitewide under a declared cluster:",
            category=UserWarning,
        )
        for _ in range(reps):
            control, total_c, n_c = _sum_arm(rng, k, 0.0, "C")
            target, total_t, n_t = _sum_arm(rng, k, effect=EFFECT, group_id="T")
            contrast = SitewideContrast.from_clusters(control, target, cluster="store")
            site_total_volume = total_c + total_t
            true_impact = EFFECT * (n_c + n_t)

            result = sitewide_impact(contrast, site_total_volume=site_total_volume)
            se = result.absolute_impact_se
            lb_norm, ub_norm = result.absolute_impact - z * se, result.absolute_impact + z * se
            covset.record(
                t=result.absolute_impact_lb <= true_impact <= result.absolute_impact_ub,
                norm=lb_norm <= true_impact <= ub_norm,
            )
    t_rate, norm_rate = covset.rates("t", "norm")
    return {"coverage_t": t_rate, "coverage_norm": norm_rate}


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_sum_impact_point_estimate_converges_to_true_impact_at_large_k():
    """Large-K sanity check that TRUE_IMPACT (module docstring) is the
    right target, not just a formula that's internally consistent with
    itself: at K=3000 clusters/arm the function's OWN point estimate,
    built from the estimated ``delta`` and the realized ``V``, should
    land within a couple of its own standard errors of TRUE_IMPACT if
    TRUE_IMPACT really is what the DGP converges to."""
    rng = np.random.default_rng(999)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*total clusters.*is below", category=RuntimeWarning
        )
        control, total_c, n_c = _sum_arm(rng, 3000, 0.0, "C")
        target, total_t, n_t = _sum_arm(rng, 3000, EFFECT, "T")
        contrast = SitewideContrast.from_clusters(control, target, cluster="store")

    site_total_volume = total_c + total_t
    true_impact = EFFECT * (n_c + n_t)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"sitewide under a declared cluster:",
            category=UserWarning,
        )
        result = sitewide_impact(contrast, site_total_volume=site_total_volume)

    # Measured |point_est - true_impact| ~ 0.36 SE; 3 SE gives margin
    # against Monte-Carlo noise while still catching a wrong formula.
    assert abs(result.absolute_impact - true_impact) < 3.0 * result.absolute_impact_se


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_sum_coverage_near_nominal_at_large_k():
    """K=360 total clusters (180/arm), 2500 reps: coverage should be
    tight around nominal since t_{358} is essentially z. Measured 95.1%;
    bounds set from that measurement with room for Monte-Carlo noise
    (binomial SE at 2500 reps is ~0.4pp)."""
    stats = _sum_coverage(reps=2500, k=180, seed=20260812)
    assert 0.93 <= stats["coverage_t"] <= 0.97, stats


def test_sum_coverage_near_nominal_at_large_k_smoke():
    """Small-N smoke twin: same K, far fewer reps, just checks the
    machinery runs and isn't grossly wrong - not a nominal-coverage
    check (reps=40 binomial noise is ~+/-8pp)."""
    stats = _sum_coverage(reps=40, k=180, seed=1)
    assert stats["coverage_t"] > 0.7, stats


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_sum_coverage_and_t_reference_hold_at_small_k():
    """K=12 total clusters (6/arm), near the check_total_clusters refusal
    floor of 10: the t-reference interval should NOT collapse (measured
    94.6%, well above a "structurally broken" bar), and swapping in a
    Normal reference should measurably under-cover relative to it
    (measured 91.8% vs 94.6%, a ~2.7pp gap) - both bounds set from these
    measurements, not the pre-implementation design estimate (see module
    docstring for why this module's number runs a bit higher)."""
    stats = _sum_coverage(reps=2500, k=6, seed=20260813)
    assert stats["coverage_t"] > 0.85, stats
    assert stats["coverage_norm"] < stats["coverage_t"], stats
    assert stats["coverage_t"] - stats["coverage_norm"] > 0.01, stats


def test_sum_coverage_and_t_reference_hold_at_small_k_smoke():
    """Small-N smoke twin of the above - same K, far fewer reps."""
    stats = _sum_coverage(reps=40, k=6, seed=2)
    assert stats["coverage_t"] > 0.7, stats
    # t vs Normal ordering is a per-rep nesting fact (t_dof's critical
    # value is never smaller than z's), not binomial noise, so holds at reps=40.
    assert stats["coverage_norm"] <= stats["coverage_t"], stats


# Ratio-metric DGP: independent numerator/denominator ICC processes with
# their own constant lifts, so the delta-method covariance term does real work.
MU_NUM, SIGMA_B_NUM, SIGMA_E_NUM = 10.0, 0.6, 0.6
MU_DEN, SIGMA_B_DEN, SIGMA_E_DEN = 5.0, 0.3, 0.3
EFFECT_NUM, EFFECT_DEN = 0.5, 0.1


def _ratio_arm(
    rng: np.random.Generator, k: int, effect_num: float, effect_den: float, group_id: str
) -> tuple[ArmStats, float, float, int]:
    """One arm's clustered ratio-metric draw: independent numerator/
    denominator ICC processes over the SAME Poisson(M_MEAN) cluster
    sizes. Returns the normalized ArmStats plus the realized numerator
    total, denominator total, and unit count."""
    sizes = np.maximum(rng.poisson(lam=M_MEAN, size=k), 2)
    n_total = int(sizes.sum())
    cluster_start = np.concatenate(([0], np.cumsum(sizes)[:-1]))

    b_num = rng.normal(0.0, SIGMA_B_NUM, size=k)
    e_num = rng.normal(0.0, SIGMA_E_NUM, size=n_total)
    num = MU_NUM + effect_num + np.repeat(b_num, sizes) + e_num
    num_g = np.add.reduceat(num, cluster_start)

    b_den = rng.normal(0.0, SIGMA_B_DEN, size=k)
    e_den = rng.normal(0.0, SIGMA_E_DEN, size=n_total)
    den = MU_DEN + effect_den + np.repeat(b_den, sizes) + e_den
    den_g = np.add.reduceat(den, cluster_start)

    arm = _cluster_ratio_arm(
        n=k,
        num_mean=float(num_g.mean()),
        num_var=float(num_g.var(ddof=1)),
        den_mean=float(den_g.mean()),
        den_var=float(den_g.var(ddof=1)),
        m_mean=float(sizes.mean()),
        m_var=float(sizes.var(ddof=1)),
        cov_num_den=float(np.cov(num_g, den_g, ddof=1)[0, 1]),
        cov_num_m=float(np.cov(num_g, sizes, ddof=1)[0, 1]),
        cov_den_m=float(np.cov(den_g, sizes, ddof=1)[0, 1]),
        group_id=group_id,
    )
    return arm, float(num_g.sum()), float(den_g.sum()), n_total


def _ratio_coverage(reps: int, k: int, seed: int) -> dict[str, float]:
    """Wald-interval coverage of the TRUE ship-to-all ratio-metric impact
    over *reps* draws. Both site totals are the REALIZED sums (``V_num``/
    ``V_den``), and - exactly the sum-metric argument in ``_sum_coverage``
    generalized to two lifts - the TRUE counterfactual baseline is
    recoverable exactly from realized data because both DGP effects are
    constants, not random draws::

        N0 = V_num - EFFECT_NUM * n_treatment
        D0 = V_den - EFFECT_DEN * n_treatment
        N1 = N0 + n_exp * EFFECT_NUM
        D1 = D0 + n_exp * EFFECT_DEN
        TRUE_IMPACT = N1/D1 - N0/D0

    the exact ``sitewide_impact_ratio`` formula with the TRUE effects
    substituted for the estimated ``delta_num``/``delta_den``.
    """
    rng = np.random.default_rng(seed)
    covset = CoverageSet()
    z = float(_norm.ppf(0.975))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*total clusters.*is below", category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore",
            message=r"sitewide under a declared cluster:",
            category=UserWarning,
        )
        for _ in range(reps):
            control, vnum_c, vden_c, n_c = _ratio_arm(rng, k, 0.0, 0.0, "C")
            target, vnum_t, vden_t, n_t = _ratio_arm(rng, k, EFFECT_NUM, EFFECT_DEN, "T")
            contrast = SitewideRatioContrast.from_clusters(control, target, cluster="store")
            site_total_numerator = vnum_c + vnum_t
            site_total_denominator = vden_c + vden_t

            n0 = site_total_numerator - EFFECT_NUM * n_t
            d0 = site_total_denominator - EFFECT_DEN * n_t
            n_exp = n_c + n_t
            n1 = n0 + n_exp * EFFECT_NUM
            d1 = d0 + n_exp * EFFECT_DEN
            true_impact = n1 / d1 - n0 / d0

            result = sitewide_impact_ratio(
                contrast,
                site_total_numerator=site_total_numerator,
                site_total_denominator=site_total_denominator,
            )
            se = result.absolute_impact_se
            lb_norm, ub_norm = result.absolute_impact - z * se, result.absolute_impact + z * se
            covset.record(
                t=result.absolute_impact_lb <= true_impact <= result.absolute_impact_ub,
                norm=lb_norm <= true_impact <= ub_norm,
            )
    t_rate, norm_rate = covset.rates("t", "norm")
    return {"coverage_t": t_rate, "coverage_norm": norm_rate}


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_ratio_impact_point_estimate_converges_to_true_impact_at_large_k():
    """Ratio-metric analogue of the sum-metric convergence check above:
    at K=3000 clusters/arm the point estimate landed within 0.1 SE of
    TRUE_IMPACT (module docstring)."""
    rng = np.random.default_rng(998)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*total clusters.*is below", category=RuntimeWarning
        )
        control, vnum_c, vden_c, n_c = _ratio_arm(rng, 3000, 0.0, 0.0, "C")
        target, vnum_t, vden_t, n_t = _ratio_arm(rng, 3000, EFFECT_NUM, EFFECT_DEN, "T")
        contrast = SitewideRatioContrast.from_clusters(control, target, cluster="store")

    site_total_numerator = vnum_c + vnum_t
    site_total_denominator = vden_c + vden_t
    n0 = site_total_numerator - EFFECT_NUM * n_t
    d0 = site_total_denominator - EFFECT_DEN * n_t
    n_exp = n_c + n_t
    n1 = n0 + n_exp * EFFECT_NUM
    d1 = d0 + n_exp * EFFECT_DEN
    true_impact = n1 / d1 - n0 / d0

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"sitewide under a declared cluster:",
            category=UserWarning,
        )
        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=site_total_numerator,
            site_total_denominator=site_total_denominator,
        )

    assert abs(result.absolute_impact - true_impact) < 3.0 * result.absolute_impact_se


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_ratio_coverage_near_nominal_at_large_k():
    """K=360 total clusters (180/arm), 2000 reps. Measured 95.1%."""
    stats = _ratio_coverage(reps=2000, k=180, seed=20260814)
    assert 0.93 <= stats["coverage_t"] <= 0.97, stats


def test_ratio_coverage_near_nominal_at_large_k_smoke():
    stats = _ratio_coverage(reps=40, k=180, seed=3)
    assert stats["coverage_t"] > 0.7, stats


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_ratio_coverage_and_t_reference_hold_at_small_k():
    """K=12 total clusters (6/arm), 2500 reps: measured 92.9% with the
    t reference (above the "does not collapse" floor) vs 90.8% with a
    Normal reference (a ~2.2pp real gap) - the ratio metric's absolute
    impact runs its own Satterthwaite dof (module docstring, "Degrees of
    freedom"), not the plain pairwise one, so it sits closer to the
    pre-implementation design estimate than the sum metric's does."""
    stats = _ratio_coverage(reps=2500, k=6, seed=20260815)
    assert stats["coverage_t"] > 0.85, stats
    assert stats["coverage_norm"] < stats["coverage_t"], stats
    assert stats["coverage_t"] - stats["coverage_norm"] > 0.01, stats


def test_ratio_coverage_and_t_reference_hold_at_small_k_smoke():
    stats = _ratio_coverage(reps=40, k=6, seed=4)
    assert stats["coverage_t"] > 0.7, stats
    assert stats["coverage_norm"] <= stats["coverage_t"], stats


# g' Sigma g cross-check, clustered multi-arm case (analogue of
# test_sitewide.py's TestSitewideImpactMultiArm variance test).


def _den_of(arm: Any) -> float:
    """Narrow a ratio-family arm's Optional mean_den to float for use
    inside a generator over a heterogeneous tuple of arms - mirrors
    sitewide_impact_ratio's own inline ``_den`` helper, since a bare
    ``arm.mean_den is not None`` assert on the loop variable doesn't
    narrow across iterations the way it does on a fixed name."""
    assert arm.mean_den is not None
    return arm.mean_den


def _quadratic_form(grad: list[float], sigma: list[list[float]]) -> float:
    """``grad' sigma grad``, spelled out independently of _combination_var
    - mirrors test_sitewide.py's own ``_quadratic_form`` helper."""
    return sum(grad[i] * sigma[i][j] * grad[j] for i in range(len(grad)) for j in range(len(grad)))


@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
def test_relative_variance_matches_g_sigma_g_clustered_multiarm_sum():
    """Three cluster arms (control/target/other, the same fixture
    TestDegreesOfFreedom::test_sum_absolute_impact_uses_two_arm_welch_not_pooled_dof
    uses), sum-metric relative impact: independently assemble Sigma from
    each arm's cluster-grain ``var_mean`` - diag(var_mean_i) +
    control.var_mean * 11' - and check the public relative impact SE
    against it, catching a bug shared between the derivation comment and
    the implementation that a same-code-path check never would."""
    control = _cluster_arm(
        n=25, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
    )
    target = _cluster_arm(
        n=25, g_mean=110.0, g_var=81.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
    )
    other = _cluster_arm(
        n=15, g_mean=105.0, g_var=49.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="O"
    )
    contrast = SitewideContrast.from_clusters(control, target, other_arms=[other], cluster="s")
    site_total_volume = 50_000.0
    result = sitewide_impact(contrast, site_total_volume=site_total_volume)

    ca, ta, oa = contrast.control, contrast.target, contrast.others[0]
    n_exp = ca.n_units + ta.n_units + oa.n_units
    other_contribution = oa.n_units * (oa.mean - ca.mean)
    baseline = site_total_volume - (ta.mean - ca.mean) * ta.n_units - other_contribution
    volume_net_of_others = site_total_volume - other_contribution
    absolute_impact = (ta.mean - ca.mean) * n_exp
    grad = [
        n_exp * volume_net_of_others / baseline**2,
        absolute_impact * oa.n_units / baseline**2,
    ]

    sigma = [
        [ta.var_mean + ca.var_mean, ca.var_mean],
        [ca.var_mean, oa.var_mean + ca.var_mean],
    ]
    quad = _quadratic_form(grad, sigma)

    assert result.relative_impact_se**2 == pytest.approx(quad, rel=1e-9)


@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
def test_absolute_variance_matches_g_sigma_g_clustered_multiarm_ratio():
    """Ratio-family analogue: three cluster arms, ratio-metric ABSOLUTE
    impact (which mixes every arm via the coef_den branch of
    _combination_var, unlike the sum metric's absolute impact). Sigma is
    now a per-arm 2x2 block (numerator, denominator) with the shared
    control structure on both the numerator AND denominator off-diagonal
    - assembled here as a flat 4x4 over (num_T, den_T, num_O, den_O)."""
    control = _cluster_ratio_arm(
        n=25,
        num_mean=100.0,
        num_var=64.0,
        den_mean=50.0,
        den_var=16.0,
        m_mean=20.0,
        m_var=4.0,
        cov_num_den=20.0,
        cov_num_m=8.0,
        cov_den_m=6.0,
        group_id="C",
    )
    target = _cluster_ratio_arm(
        n=25,
        num_mean=110.0,
        num_var=81.0,
        den_mean=52.0,
        den_var=18.0,
        m_mean=20.0,
        m_var=4.0,
        cov_num_den=21.0,
        cov_num_m=8.5,
        cov_den_m=6.5,
        group_id="T",
    )
    other = _cluster_ratio_arm(
        n=15,
        num_mean=104.0,
        num_var=49.0,
        den_mean=48.0,
        den_var=12.0,
        m_mean=20.0,
        m_var=4.0,
        cov_num_den=15.0,
        cov_num_m=7.0,
        cov_den_m=5.0,
        group_id="O",
    )
    contrast = SitewideRatioContrast.from_clusters(control, target, other_arms=[other], cluster="s")
    site_total_numerator, site_total_denominator = 5_000.0, 2_500.0
    result = sitewide_impact_ratio(
        contrast,
        site_total_numerator=site_total_numerator,
        site_total_denominator=site_total_denominator,
    )

    ca, ta, oa = contrast.control, contrast.target, contrast.others[0]
    assert ca.mean_den is not None and ta.mean_den is not None and oa.mean_den is not None
    assert (
        ca.var_mean_den is not None and ta.var_mean_den is not None and oa.var_mean_den is not None
    )
    assert (
        ca.cov_mean_den is not None and ta.cov_mean_den is not None and oa.cov_mean_den is not None
    )
    arms = (ta, oa)
    n_exp = ca.n_units + ta.n_units + oa.n_units
    num_contribution = math.fsum(a.n_units * (a.mean - ca.mean) for a in arms)
    den_contribution = math.fsum(a.n_units * (_den_of(a) - ca.mean_den) for a in arms)
    n0 = site_total_numerator - num_contribution
    d0 = site_total_denominator - den_contribution
    n_switched = n_exp - ta.n_units
    n1 = n0 + n_exp * (ta.mean - ca.mean)
    d1 = d0 + n_exp * (ta.mean_den - ca.mean_den)

    g_num = [n_switched / d1 + ta.n_units / d0, oa.n_units * (1.0 / d0 - 1.0 / d1)]
    g_den = [
        -n1 * n_switched / d1**2 - n0 * ta.n_units / d0**2,
        oa.n_units * (n1 / d1**2 - n0 / d0**2),
    ]
    components = _combination_var(ca, arms, g_num, g_den)
    components_sum = math.fsum(components)

    grad4 = [g_num[0], g_den[0], g_num[1], g_den[1]]
    sigma4 = [[0.0] * 4 for _ in range(4)]
    sigma4[0][0] = ta.var_mean + ca.var_mean
    sigma4[1][1] = ta.var_mean_den + ca.var_mean_den
    sigma4[0][1] = sigma4[1][0] = ta.cov_mean_den + ca.cov_mean_den
    sigma4[2][2] = oa.var_mean + ca.var_mean
    sigma4[3][3] = oa.var_mean_den + ca.var_mean_den
    sigma4[2][3] = sigma4[3][2] = oa.cov_mean_den + ca.cov_mean_den
    # Cross-arm blocks: only shared-control terms survive - target and
    # other are disjoint samples, so only control-anchored terms remain.
    sigma4[0][2] = sigma4[2][0] = ca.var_mean
    sigma4[0][3] = sigma4[3][0] = ca.cov_mean_den
    sigma4[1][2] = sigma4[2][1] = ca.cov_mean_den
    sigma4[1][3] = sigma4[3][1] = ca.var_mean_den

    quad = _quadratic_form(grad4, sigma4)

    assert quad == pytest.approx(components_sum, rel=1e-12)
    assert result.absolute_impact_se**2 == pytest.approx(quad, rel=1e-9)


# Dof-path divergence at unbalanced K: pairwise dof (sum-metric absolute
# impact) vs Satterthwaite dof (relative/ratio-absolute) must genuinely differ.


@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
def test_pairwise_and_satterthwaite_dof_diverge_at_unbalanced_k_sum():
    """K = 10/14/30 (control/target/other) with asymmetric cluster-size
    profiles (m_mean 15/25/40) and asymmetric per-cluster outcomes, so the
    relative-impact gradient weights are NOT hand-picked to force a
    particular dof - they fall out of the fixture. Measured: pairwise
    dof 22.0 vs Satterthwaite dof ~14.6, unambiguously different (and
    inside the reduction's own bound sum_i(K_i - 1) = 51)."""
    control = _cluster_arm(
        n=10, g_mean=100.0, g_var=64.0, m_mean=15.0, m_var=3.0, cov_gm=5.0, group_id="C"
    )
    target = _cluster_arm(
        n=14, g_mean=130.0, g_var=90.0, m_mean=25.0, m_var=6.0, cov_gm=9.0, group_id="T"
    )
    other = _cluster_arm(
        n=30, g_mean=80.0, g_var=40.0, m_mean=40.0, m_var=10.0, cov_gm=12.0, group_id="O"
    )
    contrast = SitewideContrast.from_clusters(control, target, other_arms=[other], cluster="s")
    assert contrast.n_clusters == 54
    site_total_volume = 200_000.0

    # Naive pairwise dof K_T-1 + K_C-1. Absolute impact uses a two-arm
    # Welch-Satterthwaite reduction instead; this is only the reference the
    # relative-impact dof below must diverge from.
    assert contrast.target.own_dof is not None and contrast.control.own_dof is not None
    pairwise_dof = contrast.target.own_dof + contrast.control.own_dof
    assert pairwise_dof == pytest.approx(22.0)

    # Sum-metric relative impact: Satterthwaite reduction over the same
    # contrast's per-term variance components, re-derived by hand.
    ca, ta, oa = contrast.control, contrast.target, contrast.others[0]
    arms = (ta, oa)
    n_exp = ca.n_units + ta.n_units + oa.n_units
    other_contribution = math.fsum(a.n_units * (a.mean - ca.mean) for a in (oa,))
    baseline = site_total_volume - (ta.mean - ca.mean) * ta.n_units - other_contribution
    volume_net_of_others = site_total_volume - other_contribution
    absolute_impact = (ta.mean - ca.mean) * n_exp
    grad = [
        n_exp * volume_net_of_others / baseline**2,
        absolute_impact * oa.n_units / baseline**2,
    ]
    # Delta-method terms of the gradient contracted against the arm deltas,
    # re-derived from the contrast's own arm moments.
    sum_grad = math.fsum(grad)
    components = (
        sum_grad * sum_grad * ca.var_mean,
        *(g * g * arm.var_mean for g, arm in zip(grad, arms, strict=True)),
    )
    component_dofs = (9.0, 13.0, 29.0)
    component_sum = math.fsum(components)
    expected_dof = component_sum**2 / math.fsum(
        component**2 / dof for component, dof in zip(components, component_dofs, strict=True)
    )
    result = sitewide_impact(contrast, site_total_volume=site_total_volume)

    assert result.relative_dof == pytest.approx(expected_dof)
    assert expected_dof != pytest.approx(pairwise_dof)
    # Not a coincidence of this fixture landing near the pooled dof either.
    assert expected_dof != pytest.approx(contrast.dof)


@pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
def test_pairwise_and_satterthwaite_dof_diverge_at_unbalanced_k_ratio():
    """Ratio-metric analogue: the ratio-metric ABSOLUTE impact path also
    uses the Satterthwaite reduction (module docstring, "Degrees of
    freedom" - unlike the sum metric, ratio-metric absolute impact mixes
    every arm via coef_den). Same K=10/14/30 asymmetry, translated to the
    numerator/denominator/size fixture. Measured: pairwise dof 22.0 vs
    Satterthwaite dof ~18.4 (inside the bound sum_i(K_i - 1) = 51)."""
    control = _cluster_ratio_arm(
        n=10,
        num_mean=100.0,
        num_var=64.0,
        den_mean=50.0,
        den_var=16.0,
        m_mean=15.0,
        m_var=3.0,
        cov_num_den=20.0,
        cov_num_m=5.0,
        cov_den_m=4.0,
        group_id="C",
    )
    target = _cluster_ratio_arm(
        n=14,
        num_mean=140.0,
        num_var=100.0,
        den_mean=60.0,
        den_var=25.0,
        m_mean=25.0,
        m_var=6.0,
        cov_num_den=30.0,
        cov_num_m=9.0,
        cov_den_m=7.0,
        group_id="T",
    )
    other = _cluster_ratio_arm(
        n=30,
        num_mean=90.0,
        num_var=50.0,
        den_mean=45.0,
        den_var=14.0,
        m_mean=40.0,
        m_var=10.0,
        cov_num_den=18.0,
        cov_num_m=10.0,
        cov_den_m=8.0,
        group_id="O",
    )
    contrast = SitewideRatioContrast.from_clusters(control, target, other_arms=[other], cluster="s")
    assert contrast.n_clusters == 54
    site_total_numerator, site_total_denominator = 20_000.0, 10_000.0

    assert contrast.target.own_dof is not None and contrast.control.own_dof is not None
    pairwise_dof = contrast.target.own_dof + contrast.control.own_dof
    assert pairwise_dof == pytest.approx(22.0)

    ca, ta, oa = contrast.control, contrast.target, contrast.others[0]
    assert ca.mean_den is not None and ta.mean_den is not None
    arms = (ta, oa)
    n_exp = ca.n_units + ta.n_units + oa.n_units
    num_contribution = math.fsum(a.n_units * (a.mean - ca.mean) for a in arms)
    den_contribution = math.fsum(a.n_units * (_den_of(a) - ca.mean_den) for a in arms)
    n0 = site_total_numerator - num_contribution
    d0 = site_total_denominator - den_contribution
    n_switched = n_exp - ta.n_units
    n1 = n0 + n_exp * (ta.mean - ca.mean)
    d1 = d0 + n_exp * (ta.mean_den - ca.mean_den)
    g_num = [n_switched / d1 + ta.n_units / d0, oa.n_units * (1.0 / d0 - 1.0 / d1)]
    g_den = [
        -n1 * n_switched / d1**2 - n0 * ta.n_units / d0**2,
        oa.n_units * (n1 / d1**2 - n0 / d0**2),
    ]
    # Delta-method terms of the numerator/denominator gradients, re-derived
    # from the contrast's own arm moments.
    assert ca.var_mean_den is not None and ca.cov_mean_den is not None
    sum_num = math.fsum(g_num)
    sum_den = math.fsum(g_den)
    components = [
        sum_num * sum_num * ca.var_mean
        + sum_den * sum_den * ca.var_mean_den
        + 2.0 * sum_num * sum_den * ca.cov_mean_den
    ]
    for arm, a, b in zip(arms, g_num, g_den, strict=True):
        assert arm.var_mean_den is not None and arm.cov_mean_den is not None
        components.append(
            a * a * arm.var_mean + b * b * arm.var_mean_den + 2.0 * a * b * arm.cov_mean_den
        )
    component_dofs = (9.0, 13.0, 29.0)
    component_sum = math.fsum(components)
    expected_dof = component_sum**2 / math.fsum(
        component**2 / dof for component, dof in zip(components, component_dofs, strict=True)
    )
    result = sitewide_impact_ratio(
        contrast,
        site_total_numerator=site_total_numerator,
        site_total_denominator=site_total_denominator,
    )

    assert result.absolute_dof == pytest.approx(expected_dof)
    assert expected_dof != pytest.approx(pairwise_dof)
    assert expected_dof != pytest.approx(contrast.dof)
