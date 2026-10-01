"""Tests for heterogeneity (cochran_q: Q, DerSimonian-Laird tau^2, Higgins-Thompson I^2)."""

import math

import numpy as np
import pytest
from scipy.stats import chi2
from scipy.stats import norm as norm_dist

from increment.errors import InvalidRequestError
from increment.estimation.meta import (
    HeterogeneityResult,
    MarginalizedSegmentIntervals,
    cochran_q,
    hksj_pooled_mean,
    marginalized_segment_intervals,
    pairwise_contrast_arrays,
)
from tests.oracles.test_meta_oracle import assert_segment_intervals_match, segment_posterior

# 3-segment fixture with UNEQUAL variances: an equal-variance fixture pins
# nothing about which moment estimator (DL/Paule-Mandel/REML) shipped.
FIXTURE_EST = [0.10, 0.30, -0.05]
FIXTURE_VAR = [0.02, 0.05, 0.01]


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.meta.need_least_estimable",
            lambda: cochran_q([0.1], [0.02]),
        ),
        (
            "estimation.meta.tau2_finite",
            lambda: hksj_pooled_mean([0.1, 0.2], [0.02, 0.05], tau2=-0.1),
        ),
    ],
)
def test_meta_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


def test_alpha_refusal_code_is_shared_across_meta_entry_points() -> None:
    """Cochran and marginalized interval validation share one alpha refusal."""
    with pytest.raises(InvalidRequestError) as via_cochran:
        cochran_q(FIXTURE_EST, FIXTURE_VAR, alpha=0.0)
    with pytest.raises(InvalidRequestError) as via_marginalized:
        marginalized_segment_intervals([0.1, 0.2], [0.02, 0.05], alpha=0.0)
    assert via_cochran.value.code == via_marginalized.value.code
    assert via_cochran.value.code == "estimation.meta.alpha_strictly_between"


class TestCochranQ:
    def test_hand_computed_q_df_p(self):
        """Q, df, p_value match a hand-computed inverse-variance-weighted fixture."""
        result = cochran_q(FIXTURE_EST, FIXTURE_VAR)
        assert result.k == 3
        assert result.df == 2
        assert result.q == pytest.approx(2.338235294117647)
        assert result.p_value == pytest.approx(0.3106409153020779)

    def test_hand_computed_dl_tau2(self):
        """DerSimonian-Laird tau^2 = max(0, (Q - df) / (sum(w) - sum(w^2)/sum(w)))."""
        result = cochran_q(FIXTURE_EST, FIXTURE_VAR)
        assert result.tau2 == pytest.approx(0.003593750000000001)

    def test_hand_computed_i2_point_and_ci(self):
        """I^2 point and Higgins-Thompson CI match the closed-form fixture values."""
        result = cochran_q(FIXTURE_EST, FIXTURE_VAR)
        # k=3 < 5 -> point estimate suppressed, but the CI is still reported.
        assert result.i2 is None
        assert result.i2_lb == pytest.approx(0.0)

        # Derive i2_ub from result.q/df/k with Higgins & Thompson (2002)'s normal
        # CI on ln(H), H = sqrt(Q/df), so an SE(ln H) regression cannot hide
        # behind a snapshot of result.i2_ub.
        h = max(1.0, math.sqrt(result.q / result.df))
        # Q(2.338) <= k(3): the "Q <= k" branch of the piecewise SE.
        se_ln_h = math.sqrt(1.0 / (2 * (result.k - 2)) * (1.0 - 1.0 / (3 * (result.k - 2) ** 2)))
        z = float(norm_dist.ppf(0.975))
        h_hi = math.exp(math.log(h) + z * se_ln_h)
        i2_ub_expected = max(0.0, (h_hi**2 - 1) / h_hi**2)

        assert i2_ub_expected == pytest.approx(0.9110268628545796)
        assert result.i2_ub == pytest.approx(i2_ub_expected)

    def test_i2_point_reported_at_k_5(self):
        """At k=5 the I^2 point estimate is no longer suppressed.

        More heterogeneous than FIXTURE_EST so Q > df=4 and the shrinkage
        arithmetic is genuinely exercised, not clamped to 0. Q is
        hand-computed here (inverse-variance-weighted sum of squared
        deviations from the pooled mean) from est/var directly, never from
        result.q/result.df - asserting i2 against a formula built out of
        the result's own fields would be circular and pass even if
        cochran_q's Q itself were wrong.
        """
        est = [0.10, 0.55, -0.30, 0.20, 0.45]
        var = [0.02, 0.05, 0.01, 0.03, 0.015]
        result = cochran_q(est, var)
        assert result.i2 is not None

        w = [1.0 / v for v in var]
        mu_hat = sum(wi * ei for wi, ei in zip(w, est, strict=True)) / sum(w)
        q_hand = sum(wi * (ei - mu_hat) ** 2 for wi, ei in zip(w, est, strict=True))
        df_hand = len(est) - 1
        i2_hand = max(0.0, (q_hand - df_hand) / q_hand)

        assert q_hand == pytest.approx(28.480452674897123)
        assert result.i2 == pytest.approx(i2_hand)

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_p_tau2_zero_matches_exact_chi_square_probability_k5(self):
        """P(tau^2 = 0) at tau = 0 equals the exact P(chi2_{k-1} < k-1); 59.40% at k=5."""
        rng = np.random.default_rng(0)
        k, n_per_segment, reps = 5, 400, 4000
        hits = 0
        for _ in range(reps):
            y = rng.standard_normal((k, n_per_segment))
            est = y.mean(axis=1)
            var = y.var(axis=1, ddof=1) / n_per_segment
            hits += cochran_q(est, var).tau2 == 0.0
        empirical = hits / reps
        exact = chi2.cdf(k - 1, k - 1)
        assert exact == pytest.approx(0.5939941502901616, abs=1e-4)
        assert empirical == pytest.approx(exact, abs=0.03)

    def test_p_tau2_zero_matches_exact_chi_square_probability_k5_smoke(self):
        """Fast-suite variant of the above: fewer reps, looser tolerance -
        still exercises cochran_q's tau^2-boundary-collapse behavior on
        every fast-suite run instead of only under -m parameter_recovery."""
        rng = np.random.default_rng(2)
        k, n_per_segment, reps = 5, 400, 200
        hits = 0
        for _ in range(reps):
            y = rng.standard_normal((k, n_per_segment))
            est = y.mean(axis=1)
            var = y.var(axis=1, ddof=1) / n_per_segment
            hits += cochran_q(est, var).tau2 == 0.0
        empirical = hits / reps
        exact = chi2.cdf(k - 1, k - 1)
        assert empirical == pytest.approx(exact, abs=0.10)

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_p_tau2_zero_matches_exact_chi_square_probability_k3(self):
        """P(tau^2 = 0) at tau = 0 equals the exact P(chi2_{k-1} < k-1); 63.21% at k=3.

        Actually simulates cochran_q (a prior version of this test only
        asserted the scipy chi2 identity itself, never calling cochran_q)."""
        rng = np.random.default_rng(1)
        k, n_per_segment, reps = 3, 400, 4000
        hits = 0
        for _ in range(reps):
            y = rng.standard_normal((k, n_per_segment))
            est = y.mean(axis=1)
            var = y.var(axis=1, ddof=1) / n_per_segment
            hits += cochran_q(est, var).tau2 == 0.0
        empirical = hits / reps
        exact = chi2.cdf(k - 1, k - 1)
        assert exact == pytest.approx(0.6321205588285577)
        assert empirical == pytest.approx(exact, abs=0.03)

    def test_p_tau2_zero_matches_exact_chi_square_probability_k3_smoke(self):
        """Fast-suite variant of the k=3 boundary test above."""
        rng = np.random.default_rng(3)
        k, n_per_segment, reps = 3, 400, 200
        hits = 0
        for _ in range(reps):
            y = rng.standard_normal((k, n_per_segment))
            est = y.mean(axis=1)
            var = y.var(axis=1, ddof=1) / n_per_segment
            hits += cochran_q(est, var).tau2 == 0.0
        empirical = hits / reps
        exact = chi2.cdf(k - 1, k - 1)
        assert empirical == pytest.approx(exact, abs=0.10)

    def test_k2_ci_is_none(self):
        """Higgins-Thompson I^2 CI is undefined at k=2 (k-2 denominator)."""
        result = cochran_q([0.10, 0.30], [0.02, 0.05])
        assert result.k == 2
        assert result.i2_lb is None
        assert result.i2_ub is None
        # Q/df/tau2 are unaffected by the CI being undefined.
        assert result.q >= 0.0
        assert result.tau2 >= 0.0

    @pytest.mark.parametrize(
        ("q", "k", "expected_i2", "expected_lb", "expected_ub"),
        [
            # Higgins & Thompson (2002) Table II, and heterometa's dat.higgins02
            # regression fixture (I2=29% at Q=14.1,k=11; the paper's I2=20% is a typo).
            (14.4, 24, 0.0, 0.0, 0.446),
            (14.1, 11, 0.29078, 0.0, 0.65054),
            (81.5, 19, 0.77914, 0.65979, 0.85662),
            (41.5, 7, 0.85542, 0.72188, 0.92484),
            (130.3, 3, 0.98465, 0.97291, 0.99130),
        ],
    )
    def test_higgins_thompson_worked_examples(self, q, k, expected_i2, expected_lb, expected_ub):
        """Reproduces the four/five worked meta-analyses from Higgins & Thompson (2002)."""
        # Build a trivial fixture whose Cochran Q equals the target: one outlier segment,
        # k-1 identical segments, unit variance - solved algebraically for the outlier.
        var = np.ones(k)
        est = np.zeros(k)
        est[0] = np.sqrt(q * k / (k - 1))
        result = cochran_q(est, var)
        assert result.q == pytest.approx(q, rel=1e-6)
        if k >= 5:
            assert result.i2 == pytest.approx(expected_i2, abs=1e-4)
        assert result.i2_lb == pytest.approx(expected_lb, abs=1e-3)
        assert result.i2_ub == pytest.approx(expected_ub, abs=1e-3)

    def test_guards_fewer_than_two_segments(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q([0.1], [0.02])
        assert exc_info.value.code == "estimation.meta.need_least_estimable"

    def test_guards_non_finite_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q([0.1, 0.2, 0.3], [0.02, np.nan, 0.01])
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_guards_zero_variance(self):
        """A zero-event arm gives v_k=0 -> w_k=inf; caller must exclude it first."""
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q([0.1, 0.2, 0.3], [0.02, 0.0, 0.01])
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_guards_non_finite_estimate(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q([0.1, np.nan, 0.3], [0.02, 0.05, 0.01])
        assert exc_info.value.code == "estimation.meta.est_contains_non"

    def test_guards_mismatched_shapes(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q([0.1, 0.2, 0.3], [0.02, 0.05])
        assert exc_info.value.code == "estimation.meta.est_var_same"

    def test_guards_2d_input(self):
        """A 2-D (est, var) pair must not silently pass validation and
        compute a wrong k / aggregate over all elements."""
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q(np.zeros((3, 2)), np.ones((3, 2)))
        assert exc_info.value.code == "estimation.meta.est_var_shape"

    def test_guards_scalar_input(self):
        """A 0-D/scalar input must raise ValueError, not an undocumented
        IndexError from ``.shape[0]``."""
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q(np.array(0.1), np.array(0.02))
        assert exc_info.value.code == "estimation.meta.est_var_shape"

    def test_guards_invalid_alpha(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q(FIXTURE_EST, FIXTURE_VAR, alpha=0.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"
        with pytest.raises(InvalidRequestError) as exc_info:
            cochran_q(FIXTURE_EST, FIXTURE_VAR, alpha=1.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"

    def test_alpha_010_gives_narrower_i2_ci(self):
        """A looser alpha=0.10 (90% CI) must give a strictly narrower I^2
        interval than the default alpha=0.05 (95% CI)."""
        est = [0.10, 0.30, -0.05, 0.15, 0.02]
        var = [0.02, 0.05, 0.01, 0.03, 0.015]
        default = cochran_q(est, var)
        looser = cochran_q(est, var, alpha=0.10)
        assert default.i2_lb is not None
        assert default.i2_ub is not None
        assert looser.i2_lb is not None
        assert looser.i2_ub is not None
        assert (looser.i2_ub - looser.i2_lb) < (default.i2_ub - default.i2_lb)

    def test_result_is_frozen(self):
        result = cochran_q(FIXTURE_EST, FIXTURE_VAR)
        assert isinstance(result, HeterogeneityResult)
        with pytest.raises((TypeError, ValueError)):
            result.q = 999.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_accepts_numpy_arrays(self):
        """Array-like inputs (list or ndarray) give identical results."""
        from_list = cochran_q(FIXTURE_EST, FIXTURE_VAR)
        from_array = cochran_q(np.array(FIXTURE_EST), np.array(FIXTURE_VAR))
        assert from_list == from_array


class TestPairwiseContrastArrays:
    """Closed-form log-scale contrast: diff = est_a - est_b, se = sqrt(var_a + var_b)."""

    def test_hand_computed_diff_se_interval(self):
        diff, se, lb, ub = pairwise_contrast_arrays(0.20, 0.05**2, 0.10, 0.04**2)
        assert diff == pytest.approx(0.10)
        assert se == pytest.approx(0.06403124237432849)
        assert lb == pytest.approx(-0.025498928939038795)
        assert ub == pytest.approx(0.2254989289390388)

    def test_worked_example_individually_significant_difference_not(self):
        """Two segments each individually excluding 0 can still have a
        difference interval that includes 0 - the comparison this whole
        module exists to replace the naive eyeball check for."""
        # Segment A: log lift 0.20, se 0.05 -> 95% CI excludes 0.
        a_lb = 0.20 - 1.959963984540054 * 0.05
        assert a_lb > 0
        # Segment B: log lift 0.10, se 0.04 -> 95% CI excludes 0.
        b_lb = 0.10 - 1.959963984540054 * 0.04
        assert b_lb > 0
        # Their difference does not.
        _, _, lb, ub = pairwise_contrast_arrays(0.20, 0.05**2, 0.10, 0.04**2)
        assert lb < 0 < ub

    def test_guards_non_positive_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            pairwise_contrast_arrays(0.1, 0.0, 0.2, 0.01)
        assert exc_info.value.code == "estimation.meta.var_a_finite"
        with pytest.raises(InvalidRequestError) as exc_info:
            pairwise_contrast_arrays(0.1, 0.01, 0.2, -1.0)
        assert exc_info.value.code == "estimation.meta.var_b_finite"

    def test_guards_non_finite_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            pairwise_contrast_arrays(0.1, np.nan, 0.2, 0.01)
        assert exc_info.value.code == "estimation.meta.var_a_finite"

    def test_guards_non_finite_estimate(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            pairwise_contrast_arrays(np.nan, 0.01, 0.2, 0.01)
        assert exc_info.value.code == "estimation.meta.est_a_est"

    def test_guards_invalid_alpha(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            pairwise_contrast_arrays(0.1, 0.01, 0.2, 0.01, alpha=1.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"

    def test_symmetric_swap_negates_diff(self):
        """Swapping A and B negates diff and mirrors the interval."""
        diff_ab, se_ab, lb_ab, ub_ab = pairwise_contrast_arrays(0.20, 0.05**2, 0.10, 0.04**2)
        diff_ba, se_ba, lb_ba, ub_ba = pairwise_contrast_arrays(0.10, 0.04**2, 0.20, 0.05**2)
        assert diff_ba == pytest.approx(-diff_ab)
        assert se_ba == pytest.approx(se_ab)
        assert lb_ba == pytest.approx(-ub_ab)
        assert ub_ba == pytest.approx(-lb_ab)


class TestMarginalizedSegmentIntervals:
    """Tau-marginalised random-effects posterior: shrunken estimate,
    E[lambda_k|y] shrinkage, and interval - no plug-in tau^2 anywhere.
    Agreement with continuous quadrature of the same model, across posterior
    regimes, translations and rescalings, lives in
    tests/oracles/test_meta_oracle.py."""

    # 5-segment fixture with WIDELY unequal variances, so shrink_k's
    # monotonicity in v_k is actually discriminating.
    _EST = [0.10, 0.30, -0.05, 0.20, 0.0]
    _VAR = [0.02, 0.5, 0.01, 2.0, 0.005]

    @pytest.mark.parametrize("k", [2, 3, 5, 10])
    def test_shrink_k_strictly_decreasing_in_variance(self, k):
        """E[lambda_k|y] is strictly positive and strictly decreasing in
        v_k, across a range of K with geometrically-spaced UNEQUAL
        variances - a single fixed-K fixture cannot exercise monotonicity
        across K."""
        rng = np.random.default_rng(k)
        var = 0.02 * 2.0 ** np.arange(k)
        est = 0.1 * rng.standard_normal(k)
        r = marginalized_segment_intervals(est, var)
        assert np.all(np.asarray(r.shrink_k) > 0)
        order = np.argsort(var)
        sorted_shrink = np.asarray(r.shrink_k)[order]
        assert np.all(np.diff(sorted_shrink) < 0), (
            f"shrink_k must strictly decrease as v_k increases (k={k})"
        )

    @pytest.mark.parametrize("k", [2, 3, 5, 10, 100])
    def test_positive_width_at_every_k_including_true_null(self, k):
        """Every interval is strictly positive-width, including the TRUE-NULL
        case (every est identical) - exactly where a plug-in tau^2 would
        collapse to zero and degenerate the interval to a point mass."""
        est = np.zeros(k)  # true null: no heterogeneity at all
        var = np.full(k, 0.1**2)
        r = marginalized_segment_intervals(est, var)
        assert np.all(np.asarray(r.ub) > np.asarray(r.lb))

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_coverage_smoke_k3(self):
        """Measured, not trusted from the design record: K=3, true tau=0.05,
        se=0.095 per segment, seed=0, 1500 reps -> measured coverage
        96.9%. Band is that measurement +/- a few points, not the nominal
        95% +/- 5-9pp the wider band used to allow."""
        rng = np.random.default_rng(0)
        k, se, true_tau, reps = 3, 0.095, 0.05, 1500
        var = np.full(k, se**2)
        covered = 0
        for _ in range(reps):
            theta_true = true_tau * rng.standard_normal(k)
            est = theta_true + rng.standard_normal(k) * se
            r = marginalized_segment_intervals(est, var)
            lb, ub = np.asarray(r.lb), np.asarray(r.ub)
            covered += np.mean((lb <= theta_true) & (theta_true <= ub))
        coverage = covered / reps
        assert 0.945 <= coverage <= 0.985, (
            f"K=3 coverage {coverage:.1%} far from the measured ~96.9% baseline"
        )

    def test_coverage_smoke_k3_fast(self):
        """Fast-suite variant of the coverage check above: fewer reps, wide
        band - still exercises the calibration code path on every run."""
        rng = np.random.default_rng(0)
        k, se, true_tau, reps = 3, 0.095, 0.05, 150
        var = np.full(k, se**2)
        covered = 0
        for _ in range(reps):
            theta_true = true_tau * rng.standard_normal(k)
            est = theta_true + rng.standard_normal(k) * se
            r = marginalized_segment_intervals(est, var)
            lb, ub = np.asarray(r.lb), np.asarray(r.ub)
            covered += np.mean((lb <= theta_true) & (theta_true <= ub))
        coverage = covered / reps
        assert 0.85 <= coverage <= 1.0, f"K=3 smoke coverage {coverage:.1%} implausible"

    def test_guards_fewer_than_two_segments(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1], [0.02])
        assert exc_info.value.code == "estimation.meta.need_least_estimable"

    def test_guards_non_finite_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, 0.2, 0.3], [0.02, np.nan, 0.01])
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_guards_zero_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, 0.2, 0.3], [0.02, 0.0, 0.01])
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_guards_non_finite_estimate(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, np.nan, 0.3], [0.02, 0.05, 0.01])
        assert exc_info.value.code == "estimation.meta.est_contains_non"

    def test_guards_mismatched_shapes(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, 0.2, 0.3], [0.02, 0.05])
        assert exc_info.value.code == "estimation.meta.est_var_same"

    def test_guards_non_positive_prior_scale(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, 0.2], [0.02, 0.05], tau_prior_scale=0.0)
        assert exc_info.value.code == "estimation.meta.tau_prior_scale"

    def test_guards_non_finite_prior_scale(self):
        """A NaN prior scale passes a bare ``<= 0`` check (NaN comparisons
        are always false) and used to poison every posterior array with
        NaN, which the old result validator did not reject either."""
        for bad in (float("nan"), float("inf"), -float("inf")):
            with pytest.raises(InvalidRequestError) as exc_info:
                marginalized_segment_intervals([0.1, 0.2], [0.02, 0.05], tau_prior_scale=bad)
            assert exc_info.value.code == "estimation.meta.tau_prior_scale"

    def test_result_arrays_are_finite(self):
        """theta/shrink_k/lb/ub are now validated finite -- guards against
        any future path that could otherwise let NaN through silently."""
        r = marginalized_segment_intervals(self._EST, self._VAR)
        for name in ("theta", "shrink_k", "lb", "ub"):
            assert np.all(np.isfinite(np.asarray(getattr(r, name))))

    def test_guards_invalid_alpha(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, 0.2], [0.02, 0.05], alpha=0.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals([0.1, 0.2], [0.02, 0.05], alpha=1.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"

    def test_guards_2d_input(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals(np.zeros((3, 2)), np.ones((3, 2)))
        assert exc_info.value.code == "estimation.meta.est_var_shape"

    def test_guards_scalar_input(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals(np.array(0.1), np.array(0.02))
        assert exc_info.value.code == "estimation.meta.est_var_shape"

    def test_alpha_010_gives_narrower_interval(self):
        """A looser alpha=0.10 (90% interval) must report level=0.90 and a
        strictly narrower interval than the default alpha=0.05."""
        default = marginalized_segment_intervals(self._EST, self._VAR)
        looser = marginalized_segment_intervals(self._EST, self._VAR, alpha=0.10)
        assert looser.level == pytest.approx(0.90)
        default_width = np.asarray(default.ub) - np.asarray(default.lb)
        looser_width = np.asarray(looser.ub) - np.asarray(looser.lb)
        assert np.all(looser_width < default_width)

    def test_array_length_must_match_k(self):
        """A result whose array fields don't match the declared k must be
        rejected - not silently constructed with an inconsistent shape."""
        with pytest.raises(InvalidRequestError) as exc_info:
            MarginalizedSegmentIntervals(
                k=3,
                theta=np.array([0.1, 0.2]),
                shrink_k=np.array([0.1, 0.2]),
                lb=np.array([0.0, 0.1]),
                ub=np.array([0.2, 0.3]),
                level=0.95,
                alpha=0.05,
            )
        assert exc_info.value.code == "estimation.meta.marginalized_segment.length_shape"

    def test_2d_array_field_rejected(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            MarginalizedSegmentIntervals(
                k=2,
                theta=np.array([[0.1, 0.2]]),
                shrink_k=np.array([0.1, 0.2]),
                lb=np.array([0.0, 0.1]),
                ub=np.array([0.2, 0.3]),
                level=0.95,
                alpha=0.05,
            )
        assert exc_info.value.code == "estimation.meta.marginalized_segment.length_shape"

    def test_unhashable(self):
        """frozen=True's synthesized __hash__ would raise TypeError on the
        np.ndarray fields anyway - __hash__ = None makes that explicit."""
        r = marginalized_segment_intervals(self._EST, self._VAR)
        with pytest.raises(TypeError):
            hash(r)

    def test_result_is_frozen(self):
        r = marginalized_segment_intervals(self._EST, self._VAR)
        assert isinstance(r, MarginalizedSegmentIntervals)
        with pytest.raises((TypeError, ValueError)):
            r.level = 0.5  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_accepts_numpy_arrays(self):
        from_list = marginalized_segment_intervals(self._EST, self._VAR)
        from_array = marginalized_segment_intervals(np.array(self._EST), np.array(self._VAR))
        assert list(from_list.theta) == pytest.approx(list(from_array.theta))
        assert list(from_list.shrink_k) == pytest.approx(list(from_array.shrink_k))


class TestHksjPooledMean:
    """Hartung-Knapp-Sidik-Jonkman pooled interval: t_{K-1} critical
    value, no variance floor, paired with DL tau^2."""

    _EST = [0.10, 0.30, -0.05, 0.20, 0.0]
    _VAR = [0.02, 0.05, 0.01, 0.03, 0.015]

    def test_hand_computed(self):
        mu_hat, se, lb, ub = hksj_pooled_mean(self._EST, self._VAR, tau2=0.01)
        assert mu_hat == pytest.approx(0.06565656565656565)
        assert se == pytest.approx(0.058349182332484345)
        assert lb == pytest.approx(-0.09634673602275408)
        assert ub == pytest.approx(0.2276598673358854)

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_coverage_smoke(self):
        """Measured (not trusted from the design record): K=5, unequal SE,
        true tau=0.10 - matches the design record's reported 93.4-95.1%
        coverage range."""
        rng = np.random.default_rng(0)
        se_k = np.array([0.10, 0.15, 0.09, 0.12, 0.11])
        tau_true = 0.10
        reps = 4000
        covered = 0
        for _ in range(reps):
            theta_true = tau_true * rng.standard_normal(5)
            y = theta_true + rng.standard_normal(5) * se_k
            var = se_k**2
            tau2_hat = cochran_q(y, var).tau2
            _, _, lb, ub = hksj_pooled_mean(y, var, tau2_hat)
            covered += lb <= 0.0 <= ub
        coverage = covered / reps
        assert 0.90 <= coverage <= 0.99, f"coverage {coverage:.1%} far from nominal 95%"

    def test_no_variance_floor_regression(self):
        """The HKSJ design explicitly rejects flooring its variance at the
        plug-in variance - that would suppress exactly the cases where
        HKSJ is narrower because the plug-in over-covers. Regression: at
        tau2=0 HKSJ's SE can be smaller than the naive fixed-effect SE."""
        est = [0.10, 0.12, 0.09, 0.30, 0.08]  # one outlier segment
        var = [0.01, 0.01, 0.01, 0.01, 0.01]
        naive_se = float(np.sqrt(1.0 / np.sum(1.0 / np.array(var))))
        _, hksj_se, _, _ = hksj_pooled_mean(est, var, tau2=0.0)
        assert hksj_se < naive_se, (
            "HKSJ should be free to be narrower than the naive fixed-effect SE"
        )

    def test_guards_fewer_than_two_segments(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1], [0.02], tau2=0.0)
        assert exc_info.value.code == "estimation.meta.need_least_estimable"

    def test_guards_negative_tau2(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.2], [0.02, 0.05], tau2=-0.1)
        assert exc_info.value.code == "estimation.meta.tau2_finite"

    def test_guards_nonfinite_tau2(self):
        """NaN/inf tau2 must refuse, not silently return (nan, nan, nan, nan):
        `tau2 < 0` is False for NaN, and inf makes every weight 0 (0/0 mean).
        Every other malformed input to this function raises - non-finite
        tau2 was the one silent hole."""
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.3, -0.05], [0.02, 0.05, 0.01], tau2=float("nan"))
        assert exc_info.value.code == "estimation.meta.tau2_finite"
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.3, -0.05], [0.02, 0.05, 0.01], tau2=float("inf"))
        assert exc_info.value.code == "estimation.meta.tau2_finite"

    def test_guards_invalid_alpha(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.2], [0.02, 0.05], tau2=0.0, alpha=0.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.2], [0.02, 0.05], tau2=0.0, alpha=1.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"

    def test_tiny_alpha_returns_a_finite_wide_interval(self):
        """t.ppf(1 - alpha/2, dof) saturates to inf once the complement
        rounds to 1.0; t.isf(alpha/2, dof) resolves the tail directly."""
        _, _, lb, ub = hksj_pooled_mean([0.1, 0.2, 0.3], [0.02, 0.05, 0.01], tau2=0.0, alpha=1e-20)
        assert math.isfinite(lb)
        assert math.isfinite(ub)

    def test_guards_non_positive_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.2, 0.3], [0.02, 0.0, 0.01], tau2=0.0)
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_guards_mismatched_shapes(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean([0.1, 0.2, 0.3], [0.02, 0.05], tau2=0.0)
        assert exc_info.value.code == "estimation.meta.est_var_same"

    def test_guards_2d_input(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean(np.zeros((3, 2)), np.ones((3, 2)), tau2=0.0)
        assert exc_info.value.code == "estimation.meta.est_var_shape"

    def test_guards_scalar_input(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean(np.array(0.1), np.array(0.02), tau2=0.0)
        assert exc_info.value.code == "estimation.meta.est_var_shape"

    def test_guards_degenerate_pooled_variance(self):
        """Identical (or numerically indistinguishable) segment
        estimates give an exactly (or near-)zero pooled variance - a
        zero-width '95%' interval, which is degenerate, not narrow. This
        function refuses it rather than reporting a point-mass interval."""
        est = [0.1, 0.1, 0.1]
        var = [0.02, 0.05, 0.01]
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean(est, var, tau2=0.0)
        assert exc_info.value.code == "estimation.meta.hksj_pooled_variance"

        near_identical = [0.1, 0.1 + 1e-12, 0.1 - 1e-12]
        with pytest.raises(InvalidRequestError) as exc_info:
            hksj_pooled_mean(near_identical, var, tau2=0.0)
        assert exc_info.value.code == "estimation.meta.hksj_pooled_variance"


class TestMetaGuardsIntervalsNotJustCriticalValues:
    """A finite critical value times a large finite standard error overflows, so
    validating the critical value alone leaves an infinite bound reachable."""

    def test_a_huge_finite_tau_prior_scale_is_resolved_or_refused_as_unresolved(self):
        """1e308 is a finite, accepted prior scale: effectively the flat-prior
        posterior, whose tau tail decays only like tau^-(K-1). The call must
        report the converged answer or refuse with the shared numerical code
        -- never a scale-overflow refusal, a non-finite bound, or a truncated
        answer presented as converged."""
        try:
            result = marginalized_segment_intervals(FIXTURE_EST, FIXTURE_VAR, tau_prior_scale=1e308)
        except InvalidRequestError as exc:
            assert exc.code == "estimation.meta.posterior_integration_unresolved"
        else:
            reference = segment_posterior(FIXTURE_EST, FIXTURE_VAR, 1e308)
            assert_segment_intervals_match(result, reference)

    def test_wald_bounds_refuse_an_overflowing_half_width(self):
        from increment.estimation._tails import wald_bounds

        with pytest.raises(InvalidRequestError) as exc_info:
            wald_bounds(0.0, 1e200, 1e200, what="test interval")
        assert exc_info.value.code == "estimation.tails.unresolvable"

    def test_wald_bounds_return_finite_bounds_at_ordinary_scale(self):
        from increment.estimation._tails import wald_bounds

        assert wald_bounds(1.0, 1.96, 0.5, what="test interval") == pytest.approx((0.02, 1.98))


class TestPosteriorIntegrationUnresolved:
    """An accepted input the numerical budget cannot resolve is refused with
    one shared code carrying its numerical context, never returned as a
    normalised truncated answer. No ordinary input reaches this path, so the
    node budget is shrunk to reach it deterministically."""

    def test_refusal_carries_the_shared_code_and_numerical_context(self, monkeypatch):
        monkeypatch.setattr("increment.estimation.meta._TAU_NODE_BUDGET", 1)
        with pytest.raises(InvalidRequestError) as exc_info:
            marginalized_segment_intervals(FIXTURE_EST, FIXTURE_VAR, tau_prior_scale=0.3)
        assert exc_info.value.code == "estimation.meta.posterior_integration_unresolved"
        context = exc_info.value.context
        assert context["k"] == len(FIXTURE_EST)
        assert context["tau_prior_scale"] == 0.3
        assert {"support", "integration_error", "node_budget"} <= set(context)


def test_near_float_max_location_preserves_posterior_shrinkage():
    """The location may be huge even when the centred problem is ordinary."""
    variance = np.ones(5)
    centered = marginalized_segment_intervals(np.zeros(5), variance)
    shifted = marginalized_segment_intervals(np.full(5, 1.7e308), variance)
    np.testing.assert_allclose(shifted.theta, np.full(5, 1.7e308), rtol=1e-6)
    np.testing.assert_allclose(shifted.shrink_k, centered.shrink_k, rtol=1e-6, atol=1e-10)
