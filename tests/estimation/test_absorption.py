"""Unit tests for one-way factor absorption."""

from __future__ import annotations

import math

import numpy as np
import pytest

from increment.errors import IncrementWarning, InvalidRequestError
from increment.estimation.absorption import (
    _absorption_interval,
    absorb_one_way,
)
from increment.estimation.armstats import centered_sq_sum
from tests.warning_codes import warning_codes

# Four levels, deliberately unequal cell sizes and variances: equal sizes
# would let pooled and within estimators coincide, hiding a weighting bug.
N_C = np.array([10.0, 20.0, 5.0, 40.0])
S_C = np.array([10.0, 40.0, 2.5, 120.0])
Q_C = np.array([20.0, 100.0, 6.5, 400.0])
N_T = np.array([12.0, 18.0, 6.0, 36.0])
S_T = np.array([18.0, 45.0, 4.8, 129.6])
Q_T = np.array([36.0, 130.0, 10.0, 500.0])


def _dense(n_c, s_c, q_c, n_t, s_t, q_t):
    """Expand cell moments into the unit-level (g, D, y) arrays they imply.

    Only the first two moments are reproduced exactly, which is all the
    estimator consumes.
    """
    g, d, y = [], [], []
    for k in range(len(n_c)):
        for arm, (n, s, q) in enumerate(((n_c[k], s_c[k], q_c[k]), (n_t[k], s_t[k], q_t[k]))):
            n = int(n)
            mean = s / n
            var = max((q - s * s / n) / n, 0.0)
            vals = np.full(n, mean)
            if n > 1 and var > 0:
                vals[0] += np.sqrt(var * n / 2.0)
                vals[1] -= np.sqrt(var * n / 2.0)
            g.extend([k] * n)
            d.extend([arm] * n)
            y.extend(vals.tolist())
    return np.array(g), np.array(d, dtype=float), np.array(y)


class TestEndpointNesting:
    def test_no_pooling_reproduces_unadjusted_difference_in_means(self):
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, pooling="none")
        expected = S_T.sum() / N_T.sum() - S_C.sum() / N_C.sum()
        assert res.effect == pytest.approx(expected, rel=1e-12)

    def test_hard_pooling_reproduces_the_within_estimator(self):
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, pooling="hard")
        w = N_T * N_C / (N_T + N_C)
        expected = (w * (S_T / N_T - S_C / N_C)).sum() / w.sum()
        assert res.effect == pytest.approx(expected, rel=1e-12)

    def test_hard_pooling_equals_ols_with_a_full_dummy_set(self):
        """The whole premise: cell moments carry the K-dummy fit exactly."""
        g, d, y = _dense(N_C, S_C, Q_C, N_T, S_T, Q_T)
        design = np.column_stack([d, np.eye(len(N_C))[g]])
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, pooling="hard")
        assert res.effect == pytest.approx(beta[0], abs=1e-10)

    def test_partial_pooling_lies_between_the_endpoints(self):
        none = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, pooling="none").effect
        hard = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, pooling="hard").effect
        part = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T).effect
        assert min(none, hard) - 1e-9 <= part <= max(none, hard) + 1e-9


class TestDegenerateLevels:
    def test_levels_below_two_units_per_arm_are_dropped_and_counted(self):
        # Four levels so three remain after dropping one below the
        # >= 3-usable-levels floor a t_{K-2} reference needs.
        n_c = np.array([10.0, 1.0, 20.0, 15.0])
        n_t = np.array([10.0, 8.0, 20.0, 12.0])
        s_c = np.array([10.0, 1.0, 20.0, 15.0])
        s_t = np.array([12.0, 9.0, 24.0, 14.0])
        q_c = np.array([20.0, 1.0, 40.0, 30.0])
        q_t = np.array([26.0, 12.0, 52.0, 30.0])
        res = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t)
        assert res.n_levels_used == 3
        assert res.n_levels_dropped == 1
        assert res.n_units_dropped == 9

    def test_fewer_than_three_usable_levels_raises(self):
        n_c = np.array([10.0, 20.0, 1.0])
        n_t = np.array([10.0, 20.0, 1.0])
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(n_c, n_c, n_c * 2, n_t, n_t, n_t * 2)
        assert exc_info.value.code == "estimation.absorption.need_least_levels"

    def test_exactly_two_usable_levels_is_refused_not_fabricated(self):
        """Two usable levels used to be accepted with a fabricated df=1
        critical value (the real t_{K-2} reference needs K>=3, since
        K=2 gives df=0, which has no t reference); now refused outright."""
        n_c = np.array([10.0, 20.0])
        n_t = np.array([10.0, 20.0])
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(n_c, n_c, n_c * 2, n_t, n_t, n_t * 2)
        assert exc_info.value.code == "estimation.absorption.need_least_levels"
        assert exc_info.value.context["n_usable"] == 2
        assert exc_info.value.context["min_levels"] == 3

    def test_mismatched_array_lengths_raise(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(N_C[:3], S_C, Q_C, N_T, S_T, Q_T)
        assert exc_info.value.code == "estimation.absorption.all_moment_arrays"

    def test_negative_counts_raise(self):
        bad = N_C.copy()
        bad[0] = -1.0
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(bad, S_C, Q_C, N_T, S_T, Q_T)
        assert exc_info.value.code == "estimation.absorption.non_negative"

    def test_unknown_pooling_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(
                N_C,
                S_C,
                Q_C,
                N_T,
                S_T,
                Q_T,
                pooling="bogus",  # ty: ignore[invalid-argument-type]  - proving the runtime check catches untyped callers
            )
        assert exc_info.value.code == "estimation.absorption.pooling_one"
        assert exc_info.value.context["pooling"] == "bogus"


class TestFractionalCounts:
    """Unit counts must be exact integers, converted once at ingress.

    A fractional count used to be rounded for centered moments while the
    original fractional value survived in the GLS weights, producing an
    internally inconsistent estimate.
    """

    def test_fractional_control_count_raises(self):
        bad = N_C.copy()
        bad[0] = 10.5
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(bad, S_C, Q_C, N_T, S_T, Q_T)
        assert exc_info.value.code == "estimation.absorption.contain_exact_integer"

    def test_fractional_treat_count_raises(self):
        bad = N_T.copy()
        bad[0] = 12.25
        with pytest.raises(InvalidRequestError) as exc_info:
            absorb_one_way(N_C, S_C, Q_C, bad, S_T, Q_T)
        assert exc_info.value.code == "estimation.absorption.contain_exact_integer"


class TestTinyAlpha:
    """Tiny alpha must resolve a finite (very wide) interval via isf, not a
    silent NaN from a ppf(1 - alpha/2) complement rounding to 1.0."""

    def test_tiny_alpha_returns_finite_interval_not_nan(self):
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, alpha=1e-20)
        assert np.isfinite(res.lb)
        assert np.isfinite(res.ub)
        assert res.lb < res.effect < res.ub

    def test_subnormal_tail_resolves_a_finite_critical_value_scipy_underflows(self):
        # k=12 gives dof 10; alpha=1e-300 puts the one-sided tail at 5e-301, past
        # SciPy t.isf's underflow (~1e-300) though the quantile is representable.
        # mpmath reference at ~100 digits:
        # student_t_isf(5e-301, 10) == 2.7485906095604865904463218957406e30.
        n_c = np.full(12, 20.0)
        n_t = np.full(12, 20.0)
        s_c = np.full(12, 20.0)
        s_t = np.full(12, 22.0)
        q_c = np.full(12, 60.0)
        q_t = np.full(12, 68.0)
        res = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t, alpha=1e-300)
        assert np.isfinite(res.lb)
        assert np.isfinite(res.ub)
        assert res.lb < res.effect < res.ub
        crit = (res.ub - res.effect) / res.se
        assert crit == pytest.approx(2.7485906095604865904e30, rel=1e-9, abs=0.0)


class TestResultShape:
    def test_result_is_frozen(self):
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T)
        with pytest.raises((TypeError, ValueError)):
            res.effect = 1.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_interval_brackets_the_point_estimate(self):
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T)
        assert res.lb < res.effect < res.ub
        assert res.level == pytest.approx(0.95)

    def test_lists_and_arrays_agree(self):
        a = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T)
        b = absorb_one_way(
            N_C.tolist(),
            S_C.tolist(),
            Q_C.tolist(),
            N_T.tolist(),
            S_T.tolist(),
            Q_T.tolist(),
        )
        assert a.effect == pytest.approx(b.effect)
        assert a.se == pytest.approx(b.se)

    def test_icc_is_zero_when_levels_are_identical(self):
        n = np.full(6, 20.0)
        s = np.full(6, 20.0)
        q = np.full(6, 60.0)
        res = absorb_one_way(n, s, q, n, s * 1.2, q * 1.5)
        assert res.icc == pytest.approx(0.0, abs=1e-9)
        assert res.mean_shrinkage == pytest.approx(0.0, abs=1e-9)

    def test_se_reduction_is_reported(self):
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T)
        assert res.se_unadjusted > 0
        assert res.se_reduction == pytest.approx((res.se_unadjusted - res.se) / res.se_unadjusted)


class TestSandwichSEExactness:
    def test_pooling_none_se_matches_hand_rolled_cr1_cluster_robust(self):
        """pooling='none' collapses to plain OLS on [1, D] with cluster-robust
        (CR1) SE clustered by level - verify against an independent dense
        computation, not the module's own internals."""
        g, d, y = _dense(N_C, S_C, Q_C, N_T, S_T, Q_T)
        X = np.column_stack([np.ones(len(d)), d])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = np.zeros((2, 2))
        for level in np.unique(g):
            mask = g == level
            Xg = X[mask]
            score = Xg.T @ resid[mask]
            meat += np.outer(score, score)
        k = len(np.unique(g))
        meat *= k / max(k - 1, 1)
        cov = XtX_inv @ meat @ XtX_inv
        expected_se = np.sqrt(cov[1, 1])

        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, pooling="none")
        assert res.se == pytest.approx(expected_se, rel=1e-8)

    def test_variance_components_and_shrinkage_match_hand_evaluated_moment_equation(self):
        """The one-way MoM between-level variance component (pinned via
        ``icc``) and the n_g-weighted shrinkage map it implies (pinned via
        ``mean_shrinkage``, ``lam_g = n_g*sigma_b2/(sigma_e2+n_g*sigma_b2)``),
        both evaluated by hand from the fixture's cell moments, independent
        of the module's internal accumulation path. ``mean_shrinkage`` is a
        deterministic function of ``icc`` and the ``n_g`` here (``lam_g =
        n_g*r/(1+n_g*r)`` with ``r = icc/(1-icc)``), so its real value is
        pinning that mapping and confirming this fixture lands well off
        both pooling endpoints (``lam ~= 0.967``), not adding scale
        sensitivity beyond the ``icc`` check."""
        n_g = N_C + N_T
        S_g = S_C + S_T
        w_fe = N_T * N_C / n_g
        tau_fe = float((w_fe * (S_T / N_T - S_C / N_C)).sum() / w_fe.sum())
        within_ss = float((Q_C - S_C**2 / N_C).sum() + (Q_T - S_T**2 / N_T).sum())
        k = len(N_C)
        df_w = n_g.sum() - 2 * k
        sigma_e2 = within_ss / df_w
        level_mean = S_g / n_g - tau_fe * (N_T / n_g)
        grand = float((n_g * level_mean).sum() / n_g.sum())
        ss_between = float((n_g * (level_mean - grand) ** 2).sum())
        denom = n_g.sum() - (n_g**2).sum() / n_g.sum()
        expected_sigma_b2 = max(0.0, (ss_between - (k - 1) * sigma_e2) / denom)

        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T)
        total_var = expected_sigma_b2 + sigma_e2
        expected_icc = expected_sigma_b2 / total_var if total_var > 0 else 0.0
        assert res.icc == pytest.approx(expected_icc, rel=1e-8)
        expected_lam = n_g * expected_sigma_b2 / (sigma_e2 + n_g * expected_sigma_b2)
        assert res.mean_shrinkage == pytest.approx(float(expected_lam.mean()), rel=1e-8)


def _cell(n: float, mean: float, var: float) -> tuple[float, float, float]:
    """Exact cell moments (n, sum, sumsq) for a given mean and ddof-1 variance."""
    return n, n * mean, n * mean**2 + (n - 1.0) * var


class TestImbalanceWarning:
    """Partial pooling warns when varying allocation confounds the factor.

    The identifying assumption for ``pooling="partial"`` is a treatment
    share that is constant across levels; when it varies and level effects
    correlate with it, the within (hard) and unpooled (none) estimates
    diverge far beyond noise - the Hausman signal - and the default must
    say so rather than ship the partially-confounded number silently.
    """

    def test_confounded_allocation_warns_and_names_hard_pooling(self):
        # Level effect alpha_k = k rises with treatment share (3%->97%): the
        # unpooled contrast absorbs the level gradient; within contrast is tau=1.0.
        n_c_l, n_t_l = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        n_c, s_c, q_c, n_t, s_t, q_t = ([] for _ in range(6))
        for k in range(6):
            n, s, q = _cell(n_c_l[k], float(k), 0.01)
            n_c.append(n), s_c.append(s), q_c.append(q)
            n, s, q = _cell(n_t_l[k], float(k) + 1.0, 0.01)
            n_t.append(n), s_t.append(s), q_t.append(q)

        with pytest.warns(IncrementWarning) as rec:
            res = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t, pooling="partial")
        assert "estimation.absorption.treatment_share_confounded" in warning_codes(rec)
        # The warning is advisory: the (partially confounded) result still returns.
        assert np.isfinite(res.effect)

    def test_constant_allocation_does_not_warn(self):
        import warnings

        # Reuse the fixture's per-level means/variances but resample each
        # cell at a shared n=20, so moments stay internally consistent
        # (unlike swapping in N_C's counts while keeping S_C/Q_C's sums).
        mean_c, var_c = S_C / N_C, (Q_C - S_C**2 / N_C) / (N_C - 1.0)
        mean_t, var_t = S_T / N_T, (Q_T - S_T**2 / N_T) / (N_T - 1.0)
        n_c, s_c, q_c, n_t, s_t, q_t = ([] for _ in range(6))
        for k in range(len(N_C)):
            n, s, q = _cell(20.0, float(mean_c[k]), float(var_c[k]))
            n_c.append(n), s_c.append(s), q_c.append(q)
            n, s, q = _cell(20.0, float(mean_t[k]), float(var_t[k]))
            n_t.append(n), s_t.append(s), q_t.append(q)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t, pooling="partial")

    def test_forced_hard_pooling_never_warns(self):
        import warnings

        n_c_l, n_t_l = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        n_c, s_c, q_c, n_t, s_t, q_t = ([] for _ in range(6))
        for k in range(6):
            n, s, q = _cell(n_c_l[k], float(k), 0.01)
            n_c.append(n), s_c.append(s), q_c.append(q)
            n, s, q = _cell(n_t_l[k], float(k) + 1.0, 0.01)
            n_t.append(n), s_t.append(s), q_t.append(q)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t, pooling="hard")


class TestIccUnderCancellation:
    """icc is a variance-component ratio and must stay inside [0, 1].

    At a large common offset (mean ~3e8, sd 1) the per-cell centered sums
    cancel and the within component can go negative; the estimate and SE
    are protected by the sigma_e2 <= 0 hard-absorption fallback, but the
    emitted icc used to exceed 1 (e.g. 2.93 for this construction).
    """

    @staticmethod
    def _cells(seed: int, offset: float):
        rng = np.random.default_rng(seed)
        k, m = 20, 50
        level = rng.normal(0.0, 0.7, k)
        n_c, s_c, q_c, n_t, s_t, q_t = ([] for _ in range(6))
        for j in range(k):
            yc = rng.normal(offset + level[j], 1.0, m)
            yt = rng.normal(offset + level[j], 1.0, m)
            n_c.append(float(m))
            s_c.append(float(np.sum(yc)))
            q_c.append(float(np.sum(yc * yc)))
            n_t.append(float(m))
            s_t.append(float(np.sum(yt)))
            q_t.append(float(np.sum(yt * yt)))
        return n_c, s_c, q_c, n_t, s_t, q_t

    @pytest.mark.filterwarnings(
        "ignore:.*centered (sum of squares|cross sum).*floating-point noise.*:RuntimeWarning"
    )
    def test_icc_clamped_to_unit_interval_at_extreme_mean(self):
        res = absorb_one_way(*self._cells(seed=2, offset=3e8))
        assert 0.0 <= res.icc <= 1.0

    @pytest.mark.filterwarnings(
        "ignore:.*centered (sum of squares|cross sum).*floating-point noise.*:RuntimeWarning"
    )
    def test_effect_and_se_shift_stable(self):
        """The point estimate and SE must not move with the offset (they are
        contrasts of means; the cancellation only corrupts the variance
        components)."""
        base = absorb_one_way(*self._cells(seed=2, offset=0.0))
        shifted = absorb_one_way(*self._cells(seed=2, offset=3e8))
        assert shifted.effect == pytest.approx(base.effect, abs=1e-5)
        assert shifted.se == pytest.approx(base.se, rel=1e-4)


class TestSeUnadjustedUnderCancellation:
    """``se_unadjusted`` must not silently drop a real variance component.

    At a control-arm mean far above its spread, ``sum(y**2) - n*mean**2``
    (the naive shifted sum of squares) cancels to <= 0 and used to be
    floored to 0 by a ``max(var, 0.0)`` clamp - reporting the control arm
    as if it had no variance at all. The exact centered-sum-of-squares
    route must instead recover a non-zero, right-order-of-magnitude
    variance. At this offset the summary sums (``sum(y)``, ``sum(y**2)``)
    have themselves already lost precision below the true signal before
    ``centered_sq_sum`` ever sees them, so an exact match to a directly
    centered computation is not guaranteed - only that the result is
    materially bigger than the floored-to-zero failure mode.
    """

    def test_recovers_control_variance_instead_of_zeroing_it(self):
        n_lvl, offset, sd = 500, 3e8, 0.5

        def cell_from_draws(seed: int):
            rng = np.random.default_rng(seed)
            y = offset + rng.normal(0.0, sd, n_lvl)
            return y, float(np.sum(y)), float(np.sum(y * y))

        y1, s1, q1 = cell_from_draws(1)
        y2, s2, q2 = cell_from_draws(121)
        y3, s3, q3 = cell_from_draws(233)
        n_c = np.array([float(n_lvl)] * 3)
        s_c = np.array([s1, s2, s3])
        q_c = np.array([q1, q2, q3])
        nc_tot = float(n_c.sum())
        mean_c = float(s_c.sum()) / nc_tot

        # The failure mode being fixed: the naive shifted formula really
        # does cancel to a non-positive value for this fixture.
        naive_var_c = float(q_c.sum()) - nc_tot * mean_c**2
        assert naive_var_c <= 0.0

        # Treatment arm carries no cancellation, so its variance is exact
        # and can anchor the expected combined SE.
        n_t0, s_t0, q_t0 = _cell(float(n_lvl), 2.0, 1.0)
        n_t1, s_t1, q_t1 = _cell(float(n_lvl), 2.5, 1.0)
        n_t2, s_t2, q_t2 = _cell(float(n_lvl), 2.2, 1.0)
        n_t = np.array([n_t0, n_t1, n_t2])
        s_t = np.array([s_t0, s_t1, s_t2])
        q_t = np.array([q_t0, q_t1, q_t2])
        nt_tot = float(n_t.sum())

        with pytest.warns(RuntimeWarning):
            res = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t)

        exact_var_t = centered_sq_sum(
            float(q_t.sum()), float(s_t.sum()), int(nt_tot), what="treat variance"
        ) / (nt_tot - 1.0)
        floor_se = float(np.sqrt(exact_var_t / nt_tot))
        assert res.se_unadjusted > floor_se

        # Directly centered reference variance, independent of the module's
        # accumulation. The summary sums already lost precision at this offset,
        # so only order-of-magnitude agreement is guaranteed.
        direct_var_c = float(np.sum((np.concatenate([y1, y2, y3]) - mean_c) ** 2)) / (nc_tot - 1.0)
        expected_se = float(np.sqrt(exact_var_t / nt_tot + direct_var_c / nc_tot))
        assert expected_se / 5.0 < res.se_unadjusted < expected_se * 5.0


class TestAbsorptionIntervalAtExtremeScales:
    """A tiny alpha and a huge standard error each break a different half of
    the interval contract."""

    def test_a_tiny_alpha_is_retained_even_when_level_rounds_to_one(self):
        # level = 1 - 1e-20 is exactly 1.0 in float64, so a serialized result
        # could not otherwise recover the level it was computed at.
        res = absorb_one_way(N_C, S_C, Q_C, N_T, S_T, Q_T, alpha=1e-20)
        assert res.level == 1.0
        assert res.alpha == 1e-20
        assert res.model_dump()["alpha"] == 1e-20
        # The interval must widen at that level, not collapse.
        assert res.ub - res.lb > 0.0
        assert math.isfinite(res.lb)
        assert math.isfinite(res.ub)

    @pytest.mark.parametrize("alpha", [1e-300, 1e-100, 1e-20, 0.05])
    @pytest.mark.parametrize("scale", [1.0, 1e5, 1e10, 1e150])
    def test_bounds_are_finite_or_refused_never_infinite(self, alpha, scale):
        """The stated behavior is a finite interval. Guarding only the critical
        value left the product unchecked, so a finite critical value times a
        large finite standard error could return infinite bounds. Across the
        representable range the call must either refuse or return finite
        bounds -- never hand back an infinite one."""
        n = np.array([50.0, 50.0, 50.0])
        mean_c = np.array([10.0, 11.0, 12.0]) * scale
        mean_t = np.array([11.0, 12.0, 13.0]) * scale
        try:
            res = absorb_one_way(
                n,
                mean_c * 50,
                mean_c**2 * 50 * 1.02,
                n,
                mean_t * 50,
                mean_t**2 * 50 * 1.02,
                alpha=alpha,
            )
        except ValueError:
            return
        assert math.isfinite(res.lb)
        assert math.isfinite(res.ub)
        assert res.alpha == alpha

    def test_the_half_width_guard_refuses_an_overflowing_product(self):
        """The guard itself, at its own boundary: the public path is normally
        protected by input validation reaching a non-finite moment first, so
        this exercises the interval arithmetic directly."""
        crit, se = 1e200, 1e200
        assert not math.isfinite(crit * se)
        with pytest.raises(InvalidRequestError) as exc_info:
            _absorption_interval(effect=0.0, crit=crit, se=se)
        assert exc_info.value.code == "estimation.absorption.absorption_interval_half"
        assert exc_info.value.context["crit"] == crit
        assert exc_info.value.context["se"] == se


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.absorption.one_dimensional_shape",
            lambda: absorb_one_way(N_C, np.array([[1.0, 2.0], [3.0, 4.0]]), Q_C, N_T, S_T, Q_T),
        ),
        (
            "estimation.absorption.non_negative",
            lambda: absorb_one_way(np.array([-1.0, 2.0]), S_C, Q_C, N_T, S_T, Q_T),
        ),
        (
            "estimation.absorption.absorption_interval_half",
            lambda: _absorption_interval(effect=0.0, crit=1e200, se=1e200),
        ),  # estimation/absorption.py::_absorption_interval
        (
            "estimation.absorption.pooling_one",
            lambda: absorb_one_way(
                N_C,
                S_C,
                Q_C,
                N_T,
                S_T,
                Q_T,
                pooling="bogus",  # ty: ignore[invalid-argument-type]
            ),
        ),  # estimation/absorption.py::absorb_one_way
    ],
)
def test_absorption_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
