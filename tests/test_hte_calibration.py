"""Consolidated calibration for the segment-heterogeneity machinery.

Every number pinned here was independently re-measured in this session;
floors are set with real margin below what was observed - generous enough
not to flake under normal MC noise but tight enough to catch a genuinely
broken implementation (e.g. Q type-I at 20%, or coverage collapsing to 50%).

Most tests are marked ``@pytest.mark.parameter_recovery`` and excluded from
the fast suite; unmarked fast/smoke variants below run by default.
Everything runs on (est, var)-level simulation (no raw-unit generation
needed for a property that only depends on the sufficient statistics'
sampling distribution) EXCEPT the D2a scale check, which must go through
the real ``estimate_lift``/``run_breakout``/``segment_heterogeneity``
pipeline - the risk it guards lives in the per-family absolute-scale
derivation, not the pure math, so hand-feeding moments to ``cochran_q``
would not exercise it.

Most sections also have a small, unmarked fast variant: either a
~100-400-rep Monte Carlo smoke test with a deliberately wide band (catches
"the code doesn't run", not precise calibration) or a deterministic check
that needs no simulation at all.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2, norm

from increment.breakout.estimates import run_breakout
from increment.breakout.heterogeneity import segment_heterogeneity
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.meta import (
    cochran_q,
    hksj_pooled_mean,
    marginalized_segment_intervals,
)
from increment.power import segment_pairwise_required_sample_size
from increment.power.core import (
    Baseline,
    PowerDesign,
    joint_q_power_fixed,
)
from increment.semantics.models import MeanMetric
from tests.power._procedures import make_procedure

# Unequal per-segment shares at K=3/5/10: every known failure mode in this
# machinery is worst at small K with imbalance.
_SHARES_BY_K = {
    3: [0.5, 0.3, 0.2],
    5: [0.4, 0.25, 0.15, 0.12, 0.08],
    10: [0.3, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03],
}
_BASE_SE = 0.10  # SE at share=1; SE(share) = _BASE_SE / sqrt(share)


def _unequal_se(k: int) -> np.ndarray:
    return _BASE_SE / np.sqrt(np.array(_SHARES_BY_K[k]))


# The pure helper above is exercised by every test in this file, so a
# broken share/SE fixture would silently invalidate the whole suite.


def test_unequal_se_fixture_shape():
    for k in (3, 5, 10):
        se = _unequal_se(k)
        assert se.shape == (k,)
        assert np.all(se > 0)
        assert not np.allclose(se, se[0]), f"K={k} SE fixture must be genuinely unequal"


# 1. Q type-I and power, K in {3, 5, 10}, unequal shares


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestQTypeIAndPower:
    @pytest.mark.parametrize("k", [3, 5, 10])
    def test_type_i_near_nominal(self, k):
        """Measured (2000 reps): 4.7% (K=3), 5.05% (K=5), 4.7% (K=10). Band
        is two-sided - a few MC-SE (~0.5pp at this rep count) either side
        of nominal 5%, not a one-sided floor a broken implementation
        (wrong tail, wrong df) could clear by reading far above 5% but
        under a loose cap."""
        rng = np.random.default_rng(0)
        se = _unequal_se(k)
        var = se**2
        reps = 2000
        rejections = 0
        for _ in range(reps):
            y = rng.standard_normal(k) * se
            rejections += cochran_q(y, var).p_value < 0.05
        rate = rejections / reps
        assert 0.035 <= rate <= 0.075, f"K={k} type-I {rate:.1%} far from nominal 5%"

    @pytest.mark.parametrize("k", [3, 5, 10])
    def test_power_increases_with_tau(self, k):
        """Paired Monte Carlo: the same per-rep heterogeneity SHAPE and
        sampling noise are reused for tau=0.05 and tau=0.10, isolating the
        effect of tau itself rather than comparing one tau's rejection
        rate against an arbitrary threshold.

        Measured (2000 reps): tau=0.05 / tau=0.10 rejection rates 6.35% /
        9.95% (K=3), 5.85% / 9.00% (K=5), 5.90% / 7.75% (K=10) - the
        larger-tau rate exceeds the smaller by >=1pp at every K, several
        MC-SE of margin."""
        rng = np.random.default_rng(1)
        se = _unequal_se(k)
        var = se**2
        reps = 2000
        rej_lo = rej_hi = 0
        for _ in range(reps):
            z = rng.standard_normal(k)
            noise = rng.standard_normal(k) * se
            y_lo = 0.05 * z + noise
            y_hi = 0.10 * z + noise
            rej_lo += cochran_q(y_lo, var).p_value < 0.05
            rej_hi += cochran_q(y_hi, var).p_value < 0.05
        rate_lo, rate_hi = rej_lo / reps, rej_hi / reps
        assert rate_hi - rate_lo > 0.01, (
            f"K={k} power did not increase enough with tau: "
            f"{rate_lo:.1%} (tau=0.05) -> {rate_hi:.1%} (tau=0.10)"
        )


def test_q_type_i_smoke():
    """Fast, unmarked ~400-rep variant of TestQTypeIAndPower.test_type_i_near_nominal.

    Floor 0.01 (>=4 rejections of 400) is ~3.7 SD below a true 5% rate
    (P < 2e-6), so it can't flake while still catching a near-never-
    rejecting implementation. Measured: 5.25% at K=3, 400 reps."""
    rng = np.random.default_rng(100)
    k = 3
    se = _unequal_se(k)
    var = se**2
    reps = 400
    rejections = 0
    for _ in range(reps):
        y = rng.standard_normal(k) * se
        rejections += cochran_q(y, var).p_value < 0.05
    rate = rejections / reps
    assert 0.01 <= rate <= 0.15, f"Q type-I smoke rate {rate:.1%} wildly off nominal 5%"


# 2. DL tau^2 recovery and the exact boundary-collapse rate


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestTau2Recovery:
    @pytest.mark.parametrize("k", [3, 5, 10])
    def test_boundary_collapse_rate_unequal_share_fixture(self, k):
        """P(tau2_hat = 0 at true tau = 0) = P(chi2_{K-1} < K-1) EXACTLY -
        no simulation needed for the exact value; simulation only confirms
        cochran_q's actual collapse rate matches it. Complements
        tests/estimation/test_meta.py's equal-variance/raw-unit check by
        running this file's unequal-share, summary-statistic fixture
        across K=3/5/10."""
        exact = chi2.cdf(k - 1, k - 1)
        rng = np.random.default_rng(2)
        se = _unequal_se(k)
        var = se**2
        reps = 3000
        collapsed = sum(
            cochran_q(rng.standard_normal(k) * se, var).tau2 == 0.0 for _ in range(reps)
        )
        empirical = collapsed / reps
        assert empirical == pytest.approx(exact, abs=0.03)

    @pytest.mark.parametrize("k", [5, 10])
    def test_tau2_recovers_true_value_on_average(self, k):
        """E[tau2_hat] ~= true tau^2 at true tau=0.30 - DL's max(0, ...)
        truncation gives well-documented UPWARD bias when true tau^2 is
        small relative to sampling noise (2.2x inflation measured at
        true_tau=0.10), so recovery is only checkable once that bias is
        small relative to the signal. Measured at true_tau=0.30: 0.0932 vs
        true 0.09 (K=5), 0.0898 vs 0.09 (K=10) - both within ~5%."""
        rng = np.random.default_rng(3)
        se = _unequal_se(k)
        var = se**2
        true_tau = 0.30
        true_tau2 = true_tau**2
        reps = 1500
        estimates = [
            cochran_q(true_tau * rng.standard_normal(k) + rng.standard_normal(k) * se, var).tau2
            for _ in range(reps)
        ]
        assert np.mean(estimates) == pytest.approx(true_tau2, rel=0.3)


# 3. Per-segment (marginalised) interval coverage, numeric floors per K


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPerSegmentIntervalCoverage:
    """Reference values measured (1000 reps, tau=0.05, unequal shares):
    97.2% (K=3), 98.6% (K=5), 99.5% (K=10). Both floor AND ceiling are set
    with real margin around these - a one-sided floor alone would accept
    anything up to 100% coverage. Complements
    tests/estimation/test_meta.py::TestMarginalizedSegmentIntervals's
    equal-SE K=3 check by exercising K=3/5/10 with this file's
    unequal-share fixture."""

    @pytest.mark.parametrize(
        ("k", "floor", "ceiling"),
        [(3, 0.95, 0.995), (5, 0.97, 0.998), (10, 0.985, 0.999)],
    )
    def test_coverage_band(self, k, floor, ceiling):
        rng = np.random.default_rng(4)
        se = _unequal_se(k)
        var = se**2
        reps = 1000
        covered = 0
        for _ in range(reps):
            theta_true = 0.05 * rng.standard_normal(k)
            y = theta_true + rng.standard_normal(k) * se
            r = marginalized_segment_intervals(y, var)
            lb, ub = np.asarray(r.lb), np.asarray(r.ub)
            covered += np.mean((lb <= theta_true) & (theta_true <= ub))
        coverage = covered / reps
        assert floor <= coverage <= ceiling, (
            f"K={k} coverage {coverage:.1%} outside [{floor:.1%}, {ceiling:.1%}]"
        )

    @pytest.mark.slow
    def test_coverage_floor_k100(self):
        """K=100 marginalised intervals, kept behind @slow. Measured: 90.0%
        coverage - close to, not substantially over, nominal, so a
        floor-only assertion is appropriate here."""
        rng = np.random.default_rng(5)
        k = 100
        se = 0.095 * np.ones(k)
        var = se**2
        reps = 300
        covered = 0
        for _ in range(reps):
            theta_true = 0.05 * rng.standard_normal(k)
            y = theta_true + rng.standard_normal(k) * se
            r = marginalized_segment_intervals(y, var)
            lb, ub = np.asarray(r.lb), np.asarray(r.ub)
            covered += np.mean((lb <= theta_true) & (theta_true <= ub))
        coverage = covered / reps
        assert coverage >= 0.85, f"K=100 coverage {coverage:.1%} below floor"


def test_marginalised_coverage_smoke():
    """Fast, unmarked ~100-rep variant of TestPerSegmentIntervalCoverage, at
    K=10 to exercise per-K structure the K=3 equal-SE sibling test in
    tests/estimation/test_meta.py doesn't cover. Deliberately wide band -
    catches coverage collapsed toward 0%, not precise calibration.
    Measured: 99.6% at K=10, 100 reps."""
    rng = np.random.default_rng(101)
    k = 10
    se = _unequal_se(k)
    var = se**2
    reps = 100
    covered = 0
    for _ in range(reps):
        theta_true = 0.05 * rng.standard_normal(k)
        y = theta_true + rng.standard_normal(k) * se
        r = marginalized_segment_intervals(y, var)
        lb, ub = np.asarray(r.lb), np.asarray(r.ub)
        covered += np.mean((lb <= theta_true) & (theta_true <= ub))
    coverage = covered / reps
    assert 0.90 <= coverage <= 0.999, f"marginalised coverage smoke {coverage:.1%} wildly off"


# 4. Pooled HKSJ coverage


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestHksjPooledCalibration:
    """K=3, unequal SE. Measured (1500 reps): 94.9% coverage at tau=0,
    94.2% at tau=0.10. Both floor AND ceiling are set with real margin
    around these - a floor-only assertion can't detect an always-too-wide
    interval.

    The no-variance-flooring regression (HKSJ's SE must be free to fall
    below the naive plug-in Wald SE) is a deterministic single-fixture
    check in tests/estimation/test_meta.py, strictly better for that
    purpose than a Monte Carlo version, so it is not duplicated here."""

    _SE = np.array([0.10, 0.13, 0.17])

    @pytest.mark.parametrize("tau_true", [0.0, 0.10])
    def test_coverage_band(self, tau_true):
        rng = np.random.default_rng(6)
        var = self._SE**2
        k = len(self._SE)
        reps = 1500
        covered = 0
        for _ in range(reps):
            theta = tau_true * rng.standard_normal(k)
            y = theta + rng.standard_normal(k) * self._SE
            het = cochran_q(y, var)
            _, _, lb, ub = hksj_pooled_mean(y, var, het.tau2)
            covered += lb <= 0.0 <= ub
        coverage = covered / reps
        assert 0.90 <= coverage <= 0.975, (
            f"HKSJ coverage {coverage:.1%} (tau={tau_true}) outside [90.0%, 97.5%]"
        )


def test_hksj_coverage_smoke():
    """Fast, unmarked ~100-rep variant of TestHksjPooledCalibration.
    Deliberately wide band - catches coverage collapsed toward 0%, not
    precise calibration. Measured: 97.0% at tau=0, 100 reps."""
    rng = np.random.default_rng(102)
    se_k = np.array([0.10, 0.13, 0.17])
    var = se_k**2
    k = len(se_k)
    reps = 100
    covered = 0
    for _ in range(reps):
        y = rng.standard_normal(k) * se_k
        het = cochran_q(y, var)
        _, _, lb, ub = hksj_pooled_mean(y, var, het.tau2)
        covered += lb <= 0.0 <= ub
    coverage = covered / reps
    assert 0.5 <= coverage <= 1.0, f"HKSJ coverage smoke {coverage:.1%} wildly off"


# 5. Both power solvers: predicted vs. empirical


def _ceil_arm_sizes(q: float, n_total: float, allocation: float) -> tuple[int, int]:
    """Mirrors ``increment.power.pairwise._segment_arm_sizes``'s per-segment
    integer ceiling exactly, so the closed-form power computed against it
    matches the solver's achieved power to floating-point precision rather
    than an approximate, MC-noisy estimate."""
    n_total_seg = q * n_total
    n_t = max(2, math.ceil(n_total_seg * allocation))
    n_c = max(2, math.ceil(n_total_seg * (1.0 - allocation)))
    return n_t, n_c


def test_pairwise_power_non_partition_shares():
    """A NON-PARTITION share pair (q_a=0.15, q_b=0.05, summing to 0.20,
    not 1) - the case that discriminates the correct 1/q_a+1/q_b formula
    from the old, wrong 1/(q(1-q)) one.

    The closed-form recomputation below (using the solver's own
    per-segment arm-size ceiling) is exact to floating point, but only a
    CONSISTENCY check: it re-derives from whatever n_total the solver
    returned, so it would pass even with a wrong sizing formula. The
    actual discriminator is the second assertion, which pins achieved
    power against the ORIGINAL target (0.80): a solver using the wrong
    formula sizes to n_total=50880 (vs. this solver's 46954) and achieves
    83.06% power, 3pp off target and outside the 1pp tolerance used here."""
    b = Baseline(mean=1.0, var=1.0)
    procedure = make_procedure(alpha=0.05)
    d = PowerDesign(power=0.8, allocation=0.5)
    r_a, r_b, q_a, q_b = 0.20, 0.05, 0.15, 0.05
    n_res = segment_pairwise_required_sample_size(
        r_a, r_b, q_a, q_b, b, procedure=procedure, design=d
    )

    theta = math.log1p(r_a) - math.log1p(r_b)
    n_a_t, n_a_c = _ceil_arm_sizes(q_a, n_res.n_total, d.allocation)
    n_b_t, n_b_c = _ceil_arm_sizes(q_b, n_res.n_total, d.allocation)
    se2 = b.effective_var / b.mean**2 * (1.0 / n_a_t + 1.0 / n_a_c + 1.0 / n_b_t + 1.0 / n_b_c)
    se_diff = math.sqrt(se2)
    z = norm.ppf(1.0 - procedure.compiled_tail_alpha)
    closed_form_power = norm.sf(z - theta / se_diff) + norm.cdf(-z - theta / se_diff)
    assert closed_form_power == pytest.approx(n_res.power, abs=1e-9)

    # Discriminator: achieved power must hit the target (0.80); the
    # consistency check above can't fail this way, only a wrong sizing formula can.
    assert n_res.power == pytest.approx(d.power, abs=0.01)


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPowerSolverCalibration:
    def test_joint_q_power_fixed_unequal_v(self):
        """Fixed-effects noncentral chi2 power at unequal per-segment v_k."""
        theta = np.array([0.20, 0.05, -0.05, 0.15, -0.10])
        var = np.array([0.02, 0.05, 0.01, 0.03, 0.04])
        predicted = joint_q_power_fixed(theta, var, alpha=0.05)

        rng = np.random.default_rng(9)
        k = len(theta)
        crit = chi2.ppf(0.95, k - 1)
        reps = 2000
        hits = 0
        for _ in range(reps):
            y = theta + rng.standard_normal(k) * np.sqrt(var)
            w = 1.0 / var
            ybar = (w * y).sum() / w.sum()
            q = (w * (y - ybar) ** 2).sum()
            hits += q > crit
        empirical = hits / reps
        assert abs(empirical - predicted) < 0.05, (
            f"predicted {predicted:.3f} vs empirical {empirical:.3f}"
        )


# 6. D2a scale check: through the real pipeline, not hand-fed moments


def _binomial_breakout_rows(
    baselines: list[float], effect: float, n_per_arm: int, seed: int
) -> list[dict]:
    """Synthetic group_summary rows for a constant ABSOLUTE effect over
    unequal per-segment baselines - Bernoulli sufficient stats generated
    directly (sum_y ~ Binomial(n, p), sum_y2 = sum_y since y in {0,1}), no
    per-unit simulation needed. No clamp on the treatment probability:
    this file's baselines top out at 0.40 and effect at 0.02, so it never
    nears 1.0.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, p in enumerate(baselines):
        seg = f"seg{i}"
        y_c = rng.binomial(1, p, n_per_arm)
        y_t = rng.binomial(1, p + effect, n_per_arm)
        for group_id, y in (("control", y_c), ("treatment", y_t)):
            rows.append(
                centered_row_from_raw_sums(
                    {
                        "experiment_id": "e1",
                        "metric": "conv",
                        "group_id": group_id,
                        "country": seg,
                        "n": n_per_arm,
                        "sum_y": float(y.sum()),
                        "sum_y2": float((y**2).sum()),
                        "successes": int(y.sum()),
                        "sum_x": None,
                        "sum_x2": None,
                        "sum_xy": None,
                        "sum_den": None,
                        "sum_den2": None,
                        "sum_yden": None,
                    }
                )
            )
    return rows


class TestD2aScaleCheck:
    """Must go through estimate_lift/run_breakout/segment_heterogeneity:
    the per-family absolute-scale derivation this guards lives in the
    pipeline, and hand-feeding moments to cochran_q would never exercise it.

    Measured (150 reps, n=2000/arm): relative rejects 49.3%, absolute
    rejects 5.3% - a constant absolute effect over unequal baselines reads
    as heterogeneity on the relative scale and (correctly) does not on the
    absolute scale.

    The two Monte Carlo tests below are parameter_recovery-marked
    (excluded from the fast suite); ``test_pipeline_shape_and_pooled_width_smoke``
    is not - it is what the default suite runs for this pipeline path.
    """

    _BASELINES = [0.05, 0.10, 0.20, 0.30, 0.40]
    _EFFECT = 0.02
    _METRIC = MeanMetric(name="conv", entity="user", fact="conv")

    def _rejection_rates(self, n_per_arm: int, reps: int, seed0: int) -> tuple[float, float]:
        rel_rej = abs_rej = 0
        for r in range(reps):
            rows = _binomial_breakout_rows(self._BASELINES, self._EFFECT, n_per_arm, seed0 + r)
            result = run_breakout(
                pd.DataFrame(rows), [self._METRIC], control_group="control", dimension="country"
            )
            summary, _ = segment_heterogeneity(result)
            by_scale = {s.scale: s for s in summary}
            rel_rej += by_scale["relative"].p_value < 0.05
            abs_rej += by_scale["absolute"].p_value < 0.05
        return rel_rej / reps, abs_rej / reps

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_relative_scale_rejects_at_moderate_n(self):
        rel_rate, abs_rate = self._rejection_rates(n_per_arm=2000, reps=150, seed0=1000)
        assert rel_rate > 0.30, (
            f"relative-scale rejection {rel_rate:.1%} should be well above nominal"
        )
        assert abs_rate < 0.15, (
            f"absolute-scale rejection {abs_rate:.1%} should stay near nominal 5%"
        )

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_relative_scale_saturates_at_larger_n(self):
        rel_rate, abs_rate = self._rejection_rates(n_per_arm=10_000, reps=100, seed0=2000)
        assert rel_rate > 0.90, f"relative-scale rejection {rel_rate:.1%} should be near-total"
        assert abs_rate < 0.15, (
            f"absolute-scale rejection {abs_rate:.1%} should stay near nominal 5%"
        )

    def test_pipeline_shape_and_pooled_width_smoke(self):
        """Fast, unmarked, single real pipeline call (no Monte Carlo loop).
        Not a calibration check: it asserts on structure the tests above
        never touch - absolute-scale ``shrink_k`` must not collapse toward
        0, and ``summary.pooled``'s interval width must not be
        degenerately zero on either scale.

        Measured: absolute-scale shrink_k in [0.167, 0.322]; pooled widths
        0.320 (relative) / 0.012 (absolute).
        """
        rows = _binomial_breakout_rows(
            TestD2aScaleCheck._BASELINES, TestD2aScaleCheck._EFFECT, 2000, 1000
        )
        result = run_breakout(
            pd.DataFrame(rows),
            [TestD2aScaleCheck._METRIC],
            control_group="control",
            dimension="country",
        )
        summary, segments = segment_heterogeneity(result)
        by_scale = {s.scale: s for s in summary}

        for scale, row in by_scale.items():
            assert row.pooled.lb is not None and row.pooled.ub is not None, (
                f"{scale}-scale pooled interval should always have lb/ub set"
            )
            width = row.pooled.ub - row.pooled.lb
            assert width > 1e-6, (
                f"{scale}-scale pooled interval is degenerately narrow ({width:.2e})"
            )

        abs_shrunken_k = [
            s.shrink_k for s in segments if s.scale == "absolute" and s.estimator == "shrunken"
        ]
        assert abs_shrunken_k, "expected shrunken absolute-scale segment rows"
        assert all(k is not None and k > 0.02 for k in abs_shrunken_k), (
            f"absolute-scale shrink_k collapsed toward 0: {abs_shrunken_k}"
        )


# 7. Absolute-scale Q calibration at thin cells vs. the relative scale


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_absolute_scale_q_better_calibrated_than_relative_at_thin_cells():
    """Measured (K=5, n=100/arm, p=0.10, 3000 reps, PAIRED - both scales
    computed from the SAME per-rep draws): relative 3.4% (conservative),
    absolute 5.3% (closer to nominal) - absolute-scale Q is better
    calibrated at thin cells than relative. Pairing removes the extra
    sqrt(2)-factor noise an independent comparison would carry, and
    halves the sampling work."""
    rng = np.random.default_rng(10)
    n, p, k, reps = 100, 0.10, 5, 3000

    rel_rejections = abs_rejections = estimable = 0
    for _ in range(reps):
        rel_ests: list[float] = []
        rel_vars: list[float] = []
        abs_ests: list[float] = []
        abs_vars: list[float] = []
        ok = True
        for _seg in range(k):
            y_c = rng.binomial(1, p, n)
            y_t = rng.binomial(1, p, n)  # same p - zero true effect on either scale
            p_c, p_t = y_c.mean(), y_t.mean()
            if p_c <= 0 or p_t <= 0:
                ok = False
                break
            se_c = math.sqrt((p_c * (1 - p_c)) / (n * p_c**2))
            se_t = math.sqrt((p_t * (1 - p_t)) / (n * p_t**2))
            rel_ests.append(math.log(p_t) - math.log(p_c))
            rel_vars.append(se_c**2 + se_t**2)
            abs_ests.append(p_t - p_c)
            abs_vars.append(p_c * (1 - p_c) / n + p_t * (1 - p_t) / n)
        if not ok or len(rel_ests) < 2:
            continue
        estimable += 1
        rel_rejections += cochran_q(np.array(rel_ests), np.array(rel_vars)).p_value < 0.05
        abs_rejections += cochran_q(np.array(abs_ests), np.array(abs_vars)).p_value < 0.05

    if estimable == 0:
        pytest.fail(
            "no estimable reps -- every rep hit a zero-count segment before either "
            "scale's Q could be computed; check p/n/k for this fixture"
        )
    rel_rate = rel_rejections / estimable
    abs_rate = abs_rejections / estimable
    assert rel_rate < 0.05, (
        f"relative-scale thin-cell rejection {rel_rate:.1%} should be conservative"
    )
    assert abs(abs_rate - 0.05) < abs(rel_rate - 0.05), (
        f"absolute ({abs_rate:.1%}) should be closer to nominal 5% than relative ({rel_rate:.1%})"
    )
