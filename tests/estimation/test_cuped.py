"""Steps 1 & 3d: CUPED invariance test (parameter recovery) + both-methods test.

* Step 1: With correlated pre/post data and zero true lift, CUPED recovers
  the unbiased estimate (same as unadjusted within MC tolerance) and
  ``Var(Y_cuped) ~= Var(Y) * (1 - corr(Y,X)^2)``.
* Step 3d: With a materialised covariate and both methods requested, the
  engine returns both labelled estimates; CUPED's CI is tighter; both
  recover the same unbiased point estimate under the null.
"""

from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import TypedDict

import numpy as np
import pandas as pd
import pytest

from increment.errors import InvalidRequestError
from increment.estimation.armstats import CENTERED_FIELDS, ArmStats
from increment.estimation.cuped import (
    CupedFit,
    _within_arm_theta,
    cuped_adjust,
    fit_cuped,
    fit_ratio_cuped,
    pooled_theta,
)
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.variance import ratio_abs_diff_se, ratio_log_mean_se
from increment.semantics.models import MeanMetric, Measure, RatioMetric
from tests.mc import nominal_band, replicate

# Helpers


def _make_arm_stats(
    n: int,
    mean: float,
    var: float,
    group_id: str,
    metric: str = "rev",
    sum_x: float | None = None,
    sum_x2: float | None = None,
    sum_xy: float | None = None,
) -> ArmStats:
    """Build an ArmStats from sufficient statistics.

    *y* moments are derived from (n, mean, var); *x* and *xy* moments are
    passed through as-is (or left ``None`` when not materialised).
    """
    sum_y = mean * n
    sum_y2 = var * (n - 1) + sum_y**2 / n
    return ArmStats.from_raw_sums(
        study_id="exp_cuped",
        metric=metric,
        group_id=group_id,
        n=n,
        sum_y=sum_y,
        sum_y2=sum_y2,
        sum_x=sum_x,
        sum_x2=sum_x2,
        sum_xy=sum_xy,
    )


def _mean_metric(name: str = "rev") -> MeanMetric:
    return MeanMetric(name=name, entity="user", fact=name)


def _summary_df(arms: list[ArmStats]) -> pd.DataFrame:
    """Convert list[ArmStats] to a group_summary DataFrame."""
    rows = [
        {
            "experiment_id": a.study_id,
            "metric": a.metric,
            "group_id": a.group_id,
            "n": a.n,
            **{field: getattr(a, field) for field in CENTERED_FIELDS},
        }
        for a in arms
    ]
    return pd.DataFrame(rows)


# Step 1: Invariance test (parameter_recovery mark)


def _correlated_arrays(
    rng: np.random.Generator,
    n_per_arm: int,
    rho: float = 0.7,
    true_lift: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(y_c, x_c, y_t, x_t)`` for two arms of correlated data.

    X ~ N(10, 2^2); Y = 5 + rho * scale_y * (X - 10)/scale_x + N(0, noise_sd)^2
    so corr(Y, X) ~= rho.  *true_lift* is added to the treatment Y.
    """
    scale_x = 2.0
    scale_y = 3.0
    noise_sd = scale_y * math.sqrt(1 - rho**2)

    # Common X (pre-exposure — same distribution for both arms)
    x_c = rng.normal(10, scale_x, n_per_arm)
    x_t = rng.normal(10, scale_x, n_per_arm)

    # Y = 5 + rho * scale_y * (X - 10)/scale_x + noise
    y_c = 5.0 + rho * scale_y * (x_c - 10.0) / scale_x + rng.normal(0, noise_sd, n_per_arm)
    y_t = (
        5.0
        + true_lift
        + rho * scale_y * (x_t - 10.0) / scale_x
        + rng.normal(0, noise_sd, n_per_arm)
    )
    return y_c, x_c, y_t, x_t


def _arm_from_arrays(y: np.ndarray, x: np.ndarray, group_id: str) -> ArmStats:
    """Format-1 raw sums over ``(y, x)``, adapted to the centered seam."""
    return ArmStats.from_raw_sums(
        study_id="exp_cuped",
        metric="rev",
        group_id=group_id,
        n=len(y),
        sum_y=float(y.sum()),
        sum_y2=float((y**2).sum()),
        sum_x=float(x.sum()),
        sum_x2=float((x**2).sum()),
        sum_xy=float((x * y).sum()),
    )


def _generate_correlated_data(
    rng: np.random.Generator,
    n_per_arm: int,
    rho: float = 0.7,
    true_lift: float = 0.0,
) -> tuple[ArmStats, ArmStats]:
    """Two arms of correlated (X, Y) data with the specified lift."""
    y_c, x_c, y_t, x_t = _correlated_arrays(rng, n_per_arm, rho, true_lift)
    return _arm_from_arrays(y_c, x_c, "control"), _arm_from_arrays(y_t, x_t, "treatment")


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestCupedInvariance:
    """CUPED point estimate is unbiased; variance is reduced per (1-rho^2)."""

    def test_cuped_unbiased_under_null(self):
        """CUPED recovers zero lift with correlated data — matches unadjusted."""
        rng = np.random.default_rng(42)
        control, treatment = _generate_correlated_data(rng, n_per_arm=10000, rho=0.7, true_lift=0.0)

        adjusted = cuped_adjust([control, treatment])
        c_adj, t_adj = adjusted[0], adjusted[1]

        # Point estimate should be near 0 (no lift)
        lift = t_adj.mean / c_adj.mean - 1.0
        assert lift == pytest.approx(0.0, abs=0.01), (
            f"CUPED lift estimate {lift:.4f} deviates from 0 beyond MC tolerance"
        )

    def test_cuped_variance_reduction(self):
        """Var(Y_cuped) ~= Var(Y) * (1 - rho^2) within 10% for each arm."""
        rng = np.random.default_rng(42)
        control, treatment = _generate_correlated_data(rng, n_per_arm=5000, rho=0.7, true_lift=0.0)

        adjusted = cuped_adjust([control, treatment])

        for orig_arm, adj_summary in [(control, adjusted[0]), (treatment, adjusted[1])]:
            var_y = orig_arm.var_y()
            var_cuped = adj_summary.var
            expected_var = var_y * (1 - 0.7**2)

            # Within 10% of the theoretical reduction
            assert var_cuped == pytest.approx(expected_var, rel=0.10), (
                f"Arm {orig_arm.group_id}: Var(Y_cuped)={var_cuped:.4f}, "
                f"expected ~{expected_var:.4f} (Var(Y)={var_y:.4f})"
            )


# Interval coverage over many draws, following test_cuped_late_recovery.py's
# _replicate pattern: the single-seed checks above pin the point estimate and
# Var(Y_cuped); this checks estimate_lift's CUPED CI reaches nominal coverage.

_COVERAGE_TRUE_LIFT_ADD = 0.5  # additive lift on the treatment arm's Y
_COVERAGE_BASELINE_MEAN = 5.0  # E[Y] under the DGP's control regime
_COVERAGE_TRUE_RELATIVE_LIFT = _COVERAGE_TRUE_LIFT_ADD / _COVERAGE_BASELINE_MEAN


def _cuped_coverage_trial(i: int, *, n_per_arm: int, rho: float, seed_base: int) -> bool:
    """One replicate: does the cuped-adjusted CI cover the true relative lift?"""
    rng = np.random.default_rng(seed_base + i)
    control, treatment = _generate_correlated_data(rng, n_per_arm, rho, _COVERAGE_TRUE_LIFT_ADD)
    df = _summary_df([control, treatment])
    (result,) = estimate_lift(
        metrics=[_mean_metric()],
        summary=df,
        control_group="control",
        methods=[Method(name="cuped", variance_reduction="cuped")],
    ).results
    lift = result.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return lift.lb <= _COVERAGE_TRUE_RELATIVE_LIFT <= lift.ub


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_cuped_interval_coverage_is_nominal():
    """CUPED's CI covers the true relative lift ~95% of the time over 1500
    independent draws. MCSE at 1500 reps is ~0.56pp; k=3 gives ~3x that
    headroom around the 95% target."""
    reps = 1500
    cov = replicate(
        reps,
        lambda i: _cuped_coverage_trial(i, n_per_arm=1000, rho=0.7, seed_base=20260821),
    )
    lo, hi = nominal_band(0.95, reps, k=3.0)
    assert lo <= cov.rate <= hi, cov.rate


def test_cuped_interval_coverage_smoke():
    """Fast, unmarked 40-rep small-N twin of the parameter_recovery check
    above. Bounds are nominal +/- k*mcse(nominal, reps) (see tests/mc.py)."""
    reps = 40
    cov = replicate(
        reps,
        lambda i: _cuped_coverage_trial(i, n_per_arm=200, rho=0.7, seed_base=99),
    )
    lo, _ = nominal_band(0.95, reps, k=3.0)
    assert cov.rate >= lo, cov.rate


# Fitted-theta calibration under unequal allocation and heterogeneous slopes,
# where an n-1 slope inflates the contrast variance. The DGP fixes the truth:
# additive lift _CAL_LIFT and relative lift _CAL_LIFT / _CAL_MEAN; per-arm slopes
# make theta a genuine compromise.

_CAL_MEAN = 20.0
_CAL_LIFT = 1.0
_Z975 = 1.959963984540054


class _Setting(TypedDict):
    n_c: int
    n_t: int
    slope_c: float
    slope_t: float


_CAL_SETTINGS: dict[str, _Setting] = {
    # The accepted witness as a sampling design: 1:9 allocation, slopes -1/+10.
    "adversarial": {"n_c": 100, "n_t": 900, "slope_c": -1.0, "slope_t": 10.0},
    # Unequal allocation with one shared slope: the ordinary CUPED premise.
    "unequal_homogeneous": {"n_c": 100, "n_t": 900, "slope_c": 0.7, "slope_t": 0.7},
    # Equal allocation with the heterogeneous slopes: the paired formula.
    "equal_heterogeneous": {"n_c": 500, "n_t": 500, "slope_c": -1.0, "slope_t": 10.0},
}


def _calibration_arms(
    rng: np.random.Generator, *, n_c: int, n_t: int, slope_c: float, slope_t: float
) -> tuple[ArmStats, ArmStats]:
    """X ~ N(10, 1) in both arms; Y = mean (+ lift) + slope*(X - 10) + N(0, 1).
    X is independent of assignment, so E[Y_t] - E[Y_c] = _CAL_LIFT exactly."""
    x_c = rng.normal(10.0, 1.0, n_c)
    y_c = _CAL_MEAN + slope_c * (x_c - 10.0) + rng.normal(0.0, 1.0, n_c)
    x_t = rng.normal(10.0, 1.0, n_t)
    y_t = _CAL_MEAN + _CAL_LIFT + slope_t * (x_t - 10.0) + rng.normal(0.0, 1.0, n_t)
    return _arm_from_arrays(y_c, x_c, "control"), _arm_from_arrays(y_t, x_t, "treatment")


def _lift_calibration(setting: str, *, reps: int, seed_base: int) -> dict[str, float]:
    """Replicate the public cuped lift and summarise it against the DGP truth."""
    spec = _CAL_SETTINGS[setting]
    abs_points, abs_ses, raw_points = [], [], []
    abs_hits = rel_hits = 0
    for i in range(reps):
        rng = np.random.default_rng(seed_base + i)
        control, treatment = _calibration_arms(rng, **spec)
        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
        ).results
        by_method = {r.method: r for r in results}
        cuped, raw = by_method["cuped"], by_method["unadjusted"]
        assert cuped.abs_diff is not None and cuped.abs_se is not None
        assert raw.abs_diff is not None
        cuped_lift = cuped.require_lift()
        assert cuped_lift.lb is not None and cuped_lift.ub is not None
        abs_points.append(cuped.abs_diff)
        abs_ses.append(cuped.abs_se)
        raw_points.append(raw.abs_diff)
        abs_hits += abs(cuped.abs_diff - _CAL_LIFT) <= _Z975 * cuped.abs_se
        rel_hits += cuped_lift.lb <= _CAL_LIFT / _CAL_MEAN <= cuped_lift.ub
    points = np.array(abs_points)
    return {
        "bias": float(points.mean() - _CAL_LIFT),
        "sd": float(points.std(ddof=1)),
        "mean_se": float(np.mean(abs_ses)),
        "abs_coverage": abs_hits / reps,
        "rel_coverage": rel_hits / reps,
        "sd_ratio_vs_unadjusted": float(points.std(ddof=1) / np.std(raw_points, ddof=1)),
    }


@pytest.mark.slow
@pytest.mark.parameter_recovery
@pytest.mark.parametrize("setting", sorted(_CAL_SETTINGS))
def test_fitted_theta_lift_is_calibrated_under_each_allocation(setting: str):
    """1500 replicates per setting. The fitted theta is a nuisance treated as
    fixed, so these are asymptotic claims checked at these sizes: the
    absolute contrast is unbiased inside its MC band, the reported SE tracks
    the realised SD inside the SD's own MC band (``3/sqrt(2*reps)`` = 5.5%
    at 1500 reps; the fitted-theta shortfall itself is about 1% at these
    designs, well inside that band), and both the absolute Wald interval and
    the log-relative interval (pooled-anchor SE) cover at nominal within
    3 x MCSE. No universal finite-sample variance-reduction claim is made:
    the adversarial setting checks only that the contrast is not INFLATED
    the way the n-1 slope inflated it (realised SD ratio 2.72 on this DGP
    and seed set, against 1.00 here)."""
    reps = 1500
    stats = _lift_calibration(setting, reps=reps, seed_base=20260915)
    assert abs(stats["bias"]) <= 3.0 * stats["sd"] / math.sqrt(reps), stats
    sd_band = 3.0 / math.sqrt(2.0 * reps)
    assert 1.0 - sd_band <= stats["mean_se"] / stats["sd"] <= 1.0 + sd_band, stats
    lo, hi = nominal_band(0.95, reps, k=3.0)
    assert lo <= stats["abs_coverage"] <= hi, stats
    assert lo <= stats["rel_coverage"] <= hi, stats
    assert stats["sd_ratio_vs_unadjusted"] < 1.05, stats


def test_fitted_theta_lift_calibration_smoke():
    """Fast, unmarked 40-rep twin of the adversarial cell above."""
    reps = 40
    stats = _lift_calibration("adversarial", reps=reps, seed_base=7)
    lo, _ = nominal_band(0.95, reps, k=3.0)
    assert stats["abs_coverage"] >= lo, stats
    assert stats["rel_coverage"] >= lo, stats
    assert stats["sd_ratio_vs_unadjusted"] < 1.3, stats


def test_smoke_fast():
    """Fast smoke test: CUPED runs without error and produces sensible output."""
    rng = np.random.default_rng(0)
    control, treatment = _generate_correlated_data(rng, n_per_arm=200, rho=0.5, true_lift=0.0)

    adjusted = cuped_adjust([control, treatment])
    assert len(adjusted) == 2
    # Adjusted variance should be smaller than raw
    assert adjusted[0].var < control.var_y()
    assert adjusted[1].var < treatment.var_y()
    # Mean should be close to raw (no lift under the null)
    assert adjusted[0].mean == pytest.approx(control.mean_y(), abs=0.5)
    assert adjusted[1].mean == pytest.approx(treatment.mean_y(), abs=0.5)


# Step 1: pooled-theta correctness


class TestCupedPooledTheta:
    """Pooled (not per-arm) theta is required for unbiasedness."""

    def test_pooled_theta_across_arms(self):
        """cuped_adjust uses a SINGLE theta shared across arms: the
        WITHIN-arm form -- each arm's own centered covariance and
        covariate variance, weighted by that arm's inverse count (see
        TestContrastOptimalTheta for why inverse n), NOT a single mean
        pooled jointly across arms (see the module docstring for why the
        joint form biases theta)."""
        rng = np.random.default_rng(123)
        y_c, x_c, y_t, x_t = _correlated_arrays(rng, n_per_arm=2000, rho=0.6, true_lift=0.0)
        control = _arm_from_arrays(y_c, x_c, "control")
        treatment = _arm_from_arrays(y_t, x_t, "treatment")
        assert control.ref_x is not None and control.cx2 is not None and control.cxy is not None
        assert (
            treatment.ref_x is not None and treatment.cx2 is not None and treatment.cxy is not None
        )

        expected_theta = (control.cov_yx() / control.n + treatment.cov_yx() / treatment.n) / (
            control.var_x() / control.n + treatment.var_x() / treatment.n
        )

        # Verify adjustment uses this theta by comparing adjusted means
        # to the formula mean(Y) - theta * (mean(X) - mean(X)_pooled)
        adjusted = cuped_adjust([control, treatment])
        N = control.n + treatment.n
        mean_x_pooled = (control.n * control.mean_x() + treatment.n * treatment.mean_x()) / N

        c_mean_adj = control.mean_y() - expected_theta * (control.mean_x() - mean_x_pooled)
        t_mean_adj = treatment.mean_y() - expected_theta * (treatment.mean_x() - mean_x_pooled)

        assert adjusted[0].mean == pytest.approx(c_mean_adj, rel=1e-9)
        assert adjusted[1].mean == pytest.approx(t_mean_adj, rel=1e-9)

    def test_full_quadratic_form_not_1_minus_rho_squared_shortcut(self):
        """The adjusted variance uses the FULL quadratic form
        Var(Y) - 2*theta*Cov(Y,X) + theta**2*Var(X) with the WITHIN-arm
        pooled theta, NOT the (1-rho**2) shortcut, which is only exact
        when theta is each arm's own optimal (per-arm) theta. Construct
        two arms with deliberately different Var(X)/Cov(Y,X) so the
        pooled theta is far from the control arm's own-optimal theta,
        making the two formulas diverge sharply - this fails loudly if
        the implementation ever regresses to the shortcut.
        """
        # Control: var_y=20, var_x=4, cov_yx=8 (own-optimal theta=2, rho^2=0.8)
        control = ArmStats.from_raw_sums(
            study_id="exp",
            metric="m",
            group_id="control",
            n=10,
            sum_y=100.0,
            sum_y2=1180.0,
            sum_x=50.0,
            sum_x2=286.0,
            sum_xy=572.0,
        )
        # Treatment: var_y=20, var_x=50, cov_yx=2 (very different covariate scale/correlation)
        treatment = ArmStats.from_raw_sums(
            study_id="exp",
            metric="m",
            group_id="treatment",
            n=10,
            sum_y=120.0,
            sum_y2=1620.0,
            sum_x=60.0,
            sum_x2=810.0,
            sum_xy=738.0,
        )

        adjusted = cuped_adjust([control, treatment])

        # WITHIN-arm theta: control cx2=36, cxy=72; treatment cx2=450, cxy=18
        # -> theta = (72+18)/(36+450) = 90/486
        theta = 90.0 / 486.0

        # Control's own ddof=1 moments: var_y=20, var_x=4, cov_yx=8
        expected_var_quadratic = 20.0 - 2.0 * theta * 8.0 + theta**2 * 4.0
        # The WRONG shortcut using control's OWN correlation (rho^2 = 8^2/(20*4) = 0.8):
        wrong_shortcut_var = 20.0 * (1.0 - 0.8)

        assert adjusted[0].var == pytest.approx(expected_var_quadratic, rel=1e-9)
        # The two formulas must diverge sharply here - confirms the
        # implementation is NOT silently using the (1-rho**2) shortcut.
        assert abs(adjusted[0].var - wrong_shortcut_var) > 5.0, (
            f"adjusted variance {adjusted[0].var:.4f} suspiciously close to the "
            f"(1-rho**2) shortcut value {wrong_shortcut_var:.4f} -- check for a "
            "regression to the shortcut formula"
        )


# The additive contrast's own precision: inverse-n theta


def _slope_arm(
    n: int,
    beta: float,
    group_id: str,
    *,
    noise_var: float = 0.0,
    mean_y: float = 0.0,
    metric: str = "m",
) -> ArmStats:
    """An arm with ``mean_x = 0``, ``var_x = 1``, ``cov_yx = beta`` and
    ``var_y = beta**2 + noise_var`` in ddof=1 moments: Y = mean_y + beta * X (+ noise)."""
    return ArmStats.from_raw_sums(
        study_id="s",
        metric=metric,
        group_id=group_id,
        n=n,
        sum_x=0.0,
        sum_x2=float(n - 1),
        sum_y=mean_y * n,
        sum_y2=(beta * beta + noise_var) * (n - 1) + mean_y * mean_y * n,
        sum_xy=beta * (n - 1),
    )


def _exact_contrast_variance_ratio(arms: list[ArmStats], theta: Fraction) -> Fraction:
    """``Var(A - theta*B) / Var(A)`` for the additive contrast over *arms*,
    in exact rationals from each arm's own ddof=1 moments: the objective
    the CUPED slope is supposed to minimise."""
    adjusted = sum(
        (
            (
                Fraction(a.var_y())
                - 2 * theta * Fraction(a.cov_yx())
                + theta**2 * Fraction(a.var_x())
            )
            / a.n
            for a in arms
        ),
        Fraction(0),
    )
    unadjusted = sum((Fraction(a.var_y()) / a.n for a in arms), Fraction(0))
    return adjusted / unadjusted


def _exact_inverse_n_theta(arms: list[ArmStats]) -> Fraction:
    weighted_cov = sum((Fraction(a.cov_yx()) / a.n for a in arms), Fraction(0))
    weighted_var_x = sum((Fraction(a.var_x()) / a.n for a in arms), Fraction(0))
    return weighted_cov / weighted_var_x


def _production_variance_ratio(arms: list[ArmStats]) -> float:
    adjusted = cuped_adjust(arms)
    return sum(s.var / s.n for s in adjusted) / sum(a.var_y() / a.n for a in arms)


class TestContrastOptimalTheta:
    """theta minimises ``Var(A - theta*B)`` for ``A = mean(Y_t) - mean(Y_c)``,
    ``B = mean(X_t) - mean(X_c)``: with independent arms that variance is
    ``sum_a [Var(Y_a) - 2*theta*Cov(Y_a,X_a) + theta**2*Var(X_a)] / n_a``,
    whose minimiser weights each arm's centered moments by ``1/n_a``. The
    n-1 (pooled-regression) weights estimate a different quantity and can
    make the adjusted contrast WORSE than no adjustment."""

    def test_unequal_allocation_heterogeneous_slopes_witness(self):
        """The accepted witness: n_t=900 at slope +10 against n_c=100 at slope
        -1. The n-1 pooled slope (899*10 - 99) / 998 = 8.9088 inflates the
        contrast variance 8.1179x; the contrast-optimal slope is exactly
        (10/900 - 1/100) / (1/900 + 1/100) = 1/10 and the ratio 1089/1090."""
        treatment = _slope_arm(900, 10.0, "treatment")
        control = _slope_arm(100, -1.0, "control")
        arms = [treatment, control]

        n_minus_one_theta = Fraction(899 * 10 - 99, 998)
        assert float(_exact_contrast_variance_ratio(arms, n_minus_one_theta)) == pytest.approx(
            8.117914507472781
        )

        theta, _ = pooled_theta(arms)
        assert theta == pytest.approx(0.1, abs=1e-15)
        ratio = _production_variance_ratio(arms)
        assert ratio == pytest.approx(1089 / 1090, rel=1e-12)
        assert ratio < 1.0

    def test_reconstructed_auxiliary_unequal_allocation_witness(self):
        """A second unequal-allocation witness is specified only by
        its two ratios: n-1 weights ~2.4868, contrast-optimal weights
        0.5970835450918622. Its inputs were never recorded anywhere (repo,
        history, issue tracker), so this eight-observation fixture is
        RECONSTRUCTED from those target decimals, not retained history:
        control X = (-a, a), Y = X; treatment X = (-a, a, 0, 0, 0, 0),
        Y = b*X + (0, 0, -d, d, 0, 0), with ``a = sqrt(5/2)``, ``d = sqrt(5e/2)``,
        ``q = 1 + sqrt((o - r)/(1 - r))``, ``b = (15q - 8)/(8 - q)`` and
        ``e = (15 + b)^2 / (16(1 - r)) - 15 - b^2``; ``e`` makes
        ``C^2/(W D) = 1 - r`` and ``b`` makes the n-1 slope ``q`` times the
        contrast-optimal one, so completing the square gives the old ratio
        ``r + (1 - r)(q - 1)^2 = o``. Both ratios are recomputed here from the
        raw arrays, never read back from production output."""
        r, o = 0.5970835450918622, 2.4868
        q = 1.0 + math.sqrt((o - r) / (1.0 - r))
        b = (15.0 * q - 8.0) / (8.0 - q)
        e = (15.0 + b) ** 2 / (16.0 * (1.0 - r)) - 15.0 - b * b
        a, d = math.sqrt(2.5), math.sqrt(2.5 * e)
        x_c, y_c = [-a, a], [-a, a]
        x_t = [-a, a, 0.0, 0.0, 0.0, 0.0]
        y_t = [b * x + eps for x, eps in zip(x_t, [0.0, 0.0, -d, d, 0.0, 0.0], strict=True)]

        def raw(xs, ys, group_id):
            return ArmStats.from_raw_sums(
                study_id="s",
                metric="m",
                group_id=group_id,
                n=len(xs),
                sum_x=sum(xs),
                sum_x2=sum(v * v for v in xs),
                sum_y=sum(ys),
                sum_y2=sum(v * v for v in ys),
                sum_xy=sum(u * v for u, v in zip(xs, ys, strict=True)),
            )

        control, treatment = raw(x_c, y_c, "control"), raw(x_t, y_t, "treatment")
        arms = [control, treatment]

        def var(v):
            m = sum(v) / len(v)
            return sum((u - m) ** 2 for u in v) / (len(v) - 1)

        def cov(u, v):
            mu, mv = sum(u) / len(u), sum(v) / len(v)
            return sum((p - mu) * (s - mv) for p, s in zip(u, v, strict=True)) / (len(u) - 1)

        n_c, n_t = len(x_c), len(x_t)
        d_total = var(y_c) / n_c + var(y_t) / n_t
        c_total = cov(y_c, x_c) / n_c + cov(y_t, x_t) / n_t
        w_total = var(x_c) / n_c + var(x_t) / n_t
        theta_old = ((n_c - 1) * cov(y_c, x_c) + (n_t - 1) * cov(y_t, x_t)) / (
            (n_c - 1) * var(x_c) + (n_t - 1) * var(x_t)
        )

        def ratio(theta):
            return (d_total - 2 * theta * c_total + theta * theta * w_total) / d_total

        assert ratio(theta_old) == pytest.approx(o, abs=2e-14)
        assert ratio(c_total / w_total) == pytest.approx(r, abs=2e-14)

        theta, _ = pooled_theta(arms)
        assert theta == pytest.approx(c_total / w_total, rel=1e-14)
        assert _production_variance_ratio(arms) == pytest.approx(r, abs=2e-14)

    def test_theta_is_the_exact_inverse_n_moment_ratio(self):
        """On a noisy, unequal-allocation, heterogeneous-slope pair theta equals
        ``sum(cov/n) / sum(var_x/n)`` in exact rationals, and NOT the n-1 form."""
        arms = [
            _slope_arm(2500, 1.5, "control", noise_var=4.0),
            _slope_arm(400, -0.5, "treatment", noise_var=1.0),
        ]
        theta, _ = pooled_theta(arms)
        assert theta == pytest.approx(float(_exact_inverse_n_theta(arms)), rel=1e-14)
        n_minus_one = sum((a.n - 1) * a.cov_yx() for a in arms) / sum(
            (a.n - 1) * a.var_x() for a in arms
        )
        assert abs(theta - n_minus_one) > 0.1

    def test_theta_minimises_the_additive_contrast_variance(self):
        """Any perturbation of theta increases ``Var(A - theta*B)`` on every
        fixture, including one where arms differ in covariate scale."""
        rng = np.random.default_rng(3)
        fixtures = [
            [_slope_arm(900, 10.0, "t"), _slope_arm(100, -1.0, "c")],
            [_slope_arm(150, 2.0, "t", noise_var=3.0), _slope_arm(850, 0.5, "c", noise_var=9.0)],
            [
                _arm_from_arrays(*_scaled_draw(rng, 700, y_sd=3.0, x_sd=0.5, rho=0.8), "t"),
                _arm_from_arrays(*_scaled_draw(rng, 120, y_sd=2.0, x_sd=4.0, rho=-0.4), "c"),
            ],
        ]
        for arms in fixtures:
            theta = Fraction(pooled_theta(arms)[0])
            at_theta = _exact_contrast_variance_ratio(arms, theta)
            for step in (Fraction(1, 1000), Fraction(-1, 1000), Fraction(1, 2), Fraction(-3)):
                assert _exact_contrast_variance_ratio(arms, theta + step) > at_theta

    def test_equal_allocation_preserves_the_paired_formula(self):
        """With ``n_t == n_c`` the inverse-n weights cancel: theta is the sum of
        centered cross products over the sum of centered squares."""
        rng = np.random.default_rng(11)
        y_c, x_c, y_t, x_t = _correlated_arrays(rng, n_per_arm=500, rho=0.6, true_lift=0.3)
        control = _arm_from_arrays(y_c, x_c, "control")
        treatment = _arm_from_arrays(y_t, x_t, "treatment")
        theta, _ = pooled_theta([control, treatment])
        paired = (control.cov_yx() + treatment.cov_yx()) / (control.var_x() + treatment.var_x())
        assert theta == pytest.approx(paired, rel=1e-13)

    def test_equal_allocation_opposing_slopes_cancel_to_no_adjustment(self):
        """Equal counts, equal covariate spread, slopes +b and -b: no single
        slope helps the contrast, so theta is exactly 0 and the adjusted
        moments are the raw ones."""
        arms = [_slope_arm(400, 2.0, "control", noise_var=1.0), _slope_arm(400, -2.0, "t")]
        theta, _ = pooled_theta(arms)
        assert theta == 0.0
        for arm, summary in zip(arms, cuped_adjust(arms), strict=True):
            assert summary.mean == arm.mean_y()
            assert summary.var == arm.var_y()

    def test_permutation_invariance(self):
        """theta, the anchor and each arm's adjusted summary do not depend on
        the order arms are supplied in."""
        arms = [
            _slope_arm(900, 10.0, "a"),
            _slope_arm(100, -1.0, "b", noise_var=2.0),
            _slope_arm(250, 3.0, "c", noise_var=1.0),
        ]
        reference = pooled_theta(arms)
        by_arm = dict(zip((a.group_id for a in arms), cuped_adjust(arms), strict=True))
        for order in ((1, 0, 2), (2, 1, 0), (1, 2, 0)):
            permuted = [arms[i] for i in order]
            assert pooled_theta(permuted) == reference
            for arm, summary in zip(permuted, cuped_adjust(permuted), strict=True):
                assert summary == by_arm[arm.group_id]

    def test_partition_invariance(self):
        """An arm delivered as day-slice partitions and reduced through
        ``ArmStats.combine`` adjusts identically to the same arm delivered whole."""
        rng = np.random.default_rng(5)
        y_c, x_c, y_t, x_t = _correlated_arrays(rng, n_per_arm=900, rho=0.7, true_lift=0.4)
        control = _arm_from_arrays(y_c[:150], x_c[:150], "control")
        whole = _arm_from_arrays(y_t, x_t, "treatment")
        cuts = (0, 200, 500, 900)
        slices = [
            _arm_from_arrays(y_t[lo:hi], x_t[lo:hi], "treatment")
            for lo, hi in itertools.pairwise(cuts)
        ]
        recombined = ArmStats.combine(slices)

        theta_whole, anchor_whole = pooled_theta([control, whole])
        theta_parts, anchor_parts = pooled_theta([control, recombined])
        assert theta_parts == pytest.approx(theta_whole, rel=1e-12)
        assert anchor_parts == pytest.approx(anchor_whole, rel=1e-12)
        for a, b in zip(
            cuped_adjust([control, whole]), cuped_adjust([control, recombined]), strict=True
        ):
            assert b.mean == pytest.approx(a.mean, rel=1e-12)
            assert b.var == pytest.approx(a.var, rel=1e-12)

    def test_multi_arm_family_minimises_the_sum_of_pairwise_contrast_variances(self):
        """For K arms the same inverse-n theta minimises the SUM over every
        pair of that pair's additive contrast variance (each arm enters K-1
        pairs, so the objective is (K-1) times the per-arm sum); it does not
        promise to optimise every heterogeneous pair on its own."""
        arms = [
            _slope_arm(900, 10.0, "control"),
            _slope_arm(100, -1.0, "t1", noise_var=1.0),
            _slope_arm(300, 4.0, "t2", noise_var=2.0),
        ]
        theta = Fraction(pooled_theta(arms)[0])
        assert theta == pytest.approx(float(_exact_inverse_n_theta(arms)), rel=1e-14)

        def pairwise_sum(t: Fraction) -> Fraction:
            total = Fraction(0)
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    total += _exact_contrast_variance_ratio([arms[i], arms[j]], t) * sum(
                        Fraction(a.var_y()) / a.n for a in (arms[i], arms[j])
                    )
            return total

        at_theta = pairwise_sum(theta)
        for step in (Fraction(1, 100), Fraction(-1, 100), Fraction(2)):
            assert pairwise_sum(theta + step) > at_theta
        # The family theta is not the (control, t1) pair's own optimum.
        pair_theta = _exact_inverse_n_theta(arms[:2])
        assert _exact_contrast_variance_ratio(
            arms[:2], pair_theta
        ) < _exact_contrast_variance_ratio(arms[:2], theta)


def _scaled_draw(
    rng: np.random.Generator, n: int, *, y_sd: float, x_sd: float, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(y, x)`` with the requested scales and correlation."""
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    x = 10.0 + x_sd * z1
    y = 5.0 + y_sd * (rho * z1 + math.sqrt(1.0 - rho * rho) * z2)
    return y, x


class TestCupedFit:
    """``fit_cuped`` computes theta and the pooled anchor once; the wrappers
    and the per-arm projections all read that one state."""

    @staticmethod
    def _pair() -> tuple[ArmStats, ArmStats]:
        rng = np.random.default_rng(21)
        y_c, x_c = _scaled_draw(rng, 120, y_sd=4.0, x_sd=2.0, rho=0.7)
        y_t, x_t = _scaled_draw(rng, 880, y_sd=3.0, x_sd=2.5, rho=0.5)
        return _arm_from_arrays(y_c, x_c, "control"), _arm_from_arrays(y_t + 1.5, x_t, "treatment")

    def test_wrappers_read_the_fit(self):
        control, treatment = self._pair()
        fit = fit_cuped([control, treatment])
        assert isinstance(fit, CupedFit)
        assert fit.arms == (control, treatment)
        assert pooled_theta([control, treatment]) == (fit.theta, fit.mean_x_pooled)
        assert cuped_adjust([control, treatment]) == fit.adjust()

    def test_adjusted_mean_centres_on_the_pooled_anchor(self):
        control, treatment = self._pair()
        fit = fit_cuped([control, treatment])
        n = control.n + treatment.n
        anchor = (control.n * control.mean_x() + treatment.n * treatment.mean_x()) / n
        assert fit.mean_x_pooled == pytest.approx(anchor, rel=1e-14)
        for arm in (control, treatment):
            expected = arm.mean_y() - fit.theta * (arm.mean_x() - fit.mean_x_pooled)
            assert fit.adjusted_mean(arm) == pytest.approx(expected, rel=1e-14)

    def test_residual_var_is_the_centered_quadratic_form(self):
        control, treatment = self._pair()
        fit = fit_cuped([control, treatment])
        for arm, summary in zip((control, treatment), fit.adjust(), strict=True):
            assert fit.residual_var(arm, fit.theta) == summary.var
            slope = 0.37
            expected = arm.var_y() - 2 * slope * arm.cov_yx() + slope * slope * arm.var_x()
            assert fit.residual_var(arm, slope) == pytest.approx(expected, rel=1e-14)

    def test_absolute_projection_slopes_are_theta(self):
        """``j_t=1, j_c=-1`` gives ``k=1``: both arms' scores are ``Y - theta*X``."""
        control, treatment = self._pair()
        fit = fit_cuped([control, treatment])
        assert fit.contrast_slopes(treatment, control, 1.0, -1.0) == (fit.theta, fit.theta)

    def test_log_projection_slopes_follow_the_pooled_anchor_jacobian(self):
        """With ``j_t=1/mu_t, j_c=-1/mu_c`` and ``k = j_t*w_c - j_c*w_t`` the arm
        scores are ``j_t*(Y_t - theta*k*mu_t*X_t)`` and ``j_c*(Y_c - theta*k*mu_c*X_c)``."""
        control, treatment = self._pair()
        fit = fit_cuped([control, treatment])
        mu_t, mu_c = fit.adjusted_mean(treatment), fit.adjusted_mean(control)
        w_t = treatment.n / (treatment.n + control.n)
        w_c = 1.0 - w_t
        j_t, j_c = 1.0 / mu_t, -1.0 / mu_c
        k = j_t * w_c - j_c * w_t
        slope_t, slope_c = fit.contrast_slopes(treatment, control, j_t, j_c)
        assert slope_t == pytest.approx(fit.theta * k * mu_t, rel=1e-14)
        assert slope_c == pytest.approx(fit.theta * k * mu_c, rel=1e-14)
        assert slope_t != pytest.approx(fit.theta) and slope_c != pytest.approx(fit.theta)

    def test_projection_slopes_reduce_to_theta_when_means_agree(self):
        """Equal adjusted means make the log Jacobian proportional to the
        absolute one, so the anchor's cross term vanishes and both slopes are theta."""
        arms = [
            _slope_arm(700, 1.2, "control", noise_var=1.0, mean_y=20.0),
            _slope_arm(300, 0.4, "t", noise_var=2.0, mean_y=20.0),
        ]
        fit = fit_cuped(arms)
        mu = fit.adjusted_mean(arms[0])
        assert mu == fit.adjusted_mean(arms[1])
        slope_t, slope_c = fit.contrast_slopes(arms[1], arms[0], 1.0 / mu, -1.0 / mu)
        assert slope_t == pytest.approx(fit.theta, rel=1e-14)
        assert slope_c == pytest.approx(fit.theta, rel=1e-14)


# Public projections carry the shared pooled anchor's covariance


def _moment_arm(
    n: int, mean_y: float, mean_x: float, var_y: float, var_x: float, cov: float, group_id: str
) -> ArmStats:
    """An arm with exactly these ddof=1 moments."""
    sum_y, sum_x = mean_y * n, mean_x * n
    return ArmStats.from_raw_sums(
        study_id="s",
        metric="rev",
        group_id=group_id,
        n=n,
        sum_y=sum_y,
        sum_y2=var_y * (n - 1) + sum_y * sum_y / n,
        sum_x=sum_x,
        sum_x2=var_x * (n - 1) + sum_x * sum_x / n,
        sum_xy=cov * (n - 1) + sum_x * sum_y / n,
    )


def _independent_log_lift_variance(treatment: ArmStats, control: ArmStats) -> tuple[float, float]:
    """``(V_log, V_abs)`` for the CUPED lift by an explicit delta method over
    ``(mean_y_t, mean_x_t, mean_y_c, mean_x_c)`` with a block-diagonal
    covariance: the pooled anchor ``w_t*mean_x_t + w_c*mean_x_c`` is
    differentiated like any other function of the arm means, so no formula
    from the estimator is reused."""
    theta, anchor = pooled_theta([control, treatment])
    n_t, n_c = treatment.n, control.n
    w_t = n_t / (n_t + n_c)
    w_c = 1.0 - w_t
    mu_t = treatment.mean_y() - theta * (treatment.mean_x() - anchor)
    mu_c = control.mean_y() - theta * (control.mean_x() - anchor)
    # d mu_t / d(mean_x_t) = -theta*(1-w_t); d mu_t / d(mean_x_c) = theta*w_c
    # d mu_c / d(mean_x_t) = theta*w_t;      d mu_c / d(mean_x_c) = -theta*(1-w_c)
    d_mu = np.array(
        [
            [1.0, -theta * (1.0 - w_t), 0.0, theta * w_c],
            [0.0, theta * w_t, 1.0, -theta * (1.0 - w_c)],
        ]
    )
    sigma = np.zeros((4, 4))
    sigma[:2, :2] = (
        np.array([[treatment.var_y(), treatment.cov_yx()], [treatment.cov_yx(), treatment.var_x()]])
        / n_t
    )
    sigma[2:, 2:] = (
        np.array([[control.var_y(), control.cov_yx()], [control.cov_yx(), control.var_x()]]) / n_c
    )
    grad_log = np.array([1.0 / mu_t, -1.0 / mu_c]) @ d_mu
    grad_abs = np.array([1.0, -1.0]) @ d_mu
    return float(grad_log @ sigma @ grad_log), float(grad_abs @ sigma @ grad_abs)


class TestPooledAnchorProjections:
    """Unequal positive means, unequal allocation: the log-relative lift's SE
    must carry the shared random anchor's covariance (its k-Jacobian),
    while the absolute contrast keeps the ordinary adjusted-contrast variance
    because the anchor cancels there."""

    @staticmethod
    def _fixture() -> tuple[ArmStats, ArmStats]:
        treatment = _moment_arm(900, 12.0, 5.5, 9.0, 4.0, 4.8, "treatment")
        control = _moment_arm(100, 10.0, 5.0, 16.0, 4.0, 6.0, "control")
        return treatment, control

    def _result(self):
        treatment, control = self._fixture()
        (result,) = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        ).results
        return treatment, control, result

    def test_log_relative_se_is_the_pooled_anchor_jacobian(self):
        treatment, control, result = self._result()
        v_log, _ = _independent_log_lift_variance(treatment, control)
        assert result.require_lift().log_se is not None
        assert result.require_lift().log_se == pytest.approx(math.sqrt(v_log), rel=1e-12)
        # The sum of two marginal adjusted-mean log variances is a different
        # (wrong) number here: it treats the shared anchor as fixed.
        c_adj, t_adj = cuped_adjust([control, treatment])
        marginal = t_adj.var / (t_adj.n * t_adj.mean**2) + c_adj.var / (c_adj.n * c_adj.mean**2)
        assert abs(marginal / v_log - 1.0) > 1e-3
        assert result.require_lift().log_se != pytest.approx(math.sqrt(marginal), rel=1e-6)

    def test_absolute_se_is_the_adjusted_contrast_variance(self):
        treatment, control, result = self._result()
        _, v_abs = _independent_log_lift_variance(treatment, control)
        c_adj, t_adj = cuped_adjust([control, treatment])
        assert result.abs_se is not None
        assert result.abs_se == pytest.approx(math.sqrt(v_abs), rel=1e-12)
        assert result.abs_se == pytest.approx(math.sqrt(t_adj.var / t_adj.n + c_adj.var / c_adj.n))
        assert result.abs_diff == t_adj.mean - c_adj.mean

    def test_point_estimates_stay_on_their_own_scales(self):
        treatment, control, result = self._result()
        c_adj, t_adj = cuped_adjust([control, treatment])
        assert result.require_lift().value == pytest.approx(t_adj.mean / c_adj.mean - 1.0, rel=1e-9)
        assert result.require_lift().log_mean == pytest.approx(
            math.log(t_adj.mean) - math.log(c_adj.mean)
        )

    def test_log_se_equals_marginal_sum_only_under_equal_adjusted_means(self):
        """When ``mu_t == mu_c`` the anchor cross term vanishes, so the two
        constructions coincide: the pooled-anchor form is a strict superset."""
        shifted = [
            _slope_arm(700, 1.2, "control", noise_var=1.0, mean_y=20.0, metric="rev"),
            _slope_arm(300, 0.4, "t", noise_var=2.0, mean_y=20.0, metric="rev"),
        ]
        (result,) = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df(shifted),
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        ).results
        c_adj, t_adj = cuped_adjust(shifted)
        assert t_adj.mean == c_adj.mean
        marginal = t_adj.var / (t_adj.n * t_adj.mean**2) + c_adj.var / (c_adj.n * c_adj.mean**2)
        assert result.require_lift().log_se == pytest.approx(math.sqrt(marginal), rel=1e-12)


class TestLargeOffsetSmallDifference:
    """Centered (format-2) moments at a large common offset, unequal
    allocation: theta and the per-arm adjusted variances keep exact-rational
    grade, and a representable small treatment difference survives
    adjustment."""

    @staticmethod
    def _centered_arm(y: np.ndarray, x: np.ndarray, group_id: str) -> ArmStats:
        ref_y, ref_x = float(y.mean()), float(x.mean())
        return ArmStats(
            study_id="s",
            metric="rev",
            group_id=group_id,
            n=len(y),
            ref_y=ref_y,
            cy1=float((y - ref_y).sum()),
            cy2=float(((y - ref_y) ** 2).sum()),
            ref_x=ref_x,
            cx1=float((x - ref_x).sum()),
            cx2=float(((x - ref_x) ** 2).sum()),
            cxy=float(((y - ref_y) * (x - ref_x)).sum()),
            x_role="covariate",
        )

    @pytest.mark.parametrize("offset", [0.0, 1e6, 1e9])
    def test_theta_and_contrast_match_exact_rationals(self, offset: float):
        """The treatment arm is five copies of the control arm's units shifted
        by ``difference`` (5:1 allocation, identical covariate means), so the
        covariate contrast ``B`` is zero by construction and the adjusted
        contrast is the shift whatever theta is. At offset 1e9 the 1e-3 shift
        is ~8389 ulps and grid-aligned (every shifted value rounds by the
        same ulp count), which is why the contrast equals it exactly; a
        non-aligned difference between 1e9-magnitude means is bounded by the
        ~1.2e-7 float grid regardless of estimator. The load-bearing
        assertions are theta and the per-arm variances against exact rationals,
        far inside the band where raw-sum moments would be pure noise."""
        rng = np.random.default_rng(17)
        difference = 1e-3
        y_c, x_c = _scaled_draw(rng, 150, y_sd=1.0, x_sd=1.0, rho=0.8)
        y_c, x_c = y_c + offset, x_c + offset
        y_t, x_t = np.tile(y_c, 5) + difference, np.tile(x_c, 5)
        control = self._centered_arm(y_c, x_c, "control")
        treatment = self._centered_arm(y_t, x_t, "treatment")

        def exact_var(a):
            fa = [Fraction(v) for v in a]
            m = sum(fa) / len(fa)
            return sum((v - m) ** 2 for v in fa) / (len(fa) - 1)

        def exact_cov(a, b):
            fa, fb = [Fraction(v) for v in a], [Fraction(v) for v in b]
            ma, mb = sum(fa) / len(fa), sum(fb) / len(fb)
            return sum((u - ma) * (v - mb) for u, v in zip(fa, fb, strict=True)) / (len(fa) - 1)

        n_c, n_t = len(y_c), len(y_t)
        theta_exact = (exact_cov(y_t, x_t) / n_t + exact_cov(y_c, x_c) / n_c) / (
            exact_var(x_t) / n_t + exact_var(x_c) / n_c
        )
        mean = lambda a: sum(Fraction(v) for v in a) / len(a)  # noqa: E731
        contrast_exact = (mean(y_t) - mean(y_c)) - theta_exact * (mean(x_t) - mean(x_c))
        var_exact = sum(
            (exact_var(y) - 2 * theta_exact * exact_cov(y, x) + theta_exact**2 * exact_var(x))
            / len(y)
            for y, x in ((y_t, x_t), (y_c, x_c))
        )

        theta, _ = pooled_theta([control, treatment])
        c_adj, t_adj = cuped_adjust([control, treatment])
        assert theta == pytest.approx(float(theta_exact), rel=1e-9)
        contrast = t_adj.mean - c_adj.mean
        assert contrast == pytest.approx(float(contrast_exact), rel=1e-6)
        assert contrast == pytest.approx(difference, rel=1e-3)
        assert math.sqrt(t_adj.var / n_t + c_adj.var / n_c) == pytest.approx(
            math.sqrt(float(var_exact)), rel=1e-9
        )


# Validation


class TestCupedValidation:
    """CUPED raises errors for degenerate or missing input."""

    def test_unmaterialized_covariate_raises(self):
        """ArmStats with no covariate family raises."""
        control = _make_arm_stats(n=100, mean=10.0, var=4.0, group_id="A")
        treatment = _make_arm_stats(n=100, mean=11.0, var=4.0, group_id="B")
        with pytest.raises(InvalidRequestError) as exc_info:
            cuped_adjust([control, treatment])
        assert exc_info.value.code == "estimation.cuped.arm_no_covariate"

    @pytest.mark.filterwarnings(
        "ignore:.*centered (sum of squares|cross sum).*floating-point noise.*:RuntimeWarning"
    )
    def test_constant_covariate_raises(self):
        """Zero-variance covariate raises."""
        n = 100
        x = np.full(n, 5.0)  # constant
        y = np.random.default_rng(0).normal(10, 2, n)
        control = ArmStats.from_raw_sums(
            study_id="exp",
            metric="m",
            group_id="A",
            n=n,
            sum_y=float(y.sum()),
            sum_y2=float((y**2).sum()),
            sum_x=float(x.sum()),
            sum_x2=float((x**2).sum()),
            sum_xy=float((x * y).sum()),
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp",
            metric="m",
            group_id="B",
            n=n,
            sum_y=float(y.sum()),
            sum_y2=float((y**2).sum()),
            sum_x=float(x.sum()),
            sum_x2=float((x**2).sum()),
            sum_xy=float((x * y).sum()),
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            cuped_adjust([control, treatment])
        assert exc_info.value.code == "estimation.cuped.covariate_zero_variance"

    def test_single_arm_raises(self):
        """Fewer than 2 arms raises."""
        arm = _make_arm_stats(n=100, mean=10.0, var=4.0, group_id="A")
        with pytest.raises(InvalidRequestError) as exc_info:
            cuped_adjust([arm])
        assert exc_info.value.code == "estimation.cuped.pooled_theta_least"


# Step 3d: Both-methods test


class TestCupedBothMethods:
    """estimate_lift with unadjusted + cuped methods returns both labels."""

    def test_both_methods_returned(self):
        """Both methods return correctly labeled results from one summary—no re-query needed."""
        rng = np.random.default_rng(42)
        control, treatment = _generate_correlated_data(rng, n_per_arm=2000, rho=0.7, true_lift=0.0)

        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
            methods=[
                Method(name="unadjusted"),
                Method(name="cuped", variance_reduction="cuped"),
            ],
        ).results

        methods_found = {r.method for r in results}
        assert methods_found == {"unadjusted", "cuped"}, (
            f"Expected both methods, got {methods_found}"
        )

    def test_cuped_ci_tighter(self):
        """CUPED confidence interval is narrower than unadjusted on correlated data."""
        rng = np.random.default_rng(42)
        control, treatment = _generate_correlated_data(rng, n_per_arm=5000, rho=0.7, true_lift=0.0)

        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
            methods=[
                Method(name="unadjusted"),
                Method(name="cuped", variance_reduction="cuped"),
            ],
        ).results

        cuped_result = [r for r in results if r.method == "cuped"][0]
        unadj_result = [r for r in results if r.method == "unadjusted"][0]
        cuped_lift = cuped_result.require_lift()
        unadj_lift = unadj_result.require_lift()
        assert cuped_lift.ub is not None and cuped_lift.lb is not None
        assert unadj_lift.ub is not None and unadj_lift.lb is not None

        cuped_width = cuped_lift.ub - cuped_lift.lb
        unadj_width = unadj_lift.ub - unadj_lift.lb

        assert cuped_width < unadj_width, (
            f"CUPED CI width {cuped_width:.4f} should be narrower than unadjusted {unadj_width:.4f}"
        )

    def test_both_recover_unbiased_under_null(self):
        """Both methods recover ~0% lift under the null (within MC tolerance)."""
        rng = np.random.default_rng(42)
        control, treatment = _generate_correlated_data(rng, n_per_arm=10000, rho=0.7, true_lift=0.0)

        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
            methods=[
                Method(name="unadjusted"),
                Method(name="cuped", variance_reduction="cuped"),
            ],
        ).results

        for r in results:
            assert r.require_lift().value == pytest.approx(0.0, abs=0.03), (
                f"Method '{r.method}' lift {r.require_lift().value:.4f} deviates from 0"
            )


# CUPED + ratio metrics: adjust both components, keep their induced covariance


def _ratio_arm(group_id: str, num, den, cov, metric: str = "ratio_rev") -> ArmStats:
    """A centered (format-2) ratio arm from unit-level numerator/denominator/
    covariate arrays, including the covariate-denominator cross moment."""
    n = len(num)
    ref_y, ref_den, ref_x = float(num.mean()), float(den.mean()), float(cov.mean())
    dy, dden, dx = num - ref_y, den - ref_den, cov - ref_x
    return ArmStats(
        study_id="exp",
        metric=metric,
        group_id=group_id,
        n=n,
        ref_y=ref_y,
        cy1=float(dy.sum()),
        cy2=float((dy * dy).sum()),
        ref_x=ref_x,
        cx1=float(dx.sum()),
        cx2=float((dx * dx).sum()),
        cxy=float((dx * dy).sum()),
        x_role="covariate",
        ref_den=ref_den,
        cden1=float(dden.sum()),
        cden2=float((dden * dden).sum()),
        cyden=float((dy * dden).sum()),
        cxden=float((dx * dden).sum()),
    )


def _ratio_metric() -> RatioMetric:
    return RatioMetric(
        name="ratio_rev",
        entity="user",
        numerator=Measure(fact="num"),
        denominator=Measure(fact="den"),
    )


def _moment_args(moments):
    """The five moments plus n, in the order both ratio reductions take."""
    return (
        moments.num_bar,
        moments.den_bar,
        moments.var_num,
        moments.var_den,
        moments.cov_num_den,
        moments.n,
    )


def _ratio_units(rng, n, *, load_num, load_den, lift=1.0):
    cov = rng.normal(5.0, 1.5, n)
    den = np.clip(6.0 + load_den * (cov - 5.0) + rng.normal(0.0, 1.0, n), 0.5, None)
    num = den * (2.0 + load_num * (cov - 5.0)) * lift + rng.normal(0.0, 0.5, n)
    return num, den, cov


class TestCupedRatioMetrics:
    """A ratio metric adjusts BOTH components against the covariate and reads
    the result through the same delta-method interval an unadjusted ratio
    uses."""

    @pytest.mark.parametrize("projection", ["relative", "absolute"])
    def test_contrast_se_matches_a_numerically_differentiated_contrast(self, projection):
        """The reported contrast SE equals the delta-method SE obtained by
        numerically differentiating the adjusted contrast in the six raw arm
        means and charging each arm's own 3x3 (Num, Den, X) covariance.

        That reference shares no algebra with the implementation: the
        gradient is a finite difference of the estimator's own definition,
        and the covariance comes straight from the unit-level data. It pins
        the whole construction at once -- the two per-component slopes, the
        shared pooled anchor (which does NOT cancel in a ratio), and the
        covariance the shared covariate induces between the adjusted
        components.
        """
        rng = np.random.default_rng(404)
        units = {
            "control": _ratio_units(rng, 3000, load_num=0.4, load_den=0.5),
            "treatment": _ratio_units(rng, 2200, load_num=0.4, load_den=0.5, lift=1.05),
        }
        arms = {g: _ratio_arm(g, *units[g]) for g in ("control", "treatment")}
        treatment, control = arms["treatment"], arms["control"]
        fit = fit_ratio_cuped([control, treatment])

        n_t, n_c = treatment.n, control.n
        theta_num, theta_den = fit.numerator.theta, fit.denominator.theta

        def contrast(v):
            anchor = (n_t * v[2] + n_c * v[5]) / (n_t + n_c)
            num_t = v[0] - theta_num * (v[2] - anchor)
            den_t = v[1] - theta_den * (v[2] - anchor)
            num_c = v[3] - theta_num * (v[5] - anchor)
            den_c = v[4] - theta_den * (v[5] - anchor)
            if projection == "relative":
                return math.log(num_t / den_t) - math.log(num_c / den_c)
            return num_t / den_t - num_c / den_c

        centre = np.array([units[g][j].mean() for g in ("treatment", "control") for j in (0, 1, 2)])
        gradient = np.empty(6)
        for i in range(6):
            step = abs(centre[i]) * 1e-6
            up, down = centre.copy(), centre.copy()
            up[i] += step
            down[i] -= step
            gradient[i] = (contrast(up) - contrast(down)) / (2.0 * step)

        expected = math.sqrt(
            gradient[:3] @ np.cov(np.vstack(units["treatment"]), ddof=1) @ gradient[:3] / n_t
            + gradient[3:] @ np.cov(np.vstack(units["control"]), ddof=1) @ gradient[3:] / n_c
        )

        if projection == "relative":
            moments_t, moments_c = fit.relative_moments(treatment, control)
            se_t = ratio_log_mean_se(
                *_moment_args(moments_t), group_id="treatment", metric="ratio_rev"
            )[1]
            se_c = ratio_log_mean_se(
                *_moment_args(moments_c), group_id="control", metric="ratio_rev"
            )[1]
        else:
            moments_t, moments_c = fit.absolute_moments(treatment, control)
            se_t = ratio_abs_diff_se(*_moment_args(moments_t))[1]
            se_c = ratio_abs_diff_se(*_moment_args(moments_c))[1]
        assert math.hypot(se_t, se_c) == pytest.approx(expected, rel=1e-6)

    def test_covariate_induced_covariance_is_not_dropped(self):
        """Numerator and denominator built UNCORRELATED with each other while
        both load on the covariate: after adjustment they are correlated
        through it, and that correlation widens the interval.

        Leaving the raw ``Cov(Num, Den)`` in place -- the natural way to get
        this wrong -- reports a visibly narrower interval, so this fails if
        the cross correction is ever dropped.
        """
        rng = np.random.default_rng(77)
        load_num, load_den, var_e_den = 0.9, 0.6, 1.0
        # Couple the numerator to the denominator's own noise just enough to
        # cancel the covariate-mediated covariance between them.
        couple = -load_num * load_den * 1.5**2 / var_e_den
        arms = {}
        for group in ("control", "treatment"):
            cov = rng.normal(5.0, 1.5, 20000)
            e_den = rng.normal(0.0, math.sqrt(var_e_den), 20000)
            den = 6.0 + load_den * (cov - 5.0) + e_den
            num = 12.0 + load_num * (cov - 5.0) + rng.normal(0.0, 1.0, 20000) + couple * e_den
            arms[group] = _ratio_arm(group, num, den, cov)
        control, treatment = arms["control"], arms["treatment"]
        assert abs(control.cov_yden()) < 0.05, "fixture: raw components are uncorrelated"

        fit = fit_ratio_cuped([control, treatment])
        adjusted_t, adjusted_c = fit.relative_moments(treatment, control)
        # The shared covariate induces -load_num*load_den*Var(X).
        assert adjusted_c.cov_num_den == pytest.approx(
            -load_num * load_den * control.var_x(), rel=0.05
        )

        def joint_se(cov_t, cov_c):
            return math.hypot(
                ratio_log_mean_se(
                    adjusted_t.num_bar,
                    adjusted_t.den_bar,
                    adjusted_t.var_num,
                    adjusted_t.var_den,
                    cov_t,
                    adjusted_t.n,
                    group_id="treatment",
                    metric="ratio_rev",
                )[1],
                ratio_log_mean_se(
                    adjusted_c.num_bar,
                    adjusted_c.den_bar,
                    adjusted_c.var_num,
                    adjusted_c.var_den,
                    cov_c,
                    adjusted_c.n,
                    group_id="control",
                    metric="ratio_rev",
                )[1],
            )

        retained = joint_se(adjusted_t.cov_num_den, adjusted_c.cov_num_den)
        dropped = joint_se(treatment.cov_yden(), control.cov_yden())
        assert dropped < 0.9 * retained, (
            "dropping the induced covariance must visibly narrow the interval; "
            f"retained={retained:.8f} dropped={dropped:.8f}"
        )

    def test_predictive_covariate_narrows_the_interval_through_estimate_lift(self):
        rng = np.random.default_rng(2024)
        base = rng.normal(5.0, 1.5, 2500)
        arms = []
        for group, lift in (("control", 1.0), ("treatment", 1.05)):
            # Same covariate values in both arms, so no pre-period imbalance
            # is available to correct and only precision can change.
            cov = rng.permutation(base)
            den = np.clip(6.0 + 0.5 * (cov - 5.0) + rng.normal(0.0, 1.0, 2500), 0.5, None)
            num = den * (2.0 + 0.4 * (cov - 5.0)) * lift + rng.normal(0.0, 0.5, 2500)
            arms.append(_ratio_arm(group, num, den, cov))
        df = _summary_df(arms)

        results = estimate_lift(
            metrics=[_ratio_metric()],
            summary=df,
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
        ).results
        by_method = {r.method: r for r in results}
        plain, adjusted = by_method["unadjusted"], by_method["cuped"]

        def half_width(result):
            lift = result.require_lift()
            return (lift.ub - lift.lb) / 2.0

        assert half_width(adjusted) < 0.75 * half_width(plain)
        assert adjusted.require_lift().value == pytest.approx(plain.require_lift().value, rel=1e-9)
        assert adjusted.abs_se is not None and plain.abs_se is not None
        assert adjusted.abs_se < 0.75 * plain.abs_se
        # The Welch-Satterthwaite reference survives the adjustment.
        assert adjusted.reference_kind == "t"
        assert adjusted.reference_df == pytest.approx(plain.reference_df, rel=0.01)
        assert adjusted.abs_reference_kind == "t"

    def test_zero_correlation_covariate_leaves_the_estimate_alone(self):
        rng = np.random.default_rng(31337)
        base = rng.normal(5.0, 1.5, 2500)
        arms = []
        for group, lift in (("control", 1.0), ("treatment", 1.05)):
            cov = rng.permutation(base)
            den = np.clip(rng.normal(6.0, 1.0, 2500), 0.5, None)
            num = den * rng.normal(2.0, 0.6, 2500) * lift
            arms.append(_ratio_arm(group, num, den, cov))
        df = _summary_df(arms)

        results = estimate_lift(
            metrics=[_ratio_metric()],
            summary=df,
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
        ).results
        by_method = {r.method: r for r in results}
        plain, adjusted = by_method["unadjusted"], by_method["cuped"]

        assert adjusted.require_lift().value == pytest.approx(plain.require_lift().value, rel=1e-9)
        assert adjusted.abs_se == pytest.approx(plain.abs_se, rel=0.02)

    def test_moments_without_the_denominator_cross_refuse_by_name(self):
        """Raw-sum (format-1) moments have no column for Cov(covariate,
        denominator), so the denominator slope is not identified -- refused
        rather than silently computed with the term dropped."""
        arms = [
            ArmStats.from_raw_sums(
                study_id="exp",
                metric="ratio_rev",
                group_id=group,
                n=100,
                sum_y=sum_y,
                sum_y2=sum_y2,
                sum_x=500.0,
                sum_x2=2600.0,
                sum_xy=sum_xy,
                sum_den=500.0,
                sum_den2=2600.0,
                sum_yden=sum_yden,
            )
            for group, sum_y, sum_y2, sum_xy, sum_yden in (
                ("control", 1000.0, 10100.0, 5030.0, 5030.0),
                ("treatment", 1200.0, 14500.0, 6030.0, 6030.0),
            )
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                metrics=[_ratio_metric()],
                summary=_summary_df(arms),
                control_group="control",
                methods=[Method(name="cuped", variance_reduction="cuped")],
            )
        assert exc_info.value.code == "estimation.cuped.ratio_arm_no_denominator_cross"

    def test_non_covariate_x_family_refuses_by_name(self):
        """A clustered collapse repurposes the x family to carry cluster
        totals, and its cross moment with the denominator means something
        else entirely; ratio CUPED must not read it as a covariate."""
        rng = np.random.default_rng(9)
        arms = []
        for group in ("control", "treatment"):
            arm = _ratio_arm(group, *_ratio_units(rng, 200, load_num=0.4, load_den=0.5))
            arms.append(arm.model_copy(update={"x_role": "cluster_size"}))
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_ratio_cuped(arms)
        assert exc_info.value.code == "estimation.cuped.ratio_arm_covariate_role"


# Near-perfect correlation: the quadratic form must not crash on fp noise


class TestCupedNearPerfectCorrelation:
    """adjusted_var must be clamped to >=0 - near rho~=1, floating-point
    cancellation in the quadratic form can produce a tiny negative, which
    would otherwise crash se_log_mean's sqrt with a math domain error."""

    def test_near_perfect_correlation_does_not_crash(self):
        """Y = 2*X exactly (rho=1) for both arms: the quadratic form is
        theoretically 0, but floating-point summation can push it slightly
        negative - must clamp to 0, not raise."""
        rng = np.random.default_rng(7)
        n = 500
        x_c = rng.normal(10, 2, n)
        y_c = 2.0 * x_c  # exact linear relationship, rho = 1
        x_t = rng.normal(10, 2, n)
        y_t = 2.0 * x_t

        def _to_arm(x, y, group_id):
            return ArmStats.from_raw_sums(
                study_id="exp",
                metric="rev",
                group_id=group_id,
                n=n,
                sum_y=float(y.sum()),
                sum_y2=float((y**2).sum()),
                sum_x=float(x.sum()),
                sum_x2=float((x**2).sum()),
                sum_xy=float((x * y).sum()),
            )

        control = _to_arm(x_c, y_c, "control")
        treatment = _to_arm(x_t, y_t, "treatment")

        adjusted = cuped_adjust([control, treatment])

        for summary in adjusted:
            assert summary.var >= 0.0, f"adjusted variance {summary.var} is negative"
            assert math.isfinite(summary.var)
            # Must not crash computing the SE downstream (this is what a
            # negative var would do - math domain error in sqrt).
            from increment.estimation.variance import se_log_mean

            se = se_log_mean(summary.var, summary.mean, summary.n)
            assert math.isfinite(se)
            assert se >= 0.0


def test_adjusted_variance_is_invariant_to_extreme_covariate_units():
    """Rescaling X must not underflow ``theta**2 * Var(X)`` to zero."""
    adjusted_vars = []
    for cx2, cxy in ((9.0, 9e-50), (9e300, 9e100)):
        arms = [
            ArmStats(
                study_id="e",
                metric="m",
                group_id=group_id,
                n=10,
                ref_y=10.0,
                cy1=0.0,
                cy2=18e-100,
                ref_x=0.0,
                cx1=0.0,
                cx2=cx2,
                cxy=cxy,
                x_role="covariate",
            )
            for group_id in ("control", "treatment")
        ]
        adjusted_vars.append([summary.var for summary in fit_cuped(arms).adjust()])

    for pair in adjusted_vars:
        for adjusted_var in pair:
            assert adjusted_var == pytest.approx(1e-100, rel=1e-12, abs=0.0)
    assert adjusted_vars[1] == pytest.approx(adjusted_vars[0], rel=1e-12, abs=0.0)


def test_ratio_covariate_rescaling_preserves_both_projected_covariances():
    for cx2, cxy in ((9.0, 9e-50), (9e300, 9e100)):
        arms = [
            ArmStats(
                study_id="e",
                metric="ratio",
                group_id=group,
                n=10,
                ref_y=3e-50,
                cy1=0.0,
                cy2=18e-100,
                ref_x=0.0,
                cx1=0.0,
                cx2=cx2,
                cxy=cxy,
                x_role="covariate",
                ref_den=2e-50,
                cden1=0.0,
                cden2=18e-100,
                cyden=9e-100,
                cxden=cxy,
            )
            for group in ("control", "treatment")
        ]
        fit = fit_ratio_cuped(arms)
        for project in (fit.relative_moments, fit.absolute_moments):
            for moments in project(arms[1], arms[0]):
                assert moments.var_num == pytest.approx(1e-100, rel=1e-12, abs=0.0)
                assert moments.var_den == pytest.approx(1e-100, rel=1e-12, abs=0.0)
                # Independent residuals remain uncorrelated after removing shared X.
                assert moments.cov_num_den == pytest.approx(0.0, abs=1e-112)


def test_ratio_adjustment_preserves_covariance_with_heterogeneous_component_slopes():
    arms = [
        ArmStats(
            study_id="e",
            metric="ratio",
            group_id=group,
            n=10,
            ref_y=1.0,
            cy1=0.0,
            cy2=9 * variance_y,
            ref_x=0.0,
            cx1=0.0,
            cx2=9 * variance_x,
            cxy=cross_y,
            x_role="covariate",
            ref_den=1.0,
            cden1=0.0,
            cden2=9e-308,
            cyden=0.0,
            cxden=cross_den,
        )
        for group, variance_y, variance_x, cross_y, cross_den in (
            ("control", 2e292, 1e-108, 9e92, 9e-308),
            ("treatment", 1e92, 1e-308, 0.0, 0.0),
        )
    ]
    fit = fit_ratio_cuped(arms)
    for project in (fit.relative_moments, fit.absolute_moments):
        adjusted, _ = project(arms[1], arms[0])
        # The fitted slopes are 1e200 and 1e-200; their product times Var(X) is 1e-308.
        assert adjusted.cov_num_den == pytest.approx(1e-308, rel=1e-12, abs=0.0)


def test_cuped_reciprocal_scaling_preserves_constant_covariate_arms():
    for variance_x, covariance in ((1.0, 1e50), (1e-300, 1e-100)):
        arms = [
            ArmStats(
                study_id="e",
                metric="m",
                group_id=group,
                n=10,
                ref_y=3e50,
                cy1=0.0,
                cy2=9e100 if group == "constant" else 18e100,
                ref_x=0.0,
                cx1=0.0,
                cx2=0.0 if group == "constant" else 9 * variance_x,
                cxy=0.0 if group == "constant" else 9 * covariance,
                x_role="covariate",
            )
            for group in ("control", "treatment", "constant")
        ]
        for summary in fit_cuped(arms).adjust():
            assert summary.var == pytest.approx(1e100, rel=1e-12, abs=0.0)


class TestCupedLargeOffsetCovariate:
    """Raw-sum cancellation must not conceal an invalid covariance."""

    def test_cancelled_covariate_variance_refuses_nonzero_covariance(self):
        # One ulp below the squared-mean term clamps Var(X) to zero;
        # the retained cross moment cannot then satisfy Cauchy–Schwarz.
        with pytest.warns(RuntimeWarning):
            arms = [
                ArmStats.from_raw_sums(
                    study_id="exp1",
                    metric="rev",
                    group_id=group_id,
                    n=10_000,
                    sum_y=50_000.0,
                    sum_y2=260_000.0,
                    sum_x=1e12,
                    sum_x2=math.nextafter(1e20, -math.inf),
                    sum_xy=5_000_000_009_000.0,
                )
                for group_id in ("control", "treatment")
            ]
        with pytest.raises(InvalidRequestError) as exc_info:
            cuped_adjust(arms)
        assert exc_info.value.code == "estimation.armstats.cross_moment_materially"
        assert exc_info.value.context["cross"] == 9000.0
        assert exc_info.value.context["var_b"] == 0.0

    def test_var_x_zero_message_names_cancellation(self):
        """When the centered covariate sum collapses to zero at a large
        offset, the refusal must name floating-point cancellation as a
        cause, not just a constant covariate."""
        # Exactly-constant covariate at a large offset: sum_x2 == sum_x**2/N.
        arms = []
        for group_id in ("control", "treatment"):
            arms.append(
                ArmStats.from_raw_sums(
                    study_id="exp1",
                    metric="rev",
                    group_id=group_id,
                    n=100,
                    sum_y=500.0,
                    sum_y2=2600.0,
                    sum_x=1e10,
                    sum_x2=1e18,
                    sum_xy=5e10,
                )
            )
        with pytest.raises(InvalidRequestError) as exc_info:
            cuped_adjust(arms)
        assert exc_info.value.code == "estimation.cuped.covariate_zero_variance"


class TestWithinArmThetaUsesCorrectedMoments:
    """`cxy`/`cx2` are centered on each arm's STORED reference, which need not
    be its exact mean; the residual is carried in cx1/cy1. Summing the raw
    fields biases theta and disagrees with the adjusted variance, which is
    built from the corrected moments."""

    @staticmethod
    def _arm(group_id: str, seed: int, reference_offset: float) -> ArmStats:
        rng = np.random.default_rng(seed)
        n = 200
        x = rng.normal(10.0, 2.0, n)
        y = 2.0 + 1.5 * x + rng.normal(0.0, 1.0, n)
        # A reference deliberately off the exact mean, as an upstream
        # aggregation does when it reuses a pooled reference.
        ref_y = float(y.mean()) + reference_offset
        ref_x = float(x.mean()) + reference_offset
        return ArmStats(
            study_id="s",
            metric="m",
            group_id=group_id,
            n=n,
            ref_y=ref_y,
            cy1=float((y - ref_y).sum()),
            cy2=float(((y - ref_y) ** 2).sum()),
            ref_x=ref_x,
            cx1=float((x - ref_x).sum()),
            cx2=float(((x - ref_x) ** 2).sum()),
            cxy=float(((y - ref_y) * (x - ref_x)).sum()),
            x_role="covariate",
        )

    @pytest.mark.parametrize("reference_offset", [0.0, 0.5, 2.0])
    def test_theta_matches_the_corrected_moment_ratio(self, reference_offset):
        arms = [
            self._arm("control", 7, reference_offset),
            self._arm("treatment", 8, reference_offset),
        ]
        theta = _within_arm_theta(arms)
        expected_num = sum(a.cov_yx() / a.n for a in arms)
        expected_den = sum(a.var_x() / a.n for a in arms)
        assert theta == pytest.approx(expected_num / expected_den, rel=1e-12)

    def test_an_offset_reference_no_longer_biases_theta(self):
        """The raw-field sum was ~17% low at an offset of one covariate SD; the
        corrected sum is invariant to where the reference sits."""
        centered = _within_arm_theta([self._arm("control", 7, 0.0), self._arm("treatment", 8, 0.0)])
        offset = _within_arm_theta([self._arm("control", 7, 2.0), self._arm("treatment", 8, 2.0)])
        assert offset == pytest.approx(centered, rel=1e-9)


class TestPerArmFeasibilityIsNotCancellable:
    """The adjusted-variance check evaluates the quadratic at the shared theta,
    where two arms with equal variances and opposite impossible covariances
    cancel to theta == 0 and both come out with positive adjusted variances."""

    def test_opposite_impossible_covariances_are_still_refused(self):
        arms = []
        for group_id, sign in (("control", 1.0), ("treatment", -1.0)):
            arms.append(
                ArmStats(
                    study_id="s",
                    metric="m",
                    group_id=group_id,
                    n=100,
                    ref_y=10.0,
                    cy1=0.0,
                    cy2=100.0,
                    ref_x=5.0,
                    cx1=0.0,
                    cx2=100.0,
                    # |cxy| far past sqrt(cy2 * cx2) = 100.
                    cxy=sign * 10_000.0,
                    x_role="covariate",
                )
            )
        with pytest.raises(InvalidRequestError) as exc_info:
            cuped_adjust(arms)
        assert exc_info.value.code == "estimation.armstats.cross_moment_violates"


class TestZeroVarianceArmIsExcludedFromTheta:
    """An arm whose covariate is constant WITHIN the arm identifies nothing
    about theta. Its cross moment must leave with its (zero) sum of squares:
    the feasibility check tolerates a cross moment inside its zero-variance
    rounding floor, and adding that to the numerator alone moves theta with no
    denominator behind it."""

    @staticmethod
    def _arm(group_id: str, *, cx2: float, cxy: float) -> ArmStats:
        return ArmStats(
            study_id="s",
            metric="m",
            group_id=group_id,
            n=100,
            ref_y=10.0,
            cy1=0.0,
            cy2=100.0,
            ref_x=5.0,
            cx1=0.0,
            cx2=cx2,
            cxy=cxy,
            x_role="covariate",
        )

    def test_a_constant_covariate_arm_does_not_move_theta(self):
        informative = self._arm("control", cx2=100.0, cxy=40.0)
        # Zero within-arm covariate variance with a denormal-scale cross moment,
        # which the feasibility floor accepts.
        degenerate = self._arm("treatment", cx2=0.0, cxy=5e-320)
        theta_alone = _within_arm_theta([informative, informative])
        theta_with_degenerate = _within_arm_theta([informative, degenerate])
        assert theta_with_degenerate == pytest.approx(theta_alone, rel=1e-12)

    def test_every_arm_constant_still_refuses(self):
        flat = self._arm("control", cx2=0.0, cxy=0.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            fit_cuped([flat, flat])
        assert exc_info.value.code == "estimation.cuped.covariate_zero_variance"


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.cuped.pooled_theta_least",
            lambda: cuped_adjust([_make_arm_stats(n=100, mean=10.0, var=4.0, group_id="A")]),
        ),  # estimation/cuped.py::_within_arm_theta
    ],
)
def test_cuped_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
