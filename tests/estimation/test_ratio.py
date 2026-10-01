"""Statistical validation of RatioVarianceModel via simulation.

Simulates correlated (num, den) pairs per unit across many units in one
arm, computes the ratio-metric SE via RatioVarianceModel.log_mean_se, and
compares against a bootstrap SE estimate.

This mirrors the pattern used in the paper / textbook delta-method tests
but with actual correlated data, not a single matrix multiplication.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pytest

from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.engine import estimate_lift
from increment.estimation.variance import RatioVarianceModel
from increment.semantics.models import RatioMetric
from tests.mc import nominal_band, replicate


class _RatioSimResult(NamedTuple):
    """Return type for simulate_ratio_arm: ArmStats + raw (num, den) draws."""

    arm_stats: ArmStats
    num_vals: np.ndarray
    den_vals: np.ndarray


def simulate_ratio_arm(
    n_units: int,
    num_mean: float = 100.0,
    den_mean: float = 50.0,
    num_cv: float = 0.3,
    den_cv: float = 0.2,
    rho: float = 0.5,
    rng: np.random.Generator | None = None,
) -> _RatioSimResult:
    """Simulate one arm with correlated numerator/denominator per unit.

    Parameters
    ----------
    n_units : int
        Number of units in the arm.
    num_mean, den_mean : float
        Mean of numerator and denominator.
    num_cv, den_cv : float
        Coefficient of variation for each (std / mean).
    rho : float
        Correlation between num and den.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    _RatioSimResult
        With ``.arm_stats`` (ArmStats centered on the simulated draws),
        ``.num_vals`` (per-unit numerator draws), and ``.den_vals`` (per-unit
        denominator draws).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Log-normal: easy way to get positive, right-skewed data with
    # controlled CV and correlation.
    num_sigma = math.sqrt(math.log(1.0 + num_cv**2))
    den_sigma = math.sqrt(math.log(1.0 + den_cv**2))
    cov_mat = np.array(
        [[num_sigma**2, rho * num_sigma * den_sigma], [rho * num_sigma * den_sigma, den_sigma**2]]
    )

    ln_data = rng.multivariate_normal(
        [math.log(num_mean) - 0.5 * num_sigma**2, math.log(den_mean) - 0.5 * den_sigma**2],
        cov_mat,
        size=n_units,
    )
    num_vals = np.exp(ln_data[:, 0])
    den_vals = np.exp(ln_data[:, 1])

    sum_y = float(num_vals.sum())
    sum_y2 = float((num_vals**2).sum())
    sum_den = float(den_vals.sum())
    sum_den2 = float((den_vals**2).sum())
    sum_yden = float((num_vals * den_vals).sum())

    return _RatioSimResult(
        arm_stats=ArmStats.from_raw_sums(
            study_id="sim",
            metric="ratio_test",
            group_id="A",
            n=n_units,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_den=sum_den,
            sum_den2=sum_den2,
            sum_yden=sum_yden,
        ),
        num_vals=num_vals,
        den_vals=den_vals,
    )


def bootstrap_ratio_se(
    num_vals: np.ndarray,
    den_vals: np.ndarray,
    n_reps: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """Bootstrap SE of log(num/den) = log(mean(num) / mean(den)).

    Parameters
    ----------
    num_vals, den_vals : np.ndarray
        Paired per-unit values.
    n_reps : int
        Number of bootstrap replicates.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    float
        Bootstrap standard error of log(ratio).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(num_vals)
    log_ratios = np.empty(n_reps)
    for i in range(n_reps):
        idx = rng.integers(0, n, size=n)
        log_ratios[i] = math.log(num_vals[idx].sum() / den_vals[idx].sum())

    return float(log_ratios.std(ddof=1))


class TestRatioStatisticalValidation:
    """Statistical validation: delta-method SE vs bootstrap SE."""

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_se_within_5pc_of_bootstrap(self):
        """RatioVarianceModel SE within 5% of 2000-rep bootstrap SE.

        This is a parameter-recovery / calibration test — it is
        excluded from the fast test suite.
        """
        rng = np.random.default_rng(12345)
        n_units = 2000
        result = simulate_ratio_arm(n_units, rng=rng)
        model = RatioVarianceModel()

        log_mean, se_delta = model.log_mean_se(result.arm_stats)
        _ = log_mean  # not testing the point estimate here

        # Bootstrap the SAME observed data
        se_boot = bootstrap_ratio_se(result.num_vals, result.den_vals, n_reps=2000, rng=rng)

        # Relative error
        rel_error = abs(se_delta - se_boot) / se_boot
        assert rel_error < 0.05, (
            f"Delta-method SE ({se_delta:.6f}) deviates from bootstrap "
            f"SE ({se_boot:.6f}) by {rel_error * 100:.1f}% (> 5% threshold)"
        )

    def test_small_n_finite_positive_se(self):
        """Small-N ratio arm yields a finite positive SE (smoke test)."""
        rng = np.random.default_rng(99)
        result = simulate_ratio_arm(n_units=10, rng=rng)
        model = RatioVarianceModel()

        log_mean, se = model.log_mean_se(result.arm_stats)
        assert math.isfinite(se), "SE should be finite"
        assert se > 0.0, "SE should be positive"
        assert math.isfinite(log_mean), "log_mean should be finite"


# Coverage of estimate_lift's CI (via RatioVarianceModel.log_mean_se) over many
# independent draws with a known relative lift; the SE check above uses one
# draw. This is the unclustered twin of test_cluster_ratio_coverage.py's "iid"
# reading.

_METRIC = RatioMetric(
    name="m",
    entity="u",
    numerator={"fact": "f", "aggregation": "sum"},
    denominator={"fact": "g", "aggregation": "sum"},
)
_COVERAGE_TRUE_LIFT = 0.20  # relative lift on the numerator mean; denominator mean unchanged


def _ratio_row(group: str, res: _RatioSimResult) -> dict:
    return centered_row_from_raw_sums(
        {
            "experiment_id": "e",
            "metric": "m",
            "group_id": group,
            "n": len(res.num_vals),
            "sum_y": float(res.num_vals.sum()),
            "sum_y2": float((res.num_vals**2).sum()),
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": float(res.den_vals.sum()),
            "sum_den2": float((res.den_vals**2).sum()),
            "sum_yden": float((res.num_vals * res.den_vals).sum()),
        }
    )


def _ratio_coverage_trial(i: int, *, n_units: int, seed_base: int) -> bool:
    """One replicate: does the two-arm ratio-metric CI cover the true
    relative lift?"""
    rng = np.random.default_rng(seed_base + i)
    control = simulate_ratio_arm(n_units, num_mean=100.0, den_mean=50.0, rho=0.5, rng=rng)
    treatment = simulate_ratio_arm(
        n_units, num_mean=100.0 * (1 + _COVERAGE_TRUE_LIFT), den_mean=50.0, rho=0.5, rng=rng
    )
    (result,) = estimate_lift(
        [_METRIC],
        [_ratio_row("C", control), _ratio_row("T", treatment)],
        control_group="C",
    ).results
    lift = result.require_lift()
    assert lift.lb is not None and lift.ub is not None
    return lift.lb <= _COVERAGE_TRUE_LIFT <= lift.ub


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_unclustered_ratio_interval_coverage_is_nominal():
    """The delta-method ratio CI covers the true 20% relative lift ~95% of
    the time over 1500 independent draws. MCSE at 1500 reps is ~0.56pp;
    k=3 gives ~3x that headroom around the 95% target."""
    reps = 1500
    cov = replicate(reps, lambda i: _ratio_coverage_trial(i, n_units=500, seed_base=555))
    lo, hi = nominal_band(0.95, reps, k=3.0)
    assert lo <= cov.rate <= hi, cov.rate


def test_unclustered_ratio_interval_coverage_smoke():
    """Fast, unmarked 40-rep small-N twin of the parameter_recovery check
    above. Bounds are nominal +/- k*mcse(nominal, reps) (see tests/mc.py)."""
    reps = 40
    cov = replicate(reps, lambda i: _ratio_coverage_trial(i, n_units=100, seed_base=17))
    lo, _ = nominal_band(0.95, reps, k=3.0)
    assert cov.rate >= lo, cov.rate
