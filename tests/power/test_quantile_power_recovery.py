"""Monte Carlo verification that quantile power planning's variance
matches the runtime's own construction: `required_sample_size`'s planned
power must track the empirically measured power of a fresh two-arm
quantile test repeated many times at the planned size, not the
mean-metric variance approximation that previously undersized a p90
metric by close to an order of magnitude.

The tolerance follows the same pattern as ``TestCalibration`` in
``test_calibration.py``: a fixed window around the planned power, wide
enough to absorb both Monte Carlo noise and the order-statistic quantile
estimator's own small, well-known finite-sample excess variance (a few
percent above its asymptotic value at moderate n and an extreme
quantile -- present even with an exact, noiseless baseline variance, so
it is not something the deterministic pilot projection could remove),
while remaining far narrower than the gap the mean-metric-variance defect
produced (empirical power near the null's own rejection rate instead of
the target).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from increment.estimation.quantile import log_quantile_se
from increment.frame import MetricSpec, synthesise_metric
from increment.power.core import PowerDesign, QuantileBaseline, required_sample_size
from increment.semantics.models import QuantileMetric

from ._procedures import make_procedure

_TOLERANCE = 0.08


def _two_arm_quantile_reject(
    control: np.ndarray, treatment: np.ndarray, q: float, alpha: float
) -> bool:
    """Two-sided Wald test of a zero log-ratio between two independent
    quantile estimates -- the same decision rule the arm-shape planning
    model (`_arm_log_se_sq`'s quantile branch) sizes for."""
    point_c, se_c = log_quantile_se(control, q, alpha=alpha)
    point_t, se_t = log_quantile_se(treatment, q, alpha=alpha)
    z = (math.log(point_t) - math.log(point_c)) / math.hypot(se_t, se_c)
    return abs(z) > norm.isf(alpha / 2.0)


def _measured_power(
    rng: np.random.Generator,
    *,
    log_mean: float,
    log_sigma: float,
    q: float,
    n_per_arm: int,
    relative_lift: float,
    alpha: float,
    reps: int,
) -> float:
    scale = math.exp(math.log1p(relative_lift))
    rejections = 0
    for _ in range(reps):
        control = rng.lognormal(log_mean, log_sigma, n_per_arm)
        treatment = rng.lognormal(log_mean, log_sigma, n_per_arm) * scale
        if _two_arm_quantile_reject(control, treatment, q, alpha):
            rejections += 1
    return rejections / reps


def _planned_result(rng: np.random.Generator, *, q: float, relative_lift: float):
    pilot = rng.lognormal(5.0, 0.8, 5000)
    metric = synthesise_metric(MetricSpec(name="latency_ms", type="quantile", quantile=q))
    assert isinstance(metric, QuantileMetric)
    baseline = QuantileBaseline.from_control_values(metric, pilot)
    procedure = make_procedure(
        metric_type="quantile",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
    )
    design = PowerDesign(power=0.8)
    result = required_sample_size(relative_lift, baseline, procedure, design)
    return result, procedure.compiled_alpha


class TestQuantilePowerCalibration:
    """`required_sample_size`'s planned power for a quantile metric must
    track its empirically measured power -- the p0ma defect (planning
    reused the mean-metric variance, undersizing by up to ~9x at the
    90th percentile) is fixed exactly when these agree within the same
    tolerance window this codebase already uses for a planning model with
    a known, bounded, small approximation gap (`TestCalibration` in
    `test_calibration.py`)."""

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_planned_power_matches_measured_power_at_q90(self):
        rng = np.random.default_rng(11)
        result, alpha = _planned_result(rng, q=0.9, relative_lift=0.15)

        reps = 1500
        measured = _measured_power(
            rng,
            log_mean=5.0,
            log_sigma=0.8,
            q=0.9,
            n_per_arm=result.n_per_arm,
            relative_lift=0.15,
            alpha=alpha,
            reps=reps,
        )
        assert abs(measured - result.power) <= _TOLERANCE, (
            f"planned n_per_arm={result.n_per_arm} for planned power {result.power:.4f} "
            f"measured empirical power {measured:.4f} over {reps} reps"
        )

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_planned_power_matches_measured_power_at_median(self):
        rng = np.random.default_rng(13)
        result, alpha = _planned_result(rng, q=0.5, relative_lift=0.1)

        reps = 1500
        measured = _measured_power(
            rng,
            log_mean=5.0,
            log_sigma=0.8,
            q=0.5,
            n_per_arm=result.n_per_arm,
            relative_lift=0.1,
            alpha=alpha,
            reps=reps,
        )
        assert abs(measured - result.power) <= _TOLERANCE, (
            f"planned n_per_arm={result.n_per_arm} for planned power {result.power:.4f} "
            f"measured empirical power {measured:.4f} over {reps} reps"
        )

    def test_planned_power_matches_measured_power_at_q90_smoke(self):
        """Small-N fast smoke: same construction, far fewer reps and a
        looser tolerance (absorbing the larger Monte Carlo noise at this
        rep count) -- catches a gross undersizing regression (like the
        original mean-metric-variance defect, which missed by close to an
        order of magnitude) without the slow tier's rep count."""
        rng = np.random.default_rng(11)
        result, alpha = _planned_result(rng, q=0.9, relative_lift=0.15)

        reps = 150
        measured = _measured_power(
            rng,
            log_mean=5.0,
            log_sigma=0.8,
            q=0.9,
            n_per_arm=result.n_per_arm,
            relative_lift=0.15,
            alpha=alpha,
            reps=reps,
        )
        assert abs(measured - result.power) <= _TOLERANCE + 0.08, (
            f"planned n_per_arm={result.n_per_arm} for planned power {result.power:.4f} "
            f"measured empirical power {measured:.4f} over {reps} reps"
        )
