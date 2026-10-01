"""Independent references for increment/estimation/meta.py.

DerSimonian-Laird tau^2, the HKSJ pooled mean, and the Higgins-Thompson I^2
interval are checked against a frozen R metafor oracle (see
tests/oracles/generate/gen_meta_oracle.R). No R at test time.

The tau-marginalised segment posterior shared by
``marginalized_segment_intervals`` and ``segment_rollout`` is checked
against deterministic continuous quadrature of the same model, built here
from the model's formulas alone: the REML marginal likelihood with the
pooled mean profiled out, times a HalfNormal(tau_prior_scale) prior on tau,
mixed over Morris's (1983) conditional posterior

    theta_k | y, tau ~ N(mu(tau) + lambda_k (y_k - mu(tau)),
                         lambda_k v_k + (1 - lambda_k)^2 / W(tau)),
    w_k = 1 / (v_k + tau^2),  W = sum_k w_k,  lambda_k = tau^2 w_k.

The references locate the posterior support from the density itself and
bound the omitted upper tail analytically, so they cannot inherit a
truncated or unresolved integration domain from the implementation under
test.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import log_ndtr
from scipy.stats import norm

from increment.estimation.meta import (
    cochran_q,
    hksj_pooled_mean,
    marginalized_segment_intervals,
)

_FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "meta_dl_hksj.json").read_text())
_CASES = _FIXTURE["cases"]
_TOL = _FIXTURE["tolerance"]


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_dl_tau2_matches_metafor(case):
    het = cochran_q(case["input"]["est"], case["input"]["var"], alpha=case["input"]["alpha"])
    assert het.tau2 == pytest.approx(case["dl_tau2"], rel=_TOL["dl_tau2_rel"])


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_hksj_pooled_mean_matches_metafor_knha(case):
    het = cochran_q(case["input"]["est"], case["input"]["var"], alpha=case["input"]["alpha"])
    mu_hat, se, lower, upper = hksj_pooled_mean(
        case["input"]["est"], case["input"]["var"], het.tau2, alpha=case["input"]["alpha"]
    )
    ref = case["hksj"]
    assert mu_hat == pytest.approx(ref["mu_hat"], abs=_TOL["hksj_abs"])
    assert se == pytest.approx(ref["se"], abs=_TOL["hksj_abs"])
    assert lower == pytest.approx(ref["lower"], abs=_TOL["hksj_abs"])
    assert upper == pytest.approx(ref["upper"], abs=_TOL["hksj_abs"])


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_i2_ci_matches_metafor_higgins_thompson(case):
    het = cochran_q(case["input"]["est"], case["input"]["var"], alpha=case["input"]["alpha"])
    ref = case["i2_ci"]
    assert het.i2_lb == pytest.approx(ref["i2_lb_pct"] / 100.0, abs=_TOL["i2_ci_abs"])
    assert het.i2_ub == pytest.approx(ref["i2_ub_pct"] / 100.0, abs=_TOL["i2_ci_abs"])


# ---------------------------------------------------------------------------
# Tau-marginalised segment posterior: continuous quadrature reference
# ---------------------------------------------------------------------------

RTOL = 1e-6
"""Relative accuracy of centred posterior means, variances and rollout bias."""

SHRINK_ATOL = 1e-10
"""Absolute floor for the dimensionless shrinkage factor."""

MEAN_ATOL = 1e-10
"""Absolute mean floor in units of the input spread or smallest sampling SE."""

_QUAD_RTOL = 1e-11
_QUAD_LIMIT = 1000
_GL_RTOL = 1e-12
_GL_FINE = np.polynomial.legendre.leggauss(20)
_GL_COARSE = np.polynomial.legendre.leggauss(10)
_MAX_PANELS = 4096
_SCAN_POINTS = 4000
_TAIL_RTOL = 1e-15
"""Bound on the posterior mass beyond the integrated support, relative to the
integrated mass (times the largest factor any integrand multiplies it by)."""
_LEVEL_DROPS = (0.5, 2.0, 8.0, 32.0, 128.0)
"""Log-density drops below the peak that mark quadrature breakpoints; the
last one also bounds the retained support before the tail certificate."""


@dataclass(frozen=True, eq=False)
class SegmentPosterior:
    """Reference moments of the segment posterior at one input."""

    theta: np.ndarray
    """E[theta_k | y]."""
    shrink: np.ndarray
    """E[lambda_k | y]."""
    variance: np.ndarray
    """Var[theta_k | y] by the law of total variance over tau."""
    pooled_mean: float
    """E[mu(tau) | y]."""
    selection_bias: float | None
    """E[sum_k v_k / sqrt(2 pi d_k) exp(-(c - m_k)^2 / (2 d_k)) | y] with
    m_k, d_k - v_k the conditional mean and variance, when a threshold c is
    given."""
    center: float
    """Precision-weighted input location; keeps accuracy checks translation invariant."""
    mean_scale: float
    """Maximum centred input magnitude or smallest sampling SE, in outcome units."""


class _IntervalResult(Protocol):
    @property
    def theta(self) -> np.ndarray: ...
    @property
    def shrink_k(self) -> np.ndarray: ...
    @property
    def lb(self) -> np.ndarray: ...
    @property
    def ub(self) -> np.ndarray: ...


class _PricedResult(Protocol):
    @property
    def estimated_offset(self) -> float: ...
    @property
    def recommendation(self) -> str: ...
    @property
    def policy_value(self) -> float | None: ...
    @property
    def policy_value_raw(self) -> float | None: ...
    @property
    def selection_bias(self) -> float | None: ...


def segment_posterior(
    est: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    tau_prior_scale: float,
    cost_threshold: float | None = None,
) -> SegmentPosterior:
    """Reference posterior moments for ``(est, var)`` under the HalfNormal
    prior; equal variances use the reduced one-parameter density."""
    v = np.asarray(var, dtype=float)
    if np.all(v == v[0]):
        return _equal_variance_posterior(est, float(v[0]), tau_prior_scale, cost_threshold)
    return _general_posterior(est, v, tau_prior_scale, cost_threshold)


def assert_segment_intervals_match(
    result: _IntervalResult, reference: SegmentPosterior, alpha: float = 0.05
) -> None:
    """Posterior mean, interval centre, shrinkage and variance match quadrature."""
    z = float(norm.isf(alpha / 2.0))
    half_width = (np.asarray(result.ub, dtype=float) - np.asarray(result.lb, dtype=float)) / (
        2.0 * z
    )
    np.testing.assert_allclose(
        np.asarray(result.shrink_k, dtype=float), reference.shrink, rtol=RTOL, atol=SHRINK_ATOL
    )
    theta = np.asarray(result.theta, dtype=float)
    tolerance = (
        RTOL * np.minimum(np.abs(reference.theta - reference.center), np.abs(reference.theta))
        + MEAN_ATOL * reference.mean_scale
        + 4 * np.finfo(float).eps * np.maximum(np.abs(reference.theta), abs(reference.center))
    )
    np.testing.assert_array_less(np.abs(theta - reference.theta), tolerance)
    midpoint = 0.5 * np.asarray(result.lb) + 0.5 * np.asarray(result.ub)
    np.testing.assert_array_less(np.abs(midpoint - reference.theta), tolerance)
    np.testing.assert_allclose(half_width**2, reference.variance, rtol=RTOL)


def assert_rollout_matches(
    result: _PricedResult,
    reference: SegmentPosterior,
    *,
    est: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    cost_threshold: float,
) -> None:
    """Guard statistic, winner's-curse correction and priced value agree with
    the reference at a threshold the offset guard does not refuse."""
    est_arr = np.asarray(est, dtype=float)
    var_arr = np.asarray(var, dtype=float)
    assert reference.selection_bias is not None
    s_rms = math.sqrt(float(var_arr.mean()))
    offset = (cost_threshold - reference.pooled_mean) / s_rms
    span = float(est_arr.max() - est_arr.min())
    mean_tolerance = (
        RTOL * min(abs(reference.pooled_mean - reference.center), abs(reference.pooled_mean))
        + MEAN_ATOL * reference.mean_scale
        + 4 * np.finfo(float).eps * max(abs(reference.pooled_mean), abs(reference.center))
    )
    assert result.estimated_offset == pytest.approx(offset, abs=mean_tolerance / s_rms)
    bias = reference.selection_bias
    raw = math.fsum((est_arr[est_arr > cost_threshold] - cost_threshold).tolist())
    assert result.policy_value is not None
    assert result.policy_value_raw == pytest.approx(raw, rel=1e-12, abs=1e-12 * span)
    assert result.selection_bias == pytest.approx(bias, rel=RTOL)
    assert result.policy_value == pytest.approx(
        raw - bias, abs=RTOL * bias + 1e-12 * max(abs(raw), span)
    )
    assert result.recommendation == ("rollout" if raw - bias > 0.0 else "no_net_benefit")


def _integrate(integrand: Callable[[float], float], upper: float, points: Sequence[float]) -> float:
    interior = sorted({p for p in points if 0.0 < p < upper})
    value, _ = quad(
        integrand,
        0.0,
        upper,
        points=interior or None,
        epsabs=0.0,
        epsrel=_QUAD_RTOL,
        limit=_QUAD_LIMIT,
    )
    return float(value)


def _crossing(log_density: Callable[[float], float], level: float, lo: float, hi: float) -> float:
    """The unique point in [lo, hi] where a monotone log density meets ``level``."""
    return float(brentq(lambda t: log_density(t) - level, lo, hi))


def _equal_variance_posterior(
    est: Sequence[float] | np.ndarray,
    variance: float,
    tau_prior_scale: float,
    cost_threshold: float | None,
) -> SegmentPosterior:
    """Reduced reference for a common sampling variance ``v``.

    The pooled mean is ``mean(y)`` at every tau and, in ``t = tau / sqrt(v)``
    with ``S = sum (y - mean(y))^2 / v`` and ``s = tau_prior_scale / sqrt(v)``,

        log p(t | y) = -(K-1)/2 log(1 + t^2) - S / (2 (1 + t^2)) - t^2 / (2 s^2)

    up to a constant. Every reported moment follows from the shared
    ``lambda = t^2 / (1 + t^2)``: the conditional mean is
    ``mean(y) + lambda (y_k - mean(y))`` and the Morris conditional variance
    collapses to ``v (lambda + (1 - lambda) / K)``.

    Stationary points solve ``u^2 / s^2 + (K-1) u - S = 0`` in
    ``u = 1 + t^2``: one positive root, so the density is unimodal with its
    mode at ``sqrt(u - 1)`` when ``u > 1`` and at zero otherwise. For
    ``t >= B`` the density is at most ``t^-(K-1) exp(-t^2 / (2 s^2))``, whose
    tail integral is at most ``B^-(K-1) s sqrt(2 pi) Phi(-B / s)``.
    """
    y = np.asarray(est, dtype=float)
    k = y.size
    ybar = math.fsum(y.tolist()) / k
    sigma = math.sqrt(variance)
    dev = (y - ybar) / sigma
    s = tau_prior_scale / sigma
    ss = math.fsum((dev * dev).tolist())

    def log_density(t: float) -> float:
        return -0.5 * (k - 1) * math.log1p(t * t) - 0.5 * ss / (1.0 + t * t) - 0.5 * (t / s) ** 2

    u_mode = 2.0 * ss / ((k - 1) + math.sqrt((k - 1) ** 2 + 4.0 * ss / (s * s)))
    t_mode = math.sqrt(u_mode - 1.0) if u_mode > 1.0 else 0.0
    peak = log_density(t_mode)
    points = [t_mode]
    for drop in _LEVEL_DROPS:
        level = peak - drop
        if t_mode > 0.0 and log_density(0.0) < level:
            points.append(_crossing(log_density, level, 0.0, t_mode))
        hi = max(2.0 * t_mode, 1.0)
        while log_density(hi) > level:
            hi *= 2.0
        points.append(_crossing(log_density, level, t_mode, hi))

    def density(t: float) -> float:
        return math.exp(log_density(t) - peak)

    upper = max(points)
    while True:
        mass = _integrate(density, upper, points)
        log_tail = (
            -(k - 1) * math.log(upper)
            + math.log(s)
            + 0.5 * math.log(2.0 * math.pi)
            + float(log_ndtr(-upper / s))
        )
        if log_tail - peak <= math.log(_TAIL_RTOL * mass):
            break
        upper *= 2.0

    def lam(t: float) -> float:
        return t * t / (1.0 + t * t)

    mean_lam = _integrate(lambda t: lam(t) * density(t), upper, points) / mass
    var_lam = _integrate(lambda t: (lam(t) - mean_lam) ** 2 * density(t), upper, points) / mass

    selection_bias = None
    if cost_threshold is not None:
        gap = (cost_threshold - ybar) / sigma

        def bias_density(t: float) -> float:
            lam_t = lam(t)
            spread = 1.0 + lam_t + 1.0 / ((1.0 + t * t) * k)  # (v + conditional var) / v
            z = gap - lam_t * dev
            kernel = float(np.exp(-0.5 * z * z / spread).sum())
            return kernel / math.sqrt(2.0 * math.pi * spread) * density(t)

        selection_bias = sigma * _integrate(bias_density, upper, points) / mass

    deviation = y - ybar
    return SegmentPosterior(
        theta=ybar + mean_lam * deviation,
        shrink=np.full(k, mean_lam),
        variance=variance * (mean_lam + (1.0 - mean_lam) / k) + deviation**2 * var_lam,
        pooled_mean=ybar,
        selection_bias=selection_bias,
        center=ybar,
        mean_scale=max(sigma, float(np.max(np.abs(deviation)))),
    )


def _panel_integrals(
    integrand: Callable[[np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-panel integrals under the 20-point Gauss-Legendre rule, and their
    per-component discrepancy from the 10-point rule on the same panel."""
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    estimates = []
    for nodes, weights in (_GL_FINE, _GL_COARSE):
        t = mid[:, None] + half[:, None] * nodes
        values = integrand(t.ravel()).reshape(*t.shape, -1)
        estimates.append(half[:, None] * np.tensordot(values, weights, axes=([1], [0])))
    fine, coarse = estimates
    return fine, np.abs(fine - coarse)


def _adaptive_integral(
    integrand: Callable[[np.ndarray], np.ndarray],
    edges: np.ndarray,
    scales: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Integrate a vector integrand over ``[edges[0], edges[-1]]``, bisecting
    every panel above its share of the error budget until the summed
    discrepancy, measured per component in units of ``scales(total)``, is
    below ``_GL_RTOL``."""
    lo, hi = edges[:-1], edges[1:]
    value, error = _panel_integrals(integrand, lo, hi)
    while True:
        total = value.sum(axis=0)
        panel_error = (error / scales(total)).max(axis=1)
        assert np.isfinite(panel_error).all(), "reference integrand is not finite"
        if panel_error.sum() <= _GL_RTOL:
            return total
        split = panel_error > _GL_RTOL / panel_error.size
        assert lo.size + np.count_nonzero(split) <= _MAX_PANELS, (
            "reference quadrature did not converge within its panel budget"
        )
        mid = 0.5 * (lo[split] + hi[split])
        new_lo = np.concatenate([lo[split], mid])
        new_hi = np.concatenate([mid, hi[split]])
        new_value, new_error = _panel_integrals(integrand, new_lo, new_hi)
        keep = ~split
        lo = np.concatenate([lo[keep], new_lo])
        hi = np.concatenate([hi[keep], new_hi])
        value = np.concatenate([value[keep], new_value])
        error = np.concatenate([error[keep], new_error])


def _support_edges(model: _StandardizedModel, span: float) -> tuple[np.ndarray, float, float]:
    """Breakpoints ``[0, ..., upper]``, the scanned mode and its log density.

    ``upper`` grows until the analytic tail bound beyond it is below the
    peak density by the last level drop. Breakpoints bracket every scanned
    crossing of each level drop and every scanned local maximum retained by
    the support."""
    vmax = float(model.vs.max())
    # An informative prior's scale seeds the support; an effectively flat one
    # leaves it to the likelihood tail.
    prior_reach = math.exp(min(model.log_s, 30.0))
    upper = 8.0 * max(prior_reach, math.sqrt(vmax), span, 1.0)
    lowest = 1e-4 * math.sqrt(float(model.vs.min()))
    for _ in range(64):
        grid = np.concatenate(([0.0], np.geomspace(lowest, upper, _SCAN_POINTS)))
        scan = model.log_density(grid)
        peak = float(scan.max())
        if model.log_tail_bound(upper) - peak <= -_LEVEL_DROPS[-1]:
            break
        upper *= 16.0
    else:
        raise AssertionError("reference support search did not bound the posterior tail")
    t_mode = float(grid[int(np.argmax(scan))])
    marks = {0.0, upper, t_mode}
    for drop in _LEVEL_DROPS:
        above = scan >= peak - drop
        crossing = np.flatnonzero(above[1:] != above[:-1])
        marks.update(grid[crossing].tolist())
        marks.update(grid[crossing + 1].tolist())
    rising = scan[1:-1] > scan[:-2]
    falling = scan[1:-1] >= scan[2:]
    retained = scan[1:-1] >= peak - _LEVEL_DROPS[-1]
    marks.update(grid[1:-1][rising & falling & retained].tolist())
    return np.array(sorted(marks)), t_mode, peak


@dataclass(frozen=True, eq=False)
class _StandardizedModel:
    """The model with estimates centred and every scale divided by one unit
    ``sigma``. The prior enters as ``inv_s = sigma / tau_prior_scale`` and
    ``log_s = log(tau_prior_scale / sigma)``, so an effectively flat prior
    stays representable."""

    yc: np.ndarray
    vs: np.ndarray
    inv_s: float
    log_s: float

    def evaluate(self, t: np.ndarray) -> tuple[np.ndarray, ...]:
        """``(log density, mu, lambda, conditional mean, conditional variance)``
        at each ``t``, the log density up to a constant. ``1 - lambda_k`` is
        formed as ``v_k w_k`` rather than by subtraction."""
        t2 = (t * t)[:, None]
        u = self.vs + t2
        w = 1.0 / u
        total_w = w.sum(axis=1)
        mu = (w * self.yc).sum(axis=1) / total_w
        resid = self.yc - mu[:, None]
        log_density = (
            -0.5 * (np.log(u).sum(axis=1) + np.log(total_w) + (w * resid * resid).sum(axis=1))
            - 0.5 * (t * self.inv_s) ** 2
        )
        lam = t2 * w
        cond_mean = mu[:, None] + lam * resid
        cond_var = lam * self.vs + (self.vs * w) ** 2 / total_w[:, None]
        return log_density, mu, lam, cond_mean, cond_var

    def log_density(self, t: np.ndarray) -> np.ndarray:
        return self.evaluate(t)[0]

    def log_tail_bound(self, upper: float) -> float:
        """log of an upper bound on the unnormalised posterior mass beyond ``upper``.

        For ``t >= B``: ``prod_k (v_k + t^2)^(-1/2) <= t^-K``,
        ``W(t)^(-1/2) <= sqrt((vmax + t^2) / K)`` and ``exp(-Q / 2) <= 1``, so
        the likelihood is at most ``t^-(K-1) sqrt((1 + vmax / B^2) / K)``.
        Bounding ``t^-(K-1)`` by ``B^-(K-1)`` leaves the HalfNormal kernel,
        whose tail integrates to ``s sqrt(2 pi) Phi(-B / s)``; bounding the
        kernel by one instead leaves ``B^-(K-2) / (K-2)`` when ``K >= 3``.
        """
        k = self.yc.size
        log_b = math.log(upper)
        common = 0.5 * math.log1p(float(self.vs.max()) / (upper * upper)) - 0.5 * math.log(k)
        prior_tail = (
            -(k - 1) * log_b
            + self.log_s
            + 0.5 * math.log(2.0 * math.pi)
            + float(log_ndtr(-upper * self.inv_s))
        )
        if k < 3:
            return common + prior_tail
        return common + min(prior_tail, -(k - 2) * log_b - math.log(k - 2))


def _general_posterior(
    est: Sequence[float] | np.ndarray,
    var: np.ndarray,
    tau_prior_scale: float,
    cost_threshold: float | None,
) -> SegmentPosterior:
    """Reference for arbitrary sampling variances.

    Works in units of ``sigma = sqrt(median(var))`` around ``median(est)``.
    Deviations of the conditional means are integrated relative to their
    value at the mode, in units of ``sqrt(h)``, ``h = 1 / sum_k 1/v_k``; every
    conditional variance is at least ``h / 2``, so that unit bounds the
    relative error of each reported variance. The spread of the conditional
    means is integrated in a second pass around the first pass's means, so
    no variance is formed by cancellation.
    """
    y = np.asarray(est, dtype=float)
    k = y.size
    center = float(np.median(y))
    sigma = math.sqrt(float(np.median(var)))
    model = _StandardizedModel(
        yc=(y - center) / sigma,
        vs=var / (sigma * sigma),
        inv_s=sigma / tau_prior_scale,
        log_s=math.log(tau_prior_scale) - math.log(sigma),
    )
    vs = model.vs
    h = 1.0 / float(np.sum(1.0 / vs))
    root_h = math.sqrt(h)
    span = float(model.yc.max() - model.yc.min())
    gap = None if cost_threshold is None else (cost_threshold - center) / sigma
    bias_norm = float(np.sqrt(vs / (2.0 * math.pi)).sum())

    edges, t_mode, peak = _support_edges(model, span)
    _, mu_at_mode, _, mean_at_mode, _ = model.evaluate(np.array([t_mode]))
    mu_mode = float(mu_at_mode[0])
    mean_mode = mean_at_mode[0]

    signed = np.zeros(2 + 3 * k + (0 if gap is None else 1), dtype=bool)
    signed[1] = True
    signed[2 + 2 * k : 2 + 3 * k] = True

    def first(t: np.ndarray) -> np.ndarray:
        log_density, mu, lam, cond_mean, cond_var = model.evaluate(t)
        p = np.exp(log_density - peak)[:, None]
        parts = [
            p,
            (mu[:, None] - mu_mode) / root_h * p,
            lam * p,
            cond_var / vs * p,
            (cond_mean - mean_mode) / root_h * p,
        ]
        if gap is not None:
            spread = vs + cond_var
            kernel = (
                vs
                / np.sqrt(2.0 * math.pi * spread)
                * np.exp(-0.5 * (gap - cond_mean) ** 2 / spread)
            )
            parts.append(kernel.sum(axis=1, keepdims=True) / bias_norm * p)
        return np.concatenate(parts, axis=1)

    def first_scales(total: np.ndarray) -> np.ndarray:
        scales = np.abs(total)
        scales[signed] = np.maximum(scales[signed], total[0])
        return np.maximum(scales, np.finfo(float).tiny)

    first_total = _adaptive_integral(first, edges, first_scales)
    mass = float(first_total[0])
    theta_std = mean_mode + root_h * first_total[2 + 2 * k : 2 + 3 * k] / mass

    def second(t: np.ndarray) -> np.ndarray:
        log_density, _, _, cond_mean, _ = model.evaluate(t)
        p = np.exp(log_density - peak)[:, None]
        spread = (cond_mean - theta_std) / root_h
        return np.concatenate([p, spread * spread * p], axis=1)

    def second_scales(total: np.ndarray) -> np.ndarray:
        return np.maximum(np.abs(total), total[0])

    second_total = _adaptive_integral(second, edges, second_scales)

    tail = math.exp(model.log_tail_bound(float(edges[-1])) - peak)
    assert tail * max(2.0, (span / root_h) ** 2) <= _TAIL_RTOL * mass, (
        "reference support does not certify the posterior tail"
    )
    precisions = var.min() / var
    weighted_center = center + float(np.dot(precisions, y - center) / precisions.sum())
    return SegmentPosterior(
        theta=center + sigma * theta_std,
        shrink=first_total[2 : 2 + k] / mass,
        variance=sigma
        * sigma
        * (vs * first_total[2 + k : 2 + 2 * k] / mass + h * second_total[1:] / second_total[0]),
        pooled_mean=center + sigma * (mu_mode + root_h * float(first_total[1]) / mass),
        selection_bias=None if gap is None else sigma * bias_norm * float(first_total[-1]) / mass,
        center=weighted_center,
        mean_scale=max(math.sqrt(float(var.min())), float(np.max(np.abs(y - weighted_center)))),
    )


# Inputs spanning the posterior regimes: a mode far beyond eight prior scales
# (the recorded counterexample, and absolute-scale magnitudes under the
# default prior), mass concentrated well inside the first 0.1% of that range,
# unequal variances, and K=2's heavy likelihood tail.
POSTERIOR_CASES: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
    "documented_escaped_k100": (np.linspace(-20.0, 20.0, 100), np.full(100, 4.0), 0.30),
    "sharp_shared_node_alias_k5": (
        np.array([-4000.0, -2000.0, 0.0, 2000.0, 4000.0]),
        np.full(5, 100.0),
        0.30,
    ),
    "small_reported_mean_large_center_k2": (np.array([0.0, 1000.0]), np.ones(2), 0.30),
    "narrow_at_zero_k40": (1e-3 * np.linspace(-1.5, 1.5, 40), np.full(40, 1e-6), 0.30),
    "unequal_escaped_k6": (
        np.array([-40.0, -22.0, -5.0, 8.0, 25.0, 44.0]),
        np.array([1.0, 4.0, 0.25, 2.25, 9.0, 0.5]),
        0.30,
    ),
    "unequal_ordinary_k5": (
        np.array([0.10, 0.30, -0.05, 0.20, 0.0]),
        np.array([0.02, 0.5, 0.01, 2.0, 0.005]),
        0.30,
    ),
    "unequal_ordinary_k3": (np.array([0.10, 0.30, -0.05]), np.array([0.02, 0.05, 0.01]), 0.30),
    "unequal_ordinary_k3_flat_prior": (
        np.array([0.10, 0.30, -0.05]),
        np.array([0.02, 0.05, 0.01]),
        1e6,
    ),
    "two_segments_prior_tail": (np.array([-0.4, 0.5]), np.array([0.01, 0.09]), 0.30),
}
for _case_est, _case_var, _case_scale in POSTERIOR_CASES.values():
    _case_est.setflags(write=False)
    _case_var.setflags(write=False)

# (shift, factor): est -> shift + factor * est, var -> factor^2 var,
# tau_prior_scale -> factor * tau_prior_scale. The posterior is equivariant
# under both, so every transformed input must still match its own reference.
_TRANSFORMS: dict[str, tuple[float, float]] = {
    "as_given": (0.0, 1.0),
    "translated": (37.5, 1.0),
    "rescaled_down": (0.0, 1e-3),
    "rescaled_up": (0.0, 1e6),
    "rescaled_huge": (0.0, 1e100),
    "rescaled_tiny": (0.0, 1e-100),
}


@pytest.mark.parametrize("transform", _TRANSFORMS)
@pytest.mark.parametrize("case", POSTERIOR_CASES)
def test_marginalized_intervals_match_continuous_quadrature(case: str, transform: str):
    est, var, scale = POSTERIOR_CASES[case]
    shift, factor = _TRANSFORMS[transform]
    est, var, scale = shift + factor * est, factor**2 * var, factor * scale
    result = marginalized_segment_intervals(est, var, tau_prior_scale=scale)
    assert_segment_intervals_match(result, segment_posterior(est, var, scale))


def test_documented_escaped_posterior_reports_the_recorded_quadrature_mean():
    """The recorded counterexample: y = linspace(-20, 20, 100), every
    sampling variance 4, prior scale 0.30. The posterior mode is tau ~= 5.18,
    far beyond eight prior scales; one-dimensional quadrature of the same
    model gives the last segment's posterior mean as 17.40392 (a [0, 2.4]
    tau domain reports 11.79178)."""
    result = marginalized_segment_intervals(
        np.linspace(-20.0, 20.0, 100), np.full(100, 4.0), tau_prior_scale=0.30
    )
    assert result.theta[-1] == pytest.approx(17.40392, abs=5e-6 + RTOL * 17.40392)


@pytest.mark.parametrize(
    "case", ("documented_escaped_k100", "narrow_at_zero_k40", "unequal_escaped_k6")
)
def test_partial_budget_never_returns_an_unresolved_posterior(case, monkeypatch):
    from increment.errors import InvalidRequestError

    est, var, scale = POSTERIOR_CASES[case]
    reference = segment_posterior(est, var, scale)
    for budget in (8, 64, 512):
        monkeypatch.setattr("increment.estimation.meta._TAU_NODE_BUDGET", budget)
        try:
            result = marginalized_segment_intervals(est, var, tau_prior_scale=scale)
        except InvalidRequestError as exc:
            assert exc.code == "estimation.meta.posterior_integration_unresolved"
        else:
            assert_segment_intervals_match(result, reference)
