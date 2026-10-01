"""Segment heterogeneity: Cochran's Q, DerSimonian-Laird tau^2, and Higgins-Thompson I^2.

Pure functions over ``(est, var)`` arrays, one row per declared segment;
callers own filtering and exclusion. ``increment.breakout.heterogeneity``
builds these arrays from a ``run_breakout`` call and applies design/outcome
exclusions.

Uses DerSimonian-Laird (DL), not Paule-Mandel: PM does not fix DL's
tau^2=0 boundary collapse (both truncate exactly when Cochran's
``Q(0) <= K-1``) and has worse RMSE than DL at small true tau. ``tau2``
feeds ``Q``, ``I^2``, and the HKSJ interval only;
``marginalized_segment_intervals`` marginalises over tau instead of
plugging in a point estimate (see its docstring).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.fft import dst
from scipy.special import expit, log_ndtr
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm as norm_dist

from increment.errors import (
    CodedModel,
    InvalidRequestError,
    raiser,
    refusals,
)
from increment.estimation._tails import student_t_isf, two_sided_critical_value, wald_bounds

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.meta.est_var_same": "est and var must have the same shape, got {est_arr} vs {var_arr}",
        "estimation.meta.est_var_shape": "est and var must be 1-D, got shape {est_arr}",
        "estimation.meta.need_least_estimable": "Need at least 2 estimable segments, got {k}",
        "estimation.meta.est_contains_non": "est contains non-finite values",
        "estimation.meta.var_finite_strictly": "var must be finite and strictly positive for every segment",
        "estimation.meta.tau_prior_scale": "tau_prior_scale must be finite and > 0, got {tau_prior_scale}",
        "estimation.meta.posterior_integration_unresolved": "the tau posterior (k={k}, tau_prior_scale={tau_prior_scale:.6g}) could not be integrated to the required accuracy (support tau in [{support[0]:.6g}, {support[1]:.6g}], estimated error {integration_error:.3g} times the tolerance, budget {node_budget} log-density evaluations) -- refusing rather than reporting an unconverged posterior",
        "estimation.meta.alpha_strictly_between": "alpha must be strictly between 0 and 1, got {alpha}",
        "estimation.meta.alpha_too_small": "alpha is too small: alpha / 2 underflows floating-point precision",
        "estimation.meta.var_a_finite": "var_a must be finite and > 0, got {var_a}",
        "estimation.meta.var_b_finite": "var_b must be finite and > 0, got {var_b}",
        "estimation.meta.est_a_est": "est_a and est_b must be finite",
        "estimation.meta.marginalized_segment.level_contradicts_alpha": "level={level} contradicts alpha={alpha}; expected level={expected_level}",
        "estimation.meta.marginalized_segment.length_shape": "{name} must be 1-D with length k={k}, got shape {arr_shape}",
        "estimation.meta.marginalized_segment.finite": "{name} must be finite, got {arr!r}",
        "estimation.meta.marginalized_segment.lb_exceed_ub": "lb must not exceed ub for every segment, got lb={lb!r} ub={ub!r}",
        "estimation.meta.tau2_finite": "tau2 must be finite and >= 0, got {tau2}",
        "estimation.meta.hksj_pooled_variance": "HKSJ pooled variance is degenerate (segment estimates are identical or numerically indistinguishable given their weights): the resulting interval would have zero width and is not meaningful -- check for duplicated segments",
    },
)
_raise = raiser(_REFUSALS)


def _validate_est_var(
    est: Sequence[float] | np.ndarray, var: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Shared input validation for ``(est, var)`` array pairs; returns them
    coerced to 1-D float arrays.

    Raises ``ValueError`` on mismatched shapes, non-1-D input, fewer than
    2 segments, a non-finite ``est`` value, or a non-finite/non-positive
    ``var`` value.
    """
    est_arr = np.asarray(est, dtype=float)
    var_arr = np.asarray(var, dtype=float)
    if est_arr.shape != var_arr.shape:
        _raise("estimation.meta.est_var_same", est_arr=est_arr.shape, var_arr=var_arr.shape)
    if est_arr.ndim != 1:
        _raise("estimation.meta.est_var_shape", est_arr=est_arr.shape)
    k = est_arr.shape[0]
    if k < 2:
        _raise("estimation.meta.need_least_estimable", k=k)
    if not np.all(np.isfinite(est_arr)):
        _raise("estimation.meta.est_contains_non")
    if not np.all(np.isfinite(var_arr)) or np.any(var_arr <= 0):
        _raise("estimation.meta.var_finite_strictly")
    return est_arr, var_arr


def _validate_tau_prior_scale(tau_prior_scale: float) -> None:
    """Shared validation for the HalfNormal(tau) prior scale.

    Raises ``ValueError`` unless finite and strictly positive: a non-finite
    value passes a bare ``<= 0`` check (NaN comparisons are always false)
    and would silently poison every downstream posterior quantity. Every
    finite positive scale is otherwise accepted; the posterior integration
    works in log space, so a large scale never overflows.
    """
    if not (math.isfinite(tau_prior_scale) and tau_prior_scale > 0):
        _raise("estimation.meta.tau_prior_scale", tau_prior_scale=tau_prior_scale)


def _validate_alpha(alpha: float) -> None:
    """Shared validation for a two-sided significance level.

    Raises ``ValueError`` unless ``0 < alpha < 1``: at the boundaries
    ``norm.ppf``/``t.ppf`` return an infinite- or zero-width interval
    silently rather than raising.
    """
    if not 0 < alpha < 1:
        _raise("estimation.meta.alpha_strictly_between", alpha=alpha)
    if alpha / 2.0 == 0.0:
        _raise("estimation.meta.alpha_too_small")


class HeterogeneityResult(BaseModel):
    """Cochran's Q, DerSimonian-Laird tau^2, and Higgins-Thompson I^2 over declared segments.

    ``i2`` is suppressed (``None``) below ``k=5``: under a true null,
    P(I^2 > 25%) is 26.4% at k=3, 25.5% at k=5, 21.3% at k=10, so the
    point estimate is too often a false positive there. The interval
    (``i2_lb``/``i2_ub``) is reported regardless. Both bounds are
    ``None`` at ``k=2``, where the Higgins-Thompson interval is undefined.
    """

    model_config = ConfigDict(frozen=True)

    k: int
    q: float
    df: int
    p_value: float
    tau2: float
    i2: float | None
    i2_lb: float | None
    i2_ub: float | None


def cochran_q(
    est: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> HeterogeneityResult:
    """Cochran's Q test for heterogeneity across declared segments, plus DL tau^2 and I^2.

    ``est``/``var`` are per-segment point estimates and sampling
    variances (e.g. log relative lift), same length and order; every
    variance must be finite and strictly positive - a zero-event arm
    (``v_k = 0``) must be excluded by the caller first. ``alpha`` is the
    two-sided significance level for the I^2 interval (default 0.05).
    """
    _validate_alpha(alpha)
    est_arr, var_arr = _validate_est_var(est, var)
    k = est_arr.shape[0]

    w = 1.0 / var_arr
    w_sum = w.sum()
    mu = float((w * est_arr).sum() / w_sum)
    df = k - 1
    q = float((w * (est_arr - mu) ** 2).sum())
    p_value = float(chi2_dist.sf(q, df))

    # DerSimonian-Laird: tau^2 = max(0, (Q - df) / (sum(w) - sum(w^2)/sum(w))).
    tau2 = max(0.0, (q - df) / (w_sum - (w**2).sum() / w_sum))

    i2_point = max(0.0, (q - df) / q) if q > 0 else 0.0
    i2 = i2_point if k >= 5 else None

    i2_lb, i2_ub = _higgins_thompson_i2_ci(q, k, alpha)

    return HeterogeneityResult(
        k=k, q=q, df=df, p_value=p_value, tau2=tau2, i2=i2, i2_lb=i2_lb, i2_ub=i2_ub
    )


def _higgins_thompson_i2_ci(q: float, k: int, alpha: float) -> tuple[float | None, float | None]:
    """Higgins & Thompson (2002) test-based CI for I^2, via a normal CI on ln(H).

    ``H = sqrt(Q / df)`` (floored at 1); SE[ln(H)] is piecewise (Higgins
    & Thompson 2002, p.1549): one branch for ``Q > k``, one for
    ``Q <= k`` (undefined at ``k=2``, so the CI is ``(None, None)``
    there; Q/tau^2 are unaffected).

    Reproduces the paper's Table II to its reported precision: (Q=14.4,
    k=24) -> I^2 in (0%, 45%); (81.5, 19) -> 78% (66%, 86%); (41.5, 7) ->
    86% (72%, 92%); (130.3, 3) -> 98% (97%, 99%).
    """
    df = k - 1
    if k == 2:
        return None, None

    h = max(1.0, np.sqrt(q / df))
    ln_h = float(np.log(h))
    if q > k:
        se_ln_h = 0.5 * (np.log(q) - np.log(df)) / (np.sqrt(2 * q) - np.sqrt(2 * df - 1))
    else:
        se_ln_h = np.sqrt(1.0 / (2 * (k - 2)) * (1.0 - 1.0 / (3 * (k - 2) ** 2)))

    z = two_sided_critical_value(norm_dist.isf, alpha, what="Higgins-Thompson I^2 interval")
    h_lo = max(1.0, float(np.exp(ln_h - z * se_ln_h)))
    h_hi = float(np.exp(ln_h + z * se_ln_h))
    i2_lb = max(0.0, (h_lo**2 - 1) / h_lo**2)
    i2_ub = max(0.0, (h_hi**2 - 1) / h_hi**2)
    return i2_lb, i2_ub


def pairwise_contrast_arrays(
    est_a: float, var_a: float, est_b: float, var_b: float, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """Closed-form contrast between two segments' log-scale estimates:
    two-sided Wald interval on ``diff`` at the requested ``alpha``.

    Segments are independent (each fits its own control arm and CUPED
    theta within the segment), so the variance is the plain sum with no
    covariance term.
    """
    _validate_alpha(alpha)
    if not (np.isfinite(var_a) and var_a > 0):
        _raise("estimation.meta.var_a_finite", var_a=var_a)
    if not (np.isfinite(var_b) and var_b > 0):
        _raise("estimation.meta.var_b_finite", var_b=var_b)
    if not (np.isfinite(est_a) and np.isfinite(est_b)):
        _raise("estimation.meta.est_a_est")

    diff = est_a - est_b
    se = float(np.sqrt(var_a + var_b))
    z = two_sided_critical_value(norm_dist.isf, alpha, what="pairwise segment contrast")
    lower, upper = wald_bounds(diff, z, se, what="pairwise segment contrast")
    return diff, se, lower, upper


_TAU_PRIOR_SCALE_DEFAULT = 0.30  # half-normal scale on tau, log-RR scale

_TAU_NODE_BUDGET = 4000
"""Most log-posterior evaluations one tau integration may spend, summed over
every support window and refinement level. Read at call time."""

_TAU_FIRST_LEVEL = 64
"""Fejér level opened on every new support window (63 nodes); its nested
32-level subset is the first refinement it is compared against."""

# Every reported posterior quantity (and its certified tail error) must agree
# between the last two refinement levels to within
# _TAU_RTOL * |value| + _TAU_ATOL * (the quantity's natural scale).
_TAU_RTOL = 1e-6
_TAU_ATOL = 1e-10

_TAU_TAIL_SHARE = 1e-2
"""Share of each quantity's absolute floor that a support window's certified
tails may spend when the window is chosen."""

_TAU_NARROWING_MARGIN = math.log(1e3)
"""Extra log-margin a narrowed window's tail certificates carry, so the
coarse normaliser it is chosen from may overstate the refined one up to
1000-fold without reopening the window."""

_TAU_RANGE_LIMIT = 1e150
"""Largest spread of the estimates, and largest standard-error ratio, in units
of the smallest standard error, whose squares the integrand keeps finite."""

_EXP_CAP = 700.0
_LOG2 = math.log(2.0)
_LOG_2PI = math.log(2.0 * math.pi)
_LOG_HALF_NORMAL = 0.5 * math.log(2.0 / math.pi)


class MarginalizedSegmentIntervals(CodedModel, BaseModel):
    """Per-segment shrunken estimates and intervals from a tau-marginalised
    random-effects posterior - replaces a plug-in ``tau^2`` for the
    per-segment path entirely.

    All four arrays are length ``k`` (one entry per input segment, same order).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    __hash__ = None
    """Unhashable: ``np.ndarray`` fields aren't hashable, so the
    synthesized ``frozen=True`` hash would raise ``TypeError`` anyway."""

    k: int
    theta: np.ndarray
    """Shrunken point estimate per segment: ``E[theta_k | y]``, marginalised over tau."""
    shrink_k: np.ndarray
    """``E[lambda_k | y]``, ``lambda_k = tau^2/(v_k+tau^2)`` - the
    marginalised posterior mean shrinkage factor, not a plug-in
    ``tau_hat^2/(tau_hat^2+v_k)`` (the plug-in disagrees with this
    estimate in ~45% of calls at a representative K=5 configuration).
    Strictly positive and strictly decreasing in ``v_k`` for fixed ``y``."""
    lb: np.ndarray
    ub: np.ndarray
    level: float
    alpha: float

    @model_validator(mode="after")
    def _check_array_shapes(self) -> MarginalizedSegmentIntervals:
        if not 0.0 < self.alpha < 1.0:
            _raise("estimation.meta.alpha_strictly_between", alpha=self.alpha)
        if self.alpha / 2.0 == 0.0:
            _raise("estimation.meta.alpha_too_small")
        expected_level = math.fsum((1.0, -self.alpha))
        if not math.isclose(self.level, expected_level, rel_tol=1e-12, abs_tol=1e-15):
            _raise(
                "estimation.meta.marginalized_segment.level_contradicts_alpha",
                expected_level=expected_level,
                alpha=self.alpha,
                level=self.level,
            )
        for name in ("theta", "shrink_k", "lb", "ub"):
            arr = getattr(self, name)
            if arr.ndim != 1 or arr.shape[0] != self.k:
                _raise(
                    "estimation.meta.marginalized_segment.length_shape",
                    arr_shape=arr.shape,
                    name=name,
                    k=self.k,
                )
            if not np.all(np.isfinite(arr)):
                _raise("estimation.meta.marginalized_segment.finite", arr=arr.tolist(), name=name)
        if np.any(self.lb > self.ub):
            _raise(
                "estimation.meta.marginalized_segment.lb_exceed_ub",
                lb=self.lb.tolist(),
                ub=self.ub.tolist(),
            )
        return self


class _TauPosterior(NamedTuple):
    """Converged tau-posterior expectations shared by
    :func:`marginalized_segment_intervals` and
    :mod:`increment.estimation.rollout`, in the caller's units."""

    theta: np.ndarray
    """(K,) ``E[theta_k | y]``."""
    shrink: np.ndarray
    """(K,) ``E[lambda_k | y]``, ``lambda_k = tau^2/(v_k+tau^2)``."""
    variance: np.ndarray
    """(K,) ``Var[theta_k | y] = E[Var(theta_k|y,tau)] + Var[E(theta_k|y,tau)]``."""
    pooled_mean: float
    """``E[mu_hat(tau) | y]``."""
    predictive_density: np.ndarray | None
    """(K,) posterior predictive density of a replicate estimate of segment
    ``k`` at ``predictive_at``: ``E[N(at; cond_mean_k, v_k + cond_var_k) | y]``.
    ``None`` unless requested."""


class _TauProblem(NamedTuple):
    """Moments centred on their precision-weighted mean, in smallest-SE units.
    Joint rescaling preserves numerical decisions; translating the estimates
    can change the reported-mean tolerance and therefore the refinement depth."""

    k: int
    y: np.ndarray
    v: np.ndarray
    log_v: np.ndarray
    log_v_max: float
    i_min: int
    """Index of the smallest variance: the largest weight at every tau."""
    log_s: float
    """log of the prior scale."""
    log_c: float
    """log of the transform scale ``c = min(prior scale, 1)``."""
    log_mass_const: float
    """``-K/2 log(2 pi) - 1/2 sum_k log v_k + log HalfNormal(0)``."""
    center: float
    unit: float
    log_unit: float
    var_min: float


class _TauNodes(NamedTuple):
    """Posterior pieces at nodes ``t`` of ``tau = c sinh(t)``, in Fejér
    level order (descending ``t``)."""

    t: np.ndarray
    log_tau: np.ndarray
    log_density: np.ndarray
    """(G,) unnormalised ``log[L(tau) HalfNormal(tau) dtau/dt]``."""
    log_weight_sum: np.ndarray
    """(G,) ``log sum_k w_k``, ``w_k = 1/(v_k+tau^2)``."""
    misfit: np.ndarray
    """(G,) ``Q(tau) = sum_k w_k (y_k - mu_hat(tau))^2``."""
    mu_hat: np.ndarray
    lam: np.ndarray
    """(G, K) ``lambda_k(tau)``."""
    cond_mean: np.ndarray
    """(G, K) Morris conditional mean of ``theta_k`` given ``(y, tau)``."""
    cond_var: np.ndarray
    """(G, K) Morris conditional variance of ``theta_k`` given ``(y, tau)``."""
    predictive: np.ndarray | None
    """(G, K) ``N(at; cond_mean_k, v_k + cond_var_k)`` when requested."""


class _TauLevel(NamedTuple):
    """Fine/coarse expectations and log normalisers on one support window."""

    log_z: float
    log_mass_ratio: float
    shrink: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    pooled: np.ndarray
    predictive: np.ndarray | None


class _TauWindow(NamedTuple):
    """Support ``[lo, hi]`` in ``t`` with log upper bounds on the
    unnormalised posterior mass it omits below and above."""

    lo: float
    hi: float
    log_mass_below: float
    log_mass_above: float


@functools.cache
def _fejer_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Fejér's second rule on ``[-1, 1]``: the ``n - 1`` interior Chebyshev
    extrema ``x_j = cos(j pi / n)`` (descending) and their positive weights
    ``w_j = (4/n) sin(theta_j) sum_{odd m < n} sin(m theta_j) / m``.

    Level ``n``'s nodes are exactly level ``2n``'s odd-indexed nodes, so a
    refinement reuses every earlier evaluation.
    """
    j = np.arange(1, n)
    theta = j * (math.pi / n)
    odd_reciprocals = np.where(j % 2 == 1, 1.0 / j, 0.0)
    # DST-I: out[i] = 2 sum_m a[m] sin(pi (i+1) (m+1) / n) - twice the
    # odd-harmonic sum at theta_{i+1}.
    harmonic_sums = dst(odd_reciprocals, type=1) / 2.0
    nodes = np.cos(theta)
    weights = (4.0 / n) * np.sin(theta) * harmonic_sums
    nodes.flags.writeable = False
    weights.flags.writeable = False
    return nodes, weights


def _interleave(coarse: np.ndarray, fresh: np.ndarray) -> np.ndarray:
    """Merge level-``n`` rows (``coarse``) with the ``n`` rows new at level
    ``2n`` (``fresh``) into level-``2n`` order."""
    merged = np.empty((coarse.shape[0] + fresh.shape[0], *coarse.shape[1:]))
    merged[0::2] = fresh
    merged[1::2] = coarse
    return merged


def _asinh_exp(x: float) -> float:
    """``asinh(exp(x))`` without overflow."""
    return x + _LOG2 if x > 20.0 else math.asinh(math.exp(x))


def _log_sinh(t: np.ndarray | float) -> np.ndarray:
    """``log(sinh(t))`` for ``t > 0``, accurate at both ends."""
    return t - _LOG2 + np.log(-np.expm1(-2.0 * t))


def _log_tau_at(t: float, problem: _TauProblem) -> float:
    return float(problem.log_c + _log_sinh(t))


def _caller_tau(t: float, problem: _TauProblem) -> float:
    """``tau`` at ``t`` in the caller's units, for failure reports."""
    if t <= 0.0:
        return 0.0
    log_tau = problem.log_unit + _log_tau_at(t, problem)
    return math.exp(log_tau) if log_tau < _EXP_CAP else math.inf


def _log_mass_above(log_tau: np.ndarray | float, problem: _TauProblem) -> np.ndarray:
    """log of an upper bound on the unnormalised posterior mass above
    ``tau`` (vectorised; non-increasing in ``tau``).

    For ``u >= B``: ``prod_k (v_k+u^2)^(-1/2) <= u^(-K)``,
    ``(sum_k w_k(u))^(-1/2) <= u sqrt((1 + max v/B^2)/K)`` and ``Q >= 0``,
    so ``L(u) <= (2 pi)^(-K/2) sqrt((1 + max v/B^2)/K) u^(-(K-1))``.
    Integrated against the HalfNormal density ``pi``,
    ``int_B^inf u^(-(K-1)) pi(u) du`` is at most ``B^(-(K-1)) P(tau > B)``
    and, for ``K >= 3``, at most ``pi(B) B^(-(K-2)) / (K-2)``; the smaller
    bound is used. ``tau/s`` is capped where its tail is already far below
    any floor, which only loosens the bound.
    """
    k = problem.k
    ratio = np.exp(np.minimum(log_tau - problem.log_s, 0.5 * _EXP_CAP))
    log_integral = _LOG2 + log_ndtr(-ratio) - (k - 1) * log_tau
    if k >= 3:
        log_prior_at = _LOG_HALF_NORMAL - problem.log_s - 0.5 * ratio * ratio
        log_integral = np.minimum(log_integral, log_prior_at - (k - 2) * log_tau - math.log(k - 2))
    log_envelope = -0.5 * k * _LOG_2PI + 0.5 * (
        np.logaddexp(0.0, problem.log_v_max - 2.0 * log_tau) - math.log(k)
    )
    return log_envelope + log_integral


def _log_mass_below(
    log_tau: np.ndarray,
    log_weight_sum: np.ndarray,
    misfit: np.ndarray,
    problem: _TauProblem,
) -> np.ndarray:
    """log of an upper bound on the unnormalised posterior mass below
    ``tau_a`` (vectorised; non-decreasing in ``tau_a``).

    On ``[0, tau_a]``: ``prod_k (v_k+u^2)^(-1/2) <= prod_k v_k^(-1/2)``;
    every ``w_k`` decreases in ``u``, so ``sum_k w_k(u) >= sum_k w_k(tau_a)``;
    ``Q(u) = min_mu sum_k w_k(u)(y_k-mu)^2`` is non-increasing, so
    ``Q(u) >= Q(tau_a)``; and ``pi(u) <= pi(0)``. The mass is at most
    ``tau_a`` times the product of those bounds.
    """
    return log_tau + problem.log_mass_const - 0.5 * (log_weight_sum + misfit)


def _tau_problem(
    est_arr: np.ndarray, var_arr: np.ndarray, tau_prior_scale: float
) -> _TauProblem | None:
    """Standardise validated inputs; ``None`` when the spread of ``est`` or
    the standard-error ratio exceeds what the integrand can square."""
    log_var = np.log(var_arr)
    i_min = int(np.argmin(var_arr))
    log_v = log_var - log_var[i_min]
    log_v_max = float(log_v.max())
    var_min = float(var_arr[i_min])
    log_unit = 0.5 * float(log_var[i_min])
    log_limit = math.log(_TAU_RANGE_LIMIT)
    span = float(est_arr.max()) - float(est_arr.min())
    if 0.5 * log_v_max > log_limit or (
        span > 0.0 and math.log(span) - log_unit > log_limit + _LOG2
    ):
        return None
    rel_precision = var_min / var_arr
    anchor = float(est_arr[i_min])
    center = anchor + float((rel_precision / rel_precision.sum()) @ (est_arr - anchor))
    deviation = est_arr - center
    spread = float(np.abs(deviation).max())
    if spread > 0.0 and math.log(spread) - log_unit > log_limit:
        return None
    unit = math.sqrt(var_min)
    log_s = math.log(tau_prior_scale) - log_unit
    k = est_arr.shape[0]
    return _TauProblem(
        k=k,
        y=deviation / unit,
        v=np.exp(log_v),
        log_v=log_v,
        log_v_max=log_v_max,
        i_min=i_min,
        log_s=log_s,
        log_c=min(log_s, 0.0),
        log_mass_const=(-0.5 * k * _LOG_2PI - 0.5 * float(log_v.sum()) + _LOG_HALF_NORMAL - log_s),
        center=center,
        unit=unit,
        log_unit=log_unit,
        var_min=var_min,
    )


def _tau_nodes(t: np.ndarray, problem: _TauProblem, at: float | None) -> _TauNodes:
    """Evaluate the log posterior and Morris pieces at nodes ``t`` > 0, in
    log space throughout: no ``tau^2``, weight or normaliser is formed
    where it could overflow."""
    log_tau = problem.log_c + _log_sinh(t)
    log_tau2 = 2.0 * log_tau
    log_ratio = log_tau2[:, None] - problem.log_v  # log(tau^2 / v_k)
    softplus = np.logaddexp(0.0, log_ratio)  # log(1 + tau^2/v_k) = -log(1 - lambda_k)
    lam = expit(log_ratio)
    log_w = -(problem.log_v + softplus)  # log w_k
    log_w_top = log_w[:, problem.i_min]
    w = np.exp(log_w - log_w_top[:, None])
    w_total = w.sum(axis=1)
    log_weight_sum = log_w_top + np.log(w_total)
    w /= w_total[:, None]
    mu_hat = w @ problem.y
    resid = problem.y - mu_hat[:, None]
    misfit = np.exp(log_weight_sum) * (w * resid * resid).sum(axis=1)
    # (tau/s)^2, capped where exp(-z/2) has long since vanished.
    prior_z = np.exp(np.minimum(log_tau2 - 2.0 * problem.log_s, _EXP_CAP))
    log_cosh = t - _LOG2 + np.log1p(np.exp(-2.0 * t))
    # REML marginal log-likelihood (mu profiled out) + HalfNormal log prior
    # + log dtau/dt, with sum_k log(v_k + tau^2) = sum_k log v_k + softplus.
    log_density = (problem.log_mass_const + problem.log_c + log_cosh) - 0.5 * (
        softplus.sum(axis=1) + log_weight_sum + misfit + prior_z
    )
    cond_mean = mu_hat[:, None] + lam * resid
    # (1 - lambda_k)^2 / sum_j w_j, formed in log space.
    cond_var = lam * problem.v + np.exp(-2.0 * softplus - log_weight_sum[:, None])
    predictive = None
    if at is not None:
        total = problem.v + cond_var
        predictive = np.exp(-0.5 * (at - cond_mean) ** 2 / total) / np.sqrt(2.0 * math.pi * total)
    return _TauNodes(
        t=t,
        log_tau=log_tau,
        log_density=log_density,
        log_weight_sum=log_weight_sum,
        misfit=misfit,
        mu_hat=mu_hat,
        lam=lam,
        cond_mean=cond_mean,
        cond_var=cond_var,
        predictive=predictive,
    )


def _merge_nodes(coarse: _TauNodes, fresh: _TauNodes) -> _TauNodes:
    if coarse.predictive is None:
        assert fresh.predictive is None
        predictive = None
    else:
        assert fresh.predictive is not None
        predictive = _interleave(coarse.predictive, fresh.predictive)
    return _TauNodes(
        t=_interleave(coarse.t, fresh.t),
        log_tau=_interleave(coarse.log_tau, fresh.log_tau),
        log_density=_interleave(coarse.log_density, fresh.log_density),
        log_weight_sum=_interleave(coarse.log_weight_sum, fresh.log_weight_sum),
        misfit=_interleave(coarse.misfit, fresh.misfit),
        mu_hat=_interleave(coarse.mu_hat, fresh.mu_hat),
        lam=_interleave(coarse.lam, fresh.lam),
        cond_mean=_interleave(coarse.cond_mean, fresh.cond_mean),
        cond_var=_interleave(coarse.cond_var, fresh.cond_var),
        predictive=predictive,
    )


def _tau_level(nodes: _TauNodes, fine_weights: np.ndarray, level: int, half: float) -> _TauLevel:
    """Integrate the nested rules independently before comparing their moments."""

    _, coarse_weights = _fejer_rule(level // 2)
    top = float(nodes.log_density.max())
    density = np.exp(nodes.log_density - top)
    fine_mass = fine_weights * density
    coarse_log = nodes.log_density[1::2]
    coarse_top = float(coarse_log.max())
    coarse_mass = coarse_weights * np.exp(coarse_log - coarse_top)
    fine_total = float(fine_mass.sum())
    coarse_total = float(coarse_mass.sum())
    # Normalised moments can agree when both rules collapse onto one shared node.
    log_mass_ratio = coarse_top - top + math.log(coarse_total) - math.log(fine_total)
    log_z = top + math.log(half * fine_total)
    weights = np.zeros((2, density.shape[0]))
    weights[0] = fine_mass / fine_total
    weights[1, 1::2] = coarse_mass / coarse_total

    shrink = weights @ nodes.lam
    mean = weights @ nodes.cond_mean
    theta = mean[0]
    deviation_sq = nodes.cond_mean - theta
    deviation_sq *= deviation_sq
    variance = weights @ nodes.cond_var + weights @ deviation_sq
    variance[1] -= (mean[1] - theta) ** 2
    pooled = weights @ nodes.mu_hat

    return _TauLevel(
        log_z=log_z,
        log_mass_ratio=log_mass_ratio,
        shrink=shrink,
        mean=mean,
        variance=variance,
        pooled=pooled,
        predictive=None if nodes.predictive is None else weights @ nodes.predictive,
    )


def _excess(
    pair: np.ndarray,
    tail: float | np.ndarray,
    floor: float | np.ndarray,
    *,
    relative: float | np.ndarray | None = None,
) -> float:
    """Worst ratio of (refinement difference + certified tail error) to
    ``_TAU_RTOL * relative + floor``; the default relative basis is ``|fine|``."""
    fine = pair[0]
    basis = np.abs(fine) if relative is None else relative
    return float(np.max((np.abs(fine - pair[1]) + tail) / (_TAU_RTOL * basis + floor)))


def _widened(window: _TauWindow, target: float, problem: _TauProblem) -> _TauWindow:
    """Open whichever end's certified tail exceeds ``target``."""
    lo, log_below = window.lo, window.log_mass_below
    if log_below > target:
        lo, log_below = 0.0, -math.inf
    hi, log_above = window.hi, window.log_mass_above
    if log_above > target:
        hi += max(hi - lo, 1.0)
        log_above = float(_log_mass_above(_log_tau_at(hi, problem), problem))
    return _TauWindow(lo, hi, log_below, log_above)


def _narrowed(
    nodes: _TauNodes, window: _TauWindow, target: float, problem: _TauProblem
) -> _TauWindow | None:
    """The narrowest node-bounded sub-window whose certified tails are each
    at most ``target``; ``None`` if no such window is proper."""
    above = _log_mass_above(nodes.log_tau, problem)
    below = _log_mass_below(nodes.log_tau, nodes.log_weight_sum, nodes.misfit, problem)
    # Level order descends in t: certify cuts from the top down and from the
    # bottom up, stopping at the first node either bound cannot certify.
    n_top = int(np.logical_and.accumulate(above <= target).sum())
    n_bottom = int(np.logical_and.accumulate(below[::-1] <= target).sum())
    hi, log_above = window.hi, window.log_mass_above
    if n_top:
        hi, log_above = float(nodes.t[n_top - 1]), float(above[n_top - 1])
    lo, log_below = window.lo, window.log_mass_below
    if n_bottom:
        lo, log_below = float(nodes.t[-n_bottom]), float(below[-n_bottom])
    return _TauWindow(lo, hi, log_below, log_above) if lo < hi else None


def _tau_posterior(
    est_arr: np.ndarray,
    var_arr: np.ndarray,
    tau_prior_scale: float,
    predictive_at: float | None = None,
) -> _TauPosterior:
    """Integrate the tau posterior - REML marginal likelihood (pooled mean
    profiled out) times ``HalfNormal(tau_prior_scale)`` - against Morris's
    conditional moments, to a verified accuracy or not at all.

    Numerics, all in standardised units (see :class:`_TauProblem`):

    - ``tau = c sinh(t)``, ``c = min(prior scale, smallest SE)``: linear
      near 0, where a boundary posterior lives, and logarithmic beyond,
      where an escaped one does; ``t = 0`` is ``tau = 0``, so no lower cut
      is needed. The integrand ``p(tau|y) dtau/dt`` is analytic in ``t``.
    - Nested Fejér levels on a support window ``[lo, hi]`` in ``t``, with
      the quadrature weights inside the normalised posterior weights. A
      level's estimates are accepted only when its normaliser and every reported quantity -
      ``E[theta_k]``, ``Var[theta_k]``, ``E[lambda_k]``, ``E[mu_hat]`` and,
      when requested, each predictive density - agrees with the previous
      level to ``_TAU_RTOL`` relative, over an absolute floor of
      ``_TAU_ATOL`` times its natural scale, after adding its certified
      tail error. Means use the stricter of reported and centred magnitudes.
      Otherwise the level doubles, reusing every evaluation.
    - Omitted mass is bounded in log space by :func:`_log_mass_below` and
      :func:`_log_mass_above`. With ``eps`` their sum over the integrated
      normaliser, a quantity ``f`` with range ``R`` moves by at most
      ``eps R``: ``R`` is the estimate span for means (every conditional
      mean lies between ``min y`` and ``max y``), 1 for ``lambda``,
      ``1/sqrt(2 pi v_k)`` for predictive densities, and
      ``2 (max v + span^2)`` for variances (conditional variances are at
      most ``max v``).
    - The first window ends at ``tau = max(10 s, 3 span, 1)``. After each
      unconverged level, an end whose certified tail exceeds a share of the
      absolute floors is opened, and otherwise the window is narrowed to
      the node-bounded sub-window the bounds certify whenever that is less
      than half as wide; either restarts at the first level. A peak is thus
      resolved by refinement, never by stretching a fixed grid.

    Raises ``estimation.meta.posterior_integration_unresolved`` when
    ``_TAU_NODE_BUDGET`` evaluations cannot meet that accuracy, or the
    inputs' spread cannot be represented; no truncated or unconverged
    posterior is ever returned.
    """
    budget = _TAU_NODE_BUDGET
    k = int(est_arr.shape[0])
    problem = _tau_problem(est_arr, var_arr, tau_prior_scale)
    if problem is None:
        _raise(
            "estimation.meta.posterior_integration_unresolved",
            k=k,
            tau_prior_scale=tau_prior_scale,
            support=(0.0, math.inf),
            integration_error=math.inf,
            node_budget=budget,
        )
    at = None
    if predictive_at is not None:
        # Beyond 100x the representable spread every predictive density is
        # exactly 0 in float64; clamping there keeps its square finite.
        limit = 100.0 * _TAU_RANGE_LIMIT
        at = min(max((predictive_at - problem.center) / problem.unit, -limit), limit)

    y = problem.y
    span = float(y.max() - y.min())
    y_scale = max(1.0, float(np.abs(y).max()))
    v_max = float(problem.v.max())
    v_fixed = 1.0 / float((1.0 / problem.v).sum())
    log_eps = math.log(_TAU_ATOL * _TAU_TAIL_SHARE) + min(
        0.0,
        math.log(y_scale / span) if span > 0.0 else 0.0,
        math.log(v_fixed / (2.0 * (v_max + span * span))),
    )

    log_first_hi = max(
        math.log(10.0) + problem.log_s,
        math.log(3.0 * span) if span > 0.0 else 0.0,
        0.0,
    )
    first_hi = _asinh_exp(log_first_hi - problem.log_c)
    window = _TauWindow(
        0.0,
        first_hi,
        -math.inf,
        float(_log_mass_above(_log_tau_at(first_hi, problem), problem)),
    )
    level = _TAU_FIRST_LEVEL
    nodes: _TauNodes | None = None
    used = 0
    error = math.inf
    while True:
        fresh_count = level - 1 if nodes is None else level // 2
        if used + fresh_count > budget:
            _raise(
                "estimation.meta.posterior_integration_unresolved",
                k=k,
                tau_prior_scale=tau_prior_scale,
                support=(_caller_tau(window.lo, problem), _caller_tau(window.hi, problem)),
                integration_error=error,
                node_budget=budget,
            )
        abscissae, fine_weights = _fejer_rule(level)
        half = 0.5 * (window.hi - window.lo)
        x = abscissae if nodes is None else abscissae[0::2]
        fresh = _tau_nodes(window.lo + half * (1.0 + x), problem, at)
        used += fresh_count
        nodes = fresh if nodes is None else _merge_nodes(nodes, fresh)

        level_values = _tau_level(nodes, fine_weights, level, half)
        theta = level_values.mean[0]
        reported_mean = problem.center + problem.unit * theta
        reported_pooled = problem.center + problem.unit * float(level_values.pooled[0])

        log_omitted = float(np.logaddexp(window.log_mass_below, window.log_mass_above))
        omitted = math.exp(min(log_omitted - level_values.log_z, 0.0))
        excesses = [
            abs(level_values.log_mass_ratio) / math.log1p(_TAU_RTOL),
            _excess(
                level_values.mean,
                omitted * span,
                _TAU_ATOL * y_scale,
                relative=np.minimum(problem.unit * np.abs(theta), np.abs(reported_mean))
                / problem.unit,
            ),
            _excess(level_values.shrink, omitted, _TAU_ATOL),
            _excess(
                level_values.variance, 2.0 * omitted * (v_max + span * span), _TAU_ATOL * v_fixed
            ),
            _excess(
                level_values.pooled,
                omitted * span,
                _TAU_ATOL * y_scale,
                relative=min(
                    problem.unit * abs(float(level_values.pooled[0])), abs(reported_pooled)
                )
                / problem.unit,
            ),
        ]
        if level_values.predictive is not None:
            # A predictive density lies in [0, 1/sqrt(2 pi v_k)].
            predictive_scale = 1.0 / np.sqrt(2.0 * math.pi * problem.v)
            excesses.append(
                _excess(
                    level_values.predictive,
                    omitted * predictive_scale,
                    _TAU_ATOL * predictive_scale,
                )
            )
        error = float(np.max(excesses))
        if error <= 1.0:
            return _TauPosterior(
                theta=reported_mean,
                shrink=level_values.shrink[0],
                variance=problem.var_min * level_values.variance[0],
                pooled_mean=reported_pooled,
                predictive_density=None
                if level_values.predictive is None
                else level_values.predictive[0] / problem.unit,
            )

        target = level_values.log_z + log_eps - _LOG2
        if window.log_mass_below > target or window.log_mass_above > target:
            window, level, nodes = _widened(window, target, problem), _TAU_FIRST_LEVEL, None
            continue
        narrowed = _narrowed(nodes, window, target - _TAU_NARROWING_MARGIN, problem)
        if narrowed is not None and narrowed.hi - narrowed.lo < half:
            window, level, nodes = narrowed, _TAU_FIRST_LEVEL, None
            continue
        level *= 2


def marginalized_segment_intervals(
    est: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    tau_prior_scale: float = _TAU_PRIOR_SCALE_DEFAULT,
) -> MarginalizedSegmentIntervals:
    """Per-segment shrunken estimate, shrinkage factor, and interval.

    Replaces a plug-in ``tau^2`` with a tau-marginalised random-effects
    posterior: ``p(tau | y)`` from the REML marginal likelihood (pooled
    mean profiled out) times a ``HalfNormal(tau_prior_scale)`` prior,
    mixed against Morris's (1983) conditional posterior
    ``theta_k | y, tau ~ N(mu_hat(tau) + lambda_k(tau)(y_k - mu_hat(tau)),
    lambda_k(tau)v_k + (1-lambda_k(tau))^2 / sum_j w_j(tau))``,
    ``w_j(tau) = 1/(v_j+tau^2)``. The integral over tau covers the whole
    posterior, wherever the data place it: its support is certified by
    explicit tail bounds and its quadrature refined until every reported
    moment agrees between refinements (see :func:`_tau_posterior`).

    A plug-in ``tau^2`` collapses to exactly 0 in 45-59% of calls at a
    representative K=5 configuration, degenerating every per-segment
    interval to the pooled-mean width; marginalising removes that
    failure mode structurally (verified at every ``k`` from 2 to 100).
    ``tau_prior_scale`` (default 0.30) is verified by simulation at
    coverage 94-98% out to 2x the default; passing a smaller scale to
    "tighten" the interval reintroduces the plug-in's pathology.

    Interval form ``theta_k +/- z * sqrt(mixture variance)`` is a Wald
    summary of the tau-mixture posterior, not its quantiles; measured
    on-scale coverage is nominal (95.2% at k=5, 96.9% at k=3).

    Raises ``ValueError`` on mismatched shapes, non-1-D input, fewer
    than 2 segments, any non-finite/non-positive variance, a
    non-finite or non-positive ``tau_prior_scale``, or invalid ``alpha``;
    and ``estimation.meta.posterior_integration_unresolved`` if the
    posterior cannot be integrated to that accuracy within the numerical
    budget.
    """
    _validate_alpha(alpha)
    est_arr, var_arr = _validate_est_var(est, var)
    k = est_arr.shape[0]
    _validate_tau_prior_scale(tau_prior_scale)

    posterior = _tau_posterior(est_arr, var_arr, tau_prior_scale)
    theta = posterior.theta

    z = two_sided_critical_value(norm_dist.isf, alpha, what="marginalized segment interval")
    se = np.sqrt(posterior.variance)
    return MarginalizedSegmentIntervals(
        k=k,
        theta=theta,
        shrink_k=posterior.shrink,
        lb=theta - z * se,
        ub=theta + z * se,
        level=math.fsum((1.0, -alpha)),
        alpha=alpha,
    )


def hksj_pooled_mean(
    est: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    tau2: float,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Hartung-Knapp-Sidik-Jonkman pooled random-effects mean and interval.

    Weights ``w_k = 1/(v_k + tau2)`` (``tau2`` from :func:`cochran_q`,
    DerSimonian-Laird - Paule-Mandel would make HKSJ's inflation factor
    identically 1.0). Refined variance
    ``V = sum_k w_k(est_k - mu_hat)^2 / ((K-1) * sum_k w_k)`` (Hartung &
    Knapp 2001; IntHout et al. 2014), ``t``-distributed at ``K-1`` df.
    Deliberately no variance floor at the plug-in variance: flooring
    gives 100% coverage at K=3 by suppressing exactly the cases where
    HKSJ is narrower because the plug-in over-covers. Refuses only a
    truly degenerate pooled variance (``v_hksj`` below ``1e-8`` times
    the plug-in variance).

    Verified against a 4000-rep simulation (K=5, unequal per-segment
    SE, tau=0.10): 94.2% coverage.
    """
    _validate_alpha(alpha)
    est_arr, var_arr = _validate_est_var(est, var)
    k = est_arr.shape[0]
    if not (np.isfinite(tau2) and tau2 >= 0):
        _raise("estimation.meta.tau2_finite", tau2=tau2)

    w = 1.0 / (var_arr + tau2)
    w_sum = w.sum()
    mu_hat = float((w * est_arr).sum() / w_sum)
    v_hksj = float((w * (est_arr - mu_hat) ** 2).sum() / ((k - 1) * w_sum))
    plug_in_var = 1.0 / w_sum
    if v_hksj < 1e-8 * plug_in_var:
        _raise("estimation.meta.hksj_pooled_variance")
    se = float(np.sqrt(v_hksj))

    t_crit = two_sided_critical_value(student_t_isf, alpha, k - 1, what="HKSJ pooled mean interval")
    lower, upper = wald_bounds(mu_hat, t_crit, se, what="HKSJ pooled mean interval")
    return mu_hat, se, lower, upper


ESTIMATION_META_ALPHA_TOO_SMALL = _REFUSALS["estimation.meta.alpha_too_small"]


ESTIMATION_META_VAR_FINITE_STRICTLY = _REFUSALS["estimation.meta.var_finite_strictly"]


ESTIMATION_META_POSTERIOR_INTEGRATION_UNRESOLVED = _REFUSALS[
    "estimation.meta.posterior_integration_unresolved"
]
