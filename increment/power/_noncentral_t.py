"""Scalar power of the planning reference at a signed noncentrality: the
normal limit, or the noncentral-t tails of the finite-dof reference.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.special import betainc as _betainc
from scipy.special import betaincc as _betaincc
from scipy.special import gammainc as _gammainc
from scipy.special import log_ndtr as _log_ndtr
from scipy.special import ndtr as _ndtr
from scipy.special import ndtri as _ndtri

from increment._literals import Alternative
from increment.estimation._tails import student_t_isf, tail_isf

_LOG_FLOAT_MAX = math.log(sys.float_info.max)

# Noncentral-t tails use ``T = (Z + nc) / W``, ``Z ~ N(0, 1)``, ``W = sqrt(chi2_dof / dof)``.
# The upper tail ``E[Phi(nc - c W)]`` is integrated over ``u = log W`` with density
# ``exp(K - a (expm1(2u) - 2u))``; density and integrand are log-concave for ``c >= 0``,
# enabling certified quadrature.

# Phi(-40) < 4e-350: beyond it Phi rounds to zero or one.
_NORMAL_SATURATION = 40.0
_LOG_NORMAL_SATURATION_TAIL = float(_log_ndtr(-_NORMAL_SATURATION))
_LOG_TWO = math.log(2.0)
_LOG_PI = math.log(math.pi)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_INV_SQRT_TWO = 1.0 / math.sqrt(2.0)
# A minority tail at most 2**-54 of the majority cannot move their rounded sum.
_LOG_NEGLIGIBLE_SHARE = -54.0 * math.log(2.0)
# Below it, P(a, x) = x**a / Gamma(a + 1) * (1 - x a / (a + 1)) to O(x**2).
_LOG_GAMMA_SERIES_ONSET = -40.0 * math.log(2.0)
# Relative approximation target; special-function rounding is qualified separately.
_SHORTCUT_TOLERANCE = 2.0**-53
# Each omitted side of a log-concave window holds at most e**-40 / (1 - e**-40)
# of the retained mass on that side.
_WINDOW_DROP = 40.0
# Panel criteria: max|l'| h <= 8 e**(m/33) and max|l''| h**2 <= 8 e**(m/16),
# m the panel's drop below the reference. They hold the 16-point
# Gauss-Legendre remainder of local exponential and Gaussian factors under
# 2**-60 e**m of the panel's mass.
_PANEL_SLOPE = 8.0
_PANEL_CURVATURE = 8.0
# Phi(9) > 1 - 2e-19: a panel may span the saturated normal argument freely;
# below it, resolve the nonlinear transition in spans of at most three.
_SATURATED_ARGUMENT = 9.0
_ARGUMENT_SPAN = 3.0
# Normal arguments beyond it lie far outside every window; excluding them keeps
# every square finite.
_ARGUMENT_LIMIT = 2.0**500
_PANEL_LIMIT = 256
_HALVING_LIMIT = 1100
_REFERENCE_ITERATIONS = 200
# The window and panel criteria hold about any reference; half a curvature
# standard deviation off the maximum only lowers its log value by about 1/8.
_REFERENCE_PRECISION = 0.5
_SERIES_TERM_LIMIT = 100_000
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_GL_NODES.setflags(write=False)
_GL_WEIGHTS.setflags(write=False)
# Taylor coefficients 1/k!, k = 2..17, of expm1(x) - x; the next term is below
# 1e-20 of the sum for |x| < 1/2.
_EXCESS_SERIES = np.array([1.0 / math.factorial(k) for k in range(2, 18)])
_EXCESS_SERIES.setflags(write=False)
_EXCESS_HORNER = tuple(1.0 / math.factorial(k) for k in range(17, 1, -1))
# Bernoulli coefficients B_2k / (2k (2k - 1)) of the Stirling series, k = 6..1.
_STIRLING_HORNER = (
    -691.0 / 360360.0,
    1.0 / 1188.0,
    -1.0 / 1680.0,
    1.0 / 1260.0,
    -1.0 / 360.0,
    1.0 / 12.0,
)


def _expm1_excess(x: float) -> float:
    """``expm1(x) - x``, by its Taylor series where the difference cancels."""
    if abs(x) >= 0.5:
        return math.expm1(x) - x
    total = 0.0
    for coefficient in _EXCESS_HORNER:
        total = total * x + coefficient
    return total * x * x


def _expm1_excess_array(x: np.ndarray, growth: np.ndarray) -> np.ndarray:
    """``expm1(x) - x`` given ``growth = expm1(x)``, by series where it cancels."""
    excess = growth - x
    small = np.abs(x) < 0.5
    if small.any():
        part = x[small]
        excess[small] = part * part * (np.vander(part, 16, increasing=True) @ _EXCESS_SERIES)
    return excess


def _stirling_remainder(a: float) -> float:
    """``lgamma(a) - (a - 1/2) log(a) + a - log(2 pi) / 2``.

    From ``a = 10`` the six-term Bernoulli series (remainder below
    ``1 / (156 a**13)``) avoids subtracting two large logarithms.
    """
    if a < 10.0:
        return math.lgamma(a) - (a - 0.5) * math.log(a) + a - _HALF_LOG_TWO_PI
    r = 1.0 / a
    r2 = r * r
    total = 0.0
    for coefficient in _STIRLING_HORNER:
        total = total * r2 + coefficient
    return total * r


def _log_add(x: float, y: float) -> float:
    high = max(x, y)
    return high + math.log1p(math.exp(min(x, y) - high))


def _log_chi_cdf(half: float, ratio: float) -> tuple[float, float]:
    """``(log P, P)`` for ``P = P(W < ratio) = P(half, half * ratio**2)``, ``ratio > 0``.

    From ``ratio = 2**30`` (with ``half >= 2**-50``) the Chernoff bound
    ``exp(-half * (ratio**2 - 1 - 2 log ratio))`` puts the complement below
    ``e**-1000``. A tiny gamma argument keeps the leading series term; a lower
    tail the backend underflows is the logarithm of its Kummer series.
    """
    if ratio >= 2.0**30 and half >= 2.0**-50:
        return 0.0, 1.0
    log_x = math.log(half) + 2.0 * math.log(ratio)
    if log_x < _LOG_GAMMA_SERIES_ONSET:
        x = math.exp(log_x)
        log_p = half * log_x - math.lgamma(half + 1.0) + math.log1p(-x * half / (half + 1.0))
        return log_p, math.exp(log_p)
    x = math.exp(log_x) if log_x < _LOG_FLOAT_MAX else math.inf
    p = float(_gammainc(half, x))
    if p >= sys.float_info.min:
        return math.log(p), p
    term = total = 1.0
    for k in range(1, _SERIES_TERM_LIMIT):
        term *= x / (half + k)
        total += term
        if term <= 2.0**-60 * total:
            log_p = half * log_x - x - math.lgamma(half + 1.0) + math.log(total)
            return log_p, math.exp(log_p)
    return math.nan, math.nan


def _noise_free_tail(crit: float, dof: float, nc: float) -> float | None:
    """``G(nc / crit)`` when its analytic relative error meets the shortcut target.

    For ``crit > 0`` and ``nc > B = 40``, ``S = E[H(nc + Z)]`` with
    ``H(x) = G(x / crit)``. Truncating ``Z`` symmetrically at ``+-B`` cancels
    the linear term, so ``|S - H(nc)| <= M2 / 2 + 2 Phi(-B)`` with ``M2`` the
    largest ``|H''|`` on ``[nc - B, nc + B]``. Two analytic bounds on
    ``M2 / H(nc)``, the smaller used: ``H' / H <= dof / x`` gives
    ``((nc + B) / nc)**dof * (dof |dof - 1| / (nc - B)**2 + (dof / crit)**2)``;
    once ``w = (nc - B) / crit >= 1``, ``w f_W(w)`` decreases, giving
    ``w f_W(w) (dof + |dof - 1|) / (crit**2 H(nc))``. ``None`` unless the
    relative bound meets ``2**-53``; the float value of ``G`` itself is
    the gamma backend's, not a directed-rounding enclosure.
    """
    half = 0.5 * dof
    log_g, g = _log_chi_cdf(half, nc / crit)
    log_share = _LOG_TWO + _LOG_NORMAL_SATURATION_TAIL - log_g
    if not log_share <= 0.0:
        return None
    low = nc - _NORMAL_SATURATION
    growth = dof * math.log1p(_NORMAL_SATURATION / nc)
    by_ratio = math.inf
    if growth < 700.0:
        slope = dof / crit
        by_ratio = math.exp(growth) * (dof * abs(dof - 1.0) / low / low + slope * slope)
    by_density = math.inf
    w_low = low / crit
    if w_low >= 1.0:
        two_log_w = 2.0 * math.log(w_low)
        if two_log_w > 700.0:
            by_density = 0.0
        else:
            exponent = (
                0.5 * (math.log(dof) - _LOG_PI)
                - _stirling_remainder(half)
                - half * _expm1_excess(two_log_w)
                + math.log(dof + abs(dof - 1.0))
                - 2.0 * math.log(crit)
                - log_g
            )
            by_density = math.exp(exponent) if exponent < 700.0 else math.inf
    bound = 0.5 * min(by_ratio, by_density) + math.exp(log_share)
    return g if bound <= _SHORTCUT_TOLERANCE / (1.0 + _SHORTCUT_TOLERANCE) else None


class _IntegrandPoint(NamedTuple):
    """The log integrand at one ``u = log W``, to panel-search accuracy."""

    log_value: float  # log density + log Phi(argument), without K
    slope: float  # l'(u)
    bend: float  # -l''(u)
    log_density: float  # without K
    density_slope: float
    log_cdf: float
    argument: float  # nc - crit W


def _normal_log_cdf_terms(y: float) -> tuple[float, float, float]:
    """``(log Phi(y), M, y + M)``, ``M = phi(y) / Phi(y)``, to panel-search accuracy."""
    if y >= -30.0:
        log_cdf = math.log(0.5 * math.erfc(-y * _INV_SQRT_TWO))
        mills = math.exp(-0.5 * y * y - _HALF_LOG_TWO_PI - log_cdf)
        return log_cdf, mills, y + mills
    # Mills-ratio series; its next term is below 2e-12 relative at y = -30.
    z = -y
    inverse = 1.0 / (z * z)
    tail = inverse * (1.0 - inverse * (3.0 - inverse * (15.0 - 105.0 * inverse)))
    series = 1.0 - tail
    log_cdf = -0.5 * z * z - math.log(z) - _HALF_LOG_TWO_PI + math.log(series)
    return log_cdf, z / series, z * tail / series


def _integrand_point(
    dof: float, sign: float, u: float, s: float, y: float
) -> _IntegrandPoint | None:
    """Terms at ``u`` with ``s = |crit| W`` and ``y = nc - crit W``; ``None`` beyond range."""
    two_u = 2.0 * u
    if not (two_u <= 700.0 and abs(y) <= _ARGUMENT_LIMIT and s < math.inf):
        return None
    growth = math.expm1(two_u)
    log_density = -0.5 * dof * (growth - two_u)
    density_slope = -dof * growth
    log_cdf, mills, excess = _normal_log_cdf_terms(y)
    pull = s * mills
    return _IntegrandPoint(
        log_value=log_density + log_cdf,
        slope=density_slope - sign * pull,
        bend=2.0 * dof * (growth + 1.0) + sign * pull + pull * s * excess,
        log_density=log_density,
        density_slope=density_slope,
        log_cdf=log_cdf,
        argument=y,
    )


def _point_at_log_s(
    dof: float, sign: float, log_abs_crit: float, nc: float, sigma: float
) -> _IntegrandPoint | None:
    if not sigma <= 700.0:
        return None
    s = math.exp(sigma)
    return _integrand_point(dof, sign, sigma - log_abs_crit, s, nc - sign * s)


def _reference_point(
    dof: float, sign: float, log_abs_crit: float, nc: float
) -> tuple[float, _IntegrandPoint | None]:
    """``(log s, terms there)``, ``s = |crit| W``, near the log integrand's maximum.

    The slope ``dof (1 - W**2) - sign(crit) s M`` is bracketed through
    ``M(y) <= max(0, -y) + 1``: for ``crit > 0`` it is positive once
    ``s (1 + d + s) <= dof / 2`` and ``W <= 1/2`` (``d = max(0, -nc)``) and
    negative at ``W = 1``; for ``crit < 0`` it is positive at ``W = 1`` and
    negative at twice the root of ``dof (W**2 - 1) = s (1 + d)``. Newton steps
    stay inside the bracket and stop at the evaluated point once the next step
    is within ``_REFERENCE_PRECISION`` curvature standard deviations, so its
    terms are the reference's. The point only centres the panels; its
    precision affects cost, not the value.
    """
    log_shortfall = math.log1p(max(0.0, -nc))
    log_dof = math.log(dof)
    if sign > 0.0:
        log_root = 0.5 * _log_add(2.0 * log_shortfall, _LOG_TWO + log_dof)
        low = min(log_abs_crit - _LOG_TWO, log_dof - _log_add(log_shortfall, log_root))
        high = log_abs_crit
        sigma = min(high, max(low, math.log(nc))) if nc > 0.0 else low
    else:
        log_pull = log_shortfall + log_abs_crit
        log_root = 0.5 * _log_add(2.0 * log_pull, 2.0 * (_LOG_TWO + log_dof))
        low = log_abs_crit
        high = log_abs_crit + _log_add(log_pull, log_root) - log_dof
        sigma = low
    for _ in range(_REFERENCE_ITERATIONS):
        point = _point_at_log_s(dof, sign, log_abs_crit, nc, sigma)
        if point is None or point.slope < 0.0:
            high = sigma
        elif point.slope > 0.0:
            low = sigma
        else:
            return sigma, point
        if point is not None and point.bend > 0.0:
            step = point.slope / point.bend
            if low < sigma + step < high:
                if abs(step) * math.sqrt(point.bend) <= _REFERENCE_PRECISION:
                    return sigma, point
                sigma += step
                continue
        sigma = 0.5 * (low + high)
        if not low < sigma < high:
            break
    return sigma, _point_at_log_s(dof, sign, log_abs_crit, nc, sigma)


@dataclass(frozen=True, slots=True)
class _TailIntegrand:
    """``exp(l(u))`` about a reference ``u_ref``, evaluated at offsets ``v = u - u_ref``.

    Offsets keep a transition of width ``1 / nc`` resolved even where
    ``W`` itself is tiny: ``y = y_ref - sign s_ref expm1(v)``.
    """

    dof: float
    sign: float
    u_ref: float
    s_ref: float
    y_ref: float

    def point(self, v: float) -> _IntegrandPoint | None:
        if not v <= 350.0:
            return None
        return _integrand_point(
            self.dof,
            self.sign,
            self.u_ref + v,
            self.s_ref * math.exp(v),
            self.y_ref - self.sign * self.s_ref * math.expm1(v),
        )

    def argument_step(self, v: float, argument: float, direction: float) -> float:
        """Widest step keeping a panel within ``_ARGUMENT_SPAN`` of the normal
        argument below saturation, and stopping at saturation when entering
        the transition from far above it."""
        if self.sign * direction > 0.0:
            target = (
                _SATURATED_ARGUMENT
                if argument > _SATURATED_ARGUMENT + _ARGUMENT_SPAN
                else argument - _ARGUMENT_SPAN
            )
        elif argument >= _SATURATED_ARGUMENT:
            return math.inf
        else:
            target = argument + _ARGUMENT_SPAN
        if self.s_ref == 0.0:
            return math.inf
        ratio = self.sign * (self.y_ref - target) / self.s_ref
        if not ratio > -1.0:
            return math.inf
        step = (math.log1p(ratio) - v) * direction
        return step if step > 0.0 else math.inf


def _panel_width(slope: float, bend: float, drop: float, growth: float = 0.0) -> float:
    """Widest panel the resolution criteria allow from one end's derivatives,
    with ``|l'|`` growing across it by ``growth`` per unit width."""
    relax = min(drop, 1000.0)
    budget = _PANEL_SLOPE * math.exp(relax / 33.0)
    width = math.inf
    if growth > 0.0:
        # The positive root of (slope + growth h) h = budget, without cancellation.
        reach = math.hypot(slope, 2.0 * math.sqrt(growth) * math.sqrt(budget))
        width = 2.0 * budget / (slope + reach)
    elif slope > 0.0:
        width = budget / slope
    if bend > 0.0:
        width = min(width, math.sqrt(_PANEL_CURVATURE * math.exp(relax / 16.0) / bend))
    return width


@dataclass(frozen=True, slots=True)
class _WindowEnds:
    """Stopping tests for the right and left panel marches.

    ``crit > 0``: the log-concave integrand has dropped ``_WINDOW_DROP``
    below the reference (``floor``). ``crit < 0``: the log-concave density's
    tail bound ``f(u) / |l_f'(u)|`` (times ``Phi(y(u))`` on the left, where
    ``Phi`` only decreases) is below ``floor``, ``e**-_WINDOW_DROP`` of the
    lower bound ``h Phi(y_ref) min(f(0), f(h))`` on the retained mass.
    """

    concave: bool
    floor: float

    def reached(self, point: _IntegrandPoint, direction: float) -> bool:
        if self.concave:
            return point.log_value <= self.floor
        if direction > 0.0:
            return (
                point.density_slope < 0.0
                and point.log_density - math.log(-point.density_slope) <= self.floor
            )
        return (
            point.density_slope > 0.0
            and point.log_cdf + point.log_density - math.log(point.density_slope) <= self.floor
        )


def _window_ends(
    integrand: _TailIntegrand, reference: _IntegrandPoint, widest: float
) -> _WindowEnds | None:
    if integrand.sign > 0.0:
        return _WindowEnds(concave=True, floor=reference.log_value - _WINDOW_DROP)
    width = min(widest, _panel_width(abs(reference.slope), abs(reference.bend), 0.0))
    probe = integrand.point(width)
    for _ in range(_HALVING_LIMIT):
        if probe is not None or not width > 0.0:
            break
        width *= 0.5
        probe = integrand.point(width)
    if probe is None or not width > 0.0:
        return None
    floor = (
        math.log(width)
        + reference.log_cdf
        + min(reference.log_density, probe.log_density)
        - _WINDOW_DROP
    )
    return _WindowEnds(concave=False, floor=floor)


def _panel_edges(
    integrand: _TailIntegrand,
    reference: _IntegrandPoint,
    direction: float,
    ends: _WindowEnds,
    widest: float,
) -> list[float] | None:
    """Panel boundaries from the reference outward, each panel narrowed until resolved.

    For ``crit > 0`` the curvature ``-l'' = 2 dof W**2 + s M + s**2 M (y + M)``
    is positive and nondecreasing in ``u``: ``s`` rises and ``y`` falls, and
    both ``M`` and ``M (y + M) = 1 - Var(Z | Z < y)`` fall with ``y``. So a
    panel's ``|l'|`` and ``-l''`` peak at its ends, and ``|l'|`` grows across
    it by at most its largest ``-l''`` times its width. The first proposal
    takes that growth at the inner end's curvature, the largest toward
    smaller ``u``; a rejected candidate's derivatives bound those of every
    narrower panel, whose width they then set. Otherwise a rejected candidate
    is halved. For ``crit < 0``, endpoint derivatives only estimate the panel
    maxima; that resolution criterion is qualified rather than an enclosure.
    """
    concave = ends.concave
    edges: list[float] = []
    offset, inner = 0.0, reference
    for _ in range(_PANEL_LIMIT):
        slope, bend = abs(inner.slope), abs(inner.bend)
        width = min(
            widest,
            _panel_width(
                slope, bend, reference.log_value - inner.log_value, bend if concave else 0.0
            ),
            integrand.argument_step(offset, inner.argument, direction),
        )
        outer = None
        for _ in range(_HALVING_LIMIT):
            candidate = integrand.point(offset + direction * width)
            if candidate is not None:
                relax = min(reference.log_value - max(inner.log_value, candidate.log_value), 1000.0)
                max_slope = max(slope, abs(candidate.slope))
                max_bend = max(bend, abs(candidate.bend))
                slope_budget = _PANEL_SLOPE * math.exp(relax / 33.0)
                bend_budget = _PANEL_CURVATURE * math.exp(relax / 16.0)
                if max_slope * width <= slope_budget and max_bend * width * width <= bend_budget:
                    outer = candidate
                    break
                if concave:
                    narrower = min(
                        slope_budget / max_slope if max_slope > 0.0 else math.inf,
                        math.sqrt(bend_budget / max_bend) if max_bend > 0.0 else math.inf,
                    )
                    if 0.0 < narrower < width:
                        width = narrower
                        continue
            width *= 0.5
        if outer is None:
            return None
        offset += direction * width
        edges.append(offset)
        if ends.reached(outer, direction):
            return edges
        inner = outer
    return None


def _log_panel_sum(integrand: _TailIntegrand, edges: np.ndarray) -> float:
    """``log`` of the composite Gauss-Legendre sum of ``f(u_ref + v) Phi(y) / f(u_ref)``,
    ``f`` the density of ``u``."""
    low, high = edges[:-1], edges[1:]
    centre = 0.5 * (low + high)
    radius = 0.5 * (high - low)
    nodes = (centre[:, None] + radius[:, None] * _GL_NODES).ravel()
    weights = (radius[:, None] * _GL_WEIGHTS).ravel()
    doubled = 2.0 * nodes
    growth = np.expm1(doubled)
    # E(2u_ref + 2v) - E(2u_ref) = E(2v) + expm1(2 u_ref) expm1(2v), E = expm1 - id.
    shift = (-0.5 * integrand.dof) * (
        _expm1_excess_array(doubled, growth) + math.expm1(2.0 * integrand.u_ref) * growth
    )
    arguments = integrand.y_ref - (integrand.sign * integrand.s_ref) * np.expm1(nodes)
    shift += _log_ndtr(arguments)
    top = float(shift.max())
    return top + math.log(float(weights @ np.exp(shift - top)))


def _tail_integral(crit: float, dof: float, nc: float) -> float:
    """``P(T > crit)`` as the positive integral of ``exp(l(u))``; NaN if unresolved.

    Composite 16-point Gauss-Legendre panels march out from a point near the
    maximum (``_panel_edges``) until ``_window_ends`` bounds each omitted
    side below ``e**-40`` of the retained mass; one vectorized pass then sums
    the panels in log space. The truncation bound is analytic; the panel
    criteria are a qualified remainder estimate, not an enclosure. Binary64
    log-space evaluation leaves a relative error of about
    ``eps (|log S| + 10)``.
    """
    sign = 1.0 if crit > 0.0 else -1.0
    log_abs_crit = math.log(abs(crit))
    sigma, reference = _reference_point(dof, sign, log_abs_crit, nc)
    if reference is None:
        return math.nan
    s_ref = math.exp(sigma)
    integrand = _TailIntegrand(
        dof=dof, sign=sign, u_ref=sigma - log_abs_crit, s_ref=s_ref, y_ref=nc - sign * s_ref
    )
    widest = 64.0 / min(dof, math.sqrt(dof))
    ends = _window_ends(integrand, reference, widest)
    if ends is None:
        return math.nan
    right = _panel_edges(integrand, reference, 1.0, ends, widest)
    left = _panel_edges(integrand, reference, -1.0, ends, widest)
    if right is None or left is None:
        return math.nan
    half = 0.5 * dof
    log_value = (
        0.5 * (math.log(dof) - _LOG_PI)
        - _stirling_remainder(half)
        - half * _expm1_excess(2.0 * integrand.u_ref)
        + _log_panel_sum(integrand, np.array([*reversed(left), 0.0, *right]))
    )
    # A probability; rounding alone can carry the log above zero.
    return 1.0 if log_value >= 0.0 else math.exp(log_value)


def _negative_crit_tail_rounds_to_zero(crit: float, dof: float, nc: float) -> bool:
    """For ``crit < 0``: ``S <= Phi(nc / 2) + P(W > -nc / (2 |crit|))``, and
    both below ``2**-1076`` round ``S`` to zero (Chernoff for the second)."""
    if nc > -2.0 * _NORMAL_SATURATION:
        return False
    ratio = nc / (2.0 * crit)
    if not ratio > 1.0:
        return False
    two_log_ratio = 2.0 * math.log(ratio)
    return two_log_ratio > 700.0 or 0.5 * dof * _expm1_excess(two_log_ratio) >= 1076.0 * _LOG_TWO


def _noncentral_t_tail(crit: float, dof: float, nc: float) -> float:
    """``P(T > crit)``, ``T = (Z + nc) / W``, ``W = sqrt(chi2_dof / dof)``; NaN if unresolved.

    The exact limits ``crit = 0`` (``Phi(nc)``) and the saturated normal
    (``S <= Phi(nc)`` for ``crit > 0``, ``S >= Phi(nc)`` for ``crit < 0``)
    round exactly; a far positive ``nc`` takes the noise-free limit only when
    its relative error bound meets the shortcut target; every other case is the positive integral.
    """
    if crit == 0.0:
        if abs(nc) >= _NORMAL_SATURATION:
            return 1.0 if nc > 0.0 else 0.0
        return float(_ndtr(nc))
    if crit > 0.0:
        if nc <= -_NORMAL_SATURATION:
            return 0.0
        if nc > _NORMAL_SATURATION:
            limit = _noise_free_tail(crit, dof, nc)
            if limit is not None:
                return limit
    elif nc >= _NORMAL_SATURATION:
        return 1.0
    elif _negative_crit_tail_rounds_to_zero(crit, dof, nc):
        return 0.0
    return _tail_integral(crit, dof, nc)


def _minority_tail_is_negligible(distance: float, majority: float) -> bool:
    """For ``crit >= 0`` the minority tail is at most ``Phi(-distance)``."""
    if distance >= _NORMAL_SATURATION:
        return True
    if not majority > 0.0:
        return False
    return float(_log_ndtr(-distance)) <= math.log(majority) + _LOG_NEGLIGIBLE_SHARE


# Gil, Segura and Temme (2023, eqs. 1.1, 1.5, 1.6) give positive incomplete-beta series for
# ``c > 0`` and ``nc > 0``: ``P(T > c) = (1/2) sum_j (p_j I_y(a,j+1/2) + q_j I_y(a,j+1))``
# and ``P(|T| > c) = sum_j p_j I_y(a,j+1/2)``, so every term is positive and two-sided tails exact.
_SERIES_TERMS = (16, 32, 64, 128)
_SERIES_SHAPES = np.arange(_SERIES_TERMS[-1] + 1.0) + np.array([[0.5], [1.0]])
_SERIES_SHAPES.setflags(write=False)
_SERIES_DIVISORS = _SERIES_SHAPES[:, :-1] + 0.5
_SERIES_DIVISORS.setflags(write=False)


def _positive_series(crit: float, dof: float, distance: float, *, two_sided: bool) -> float | None:
    """``P(|T| > crit)`` if ``two_sided``, else ``P(T > crit)``, at ``nc =
    distance`` from the series above; ``None`` unless ``crit > 0``,
    ``distance > 0`` and a positive expansion meets the relative truncation target.

    For the survival series, at the first omitted index ``n`` the weight ratio is
    at most ``r = lam / (b_n + 1/2)``. The bound ``I_y(a,b+1)/I_y(a,b) <= (a+b)/b``
    gives term ratio ``rho = r (a + b_n) / b_n``; both fall with the index,
    so a row omits at most ``min(w_n / (1 - r), w_n I_n / (1 - rho))``. That
    bound plus the smallest normal float must stay at most ``2**-53`` times
    the retained sum. The weight rows have total mass at most one, and the
    added float covers underflow. Since ``I_y(a, b)`` rises with ``b``, a
    row's bound is at least ``w_n`` times its retained sum: survival counts
    whose first omitted weights all exceed ``2**-53`` are skipped without
    evaluating beta values. Each ``I`` is the backend's direct ``betaincc`` or
    ``betainc`` at the smaller of ``c**2 / (dof + c**2)`` and ``y``, formed
    from a normal square; a nonfinite sum is declined, never read as zero
    power.

    For ``distance > crit`` the two-sided interior mass is ``sum p_j I_x(b_j, a)``;
    the one-sided CDF adds ``Phi(-distance)`` to
    ``sum(p_j I_x(b_j, a) + q_j I_x(b_j + 1/2, a)) / 2``.
    These beta terms decrease, with ratio at most ``x (a + b) / b``. Only the
    first omitted term is evaluated until the bound could meet tolerance.
    Subtraction requires computed power at least one half; the error is
    compared with its lower bound, ``power - error``. An underflowed omitted
    beta is covered by its upper bound on all later betas. Floating-point
    evaluation is qualified, not a directed-rounding enclosure.
    """
    lam = 0.5 * distance * distance
    if not (distance > 0.0 and 0.0 < lam < _SERIES_TERMS[-1] and crit > 0.0):
        return None
    ratio, half = crit / math.sqrt(dof), 0.5 * dof
    lower = ratio <= 1.0
    small = ratio if lower else 1.0 / ratio
    square = small * small
    if not (square >= sys.float_info.min and half > 0.0):
        return None
    coordinate = square / (1.0 + square)
    complement = distance > crit
    x = coordinate if lower else 1.0 / (1.0 + square)
    rows, scale = (1, 1.0) if two_sided else (2, 0.5)
    shapes = _SERIES_SHAPES[:rows]
    weights = np.empty((rows, _SERIES_TERMS[-1] + 1))
    weights[:, 0] = math.exp(-lam)
    if not two_sided:
        weights[1, 0] *= math.sqrt(4.0 * lam / math.pi)
    np.divide(lam, _SERIES_DIVISORS[:rows], out=weights[:, 1:])
    np.cumprod(weights, axis=1, out=weights)
    betas = np.empty_like(weights)
    beta = _betainc if lower == complement else _betaincc
    evaluated = 0
    for n in _SERIES_TERMS:
        if not complement and (lam >= n or min(weights[:, n].tolist()) > _SHORTCUT_TOLERANCE):
            continue
        span = slice(n if complement else evaluated, n + 1)
        if lower:
            beta(shapes[:, span], half, coordinate, out=betas[:, span])
        else:
            beta(half, shapes[:, span], coordinate, out=betas[:, span])
        evaluated = evaluated if complement else n + 1
        omitted = 0.0
        for row in range(rows):
            shape = float(shapes[row, n])
            weight = float(weights[row, n])
            weight_ratio = lam / (shape + 0.5)
            first = weight * float(betas[row, n])
            bound = (
                (first if complement else weight) / (1.0 - weight_ratio)
                if weight_ratio < 1.0
                else math.inf
            )
            term_ratio = weight_ratio * (half + shape) / shape
            if complement:
                term_ratio *= x
            if term_ratio < 1.0:
                bound = min(bound, first / (1.0 - term_ratio))
            omitted += bound
        error = scale * omitted + sys.float_info.min
        # Power cannot exceed one, so this count cannot meet relative tolerance.
        if not error <= _SHORTCUT_TOLERANCE:
            continue
        if complement:
            span = slice(evaluated, n)
            if lower:
                beta(shapes[:, span], half, coordinate, out=betas[:, span])
            else:
                beta(half, shapes[:, span], coordinate, out=betas[:, span])
            evaluated = n + 1
        total = scale * float(np.vdot(weights[:, :n], betas[:, :n]))
        if complement:
            if not two_sided:
                total += float(_ndtr(-distance))
            total = 1.0 - total
            if not 0.5 <= total <= 1.0:
                return None
        retained = total - error if complement else total
        if 0.0 < total < math.inf and error <= _SHORTCUT_TOLERANCE * retained:
            # A probability; rounding alone can carry the sum above one.
            return min(total, 1.0)
    return None


def _scalar_power_from_nc(
    nc: float,
    *,
    alternative: Alternative,
    tail_alpha: float,
    dof: float | None,
) -> float:
    """Scalar power kernel; a tail it cannot resolve stays NaN.

    The Student-t reference reads every direction from the upper tail:
    greater at ``nc``, less at ``-nc`` (``T(dof, nc) = -T(dof, -nc)``),
    two-sided their sum. A favorable direction and two-sided power first try
    ``_positive_series``, whose two-sided sum holds both tails; otherwise
    ``_noncentral_t_tail`` answers, with the minority tail dropped only when
    ``Phi(-|nc|)`` bounds it below ``2**-54`` of the majority. For
    ``crit < 0`` the two-sided regions cover the line and the sum is capped
    at one. Callers distinguish an unresolved tail from certified zero power.
    """
    if nc == 0.0:
        # The null rejection probability is the compiled tail allocation.
        return tail_alpha * (2.0 if alternative == "two-sided" else 1.0)
    if math.isinf(nc):
        if alternative == "two-sided":
            return 1.0
        return 1.0 if (nc > 0.0) == (alternative == "greater") else 0.0
    if dof is None or math.isinf(dof):
        z_alpha = -float(_ndtri(tail_alpha))
        if alternative == "two-sided":
            return float(_ndtr(nc - z_alpha) + _ndtr(-z_alpha - nc))
        if alternative == "greater":
            return float(_ndtr(nc - z_alpha))
        return float(_ndtr(-z_alpha - nc))

    crit = tail_isf(student_t_isf, tail_alpha, dof, what="power noncentral-t critical value")
    if alternative != "two-sided":
        upper = nc if alternative == "greater" else -nc
        series = _positive_series(crit, dof, upper, two_sided=False)
        return _noncentral_t_tail(crit, dof, upper) if series is None else series
    distance = abs(nc)
    series = _positive_series(crit, dof, distance, two_sided=True)
    if series is not None:
        return series
    majority = _noncentral_t_tail(crit, dof, distance)
    if crit >= 0.0 and _minority_tail_is_negligible(distance, majority):
        return majority
    total = majority + _noncentral_t_tail(crit, dof, -distance)
    return 1.0 if total > 1.0 else total
