"""Boundary-crossing power for sequential designs.

Under a true log-lift ``delta`` with full-information standard error
``se_full``, the z-statistic at information fraction ``t_k`` has mean
``drift_z * sqrt(t_k)`` where ``drift_z = delta / se_full``, so the score
``S_k = Z_k * sqrt(t_k)`` gains independent ``N(drift_z * dt, dt)``
increments. Power is the probability the path exits a KNOWN boundary at
some look.

The recursion (Armitage-McPherson-Rowe / Jennison-Turnbull) integrates the
surviving sub-density on exactly ``[-b_k, b_k]`` with composite
Gauss-Legendre panels and takes every exit tail from the normal survival
function, so no finite-domain truncation exists. Its quadrature enclosure uses
the classical Gauss-Legendre remainder theorem on each panel. Derivatives of
the surviving density are bounded by derivatives of the most recent Gaussian
increment; Leibniz' rule then bounds each propagated density and exit integral.
The bound therefore covers every look rather than inferring the remaining
error from observed refinement differences.

Rounding and normal-tail allowances are added separately. Tiny probabilities
keep relative accuracy: no fixed absolute probability tolerance erases them.
Cheap bounds without quadrature (the first-look exit from below, the union of
marginal exceedances from above) remain valid pre-checks. An evaluation that
cannot meet the quadrature work ceiling is unresolved and no public point
estimate or inverse decision consumes it.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.special import ndtr as _ndtr
from scipy.special import ndtri as _ndtri

from increment.errors import InvalidRequestError, RefusalSpec, raiser, refusals
from increment.estimation.asymptotic_mean import mixture_r_star
from increment.estimation.sequential import (
    GaussianScoreMixture,
    _validate_information_fractions,
)
from increment.power._search import (
    _MDE_NUMERICAL_RESOLUTION,
    _MdeRefusal,
    _noncentrality,
    _se_from_log,
    float_from_ordinal,
    float_ordinal,
)
from increment.power._validation import _validate_planned_looks

if TYPE_CHECKING:
    from increment.power.core import _MdeSearch

DEFAULT_PLANNED_LOOKS = 14
PANEL_NODES = 8
NODES_MIN = 16
NODES_MAX = 4096
MAX_WALK_WORK = 300_000_000
MAX_WALK_LOOKS = 256
MATRIX_ROWS = 256
C_FP = 4.0
# cephes ndtr/erfc documented peak relative error 5.7e-14, rounded up.
SF_REL = 1e-13
# sup_A |E[1_A Z]| = phi(0); retained as a diagnostic slope clamp.
SLOPE_CAP = 1.0 / math.sqrt(2.0 * math.pi)
_EPS = 2.0**-53
_SUBNORMAL = 2.0**-1022
_ROOT_2PI = math.sqrt(2.0 * math.pi)
_LOG_FLOAT_MAX = math.log(float.fromhex("0x1.fffffffffffffp+1023"))

ExitSide = Literal["both", "upper"]
SequentialPlanningSpec = GaussianScoreMixture


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "power.nodes_multiple": "nodes must be a multiple of {panel_nodes}, got {nodes}",
        "power.alpha_finite": "alpha must be finite and in (0, 1) (got {alpha!r})",
        "power.se_full_finite": "se_full must be finite and > 0 (got {se_full!r})",
        "power.boundary_crossing_quadrature": "boundary-crossing quadrature did not converge: {reason}",
        "power.sequential_mde.design_search_minimum": "sequential design cannot search for a minimum detectable effect when target_power={target} is at or below this boundary's own null-crossing probability ({estimate:.6g}) -- the zero-distance effect already meets it, so no positive relative lift is more detectable than the null itself",
    },
)
_raise = raiser(_REFUSALS)


@dataclass(frozen=True, slots=True)
class _PlannedLooks:
    """A sequential boundary family with its resolved information fractions."""

    spec: SequentialPlanningSpec
    fractions: tuple[float, ...]


def _require_prospective(spec: object) -> SequentialPlanningSpec:
    """Brownian planning kernels cannot size a registered likelihood process."""
    if not isinstance(spec, GaussianScoreMixture):
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "registered sequential power requires a matching count/model calculation; Gaussian score approximations do not size the deployed process",
        )
    return spec


def _resolve_looks(spec: SequentialPlanningSpec, planned_looks: int | None) -> _PlannedLooks:
    """Resolve the look schedule once, before any numerical work: equal
    batches at ``planned_looks`` (default 14)."""
    if planned_looks is not None:
        _validate_planned_looks(planned_looks)
    _require_prospective(spec)
    looks = DEFAULT_PLANNED_LOOKS if planned_looks is None else planned_looks
    fractions = tuple(k / looks for k in range(1, looks + 1))
    return _PlannedLooks(spec, fractions)


def _sf(z: np.ndarray | float) -> np.ndarray | float:
    return _ndtr(-z)


def _pdf(z: np.ndarray | float) -> np.ndarray | float:
    """Standard-normal density without overflowing ``z * z`` in the tails."""
    magnitude = np.abs(z)
    safe = np.minimum(magnitude, 40.0)
    value = np.exp(-0.5 * safe * safe) / _ROOT_2PI
    return np.where(magnitude > 40.0, 0.0, value)


@functools.cache
def _legendre(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite eight-point Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    if nodes < PANEL_NODES or nodes % PANEL_NODES:
        _raise("power.nodes_multiple", panel_nodes=PANEL_NODES, nodes=nodes)
    local_x, local_w = np.polynomial.legendre.leggauss(PANEL_NODES)
    panels = nodes // PANEL_NODES
    width = 2.0 / panels
    centers = -1.0 + width * (np.arange(panels) + 0.5)
    x = (centers[:, None] + 0.5 * width * local_x[None, :]).reshape(-1)
    w = np.broadcast_to(0.5 * width * local_w, (panels, PANEL_NODES)).reshape(-1).copy()
    x.setflags(write=False)
    w.setflags(write=False)
    return x, w


@functools.cache
def _gaussian_derivative_log_bound(order: int) -> float:
    """Log of a Fourier-integral bound for ``sup |phi^(order)|``."""
    return 0.5 * (order - 1) * math.log(2.0) + math.lgamma(0.5 * (order + 1)) - math.log(math.pi)


def _product_derivative_log_bound(previous_sd: float, increment_sd: float, *, tail: bool) -> float:
    """Bound the 16th derivative of a surviving density times a Gaussian
    transition density or tail.

    The surviving density at a look is a sub-probability mixture of the most
    recent Gaussian increment, so its derivative bounds do not depend on an
    unobserved earlier density.
    """
    order = 2 * PANEL_NODES
    terms: list[float] = []
    for k in range(order + 1):
        left = _gaussian_derivative_log_bound(k) - (k + 1) * math.log(previous_sd)
        right_order = order - k
        if tail:
            right = (
                0.0
                if right_order == 0
                else _gaussian_derivative_log_bound(right_order - 1)
                - right_order * math.log(increment_sd)
            )
        else:
            right = _gaussian_derivative_log_bound(right_order) - (right_order + 1) * math.log(
                increment_sd
            )
        terms.append(
            math.lgamma(order + 1) - math.lgamma(k + 1) - math.lgamma(order - k + 1) + left + right
        )
    top = max(terms)
    return top + math.log(math.fsum(math.exp(term - top) for term in terms))


def _quadrature_remainder(
    interval_length: float,
    nodes: int,
    previous_sd: float,
    increment_sd: float,
    *,
    tail: bool,
) -> float:
    """Composite Gauss-Legendre remainder from the classical derivative bound."""
    panels = nodes // PANEL_NODES
    panel_width = interval_length / panels
    order = 2 * PANEL_NODES
    log_coefficient = (
        4.0 * math.lgamma(PANEL_NODES + 1) - math.log(order + 1) - 3.0 * math.lgamma(order + 1)
    )
    log_error = (
        math.log(panels)
        + (order + 1) * math.log(panel_width)
        + log_coefficient
        + _product_derivative_log_bound(previous_sd, increment_sd, tail=tail)
    )
    return math.inf if log_error >= _LOG_FLOAT_MAX else math.exp(log_error)


@dataclass(frozen=True, slots=True)
class _Walk:
    """Per-look first-exit masses and rigorous quadrature error bounds."""

    upper: tuple[float, ...]
    total: tuple[float, ...]
    upper_error: tuple[float, ...]
    total_error: tuple[float, ...]
    surviving: float
    expected_error: float
    z_max: float
    upper_score: tuple[float, ...]
    total_score: tuple[float, ...]
    score_abs: float


def _gl_walk(  # noqa: PLR0915
    bounds: Sequence[float], fractions: Sequence[float], drift_z: float, nodes: int
) -> _Walk:
    x_ref, w_ref = _legendre(nodes)
    prev_t = 0.0
    prev_b = 0.0
    prev_dt = 0.0
    density = density_error = grid = weights = None
    upper: list[float] = []
    total: list[float] = []
    upper_error: list[float] = []
    total_error: list[float] = []
    upper_score: list[float] = []
    total_score: list[float] = []
    score_abs = 0.0
    z_max = 0.0
    for c, t in zip(bounds, fractions, strict=True):
        dt = t - prev_t
        b = c * math.sqrt(t)
        sd = math.sqrt(dt)
        mean = drift_z * dt
        z_max = max(z_max, (b + prev_b + abs(mean)) / sd)
        new_grid = b * x_ref
        new_weights = b * w_ref
        if density is None:
            z_top = (b - mean) / sd
            z_bot = (b + mean) / sd
            top = float(_sf(z_top))
            bot = float(_sf(z_bot))
            new_density = _pdf((new_grid - mean) / sd) / sd
            new_density_error = np.zeros_like(new_density)
            top_error = bot_error = 0.0
            # y = x - drift * prev_t is zero at the first look.
            pdf_top = float(_pdf(z_top))
            pdf_bot = float(_pdf(z_bot))
            top_score = sd * pdf_top
            bot_score = -sd * pdf_bot
            score_abs += sd * (pdf_top + pdf_bot)
        else:
            assert density_error is not None and grid is not None and weights is not None
            weighted = weights * density
            weighted_error = weights * density_error
            density_remainder = _quadrature_remainder(
                2.0 * prev_b, nodes, math.sqrt(prev_dt), sd, tail=False
            )
            new_density = np.empty_like(new_grid)
            new_density_error = np.empty_like(new_grid)
            for start in range(0, nodes, MATRIX_ROWS):
                stop = min(nodes, start + MATRIX_ROWS)
                z = (new_grid[start:stop, None] - grid[None, :] - mean) / sd
                kernel = np.exp(-0.5 * z * z) / (sd * _ROOT_2PI)
                new_density[start:stop] = kernel @ weighted
                new_density_error[start:stop] = kernel @ weighted_error + density_remainder
            z_top = (b - grid - mean) / sd
            z_bot = (b + grid + mean) / sd
            sf_top = _sf(z_top)
            sf_bot = _sf(z_bot)
            top = float(np.dot(weighted, sf_top))
            bot = float(np.dot(weighted, sf_bot))
            tail_remainder = _quadrature_remainder(
                2.0 * prev_b, nodes, math.sqrt(prev_dt), sd, tail=True
            )
            top_error = float(np.dot(weighted_error, sf_top)) + tail_remainder
            bot_error = float(np.dot(weighted_error, sf_bot)) + tail_remainder
            y = grid - drift_z * prev_t
            pdf_top = _pdf(z_top)
            pdf_bot = _pdf(z_bot)
            top_score = float(np.dot(weighted, y * sf_top + sd * pdf_top))
            bot_score = float(np.dot(weighted, y * sf_bot - sd * pdf_bot))
            score_abs += float(
                np.dot(weighted, np.abs(y) * (sf_top + sf_bot) + sd * (pdf_top + pdf_bot))
            )
        upper.append(top)
        total.append(top + bot)
        upper_error.append(top_error)
        total_error.append(top_error + bot_error)
        upper_score.append(top_score)
        total_score.append(top_score + bot_score)
        density, density_error = new_density, new_density_error
        grid, weights, prev_t, prev_b, prev_dt = new_grid, new_weights, t, b, dt
    assert weights is not None and density is not None
    expected_error = math.fsum(
        error * (1.0 - t) for error, t in zip(total_error, fractions, strict=True)
    )
    return _Walk(
        tuple(upper),
        tuple(total),
        tuple(upper_error),
        tuple(total_error),
        float(np.dot(weights, density)),
        expected_error,
        z_max,
        tuple(upper_score),
        tuple(total_score),
        score_abs,
    )


def _nodes_floor(bounds: Sequence[float], fractions: Sequence[float]) -> int:
    """Smallest power of two giving at least one node per standard deviation
    of the integrand at every look.

    The first look samples a density of scale ``sd_1`` on ``[-b_1, b_1]``
    (interior node spacing ``~ pi b_1 / N``). Look ``k`` samples its kernel,
    of scale ``sd_k``, at the previous interval's nodes and integrates it
    against the previous density, whose narrowest feature has scale
    ``sd_{k-1}``; the product's scale is ``1 / sqrt(1/dt_{k-1} + 1/dt_k)``.
    Returns ``2 * NODES_MAX`` for a need beyond the ceiling.
    """
    need = float(NODES_MIN)
    prev_t = 0.0
    prev_b = 0.0
    prev_dt = 0.0
    for c, t in zip(bounds, fractions, strict=True):
        dt = t - prev_t
        b = c * math.sqrt(t)
        if prev_t == 0.0:
            need = max(need, math.pi * b / math.sqrt(dt))
        else:
            need = max(need, math.pi * prev_b * math.sqrt(1.0 / prev_dt + 1.0 / dt))
        prev_t, prev_b, prev_dt = t, b, dt
    if not need <= NODES_MAX:
        return 2 * NODES_MAX
    nodes = NODES_MIN
    while nodes < need:
        nodes *= 2
    return nodes


def _cheap_bounds(
    bounds: Sequence[float], fractions: Sequence[float], drift_z: float, exit_side: ExitSide
) -> tuple[float, float]:
    """Certified bounds without quadrature: one-look marginal exits are
    subsets of either-side crossing (only the first upper tail is a subset of
    first-upper-exit power); their union contains the crossing event."""
    upper = 0.0
    lower = 0.0
    for k, (c, t) in enumerate(zip(bounds, fractions, strict=True)):
        mu = drift_z * math.sqrt(t)
        top = float(_sf(c - mu))
        bot = float(_sf(c + mu))
        exceed = top if exit_side == "upper" else top + bot
        upper += exceed
        if exit_side == "both":
            # Being outside at any one look implies an earlier-or-current
            # boundary crossing. For directional first-upper-exit power, only
            # the first-look upper tail has that subset relation.
            lower = max(lower, exceed)
        elif k == 0:
            lower = exceed
    return lower * (1.0 - SF_REL), min(1.0, upper * (1.0 + SF_REL))


@dataclass(frozen=True, slots=True)
class _CrossingEnclosure:
    """A crossing probability with explicit numerical error bounds.

    ``quadrature`` is the propagated composite Gauss-Legendre remainder;
    ``rounding`` and ``sf_rel`` cover floating arithmetic and normal tails.
    ``expected_half`` independently bounds the expected-information estimate.
    ``resolved`` means the enclosure itself is rigorous; ``converged`` and
    ``expected_converged`` certify the declared point-estimate tolerances.
    """

    estimate: float
    lower: float
    upper: float
    quadrature: float
    rounding: float
    sf_rel: float
    slope: float
    slope_half: float
    nodes: int
    resolved: bool
    converged: bool
    expected_converged: bool
    expected_fraction: float | None
    expected_half: float | None
    levels: tuple[float, float, float] | None
    resolution_reason: str | None = None

    @property
    def half_width(self) -> float:
        return self.quadrature + self.rounding + self.sf_rel

    @property
    def cheap(self) -> bool:
        return self.levels is None


def _enclosure_from_walks(
    bounds: Sequence[float],
    fractions: Sequence[float],
    exit_side: ExitSide,
    nodes: int,
    walks: tuple[_Walk, _Walk, _Walk],
) -> _CrossingEnclosure:
    """Enclosure at ``nodes`` from rigorous per-integral remainder bounds."""
    looks = len(bounds)
    if exit_side == "upper":
        values = [math.fsum(w.upper) for w in walks]
        slopes = [math.fsum(w.upper_score) for w in walks]
        quadrature = math.fsum(walks[2].upper_error)
    else:
        values = [math.fsum(w.total) for w in walks]
        slopes = [math.fsum(w.total_score) for w in walks]
        quadrature = math.fsum(walks[2].total_error)
    estimate = values[2]
    finest = walks[2]
    scale = C_FP * looks * (nodes + finest.z_max * finest.z_max + 8.0) * _EPS
    subnormal = looks * nodes * _SUBNORMAL
    rounding = scale * estimate + subnormal
    sf_rel = SF_REL * estimate
    half = quadrature + rounding + sf_rel
    unused = math.fsum(mass * (1.0 - t) for mass, t in zip(finest.total, fractions, strict=True))
    expected = 1.0 - unused
    expected_rounding = scale * unused + C_FP * _EPS * max(1.0, abs(expected)) + subnormal
    expected_sf = SF_REL * unused
    expected_half = finest.expected_error + expected_rounding + expected_sf
    resolved = (
        nodes >= _nodes_floor(bounds, fractions)
        and math.isfinite(half)
        and math.isfinite(expected_half)
    )
    expected_budget = C_FP * looks * (nodes + 8.0) * _EPS + SF_REL
    converged = resolved and quadrature <= rounding + sf_rel
    expected_converged = resolved and expected_half <= expected_budget
    s1 = abs(slopes[2] - slopes[1])
    slope_rounding = (scale + SF_REL) * finest.score_abs + subnormal
    return _CrossingEnclosure(
        estimate=estimate,
        lower=max(0.0, estimate - half),
        upper=min(1.0, estimate + half),
        quadrature=quadrature,
        rounding=rounding,
        sf_rel=sf_rel,
        slope=max(-SLOPE_CAP, min(SLOPE_CAP, slopes[2])),
        slope_half=s1 + slope_rounding,
        nodes=nodes,
        resolved=resolved,
        converged=converged,
        expected_converged=expected_converged,
        expected_fraction=min(1.0, max(0.0, expected)),
        expected_half=expected_half,
        levels=(values[0], values[1], values[2]),
    )


def _crossing_enclosure(
    bounds: Sequence[float],
    fractions: Sequence[float],
    drift_z: float,
    exit_side: ExitSide,
    nodes: int,
) -> _CrossingEnclosure:
    """Enclosure from fresh walks at ``nodes/4, nodes/2, nodes``."""
    fr = list(fractions)
    walks = (
        _gl_walk(bounds, fr, drift_z, nodes // 4),
        _gl_walk(bounds, fr, drift_z, nodes // 2),
        _gl_walk(bounds, fr, drift_z, nodes),
    )
    return _enclosure_from_walks(bounds, fr, exit_side, nodes, walks)


def _cheap_enclosure(
    bounds: Sequence[float],
    fractions: Sequence[float],
    drift_z: float,
    exit_side: ExitSide,
    reason: str | None = None,
) -> _CrossingEnclosure:
    lower, upper = _cheap_bounds(bounds, fractions, drift_z, exit_side)
    return _CrossingEnclosure(
        estimate=0.5 * (lower + upper),
        lower=lower,
        upper=upper,
        quadrature=0.0,
        rounding=0.0,
        sf_rel=0.5 * (upper - lower),
        slope=0.0,
        slope_half=math.inf,
        nodes=0,
        resolved=False,
        converged=False,
        expected_converged=False,
        expected_fraction=None,
        expected_half=None,
        levels=None,
        resolution_reason=reason,
    )


def _certified_crossing(
    bounds: Sequence[float],
    fractions: Sequence[float],
    drift_z: float,
    exit_side: ExitSide,
    nodes: int | None = None,
    *,
    tight: bool = True,
) -> _CrossingEnclosure:
    """Return a rigorous enclosure, optionally refining its error budget."""
    fr = list(fractions)
    if math.isinf(drift_z):
        estimate = 1.0 if exit_side == "both" or drift_z > 0.0 else 0.0
        return _CrossingEnclosure(
            estimate=estimate,
            lower=estimate,
            upper=estimate,
            quadrature=0.0,
            rounding=0.0,
            sf_rel=0.0,
            slope=0.0,
            slope_half=0.0,
            nodes=0,
            resolved=True,
            converged=True,
            expected_converged=True,
            expected_fraction=fr[0],
            expected_half=0.0,
            levels=(estimate, estimate, estimate),
        )
    first_mu = drift_z * math.sqrt(fr[0])
    first_top = float(_sf(bounds[0] - first_mu))
    first_bottom = float(_sf(bounds[0] + first_mu))
    top_saturated = first_top == 1.0 and first_bottom == 0.0
    bottom_saturated = first_bottom == 1.0 and first_top == 0.0
    if top_saturated or bottom_saturated:
        detected = exit_side == "both" or top_saturated
        estimate = 1.0 if detected else 0.0
        expected_half = (1.0 - fr[0]) * SF_REL
        return _CrossingEnclosure(
            estimate=estimate,
            lower=max(0.0, estimate - SF_REL),
            upper=min(1.0, estimate + SF_REL),
            quadrature=0.0,
            rounding=0.0,
            sf_rel=SF_REL,
            slope=0.0,
            slope_half=0.0,
            nodes=0,
            resolved=True,
            converged=True,
            expected_converged=True,
            expected_fraction=fr[0],
            expected_half=expected_half,
            levels=(estimate, estimate, estimate),
        )
    floor = _nodes_floor(bounds, fr)
    if floor > NODES_MAX:
        return _cheap_enclosure(
            bounds,
            fr,
            drift_z,
            exit_side,
            "node floor exceeds the quadrature ceiling",
        )
    n = max(nodes or 0, floor, 4 * NODES_MIN)
    if len(fr) > MAX_WALK_LOOKS or len(fr) * n * n > MAX_WALK_WORK:
        return _cheap_enclosure(
            bounds,
            fr,
            drift_z,
            exit_side,
            "quadrature work ceiling reached",
        )
    walks = (
        _gl_walk(bounds, fr, drift_z, n // 4),
        _gl_walk(bounds, fr, drift_z, n // 2),
        _gl_walk(bounds, fr, drift_z, n),
    )
    while True:
        enclosure = _enclosure_from_walks(bounds, fr, exit_side, n, walks)
        accuracy_reached = enclosure.converged and enclosure.expected_converged
        if not tight:
            return enclosure
        if accuracy_reached:
            return enclosure
        if n >= NODES_MAX:
            return replace(enclosure, resolution_reason="quadrature node ceiling reached")
        next_n = 2 * n
        if len(fr) * next_n * next_n > MAX_WALK_WORK:
            return replace(
                enclosure,
                resolution_reason="quadrature work ceiling reached after a valid enclosure",
            )
        n = next_n
        walks = (walks[1], walks[2], _gl_walk(bounds, fr, drift_z, n))


def _always_valid_z_bound(
    information_fraction: float, alpha: float, exit_side: ExitSide, *, e_value_dual: bool = False
) -> float:
    """Always-valid z boundary at the mixture's own se-independent optimal
    tuning, computed from the RAW alpha. Its log term is built at the level
    estimation.asymptotic_mean.boundary_alpha gives the runtime cell: alpha
    for a two-sided exit or an ``e_value_dual`` family member (the dual of
    its sign-gated e-value), otherwise twice alpha for a one-sided exit --
    bit-identical to count_boundary at the runtime's derived rho."""
    r_star = float(mixture_r_star(Fraction(alpha)))
    boundary_alpha = alpha if exit_side == "both" or e_value_dual else 2.0 * alpha
    log_r = math.log(information_fraction) + math.log(r_star)
    log_one_plus_r = (
        log_r + math.log1p(math.exp(-log_r)) if log_r > 0.0 else math.log1p(math.exp(log_r))
    )
    log_one_plus_inverse = (
        math.log1p(math.exp(-log_r)) if log_r >= 0.0 else -log_r + math.log1p(math.exp(log_r))
    )
    log_c_sq = log_one_plus_inverse + math.log(log_one_plus_r - 2.0 * math.log(boundary_alpha))
    return math.inf if 0.5 * log_c_sq >= _LOG_FLOAT_MAX else math.exp(0.5 * log_c_sq)


def _planning_bounds_from_log_se(
    spec: SequentialPlanningSpec,
    fractions: Sequence[float],
    log_se_sq: float,
    alpha: float,
    exit_side: ExitSide = "both",
    *,
    e_value_dual: bool = False,
) -> list[float]:
    """Sequential z boundaries with a log-scale full-information variance."""
    _require_prospective(spec)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        _raise("power.alpha_finite", alpha=alpha)
    fr = _validate_information_fractions(fractions)
    return [_always_valid_z_bound(t, alpha, exit_side, e_value_dual=e_value_dual) for t in fr]


def _standardized_drift(delta: float, log_se_sq: float) -> float:
    """``delta / sqrt(exp(log_se_sq))`` without underflowing the denominator."""
    if delta == 0.0:
        return 0.0
    log_abs = math.log(abs(delta)) - 0.5 * log_se_sq
    value = math.inf if log_abs >= _LOG_FLOAT_MAX else math.exp(log_abs)
    return math.copysign(value, delta)


def _sequential_enclosure_from_log_se(
    looks: _PlannedLooks,
    delta: float,
    log_se_sq: float,
    alpha: float,
    exit_side: ExitSide = "both",
    *,
    e_value_dual: bool = False,
) -> _CrossingEnclosure:
    """Internal log-variance entry point shared by all planning solvers."""
    bounds = _planning_bounds_from_log_se(
        looks.spec, looks.fractions, log_se_sq, alpha, exit_side, e_value_dual=e_value_dual
    )
    return _certified_crossing(
        bounds, looks.fractions, _standardized_drift(delta, log_se_sq), exit_side
    )


def _sequential_estimates_from_log_se(
    looks: _PlannedLooks,
    delta: float,
    log_se_sq: float,
    alpha: float,
    exit_side: ExitSide = "both",
    *,
    e_value_dual: bool = False,
) -> tuple[float, float]:
    enclosure = _sequential_enclosure_from_log_se(
        looks, delta, log_se_sq, alpha, exit_side, e_value_dual=e_value_dual
    )
    return (
        _point_estimate(enclosure, "estimate"),
        _point_estimate(enclosure, "expected_fraction"),
    )


def planning_bounds(
    spec: SequentialPlanningSpec,
    fractions: Sequence[float],
    se_full: float,
    alpha: float,
    exit_side: ExitSide = "both",
) -> list[float]:
    """Per-look z-scale boundary for a planned look schedule.

    ``exit_side="both"`` is the two-sided boundary at ``alpha``.
    ``exit_side="upper"`` is the boundary of a one-sided cell outside a
    registered family, which spends one tail of the two-sided boundary at
    ``2 * alpha``. A registered family member -- a sequential secondary or a
    registered breakout cell -- is built at ``alpha`` on its tested side
    instead; plan it with ``ArmPlanningProcedure.standard(role="secondary",
    ...)`` through the arm solvers, which use that boundary. ``fractions``
    may be a partial schedule.
    """
    _require_prospective(spec)
    if not math.isfinite(se_full) or se_full <= 0.0:
        _raise("power.se_full_finite", se_full=se_full)
    return _planning_bounds_from_log_se(spec, fractions, 2.0 * math.log(se_full), alpha, exit_side)


def sequential_power_enclosure(
    spec: SequentialPlanningSpec,
    delta: float,
    se_full: float,
    alpha: float,
    planned_looks: int | None = None,
    exit_side: ExitSide = "both",
) -> _CrossingEnclosure:
    """Certified enclosure of the sequential design's power at the resolved
    look schedule; see ``sequential_power``."""
    looks = _resolve_looks(spec, planned_looks)
    if not math.isfinite(se_full) or se_full <= 0.0:
        _raise("power.se_full_finite", se_full=se_full)
    return _sequential_enclosure_from_log_se(
        looks, delta, 2.0 * math.log(se_full), alpha, exit_side
    )


def _point_estimate(
    enclosure: _CrossingEnclosure, field: Literal["estimate", "expected_fraction"]
) -> float:
    value = getattr(enclosure, field)
    converged = enclosure.converged if field == "estimate" else enclosure.expected_converged
    if (
        not enclosure.resolved
        or not converged
        or value is None
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        reason = enclosure.resolution_reason or "declared point-estimate tolerance was not met"
        _raise("power.boundary_crossing_quadrature", reason=reason)
    return value


def sequential_power(
    spec: SequentialPlanningSpec,
    delta: float,
    se_full: float,
    alpha: float,
    planned_looks: int | None = None,
    exit_side: ExitSide = "both",
) -> float:
    """Power of the sequential design at its planned looks.

    ``GaussianScoreMixture`` uses equal batches (default 14 looks).

    ``exit_side="both"`` (default) is the either-side crossing probability:
    the two-sided sense of "power". ``exit_side="upper"`` is the probability
    that the FIRST boundary exit is through the top (``+c*sqrt(t)``):
    directional, one-sided power. A lower-boundary exit still stops the
    experiment (a harm-side stop is a real decision) but is not counted as
    a detection, matching the fixed-horizon one-sided power formula.
    The boundary doubles ``alpha`` itself for ``exit_side="upper"``, so pass
    the one-sided ``alpha``; for ``"less"``, negate ``delta`` (the boundary
    is symmetric, so a lower-exit probability at drift ``d`` equals an
    upper-exit probability at drift ``-d``). That one-sided boundary is the
    one a cell outside a registered family uses; a family member such as a
    sequential secondary is built at ``alpha`` on its tested side and is
    planned with ``ArmPlanningProcedure.standard(role="secondary", ...)``
    through the arm solvers (see ``planning_bounds``).

    Returns the point estimate only from a resolved numerical enclosure.
    A work or node ceiling raises rather than returning an arbitrary midpoint
    of cheap bounds. A supplied ``planned_looks`` must be a positive integer.
    """
    enclosure = sequential_power_enclosure(spec, delta, se_full, alpha, planned_looks, exit_side)
    return _point_estimate(enclosure, "estimate")


def sequential_expected_information_fraction(
    spec: SequentialPlanningSpec,
    delta: float,
    se_full: float,
    alpha: float,
    planned_looks: int | None = None,
    exit_side: ExitSide = "both",
) -> float:
    """E[T] in [0, 1]: expected information fraction the sequential design
    stops at, under true log-lift ``delta``.

    Multiply by the fixed-horizon (``n_max``) sample size to get the
    expected sample size -- the honest economic counterpart to
    ``sequential_power``'s ``n_max``: early stopping is the entire
    argument for monitoring sequentially, and this is always <= 1 (a
    design that never crosses runs to the planned final look). Every
    look's first-exit mass is charged its own information fraction and
    the surviving mass the final planned look's.

    ``exit_side`` selects the SAME physical boundary
    ``sequential_power`` uses at this alpha (a one-sided decision tunes a
    narrower boundary -- see ``_always_valid_z_bound``); EITHER boundary
    exit halts the trial regardless of ``exit_side``, so this is not a
    directional crossing probability the way ``sequential_power``'s
    ``exit_side="upper"`` is.

    The look schedule resolves exactly as in ``sequential_power``.
    """
    enclosure = sequential_power_enclosure(spec, delta, se_full, alpha, planned_looks, exit_side)
    return _point_estimate(enclosure, "expected_fraction")


_SEQUENTIAL_MDE_UNREPRESENTABLE = RefusalSpec(
    "power.sequential_mde.unrepresentable",
    InvalidRequestError,
    template="sequential design at power={target_power}, se={se:.6g}: the minimum detectable log-effect ({mde_theta:.6g}) is too large to express as a relative lift in float64 -- the standard error is too large for a relative-scale minimum detectable effect",
)
_SEQUENTIAL_MDE_COMPLIANCE_UNREPRESENTABLE = RefusalSpec(
    "power.sequential_mde.compliance_unrepresentable",
    InvalidRequestError,
    template="sequential design at power={target_power}: the raw minimum detectable relative lift ({mde_relative:.6g}) adjusted onto the complier scale by compliance {compliance:.6g} overflows float64 -- the compliance is too small for a representable complier-scale effect",
)
_SEQUENTIAL_MDE_DECREASE_UNREPRESENTABLE = RefusalSpec(
    "power.sequential_mde.decrease_unrepresentable",
    InvalidRequestError,
    template="sequential design at power={target_power}: the raw minimum detectable relative lift ({mde_relative:.6g}) adjusted onto the complier scale by compliance {compliance:.6g} reaches or exceeds a 100% decrease, which is not a representable relative effect -- the detectable decrease is too large to resolve at this standard error",
)

RESOLUTION_MULTIPLE = 4.0
# K=14 shifted-null/high-power searches need about 2,000 certified evaluations;
# retain 2x headroom while bounding pathological searches.
MAX_EVALUATIONS = 4096
# sup_A |E[1_A (Z^2 - 1)]| = E[(Z^2 - 1)^+] = 2 phi(1): no exit event's second
# drift derivative exceeds it.
SECOND_DERIVATIVE_CAP = 2.0 * math.exp(-0.5) / math.sqrt(2.0 * math.pi)

_NODE_CEILING = "node ceiling reached without a certified quadrature remainder"
_FALLS_BELOW = (
    "an earlier interval straddles target within numerical accuracy and power then falls below it"
)


@dataclass(frozen=True, slots=True)
class _StraddleGap:
    """An ordinal interval whose resolved enclosure straddles the target:
    every candidate inside has power within ``RESOLUTION_MULTIPLE`` times
    ``half`` (the midpoint enclosure's half-width) of the target, and none
    is certified detected."""

    start: float
    end: float
    lower: float
    upper: float
    half: float


@dataclass(frozen=True, slots=True)
class _SequentialMde:
    """A certified sequential minimum detectable effect.

    ``mde_relative`` is the first candidate whose point enclosure reaches
    the target with an excess of at most ``RESOLUTION_MULTIPLE`` times its
    half-width; every admissible candidate at or before ``bracket[0]`` in
    distance order has certified power below the target, except inside the
    recorded ``straddle_gaps``. ``power`` is the enclosure's estimate;
    ``distance``/``theta`` locate the alternative.
    """

    mde_relative: float
    power: float
    enclosure: _CrossingEnclosure
    bracket: tuple[float, float]
    straddle_gaps: tuple[_StraddleGap, ...]
    distance: float
    theta: float


@dataclass(frozen=True, slots=True)
class _Box:
    """Drift and z-boundary ranges over an ordinal interval of candidates,
    with ``band[k]`` the boundary half-range at look ``k`` (zero for a
    group-sequential planning boundary)."""

    drift_lo: float
    drift_hi: float
    c_lo: tuple[float, ...]
    c_hi: tuple[float, ...]
    band: tuple[float, ...]

    @property
    def drift_mid(self) -> float:
        return 0.5 * (self.drift_lo + self.drift_hi)

    @property
    def half_drift(self) -> float:
        return 0.5 * (self.drift_hi - self.drift_lo)

    @property
    def c_mid(self) -> tuple[float, ...]:
        return tuple(0.5 * (lo + hi) for lo, hi in zip(self.c_lo, self.c_hi, strict=True))


@dataclass(frozen=True, slots=True)
class _Interval:
    """Certified power range over a box around its midpoint enclosure."""

    lower: float
    upper: float
    variation: float
    midpoint: _CrossingEnclosure


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


@dataclass(slots=True)
class _SequentialMdeSearch:
    """Left-first search for the first candidate whose sequential power is
    certified to reach the target.

    Candidates are ``_MdeSearch``'s public floats, subdivided at float64
    ordinal midpoints in distance order. Each candidate's drift and (for
    GaussianScoreMixture) boundary use the alternative's own variance. An interval is
    excluded only when a certified upper bound over it -- the union bound
    over marginal exceedances, or the midpoint enclosure widened by the
    same-Gaussian-path variation bound -- lies below the target; a
    candidate is detected only when the lower bound of its own resolved
    point enclosure reaches the target. A straddling interval is refined by
    subdivision while the variation term dominates and by node doubling
    while the quadrature term does; when neither can separate it from the
    target it is recorded as a gap and the search continues to the right,
    refusing rather than returning a later root if power then falls below
    the gap's own lower bound. The evaluation ceiling and the node ceiling
    refuse as unresolved; they never become an unattainable claim.
    """

    search: _MdeSearch
    inference: SequentialPlanningSpec
    alpha_seq: float
    exit_side: Literal["both", "upper"]
    fractions: tuple[float, ...]
    sqrt_t: tuple[float, ...]
    bounds: tuple[float, ...]
    require_expected: bool
    evaluations: int = 0
    # Node level at which this boundary family's quadrature term last had to
    # be halved below the rounding budget; later evaluations start there.
    nodes: int | None = None
    cache: dict[tuple[float, int | None], _CrossingEnclosure] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        looks: _PlannedLooks,
        search: _MdeSearch,
        *,
        alpha_seq: float,
        exit_side: Literal["both", "upper"],
        e_value_dual: bool = False,
        require_expected: bool = True,
    ) -> _SequentialMdeSearch:
        inference, fractions = looks.spec, looks.fractions
        # Every remaining planning spec's boundary depends only on
        # (fractions, alpha_seq, exit_side, e_value_dual): GaussianScoreMixture
        # is tuned at its own se-independent optimum (mixture_r_star), so it no
        # longer varies with theta.
        bounds = tuple(
            _planning_bounds_from_log_se(
                inference, fractions, 0.0, alpha_seq, exit_side, e_value_dual=e_value_dual
            )
        )
        return cls(
            search=search,
            inference=inference,
            alpha_seq=alpha_seq,
            exit_side=exit_side,
            fractions=fractions,
            sqrt_t=tuple(math.sqrt(t) for t in fractions),
            bounds=bounds,
            require_expected=require_expected,
        )

    # -- candidate geometry -------------------------------------------------

    @property
    def target(self) -> float:
        return self.search.target

    def log_se_sq(self, theta: float) -> float:
        return self.search.plan.log_se_sq(theta)

    def se_full(self, theta: float) -> float:
        return _se_from_log(self.log_se_sq(theta))

    def drift(self, point: tuple[float, float]) -> float:
        distance, theta = point
        return _noncentrality(abs(distance), self.log_se_sq(theta))

    def bounds_at(self, log_se_sq: float) -> tuple[float, ...]:
        return self.bounds

    def box(self, a: float, b: float) -> _Box:
        """Ranges over the candidates between ``a`` and ``b``: the variance is
        monotone in ``theta`` and the drift is monotone except across the
        decreasing direction's noncentrality peak. Every look's boundary is
        fixed by ``(fractions, alpha_seq, exit_side, e_value_dual)`` alone."""
        pa, pb = self.search.candidate(a), self.search.candidate(b)
        assert pa is not None and pb is not None
        drifts = [self.drift(pa), self.drift(pb)]
        if self.search.sigma < 0.0:
            d_peak = self.search.peak_distance()
            d_lo, d_hi = sorted((abs(pa[0]), abs(pb[0])))
            if d_lo < d_peak < d_hi:
                drifts.append(_noncentrality(d_peak, self.log_se_sq(self.search.theta0 - d_peak)))
        return _Box(
            min(drifts), max(drifts), self.bounds, self.bounds, (0.0,) * len(self.fractions)
        )

    # -- ordinals in distance order -------------------------------------------

    def ordinal(self, m: float) -> int:
        return int(self.search.sigma) * float_ordinal(m)

    def from_ordinal(self, ordinal: int) -> float:
        return float_from_ordinal(int(self.search.sigma) * ordinal)

    # -- certified variation over a box ---------------------------------------

    @staticmethod
    def slope_bound(lower: float, upper: float) -> float:
        """``sup |dP/d drift|`` over exit events whose power lies in
        ``[lower, upper]``: ``|E[1_A Z]| <= phi(Phi^{-1}(P(A)))``, largest
        at one half."""
        p = 0.5 if lower <= 0.5 <= upper else (upper if upper < 0.5 else lower)
        p = min(max(p, 0.0), 1.0)
        return _phi(float(_ndtri(p))) if 0.0 < p < 1.0 else 0.0

    def split_ordinal(self, a: float, b: float, a_ordinal: int, b_ordinal: int) -> int:
        """Choose a value-space midpoint, falling back to ordinal space only
        when rounding leaves no interior float. Raw ordinal bisection spends
        thousands of evaluations traversing the exponent range near zero."""
        midpoint = a + 0.5 * (b - a)
        candidate = self.ordinal(midpoint)
        if a_ordinal < candidate < b_ordinal:
            return candidate
        return a_ordinal + (b_ordinal - a_ordinal) // 2

    def band_term(self, box: _Box) -> float:
        """Same-Gaussian-path bound for the boundary range: a classification
        changes only if a marginal ``Z_k`` lies between the midpoint
        threshold and the moved one."""
        total = 0.0
        c_mid = box.c_mid
        for k, half_c in enumerate(box.band):
            if half_c == 0.0:
                continue
            slack = half_c + box.half_drift * self.sqrt_t[k]
            mu = box.drift_mid * self.sqrt_t[k]
            top = max(0.0, abs(c_mid[k] - mu) - slack)
            bottom = max(0.0, abs(c_mid[k] + mu) - slack)
            total += half_c * (_phi(top) + _phi(bottom))
        return total

    def variation(
        self,
        box: _Box,
        lower: float = 0.0,
        upper: float = 1.0,
        midpoint: _CrossingEnclosure | None = None,
    ) -> float:
        """Rigorous event-probability variation over a drift/boundary box.

        The universal Gaussian score bound avoids relying on a numerically
        differentiated midpoint slope. ``midpoint`` remains an accepted
        argument so callers can share the same contraction loop.
        """
        slope = self.slope_bound(lower, upper)
        if self.exit_side == "both":
            # With a fixed symmetric boundary, either-side crossing power is
            # even in drift, so P'(0)=0. The universal second-derivative bound
            # gives a much tighter rigorous slope near the null.
            slope = min(
                slope,
                SECOND_DERIVATIVE_CAP * (abs(box.drift_mid) + box.half_drift),
            )
        drift_term = box.half_drift * slope
        return min(1.0, drift_term + self.band_term(box))

    def union_upper(self, box: _Box) -> float:
        """Certified upper bound over the box without quadrature: the union
        over looks of each marginal exceedance at its most favourable
        configuration."""

        total = 0.0
        for k, c in enumerate(box.c_lo):
            top = float(_ndtr(box.drift_hi * self.sqrt_t[k] - c))
            bottom = float(_ndtr(-c - box.drift_lo * self.sqrt_t[k]))
            total += top if self.exit_side == "upper" else top + bottom
        return min(1.0, total * (1.0 + SF_REL))

    # -- certified enclosures -------------------------------------------------

    def point(self, m: float, nodes: int | None = None) -> _CrossingEnclosure:
        """Enclosure of candidate ``m``'s power. Without ``nodes`` the cheap
        bounds are tried first and returned when they already decide the
        candidate against the target; a quadrature enclosure starts at the
        node level the solve has already needed."""
        key = (m, nodes)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        point = self.search.candidate(m)
        assert point is not None
        log_se_sq = self.log_se_sq(point[1])
        bounds = self.bounds_at(log_se_sq)
        drift = _noncentrality(abs(point[0]), log_se_sq)
        if nodes is None:
            cheap = _cheap_enclosure(bounds, self.fractions, drift, self.exit_side)
            if cheap.upper < self.target or cheap.lower >= self.target:
                self.cache[key] = cheap
                return cheap
        self.evaluations += 1
        level = max(nodes or 0, self.nodes or 0) or None
        enclosure = _certified_crossing(
            bounds,
            self.fractions,
            drift,
            self.exit_side,
            level,
            tight=False,
        )
        self.cache[key] = enclosure
        return enclosure

    def interval(self, box: _Box, nodes: int | None = None, cheap_up: float = 1.0) -> _Interval:
        """Certified power range over ``box`` around its midpoint enclosure.
        The first-order constant is contracted with the box's own certified
        range (monotone, at most four passes), seeded by the union bound."""
        self.evaluations += 1
        level = max(nodes or 0, self.nodes or 0) or None
        midpoint = _certified_crossing(
            box.c_mid,
            self.fractions,
            box.drift_mid,
            self.exit_side,
            level,
            tight=False,
        )
        lower, upper = 0.0, cheap_up
        variation = self.variation(box, lower, upper, midpoint)
        for _ in range(4):
            lower = max(0.0, midpoint.lower - variation)
            upper = min(1.0, midpoint.upper + variation)
            contracted = self.variation(box, lower, upper, midpoint)
            if contracted >= variation:
                break
            variation = contracted
        return _Interval(
            max(0.0, midpoint.lower - variation),
            min(1.0, midpoint.upper + variation),
            variation,
            midpoint,
        )

    def can_refine(self, enclosure: _CrossingEnclosure) -> bool:
        next_nodes = 2 * enclosure.nodes
        return (
            next_nodes <= NODES_MAX
            and len(self.fractions) * next_nodes * next_nodes <= MAX_WALK_WORK
            and enclosure.resolution_reason is None
        )

    def quadrature_dominates(self, enclosure: _CrossingEnclosure) -> bool:
        return self.can_refine(enclosure) and (
            enclosure.quadrature > enclosure.rounding + enclosure.sf_rel
        )

    def doubled(self, enclosure: _CrossingEnclosure) -> int:
        """Remember the next feasible node level for this boundary family."""
        next_nodes = 2 * enclosure.nodes
        self.nodes = max(self.nodes or 0, next_nodes)
        return self.nodes

    def refine_point(self, m: float, enclosure: _CrossingEnclosure) -> _CrossingEnclosure:
        """Upgrade cheap bounds and refine uncertainty that straddles or lies
        within the declared answer-resolution band around the target."""
        if enclosure.cheap:
            enclosure = self.point(m, 4 * NODES_MIN)
            if enclosure.cheap:
                return enclosure
        while self.quadrature_dominates(enclosure) and (
            enclosure.lower < self.target <= enclosure.upper
            or abs(enclosure.estimate - self.target) <= RESOLUTION_MULTIPLE * enclosure.half_width
        ):
            enclosure = self.point(m, self.doubled(enclosure))
        return enclosure

    def certify(
        self, m: float, enclosure: _CrossingEnclosure, interval: tuple[float, float]
    ) -> _CrossingEnclosure | _MdeRefusal:
        """Return a point after its power enclosure meets its tolerance.
        Cheap and quadrature lower bounds may be intersected because both are
        rigorous."""
        cheap_lower = enclosure.lower if enclosure.cheap else None
        refined = self.refine_point(m, enclosure)
        while refined.resolved and not refined.converged and self.can_refine(refined):
            refined = self.point(m, self.doubled(refined))
        if not refined.resolved or not refined.converged:
            return self.unresolved(
                interval,
                (refined.lower, refined.upper),
                refined.resolution_reason or _NODE_CEILING,
            )
        if cheap_lower is not None and cheap_lower > refined.lower:
            refined = replace(refined, lower=cheap_lower)
        return refined

    def certify_expected(
        self, m: float, enclosure: _CrossingEnclosure, interval: tuple[float, float]
    ) -> _CrossingEnclosure | _MdeRefusal:
        """Refine expected information only for a returned direct-MDE answer.

        Every power enclosure is rigorous, so intersect successive bounds
        instead of discarding a stronger bound established at fewer nodes.
        """
        refined = enclosure
        while refined.resolved and not refined.expected_converged and self.can_refine(refined):
            next_enclosure = self.point(m, self.doubled(refined))
            lower = max(refined.lower, next_enclosure.lower)
            upper = min(refined.upper, next_enclosure.upper)
            # The finer estimate may fall outside the older, tighter bound;
            # the reported power must stay inside the enclosure it certifies.
            refined = replace(
                next_enclosure,
                lower=lower,
                upper=upper,
                estimate=min(max(next_enclosure.estimate, lower), upper),
            )
        if not refined.resolved or not refined.converged or not refined.expected_converged:
            return self.unresolved(
                interval,
                (refined.lower, refined.upper),
                refined.resolution_reason or _NODE_CEILING,
            )
        return refined

    @staticmethod
    def certified_lower(enclosure: _CrossingEnclosure, target: float) -> bool:
        """A lower bound counts only from cheap bounds or a resolved enclosure."""
        return enclosure.lower >= target and (enclosure.cheap or enclosure.resolved)

    # -- refusals -------------------------------------------------------------

    def unresolved(
        self,
        interval: tuple[float, float],
        enclosure: tuple[float, float] | None,
        stopping_reason: str,
    ) -> _MdeRefusal:
        return _MdeRefusal(
            "numerical_resolution",
            _MDE_NUMERICAL_RESOLUTION,
            {
                "target_power": self.target,
                "direction": self.search.direction,
                "unresolved_interval": interval,
                "power_enclosure": enclosure,
                "stopping_reason": stopping_reason,
            },
        )

    def beyond_float64(self, m_max: float) -> _MdeRefusal:
        """Every representable increase is certified below the target while
        the drift grows without bound beyond float64, so the answer exists
        but has no representation: named after the ceiling that ended the
        admissible candidates -- the compliance division when compliance
        dilutes the lift, float64 itself otherwise."""
        top = self.search.candidate(m_max)
        assert top is not None
        theta = top[1]
        compliance = self.search.compliance
        if compliance < 1.0:
            return _MdeRefusal(
                "unrepresentable",
                _SEQUENTIAL_MDE_COMPLIANCE_UNREPRESENTABLE,
                {
                    "target_power": self.target,
                    "mde_relative": math.expm1(theta),
                    "compliance": compliance,
                },
            )
        return _MdeRefusal(
            "unrepresentable",
            _SEQUENTIAL_MDE_UNREPRESENTABLE,
            {"target_power": self.target, "se": self.se_full(theta), "mde_theta": theta},
        )

    # -- the search -----------------------------------------------------------

    def finish(
        self,
        a_excl: float,
        b: float,
        enclosure: _CrossingEnclosure,
        gaps: list[_StraddleGap],
    ) -> _SequentialMde:
        point = self.search.candidate(b)
        assert point is not None
        return _SequentialMde(
            mde_relative=b,
            power=enclosure.estimate,
            enclosure=enclosure,
            bracket=(a_excl, b),
            straddle_gaps=tuple(gaps),
            distance=point[0],
            theta=point[1],
        )

    def resolve_lower_endpoint(
        self,
    ) -> tuple[float, _CrossingEnclosure] | _SequentialMde | _MdeRefusal:
        """Classify the first admissible candidate before interval search."""
        target = self.target
        m_min = self.search.lower_endpoint()
        if isinstance(m_min, _MdeRefusal):
            return m_min
        low = self.point(m_min)
        if low.cheap and low.upper < target:
            return m_min, low
        certified = self.certify(m_min, low, (m_min, m_min))
        if isinstance(certified, _MdeRefusal):
            return certified
        if certified.lower >= target:
            if m_min == 0.0:
                _raise(
                    "power.sequential_mde.design_search_minimum",
                    estimate=certified.estimate,
                    target=target,
                )
            if self.require_expected:
                expected = self.certify_expected(m_min, certified, (m_min, m_min))
                if isinstance(expected, _MdeRefusal):
                    return expected
                assert expected.lower >= target, (
                    "unreachable: certify_expected intersects bounds, so the lower bound cannot fall"
                )
                certified = expected
            return self.finish(m_min, m_min, certified, [])
        if certified.upper >= target:
            return self.unresolved(
                (m_min, m_min),
                (certified.lower, certified.upper),
                "lower endpoint enclosure straddles target",
            )
        return m_min, certified

    def assess_candidate(
        self,
        *,
        a_excl: float,
        b: float,
        enclosure: _CrossingEnclosure,
        adjacent: bool,
        gaps: list[_StraddleGap],
    ) -> bool | _SequentialMde | _MdeRefusal:
        """Return False when undetected, True when detected but unresolved,
        or the terminal answer/refusal."""
        if not self.certified_lower(enclosure, self.target):
            return False
        if enclosure.cheap and not adjacent:
            # A cheap lower bound is sufficient to bracket a detection.
            # Delay candidate-dependent quadrature until the left-first
            # search isolates the crossing.
            return True
        if not adjacent and enclosure.resolved and not enclosure.converged:
            # Rigorous lower bounds still bracket the crossing. Subdivide to
            # a better-conditioned boundary instead of returning this
            # unconverged point estimate.
            return True
        certified = self.certify(b, enclosure, (a_excl, b))
        if isinstance(certified, _MdeRefusal):
            return certified
        if certified.lower < self.target:
            # Refining expected information can widen the power allowance.
            # This candidate is no longer a certified detection; continue the
            # left-first search rather than consuming its point estimate.
            return False
        finishing = (
            certified.estimate - self.target <= RESOLUTION_MULTIPLE * certified.half_width
            or adjacent
        )
        if finishing and self.require_expected:
            expected = self.certify_expected(b, certified, (a_excl, b))
            if isinstance(expected, _MdeRefusal):
                return expected
            if expected.lower < self.target:
                return False
            certified = expected
            finishing = (
                certified.estimate - self.target <= RESOLUTION_MULTIPLE * certified.half_width
                or adjacent
            )
        if finishing:
            return self.finish(a_excl, b, certified, gaps)
        return True

    def exclude_interval(
        self,
        *,
        a_excl: float,
        b: float,
        upper: float,
        max_upper: float,
        gaps: list[_StraddleGap],
    ) -> tuple[float, float] | _MdeRefusal:
        """Advance the excluded prefix or refuse after an unresolved gap."""
        if gaps and upper < gaps[-1].lower:
            return self.unresolved(
                (a_excl, b),
                (gaps[-1].lower, gaps[-1].upper),
                _FALLS_BELOW,
            )
        return b, max(max_upper, upper)

    def decrease_floor_refusal(self) -> _MdeRefusal | None:
        """Return the legacy code only after certifying a crossing beyond the
        public ``-100%`` floor.

        The physical floor preceding the fixed-reference peak is necessary
        but not sufficient: low-information designs may miss the target even
        at that peak. In that case the answer is genuinely unattainable, not
        merely unrepresentable.
        """
        search = self.search
        if search.sigma > 0.0 or search.compliance == 1.0:
            return None
        d_floor = min(-search.theta_floor, search.theta0 - search.theta_floor)
        d_peak = search.peak_distance()
        if d_floor >= d_peak:
            return None

        def enclosure(distance: float) -> _CrossingEnclosure:
            theta = search.theta0 - distance
            log_se_sq = self.log_se_sq(theta)
            return _certified_crossing(
                self.bounds_at(log_se_sq),
                self.fractions,
                _noncentrality(distance, log_se_sq),
                self.exit_side,
            )

        peak = enclosure(d_peak)
        m_floor = math.expm1(-d_floor) / search.compliance
        m_peak = math.expm1(-d_peak) / search.compliance
        if not peak.resolved:
            return self.unresolved(
                (m_floor, m_peak),
                (peak.lower, peak.upper),
                peak.resolution_reason or _NODE_CEILING,
            )
        if peak.upper < self.target:
            return None
        if not peak.converged or peak.lower < self.target:
            return self.unresolved(
                (m_floor, m_peak),
                (peak.lower, peak.upper),
                peak.resolution_reason or "peak power enclosure straddles target",
            )

        low, high = d_floor, d_peak
        for _ in range(80):
            middle = low + 0.5 * (high - low)
            if middle == low or middle == high:
                break
            current = enclosure(middle)
            if not current.resolved or not current.converged:
                return self.unresolved(
                    (
                        math.expm1(-low) / search.compliance,
                        math.expm1(-high) / search.compliance,
                    ),
                    (current.lower, current.upper),
                    current.resolution_reason or _NODE_CEILING,
                )
            if current.lower >= self.target:
                high = middle
            elif current.upper < self.target:
                low = middle
            else:
                # The crossing is inside this rigorous probability enclosure;
                # ``high`` remains the nearest certified detected candidate.
                break
        mde_relative = math.expm1(-high) / search.compliance
        if not math.isfinite(mde_relative) or mde_relative >= -1.0:
            return None
        return _MdeRefusal(
            "unattainable",
            _SEQUENTIAL_MDE_DECREASE_UNREPRESENTABLE,
            {
                "target_power": self.target,
                "mde_relative": mde_relative,
                "compliance": search.compliance,
            },
        )

    def no_answer(
        self,
        *,
        m_max: float,
        a_excl: float,
        max_upper: float,
        gaps: list[_StraddleGap],
    ) -> _MdeRefusal:
        """Classify a fully searched domain with no certified candidate."""

        if gaps:
            return self.unresolved(
                (a_excl, gaps[-1].end),
                (gaps[0].lower, gaps[0].upper),
                "power reaches target only within numerical accuracy; no candidate is certified "
                "detected",
            )
        if self.search.sigma > 0.0 and not self.search.plan.bounded:
            return self.beyond_float64(m_max)
        decrease_floor = self.decrease_floor_refusal()
        if decrease_floor is not None:
            return decrease_floor
        return self.search.unattainable(max_upper, "sequential_exclusion")

    def solve(self) -> _SequentialMde | _MdeRefusal:  # noqa: PLR0915
        target = self.target
        lower = self.resolve_lower_endpoint()
        if not isinstance(lower, tuple):
            return lower
        m_min, low = lower
        m_max = self.search.upper_endpoint(m_min)
        top = self.point(m_max)
        stack = [(self.ordinal(m_min), self.ordinal(m_max), self.certified_lower(top, target))]
        a_excl = m_min
        max_upper = low.upper
        gaps: list[_StraddleGap] = []

        while stack:
            a_ord, b_ord, b_certified = stack.pop()
            a, b = self.from_ordinal(a_ord), self.from_ordinal(b_ord)
            if self.evaluations > MAX_EVALUATIONS:
                return self.unresolved((a_excl, b), None, "evaluation budget exhausted")
            adjacent = b_ord - a_ord <= 1
            mid = self.split_ordinal(a, b, a_ord, b_ord)
            pb = self.point(b)
            if b_certified or self.certified_lower(pb, target):
                candidate = self.assess_candidate(
                    a_excl=a_excl,
                    b=b,
                    enclosure=pb,
                    adjacent=adjacent,
                    gaps=gaps,
                )
                if candidate is not False:
                    if candidate is not True:
                        return candidate
                    stack.append((mid, b_ord, True))
                    stack.append((a_ord, mid, False))
                    continue
            box = self.box(a, b)
            cheap_up = self.union_upper(box)
            if cheap_up < target:
                excluded = self.exclude_interval(
                    a_excl=a_excl,
                    b=b,
                    upper=cheap_up,
                    max_upper=max_upper,
                    gaps=gaps,
                )
                if isinstance(excluded, _MdeRefusal):
                    return excluded
                a_excl, max_upper = excluded
                continue
            enclosed: _Interval | None = None
            if self.variation(box, 0.0, cheap_up) < 1.0:
                enclosed = self.interval(box, cheap_up=cheap_up)
                if not enclosed.midpoint.resolved:
                    if adjacent:
                        return self.unresolved(
                            (a, b),
                            (enclosed.midpoint.lower, enclosed.midpoint.upper),
                            enclosed.midpoint.resolution_reason or _NODE_CEILING,
                        )
                    # The candidate-dependent boundary may become tractable in
                    # a smaller interval; subdivide without consuming its
                    # unresolved midpoint.
                    stack.append((mid, b_ord, False))
                    stack.append((a_ord, mid, False))
                    continue
                if enclosed.upper < target:
                    excluded = self.exclude_interval(
                        a_excl=a_excl,
                        b=b,
                        upper=enclosed.upper,
                        max_upper=max_upper,
                        gaps=gaps,
                    )
                    if isinstance(excluded, _MdeRefusal):
                        return excluded
                    a_excl, max_upper = excluded
                    continue
            resolved = enclosed is not None and (
                enclosed.variation <= enclosed.midpoint.half_width or adjacent
            )
            if not resolved:
                stack.append((mid, b_ord, False))
                stack.append((a_ord, mid, False))
                continue
            assert enclosed is not None
            # Resolved to the evaluator's accuracy yet undecided: refine the
            # point, then the midpoint, while quadrature dominates.
            pb = self.refine_point(b, pb)
            if not pb.resolved:
                return self.unresolved(
                    (a_excl, b), (pb.lower, pb.upper), pb.resolution_reason or _NODE_CEILING
                )
            candidate = self.assess_candidate(
                a_excl=a_excl,
                b=b,
                enclosure=pb,
                adjacent=adjacent,
                gaps=gaps,
            )
            if candidate is True:
                stack.append((mid, b_ord, True))
                stack.append((a_ord, mid, False))
                continue
            if candidate is not False:
                return candidate
            midpoint = enclosed.midpoint
            if self.quadrature_dominates(midpoint):
                enclosed = self.interval(box, nodes=self.doubled(midpoint), cheap_up=cheap_up)
                if not enclosed.midpoint.resolved:
                    if adjacent:
                        return self.unresolved(
                            (a, b),
                            (enclosed.midpoint.lower, enclosed.midpoint.upper),
                            enclosed.midpoint.resolution_reason or _NODE_CEILING,
                        )
                    stack.append((mid, b_ord, False))
                    stack.append((a_ord, mid, False))
                    continue
                if enclosed.upper < target:
                    excluded = self.exclude_interval(
                        a_excl=a_excl,
                        b=b,
                        upper=enclosed.upper,
                        max_upper=max_upper,
                        gaps=gaps,
                    )
                    if isinstance(excluded, _MdeRefusal):
                        return excluded
                    a_excl, max_upper = excluded
                    continue
                if enclosed.variation > enclosed.midpoint.half_width and not adjacent:
                    # The variation term dominates the refined enclosure:
                    # subdivision, not node doubling, is what resolves it.
                    stack.append((mid, b_ord, False))
                    stack.append((a_ord, mid, False))
                    continue
            gaps.append(
                _StraddleGap(a, b, enclosed.lower, enclosed.upper, enclosed.midpoint.half_width)
            )

        return self.no_answer(
            m_max=m_max,
            a_excl=a_excl,
            max_upper=max_upper,
            gaps=gaps,
        )
