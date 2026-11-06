"""Score inference for the difference of two independent binomial proportions.

For ``delta = p_t - p_c`` the Miettinen-Nurminen score statistic is::

    z(delta) = (x_t/n_t - x_c/n_c - delta) / sqrt(V(delta))
    V(delta) = (p_t (1 - p_t) / n_t + p_c (1 - p_c) / n_c) * N / (N - 1),   N = n_c + n_t

where ``(p_c, p_t)`` maximize the two-binomial likelihood subject to ``p_t - p_c = delta``
(Miettinen & Nurminen 1985). ``z`` is referred to the standard Normal and decreases in
``delta``, so the interval ``{delta : |z(delta)| < crit}`` is cut on each side of the observed
difference by bracketed root finding. It lies inside ``[-1, 1]``, stays finite with zero
events, and keeps near-nominal size at sparse counts where the Wald interval does not.
"""

from __future__ import annotations

import math
from typing import Literal

from scipy.optimize import brentq
from scipy.special import ndtr

__all__ = ["score_bounds", "score_p_value", "score_restricted_mle"]

# Absolute endpoint tolerance on [-1, 1]: below the spacing of floats near one.
_ROOT_XTOL = 2.0**-60
# Newton steps from the cubic root rarely exceed three; bisection covers the rest.
_POLISH_STEPS = 64


def _cubic_root(x_c: int, n_c: int, x_t: int, n_t: int, delta: float) -> float:
    """The trigonometric root for ``p_c`` of ``L3 p^3 + L2 p^2 + L1 p + L0`` with
    ``L3 = N``, ``L2 = (n_t + 2 n_c) delta - N - M``, ``L1 = (n_c delta - N - 2 x_c) delta + M``
    and ``L0 = x_c delta (1 - delta)``, ``M = x_c + x_t`` (Miettinen & Nurminen 1985, appendix).
    Rounding degrades it as ``|delta|`` nears one, so it only starts the refinement."""
    total = n_c + n_t
    events = x_c + x_t
    b = ((n_t + 2 * n_c) * delta - total - events) / total
    c = ((n_c * delta - total - 2 * x_c) * delta + events) / total
    d = x_c * delta * (1.0 - delta) / total
    v = (b / 3.0) ** 3 - b * c / 6.0 + d / 2.0
    u = math.copysign(math.sqrt(max((b / 3.0) ** 2 - c / 3.0, 0.0)), v)
    if u == 0.0:
        return -b / 3.0
    cosine = min(max(v / u**3, -1.0), 1.0)
    return 2.0 * u * math.cos((math.pi + math.acos(cosine)) / 3.0) - b / 3.0


def _ratio(count: int, rate: float) -> float:
    """``count / rate`` with a zero count contributing nothing and a positive count at a zero
    (or rounded-negative) rate contributing an infinite likelihood slope."""
    if count == 0:
        return 0.0
    return count / rate if rate > 0.0 else math.inf


def _slope(x_c: int, n_c: int, x_t: int, n_t: int, delta: float, p_c: float) -> float:
    """Derivative of the restricted log-likelihood in ``p_c``; decreasing, since the
    likelihood is concave."""
    return (
        _ratio(x_t, p_c + delta)
        - _ratio(n_t - x_t, (1.0 - p_c) - delta)
        + _ratio(x_c, p_c)
        - _ratio(n_c - x_c, 1.0 - p_c)
    )


def _curvature(x_c: int, n_c: int, x_t: int, n_t: int, delta: float, p_c: float) -> float:
    """Minus the second derivative of the restricted log-likelihood in ``p_c``."""
    rates = (
        (x_t, p_c + delta),
        (n_t - x_t, (1.0 - p_c) - delta),
        (x_c, p_c),
        (n_c - x_c, 1.0 - p_c),
    )
    return math.fsum(
        (count / rate / rate if rate > 0.0 else math.inf) for count, rate in rates if count
    )


def score_restricted_mle(
    x_c: int, n_c: int, x_t: int, n_t: int, delta: float
) -> tuple[float, float]:
    """``(p_c, p_t)`` maximizing the likelihood of ``x_c`` of ``n_c`` and ``x_t`` of ``n_t``
    subject to ``p_t - p_c = delta``, for ``-1 <= delta <= 1``.

    The maximum lies on the boundary of the feasible ``p_c`` range when the likelihood slope
    does not change sign there; otherwise the cubic root is refined by safeguarded Newton
    steps on the likelihood equation, which stay within a shrinking sign bracket.
    """
    low, high = max(0.0, -delta), min(1.0, 1.0 - delta)
    if not _slope(x_c, n_c, x_t, n_t, delta, low) > 0.0:
        p_c = low
    elif not _slope(x_c, n_c, x_t, n_t, delta, high) < 0.0:
        p_c = high
    else:
        p_c = _cubic_root(x_c, n_c, x_t, n_t, delta)
        if not low < p_c < high:
            p_c = 0.5 * (low + high)
        for _ in range(_POLISH_STEPS):
            slope = _slope(x_c, n_c, x_t, n_t, delta, p_c)
            if slope > 0.0:
                low = p_c
            elif slope < 0.0:
                high = p_c
            else:
                break
            step = slope / _curvature(x_c, n_c, x_t, n_t, delta, p_c)
            if abs(step) <= math.ulp(p_c):
                break
            candidate = p_c + step
            if not low < candidate < high:
                candidate = 0.5 * (low + high)
                if not low < candidate < high:
                    break
            p_c = candidate
    return p_c, min(max(p_c + delta, 0.0), 1.0)


def _score_statistic(x_c: int, n_c: int, x_t: int, n_t: int, delta: float) -> float:
    """``z(delta)``; a null outside ``[-1, 1]`` lies beyond every feasible difference, so ``z``
    is ``+inf`` below it and ``-inf`` above. A zero restricted variance occurs only where the
    observed difference equals the null (then 0) or on the boundary of the parameter space
    (then infinite)."""
    if not -1.0 <= delta <= 1.0:
        return math.copysign(math.inf, -delta)
    gap = math.fsum((x_t / n_t, -(x_c / n_c), -delta))
    if gap == 0.0:
        return 0.0
    p_c, p_t = score_restricted_mle(x_c, n_c, x_t, n_t, delta)
    total = n_c + n_t
    variance = (p_t * (1.0 - p_t) / n_t + p_c * (1.0 - p_c) / n_c) * total / (total - 1)
    if not variance > 0.0:
        return math.copysign(math.inf, gap)
    return gap / math.sqrt(variance)


def score_p_value(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    delta: float,
    alternative: Literal["two-sided", "greater", "less"],
) -> float:
    """The score test's p-value against ``p_t - p_c = delta``: ``"greater"`` tests for a
    difference above ``delta``, ``"less"`` below it, and two-sided doubles the smaller tail."""
    z = _score_statistic(x_c, n_c, x_t, n_t, delta)
    if alternative == "greater":
        return float(ndtr(-z))
    if alternative == "less":
        return float(ndtr(z))
    return min(1.0, 2.0 * float(ndtr(-abs(z))))


def _endpoint(
    x_c: int, n_c: int, x_t: int, n_t: int, crit: float, observed: float, end: float
) -> float:
    """The null between ``observed`` and ``end`` where ``|z|`` reaches ``crit``. The search
    runs on ``atan(z)``, which stays finite where ``z`` is infinite at the boundary."""
    if observed == end:
        return end
    target = math.atan(crit if end < observed else -crit)

    def excess(delta: float) -> float:
        return math.atan(_score_statistic(x_c, n_c, x_t, n_t, delta)) - target

    low, high = (end, observed) if end < observed else (observed, end)
    return float(brentq(excess, low, high, xtol=_ROOT_XTOL, maxiter=200))


def score_bounds(x_c: int, n_c: int, x_t: int, n_t: int, crit: float) -> tuple[float, float]:
    """The two-sided score interval for ``p_t - p_c`` at the Normal critical value ``crit``:
    the nulls whose score statistic lies strictly within ``(-crit, crit)``."""
    observed = math.fsum((x_t / n_t, -(x_c / n_c)))
    return (
        _endpoint(x_c, n_c, x_t, n_t, crit, observed, -1.0),
        _endpoint(x_c, n_c, x_t, n_t, crit, observed, 1.0),
    )
