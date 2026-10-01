"""Independent boundary-crossing references sharing no code with
``increment.power.sequential``.

* ``exact_one_look``: closed-form normal tails (one look).
* ``quad_two_look``: two-look conditional-normal integral with
  ``scipy.integrate.quad`` -- the first-look density integrated over the
  surviving interval times the second look's conditional exit probability,
  plus the first-look exits. Reports quad's own error estimates separately.
* ``simpson_walk``: Jennison-Turnbull recursion with Simpson's rule on the
  surviving interval (odd point count, endpoints on the boundary), O(h^4)
  and independent of Gauss-Legendre node placement; the reference for three
  or more looks. Returns the per-look first-exit masses as well, so the
  expected information fraction can be checked from the same recursion.
* ``joint_normal_crossing``: no recursion at all -- the per-look z statistics
  are jointly normal with ``Cov(Z_i, Z_j) = sqrt(t_i / t_j)``, so survival
  through the first ``k`` looks is one rectangle probability of that joint
  law (Genz quasi-Monte-Carlo integration). First-exit masses are
  differences of successive rectangles; a non-recursive reference for the
  either-side exit probability on unequal look schedules. Its upper-only
  first exit is instead a small difference of two large rectangle
  probabilities, so Genz's own integration noise (observed to drift up to
  ~2.6e-7 across SciPy's Genz implementations) dominates the result --
  prefer ``simpson_walk`` for upper-only validation.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

JOINT_NORMAL_ABS_TOL = 2e-7


def exact_one_look(c: float, drift_z: float) -> tuple[float, float]:
    """``(upper_first_exit, either_side_exit)`` of a single look at ``c``."""
    top = float(norm.sf(c - drift_z))
    bottom = float(norm.sf(c + drift_z))
    return top, top + bottom


def quad_two_look(
    bounds: Sequence[float],
    fractions: Sequence[float],
    drift_z: float,
    *,
    epsabs: float = 0.0,
    epsrel: float = 1e-13,
) -> tuple[float, float, float, float]:
    """``(upper, both, quad_err_upper, quad_err_both)`` for two looks."""
    c1, c2 = bounds
    t1, t2 = fractions
    b1 = c1 * math.sqrt(t1)
    b2 = c2 * math.sqrt(t2)
    sd1 = math.sqrt(t1)
    sd2 = math.sqrt(t2 - t1)
    m1 = drift_z * t1
    m2 = drift_z * (t2 - t1)
    top1 = float(norm.sf((b1 - m1) / sd1))
    bot1 = float(norm.sf((b1 + m1) / sd1))

    def f1(s: float) -> float:
        return float(norm.pdf((s - m1) / sd1)) / sd1

    def top2(s: float) -> float:
        return f1(s) * float(norm.sf((b2 - s - m2) / sd2))

    def bot2(s: float) -> float:
        return f1(s) * float(norm.sf((b2 + s + m2) / sd2))

    # Splitting at the mode keeps each piece smooth for the adaptive rule.
    points = [m1] if -b1 < m1 < b1 else None
    up2, err_up = quad(top2, -b1, b1, epsabs=epsabs, epsrel=epsrel, limit=500, points=points)
    lo2, err_lo = quad(bot2, -b1, b1, epsabs=epsabs, epsrel=epsrel, limit=500, points=points)
    return top1 + up2, top1 + bot1 + up2 + lo2, err_up, err_up + err_lo


@dataclass(frozen=True)
class SimpsonWalk:
    upper: float
    both: float
    total: tuple[float, ...]
    surviving: float

    def expected_fraction(self, fractions: Sequence[float]) -> float:
        """``E[T]`` from the per-look first-exit masses: every look's mass
        is charged its own fraction and the survivors the final one."""
        exits = math.fsum(m * t for m, t in zip(self.total, fractions, strict=True))
        return exits + self.surviving * fractions[-1]


def simpson_walk(
    bounds: Sequence[float], fractions: Sequence[float], drift_z: float, points: int = 2001
) -> SimpsonWalk:
    assert points % 2 == 1
    w_ref = np.ones(points)
    w_ref[1:-1:2] = 4.0
    w_ref[2:-1:2] = 2.0
    prev_t = 0.0
    f = nodes = weights = None
    upper = 0.0
    both = 0.0
    total: list[float] = []
    for c, t in zip(bounds, fractions, strict=True):
        dt = t - prev_t
        b = c * math.sqrt(t)
        sd = math.sqrt(dt)
        mu = drift_z * dt
        new_nodes = np.linspace(-b, b, points)
        h = new_nodes[1] - new_nodes[0]
        new_w = w_ref * h / 3.0
        if f is None:
            top = float(norm.sf((b - mu) / sd))
            bot = float(norm.sf((b + mu) / sd))
            dens = norm.pdf((new_nodes - mu) / sd) / sd
        else:
            assert nodes is not None and weights is not None
            weighted = weights * f
            dens = np.empty_like(new_nodes)
            # Bound peak memory: the 8001-point tiny-tail references otherwise
            # allocate an 8001x8001 kernel (>500 MiB) per xdist worker.
            for start in range(0, points, 256):
                stop = min(points, start + 256)
                z = (new_nodes[start:stop, None] - nodes[None, :] - mu) / sd
                kernel = np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi))
                dens[start:stop] = kernel @ weighted
            top = float(np.dot(weights * f, norm.sf((b - nodes - mu) / sd)))
            bot = float(np.dot(weights * f, norm.sf((b + nodes + mu) / sd)))
        upper += top
        both += top + bot
        total.append(top + bot)
        f, nodes, weights, prev_t = dens, new_nodes, new_w, t
    assert weights is not None and f is not None
    return SimpsonWalk(upper, both, tuple(total), float(np.dot(weights, f)))


@dataclass(frozen=True)
class JointNormalCrossing:
    both: float
    upper: float
    exits: tuple[float, ...]
    surviving: float

    def expected_fraction(self, fractions: Sequence[float]) -> float:
        exits = math.fsum(m * t for m, t in zip(self.exits, fractions, strict=True))
        return exits + self.surviving * fractions[-1]


def _rectangle(upper: np.ndarray, lower: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    from scipy.stats import multivariate_normal

    kwargs = {
        "mean": mean,
        "cov": cov,
        "lower_limit": lower,
        "abseps": 1e-10,
        "releps": 1e-10,
        "maxpts": 4_000_000,
    }
    if "rng" in inspect.signature(multivariate_normal.cdf).parameters:
        kwargs["rng"] = np.random.default_rng(0)
    return float(multivariate_normal.cdf(upper, **kwargs))


def joint_normal_crossing(
    bounds: Sequence[float], fractions: Sequence[float], drift_z: float
) -> JointNormalCrossing:
    """Either-side and upper first-exit probabilities from the joint law of
    ``Z_k = S_k / sqrt(t_k)``: mean ``drift_z * sqrt(t_k)`` and correlation
    ``sqrt(t_i / t_j)`` for ``i <= j``. Supported SciPy versions use
    different Genz integrations; comparisons allow ``JOINT_NORMAL_ABS_TOL``."""
    t = np.asarray(fractions, dtype=float)
    c = np.asarray(bounds, dtype=float)
    mean = drift_z * np.sqrt(t)
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    surviving = [1.0]
    upper = 0.0
    for k in range(1, t.size + 1):
        surviving.append(_rectangle(c[:k], -c[:k], mean[:k], cov[:k, :k]))
        # Survive the first k-1 looks, then finish at or below +c_k: the
        # complement within the survivors is the upper first exit at look k.
        below = np.concatenate((-c[: k - 1], [-np.inf]))
        upper += surviving[k - 1] - _rectangle(c[:k], below, mean[:k], cov[:k, :k])
    exits = tuple(surviving[k - 1] - surviving[k] for k in range(1, t.size + 1))
    return JointNormalCrossing(1.0 - surviving[-1], upper, exits, surviving[-1])
