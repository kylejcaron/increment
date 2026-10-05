"""Route selection for unadjusted conversion and retention inference.

Two routes serve an unadjusted, unit-grain, fixed-horizon conversion or retention
contrast, and a count-only rule picks between them before any interval is built:

* ``asymptotic`` -- the delta-method log risk ratio against a Welch-Satterthwaite
  ``t`` reference that every unadjusted mean metric uses (``reference_kind="t"``,
  ``scale="log"``). It carries no finite-sample guarantee.
* ``finite_sample`` -- the Berger-Boos independent-binomial test inversion in
  ``binomial_rr`` (``reference_kind="binomial"``), valid at every count and
  therefore the route for sparse cells.

``Method.conversion_inference`` is ``"auto"`` (the default) or ``"finite_sample"``.
``"finite_sample"`` always takes the finite-sample route. ``"auto"`` takes the
asymptotic route only when every one of the four per-arm success and failure counts
reaches ``dense_min_count(tail_alpha)``. The rule reads those counts and the tail
allocation (``alpha / 2`` per tail two-sided, ``alpha`` directional) and nothing
else: it is fixed before inference and never compares an interval or a p-value.

The log risk ratio's Wald interval is skewed at small counts: a count ``s`` gives
the log-scale statistic skewness of order ``s ** -0.5``, which moves a one-sided
tail at standard normal quantile ``z`` by roughly ``(z ** 2 - 1) * phi(z) / Q(z)``
times that skewness, a relative error that grows like ``z ** 3``. Holding that error
inside ``scientific_delta(tail)`` (a tenth of the tail below 0.05, 0.005 at and above
it) therefore needs a count that grows like ``z ** 6`` asymptotically. Over the tails
this package computes, the measured requirement grows more slowly, and
``dense_min_count`` is a ``z ** 4`` envelope of it: its two constants keep it at least
1.25 times that requirement at every production tail, as the boundary coverage scan in
``calibration.conversion_route`` measures it (``docs/limitations.md`` tabulates both).

Every routed-asymptotic cell has all four counts at least ``dense_min_count >= 9``,
so the combined log-scale standard error is below ``sqrt(2 / 9) < 0.5`` and neither
the ``log_se >= 0.5`` nor the zero-variance guard of ``infer_lift`` can fire.
"""

from __future__ import annotations

import math
from typing import Literal, NoReturn

import numpy as np
from scipy.stats import norm

from increment._literals import ConversionInference
from increment.compatibility import _conservative_divide
from increment.errors import InvalidRequestError, refusals, refuse

Route = Literal["asymptotic", "finite_sample"]

#: Planning's classification of a rate pair against the runtime rule: every draw
#: routes asymptotic (``dense``), none does (``sparse``), or the draw decides
#: (``borderline``).
PlanningRoute = Literal["dense", "sparse", "borderline"]

#: Largest probability of the unlikely route at which a plan counts as certain.
PLANNING_ROUTE_CERTAINTY = 1e-6

#: Counts below which the log-scale delta method is not admitted at all: with every
#: count at least this, the combined log standard error is below ``sqrt(2 / 9)``.
_GUARD_FLOOR = 9

#: ``dense_min_count(tail) = max(_DENSE_FLOOR, ceil(_DENSE_SLOPE * z ** 4))``, ``z = Phi^{-1}(1 - tail)``:
#: at least 1.25 times the requirement ``calibration.conversion_route`` measures at each production
#: tail (``docs/limitations.md``); the floor is set by the 0.1 tail, the slope by the 0.0005 tail.
#: A tail below 0.0005 is extrapolated by the same formula, not measured.
_DENSE_FLOOR = 412
_DENSE_SLOPE = 145


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.binomial.finite_sample_unavailable": (
            "conversion_inference='finite_sample' is unavailable for metric {metric!r}: "
            "{reason}. The finite-sample route is the independent-binomial test inversion "
            "for an unadjusted, unclustered, fixed-horizon conversion or retention "
            "contrast without an informative prior; use conversion_inference='auto' "
            "(the default), or drop the incompatible setting"
        ),
    },
)
FINITE_SAMPLE_UNAVAILABLE = _REFUSALS["estimation.binomial.finite_sample_unavailable"]


def refuse_finite_sample_unavailable(metric: str, reason: str) -> NoReturn:
    """Refuse an explicit ``finite_sample`` request on a contrast the finite-sample
    route cannot serve; ``reason`` names the incompatible setting."""
    refuse(FINITE_SAMPLE_UNAVAILABLE, metric=metric, reason=reason)


def finite_sample_blocker(
    metric_type: str, *, cluster: str | None, prior_present: bool, sequential: bool
) -> str | None:
    """Why ``conversion_inference="finite_sample"`` cannot serve one metric's request,
    or ``None`` when it can: the finite-sample route is the independent-binomial test
    inversion of an unclustered conversion or retention rate at a fixed horizon, with
    no informative prior."""
    if metric_type not in ("conversion", "retention"):
        return (
            f"the metric type is {metric_type!r}, and the finite-sample route serves "
            "conversion and retention rates only"
        )
    if cluster is not None:
        return (
            f"the units are clustered by {cluster!r}, which the independent-binomial test "
            "inversion does not model"
        )
    if prior_present:
        return (
            "an informative prior is set, and a test inversion has no posterior a prior "
            "could act on"
        )
    if sequential:
        return (
            "sequential inference is requested, and the finite-sample route is fixed-horizon only"
        )
    return None


def dense_min_count(tail_alpha: float) -> int:
    """Least per-arm success and failure count at which ``auto`` takes the
    asymptotic route for a one-sided tail allocation ``tail_alpha`` in ``(0, 1)``."""
    z = float(norm.isf(tail_alpha))
    return max(_GUARD_FLOOR, _DENSE_FLOOR, math.ceil(_DENSE_SLOPE * z**4))


def route_for_counts(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    *,
    tail_alpha: float,
    mode: ConversionInference,
) -> Route:
    """The route an unadjusted conversion contrast takes, from its four counts.

    ``mode="finite_sample"`` is always finite-sample. Under ``"auto"`` the route is
    asymptotic iff the smallest of ``x_c``, ``n_c - x_c``, ``x_t`` and ``n_t - x_t``
    is at least ``dense_min_count(tail_alpha)``. A tail allocation the delta method
    cannot resolve (not strictly positive) takes the finite-sample route, which then
    refuses it by its own coded reason.
    """
    if mode == "finite_sample" or not tail_alpha > 0.0:
        return "finite_sample"
    smallest = min(x_c, n_c - x_c, x_t, n_t - x_t)
    return "asymptotic" if smallest >= dense_min_count(tail_alpha) else "finite_sample"


def family_route_alpha(q: float, hypotheses: int) -> float | None:
    """The smallest level a BH family of ``hypotheses`` decides a p-value at, in ``alpha``'s
    convention: ``q / hypotheses`` rounded downward exactly as the selection threshold is, so a
    row is never routed at a looser level than the one its p-value is read at. ``None`` for a
    family with no hypotheses."""
    return _conservative_divide(q, hypotheses) if hypotheses > 0 else None


def _outside_mass(n: int, p: float, m: int) -> float:
    """``P(X < m or X > n - m)`` for ``X ~ Bin(n, p)``, summed from the two tails so that
    it keeps its relative precision when small. An arm too small to hold ``m`` successes
    and ``m`` failures lies outside with certainty."""
    from increment.estimation import binomial_rr

    if n < 2 * m:
        return 1.0
    below = float(binomial_rr._fast_binom_cdf(np.asarray(m - 1), n, p))
    above = float(binomial_rr._fast_binom_sf(np.asarray(n - m), n, p))
    return min(1.0, below + above)


def _routed(outside_c: float, outside_t: float) -> float:
    """Probability that every count of independent arms reaches the dense band, from each arm's
    outside mass: the product of the arms' inside masses, which keeps its relative precision
    where it is small."""
    return (1.0 - outside_c) * (1.0 - outside_t)


def _planning_class(outside_c: float, outside_t: float) -> PlanningRoute:
    """Classify a plan from each arm's probability of falling outside the dense band."""
    # The complement of both arms inside, summed without cancellation.
    outside = outside_c + outside_t - outside_c * outside_t
    if outside <= PLANNING_ROUTE_CERTAINTY:
        return "dense"
    if _routed(outside_c, outside_t) <= PLANNING_ROUTE_CERTAINTY:
        return "sparse"
    return "borderline"


def routed_share(n_c: int, n_t: int, p_c: float, p_t: float, *, tail_alpha: float) -> float:
    """Probability that the runtime rule takes the asymptotic route for arm sizes ``(n_c, n_t)``
    and true rates ``(p_c, p_t)``: ``P(m <= X_c <= n_c - m) * P(m <= X_t <= n_t - m)`` with
    ``m = dense_min_count(tail_alpha)`` and independent binomial arms. ``planning_route``
    classifies a plan from this probability."""
    m = dense_min_count(tail_alpha)
    return _routed(_outside_mass(n_c, p_c, m), _outside_mass(n_t, p_t, m))


def planning_route(
    n_c: int,
    n_t: int,
    p_c: float,
    p_t: float,
    *,
    tail_alpha: float,
    mode: ConversionInference,
) -> PlanningRoute:
    """Which route a plan at arm sizes ``(n_c, n_t)`` and true rates ``(p_c, p_t)`` takes.

    The runtime rule reads the four realised counts, so the plan's probability of the
    asymptotic route is ``routed_share``. A plan is ``dense`` when that probability is at
    least ``1 - PLANNING_ROUTE_CERTAINTY``, ``sparse`` when it is at most
    ``PLANNING_ROUTE_CERTAINTY``, and ``borderline`` between. ``mode="finite_sample"`` is
    always ``sparse``: the runtime never leaves the finite-sample route.
    """
    if mode == "finite_sample":
        return "sparse"
    m = dense_min_count(tail_alpha)
    return _planning_class(_outside_mass(n_c, p_c, m), _outside_mass(n_t, p_t, m))


def dense_extent(
    n_c: int,
    n_t: int,
    p_c: float,
    p_lo: float,
    p_hi: float,
    *,
    tail_alpha: float,
    mode: ConversionInference,
) -> tuple[bool, bool]:
    """``(some, every)``: whether some, and whether every, treatment rate in
    ``[p_lo, p_hi]`` is ``dense`` at control rate ``p_c``.

    ``P(m <= X <= n - m)`` is increasing in the rate up to one half and decreasing after
    it (its derivative has the sign of ``(1 - 2 p)``), so over an interval its minimum
    sits at an end and its maximum at the rate nearest one half. A classification that
    holds on a whole interval therefore needs only those points.
    """
    if mode == "finite_sample":
        return False, False
    m = dense_min_count(tail_alpha)
    outside_c = _outside_mass(n_c, p_c, m)

    def dense(p_t: float) -> bool:
        return _planning_class(outside_c, _outside_mass(n_t, p_t, m)) == "dense"

    return dense(min(max(0.5, p_lo), p_hi)), dense(p_lo) and dense(p_hi)
