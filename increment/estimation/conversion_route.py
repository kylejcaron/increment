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
reaches ``dense_min_count(tail_alpha)``. The rule reads those counts and the row's own
tail allocation (``alpha / 2`` per tail two-sided, ``alpha`` directional) and nothing
else: it is fixed before inference, never compares an interval or a p-value, and is the
same whether or not the row belongs to a multiplicity family.

A family reads a row's p-value at levels below the row's own, where the delta p-value
is not validated. A row the rule routes asymptotic therefore reports the composite
``composite_p_value``: its delta p-value when that is at least ``U``, the smallest level
at which its own counts are dense (``dense_level``, the rule inverted count by count),
and otherwise the smaller of its exact p-value (the finite-sample route's, read at the
nominal tail) and ``U``. Read at any level ``u``, the composite rejects only where the
pipeline routed by counts at ``u`` rejects: by the delta row at ``u`` on the counts
dense at ``u``, by the finite-sample test at ``u`` elsewhere (``docs/limitations.md``).

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

import functools
import math
import re
import struct
from collections.abc import Callable
from fractions import Fraction
from typing import Literal, NoReturn

import numpy as np

from increment._literals import ConversionInference
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

#: Prefixes of the note a delta-route row carries when its p-value is not its delta p-value
#: (`composite_p_value`): the exact p-value read in its place, or the dense level itself.
ESCALATED_NOTE_PREFIX = "conversion p-value escalated to the exact p-value"
CAPPED_NOTE_PREFIX = "conversion p-value capped at its dense level"
_COMPOSITE_PREFIXES = (ESCALATED_NOTE_PREFIX, CAPPED_NOTE_PREFIX)
_NOTE_SEPARATOR = re.compile(r"( \| |; )")


def without_composite_note(note: str | None) -> str | None:
    """*note* with any composite p-value note removed, every other disclosure and its separator
    intact: a row reissued as an exact binomial set reports the exact p-value at that level,
    so the note explaining its composite no longer applies while its other qualifications do."""
    if not note:
        return None
    pieces = _NOTE_SEPARATOR.split(note)
    kept: list[str] = []
    for index in range(0, len(pieces), 2):
        if pieces[index].startswith(_COMPOSITE_PREFIXES):
            continue
        if kept:
            kept.append(pieces[index - 1])
        kept.append(pieces[index])
    return "".join(kept) or None


def fcr_route_tail(alternative: str, fcr_alpha: float | Fraction) -> float:
    """The one-sided tail the count rule routes a selected row's reissue at: the tail of the
    interval that row finally reports at the corrected level ``fcr_alpha`` -- half of it for a
    central two-sided interval, all of it for the single open bound a directional row reports
    (`open_bound_from_two_sided_at_target` opens the two-sided parent at that full tail)."""
    return float(fcr_alpha) / 2.0 if alternative == "two-sided" else float(fcr_alpha)


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
    *, cluster: str | None, prior_present: bool, sequential: bool
) -> str | None:
    """Why ``conversion_inference="finite_sample"`` cannot serve one conversion or retention
    metric's request, or ``None`` when it can: the finite-sample route is the
    independent-binomial test inversion of an unclustered rate at a fixed horizon, with no
    informative prior. A metric that is not a conversion or retention rate is the shared
    ``increment._finite_sample_refusals`` hazard, refused before this is asked."""
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
    from scipy.stats import norm

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


#: Counts from which every representable tail is dense: ``dense_min_count`` of the smallest
#: positive float, past which the closed-form inversion below would evaluate the rule at a tail
#: of zero. A contrast with such counts has a dense level of zero and is never escalated.
@functools.cache
def _dense_ceiling() -> int:
    return dense_min_count(math.nextafter(0.0, 1.0))


def __getattr__(name: str) -> int:
    if name == "_DENSE_CEILING":
        return _dense_ceiling()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _float_bisect(lo: float, hi: float, dense: Callable[[float], bool]) -> float:
    """The least float in ``(lo, hi]`` at which ``dense`` holds, ``dense`` being false at
    ``lo`` and true at ``hi`` and monotone between (up to the isolated one-float steps the
    caller settles afterwards), by bisection over the ordered bit patterns of the floats."""
    pack = struct.Struct("<d")
    unpack = struct.Struct("<q")
    low, high = unpack.unpack(pack.pack(lo))[0], unpack.unpack(pack.pack(hi))[0]
    while high - low > 1:
        mid = (low + high) // 2
        if dense(pack.unpack(unpack.pack(mid))[0]):
            high = mid
        else:
            low = mid
    return pack.unpack(unpack.pack(high))[0]


def dense_level_tail(smallest: int) -> float | None:
    """The least one-sided tail at which a contrast whose smallest per-arm success or failure
    count is ``smallest`` is dense: the smallest float ``t`` with ``dense_min_count(t) <=
    smallest``, so the contrast is dense at every tail from it up and at none below it
    (``dense_min_count`` does not increase with the tail on ``(0, 1/2)``, apart from isolated
    floats where the normal quantile's own rounding steps the wrong way, which
    ``calibration.conversion_route.dense_level_check`` counts). The boundary is bisected over
    the floats of ``(0, 1/2)`` against the rule itself and then settled float by float on both
    sides, so it is the rule's own boundary, never a closed form's (the closed-form inversion
    of the ``z ** 4`` law lands on a plateau of the rule that can be billions of floats wide at
    large counts, and underflows to zero near the ceiling). ``None`` below the floor, where
    no tail is dense; ``0.0`` from ``_DENSE_CEILING`` on, where every representable tail is.
    Every tail evaluated lies in ``(0, 1/2)``, the domain on which the rule is monotone (it is
    not above one half)."""
    if smallest < _DENSE_FLOOR:
        return None
    if smallest >= _dense_ceiling():
        return 0.0
    tail = _float_bisect(
        math.nextafter(0.0, 1.0),
        math.nextafter(0.5, 0.0),
        lambda t: dense_min_count(t) <= smallest,
    )
    while dense_min_count(tail) > smallest:
        tail = math.nextafter(tail, 1.0)
    while (below := math.nextafter(tail, 0.0)) > 0.0 and dense_min_count(below) <= smallest:
        tail = below
    assert 0.0 < tail < 0.5, "the inversion never leaves the rule's monotone domain"
    return tail


def dense_level(x_c: int, n_c: int, x_t: int, n_t: int, *, alternative: str) -> float | None:
    """``U``: the smallest level, in the convention of the level a p-value is compared with
    (``alpha`` two-sided, the single directional tail otherwise), at which the four counts are
    dense. Twice ``dense_level_tail`` two-sided (an exact doubling, so ``alpha >= U`` is
    ``alpha / 2 >= dense_level_tail`` float for float), the tail itself directional. ``None``
    where no level is dense."""
    tail = dense_level_tail(min(x_c, n_c - x_c, x_t, n_t - x_t))
    if tail is None:
        return None
    return 2.0 * tail if alternative == "two-sided" else tail


def dense_level_holds(
    x_c: int, n_c: int, x_t: int, n_t: int, *, level: float, alternative: str
) -> bool:
    """Whether a persisted level matches the count rule's level under ``alternative``.

    The exact boundary check is the fast path. When it misses, the normal inverse may differ
    by a few floating-point units between SciPy builds, so compare the independently inverted
    level with a consistency tolerance (not a statistical tolerance).
    """
    smallest = min(x_c, n_c - x_c, x_t, n_t - x_t)
    if level == 0.0 and smallest >= _dense_ceiling():
        return True
    tail = level / 2.0 if alternative == "two-sided" else level
    if 0.0 < tail < 0.5 and smallest < _dense_ceiling() and dense_min_count(tail) <= smallest:
        below = math.nextafter(tail, 0.0)
        if below == 0.0 or dense_min_count(below) > smallest:
            return True
    expected = dense_level(x_c, n_c, x_t, n_t, alternative=alternative)
    return expected is not None and math.isclose(level, expected, rel_tol=1e-12, abs_tol=1e-15)


def composite_p_value(p_delta: float, dense_level: float, escalated_p_value: float | None) -> float:
    """The p-value a row the rule routed asymptotic reports: ``p_delta`` when that is at least
    ``dense_level`` (the row is dense at every level that could reject it), otherwise the
    smaller of its exact p-value and ``dense_level``, the level itself when no exact p-value
    was read (the exact route unavailable, or its p-value unable to fall below the level)."""
    if p_delta >= dense_level:
        return p_delta
    if escalated_p_value is None:
        return dense_level
    return min(escalated_p_value, dense_level)


def composite_exact_reading(
    counts: tuple[int, int, int, int],
    *,
    dense_level: float,
    alpha: float,
    null_lift: float,
    alternative: str,
) -> tuple[float | None, str | None]:
    """Read exact count evidence below its dense level, retaining any route refusal."""
    from increment.estimation import binomial_rr

    beta = binomial_rr.nuisance_beta(alpha)
    if dense_level <= beta:
        return None, None
    x_c, n_c, x_t, n_t = counts
    if max(n_c, n_t) > binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE:
        return None, "estimation.binomial.finite_sample_arm_ceiling_exceeded"
    tail = alpha / 2.0 if alternative == "two-sided" else alpha
    try:
        exact = binomial_rr.null_p_value(
            1.0 + null_lift, x_c, n_c, x_t, n_t, beta, alternative=alternative, tail=tail
        )
    except binomial_rr.BinomialDataError as exc:
        return None, exc.code
    return exact, None


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


def _mass_between(n: int, p: float, lo: int, hi: int) -> float:
    """``P(lo <= X <= hi)`` for ``X ~ Bin(n, p)``, zero for an empty range, taken from the tail
    the range sits in so the two tails subtracted are the smaller ones."""
    from increment.estimation import binomial_rr

    lo, hi = max(lo, 0), min(hi, n)
    if lo > hi:
        return 0.0
    if lo > n * p:
        below = float(binomial_rr._fast_binom_sf(np.asarray(lo - 1), n, p))
        above = float(binomial_rr._fast_binom_sf(np.asarray(hi), n, p))
        return max(0.0, below - above)
    upper = float(binomial_rr._fast_binom_cdf(np.asarray(hi), n, p))
    lower = float(binomial_rr._fast_binom_cdf(np.asarray(lo - 1), n, p)) if lo > 0 else 0.0
    return max(0.0, upper - lower)


def unrouted_share(
    n_c: int, n_t: int, p_c: float, p_t: float, *, floor: int, control_below: int | None = None
) -> float:
    """Probability that the runtime rule keeps a draw on the finite-sample route: some arm's
    success or failure count falls below ``floor``. The complement of ``routed_share``, summed
    from the arms' outside masses so it keeps its relative precision where it is small.

    With ``control_below`` only draws whose control count is below it count: those the
    finite-sample route refuses when it decides from that count on. The control arm's own
    off-route mass below it (below ``floor``, or above ``n_c - floor`` yet below the count) is
    taken whole, its routed mass below the count times the treatment arm's off-route mass; each
    term is a tail or a range read from its nearer tail, never a total less an inside sum.
    """
    outside_t = _outside_mass(n_t, p_t, floor)
    if control_below is None:
        outside_c = _outside_mass(n_c, p_c, floor)
        return min(1.0, outside_c + outside_t - outside_c * outside_t)
    control_off = _mass_between(n_c, p_c, 0, min(control_below, floor) - 1) + _mass_between(
        n_c, p_c, max(n_c - floor + 1, floor), control_below - 1
    )
    control_in = _mass_between(n_c, p_c, floor, min(control_below - 1, n_c - floor))
    return min(1.0, control_off + control_in * outside_t)


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
