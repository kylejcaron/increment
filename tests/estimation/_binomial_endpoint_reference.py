"""Finer-resolution reference for the exact binomial endpoint search.

The reference searches the same evaluated p-envelope (`binomial_rr.p_plus` / `p_minus`) as
production, but with a different inversion (doubling, then arithmetic bisection) taken to a
relative bracket of ``2**-40``, so a reported endpoint can be compared against a near-exact
crossing instead of a pinned bit pattern.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

from increment.estimation import binomial_rr as brr
from increment.estimation.results import BinomialConfidenceSet

REFERENCE_RELATIVE_WIDTH = 2.0**-40

#: Slack for the float arithmetic that converts lift bounds back to risk ratios.
_CONVERSION_SLACK = 1e-12


def crossing_bracket(
    f: Callable[[float], float], target: float, *, increasing: bool
) -> tuple[float, float]:
    """``(lo, hi)`` around the crossing of ``f`` at *target*, ``hi - lo <= 2**-40 * hi``.

    *increasing*: ``f(lo) < target <= f(hi)``. Otherwise ``f(lo) >= target > f(hi)``. A
    crossing at the origin returns ``(0.0, 0.0)``.
    """

    def beyond(r: float) -> bool:
        return (f(r) >= target) == increasing

    if beyond(0.0):
        return 0.0, 0.0
    lo, hi = 0.0, 1.0
    while not beyond(hi):
        lo, hi = hi, hi * 2.0
    while hi - lo > REFERENCE_RELATIVE_WIDTH * hi:
        mid = (lo + hi) / 2.0
        if beyond(mid):
            hi = mid
        else:
            lo = mid
    return lo, hi


def assert_endpoints_contain_finer_reference(
    lower: float,
    upper: float | None,
    *,
    counts: tuple[int, int, int, int],
    tail_alpha: float,
    geometry: Literal["central", "lower_bound", "upper_bound"],
    beta: float,
) -> None:
    """Risk-ratio endpoints enclose the finer crossings, within the declared resolution.

    Containment compares against the reference bracket's confirmed-inside end (which lies at
    or beyond the crossing), and the gap against its confirmed-outside end (at or before it),
    so neither assertion depends on where inside the reference bracket the crossing falls.
    The structural floor (``lower == 0``) and unbounded ceiling (``upper is None``) have no
    crossing to compare.
    """
    x_c, n_c, x_t, n_t = counts
    target = tail_alpha / 2.0 if geometry == "central" else tail_alpha
    tau = brr._endpoint_tolerance(brr._count_scale(x_c, n_c, x_t, n_t))

    if geometry != "upper_bound":
        ref_lo, ref_hi = crossing_bracket(
            lambda r: brr.p_plus(r, x_c, n_c, x_t, n_t, beta), target, increasing=True
        )
        assert lower <= ref_hi, (lower, ref_hi)
        if ref_lo > 0.0:
            assert lower > 0.0
            assert math.log(ref_lo / lower) <= tau + _CONVERSION_SLACK, (lower, ref_lo, tau)

    if geometry != "lower_bound" and upper is not None:
        ref_lo, ref_hi = crossing_bracket(
            lambda r: brr.p_minus(r, x_c, n_c, x_t, n_t, beta), target, increasing=False
        )
        assert upper >= ref_lo, (upper, ref_lo)
        if ref_hi > 0.0:
            assert math.log(upper / ref_hi) <= tau + _CONVERSION_SLACK, (upper, ref_hi, tau)


def assert_set_contains_finer_reference(bset: BinomialConfidenceSet) -> None:
    """The persisted lift-scale set encloses the finer crossings of its own p-envelope."""
    assert_endpoints_contain_finer_reference(
        bset.lower + 1.0,
        None if bset.upper is None else bset.upper + 1.0,
        counts=(bset.x_c, bset.n_c, bset.x_t, bset.n_t),
        tail_alpha=bset.decision_alpha,
        geometry=bset.geometry,
        beta=bset.nuisance_beta,
    )
