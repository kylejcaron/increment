"""Rate-aware Chernoff cut of a binomial count's support.

For ``X ~ Bin(n, q)`` and the Bernoulli relative entropy
``D(x || q) = x ln(x/q) + (1-x) ln((1-x)/(1-q))``, the Chernoff-Hoeffding bounds are::

    P(X <= m) <= exp(-n D(m/n || q))   for m/n <= q,
    P(X >= m) <= exp(-n D(m/n || q))   for m/n >= q.

``D(x || q)`` rises with ``q`` while ``q > x`` and falls with ``q`` while ``q < x``. Over a
nuisance interval ``[q_lo, q_hi]`` the mass below a count is therefore largest at ``q_lo``
and the mass above a count at ``q_hi``, so one cut per end covers every rate in the
interval at once. The kept range spans about ``sqrt(n q (1-q))`` counts around ``n q``,
not the ``sqrt(n)`` of a rate-blind bound.

A cut is certified in float64: a count is cut only when a lower bound on
``n D(m/n || q)``, built from the evaluation error derived in ``exponent_lower_bound``,
reaches ``-ln(tail)`` rounded up for the error of ``ln``.
"""

from __future__ import annotations

import math
import sys

_EPS = 2.0**-52  # float64 epsilon: one unit in the last place is at most this relative error

#: ``2**-1022``: a float quotient below it is subnormal and rounds by more than ``_EPS / 2``
#: of itself.
_MIN_NORMAL = sys.float_info.min

#: Relative error assumed of ``math.log`` and ``math.log1p``, in units of ``_EPS``. glibc and
#: Apple libm stay within one unit in the last place; four leaves headroom, and
#: ``tests/estimation/test_binomial_rr.py::TestChernoffSupportCut`` checks the bound's
#: tightness against exact decimal arithmetic.
_LIBM_ULPS = 4.0

#: One unit of the smallest subnormal, ``2**-1074``, times ``2 * _LIBM_ULPS + 2``: the absolute
#: error of the summands per unit of ``n`` when they underflow (see ``exponent_lower_bound``).
_SUBNORMAL_ALLOWANCE = (2.0 * _LIBM_ULPS + 2.0) * 2.0**-1074


def _log_ratio(x: float, q: float) -> tuple[float, float]:
    """``ln(x/q)`` for ``x > 0`` and ``q > 0``, with the absolute error of ``x * ln(x/q)`` per
    unit of ``x`` in units of ``_EPS``, as derived in ``exponent_lower_bound``.

    The logarithm of the quotient is used while the float quotient is a normal number; once it
    overflows or turns subnormal the difference of the logarithms is.
    """
    ratio = x / q
    if _MIN_NORMAL <= ratio < math.inf:
        log_ratio = math.log(ratio)
        return log_ratio, (_LIBM_ULPS + 1.0) * abs(log_ratio) + 1.0
    log_x = math.log(x)
    log_q = math.log(q)
    return log_x - log_q, (_LIBM_ULPS + 2.0) * (abs(log_x) + abs(log_q))


def exponent_lower_bound(n: int, x: float, q: float) -> float:
    """A float64 lower bound on ``n * D(x || q)`` for ``0 <= x <= 1`` and ``0 <= q <= 1``.

    ``D`` is evaluated as ``x ln(x/q) + (1-x) (log1p(-x) - log1p(-q))``; it is 0 at ``x = q``
    and infinite where ``x > 0`` meets ``q = 0`` or ``x < 1`` meets ``q = 1``. Write
    ``u = _EPS / 2`` for the rounding unit and ``L = _LIBM_ULPS * _EPS`` for the relative
    error of ``ln``/``log1p``. To first order the summands' absolute errors are

    * ``x ln(x/q)`` as ``x ln(fl(x/q))``, used while the float quotient is a normal number, so
      that it rounds by at most ``1.01u`` of itself: ``x (L + 1.01u) |ln(x/q)| + 1.02u x``. The
      quotient's rounding is an absolute error of ``ln`` that does not vanish as ``x -> q``,
      hence the second term.
    * ``x ln(x/q)`` as ``x (ln x - ln q)``, used once the quotient overflows or is subnormal,
      which needs ``|ln(x/q)| >= 708``: ``x (L + 2.01u) (|ln x| + |ln q|)``. Each logarithm
      rounds by ``L`` of itself and the difference and the product by ``u`` of themselves, so
      every error scales with the logarithms' magnitudes. Both operands lie in ``(0, 1]`` and
      neither is below ``2**-1074``, so the sum is ``2 max(|ln x|, |ln q|) - |ln(x/q)|``, at
      most ``2 * 744.5 - |ln(x/q)|``, which is at most ``1.11 |ln(x/q)|``: this bound is as
      tight as the quotient form's. Near ``x = q`` the form would cancel down to its own
      error, which is why the quotient form is kept wherever it holds.
    * ``(1-x)(log1p(-x) - log1p(-q))``: ``(1-x) (L + 3.1u) (|ln(1-x)| + |ln(1-q)|)``. The
      difference rounds against the logarithms' magnitudes, not against their difference.

    The code rounds these coefficients up to ``_LIBM_ULPS + 1`` and ``1``, ``_LIBM_ULPS + 2``,
    and ``_LIBM_ULPS + 2`` in units of ``_EPS``, which also covers evaluating the bound itself in
    float64. The sum and the product by ``n`` each round by at most ``u`` of their value;
    ``2 * _EPS`` of it is allowed.

    Below ``2**-1022`` a result rounds by an absolute ``2**-1075`` instead, and a libm result
    is off by up to ``_LIBM_ULPS`` units of ``2**-1074``. That reaches the sum only through
    operands below about ``2**-969``: the two summand products (each scaled by ``n``), the two
    ``log1p`` values that enter ``second``, the product by ``n``, and the underflow of the
    allowance itself. They total at most ``(2 * _LIBM_ULPS + 1) * n + 2`` units of ``2**-1074``;
    ``_SUBNORMAL_ALLOWANCE * (n + 1)`` is added to the error, which is rounded up from that.
    The result is rounded down.
    """
    if q == 0.0:
        return math.inf if x > 0.0 else 0.0
    if q == 1.0:
        return math.inf if x < 1.0 else 0.0
    first = first_error = second = second_error = 0.0
    if x > 0.0:
        log_ratio, log_ratio_error = _log_ratio(x, q)
        first = x * log_ratio
        first_error = x * log_ratio_error
    if x < 1.0:
        log_x = math.log1p(-x)
        log_q = math.log1p(-q)
        second = (1.0 - x) * (log_x - log_q)
        second_error = (_LIBM_ULPS + 2.0) * (1.0 - x) * (abs(log_x) + abs(log_q))
    value = n * (first + second)
    error = _EPS * (n * (first_error + second_error) + 2.0 * value) + _SUBNORMAL_ALLOWANCE * (n + 1)
    return math.nextafter(value - error, -math.inf)


def _below(n: int, q_lo: float, target: float) -> int:
    """Smallest count kept at the low end: one past the largest ``m`` for which
    ``P(X <= m) <= tail`` is certified at ``q_lo``, or 0 when no ``m`` is."""

    def certified(m: int) -> bool:
        # The next float above ``m/n`` is at least ``m/n``, and the bound only weakens as
        # its argument rises toward ``q_lo``, so the rounded argument keeps it valid.
        x = 0.0 if m == 0 else math.nextafter(m / n, math.inf)
        return x <= q_lo and exponent_lower_bound(n, x, q_lo) >= target

    if not certified(0):
        return 0
    cut, kept = 0, min(n, int(n * q_lo) + 1)  # beyond ``n q_lo`` the count exceeds ``q_lo``
    while kept - cut > 1:
        mid = (cut + kept) // 2
        if certified(mid):
            cut = mid
        else:
            kept = mid
    return cut + 1


def _above(n: int, q_hi: float, target: float) -> int:
    """Largest count kept at the high end: one below the smallest ``m`` for which
    ``P(X >= m) <= tail`` is certified at ``q_hi``, or ``n`` when no ``m`` is."""

    def certified(m: int) -> bool:
        x = 1.0 if m == n else math.nextafter(m / n, -math.inf)
        return x >= q_hi and exponent_lower_bound(n, x, q_hi) >= target

    if not certified(n):
        return n
    cut, kept = n, max(0, math.ceil(n * q_hi) - 1)  # below ``n q_hi`` the count is under ``q_hi``
    while cut - kept > 1:
        mid = (cut + kept) // 2
        if certified(mid):
            cut = mid
        else:
            kept = mid
    return cut - 1


def chernoff_support(n: int, q_lo: float, q_hi: float, tail: float) -> tuple[int, int]:
    """Counts ``(i_lo, i_hi)`` such that, for every ``q`` in ``[q_lo, q_hi]`` and
    ``X ~ Bin(n, q)``, ``P(X < i_lo) <= tail`` and ``P(X > i_hi) <= tail``.

    ``tail`` must lie in ``(0, 1/2)``, which keeps ``i_lo <= i_hi``. An end the bound cannot
    certify keeps its whole range: a ``q_lo`` of 0 keeps ``X = 0`` reachable, a ``q_hi`` of 1
    keeps ``X = n``, and so does any end whose rate leaves every count above ``-ln(tail)``
    short of the exponent.
    """
    # ``ln(tail)`` is within ``L = _LIBM_ULPS * _EPS`` (relative) of its true value, so its true
    # magnitude is at most the computed one over ``1 - L``, which is below ``1 + 2L`` times it.
    target = math.nextafter(-math.log(tail) * (1.0 + 2.0 * _LIBM_ULPS * _EPS), math.inf)
    return _below(n, q_lo, target), _above(n, q_hi, target)
