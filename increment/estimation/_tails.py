"""Tail probability numerics and critical values via the survival (isf) form.

A two-sided critical value at significance level ``alpha`` is the point
whose one-sided tail probability is ``alpha / 2``. Computing it as
``dist.ppf(1 - alpha / 2)`` rounds the complement to exactly ``1.0``
once ``alpha`` is small enough (already losing significant digits by
``alpha ~ 1e-12``; ``1 - alpha / 2 == 1.0`` in binary64 by
``alpha ~ 1e-16``), silently returning ``inf`` or a badly under-resolved
quantile. ``dist.isf(alpha / 2)`` evaluates the tail probability
directly instead of through its rounded complement, and stays accurate
to the reference distribution's own tail resolution.

An inverse-survival backend can fail even when the mathematical quantile
is representable. :func:`student_t_isf` handles Student-t tails with
complementary beta coordinates and bounded numerical fallbacks, including
subnormal probabilities. :func:`tail_isf` requires a FINITE result so an
unavailable or unrepresentable quantile cannot silently degrade an interval.
It does NOT require positivity, because a tail above 0.5 legitimately
yields a negative quantile.
:func:`two_sided_critical_value` additionally requires a positive result,
since it halves *alpha* and a non-positive critical value there means the
same resolution failure.

Per AGENTS.md: never derive a tail probability from ``1 - level`` or
recover ``alpha`` from a rounded confidence level -- pass the tail
probability (or ``alpha``) through untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from increment.errors import InvalidRequestError, RefusalSpec, refuse
from increment.estimation._student_t import student_t_isf

__all__ = [
    "SCIPY_BINOMIAL_ULP_ALLOWANCE",
    "resolvable_expm1",
    "student_t_isf",
    "tail_isf",
    "two_sided_critical_value",
    "wald_bounds",
]

# Floor of the SciPy binomial pmf/cdf/sf relative-error allowance, in ULPs. Measured at `n <= 1000`
# trials; `binomial_rr._ulp_allowance` raises it to `n` ULPs above that (error grows with `n`).
SCIPY_BINOMIAL_ULP_ALLOWANCE = 2048


def _render_unresolvable(*, what: str, tail_or_alpha: float, value: object) -> str:
    return f"{what} (tail_or_alpha={tail_or_alpha!r}): {value}"


_TAILS_UNRESOLVABLE = RefusalSpec(
    "estimation.tails.unresolvable", InvalidRequestError, _render_unresolvable
)


def tail_isf(
    isf: Callable[..., float],
    tail: float,
    *shape_args: float,
    what: str,
) -> float:
    """The value whose upper-tail probability is exactly *tail*.

    *isf* is a distribution's inverse survival function -- e.g.
    ``scipy.stats.norm.isf``, ``student_t_isf``, or a frozen
    distribution's bound ``.isf`` (``scipy.stats.beta(a, b).isf``).
    *shape_args* pass through positionally after *tail* (e.g. degrees
    of freedom for Student-t). *tail* must be a probability the caller
    computed directly -- never a complement recovered from
    ``1 - something``.

    Raises ``InvalidRequestError`` (code ``estimation.tails.unresolvable``) if
    *tail* is not in ``(0, 1)`` or the resulting value is non-finite. A backend's
    non-finite result does not by itself prove mathematical unrepresentability.
    """
    if not math.isfinite(tail) or not (0.0 < tail < 1.0):
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=tail,
            value=f"tail probability must be in (0, 1), got {tail!r}",
        )
    value = float(isf(tail, *shape_args))
    if not math.isfinite(value):
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=tail,
            value=(
                f"inverse survival at tail probability {tail:.3g} returned "
                f"{value!r}; a finite quantile is required to build an interval"
            ),
        )
    return value


def two_sided_critical_value(
    isf: Callable[..., float],
    alpha: float,
    *shape_args: float,
    what: str,
) -> float:
    """Two-sided critical value for significance level *alpha*.

    Equivalent to ``dist.ppf(1 - alpha / 2)`` but evaluated as
    ``dist.isf(alpha / 2)`` (see the module docstring for why). Pass
    *alpha* itself, never a level's complement.

    Raises ``InvalidRequestError`` (code ``estimation.tails.unresolvable``) if
    *alpha* is not in ``(0, 1)`` or the resulting critical value is non-finite
    or non-positive.
    """
    if not math.isfinite(alpha) or not (0.0 < alpha < 1.0):
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=alpha,
            value=f"alpha must be in (0, 1), got {alpha!r}",
        )
    crit = tail_isf(isf, alpha / 2.0, *shape_args, what=what)
    if crit <= 0.0:
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=alpha,
            value=(
                f"critical value at alpha={alpha:.3g} is {crit!r}, not positive -- "
                "the tail probability has underflowed the reference distribution's "
                "resolution"
            ),
        )
    return crit


def wald_bounds(point: float, crit: float, se: float, *, what: str) -> tuple[float, float]:
    """Two-sided bounds ``point +- crit * se``, refusing a non-finite width.

    A finite critical value times a large finite standard error still
    overflows, so validating the critical value alone leaves an infinite bound
    reachable from finite inputs. The half-width is judged here, once, for
    every caller building an interval from a tail critical value.
    """
    half_width = crit * se
    if not math.isfinite(half_width):
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=crit,
            value=(
                f"interval half-width is not representable in float64 (critical "
                f"value {crit:.6g} x se {se:.6g}) -- the interval cannot be "
                f"reported at this scale"
            ),
        )
    lower, upper = point - half_width, point + half_width
    if not (math.isfinite(lower) and math.isfinite(upper)):
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=crit,
            value=(
                f"interval bound is not representable in float64 (point "
                f"{point:.6g} +- {half_width:.6g})"
            ),
        )
    return lower, upper


def resolvable_expm1(x: float, *, what: str) -> float:
    """``math.expm1(x)``, refusing rather than reporting a domain-floor
    artifact: a result of exactly -1.0 means *x* underflowed there in
    float64, not that the true effect is a total loss; an overflow means
    *x* is not representable on the ratio scale at all. Both mean no
    honest bound exists at this scale (mirrors this module's isf guards).
    """
    try:
        value = math.expm1(x)
    except OverflowError:
        refuse(_TAILS_UNRESOLVABLE, what=what, tail_or_alpha=x, value=f"expm1({x!r}) overflows")
        raise  # unreachable; refuse() never returns
    if value == -1.0:
        refuse(
            _TAILS_UNRESOLVABLE,
            what=what,
            tail_or_alpha=x,
            value=f"expm1({x!r}) underflows to the excluded -1.0 floor",
        )
    return value
