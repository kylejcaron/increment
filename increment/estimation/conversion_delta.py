"""The delta-method decision of an unadjusted conversion or retention contrast over arrays of
count pairs.

Counts whose four per-arm success and failure counts all reach ``dense_min_count(tail)`` are
decided by the delta method (``conversion_route``): the log risk ratio against a
Welch-Satterthwaite ``t`` reference, rejecting when the interval of the row's alternative lies
beyond the null (``LiftEstimate.stat_sig``). `delta_decision` evaluates that decision for many
count pairs at once. Each arm's mean and log standard error are the runtime's own functions of
its counts (`_arm`, the shared primitive); the pair-level arithmetic repeats the runtime's
operations in NumPy, which differs from the runtime's ``math`` calls only in the last places of
its transcendental functions, and the reference quantile is the runtime's own `student_t_isf`.
It certifies a pair only beyond a radius derived from those rounding units and the
runtime's conjugate update (see `delta_decision`), with a bracket of the quantile that follows
from its monotonicity in the degrees of freedom; nothing is calibrated against the runtime's
output. Every pair it cannot certify is left to `production_decision`, the unchanged runtime
row, or to the caller's ambiguous mass when the rows are not affordable.

`delta_interval` and `production_decision` are the runtime's own calculation for one pair, and
the only decision planning uses. `delta_decision`, `delta_statistic`, `critical_values` and
`delta_log_bounds` are vectorised approximations for the calibration campaigns' enumerations and
noncoverage tables: a measurement, never a planning decision and never a certificate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.stats import t as _student_t

from increment._literals import Alternative

_UNIT_ROUNDOFF = 2.0**-53

#: Degree-of-freedom floor and node count of the Chebyshev interpolation in ``1 / df`` that
#: stands in for a per-point ``t`` quantile; above the floor the quantile is analytic in
#: ``1 / df`` over a lattice's narrow range and the interpolant agrees with ``t.isf`` to
#: `CRITICAL_AGREEMENT`.
INTERPOLATION_DF_FLOOR = 30.0
INTERPOLATION_NODES = 24
CRITICAL_AGREEMENT = 1e-12
_INTERPOLATION_MIN_POINTS = 4096


def delta_statistic(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(log_rr, se, df)`` of the delta-method route for count arrays with ``0 < x < n``."""
    with np.errstate(all="ignore"):
        mean_c, mean_t = x_c / n_c, x_t / n_t
        var_c = x_c * (n_c - x_c) / (n_c * (n_c - 1.0))
        var_t = x_t * (n_t - x_t) / (n_t * (n_t - 1.0))
        se_c = np.sqrt(var_c / (n_c * mean_c**2))
        se_t = np.sqrt(var_t / (n_t * mean_t**2))
        log_rr = np.log(mean_t) - np.log(mean_c)
        se = np.hypot(se_c, se_t)
        scale = np.maximum(se_c, se_t)
        a, b = se_t / scale, se_c / scale
        df = (a * a + b * b) ** 2 / (a**4 / (n_t - 1) + b**4 / (n_c - 1))
    return log_rr, se, df


def critical_values(df: np.ndarray, tail: float) -> np.ndarray:
    """Student ``t`` upper-tail critical values at ``df``."""
    finite = df[np.isfinite(df)]
    if finite.size < _INTERPOLATION_MIN_POINTS or finite.min() < INTERPOLATION_DF_FLOOR:
        return _student_t.isf(tail, df)
    lo, hi = 1.0 / finite.max(), 1.0 / finite.min()
    nodes = np.cos(np.pi * (np.arange(INTERPOLATION_NODES) + 0.5) / INTERPOLATION_NODES)
    inverse = 0.5 * (lo + hi) + 0.5 * (hi - lo) * nodes
    coefficients = np.polynomial.chebyshev.chebfit(
        nodes, _student_t.isf(tail, 1.0 / inverse), INTERPOLATION_NODES - 1
    )
    with np.errstate(all="ignore"):
        position = (1.0 / df - 0.5 * (lo + hi)) * (2.0 / (hi - lo)) if hi > lo else 0.0 * df
    return np.polynomial.chebyshev.chebval(position, coefficients)


def delta_log_bounds(
    x_c: np.ndarray, n_c: int, x_t: np.ndarray, n_t: int, tail: float
) -> tuple[np.ndarray, np.ndarray]:
    """Log risk-ratio interval bounds the production delta-method route reports at one-sided
    ``tail``, for count arrays with ``0 < x < n``."""
    log_rr, se, df = delta_statistic(x_c, n_c, x_t, n_t)
    crit = critical_values(df, tail)
    return log_rr - crit * se, log_rr + crit * se


@dataclass(frozen=True, slots=True)
class DeltaDecision:
    """The rejections of count pairs, by direction: ``plus`` where the interval lies above the
    null and ``minus`` where it lies below it, each only for an alternative that reads it.
    ``settled`` is false where the rejection is not certain either way (`delta_decision`), so
    the runtime row decides the pair; ``plus`` and ``minus`` are false there."""

    plus: np.ndarray
    minus: np.ndarray
    settled: np.ndarray


#: Relative accuracy assumed of `student_t_isf` (the runtime's own reference quantile), and so of
#: the monotonicity in the degrees of freedom the bracket below relies on.
CRITICAL_RELATIVE_ACCURACY = 1e-12

# prose: allow-long derivation of a constant
#: Rounding units allowed between the vectorised margin and the runtime's, in ``u = 2**-53``:
#: about twenty IEEE operations at one unit each (the runtime's moments, ratio, hypot, product,
#: subtraction and the conjugate update; NumPy's and the runtime's differ only where they call a
#: transcendental function) and three such calls (log or log1p, hypot, expm1) at no more than
#: 8 ulp, 16 units, each, doubled.
ROUNDING_UNITS = 128

_COLUMN_CHUNK = 64


@lru_cache(maxsize=1 << 18)
def _arm(x: int, n: int) -> tuple[float, float]:
    """``(mean, log standard error)`` of an arm of ``x`` successes in ``n`` units as the runtime
    forms them: the centered moments of ``ArmStats.from_raw_sums``, ``to_summary``, and
    ``se_log_mean`` -- the shared primitive, not a restatement of it."""
    from increment.estimation.armstats import ArmStats
    from increment.estimation.variance import se_log_mean

    summary = ArmStats.from_raw_sums(
        study_id="e", metric="conv", group_id="g", n=n, sum_y=float(x), sum_y2=float(x)
    ).to_summary()
    return summary.mean, se_log_mean(summary.var, summary.mean, summary.n)


def _arms(counts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    flat = [_arm(int(x), n) for x in counts.ravel()]
    mean = np.array([m for m, _ in flat]).reshape(counts.shape)
    se = np.array([s for _, s in flat]).reshape(counts.shape)
    return mean, se


def _welch_df(se_t: np.ndarray, se_c: np.ndarray, n_t: int, n_c: int) -> np.ndarray:
    """``_resolve_fixed_horizon``'s Welch-Satterthwaite degrees of freedom, operation for
    operation."""
    scale = np.maximum(se_t, se_c)
    a, b = se_t / scale, se_c / scale
    return (a * a + b * b) ** 2 / (a**4 / (n_t - 1) + b**4 / (n_c - 1))


def _crit(df: float, tail: float) -> float:
    from increment.estimation._student_t import student_t_isf

    return float(student_t_isf(tail, df))


def delta_decision(
    x_c: np.ndarray,
    n_c: int,
    x_t: np.ndarray,
    n_t: int,
    *,
    tail: float,
    alternative: Alternative,
    null_lift: float,
) -> DeltaDecision:
    """The delta-method rejection of every count pair (broadcast arrays, ``x_c`` down the rows
    and ``x_t`` along the columns, counts inside the routed rectangle) at one-sided ``tail``
    against ``1 + null_lift``: the lower interval end above the null for ``greater`` and
    ``two-sided``, the upper end below it for ``less`` and ``two-sided``.

    Each arm's mean and log standard error are the runtime's own (`_arm`). The pair's log risk
    ratio, combined standard error and Welch degrees of freedom repeat the runtime's operations
    in NumPy. The reference quantile is the runtime's own `student_t_isf`, decreasing in the
    degrees of freedom, so over each run of `_COLUMN_CHUNK` columns of a row it lies between its
    values at the run's largest and smallest degrees of freedom (each widened by
    `CRITICAL_RELATIVE_ACCURACY`). A pair is settled only when its interval end is on the same
    side of the null at both ends of that bracket, beyond a radius that covers the difference
    between the vectorised margin and the runtime's (`ROUNDING_UNITS` of rounding on the terms
    of the margin, and the near-flat conjugate update the runtime applies: a relative
    ``(se / prior_sigma) ** 2``). Every other pair is left to the runtime (`settled` false).
    """
    from increment.estimation.inference import _DEFAULT_PRIOR

    rows, cols = np.broadcast(x_c, x_t).shape
    mean_c, se_c = _arms(np.asarray(x_c).reshape(-1), n_c)
    mean_t, se_t = _arms(np.asarray(x_t).reshape(-1), n_t)
    mc, mt = mean_c[:, None], mean_t[None, :]
    sc, st = se_c[:, None], se_t[None, :]
    with np.errstate(all="ignore"):
        lo, hi = np.minimum(mc, mt), np.maximum(mc, mt)
        log_rr = np.where(lo >= hi / 2.0, np.log1p((mt - mc) / mc), np.log(mt) - np.log(mc))
        se = np.hypot(st, sc)
        df = _welch_df(st, sc, n_t, n_c)
    log_null = math.log1p(null_lift)
    update = (float(np.max(se)) / _DEFAULT_PRIOR.sigma) ** 2
    slack = ROUNDING_UNITS * _UNIT_ROUNDOFF + update
    starts = np.arange(0, cols, _COLUMN_CHUNK)
    df_low = np.minimum.reduceat(df, starts, axis=1)
    df_high = np.maximum.reduceat(df, starts, axis=1)
    crit_hi = np.empty(df_low.shape)
    crit_lo = np.empty(df_low.shape)
    for index in np.ndindex(df_low.shape):
        crit_hi[index] = _crit(float(df_low[index]) * (1.0 - slack), tail) * (
            1.0 + CRITICAL_RELATIVE_ACCURACY
        )
        crit_lo[index] = _crit(float(df_high[index]) * (1.0 + slack), tail) * (
            1.0 - CRITICAL_RELATIVE_ACCURACY
        )
    widths = np.diff(np.append(starts, cols))
    crit_hi, crit_lo = np.repeat(crit_hi, widths, axis=1), np.repeat(crit_lo, widths, axis=1)
    radius = slack * (np.abs(log_rr) + crit_hi * se + abs(log_null))
    settled = np.ones((rows, cols), bool)
    plus = np.zeros((rows, cols), bool)
    minus = np.zeros((rows, cols), bool)
    certain: list[np.ndarray] = []
    if alternative != "less":
        up = log_rr - log_null
        rejects = up - crit_hi * se > radius
        accepts = up - crit_lo * se < -radius
        plus, certain = rejects, [rejects | accepts]
    if alternative != "greater":
        down = log_null - log_rr
        rejects = down - crit_hi * se > radius
        accepts = down - crit_lo * se < -radius
        minus, certain = rejects, [*certain, rejects | accepts]
    # A pair is settled when every direction read is certain, or one of them certainly rejects.
    everywhere = np.logical_and.reduce(certain)
    settled = everywhere | plus | minus
    return DeltaDecision(plus & settled, minus & settled, settled)


def delta_interval(
    x_c: int, n_c: int, x_t: int, n_t: int, *, tail: float, alternative: Alternative
) -> tuple[float, float] | None:
    """The relative-lift interval ``(lb, ub)`` the runtime reports for a routed count pair, or
    ``None`` when the runtime guards refuse it (a decision failure, never a rejection).

    This is the runtime's own calculation, call for call: the arms' moments and log standard
    errors (`_arm`, as ``_compute_lift_arm_moments`` forms them for a mean metric),
    ``stable_log_ratio``, and ``infer_lift`` with the Welch reference ``arm_ns=(n_t, n_c)`` at
    the row's alpha. Nothing is restated, so the interval is the runtime's to the last bit."""
    from increment.estimation.inference import LiftGuardError, infer_lift
    from increment.estimation.variance import stable_log_ratio

    mean_c, se_c = _arm(x_c, n_c)
    mean_t, se_t = _arm(x_t, n_t)
    try:
        estimate = infer_lift(
            metric="conv",
            group_id="treatment",
            method="unadjusted",
            log_rr=stable_log_ratio(mean_c, mean_t),
            se_t=se_t,
            se_c=se_c,
            alpha=2.0 * tail if alternative == "two-sided" else tail,
            alternative=alternative,
            arm_ns=(n_t, n_c),
            method_role="decision",
        )
    except LiftGuardError:
        return None
    lift = estimate.lift
    assert lift is not None and lift.lb is not None and lift.ub is not None
    return lift.lb, lift.ub


def production_decision(
    x_c: int,
    n_c: int,
    x_t: int,
    n_t: int,
    *,
    tail: float,
    alternative: Alternative,
    null_lift: float,
) -> tuple[bool, bool]:
    """``(plus, minus)`` of one count pair as the runtime decides it: the interval of
    `delta_interval` lies above (``plus``) or below (``minus``) ``null_lift``, each only for an
    alternative that reads it (``LiftEstimate.stat_sig``'s comparisons). A pair the runtime
    guards refuse rejects in neither direction."""
    interval = delta_interval(x_c, n_c, x_t, n_t, tail=tail, alternative=alternative)
    if interval is None:
        return False, False
    lower, upper = interval
    return (
        alternative != "less" and lower > null_lift,
        alternative != "greater" and upper < null_lift,
    )
