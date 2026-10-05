"""The delta-method decision of an unadjusted conversion or retention contrast over arrays of
count pairs.

Counts whose four per-arm success and failure counts all reach ``dense_min_count(tail)`` are
decided by the delta method (``conversion_route``): the log risk ratio against a
Welch-Satterthwaite ``t`` reference, rejecting when the interval of the row's alternative lies
beyond the null (``LiftEstimate.stat_sig``). This module evaluates that decision for many count
pairs at once, from the same arm moments, the same standard error and the same reference, so
planning and calibration read the production rule instead of restating it.

Each arm's centered moments give ``mean = x / n`` and ``var = x (n - x) / (n (n - 1))``; its log
standard error is ``sqrt(var / (n mean ** 2))``, the combined one the hypotenuse of the two, and
the reference's degrees of freedom are Welch-Satterthwaite. The runtime forms the same
quantities through ``ArmStats`` moments and ``math`` functions, so the two agree to a few units
in the last place, not bit for bit. `delta_decision` therefore settles a pair only when its
interval end clears the null by `DELTA_AGREEMENT` on the log scale, which is far above the
disagreement (checked pair by pair against ``estimate_lift`` in the tests); `production_decision`
is the unchanged runtime row, and decides every pair the vectorised rule leaves open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as _student_t

from increment._literals import Alternative

#: Log-scale distance from the null within which the vectorised decision defers to the runtime.
#: The two evaluate the interval ends to a few ULP of a quantity below ten and the reference
#: quantile to `CRITICAL_AGREEMENT`; this is a hundred thousand times the largest gap measured.
DELTA_AGREEMENT = 1e-10

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
    ``settled`` is false where a read interval end is within `DELTA_AGREEMENT` of the null,
    so the runtime decides the pair (`production_decision`)."""

    plus: np.ndarray
    minus: np.ndarray
    settled: np.ndarray


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
    """The delta-method rejection of every count pair (broadcast arrays with ``0 < x < n``) at
    one-sided ``tail`` against ``1 + null_lift``: the lower interval end above the null for
    ``greater`` and ``two-sided``, the upper end below it for ``less`` and ``two-sided``."""
    lower, upper = delta_log_bounds(x_c, n_c, x_t, n_t, tail)
    log_null = math.log1p(null_lift)
    reads_plus, reads_minus = alternative != "less", alternative != "greater"
    above, below = lower - log_null, log_null - upper
    plus = reads_plus & (above > 0.0)
    minus = reads_minus & (below > 0.0)
    open_ = np.zeros(np.broadcast(x_c, x_t).shape, bool)
    if reads_plus:
        open_ |= ~(np.abs(above) > DELTA_AGREEMENT)
    if reads_minus:
        open_ |= ~(np.abs(below) > DELTA_AGREEMENT)
    return DeltaDecision(
        np.broadcast_to(plus, open_.shape), np.broadcast_to(minus, open_.shape), ~open_
    )


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
    """``(plus, minus)`` of one count pair as the runtime decides it: ``estimate_lift`` on the
    contrast, whose interval end lies above (``plus``) or below (``minus``) the null, each only
    for an alternative that reads it. Only a pair the count rule routes to the delta method has
    that row (``ValueError`` otherwise); one the runtime cannot form a row for rejects in neither
    direction."""
    from increment.estimation.armstats import ArmStats
    from increment.estimation.engine import Method, estimate_lift
    from increment.semantics.models import ConversionMetric

    rows = []
    for group_id, n, x in (("control", n_c, x_c), ("treatment", n_t, x_t)):
        arm = ArmStats.from_raw_sums(
            study_id="e", metric="conv", group_id=group_id, n=n, sum_y=float(x), sum_y2=float(x)
        )
        rows.append(
            {
                "experiment_id": "e",
                "metric": "conv",
                "group_id": group_id,
                "n": float(arm.n),
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
            }
        )
    computation = estimate_lift(
        metrics=[ConversionMetric(name="conv", entity="user", fact="conv")],
        summary=rows,
        control_group="control",
        methods=[Method(name="unadjusted", conversion_inference="auto")],
        alpha=2.0 * tail if alternative == "two-sided" else tail,
        alternative=alternative,
        null_lift=null_lift,
    )
    if computation.failures or not computation.results:
        return False, False
    (row,) = computation.results
    if row.reference_kind != "t":
        raise ValueError(
            f"counts {(x_c, n_c, x_t, n_t)} are not decided by the delta method: the count rule "
            "keeps them on the finite-sample route"
        )
    assert row.lift is not None
    plus = alternative != "less" and row.lift.lb is not None and row.lift.lb > null_lift
    minus = alternative != "greater" and row.lift.ub is not None and row.lift.ub < null_lift
    return bool(plus), bool(minus)
