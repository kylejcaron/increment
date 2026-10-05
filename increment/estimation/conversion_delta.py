"""The delta-method decision of an unadjusted conversion or retention count pair, as the
runtime makes it.

Counts whose four per-arm success and failure counts all reach ``dense_min_count(tail)`` are
decided by the delta method (``conversion_route``): the log risk ratio against a
Welch-Satterthwaite ``t`` reference, rejecting when the interval of the row's alternative lies
beyond the null (``LiftEstimate.stat_sig``). `delta_interval` and `production_decision` are the
runtime's own calculation for one count pair, call for call (the arms' moments and log standard
errors, ``stable_log_ratio``, ``infer_lift``); planning decides every routed pair with them.
"""

from __future__ import annotations

from functools import lru_cache

from increment._literals import Alternative


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
