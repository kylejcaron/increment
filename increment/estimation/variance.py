"""Variance models for the delta-method SE of log(mean) / log(ratio).

``se_log_mean`` is the one formula every mean-type variance model uses;
``RatioVarianceModel`` adds the -2*Cov cross term for ratio metrics. The
``Registry`` is a small generic dispatch class shared by the metric-type
and variance-reduction axes.
"""

from __future__ import annotations

import math
import sys
from typing import Protocol

from increment._moment_plan import X_ROLE_VARIABLES, X_SLOT_ROLES
from increment.errors import (
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.estimation.armstats import (
    _EPS,
    ArmStats,
    clamp_negative_variance,
    variance_slack,
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.variance.metric_group_log": "metric={metric!r} group={group_id!r}: {what} {mean:.6g} <= 0 -- log-scale relative-lift inference is undefined (zero events, or a signed metric that needs an absolute-scale estimand)",
        "estimation.variance.mean_positive_log": "mean must be positive for log(mean), got {mean}",
        "estimation.variance.positive_se_log": "n must be positive for se_log_mean, got {n}",
        "estimation.variance.var_non_negative": "var must be non-negative, got {var}",
        "estimation.variance.se_log_mean_out_of_range": "se_log_mean(var={var:.6g}, mean={mean:.6g}, n={n:.6g}): the true standard error is outside the float64 exponent range (log SE = {half:.6g}) -- it is not representable at this scale",
        "estimation.variance.se_log_mean_not_representable": "se_log_mean(var={var:.6g}, mean={mean:.6g}, n={n:.6g}) is {se!r} for a positive variance -- the true standard error is not representable in float64",
        "estimation.variance.ratio_moments_needs_ref_den": "ratio-family moments require ref_den/cden1/cden2/cyden (not None)",
        "estimation.variance.ratio_moments_least_two_units": "n must be at least 2 to compute ddof=1 ratio moments (metric {metric!r}, arm {group_id!r} has n={n})",
        "estimation.variance.ratio_moments_nonpositive_denominator_mean": "ratio metric {metric!r} arm {group_id!r}: nonpositive denominator mean ({d_bar!r}) -- the ratio estimand sum(num)/sum(den) is undefined or sign-ambiguous; check the denominator fact/window",
        "estimation.variance.ratio_variance_term": "{what}: the ratio variance term is not representable in float64 for num_bar={num_bar:.6g}, den_bar={den_bar:.6g}, var_den={var_den:.6g} -- r*sqrt(var_den)={rooted:.6g} squares past the exponent range",
        "estimation.variance.ratio_residual_variance": "{what}: ratio residual variance is not representable in float64 (residual={residual!r}) for num_bar={num_bar:.6g}, den_bar={den_bar:.6g} -- the quadratic form overflowed at this scale",
        "estimation.variance.ratio_residual_variance_negative": "{what}: ratio residual variance is {residual:.6g}, negative beyond floating-point rounding -- these moments are corrupt or were not produced together",
        "estimation.variance.ratio_se_representable": "{what}: ratio SE is not representable in float64 for num_bar={num_bar:.6g}, den_bar={den_bar:.6g} -- a positive residual variance ({clamped:.6g}) underflows to a zero standard error at this scale",
        "estimation.variance.ratio_se_sqrt_not_representable": "{what}: ratio SE is not representable in float64 for num_bar={num_bar:.6g}, den_bar={den_bar:.6g} -- sqrt(residual/n)={root:.6g} over a denominator this small overflows the exponent range",
        "estimation.variance.ratio_se_final_not_representable": "{what}: ratio SE is not representable in float64 (R={r!r}, se={se!r}) for num_bar={num_bar:.6g}, den_bar={den_bar:.6g} -- the ratio or its variance has overflowed at this scale",
        "estimation.variance.cluster_robust_late": "cluster-robust LATE needs a cluster uptake family, but this row declares x_role={x_role!r} (x carries {carries}, not a cluster uptake total) -- the cluster-robust LATE first stage cannot be computed. Rebuild the summary with a declared cluster AND uptake fact on an Encouragement design.",
        "estimation.variance.cluster_robust_late_needs_cluster_uptake_family": "cluster-robust LATE needs the cluster-grain uptake family (ref_x/cx1/cx2/cxy) -- rebuild the summary with a declared cluster AND uptake fact on an Encouragement design.",
        "estimation.variance.cluster_robust_late_needs_cxden": "cluster-robust LATE needs cxden (cov of cluster uptake total and cluster size) -- rebuild the summary through group_summary(cluster=...) / from_unit_summary(cluster=...).",
        "estimation.variance.cluster_outcome_moments_least_two_units": "n must be at least 2 to compute ddof=1 cluster-outcome moments (metric {metric!r}, arm {group_id!r} has n={n})",
        "estimation.variance.cluster_outcome_moments_nonpositive_size": "cluster metric {metric!r} arm {group_id!r}: nonpositive mean cluster size ({size_bar!r}) -- corrupt cluster moments",
        "estimation.variance.cluster_uptake_moments_least_two_units": "n must be at least 2 to compute ddof=1 cluster-uptake moments (metric {metric!r}, arm {group_id!r} has n={n})",
        "estimation.variance.cluster_uptake_moments_needs_ref_den": "ratio-family moments require ref_den/cden1/cden2 (not None)",
        "estimation.variance.ratio_variance.correlation_not_representable": "var_log_r for metric={metric!r} group={group_id!r}: the implied numerator/denominator correlation has log magnitude {log_rho:.6g}, astronomically beyond the Cauchy-Schwarz bound of 1 -- these moments are corrupt or were not produced together",
        "estimation.variance.ratio_variance.variance_scale_not_representable": "var_log_r for metric={metric!r} group={group_id!r}: log-ratio variance is {value:.6g} of the terms' own scale (log scale {log_scale:.6g}), negative beyond their floating-point rounding (~{variance_slack:.3g}) -- the numerator and denominator moments are inconsistent (mis-aggregated upstream or not produced together), not merely cancelled.",
        "estimation.variance.ratio_variance.true_se_out_of_range": "var_log_r for metric={metric!r} group={group_id!r}: the true standard error is outside the float64 exponent range (log SE = {log_se:.6g}) -- it is not representable at this scale",
        "estimation.variance.ratio_variance.se_not_representable": "var_log_r for metric={metric!r} group={group_id!r} is {se!r} for a positive variance -- the true standard error is not representable in float64",
        "estimation.variance.registry.no_registered_available": RefusalSpec(
            "estimation.variance.registry.no_registered_available",
            UnsupportedRequestError,
            template="No {registry_name} registered for '{key}'. Available keys: {available}",
        ),
    },
)
_raise = raiser(_REFUSALS)


def ratio_residual_slack(magnitude: float, n: int) -> float:
    """Rounding tolerance for a delta-method ratio variance.

    :func:`~increment.estimation.armstats.variance_slack` budgets the
    accumulation of ``n`` addends into each moment. A ratio's delta-method
    terms then divide by a computed mean, and that quotient's rounding
    propagates into the terms on top of the reduction error already budgeted.

    Counting the operations that a perfectly-constant ratio -- whose exact
    residual is zero, so every computed bit is error -- accumulates AFTER the
    moments are formed, each contributing at most half an ulp relatively:

    - cross term ``2 * r * cov``: the quotient, then the two products (3).
    - squared term ``(r * sqrt(var_den)) ** 2``: the quotient, the root, the
      product, then the squaring (4).
    - summing the three terms into the residual (2).

    Nine half-ulps bound the relative error, so the budget beyond the
    reduction slack is ``4.5 * eps`` of the terms' combined magnitude; it is
    carried as 8 to leave the derivation's own slack. This stays ~1e-15
    RELATIVE, far tighter than any genuine Cauchy-Schwarz violation, which is
    an O(1) relative deficit.

    The residual's exact value depends on the order the warehouse summed the
    moments, so this must be an error bound rather than a tolerance fitted to
    an observed value.
    """
    return variance_slack(magnitude, n) + 8.0 * _EPS * magnitude


# Delta-method SE of log(mean)


def check_positive_mean(metric: str, group_id: str, mean: float, *, what: str = "arm mean") -> None:
    """Refuse a non-positive mean before it reaches ``math.log``.

    Shared by the unadjusted mean-family path (``MeanVarianceModel``) and
    CUPED's adjusted-mean path so both name the metric, arm, and value
    instead of a bare ``math domain error``.
    """
    if mean <= 0:
        _raise(
            "estimation.variance.metric_group_log",
            metric=metric,
            group_id=group_id,
            what=what,
            mean=mean,
        )


def stable_log_ratio(mean_c: float, mean_t: float) -> float:
    """Return ``log(mean_t / mean_c)`` stably from two raw arm means.

    For means within a factor of two, ``log1p((mean_t - mean_c) / mean_c)``
    avoids cancellation when the effect is tiny.  Outside that range,
    subtracting the two arm logs avoids rounding a large decrease to
    ``log1p(-1)`` and is more accurate.  This helper is used by mean and
    quantile paths that have one raw point estimate per arm; ratio metrics
    instead preserve their numerator/denominator cross-ratio before
    applying the equivalent one-log conversion.

    Either mean at or below zero is refused by name
    (``estimation.variance.mean_positive_log``, as ``se_log_mean`` does)
    rather than surfacing a bare ``ZeroDivisionError`` or ``math domain
    error``; production callers already refuse it with the metric and arm
    attached (``check_positive_mean``) before reaching here.
    """
    if mean_c <= 0:
        _raise("estimation.variance.mean_positive_log", mean=mean_c)
    if mean_t <= 0:
        _raise("estimation.variance.mean_positive_log", mean=mean_t)
    if min(mean_c, mean_t) >= max(mean_c, mean_t) / 2.0:
        return math.log1p((mean_t - mean_c) / mean_c)
    return math.log(mean_t) - math.log(mean_c)


_MAX_FLOAT = sys.float_info.max
_SQRT_MAX_FLOAT = math.sqrt(_MAX_FLOAT)
#: An implied |correlation| past this cannot be rounding, and exponentiating
#: it would overflow instead of yielding a number the variance path can judge.
_MAX_LOG_IMPLIED_RHO = 300.0


def se_log_mean(var: float, mean: float, n: float) -> float:
    """SE of log(mean) under the delta method: ``sqrt(var / (n * mean**2))``.

    Estimation calls this forward (n known); power analysis inverts it.
    First-order accuracy degrades with expected event count: <1.5% error
    above ~50 events, ~8.5% at n*p~10, ~30% at n*p~2 - see ``infer_lift``
    for the matching point-estimate bias in that regime.
    """
    if mean <= 0:
        _raise("estimation.variance.mean_positive_log", mean=mean)
    if n <= 0:
        _raise("estimation.variance.positive_se_log", n=n)
    if var < 0:
        _raise("estimation.variance.var_non_negative", var=var)
    if var == 0.0:
        return 0.0
    # Work in logs: `mean**2` overflows before the SE does (mean ~ 1e200, var ~ 1e308 and n=2
    # give SE ~ 7e-47), and an infinite square would report se=0.0.
    log_se2 = math.log(var) - math.log(n) - 2.0 * math.log(mean)
    half = 0.5 * log_se2
    # Judge the exponentiated result, not a guessed log bound: `math.exp` stays finite to 709.78
    # and nonzero to -744, so a safe constant would refuse representable answers. Overflow and
    # zero (maximal confidence from a positive variance) are refused by name.
    try:
        se = math.exp(half)
    except OverflowError:
        _raise("estimation.variance.se_log_mean_out_of_range", half=half, mean=mean, n=n, var=var)
    if not math.isfinite(se) or se == 0.0:
        _raise("estimation.variance.se_log_mean_not_representable", mean=mean, n=n, se=se, var=var)
    return se


# Shared by RatioVarianceModel.log_mean_se (log scale) and
# engine._ratio_abs_diff_se (absolute scale); same ddof=1 convention.


def ratio_moments(arm: ArmStats) -> tuple[float, float, float, float, float]:
    """Per-arm (n_bar, d_bar, var_n, var_d, cov_nd) for a ratio-family metric.

    ddof=1 (Bessel-corrected), matching every variance reduction on
    ``ArmStats``. Callers clamp their own derived variance separately:
    ``RatioVarianceModel`` clamps the log-scale ``var_log_r``,
    ``_ratio_abs_diff_se`` the absolute-scale ``var_r`` - they can diverge
    under floating point, so neither substitutes for the other.
    """
    if arm.ref_den is None or arm.cden1 is None or arm.cden2 is None or arm.cyden is None:
        _raise("estimation.variance.ratio_moments_needs_ref_den")
    if arm.n < 2:
        _raise(
            "estimation.variance.ratio_moments_least_two_units",
            group_id=arm.group_id,
            metric=arm.metric,
            n=arm.n,
        )

    n_bar = arm.mean_y()
    d_bar = arm.mean_den()
    if d_bar <= 0.0:
        _raise(
            "estimation.variance.ratio_moments_nonpositive_denominator_mean",
            group_id=arm.group_id,
            metric=arm.metric,
            d_bar=d_bar,
        )

    # ddof=1 moments, every one an analytic expansion in the centered fields
    var_n = arm.var_y()
    var_d = arm.var_den()
    cov_nd = arm.cov_yden()

    return n_bar, d_bar, var_n, var_d, cov_nd


#: Calibrated across sample sizes 20-2000 and lognormal, exponential, and gamma
#: denominators: cells below 93% coverage at nominal 95% usually exceed this
#: threshold, while cells at or above 94.5% do not.
RATIO_DENOMINATOR_PRECISION_THRESHOLD = 0.15


def ratio_denominator_precision(arm: ArmStats) -> float:
    """Relative standard error of the arm's denominator mean.

    ``sqrt(var_d / n) / d_bar``: the denominator's coefficient of variation
    over ``sqrt(n)``. The delta method linearises ``1 / d_bar``; the larger
    this quantity, the further a right-skewed denominator pushes the
    sampling law of ``1 / d_bar`` from that line and the more the interval
    undercovers. Second-order moments cannot see the skew itself; this is
    the quantity they do carry that tracks the failure. Shares
    :func:`ratio_moments`' domain (positive means, ``n >= 2``); float
    division overflow yields ``inf`` rather than raising.
    """
    _, d_bar, _, var_d, _ = ratio_moments(arm)
    return math.sqrt(var_d / arm.n) / d_bar


#: Standardized skewness of a denominator MEAN, ``g1 / sqrt(n)``, from which the
#: pooled coverage of admitted ratio rows falls below 93.5% at nominal 95%
#: (``calibration/ratio_skew.py``: lognormal denominators sigma 0.5-2.0, n 50-400;
#: rows at or above it covered 92.2%, rows below it 94.3%).
RATIO_DENOMINATOR_SKEW_THRESHOLD = 0.30


def ratio_denominator_mean_skewness(arm: ArmStats) -> float:
    """Standardized skewness of the arm's denominator mean, ``g1 / sqrt(n)``.

    ``g1`` is the arm's sample skewness ``m3 / m2**1.5`` (:meth:`ArmStats.skew_den`);
    dividing by ``sqrt(n)`` gives the skewness of ``d_bar``'s sampling law to
    first order, the leading Edgeworth term of the mean's departure from
    normality and so of the delta method's failure to see how ``1 / d_bar``
    is distributed. Refuses when the third moment was not carried.
    """
    return arm.skew_den() / math.sqrt(arm.n)


def ratio_log_mean_se(
    num_bar: float,
    den_bar: float,
    var_num: float,
    var_den: float,
    cov_num_den: float,
    n: int,
    *,
    group_id: str,
    metric: str,
) -> tuple[float, float]:
    """Log-scale ``(log R, SE(log R))`` for ``R = num_bar / den_bar`` from one
    arm's ddof=1 moments -- the -2*Cov delta method :class:`RatioVarianceModel`
    documents.

    Separate from :class:`RatioVarianceModel` so a caller holding adjusted
    moments rather than an ``ArmStats`` (CUPED on a ratio metric, which
    replaces all five moments) reaches the same interval, not a second one.

    Delta-method variance of log(R), carried so no intermediate leaves
    float64 while the standard error it feeds is representable.
    Normalizing each standard deviation by its own mean and by sqrt(n)
    gives a = sd_n/(num_bar*sqrt(n)) and b = sd_d/(den_bar*sqrt(n)), and the
    cross term is 2*rho*a*b for the moments' own correlation, so

        var_log_r = a**2 + b**2 - 2*rho*a*b

    The magnitudes of a and b are held as LOGS and only their ratio is
    exponentiated, leaving a bracket in [0, 4] at any scale. Forming a
    directly overflows near num_bar ~ 1e-308 and squaring it overflows near
    1e-200, both while the returned standard error is still finite; the
    correlation is bounded by Cauchy-Schwarz, so it is safe to exponentiate.

    Judging the bracket also conditions the negative-variance decision
    better: this seam combines ALREADY-CENTERED moments, so police_sq_sum's
    raw-sum advice ("centre the values before aggregation") is not
    actionable here -- the caller has no further centering available.
    Perfectly proportional numerator and denominator cancel the bracket
    completely, which is a true zero variance.
    """
    log_mean = math.log(num_bar / den_bar)
    if var_num <= 0.0 and var_den <= 0.0:
        return log_mean, 0.0
    half_log_n = 0.5 * math.log(n)
    log_a = 0.5 * math.log(var_num) - math.log(num_bar) - half_log_n if var_num > 0.0 else -math.inf
    log_b = 0.5 * math.log(var_den) - math.log(den_bar) - half_log_n if var_den > 0.0 else -math.inf
    log_scale = max(log_a, log_b)
    ua = math.exp(log_a - log_scale) if log_a > -math.inf else 0.0
    ub = math.exp(log_b - log_scale) if log_b > -math.inf else 0.0
    if var_num > 0.0 and var_den > 0.0 and cov_num_den != 0.0:
        # Do not clamp to the Cauchy-Schwarz bound; that would repair corrupt moments. A
        # correlation slightly over 1 reaches the rounding-aware negative-variance check below;
        # one past _MAX_LOG_IMPLIED_RHO cannot be rounding and would overflow exp, so refuse it.
        log_rho = math.log(abs(cov_num_den)) - 0.5 * math.log(var_num) - 0.5 * math.log(var_den)
        if log_rho > _MAX_LOG_IMPLIED_RHO:
            _raise(
                "estimation.variance.ratio_variance.correlation_not_representable",
                group_id=group_id,
                metric=metric,
                log_rho=log_rho,
            )
        rho = math.copysign(math.exp(log_rho), cov_num_den)
    else:
        # Cauchy-Schwarz forces a zero cross moment against a zero variance.
        rho = 0.0
    cross = 2.0 * rho * ua * ub
    value = ua * ua + ub * ub - cross
    magnitude = ua * ua + ub * ub + abs(cross)
    if value < 0.0 and -value <= ratio_residual_slack(magnitude, n):
        clamped: float | None = 0.0
    else:
        clamped = clamp_negative_variance(value, magnitude=magnitude, n=n)
    if clamped is None:
        _raise(
            "estimation.variance.ratio_variance.variance_scale_not_representable",
            group_id=group_id,
            metric=metric,
            value=value,
            log_scale=log_scale,
            variance_slack=variance_slack(magnitude, n),
        )
    if clamped == 0.0:
        return log_mean, 0.0
    # One exponentiation of the whole result, judged the same way the
    # log-scale mean SE is: past the exponent range the true value is not
    # representable and must refuse, never round to a zero standard error.
    log_se = log_scale + 0.5 * math.log(clamped)
    try:
        se = math.exp(log_se)
    except OverflowError:
        _raise(
            "estimation.variance.ratio_variance.true_se_out_of_range",
            group_id=group_id,
            metric=metric,
            log_se=log_se,
        )
    if not math.isfinite(se) or se == 0.0:
        _raise(
            "estimation.variance.ratio_variance.se_not_representable",
            group_id=group_id,
            metric=metric,
            se=se,
        )
    return log_mean, se


def ratio_abs_diff_se(
    num_bar: float,
    den_bar: float,
    var_num: float,
    var_den: float,
    cov_num_den: float,
    n: int,
    *,
    what: str = "ratio absolute-scale SE",
) -> tuple[float, float]:
    """Absolute-scale ``(R, SE)`` for ``R = num_bar / den_bar``, stable
    "ratio-residual" delta-method form:
    ``Var(R) = Var(Num - R*Den) / (n * den_bar**2)``.

    Algebraically identical to the direct three-term expansion
    (``var_num/den_bar**2 - 2*R*cov_num_den/den_bar**2 +
    R**2*var_den/den_bar**2``) but evaluated as ONE combined residual at
    the scale of ``Num``'s own variance units, rather than three terms
    each independently divided by ``den_bar**2``/``**3``/``**4`` and
    expected to cancel to the true (small) answer AFTER each has lost
    its own precision at its own scale. The residual cancels once,
    before any division, so the quotient carries only the rounding of
    the one final division -- and it never divides by ``den_bar**3``,
    so a tiny ``den_bar`` that underflows ``den_bar**3`` to exactly
    ``0.0`` cannot raise ``ZeroDivisionError``.

    Reuses :func:`~increment.estimation.armstats.clamp_negative_variance`
    for the residual's own refuse-vs-clamp boundary (the same
    magnitude-relative rounding tolerance every centered sum of squares
    on this seam uses) -- NOT :func:`~increment.estimation.armstats.
    police_sq_sum`'s extra positive-but-below-noise-floor clamp: unlike
    a raw-sum reduction, ``var_num``/``var_den``/``cov_num_den`` arrive
    already reduced, so a small positive residual here is the genuine
    answer, not summation noise, and must not be zeroed.

    Raises ``ValueError`` if the resulting SE is not finite -- the true
    value is not representable in float64 at this scale, which must be
    refused rather than silently returned as ``inf``.
    """
    r = num_bar / den_bar
    cross = 2.0 * r * cov_num_den
    # (r * sqrt(var_den))**2 rather than r*r*var_den: halving each factor's
    # exponent before squaring keeps a representable term from overflowing on
    # the way to it. Zero variance short-circuits, so a huge r costs nothing.
    if var_den == 0.0:
        square = 0.0
    else:
        # Halve each factor's exponent before squaring so a representable term
        # never overflows on the way to itself; refuse when the term genuinely
        # is not representable rather than raising a bare range error.
        rooted = r * math.sqrt(var_den)
        if abs(rooted) > _SQRT_MAX_FLOAT:
            _raise(
                "estimation.variance.ratio_variance_term",
                what=what,
                num_bar=num_bar,
                den_bar=den_bar,
                var_den=var_den,
                rooted=rooted,
            )
        square = rooted * rooted
    residual = var_num - cross + square
    magnitude = abs(var_num) + abs(cross) + abs(square)
    if not math.isfinite(residual) or not math.isfinite(magnitude):
        # NaN fails every comparison below, so an overflowed quadratic form would
        # otherwise clamp to a zero SE -- maximal confidence from a broken number.
        _raise(
            "estimation.variance.ratio_residual_variance",
            what=what,
            num_bar=num_bar,
            den_bar=den_bar,
            residual=residual,
        )
    if residual < 0.0 and -residual <= ratio_residual_slack(magnitude, n):
        clamped: float | None = 0.0
    else:
        clamped = clamp_negative_variance(residual, magnitude=magnitude, n=n)
    if clamped is None:
        _raise("estimation.variance.ratio_residual_variance_negative", what=what, residual=residual)
    # Take each square root separately rather than dividing first: `clamped / n`
    # underflows to exactly zero for a subnormal residual, and a zero SE from a
    # positive variance reads as maximal confidence. sqrt halves the exponent,
    # so both roots stay representable wherever the true SE is.
    root = math.sqrt(clamped) / math.sqrt(n)
    if clamped > 0.0 and root == 0.0:
        _raise(
            "estimation.variance.ratio_se_representable",
            what=what,
            num_bar=num_bar,
            den_bar=den_bar,
            clamped=clamped,
        )
    # Divide by |den_bar| AFTER the square root rather than by den_bar**2 before
    # it: squaring overflows (or underflows to exactly zero) well before the true
    # SE leaves float64, which would refuse a representable answer.
    if root > 0.0 and abs(den_bar) < root / _MAX_FLOAT:
        # The quotient itself is past float64; refuse with the scale named
        # rather than letting the division raise a bare range error.
        _raise(
            "estimation.variance.ratio_se_sqrt_not_representable",
            what=what,
            num_bar=num_bar,
            den_bar=den_bar,
            root=root,
        )
    se = root / abs(den_bar)
    if not math.isfinite(r) or not math.isfinite(se):
        _raise(
            "estimation.variance.ratio_se_final_not_representable",
            what=what,
            num_bar=num_bar,
            den_bar=den_bar,
            r=r,
            se=se,
        )
    return r, se


def cluster_uptake_moments(arm: ArmStats) -> tuple[float, float, float, float, float]:
    """Cluster uptake-ratio (D_j / M_j) moments for the LATE first stage. Refuses unless the x
    slot carries the uptake total (a size there collapses LATE to ITT); uptake_bar may be 0."""
    # Only a declared uptake total can name the first-stage x family. Refuse
    # other declarations before constructing the moments view, which asserts
    # when a copied row has a partial non-uptake family.
    if arm.x_role is not None and arm.x_role not in X_ROLE_VARIABLES:
        _raise(
            "estimation.variance.cluster_robust_late",
            x_role=arm.x_role,
            carries=f"an unrecognized role {arm.x_role!r}",
        )
    if arm.x_role != X_SLOT_ROLES["uptake"]:
        if arm.x_role == X_SLOT_ROLES["size"]:
            carries = "a cluster size"
        elif arm.x_role == X_SLOT_ROLES["x"]:
            carries = "a covariate"
        else:
            carries = "no declared role (a summary predating the declaration)"
        _raise("estimation.variance.cluster_robust_late", x_role=arm.x_role, carries=carries)
    if arm.ref_x is None or arm.cx1 is None or arm.cx2 is None or arm.cxy is None:
        _raise("estimation.variance.cluster_robust_late_needs_cluster_uptake_family")
    if not arm.moments.has("uptake"):
        _raise(
            "estimation.variance.cluster_robust_late",
            x_role=arm.x_role,
            carries="no declared role (a summary predating the declaration)",
        )
    if arm.cxden is None:
        _raise("estimation.variance.cluster_robust_late_needs_cxden")
    if arm.ref_den is None or arm.cden1 is None or arm.cden2 is None:
        _raise("estimation.variance.cluster_uptake_moments_needs_ref_den")
    if arm.n < 2:
        _raise(
            "estimation.variance.cluster_uptake_moments_least_two_units",
            group_id=arm.group_id,
            metric=arm.metric,
            n=arm.n,
        )

    uptake_bar = arm.mean_x()
    size_bar = arm.mean_den()
    if size_bar <= 0.0:
        _raise(
            "estimation.variance.cluster_outcome_moments_nonpositive_size",
            group_id=arm.group_id,
            metric=arm.metric,
            size_bar=size_bar,
        )

    var_uptake = arm.var_x()
    var_size = arm.var_den()
    cov_uptake_size = arm.cov_xden()

    return uptake_bar, size_bar, var_uptake, var_size, cov_uptake_size


def cluster_outcome_moments(arm: ArmStats) -> tuple[float, float, float, float, float]:
    """Per-arm (mean_bar, size_bar, var_mean, var_size, cov_mean_size) for
    the cluster outcome ratio Y_j / M_j - the LATE additive numerator's
    cluster-robust analogue of :func:`ratio_moments`. Unlike that
    function, ``mean_bar`` need not be positive: this is the additive
    Wald-ratio numerator, which never takes ``log()``, so a signed or
    near-zero outcome (profit, an already-differenced metric) must not
    refuse here.
    """
    if arm.ref_den is None or arm.cden1 is None or arm.cden2 is None or arm.cyden is None:
        _raise("estimation.variance.ratio_moments_needs_ref_den")
    if arm.n < 2:
        _raise(
            "estimation.variance.cluster_outcome_moments_least_two_units",
            group_id=arm.group_id,
            metric=arm.metric,
            n=arm.n,
        )

    mean_bar = arm.mean_y()
    size_bar = arm.mean_den()
    if size_bar <= 0.0:
        _raise(
            "estimation.variance.cluster_outcome_moments_nonpositive_size",
            group_id=arm.group_id,
            metric=arm.metric,
            size_bar=size_bar,
        )

    var_mean = arm.var_y()
    var_size = arm.var_den()
    cov_mean_size = arm.cov_yden()

    return mean_bar, size_bar, var_mean, var_size, cov_mean_size


# VarianceModel Protocol


class VarianceModel(Protocol):
    """A variance model for one metric *family*.

    Stateless protocol (not ABC): any object satisfying the interface works.
    """

    consumes: str  # "moments" | "unit_totals" (percentile declares "unit_totals")

    def log_mean_se(self, arm: ArmStats) -> tuple[float, float]:
        """Return (log_mean, se_of_log_mean) for *arm*."""
        ...


# Concrete models


class MeanVarianceModel:
    """Variance model for mean / conversion / retention metrics.

    These differ at the query layer (0/1 vs real-valued) but share the same
    delta-method SE on the log scale::
        Var(log(Y_bar)) approx Var(Y) / (n * Y_bar**2)
    """

    consumes = "moments"
    # The variance is estimated from the arms, so the sampling reference is
    # Welch-Satterthwaite t, not the known-variance Normal.
    supports_welch_reference = True

    def log_mean_se(self, arm: ArmStats) -> tuple[float, float]:
        s = arm.to_summary()
        check_positive_mean(arm.metric, arm.group_id, s.mean)
        return math.log(s.mean), se_log_mean(s.var, s.mean, s.n)


class RatioVarianceModel:
    """Delta-method variance of log(R), R = numerator / denominator.

    Uses the -2*Cov form::

        Var(log(R)) approx (1/n) * [
            Var(Num) / Num_bar**2
            + Var(Den) / Den_bar**2
            - 2 * Cov(Num, Den) / (Num_bar * Den_bar)
        ]

    Asymptotic: undercovers under heavily right-skewed denominators - with
    a lognormal(sigma=1.5) denominator and an independent numerator,
    measured 90.2% coverage at nominal 95% with n=50 units/arm, 92.7% at
    n=100, 93.1% at n=200 and 93.7% at n=400, and worse at sigma=2.0
    (85.1% at n=50, 91.7% at n=400); lognormal(sigma=0.5) is nominal at
    every n (``calibration/ratio_skew.py``). The carried third moment of
    the denominator exposes the regime: the engine emits the coded
    ``estimation.engine.ratio_denominator_skew`` warning and attaches a
    ``ratio_denominator_skew`` row note when either arm's
    :func:`ratio_denominator_mean_skewness` exceeds
    ``RATIO_DENOMINATOR_SKEW_THRESHOLD``, and attaches a
    ``ratio_denominator_precision`` note when either arm's
    :func:`ratio_denominator_precision` exceeds
    ``RATIO_DENOMINATOR_PRECISION_THRESHOLD``. The interval itself is
    unchanged: a flagged row is still anticonservative. Point estimate
    carries O(1/n) Jensen bias (+1-4pp at n<=200, vanishing by n=2000).
    """

    consumes = "moments"
    # Variance is estimated from the arms, so the reference is Welch t. That
    # corrects the small-sample reference only; the delta-method skew
    # undercoverage documented above is flagged, not corrected.
    supports_welch_reference = True

    def log_mean_se(self, arm: ArmStats) -> tuple[float, float]:
        n_bar, d_bar, var_n, var_d, cov_nd = ratio_moments(arm)
        check_positive_mean(arm.metric, arm.group_id, n_bar, what="ratio numerator mean")
        return ratio_log_mean_se(
            n_bar, d_bar, var_n, var_d, cov_nd, arm.n, group_id=arm.group_id, metric=arm.metric
        )


class ClusterVarianceModel(RatioVarianceModel):
    """Cluster-robust variance for a metric randomized at a coarser grain
    than it is measured at.

    A clustered ``group_summary`` row is a ratio row over clusters
    (numerator ``g_j`` cluster total of y, denominator ``m_j`` cluster
    unit count), so this model adds nothing to
    :class:`RatioVarianceModel` - it reduces exactly to
    :class:`MeanVarianceModel` when every cluster is a singleton. A
    clustered ratio metric reuses the same shape with its own
    per-cluster denominator total in place of ``m_j``.

    Selected by the declared cluster, never inferred from row shape.
    Additive intervals use Welch-Satterthwaite degrees of freedom; relative
    Fieller sets use a fixed ``min(K_T - 1, K_C - 1)`` reference. Both are
    working approximations. Each arm needs at least two clusters; fewer
    than 40 total clusters produces a qualified-reference warning, not a
    further admission floor or a finite-sample coverage guarantee.
    """

    consumes = "moments"


# Generic registry for orthogonal dispatch axes


class Registry[T]:
    """A small generic registry for orthogonal dispatch axes.

    ``register(key, value)`` — store a value under a string key.
    ``get(key)`` — retrieve; raises ``NotImplementedError`` naming the
    missing key and all registered keys (helpful debugging).
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._entries: dict[str, T] = {}

    def register(self, key: str, value: T) -> None:
        self._entries[key] = value

    def get(self, key: str) -> T:
        if key not in self._entries:
            available = sorted(self._entries)
            _raise(
                "estimation.variance.registry.no_registered_available",
                key=key,
                registry_name=self._name,
                available=available,
            )
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries


# Pre-built registries

# Metric-type -> VarianceModel dispatch
VARIANCE_MODELS: Registry[VarianceModel] = Registry("variance model")
VARIANCE_MODELS.register("mean", MeanVarianceModel())
VARIANCE_MODELS.register("conversion", MeanVarianceModel())
VARIANCE_MODELS.register("retention", MeanVarianceModel())
VARIANCE_MODELS.register("ratio", RatioVarianceModel())
# Selected via the declared cluster (never a metric type on the wire).
VARIANCE_MODELS.register("cluster", ClusterVarianceModel())

# Method.variance_reduction dispatch
VARIANCE_REDUCTION: Registry[str] = Registry("variance reduction")
VARIANCE_REDUCTION.register("none", "none")
VARIANCE_REDUCTION.register("cuped", "cuped")
