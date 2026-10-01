"""One-way factor absorption - variance reduction on the ATE.

Absorbs a high-cardinality discrete factor (market, device model,
merchant) as a covariate to sharpen the estimate of the overall
treatment effect - CUPED with a categorical covariate. Says nothing
about whether the factor's levels respond differently (a heterogeneity
question, a different module).

One-way random-effects GLS for the treatment contrast::

    y_i = mu + tau * D_i + b_g(i) + e_i,   b_g ~ N(0, sigma_b^2)

Woodbury turns each level's contribution into a rank-1 downdate of a
2x2 normal-equation system, so the fit is O(K) over per-(level, arm)
scalar moments (the shape ``group_summary(by=...)`` emits) - no p x p
Gram matrix is involved.

The pooling weight is estimated from the data: ``sigma_b^2 -> 0``
reproduces the unadjusted difference in means, ``sigma_b^2 -> inf``
the hard within (fixed-effects) estimator. Under the >= 2-units-per-arm
floor this module enforces, hard absorption of a weak factor costs at
most a percent or two of precision against no adjustment, so absorbing
a factor is never a meaningfully worse bet than not absorbing it. A
separate >= 3-usable-levels floor keeps the sandwich's ``t_{K-2}``
reference at a real, non-fabricated degree of freedom (K=2 would need
df=0, undefined).

Inference is a sandwich clustered at the factor level against a
``t_{K-2}`` reference; a homoskedastic sigma^2 would be a severe
coverage regression under skewed allocation (63.8% coverage at 90/10
against a nominal 95%).

References
----------
Deng et al. 2013. "Controlled-experiment Using Pre-Experiment Data."
Searle, Casella, McCulloch 1992. "Variance Components", ch. 3.
Liang & Zeger 1986. "Longitudinal data analysis using generalized
    linear models" (the clustered sandwich).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from increment.errors import (
    IncrementWarning,
    InvalidRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.armstats import centered_sq_sum
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.absorption.one_dimensional_shape": "{name} must be one-dimensional, got shape {arr}",
        "estimation.absorption.contains_non_finite": "{name} contains non-finite values",
        "estimation.absorption.non_negative": "{name} must be non-negative, got {arr}",
        "estimation.absorption.contain_exact_integer": "{name} must contain exact integer unit counts, got {arr}",
        "estimation.absorption.absorption_interval_half": "absorption interval half-width is not representable in float64 (critical value {crit:.6g} x se {se:.6g}) -- the effect is not estimable at this scale",
        "estimation.absorption.pooling_one": "pooling must be one of {pooling_options}, got {pooling!r}",
        "estimation.absorption.all_moment_arrays": "all moment arrays must have the same length, got sizes {lengths}",
        "estimation.absorption.need_least_levels": "need at least {min_levels} levels with >= {min_cell_n} units in both arms; got {n_usable}",
    },
)

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA
_raise = raiser(_REFUSALS)

_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(code: str, warning_type: type[IncrementWarning], render) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "estimation.absorption.treatment_share_confounded",
    IncrementWarning,
    lambda *, share_min, share_max, tau_fe, tau_none: (
        f"treatment share varies across factor levels (min {share_min:.3f}, "
        f"max {share_max:.3f}) and the within (pooling='hard') estimate "
        f"{tau_fe:.6g} disagrees with the unpooled (pooling='none') estimate "
        f"{tau_none:.6g} beyond noise -- the factor is confounded with "
        "assignment, and partial pooling removes only part of that "
        "confounding; use pooling='hard' for this comparison"
    ),
)


# A level needs >= 2 units in both arms: one arm alone carries no
# treatment contrast, and a single unit carries no within-cell variance.
_MIN_CELL_N = 2
# t_{K-2} needs K >= 3 for a real (non-fabricated) degree of freedom;
# K=2 would need df=0, which has no t reference.
_MIN_LEVELS = 3

_POOLING = ("partial", "hard", "none")


class AbsorptionResult(BaseModel):
    """The absorbed average treatment effect and its diagnostics.

    ``effect`` is on the absolute scale (treated mean minus control
    mean, factor absorbed). When the effect is homogeneous across
    levels this is the ATE; when it varies, ``effect`` converges to
    the precision-weighted average of per-level effects (weights
    ``n_t * n_c / n_g``, Angrist 1998), down-weighting skewed-allocation
    levels relative to the unit-weighted ATE.

    ``icc`` is always the estimated variance-component ratio;
    ``mean_shrinkage`` is the pooling weight actually applied, pinned
    to 0.0 or 1.0 when pooling is forced rather than estimated - the
    two can disagree when ``pooling`` is not ``"partial"``.
    """

    model_config = ConfigDict(frozen=True)

    effect: float
    se: float
    lb: float
    ub: float
    level: float
    #: The requested two-sided error rate. Retained because `level` is
    #: `1 - alpha`, which rounds to exactly 1.0 for a small but valid alpha
    #: (1e-20 under a family correction), so a serialized result could not
    #: otherwise recover the significance level it was computed at.
    alpha: float
    n_levels_used: int
    n_levels_dropped: int
    n_units_used: int
    n_units_dropped: int
    icc: float
    mean_shrinkage: float
    se_unadjusted: float

    @property
    def se_reduction(self) -> float:
        """Fractional reduction in standard error against no adjustment.

        Negative means absorbing the factor cost precision.
        """
        if self.se_unadjusted <= 0:
            return 0.0
        return (self.se_unadjusted - self.se) / self.se_unadjusted


def _as_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        _raise("estimation.absorption.one_dimensional_shape", name=name, arr=arr.shape)
    if not np.all(np.isfinite(arr)):
        _raise("estimation.absorption.contains_non_finite", name=name)
    return arr


def _as_count_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate *values* as one-dimensional, finite, non-negative, exact
    integer unit counts -- converted once here so no caller downstream
    needs its own ``round()``/``int()`` conversion of the same array.

    A fractional count rounded in one place (centered moments) and
    truncated in another (``n_units_used``) while the original
    fractional value survives in a third (the GLS weights) produces an
    internally inconsistent result; refusing at ingress is the only
    fix that cannot drift out of sync across the module.
    """
    arr = _as_array(values, name)
    if np.any(arr < 0):
        _raise("estimation.absorption.non_negative", name=name, arr=arr.tolist())
    if not np.all(arr == np.round(arr)):
        _raise("estimation.absorption.contain_exact_integer", name=name, arr=arr.tolist())
    return arr


def _absorption_interval(*, effect: float, crit: float, se: float) -> tuple[float, float]:
    """Two-sided bounds, refusing an unrepresentable half-width.

    A finite critical value times a large finite standard error still
    overflows, so the interval is judged rather than only its critical value:
    returning an infinite bound would contradict this estimator's stated
    finite-interval behavior.
    """
    half_width = crit * se
    if not math.isfinite(half_width):
        _raise("estimation.absorption.absorption_interval_half", crit=crit, se=se)
    return effect - half_width, effect + half_width


def absorb_one_way(  # noqa: PLR0915
    n_control: Sequence[float] | np.ndarray,
    sum_control: Sequence[float] | np.ndarray,
    sumsq_control: Sequence[float] | np.ndarray,
    n_treat: Sequence[float] | np.ndarray,
    sum_treat: Sequence[float] | np.ndarray,
    sumsq_treat: Sequence[float] | np.ndarray,
    *,
    pooling: Literal["partial", "hard", "none"] = "partial",
    alpha: float = 0.05,
) -> AbsorptionResult:
    """Absorb a one-way factor to sharpen the average treatment effect.

    All six arrays are parallel and indexed by factor level: entry
    ``k`` is that level's per-arm count, sum of ``y``, and sum of
    ``y**2``.

    ``pooling="partial"`` (default) estimates the between-level
    variance from the data; ``"hard"`` forces full absorption (the
    within estimator); ``"none"`` forces no absorption. ``"partial"``
    and ``"none"`` are unbiased only when the treatment share is
    constant across levels - if allocation differs by level (staged
    rollouts, per-stratum allocation, observational fixed effects),
    only ``"hard"`` removes the level confounding; use it for those
    designs. ``"partial"`` self-diagnoses the violation: it emits a
    ``UserWarning`` when the treatment share varies across levels and
    the within and unpooled contrasts disagree beyond noise.

    ``alpha`` is the two-sided significance level for the interval
    (default 0.05).

    Raises ``ValueError`` on ragged or non-finite inputs, negative or
    fractional counts, an unknown *pooling* mode, fewer than three
    usable levels, or moment sums too inconsistent to center (a
    level's centered sum of squares comes out negative beyond
    floating-point rounding). Emits a ``RuntimeWarning`` when centering
    hits catastrophic cancellation (mean large relative to spread) but
    still recovers a usable value.
    """
    if pooling not in _POOLING:
        _raise("estimation.absorption.pooling_one", pooling=pooling, pooling_options=_POOLING)
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)

    n_c = _as_count_array(n_control, "n_control")
    s_c = _as_array(sum_control, "sum_control")
    q_c = _as_array(sumsq_control, "sumsq_control")
    n_t = _as_count_array(n_treat, "n_treat")
    s_t = _as_array(sum_treat, "sum_treat")
    q_t = _as_array(sumsq_treat, "sumsq_treat")

    lengths = {a.size for a in (n_c, s_c, q_c, n_t, s_t, q_t)}
    if len(lengths) != 1:
        _raise("estimation.absorption.all_moment_arrays", lengths=sorted(lengths))

    keep = (n_c >= _MIN_CELL_N) & (n_t >= _MIN_CELL_N)
    n_levels_dropped = int((~keep).sum())
    n_units_dropped = int((n_c[~keep] + n_t[~keep]).sum())
    if int(keep.sum()) < _MIN_LEVELS:
        _raise(
            "estimation.absorption.need_least_levels",
            min_levels=_MIN_LEVELS,
            min_cell_n=_MIN_CELL_N,
            n_usable=int(keep.sum()),
        )

    n_c, s_c, q_c = n_c[keep], s_c[keep], q_c[keep]
    n_t, s_t, q_t = n_t[keep], s_t[keep], q_t[keep]
    k = n_c.size

    n_g = n_c + n_t
    s_g = s_c + s_t
    n_total = float(n_g.sum())

    # Variance components, method of moments. Within: pooled
    # ddof-corrected variance across all 2K cells, centred per level
    # through the exact-Fraction helper so a level whose mean dwarfs its
    # spread doesn't cancel to a wrong (or falsely negative) sum of squares.
    within_ss = 0.0
    for i in range(k):
        within_ss += centered_sq_sum(
            float(q_c[i]),
            float(s_c[i]),
            int(n_c[i]),
            what=f"control within-level variance (level {i})",
        )
        within_ss += centered_sq_sum(
            float(q_t[i]),
            float(s_t[i]),
            int(n_t[i]),
            what=f"treat within-level variance (level {i})",
        )
    df_within = n_total - 2.0 * k
    sigma_e2 = within_ss / df_within if df_within > 0 else 0.0

    # Between: one-way ANOVA moment equation on treatment-adjusted level
    # means, centred with the within estimator so tau does not leak in.
    w_fe = n_t * n_c / n_g
    tau_fe = float((w_fe * (s_t / n_t - s_c / n_c)).sum() / w_fe.sum())
    level_mean = s_g / n_g - tau_fe * (n_t / n_g)
    grand = float((n_g * level_mean).sum() / n_total)
    ss_between = float((n_g * (level_mean - grand) ** 2).sum())
    denom = n_total - float((n_g**2).sum()) / n_total
    sigma_b2 = max(0.0, (ss_between - (k - 1) * sigma_e2) / denom) if denom > 0 else 0.0

    if pooling == "hard":
        lam = np.ones(k)
    elif pooling == "none":
        lam = np.zeros(k)
    elif sigma_e2 <= 0:
        lam = np.ones(k)
    else:
        lam = n_g * sigma_b2 / (sigma_e2 + n_g * sigma_b2)

    # Rank-1 GLS per level: A_g = X_g'X_g - (lam_g/n_g)(X_g'1)(1'X_g),
    # X = [1, D]; for binary D this is closed form in the cell counts.
    bread = np.zeros((2, 2))
    rhs = np.zeros(2)
    a_blocks = np.empty((k, 2, 2))
    b_blocks = np.empty((k, 2))
    for i in range(k):
        xtx = np.array([[n_g[i], n_t[i]], [n_t[i], n_t[i]]])
        xt1 = np.array([n_g[i], n_t[i]])
        a_g = xtx - (lam[i] / n_g[i]) * np.outer(xt1, xt1)
        b_g = np.array([s_g[i], s_t[i]]) - (lam[i] / n_g[i]) * xt1 * s_g[i]
        a_blocks[i] = a_g
        b_blocks[i] = b_g
        bread += a_g
        rhs += b_g

    # bread is singular by construction when lam -> 1 (the intercept is
    # fully absorbed); the minimum-norm solution IS the within estimator.
    coef, *_ = np.linalg.lstsq(bread, rhs, rcond=None)
    effect = float(coef[1])

    # Sandwich, clustered at the factor level.
    bread_inv = np.linalg.pinv(bread)
    meat = np.zeros((2, 2))
    for i in range(k):
        score = b_blocks[i] - a_blocks[i] @ coef
        meat += np.outer(score, score)
    # k/(k-2) is undefined at k<=2 and non-monotone near there; k/(k-1)
    # is the standard CR1 finite-sample correction, monotone to 1.
    meat *= k / max(k - 1, 1)
    cov = bread_inv @ meat @ bread_inv
    se = float(np.sqrt(max(cov[1, 1], 0.0)))

    crit = two_sided_critical_value(student_t_isf, alpha, k - 2, what="absorption interval")

    # Unadjusted comparison on the same retained cells.
    nc_tot, nt_tot = float(n_c.sum()), float(n_t.sum())
    mean_c, mean_t = float(s_c.sum()) / nc_tot, float(s_t.sum()) / nt_tot
    var_c = centered_sq_sum(
        float(q_c.sum()), float(s_c.sum()), int(nc_tot), what="unadjusted control variance"
    ) / max(nc_tot - 1.0, 1.0)
    var_t = centered_sq_sum(
        float(q_t.sum()), float(s_t.sum()), int(nt_tot), what="unadjusted treat variance"
    ) / max(nt_tot - 1.0, 1.0)
    se_unadjusted = float(np.sqrt(var_t / nt_tot + var_c / nc_tot))

    # Self-diagnosis: a within/unpooled gap beyond a conservative 2x
    # noise threshold (correlated SEs) signals share-confounded levels.
    if pooling == "partial":
        share = n_t / n_g
        tau_none = mean_t - mean_c
        gap = tau_fe - tau_none
        scale = float(np.hypot(se, se_unadjusted))
        if float(share.max() - share.min()) > 1e-3 and scale > 0.0 and abs(gap) > 2.0 * scale:
            _warn(
                "estimation.absorption.treatment_share_confounded",
                share_min=share.min(),
                share_max=share.max(),
                tau_fe=tau_fe,
                tau_none=tau_none,
                stacklevel=2,
            )

    # sigma_e2 can only go negative through floating-point cancellation;
    # clamp it here too so the emitted icc stays inside its [0, 1] contract.
    total_var = sigma_b2 + max(sigma_e2, 0.0)
    lb, ub = _absorption_interval(effect=effect, crit=crit, se=se)
    return AbsorptionResult(
        effect=effect,
        se=se,
        lb=lb,
        ub=ub,
        level=math.fsum((1.0, -alpha)),
        alpha=alpha,
        n_levels_used=k,
        n_levels_dropped=n_levels_dropped,
        n_units_used=int(n_total),
        n_units_dropped=n_units_dropped,
        icc=min(float(sigma_b2 / total_var), 1.0) if total_var > 0 else 0.0,
        mean_shrinkage=float(lam.mean()),
        se_unadjusted=se_unadjusted,
    )
