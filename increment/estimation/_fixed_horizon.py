"""Fixed-horizon interval references: the resolved alpha, critical value and sampling law.

Estimators and the persisted rows that rebuild their intervals read one resolution, so a
fresh row and a reissued one cut the same interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from increment._literals import ALTERNATIVE_VALUES
from increment.errors import InvalidRequestError, raiser, refusals, refuse
from increment.estimation._tails import student_t_isf, two_sided_critical_value, wald_bounds
from increment.estimation.armstats import _raise as component_refuse
from increment.estimation.binomial_rr import UNKNOWN_ALTERNATIVE
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.risk_difference import score_bounds

if TYPE_CHECKING:
    from increment.estimation.armstats import IndependentMeanReference
    from increment.estimation.priors import Prior

# The codes predate this shared home; estimators raise these same spec objects.
_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.inference.arm_ns_each": "arm_ns must each be > 1 for a Welch-Satterthwaite df, got {arm_ns}",
        "estimation.inference.prior_excludes_welch_reference": "arm_ns={arm_ns!r} requests a Welch-Satterthwaite sampling reference, which cannot be combined with an informative prior: a t critical value on a prior-updated scale is neither a sampling interval nor a posterior interval",
        "estimation.inference.one_sided_alpha_doubles": "one-sided alpha={alpha} doubles to alpha_eff={alpha_eff} >= 1: no displayable two-sided interval exists at that level",
        "estimation.inference.prior_excludes_cluster_robust_t": "prior and a cluster-robust t or joint reference are mutually exclusive: the available covariance does not specify the requested Normal conjugate update",
        "estimation.inference.infer_lift_abs_diff_finite": "abs_diff must be finite, got {abs_diff}",
        "estimation.inference.infer_lift_abs_se_finite": "abs_se must be finite, got {abs_se}",
        "estimation.inference.infer_lift_abs_se_positive": "abs_se must be > 0.0, got {abs_se}",
        "estimation.encouragement.alpha_eff_too": "alpha_eff is too small: alpha_eff / 2 underflows floating-point precision",
        "estimation.encouragement.dof_positive_reference": "dof must be positive for a t reference, got {dof}",
    },
)
_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA
_raise = raiser(_REFUSALS)

ARM_NS_EACH = _REFUSALS["estimation.inference.arm_ns_each"]
PRIOR_EXCLUDES_WELCH_REFERENCE = _REFUSALS["estimation.inference.prior_excludes_welch_reference"]
ONE_SIDED_ALPHA_DOUBLES = _REFUSALS["estimation.inference.one_sided_alpha_doubles"]
PRIOR_EXCLUDES_CLUSTER_ROBUST_T = _REFUSALS["estimation.inference.prior_excludes_cluster_robust_t"]
INFER_LIFT_ABS_DIFF_FINITE = _REFUSALS["estimation.inference.infer_lift_abs_diff_finite"]
INFER_LIFT_ABS_SE_FINITE = _REFUSALS["estimation.inference.infer_lift_abs_se_finite"]
INFER_LIFT_ABS_SE_POSITIVE = _REFUSALS["estimation.inference.infer_lift_abs_se_positive"]
ALPHA_EFF_TOO = _REFUSALS["estimation.encouragement.alpha_eff_too"]
DOF_POSITIVE_REFERENCE = _REFUSALS["estimation.encouragement.dof_positive_reference"]


def _alpha_eff_for(alternative: str, alpha: float) -> float:
    """The central-equivalent alpha a call-level *alpha* displays under.

    The inverse of ``_fcr_alpha_for``: directional alternatives double it.
    """
    return alpha if alternative == "two-sided" else 2.0 * alpha


@dataclass(frozen=True, slots=True)
class SamplingReference:
    """Distribution used to cut an interval or sequential boundary."""

    kind: Literal["normal", "t", "sequential"]
    df: float | None


@dataclass(frozen=True, slots=True)
class FixedHorizonReference:
    """Resolved alpha, critical value, and sampling reference."""

    alpha_eff: float
    crit: float | None
    reference: SamplingReference


def _resolve_fixed_horizon(
    alpha: float,
    alternative: str,
    *,
    dof: float | None,
    arm_ns: tuple[int, int] | None,
    se_t: float | None = None,
    se_c: float | None = None,
    prior: Prior | None,
    independent_means: IndependentMeanReference | None = None,
) -> FixedHorizonReference:
    """Validate shared options and resolve one interval reference."""
    from scipy.stats import norm as _norm

    if alternative not in ALTERNATIVE_VALUES:
        refuse(UNKNOWN_ALTERNATIVE, alternative=alternative)
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)
    alpha_eff = _alpha_eff_for(alternative, alpha)
    if alpha_eff >= 1.0:
        _raise("estimation.inference.one_sided_alpha_doubles", alpha=alpha, alpha_eff=alpha_eff)
    if alpha_eff / 2.0 == 0.0:
        refuse(ALPHA_EFF_TOO)
    if independent_means is not None:
        if dof is not None or arm_ns is not None or prior is not None:
            component_refuse(
                "estimation.armstats.independent_mean_contract",
                reason="Independent mean reference cannot be combined with another sampling reference.",
            )
        df = independent_means.df
        return FixedHorizonReference(
            alpha_eff=alpha_eff,
            crit=two_sided_critical_value(
                student_t_isf, alpha_eff, df, what="independent mean Welch reference"
            ),
            reference=SamplingReference(kind="t", df=df),
        )
    if dof is not None:
        if dof <= 0:
            refuse(DOF_POSITIVE_REFERENCE, dof=dof)
        if prior is not None:
            _raise("estimation.inference.prior_excludes_cluster_robust_t")
    if dof is not None:
        crit = two_sided_critical_value(
            student_t_isf,
            alpha_eff,
            dof,
            what="infer_lift dof reference",
        )
        reference = SamplingReference(kind="t", df=dof)
    elif arm_ns is not None:
        n_t, n_c = arm_ns
        if n_t <= 1 or n_c <= 1:
            _raise("estimation.inference.arm_ns_each", arm_ns=arm_ns)
        if prior is not None:
            _raise("estimation.inference.prior_excludes_welch_reference", arm_ns=arm_ns)
        assert se_t is not None and se_c is not None, "unreachable: arm_ns callers pass se_t/se_c"
        # Welch-Satterthwaite df is a ratio of degree-4 homogeneous SE polynomials, so dividing
        # by the larger SE cancels algebraically while keeping terms near O(1) instead of
        # under/overflowing at extreme SE magnitudes.
        se_scale = max(se_t, se_c)
        a, b = se_t / se_scale, se_c / se_scale
        welch_df = (a * a + b * b) ** 2 / (a**4 / (n_t - 1) + b**4 / (n_c - 1))
        crit = two_sided_critical_value(
            student_t_isf,
            alpha_eff,
            welch_df,
            what="infer_lift Welch-Satterthwaite reference",
        )
        reference = SamplingReference(kind="t", df=welch_df)
    else:
        crit = two_sided_critical_value(
            _norm.isf,
            alpha_eff,
            what="infer_lift Normal reference",
        )
        reference = SamplingReference(kind="normal", df=None)
    return FixedHorizonReference(alpha_eff=alpha_eff, crit=crit, reference=reference)


def _joint_additive_bounds(
    point: float | None, se: float | None, alpha: float, alternative: str, df: float | None
) -> tuple[float | None, float | None]:
    if point is None or se is None:
        return None, None
    if not math.isfinite(point):
        refuse(INFER_LIFT_ABS_DIFF_FINITE, abs_diff=point)
    if not math.isfinite(se):
        refuse(INFER_LIFT_ABS_SE_FINITE, abs_se=se)
    if se < 0.0:
        refuse(INFER_LIFT_ABS_SE_POSITIVE, abs_se=se)
    if se == 0.0:
        return None, None
    reference = _resolve_fixed_horizon(alpha, alternative, dof=df, arm_ns=None, prior=None)
    return (
        wald_bounds(point, reference.crit, se, what="joint additive sidecar")
        if reference.crit is not None
        else (None, None)
    )


def binomial_abs_interval(
    x_c: int, n_c: int, x_t: int, n_t: int, alpha_eff: float
) -> tuple[float, float, Literal["score"]]:
    """``(abs_lb, abs_ub, abs_reference_kind)`` of a binomial row at its central-equivalent
    ``alpha_eff``: the Miettinen-Nurminen score interval for ``p_t - p_c`` from the four
    counts. Every binomial row's additive bounds come from here, so a fresh row, a family
    reissue and a rerouted delta row agree. The critical value takes the shared survival
    form: an alpha below its resolution refuses with ``estimation.tails.unresolvable``."""
    from scipy.stats import norm

    crit = two_sided_critical_value(norm.isf, alpha_eff, what="binomial additive interval")
    lower, upper = score_bounds(x_c, n_c, x_t, n_t, crit)
    return lower, upper, "score"
