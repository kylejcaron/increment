"""Bayesian Normal-Normal inference for lift estimation.

``Normal`` is a small frozen pydantic model for scalar distributions;
``normal_posterior`` performs a conjugate update.

``infer_lift`` is the shared tail for every delta-metric lift: a joint
``log_rr`` point estimate and the two per-arm standard errors update a
flat Normal prior, returning a ``LiftEstimate`` on the RELATIVE scale
from the closed-form quantile of the log-scale posterior.

``infer_ate`` is its sibling for observational (IPTW/DML/AIPW) estimands:
one contrast's collapsed ``ScoreStats`` is already a point/SE pair on the
reported scale, so the update runs directly on the ADDITIVE scale - no
log transform, and lifts ``<= -1`` are representable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import norm as _norm

from increment._literals import (
    ALTERNATIVE_VALUES,
    VALUE_SCALE_VALUES,
    Alternative,
    PreferredDirection,
    ValueScale,
)
from increment.errors import CodedError, InvalidRequestError, RefusalSpec, raiser, refusals, refuse
from increment.estimation.binomial_rr import UNKNOWN_ALTERNATIVE
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.priors import (
    EXPECTED_NEGATIVE_PART_SCALE,
    EXPECTED_POSITIVE_PART_SCALE,
    MixturePrior,
    StudentTPrior,
    mixture_posterior,
)

if TYPE_CHECKING:
    from increment._readout_request import ReadoutRequest

from increment.estimation._tails import (
    resolvable_expm1,
    student_t_isf,
    two_sided_critical_value,
    wald_bounds,
)
from increment.estimation.armstats import (
    ArmStats,
    IndependentMeanComponent,
    IndependentMeanReference,
    ScoreStats,
)
from increment.estimation.results import (
    Estimate,
    JointContrastReference,
    LiftEstimate,
    RelativeUnavailableReason,
    relative_confidence_set,
)
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.inference.arm_ns_each": "arm_ns must each be > 1 for a Welch-Satterthwaite df, got {arm_ns}",
        "estimation.inference.prior_excludes_welch_reference": "arm_ns={arm_ns!r} requests a Welch-Satterthwaite sampling reference, which cannot be combined with an informative prior: a t critical value on a prior-updated scale is neither a sampling interval nor a posterior interval",
        "estimation.inference.null_lift_log": "null_lift must be > -1.0 (log scale), got {null_lift}",
        "estimation.diagnostics.alpha": "alpha must be in (0, 1), got {alpha}",
        "estimation.inference.one_sided_alpha_doubles": "one-sided alpha={alpha} doubles to alpha_eff={alpha_eff} >= 1: no displayable two-sided interval exists at that level",
        "estimation.inference.prior_excludes_cluster_robust_t": "prior and a cluster-robust t or joint reference are mutually exclusive: the available covariance does not specify the requested Normal conjugate update",
        "estimation.inference.prior_excludes_joint_reference": "an informative prior cannot be combined with a joint frequentist relative set",
        "estimation.inference.joint_reference_invalid": RefusalSpec(
            "estimation.inference.joint_reference_invalid",
            InvalidRequestError,
            lambda *, reason: reason,
        ),
        "estimation.inference.infer_lift_abs_diff_finite": "abs_diff must be finite, got {abs_diff}",
        "estimation.inference.infer_lift_abs_se_finite": "abs_se must be finite, got {abs_se}",
        "estimation.inference.infer_lift_abs_se_positive": "abs_se must be > 0.0, got {abs_se}",
        "estimation.inference.unknown_value_scale": 'Unknown value_scale={value_scale!r}: must be "relative" or "absolute"',
        "estimation.inference.infer_ate_null_abs_on_absolute_metric": RefusalSpec(
            "estimation.inference.infer_ate_null_abs_on_absolute_metric",
            InvalidRequestError,
            lambda *, name: ABSOLUTE_NULL_ABS_REFUSAL.format(name=name),
        ),
        "estimation.inference.infer_ate_null_lift_on_absolute_metric": RefusalSpec(
            "estimation.inference.infer_ate_null_lift_on_absolute_metric",
            InvalidRequestError,
            lambda *, name: ABSOLUTE_NULL_LIFT_REFUSAL.format(name=name),
        ),
        "estimation.inference.degenerate_data_zero": "Degenerate data: zero variance, cannot estimate lift uncertainty",
        "estimation.inference.abs_dof_requires_dof": "abs_dof is only meaningful alongside a cluster-robust dof (the absolute sidecar's own reference df exists only when the primary contrast is already on the cluster-robust t-reference path); got abs_dof set with dof=None.",
        "estimation.inference.infer_ate_null_lift_finite": "null_lift must be finite, got {null_lift}",
        "estimation.inference.infer_ate_null_abs_finite": "null_abs must be finite, got {null_abs}",
    },
)

INFER_LIFT_ABS_DIFF_FINITE = _REFUSALS["estimation.inference.infer_lift_abs_diff_finite"]
INFER_LIFT_ABS_SE_FINITE = _REFUSALS["estimation.inference.infer_lift_abs_se_finite"]
INFER_LIFT_ABS_SE_POSITIVE = _REFUSALS["estimation.inference.infer_lift_abs_se_positive"]
INFER_ATE_ABS_DIFF_FINITE = INFER_LIFT_ABS_DIFF_FINITE
INFER_ATE_ABS_SE_FINITE = INFER_LIFT_ABS_SE_FINITE
INFER_ATE_ABS_SE_POSITIVE = INFER_LIFT_ABS_SE_POSITIVE
INFER_ATE_NULL_LIFT_FINITE = _REFUSALS["estimation.inference.infer_ate_null_lift_finite"]
INFER_ATE_NULL_ABS_FINITE = _REFUSALS["estimation.inference.infer_ate_null_abs_finite"]
ONE_SIDED_ALPHA_DOUBLES = _REFUSALS["estimation.inference.one_sided_alpha_doubles"]

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA
_raise = raiser(_REFUSALS)

# An absolute-native row carries no abs_diff/abs_se sidecar (it would
# merely re-represent ``lift``), so an absolute margin on one reads empty.
ABSOLUTE_NULL_ABS_REFUSAL = (
    "margins_abs/null_abs cannot target metric {name!r}: its rows are "
    "absolute-native (value_scale='absolute'), so the abs_diff/abs_se fields the "
    "absolute-margin decision reads are None by design. A shifted null for an "
    "absolute-native row is a null_lift in the metric's own units, which this row "
    "does not support. Testing against 0 with alternative= runs but drops the "
    "margin, so it answers a different question."
)

# A relative shifted null on an absolute-native row compares a unitless
# fraction against additive units - a silent scale mismatch; zero is exempt.
ABSOLUTE_NULL_LIFT_REFUSAL = (
    "null_lifts/margins cannot target metric {name!r} with a nonzero value: its "
    "rows are absolute-native (value_scale='absolute'), so `lift` is in the "
    "metric's own units and a unitless relative null would be compared against "
    "an additive interval. A nonzero shifted null is not available on an "
    "absolute-native row. Testing against 0 with alternative= runs but drops the "
    "margin, so it answers a different question."
)


class LiftGuardError(CodedError):
    """A data-quality guard in ``infer_lift`` refused to estimate.

    ``reason`` is a stable short code identifying which guard fired; callers
    bucket on it rather than parsing ``str(self)``.
    """

    _CODE = "estimation.inference.lift_guard"

    def __init__(
        self,
        message: str,
        *,
        reason: Literal[
            "zero_variance",
            "nonfinite_se",
            "delta_method_unreliable",
            "non_positive_mean",
        ],
    ) -> None:
        super().__init__(message, code=self._CODE, context={"reason": reason})
        self.reason = reason


@runtime_checkable
class LiftPosterior(Protocol):
    """Distribution operations consumed by fixed-horizon decision statistics."""

    def cdf(self, value: float) -> float: ...

    def survival(self, value: float) -> float: ...

    def quantile(self, probability: float) -> float: ...

    def isf(self, tail_probability: float) -> float: ...

    def probability_between(self, lower: float, upper: float) -> float: ...

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float: ...

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float: ...


class Normal(BaseModel):
    """Normal distribution: scalar parameters (no array support needed;
    estimates are scalar per metric x method x arm).
    """

    model_config = ConfigDict(frozen=True)

    mu: float
    sigma: float = Field(gt=0)  # positivity via pydantic validation

    @property
    def precision(self) -> float:
        """Inverse variance."""
        return 1.0 / self.sigma**2

    def cdf(self, value: float) -> float:
        """Probability mass at or below ``value``."""
        return float(_norm.cdf((value - self.mu) / self.sigma))

    def survival(self, value: float) -> float:
        """Probability mass strictly above ``value``."""
        return float(_norm.sf((value - self.mu) / self.sigma))

    def quantile(self, probability: float) -> float:
        """Value whose cumulative probability is ``probability``."""
        return self.mu + self.sigma * float(_norm.ppf(probability))

    def isf(self, tail_probability: float) -> float:
        """Value whose upper-tail probability is ``tail_probability``."""
        return self.mu + self.sigma * float(_norm.isf(tail_probability))

    def probability_between(self, lower: float, upper: float) -> float:
        """Probability mass between two inclusive bounds."""
        if lower >= self.mu:
            return self.survival(lower) - self.survival(upper)
        return self.cdf(upper) - self.cdf(lower)

    def expected_negative_part(self, *, scale: Literal["linear", "log"]) -> float:
        """Expected magnitude below zero on the requested reporting scale."""
        z = self.mu / self.sigma
        if scale == "linear":
            return float(self.sigma * _norm.pdf(z) - self.mu * _norm.cdf(-z))
        if scale == "log":
            a = -z
            return float(
                _norm.cdf(a) - math.exp(self.mu + 0.5 * self.sigma**2) * _norm.cdf(a - self.sigma)
            )
        refuse(EXPECTED_NEGATIVE_PART_SCALE, scale=scale)

    def expected_positive_part(self, *, scale: Literal["linear", "log"]) -> float:
        """Expected magnitude above zero on the requested reporting scale."""
        z = self.mu / self.sigma
        if scale == "linear":
            return float(self.sigma * _norm.pdf(z) + self.mu * _norm.cdf(z))
        if scale == "log":
            a = -z
            return float(
                math.exp(self.mu + 0.5 * self.sigma**2) * _norm.cdf(self.sigma - a) - _norm.cdf(-a)
            )
        refuse(EXPECTED_POSITIVE_PART_SCALE, scale=scale)


type Prior = Normal | StudentTPrior | MixturePrior
# Composes with a mixture everywhere except normal_posterior (the single-
# Normal update) and paths that don't persist state to recompute it.

_DEFAULT_PRIOR = Normal(mu=0.0, sigma=1e6)


def validate_readout_inference(request: ReadoutRequest) -> None:
    """Validate static prior/inference combinations before moments are read."""
    from increment.estimation.decision_types import FixedInference

    plan = request.plan
    inference = plan.inference
    configs = tuple(request.configs)
    if not isinstance(inference, FixedInference) and any(
        getattr(config, "prior", None) is not None for config in configs
    ):
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(
            Unsupported("arm.adjustment.sequential_prior"),
            metrics=tuple(c.metric.name for c in configs if getattr(c, "prior", None) is not None),
        )


def normal_posterior(
    estimate: float,
    standard_error: float,
    prior: Normal | None = None,
) -> Normal:
    """Return the conjugate posterior for a normally distributed estimate."""
    observed = Normal(mu=estimate, sigma=standard_error)
    base = prior if prior is not None else _DEFAULT_PRIOR
    # Form the posterior around the smaller scale so no absolute sigma is
    # ever squared: squaring/hypot-ing raw sigmas overflows or underflows
    # to zero long before the sigmas themselves become unrepresentable.
    small, large = (base, observed) if base.sigma <= observed.sigma else (observed, base)
    r = small.sigma / large.sigma
    if r == 0.0:
        # r underflowed to exactly zero: small.sigma is the representable
        # dominant limit, and large's contribution is unresolvable at
        # this scale.
        return small
    denominator = 1.0 + r * r
    posterior_sigma = small.sigma / math.hypot(1.0, r)
    if (small.mu >= 0.0) == (large.mu >= 0.0):
        # Same sign: difference first to avoid catastrophic cancellation
        # when small.mu and large.mu are both large.
        delta = large.mu - small.mu
        posterior_mu = small.mu + r * ((r * delta) / denominator)
    else:
        posterior_mu = small.mu / denominator + r * ((r * large.mu) / denominator)
    return Normal(mu=posterior_mu, sigma=posterior_sigma)


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
    if alternative not in ALTERNATIVE_VALUES:
        refuse(UNKNOWN_ALTERNATIVE, alternative=alternative)
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)
    alpha_eff = alpha if alternative == "two-sided" else 2.0 * alpha
    if alpha_eff >= 1.0:
        _raise("estimation.inference.one_sided_alpha_doubles", alpha=alpha, alpha_eff=alpha_eff)
    if alpha_eff / 2.0 == 0.0:
        from increment.estimation.encouragement import ALPHA_EFF_TOO

        refuse(ALPHA_EFF_TOO)
    if independent_means is not None:
        from increment.estimation.armstats import _raise as component_refuse

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
            from increment.estimation.encouragement import DOF_POSITIVE_REFERENCE

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


# Shared lift-inference function


# Public inference signature is the API for estimate construction.
def infer_lift(  # noqa: PLR0913
    metric: str,
    group_id: str,
    method: str,
    log_rr: float,
    se_t: float,
    se_c: float,
    prior: Prior | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    inference_spec: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    n_comparison: int | None = None,
    abs_diff: float | None = None,
    abs_se: float | None = None,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    preferred_direction: PreferredDirection | None = None,
    dof: float | None = None,
    arm_ns: tuple[int, int] | None = None,
    n_clusters: int | None = None,
    *,
    abs_dof: float | None = None,
    method_role: Literal["decision", "sensitivity"],
) -> LiftEstimate:
    """Infer a relative lift from a caller-supplied joint log ratio and
    per-arm log-scale SEs: the shared tail for every delta-metric lift
    computation.

    ``log_rr`` is the caller's ``log(point_t / point_c)``.  Callers with raw
    arm point estimates should form it jointly (``stable_log_ratio`` for a
    single point per arm, or the ratio metric's exact component cross-ratio)
    rather than subtracting independently rounded per-arm logs.  Independent
    logs carry an absolute error of ``|log(point)| * eps`` that can swallow a
    small effect at a large offset.  ``se_t``/``se_c`` remain the per-arm
    delta-method SEs of ``log(point)``, which the offset does not disturb.

    Combines a flat (or supplied) Normal prior on the log risk-ratio with
    the arms' SEs via ``normal_posterior`` (or a K-component mixture
    posterior when ``prior`` is a ``StudentTPrior``/``MixturePrior``), then
    returns the closed-form quantile of that posterior, back-transformed
    by ``exp(x) - 1``. Because ``exp`` is strictly increasing, the
    transform of the quantile equals the quantile of the transform, so
    the CI is exact, not sampled.

    For ``alternative != "two-sided"``, the interval is computed at
    ``alpha_eff = 2 * alpha`` through the same two-sided path (both
    bounds kept), labeled with the honest two-sided coverage ``1 -
    alpha_eff`` and the caller's direction - the standard one-sided
    display convention. ``prior`` composes with a one-sided
    ``alternative``, but is mutually exclusive with ``inference_spec``
    (frequentist sequential coverage would be voided by a prior-shifted
    center) and with a cluster-robust ``dof`` (no Normal-Normal conjugate
    update exists against a t reference; the raw stats become the
    inference inputs directly).

    ``null_lift``/``null_abs`` are decision metadata only (stamped onto
    the result for ``stat_sig``/``prob_favorable``) and never shift the
    posterior or either interval; both must be finite, and ``null_lift``
    must be ``> -1`` (the log scale has no representation at or below
    total loss).

    Refuses (``LiftGuardError``) when either arm has zero variance or a
    non-finite SE, or the combined log-scale SE is ``>= 0.5`` (a 95% interval already
    spans a ~7x ratio range there, past where the log-Normal
    approximation holds). These guards do not catch the ratio-of-means'
    first-order small-sample bias, approximately ``(1 + lift) *
    CV_c**2 / n_c``, material only in the low-event-count regime.

    ``arm_ns``, when supplied (each arm's unit count), swaps the fixed-
    horizon Normal critical value for a Welch-Satterthwaite t reference
    fitted to ``se_t``/``se_c`` -- better small-n calibration than the
    Normal approximation, which is anticonservative there, though still
    approximate rather than exact; omit it to keep the Normal approximation.

    ``abs_diff``/``abs_se``, when supplied, get their own prior-free Wald
    interval (fixed-horizon only) alongside the relative one. A Welch
    reference requires an independent ``abs_dof`` to publish those bounds:
    log-scale degrees of freedom do not define an additive reference. The
    sidecar's reference is persisted separately from the primary reference.
    """
    if inference_spec is not None:
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "SE-only sequential inference was removed; use estimate_sequential with registered exact state",
        )
    # A zero-variance arm silently drops its uncertainty from se_log_rr,
    # collapsing coverage; refuse per-arm, not only when both are degenerate.
    if se_t <= 0 or se_c <= 0:
        raise LiftGuardError(
            "Degenerate data: an arm has zero variance (e.g. all-identical "
            "outcomes or a saturated conversion arm), cannot estimate lift "
            "uncertainty for that arm",
            reason="zero_variance",
        )

    if not math.isfinite(se_t) or not math.isfinite(se_c):
        raise LiftGuardError(
            "Cannot estimate lift uncertainty from a non-finite arm SE",
            reason="nonfinite_se",
        )

    # Combine SEs (arms are independent).
    se_log_rr = math.hypot(se_t, se_c)

    # The log-Normal approximation degrades with se_log_rr, not with the
    # ratio; at 0.5 a 95% interval already spans ~7x, past where it holds.
    if se_log_rr >= 0.5:
        raise LiftGuardError(
            f"Log delta method unreliable: combined log-scale SE "
            f"{se_log_rr:.3f} >= 0.5 (too few events / too small arms)",
            reason="delta_method_unreliable",
        )

    ref = _resolve_fixed_horizon(
        alpha,
        alternative,
        dof=dof,
        arm_ns=arm_ns,
        se_t=se_t,
        se_c=se_c,
        prior=prior,
    )
    alpha_eff = ref.alpha_eff
    if not math.isfinite(null_lift):
        refuse(INFER_ATE_NULL_LIFT_FINITE, null_lift=null_lift)
    if null_lift <= -1.0:
        _raise("estimation.inference.null_lift_log", null_lift=null_lift)
    if null_abs is not None and not math.isfinite(null_abs):
        refuse(INFER_ATE_NULL_ABS_FINITE, null_abs=null_abs)

    mixture_post = None
    if dof is not None:
        # Cluster-robust path: the raw stats ARE the inference inputs; no
        # conjugate update (see the dof parameter's docstring above).
        mu_n = log_rr
        sigma_n = se_log_rr
    elif isinstance(prior, (StudentTPrior, MixturePrior)):
        # Mixture prior: quantiles of the K-component posterior replace
        # mu +/- z*sigma; back-transform stays exact (exp is increasing).
        mixture_post = mixture_posterior(log_rr, se_log_rr, prior.components())
        mu_n = mixture_post.quantile(0.5)
        sigma_n = float("nan")  # never read on this branch
    else:
        # Conjugate Normal update
        posterior = normal_posterior(log_rr, se_log_rr, prior=prior)
        mu_n = posterior.mu
        sigma_n = posterior.sigma

    # exp(mu_n) - 1 is the standard delta-method lift (the CI's closed
    # form); averaging E[exp(X)-1] instead would inflate with SE alone.
    assert ref.crit is not None
    half_width = ref.crit * sigma_n
    inference_label = "fixed"

    if mixture_post is not None:
        value = resolvable_expm1(mu_n, what="infer_lift relative point estimate")
        lb = resolvable_expm1(
            mixture_post.quantile(alpha_eff / 2.0), what="infer_lift mixture posterior lower bound"
        )
        ub = resolvable_expm1(
            mixture_post.isf(alpha_eff / 2.0), what="infer_lift mixture posterior upper bound"
        )
    else:
        lower_log, upper_log = wald_bounds(
            mu_n, half_width, 1.0, what="infer_lift relative interval"
        )
        value = resolvable_expm1(mu_n, what="infer_lift relative point estimate")
        lb = resolvable_expm1(lower_log, what="infer_lift relative interval lower bound")
        ub = resolvable_expm1(upper_log, what="infer_lift relative interval upper bound")

    # Additive Wald endpoints: the prior-free absolute-scale reading,
    # fixed-horizon only (no sequential coverage guarantee exists for them).
    abs_lb = abs_ub = None
    abs_ref = None
    if inference_spec is None and abs_diff is not None and abs_se is not None:
        if not math.isfinite(abs_diff):
            refuse(INFER_LIFT_ABS_DIFF_FINITE, abs_diff=abs_diff)
        if not math.isfinite(abs_se):
            refuse(INFER_LIFT_ABS_SE_FINITE, abs_se=abs_se)
        if abs_se <= 0.0:
            refuse(INFER_LIFT_ABS_SE_POSITIVE, abs_se=abs_se)
        if abs_dof is not None or ref.reference.kind != "t":
            abs_ref = _resolve_fixed_horizon(
                alpha, alternative, dof=abs_dof, arm_ns=None, prior=None
            )
            assert abs_ref.crit is not None
            abs_lb, abs_ub = wald_bounds(
                abs_diff, abs_ref.crit, abs_se, what="infer_lift absolute interval"
            )

    estimate = Estimate(
        value=value,
        lb=lb,
        ub=ub,
        level=math.fsum((1.0, -alpha_eff)),
        alpha=alpha_eff,
        # Raw pre-update stats (log_rr/se_log_rr), not the conjugately
        # updated posterior - the two differ only under an informative prior.
        log_mean=log_rr,
        log_se=se_log_rr,
    )

    return LiftEstimate(
        metric=metric,
        group_id=group_id,
        method=method,
        method_role=method_role,
        inference=inference_label,
        alternative=alternative,
        null_lift=null_lift,
        preferred_direction=preferred_direction,
        lift=estimate,
        prior_shrunk=prior is not None,
        prior_spec=prior if isinstance(prior, (StudentTPrior, MixturePrior)) else None,
        scale="log",
        abs_diff=abs_diff,
        abs_se=abs_se,
        null_abs=null_abs,
        abs_lb=abs_lb,
        abs_ub=abs_ub,
        abs_reference_kind=("t" if abs_ref.reference.df is not None else "normal")
        if abs_ref is not None
        else None,
        abs_reference_df=abs_ref.reference.df if abs_ref is not None else None,
        n_clusters=n_clusters,
        dof=dof,
        reference_kind=ref.reference.kind,
        reference_df=ref.reference.df,
    )


def _joint_additive_bounds(
    point: float | None, se: float | None, alpha: float, alternative: str, df: float | None
) -> tuple[float | None, float | None]:
    if point is None or se is None:
        return None, None
    if not math.isfinite(point):
        refuse(INFER_ATE_ABS_DIFF_FINITE, abs_diff=point)
    if not math.isfinite(se):
        refuse(INFER_ATE_ABS_SE_FINITE, abs_se=se)
    if se < 0.0:
        refuse(INFER_ATE_ABS_SE_POSITIVE, abs_se=se)
    if se == 0.0:
        return None, None
    reference = _resolve_fixed_horizon(alpha, alternative, dof=df, arm_ns=None, prior=None)
    return (
        wald_bounds(point, reference.crit, se, what="joint additive sidecar")
        if reference.crit is not None
        else (None, None)
    )


def _validate_additive_sidecar(abs_diff: float, abs_se: float) -> None:
    """Require finite additive inputs and positive uncertainty."""
    if not math.isfinite(abs_diff):
        refuse(INFER_ATE_ABS_DIFF_FINITE, abs_diff=abs_diff)
    if not math.isfinite(abs_se):
        refuse(INFER_ATE_ABS_SE_FINITE, abs_se=abs_se)
    if abs_se <= 0.0:
        refuse(INFER_ATE_ABS_SE_POSITIVE, abs_se=abs_se)


# Public inference signature is the API for estimate construction.
def infer_ate(  # noqa: PLR0913, PLR0915
    metric: str,
    group_id: str,
    method: str,
    point: float | None,
    scores: ScoreStats | IndependentMeanReference,
    prior: Prior | None = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    population: str | None = None,
    null_lift: float = 0.0,
    preferred_direction: PreferredDirection | None = None,
    abs_diff: float | None = None,
    abs_se: float | None = None,
    null_abs: float | None = None,
    value_scale: ValueScale = "relative",
    dof: float | None = None,
    n_clusters: int | None = None,
    estimand: Literal["ate", "plr_slope", "overlap_subpopulation_ate"] = "ate",
    abs_dof: float | None = None,
    joint_reference: JointContrastReference | None = None,
    relative_unavailable_reason: RelativeUnavailableReason | None = None,
    *,
    method_role: Literal["decision", "sensitivity"],
) -> LiftEstimate:
    """Infer one contrast's lift from its collapsed influence scores.

    ``IndependentMeanReference`` explicitly selects unweighted independent
    means on the absolute scale, with within-component ddof=1 and Welch df.
    Adjusted ``ScoreStats`` retain their own normal or cluster reference.

    Unlike ``infer_lift`` (two independent arms on the LOG scale, needing
    a transform plus degeneracy/precision guards), ``infer_ate`` operates
    directly on the ADDITIVE (already-relative-lift) scale: ``scores``
    already collapses one contrast's per-unit influence function into a
    single lift-scale point/SE pair, so lifts ``<= -1`` are representable
    (there is no log floor).

    ``value_scale`` picks what ``lift.value`` reports: the relative lift
    (default, with the additive pair riding alongside as
    ``abs_diff``/``abs_se``), or - when the control mean is
    indistinguishable from 0 and the relative scale is unidentified - the
    additive ATE itself in the metric's own units. On an ``"absolute"``
    row with ``prior=None`` the conjugate update is skipped and the
    posterior is ``Normal(point, scores.se())`` verbatim:
    ``normal_posterior``'s ``Normal(0, 1e6)`` default is only near-flat
    in unitless terms, and would otherwise silently shrink a large-SE
    additive estimate toward 0.

    When ``joint_reference`` is available, it is authoritative for the additive
    point and SE as well as relative inference; redundant caller ``abs_diff``
    and ``abs_se`` inputs are replaced by its canonical projection. Zero
    variance retains a numeric-null SE and no additive bounds. ``abs_dof``
    selects a separate additive reference (Normal when omitted on this path).

    ``prior`` here is on the relative-lift scale (not log-RR); a mixture
    prior is refused since this path doesn't persist the raw pre-prior
    statistics a mixture posterior needs - use ``infer_lift`` instead.
    ``alternative``/``alpha_eff`` follow the same one-sided doubling
    convention as ``infer_lift``, with no log/exp transform to carry
    through. ``null_lift`` has no floor at -1 and, like ``null_abs``, is
    decision metadata only; both must be finite and are refused on
    ``"absolute"`` rows where the corresponding sidecar field is ``None``.

    ``dof``/``n_clusters`` mirror ``infer_lift``'s cluster-robust path: a
    set ``dof`` bypasses the conjugate update entirely (``mu = point,
    sigma = scores.se()``) in favor of a t reference, and is mutually
    exclusive with an informative ``prior``. Outside the joint-reference path,
    ``abs_dof`` requires a set ``dof`` and cuts the additive
    ``abs_diff``/``abs_se`` sidecar's Wald interval at its OWN reference
    instead of reusing the relative lift's critical value: the two
    scales can have different variance-component geometry. Omitting it on
    that scalar path retains the shared reference; callers with independent
    scale-specific references must supply both. The sidecar reference is
    persisted separately in the result.

    ``estimand`` labels which quantity this contrast identifies: ``"ate"``
    (default) for a nonparametric average treatment effect, ``"plr_slope"``
    for DML's partially-linear model coefficient, or
    ``"overlap_subpopulation_ate"`` when ``gate.overlap="trim"`` dropped
    units and the ATE is only identified over the retained overlap band.
    The caller (one of the three contrast handlers in ``_adjust/``)
    derives this from its own method and trim outcome; every observational
    row that reaches this function carries one of these three values,
    never the randomized/encouragement estimands ``"itt"``/``"compliance"``/
    ``"late"``.
    """
    if isinstance(prior, (StudentTPrior, MixturePrior)):
        from increment.estimation.adjust import MIXTURE_PRIORS_ARE

        refuse(MIXTURE_PRIORS_ARE)
    has_joint_result = joint_reference is not None or relative_unavailable_reason is not None
    if has_joint_result and (
        value_scale != "relative"
        or (joint_reference is not None and relative_unavailable_reason is not None)
    ):
        _raise(
            "estimation.inference.joint_reference_invalid",
            reason="joint relative inference requires one reference or one unavailable reason",
        )
    if abs_dof is not None and dof is None and not has_joint_result:
        _raise("estimation.inference.abs_dof_requires_dof")
    if value_scale not in VALUE_SCALE_VALUES:
        _raise("estimation.inference.unknown_value_scale", value_scale=value_scale)
    if value_scale == "absolute" and null_abs is not None:
        _raise("estimation.inference.infer_ate_null_abs_on_absolute_metric", name=metric)
    if value_scale == "absolute" and null_lift != 0.0:
        _raise("estimation.inference.infer_ate_null_lift_on_absolute_metric", name=metric)
    if not math.isfinite(null_lift):
        refuse(INFER_ATE_NULL_LIFT_FINITE, null_lift=null_lift)
    if null_abs is not None and not math.isfinite(null_abs):
        refuse(INFER_ATE_NULL_ABS_FINITE, null_abs=null_abs)
    independent = scores if isinstance(scores, IndependentMeanReference) else None
    if independent is not None:
        from increment.estimation.armstats import _raise as component_refuse

        if (
            method != "independent_mean"
            or value_scale != "absolute"
            or prior is not None
            or dof is not None
            or n_clusters is not None
            or abs_diff is not None
            or abs_se is not None
            or point != independent.point
        ):
            component_refuse(
                "estimation.armstats.independent_mean_contract",
                reason="Welch components require their unadjusted independent mean point, absolute scale, and no prior or cluster reference.",
            )
    if has_joint_result:
        if prior is not None:
            _raise("estimation.inference.prior_excludes_joint_reference")
        confidence_set = (
            relative_confidence_set(joint_reference, alpha=alpha, alternative=alternative)
            if joint_reference is not None
            else None
        )
        displayed = (
            confidence_set.estimate()
            if confidence_set is not None
            else Estimate(value=point)
            if point is not None and math.isfinite(point)
            else None
        )
        if joint_reference is not None:
            # Persist exactly the same projection that row validation reconstructs.
            abs_diff = joint_reference.a
            abs_se = math.sqrt(joint_reference.var_a) or None
        elif abs_se is None:
            abs_se = scores.se() or None
        additive_lb, additive_ub = _joint_additive_bounds(
            abs_diff, abs_se, alpha, alternative, abs_dof
        )
        return LiftEstimate(
            metric=metric,
            group_id=group_id,
            method=method,
            method_role=method_role,
            alternative=alternative,
            null_lift=null_lift,
            preferred_direction=preferred_direction,
            lift=displayed,
            population=population,
            scale="linear",
            value_scale="relative",
            abs_diff=abs_diff,
            abs_se=abs_se,
            null_abs=null_abs,
            abs_lb=additive_lb,
            abs_ub=additive_ub,
            abs_reference_kind=("t" if abs_dof is not None else "normal")
            if abs_se is not None
            else None,
            abs_reference_df=abs_dof if abs_se is not None else None,
            n_clusters=n_clusters,
            dof=joint_reference.df if joint_reference is not None else dof,
            reference_kind=joint_reference.kind
            if joint_reference is not None
            else ("t" if dof is not None else "normal"),
            reference_df=joint_reference.df if joint_reference is not None else dof,
            estimand=estimand,
            relative_confidence_set=confidence_set,
            relative_unavailable_reason=relative_unavailable_reason,
        )
    if point is None:
        _raise("estimation.inference.degenerate_data_zero")
    se = scores.se()

    # Degenerate guard: zero variability -> no uncertainty to propagate.
    if se <= 0:
        _raise("estimation.inference.degenerate_data_zero")
    ref = _resolve_fixed_horizon(
        alpha,
        alternative,
        dof=dof,
        arm_ns=None,
        prior=prior,
        independent_means=independent,
    )
    alpha_eff = ref.alpha_eff

    if dof is not None or independent is not None or (value_scale == "absolute" and prior is None):
        # Cluster/Welch references and flat additive inference use raw statistics.
        mu_n, sigma_n = point, se
    else:
        posterior = normal_posterior(point, se, prior=prior)
        mu_n = posterior.mu
        sigma_n = posterior.sigma

    assert ref.crit is not None, "unreachable: infer_ate never passes a sequential spec"
    z = ref.crit
    lb = mu_n - z * sigma_n
    ub = mu_n + z * sigma_n

    # The additive scale uses its own reference when its variance components
    # differ from those of the relative contrast.
    abs_lb = abs_ub = None
    abs_ref = None
    if abs_diff is not None and abs_se is not None:
        _validate_additive_sidecar(abs_diff, abs_se)
        abs_ref = _resolve_fixed_horizon(
            alpha,
            alternative,
            dof=abs_dof if abs_dof is not None else dof,
            arm_ns=None,
            prior=None,
        )
        assert abs_ref.crit is not None
        abs_lb, abs_ub = wald_bounds(
            abs_diff, abs_ref.crit, abs_se, what="infer_ate absolute interval"
        )

    estimate = Estimate(
        value=mu_n,
        lb=lb,
        ub=ub,
        level=math.fsum((1.0, -alpha_eff)),
        alpha=alpha_eff,
        log_mean=point,
        log_se=se,
    )

    result = LiftEstimate(
        metric=metric,
        group_id=group_id,
        method=method,
        method_role=method_role,
        alternative=alternative,
        lift=estimate,
        population=population,
        scale="linear",
        value_scale=value_scale,
        null_lift=null_lift,
        preferred_direction=preferred_direction,
        prior_shrunk=prior is not None,
        abs_diff=abs_diff,
        abs_se=abs_se,
        null_abs=null_abs,
        abs_lb=abs_lb,
        abs_ub=abs_ub,
        abs_reference_kind=("t" if abs_ref.reference.df is not None else "normal")
        if abs_ref is not None
        else None,
        abs_reference_df=abs_ref.reference.df if abs_ref is not None else None,
        n_clusters=n_clusters,
        dof=dof,
        reference_kind=ref.reference.kind,
        reference_df=ref.reference.df,
        estimand=estimand,
        independent_mean_reference=(
            independent.model_copy(
                update={"alpha": alpha, "alternative": alternative, "interval": "central"}
            )
            if independent is not None
            else None
        ),
    )
    return result


def infer_independent_mean(
    control: ArmStats,
    treatment: ArmStats,
    *,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
) -> LiftEstimate:
    """Absolute difference of independent means from native/frame centered moments.

    The caller declares independent iid unweighted units by selecting this
    operation. Adjusted, clustered and estimated-cutoff moments are refused.
    """
    from increment.estimation.armstats import _raise as component_refuse

    if control.group_id == treatment.group_id or (control.study_id, control.metric) != (
        treatment.study_id,
        treatment.metric,
    ):
        component_refuse(
            "estimation.armstats.independent_mean_contract",
            reason="Independent mean arms must be distinct and belong to the same study and metric.",
        )
    reference = IndependentMeanReference(
        components=(
            IndependentMeanComponent.from_arm_stats(control, coefficient=-1),
            IndependentMeanComponent.from_arm_stats(treatment, coefficient=1),
        )
    )
    return infer_ate(
        control.metric,
        treatment.group_id,
        "independent_mean",
        reference.point,
        reference,
        alpha=alpha,
        alternative=alternative,
        value_scale="absolute",
        method_role="decision",
    )
