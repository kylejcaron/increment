"""Encouragement-design estimation: ITT, compliance (first stage), LATE.

ITT delegates to ``estimate_lift`` untouched: assignment-level lift IS
the ITT. Compliance reuses the same machinery on the uptake moments
when the control arm has uptake; a one-sided design's structurally-zero
control uptake reports the treated-arm uptake rate directly. LATE is
the Wald ratio of ITTs (delta-method SE) plus the Imbens-Rubin
complier-relative lift (log scale) when its precision guard passes.

``Method(variance_reduction="cuped")`` adjusts the LATE numerator only
(adjusting the first stage too would break ``LATE == cuped ITT /
compliance``). A weak first stage suppresses LATE; a one-sided design
hard-errors on any control uptake; as-treated comparisons are never
computed.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from narwhals.typing import IntoDataFrame
from scipy.stats import norm as _norm

from increment.estimation._readout_refusals import READOUT_REFUSALS as _READOUT_REFUSALS
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.semantics.design import READOUT_ENCOURAGEMENT_RETENTION
from increment.sequential_state import SequentialSnapshot, sequential_refuse

if TYPE_CHECKING:
    from increment._readout_request import ReadoutRequest
    from increment.decision import DecisionComputation
    from increment.estimation.results import LiftEstimate
from increment._literals import ALTERNATIVE_VALUES
from increment._moment_plan import UPTAKE_MASK
from increment._source_types import ComplianceArm, ComplianceSummary
from increment._window import resolve_window_days
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
    refuse,
)
from increment.estimation._adjust.overlap import IdentificationError
from increment.estimation._tails import student_t_isf, two_sided_critical_value
from increment.estimation.adjust import MIXTURE_PRIORS_ARE
from increment.estimation.armstats import (
    ArmStats,
    CenteredMoments,
    clamp_negative_variance,
    variance_slack,
    welch_satterthwaite_df,
)
from increment.estimation.binomial_rr import UNKNOWN_ALTERNATIVE
from increment.estimation.cuped import fit_cuped
from increment.estimation.engine import (
    Method,
    _df_to_arms,
    _validate_methods,
    _warn_if_open_ended_sequential,
    _winsorization_result_fields,
    check_total_clusters,
    estimate_lift,
    ratio_abs_diff_se,
    ratio_pair_cov,
)
from increment.estimation.inference import (
    ONE_SIDED_ALPHA_DOUBLES,
    LiftGuardError,
    Normal,
    Prior,
    normal_posterior,
)
from increment.estimation.priors import MixturePrior, StudentTPrior
from increment.estimation.results import Estimate, LiftEstimate, _reference_fields
from increment.estimation.sequential import (
    SEQUENTIAL_POLICIES,
    UPTAKE_COMPLETION_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
    SequentialSupportRequest,
    sequential_support_refusal,
)
from increment.estimation.variance import cluster_outcome_moments, cluster_uptake_moments
from increment.semantics.design import RETENTION_UNDER_ENCOURAGEMENT, Encouragement
from increment.semantics.models import MeanMetric, Metric

ESTIMANDS = (
    "itt",
    "compliance",
    "late",
)  # public: the shared default estimands, reused by readouts.py

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "identification.encouragement.exclusion_required": RefusalSpec(
            "identification.encouragement.exclusion_required",
            IdentificationError,
            template=(
                "LATE requires an acknowledged ExclusionRestriction; request "
                "estimands=('itt', 'compliance') or declare exclusion_restriction "
                "on the encouragement design (requested estimands={estimands!r})"
            ),
            keys=frozenset({"available_estimands", "required_assumption"}),
        ),
        "estimation.encouragement.cluster.ratio": RefusalSpec(
            "estimation.encouragement.cluster.ratio",
            CapabilityError,
            template="ratio metric(s) {names} are not supported with a declared cluster ('{cluster}') under an encouragement design -- the first stage reads cluster SIZES out of the den family a clustered ratio metric needs for its own denominator, so the two cannot share one row. Analyze the ratio metric without the encouragement design, or without the cluster.",
        ),
        "estimation.encouragement.late.ratio": RefusalSpec(
            "estimation.encouragement.late.ratio",
            UnsupportedRequestError,
            template="LATE is not supported for ratio metric '{metric}' -- the Wald numerator is the difference of arm mean_y, which for a ratio metric is the NUMERATOR mean, so the result would be the numerator's LATE silently mislabelled as the ratio's. Request estimands=('itt',) (ratio-aware), or model the numerator and denominator as separate mean metrics",
        ),
        "readout.encouragement.value_scale": RefusalSpec(
            "readout.encouragement.value_scale",
            UnsupportedRequestError,
            template="value_scale= is an observational-only reporting selector -- an encouragement design's LATE rows already report the additive estimand in the outcome's own units (value_scale='absolute'); select it with estimands= instead.",
        ),
        "readout.encouragement.cluster_ratio": RefusalSpec(
            "readout.encouragement.cluster_ratio",
            CapabilityError,
            template="ratio metric(s) {names} are not supported with a declared cluster ('{cluster}') under an encouragement design",
        ),
        "readout.encouragement.asof_completion": "{view}: AlwaysValid under an encouragement design requires completed_windows_only=True so repeated looks use finalized observations",
        "readout.encouragement.asof_unbounded": "as-of completed encouragement inference requires bounded outcome and uptake windows; unbounded metrics={metrics!r}, uptake_window_days={uptake_window_days!r}",
        "readout.inference.sequential_view": RefusalSpec(
            "readout.inference.sequential_view",
            UnsupportedRequestError,
            template="sequential inference is not supported on the segmented {view} view",
        ),
        "estimation.encouragement.control.missing": "control_group '{control_group}' not found in summary",
        "estimation.encouragement.negative_variance": "{what} computed as {value:.6g}, negative beyond floating-point rounding (~{slack:.3g}) -- this scale of deficit means the upstream moments feeding the Wald ratio's delta-method variance are infeasible or inconsistent (mis-aggregated or corrupted), not merely cancelled.",
        "estimation.encouragement.cuped_adjusted_late": "CUPED-adjusted LATE needs the covariate x uptake cross-moment cxd, which is None for arm {arm_group}/{arm_metric} although its covariate moments are present -- the moments were aggregated before this estimator existed. Re-aggregate the summary (re-run the query, or rebuild from_unit_summary with both covariate= and uptake=), or request the late estimand with variance_reduction='none'.",
        "estimation.encouragement.needs_uptake_cyd": "needs uptake: cyd is None (uptake fact not materialised)",
        "estimation.encouragement.unknown_estimand_supported": "unknown estimand(s) {unknown}; supported: {estimands}. As-treated / per-protocol comparisons are deliberately not offered.",
        "estimation.encouragement.prior_inference_are": "prior and inference are mutually exclusive: a conjugate posterior has no always-valid guarantee under repeated looks",
        "estimation.encouragement.one_sided_encouragement": "one_sided encouragement declared but control arm has uptake (uptake_total={control_uptake_total:g}, metric '{metric}') -- control must be structurally unable to take up; this is an instrumentation bug, refusing to estimate",
        "estimation.encouragement.cluster.arm_needs_two": "metric '{metric}': cluster contrast ('{cluster}') needs at least 2 clusters in EACH arm to estimate a between-cluster variance for that arm, got {k_t} treatment cluster(s) and {k_c} control cluster(s) -- a single cluster carries no between-cluster variance contribution to estimate its own arm's component from.",
        "estimation.encouragement.alpha_eff_too": "alpha_eff is too small: alpha_eff / 2 underflows floating-point precision",
        "estimation.encouragement.dof_positive_reference": "dof must be positive for a t reference, got {dof}",
        "estimation.encouragement.prior_cluster_robust": "prior and a cluster-robust t reference are mutually exclusive: a Normal prior has no coherent conjugate update against a t sampling distribution",
        "estimation.encouragement.sequential_inference_cluster": "sequential inference and a cluster-robust t reference are mutually exclusive -- the sequential boundaries assume independent units at unit grain",
    },
)

_REFUSALS["readout.encouragement.retention"] = READOUT_ENCOURAGEMENT_RETENTION

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA

ARM_NEEDS_TWO = _REFUSALS["estimation.encouragement.cluster.arm_needs_two"]
ALPHA_EFF_TOO = _REFUSALS["estimation.encouragement.alpha_eff_too"]
DOF_POSITIVE_REFERENCE = _REFUSALS["estimation.encouragement.dof_positive_reference"]
SEQUENTIAL_INFERENCE_CLUSTER = _REFUSALS["estimation.encouragement.sequential_inference_cluster"]
_REFUSALS[RETENTION_UNDER_ENCOURAGEMENT.code] = RETENTION_UNDER_ENCOURAGEMENT
_REFUSALS.update(
    {
        code: _READOUT_REFUSALS[code]
        for code in ("readout.estimands.unknown", "readout.encouragement.margin")
    }
)
_refuse = raiser(_REFUSALS)


def _require_exclusion_for_late(design: Encouragement, estimands: Sequence[str]) -> None:
    """Refuse LATE before touching source data when exclusion is undeclared."""
    if "late" in estimands and design.exclusion_restriction is None:
        _refuse(
            "identification.encouragement.exclusion_required",
            estimands=tuple(estimands),
            available_estimands=("itt", "compliance"),
            required_assumption="exclusion_restriction",
        )


def validate_readout_encouragement_identification(request: ReadoutRequest) -> None:
    """Validate encouragement assumptions before compatibility or source checks."""
    if request.view != "daily" and isinstance(request.design, Encouragement):
        estimands = ESTIMANDS if request.estimands is None else request.estimands
        _require_exclusion_for_late(request.design, estimands)


def _clamp_variance(value: float, *, magnitude: float, n: int, what: str) -> float:
    """Clamp a mathematically non-negative delta-method LATE variance
    within floating-point noise of zero; refuse (coded error) beyond it.

    Thin wrapper around ``armstats.clamp_negative_variance`` (the shared
    clamp-vs-refuse tolerance every such variance in the package should
    use) that renders this module's own named refusal on overflow,
    since the message/code differ per call site.
    """
    result = clamp_negative_variance(value, magnitude=magnitude, n=n)
    if result is None:
        slack = variance_slack(magnitude, max(n, 1))
        _refuse("estimation.encouragement.negative_variance", what=what, value=value, slack=slack)
    return result


_N0_MIN_Z = 3.0  # precision floor for the complier control mean (relative form)
_N1_MIN_Z = 3.0  # same floor for the complier treated mean (equally fragile)
# Near this z-margin above the first-stage gate, estimates lean toward the
# as-treated value under confounding; LATE rows carry a caveat naming that.
_NEAR_GATE_Z_MARGIN = 1.0


def _first_stage(t: ArmStats, c: ArmStats) -> tuple[float, float]:
    """(compliance lift B, Var(B))."""
    b = t.mean_d() - c.mean_d()
    var_b = t.var_d() / t.n + c.var_d() / c.n
    return b, var_b


def _first_stage_cluster(t: ArmStats, c: ArmStats) -> tuple[float, float, float]:
    """(compliance lift B, Var(B), dof) from CLUSTER totals: the declared-
    cluster analogue of :func:`_first_stage`, a ratio delta method over
    each arm's cluster uptake sums / cluster sizes (see
    ``variance.cluster_uptake_moments``) instead of unit-grain
    ``mean_d``/``var_d``, which assume binary *d* at unit grain.

    ``dof`` uses Welch-Satterthwaite over each arm's own Bessel-corrected
    between-cluster variance component. The additive variance therefore uses
    ``se_t**2 + se_c**2`` with arm-specific degrees of freedom, not a pooled
    ``t.n + c.n - 2`` reference.
    """
    uptake_t, size_t, var_uptake_t, var_size_t, cov_us_t = cluster_uptake_moments(t)
    uptake_c, size_c, var_uptake_c, var_size_c, cov_us_c = cluster_uptake_moments(c)
    rate_t, se_t = ratio_abs_diff_se(uptake_t, size_t, var_uptake_t, var_size_t, cov_us_t, t.n)
    rate_c, se_c = ratio_abs_diff_se(uptake_c, size_c, var_uptake_c, var_size_c, cov_us_c, c.n)
    b = rate_t - rate_c
    var_b = se_t**2 + se_c**2
    dof = welch_satterthwaite_df(se_t * se_t, float(t.n - 1), se_c * se_c, float(c.n - 1))
    return b, var_b, dof


def _arm_stats_from_compliance(
    arm: ComplianceArm, *, study_id: str, cluster: str | None
) -> ArmStats:
    """Represent the real uptake response as complete centered moments.

    At cluster grain the uptake total is both the outcome and what the x
    slot carries (``x_role="uptake_total"``) and cluster size is the
    denominator; at unit grain the binary uptake indicator is both the
    outcome and the mask, with reference zero and exact residual sums.
    No outcome data is invented.
    """
    if cluster is not None:
        assert (
            arm.n_clusters is not None
            and arm.ref_uptake is not None
            and arm.cluster_uptake1 is not None
            and arm.cluster_uptake2 is not None
            and arm.ref_size is not None
            and arm.cluster_size1 is not None
            and arm.cluster_size2 is not None
            and arm.cluster_cross is not None
        ), f"clustered ComplianceArm {arm.group_id!r} missing its bivariate family"
        bivariate = CenteredMoments(
            n=arm.n_clusters,
            variables=("uptake", "size"),
            ref={"uptake": arm.ref_uptake, "size": arm.ref_size},
            c1={("uptake", None): arm.cluster_uptake1, ("size", None): arm.cluster_size1},
            c2={
                ("uptake", "uptake", None): arm.cluster_uptake2,
                ("size", "size", None): arm.cluster_size2,
                ("uptake", "size", None): arm.cluster_cross,
            },
            count={},
        )
        outcome = bivariate.renamed({"size": "den"}).with_alias("y", of="uptake")
        return ArmStats.from_moments(
            outcome, study_id=study_id, metric="uptake", group_id=arm.group_id
        )
    indicator = CenteredMoments(
        n=arm.n_units,
        variables=("y",),
        ref={"y": 0.0},
        c1={("y", None): arm.uptake_total, ("y", UPTAKE_MASK): arm.uptake_total},
        c2={("y", "y", None): arm.uptake_total, ("y", "y", UPTAKE_MASK): arm.uptake_total},
        count={UPTAKE_MASK: arm.uptake_total},
    )
    return ArmStats.from_moments(
        indicator, study_id=study_id, metric="uptake", group_id=arm.group_id
    )


def estimate_compliance(
    compliance: ComplianceSummary,
    design: Encouragement,
    *,
    alpha: float = 0.05,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    prior: Normal | None = None,
    compliance_requested: bool = True,
) -> DecisionComputation[LiftEstimate]:
    """Design-level compliance rows built directly from a ``MomentSource``'s
    :class:`~increment._source_types.ComplianceSummary` -- independent of
    any outcome metric's declaration, order, count, or missingness. Reuses
    the exact same first-stage/compliance-row machinery
    (:func:`_first_stage_context`/:func:`_estimate_compliance_row`) as
    ``estimate_encouragement``'s own ``compliance`` estimand, fed by the
    design's declared uptake cohort instead of any one metric's
    outcome-filtered moments -- this is what fixes metric-order/count/
    missingness sensitivity in the design-level compliance row while
    leaving each metric's own cohort-matched LATE first stage untouched.

    LATE suppression is determined from each outcome's cohort by
    ``estimate_encouragement``; this design-wide row never diagnoses it.
    """
    from increment._source_types import validate_compliance_design_match
    from increment.estimation.decision_types import DecisionComputation
    from increment.estimation.engine import _lift_decision_bundle

    validate_compliance_design_match(compliance, design)
    cluster = compliance.cluster
    if cluster is not None and inference is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported("arm.inference.cluster"), cluster=cluster)
    if cluster is not None and prior is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported("arm.adjustment.cluster_prior"), cluster=cluster)
    control_id = str(design.control_group)
    control_arm = compliance.arm(control_id)
    if control_arm is None:
        _refuse("estimation.encouragement.control.missing", control_group=control_id)
    c = _arm_stats_from_compliance(control_arm, study_id=compliance.study_id, cluster=cluster)
    results: list[LiftEstimate] = []
    for arm in compliance.arms:
        if arm.group_id == control_id:
            continue
        t = _arm_stats_from_compliance(arm, study_id=compliance.study_id, cluster=cluster)
        context = _first_stage_context(t, c, design, cluster=cluster, late_requested=False)
        if not compliance_requested:
            continue
        results.append(
            _estimate_compliance_row(
                t,
                c,
                design,
                context,
                inference=inference,
                prior=prior,
                alpha=alpha,
                cluster=cluster,
            )
        )
    bundle = _lift_decision_bundle(results, inference=inference, allow_linear=True)
    return DecisionComputation(
        results=tuple(results), evidence=bundle.evidence, failures=bundle.failures
    )


def _late_additive(t: ArmStats, c: ArmStats) -> tuple[float, float]:
    """(tau_hat, se): Wald ratio with full delta-method variance."""
    a = t.mean_y() - c.mean_y()
    b, var_b = _first_stage(t, c)
    tau = a / b
    var_a = t.var_y() / t.n + c.var_y() / c.n
    cov_ab = t.cov_yd() / t.n + c.cov_yd() / c.n
    cross, tail = 2 * tau * cov_ab, tau**2 * var_b
    numerator = _clamp_variance(
        var_a - cross + tail,
        magnitude=abs(var_a) + abs(cross) + abs(tail),
        n=t.n + c.n,
        what="LATE (additive) delta-method variance",
    )
    return tau, math.sqrt(numerator / b**2)


def _late_additive_cluster(t: ArmStats, c: ArmStats) -> tuple[float, float, float]:
    """(tau_hat, se, dof): cluster-robust Wald ratio, the declared-cluster
    analogue of :func:`_late_additive`.

    ``mean_y``/``mean_d`` become ratios of cluster sums (``sum(g_j) /
    sum(m_j)``, ``sum(d_j) / sum(m_j)``), the same linearized delta
    method :class:`~increment.estimation.variance.ClusterVarianceModel`
    uses for the ITT row; ``cov_yd`` becomes the matching cluster-grain
    covariance of those two ratio numerators (``ratio_pair_cov``) rather
    than the unit-grain ``cov_yd()``, which assumes binary *d*.

    Uses :func:`~increment.estimation.variance.cluster_outcome_moments`,
    not :func:`~increment.estimation.engine._ratio_abs_diff_se`: the
    latter's numerator-positivity guard exists for the log-relative lift
    and would wrongly refuse a signed or zero-mean outcome here, where
    the unclustered ``_late_additive`` has no such restriction.

    ``dof`` is a Welch-Satterthwaite reference over the same arm-specific terms
    that the pooled numerator sums. Each component decomposes into treatment
    and control contributions, each paired with that arm's ``K_a - 1`` degrees
    of freedom rather than the pooled ``t.n + c.n - 2``. Negative component
    variances are floored only for the reference; the reported SE keeps the
    exact unclamped arithmetic.
    """
    mean_t, size_yt, var_mean_t, var_size_yt, cov_ms_t = cluster_outcome_moments(t)
    mean_c, size_yc, var_mean_c, var_size_yc, cov_ms_c = cluster_outcome_moments(c)
    a_t, se_a_t = ratio_abs_diff_se(mean_t, size_yt, var_mean_t, var_size_yt, cov_ms_t, t.n)
    a_c, se_a_c = ratio_abs_diff_se(mean_c, size_yc, var_mean_c, var_size_yc, cov_ms_c, c.n)
    uptake_t, size_t, var_uptake_t, var_size_t, cov_us_t = cluster_uptake_moments(t)
    uptake_c, size_c, var_uptake_c, var_size_c, cov_us_c = cluster_uptake_moments(c)
    b_t, se_b_t = ratio_abs_diff_se(uptake_t, size_t, var_uptake_t, var_size_t, cov_us_t, t.n)
    b_c, se_b_c = ratio_abs_diff_se(uptake_c, size_c, var_uptake_c, var_size_c, cov_us_c, c.n)

    a = a_t - a_c
    b = b_t - b_c
    var_a = se_a_t**2 + se_a_c**2
    var_b = se_b_t**2 + se_b_c**2

    cov_ab_t = ratio_pair_cov(
        a_t, b_t, size_t, t.cov_yx(), t.cov_yden(), t.cov_xden(), var_size_t, t.n
    )
    cov_ab_c = ratio_pair_cov(
        a_c, b_c, size_c, c.cov_yx(), c.cov_yden(), c.cov_xden(), var_size_c, c.n
    )
    cov_ab = cov_ab_t + cov_ab_c

    tau = a / b
    cross, tail = 2 * tau * cov_ab, tau**2 * var_b
    numerator = _clamp_variance(
        var_a - cross + tail,
        magnitude=abs(var_a) + abs(cross) + abs(tail),
        n=t.n + c.n,
        what="LATE (cluster-robust additive) delta-method variance",
    )
    component_t = max(se_a_t**2 - 2 * tau * cov_ab_t + tau**2 * se_b_t**2, 0.0)
    component_c = max(se_a_c**2 - 2 * tau * cov_ab_c + tau**2 * se_b_c**2, 0.0)
    dof = welch_satterthwaite_df(component_t, float(t.n - 1), component_c, float(c.n - 1))
    return tau, math.sqrt(numerator / b**2), dof


def _late_additive_cuped(t: ArmStats, c: ArmStats) -> tuple[float, float, float]:
    """(tau_hat, se, se_unadjusted): Wald ratio on a CUPED-adjusted numerator.

    Numerator-only: the first stage stays raw, so ``min_first_stage_z``
    reads exactly the z it would without a covariate, and the compliance
    row is byte-identical to the unadjusted one. The adjusted per-arm
    means come from one :func:`fit_cuped` state (adjusted-per-arm, then
    differenced), so this numerator is bit-for-bit the cuped ITT row's
    ``abs_diff`` and the reported LATE equals its cuped ITT over its
    compliance exactly.

    The delta method carries the full outcome/uptake/covariate Jacobian:
    ``Var(A)`` is the adjusted contrast's variance (the pooled anchor cancels
    in the difference, so the per-arm ``Var(Y - theta*X)`` terms are exact
    here), ``Var(B)`` the raw first stage's, and ``Cov(A, B)`` the per-arm
    ``Cov(Y - theta*X, D) = Cov(Y, D) - theta*Cov(X, D)`` -- dropping the
    covariate-uptake term measured SE/SD 0.944, coverage 0.938 versus
    1.001/0.951 with it.

    ``se_unadjusted`` is the same delta-method SE with no adjustment,
    returned so the caller can annotate the backfire regime:
    numerator-only CUPED carries excess influence-function variance
    ``(tau * theta_D)^2 Var(X)`` where the variance-optimal coefficient
    would carry ``(theta_Y - tau * theta_D)^2 Var(X)``, so it inflates
    variance whenever the covariate drives uptake strongly, predicts the
    outcome weakly net of uptake, and tau is large.
    """
    # fit_cuped refuses first when x-moments are absent, so the cxd check
    # below only fires on a substrate with a covariate that predates it.
    fit = fit_cuped([c, t])
    for arm in (c, t):
        if arm.cxd is None:
            _refuse(
                "estimation.encouragement.cuped_adjusted_late",
                arm_group=arm.group_id,
                arm_metric=arm.metric,
            )
    c_adj, t_adj = fit.adjust()
    a = t_adj.mean - c_adj.mean
    b, var_b = _first_stage(t, c)
    tau = a / b
    var_a = t_adj.var / t.n + c_adj.var / c.n
    cov_ab = sum((arm.cov_yd() - fit.theta * arm.cov_xd()) / arm.n for arm in (t, c))
    cross, tail = 2 * tau * cov_ab, tau**2 * var_b
    numerator = _clamp_variance(
        var_a - cross + tail,
        magnitude=abs(var_a) + abs(cross) + abs(tail),
        n=t.n + c.n,
        what="LATE (CUPED-adjusted additive) delta-method variance",
    )
    _, se_unadjusted = _late_additive(t, c)
    return tau, math.sqrt(numerator / b**2), se_unadjusted


def _late_relative(t: ArmStats, c: ArmStats) -> tuple[float, float] | str:
    """(theta = log complier ratio, se) or a fallback-reason string.

    theta = log(N1) - log(N0); the first stage cancels in the ratio.

    The log delta method's validity is a precision property of log(N1)
    and log(N0), not of the ratio itself: an extreme but precisely-
    measured theta is fine, a modest theta built on a noisy complier
    mean is not. The two z-floors below enforce that on both flanks, and
    also bound the combined log-scale SE: binary uptake makes the
    complier (yd) and never-taker (y - yd) moment components
    nonnegatively correlated across the N1/N0 split, so the cross term
    in var(theta) only shrinks it - with both floors at z >= 3,
    se_theta <= sqrt(1/z1^2 + 1/z0^2) <= sqrt(2)/3, inside the same
    < 0.5 ceiling ``infer_lift``'s delta-method guard enforces.
    """
    if t.cyd is None or c.cyd is None:
        _refuse("estimation.encouragement.needs_uptake_cyd")
    n1 = t.mean_yd() - c.mean_yd()
    # mean_y_untaken() is the mean of y*(1-d) by analytic expansion, the
    # per-arm never-taker term, never a difference of two large means.
    n0 = c.mean_y_untaken() - t.mean_y_untaken()
    # `_clamp_variance` clamps float-noise deficits to zero (the precision gate then reads an
    # exact mean) and refuses material deficits as corrupt moments. It runs per arm against
    # constituent magnitudes: a cancelled residual's own scale would refuse the rounding being
    # absorbed, and summing first would let one negative arm hide behind a positive one.
    var_n0 = math.fsum(
        _clamp_variance(
            (a.var_y() + a.var_yd() - 2 * a.cov_y_yd()) / a.n,
            magnitude=(abs(a.var_y()) + abs(a.var_yd()) + 2 * abs(a.cov_y_yd())) / a.n,
            n=a.n,
            what=f"complier control mean variance for arm {a.group_id!r}",
        )
        for a in (t, c)
    )
    if n0 <= 0:
        return "complier control mean estimated <= 0"
    if var_n0 > 0.0 and n0 / math.sqrt(var_n0) < _N0_MIN_Z:
        return "complier control mean too imprecise"
    if n1 <= 0:
        return "complier treated mean estimated <= 0"
    # var_yd is a single non-negative moment, so there is nothing to cancel;
    # its own magnitude is the right scale.
    var_n1 = math.fsum(
        _clamp_variance(
            a.var_yd() / a.n,
            magnitude=abs(a.var_yd()) / a.n,
            n=a.n,
            what=f"complier treated mean variance for arm {a.group_id!r}",
        )
        for a in (t, c)
    )
    if var_n1 > 0.0 and n1 / math.sqrt(var_n1) < _N1_MIN_Z:
        return "complier treated mean too imprecise"
    theta = math.log(n1) - math.log(n0)
    # Delta-method gradient of theta=log(N1)-log(N0): treatment (1/N0,
    # 1/N1-1/N0), control (-1/N0, -1/N1+1/N0), for (dtheta/dy, dtheta/dyd).
    var_theta = 0.0
    for arm, g_y, g_yd in (
        (t, 1.0 / n0, 1.0 / n1 - 1.0 / n0),
        (c, -1.0 / n0, -1.0 / n1 + 1.0 / n0),
    ):
        var_theta += (
            g_y**2 * arm.var_y() + g_yd**2 * arm.var_yd() + 2 * g_y * g_yd * arm.cov_y_yd()
        ) / arm.n
    return theta, math.sqrt(var_theta)


def _nn_estimate(
    point: float,
    se: float,
    prior: Normal | None,
    alpha: float,
    alternative: str = "two-sided",
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    n_comparison: int | None = None,
    dof: float | None = None,
) -> Estimate:
    """Conjugate Normal-Normal update -> closed-form interval (shared tail).

    ``alternative`` follows the same alpha-doubling identity ``infer_lift``/
    ``infer_ate`` use: a one-sided test at level ``alpha`` displays the
    two-sided interval at ``alpha_eff = 2 * alpha``, with ``level`` set to
    that interval's honest coverage. Stamping ``LiftEstimate.alternative``
    is the caller's job - this helper only builds the ``Estimate``.

    With ``inference`` set, the interval is the sequential boundary applied
    to the raw delta-method SE around the raw point; the conjugate
    posterior is bypassed entirely (a prior-shifted center would void the
    frequentist time-uniform guarantee). ``n_comparison`` is the combined
    two-arm count the spec's ``radius`` contract requires.

    ``dof`` mirrors ``infer_lift``'s cluster-robust reference: when set,
    the interval uses a t_{dof} critical value in place of the Normal z,
    and the conjugate tail is bypassed entirely (``mu = point, sigma =
    se``), since a Normal prior has no coherent update against a t
    sampling distribution. Mutually exclusive with ``prior`` and
    ``inference``; callers enforce that exclusion up front, this only
    re-checks it.
    """
    if alternative not in ALTERNATIVE_VALUES:
        refuse(UNKNOWN_ALTERNATIVE, alternative=alternative)
    if not 0.0 < alpha < 1.0:
        _refuse("estimation.diagnostics.alpha", alpha=alpha)
    alpha_eff = alpha if alternative == "two-sided" else 2.0 * alpha
    if alpha_eff >= 1.0:
        refuse(ONE_SIDED_ALPHA_DOUBLES, alpha=alpha, alpha_eff=alpha_eff)
    if alpha_eff / 2.0 == 0.0:
        refuse(ALPHA_EFF_TOO)
    if dof is not None:
        if dof <= 0:
            refuse(DOF_POSITIVE_REFERENCE, dof=dof)
        if prior is not None:
            _refuse("estimation.encouragement.prior_cluster_robust")
        if inference is not None:
            refuse(SEQUENTIAL_INFERENCE_CLUSTER)
        crit = two_sided_critical_value(
            student_t_isf, alpha_eff, dof, what="encouragement cluster-robust t reference"
        )
        half = crit * se
        return Estimate(
            value=point,
            lb=point - half,
            ub=point + half,
            level=math.fsum((1.0, -alpha_eff)),
            alpha=alpha_eff,
            log_mean=point,
            log_se=se,
        )
    if inference is not None:
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "additive SE-only sequential inference has no matching raw likelihood",
        )
    posterior = normal_posterior(point, se, prior=prior)
    z = _norm.isf(alpha_eff / 2.0)
    mu, sig = posterior.mu, posterior.sigma
    return Estimate(
        value=mu,
        lb=mu - z * sig,
        ub=mu + z * sig,
        level=math.fsum((1.0, -alpha_eff)),
        alpha=alpha_eff,
        log_mean=point,
        log_se=se,
    )


def _nn_estimate_or_point(
    point: float,
    se: float,
    prior: Normal | None,
    alpha: float,
    alternative: str = "two-sided",
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    n_comparison: int | None = None,
    dof: float | None = None,
) -> Estimate:
    """``_nn_estimate``, degenerating to a point-only ``Estimate`` at se == 0.

    A legitimate zero-SE point (e.g. an exactly-zero additive LATE with
    an exactly-estimable first stage) has no displayable interval --
    ``normal_posterior``'s ``Normal`` requires strictly positive sigma.
    Mirrors the existing zero-SE compliance rows
    (``_zero_control_compliance_row``, ``_relative_compliance_row``).
    """
    if se <= 0.0:
        return Estimate(value=point)
    return _nn_estimate(
        point,
        se,
        prior,
        alpha,
        alternative,
        inference=inference,
        n_comparison=n_comparison,
        dof=dof,
    )


def _compliance_metric(uptake_window_days: int | None = None) -> MeanMetric:
    """The synthetic uptake metric for the two-sided compliance delegation.

    Carries the design's uptake window so a windowed uptake is not
    misflagged as open-ended by ``estimate_lift``'s sequential accrual
    warning. Its ``name`` is the user-facing compliance metric identity
    (``LiftEstimate.metric`` on every compliance row) -- see
    ``docs/guides/metric-types.md`` for the naming convention.
    """
    # Retain compliance's prior/sequential support and absolute fallback;
    # exact-binomial dispatch is for user-declared binary estimands.
    return MeanMetric(
        name="uptake",
        entity="__uptake__",
        fact="__uptake__",
        aggregation="avg_event",
        window_days=uptake_window_days,
    )


def validate_readout_encouragement(request: ReadoutRequest) -> None:
    """Validate encouragement-specific static compatibility at the seam."""

    design = request.design
    if getattr(design, "mechanism", None) != "encouragement":
        return
    validate_readout_encouragement_identification(request)
    metrics = tuple(request.metrics)
    configs = tuple(request.configs)
    estimands = request.estimands
    plan = request.plan
    raw_inference = plan.inference
    inference = raw_inference if isinstance(raw_inference, SEQUENTIAL_POLICIES) else None
    if inference is not None and estimands is not None and set(estimands) == {"compliance"}:
        metrics, configs = (), ()
    cluster = request.cluster
    value_scale = request.value_scale

    if value_scale:
        _refuse("readout.encouragement.value_scale")
    if estimands is not None:
        unknown = set(estimands) - set(ESTIMANDS)
        if unknown:
            _refuse(
                "readout.estimands.unknown",
                unknown=unknown,
                supported=ESTIMANDS,
            )
    if (
        any(getattr(config, "prior", None) is not None for config in configs)
        and inference is not None
    ):
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(
            Unsupported("arm.adjustment.sequential_prior"),
            metrics=tuple(c.metric.name for c in configs if getattr(c, "prior", None) is not None),
        )
    if cluster is not None:
        ratios = [metric.name for metric in metrics if getattr(metric, "type", None) == "ratio"]
        if ratios:
            _refuse(
                "readout.encouragement.cluster_ratio",
                names=ratios,
                cluster=cluster,
            )
    retention = [metric.name for metric in metrics if getattr(metric, "type", None) == "retention"]
    if retention:
        _refuse("readout.encouragement.retention", names=retention)
    # A margin applies to the whole-window ITT row; a run request without
    # itt has nothing to apply it to. Other views own their margin rules.
    if request.view == "run" and estimands is not None and "itt" not in estimands:
        declared = [
            metric.name
            for metric in metrics
            if getattr(metric, "margin", None) is not None
            or getattr(metric, "margin_abs", None) is not None
        ]
        shifted = [
            metric.name
            for metric in metrics
            if (
                getattr(plan.procedures[metric.name], "null_lift", 0.0) != 0.0
                or getattr(plan.procedures[metric.name], "null_abs", None) is not None
            )
        ]
        combined = list(dict.fromkeys([*shifted, *declared]))
        if combined:
            _refuse("readout.encouragement.margin", names=combined)
    if request.view == "asof" and isinstance(inference, UPTAKE_COMPLETION_POLICIES):
        if not request.completion_policy:
            _refuse("readout.encouragement.asof_completion", view="asof_lift")
        unbounded = [metric.name for metric in metrics if resolve_window_days(metric) is None]
        uptake_window_days = getattr(design.uptake, "window_days", None)
        if unbounded or uptake_window_days is None:
            _refuse(
                "readout.encouragement.asof_unbounded",
                metrics=unbounded,
                uptake_window_days=uptake_window_days,
            )
    if request.view == "breakout" and inference is not None:
        _refuse("readout.inference.sequential_view", view="breakout")


def _encouragement_decision_bundle(
    results: Sequence[LiftEstimate],
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    late_failures: Sequence[tuple[str, str, float]],
    itt_bundle: DecisionComputation[LiftEstimate] | None,
) -> DecisionComputation[LiftEstimate]:
    """Attach one decision value to each encouragement hypothesis.

    Additive LATE is the canonical LATE hypothesis row.  The relative LATE
    presentation is the same null and therefore deliberately emits no second
    evidence value.
    """
    from increment.estimation.decision_types import (
        ArmHypothesisKey,
        DecisionComputation,
        DecisionFailure,
    )
    from increment.estimation.engine import _lift_decision_bundle

    evidence_results = [
        result
        for result in results
        if not (result.estimand == "late" and result.value_scale == "relative")
    ]
    non_itt = [result for result in evidence_results if result.estimand != "itt"]
    bundle = _lift_decision_bundle(
        non_itt,
        inference=inference,
        allow_linear=True,
    )
    evidence = dict(itt_bundle.evidence) if itt_bundle is not None else {}
    failures = dict(itt_bundle.failures) if itt_bundle is not None else {}
    evidence.update(bundle.evidence)
    failures.update(bundle.failures)
    for metric, group_id, z_fs in late_failures:
        hypothesis = ArmHypothesisKey(metric, group_id, "late")
        failures[hypothesis] = DecisionFailure(
            hypothesis,
            "estimation.encouragement.late.weak_first_stage",
            {"metric": metric, "group_id": group_id, "first_stage_z": z_fs},
        )
    return DecisionComputation(results=tuple(results), evidence=evidence, failures=failures)


@dataclass(frozen=True, slots=True)
class _PreparedEncouragementEstimation:
    """Validated request and arm inventory shared by every contrast."""

    metric_types: Mapping[str, str]
    estimands: tuple[str, ...]
    late_methods: tuple[Method, ...]
    resolved_method_roles: Mapping[str, Literal["decision", "sensitivity"]]
    control: Mapping[str, ArmStats]
    treatments: tuple[ArmStats, ...]
    itt_bundle: DecisionComputation[LiftEstimate] | None
    results: tuple[LiftEstimate, ...]


@dataclass(frozen=True, slots=True)
class _EncouragementFirstStageContext:
    """One contrast's first-stage statistics and emission policy."""

    b: float
    var_b: float
    z_fs: float
    weak: bool
    n_clusters: int | None
    dof: float | None
    suppression: str | None
    late_caveat: str
    pop: str
    iv_assumes: str


@dataclass(frozen=True, slots=True)
class _EncouragementMethodStrategy:
    """One method's fixed LATE operation."""

    method: Method
    method_role: Literal["decision", "sensitivity"]
    use_cuped: bool


def _empty_encouragement_decision(
    summary: IntoDataFrame | Iterable[Mapping[str, Any]],
    design: Encouragement,
    estimands: Sequence[str],
) -> DecisionComputation[LiftEstimate]:
    """Build the refusal bundle for an explicitly empty method list."""
    from increment.estimation.decision_types import (
        ArmHypothesisKey,
        DecisionComputation,
        DecisionFailure,
    )

    arms = _df_to_arms(summary)
    controls = {arm.metric for arm in arms if arm.group_id == design.control_group}
    failures: dict[Any, DecisionFailure] = {}
    for arm in arms:
        if arm.group_id == design.control_group or arm.metric not in controls:
            continue
        keys = []
        if "itt" in estimands:
            keys.append(ArmHypothesisKey(arm.metric, arm.group_id, "itt"))
        if "late" in estimands:
            keys.append(ArmHypothesisKey(arm.metric, arm.group_id, "late"))
        if "compliance" in estimands:
            keys.append(ArmHypothesisKey("uptake", arm.group_id, "compliance"))
        for hypothesis in keys:
            failures[hypothesis] = DecisionFailure(
                hypothesis,
                "estimation.encouragement.no_decision_method",
                {"metric": hypothesis.metric, "group_id": hypothesis.group_id},
            )
    return DecisionComputation(results=(), evidence={}, failures=failures)


def _resolve_encouragement_methods(
    methods: list[Method],
    estimands: Sequence[str],
    prior: Prior | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    cluster: str | None,
    metrics: Sequence[Metric],
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
) -> dict[str, Literal["decision", "sensitivity"]]:
    """Validate method policy and resolve roles before contrast processing."""
    _validate_methods(methods)
    resolved_method_roles: dict[str, Literal["decision", "sensitivity"]] = dict(method_roles or {})
    from increment.estimation.engine import resolve_method_roles

    if method_roles is None:
        resolved_method_roles = resolve_method_roles(methods)
    code = sequential_support_refusal(
        SequentialSupportRequest(
            inference=inference,
            uses_cuped=bool({"itt", "late"} & set(estimands))
            and any(method.variance_reduction == "cuped" for method in methods),
        )
    )
    if code is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported(code), metrics=tuple(m.name for m in metrics))
    if cluster is None:
        return resolved_method_roles
    if inference is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported("arm.inference.cluster"), cluster=cluster)
    if prior is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported("arm.adjustment.cluster_prior"), cluster=cluster)
    if any(m.variance_reduction == "cuped" for m in methods):
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(
            Unsupported("arm.adjustment.cluster_cuped"), cluster=cluster, encouragement=True
        )
    ratios = sorted(m.name for m in metrics if m.type == "ratio")
    if ratios:
        _refuse(
            "estimation.encouragement.cluster.ratio",
            names=ratios,
            cluster=cluster,
        )
    return resolved_method_roles


# Mirrors estimate_encouragement's policy arguments one-for-one.
def _prepare_encouragement_estimation(  # noqa: PLR0913
    metrics: Sequence[Metric],
    summary: IntoDataFrame | Iterable[Mapping[str, Any]],
    design: Encouragement,
    estimands: Sequence[str],
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    null_lift: float | None,
    null_abs: float | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    cluster: str | None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
) -> _PreparedEncouragementEstimation | DecisionComputation[LiftEstimate]:
    """Resolve request policy, index arms, and delegate ITT construction."""
    if isinstance(prior, (StudentTPrior, MixturePrior)):
        refuse(MIXTURE_PRIORS_ARE)
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        _refuse(
            "estimation.encouragement.unknown_estimand_supported",
            unknown=sorted(unknown),
            estimands=ESTIMANDS,
        )
    if alternative not in ALTERNATIVE_VALUES:
        refuse(UNKNOWN_ALTERNATIVE, alternative=alternative)
    if prior is not None and inference is not None:
        _refuse("estimation.encouragement.prior_inference_are")
    late_methods = methods if methods is not None else [Method(name="unadjusted")]
    if methods == []:
        return _empty_encouragement_decision(summary, design, estimands)
    resolved_method_roles = _resolve_encouragement_methods(
        late_methods,
        estimands,
        prior,
        inference,
        cluster,
        metrics,
        method_roles,
    )
    retention = sorted(m.name for m in metrics if getattr(m, "type", None) == "retention")
    if retention:
        _refuse("readout.encouragement.retention", names=retention)
    arms = _df_to_arms(summary)
    metric_types = {m.name: m.type for m in metrics}
    metrics_by_name = {m.name: m for m in metrics}
    control = {a.metric: a for a in arms if a.group_id == design.control_group}
    treatments = tuple(a for a in arms if a.group_id != design.control_group)
    if not control:
        _refuse(
            "estimation.encouragement.control.missing",
            control_group=design.control_group,
        )
    if inference is not None and "late" in estimands and "itt" not in estimands:
        _warn_if_open_ended_sequential(
            (m for m in metrics if m.name in {a.metric for a in arms}), stacklevel=4
        )
    itt_bundle: DecisionComputation[LiftEstimate] | None = None
    results: tuple[LiftEstimate, ...] = ()
    if "itt" in estimands:
        itt_bundle = estimate_lift(
            metrics,
            summary,
            control_group=design.control_group,
            methods=methods,
            prior=prior,
            alpha=alpha,
            alternative=alternative,
            null_lift=null_lift,
            null_abs=null_abs,
            inference=inference,
            cluster=cluster,
            method_roles=resolved_method_roles,
        )
        results = tuple(
            r.model_copy(
                update={
                    "preferred_direction": metrics_by_name[r.metric].declared_preferred_direction
                }
            )
            for r in itt_bundle.results
        )
    return _PreparedEncouragementEstimation(
        metric_types=metric_types,
        estimands=tuple(estimands),
        late_methods=tuple(late_methods),
        resolved_method_roles=resolved_method_roles,
        control=control,
        treatments=treatments,
        itt_bundle=itt_bundle,
        results=results,
    )


def _first_stage_context(
    t: ArmStats,
    c: ArmStats,
    design: Encouragement,
    *,
    cluster: str | None,
    late_requested: bool,
) -> _EncouragementFirstStageContext:
    """Compute first-stage statistics and fixed guard annotations."""
    n_clusters: int | None = None
    dof: float | None = None
    if cluster is not None:
        n_clusters = t.n + c.n
        check_total_clusters(t.metric, cluster, n_clusters, stacklevel=4)
        if t.n < 2 or c.n < 2:
            refuse(
                ARM_NEEDS_TWO,
                metric=t.metric,
                cluster=cluster,
                k_t=t.n,
                k_c=c.n,
            )
        control_uptake_total = c.n * c.mean_x()
    else:
        control_uptake_total = c.sum_d
    if design.one_sided and control_uptake_total and control_uptake_total > 0:
        _refuse(
            "estimation.encouragement.one_sided_encouragement",
            control_uptake_total=control_uptake_total,
            metric=t.metric,
        )
    if cluster is not None:
        b, var_b, dof = _first_stage_cluster(t, c)
    else:
        b, var_b = _first_stage(t, c)
    if var_b > 0:
        z_fs = b / math.sqrt(var_b)
    elif b > 0:
        z_fs = math.inf  # exactly-known positive first stage: never weak
    elif b < 0:
        z_fs = -math.inf  # exactly-known negative first stage: always weak
    else:
        z_fs = 0.0  # no first stage at all
    weak = z_fs < design.min_first_stage_z
    suppression = None
    late_caveat = ""
    if weak and late_requested:
        reason = (
            "weak instrument"
            if z_fs > -design.min_first_stage_z
            else "NEGATIVE first stage: encouragement reduced uptake"
        )
        suppression = (
            f"late suppressed: first-stage z={z_fs:.2f} < {design.min_first_stage_z:g} ({reason})"
        )
    elif not weak and z_fs < design.min_first_stage_z + _NEAR_GATE_Z_MARGIN:
        late_caveat = (
            f"; first-stage z={z_fs:.2f} clears the emission gate "
            f"({design.min_first_stage_z:g}) only narrowly: conditional on "
            "emission, near-gate estimates select on a lucky first stage "
            "and lean toward the as-treated value under confounding -- "
            "treat the point as interval-only evidence"
        )
    return _EncouragementFirstStageContext(
        b=b,
        var_b=var_b,
        z_fs=z_fs,
        weak=weak,
        n_clusters=n_clusters,
        dof=dof,
        suppression=suppression,
        late_caveat=late_caveat,
        pop=f"{design.uptake.fact} takers" if design.one_sided else "compliers",
        iv_assumes=(
            "assumes exclusion restriction"
            if design.one_sided
            else "assumes exclusion restriction and monotonicity (no defiers)"
        ),
    )


def _cluster_compliance_row(
    t: ArmStats,
    c: ArmStats,
    context: _EncouragementFirstStageContext,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
) -> LiftEstimate:
    """Build the absolute-scale compliance row for clustered moments."""
    notes = [
        "treated-arm compliance lift over control (cluster-robust, "
        "absolute scale); relative (log-RR) uptake lift is not "
        "computed for a declared cluster"
    ]
    if context.suppression:
        notes.append(context.suppression)
    return LiftEstimate(
        metric="uptake",
        group_id=t.group_id,
        method="unadjusted",
        method_role="decision",
        estimand="compliance",
        inference=inference.label if inference is not None else "fixed",
        lift=_nn_estimate(
            context.b,
            math.sqrt(context.var_b),
            prior,
            alpha,
            inference=inference,
            n_comparison=t.n + c.n,
            dof=context.dof,
        ),
        value_scale="absolute",
        scale="linear",
        note="; ".join(notes),
        n_clusters=context.n_clusters,
        **_reference_fields(context.dof),
        prior_shrunk=prior is not None,
    )


def _zero_control_compliance_row(
    t: ArmStats,
    c: ArmStats,
    design: Encouragement,
    context: _EncouragementFirstStageContext,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
) -> LiftEstimate:
    """Build the direct treated uptake-rate compliance row."""
    notes = [
        "treated-arm uptake rate (equals the compliance lift; control uptake is structurally 0)"
        if design.one_sided
        else "treated-arm uptake rate (equals the compliance lift; control uptake is zero)",
    ]
    if not design.one_sided:
        notes.append(
            "control uptake observed at zero although the design "
            "is declared two-sided -- effectively one-sided"
        )
    if context.suppression:
        notes.append(context.suppression)
    rate = t.mean_d()
    se_rate = math.sqrt(t.var_d() / t.n)
    lift = (
        _nn_estimate(
            rate,
            se_rate,
            prior,
            alpha,
            inference=inference,
            n_comparison=t.n + c.n,
        )
        if se_rate > 0
        else Estimate(value=rate)
    )
    return LiftEstimate(
        metric="uptake",
        group_id=t.group_id,
        method="unadjusted",
        estimand="compliance",
        method_role="decision",
        inference=inference.label if inference is not None else "fixed",
        lift=lift,
        value_scale="absolute",
        scale="linear",
        note="; ".join(notes),
        prior_shrunk=prior is not None,
    )


def _relative_compliance_row(
    t: ArmStats,
    c: ArmStats,
    design: Encouragement,
    context: _EncouragementFirstStageContext,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
) -> LiftEstimate:
    """Build relative compliance, falling back to its absolute lift."""
    d_rows: list[dict[str, Any]] = []
    for a in (c, t):
        assert a.sum_d is not None
        d_rows.append(
            {
                "experiment_id": a.study_id,
                "metric": "uptake",
                "group_id": a.group_id,
                "n": a.n,
                "ref_y": a.sum_d / a.n,
                "cy1": 0.0,
                "cy2": a.sum_d * (a.n - a.sum_d) / a.n,
            }
        )
    comp_metric = _compliance_metric(design.uptake.window_days)
    comp_guard_reason = "relative evidence unavailable"
    with warnings.catch_warnings():
        if design.uptake.window_days is None:
            warnings.filterwarnings("ignore", message=".*open-ended.*", category=UserWarning)
        try:
            comp = estimate_lift(
                [comp_metric],
                d_rows,
                control_group=design.control_group,
                prior=prior,
                alpha=alpha,
                inference=inference,
            )
        except LiftGuardError as exc:
            comp = None
            comp_guard_reason = str(exc)
    if comp is not None and not comp.results:
        comp_guard_reason = (
            next(iter(comp.failures.values())).display() if comp.failures else comp_guard_reason
        )
    if comp is not None and comp.results:
        return cast("Sequence[LiftEstimate]", comp.results)[0].model_copy(
            update={"estimand": "compliance", "note": context.suppression}
        )
    notes = [
        f"uptake lift on the absolute scale; relative uptake lift withheld: {comp_guard_reason}"
    ]
    if context.suppression:
        notes.append(context.suppression)
    return LiftEstimate(
        metric="uptake",
        group_id=t.group_id,
        method="unadjusted",
        estimand="compliance",
        method_role="decision",
        inference=inference.label if inference is not None else "fixed",
        lift=(
            _nn_estimate(
                context.b,
                math.sqrt(context.var_b),
                prior,
                alpha,
                inference=inference,
                n_comparison=t.n + c.n,
            )
            if context.var_b > 0
            else Estimate(value=context.b)
        ),
        value_scale="absolute",
        scale="linear",
        note="; ".join(notes),
        prior_shrunk=prior is not None,
    )


def _estimate_compliance_row(
    t: ArmStats,
    c: ArmStats,
    design: Encouragement,
    context: _EncouragementFirstStageContext,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
    cluster: str | None,
) -> LiftEstimate:
    """Select the fixed compliance reporting strategy for one contrast."""
    if cluster is not None:
        return _cluster_compliance_row(t, c, context, inference=inference, prior=prior, alpha=alpha)
    if c.sum_d == 0:
        return _zero_control_compliance_row(
            t, c, design, context, inference=inference, prior=prior, alpha=alpha
        )
    return _relative_compliance_row(
        t, c, design, context, inference=inference, prior=prior, alpha=alpha
    )


def _estimate_unadjusted_late_rows(
    t: ArmStats,
    c: ArmStats,
    context: _EncouragementFirstStageContext,
    method_strategy: _EncouragementMethodStrategy,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
    alternative: str,
    cluster: str | None,
) -> list[LiftEstimate]:
    """Build additive and relative LATE rows for one unadjusted method."""
    if cluster is not None:
        tau, se, late_dof = _late_additive_cluster(t, c)
    else:
        tau, se = _late_additive(t, c)
        late_dof = None
    row = LiftEstimate(
        **cast(dict[str, Any], _winsorization_result_fields(c, t)),
        metric=t.metric,
        group_id=t.group_id,
        method=method_strategy.method.name,
        method_role=method_strategy.method_role,
        estimand="late",
        inference=inference.label if inference is not None else "fixed",
        alternative=alternative,
        value_scale="absolute",
        scale="linear",
        lift=_nn_estimate_or_point(
            tau,
            se,
            prior,
            alpha,
            alternative,
            inference=inference,
            n_comparison=t.n + c.n,
            dof=late_dof,
        ),
        note=f"effect of uptake on {context.pop}; {context.iv_assumes}{context.late_caveat}",
        n_clusters=context.n_clusters,
        **(
            _reference_fields(late_dof)
            if inference is None
            else {"reference_kind": "sequential", "reference_df": None}
        ),
        prior_shrunk=prior is not None,
    )
    if cluster is not None:
        rel: tuple[float, float] | str = (
            "cluster-robust relative (complier-ratio) LATE is not "
            "built -- it needs the cluster-grain "
            "sum(y**2*d)/sum(x*y*d) moment family; drop cluster= "
            "to get the relative row, or read the additive row above"
        )
    else:
        rel = _late_relative(t, c)
    if isinstance(rel, str):
        row = row.model_copy(
            update={
                "note": f"effect of uptake on {context.pop}; {context.iv_assumes}; "
                f"relative form withheld: {rel}{context.late_caveat}"
            }
        )
        return [row]
    theta, se_theta = rel
    if se_theta <= 0.0:
        rel_lift = Estimate(value=math.exp(theta) - 1.0)
    else:
        est = _nn_estimate(
            theta,
            se_theta,
            prior,
            alpha,
            alternative,
            inference=inference,
            n_comparison=t.n + c.n,
        )
        assert est.lb is not None and est.ub is not None
        rel_lift = Estimate(
            value=math.exp(est.value) - 1.0,
            lb=math.exp(est.lb) - 1.0,
            ub=math.exp(est.ub) - 1.0,
            level=est.level,
            alpha=est.alpha,
        )
    return [
        row,
        LiftEstimate(
            **cast(dict[str, Any], _winsorization_result_fields(c, t)),
            metric=t.metric,
            group_id=t.group_id,
            method=method_strategy.method.name,
            method_role=method_strategy.method_role,
            estimand="late",
            inference=inference.label if inference is not None else "fixed",
            alternative=alternative,
            value_scale="relative",
            lift=rel_lift,
            note=f"relative to {context.pop} control mean; "
            f"{context.iv_assumes}{context.late_caveat}",
            prior_shrunk=prior is not None,
        ),
    ]


def _estimate_cuped_late_row(
    t: ArmStats,
    c: ArmStats,
    context: _EncouragementFirstStageContext,
    method_strategy: _EncouragementMethodStrategy,
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
    alternative: str,
) -> LiftEstimate:
    """Build the additive numerator-only CUPED LATE row."""
    tau, se, se_unadjusted = _late_additive_cuped(t, c)
    notes = [
        f"effect of uptake on {context.pop}",
        context.iv_assumes,
        "CUPED-adjusted ITT numerator over the RAW first stage",
        "relative form withheld: CUPED adjusts the additive LATE only "
        "-- the complier-relative form needs the sum(x*y*d) moment "
        "family, so it is never emitted under a cuped label",
    ]
    if se_unadjusted > 0.0 and se > se_unadjusted:
        notes.append(
            f"CUPED INFLATED this LATE's variance versus no adjustment "
            f"(SE {se / se_unadjusted:.2f}x the unadjusted SE): the "
            f"covariate drives uptake more than it predicts the outcome "
            f"net of uptake -- pass variance_reduction='none' for metric "
            f"'{t.metric}' to recover the tighter interval"
        )
    return LiftEstimate(
        **cast(dict[str, Any], _winsorization_result_fields(c, t)),
        metric=t.metric,
        group_id=t.group_id,
        method=method_strategy.method.name,
        method_role=method_strategy.method_role,
        estimand="late",
        inference=inference.label if inference is not None else "fixed",
        alternative=alternative,
        value_scale="absolute",
        scale="linear",
        lift=_nn_estimate_or_point(
            tau,
            se,
            prior,
            alpha,
            alternative,
            inference=inference,
            n_comparison=t.n + c.n,
        ),
        note="; ".join(notes) + context.late_caveat,
        prior_shrunk=prior is not None,
    )


def _estimate_late_rows(
    t: ArmStats,
    c: ArmStats,
    context: _EncouragementFirstStageContext,
    method_strategies: tuple[_EncouragementMethodStrategy, ...],
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Normal | None,
    alpha: float,
    alternative: str,
    cluster: str | None,
) -> list[LiftEstimate]:
    """Build LATE rows in the configured method order."""
    rows: list[LiftEstimate] = []
    for method_strategy in method_strategies:
        if method_strategy.use_cuped:
            rows.append(
                _estimate_cuped_late_row(
                    t,
                    c,
                    context,
                    method_strategy,
                    inference=inference,
                    prior=prior,
                    alpha=alpha,
                    alternative=alternative,
                )
            )
        else:
            rows.extend(
                _estimate_unadjusted_late_rows(
                    t,
                    c,
                    context,
                    method_strategy,
                    inference=inference,
                    prior=prior,
                    alpha=alpha,
                    alternative=alternative,
                    cluster=cluster,
                )
            )
    return rows


# Public estimator signature is the API for encouragement results.
def estimate_encouragement(  # noqa: PLR0913
    metrics: Sequence[Metric],
    summary: SequentialSnapshot | IntoDataFrame | Iterable[Mapping[str, Any]],
    design: Encouragement,
    *,
    estimands: Sequence[str] = ESTIMANDS,
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    alpha: float | None = None,
    alternative: str | None = None,
    null_lift: float | None = None,
    null_abs: float | None = None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    cluster: str | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
) -> DecisionComputation[LiftEstimate]:
    """Estimate ITT, compliance (first stage), and LATE for an encouragement design.

    ``methods`` is one reporting configuration per emitted row family
    (default ``Method(name="unadjusted")``). ``variance_reduction="cuped"``
    adjusts the ``itt`` rows and the additive ``late`` row (same pooled
    theta, so cuped LATE == cuped ITT's ``abs_diff`` over compliance
    exactly); it withholds the complier-relative ``late`` row (unbuilt
    moment family) and never adjusts ``compliance`` (its theta would be
    wrong for D, and it is the weak-first-stage gate's own diagnostic).
    Refused with ``cluster``. A near-zero metric under a strong ``itt``
    adjustment can push a CUPED-adjusted arm mean non-positive, aborting the call.

    ``prior`` forwards to every requested estimand (default: approximately
    flat), but each row lives on a different scale (``itt``/two-sided
    ``compliance``: log-RR; rate-form ``compliance``: absolute rate;
    additive ``late``: absolute tau; relative ``late``: log complier
    ratio), so an informative prior only makes sense on a single-scale
    request. Refused with ``cluster`` (t reference) or as a mixture prior
    (multi-scale rows don't persist raw pre-prior statistics).

    ``alternative`` forwards to ``itt`` and both ``late`` rows (the same
    alpha-doubling identity as ``infer_lift``); ``compliance`` is
    unaffected, since it diagnoses instrument strength, not a hypothesis.

    ``null_lift``/``null_abs`` forward to ``itt`` only: the ITT row is the
    randomized assignment contrast, so a declared non-inferiority margin
    applies to it unchanged. ``late`` and ``compliance`` carry no shifted
    null. Fixed-horizon only; sequential ``inference`` refuses a shifted null.

    Registered inference consumes a finalized exact snapshot for raw ITT and
    relative Bernoulli uptake in two-sided designs. Binary-uptake LATE, adjusted
    scores, structural-zero uptake rate targets, priors and clusters require
    other proofs and refuse; their fixed-horizon behavior remains available.

    ``cluster`` declares the randomization-grain column, mirroring
    ``estimate_lift(cluster=...)``: *summary* must carry the clustered
    collapse for every metric and the uptake fact. ``itt`` and additive
    ``late`` switch to the cluster-robust ratio delta method and a
    ``t_{K_T + K_C - 2}`` working reference; each arm needs at least 2
    clusters and below 40 total clusters emits a warning. ``compliance``
    reports directly on the absolute scale instead of the log-RR delegation
    (point estimates unchanged).
    Refuses a CUPED method, an informative ``prior`` or sequential
    ``inference``, and any ratio metric on any estimand; withholds the
    complier-relative ``late`` row (unbuilt moment family).
    """
    _require_exclusion_for_late(design, estimands)
    if inference is not None:
        from increment.estimation.decision_types import DecisionComputation
        from increment.estimation.sequential_runtime import estimate_sequential

        if methods == []:
            return DecisionComputation(results=(), evidence={}, failures={})

        if (
            "late" in estimands
            or prior is not None
            or cluster is not None
            or null_lift not in (None, 0.0)
            or null_abs is not None
            or any(
                m.variance_reduction != "none" or m.name != "unadjusted" for m in (methods or ())
            )
        ):
            sequential_refuse(
                "route.unsupported",
                "raw ITT and Bernoulli relative uptake are supported; binary-uptake LATE, adjusted scores and shifted nulls require another proof",
            )
        if not isinstance(summary, SequentialSnapshot):
            sequential_refuse(
                "source.invalid", "sequential encouragement requires a finalized exact checkpoint"
            )
        if {c.estimand for c in summary.registration.roster} != set(estimands):
            sequential_refuse(
                "source.invalid", "requested estimands differ from the retained roster"
            )
        if design.control_group != summary.registration.control_group:
            sequential_refuse("source.invalid", "encouragement control differs from registration")
        if design.one_sided and any(
            c.estimand == "compliance" for c in summary.registration.roster
        ):
            sequential_refuse(
                "route.unsupported", "structural-zero uptake needs the existing C03 rate target"
            )
        from increment.estimation.sequential_runtime import validate_engine_request

        validate_engine_request(
            summary, metrics if "itt" in estimands else (), alpha=alpha, alternative=alternative
        )
        return estimate_sequential(summary, inference)

    if isinstance(summary, SequentialSnapshot):
        sequential_refuse(
            "source.invalid", "an exact snapshot requires its registered runtime policy"
        )
    alpha = 0.05 if alpha is None else alpha
    alternative = "two-sided" if alternative is None else alternative
    prepared = _prepare_encouragement_estimation(
        metrics,
        summary,
        design,
        estimands,
        methods,
        prior,
        alpha,
        alternative,
        null_lift,
        null_abs,
        inference,
        cluster,
        method_roles,
    )
    if not isinstance(prepared, _PreparedEncouragementEstimation):
        return prepared
    normal_prior = cast("Normal | None", prior)
    method_strategies = tuple(
        _EncouragementMethodStrategy(
            method=method,
            method_role=prepared.resolved_method_roles.get(method.name, "decision"),
            use_cuped=method.variance_reduction == "cuped",
        )
        for method in prepared.late_methods
    )
    out = list(prepared.results)
    late_failures: list[tuple[str, str, float]] = []
    reported_uptake_arms: set[str] = set()
    for t in prepared.treatments:
        c = prepared.control.get(t.metric)
        if c is None:
            continue
        context = _first_stage_context(
            t,
            c,
            design,
            cluster=cluster,
            late_requested="late" in prepared.estimands,
        )
        if context.weak and "late" in prepared.estimands:
            late_failures.append((t.metric, t.group_id, context.z_fs))
        if context.suppression is not None or (
            t.group_id not in reported_uptake_arms and "compliance" in prepared.estimands
        ):
            reported_uptake_arms.add(t.group_id)
            out.append(
                _estimate_compliance_row(
                    t,
                    c,
                    design,
                    context,
                    inference=inference,
                    prior=normal_prior,
                    alpha=alpha,
                    cluster=cluster,
                ).model_copy(update={"metric": t.metric} if context.suppression else {})
            )
        if "late" not in prepared.estimands or context.weak:
            continue
        if prepared.metric_types.get(t.metric) == "ratio":
            _refuse("estimation.encouragement.late.ratio", metric=t.metric)
        out.extend(
            _estimate_late_rows(
                t,
                c,
                context,
                method_strategies,
                inference=inference,
                prior=normal_prior,
                alpha=alpha,
                alternative=alternative,
                cluster=cluster,
            )
        )
    return _encouragement_decision_bundle(
        out,
        inference=inference,
        late_failures=late_failures,
        itt_bundle=prepared.itt_bundle,
    )


READOUT_ENCOURAGEMENT_VALUE_SCALE = _REFUSALS["readout.encouragement.value_scale"]


ESTIMATION_ENCOURAGEMENT_UNKNOWN_ESTIMAND_SUPPORTED = _REFUSALS[
    "estimation.encouragement.unknown_estimand_supported"
]
