"""Power analysis — sample size, achieved power, minimum detectable effect.

The three arm solvers invert the log-ratio-of-means estimator's variance
with each arm evaluated at ITS OWN mean under the alternative. With
``v = baseline.effective_var``, ``m0 = baseline.mean``, the effective log
ratio ``theta = log1p(relative_lift * compliance)`` and ``m1 = m0 * exp(theta)``,
the planning variance at analyzed arm counts is::

    S^2(theta) = v1 / (n_T * m1^2) + v / (n_C * m0^2)

where ``v1 = v`` for a mean-like metric (equal absolute variance in both
arms) and ``v1 = v * m1 * (1 - m1) / (m0 * (1 - m0))`` for a conversion or
retention metric (a Bernoulli shape sharing the baseline's declared
variance multiplier). Both are planning assumptions, not guarantees for
every data-generating process. A bounded metric's alternative rate must
stay at or below one; the treatment rate exactly one contributes zero
treatment variance.

A conversion or retention plan the runtime decides with the exact binomial
risk-ratio test (unadjusted, unclustered, fixed horizon, no prior, raw counts)
does not use this model. ``power_basis="exact"`` integrates that decision's
rejection probability at the analyzed integer counts. ``"approximate"``
instead integrates a Normal-conditional-tail decision model; its numerical
certificates do not bound the model's departure from runtime inference.
The Bernoulli shape above applies to the remaining conversion and retention plans.

Because the variance depends on the alternative, a decreasing direction's
noncentrality ``d / S(theta0 - d)`` rises to a single peak and then falls:
the minimum detectable effect is the FIRST distance reaching the target
power inside the admissible distance interval, which can start above zero
or be empty when a shifted null lies outside the effective alternatives
partial compliance can reach. A target above the reachable maximum has no
minimum detectable effect; supplied-effect queries still answer and carry
the missing companion as a numeric null with its reason.

The segment-pairwise solvers keep the baseline-only variance
``v / m^2 * (1/n_T + 1/n_C)`` for each segment: an explicitly documented
approximation, not the arm model above. They refuse a quantile metric with
the readout's own breakout refusal, since a quantile readout has no segments.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal, NoReturn, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from scipy.special import ndtr as _ndtr
from scipy.special import ndtri as _ndtri
from scipy.stats import chi2 as _chi2
from scipy.stats import ncx2 as _ncx2
from scipy.stats import norm as _norm

from increment._literals import Alternative
from increment.compatibility import (
    PowerDesign,
    Unsupported,
    refuse_unsupported,
)
from increment.decision import FixedInference
from increment.errors import (
    CodedError,
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.estimation._readout_refusals import READOUT_REFUSALS
from increment.estimation.arm_contract import (
    ArmPlanningProcedure,
    RelativeDecisionPolicy,
    arm_planning_support,
)
from increment.estimation.armstats import SummaryStats
from increment.estimation.binomial_rr import FINITE_SAMPLE_MAX_ARM_SIZE, nuisance_beta
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.meta import ESTIMATION_META_VAR_FINITE_STRICTLY
from increment.estimation.quantile import (
    QuantileArm,
    _bracket_half_width,
    _bracket_ranks,
    _quantile_n_min,
    quantile_half_width,
)
from increment.estimation.quantile import _raise as _quantile_raise
from increment.estimation.sequential import GaussianScoreMixture
from increment.power._binomial import (
    PLANNING_CELL_CEILING,
    BinomialDecision,
    BinomialPower,
    RejectionGeometry,
    ReplayBoundExceeded,
    Route,
    refused,
    route_for,
    solver_floor,
    solver_refuses,
    tail_margin,
    window_cells,
)
from increment.power._noncentral_t import _scalar_power_from_nc
from increment.power._search import (
    _LOG_FLOAT_MAX,
    _MDE_NUMERICAL_RESOLUTION,
    _MdeRefusal,
    _noncentrality,
    bisect_first_true,
    float_from_ordinal,
    float_ordinal,
)
from increment.power._validation import (
    _require_finite,
    _require_relative_domain,
    _validate_planned_looks,
)
from increment.power.sequential import (
    _NODE_CEILING,
    SequentialPlanningSpec,
    _certified_crossing,
    _CrossingEnclosure,
    _PlannedLooks,
    _planning_bounds_from_log_se,
    _require_prospective,
    _resolve_looks,
    _sequential_estimates_from_log_se,
    _SequentialMde,
    _SequentialMdeSearch,
)
from increment.semantics.models import InferenceSpec, MethodSpec, QuantileMetric

_FLOAT_MAX = sys.float_info.max
# Largest treatment arm the quantile size search probes: the binomial bracket
# ranks stay accurate to well past it, and it exceeds any real population.
_MAX_QUANTILE_ARM = 2**40

MdeUnavailableReason = Literal["unattainable", "unrepresentable", "numerical_resolution"]
PowerBasis = Literal["asymptotic", "exact", "approximate"]


# ``limiting_condition`` is "recording_grid" when target power is at or above
# ``maximum_power``, the largest power the size search found on its grid of
# sizes or in the large-sample limit; "sample_size_limit" when the target is
# reachable only beyond 2**40 units per arm.
_SEGMENT_PAIRWISE_N_PER_ARM_TOO_SMALL = RefusalSpec(
    "power.segment_pairwise_achieved_n_per_arm_too_small",
    InvalidRequestError,
    template=(
        "n_per_arm={n_per_arm} is too small to split segments A (q={q_a}) and B "
        "(q={q_b}) into 2-arm designs ({exc}); increase n_per_arm or the "
        "smaller segment's share"
    ),
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "power.mde_relative_unavailable": "mde_relative is unavailable but mde_unavailable_reason is None: a missing minimum detectable effect must state why",
        "power.mde_unavailable_reason": "mde_unavailable_reason={mde_unavailable_reason!r} contradicts an available mde_relative={mde_relative}",
        "power.minimum_detectable_effect.design_search_minimum": "fixed-horizon design cannot search for a minimum detectable effect when target_power={target} is at or below the null's own crossing probability ({estimate:.6g}) -- zero distance already qualifies, so no strictly nonzero minimum detectable effect exists",
        "power.baseline.mean": "mean must be > 0",
        "power.baseline.var_zero_variance": "var must be > 0 -- a zero-variance (deterministic) metric has no power question to answer (every solver divides by the log-ratio variance); from_proportion(p) requires 0 < p < 1",
        "power.baseline.cuped_rho": "cuped_rho must be in (-1, 1)",
        "power.baseline.compliance": "compliance must be in (0, 1]",
        "power.baseline.trigger_rate": "trigger_rate must be in (0, 1]",
        "power.baseline.cluster_participation": RefusalSpec(
            "power.baseline.cluster_participation",
            InvalidRequestError,
            template="cluster_participation must be finite in (0, 1]",
            keys=frozenset({"cluster_participation", "route"}),
        ),
        "power.baseline.cluster_participation_triggered": RefusalSpec(
            "power.baseline.cluster_participation_triggered",
            InvalidRequestError,
            template="clustered triggered planning requires cluster_participation: count contributing analyzed clusters and recruited assigned clusters in the pilot",
            keys=frozenset({"avg_cluster_size", "cluster_participation", "trigger_rate", "route"}),
        ),
        "power.baseline.cluster_participation_untriggered": RefusalSpec(
            "power.baseline.cluster_participation_untriggered",
            InvalidRequestError,
            template="cluster_participation must be 1.0 for an untriggered baseline",
            keys=frozenset({"cluster_participation", "trigger_rate", "route"}),
        ),
        "power.baseline.cluster_analyzed_mean": RefusalSpec(
            "power.baseline.cluster_analyzed_mean",
            InvalidRequestError,
            template="assigned mean {avg_cluster_size}, trigger fraction {trigger_rate}, and cluster participation {cluster_participation} imply analyzed mean {analyzed_mean}; it must be finite and at least one -- recount assigned units and contributing pilot clusters",
            keys=frozenset({"route"}),
        ),
        "power.baseline.icc": "icc must be in [0, 1)",
        "power.baseline.cluster_icc": "cluster_icc must be in [0, 1)",
        "power.baseline.avg_cluster_size": "avg_cluster_size must be >= 1.0 (expected assigned units per randomized cluster), got {avg_cluster_size}",
        "power.baseline.cluster_size_cv": "cluster_size_cv must be >= 0",
        "power.baseline.cluster_icc_avg": RefusalSpec(
            "power.baseline.cluster_icc_avg",
            InvalidRequestError,
            template="cluster_icc={cluster_icc} > 0 requires analyzed mean cluster size > 1 (got {analyzed_mean}); a one-unit contributing cluster does not identify within-cluster correlation -- use a pilot with repeated observations within contributing clusters, or leave cluster_icc at 0",
            keys=frozenset({"route"}),
        ),
        "power.baseline.summarystats_bound_pilot": "SummaryStats.n must be >= 2 to bound the pilot variance's own sampling uncertainty (got n={n}); a single unit has no degrees of freedom for a variance estimate",
        "power.baseline.confidence": "confidence must be in (0, 1)",
        "power.baseline.confidence_produces_non": "confidence produces a non-finite variance inflation",
        "power.baseline.mean_den": "mean_den must be > 0, got {mean_den}",
        "power.quantile_baseline.cuped_unsupported": "cuped_rho={cuped_rho} was declared, but a quantile metric has no runtime CUPED route to credit -- QuantileBaseline always plans without variance reduction",
        "power.quantile_baseline.metric_not_quantile": "a QuantileBaseline plans a quantile metric, but {metric!r} is declared as {metric_type!r} -- plan a quantile metric with ArmPlanningProcedure.standard('quantile'), and any other metric with Analysis.planning_baseline(metric) or a Baseline constructor",
        "power.quantile_size_search_unreachable": RefusalSpec(
            "power.quantile_size_search_unreachable",
            InvalidRequestError,
            lambda *, power, maximum_power, limiting_condition: (
                f"target power {power} is not reachable for this quantile metric: "
                + (
                    "the pilot's recording grid keeps the quantile's standard error above "
                    f"a floor, so no sample size exceeds power {maximum_power:.6g} -- lower "
                    "the target power, plan a larger relative_lift, or record the metric at "
                    "a finer resolution"
                    if limiting_condition == "recording_grid"
                    else "it needs more than 2**40 (about 1.1e12) units per arm -- plan a "
                    "larger relative_lift"
                )
            ),
        ),
        "power.power.check": "power must be in [0, 1], got {power}",
        "power.power.effective_var": "effective_var must be > 0, got {effective_var}",
        "power.log_exp_needs": "log(1 - exp(x)) needs x <= 0, got {x}",
        "power.procedure_armplanningprocedure": "procedure must be ArmPlanningProcedure, got {procedure_type}",
        "power.power_solvers_relative": "power solvers require a relative ArmPlanningProcedure decision",
        "power.core.n_per_arm_int": "n_per_arm must be an integer (got {n_per_arm!r})",
        "power.core.n_per_arm_min": "n_per_arm must be >= {minimum} for metric {metric_type!r} (got {n_per_arm})",
        "power.sequential_sample_size": "sequential sample-size power could not meet its numerical tolerance at n_per_arm={n_per_arm}: {reason}",
        "power.noncentral_t_unresolved": "noncentral-t power at noncentrality {nc!r} with {dof!r} degrees of freedom and tail allocation {tail_alpha!r} is unresolved: its tail integral did not resolve within the quadrature's panel limits",
        "power.sequential_sample_size_target_inside_enclosure": "sequential sample-size target lies inside the certified power enclosure at n_per_arm={n_per_arm}: ({lower}, {upper})",
        "power.relative_lift_lies": "relative_lift={relative_lift} lies below the null boundary (null_lift={null_lift}) but alternative='greater' -- no sample size gives this design more than alpha power; flip the alternative or the sign of relative_lift",
        "power.relative_lift_lies_above_null": "relative_lift={relative_lift} lies above the null boundary (null_lift={null_lift}) but alternative='less' -- no sample size gives this design more than alpha power; flip the alternative or the sign of relative_lift",
        "power.size_design_relative": "cannot size a design for a relative_lift exactly at the null boundary (relative_lift={relative_lift}, null_lift={null_lift}): the distance to detect is zero, so no finite sample size reaches any power above alpha -- pass a lift away from the null (the same refusal segment_pairwise_required_sample_size makes for theta=0)",
        "power.sample_size_detecting": "the sample size detecting relative_lift={relative_lift} against null_lift={null_lift} at this baseline exceeds the float64 range: the log-scale distance is too small for the planning variance",
        "power.sequential_design_more": "sequential design requires more than 1024x the fixed-horizon sample size to reach power={power} -- check the InferenceSpec(kind='asymptotic_mean') declared on ArmPlanningProcedure.standard() for a mismatch with the target relative_lift",
        "power.segment_share_n": "segment share q={q} of n_total={n_total} implies only {n_total_seg:.3g} units, too few for a 2-arm design at allocation={allocation} (need >= {min_n_total_seg:.3g} total at this allocation); increase n_total or the segment's share",
        "power.q_a": "q_a must be in (0, 1)",
        "power.q_b": "q_b must be in (0, 1)",
        "power.q_a_q": "q_a + q_b must be <= 1, got {q_a_plus_q_b}",
        "power.segment_pairwise_solvers": RefusalSpec(
            "power.segment_pairwise_solvers",
            InvalidRequestError,
            lambda *, unsupported: (
                "segment-pairwise solvers do not support non-default Baseline fields: "
                + ", ".join(unsupported)
            ),
        ),
        "power.segment_clustered_baseline": "segment {segment}'s clustered baseline requires at least two clusters per arm (got {k_total} total, n_t={n_t}, n_c={n_c}); increase n_total or the segment's share",
        "power.supports_fixed_horizon": "{caller} supports fixed-horizon ArmPlanningProcedure only",
        "power.segment_pairwise_required": "segment_pairwise_required_sample_size does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here",
        "power.r_a_below": "r_a ({r_a}) is below r_b ({r_b}) (theta={theta:.6g}) but alternative='greater' -- no sample size gives this design more than alpha power; flip the alternative or the order of r_a and r_b",
        "power.r_a_above": "r_a ({r_a}) is above r_b ({r_b}) (theta={theta:.6g}) but alternative='less' -- no sample size gives this design more than alpha power; flip the alternative or the order of r_a and r_b",
        "power.r_a_r": "r_a ({r_a}) and r_b ({r_b}) give the same relative lift (theta=0); no finite sample size can distinguish segment A from segment B in this design",
        "power.solved_too_small": "the solved-for N ({n_total}) is too small to split segments A (q={q_a}) and B (q={q_b}) into 2-arm designs ({exc}); this can happen when the contrast between r_a and r_b is large relative to the smaller segment's share, needing more of the solved-for N per segment than a 2-arm split allows -- narrow the contrast, raise the smaller segment's share, or accept that an easy-to-detect contrast needs a manually-chosen larger N",
        "power.segment_pairwise_achieved": "segment_pairwise_achieved_power does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here",
        "power.segment_pairwise_minimum": "segment_pairwise_minimum_detectable_effect does not support a shifted null (null_lift={null_lift}): the underlying formula has no theta0 term, so a nonzero null is not well-defined here",
        "power.segment_pairwise_achieved_n_per_arm_too_small": _SEGMENT_PAIRWISE_N_PER_ARM_TOO_SMALL,
        "power.tau_b": "tau_b must be >= 0, got {tau_b}",
        "power.theta_var_one": "theta and var must be 1-D (one value per segment), got shapes {theta_shape} and {var_shape}",
        "power.theta_var_same": "theta and var must have the same shape, got {theta_shape} vs {var_shape}",
        "power.need_least_segments": "Need at least 2 segments for a joint Q-test power calculation, got {k}",
        "power.theta_contains_non": "theta contains non-finite values",
    },
)

_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA

_REFUSALS["estimation.meta.var_finite_strictly"] = ESTIMATION_META_VAR_FINITE_STRICTLY
_raise = raiser(_REFUSALS)


def _require_companion_mde(
    mde_relative: float | None, mde_unavailable_reason: MdeUnavailableReason | None
) -> None:
    """An available minimum detectable effect stays above the shared
    ``-1.0`` floor and carries no unavailability reason; a missing one
    always carries a reason. Shared by every result row that reports the
    companion effect."""
    from increment.power._search import require_reason_when_null

    def invalid(_message: str) -> None:
        if mde_relative is None:
            _raise("power.mde_relative_unavailable")
        _raise(
            "power.mde_unavailable_reason",
            mde_relative=mde_relative,
            mde_unavailable_reason=mde_unavailable_reason,
        )

    require_reason_when_null(
        mde_relative, mde_unavailable_reason, name="mde_relative", invalid=invalid
    )
    if mde_relative is not None:
        _require_relative_domain("mde_relative", mde_relative)


# Domain models


def ratio_linearized_variance(
    mean_num: float, mean_den: float, var_num: float, var_den: float, cov_num_den: float
) -> float:
    """Per-unit variance of a ratio metric, ``Var(Num - R*Den) / mean_den**2`` with
    ``R = mean_num / mean_den``, unvalidated so callers can inspect a degenerate value."""
    r = mean_num / mean_den
    return (var_num + r * r * var_den - 2.0 * r * cov_num_den) / (mean_den**2)


# Three pilot ratios, two operations, and comparison incur at most six
# roundings; eight unit roundoffs bound their error for normal inputs.
_CLUSTER_MEAN_ROUNDOFF = 4 * math.ulp(1.0)


def _analyzed_cluster_mean(
    assigned_mean: float, trigger_rate: float, participation: float
) -> float:
    """Recover a one-unit mean obscured by rounded pilot ratios."""
    mean = assigned_mean * trigger_rate / participation
    return 1.0 if abs(mean - 1.0) <= _CLUSTER_MEAN_ROUNDOFF else mean


class Baseline(CodedModel, BaseModel):
    """Control-arm assumptions.

    Parameters
    ----------
    mean : float
        Control mean; relative lift is expressed against this.
    var : float
        Per-unit outcome variance (control arm, ddof=1 scale).
    cuped_rho : float
        |correlation| with a pre-period covariate; reduces variance by
        ``(1 - rho^2)``. Default 0.0.
    compliance : float
        Expected first-stage uptake lift under encouragement, in
        ``(0, 1]``; inflates required n by ``1 / compliance**2`` since
        the ITT machinery must detect ``delta * compliance``. Default 1.0.
    icc : float
        Intraclass correlation of a factor absorbed via
        ``absorb_one_way`` (categorical CUPED); reduces variance by
        ``(1 - icc)``. See ``from_absorption``. Opposite direction from
        ``cluster_icc``, which inflates variance. Default 0.0.
    cluster_icc : float
        Intraclass correlation of the analyzed score within contributing
        analyzed clusters; inflates variance through the design effect.
    avg_cluster_size : float
        Expected assigned units per randomized cluster; drives recruitment
        counts. Triggered analyses convert this to an analyzed mean size using
        ``avg_cluster_size * trigger_rate / cluster_participation``.
    cluster_size_cv : float
        Coefficient of variation of analyzed sizes among contributing clusters.
    cluster_participation : float | None
        Pilot fraction of recruited clusters contributing analyzed observations.
        Required for clustered triggered planning, and defaults to 1.0 only for
        untriggered baselines.
    trigger_rate : float
        Share of assigned units expected to become treatment-eligible,
        in ``(0, 1]``; inflates required n by 1 / ``trigger_rate``.
        Distinct from ``compliance``, which inflates by the square - use
        ``compliance`` for a diluted-effect readout instead. Default 1.0.
    """

    model_config = ConfigDict(frozen=True)

    mean: float
    var: float
    cuped_rho: float = 0.0
    compliance: float = 1.0
    trigger_rate: float = 1.0
    icc: float = 0.0
    # Cluster knobs live on Baseline, not PowerDesign: every solver reads variance
    # only through Baseline.effective_var, and `compliance` set that precedent.
    cluster_icc: float = 0.0
    avg_cluster_size: float = 1.0
    cluster_size_cv: float = 0.0
    cluster_participation: float | None = None

    @model_validator(mode="after")
    def _check(self) -> Baseline:
        _require_finite("mean", self.mean)
        if self.mean <= 0:
            _raise("power.baseline.mean")
        _require_finite("var", self.var)
        if self.var <= 0:
            _raise("power.baseline.var_zero_variance")
        if not -1 < self.cuped_rho < 1:
            _raise("power.baseline.cuped_rho")
        if not 0 < self.compliance <= 1:
            _raise("power.baseline.compliance")
        if not 0 < self.trigger_rate <= 1:
            _raise("power.baseline.trigger_rate")
        if not 0.0 <= self.icc < 1.0:
            _raise("power.baseline.icc")
        if not 0.0 <= self.cluster_icc < 1.0:
            _raise("power.baseline.cluster_icc")
        if self.cluster_participation is not None:
            _require_finite("cluster_participation", self.cluster_participation)
            if not 0.0 < self.cluster_participation <= 1.0:
                _raise(
                    "power.baseline.cluster_participation",
                    cluster_participation=self.cluster_participation,
                    route="count contributing clusters / assigned clusters in the pilot",
                )
        if self.trigger_rate == 1.0:
            if self.cluster_participation not in (None, 1.0):
                _raise(
                    "power.baseline.cluster_participation_untriggered",
                    trigger_rate=self.trigger_rate,
                    cluster_participation=self.cluster_participation,
                    route="use full participation for an untriggered population",
                )
        elif self.avg_cluster_size > 1.0 and self.cluster_participation is None:
            _raise(
                "power.baseline.cluster_participation_triggered",
                trigger_rate=self.trigger_rate,
                avg_cluster_size=self.avg_cluster_size,
                cluster_participation=None,
                route="count contributing clusters / assigned clusters in the pilot",
            )
        _require_finite("avg_cluster_size", self.avg_cluster_size)
        if self.avg_cluster_size < 1.0:
            _raise("power.baseline.avg_cluster_size", avg_cluster_size=self.avg_cluster_size)
        _require_finite("cluster_size_cv", self.cluster_size_cv)
        if self.cluster_size_cv < 0.0:
            _raise("power.baseline.cluster_size_cv")
        analyzed_mean = self.avg_cluster_size
        if self.cluster_participation is not None:
            analyzed_mean = _analyzed_cluster_mean(
                self.avg_cluster_size, self.trigger_rate, self.cluster_participation
            )
            if not math.isfinite(analyzed_mean) or analyzed_mean < 1.0:
                _raise(
                    "power.baseline.cluster_analyzed_mean",
                    avg_cluster_size=self.avg_cluster_size,
                    trigger_rate=self.trigger_rate,
                    cluster_participation=self.cluster_participation,
                    analyzed_mean=analyzed_mean,
                    route="recount assigned units and contributing pilot clusters",
                )
        if self.cluster_icc > 0.0 and analyzed_mean <= 1.0:
            _raise(
                "power.baseline.cluster_icc_avg",
                cluster_icc=self.cluster_icc,
                analyzed_mean=analyzed_mean,
                route="use a pilot with within-cluster replication, or leave cluster_icc at 0",
            )
        return self

    # Computed
    @property
    def design_effect(self) -> float:
        """Kish DEFF on analyzed sizes of contributing clusters."""
        if self.avg_cluster_size <= 1.0 or self.cluster_icc == 0.0:
            return 1.0
        participation = 1.0 if self.cluster_participation is None else self.cluster_participation
        analyzed_mean = _analyzed_cluster_mean(
            self.avg_cluster_size, self.trigger_rate, participation
        )
        m_eff = (1.0 + self.cluster_size_cv**2) * analyzed_mean
        return 1.0 + (m_eff - 1.0) * self.cluster_icc

    @property
    def effective_var(self) -> float:
        """Variance after CUPED / absorption reduction and cluster inflation.

        ``var * (1 - rho^2) * (1 - icc) * design_effect``. The factors are
        applied independently for planning purposes only: the underlying
        estimators (CUPED regression, factor absorption, cluster-robust
        variance) are not jointly fit and do not compose in production
        (see ``docs/guides/cuped.md``). ``icc`` reduces variance by
        absorbing a nuisance factor; ``cluster_icc`` inflates it because
        randomization happened at a coarser grain - stacking them is an
        independent-factors approximation, not a fitted joint model.
        """
        return self.var * (1 - self.cuped_rho**2) * (1 - self.icc) * self.design_effect

    # Constructors

    @classmethod
    def from_summary(
        cls, s: SummaryStats, cuped_rho: float = 0.0, *, confidence: float = 0.80
    ) -> Baseline:
        """Build from a pilot experiment's summary statistics.

        ``s.var`` is inflated to its ``confidence``-level one-sided upper
        confidence bound (chi-square, ``s.n - 1`` degrees of freedom)
        before use: a point-estimate variance from a small pilot
        understates the true sampling uncertainty in the variance itself,
        so a naive plug-in would size/power a 20-unit pilot identically to
        a 2M-unit one. The inflation factor is ``df / chi2.isf(confidence, df)``
        and vanishes as ``s.n -> inf`` (that ratio -> 1).

        Raises ``ValueError`` if ``s.n < 2``: a variance has no degree of
        freedom to bound below that.
        """
        if s.n < 2:
            _raise("power.baseline.summarystats_bound_pilot", n=s.n)
        if not 0.0 < confidence < 1.0:
            _raise("power.baseline.confidence")
        df = s.n - 1
        inflated_var = s.var * df / _chi2.isf(confidence, df)
        if not math.isfinite(inflated_var):
            _raise("power.baseline.confidence_produces_non")
        return cls(mean=s.mean, var=inflated_var, cuped_rho=cuped_rho)

    @classmethod
    def from_proportion(cls, p: float, cuped_rho: float = 0.0) -> Baseline:
        """Build from a Bernoulli proportion (mean=p, var=p*(1-p)).

        This is the natural constructor for conversion / retention metrics.
        """
        return cls(mean=p, var=p * (1 - p), cuped_rho=cuped_rho)

    @classmethod
    def from_absorption(cls, mean: float, var: float, icc: float) -> Baseline:
        """Build a Baseline that credits an absorbed factor's SE reduction.

        Absorption-aware analogue of ``cuped_rho``: ``icc`` is the
        intraclass correlation a live ``AbsorptionResult.icc`` (from
        ``absorb_one_way``) measures for a one-way factor - the share of
        ``var`` between factor levels. ``effective_var`` applies the
        first-order reduction ``var * (1 - icc)``, the absorption
        analogue of CUPED's ``var * (1 - rho^2)``; not to be confused
        with ``cluster_icc``, which inflates variance via the design
        effect instead of reducing it.

        Only changes ``effective_var`` on the input side, so it feeds
        every solver's ``se2`` the same way regardless of inference axis.

        The ``(1 - icc)`` mapping is measured, not asserted: cross-checked
        against ``absorb_one_way`` on simulated one-way-factor data at
        ICC 0.04-0.80, the predicted SE cut ``1 - sqrt(1 - icc)`` tracks
        the measured ``AbsorptionResult.se_reduction`` within about a
        percentage point (see ``tests/power/test_absorption_knob.py``).
        At ``icc=0`` this returns ``effective_var == var`` exactly:
        ``absorb_one_way`` measures an apparent ~0.5-1% SE cut even at
        true ICC 0, but that is a finite-K sandwich-vs-naive-SE artifact,
        not real variance reduction, and the ``icc`` knob here must not
        credit it.

        Does not compose with CUPED in the estimator: ``absorb_one_way``
        and a CUPED regression are not jointly fit against the same
        data, so stacking ``icc`` and ``cuped_rho`` is a planning
        approximation, not a guarantee both reductions are
        simultaneously achievable.
        """
        return cls(mean=mean, var=var, icc=icc)

    @classmethod
    def from_ratio(
        cls,
        mean_num: float,
        mean_den: float,
        var_num: float,
        var_den: float,
        cov_num_den: float,
        cuped_rho: float = 0.0,
    ) -> Baseline:
        """Build from a ratio metric's pilot moments (numerator over denominator).

        The per-unit quantity a ratio's power question needs is NOT the
        numerator's variance -- it is the linearized ``Var(y - R*den) /
        mean_den**2`` (``R = mean_num / mean_den``), which also charges
        for the denominator's own variance and its covariance with the
        numerator. Passing the numerator's variance alone (e.g. via the
        bare constructor) silently under- or over-states the required
        sample size for any ratio metric whose denominator varies across
        units (sessions per user, items per cart, and so on).

        ``mean_num``/``mean_den``: per-unit numerator/denominator means.
        ``var_num``/``var_den``: per-unit numerator/denominator variances
        (ddof=1). ``cov_num_den``: per-unit covariance between numerator
        and denominator (ddof=1).

        This is the same linearization the estimation-side
        ``RatioVarianceModel`` uses from ``ArmStats``' ``cyden`` / ``cden2``
        / ``ref_den`` moments (see
        ``increment.estimation.variance.ratio_moments``); it is not a new
        statistic, just a missing constructor for the same math.
        """
        if mean_den <= 0:
            _raise("power.baseline.mean_den", mean_den=mean_den)
        var = ratio_linearized_variance(mean_num, mean_den, var_num, var_den, cov_num_den)
        return cls(mean=mean_num / mean_den, var=var, cuped_rho=cuped_rho)


class _ProjectionMemo:
    """A cache that never takes part in equality: two baselines from the same
    pilot compare equal whether or not either has been planned yet."""

    __slots__ = ("entries",)

    def __init__(self) -> None:
        self.entries: dict[tuple[float, float], _QuantileProjection] = {}

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ProjectionMemo)

    __hash__ = None  # type: ignore[assignment]


class QuantileBaseline(Baseline):
    """A ``Baseline`` for quantile-metric planning.

    Build one with ``Analysis.planning_baseline(metric)``, which passes the
    declared metric and the pilot's control-arm values to
    ``from_control_values``. Its variance comes from the
    runtime's own order-statistic construction
    (``increment.estimation.quantile.log_quantile_se``) re-evaluated at
    each candidate sample size, not from a single statistical quantity a
    caller could otherwise be asked to supply.

    Records which metric it was built for (``metric_name``) alongside the
    quantile level (``quantile_q``); every solver echoes both on
    ``PowerResult`` so reusing one metric's baseline to plan a different
    metric is visible in the answer rather than silently mis-sizing a
    design. ``cuped_rho`` is refused (not merely defaulted and silently
    trusted): a quantile metric has no runtime CUPED route to credit.
    """

    model_config = ConfigDict(frozen=True)
    metric_name: str
    quantile_q: float = Field(gt=0.0, lt=1.0)
    cuped_rho: float = Field(default=0.0, frozen=True)
    # The pilot's own control-arm values, ascending -- the deterministic
    # SE projection's only input: not a fitted parametric quantity, not a
    # precomputed SE-at-n table.
    _pilot_sorted: tuple[float, ...] = PrivateAttr(default=())
    # The pilot's projection at each (quantile, alpha) it has been planned at.
    _projections: _ProjectionMemo = PrivateAttr(default_factory=lambda: _ProjectionMemo())

    @model_validator(mode="after")
    def _refuse_cuped(self) -> QuantileBaseline:
        if self.cuped_rho != 0.0:
            _raise("power.quantile_baseline.cuped_unsupported", cuped_rho=self.cuped_rho)
        return self

    @classmethod
    def from_control_values(cls, metric: QuantileMetric, values: np.ndarray) -> QuantileBaseline:
        """Plan the declared quantile ``metric`` from a pilot's control-arm
        per-unit values.

        ``mean`` is the pilot's sample quantile. ``var`` is the per-unit
        variance the runtime's standard error implies at the pilot's own
        size ``m`` and a reference alpha of 0.05, ``m * (mean * se)**2``
        (the delta method on ``log Q``). The solvers never read ``var``:
        they re-evaluate the order-statistic construction at each
        candidate sample size and the procedure's own alpha.
        """
        if not isinstance(metric, QuantileMetric):
            _raise(
                "power.quantile_baseline.metric_not_quantile",
                metric=getattr(metric, "name", None),
                metric_type=getattr(metric, "type", type(metric).__name__),
            )
        raw = np.asarray(values, dtype=float)
        if raw.ndim != 1:
            _quantile_raise("estimation.quantile.values_one_dimensional", shape=raw.shape)
        if not np.isfinite(raw).all():
            _quantile_raise(
                "estimation.quantile.values_contain_non", n_bad=int((~np.isfinite(raw)).sum())
            )
        pilot = np.sort(raw)
        projection = _QuantileProjection.of(pilot, metric.quantile, 0.05)
        obj = cls(
            metric_name=metric.name,
            quantile_q=metric.quantile,
            mean=projection.arm.point,
            var=pilot.size * (projection.arm.point * projection.se(pilot.size)) ** 2,
        )
        # ty: ignore[invalid-assignment] -- pydantic PrivateAttr; frozen applies to fields, not this
        obj._pilot_sorted = tuple(float(v) for v in pilot)
        return obj

    def _projection(self, alpha: float) -> _QuantileProjection:
        key = (self.quantile_q, alpha)
        projection = self._projections.entries.get(key)
        if projection is None:
            projection = _QuantileProjection.of(
                np.asarray(self._pilot_sorted), self.quantile_q, alpha
            )
            self._projections.entries[key] = projection
        return projection


class PowerResult(CodedModel, BaseModel):
    """Result of a power-analysis solver.

    Parameters
    ----------
    n_per_arm : int
        Number of units in the treatment arm.
    n_total : int
        Total N across both arms.
    power : float
        Planned power at the computed / given sample size, under the model
        named by ``power_basis``. For ``required_sample_size`` and
        ``achieved_power`` it describes the SUPPLIED effect; for
        ``minimum_detectable_effect`` it describes the returned effect's
        implied absolute alternative, which can exceed the target when the
        answer is a domain endpoint.
    power_basis : {"asymptotic", "exact", "approximate"}
        How ``power`` was computed. ``"asymptotic"``: the log-ratio
        Normal/noncentral-t planning model (every plan the runtime does not
        decide with the exact binomial risk-ratio test). ``"exact"``: the
        probability that the runtime's unchanged exact binomial decision
        rejects, at the analyzed integer counts (up to at most about ``1e-12``
        of omitted outer count mass, and the numerical error of its sum, about
        ``1e-12`` of it at 1,000 units per arm and ``4e-7`` at a billion).
        ``"approximate"``: the same decision replayed with Normal conditional
        tails, for binomial plans whose exact geometry exceeds the planning
        cell budget.
    mde_relative : float | None
        Minimum detectable relative effect on the complier scale, expressed
        RELATIVE TO the declared null: ``(exp(distance) - 1) /
        baseline.compliance``, where ``distance`` is the search's own
        log-scale gap from ``theta0 = log1p(decision.null_lift)``. At
        ``compliance=1.0`` this is exactly ``exp(distance) - 1``; a lower
        compliance rescales it up. To recover the implied ABSOLUTE
        alternative, compose against the null rather than adding lifts:
        ``expm1(log1p(decision.null_lift) + log1p(mde_relative *
        baseline.compliance)) / baseline.compliance``. Positive for two-sided and one-sided
        "greater"; negative for one-sided "less". ``None`` when no minimum
        detectable effect exists at the design's target power -- a valid
        supplied-effect answer can still be reported then -- with the cause
        in ``mde_unavailable_reason``.
    mde_unavailable_reason : {"unattainable", "unrepresentable", "numerical_resolution"} | None
        Why ``mde_relative`` is ``None``: the target power exceeds every
        admissible alternative's power (``unattainable``), the admissible
        answer has no float64 representation (``unrepresentable``), or the
        search could not resolve it within its numerical limits
        (``numerical_resolution``). ``None`` whenever ``mde_relative`` is
        available; a missing effect always carries a reason.
    effective_var : float
        Per-unit variance for asymptotic planning, after the effective decision
        method's CUPED / factor-absorption reduction and cluster design effect.
        Sensitivity-only CUPED receives no reduction, so this may differ from
        the caller's ``Baseline.effective_var``. Exact and approximate binomial
        plans report it as metadata; their power uses event rates and counts.
        For a ``QuantileBaseline``, the per-unit variance its pilot's standard
        error implies (see ``QuantileBaseline.from_control_values``).
    n_clusters_per_arm : int | None
        Randomization clusters needed in the treatment arm,
        ``ceil(n_per_arm / baseline.avg_cluster_size)``. ``None`` under
        unit randomization.
    n_clusters_total : int | None
        Randomization clusters across both arms, ceiled per arm and
        summed (not ceiled once on ``n_total``, since a part-cluster in
        each arm costs two whole clusters). ``None`` under unit
        randomization.
    n_triggered_per_arm, n_triggered_total : int | None
        Units actually entering a triggered analysis
        (``n_per_arm``/``n_total`` times ``trigger_rate``). ``None`` when
        no ``trigger_rate`` was declared; ``n_per_arm``/``n_total``
        always count assigned units.
    expected_n_total : int | None
        Expected total N a sequential design stops at under the solved-for
        effect: ``E[T] * n_total``, where ``E[T]`` charges each look's
        first-boundary-exit mass its own information fraction and the
        never-crossing remainder the final planned look (see
        ``increment.power.sequential.sequential_expected_information_fraction``).
        This is the honest economic counterpart to ``n_total`` (the
        worst-case, never-stops-early size) -- early stopping is the
        entire argument for monitoring sequentially. ``None`` when no
        ``inference`` spec was given (a fixed-horizon design always runs
        to ``n_total``).
    inference_to_declare : InferenceSpec | None
        The exact runtime ``InferenceSpec`` this plan assumed -- pass it to
        ``InferenceSpec`` (or a YAML ``inference:`` block) at runtime
        declaration so the runtime's boundary is tuned from the same N
        planning assumed. ``None`` for a fixed-horizon result.
    planned_metric_name, planned_quantile : str | None, float | None
        The metric name and quantile level a ``QuantileBaseline`` was
        built for (``Analysis.planning_baseline(metric)``), echoed here so
        a caller who reuses one metric's baseline to plan a different
        metric sees the mismatch stated in the answer instead of
        discovering it, if at all, from a silently mis-sized design.
        ``None`` for every non-quantile metric.
    """

    model_config = ConfigDict(frozen=True)

    n_per_arm: int
    n_total: int
    power: float
    power_basis: PowerBasis
    mde_relative: float | None
    mde_unavailable_reason: MdeUnavailableReason | None = None
    effective_var: float
    n_clusters_per_arm: int | None = None  # None = unit randomization
    n_clusters_total: int | None = None
    n_triggered_per_arm: int | None = None  # None = no trigger rate declared
    n_triggered_total: int | None = None
    expected_n_total: int | None = None  # None = no inference spec (fixed-horizon)
    inference_to_declare: InferenceSpec | None = None
    planned_metric_name: str | None = None
    planned_quantile: float | None = None

    @property
    def planned_for(self) -> str | None:
        """``"planned for metric <name> at quantile <q>"`` when this
        result was sized for a quantile metric, else ``None``."""
        if self.planned_metric_name is None:
            return None
        name, q = self.planned_metric_name, self.planned_quantile
        return f"planned for metric {name!r} at quantile {q}"

    @model_validator(mode="after")
    def _check(self) -> PowerResult:
        """Closing invariant every solver's result satisfies regardless
        of path: a probability stays in ``[0, 1]``, an available relative
        effect stays above the shared ``-1.0`` floor and carries no
        unavailability reason, a missing one always does, and every
        numeric field is finite."""
        _require_finite("power", self.power)
        if not 0.0 <= self.power <= 1.0:
            _raise("power.power.check", power=self.power)
        _require_companion_mde(self.mde_relative, self.mde_unavailable_reason)
        _require_finite("effective_var", self.effective_var)
        if self.effective_var <= 0.0:
            _raise("power.power.effective_var", effective_var=self.effective_var)
        return self


# Internal helpers


def _expected_n_total(expected_t: float | None, n_total: int) -> int | None:
    """Round ``expected_t * n_total`` to an int, clamped to ``[0, n_total]``.

    ``expected_t`` is ``None`` for fixed-horizon designs (no ``inference``
    spec); floating-point error at ``expected_t`` near the boundary can
    round fractionally outside ``[0, n_total]``, which the clamp corrects.
    """
    if expected_t is None:
        return None
    return max(0, min(n_total, round(expected_t * n_total)))


def _planned_quantile_fields(baseline: Baseline) -> tuple[str | None, float | None]:
    """``(planned_metric_name, planned_quantile)``: the metric and
    quantile level a ``QuantileBaseline`` was built for, or ``(None,
    None)`` for every other baseline."""
    if isinstance(baseline, QuantileBaseline):
        return baseline.metric_name, baseline.quantile_q
    return None, None


def _se_sq(n_T: float, n_C: float, baseline: Baseline) -> float:
    """Baseline-only log-ratio variance ``v / m^2 * (1/n_T + 1/n_C)``.

    Retained for the segment-pairwise solvers, whose documented model
    evaluates every arm at its segment's baseline. The arm trio uses
    ``_arm_log_se_sq`` instead.
    """
    return baseline.effective_var / (baseline.mean**2) * (1.0 / n_T + 1.0 / n_C)


def _is_bounded_metric(procedure: ArmPlanningProcedure) -> bool:
    """Conversion and retention rates live in ``(0, 1]`` and carry a Bernoulli shape."""
    return procedure.metric.metric_type in ("conversion", "retention")


def _log1mexp(x: float) -> float:
    """``log(1 - exp(x))`` for ``x <= 0``; ``-inf`` exactly at ``x == 0``."""
    if x == 0.0:
        return -math.inf
    if x > 0.0:
        _raise("power.log_exp_needs", x=x)
    if x < -math.log(2.0):
        return math.log1p(-math.exp(x))
    return math.log(-math.expm1(x))


def _require_bracket(n: int, q: float, alpha: float) -> tuple[int, int]:
    """1-based ranks of the runtime's order-statistic bracket at size ``n``,
    refusing with the runtime's own code where none exists."""
    ranks = _bracket_ranks(n, q, alpha)
    if ranks is None:
        _quantile_raise(
            "estimation.quantile.too_small_bound",
            n=n,
            q=q,
            level=1.0 - alpha,
            n_min=_quantile_n_min(q, alpha),
        )
    return ranks


# Shift standard deviations spanned by the exact integration below: the
# normal mass beyond eight is under 1.3e-15, folded into the end pieces.
_SHIFT_SPAN = 8.0


@dataclass(frozen=True, slots=True)
class _QuantileProjection:
    """The runtime's quantile SE on a pilot of size ``m``, projected onto
    another sample size ``n`` and evaluated by the runtime's own rule,
    ``increment.estimation.quantile.quantile_half_width``.

    Where the pilot's bracket holds no tie, its log distances from the point
    shrink by ``s = sqrt(m / n)``, the asymptotic quantile-SE rate. Where it
    holds one, the data are recorded on a grid and the runtime's point and
    bracket ends fall on recorded values, so they are read from the pilot's
    recorded distribution at the bracket's rank probabilities ``a / n`` and
    ``b / n``. A sample of size ``n`` sharing ``min(m, n)`` units with the
    pilot sees that distribution through an ECDF differing from the pilot's
    near ``q`` by a normal shift of variance ``q (1 - q) |1/m - 1/n|``, so
    the half-width is averaged over that shift, which moves the point and
    both ends together: where the quantile falls within its recording cell
    is known only to the pilot's own precision. The shared units also bound
    the shift: a subsample's ``a``-th smallest value is at least the pilot's
    ``a``-th, and a sample extending the pilot has its ``a``-th smallest at
    most the pilot's ``a``-th, with the mirror bounds at ``b``.

    The integrand is constant between the shifts at which one of the three
    reads crosses a change of recorded value, so the average is the sum of
    each piece's half-width times its normal mass, exactly.

    The pilot's tie flag is kept, and the count of repeated values the 95%
    bracket spans shrinks with the bracket (``1 + (r - 1) s``). Beyond the
    pilot's size the point settles on a recorded value, so the floor and
    cell move from the pilot's own values by ``1 - m / n`` toward ``limit``:
    every value repeats, and the floor is one recording step from the point.
    At ``n == m`` the shift is zero and this is the runtime's SE on the
    pilot, bit for bit.
    """

    pilot: np.ndarray  # ascending
    alpha: float
    z: float
    arm: QuantileArm  # the pilot's own facts
    limit: QuantileArm  # the facts as n grows without bound
    lower: float  # Y_(a)
    upper: float  # Y_(b)
    tied: bool
    # Probabilities at which the pilot's recorded value changes, reading
    # rank ``k`` for probabilities in ``[(k - 1/2) / m, (k + 1/2) / m)``.
    edges: np.ndarray

    @classmethod
    def of(cls, pilot_sorted: np.ndarray, q: float, alpha: float) -> _QuantileProjection:
        a, b = _require_bracket(pilot_sorted.size, q, alpha)
        arm = QuantileArm.of(pilot_sorted, q)
        lower, upper = float(pilot_sorted[a - 1]), float(pilot_sorted[b - 1])
        if arm.point <= 0.0 or lower <= 0.0:
            _quantile_raise(
                "estimation.quantile.quantile_positive_log", q=q, point=arm.point, lo=lower
            )
        bracket = pilot_sorted[a - 1 : b]
        # With every value recorded twice, the runtime's cell is the full step.
        step = QuantileArm.of(np.repeat(pilot_sorted, 2), q).cell
        limit = replace(
            arm,
            floor=math.log1p(step / arm.point),
            cell=step,
            repeated=min(arm.repeated, 1),
        )
        changes = np.flatnonzero(pilot_sorted[1:] != pilot_sorted[:-1]) + 1
        return cls(
            pilot=pilot_sorted,
            alpha=alpha,
            z=float(_norm.isf(alpha / 2.0)),
            arm=arm,
            limit=limit,
            lower=lower,
            upper=upper,
            tied=bool(np.any(bracket[1:] == bracket[:-1])),
            edges=(changes + 0.5) / pilot_sorted.size,
        )

    def arm_at(self, n: int) -> QuantileArm:
        """The runtime's arm facts projected to size ``n``."""
        m, r = self.pilot.size, self.arm.repeated
        repeated = 0 if r == 0 else math.floor(1.0 + (r - 1) * math.sqrt(m / n))
        settle = max(0.0, 1.0 - m / n)
        return replace(
            self.arm,
            floor=self.arm.floor + (self.limit.floor - self.arm.floor) * settle,
            cell=self.arm.cell + (self.limit.cell - self.arm.cell) * settle,
            repeated=repeated,
        )

    def _half_width(self, arm: QuantileArm, lower: float, upper: float) -> float:
        """The runtime's rule before its positivity check: zero where the
        bracket has collapsed onto one value; refuses a non-positive end."""
        if lower <= 0.0:
            _quantile_raise(
                "estimation.quantile.quantile_positive_log", q=arm.q, point=arm.point, lo=lower
            )
        return _bracket_half_width(arm, lower, upper, tied=self.tied)

    def _recorded(self, probability: float) -> float:
        """The pilot's recorded value at ``probability``: its order statistic
        of rank ``round(m * probability)``, clipped to the sample."""
        m = self.pilot.size
        return float(self.pilot[min(max(math.floor(m * probability + 0.5), 1), m) - 1])

    def _cuts(self, offset: float, lo: float, hi: float) -> np.ndarray:
        """Shifts strictly inside ``(lo, hi)`` at which the read at
        ``offset + shift`` crosses a change of recorded value."""
        start, stop = np.searchsorted(self.edges, (lo + offset, hi + offset), side="right")
        cuts = self.edges[start:stop] - offset
        return cuts[cuts < hi]

    def _shifted_half_width(
        self,
        arm: QuantileArm,
        lower_p: float,
        upper_p: float,
        shift_sd: float,
        shift_lo: float,
        shift_hi: float,
    ) -> float:
        """The runtime's half-width on the pilot's recorded distribution with
        the bracket at probabilities ``(lower_p, upper_p)``, averaged exactly
        over a normal shift of the ECDF near ``q`` with standard deviation
        ``shift_sd`` and support clipped to ``[shift_lo, shift_hi]``."""
        lo = max(shift_lo, -_SHIFT_SPAN * shift_sd)
        hi = min(shift_hi, _SHIFT_SPAN * shift_sd)
        cuts = np.unique(np.concatenate([self._cuts(p, lo, hi) for p in (lower_p, upper_p, arm.q)]))
        # Piece j lies between cuts j-1 and j; the end pieces take the tails.
        below = np.concatenate(([-math.inf], cuts)) / shift_sd
        above = np.concatenate((cuts, [math.inf])) / shift_sd
        mass = np.where(below >= 0.0, _ndtr(-below) - _ndtr(-above), _ndtr(above) - _ndtr(below))
        mids = (np.concatenate(([lo], cuts)) + np.concatenate((cuts, [hi]))) / 2.0
        total = 0.0
        for weight, shift in zip(mass.tolist(), mids.tolist(), strict=True):
            total += weight * self._half_width(
                replace(arm, point=self._recorded(arm.q + shift)),
                self._recorded(lower_p + shift),
                self._recorded(upper_p + shift),
            )
        return total

    def _positive(self, half: float, q: float) -> float:
        if half <= 0.0:
            _quantile_raise("estimation.quantile.degenerate_spread_order", q=q)
        return half

    def shift(self, n: int) -> tuple[float, float, float, float, float]:
        """``(lower_p, upper_p, shift_sd, shift_lo, shift_hi)`` of the average
        at a tied size ``n != m``: the bracket's rank probabilities, the
        shift's standard deviation and its support from the shared units."""
        a, b = _require_bracket(n, self.arm.q, self.alpha)
        m, q = self.pilot.size, self.arm.q
        shift_sd = math.sqrt(q * (1.0 - q) * abs(1.0 / m - 1.0 / n))
        if n < m:
            shift_lo, shift_hi = (a - 0.5) / m - a / n, (m - n + b + 0.5) / m - b / n
        else:
            shift_lo, shift_hi = (b - n + m - 0.5) / m - b / n, (a + 0.5) / m - a / n
        return a / n, b / n, shift_sd, shift_lo, shift_hi

    def se(self, n: int) -> float:
        """The runtime's log-scale quantile SE projected to size ``n``."""
        _require_bracket(n, self.arm.q, self.alpha)
        m, q = self.pilot.size, self.arm.q
        arm = self.arm_at(n)
        if not self.tied:
            keep = 1.0 - math.sqrt(m / n)
            log_point = math.log(self.arm.point)
            lower = self.lower * math.exp(keep * (log_point - math.log(self.lower)))
            upper = self.upper * math.exp(-keep * (math.log(self.upper) - log_point))
            return self._positive(self._half_width(arm, lower, upper), q) / self.z
        if n == m:
            return quantile_half_width(arm, self.lower, self.upper, tied=True) / self.z
        half = self._shifted_half_width(arm, *self.shift(n))
        return self._positive(half, q) / self.z

    @property
    def se_floor(self) -> float:
        """The limit of ``se(n)`` as ``n`` grows: zero for an untied bracket,
        else the averaged half-width of a bracket closed onto ``q``."""
        if not self.tied:
            return 0.0
        q = self.arm.q
        shift_sd = math.sqrt(q * (1.0 - q) / self.pilot.size)
        half = self._shifted_half_width(self.limit, q, q, shift_sd, -math.inf, math.inf)
        return self._positive(half, q) / self.z


def _deterministic_quantile_se(
    pilot_sorted: np.ndarray, q: float, alpha: float, n_candidate: int
) -> float:
    """The runtime's quantile SE (``increment.estimation.quantile.
    log_quantile_se``) projected from a pilot onto ``n_candidate``; see
    ``_QuantileProjection``. Refuses with the runtime's own codes where the
    runtime would refuse: no bracket at either size, a non-positive
    quantile, or a collapsed spread."""
    return _QuantileProjection.of(pilot_sorted, q, alpha).se(n_candidate)


def _arm_log_se_sq(
    n_T: float, n_C: float, baseline: Baseline, *, theta: float, bounded: bool, alpha: float = 0.05
) -> float:
    """``log S^2(theta)``: the log-ratio planning variance with the treatment
    arm evaluated at its own mean ``m1 = m0 * exp(theta)``.

    ``theta`` is the absolute effective log ratio ``log1p(relative_lift *
    compliance)``, not a distance from a shifted null. ``n_T``/``n_C`` are
    ANALYZED counts (or allocation fractions for a per-unit coefficient).
    The two arm terms are formed in log space and combined with
    log-add-exp so a huge or tiny mean never squares into overflow first.
    ``bounded`` selects the Bernoulli shape ``v1 = v * m1 (1 - m1) / (m0 (1 -
    m0))``, whose treatment term vanishes (``-inf``) at a rate of exactly
    one; the control term keeps the total positive. Callers preflight a
    bounded baseline to ``0 < m0 < 1`` and every rate to ``<= 1``.

    A ``QuantileBaseline`` takes a separate branch: the variance is not a
    fixed per-unit quantity but a function of the arm's own analyzed
    count, the runtime's own construction on the pilot projected to that
    count (``_QuantileProjection``) at ``alpha``, the procedure's own
    compiled alpha, which the runtime's interval also tracks.
    Under this module's multiplicative-effect convention (``Q_t = Q_c *
    exp(theta)``), the treatment arm's log-scale quantile SE has the SAME
    asymptotic shape as control's own construction at its own n -- an
    exact algebraic identity, not the default "same absolute variance"
    assumption the ``else`` branch below applies -- so both arms reuse
    the SAME pilot values and recording grid (a property of the
    instrument, not of the hypothesized shift; ``n_T`` alone changes what
    that instrument's construction resolves to for treatment).
    """
    if isinstance(baseline, QuantileBaseline):
        projection = baseline._projection(alpha)
        se_c = projection.se(int(round(n_C)))
        se_t = projection.se(int(round(n_T)))
        return float(np.logaddexp(2.0 * math.log(se_t), 2.0 * math.log(se_c)))
    log_v = math.log(baseline.effective_var)
    log_m0 = math.log(baseline.mean)
    per_unit = log_v - 2.0 * log_m0
    control = per_unit - math.log(n_C)
    if bounded:
        treatment = (
            per_unit
            - math.log(n_T)
            - math.log1p(-baseline.mean)
            - theta
            + _log1mexp(log_m0 + theta)
        )
    else:
        treatment = per_unit - math.log(n_T) - 2.0 * theta
    return float(np.logaddexp(treatment, control))


def _distance_from_null(effective_lift: float, null_lift: float) -> float:
    """Signed log-scale distance ``log1p(effective_lift) - log1p(null_lift)``.

    Formed as ``log1p((effective - null) / (1 + null))`` so two adjacent
    representable lifts keep their distance instead of cancelling in a
    difference of two nearly equal logarithms; the difference form is used
    only when that ratio overflows or rounds onto the log domain boundary.
    """
    ratio = (effective_lift - null_lift) / (1.0 + null_lift)
    if math.isfinite(ratio) and ratio > -1.0:
        return math.log1p(ratio)
    return math.log1p(effective_lift) - math.log1p(null_lift)


_BOUNDED_METRIC_BASELINE = RefusalSpec(
    "power.bounded_metric.baseline",
    InvalidRequestError,
    template="{metric_type} planning needs a baseline rate strictly inside (0, 1), got mean={mean!r}: a bounded metric's control mean is a probability, and an alternative rate p0 * (1 + lift) has no room to move from a rate at or above one",
)
_BOUNDED_METRIC_RATE_ABOVE_ONE = RefusalSpec(
    "power.bounded_metric.rate_above_one",
    InvalidRequestError,
    template="{metric_type} planning at baseline rate {mean:.6g}: the {role} lift implies a rate of {implied_rate:.6g}, above one -- an alternative that cannot occur is not a conservative approximation; lower the lift or the baseline rate",
)


def _bounded_preflight(
    procedure: ArmPlanningProcedure, baseline: Baseline, *, null_lift: float
) -> bool:
    """Whether the Bernoulli variance shape applies, refusing a bounded
    baseline outside ``(0, 1)`` or a null rate above one before any
    calculation. Invalid bounded inputs never fall through to the
    unbounded model."""
    if not _is_bounded_metric(procedure):
        return False
    if not 0.0 < baseline.mean < 1.0:
        refuse(
            _BOUNDED_METRIC_BASELINE,
            metric_type=procedure.metric.metric_type,
            mean=baseline.mean,
        )
    _require_bounded_rate(procedure, baseline, theta=math.log1p(null_lift), role="null")
    return True


def _require_bounded_rate(
    procedure: ArmPlanningProcedure, baseline: Baseline, *, theta: float, role: str, **lifts: float
) -> None:
    """Refuse a bounded rate ``m0 * exp(theta)`` above one, judged in log space."""
    log_rate = math.log(baseline.mean) + theta
    if log_rate > 0.0:
        refuse(
            _BOUNDED_METRIC_RATE_ABOVE_ONE,
            metric_type=procedure.metric.metric_type,
            mean=baseline.mean,
            role=role,
            implied_rate=math.exp(log_rate),
            **lifts,
        )


def _cluster_counts(n_T: int, n_C: int, baseline: Baseline) -> tuple[int | None, int | None]:
    """Required randomized clusters from assigned units and assigned mean size."""
    m = baseline.avg_cluster_size
    if m <= 1.0:
        return None, None
    k_t = math.ceil(n_T / m)
    return k_t, k_t + math.ceil(n_C / m)


_CLUSTER_TOO_FEW = RefusalSpec(
    "power.cluster.too_few_clusters",
    InvalidRequestError,
    template="cluster-randomized fixed-horizon power requires at least two represented clusters per arm (got k_t={k_t}, k_c={k_c}, {k_total} total)",
    keys=frozenset({"recruited_k_total", "route"}),
)


def _represented_clusters(k_recruited: int, baseline: Baseline) -> int:
    """Floor represented clusters, allowing for rounded pilot ratios."""
    p = baseline.cluster_participation
    if p is None:
        return k_recruited
    recruited = float(k_recruited)
    product = recruited * p
    nearest = round(product)
    # The source ratio, integer conversion, and product each round.
    # Propagating one ULP per input plus one product ULP bounds those
    # errors, including their smaller cross term, on either side.
    tolerance = math.ulp(product) + recruited * math.ulp(p) + p * math.ulp(recruited)
    return nearest if abs(product - nearest) <= tolerance else math.floor(product)


def _cluster_dof(
    n_T: int, n_C: int, baseline: Baseline, *, enforce_floor: bool = True
) -> float | None:
    """Return conservative ``min(Krepresented_t - 1, Krepresented_c - 1)``."""
    k_t, k_total = _cluster_counts(n_T, n_C, baseline)
    if k_total is None:
        return None
    assert k_t is not None
    k_c = k_total - k_t
    represented_t = _represented_clusters(k_t, baseline)
    represented_c = _represented_clusters(k_c, baseline)
    if enforce_floor and (represented_t < 2 or represented_c < 2):
        refuse(
            _CLUSTER_TOO_FEW,
            k_total=represented_t + represented_c,
            k_t=represented_t,
            k_c=represented_c,
            recruited_k_total=k_total,
            route="recruit enough clusters for at least two contributing clusters per arm "
            "at the pilot participation rate",
        )
    return float(min(represented_t - 1, represented_c - 1))


def _compute_arms(n_T: int, design: PowerDesign, *, minimum_per_arm: int = 2) -> tuple[int, int]:
    """Return (n_T, n_C) given the treatment-arm size.

    Control-arm size is derived from the allocation ratio so that
    n_T : n_C approx allocation : (1 - allocation).
    """
    if n_T < minimum_per_arm:
        n_T = minimum_per_arm
    n_C = max(minimum_per_arm, math.ceil(n_T * (1.0 - design.allocation) / design.allocation))
    return n_T, n_C


def _power_from_nc(
    nc: float,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    """Power at noncentrality ``nc`` (signed toward the alternative).

    Parameterizing on ``nc`` directly avoids reconstructing an absolute
    ``theta`` around a shifted null and subtracting it back off: when the null
    is large relative to the standard error, that round trip rounds the
    increment away and turns a root-find objective into a step function.
    An infinite ``nc`` (a variance that underflowed) is the exact limit:
    certain detection on its side, none on the other. The normal reference
    uses the survival ufuncs behind ``scipy.stats.norm`` directly: the
    minimum-detectable-effect search evaluates this dozens of times. A tail
    the kernel cannot resolve is refused, never reported as zero power.
    """
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    tail_alpha = procedure.compiled_tail_alpha
    power = _scalar_power_from_nc(
        nc, alternative=decision.alternative, tail_alpha=tail_alpha, dof=dof
    )
    if math.isnan(power):
        _raise("power.noncentral_t_unresolved", nc=nc, dof=dof, tail_alpha=tail_alpha)
    return power


def _power_at(
    theta: float,
    se2: float,
    design: PowerDesign,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    theta0 = math.log1p(decision.null_lift)
    nc = (theta - theta0) / math.sqrt(se2)
    return _power_from_nc(nc, procedure, dof=dof)


def _derive_axes_from_baseline(
    procedure: ArmPlanningProcedure, baseline: Baseline
) -> ArmPlanningProcedure:
    """Auto-derive the quantile metric type and CUPED/compliance/
    triggering/absorption from the Baseline the solver call already
    received, wherever the procedure still carries standard()'s
    undeclared default for that axis. A procedure whose axis was
    explicitly set to something else is left untouched here; a genuine
    conflict with the baseline is then reported by arm_planning_support's
    baseline-compatibility check (or, for a ``QuantileBaseline``, by
    ``_prepare_solver``), which fires only on a real mismatch."""
    if isinstance(baseline, QuantileBaseline) and procedure.metric.metric_type == "mean":
        procedure = procedure.model_copy(
            update={"metric": procedure.metric.model_copy(update={"metric_type": "quantile"})}
        )
    analysis = procedure.analysis
    analysis_updates: dict[str, str] = {}
    if baseline.compliance != 1.0 and analysis.identification == "randomized":
        analysis_updates["identification"] = "encouragement"
    if baseline.trigger_rate != 1.0 and analysis.population == "assigned":
        analysis_updates["population"] = "triggered"
    if baseline.icc != 0.0 and analysis.variance_adjustment == "none":
        analysis_updates["variance_adjustment"] = "factor_absorption"
    if analysis_updates:
        procedure = procedure.model_copy(
            update={"analysis": analysis.model_copy(update=analysis_updates)}
        )
    already_cuped = any(
        method.variance_reduction != "none"
        for method in (procedure.decision_method, *procedure.sensitivity_methods)
    )
    if baseline.cuped_rho != 0.0 and not already_cuped:
        procedure = procedure.model_copy(
            update={"decision_method": MethodSpec(name="cuped", variance_reduction="cuped")}
        )
    return procedure


def _prepare_solver(
    procedure: ArmPlanningProcedure,
    baseline: Baseline,
) -> tuple[ArmPlanningProcedure, Baseline]:
    """Revalidate planning policy, derive undeclared axes from the
    baseline, and refuse unsupported baseline combinations."""
    if not isinstance(procedure, ArmPlanningProcedure):
        _raise("power.procedure_armplanningprocedure", procedure_type=type(procedure).__name__)
    procedure = ArmPlanningProcedure.model_validate(procedure)
    if not isinstance(procedure.decision, RelativeDecisionPolicy):
        _raise("power.power_solvers_relative")
    baseline = Baseline.model_validate(baseline)
    procedure = _derive_axes_from_baseline(procedure, baseline)
    # Sensitivity-only CUPED must not subsidize an unadjusted decision.
    if procedure.decision_method.variance_reduction == "none" and baseline.cuped_rho != 0.0:
        baseline = baseline.model_copy(update={"cuped_rho": 0.0})
    support = arm_planning_support(procedure, baseline=baseline)
    if isinstance(support, Unsupported):
        refuse_unsupported(support)
    if procedure.metric.metric_type == "quantile":
        _refuse_unanalyzable_quantile_plan(procedure, baseline)
    elif isinstance(baseline, QuantileBaseline):
        _raise(
            "power.quantile_baseline.metric_not_quantile",
            metric=baseline.metric_name,
            metric_type=procedure.metric.metric_type,
        )
    return procedure, baseline


def _refuse_unanalyzable_quantile_plan(procedure: ArmPlanningProcedure, baseline: Baseline) -> None:
    """The quantile readout tests two-sided against a zero null only, and
    refuses anything else with this same code; planning refuses it too."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    if decision.alternative != "two-sided" or decision.null_lift != 0.0:
        refuse(
            READOUT_REFUSALS["readout.metric.quantile_alternative"],
            metric=baseline.metric_name if isinstance(baseline, QuantileBaseline) else None,
            alternative=decision.alternative,
            null_lift=decision.null_lift,
            route=(
                "plan the two-sided test against a zero null the quantile readout runs: "
                "ArmPlanningProcedure.standard('quantile') with no one-sided alternative, "
                "null_lift or guardrail role"
            ),
        )


def _procedure_inference(procedure: ArmPlanningProcedure) -> SequentialPlanningSpec | None:
    """Return the declared planning approximation, or the fixed-horizon sentinel."""
    inference = procedure.inference
    if isinstance(inference, FixedInference):
        return None
    return _require_prospective(inference)


def _planned_looks(
    procedure: ArmPlanningProcedure, planned_looks: int | None
) -> _PlannedLooks | None:
    """Resolve the procedure's look schedule once; a fixed-horizon procedure
    plans no looks but still rejects an invalid supplied count."""
    inference = _procedure_inference(procedure)
    if inference is None:
        if planned_looks is not None:
            _validate_planned_looks(planned_looks)
        return None
    return _resolve_looks(inference, planned_looks)


def _minimum_per_arm(procedure: ArmPlanningProcedure) -> int:
    return 20 if procedure.metric.metric_type == "quantile" else 2


def _assigned_minimum_per_arm(procedure: ArmPlanningProcedure, baseline: Baseline) -> int:
    return math.ceil(_minimum_per_arm(procedure) / baseline.trigger_rate)


def _validate_per_arm_floor(
    n_per_arm: int, procedure: ArmPlanningProcedure, baseline: Baseline
) -> None:
    if isinstance(n_per_arm, bool) or not isinstance(n_per_arm, int):
        _raise("power.core.n_per_arm_int", n_per_arm=n_per_arm)
    minimum = _assigned_minimum_per_arm(procedure, baseline)
    if n_per_arm < minimum:
        _raise(
            "power.core.n_per_arm_min",
            minimum=minimum,
            n_per_arm=n_per_arm,
            metric_type=procedure.metric.metric_type,
        )


def _quantile_search_floor(procedure: ArmPlanningProcedure, baseline: QuantileBaseline) -> int:
    """The search's lower probing bound, clamped to the closed-form
    bracket-existence floor: a search seeded low
    by a large MDE must not refuse merely because its first probed n
    sits below the bracket's own feasibility floor when a larger, still
    reachable n is feasible. Reuses ``log_quantile_se``'s own formula,
    not a reimplementation, so the two floors can never disagree."""
    return max(
        _minimum_per_arm(procedure), _quantile_n_min(baseline.quantile_q, procedure.compiled_alpha)
    )


# Grid ratio of the quantile size search: four steps per doubling, finer than
# the projection's variation between the sizes it probes directly.
_QUANTILE_GRID_STEPS_PER_DOUBLING = 4


def _quantile_size_search(
    procedure: ArmPlanningProcedure,
    baseline: QuantileBaseline,
    design: PowerDesign,
    *,
    theta: float,
    distance: float,
    bounded: bool,
) -> tuple[int, int]:
    """The smallest treatment-arm size reaching target power for a quantile
    metric. Its per-arm variance (the pilot's projection) is a function of
    the candidate n itself, so it cannot be solved in the single closed-form
    step a fixed-variance baseline uses -- the search re-evaluates it at
    each candidate n it probes.

    Power is not monotone in n: it passes through the pilot's own power at
    the pilot's size, drops where an arm's bracket stops spanning enough
    repeated values to be resolved (the runtime switches to its wider tied
    interval), and on a coarse grid wavers with the runtime's own bracket
    ranks. The search therefore evaluates a grid of sizes in increasing
    order -- the bracket-existence floor, the pilot's size, the last size
    each arm stays resolved at and the one after, and a geometric ladder of
    ``_QUANTILE_GRID_STEPS_PER_DOUBLING`` steps per doubling up to
    ``_MAX_QUANTILE_ARM`` -- and bisects between the first grid size reaching
    the target and the one before it, so the answer's predecessor falls
    short of the target and no coarser feature hides a smaller size; only a
    rank-level waver within one grid step can.

    A size at which the projected bracket reaches a non-positive value has
    no log-scale interval, as at the runtime; the search passes over it. On
    tied data the projected SE falls to a floor fixed by the recording step
    (``_QuantileProjection.se_floor``), so power has a ceiling below one; a
    target never reached is refused naming the largest power found on the
    grid or in that limit, or the count limit when the target lies below it.
    """
    floor = _quantile_search_floor(procedure, baseline)
    q, alpha = baseline.quantile_q, procedure.compiled_alpha
    projection = baseline._projection(alpha)
    unanswerable: list[InvalidRequestError] = []
    # Sizes per turn of the slower arm's bracket-rank rounding: an arm's ranks
    # advance by about q times its share of a treatment unit per size.
    share = min(1.0, (1.0 - design.allocation) / design.allocation)
    turn = math.ceil(1.0 / (min(q, 1.0 - q) * share))

    def _arms(n_t: int) -> tuple[int, int]:
        return _compute_arms(n_t, design, minimum_per_arm=floor)

    def _reaches(n_t: int) -> bool:
        power = _power_at_n(n_t)
        return power is not None and power >= design.power

    def _power_at_n(n_t: int) -> float | None:
        n_t2, n_c2 = _arms(n_t)
        plan = _ArmPlan(procedure, baseline, n_t2, n_c2, None, bounded)
        try:
            return plan.power(distance, theta)
        except InvalidRequestError as refusal:
            # The runtime's two "no log-scale interval at this size" refusals.
            if refusal.code not in (
                "estimation.quantile.quantile_positive_log",
                "estimation.quantile.degenerate_spread_order",
            ):
                raise
            unanswerable.append(refusal)
            return None

    def _first_reaching(lo: int, hi: int) -> tuple[int, int]:
        lo = bisect_first_true(lo, hi, _reaches)
        # Below a crossing, power wavers with the runtime's bracket ranks:
        # walk down until a whole turn of sizes past the last reaching one falls short.
        first = n_t = lo
        misses = 0
        while n_t > floor and misses < turn:
            n_t -= 1
            if _reaches(n_t):
                first, misses = n_t, 0
            else:
                misses += 1
        return _arms(first)

    def _last_resolved(arm: int) -> int | None:
        def resolved(n_t: int) -> bool:
            return projection.arm_at(_arms(n_t)[arm]).resolved

        if not resolved(floor) or resolved(_MAX_QUANTILE_ARM):
            return None
        first_unresolved = bisect_first_true(floor, _MAX_QUANTILE_ARM, lambda n: not resolved(n))
        return first_unresolved - 1

    steps = _QUANTILE_GRID_STEPS_PER_DOUBLING
    doublings = math.ceil(steps * math.log2(_MAX_QUANTILE_ARM / floor))
    grid = {floor, projection.pilot.size, _MAX_QUANTILE_ARM}
    grid.update(math.ceil(floor * 2.0 ** (k / steps)) for k in range(doublings))
    for end in (_last_resolved(0), _last_resolved(1)):
        if end is not None:
            grid.update((end, end + 1))

    peak = 0.0
    previous = floor - 1
    answered = False
    for n_t in sorted(n for n in grid if floor <= n <= _MAX_QUANTILE_ARM):
        power = _power_at_n(n_t)
        if power is not None and power >= design.power:
            return _first_reaching(previous + 1, n_t)
        if power is not None:
            answered = True
            peak = max(peak, power)
        previous = n_t
    if not answered:
        # No size up to the count limit has a log-scale interval.
        raise unanswerable[-1]
    _refuse_unreached_quantile_power(projection.se_floor, peak, distance, procedure, design)


def _refuse_unreached_quantile_power(
    se_floor: float,
    peak: float,
    distance: float,
    procedure: ArmPlanningProcedure,
    design: PowerDesign,
) -> NoReturn:
    """Refuse a target no searched size reached, naming the largest power any
    size reaches: the grid's peak or the limit with both arms at the SE floor."""
    ceiling = (
        1.0
        if se_floor == 0.0
        else _power_from_nc(
            _noncentrality(distance, math.log(2.0) + 2.0 * math.log(se_floor)),
            procedure,
            dof=None,
        )
    )
    maximum_power = max(peak, ceiling)
    _raise(
        "power.quantile_size_search_unreachable",
        power=design.power,
        maximum_power=maximum_power,
        limiting_condition=(
            "recording_grid" if design.power >= maximum_power else "sample_size_limit"
        ),
    )


def _mde_theta(
    se2: float,
    design: PowerDesign,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    """Minimum detectable log-scale effect (distance from the null) at a
    FIXED variance ``se2``.

    This is the segment-pairwise model's inverse and, at ``se2=1.0``, the
    dimensionless fixed-reference quantile sum the arm solvers seed their
    searches with. It is not the arm trio's inverse: that variance moves
    with the alternative (see ``_solve_arm_mde``).

    A test's own null-boundary power (``2*tail_alpha`` two-sided,
    ``tail_alpha`` one-sided) already exceeds any target power at or
    below it: the zero-distance effect trivially clears that target, so
    no finite closed-form or root-found distance is meaningful there.
    That domain edge is checked once, before either solve path, instead
    of letting the closed form return a wrong-signed effect or the root
    finder fail for lack of a sign change at the endpoint.
    """
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    se = math.sqrt(se2)
    theta0 = math.log1p(decision.null_lift)
    null_power = _power_at(theta0, se2, design, procedure, dof=dof)
    if design.power <= null_power:
        return 0.0
    z_alpha = float(_norm.isf(procedure.compiled_tail_alpha))
    z_power = float(_norm.ppf(design.power))
    alternative = decision.alternative
    direction = 1.0 if alternative in ("two-sided", "greater") else -1.0
    if dof is None and alternative != "two-sided":
        return direction * se * (z_alpha + z_power)

    from scipy.optimize import brentq

    target = design.power

    def gap(nc_abs: float) -> float:
        return _power_from_nc(direction * nc_abs, procedure, dof=dof) - target

    lo, hi = 0.0, max(1.0, z_alpha + z_power)
    while gap(hi) < 0.0:
        hi *= 2.0
    return direction * brentq(gap, lo, hi, xtol=1e-12) * se


def _mde_relative(
    se2: float,
    design: PowerDesign,
    procedure: ArmPlanningProcedure,
    *,
    dof: float | None = None,
) -> float:
    """Minimum detectable relative effect at a fixed variance ``se2``
    (segment-pairwise model)."""
    return math.expm1(_mde_theta(se2, design, procedure, dof=dof))


def _validate_relative_lift(relative_lift: float) -> None:
    _require_relative_domain("relative_lift", relative_lift)


def _supplied_effect(
    procedure: ArmPlanningProcedure, baseline: Baseline, relative_lift: float, *, bounded: bool
) -> tuple[float, float]:
    """``(theta, distance)`` of a supplied complier-scale ``relative_lift``:
    its absolute effective log ratio ``log1p(relative_lift * compliance)``
    and its signed log-distance from the null. Validates the public domain
    and, for a bounded metric, refuses an implied rate above one. Sizing
    and achieved power share this conversion, so their supplied effect is
    the same alternative the companion solver measures distance from.
    """
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    _validate_relative_lift(relative_lift)
    effective_lift = relative_lift * baseline.compliance
    theta = math.log1p(effective_lift)
    if bounded:
        _require_bounded_rate(
            procedure,
            baseline,
            theta=theta,
            role="alternative",
            relative_lift=relative_lift,
            compliance=baseline.compliance,
        )
    return theta, _distance_from_null(effective_lift, decision.null_lift)


def _sequential_side(
    procedure: ArmPlanningProcedure,
) -> tuple[float, Literal["both", "upper"], bool]:
    """(alpha, exit side, e-value dual) of the sequential boundary the runtime
    builds for this procedure's registered cell. The mixture is tuned at the
    raw compiled alpha; its log term follows
    estimation.asymptotic_mean.boundary_alpha (see
    power.sequential._always_valid_z_bound). A registered family member is
    the dual of its sign-gated e-value, built at alpha on either side; any
    other one-sided cell spends one tail of the two-sided boundary at twice
    alpha. A planning procedure declares membership through its family axes:
    a metric-axis family is a secondary family (``standard(role="secondary")``)
    and a segment-axis family a registered breakout family, whose cells the
    runtime registers with ``family=True``; an arm-only family is a primary's
    split across its own arms, whose cells are not family members. Sequential
    planning refuses a prior, so a planned secondary is never the prior-bound
    secondary the runtime leaves outside its family."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    exit_side: Literal["both", "upper"] = "both" if decision.alternative == "two-sided" else "upper"
    member = not {"metric", "segment"}.isdisjoint(decision.family.axes)
    return procedure.compiled_alpha, exit_side, member


def _sequential_drift(procedure: ArmPlanningProcedure, distance: float) -> float:
    """Drift toward the decision side, measured from the null as at
    analysis: a ``less`` alternative's decrease exits through the upper
    boundary."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    return -distance if decision.alternative == "less" else distance


# Context: target_power, direction, limiting_condition, and maximum_power (the most detectable
# admissible alternative's power; None for "empty_admissible_direction"). Sequential designs
# report the largest certified upper bound instead ("sequential_exclusion").
_MDE_UNATTAINABLE = RefusalSpec(
    "power.minimum_detectable_effect.unattainable",
    InvalidRequestError,
    lambda *, target_power, maximum_power, direction, limiting_condition: (
        f"no minimum detectable effect reaches power={target_power} in the {direction} "
        "direction: "
        + (
            "no admissible alternative exists on that side of the null (empty_admissible_direction)"
            if maximum_power is None
            else (
                f"the most detectable admissible alternative reaches power "
                f"{maximum_power:.6g}, limited by {limiting_condition}"
            )
        )
        + " -- raise n_per_arm, lower the target power, or revisit the null and compliance"
    ),
)
_MDE_UNREPRESENTABLE = RefusalSpec(
    "power.minimum_detectable_effect.unrepresentable",
    InvalidRequestError,
    template="the minimum detectable effect at power={target_power} in the {direction} direction exists but has no float64 representation as a relative lift ({limiting_condition}) -- the standard error or compliance is too extreme for a relative-scale answer",
)


@dataclass(frozen=True, slots=True)
class _ArmPlan:
    """One arm trio's fixed-horizon planning model at ANALYZED counts."""

    procedure: ArmPlanningProcedure
    baseline: Baseline
    n_T: float
    n_C: float
    dof: float | None
    bounded: bool

    def log_se_sq(self, theta: float) -> float:
        return _arm_log_se_sq(
            self.n_T,
            self.n_C,
            self.baseline,
            theta=theta,
            bounded=self.bounded,
            alpha=self.procedure.compiled_alpha,
        )

    def power(self, distance: float, theta: float) -> float:
        """Power at the signed log-distance ``distance`` from the null of an
        alternative whose absolute effective log ratio is ``theta``."""
        nc = _noncentrality(distance, self.log_se_sq(theta))
        return _power_from_nc(nc, self.procedure, dof=self.dof)


def _first_ordinal_accepted(rejected: int, accepted: int, accept: Callable[[float], bool]) -> int:
    """Bisect float64 ordinals from a rejected float to an accepted one and
    return the accepted ordinal adjacent to a rejected one. ``accept`` takes
    the candidate float and must switch exactly once; at most 64 evaluations."""
    lo, hi = rejected, accepted
    while abs(hi - lo) > 1:
        mid = lo + (hi - lo) // 2
        if accept(float_from_ordinal(mid)):
            hi = mid
        else:
            lo = mid
    return hi


def _decreasing_noncentrality_peak(r: float, q: float, log_q: float) -> float:
    """Distance ``d > 0`` maximizing ``d / sqrt(A e^{r d} + B)`` for ``q = B / A``.

    The derivative changes sign exactly once, at ``A e^{r d} (2 - r d) + 2 B
    = 0``. For ``B >= 0`` the root lies at or beyond ``2 / r``: with ``u =
    r d - 2`` it solves ``u + log u = log(2 B / A) - 2``, handled in ``l =
    log u`` as ``e^l + l = rhs`` so a tiny ``u`` keeps its precision and a
    huge ``B / A`` enters only through ``log_q``. For ``B < 0`` (a Bernoulli
    control term below the treatment constant) the root lies in ``(1 / r,
    2 / r)`` and is bracketed there, away from the cancellation-prone
    origin.
    """
    from scipy.optimize import brentq

    if q < 0.0:
        return float(
            brentq(
                lambda d: math.exp(r * d) * (2.0 - r * d) + 2.0 * q, 1.0 / r, 2.0 / r, xtol=1e-14
            )
        )
    if q == 0.0:
        return 2.0 / r
    rhs = math.log(2.0) + log_q - 2.0
    if rhs >= 1.0:
        lo, hi = 0.0, math.log(rhs)
    else:
        lo, hi = rhs - 1.0, rhs
    if hi <= lo:
        log_u = lo
    else:
        log_u = float(brentq(lambda l: math.exp(l) + l - rhs, lo, hi, xtol=1e-14))
    return (math.exp(log_u) + 2.0) / r


@dataclass(frozen=True, slots=True)
class _MdeSearch:
    """Candidate space of one fixed-horizon minimum-detectable-effect solve.

    A candidate is a public float ``m`` (the emitted ``mde_relative``);
    its signed log-distance from the null is ``log1p(m * compliance)`` and
    its implied absolute alternative ``expm1(theta0 + distance) /
    compliance`` must itself be a representable relative lift above the
    ``-1`` floor and, for a bounded metric, a rate at most one. The
    admissible candidates form one interval along the direction of
    increasing distance; its endpoints are located by bisecting float64
    ordinals between a rejected and an accepted float, as is the first
    crossing of the target power.
    """

    plan: _ArmPlan
    target: float
    theta0: float
    sigma: float
    compliance: float
    log_m0: float
    # Absolute effective log ratios keep their implied complier lift above -1
    # only above this floor (exclusive); compliance 1 has none.
    theta_floor: float

    @classmethod
    def build(
        cls,
        plan: _ArmPlan,
        *,
        target: float,
        null_lift: float,
        alternative: Alternative,
    ) -> _MdeSearch:
        compliance = plan.baseline.compliance
        return cls(
            plan=plan,
            target=target,
            theta0=math.log1p(null_lift),
            sigma=-1.0 if alternative == "less" else 1.0,
            compliance=compliance,
            log_m0=math.log(plan.baseline.mean),
            theta_floor=math.log1p(-compliance) if compliance < 1.0 else -math.inf,
        )

    @property
    def direction(self) -> str:
        return "decreasing" if self.sigma < 0.0 else "increasing"

    def candidate(self, m: float) -> tuple[float, float] | None:
        """``(signed distance, theta)`` of public candidate ``m``, or ``None``
        when it or its implied absolute alternative leaves the public domain."""
        if not math.isfinite(m) or m <= -1.0:
            return None
        if (m < 0.0) if self.sigma > 0.0 else (m > 0.0):
            return None
        distance = math.log1p(m * self.compliance)
        theta = self.theta0 + distance
        if theta >= _LOG_FLOAT_MAX:
            return None
        implied = math.expm1(theta) / self.compliance
        if not math.isfinite(implied) or implied <= -1.0:
            return None
        if self.plan.bounded and self.log_m0 + theta > 0.0:
            return None
        return distance, theta

    def admissible(self, m: float) -> bool:
        return self.candidate(m) is not None

    def power_of(self, point: tuple[float, float]) -> float:
        distance, theta = point
        return self.plan.power(distance, theta)

    def reaches(self, m: float) -> bool:
        point = self.candidate(m)
        return point is not None and self.power_of(point) >= self.target

    def unattainable(self, maximum_power: float | None, limiting_condition: str) -> _MdeRefusal:
        return _MdeRefusal(
            "unattainable",
            _MDE_UNATTAINABLE,
            {
                "target_power": self.target,
                "maximum_power": maximum_power,
                "direction": self.direction,
                "limiting_condition": limiting_condition,
            },
        )

    def unrepresentable(self, limiting_condition: str) -> _MdeRefusal:
        return _MdeRefusal(
            "unrepresentable",
            _MDE_UNREPRESENTABLE,
            {
                "target_power": self.target,
                "direction": self.direction,
                "limiting_condition": limiting_condition,
            },
        )

    def lower_endpoint(self) -> float | _MdeRefusal:
        """First admissible candidate in increasing-distance order.

        Zero when the null itself is reachable. Under partial compliance a
        shifted null can sit below every reachable effective lift: the
        decreasing direction is then empty, and the increasing one opens
        where the implied complier lift first exceeds -1 and closes at the
        smaller of float64 range and the bounded rate ceiling. A null above
        that floor is rejected only when its own complier-scale lift
        overflows float64 (``overflowed_null``).
        """
        if self.candidate(0.0) is not None:
            return 0.0
        if self.theta0 > self.theta_floor:
            return self.overflowed_null()
        if self.sigma < 0.0:
            return self.unattainable(None, "empty_admissible_direction")
        d_floor = self.theta_floor - self.theta0
        d_ceiling = math.log1p(self.compliance * _FLOAT_MAX) - self.theta0
        if self.plan.bounded:
            d_ceiling = min(d_ceiling, -self.log_m0 - self.theta0)
        if d_ceiling <= d_floor:
            return self.unattainable(None, "empty_admissible_direction")
        # An admissible anchor inside the physical interval, approaching the
        # floor until one is representable.
        span = min(1.0, 0.5 * (d_ceiling - d_floor))
        for _ in range(64):
            inside = math.expm1(d_floor + span) / self.compliance
            if self.admissible(inside):
                return float_from_ordinal(
                    _first_ordinal_accepted(
                        float_ordinal(0.0), float_ordinal(inside), self.admissible
                    )
                )
            span *= 0.5
        return self.unrepresentable("float64_relative_lift")

    def overflowed_null(self) -> float | _MdeRefusal:
        """Lower endpoint when the null is physical but its complier-scale
        lift overflows float64. Every increase overflows too. A decrease is
        representable only past the band of small distances whose implied
        lift still overflows; that band is physical, so when its most
        detectable distance already reaches the target the answer lies
        inside it without a representation, and otherwise the first
        representable decrease opens the interval.
        """
        if self.sigma > 0.0:
            return self.unrepresentable("float64_relative_lift")
        edge = math.nextafter(-1.0, 0.0)
        if not self.admissible(edge):
            return self.unrepresentable("float64_relative_lift")
        m_min = float_from_ordinal(
            _first_ordinal_accepted(float_ordinal(0.0), float_ordinal(edge), self.admissible)
        )
        # The band ends at m_min's neighbour toward zero.
        d_band = -math.log1p(float_from_ordinal(float_ordinal(m_min) + 1) * self.compliance)
        d_check = min(d_band, self.peak_distance())
        if self.plan.power(-d_check, self.theta0 - d_check) >= self.target:
            return self.unrepresentable("float64_relative_lift")
        return m_min

    def peak_distance(self) -> float:
        """Distance of the decreasing direction's noncentrality peak
        (``_decreasing_noncentrality_peak``). Along ``theta = theta0 - d``
        the variance is ``A e^{r d} + B``. With constant absolute variance
        ``r = 2`` and ``B / A = (n_T / n_C) e^{2 theta0}``. With the
        Bernoulli shape the treatment term is ``v (e^{-theta} - m0) / (n_T
        m0^2 (1 - m0))``, so ``r = 1`` and, at the null rate ``p = m0
        e^{theta0}``, ``A = v / (n_T m0 (1 - m0) p)`` and ``B / A = p ((1 -
        m0) n_T / (m0 n_C) - 1)``; the rate ratio is formed in log space.
        """
        plan = self.plan
        if plan.bounded:
            ratio = (
                math.log1p(-plan.baseline.mean)
                + math.log(plan.n_T)
                - self.log_m0
                - math.log(plan.n_C)
            )
            log_null_rate = self.log_m0 + self.theta0
            log_q = log_null_rate + ratio + _log1mexp(-ratio) if ratio > 0.0 else -math.inf
            q = (
                (math.exp(log_q) if log_q < _LOG_FLOAT_MAX else math.inf)
                if ratio > 0.0
                else math.exp(log_null_rate) * math.expm1(ratio)
            )
            return _decreasing_noncentrality_peak(1.0, q, log_q)
        log_q = math.log(plan.n_T) - math.log(plan.n_C) + 2.0 * self.theta0
        q = math.exp(log_q) if log_q < _LOG_FLOAT_MAX else math.inf
        return _decreasing_noncentrality_peak(2.0, q, log_q)

    def upper_endpoint(self, m_min: float) -> float:
        """Last admissible candidate before the far side of the domain."""
        beyond = math.inf if self.sigma > 0.0 else -1.0
        return float_from_ordinal(
            _first_ordinal_accepted(float_ordinal(beyond), float_ordinal(m_min), self.admissible)
        )

    def increasing_top(self, m_max: float, power_max: float) -> float | _MdeRefusal:
        """The increasing direction's noncentrality grows with distance, so
        the upper endpoint bounds the reachable power."""
        if power_max >= self.target:
            return m_max
        if not self.plan.bounded:
            # Noncentrality grows without bound in exact arithmetic.
            return self.unrepresentable("float64_relative_lift")
        ceiling = -self.log_m0 - self.theta0
        if self.plan.power(ceiling, -self.log_m0) >= self.target:
            return self.unrepresentable("float64_relative_lift")
        return self.unattainable(power_max, "bounded_rate_ceiling")

    def decreasing_top(self, m_max: float, d_max: float) -> float | _MdeRefusal:
        """The decreasing direction's variance grows with distance, so its
        noncentrality has a single peak (``peak_distance``) located
        analytically and clamped into the interval: sampling could skip a
        narrow reachable band. An interval opening past the peak (after an
        overflow band) lies on the falling side, where its endpoint's power
        bounds it."""
        plan = self.plan
        d_peak = self.peak_distance()
        m_top = m_max
        if d_peak < d_max:
            inside = math.expm1(-d_peak) / self.compliance
            if self.admissible(inside):
                m_top = inside
        top = self.candidate(m_top)
        assert top is not None
        if self.power_of(top) >= self.target:
            return m_top
        # Exclusive physical limit: both the emitted decrease and its implied
        # absolute alternative stay above the -1 floor.
        d_physical = min(-self.theta_floor, self.theta0 - self.theta_floor)
        d_limit = min(d_peak, d_physical)
        physical = plan.power(-d_limit, self.theta0 - d_limit)
        if physical >= self.target:
            return self.unrepresentable("float64_relative_lift")
        limiting = "noncentrality_peak" if d_peak < d_physical else "relative_lift_floor"
        return self.unattainable(physical, limiting)


def _solve_arm_mde(
    plan: _ArmPlan,
    *,
    target: float,
    null_lift: float,
    alternative: Alternative,
) -> tuple[float, float] | _MdeRefusal:
    """Smallest admissible complier-scale relative effect reaching ``target``
    power at the plan's counts, with that effect's own power.

    A lower endpoint already reaching the target is returned with its
    actual power, including a zero endpoint when the null is admissible and
    the target is at or below the null's own power. The public MDE entry point
    refuses that zero-distance result because no strictly nonzero minimum exists;
    supplied-effect entry points may still report it as their companion MDE.
    A partial-compliance-shifted null's nonzero lower endpoint is also
    returned with its power. A peak or endpoint below target means no answer
    exists. Otherwise the first crossing is isolated to adjacent
    float64 candidates by bisecting their ordinals between the failing
    lower endpoint and the passing peak or endpoint, so the answer reaches
    the target and its predecessor in distance order does not.
    """
    search = _MdeSearch.build(plan, target=target, null_lift=null_lift, alternative=alternative)
    m_min = search.lower_endpoint()
    if isinstance(m_min, _MdeRefusal):
        return m_min
    low = search.candidate(m_min)
    assert low is not None
    power_min = search.power_of(low)
    if power_min >= target:
        return m_min, power_min

    m_max = search.upper_endpoint(m_min)
    high = search.candidate(m_max)
    assert high is not None
    if search.sigma > 0.0:
        m_top = search.increasing_top(m_max, search.power_of(high))
    else:
        m_top = search.decreasing_top(m_max, abs(high[0]))
    if isinstance(m_top, _MdeRefusal):
        return m_top

    m = float_from_ordinal(
        _first_ordinal_accepted(float_ordinal(m_min), float_ordinal(m_top), search.reaches)
    )
    point = search.candidate(m)
    assert point is not None
    return m, search.power_of(point)


def _sequential_sample_size_detected(
    enclosure: _CrossingEnclosure, target: float, n_per_arm: int
) -> bool:
    """Classify one integer size only from rigorous enclosure bounds."""
    if enclosure.lower >= target and (enclosure.cheap or enclosure.resolved):
        return True
    if enclosure.upper < target:
        return False
    if not enclosure.resolved or not enclosure.converged or not enclosure.expected_converged:
        reason = enclosure.resolution_reason or _NODE_CEILING
        _raise("power.sequential_sample_size", n_per_arm=n_per_arm, reason=reason)
    _raise(
        "power.sequential_sample_size_target_inside_enclosure",
        n_per_arm=n_per_arm,
        lower=enclosure.lower,
        upper=enclosure.upper,
    )


def _solve_sequential_mde(
    looks: _PlannedLooks,
    plan: _ArmPlan,
    *,
    target: float,
    null_lift: float,
    alternative: Alternative,
    alpha_seq: float,
    exit_side: Literal["both", "upper"],
    e_value_dual: bool,
    require_expected: bool = True,
) -> _SequentialMde | _MdeRefusal:
    """Smallest admissible complier-scale relative effect whose sequential
    boundary-crossing power is certified to reach ``target`` at the plan's
    counts, with the alternative's own variance behind the drift and the
    boundary at every candidate. A target at or below the null's own
    certified crossing probability raises ``ValueError`` when zero is
    admissible; every other missing answer is an ``_MdeRefusal``."""
    search = _MdeSearch.build(plan, target=target, null_lift=null_lift, alternative=alternative)
    return _SequentialMdeSearch.build(
        looks,
        search,
        alpha_seq=alpha_seq,
        exit_side=exit_side,
        e_value_dual=e_value_dual,
        require_expected=require_expected,
    ).solve()


# Runtime-binomial planning: the exact binomial risk-ratio decision the runtime
# applies to eligible conversion and retention contrasts (see
# ``increment.power._binomial``).

_GeometryCache = dict[tuple[BinomialDecision, Route], RejectionGeometry]


def _runtime_binomial(procedure: ArmPlanningProcedure) -> bool:
    """Whether the runtime decides this plan with the exact binomial test.

    Mirrors ``increment.estimation.engine._binomial_eligible`` -- a declared
    conversion or retention metric, unclustered, not CUPED-adjusted, with raw
    binary counts -- plus that route's own entry conditions: fixed-horizon
    inference and no prior. An absorbed factor or winsorized outcome is no
    longer a raw binomial count, so those plans keep the log-ratio model.
    """
    return (
        _is_bounded_metric(procedure)
        and procedure.dependence == "iid"
        and procedure.decision_method.variance_reduction == "none"
        and isinstance(procedure.inference, FixedInference)
        and not procedure.prior_present
        and procedure.analysis.variance_adjustment == "none"
        and procedure.metric.winsorization == "none"
    )


def _alternative_rate(log_p_c: float, theta: float) -> float:
    """Treatment rate of effect ``theta`` over the control rate ``exp(log_p_c)``."""
    return min(1.0, math.exp(log_p_c + theta))


@dataclass(frozen=True, slots=True)
class _BinomialPlan:
    """The runtime's rejection geometry at one analyzed integer pair, with
    the baseline's control rate; effects enter through the treatment rate
    ``p_c * exp(theta)``."""

    geometry: RejectionGeometry
    p_c: float
    log_p_c: float

    @property
    def basis(self) -> PowerBasis:
        return self.geometry.route

    def rate(self, theta: float) -> float:
        return _alternative_rate(self.log_p_c, theta)

    def evaluate(self, theta: float) -> BinomialPower:
        """The rejection mass of the replayed decision set at effect ``theta`` (the runtime's
        rejection probability on the exact route), enclosed by its numerical error; raises
        `ReplayBoundExceeded` when the cells its alternative window adds would leave the geometry
        storing more than the planning bound."""
        return self.geometry.evaluate(self.p_c, self.rate(theta))

    def supplied(self, theta: float, **sizing: float) -> BinomialPower:
        """`evaluate` of an effect the caller supplied: beyond the planning bound it is refused."""
        try:
            return self.evaluate(theta)
        except ReplayBoundExceeded as exceeded:
            decision = self.geometry.decision
            _refuse_replay_bound(
                self.p_c, decision.n_t, decision.n_c, exceeded.cells, p_t=exceeded.p_t, **sizing
            )

    def bound(self, theta_a: float, theta_b: float) -> float:
        """Upper bound on every computed point power between the effects."""
        low, high = sorted((self.rate(theta_a), self.rate(theta_b)))
        bound = self.geometry.closure_bound(self.p_c, low, high)
        return self.geometry.point_upper(self.p_c, low, high, bound)


def _render_replay_bound(
    *,
    n_c: int,
    n_t: int,
    p_c: float,
    cells: int,
    max_cells: int,
    max_arm_size: int,
    p_t: float | None = None,
    power: float | None = None,
    power_reached: float | None = None,
) -> str:
    rates = f"a {p_c:.6g} control rate" + (
        "" if p_t is None else f" and a {p_t:.6g} alternative treatment rate"
    )
    where = f"n_c={n_c}/n_t={n_t} analyzed units at {rates}"
    replay = f"{cells:,} (control, treatment) count cells against a bound of {max_cells:,}"
    scope = (
        f"The bound is the replay's cost, not a limit of the analysis: the runtime decides "
        f"arms of up to {max_arm_size:,} analyzed units"
    )
    if power is None:
        return (
            f"planning the selected binomial decision model at {where} is not supported: its "
            f"replay spans {replay}. {scope} -- plan fewer analyzed units per arm (a larger "
            "relative_lift needs fewer)"
        )
    return (
        f"target power {power} is not reached by the selected binomial decision model within "
        f"the planning replay bound (power {power_reached:.6g} at {where}, whose replay spans "
        f"{replay}). {scope} -- plan a larger relative_lift or a lower target power"
    )


_BINOMIAL_REPLAY_BOUND = RefusalSpec(
    "power.binomial_replay_bound_exceeded", InvalidRequestError, _render_replay_bound
)


def _replay_bound_context(
    p_c: float, n_T: int, n_C: int, cells: int, **extra: float | None
) -> dict[str, object]:
    """Context of the replay-bound refusal for ``cells`` count cells at analyzed counts
    ``(n_T, n_C)``; ``p_t`` names the alternative rate behind them, and a size search adds its
    target ``power`` and the ``power_reached`` at its ceiling."""
    return {
        "n_c": n_C,
        "n_t": n_T,
        "p_c": p_c,
        "cells": cells,
        "max_cells": PLANNING_CELL_CEILING,
        "max_arm_size": FINITE_SAMPLE_MAX_ARM_SIZE,
        **{key: value for key, value in extra.items() if value is not None},
    }


def _refuse_replay_bound(
    p_c: float, n_T: int, n_C: int, cells: int, **extra: float | None
) -> NoReturn:
    refuse(_BINOMIAL_REPLAY_BOUND, **_replay_bound_context(p_c, n_T, n_C, cells, **extra))


def _binomial_key(procedure: ArmPlanningProcedure, n_T: int, n_C: int) -> BinomialDecision:
    """The runtime decision at analyzed counts ``(n_T, n_C)``: the compiled decision alpha
    fixes the nuisance budget, the compiled tail allocation the rejection threshold, and
    ``1 + null_lift`` the tested risk ratio."""
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    return BinomialDecision(
        n_c=n_C,
        n_t=n_T,
        null_ratio=1.0 + decision.null_lift,
        beta=nuisance_beta(procedure.compiled_alpha),
        tail_alpha=procedure.compiled_tail_alpha,
        alternative=decision.alternative,
    )


def _binomial_plan(
    procedure: ArmPlanningProcedure,
    baseline: Baseline,
    n_T: int,
    n_C: int,
    cache: _GeometryCache,
    *,
    route: Route | None = None,
) -> _BinomialPlan:
    """The runtime decision at analyzed counts ``(n_T, n_C)``. A decision the runtime refuses in
    full decides no count pair, so it has no power to plan and is refused (`_refuse_undecided`).
    One whose replay would span more than ``PLANNING_CELL_CEILING`` count cells at the null rate
    is refused before any is built; its geometry refuses an alternative whose window would take
    it past the bound. ``route`` overrides the budgeted route (sizing proposals only)."""
    key = _binomial_key(procedure, n_T, n_C)
    if refused(key):
        _refuse_undecided(procedure, key, scope="requested")
    cells = window_cells(key, baseline.mean)
    if cells > PLANNING_CELL_CEILING:
        _refuse_replay_bound(baseline.mean, n_T, n_C, cells)
    route = route_for(cells) if route is None else route
    geometry = cache.get((key, route))
    if geometry is None:
        geometry = cache[key, route] = RejectionGeometry(key, route, PLANNING_CELL_CEILING)
    geometry.begin_solve()
    return _BinomialPlan(geometry, baseline.mean, math.log(baseline.mean))


def _analyzed_counts(n_T: int, n_C: int, baseline: Baseline) -> tuple[int, int]:
    """Integer analyzed counts behind assigned arms: the binomial law needs
    whole units, and these are the triggered counts the result reports."""
    if baseline.trigger_rate == 1.0:
        return n_T, n_C
    return round(n_T * baseline.trigger_rate), round(n_C * baseline.trigger_rate)


# Evaluations and interval bounds one binomial effect search may spend.
_BINOMIAL_MDE_EVALUATIONS = 512
_BINOMIAL_MDE_ATOL = 1e-8
_BINOMIAL_MDE_RTOL = 1e-8


class _SearchBudgetExhausted(Exception):
    """The ordered interval search exhausted its evaluation allowance."""


def _binomial_curvature_gap(n: int, low: float, high: float) -> Fraction:
    """Bound interpolation error of a fixed binomial rejection probability.

    Bernstein second differences give ``|P''| <= 2n(n-1)`` everywhere.
    Inside (0,1), the score bound ``2n/min(p(1-p))`` can be smaller.
    Multiply the smaller bound by the squared interval width divided by eight.
    """
    a, b = Fraction(low), Fraction(high)
    curvature = Fraction(2 * n * (n - 1))
    if 0 < a <= b < 1:
        curvature = min(curvature, 2 * n / min(a * (1 - a), b * (1 - b)))
    return curvature * (b - a) ** 2 / 8


@dataclass(slots=True)
class _BinomialMdeState:
    evaluations: int = 0
    unresolved: tuple[float, float] | None = None
    lower: float = 0.0
    upper: float = 0.0


@dataclass(frozen=True, slots=True)
class _BinomialMdeSearch(_MdeSearch):
    """Search intervals in distance order without assuming point-power monotonicity.

    Exclude an interval only with an upper bound on every computed point.
    Otherwise subdivide left first. Unexcluded earlier effects must be within
    ``atol + rtol * abs(effect)`` of the returned, evaluated passing point.
    """

    model: _BinomialPlan
    state: _BinomialMdeState

    @classmethod
    def of(
        cls,
        plan: _ArmPlan,
        model: _BinomialPlan,
        *,
        target: float,
        null_lift: float,
        alternative: Alternative,
    ) -> _BinomialMdeSearch:
        base = _MdeSearch.build(plan, target=target, null_lift=null_lift, alternative=alternative)
        return cls(
            plan=base.plan,
            target=base.target,
            theta0=base.theta0,
            sigma=base.sigma,
            compliance=base.compliance,
            log_m0=base.log_m0,
            theta_floor=base.theta_floor,
            model=model,
            state=_BinomialMdeState(),
        )

    def _charge(self) -> None:
        self.state.evaluations += 1
        if self.state.evaluations > _BINOMIAL_MDE_EVALUATIONS:
            raise _SearchBudgetExhausted

    def overflowed_null(self) -> float | _MdeRefusal:
        """Exclude an unrepresentable initial band before searching representable effects."""
        if self.sigma > 0.0:
            return self.unrepresentable("float64_relative_lift")
        edge = math.nextafter(-1.0, 0.0)
        if not self.admissible(edge):
            return self.unrepresentable("float64_relative_lift")
        m_min = float_from_ordinal(
            _first_ordinal_accepted(float_ordinal(0.0), float_ordinal(edge), self.admissible)
        )
        d_band = -math.log1p(float_from_ordinal(float_ordinal(m_min) + 1) * self.compliance)
        if self.model.bound(self.theta0, self.theta0 - d_band) >= self.target:
            return self.unrepresentable("float64_relative_lift")
        return m_min

    def evaluate(self, effect: float) -> BinomialPower:
        self._charge()
        point = self.candidate(effect)
        assert point is not None
        return self.model.evaluate(point[1])

    def detected(self, power: BinomialPower) -> bool:
        return power.power >= self.target

    @staticmethod
    def tolerance(effect: float) -> float:
        return _BINOMIAL_MDE_ATOL + _BINOMIAL_MDE_RTOL * abs(effect)

    def unresolved(self, lo: float, hi: float, reason: str) -> _MdeRefusal:
        state = self.state
        interval = state.unresolved or (lo, hi)
        return _MdeRefusal(
            "numerical_resolution",
            _MDE_NUMERICAL_RESOLUTION,
            {
                "target_power": self.target,
                "direction": self.direction,
                "unresolved_interval": tuple(sorted(interval)),
                "power_enclosure": (state.lower, state.upper) if state.unresolved else None,
                "stopping_reason": reason,
            },
        )

    def _bound(self, lo: float, hi: float, left: BinomialPower, right: BinomialPower) -> float:
        self._charge()
        a, b = self.candidate(lo), self.candidate(hi)
        assert a is not None and b is not None
        low_rate, high_rate = sorted((self.model.rate(a[1]), self.model.rate(b[1])))
        gap = _binomial_curvature_gap(self.model.geometry.decision.n_t, low_rate, high_rate)
        true_upper = min(
            1.0, math.nextafter(float(Fraction(max(left.upper, right.upper)) + gap), math.inf)
        )
        return self.model.geometry.point_upper(self.model.p_c, low_rate, high_rate, true_upper)

    def _midpoint(self, lo: float, hi: float) -> float:
        """Bisect log distance without overflowing a large relative effect."""
        a = math.log1p(lo * self.compliance)
        b = math.log1p(hi * self.compliance)
        mid = math.expm1(a + (b - a) / 2.0) / self.compliance
        if min(lo, hi) < mid < max(lo, hi):
            return mid
        first, last = float_ordinal(lo), float_ordinal(hi)
        return float_from_ordinal(first + (last - first) // 2)

    def between(
        self, lo: float, hi: float, left: BinomialPower, right: BinomialPower
    ) -> tuple[float, float] | _MdeRefusal | None:
        pending = self.state.unresolved
        if pending is not None and abs(lo - pending[0]) > self.tolerance(lo):
            return self.unresolved(lo, hi, "an earlier detectable region remains unresolved")
        if self.detected(left):
            return lo, left.power
        bound = self._bound(lo, hi, left, right)
        if bound < self.target:
            return None
        adjacent = abs(float_ordinal(hi) - float_ordinal(lo)) <= 1
        if abs(hi - lo) <= self.tolerance(hi) / 4.0 or adjacent:
            if self.detected(right):
                if pending is not None and abs(hi - pending[0]) > self.tolerance(hi):
                    return self.unresolved(
                        lo, hi, "an earlier detectable region remains unresolved"
                    )
                return hi, right.power
            if pending is None:
                self.state.unresolved = (lo, hi)
                self.state.lower = max(left.lower, right.lower)
                self.state.upper = bound
            return None
        mid = self._midpoint(lo, hi)
        middle = self.evaluate(mid)
        earlier = self.between(lo, mid, left, middle)
        if earlier is not None:
            return earlier
        return self.between(mid, hi, middle, right)


def _solve_binomial_mde(
    plan: _ArmPlan,
    model: _BinomialPlan,
    *,
    target: float,
    null_lift: float,
    alternative: Alternative,
) -> tuple[float, float] | _MdeRefusal:
    """Earliest detectable region to numerical effect tolerance, with point power.

    Wider unresolved earlier intervals refuse instead of being skipped.
    Solved searches are memoized for a curve's companion effects.
    """
    memo = model.geometry.effects
    key = (model.p_c, plan.baseline.compliance, target)
    if key not in memo:
        memo[key] = _search_binomial_mde(
            plan, model, target=target, null_lift=null_lift, alternative=alternative
        )
    return cast("tuple[float, float] | _MdeRefusal", memo[key])


def _search_binomial_mde(
    plan: _ArmPlan,
    model: _BinomialPlan,
    *,
    target: float,
    null_lift: float,
    alternative: Alternative,
) -> tuple[float, float] | _MdeRefusal:
    geometry = model.geometry
    model = replace(
        model,
        geometry=RejectionGeometry(geometry.decision, geometry.route, geometry.max_cells),
    )
    search = _BinomialMdeSearch.of(
        plan, model, target=target, null_lift=null_lift, alternative=alternative
    )
    try:
        return _ordered_exclusion(search)
    except ReplayBoundExceeded as exceeded:
        # A candidate's alternative window takes the replay past the planning bound: the search
        # ends unresolved, so a companion effect is unavailable (``numerical_resolution``) and a
        # direct request is refused with the bound.
        decision = model.geometry.decision
        context = _replay_bound_context(
            model.p_c, decision.n_t, decision.n_c, exceeded.cells, p_t=exceeded.p_t
        )
        return _MdeRefusal("numerical_resolution", _BINOMIAL_REPLAY_BOUND, context)


def _ordered_exclusion(search: _BinomialMdeSearch) -> tuple[float, float] | _MdeRefusal:
    m_min = search.lower_endpoint()
    if isinstance(m_min, _MdeRefusal):
        return m_min
    m_max = search.upper_endpoint(m_min)
    try:
        first = search.evaluate(m_min)
        if search.detected(first):
            return m_min, first.power
        last = search.evaluate(m_max)
        found = search.between(m_min, m_max, first, last)
        if found is not None:
            return found
        if search.state.unresolved is not None:
            return search.unresolved(m_min, m_max, "no evaluated point reaches the target")
        limiting = "bounded_rate_ceiling" if search.sigma > 0.0 else "relative_lift_floor"
        return search.unattainable(last.power, limiting)
    except _SearchBudgetExhausted:
        return search.unresolved(m_min, m_max, "the ordered interval search exhausted its budget")


def _fixed_mde(
    plan: _ArmPlan,
    model: _BinomialPlan | None,
    *,
    target: float,
    decision: RelativeDecisionPolicy,
) -> tuple[float, float] | _MdeRefusal:
    if model is not None:
        return _solve_binomial_mde(
            plan,
            model,
            target=target,
            null_lift=decision.null_lift,
            alternative=decision.alternative,
        )
    return _solve_arm_mde(
        plan, target=target, null_lift=decision.null_lift, alternative=decision.alternative
    )


def _companion_mde(
    plan: _ArmPlan,
    design: PowerDesign,
    decision: RelativeDecisionPolicy,
    model: _BinomialPlan | None = None,
) -> tuple[float | None, MdeUnavailableReason | None]:
    """Fixed-horizon companion effect for a supplied-effect answer: the
    minimum detectable effect at the plan's counts, or a numeric null with
    the reason it does not exist. Only the solver's own recognized
    unavailability outcomes become nulls; anything else propagates."""
    solved = _fixed_mde(plan, model, target=design.power, decision=decision)
    if isinstance(solved, _MdeRefusal):
        return None, solved.reason
    mde_relative, _ = solved
    return mde_relative, None


_BINOMIAL_SIZE_LIMIT = RefusalSpec(
    "power.binomial_size_search_unreachable",
    InvalidRequestError,
    template=(
        "target power {power} is not reached by the selected binomial decision model "
        "within its arm ceiling of {max_arm_size} analyzed units per arm (power "
        "{maximum_power:.6g} at n_per_arm={n_per_arm}) -- plan a larger relative_lift "
        "or a lower target power"
    ),
)


def _render_tail_level(
    *,
    alpha: float,
    beta: float,
    tail_alpha: float,
    margin: float,
    n_c: int,
    n_t: int,
    solver_floor: float,
    cause: str,
    scope: str,
) -> str:
    if cause == "solver_floor":
        why = (
            f"its nuisance budget {beta:.3g} (alpha / 32) is below the {solver_floor:g} floor of "
            "the Clopper-Pearson endpoint solver"
        )
    else:
        where = (
            f"at the smallest plannable design (n_c={n_c}, n_t={n_t} analyzed units)"
            if scope == "smallest"
            else f"at n_c={n_c}, n_t={n_t} analyzed units"
        )
        why = (
            f"its float margin {margin:.3g} {where} reaches what the tail level {tail_alpha:.3g} "
            f"leaves after the nuisance budget {beta:.3g}"
        )
    way = (
        "so no arm size has power -- plan a larger alpha"
        if cause == "solver_floor" or scope == "smallest"
        else "so this size has no power to plan -- plan fewer analyzed units per arm or a "
        "larger alpha"
    )
    return (
        f"the runtime's exact binomial decision refuses every count pair at alpha={alpha}: "
        f"{why}, {way}"
    )


_BINOMIAL_TAIL_LEVEL = RefusalSpec(
    "power.binomial_tail_level_unrepresentable", InvalidRequestError, _render_tail_level
)

_BINOMIAL_ARM_AT_FLOOR = RefusalSpec(
    "power.binomial_arm_ceiling_below_smallest_design",
    InvalidRequestError,
    template=(
        "the smallest plannable design (n_t={n_t}, n_c={n_c} analyzed units at allocation "
        "{allocation}) already has an arm above the runtime's ceiling of {max_arm_size} analyzed "
        "units, so no size can be planned -- use a less lopsided allocation"
    ),
)

_BINOMIAL_ARM_CEILING = RefusalSpec(
    "power.binomial_arm_ceiling_exceeded",
    InvalidRequestError,
    template=(
        "n_c={n_c}/n_t={n_t} analyzed units has an arm above the runtime's ceiling of "
        "{max_arm_size} analyzed units, where it refuses every count pair, so no power can be "
        "planned -- plan fewer units per arm"
    ),
)


def _refuse_undecided(
    procedure: ArmPlanningProcedure,
    key: BinomialDecision,
    *,
    scope: Literal["smallest", "requested"],
    allocation: float | None = None,
) -> NoReturn:
    """Refuse a decision the runtime refuses in full, by its cause (`refused`): an arm above
    its ceiling, a nuisance budget below the solver's floor, or a tail level the float margin
    dominates. ``scope`` is ``smallest`` for the smallest design a size search can plan (then no
    size has power) and ``requested`` for the analyzed counts a caller asked about."""
    if max(key.n_c, key.n_t) > FINITE_SAMPLE_MAX_ARM_SIZE:
        if scope == "smallest":
            refuse(
                _BINOMIAL_ARM_AT_FLOOR,
                n_t=key.n_t,
                n_c=key.n_c,
                allocation=allocation,
                max_arm_size=FINITE_SAMPLE_MAX_ARM_SIZE,
            )
        refuse(
            _BINOMIAL_ARM_CEILING,
            n_t=key.n_t,
            n_c=key.n_c,
            max_arm_size=FINITE_SAMPLE_MAX_ARM_SIZE,
        )
    refuse(
        _BINOMIAL_TAIL_LEVEL,
        alpha=procedure.compiled_alpha,
        beta=key.beta,
        tail_alpha=key.tail_alpha,
        margin=tail_margin(key),
        n_c=key.n_c,
        n_t=key.n_t,
        solver_floor=solver_floor(),
        cause="solver_floor" if solver_refuses(key) else "float_margin",
        scope=scope,
    )


# Proposal, verification and bracketing share one search state.
def _binomial_size(  # noqa: PLR0915
    procedure: ArmPlanningProcedure,
    baseline: Baseline,
    design: PowerDesign,
    *,
    theta: float,
    proposal: int,
    cache: _GeometryCache,
) -> tuple[int, int, _BinomialPlan, float]:
    """Assigned ``(n_T, n_C)`` whose runtime binomial power is certified to reach
    the target (the lower end of its numerical enclosure does) while assigned
    ``n_T - 1`` is not (or ``n_T`` is the per-arm floor), with that size's plan
    and computed power. Each candidate's binomial law uses its analyzed counts
    (``_analyzed_counts``), exactly as ``achieved_power`` evaluates the returned
    size.

    Every candidate is evaluated with the route selected at its own size. The
    search starts at ``proposal`` and brackets upward or downward, stepping
    by a probit-secant in ``sqrt(n)`` (power ``~ Phi(b sqrt(n) - z_alpha)``),
    then narrows the bracket to adjacent sizes, halving whenever the secant
    stalls. Power need not be monotone in the sample size, so the answer is
    the verified smallest certified size of its bracket, not a global minimum.
    """
    target = design.power
    floor = _assigned_minimum_per_arm(procedure, baseline)
    z_target = float(_ndtri(target))
    z_alpha = float(_norm.isf(procedure.compiled_tail_alpha))
    rate = _alternative_rate(math.log(baseline.mean), theta)
    arm_ceiling = _binomial_admitted_ceiling(
        procedure, baseline, design, floor, _binomial_arm_ceiling(design, floor, baseline)
    )
    ceiling = _binomial_replay_ceiling(procedure, baseline, design, floor, arm_ceiling, rate)
    powers: dict[int, BinomialPower] = {}

    def plan_at(n: int, route: Route | None = None) -> _BinomialPlan:
        n_T, n_C = _analyzed_counts(*_compute_arms(n, design, minimum_per_arm=floor), baseline)
        return _binomial_plan(procedure, baseline, n_T, n_C, cache, route=route)

    def power_at(n: int) -> BinomialPower:
        if n not in powers:
            powers[n] = plan_at(n).supplied(theta)
        return powers[n]

    def crossing(evaluate: Callable[[int], BinomialPower], n: int, *, final: bool) -> int:
        """Smallest certified size of the bracket the search closes from ``n``.
        A search reaching the arm ceiling without certifying refuses when
        ``final``; a proposal search returns the ceiling instead, leaving the
        refusal to the selected-route search."""
        values: dict[int, BinomialPower] = {}

        def value(m: int) -> BinomialPower:
            if m not in values:
                values[m] = evaluate(m)
            return values[m]

        def probit(m: int) -> float:
            return float(_ndtri(min(max(values[m].power, 1e-15), 1.0 - 1e-15)))

        def secant(a: int, b: int | None) -> float:
            """Size where the probit line through the evaluated sizes reaches
            the target; one size uses the line through ``(0, -z_alpha)``."""
            if b is None or a == b:
                slope = (probit(a) + z_alpha) / math.sqrt(a)
                root_n = (z_target + z_alpha) / slope if slope > 0.0 else math.inf
            else:
                slope = (probit(b) - probit(a)) / (math.sqrt(b) - math.sqrt(a))
                root_n = math.sqrt(a) + (z_target - probit(a)) / slope if slope > 0.0 else math.inf
            return root_n * root_n if math.isfinite(root_n) and root_n > 0.0 else math.inf

        lo: int | None = None  # largest failing size of the bracket
        hi: int | None = None  # smallest passing size of the bracket
        evaluated: list[int] = []
        stalls = 0
        while True:
            width = None if lo is None or hi is None else hi - lo
            if value(n).lower >= target:
                hi = n if hi is None else min(hi, n)
            else:
                lo = n if lo is None else max(lo, n)
            evaluated.append(n)
            if hi is not None and (hi <= floor or (lo is not None and hi - lo <= 1)):
                return hi
            previous = evaluated[-2] if len(evaluated) > 1 else None
            if hi is None:
                assert lo is not None
                if lo >= ceiling:
                    if not final:
                        return ceiling
                    if ceiling < arm_ceiling:
                        n_T, n_C = _analyzed_counts(
                            *_compute_arms(lo, design, minimum_per_arm=floor), baseline
                        )
                        cells, p_t = _sizing_cells(
                            _binomial_key(procedure, n_T, n_C), baseline.mean, rate
                        )
                        _refuse_replay_bound(
                            baseline.mean,
                            n_T,
                            n_C,
                            cells,
                            p_t=p_t,
                            power=target,
                            power_reached=values[lo].power,
                        )
                    refuse(
                        _BINOMIAL_SIZE_LIMIT,
                        power=target,
                        max_arm_size=max(
                            _analyzed_counts(
                                *_compute_arms(arm_ceiling, design, minimum_per_arm=floor), baseline
                            )
                        ),
                        maximum_power=values[lo].power,
                        n_per_arm=_compute_arms(lo, design, minimum_per_arm=floor)[0],
                    )
                guess = min(secant(lo, previous), 4.0 * lo + 4.0)
                n = min(ceiling, max(lo + 1, math.ceil(1.05 * lo), math.ceil(guess)))
            elif lo is None:
                if previous is None:
                    # A proposal that passes is checked against its predecessor first.
                    n = hi - 1
                    continue
                guess = secant(hi, previous)
                guess = guess if math.isfinite(guess) else 0.0
                n = max(floor, min(hi - 1, math.floor(0.95 * hi), max(math.floor(guess), hi // 4)))
            else:
                if width is not None and 2 * (hi - lo) > width:
                    stalls += 1
                else:
                    stalls = 0
                guess = secant(lo, hi)
                if stalls >= 2 or not math.isfinite(guess):
                    n = lo + (hi - lo) // 2
                else:
                    n = min(hi - 1, max(lo + 1, round(guess)))

    start = min(max(proposal, floor), ceiling)
    if plan_at(start).basis == "exact":
        # The approximate replay costs a fraction of the exact one and agrees
        # with it closely; its crossing proposes the size the exact decision
        # then verifies (the proposal and its predecessor) and corrects.
        start = crossing(lambda m: plan_at(m, "approximate").supplied(theta), start, final=False)
    hi = crossing(power_at, start, final=True)
    n_T, n_C = _compute_arms(hi, design, minimum_per_arm=floor)
    return n_T, n_C, plan_at(hi), powers[hi].power


def _binomial_arm_ceiling(design: PowerDesign, floor: int, baseline: Baseline) -> int:
    """Largest assigned treatment size whose analyzed arms both stay within
    the runtime's arm ceiling (beyond it every count pair is refused)."""
    lo, hi = floor, math.ceil(FINITE_SAMPLE_MAX_ARM_SIZE / baseline.trigger_rate) + 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        arms = _compute_arms(mid, design, minimum_per_arm=floor)
        if max(_analyzed_counts(*arms, baseline)) <= FINITE_SAMPLE_MAX_ARM_SIZE:
            lo = mid
        else:
            hi = mid - 1
    return lo


# The retained-cell count is not monotone in the size (each window's integer edges move
# independently), so a bisection's crossing is not the largest size that fits: the search
# ceiling stays a fraction under it, so `_binomial_plan` never refuses a proposed size. The
# predicate counts the rectangles an evaluation classifies (`_sizing_cells`).
_REPLAY_CEILING_JITTER = 128


def _sizing_cells(key: BinomialDecision, p_c: float, rate: float) -> tuple[int, float | None]:
    """Cells of the larger of the rectangles a size search evaluates at ``key``: the null
    rectangle every plan is checked against, and the one at the supplied effect's treatment
    ``rate``. The rate comes back when its rectangle is the larger."""
    null, alternative = window_cells(key, p_c), window_cells(key, p_c, rate)
    return (alternative, rate) if alternative > null else (null, None)


def _binomial_admitted_ceiling(
    procedure: ArmPlanningProcedure,
    baseline: Baseline,
    design: PowerDesign,
    floor: int,
    upper: int,
) -> int:
    """Largest assigned treatment size at most ``upper`` whose decision the runtime does not
    refuse in full: from the size where the float margin dominates the tail level, the runtime
    decides no count pair. A decision refused even at the smallest arms has no such size and is
    refused."""

    def key_at(n: int) -> BinomialDecision:
        n_T, n_C = _analyzed_counts(*_compute_arms(n, design, minimum_per_arm=floor), baseline)
        return _binomial_key(procedure, n_T, n_C)

    def admitted(n: int) -> bool:
        return not refused(key_at(n))

    if not admitted(floor):
        _refuse_undecided(procedure, key_at(floor), scope="smallest", allocation=design.allocation)
    if admitted(upper):
        return upper
    lo, hi = floor, upper - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if admitted(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _binomial_replay_ceiling(
    procedure: ArmPlanningProcedure,
    baseline: Baseline,
    design: PowerDesign,
    floor: int,
    upper: int,
    rate: float,
) -> int:
    """The ceiling of a size search: the assigned treatment size at most ``upper`` whose analyzed
    arms span at most ``PLANNING_CELL_CEILING`` count cells at the null rate and at the supplied
    effect's treatment ``rate``, found by bisection (``upper`` itself when it fits), less
    ``1 / _REPLAY_CEILING_JITTER`` of it. The count of cells is not monotone in the size, so that
    size is not the largest that fits, and sizes above the ceiling may fit too."""

    def within(n: int) -> bool:
        n_T, n_C = _analyzed_counts(*_compute_arms(n, design, minimum_per_arm=floor), baseline)
        cells, _ = _sizing_cells(_binomial_key(procedure, n_T, n_C), baseline.mean, rate)
        return cells <= PLANNING_CELL_CEILING

    if within(upper):
        return upper
    lo, hi = floor, upper - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if within(mid):
            lo = mid
        else:
            hi = mid - 1
    return max(floor, lo - lo // _REPLAY_CEILING_JITTER)


def _sequential_companion(
    looks: _PlannedLooks,
    plan: _ArmPlan,
    *,
    design: PowerDesign,
    alpha_seq: float,
    exit_side: Literal["both", "upper"],
    e_value_dual: bool,
) -> tuple[float | None, float | None, MdeUnavailableReason | None]:
    """Sequential companion effect for a supplied-effect answer as
    ``(complier-scale mde_relative, its crossing power, None)``, or ``(None,
    None, reason)`` when none exists at the plan's counts. Only the solver's
    own recognized unavailability outcomes become nulls; anything else
    propagates."""
    decision = cast("RelativeDecisionPolicy", plan.procedure.decision)
    solved = _solve_sequential_mde(
        looks,
        plan,
        target=design.power,
        null_lift=decision.null_lift,
        alternative=decision.alternative,
        alpha_seq=alpha_seq,
        exit_side=exit_side,
        e_value_dual=e_value_dual,
        require_expected=False,
    )
    if isinstance(solved, _MdeRefusal):
        return None, None, solved.reason
    return solved.mde_relative, solved.power, None


# Solvers


# Public sizing signature is the API for power planning.
def required_sample_size(  # noqa: PLR0915
    relative_lift: float,
    baseline: Baseline,
    procedure: ArmPlanningProcedure,
    design: PowerDesign | None = None,
    *,
    planned_looks: int | None = None,
) -> PowerResult:
    """Compute the sample size needed to detect *relative_lift* with the
    given *baseline* assumptions and *design* parameters.

    ``n_per_arm`` is the treatment arm size after ceiling to whole units;
    ``power`` is the actual (slightly >= target) power at that integer size,
    with the treatment arm's variance evaluated at *relative_lift*.
    ``mde_relative`` is the companion minimum detectable effect at that
    size, ``None`` with ``mde_unavailable_reason`` when none exists at
    ``design.power``.

    ``GaussianScoreMixture`` computes the always-valid Gaussian-model
    planning approximation the runtime's ``asymptotic_mean`` boundary
    executes; equal-look batches default to 14 looks. Its crossing
    enclosures do not certify power for Beta/NIG/NIW runtime likelihood
    evidence. Registered runtime policies are refused.

    For triggered experiments, sequential looks count analyzed units:
    configure the planning maximum from ``PowerResult.n_triggered_total``,
    not the assigned-unit ``n_total``.

    A *relative_lift* exactly at the null boundary raises ``ValueError``
    instead of returning a degenerate sentinel: no finite sample size
    reaches above-alpha power at zero distance. A one-sided design whose
    *relative_lift* lies on the wrong side of the null boundary is
    refused too - the closed form squares the distance, so it would
    otherwise size for the reflected effect at ~alpha power.

    For a cluster-randomized design, set ``baseline.cluster_icc`` and
    ``baseline.avg_cluster_size``: the design effect inflates
    ``effective_var``, so the solved ``n_total`` is already
    cluster-corrected, and ``n_clusters_per_arm``/``n_clusters_total``
    report the clusters to recruit. Triggered designs also require
    ``baseline.cluster_participation``: contributing analyzed pilot clusters
    divided by recruited pilot clusters. ``Analysis.planning_baseline`` derives
    it from the source; manual baselines use those pilot counts.
    Omitting clustering is anticonservative by the design effect -- at ICC
    0.05 and 50 units per cluster it under-sizes by 3.45x.

    A supplied ``planned_looks`` must be a positive integer, validated even
    when ``inference`` is fixed.
    """
    procedure, baseline = _prepare_solver(procedure, baseline)
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    if design is None:
        design = PowerDesign()
    else:
        design = PowerDesign.model_validate(design)
    looks = _planned_looks(procedure, planned_looks)
    bounded = _bounded_preflight(procedure, baseline, null_lift=decision.null_lift)
    theta, distance = _supplied_effect(procedure, baseline, relative_lift, bounded=bounded)

    alternative = decision.alternative
    if alternative == "greater" and distance < 0.0:
        _raise(
            "power.relative_lift_lies", relative_lift=relative_lift, null_lift=decision.null_lift
        )
    if alternative == "less" and distance > 0.0:
        _raise(
            "power.relative_lift_lies_above_null",
            relative_lift=relative_lift,
            null_lift=decision.null_lift,
        )
    alpha_seq, exit_side, e_value_dual = _sequential_side(procedure)
    theta_dir = _sequential_drift(procedure, distance)

    clustered = baseline.avg_cluster_size > 1.0 and looks is None
    binomial = _runtime_binomial(procedure)
    model: _BinomialPlan | None = None
    binomial_power = math.nan
    if distance == 0.0:
        _raise(
            "power.size_design_relative",
            relative_lift=relative_lift,
            null_lift=decision.null_lift,
        )
    elif isinstance(baseline, QuantileBaseline):
        # A QuantileBaseline's per-arm variance depends on the arm's analyzed count, so bisect
        # instead of taking the closed-form step below. Quantile metrics refuse clustering and
        # sequential inference upstream (arm_planning_support): looks is None and clustered false.
        n_T, n_C = _quantile_size_search(
            procedure, baseline, design, theta=theta, distance=distance, bounded=bounded
        )
    else:
        # Normal seed: the per-unit H1 variance coefficient at the allocation
        # fractions times the dimensionless quantile sum, over the squared
        # distance, formed in log space. Integer rounding, the cluster floor
        # and reference, triggering, and the sequential search refine it.
        z_sum = abs(_mde_theta(1.0, design, procedure))
        if z_sum == 0.0:
            n_total_float = 0.0
        else:
            log_n_total = (
                _arm_log_se_sq(
                    design.allocation,
                    1.0 - design.allocation,
                    baseline,
                    theta=theta,
                    bounded=bounded,
                )
                + 2.0 * math.log(z_sum)
                - 2.0 * math.log(abs(distance))
            )
            if log_n_total >= _LOG_FLOAT_MAX:
                _raise(
                    "power.sample_size_detecting",
                    relative_lift=relative_lift,
                    null_lift=decision.null_lift,
                )
            n_total_float = math.exp(log_n_total)
        n_T = max(_minimum_per_arm(procedure), math.ceil(n_total_float * design.allocation))
        n_T, n_C = _compute_arms(n_T, design, minimum_per_arm=_minimum_per_arm(procedure))

        if binomial:
            # The Normal seed proposes; the runtime binomial decision decides,
            # searching assigned sizes whose analyzed counts it integrates.
            n_T, n_C, model, binomial_power = _binomial_size(
                procedure,
                baseline,
                design,
                theta=theta,
                proposal=math.ceil(n_T / baseline.trigger_rate),
                cache={},
            )
        elif looks is not None:
            spec, fractions = looks.spec, looks.fractions
            bounds_cache = tuple(
                _planning_bounds_from_log_se(
                    spec, fractions, 0.0, alpha_seq, exit_side, e_value_dual=e_value_dual
                )
            )

            def _seq_enclosure_at(n_t: int) -> _CrossingEnclosure:
                nt, nc = _compute_arms(n_t, design, minimum_per_arm=_minimum_per_arm(procedure))
                log_se_sq = _arm_log_se_sq(nt, nc, baseline, theta=theta, bounded=bounded)
                return _certified_crossing(
                    bounds_cache,
                    fractions,
                    _noncentrality(theta_dir, log_se_sq),
                    exit_side,
                )

            lo, hi = n_T, 8 * n_T
            cap = 1024 * n_T
            while not _sequential_sample_size_detected(_seq_enclosure_at(hi), design.power, hi):
                hi *= 2
                if hi > cap:
                    _raise("power.sequential_design_more", power=design.power)
            n_T, n_C = _compute_arms(
                bisect_first_true(
                    lo,
                    hi,
                    lambda mid: _sequential_sample_size_detected(
                        _seq_enclosure_at(mid), design.power, mid
                    ),
                ),
                design,
                minimum_per_arm=_minimum_per_arm(procedure),
            )
        elif clustered:
            # Search over ASSIGNED treatment units. Triggering is applied
            # after recruitment, so both the analyzed SE and cluster
            # reference must be reconstructed from the same assigned pair.
            initial_assigned_t = max(
                math.ceil(n_T / baseline.trigger_rate),
                _assigned_minimum_per_arm(procedure, baseline),
            )

            def _cluster_meets_target(assigned_t: int) -> bool:
                assigned_t, assigned_c = _compute_arms(
                    assigned_t,
                    design,
                    minimum_per_arm=_assigned_minimum_per_arm(procedure, baseline),
                )
                dof = _cluster_dof(
                    assigned_t,
                    assigned_c,
                    baseline,
                    enforce_floor=False,
                )
                assert dof is not None
                # Candidates without a t reference lie outside the search domain.
                if dof <= 0.0:
                    return False
                plan = _ArmPlan(
                    procedure,
                    baseline,
                    assigned_t * baseline.trigger_rate,
                    assigned_c * baseline.trigger_rate,
                    dof,
                    bounded,
                )
                return plan.power(distance, theta) >= design.power

            lo, hi = initial_assigned_t, max(2, initial_assigned_t)
            while not _cluster_meets_target(hi):
                hi *= 2
            n_T, n_C = _compute_arms(
                bisect_first_true(lo, hi, _cluster_meets_target),
                design,
                minimum_per_arm=_assigned_minimum_per_arm(procedure, baseline),
            )
    # Re-compute power and MDE at the actual integer n.
    if clustered:
        # The cluster branch already searched assigned units; do not ceil a
        # second, independently rounded analyzed pair.
        assigned_T, assigned_C = n_T, n_C
        analyzed_T = assigned_T * baseline.trigger_rate
        analyzed_C = assigned_C * baseline.trigger_rate
    elif model is not None:
        # The binomial search already chose assigned units; report the
        # analyzed counts its power was integrated at.
        assigned_T, assigned_C = n_T, n_C
        n_T, n_C = _analyzed_counts(assigned_T, assigned_C, baseline)
        analyzed_T, analyzed_C = n_T, n_C
    else:
        assigned_T = math.ceil(n_T / baseline.trigger_rate)
        assigned_C = math.ceil(n_C / baseline.trigger_rate)
        analyzed_T, analyzed_C = n_T, n_C
    cluster_dof = _cluster_dof(assigned_T, assigned_C, baseline) if clustered else None
    plan = _ArmPlan(procedure, baseline, analyzed_T, analyzed_C, cluster_dof, bounded)

    # Achieved power at this n, for the SUPPLIED effect.
    if looks is not None:
        power, expected_t = _sequential_estimates_from_log_se(
            looks,
            theta_dir,
            plan.log_se_sq(theta),
            alpha_seq,
            exit_side,
            e_value_dual=e_value_dual,
        )
    elif model is not None:
        power = binomial_power
        expected_t = None
    else:
        power = plan.power(distance, theta)
        expected_t = None

    # Companion MDE at this n, on the complier scale.
    if looks is not None:
        mde_relative, _, mde_unavailable_reason = _sequential_companion(
            looks,
            plan,
            design=design,
            alpha_seq=alpha_seq,
            exit_side=exit_side,
            e_value_dual=e_value_dual,
        )
    else:
        mde_relative, mde_unavailable_reason = _companion_mde(plan, design, decision, model)

    triggered = baseline.trigger_rate < 1.0
    n_total = assigned_T + assigned_C
    k_per_arm, k_total = _cluster_counts(assigned_T, assigned_C, baseline)
    planned_metric_name, planned_quantile = _planned_quantile_fields(baseline)
    return PowerResult(
        n_per_arm=assigned_T,
        n_total=n_total,
        power=min(power, 1.0),
        power_basis="asymptotic" if model is None else model.basis,
        mde_relative=mde_relative,
        mde_unavailable_reason=mde_unavailable_reason,
        effective_var=baseline.effective_var,
        n_clusters_per_arm=k_per_arm,
        n_clusters_total=k_total,
        n_triggered_per_arm=(round(analyzed_T) if clustered else n_T) if triggered else None,
        n_triggered_total=(round(analyzed_T + analyzed_C) if clustered else (n_T + n_C))
        if triggered
        else None,
        expected_n_total=_expected_n_total(expected_t, n_total),
        inference_to_declare=(
            InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=n_total)
            if isinstance(procedure.inference, GaussianScoreMixture)
            else None
        ),
        planned_metric_name=planned_metric_name,
        planned_quantile=planned_quantile,
    )


def achieved_power(
    n_per_arm: int,
    relative_lift: float,
    baseline: Baseline,
    procedure: ArmPlanningProcedure,
    design: PowerDesign | None = None,
    *,
    planned_looks: int | None = None,
) -> PowerResult:
    """Compute achieved power at a fixed arm size under a planning procedure.

    ``power`` describes the supplied ``relative_lift``. For eligible binomial
    plans, ``power_basis="exact"`` integrates the runtime decision's rejection
    probability at the analyzed counts; ``"approximate"`` instead integrates
    the Normal-conditional-tail decision model and certifies only that model.
    Other plans use the log-ratio model with treatment-arm variance evaluated
    at the alternative (``"asymptotic"``). ``mde_relative`` is the companion
    minimum detectable effect at the same size and target under the same
    model; when none exists there it is ``None`` with
    ``mde_unavailable_reason`` set, and the supplied-effect answer stands.
    A binomial plan whose decision the runtime refuses in full (an arm above
    its ceiling, a nuisance budget below the endpoint solver's floor, a tail
    level its float margin dominates) decides no count pair and is refused
    with a ``power.binomial_*`` code, not given a power.
    The look schedule resolves as in ``required_sample_size``.
    """
    return _achieved_power(
        n_per_arm, relative_lift, baseline, procedure, design, planned_looks, cache={}
    )


def _achieved_power(
    n_per_arm: int,
    relative_lift: float,
    baseline: Baseline,
    procedure: ArmPlanningProcedure,
    design: PowerDesign | None,
    planned_looks: int | None,
    *,
    cache: _GeometryCache,
) -> PowerResult:
    procedure, baseline = _prepare_solver(procedure, baseline)
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    if design is None:
        design = PowerDesign()
    else:
        design = PowerDesign.model_validate(design)
    looks = _planned_looks(procedure, planned_looks)
    bounded = _bounded_preflight(procedure, baseline, null_lift=decision.null_lift)
    _validate_per_arm_floor(n_per_arm, procedure, baseline)
    theta, distance = _supplied_effect(procedure, baseline, relative_lift, bounded=bounded)
    n_T, n_C = _compute_arms(
        n_per_arm, design, minimum_per_arm=_assigned_minimum_per_arm(procedure, baseline)
    )
    analyzed_T = n_T * baseline.trigger_rate
    analyzed_C = n_C * baseline.trigger_rate
    alpha_seq, exit_side, e_value_dual = _sequential_side(procedure)
    theta_dir = _sequential_drift(procedure, distance)

    model: _BinomialPlan | None = None
    if _runtime_binomial(procedure):
        analyzed_T, analyzed_C = _analyzed_counts(n_T, n_C, baseline)
        model = _binomial_plan(procedure, baseline, analyzed_T, analyzed_C, cache)
    clustered = baseline.avg_cluster_size > 1.0 and looks is None
    cluster_dof = _cluster_dof(n_T, n_C, baseline) if clustered else None
    plan = _ArmPlan(procedure, baseline, analyzed_T, analyzed_C, cluster_dof, bounded)
    if looks is not None:
        power, expected_t = _sequential_estimates_from_log_se(
            looks,
            theta_dir,
            plan.log_se_sq(theta),
            alpha_seq,
            exit_side,
            e_value_dual=e_value_dual,
        )
    elif model is not None:
        power = model.supplied(theta).power
        expected_t = None
    else:
        power = plan.power(distance, theta)
        expected_t = None

    if looks is not None:
        mde_relative, _, mde_unavailable_reason = _sequential_companion(
            looks,
            plan,
            design=design,
            alpha_seq=alpha_seq,
            exit_side=exit_side,
            e_value_dual=e_value_dual,
        )
    else:
        mde_relative, mde_unavailable_reason = _companion_mde(plan, design, decision, model)

    triggered = baseline.trigger_rate < 1.0
    k_per_arm, k_total = _cluster_counts(n_T, n_C, baseline)
    n_total = n_T + n_C
    planned_metric_name, planned_quantile = _planned_quantile_fields(baseline)
    return PowerResult(
        n_per_arm=n_T,
        n_total=n_total,
        power=min(power, 1.0),
        power_basis="asymptotic" if model is None else model.basis,
        mde_relative=mde_relative,
        mde_unavailable_reason=mde_unavailable_reason,
        effective_var=baseline.effective_var,
        n_clusters_per_arm=k_per_arm,
        n_clusters_total=k_total,
        n_triggered_per_arm=round(analyzed_T) if triggered else None,
        n_triggered_total=round(analyzed_T + analyzed_C) if triggered else None,
        expected_n_total=_expected_n_total(expected_t, n_total),
        inference_to_declare=(
            InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=n_total)
            if isinstance(procedure.inference, GaussianScoreMixture)
            else None
        ),
        planned_metric_name=planned_metric_name,
        planned_quantile=planned_quantile,
    )


def minimum_detectable_effect(
    n_per_arm: int,
    baseline: Baseline,
    procedure: ArmPlanningProcedure,
    design: PowerDesign | None = None,
    *,
    planned_looks: int | None = None,
) -> PowerResult:
    """Compute the minimum detectable relative effect at a fixed arm size.

    The answer is the first admissible complier-scale effect reaching
    ``design.power`` under the planning model named by ``power_basis`` (see
    ``achieved_power``); ``power`` is that effect's own power, which exceeds
    the target when the answer is the admissible interval's lower endpoint.
    For a runtime-binomial plan the answer is the first effect whose power is
    certified to reach the target (the lower end of its numerical enclosure
    does, so ``power`` exceeds the target by about that error), whatever the
    shape of power along the effects: every earlier candidate is excluded by
    its own power or by the monotone closure of the decision's rejection set,
    a bound on the rejection probability over an interval of effects, except
    those whose power lies within the enclosure of the target, so a band of
    effects that reaches it is found whether or not the largest admissible
    effect does. The enclosure is of the numerical integration of the decision
    set the route replays: for ``power_basis="exact"`` that is the runtime's
    own decision, for ``"approximate"`` its Normal-tail model, so "certified"
    and "unattainable" there describe that model and are not bounds on the
    runtime. A target no admissible alternative can reach is ``unattainable``;
    when no effect certifies it and the bound cannot exclude every effect, it
    lies within the enclosure of the greatest power any effect may reach and
    can neither be certified nor ruled out (``numerical_resolution``, whose
    ``power_enclosure`` is that enclosure and ``unresolved_interval`` the
    effects the bound could not exclude). A target whose answer has no float64
    representation is refused with a
    ``power.minimum_detectable_effect.*`` code. The look schedule resolves
    as in ``required_sample_size``. For fixed-horizon inference, a target at
    or below the null's own crossing probability is also refused: zero
    distance already qualifies, so no strictly nonzero minimum detectable
    effect exists.
    """
    return _minimum_detectable_effect(
        n_per_arm, baseline, procedure, design, planned_looks, cache={}
    )


def _minimum_detectable_effect(
    n_per_arm: int,
    baseline: Baseline,
    procedure: ArmPlanningProcedure,
    design: PowerDesign | None,
    planned_looks: int | None,
    *,
    cache: _GeometryCache,
) -> PowerResult:
    procedure, baseline = _prepare_solver(procedure, baseline)
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    if design is None:
        design = PowerDesign()
    else:
        design = PowerDesign.model_validate(design)
    looks = _planned_looks(procedure, planned_looks)
    bounded = _bounded_preflight(procedure, baseline, null_lift=decision.null_lift)
    _validate_per_arm_floor(n_per_arm, procedure, baseline)
    n_T, n_C = _compute_arms(
        n_per_arm, design, minimum_per_arm=_assigned_minimum_per_arm(procedure, baseline)
    )
    analyzed_T = n_T * baseline.trigger_rate
    analyzed_C = n_C * baseline.trigger_rate
    model: _BinomialPlan | None = None
    if _runtime_binomial(procedure):
        analyzed_T, analyzed_C = _analyzed_counts(n_T, n_C, baseline)
        model = _binomial_plan(procedure, baseline, analyzed_T, analyzed_C, cache)
    clustered = baseline.avg_cluster_size > 1.0 and looks is None
    cluster_dof = _cluster_dof(n_T, n_C, baseline) if clustered else None
    plan = _ArmPlan(procedure, baseline, analyzed_T, analyzed_C, cluster_dof, bounded)

    if looks is not None:
        alpha_seq, exit_side, e_value_dual = _sequential_side(procedure)
        sequential = _solve_sequential_mde(
            looks,
            plan,
            target=design.power,
            null_lift=decision.null_lift,
            alternative=decision.alternative,
            alpha_seq=alpha_seq,
            exit_side=exit_side,
            e_value_dual=e_value_dual,
        )
        if isinstance(sequential, _MdeRefusal):
            refuse(sequential.spec, **sequential.context)
        mde_relative, power = sequential.mde_relative, sequential.power
        # The answer's own enclosure carries E[T] at its alternative's variance.
        expected_t = sequential.enclosure.expected_fraction
    else:
        solved = _fixed_mde(plan, model, target=design.power, decision=decision)
        if isinstance(solved, _MdeRefusal):
            refuse(solved.spec, **solved.context)
        mde_relative, power = solved
        if mde_relative == 0.0:
            _raise(
                "power.minimum_detectable_effect.design_search_minimum",
                estimate=power,
                target=design.power,
            )
        expected_t = None

    triggered = baseline.trigger_rate < 1.0
    k_per_arm, k_total = _cluster_counts(n_T, n_C, baseline)
    n_total = n_T + n_C
    planned_metric_name, planned_quantile = _planned_quantile_fields(baseline)
    return PowerResult(
        n_per_arm=n_T,
        n_total=n_total,
        power=min(power, 1.0),
        power_basis="asymptotic" if model is None else model.basis,
        mde_relative=mde_relative,
        effective_var=baseline.effective_var,
        n_clusters_per_arm=k_per_arm,
        n_clusters_total=k_total,
        n_triggered_per_arm=round(analyzed_T) if triggered else None,
        n_triggered_total=round(analyzed_T + analyzed_C) if triggered else None,
        expected_n_total=_expected_n_total(expected_t, n_total),
        inference_to_declare=(
            InferenceSpec(kind="asymptotic_mean", expected_decision_sample_size=n_total)
            if isinstance(procedure.inference, GaussianScoreMixture)
            else None
        ),
        planned_metric_name=planned_metric_name,
        planned_quantile=planned_quantile,
    )


@dataclass(frozen=True, slots=True)
class _GridQuery:
    """One scalar solve of a power grid: power for ``relative_lift``, or the
    minimum detectable effect when it is ``None``."""

    n_per_arm: int
    relative_lift: float | None
    procedure: ArmPlanningProcedure
    design: PowerDesign


def _solve_together(
    queries: Sequence[_GridQuery], *, baseline: Baseline, planned_looks: int | None
) -> list[PowerResult]:
    """Solve ``queries`` in order, each exactly as its public solver would,
    reusing one runtime-binomial rejection geometry (and its solved companion
    effects) across every query with the same size and procedure."""
    cache: _GeometryCache = {}
    return [
        _achieved_power(
            query.n_per_arm,
            query.relative_lift,
            baseline,
            query.procedure,
            query.design,
            planned_looks,
            cache=cache,
        )
        if query.relative_lift is not None
        else _minimum_detectable_effect(
            query.n_per_arm, baseline, query.procedure, query.design, planned_looks, cache=cache
        )
        for query in queries
    ]


# Pairwise segment-difference solvers - detect a difference between two
# segments' lifts; see segment_pairwise_required_sample_size for the formula.


def _segment_arm_sizes(q: float, n_total: float, design: PowerDesign) -> tuple[int, int]:
    """Split a segment share into treatment/control arms.

    Every derived arm holds at least two units, so the allocation-aware
    segment threshold is ``2 / min(allocation, 1 - allocation)``; below it,
    flooring the arms would fabricate units not present in the segment.
    """
    n_total_seg = q * n_total
    min_alloc = min(design.allocation, 1.0 - design.allocation)
    min_n_total_seg = 2 / min_alloc
    if n_total_seg < min_n_total_seg:
        _raise(
            "power.segment_share_n",
            q=q,
            n_total=n_total,
            n_total_seg=n_total_seg,
            allocation=design.allocation,
            min_n_total_seg=min_n_total_seg,
        )
    n_t = max(2, math.ceil(n_total_seg * design.allocation))
    n_c = max(2, math.ceil(n_total_seg * (1.0 - design.allocation)))
    return n_t, n_c


def _pairwise_theta(r_a: float, r_b: float) -> float:
    _validate_relative_lift(r_a)
    _validate_relative_lift(r_b)
    return math.log1p(r_a) - math.log1p(r_b)


def _validate_segment_shares(q_a: float, q_b: float) -> None:
    if not 0 < q_a < 1:
        _raise("power.q_a")
    if not 0 < q_b < 1:
        _raise("power.q_b")
    if q_a + q_b > 1.0 + 1e-9:
        _raise("power.q_a_q", q_a_plus_q_b=q_a + q_b)


def _validate_pairwise_baselines(baseline_a: Baseline, baseline_b: Baseline) -> None:
    unsupported = [
        field
        for field in ("compliance", "trigger_rate")
        if any(getattr(baseline, field) != 1.0 for baseline in (baseline_a, baseline_b))
    ]
    if unsupported:
        _raise("power.segment_pairwise_solvers", unsupported=unsupported)


def _validate_pairwise_cluster_floor(
    n_t: int, n_c: int, baseline: Baseline, *, segment: str
) -> None:
    """Require the analyzer's structural minimum of two clusters per arm."""
    if baseline.avg_cluster_size <= 1.0:
        return
    k_t, k_total = _cluster_counts(n_t, n_c, baseline)
    assert k_t is not None and k_total is not None
    k_c = k_total - k_t
    if k_t < 2 or k_c < 2:
        _raise(
            "power.segment_clustered_baseline",
            segment=segment,
            k_total=k_total,
            n_t=n_t,
            n_c=n_c,
        )


def _pairwise_se_sq(
    n_total: float,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    baseline_b: Baseline,
    design: PowerDesign,
) -> float:
    """Variance of the segment-A-minus-segment-B log-lift contrast at total N."""
    n_a_t, n_a_c = _segment_arm_sizes(q_a, n_total, design)
    n_b_t, n_b_c = _segment_arm_sizes(q_b, n_total, design)
    _validate_pairwise_cluster_floor(n_a_t, n_a_c, baseline_a, segment="A")
    _validate_pairwise_cluster_floor(n_b_t, n_b_c, baseline_b, segment="B")
    return _se_sq(n_a_t, n_a_c, baseline_a) + _se_sq(n_b_t, n_b_c, baseline_b)


def _prepare_pairwise(
    procedure: ArmPlanningProcedure,
    baseline_a: Baseline,
    baseline_b: Baseline | None,
    design: PowerDesign | None,
    *,
    q_a: float,
    q_b: float,
    caller: str,
) -> tuple[ArmPlanningProcedure, Baseline, Baseline, PowerDesign, RelativeDecisionPolicy]:
    """Shared preamble for the three segment-pairwise solvers: baseline
    prep, design defaulting, and the fixed-horizon/segment-share guards.

    A quantile metric is refused with the readout's own breakout code: its
    segments have no readout to plan for, and a ``QuantileBaseline``
    carries no per-unit variance the segment model could scale."""
    quantile_baseline = next(
        (b for b in (baseline_a, baseline_b) if isinstance(b, QuantileBaseline)), None
    )
    if quantile_baseline is not None or (
        isinstance(procedure, ArmPlanningProcedure) and procedure.metric.metric_type == "quantile"
    ):
        refuse(
            READOUT_REFUSALS["readout.metric.quantile_breakout"],
            metric=None if quantile_baseline is None else quantile_baseline.metric_name,
            solver=caller,
            route=(
                "plan the whole-population quantile lift with required_sample_size, "
                "achieved_power or minimum_detectable_effect"
            ),
        )
    procedure, baseline_a = _prepare_solver(procedure, baseline_a)
    if baseline_b is None:
        baseline_b = baseline_a
    else:
        _, baseline_b = _prepare_solver(procedure, baseline_b)
    _validate_pairwise_baselines(baseline_a, baseline_b)
    if design is None:
        design = PowerDesign()
    else:
        design = PowerDesign.model_validate(design)
    if not isinstance(procedure.inference, FixedInference):
        _raise("power.supports_fixed_horizon", caller=caller)
    _validate_segment_shares(q_a, q_b)
    decision = cast("RelativeDecisionPolicy", procedure.decision)
    return procedure, baseline_a, baseline_b, design, decision


def _reraise_cluster_floor(exc: ValueError) -> None:
    """The cluster-floor refusal keeps its own code; other split failures
    become the calling solver's own refusal."""
    if isinstance(exc, CodedError) and exc.code == "power.segment_clustered_baseline":
        raise exc


def segment_pairwise_required_sample_size(
    r_a: float,
    r_b: float,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    procedure: ArmPlanningProcedure,
    baseline_b: Baseline | None = None,
    design: PowerDesign | None = None,
) -> PowerResult:
    """Experiment-wide N needed to detect segment A's lift differing from segment B's.

    ``r_a``/``r_b`` are the two segments' relative lifts; ``q_a``/``q_b``
    are their shares of the *whole* experiment (a 10% segment out of ten
    still has ``q=0.10`` regardless of how many other segments exist).
    Cost is ``n_total = n_ATE(delta) * (1/q_a + 1/q_b)``, where
    ``delta = log1p(r_a) - log1p(r_b)`` is the log-scale contrast (not
    ``log1p(r_a - r_b)``: that errs -27.7% at (0.50, 0.20) and +19.4% at
    (0.30, -0.10)), and ``n_ATE(delta)`` is the total N a standard
    50/50-allocation experiment would need to detect ``delta`` as a plain
    ATE. ``1/q_a + 1/q_b`` reduces to the simpler
    ``n_ATE(delta)/(q(1-q))`` form only when ``q_a + q_b = 1`` (a
    2-segment breakout); with more segments the two forms diverge and the
    simpler one understates N.

    ``n_per_arm``/``n_total`` are experiment-wide (summed across every
    segment, not just A and B). ``effective_var`` reflects segment A's
    baseline only. ``baseline_b`` defaults to ``baseline_a``;
    ``design.allocation`` governs the treatment/control split within
    each segment. Cluster design effects flow in via each baseline's
    ``effective_var``, but ``PowerResult.n_clusters_*`` stays ``None``:
    no single cluster count is meaningful across two possibly-different
    baselines. Fixed-horizon only; sequential planning for segment
    contrasts is not yet supported.

    Raises
    ------
    ValueError
        If ``r_a`` and ``r_b`` give the same relative lift (theta=0), the
        same refusal ``required_sample_size`` makes for a lift exactly
        at its null boundary. Also raised if the solved-for N is too
        small for either segment's 2-arm split (see ``_segment_arm_sizes``).
    """
    procedure, baseline_a, baseline_b, design, decision = _prepare_pairwise(
        procedure,
        baseline_a,
        baseline_b,
        design,
        q_a=q_a,
        q_b=q_b,
        caller="segment_pairwise_required_sample_size",
    )
    theta = _pairwise_theta(r_a, r_b)
    if decision.null_lift != 0.0:
        _raise("power.segment_pairwise_required", null_lift=decision.null_lift)
    alternative = decision.alternative
    if alternative == "greater" and theta < 0.0:
        _raise("power.r_a_below", r_a=r_a, r_b=r_b, theta=theta)
    if alternative == "less" and theta > 0.0:
        _raise("power.r_a_above", r_a=r_a, r_b=r_b, theta=theta)

    k = 1.0 / design.allocation + 1.0 / (1.0 - design.allocation)
    term_a = baseline_a.effective_var / (baseline_a.mean**2 * q_a)
    term_b = baseline_b.effective_var / (baseline_b.mean**2 * q_b)
    z_sum = float(_norm.isf(procedure.compiled_tail_alpha)) + float(_norm.ppf(design.power))

    if abs(theta) < 1e-15:
        _raise("power.r_a_r", r_a=r_a, r_b=r_b)

    n_total_float = k * z_sum**2 * (term_a + term_b) / theta**2
    n_t = max(2, math.ceil(n_total_float * design.allocation))
    n_t, n_c = _compute_arms(n_t, design)

    try:
        se2 = _pairwise_se_sq(n_t + n_c, q_a, q_b, baseline_a, baseline_b, design)
    except ValueError as exc:
        _reraise_cluster_floor(exc)
        _raise("power.solved_too_small", n_total=n_t + n_c, q_a=q_a, q_b=q_b, exc=str(exc))
    return PowerResult(
        n_per_arm=n_t,
        n_total=n_t + n_c,
        power=min(_power_at(theta, se2, design, procedure), 1.0),
        power_basis="asymptotic",
        mde_relative=_mde_relative(se2, design, procedure),
        effective_var=baseline_a.effective_var,
    )


def segment_pairwise_achieved_power(
    n_per_arm: int,
    r_a: float,
    r_b: float,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    procedure: ArmPlanningProcedure,
    baseline_b: Baseline | None = None,
    design: PowerDesign | None = None,
) -> PowerResult:
    """Achieved power for detecting segment A's lift differing from segment B's
    at an experiment-wide treatment-arm size of ``n_per_arm``.

    ``PowerResult.effective_var`` reflects segment A's baseline only; it
    does not summarize ``baseline_b``. Fixed-horizon only; sequential
    planning for segment contrasts is not yet supported.

    Raises
    ------
    ValueError
        If ``n_per_arm`` implies too few units in either segment's share
        for a 2-arm split (see ``_segment_arm_sizes``); increase
        ``n_per_arm`` or the smaller segment's share.
    """
    procedure, baseline_a, baseline_b, design, decision = _prepare_pairwise(
        procedure,
        baseline_a,
        baseline_b,
        design,
        q_a=q_a,
        q_b=q_b,
        caller="segment_pairwise_achieved_power",
    )
    if decision.null_lift != 0.0:
        _raise("power.segment_pairwise_achieved", null_lift=decision.null_lift)
    theta = _pairwise_theta(r_a, r_b)

    n_t, n_c = _compute_arms(n_per_arm, design)
    try:
        se2 = _pairwise_se_sq(n_t + n_c, q_a, q_b, baseline_a, baseline_b, design)
    except ValueError as exc:
        _reraise_cluster_floor(exc)
        _raise(
            "power.segment_pairwise_achieved_n_per_arm_too_small",
            n_per_arm=n_per_arm,
            q_a=q_a,
            q_b=q_b,
            exc=str(exc),
        )
    return PowerResult(
        n_per_arm=n_t,
        n_total=n_t + n_c,
        power=min(_power_at(theta, se2, design, procedure), 1.0),
        power_basis="asymptotic",
        mde_relative=_mde_relative(se2, design, procedure),
        effective_var=baseline_a.effective_var,
    )


def segment_pairwise_minimum_detectable_effect(
    n_per_arm: int,
    q_a: float,
    q_b: float,
    baseline_a: Baseline,
    procedure: ArmPlanningProcedure,
    baseline_b: Baseline | None = None,
    design: PowerDesign | None = None,
) -> PowerResult:
    """Smallest segment-A-vs-segment-B difference detectable at ``n_per_arm``.

    ``mde_relative`` is ``exp(delta) - 1`` for the smallest detectable
    log-scale contrast ``delta = log(1+r_A) - log(1+r_B)``: the smallest
    detectable ratio ``(1+r_A)/(1+r_B) - 1``, not a lift against a
    single baseline mean.

    ``PowerResult.effective_var`` reflects segment A's baseline only; it
    does not summarize ``baseline_b``. Fixed-horizon only; sequential
    planning for segment contrasts is not yet supported.

    Raises
    ------
    ValueError
        If ``n_per_arm`` implies too few units in either segment's share
        for a 2-arm split (see ``_segment_arm_sizes``); increase
        ``n_per_arm`` or the smaller segment's share.
    """
    procedure, baseline_a, baseline_b, design, decision = _prepare_pairwise(
        procedure,
        baseline_a,
        baseline_b,
        design,
        q_a=q_a,
        q_b=q_b,
        caller="segment_pairwise_minimum_detectable_effect",
    )
    if decision.null_lift != 0.0:
        _raise("power.segment_pairwise_minimum", null_lift=decision.null_lift)

    n_t, n_c = _compute_arms(n_per_arm, design)
    try:
        se2 = _pairwise_se_sq(n_t + n_c, q_a, q_b, baseline_a, baseline_b, design)
    except ValueError as exc:
        _reraise_cluster_floor(exc)
        _raise(
            "power.segment_pairwise_achieved_n_per_arm_too_small",
            n_per_arm=n_per_arm,
            q_a=q_a,
            q_b=q_b,
            exc=str(exc),
        )
    mde_theta = _mde_theta(se2, design, procedure)
    return PowerResult(
        n_per_arm=n_t,
        n_total=n_t + n_c,
        power=min(_power_at(mde_theta, se2, design, procedure), 1.0),
        power_basis="asymptotic",
        mde_relative=math.expm1(mde_theta),
        effective_var=baseline_a.effective_var,
    )


# Cochran's Q (increment.estimation.meta.cochran_q) tests whether K
# segments' effects are equal; see each solver below for its exact/approximate power formula.


def joint_q_power_fixed(
    theta: Sequence[float] | np.ndarray,
    var: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Exact power of Cochran's Q to detect FIXED, named per-segment deviations.

    Parameters
    ----------
    theta : array-like of float
        Each segment's true log-scale effect (or any per-segment quantity
        Q is computed over). Only relative differences matter: a common
        shift added to every ``theta_k`` does not change the result.
    var : array-like of float
        Each segment's sampling variance, same length and order as ``theta``.
    alpha : float
        Significance level for Cochran's Q test (default 0.05).

    Returns
    -------
    float
        Power = P(Q > chi2_crit(K-1, alpha)) under the noncentral chi2(K-1,
        lambda) distribution Q follows exactly at these ``theta``/``var``.

    Fixed-horizon only; sequential planning for segment contrasts is not
    yet supported.

    Verified against a 100k-rep simulation at a 5-segment unequal-share
    fixture: predicted 0.0533 vs. empirical 0.0543 (see
    ``tests/power/test_core.py``).
    """
    theta_arr = np.asarray(theta, dtype=float)
    var_arr = np.asarray(var, dtype=float)
    k = _validate_joint_q_inputs(theta_arr, var_arr, alpha)

    w = 1.0 / var_arr
    # theta_bar_w must be the precision-weighted mean: any other centre
    # strictly overstates lambda (lambda_w = min_c sum_k w_k (theta_k - c)^2).
    theta_bar_w = float((w * theta_arr).sum() / w.sum())
    lam = float((w * (theta_arr - theta_bar_w) ** 2).sum())
    crit = _chi2.isf(alpha, k - 1)
    return float(_ncx2.sf(crit, k - 1, lam))


def joint_q_power_random(
    tau_b: float,
    var: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Power of Cochran's Q under a random-effects model of segment spread.

    Segment effects are modelled as ``theta_k ~ iid N(mu, tau_b^2)``: the
    design-time question "if segments typically differ by about
    ``tau_b``, what's my power to detect that?", vs.
    ``joint_q_power_fixed``'s "if they differ by exactly these amounts".

    Parameters
    ----------
    tau_b : float
        Standard deviation of segment effects around their common mean
        (same scale as ``joint_q_power_fixed``'s ``theta``).
    var : array-like of float
        Each segment's sampling variance.
    alpha : float
        Significance level for Cochran's Q test (default 0.05).

    Returns
    -------
    float
        Power estimate. Exact when every ``var`` entry is equal: Q is
        then a scaled central chi-square, ``Q ~ (1 + tau_b^2/v) *
        chi2(K-1)`` (verified against a 100k-rep simulation: 0.1078
        exact vs. 0.1099 empirical at K=5, v=1.0, tau_b=0.5).

        Approximate for unequal ``var``: Q's true distribution is a
        generalised (Satterthwaite-type) weighted sum of independent
        central chi2(1) variables, not a noncentral chi-square exactly.
        This returns a mean-matched noncentral-chi2(K-1, E[lambda])
        approximation, ``E[lambda] = tau_b^2 * (sum(w) - sum(w^2)/sum(w))``
        (the same building block as ``cochran_q``'s DerSimonian-Laird
        denominator). Measured against 60k-rep simulation across 7
        configurations (K=5-10, per-segment variance ratios 1x-50x):
        relative error within +/-4% up to an 8x ratio, growing to +12.8%
        at a 50x ratio - treat as an approximation, not a calibrated
        design tool, when segment variances are wildly unequal.

    Fixed-horizon only; sequential planning for segment contrasts is not
    yet supported.
    """
    var_arr = np.asarray(var, dtype=float)
    k = _validate_joint_q_inputs(np.zeros_like(var_arr), var_arr, alpha)
    _require_finite("tau_b", tau_b)
    if tau_b < 0:
        _raise("power.tau_b", tau_b=tau_b)

    crit = _chi2.isf(alpha, k - 1)
    if np.ptp(var_arr) < 1e-12 * var_arr.max():
        # Equal variances -> exact scaled-central-chi2 closed form.
        v0 = float(var_arr[0])
        scale = 1.0 + tau_b**2 / v0
        return float(_chi2.sf(crit / scale, k - 1))

    w = 1.0 / var_arr
    lam = float(tau_b**2 * (w.sum() - (w**2).sum() / w.sum()))
    return float(_ncx2.sf(crit, k - 1, lam))


def _validate_joint_q_inputs(theta: np.ndarray, var: np.ndarray, alpha: float) -> int:
    if theta.ndim != 1 or var.ndim != 1:
        _raise("power.theta_var_one", theta_shape=theta.shape, var_shape=var.shape)
    if theta.shape != var.shape:
        _raise("power.theta_var_same", theta_shape=theta.shape, var_shape=var.shape)
    k = theta.shape[0]
    if k < 2:
        _raise("power.need_least_segments", k=k)
    if not np.all(np.isfinite(var)) or np.any(var <= 0):
        _raise("estimation.meta.var_finite_strictly")
    if not np.all(np.isfinite(theta)):
        _raise("power.theta_contains_non")
    if not 0.0 < alpha < 1.0:
        _raise("estimation.diagnostics.alpha", alpha=alpha)
    return k
