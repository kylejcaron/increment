"""Estimate and LiftEstimate: the value-object types for all estimation output.

``Estimate`` is a point value with an optional interval. ``LiftEstimate``
wraps one and carries the metadata to identify which (metric, method,
group) it describes, on the relative scale by default
(``value_scale="relative"``) or the additive scale for an encouragement
design's LATE estimand or an additively-reported observational metric
(``value_scale="absolute"``; see the class docstring).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.stats import norm as _norm
from scipy.stats import t as _t

from increment._literals import (
    ALTERNATIVE_VALUES,
    Alternative,
    PreferredDirection,
    RowRole,
    ValueScale,
)
from increment.errors import (
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)
from increment.estimation._student_t import _log_tail_bounds
from increment.estimation._tails import resolvable_expm1, student_t_isf, tail_isf, wald_bounds
from increment.estimation.armstats import IndependentMeanReference
from increment.estimation.priors import MixturePrior, StudentTPrior
from increment.estimation.sequential_result import (
    AsymptoticSequentialResult,
    SequentialInferenceResult,
    SequentialResult,
)
from increment.winsor import BootstrapReference, WinsorConfidenceSet, winsor_refuse

RelativeUnavailableReason = Literal[
    "joint_covariance_indefinite",
    "joint_covariance_unrepresentable",
    "nonpositive_arm_mean",
    "zero_relative_variance",
]


if TYPE_CHECKING:
    from increment.estimation.binomial_rr import BinomialInterval
    from increment.estimation.inference import LiftPosterior, Normal


class Estimate(CodedModel, BaseModel):
    """A number with quantified uncertainty.

    ``lb``/``ub`` (when set) represent an interval at ``level`` confidence.
    ``open_side`` names an unbounded endpoint of a one-sided interval.

    Parameters
    ----------
    value : float
        Point estimate.
    lb, ub : float | None
        Interval bounds. Closed intervals set both; open intervals leave the
        endpoint named by ``open_side`` unset.
    open_side : {"lower", "upper"} | None
        Unbounded endpoint for a one-sided interval, or ``None`` for a closed
        or unavailable interval.
    level : float | None
        Confidence/credible level, e.g. 0.95. Required iff ``lb``/``ub`` are set.
        At extreme alpha values this may round to 1.0; ``alpha`` retains the
        exact allocated noncoverage budget.
    alpha : float | None
        Allocated noncoverage budget; level is its nominal complement.
        Fixed open endpoints use this full quantile tail. Sequential open
        endpoints retain the symmetric parent budget, without a fixed-time
        tail interpretation. Attained coverage may be higher than nominal.
        Unlike ``level``, it remains recoverable when the confidence level
        rounds to 1.0.
    log_mean, log_se : float | None
        Point estimate and standard error on the natural-log scale. Set by
        ``infer_lift`` to the raw pre-prior statistics, not the posterior
        mean/sd - the two differ under an informative prior.
    """

    model_config = ConfigDict(frozen=True)

    value: float = Field(allow_inf_nan=False)
    lb: float | None = Field(default=None, allow_inf_nan=False)
    ub: float | None = Field(default=None, allow_inf_nan=False)
    open_side: Literal["lower", "upper"] | None = Field(default=None)
    level: float | None = None
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)
    log_mean: float | None = Field(default=None, allow_inf_nan=False)
    log_se: float | None = Field(default=None, allow_inf_nan=False)

    @model_validator(mode="after")
    def _lb_ub_paired(self):
        if self.open_side is not None:
            calibrated, open_field, open_name = (
                (self.ub, self.lb, "lb") if self.open_side == "lower" else (self.lb, self.ub, "ub")
            )
            if calibrated is None:
                _raise("estimation.results.estimate.open_side_needs_calibrated_bound")
            if open_field is not None:
                _raise(
                    "estimation.results.estimate.open_side_bound_must_be_none",
                    field=open_name,
                )
            if self.level is None or not (0.0 < self.level <= 1.0):
                _raise("estimation.results.estimate.level_interval", level=self.level)
            if self.level == 1.0 and self.alpha is None:
                _raise("estimation.results.estimate.level_alpha_because")
            if self.alpha is not None:
                expected_level = math.fsum((1.0, -self.alpha))
                if not math.isclose(
                    self.level,
                    expected_level,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ):
                    _raise(
                        "estimation.results.estimate.level_contradicts_effective",
                        expected_level=expected_level,
                        alpha=self.alpha,
                        level=self.level,
                    )
            return self
        if (self.lb is None) != (self.ub is None):
            _raise("estimation.results.estimate.lb_ub_both")
        if self.lb is None:
            if self.level is not None:
                _raise("estimation.results.estimate.level_none_lb")
            if self.alpha is not None:
                _raise("estimation.results.estimate.alpha_none_lb")
        else:
            lb, ub = self.lb, self.ub
            if self.level is None:
                _raise("estimation.results.estimate.level_lb_ub")
            if ub is None:
                _raise("estimation.results.estimate.lb_ub_both")
            if not (0.0 < self.level <= 1.0):
                _raise("estimation.results.estimate.level_interval", level=self.level)
            if self.level == 1.0 and self.alpha is None:
                _raise("estimation.results.estimate.level_alpha_because")
            if self.alpha is not None:
                expected_level = math.fsum((1.0, -self.alpha))
                if not math.isclose(self.level, expected_level, rel_tol=1e-12, abs_tol=1e-15):
                    _raise(
                        "estimation.results.estimate.level_contradicts_effective",
                        expected_level=expected_level,
                        alpha=self.alpha,
                        level=self.level,
                    )
            if lb > ub:
                _raise("estimation.results.estimate.interval_inverted_lb", lb=lb, ub=ub)
        return self

    def excludes(self, null: float = 0.0) -> bool:
        """Whether the interval lies entirely on one side of *null*.

        ``open_side="lower"`` uses only the finite upper bound;
        ``open_side="upper"`` uses only the finite lower bound.
        """
        if self.open_side == "lower":
            return self.ub is not None and self.ub < null
        if self.open_side == "upper":
            return self.lb is not None and self.lb > null
        if self.lb is None or self.ub is None:
            return False
        return self.lb > null or self.ub < null


BinomialMethod = Literal["binomial_bb_difference_v3"]
#: The construction `BinomialConfidenceSet` reads: its nuisance envelope, integer thresholds
#: and numerical stop rule. A producer stamps it on every set it cuts.
BINOMIAL_METHOD: BinomialMethod = "binomial_bb_difference_v3"


class BinomialConfidenceSet(CodedModel, BaseModel):
    """A typed relative-lift confidence set from the exact independent-
    binomial risk-ratio method (see ``binomial_rr.py``).

    Persisted on EVERY ``reference_kind="binomial"`` row -- including a
    point-backed one -- as the sufficient-count/method provenance
    ``LiftEstimate.stat_sig()``/``p_value()`` recompute a shifted-null
    verdict from; used as the row's ONLY confidence-set representation
    when no finite point exists (``x_c == 0``: the empirical ratio has a
    zero denominator), in which case ``LiftEstimate.lift`` is ``None``.

    ``lower``/``upper`` are on the LIFT scale (``R - 1``), matching
    ``Estimate.lb``/``ub`` elsewhere in this module. ``upper is None``
    means genuinely unbounded (no finite ceiling exists in the identified
    set, e.g. an observed zero control count) -- never serialized as
    infinity. ``lower`` is never ``None``: the relative-lift scale's
    natural floor, ``-1`` (``R = 0``), is always a legitimate finite
    value, attainable and closed.

    ``method`` is required and names the construction the endpoints were cut under, nuisance
    stop rule included: ``LiftEstimate.stat_sig()``/``p_value()`` recompute from the counts
    with the current rule, so a set cut under another one would sit beside a verdict its own
    interval can contradict. A set that names another construction, or none, is refused when
    read, not interpreted with the current rule. The marker is never a default: a payload
    dumped without its defaults would otherwise read as the current construction whatever
    cut it.
    """

    model_config = ConfigDict(frozen=True)

    lower: float = Field(allow_inf_nan=False)
    upper: float | None = Field(default=None, allow_inf_nan=False)
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    level: float = Field(gt=0.0, le=1.0)
    decision_alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    # Alpha passed to the exact inversion. Directional rows retain the
    # sitewide central-equivalent display alpha above, so these differ in
    # ordinary inference and coincide after directional FCR reinversion.
    geometry: Literal["central", "lower_bound", "upper_bound"]
    method: BinomialMethod
    x_c: int = Field(ge=0)
    n_c: int = Field(ge=1)
    x_t: int = Field(ge=0)
    n_t: int = Field(ge=1)
    nuisance_beta: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def _cut_under_the_current_construction(cls, data: Any) -> Any:
        recorded = (
            data.get("method") if isinstance(data, Mapping) else getattr(data, "method", None)
        )
        if recorded != BINOMIAL_METHOD:
            _raise(
                "estimation.results.binomial.obsolete_construction",
                method=recorded
                if recorded is None or isinstance(recorded, str)
                else repr(recorded),
                supported=BINOMIAL_METHOD,
            )
        return data

    @model_validator(mode="after")
    def _validate_binomial_set(self):
        from increment.estimation.binomial_rr import nuisance_beta as _nuisance_beta

        expected_level = math.fsum((1.0, -self.alpha))
        if not math.isclose(self.level, expected_level, rel_tol=0.0, abs_tol=1e-15):
            _raise(
                "estimation.results.binomial.decision_metadata",
                reason=f"level={self.level!r} does not match alpha={self.alpha!r}",
            )
        valid_decision_alphas = (
            (self.alpha,) if self.geometry == "central" else (self.alpha / 2.0, self.alpha)
        )
        if not any(
            math.isclose(self.decision_alpha, valid, rel_tol=0.0, abs_tol=0.0)
            for valid in valid_decision_alphas
        ):
            _raise(
                "estimation.results.binomial.decision_metadata",
                reason=(
                    f"decision_alpha={self.decision_alpha!r} is incompatible with "
                    f"alpha={self.alpha!r} and geometry={self.geometry!r}"
                ),
            )
        expected_beta = _nuisance_beta(self.decision_alpha)
        if self.nuisance_beta != expected_beta:
            _raise(
                "estimation.results.binomial.decision_metadata",
                reason=(
                    f"nuisance_beta={self.nuisance_beta!r} does not match "
                    f"decision_alpha={self.decision_alpha!r}"
                ),
            )
        if self.upper is not None and self.lower > self.upper:
            _raise(
                "estimation.results.binomial.interval_inverted",
                lower=self.lower,
                upper=self.upper,
            )
        if self.x_c > self.n_c or self.x_t > self.n_t:
            _raise(
                "estimation.results.binomial.counts_out_of_range",
                x_c=self.x_c,
                n_c=self.n_c,
                x_t=self.x_t,
                n_t=self.n_t,
            )
        if self.x_c == 0 and self.upper is not None:
            # A zero control count keeps q=0 in the nuisance domain for every R, so every
            # method-consistent upper endpoint is unbounded (binomial_rr.py's module
            # docstring): a finite upper is impossible, not tighter.
            _raise(
                "estimation.results.binomial.counts_out_of_range",
                x_c=self.x_c,
                n_c=self.n_c,
                x_t=self.x_t,
                n_t=self.n_t,
            )
        return self

    @property
    def point_available(self) -> bool:
        """Whether these counts admit a finite empirical-lift point (``x_c > 0``)."""
        return self.x_c > 0

    def null_p_value(self, null_lift: float, alternative: str) -> float:
        """The exact p-value for ``H0: R = 1 + null_lift`` under *alternative*, from these
        persisted counts and nuisance budget, refined against the level the verdict compares
        it with (``decision_alpha``, halved for the two-sided test): the recompute behind
        ``LiftEstimate.stat_sig()``/``p_value()`` and the tables' twin of them."""
        from increment.estimation.binomial_rr import null_p_value

        tail = self.decision_alpha / 2.0 if alternative == "two-sided" else self.decision_alpha
        return null_p_value(
            1.0 + null_lift,
            self.x_c,
            self.n_c,
            self.x_t,
            self.n_t,
            self.nuisance_beta,
            alternative=alternative,
            tail=tail,
        )


def _reference_fields(dof: float | None) -> dict[str, Any]:
    """Build the persisted reference triple from a legacy degrees of freedom."""
    return {
        "dof": dof,
        "reference_kind": "t" if dof is not None else "normal",
        "reference_df": dof,
    }


def _infer_reference_from_legacy_dof(data: Any) -> Any:
    """Recover only the released degrees-of-freedom wire fields.

    Independent-mean references are versioned contracts. Obsolete v1 payloads
    and incomplete references must be rejected instead of silently upgraded.
    """
    if not isinstance(data, dict):
        return data
    reference = data.get("independent_mean_reference")
    if isinstance(reference, dict):
        kind = reference.get("kind")
        missing = {"alpha", "alternative", "interval"} - reference.keys()
        if kind == "independent-means-welch-v1" or missing:
            winsor_refuse(
                "invalid_state",
                "Obsolete or incomplete independent mean reference; use v2 fields.",
            )
    if "reference_kind" in data or "reference_df" in data:
        return data
    if data.get("inference", "fixed") != "fixed":
        return {**data, "reference_kind": "sequential", "reference_df": None}
    dof = data.get("dof")
    if dof is None:
        return data
    return {**data, "reference_kind": "t", "reference_df": dof}


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.results.estimate.lb_ub_both": "lb and ub must both be set, or both be None",
        "estimation.results.estimate.level_none_lb": "level must be None when lb/ub are None",
        "estimation.results.estimate.alpha_none_lb": "alpha must be None when lb/ub are None",
        "estimation.results.estimate.level_lb_ub": "level is required when lb/ub are set",
        "estimation.results.estimate.level_interval": "level must be in (0, 1] for an interval, got {level}",
        "estimation.results.estimate.level_alpha_because": "level=1.0 requires alpha because the effective tail is unrepresentable from the rounded confidence level",
        "estimation.results.estimate.level_contradicts_effective": "level={level} contradicts effective alpha={alpha}; expected level={expected_level}",
        "estimation.results.estimate.interval_inverted_lb": "interval is inverted: lb={lb} > ub={ub}",
        "estimation.results.estimate.open_side_needs_calibrated_bound": "open_side is set but the calibrated bound on that side is None",
        "estimation.results.estimate.open_side_bound_must_be_none": "open_side is set; {field} must be None (never a numeric sentinel), got a finite value",
        "estimation.results.lift.open_side_alternative_mismatch": "open_side={open_side!r} does not match alternative={alternative!r}",
        "estimation.results.lift.posterior_decision_stats_sequential": "LiftEstimate for {metric!r}/{group_id!r}: decision stats are undefined for inference={inference!r} -- a sequential confidence sequence's radius is not a posterior standard deviation, so recovering (mu, sigma) from it would give a silently wrong probability. Only inference='fixed' (single-look) estimates support chance_to_beat/prob_beyond/prob_within/risk_if_shipped/p_value.",
        "estimation.results.lift.posterior_decision_stats_cluster_robust": RefusalSpec(
            "estimation.results.lift.posterior_decision_stats_cluster_robust",
            InvalidRequestError,
            lambda *, reference_df, group_id, metric: (
                f"LiftEstimate for {metric!r}/{group_id!r}: this decision statistic is "
                "unavailable for an asymptotic cluster sandwich; read the interval directly."
                if reference_df is None
                else f"LiftEstimate for {metric!r}/{group_id!r}: decision stats are "
                f"undefined for a cluster-robust t_{{{reference_df:g}}} reference -- the "
                "sandwich's relative evidence is a joint Fieller set, not a scalar "
                "(mu, sigma) pair to recover. Read relative_confidence_set directly, or "
                "decide with stat_sig()/p_value(), which honour this reference."
            ),
        ),
        "estimation.results.lift.t_posterior_sufficient_statistics": "LiftEstimate for {metric!r}/{group_id!r}: the posterior behind a t_{{{reference_df:g}}}-reference estimate is read from the working-scale moments its interval was cut from, and this row is missing {missing}. Its endpoints are t quantiles, so inverting them with a Normal z would report sigma inflated by the t/z ratio instead -- there is no fallback.",
        "estimation.results.lift.liftestimate_level_contradicts": "LiftEstimate for {metric!r}/{group_id!r}: level={e_level} contradicts effective alpha={e_alpha}; expected level={expected_level}",
        "estimation.results.lift.liftestimate_prior_spec": "LiftEstimate for {metric!r}/{group_id!r}: prior_spec is set but the raw log-scale sufficient statistics are missing -- not a genuine infer_lift output",
        "estimation.results.lift.liftestimate_carries_no": "LiftEstimate for {metric!r}/{group_id!r} carries no interval -- decision stats need lb/ub/level (run inference with an alpha, which is the default)",
        "estimation.results.lift.liftestimate_value_lb": "LiftEstimate for {metric!r}/{group_id!r}: value/lb <= -1 is not representable on the log scale (value={e_value}, lb={e_lb}) -- not a genuine infer_lift output",
        "estimation.results.lift.liftestimate_interval_symmetric": "LiftEstimate for {metric!r}/{group_id!r}: interval is not a symmetric Normal quantile interval on the {scale} scale -- decision stats are only defined for fixed-horizon estimates produced by infer_lift/infer_ate",
        "estimation.results.lift.threshold_representable_log": "threshold <= -1 is not representable on the log scale, got {threshold}",
        "estimation.results.lift.liftestimate_prob_favorable": "LiftEstimate for {metric!r}/{group_id!r}: prob_favorable() requires preferred_direction to be set -- it was not resolved from a Metric declaration for this estimate",
        "estimation.results.lift.prob_within_threshold_positive": "threshold must be > 0, got {threshold}",
        "estimation.results.lift.prob_within_threshold_unit_interval": "threshold must be in (0, 1), got {threshold}",
        "estimation.results.lift.liftestimate_chance_to": "LiftEstimate for {metric!r}/{group_id!r}: chance_to_beat_favorable() requires preferred_direction to be set -- it was not resolved from a Metric declaration for this estimate",
        "estimation.results.lift.liftestimate_risk_if": "LiftEstimate for {metric!r}/{group_id!r}: risk_if_shipped_favorable() requires preferred_direction to be set -- it was not resolved from a Metric declaration for this estimate",
        "estimation.results.lift.p_value_cluster_robust_null_abs": RefusalSpec(
            "estimation.results.lift.p_value_cluster_robust_null_abs",
            InvalidRequestError,
            lambda *, reference_df, group_id, metric: (
                f"LiftEstimate for {metric!r}/{group_id!r}: this decision statistic is "
                "unavailable for an asymptotic cluster sandwich; read the interval directly."
                if reference_df is None
                else f"LiftEstimate for {metric!r}/{group_id!r}: additive decision stats "
                f"are unavailable for a t_{{{reference_df:g}}}-reference estimate. "
                "A Normal additive tail is not justified by this reference; "
                "read an available additive interval directly or decide on the relative scale."
            ),
        ),
        "estimation.results.lift.p_value_null_abs_missing_abs_se": "LiftEstimate for {metric!r}/{group_id!r}: null_abs is set but abs_se is unavailable (degenerate arm) -- the additive decision is undefined and there is no silent fallback to the relative interval",
        "estimation.results.lift.liftestimate_dof_set": "LiftEstimate for {metric!r}/{group_id!r}: reference_kind='t' but this row carries no log-scale sufficient statistics (log_mean/log_se) -- p_value() via the t-reference formula is undefined for linear/absolute-scale estimates (e.g. infer_ate or clustered encouragement rows)",
        "estimation.results.lift.liftestimate_log_se": "LiftEstimate for {metric!r}/{group_id!r}: log_se must be positive to compute a t-reference p-value, got log_se={e_log_se}",
        "estimation.results.lift.p_value_null_lift_not_representable_dof": "LiftEstimate for {metric!r}/{group_id!r}: null_lift <= -1 is not representable on the log scale, got {null_lift}",
        "estimation.results.lift.reference_df_required_for_t": "estimate row for {metric!r}/{group_id!r}: reference_kind='t' requires a finite reference_df > 0, got {reference_df!r}",
        "estimation.results.lift.reference_df_set_for_normal": "estimate row for {metric!r}/{group_id!r}: reference_kind={reference_kind!r} requires reference_df=None, got {reference_df!r}",
        "estimation.results.lift.dof_reference_df_mismatch": "estimate row for {metric!r}/{group_id!r}: dof={dof!r} does not match reference_df={reference_df!r} -- a t-reference row's legacy dof column must agree with the reference it was actually cut from",
        "estimation.results.lift.inference_reference_kind_mismatch": "estimate row for {metric!r}/{group_id!r}: inference={inference!r} and reference_kind={reference_kind!r} disagree; a sequential interval has a sequential reference and a fixed-horizon interval a normal or t one",
        "estimation.results.lift.fcr_prior_unsupported": "directional FCR conversion requires a prior-free fixed reference",
        "estimation.results.lift.open_interval_unrecoverable": "open fixed intervals require stored alpha and, near alpha=0.5, raw sufficient statistics to recover uncertainty",
        "estimation.results.joint.invalid_reference": "Invalid joint relative reference: {reason}",
        "estimation.results.joint.invalid_set": "Invalid relative confidence set: {reason}",
        "estimation.results.joint.unavailable": "Joint relative inference is unavailable: {reason}",
        "estimation.results.binomial.interval_inverted": "binomial confidence set has lower={lower} > upper={upper}",
        "estimation.results.binomial.counts_out_of_range": "binomial confidence set counts out of range: x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}",
        "estimation.results.binomial.decision_metadata": "binomial confidence set has inconsistent decision metadata: {reason}",
        "estimation.results.binomial.obsolete_construction": "binomial confidence set records method {method!r} (None: none recorded); this version reads only {supported!r}. A set cut under another construction, or persisted without naming one, may hold endpoints from a different nuisance stop rule than the p-value recomputed from its counts now uses, so its interval and verdict could disagree. Re-run the analysis to cut the set again.",
        "estimation.results.binomial.posterior_unavailable": "metric={metric!r} group_id={group_id!r}: no Normal/lognormal posterior exists for a reference_kind='binomial' row -- the exact binomial method is a frequentist test-inversion, not a posterior; chance_to_beat/prob_beyond/prob_within/risk_if_shipped and their favorable variants are unavailable here. Use stat_sig()/p_value() (both binomial-set-aware) or the persisted lift/binomial_set bounds directly.",
        "estimation.results.lift.binomial_lift_availability": "metric={metric!r} group_id={group_id!r}: {reason}",
        "estimation.results.lift.absolute_reference_mismatch": "absolute reference {kind!r} requires df exactly for t, got {df!r}",
        "estimation.results.lift.absolute_alpha_without_interval": "abs_alpha={abs_alpha!r} is the alpha an additive interval was cut at, but this row carries no additive interval (abs_lb/abs_ub)",
        "estimation.results.lift.absolute_alpha_mismatch": "abs_alpha={abs_alpha!r} contradicts the alpha {alpha!r} of the relative interval the same call cut",
    },
)
_raise = raiser(_REFUSALS)


def _validate_absolute_reference_fields(row: Any) -> Any:
    if (row.abs_reference_kind == "t") != (row.abs_reference_df is not None):
        _raise(
            "estimation.results.lift.absolute_reference_mismatch",
            kind=row.abs_reference_kind,
            df=row.abs_reference_df,
        )
    if row.abs_alpha is None:
        return row
    if row.abs_lb is None or row.abs_ub is None:
        _raise("estimation.results.lift.absolute_alpha_without_interval", abs_alpha=row.abs_alpha)
    lift = row.lift
    if (
        lift is not None
        and lift.alpha is not None
        and not math.isclose(row.abs_alpha, lift.alpha, rel_tol=1e-12, abs_tol=0.0)
    ):
        _raise(
            "estimation.results.lift.absolute_alpha_mismatch",
            abs_alpha=row.abs_alpha,
            alpha=lift.alpha,
        )
    return row


def _validate_reference_fields(row: Any) -> Any:
    """Validate that the persisted and legacy reference fields agree."""
    if row.reference_kind == "t":
        if row.reference_df is None or not math.isfinite(row.reference_df) or row.reference_df <= 0:
            _raise(
                "estimation.results.lift.reference_df_required_for_t",
                metric=row.metric,
                group_id=row.group_id,
                reference_df=row.reference_df,
            )
    elif row.reference_df is not None:
        _raise(
            "estimation.results.lift.reference_df_set_for_normal",
            metric=row.metric,
            group_id=row.group_id,
            reference_kind=row.reference_kind,
            reference_df=row.reference_df,
        )
    if row.dof is not None and row.dof != row.reference_df:
        _raise(
            "estimation.results.lift.dof_reference_df_mismatch",
            metric=row.metric,
            group_id=row.group_id,
            dof=row.dof,
            reference_df=row.reference_df,
        )
    if (row.inference != "fixed") != (row.reference_kind == "sequential"):
        _raise(
            "estimation.results.lift.inference_reference_kind_mismatch",
            metric=row.metric,
            group_id=row.group_id,
            inference=row.inference,
            reference_kind=row.reference_kind,
        )
    return row


class _RowIdentity(CodedModel, BaseModel):
    """Leading identity fields shared, in the same declared order, by
    LiftEstimate, BreakoutEstimate and DailyLiftEstimate. Only a field
    already first in every subclass belongs here: pydantic prepends
    inherited fields ahead of a subclass's own, so a later-declared
    field (e.g. `null_lift` on DailyLiftEstimate) must stay put.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    group_id: str
    method: str  # which Method produced this estimate
    method_role: Literal["decision", "sensitivity"]
    inference: str = "fixed"  # "fixed" | "always_valid" | "asymptotic_mean"
    alternative: Alternative = "two-sided"  # "two-sided" | "greater" | "less"

    @field_validator("alternative", mode="before")
    @classmethod
    def _validated_alternative(cls, value: object) -> object:
        from increment.estimation.binomial_rr import validate_alternative

        return validate_alternative(value) if isinstance(value, str) else value


class LiftEstimate(_RowIdentity):
    """A lift estimate, carrying the metadata to identify which (metric,
    method, group) and its persisted inference reference.

    ``value`` is ``exp(mu_n) - 1`` for ``scale="log"`` estimates (under a
    near-flat prior this equals ``treatment_mean / control_mean - 1``), or
    ``mu_n`` directly for ``scale="linear"`` estimates, which already carry
    the posterior on the relative scale. The CI is the same posterior's
    quantile, back-transformed to this scale.

    The Normal posterior is never persisted: with a stored ``alpha`` the
    ``mu``/``sigma`` recovery from ``value``/``lb``/``alpha``/``scale`` is
    accurate up to rounding away from a zero critical value; open fixed rows use raw
    statistics to recover uncertainty at alpha=0.5. A closed
    ``alpha``-less interval falls back to the ``level`` tail, which
    is exact at ordinary alphas but loses precision as ``alpha`` approaches
    the representable floor. A mixture posterior (``prior_spec`` set) is not
    recoverable, so it persists the prior instead.

    ``reference_kind`` persists the reference: "normal", "t", "sequential",
    "binomial", or "confidence_set". Winsor confidence sets persist raw
    construction state and endpoint statuses, expose no posterior, and
    reinvert through ``reintervalize(alpha)``. Their public
    ``confidence_set.qualification`` is either
    ``pointwise_asymptotic_model_conditioned_v1`` (bootstrap candidate) or
    ``uniform_support_conditioned_v1`` (rank method); neither is an
    unqualified finite-sample promise. Bootstrap p-values invert stored roots;
    rank p-values are alpha when the set excludes the null, else one.
    ``dof`` retains cluster degrees of freedom, so ``dof=None`` does not
    imply Normal inference: a Welch reference has its own ``reference_df``.
    Relative t p-values use this reference; posterior-derived stats refuse it.

    ``inference`` labels the interval semantics: "fixed" is a single-look
    interval; "always_valid"/"asymptotic_mean" are valid at every look.
    ``value`` carries no stopping adjustment (winner's curse); the interval
    endpoints are the safe summary. Decision-stat methods refuse on any
    non-"fixed" inference.

    ``value_scale="absolute"`` marks a row whose ``value`` is not a
    relative lift at all - an encouragement design's additive LATE, or an
    observational metric reported additively.
    """

    null_lift: float = Field(default=0.0, allow_inf_nan=False)
    # H0 boundary this estimate was resolved against (relative scale, same
    # units as `lift.value`); 0.0 (default) is the plain zero-null case.
    preferred_direction: PreferredDirection | None = None
    # The metric's declared favorability. Separate from `alternative`, which
    # can point the opposite way on a harm/futility test.
    lift: Estimate | None = None
    # A set-backed reference may have no finite empirical ratio while its
    # confidence set remains available with explicit endpoint statuses.
    estimand: Literal[
        "itt", "compliance", "late", "ate", "plr_slope", "overlap_subpopulation_ate"
    ] = "itt"
    # itt/compliance/late are encouragement.ESTIMANDS. Only infer_ate stamps "ate" (iptw/aipw),
    # "overlap_subpopulation_ate" (when gate.overlap="trim" drops units) or "plr_slope" (dml).
    #: Population this row was estimated over. "assigned" is every enrolled
    #: unit; "triggered" is the subset that produced the declared trigger.
    analysis_population: Literal["assigned", "triggered"] = "assigned"
    value_scale: ValueScale = "relative"
    note: str | None = None  # guard fallbacks / suppression reasons / caveats
    abs_diff: float | None = Field(default=None, allow_inf_nan=False)
    abs_se: float | None = Field(default=None, allow_inf_nan=False)
    null_abs: float | None = Field(default=None, allow_inf_nan=False)
    # H0 boundary on the absolute scale (same units as `abs_diff`); the
    # additive sibling of `null_lift`. None = no absolute margin in play.
    abs_lb: float | None = Field(default=None, allow_inf_nan=False)
    abs_ub: float | None = Field(default=None, allow_inf_nan=False)
    # None means the sidecar reference is unavailable, including legacy rows.
    abs_reference_kind: Literal["normal", "t"] | None = None
    abs_reference_df: float | None = Field(default=None, allow_inf_nan=False, gt=0)
    abs_alpha: float | None = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)
    # The central-equivalent alpha the additive interval was cut at, in `Estimate.alpha`'s
    # convention: a directional row carries the doubled call alpha. Set with abs_lb/abs_ub;
    # None for a row serialized before it was persisted.

    winsor_lower_percentile: float | None = None
    winsor_upper_percentile: float | None = None
    winsor_lower_bound: float | None = None
    winsor_upper_bound: float | None = None
    winsor_control_n: int | None = None
    winsor_control_n_lower: int | None = None
    winsor_control_n_upper: int | None = None
    winsor_treatment_n: int | None = None
    winsor_treatment_n_lower: int | None = None
    winsor_treatment_n_upper: int | None = None
    # Widest influence support for this row; a smaller comparison count can
    # appear in its reliability advisory. None means unit grain.
    n_clusters: int | None = None
    dof: float | None = None
    reference_kind: Literal["normal", "t", "sequential", "binomial", "confidence_set"] = "normal"
    confidence_set: WinsorConfidenceSet | None = Field(default=None, exclude_if=lambda v: v is None)
    independent_mean_reference: IndependentMeanReference | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    reference_df: float | None = None
    # `dof` remains for compatibility; persisted reference fields drive consumers.
    population: str | None = None  # trimmed identification population, e.g.
    # "overlap e in [0.01, 0.99] (1834 of 2000 units)"; None = full source population
    relative_confidence_set: RelativeConfidenceSet | None = None
    relative_unavailable_reason: RelativeUnavailableReason | None = None
    # A quantile row's p-value, computed once by inverting the quantile
    # estimator's own interval construction at null=0.0 -- alpha-independent
    # by construction, unlike this row's own se/interval. None for every
    # non-quantile row.
    quantile_p_value: float | None = None
    sequential_result: SequentialResult | None = None
    # Sufficient counts/method for every reference_kind="binomial" row
    # (point-backed or not); see BinomialConfidenceSet. None for every
    # other reference_kind.
    binomial_set: BinomialConfidenceSet | None = None
    ds: date | datetime | str | int | float | None = None  # None for total-grain estimates
    scale: Literal["log", "linear"] = "log"
    # Scale the underlying Normal posterior lives on: "log" (posterior on
    # log-RR) or "linear" (posterior directly on the relative lift).
    prior_spec: StudentTPrior | MixturePrior | None = None
    # Set only for mixture-prior estimates, persisted as declared: the
    # posterior is recomputed from lift.log_mean/lift.log_se + this spec.
    prior_shrunk: bool = False
    # True when `prior is not None`: lift.log_mean/log_se are raw pre-prior
    # statistics while value/lb/ub are the prior-informed posterior.
    role: RowRole | None = None
    # The declared-plan role this row was estimated under, or "exploratory" for a metric
    # added after the plan; None only when `src.plan.declared` is False.
    discovery: bool | None = None
    # None outside a tested family; otherwise BH/e-BH selection or qualified
    # fixed-roster Bonferroni selection for asymptotic mean inference.
    # Family selection need not equal stat_sig(): capped intervals and additive
    # margins can disagree with a BH/e-BH discovery.
    family_axes: tuple[str, ...] | None = None
    # Axes of the correcting family, or None when no family correction applies.
    # Primary Bonferroni allocation is recorded in lift.level and the plan.
    family_q: float | None = None  # FDR level, or qualified asymptotic-mean FWER level.
    family_threshold: float | None = None
    # Family-wide values: family_guarantee is "asymptotic_sequential" if any cell is asymptotic,
    # else "finite_sample"; family_nominal_alpha is the overall test level in
    # fcr_alpha = min(realized_threshold, family_nominal_alpha).
    family_guarantee: Literal["finite_sample", "asymptotic_sequential"] | None = None
    family_nominal_alpha: float | None = None
    family_size: int | None = Field(default=None, ge=1)
    # Hypotheses in the correcting family (BH's m) where the family records it; None otherwise.

    # family_threshold is R*q/m for BH/e-BH; unset for asymptotic Bonferroni.
    # Reported intervals never receive more than their nominal alpha.
    _infer_reference: ClassVar[Any] = model_validator(mode="before")(
        _infer_reference_from_legacy_dof
    )
    _check_reference: ClassVar[Any] = model_validator(mode="after")(_validate_reference_fields)
    _check_absolute_reference: ClassVar[Any] = model_validator(mode="after")(
        _validate_absolute_reference_fields
    )

    @field_serializer("ds", when_used="json")
    def _serialize_day_axis(self, value: date | datetime | str | int | float | None) -> object:
        if isinstance(value, datetime):
            return {"datetime": value.isoformat()}
        if isinstance(value, str):
            return {"label": value}
        return value

    @field_validator("ds", mode="before")
    @classmethod
    def _deserialize_day_axis(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, Mapping):
            if set(value) == {"label"} and isinstance(value["label"], str):
                return value["label"]
            if set(value) == {"datetime"} and isinstance(value["datetime"], str):
                return datetime.fromisoformat(value["datetime"])
        if info.mode == "json" and isinstance(value, str):
            return date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _open_side_matches_alternative(self):
        if self.sequential_result is not None:
            return self
        if self.lift is None or self.relative_confidence_set is not None:
            return self
        open_side = self.lift.open_side
        if open_side == "upper" and self.alternative != "greater":
            _raise(
                "estimation.results.lift.open_side_alternative_mismatch",
                open_side=open_side,
                alternative=self.alternative,
            )
        if open_side == "lower" and self.alternative != "less":
            _raise(
                "estimation.results.lift.open_side_alternative_mismatch",
                open_side=open_side,
                alternative=self.alternative,
            )
        return self

    @model_validator(mode="after")
    def _independent_mean_consistency(self):
        if self.method == "independent_mean" and self.independent_mean_reference is None:
            winsor_refuse(
                "invalid_state", "Independent mean results require their component reference."
            )
        if self.method != "independent_mean" and self.independent_mean_reference is not None:
            winsor_refuse(
                "invalid_state", "Independent mean component references require that method."
            )
        if self.independent_mean_reference is not None:
            from increment.estimation.inference import _resolve_fixed_horizon

            ref = self.independent_mean_reference
            resolved = _resolve_fixed_horizon(
                ref.alpha,
                ref.alternative,
                dof=None,
                arm_ns=None,
                prior=None,
                independent_means=ref,
            )
            assert resolved.crit is not None
            expected_bounds: tuple[float | None, float | None] = (
                ref.point - resolved.crit * ref.se(),
                ref.point + resolved.crit * ref.se(),
            )
            expected_open = None
            if ref.interval == "directional-fcr":
                if ref.alternative == "two-sided":
                    winsor_refuse("invalid_state", "Directional mean interval needs a direction.")
                critical = tail_isf(
                    student_t_isf, resolved.alpha_eff, ref.df, what="fixed FCR endpoint"
                )
                bounds = wald_bounds(ref.point, critical, ref.se(), what="fixed FCR endpoint")
                expected_bounds = (
                    (bounds[0], None) if ref.alternative == "greater" else (None, bounds[1])
                )
                expected_open = "upper" if ref.alternative == "greater" else "lower"
            if (
                self.reference_kind != "t"
                or self.reference_df != ref.df
                or self.value_scale != "absolute"
                or self.scale != "linear"
                or self.lift is None
                or self.lift.value != ref.point
                or self.lift.log_se != ref.se()
                or self.lift.log_mean != ref.point
                or (self.lift.lb, self.lift.ub) != expected_bounds
                or self.lift.alpha != resolved.alpha_eff
                or self.lift.level != math.fsum((1.0, -resolved.alpha_eff))
                or self.lift.open_side != expected_open
                or self.alternative != ref.alternative
                or self.null_lift != 0
                or self.null_abs is not None
                or self.prior_spec is not None
                or self.prior_shrunk
                or self.dof is not None
                or self.n_clusters is not None
                or any(
                    x is not None
                    for x in (self.abs_diff, self.abs_se, self.abs_lb, self.abs_ub, self.abs_alpha)
                )
                or self.method != "independent_mean"
            ):
                winsor_refuse(
                    "invalid_state", "Independent mean result contradicts its component reference."
                )
        return self

    @model_validator(mode="after")
    def _binomial_lift_availability(self):
        """Enforce the documented binomial "iff" contract in BOTH
        directions, and that every displayed field on this row agrees
        with its own persisted ``binomial_set``: ``lift is None`` iff
        this is a genuinely point-unavailable binomial row (never a bare
        ``None/None`` claiming full-space availability, never a silently
        dropped set on a point-backed row, never a non-binomial row with
        no point at all); and, whenever both are present, ``lift``'s
        value/bounds/alpha/level and ``binomial_set``'s geometry cannot
        contradict the persisted counts -- otherwise a `p_value()`/
        `stat_sig()` caller (which uses the counts) and a display
        consumer (which uses the bounds) could disagree about this same
        row.
        """
        if self.reference_kind == "confidence_set":
            region = self.confidence_set
            if (
                region is None
                or self.binomial_set is not None
                or self.relative_confidence_set is not None
                or self.relative_unavailable_reason is not None
            ):
                winsor_refuse(
                    "invalid_state", "Confidence-set reference requires its own persisted region."
                )
            if (
                self.alternative != "two-sided"
                or self.prior_spec is not None
                or self.prior_shrunk
                or self.value_scale != "relative"
                or self.estimand != "itt"
                or self.metric != region.raw.metric
                or self.group_id != region.treatment
                or self.analysis_population != region.raw.population
                or self.abs_diff != region.additive_point
                or self.abs_lb != region.additive.lower.value
                or self.abs_ub != region.additive.upper.value
                or (self.abs_alpha is not None and self.abs_alpha != region.alpha)
                or self.abs_se is not None
            ):
                winsor_refuse(
                    "invalid_state", "Confidence-set row contradicts its persisted region."
                )
            if (self.lift is None) != (region.point is None):
                winsor_refuse("invalid_state", "Confidence-set point availability is inconsistent.")
            if self.lift is not None:
                finite = (
                    region.lower is not None and region.upper is not None and region.alpha / 2 > 0
                )
                if (
                    self.lift.value != region.point
                    or self.lift.log_mean is not None
                    or self.lift.log_se is not None
                    or self.lift.open_side is not None
                    or (self.lift.lb, self.lift.ub)
                    != ((region.lower, region.upper) if finite else (None, None))
                    or self.lift.alpha != (region.alpha if finite else None)
                ):
                    winsor_refuse(
                        "invalid_state", "Displayed estimate contradicts its confidence region."
                    )
            return self
        if self.confidence_set is not None:
            winsor_refuse(
                "invalid_state", "A confidence set cannot be relabeled as a sampling posterior."
            )
        if _validate_joint_relative_row(self):
            return self
        if self.reference_kind == "sequential":
            return self
        if self.reference_kind != "binomial":
            if self.lift is None:
                _raise(
                    "estimation.results.lift.binomial_lift_availability",
                    group_id=self.group_id,
                    metric=self.metric,
                    reason="lift is None outside the binomial point-unavailable case",
                )
            return self
        bset = self.binomial_set
        if bset is None:
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason="reference_kind='binomial' row is missing its sufficient-count set",
            )
        expected_geometry = {
            "two-sided": "central",
            "greater": "lower_bound",
            "less": "upper_bound",
        }.get(self.alternative)
        if expected_geometry is not None and bset.geometry != expected_geometry:
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason=(
                    f"alternative={self.alternative!r} implies geometry="
                    f"{expected_geometry!r}, got {bset.geometry!r}"
                ),
            )
        if self.lift is None:
            if bset.point_available:
                _raise(
                    "estimation.results.lift.binomial_lift_availability",
                    group_id=self.group_id,
                    metric=self.metric,
                    reason="binomial_set claims a point-available count pair with lift=None",
                )
            return self
        if not bset.point_available:
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason="lift is a finite point but binomial_set has no available point (x_c == 0)",
            )
        from increment.estimation.binomial_rr import point_lift as _point_lift

        expected_value = _point_lift(bset.x_c, bset.n_c, bset.x_t, bset.n_t)
        if expected_value is None or not math.isclose(
            self.lift.value, expected_value, rel_tol=0.0, abs_tol=1e-9
        ):
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason=(
                    f"lift.value={self.lift.value!r} does not match the persisted counts' "
                    f"empirical ratio ({expected_value!r})"
                ),
            )
        if self.lift.alpha != bset.alpha or self.lift.level != bset.level:
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason=(
                    f"lift alpha/level ({self.lift.alpha!r}/{self.lift.level!r}) does not "
                    f"match binomial_set ({bset.alpha!r}/{bset.level!r})"
                ),
            )
        if self.lift.lb != bset.lower or self.lift.ub != bset.upper:
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason=(
                    f"lift bounds ({self.lift.lb!r}, {self.lift.ub!r}) do not match "
                    f"binomial_set bounds ({bset.lower!r}, {bset.upper!r})"
                ),
            )
        return self

    @model_validator(mode="after")
    def _sequential_consistency(self):
        """Keep checkpoint authority independent of fixed-kind early returns."""
        from increment.sequential_state import require_public_laws, sequential_refuse

        result = self.sequential_result
        if self.reference_kind != "sequential":
            if result is not None:
                sequential_refuse("source.invalid", "fixed result cannot carry sequential evidence")
            return self
        if result is None:
            sequential_refuse(
                "continuation.legacy", "sequential result lacks certified raw-state evidence"
            )
        if isinstance(result, AsymptoticSequentialResult) != (self.inference == "asymptotic_mean"):
            sequential_refuse(
                "source.invalid", "displayed validity regime differs from its checkpoint"
            )
        require_public_laws((result.checkpoint.model,), "LiftEstimate replay")
        if (
            self.value_scale != "relative"
            or self.scale != "linear"
            or self.binomial_set is not None
            or self.prior_spec is not None
            or self.prior_shrunk
            or any(
                value is not None
                for value in (
                    self.null_abs,
                    self.abs_diff,
                    self.abs_se,
                    self.abs_lb,
                    self.abs_ub,
                    self.abs_reference_kind,
                    self.abs_reference_df,
                    self.abs_alpha,
                    self.n_clusters,
                )
            )
        ):
            sequential_refuse(
                "source.invalid", "sequential result cannot carry fixed-horizon inference"
            )
        cell = result.checkpoint.cell
        if (self.metric, self.group_id, self.estimand, self.alternative, self.null_lift) != (
            cell.metric,
            cell.group_id,
            cell.estimand,
            cell.alternative,
            float(cell.null_lift),
        ):
            sequential_refuse("source.invalid", "result and checkpoint hypothesis disagree")
        from increment.estimation.sequential_result import _point
        from increment.estimation.sequential_runtime import (
            display_estimate,
            require_selected_widening,
        )

        if (
            self.lift != display_estimate(result)
            or result.point_reason != _point(result.checkpoint)[1]
        ):
            sequential_refuse(
                "source.invalid",
                "displayed sequential point or interval differs from its exact stopped state",
            )
        require_selected_widening(
            result,
            discovery=self.discovery,
            family_threshold=self.family_threshold,
            family_nominal_alpha=self.family_nominal_alpha,
        )
        return self

    def require_lift(self) -> Estimate:
        """Return the finite point estimate or refuse for a set-only row."""
        if self.lift is None:
            if self.confidence_set is not None:
                winsor_refuse(
                    "invalid_state", "Point estimate is undefined; use confidence_set endpoints."
                )
            if self.relative_unavailable_reason is not None:
                _raise(
                    "estimation.results.joint.unavailable",
                    reason=self.relative_unavailable_reason,
                )
            if self.relative_confidence_set is not None:
                _raise(
                    "estimation.results.joint.unavailable",
                    reason=self.relative_confidence_set.point_unavailable_reason,
                )
            if self.sequential_result is not None:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported",
                    self.sequential_result.point_reason
                    or "use sequential_result.bounds for the confidence set",
                )
            _raise(
                "estimation.results.lift.binomial_lift_availability",
                group_id=self.group_id,
                metric=self.metric,
                reason="a finite point estimate is unavailable; use binomial_set",
            )
        return self.lift

    def _family_view(self) -> LiftEstimate:
        """The estimate exploratory-family evidence and FCR reissue read: this row itself."""
        return self

    def reintervalize(self, alpha: float) -> LiftEstimate:
        """Reinvert persisted inference; bootstrap roots are never regenerated."""
        if self.confidence_set is None:
            winsor_refuse(
                "reinversion_required", "This operation requires a persisted winsor confidence set."
            )
        from increment.estimation.winsor import estimate_winsor_lift

        region = self.confidence_set
        result = estimate_winsor_lift(
            region.raw,
            region.control,
            region.treatment,
            alpha=alpha,
            method=self.method,
            method_role=self.method_role,
            null_lift=self.null_lift,
            null_abs=self.null_abs,
            preferred_direction=self.preferred_direction,
            reference=region.reference
            if isinstance(region.reference, BootstrapReference)
            else None,
        )
        return self.model_copy(
            update={
                "confidence_set": result.confidence_set,
                "lift": result.lift,
                "abs_lb": result.abs_lb,
                "abs_ub": result.abs_ub,
                "abs_alpha": result.abs_alpha,
            }
        )

    def __repr__(self) -> str:
        """Compact interactive representation of the published estimate."""
        if self.confidence_set is not None:
            region = self.confidence_set
            return (
                f"LiftEstimate(metric={self.metric!r}, group={self.group_id!r}, "
                f"value={region.point!r}, lower={region.relative.lower!r}, "
                f"upper={region.relative.upper!r}, reference='confidence_set')"
            )
        unit = self.value_scale
        if self.relative_unavailable_reason is not None:
            return (
                f"LiftEstimate(metric={self.metric!r}, group={self.group_id!r}, "
                f"value={None if self.lift is None else self.lift.value!r}, "
                f"relative_unavailable={self.relative_unavailable_reason!r}, "
                f"abs_diff={self.abs_diff!r})"
            )
        if self.relative_confidence_set is not None:
            relative = self.relative_confidence_set
            return (
                f"LiftEstimate(metric={self.metric!r}, group={self.group_id!r}, "
                f"estimand={self.estimand!r}, value="
                f"{None if self.lift is None else self.lift.value!r} ({unit}), "
                f"relative_set={relative.geometry}:{relative.intervals!r}, "
                f"stat_sig={self.stat_sig()}, discovery={self.discovery!r})"
            )
        if self.sequential_result is not None:
            value = "unavailable" if self.lift is None else f"{self.lift.value:+.4g}"
            bounds = self.sequential_result.bounds
            interval = f"ratio_set=({bounds.lower}, {bounds.upper}), empty={bounds.empty}"
        elif self.lift is None:
            value = "unavailable"
            bset = self.binomial_set
            assert bset is not None, "validated: lift=None implies a binomial_set"
            interval = (
                f"interval=({bset.lower:+.4g}, "
                f"{'+inf' if bset.upper is None else f'{bset.upper:+.4g}'})"
            )
        else:
            value = f"{self.lift.value:+.4g}"
            if self.lift.open_side == "lower" and self.lift.ub is not None:
                interval = f"interval=(-inf, {self.lift.ub:+.4g})"
            elif self.lift.open_side == "upper" and self.lift.lb is not None:
                interval = f"interval=({self.lift.lb:+.4g}, +inf)"
            elif self.lift.lb is None or self.lift.ub is None:
                interval = "interval=None"
            else:
                interval = f"interval=({self.lift.lb:+.4g}, {self.lift.ub:+.4g})"
        stat_sig = self.stat_sig()
        return (
            f"LiftEstimate(metric={self.metric!r}, group={self.group_id!r}, "
            f"estimand={self.estimand!r}, value={value} ({unit}), "
            f"{interval}, stat_sig={stat_sig}, discovery={self.discovery!r})"
        )

    @property
    def winsor_control_fraction_lower(self) -> float | None:
        if self.winsor_control_n is None or self.winsor_control_n_lower is None:
            return None
        return self.winsor_control_n_lower / self.winsor_control_n

    @property
    def winsor_control_fraction_upper(self) -> float | None:
        if self.winsor_control_n is None or self.winsor_control_n_upper is None:
            return None
        return self.winsor_control_n_upper / self.winsor_control_n

    @property
    def winsor_treatment_fraction_lower(self) -> float | None:
        if self.winsor_treatment_n is None or self.winsor_treatment_n_lower is None:
            return None
        return self.winsor_treatment_n_lower / self.winsor_treatment_n

    @property
    def winsor_treatment_fraction_upper(self) -> float | None:
        if self.winsor_treatment_n is None or self.winsor_treatment_n_upper is None:
            return None
        return self.winsor_treatment_n_upper / self.winsor_treatment_n

    def _t_posterior(self, e: Estimate, tail_alpha: float) -> Normal:
        """Posterior behind a t-reference interval.

        A t reference corrects for a variance estimated from the data in the
        SAMPLING distribution; it does not widen the posterior. The stored
        endpoints are ``t_{reference_df}`` quantiles, so inverting them with a
        Normal z would report sigma inflated by the t/z ratio. The
        working-scale moments the interval was cut from are persisted on the
        row: read those, and use the endpoints only to confirm they are the
        pair those moments imply.
        """
        from increment.estimation.inference import Normal

        assert self.reference_df is not None, "validated: a t row carries reference_df"
        assert e.lb is not None and e.ub is not None, "checked by the caller"
        parameters = _fixed_fcr_parameters(self)
        assert parameters is not None, "refused above: a t row carries log_mean/log_se"
        mu, sigma = parameters
        crit = tail_isf(
            student_t_isf,
            tail_alpha / 2.0,
            self.reference_df,
            what="fixed t posterior reconstruction",
        )
        if self.scale == "log":
            lb_implied = math.expm1(mu - crit * sigma)
            ub_implied = math.expm1(mu + crit * sigma)
        else:
            lb_implied = mu - crit * sigma
            ub_implied = mu + crit * sigma
        if (
            sigma <= 0
            or not math.isclose(lb_implied, e.lb, rel_tol=1e-6, abs_tol=1e-9)
            or not math.isclose(ub_implied, e.ub, rel_tol=1e-6, abs_tol=1e-9)
        ):
            _raise(
                "estimation.results.lift.liftestimate_interval_symmetric",
                group_id=self.group_id,
                metric=self.metric,
                scale=self.scale,
            )
        return Normal(mu=mu, sigma=sigma)

    def _posterior(self) -> LiftPosterior:
        """Recover the posterior (Normal or mixture) this estimate was cut from.

        A ``reference_kind="normal"`` row inverts its endpoints: with stored
        ``alpha``, closed (value, lb, alpha) intervals give back (mu, sigma) up
        to rounding, and that inversion is the only route for a Normal-prior row
        whose prior is not persisted. An ``alpha``-less closed interval (e.g. a
        hand-built or legacy Estimate) recovers the tail from ``level`` instead
        -- backward-compatible, exact at ordinary alphas.

        A ``reference_kind="t"`` row must NOT be inverted that way: its
        endpoints are t quantiles, so a Normal z would divide by too small a
        critical value and report sigma inflated by the t/z ratio. It reads the
        persisted working-scale moments instead, the same ones open fixed
        intervals prefer, and checks the endpoints against them.
        """
        from increment.estimation.inference import Normal

        if self.reference_kind == "confidence_set":
            winsor_refuse(
                "posterior_unavailable",
                "Rank confidence sets do not define posterior probabilities.",
            )
        if self.sequential_result is not None:
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported",
                "likelihood evidence is not an effect posterior; use sequential_result and stat_sig()",
            )
        e = self.lift
        if self.reference_kind == "binomial":
            _raise(
                "estimation.results.binomial.posterior_unavailable",
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.inference != "fixed":
            _raise(
                "estimation.results.lift.posterior_decision_stats_sequential",
                group_id=self.group_id,
                inference=self.inference,
                metric=self.metric,
            )
        if self.n_clusters is not None:
            _raise(
                "estimation.results.lift.posterior_decision_stats_cluster_robust",
                reference_df=self.reference_df,
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.reference_kind == "t" and (e is None or e.log_mean is None or e.log_se is None):
            missing = " and ".join(
                name
                for name, present in (
                    ("lift.log_mean", e is not None and e.log_mean is not None),
                    ("lift.log_se", e is not None and e.log_se is not None),
                )
                if not present
            )
            _raise(
                "estimation.results.lift.t_posterior_sufficient_statistics",
                missing=missing,
                reference_df=self.reference_df,
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.relative_confidence_set is not None or self.relative_unavailable_reason is not None:
            _raise(
                "estimation.results.joint.unavailable",
                reason="a joint frequentist reference does not define a scalar posterior",
            )
        assert e is not None, "validated: fixed scalar inference carries a point"
        if e.alpha is not None and e.level is not None:
            expected_level = math.fsum((1.0, -e.alpha))
            if not math.isclose(e.level, expected_level, rel_tol=1e-12, abs_tol=1e-15):
                _raise(
                    "estimation.results.lift.liftestimate_level_contradicts",
                    e_alpha=e.alpha,
                    e_level=e.level,
                    expected_level=expected_level,
                    group_id=self.group_id,
                    metric=self.metric,
                )
        if e.open_side is not None and self.prior_spec is not None:
            _raise("estimation.results.lift.fcr_prior_unsupported")
        if self.prior_spec is not None:
            if e.log_mean is None or e.log_se is None:
                _raise(
                    "estimation.results.lift.liftestimate_prior_spec",
                    group_id=self.group_id,
                    metric=self.metric,
                )
            from increment.estimation.priors import mixture_posterior

            return mixture_posterior(e.log_mean, e.log_se, self.prior_spec.components())
        if e.open_side is not None:
            if self.prior_shrunk:
                _raise("estimation.results.lift.fcr_prior_unsupported")
            calibrated = e.ub if e.open_side == "lower" else e.lb
            assert calibrated is not None, "validated open intervals have a finite endpoint"
            mu, sigma = _recover_fixed_open_interval_parameters(
                e,
                center=_working_value(self, e.value),
                calibrated=_working_value(self, calibrated),
                persisted=_fixed_fcr_parameters(self),
            )
            if not math.isfinite(sigma) or sigma <= 0:
                _raise(
                    "estimation.results.lift.liftestimate_interval_symmetric",
                    group_id=self.group_id,
                    metric=self.metric,
                    scale=self.scale,
                )
            return Normal(mu=mu, sigma=sigma)
        if e.lb is None or e.ub is None or e.level is None:
            _raise(
                "estimation.results.lift.liftestimate_carries_no",
                group_id=self.group_id,
                metric=self.metric,
            )
        # A level-only interval (no stored alpha) recovers the two-sided tail
        # from the confidence level; exact enough at decision-stat alphas.
        tail_alpha = e.alpha if e.alpha is not None else math.fsum((1.0, -e.level))
        if self.reference_kind == "t":
            return self._t_posterior(e, tail_alpha)
        z = tail_isf(_norm.isf, tail_alpha / 2.0, what="fixed Normal posterior reconstruction")
        if self.scale == "log":
            if e.value <= -1.0 or e.lb <= -1.0:
                _raise(
                    "estimation.results.lift.liftestimate_value_lb",
                    e_lb=e.lb,
                    e_value=e.value,
                    group_id=self.group_id,
                    metric=self.metric,
                )
            mu = math.log1p(e.value)
            sigma = (mu - math.log1p(e.lb)) / z
            ub_implied = math.expm1(mu + z * sigma)
        else:
            mu = e.value
            sigma = (mu - e.lb) / z
            ub_implied = mu + z * sigma
        if sigma <= 0 or not math.isclose(ub_implied, e.ub, rel_tol=1e-6, abs_tol=1e-9):
            _raise(
                "estimation.results.lift.liftestimate_interval_symmetric",
                group_id=self.group_id,
                metric=self.metric,
                scale=self.scale,
            )
        return Normal(mu=mu, sigma=sigma)

    def chance_to_beat(self) -> float:
        """P(lift > 0). Under the default flat prior this equals
        1 - one-sided p-value exactly - decision-grade only with a real
        prior."""
        return self.prob_beyond(0.0)

    def prob_beyond(self, threshold: float) -> float:
        """P(lift > threshold). prob_beyond(mde) is the chance of a
        meaningful win; prob_beyond(-tolerance) is the non-inferiority
        probability a guardrail wants."""
        p = self._posterior()
        if self.scale == "log" and threshold <= -1.0:
            _raise("estimation.results.lift.threshold_representable_log", threshold=threshold)
        t = math.log1p(threshold) if self.scale == "log" else threshold
        return p.survival(t)

    def prob_favorable(self) -> float:
        """P(lift is on the metric's declared good side of ``null_lift``).

        Not derived from ``alternative``, which can point the opposite way
        on a harm/futility test. When ``null_abs`` is set, the decision
        moves to the additive scale instead.
        """
        if self.reference_kind == "confidence_set":
            winsor_refuse(
                "posterior_unavailable",
                "Rank confidence sets do not define posterior probabilities.",
            )
        if self.preferred_direction is None:
            _raise(
                "estimation.results.lift.liftestimate_prob_favorable",
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.null_abs is not None:
            # Absolute tails retain their unsupported-reference refusal.
            if (
                self.reference_kind == "t"
                or self.abs_reference_kind == "t"
                or self.n_clusters is not None
            ):
                _raise(
                    "estimation.results.lift.p_value_cluster_robust_null_abs",
                    reference_df=self.abs_reference_df
                    if self.abs_reference_kind == "t"
                    else self.reference_df,
                    group_id=self.group_id,
                    metric=self.metric,
                )
            if self.abs_se is None or self.abs_diff is None:
                _raise(
                    "estimation.results.lift.p_value_null_abs_missing_abs_se",
                    group_id=self.group_id,
                    metric=self.metric,
                )
            p_beyond = float(_norm.sf((self.null_abs - self.abs_diff) / self.abs_se))
            return (
                p_beyond
                if self.preferred_direction != "decrease"
                else float(_norm.cdf((self.null_abs - self.abs_diff) / self.abs_se))
            )
        p_beyond = self.prob_beyond(self.null_lift)
        return (
            p_beyond
            if self.preferred_direction != "decrease"
            else float(
                self._posterior().cdf(
                    math.log1p(self.null_lift) if self.scale == "log" else self.null_lift
                )
            )
        )

    def prob_within(self, threshold: float) -> float:
        """ROPE: P(|lift| < threshold), the 'confidently flat' verdict.

        ``threshold`` is a relative-lift fraction in (0, 1) on a
        ``value_scale="relative"`` row (both ``scale`` variants). On a
        ``value_scale="absolute"`` row -- an encouragement LATE or an
        additively-reported observational metric -- the window is read
        in the row's own additive units instead, so any positive
        threshold is accepted (e.g. a $5 ROPE on revenue-per-user).
        """
        if self.value_scale == "absolute":
            if not threshold > 0.0:
                _raise(
                    "estimation.results.lift.prob_within_threshold_positive", threshold=threshold
                )
        elif not 0.0 < threshold < 1.0:
            _raise(
                "estimation.results.lift.prob_within_threshold_unit_interval", threshold=threshold
            )
        p = self._posterior()
        if self.scale == "log":
            hi, lo = math.log1p(threshold), math.log1p(-threshold)
        else:
            hi, lo = threshold, -threshold
        return p.probability_between(lo, hi)

    def chance_to_beat_favorable(self) -> float:
        """Direction-aware companion to ``chance_to_beat()``: P(lift is on
        the favorable side of 0). Always compares against 0, not
        ``null_lift``, unlike ``prob_favorable()``."""
        if self.preferred_direction is None:
            _raise(
                "estimation.results.lift.liftestimate_chance_to",
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.preferred_direction != "decrease":
            return self.chance_to_beat()
        return 1.0 - self.chance_to_beat()

    def risk_if_shipped(self) -> float:
        """Expected loss if shipped and the true lift is negative:
        E[max(0, -lift)]. Closed form under the lognormal (scale='log') or
        Normal (scale='linear') posterior - no sampling."""
        return self._posterior().expected_negative_part(scale=self.scale)

    def risk_if_shipped_favorable(self) -> float:
        """Direction-aware companion to ``risk_if_shipped()``: expected
        magnitude of the move to the unfavorable side of 0. Always
        compares against 0, not ``null_lift``."""
        if self.preferred_direction is None:
            _raise(
                "estimation.results.lift.liftestimate_risk_if",
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.preferred_direction != "decrease":
            return self.risk_if_shipped()
        return self._posterior().expected_positive_part(scale=self.scale)

    def require_sequential_result(self) -> SequentialResult:
        """Return the authoritative stopped likelihood state or refuse a fixed row."""
        if self.sequential_result is None:
            from increment.sequential_state import sequential_refuse

            sequential_refuse("source.invalid", "result has no registered sequential checkpoint")
        return self.sequential_result

    def require_exact_sequential_result(self) -> SequentialInferenceResult:
        """Narrow to exact likelihood evidence without relabeling an AsympCS."""
        result = self.require_sequential_result()
        if not isinstance(result, SequentialInferenceResult):
            from increment.sequential_state import sequential_refuse

            sequential_refuse(
                "route.unsupported", "asymptotic confidence sets are not exact evidence"
            )
        return result

    def require_asymptotic_sequential_result(self) -> AsymptoticSequentialResult:
        """Return the scalar mean construction and its complete confidence geometry."""
        result = self.require_sequential_result()
        if not isinstance(result, AsymptoticSequentialResult):
            from increment.sequential_state import sequential_refuse

            sequential_refuse("route.unsupported", "result is not a scalar mean AsympCS")
        return result

    def stat_sig(self) -> bool:
        """Whether this row's own interval excludes its own null, honoring
        ``alternative`` -- the LiftEstimate-specific twin of
        `increment.tables._stat_sig` (which delegates here), exposed as a
        method so a caller outside `tables` (a gated extra) can read a
        row's own interval-exclusion verdict directly. Independent of,
        and never gated by, this row's family ``discovery`` verdict (see
        `increment.estimation.family.family_discovery`): a selected
        decision row keeps ``discovery=True`` even when a binding
        nominal-alpha cap leaves ``stat_sig()`` False, and family
        selection never demotes it to match. A ``null_abs`` row decides
        against the additive interval (``abs_lb``/``abs_ub``) instead of
        the relative one; missing additive endpoints report False rather
        than falling back to the relative interval.

        A ``reference_kind="binomial"`` row (point-backed or not) instead
        tests ``r0 = 1 + null_lift`` for exclusion from the exact
        confidence set directly via the persisted counts -- the same
        Berger-Boos tail functions the set was inverted from, so this is
        exact even when ``null_lift`` differs from the null the row's own
        ``lift``/``binomial_set`` bounds were built against.
        """
        if self.confidence_set is not None:
            if self.null_abs is not None:
                return self.confidence_set.additive.excludes(self.null_abs)
            return self.confidence_set.relative.excludes(self.null_lift)
        if self.sequential_result is not None:
            return self.sequential_result.rejects()
        if self.null_abs is not None:
            if self.abs_lb is None or self.abs_ub is None:
                return False
            if self.alternative == "greater":
                return self.abs_lb > self.null_abs
            if self.alternative == "less":
                return self.abs_ub < self.null_abs
            return not (self.abs_lb <= self.null_abs <= self.abs_ub)
        if self.relative_unavailable_reason is not None:
            return False
        if self.relative_confidence_set is not None:
            return self.relative_confidence_set.contains(self.null_lift) is False
        if self.reference_kind == "binomial":
            bset = self.binomial_set
            assert bset is not None
            # ``p_two < alpha`` is ``min(p_+, p_-) < alpha / 2``: doubling is exact.
            return bset.null_p_value(self.null_lift, self.alternative) < bset.decision_alpha
        if self.alternative == "greater":
            assert self.lift is not None
            return self.lift.lb is not None and self.lift.lb > self.null_lift
        if self.alternative == "less":
            assert self.lift is not None
            return self.lift.ub is not None and self.lift.ub < self.null_lift
        assert self.lift is not None
        return self.lift.excludes(self.null_lift)

    def _null_abs_p_value(self) -> float:
        """The additive-margin tail from ``Normal(abs_diff, abs_se)``
        against ``null_abs`` -- split out of ``p_value`` purely to keep
        that method's branch count readable; see its docstring for the
        contract."""
        if (
            self.reference_kind == "t"
            or self.abs_reference_kind == "t"
            or self.n_clusters is not None
        ):
            _raise(
                "estimation.results.lift.p_value_cluster_robust_null_abs",
                reference_df=self.abs_reference_df
                if self.abs_reference_kind == "t"
                else self.reference_df,
                group_id=self.group_id,
                metric=self.metric,
            )
        if self.abs_se is None or self.abs_diff is None:
            _raise(
                "estimation.results.lift.p_value_null_abs_missing_abs_se",
                group_id=self.group_id,
                metric=self.metric,
            )
        assert self.null_abs is not None
        z_abs = (self.null_abs - self.abs_diff) / self.abs_se
        if self.alternative == "greater":
            return float(_norm.cdf(z_abs))
        if self.alternative == "less":
            return float(_norm.sf(z_abs))
        return float(2.0 * min(_norm.cdf(z_abs), _norm.sf(z_abs)))

    def p_value(self) -> float:
        """Return a presentation-only p-value for this displayed result.
        Typed decision evidence is emitted by the estimator computation bundle;
        this convenience method remains for result-table presentation and
        compatibility with the existing family layer.
        ``alternative`` ("two-sided" default; "greater"/"less" read one
        tail, matching scipy's own alternative= convention) and
        ``null_lift`` (0.0 default: the plain zero-null case) -- the exact
        same declared tail and shifted null as the interval. Family selection
        remains distinct from interval significance: nominal caps and additive
        sidecars can impose different cutoffs.

        A ``null_abs`` row (an absolute-margin secondary) instead reads
        the additive tail from ``Normal(abs_diff, abs_se)`` against
        ``null_abs`` -- the same construction `prob_favorable` uses for
        its own additive branch -- so this p-value and ``stat_sig``
        (``abs_lb``/``abs_ub`` vs ``null_abs``) test the same null on the
        same scale; a ``reference_kind="t"`` ``null_abs`` row refuses,
        same as `prob_favorable`, since a Normal tail would understate
        the t-reference uncertainty.

        A quantile row (``quantile_p_value`` set) instead returns that
        stored value directly: it was computed once, independent of
        ``null_lift``/``alpha``, by inverting the quantile estimator's own
        interval construction, so it cannot change with the alpha a
        multiplicity allocation assigns this row.

        Otherwise, a ``reference_kind="t"`` row reads
        ``lift.log_mean``/``lift.log_se`` against its ``reference_df`` t
        reference directly, tested against this row's own ``null_lift`` (transformed
        to this row's own ``scale``, exactly as the Normal branch
        below transforms its threshold) -- ``_posterior()`` refuses those
        rows, since the interval is a t quantile pair, not a Normal
        posterior. A sequential reference refuses through ``_posterior()``.
        Otherwise (``reference_kind == "normal"``) this reads the recovered
        Normal posterior's cdf/survival at ``null_lift`` (transformed to this
        row's own ``scale``, exactly as ``prob_beyond`` transforms its
        threshold): under the default flat prior the posterior is
        Normal(MLE, SE), so this is the exact frequentist p-value; under
        an informative prior it remains the unshrunk frequentist report
        only if the estimate was produced with a flat prior."""
        if self.confidence_set is not None:
            # Invert the declared single-level test: P(p <= u) <= u
            # for every u. No normal tail is inferred from endpoints.
            if isinstance(self.confidence_set.reference, BootstrapReference):
                from increment.estimation._winsor_bootstrap import bootstrap_p_value

                return bootstrap_p_value(
                    self.confidence_set.reference,
                    self.null_abs if self.null_abs is not None else self.null_lift,
                    relative=self.null_abs is None,
                )
            return self.confidence_set.alpha if self.stat_sig() else 1.0
        if self.null_abs is not None:
            return self._null_abs_p_value()
        if self.relative_unavailable_reason is not None:
            return _nonpositive_arm_mean_p_value(self.relative_unavailable_reason)
        if self.relative_confidence_set is not None:
            return self.relative_confidence_set.p_value(self.null_lift)
        if (quantile_p_value := self.quantile_p_value) is not None:
            return quantile_p_value
        if self.reference_kind == "binomial":
            if self.null_lift <= -1.0:
                _raise(
                    "estimation.results.lift.p_value_null_lift_not_representable_dof",
                    group_id=self.group_id,
                    metric=self.metric,
                    null_lift=self.null_lift,
                )
            bset = self.binomial_set
            assert bset is not None
            return bset.null_p_value(self.null_lift, self.alternative)
        if self.reference_kind == "t" or self.n_clusters is not None:
            if self.inference != "fixed":
                self._posterior()  # raises the sequential-inference refusal
            e = self.lift
            assert e is not None, "validated: lift is None only for reference_kind='binomial'"
            if e.log_mean is None or e.log_se is None:
                _raise(
                    "estimation.results.lift.liftestimate_dof_set",
                    group_id=self.group_id,
                    metric=self.metric,
                )
            if e.log_se <= 0:
                _raise(
                    "estimation.results.lift.liftestimate_log_se",
                    e_log_se=e.log_se,
                    group_id=self.group_id,
                    metric=self.metric,
                )
            if self.scale == "log" and self.null_lift <= -1.0:
                _raise(
                    "estimation.results.lift.p_value_null_lift_not_representable_dof",
                    group_id=self.group_id,
                    metric=self.metric,
                    null_lift=self.null_lift,
                )
            null = math.log1p(self.null_lift) if self.scale == "log" else self.null_lift
            z = (e.log_mean - null) / e.log_se
            reference_df = self.reference_df
            distribution = _t if reference_df is not None else _norm
            shapes = (reference_df,) if reference_df is not None else ()
            if self.alternative == "greater":
                return float(distribution.sf(z, *shapes))
            if self.alternative == "less":
                return float(distribution.cdf(z, *shapes))
            return float(2.0 * distribution.sf(abs(z), *shapes))
        p = self._posterior()
        if self.scale == "log" and self.null_lift <= -1.0:
            _raise(
                "estimation.results.lift.p_value_null_lift_not_representable_dof",
                group_id=self.group_id,
                metric=self.metric,
                null_lift=self.null_lift,
            )
        null = math.log1p(self.null_lift) if self.scale == "log" else self.null_lift
        if self.alternative == "greater":
            return p.cdf(null)
        if self.alternative == "less":
            return p.survival(null)
        return 2.0 * min(p.cdf(null), p.survival(null))


def _nonpositive_arm_mean_p_value(reason: RelativeUnavailableReason) -> float:
    """``p_value()``'s ``relative_unavailable_reason`` special case, split
    out to keep that method's own branching flat. Unlike unavailable
    covariance or zero relative variance, a nonpositive arm mean's log route
    is undefined, not indeterminate: the maximal, never-rejecting p-value
    IS the correct answer, not a missing one."""
    if reason == "nonpositive_arm_mean":
        return 1.0
    _raise("estimation.results.joint.unavailable", reason=reason)


def _fcr_alpha_for(alternative: str, fcr_alpha: float | Fraction) -> float:
    """Return the requested alpha whose effective value is ``fcr_alpha``.

    Directional alternatives double internally; two-sided alternatives do not.
    """
    exact = Fraction(fcr_alpha) / (2 if alternative != "two-sided" else 1)
    value = float(exact)
    return math.nextafter(value, 0.0) if Fraction(value) > exact else value


def _alpha_eff_for(alternative: str, alpha: float) -> float:
    """The central-equivalent alpha a call-level *alpha* displays under.

    The inverse of ``_fcr_alpha_for``: directional alternatives double it.
    """
    return alpha if alternative == "two-sided" else 2.0 * alpha


def _working_value(estimate: LiftEstimate, value: float) -> float:
    if estimate.scale != "log":
        return value
    if value <= -1.0:
        e = estimate.lift
        assert e is not None, "validated: lift is None only for reference_kind='binomial'"
        _raise(
            "estimation.results.lift.liftestimate_value_lb",
            e_lb=value,
            e_value=e.value,
            group_id=estimate.group_id,
            metric=estimate.metric,
        )
    return math.log1p(value)


def _fixed_fcr_parameters(estimate: LiftEstimate) -> tuple[float, float] | None:
    """Reconstruct the prior-free constructor's center and working SE."""
    from increment.estimation.inference import normal_posterior

    e = estimate.lift
    assert e is not None, "validated: lift is None only for reference_kind='binomial'"
    if e.log_mean is None or e.log_se is None:
        return None
    if e.log_se <= 0:
        _raise(
            "estimation.results.lift.liftestimate_interval_symmetric",
            group_id=estimate.group_id,
            metric=estimate.metric,
            scale=estimate.scale,
        )
    # Cluster t and additive infer_ate bypass the near-flat Normal update.
    # Welch and encouragement's additive rows still perform that update.
    if estimate.dof is not None or (
        estimate.scale == "linear"
        and estimate.value_scale == "absolute"
        and estimate.estimand not in ("itt", "compliance", "late")
    ):
        return e.log_mean, e.log_se
    posterior = normal_posterior(e.log_mean, e.log_se)
    return posterior.mu, posterior.sigma


def _recover_fixed_open_interval_parameters(
    estimate: Estimate,
    *,
    center: float,
    calibrated: float,
    persisted: tuple[float, float] | None,
) -> tuple[float, float]:
    """Recover an open fixed-Normal interval's center and standard error.

    Persisted constructor moments are authoritative. Endpoint inversion is a
    fallback only when the full-tail critical value is safely separated from
    zero; near the Normal median, endpoint rounding overwhelms that quotient.
    """
    if estimate.alpha is None:
        _raise("estimation.results.lift.open_interval_unrecoverable")
    if persisted is not None:
        return persisted
    z = tail_isf(_norm.isf, estimate.alpha, what="open fixed Normal interval")
    if abs(z) <= math.sqrt(math.ulp(1.0)):
        _raise("estimation.results.lift.open_interval_unrecoverable")
    sigma = (
        (calibrated - center) / z if estimate.open_side == "lower" else (center - calibrated) / z
    )
    return center, sigma


def _open_joint_far_bound(relative: RelativeConfidenceSet) -> Estimate | None:
    """A joint set's FCR display: its central interval with the far end opened.

    The retained endpoint and the stored total alpha/level are exactly the
    ones ``RelativeConfidenceSet.estimate`` publishes; opening the far end
    only enlarges the set, so it cannot raise noncoverage. A row whose
    central inversion already failed to bound that side is returned as-is:
    the ``None`` there is the inversion's own, not this relabeling's.
    """
    lift = relative.estimate()
    if lift is None or relative.alternative == "two-sided":
        return lift
    if lift.lb is None or lift.ub is None:
        return lift
    if relative.alternative == "greater":
        return lift.model_copy(update={"ub": None, "open_side": "upper"})
    return lift.model_copy(update={"lb": None, "open_side": "lower"})


def _reinverted_binomial_note(note: str | None, ci: BinomialInterval) -> str | None:
    """*note* with any superseded precision disclosure replaced by *ci*'s own."""
    from increment.estimation.binomial_rr import precision_note, without_precision_note

    return " | ".join(filter(None, (without_precision_note(note), precision_note(ci)))) or None


def open_bound_from_two_sided_at_target(estimate: LiftEstimate) -> LiftEstimate:
    """Convert a directional FCR parent built at total ``lift.alpha``.

    Callers request target_alpha/2 to compensate for directional doubling.
    Fixed Normal/t rows use the full allocated tail on their working scale;
    certified sequential rows are already inverted at their exact tail and
    are returned unchanged. A
    ``reference_kind="binomial"`` row instead reinverts the relevant
    Berger-Boos tail directly at the full stored alpha, placing its endpoint
    against the row's own ``null_lift`` as the first pass did (never a Normal/t
    endpoint reconstruction -- that would silently drop the exact method's
    coverage guarantee). Alpha and its nominal level remain unchanged.
    Absolute sidecars retain their central parent intervals. Two-sided,
    unavailable and already-open Normal/t/sequential rows are no-ops. Exact
    binomial rows are always reinverted, including greater and set-only rows,
    because their first pass spent the caller's half-alpha while FCR needs the
    full target alpha stored as the central-equivalent display alpha.
    Joint references reinvert their own set and open its far endpoint from the
    central inversion, never scalar-Wald endpoint recovery.
    """
    if estimate.relative_unavailable_reason is not None:
        return estimate
    relative = estimate.relative_confidence_set
    if relative is not None:
        if estimate.alternative not in ALTERNATIVE_VALUES:
            _raise("estimation.results.joint.invalid_set", reason="unknown alternative")
        updated = relative_confidence_set(
            relative.reference,
            alpha=relative.alpha,
            alternative=estimate.alternative,
        )
        return estimate.model_copy(
            update={
                "relative_confidence_set": updated,
                "lift": _open_joint_far_bound(updated),
            }
        )
    if estimate.sequential_result is not None:
        return estimate
    e = estimate.lift
    if estimate.confidence_set is not None and estimate.alternative != "two-sided":
        winsor_refuse(
            "reinversion_required", "Directional conversion requires confidence-set reinversion."
        )
    if estimate.alternative == "two-sided":
        return estimate
    if estimate.reference_kind == "binomial":
        from increment.estimation.binomial_rr import confidence_interval as _binomial_ci
        from increment.estimation.binomial_rr import nuisance_beta as _nuisance_beta
        from increment.estimation.binomial_rr import to_lift_bounds as _to_lift_bounds
        from increment.estimation.binomial_rr import validate_alternative as _validate_alternative

        bset = estimate.binomial_set
        assert bset is not None, "validated: reference_kind='binomial' rows carry a set"

        ci = _binomial_ci(
            bset.x_c,
            bset.n_c,
            bset.x_t,
            bset.n_t,
            alpha=bset.alpha,
            alternative=_validate_alternative(estimate.alternative),
            null_r=1.0 + estimate.null_lift,
        )
        ci_lower, ci_upper = _to_lift_bounds(ci)
        new_lift = (
            None
            if e is None
            else e.model_copy(
                update={
                    "lb": ci_lower,
                    "ub": ci_upper,
                    "open_side": "upper" if ci_upper is None else None,
                }
            )
        )
        new_set = bset.model_copy(
            update={
                "lower": ci_lower,
                "upper": ci_upper,
                "geometry": ci.geometry,
                "decision_alpha": bset.alpha,
                "nuisance_beta": _nuisance_beta(bset.alpha),
            }
        )
        return estimate.model_copy(
            update={
                "lift": new_lift,
                "binomial_set": new_set,
                "note": _reinverted_binomial_note(estimate.note, ci),
            }
        )
    if e is None or e.lb is None or e.ub is None:
        return estimate
    lower = estimate.alternative == "greater"
    bound = e.lb if lower else e.ub
    if estimate.inference == "fixed":
        if estimate.prior_shrunk or estimate.prior_spec is not None:
            _raise("estimation.results.lift.fcr_prior_unsupported")
        if e.alpha is None:
            _raise("estimation.results.lift.open_interval_unrecoverable")
        isf = student_t_isf if estimate.reference_kind == "t" else _norm.isf
        shapes = (estimate.reference_df,) if estimate.reference_df is not None else ()
        critical = tail_isf(isf, e.alpha, *shapes, what="fixed FCR endpoint")
        parameters = _fixed_fcr_parameters(estimate)
        if parameters is None:
            mu = _working_value(estimate, e.value)
            parent_critical = tail_isf(isf, e.alpha / 2, *shapes, what="fixed FCR parent")
            sigma = (mu - _working_value(estimate, e.lb)) / parent_critical
        else:
            mu, sigma = parameters
        if not math.isfinite(sigma) or sigma <= 0:
            _raise(
                "estimation.results.lift.liftestimate_interval_symmetric",
                group_id=estimate.group_id,
                metric=estimate.metric,
                scale=estimate.scale,
            )
        bounds = wald_bounds(mu, critical, sigma, what="fixed FCR endpoint")
        bound = bounds[0] if lower else bounds[1]
        if estimate.scale == "log":
            bound = resolvable_expm1(bound, what="fixed FCR relative endpoint")
    new_lift = e.model_copy(
        update={
            "lb": bound if lower else None,
            "ub": None if lower else bound,
            "open_side": "upper" if lower else "lower",
        }
    )
    reference = estimate.independent_mean_reference
    return estimate.model_copy(
        update={
            "lift": new_lift,
            "independent_mean_reference": reference.model_copy(
                update={"interval": "directional-fcr"}
            )
            if reference is not None
            else None,
        }
    )


class JointContrastReference(CodedModel, BaseModel):
    """Working joint reference for the additive numerator and denominator."""

    model_config = ConfigDict(frozen=True)
    a: float = Field(allow_inf_nan=False)
    c: float = Field(allow_inf_nan=False)
    var_a: float = Field(ge=0.0, allow_inf_nan=False)
    var_c: float = Field(ge=0.0, allow_inf_nan=False)
    cov_ac: float = Field(allow_inf_nan=False)
    kind: Literal["normal", "t"] = "normal"
    df: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _reference_consistent(self):
        if (self.kind == "t") != (self.df is not None):
            _raise("estimation.results.joint.invalid_reference", reason="kind and df disagree")
        if Fraction(self.cov_ac) ** 2 > Fraction(self.var_a) * Fraction(self.var_c):
            _raise(
                "estimation.results.joint.invalid_reference",
                reason="covariance is not positive semidefinite",
            )
        return self


def _joint_reference_from_exact(
    *,
    a: Fraction | float,
    c: Fraction | float,
    var_a: Fraction,
    var_c: Fraction,
    cov_ac: Fraction,
    kind: Literal["normal", "t"] = "normal",
    df: float | None = None,
) -> tuple[JointContrastReference | None, RelativeUnavailableReason | None]:
    """Enclose sampled PSD covariance without certifying a zero-variance effect."""
    determinant = var_a * var_c - cov_ac * cov_ac
    if var_a < 0 or var_c < 0 or determinant < 0:
        return None, "joint_covariance_indefinite"
    try:
        if determinant == 0 and c != 0.0:
            point_a, point_c = Fraction(a), Fraction(c)
            if (
                var_a * point_c * point_c + var_c * point_a * point_a
                == 2 * cov_ac * point_a * point_c
            ):
                return None, "zero_relative_variance"
        rounded_a, rounded_c = float(a), float(c)
        cross = float(cov_ac)
        error = abs(Fraction(cross) - cov_ac)
        scale = Fraction(1)
        if error:
            exponent = (
                var_a.numerator.bit_length()
                - var_a.denominator.bit_length()
                - var_c.numerator.bit_length()
                + var_c.denominator.bit_length()
            ) // 2
            scale = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
        # Balanced diagonal margins have product error**2, enclosing the cross-term error.
        diagonal = []
        for target in (var_a + error * scale, var_c + error / scale):
            value = float(target)
            if Fraction(value) < target:
                value = math.nextafter(value, math.inf)
            if not math.isfinite(value):
                return None, "joint_covariance_unrepresentable"
            diagonal.append(value)
    except (OverflowError, ValueError):
        return None, "joint_covariance_unrepresentable"
    return JointContrastReference(
        a=rounded_a,
        c=rounded_c,
        var_a=diagonal[0],
        var_c=diagonal[1],
        cov_ac=cross,
        kind=kind,
        df=df,
    ), None


def _decimal_fraction(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _fraction_endpoint(value: Fraction, *, upper: bool) -> float:
    try:
        result = float(value)
    except OverflowError as exc:
        raise ArithmeticError("endpoint_unrepresentable") from exc
    if (Fraction(result) < value) if upper else (Fraction(result) > value):
        result = math.nextafter(result, math.inf if upper else -math.inf)
    if not math.isfinite(result):
        raise ArithmeticError("endpoint_unrepresentable")
    return result


def _quadratic_endpoint(
    a: Fraction,
    b: Fraction,
    c: Fraction,
    approximation: Decimal,
    *,
    left: bool,
    upper: bool,
) -> float:
    result = float(approximation)
    midpoint = -b / (2 * a)
    for _ in range(4):
        if not math.isfinite(result):
            raise ArithmeticError("endpoint_unrepresentable")
        x = Fraction(result)
        polynomial = (a * x * x + b * x + c) * (1 if a > 0 else -1)
        if left:
            side = 1 if x > midpoint else (1 if polynomial < 0 else -1 if polynomial > 0 else 0)
        else:
            side = -1 if x < midpoint else (-1 if polynomial < 0 else 1 if polynomial > 0 else 0)
        if side >= 0 if upper else side <= 0:
            return result
        result = math.nextafter(result, math.inf if upper else -math.inf)
    raise ArithmeticError("endpoint_enclosure_unresolved")


def _quadratic_intervals(a: Fraction, b: Fraction, c: Fraction):
    if a == 0:
        if b == 0:
            return ((None, None),) if c <= 0 else ()
        root = _fraction_endpoint(-c / b, upper=b > 0)
        return ((None, root),) if b > 0 else ((root, None),)
    discriminant = b * b - 4 * a * c
    if discriminant <= 0:
        if a < 0:
            return ((None, None),)
        if discriminant < 0:
            return ()
        root = -b / (2 * a)
        return ((_fraction_endpoint(root, upper=False), _fraction_endpoint(root, upper=True)),)
    with localcontext() as context:
        context.prec = 80
        root_disc = _decimal_fraction(discriminant).sqrt()
        q = -(_decimal_fraction(b) + (root_disc if b >= 0 else -root_disc)) / 2
        low, high = sorted((q / _decimal_fraction(a), _decimal_fraction(c) / q))
    lower = _quadratic_endpoint(a, b, c, low, left=True, upper=a < 0)
    upper = _quadratic_endpoint(a, b, c, high, left=False, upper=a > 0)
    if a > 0:
        return ((lower, upper),)
    if lower >= upper:
        raise ArithmeticError("endpoints_not_separable")
    return ((None, lower), (upper, None))


def _relative_geometry(intervals):
    if not intervals:
        return "empty"
    if intervals == ((None, None),):
        return "all_real"
    if len(intervals) == 2:
        return "disconnected"
    return "one_sided" if None in intervals[0] else "bounded"


def _joint_statistic(reference: JointContrastReference, ratio: Fraction) -> Fraction | None:
    difference = Fraction(reference.a) - ratio * Fraction(reference.c)
    variance = (
        Fraction(reference.var_a)
        - 2 * ratio * Fraction(reference.cov_ac)
        + ratio * ratio * Fraction(reference.var_c)
    )
    if difference == 0:
        return Fraction(0)
    return difference * difference / variance if variance else None


def _joint_p_value(reference: JointContrastReference, null: float, alternative: str) -> float:
    boundary = Fraction(null)
    statistic = _joint_statistic(reference, boundary)
    if alternative != "two-sided":
        a, c = Fraction(reference.a), Fraction(reference.c)
        va, vc, cov = map(Fraction, (reference.var_a, reference.var_c, reference.cov_ac))

        def included(value):
            return value <= boundary if alternative == "greater" else value >= boundary

        if (c and included(a / c)) or (a == 0 and c == 0):
            return 1.0
        candidates = [statistic]
        if vc:
            candidates.append(c * c / vc)
        elif c == 0:
            candidates.append(a * a / va if va else None)
        slope = a * vc - c * cov
        if slope:
            turning_point = (a * cov - c * va) / slope
            if included(turning_point):
                candidates.append(_joint_statistic(reference, turning_point))
        statistic = min((value for value in candidates if value is not None), default=None)
    if statistic is None:
        return 0.0
    if statistic == 0:
        return 1.0
    if reference.kind == "normal" and statistic >= 1600:
        return math.ulp(0.0)  # The two-sided Normal tail beyond 40 is below one subnormal.
    with localcontext() as context:
        context.prec = 80
        if reference.kind == "t" and statistic >= 256:
            assert reference.df is not None, "validated Student reference"
            # At z >= 16 the alternating remainder reaches binary64 precision.
            log_z = math.nextafter(float(_decimal_fraction(statistic).ln() / 2), -math.inf)
            try:
                _, log_tail, _ = _log_tail_bounds(log_z, reference.df)
            except ArithmeticError as exc:
                _raise("estimation.results.joint.unavailable", reason=str(exc))
        else:
            z = _quadratic_endpoint(
                Fraction(1),
                Fraction(0),
                -statistic,
                _decimal_fraction(statistic).sqrt(),
                left=False,
                upper=False,
            )
            log_tail = float(_t.logsf(z, reference.df) if reference.kind == "t" else _norm.logsf(z))
    if not math.isfinite(log_tail):
        _raise("estimation.results.joint.unavailable", reason="tail_unresolvable")
    multiplier = 2.0 if alternative == "two-sided" else 1.0
    probability = math.exp(min(0.0, math.log(multiplier) + log_tail))
    return min(1.0, math.nextafter(probability, math.inf))


class RelativeConfidenceSet(CodedModel, BaseModel):
    """Outward-enclosed Fieller set under an explicitly approximate joint reference."""

    model_config = ConfigDict(frozen=True)
    reference: JointContrastReference
    alpha: float = Field(gt=0.0, lt=1.0, allow_inf_nan=False)
    alternative: Alternative = "two-sided"
    intervals: tuple[tuple[float | None, float | None], ...] = ()
    geometry: Literal[
        "bounded", "one_sided", "disconnected", "all_real", "empty", "unavailable"
    ] = "empty"
    reason: str | None = None
    qualification: Literal["working_approximation"] = "working_approximation"
    # The pre-closure inversion, recomputed by the validator on every
    # construction (including deserialization) rather than persisted: it is
    # a projection of `reference`/`alpha`/`alternative`, not independent state.
    _central: tuple[tuple[float | None, float | None], ...] = PrivateAttr(default=())

    @model_validator(mode="after")
    def _validate_geometry(self):
        central, intervals, geometry, reason = _relative_inversion(
            self.reference, self.alpha, self.alternative
        )
        for name, expected in (
            ("intervals", intervals),
            ("geometry", geometry),
            ("reason", reason),
        ):
            if name in self.model_fields_set and getattr(self, name) != expected:
                _raise(
                    "estimation.results.joint.invalid_set",
                    reason=f"{name} contradicts the persisted joint reference and alpha",
                )
            object.__setattr__(self, name, expected)
        # ty: ignore[invalid-assignment] -- pydantic PrivateAttr; frozen applies to fields, not this
        self._central = central
        return self

    def contains(self, value: float) -> bool | None:
        if not math.isfinite(value):
            _raise(
                "estimation.results.joint.invalid_set", reason="membership requires a finite value"
            )
        if self.geometry == "unavailable":
            return None
        return any(
            (lo is None or value >= lo) and (hi is None or value <= hi) for lo, hi in self.intervals
        )

    def p_value(self, null: float) -> float:
        """Reference tail; directional tests use one tail over the composite null."""
        if not math.isfinite(null):
            _raise("estimation.results.joint.invalid_set", reason="null must be finite")
        return _joint_p_value(self.reference, null, self.alternative)

    @computed_field
    @property
    def point_unavailable_reason(self) -> str | None:
        if self.reference.c == 0:
            return "zero_denominator"
        try:
            float(Fraction(self.reference.a) / Fraction(self.reference.c))
        except OverflowError:
            return "point_unrepresentable"
        return None

    @property
    def alpha_eff(self) -> float:
        """The displayed interval's total noncoverage budget.

        A directional set spends ``alpha`` on its single tail, so the
        central interval built from the same critical value carries
        ``2 * alpha`` across both -- the alpha-doubling identity every
        other inference path in the library displays under.
        """
        return _alpha_eff_for(self.alternative, self.alpha)

    def estimate(self) -> Estimate | None:
        """The displayed interval: central at ``alpha_eff``, never closed toward
        the alternative.

        A directional row displays both endpoints of the central inversion and
        labels them with the honest two-sided coverage, matching ``infer_lift``,
        ``infer_ate``'s scalar path, and this row's own additive sidecar. Only
        the FCR-selected re-estimation path opens the far endpoint, through
        ``open_bound_from_two_sided_at_target``. An endpoint that is ``None``
        here is one the inversion genuinely could not bound, recorded by
        ``open_side`` and by the set's own ``geometry``.
        """
        if self.reference.c == 0:
            return None
        try:
            point = float(Fraction(self.reference.a) / Fraction(self.reference.c))
        except OverflowError:
            return None
        if _relative_geometry(self._central) not in ("bounded", "one_sided"):
            return Estimate(value=point)
        lower, upper = self._central[0]
        alpha_eff = self.alpha_eff
        return Estimate(
            value=point,
            lb=lower,
            ub=upper,
            alpha=alpha_eff,
            level=math.fsum((1.0, -alpha_eff)),
            open_side="lower" if lower is None else "upper" if upper is None else None,
        )


def _validate_joint_relative_row(row: Any) -> bool:
    """Validate joint evidence consistently on primary, breakout, and daily rows."""
    relative = row.relative_confidence_set
    if relative is None and row.relative_unavailable_reason is None:
        return False
    common_invalid = (
        row.inference != "fixed"
        or row.value_scale != "relative"
        or getattr(row, "scale", "linear") != "linear"
        or row.binomial_set is not None
        or getattr(row, "prior_spec", None) is not None
        or getattr(row, "prior_shrunk", False)
    )
    if row.relative_unavailable_reason is not None:
        if (
            common_invalid
            or relative is not None
            or row.reference_kind not in ("normal", "t")
            or row.abs_diff is None
            or (
                row.lift is not None
                and any(
                    value is not None
                    for value in (
                        row.lift.lb,
                        row.lift.ub,
                        row.lift.level,
                        row.lift.alpha,
                        row.lift.open_side,
                        row.lift.log_mean,
                        row.lift.log_se,
                    )
                )
            )
        ):
            _raise(
                "estimation.results.joint.invalid_set",
                reason="unavailable relative covariance permits only additive and point output",
            )
        return True
    if relative is not None:
        if (
            common_invalid
            or row.reference_kind != relative.reference.kind
            or row.reference_df != relative.reference.df
            or row.alternative != relative.alternative
            or row.lift not in (relative.estimate(), _open_joint_far_bound(relative))
        ):
            _raise(
                "estimation.results.joint.invalid_set",
                reason="display, reference, alternative or prior contradicts the joint set",
            )
        from increment.estimation.inference import _joint_additive_bounds

        # Producers use this same projection; binary64 serialization is exact.
        ref = relative.reference
        if row.abs_diff != ref.a:
            _raise(
                "estimation.results.joint.invalid_set",
                reason="additive point contradicts joint reference",
            )
        expected_se = math.sqrt(ref.var_a)
        if row.abs_se != (expected_se or None):
            _raise(
                "estimation.results.joint.invalid_set",
                reason="additive variance contradicts joint reference",
            )
        if expected_se > 0.0 and row.abs_reference_kind not in ("normal", "t"):
            _raise(
                "estimation.results.joint.invalid_set",
                reason="additive uncertainty requires a persisted reference",
            )
        if expected_se == 0.0 and row.abs_reference_kind is not None:
            _raise(
                "estimation.results.joint.invalid_set",
                reason="zero-variance additive projection has no interval reference",
            )
        expected_abs = _joint_additive_bounds(
            row.abs_diff,
            row.abs_se,
            relative.alpha,
            row.alternative,
            row.abs_reference_df if row.abs_reference_kind == "t" else None,
        )
        if (row.abs_lb, row.abs_ub) != expected_abs:
            _raise(
                "estimation.results.joint.invalid_set",
                reason="additive bounds contradict joint reference and alpha",
            )
        if row.abs_alpha is not None and row.abs_alpha != relative.alpha_eff:
            _raise(
                "estimation.results.joint.invalid_set",
                reason="additive alpha contradicts the joint set",
            )
        return True
    return False


def relative_confidence_set(
    reference: JointContrastReference,
    *,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> RelativeConfidenceSet:
    """Invert the joint quadratic without floating-point topology decisions."""
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        _raise("estimation.results.joint.invalid_set", reason="alpha must be in (0, 1)")
    if alternative not in ALTERNATIVE_VALUES:
        _raise("estimation.results.joint.invalid_set", reason="unknown alternative")
    return RelativeConfidenceSet(
        reference=reference, alpha=alpha, alternative=cast(Alternative, alternative)
    )


def _central_inversion(reference: JointContrastReference, alpha: float, alternative: str):
    """Invert the joint quadratic at the reference tail, before any directional closure.

    A directional ``alternative`` spends its full ``alpha`` on one tail,
    which is the same critical value a two-sided read spends at
    ``alpha_eff = 2 * alpha`` -- so this is simultaneously the directional
    set's pre-closure geometry and the central interval the row displays
    under the library's one-sided display convention.
    """
    if alternative != "two-sided" and alpha >= 0.5:
        _raise("estimation.results.joint.invalid_set", reason="directional alpha must be below 0.5")
    tail = alpha / 2.0 if alternative == "two-sided" else alpha
    if 2.0 * tail > alpha and alternative == "two-sided":
        tail = math.nextafter(tail, 0.0)
    try:
        q = tail_isf(
            student_t_isf if reference.kind == "t" else _norm.isf,
            tail,
            *((reference.df,) if reference.df is not None else ()),
            what="joint relative reference",
        )
        a, c = Fraction(reference.a), Fraction(reference.c)
        q2 = Fraction(q) ** 2
        intervals = _quadratic_intervals(
            c * c - q2 * Fraction(reference.var_c),
            -2 * a * c + 2 * q2 * Fraction(reference.cov_ac),
            a * a - q2 * Fraction(reference.var_a),
        )
    except (ArithmeticError, InvalidRequestError) as exc:
        reason = str(exc) if isinstance(exc, ArithmeticError) else "critical_value_unavailable"
        return (), "unavailable", reason
    return intervals, _relative_geometry(intervals), None


def _relative_inversion(reference: JointContrastReference, alpha: float, alternative: str):
    """The published set: the central inversion, closed toward the alternative.

    The closure is what makes ``contains``/``stat_sig`` a one-sided test and
    keeps a disconnected central set from excluding a null that lies in its
    far component. It shapes the SET only; the displayed interval comes from
    the central inversion (see ``RelativeConfidenceSet.estimate``).
    """
    central, geometry, reason = _central_inversion(reference, alpha, alternative)
    if central and alternative == "greater":
        intervals = ((central[0][0], None),)
    elif central and alternative == "less":
        intervals = ((None, central[-1][1]),)
    else:
        return central, central, geometry, reason
    return central, intervals, _relative_geometry(intervals), None


LiftEstimate.model_rebuild()
