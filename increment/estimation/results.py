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
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    Self,
    TypedDict,
    assert_never,
    cast,
)

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

from increment._canonical import canonical_json_bytes, canonical_json_loads
from increment._display import format_interval
from increment._immutable import _FrozenMapping
from increment._literals import (
    ALTERNATIVE_VALUES,
    Alternative,
    MetricType,
    PreferredDirection,
    RowRole,
    ValueScale,
)
from increment.errors import (
    CodedError,
    CodedModel,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
)
from increment.estimation import binomial_rr, conversion_route
from increment.estimation._fcr_reinterval import (
    _additive_reference_df,
    _has_additive_interval,
    classify,
    nominal_alpha,
)
from increment.estimation._fixed_horizon import (
    _alpha_eff_for,
    _joint_additive_bounds,
    _resolve_fixed_horizon,
    binomial_abs_interval,
)
from increment.estimation._readout_refusals import refuse_legacy_sampling
from increment.estimation._student_t import _log_tail_bounds
from increment.estimation._tails import resolvable_expm1, student_t_isf, tail_isf, wald_bounds
from increment.estimation._winsor_bootstrap import bootstrap_p_value
from increment.estimation.armstats import IndependentMeanReference
from increment.estimation.conversion_route import CAPPED_NOTE_PREFIX, ESCALATED_NOTE_PREFIX
from increment.estimation.priors import MixturePosterior, MixturePrior, Normal, StudentTPrior
from increment.estimation.risk_difference import score_p_value
from increment.estimation.sequential_result import (
    AsymptoticSequentialResult,
    SequentialInferenceResult,
    SequentialResult,
    _point,
)
from increment.estimation.winsor import winsor_lift_parts
from increment.semantics.models import favorable_alternative
from increment.sequential_state import require_public_laws, sequential_refuse
from increment.winsor import (
    BootstrapReference,
    InfluenceReference,
    WinsorConfidenceSet,
    influence_p_value,
    winsor_refuse,
)


def _sequential_display_parts(result: SequentialResult, *, point_available: bool) -> list[str]:
    """Render persisted confidence geometry and reasons without recomputing inference."""
    if isinstance(result, AsymptoticSequentialResult):
        bounds = result.bounds
        intervals = (
            " ∪ ".join(
                format_interval(component.lower, component.upper) for component in bounds.components
            )
            or "∅"
        )
        parts = [f"ratio_interval={intervals}", f"geometry={bounds.status!r}"]
        if bounds.reason is not None:
            parts.append(f"bounds_reason={bounds.reason!r}")
    else:
        bounds = result.bounds
        interval = format_interval(bounds.lower, bounds.upper, empty=bounds.empty)
        parts = [f"ratio_interval={interval}"]
    if not point_available and result.point_reason is not None:
        parts.append(f"point_reason={result.point_reason!r}")
    parts.extend((f"confidence={float(1 - bounds.alpha):.1%}", f"status={bounds.status!r}"))
    return parts


RelativeUnavailableReason = Literal[
    # An arm mean too close to zero relative to its standard error for the
    # log-scale delta method; the additive difference decides the row.
    "delta_method_unreliable",
    "joint_covariance_indefinite",
    "joint_covariance_unrepresentable",
    "nonpositive_arm_mean",
    # Relative projection refused for representability or finiteness
    # (``relative_error_code`` names it); the additive projection stays usable.
    "relative_unrepresentable",
    "zero_relative_variance",
    # Exactly one arm is constant: the log scale has no uncertainty for it,
    # while the additive difference keeps the other arm's.
    "zero_variance",
]

# Reasons for which the arm summaries block the log scale itself: with no
# declared margin the additive difference tested against zero is the decision.
_ADDITIVE_DECISION_REASONS = frozenset(
    {"delta_method_unreliable", "nonpositive_arm_mean", "zero_variance"}
)


if TYPE_CHECKING:
    from increment.estimation._fcr_reinterval import Construction
    from increment.estimation.binomial_rr import BinomialInterval
    from increment.estimation.inference import LiftPosterior
    from increment.estimation.readout_types import ReadoutMetadata
    from increment.result_envelope import ResultCell
    from increment.winsor import WinsorRawState


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
#: The arithmetic assumption under which finite-sample labels are numerically qualified.
BinomialNumericalQualification = Literal[
    "scipy_special_function_error_model_conditional_v1", "legacy_unrecorded_v1"
]
#: The construction `BinomialConfidenceSet` reads: its nuisance envelope, integer thresholds
#: and numerical stop rule. A producer stamps it on every set it cuts.
BINOMIAL_METHOD: BinomialMethod = "binomial_bb_difference_v3"
BINOMIAL_NUMERICAL_QUALIFICATION: BinomialNumericalQualification = (
    "scipy_special_function_error_model_conditional_v1"
)


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
    stop rule included. ``numerical_qualification`` records that a newly produced finite-sample
    construction depends on the deployed SciPy/Boost error allowance; it is not a cross-build
    proof. Rows saved before that field existed default to ``legacy_unrecorded_v1``, which makes
    no arithmetic qualification claim. ``LiftEstimate.stat_sig()``/``p_value()`` recompute from
    the counts with the current rule, so a set cut under another construction would sit beside a
    verdict its own interval can contradict. A set that names another construction or
    qualification is refused when read.
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
    numerical_qualification: BinomialNumericalQualification = "legacy_unrecorded_v1"
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
        expected_beta = binomial_rr.nuisance_beta(self.decision_alpha)
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
        ``LiftEstimate.stat_sig()``/``p_value()`` and the tables' twin of them. Refused with
        ``estimation.binomial.tail_unrepresentable`` exactly where a fresh inversion of these
        counts at this null and level is: a set admitted because its own null was rejected on
        the nuisance domain alone does not answer a null the float margin leaves undecidable."""
        tail = self.decision_alpha / 2.0 if alternative == "two-sided" else self.decision_alpha
        return binomial_rr.null_p_value(
            1.0 + null_lift,
            self.x_c,
            self.n_c,
            self.x_t,
            self.n_t,
            self.nuisance_beta,
            alternative=alternative,
            tail=tail,
        )

    def null_rejection_tail(self, null_lift: float) -> Literal["greater", "less"]:
        """The tail a two-sided rejection of ``H0: R = 1 + null_lift`` lies on: the
        smaller of the two one-sided p-values the two-sided ``null_p_value`` doubles."""
        return binomial_rr.null_rejection_tail(
            1.0 + null_lift,
            self.x_c,
            self.n_c,
            self.x_t,
            self.n_t,
            self.nuisance_beta,
            tail=self.decision_alpha / 2.0,
        )


class DeltaConversionEvidence(CodedModel, BaseModel):
    """The count evidence persisted on every conversion or retention row the count rule
    routed to the delta method (``reference_kind="t"``, ``scale="log"``, no ``binomial_set``).

    ``dense_level`` is ``U``, the smallest level at which the four counts are dense
    (`conversion_route.dense_level`, in the convention of the level the row's p-value is
    compared with: ``alpha`` two-sided, the single directional tail otherwise; ``0.0`` for
    counts dense at every representable level). Every delta-route conversion row persists
    its counts and ``U``; its p-value is its delta p-value.

    ``composite`` marks a decision row read by a multiplicity family below its own level
    (a declared secondary family, ``run_breakout(correction="bh")``,
    ``select_exploratory_family``). Its ``p_value()`` is then the composite
    `conversion_route.composite_p_value`: its delta p-value when that is at least ``U``, else
    the smaller of ``escalated_p_value`` and ``U``. ``escalated_p_value`` is the finite-sample
    route's exact p-value against the row's own null, read at the row's nominal tail with the
    nuisance budget of its nominal level (never a finer reading), persisted only when the delta
    p-value is below ``U`` and the exact route could serve it with a p-value able to fall below
    ``U``. ``exact_unavailable`` is the refusal code of the exact route where it could not (an
    arm above its ceiling, a tail its float margin dominates); the row's p-value is then ``U``
    itself, and its ``note`` says so. At most one of the two is set, and only on a composite
    row. ``reading_null_lift`` and ``reading_alpha`` bind that reading to its null and
    nominal level. Retesting a copied row against a different relative null reads
    the exact evidence from its counts at that retained nominal level.

    A declared family and ``select_exploratory_family`` read the same composite from the same
    persisted counts (the exploratory path reads the exact p-value on demand at selection), so
    a row corrected after the fact carries the evidence the family selected on.
    """

    model_config = ConfigDict(frozen=True)

    x_c: int = Field(ge=0)
    n_c: int = Field(ge=1)
    x_t: int = Field(ge=0)
    n_t: int = Field(ge=1)
    dense_level: float = Field(ge=0.0, lt=1.0, allow_inf_nan=False)
    composite: bool = False
    escalated_p_value: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    exact_unavailable: str | None = None
    reading_null_lift: float | None = Field(
        default=None, allow_inf_nan=False, exclude_if=lambda v: v is None
    )
    reading_alpha: float | None = Field(
        default=None, gt=0.0, lt=1.0, allow_inf_nan=False, exclude_if=lambda v: v is None
    )

    @model_validator(mode="after")
    def _validate_counts(self):
        if self.x_c > self.n_c or self.x_t > self.n_t:
            _raise(
                "estimation.results.delta.counts_out_of_range",
                x_c=self.x_c,
                n_c=self.n_c,
                x_t=self.x_t,
                n_t=self.n_t,
            )
        if self.escalated_p_value is not None and self.exact_unavailable is not None:
            _raise(
                "estimation.results.delta.dense_level",
                reason="an exact p-value and the exact route's refusal cannot both be recorded",
            )
        if not self.composite and (
            self.escalated_p_value is not None or self.exact_unavailable is not None
        ):
            _raise(
                "estimation.results.delta.dense_level",
                reason="an exact reading belongs only to a row read by a family (composite)",
            )
        if self.composite and (self.reading_null_lift is None or self.reading_alpha is None):
            _raise(
                "estimation.results.delta.dense_level",
                reason="composite evidence requires its reading null and nominal level",
            )
        return self

    @property
    def counts(self) -> tuple[int, int, int, int]:
        """``(x_c, n_c, x_t, n_t)``."""
        return self.x_c, self.n_c, self.x_t, self.n_t

    def dense_at(self, alpha: float) -> bool:
        """Whether the counts are dense at level ``alpha`` (in ``dense_level``'s convention)."""
        return alpha >= self.dense_level


class _ReferenceFields(TypedDict):
    dof: float | None
    reference_kind: Literal["normal", "t", "sequential"]
    reference_df: float | None


def _reference_fields(
    dof: float | None, *, kind: Literal["normal", "t", "sequential"] | None = None
) -> _ReferenceFields:
    """Build the persisted reference triple from a legacy degrees of freedom."""
    return {
        "dof": dof,
        "reference_kind": kind or ("t" if dof is not None else "normal"),
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
        "estimation.results.lift.open_interval_unrecoverable": "The selected interval endpoint cannot be recovered from its persisted sampling reference; recompute the analysis from source data.",
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
        "estimation.results.lift.fcr_prior_unsupported": "This legacy prior-bound interval has no proven prior-free sampling state for FCR conversion; recompute the analysis from source data.",
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
        "estimation.results.lift.p_value_missing_log_mean_or_se": "LiftEstimate for {metric!r}/{group_id!r}: the sampling reference is unavailable because log_mean or log_se is missing; recompute the analysis from sufficient source data.",
        "estimation.results.lift.liftestimate_log_se": "LiftEstimate for {metric!r}/{group_id!r}: log_se must be positive to compute a t-reference p-value, got log_se={e_log_se}",
        "estimation.results.lift.p_value_null_lift_not_representable_dof": "LiftEstimate for {metric!r}/{group_id!r}: null_lift <= -1 is not representable on the log scale, got {null_lift}",
        "estimation.results.lift.reference_df_required_for_t": "estimate row for {metric!r}/{group_id!r}: reference_kind='t' requires a finite reference_df > 0, got {reference_df!r}",
        "estimation.results.lift.reference_df_set_for_normal": "estimate row for {metric!r}/{group_id!r}: reference_kind={reference_kind!r} requires reference_df=None, got {reference_df!r}",
        "estimation.results.lift.dof_reference_df_mismatch": "estimate row for {metric!r}/{group_id!r}: dof={dof!r} does not match reference_df={reference_df!r} -- a t-reference row's legacy dof column must agree with the reference it was actually cut from",
        "estimation.results.lift.inference_reference_kind_mismatch": "estimate row for {metric!r}/{group_id!r}: inference={inference!r} and reference_kind={reference_kind!r} disagree; a sequential interval has a sequential reference and a fixed-horizon interval a normal or t one",
        "estimation.results.lift.posterior_components_invalid": "LiftEstimate for {metric!r}/{group_id!r}: stored posterior components are invalid: {reason}",
        "estimation.results.joint.invalid_reference": "Invalid joint relative reference: {reason}",
        "estimation.results.joint.invalid_set": "Invalid relative confidence set: {reason}",
        "estimation.results.joint.unavailable": "Joint relative inference is unavailable: {reason}",
        "estimation.results.binomial.interval_inverted": "binomial confidence set has lower={lower} > upper={upper}",
        "estimation.results.binomial.counts_out_of_range": "binomial confidence set counts out of range: x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}",
        "estimation.results.binomial.decision_metadata": "binomial confidence set has inconsistent decision metadata: {reason}",
        "estimation.results.binomial.obsolete_construction": "binomial confidence set records method {method!r} (None: none recorded); this version reads only {supported!r}. A set naming an incompatible construction, or persisted without naming one, may use different envelopes, thresholds or numerical rules, so its stored interval and a recomputed verdict could disagree. Re-run the analysis to cut the set again.",
        "estimation.results.binomial.posterior_unavailable": "metric={metric!r} group_id={group_id!r}: no Normal/lognormal posterior exists for a reference_kind='binomial' row -- the exact binomial method is a frequentist test-inversion, not a posterior; chance_to_beat/prob_beyond/prob_within/risk_if_shipped and their favorable variants are unavailable here. Use stat_sig()/p_value() (both binomial-set-aware) or the persisted lift/binomial_set bounds directly.",
        "estimation.results.lift.binomial_lift_availability": "metric={metric!r} group_id={group_id!r}: {reason}",
        "estimation.results.lift.absolute_reference_mismatch": "absolute reference {kind!r} with df {df!r} does not fit this row: a t reference carries its df, and a score reference belongs only to an exact binomial row",
        "estimation.results.lift.absolute_alpha_without_interval": "abs_alpha={abs_alpha!r} is the alpha an additive interval was cut at, but this row carries no additive interval (abs_lb/abs_ub)",
        "estimation.results.lift.absolute_alpha_mismatch": "abs_alpha={abs_alpha!r} contradicts the alpha {alpha!r} of the relative interval the same call cut",
        "estimation.results.delta.counts_out_of_range": "delta-route conversion evidence counts out of range: x_c={x_c}, n_c={n_c}, x_t={x_t}, n_t={n_t}",
        "estimation.results.delta.dense_level": "delta-route conversion evidence is inconsistent: {reason}",
        "estimation.results.delta.reference_mismatch": "estimate row for {metric!r}/{group_id!r}: delta-route conversion evidence belongs only to a reference_kind='t' row on the log scale with no binomial_set, got reference_kind={reference_kind!r}",
    },
)
_raise = raiser(_REFUSALS)


def _validate_absolute_reference_fields(row: Any) -> Any:
    if (row.abs_reference_kind == "t") != (row.abs_reference_df is not None) or (
        row.abs_reference_kind == "score"
        and (row.reference_kind != "binomial" or row.binomial_set is None)
    ):
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


def _validate_delta_evidence(row: Any) -> Any:
    """Delta-route conversion evidence sits only on a t-reference log-scale row with no binomial
    set, and its dense level is the count rule's own inversion of the row's counts under the
    row's alternative (every row type shares this check)."""
    evidence = row.delta_evidence
    if evidence is None:
        return row
    if (
        row.reference_kind != "t"
        or getattr(row, "scale", "log") != "log"
        or row.binomial_set is not None
    ):
        _raise(
            "estimation.results.delta.reference_mismatch",
            metric=row.metric,
            group_id=row.group_id,
            reference_kind=row.reference_kind,
        )
    if not conversion_route.dense_level_holds(
        *evidence.counts, level=evidence.dense_level, alternative=row.alternative
    ):
        _raise(
            "estimation.results.delta.dense_level",
            reason=(
                f"dense_level={evidence.dense_level!r} is not the rule's level for counts "
                f"{evidence.counts} under {row.alternative!r} (re-run the analysis to cut "
                "the row under this version's rule)"
            ),
        )
    return row


class PosteriorComponents(CodedModel, BaseModel):
    """Exact stored normal-mixture posterior components, in original order."""

    model_config = ConfigDict(frozen=True, revalidate_instances="always", extra="forbid")

    weights: tuple[float, ...]
    means: tuple[float, ...]
    sigmas: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_components(self):
        lengths = (len(self.weights), len(self.means), len(self.sigmas))
        if not lengths[0] or len(set(lengths)) != 1:
            _raise(
                "estimation.results.lift.posterior_components_invalid",
                metric="<posterior>",
                group_id="<posterior>",
                reason=f"component lengths must be equal and nonempty; got {lengths}",
            )
        if not (
            all(math.isfinite(weight) for weight in self.weights)
            and all(math.isfinite(mean) for mean in self.means)
            and all(math.isfinite(sigma) for sigma in self.sigmas)
        ):
            _raise(
                "estimation.results.lift.posterior_components_invalid",
                metric="<posterior>",
                group_id="<posterior>",
                reason="weights, means, and sigmas must be finite",
            )
        if any(weight < 0.0 for weight in self.weights):
            _raise(
                "estimation.results.lift.posterior_components_invalid",
                metric="<posterior>",
                group_id="<posterior>",
                reason="weights must be nonnegative",
            )
        if any(sigma <= 0.0 for sigma in self.sigmas):
            _raise(
                "estimation.results.lift.posterior_components_invalid",
                metric="<posterior>",
                group_id="<posterior>",
                reason="sigmas must be positive",
            )
        try:
            total = math.fsum(self.weights)
        except (OverflowError, ValueError):
            total = math.inf
        if not math.isfinite(total) or abs(total - 1.0) > 1e-9:
            _raise(
                "estimation.results.lift.posterior_components_invalid",
                metric="<posterior>",
                group_id="<posterior>",
                reason=f"weights must sum to one within 1e-9; got {total}",
            )
        return self


_ROW_CONTEXT_FIELDS = frozenset(
    {
        "sampling_reason_context",
        "posterior_reason_context",
        "absolute_error_context",
        "relative_error_context",
        "failure_context",
        "decision_scope_reason_context",
        "weight_diagnostics_reason_context",
    }
)


def _freeze_row_contexts(row: Any) -> None:
    """Replace each reason context on *row* with an immutable, canonical,
    picklable mapping, so a refusal survives pickle, deepcopy and JSON."""

    def freeze(value):
        if isinstance(value, Mapping):
            frozen = _FrozenMapping({key: freeze(item) for key, item in value.items()})
            canonical_json_bytes(dict(frozen))
            return frozen
        if isinstance(value, (tuple, list)):
            return tuple(freeze(item) for item in value)
        return value

    for name in _ROW_CONTEXT_FIELDS:
        value = getattr(row, name)
        if value is not None:
            object.__setattr__(row, name, freeze(value))


class _RowIdentity(CodedModel, BaseModel):
    """Leading identity fields shared, in the same declared order, by
    LiftEstimate, BreakoutEstimate and DailyLiftEstimate. Only a field
    already first in every subclass belongs here: pydantic prepends
    inherited fields ahead of a subclass's own, so a later-declared
    field (e.g. `null_lift` on DailyLiftEstimate) must stay put.
    """

    model_config = ConfigDict(frozen=True, revalidate_instances="always")

    metric: str
    group_id: str
    method: str  # which Method produced this estimate
    method_role: Literal["decision", "sensitivity"]
    inference: str = "fixed"  # "fixed" | "always_valid" | "asymptotic_mean"
    alternative: Alternative = "two-sided"  # "two-sided" | "greater" | "less"

    sampling_available: bool | None = None
    sampling_reason_code: str | None = None
    sampling_reason_context: Mapping[str, object] | None = None
    posterior_available: bool | None = None
    posterior_components: PosteriorComponents | None = None
    posterior_reason_code: str | None = None
    posterior_reason_context: Mapping[str, object] | None = None
    posterior_model: Literal["normal", "mixture"] | None = None
    posterior_scale: Literal["log", "linear"] | None = None
    posterior_estimate: float | None = None
    posterior_lb: float | None = None
    posterior_ub: float | None = None
    posterior_level: float | None = None
    posterior_alpha: float | None = None
    posterior_latent_mean: float | None = None
    posterior_latent_sd: float | None = None
    # Stored at the declared relative null on the metric's good side; None for
    # absolute-margin nulls.
    posterior_prob_favorable: float | None = None
    posterior_prob_favorable_reason_code: str | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    absolute_error_code: str | None = None
    absolute_error_context: Mapping[str, object] | None = None
    relative_error_code: str | None = None
    relative_error_context: Mapping[str, object] | None = None
    failure_code: str | None = None
    failure_context: Mapping[str, object] | None = None
    source_snapshot_id: str | None = None
    #: The declared ``Metric.type`` behind ``metric``, stamped by every readout. None
    #: when unknown: a row built by a low-level estimator, the design-level compliance
    #: row, or a result saved before this field existed.
    metric_type: MetricType | None = Field(default=None, exclude_if=lambda value: value is None)
    decision_scope_complete: bool | None = None

    decision_scope_reason_code: str | None = None
    decision_scope_reason_context: Mapping[str, object] | None = None
    family_id: str | None = None
    multiplicity_status: (
        Literal[
            "declared_plan",
            "unassigned_in_plan",
            "undeclared_plan",
            "exploratory_unadjusted",
            "exploratory_family",
        ]
        | None
    ) = None
    weight_diagnostics_available: bool | None = None
    weight_diagnostics_reason_code: str | None = None
    weight_diagnostics_reason_context: Mapping[str, object] | None = None
    weight_definition: str | None = None
    weight_grain: Literal["unit", "cluster"] | None = None
    control_weight_ess: float | None = None
    treatment_weight_ess: float | None = None
    control_weight_max_share: float | None = None
    treatment_weight_max_share: float | None = None
    control_weight_n: int | None = None
    treatment_weight_n: int | None = None

    n_control: int | None = Field(default=None, exclude_if=lambda v: v is None)
    n_treat: int | None = Field(default=None, exclude_if=lambda v: v is None)
    mean_control: float | None = Field(default=None, exclude_if=lambda v: v is None)
    mean_treat: float | None = Field(default=None, exclude_if=lambda v: v is None)

    def _posterior(self) -> LiftPosterior | None:
        """Reconstruct the stored posterior shared by every decision-row view."""
        if self.posterior_available is not True or self.inference != "fixed":
            return None
        if self.posterior_model == "normal":
            if self.posterior_latent_mean is None or self.posterior_latent_sd is None:
                return None
            return Normal(mu=self.posterior_latent_mean, sigma=self.posterior_latent_sd)
        if self.posterior_model == "mixture" and self.posterior_components is not None:
            import numpy as np

            return MixturePosterior(
                weights=np.asarray(self.posterior_components.weights, dtype=float),
                means=np.asarray(self.posterior_components.means, dtype=float),
                sigmas=np.asarray(self.posterior_components.sigmas, dtype=float),
            )
        return None

    @model_validator(mode="after")
    def _freeze_readout_contexts(self):
        _freeze_row_contexts(self)
        return self

    @model_validator(mode="after")
    def _validate_posterior_payload(self):
        payload = (
            self.posterior_estimate,
            self.posterior_lb,
            self.posterior_ub,
            self.posterior_level,
            self.posterior_alpha,
            self.posterior_latent_mean,
            self.posterior_latent_sd,
            self.posterior_prob_favorable,
        )

        def invalid(reason: str) -> NoReturn:
            _raise(
                "estimation.results.lift.posterior_components_invalid",
                metric=self.metric,
                group_id=self.group_id,
                reason=reason,
            )

        if self.posterior_available is not True:
            if (
                self.posterior_model is not None
                or self.posterior_scale is not None
                or self.posterior_components is not None
                or any(value is not None for value in payload)
            ):
                invalid("posterior state is present while posterior_available is not true")
            return self
        required = (
            self.posterior_estimate,
            self.posterior_lb,
            self.posterior_ub,
            self.posterior_level,
            self.posterior_alpha,
        )
        if self.posterior_model is None or self.posterior_scale is None:
            invalid("available posterior is missing its model or scale")
        if any(value is None or not math.isfinite(value) for value in required):
            invalid("available posterior is missing finite interval payload")
        if self.posterior_model == "normal":
            if (
                self.posterior_components is not None
                or self.posterior_latent_mean is None
                or self.posterior_latent_sd is None
                or not math.isfinite(self.posterior_latent_mean)
                or not math.isfinite(self.posterior_latent_sd)
                or self.posterior_latent_sd <= 0.0
                or getattr(self, "prior_spec", None) is not None
            ):
                invalid("Normal posterior requires finite latent parameters and no mixture state")
        elif self.posterior_model == "mixture":
            if (
                self.posterior_components is None
                or self.posterior_latent_mean is not None
                or self.posterior_latent_sd is not None
                or getattr(self, "prior_spec", None) is None
            ):
                invalid(
                    "mixture posterior requires stored components and a separate declared prior"
                )
        else:
            invalid("available posterior model is unsupported")
        return self

    @field_serializer(
        "sampling_reason_context",
        "posterior_reason_context",
        "absolute_error_context",
        "relative_error_context",
        "failure_context",
        "decision_scope_reason_context",
        "weight_diagnostics_reason_context",
        when_used="always",
    )
    def _serialize_readout_context(self, value: Mapping[str, object] | None) -> object:
        def thaw(item: object) -> object:
            if isinstance(item, Mapping):
                return {key: thaw(nested) for key, nested in item.items()}
            if isinstance(item, (tuple, list)):
                return [thaw(nested) for nested in item]
            return item

        return None if value is None else thaw(value)

    @field_validator("alternative", mode="before")
    @classmethod
    def _validated_alternative(cls, value: object) -> object:
        return binomial_rr.validate_alternative(value) if isinstance(value, str) else value

    @field_validator("posterior_components", mode="before")
    @classmethod
    def _decode_posterior_components(cls, value: object, info: ValidationInfo) -> object:
        if value is None or isinstance(value, PosteriorComponents):
            return value
        if isinstance(value, str):
            metric = str(info.data.get("metric", "<frame>"))
            group_id = str(info.data.get("group_id", "<frame>"))
            try:
                parsed = canonical_json_loads(value)
            except (TypeError, ValueError):
                _raise(
                    "estimation.results.lift.posterior_components_invalid",
                    metric=metric,
                    group_id=group_id,
                    reason="frame payload is not valid canonical JSON",
                )
            if not isinstance(parsed, Mapping) or canonical_json_bytes(parsed).decode() != value:
                _raise(
                    "estimation.results.lift.posterior_components_invalid",
                    metric=metric,
                    group_id=group_id,
                    reason="frame payload must be a canonical JSON object",
                )
            value = parsed
        if isinstance(value, Mapping):
            return PosteriorComponents.model_validate(value)
        return value

    @property
    def reason(self) -> str | None:
        """Return the concise persisted reason for a coded cell failure."""
        if self.failure_code is None:
            return None
        context = self.failure_context or {}
        reason = context.get("reason")
        return reason if isinstance(reason, str) else self.failure_code

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Copy the row, freezing updated contexts and rebuilding composite readings."""
        copied = super().model_copy(update=update, deep=deep)
        if update and not _ROW_CONTEXT_FIELDS.isdisjoint(update):
            _freeze_row_contexts(copied)
        evidence = getattr(copied, "delta_evidence", None)
        if (
            update is None
            or "null_lift" not in update
            or update["null_lift"] == getattr(self, "null_lift", None)
            or evidence is None
            or not evidence.composite
        ):
            return copied

        assert evidence.reading_alpha is not None
        evidence = evidence.model_copy(
            update={"escalated_p_value": None, "exact_unavailable": None}
        )
        delta_row = copied.model_copy(update={"delta_evidence": None})
        retested_evidence, composite_note = _composite_evidence(
            cast(LiftEstimate, delta_row), evidence, evidence.reading_alpha, copied.alternative
        )
        note_parts = [
            part
            for part in (
                conversion_route.without_composite_note(getattr(copied, "note", None)),
                composite_note,
            )
            if part is not None
        ]
        return super(_RowIdentity, copied).model_copy(
            update={
                "delta_evidence": retested_evidence,
                "note": "; ".join(note_parts) or None,
            },
            deep=deep,
        )


#: Display names of the recorded family procedures, shared by every family label.
_FAMILY_PROCEDURE_LABELS = {"e_bh": "e-BH", "bh": "Benjamini-Hochberg", "bonferroni": "Bonferroni"}


def _selection_qualification(row: Any) -> str | None:
    """Describe an affirmative family selection from the row's retained provenance."""
    if getattr(row, "discovery", None) is not True:
        return None
    status = getattr(row, "multiplicity_status", None)
    q = getattr(row, "family_q", None)
    size = getattr(row, "family_size", None)
    family = f"retained family size {size}" if size is not None else "retained family"
    level = f" at q={q:g}" if q is not None else ""
    procedure = getattr(row, "family_procedure", None)
    procedure_name = _FAMILY_PROCEDURE_LABELS.get(procedure)
    procedure_label = (
        f"{procedure_name} family selection"
        if procedure_name is not None
        else "Family selection; procedure not recorded"
    )
    if getattr(row, "sequential_result", None) is not None:
        guarantee = getattr(row, "family_guarantee", None)
        qualification = (
            "finite-sample guarantee"
            if guarantee == "finite_sample"
            else "asymptotic-sequential guarantee"
            if guarantee == "asymptotic_sequential"
            else "no controlled guarantee"
        )
        trigger = (
            "; triggered family guarantee under arm-independent triggering, no pre-trigger "
            "treatment effect, trigger-order stationarity, and looks/stops/freezes using only "
            "revealed triggered-chain information, not early assigned results (including "
            "the same metric)"
            if getattr(row, "analysis_population", None) == "triggered" and guarantee is not None
            else ""
        )
        guarantee_kind = (
            "FWER"
            if procedure == "bonferroni"
            else "FDR"
            if procedure in ("e_bh", "bh")
            else "family"
        )
        return (
            f"{procedure_label}{level}; arbitrary-dependence {guarantee_kind} selection{trigger}; "
            f"{qualification}; {family}"
        )
    if status == "exploratory_family" and q is not None and procedure == "bh":
        return (
            f"{procedure_label}{level}; {family}; selected interval is FCR-adjusted; "
            "FDR qualification requires independence or positive dependence; exploratory, "
            "not confirmatory"
        )
    if status == "exploratory_family":
        return f"{procedure_label}{level}; {family}; exploratory, not confirmatory"
    if q is not None and procedure == "bh":
        return (
            f"{procedure_label}{level}; {family}; FDR qualification requires independence or "
            "positive dependence"
        )
    return f"{procedure_label}{level}; {family}"


class LiftEstimate(_RowIdentity):
    """A lift estimate, carrying the metadata to identify which (metric,
    method, group) and its persisted inference reference.

    ``value`` is the prior-free sampling estimate: ``exp(log_mean) - 1``
    for ``scale="log"`` or ``log_mean`` for ``scale="linear"``. The CI is
    the corresponding sampling interval on this scale. Posterior estimates
    and decision probabilities are persisted separately and never replace
    the sampling estimate or interval.

    Normal posterior latent parameters are stored separately from the sampling
    interval. A mixture posterior is stored exactly in
    ``posterior_components`` as equal-length ``weights``, ``means`` and
    ``sigmas`` tuples, in component order. Zero-weight components are retained
    with their original means and sigmas so persisted positions are stable;
    weights are validated but never normalized. ``prior_spec`` remains the
    declared Student-t/mixture prior and is not used to reconstruct the
    already-updated posterior.

    ``reference_kind`` persists the reference: "normal", "t", "sequential",
    "binomial", or "confidence_set". Winsor confidence sets persist raw
    construction state and endpoint statuses, expose no posterior, and
    reinvert through ``reintervalize(alpha)``. Their public
    ``confidence_set.qualification`` names the construction that ran:
    ``pointwise_asymptotic_model_conditioned_v1`` (bootstrap candidate),
    ``pointwise_asymptotic_influence_v1`` (analytic influence interval) or
    ``uniform_support_conditioned_v1`` (rank method); none is an unqualified
    finite-sample promise. Bootstrap p-values invert stored roots; analytic
    p-values are the two-sided normal tail of the stored pivot; rank p-values
    are alpha when the set excludes the null, else one.
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
    # The reference abs_lb/abs_ub were cut from: Welch or cluster "t" (with abs_reference_df),
    # "normal", or "score" (the risk-difference score interval of a binomial row). None means
    # no reference was persisted, as on a row saved before binomial rows carried one.
    abs_reference_kind: Literal["normal", "t", "score"] | None = None
    abs_reference_df: float | None = Field(default=None, allow_inf_nan=False, gt=0)
    abs_alpha: float | None = Field(default=None, gt=0.0, lt=1.0, allow_inf_nan=False)
    allocation_alpha: float | None = Field(
        default=None, exclude_if=lambda value: value is None, gt=0.0, lt=1.0, allow_inf_nan=False
    )
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
    # The four counts and composite p-value evidence of every conversion or retention row the
    # count rule routed to the delta method; see DeltaConversionEvidence. None for every other
    # row, including a delta-route conversion row persisted before the evidence existed, which
    # reads as a plain Wald row (its p-value is the delta p-value at every level).
    delta_evidence: DeltaConversionEvidence | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    ds: date | datetime | str | int | float | None = None  # None for total-grain estimates
    scale: Literal["log", "linear"] = "log"
    # Scale on which the underlying posterior law is parameterized.
    # Declared Student-t/MixturePrior input; exact updated posterior state is
    # separately persisted in posterior_components.
    prior_spec: StudentTPrior | MixturePrior | None = None
    # Historical persisted marker that an informative prior was supplied.
    # Modern sampling availability and posterior availability are authoritative.
    prior_shrunk: bool = False
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
    family_procedure: Literal["e_bh", "bonferroni", "bh"] | None = None
    # Canonical procedure used by the retained family; absent in older payloads.
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

    _check_delta_evidence: ClassVar[Any] = model_validator(mode="after")(_validate_delta_evidence)

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
            ref = self.independent_mean_reference
            lift = self.lift

            def expected_interval() -> tuple[
                tuple[float | None, float | None], float | None, float | None, str | None
            ]:
                resolved = _resolve_fixed_horizon(
                    ref.alpha,
                    ref.alternative,
                    dof=None,
                    arm_ns=None,
                    prior=None,
                    independent_means=ref,
                )
                assert resolved.crit is not None
                bounds: tuple[float | None, float | None] = (
                    ref.point - resolved.crit * ref.se(),
                    ref.point + resolved.crit * ref.se(),
                )
                open_side = None
                if ref.interval == "directional-fcr":
                    if ref.alternative == "two-sided":
                        winsor_refuse(
                            "invalid_state", "Directional mean interval needs a direction."
                        )
                    critical = tail_isf(
                        student_t_isf, resolved.alpha_eff, ref.df, what="fixed FCR endpoint"
                    )
                    one_sided = wald_bounds(
                        ref.point, critical, ref.se(), what="fixed FCR endpoint"
                    )
                    bounds = (
                        (one_sided[0], None)
                        if ref.alternative == "greater"
                        else (None, one_sided[1])
                    )
                    open_side = "upper" if ref.alternative == "greater" else "lower"
                return bounds, resolved.alpha_eff, math.fsum((1.0, -resolved.alpha_eff)), open_side

            # A withheld interval (``relative_error_code`` names the refusal)
            # keeps the point and statistics only; it is accepted only when the
            # reference's level really cannot be represented.
            withheld = (
                lift is not None
                and lift.lb is None
                and lift.ub is None
                and self.relative_error_code is not None
            )
            if withheld:
                try:
                    expected_interval()
                except CodedError as exc:
                    if exc.code != "estimation.tails.unresolvable":
                        raise
                else:
                    winsor_refuse(
                        "invalid_state",
                        "Independent mean interval withheld at a representable level.",
                    )
                expected_bounds: tuple[float | None, float | None] = (None, None)
                expected_alpha: float | None = None
                expected_level: float | None = None
                expected_open: str | None = None
            else:
                expected_bounds, expected_alpha, expected_level, expected_open = expected_interval()
            if (
                self.reference_kind != "t"
                or self.reference_df != ref.df
                or self.value_scale != "absolute"
                or self.scale != "linear"
                or lift is None
                or lift.value != ref.point
                or lift.log_se != ref.se()
                or lift.log_mean != ref.point
                or (lift.lb, lift.ub) != expected_bounds
                or lift.alpha != expected_alpha
                or lift.level != expected_level
                or lift.open_side != expected_open
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
        if self.failure_code is not None and self.lift is None:
            if self.binomial_set is not None or self.confidence_set is not None:
                _raise(
                    "estimation.results.lift.binomial_lift_availability",
                    group_id=self.group_id,
                    metric=self.metric,
                    reason="failed cell cannot carry a confidence set",
                )
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
        expected_value = binomial_rr.point_lift(bset.x_c, bset.n_c, bset.x_t, bset.n_t)
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
        result = self.sequential_result
        if self.reference_kind != "sequential":
            if result is not None:
                sequential_refuse("source.invalid", "fixed result cannot carry sequential evidence")
            return self
        if result is None:
            if (
                self.analysis_population == "triggered"
                and self.failure_code == "readout.cell.unsupported_request"
                and self.failure_context is not None
                and self.failure_context.get("reason")
                in ("triggered_chain_uncommitted", "triggered_uptake_unsupported")
                and self.sampling_available is False
                and self.decision_scope_complete is False
                and self.lift is None
            ):
                # This narrowly typed row is an explicit unsupported request,
                # not a sequential estimate; it must not invent a checkpoint.
                return self
            sequential_refuse(
                "continuation.legacy", "sequential result lacks certified raw-state evidence"
            )
        if isinstance(result, AsymptoticSequentialResult) != (self.inference == "asymptotic_mean"):
            sequential_refuse(
                "source.invalid", "displayed validity regime differs from its checkpoint"
            )
        require_public_laws((result.checkpoint.model,), "LiftEstimate replay")
        if self.analysis_population != result.checkpoint.population:
            sequential_refuse("source.invalid", "result and checkpoint populations disagree")
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

    def as_cells(
        self,
        metadata: ReadoutMetadata | None = None,
        *,
        created_at: datetime | None = None,
    ) -> list[ResultCell]:
        """Return this finished row as one portable result cell."""
        from importlib import import_module

        # Upward: as_cells builds a ResultCell, and ResultCell.retained_row rebuilds this row.
        ResultCell = import_module("increment.result_envelope").ResultCell

        return [ResultCell.from_row(self, metadata, created_at=created_at)]

    def _reissue_allocation(self) -> float | None:
        allocations: list[float] = []
        if self.family_procedure is not None:
            if self.family_nominal_alpha is not None:
                allocations.append(self.family_nominal_alpha)
            if self.allocation_alpha is not None:
                allocations.append(self.allocation_alpha)
            if self.discovery and self.family_threshold is not None:
                allocations.append(self.family_threshold)
        elif self.role == "primary":
            if self.allocation_alpha is not None:
                allocations.append(self.allocation_alpha)
            if self.lift is not None and self.lift.alpha is not None:
                allocations.append(self.lift.alpha)
            if self.abs_alpha is not None:
                allocations.append(self.abs_alpha)
            if self.confidence_set is not None:
                allocations.append(self.confidence_set.alpha)
            if self.relative_confidence_set is not None:
                allocations.append(self.relative_confidence_set.alpha)
            if self.alternative != "two-sided":
                allocations = [value / 2.0 for value in allocations]
        return min(allocations) if allocations else None

    def _with_reissued_allocation(self, row: LiftEstimate, alpha: float) -> LiftEstimate:
        allocation = self._reissue_allocation()
        directional_primary = (
            row.role == "primary"
            and row.alternative != "two-sided"
            and self.family_procedure is None
        )
        stored_alpha = (
            row.lift.alpha
            if row.lift is not None and row.lift.alpha is not None
            else row.abs_alpha
            if row.abs_alpha is not None
            else alpha * 2.0
            if directional_primary
            else alpha
        )
        call_alpha = stored_alpha / 2.0 if directional_primary else stored_alpha
        if allocation is None or call_alpha < allocation:
            allocation = call_alpha
        if directional_primary:
            allocation *= 2.0
        return row.model_copy(update={"allocation_alpha": allocation})

    def reintervalize(
        self, alpha: float, *, interval_alternative: str | None = None
    ) -> LiftEstimate:
        """Rebuild this row's interval using its stored decision allocation."""
        return self.reissue(alpha, interval_alternative=interval_alternative)

    def reissue(self, alpha: float, *, interval_alternative: str | None = None) -> LiftEstimate:
        """Rebuild persisted inference without widening its declared allocation.

        Selected family rows use their stored construction and threshold: conversion
        rows are rerouted against retained counts, while constructions without a
        retained reissue path refuse. Unadjusted rows retain the general
        reintervalization behavior below. A ``t`` row rebuilds its log-scale and
        retained additive intervals; endpoints outside float64 withhold only the
        affected interval and record the refusal code.
        """
        allocation = self._reissue_allocation()
        if allocation is not None and alpha > allocation:
            raise CodedError(
                "Reissued interval exceeds the row's declared or selected alpha allocation.",
                code="results.operation.alpha_above_allocation",
                context={"requested_alpha": alpha, "allocated_alpha": allocation},
            )
        if self.family_procedure is not None and self.discovery is True:
            if interval_alternative is not None and interval_alternative != self.alternative:
                raise CodedError(
                    "A selected family interval must retain its registered alternative.",
                    code="results.operation.unsupported",
                    context={
                        "operation": "reinterval",
                        "reason": "family reissue cannot change the registered alternative",
                    },
                )
            construction, reason = classify(self)
            if construction is None:
                raise CodedError(
                    "The selected family interval cannot be reissued from retained state.",
                    code="results.operation.unsupported",
                    context={"operation": "reinterval", "reason": reason},
                )
            return self._with_reissued_allocation(
                reinterval_at_fcr(self, construction, alpha), alpha
            )

        relative = (
            self.reference_kind == "t"
            and self.reference_df is not None
            and self.lift is not None
            and self.lift.log_mean is not None
            and self.lift.log_se is not None
        )
        additive = (
            self.abs_diff is not None
            and self.abs_se is not None
            and self.abs_reference_kind == "t"
            and self.abs_reference_df is not None
        )
        if relative or (
            additive and self.lift is None and self.relative_unavailable_reason is not None
        ):
            return self._with_reissued_allocation(
                self._reintervalize_t(
                    alpha, interval_alternative, relative=relative, additive=additive
                ),
                alpha,
            )
        if self.confidence_set is None:
            winsor_refuse(
                "reinversion_required", "This operation requires a persisted winsor confidence set."
            )
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
            if isinstance(region.reference, BootstrapReference | InfluenceReference)
            else None,
        )
        return self._with_reissued_allocation(
            self.model_copy(
                update={
                    "confidence_set": result.confidence_set,
                    "lift": result.lift,
                    "abs_lb": result.abs_lb,
                    "abs_ub": result.abs_ub,
                    "abs_alpha": result.abs_alpha,
                }
            ),
            alpha,
        )

    def _reintervalize_t(
        self,
        alpha: float,
        interval_alternative: str | None,
        *,
        relative: bool,
        additive: bool,
    ) -> LiftEstimate:
        alternative = interval_alternative or self.alternative
        alpha_eff = alpha if alternative == "two-sided" else 2.0 * alpha
        update: dict[str, Any] = {
            "alternative": alternative,
            "independent_mean_reference": (
                self.independent_mean_reference.model_copy(
                    update={"alpha": alpha, "alternative": alternative, "interval": "central"}
                )
                if self.independent_mean_reference is not None
                else None
            ),
        }
        if additive:
            assert (
                self.abs_diff is not None
                and self.abs_se is not None
                and self.abs_reference_df is not None
            )
            abs_critical = tail_isf(
                student_t_isf,
                alpha_eff / 2.0,
                self.abs_reference_df,
                what="absolute t reintervalization",
            )
            abs_lb, abs_ub = wald_bounds(
                self.abs_diff,
                abs_critical,
                self.abs_se,
                what="absolute t reintervalization endpoint",
            )
            update.update({"abs_lb": abs_lb, "abs_ub": abs_ub, "abs_alpha": alpha_eff})
        if not relative:
            return self.model_copy(update=update)
        lift = self.lift
        assert lift is not None and lift.log_mean is not None and lift.log_se is not None
        assert self.reference_df is not None
        try:
            critical = tail_isf(
                student_t_isf, alpha_eff / 2.0, self.reference_df, what="t reintervalization"
            )
            lower, upper = wald_bounds(
                lift.log_mean, critical, lift.log_se, what="t reintervalization endpoint"
            )
            if self.scale == "log":
                lower = resolvable_expm1(lower, what="t reintervalization endpoint")
                upper = resolvable_expm1(upper, what="t reintervalization endpoint")
        except CodedError as exc:
            # The level, not the evidence, is what cannot be represented: keep
            # the point and the log-scale statistics the selection was decided
            # on (p_value() still answers from them) and withhold only the
            # interval, recording the exact refusal beside it.
            update.update(
                {
                    "lift": lift.model_copy(
                        update={
                            "lb": None,
                            "ub": None,
                            "alpha": None,
                            "level": None,
                            "open_side": None,
                        }
                    ),
                    "relative_error_code": exc.code,
                    "relative_error_context": exc.context,
                }
            )
            return self.model_copy(update=update)
        update["lift"] = lift.model_copy(
            update={
                "lb": lower,
                "ub": upper,
                "alpha": alpha_eff,
                "level": 1.0 - alpha_eff,
                "open_side": None,
            }
        )
        # The interval is representable at this level: an earlier level's
        # withholding no longer describes the row.
        update.update({"relative_error_code": None, "relative_error_context": None})
        return self.model_copy(update=update)

    def __repr__(self) -> str:
        """Compact interactive representation of the published estimate."""
        identity = (
            f"metric={self.metric!r}, group={self.group_id!r}, method={self.method!r}, "
            f"role={self.method_role!r}, estimand={self.estimand!r}, "
            f"population={self.analysis_population!r}, inference={self.inference!r}, "
            f"alternative={self.alternative!r}, scale={self.value_scale!r}"
        )
        if self.lift is None and self.failure_code is not None:
            context = self.failure_context or {}
            reason = context.get("reason")
            if reason is None:
                reason = {
                    key: value
                    for key, value in context.items()
                    if key not in {"metric", "group_id"}
                }
            if not reason:
                reason = self.failure_code
            details = [identity]
            details.append(f"failure={self.failure_code!r}")
            if reason is not None:
                details.append(f"reason={reason!r}")
            return f"LiftEstimate({', '.join(details)})"
        if self.confidence_set is not None:
            region = self.confidence_set
            qualification = _selection_qualification(self)
            suffix = (
                f", selection_qualification={qualification!r}" if qualification is not None else ""
            )
            return (
                f"LiftEstimate({identity}, "
                f"value={region.point!r}, lower={region.relative.lower!r}, "
                f"upper={region.relative.upper!r}, reference='confidence_set'{suffix})"
            )
        unit = self.value_scale
        if self.relative_unavailable_reason is not None:
            qualification = _selection_qualification(self)
            suffix = (
                f", selection_qualification={qualification!r}" if qualification is not None else ""
            )
            return (
                f"LiftEstimate({identity}, "
                f"value={None if self.lift is None else self.lift.value!r}, "
                f"relative_unavailable={self.relative_unavailable_reason!r}, "
                f"abs_diff={self.abs_diff!r}{suffix})"
            )
        if self.relative_confidence_set is not None:
            relative = self.relative_confidence_set
            qualification = _selection_qualification(self)
            suffix = (
                f", selection_qualification={qualification!r}" if qualification is not None else ""
            )
            return (
                f"LiftEstimate({identity}, "
                f"value="
                f"{None if self.lift is None else self.lift.value!r} ({unit}), "
                f"relative_set={relative.geometry}:{relative.intervals!r}, "
                f"discovery={self.discovery!r}{suffix})"
            )
        if self.sequential_result is not None:
            value = "unavailable" if self.lift is None else f"{self.lift.value:+.4g}"
            interval = ", ".join(
                _sequential_display_parts(
                    self.sequential_result, point_available=self.lift is not None
                )
            )
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
            decision_alpha = row_decision_alpha(self)
            # The level this construction decides at, shown where it is not simply the
            # interval's own level complement: a plan-allocated (Bonferroni-split) or
            # family-reissued row, or a closed directional interval spending one tail.
            if decision_alpha is not None and (
                self.role is not None
                or self.family_procedure is not None
                or not math.isclose(decision_alpha, self.lift.alpha or decision_alpha)
            ):
                interval += f", decision_alpha={decision_alpha:.4g}"
        route = conversion_route_label(self)
        if route is not None:
            interval += f", route={route!r}"
        qualification = _selection_qualification(self)
        suffix = f", selection_qualification={qualification!r}" if qualification is not None else ""
        return (
            f"LiftEstimate({identity}, "
            f"value={value} ({unit}), "
            f"{interval}, discovery={self.discovery!r}{suffix})"
        )

    __str__ = __repr__

    def __repr_args__(self) -> list[tuple[str, object]]:
        """Give Rich a compact row summary instead of the full persisted model."""
        return [
            ("metric", self.metric),
            ("group", self.group_id),
            ("population", self.analysis_population),
            ("scale", self.value_scale),
            ("summary", self.__repr__()),
        ]

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

    def chance_to_beat(self) -> float | None:
        """Stored posterior probability that lift is positive, if available."""
        return row_chance_to_beat(self)

    def prob_beyond(self, threshold: float) -> float | None:
        """Stored-posterior probability that lift exceeds ``threshold``."""
        self._refuse_selected_sequential_probability()
        posterior = self._posterior()
        if posterior is None:
            return None
        if self.posterior_scale == "log":
            if threshold <= -1.0:
                _raise("estimation.results.lift.threshold_representable_log", threshold=threshold)
            threshold = math.log1p(threshold)
        return posterior.survival(threshold)

    def _refuse_selected_sequential_probability(self) -> None:
        result = self.sequential_result
        if result is not None and result.bounds.alpha != result.checkpoint.cell.alpha:
            sequential_refuse(
                "route.unsupported",
                "selected sequential intervals do not define posterior decision probabilities",
            )

    def prob_favorable(self) -> float | None:
        """Stored posterior probability on the metric's declared good side, at its
        null; a sequential row refuses, as ``p_value()`` does."""
        _require_fixed_horizon(
            self, "sequential confidence sequences do not define posterior probabilities"
        )
        posterior = self._posterior()
        if posterior is None:
            return None
        if self.null_abs is not None and (
            self.reference_kind == "t" or self.n_clusters is not None
        ):
            _raise(
                "estimation.results.lift.p_value_cluster_robust_null_abs",
                reference_df=self.abs_reference_df
                if self.abs_reference_kind == "t"
                else self.reference_df,
                group_id=self.group_id,
                metric=self.metric,
            )
        side = row_favorable_side(self)
        if side is None or self.null_abs is not None:
            return None
        threshold = math.log1p(self.null_lift) if self.posterior_scale == "log" else self.null_lift
        return posterior.cdf(threshold) if side == "less" else posterior.survival(threshold)

    def prob_within(self, threshold: float) -> float | None:
        """Stored-posterior probability that lift lies inside a symmetric band."""
        if self.value_scale == "absolute":
            if not threshold > 0.0:
                _raise(
                    "estimation.results.lift.prob_within_threshold_positive", threshold=threshold
                )
        elif not 0.0 < threshold < 1.0:
            _raise(
                "estimation.results.lift.prob_within_threshold_unit_interval", threshold=threshold
            )
        posterior = self._posterior()
        if posterior is None:
            return None
        if self.posterior_scale == "log":
            lower, upper = math.log1p(-threshold), math.log1p(threshold)
        else:
            lower, upper = -threshold, threshold
        return posterior.probability_between(lower, upper)

    def chance_to_beat_favorable(self) -> float | None:
        """Stored posterior probability on the favorable side of zero; see
        :func:`row_chance_to_beat_favorable`."""
        return row_chance_to_beat_favorable(self)

    def risk_if_shipped(self) -> float | None:
        """Stored-posterior expected negative lift, when a posterior is available."""
        posterior = self._posterior()
        if posterior is None or self.posterior_scale is None:
            return None
        return posterior.expected_negative_part(scale=self.posterior_scale)

    def risk_if_shipped_favorable(self) -> float | None:
        """Stored-posterior expected loss on the declared unfavorable side."""
        posterior = self._posterior()
        if posterior is None or self.posterior_scale is None or self.preferred_direction is None:
            return None
        if self.preferred_direction == "decrease":
            return posterior.expected_positive_part(scale=self.posterior_scale)
        return posterior.expected_negative_part(scale=self.posterior_scale)

    def require_sequential_result(self) -> SequentialResult:
        """Return the authoritative stopped likelihood state or refuse a fixed row."""
        if self.sequential_result is None:
            sequential_refuse("source.invalid", "result has no registered sequential checkpoint")
        return self.sequential_result

    def require_exact_sequential_result(self) -> SequentialInferenceResult:
        """Narrow to exact likelihood evidence without relabeling an AsympCS."""
        result = self.require_sequential_result()
        if not isinstance(result, SequentialInferenceResult):
            sequential_refuse(
                "route.unsupported", "asymptotic confidence sets are not exact evidence"
            )
        return result

    def require_asymptotic_sequential_result(self) -> AsymptoticSequentialResult:
        """Return the scalar mean construction and its complete confidence geometry."""
        result = self.require_sequential_result()
        if not isinstance(result, AsymptoticSequentialResult):
            sequential_refuse("route.unsupported", "result is not a scalar mean AsympCS")
        return result

    def stat_sig(self) -> bool | None:
        """Whether this row's own interval excludes its own null, honoring
        ``alternative``; ``None`` when the row carries no usable sampling
        evidence (a failed cell, or sampling marked unavailable). The same
        verdict ``BreakoutEstimate``/``DailyLiftEstimate`` and the readout
        frame report: see :func:`row_stat_sig`. Independent of, and never
        gated by, this row's family ``discovery`` verdict (see
        `increment.estimation.family.family_discovery`): a selected
        decision row keeps ``discovery=True`` even when a binding
        nominal-alpha cap leaves ``stat_sig()`` False, and family
        selection never demotes it to match.
        """
        return row_stat_sig(self)

    def rejection_side(self) -> RejectionSide | None:
        """The side of its null this row's significant result lies on; ``None``
        exactly when ``stat_sig()`` is not True. See :func:`row_rejection_side`."""
        return row_rejection_side(self)

    def p_value(self) -> float | None:
        """Return a presentation-only sampling p-value for this displayed result.

        Sampling evidence is computed from the persisted prior-free reference,
        independently of any separately stored posterior, and is never divided
        by a multiplicity factor: compare it with the row's own decision alpha
        (``lift.alpha``, halved for a directional row) or read ``discovery``
        for the family verdict. See :func:`row_p_value`.
        """
        return row_p_value(self)


def failed_cell_row(
    *,
    metric: str,
    group_id: str,
    method: str,
    method_role: Literal["decision", "sensitivity"],
    estimand: Literal["itt", "late"],
    alternative: str,
    code: str,
    context: Mapping[str, object],
    prior_declared: bool,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    preferred_direction: PreferredDirection | None = None,
) -> LiftEstimate:
    """An unavailable cell whose refusal is its failure and its sampling reason, and
    its posterior reason under a declared prior; a LATE cell is on its additive scale.
    The cell keeps the null and favorable side of the hypothesis it was to test."""
    return LiftEstimate(
        metric=metric,
        group_id=group_id,
        method=method,
        estimand=estimand,
        value_scale="absolute" if estimand == "late" else "relative",
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        preferred_direction=preferred_direction,
        method_role=method_role,
        failure_code=code,
        failure_context=context,
        sampling_available=False,
        sampling_reason_code=code,
        sampling_reason_context=context,
        posterior_available=False if prior_declared else None,
        posterior_reason_code=code if prior_declared else None,
        posterior_reason_context=context if prior_declared else None,
    )


if TYPE_CHECKING:
    from increment.breakout.estimates import BreakoutEstimate, DailyLiftEstimate
    from increment.estimation.contrast_results import ContrastResult

    type DecisionRow = LiftEstimate | BreakoutEstimate | DailyLiftEstimate


def row_chance_to_beat(row: DecisionRow) -> float | None:
    """Stored-posterior probability that a decision row's lift is positive."""
    return _row_chance_of_side(row, "greater")


def row_favorable_side(row: DecisionRow) -> Literal["greater", "less"] | None:
    """The side of zero a decision row's metric declares favorable, read from its
    ``preferred_direction`` alone (never the test ``alternative``); ``None`` for a
    metric that declares none."""
    return favorable_alternative(getattr(row, "preferred_direction", None))


def row_chance_to_beat_favorable(row: DecisionRow) -> float | None:
    """Stored-posterior probability that a decision row's lift lies on its
    :func:`row_favorable_side` of zero, or ``None`` for a metric that declares no
    direction. The decrease side reads the posterior's lower tail, never
    ``1 - P(lift > 0)``, so a vanishing probability keeps its magnitude."""
    side = row_favorable_side(row)
    return None if side is None else _row_chance_of_side(row, side)


def _row_chance_of_side(row: DecisionRow, side: Literal["greater", "less"]) -> float | None:
    if isinstance(row, LiftEstimate):
        row._refuse_selected_sequential_probability()
    posterior = row._posterior()
    if posterior is None:
        return None
    # Zero lift is zero on the log scale as well: log1p(0) == 0.
    return posterior.survival(0.0) if side == "greater" else posterior.cdf(0.0)


def _require_fixed_horizon(row: DecisionRow, reason: str) -> None:
    """Refuse a sequential row: a confidence sequence has no fixed-look endpoint
    distribution or posterior to summarize."""
    if row.inference != "fixed":
        sequential_refuse("route.unsupported", reason)


#: Per-arm unit count below which a breakout or daily estimate is flagged ``low_reliability``:
#: a disclosed caveat on an approximate interval's coverage. The row's verdict, p-value and
#: family membership are those of any other row.
DEFAULT_RELIABILITY_FLOOR = 50


#: The side of the null a rejecting confidence set lies on: above it, below it, or on both
#: sides (a set disconnected around the null, or an empty one), where no side is established.
RejectionSide = Literal["greater", "less", "split"]


def _interval_side(
    lower: float | Fraction | None, upper: float | Fraction | None, null: float | Fraction
) -> RejectionSide:
    if lower is not None and lower > null:
        return "greater"
    if upper is not None and upper < null:
        return "less"
    return "split"


def _set_side(
    intervals: tuple[tuple[float | None, float | None], ...], null: float
) -> RejectionSide:
    if intervals and all(lo is not None and lo > null for lo, _ in intervals):
        return "greater"
    if intervals and all(hi is not None and hi < null for _, hi in intervals):
        return "less"
    return "split"


def _sequential_decision(result: SequentialResult) -> RejectionSide | Literal[False]:
    if not result.rejects():
        return False
    if isinstance(result, AsymptoticSequentialResult):
        # The crossed tail is the sign of the contrast at the registered null.
        contrast = result.bounds.estimator_contrast
        assert contrast is not None
        return "greater" if contrast > 0 else "less"
    bounds = result.bounds
    if bounds.empty:
        return "split"
    return _interval_side(bounds.lower, bounds.upper, 1 + result.checkpoint.cell.null_lift)


def _additive_decision(row: DecisionRow, null: float) -> RejectionSide | Literal[False]:
    if not _additive_excludes(row, null):
        return False
    return _interval_side(row.abs_lb, row.abs_ub, null)


def _row_decision(row: DecisionRow) -> RejectionSide | Literal[False] | None:
    """The row's own verdict: the side of a rejection, ``False`` for a genuine
    non-rejection, ``None`` for no usable sampling verdict. Both
    :func:`row_stat_sig` and :func:`row_rejection_side` project it, so a
    rejection always names its side."""
    if row.failure_code is not None or _sampling_availability(row) is False:
        return None
    if row.sequential_result is not None:
        return _sequential_decision(row.sequential_result)
    confidence_set = getattr(row, "confidence_set", None)
    if confidence_set is not None:
        null = row.null_abs if row.null_abs is not None else row.null_lift
        interval = confidence_set.additive if row.null_abs is not None else confidence_set.relative
        if not interval.excludes(null):
            return False
        return _interval_side(interval.lower.value, interval.upper.value, null)
    if row.null_abs is not None:
        return _additive_decision(row, row.null_abs)
    if row.relative_unavailable_reason is not None:
        return _additive_decision(row, 0.0) if _additive_zero_reference(row) is not None else False
    if row.relative_confidence_set is not None:
        if row.relative_confidence_set.contains(row.null_lift) is not False:
            return False
        return _set_side(row.relative_confidence_set.intervals, row.null_lift)
    if row.reference_kind == "binomial":
        bset = row.binomial_set
        assert bset is not None
        # ``p_two < alpha`` is ``min(p_+, p_-) < alpha / 2``: doubling is exact.
        if not bset.null_p_value(row.null_lift, row.alternative) < bset.decision_alpha:
            return False
        if row.alternative != "two-sided":
            return row.alternative
        return bset.null_rejection_tail(row.null_lift)
    lift = row.lift
    if lift is None:
        return None
    if row.alternative == "greater":
        rejected = lift.lb is not None and lift.lb > row.null_lift
    elif row.alternative == "less":
        rejected = lift.ub is not None and lift.ub < row.null_lift
    else:
        rejected = lift.excludes(row.null_lift)
    return _interval_side(lift.lb, lift.ub, row.null_lift) if rejected else False


def row_stat_sig(row: DecisionRow) -> bool | None:
    """Whether ``row``'s own interval excludes its own null, honoring
    ``alternative`` -- the one verdict every row type and the readout frame
    report.

    ``None`` means no usable sampling verdict: an explicit producer failure,
    sampling marked unavailable, or a breakout cell excluded before estimation.
    ``False`` is a genuine non-rejection. A ``low_reliability`` cell decides the
    same way: the flag is a caveat, not a verdict.

    A ``null_abs`` row decides against the additive interval
    (``abs_lb``/``abs_ub``) instead of the relative one, and that takes
    precedence over a persisted ``binomial_set``; missing additive endpoints
    report False rather than falling back to the relative interval. A row whose
    relative scale is undefined because an arm mean is not positive, with no
    declared margin, decides the same additive interval against zero (see
    ``_additive_zero_reference``).

    A ``reference_kind="binomial"`` row (point-backed or not) instead tests
    ``r0 = 1 + null_lift`` for exclusion from the exact confidence set directly
    via the persisted counts -- the same Berger-Boos tail functions the set was
    inverted from, so this is exact even when ``null_lift`` differs from the
    null the row's own ``lift``/``binomial_set`` bounds were built against.
    """
    decision = _row_decision(row)
    return True if isinstance(decision, str) else decision


def row_rejection_side(row: DecisionRow) -> RejectionSide | None:
    """The side of its own null ``row``'s rejection lies on, read from the same
    verdict as :func:`row_stat_sig`; ``None`` exactly when that verdict is not True.

    A two-sided binomial rejection lies on the tail with the smaller one-sided
    p-value; a sequential exact set on the side of the registered null ratio its
    bounds exclude; an asymptotic set on the sign of its contrast at that null.
    ``"split"`` names a rejecting set that lies on both sides of the null.
    """
    decision = _row_decision(row)
    return decision if isinstance(decision, str) else None


def row_verdict(row: Any) -> bool | None:
    """The row's ``stat_sig()`` verdict, or None when the row cannot compute one."""
    stat_sig = getattr(row, "stat_sig", None)
    if not callable(stat_sig):
        return None
    try:
        return stat_sig()
    except CodedError:
        return None


def contrast_evidence_available(result: ContrastResult) -> bool:
    """Whether the declared reference supplies typed evidence.

    Envelope evidence does not depend on the descriptive sample standard error.
    """
    return (
        result.residual_p_value is not None
        if result.method == "switchback_unit_variance_envelope"
        else result.standard_error is not None and result.standard_error > 0.0
    )


def contrast_stat_sig(result: ContrastResult) -> bool:
    """Return the contrast's table/readout decision verdict."""
    if not contrast_evidence_available(result):
        return False
    estimate = result.estimate
    if result.method == "switchback_unit_variance_envelope":
        assert result.residual_p_value is not None
        return result.residual_p_value < result.alpha
    if result.alternative == "greater":
        return estimate.lb is not None and estimate.lb > result.null_abs
    if result.alternative == "less":
        return estimate.ub is not None and estimate.ub < result.null_abs
    return (
        estimate.lb is not None
        and estimate.ub is not None
        and not (estimate.lb <= result.null_abs <= estimate.ub)
    )


def contrast_rejection_side(result: ContrastResult) -> RejectionSide | None:
    """The side of its null the contrast's rejection lies on; ``None`` exactly when
    :func:`contrast_stat_sig` is False.

    The envelope's point is the retained-window total, whose effect is the point
    over the positive slope, so its side is the sign of the exact residual
    ``point - null_abs * slope`` its p-value tests, recorded when the evidence was
    computed; the stored point and slope are roundings that can flip that sign.
    """
    if not contrast_stat_sig(result):
        return None
    estimate = result.estimate
    if result.alternative != "two-sided":
        return result.alternative
    if result.method == "switchback_unit_variance_envelope":
        assert result.residual_side is not None
        return result.residual_side
    if estimate.lb is not None and estimate.lb > result.null_abs:
        return "greater"
    return "less"


def row_p_value(row: DecisionRow) -> float | None:
    """``row``'s prior-free sampling p-value against its own null, or ``None``
    when the row carries no usable sampling evidence (the cases
    :func:`row_stat_sig` reports as ``None``). It is never divided by a
    multiplicity factor; the row's decision alpha is what it is compared with.
    A fixed-horizon row with an interval satisfies ``p_value() <= alpha``
    (``<`` for an exact binomial row) exactly when ``stat_sig()``, for the
    alpha its construction decides at (:func:`row_decision_alpha`). A
    delta-route conversion row read by a multiplicity family
    (``delta_evidence.composite``) reports the composite p-value of its counts
    (`conversion_route.composite_p_value`): its delta p-value when that is at
    least the level its counts become dense at, else the smaller of its exact
    p-value and that level. The composite agrees with ``stat_sig()`` at the
    row's own alpha and is valid at every level the family reads it at; a
    row outside a family reports its delta p-value.
    """
    _require_fixed_horizon(
        row, "sequential confidence sequences do not define fixed-look sampling p-values"
    )
    if row.failure_code is not None or _sampling_availability(row) is False:
        return None
    confidence_set = getattr(row, "confidence_set", None)
    if confidence_set is not None:
        reference = confidence_set.reference
        null = row.null_abs if row.null_abs is not None else row.null_lift
        if isinstance(reference, BootstrapReference):
            return bootstrap_p_value(reference, null, relative=row.null_abs is None)
        if isinstance(reference, InfluenceReference):
            return influence_p_value(reference, null, relative=row.null_abs is None)
        return confidence_set.alpha if row_stat_sig(row) else 1.0
    if row.null_abs is not None:
        return _null_abs_p_value(row)
    if row.relative_unavailable_reason is not None:
        return _relative_unavailable_p_value(row)
    if row.relative_confidence_set is not None:
        return row.relative_confidence_set.p_value(row.null_lift)
    if (quantile_p_value := getattr(row, "quantile_p_value", None)) is not None:
        return quantile_p_value
    if row.reference_kind == "binomial":
        if row.null_lift <= -1.0:
            _raise(
                "estimation.results.lift.p_value_null_lift_not_representable_dof",
                group_id=row.group_id,
                metric=row.metric,
                null_lift=row.null_lift,
            )
        bset = row.binomial_set
        assert bset is not None
        return bset.null_p_value(row.null_lift, row.alternative)
    e = row.lift
    if e is None:
        return None
    # A breakout or daily cell carries no ``scale``: its Wald statistics are on
    # the log scale unless the row itself is additive.
    scale = getattr(row, "scale", None) or ("linear" if row.value_scale == "absolute" else "log")
    clustered = row.reference_kind == "t" or getattr(row, "n_clusters", None) is not None
    if e.log_mean is None or e.log_se is None:
        _raise(
            "estimation.results.lift.liftestimate_dof_set"
            if clustered
            else "estimation.results.lift.p_value_missing_log_mean_or_se",
            group_id=row.group_id,
            metric=row.metric,
        )
    if clustered and e.log_se <= 0:
        _raise(
            "estimation.results.lift.liftestimate_log_se",
            e_log_se=e.log_se,
            group_id=row.group_id,
            metric=row.metric,
        )
    if scale == "log" and row.null_lift <= -1.0:
        _raise(
            "estimation.results.lift.p_value_null_lift_not_representable_dof",
            group_id=row.group_id,
            metric=row.metric,
            null_lift=row.null_lift,
        )
    null = math.log1p(row.null_lift) if scale == "log" else row.null_lift
    z = (e.log_mean - null) / e.log_se
    reference_df = row.reference_df
    from scipy.stats import norm as _norm
    from scipy.stats import t as _t

    distribution = _t if reference_df is not None else _norm
    shapes = (reference_df,) if reference_df is not None else ()
    if row.alternative == "greater":
        p_delta = float(distribution.sf(z, *shapes))
    elif row.alternative == "less":
        p_delta = float(distribution.cdf(z, *shapes))
    else:
        p_delta = float(2.0 * distribution.sf(abs(z), *shapes))
    evidence = row.delta_evidence
    if evidence is None or not evidence.composite:
        return p_delta
    exact = evidence.escalated_p_value
    if p_delta < evidence.dense_level and row.null_lift != evidence.reading_null_lift:
        assert evidence.reading_alpha is not None
        exact, _reason = conversion_route.composite_exact_reading(
            evidence.counts,
            dense_level=evidence.dense_level,
            alpha=evidence.reading_alpha,
            null_lift=row.null_lift,
            alternative=row.alternative,
        )
    return conversion_route.composite_p_value(p_delta, evidence.dense_level, exact)


def _sampling_availability(row: DecisionRow) -> bool | None:
    """The row's persisted sampling-availability marker; a prior-bound row
    serialized before the marker existed is refused rather than guessed."""
    available = row.sampling_available
    if available is None and getattr(row, "prior_shrunk", False):
        refuse_legacy_sampling(row)
    return available


def _additive_excludes(row: DecisionRow, null: float) -> bool:
    """Whether the additive interval lies beyond ``null`` on the tested side."""
    if row.abs_lb is None or row.abs_ub is None:
        return False
    if row.alternative == "greater":
        return row.abs_lb > null
    if row.alternative == "less":
        return row.abs_ub < null
    return not (row.abs_lb <= null <= row.abs_ub)


def _additive_zero_reference(row: DecisionRow) -> tuple[float, float, float | None] | None:
    """``(abs_diff, abs_se, df)`` when the row's decision is the additive
    difference tested against zero: the arm summaries block the log scale (an
    arm mean is not positive, one arm is constant, or an arm mean is close to
    zero relative to its standard error), the row's declared null is the
    unshifted zero on both scales (no absolute margin, no relative margin -- a
    declared relative margin has no additive counterpart, so such a row stays a
    conservative non-rejection), and the additive sidecar carries an interval
    with its own reference. The reference is the one the interval was cut from
    -- Welch or cluster ``t`` at ``abs_reference_df``, Normal otherwise -- so
    ``p_value()`` and ``stat_sig()`` read the same construction. ``None`` leaves
    the row a conservative non-rejection."""
    if (
        row.relative_unavailable_reason not in _ADDITIVE_DECISION_REASONS
        or row.null_abs is not None
        or row.null_lift != 0.0
        or row.abs_diff is None
        or row.abs_se is None
        or not row.abs_se > 0.0
        or row.abs_lb is None
        or row.abs_ub is None
        or row.abs_reference_kind is None
    ):
        return None
    return row.abs_diff, row.abs_se, row.abs_reference_df


#: Sequential bound statuses that carry a usable decision: a non-rejection under
#: one of these is a genuine non-rejection. ``unresolved``, ``abstained`` and
#: ``unavailable`` bounds decide nothing.
DECIDABLE_SEQUENTIAL_STATUSES = frozenset(
    {"empty", "full-domain", "interval", "full", "ray", "bounded", "disconnected"}
)


def _additive_sidecar_decides(row: DecisionRow) -> bool:
    """Whether the row's decision reads the additive sidecar (``abs_*``): a declared
    absolute margin or the automatic additive-zero decision. A native absolute row
    (``value_scale="absolute"``) is absolute too, but its ``lift`` and ``null_lift``
    already live on that scale."""
    return row.null_abs is not None or _additive_zero_reference(row) is not None


def row_decision_scale(row: DecisionRow) -> Literal["relative", "absolute"]:
    """The scale ``row``'s own decision is made on: ``"absolute"`` for a declared
    absolute margin, an automatic additive-zero decision, or a native absolute row
    whose ``lift``/``null_lift`` are in the metric's own units; ``"relative"``
    otherwise. The readout adapter persists it as ``decision_scale`` so table
    and dashboard consumers label the tested null without re-deriving the rule."""
    if row.value_scale == "absolute" or _additive_sidecar_decides(row):
        return "absolute"
    return "relative"


def row_decision_available(row: DecisionRow, *, verdict: bool | None = None) -> bool:
    """Whether ``row`` carries a usable decision construction, so that a
    ``False`` verdict from :func:`row_stat_sig` is a genuine non-rejection
    rather than the absence of a test. Persisted by the readout adapter as
    ``decision_available``."""
    verdict = row_stat_sig(row) if verdict is None else verdict
    if verdict is None:
        return False
    sequential = row.sequential_result
    if sequential is not None:
        return (
            verdict or getattr(sequential.bounds, "status", None) in DECIDABLE_SEQUENTIAL_STATUSES
        )
    if verdict:
        return True
    absolute = _additive_sidecar_decides(row)
    confidence_set = getattr(row, "confidence_set", None)
    if confidence_set is not None:
        interval = (
            confidence_set.additive
            if absolute or row.value_scale == "absolute"
            else confidence_set.relative
        )
        return all(endpoint.status != "undefined" for endpoint in (interval.lower, interval.upper))
    if absolute:
        lower, upper = row.abs_lb is not None, row.abs_ub is not None
        return (
            (lower and upper)
            or (lower and row.alternative == "greater")
            or (upper and row.alternative == "less")
        )
    if row.relative_confidence_set is not None:
        return row.relative_confidence_set.geometry not in ("empty", "unavailable")
    if row.binomial_set is not None:
        return True
    lift = row.lift
    if lift is None:
        return False
    lower, upper = lift.lb is not None, lift.ub is not None
    return (
        (lower and upper)
        or (lower and lift.open_side == "upper")
        or (upper and lift.open_side == "lower")
    )


def conversion_route_label(row: DecisionRow) -> str | None:
    """Which route an unadjusted conversion or retention row took, read from what it persists:
    ``"finite_sample"`` for an exact binomial set, ``"delta"`` for delta-route count evidence,
    ``None`` for every other row (a mean metric, an adjusted or clustered conversion, or a
    delta-route row persisted before its evidence existed)."""
    if row.binomial_set is not None:
        return "finite_sample"
    if row.delta_evidence is not None:
        return "delta"
    return None


def carry_nominal_evidence(nominal: LiftEstimate, reissued: LiftEstimate) -> LiftEstimate:
    """*reissued*, a selected row re-estimated at its family's corrected level, carrying the
    composite evidence of *nominal*, the row the family selected on, when both took the delta
    route: the exact reading is persisted at the nominal tail only, never the finer one the
    re-estimation would read, so the selected row's p-value is the one that selected it. A
    re-estimation the counts route finite-sample at the corrected level keeps its exact set
    and p-value, as every finite-sample reissue does."""
    if nominal.delta_evidence is None or reissued.delta_evidence is None:
        return reissued
    if nominal.delta_evidence.counts != reissued.delta_evidence.counts:
        return reissued
    return reissued.model_copy(
        update={"delta_evidence": nominal.delta_evidence, "note": nominal.note}
    )


def row_decision_alpha(row: DecisionRow) -> float | None:
    """The level ``row``'s own fixed-horizon construction decides at: the single
    tail a directional test spends, so ``p_value() <= row_decision_alpha(row)``
    (``<`` for an exact binomial row) agrees with ``stat_sig()``. An additive
    decision reads ``abs_alpha``; an exact binomial row its set's
    ``decision_alpha``; a joint set its inversion alpha; a scalar interval its
    ``lift.alpha``, halved for a closed directional interval (the display alpha
    is central-equivalent) and taken whole for an open directional bound, which
    a family reissue already cut at the full tail. ``None`` without a fixed
    construction."""
    if row.inference != "fixed":
        return None
    confidence_set = getattr(row, "confidence_set", None)
    if confidence_set is not None:
        return confidence_set.alpha
    if _additive_sidecar_decides(row):
        abs_alpha = row.abs_alpha
        if abs_alpha is None:
            return None
        return abs_alpha if row.alternative == "two-sided" else abs_alpha / 2.0
    if row.relative_unavailable_reason is not None:
        return None
    if row.relative_confidence_set is not None:
        return row.relative_confidence_set.alpha
    if row.binomial_set is not None:
        return row.binomial_set.decision_alpha
    lift = row.lift
    if lift is None or lift.alpha is None:
        return None
    if row.alternative == "two-sided" or lift.open_side is not None:
        return lift.alpha
    return lift.alpha / 2.0


def _null_abs_p_value(row: DecisionRow) -> float:
    """Tail against ``null_abs`` under the additive interval's reference: the score test of
    the binomial counts for a score interval, else the Normal or t tail of ``abs_diff``."""
    assert row.null_abs is not None
    if row.abs_reference_kind == "score":
        bset = row.binomial_set
        assert bset is not None, "validated: a score reference rides on a binomial set"
        return score_p_value(bset.x_c, bset.n_c, bset.x_t, bset.n_t, row.null_abs, row.alternative)
    if row.abs_se is None or row.abs_diff is None or not row.abs_se > 0.0:
        _raise(
            "estimation.results.lift.p_value_null_abs_missing_abs_se",
            group_id=row.group_id,
            metric=row.metric,
        )
    abs_kind = row.abs_reference_kind
    reference_df = row.abs_reference_df
    if getattr(row, "n_clusters", None) is not None or (
        row.reference_kind == "t" and abs_kind is None
    ):
        _raise(
            "estimation.results.lift.p_value_cluster_robust_null_abs",
            reference_df=row.abs_reference_df
            if row.abs_reference_kind == "t"
            else row.reference_df,
            group_id=row.group_id,
            metric=row.metric,
        )
    if abs_kind == "t" and reference_df is None:
        _raise(
            "estimation.results.lift.p_value_cluster_robust_null_abs",
            reference_df=row.reference_df,
            group_id=row.group_id,
            metric=row.metric,
        )
    return _additive_tail_p_value(
        row.abs_diff, row.abs_se, row.null_abs, row.alternative, reference_df
    )


def _relative_unavailable_p_value(row: DecisionRow) -> float:
    """``p_value()`` for a row with no relative interval. When the arm
    summaries block the log scale, the relative route is unavailable, not
    indeterminate: with an additive reference the row is decided on that scale
    against zero, and without one the maximal, never-rejecting p-value is the
    correct answer. Unavailable joint covariance and zero relative variance
    remain indeterminate."""
    reason = row.relative_unavailable_reason
    if reason not in _ADDITIVE_DECISION_REASONS:
        _raise("estimation.results.joint.unavailable", reason=reason)
    reference = _additive_zero_reference(row)
    if reference is None:
        return 1.0
    abs_diff, abs_se, df = reference
    return _additive_tail_p_value(abs_diff, abs_se, 0.0, row.alternative, df)


def _additive_tail_p_value(
    abs_diff: float, abs_se: float, null: float, alternative: str, df: float | None
) -> float:
    """The tail of ``abs_diff`` against ``null`` under the additive reference the
    interval was cut from: Student-t at ``df`` when given, Normal otherwise.
    Directional alternatives use one tail over the composite null."""
    z = (abs_diff - null) / abs_se
    from scipy.stats import norm as _norm
    from scipy.stats import t as _t

    distribution = _t if df is not None else _norm
    shapes = (df,) if df is not None else ()
    if alternative == "greater":
        return float(distribution.sf(z, *shapes))
    if alternative == "less":
        return float(distribution.cdf(z, *shapes))
    return float(2.0 * distribution.sf(abs(z), *shapes))


def _fcr_alpha_for(alternative: str, fcr_alpha: float | Fraction) -> float:
    """Return the requested alpha whose effective value is ``fcr_alpha``.

    Directional alternatives double internally; two-sided alternatives do not.
    """
    exact = Fraction(fcr_alpha) / (2 if alternative != "two-sided" else 1)
    value = float(exact)
    return math.nextafter(value, 0.0) if Fraction(value) > exact else value


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
    return e.log_mean, e.log_se


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
    from scipy.stats import norm as _norm

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
    return (
        " | ".join(
            filter(None, (binomial_rr.without_precision_note(note), binomial_rr.precision_note(ci)))
        )
        or None
    )


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
        bset = estimate.binomial_set
        assert bset is not None, "validated: reference_kind='binomial' rows carry a set"

        ci = binomial_rr.confidence_interval(
            bset.x_c,
            bset.n_c,
            bset.x_t,
            bset.n_t,
            alpha=bset.alpha,
            alternative=binomial_rr.validate_alternative(estimate.alternative),
            null_r=1.0 + estimate.null_lift,
        )
        ci_lower, ci_upper = binomial_rr.to_lift_bounds(ci)
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
                "nuisance_beta": binomial_rr.nuisance_beta(bset.alpha),
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
        if estimate.sampling_available is not True and (
            estimate.prior_shrunk or estimate.prior_spec is not None
        ):
            _raise("estimation.results.lift.fcr_prior_unsupported")
        if e.alpha is None:
            _raise("estimation.results.lift.open_interval_unrecoverable")
        from scipy.stats import norm as _norm

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
    from scipy.stats import norm as _norm
    from scipy.stats import t as _t

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
    from scipy.stats import norm as _norm

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


def estimate_winsor_lift(
    raw: WinsorRawState,
    control: str,
    treatment: str,
    *,
    alpha: float = 0.05,
    method: str = "unadjusted",
    method_role: Literal["decision", "sensitivity"] = "decision",
    null_lift: float = 0.0,
    null_abs: float | None = None,
    preferred_direction: PreferredDirection | None = None,
    reference: BootstrapReference | InfluenceReference | None = None,
) -> LiftEstimate:
    """Build the public row without manufacturing a posterior or a standard error."""
    parts = winsor_lift_parts(raw, control, treatment, alpha=alpha, reference=reference)
    region = parts.region
    point = None
    if region.point is not None:
        if region.lower is not None and region.upper is not None and alpha / 2 > 0:
            point = Estimate(
                value=region.point,
                lb=region.lower,
                ub=region.upper,
                alpha=alpha,
                level=region.level,
            )
        else:
            point = Estimate(value=region.point)
    row = {
        "metric": raw.metric,
        "group_id": treatment,
        "method": method,
        "method_role": method_role,
        "reference_kind": "confidence_set",
        "confidence_set": region,
        "lift": point,
        "abs_diff": region.additive_point,
        "abs_lb": region.additive.lower.value,
        "abs_ub": region.additive.upper.value,
        "abs_alpha": (
            region.alpha
            if region.additive.lower.value is not None and region.additive.upper.value is not None
            else None
        ),
        "null_lift": null_lift,
        "null_abs": null_abs,
        "preferred_direction": preferred_direction,
        "analysis_population": raw.population,
        "winsor_upper_percentile": raw.quantile,
        "winsor_upper_bound": parts.cutoff,
        "winsor_control_n_lower": 0,
        "winsor_treatment_n_lower": 0,
        "winsor_control_n_upper": parts.control_n_upper,
        "winsor_treatment_n_upper": parts.treatment_n_upper,
        "winsor_control_n": parts.control_n,
        "winsor_treatment_n": parts.treatment_n,
    }
    return LiftEstimate.model_validate(row, context=parts.validation_context)


def _outward(value: Fraction | None, *, lower: bool) -> float | None:
    if value is None:
        return None
    exact = value - 1
    try:
        result = float(exact)
    except OverflowError:
        return None
    if not math.isfinite(result):
        return None
    if (lower and Fraction(result) > exact) or (not lower and Fraction(result) < exact):
        result = math.nextafter(result, -math.inf if lower else math.inf)
    return result if math.isfinite(result) else None


def display_estimate(result: SequentialResult):
    point, _ = _point(result.checkpoint)
    bounds = result.bounds
    lift = None
    if point is not None:
        # The full exact geometry remains in sequential_result even when a
        # finite display cannot represent an endpoint or the alpha allocation.
        lower, upper = _outward(bounds.lower, lower=True), _outward(bounds.upper, lower=False)
        alpha = float(bounds.alpha)
        open_side = (
            "upper"
            if upper is None and lower is not None
            else "lower"
            if lower is None and upper is not None
            else None
        )
        if (
            not (
                isinstance(result, AsymptoticSequentialResult)
                and len(result.bounds.components) != 1
            )
            and not bounds.empty
            and (lower is not None or upper is not None)
            and 0 < alpha < 1
            and 0 < 1 - alpha < 1
            and alpha / 2 > 0
        ):
            lift = Estimate(
                value=point, lb=lower, ub=upper, open_side=open_side, alpha=alpha, level=1 - alpha
            )
        else:
            lift = Estimate(value=point)
    return lift


def require_selected_widening(
    result: SequentialResult,
    *,
    discovery: bool | None,
    family_threshold: float | None,
    family_nominal_alpha: float | None,
) -> None:
    """A row's decision alpha may pass its registered cell allocation only as a
    selected family member, and never past ``min(family_threshold, family_nominal_alpha)``."""
    if result.decision_alpha == result.checkpoint.cell.alpha:
        return
    if not discovery:
        sequential_refuse(
            "source.invalid",
            "only a selected row may carry a decision alpha past its registered cell allocation",
        )
    if family_threshold is None or family_nominal_alpha is None:
        sequential_refuse(
            "source.invalid",
            "a widened decision alpha requires a recorded family threshold and nominal alpha",
        )
    permitted = min(family_threshold, family_nominal_alpha)
    if float(result.decision_alpha) > permitted * (1 + 1e-9):
        sequential_refuse(
            "source.invalid",
            "reinverted decision alpha exceeds the family's realized threshold or nominal alpha",
        )


def _composite_evidence(
    row: LiftEstimate,
    evidence: DeltaConversionEvidence,
    alpha: float,
    alternative: Alternative,
) -> tuple[DeltaConversionEvidence, str | None]:
    """The composite evidence of a family-member conversion row routed to the delta method, and
    the note it carries when its p-value is not its delta p-value.

    ``U`` is the smallest level at which the counts are dense (`conversion_route.dense_level`).
    A delta p-value at or above ``U`` is the row's p-value. Below ``U`` the exact p-value is read
    at the row's nominal tail with its nominal nuisance budget ``beta`` -- only where it can
    matter: when ``U <= beta`` no exact p-value falls below ``U``, and the row reports ``U``
    without the work. The exact route's own refusals (an arm above `FINITE_SAMPLE_MAX_ARM_SIZE`,
    a tail its float margin dominates) are recorded as the reason the row reports ``U``: a
    p-value of ``U`` is valid at every level (the row rejects nowhere below ``U``, and at ``U``
    and above only where its delta p-value does), and is never a silent fallback.
    """
    level = evidence.dense_level
    null_lift = row.null_lift
    p_delta = row_p_value(row)
    assert p_delta is not None, "a delta-route row carries its sampling reference"
    evidence = evidence.model_copy(
        update={"composite": True, "reading_null_lift": null_lift, "reading_alpha": alpha}
    )
    if p_delta >= level:
        return evidence, None
    below = (
        f"the delta p-value {p_delta:.3g} lies below {level:.3g}, the smallest level at "
        "which these counts are dense"
    )
    beta = binomial_rr.nuisance_beta(alpha)
    if level <= beta:
        return evidence, (
            f"{CAPPED_NOTE_PREFIX}: {below}, and no exact p-value falls below the nuisance "
            f"budget {beta:.3g}, so the row reports {level:.3g}"
        )
    exact, code = conversion_route.composite_exact_reading(
        evidence.counts,
        dense_level=level,
        alpha=alpha,
        null_lift=null_lift,
        alternative=alternative,
    )
    if exact is not None:
        evidence = evidence.model_copy(update={"escalated_p_value": exact})
        if exact < level:
            return evidence, (
                f"{ESCALATED_NOTE_PREFIX}: {below}, so the row reports the exact p-value "
                f"{exact:.3g} read at its nominal tail"
            )
        return evidence, (
            f"{CAPPED_NOTE_PREFIX}: {below}, and the exact p-value {exact:.3g} does not, "
            f"so the row reports {level:.3g}"
        )
    assert code is not None, "a readable dense level either has exact evidence or a refusal"
    return evidence.model_copy(update={"exact_unavailable": code}), (
        f"{CAPPED_NOTE_PREFIX}: {below}, and the exact route is unavailable ({code}), so the "
        f"row reports {level:.3g}"
    )


# FCR reissue rebuilds each persisted construction as its estimator does, then opens it
# with ``open_bound_from_two_sided_at_target`` exactly as the moment path does.


def _wald_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """``infer_lift``'s interval from persisted constructor moments at *alpha*."""
    assert view.lift is not None
    parameters = _fixed_fcr_parameters(view)
    assert parameters is not None, "classified: raw statistics are persisted"
    mu, sigma = parameters
    reference = _resolve_fixed_horizon(
        alpha,
        view.alternative,
        dof=view.reference_df if view.reference_kind == "t" else None,
        arm_ns=None,
        prior=None,
    )
    assert reference.crit is not None
    updates: dict[str, object] = {}
    try:
        lower, upper = wald_bounds(mu, reference.crit * sigma, 1.0, what="FCR relative interval")
        if view.scale == "log":
            lower = resolvable_expm1(lower, what="FCR relative lower bound")
            upper = resolvable_expm1(upper, what="FCR relative upper bound")
    except CodedError as exc:
        updates.update(
            {
                "lift": view.lift.model_copy(
                    update={
                        "lb": None,
                        "ub": None,
                        "alpha": None,
                        "level": None,
                        "open_side": None,
                    }
                ),
                "relative_error_code": exc.code,
                "relative_error_context": exc.context,
            }
        )
    else:
        updates["lift"] = view.lift.model_copy(
            update={
                "lb": lower,
                "ub": upper,
                "level": math.fsum((1.0, -reference.alpha_eff)),
                "alpha": reference.alpha_eff,
            }
        )
    if view.abs_lb is not None or view.abs_ub is not None:
        assert view.abs_diff is not None and view.abs_se is not None
        abs_reference = _resolve_fixed_horizon(
            alpha,
            view.alternative,
            dof=view.abs_reference_df if view.abs_reference_kind == "t" else None,
            arm_ns=None,
            prior=None,
        )
        assert abs_reference.crit is not None
        updates["abs_lb"], updates["abs_ub"] = wald_bounds(
            view.abs_diff, abs_reference.crit, view.abs_se, what="FCR additive interval"
        )
        updates["abs_alpha"] = abs_reference.alpha_eff
    return view.model_copy(update=updates)


def _score_additive(x_c: int, n_c: int, x_t: int, n_t: int, alpha_eff: float) -> dict[str, object]:
    """The additive fields of a binomial row cut at its central-equivalent *alpha_eff*."""
    lower, upper, kind = binomial_abs_interval(x_c, n_c, x_t, n_t, alpha_eff)
    return {"abs_lb": lower, "abs_ub": upper, "abs_reference_kind": kind, "abs_alpha": alpha_eff}


def with_score_margin(view: LiftEstimate) -> LiftEstimate:
    """*view* as a family member: a binomial margin row saved with Wald additive bounds and no
    reference is read with the score interval its own level cuts, so a family selects it on the
    score test its reissued interval inverts. Any other row is returned unchanged."""
    bset = view.binomial_set
    if (
        view.null_abs is None
        or view.abs_reference_kind is not None
        or view.reference_kind != "binomial"
        or bset is None
    ):
        return view
    return view.model_copy(
        update=_score_additive(bset.x_c, bset.n_c, bset.x_t, bset.n_t, bset.alpha)
    )


def _binomial_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """The exact binomial row ``estimate_lift`` builds at *alpha* from the persisted counts, its
    score additive interval included (a row persisted with Wald additive bounds gets one too)."""
    bset = view.binomial_set
    assert bset is not None
    alpha_eff = alpha if view.alternative == "two-sided" else 2.0 * alpha
    interval = binomial_rr.confidence_interval(
        bset.x_c,
        bset.n_c,
        bset.x_t,
        bset.n_t,
        alpha=alpha,
        alternative=binomial_rr.validate_alternative(view.alternative),
        null_r=1.0 + view.null_lift,
    )
    lower, upper = binomial_rr.to_lift_bounds(interval)
    level = math.fsum((1.0, -alpha_eff))
    updates: dict[str, object] = {
        "binomial_set": BinomialConfidenceSet(
            lower=lower,
            upper=upper,
            alpha=alpha_eff,
            level=level,
            geometry=interval.geometry,
            method=bset.method,
            numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
            x_c=bset.x_c,
            n_c=bset.n_c,
            x_t=bset.x_t,
            n_t=bset.n_t,
            nuisance_beta=binomial_rr.nuisance_beta(alpha),
            decision_alpha=alpha,
        ),
        "note": _reinverted_binomial_note(view.note, interval),
    }
    if view.lift is not None:
        updates["lift"] = view.lift.model_copy(
            update={
                "lb": lower,
                "ub": upper,
                "open_side": "upper" if upper is None else None,
                "level": level,
                "alpha": alpha_eff,
            }
        )
    updates.update(_score_additive(bset.x_c, bset.n_c, bset.x_t, bset.n_t, alpha_eff))
    return view.model_copy(update=updates)


def _conversion_parent(
    view: LiftEstimate, alpha: float, fcr_alpha: float | Fraction
) -> LiftEstimate:
    """The row ``estimate_lift`` builds at *alpha* for a conversion contrast the count rule
    routed to the delta method: routed again by its persisted counts at the tail of the bound
    the row finally reports at the corrected level *fcr_alpha* (`fcr_route_tail`), the Wald
    row where they are dense there and the exact binomial row from the counts where they are
    not (the corrected level fell below the row's dense level), as a fresh finite-sample row
    would be cut: the score additive interval with no t reference, and only the disclosures
    that still apply."""
    evidence = view.delta_evidence
    assert evidence is not None, "classified: the row carries its counts"
    tail = conversion_route.fcr_route_tail(view.alternative, float(fcr_alpha))
    if (
        conversion_route.route_for_counts(*evidence.counts, tail_alpha=tail, mode="auto")
        == "asymptotic"
    ):
        return _wald_parent(view, alpha)
    x_c, n_c, x_t, n_t = evidence.counts
    alternative = binomial_rr.validate_alternative(view.alternative)
    interval = binomial_rr.confidence_interval(
        x_c, n_c, x_t, n_t, alpha=alpha, alternative=alternative, null_r=1.0 + view.null_lift
    )
    point = binomial_rr.point_lift(x_c, n_c, x_t, n_t)
    assert point is not None, "a delta-route row has a positive control count"
    lower, upper = binomial_rr.to_lift_bounds(interval)
    alpha_eff = _alpha_eff_for(view.alternative, alpha)
    level = math.fsum((1.0, -alpha_eff))
    updates: dict[str, object] = {
        "lift": Estimate(
            value=point,
            lb=lower,
            ub=upper,
            open_side="upper" if upper is None else None,
            level=level,
            alpha=alpha_eff,
        ),
        "binomial_set": BinomialConfidenceSet(
            lower=lower,
            upper=upper,
            alpha=alpha_eff,
            level=level,
            geometry=interval.geometry,
            method=BINOMIAL_METHOD,
            numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
            x_c=x_c,
            n_c=n_c,
            x_t=x_t,
            n_t=n_t,
            nuisance_beta=binomial_rr.nuisance_beta(alpha),
            decision_alpha=alpha,
        ),
        "delta_evidence": None,
        "reference_kind": "binomial",
        "reference_df": None,
        "dof": None,
        "scale": "linear",
        "abs_reference_df": None,
        "note": " | ".join(
            part
            for part in (
                conversion_route.without_composite_note(view.note),
                binomial_rr.precision_note(interval),
            )
            if part is not None
        )
        or None,
    }
    updates.update(_score_additive(x_c, n_c, x_t, n_t, alpha_eff))
    return view.model_copy(update=updates)


def _joint_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """The Fieller row ``infer_ate`` builds at *alpha* from the persisted joint reference."""
    relative = view.relative_confidence_set
    assert relative is not None
    reissued = relative_confidence_set(
        relative.reference, alpha=alpha, alternative=view.alternative
    )
    abs_lb, abs_ub = _joint_additive_bounds(
        view.abs_diff,
        view.abs_se,
        reissued.alpha,
        view.alternative,
        view.abs_reference_df if view.abs_reference_kind == "t" else None,
    )
    return view.model_copy(
        update={
            "relative_confidence_set": reissued,
            "lift": reissued.estimate(),
            "abs_lb": abs_lb,
            "abs_ub": abs_ub,
            "abs_alpha": reissued.alpha_eff if abs_lb is not None else None,
        }
    )


def _unavailable_parent(view: LiftEstimate, alpha: float) -> LiftEstimate:
    """The central additive interval an estimator cuts for such a row at *alpha*."""
    if not _has_additive_interval(view):
        return view
    assert view.abs_diff is not None and view.abs_se is not None
    abs_lb, abs_ub = _joint_additive_bounds(
        view.abs_diff, view.abs_se, alpha, view.alternative, _additive_reference_df(view)
    )
    return view.model_copy(
        update={
            "abs_lb": abs_lb,
            "abs_ub": abs_ub,
            "abs_alpha": _alpha_eff_for(view.alternative, alpha) if abs_lb is not None else None,
        }
    )


def reinterval_at_fcr(
    view: LiftEstimate, construction: Construction, fcr_alpha: float | Fraction
) -> LiftEstimate:
    """*view* reissued at the total noncoverage budget *fcr_alpha*.

    Two-sided rows get a central interval at *fcr_alpha*; a directional row gets the open bound
    ``open_bound_from_two_sided_at_target`` derives from the parent built at the halved alpha. A
    row without a relative interval has its additive interval reissued centrally.
    """
    alpha = _fcr_alpha_for(view.alternative, fcr_alpha)
    match construction:
        case "wald":
            parent = _wald_parent(view, alpha)
        case "conversion":
            parent = _conversion_parent(view, alpha, fcr_alpha)
        case "binomial":
            parent = _binomial_parent(view, alpha)
        case "joint":
            parent = _joint_parent(view, alpha)
        case "unavailable":
            parent = _unavailable_parent(view, alpha)
        case _:
            assert_never(construction)
    return open_bound_from_two_sided_at_target(parent)


def reinterval_selected(
    view: LiftEstimate, construction: Construction, realized_threshold: float
) -> LiftEstimate:
    """*view* reissued at ``min(realized_threshold, nominal)``, the level a selected row reports.

    A row with no relative and no additive interval has nothing to reissue.
    """
    if construction == "unavailable" and not _has_additive_interval(view):
        return view
    return reinterval_at_fcr(
        view, construction, min(realized_threshold, nominal_alpha(view, construction))
    )
