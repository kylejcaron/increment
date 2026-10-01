"""Segment heterogeneity and pairwise contrast over a ``run_breakout`` result.

Thin model-based adapter over ``increment.estimation.meta``'s pure array
math; lives here because it consumes ``BreakoutEstimate``, which
``increment.estimation`` cannot import without inverting the dependency.

``segment_contrast`` answers "is segment A's lift actually different from
segment B's?" - a calibrated Wald contrast, not the naive eyeball
comparison of two intervals in a breakout table.

``segment_heterogeneity`` answers "of the segments I declared, do any
differ?": Cochran's Q, DerSimonian-Laird tau^2, Higgins-Thompson I^2, an
HKSJ pooled effect, and tau-marginalised shrinkage, on both the relative
(log-RR) and absolute scales - heterogeneity is scale-dependent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, model_validator

from increment._literals import ValueScale
from increment.breakout.estimates import (
    DESIGN_BASED_REASONS,
    BreakoutEstimate,
    BreakoutEstimates,
    EstimateList,
    ExclusionReason,
    _relative_meta_moments,
)
from increment.errors import (
    CodedModel,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation._tails import student_t_isf
from increment.estimation.binomial_rr import confidence_interval, to_lift_bounds
from increment.estimation.meta import (
    _TAU_PRIOR_SCALE_DEFAULT,
    ESTIMATION_META_POSTERIOR_INTEGRATION_UNRESOLVED,
    MarginalizedSegmentIntervals,
    cochran_q,
    hksj_pooled_mean,
    marginalized_segment_intervals,
    pairwise_contrast_arrays,
)
from increment.estimation.results import Estimate, _recover_fixed_open_interval_parameters

Renderer = Callable[..., str]


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "breakout.heterogeneity.posterior_integration_unresolved",
    IncrementWarning,
    lambda *, metric, group_id, dimension, source, estimand, value_scale, scale, k, tau_prior_scale, support, integration_error, node_budget: (
        f"segment_heterogeneity: metric={metric!r} group_id={group_id!r} "
        f"dimension={dimension!r} source={source!r} estimand={estimand!r} "
        f"value_scale={value_scale!r} scale={scale!r}: the shrinkage "
        f"posterior over tau (k={k}, tau_prior_scale={tau_prior_scale:.6g}) "
        "could not be integrated to the required accuracy within "
        f"{node_budget} log-density evaluations (last support tau in "
        f"[{support[0]:.6g}, {support[1]:.6g}], estimated error "
        f"{integration_error:.3g} times the tolerance). Shrunken rows "
        "withheld with lift=None and excluded='estimation_failed'; the raw "
        "rows are unaffected."
    ),
)
_register_warning(
    "breakout.heterogeneity.hksj_degenerate_variance_fallback",
    IncrementWarning,
    lambda *, metric, group_id, dimension, source, estimand, value_scale, scale: (
        f"segment_heterogeneity: metric={metric!r} group_id={group_id!r} "
        f"dimension={dimension!r} source={source!r} estimand={estimand!r} "
        f"value_scale={value_scale!r} scale={scale!r} has a "
        "degenerate HKSJ pooled variance (segment estimates are identical "
        "or numerically indistinguishable given their weights) -- falling "
        "back to the plug-in pooled interval for this group."
    ),
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "breakout.segment_contrast_absolute_interval": "segment_contrast requires an interval on each segment's lift (lb/ub/level all set) to recover its variance",
        "breakout.segment_contrast_open_interval_variance": "segment_contrast could not recover a positive, finite standard error from the open interval's calibrated endpoint",
        "breakout.segment_contrast_expected": "segment_contrast: expected exactly 1 row for dimension_value={label!r}, got {row_count}. Scope `estimates` to a single (metric, method, group_id, dimension, source, estimand, value_scale) combination before calling segment_contrast.",
        "breakout.segment_contrast_dimension": "segment_contrast: dimension_value={dimension_value_a!r} and {dimension_value_b!r} do not share the same (metric, method, group_id, dimension, source, estimand, value_scale) grouping ({key_a!r} vs {key_b!r}). Scope `estimates` to a single grouping before calling segment_contrast.",
        "breakout.segment_contrast_reference_fixed_horizon": RefusalSpec(
            "breakout.segment_contrast_reference_fixed_horizon",
            InvalidRequestError,
            lambda *, dimension_value, dof, inference, reference_kind, reference_df: (
                f"segment_contrast: dimension_value={dimension_value!r} has "
                f"dof={dof!r}, inference={inference!r}, "
                f"reference_kind={reference_kind!r}, reference_df={reference_df!r}; "
                + (
                    "an exact-binomial confidence set does not encode the Normal "
                    "variance this contrast requires. Re-run run_breakout with a "
                    "fixed-horizon Normal-reference method (for example, CUPED with "
                    "valid covariate moments) before calling segment_contrast."
                    if reference_kind == "binomial"
                    else "a sequential width is a boundary crossing, not a critical value "
                    "times a standard error, so no variance can be recovered from it. "
                    "Re-run run_breakout without `inference` before calling "
                    "segment_contrast."
                    if inference != "fixed"
                    else "a t sampling reference (cluster-robust or Welch) cuts its "
                    "endpoints at t quantiles, so recovering the variance from lb/ub "
                    "would inflate it by the t/z ratio -- this contrast reads "
                    "lift.log_mean/lift.log_se instead, and this row carries neither a "
                    "log_mean nor a positive log_se. Re-run run_breakout so they are "
                    "persisted, or exclude this segment."
                )
            ),
        ),
        "breakout.segment_contrast_estimate": "segment_contrast requires an estimate on both segments; unavailable rows cannot be contrasted",
        "breakout.segment.lift_none_excluded": "lift=None requires excluded to explain the unavailable row",
        "breakout.segment.point_unavailable": "a finite point estimate is unavailable for this excluded segment row",
        "breakout.segment_heterogeneity_rebuild": "segment_heterogeneity: cannot rebuild {metric!r}/{dimension_value!r}'s relative-scale interval at alpha={alpha} -- the row is alternative={alternative!r}, dof={dof!r}, inference={inference!r}, reference_kind={reference_kind!r}, reference_df={reference_df!r}, and a symmetric two-sided Normal interval would silently misrepresent its shape or reference, including a t sampling reference from clustering or Welch. Pass an alpha matching this row's own level, or exclude it before correcting.",
        "breakout.tau_prior_scale": "tau_prior_scale must be > 0, got {tau_prior_scale}",
    },
)
_raise = raiser(_REFUSALS)


def _log_scale_moments(estimate: Estimate, *, reference_kind: str) -> tuple[float, float]:
    """Recover (log1p(value), se) on the log scale from a relative-scale Estimate.

    Exact (up to floating point) under a Normal reference: ``infer_lift``
    computes ``lb``/``ub`` as closed-form quantiles of the log-scale Normal
    posterior, so ``log1p(lb)``/``log1p(ub)`` recover its mean/se exactly. An
    open interval (``open_side`` set, one bound ``None``) recovers the same
    se from its single calibrated bound at the FULL allocated tail
    ``norm.isf(alpha)`` -- a fixed-horizon open row's stored ``alpha`` is
    already the one-sided allocation, not a two-sided total (see
    ``Estimate.open_side``).

    Under a ``"t"`` reference those endpoints are t quantiles, so the same
    back-solve would divide by too small a critical value and inflate the
    recovered se by the t/z ratio. Such a row is read from the working-scale
    moments its interval was cut from instead; :func:`segment_contrast`
    admits a t row only once it carries them.

    Assumes a fixed-horizon interval - a sequential interval's width
    would silently inflate the recovered se, so :func:`segment_contrast`
    refuses any non-``"fixed"`` row before calling this.
    """
    log_value = math.log1p(estimate.value)
    persisted = _persisted_open_parameters(estimate, near_flat_update=True)
    if estimate.open_side is not None:
        calibrated = estimate.ub if estimate.open_side == "lower" else estimate.lb
        if calibrated is None or estimate.level is None:
            _raise("breakout.segment_contrast_absolute_interval")
        log_calibrated = math.log1p(calibrated)
        _, se = _recover_fixed_open_interval_parameters(
            estimate,
            center=log_value,
            calibrated=log_calibrated,
            persisted=persisted,
        )
        if not math.isfinite(se) or se <= 0:
            _raise("breakout.segment_contrast_open_interval_variance")
        return log_value, se
    if reference_kind == "t":
        assert persisted is not None, "admitted: a t row carries its working-scale moments"
        return persisted
    if estimate.lb is None or estimate.ub is None or estimate.level is None:
        _raise("breakout.segment_contrast_absolute_interval")
    log_lb = math.log1p(estimate.lb)
    log_ub = math.log1p(estimate.ub)
    z = _z_for_alpha(_tail_alpha(estimate))
    se = (log_ub - log_lb) / (2.0 * z)
    return log_value, se


def _absolute_scale_moments(
    estimate: Estimate, *, near_flat_update: bool, reference_kind: str
) -> tuple[float, float]:
    """Recover (value, se) directly for an additive-scale Estimate.

    No log1p transform: an additive LATE or additively-reported metric's
    ``value``/``lb``/``ub`` already live on the linear scale, so shifting
    through ``log1p`` (defined only above -1) would raise on a legitimate
    effect at or below -1 and misstate every other one. An open interval
    recovers se from its single calibrated bound at the full allocated
    tail, and a ``"t"``-reference row from its own working-scale moments,
    both same as :func:`_log_scale_moments`.
    """
    persisted = _persisted_open_parameters(estimate, near_flat_update=near_flat_update)
    if estimate.open_side is not None:
        calibrated = estimate.ub if estimate.open_side == "lower" else estimate.lb
        if calibrated is None or estimate.level is None:
            _raise("breakout.segment_contrast_absolute_interval")
        _, se = _recover_fixed_open_interval_parameters(
            estimate,
            center=estimate.value,
            calibrated=calibrated,
            persisted=persisted,
        )
        if not math.isfinite(se) or se <= 0:
            _raise("breakout.segment_contrast_open_interval_variance")
        return estimate.value, se
    if reference_kind == "t":
        assert persisted is not None, "admitted: a t row carries its working-scale moments"
        return persisted
    if estimate.lb is None or estimate.ub is None or estimate.level is None:
        _raise("breakout.segment_contrast_absolute_interval")
    z = _z_for_alpha(_tail_alpha(estimate))
    se = (estimate.ub - estimate.lb) / (2.0 * z)
    return estimate.value, se


def _absolute_meta_moments(row: BreakoutEstimate) -> tuple[float, float] | None:
    """Use additive primary moments, reconstructing only fixed Normal intervals.

    Absolute rows store additive moments in the shared log_mean/log_se fields.
    """
    lift = row.lift
    if lift is None:
        return None
    if (
        lift.log_mean is not None
        and lift.log_se is not None
        and math.isfinite(lift.log_mean)
        and lift.log_se > 0
        and math.isfinite(lift.log_se)
    ):
        return lift.log_mean, lift.log_se
    if row.inference != "fixed" or row.reference_kind != "normal":
        return None
    if (
        lift.level is None
        or (lift.lb is None and lift.open_side != "lower")
        or (lift.ub is None and lift.open_side != "upper")
    ):
        return None
    return _absolute_scale_moments(
        lift,
        near_flat_update=True,
        reference_kind=row.reference_kind,
    )


def _persisted_open_parameters(
    estimate: Estimate, *, near_flat_update: bool
) -> tuple[float, float] | None:
    """Recover constructor moments persisted on a fixed open estimate."""
    if estimate.log_mean is None or estimate.log_se is None:
        return None
    if estimate.log_se <= 0:
        return None
    if not near_flat_update:
        return estimate.log_mean, estimate.log_se
    from increment.estimation.inference import normal_posterior

    posterior = normal_posterior(estimate.log_mean, estimate.log_se)
    return posterior.mu, posterior.sigma


def _tail_alpha(estimate: Estimate) -> float:
    if estimate.alpha is not None:
        return estimate.alpha
    assert estimate.level is not None, "callers check level is not None first"
    return math.fsum((1.0, -estimate.level))


def _z_for_alpha(alpha: float) -> float:
    """Critical value for a closed two-sided interval at *alpha*."""
    from scipy.stats import norm

    return float(norm.isf(alpha / 2.0))


def _critical_for_reference(
    alpha: float, *, reference_kind: str | None, reference_df: float | None
) -> float:
    """Two-sided critical value on the reference an interval was cut from.

    Rebuilding a t-reference interval with a Normal quantile would narrow it
    by the z/t ratio, understating exactly the small-sample uncertainty the t
    reference exists to carry. A row whose reference was never persisted
    (``None``) keeps the Normal reading its endpoints were written under.
    """
    if reference_kind == "t":
        assert reference_df is not None, "validated: a t reference carries its df"
        return float(student_t_isf(alpha / 2.0, reference_df))
    return _z_for_alpha(alpha)


def segment_contrast(
    estimates: BreakoutEstimates,
    dimension_value_a: str,
    dimension_value_b: str,
    alpha: float = 0.05,
) -> Estimate:
    """Contrast segment ``dimension_value_a``'s lift against ``dimension_value_b``'s.

    ``estimates`` must already be scoped to ONE (metric, method,
    group_id, dimension, source, estimand, value_scale) combination -
    exactly two rows, one per label - or this raises rather than
    guessing which pair to contrast.

    Relative rows (``value_scale="relative"``) return a value on the
    same back-transformed scale as
    :func:`increment.power.core.segment_pairwise_required_sample_size`'s
    contrast: ``value = exp(delta) - 1`` where
    ``delta = log1p(r_A) - log1p(r_B)``. Absolute rows
    (``value_scale="absolute"``, e.g. an encouragement LATE) return the
    plain additive difference ``value_A - value_B`` with no log
    transform - ``log1p`` is undefined at or below an additive effect of
    -1, which a valid absolute effect routinely is.

    Segments partition a breakout's rows with independent control arms
    and CUPED thetas, so the contrast variance is the plain sum
    ``var_a + var_b``.

    Running this on every one of a dimension's pairs without correction
    re-introduces the multiple-comparisons problem it exists to avoid
    for one pre-specified pair; use :func:`increment.estimation.meta.
    cochran_q`'s joint test when scanning every pair.

    Raises
    ------
    ValueError
        Not exactly one row per label, the two rows come from different
        groupings, either is missing an interval, or either carries a
        reference this rebuild cannot invert: non-fixed-horizon ``inference``
        (a sequential width is not a critical value times a standard error) or
        a t sampling reference (built with a t critical value, not a Normal
        one). Exact-binomial rows are also refused because their confidence
        sets do not encode a Normal variance; rerun the breakout with a
        fixed-horizon Normal-reference method, such as CUPED with valid
        covariate moments, before contrasting those segments. A directional
        (open) row IS accepted: its single calibrated bound recovers the
        variance at the row's own full allocated tail (``Estimate.open_side``);
        an unavailable row (both bounds ``None``) is not.
    """
    rows_a = [e for e in estimates if e.dimension_value == dimension_value_a]
    rows_b = [e for e in estimates if e.dimension_value == dimension_value_b]
    for label, rows in ((dimension_value_a, rows_a), (dimension_value_b, rows_b)):
        if len(rows) != 1:
            _raise("breakout.segment_contrast_expected", label=label, row_count=len(rows))

    row_a, row_b = rows_a[0], rows_b[0]
    key_a, key_b = _group_key(row_a), _group_key(row_b)
    if key_a != key_b:
        _raise(
            "breakout.segment_contrast_dimension",
            dimension_value_a=dimension_value_a,
            dimension_value_b=dimension_value_b,
            key_a=key_a,
            key_b=key_b,
        )

    for row in (row_a, row_b):
        # Exact-binomial and sequential widths do not provide Normal variance or
        # a critical-value-times-SE representation, so they cannot enter a Wald
        # contrast. A t reference can enter when its cut moments were persisted;
        # otherwise no working-scale variance is available.
        lift = row.lift
        if (
            row.reference_kind == "binomial"
            or row.inference != "fixed"
            or (
                row.reference_kind == "t"
                and (lift is None or lift.log_mean is None or not (lift.log_se or 0.0) > 0.0)
            )
        ):
            _raise(
                "breakout.segment_contrast_reference_fixed_horizon",
                dimension_value=row.dimension_value,
                dof=row.dof,
                inference=row.inference,
                reference_kind=row.reference_kind,
                reference_df=row.reference_df,
            )

    lift_a, lift_b = row_a.lift, row_b.lift
    if lift_a is None or lift_b is None:
        _raise("breakout.segment_contrast_estimate")
    if row_a.value_scale == "absolute":
        randomized_estimands = ("itt", "compliance", "late")
        a, se_a = _absolute_scale_moments(
            lift_a,
            near_flat_update=row_a.estimand in randomized_estimands,
            reference_kind=row_a.reference_kind,
        )
        b, se_b = _absolute_scale_moments(
            lift_b,
            near_flat_update=row_b.estimand in randomized_estimands,
            reference_kind=row_b.reference_kind,
        )
        diff, se, lb, ub = pairwise_contrast_arrays(a, se_a**2, b, se_b**2, alpha=alpha)
        return Estimate(
            value=diff,
            lb=lb,
            ub=ub,
            level=math.fsum((1.0, -alpha)),
            alpha=alpha,
        )
    log_a, se_a = _log_scale_moments(lift_a, reference_kind=row_a.reference_kind)
    log_b, se_b = _log_scale_moments(lift_b, reference_kind=row_b.reference_kind)

    diff, se, lb, ub = pairwise_contrast_arrays(log_a, se_a**2, log_b, se_b**2, alpha=alpha)
    return Estimate(
        value=math.expm1(diff),
        lb=math.expm1(lb),
        ub=math.expm1(ub),
        level=math.fsum((1.0, -alpha)),
        alpha=alpha,
    )


class HeterogeneitySummary(BaseModel):
    """One row per ``(metric, method, group_id, dimension, source,
    estimand, value_scale)`` grouping key: Cochran's Q, DerSimonian-Laird
    tau^2, Higgins-Thompson I^2, and an HKSJ pooled effect, over the
    segments declared for that key.

    ``scale`` is ``"relative"`` (log-RR) or ``"absolute"`` (risk
    difference) - the two frequently disagree, so both ship.
    ``value_scale`` is the upstream rows' own value scale; ``scale`` is
    which of the two heterogeneity passes this row belongs to.
    ``tau2``/``i2``/``i2_lb``/``i2_ub`` are ``None`` whenever
    ``n_excluded_outcome > 0`` for this (key, ``scale``) row: an
    outcome-based exclusion biases tau^2, so it is suppressed rather than
    reported as trustworthy. ``q``/``p_value``/``pooled`` are not.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    method: str
    group_id: str
    dimension: str
    source: str | None
    method_role: Literal["decision", "sensitivity"]
    estimand: str = "itt"  # "itt" | "compliance" | "late" (mirrors BreakoutEstimate)
    value_scale: ValueScale = "relative"  # scale of the rows' own lift values
    scale: ValueScale
    k: int
    q: float
    p_value: float
    tau2: float | None
    i2: float | None
    i2_lb: float | None
    i2_ub: float | None
    n_excluded_design: int
    n_excluded_outcome: int
    pooled: Estimate


class SegmentEstimate(CodedModel, BaseModel):
    """One row per (segment, scale, estimator) for a ``HeterogeneitySummary``
    key: two rows per estimable (segment, scale) (``estimator="raw"``
    and ``"shrunken"``), plus an unavailable pair per excluded segment.

    ``excluded`` is set from an upstream ``BreakoutEstimate.excluded``, a
    live segment unusable on this scale only
    (``excluded="zero_variance"``), or a ``shrunken`` row withheld because
    that scale's tau posterior could not be integrated within the
    numerical budget (``excluded="estimation_failed"``).

    ``raw`` reuses the segment's own estimate (for ``scale="relative"``,
    ``BreakoutEstimate.lift`` verbatim). ``shrunken`` is the
    tau-marginalised posterior estimate, with its own ``shrink_k``.
    ``baseline`` is the segment's control-arm absolute mean, recovered
    from ``abs_diff`` and the raw log-scale ``lift.log_mean``.

    ``value_scale`` is the upstream rows' own value scale; ``scale`` is
    which of the two heterogeneity passes this row belongs to.

    ``role``, ``discovery``, ``family_axes``, ``family_q``, and
    ``family_threshold`` mirror the source ``BreakoutEstimate``'s fields
    of the same name verbatim (see there) -- a reader of this frame alone
    can otherwise not tell a discovery from a non-discovery, or a
    multiplicity-corrected interval from an uncorrected one. Both
    ``estimator`` rows for a segment carry the same source values.
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    method: str
    method_role: Literal["decision", "sensitivity"]
    group_id: str
    dimension: str
    dimension_value: str
    source: str | None
    estimand: str = "itt"
    value_scale: ValueScale = "relative"
    scale: ValueScale
    estimator: Literal["raw", "shrunken"]
    shrink_k: float | None
    baseline: float | None
    excluded: ExclusionReason | None
    lift: Estimate | None
    role: str | None = None  # mirrors BreakoutEstimate.role; "exploratory" for a breakout row
    discovery: bool | None = None
    # Whether the source BreakoutEstimate's (metric, arm, segment) cell
    # survived BH/FCR family selection. None when that selection never ran.
    family_axes: tuple[str, ...] | None = None
    family_q: float | None = None
    family_threshold: float | None = None

    # Which family corrected the source row, at what FDR level, and that
    # family's own realized cutoff -- mirrors BreakoutEstimate's fields of
    # the same name; see there for the nominal-alpha cap.
    @model_validator(mode="after")
    def _lift_requires_exclusion(self):
        if self.lift is None and self.excluded is None:
            _raise("breakout.segment.lift_none_excluded")
        return self

    def require_lift(self) -> Estimate:
        """Return the finite point estimate or refuse for an excluded row."""
        if self.lift is None:
            _raise("breakout.segment.point_unavailable")
        return self.lift


class HeterogeneitySummaries(EstimateList[HeterogeneitySummary]):
    """``list[HeterogeneitySummary]`` with ``.to_frame()`` - see
    :class:`~increment.breakout.estimates.EstimateList`."""

    _model = HeterogeneitySummary


class SegmentEstimates(EstimateList[SegmentEstimate]):
    """``list[SegmentEstimate]`` with ``.to_frame()`` - see
    :class:`~increment.breakout.estimates.EstimateList`."""

    _model = SegmentEstimate


class SegmentHeterogeneityResult(NamedTuple):
    """:func:`segment_heterogeneity`'s return value - two independent,
    row-aligned-by-grouping-key result sets."""

    summary: HeterogeneitySummaries
    segments: SegmentEstimates


def _group_key(
    e: BreakoutEstimate,
) -> tuple[str, str, str, str, str | None, str, ValueScale]:
    return (e.metric, e.method, e.group_id, e.dimension, e.source, e.estimand, e.value_scale)


def _baseline_mean(row: BreakoutEstimate) -> float | None:
    """Control-arm absolute mean, recovered from ``log_mean =
    log(t_mean/c_mean)`` and ``abs_diff = t_mean - c_mean``. ``None``
    when unavailable or the true lift is exactly zero (divide by zero)."""
    lift = row.lift
    if lift is None:
        return None
    if row.reference_kind == "binomial" and row.binomial_set is not None:
        return row.binomial_set.x_c / row.binomial_set.n_c
    log_mean = lift.log_mean
    abs_diff = row.abs_diff
    if log_mean is None or abs_diff is None:
        return None
    denom = math.expm1(log_mean)
    if denom == 0.0:
        return None
    return abs_diff / denom


def _misrepresented_by_a_rebuild(row: BreakoutEstimate) -> bool:
    """Whether rebuilding this row's interval at another alpha would misstate
    it: a one-sided row has the wrong shape, and a sequential row a width that
    is not a critical value times a standard error at all.

    A t sampling reference is NOT in that set. It is rebuilt at its own
    persisted degrees of freedom -- per scale, since the additive sidecar
    carries a Welch reference independent of the relative interval's (see
    :func:`_critical_for_reference`) -- never at a Normal quantile, which
    would narrow the interval by the z/t ratio.
    """
    return row.alternative != "two-sided" or row.inference != "fixed"


def _family_selection_corrected(row: BreakoutEstimate) -> bool:
    """Whether *row*'s stored interval was actually re-estimated by a
    BH/FCR family-selection pass. ``run_breakout``'s ``correction="bh"``
    path re-estimates ONLY the selected (``discovery=True``) decision
    cells at the FCR level ``1 - R*q/m`` (Benjamini-Yekutieli); an
    unselected sibling in the same family carries the same
    ``family_q``/``family_axes`` bookkeeping but keeps its ORIGINAL
    nominal-alpha interval untouched, so it is not in this set.
    Rebuilding a selected row's interval at a different, uncorrected
    alpha would silently re-narrow (or widen) exactly the segment
    selected by looking at the data back to nominal coverage -- the
    failure the correction exists to prevent. Bonferroni composes fine
    on rebuild instead (it only divides alpha, with no data-dependent
    selection to undo), so it is not in this set either.
    """
    return row.discovery is True


def _refuse_rebuild(row: BreakoutEstimate, alpha: float, *, strict: bool) -> Estimate | None:
    """Withhold an informational-only rebuild, otherwise raise."""
    if not strict:
        return None
    _raise(
        "breakout.segment_heterogeneity_rebuild",
        metric=row.metric,
        dimension_value=row.dimension_value,
        alpha=alpha,
        alternative=row.alternative,
        dof=row.dof,
        inference=row.inference,
        reference_kind=row.reference_kind,
        reference_df=row.reference_df,
    )


def _stored_alpha(lift: Estimate) -> float | None:
    """Recover *lift*'s own alpha from its ``alpha`` field, or derive it
    from ``level`` when only that was stamped."""
    if lift.alpha is not None:
        return lift.alpha
    if lift.level is not None:
        return math.fsum((1.0, -lift.level))
    return None


def _row_alpha_matches(row: BreakoutEstimate, alpha: float) -> bool:
    """Whether *row* already carries an interval at *alpha*, so a rebuild
    at *alpha* would reproduce it rather than change anything. Read from
    ``row.lift`` (the relative-scale Estimate) even to answer this for the
    absolute scale: a family-selection pass re-estimates both scales'
    sufficient statistics together at the same alpha, so ``lift``'s
    stored alpha is authoritative for either one.
    """
    lift = row.lift
    if lift is None or lift.level is None:
        return False
    stored_alpha = _stored_alpha(lift)
    return stored_alpha is not None and math.isclose(stored_alpha, alpha)


def _relative_raw_lift(
    row: BreakoutEstimate, alpha: float, *, strict: bool = True
) -> Estimate | None:
    """Return the relative raw lift, rebuilding only when valid."""
    lift = row.lift
    if lift is None:
        return None
    if lift.level is None or _row_alpha_matches(row, alpha):
        return lift
    if _family_selection_corrected(row):
        return _refuse_rebuild(row, alpha, strict=strict)
    if row.reference_kind == "binomial" and row.binomial_set is not None:
        exact = row.binomial_set
        tail_alpha = alpha if row.alternative == "two-sided" else alpha / 2.0
        interval = confidence_interval(
            exact.x_c,
            exact.n_c,
            exact.x_t,
            exact.n_t,
            alpha=tail_alpha,
            alternative=row.alternative,
        )
        lower, upper = to_lift_bounds(interval)
        return Estimate(
            value=lift.value,
            lb=lower,
            ub=upper,
            open_side="upper" if upper is None else None,
            level=math.fsum((1.0, -alpha)),
            alpha=alpha,
        )
    if _misrepresented_by_a_rebuild(row):
        return _refuse_rebuild(row, alpha, strict=strict)
    log_mean, log_se = lift.log_mean, lift.log_se
    if (
        log_mean is None
        or log_se is None
        or not math.isfinite(log_mean)
        or not (log_se > 0 and math.isfinite(log_se))
    ):
        return None
    crit = _critical_for_reference(
        alpha, reference_kind=row.reference_kind, reference_df=row.reference_df
    )
    return Estimate(
        value=math.expm1(log_mean),
        lb=math.expm1(log_mean - crit * log_se),
        ub=math.expm1(log_mean + crit * log_se),
        level=math.fsum((1.0, -alpha)),
        alpha=alpha,
        log_mean=log_mean,
        log_se=log_se,
    )


def _absolute_raw_lift(
    row: BreakoutEstimate, alpha: float, mean: float, variance: float
) -> Estimate | None:
    """Preserve primary intervals; rebuild only on the applicable reference."""
    primary = row.value_scale == "absolute"
    if primary and _row_alpha_matches(row, alpha):
        return row.lift
    if (
        _misrepresented_by_a_rebuild(row)
        or (_family_selection_corrected(row) and not _row_alpha_matches(row, alpha))
        or (primary and row.reference_kind not in {"normal", "t"})
    ):
        return None
    critical = _critical_for_reference(
        alpha,
        reference_kind=row.reference_kind if primary else row.abs_reference_kind,
        reference_df=row.reference_df if primary else row.abs_reference_df,
    )
    half_width = critical * math.sqrt(variance)
    return Estimate(
        value=mean,
        lb=mean - half_width,
        ub=mean + half_width,
        level=math.fsum((1.0, -alpha)),
        alpha=alpha,
    )


def _shrinkage_intervals(
    est: list[float],
    var: list[float],
    row: BreakoutEstimate,
    *,
    alpha: float,
    tau_prior_scale: float,
    scale: str,
) -> MarginalizedSegmentIntervals | None:
    """Withhold only this scale's shrinkage when posterior integration fails."""
    try:
        return marginalized_segment_intervals(
            est, var, alpha=alpha, tau_prior_scale=tau_prior_scale
        )
    except InvalidRequestError as exc:
        if exc.code != ESTIMATION_META_POSTERIOR_INTEGRATION_UNRESOLVED.code:
            raise
        failure = exc.context
        _warn(
            "breakout.heterogeneity.posterior_integration_unresolved",
            metric=row.metric,
            group_id=row.group_id,
            dimension=row.dimension,
            source=row.source,
            estimand=row.estimand,
            value_scale=row.value_scale,
            scale=scale,
            k=failure["k"],
            tau_prior_scale=failure["tau_prior_scale"],
            support=failure["support"],
            integration_error=failure["integration_error"],
            node_budget=failure["node_budget"],
            stacklevel=3,
        )
        return None


def segment_heterogeneity(  # noqa: PLR0915
    estimates: BreakoutEstimates,
    alpha: float = 0.05,
    tau_prior_scale: float = _TAU_PRIOR_SCALE_DEFAULT,
) -> SegmentHeterogeneityResult:
    """Cochran's Q, tau^2, I^2, an HKSJ pooled effect, and tau-marginalised
    per-segment shrinkage, over every declared segment for each ``(metric,
    method, group_id, dimension, source, estimand, value_scale)`` grouping
    key in *estimates*.

    *estimates* must come from a SINGLE ``run_breakout`` call - a
    standalone call stamps ``source=None``, so two independent calls on
    the same dimension would silently merge into one Q.

    A key needs at least 2 live segments on a scale to run a test;
    absolute encouragement rows use their persisted additive moments for
    the absolute scale, while rows without usable moments are dropped from
    that scale only. A segment unusable on ONE scale only is dropped from
    that scale's math (``excluded="zero_variance"``) and tested normally on
    the other. A degenerate HKSJ pooled variance falls back to the wider
    plug-in interval for that row only.
    Only when posterior integration cannot be resolved within the numerical
    budget are that scale's ``shrunken`` rows withheld
    (``excluded="estimation_failed"``, with a
    ``breakout.heterogeneity.posterior_integration_unresolved`` warning
    carrying the failure's context); the ``raw`` rows are unaffected.

    The relative-scale ``raw`` row reuses ``BreakoutEstimate.lift``
    verbatim when no rebuild is needed; Q/tau^2/pooled/shrunken use the
    raw pre-update ``lift.log_mean``/``lift.log_se`` (see
    :func:`_relative_raw_lift`). A row whose stored interval was set by
    a BH/FCR family-selection correction is withheld instead of rebuilt
    (``excluded="reference_not_normal"``, ``lift=None``) only when the
    requested *alpha* differs from that row's own corrected alpha --
    rebuilding it at a different, uncorrected alpha would silently
    re-narrow or widen it back to nominal coverage. A matching-alpha
    family-corrected row needs no such withhold and reuses its stored
    interval verbatim instead.

    Absolute-primary raw rows likewise preserve their stored intervals at a
    matching alpha. Allowed rebuilds use the primary sampling reference,
    not a relative row's additive-sidecar reference.

    Parameters
    ----------
    estimates : BreakoutEstimates
        A single ``run_breakout`` call's dense output.
    alpha : float
        Two-sided significance level, shared by every interval.
    tau_prior_scale : float
        HalfNormal prior scale on tau for the tau-marginalised shrinkage,
        forwarded to :func:`~increment.estimation.meta.
        marginalized_segment_intervals`. The default (0.30) is calibrated
        for the log-RR (relative) scale; an absolute-scale metric whose
        true between-segment tau is not of that order (e.g. a dollar
        metric) over-shrinks under the default with no warning -- pass a
        scale matching that metric's own units.
    Raises
    ------
    ValueError
        Propagated from ``cochran_q``/``hksj_pooled_mean`` on malformed
        moment inputs; also raised when a live row's stored relative-scale
        interval must be rebuilt at *alpha* but the rebuild would
        misrepresent it (wrong shape or reference distribution -- see
        :func:`_relative_raw_lift`). A row corrected by a BH/FCR
        family-selection pass is withheld instead of raising when its
        own corrected alpha differs from the requested one (see above).
    Returns
    -------
    SegmentHeterogeneityResult
        ``(summary, segments)`` - see :class:`HeterogeneitySummary` and
        :class:`SegmentEstimate`.
    """
    if any(row.sequential_result is not None for row in estimates):
        from increment.sequential_state import sequential_refuse

        sequential_refuse(
            "route.unsupported",
            "segment heterogeneity and shrinkage need fixed-horizon covariance evidence; use the registered breakout confidence sets",
        )
    # Fail fast and uniformly: an out-of-range scale would otherwise surface
    # only once some grouping reaches the shrinkage step, and not at all when
    # none does.
    if not tau_prior_scale > 0:
        _raise("breakout.tau_prior_scale", tau_prior_scale=tau_prior_scale)
    groups: dict[
        tuple[str, str, str, str, str | None, str, ValueScale],
        list[BreakoutEstimate],
    ] = {}
    for e in estimates:
        groups.setdefault(_group_key(e), []).append(e)

    summary_rows: list[HeterogeneitySummary] = []
    segment_rows: list[SegmentEstimate] = []

    for (
        metric,
        method,
        group_id,
        dimension,
        source,
        estimand,
        value_scale,
    ), rows in groups.items():
        n_design = sum(1 for r in rows if r.excluded in DESIGN_BASED_REASONS)
        n_outcome_upstream = sum(
            1 for r in rows if r.excluded is not None and r.excluded not in DESIGN_BASED_REASONS
        )
        live = [r for r in rows if r.excluded is None]
        excluded = [r for r in rows if r.excluded is not None]
        baseline_by_row = {id(r): _baseline_mean(r) for r in live}

        if len(live) < 2:
            continue

        for scale in ("relative", "absolute"):
            if scale == "relative":
                raw_est: list[float | None] = []
                raw_var: list[float | None] = []
                for row in live:
                    log_mean, log_se = _relative_meta_moments(row)
                    raw_est.append(log_mean)
                    raw_var.append(log_se**2 if log_se is not None else None)
            elif value_scale == "absolute":
                additive = [_absolute_meta_moments(row) for row in live]
                raw_est = [moments[0] if moments is not None else None for moments in additive]
                raw_var = [moments[1] ** 2 if moments is not None else None for moments in additive]
            else:
                raw_est = [r.abs_diff for r in live]
                raw_var = [r.abs_se**2 if r.abs_se is not None else None for r in live]

            # A row missing, or non-positive/non-finite, variance on this
            # scale is dropped from this scale's math only (other rows unaffected).
            good_rows: list[BreakoutEstimate] = []
            bad_rows: list[BreakoutEstimate] = []
            est: list[float] = []
            var: list[float] = []
            for row, e, v in zip(live, raw_est, raw_var, strict=True):
                if (
                    e is None
                    or v is None
                    or not math.isfinite(e)
                    or not (v > 0 and math.isfinite(v))
                ):
                    bad_rows.append(row)
                else:
                    good_rows.append(row)
                    est.append(e)
                    var.append(v)

            if len(good_rows) < 2:
                continue
            n_outcome = n_outcome_upstream + len(bad_rows)

            het = cochran_q(est, var, alpha=alpha)
            marg = _shrinkage_intervals(
                est,
                var,
                good_rows[0],
                alpha=alpha,
                tau_prior_scale=tau_prior_scale,
                scale=scale,
            )
            tau2_reported = het.tau2 if n_outcome == 0 else None
            i2_reported = het.i2 if n_outcome == 0 else None
            i2_lb_reported = het.i2_lb if n_outcome == 0 else None
            i2_ub_reported = het.i2_ub if n_outcome == 0 else None

            try:
                mu_hat, _pooled_se, pooled_lb, pooled_ub = hksj_pooled_mean(
                    est, var, het.tau2, alpha=alpha
                )
            except ValueError as exc:
                if "degenerate" not in str(exc):
                    raise
                # hksj_pooled_mean refuses a degenerate pooled variance;
                # fall back to the wider plug-in interval for this row only.
                _warn(
                    "breakout.heterogeneity.hksj_degenerate_variance_fallback",
                    metric=metric,
                    group_id=group_id,
                    dimension=dimension,
                    source=source,
                    estimand=estimand,
                    value_scale=value_scale,
                    scale=scale,
                    stacklevel=2,
                )
                w_arr = [1.0 / (v + het.tau2) for v in var]
                w_sum = sum(w_arr)
                mu_hat = sum(wk * ek for wk, ek in zip(w_arr, est, strict=True)) / w_sum
                se_plug_in = math.sqrt(1.0 / w_sum)
                z_pooled = _z_for_alpha(alpha)
                pooled_lb = mu_hat - z_pooled * se_plug_in
                pooled_ub = mu_hat + z_pooled * se_plug_in

            pooled = (
                Estimate(
                    value=math.expm1(mu_hat),
                    lb=math.expm1(pooled_lb),
                    ub=math.expm1(pooled_ub),
                    level=math.fsum((1.0, -alpha)),
                    alpha=alpha,
                )
                if scale == "relative"
                else Estimate(
                    value=mu_hat,
                    lb=pooled_lb,
                    ub=pooled_ub,
                    level=math.fsum((1.0, -alpha)),
                    alpha=alpha,
                )
            )

            summary_rows.append(
                HeterogeneitySummary(
                    metric=metric,
                    method=method,
                    method_role=rows[0].method_role,
                    group_id=group_id,
                    dimension=dimension,
                    source=source,
                    estimand=estimand,
                    value_scale=value_scale,
                    scale=scale,
                    k=het.k,
                    q=het.q,
                    p_value=het.p_value,
                    tau2=tau2_reported,
                    i2=i2_reported,
                    i2_lb=i2_lb_reported,
                    i2_ub=i2_ub_reported,
                    n_excluded_design=n_design,
                    n_excluded_outcome=n_outcome,
                    pooled=pooled,
                )
            )

            for idx, (row, e_k, v_k) in enumerate(zip(good_rows, est, var, strict=True)):
                base = baseline_by_row[id(row)]
                raw_excluded: ExclusionReason | None = None
                if scale == "relative":
                    corrected = _family_selection_corrected(row)
                    raw_lift = _relative_raw_lift(row, alpha, strict=not corrected)
                    if raw_lift is None and corrected:
                        raw_excluded = "reference_not_normal"
                else:
                    raw_lift = _absolute_raw_lift(row, alpha, e_k, v_k)
                    if raw_lift is None:
                        raw_excluded = "reference_not_normal"
                if marg is None:
                    # Unresolved posterior integration on this scale: the
                    # shrunken estimate is withheld - see the warning above.
                    shrink_value: float | None = None
                    shrunken_excluded: ExclusionReason | None = "estimation_failed"
                    shrunken_lift = None
                else:
                    theta_k, lb_k, ub_k = marg.theta[idx], marg.lb[idx], marg.ub[idx]
                    shrink_value = float(marg.shrink_k[idx])
                    shrunken_excluded = None
                    shrunken_lift = (
                        Estimate(
                            value=math.expm1(theta_k),
                            lb=math.expm1(lb_k),
                            ub=math.expm1(ub_k),
                            level=math.fsum((1.0, -alpha)),
                            alpha=alpha,
                        )
                        if scale == "relative"
                        else Estimate(
                            value=float(theta_k),
                            lb=float(lb_k),
                            ub=float(ub_k),
                            level=math.fsum((1.0, -alpha)),
                            alpha=alpha,
                        )
                    )
                segment_rows.append(
                    SegmentEstimate(
                        metric=metric,
                        method=method,
                        method_role=row.method_role,
                        group_id=group_id,
                        dimension=dimension,
                        dimension_value=row.dimension_value,
                        source=source,
                        estimand=estimand,
                        value_scale=value_scale,
                        scale=scale,
                        estimator="raw",
                        shrink_k=None,
                        baseline=base,
                        excluded=raw_excluded,
                        lift=raw_lift,
                        role=row.role,
                        discovery=row.discovery,
                        family_axes=row.family_axes,
                        family_q=row.family_q,
                        family_threshold=row.family_threshold,
                    )
                )
                segment_rows.append(
                    SegmentEstimate(
                        metric=metric,
                        method=method,
                        group_id=group_id,
                        dimension=dimension,
                        method_role=row.method_role,
                        dimension_value=row.dimension_value,
                        source=source,
                        estimand=estimand,
                        value_scale=value_scale,
                        scale=scale,
                        estimator="shrunken",
                        shrink_k=shrink_value,
                        baseline=base,
                        excluded=shrunken_excluded,
                        lift=shrunken_lift,
                        role=row.role,
                        discovery=row.discovery,
                        family_axes=row.family_axes,
                        family_q=row.family_q,
                        family_threshold=row.family_threshold,
                    )
                )

            for row in bad_rows:
                # A bad row's verbatim relative lift is informational only;
                # a mismatched level rebuilds (or withholds) via the same helper.
                for estimator in ("raw", "shrunken"):
                    segment_rows.append(
                        SegmentEstimate(
                            metric=metric,
                            method=method,
                            group_id=group_id,
                            dimension=dimension,
                            method_role=row.method_role,
                            dimension_value=row.dimension_value,
                            source=source,
                            estimand=estimand,
                            value_scale=value_scale,
                            scale=scale,
                            estimator=estimator,
                            shrink_k=None,
                            baseline=baseline_by_row[id(row)],
                            excluded="zero_variance",
                            lift=(
                                _relative_raw_lift(row, alpha, strict=False)
                                if (scale == "relative" and estimator == "raw")
                                else None
                            ),
                            role=row.role,
                            discovery=row.discovery,
                            family_axes=row.family_axes,
                            family_q=row.family_q,
                            family_threshold=row.family_threshold,
                        )
                    )
            for row in excluded:
                for estimator in ("raw", "shrunken"):
                    segment_rows.append(
                        SegmentEstimate(
                            metric=metric,
                            method=method,
                            group_id=group_id,
                            dimension=dimension,
                            dimension_value=row.dimension_value,
                            source=source,
                            method_role=row.method_role,
                            estimand=estimand,
                            value_scale=value_scale,
                            scale=scale,
                            estimator=estimator,
                            shrink_k=None,
                            baseline=None,
                            excluded=row.excluded,
                            lift=(
                                _relative_raw_lift(row, alpha, strict=False)
                                if (scale == "relative" and estimator == "raw")
                                else None
                            ),
                            role=row.role,
                            discovery=row.discovery,
                            family_axes=row.family_axes,
                            family_q=row.family_q,
                            family_threshold=row.family_threshold,
                        )
                    )

    return SegmentHeterogeneityResult(
        HeterogeneitySummaries(summary_rows), SegmentEstimates(segment_rows)
    )
