"""Per-segment and per-day estimation, thin wrappers over `estimate_lift`.

`estimate_lift` iterates every `(metric, group_id)` row in whatever frame it
is given, keyed only by metric name in its control-arm lookup. Passing it
rows from more than one segment or day at once corrupts that lookup: it
keeps only the last control arm it sees for a metric, and compares every
treatment arm against that one regardless of which segment or day it
actually belongs to. `run_breakout` and `run_daily_lift` own the segment
and day partitioning, so `estimate_lift` itself never sees a mixed slice.

`run_daily` never calls `estimate_lift` at all - it just reduces each
already-one-row-per-day moment directly, so it has no such hazard.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    SupportsIndex,
    cast,
    get_args,
    get_origin,
    overload,
)

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.stats import norm as _norm

from increment._literals import Alternative, Correction, Role, ValueScale
from increment._moment_plan import OPTIONAL_SLOTS, SLOTS, X_SLOT_ROLES
from increment._policy_alpha import resolve_cell_alpha
from increment.compatibility import _conservative_divide
from increment.errors import (
    CapabilityError,
    CodedModel,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)
from increment.estimation.armstats import ArmStats
from increment.estimation.diagnostics import ESTIMATION_DIAGNOSTICS_ALPHA
from increment.estimation.encouragement import ESTIMANDS, estimate_encouragement
from increment.estimation.engine import (
    Method,
    _validate_methods,
    estimate_lift,
)
from increment.estimation.engine import (
    merge_decision_computations as _merge_decision_computations,
)
from increment.estimation.family import decision_cells, family_discovery, select_family
from increment.estimation.inference import LiftGuardError, Prior
from increment.estimation.meta import ESTIMATION_META_ALPHA_TOO_SMALL
from increment.estimation.results import (
    BinomialConfidenceSet,
    Estimate,
    LiftEstimate,
    RelativeConfidenceSet,
    RelativeUnavailableReason,
    _fcr_alpha_for,
    _infer_reference_from_legacy_dof,
    _RowIdentity,
    _validate_absolute_reference_fields,
    _validate_joint_relative_row,
    _validate_reference_fields,
    open_bound_from_two_sided_at_target,
)
from increment.estimation.sequential import (
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
)
from increment.estimation.sequential_result import SequentialInferenceResult, SequentialResult
from increment.estimation.variance import VARIANCE_MODELS
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import Metric, RetentionMetric
from increment.semantics.sequential import ASYMPTOTIC_LAWS
from increment.sequential_state import SequentialSnapshot, require_public_laws, sequential_refuse

if TYPE_CHECKING:
    from increment.decision import CompiledDecisionPlan, DecisionComputation

Renderer = Callable[..., str]


_RETENTION_UNBOUNDED_REMEDY = (
    "Use the as-of view (run_asof/run_asof_lift), which reports the cumulative ratchet "
    "honestly, or declare threshold_days: [a, b] to bound the band."
)
_REFUSALS = refusals(
    InvalidRequestError,
    {
        "breakout.retention.unbounded": RefusalSpec(
            "breakout.retention.unbounded",
            InvalidRequestError,
            template="{fn_name}: retention metric(s) {names!r} declare an unbounded band, so their outcome never completes -- on this view every value would be a lower bound on itself, biased hardest for the least-observed units. {remedy}",
            keys=frozenset({"supported_view"}),
        ),
        "breakout.retention.daily": "{fn_name}: RetentionMetric(s) {names!r} have no valid independent-per-day snapshot -- 'returned within the observation band' is not a property of a single day. Use the as-of view (run_asof/run_asof_lift, calendar axis) or the cohort view (run_daily/run_daily_lift on a retention metric, indexed by exposure date), or pass metrics= to exclude it.",
        "breakout.retention.completion": "{fn_name}: completed_windows_only=True is contradictory for retention metric(s) {names!r} -- their bands are open on the right, so no unit's window ever completes and the gate would admit nothing. Drop completed_windows_only to get the cumulative monitoring series, or declare threshold_days: [a, b] to bound the band.",
        "breakout.quantile": RefusalSpec(
            "breakout.quantile",
            CapabilityError,
            template="quantile metric(s) {names!r} have no moments representation, so {method} cannot serve them -- {reason}. {remedy}",
        ),
        "breakout.metric.daily_winsorization": RefusalSpec(
            "breakout.metric.daily_winsorization",
            CapabilityError,
            lambda *, method, names: (
                f"{method} does not support winsorized metrics ({', '.join(names)}); "
                "use total-grain run() or run_breakout()"
            ),
        ),
        "breakout.retention.encouragement": RefusalSpec(
            "breakout.retention.encouragement",
            CapabilityError,
            template="{fn_name}: retention metric(s) {names!r} are not supported under an encouragement design; its maturity gate changes the first-stage population by outcome metric",
        ),
        "readout.inference.disjoint_slices": RefusalSpec(
            "readout.inference.disjoint_slices",
            UnsupportedRequestError,
            template="run_daily_lift: per-day/per-cohort slices are disjoint, so a sequential guarantee ({inference}) cannot apply to them -- an always-valid sequence does apply to the cumulative as-of view, so pass view='asof' (Analysis.run_asof_lift) or drop inference",
            keys=frozenset({"view"}),
        ),
        "breakout.breakout.exactly_one_lift": RefusalSpec(
            "breakout.breakout.exactly_one_lift",
            InvalidRequestError,
            lambda *, reason="exactly one of lift or excluded must be set": reason,
        ),
        "breakout.breakout.point_unavailable": "a finite point estimate is unavailable for this breakout row",
        "breakout.to_frame_model": "to_frame(): model={model} does not match estimates[0]'s actual type {inferred}",
        "breakout.to_frame_infer": "to_frame(): cannot infer the schema from an empty sequence -- pass model=<the result model class> explicitly, or call <Model>Estimates(estimates).to_frame() instead, which always knows its own model even when empty.",
        "breakout.daily_metric.exactly_one_value": "exactly one of value or unavailable must be set",
        "breakout.daily_lift.exactly_one_unavailable": "exactly one of lift or unavailable must be set",
        "breakout.daily_lift.point_unavailable": "a finite point estimate is unavailable for this daily lift row",
        "breakout.run_breakout_duplicate": "run_breakout: duplicate summary row for segment {dimension}={value!r}, metric={metric!r}, group_id={group_id!r} -- each (segment, metric, arm) cell's moments must arrive as one pre-aggregated row; duplicates would silently discard one row's units (duplicate control: last row wins the control lookup) or double-count the segment downstream (duplicate treatment: two output rows for one cell inflate segment_heterogeneity's k).",
        "breakout.metric_found_group": "Metric(s) {unknown_metrics} found in group_summary but not in the metrics list. Declared metrics: {declared_names}",
        "breakout.run_breakout_correction": "run_breakout: correction must be 'none', 'bonferroni', or 'bh', got {correction!r}",
        "breakout.run_breakout_bh_excludes_prior": "run_breakout: correction='bh' cannot use an informative prior; BH/e-BH family selection requires frequentist p-values/e-values",
        "breakout.run_breakout_methods": "run_breakout: methods=[] requests zero estimation methods -- every declared (metric, arm) pair would then be reported as unestimable with a misleading 'no data' warning, when the real reason is that no method was asked for. Pass methods=None for the default ([Method(name='unadjusted')]) or a non-empty list of Methods.",
        "breakout.run_breakout_dimension": "run_breakout: dimension column {dimension!r} not found in summary. Available columns: {columns}",
        "breakout.run_breakout_group": "run_breakout: group_summary missing required columns: {missing}",
        "breakout.coerce_type_datetime": "cannot coerce {v!r} (type {type_name}) to datetime.date",
        "breakout.daily_group_summary": "daily_group_summary missing required columns: {missing}",
        "breakout.run_daily_lift": "run_daily_lift: correction must be 'none' or 'bonferroni', got {correction!r}",
        "breakout.unknown_estimand_supported": "unknown estimand(s) {unknown_estimands}; supported: {supported}",
        "breakout.compliance_late_view": "compliance and late require view='asof' with an encouragement design",
        "breakout.run_daily_lift_ds_column_found": "run_daily_lift: 'ds' column not found in summary. Available columns: {columns}",
        "breakout.run_daily_lift_dimension_found": "run_daily_lift: dimension column {dimension!r} not found in summary. Available columns: {columns}",
        "breakout.run_daily_lift_metric_declared": "Metric(s) {unknown} found in group_summary but not in the metrics list. Declared metrics: {declared}",
        "breakout.run_daily_lift_plan_missing_procedures": "run_daily_lift: compiled plan is missing procedure(s) for {missing_procedures!r}",
    },
)

_REFUSALS["estimation.meta.alpha_too_small"] = ESTIMATION_META_ALPHA_TOO_SMALL
_REFUSALS["estimation.diagnostics.alpha"] = ESTIMATION_DIAGNOSTICS_ALPHA
_refuse = raiser(_REFUSALS)


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
    "breakout.estimates.row_few_units",
    IncrementWarning,
    lambda *, label, metric, group_id, n: (
        f"{label} metric={metric!r} group_id={group_id!r} "
        f"has fewer than 2 units (n={n}) -- a ddof=1 variance is "
        "undefined, skipped."
    ),
)
_register_warning(
    "breakout.estimates.row_nonpositive_mean",
    IncrementWarning,
    lambda *, label, metric, group_id, mean: (
        f"{label} metric={metric!r} group_id={group_id!r} "
        f"has a non-positive mean (mean={mean}) -- log(mean) is "
        "undefined for a non-positive mean, skipped."
    ),
)
_register_warning(
    "breakout.estimates.row_nonpositive_denominator_mean",
    IncrementWarning,
    lambda *, label, metric, group_id, den_desc: (
        f"{label} metric={metric!r} group_id={group_id!r} "
        f"has a non-positive denominator mean (denominator mean={den_desc}) "
        "-- log(mean) is undefined for a non-positive denominator, skipped."
    ),
)
_register_warning(
    "breakout.estimates.lost_control_metric",
    IncrementWarning,
    lambda *, label, metric_name: (
        f"{label} metric={metric_name!r} lost its control arm to the "
        "per-row estimability gate above (fewer than 2 units, or a "
        "non-positive numerator/denominator mean) -- that metric's "
        "cell for this slice comes back as an unavailable row with "
        "excluded='no_control_arm', unless that arm's own row was "
        "also dropped above, in which case its own reason wins."
    ),
)
_register_warning(
    "breakout.estimates.slice_no_live_control_dropped",
    IncrementWarning,
    lambda *, warning_label, control_group: (
        f"{warning_label} has no usable '{control_group}' control arm remaining for "
        f"any metric (dropped by the estimability gate above) -- skipped."
    ),
)
_register_warning(
    "breakout.estimates.slice_no_control_arm",
    IncrementWarning,
    lambda *, warning_label, control_group: (
        f"{warning_label} has no '{control_group}' control arm for any metric -- skipped."
    ),
)
_register_warning(
    "breakout.estimates.metric_guarded_excluded",
    IncrementWarning,
    lambda *, dimension, segment_value, metric_name, skip_reason: (
        f"run_breakout: segment {dimension}={segment_value!r} "
        f"metric={metric_name!r} -- {skip_reason} -- excluded."
    ),
)
_register_warning(
    "breakout.estimates.route2_no_live_control_row",
    IncrementWarning,
    lambda *, dimension, segment_value, metric_name, control_group, group_id: (
        f"run_breakout: segment {dimension}={segment_value!r} "
        f"metric={metric_name!r} has no live '{control_group}' control "
        f"row in this segment to compare group_id={group_id!r} against -- "
        "no comparison possible, excluded."
    ),
)
_register_warning(
    "breakout.estimates.no_row_for_group_metric",
    IncrementWarning,
    lambda *, dimension, segment_value, group_id, metric_name: (
        f"run_breakout: segment {dimension}={segment_value!r} has no "
        f"'{group_id}' row for metric={metric_name!r} at all -- no comparison "
        "possible, excluded."
    ),
)
_register_warning(
    "breakout.estimates.daily_partial_guarded_arms",
    IncrementWarning,
    lambda *, metric_name: (
        f"run_daily_lift: metric={metric_name!r} retained "
        "estimable arms while excluding guarded arms"
    ),
)


def reject_retention_metrics(
    metrics: Sequence[Metric], fn_name: str, *, view: DayAxisView = "daily"
) -> None:
    """Raise ``ValueError`` for a `RetentionMetric` this *view* cannot report.

    An unbounded band (no ``threshold_days`` upper bound) has no completion
    date, so ``view="daily"``/``"cohort"`` would read it as a lower bound
    that worsens the less a unit has been observed. ``view="daily"`` also
    rejects every retention metric outright: a single day has no retention
    reading at all, bounded or not. Only ``view="asof"`` accepts both band
    shapes - its cumulative series has an honest reading either way.
    """
    if view != "asof":
        unbounded = [
            m.name for m in metrics if isinstance(m, RetentionMetric) and m.band[1] is None
        ]
        if unbounded:
            _refuse(
                "breakout.retention.unbounded",
                fn_name=fn_name,
                names=unbounded,
                remedy=_RETENTION_UNBOUNDED_REMEDY,
                supported_view="asof",
            )

    if view != "daily":
        return

    names = [m.name for m in metrics if isinstance(m, RetentionMetric)]
    if names:
        _refuse("breakout.retention.daily", fn_name=fn_name, names=names)


def reject_winsorized_day_axis(metrics: Sequence[Metric], method: str) -> None:
    names = [metric.name for metric in metrics if getattr(metric, "winsorization", None)]
    if names:
        _refuse("breakout.metric.daily_winsorization", method=method, names=names)


def reject_quantile_metrics(
    metrics: Sequence[Metric], method: str, *, reason: str, remedy: str = "Use run()."
) -> None:
    """A quantile has no moments representation, so only ``run()`` (from
    retained per-unit rows) can serve it. ``remedy`` overrides the default
    ``run()`` pointer for a caller that already is ``run()``.
    """
    names = [metric.name for metric in metrics if metric.type == "quantile"]
    if names:
        _refuse(
            "breakout.quantile",
            names=names,
            method=method,
            reason=reason,
            remedy=remedy,
        )


def reject_contradictory_completed_windows(metrics: Sequence[Metric], fn_name: str) -> None:
    """Raise when completed windows meet an unbounded retention band."""
    open_bands = [m.name for m in metrics if isinstance(m, RetentionMetric) and m.band[1] is None]
    if open_bands:
        _refuse(
            "breakout.retention.completion",
            fn_name=fn_name,
            names=open_bands,
        )


def reject_retention_under_encouragement(metrics: Sequence[Metric], fn_name: str) -> None:
    """Raise when a `RetentionMetric` is used under an ``Encouragement``
    design: retention's maturity gate changes the first-stage population
    by outcome metric, which this function's contract does not support."""
    retention = [m.name for m in metrics if isinstance(m, RetentionMetric)]
    if retention:
        _refuse(
            "breakout.retention.encouragement",
            fn_name=fn_name,
            names=retention,
        )


ExclusionReason = Literal[
    "few_units",
    "no_control_arm",
    "nonpositive_mean",
    "zero_variance",
    "extreme_ratio",
    "estimation_failed",
    "prior_grid_truncated",
    "reference_not_normal",
]
_LIFT_GUARD_EXCLUSION_REASONS: Mapping[str, ExclusionReason] = {
    "non_positive_mean": "nonpositive_mean",
    "zero_variance": "zero_variance",
    "nonfinite_se": "extreme_ratio",
    "delta_method_unreliable": "extreme_ratio",
}


def _failure_exclusion_reason(failure: Any) -> ExclusionReason:
    """Exclude lift-guard failures by their stable reason; preserve other failures."""
    if failure.code != "estimation.engine.lift_guard":
        return "estimation_failed"
    return _LIFT_GUARD_EXCLUSION_REASONS[str(failure.context["reason"])]


"""Why a (segment, metric, method, arm) cell in a breakout came back as unavailable.

``few_units`` (n < 2, no ddof=1 variance) and ``no_control_arm`` (nothing to
compare against -- the control row is missing, dropped, or the pair never
appeared in this segment's raw data at all) are DESIGN-based: they condition
on arm counts, ancillary to the outcome, and are safe to keep in an analysis.
``nonpositive_mean`` (log(mean) undefined), ``zero_variance`` (both arms
degenerate), and ``extreme_ratio`` (log ratio too extreme for the delta
method) are OUTCOME-based: they truncate a tail and would bias a downstream
``tau^2``/``I^2`` estimate over segments if not accounted for (D7).
``estimation_failed`` marks a computation failure unrelated to a lift guard;
the original coded failure is retained by the decision computation.

``zero_variance`` is also reused, at a different granularity, by
:func:`increment.breakout.heterogeneity.segment_heterogeneity`: there it
tags a LIVE segment (``excluded is None`` here) whose sufficient
statistic or variance is simply unusable on ONE scale (relative or
absolute) of that function's own math, not a claim that both arms are
degenerate at the ``run_breakout`` level.

``prior_grid_truncated`` remains in the schema for existing serialized rows.
Posterior integration no longer has a fixed grid ceiling. A numerical
integration failure withholds ``shrunken`` rows as ``estimation_failed``;
the raw sibling rows stay live.

``reference_not_normal`` is likewise only ever produced by
:func:`~increment.breakout.heterogeneity.segment_heterogeneity`. On the
absolute scale it tags a withheld row this scale's interval cannot be
rebuilt for -- a one-sided row, a sequential row whose width is not a
critical value times a standard error, or a row whose additive sidecar
claims a t reference without the degrees of freedom to cut it at. A
sidecar that does carry its own Welch df is rebuilt on that reference,
not on a Normal one. A misrepresented row on the relative scale instead
raises, since its own stored interval could be reused verbatim once a
matching alpha is supplied.

On EITHER scale, it also tags a row whose sufficient statistics were
re-estimated by a BH/FCR family-selection correction, when the
requested alpha differs from that row's own corrected alpha:
rebuilding it at the requested, uncorrected alpha would silently undo
the correction. A family-corrected row whose own alpha already matches
the requested one needs no such withhold on that account -- it reuses
its stored interval verbatim on the relative scale, or, absent any
other ``reference_not_normal`` condition (e.g. being one-sided), is
built fresh at the (already-matching) requested alpha on the absolute
scale.
"""

DESIGN_BASED_REASONS = frozenset({"few_units", "no_control_arm"})


def _validate_flat_binomial_row(row: Any, *, refusal_code: str) -> None:
    """Keep a flattened row's displayed interval and exact evidence coherent."""
    bset = row.binomial_set
    if row.reference_kind != "binomial":
        if bset is not None:
            _refuse(
                refusal_code,
                reason="binomial_set is present on a non-binomial reference row",
            )
        return
    if bset is None:
        _refuse(
            refusal_code,
            reason="reference_kind='binomial' row is missing its sufficient-count set",
        )
    expected_geometry = {
        "two-sided": "central",
        "greater": "lower_bound",
        "less": "upper_bound",
    }.get(row.alternative)
    if expected_geometry is not None and bset.geometry != expected_geometry:
        _refuse(
            refusal_code,
            reason=(
                f"alternative={row.alternative!r} implies geometry="
                f"{expected_geometry!r}, got {bset.geometry!r}"
            ),
        )
    if row.lift is None:
        if bset.point_available:
            _refuse(
                refusal_code,
                reason="binomial_set claims a point-available count pair with lift=None",
            )
        return
    if not bset.point_available:
        _refuse(
            refusal_code,
            reason="lift is finite but binomial_set has no available point (x_c == 0)",
        )
    from increment.estimation.binomial_rr import point_lift

    expected_value = point_lift(bset.x_c, bset.n_c, bset.x_t, bset.n_t)
    if expected_value is None or not math.isclose(
        row.lift.value, expected_value, rel_tol=0.0, abs_tol=1e-9
    ):
        _refuse(
            refusal_code,
            reason=(
                f"lift.value={row.lift.value!r} does not match the persisted counts' "
                f"empirical ratio ({expected_value!r})"
            ),
        )
    if row.lift.alpha != bset.alpha or row.lift.level != bset.level:
        _refuse(
            refusal_code,
            reason=(
                f"lift alpha/level ({row.lift.alpha!r}/{row.lift.level!r}) does not "
                f"match binomial_set ({bset.alpha!r}/{bset.level!r})"
            ),
        )
    if row.lift.lb != bset.lower or row.lift.ub != bset.upper:
        _refuse(
            refusal_code,
            reason=(
                f"lift bounds ({row.lift.lb!r}, {row.lift.ub!r}) do not match "
                f"binomial_set bounds ({bset.lower!r}, {bset.upper!r})"
            ),
        )


def _reject_sequential_mixed_authority(row: Any) -> None:
    """Reject fixed/joint sidecars on a checkpoint-backed sequential view."""
    fields = (
        "relative_confidence_set",
        "relative_unavailable_reason",
        "null_abs",
        "abs_diff",
        "abs_se",
        "abs_lb",
        "abs_ub",
        "abs_reference_kind",
        "abs_reference_df",
    )
    if any(getattr(row, field) is not None for field in fields):
        sequential_refuse(
            "source.invalid",
            "sequential view cannot carry fixed or joint authoritative fields",
        )


class BreakoutEstimate(_RowIdentity):
    """A lift estimate for ONE segment (dimension value) of a breakout.

    Carries the same identity fields as ``LiftEstimate`` (via
    ``_RowIdentity``) plus ``dimension``/``dimension_value`` to identify
    the segment and ``source`` to disambiguate the rare same-name,
    different-source case.

    ``reference_kind``/``reference_df`` preserve the source row's sampling
    reference through serialization. A Welch t reference can have ``dof=None``;
    it must not be reconstructed as a Normal interval.

    ``run_breakout`` returns one row per (segment, metric, method,
    non-control arm) cell, dense - an unestimable cell still gets a row,
    with ``lift=None`` and ``excluded`` naming why (see
    :data:`ExclusionReason`).

    ``low_reliability`` flags a real estimate whose control or
    treatment arm has fewer than ``reliability_floor`` units: estimable,
    but the Wald interval's nominal coverage is not trustworthy there.

    Call ``.to_frame()`` on the :class:`BreakoutEstimates` this returns
    rather than constructing a plain ``list[BreakoutEstimate]``.
    """

    null_lift: float = Field(default=0.0, allow_inf_nan=False)
    dof: float | None = None
    reference_kind: Literal["normal", "t", "sequential", "binomial"] = "normal"
    reference_df: float | None = None
    # Persisted sampling reference; dof remains for compatibility.
    family_axes: tuple[str, ...] | None = None
    family_q: float | None = None
    family_threshold: float | None = None
    # Which family corrected this row, at what FDR level, and that family's
    # own realized cutoff before capping at the nominal alpha. Mirrors
    # LiftEstimate's fields of the same name; see there for the cap.
    family_guarantee: Literal["finite_sample", "asymptotic_sequential"] | None = None
    family_nominal_alpha: float | None = None
    # Mirrors LiftEstimate: a sequential family's regime and nominal alpha.
    dimension: str  # the breakout column name, e.g. "country"
    dimension_value: str  # the segment, e.g. "US"
    null_abs: float | None = Field(default=None, allow_inf_nan=False)
    source: str | None = None  # resolved FactSource name, when known
    lift: Estimate | None
    relative_confidence_set: RelativeConfidenceSet | None = None
    relative_unavailable_reason: RelativeUnavailableReason | None = None
    # None either for a genuinely excluded/unestimable cell (`excluded`
    # set) or for an admitted `reference_kind="binomial"` row with an
    # observed zero control count (`binomial_set` set, point unavailable
    # per `binomial_set.point_available`); see `_lift_or_excluded`.
    sequential_result: SequentialResult | None = None
    binomial_set: BinomialConfidenceSet | None = None
    estimand: str = "itt"
    value_scale: ValueScale = "relative"  # mirrors LiftEstimate.value_scale
    note: str | None = None  # guard fallbacks / suppression reasons / caveats
    abs_diff: float | None = Field(default=None, allow_inf_nan=False)
    abs_se: float | None = Field(default=None, allow_inf_nan=False)
    abs_lb: float | None = Field(default=None, allow_inf_nan=False)
    abs_ub: float | None = Field(default=None, allow_inf_nan=False)
    abs_reference_kind: Literal["normal", "t"] | None = None
    abs_reference_df: float | None = Field(default=None, allow_inf_nan=False, gt=0)
    excluded: ExclusionReason | None = None

    @model_validator(mode="after")
    def _lift_or_excluded(self):
        """Exactly one of three states: excluded (neither lift nor a set),
        point-backed (lift set, no excluded), or -- exclusively for an
        admitted ``reference_kind="binomial"`` zero-control-count row --
        set-only (neither lift nor excluded, a typed ``binomial_set``
        with no available point). Never a bare lift=None/excluded=None
        claiming full-space availability, and never a set attached to a
        genuinely excluded row.
        """
        if self.sequential_result is not None and self.reference_kind != "sequential":
            from increment.sequential_state import sequential_refuse

            sequential_refuse("source.invalid", "fixed view cannot carry sequential evidence")
        if self.reference_kind == "sequential":
            _reject_sequential_mixed_authority(self)
            from increment.sequential_state import sequential_refuse

            if self.sequential_result is None:
                sequential_refuse(
                    "continuation.legacy", "sequential view lost its exact checkpoint"
                )
            require_public_laws(
                (self.sequential_result.checkpoint.model,), "BreakoutEstimate replay"
            )
            from increment.estimation.sequential_runtime import (
                display_estimate,
                require_selected_widening,
            )

            if self.lift != display_estimate(self.sequential_result):
                sequential_refuse(
                    "source.invalid", "view point or interval differs from its stopped state"
                )
            require_selected_widening(
                self.sequential_result,
                discovery=self.discovery,
                family_threshold=self.family_threshold,
                family_nominal_alpha=self.family_nominal_alpha,
            )
            cp = self.sequential_result.checkpoint
            if (cp.model.law in ASYMPTOTIC_LAWS) != (self.inference == "asymptotic_mean"):
                sequential_refuse(
                    "source.invalid", "displayed validity regime differs from its checkpoint"
                )
            if (self.metric, self.group_id, self.estimand, self.alternative, self.null_lift) != (
                cp.cell.metric,
                cp.cell.group_id,
                cp.cell.estimand,
                cp.cell.alternative,
                float(cp.cell.null_lift),
            ):
                sequential_refuse(
                    "source.invalid", "view hypothesis differs from its certified checkpoint"
                )
            if self.excluded is not None or self.binomial_set is not None:
                sequential_refuse(
                    "source.invalid",
                    "certified sequential abstention belongs in its confidence geometry",
                )
            if cp.cell.segment != ((self.dimension, self.dimension_value),):
                sequential_refuse(
                    "source.invalid", "breakout segment differs from its certified checkpoint"
                )
            return self
        if self.excluded is not None:
            if (
                self.lift is not None
                or self.binomial_set is not None
                or self.relative_confidence_set is not None
                or self.relative_unavailable_reason is not None
            ):
                _refuse(
                    "breakout.breakout.exactly_one_lift",
                    reason="excluded is set; lift and binomial_set must both be None",
                )
            return self
        if _validate_joint_relative_row(self):
            return self
        if self.lift is None:
            if self.reference_kind != "binomial" or self.binomial_set is None:
                _refuse(
                    "breakout.breakout.exactly_one_lift",
                    reason="lift is None outside a retained confidence-set or binomial point-unavailable case",
                )
            if self.binomial_set.point_available:
                _refuse(
                    "breakout.breakout.exactly_one_lift",
                    reason="binomial_set claims a point-available count pair with lift=None",
                )
        elif self.reference_kind == "binomial" and self.binomial_set is None:
            _refuse(
                "breakout.breakout.exactly_one_lift",
                reason="reference_kind='binomial' row is missing its sufficient-count set",
            )
        _validate_flat_binomial_row(
            self,
            refusal_code="breakout.breakout.exactly_one_lift",
        )
        return self

    def require_lift(self) -> Estimate:
        """Return the finite point estimate or refuse for excluded/set-only rows."""
        if self.lift is None:
            _refuse("breakout.breakout.point_unavailable")
        return self.lift

    low_reliability: bool = False
    n_treat: float | None = None
    n_control: float | None = None
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
    role: str | None = None
    # Always "exploratory" from run_breakout()/readouts.breakout(): plan roles are not
    # meaningful per segment. Optional for compatibility with rows built without a role.
    discovery: bool | None = None
    # Whether this (metric, arm, segment) cell survived the call-wide BH/e-BH selection
    # (`correction="bh"`); None for "none"/"bonferroni" rows and untested excluded cells. This is
    # family selection, not stat_sig: select_family's `fcr_alpha = min(realized, nominal_alpha)`
    # cap can leave a selected cell's interval containing the null.
    policy_name: Literal["compiled_plan", "default_exploratory"] = "default_exploratory"
    # Which policy source produced this row. Standalone calls use the named
    # lightweight default; readouts.breakout stamps its compiled-plan boundary.

    _infer_reference: ClassVar[Any] = model_validator(mode="before")(
        _infer_reference_from_legacy_dof
    )
    _check_reference: ClassVar[Any] = model_validator(mode="after")(_validate_reference_fields)
    _check_absolute_reference: ClassVar[Any] = model_validator(mode="after")(
        _validate_absolute_reference_fields
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


Backend = Literal["pandas", "polars", "pyarrow"]

# Which day-axis view produced the moments passed to run_daily/
# run_daily_lift; governs the retention guard and emitted ds_basis.
DayAxisView = Literal["daily", "asof", "cohort"]


def _scalar_dtype(annotation: Any) -> nw.dtypes.DType:
    """Map a pydantic field's declared type to a narwhals dtype.

    Unwraps ``X | None`` to ``X`` first. ``bool`` is checked before
    ``int`` since Python's ``bool`` is a subclass of ``int``.
    """
    args = [a for a in get_args(annotation) if a is not type(None)]
    base = args[0] if args else annotation
    if base is date or base is datetime:
        return nw.Datetime()
    if base is bool:
        # An OPTIONAL bool needs a null-carrying pandas column - see
        # _is_optional_bool and to_frame's use of it.
        return nw.Boolean()
    if base is int:
        # An OPTIONAL int must survive a None cell; pandas' numpy-backed
        # Int64 cannot, so it rides as Float64 instead.
        return nw.Float64() if type(None) in get_args(annotation) else nw.Int64()
    if base is float:
        return nw.Float64()
    return nw.String()


def _is_optional_float(annotation: Any) -> bool:
    """Whether a field is declared ``float | None``."""
    args = get_args(annotation)
    return float in args and type(None) in args


def _is_optional_bool(annotation: Any) -> bool:
    """Whether a field is declared ``bool | None`` - the one type
    pandas' numpy-backed boolean column cannot represent a null for."""
    args = get_args(annotation)
    return bool in args and type(None) in args


def _is_optional_string(annotation: Any) -> bool:
    """Whether a field is a nullable string or string-valued literal."""
    args = get_args(annotation)
    if type(None) not in args:
        return False
    return any(arg is str or get_origin(arg) is Literal for arg in args if arg is not type(None))


def _is_optional_model(annotation: Any) -> bool:
    args = get_args(annotation)
    return type(None) in args and any(
        isinstance(arg, type) and issubclass(arg, BaseModel)
        for arg in args
        if arg is not type(None)
    )


def _is_estimate_annotation(annotation: Any) -> bool:
    """Whether a field is an ``Estimate`` or optional ``Estimate``."""
    return annotation is Estimate or Estimate in get_args(annotation)


def _is_optional_binomial_set(annotation: Any) -> bool:
    """Whether a field is declared ``BinomialConfidenceSet | None`` - the
    sole confidence-set-typed field. Detected by name, not structurally
    like :func:`_is_optional_model`: it needs typed ``set_lower``/
    ``set_upper``/``set_level`` columns in :func:`to_frame`, not that
    generic BaseModel-valued-scalar's ``repr()`` string fallback (which
    every OTHER model-valued field, e.g. ``prior_spec``, still gets)."""
    args = get_args(annotation)
    return type(None) in args and any(
        isinstance(arg, type) and issubclass(arg, BinomialConfidenceSet)
        for arg in args
        if arg is not type(None)
    )


def _frame_columns(
    model: type[BaseModel], estimate_field: str | None, binomial_set_field: str | None
) -> list[str]:
    columns: list[str] = []
    for name in model.model_fields:
        if name == estimate_field:
            columns.extend((name, "lb", "ub", "open_side"))
        elif name == "sequential_result":
            columns.extend(
                (
                    name,
                    "sequential_lower",
                    "sequential_upper",
                    "sequential_status",
                    "sequential_log_e",
                    "sequential_point_reason",
                    "sequential_validity_regime",
                    "sequential_alpha",
                    "sequential_components",
                )
            )
        elif name == binomial_set_field:
            columns.extend(("set_lower", "set_upper", "set_level"))
        else:
            columns.append(name)
    return columns


def _append_frame_value(
    data: dict[str, list[Any]],
    name: str,
    value: Any,
    estimate_field: str | None,
    binomial_set_field: str | None,
) -> None:
    if name == estimate_field:
        data[name].append(None if value is None else value.value)
        data["lb"].append(None if value is None else value.lb)
        data["ub"].append(None if value is None else value.ub)
        data["open_side"].append(None if value is None else value.open_side)
    elif name == "sequential_result":
        from increment.estimation.sequential_runtime import _outward

        data[name].append(None if value is None else value.model_dump_json())
        data["sequential_lower"].append(
            None if value is None else _outward(value.bounds.lower, lower=True)
        )
        data["sequential_upper"].append(
            None if value is None else _outward(value.bounds.upper, lower=False)
        )
        data["sequential_status"].append(None if value is None else value.bounds.status)
        data["sequential_log_e"].append(
            str(value.log_e) if isinstance(value, SequentialInferenceResult) else None
        )
        data["sequential_point_reason"].append(None if value is None else value.point_reason)
        data["sequential_validity_regime"].append(
            None if value is None else getattr(value, "validity_regime", "finite_sample")
        )
        data["sequential_alpha"].append(None if value is None else str(value.decision_alpha))
        data["sequential_components"].append(
            None
            if value is None or isinstance(value, SequentialInferenceResult)
            else value.bounds.model_dump_json(include={"components"})
        )
    elif name == binomial_set_field:
        data["set_lower"].append(None if value is None else value.lower)
        data["set_upper"].append(None if value is None else value.upper)
        data["set_level"].append(None if value is None else value.level)
    elif value is None:
        data[name].append(None)
    elif isinstance(value, date) and not isinstance(value, datetime):
        data[name].append(datetime.combine(value, datetime.min.time()))
    elif isinstance(value, BaseModel):
        data[name].append(repr(value))
    elif isinstance(value, Sequence) and not isinstance(value, str):
        data[name].append(repr(value))
    else:
        data[name].append(value)


def _copy_common_fields(source: LiftEstimate, /, **overrides: Any) -> dict[str, Any]:
    """Extract the `_RowIdentity` fields plus every other field
    BreakoutEstimate/DailyLiftEstimate mirror from a LiftEstimate-shaped
    source, as a kwargs dict ready for the target model's constructor.
    `overrides` wins over the copied value for any key present in both,
    and may also add keys `_copy_common_fields` itself doesn't copy
    (e.g. BreakoutEstimate's winsor_* sidecar).
    """
    copied = {
        "metric": source.metric,
        "group_id": source.group_id,
        "method": source.method,
        "method_role": source.method_role,
        "inference": source.inference,
        "alternative": source.alternative,
        "null_lift": source.null_lift,
        "dof": source.dof,
        "reference_kind": source.reference_kind,
        "reference_df": source.reference_df,
        "family_axes": source.family_axes,
        "family_q": source.family_q,
        "family_threshold": source.family_threshold,
        "family_guarantee": source.family_guarantee,
        "family_nominal_alpha": source.family_nominal_alpha,
        "null_abs": source.null_abs,
        "lift": source.lift,
        "relative_confidence_set": source.relative_confidence_set,
        "relative_unavailable_reason": source.relative_unavailable_reason,
        "sequential_result": source.sequential_result,
        "binomial_set": source.binomial_set,
        "estimand": source.estimand,
        "value_scale": source.value_scale,
        "note": source.note,
        "abs_diff": source.abs_diff,
        "abs_se": source.abs_se,
        "abs_lb": source.abs_lb,
        "abs_ub": source.abs_ub,
        "abs_reference_kind": source.abs_reference_kind,
        "abs_reference_df": source.abs_reference_df,
    }
    copied.update(overrides)
    return copied


def to_frame[M: BaseModel](
    estimates: Sequence[M],
    model: type[M] | None = None,
    backend: Backend = "pandas",
) -> IntoDataFrame:
    """Convert a sequence of result models (:class:`LiftEstimate`,
    :class:`BreakoutEstimate`, :class:`DailyMetricValue`,
    :class:`DailyLiftEstimate`) to a native ``backend`` frame.

    Generic over pydantic's ``model_fields``: an ``Estimate``-typed field
    flattens into four columns (its name, plus ``lb``/``ub``/``open_side``);
    a ``binomial_set`` field (see :class:`~increment.estimation.results.
    BinomialConfidenceSet`) flattens into ``set_lower``/``set_upper``/
    ``set_level`` -- always the row's confidence-set bounds/level, even
    for a set-only row with no finite point (``<estimate field>`` and
    ``lb``/``ub`` stay ``None`` there; ``set_lower``/``set_upper``/
    ``set_level`` are the row's ONLY confidence-set representation in
    that case; see :class:`BinomialConfidenceSet`); every other field
    passes through as a scalar column in declaration order. Most callers
    should use ``results.to_frame()`` on a pipeline's own result rather
    than calling this function directly.

    Parameters
    ----------
    estimates : Sequence[M]
        Any sequence of one supported result model. May be empty.
    model : type[M] | None
        Which model *estimates* holds. Required when *estimates* is
        empty, since an empty sequence carries no runtime type trace.
    backend : {"pandas", "polars", "pyarrow"}
        Which native library to build.

    Returns
    -------
    IntoDataFrame
        One row per estimate, columns in the model's field order, with
        the ``Estimate``-typed field expanded to
        ``<field name>``/``lb``/``ub``/``open_side`` and a
        ``binomial_set`` field expanded to ``set_lower``/``set_upper``/
        ``set_level``. ``open_side`` is ``"lower"``/``"upper"`` for a
        genuinely unbounded one-sided endpoint, and ``None`` both for a
        closed interval (``lb``/``ub`` both set) and for an unavailable
        one (``lb``/``ub``/``value`` all ``None``) -- distinguish the
        two by whether ``<field name>`` (the point estimate) is
        ``None``.
    """
    if estimates:
        inferred = type(estimates[0])
        if model is None:
            model = inferred
        elif not isinstance(estimates[0], model):
            _refuse("breakout.to_frame_model", model=model.__name__, inferred=inferred.__name__)
    elif model is None:
        _refuse("breakout.to_frame_infer")

    estimate_field = next(
        (
            name
            for name, info in model.model_fields.items()
            if _is_estimate_annotation(info.annotation)
        ),
        None,
    )
    binomial_set_field = next(
        (
            name
            for name, info in model.model_fields.items()
            if _is_optional_binomial_set(info.annotation)
        ),
        None,
    )
    columns = _frame_columns(model, estimate_field, binomial_set_field)
    data: dict[str, list[Any]] = {c: [] for c in columns}
    for est in estimates:
        for name in model.model_fields:
            _append_frame_value(
                data,
                name,
                getattr(est, name),
                estimate_field,
                binomial_set_field,
            )

    schema = {
        name: (
            nw.Float64()
            if name
            in (
                estimate_field,
                "lb",
                "ub",
                "set_lower",
                "set_upper",
                "set_level",
                "sequential_lower",
                "sequential_upper",
            )
            else nw.String()
            if name
            in (
                "open_side",
                "sequential_status",
                "sequential_log_e",
                "sequential_point_reason",
                "sequential_validity_regime",
                "sequential_alpha",
                "sequential_components",
            )
            else _scalar_dtype(model.model_fields[name].annotation)
        )
        for name in columns
    }
    if data.get("ds"):
        non_null = [day for day in data["ds"] if day is not None]
        if non_null:
            day_type = (
                datetime
                if isinstance(non_null[0], datetime)
                else float
                if any(isinstance(day, float) for day in non_null)
                else type(non_null[0])
            )
            # A day type mixed with nulls needs a nullable dtype (an
            # optional int maps to Float64, not numpy-backed Int64,
            # which cannot hold a null cell).
            has_nulls = len(non_null) < len(data["ds"])
            schema["ds"] = _scalar_dtype(day_type | None if has_nulls else day_type)
        # else: every value is None -- keep the annotation-derived Datetime schema.
    frame = nw.from_dict(data, schema=schema, backend=backend).to_native()

    nullable_strings = [
        name
        for name, info in model.model_fields.items()
        if name != estimate_field
        and name != binomial_set_field
        and (_is_optional_string(info.annotation) or _is_optional_model(info.annotation))
        and schema[name] == nw.String()
    ]
    if estimate_field is not None:
        nullable_strings.append("open_side")
    if "sequential_result" in model.model_fields:
        nullable_strings.extend(
            (
                "sequential_status",
                "sequential_log_e",
                "sequential_point_reason",
                "sequential_validity_regime",
                "sequential_alpha",
                "sequential_components",
            )
        )
    if backend == "pandas" and nullable_strings:
        # Rewritten after construction: narwhals may otherwise infer a
        # floating object column for all-null optional strings.  Pandas'
        # nullable StringDtype retains real strings and native <NA> values.
        import pandas as pd

        for name in nullable_strings:
            frame[name] = pd.array(data[name], dtype="string")

    nullable_bools = [
        name for name, info in model.model_fields.items() if _is_optional_bool(info.annotation)
    ]
    if backend == "pandas" and nullable_bools:
        # Rewritten after construction: narwhals resolves nw.Boolean() to
        # whichever pandas dtype the values happen to allow.
        import pandas as pd

        for name in nullable_bools:
            frame[name] = pd.array(data[name], dtype="boolean")
    nullable_floats = [
        name
        for name, info in model.model_fields.items()
        if name != estimate_field
        and name != binomial_set_field
        and _is_optional_float(info.annotation)
        and schema[name].is_float()
    ]
    if binomial_set_field is not None:
        nullable_floats.extend(("set_lower", "set_upper", "set_level"))
    if backend == "pandas" and nullable_floats:
        # Pandas' nullable Float64 dtype preserves unavailable values as
        # <NA>, rather than conflating them with a floating-point NaN.
        import pandas as pd

        for name in nullable_floats:
            frame[name] = pd.array(data[name], dtype="Float64")

    return frame


class EstimateList[M: BaseModel](list[M]):
    """A ``list`` subclass that carries its element model as a class
    attribute (``_model``), so :meth:`to_frame` resolves the right schema
    even when the instance holds zero elements.

    Slicing and concatenation (``results[1:3]``, ``results + other``)
    preserve the subclass; a list comprehension over the results does
    not - use :func:`to_frame` directly with an explicit ``model=`` for
    that case.
    """

    _model: ClassVar[type[BaseModel]]

    def to_frame(self, backend: Backend = "pandas") -> IntoDataFrame:
        """Convert this list to a native ``backend`` frame - see :func:`to_frame`."""
        return to_frame(self, model=self._model, backend=backend)

    @overload
    def __getitem__(self, key: SupportsIndex) -> M: ...
    @overload
    def __getitem__(self, key: slice) -> EstimateList[M]: ...
    def __getitem__(self, key: SupportsIndex | slice) -> M | EstimateList[M]:
        result = super().__getitem__(key)
        if isinstance(key, slice):
            return cast("EstimateList[M]", type(self)(cast("list[M]", result)))
        return cast("M", result)

    def __add__(self, other: list[M]) -> EstimateList[M]:
        return cast("EstimateList[M]", type(self)([*self, *other]))


class DailyMetricValue(CodedModel, BaseModel):
    """One arm's absolute metric mean (with CI) for ONE day.

    The time-series counterpart to a plain arm-level mean - what plots
    on a per-day chart of the raw value, as opposed to the *relative*
    lift between arms (see :class:`DailyLiftEstimate`). Produced by
    :func:`run_daily` directly from one ``daily_group_summary`` row.

    ``dimension``/``dimension_value``/``source`` are populated together
    when :func:`run_daily` is given a ``dimension``, and ``None``
    otherwise.

    ``ds_basis`` says what ``ds`` indexes: ``"calendar"`` (default) is
    an observation date; ``"cohort"`` means ``ds`` is the unit's own
    exposure date, how a retention series is conventionally indexed.

    Call ``.to_frame()`` on the :class:`DailyMetricValues` this function
    returns rather than constructing a plain list.
    """

    model_config = ConfigDict(frozen=True)

    ds: date
    metric: str
    group_id: str
    value: Estimate | None  # absolute mean; Wald CI: mean +/- z * sqrt(var / n)
    unavailable: ExclusionReason | None = None
    n: int
    ds_basis: Literal["calendar", "cohort"] = "calendar"
    dimension: str | None = None  # the breakout column name, e.g. "country"
    dimension_value: str | None = None  # the segment, e.g. "US"
    source: str | None = None  # resolved FactSource name, when known

    @model_validator(mode="after")
    def _value_or_unavailable(self):
        if (self.value is None) == (self.unavailable is None):
            _refuse("breakout.daily_metric.exactly_one_value")
        return self


class DailyLiftEstimate(_RowIdentity):
    """A relative lift estimate for ONE day of a daily time series.

    Carries the same identity fields as :class:`BreakoutEstimate` and
    :class:`~increment.estimation.results.LiftEstimate` (via
    ``_RowIdentity``) plus ``ds`` to identify which day's slice of
    moments produced it.

    ``estimand``/``value_scale``/``note`` mirror :class:`BreakoutEstimate`'s
    fields of the same name.
    ``ds`` retains calendar dates, numeric day indices, or structured string
    labels for as-of frame readouts.
    ``low_reliability`` marks a real estimate whose control or
    treatment arm has fewer than ``reliability_floor`` units that day.

    ``dimension``/``dimension_value``/``source`` are populated together
    when :func:`run_daily_lift` is given a ``dimension``, and ``None``
    otherwise. ``ds_basis`` - see :class:`DailyMetricValue`.
    """

    dof: float | None = None
    reference_kind: Literal["normal", "t", "sequential", "binomial"] = "normal"
    reference_df: float | None = None
    ds: date | datetime | str | int | float
    lift: Estimate | None
    relative_confidence_set: RelativeConfidenceSet | None = None
    relative_unavailable_reason: RelativeUnavailableReason | None = None
    unavailable: ExclusionReason | None = None
    # None either for a genuinely unavailable cell (`unavailable` set) or
    # for an admitted `reference_kind="binomial"` row with an observed
    # zero control count (`binomial_set` set, point unavailable per
    # `binomial_set.point_available`); see `_lift_or_unavailable`.
    sequential_result: SequentialResult | None = None
    binomial_set: BinomialConfidenceSet | None = None
    # Sufficient counts/method for every exact-binomial-estimated row
    # (point-backed or not); None for every other row.
    estimand: str = "itt"  # "itt" | "compliance" | "late" (mirrors LiftEstimate)
    value_scale: ValueScale = "relative"  # mirrors LiftEstimate.value_scale
    note: str | None = None  # guard fallbacks / suppression reasons / caveats
    ds_basis: Literal["calendar", "cohort"] = "calendar"
    dimension: str | None = None  # the breakout column name, e.g. "country"
    dimension_value: str | None = None  # the segment, e.g. "US"
    source: str | None = None  # resolved FactSource name, when known
    low_reliability: bool = False
    role: Role | None = None
    # The declared-plan role this day/as-of cell was estimated under; drives the
    # per-cell alpha split. None when no plan was declared (exploratory default).
    null_lift: float = Field(default=0.0, allow_inf_nan=False)
    null_abs: float | None = Field(default=None, allow_inf_nan=False)
    # The shifted null carried by the procedure that produced this row. Exactly
    # one axis is meaningful for a given procedure; unavailable rows retain it.
    abs_diff: float | None = Field(default=None, allow_inf_nan=False)
    abs_se: float | None = Field(default=None, allow_inf_nan=False)
    abs_lb: float | None = Field(default=None, allow_inf_nan=False)
    abs_ub: float | None = Field(default=None, allow_inf_nan=False)
    abs_reference_kind: Literal["normal", "t"] | None = None
    abs_reference_df: float | None = Field(default=None, allow_inf_nan=False, gt=0)
    # Additive evidence for null_abs decisions. None when that sidecar is
    # unavailable; never reconstructed from the relative interval.
    policy_name: Literal["compiled_plan", "default_exploratory"] = "default_exploratory"
    # Output provenance for the lightweight standalone default versus the
    # compiled policy supplied by Analysis.

    n_treat: int | None = None
    n_control: int | None = None
    discovery: bool | None = None
    family_axes: tuple[str, ...] | None = None
    family_q: float | None = None
    family_threshold: float | None = None
    family_guarantee: Literal["finite_sample", "asymptotic_sequential"] | None = None
    family_nominal_alpha: float | None = None

    @field_serializer("ds", when_used="json")
    def _serialize_day_axis(self, value: date | datetime | str | int | float) -> object:
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

    _infer_reference: ClassVar[Any] = model_validator(mode="before")(
        _infer_reference_from_legacy_dof
    )
    _check_reference: ClassVar[Any] = model_validator(mode="after")(_validate_reference_fields)
    _check_absolute_reference: ClassVar[Any] = model_validator(mode="after")(
        _validate_absolute_reference_fields
    )

    @model_validator(mode="after")
    def _lift_or_unavailable(self):
        if self.sequential_result is not None and self.reference_kind != "sequential":
            from increment.sequential_state import sequential_refuse

            sequential_refuse("source.invalid", "fixed view cannot carry sequential evidence")
        if self.reference_kind == "sequential":
            from increment.sequential_state import sequential_refuse

            if self.sequential_result is None:
                sequential_refuse(
                    "continuation.legacy", "sequential view lost its exact checkpoint"
                )
            require_public_laws(
                (self.sequential_result.checkpoint.model,), "DailyLiftEstimate replay"
            )
            from increment.estimation.sequential_runtime import (
                display_estimate,
                require_selected_widening,
            )

            if self.lift != display_estimate(self.sequential_result):
                sequential_refuse(
                    "source.invalid", "view point or interval differs from its stopped state"
                )
            require_selected_widening(
                self.sequential_result,
                discovery=self.discovery,
                family_threshold=self.family_threshold,
                family_nominal_alpha=self.family_nominal_alpha,
            )
            _reject_sequential_mixed_authority(self)
            cp = self.sequential_result.checkpoint
            if (cp.model.law in ASYMPTOTIC_LAWS) != (self.inference == "asymptotic_mean"):
                sequential_refuse(
                    "source.invalid", "displayed validity regime differs from its checkpoint"
                )
            if (self.metric, self.group_id, self.estimand, self.alternative, self.null_lift) != (
                cp.cell.metric,
                cp.cell.group_id,
                cp.cell.estimand,
                cp.cell.alternative,
                float(cp.cell.null_lift),
            ):
                sequential_refuse(
                    "source.invalid", "view hypothesis differs from its certified checkpoint"
                )
            if self.unavailable is not None or self.binomial_set is not None:
                sequential_refuse(
                    "source.invalid",
                    "certified sequential abstention belongs in its confidence geometry",
                )
            return self
        if self.unavailable is not None:
            if (
                self.lift is not None
                or self.relative_confidence_set is not None
                or self.binomial_set is not None
                or self.relative_unavailable_reason is not None
            ):
                _refuse("breakout.daily_lift.exactly_one_unavailable")
            return self
        if _validate_joint_relative_row(self):
            return self
        if self.lift is None:
            if (
                self.reference_kind != "binomial"
                or self.binomial_set is None
                or self.binomial_set.point_available
            ):
                _refuse("breakout.daily_lift.exactly_one_unavailable")
        _validate_flat_binomial_row(
            self,
            refusal_code="breakout.daily_lift.exactly_one_unavailable",
        )
        return self

    def require_lift(self) -> Estimate:
        """Return the finite point estimate or refuse for an unavailable row."""
        if self.lift is None:
            _refuse("breakout.daily_lift.point_unavailable")
        return self.lift


class LiftEstimates(EstimateList[LiftEstimate]):
    """``list[LiftEstimate]`` with ``.to_frame()`` - :meth:`~increment.
    analysis.Analysis.run`'s return type."""

    _model = LiftEstimate


class BreakoutEstimates(EstimateList[BreakoutEstimate]):
    """``list[BreakoutEstimate]`` with ``.to_frame()`` - :func:`run_breakout`'s
    (and :meth:`~increment.analysis.Analysis.run_breakout`'s)
    return type."""

    _model = BreakoutEstimate


class DailyMetricValues(EstimateList[DailyMetricValue]):
    """``list[DailyMetricValue]`` with ``.to_frame()`` - :func:`run_daily`'s
    (and :meth:`~increment.analysis.Analysis.run_daily`/
    ``run_asof``'s) return type."""

    _model = DailyMetricValue


class DailyLiftEstimates(EstimateList[DailyLiftEstimate]):
    """``list[DailyLiftEstimate]`` with ``.to_frame()`` - :func:`run_daily_lift`'s
    (and :meth:`~increment.analysis.Analysis.run_daily_lift`/
    ``run_asof_lift``'s) return type."""

    _model = DailyLiftEstimate


def daily_sequential_projection(rows: Sequence[LiftEstimate]) -> DailyLiftEstimates:
    output = []
    for row in rows:
        result = row.require_sequential_result()
        if row.ds is None:
            sequential_refuse("source.invalid", "as-of checkpoint has no reveal label")
        output.append(
            DailyLiftEstimate(
                ds=row.ds,
                metric=row.metric,
                group_id=row.group_id,
                method=row.method,
                method_role=row.method_role,
                inference=row.inference,
                alternative=row.alternative,
                null_lift=row.null_lift,
                role=row.role,
                policy_name="compiled_plan",
                reference_kind="sequential",
                lift=row.lift,
                sequential_result=result,
                estimand=row.estimand,
                note=row.note,
                n_treat=result.checkpoint.treatment.n,
                n_control=result.checkpoint.control.n,
                discovery=row.discovery,
                family_axes=row.family_axes,
                family_q=row.family_q,
                family_threshold=row.family_threshold,
                family_guarantee=row.family_guarantee,
                family_nominal_alpha=row.family_nominal_alpha,
            )
        )
    return DailyLiftEstimates(output)


def _relative_meta_moments(row: BreakoutEstimate) -> tuple[float | None, float | None]:
    """Return log-risk-ratio moments for downstream Gaussian meta models.

    Exact binomial rows deliberately do not present a Normal standard error
    as part of their primary inference. Downstream heterogeneity and rollout
    models are themselves Gaussian approximations, however, so they may use
    the ordinary independent-binomial delta variance when both event counts
    are positive. Boundary rows remain unavailable to those models.
    """
    if row.value_scale != "relative":
        return None, None
    lift = row.lift
    if lift is None:
        return None, None
    if row.reference_kind != "binomial":
        return lift.log_mean, lift.log_se
    bset = row.binomial_set
    if bset is None or bset.x_c == 0 or bset.x_t == 0:
        return None, None
    log_mean = math.fsum(
        (math.log(bset.x_t), -math.log(bset.n_t), -math.log(bset.x_c), math.log(bset.n_c))
    )
    variance = math.fsum((1.0 / bset.x_t, -1.0 / bset.n_t, 1.0 / bset.x_c, -1.0 / bset.n_c))
    if not (variance > 0.0 and math.isfinite(variance)):
        return None, None
    return log_mean, math.sqrt(variance)


# Columns run_breakout reads directly once past the dimension check
# below; a missing one would otherwise surface as a bare KeyError.
_REQUIRED_BREAKOUT_COLUMNS = ["metric", "n", "ref_y", "cy1"]


class _DroppedRow(NamedTuple):
    """One row excluded before ``estimate_lift`` ever sees it, with why."""

    row: Mapping[str, Any]
    reason: ExclusionReason  # "few_units" | "nonpositive_mean"


class _MeanPartition(NamedTuple):
    """How a slice's rows split on the per-row estimability guard.

    Pure data - no policy. ``run_breakout`` turns this into warnings and
    a typed :data:`ExclusionReason`; ``run_daily_lift`` retains the dense
    row with ``lift=None`` and the explicit ``unavailable`` reason.
    """

    live: list[Mapping[str, Any]]
    dropped: list[_DroppedRow]
    lost_control_metrics: set[str]
    live_control_metrics: set[str]


def _mean_y(row: Mapping[str, Any]) -> float:
    """The row's outcome mean ``ref_y + cy1 / n``. Required on every row, so
    a missing/``None`` reference here is a malformed row, not an absent family.
    """
    return float(row["ref_y"]) + float(row["cy1"]) / float(row["n"])


def _mean_den(row: Mapping[str, Any]) -> float | None:
    """The row's ratio-denominator mean ``ref_den + cden1 / n``, or ``None``
    when the family is absent - as it is on every non-ratio row.
    """
    ref_den = row.get("ref_den")
    if ref_den is None:
        return None
    cden1 = row.get("cden1")
    return float(ref_den) + (0.0 if cden1 is None else float(cden1)) / float(row["n"])


def _is_absent(value: Any) -> bool:
    """Use dataframe-null semantics for every optional summary field."""
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _binomial_gate_exempt_metrics(
    metrics: Sequence[Metric],
    *,
    methods: list[Method] | None,
    methods_by_metric: Mapping[str, list[Method]] | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    prior: Prior | None = None,
    prior_by_metric: Mapping[str, Prior | None] | None = None,
) -> frozenset[str]:
    """Conversion/retention metrics exempt from the ddof/positive-mean gate
    below because the exact binomial risk-ratio method admits them at ``n=1``
    and with zero treatment/control means. The log-Normal delta method still
    needs a ddof=1 variance and ``math.log`` of a positive mean.

    This mirrors ``estimation.engine._binomial_eligible`` at this
    row-partitioning layer. Mixed requests are exempt when an unadjusted method
    can use the exact path; adjusted methods are partitioned per method.
    """
    exempt: set[str] = set()
    for metric in metrics:
        if metric.type not in ("conversion", "retention"):
            continue
        metric_prior = (prior_by_metric or {}).get(metric.name, prior)
        if inference is not None or metric_prior is not None:
            continue
        configured = (methods_by_metric or {}).get(metric.name, methods)
        configured = configured or [Method(name="unadjusted")]
        if any(method.variance_reduction != "cuped" for method in configured):
            exempt.add(metric.name)
    return frozenset(exempt)


def _row_x_slot_is_covariate(row: Mapping[str, Any]) -> bool:
    """Whether a row's materialized x slot is a CUPED covariate: a present ``x_role`` decides;
    a frame predating the column implied one only without the legacy clustered ``cxden``."""
    if _opt(row.get("ref_x")) is None:
        return False
    raw_role = row.get("x_role")
    role = None if _is_absent(raw_role) else raw_role
    if role == X_SLOT_ROLES["x"]:
        return True
    return "x_role" not in row and _opt(row.get("cxden")) is None


def _row_has_independent_binomial_family(row: Mapping[str, Any]) -> bool:
    """Whether a summary row lacks denominator or dependent-cluster coordinates."""
    if _opt(row.get("ref_den")) is not None:
        return False
    return _opt(row.get("ref_x")) is None or _row_x_slot_is_covariate(row)


def _partition_nonpositive_mean_rows(
    rows: list[Mapping[str, Any]],
    *,
    control_group: str,
    metric_types: Mapping[str, str],
    binomial_gate_exempt_metrics: frozenset[str],
) -> _MeanPartition:
    """Split *rows* on a per-row, family-dependent estimability gate.

    Drop rows with ``n < 2`` or a nonpositive outcome/denominator mean when
    log-scale variance models require those quantities. Exact binomial rows
    remain eligible at ``n=1`` and with zero means. Mixed adjusted/unadjusted
    requests receive a method-specific partition; this function only identifies
    excluded rows.

    ``lost_control_metrics`` is every metric with a control row among
    *rows* but not among ``live``; ``live_control_metrics`` is every
    metric that still has one.
    """
    live: list[Mapping[str, Any]] = []
    dropped: list[_DroppedRow] = []
    for row in rows:
        metric_name = str(row["metric"])
        if metric_name in binomial_gate_exempt_metrics and _row_has_independent_binomial_family(
            row
        ):
            live.append(row)
            continue
        if int(row["n"]) < 2:
            dropped.append(_DroppedRow(row, "few_units"))
            continue
        if _mean_y(row) <= 0.0:
            dropped.append(_DroppedRow(row, "nonpositive_mean"))
            continue
        if metric_types.get(metric_name) == "ratio":
            mean_den = _mean_den(row)
            if mean_den is None or mean_den <= 0.0:
                dropped.append(_DroppedRow(row, "nonpositive_mean"))
                continue
        live.append(row)

    original_control_metrics = {
        str(row["metric"]) for row in rows if str(row.get("group_id")) == control_group
    }
    live_control_metrics = {
        str(row["metric"]) for row in live if str(row.get("group_id")) == control_group
    }
    return _MeanPartition(
        live=live,
        dropped=dropped,
        lost_control_metrics=original_control_metrics - live_control_metrics,
        live_control_metrics=live_control_metrics,
    )


def _warn_nonpositive_mean_partition(part: _MeanPartition, *, label: str) -> None:
    """Emit ``run_breakout``'s per-row and per-metric drop warnings.

    Called only by ``run_breakout``; ``run_daily_lift`` stays silent
    and returns unavailable rows instead. ``stacklevel=4`` points at
    user's own call site.
    """
    for dropped in part.dropped:
        row = dropped.row
        if dropped.reason == "few_units":
            _warn(
                "breakout.estimates.row_few_units",
                label=label,
                metric=row.get("metric"),
                group_id=row.get("group_id"),
                n=int(row["n"]),
                stacklevel=4,
            )
        elif _mean_y(row) <= 0.0:
            _warn(
                "breakout.estimates.row_nonpositive_mean",
                label=label,
                metric=row.get("metric"),
                group_id=row.get("group_id"),
                mean=_mean_y(row),
                stacklevel=4,
            )
        else:
            # Only reachable via the ratio-metric denominator guard: the
            # outcome mean is fine, so name the denominator mean instead.
            mean_den = _mean_den(row)
            den_desc = "missing/None" if mean_den is None else str(mean_den)
            _warn(
                "breakout.estimates.row_nonpositive_denominator_mean",
                label=label,
                metric=row.get("metric"),
                group_id=row.get("group_id"),
                den_desc=den_desc,
                stacklevel=4,
            )
    if part.live_control_metrics:
        # Only accurate when the caller proceeds to call estimate_lift on
        # `live`; an empty live_control_metrics skips the slice instead.
        for metric_name in sorted(part.lost_control_metrics):
            _warn(
                "breakout.estimates.lost_control_metric",
                label=label,
                metric_name=metric_name,
                stacklevel=4,
            )


def _estimate_lift_or_reason(
    *,
    metrics: Sequence[Metric],
    summary: list[Mapping[str, Any]],
    control_group: str,
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str = "two-sided",
    null_lift: float = 0.0,
    null_abs: float | None = None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
) -> tuple[list[LiftEstimate] | None, str | None, Mapping[Any, Any]]:
    """Call ``estimate_lift`` and retain keyed failures for dense callers.

    Catches lift guards and returns their keyed failures rather than dropping
    otherwise estimable sibling arms.
    """

    try:
        metrics_by_name = {m.name: m for m in metrics}
        computation = estimate_lift(
            metrics=metrics,
            summary=summary,
            control_group=control_group,
            methods=methods,
            prior=prior,
            alpha=alpha,
            alternative=alternative,
            null_lift=null_lift,
            null_abs=null_abs,
            inference=inference,
            method_roles=method_roles,
        )
        results = [
            r.model_copy(
                update={
                    "preferred_direction": metrics_by_name[r.metric].declared_preferred_direction
                }
            )
            for r in computation.results
        ]
        if computation.failures and not results:
            reason = next(iter(computation.failures.values())).display()
            return None, reason, computation.failures
        return results, None, computation.failures
    except LiftGuardError as exc:
        from increment.decision import ArmHypothesisKey, DecisionFailure

        failures: dict[Any, Any] = {}
        groups = {str(row["group_id"]) for row in summary if str(row["group_id"]) != control_group}
        for metric in metrics:
            for group_id in groups:
                hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
                failures[hypothesis] = DecisionFailure(
                    hypothesis,
                    "estimation.engine.lift_guard",
                    {
                        "metric": metric.name,
                        "group_id": group_id,
                        "reason": exc.reason,
                        "display": str(exc),
                    },
                )
        return None, str(exc), failures


def _rekey_segment_computation(
    bundle: DecisionComputation[LiftEstimate],
    *,
    dimension: str,
    dimension_value: str,
) -> DecisionComputation[LiftEstimate]:
    from increment.decision import (
        DecisionComputation,
        SegmentHypothesisKey,
    )

    def key_for(hypothesis: Any) -> SegmentHypothesisKey:
        return SegmentHypothesisKey(
            hypothesis.metric,
            hypothesis.group_id,
            hypothesis.estimand,
            dimension,
            dimension_value,
        )

    evidence: dict[Any, Any] = {}
    for value in bundle.evidence.values():
        hypothesis = key_for(value.hypothesis)
        evidence[hypothesis] = replace(value, hypothesis=hypothesis)
    failures: dict[Any, Any] = {
        key_for(value.hypothesis): type(value)(key_for(value.hypothesis), value.code, value.context)
        for value in bundle.failures.values()
    }
    return DecisionComputation(
        results=bundle.results,
        evidence=evidence,
        failures=failures,
        sequential_snapshot=bundle.sequential_snapshot,
    )


def _estimate_lift_computation_or_reason(
    *,
    metric: Metric,
    summary: list[Mapping[str, Any]],
    control_group: str,
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
) -> tuple[DecisionComputation[LiftEstimate] | None, str | None]:
    try:
        computation = estimate_lift(
            metrics=[metric],
            summary=summary,
            control_group=control_group,
            methods=methods,
            prior=prior,
            alpha=alpha,
            alternative=alternative,
            inference=inference,
            method_roles=method_roles,
        )
        from increment.decision import DecisionComputation

        rows = tuple(
            result.model_copy(update={"preferred_direction": metric.declared_preferred_direction})
            for result in computation.results
        )
        return DecisionComputation(
            results=rows,
            evidence=computation.evidence,
            failures=computation.failures,
        ), None
    except LiftGuardError as exc:
        from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

        failures: dict[Any, Any] = {}
        groups = {str(row["group_id"]) for row in summary if str(row["group_id"]) != control_group}
        for group_id in groups:
            hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
            failures[hypothesis] = DecisionFailure(
                hypothesis,
                "estimation.engine.lift_guard",
                {
                    "metric": metric.name,
                    "group_id": group_id,
                    "reason": exc.reason,
                    "display": str(exc),
                },
            )
        return DecisionComputation(results=(), evidence={}, failures=failures), str(exc)


def _estimate_encouragement_or_reason(
    *,
    metrics: Sequence[Metric],
    summary: list[Mapping[str, Any]],
    design: Encouragement,
    estimands: Sequence[str],
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str = "two-sided",
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
) -> tuple[list[LiftEstimate], str | None]:
    """Call ``estimate_encouragement`` for *estimands*, reporting why a
    slice's compliance/late row(s) came back missing.

    A weak first stage omits a ``late`` row without raising (an
    explanatory ``compliance`` row is still emitted); this helper only
    catches the case where the whole slice lacks *design.control_group*
    entirely. Every other ``ValueError`` propagates unchanged.

    Returns
    -------
    tuple[list[LiftEstimate], str | None]
        ``(results, None)`` on success, or ``([], reason)`` when the control arm is missing.
    """
    try:
        results = list(
            estimate_encouragement(
                metrics,
                summary,
                design,
                estimands=estimands,
                methods=methods,
                prior=prior,
                alpha=alpha,
                alternative=alternative,
                inference=inference,
                method_roles=method_roles,
            ).results
        )
    except InvalidRequestError as exc:
        if exc.code != "estimation.encouragement.control.missing":
            raise
        return [], str(exc)
    return results, None


# Cohesive encouragement row assembly; splitting this path is deferred.
def _encouragement_rows_for_slice(  # noqa: PLR0913
    day_rows: list[Mapping[str, Any]],
    *,
    metric: Metric,
    design: Encouragement,
    estimands: Sequence[str],
    methods: list[Method] | None,
    prior: Prior | None,
    policy: _DailyCellPolicy,
    ds_value: date | datetime | str | int | float,
    ds_basis: Literal["calendar", "cohort"],
    dimension: str | None,
    dim_value: str | None,
    source: str | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    reliability_floor: int,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
) -> list[DailyLiftEstimate]:
    """One day-slice's compliance/LATE rows for one outcome metric."""
    if methods == [] or not estimands:
        return []
    estimates, _ = _estimate_encouragement_or_reason(
        metrics=[metric],
        summary=day_rows,
        design=design,
        estimands=estimands,
        methods=methods,
        prior=prior,
        alpha=policy.alpha,
        alternative=policy.alternative,
        inference=inference,
        method_roles=method_roles,
    )
    control_n = next(
        (float(row["n"]) for row in day_rows if str(row["group_id"]) == design.control_group),
        None,
    )
    arm_n = {str(row["group_id"]): float(row["n"]) for row in day_rows}
    rows: list[DailyLiftEstimate] = []
    for estimate in estimates:
        treatment_n = arm_n.get(estimate.group_id)
        below_floor = (treatment_n is not None and treatment_n < reliability_floor) or (
            control_n is not None and control_n < reliability_floor
        )
        result_metric = (
            f"{metric.name}_uptake" if estimate.estimand == "compliance" else estimate.metric
        )
        rows.append(
            DailyLiftEstimate(
                **_copy_common_fields(
                    estimate,
                    metric=result_metric,
                    reference_kind=_slice_reference_kind(estimate),
                    role=policy.role,
                    null_lift=policy.null_lift,
                    null_abs=policy.null_abs,
                    policy_name=policy.policy_name,
                    ds=ds_value,
                    ds_basis=ds_basis,
                    dimension=dimension,
                    dimension_value=dim_value,
                    source=source,
                    low_reliability=below_floor,
                    # The encouragement estimation pass produces no
                    # sequential checkpoint or family-correction metadata.
                    sequential_result=None,
                    binomial_set=None,
                    family_axes=None,
                    family_q=None,
                    family_threshold=None,
                    family_guarantee=None,
                    family_nominal_alpha=None,
                )
            )
        )
    return rows


def _slice_reference_kind(
    estimate: LiftEstimate,
) -> Literal["normal", "t", "sequential", "binomial"]:
    if estimate.reference_kind == "confidence_set":
        from increment.winsor import winsor_refuse

        winsor_refuse(
            "design_unsupported", "Pooled winsor confidence sets require the complete unsplit pool."
        )
    return estimate.reference_kind


def _breakout_estimate_row(
    lift_estimate: LiftEstimate,
    *,
    dimension: str,
    dimension_value: str,
    source: str | None,
    n_treat: float | None,
    n_control: float | None,
    low_reliability: bool,
    role: str | None = "exploratory",
    discovery: bool | None = None,
    family_axes: tuple[str, ...] | None = None,
    family_q: float | None = None,
    family_threshold: float | None = None,
) -> BreakoutEstimate:
    """Build one real ``BreakoutEstimate`` row from a
    ``LiftEstimate`` plus this call's breakout-specific context (which
    never changes between passes for the same cell). Shared by
    ``run_breakout``'s nominal per-segment pass, its ``"bh"`` branch's
    FCR re-estimation pass, and readouts' encouragement branch, so every
    real row carries exactly the same field set. ``estimand``/
    ``value_scale``/``note`` and a registered sequential family's
    ``family_guarantee``/``family_nominal_alpha`` come from the
    ``LiftEstimate`` itself (model defaults ``"itt"``/``"relative"``/``None``
    on the randomized path, which never sets them).
    """
    return BreakoutEstimate(
        **_copy_common_fields(
            lift_estimate,
            reference_kind=_slice_reference_kind(lift_estimate),
            dimension=dimension,
            dimension_value=dimension_value,
            source=source,
            # BreakoutEstimate-only sidecar: _copy_common_fields doesn't
            # carry it (DailyLiftEstimate has no winsor_* fields).
            winsor_lower_percentile=lift_estimate.winsor_lower_percentile,
            winsor_upper_percentile=lift_estimate.winsor_upper_percentile,
            winsor_lower_bound=lift_estimate.winsor_lower_bound,
            winsor_upper_bound=lift_estimate.winsor_upper_bound,
            winsor_control_n=lift_estimate.winsor_control_n,
            winsor_control_n_lower=lift_estimate.winsor_control_n_lower,
            winsor_control_n_upper=lift_estimate.winsor_control_n_upper,
            winsor_treatment_n=lift_estimate.winsor_treatment_n,
            winsor_treatment_n_lower=lift_estimate.winsor_treatment_n_lower,
            winsor_treatment_n_upper=lift_estimate.winsor_treatment_n_upper,
            low_reliability=low_reliability,
            n_treat=n_treat,
            n_control=n_control,
            role=role,
            discovery=discovery,
            family_axes=family_axes,
            family_q=family_q,
            family_threshold=family_threshold,
        )
    )


# Per-arm unit floor under which a real estimate is flagged low_reliability;
# shared by run_breakout, run_daily_lift, and readouts' encouragement branch.
DEFAULT_RELIABILITY_FLOOR = 50


def _derive_method_roles(
    methods: Sequence[Method] | None,
    roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
) -> dict[str, Literal["decision", "sensitivity"]]:
    from increment.estimation.engine import resolve_method_roles

    return resolve_method_roles(list(methods or [Method(name="unadjusted")]), override=roles)


def _configured_methods_for_metric(
    metric_name: str,
    *,
    methods: list[Method] | None,
    methods_by_metric: Mapping[str, list[Method]] | None,
) -> list[Method]:
    if methods_by_metric is not None:
        return list(methods_by_metric.get(metric_name, []))
    return list(methods or [Method(name="unadjusted")])


def _mixed_binomial_method_groups(
    metric: Metric,
    configured: list[Method],
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Prior | None,
) -> tuple[list[Method], list[Method]] | None:
    """Return exact and legacy lanes for a mixed binary-metric request."""
    if metric.type not in ("conversion", "retention"):
        return None
    if inference is not None or prior is not None:
        return None
    exact = [method for method in configured if method.variance_reduction != "cuped"]
    legacy = [method for method in configured if method.variance_reduction == "cuped"]
    return (exact, legacy) if exact and legacy else None


_BINOMIAL_UNUSED_ADJUSTMENT_FIELDS = frozenset(
    {"ref_x", "cx1", "cx2", "cxy", "x_role", "cxden", "cxd"}
)

#: Slots a per-day metric value reads: the unmasked families. The uptake
#: mask is an encouragement first-stage input, never a per-day value.
_DAY_VALUE_SLOTS = tuple(slot for slot in OPTIONAL_SLOTS if SLOTS[slot].mask is None)


def _unadjusted_binomial_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Project mixed-request rows onto the unadjusted Bernoulli family."""
    projected: list[Mapping[str, Any]] = []
    for row in rows:
        if _row_x_slot_is_covariate(row):
            projected.append(
                {
                    key: value
                    for key, value in row.items()
                    if key not in _BINOMIAL_UNUSED_ADJUSTMENT_FIELDS
                }
            )
        else:
            projected.append(row)
    return projected


class _PreparedLiftSlice(NamedTuple):
    live_rows: list[Mapping[str, Any]]
    control_n: dict[str, float]
    arm_n: dict[tuple[str, str], float]
    unestimable: dict[tuple[str, str], ExclusionReason]
    route2_pairs: set[tuple[str, str]]
    part: _MeanPartition


class _EstimatedCell(NamedTuple):
    metric: str
    group_id: str
    estimate: LiftEstimate


class _UnavailableCell(NamedTuple):
    metric: str
    group_id: str
    method: Method
    method_role: Literal["decision", "sensitivity"]
    reason: ExclusionReason


_BreakoutCellOutcome = _EstimatedCell | _UnavailableCell


class _BreakoutMetricPass(NamedTuple):
    outcomes: tuple[_BreakoutCellOutcome, ...]
    computations: dict[tuple[str, str], DecisionComputation]
    metric_rows: dict[tuple[str, str], list[Mapping[str, Any]]]


class _BreakoutContext(NamedTuple):
    metrics: Sequence[Metric]
    correction: Correction
    segment_value: str
    dimension: str
    control_group: str
    source: str | None
    methods: list[Method] | None
    methods_by_metric: Mapping[str, list[Method]] | None
    prior: Prior | None
    alpha: float
    alternative: str
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None
    method_roles_by_metric: Mapping[str, Mapping[str, Literal["decision", "sensitivity"]]] | None
    warned_open_ended: set[str]
    rewarn_registry: dict[Any, int]
    all_pairs: set[tuple[str, str]]
    reliability_floor: int
    resolved_methods: list[Method]


class _BreakoutRowsPass(NamedTuple):
    rows: list[BreakoutEstimate]
    family_entries: list[tuple[str, LiftEstimate, int]]


def _partition_breakout_summary(
    rows: Iterable[Mapping[str, Any]],
    *,
    dimension: str,
    control_group: str,
    methods_by_metric: Mapping[str, list[Method]] | None,
) -> tuple[dict[str, list[Mapping[str, Any]]], set[tuple[str, str]]]:
    """Partition breakout rows and validate each segment/metric/arm cell."""
    segments: dict[str, list[Mapping[str, Any]]] = {}
    all_pairs: set[tuple[str, str]] = set()
    seen_cells: set[tuple[str, str, str]] = set()
    for row in rows:
        value = str(row[dimension])
        segments.setdefault(value, []).append(row)
        group_id = str(row.get("group_id"))
        cell = (value, str(row["metric"]), group_id)
        if cell in seen_cells:
            _refuse(
                "breakout.run_breakout_duplicate",
                dimension=dimension,
                value=value,
                metric=row["metric"],
                group_id=group_id,
            )
        seen_cells.add(cell)
        if group_id != control_group and (
            methods_by_metric is None or methods_by_metric.get(str(row["metric"]), [])
        ):
            all_pairs.add((str(row["metric"]), group_id))
    return segments, all_pairs


def _prepare_lift_slice(
    rows: list[Mapping[str, Any]],
    *,
    control_group: str,
    metric_types: Mapping[str, str],
    binomial_gate_exempt_metrics: frozenset[str],
    warning_label: str | None = None,
) -> _PreparedLiftSlice:
    """Apply shared row/control-arm gates for one breakout or day slice."""
    part = _partition_nonpositive_mean_rows(
        rows,
        control_group=control_group,
        metric_types=metric_types,
        binomial_gate_exempt_metrics=binomial_gate_exempt_metrics,
    )
    if warning_label is not None:
        _warn_nonpositive_mean_partition(part, label=warning_label)
    control_n: dict[str, float] = {}
    arm_n: dict[tuple[str, str], float] = {}
    for row in part.live:
        row_metric = str(row["metric"])
        row_group = str(row.get("group_id"))
        arm_n[(row_metric, row_group)] = float(row["n"])
        if row_group == control_group:
            control_n[row_metric] = float(row["n"])
    unestimable: dict[tuple[str, str], ExclusionReason] = {}
    for dropped in part.dropped:
        group_id = str(dropped.row.get("group_id"))
        if group_id != control_group:
            unestimable[(str(dropped.row["metric"]), group_id)] = dropped.reason
    route2_pairs: set[tuple[str, str]] = set()
    for row in rows:
        group_id = str(row.get("group_id"))
        metric_name = str(row["metric"])
        if group_id != control_group and metric_name not in part.live_control_metrics:
            pair = (metric_name, group_id)
            if pair not in unestimable and metric_name not in part.lost_control_metrics:
                route2_pairs.add(pair)
            unestimable.setdefault(pair, "no_control_arm")
    if not part.live_control_metrics:
        for row in rows:
            group_id = str(row.get("group_id"))
            if group_id != control_group:
                unestimable.setdefault((str(row["metric"]), group_id), "no_control_arm")
        if warning_label is not None:
            if part.lost_control_metrics:
                _warn(
                    "breakout.estimates.slice_no_live_control_dropped",
                    warning_label=warning_label,
                    control_group=control_group,
                    stacklevel=3,
                )
            else:
                _warn(
                    "breakout.estimates.slice_no_control_arm",
                    warning_label=warning_label,
                    control_group=control_group,
                    stacklevel=3,
                )
    return _PreparedLiftSlice(part.live, control_n, arm_n, unestimable, route2_pairs, part)


def _finalize_breakout_metric_warnings(
    caught: Sequence[warnings.WarningMessage],
    skip_reasons: Mapping[str, str],
    skip_failures: Mapping[str, Any],
    prepared: _PreparedLiftSlice,
    context: _BreakoutContext,
    unestimable: dict[tuple[str, str], ExclusionReason],
    failures: dict[tuple[str, str], Any],
) -> dict[tuple[str, str], ExclusionReason]:
    """Replay metric warnings and map guarded metrics to exclusions."""
    for warning in caught:
        message = str(warning.message)
        if message.startswith("metric '") and "is open-ended" in message:
            metric_name = message.split("'", 2)[1]
            if metric_name in context.warned_open_ended:
                continue
            context.warned_open_ended.add(metric_name)
        warnings.warn_explicit(
            warning.message,
            warning.category,
            warning.filename,
            warning.lineno,
            registry=context.rewarn_registry,
        )
    for metric_name in sorted(skip_reasons):
        skip_reason = skip_reasons[metric_name]
        _warn(
            "breakout.estimates.metric_guarded_excluded",
            dimension=context.dimension,
            segment_value=context.segment_value,
            metric_name=metric_name,
            skip_reason=skip_reason,
            stacklevel=4,
        )
        failure = skip_failures.get(metric_name)
        reason = _failure_exclusion_reason(failure) if failure is not None else "estimation_failed"
        for row in prepared.live_rows:
            group_id = str(row.get("group_id"))
            if group_id != context.control_group and str(row["metric"]) == metric_name:
                pair = (metric_name, group_id)
                unestimable.setdefault(pair, reason)
                if failure is not None and failure.code != "estimation.engine.lift_guard":
                    failures.setdefault(pair, failure)
    return unestimable


def _failed_pairs(failures: Mapping[Any, Any]) -> list[tuple[tuple[str, str], Any]]:
    """(metric, arm) pair of each failure, taken from its hypothesis key; failure
    contexts come from the raising estimator and need not name the pair."""
    return [
        ((str(hypothesis.metric), str(hypothesis.group_id)), failure)
        for hypothesis, failure in failures.items()
    ]


def _record_failed_pairs(
    failures: Mapping[Any, Any], unestimable: dict[tuple[str, str], ExclusionReason]
) -> None:
    """Label each failed (metric, arm) pair even when other arms of the metric estimated."""
    for pair, failure in _failed_pairs(failures):
        unestimable.setdefault(pair, _failure_exclusion_reason(failure))


def _keep_non_guard_failures(failures: Mapping[Any, Any], kept: dict[tuple[str, str], Any]) -> None:
    """Record failures that are not lift guards so the breakout computation reports
    them under their own code instead of an exclusion label."""
    for pair, failure in _failed_pairs(failures):
        if failure.code != "estimation.engine.lift_guard":
            kept.setdefault(pair, failure)


def _breakout_unavailable_computation(
    unestimable: Mapping[tuple[str, str], ExclusionReason],
    context: _BreakoutContext,
    original_failures: Mapping[tuple[str, str], Any] | None = None,
) -> DecisionComputation:
    from increment.decision import DecisionComputation, DecisionFailure, SegmentHypothesisKey

    failures: dict[Any, Any] = {}
    for (metric_name, group_id), reason in sorted(unestimable.items()):
        hypothesis = SegmentHypothesisKey(
            metric_name, group_id, "itt", context.dimension, context.segment_value
        )
        original = (original_failures or {}).get((metric_name, group_id))
        if original is not None:
            failure_context = dict(original.context)
            failure_context.update(
                {
                    "metric": metric_name,
                    "group_id": group_id,
                    "dimension": context.dimension,
                    "dimension_value": context.segment_value,
                }
            )
            failures[hypothesis] = DecisionFailure(hypothesis, original.code, failure_context)
        else:
            failures[hypothesis] = DecisionFailure(
                hypothesis,
                f"breakout.{reason}",
                {
                    "metric": metric_name,
                    "group_id": group_id,
                    "dimension": context.dimension,
                    "dimension_value": context.segment_value,
                    "reason": reason,
                },
            )
    return DecisionComputation(results=(), evidence={}, failures=failures)


def _breakout_cell_outcomes(
    estimates: Sequence[LiftEstimate],
    unestimable: Mapping[tuple[str, str], ExclusionReason],
    method_unestimable: Mapping[tuple[str, str, str], ExclusionReason],
    context: _BreakoutContext,
) -> tuple[_BreakoutCellOutcome, ...]:
    """Create one immutable result-or-reason outcome for every requested cell."""
    successful = {(estimate.metric, estimate.group_id, estimate.method) for estimate in estimates}
    outcomes: list[_BreakoutCellOutcome] = [
        _EstimatedCell(estimate.metric, estimate.group_id, estimate) for estimate in estimates
    ]
    for (metric_name, group_id, method_name), reason in sorted(method_unestimable.items()):
        configured = _configured_methods_for_metric(
            metric_name,
            methods=context.methods,
            methods_by_metric=context.methods_by_metric,
        )
        roles = (context.method_roles_by_metric or {}).get(metric_name, context.method_roles)
        role_map = _derive_method_roles(configured, roles)
        method = next(method for method in configured if method.name == method_name)
        if (metric_name, group_id, method_name) not in successful:
            outcomes.append(
                _UnavailableCell(
                    metric_name,
                    group_id,
                    method,
                    role_map.get(method_name, "decision"),
                    reason,
                )
            )
    for (metric_name, group_id), reason in sorted(unestimable.items()):
        configured = _configured_methods_for_metric(
            metric_name,
            methods=context.methods,
            methods_by_metric=context.methods_by_metric,
        )
        roles = (context.method_roles_by_metric or {}).get(metric_name, context.method_roles)
        role_map = _derive_method_roles(configured, roles)
        outcomes.extend(
            _UnavailableCell(
                metric_name, group_id, method, role_map.get(method.name, "decision"), reason
            )
            for method in configured
            # A method-specific exclusion already produced this method's row.
            if (metric_name, group_id, method.name) not in successful
            and (metric_name, group_id, method.name) not in method_unestimable
        )
    return tuple(outcomes)


def _warn_breakout_route2_pairs(
    prepared: _PreparedLiftSlice,
    context: _BreakoutContext,
) -> None:
    """Warn for live treatment rows lacking a metric's control row."""
    if not prepared.part.live_control_metrics:
        return
    for metric_name, group_id in sorted(prepared.route2_pairs):
        _warn(
            "breakout.estimates.route2_no_live_control_row",
            dimension=context.dimension,
            segment_value=context.segment_value,
            metric_name=metric_name,
            control_group=context.control_group,
            group_id=group_id,
            stacklevel=4,
        )


def _estimate_breakout_slice_metrics(  # noqa: PLR0915
    prepared: _PreparedLiftSlice,
    context: _BreakoutContext,
) -> _BreakoutMetricPass:
    """Estimate each live breakout metric and retain immutable cell outcomes."""
    declared_names = {m.name for m in context.metrics}
    unknown_metrics = {str(r["metric"]) for r in prepared.live_rows} - declared_names
    if unknown_metrics:
        _refuse(
            "breakout.metric_found_group",
            unknown_metrics=sorted(unknown_metrics),
            declared_names=sorted(declared_names),
        )
    unestimable = dict(prepared.unestimable)
    _warn_breakout_route2_pairs(prepared, context)
    estimates: list[LiftEstimate] = []
    method_unestimable: dict[tuple[str, str, str], ExclusionReason] = {}
    computations: dict[tuple[str, str], DecisionComputation] = {}
    metric_rows: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    skip_reason_by_metric: dict[str, str] = {}
    skip_failure_by_metric: dict[str, Any] = {}
    original_failures: dict[tuple[str, str], Any] = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for metric in context.metrics:
            if context.methods_by_metric is not None and not context.methods_by_metric.get(
                metric.name, []
            ):
                continue
            if metric.name not in prepared.part.live_control_metrics:
                continue
            rows = [r for r in prepared.live_rows if str(r["metric"]) == metric.name]
            if not rows:
                continue
            if context.correction == "bh":
                metric_rows[(context.segment_value, metric.name)] = rows
            configured = _configured_methods_for_metric(
                metric.name,
                methods=context.methods,
                methods_by_metric=context.methods_by_metric,
            )
            requested_roles = (context.method_roles_by_metric or {}).get(
                metric.name, context.method_roles
            )
            resolved_roles = _derive_method_roles(configured, requested_roles)
            mixed_groups = _mixed_binomial_method_groups(
                metric,
                configured,
                inference=context.inference,
                prior=context.prior,
            )
            groups = list(mixed_groups) if mixed_groups is not None else [configured]
            for group_index, method_group in enumerate(groups):
                method_rows = rows
                if mixed_groups is not None and group_index == 0:
                    method_rows = _unadjusted_binomial_rows(rows)
                elif mixed_groups is not None:
                    legacy_prepared = _prepare_lift_slice(
                        rows,
                        control_group=context.control_group,
                        metric_types={metric.name: metric.type},
                        binomial_gate_exempt_metrics=frozenset(),
                        warning_label=(
                            f"run_breakout: segment {context.dimension}="
                            f"{context.segment_value!r} method={method_group[0].name!r}"
                        ),
                    )
                    method_rows = legacy_prepared.live_rows
                    for pair, reason in legacy_prepared.unestimable.items():
                        for method in method_group:
                            method_unestimable[(*pair, method.name)] = reason
                    if not legacy_prepared.part.live_control_metrics:
                        continue
                computation, skip_reason = _estimate_lift_computation_or_reason(
                    metric=metric,
                    summary=method_rows,
                    control_group=context.control_group,
                    prior=context.prior,
                    alpha=context.alpha,
                    alternative=context.alternative,
                    methods=method_group,
                    inference=context.inference,
                    method_roles=resolved_roles,
                )
                if computation is not None and computation.failures:
                    _keep_non_guard_failures(computation.failures, original_failures)
                    _record_failed_pairs(computation.failures, unestimable)
                    if not computation.results:
                        failure = next(iter(computation.failures.values()))
                        skip_reason = failure.display()
                        skip_failure_by_metric[metric.name] = failure
                if computation is not None and (
                    (context.segment_value, metric.name) not in computations
                    or any(result.method_role == "decision" for result in computation.results)
                ):
                    computations[(context.segment_value, metric.name)] = _rekey_segment_computation(
                        computation,
                        dimension=context.dimension,
                        dimension_value=context.segment_value,
                    )
                if skip_reason is not None:
                    skip_reason_by_metric[metric.name] = skip_reason
                elif computation is not None:
                    estimates.extend(computation.results)
    unestimable = _finalize_breakout_metric_warnings(
        caught,
        skip_reason_by_metric,
        skip_failure_by_metric,
        prepared,
        context,
        unestimable,
        original_failures,
    )
    produced = {(estimate.metric, estimate.group_id) for estimate in estimates}
    for pair in context.all_pairs - produced:
        if pair not in unestimable:
            metric_name, group_id = pair
            if prepared.part.live_control_metrics:
                _warn(
                    "breakout.estimates.no_row_for_group_metric",
                    dimension=context.dimension,
                    segment_value=context.segment_value,
                    group_id=group_id,
                    metric_name=metric_name,
                    stacklevel=3,
                )
            unestimable[pair] = "no_control_arm"
    if unestimable:
        computations[(context.segment_value, "__unavailable__")] = (
            _breakout_unavailable_computation(unestimable, context, original_failures)
        )
    return _BreakoutMetricPass(
        _breakout_cell_outcomes(estimates, unestimable, method_unestimable, context),
        computations,
        metric_rows,
    )


def _append_breakout_slice_rows(
    prepared: _PreparedLiftSlice,
    context: _BreakoutContext,
    metric_pass: _BreakoutMetricPass,
    result_start: int,
) -> _BreakoutRowsPass:
    """Project immutable cell outcomes into one segment's output rows."""
    rows: list[BreakoutEstimate] = []
    family_entries: list[tuple[str, LiftEstimate, int]] = []
    for outcome in metric_pass.outcomes:
        if isinstance(outcome, _EstimatedCell):
            estimate = outcome.estimate
            treatment_n = prepared.arm_n.get((estimate.metric, estimate.group_id))
            control_n = prepared.control_n.get(estimate.metric)
            rows.append(
                _breakout_estimate_row(
                    estimate,
                    dimension=context.dimension,
                    dimension_value=context.segment_value,
                    source=context.source,
                    n_treat=treatment_n,
                    n_control=control_n,
                    low_reliability=(
                        treatment_n is not None and treatment_n < context.reliability_floor
                    )
                    or (control_n is not None and control_n < context.reliability_floor),
                )
            )
            if context.correction == "bh":
                family_entries.append(
                    (context.segment_value, estimate, result_start + len(rows) - 1)
                )
            continue
        rows.append(
            BreakoutEstimate(
                metric=outcome.metric,
                group_id=outcome.group_id,
                method=outcome.method.name,
                method_role=outcome.method_role,
                dimension=context.dimension,
                dimension_value=context.segment_value,
                source=context.source,
                alternative=context.alternative,
                lift=None,
                excluded=outcome.reason,
                role="exploratory",
            )
        )
    return _BreakoutRowsPass(rows, family_entries)


class _BreakoutFamilyContext(NamedTuple):
    segments: dict[str, list[Mapping[str, Any]]]
    all_pairs: set[tuple[str, str]]
    family_entries: list[tuple[str, LiftEstimate, int]]
    metric_rows_by_segment_metric: dict[tuple[str, str], list[Mapping[str, Any]]]
    segment_computations: dict[tuple[str, str], DecisionComputation[LiftEstimate]]
    metrics: Sequence[Metric]
    correction: Correction
    q: float
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None
    alpha: float
    control_group: str
    dimension: str
    methods: list[Method] | None
    methods_by_metric: Mapping[str, list[Method]] | None
    prior: Prior | None
    alternative: str
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None
    method_roles_by_metric: Mapping[str, Mapping[str, Literal["decision", "sensitivity"]]] | None


def _apply_breakout_family_correction(
    results: list[BreakoutEstimate],
    context: _BreakoutFamilyContext,
) -> list[BreakoutEstimate]:
    """Select the flat breakout family and re-estimate selected cells."""
    segments = context.segments
    all_pairs = context.all_pairs
    family_entries = context.family_entries
    metric_rows_by_segment_metric = context.metric_rows_by_segment_metric
    segment_computations = context.segment_computations
    metrics = context.metrics
    correction = context.correction
    q = context.q
    inference = context.inference
    alpha = context.alpha
    control_group = context.control_group
    dimension = context.dimension
    methods = context.methods
    methods_by_metric = context.methods_by_metric
    prior = context.prior
    method_roles = context.method_roles
    alternative = context.alternative
    method_roles_by_metric = context.method_roles_by_metric
    from increment.decision import SegmentHypothesisKey

    family_keys = [
        SegmentHypothesisKey(metric_name, group_id, "itt", dimension, segment_value)
        for segment_value in sorted(segments)
        for metric_name, group_id in sorted(all_pairs)
        if metric_name in {str(row["metric"]) for row in segments[segment_value]}
    ]
    if correction != "bh" or not family_keys:
        return results
    canonical_rows = {
        (segment_value, estimate.metric, estimate.group_id): estimate
        for segment_value, estimate, _idx in family_entries
        if estimate.method_role == "decision"
    }
    family_cells = [
        (key, canonical_rows.get((key.dimension_value, key.metric, key.group_id)))
        for key in family_keys
    ]
    outcome = select_family(
        decision_cells(family_cells),
        q,
        inference,
        alpha,
        computation=_merge_decision_computations(tuple(segment_computations.values())),
    )
    family_record = {
        "family_axes": ("metric", "arm", "segment"),
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
    }
    selected_keys = outcome.selected
    if outcome.fcr_alpha is None:
        for segment_value, estimate, idx in family_entries:
            key = SegmentHypothesisKey(
                estimate.metric, estimate.group_id, "itt", dimension, segment_value
            )
            results[idx] = results[idx].model_copy(
                update={
                    "role": "exploratory",
                    "discovery": (
                        family_discovery(outcome, key)
                        if estimate.method_role == "decision"
                        else None
                    ),
                    "family_axes": family_record["family_axes"]
                    if estimate.method_role == "decision"
                    else None,
                    "family_q": family_record["family_q"]
                    if estimate.method_role == "decision"
                    else None,
                    "family_threshold": family_record["family_threshold"]
                    if estimate.method_role == "decision"
                    else None,
                }
            )
        return results
    metrics_by_name = {metric.name: metric for metric in metrics}
    selected_segment_metrics = {(key.dimension_value, key.metric) for key in selected_keys}
    reestimated: dict[tuple[str, str], list[LiftEstimate]] = {}
    for segment_value, metric_name in selected_segment_metrics:
        metric_rows = metric_rows_by_segment_metric[(segment_value, metric_name)]
        metric_methods = (methods_by_metric or {}).get(metric_name, methods) or [
            Method(name="unadjusted")
        ]
        requested_roles = (method_roles_by_metric or {}).get(metric_name, method_roles)
        metric_roles = _derive_method_roles(metric_methods, requested_roles)
        decision_name = next(name for name, role in metric_roles.items() if role == "decision")
        decision_method = next(
            (method for method in metric_methods if method.name == decision_name),
            metric_methods[0] if metric_methods else Method(name="unadjusted"),
        )
        re_estimates, _reason, _failures = _estimate_lift_or_reason(
            metrics=[metrics_by_name[metric_name]],
            summary=metric_rows,
            control_group=control_group,
            prior=prior,
            alpha=_fcr_alpha_for(alternative, outcome.fcr_alpha),
            methods=[decision_method],
            alternative=alternative,
            inference=None,
            method_roles={decision_name: "decision"},
        )
        if re_estimates:
            reestimated[(segment_value, metric_name)] = [
                open_bound_from_two_sided_at_target(r) for r in re_estimates
            ]
    for segment_value, estimate, idx in family_entries:
        key = SegmentHypothesisKey(
            estimate.metric, estimate.group_id, "itt", dimension, segment_value
        )
        if key not in selected_keys:
            results[idx] = results[idx].model_copy(
                update={
                    "role": "exploratory",
                    "discovery": (
                        family_discovery(outcome, key)
                        if estimate.method_role == "decision"
                        else None
                    ),
                    "family_axes": family_record["family_axes"]
                    if estimate.method_role == "decision"
                    else None,
                    "family_q": family_record["family_q"]
                    if estimate.method_role == "decision"
                    else None,
                    "family_threshold": family_record["family_threshold"]
                    if estimate.method_role == "decision"
                    else None,
                }
            )
            continue
        if estimate.method_role != "decision":
            results[idx] = results[idx].model_copy(
                update={
                    "role": "exploratory",
                    "discovery": None,
                    "family_axes": None,
                    "family_q": None,
                    "family_threshold": None,
                }
            )
            continue
        candidates = reestimated.get((segment_value, estimate.metric), [])
        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.group_id == estimate.group_id and candidate.method == estimate.method
            ),
            None,
        )
        if match is None:
            raise AssertionError(
                f"run_breakout: selected cell {key!r} has no matching "
                "FCR re-estimate -- pass-2 estimation diverged from "
                "pass-1 for the same retained moments"
            )
        original = results[idx]
        results[idx] = _breakout_estimate_row(
            match,
            dimension=dimension,
            dimension_value=segment_value,
            source=original.source,
            n_treat=original.n_treat,
            n_control=original.n_control,
            low_reliability=original.low_reliability,
            role="exploratory",
            discovery=family_discovery(outcome, key),
            family_axes=("metric", "arm", "segment"),
            family_q=outcome.q,
            family_threshold=outcome.realized_threshold,
        )
    return results


class _SequentialBreakoutRequest(NamedTuple):
    control_group: str
    dimension: str
    alpha: float
    alternative: Alternative
    correction: Correction
    q: float
    source: str | None
    reliability_floor: int


def _snapshot_breakout(
    snapshot: SequentialSnapshot,
    metrics: Sequence[Metric],
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
    request: _SequentialBreakoutRequest,
) -> BreakoutEstimates:
    from increment.estimation.sequential_runtime import (
        selected_snapshot_results,
        validate_engine_request,
    )
    from increment.sequential_source import validate_breakout_registration

    reg = inference.registration
    validate_engine_request(snapshot, metrics, alpha=None, alternative=request.alternative)
    validate_breakout_registration(
        reg,
        control_group=request.control_group,
        dimension=request.dimension,
        correction=request.correction,
        q=request.q,
        alpha_by_metric={m.metric: request.alpha for m in reg.models},
    )
    rows = selected_snapshot_results(snapshot, inference, nominal_alpha=request.alpha)
    output = []
    for row in rows:
        cp = row.require_sequential_result().checkpoint
        output.append(
            _breakout_estimate_row(
                row.model_copy(update={"role": "exploratory"}),
                dimension=request.dimension,
                dimension_value=cp.cell.segment[0][1],
                source=request.source,
                n_treat=cp.treatment.n,
                n_control=cp.control.n,
                low_reliability=min(cp.treatment.n, cp.control.n) < request.reliability_floor,
                discovery=row.discovery,
                family_axes=row.family_axes,
                family_q=row.family_q,
                family_threshold=row.family_threshold,
            )
        )
    return BreakoutEstimates(output)


def run_breakout(  # noqa: PLR0913
    summary: SequentialSnapshot | IntoDataFrame | Iterable[Mapping[str, Any]],
    metrics: Sequence[Metric],
    *,
    control_group: str,
    dimension: str,
    source: str | None = None,
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    alpha: float = 0.05,
    alternative: Alternative = "two-sided",
    correction: Correction = "none",
    q: float = 0.10,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    reliability_floor: int = DEFAULT_RELIABILITY_FLOOR,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    method_roles_by_metric: Mapping[str, Mapping[str, Literal["decision", "sensitivity"]]]
    | None = None,
    methods_by_metric: Mapping[str, list[Method]] | None = None,
    policy_name: Literal["compiled_plan", "default_exploratory"] = "default_exploratory",
) -> BreakoutEstimates:
    """Estimate lift for every distinct value of a dimension. Partitions
    ``summary`` on ``dimension`` before calling ``estimate_lift`` once
    per value - mixing segments in one call would corrupt results (see
    the module docstring).

    Parameters
    ----------
    summary : IntoDataFrame | Iterable[Mapping[str, Any]]
        ``group_summary`` rows with an extra ``dimension`` column.
    metrics, control_group, methods, prior, alternative
        Forwarded to ``estimate_lift`` unchanged, once per segment.
        ``methods`` defaults to unadjusted; an explicit ``[]`` raises.
    dimension : str
        The column to split on, e.g. ``"country"``.
    source : str | None
        Resolved ``FactSource`` name, stamped onto every result.
    alpha : float
        Per-segment level, Bonferroni-adjusted if requested; the nominal
        (pass-1) level under ``correction="bh"``.
    correction : Correction
        ``"bonferroni"`` divides *alpha* by the segment count.
        ``"bh"`` selects a flat family and reinverts selected intervals at
        min(q*R/m, nominal alpha). Sequential inputs use the predeclared full
        roster and current/frozen likelihood, including missing cells with
        log evidence -infinity. Fixed-horizon inputs retain the present-row
        family and require complete p-value evidence. Every row is exploratory.
        Posterior effect priors are excluded from family testing.
    q : float
        FDR level for ``correction="bh"``'s family selection. Unused
        otherwise.
    inference : AsymptoticMean | AlwaysValid | MixedFamily | None
        Registered sequential evidence for the full predeclared roster.
    reliability_floor : int
        Per-arm floor flagging ``low_reliability=True`` (default 50).

    Raises
    ------
    ValueError
        Invalid ``correction``, or an explicit ``methods=[]``.
    CapabilityError
        Under ``correction="bh"``, a family member carrying failed, missing,
        or invalid typed evidence (code ``family.evidence.incomplete``) --
        which includes a cell present in a segment that could not be
        estimated. Only a metric ABSENT from a segment contributes no cell
        and cannot trigger this.

    Returns
    -------
    BreakoutEstimates
        One row per (segment, metric, method, arm) cell, dense: an
        excluded cell gets ``lift=None`` (see :data:`ExclusionReason`).
    """
    if correction not in ("none", "bonferroni", "bh"):
        _refuse("breakout.run_breakout_correction", correction=correction)
    if correction == "bh" and prior is not None:
        _refuse("breakout.run_breakout_bh_excludes_prior")
    if methods is not None and not methods:
        _refuse("breakout.run_breakout_methods")
    if inference is not None:
        declared_methods = [
            *(methods or ()),
            *(m for group in (methods_by_metric or {}).values() for m in group),
        ]
        if prior is not None or any(
            m.name != "unadjusted" or m.variance_reduction != "none" for m in declared_methods
        ):
            sequential_refuse(
                "route.unsupported", "certified breakout requires raw unadjusted observations"
            )
        if not isinstance(summary, SequentialSnapshot):
            sequential_refuse(
                "source.invalid", "certified breakout requires a finalized exact snapshot"
            )
        return _snapshot_breakout(
            summary,
            metrics,
            inference,
            _SequentialBreakoutRequest(
                control_group,
                dimension,
                alpha,
                alternative,
                correction,
                q,
                source,
                reliability_floor,
            ),
        )
    if isinstance(summary, SequentialSnapshot):
        sequential_refuse(
            "source.invalid", "exact breakout snapshot requires its registered runtime"
        )
    resolved_methods = methods if methods is not None else [Method(name="unadjusted")]
    _validate_methods(resolved_methods)
    binomial_gate_exempt_metrics = _binomial_gate_exempt_metrics(
        metrics,
        methods=methods,
        methods_by_metric=methods_by_metric,
        inference=inference,
        prior=prior,
    )
    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        if dimension not in frame.columns:
            _refuse("breakout.run_breakout_dimension", dimension=dimension, columns=frame.columns)
        missing = [c for c in _REQUIRED_BREAKOUT_COLUMNS if c not in frame.columns]
        if missing:
            _refuse("breakout.run_breakout_group", missing=missing)
        rows: Iterable[Mapping[str, Any]] = frame.iter_rows(named=True)
    else:
        rows = cast("Iterable[Mapping[str, Any]]", summary)
    segments, all_pairs = _partition_breakout_summary(
        rows,
        dimension=dimension,
        control_group=control_group,
        methods_by_metric=methods_by_metric,
    )
    segment_count = len(segments)
    k = max(segment_count, 1) if correction == "bonferroni" else 1
    alpha_seg = _conservative_divide(alpha, k)
    results: list[BreakoutEstimate] = []
    segment_computations: dict[tuple[str, str], DecisionComputation[LiftEstimate]] = {}
    warned_open_ended: set[str] = set()
    rewarn_registry: dict[Any, int] = {}
    family_entries: list[tuple[str, LiftEstimate, int]] = []
    metric_rows_by_segment_metric: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for value in sorted(segments):
        segment_rows = segments[value]
        context = _BreakoutContext(
            metrics,
            correction,
            value,
            dimension,
            control_group,
            source,
            methods,
            methods_by_metric,
            prior,
            alpha_seg,
            alternative,
            inference,
            method_roles,
            method_roles_by_metric,
            warned_open_ended,
            rewarn_registry,
            all_pairs,
            reliability_floor,
            resolved_methods,
        )
        prepared = _prepare_lift_slice(
            segment_rows,
            control_group=control_group,
            metric_types={m.name: m.type for m in metrics},
            binomial_gate_exempt_metrics=binomial_gate_exempt_metrics,
            warning_label=f"run_breakout: segment {dimension}={value!r}",
        )
        pass_result = _estimate_breakout_slice_metrics(prepared, context)
        segment_computations.update(pass_result.computations)
        metric_rows_by_segment_metric.update(pass_result.metric_rows)
        row_pass = _append_breakout_slice_rows(prepared, context, pass_result, len(results))
        results.extend(row_pass.rows)
        family_entries.extend(row_pass.family_entries)
    family_context = _BreakoutFamilyContext(
        segments,
        all_pairs,
        family_entries,
        metric_rows_by_segment_metric,
        segment_computations,
        metrics,
        correction,
        q,
        inference,
        alpha,
        control_group,
        dimension,
        methods,
        methods_by_metric,
        prior,
        alternative,
        method_roles,
        method_roles_by_metric,
    )
    results = _apply_breakout_family_correction(results, family_context)
    if policy_name != "default_exploratory":
        results = [row.model_copy(update={"policy_name": policy_name}) for row in results]

    output = BreakoutEstimates(results)
    return output


def _opt(v: Any) -> float | None:
    """Convert None/NaN to None for optional float columns - mirrors
    ``increment.estimation.engine._opt`` (pandas has no float NULL, so
    an unmaterialised optional moment arrives as NaN)."""
    if _is_absent(v):
        return None
    return float(v)


def _coerce_date(v: Any) -> date:
    """Coerce a ``ds`` cell to ``datetime.date``.

    Its concrete Python type depends on the native frame that carried
    it: pyarrow decodes to ``date`` already, pandas commonly carries a
    ``Timestamp``/``datetime`` (always midnight-truncated). ``datetime``
    is checked before ``date`` since it is a subclass of ``date``.
    """
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        return date.fromisoformat(v)
    _refuse("breakout.coerce_type_datetime", v=v, type_name=type(v).__name__)


# daily_group_summary's columns run_daily actually reads (plus `ds`,
# which group_summary doesn't need).
_REQUIRED_DAILY_COLUMNS = [
    "ds",
    "experiment_id",
    "metric",
    "group_id",
    "n",
    "ref_y",
    "cy1",
    "cy2",
]


def run_daily(
    summary: IntoDataFrame | Iterable[Mapping[str, Any]],
    metrics: Sequence[Metric],
    *,
    alpha: float = 0.05,
    dimension: str | None = None,
    source: str | None = None,
    view: DayAxisView = "daily",
) -> DailyMetricValues:
    """Reduce per-day moments to absolute per-arm metric values with CIs.

    Unlike :func:`run_breakout`/:func:`run_daily_lift`, this does not
    call ``estimate_lift`` - it is a per-row moment reduction, not a
    relative-lift computation. Each row of *summary* is already exactly
    one ``(ds, metric, group_id)`` group, so no grouping is needed.

    Routes each row through the same ``VARIANCE_MODELS`` dispatch
    ``estimate_lift`` uses (by the row's declared ``Metric.type``), so a
    ratio metric's value is ``mean(y) / mean(den)``, not the raw
    numerator mean.

    Parameters
    ----------
    summary : IntoDataFrame | Iterable[Mapping[str, Any]]
        ``daily_group_summary`` rows (one per ``(ds, metric,
        group_id)``).
    metrics : Sequence[Metric]
        Declared metric definitions.
    alpha : float
        Two-sided CI size (default 0.05 -> 95% CI), on the log scale.
    dimension : str | None
        When given, stamps each result with the named segment column,
        mirroring :func:`run_breakout`'s string-coercion convention.
        Does not partition *summary* - only changes what gets stamped.
    source : str | None
        Resolved ``FactSource`` name backing *dimension*.
    view : Literal["daily", "asof", "cohort"]
        Which day-axis view produced *summary*; governs the retention
        guard and the emitted ``ds_basis``.

    Returns
    -------
    DailyMetricValues
        One per ``(ds, metric, group_id)`` row, always - never drops a
        row. A row with ``n < 2`` or a non-positive mean is unestimable
        and comes back with ``value=None``, ``unavailable`` naming the
        reason, and its real ``n``, rather than being dropped or raising.
    """
    reject_retention_metrics(metrics, "run_daily", view=view)
    ds_basis: Literal["calendar", "cohort"] = "cohort" if view == "cohort" else "calendar"
    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        required = (
            [*_REQUIRED_DAILY_COLUMNS, dimension]
            if dimension is not None
            else _REQUIRED_DAILY_COLUMNS
        )
        missing = [c for c in required if c not in frame.columns]
        if missing:
            _refuse("breakout.daily_group_summary", missing=missing)
        rows: Iterable[Mapping[str, Any]] = frame.iter_rows(named=True)
    else:
        rows = cast("Iterable[Mapping[str, Any]]", summary)

    metric_types = {m.name: m.type for m in metrics}
    if not 0.0 < alpha < 1.0:
        _refuse("estimation.diagnostics.alpha", alpha=alpha)
    if alpha / 2.0 == 0.0:
        _refuse("estimation.meta.alpha_too_small")
    z = _norm.isf(alpha / 2.0)

    results: list[DailyMetricValue] = []
    for row in rows:
        n = int(row["n"])
        metric = str(row["metric"])
        group_id = str(row["group_id"])
        mean_y = _mean_y(row) if n > 0 else 0.0
        mean_den = _mean_den(row) if n > 0 and metric_types.get(metric) == "ratio" else None
        unavailable: ExclusionReason | None = (
            "few_units"
            if n < 2
            else "nonpositive_mean"
            if mean_y <= 0.0
            or (metric_types.get(metric) == "ratio" and (mean_den is None or mean_den <= 0.0))
            else None
        )
        if unavailable is not None:
            results.append(
                DailyMetricValue(
                    ds=_coerce_date(row["ds"]),
                    metric=metric,
                    group_id=group_id,
                    value=None,
                    unavailable=unavailable,
                    n=n,
                    ds_basis=ds_basis,
                    dimension=dimension,
                    dimension_value=str(row[dimension]) if dimension is not None else None,
                    source=source,
                )
            )
            continue

        arm = ArmStats(
            study_id=str(row["experiment_id"]),
            metric=metric,
            group_id=group_id,
            n=n,
            ref_y=float(row["ref_y"]),
            cy1=float(row["cy1"]),
            cy2=float(row["cy2"]),
            x_role=X_SLOT_ROLES["x"] if _opt(row.get("ref_x")) is not None else None,
            **{slot: _opt(row.get(slot)) for slot in _DAY_VALUE_SLOTS},
        )
        metric_type = metric_types[metric]
        variance_model = VARIANCE_MODELS.get(metric_type)
        log_mean, se = variance_model.log_mean_se(arm)
        results.append(
            DailyMetricValue(
                ds=_coerce_date(row["ds"]),
                metric=metric,
                group_id=group_id,
                value=Estimate(
                    value=math.exp(log_mean),
                    lb=math.exp(log_mean - z * se),
                    ub=math.exp(log_mean + z * se),
                    level=math.fsum((1.0, -alpha)),
                    alpha=alpha,
                ),
                n=n,
                ds_basis=ds_basis,
                dimension=dimension,
                dimension_value=str(row[dimension]) if dimension is not None else None,
                source=source,
            )
        )
    return DailyMetricValues(results)


# Cohesive unavailable-row assembly; splitting this path is deferred.
def _nan_lift_rows(  # noqa: PLR0913
    unestimable: Mapping[tuple[str, str], ExclusionReason],
    *,
    methods: list[Method] | None,
    ds_value: date | datetime | str | int | float,
    ds_basis: Literal["calendar", "cohort"],
    dimension: str | None,
    dim_value: str | None,
    source: str | None,
    inference: str = "fixed",
    alternative: str = "two-sided",
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    methods_by_metric: Mapping[str, list[Method]] | None = None,
    method_roles_by_metric: Mapping[str, Mapping[str, Literal["decision", "sensitivity"]]]
    | None = None,
    unavailable_methods: Mapping[tuple[str, str], set[str]] | None = None,
    policy_by_metric: Mapping[str, _DailyCellPolicy] | None = None,
) -> list[DailyLiftEstimate]:
    """One unavailable ``DailyLiftEstimate`` per (metric, arm) x method.

    The reason mapping is deduplicated by the caller, since the same pair
    can be reached by more than one route. Full policy provenance remains
    present even when no numerical estimate is available.
    """
    resolved = methods if methods is not None else [Method(name="unadjusted")]
    role_map = _derive_method_roles(resolved, method_roles)
    rows: list[DailyLiftEstimate] = []
    for (metric, group_id), reason in sorted(unestimable.items()):
        policy = (policy_by_metric or {}).get(
            metric,
            _DailyCellPolicy(0.05, alternative, 0.0, None, None, "default_exploratory"),
        )
        for method in (methods_by_metric or {}).get(metric, resolved):
            if unavailable_methods is not None and method.name not in unavailable_methods.get(
                (metric, group_id), {method.name}
            ):
                continue
            rows.append(
                DailyLiftEstimate(
                    metric=metric,
                    group_id=group_id,
                    method=method.name,
                    method_role=(method_roles_by_metric or {})
                    .get(metric, role_map)
                    .get(method.name, "decision"),
                    inference=inference,
                    reference_kind="sequential" if inference != "fixed" else "normal",
                    alternative=policy.alternative,
                    null_lift=policy.null_lift,
                    null_abs=policy.null_abs,
                    policy_name=policy.policy_name,
                    role=policy.role,
                    ds=ds_value,
                    lift=None,
                    unavailable=reason,
                    ds_basis=ds_basis,
                    dimension=dimension,
                    dimension_value=dim_value,
                    source=source,
                )
            )
    return rows


def _asof_monitoring_note(
    note: str | None,
    *,
    estimand: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    design: Randomized | Encouragement | Observational | None,
    secondary_fixed_horizon: bool = False,
) -> str | None:
    """Append the as-of monitoring caveats this row's inference/design
    combination warrants, without duplicating text already present.

    Fixed-alpha rows get the repeated-looks caveat; unwindowed
    encouragement uptake rows get the complier-drift caveat.
    ``secondary_fixed_horizon=True`` additionally appends the disclosure
    that a fixed-horizon secondary's `discovery` verdict is never stamped
    on this per-date view - `readouts.asof_lift` never runs BH/FCR family
    selection across dates or cells, only `readouts.run` does.
    """
    notes = [note] if note else []
    if inference is None:
        notes.append("monitoring readout: fixed-alpha, not valid under repeated looks")
    if (
        estimand in ("compliance", "late")
        and isinstance(design, Encouragement)
        and design.uptake.window_days is None
    ):
        notes.append("uptake unwindowed: complier definition drifts as the window grows")
    if secondary_fixed_horizon:
        notes.append("family verdicts at run()")
    return "; ".join(dict.fromkeys(notes)) or None


class _DailyCellPolicy(NamedTuple):
    """Complete decision policy for one metric in one date/segment slice."""

    alpha: float
    alternative: str
    null_lift: float
    null_abs: float | None
    role: Role | None
    policy_name: Literal["compiled_plan", "default_exploratory"]


class _DailyLiftContext(NamedTuple):
    control_group: str
    methods: list[Method] | None
    prior: Prior | None
    alpha: float
    alternative: Alternative
    design: Randomized | Encouragement | Observational | None
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None
    inference_label: str
    view: DayAxisView
    dimension: str | None
    source: str | None
    ds_basis: Literal["calendar", "cohort"]
    reliability_floor: int
    methods_by_metric: Mapping[str, list[Method]] | None
    prior_by_metric: Mapping[str, Prior | None] | None
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None
    method_roles_by_metric: Mapping[str, Mapping[str, Literal["decision", "sensitivity"]]] | None
    plan: CompiledDecisionPlan | None = None
    segment_count_by_metric: Mapping[str, int] | None = None


class _DailyMetricPass(NamedTuple):
    estimates: list[LiftEstimate]
    unestimable: dict[tuple[str, str], ExclusionReason]
    unavailable_methods: dict[tuple[str, str], set[str]]


def _plan_daily_lift_slices(
    raw_rows: list[Mapping[str, Any]],
    *,
    dimension: str | None,
    per_metric_correction: bool,
    preserve_day_axis: bool = False,
) -> dict[tuple[Any, str | None, str | None], list[Mapping[str, Any]]]:
    """Partition daily moments before any estimation call."""
    slices: dict[tuple[Any, str | None, str | None], list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        ds_value = row["ds"] if preserve_day_axis else _coerce_date(row["ds"])
        dim_value = str(row[dimension]) if dimension is not None else None
        metric_name = str(row["metric"]) if per_metric_correction else None
        slices.setdefault((ds_value, dim_value, metric_name), []).append(row)
    return slices


def _estimate_itt_slice_or_unavailable(
    day_rows: list[Mapping[str, Any]],
    slice_metrics: Sequence[Metric],
    prepared: _PreparedLiftSlice,
    context: _DailyLiftContext,
    policy_by_metric: Mapping[str, _DailyCellPolicy],
) -> _DailyMetricPass:
    """Estimate ITT cells and track method-specific unavailable cells."""
    unestimable = dict(prepared.unestimable)
    unavailable_methods: dict[tuple[str, str], set[str]] = {}
    estimates: list[LiftEstimate] = []
    for metric in slice_metrics:
        metric_rows = [row for row in prepared.live_rows if str(row["metric"]) == metric.name]
        policy = policy_by_metric[metric.name]
        if not any(str(row.get("group_id")) == context.control_group for row in metric_rows):
            continue
        configured = _configured_methods_for_metric(
            metric.name,
            methods=context.methods,
            methods_by_metric=context.methods_by_metric,
        )
        requested_roles = (context.method_roles_by_metric or {}).get(
            metric.name, context.method_roles
        )
        resolved_roles = _derive_method_roles(configured, requested_roles)
        metric_prior = (context.prior_by_metric or {}).get(metric.name, context.prior)
        mixed_groups = _mixed_binomial_method_groups(
            metric,
            configured,
            inference=context.inference,
            prior=metric_prior,
        )
        groups = list(mixed_groups) if mixed_groups is not None else [configured]
        for group_index, method_group in enumerate(groups):
            group_rows = metric_rows
            if mixed_groups is not None and group_index == 0:
                group_rows = _unadjusted_binomial_rows(metric_rows)
            elif mixed_groups is not None:
                legacy_prepared = _prepare_lift_slice(
                    metric_rows,
                    control_group=context.control_group,
                    metric_types={metric.name: metric.type},
                    binomial_gate_exempt_metrics=frozenset(),
                )
                group_rows = legacy_prepared.live_rows
                for pair, reason in legacy_prepared.unestimable.items():
                    unestimable.setdefault(pair, reason)
                    unavailable_methods.setdefault(pair, set()).update(
                        method.name for method in method_group
                    )
                if not legacy_prepared.part.live_control_metrics:
                    continue
            metric_estimates, metric_skip, metric_failures = _estimate_lift_or_reason(
                metrics=[metric],
                summary=group_rows,
                control_group=context.control_group,
                methods=method_group,
                prior=metric_prior,
                alpha=policy.alpha,
                alternative=policy.alternative,
                null_lift=policy.null_lift,
                null_abs=policy.null_abs,
                inference=context.inference,
                method_roles=resolved_roles,
            )
            if metric_failures and metric_estimates and context.view == "daily":
                _warn(
                    "breakout.estimates.daily_partial_guarded_arms",
                    metric_name=metric.name,
                    stacklevel=3,
                )
            for hypothesis, failure in metric_failures.items():
                reason: ExclusionReason = _failure_exclusion_reason(failure)
                pair = (str(hypothesis.metric), str(hypothesis.group_id))
                unestimable.setdefault(pair, reason)
                method_name = failure.context.get("method")
                if isinstance(method_name, str):
                    unavailable_methods.setdefault(pair, set()).add(method_name)
            if metric_skip is not None:
                reason = _failure_exclusion_reason(metric_failures[next(iter(metric_failures))])
                for row in day_rows:
                    group_id = str(row.get("group_id"))
                    if str(row["metric"]) == metric.name and group_id != context.control_group:
                        pair = (metric.name, group_id)
                        unestimable.setdefault(pair, reason)
                        unavailable_methods.setdefault(pair, set()).update(
                            method.name for method in method_group
                        )
            else:
                estimates.extend(metric_estimates or [])
    successful_methods = {
        (estimate.metric, estimate.group_id, estimate.method) for estimate in estimates
    }
    declared_names = {metric.name for metric in slice_metrics}
    for row in day_rows:
        metric_name = str(row["metric"])
        group_id = str(row.get("group_id"))
        if metric_name not in declared_names or group_id == context.control_group:
            continue
        pair = (metric_name, group_id)
        configured = _configured_methods_for_metric(
            metric_name,
            methods=context.methods,
            methods_by_metric=context.methods_by_metric,
        )
        missing = {
            method.name
            for method in configured
            if (metric_name, group_id, method.name) not in successful_methods
        }
        if missing:
            unestimable.setdefault(pair, "zero_variance")
            unavailable_methods.setdefault(pair, set()).update(missing)
    for metric_name, group_id in unestimable:
        configured = _configured_methods_for_metric(
            metric_name,
            methods=context.methods,
            methods_by_metric=context.methods_by_metric,
        )
        unavailable_methods.setdefault((metric_name, group_id), set()).update(
            method.name
            for method in configured
            if (metric_name, group_id, method.name) not in successful_methods
        )
    return _DailyMetricPass(estimates, unestimable, unavailable_methods)


def _daily_lift_rows_for_slice(
    day_rows: list[Mapping[str, Any]],
    prepared: _PreparedLiftSlice,
    metric_pass: _DailyMetricPass,
    *,
    ds_value: date | datetime | str | int | float,
    dim_value: str | None,
    context: _DailyLiftContext,
    policy_by_metric: Mapping[str, _DailyCellPolicy],
) -> list[DailyLiftEstimate]:
    """Project estimated and unavailable ITT cells for one day slice."""
    rows: list[DailyLiftEstimate] = []
    for estimate in metric_pass.estimates:
        treatment_n = prepared.arm_n.get((estimate.metric, estimate.group_id))
        control_n = prepared.control_n.get(estimate.metric)
        rows.append(
            DailyLiftEstimate(
                **_copy_common_fields(
                    estimate,
                    role=policy_by_metric[estimate.metric].role,
                    null_lift=policy_by_metric[estimate.metric].null_lift,
                    null_abs=policy_by_metric[estimate.metric].null_abs,
                    policy_name=policy_by_metric[estimate.metric].policy_name,
                    reference_kind=_slice_reference_kind(estimate),
                    ds=ds_value,
                    ds_basis=context.ds_basis,
                    dimension=context.dimension,
                    dimension_value=dim_value,
                    source=context.source,
                    low_reliability=(
                        treatment_n is not None and treatment_n < context.reliability_floor
                    )
                    or (control_n is not None and control_n < context.reliability_floor),
                    # The per-day ITT estimation pass produces no
                    # family-correction metadata.
                    family_axes=None,
                    family_q=None,
                    family_threshold=None,
                    family_guarantee=None,
                    family_nominal_alpha=None,
                )
            )
        )
    rows.extend(
        _nan_lift_rows(
            metric_pass.unestimable,
            unavailable_methods=metric_pass.unavailable_methods,
            methods=context.methods,
            methods_by_metric=context.methods_by_metric,
            method_roles_by_metric=context.method_roles_by_metric,
            policy_by_metric=policy_by_metric,
            ds_value=ds_value,
            ds_basis=context.ds_basis,
            dimension=context.dimension,
            dim_value=dim_value,
            source=context.source,
            inference=context.inference_label,
            alternative=context.alternative,
            method_roles=context.method_roles,
        )
    )
    return rows


def _stamp_asof_monitoring_notes(
    results: list[DailyLiftEstimate],
    *,
    view: DayAxisView,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    design: Randomized | Encouragement | Observational | None,
) -> list[DailyLiftEstimate]:
    """Append monitoring caveats to as-of rows."""
    if view != "asof":
        return results
    return [
        row.model_copy(
            update={
                "note": _asof_monitoring_note(
                    row.note, estimand=row.estimand, inference=inference, design=design
                )
            }
        )
        for row in results
    ]


def _append_daily_encouragement_rows(
    results: list[DailyLiftEstimate],
    day_rows: list[Mapping[str, Any]],
    slice_metrics: Sequence[Metric],
    estimands: Sequence[str],
    context: _DailyLiftContext,
    policy_by_metric: Mapping[str, _DailyCellPolicy],
    *,
    ds_value: date | datetime | str | int | float,
    dim_value: str | None,
) -> None:
    """Append as-of encouragement estimates for the current day slice."""
    for metric in slice_metrics:
        metric_rows = [row for row in day_rows if str(row["metric"]) == metric.name]
        results.extend(
            _encouragement_rows_for_slice(
                metric_rows,
                metric=metric,
                design=cast(Encouragement, context.design),
                estimands=estimands,
                methods=(context.methods_by_metric or {}).get(metric.name, context.methods),
                prior=(context.prior_by_metric or {}).get(metric.name, context.prior),
                policy=policy_by_metric[metric.name],
                ds_value=ds_value,
                ds_basis=context.ds_basis,
                dimension=context.dimension,
                dim_value=dim_value,
                source=context.source,
                inference=context.inference,
                reliability_floor=context.reliability_floor,
                method_roles=(context.method_roles_by_metric or {}).get(
                    metric.name, context.method_roles
                ),
            )
        )


def _validate_daily_lift_request(
    metrics: Sequence[Metric],
    *,
    correction: Literal["none", "bonferroni"],
    view: DayAxisView,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    design: Randomized | Encouragement | Observational | None,
    estimands: Sequence[str] | None,
) -> tuple[str, bool, tuple[str, ...]]:
    """Validate daily-lift policy and resolve requested estimands."""
    if correction not in ("none", "bonferroni"):
        _refuse("breakout.run_daily_lift", correction=correction)
    reject_retention_metrics(metrics, "run_daily_lift", view=view)
    if inference is not None and view != "asof":
        _refuse("readout.inference.disjoint_slices", view=view, inference=type(inference).__name__)
    encouragement_asof = view == "asof" and isinstance(design, Encouragement)
    requested_estimands = tuple(
        estimands if estimands is not None else (ESTIMANDS if encouragement_asof else ("itt",))
    )
    unknown_estimands = set(requested_estimands) - set(ESTIMANDS)
    if unknown_estimands:
        _refuse(
            "breakout.unknown_estimand_supported",
            unknown_estimands=sorted(unknown_estimands),
            supported=ESTIMANDS,
        )
    if not encouragement_asof and requested_estimands != ("itt",):
        _refuse("breakout.compliance_late_view")
    return (
        inference.label if inference is not None else "fixed",
        encouragement_asof,
        requested_estimands,
    )


def _resolve_daily_cell_policy(
    context: _DailyLiftContext,
    metric: Metric,
    metric_rows: Sequence[Mapping[str, Any]],
) -> _DailyCellPolicy:
    """Resolve the complete policy for one metric/date/segment slice.

    Arm cardinality is date-local: a later arm must not spend an earlier date's
    alpha. Segment cardinality stays readout-wide, because every segment belongs
    to the same predeclared view family even when one date has sparse rows.
    """
    if context.plan is None:
        return _DailyCellPolicy(
            context.alpha, context.alternative, 0.0, None, None, "default_exploratory"
        )
    procedure = context.plan.procedures[metric.name]
    n_arms = len({str(row["group_id"]) for row in metric_rows} - {str(context.control_group)})
    n_segments = (context.segment_count_by_metric or {}).get(metric.name, 1)
    resolver_view: Literal["asof"] | None = "asof" if context.view == "asof" else None
    alpha = resolve_cell_alpha(
        context.plan,
        procedure,
        n_arms=n_arms,
        n_segments=n_segments,
        view=resolver_view,
        mechanism=getattr(context.design, "mechanism", None),
    )
    return _DailyCellPolicy(
        alpha=alpha,
        alternative=procedure.alternative,
        null_lift=float(getattr(procedure, "null_lift", 0.0)),
        null_abs=getattr(procedure, "null_abs", None),
        role=procedure.role if context.plan.declared else None,
        policy_name="compiled_plan",
    )


def _snapshot_daily_lift(
    summary, metrics, requested_estimands, context: _DailyLiftContext
) -> DailyLiftEstimates:
    assert isinstance(context.inference, SEQUENTIAL_POLICIES)
    from increment.estimation.sequential_runtime import (
        selected_snapshot_results,
        validate_engine_request,
    )
    from increment.sequential_source import validate_sequential_plan
    from increment.sequential_state import validate_sequential_methods

    if (
        context.dimension is not None
        or "late" in requested_estimands
        or isinstance(context.design, Observational)
        or context.prior is not None
        or any(p is not None for p in (context.prior_by_metric or {}).values())
    ):
        sequential_refuse(
            "route.unsupported",
            "current sequential as-of requires raw unsegmented ITT or Bernoulli uptake",
        )
    registration = context.inference.registration
    for metric in metrics:
        validate_sequential_methods(
            registration,
            metric.name,
            (context.methods_by_metric or {}).get(metric.name, context.methods or ()),
        )
    if not isinstance(summary, SequentialSnapshot) or summary.reveal_cursor is None:
        sequential_refuse(
            "source.invalid", "as-of likelihood requires a labeled finalized exact snapshot"
        )
    if summary.registration.control_group != context.control_group or any(
        c.segment for c in summary.registration.roster
    ):
        sequential_refuse("source.invalid", "as-of request differs from the retained registration")
    if any(c.estimand not in requested_estimands for c in summary.registration.roster):
        sequential_refuse("source.invalid", "as-of estimands differ from the retained roster")
    if (
        isinstance(context.design, Encouragement)
        and context.design.one_sided
        and "compliance" in requested_estimands
    ):
        sequential_refuse(
            "route.unsupported", "structural-zero uptake requires the C03 fixed-horizon rate target"
        )
    if context.plan is not None:
        if context.plan.inference != context.inference:
            sequential_refuse("source.invalid", "as-of compiled and runtime policies differ")
        validate_sequential_plan(context.plan, metrics, context.design)
    validate_engine_request(
        summary,
        metrics if "itt" in requested_estimands else (),
        alpha=context.alpha if context.plan is None else None,
        alternative=context.alternative if context.plan is None else None,
    )
    results = selected_snapshot_results(
        summary,
        context.inference,
        nominal_alpha=context.alpha if context.plan is None else context.plan.alpha,
    )
    return daily_sequential_projection(
        [row.model_copy(update={"ds": summary.reveal_cursor}) for row in results]
    )


def run_daily_lift(  # noqa: PLR0913
    summary: SequentialSnapshot | IntoDataFrame | Iterable[Mapping[str, Any]],
    metrics: Sequence[Metric],
    *,
    control_group: str,
    dimension: str | None = None,
    source: str | None = None,
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    alpha: float = 0.05,
    correction: Literal["none", "bonferroni"] = "none",
    alternative: Alternative = "two-sided",
    design: Randomized | Encouragement | Observational | None = None,
    view: DayAxisView = "daily",
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    estimands: Sequence[str] | None = None,
    reliability_floor: int = DEFAULT_RELIABILITY_FLOOR,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    methods_by_metric: Mapping[str, list[Method]] | None = None,
    prior_by_metric: Mapping[str, Prior | None] | None = None,
    method_roles_by_metric: Mapping[str, Mapping[str, Literal["decision", "sensitivity"]]]
    | None = None,
    plan: CompiledDecisionPlan | None = None,
) -> DailyLiftEstimates:
    """Estimate relative lift separately per day slice -
    ``run_breakout``'s day-axis twin. Partitions ``summary`` on ``ds``
    alone, or on each (day, dimension value) pair when ``dimension`` is
    set, calling ``estimate_lift`` once per metric within each slice.

    When ``plan`` is supplied, each metric/date slice uses that compiled
    procedure's alpha, alternative, shifted null, and role. On
    ``view="asof"``, the plan's as-of view policy additionally owns segment
    correction. Plan-driven daily/cohort slices apply no compiled segment
    correction. Call-wide ``alpha``, ``alternative``, and ``correction`` are
    the lightweight standalone policy only when ``plan=None``; there,
    ``correction="bonferroni"`` with a dimension divides each metric's alpha
    by its readout-wide segment count.

    Parameters
    ----------
    summary : IntoDataFrame | Iterable[Mapping[str, Any]]
        ``daily_group_summary`` rows carrying a ``ds`` column.
    metrics, control_group, methods, prior
        Runtime metric/estimator inputs applied once per day.
    alpha, alternative, correction
        Standalone exploratory policy used only when ``plan=None``.
    plan : CompiledDecisionPlan | None
        Compiled per-metric policy. Takes precedence over call-wide policy
        scalars and records ``policy_name="compiled_plan"`` on every row.
    design : Randomized | Encouragement | Observational | None
        Under an asof ``Encouragement`` design, additionally estimates
        ``"late"`` rows alongside ``itt``.
    view : Literal["daily", "asof", "cohort"]
        Which day-axis view produced *summary*; governs the retention
        guard and the emitted ``ds_basis``.
    inference : AsymptoticMean | AlwaysValid | MixedFamily | None
        Meaningful only on ``view="asof"``.
    estimands : Sequence[str] | None
        Defaults to ``("itt", "compliance", "late")`` under asof
        encouragement, else ``("itt",)``.
    reliability_floor : int
        Per-arm floor flagging ``low_reliability=True`` (default 50).
    Raises
    ------
    ValueError
        Invalid *correction* or *estimands*, retention metric or incompatible
        design request, invalid methods, or a missing ``ds``/dimension column
        in *summary*.
    NotImplementedError
        *inference* outside ``view="asof"`` (code
        ``readout.inference.disjoint_slices``).
    Returns
    -------
    DailyLiftEstimates
        Dense over ITT cells: an unestimable (day, metric, method, arm)
        gets ``lift=None`` plus an ``unavailable`` reason rather than
        being dropped. A slice that keeps some estimable arms while
        excluding guarded arms emits a per-metric ``UserWarning``;
        pre-estimation drops (too few units, non-positive means) stay
        silent and surface only as unavailable rows. Encouragement
        ``late`` rows are not dense: a weak first stage suppresses the
        requested ``late`` row, leaving only the explanatory
        ``compliance`` row.
    """
    inference_label, encouragement_asof, requested_estimands = _validate_daily_lift_request(
        metrics,
        correction=correction,
        view=view,
        inference=inference,
        design=design,
        estimands=estimands,
    )
    if inference is not None:
        context = _DailyLiftContext(
            control_group,
            methods,
            prior,
            alpha,
            alternative,
            design,
            inference,
            inference_label,
            view,
            dimension,
            source,
            "calendar",
            reliability_floor,
            methods_by_metric,
            prior_by_metric,
            method_roles,
            method_roles_by_metric,
            plan,
        )
        return _snapshot_daily_lift(summary, metrics, requested_estimands, context)
    if isinstance(summary, SequentialSnapshot):
        sequential_refuse("source.invalid", "as-of exact snapshot requires its registered policy")
    _validate_methods(methods if methods is not None else [Method(name="unadjusted")])
    metric_types = {metric.name: metric.type for metric in metrics}
    binomial_gate_exempt_metrics = _binomial_gate_exempt_metrics(
        metrics,
        methods=methods,
        methods_by_metric=methods_by_metric,
        inference=inference,
        prior=prior,
        prior_by_metric=prior_by_metric,
    )
    ds_basis: Literal["calendar", "cohort"] = "cohort" if view == "cohort" else "calendar"
    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    if isinstance(frame, nw.DataFrame):
        if "ds" not in frame.columns:
            _refuse("breakout.run_daily_lift_ds_column_found", columns=frame.columns)
        if dimension is not None and dimension not in frame.columns:
            _refuse(
                "breakout.run_daily_lift_dimension_found",
                dimension=dimension,
                columns=frame.columns,
            )
        rows: Iterable[Mapping[str, Any]] = frame.iter_rows(named=True)
    else:
        rows = cast("Iterable[Mapping[str, Any]]", summary)
    raw_rows = list(rows)
    metric_by_name = {metric.name: metric for metric in metrics}
    unknown = {str(row["metric"]) for row in raw_rows} - metric_by_name.keys()
    if unknown:
        _refuse(
            "breakout.run_daily_lift_metric_declared",
            unknown=sorted(unknown),
            declared=sorted(metric_by_name),
        )
    if plan is not None:
        missing_procedures = sorted(metric_by_name.keys() - plan.procedures.keys())
        if missing_procedures:
            _refuse(
                "breakout.run_daily_lift_plan_missing_procedures",
                missing_procedures=missing_procedures,
            )
    segment_count_by_metric: dict[str, int] = {}
    if dimension is not None:
        segments_by_metric: dict[str, set[str]] = {}
        for row in raw_rows:
            segments_by_metric.setdefault(str(row["metric"]), set()).add(str(row[dimension]))
        segment_count_by_metric = {
            name: max(len(values), 1) for name, values in segments_by_metric.items()
        }
    # A compiled plan makes the resolver own the full per-cell composition;
    # direct plan-less callers retain the standalone call-wide correction.
    # per_metric_correction's own unknown-metric check is unreachable: `unknown`
    # (computed unconditionally above) already refused before this point.
    per_metric_correction = correction == "bonferroni" and dimension is not None and plan is None
    alpha_by_metric: dict[str, float] = {}
    if per_metric_correction:
        alpha_by_metric = {
            name: _conservative_divide(alpha, count)
            for name, count in segment_count_by_metric.items()
        }
    slices = _plan_daily_lift_slices(
        raw_rows,
        dimension=dimension,
        per_metric_correction=per_metric_correction,
        preserve_day_axis=view == "asof",
    )
    results: list[DailyLiftEstimate] = []
    for (ds_value, dim_value, slice_metric_name), day_rows in slices.items():
        slice_metrics = (
            [metric_by_name[slice_metric_name]] if slice_metric_name is not None else metrics
        )
        alpha_segment = (
            alpha_by_metric[slice_metric_name] if slice_metric_name is not None else alpha
        )
        context = _DailyLiftContext(
            control_group,
            methods,
            prior,
            alpha_segment,
            alternative,
            design,
            inference,
            inference_label,
            view,
            dimension,
            source,
            ds_basis,
            reliability_floor,
            methods_by_metric,
            prior_by_metric,
            method_roles,
            method_roles_by_metric,
            plan,
            segment_count_by_metric,
        )
        policy_by_metric = {
            metric.name: _resolve_daily_cell_policy(
                context,
                metric,
                [row for row in day_rows if str(row["metric"]) == metric.name],
            )
            for metric in slice_metrics
        }
        encouragement_estimands = tuple(
            name for name in requested_estimands if name in ("compliance", "late")
        )
        if encouragement_asof and encouragement_estimands:
            _append_daily_encouragement_rows(
                results,
                day_rows,
                slice_metrics,
                encouragement_estimands,
                context,
                policy_by_metric,
                ds_value=ds_value,
                dim_value=dim_value,
            )
        if "itt" not in requested_estimands:
            continue
        prepared = _prepare_lift_slice(
            day_rows,
            control_group=control_group,
            metric_types=metric_types,
            binomial_gate_exempt_metrics=binomial_gate_exempt_metrics,
        )
        if prepared.part.live_control_metrics:
            metric_pass = _estimate_itt_slice_or_unavailable(
                day_rows, slice_metrics, prepared, context, policy_by_metric
            )
        else:
            metric_pass = _DailyMetricPass([], dict(prepared.unestimable), {})
        results.extend(
            _daily_lift_rows_for_slice(
                day_rows,
                prepared,
                metric_pass,
                ds_value=ds_value,
                dim_value=dim_value,
                context=context,
                policy_by_metric=policy_by_metric,
            )
        )
    return DailyLiftEstimates(
        _stamp_asof_monitoring_notes(results, view=view, inference=inference, design=design)
    )
