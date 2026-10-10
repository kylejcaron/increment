"""The estimation orchestrator: ``estimate_lift``.

Consumes ``group_summary`` rows (any narwhals-supported frame or plain
row mappings) and produces ``list[LiftEstimate]``, one per (metric x
method x non-control arm), dispatching on two axes: metric type (mean/
conversion/retention vs ratio, via ``VarianceModel``) and each method's
``variance_reduction`` (a ``VARIANCE_REDUCTION`` registry key, resolved
once into a frozen per-contrast strategy before the estimation loop).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel, ConfigDict, model_validator

from increment._finite_sample_refusals import (
    refuse_finite_sample_cuped,
    refuse_finite_sample_metric_type,
)
from increment._literals import Alternative, ConversionInference, PreferredDirection
from increment._moment_plan import CORE_SLOTS, OPTIONAL_SLOTS, X_ROLE_VARIABLES
from increment.errors import (
    PACKAGE_FRAMES,
    CapabilityError,
    CodedError,
    CodedModel,
    IncrementRuntimeWarning,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    raiser,
    refusals,
    refuse,
    warn,
)
from increment.estimation import binomial_rr
from increment.estimation._readout_refusals import READOUT_REFUSALS as _READOUT_REFUSALS
from increment.estimation._tails import two_sided_critical_value, wald_bounds
from increment.estimation.armstats import (
    ArmStats,
    binary_counts,
    canonical_bernoulli_arm,
    welch_satterthwaite_df,
)
from increment.estimation.conversion_route import (
    finite_sample_blocker,
    refuse_finite_sample_unavailable,
    route_for_counts,
)
from increment.estimation.cuped import AdjustedRatioMoments, fit_cuped, fit_ratio_cuped
from increment.estimation.inference import (
    LiftGuardError,
    PosteriorFields,
    Prior,
    infer_lift,
    posterior_fields,
)
from increment.estimation.results import (
    BINOMIAL_METHOD,
    BINOMIAL_NUMERICAL_QUALIFICATION,
    BinomialConfidenceSet,
    Estimate,
    LiftEstimate,
    _alpha_eff_for,
)
from increment.estimation.sequential import (
    ASYMPTOTIC_PROCEDURE_POLICIES,
    SEQUENTIAL_POLICIES,
    AlwaysValid,
    AsymptoticMean,
    MixedFamily,
    SequentialSupportRequest,
    sequential_support_refusal,
)
from increment.estimation.variance import (
    RATIO_DENOMINATOR_PRECISION_THRESHOLD,
    VARIANCE_MODELS,
    VARIANCE_REDUCTION,
    MeanVarianceModel,
    check_positive_mean,
    ratio_denominator_precision,
    ratio_log_mean_se,
    ratio_moments,
    se_log_mean,
    stable_log_ratio,
)
from increment.estimation.variance import ratio_abs_diff_se as variance_ratio_abs_diff_se
from increment.semantics.models import Metric, RetentionMetric
from increment.sequential_state import SequentialSnapshot, sequential_refuse

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.decision import DecisionComputation, DecisionFailure, PValueEvidence
    from increment.winsor import BootstrapReference, WinsorRawState
# 40 (sandwich variance itself noisy, so borderline significance is fragile).
_WARN_TOTAL_CLUSTERS = 40


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "estimation.engine.cluster.denominator": RefusalSpec(
            "estimation.engine.cluster.denominator",
            CapabilityError,
            template="metric '{metric}' (type '{metric_type}', contrast '{treatment_group}' vs '{control_group}') carries a populated denominator family (ref_den) with no cluster= declared -- this group_summary was built with cluster=... (so 'n' counts clusters, not units), and reading it here as unit grain would silently mislabel a cluster-level point estimate and SE as unit-level. Pass estimate_lift(cluster=<the same column>, ...).",
            keys=frozenset({"cluster"}),
        ),
        "estimation.engine.ratio.cuped": RefusalSpec(
            "estimation.engine.ratio.cuped",
            UnsupportedRequestError,
            template="CUPED cannot be applied to ratio metric '{metric}' from this source: experiment '{experiment}' declares n_pre_periods=0, and the warehouse covariate for a ratio metric is its numerator's pre-period total, so no covariate moments (ref_x/cx1/cx2/cxy) or covariate-denominator cross moment (cxden) can be materialized. Declare n_pre_periods > 0 on the experiment, or supply the covariate column directly via from_unit_summary(metrics=[MetricSpec(type='ratio', numerator=..., denominator=..., covariate='<pre-period column>')]).",
        ),
        "estimation.engine.arm.duplicate_rows": "group_summary carries more than one row for (metric, group_id) {keys} -- an unchecked duplicate control row silently overwrites an earlier one in the lookup dict, and an unchecked duplicate treatment row is silently iterated as if it were a second arm, either way returning an arbitrary lift instead of a refusal. Deduplicate group_summary to exactly one row per (metric, group_id) before calling estimate_lift.",
        "estimation.engine.arm.invalid_count": "{what} is {value!r}, not a finite integer-valued count -- ArmStats.n rejects a fractional value at direct construction, so the untyped dataframe ingress must too instead of silently truncating it (int(2.9) == 2) into a different arm size than the row declared. Round or fix the upstream aggregation so 'n' is integral.",
        "estimation.engine.ratio.negative_variance": "ratio_abs_diff_se refused these moments: {reason} -- a deficit at this scale means the numerator/denominator moments violate Cauchy-Schwarz (an infeasible covariance), not merely cancelled, so reporting a clamped zero-SE (maximal confidence) would be the most dangerous possible failure mode for a decision number.",
        "estimation.engine.method.name_without_variance": "Method(name={self!r}) without variance_reduction='cuped' would label an unadjusted estimate as CUPED-adjusted; pass Method(name='cuped', variance_reduction='cuped') or rename.",
        "estimation.engine.method.name_iptw_does": "Method(name='iptw') does not accept outcome_learner/folds -- IPTW fits one propensity model with no cross-fitting; set propensity_learner (mapped to iptw_estimate's learner=) instead, or use Method(name='dml') / Method(name='aipw') for cross-fit outcome-model pluggability.",
        "estimation.engine.carries_format_moments": "{what} carries format-1 moments (raw additive sums: 'sum_y2' present, 'cy2' absent). increment consumes CENTERED moments -- ref_y/cy1/cy2 and the c-form families -- because raw second moments lose the variance signal to floating-point cancellation once the mean dwarfs the spread, so they are never reinterpreted here. Two ways in: Analysis.from_moments(...), which adapts stamped or unstamped format-1 rows automatically, or ArmStats.from_raw_sums(...) for hand-built rows.",
        "estimation.engine.carries_unsupported_weighted": "{what} carries unsupported weighted ArmStats fields: {unsupported}; weighted estimands must use ScoreStats.",
        "estimation.engine.x_role_declared": "x_role must be declared when the x family is present; an explicit null x_role is not a legacy declaration",
        "estimation.engine.group_summary_missing": "group_summary missing required columns: {missing}",
        "estimation.engine.group_summary_row": "group_summary row missing required columns: {missing}",
        "estimation.engine.winsorization_metadata_present": "winsorization metadata must be present on both contrast arms",
        "estimation.engine.winsorization_metadata_differs": "winsorization metadata differs between arms in {field}",
        "estimation.engine.method_names_unique": "{caller}: method names must be unique within one metric request; duplicates: {duplicates!r}",
        "estimation.engine.method_name_observational": "Method(name={m!r}) is an observational adjustment (estimate_ate / Observational design); this randomized path would label an unadjusted randomized estimate with it.",
        "estimation.engine.metric_found_group": "Metric(s) {unknown} found in group_summary but not in the metrics list. Declared metrics: {metric_types}",
        "estimation.engine.control_group_found": "control_group '{control_group}' not found in arms. Available groups: {known_groups}",
        "estimation.engine.route_alpha": "route_alpha must lie in (0, 1], got {route_alpha!r}",
    },
)
_refuse = raiser(_REFUSALS)


_REFUSALS.update(
    {
        code: _READOUT_REFUSALS[code]
        for code in (
            "breakout.retention.completion",
            "breakout.retention.unbounded",
            "readout.assignment.estimands",
            "readout.margin.breakout",
            "readout.metric.daily_winsorization",
            "readout.metric.percentile_winsorization",
            "readout.metric.quantile_alternative",
            "readout.metric.quantile_breakout",
            "readout.randomized.value_scale",
            "readout.value_scale.unknown_metric",
        )
    }
)


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Callable[..., str]
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(
    code: str,
    /,
    *,
    stacklevel: int = 2,
    skip_file_prefixes: tuple[str, ...] = (),
    **context: object,
) -> None:
    # +1 absorbs this helper's own frame; errors.warn() absorbs its own.
    warn(
        _WARNINGS[code],
        stacklevel=stacklevel + 1,
        skip_file_prefixes=skip_file_prefixes,
        context=context,
    )


_register_warning(
    "estimation.engine.small_total_clusters",
    IncrementRuntimeWarning,
    lambda *, metric_name, n_clusters, cluster, floor: (
        f"metric '{metric_name}': {n_clusters} total clusters "
        f"across arms (cluster '{cluster}') is below "
        f"{floor} -- the cluster-robust variance is "
        f"itself noisy at this K and asymptotic references can "
        f"over-reject; treat borderline "
        f"significance as fragile."
    ),
)

_register_warning(
    "estimation.engine.open_ended_sequential",
    IncrementWarning,
    lambda *, metric_name, windows, fix: (
        f"metric '{metric_name}' is open-ended ({windows}): unit "
        "totals are still accruing between analyses, so the "
        "sequential guarantee -- derived for finalized per-unit "
        "values -- is approximate and the estimand drifts with "
        f"the enrollment mix. {fix} to remove the changing-unit "
        "approximation; the plug-in standard error keeps coverage "
        "asymptotic either way."
    ),
)


def check_total_clusters(
    metric_name: str,
    cluster: str,
    n_clusters: int,
    *,
    warn: bool = True,
) -> None:
    """Emit the small-K advisory without imposing an arbitrary floor.

    Estimator-specific support and positive-df requirements remain enforced
    by each admitted path; generic cluster sandwiches are explicitly
    qualified as working asymptotic references. The warning names the first caller outside
    the package, whatever private layers the estimate passed through.
    """
    if warn and n_clusters < _WARN_TOTAL_CLUSTERS:
        _warn(
            "estimation.engine.small_total_clusters",
            metric_name=metric_name,
            n_clusters=n_clusters,
            cluster=cluster,
            floor=_WARN_TOTAL_CLUSTERS,
            skip_file_prefixes=PACKAGE_FRAMES,
        )


class Method(CodedModel, BaseModel):
    """An estimation configuration: named method with optional adjustments.

    ``name`` is a free-form label stamped onto every ``LiftEstimate`` this
    Method produces, except it must not contradict its own configuration:
    ``name="cuped"`` without ``variance_reduction="cuped"``, or an
    unregistered ``variance_reduction``, are both refused at construction.
    The reserved observational names (``"iptw"``/``"dml"``/``"aipw"``) are
    valid here - ``estimate_ate`` dispatches on them - but refused per-call
    by ``_validate_methods`` on the randomized path.

    ``conversion_inference`` chooses the sampling route for eligible
    unadjusted, unit-grain, fixed-horizon conversion or retention contrasts
    (it has no effect on other contrasts):

    * ``"auto"`` (the default): rows whose four per-arm success and failure
      counts are all dense for the requested tail take the delta-method
      route (``reference_kind="t"``, ``scale="log"``); every other row
      takes the finite-sample route. The prior-free route is selected from
      counts before any interval is computed, whether or not a prior was
      declared. A supported posterior is stored separately; a posterior
      working-likelihood guard does not discard valid exact sampling
      inference.
    * ``"finite_sample"``: every row takes the finite-sample independent-
      binomial route (``reference_kind="binomial"``), valid at any count and
      validated to ``binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE`` units per arm.
      It is refused with ``variance_reduction="cuped"`` at construction and
      with an informative prior, clustered units, or sequential inference
      when the request is made.

    ``propensity_learner``/``outcome_learner``/``folds`` are the
    pluggable-nuisance seam for ``estimate_ate``'s adjustments (ignored by
    the randomized ``estimate_lift``):

    * ``"dml"``/``"aipw"``: both factories pass straight through, each
      cross-fitting a propensity model plus the method's own outcome
      model(s).
    * ``"iptw"``: fits one propensity model with no cross-fitting;
      ``outcome_learner``/``folds`` are refused since IPTW has neither.
    * ``AdjustmentSet(missing="allow")`` requires explicitly supplied
      NaN-native learner(s) here, since the package defaults silently
      emit non-finite predictions under NaN input.
    """

    model_config = ConfigDict(frozen=True)

    name: str  # label carried on the LiftEstimate
    variance_reduction: str = "none"  # "none" | "cuped" (same Registry)
    conversion_inference: ConversionInference = "auto"
    propensity_learner: Callable[[], Any] | None = None
    outcome_learner: Callable[[], Any] | None = None
    folds: int | None = None

    @model_validator(mode="after")
    def _check_label_matches_configuration(self) -> Method:
        VARIANCE_REDUCTION.get(self.variance_reduction)  # raises if unregistered
        if self.conversion_inference == "finite_sample" and self.variance_reduction == "cuped":
            refuse_finite_sample_cuped(self.name)
        if self.name == "cuped" and self.variance_reduction != "cuped":
            _refuse("estimation.engine.method.name_without_variance", self=self.name)
        if self.name == "iptw" and (self.outcome_learner is not None or self.folds is not None):
            _refuse("estimation.engine.method.name_iptw_does")
        return self


# ---------------------------------------------------------------------------
# DataFrame-to-ArmStats conversion


def _opt(v: Any) -> float | None:
    """Convert None/NaN -> None for optional float columns.

    NaN is treated as "absent", not refused: pandas has no float NULL, so
    an unmaterialised optional moment (e.g. a CUPED covariate) arrives as
    NaN, indistinguishable from any other NaN. Required moments
    (n/ref_y/cy1/cy2) differ: they are never legitimately absent, so
    non-finite values there are refused by name in ``_df_to_arms``.
    """
    if v is None or v != v:  # NaN != NaN is True for floats
        return None
    return float(v)


def _opt_int(v: Any) -> int | None:
    """Convert an optional numeric count column to an integer."""
    if v is None or v != v:
        return None
    return int(v)


def _refuse_format_one(names: Collection[str], what: str) -> None:
    """Refuse format-1 (raw additive sums) moments by name.

    A REFUSAL, never schema detection: reading a raw-sum row as a centered
    one would report a variance that is not the data's.
    """
    if "sum_y2" in names and "cy2" not in names:
        _refuse("estimation.engine.carries_format_moments", what=what)


def _refuse_weighted_fields(names: Collection[str], what: str) -> None:
    """Refuse obsolete weighted fields instead of silently ignoring them."""
    unsupported = sorted({"sum_w", "sum_w2"}.intersection(names))
    if unsupported:
        _refuse(
            "estimation.engine.carries_unsupported_weighted", unsupported=unsupported, what=what
        )


def _nullish(value: Any) -> bool:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _infer_x_role(row: Mapping[str, Any]) -> str | None:
    """Read an x-family declaration, inferring only when its key is absent.

    A present ``x_role`` key is authoritative. Native dataframe null sentinels
    normalize to ``None``, but remain present: rows with a materialized x family
    must therefore declare a role. A frame with no role column is a legacy
    producer; ``cxden`` marks an ambiguous clustered row whose role cannot be
    inferred safely.
    """

    def _present(col: str) -> bool:
        return not _nullish(row.get(col))

    if "x_role" in row:
        declared = row["x_role"]
        if _nullish(declared):
            if _present("ref_x"):
                _refuse("estimation.engine.x_role_declared")
            return None
        return cast("str | None", declared)

    if not _present("ref_x"):
        return None
    return None if _present("cxden") else "covariate"


def _validate_required_count(value: Any, *, what: str) -> int:
    """Require a finite, non-fractional count; refuse a decimal one.

    Mirrors ``ArmStats.n``'s own ``int`` field, which pydantic already
    rejects a fractional float against at direct construction -- this
    untyped dataframe ingress must refuse the same value by name instead
    of silently truncating it (``int(2.9) == 2``) into a different arm
    size than the row declared.
    """
    if isinstance(value, bool):
        _refuse("estimation.engine.arm.invalid_count", what=what, value=value)
    if isinstance(value, Integral):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        _refuse("estimation.engine.arm.invalid_count", what=what, value=value)
    if not math.isfinite(numeric) or numeric != math.trunc(numeric):
        _refuse("estimation.engine.arm.invalid_count", what=what, value=value)
    return int(numeric)


def _validate_arm_uniqueness(arms: Sequence[ArmStats]) -> None:
    """Refuse a malformed arm inventory with more than one row per
    (metric, group_id).

    Every downstream consumer -- ``_prepare_lift_estimation``'s
    ``control_by_metric`` dict, the encouragement first-stage/LATE
    lookups -- assumes exactly one row per (metric, group_id). An
    unchecked duplicate silently loses an earlier control row to
    "last one wins", or is silently iterated twice as if it were a
    second arm, either way returning an arbitrary answer instead of a
    refusal. Called once, here, at the single ingress point every
    estimator shares.
    """
    counts: dict[tuple[str, str], int] = {}
    for arm in arms:
        key = (arm.metric, arm.group_id)
        counts[key] = counts.get(key, 0) + 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        _refuse("estimation.engine.arm.duplicate_rows", keys=duplicates)


# ---------------------------------------------------------------------------


def _df_to_arms(summary: IntoDataFrame | Iterable[Mapping[str, Any]]) -> list[ArmStats]:
    """Validate and convert ``group_summary`` rows to ``list[ArmStats]``.

    Accepts any narwhals-supported native frame (pandas / polars / pyarrow /
    ...) or a plain iterable of row mappings (e.g. ``list[dict]``). The
    query layer emits CENTERED moments (format 2); the conversion at the
    edge is the **only** untyped boundary in the estimation pipeline.
    """
    required = ["experiment_id", "metric", "group_id", "n", *CORE_SLOTS]

    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    native_frame = isinstance(frame, nw.DataFrame)
    if native_frame:
        _refuse_format_one(frame.columns, "group_summary")
        _refuse_weighted_fields(frame.columns, "group_summary")
        missing = [c for c in required if c not in frame.columns]
        if missing:
            _refuse("estimation.engine.group_summary_missing", missing=missing)
        rows: Iterable[Mapping[str, Any]] = frame.iter_rows(named=True)
    else:
        rows = cast("Iterable[Mapping[str, Any]]", summary)

    arms: list[ArmStats] = []
    for row in rows:
        _refuse_format_one(row, "group_summary row")
        _refuse_weighted_fields(row, "group_summary row")
        missing = [c for c in required if c not in row]
        if missing:
            # The frame path validates columns up front; row mappings have no
            # shared schema, so this defect needs the same named error too.
            _refuse("estimation.engine.group_summary_row", missing=missing)
        metric = row["metric"]
        group_id = row["group_id"]
        successes = row.get("successes")
        if isinstance(successes, Integral) and not isinstance(successes, bool):
            successes = int(successes)
        elif _nullish(successes):
            successes = None
        arms.append(
            # Internal producers are already centered: construct directly,
            # never through the format-1 adapter.
            ArmStats(
                study_id=str(row["experiment_id"]),
                metric=str(metric),
                group_id=str(group_id),
                n=_validate_required_count(
                    row["n"], what=f"group_summary row 'n' ({metric}/{group_id})"
                ),
                successes=successes,
                ref_y=float(row["ref_y"]),
                cy1=float(row["cy1"]),
                cy2=float(row["cy2"]),
                # The declared x-family role, defaulted for a legacy frame
                # that predates the column - see _infer_x_role.
                x_role=_infer_x_role(row),
                **{slot: _opt(row.get(slot)) for slot in OPTIONAL_SLOTS},
                winsor_lower_percentile=_opt(row.get("winsor_lower_percentile")),
                winsor_upper_percentile=_opt(row.get("winsor_upper_percentile")),
                winsor_lower_bound=_opt(row.get("winsor_lower_bound")),
                winsor_upper_bound=_opt(row.get("winsor_upper_bound")),
                winsor_n=_opt_int(row.get("winsor_n")),
                winsor_n_lower=_opt_int(row.get("winsor_n_lower")),
                winsor_n_upper=_opt_int(row.get("winsor_n_upper")),
            )
        )
    _validate_arm_uniqueness(arms)
    return arms


def _validate_mixed_winsor_identity(
    summary: SequentialSnapshot | IntoDataFrame | Iterable[Mapping[str, Any]],
    raw: WinsorRawState,
    *,
    summary_population: Literal["assigned", "triggered"] | None,
) -> list[Mapping[str, Any]]:
    """Retain ordinary rows once and validate their explicit study/population identity."""
    from increment.winsor import winsor_refuse

    if summary_population not in ("assigned", "triggered"):
        winsor_refuse(
            "pool_mismatch",
            "Mixed percentile and ordinary metrics require explicit summary_population.",
        )
    if summary_population != raw.population:
        winsor_refuse(
            "pool_mismatch",
            "Ordinary summary population differs from the percentile raw population.",
        )
    if isinstance(summary, SequentialSnapshot):
        sequential_refuse(
            "source.invalid", "an exact snapshot requires its registered runtime policy"
        )
    frame = nw.from_native(summary, eager_only=True, pass_through=True)
    rows: list[Mapping[str, Any]]
    if isinstance(frame, nw.DataFrame):
        if "experiment_id" not in frame.columns:
            winsor_refuse(
                "pool_mismatch",
                "Mixed ordinary summary rows must carry experiment_id for study identity.",
            )
        rows = list(frame.iter_rows(named=True))
    else:
        rows = list(cast("Iterable[Mapping[str, Any]]", summary))
    for row in rows:
        if "experiment_id" not in row:
            winsor_refuse(
                "pool_mismatch",
                "Mixed ordinary summary rows must carry experiment_id for study identity.",
            )
        if str(row["experiment_id"]) != raw.study_id:
            winsor_refuse(
                "pool_mismatch",
                "Percentile raw states and ordinary summary rows must share one study.",
            )
    return rows


class _WinsorizationFields(TypedDict):
    winsor_lower_percentile: float | None
    winsor_upper_percentile: float | None
    winsor_lower_bound: float | None
    winsor_upper_bound: float | None
    winsor_control_n: int | None
    winsor_control_n_lower: int | None
    winsor_control_n_upper: int | None
    winsor_treatment_n: int | None
    winsor_treatment_n_lower: int | None
    winsor_treatment_n_upper: int | None


def _winsorization_result_fields(control: ArmStats, treatment: ArmStats) -> _WinsorizationFields:
    """Map one arm pair's winsorization metadata onto a result row."""
    arm_fields = (
        "winsor_lower_percentile",
        "winsor_upper_percentile",
        "winsor_lower_bound",
        "winsor_upper_bound",
    )
    configured = control.winsor_n is not None or treatment.winsor_n is not None
    if not configured:
        return {
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_control_n": None,
            "winsor_control_n_lower": None,
            "winsor_control_n_upper": None,
            "winsor_treatment_n": None,
            "winsor_treatment_n_lower": None,
            "winsor_treatment_n_upper": None,
        }
    if control.winsor_n is None or treatment.winsor_n is None:
        _refuse("estimation.engine.winsorization_metadata_present")
    for field in arm_fields:
        if getattr(control, field) != getattr(treatment, field):
            _refuse("estimation.engine.winsorization_metadata_differs", field=field)
    return {
        "winsor_lower_percentile": control.winsor_lower_percentile,
        "winsor_upper_percentile": control.winsor_upper_percentile,
        "winsor_lower_bound": control.winsor_lower_bound,
        "winsor_upper_bound": control.winsor_upper_bound,
        "winsor_control_n": control.winsor_n,
        "winsor_control_n_lower": control.winsor_n_lower,
        "winsor_control_n_upper": control.winsor_n_upper,
        "winsor_treatment_n": treatment.winsor_n,
        "winsor_treatment_n_lower": treatment.winsor_n_lower,
        "winsor_treatment_n_upper": treatment.winsor_n_upper,
    }


def _validate_unique_method_names(
    methods: Sequence[Method] | None,
    *,
    caller: str,
) -> None:
    """Refuse duplicate effective method labels within one metric request."""
    if methods is None:
        return
    seen: set[str] = set()
    duplicates: set[str] = set()
    for method in methods:
        (duplicates if method.name in seen else seen).add(method.name)
    if duplicates:
        _refuse(
            "estimation.engine.method_names_unique",
            caller=caller,
            duplicates=sorted(duplicates),
        )


def resolve_method_roles(
    methods: Sequence[Method],
    *,
    prefer: Callable[[Method], bool] = lambda m: m.name == "unadjusted",
    override: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    decision: Method | None = None,
) -> dict[str, Literal["decision", "sensitivity"]]:
    """Resolve which method takes the decision role.

    `override`, when given, wins verbatim. Otherwise `decision`, when given,
    forces that method into the decision role. Otherwise the first method
    satisfying `prefer` wins, falling back to `methods[0]`.
    """
    if override is not None:
        return dict(override)
    if not methods:
        return {}
    chosen = (
        decision if decision is not None else next((m for m in methods if prefer(m)), methods[0])
    )
    return {m.name: ("decision" if m is chosen else "sensitivity") for m in methods}


def _validate_methods(methods: Sequence[Method]) -> None:
    """Refuse duplicate labels and observational estimator names on a randomized path.

    Every name registered in ``ADJUSTMENTS`` (``"iptw"``/``"dml"``/``"aipw"``
    built in, plus any adjustment registered later) is an ``estimate_ate``
    dispatch key the randomized path never applies, so the label would
    always be a mislabel here -- but they are required names on the
    observational path, which is why this rule is per-call rather than a
    ``Method`` constructor rule. Shared with ``estimate_encouragement``,
    whose ``estimands=("late",)`` calls never reach ``estimate_lift``.
    """
    _validate_unique_method_names(methods, caller="estimate_lift")
    from increment.estimation.adjust import ADJUSTMENTS

    for m in methods:
        if m.name in ADJUSTMENTS:
            _refuse("estimation.engine.method_name_observational", m=m.name)


def _validate_conversion_inference(
    metrics: Sequence[Metric],
    methods_by_metric: Sequence[Sequence[Method]],
    *,
    cluster: str | None,
    priors: Sequence[Prior | None],
    sequential: bool,
) -> None:
    """Refuse an explicit ``finite_sample`` method on a metric whose request the
    finite-sample route cannot serve, before any source is read."""
    from increment.estimation.adjust import ADJUSTMENTS

    for metric, methods, prior in zip(metrics, methods_by_metric, priors, strict=True):
        explicit = [method for method in methods if method.conversion_inference == "finite_sample"]
        if not explicit:
            continue
        if metric.type not in ("conversion", "retention"):
            refuse_finite_sample_metric_type(metric.type, metric=metric.name)
        if any(method.name in ADJUSTMENTS for method in explicit):
            refuse_finite_sample_unavailable(
                metric.name,
                "the method is an observational adjustment, which fits propensity or "
                "outcome models instead of comparing raw binomial counts",
            )
        reason = finite_sample_blocker(
            cluster=cluster, prior_present=prior is not None, sequential=sequential
        )
        if reason is not None:
            refuse_finite_sample_unavailable(metric.name, reason)


def _validate_typed_direct_compatibility(
    metrics: Sequence[Metric],
    methods: Sequence[Method],
    *,
    cluster: str | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    prior: Prior | None,
    alpha: float,
    alternative: Alternative,
    null_lift: float,
    null_abs: float | None,
) -> None:
    from increment.compatibility import Unsupported
    from increment.estimation.arm_contract import (
        ARM_EVIDENCE_CONTRACT,
        AbsoluteDecisionPolicy,
        AnalysisAxes,
        ArmCompatibilityRequest,
        FamilyPolicy,
        MethodCapability,
        MetricCapabilities,
        RelativeDecisionPolicy,
        cuped_capability,
    )
    from increment.estimation.decision_types import FixedInference
    from increment.semantics.assignment import ParallelAssignment
    from increment.sequential_state import adjustment_kind

    roles = resolve_method_roles(methods)
    decision_index = next(i for i, method in enumerate(methods) if roles[method.name] == "decision")
    registration = getattr(inference, "registration", None)
    for metric in metrics:
        if metric.type in ("total", "active"):
            # Report-only metrics keep their established downstream handling;
            # the arm evidence contract covers estimator-backed types only.
            continue
        adjustment = (
            adjustment_kind(registration, metric.name) if registration is not None else None
        )
        method_capabilities = tuple(
            MethodCapability(
                role="decision" if i == decision_index else "sensitivity",
                estimator=method.name,
                variance_reduction=cuped_capability(method, adjustment=adjustment),
            )
            for i, method in enumerate(methods)
        )
        family = FamilyPolicy(kind="none", axes=(), nominal_alpha=alpha)
        decision = (
            AbsoluteDecisionPolicy(
                alternative=alternative,
                null_abs=null_abs,
                family=family,
            )
            if null_abs is not None
            else RelativeDecisionPolicy(
                alternative=alternative,
                null_lift=null_lift,
                signed_ratio=isinstance(inference, ASYMPTOTIC_PROCEDURE_POLICIES),
                family=family,
            )
        )
        typed = ArmCompatibilityRequest(
            assignment=ParallelAssignment(),
            analysis=AnalysisAxes(
                identification="randomized",
                view="total",
                segmented=False,
                completed_windows_only=False,
                population="assigned",
                variance_adjustment="none",
            ),
            dependence="cluster" if cluster is not None else "iid",
            inference=inference or FixedInference(),
            estimand="itt",
            metric=MetricCapabilities(
                metric_type=metric.type,
                value_scale="absolute" if null_abs is not None else "relative",
                winsorization=(
                    "percentile"
                    if getattr(getattr(metric, "winsorization", None), "has_percentile", False)
                    else "fixed"
                    if getattr(metric, "winsorization", None) is not None
                    else "none"
                ),
                outcome_window="unbounded" if _is_open_ended(metric) else "bounded",
                uptake_window="not_applicable",
            ),
            decision=decision,
            methods=method_capabilities,
            prior_present=prior is not None,
        )
        support = ARM_EVIDENCE_CONTRACT.runtime_support(typed)
        if not isinstance(support, Unsupported):
            continue
        from increment.compatibility import refuse_unsupported

        context: dict[str, object] = {}
        if support.refusal_code in (
            "arm.inference.cluster",
            "arm.adjustment.cluster_cuped",
            "arm.adjustment.cluster_prior",
            "arm.metric.quantile_cluster",
        ):
            context["cluster"] = cluster
        if support.refusal_code == "arm.metric.quantile_cluster":
            context["metric"] = metric.name
            context["metric_type"] = metric.type
        if support.refusal_code in ("arm.metric.quantile_cuped", "arm.metric.quantile_sequential"):
            context["metric"] = metric.name
        if support.refusal_code == "arm.adjustment.sequential_cuped":
            context["metrics"] = (metric.name,)
        if support.refusal_code == "arm.adjustment.sequential_prior":
            context["metrics"] = (metric.name,)
        refuse_unsupported(support, **context)


def _validate_percentile_winsor_readout(
    request: ReadoutRequest,
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    method_catalog: Sequence[Sequence[Method]],
    names: Sequence[str],
) -> None:
    if not names:
        return
    from increment.estimation.decision_types import FixedInference
    from increment.estimation.winsor import validate_winsor_metric

    selected = set(names)
    methods = tuple(
        method
        for config, metric_methods in zip(configs, method_catalog, strict=True)
        if config.metric.name in selected
        for method in metric_methods
    )
    configs = tuple(config for config in configs if config.metric.name in selected)
    unsupported = (
        request.view != "run"
        or bool(request.by)
        or request.cluster is not None
        or getattr(request.design, "mechanism", None) != "randomized"
        or not isinstance(request.plan.inference, FixedInference)
        or any(config.prior is not None for config in configs)
        or any(method.variance_reduction == "cuped" for method in methods)
        or any(request.plan.procedures[name].alternative != "two-sided" for name in names)
    )
    if unsupported:
        _refuse("readout.metric.percentile_winsorization", view=request.view, names=names)
    for metric in metrics:
        if metric.name in selected:
            validate_winsor_metric(metric)


UNBOUNDED_RETENTION_DAILY_REMEDY = (
    "Use the as-of view (run_asof/run_asof_lift), which reports the cumulative ratchet "
    "honestly, or declare threshold_days: [a, b] to bound the band."
)


def reject_unbounded_retention(
    metrics: Sequence[Metric], fn_name: str, *, remedy: str, supported_view: str
) -> None:
    """Refuse a retention band with no completion date on a non-cumulative view.

    The one implementation of this rule: the shared readout gate and the
    day-axis facade both call it, so every route raises the same code before
    any source read.
    """
    unbounded = [m.name for m in metrics if isinstance(m, RetentionMetric) and m.band[1] is None]
    if unbounded:
        _refuse(
            "breakout.retention.unbounded",
            fn_name=fn_name,
            names=unbounded,
            remedy=remedy,
            supported_view=supported_view,
        )


def reject_completed_windows_on_unbounded_retention(
    metrics: Sequence[Metric], fn_name: str
) -> None:
    """Refuse completed windows over a retention band with no completion date.

    No unit's window ever completes, so the completion gate would admit nothing.
    Like `reject_unbounded_retention`, the shared readout gate and the day-axis
    facade both call this, so every route raises one code before any source read.
    """
    open_bands = [m.name for m in metrics if isinstance(m, RetentionMetric) and m.band[1] is None]
    if open_bands:
        _refuse("breakout.retention.completion", fn_name=fn_name, names=open_bands)


def validate_readout_engine(request: ReadoutRequest) -> None:
    """Validate metric/method compatibility before source moments are read."""
    metrics = tuple(request.metrics)
    configs = tuple(request.configs)
    view = request.view
    by = tuple(request.by)
    cluster = request.cluster
    design = request.design
    mechanism = getattr(design, "mechanism", None)
    method_catalog = request.estimation_methods
    methods = [method for metric_methods in method_catalog for method in metric_methods]
    if mechanism != "observational":
        for metric_methods in method_catalog:
            _validate_methods(metric_methods)
    _validate_conversion_inference(
        metrics,
        method_catalog,
        cluster=cluster,
        priors=[config.prior for config in configs],
        sequential=isinstance(request.plan.inference, SEQUENTIAL_POLICIES),
    )
    winsorized_names = [
        metric.name for metric in metrics if getattr(metric, "winsorization", None) is not None
    ]
    if view == "daily":
        if winsorized_names:
            _refuse(
                "readout.metric.daily_winsorization",
                view=view,
                names=winsorized_names,
            )
        reject_unbounded_retention(
            metrics, view, remedy=UNBOUNDED_RETENTION_DAILY_REMEDY, supported_view="asof"
        )
        # Daily values are descriptive; skip inferential compatibility rules.
        return

    # A compliance-only request never reads outcome moments, so an outcome band's
    # completion date is irrelevant to it.
    estimands = request.estimands
    compliance_only = estimands is not None and set(estimands) == {"compliance"}
    if view == "asof" and request.completion_policy and not compliance_only:
        reject_completed_windows_on_unbounded_retention(metrics, view)

    percentile_names = [
        metric.name
        for metric in metrics
        if (
            (winsorization := getattr(metric, "winsorization", None)) is not None
            and getattr(winsorization, "has_percentile", False)
        )
    ]
    _validate_percentile_winsor_readout(request, metrics, configs, method_catalog, percentile_names)

    if cluster is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        if isinstance(request.plan.inference, SEQUENTIAL_POLICIES):
            refuse_unsupported(Unsupported("arm.inference.cluster"), cluster=cluster)
        if any(getattr(method, "variance_reduction", None) == "cuped" for method in methods):
            refuse_unsupported(Unsupported("arm.adjustment.cluster_cuped"), cluster=cluster)
        if any(getattr(config, "prior", None) is not None for config in configs):
            refuse_unsupported(Unsupported("arm.adjustment.cluster_prior"), cluster=cluster)
        for metric in metrics:
            if getattr(metric, "type", None) == "quantile":
                refuse_unsupported(
                    Unsupported("arm.metric.quantile_cluster"),
                    metric=metric.name,
                    metric_type=metric.type,
                    cluster=cluster,
                )

    for metric, procedure in ((metric, request.plan.procedures[metric.name]) for metric in metrics):
        if getattr(metric, "type", None) != "quantile":
            continue
        if by:
            _refuse("readout.metric.quantile_breakout", metric=metric.name)
        if (
            procedure.alternative != "two-sided"
            or getattr(procedure, "null_lift", 0.0) != 0.0
            or getattr(procedure, "null_abs", None) is not None
        ):
            _refuse(
                "readout.metric.quantile_alternative",
                metric=metric.name,
            )

    if view == "breakout" and not isinstance(request.plan.inference, SEQUENTIAL_POLICIES):
        # Fixed-horizon segment rows build no shifted null. Plan-bound ExperimentMetric margins
        # resolve onto the compiled procedure, not the Metric, so check both. Sequential breakout
        # tests each cell against its registered null, which must equal the compiled one.
        procedures = request.plan.procedures
        declared = [
            metric.name
            for metric in metrics
            if getattr(metric, "margin", None) is not None
            or getattr(metric, "margin_abs", None) is not None
            or getattr(procedures[metric.name], "null_lift", 0.0) != 0.0
            or getattr(procedures[metric.name], "null_abs", None) is not None
        ]
        if declared:
            _refuse("readout.margin.breakout", names=declared)

    if view == "run" and mechanism in {"randomized", "observational"}:
        value_scale = request.value_scale
        if value_scale and mechanism == "randomized":
            _refuse("readout.randomized.value_scale")
        if value_scale and mechanism == "observational":
            selected = {metric.name for metric in metrics}
            unknown = set(value_scale) - selected
            if unknown:
                _refuse(
                    "readout.value_scale.unknown_metric",
                    unknown=unknown,
                    selected=selected,
                )

    if estimands is not None and tuple(estimands) != ("itt",) and mechanism != "encouragement":
        _refuse(
            "readout.assignment.estimands",
            estimands=estimands,
            mechanism=mechanism,
        )


def _is_open_ended(metric: Metric) -> bool:
    """A metric whose per-unit value can still change between analyses."""
    if metric.type == "ratio":
        return metric.numerator.window_days is None or metric.denominator.window_days is None
    if metric.type == "retention":
        # The band's right edge, not window_days - vacated on retention.
        return metric.band[1] is None
    return metric.window_days is None


def _warn_if_open_ended_sequential(metrics: Iterable[Metric]) -> None:
    """Warn per open-ended metric that the sequential guarantee is approximate.

    Shared by every sequential entry point that estimates from arm moments:
    unit totals still accrue between analyses on an open-ended metric, so a
    boundary derived for finalized per-unit values only approximately covers,
    and the estimand drifts with the enrollment mix.
    """
    for m in metrics:
        if not _is_open_ended(m):
            continue
        if m.type == "ratio":
            windows = "one or both windows unset"
        elif m.type == "retention":
            windows = "unbounded band"
        else:
            windows = "window_days=None"
        fix = (
            "Bound the band (threshold_days: [a, b])"
            if m.type == "retention"
            else "Set window_days"
        )
        _warn(
            "estimation.engine.open_ended_sequential",
            metric_name=m.name,
            windows=windows,
            fix=fix,
            skip_file_prefixes=PACKAGE_FRAMES,
        )


# Orchestrator


def _mean_abs_from_log(mean: float, se_log: float) -> tuple[float, float]:
    """Absolute-scale (mean, SE) from an unadjusted arm's mean and its
    log-scale SE.

    Exact, not approximate, for the SE: ``mean * se_log === sqrt(var/n)``
    algebraically, recovering the absolute-scale SE bit-for-bit (max
    observed error 3.5e-18). ``mean`` passes through untouched; callers
    must pass the arm's actual mean, not one reconstructed from a
    log-scale point estimate, since ``abs_t - abs_c`` would otherwise
    amplify round-trip error for a small lift. The CUPED branch does not
    route through here: its log-scale SE carries the pooled anchor's cross
    term, so the identity above does not hold there.
    """
    return mean, mean * se_log


def _ratio_abs_diff_se(arm: ArmStats) -> tuple[float, float]:
    """Absolute-scale (R, SE) for a ratio arm, DIRECT form (not ``R *
    sqrt(var_log_r)``).

    The two forms are algebraically identical but not equal in floating
    point: ``var_log_r`` carries a ``var_n/n_bar^2`` term that ``R^2``
    cancels analytically, not numerically. On the package's pinned
    ``n=200``/``d_bar=10`` test configuration, the relative route divides
    by an underflowed ``n_bar**2``/``n_bar*d_bar`` and raises
    ``ZeroDivisionError`` once ``n_bar`` drops below roughly ``1e-161``;
    this direct form divides only by ``d_bar``, so it survives there,
    though below roughly ``1e-159`` it can itself correctly underflow to
    exactly zero - a legitimately negligible variance, not an error.
    That range is precisely what the signed/near-zero-numerator ratio
    metric exists to rescue, though through ``estimate_lift`` the rescue
    window is only ``n_bar`` in roughly ``(1e-161, 1e-159)``: below it
    the relative path's ``ZeroDivisionError`` aborts the whole call first.
    """
    n_bar, d_bar, var_n, var_d, cov_nd = ratio_moments(arm)
    return ratio_abs_diff_se(n_bar, d_bar, var_n, var_d, cov_nd, arm.n)


def ratio_abs_diff_se(
    num_bar: float, den_bar: float, var_num: float, var_den: float, cov_num_den: float, n: int
) -> tuple[float, float]:
    """Absolute-scale (R, SE) for R = num/den, factored out so the
    cluster-robust LATE reduction reuses the identical delta method for
    cluster-grain ratios without duplicating the formula.

    Delegates to :func:`~increment.estimation.variance.ratio_abs_diff_se`,
    which evaluates the stable ratio-residual form, and re-raises its
    refusal under this module's registered code so the public refusal
    contract on this call path is unchanged.
    """
    try:
        return variance_ratio_abs_diff_se(
            num_bar, den_bar, var_num, var_den, cov_num_den, n, what="ratio absolute-scale SE"
        )
    except ValueError as exc:
        _refuse("estimation.engine.ratio.negative_variance", reason=str(exc))


def ratio_pair_cov(
    r1: float,
    r2: float,
    den_bar: float,
    cov_num1_num2: float,
    cov_num1_den: float,
    cov_num2_den: float,
    var_den: float,
    n: int,
) -> float:
    """Delta-method covariance of two ratios R1 = N1/D, R2 = N2/D sharing
    the same denominator D, linearized the same way as
    :func:`ratio_abs_diff_se` (whose single-ratio variance is the N1 == N2
    special case: r1=r2=R, cov_num1_num2=var_num, cov_num1_den=
    cov_num2_den=cov_num_den reproduces it exactly).

    Combines the cluster-robust LATE reduction's two cluster-grain ratio
    estimators into the Wald ratio's covariance term, mirroring the
    unit-grain ``cov_yd()`` role in ``_late_additive``.
    """
    return (1.0 / n) * (
        cov_num1_num2 / den_bar**2
        - r2 * cov_num1_den / den_bar**2
        - r1 * cov_num2_den / den_bar**2
        + r1 * r2 * var_den / den_bar**2
    )


def _raw_stats_evidence(
    result: LiftEstimate, hypothesis: Any
) -> tuple[PValueEvidence | None, DecisionFailure | None]:
    """Fixed-horizon p-value evidence from a row's own raw (point, se): the
    fallback decision statistic once always_valid/asymptotic_mean/absolute
    margin paths don't apply."""
    from increment.estimation.decision_types import DecisionFailure, PValueEvidence

    if result.relative_confidence_set is not None or result.relative_unavailable_reason is not None:
        try:
            p_value = result.p_value()
        except CodedError as exc:
            return None, DecisionFailure(
                hypothesis,
                "evidence.p_value.unavailable",
                {
                    "metric": result.metric,
                    "group_id": result.group_id,
                    "reason": result.relative_unavailable_reason or str(exc),
                },
            )
        if p_value is None:
            return None, DecisionFailure(
                hypothesis,
                "evidence.p_value.unavailable",
                {
                    "metric": result.metric,
                    "group_id": result.group_id,
                    "reason": "missing_sampling_statistic",
                },
            )
        return PValueEvidence(hypothesis, result.method, p_value, "relative_confidence_set"), None

    if result.lift is None:
        return None, DecisionFailure(
            hypothesis,
            "evidence.p_value.unavailable",
            {
                "metric": result.metric,
                "group_id": result.group_id,
                "reason": "missing_raw_stats",
            },
        )
    raw_point, raw_se = result.lift.log_mean, result.lift.log_se
    if raw_point is None or raw_se is None or not raw_se > 0.0:
        return None, DecisionFailure(
            hypothesis,
            "evidence.p_value.unavailable",
            {
                "metric": result.metric,
                "group_id": result.group_id,
                "reason": "missing_raw_stats",
            },
        )
    null = (
        result.null_abs
        if result.scale == "linear" and result.null_abs is not None
        else result.null_lift
        if result.scale == "linear"
        else math.log1p(result.null_lift)
    )
    z = (raw_point - null) / raw_se
    if result.reference_kind != "t":
        from scipy.stats import norm

        if result.alternative == "greater":
            p_value = float(norm.sf(z))
        elif result.alternative == "less":
            p_value = float(norm.cdf(z))
        else:
            p_value = float(2.0 * min(norm.cdf(z), norm.sf(z)))
        reference = "normal"
    else:
        from scipy.stats import t

        if result.alternative == "greater":
            p_value = float(t.sf(z, result.reference_df))
        elif result.alternative == "less":
            p_value = float(t.cdf(z, result.reference_df))
        else:
            p_value = float(2.0 * t.sf(abs(z), result.reference_df))
        reference = f"t_{result.reference_df:g}"
    return PValueEvidence(hypothesis, result.method, p_value, reference), None


def merge_decision_computations(
    computations: Sequence[DecisionComputation[LiftEstimate]],
) -> DecisionComputation[LiftEstimate]:
    """Combine disjoint estimator bundles into one DecisionComputation,
    dropping any retained sequential snapshot."""
    from increment.estimation.decision_types import DecisionComputation

    results: list[Any] = []
    evidence: dict[Any, Any] = {}
    failures: dict[Any, Any] = {}
    for computation in computations:
        results.extend(computation.results)
        evidence.update(computation.evidence)
        failures.update(computation.failures)
    return DecisionComputation(results=tuple(results), evidence=evidence, failures=failures)


def _lift_decision_bundle(  # noqa: PLR0915
    results: Sequence[LiftEstimate],
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    sequential_snapshot=None,
) -> DecisionComputation[LiftEstimate]:
    """Attach one typed decision value to every decision presentation row."""
    from increment.estimation.decision_types import (
        ArmHypothesisKey,
        DecisionComputation,
        DecisionFailure,
        EValueEvidence,
        PValueEvidence,
    )

    evidence: dict[Any, Any] = {}
    failures: dict[Any, DecisionFailure] = {}
    seen: set[Any] = set()
    output_results = list(results)
    for result in output_results:
        if result.method_role != "decision":
            continue
        hypothesis = ArmHypothesisKey(result.metric, result.group_id, result.estimand)
        if hypothesis in seen:
            evidence.pop(hypothesis, None)
            failures[hypothesis] = DecisionFailure(
                hypothesis,
                "evidence.duplicate_hypothesis",
                {"metric": result.metric, "group_id": result.group_id},
            )
            continue
        seen.add(hypothesis)
        if result.reference_kind == "confidence_set":
            assert result.confidence_set is not None
            inference_spec = result.confidence_set.raw.inference
            if inference_spec.status == "experimental":
                failures[hypothesis] = DecisionFailure(
                    hypothesis,
                    "evidence.experimental_reference",
                    {
                        "metric": result.metric,
                        "method": result.method,
                        "status": inference_spec.status,
                        "qualification": result.confidence_set.qualification,
                    },
                )
                continue
            evidence[hypothesis] = PValueEvidence(
                hypothesis, result.method, result.p_value(), result.confidence_set.method
            )
            continue
        if result.quantile_p_value is not None:
            evidence[hypothesis] = PValueEvidence(
                hypothesis,
                result.method,
                result.quantile_p_value,
                "quantile_inversion",
            )
            continue
        if result.inference in ("always_valid", "asymptotic_mean"):
            from increment.sequential_state import sequential_refuse

            sequential = result.sequential_result
            if sequential is None:
                sequential_refuse(
                    "continuation.legacy", "sequential decision lacks raw likelihood evidence"
                )
            from increment.estimation.decision_types import AsymptoticSequentialEvidence
            from increment.estimation.sequential_result import AsymptoticSequentialResult

            if isinstance(sequential, AsymptoticSequentialResult):
                evidence[hypothesis] = AsymptoticSequentialEvidence(
                    hypothesis=hypothesis, method=result.method, result=sequential
                )
                continue
            evidence[hypothesis] = EValueEvidence(
                hypothesis=hypothesis,
                method=result.method,
                log_e=sequential.log_e,
                process="raw_likelihood_v1",
                checkpoint=sequential.checkpoint,
                certificate=sequential.certificate,
            )
            continue
        if result.null_abs is not None:
            # The margin test must use the same reference as the margin
            # interval, so a t reference needs its own degrees of freedom.
            # Only an unknown df leaves the p-value undefined.
            abs_kind = result.abs_reference_kind or result.reference_kind
            abs_df = result.abs_reference_df
            if (abs_kind == "t" and abs_df is None) or (
                abs_kind != "t" and result.n_clusters is not None
            ):
                failures[hypothesis] = DecisionFailure(
                    hypothesis,
                    "evidence.p_value.unavailable",
                    {
                        "metric": result.metric,
                        "group_id": result.group_id,
                        "reason": "clustered_absolute_margin",
                        "reference_kind": result.reference_kind,
                        "reference_df": result.reference_df,
                        "display": (
                            "absolute-margin p-value unavailable under a t sampling reference "
                            "(cluster-robust or Welch); decide on the relative scale or drop the margin"
                        ),
                    },
                )
                continue
            if result.abs_diff is None or result.abs_se is None or not result.abs_se > 0.0:
                failures[hypothesis] = DecisionFailure(
                    hypothesis,
                    "evidence.p_value.unavailable",
                    {
                        "metric": result.metric,
                        "group_id": result.group_id,
                        "reason": "missing_absolute_stats",
                    },
                )
                continue
            if abs_kind == "t":
                from scipy.stats import t as _student_t

                dist = _student_t(df=abs_df)
                reference_name = "student_t"
            else:
                from scipy.stats import norm as dist  # type: ignore[assignment]

                reference_name = "normal"
            z_abs = (result.null_abs - result.abs_diff) / result.abs_se
            if result.alternative == "greater":
                p_value = float(dist.cdf(z_abs))
            elif result.alternative == "less":
                p_value = float(dist.sf(z_abs))
            else:
                p_value = float(2.0 * min(dist.cdf(z_abs), dist.sf(z_abs)))
            evidence[hypothesis] = PValueEvidence(
                hypothesis, result.method, p_value, reference_name
            )
            continue
        if result.reference_kind == "binomial":
            bset = result.binomial_set
            assert bset is not None, "validated: reference_kind='binomial' rows carry a set"
            p_value = result.p_value()
            if p_value is None:
                failures[hypothesis] = DecisionFailure(
                    hypothesis,
                    "evidence.p_value.unavailable",
                    {
                        "metric": result.metric,
                        "group_id": result.group_id,
                        "reason": "missing_sampling_statistic",
                    },
                )
            else:
                evidence[hypothesis] = PValueEvidence(
                    hypothesis, result.method, p_value, bset.method
                )
            continue
        evidence_row, failure_row = _raw_stats_evidence(result, hypothesis)
        if failure_row is not None:
            failures[hypothesis] = failure_row
        else:
            evidence[hypothesis] = evidence_row
    return DecisionComputation(
        results=tuple(output_results),
        evidence=evidence,
        failures=failures,
        sequential_snapshot=sequential_snapshot,
    )


def _infer_clustered_lift_result(
    contrast: tuple[ArmStats, ArmStats],
    method: Method,
    strategy: _LiftVarianceStrategy,
    alpha: float,
    alternative: str,
    method_role: Literal["decision", "sensitivity"],
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
) -> LiftEstimate:
    from fractions import Fraction

    from increment.estimation.inference import _joint_additive_bounds
    from increment.estimation.results import _joint_reference_from_exact, relative_confidence_set

    treatment, control = contrast
    assert strategy.absolute_moments is not None and strategy.dof is not None
    abs_t, se_t, abs_c, se_c = strategy.absolute_moments
    abs_diff, abs_se = abs_t - abs_c, math.hypot(se_t, se_c)
    vt, vc = Fraction(se_t) ** 2, Fraction(se_c) ** 2
    reference, reason = _joint_reference_from_exact(
        a=Fraction(abs_t) - Fraction(abs_c),
        c=abs_c,
        var_a=vt + vc,
        var_c=vc,
        cov_ac=-vc,
        kind="t",
        df=strategy.dof,
    )
    confidence_set = (
        relative_confidence_set(reference, alpha=alpha, alternative=alternative)
        if reference is not None
        else None
    )
    point = abs_diff / abs_c if abs_c != 0.0 else None
    displayed = (
        confidence_set.estimate()
        if confidence_set is not None
        else Estimate(value=point)
        if point is not None and math.isfinite(point)
        else None
    )
    abs_df = (
        welch_satterthwaite_df(
            (se_t / abs_se) ** 2,
            float(treatment.n - 1),
            (se_c / abs_se) ** 2,
            float(control.n - 1),
        )
        if abs_se > 0.0
        else None
    )
    if reference is not None:
        abs_diff = reference.a
        abs_se = math.sqrt(reference.var_a)
    lower, upper = _joint_additive_bounds(abs_diff, abs_se, alpha, alternative, abs_df)
    return LiftEstimate(
        metric=treatment.metric,
        group_id=treatment.group_id,
        method=method.name,
        method_role=method_role,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        preferred_direction=preferred_direction,
        sampling_available=True,
        lift=displayed,
        scale="linear",
        relative_confidence_set=confidence_set,
        relative_unavailable_reason=reason,
        abs_diff=abs_diff,
        abs_se=abs_se if abs_se > 0.0 else None,
        abs_lb=lower,
        abs_ub=upper,
        abs_reference_kind="t" if abs_df is not None else None,
        abs_reference_df=abs_df,
        abs_alpha=_alpha_eff_for(alternative, alpha) if lower is not None else None,
        n_clusters=strategy.n_clusters,
        dof=strategy.dof,
        reference_kind="t",
        reference_df=strategy.dof,
        note="Independent-arm covariance; additive Welch and relative fixed-t working approximations.",
    )


def _welch_arm_ns(
    contrast: tuple[ArmStats, ArmStats], strategy: _LiftVarianceStrategy, prior: Prior | None
) -> tuple[int, int] | None:
    """Arm sizes selecting the prior-free Welch-Satterthwaite sampling reference."""
    if not getattr(strategy.variance_model, "supports_welch_reference", False):
        return None
    treatment, control = contrast
    return treatment.n, control.n


def _additive_welch_df(
    contrast: tuple[ArmStats, ArmStats], abs_se_t: float, abs_se_c: float
) -> float:
    """Welch-Satterthwaite df of the additive difference from each arm's own
    absolute-scale SE and size. Only centered moments enter, so translating
    every outcome by a constant leaves it unchanged."""
    treatment, control = contrast
    abs_se = math.hypot(abs_se_t, abs_se_c)
    return welch_satterthwaite_df(
        (abs_se_t / abs_se) ** 2,
        float(treatment.n - 1),
        (abs_se_c / abs_se) ** 2,
        float(control.n - 1),
    )


def _infer_lift_result(
    contrast: tuple[ArmStats, ArmStats],
    method: Method,
    moments: tuple[float, float, float, float, float, float, float] | None,
    strategy: _LiftVarianceStrategy,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    method_role: Literal["decision", "sensitivity"],
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
) -> tuple[LiftEstimate | None, DecisionFailure | None]:
    """Infer, decorate, and translate a single contrast's guarded result."""
    treatment, control = contrast
    if moments is None:
        result = _infer_clustered_lift_result(
            contrast,
            method,
            strategy,
            alpha,
            alternative,
            method_role,
            null_lift,
            null_abs,
            preferred_direction,
        )
        return result.model_copy(update=_winsorization_result_fields(control, treatment)), None
    log_rr, se_t, se_c, abs_t, abs_se_t, abs_c, abs_se_c = moments
    abs_diff = abs_t - abs_c
    abs_se: float | None = math.hypot(abs_se_t, abs_se_c)
    if abs_se == 0.0:
        # Both arms clamped to exactly zero variance (degenerate ratio arm).
        abs_se = None
    try:
        arm_ns = _welch_arm_ns(contrast, strategy, prior)
        result = infer_lift(
            metric=treatment.metric,
            group_id=treatment.group_id,
            method=method.name,
            log_rr=log_rr,
            se_t=se_t,
            se_c=se_c,
            prior=None if prior is not None and arm_ns is not None else prior,
            alpha=alpha,
            alternative=alternative,
            inference_spec=inference,
            method_role=method_role,
            n_comparison=treatment.n + control.n,
            abs_diff=abs_diff,
            abs_se=abs_se,
            null_lift=null_lift,
            null_abs=null_abs,
            preferred_direction=preferred_direction,
            dof=strategy.dof,
            abs_dof=(
                _additive_welch_df(contrast, abs_se_t, abs_se_c)
                # The additive sidecar is dropped when the primary reference
                # is t and this is None, so it must follow arm_ns as well.
                if (strategy.dof is not None or arm_ns is not None) and abs_se is not None
                else None
            ),
            n_clusters=strategy.n_clusters,
            arm_ns=arm_ns,
        )
        if prior is not None and arm_ns is not None:
            result = result.model_copy(
                update=posterior_fields(
                    log_rr,
                    math.hypot(se_t, se_c),
                    prior,
                    alpha=alpha,
                    alternative=alternative,
                    scale="log",
                    preferred_direction=preferred_direction,
                    null_lift=null_lift,
                    null_abs=null_abs,
                )
            )
    except LiftGuardError as exc:
        result = None
        guard_reason = exc.reason
        guard_display = str(exc)
    if result is None:
        if method_role != "decision":
            return None, None
        from increment.estimation.decision_types import ArmHypothesisKey, DecisionFailure

        hypothesis = ArmHypothesisKey(treatment.metric, treatment.group_id, "itt")
        return None, DecisionFailure(
            hypothesis,
            "estimation.engine.lift_guard",
            {
                "metric": treatment.metric,
                "group_id": treatment.group_id,
                "method": method.name,
                "reason": guard_reason or "lift guard",
                "display": guard_display,
            },
        )
    update: dict[str, Any] = dict(_winsorization_result_fields(control, treatment))
    if inference is None and strategy.ratio_precision_note is not None:
        # Fixed-horizon ratio rows carry the advisory; infer_lift sets no note of its own.
        update["note"] = strategy.ratio_precision_note
    return result.model_copy(update=update), None


@dataclass(frozen=True, slots=True)
class _PreparedLiftEstimation:
    """Validated inputs and arm inventory shared by every contrast."""

    metrics: tuple[Metric, ...]
    methods: tuple[Method, ...]
    metric_types: Mapping[str, str]
    control_by_metric: Mapping[str, ArmStats]
    treatment_arms: tuple[ArmStats, ...]
    resolved_method_roles: Mapping[str, Literal["decision", "sensitivity"]]


@dataclass(frozen=True, slots=True)
class _LiftMethodStrategy:
    """One method's moment operation, fixed before a contrast's loop."""

    method: Method
    use_cuped: bool


@dataclass(frozen=True, slots=True)
class _LiftVarianceStrategy:
    """One contrast's variance policy, resolved before its method loop."""

    variance_model: Any
    metric_type: str
    dof: float | None
    n_clusters: int | None
    absolute_moments: tuple[float, float, float, float] | None
    methods: tuple[_LiftMethodStrategy, ...]
    # Advisory for a unit-grain ratio contrast whose denominator mean is
    # poorly resolved; None for every other contrast.
    ratio_precision_note: str | None


def _prepare_lift_estimation(
    metrics: Sequence[Metric],
    summary: IntoDataFrame | Iterable[Mapping[str, Any]],
    control_group: str,
    methods: list[Method] | None,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    null_lift: float,
    null_abs: float | None,
    cluster: str | None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
) -> _PreparedLiftEstimation | DecisionComputation[LiftEstimate]:
    """Resolve request policy and validate the arm inventory before contrasts."""
    if methods is None:
        methods = [Method(name="unadjusted")]
    if methods == []:
        from increment.estimation.decision_types import (
            ArmHypothesisKey,
            DecisionComputation,
            DecisionFailure,
        )

        arms = _df_to_arms(summary)
        controls = {arm.metric for arm in arms if arm.group_id == control_group}
        failures: dict[Any, DecisionFailure] = {
            (hypothesis := ArmHypothesisKey(arm.metric, arm.group_id, "itt")): DecisionFailure(
                hypothesis,
                "estimation.engine.no_decision_method",
                {"metric": arm.metric, "group_id": arm.group_id},
            )
            for arm in arms
            if arm.group_id != control_group and arm.metric in controls
        }
        return DecisionComputation(results=(), evidence={}, failures=failures)
    resolved_method_roles: dict[str, Literal["decision", "sensitivity"]] = dict(method_roles or {})
    if method_roles is None:
        resolved_method_roles = resolve_method_roles(methods)
    _validate_methods(methods)
    _validate_typed_direct_compatibility(
        metrics,
        methods,
        cluster=cluster,
        inference=inference,
        prior=prior,
        alpha=alpha,
        alternative=cast("Literal['two-sided', 'greater', 'less']", alternative),
        null_lift=null_lift,
        null_abs=null_abs,
    )
    _validate_conversion_inference(
        metrics,
        [methods] * len(metrics),
        cluster=cluster,
        priors=[prior] * len(metrics),
        sequential=inference is not None,
    )
    code = sequential_support_refusal(
        SequentialSupportRequest(
            inference=inference,
            uses_cuped=any(method.variance_reduction == "cuped" for method in methods),
        )
    )
    if code is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported(code), metrics=tuple(m.name for m in metrics))
    if cluster is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        if inference is not None:
            refuse_unsupported(Unsupported("arm.inference.cluster"), cluster=cluster)
        if any(m.variance_reduction == "cuped" for m in methods):
            refuse_unsupported(Unsupported("arm.adjustment.cluster_cuped"), cluster=cluster)
        if prior is not None:
            refuse_unsupported(Unsupported("arm.adjustment.cluster_prior"), cluster=cluster)

    arms = _df_to_arms(summary)
    metric_types = {m.name: m.type for m in metrics}
    arm_metrics = {a.metric for a in arms}
    unknown = arm_metrics - metric_types.keys()
    if unknown:
        _refuse(
            "estimation.engine.metric_found_group",
            unknown=sorted(unknown),
            metric_types=sorted(metric_types),
        )
    if inference is not None:
        _warn_if_open_ended_sequential(m for m in metrics if m.name in arm_metrics)
    control_arms = [a for a in arms if a.group_id == control_group]
    if not control_arms:
        known_groups = sorted({a.group_id for a in arms})
        _refuse(
            "estimation.engine.control_group_found",
            control_group=control_group,
            known_groups=known_groups,
        )
    control_by_metric = {a.metric: a for a in control_arms}
    treatment_arms = tuple(a for a in arms if a.group_id != control_group)
    return _PreparedLiftEstimation(
        metrics=tuple(metrics),
        methods=tuple(methods),
        metric_types=metric_types,
        control_by_metric=control_by_metric,
        treatment_arms=treatment_arms,
        resolved_method_roles=resolved_method_roles,
    )


def _select_lift_variance_strategy(
    treatment: ArmStats,
    control: ArmStats,
    metric_type: str,
    control_group: str,
    cluster: str | None,
    method_strategies: tuple[_LiftMethodStrategy, ...],
) -> _LiftVarianceStrategy:
    """Resolve one contrast's grain, capability, and hoisted moments."""
    dof: float | None = None
    n_clusters: int | None = None
    if cluster is not None:
        if metric_type not in ("mean", "conversion", "retention", "ratio"):
            from increment.compatibility import Unsupported, refuse_unsupported

            refuse_unsupported(
                Unsupported("arm.metric.quantile_cluster"),
                metric=treatment.metric,
                metric_type=metric_type,
                cluster=cluster,
            )
        variance_model = VARIANCE_MODELS.get("cluster")
        k_t, k_c = treatment.n, control.n
        n_clusters = k_t + k_c
        check_total_clusters(treatment.metric, cluster, n_clusters)
        if k_t < 2 or k_c < 2:
            from increment.estimation.encouragement import ARM_NEEDS_TWO

            refuse(
                ARM_NEEDS_TWO,
                metric=treatment.metric,
                cluster=cluster,
                k_t=k_t,
                k_c=k_c,
            )
        # One fixed t reference for the entire relative test inversion; additive inference is Welch.
        dof = float(min(k_t - 1, k_c - 1))
    else:
        if metric_type in ("mean", "conversion", "retention") and treatment.ref_den is not None:
            _refuse(
                "estimation.engine.cluster.denominator",
                metric=treatment.metric,
                metric_type=metric_type,
                treatment_group=treatment.group_id,
                control_group=control_group,
                cluster=None,
            )
        variance_model = VARIANCE_MODELS.get(metric_type)
    absolute_moments = None
    # The unadjusted absolute sidecar serves the methods that do not adjust;
    # a CUPED-adjusted ratio builds its own from the adjusted components.
    if cluster is not None or (
        metric_type == "ratio" and any(not item.use_cuped for item in method_strategies)
    ):
        abs_t, abs_se_t = _ratio_abs_diff_se(treatment)
        abs_c, abs_se_c = _ratio_abs_diff_se(control)
        absolute_moments = (abs_t, abs_se_t, abs_c, abs_se_c)
    return _LiftVarianceStrategy(
        variance_model=variance_model,
        metric_type=metric_type,
        dof=dof,
        n_clusters=n_clusters,
        absolute_moments=absolute_moments,
        methods=method_strategies,
        ratio_precision_note=(
            _ratio_denominator_precision_note(treatment, control)
            if cluster is None and metric_type == "ratio"
            else None
        ),
    )


def _ratio_denominator_precision_note(treatment: ArmStats, control: ArmStats) -> str | None:
    """Advisory text for a ratio contrast whose denominator mean is poorly
    resolved in either arm, naming each flagged arm's statistic and the
    threshold; ``None`` when both arms resolve it well enough."""
    flagged = [
        f"{arm.group_id}={statistic:.3g}"
        for arm in (treatment, control)
        if (statistic := ratio_denominator_precision(arm)) > RATIO_DENOMINATOR_PRECISION_THRESHOLD
    ]
    if not flagged:
        return None
    return (
        "ratio_denominator_precision: the denominator is too noisy at this sample size for "
        "the interval to hold its stated level (relative standard error of the denominator mean "
        f"{', '.join(flagged)} exceeds {RATIO_DENOMINATOR_PRECISION_THRESHOLD:.3g}). Expect "
        "under-coverage; add units or check the denominator for heavy tails before deciding."
    )


def _binomial_eligible(
    metric_type: str,
    cluster: str | None,
    method_strategy: _LiftMethodStrategy,
    treatment: ArmStats,
    control: ArmStats,
) -> bool:
    """Whether this (metric, method) contrast is a genuinely-binary,
    unadjusted, unit-grain arm pair whose route ``conversion_route`` chooses
    from its counts: the finite-sample binomial method (see ``binomial_rr.py``)
    or the delta-method contrast.

    A declared ``conversion``/``retention`` metric type is the sole
    structural provenance signal (a 0/1-per-unit fact by construction --
    matching moments alone never proves this). CUPED uses adjusted
    asymptotic inference rather than raw Bernoulli sufficient statistics.
    Clustered conversion/retention uses the linear joint Fieller path with
    cluster-robust covariance and a fixed-t working reference, not this
    binomial construction. An attached uptake moment (``arm.sum_d``) does not
    disqualify eligibility either: it describes a different random
    variable over the same units and is stripped before either route
    reads the arm (see ``_without_unused_binomial_uptake``).
    """
    return (
        metric_type in ("conversion", "retention")
        and cluster is None
        and not method_strategy.use_cuped
        and all(arm.ref_den is None for arm in (treatment, control))
    )


def _without_unused_binomial_covariate(arm: ArmStats) -> ArmStats:
    """Drop a materialized covariate from an unadjusted binomial arm."""
    # An unrecognized declaration has no view to ask; it is not a covariate.
    if arm.x_role not in X_ROLE_VARIABLES or not arm.moments.has("x"):
        return arm
    return arm.model_copy(
        update={
            "ref_x": None,
            "cx1": None,
            "cx2": None,
            "cxy": None,
            "x_role": None,
            "cxden": None,
            "cxd": None,
        }
    )


def _without_unused_binomial_uptake(arm: ArmStats) -> ArmStats:
    """Drop uptake (first-stage compliance) moments from a binomial-
    eligible ITT arm.

    An encouragement design's uptake facts do not change the ITT's
    sufficient statistics for a binary outcome (x_c, n_c, x_t, n_t); only
    the assignment-level y family does. Stripping them here keeps the
    independent-binomial counts available to the ITT, while the
    design's own uptake/compliance/LATE estimation (encouragement.py)
    reads the untouched arm from the original summary.
    """
    if arm.sum_d is None:
        return arm
    return arm.model_copy(update={"sum_d": None, "cyd": None, "cy2d": None, "cxd": None})


def _binomial_abs_sidecar(treatment: ArmStats, control: ArmStats) -> tuple[float, float | None]:
    """Wald absolute-scale ``(diff, SE)`` from the arms' own mean/variance,
    independent of the exact binomial method (which needs neither and
    tolerates zero events/variance that this classical SE cannot).

    ``None`` SE when either arm has fewer than 2 units (no ddof=1
    variance -- the binomial method itself is admissible at ``n=1``, but
    this additive Wald sidecar is a distinct, unrelated feature with its
    own long-standing sample-size floor) or both arms are simultaneously
    zero-variance, matching the ``abs_se == 0.0 -> None`` convention
    ``_infer_lift_result`` already uses for a degenerate ratio arm.
    """
    if treatment.n < 2 or control.n < 2:
        return treatment.mean_y() - control.mean_y(), None
    t_summary, c_summary = treatment.to_summary(), control.to_summary()
    abs_diff = t_summary.mean - c_summary.mean
    abs_se = math.hypot(
        math.sqrt(t_summary.var / t_summary.n), math.sqrt(c_summary.var / c_summary.n)
    )
    return abs_diff, (abs_se if abs_se > 0.0 else None)


def _binomial_abs_bounds(abs_diff: float, abs_se: float, alpha_eff: float) -> tuple[float, float]:
    """Two-sided Wald bounds for the binomial row's additive sidecar at its
    central-equivalent ``alpha_eff``, through the shared survival-form
    critical value: an extreme alpha resolves to a finite interval or
    refuses with ``estimation.tails.unresolvable`` instead of overflowing."""
    from scipy.stats import norm

    crit = two_sided_critical_value(norm.isf, alpha_eff, what="binomial additive sidecar")
    return wald_bounds(abs_diff, crit, abs_se, what="binomial additive sidecar")


def _validate_binomial_request(
    treatment: ArmStats,
    *,
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    null_lift: float,
) -> tuple[Alternative, float]:
    """Validate the request-level inputs every route of an eligible conversion contrast
    shares and return ``(alternative, alpha_eff)``.

    Request-level misconfigurations (an informative prior; a sequential guarantee) raise
    immediately, matching how ``infer_lift`` already raises immediately for the analogous
    cluster/sequential/prior combinations on the log-Normal path -- these are not per-arm
    data guards. ``alpha_eff`` is the persisted central-equivalent display level: a
    directional alternative doubles ``alpha``, while each inversion still spends ``alpha``.
    """
    from increment.estimation.inference import INFER_ATE_NULL_LIFT_FINITE

    alternative = binomial_rr.validate_alternative(alternative)
    if not math.isfinite(null_lift):
        refuse(INFER_ATE_NULL_LIFT_FINITE, null_lift=null_lift)
    if null_lift <= -1.0:
        raise InvalidRequestError(
            f"null_lift must be > -1.0 (log scale), got {null_lift}",
            code="estimation.inference.null_lift_log",
            context={"null_lift": null_lift},
        )
    if inference is not None:
        raise InvalidRequestError(
            "sequential inference is unsupported for the exact binomial method "
            f"(metric={treatment.metric!r}, group_id={treatment.group_id!r}): it "
            "supports fixed-horizon inference only",
            code="estimation.binomial.fixed_horizon_required",
            context={"metric": treatment.metric, "group_id": treatment.group_id},
        )
    if prior is not None:
        raise InvalidRequestError(
            "an informative prior is unsupported for the exact binomial method "
            f"(metric={treatment.metric!r}, group_id={treatment.group_id!r}): it "
            "is a frequentist test-inversion with no posterior a prior could act on",
            code="estimation.binomial.prior_unsupported",
            context={"metric": treatment.metric, "group_id": treatment.group_id},
        )
    alpha_eff = alpha if alternative == "two-sided" else 2.0 * alpha
    if not (0.0 < alpha_eff < 1.0):
        raise InvalidRequestError(
            f"alpha={alpha!r} doubled to {alpha_eff!r} for a directional alternative "
            "is not representable in (0, 1)",
            code="estimation.binomial.alpha_doubling_unrepresentable",
            context={"alpha": alpha, "alternative": alternative},
        )
    return alternative, alpha_eff


def _binomial_arm(arm: ArmStats) -> ArmStats:
    """``arm`` as the binomial reading sees it: its unused covariate and uptake moments dropped."""
    return _without_unused_binomial_uptake(_without_unused_binomial_covariate(arm))


def _contrast_counts(
    treatment: ArmStats, control: ArmStats, metric_type: str
) -> tuple[int, int, int, int]:
    """``(x_c, n_c, x_t, n_t)`` of an eligible conversion contrast, reconstructed from its
    arm moments before any route is chosen, so a corrupted arm refuses identically under
    every ``conversion_inference``."""
    x_c, n_c = binary_counts(_binomial_arm(control), metric_type)
    x_t, n_t = binary_counts(_binomial_arm(treatment), metric_type)
    return x_c, n_c, x_t, n_t


def _contrast_of_counts(
    treatment: ArmStats, control: ArmStats, counts: tuple[int, int, int, int]
) -> tuple[ArmStats, ArmStats]:
    """The ``(treatment, control)`` arms of an eligible contrast the delta method reads: the
    binomial arms with their y family formed from the counts (`canonical_bernoulli_arm`), so a
    producer's accepted rounding in the stored moments never reaches the interval, and the
    interval is a function of the four counts alone."""
    x_c, _, x_t, _ = counts
    return (
        canonical_bernoulli_arm(_binomial_arm(treatment), x_t),
        canonical_bernoulli_arm(_binomial_arm(control), x_c),
    )


def _binomial_data_failure(
    treatment: ArmStats, exc: binomial_rr.BinomialDataError, method_role: str
) -> tuple[None, DecisionFailure | None]:
    """A per-arm data/numerical guard (bad reconstructed counts, an unrepresentable tail)
    becomes a keyed ``DecisionFailure`` for the decision method, matching
    ``LiftGuardError``'s existing soft-failure treatment."""
    from increment.estimation.decision_types import ArmHypothesisKey, DecisionFailure

    if method_role != "decision":
        return None, None
    hypothesis = ArmHypothesisKey(treatment.metric, treatment.group_id, "itt")
    return None, DecisionFailure(hypothesis, exc.code, dict(exc.context))


def _infer_binomial_lift_result(
    contrast: tuple[ArmStats, ArmStats],
    counts: tuple[int, int, int, int],
    method: Method,
    alpha: float,
    alternative: Alternative,
    method_role: Literal["decision", "sensitivity"],
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
) -> tuple[LiftEstimate | None, DecisionFailure | None]:
    """Exact independent-binomial risk-ratio inference for one eligible
    (conversion/retention, unadjusted, unit-grain) contrast routed finite-sample.

    Bypasses the log-Normal delta method and its ``log_se >= .5``
    admission rule entirely for this contrast -- that guard is a
    computational admission rule for the Normal approximation, not a
    scientific boundary this exact method needs. See ``binomial_rr.py``
    for the method and its coverage argument. ``counts`` and the request
    were validated by the caller.
    """
    treatment, control = contrast
    x_c, n_c, x_t, n_t = counts
    alpha_eff = alpha if alternative == "two-sided" else 2.0 * alpha
    try:
        ci = binomial_rr.confidence_interval(
            x_c, n_c, x_t, n_t, alpha=alpha, alternative=alternative, null_r=1.0 + null_lift
        )
        point = binomial_rr.point_lift(x_c, n_c, x_t, n_t)
    except binomial_rr.BinomialDataError as exc:
        return _binomial_data_failure(treatment, exc, method_role)

    lift_lower, lift_upper = binomial_rr.to_lift_bounds(ci)
    level = math.fsum((1.0, -alpha_eff))
    binomial_set = BinomialConfidenceSet(
        lower=lift_lower,
        upper=lift_upper,
        alpha=alpha_eff,
        level=level,
        geometry=ci.geometry,
        method=BINOMIAL_METHOD,
        numerical_qualification=BINOMIAL_NUMERICAL_QUALIFICATION,
        x_c=x_c,
        n_c=n_c,
        x_t=x_t,
        n_t=n_t,
        nuisance_beta=binomial_rr.nuisance_beta(alpha),
        decision_alpha=alpha,
    )
    lift_estimate = None
    if point is not None:
        lift_estimate = Estimate(
            value=point,
            lb=lift_lower,
            ub=lift_upper,
            open_side="upper" if lift_upper is None else None,
            level=level,
            alpha=alpha_eff,
        )
    abs_diff, abs_se = _binomial_abs_sidecar(treatment, control)
    abs_lb = abs_ub = None
    if abs_se is not None:
        abs_lb, abs_ub = _binomial_abs_bounds(abs_diff, abs_se, alpha_eff)
    result = LiftEstimate(
        metric=treatment.metric,
        group_id=treatment.group_id,
        method=method.name,
        method_role=method_role,
        inference="fixed",
        alternative=alternative,
        null_lift=null_lift,
        preferred_direction=preferred_direction,
        sampling_available=True,
        lift=lift_estimate,
        scale="linear",
        abs_diff=abs_diff,
        abs_se=abs_se,
        null_abs=null_abs,
        abs_lb=abs_lb,
        abs_ub=abs_ub,
        abs_alpha=alpha_eff if abs_lb is not None else None,
        reference_kind="binomial",
        binomial_set=binomial_set,
        note=binomial_rr.precision_note(ci),
    )
    return result.model_copy(update=_winsorization_result_fields(control, treatment)), None


class _NonPositiveMeanFailure(LiftGuardError):
    """A per-arm non-positive mean blocks only the log-scale relative
    lift for this contrast; the additive sidecar, computed from the same
    arm summaries before the guard fired, rides along so the caller can
    build a real additive-only row instead of losing the cell entirely.
    """

    def __init__(
        self,
        message: str,
        *,
        abs_diff: float,
        abs_se_t: float,
        abs_se_c: float,
    ) -> None:
        super().__init__(message, reason="non_positive_mean")
        self.abs_diff = abs_diff
        self.abs_se_t = abs_se_t
        self.abs_se_c = abs_se_c


def _ratio_log_lift(num_c: float, den_c: float, num_t: float, den_t: float) -> float:
    """Preserve the joint contrast before rounding finite positive component means."""
    ratio = (Fraction(num_t) * Fraction(den_c)) / (Fraction(num_c) * Fraction(den_t))
    relative = ratio - 1
    if abs(relative) <= 0.5:
        return math.log1p(float(relative))
    # A normal float preserves relative precision; subnormal ratios need exact logs.
    try:
        ratio_float = float(ratio)
    except OverflowError:
        ratio_float = 0.0
    if math.isfinite(ratio_float) and ratio_float >= sys.float_info.min:
        return math.log(ratio_float)
    return math.log(ratio.numerator) - math.log(ratio.denominator)


def _compute_lift_arm_moments(
    treatment: ArmStats,
    control: ArmStats,
    method_strategy: _LiftMethodStrategy,
    strategy: _LiftVarianceStrategy,
) -> tuple[float, float, float, float, float, float, float]:
    """Return the joint log lift, arm log-scale SEs, and absolute sidecars.

    Mean paths compare raw means; ratio paths preserve the exact cross ratio.
    Clustered inference bypasses these log moments.
    """
    if method_strategy.use_cuped and strategy.metric_type == "ratio":
        return _ratio_cuped_arm_moments(treatment, control)
    if method_strategy.use_cuped:
        fit = fit_cuped([control, treatment])
        c_summary, t_summary = fit.adjust()
        abs_t, abs_se_t = t_summary.mean, math.sqrt(t_summary.var / t_summary.n)
        abs_c, abs_se_c = c_summary.mean, math.sqrt(c_summary.var / c_summary.n)
        try:
            check_positive_mean(
                control.metric, control.group_id, c_summary.mean, what="CUPED-adjusted arm mean"
            )
            check_positive_mean(
                treatment.metric,
                treatment.group_id,
                t_summary.mean,
                what="CUPED-adjusted arm mean",
            )
        except InvalidRequestError as exc:
            raise _NonPositiveMeanFailure(
                str(exc), abs_diff=abs_t - abs_c, abs_se_t=abs_se_t, abs_se_c=abs_se_c
            ) from exc
        log_rr = stable_log_ratio(c_summary.mean, t_summary.mean)
        # The log ratio's arm scores carry the shared pooled anchor: each arm
        # is linearised at its own slope (theta only when the adjusted means
        # agree), so the per-arm variance is NOT the adjusted-mean variance
        # that serves the absolute contrast below, where the anchor cancels.
        slope_t, slope_c = fit.contrast_slopes(
            treatment, control, 1.0 / t_summary.mean, -1.0 / c_summary.mean
        )
        se_t = se_log_mean(fit.residual_var(treatment, slope_t), t_summary.mean, t_summary.n)
        se_c = se_log_mean(fit.residual_var(control, slope_c), c_summary.mean, c_summary.n)
    elif type(strategy.variance_model) is MeanVarianceModel:
        # The exact built-in mean model can share one summary per arm. Custom
        # protocol implementations and subclasses retain their model-owned SE.
        c_summary = control.to_summary()
        t_summary = treatment.to_summary()
        # sqrt(var/n) equals mean*se_log (see _mean_abs_from_log); computing it directly keeps
        # the absolute SE available when the positivity guard below fails.
        abs_c, abs_se_c = c_summary.mean, math.sqrt(c_summary.var / c_summary.n)
        abs_t, abs_se_t = t_summary.mean, math.sqrt(t_summary.var / t_summary.n)
        try:
            check_positive_mean(control.metric, control.group_id, c_summary.mean)
            check_positive_mean(treatment.metric, treatment.group_id, t_summary.mean)
        except InvalidRequestError as exc:
            raise _NonPositiveMeanFailure(
                str(exc), abs_diff=abs_t - abs_c, abs_se_t=abs_se_t, abs_se_c=abs_se_c
            ) from exc
        se_c = se_log_mean(c_summary.var, c_summary.mean, c_summary.n)
        se_t = se_log_mean(t_summary.var, t_summary.mean, t_summary.n)
        log_rr = stable_log_ratio(c_summary.mean, t_summary.mean)
    else:
        try:
            log_c, se_c = strategy.variance_model.log_mean_se(control)
            log_t, se_t = strategy.variance_model.log_mean_se(treatment)
        except InvalidRequestError as exc:
            if (
                exc.code == "estimation.variance.metric_group_log"
                and strategy.absolute_moments is not None
            ):
                abs_t, abs_se_t, abs_c, abs_se_c = strategy.absolute_moments
                raise _NonPositiveMeanFailure(
                    str(exc), abs_diff=abs_t - abs_c, abs_se_t=abs_se_t, abs_se_c=abs_se_c
                ) from exc
            raise
        if strategy.absolute_moments is None:
            mean_c = control.to_summary().mean
            mean_t = treatment.to_summary().mean
            log_rr = stable_log_ratio(mean_c, mean_t)
            abs_t, abs_se_t = _mean_abs_from_log(mean_t, se_t)
            abs_c, abs_se_c = _mean_abs_from_log(mean_c, se_c)
        else:
            abs_t, abs_se_t, abs_c, abs_se_c = strategy.absolute_moments
            if strategy.metric_type == "ratio":
                log_rr = _ratio_log_lift(
                    control.mean_y(), control.mean_den(), treatment.mean_y(), treatment.mean_den()
                )
            else:
                log_rr = log_t - log_c
    return log_rr, se_t, se_c, abs_t, abs_se_t, abs_c, abs_se_c


class _PosteriorGuardFields(TypedDict):
    posterior_available: Literal[False]
    posterior_reason_code: str
    posterior_reason_context: dict[str, Any]


def _posterior_fields_for_contrast(
    contrast: tuple[ArmStats, ArmStats],
    method_strategy: _LiftMethodStrategy,
    strategy: _LiftVarianceStrategy,
    prior: Prior | None,
    *,
    alpha: float,
    alternative: str,
    preferred_direction: PreferredDirection | None,
    null_lift: float,
    null_abs: float | None,
) -> PosteriorFields | _PosteriorGuardFields:
    """Return a supported posterior, or its existing working-likelihood guard."""
    if prior is None:
        return PosteriorFields()
    treatment, control = contrast
    try:
        log_rr, se_t, se_c, *_ = _compute_lift_arm_moments(
            treatment, control, method_strategy, strategy
        )
        infer_lift(
            metric=treatment.metric,
            group_id=treatment.group_id,
            method=method_strategy.method.name,
            log_rr=log_rr,
            se_t=se_t,
            se_c=se_c,
            alpha=alpha,
            alternative=alternative,
            method_role="decision",
        )
        return posterior_fields(
            log_rr,
            math.hypot(se_t, se_c),
            prior,
            alpha=alpha,
            alternative=alternative,
            scale="log",
            preferred_direction=preferred_direction,
            null_lift=null_lift,
            null_abs=null_abs,
        )
    except LiftGuardError as exc:
        return _PosteriorGuardFields(
            posterior_available=False,
            posterior_reason_code="estimation.engine.lift_guard",
            posterior_reason_context={
                "metric": treatment.metric,
                "group_id": treatment.group_id,
                "method": method_strategy.method.name,
                "reason": exc.reason,
                "display": str(exc),
            },
        )
    except InvalidRequestError as exc:
        if exc.code != "estimation.armstats.arm_stats.least_compute_metric":
            raise
        return _PosteriorGuardFields(
            posterior_available=False,
            posterior_reason_code=exc.code,
            posterior_reason_context={
                "metric": treatment.metric,
                "group_id": treatment.group_id,
                "method": method_strategy.method.name,
                "reason": exc.code,
                "display": str(exc),
            },
        )


def _nonpositive_mean_additive_row(
    contrast: tuple[ArmStats, ArmStats],
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
    exc: _NonPositiveMeanFailure,
    *,
    welch: bool,
) -> LiftEstimate:
    """The additive-scale row for a contrast whose per-arm mean blocks
    the log-scale relative lift: the relative estimate is genuinely
    undefined here (math.log of a non-positive number, not merely
    numerically fragile like the clustered Fieller path's own
    covariance-availability check), but the additive difference is a
    real, well-defined Wald estimate from the same arm summaries --
    reported here rather than discarded, mirroring the clustered
    lift=None/relative_unavailable_reason row shape.

    The additive reference is the one the ordinary sidecar uses for the
    same arms: a Welch-Satterthwaite ``t`` over each arm's own SE and size
    when ``welch`` (see ``_welch_arm_ns``), Normal otherwise. Every input is
    a centered moment, so translating all outcomes -- which is how an arm
    mean reaches zero or below -- changes neither the interval nor its
    reference.
    """
    from increment.estimation.inference import _joint_additive_bounds

    treatment, control = contrast
    abs_se = math.hypot(exc.abs_se_t, exc.abs_se_c)
    abs_dof = (
        _additive_welch_df(contrast, exc.abs_se_t, exc.abs_se_c) if welch and abs_se > 0.0 else None
    )
    lower, upper = _joint_additive_bounds(exc.abs_diff, abs_se, alpha, alternative, abs_dof)
    return LiftEstimate(
        metric=treatment.metric,
        group_id=treatment.group_id,
        method=method.name,
        method_role=method_role,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        preferred_direction=preferred_direction,
        sampling_available=True,
        lift=None,
        scale="linear",
        relative_confidence_set=None,
        relative_unavailable_reason="nonpositive_arm_mean",
        abs_diff=exc.abs_diff,
        abs_se=abs_se if abs_se > 0.0 else None,
        abs_lb=lower,
        abs_ub=upper,
        abs_reference_kind=("t" if abs_dof is not None else "normal") if abs_se > 0.0 else None,
        abs_reference_df=abs_dof,
        abs_alpha=_alpha_eff_for(alternative, alpha) if lower is not None else None,
        reference_kind="normal",
        reference_df=None,
        note="Arm mean is non-positive; the log-scale relative lift is undefined. "
        + (
            "Additive Welch-t working approximation."
            if abs_dof is not None
            else "Additive Normal working approximation."
        ),
    )


def _nonpositive_mean_outcome(
    exc: _NonPositiveMeanFailure,
    contrast: tuple[ArmStats, ArmStats],
    strategy: _LiftVarianceStrategy,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    prior: Prior | None,
    alpha: float,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
) -> tuple[LiftEstimate | None, DecisionFailure | None]:
    """Return an additive row or the prior-backed decision's guarded failure."""
    treatment, control = contrast
    if prior is not None and method_role == "decision":
        from increment.estimation.decision_types import ArmHypothesisKey, DecisionFailure

        hypothesis = ArmHypothesisKey(treatment.metric, treatment.group_id, "itt")
        return None, DecisionFailure(
            hypothesis,
            "estimation.engine.lift_guard",
            {
                "metric": treatment.metric,
                "group_id": treatment.group_id,
                "method": method.name,
                "reason": exc.reason,
                "display": str(exc),
            },
        )
    return (
        _nonpositive_mean_additive_row(
            contrast,
            method,
            method_role,
            alpha,
            alternative,
            null_lift,
            null_abs,
            preferred_direction,
            exc,
            welch=_welch_arm_ns(contrast, strategy, prior) is not None,
        ),
        None,
    )


def _ratio_cuped_arm_moments(
    treatment: ArmStats, control: ArmStats
) -> tuple[float, float, float, float, float, float, float]:
    """A CUPED-adjusted ratio metric's log ratio, per-arm log-scale SEs, and
    absolute-scale sidecars.

    Both scales read the adjusted components through the SAME delta-method
    reductions an unadjusted ratio uses -- only the five moments they are
    given differ. The relative and absolute projections carry different
    Jacobians through the shared pooled anchor, so each asks the fit for its
    own adjusted moments rather than reusing one set.
    """
    fit = fit_ratio_cuped([control, treatment])
    for arm in (control, treatment):
        num_bar, den_bar = fit.adjusted_components(arm)
        check_positive_mean(
            arm.metric, arm.group_id, num_bar, what="CUPED-adjusted ratio numerator"
        )
        check_positive_mean(
            arm.metric, arm.group_id, den_bar, what="CUPED-adjusted ratio denominator"
        )
    rel_t, rel_c = fit.relative_moments(treatment, control)
    log_t, se_t = _adjusted_log_mean_se(treatment, rel_t)
    log_c, se_c = _adjusted_log_mean_se(control, rel_c)
    abs_t, abs_c = fit.absolute_moments(treatment, control)
    ratio_t, abs_se_t = _adjusted_abs_diff_se(abs_t)
    ratio_c, abs_se_c = _adjusted_abs_diff_se(abs_c)
    log_rr = _ratio_log_lift(rel_c.num_bar, rel_c.den_bar, rel_t.num_bar, rel_t.den_bar)
    return log_rr, se_t, se_c, ratio_t, abs_se_t, ratio_c, abs_se_c


def _adjusted_log_mean_se(arm: ArmStats, moments: AdjustedRatioMoments) -> tuple[float, float]:
    return ratio_log_mean_se(
        moments.num_bar,
        moments.den_bar,
        moments.var_num,
        moments.var_den,
        moments.cov_num_den,
        moments.n,
        group_id=arm.group_id,
        metric=arm.metric,
    )


def _adjusted_abs_diff_se(moments: AdjustedRatioMoments) -> tuple[float, float]:
    return ratio_abs_diff_se(
        moments.num_bar,
        moments.den_bar,
        moments.var_num,
        moments.var_den,
        moments.cov_num_den,
        moments.n,
    )


def _raw_winsor_lift_bundle(
    metrics,
    control,
    raw_outcomes,
    winsor_references,
    methods,
    method_roles,
    alpha,
    null_lift,
    null_abs,
    preferred_direction,
):
    from increment.estimation.decision_types import (
        ArmHypothesisKey,
        DecisionComputation,
        DecisionFailure,
    )
    from increment.estimation.winsor import estimate_winsor_lift
    from increment.winsor import winsor_refuse

    resolved = methods
    roles = dict(method_roles or {})
    if method_roles is None and resolved:
        roles = resolve_method_roles(resolved)
    results, failures = [], {}
    for metric in metrics:
        raw = raw_outcomes[metric.name]
        raw.arm(control)
        if (
            raw.metric != metric.name
            or raw.quantile != metric.winsorization.upper_percentile
            or raw.support != metric.winsorization.support
            or raw.inference != metric.winsorization.inference
        ):
            winsor_refuse(
                "pool_mismatch", "Metric declaration differs from raw construction state."
            )
        for arm in raw.arms:
            if arm.group_id == control:
                continue
            if not resolved:
                key = ArmHypothesisKey(metric.name, arm.group_id, "itt")
                failures[key] = DecisionFailure(
                    key,
                    "estimation.engine.no_decision_method",
                    {"metric": metric.name, "group_id": arm.group_id},
                )
            for method in resolved:
                results.append(
                    estimate_winsor_lift(
                        raw,
                        control,
                        arm.group_id,
                        alpha=alpha,
                        method=method.name,
                        method_role=cast(
                            "Literal['decision', 'sensitivity']",
                            roles.get(method.name, "sensitivity"),
                        ),
                        null_lift=null_lift,
                        null_abs=null_abs,
                        preferred_direction=preferred_direction,
                        reference=(winsor_references or {}).get((metric.name, arm.group_id)),
                    )
                )
    bundle = _lift_decision_bundle(results, inference=None)
    return DecisionComputation(
        results=bundle.results, evidence=bundle.evidence, failures={**bundle.failures, **failures}
    )


def _estimate_registered_sequential(
    metrics: Sequence[Metric],
    summary: SequentialSnapshot | IntoDataFrame | Iterable[Mapping[str, Any]],
    control_group: str,
    methods: list[Method] | None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None,
    prior: Prior | None,
    alpha: float | None,
    alternative: str | None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily,
    null_lift: float | None,
    null_abs: float | None,
    cluster: str | None,
) -> DecisionComputation[LiftEstimate]:
    from increment.estimation.decision_types import DecisionComputation
    from increment.estimation.sequential_runtime import estimate_sequential, validate_engine_request

    if methods == []:
        return DecisionComputation(results=(), evidence={}, failures={})

    if method_roles and any(role != "decision" for role in method_roles.values()):
        sequential_refuse(
            "route.unsupported",
            "raw sequential inference does not support nondecision method roles",
        )

    if cluster is not None or null_abs is not None:
        sequential_refuse(
            "route.unsupported",
            "raw sequential inference requires independent units and a relative null",
        )
    from increment.sequential_state import validate_sequential_methods

    for metric in metrics:
        validate_sequential_methods(inference.registration, metric.name, methods or (), prior=prior)
    _validate_conversion_inference(
        metrics,
        [methods or ()] * len(metrics),
        cluster=cluster,
        priors=[prior] * len(metrics),
        sequential=True,
    )
    if not isinstance(summary, SequentialSnapshot):
        sequential_refuse(
            "source.invalid",
            "rounded moment rows cannot establish exact sequential state; supply a registered snapshot",
        )
    if control_group != summary.registration.control_group:
        sequential_refuse("source.invalid", "control identity differs from registration")

    validate_engine_request(
        summary,
        metrics,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
    )
    return estimate_sequential(summary, inference)


def _asymptotic_lift_outcome(
    contrast: tuple[ArmStats, ArmStats],
    method_strategy: _LiftMethodStrategy,
    strategy: _LiftVarianceStrategy,
    method_role: Literal["decision", "sensitivity"],
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
) -> tuple[LiftEstimate | None, DecisionFailure | None]:
    """The delta-method contrast every unadjusted mean metric takes: log risk ratio
    against a Welch-Satterthwaite ``t`` reference, with the additive Wald sidecar."""
    treatment, control = contrast
    try:
        moments = (
            _compute_lift_arm_moments(treatment, control, method_strategy, strategy)
            if strategy.n_clusters is None
            else None
        )
    except _NonPositiveMeanFailure as exc:
        return _nonpositive_mean_outcome(
            exc,
            contrast,
            strategy,
            method_strategy.method,
            method_role,
            prior,
            alpha,
            alternative,
            null_lift,
            null_abs,
            preferred_direction,
        )
    return _infer_lift_result(
        contrast=contrast,
        method=method_strategy.method,
        moments=moments,
        strategy=strategy,
        prior=prior,
        alpha=alpha,
        alternative=alternative,
        inference=inference,
        method_role=method_role,
        null_lift=null_lift,
        null_abs=null_abs,
        preferred_direction=preferred_direction,
    )


def _lift_for_method(  # noqa: PLR0913
    contrast: tuple[ArmStats, ArmStats],
    metric_type: str,
    cluster: str | None,
    method_strategy: _LiftMethodStrategy,
    strategy: _LiftVarianceStrategy,
    method_role: Literal["decision", "sensitivity"],
    prior: Prior | None,
    alpha: float,
    alternative: str,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    null_lift: float,
    null_abs: float | None,
    preferred_direction: PreferredDirection | None,
    route_alpha: float | None = None,
) -> tuple[LiftEstimate | None, DecisionFailure | None]:
    """One (contrast, method) row.

    Auto conversion routing is selected from the counts independently of
    the prior. The prior can add a separately guarded posterior, but cannot
    replace the prior-free sampling construction.
    """
    treatment, control = contrast
    method = method_strategy.method
    eligible = (
        _binomial_eligible(metric_type, cluster, method_strategy, treatment, control)
        and inference is None
        and (prior is None or method.conversion_inference == "auto")
    )
    if not eligible:
        if method.conversion_inference == "finite_sample":
            if metric_type not in ("conversion", "retention"):
                refuse_finite_sample_metric_type(metric_type, metric=treatment.metric)
            refuse_finite_sample_unavailable(
                treatment.metric,
                finite_sample_blocker(
                    cluster=cluster,
                    prior_present=prior is not None,
                    sequential=inference is not None,
                )
                or "the arm moments carry a ratio denominator or a CUPED adjustment",
            )
        return _asymptotic_lift_outcome(
            contrast,
            method_strategy,
            strategy,
            method_role,
            prior,
            alpha,
            alternative,
            inference,
            null_lift,
            null_abs,
            preferred_direction,
        )
    valid_alternative, alpha_eff = _validate_binomial_request(
        treatment,
        prior=None if method.conversion_inference == "auto" else prior,
        alpha=alpha,
        alternative=alternative,
        inference=inference,
        null_lift=null_lift,
    )
    try:
        counts = _contrast_counts(treatment, control, metric_type)
    except binomial_rr.BinomialDataError as exc:
        return _binomial_data_failure(treatment, exc, method_role)
    # A multiplicity procedure may read this row's p-value at a smaller family level
    # (``route_alpha``, in ``alpha``'s convention): the route is the one valid at the smallest
    # level the row can be decided at, never a looser one.
    route_level = alpha_eff
    if route_alpha is not None and method_role == "decision":
        floor = route_alpha if valid_alternative == "two-sided" else 2.0 * route_alpha
        route_level = min(alpha_eff, floor)
    route = route_for_counts(
        *counts, tail_alpha=route_level / 2.0, mode=method.conversion_inference
    )
    if route == "finite_sample":
        result, failure = _infer_binomial_lift_result(
            contrast,
            counts,
            method,
            alpha,
            valid_alternative,
            method_role,
            null_lift,
            null_abs,
            preferred_direction,
        )
        if result is not None and prior is not None:
            posterior_contrast = _contrast_of_counts(treatment, control, counts)
            result = result.model_copy(
                update=_posterior_fields_for_contrast(
                    posterior_contrast,
                    method_strategy,
                    strategy,
                    prior,
                    alpha=alpha,
                    alternative=valid_alternative,
                    preferred_direction=preferred_direction,
                    null_lift=null_lift,
                    null_abs=null_abs,
                )
            )
        return result, failure
    return _asymptotic_lift_outcome(
        _contrast_of_counts(treatment, control, counts),
        method_strategy,
        strategy,
        method_role,
        prior,
        alpha,
        alternative,
        inference,
        null_lift,
        null_abs,
        preferred_direction,
    )


def estimate_lift(  # noqa: PLR0913
    metrics: Sequence[Metric],
    summary: SequentialSnapshot
    | IntoDataFrame
    | Iterable[Mapping[str, Any]],  # group_summary rows for ONE experiment
    control_group: str,  # REQUIRED explicit control
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    alpha: float | None = None,
    alternative: str | None = None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    null_lift: float | None = None,
    null_abs: float | None = None,
    preferred_direction: PreferredDirection | None = None,
    cluster: str | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    raw_outcomes: Mapping[str, WinsorRawState] | None = None,
    winsor_references: Mapping[tuple[str, str], BootstrapReference] | None = None,
    *,
    summary_population: Literal["assigned", "triggered"] | None = None,
) -> DecisionComputation[LiftEstimate]:
    """Estimate relative lift for every (metric x method x non-control arm).

    Percentile winsorization requires ``raw_outcomes`` with every cutoff-pool
    arm and matching inference specification. It returns a confidence set without a
    posterior; percentile-clipped summaries without ``raw_outcomes`` refuse before estimation.
    Mixed percentile and ordinary requests additionally require the typed
    ``summary_population`` keyword. Ordinary summary rows identify the study
    through ``experiment_id`` but do not provide trustworthy population
    metadata, so population is never inferred from them.

    Ratio metrics use a first-order delta-method SE (``RatioVarianceModel``):
    under a heavily right-skewed denominator the interval undercovers at
    small n (see that class). The carried moments cannot reveal skew, but
    they do reveal how precisely each arm's denominator mean is resolved:
    a fixed-horizon unit-grain ratio row whose arm exceeds
    ``RATIO_DENOMINATOR_PRECISION_THRESHOLD`` on ``ratio_denominator_precision``
    carries a ``ratio_denominator_precision`` note naming the arm, the
    statistic and the threshold. The note is advisory: the interval is
    reported unchanged.

    An unadjusted, unclustered, fixed-horizon conversion or retention row
    takes the route ``Method.conversion_inference`` selects whether or not
    a prior is declared. Under ``"auto"`` the four per-arm success and
    failure counts and the tail allocation alone decide it
    (``conversion_route.route_for_counts``): dense counts take the same
    delta-method contrast and Welch sampling reference as unadjusted means;
    other counts take the finite-sample independent-binomial inversion.
    Only the ``"binomial"`` rows carry the finite-sample guarantee. A
    delta-method row is a function of the four counts alone: the arms'
    moments are formed from the counts, not read as stored, so the same
    counts give the same sampling interval through every ingress. A prior
    adds a separate posterior where its working likelihood is supported.

    ``control_group`` is required - the engine never guesses by sort
    order. ``prior``/``alpha``/``alternative``/``null_lift``/``null_abs``/
    ``preferred_direction`` forward to fixed-horizon inference. Registered
    sequential policies consume an exact SequentialSnapshot and its committed
    null, direction, allocation and predictive law. Explicit effect-policy
    arguments must agree with registration. They never consume rounded arm
    moments or infer a sampling law from available columns.

    ``cluster``, when set, declares the experiment's randomization-grain
    column: *summary* must carry the clustered two-stage collapse
    (``group_summary(..., cluster=...)``), where each row's ``n`` is
    the arm's cluster count and the ratio-moment fields hold each
    cluster's outcome total over its denominator (cluster size for a mean-family
    metric, the metric's own per-cluster denominator total for a ratio
    metric). Each cluster must belong to exactly one arm; preaggregated
    summaries cannot establish that assignment property. The engine uses
    ``ClusterVarianceModel`` with each arm's Bessel-corrected between-cluster
    variance. Additive intervals use Welch-Satterthwaite degrees of freedom;
    relative Fieller sets use the fixed ``min(K_T - 1, K_C - 1)`` reference.
    Both are working approximations, not exact small-cluster guarantees;
    point estimates are unchanged. Refuses sequential ``inference``, a CUPED method, and
    quantile metrics; each arm still needs at least 2 clusters, and below 40
    total clusters emits the qualified-reference warning (see
    ``check_total_clusters``).
    """
    return _estimate_lift(
        metrics,
        summary,
        control_group,
        methods,
        prior,
        alpha,
        alternative,
        inference,
        null_lift,
        null_abs,
        preferred_direction,
        cluster,
        method_roles,
        raw_outcomes,
        winsor_references,
        summary_population=summary_population,
    )


def _estimate_lift(  # noqa: PLR0913
    metrics: Sequence[Metric],
    summary: SequentialSnapshot
    | IntoDataFrame
    | Iterable[Mapping[str, Any]],  # group_summary rows for ONE experiment
    control_group: str,  # REQUIRED explicit control
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    alpha: float | None = None,
    alternative: str | None = None,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None = None,
    null_lift: float | None = None,
    null_abs: float | None = None,
    preferred_direction: PreferredDirection | None = None,
    cluster: str | None = None,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    raw_outcomes: Mapping[str, WinsorRawState] | None = None,
    winsor_references: Mapping[tuple[str, str], BootstrapReference] | None = None,
    *,
    summary_population: Literal["assigned", "triggered"] | None = None,
    route_alpha: float | None = None,
) -> DecisionComputation[LiftEstimate]:
    """``estimate_lift`` with the multiplicity routing level its families pass.

    ``route_alpha`` is the smallest level (in ``alpha``'s convention) a multiplicity procedure
    can later decide a decision row's p-value at, in ``(0, 1]``: a conversion decision row
    takes the delta-method route only if its counts are dense at that level too, so an
    approximate p-value is never read at a tail it was not validated at. Omitted, the row's
    own ``alpha`` decides. Only the package's own families set it; the public function
    never does, so a caller cannot move a row's ``reference_kind`` apart from a family.
    """
    if route_alpha is not None and not 0.0 < route_alpha <= 1.0:
        _refuse("estimation.engine.route_alpha", route_alpha=route_alpha)
    percentile_metrics = [
        m for m in metrics if getattr(getattr(m, "winsorization", None), "has_percentile", False)
    ]
    if inference is not None and not percentile_metrics:
        return _estimate_registered_sequential(
            metrics,
            summary,
            control_group,
            methods,
            method_roles,
            prior,
            alpha,
            alternative,
            inference,
            null_lift,
            null_abs,
            cluster,
        )
    alpha = 0.05 if alpha is None else alpha
    alternative = "two-sided" if alternative is None else alternative
    null_lift = 0.0 if null_lift is None else null_lift

    if percentile_metrics:
        from increment.estimation.winsor import validate_winsor_metric
        from increment.winsor import WinsorRawState, winsor_refuse

        winsor_methods = methods if methods is not None else [Method(name="unadjusted")]
        _validate_methods(winsor_methods)

        for metric in percentile_metrics:
            if raw_outcomes is None or not isinstance(
                raw_outcomes.get(metric.name), WinsorRawState
            ):
                winsor_refuse(
                    "raw_state_required",
                    "Percentile inference requires typed exact raw unit state.",
                )
            validate_winsor_metric(metric)
        assert raw_outcomes is not None
        if (
            cluster is not None
            or inference is not None
            or prior is not None
            or alternative != "two-sided"
            or any(m.variance_reduction == "cuped" for m in methods or [])
        ):
            winsor_refuse(
                "design_unsupported",
                "Pooled winsor inference requires unadjusted fixed two-sided unit contrasts.",
            )
        raw_states = [raw_outcomes[metric.name] for metric in percentile_metrics]
        identities = {(raw.study_id, raw.population) for raw in raw_states}
        if len(identities) != 1:
            winsor_refuse(
                "pool_mismatch",
                "All percentile raw states must share one study and population identity.",
            )
        ordinary = [m for m in metrics if m not in percentile_metrics]
        if ordinary:
            summary = _validate_mixed_winsor_identity(
                summary, raw_states[0], summary_population=summary_population
            )
        bundle = _raw_winsor_lift_bundle(
            percentile_metrics,
            control_group,
            raw_outcomes,
            winsor_references,
            winsor_methods,
            method_roles,
            alpha,
            null_lift,
            null_abs,
            preferred_direction,
        )
        if ordinary:
            remaining = _estimate_lift(
                metrics=ordinary,
                summary=summary,
                control_group=control_group,
                methods=methods,
                prior=prior,
                alpha=alpha,
                alternative=alternative,
                inference=inference,
                null_lift=null_lift,
                null_abs=null_abs,
                preferred_direction=preferred_direction,
                cluster=cluster,
                method_roles=method_roles,
                route_alpha=route_alpha,
            )
            from increment.estimation.decision_types import DecisionComputation

            return DecisionComputation(
                results=(*bundle.results, *remaining.results),
                evidence={**bundle.evidence, **remaining.evidence},
                failures={**bundle.failures, **remaining.failures},
            )
        return bundle

    if isinstance(summary, SequentialSnapshot):
        sequential_refuse(
            "source.invalid", "an exact snapshot requires its registered runtime policy"
        )
    prepared = _prepare_lift_estimation(
        metrics,
        summary,
        control_group,
        methods,
        prior,
        alpha,
        alternative,
        inference,
        null_lift,
        null_abs,
        cluster,
        method_roles,
    )
    if not isinstance(prepared, _PreparedLiftEstimation):
        return prepared
    if any(
        a.winsor_lower_percentile is not None or a.winsor_upper_percentile is not None
        for a in (*prepared.treatment_arms, *prepared.control_by_metric.values())
    ):
        from increment.winsor import winsor_refuse

        winsor_refuse(
            "raw_state_required",
            "Legacy percentile metadata cannot be analyzed as fixed clipped moments.",
        )
    method_strategies = tuple(
        _LiftMethodStrategy(method=method, use_cuped=method.variance_reduction == "cuped")
        for method in prepared.methods
    )
    results: list[LiftEstimate] = []
    guard_failures: dict[Any, Any] = {}
    for treatment in prepared.treatment_arms:
        control = prepared.control_by_metric.get(treatment.metric)
        if control is None:
            from increment.estimation.decision_types import ArmHypothesisKey, DecisionFailure

            hypothesis = ArmHypothesisKey(treatment.metric, treatment.group_id, "itt")
            guard_failures[hypothesis] = DecisionFailure(
                hypothesis,
                "estimation.engine.missing_control",
                {
                    "metric": treatment.metric,
                    "group_id": treatment.group_id,
                    "control_group": control_group,
                },
            )
            continue
        strategy = _select_lift_variance_strategy(
            treatment,
            control,
            prepared.metric_types[treatment.metric],
            control_group,
            cluster,
            method_strategies,
        )
        metric_type = prepared.metric_types[treatment.metric]
        for method_strategy in strategy.methods:
            method_role = prepared.resolved_method_roles.get(
                method_strategy.method.name, "decision"
            )
            result, failure = _lift_for_method(
                (treatment, control),
                metric_type,
                cluster,
                method_strategy,
                strategy,
                method_role,
                prior,
                alpha,
                alternative,
                inference,
                null_lift,
                null_abs,
                preferred_direction,
                route_alpha,
            )
            if failure is not None:
                guard_failures[failure.hypothesis] = failure
            if result is not None:
                results.append(result)
    bundle = _lift_decision_bundle(
        results,
        inference=inference,
    )
    if not guard_failures:
        return bundle
    from increment.estimation.decision_types import DecisionComputation

    return DecisionComputation(
        results=bundle.results,
        evidence=dict(bundle.evidence),
        failures={**bundle.failures, **guard_failures},
    )
