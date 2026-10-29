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
from typing import TYPE_CHECKING, Any, Literal, NoReturn, TypedDict, cast

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import (
    BaseModel as BaseModel,
)
from pydantic import (
    ConfigDict as ConfigDict,
)
from pydantic import (
    Field as Field,
)
from pydantic import (
    field_validator as field_validator,
)
from pydantic import (
    model_validator as model_validator,
)

from increment._evidence_types import unit_frame_covariate_columns
from increment._finite_sample_refusals import (
    refuse_finite_sample_cuped as refuse_finite_sample_cuped,
)
from increment._finite_sample_refusals import (
    refuse_finite_sample_metric_type,
)
from increment._literals import Alternative, PreferredDirection
from increment._literals import ConversionInference as ConversionInference
from increment._moment_plan import CORE_SLOTS, OPTIONAL_SLOTS, X_ROLE_VARIABLES
from increment.declarations import (
    VARIANCE_REDUCTION as VARIANCE_REDUCTION,
)
from increment.declarations import (
    CovariateArg as CovariateArg,
)
from increment.declarations import (
    Method,
)
from increment.declarations import (
    _as_covariate as _as_covariate,
)
from increment.declarations import (
    _normalize_method_covariates as _normalize_method_covariates,
)
from increment.declarations import (
    _normalize_method_covariates_input as _normalize_method_covariates_input,
)
from increment.errors import (
    PACKAGE_FRAMES,
    CapabilityError,
    CodedError,
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
from increment.errors import (
    CodedModel as CodedModel,
)
from increment.estimation import binomial_rr
from increment.estimation._readout_refusals import READOUT_REFUSALS as _READOUT_REFUSALS
from increment.estimation._tails import (
    resolvable_expm1,
    two_sided_critical_value,
    wald_bounds,
)
from increment.estimation.advisory import ReadoutAdvisory
from increment.estimation.armstats import (
    ArmStats,
    binary_counts,
    canonical_bernoulli_arm,
    welch_satterthwaite_df,
)
from increment.estimation.conversion_route import (
    CAPPED_NOTE_PREFIX,
    ESCALATED_NOTE_PREFIX,
    dense_level,
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
    DeltaConversionEvidence,
    Estimate,
    LiftEstimate,
    _alpha_eff_for,
    row_decision_alpha,
    row_p_value,
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
    RATIO_DENOMINATOR_SKEW_THRESHOLD,
    VARIANCE_MODELS,
    MeanVarianceModel,
    check_positive_mean,
    ratio_denominator_mean_skewness,
    ratio_denominator_precision,
    ratio_log_mean_se,
    ratio_moments,
    require_positive_denominator_mean,
    se_log_mean,
    stable_log_ratio,
)
from increment.estimation.variance import ratio_abs_diff_se as variance_ratio_abs_diff_se
from increment.semantics.models import (
    CupedCovariate as CupedCovariate,
)
from increment.semantics.models import (
    CupedCovariateBinding as CupedCovariateBinding,
)
from increment.semantics.models import (
    Metric,
    RetentionMetric,
)
from increment.sequential_state import SequentialSnapshot, sequential_refuse

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.decision import DecisionComputation, DecisionFailure, PValueEvidence
    from increment.estimation.results import RelativeUnavailableReason
    from increment.winsor import BootstrapReference, InfluenceReference, WinsorRawState
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
            template="CUPED cannot be applied to ratio metric '{metric}' from this source: experiment '{experiment}' declares n_pre_periods=0, and the warehouse covariate for a ratio metric is its numerator's pre-period total, so no covariate moments (ref_x/cx1/cx2/cxy) or covariate-denominator cross moment (cxden) can be materialized. Declare n_pre_periods > 0 on the experiment, or supply the covariate through Method.cuped(covariates=...) in the frame's MetricSpec method declaration.",
        ),
        "estimation.engine.arm.duplicate_rows": "group_summary carries more than one row for (metric, group_id) {keys} -- an unchecked duplicate control row silently overwrites an earlier one in the lookup dict, and an unchecked duplicate treatment row is silently iterated as if it were a second arm, either way returning an arbitrary lift instead of a refusal. Deduplicate group_summary to exactly one row per (metric, group_id) before calling estimate_lift.",
        "estimation.engine.arm.invalid_count": "{what} is {value!r}, not a finite integer-valued count -- ArmStats.n rejects a fractional value at direct construction, so the untyped dataframe ingress must too instead of silently truncating it (int(2.9) == 2) into a different arm size than the row declared. Round or fix the upstream aggregation so 'n' is integral.",
        "estimation.engine.ratio.negative_variance": "ratio_abs_diff_se refused these moments: {reason} -- a deficit at this scale means the numerator/denominator moments violate Cauchy-Schwarz (an infeasible covariance), not merely cancelled, so reporting a clamped zero-SE (maximal confidence) would be the most dangerous possible failure mode for a decision number.",
        "estimation.engine.cuped_linear_unit_evidence": RefusalSpec(
            "estimation.engine.cuped_linear_unit_evidence",
            UnsupportedRequestError,
            template="multicovariate CUPED for metric {metric!r} requires per-unit outcomes and covariates declared on the method; use a per-unit analysis entry point instead of aggregate moments.",
            keys=frozenset({"metric"}),
        ),
        "estimation.engine.cuped_linear_observational": RefusalSpec(
            "estimation.engine.cuped_linear_observational",
            UnsupportedRequestError,
            template="multicovariate CUPED is a randomized mean/ratio estimator and cannot be used with observational metric {metric!r}; choose an observational estimator or a randomized design.",
            keys=frozenset({"metric"}),
        ),
        "estimation.engine.cuped_linear_cluster": RefusalSpec(
            "estimation.engine.cuped_linear_cluster",
            UnsupportedRequestError,
            template="multicovariate CUPED does not support clustered assignment (cluster={cluster!r}); choose an unadjusted clustered method or scalar CUPED where supported.",
            keys=frozenset({"cluster"}),
        ),
        "estimation.engine.cuped_linear_sequential": RefusalSpec(
            "estimation.engine.cuped_linear_sequential",
            UnsupportedRequestError,
            template="multicovariate CUPED is fixed-horizon only and cannot be used with sequential inference; use an unadjusted sequential method or scalar CUPED's registered sequential route.",
            keys=frozenset(),
        ),
        "estimation.engine.cuped_linear_winsorization": RefusalSpec(
            "estimation.engine.cuped_linear_winsorization",
            UnsupportedRequestError,
            template="multicovariate CUPED cannot be composed with winsorization for metric {metric!r}; choose one supported construction.",
            keys=frozenset({"metric"}),
        ),
        "estimation.engine.cuped_linear_encouragement": RefusalSpec(
            "estimation.engine.cuped_linear_encouragement",
            UnsupportedRequestError,
            template="multicovariate CUPED is not available for encouragement design metric {metric!r}; use an encouragement-supported method without CUPED.",
            keys=frozenset({"metric"}),
        ),
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
    code: str,
    warning_type: type[IncrementWarning],
    render: Callable[..., str],
    *,
    dedupe_fields: tuple[str, ...] | None = None,
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render, dedupe_fields=dedupe_fields)
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
    dedupe_fields=("metric_name",),
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
    dedupe_fields=("metric_name",),
)

_register_warning(
    "estimation.engine.ratio_denominator_skew",
    IncrementRuntimeWarning,
    lambda *, metric, arm, n, skewness, statistic, threshold: (
        f"metric '{metric}' arm '{arm}': the denominator is too right-skewed for its sample "
        f"size (sample skewness {skewness:.3g} over sqrt(n={n}) is {statistic:.3g}, above "
        f"{threshold:.3g}); the delta-method interval undercovers in this region (about 92% "
        "pooled at nominal 95% in calibration) and is reported unchanged -- add units or check "
        "the denominator for heavy tails before deciding."
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


def _is_unadjusted_method(method: Method) -> bool:
    """Whether a declared procedure or historical label selects unadjusted inference."""
    return method.variance_reduction == "none" and (
        method.name in {"naive", "unadjusted"}
        or method.procedure == "unadjusted"
        and not method.is_observational
    )


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
    prefer: Callable[[Method], bool] = _is_unadjusted_method,
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
    from increment.declarations import ADJUSTMENTS

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
                outcome_window="unbounded" if is_open_ended(metric) else "bounded",
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


def _reject_cuped_linear_aggregate(
    metrics: Sequence[Metric], methods: Sequence[Method] | None
) -> None:
    if methods is None:
        return
    for metric in metrics:
        if any(method.variance_reduction == "cuped_multicovariate" for method in methods):
            _refuse("estimation.engine.cuped_linear_unit_evidence", metric=metric.name)


def refuse_cuped_linear_sequential() -> NoReturn:
    """Use the engine-owned CUPED-linear refusal from sequential preflight."""
    _refuse("estimation.engine.cuped_linear_sequential")


def _validate_cuped_linear_compatibility(request: ReadoutRequest) -> None:
    metrics = tuple(request.metrics)
    methods = request.estimation_methods
    linear_by_metric = {
        metric.name: any(
            method.variance_reduction == "cuped_multicovariate" for method in metric_methods
        )
        for metric, metric_methods in zip(metrics, methods, strict=True)
    }
    mechanism = getattr(request.design, "mechanism", None)
    if request.cluster is not None and any(linear_by_metric.values()):
        _refuse("estimation.engine.cuped_linear_cluster", cluster=request.cluster)
    if isinstance(request.plan.inference, SEQUENTIAL_POLICIES) and any(linear_by_metric.values()):
        refuse_cuped_linear_sequential()
    for metric in metrics:
        if not linear_by_metric[metric.name]:
            continue
        if getattr(getattr(metric, "winsorization", None), "has_percentile", False):
            _refuse("estimation.engine.cuped_linear_winsorization", metric=metric.name)
        if mechanism == "encouragement":
            _refuse("estimation.engine.cuped_linear_encouragement", metric=metric.name)
        if mechanism == "observational":
            _refuse("estimation.engine.cuped_linear_observational", metric=metric.name)


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
    _validate_cuped_linear_compatibility(request)
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
            from increment._evidence_refusals import OPERATION_UNAVAILABLE
            from increment.errors import refuse as refuse_spec

            refuse_spec(
                OPERATION_UNAVAILABLE,
                constructor=request.source_name,
                requested="daily_winsorization",
                offered=(),
                route="use total-grain run() or run_breakout()",
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


def is_open_ended(metric: Metric) -> bool:
    """Whether a metric's per-unit value can still change between analyses."""
    if metric.type == "ratio":
        return metric.numerator.window_days is None or metric.denominator.window_days is None
    if metric.type == "retention":
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
        if not is_open_ended(m):
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


def _open_ended_advisory(metric: Metric) -> ReadoutAdvisory | None:
    if not is_open_ended(metric):
        return None
    if metric.type == "ratio":
        windows = "one or both windows unset"
    elif metric.type == "retention":
        windows = "unbounded band"
    else:
        windows = "window_days=None"
    fix = (
        "Bound the band (threshold_days: [a, b])"
        if metric.type == "retention"
        else "Set window_days"
    )
    context = {"metric_name": metric.name, "windows": windows, "fix": fix}
    spec = _WARNINGS["estimation.engine.open_ended_sequential"]
    return ReadoutAdvisory(code=spec.code, context=context, message=spec.render(**context))


def _small_total_clusters_advisory(
    metric_name: str, cluster: str | None, n_clusters: int | None
) -> ReadoutAdvisory | None:
    if n_clusters is None or n_clusters >= _WARN_TOTAL_CLUSTERS:
        return None
    context = {
        "metric_name": metric_name,
        "n_clusters": n_clusters,
        "cluster": cluster,
        "floor": _WARN_TOTAL_CLUSTERS,
    }
    spec = _WARNINGS["estimation.engine.small_total_clusters"]
    return ReadoutAdvisory(code=spec.code, context=context, message=spec.render(**context))


def _lift_advisories(
    metric: Metric,
    treatment: ArmStats,
    control: ArmStats,
    strategy: _LiftVarianceStrategy,
    *,
    cluster: str | None,
    sequential: bool,
) -> tuple[ReadoutAdvisory, ...]:
    advisories: list[ReadoutAdvisory] = []
    cluster_advisory = _small_total_clusters_advisory(metric.name, cluster, strategy.n_clusters)
    if cluster_advisory is not None:
        advisories.append(cluster_advisory)
    for note in strategy.ratio_notes:
        if note.startswith("ratio_denominator_precision:"):
            stats = {
                arm.group_id: ratio_denominator_precision(arm)
                for arm in (treatment, control)
                if ratio_denominator_precision(arm) > RATIO_DENOMINATOR_PRECISION_THRESHOLD
            }
            advisories.append(
                ReadoutAdvisory(
                    code="estimation.engine.ratio_denominator_precision",
                    context={
                        "metric_name": metric.name,
                        "threshold": RATIO_DENOMINATOR_PRECISION_THRESHOLD,
                        "relative_standard_errors": stats,
                    },
                    message=note,
                )
            )
        elif note.startswith("ratio_denominator_skew:"):
            advisories.append(
                ReadoutAdvisory(
                    code="estimation.engine.ratio_denominator_skew",
                    context={
                        "metric_name": metric.name,
                        "threshold": RATIO_DENOMINATOR_SKEW_THRESHOLD,
                    },
                    message=note,
                )
            )
    if sequential:
        advisory = _open_ended_advisory(metric)
        if advisory is not None:
            advisories.append(advisory)
    return tuple(advisories)


def _treatment_lift_advisories(metrics, treatment, control, strategy, cluster, sequential):
    metric = next(item for item in metrics if item.name == treatment.metric)
    return _lift_advisories(
        metric,
        treatment,
        control,
        strategy,
        cluster=cluster,
        sequential=sequential,
    )


def _attach_lift_advisories(advisory_map, treatment, advisories, result, failure):
    if advisories and (result is not None or failure is not None):
        from increment.estimation.decision_types import ArmHypothesisKey

        advisory_map[ArmHypothesisKey(treatment.metric, treatment.group_id, "itt")] = advisories


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
    margin paths don't apply. A row with no relative interval reads the
    row's own ``p_value()``: its joint relative set, or -- for a
    nonpositive arm mean with no declared margin -- the additive difference
    against zero under the reference its interval was cut from."""
    from increment.estimation.decision_types import DecisionFailure, PValueEvidence
    from increment.estimation.results import _additive_zero_reference

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
        if result.relative_confidence_set is not None:
            reference = "relative_confidence_set"
        elif _additive_zero_reference(result) is None:
            reference = (
                "relative_margin_conservative_nonrejection"
                if result.null_lift != 0.0 and result.null_abs is None
                else "degenerate_arm_conservative_nonrejection"
            )
        else:
            reference = "student_t" if result.abs_reference_kind == "t" else "normal"
        return PValueEvidence(hypothesis, result.method, p_value, reference), None

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
    advisories: dict[Any, list[ReadoutAdvisory]] = {}
    for computation in computations:
        results.extend(computation.results)
        evidence.update(computation.evidence)
        failures.update(computation.failures)
        for hypothesis, items in computation.advisories.items():
            advisories.setdefault(hypothesis, []).extend(items)
    merged_advisories = {}
    for key, items in advisories.items():
        distinct = {item.model_dump_json(): item for item in items}
        merged_advisories[key] = tuple(distinct[value] for value in sorted(distinct))
    return DecisionComputation(
        results=tuple(results),
        evidence=evidence,
        failures=failures,
        advisories=merged_advisories,
    )


def _lift_decision_bundle(  # noqa: PLR0915
    results: Sequence[LiftEstimate],
    *,
    inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    sequential_snapshot=None,
    cuped_advisory: ReadoutAdvisory | None = None,
    cuped_failures: Sequence[tuple[str, str, str, str, Mapping[str, Any]]] = (),
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
        if result.delta_evidence is not None and result.delta_evidence.composite:
            # The composite p-value the row itself reports (its persisted counts, dense level
            # and exact reading), the same evidence a family assembled after the fact reads.
            p_value = result.p_value()
            assert p_value is not None, "a delta-route row carries its sampling reference"
            evidence[hypothesis] = PValueEvidence(
                hypothesis, result.method, p_value, "conversion_composite"
            )
            continue
        evidence_row, failure_row = _raw_stats_evidence(result, hypothesis)
        if failure_row is not None:
            failures[hypothesis] = failure_row
        else:
            evidence[hypothesis] = evidence_row
    for metric, group_id, estimand, code, context in cuped_failures:
        hypothesis = ArmHypothesisKey(metric, group_id, estimand)
        evidence.pop(hypothesis, None)
        failures[hypothesis] = DecisionFailure(hypothesis, code, dict(context))
    advisories: dict[Any, tuple[ReadoutAdvisory, ...]] = (
        {
            ArmHypothesisKey(result.metric, result.group_id, result.estimand): (cuped_advisory,)
            for result in output_results
        }
        if cuped_advisory is not None
        else {}
    )
    return DecisionComputation(
        results=tuple(output_results),
        evidence=evidence,
        failures=failures,
        sequential_snapshot=sequential_snapshot,
        advisories=advisories,
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


def _cluster_arm_unit_facts(arm: ArmStats) -> tuple[int | None, float | None]:
    """Return unit count and raw unit outcome mean from cluster totals."""
    if arm.x_role != "cluster_size" or arm.ref_x is None or arm.cx1 is None:
        return None, None
    unit_count = round(arm.n * arm.ref_x + arm.cx1)
    if unit_count <= 0:
        return None, None
    outcome_total = arm.n * arm.ref_y + arm.cy1
    return unit_count, outcome_total / unit_count


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
        cluster_result = result.model_copy(update=_winsorization_result_fields(control, treatment))
        if strategy.n_clusters is not None:
            control_n, control_mean = _cluster_arm_unit_facts(control)
            treatment_n, treatment_mean = _cluster_arm_unit_facts(treatment)
            if strategy.metric_type in {"ratio", "quantile"}:
                # A per-unit outcome mean is not the arm's value of a ratio metric.
                control_mean = treatment_mean = None
            cluster_result = cluster_result.model_copy(
                update={
                    "n_control": control_n,
                    "n_treat": treatment_n,
                    "mean_control": control_mean,
                    "mean_treat": treatment_mean,
                }
            )
        return cluster_result, None
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
    if strategy.n_clusters is None and strategy.metric_type not in {"ratio", "quantile"}:
        update.update(
            n_control=control.n,
            n_treat=treatment.n,
            mean_control=control.mean_y(),
            mean_treat=treatment.mean_y(),
        )
    if inference is None and strategy.ratio_notes:
        update["note"] = "; ".join(strategy.ratio_notes)
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
    # Advisories for a fixed-horizon unit-grain ratio contrast: the
    # denominator-precision note, then the denominator-skew coverage note or
    # its unavailability; empty for every other contrast.
    ratio_notes: tuple[str, ...]


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
    *,
    fixed_horizon: bool,
) -> _LiftVarianceStrategy:
    """Resolve one contrast's grain, capability, hoisted moments and ratio advisories.

    The ratio advisories are evaluated for a fixed-horizon unit-grain ratio
    contrast only: the denominator-skew region was measured for the
    single-look interval, and a clustered row's denominator is a cluster
    total whose skew/count regime is not the measured one.
    """
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
    ratio_notes: tuple[str, ...] = ()
    if fixed_horizon and cluster is None and metric_type == "ratio":
        ratio_notes = tuple(
            note
            for note in (
                _ratio_denominator_precision_note(treatment, control),
                _ratio_denominator_skew_note(treatment, control),
            )
            if note is not None
        )
    return _LiftVarianceStrategy(
        variance_model=variance_model,
        metric_type=metric_type,
        dof=dof,
        n_clusters=n_clusters,
        absolute_moments=absolute_moments,
        methods=method_strategies,
        ratio_notes=ratio_notes,
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
        "under-coverage; add units or check the denominator for heavy tails before deciding"
    )


def _ratio_denominator_skew_note(treatment: ArmStats, control: ArmStats) -> str | None:
    """Coverage advisory for a ratio contrast whose denominator mean is skewed
    beyond the measured under-covering region in either arm.

    An arm whose third moment overflowed or is inconsistent with its second cannot be
    checked; the note reports that gap while valid moments in the other arm are still
    checked. The estimate proceeds in either case.
    """
    unavailable = [
        (arm.group_id, arm.skew_den_unavailable())
        for arm in (treatment, control)
        if arm.skew_den_unavailable() is not None
    ]
    reasons = []
    missing = [group_id for group_id, reason in unavailable if reason == "missing"]
    underflow = [group_id for group_id, reason in unavailable if reason == "underflow"]
    inconsistent = [group_id for group_id, reason in unavailable if reason == "inconsistent"]
    if missing:
        reasons.append(
            f"the moments for {', '.join(missing)} do not carry the denominator's third moment "
            "(cden3)"
        )
    if underflow:
        reasons.append(
            f"the carried third moment cden3 for {', '.join(underflow)} is zero at a "
            "denominator scale where cubed residuals underflow"
        )
    if inconsistent:
        reasons.append(
            f"the third moment for {', '.join(inconsistent)} is non-finite or inconsistent "
            "with the second (overflowed or mis-aggregated upstream)"
        )
    unavailable_note = (
        "ratio_denominator_skew: unavailable -- "
        + "; ".join(reasons)
        + ", so those arms could not be checked"
        if reasons
        else None
    )
    flagged: list[str] = []
    unavailable_groups = {group_id for group_id, _reason in unavailable}
    for arm in (treatment, control):
        if arm.group_id in unavailable_groups:
            continue
        statistic = ratio_denominator_mean_skewness(arm)
        if statistic <= RATIO_DENOMINATOR_SKEW_THRESHOLD:
            continue
        skewness = arm.skew_den()
        flagged.append(f"{arm.group_id}={statistic:.3g} (skewness {skewness:.3g}, n={arm.n})")
        _warn(
            "estimation.engine.ratio_denominator_skew",
            metric=arm.metric,
            arm=arm.group_id,
            n=arm.n,
            skewness=skewness,
            statistic=statistic,
            threshold=RATIO_DENOMINATOR_SKEW_THRESHOLD,
            skip_file_prefixes=PACKAGE_FRAMES,
        )
    warning_note = None
    if flagged:
        warning_note = (
            "ratio_denominator_skew: the denominator is too right-skewed for its sample size for "
            "the interval to hold its stated level (standardized skewness of the denominator mean "
            f"{', '.join(flagged)} exceeds {RATIO_DENOMINATOR_SKEW_THRESHOLD:.3g}; about 92% "
            "pooled coverage at nominal 95% in calibration). The estimate and interval are "
            "reported unchanged; add units or check the denominator for heavy tails before deciding"
        )
    if unavailable_note and warning_note:
        return f"{warning_note}; {unavailable_note}"
    return warning_note or unavailable_note


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
    *,
    family_member: bool = False,
    route_tail: float | None = None,
) -> tuple[LiftEstimate | None, DecisionFailure | None]:
    """One (contrast, method) row.

    Auto conversion routing is selected from the counts and the row's own tail allocation
    (``route_tail`` when a family's corrected pass opens the resulting parent at another tail:
    `conversion_route.fcr_route_tail`), independently of the prior and of any family the row
    belongs to. The prior can add a separately guarded posterior, but cannot replace the
    prior-free sampling construction. A row routed to the delta method carries its four counts
    and dense level; a decision row of a multiplicity family (``family_member``) additionally
    reads the composite p-value the family decides on (`escalate_family_member`).
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
    route = route_for_counts(
        *counts,
        tail_alpha=alpha_eff / 2.0 if route_tail is None else route_tail,
        mode=method.conversion_inference,
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
    result, failure = _asymptotic_lift_outcome(
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
    if result is None or null_abs is not None:
        # An additive margin decides on the additive sidecar, which neither route cuts.
        return result, failure
    level = dense_level(*counts, alternative=valid_alternative)
    assert level is not None, "routed asymptotic: the counts are dense at the nominal level"
    x_c, n_c, x_t, n_t = counts
    result = result.model_copy(
        update={
            "delta_evidence": DeltaConversionEvidence(
                x_c=x_c, n_c=n_c, x_t=x_t, n_t=n_t, dense_level=level
            )
        }
    )
    if family_member and method_role == "decision":
        result = escalate_family_member(result)
    return result, failure


def escalate_family_member(row: LiftEstimate) -> LiftEstimate:
    """*row*, a delta-route conversion decision row with its counts, read as a member of a
    multiplicity family: its ``delta_evidence`` becomes composite, carrying the exact p-value
    where the composite needs it, and its note says when its p-value is not its delta p-value.
    A row already read as a member, or one without counts, is returned unchanged."""
    evidence = row.delta_evidence
    if evidence is None or evidence.composite:
        return row
    alpha = row_decision_alpha(row)
    assert alpha is not None, "a delta-route row carries its decision alpha"
    composite, note = _composite_evidence(
        row, evidence, alpha, binomial_rr.validate_alternative(row.alternative)
    )
    notes = [part for part in (row.note, note) if part is not None]
    return row.model_copy(update={"delta_evidence": composite, "note": "; ".join(notes) or None})


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
    from increment.estimation.conversion_route import composite_exact_reading

    exact, code = composite_exact_reading(
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


def _ratio_denominator_refusals(frame: Any, metric_name: str) -> dict[str, CodedError]:
    """The variance owner's refusal for every arm whose raw denominator mean is not positive."""
    means = frame.group_by("group_id").agg(nw.col("y_den").mean().alias("d_bar"))
    refused: dict[str, CodedError] = {}
    for group_id, d_bar in zip(
        means.get_column("group_id").to_list(), means.get_column("d_bar").to_list(), strict=True
    ):
        try:
            require_positive_denominator_mean(
                metric=metric_name, group_id=str(group_id), d_bar=float(d_bar)
            )
        except CodedError as exc:
            refused[str(group_id)] = exc
    return refused


# Numeric refusals proven local to one (arm, method) cell: a readout keeps the
# cell as an unavailable row while its siblings stay usable. Capability,
# request, source, and wire refusals abort the readout instead.
CELL_ESTIMATION_GUARD_CODES = frozenset(
    {
        "estimation.cuped.covariate_zero_variance",
        "estimation.variance.ratio_moments_nonpositive_denominator_mean",
        "estimation.cuped_linear.arm_counts",
        "estimation.cuped_linear.invalid_arms",
        "estimation.cuped_linear.nonfinite",
        "estimation.cuped_linear.missing_refused",
        "estimation.cuped_linear.level_support",
        "estimation.cuped_linear.rank_deficiency",
        "estimation.cuped_linear.leverage",
        "estimation.cuped_linear.support_region",
        "estimation.cuped_linear.residual_df",
        "estimation.cuped_linear.nonpositive_component",
        "estimation.cuped_linear.nonfinite_projection",
    }
)


def _cuped_linear_posterior_fields(
    metric: Metric, arm: str, prior: Prior | None, *, reference_df: float | None
) -> dict[str, Any]:
    """A declared prior is unavailable on every linear row, with its exact reason:
    there are no HC2/BM score statistics to compose it from."""
    if prior is None:
        return {
            "posterior_available": None,
            "posterior_reason_code": None,
            "posterior_reason_context": None,
        }
    return {
        "posterior_available": False,
        "posterior_reason_code": "estimation.results.lift.t_posterior_sufficient_statistics",
        "posterior_reason_context": {
            "metric": metric.name,
            "group_id": arm,
            "reference_df": reference_df,
            "missing": "HC2/BM score statistics needed to compose the declared prior",
        },
    }


def _cuped_linear_unavailable_row(
    metric: Metric,
    arm: str,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    *,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
    prior: Prior | None,
    code: str,
    context: Mapping[str, object],
) -> LiftEstimate:
    """One unavailable linear row carrying its exact refusal."""
    return LiftEstimate(
        metric=metric.name,
        group_id=arm,
        method=method.name,
        method_role=method_role,
        estimand="itt",
        alternative=alternative,
        preferred_direction=metric.declared_preferred_direction,
        null_lift=null_lift,
        null_abs=null_abs,
        failure_code=code,
        failure_context=context,
        **_cuped_linear_posterior_fields(metric, arm, prior, reference_df=None),
    )


def _cuped_linear_withheld_relative_row(  # noqa: PLR0913
    metric: Metric,
    arm: str,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    *,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
    prior: Prior | None,
    absolute: Any,
    alpha_eff: float,
    reason: RelativeUnavailableReason,
    code: str,
    context: Mapping[str, object],
) -> LiftEstimate:
    """The additive projection of an arm whose relative projection is refused.

    The absolute contrast, its HC2/BM ``t`` interval and any absolute-margin
    decision stay usable; the relative refusal is recorded beside them rather
    than failing the cell, mirroring the clustered ``lift=None`` row shape.
    """
    return LiftEstimate(
        metric=metric.name,
        group_id=arm,
        method=method.name,
        method_role=method_role,
        estimand="itt",
        alternative=alternative,
        preferred_direction=metric.declared_preferred_direction,
        null_lift=null_lift,
        null_abs=null_abs,
        abs_diff=absolute.estimate,
        abs_se=absolute.se if absolute.se > 0.0 else None,
        abs_lb=absolute.lower,
        abs_ub=absolute.upper,
        abs_reference_kind="t",
        abs_reference_df=absolute.dof,
        abs_alpha=alpha_eff,
        reference_kind="t",
        reference_df=absolute.dof,
        dof=absolute.dof,
        scale="linear",
        relative_unavailable_reason=reason,
        relative_error_code=code,
        relative_error_context=context,
        **_cuped_linear_posterior_fields(metric, arm, prior, reference_df=absolute.dof),
    )


def _withhold_ratio_arms(
    frame: Any,
    arm_order: tuple[str, ...],
    *,
    metric: Metric,
    control_group: str,
    refuse_arm: Callable[[str, str, Mapping[str, object]], None],
) -> tuple[Any, tuple[str, ...]]:
    """Withhold ratio arms that cannot form the estimand, as their own cell failures.

    Returns the per-unit evidence and arm order left to fit. When nothing is
    left to contrast -- the control arm is withheld, or every non-control arm
    is -- the evidence comes back as ``None`` and every contrast carries its
    refusal.
    """
    refused = _ratio_denominator_refusals(frame, metric.name)
    remaining = _withhold_refused_arms(
        refused, arm_order, control_group=control_group, refuse_arm=refuse_arm
    )
    if not remaining:
        return None, ()
    if refused:
        frame = frame.filter(~nw.col("group_id").is_in(list(refused)))
    return frame, remaining


def _withhold_refused_arms(
    refused: Mapping[str, CodedError],
    arm_order: tuple[str, ...],
    *,
    control_group: str,
    refuse_arm: Callable[[str, str, Mapping[str, object]], None],
) -> tuple[str, ...]:
    """Record each refused arm's own failure; the arms left to fit, or none to contrast."""
    if control_group in refused:
        refusal = refused[control_group]
        for arm in arm_order:
            if arm != control_group:
                refuse_arm(arm, refusal.code, refusal.context)
        return ()
    for arm in arm_order:
        if arm in refused:
            refuse_arm(arm, refused[arm].code, refused[arm].context)
    remaining = tuple(arm for arm in arm_order if arm not in refused)
    return () if remaining == (control_group,) else remaining


def _cell_denominator_refusals(cells: Any, metric_name: str) -> dict[str, CodedError]:
    """The variance owner's refusal for every cell arm whose raw denominator mean is not positive."""
    refused: dict[str, CodedError] = {}
    for a, arm in enumerate(cells.arms):
        count = sum(cells.counts[a])
        if not count:
            continue
        d_bar = cells.outcome_references[1][a] + math.ldexp(
            math.fsum(cells.centered_first_moments[a][1]) / count, cells.outcome_exponents[1][a]
        )
        try:
            require_positive_denominator_mean(metric=metric_name, group_id=str(arm), d_bar=d_bar)
        except CodedError as exc:
            refused[str(arm)] = exc
    return refused


def _fit_cuped_multicovariate_frame(
    frame: Any,
    covariates: Sequence[Any],
    *,
    control_group: str,
    arm_order: tuple[str, ...],
    ratio: bool,
    covariate_columns: Mapping[str, str] | None = None,
) -> Any:
    """Fit the per-unit evidence frame once, returning the fit or its coded refusal."""
    import numpy as np

    from increment.estimation.cuped.multicovariate import fit_cuped_multicovariate

    declarations = tuple(covariates)
    # The frame carries a covariate named like one of its own columns under
    # a scratch column; the declared name still identifies it in the fit.
    columns = covariate_columns or unit_frame_covariate_columns(
        [item.name for item in declarations]
    )
    y = np.asarray(frame.get_column("y").to_list(), dtype=float)
    try:
        return fit_cuped_multicovariate(
            None if ratio else y,
            frame.get_column("group_id").to_list(),
            control=control_group,
            arm_order=arm_order,
            covariates={
                item.name: np.asarray(frame.get_column(columns[item.name]).to_list(), dtype=object)
                for item in declarations
            },
            kinds={item.name: item.kind for item in declarations},
            missing={item.name: item.missing for item in declarations},
            numerator=y if ratio else None,
            denominator=(
                np.asarray(frame.get_column("y_den").to_list(), dtype=float) if ratio else None
            ),
        )
    except CodedError as exc:
        return exc


def estimate_cuped_linear(
    metric: Metric,
    unit_frame: Any,
    *,
    control_group: str,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    covariates: Sequence[Any],
    covariate_columns: Mapping[str, str] | None = None,
    alpha: float,
    alternative: str,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    prior: Prior | None = None,
) -> Any:
    """Fit mixed-covariate CUPED once and project one randomized metric.

    A ratio arm whose raw denominator mean is not positive cannot form the
    estimand: it is withheld from the fit and reported as its own cell
    failure, so its siblings keep their adjusted evidence. A control arm in
    that state fails every contrast the same way. A fit that then fails on
    the remaining arms is reported on those arms only; each withheld arm
    keeps its own refusal. An arm whose absolute projection is usable keeps
    it when only the relative projection is refused.
    """
    frame = nw.from_native(unit_frame, eager_only=True)
    ratio = getattr(metric, "type", None) == "ratio"
    arm_order = tuple(dict.fromkeys((control_group, *map(str, frame.get_column("group_id")))))

    refusals = _cuped_linear_refusals(
        metric,
        method,
        method_role,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
    )
    results, cuped_failures, refuse_arm = refusals
    if ratio:
        frame, arm_order = _withhold_ratio_arms(
            frame,
            arm_order,
            metric=metric,
            control_group=control_group,
            refuse_arm=refuse_arm,
        )
        if frame is None:
            return _lift_decision_bundle(results, inference=None, cuped_failures=cuped_failures)
    fit = _fit_cuped_multicovariate_frame(
        frame,
        covariates,
        control_group=control_group,
        arm_order=arm_order,
        ratio=ratio,
        covariate_columns=covariate_columns,
    )
    return _project_cuped_linear(
        fit,
        metric,
        arm_order,
        control_group=control_group,
        method=method,
        method_role=method_role,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
        refusals=refusals,
    )


def estimate_cuped_linear_cells(
    metric: Metric,
    cells: Any,
    *,
    control_group: str,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    prior: Prior | None = None,
) -> Any:
    """Fit joint categorical evidence and project it exactly as the per-unit route does.

    Ratio arms whose raw denominator mean is not positive are withheld with
    their own refusal before the fit, as on the per-unit route.
    """
    from increment.estimation.cuped.cells import fit_categorical_cells, select_cell_arms

    arm_order = tuple(dict.fromkeys((control_group, *map(str, cells.arms))))
    refusals = _cuped_linear_refusals(
        metric,
        method,
        method_role,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
    )
    results, cuped_failures, refuse_arm = refusals
    if getattr(metric, "type", None) == "ratio":
        arm_order = _withhold_refused_arms(
            _cell_denominator_refusals(cells, metric.name),
            arm_order,
            control_group=control_group,
            refuse_arm=refuse_arm,
        )
        if not arm_order:
            return _lift_decision_bundle(results, inference=None, cuped_failures=cuped_failures)
    try:
        selected = cells if arm_order == tuple(cells.arms) else select_cell_arms(cells, arm_order)
        fit = fit_categorical_cells(selected, control=control_group)
    except CodedError as exc:
        fit = exc
    return _project_cuped_linear(
        fit,
        metric,
        arm_order,
        control_group=control_group,
        method=method,
        method_role=method_role,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
        refusals=refusals,
    )


def estimate_cuped_linear_fit(
    metric: Metric,
    fit: Any,
    *,
    arm_order: Sequence[str] | None = None,
    control_group: str,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float = 0.0,
    null_abs: float | None = None,
    prior: Prior | None = None,
) -> Any:
    """Project a finalized compact fit without reacquiring per-unit observations."""
    refusals = _cuped_linear_refusals(
        metric,
        method,
        method_role,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
    )
    results, cuped_failures, refuse_arm = refusals
    withheld = dict(getattr(fit, "withheld", ()))
    resolved_order = tuple(map(str, getattr(fit, "arm_order", arm_order or ())))
    if withheld:
        resolved_order = _withhold_refused_arms(
            withheld,
            resolved_order,
            control_group=control_group,
            refuse_arm=refuse_arm,
        )
        if not resolved_order:
            return _lift_decision_bundle(results, inference=None, cuped_failures=cuped_failures)
    fit = getattr(fit, "fit", fit)
    if fit is None:
        return _lift_decision_bundle(results, inference=None, cuped_failures=cuped_failures)
    return _project_cuped_linear(
        fit,
        metric,
        resolved_order,
        control_group=control_group,
        method=method,
        method_role=method_role,
        alpha=alpha,
        alternative=alternative,
        null_lift=null_lift,
        null_abs=null_abs,
        prior=prior,
        refusals=(results, cuped_failures, refuse_arm),
    )


_LinearRefusals = tuple[
    list[LiftEstimate],
    list[tuple[Any, ...]],
    Callable[[str, str, Mapping[str, object]], None],
]


def _cuped_linear_refusals(
    metric: Metric,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    *,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
    prior: Prior | None,
) -> _LinearRefusals:
    """Unavailable cell rows and decision failures recorded for one linear method."""
    results: list[LiftEstimate] = []
    cuped_failures: list[tuple[Any, ...]] = []

    def refuse_arm(arm: str, code: str, context: Mapping[str, object]) -> None:
        results.append(
            _cuped_linear_unavailable_row(
                metric,
                arm,
                method,
                method_role,
                alternative=alternative,
                null_lift=null_lift,
                null_abs=null_abs,
                prior=prior,
                code=code,
                context=context,
            )
        )
        if method_role == "decision":
            cuped_failures.append((metric.name, arm, "itt", code, context))

    return results, cuped_failures, refuse_arm


def _project_cuped_linear(
    fit: Any,
    metric: Metric,
    arm_order: tuple[str, ...],
    *,
    control_group: str,
    method: Method,
    method_role: Literal["decision", "sensitivity"],
    alpha: float,
    alternative: str,
    null_lift: float,
    null_abs: float | None,
    prior: Prior | None,
    refusals: _LinearRefusals,
) -> Any:
    """Project one fit, or the coded refusal it raised, onto per-arm linear CUPED rows."""
    results, cuped_failures, refuse_arm = refusals
    ratio = getattr(metric, "type", None) == "ratio"
    if isinstance(fit, CodedError):
        if not results or not fit.code.startswith("estimation.cuped_linear."):
            raise fit
        # The fit-wide refusal belongs to the arms that were fitted; the withheld
        # arms already carry their own.
        for arm in arm_order:
            if arm != control_group:
                refuse_arm(arm, fit.code, fit.context)
        return _lift_decision_bundle(results, inference=None, cuped_failures=cuped_failures)
    alpha_eff = alpha if alternative == "two-sided" else 2.0 * alpha
    context = {
        "metric": metric.name,
        "n_min": fit.n_min,
        "p": fit.p,
        "max_leverage": fit.max_leverage,
        "mean_leverage": fit.mean_leverage,
        "reference_df": fit.dof,
        "structural_drops": fit.layout.structural_omissions,
        "shared_missingness_indicators": fit.layout.shared_indicators,
    }
    advisory = ReadoutAdvisory(
        code="estimation.cuped_linear.fit",
        context=context,
        message=(
            f"Mixed CUPED fit: minimum arm size {fit.n_min}, {fit.p} encoded covariates, "
            f"maximum leverage {fit.max_leverage:.3g}, reference df {fit.dof:.3g}."
        ),
    )

    def withhold_relative(arm: str, absolute: Any, exc: CodedError) -> None:
        """Record a relative-projection refusal for one arm, keeping its
        additive projection when there is one; otherwise the cell fails."""
        if absolute is None:
            failure_context = {**exc.context, "metric": metric.name, "group_id": arm}
            refuse_arm(arm, exc.code, failure_context)
            return
        results.append(
            _cuped_linear_withheld_relative_row(
                metric,
                arm,
                method,
                method_role,
                alternative=alternative,
                null_lift=null_lift,
                null_abs=null_abs,
                prior=prior,
                absolute=absolute,
                alpha_eff=alpha_eff,
                reason=(
                    "nonpositive_arm_mean"
                    if exc.code == "estimation.cuped_linear.nonpositive_component"
                    else "relative_unrepresentable"
                ),
                code=exc.code,
                context=exc.context,
            )
        )

    for arm in arm_order:
        if arm == control_group:
            continue
        absolute = None
        absolute_error = None
        try:
            absolute = (
                fit.ratio(arm, scale="absolute", alpha=alpha_eff)
                if ratio
                else fit.absolute(arm, alpha=alpha_eff)
            )
        except CodedError as exc:
            absolute_error = exc
        try:
            relative = (
                fit.ratio(arm, scale="log", alpha=alpha_eff)
                if ratio
                else fit.relative(arm, alpha=alpha_eff)
            )
        except CodedError as exc:
            withhold_relative(arm, absolute, exc)
            continue
        if absolute_error is not None and null_abs is not None and method_role == "decision":
            failure_context = {
                **absolute_error.context,
                "metric": metric.name,
                "group_id": arm,
                "projection": "absolute",
            }
            cuped_failures.append((metric.name, arm, "itt", absolute_error.code, failure_context))
        try:
            point = resolvable_expm1(relative.estimate, what="CUPED relative point estimate")
        except CodedError as exc:
            withhold_relative(arm, absolute, exc)
            continue
        relative_error = None
        try:
            lift = Estimate(
                value=point,
                lb=resolvable_expm1(relative.lower, what="CUPED relative lower bound"),
                ub=resolvable_expm1(relative.upper, what="CUPED relative upper bound"),
                level=1.0 - alpha_eff,
                alpha=alpha_eff,
                log_mean=relative.estimate,
                log_se=relative.se,
            )
        except CodedError as exc:
            # Only the interval is unrepresentable at this level: keep the point
            # and the log-scale statistics its sampling test uses, withhold the
            # bounds, and record the refusal beside them.
            relative_error = exc
            lift = Estimate(value=point, log_mean=relative.estimate, log_se=relative.se)
        results.append(
            LiftEstimate(
                metric=metric.name,
                group_id=arm,
                method=method.name,
                method_role=method_role,
                estimand="itt",
                lift=lift,
                alternative=alternative,
                preferred_direction=metric.declared_preferred_direction,
                null_lift=null_lift,
                null_abs=null_abs,
                abs_diff=None if absolute is None else absolute.estimate,
                abs_se=None if absolute is None or not absolute.se > 0.0 else absolute.se,
                abs_lb=None if absolute is None else absolute.lower,
                abs_ub=None if absolute is None else absolute.upper,
                abs_reference_kind="t" if absolute is not None else None,
                abs_reference_df=None if absolute is None else absolute.dof,
                abs_alpha=alpha_eff if absolute is not None else None,
                reference_kind="t",
                reference_df=relative.dof,
                dof=relative.dof,
                scale="log",
                absolute_error_code=None if absolute_error is None else absolute_error.code,
                absolute_error_context=None if absolute_error is None else absolute_error.context,
                relative_error_code=None if relative_error is None else relative_error.code,
                relative_error_context=None if relative_error is None else relative_error.context,
                **_cuped_linear_posterior_fields(metric, arm, prior, reference_df=relative.dof),
            )
        )
    return _lift_decision_bundle(
        results,
        inference=None,
        cuped_advisory=advisory,
        cuped_failures=cuped_failures,
    )


def reintervalize_cuped_decision(
    row: LiftEstimate,
    alpha: float,
    *,
    alternative: str,
    metric: Metric,
    method: Method,
    prior: Prior | None,
) -> LiftEstimate:
    """Rebuild one selected linear decision row at its family's FCR level.

    Only a selected cell is rebuilt, from its retained HC2/BM statistics
    without refitting (``LiftEstimate.reintervalize``: a level whose relative
    bounds are unrepresentable withholds the interval and keeps the point and
    sampling statistics). A row the level cannot rebuild at all becomes an
    unavailable cell carrying the refusal, so its siblings' rows survive.
    """
    try:
        return row.reintervalize(alpha, interval_alternative=alternative).model_copy(
            update={"allocation_alpha": None}
        )
    except CodedError as exc:
        return _cuped_linear_unavailable_row(
            metric,
            row.group_id,
            method,
            "decision",
            alternative=alternative,
            null_lift=row.null_lift,
            null_abs=row.null_abs,
            prior=prior,
            code=exc.code,
            context={**exc.context, "metric": metric.name, "group_id": row.group_id},
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
    winsor_references: Mapping[tuple[str, str], BootstrapReference | InfluenceReference]
    | None = None,
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
    under a heavily right-skewed denominator the interval undercovers (see
    that class). A fixed-horizon unit-grain ratio row is checked against the
    measured region on each arm's denominator moments. An arm whose
    denominator-mean skewness ``ratio_denominator_mean_skewness`` exceeds
    ``RATIO_DENOMINATOR_SKEW_THRESHOLD`` raises the coded
    ``estimation.engine.ratio_denominator_skew`` warning (context: metric,
    arm, n, skewness, statistic, threshold) and the row carries a
    ``ratio_denominator_skew`` note naming each flagged arm; a row whose
    moments lack the denominator's third moment carries the note's
    unavailability instead of passing silently. An arm exceeding
    ``RATIO_DENOMINATOR_PRECISION_THRESHOLD`` on ``ratio_denominator_precision``
    additionally carries the ``ratio_denominator_precision`` note. Both are
    advisory: the estimate and interval are reported unchanged. Sequential
    and clustered ratio rows carry neither (the region was measured for
    the single-look unit-grain interval).

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
    _reject_cuped_linear_aggregate(metrics, methods)

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
    winsor_references: Mapping[tuple[str, str], BootstrapReference | InfluenceReference]
    | None = None,
    *,
    summary_population: Literal["assigned", "triggered"] | None = None,
    family_member: bool = False,
    route_tail: float | None = None,
) -> DecisionComputation[LiftEstimate]:
    """``estimate_lift`` with the family membership and corrected-pass routing tail its
    multiplicity families pass.

    ``family_member`` marks the decision rows as members of a multiplicity family that reads
    their p-values below their own level: a delta-route conversion decision row then reports
    the composite p-value of its counts (`escalate_family_member`). ``route_tail`` is the
    one-sided tail a selected row's corrected pass routes conversion counts at, the tail of the
    bound it finally reports (`conversion_route.fcr_route_tail`; a directional reissue is
    requested at half the corrected level and opened at the whole of it). Only the package's
    own families set either; the public function never does, so a row's p-value outside a
    family is its own delta p-value and its route its own tail's.
    """
    from increment.estimation.decision_types import DecisionComputation

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
                family_member=family_member,
                route_tail=route_tail,
            )
            from increment.estimation.decision_types import DecisionComputation

            return DecisionComputation(
                results=(*bundle.results, *remaining.results),
                evidence={**bundle.evidence, **remaining.evidence},
                failures={**bundle.failures, **remaining.failures},
                advisories={**bundle.advisories, **remaining.advisories},
            )
        return bundle

    if isinstance(summary, SequentialSnapshot):
        sequential_refuse(
            "source.invalid", "an exact snapshot requires its registered runtime policy"
        )
    _reject_cuped_linear_aggregate(metrics, methods)
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
    advisory_map: dict[Any, tuple[ReadoutAdvisory, ...]] = {}
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
            fixed_horizon=inference is None,
        )
        advisories = _treatment_lift_advisories(
            metrics, treatment, control, strategy, cluster, inference is not None
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
                family_member=family_member,
                route_tail=route_tail,
            )
            _attach_lift_advisories(advisory_map, treatment, advisories, result, failure)
            if failure is not None:
                guard_failures[failure.hypothesis] = failure
            if result is not None:
                results.append(result)
    bundle = _lift_decision_bundle(
        results,
        inference=inference,
    )
    return DecisionComputation(
        results=bundle.results,
        evidence=dict(bundle.evidence),
        failures={**bundle.failures, **guard_failures},
        advisories={**bundle.advisories, **advisory_map},
    )
