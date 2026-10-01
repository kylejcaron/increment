"""Validation and input-normalization internals for frame sources."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import narwhals as nw
import numpy as np

from increment._metric_specs import MetricsArg, MetricSpec, coerce_metrics
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    _safe_error_value,
    raiser,
    refusals,
    refuse,
    warn,
)
from increment.estimation.armstats import CENTERED_FIELDS
from increment.plan import (
    bind_automatic_sequential_plan,
    compile_decision_plan,
    refuse_observational_relative_margin,
)
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import AnalysisPlan, Metric
from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan

CAPABILITY_TABLE: dict[str, str] = {
    "window_days": (
        "window_days is not supported on from_unit_summary -- a one-row-per-unit "
        "summary carries no dates to window against; use from_unit_panel for a "
        "windowed metric."
    ),
    "type=retention": (
        "type='retention' is not supported on from_unit_summary -- a retention "
        "metric needs threshold_days and a date to compute the observation band "
        "against, and a one-row-per-unit summary has neither; use from_unit_panel "
        "for a retention metric."
    ),
}

_CONSTRUCTOR_CAPABILITY = RefusalSpec(
    "source.frame.constructor",
    CapabilityError,
    lambda *, metric, capability, value, route: (
        f"metric {metric!r}: {CAPABILITY_TABLE[capability]} Requested {value!r}. {route}"
    ),
)
_CLUSTER_CAPABILITY = RefusalSpec(
    "source.frame.cluster_capability",
    CapabilityError,
    lambda *, cluster, feature, metrics, uptake, mechanism, route: (
        f"cluster={cluster!r} cannot serve {feature!r} for metrics={metrics!r}, "
        f"uptake={uptake!r}, mechanism={mechanism!r}. {route}"
    ),
)
_PANEL_MISSING_DROP = RefusalSpec(
    "frame.missing_policy.panel_drop",
    UnsupportedRequestError,
    template="missing={missing!r} is not supported on the panel shape -- the panel is densified to a unit x day spine with zero-filled cells, so a dropped (unit, day) row silently means zero, not complete-case. Declare missing='zero' if null means \"no events\", or repair the rows upstream (increment.impute.drop_null before collapsing to one row per unit).",
)
_SOURCE_FRAME_COLUMN_DTYPE = RefusalSpec(
    "source.frame.column_dtype",
    InvalidRequestError,
    template="column {column!r} has dtype {dtype} which increment cannot interpret as a numeric metric value on every backend by construction; offending value(s): {offending_values}",
)
_METRIC_MISSING = RefusalSpec(
    "source.frame.metric_missing",
    InvalidRequestError,
    lambda *, message, missing, available: message,
)
_DUPLICATE_UNITS = RefusalSpec(
    "source.frame.duplicate_units",
    InvalidRequestError,
    template="from_unit_summary requires one row per unit, but {count} unit(s) have duplicates; sample unit ids: {units!r}. {route}",
)
_CONTROL_MISSING = RefusalSpec(
    "source.frame.control_missing",
    InvalidRequestError,
    template="control group {control!r} not found in column {group!r}; observed groups: {observed!r}. {route}",
)
_GROUP_LABEL_COLLISION = RefusalSpec(
    "frame.validation.group_label_collision",
    InvalidRequestError,
    lambda *, group, collision, route: (
        f"column {group!r} has a group-label identity collision: {collision}. {route}"
    ),
)
_GROUP_LABEL_RESERVED = RefusalSpec(
    "frame.validation.group_label_reserved",
    InvalidRequestError,
    lambda *, group, labels, route: (
        f"column {group!r} uses reserved accounting label(s) {labels!r} as real arms. {route}"
    ),
)
_UPTAKE_NOT_BINARY = RefusalSpec(
    "source.frame.uptake_not_binary",
    InvalidRequestError,
    lambda *, uptake, values, route: (
        f"uptake column {uptake!r} must be binary (0/1 or True/False); "
        f"found non-binary value(s): {values!r}. {route}"
    ),
)
_CONVERSION_NOT_BINARY = RefusalSpec(
    "source.frame.conversion_not_binary",
    InvalidRequestError,
    lambda *, metric, column, values, route: (
        f"metric {metric!r}: type='conversion' expects a 0/1 or True/False "
        f"column, but {column!r} has non-binary value(s): {values!r}. {route}"
    ),
)
_CLUSTER_LABELS = RefusalSpec(
    "source.frame.cluster_labels",
    InvalidRequestError,
    lambda *, cluster, reason, route, missing=None, spanning=None, examples=(): (
        (
            f"cluster column {cluster!r} has {missing} null/NaN labels."
            if reason == "null_label"
            else f"cluster column {cluster!r} has {spanning} labels spanning arms "
            f"(examples={examples!r})."
        )
        + f" {route}"
    ),
)
_DAY_GRAIN_COLUMNS = RefusalSpec(
    "source.panel.day_grain_columns",
    InvalidRequestError,
    lambda *, columns, route: (
        f"day-grain columns have unsupported timezone/axis semantics: {columns!r}. {route}"
    ),
)
_NULL_IDENTITY = RefusalSpec(
    "source.frame.null_identity",
    InvalidRequestError,
    lambda *, role, column, count, route: (
        f"{count} row(s) have a null value in {role} column {column!r}. {route}"
    ),
)
_NON_FINITE = RefusalSpec(
    "source.frame.non_finite",
    InvalidRequestError,
    lambda *, role, column, count, route: (
        f"{role} column {column!r} has {count} non-finite value(s). {route}"
    ),
)
_UNASSIGNED = RefusalSpec(
    "source.frame.unassigned",
    InvalidRequestError,
    template="{count} row(s) have a null value in group column {group!r}; sample unit ids: {units!r}. {route}",
    keys=frozenset({"constructor"}),
)
_NULL_EXPOSURE = RefusalSpec(
    "source.panel.null_exposure",
    InvalidRequestError,
    lambda *, count, column, constructor, route: (
        f"{count} unit(s) have null exposure_date values in {column!r}. {route} "
        f"Use {constructor!r} with on_unassigned='exclude' for accounted exclusion."
    ),
)
_EXPOSURE_DATE = RefusalSpec(
    "source.panel.exposure_date",
    InvalidRequestError,
    lambda *, issue, column, count, examples, route: (
        f"exposure_date validation failed ({issue}; column={column!r}, count={count}, "
        f"examples={examples!r}). {route}"
    ),
)
_DUPLICATE_UNIT_DAYS = RefusalSpec(
    "source.panel.duplicate_unit_days",
    InvalidRequestError,
    lambda *, unit, date, count, examples, route: (
        f"from_unit_panel found {count} duplicate (unit, day) pair(s) for "
        f"({unit!r}, {date!r}), examples={examples!r}. {route}"
    ),
)
_MULTI_GROUP_UNIT = RefusalSpec(
    "source.panel.multi_group_unit",
    InvalidRequestError,
    lambda *, unit, group, count, examples, route: (
        f"{count} unit(s) appear in multiple groups across the panel "
        f"({unit!r}, {group!r}), examples={examples!r}. {route}"
    ),
)
_RATIO_SAME_COLUMN = RefusalSpec(
    "source.frame.ratio_same_column",
    InvalidRequestError,
    template="metric {metric!r} reads column {column!r} for both numerator and denominator -- that ratio is identically 1, so there is nothing to estimate; point the two sides at different columns",
)
_ON_UNASSIGNED_INVALID = RefusalSpec(
    "source.frame.on_unassigned_invalid",
    InvalidRequestError,
    lambda *, value, allowed: f"on_unassigned must be one of {sorted(allowed)!r}, got {value!r}",
)


_METRIC_COVARIATE_IMPUTE_ALL_NULL = RefusalSpec(
    "frame.validation.metric_covariate_impute_all_null",
    InvalidRequestError,
    template="metric {metric!r}: covariate column {covariate!r} has no observed values to impute a pooled mean from -- every row is null/NaN. Drop covariate= or repair the column upstream.",
)

Renderer = Callable[..., str]


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Renderer
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


# Shared by both the dataframe path (_frame_panel.py) and the warehouse path
# (query/builders.py): the same censoring hazard must raise the same code
# from either entry point, so both import and reuse this one spec rather
# than each registering an independently constructed duplicate.
CENSORING_DROPPED_UNITS = _register_warning(
    "frame.censoring.dropped_units",
    IncrementWarning,
    lambda *, metric_name, dropped, enrolled, cause: (
        f"metric {metric_name!r}: censoring dropped {dropped} of {enrolled} "
        f"enrolled units ({dropped / enrolled:.0%}) whose window closes "
        f"after the observable bound. Bound is {cause}."
    ),
)


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "frame.validation.metric_missing_drop",
    IncrementWarning,
    lambda *, name, dropped: (
        f"metric {name!r}: missing='drop' excludes rows with "
        f"{dropped} null/NaN value(s) from this metric's moments "
        f"(complete-case). This is unbiased only when missingness "
        f'is unrelated to the values; if null means "no events", '
        f"missing='zero' keeps every unit."
    ),
)
_register_warning(
    "frame.validation.metric_covariate_missing_imputed",
    IncrementWarning,
    lambda *, name, covariate, n_missing, filled: (
        f"metric {name!r}: covariate column {covariate!r} "
        f"has {n_missing} null/NaN value(s); each is counted as "
        f"{filled} -- a deterministic imputation computed without "
        f"the arm or the outcome, so randomization keeps the "
        f"estimate unbiased and only the imputed units' variance "
        f"reduction is forgone. Set covariate_missing='error' to "
        f"refuse instead."
    ),
)


# Shared by frame panels and warehouse artifacts: the same unit_frame hazard.
FRAME_UNIT_FRAME_PANEL = RefusalSpec(
    "source.frame.unit_frame_panel",
    CapabilityError,
    lambda *, metric, shape, route: (
        f"unit_frame for metric {metric!r} is unavailable on {shape}. {route}"
    ),
)

_REFUSALS = refusals(
    InvalidRequestError,
    {
        "frame.validation.metric_covariate_impute_all_null": _METRIC_COVARIATE_IMPUTE_ALL_NULL,
        "frame.validation.design_control_group": "design.control_group={control_group!r} disagrees with control={control!r} -- one control, declared once",
        "frame.validation.dtype_disagree_day": "{ds_name!r} (dtype {ds_dtype}) and {anchor_name!r} (dtype {anchor_dtype}) disagree on day-axis kind -- one is an explicit numeric day index and the other is a calendar date/datetime. Declare both the same kind.",
        "frame.validation.metric_covariate_column": "metric {metric!r}: covariate column {covariate!r} has {n_missing} null/NaN value(s) and covariate_missing='error'. Sums skip nulls while n counts every row, which would silently bias the CUPED estimate (an artificial covariate imbalance). Drop covariate_missing='error' to pooled-mean impute them (unbiased under randomization), declare covariate_missing='zero' if null means \"no pre-period activity\", or repair upstream with increment.impute.pooled_mean.",
        "frame.validation.metric_value_missing": "metric {metric!r}: column {column!r} has {n_missing} null/NaN value(s). An undeclared null is silently averaged as 0 (n counts the row, sum skips it) and NaN sums diverge by backend. If null means \"no events\", declare MetricSpec(missing='zero') or run df, n = increment.impute.zeros(df, {column!r}). If \"not observed\", declare missing='drop' (complete-case; unbiased only when missingness is unrelated to the values) or run df, n = increment.impute.drop_null(df, {column!r}).",
        "frame.validation.from_unit_panel": RefusalSpec(
            "frame.validation.from_unit_panel",
            InvalidRequestError,
            lambda *, covered: (
                f"from_unit_panel does not support a CUPED covariate on a windowed "
                f"or retention metric: {', '.join(map(repr, covered))}. Those "
                f"metrics have no per-unit collapse this frame can attach a "
                f"pre-period covariate to. Aggregate to one row per unit yourself "
                f"(taking the pre-period value once) and use from_unit_summary "
                f"instead."
            ),
        ),
        "frame.validation.duplicate_breakout_column": "duplicate breakout column(s): {duplicates!r}",
        "frame.validation.breakout_reserved_internal": "breakout {name!r} is reserved for internal panel/moment fields",
        "frame.validation.breakout_overlaps_column": "breakout {name!r} overlaps {role} column {name!r}",
        "frame.validation.breakout_overlaps_metric": "breakout {name!r} overlaps metric input column {name!r}",
        "frame.validation.breakout_unit_stable": "breakout {name!r} is not unit-stable: {count} unit(s) have several values, including {shown}{more}",
    },
)
_raise = raiser(_REFUSALS)
COVARIATE_IMPUTE_ALL_NULL = _REFUSALS["frame.validation.metric_covariate_impute_all_null"]


_NULL_BREAKOUT = "__null__"


_BREAKOUT_RESERVED_COLUMNS = frozenset(
    {
        "unit_id",
        "group_id",
        "ds",
        "experiment_id",
        "metric",
        "n",
        "y",
        "x",
        "y_den",
        "d",
        *CENTERED_FIELDS,
    }
)


def _validate_columns(
    frame: nw.DataFrame[Any],
    *,
    roles: Sequence[tuple[str, str]],
    metrics: Sequence[MetricSpec],
    reject: Callable[..., NoReturn] | None = None,
) -> None:
    """Report every missing column at once, not one per round trip.

    *roles* is ``[(role_label, column_name), ...]`` for the non-metric
    columns (``unit``/``group`` for the summary shape; ``unit``/``group``/
    ``date`` for the panel shape).
    """
    present = set(frame.columns)
    missing: list[str] = []

    for role, col in roles:
        if col not in present:
            missing.append(f"{col!r} ({role})")

    for spec in metrics:
        for col in spec.source_columns:
            if col not in present:
                missing.append(f"{col!r} (metric {spec.name!r})")
    if missing:
        message = (
            f"columns not found in frame: {', '.join(missing)}. "
            f"Available columns: {sorted(present)}"
        )
        if reject is not None:
            reject(
                message=message,
                missing=missing,
                available=sorted(present),
                reason="missing_column",
            )
        refuse(_METRIC_MISSING, message=message, missing=missing, available=sorted(present))
    for spec in metrics:
        # Only meaningful here, where these names ARE frame columns: both sides
        # sum the same column, so the ratio is identically 1 with zero variance
        # and nothing to estimate. On the from_moments path the same names are
        # labels on precomputed moments and may legitimately coincide.
        if spec.type == "ratio" and spec.numerator == spec.denominator:
            if reject is not None:
                reject(
                    message=_RATIO_SAME_COLUMN.render(metric=spec.name, column=spec.numerator),
                    reason="degenerate_ratio",
                )
            refuse(_RATIO_SAME_COLUMN, metric=spec.name, column=spec.numerator)


def _is_nan_real(value: object) -> bool:
    """True only for a NaN real; an int or rational too large for a float is
    not NaN and falls through to the parser, which refuses it by value."""
    if not isinstance(value, Real) or isinstance(value, Integral):
        return False
    try:
        return math.isnan(float(value))
    except OverflowError:
        return False


def _parse_metric_value(value: object) -> float | None:
    """Parse one non-null metric cell as a finite float, or ``None`` on failure."""
    try:
        parsed = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return None if math.isnan(parsed) or math.isinf(parsed) else parsed


def _is_native_inferred_null(frame: nw.DataFrame[Any], column: str) -> bool:
    """Distinguish native null types from other unrecognized schemas."""
    if frame.implementation.value == "pyarrow":
        from pyarrow.types import is_null

        return is_null(frame.to_native().schema.field(column).type)
    if frame.implementation.value == "pandas":
        dtype = getattr(frame.to_native()[column].dtype, "pyarrow_dtype", None)
        if dtype is not None:
            from pyarrow import DataType
            from pyarrow.types import is_null

            return isinstance(dtype, DataType) and is_null(dtype)
    if frame.implementation.value == "polars":
        from polars import Null

        return frame.to_native().schema[column] == Null
    return False


def _coerce_metric_columns(frame: nw.DataFrame[Any], columns: Sequence[str]) -> nw.DataFrame[Any]:
    """Preserve numeric/null inputs and parse textual metrics consistently across backends."""
    for column in dict.fromkeys(columns):
        dtype = frame.schema[column]
        if isinstance(dtype, nw.Unknown) and _is_native_inferred_null(frame, column):
            frame = frame.with_columns(nw.col(column).cast(nw.Float64))
            continue
        if dtype.is_numeric():
            continue
        if isinstance(dtype, nw.Boolean):
            frame = frame.with_columns(nw.col(column).cast(nw.Float64))
            continue
        if isinstance(dtype, (nw.String, nw.Object)):
            parsed: list[float | None] = []
            offending: list[str] = []
            values = frame[column].to_list()
            is_null = frame.select(nw.col(column).is_null())[column].to_list()
            for value, null in zip(values, is_null, strict=True):
                if null or _is_nan_real(value):
                    parsed.append(None)
                    continue
                candidate = _parse_metric_value(value)
                if candidate is None:
                    offending.append(str(value))
                    parsed.append(None)
                else:
                    parsed.append(candidate)
            if offending:
                refuse(
                    _SOURCE_FRAME_COLUMN_DTYPE,
                    column=column,
                    dtype=str(dtype),
                    offending_values=offending[:5],
                )
            frame = frame.with_columns(
                nw.new_series(column, parsed, dtype=nw.Float64, backend=frame.implementation)
            )
            continue
        sample = [str(v) for v in frame[column].head(5).to_list()]
        refuse(_SOURCE_FRAME_COLUMN_DTYPE, column=column, dtype=str(dtype), offending_values=sample)
    return frame


def _reject_reserved_labels(labels: Sequence[str], *, group: str) -> None:
    reserved = sorted(set(labels) & {UNASSIGNED_LABEL, MIXED_ASSIGNMENT_LABEL})
    if reserved:
        refuse(
            _GROUP_LABEL_RESERVED,
            group=group,
            labels=tuple(reserved),
            route="rename these arms upstream before constructing the source",
        )


def _canonicalize_group_identity(frame: nw.DataFrame[Any], group: str) -> nw.DataFrame[Any]:
    """Validate wire identities and preserve missing groups before reduction."""
    if frame.schema[group] == nw.String and not frame.implementation.is_pandas_like():
        # A typed string column is already its canonical form, so distinct values
        # cannot collide. pandas reports a sniffed object column as String too,
        # so it always takes the pairing scan and the cast below.
        _reject_reserved_labels(frame[group].unique().drop_nulls().to_list(), group=group)
        return frame
    # No ``otherwise``: a when-without-else yields a true null on every backend,
    # whereas pandas 2.x casts a ``lit(None)`` branch to the string "None".
    canonical = nw.when(~_is_missing(frame, group)).then(nw.col(group).cast(nw.String)).alias(group)
    # Pair before deduplication: raw equality can hide differently cast labels.
    identities = (
        frame.select(
            nw.col(group).alias("__raw_group"),
            canonical.alias("__canonical_group"),
        )
        .filter(~nw.col("__canonical_group").is_null())
        .unique()
        .rows()
    )
    by_label: dict[str, list[object]] = {}
    for raw, label in identities:
        by_label.setdefault(label, []).append(_safe_error_value(raw))
    if len(by_label) != len(identities):
        collisions = tuple(
            (label, tuple(sorted(raws, key=repr)[:5]))
            for label, raws in sorted(by_label.items())
            if len(raws) > 1
        )[:5]
        refuse(
            _GROUP_LABEL_COLLISION,
            group=group,
            collision=collisions,
            route="cast the column to one consistent type upstream",
        )
    _reject_reserved_labels(list(by_label), group=group)
    return frame.with_columns(canonical)


def _group_counts(frame: nw.DataFrame[Any], group: str) -> dict[str, int]:
    """Row count per *group* value - callers guarantee one row per unit."""
    counts = frame.group_by(group).agg(nw.len().alias("__n__"))
    return {str(row[group]): int(row["__n__"]) for row in counts.iter_rows(named=True)}


def _reject_duplicate_units(frame: nw.DataFrame[Any], unit: str) -> None:
    """Duplicates inflate ``n`` and corrupt the variance, silently.

    They are also the signature of panel or event data handed to the summary
    constructor, so the error points at the constructor that wants that shape.
    """
    # Every key distinct (nulls count as one key here and in the group-by).
    if frame[unit].n_unique() == frame.shape[0]:
        return
    dupes = (
        frame.group_by(unit)
        .agg(nw.len().alias("__rows__"))
        .filter(nw.col("__rows__") > 1)
        .sort("__rows__", descending=True)
    )
    offenders = dupes[unit].to_list()
    refuse(
        _DUPLICATE_UNITS,
        count=len(offenders),
        units=tuple(offenders[:5]),
        route="If this frame is one row per unit per day, use from_unit_panel; otherwise repair duplicate rows.",
    )


def _reject_windowed_specs(specs: Sequence[MetricSpec]) -> None:
    """Reject windowed metrics on the one-row-per-unit constructor."""
    for spec in specs:
        if spec.window_days is not None:
            refuse(
                _CONSTRUCTOR_CAPABILITY,
                metric=spec.name,
                capability="window_days",
                value=spec.window_days,
                route="use from_unit_panel for a windowed metric",
            )
        if spec.type == "retention":
            refuse(
                _CONSTRUCTOR_CAPABILITY,
                metric=spec.name,
                capability="type=retention",
                value=spec.type,
                route="use from_unit_panel for a retention metric",
            )


def _validate_control(frame: nw.DataFrame[Any], group: str, control: str) -> None:
    observed = frame[group].unique().to_list()
    if control in observed:
        return
    if not observed:
        refuse(
            _CONTROL_MISSING,
            control=control,
            group=group,
            observed=(),
            route="declare a control present in the frame",
        )
    refuse(
        _CONTROL_MISSING,
        control=control,
        group=group,
        observed=tuple(sorted(observed, key=repr)),
        route="declare a control present in the frame",
    )


def _coerce_source_metrics(
    metrics: MetricsArg,
    *,
    design: Randomized | Encouragement | Observational | None,
) -> list[MetricSpec]:
    """Admit empty encouragement catalogs; compiled plans validate required outcomes."""
    if not metrics and isinstance(design, Encouragement):
        return []
    return coerce_metrics(metrics)


def _resolve_uptake_column(
    design: Randomized | Encouragement | Observational | None, uptake: str | None
) -> tuple[str | None, str]:
    """The uptake column to analyze, and the role label naming where it came from.

    An ``Encouragement`` design already declares its binary uptake fact, and
    on a dataframe the fact IS a column - so the design names the column and
    ``uptake=`` is only needed to override it, exactly as
    ``MetricSpec.value_column`` overrides ``MetricSpec.name``. The label rides
    along so a name that matches no column reports which declaration it came
    from.
    """
    if uptake is not None:
        return uptake, "uptake"
    if isinstance(design, Encouragement):
        return design.uptake.fact, "uptake, from design.uptake.fact"
    return None, "uptake"


def _resolve_design_and_plan(
    design: Randomized | Encouragement | Observational | None,
    control: str,
    plan: AnalysisPlan | None,
    synthesised_metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig] | None = None,
    *,
    source_id: str = "frame",
    source_mapping: dict[str, object] | None = None,
    transformations=(),
) -> tuple[Randomized | Encouragement | Observational, CompiledDecisionPlan]:
    """Derive/validate *design* against *control*, and resolve *plan* against
    *synthesised_metrics* (already built once by the caller from ``specs`` -
    resolving here too would synthesise every metric twice per construction).

    ``control`` must be the canonical string label: group labels are
    canonicalized to strings before control validation.
    ``design=None`` derives ``Randomized(control_group=str(control))`` - the
    common case, an explicit design is only needed for an encouragement or
    observational analysis. An explicit *design* whose ``control_group``
    disagrees with *control* is refused outright: a source has exactly one
    control arm, declared once, not twice with room to disagree.

    Shared by both ``FrameTotalsSource.from_frame`` and
    ``FramePanelSource.from_frame`` so the derivation rule and the refusal
    message have exactly one definition. Also calls
    ``increment.plan.refuse_observational_relative_margin`` (mirrors
    ``FramePanelSource.from_frame``'s encouragement/retention refusal
    below): both mechanism x capability checks fire here, at construction,
    once design and the resolved plan are both real construction state -
    not at first readout. The margin refusal is shared with every other
    source constructor that can carry ``design=Observational(...)``
    (``MomentsSource``, ``SqlPanelSource``, the native definitions path),
    not just this module's two.
    """
    if design is None:
        design = Randomized(control_group=str(control))
    elif str(design.control_group) != str(control):
        _raise(
            "frame.validation.design_control_group",
            control_group=design.control_group,
            control=control,
        )
    bound_plan = bind_automatic_sequential_plan(
        plan,
        synthesised_metrics,
        design=design,
        source_id=source_id,
        source_mapping=source_mapping or {},
        transformations=transformations,
        path="frame",
    )
    compiled_plan = compile_decision_plan(
        bound_plan,
        synthesised_metrics,
        path="frame",
        design=design,
        configs=configs,
        estimands=tuple(
            sorted({cell.estimand for cell in bound_plan.inference.registration.roster})
        )
        if (
            bound_plan is not None
            and bound_plan.inference is not None
            and bound_plan.inference.registration is not None
        )
        else None,
    )
    refuse_observational_relative_margin(design, compiled_plan)
    return design, compiled_plan


def _validate_uptake_binary(frame: nw.DataFrame[Any], uptake: str) -> None:
    """v1 encouragement analysis is binary-uptake-only: ``ArmStats.var_d()``
    relies on the closed form ``sum_d * (n - sum_d) / n``, which only holds
    when every value is exactly 0/1 (or True/False) - anything else
    silently corrupts that reduction downstream instead of failing here, at
    the edge.
    """
    observed = frame[uptake].unique().to_list()
    bad = [v for v in observed if v not in (0, 1, True, False)]
    if bad:
        refuse(
            _UPTAKE_NOT_BINARY,
            uptake=uptake,
            values=tuple(sorted(map(_safe_error_value, bad), key=repr)),
            route="provide one 0/1 or True/False uptake value per row",
        )


def _validate_conversion_binary(frame: nw.DataFrame[Any], specs: Sequence[MetricSpec]) -> None:
    """``type="conversion"`` names a Bernoulli rate, but it dispatches the
    same mean-variance model as ``type="mean"`` - nothing downstream would
    ever notice a magnitude column wearing the label; only the published
    number's NAME would be wrong. Mirror ``_validate_uptake_binary`` at the
    summary chokepoint (the panel path enforces the same contract at its
    unit collapse). Nulls/NaN are ruled on by the spec's declared missing
    policy, not here.
    """
    for spec in specs:
        if spec.type != "conversion":
            continue
        observed = frame[spec.y_column].unique().to_list()
        bad = [
            _safe_error_value(v)
            for v in observed
            if v is not None and v == v and v not in (0, 1, True, False)
        ]
        if bad:
            refuse(
                _CONVERSION_NOT_BINARY,
                metric=spec.name,
                column=spec.y_column,
                values=tuple(sorted(bad, key=repr)[:5]),
                route="declare type='mean' for magnitudes or provide a 0/1 conversion flag",
            )


def _reject_cluster_request(
    specs: Sequence[MetricSpec],
    *,
    cluster: str,
    design: Randomized | Encouragement | Observational | None,
    uptake: str | None,
) -> None:
    """Structural refusals for a declared cluster, before any data is read.

    Cluster-robust inference is built for mean/conversion/ratio metrics
    on the unadjusted randomized path, the encouragement path (ITT and
    the ADDITIVE ``late`` row; CUPED, ratio metrics and the
    complier-relative ``late`` row still refuse below), and the
    observational adjusted path (IPTW/DML/AIPW read the declaration from
    the source and cluster their influence-function variance); everything
    else refuses by name here rather than producing a silently
    unclustered (or misfiled) number.
    """
    mechanism = getattr(design, "mechanism", "randomized")
    if uptake is not None and mechanism != "encouragement":
        refuse(
            _CLUSTER_CAPABILITY,
            cluster=cluster,
            feature="uptake",
            metrics=(),
            uptake=uptake,
            mechanism=mechanism,
            route="drop cluster= or use an encouragement design",
        )
    covered = [s.name for s in specs if s.covariate is not None]
    if covered:
        refuse(
            _CLUSTER_CAPABILITY,
            cluster=cluster,
            feature="cuped",
            metrics=tuple(covered),
            uptake=uptake,
            mechanism=mechanism,
            route="drop cluster= or the covariate",
        )
    quantiles = [s.name for s in specs if s.type == "quantile"]
    if quantiles:
        refuse(
            _CLUSTER_CAPABILITY,
            cluster=cluster,
            feature="quantile",
            metrics=tuple(quantiles),
            uptake=uptake,
            mechanism=mechanism,
            route="drop cluster= or use mean/conversion/ratio metrics",
        )
    if mechanism == "encouragement" or uptake is not None:
        ratios = [s.name for s in specs if s.type == "ratio"]
        if ratios:
            refuse(
                _CLUSTER_CAPABILITY,
                cluster=cluster,
                feature="ratio",
                metrics=tuple(ratios),
                uptake=uptake,
                mechanism=mechanism,
                route="drop cluster= or ratio metrics under encouragement",
            )


def _validate_cluster_labels(
    frame: nw.DataFrame[Any], *, cluster: str, group: str, enforce_purity: bool = True
) -> None:
    """Data-quality refusals for the cluster column, mirroring the group-null
    policy: loud and named, never silently absorbed (there is no "exclude"
    knob for cluster labels - a unit without a randomization-grain label has
    no variance contribution to file anywhere).

    ``enforce_purity`` gates the arm-purity check only: a declared
    cluster-randomization (or encouragement) requires every cluster to be a
    pure arm; a declared observational design carries dependence clusters
    that may legitimately span both treatment values, so its caller passes
    ``enforce_purity=False``. The null-label check always applies -- an
    observational design still needs every unit's dependence cluster known.
    """
    n_missing = _missing_count(frame, cluster)
    if n_missing:
        refuse(
            _CLUSTER_LABELS,
            cluster=cluster,
            reason="null_label",
            missing=n_missing,
            route="repair cluster labels upstream; there is no exclude option",
        )
    if not enforce_purity:
        return
    groups_per_cluster = frame.group_by(cluster).agg(nw.col(group).n_unique().alias("__n_groups__"))
    if (groups_per_cluster["__n_groups__"] > 1).any():
        span = groups_per_cluster.filter(nw.col("__n_groups__") > 1)
        offenders = sorted(str(v) for v in span[cluster].to_list())[:5]
        refuse(
            _CLUSTER_LABELS,
            cluster=cluster,
            reason="spanning_label",
            spanning=int(span.shape[0]),
            examples=tuple(offenders),
            route="ensure each randomization cluster belongs to exactly one arm",
        )


def _day_axis_is_numeric(dtype: nw.dtypes.DType) -> bool:
    """A day axis is an explicit numeric day index unless it is a real
    calendar Date/Datetime - including a pandas date column, which
    narwhals reports as an opaque non-numeric (object) dtype rather than
    Date/Datetime on its own, so it still measures like a calendar axis.
    """
    return dtype.is_numeric() and not isinstance(dtype, nw.Date | nw.Datetime)


def _validate_day_axis_match(
    ds_dtype: nw.dtypes.DType, anchor_dtype: nw.dtypes.DType, *, ds_name: str, anchor_name: str
) -> None:
    """Refuse when a day axis and its anchor disagree on numeric vs.
    calendar semantics - measuring elapsed days between an explicit
    numeric day index and a real calendar date is nonsensical either way:
    a numeric axis read as microseconds turns day 1 into ~1.16e-11
    elapsed days, and a calendar axis read as a raw number filters every
    row as pre- or post-exposure depending on which side is which.
    """
    if _day_axis_is_numeric(ds_dtype) != _day_axis_is_numeric(anchor_dtype):
        _raise(
            "frame.validation.dtype_disagree_day",
            ds_name=ds_name,
            ds_dtype=ds_dtype,
            anchor_name=anchor_name,
            anchor_dtype=anchor_dtype,
        )


def _validate_day_grain_columns(
    frame: nw.DataFrame[Any], columns: Sequence[tuple[str, str]]
) -> None:
    """Validate every day-grain column named in *columns*, once, at
    construction - the boundary that owns both invariants below, rather
    than a per-method check inside daily/asof windowing.

    ``from_unit_panel`` is day-grain and the caller owns day bucketing;
    no timezone or day-boundary shifting is applied here. A timezone-aware
    column has no unambiguous calendar day until the caller picks that
    boundary, and letting it through used to crash deep inside the day
    arithmetic with a raw backend cast error naming nothing.

    When both a ``"date"`` and an ``"exposure_date"`` role are present in
    *columns*, also refuse here if they disagree on numeric-vs-calendar
    day-axis kind (:func:`_validate_day_axis_match`). Checking this at
    construction - rather than lazily inside ``_with_day_index`` - means a
    mismatched source can no longer construct successfully and answer
    ``unit_counts()`` before a later ``daily()``/``asof()`` call on the
    SAME source refuses it.
    """
    offenders = [
        (role, col, dtype.time_zone)
        for role, col in columns
        if isinstance(dtype := frame.schema[col], nw.Datetime) and dtype.time_zone is not None
    ]
    if offenders:
        refuse(
            _DAY_GRAIN_COLUMNS,
            columns=tuple(offenders),
            route="bucket timestamps upstream and pass naive day-grain values",
        )
    dtypes = {role: frame.schema[col] for role, col in columns}
    if "date" in dtypes and "exposure_date" in dtypes:
        _validate_day_axis_match(
            dtypes["date"], dtypes["exposure_date"], ds_name="date", anchor_name="exposure_date"
        )


def _is_missing(frame: nw.DataFrame[Any], col: str) -> nw.Expr:
    """Null-or-NaN as one predicate; ``is_nan`` only types on float columns."""
    expr = nw.col(col).is_null()
    if frame.schema[col] in (nw.Float32, nw.Float64):
        expr = expr | nw.col(col).is_nan()
    return expr


def _missing_count(frame: nw.DataFrame[Any], col: str) -> int:
    return int(frame.select(_is_missing(frame, col).cast(nw.Int64).sum()).item())


def _reject_null_identity(frame: nw.DataFrame[Any], col: str, *, role: str) -> None:
    """Refuse a null value in an identity column.

    A null unit id or day silently manufactures a phantom key: it fails
    the densification join on some backends and is zero-filled on
    others, hashes as ``str(None)`` into a CATE fold, or drops the row's
    outcome with no accounting at all - a different, backend-dependent
    answer for the same input.
    """
    n_missing = _missing_count(frame, col)
    if n_missing:
        refuse(
            _NULL_IDENTITY,
            role=role,
            column=col,
            count=n_missing,
            route=f"repair the {role} column upstream before construction",
        )


def _reject_non_finite_metrics(frame: nw.DataFrame[Any], specs: Sequence[MetricSpec]) -> None:
    """Refuse a non-finite (+-inf) metric, denominator, or covariate value.

    An undeclared null is at least a documented, backend-consistent
    policy (see :func:`_enforce_missing_policy`); inf/-inf is neither
    null nor NaN, so it survives every missing-value check and only
    diverges by backend at the first arithmetic sum (inf on pandas, NaN
    on polars/pyarrow once summed against other values).
    """
    checked: set[str] = set()
    for spec in specs:
        for col in (spec.y_column, spec.denominator, spec.covariate):
            if col is None or col in checked:
                continue
            checked.add(col)
            if not frame.schema[col].is_numeric():
                continue
            bad = int(
                frame.select(
                    ((~nw.col(col).is_finite()) & (~_is_missing(frame, col))).cast(nw.Int64).sum()
                ).item()
            )
            if bad:
                refuse(
                    _NON_FINITE,
                    role="metric/denominator/covariate",
                    column=col,
                    count=bad,
                    route="replace inf/-inf with finite values or declared nulls upstream",
                )


def _reject_non_finite_day_axis(frame: nw.DataFrame[Any], column: str, *, role: str) -> None:
    """Refuse a non-finite (+-inf) value on a numeric day-axis column.

    A numeric day axis is compared and subtracted directly (see
    :func:`_day_axis_elapsed`); an infinite value survives every null/NaN
    missing-value check and only diverges by backend at the first
    arithmetic comparison - silently distorting window maturity math or
    excluding outcomes during day-index filtering while the unit still
    counts. Applies to the day column and to the exposure anchor, which
    is the origin every day index is measured from. A calendar
    Date/Datetime column has no literal +-inf representation, so this
    only applies to a numeric axis.
    """
    if not _day_axis_is_numeric(frame.schema[column]):
        return
    bad = int(
        frame.select(
            ((~nw.col(column).is_finite()) & (~_is_missing(frame, column))).cast(nw.Int64).sum()
        ).item()
    )
    if bad:
        refuse(
            _NON_FINITE,
            role=role,
            column=column,
            count=bad,
            route="replace inf/-inf with finite day-axis values upstream",
        )


def _validate_frame_boundary(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    date: str | None,
    exposure_date: str | None = None,
    specs: Sequence[MetricSpec],
) -> None:
    """Refuse a null unit id, a null/NaN or non-finite date or exposure
    anchor, or a non-finite metric value before any duplicate check or
    spine construction touches them - the single boundary that owns these
    invariants, rather than scattering the checks across each constructor.
    """
    _reject_null_identity(frame, unit, role="unit")
    if date is not None:
        _reject_null_identity(frame, date, role="date")
        _reject_non_finite_day_axis(frame, date, role="date")
    if exposure_date is not None:
        # The anchor is the origin every day index is measured from, so an
        # infinite value there corrupts the same arithmetic as an infinite day.
        _reject_non_finite_day_axis(frame, exposure_date, role="exposure_date")
    _reject_non_finite_metrics(frame, specs)


def _resolve_unassigned(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    group: str,
    on_unassigned: Literal["error", "exclude"],
    constructor: str,
) -> tuple[nw.DataFrame[Any], int]:
    """Enforce the assignment-null policy; returns ``(frame, excluded_units)``.

    Default ``"error"``: refuse, naming the ``on_unassigned="exclude"``
    knob. ``"exclude"``: drop every row for a UNIT with any null-labelled
    row - the unit is the entity, so a unit with a valid label on one row
    (e.g. one day of a panel) and a null label on another must not keep
    its arm while the null row's outcome silently vanishes into a
    densified zero. Returns how many units that removed - the caller
    surfaces the count in ``unit_counts()`` (under ``UNASSIGNED_LABEL``)
    and beside the SRM check, so the exclusion is always visible and
    never earns an arm, an estimate, or a chi-square degree of freedom.
    """
    if on_unassigned not in ("error", "exclude"):
        refuse(_ON_UNASSIGNED_INVALID, value=on_unassigned, allowed={"error", "exclude"})
    predicate = _is_missing(frame, group)
    n_rows = int(frame.select(predicate.cast(nw.Int64).sum()).item())
    if n_rows == 0:
        return frame, 0
    if on_unassigned == "error":
        refuse(
            _UNASSIGNED,
            count=n_rows,
            group=group,
            units=tuple(sorted(set(frame.filter(predicate)[unit].to_list()), key=repr)[:5]),
            route=f"pass on_unassigned='exclude' to {constructor}, or repair the assignment upstream",
            constructor=constructor,
        )
    unassigned_units = list(set(frame.filter(predicate)[unit].to_list()))
    kept = frame.filter(~nw.col(unit).is_in(unassigned_units))
    return kept, len(unassigned_units)


def _resolve_null_exposure(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    exposure_date: str,
    on_unassigned: Literal["error", "exclude"],
    constructor: str,
) -> tuple[nw.DataFrame[Any], int]:
    """Enforce the exposure-null policy; returns ``(frame, excluded_units)``.

    A unit with a null exposure date has no day 0: window/band arithmetic
    and the encouragement uptake window all read this anchor, so the
    unit's rows would silently vanish from windowed moments while
    ``unit_counts()``/``srm()`` still counted it - an invisibly selected
    subpopulation (null exposure usually means late/never-activated, so
    the selection correlates with the outcome). Same single unusable-unit
    accounting as :func:`_resolve_unassigned`: ``"error"`` refuses naming
    the knob; ``"exclude"`` drops the unit's rows and counts it once under
    ``UNASSIGNED_LABEL``. The per-unit constancy check
    (:func:`_validate_exposure_date`) runs first, so a null-exposure unit
    is null on EVERY row - excluding the null rows excludes the unit
    whole, keeping moments and SRM consistent.
    """
    predicate = _is_missing(frame, exposure_date)
    n_units = int(frame.filter(predicate)[unit].n_unique())
    if n_units == 0:
        return frame, 0
    if on_unassigned == "error":
        refuse(
            _NULL_EXPOSURE,
            count=n_units,
            column=exposure_date,
            constructor=constructor,
            route="repair exposure dates upstream or pass on_unassigned='exclude'",
        )
    return frame.filter(~predicate), n_units


def _metric_missing_error(spec: MetricSpec, col: str, n_missing: int) -> NoReturn:
    _raise(
        "frame.validation.metric_value_missing",
        metric=spec.name,
        column=col,
        n_missing=n_missing,
    )


def _enforce_missing_policy(
    frame: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    *,
    shape: Literal["summary", "panel"],
) -> nw.DataFrame[Any]:
    """Validate every spec's declared null/NaN policy against the data.

    Summary shape: purely a validation - ``"zero"``/``"drop"`` are applied
    per spec at moment construction (``_moment_rows``/``unit_frame``), so
    two metrics reading one column under different declarations never
    contaminate each other. Panel shape: a declared ``"zero"`` column has
    its NaN normalized to null here (NaN is a value on polars/pyarrow but
    missing on pandas - normalizing before ``_fact_max_observed_dates``
    and ``_densify_panel`` makes "not genuinely observed" mean the same
    thing on every backend), and the actual zero is applied by
    densification's own ``fill_null(0.0)``, exactly as for absent
    (unit, day) cells. ``"drop"`` is refused outright on the panel;
    densification zero-fills absent cells, so dropping a null row would
    silently mean zero anyway, not complete-case.
    """
    counts: dict[str, int] = {}

    def miss(col: str) -> int:
        if col not in counts:
            counts[col] = _missing_count(frame, col)
        return counts[col]

    fill_zero: set[str] = set()
    for spec in specs:
        value_cols = [spec.y_column] + ([spec.denominator] if spec.denominator else [])
        if spec.missing == "drop" and shape == "panel":
            refuse(_PANEL_MISSING_DROP, missing=spec.missing)
        dropped = 0
        for col in dict.fromkeys(value_cols):
            n_missing = miss(col)
            if n_missing == 0:
                continue
            if spec.missing == "error":
                _metric_missing_error(spec, col, n_missing)
            if spec.missing == "zero" and shape == "panel":
                fill_zero.add(col)
            if spec.missing == "drop":
                dropped += n_missing
        if dropped:
            _warn(
                "frame.validation.metric_missing_drop",
                name=spec.name,
                dropped=dropped,
                stacklevel=4,
            )
        if spec.covariate is not None:
            n_missing = miss(spec.covariate)
            if n_missing == 0:
                continue
            if spec.covariate_missing == "error":
                _raise(
                    "frame.validation.metric_covariate_column",
                    metric=spec.name,
                    covariate=spec.covariate,
                    n_missing=n_missing,
                )
            n_observed = frame.shape[0] - n_missing
            if spec.covariate_missing == "impute" and n_observed == 0:
                _raise(
                    "frame.validation.metric_covariate_impute_all_null",
                    metric=spec.name,
                    covariate=spec.covariate,
                )
            filled = (
                "the pooled mean of the observed values"
                if spec.covariate_missing == "impute"
                else "zero"
            )
            _warn(
                "frame.validation.metric_covariate_missing_imputed",
                name=spec.name,
                covariate=spec.covariate,
                n_missing=n_missing,
                filled=filled,
                stacklevel=4,
            )
    if fill_zero:
        nan_cols = [
            col for col in sorted(fill_zero) if frame.schema[col] in (nw.Float32, nw.Float64)
        ]
        if nan_cols:
            frame = frame.with_columns(*(nw.col(col).fill_nan(None).alias(col) for col in nan_cols))
    return frame


def _reject_covariates(specs: Sequence[MetricSpec]) -> None:
    """A CUPED covariate is a single pre-period value per unit.

    `FramePanelSource._resolve_panel_covariate` resolves exactly this for a
    plain (unwindowed, non-retention) metric -- taking a panel covariate's
    per-unit value when it is constant across a unit's own rows, refusing
    by name when it genuinely varies -- and `moments(grain="total")` joins
    the resolved value onto the same per-unit collapse `unit_frame()`
    already uses. A windowed or retention metric has no per-unit collapse
    to attach a covariate to (see `_windowed_moment_rows`): that
    combination still refuses here.
    """
    covered = [
        s.name
        for s in specs
        if s.covariate is not None and (s.window_days is not None or s.type == "retention")
    ]
    if covered:
        _raise("frame.validation.from_unit_panel", covered=covered)


def _normalize_breakout_names(
    breakouts: Sequence[str],
    *,
    unit: str,
    group: str,
    date: str,
    uptake: str | None,
    exposure_date: str | None,
    specs: Sequence[MetricSpec],
) -> tuple[str, ...]:
    """Validate a declared set of unit-stable breakout column names."""
    names = tuple(breakouts)
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        _raise("frame.validation.duplicate_breakout_column", duplicates=duplicates)

    roles = {unit: "unit", group: "group", date: "date"}
    if uptake is not None:
        roles[uptake] = "uptake"
    if exposure_date is not None:
        roles[exposure_date] = "exposure_date"
    metric_columns = {column for spec in specs for column in spec.source_columns}
    for name in names:
        if name.startswith("_") or name in _BREAKOUT_RESERVED_COLUMNS:
            _raise("frame.validation.breakout_reserved_internal", name=name)
        if name in roles:
            _raise("frame.validation.breakout_overlaps_column", name=name, role=roles[name])
        if name in metric_columns:
            _raise("frame.validation.breakout_overlaps_metric", name=name)
    return names


def _normalized_breakout_expr(frame: nw.DataFrame[Any], name: str) -> nw.Expr:
    """A string identity value, with nulls visible in the declared segment."""
    value = nw.col(name).cast(nw.String)
    if isinstance(frame.schema[name], nw.Boolean):
        value = nw.when(nw.col(name)).then(nw.lit("true")).otherwise(nw.lit("false"))
    return (
        nw.when(_is_missing(frame, name)).then(nw.lit(_NULL_BREAKOUT)).otherwise(value).alias(name)
    )


def _unit_identity(
    frame: nw.DataFrame[Any], *, unit: str, group: str, breakouts: Sequence[str]
) -> nw.DataFrame[Any]:
    """One validated group/breakout identity row per unit."""
    normalized = frame.select(
        nw.col(unit).alias("unit_id"),
        nw.col(group).alias("group_id"),
        *[_normalized_breakout_expr(frame, name) for name in breakouts],
    )
    for name in breakouts:
        values = normalized.select("unit_id", name).unique()
        conflicts = (
            values.group_by("unit_id").agg(nw.len().alias("__n__")).filter(nw.col("__n__") > 1)
        )
        if conflicts.is_empty():
            continue
        offenders = conflicts["unit_id"].to_list()
        shown = ", ".join(repr(value) for value in offenders[:5])
        more = f" (and {len(offenders) - 5} more)" if len(offenders) > 5 else ""
        _raise(
            "frame.validation.breakout_unit_stable",
            name=name,
            count=len(offenders),
            shown=shown,
            more=more,
        )
    return normalized.unique(subset=["unit_id"])


def _validate_exposure_date(
    frame: nw.DataFrame[Any], *, unit: str, exposure_date: str | None, specs: Sequence[MetricSpec]
) -> None:
    """A windowed or retention metric's day-0 anchor must be an explicit,
    per-unit-constant exposure date.

    Deriving it from a unit's first OBSERVED row (as the encouragement
    design's uptake window used to, before this task) would let a
    densified zero-fill row or the frame's own row order silently define
    the window - required whenever any spec carries ``window_days`` or
    ``type="retention"``. When given (whether or not it is required),
    every unit must map to exactly one distinct value - a unit whose
    exposure date varies by row has no single day 0 to window against.
    """
    needs_exposure = any(s.window_days is not None or s.type == "retention" for s in specs)
    if exposure_date is None:
        if needs_exposure:
            refuse(
                _EXPOSURE_DATE,
                issue="missing required declaration for windowed/retention metrics",
                column=None,
                count=0,
                examples=tuple(
                    s.name for s in specs if s.window_days is not None or s.type == "retention"
                ),
                route="pass exposure_date=<column name>",
            )
        return

    conflicts = (
        frame.select(nw.col(unit).alias("__u__"), nw.col(exposure_date).alias("__e__"))
        .unique()
        .group_by("__u__")
        .agg(nw.len().alias("__n__"))
        .filter(nw.col("__n__") > 1)
    )
    if conflicts.is_empty():
        return

    offenders = conflicts["__u__"].to_list()
    refuse(
        _EXPOSURE_DATE,
        issue="per-unit values are not constant",
        column=exposure_date,
        count=len(offenders),
        examples=tuple(_safe_error_value(unit_id) for unit_id in offenders[:5]),
        route="provide one exposure_date value per unit across the panel",
    )


def _reject_duplicate_unit_days(frame: nw.DataFrame[Any], unit: str, date: str) -> None:
    """At most one row per (unit, day); duplicates are a data-quality signature.

    ``daily()`` and the whole-panel collapse both sum values per (unit, day)
    slot - a duplicate silently doubles that slot, the same class of hazard
    as duplicate units on the summary path.
    """
    dupes = (
        frame.group_by([unit, date])
        .agg(nw.len().alias("__rows__"))
        .filter(nw.col("__rows__") > 1)
        .sort("__rows__", descending=True)
    )
    if dupes.is_empty():
        return

    offenders = [(row[unit], row[date]) for row in dupes.iter_rows(named=True)]
    refuse(
        _DUPLICATE_UNIT_DAYS,
        unit=unit,
        date=date,
        count=len(offenders),
        examples=tuple(
            (_safe_error_value(unit_id), _safe_error_value(day)) for unit_id, day in offenders[:5]
        ),
        route="aggregate same-day events onto one row before from_unit_panel",
    )


def _validate_single_group_per_unit(frame: nw.DataFrame[Any], unit: str, group: str) -> None:
    """A unit's variant must be constant for the whole panel.

    Silently picking one of several observed groups for a unit would be
    exactly the kind of undocumented convention this path exists to avoid.
    """
    pairs = frame.select(nw.col(unit).alias("__u__"), nw.col(group).alias("__g__")).unique()
    conflicts = pairs.group_by("__u__").agg(nw.len().alias("__n__")).filter(nw.col("__n__") > 1)
    if conflicts.is_empty():
        return

    offenders = conflicts["__u__"].to_list()
    refuse(
        _MULTI_GROUP_UNIT,
        unit=unit,
        group=group,
        count=len(offenders),
        examples=tuple(_safe_error_value(unit_id) for unit_id in offenders[:5]),
        route="ensure each unit has one group across the whole panel",
    )


def _validate_switchback_labels(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    group: str,
    reject_unit: Callable[..., NoReturn],
    reject_group: Callable[..., NoReturn],
) -> None:
    """Require non-missing switchback unit and group labels to be nonempty strings."""
    for field, column, reject in (
        ("unit", unit, reject_unit),
        ("group", group, reject_group),
    ):
        if _missing_count(frame, column):
            continue
        series = frame[column]
        if frame.schema[column] == nw.String:
            # A string column holds nothing but strings; only emptiness needs a scan.
            non_strings: list[object] = []
            empty_count = int((series.str.len_chars() == 0).sum())
            empty: list[object] = [""] * min(empty_count, 5)
        else:
            values = series.to_list()
            non_strings = [value for value in values if not isinstance(value, str)]
            empty = [value for value in values if isinstance(value, str) and not value]
        if non_strings:
            reject(
                message=(
                    f"switchback {field} column {column!r} must contain only "
                    f"nonempty strings; found {non_strings[:5]!r}"
                ),
                column=column,
                field=field,
                values=non_strings[:5],
                reason="non_string_label",
            )
        if empty:
            reject(
                message=f"switchback {field} column {column!r} contains empty labels",
                column=column,
                field=field,
                values=empty[:5],
                reason="empty_label",
            )


def _range_cardinality(values: range) -> int:
    """Count a range with integer arithmetic, without materializing it."""
    if values.step > 0:
        distance = values.stop - values.start
        return 0 if distance <= 0 else (distance - 1) // values.step + 1
    distance = values.start - values.stop
    step = -values.step
    return 0 if distance <= 0 else (distance - 1) // step + 1


def _first_missing_switchback_step(observed_steps: set[int], total_steps: int) -> int | None:
    """Find the first missing nonnegative step without scanning the whole range."""
    candidate = 0
    for current_step in sorted(observed_steps):
        if current_step < candidate:
            continue
        if current_step >= total_steps:
            break
        if current_step > candidate:
            return candidate
        candidate += 1
    return candidate if candidate < total_steps else None


def _switchback_integral_domain(
    frame: nw.DataFrame[Any],
    *,
    column: str,
    expected: Sequence[int] | range | None = None,
    label: str,
    reject: Callable[..., NoReturn],
) -> tuple[int, ...]:
    """Return an exact integer domain, rejecting nulls, booleans, and floats."""
    series = frame[column]
    strict_integer = frame.schema[column].is_integer() and series.null_count() == 0
    values = series.unique().to_list() if strict_integer else series.to_list()
    if not values:
        reject(
            message=f"switchback {label} column {column!r} is empty",
            column=column,
            field=label,
            reason="empty_domain",
        )
    bad = (
        []
        if strict_integer
        else [
            value for value in values if isinstance(value, bool) or not isinstance(value, Integral)
        ]
    )
    if bad:
        shown = ", ".join(repr(value) for value in bad[:5])
        reject(
            message=(
                f"switchback {label} column {column!r} must contain exact "
                f"non-boolean integers; found {shown}"
            ),
            column=column,
            field=label,
            values=bad[:5],
            reason="non_integral_domain",
        )
    observed_set = {int(value) for value in values}
    observed = tuple(sorted(observed_set))
    if expected is not None:
        if isinstance(expected, range):
            expected_count = _range_cardinality(expected)
            matches_expected = len(observed_set) == expected_count and all(
                value in expected for value in observed_set
            )
        else:
            expected_count = len(expected)
            matches_expected = len(observed_set) == expected_count and all(
                value in observed_set for value in expected
            )
        if not matches_expected:
            if isinstance(expected, range):
                expected_display: object = (
                    list(expected) if expected_count <= 1000 else repr(expected)
                )
            else:
                expected_display = list(expected)
            expected_text = (
                expected_display if isinstance(expected_display, str) else repr(expected_display)
            )
            reject(
                message=(
                    f"switchback {label} domain must be {expected_text}; found {list(observed)!r}"
                ),
                column=column,
                field=label,
                expected=expected_display,
                observed=list(observed),
                reason="unexpected_domain",
            )
    return observed


def _validate_switchback_domains(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    cycle: str,
    period: str,
    step: str,
    total_steps: int,
    reject_missingness: Callable[..., NoReturn],
    reject_domain: Callable[..., NoReturn],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Validate the fixed cycle, period, and step axes and non-null units."""
    if _missing_count(frame, unit):
        reject_missingness(
            message=f"switchback unit column {unit!r} contains null/NaN values",
            column=unit,
            reason="null_unit",
        )
    cycles = _switchback_integral_domain(
        frame,
        column=cycle,
        label="cycle",
        reject=reject_domain,
    )
    if cycles[0] != 0 or len(cycles) != cycles[-1] + 1:
        reject_domain(
            message=(f"switchback cycle domain must be contiguous 0..C-1; found {list(cycles)!r}"),
            column=cycle,
            field="cycle",
            observed=list(cycles),
            reason="non_contiguous_cycle_domain",
        )
    periods = _switchback_integral_domain(
        frame,
        column=period,
        expected=(0, 1),
        label="period",
        reject=reject_domain,
    )
    steps = _switchback_integral_domain(
        frame,
        column=step,
        expected=range(total_steps),
        label="step",
        reject=reject_domain,
    )
    return cycles, periods, steps


@dataclass(frozen=True, slots=True)
class SwitchbackSchedule:
    """Realized two-period orders of a validated, complete switchback schedule.

    ``units`` lists the distinct unit labels in frame-engine sort order and
    ``control_first`` holds one flag per unit-cycle in that unit order with
    cycles ascending: True when the realized order is control then treatment.
    """

    units: tuple[object, ...]
    control_first: np.ndarray

    @property
    def ct_cycles(self) -> int:
        return int(self.control_first.sum())

    @property
    def tc_cycles(self) -> int:
        return int(self.control_first.size) - self.ct_cycles


def _validate_switchback_schedule(
    frame: nw.DataFrame[Any],
    *,
    unit: str,
    cycle: str,
    period: str,
    step: str,
    group: str,
    control: str,
    treatment: str,
    cycles: Sequence[int],
    total_steps: int,
    reject_missingness: Callable[..., NoReturn],
    reject_schedule: Callable[..., NoReturn],
) -> SwitchbackSchedule:
    """Validate one complete two-period schedule and return its realized orders."""
    if _missing_count(frame, group):
        reject_missingness(
            message=f"switchback group column {group!r} contains null/NaN values",
            column=group,
            reason="null_group",
        )
    keys = [unit, cycle, period, step]
    duplicates = frame.group_by(keys).agg(nw.len().alias("__rows__")).filter(nw.col("__rows__") > 1)
    if not duplicates.is_empty():
        reject_schedule(
            message=(
                "switchback schedule contains duplicate rows for a "
                f"(unit, cycle, period, step) cell: {duplicates.shape[0]} duplicate cell(s)"
            ),
            unit=unit,
            cycle=cycle,
            period=period,
            step=step,
            duplicate_cells=int(duplicates.shape[0]),
            reason="duplicate_schedule_cells",
        )

    unit_rows = frame.group_by(unit).agg(nw.len().alias(step))
    observed_by_unit: dict[object, int] = {
        unit_value: int(count) for unit_value, count in unit_rows.select(unit, step).iter_rows()
    }

    ordered_units = sorted(observed_by_unit, key=repr)
    cycle_count = _range_cardinality(cycles) if isinstance(cycles, range) else len(cycles)
    expected_per_unit = cycle_count * 2 * total_steps
    missing_by_unit = {
        value: expected_per_unit - observed_by_unit[value]
        for value in ordered_units
        if observed_by_unit[value] < expected_per_unit
    }
    missing_count = sum(missing_by_unit.values())
    if missing_count:
        # Exact step identities are needed only to render an incomplete schedule.
        observed_steps: dict[object, dict[tuple[int, int], set[int]]] = {}
        for unit_value, current_cycle, current_period, current_step in frame.select(
            *keys
        ).iter_rows():
            if unit_value in missing_by_unit:
                observed_steps.setdefault(unit_value, {}).setdefault(
                    (int(current_cycle), int(current_period)), set()
                ).add(int(current_step))
        missing_examples: list[tuple[object, int, int, int]] = []
        for unit_value in missing_by_unit:
            unit_steps = observed_steps.get(unit_value, {})
            for current_cycle in cycles:
                for current_period in (0, 1):
                    missing_step = _first_missing_switchback_step(
                        unit_steps.get((current_cycle, current_period), set()),
                        total_steps,
                    )
                    if missing_step is not None:
                        missing_examples.append(
                            (unit_value, current_cycle, current_period, missing_step)
                        )
                        if len(missing_examples) == 5:
                            break
                if len(missing_examples) == 5:
                    break
            if len(missing_examples) == 5:
                break
        shown = ", ".join(repr(item) for item in missing_examples)
        reject_schedule(
            message=(
                f"switchback schedule is incomplete: missing {missing_count} "
                f"(unit, cycle, period, step) cell(s), including {shown}"
            ),
            unit=unit,
            cycle=cycle,
            period=period,
            step=step,
            missing_cells=missing_count,
            reason="incomplete_schedule",
        )

    # The schedule is complete: every unit holds every cycle and both periods,
    # so one row per cell sorted by unit, cycle, period reshapes exactly.
    cells = (
        frame.group_by([unit, cycle, period])
        .agg(nw.col(group).n_unique().alias("__arms__"), nw.col(group).min().alias(group))
        .sort(unit, cycle, period)
    )
    shape = (len(observed_by_unit), cycle_count, 2)
    single = np.asarray(cells["__arms__"].to_list(), dtype=np.int64).reshape(shape) == 1
    labels = np.asarray(cells[group].to_list(), dtype=object).reshape(shape)
    units = tuple(cells[unit].gather_every(cycle_count * 2).to_list())
    both_single = single.all(axis=2)
    control_first = both_single & (labels[..., 0] == control) & (labels[..., 1] == treatment)
    treatment_first = both_single & (labels[..., 0] == treatment) & (labels[..., 1] == control)
    valid = control_first | treatment_first
    if not valid.all():
        # Report the first offending unit-cycle in stable unit order.
        offending = next(
            (index, offset)
            for index in sorted(range(len(units)), key=lambda index: repr(units[index]))
            for offset in range(cycle_count)
            if not valid[index, offset]
        )
        unit_value, current_cycle = units[offending[0]], cycles[offending[1]]
        if not single[offending].all():
            reject_schedule(
                message=(
                    "switchback schedule has duplicate/conflicting arms within "
                    "a unit-cycle-period cell"
                ),
                unit=unit_value,
                cycle=current_cycle,
                period=period,
                reason="conflicting_arms",
            )
        sequence = tuple(labels[offending].tolist())
        if any(label not in (control, treatment) for label in sequence):
            reject_schedule(
                message=(
                    f"switchback schedule contains a third arm in unit {unit_value!r}, "
                    f"cycle {current_cycle}: {sequence!r}"
                ),
                unit=unit_value,
                cycle=current_cycle,
                period=period,
                observed_arms=sequence,
                reason="third_arm",
            )
        reject_schedule(
            message=(
                "switchback schedule requires exactly one control/treatment "
                f"period per unit-cycle; got {sequence!r}"
            ),
            unit=unit_value,
            cycle=current_cycle,
            period=period,
            observed_arms=sequence,
            reason="invalid_period_assignment",
        )
    return SwitchbackSchedule(units=units, control_first=control_first)


def _validate_switchback_metric_values(
    frame: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    *,
    reject_missingness: Callable[..., NoReturn],
    reject_numeric: Callable[..., NoReturn],
) -> None:
    """Reject missing, non-finite, and non-binary switchback outcomes."""
    for spec in specs:
        n_missing = _missing_count(frame, spec.y_column)
        if n_missing:
            reject_missingness(
                message=(
                    f"metric {spec.name!r}: column {spec.y_column!r} has "
                    f"{n_missing} null/NaN value(s); switchback outcomes must be "
                    "observed and finite"
                ),
                metric=spec.name,
                column=spec.y_column,
                missing_values=n_missing,
                reason="null_metric",
            )
        series = frame[spec.y_column]
        dtype = frame.schema[spec.y_column]
        if spec.type == "conversion":
            values = series.to_list()
            bad = [
                value
                for value in values
                if not (
                    isinstance(value, bool)
                    or (isinstance(value, Real) and not isinstance(value, bool) and value in (0, 1))
                )
            ]
            if bad:
                reject_numeric(
                    message=(
                        f"metric {spec.name!r}: conversion column {spec.y_column!r} "
                        f"must be boolean or 0/1; found {bad[:5]!r}"
                    ),
                    metric=spec.name,
                    column=spec.y_column,
                    invalid_values=bad[:5],
                    reason="non_binary_conversion",
                )
        else:
            bad: list[object] = []
            if dtype.is_float():
                # Nulls and NaN were rejected above, so only infinities remain.
                bad = series.filter(~series.is_finite()).to_list()
            elif not dtype.is_integer():
                for value in series.to_list():
                    try:
                        finite = math.isfinite(float(value))
                    except (TypeError, ValueError, OverflowError):
                        finite = False
                    if not finite:
                        bad.append(value)
            if bad:
                reject_numeric(
                    message=(
                        f"metric {spec.name!r}: column {spec.y_column!r} must contain "
                        f"finite numeric values; found {bad[:5]!r}"
                    ),
                    metric=spec.name,
                    column=spec.y_column,
                    invalid_values=bad[:5],
                    reason="nonfinite_numeric",
                )
