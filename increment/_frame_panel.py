"""Panel and windowing internals for frame sources."""

from __future__ import annotations

import datetime as dt
import math
import numbers
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import narwhals as nw
import numpy as np

from increment._day_axis_labels import _DAY_AXIS_LABEL_UNORDERABLE, _day_axis_label_order
from increment._frame_validation import (
    _SIGNED_DAY_MAX,
    CENSORING_DROPPED_UNITS,
    _day_axis_is_numeric,
    _group_counts,
    _is_missing,
)
from increment._window import final_maturity_day as _final_maturity_day  # noqa: F401
from increment._window import resolve_window_days as _resolve_window_days  # noqa: F401
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    WarningSpec,
    _safe_error_value,
    refuse,
    warn,
)
from increment.sequential_state import sequential_refuse
from increment.sources import CENSOR_WARN_FRACTION


def _prepare_panel(
    frame: nw.DataFrame[Any],
    *,
    identity: nw.DataFrame[Any],
    unit: str,
    date: str,
    value_columns: Sequence[str],
    zero_fill: bool = True,
    exposure: nw.DataFrame[Any] | None = None,
    observation_end: Any = None,
) -> tuple[nw.DataFrame[Any], dict[str, int], int]:
    """Build sparse unit-day rows, zero-fill explicit nulls and count the cells filled.

    Values dated after *observation_end* become zero; their rows stay, so a unit
    observed only later still counts as a unit with no outcome. The count covers each
    unit's eligible days: from its *exposure* (else the first panel day) through
    *observation_end* (else the last panel day). Exposures cannot be placed on day labels
    without a chronological order, so such a panel counts every day.
    """
    observed = frame.select(
        nw.col(unit).alias("unit_id"),
        nw.col(date).alias("ds"),
        *[nw.col(c).cast(nw.Float64).alias(c) for c in value_columns],
    )
    labels = observed.get_column("ds").unique().to_list()
    days = labels if observation_end is None else _labels_through(labels, observation_end)
    window = observed if observation_end is None else _rows_on_days(observed, days)
    cells = identity.shape[0] * len(days)
    if exposure is not None and _exposures_placeable(observed, exposure):
        cells = _eligible_cells(window.select(nw.col("ds").alias("__exposure__")), exposure)
        indexed, day_index = _with_day_index(window, exposure)
        window = indexed.filter(nw.col(day_index) >= 0)
    n_filled = cells - window.shape[0]
    if value_columns:
        any_null = _is_missing(window, value_columns[0])
        for column in value_columns[1:]:
            any_null = any_null | _is_missing(window, column)
        n_filled += int(window.select(any_null.cast(nw.Int64).sum()).item())
        if zero_fill:
            observed = observed.with_columns(*[nw.col(c).fill_null(0.0) for c in value_columns])
        if len(days) < len(labels):
            observable = nw.col("ds").is_in(days)
            observed = observed.with_columns(
                nw.when(observable).then(nw.col(c)).otherwise(0.0).alias(c) for c in value_columns
            )
    return (
        observed.join(identity, on="unit_id", how="left"),
        _group_counts(identity, "group_id"),
        n_filled,
    )


def _rows_on_days(panel: nw.DataFrame[Any], days: Sequence[Any]) -> nw.DataFrame[Any]:
    """The rows of *panel* dated on one of *days*. An empty list of days carries no type to
    compare with the day column, so it selects no row without a comparison."""
    return panel.filter(nw.col("ds").is_in(days)) if days else panel.head(0)


def _exposures_placeable(panel: nw.DataFrame[Any], exposure: nw.DataFrame[Any]) -> bool:
    """Whether every exposure date has a place on the panel's day axis."""
    labels = panel.get_column("ds").unique().to_list()
    try:
        _day_axis_label_order([*labels, *exposure.get_column("__exposure__").unique().to_list()])
    except CapabilityError as exc:
        if exc.code != _DAY_AXIS_LABEL_UNORDERABLE.code:
            raise
        return False
    return True


def _eligible_cells(days: nw.DataFrame[Any], exposure: nw.DataFrame[Any]) -> int:
    """Count eligible cells by exact axis coordinates, never by elapsed differences."""
    days = days.unique()
    if days.is_empty():
        return 0
    starts = exposure.group_by("__exposure__").agg(nw.len().alias("__units__"))
    day_column = days.get_column("__exposure__")
    start_column = starts.get_column("__exposure__")
    labels = isinstance(day_column.dtype, nw.String)
    if not labels and not _day_axis_is_numeric(day_column.dtype):
        day_column = day_column.cast(nw.Datetime("us")).cast(nw.Int64)
        start_column = start_column.cast(nw.Datetime("us")).cast(nw.Int64)
    day_values, start_values = day_column.to_list(), start_column.to_list()
    if labels:
        order = _day_axis_label_order([*day_values, *start_values])
        day_values = [order[value][0] for value in day_values]
        start_values = [order[value][0] for value in start_values]
    coordinates = np.sort(np.asarray(day_values))
    eligible = len(coordinates) - np.searchsorted(coordinates, start_values, side="left")
    return int((eligible * starts.get_column("__units__").to_numpy()).sum())


def _labels_through(labels: Sequence[Any], end: Any) -> list[Any]:
    """The day labels on or before *end*, in their input order."""
    order = _day_axis_label_order([*labels, end])
    return [label for label in labels if order[label] <= order[end]]


_SIGNED_DAY_RANGE = f"the signed 64-bit range ({-_SIGNED_DAY_MAX - 1} to {_SIGNED_DAY_MAX})"

_OBSERVATION_END_ISSUES = {
    "unparseable": "is not an ISO-8601 date",
    "wrong_kind": "is not a value of the date column's kind",
    "out_of_range": f"is outside {_SIGNED_DAY_RANGE}",
    "before_first_day": "is before the panel's first day",
    "unknown_label": "cannot be ordered among the date column's labels",
}

_OBSERVATION_END = RefusalSpec(
    "source.panel.observation_end",
    InvalidRequestError,
    lambda *, value, axis_kind, issue: (
        f"observation_end={value!r} {_OBSERVATION_END_ISSUES[issue]} (a {axis_kind} day "
        "axis). Pass the last observed day: a date, naive datetime or ISO-8601 string for "
        "a calendar date column, a number for a numeric day index, or a label spelled like "
        "the column's own for a string day column."
    ),
)

_PANEL_AS_OF = RefusalSpec(
    "source.panel.as_of",
    InvalidRequestError,
    lambda *, value, axis_kind, issue: (
        f"as_of={value!r} {_OBSERVATION_END_ISSUES[issue]} (a {axis_kind} day axis). "
        "Pass a day in the date column's form."
    ),
)


@dataclass(frozen=True, slots=True)
class _PanelDay:
    value: Any
    axis_kind: Literal["numeric", "calendar", "label"]
    issue: str | None = None


def _resolve_empty_panel_axis(
    frame: nw.DataFrame[Any], *, date: str, exposure_date: str | None, observation_end: Any
) -> nw.DataFrame[Any]:
    """Resolve only untyped empty pandas object days from explicit calendar information."""
    if not frame.is_empty() or frame.implementation.value != "pandas":
        return frame
    native = frame.to_native()
    ambiguous = [
        name for name in (date, exposure_date) if name is not None and native[name].dtype == object
    ]
    if not ambiguous:
        return frame
    calendar = any(
        name is not None and isinstance(frame.schema[name], nw.Date | nw.Datetime)
        for name in (date, exposure_date)
    )
    cutoff, issue = _calendar_day(observation_end, datetime_axis=False)
    if not calendar and (issue is not None or cutoff is None):
        return frame
    return frame.with_columns(nw.col(name).cast(nw.Datetime("us")) for name in ambiguous)


def _beyond_day_range(dtype: nw.dtypes.DType, value: Any) -> bool:
    """Whether *value* is a number outside the signed 64-bit range a numeric day axis
    counts days in, which the backends' day arithmetic cannot hold.

    The range is checked against -2**63 and 2**63, which every binary float type holds
    exactly; comparing a NumPy float with 2**63 - 1 rounds that limit up to 2**63.
    """
    limit = _SIGNED_DAY_MAX + 1
    return (
        _day_axis_is_numeric(dtype)
        and isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and (value < -limit or value >= limit)
    )


def _classify_panel_day(
    frame: nw.DataFrame[Any], *, date: str, value: Any, construction: bool
) -> _PanelDay:
    """Classify a cutoff once using the source's validated day axis."""
    dtype = frame.schema[date]
    labels = frame.get_column(date).unique().to_list()
    issue = None
    normalized = value
    if isinstance(dtype, nw.String):
        axis_kind = "label"
        if not isinstance(value, str):
            issue = "wrong_kind"
        else:
            # An axis that cannot be ordered at all refuses with its own code.
            _day_axis_label_order(labels)
            try:
                order = _day_axis_label_order([*labels, value])
            except CapabilityError as exc:
                if exc.code != _DAY_AXIS_LABEL_UNORDERABLE.code:
                    raise
                issue = "unknown_label"
            else:
                if construction and labels and order[value] < min(order[label] for label in labels):
                    issue = "before_first_day"
    else:
        if _day_axis_is_numeric(dtype):
            axis_kind = "numeric"
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                issue = "wrong_kind"
            elif not -math.inf < value < math.inf:
                # Compared rather than converted: an integer too large for a float is finite.
                issue = "unparseable"
            elif _beyond_day_range(dtype, value):
                issue = "out_of_range"
        else:
            axis_kind = "calendar"
            normalized, issue = _calendar_day(value, datetime_axis=_datetime_axis(dtype, labels))
        if construction and issue is None and labels and normalized < min(labels):
            issue = "before_first_day"

    return _PanelDay(normalized, axis_kind, issue)


def _normalize_panel_day(
    frame: nw.DataFrame[Any],
    *,
    date: str,
    value: Any,
    purpose: Literal["observation_end", "capture", "compliance"],
) -> Any:
    """Normalize a cutoff at the source boundary, retaining each operation's refusal."""
    if value is None:
        return None
    day = _classify_panel_day(
        frame, date=date, value=value, construction=purpose == "observation_end"
    )
    if day.issue is None:
        return day.value
    if purpose == "capture":
        if _beyond_day_range(frame.schema[date], value):
            sequential_refuse(
                "source.invalid", f"as_of is outside {_SIGNED_DAY_RANGE} of the numeric day column"
            )
        if day.axis_kind == "label" and day.issue == "unknown_label":
            _day_axis_label_order([*frame.get_column(date).unique().to_list(), value])
        sequential_refuse(
            "source.invalid",
            f"as_of={_safe_error_value(value)!r} {_OBSERVATION_END_ISSUES[day.issue]} "
            f"(a {day.axis_kind} day axis)",
        )
    safe_value = _safe_error_value(value)
    if (
        purpose == "compliance"
        and isinstance(value, numbers.Real)
        and not -math.inf < value < math.inf
    ):
        safe_value = str(value)
    if purpose == "observation_end":
        refuse(_OBSERVATION_END, value=safe_value, axis_kind=day.axis_kind, issue=day.issue)
    refuse(_PANEL_AS_OF, value=safe_value, axis_kind=day.axis_kind, issue=day.issue)


def _normalize_observation_end(frame: nw.DataFrame[Any], *, date: str, value: Any) -> Any:
    """Normalize the construction cutoff without changing its refusal contract."""
    return _normalize_panel_day(frame, date=date, value=value, purpose="observation_end")


def _datetime_axis(dtype: nw.dtypes.DType, labels: Sequence[Any]) -> bool:
    """Whether a calendar axis holds datetimes rather than dates."""
    if isinstance(dtype, nw.Datetime):
        return True
    if isinstance(dtype, nw.Date):
        return False
    return any(isinstance(label, dt.datetime) for label in labels)


def _calendar_day(
    value: Any, *, datetime_axis: bool
) -> tuple[Any, Literal["unparseable", "wrong_kind"] | None]:
    """The calendar day *value* names, typed like the axis, or why it names none."""
    if isinstance(value, str):
        try:
            value = (
                dt.datetime.fromisoformat(value)
                if "T" in value or " " in value
                else dt.date.fromisoformat(value)
            )
        except ValueError:
            return None, "unparseable"
    if not isinstance(value, dt.date):
        return None, "wrong_kind"
    if isinstance(value, dt.datetime):
        if value.tzinfo is not None:
            return None, "wrong_kind"
        value = value.date()
    return (dt.datetime.combine(value, dt.time()) if datetime_axis else value), None


def _partition_panel_days(panel: nw.DataFrame[Any]) -> dict[Any, nw.DataFrame[Any]]:
    """Partition rows once by first-seen day label for reuse by reductions."""
    labels = panel.get_column("ds").unique().to_list()
    index_name = _scratch_name(panel, "__day_partition__")
    ordered = panel.with_columns(
        nw.col("ds").replace_strict(labels, list(range(len(labels)))).alias(index_name)
    ).sort(index_name)
    indices = ordered.get_column(index_name).to_numpy()
    day_slices: dict[Any, nw.DataFrame[Any]] = {}
    start = 0
    while start < len(indices):
        day_index = int(indices[start])
        stop = start + 1
        while stop < len(indices) and indices[stop] == day_index:
            stop += 1
        day_slices[labels[day_index]] = ordered[start:stop].drop(index_name)
        start = stop
    return day_slices


def _day_population(
    panel: nw.DataFrame[Any],
    *,
    identity: nw.DataFrame[Any],
    ds: Any,
    value_columns: Sequence[str],
    ordinal: str,
    day_slices: Mapping[Any, nw.DataFrame[Any]] | None = None,
) -> nw.DataFrame[Any]:
    """Construct one logical zero-filled day without retaining the date spine."""
    observed_day = day_slices[ds] if day_slices is not None else panel.filter(nw.col("ds") == ds)
    observed = observed_day.select("unit_id", *value_columns)
    population = identity.with_columns(nw.lit(ds).alias("ds")).join(
        observed, on="unit_id", how="left"
    )
    return population.with_columns(
        *[nw.col(column).fill_null(0.0).alias(column) for column in value_columns]
    ).sort(ordinal)


def _day_labels(panel: nw.DataFrame[Any]) -> list[Any]:
    """Return the observed date axis in its justified chronological order."""
    labels = panel.get_column("ds").unique().to_list()
    order = _day_axis_label_order(labels)
    return sorted(labels, key=order.__getitem__)


def _densify_panel(
    panel: nw.DataFrame[Any],
    *,
    identity: nw.DataFrame[Any],
    value_columns: Sequence[str],
) -> nw.DataFrame[Any]:
    """Materialize a day spine for the byte-bounded small-panel kernel only."""
    dates = panel.select("ds").unique()
    observed = panel.select("unit_id", "ds", *value_columns)
    dense = identity.join(dates, how="cross").join(observed, on=["unit_id", "ds"], how="left")
    return dense.with_columns(*[nw.col(column).fill_null(0.0) for column in value_columns])


def _uptake_day_elapsed(panel: nw.DataFrame[Any], anchor: str) -> nw.Expr:
    """Elapsed uptake days, including canonical structured string labels."""
    if isinstance(panel.schema["ds"], nw.String):
        labels = panel.get_column("ds").unique().to_list()
        order = _day_axis_label_order([*labels, *panel.get_column(anchor).unique().to_list()])
        if not order:
            # Only a frame with no rows has no label to map; its elapsed days need only a type.
            return nw.lit(0, dtype=nw.Int64)
        values = [
            key[0] if isinstance(key[0], int) else dt.date.fromisoformat(key[0]).toordinal()
            for key in order.values()
        ]
        ds = nw.col("ds").replace_strict(list(order), values)
        anchor_day = nw.col(anchor).replace_strict(list(order), values)
        return _day_axis_elapsed(ds, anchor_day, numeric=True, integer=True)
    return _day_axis_elapsed(
        _signed_day(nw.col("ds"), panel.schema["ds"]),
        _signed_day(nw.col(anchor), panel.schema[anchor]),
        numeric=_day_axis_is_numeric(panel.schema["ds"]),
        integer=panel.schema["ds"].is_integer() and panel.schema[anchor].is_integer(),
    )


def _signed_day(day: nw.Expr, dtype: nw.dtypes.DType) -> nw.Expr:
    """Widen integer days before arithmetic, preserving negative offsets on unsigned axes."""
    return day.cast(nw.Int64) if dtype.is_integer() else day


def _day_axis_elapsed(
    ds: nw.Expr, anchor: nw.Expr, *, numeric: bool, integer: bool = False
) -> nw.Expr:
    """Whole days from *anchor* to *ds* on a day axis of the given kind.

    An explicit numeric day axis subtracts directly - casting it through
    ``Datetime('us')`` first would silently reinterpret whole-day units
    as whole-microsecond units. A calendar axis casts both sides through
    ``Datetime('us')`` (pandas stores a raw ``datetime.date`` column as
    opaque ``object`` dtype, invisible to narwhals as Date/Datetime on
    its own) and divides the resulting duration by 86400 seconds.

    Integer offsets saturate only outside int64, for private bounded predicates.
    Masked operands prevent wrapping even when a backend evaluates both branches.
    """
    if integer:
        minimum = -_SIGNED_DAY_MAX - 1
        negative_anchor = nw.when(anchor < 0).then(anchor).otherwise(0)
        positive_anchor = nw.when(anchor > 0).then(anchor).otherwise(0)
        above = (anchor < 0) & (ds > _SIGNED_DAY_MAX + negative_anchor)
        below = (anchor > 0) & (ds < minimum + positive_anchor)
        overflow = above | below
        safe_ds = nw.when(overflow).then(0).otherwise(ds)
        safe_anchor = nw.when(overflow).then(0).otherwise(anchor)
        return (
            nw.when(above)
            .then(nw.lit(_SIGNED_DAY_MAX, dtype=nw.Int64))
            .when(below)
            .then(nw.lit(minimum, dtype=nw.Int64))
            .otherwise(safe_ds - safe_anchor)
        )
    if numeric:
        return ds - anchor
    return (
        ds.cast(nw.Datetime("us")) - anchor.cast(nw.Datetime("us"))
    ).dt.total_seconds() / 86400.0


Renderer = Callable[..., str]


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Renderer
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
    warn(
        _WARNINGS[code],
        stacklevel=stacklevel + 1,
        skip_file_prefixes=skip_file_prefixes,
        context=context,
    )


_WARNINGS["frame.censoring.dropped_units"] = CENSORING_DROPPED_UNITS


def _scratch_name(frame: nw.DataFrame[Any], base: str) -> str:
    columns = frame.columns
    name = base
    while name in columns:
        name = f"_{name}"
    return name


_PANEL_DECLARATION_COLLISION = RefusalSpec(
    "frame.panel.declaration_canonical_collision",
    InvalidRequestError,
    template=(
        "from_unit_panel declaration {role!r} uses canonical column name "
        "{canonical_name!r} as {source_name!r}; rename that declaration column "
        "before constructing the panel. {route}"
    ),
)


def _validate_panel_declarations(declarations: Iterable[tuple[str, str]]) -> None:
    """Reject value declarations that duplicate projected panel identity columns."""
    for role, source_name in declarations:
        if source_name in {"unit_id", "group_id", "ds"}:
            refuse(
                _PANEL_DECLARATION_COLLISION,
                role=role,
                canonical_name=source_name,
                source_name=source_name,
                route=(
                    "Rename the declared metric value, denominator, or uptake column; "
                    "canonical role columns and separately resolved covariates remain valid."
                ),
            )


def _with_day_index(
    panel: nw.DataFrame[Any], exposure: nw.DataFrame[Any]
) -> tuple[nw.DataFrame[Any], str]:
    """Join the per-unit exposure anchor and attach a collision-free day index."""
    anchor = _scratch_name(panel, "__exposure__")
    index = _scratch_name(panel, "__day_idx__")
    joined = panel.join(
        exposure.select("unit_id", nw.col("__exposure__").alias(anchor)),
        on="unit_id",
        how="left",
    )
    elapsed = _uptake_day_elapsed(joined, anchor)
    return joined.with_columns(elapsed.alias(index)), index


def _admit_panel_totals(
    panel: nw.DataFrame[Any],
    exposure: nw.DataFrame[Any] | None,
    *,
    value_columns: Sequence[str],
    observation_end: Any = None,
) -> nw.DataFrame[Any]:
    """Preserve observable zero-outcome units while excluding pre-exposure values.

    A unit is observable once exposed by *observation_end*, else by the last panel day.
    """
    if exposure is None or panel.is_empty():
        return panel
    last_day = observation_end
    if last_day is None:
        labels = panel.get_column("ds").unique().to_list()
        order = _day_axis_label_order(labels)
        last_day = max(labels, key=order.__getitem__)
    observable = _observable_end_index(exposure, last_day)
    units = observable.filter(nw.col("__observable_days__") >= 0).select("unit_id")
    admitted = panel.join(units, on="unit_id", how="semi")
    indexed, day_index = _with_day_index(admitted.select("unit_id", "ds"), exposure)
    mask_name = _scratch_name(admitted, "__post_exposure__")
    mask = indexed.select("unit_id", "ds", (nw.col(day_index) >= 0).alias(mask_name))
    return (
        admitted.join(mask, on=["unit_id", "ds"], how="left")
        .with_columns(
            nw.when(nw.col(mask_name)).then(nw.col(column)).otherwise(0.0).alias(column)
            for column in value_columns
        )
        .drop(mask_name)
    )


def _observable_end_index(exposure: nw.DataFrame[Any], observable_end: Any) -> nw.DataFrame[Any]:
    """Per-unit whole days from the unit's own exposure to *observable_end*
    - a single date shared by every unit computing THIS metric: an
    explicit ``observation_end``, or (for a running experiment) that
    metric's own fact-scoped observed extent (see
    :func:`_fact_max_observed_dates`) - never the whole panel's.

    *observable_end* is a day of the panel's own axis: a panel label, or an
    ``observation_end`` the builder normalized to that axis.
    """
    if isinstance(exposure.schema["__exposure__"], nw.String):
        labeled = exposure.with_columns(nw.lit(observable_end).alias("ds"))
        return labeled.with_columns(
            _uptake_day_elapsed(labeled, "__exposure__").alias("__observable_days__")
        ).drop("ds")
    dtype = exposure.schema["__exposure__"]
    elapsed = _day_axis_elapsed(
        nw.lit(observable_end),
        _signed_day(nw.col("__exposure__"), dtype),
        numeric=_day_axis_is_numeric(dtype),
        integer=dtype.is_integer() and isinstance(observable_end, numbers.Integral),
    )
    return exposure.with_columns(elapsed.alias("__observable_days__"))


def _exposed_through(
    frame: nw.DataFrame[Any], *, unit: str, exposure_date: str, observation_end: Any
) -> tuple[nw.DataFrame[Any], int]:
    """The rows of units exposed on or before *observation_end*, and how many units left."""
    exposure = frame.group_by(nw.col(unit).alias("unit_id")).agg(
        nw.col(exposure_date).alias("__exposure__").min()
    )
    enrolled = (
        _observable_end_index(exposure, observation_end)
        .filter(nw.col("__observable_days__") >= 0)
        .select(nw.col("unit_id").alias(unit))
    )
    excluded = exposure.shape[0] - enrolled.shape[0]
    if not excluded:
        return frame, 0
    return frame.join(enrolled, on=unit, how="semi"), excluded


def _censor_units(
    indexed: nw.DataFrame[Any],
    *,
    final_maturity_day: int,
    observable_end_idx: nw.DataFrame[Any],
) -> tuple[nw.DataFrame[Any], int, int]:
    """Drop whole units whose last required day exceeds the observable end.

    Mirrors ``unit_totals``'s late-enrollee censoring block exactly: a
    unit is kept only once the LAST calendar day its outcome depends on
    (``final_maturity_day`` days after its own exposure - see
    :func:`_final_maturity_day`) is itself observable; otherwise it is
    dropped entirely - not zeroed - so an immature unit never reads as
    "observed, did not return". Returns ``(kept, enrolled, dropped)``.
    """
    observable_column = _scratch_name(indexed, "__observable_days__")
    observable = observable_end_idx.select(
        "unit_id", nw.col("__observable_days__").alias(observable_column)
    )
    scoped = indexed.join(observable, on="unit_id", how="left")
    enrolled = int(scoped["unit_id"].n_unique())
    kept = scoped.filter(nw.col(observable_column) >= final_maturity_day)
    return kept, enrolled, enrolled - int(kept["unit_id"].n_unique())


def _censoring_warning(
    metric_name: str,
    *,
    enrolled: int,
    dropped: int,
    observation_end: dt.date | dt.datetime | str | int | float | None,
) -> None:
    """Warn when censoring drops a material share of enrolled units.

    Same >``CENSOR_WARN_FRACTION`` threshold and two-cause taxonomy as
    ``unit_totals``'s own warning (this path has no ``data_as_of``
    concept, so only two causes apply here, not three): a declared
    ``observation_end`` that closed too early, or no declared end at all
    (a running experiment - these units simply have not matured yet).

    Attributed to the first stack frame outside this package rather than
    a hardcoded ``stacklevel`` - a fixed frame count silently
    mis-attributes the warning's source line the moment the internal
    call chain gains or loses a frame.
    """
    if not enrolled or dropped / enrolled <= CENSOR_WARN_FRACTION:
        return
    cause = (
        "the declared observation_end -- extend it if this intervention's "
        "effect persists past the experiment; if it reverts, this censoring "
        "is correct"
        if observation_end is not None
        else "no observation_end declared -- these units have not had time "
        "to mature within the observed dates; this resolves as more data "
        "arrives, not by changing any argument"
    )
    _warn(
        "frame.censoring.dropped_units",
        metric_name=metric_name,
        dropped=dropped,
        enrolled=enrolled,
        cause=cause,
        skip_file_prefixes=(os.path.dirname(os.path.abspath(__file__)),),
    )


def _fact_max_observed_dates(
    raw: nw.DataFrame[Any], *, date: str, columns: Iterable[str]
) -> dict[str, Any]:
    """Per-column latest date with a genuinely OBSERVED (non-null) value,
    read from *raw* - the one-row-per-(unit, date) input :func:`from_unit_panel`
    was actually given, before :func:`_densify_panel` zero-fills every unit
    at every globally-observed date.

    Zero-filling is exactly why this cannot be computed from the densified
    panel afterwards: a padded cell and a real, present zero are both
    ``0.0`` there, so the densified panel has no per-column "last observed"
    concept at all - every column's max ``ds`` collapses to the same
    panel-wide date regardless of which column actually had a row on it.
    On the raw input, a column absent for a given (unit, date) is ``null``
    (the row exists for whichever OTHER metric was observed that day; this
    one was not), so ``~is_null()`` recovers exactly this metric's own
    fact stream. Mirrors ``unit_day_spine_stats``'s ``events.ts.max()``
    fallback in builders.py, where ``events`` is already scoped to one
    metric's own event stream.

    A *columns* entry with zero non-null rows across the whole input is
    omitted from the result - callers fall back further (to the panel's
    own extent) rather than crash on a metric with literally no data.
    """
    out: dict[str, Any] = {}
    order = _day_axis_label_order(raw.get_column(date).unique().to_list())
    for col in columns:
        observed = raw.filter(~nw.col(col).is_null())
        if observed.shape[0]:
            labels = observed.get_column(date).unique().to_list()
            out[col] = max(labels, key=order.__getitem__)
    return out
