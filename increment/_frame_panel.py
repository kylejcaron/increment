"""Panel and windowing internals for frame sources."""

from __future__ import annotations

import datetime as dt
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import narwhals as nw

from increment._frame_validation import (
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
    raiser,
    refusals,
    refuse,
    warn,
)
from increment.sources import CENSOR_WARN_FRACTION


def _prepare_panel(
    frame: nw.DataFrame[Any],
    *,
    identity: nw.DataFrame[Any],
    unit: str,
    date: str,
    value_columns: Sequence[str],
) -> tuple[nw.DataFrame[Any], dict[str, int], int]:
    """Canonical sparse observations and analytical zero-filled cell counts."""
    observed = frame.select(
        nw.col(unit).alias("unit_id"),
        nw.col(date).alias("ds"),
        *[nw.col(c).cast(nw.Float64).alias(c) for c in value_columns],
    )
    n_filled = identity.shape[0] * observed.get_column("ds").n_unique() - observed.shape[0]
    if value_columns:
        any_null = _is_missing(observed, value_columns[0])
        for column in value_columns[1:]:
            any_null = any_null | _is_missing(observed, column)
        n_filled += int(observed.select(any_null.cast(nw.Int64).sum()).item())
        observed = observed.with_columns(*[nw.col(c).fill_null(0.0) for c in value_columns])
    return (
        observed.join(identity, on="unit_id", how="left"),
        _group_counts(identity, "group_id"),
        n_filled,
    )


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
        values = [
            key[0] if isinstance(key[0], int) else dt.date.fromisoformat(key[0]).toordinal()
            for key in order.values()
        ]
        return nw.col("ds").replace_strict(list(order), values) - nw.col(anchor).replace_strict(
            list(order), values
        )
    return _day_axis_elapsed(
        nw.col("ds"), nw.col(anchor), numeric=_day_axis_is_numeric(panel.schema["ds"])
    )


def _day_axis_elapsed(ds: nw.Expr, anchor: nw.Expr, *, numeric: bool) -> nw.Expr:
    """Whole days from *anchor* to *ds* on a day axis of the given kind.

    An explicit numeric day axis subtracts directly - casting it through
    ``Datetime('us')`` first would silently reinterpret whole-day units
    as whole-microsecond units. A calendar axis casts both sides through
    ``Datetime('us')`` (pandas stores a raw ``datetime.date`` column as
    opaque ``object`` dtype, invisible to narwhals as Date/Datetime on
    its own) and divides the resulting duration by 86400 seconds.
    """
    if numeric:
        return ds - anchor
    return (
        ds.cast(nw.Datetime("us")) - anchor.cast(nw.Datetime("us"))
    ).dt.total_seconds() / 86400.0


_ISO_DATE_LABEL = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PREFIXED_INTEGER_LABEL = re.compile(r"^(?P<prefix>[^+\-0-9]*)(?P<sign>[+-]?)(?P<digits>\d+)$")

_DAY_AXIS_LABEL_UNORDERABLE = RefusalSpec(
    "frame.asof.day_axis_unorderable",
    CapabilityError,
    template="cannot order day-axis label(s) {labels!r} chronologically -- string day-axis labels are ordered only when every distinct label is an unambiguous ISO-8601 date ('YYYY-MM-DD', lexicographic order is chronological) or every label shares one consistent '<prefix><signed integer>' shape (e.g. 'd1', 'd-1'), with no two distinct labels parsing to the same integer. Declare an explicit date or numeric day axis instead of a free-form string label.",
)

Renderer = Callable[..., str]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "frame.panel.observable_end_dtype": "observable_end {observable_end!r} (dtype {end_dtype}, {end_kind} kind) does not match the {axis_kind} day axis (exposure anchor dtype {anchor_dtype}). Pass an observable_end of the same kind as the panel's date column.",
    },
)
_raise = raiser(_REFUSALS)


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


def _is_iso_date_label(value: str) -> bool:
    if not _ISO_DATE_LABEL.match(value):
        return False
    try:
        dt.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _day_axis_label_order(labels: Iterable[Any]) -> dict[Any, tuple[Any, ...]]:
    """A chronologically-justified rank for every distinct day-axis label.

    This is a COLLECTION-level decision - it depends on the full set of
    distinct labels, not any one value in isolation - because the only
    orders this accepts are ones the ENTIRE set jointly justifies. Guessing
    at anything else (raw string order misreads ``"01/01/2026"`` as before
    ``"12/31/2025"``; naive natural-sort digit-chunking misreads ``"d0"``
    as after ``"d-1"``) is exactly the class of bug this function exists
    to prevent, so it refuses instead of guessing.

    Date/Datetime/numeric labels already compare correctly on their own.
    A string label orders correctly only when the whole set is either:

    - every label an unambiguous ISO-8601 date (``"YYYY-MM-DD"``, where
      lexicographic order IS chronological), or
    - every label one consistent ``<prefix><signed integer>`` shape (e.g.
      ``"d1"``, ``"d-1"``, ``"day_10"``), ranked by the parsed integer -
      REFUSED if two distinct labels parse to the same integer (``"d1"``
      and ``"d01"``), since their cumulative order would then silently
      depend on backend/input row order.

    Any other shape refuses, naming the offending label(s).
    """
    # Reject bytes-like labels before deduplication: they cannot justify a
    # chronological order, and bytearray/memoryview values would otherwise
    # fail with an unhelpful hashing error.
    materialized = list(labels)
    if any(isinstance(v, bytes | bytearray | memoryview) for v in materialized):
        refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=tuple(sorted(map(str, materialized))))
    distinct = list(dict.fromkeys(materialized))
    strings = [v for v in distinct if isinstance(v, str)]
    if not strings:
        return {v: (v,) for v in distinct}
    if len(strings) != len(distinct):
        refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=tuple(sorted(map(str, distinct))))

    if all(_is_iso_date_label(v) for v in strings):
        return {v: (v,) for v in distinct}

    matches = {v: _PREFIXED_INTEGER_LABEL.match(v) for v in strings}
    if all(matches.values()):
        prefixes = {m.group("prefix") for m in matches.values() if m is not None}
        if len(prefixes) == 1:
            parsed = {
                v: (-1 if m.group("sign") == "-" else 1) * int(m.group("digits"))
                for v, m in matches.items()
                if m is not None
            }
            by_value: dict[int, list[str]] = {}
            for v, n in parsed.items():
                by_value.setdefault(n, []).append(v)
            collisions = [vs for vs in by_value.values() if len(vs) > 1]
            if not collisions:
                return {v: (parsed[v],) for v in distinct}
            offending = tuple(sorted(v for vs in collisions for v in vs))
            refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=offending)

    refuse(_DAY_AXIS_LABEL_UNORDERABLE, labels=tuple(sorted(strings)))


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
) -> nw.DataFrame[Any]:
    """Preserve observable zero-outcome units while excluding pre-exposure values."""
    if exposure is None or panel.is_empty():
        return panel
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

    *observable_end* is validated against the exposure anchor's axis kind
    first - a numeric axis combined with the publicly documented
    ``datetime.date`` default otherwise reaches unsupported
    date-minus-number backend arithmetic.
    """
    if isinstance(exposure.schema["__exposure__"], nw.String):
        labeled = exposure.with_columns(nw.lit(observable_end).alias("ds"))
        return labeled.with_columns(
            _uptake_day_elapsed(labeled, "__exposure__").alias("__observable_days__")
        ).drop("ds")
    anchor_dtype = exposure.schema["__exposure__"]
    numeric = _day_axis_is_numeric(anchor_dtype)
    end_dtype = exposure.select(nw.lit(observable_end).alias("__end__")).schema["__end__"]
    if _day_axis_is_numeric(end_dtype) != numeric:
        axis_kind = "numeric" if numeric else "calendar"
        end_kind = "numeric" if _day_axis_is_numeric(end_dtype) else "calendar"
        _raise(
            "frame.panel.observable_end_dtype",
            observable_end=observable_end,
            end_dtype=end_dtype,
            end_kind=end_kind,
            axis_kind=axis_kind,
            anchor_dtype=anchor_dtype,
        )
    elapsed = _day_axis_elapsed(nw.lit(observable_end), nw.col("__exposure__"), numeric=numeric)
    return exposure.with_columns(elapsed.alias("__observable_days__"))


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
