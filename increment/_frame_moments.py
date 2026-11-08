"""Moment-construction internals for frame sources."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, cast

import narwhals as nw
import numpy as np

from increment._frame_panel import (
    _admit_panel_totals,
    _censor_units,
    _censoring_warning,
    _day_axis_label_order,
    _observable_end_index,
    _partition_panel_days,
    _scratch_name,
    _uptake_day_elapsed,
    _with_day_index,
)
from increment._frame_validation import (
    COVARIATE_COLUMN_MISSING,
    COVARIATE_IMPUTE_ALL_NULL,
    SCALAR_COVARIATE_REFUSALS,
)
from increment._moment_plan import (
    CLUSTER_SIZE_GRAIN,
    CLUSTER_UPTAKE_GRAIN,
    COMPLIANCE_CLUSTER,
    DAY_GRAIN,
    SLOTS,
    UNIT_GRAIN,
    X_SLOT_ROLES,
    MomentPlan,
)
from increment._window import final_maturity_day as _final_maturity_day
from increment._window import resolve_window_days as _resolve_window_days
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    raiser,
    refusals,
    refuse,
)
from increment.semantics.models import Metric, QuantileMetric, RetentionMetric
from increment.sources import WINSORIZATION_MOMENT_FIELDS

_ASOF_QUANTILE_UNSUPPORTED = RefusalSpec(
    "frame.asof.quantile_unsupported",
    CapabilityError,
    template="metric {metric!r} is a quantile metric -- quantiles do not decompose into as-of moments",
)

_WINSOR_COLLAPSED_BOUND = RefusalSpec(
    "frame.winsorization.collapsed_bound",
    CapabilityError,
    template="metric {metric!r}: percentile winsorization resolved a collapsed bound (lower={lower!r}, upper={upper!r}) on {date!r} while still clipping {clipped} value(s) -- pooling more data per date or using a fixed-value cap avoids this.",
)


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "frame.moments.asof_moments_completed": "asof moments: completed_windows_only=True is contradictory for retention metric {metric!r} with unbounded band {band!r}",
        "frame.moments.asof_completed_requires_uptake_window": "asof moments: completed_windows_only=True under an encouragement design requires bounded outcome and uptake windows",
        "frame.moments.metric_type_conversion": "metric {metric!r}: type='conversion' collapsed to a per-unit value of {value:g} (> 1) -- summing per-day rows yields a day count, not a 0/1 conversion flag. If you want the number of converting days, declare a mean metric with a count aggregation instead; for any-occurrence conversion, set window_days on this metric or pre-reduce the column to one 0/1 row per unit.",
    },
)
_raise = raiser(_REFUSALS)

# A data-dependent refusal is retained until its metric is requested.
_DeferredRefusal = tuple[RefusalSpec, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class _CovariateReduction:
    """Method-scoped covariate preparation and retained reduction failures."""

    tables: Mapping[str, nw.DataFrame[Any]] | None = None
    columns: Mapping[str, str] | None = None
    means: Mapping[str, float | None] | None = None
    failures: dict[str, _DeferredRefusal] | None = None
    cell_failures: dict[tuple[str, str, str], _DeferredRefusal] | None = None


if TYPE_CHECKING:
    from increment.frame import MetricSpec


# A finite float is an integer times a power of two: integer parts summed on a shared
# scale give one total in any order, partitioning or dataframe library, rounded once.

#: Bounded row scratch; below the 2**26 exact-bin ceiling for 27/26-bit mantissa halves.
_SUM_ROWS = 1 << 16
_EXPONENTS = 2048
#: Biased exponents from which 2**26 binned values could overflow a float64 bin sum.
_BIN_OVERFLOW = 2046 - 26
_LOW_HALF = np.uint64((1 << 26) - 1)
_FRACTION = np.uint64((1 << 52) - 1)
_LIMB = np.uint64(32)
_LIMB_MASK = np.uint64(0xFFFFFFFF)
#: Limbs held per group: values up to 2**96 apart. A group whose values span more (such
#: as 1e300 beside 1e-300) sums by ``exact_sum``, so memory stays at 8 limbs per group.
_LIMB_WIDTH = 8


def exact_sum(values: Iterable[float]) -> float:
    """The sum of *values* rounded once to the nearest float, in any order.

    ``math.fsum`` is exact unless a partial sum overflows; the rational fallback then
    rounds the exact total, to ±inf beyond the largest float. Non-finite values follow
    IEEE: nan if any is nan or both infinities occur, otherwise that infinity.
    """
    values = [float(value) for value in values]
    special = [value for value in values if not math.isfinite(value)]
    if special:
        return sum(special, 0.0)
    try:
        return math.fsum(values) + 0.0
    except OverflowError:
        total = sum(map(Fraction, values), Fraction(0))
        try:
            return float(total)
        except OverflowError:
            return math.inf if total > 0 else -math.inf


def _exact_group_sums(codes: np.ndarray, n_groups: int, values: np.ndarray) -> np.ndarray:
    """``exact_sum`` of each group's *values* in code order, with bounded exponent scratch.

    Nonnegative integers use float64 accumulation only when a conservative bound proves
    every partial total is below 2**53. Other values retain the rounded-once exact reducer.
    """
    values = np.ascontiguousarray(values, dtype=np.float64)
    codes = np.asarray(codes, dtype=np.int64)
    integer_sums = _integer_group_sums(codes, n_groups, values)
    if integer_sums is not None:
        return integer_sums
    finite = np.isfinite(values)
    special_codes, special = codes[~finite], values[~finite]
    live = finite & (values != 0.0)
    if not live.all():
        codes, values = codes[live], values[live]
    if n_groups * _EXPONENTS < len(values):
        codes, values = _exponent_partials(codes, n_groups, values)
    sums = _limb_sums(codes, n_groups, values)
    if len(special):
        nan = np.bincount(special_codes[np.isnan(special)], minlength=n_groups) > 0
        high = np.bincount(special_codes[special == np.inf], minlength=n_groups) > 0
        low = np.bincount(special_codes[special == -np.inf], minlength=n_groups) > 0
        sums[high] = np.inf
        sums[low] = -np.inf
        sums[nan | (high & low)] = np.nan
    return sums


def _integer_group_sums(codes: np.ndarray, n_groups: int, values: np.ndarray) -> np.ndarray | None:
    """Exact sums when nonnegative integral values have provably representable totals.

    For maximum M > 0 and group row count c, c <= (2**53 - 1) // M bounds every
    accumulation prefix below 2**53. Python integer division cannot overflow.
    """
    maximum = float(values.max(initial=0.0))
    if not maximum < 2**53 or not values.min(initial=0.0) >= 0.0:
        return None
    for start in range(0, len(values), _SUM_ROWS):
        chunk = values[start : start + _SUM_ROWS]
        if np.any(chunk != np.floor(chunk)):
            return None
    if maximum:
        count = int(np.bincount(codes, minlength=n_groups).max(initial=0))
        if count > ((1 << 53) - 1) // int(maximum):
            return None
    return np.bincount(codes, weights=values, minlength=n_groups).astype(np.float64, copy=False)


def _exponent_partials(
    codes: np.ndarray, n_groups: int, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fewer finite nonzero values with the same group sums.

    Per group and exponent, the high and low mantissa halves each sum to a multiple of
    the bin's scale below 2**53 of it, so every float64 partial sum is exact. Exponents
    whose bin sums could overflow pass through unchanged. Row and bin scratch are bounded;
    occupied bins are compacted when the global bin space exceeds the row slice. Every
    nonzero bin partial is retained separately for the final rounded-once exact reduction.
    """
    size = n_groups * _EXPONENTS
    partial_codes = [np.zeros(0, dtype=np.int64)]
    partial_values = [np.zeros(0)]
    for start in range(0, len(values), _SUM_ROWS):
        chunk_codes = codes[start : start + _SUM_ROWS]
        chunk_values = values[start : start + _SUM_ROWS]
        bits = chunk_values.view(np.uint64)
        exponent = ((bits >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
        passed = exponent >= _BIN_OVERFLOW
        if passed.any():
            partial_codes.append(chunk_codes[passed])
            partial_values.append(chunk_values[passed])
            binned = ~passed
            chunk_codes, chunk_values, bits = (
                chunk_codes[binned],
                chunk_values[binned],
                bits[binned],
            )
            exponent = exponent[binned]
        bins = chunk_codes * _EXPONENTS + exponent
        occupied = None
        if size > _SUM_ROWS:
            occupied, bins = np.unique(bins, return_inverse=True)
        high = (bits & ~_LOW_HALF).view(np.float64)
        low = chunk_values - high
        for half in (high, low):
            partial = np.bincount(bins, weights=half)
            nonzero = np.flatnonzero(partial)
            original = nonzero if occupied is None else occupied[nonzero]
            partial_codes.append(original // _EXPONENTS)
            partial_values.append(partial[nonzero])
    return np.concatenate(partial_codes), np.concatenate(partial_values)


def _limb_sums(codes: np.ndarray, n_groups: int, values: np.ndarray) -> np.ndarray:
    """Group sums of finite nonzero *values* through exact 32-bit limb integers.

    A value is ``±m * 2**(p - 1074)`` with ``m < 2**53``; ``m << (p % 32)`` splits into
    three limbs from limb ``p // 32``. Each group adds its limbs in int64 (exact below
    2**31 rows), carries them into 32-bit limbs and rounds the integer once.
    """
    if not len(values):
        return np.zeros(n_groups)
    bits = values.view(np.uint64)
    biased = (bits >> np.uint64(52)) & np.uint64(0x7FF)
    mantissa = (bits & _FRACTION) | ((biased != 0).astype(np.uint64) << np.uint64(52))
    position = np.maximum(biased, np.uint64(1)) - np.uint64(1)
    limb = (position >> np.uint64(5)).astype(np.int64)
    offset = position & np.uint64(31)
    sign = np.where(bits >> np.uint64(63) != 0, -1, 1)
    parts = (
        (mantissa & (_LIMB_MASK >> offset)) << offset,
        (mantissa >> (_LIMB - offset)) & _LIMB_MASK,
        (mantissa >> _LIMB) >> (_LIMB - offset),
    )
    bottom = np.full(n_groups, np.iinfo(np.int64).max)
    np.minimum.at(bottom, codes, limb)
    top = np.full(n_groups, -1)
    np.maximum.at(top, codes, limb)
    present = top >= 0
    span = np.where(present, top - bottom, 0)
    # Three limbs per value and two for carries: fewer than 2**31 rows add under 32 bits.
    width = min(int(span.max()) + 5, _LIMB_WIDTH)
    wide = span + 5 > width
    bottom = np.where(present, bottom, 0)
    column = codes * width + (limb - bottom[codes])
    wide_rows = wide[codes]
    if wide_rows.any():
        narrow = ~wide_rows
        column, sign = column[narrow], sign[narrow]
        parts = tuple(part[narrow] for part in parts)
    limbs = np.zeros(n_groups * width, dtype=np.int64)
    for shift, part in enumerate(parts):
        np.add.at(limbs, column + shift, sign * part.astype(np.int64))
    limbs = limbs.reshape(n_groups, width)
    _carry(limbs)
    negative = limbs[:, -1] < 0
    limbs[negative] = -limbs[negative]
    _carry(limbs)
    magnitude = _round_limbs(limbs.astype(np.uint64), bottom)
    sums = np.where(negative, -magnitude, magnitude)
    if wide_rows.any():
        # One sort splits the wide groups' rows: the cost grows with rows, not rows x groups.
        rows = np.flatnonzero(wide_rows)
        rows = rows[np.argsort(codes[rows])]
        for group_rows in np.split(rows, np.flatnonzero(np.diff(codes[rows])) + 1):
            sums[codes[group_rows[0]]] = exact_sum(values[group_rows].tolist())
    return sums


def _carry(cells: np.ndarray) -> None:
    """Carry each limb's bits above 32 into the next, leaving lower limbs in [0, 2**32)."""
    for column in range(cells.shape[1] - 1):
        carry = cells[:, column] >> 32
        cells[:, column] -= carry << 32
        cells[:, column + 1] += carry


def _round_limbs(cells: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """Each row's nonnegative limb integer as the nearest float, ties to even.

    Row ``i`` holds ``sum(cells[i, j] * 2**(32 * (bottom[i] + j)))`` in units of
    ``2**-1074``. A sum below the smallest normal float is a multiple of that unit and
    so exact; otherwise its top 53 bits are rounded on the bits below them.
    """
    rows = np.arange(len(cells))
    padded = np.concatenate([np.zeros((len(cells), 2), dtype=np.uint64), cells], axis=1)
    nonzero = padded != 0
    lead = np.maximum(padded.shape[1] - 1 - np.argmax(nonzero[:, ::-1], axis=1), 2)
    head, middle, tail = padded[rows, lead], padded[rows, lead - 1], padded[rows, lead - 2]
    sticky = (nonzero & (np.arange(padded.shape[1]) < (lead - 2)[:, None])).any(axis=1)
    window = (head << _LIMB) | middle
    # Drop the low bits of the 96-bit window (head, middle, tail) past 53 significant bits.
    dropped = np.frexp(head.astype(np.float64))[1].astype(np.uint64) + np.uint64(11)
    short = dropped <= _LIMB
    inner = np.where(short, dropped, _LIMB)
    outer = np.where(short, _LIMB + np.uint64(1), dropped) - _LIMB
    kept = np.where(short, (window << (_LIMB - inner)) | (tail >> inner), window >> outer)
    rest = np.where(
        short,
        tail & ((np.uint64(1) << inner) - np.uint64(1)),
        ((window & ((np.uint64(1) << outer) - np.uint64(1))) << _LIMB) | tail,
    )
    half = np.uint64(1) << (dropped - np.uint64(1))
    kept += (rest > half) | ((rest == half) & (sticky | ((kept & np.uint64(1)) != 0)))
    exponent = 32 * (bottom + lead - 4) + dropped.astype(np.int64) - 1074
    with np.errstate(over="ignore"):
        rounded = np.ldexp(kept.astype(np.float64), exponent)
    return np.where(nonzero.any(axis=1), rounded, 0.0)


def _float_values(series: nw.Series[Any]) -> tuple[np.ndarray, np.ndarray | None]:
    """A column's float64 values and, when it has nulls, which rows hold a value."""
    series = series.cast(nw.Float64)
    if not series.null_count():
        return np.ascontiguousarray(series.to_numpy(), dtype=np.float64), None
    valid = ~np.asarray(series.is_null().to_numpy(), dtype=bool)
    return np.ascontiguousarray(series.fill_null(0.0).to_numpy(), dtype=np.float64), valid


def _group_totals(
    codes: np.ndarray, n_groups: int, values: np.ndarray, valid: np.ndarray | None
) -> np.ndarray:
    """Exact per-group sums of the rows of *values* that hold a value."""
    if valid is None:
        return _exact_group_sums(codes, n_groups, values)
    return _exact_group_sums(codes[valid], n_groups, values[valid])


def _exact_mean(series: nw.Series[Any]) -> float | None:
    """The mean of a column's non-null values from their exact sum; None when it has none."""
    values, valid = _float_values(series)
    if valid is not None:
        values = values[valid]
    if not len(values):
        return None
    return float(_exact_group_sums(np.zeros(len(values), dtype=np.int64), 1, values)[0]) / len(
        values
    )


def _grouped(
    frame: nw.DataFrame[Any],
    keys: Sequence[str],
    aggregations: Mapping[str, nw.Expr] | None = None,
) -> tuple[np.ndarray, nw.DataFrame[Any]]:
    """Each row's dense group code, and one row per group in code order: its *keys* and
    *aggregations*, each evaluated over the group. Codes follow each group's first row."""
    aggregations = aggregations or {}
    row = _scratch_name(frame, "__row__")
    first = _scratch_name(frame, "__first__")
    indexed = frame.with_row_index(row)
    implementation = frame.implementation
    if implementation.is_polars() or implementation.is_pandas_like():
        # These evaluate a window on the rows in place. pyarrow joins a window's values
        # back in thread order, so elsewhere each row's index travels through the join.
        windowed = indexed.with_columns(
            nw.col(row).min().over(*keys).alias(first),
            *(expression.over(*keys).alias(name) for name, expression in aggregations.items()),
        )
        leading = np.asarray(windowed.get_column(first).to_numpy(), dtype=np.int64)
        leaders = np.flatnonzero(leading == np.arange(len(leading)))
        lookup = np.zeros(len(leading), dtype=np.int64)
        lookup[leaders] = np.arange(len(leaders))
        groups = windowed.filter(nw.col(row) == nw.col(first)).select(*keys, *aggregations)
        return lookup[leading], groups
    code = _scratch_name(frame, "__code__")
    groups = (
        indexed.group_by(*keys)
        .agg(
            nw.col(row).min().alias(first),
            *(expression.alias(name) for name, expression in aggregations.items()),
        )
        .sort(first)
    )
    coded = indexed.select(row, *keys).join(
        groups.select(*keys).with_row_index(code), on=list(keys), how="inner"
    )
    # Identity keys are validated non-null, so every row joins its own group.
    assert len(coded) == len(frame)
    codes = np.empty(len(frame), dtype=np.int64)
    codes[coded.get_column(row).to_numpy()] = coded.get_column(code).to_numpy()
    return codes, groups.select(*keys, *aggregations)


def _exact_totals(
    frame: nw.DataFrame[Any],
    keys: Sequence[str],
    sums: Mapping[str, str],
    *,
    maxima: Mapping[str, str] | None = None,
    counts: Sequence[str] = (),
) -> nw.DataFrame[Any]:
    """One row per *keys* group: the *sums* columns' totals, *maxima* and row *counts*.

    Float columns sum exactly; integer and other columns sum natively, which is already
    independent of row order.
    """
    maxima = maxima or {}
    floats = {name: column for name, column in sums.items() if frame.schema[column].is_float()}
    native = {
        **{name: nw.col(column).sum() for name, column in sums.items() if name not in floats},
        **{name: nw.col(column).max() for name, column in maxima.items()},
    }
    needed = dict.fromkeys([*keys, *sums.values(), *maxima.values()])
    codes, groups = _grouped(frame.select(*needed), keys, native)
    n_groups = len(groups)
    backend = groups.implementation
    columns = [
        nw.new_series(
            name,
            _group_totals(codes, n_groups, *_float_values(frame.get_column(column))),
            frame.schema[column],
            backend=backend,
        )
        for name, column in floats.items()
    ]
    columns += [
        nw.new_series(name, np.bincount(codes, minlength=n_groups), nw.Int64, backend=backend)
        for name in counts
    ]
    return groups.with_columns(*columns) if columns else groups


# Residuals and products overflow to inf as native column arithmetic does; an estimator
# reads an overflowed moment as unavailable and refuses with its own code.
@np.errstate(over="ignore", invalid="ignore", divide="ignore")
def emit_centered_moments(
    long: nw.DataFrame[Any],
    plan: MomentPlan,
    *,
    keys: Sequence[str],
    columns: Mapping[str, str],
    successes_of: str | None = None,
) -> list[dict[str, Any]]:
    """Two-phase centered moments of *plan*'s materialized variables, one record per *keys*.

    *columns* maps each materialized plan variable and mask to its column
    in *long*; a plan variable absent from it is not emitted at all (never
    a column of nulls -- see ``_moment_fields``). Phase 1 takes each
    variable's per-partition mean over the same rows and keys; phase 2 sums
    residuals and their products. Every sum is exact (``exact_sum``), so the
    moments are a function of the rows, whatever their order, partitioning or
    dataframe library. Masked moments stay centered on the partition reference,
    never on the masked subgroup's own mean (see ``ArmStats``). Count fields
    stay raw; a null value leaves its row out of that variable's sums.

    *successes_of* names a plan variable whose per-row values are a declared
    0/1 outcome at the grain *long* is reduced at. The aggregate then also
    carries :data:`_SUCCESSES` (the exact integer count of rows equal to 1) and
    :data:`_NONBINARY` (rows equal to neither 0 nor 1, nulls and NaN
    included); the count is meaningful only where ``_NONBINARY`` is 0 (see
    :func:`_exact_successes`). Both are integer counts, never routed through
    float. Callers declare the outcome from the metric type, never from the
    values: 0/1-valued mean data is not a conversion.
    """
    group_cols = list(keys)
    codes, groups = _grouped(long.select(*group_cols), group_cols)
    n_groups = len(groups)
    live = [variable for variable in plan.variables if variable in columns]
    masks = [mask for mask in plan.masked if mask in columns]
    data = {name: _float_values(long.get_column(columns[name])) for name in (*live, *masks)}
    references: dict[str, np.ndarray] = {}
    residuals: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for variable in live:
        values, valid = data[variable]
        count = np.bincount(codes if valid is None else codes[valid], minlength=n_groups)
        references[variable] = _group_totals(codes, n_groups, values, valid) / count
        residuals[variable] = (values - references[variable][codes], valid)
    aggregates: dict[str, np.ndarray] = {
        plan.name(("n", (), None)): np.bincount(codes, minlength=n_groups)
    }
    for moment in (*plan.unmasked_moments(), *plan.masked_moments()):
        kind, variables, mask = moment
        if not all(v in residuals for v in variables) or (mask is not None and mask not in masks):
            continue
        name = plan.name(moment)
        if kind == "ref":
            aggregates[name] = references[variables[0]]
        elif kind == "count":
            assert mask is not None  # a count is always some mask's count
            aggregates[name] = _group_totals(codes, n_groups, *data[mask])
        else:
            product, valid = data[mask] if mask is not None else (None, None)
            for variable in variables:
                term, present = residuals[variable]
                product = term if product is None else product * term
                valid = present if valid is None else valid if present is None else valid & present
            assert product is not None
            aggregates[name] = _group_totals(codes, n_groups, product, valid)
    if successes_of is not None:
        outcome, valid = data[successes_of]
        is_one = outcome == 1.0
        is_zero = outcome == 0.0
        if valid is not None:
            is_one &= valid
            is_zero &= valid
        aggregates[_SUCCESSES] = np.bincount(codes[is_one], minlength=n_groups)
        aggregates[_NONBINARY] = np.bincount(codes[~(is_one | is_zero)], minlength=n_groups)
    return [
        {**key, **{name: values[index].item() for name, values in aggregates.items()}}
        for index, key in enumerate(groups.iter_rows(named=True))
    ]


#: Aggregate columns :func:`emit_centered_moments` adds for a declared binary outcome.
_SUCCESSES = "successes"
_NONBINARY = "__nonbinary_rows"

#: Metric types whose per-unit outcome is a declared 0/1 indicator. The
#: declaration, never the observed values, decides whether a count is carried.
_BINARY_OUTCOME_TYPES = frozenset({"conversion", "retention"})


def _binary_outcome(spec: MetricSpec) -> str | None:
    """The plan variable to count successes of for *spec*, or ``None`` when undeclared."""
    return "y" if spec.type in _BINARY_OUTCOME_TYPES else None


#: Frame row slot order, pinned rather than taken from SLOTS: the frame has emitted
#: cxden right after the x family since format 2, and it is the exported parquet order.
_FRAME_ROW_SLOTS = (
    "ref_y",
    "cy1",
    "cy2",
    "ref_x",
    "cx1",
    "cx2",
    "cxy",
    "cxden",
    "ref_den",
    "cden1",
    "cden2",
    "cden3",
    "cyden",
    "sum_d",
    "cyd",
    "cy2d",
    "cxd",
)
assert frozenset(_FRAME_ROW_SLOTS) == frozenset(SLOTS)


def _exact_successes(record: Mapping[str, Any]) -> int | None:
    """The exact integer success count of an aggregated row, or ``None``.

    Present only when the reduction was declared binary (``successes_of``)
    and every row of this group is exactly 0 or 1; any other value (a cluster
    sum, a fractional or missing outcome) leaves no count that could be read
    as the success total of the ``n`` rows.
    """
    if _SUCCESSES not in record or int(record[_NONBINARY]) != 0:
        return None
    return int(record[_SUCCESSES])


def _moment_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    """Format one aggregated row, emitting literal ``None`` for unpopulated slots.

    Never aggregate a column of nulls: narwhals ``sum()`` over an all-null
    column returns ``0.0`` on pandas, polars and pyarrow while ibis returns
    NULL, and ``_opt`` (engine.py) only maps NaN to None.  A ``0.0`` would
    slip past the ``is None`` guards in cuped.py and variance.py: with a
    zeroed ``ref_x``/``cx1``/``cx2``/``cxy`` family CUPED would skip its
    "covariate not materialised" error and die later at ``var_x_pop <= 0``,
    and with a zeroed ``ref_den``/``cden*`` family ``RatioVarianceModel``
    would proceed to ``d_bar = 0``. The same hazard applies to the uptake
    family: ``ArmStats.mean_d()`` guards on ``sum_d is None``. The emitter
    only aggregates materialized slots, so a slot absent from *record* was
    never computed and is filled in as ``None`` here.

    ``n`` and ``successes`` stay Python integers: counts are never routed
    through float.
    """
    return {
        "n": int(record["n"]),
        "successes": _exact_successes(record),
        **{slot: (float(record[slot]) if slot in record else None) for slot in _FRAME_ROW_SLOTS},
    }


def _apply_spec_missing(
    long: nw.DataFrame[Any],
    spec: MetricSpec,
    *,
    has_x: bool,
    has_den: bool,
    covariate: Any | None = None,
) -> nw.DataFrame[Any]:
    """Apply *spec*'s declared null/NaN policy to its own selected columns.

    *long* is the per-spec projection (``y``/``x``/``y_den`` already cast
    to Float64), so one metric's declaration can never touch another
    metric's values even when both read the same source column.
    ``missing="zero"`` counts a missing value as 0 with the row still in
    ``n``; ``missing="skip"`` removes the row from THIS metric's moments
    (observed units only). A missing covariate value is filled with the pooled
    mean of the observed values (or 0 under ``missing="zero"``), computed from
    the covariate column alone -- BEFORE ``missing="skip"`` removes any
    outcome-null rows below, never from the arm or the outcome. The adjusted
    estimate stays unbiased when the filled covariate keeps equal means across
    arms: for a zero fill, when values and their missingness are distributed alike
    in both arms (equal missing shares are not enough); for a mean fill, when
    which values are missing is unrelated to their size. NaN is normalized to
    null first so every backend agrees on what "missing" is.
    """
    value_cols = ["y"] + (["y_den"] if has_den else [])
    if has_x:
        x_missing = int(
            long.select((nw.col("x").is_null() | nw.col("x").is_nan()).cast(nw.Int64).sum()).item()
        )
        if x_missing:
            assert covariate is not None
            if covariate.missing == "refuse":
                refuse(
                    COVARIATE_COLUMN_MISSING,
                    metric=spec.name,
                    covariate=covariate.name,
                    n_missing=x_missing,
                )
            if covariate.missing == "zero":
                fill = 0.0
            else:
                mean = _exact_mean(long.get_column("x").fill_nan(None))
                if mean is None:
                    refuse(
                        COVARIATE_IMPUTE_ALL_NULL,
                        metric=spec.name,
                        covariate=covariate.name,
                    )
                fill = float(mean)
            long = long.with_columns(nw.col("x").fill_nan(None).fill_null(fill).alias("x"))
    if spec.missing == "zero":
        long = long.with_columns(
            *(nw.col(c).fill_nan(None).fill_null(0.0).alias(c) for c in value_cols)
        )
    elif spec.missing == "skip":
        predicate = nw.col(value_cols[0]).is_null() | nw.col(value_cols[0]).is_nan()
        for c in value_cols[1:]:
            predicate = predicate | nw.col(c).is_null() | nw.col(c).is_nan()
        long = long.filter(~predicate)
    return long


def _apply_winsorization(
    long: nw.DataFrame[Any],
    spec: MetricSpec,
    *,
    key_cols: Sequence[str],
) -> tuple[nw.DataFrame[Any], dict[tuple[Any, ...], dict[str, float | int | None]]]:
    """Clip pooled per-unit outcomes and count caps within each result key."""
    config = spec.winsorization
    if config is None:
        return long, {}
    long = long.with_columns(nw.col("y").alias("y_raw"))

    lower = config.lower_value
    if config.lower_percentile is not None:
        lower = float(
            long.select(
                nw.col("y").quantile(config.lower_percentile, interpolation="linear")
            ).item()
        )
    upper = config.upper_value
    if config.upper_percentile is not None:
        upper = float(
            long.select(
                nw.col("y").quantile(config.upper_percentile, interpolation="linear")
            ).item()
        )
    flagged = long
    count_exprs = [nw.len().alias("winsor_n")]
    if lower is not None:
        flagged = flagged.with_columns((nw.col("y") < lower).cast(nw.Int64).alias("__winsor_lower"))
        count_exprs.append(nw.col("__winsor_lower").sum().alias("winsor_n_lower"))
    if upper is not None:
        flagged = flagged.with_columns((nw.col("y") > upper).cast(nw.Int64).alias("__winsor_upper"))
        count_exprs.append(nw.col("__winsor_upper").sum().alias("winsor_n_upper"))
    counts = flagged.group_by(list(key_cols)).agg(*count_exprs)
    clipped = flagged.with_columns(nw.col("y").clip(lower, upper).alias("y"))
    helper_columns = [
        name for name in ("__winsor_lower", "__winsor_upper") if name in clipped.columns
    ]
    if helper_columns:
        clipped = clipped.drop(*helper_columns)
    metadata: dict[tuple[Any, ...], dict[str, float | int | None]] = {}
    for row in counts.iter_rows(named=True):
        metadata[tuple(row[name] for name in key_cols)] = {
            "winsor_lower_percentile": config.lower_percentile,
            "winsor_upper_percentile": config.upper_percentile,
            "winsor_lower_bound": lower,
            "winsor_upper_bound": upper,
            "winsor_n": int(row["winsor_n"]),
            "winsor_n_lower": int(row.get("winsor_n_lower") or 0),
            "winsor_n_upper": int(row.get("winsor_n_upper") or 0),
        }
    return clipped, metadata


def _winsor_metadata_for_record(
    record: Mapping[str, Any],
    metadata: Mapping[tuple[Any, ...], dict[str, float | int | None]],
    *,
    key_cols: Sequence[str],
) -> dict[str, float | int | None]:
    return metadata.get(
        tuple(record[name] for name in key_cols),
        dict.fromkeys(WINSORIZATION_MOMENT_FIELDS, None),
    )


def _moment_group_cols(*fixed: str, by: Sequence[str]) -> list[str]:
    """Return a moment grouping key with an optional declared breakout."""
    return [*fixed, *by]


def _drop_degenerate_winsor_bounds(
    metadata: dict[tuple[Any, ...], dict[str, float | int | None]],
    *,
    metric_name: str,
) -> dict[tuple[Any, ...], dict[str, float | int | None]]:
    """A percentile cutoff is resolved once per date, pooled across every
    arm and breakout group there - a collapsed pair (lower >= upper) is a
    property of the WHOLE date, never one key alone. Deciding per key
    would report different metadata for different arms on the same date,
    which the estimation engine rejects as inconsistent between arms.

    Nothing on this date actually fell outside ``[lower, upper]`` (every
    key's cap counts are zero): clipping to a collapsed pair changed
    nothing, so report the same "no cap" representation an unconfigured
    metric carries. Something WAS clipped to a collapsed pair: that
    silently drags a real value to the wrong number while looking
    uncapped, so refuse instead of returning it.
    """
    if not metadata:
        return metadata
    sample = next(iter(metadata.values()))
    lower, upper = sample.get("winsor_lower_bound"), sample.get("winsor_upper_bound")
    if lower is None or upper is None or lower < upper:
        return metadata
    clipped = sum(
        int(meta.get("winsor_n_lower") or 0) + int(meta.get("winsor_n_upper") or 0)
        for meta in metadata.values()
    )
    if clipped == 0:
        no_cap = dict.fromkeys(WINSORIZATION_MOMENT_FIELDS, None)
        return dict.fromkeys(metadata, no_cap)
    ds = next(iter(metadata))[0]
    refuse(
        _WINSOR_COLLAPSED_BOUND,
        metric=metric_name,
        date=ds,
        lower=lower,
        upper=upper,
        clipped=clipped,
    )


def _day_moment_records(
    long: nw.DataFrame[Any],
    spec: MetricSpec,
    *,
    by: Sequence[str],
    has_den: bool,
    has_d: bool = False,
) -> Iterator[tuple[dict[str, Any], dict[str, float | int | None]]]:
    """Centered moments per ``ds``, winsorized within each date.

    A percentile cutoff is estimated separately per date - pooling across
    dates would leak future/past data into today's bound (D4). A fixed-value
    cap or an unconfigured metric has no such leakage risk, so it stays one
    vectorized reduction over every date, matching the total-grain contract.
    """
    if long.is_empty():
        return
    key_cols = ("ds", "group_id", *by)
    config = spec.winsorization
    percentile = config is not None and (
        config.lower_percentile is not None or config.upper_percentile is not None
    )
    partitions = (part for _, part in long.group_by("ds")) if percentile else (long,)
    for part in partitions:
        clipped, metadata = _apply_winsorization(part, spec, key_cols=key_cols)
        if percentile:
            metadata = _drop_degenerate_winsor_bounds(metadata, metric_name=spec.name)
        columns = {"y": "y"}
        if has_den:
            columns["den"] = "y_den"
        if has_d:
            columns["d"] = "d"
        summary = emit_centered_moments(
            clipped,
            DAY_GRAIN,
            keys=list(key_cols),
            columns=columns,
            successes_of=_binary_outcome(spec),
        )
        for record in summary:
            yield record, _winsor_metadata_for_record(record, metadata, key_cols=key_cols)


def _partitioned_moment_rows(
    frame: nw.DataFrame[Any],
    *,
    group: str,
    metrics: Sequence[MetricSpec],
    experiment_id: str,
    uptake: str | None,
    cluster: str | None,
    by: Sequence[str],
    outcome_column: str | None,
    denominator_column: str | None,
    covariates: _CovariateReduction,
) -> list[dict[str, Any]]:
    assert len(by) == 1 and covariates.cell_failures is not None
    rows = []
    for key, partition in frame.group_by(list(by)):
        label = "__null__" if key[0] is None else str(key[0])
        admitted = tuple(
            spec for spec in metrics if (spec.name, by[0], label) not in covariates.cell_failures
        )
        if not admitted:
            continue
        failures: dict[str, _DeferredRefusal] = {}
        try:
            rows.extend(
                _moment_rows(
                    partition,
                    group=group,
                    metrics=admitted,
                    experiment_id=experiment_id,
                    uptake=uptake,
                    cluster=cluster,
                    by=by,
                    outcome_column=outcome_column,
                    denominator_column=denominator_column,
                    covariates=replace(covariates, failures=failures, cell_failures=None),
                )
            )
        except InvalidRequestError as exc:
            if exc.code not in SCALAR_COVARIATE_REFUSALS:
                raise
            failures[str(exc.context["metric"])] = (
                SCALAR_COVARIATE_REFUSALS[exc.code],
                exc.context,
            )
        for metric, failure in failures.items():
            covariates.cell_failures[(metric, by[0], label)] = failure
    return rows


def _moment_rows(
    frame: nw.DataFrame[Any],
    *,
    group: str,
    metrics: Sequence[MetricSpec],
    experiment_id: str,
    uptake: str | None = None,
    cluster: str | None = None,
    by: Sequence[str] = (),
    outcome_column: str | None = None,
    denominator_column: str | None = None,
    covariates: _CovariateReduction | None = None,
) -> list[dict[str, Any]]:
    """Build centered moments after each metric's pooled unit transform."""
    if covariates is not None and covariates.cell_failures is not None and by:
        return _partitioned_moment_rows(
            frame,
            group=group,
            metrics=metrics,
            experiment_id=experiment_id,
            uptake=uptake,
            cluster=cluster,
            by=by,
            outcome_column=outcome_column,
            denominator_column=denominator_column,
            covariates=covariates,
        )
    covariate_columns = None if covariates is None else covariates.columns
    covariate_means = None if covariates is None else covariates.means
    covariate_failures = None if covariates is None else covariates.failures
    rows: list[dict[str, Any]] = []
    has_d = uptake is not None

    for spec in metrics:
        covariate = next(
            (
                covariate
                for method in spec._methods
                if method.variance_reduction == "cuped"
                for covariate in method.covariates or ()
            ),
            None,
        )
        has_x = covariate is not None
        has_den = spec.denominator is not None
        key_cols = ("group_id", *by)

        if cluster is not None:
            has_uptake = uptake is not None
            select_exprs = [
                *[nw.col(name) for name in by],
                nw.col(group).alias("group_id"),
                nw.col(cluster).alias("__cluster__"),
                nw.col(outcome_column or spec.y_column).cast(nw.Float64).alias("y"),
            ]
            if has_den:
                select_exprs.append(
                    nw.col(denominator_column or spec.denominator).cast(nw.Float64).alias("y_den")
                )
            if has_uptake:
                select_exprs.append(nw.col(uptake).cast(nw.Float64).alias("d"))
            long = frame.select(*select_exprs)
            long = _apply_spec_missing(long, spec, has_x=False, has_den=has_den, covariate=None)
            long, winsor_metadata = _apply_winsorization(long, spec, key_cols=key_cols)
            per_cluster = _exact_totals(
                long,
                _moment_group_cols("group_id", "__cluster__", by=by),
                {
                    "y": "y",
                    **({"y_den": "y_den"} if has_den else {}),
                    **({"x": "d"} if has_uptake else {}),
                },
                counts=[*(() if has_den else ("y_den",)), *(() if has_uptake else ("x",))],
            )
            plan = CLUSTER_UPTAKE_GRAIN if has_uptake else CLUSTER_SIZE_GRAIN
            x_var = plan.x_variable()
            assert x_var is not None
            summary = emit_centered_moments(
                per_cluster.select(
                    *by,
                    "group_id",
                    "y",
                    nw.col("y_den").cast(nw.Float64),
                    nw.col("x").cast(nw.Float64),
                ),
                plan,
                keys=_moment_group_cols("group_id", by=by),
                columns={"y": "y", "den": "y_den", x_var: "x"},
            )
            for record in summary:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "metric": spec.name,
                        **{name: str(record[name]) for name in by},
                        "group_id": record["group_id"],
                        "x_role": X_SLOT_ROLES[x_var],
                        **_moment_fields(record),
                        **_winsor_metadata_for_record(record, winsor_metadata, key_cols=key_cols),
                    }
                )
            continue

        exprs = [nw.col(outcome_column or spec.y_column).cast(nw.Float64).alias("y")]
        if has_x:
            x_column = (covariate_columns or {}).get(covariate.name, covariate.name)
            exprs.append(nw.col(x_column).cast(nw.Float64).alias("x"))
        if has_den:
            exprs.append(
                nw.col(denominator_column or spec.denominator).cast(nw.Float64).alias("y_den")
            )
        if uptake is not None:
            exprs.append(nw.col(uptake).cast(nw.Float64).alias("d"))

        long = frame.select(
            *[nw.col(name) for name in by],
            nw.col(group).alias("group_id"),
            *exprs,
        )
        if has_x and covariate.missing == "impute" and covariate_means is not None:
            mean = covariate_means[covariate.name]
            if mean is not None:
                long = long.with_columns(nw.col("x").fill_nan(None).fill_null(mean).alias("x"))
        try:
            long = _apply_spec_missing(
                long, spec, has_x=has_x, has_den=has_den, covariate=covariate
            )
        except InvalidRequestError as exc:
            if covariate_failures is None or exc.code not in SCALAR_COVARIATE_REFUSALS:
                raise
            covariate_failures[spec.name] = (SCALAR_COVARIATE_REFUSALS[exc.code], exc.context)
            has_x = False
            long = _apply_spec_missing(long.drop("x"), spec, has_x=False, has_den=has_den)
        long, winsor_metadata = _apply_winsorization(long, spec, key_cols=key_cols)
        columns = {"y": "y"}
        if has_x:
            columns["x"] = "x"
        if has_den:
            columns["den"] = "y_den"
        if has_d:
            columns["d"] = "d"
        summary = emit_centered_moments(
            long,
            UNIT_GRAIN,
            keys=_moment_group_cols("group_id", by=by),
            columns=columns,
            successes_of=_binary_outcome(spec),
        )

        for record in summary:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "metric": spec.name,
                    **{name: str(record[name]) for name in by},
                    "group_id": record["group_id"],
                    "x_role": X_SLOT_ROLES["x"] if has_x else None,
                    **_moment_fields(record),
                    **_winsor_metadata_for_record(record, winsor_metadata, key_cols=key_cols),
                }
            )

    return rows


def _daily_moment_rows(
    panel: nw.DataFrame[Any],
    *,
    bounded_dense: bool,
    identity: nw.DataFrame[Any],
    identity_ordinal: str,
    metrics: Sequence[MetricSpec],
    synthesised: Sequence[Metric],
    experiment_id: str,
    exposure: nw.DataFrame[Any],
    by: Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, _DeferredRefusal]]:
    """Reduce bounded dense workspaces at once and larger panels one day at a time.

    Each day's population is the units exposed by that day.
    """
    from increment._frame_panel import _day_population

    labels = panel.get_column("ds").unique().to_list()

    axis_labels = (
        [*labels, *exposure.get_column("__exposure__").unique().to_list()]
        if isinstance(panel.schema["ds"], nw.String)
        else None
    )
    day_slices = _partition_panel_days(panel) if not bounded_dense else None
    rows: list[dict[str, Any]] = []
    failures: dict[str, _DeferredRefusal] = {}
    for spec, metric in zip(metrics, synthesised, strict=True):
        if spec.type in ("retention", "quantile"):
            continue
        has_den = spec.denominator is not None
        right_edge = _resolve_window_days(metric)
        spec_rows: list[dict[str, Any]] = []
        try:
            if axis_labels is not None:
                _day_axis_label_order(axis_labels)
            for ds in labels if not bounded_dense else [None]:
                if bounded_dense:
                    population = panel
                else:
                    assert day_slices is not None
                    values = [spec.y_column]
                    if spec.denominator is not None:
                        values.append(spec.denominator)
                    population = _day_population(
                        panel,
                        identity=identity,
                        ds=ds,
                        value_columns=values,
                        ordinal=identity_ordinal,
                        day_slices=day_slices,
                    )
                indexed, day_index = _with_day_index(population, exposure)
                population = indexed.filter(nw.col(day_index) >= 0)
                if right_edge is not None:
                    population = population.filter(nw.col(day_index) < right_edge)
                long = population.select(
                    nw.col("ds"),
                    *[nw.col(name) for name in by],
                    nw.col("group_id"),
                    nw.col(spec.y_column).alias("y"),
                    *(
                        [nw.col(spec.denominator).alias("y_den")]
                        if spec.denominator is not None
                        else []
                    ),
                )
                for record, winsor_metadata in _day_moment_records(
                    long, spec, by=by, has_den=has_den, has_d=False
                ):
                    spec_rows.append(
                        {
                            "ds": record["ds"],
                            **{name: str(record[name]) for name in by},
                            "experiment_id": experiment_id,
                            "metric": spec.name,
                            "group_id": record["group_id"],
                            **_moment_fields(record),
                            **winsor_metadata,
                        }
                    )
        except CapabilityError as exc:
            if exc.code != _WINSOR_COLLAPSED_BOUND.code:
                raise
            failures[spec.name] = (_WINSOR_COLLAPSED_BOUND, exc.context)
            continue
        rows.extend(spec_rows)
    return rows, failures


def _asof_day_rank(ds: nw.Series[Any]) -> tuple[np.ndarray, int]:
    """Chronological rank of every row's day label, dense from 0, and the day count.

    Ranks come from ``_day_axis_label_order`` over the distinct labels, so a
    string axis orders the way the constructor justified (``d9`` before
    ``d10``) and an unorderable one refuses by the same code.
    """
    order = _day_axis_label_order(ds.unique().to_list())
    ranked = sorted(order, key=order.__getitem__)
    ranks: list[int] = []
    position = -1
    previous: object = object()
    for label in ranked:
        if order[label] != previous:
            position += 1
            previous = order[label]
        ranks.append(position)
    coded = ds.replace_strict(ranked, ranks, return_dtype=nw.Int64).to_numpy()
    return coded.astype(np.int64, copy=False), position + 1


def _asof_unit_code(ordinal: nw.Series[Any]) -> tuple[np.ndarray, int]:
    """Dense unit codes from the stable identity ordinal, never public identifiers."""
    labels = sorted(ordinal.unique().to_list())
    coded = ordinal.replace_strict(
        labels, list(range(len(labels))), return_dtype=nw.Int64
    ).to_numpy()
    return coded.astype(np.int64, copy=False), len(labels)


def _asof_float_column(source: nw.DataFrame[Any], name: str) -> np.ndarray:
    return source.get_column(name).cast(nw.Float64).to_numpy().astype(np.float64, copy=False)


def _asof_running_sum(values: np.ndarray) -> np.ndarray:
    """Left-to-right cumulative sum of each row, from a state that starts at +0.0.

    Adding +0.0 afterwards changes no value except a negative zero, which
    a sum that started from +0.0 could never have produced.
    """
    return np.cumsum(values, axis=1) + 0.0


def _asof_spine_cells(
    day_rank: np.ndarray, unit_code: np.ndarray, *, n_days: int, n_units: int
) -> tuple[np.ndarray, np.ndarray]:
    """Source row index of every ``(unit, day)`` cell of the dense spine, and which cells exist.

    Callers normally hand in the densified unit x day spine, where every
    cell is filled exactly once; a direct caller may pass a sparser panel,
    so an unfilled cell reports ``filled=False`` and a harmless index of 0
    (never read, because ``filled`` gates every downstream mask).
    """
    n_rows = day_rank.shape[0]
    cells = np.full(n_units * n_days, -1, dtype=np.int64)
    cells[unit_code * n_days + day_rank] = np.arange(n_rows, dtype=np.int64)
    filled = cells >= 0
    assert filled.sum() == n_rows
    return np.where(filled, cells, 0).reshape(n_units, n_days), filled.reshape(n_units, n_days)


def _asof_grid(source: nw.DataFrame[Any], cells: np.ndarray, name: str) -> np.ndarray:
    return _asof_float_column(source, name)[cells]


def _asof_outcome_state(
    source: nw.DataFrame[Any],
    cells: np.ndarray,
    admitted: np.ndarray,
    day_index: np.ndarray,
    *,
    spec: MetricSpec,
    is_retention: bool,
    band_start: int,
    band_end: int | None,
    right_edge: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The outcome-open gate for each cell, and the resulting cumulative ``y`` state."""
    y = _asof_grid(source, cells, spec.y_column)
    if is_retention:
        outcome_open = admitted & (day_index >= band_start)
        if band_end is not None:
            outcome_open &= day_index < band_end
    elif right_edge is None:
        outcome_open = admitted
    else:
        outcome_open = admitted & (day_index < right_edge)
    if is_retention or spec.type == "conversion":
        y_state = np.maximum.accumulate(outcome_open & (y != 0.0), axis=1).astype(np.float64)
    else:
        y_state = _asof_running_sum(np.where(outcome_open, y, 0.0))
    return outcome_open, y_state


def _asof_uptake_state(
    source: nw.DataFrame[Any],
    cells: np.ndarray,
    admitted: np.ndarray,
    day_index: np.ndarray,
    *,
    uptake: str,
    uptake_window_days: int | None,
) -> np.ndarray:
    """The cumulative uptake ratchet, gated on the exposure-relative day index."""
    uptake_open = admitted & (day_index >= 0)
    if uptake_window_days is not None:
        uptake_open &= day_index < uptake_window_days
    taken = _asof_grid(source, cells, uptake) != 0.0
    return np.maximum.accumulate(uptake_open & taken, axis=1).astype(np.float64)


def _asof_unit_rows(
    panel: nw.DataFrame[Any],
    *,
    spec: MetricSpec,
    metric: Metric,
    by: Sequence[str],
    exposure: nw.DataFrame[Any],
    uptake: str | None,
    uptake_window_days: int | None,
    completed_windows_only: bool,
    identity_ordinal: str,
) -> nw.DataFrame[Any]:
    """One cumulative, admitted outcome row per unit and observed day.

    The panel is the dense unit x day spine, so each unit's day-ordered
    values form one row of a ``(units, days)`` grid; a cumulative sum along
    that row adds values strictly left to right from ``+0.0``, the order
    the per-unit state advanced in, and a masked day adds an exact
    ``+0.0``. Retention
    enters once its band opens (or closes when final-only rows are
    requested), then ratchets as a binary outcome; its bounded band freezes
    after the right edge. Rows come out day-major, units in identity-ordinal order.
    """
    if isinstance(metric, QuantileMetric):
        refuse(_ASOF_QUANTILE_UNSUPPORTED, metric=metric.name)
    is_retention = isinstance(metric, RetentionMetric)
    band_start, band_end = metric.band if is_retention else (0, None)
    if is_retention and completed_windows_only and band_end is None:
        _raise("frame.moments.asof_moments_completed", metric=metric.name, band=metric.band)

    source, day_index_name = _with_day_index(panel, exposure)
    right_edge = _resolve_window_days(metric)
    completion_days: int | None = None
    if completed_windows_only:
        if uptake is not None and (right_edge is None or uptake_window_days is None):
            _raise("frame.moments.asof_completed_requires_uptake_window")
        if uptake is not None or (not is_retention and right_edge is not None):
            assert right_edge is not None  # guaranteed by the raise above when uptake is set
            completion_days = max(right_edge, uptake_window_days or 0)
    gate = band_end if completed_windows_only else band_start
    columns = ["ds", "unit_id", "group_id", "y", *by]
    if spec.denominator is not None:
        columns.append("y_den")
    if uptake is not None:
        columns.append("d")
    if source.shape[0] == 0:
        return nw.from_dict({name: [] for name in columns}, backend=panel.implementation)

    day_rank, n_days = _asof_day_rank(source.get_column("ds"))
    unit_code, n_units = _asof_unit_code(source.get_column(identity_ordinal))
    cells, filled = _asof_spine_cells(day_rank, unit_code, n_days=n_days, n_units=n_units)

    day_index = _asof_grid(source, cells, day_index_name)
    admitted = filled & ~(day_index < 0)
    outcome_open, y_state = _asof_outcome_state(
        source,
        cells,
        admitted,
        day_index,
        spec=spec,
        is_retention=is_retention,
        band_start=band_start,
        band_end=band_end,
        right_edge=right_edge,
    )
    values: dict[str, np.ndarray] = {"y": y_state}
    if spec.denominator is not None:
        den = _asof_grid(source, cells, spec.denominator)
        values["y_den"] = _asof_running_sum(np.where(outcome_open, den, 0.0))
    if uptake is not None:
        values["d"] = _asof_uptake_state(
            source,
            cells,
            admitted,
            day_index,
            uptake=uptake,
            uptake_window_days=uptake_window_days,
        )

    keep = admitted
    if completion_days is not None:
        keep = keep & ~(day_index < completion_days)
    if is_retention:
        assert gate is not None
        keep = keep & ~(day_index < gate)

    # Day-major emission (ds rank, then identity ordinal), the order the row loop sorted in.
    emitted = keep.T.reshape(-1)
    rows = source.select("ds", "unit_id", "group_id", *by)[cells.T.reshape(-1)[emitted]]
    rows = rows.with_columns(
        *[
            nw.new_series(name, state.T.reshape(-1)[emitted], backend=panel.implementation)
            for name, state in values.items()
        ]
    )
    return rows.select(*columns)


def _effective_observable_end(
    spec: MetricSpec,
    *,
    panel_ds_max: Any,
    panel_order: Mapping[Any, tuple[Any, ...]],
    observation_end: Any,
    fact_max_ds: Mapping[str, Any],
) -> Any:
    """Use the earlier component horizon; an absent component uses the panel extent."""
    if observation_end is not None:
        return observation_end
    if panel_ds_max is None:
        return None
    if spec.denominator is None:
        return fact_max_ds.get(spec.y_column, panel_ds_max)
    numerator_end = fact_max_ds.get(spec.y_column, panel_ds_max)
    denominator_end = fact_max_ds.get(spec.denominator, panel_ds_max)
    return min((numerator_end, denominator_end), key=panel_order.__getitem__)


@dataclass(frozen=True)
class _AsOfSettings:
    identity: nw.DataFrame[Any]
    identity_ordinal: str
    bounded_dense: bool
    synthesised: Sequence[Metric]
    experiment_id: str
    uptake: str | None
    uptake_window_days: int | None
    exposure: nw.DataFrame[Any]
    completed_windows_only: bool
    observation_end: dt.date | dt.datetime | str | int | float | None
    fact_max_ds: Mapping[str, Any]
    by: Sequence[str]


@dataclass
class _AsOfState:
    y: np.ndarray
    denominator: np.ndarray
    uptake: np.ndarray


def _asof_completion_days(metric: Metric, settings: _AsOfSettings) -> int | None:
    if not settings.completed_windows_only:
        return None
    right_edge = _resolve_window_days(metric)
    if settings.uptake is not None and (right_edge is None or settings.uptake_window_days is None):
        _raise("frame.moments.asof_completed_requires_uptake_window")
    if settings.uptake is not None or (
        not isinstance(metric, RetentionMetric) and right_edge is not None
    ):
        assert right_edge is not None
        return max(right_edge, settings.uptake_window_days or 0)
    return None


def _asof_update_uptake(
    population: nw.DataFrame[Any],
    settings: _AsOfSettings,
    admitted: np.ndarray,
    day_index_values: np.ndarray,
    state: _AsOfState,
) -> None:
    if settings.uptake is None:
        return
    uptake_open = admitted & (day_index_values >= 0)
    if settings.uptake_window_days is not None:
        uptake_open &= day_index_values < settings.uptake_window_days
    taken = _asof_float_column(population, settings.uptake) != 0.0
    state.uptake = np.maximum(state.uptake, (uptake_open & taken).astype(np.float64))


def _asof_stream_day(
    panel: nw.DataFrame[Any],
    identity: nw.DataFrame[Any],
    spec: MetricSpec,
    metric: Metric,
    ds: Any,
    settings: _AsOfSettings,
    state: _AsOfState,
    completion_days: int | None,
    day_slices: Mapping[Any, nw.DataFrame[Any]],
) -> nw.DataFrame[Any]:
    from increment._frame_panel import _day_population

    is_retention = isinstance(metric, RetentionMetric)
    band_start, band_end = metric.band if is_retention else (0, None)
    right_edge = _resolve_window_days(metric)
    value_columns = list(
        dict.fromkeys(
            [
                spec.y_column,
                *([spec.denominator] if spec.denominator is not None else []),
                *([settings.uptake] if settings.uptake is not None else []),
            ]
        )
    )
    population = _day_population(
        panel,
        identity=identity,
        ds=ds,
        value_columns=value_columns,
        ordinal=settings.identity_ordinal,
        day_slices=day_slices,
    )
    population, day_index = _with_day_index(population, settings.exposure)
    population = population.sort(settings.identity_ordinal)
    day_index_values = population.get_column(day_index).to_numpy().astype(np.float64, copy=False)
    admitted = day_index_values >= 0

    y = _asof_float_column(population, spec.y_column)
    if is_retention:
        outcome_open = admitted & (day_index_values >= band_start)
        if band_end is not None:
            outcome_open &= day_index_values < band_end
    elif right_edge is None:
        outcome_open = admitted
    else:
        outcome_open = admitted & (day_index_values < right_edge)
    if is_retention or spec.type == "conversion":
        state.y = np.maximum(state.y, (outcome_open & (y != 0.0)).astype(np.float64))
    else:
        state.y += np.where(outcome_open, y, 0.0)
    if spec.denominator is not None:
        den = _asof_float_column(population, spec.denominator)
        state.denominator += np.where(outcome_open, den, 0.0)

    _asof_update_uptake(population, settings, admitted, day_index_values, state)
    keep = admitted.copy()
    if completion_days is not None:
        keep &= day_index_values >= completion_days
    if is_retention:
        gate = band_end if settings.completed_windows_only else band_start
        assert gate is not None
        keep &= day_index_values >= gate
    state_columns = [nw.new_series("y", state.y, backend=panel.implementation)]
    if spec.denominator is not None:
        state_columns.append(
            nw.new_series("y_den", state.denominator, backend=panel.implementation)
        )
    if settings.uptake is not None:
        state_columns.append(nw.new_series("d", state.uptake, backend=panel.implementation))
    return (
        population.with_columns(
            *state_columns,
            nw.new_series("__keep", keep, backend=panel.implementation),
        )
        .filter(nw.col("__keep"))
        .select(
            "ds",
            *settings.by,
            "group_id",
            "y",
            *(["y_den"] if spec.denominator is not None else []),
            *(["d"] if settings.uptake is not None else []),
        )
    )


def _asof_metric_rows(
    records: nw.DataFrame[Any],
    spec: MetricSpec,
    settings: _AsOfSettings,
) -> list[dict[str, Any]]:
    result = []
    if records.is_empty():
        return result
    for record, winsor_metadata in _day_moment_records(
        records,
        spec,
        by=settings.by,
        has_den=spec.denominator is not None,
        has_d=settings.uptake is not None,
    ):
        result.append(
            {
                "ds": record["ds"],
                **{name: str(record[name]) for name in settings.by},
                "experiment_id": settings.experiment_id,
                "metric": spec.name,
                "group_id": record["group_id"],
                **_moment_fields(record),
                **winsor_metadata,
            }
        )
    return result


def _asof_streamed_rows(
    panel: nw.DataFrame[Any],
    identity: nw.DataFrame[Any],
    spec: MetricSpec,
    metric: Metric,
    labels: Sequence[Any],
    settings: _AsOfSettings,
    day_slices: Mapping[Any, nw.DataFrame[Any]],
) -> list[dict[str, Any]]:
    state = _AsOfState(
        np.zeros(identity.shape[0], dtype=np.float64),
        np.zeros(identity.shape[0], dtype=np.float64),
        np.zeros(identity.shape[0], dtype=np.float64),
    )
    completion_days = _asof_completion_days(metric, settings)
    result = []
    for ds in labels:
        records = _asof_stream_day(
            panel, identity, spec, metric, ds, settings, state, completion_days, day_slices
        )
        result.extend(_asof_metric_rows(records, spec, settings))
    return result


def _asof_dense_rows(
    panel: nw.DataFrame[Any],
    spec: MetricSpec,
    metric: Metric,
    metric_identity: nw.DataFrame[Any],
    settings: _AsOfSettings,
) -> list[dict[str, Any]]:
    metric_panel = panel.join(metric_identity.select("unit_id"), on="unit_id", how="semi")
    records = _asof_unit_rows(
        metric_panel,
        spec=spec,
        metric=metric,
        by=settings.by,
        exposure=settings.exposure,
        uptake=settings.uptake,
        uptake_window_days=settings.uptake_window_days,
        completed_windows_only=settings.completed_windows_only,
        identity_ordinal=settings.identity_ordinal,
    )
    return _asof_metric_rows(records, spec, settings)


def _asof_moment_rows(
    panel: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    settings: _AsOfSettings,
) -> tuple[list[dict[str, Any]], dict[str, _DeferredRefusal]]:
    """Advance cumulative states chronologically with O(units) retained state."""
    from increment._frame_panel import _day_labels

    labels = _day_labels(panel)
    rows: list[dict[str, Any]] = []
    failures: dict[str, _DeferredRefusal] = {}
    if not labels:
        return rows, failures
    day_slices = _partition_panel_days(panel) if not settings.bounded_dense else None
    panel_order = _day_axis_label_order(labels)
    panel_ds_max = max(labels, key=panel_order.__getitem__)
    for spec, metric in zip(specs, settings.synthesised, strict=True):
        if spec.type == "quantile" or (
            settings.completed_windows_only
            and isinstance(metric, RetentionMetric)
            and metric.band[1] is None
        ):
            continue
        metric_identity = settings.identity
        maturity = _final_maturity_day(metric)
        if (
            settings.completed_windows_only
            and spec.denominator is not None
            and maturity is not None
        ):
            observable_end = _effective_observable_end(
                spec,
                panel_ds_max=panel_ds_max,
                panel_order=panel_order,
                observation_end=settings.observation_end,
                fact_max_ds=settings.fact_max_ds,
            )
            if observable_end is not None:
                observable = _observable_end_index(settings.exposure, observable_end)
                eligible = observable.filter(nw.col("__observable_days__") >= maturity).select(
                    "unit_id"
                )
                metric_identity = settings.identity.join(eligible, on="unit_id", how="semi")
        try:
            if settings.bounded_dense:
                metric_rows = _asof_dense_rows(panel, spec, metric, metric_identity, settings)
            else:
                assert day_slices is not None
                metric_rows = _asof_streamed_rows(
                    panel,
                    metric_identity,
                    spec,
                    metric,
                    labels,
                    settings,
                    day_slices=day_slices,
                )
        except CapabilityError as exc:
            if exc.code != _WINSOR_COLLAPSED_BOUND.code:
                raise
            failures[spec.name] = (_WINSOR_COLLAPSED_BOUND, exc.context)
            continue
        rows.extend(metric_rows)
    return rows, failures


def _collapse_to_unit_totals(
    panel: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    *,
    uptake: str | None = None,
    window_days: int | None = None,
    exposure: nw.DataFrame[Any] | None = None,
    by: Sequence[str] = (),
) -> nw.DataFrame[Any]:
    """Sum sparse or dense daily values; collapse uptake with MAX.

    With a declared *exposure*, uptake counts on ``[exposure, exposure +
    window_days)``, with no end if unbounded. Without one, every observed uptake
    row counts; a bounded uptake window always comes with its exposure anchor.
    """
    sums = {c: c for c in sorted({c for s in specs for c in (s.y_column, s.denominator) if c})}
    unit_keys = _moment_group_cols("unit_id", "group_id", by=by)

    if uptake is None:
        return _exact_totals(panel, unit_keys, sums)

    assert window_days is None or exposure is not None
    scoped = panel
    if exposure is not None:
        anchor = _scratch_name(panel, "__exposure__")
        scoped = panel.join(
            exposure.select("unit_id", nw.col("__exposure__").alias(anchor)),
            on="unit_id",
            how="left",
        )
        elapsed = _uptake_day_elapsed(scoped, anchor)
        # Uptake is only read on or after the unit's own exposure (the
        # band's left edge).
        in_band = elapsed >= 0
        if window_days is not None:
            in_band = in_band & (elapsed < window_days)
        scoped = scoped.with_columns(
            nw.when(in_band).then(nw.col(uptake)).otherwise(0.0).alias(uptake)
        )

    return _exact_totals(scoped, unit_keys, sums, maxima={uptake: uptake})


def _reduce_spec(
    windowed: nw.DataFrame[Any],
    spec: MetricSpec,
    band: tuple[int, int | None] | None,
    *,
    units_source: nw.DataFrame[Any] | None = None,
    by: Sequence[str] = (),
    day_index: str = "__day_idx__",
) -> nw.DataFrame[Any]:
    """Per-unit ``__y__`` (and ``__y_den__`` for a ratio), one row per
    surviving ``(unit_id, group_id)``.

    *windowed* is already censored and window/band-bounded. Retention and
    conversion both read "any occurrence" as ``value != 0`` (this frame
    data model has no raw event count); mean and ratio always sum.

    *units_source*, when given, is the post-censoring, pre-window-bound
    frame: every surviving unit has a row there for some day, so it is
    the correct denominator universe even when a unit's own window/band
    contains no globally-observed date. Missing units are left-joined
    back in with ``__y__``/``__y_den__`` filled to 0.0 (observed, did not
    return/convert/accrue - not "not observable", which censoring already
    decided). ``None`` is the neutral default: return *per_unit* unjoined.
    """
    unit_keys = _moment_group_cols("unit_id", "group_id", by=by)
    if spec.type == "retention":
        assert band is not None
        active = windowed.filter(nw.col(day_index) >= band[0]).with_columns(
            (nw.col(spec.y_column) != 0).cast(nw.Float64).alias("__occurred__")
        )
        per_unit = active.group_by(unit_keys).agg(nw.col("__occurred__").max().alias("__y__"))
        fill_zero = ("__y__",)
    elif spec.type == "conversion":
        per_unit = (
            windowed.with_columns(
                (nw.col(spec.y_column) != 0).cast(nw.Float64).alias("__occurred__")
            )
            .group_by(unit_keys)
            .agg(nw.col("__occurred__").max().alias("__y__"))
        )
        fill_zero = ("__y__",)
    else:
        sums = {"__y__": spec.y_column}
        fill_zero = ("__y__",)
        if spec.denominator is not None:
            sums["__y_den__"] = spec.denominator
            fill_zero = ("__y__", "__y_den__")
        per_unit = _exact_totals(windowed, unit_keys, sums)

    if units_source is None:
        return per_unit
    units = units_source.select(*unit_keys).unique()
    out = units.join(per_unit, on=unit_keys, how="left")
    return out.with_columns([nw.col(c).fill_null(0.0) for c in fill_zero])


def _windowed_moment_rows(
    panel: nw.DataFrame[Any],
    *,
    specs: Sequence[MetricSpec],
    synthesised: Sequence[Metric],
    experiment_id: str,
    exposure: nw.DataFrame[Any],
    observation_end: dt.date | dt.datetime | str | int | float | None,
    fact_max_ds: Mapping[str, Any],
    by: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Reduce each metric on its own mature exposure-relative population.
    A declared end overrides raw horizons; ratios use the earlier component.
    An entirely absent component retains the panel-extent fallback.
    """
    indexed, day_index = _with_day_index(panel, exposure)

    rows: list[dict[str, Any]] = []
    # Computed once outside the loop: every spec sharing this fallback
    # gets the same panel-wide max, so recomputing it per-spec was waste.
    panel_labels = panel.get_column("ds").unique().to_list()
    panel_order = _day_axis_label_order(panel_labels)
    panel_ds_max = max(panel_labels, key=panel_order.__getitem__, default=None)
    for spec, metric in zip(specs, synthesised, strict=True):
        final_maturity_day = _final_maturity_day(metric)
        assert final_maturity_day is not None  # every windowed/retention spec has one
        right_edge = _resolve_window_days(metric)
        band = metric.band if isinstance(metric, RetentionMetric) else None

        observable_end = _effective_observable_end(
            spec,
            panel_ds_max=panel_ds_max,
            panel_order=panel_order,
            observation_end=observation_end,
            fact_max_ds=fact_max_ds,
        )
        observable_idx = _observable_end_index(exposure, observable_end)
        kept, enrolled, dropped = _censor_units(
            indexed, final_maturity_day=final_maturity_day, observable_end_idx=observable_idx
        )
        _censoring_warning(
            spec.name, enrolled=enrolled, dropped=dropped, observation_end=observation_end
        )

        # Bound day 0..right-edge; retention's denominator uses `kept`
        # (pre-bound) so a mature unit with no rows in its band isn't dropped.
        bounded = (
            kept.filter((nw.col(day_index) >= 0) & (nw.col(day_index) < right_edge))
            if right_edge is not None
            else kept
        )
        totals = _reduce_spec(bounded, spec, band, units_source=kept, by=by, day_index=day_index)

        has_den = spec.denominator is not None
        exprs = [nw.col("__y__").alias("y")]
        if has_den:
            exprs.append(nw.col("__y_den__").alias("y_den"))
        long = totals.select(*[nw.col(name) for name in by], nw.col("group_id"), *exprs)
        long, winsor_metadata = _apply_winsorization(long, spec, key_cols=("group_id", *by))
        columns = {"y": "y"}
        if has_den:
            columns["den"] = "y_den"
        summary = emit_centered_moments(
            long,
            UNIT_GRAIN,
            keys=_moment_group_cols("group_id", by=by),
            columns=columns,
            successes_of=_binary_outcome(spec),
        )

        for record in summary:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "metric": spec.name,
                    **{name: str(record[name]) for name in by},
                    "group_id": record["group_id"],
                    **_moment_fields(record),
                    **_winsor_metadata_for_record(
                        record, winsor_metadata, key_cols=("group_id", *by)
                    ),
                }
            )
    return rows


def _conversion_total_failures(
    collapsed: nw.DataFrame[Any], specs: Sequence[MetricSpec]
) -> dict[str, _DeferredRefusal]:
    """Reject day counts where total-grain conversion requires a binary value."""
    failures: dict[str, _DeferredRefusal] = {}
    for spec in specs:
        if spec.type != "conversion":
            continue
        collapsed_max = collapsed[spec.y_column].max()
        if collapsed_max is not None and float(collapsed_max) > 1.0:
            failures[spec.name] = (
                _REFUSALS["frame.moments.metric_type_conversion"],
                {"metric": spec.name, "value": float(collapsed_max)},
            )
    return failures


def _total_moment_rows(
    panel: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    *,
    synthesised: Sequence[Metric],
    experiment_id: str,
    uptake: str | None,
    uptake_window_days: int | None,
    exposure: nw.DataFrame[Any] | None,
    observation_end: dt.date | dt.datetime | str | int | float | None,
    fact_max_ds: Mapping[str, Any],
    by: Sequence[str] = (),
    covariates: _CovariateReduction | None = None,
) -> tuple[list[dict[str, Any]], dict[str, _DeferredRefusal]]:
    """``"total"`` grain moments, split into the unwindowed fast path (one
    shared collapse + group_by) and the per-spec windowed/retention path
    (:func:`_windowed_moment_rows`) - censoring differs per metric, so a
    windowed spec cannot share the unwindowed specs' one collapsed frame.
    """
    plain: list[MetricSpec] = []
    windowed: list[MetricSpec] = []
    windowed_metrics: list[Metric] = []
    for spec, metric in zip(specs, synthesised, strict=True):
        if spec.window_days is not None or spec.type == "retention":
            windowed.append(spec)
            windowed_metrics.append(metric)
        else:
            plain.append(spec)

    rows: list[dict[str, Any]] = []
    failures: dict[str, _DeferredRefusal] = {}

    if plain:
        admitted = _admit_panel_totals(
            panel,
            exposure,
            value_columns=sorted(
                {c for spec in plain for c in (spec.y_column, spec.denominator) if c}
            ),
            observation_end=observation_end,
        )
        collapsed = _collapse_to_unit_totals(
            admitted,
            plain,
            uptake=uptake,
            window_days=uptake_window_days,
            exposure=exposure,
            by=by,
        )
        # Join caller-resolved per-unit covariates onto the collapsed totals that
        # `from_unit_summary` also uses, so one moment implementation serves both.
        covariate_aliases: dict[str, str] = {}
        covariate_means: dict[str, float | None] = {}
        imputed_names = {
            covariate.name
            for spec in plain
            for method in spec._methods
            if method.variance_reduction == "cuped"
            for covariate in method.covariates or ()
            if covariate.missing == "impute"
        }
        for name, per_unit in ((None if covariates is None else covariates.tables) or {}).items():
            covariate_aliases[name] = _scratch_name(collapsed, f"__covariate__{name}")
            if name in imputed_names:
                covariate_means[covariate_aliases[name]] = _exact_mean(
                    per_unit.get_column(name).fill_nan(None)
                )
            collapsed = collapsed.join(
                per_unit.rename({name: covariate_aliases[name]}), on="unit_id", how="left"
            )

        def remap_method(method: Any) -> Any:
            if not method.covariates:
                return method
            return method.model_copy(
                update={
                    "covariates": tuple(
                        covariate.model_copy(update={"name": covariate_aliases[covariate.name]})
                        if covariate.name in covariate_aliases
                        else covariate
                        for covariate in method.covariates
                    )
                }
            )

        moment_specs = [
            spec.model_copy(
                update={
                    "decision_method": (
                        None if spec.decision_method is None else remap_method(spec.decision_method)
                    ),
                    "sensitivity_methods": tuple(
                        remap_method(method) for method in spec.sensitivity_methods
                    ),
                }
            )
            for spec in plain
        ]
        # A conversion is any-occurrence (0/1). The unwindowed sum would
        # silently turn a multi-day converter into a day count - defer the
        # refusal until that metric is selected, while allowing valid siblings
        # to use this shared collapsed frame.
        failures.update(_conversion_total_failures(collapsed, plain))
        if failures:
            moment_specs = [spec for spec in moment_specs if spec.name not in failures]
        rows += _moment_rows(
            collapsed,
            group="group_id",
            metrics=moment_specs,
            experiment_id=experiment_id,
            uptake=uptake,
            by=by,
            covariates=_CovariateReduction(
                means=covariate_means,
                failures=failures,
                cell_failures=None if covariates is None else covariates.cell_failures,
            ),
        )
    if windowed:
        assert exposure is not None  # enforced by _validate_exposure_date
        rows += _windowed_moment_rows(
            panel,
            specs=windowed,
            synthesised=windowed_metrics,
            experiment_id=experiment_id,
            exposure=exposure,
            observation_end=observation_end,
            fact_max_ds=fact_max_ds,
            by=by,
        )
    return rows, failures


def _compliance_arm_rows_totals(
    frame: nw.DataFrame[Any],
    *,
    group: str,
    uptake: str,
    cluster: str | None,
) -> list[dict[str, Any]]:
    """Per-arm design-level compliance sufficient state off a one-row-
    per-unit frame -- independent of any metric: ``n_units``,
    ``uptake_total``, and (when *cluster* is declared) the bivariate
    centered moments of cluster uptake totals and cluster sizes, keyed by
    the field names :class:`~increment._evidence_types.ComplianceArm` takes
    directly as constructor kwargs.
    """
    unit_rows = frame.select(
        nw.col(group).alias("group_id"), nw.col(uptake).cast(nw.Float64).alias("d")
    )
    totals = unit_rows.group_by("group_id").agg(
        nw.len().alias("n_units"), nw.col("d").sum().alias("uptake_total")
    )
    rows: dict[str, dict[str, Any]] = {
        str(record["group_id"]): {
            "group_id": str(record["group_id"]),
            "n_units": int(record["n_units"]),
            "uptake_total": float(record["uptake_total"]),
        }
        for record in totals.iter_rows(named=True)
    }
    if cluster is not None:
        per_cluster_source = frame.select(
            nw.col(group).alias("group_id"),
            nw.col(cluster).alias("__cluster__"),
            nw.col(uptake).cast(nw.Float64).alias("__d__"),
        )
        per_cluster = per_cluster_source.group_by("group_id", "__cluster__").agg(
            nw.col("__d__").sum().alias("u"), nw.len().alias("m")
        )
        per_cluster = per_cluster.with_columns(nw.col("m").cast(nw.Float64))
        cluster_moments = emit_centered_moments(
            per_cluster, COMPLIANCE_CLUSTER, keys=["group_id"], columns={"uptake": "u", "size": "m"}
        )
        for record in cluster_moments:
            rows[str(record["group_id"])].update(
                {
                    name: int(record[name]) if name == "n_clusters" else float(record[name])
                    for name in COMPLIANCE_CLUSTER.names.values()
                }
            )
    return list(rows.values())


def _compliance_arm_rows_panel(
    panel: nw.DataFrame[Any],
    *,
    uptake: str,
    window_days: int | None,
    exposure: nw.DataFrame[Any] | None,
    as_of: object | None,
    completed_windows_only: bool = False,
) -> list[dict[str, Any]]:
    """Metric-independent uptake totals from sparse or dense panel rows.
    A snapshot admits units exposed by *as_of* and zeros later uptake, retaining
    exposed units with no earlier observations; completion further filters whole
    units. A snapshot needs the declared *exposure*.
    """
    scoped = panel
    if as_of is not None:
        assert exposure is not None
        labels = panel.get_column("ds").unique().to_list()
        enrollment = exposure.get_column("__exposure__").unique().to_list()
        order = _day_axis_label_order([*labels, *enrollment, as_of])
        admitted = [label for label in order if order[label] <= order[as_of]]
        if not any(order[label] <= order[as_of] for label in labels):
            return []
        scoped = panel.with_columns(
            nw.when(nw.col("ds").is_in(admitted)).then(nw.col(uptake)).otherwise(0.0).alias(uptake)
        )
        enrolled = exposure.filter(nw.col("__exposure__").is_in(admitted)).select("unit_id")
        scoped = scoped.join(enrolled, on="unit_id", how="inner")
    if completed_windows_only:
        assert exposure is not None and window_days is not None and as_of is not None
        anchors = exposure.with_columns(nw.lit(cast(Any, as_of)).alias("ds"))
        completed = anchors.filter(
            _uptake_day_elapsed(anchors, "__exposure__") >= window_days
        ).select("unit_id")
        scoped = scoped.join(completed, on="unit_id", how="inner")
    if scoped.is_empty():
        return []
    collapsed = _collapse_to_unit_totals(
        scoped,
        (),
        uptake=uptake,
        window_days=window_days,
        exposure=exposure,
    )
    totals = collapsed.group_by("group_id").agg(
        nw.len().alias("n_units"), nw.col(uptake).sum().alias("uptake_total")
    )
    return [
        {
            "group_id": str(record["group_id"]),
            "n_units": int(record["n_units"]),
            "uptake_total": float(record["uptake_total"]),
        }
        for record in totals.iter_rows(named=True)
    ]
