"""Explicit missing-value repairs for dataframe entry paths.

`from_unit_summary`/`from_unit_panel` refuse null or NaN metric values by
default, since either can silently produce a wrong number (zero-averaged
or backend-inconsistent sums) with no warning. This module is the helper
tier of the named fixes for callers holding the dataframe; warehouse users
instead declare `MetricSpec(missing=...)`, which needs no source mutation.

Helpers are narwhals-based and backend-agnostic, returning the same native
frame type that came in, and treat null/NaN identically as "missing" since
NaN is a real value (not skipped by sums) on polars/pyarrow but not
pandas. Each helper returns `(frame, affected_count)`, so a repair is
always visible and loggable.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw

from increment.errors import (
    IncrementRuntimeWarning,
    InvalidRequestError,
    WarningSpec,
    raiser,
    refusals,
    warn,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.typing import IntoDataFrame

__all__ = ["drop_null", "pooled_mean", "zeros"]


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "impute.impute_name_least": "impute.{helper}: name at least one column to repair",
        "impute.impute_column_found": "impute.{helper}: column(s) {missing!r} not found in frame; available columns: {present}",
        "impute.impute_pooled_mean": "impute.pooled_mean: column {col!r} has no observed values to compute a mean from -- every row is null/NaN. Repair the column upstream or drop it.",
        "impute.impute_pooled_mean_not_finite": "impute.pooled_mean: column {col!r} has {observed} observed value(s) whose mean is {mean!r}, not a finite number -- the values are non-finite or large enough to overflow when averaged, so there is no usable fill value. Repair the column upstream.",
        "impute.impute_roles_columns": "impute.pooled_mean: roles name column(s) {columns!r} that are not among the columns to repair {requested!r}; declare a role only for a column you pass.",
        "impute.impute_roles_value": "impute.pooled_mean: roles for column(s) {columns!r} must be 'covariate' or 'outcome'.",
        "impute.pooled_mean_outcome": "impute.pooled_mean: column(s) {columns!r} are declared outcomes, and filling an outcome with its pooled mean shrinks variance and can bias the estimate. Route: {route}",
    },
)
_raise = raiser(_REFUSALS)

_WARNINGS: dict[str, WarningSpec] = {}

_POOLED_MEAN_ROLE_UNDECLARED = _WARNINGS["impute.pooled_mean_role_undeclared"] = WarningSpec(
    "impute.pooled_mean_role_undeclared",
    IncrementRuntimeWarning,
    lambda *, columns, affected_counts, route: (
        f"impute.pooled_mean filled {affected_counts!r} value(s) in column(s) {list(columns)!r} "
        f"without a declared role. Pooled-mean fill is valid for pre-period covariates only. "
        f"Route: {route}"
    ),
)
_OUTCOME_ROUTE = (
    "declare MetricSpec(missing='zero') if null means no events, MetricSpec(missing='drop') "
    "if not observed, or repair upstream with increment.impute.zeros / drop_null"
)
_UNDECLARED_ROUTE = (
    "pass roles={column: 'covariate'} for a pre-period covariate; an outcome column "
    "must use impute.zeros, impute.drop_null or MetricSpec(missing=...)"
)


def _validate_columns(frame: nw.DataFrame[Any], cols: tuple[str, ...], helper: str) -> None:
    if not cols:
        _raise("impute.impute_name_least", helper=helper)
    present = set(frame.columns)
    missing = [c for c in cols if c not in present]
    if missing:
        _raise(
            "impute.impute_column_found", helper=helper, missing=missing, present=sorted(present)
        )


def _missing_expr(frame: nw.DataFrame[Any], col: str) -> nw.Expr:
    """Null-or-NaN as one predicate: ``is_nan`` only types on float columns."""
    expr = nw.col(col).is_null()
    if frame.schema[col] in (nw.Float32, nw.Float64):
        expr = expr | nw.col(col).is_nan()
    return expr


def _affected_rows(frame: nw.DataFrame[Any], cols: tuple[str, ...]) -> int:
    """Rows with at least one missing value across *cols*."""
    predicate = _missing_expr(frame, cols[0])
    for col in cols[1:]:
        predicate = predicate | _missing_expr(frame, col)
    return int(frame.select(predicate.cast(nw.Int64).sum()).item())


def _missing_count(frame: nw.DataFrame[Any], col: str) -> int:
    return int(frame.select(_missing_expr(frame, col).cast(nw.Int64).sum()).item())


def _validated_roles(
    roles: Mapping[str, Literal["covariate", "outcome"]] | None, cols: tuple[str, ...]
) -> dict[str, str]:
    declared = dict(roles or {})
    stray = tuple(col for col in declared if col not in cols)
    if stray:
        _raise("impute.impute_roles_columns", columns=stray, requested=cols)
    invalid = tuple(col for col, role in declared.items() if role not in ("covariate", "outcome"))
    if invalid:
        _raise("impute.impute_roles_value", columns=invalid)
    return declared


def zeros[FrameT: "IntoDataFrame"](frame: FrameT, *cols: str) -> tuple[FrameT, int]:
    """Replace null/NaN with 0 in `cols`: the "null means no events" repair.

    Use when a missing value genuinely records absence of activity (e.g. a
    revenue column left null for units that never purchased). If null means
    "not observed" instead, zero-filling biases the mean toward zero - use
    `drop_null` or repair upstream. Returns `(frame, affected_count)`.
    """
    nwf = nw.from_native(frame, eager_only=True)
    _validate_columns(nwf, cols, "zeros")
    affected = _affected_rows(nwf, cols)
    if affected:
        exprs = []
        for col in cols:
            expr = nw.col(col)
            if nwf.schema[col] in (nw.Float32, nw.Float64):
                expr = expr.fill_nan(None)
            exprs.append(expr.fill_null(0).alias(col))
        nwf = nwf.with_columns(*exprs)
    return nwf.to_native(), affected


def pooled_mean[FrameT: "IntoDataFrame"](
    frame: FrameT,
    *cols: str,
    roles: Mapping[str, Literal["covariate", "outcome"]] | None = None,
) -> tuple[FrameT, int]:
    """Replace null/NaN in each of `cols` with that column's mean over the
    observed values: deterministic mean imputation.

    Safe for a randomized design's pre-period covariate (e.g. a CUPED
    baseline): the mean is computed from the pooled column only, never
    from the arm or outcome, so randomization keeps the treatment effect
    unbiased - the only cost is that imputed units add no variance
    reduction. Not safe for an observational adjustment covariate, where
    imputing a confounder without its missingness indicator hides residual
    confounding; use `AdjustmentSet(missing="impute-indicator")` instead.

    Integer columns are promoted to Float64. A column with no observed
    values raises. Returns `(frame, affected_count)`.

    `roles` declares what each column is: `"covariate"` or `"outcome"`. A
    column declared `"outcome"` raises `impute.pooled_mean_outcome` (an
    `InvalidRequestError`) before any fill; the route is `zeros`,
    `drop_null` or `MetricSpec(missing=...)`. Filling a column with no
    declared role still succeeds but emits the coded
    `impute.pooled_mean_role_undeclared` warning when any value was
    filled. Keys must be among `cols` (`impute.impute_roles_columns`) and
    values one of the two literals (`impute.impute_roles_value`).
    """
    nwf = nw.from_native(frame, eager_only=True)
    _validate_columns(nwf, cols, "pooled_mean")
    declared = _validated_roles(roles, cols)
    outcomes = tuple(col for col in cols if declared.get(col) == "outcome")
    if outcomes:
        _raise("impute.pooled_mean_outcome", columns=outcomes, route=_OUTCOME_ROUTE)
    affected = _affected_rows(nwf, cols)
    if affected:
        exprs = []
        for col in cols:
            expr = nw.col(col).cast(nw.Float64).fill_nan(None)
            # Count observed values rather than inferring emptiness from the
            # mean: an all-null column averages to None on some backends and to
            # NaN on others, and a NaN mean is separately reachable from
            # observed values that are non-finite or overflow when averaged.
            observed = int(nwf.select(expr.count().alias("c")).item())
            if observed == 0:
                _raise("impute.impute_pooled_mean", col=col)
            mean = nwf.select(expr.mean().alias("m")).item()
            if mean is None or not math.isfinite(mean):
                _raise(
                    "impute.impute_pooled_mean_not_finite", col=col, observed=observed, mean=mean
                )
            exprs.append(expr.fill_null(float(mean)).alias(col))
        undeclared_counts = {
            col: count
            for col in cols
            if col not in declared and (count := _missing_count(nwf, col))
        }
        nwf = nwf.with_columns(*exprs)
        if undeclared_counts:
            warn(
                _POOLED_MEAN_ROLE_UNDECLARED,
                context={
                    "columns": tuple(undeclared_counts),
                    "affected_counts": undeclared_counts,
                    "route": _UNDECLARED_ROUTE,
                },
            )
    return nwf.to_native(), affected


def drop_null[FrameT: "IntoDataFrame"](frame: FrameT, *cols: str) -> tuple[FrameT, int]:
    """Drop every row with a null/NaN in any of `cols`: explicit complete-case
    analysis, with the dropped-row count returned.

    Unbiased only when missingness is unrelated to the values (MCAR); if
    units missing a value differ systematically from those that don't,
    dropping them changes the answer. Prefer `zeros` when null means
    "no events". Returns `(frame, dropped_count)`.
    """
    nwf = nw.from_native(frame, eager_only=True)
    _validate_columns(nwf, cols, "drop_null")
    predicate = _missing_expr(nwf, cols[0])
    for col in cols[1:]:
        predicate = predicate | _missing_expr(nwf, col)
    dropped = int(nwf.select(predicate.cast(nw.Int64).sum()).item())
    if dropped:
        nwf = nwf.filter(~predicate)
    return nwf.to_native(), dropped
