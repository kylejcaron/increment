"""Reduce retained measures and independently frozen forecasts over the randomized draws."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from functools import cache
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, NoReturn

import narwhals as nw
import numpy as np

from increment._frame_validation import (
    _is_missing,
    _missing_count,
    eager_frame,
    require_eager_frame,
)
from increment.estimation.contrast import (
    SwitchbackMetricStats,
    UnavailableContrastStats,
    _exact_sum,
)
from increment.semantics.assignment import FixedCountOrder, SwitchbackAssignment
from increment.semantics.models import CupedCovariateBinding

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from increment._metric_specs import MetricSpec


type _Reject = Callable[..., NoReturn]


def forecast_column(spec: MetricSpec, reject: _Reject) -> str | None:
    method = spec.decision_method
    if method is None or method.variance_reduction != "cuped":
        return None
    if spec.type != "mean":
        reject(
            message=f"CUPED on this switchback requires a mean, not {spec.type!r}.",
            metric=spec.name,
            field="decision_method",
            reason="cuped_requires_mean",
        )
    if (
        method.name != "cuped"
        or method.conversion_inference != "auto"
        or method.propensity_learner is not None
        or method.outcome_learner is not None
        or method.folds is not None
        or method.procedure is not None
    ):
        reject(
            message="Declare the switchback adjustment with Method.cuped(covariates='column').",
            metric=spec.name,
            field="decision_method",
            reason="ignored_metric_option",
        )
    covariates = method.covariates
    if (
        not covariates
        or len(covariates) != 1
        or covariates[0].kind != "continuous"
        or covariates[0].missing != "refuse"
        or isinstance(covariates[0], CupedCovariateBinding)
    ):
        reject(
            message="Declare one complete numeric forecast column for switchback CUPED.",
            metric=spec.name,
            field="covariates",
            reason="invalid_forecast_declaration",
        )
    return covariates[0].name


def _numeric_values(series: nw.Series[Any]) -> bool:
    if series.dtype.is_numeric():
        return True
    return series.dtype == nw.Object and all(
        isinstance(value, Real | Decimal) and not isinstance(value, bool)
        for value in series.to_list()
    )


def validate_metric_columns(
    frame: nw.DataFrame[Any],
    specs: Sequence[MetricSpec],
    *,
    reject_numeric: _Reject,
    reject_missingness: _Reject,
) -> None:
    """Validate both raw ratio columns before computing any period sums."""
    checked: set[str] = set()
    for spec in specs:
        for column in (spec.y_column, spec.denominator):
            if column is None or column in checked:
                continue
            checked.add(column)
            missing = _missing_count(frame, column)
            if missing:
                reject_missingness(
                    message=f"Metric {spec.name!r} has missing values in {column!r}.",
                    metric=spec.name,
                    column=column,
                    missing_values=missing,
                    reason="null_metric",
                )
            if not _numeric_values(frame[column]):
                reject_numeric(
                    message=f"Metric {spec.name!r} requires a numeric column, not {column!r} of type {frame.schema[column]}.",
                    metric=spec.name,
                    column=column,
                    reason="nonnumeric_metric",
                )
            if not frame.schema[column].is_integer():
                bad = [v for v in frame[column].to_list() if not _finite(v)]
                if bad:
                    reject_numeric(
                        message=f"Metric {spec.name!r} needs finite values in {column!r}.",
                        metric=spec.name,
                        column=column,
                        invalid_values=bad[:5],
                        reason="nonfinite_numeric",
                    )


def _finite(value: Any) -> bool:
    if isinstance(value, Integral | Fraction):
        return True
    if isinstance(value, Decimal):
        return value.is_finite()
    return math.isfinite(value)


def _forecast_clock(frame: nw.DataFrame[Any], reject: _Reject) -> None:
    before, after = frame.schema["forecast_at"], frame.schema["assigned_at"]
    forecast, assignment = nw.col("forecast_at"), nw.col("assigned_at")
    complete = len(frame) > 0 and not (
        frame["forecast_at"].null_count() or frame["assigned_at"].null_count()
    )
    if complete and before == nw.Date and after == nw.Date:
        if frame.select(forecast.max() < assignment.min()).item():
            return
    elif (
        complete
        and isinstance(before, nw.Datetime)
        and isinstance(after, nw.Datetime)
        and before.time_zone == after.time_zone
    ):
        # Scale native ticks as Python integers: int64 nanoseconds can overflow,
        # and comparing in the coarser unit can erase a strict gap.
        latest, first = frame.select(
            forecast.cast(nw.Int64).max(), assignment.cast(nw.Int64).min()
        ).row(0)
        nanos = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}
        if int(latest) * nanos[before.time_unit] < int(first) * nanos[after.time_unit]:
            return
    reject(
        message="All forecast timestamps must strictly precede the first assignment on the same clock.",
        reason="forecast_not_pre_assignment",
    )


@dataclass(frozen=True, slots=True)
class _MissingForecast:
    count: int
    first_key: tuple[str, int, int]


def _forecast_values(
    table: nw.DataFrame[Any],
    column: str,
    keys: tuple[str, str, str],
    ordered: list[int],
    shape: tuple[int, int, int],
    reject: _Reject,
) -> np.ndarray | _MissingForecast:
    series = table[column]
    missing = table.select(_is_missing(table, column))[column]
    count = int(missing.sum())
    observed = series.filter(~missing) if count else series
    if len(observed) and not _numeric_values(observed):
        reject(
            message=f"Forecast column {column!r} must be numeric.",
            column=column,
            reason="nonnumeric_forecast",
        )
    values = observed.to_list()
    if not all(map(_finite, values)):
        reject(
            message=f"Forecast column {column!r} must be finite.",
            column=column,
            reason="nonfinite_forecast",
        )
    if count:
        u, c, p = min(table.select(*keys).filter(missing).iter_rows())
        return _MissingForecast(count, (u, int(c), int(p)))
    return np.asarray([values[i] for i in ordered], dtype=object).reshape(shape)


def _forecast_data(
    frame: IntoDataFrame | None,
    forecasts: Mapping[str, str],
    *,
    keys: tuple[str, str, str],
    units: Sequence[object],
    cycles: Sequence[int],
    periods: int,
    reject: _Reject,
) -> dict[str, np.ndarray | _MissingForecast]:
    if not forecasts:
        return {}
    if frame is None:
        reject(
            message="Provide pre_assignment_forecasts for the declared switchback CUPED column.",
            reason="pre_assignment_forecasts_required",
            metric=next(iter(forecasts)),
            field="pre_assignment_forecasts",
        )
    require_eager_frame(frame, constructor="pre_assignment_forecasts")
    table = eager_frame(frame)
    columns = tuple(dict.fromkeys(forecasts.values()))
    clock = ("forecast_at", "assigned_at")
    required = (*keys, *clock, *columns)
    if len(set(required)) != len(required) or not set(required) <= set(table.columns):
        reject(
            message="Provide distinct roster keys, forecast_at, assigned_at and the declared forecast columns.",
            required=required,
            reason="forecast_columns",
        )
    _forecast_clock(table, reject)
    shape = (len(units), len(cycles), periods)
    if len(table) != math.prod(shape):
        reject(
            message="Supply one forecast per scheduled unit, cycle and period.",
            reason="forecast_keys",
        )
    u_index = {v: i for i, v in enumerate(units)}
    c_index = {v: i for i, v in enumerate(cycles)}
    ordered: list[int] = [-1] * len(table)
    for row, (u, c, p) in enumerate(table.select(*keys).iter_rows()):
        if (
            not isinstance(u, str)
            or u not in u_index
            or isinstance(c, bool)
            or not isinstance(c, Integral)
            or c not in c_index
            or isinstance(p, bool)
            or not isinstance(p, Integral)
            or not 0 <= p < periods
        ):
            reject(
                message="A forecast key lies outside the scheduled roster or period range.",
                unit=u,
                cycle=c,
                period=p,
                reason="forecast_keys",
            )
        i = (u_index[u] * len(cycles) + c_index[c]) * periods + int(p)
        if ordered[i] != -1:
            reject(
                message="A scheduled period has two forecasts.",
                unit=u,
                cycle=c,
                period=p,
                reason="forecast_keys",
            )
        ordered[i] = row
    return {col: _forecast_values(table, col, keys, ordered, shape, reject) for col in columns}


def _arm_averages(
    values: np.ndarray, treated: np.ndarray, assignment: SwitchbackAssignment
) -> list[tuple[Fraction, Fraction]]:
    """Exact treated and control levels per unit (averaged cycles) or block (averaged roster)."""
    units, cycles, k = treated.shape
    sequence = assignment.sequence
    shared = sequence.independence_unit == "shared_block"
    divisor = units if shared else cycles
    totals = [[Fraction(0), Fraction(0)] for _ in range(cycles if shared else units)]
    if isinstance(sequence, FixedCountOrder):
        weights = (
            Fraction(1, (k - sequence.treated_periods) * divisor),
            Fraction(1, sequence.treated_periods * divisor),
        )
        by_order = (weights, weights)
    else:
        p = Fraction(sequence.probability_ct)
        # Within CT (or TC), both observed arms have the same positional inclusion probability.
        ct, tc = 1 / (2 * p * divisor), 1 / (2 * (1 - p) * divisor)
        by_order = ((ct, ct), (tc, tc))
    for u in range(units):
        for c in range(cycles):
            arms = treated[u, c]
            accum = totals[c if shared else u]
            weights = by_order[int(arms[0])]
            period_values = values[u, c].tolist()
            for arm in (0, 1):
                subtotal = _exact_sum(
                    v.as_integer_ratio()
                    for v, z in zip(period_values, arms, strict=True)
                    if z == arm
                )
                accum[arm] += weights[arm] * subtotal
    return [(t, c) for c, t in totals]


def reduce_joint_metrics(
    specs: Sequence[MetricSpec],
    *,
    assignment: SwitchbackAssignment,
    totals: Mapping[str, np.ndarray],
    treated: np.ndarray,
    forecast_columns: Mapping[str, str],
    forecast_frame: IntoDataFrame | None,
    keys: tuple[str, str, str],
    units: Sequence[object],
    cycles: Sequence[int],
    metadata: Mapping[str, Any],
    reject: _Reject,
) -> dict[str, SwitchbackMetricStats | UnavailableContrastStats]:
    forecasts = _forecast_data(
        forecast_frame,
        forecast_columns,
        keys=keys,
        units=units,
        cycles=cycles,
        periods=assignment.periods_per_cycle,
        reject=reject,
    )

    @cache
    def arm_levels(column: str) -> list[tuple[Fraction, Fraction]]:
        return _arm_averages(totals[column].reshape(treated.shape), treated, assignment)

    @cache
    def signed_forecast(column: str) -> tuple[Fraction, ...] | _MissingForecast:
        data = forecasts[column]
        if isinstance(data, _MissingForecast):
            return data
        return tuple(t - c for t, c in _arm_averages(data, treated, assignment))

    results: dict[str, SwitchbackMetricStats | UnavailableContrastStats] = {}
    for spec in specs:
        if spec.type == "ratio":
            assert spec.denominator is not None
            results[spec.name] = SwitchbackMetricStats(
                metric=spec.name,
                aggregation="ratio",
                **metadata,
                ratio_components=tuple(
                    (yt, dt, yc, dc)
                    for (yt, yc), (dt, dc) in zip(
                        arm_levels(spec.y_column), arm_levels(spec.denominator), strict=True
                    )
                ),
            )
        else:
            col = forecast_columns[spec.name]
            predictor = signed_forecast(col)
            if isinstance(predictor, _MissingForecast):
                results[spec.name] = UnavailableContrastStats(
                    metric=spec.name,
                    aggregation="sum",
                    forecast_column=col,
                    failure_code="source.frame.switchback.metric",
                    failure_context={
                        "metric": spec.name,
                        "column": col,
                        "reason": "missing_forecast",
                        "missing_count": predictor.count,
                        "first_missing_key": predictor.first_key,
                    },
                    **metadata,
                )
                continue
            results[spec.name] = SwitchbackMetricStats(
                metric=spec.name,
                aggregation="sum",
                forecast_column=col,
                cuped_outcomes=tuple(t - c for t, c in arm_levels(spec.y_column)),
                cuped_predictors=predictor,
                **metadata,
            )
    return results
