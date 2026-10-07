from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import ibis
import ibis.expr.types as ir

from increment._source_operations import DashboardGroupData
from increment._window import resolve_window_days
from increment.estimation.armstats import ArmStats
from increment.query._native_refusals import _refuse_operation
from increment.semantics.models import (
    ConversionMetric,
    Metric,
    QuantileMetric,
    RatioMetric,
    RetentionMetric,
)

if TYPE_CHECKING:
    from increment.sequential_state import SequentialCheckpoint, SequentialSnapshot


def _dashboard_value(value, name: str, unavailable: dict[str, str]) -> float | None:
    if value is None:
        unavailable.setdefault(name, "not applicable to this metric or retained evidence")
        return None
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result):
        unavailable[name] = "outside the finite numeric range"
        return None
    return result


def _retained_dashboard_groups(
    metrics: Sequence[Metric],
    checkpoints: Mapping[str, SequentialCheckpoint],
    snapshot: SequentialSnapshot,
) -> tuple[DashboardGroupData, ...]:
    """Project retained observations without consulting current warehouse data."""
    if set(checkpoints) != {metric.name for metric in metrics}:
        _refuse_operation(
            operation="dashboard_group_data",
            request={"metrics": tuple(metric.name for metric in metrics)},
            offered=tuple(checkpoints),
            route="supply each displayed metric's retained checkpoint",
        )
    rows: list[DashboardGroupData] = []
    for metric in metrics:
        checkpoint = checkpoints[metric.name]
        if (
            checkpoint.cell.metric != metric.name
            or checkpoint.model.observable != "outcome"
            or checkpoint.cell.estimand != "itt"
            or checkpoint.cell.segment
        ):
            _refuse_operation(
                operation="dashboard_group_data",
                request={"metric": metric.name},
                offered=(checkpoint.cell.metric,),
                route="supply this metric's unsegmented outcome checkpoint",
            )
        checkpoint.verify_snapshot(snapshot)
        transformed = (
            getattr(checkpoint.model, "adjustment", None) is not None
            or getattr(metric, "winsorization", None) is not None
        )
        window = (
            metric.band if isinstance(metric, RetentionMetric) else (0, resolve_window_days(metric))
        )
        for state in (checkpoint.control, checkpoint.treatment):
            unavailable = dict.fromkeys(
                (
                    "assigned_units",
                    "event_count",
                    "excluded_not_mature",
                    "excluded_no_observed_day",
                    "excluded_other",
                    "observation_end",
                ),
                "historical accounting was not retained in this checkpoint",
            )
            observed = analysis_input = total = numerator = denominator = None
            retained_units = None
            if state.n:
                if state.law == "bernoulli":
                    assert state.successes is not None
                    retained_units = total = state.successes
                    observed = Fraction(state.successes, state.n)
                else:
                    first = state.mean[0]
                    is_ratio = state.law in ("ratio_mean", "adjusted_ratio_mean", "gaussian_ratio")
                    value = (
                        first / state.mean[1]
                        if is_ratio and state.mean[1]
                        else (None if is_ratio else first)
                    )
                    if transformed:
                        analysis_input = value
                        for name in ("observed_value", "sum_value", "numerator", "denominator"):
                            unavailable[name] = "pre-transform outcomes were not retained"
                    else:
                        observed, total = value, first * state.n
                        if is_ratio:
                            numerator, denominator = total, state.mean[1] * state.n
                            if state.mean[1] == 0:
                                unavailable["observed_value"] = "retained denominator is zero"
                        if isinstance(metric, (ConversionMetric, RetentionMetric)):
                            retained_units = int(total)
            else:
                for name in (
                    "observed_value",
                    "analysis_input_value",
                    "sum_value",
                    "numerator",
                    "denominator",
                ):
                    unavailable[name] = "no eligible units in the retained checkpoint"
                if isinstance(metric, (ConversionMetric, RetentionMetric)):
                    retained_units = 0
            if retained_units is None:
                unavailable["retained_units"] = "this metric is not a binary unit outcome"
            if window[1] is None:
                unavailable["window_end_days"] = "the metric has no fixed end day"
            rows.append(
                DashboardGroupData(
                    metric=metric.name,
                    group_id=state.group_id,
                    assigned_units=None,
                    eligible_units=state.n,
                    observed_value=_dashboard_value(observed, "observed_value", unavailable),
                    analysis_input_value=_dashboard_value(
                        analysis_input, "analysis_input_value", unavailable
                    ),
                    sum_value=_dashboard_value(total, "sum_value", unavailable),
                    event_count=None,
                    retained_units=retained_units,
                    numerator=_dashboard_value(numerator, "numerator", unavailable),
                    denominator=_dashboard_value(denominator, "denominator", unavailable),
                    excluded_not_mature=None,
                    excluded_no_observed_day=None,
                    excluded_other=None,
                    observation_end=None,
                    window_start_days=window[0],
                    window_end_days=window[1],
                    source_kind="retained_checkpoint",
                    prefix_id=checkpoint.prefix_id,
                    unavailable=unavailable,
                )
            )
    return tuple(rows)


def _dashboard_quantiles(totals: ir.Table, quantile: float) -> ir.Table:
    """Reduce linear sample quantiles using portable ordered ranks."""
    observed = totals.select("group_id", "y")
    observed = observed.filter(observed.y.notnull())
    ranked = observed.mutate(
        _rank=ibis.row_number().over(ibis.window(group_by=observed.group_id, order_by=observed.y)),
        _size=observed.y.count().over(ibis.window(group_by=observed.group_id)),
    )
    position = (ranked._size - 1) * quantile
    bounds = ranked.group_by("group_id").aggregate(
        lower=ranked.y.max(where=ranked._rank == position.floor()),
        upper=ranked.y.max(where=ranked._rank == position.ceil()),
        weight=(position - position.floor()).max(),
    )
    return bounds.select(
        "group_id",
        quantile_value=(1 - bounds.weight) * bounds.lower + bounds.weight * bounds.upper,
    )


def _observed_dashboard_group(
    metric: Metric,
    record: Mapping[str, Any],
    window: tuple[int | None, int | None],
) -> DashboardGroupData:
    """Decode one bounded warehouse aggregate into observed group evidence."""
    eligible = int(record["n"] or 0)
    assigned, no_observed, not_mature = (
        int(record[name]) for name in ("assigned", "no_observed", "not_mature")
    )
    unavailable: dict[str, str] = {}
    observed = total = numerator = denominator = None
    retained_units = 0 if isinstance(metric, (ConversionMetric, RetentionMetric)) else None
    if retained_units is None:
        unavailable["retained_units"] = "this metric is not a binary unit outcome"
    if window[1] is None:
        unavailable["window_end_days"] = "the metric has no fixed end day"
    if record["observation_end"] is None:
        unavailable["observation_end"] = "the running source has no observed metric days"
    unavailable["prefix_id"] = "pinned warehouse evidence is not a retained checkpoint"
    if eligible:
        arm = ArmStats.model_validate(
            {
                **{name: record[name] for name in ArmStats.model_fields if name in record},
                "study_id": record["experiment_id"],
            }
        )
        mean = arm.mean_y() * record["scale_y"]
        total = mean * eligible
        if isinstance(metric, RatioMetric):
            den_mean = arm.mean_den() * record["scale_y_den"]
            numerator, denominator = total, den_mean * eligible
            observed = mean / den_mean if den_mean else None
            if den_mean == 0:
                unavailable["observed_value"] = "group denominator is zero"
        elif isinstance(metric, QuantileMetric):
            observed = record["quantile_value"]
        else:
            observed = mean
            if retained_units is not None:
                retained_units = round(total)
    else:
        unavailable["observed_value"] = "no eligible observed units"
        unavailable["sum_value"] = "no eligible observed units"
    return DashboardGroupData(
        metric=metric.name,
        group_id=str(record["group_id"]),
        assigned_units=assigned,
        eligible_units=eligible,
        observed_value=_dashboard_value(observed, "observed_value", unavailable),
        analysis_input_value=_dashboard_value(None, "analysis_input_value", unavailable),
        sum_value=_dashboard_value(total, "sum_value", unavailable),
        event_count=int(record["event_count"] or 0),
        retained_units=retained_units,
        numerator=_dashboard_value(numerator, "numerator", unavailable),
        denominator=_dashboard_value(denominator, "denominator", unavailable),
        excluded_not_mature=not_mature,
        excluded_no_observed_day=no_observed,
        excluded_other=assigned - eligible - no_observed - not_mature,
        observation_end=record["observation_end"],
        window_start_days=window[0],
        window_end_days=window[1],
        source_kind="pinned_warehouse",
        prefix_id=None,
        unavailable=unavailable,
    )
