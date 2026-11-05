"""Build portable moments rows from per-arm summary statistics."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, NoReturn

from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, refuse

if TYPE_CHECKING:
    import numpy as np

_UNSUPPORTED = RefusalSpec(
    "moments.summaries.unsupported",
    CapabilityError,
    template="{feature} cannot be represented by per-arm mean/conversion summaries; use {route}.",
)


def _render_invalid(
    *,
    field: str,
    metric: object,
    group_id: object,
    value: object,
    reason: str,
    route: str,
    expected: tuple[str, ...] | None = None,
) -> str:
    accepted = "" if expected is None else f"; accepted: {', '.join(expected)}"
    return (
        f"invalid summary {field} for metric {metric!r}, group {group_id!r}: {value!r} "
        f"({reason.replace('_', ' ')}{accepted}). {route}"
    )


_INVALID = RefusalSpec("moments.summaries.invalid", InvalidRequestError, _render_invalid)
_ROW_FIELDS: Mapping[str, tuple[str, ...]] = {
    "mean": ("group_id", "mean", "metric", "n", "var"),
    "conversion": ("group_id", "metric", "n", "successes"),
}
_ROW_FIELD_ROUTES: Mapping[str, str] = {
    "mean": (
        "Rename or remove the other fields; var is the ddof=1 sample variance, "
        "the square of the standard deviation."
    ),
    "conversion": (
        "Rename or remove the other fields; a conversion row carries its integer successes count."
    ),
}


def _invalid(
    reason: str,
    route: str,
    *,
    field: str,
    value: object,
    metric: object = "<none>",
    group_id: object = "<none>",
    expected: tuple[str, ...] | None = None,
) -> NoReturn:
    """Refuse one summary input, naming why (``reason``) and how to fix it (``route``)."""
    extra = {} if expected is None else {"expected": expected}
    refuse(
        _INVALID,
        field=field,
        metric=metric,
        group_id=group_id,
        value=value,
        reason=reason,
        route=route,
        **extra,
    )


def _wire_metadata(
    metric_types: Mapping[str, Literal["mean", "conversion"]],
    control: str,
    study_id: str,
    assignment_counts: Mapping[str, int] | None,
) -> dict[str, object]:
    from increment._analysis_config import resolve_configs
    from increment._source_identity import source_identity
    from increment.decision_wire import compiled_plan_to_json
    from increment.frame import MetricSpec, synthesise_metric
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.sources import (
        ASSIGNMENT_COUNTS_FIELD,
        MOMENTS_FORMAT,
        WINSORIZATION_MOMENT_FIELDS,
    )

    specs = {
        name: MetricSpec(name=name, type=metric_type) for name, metric_type in metric_types.items()
    }
    catalog = [synthesise_metric(spec) for spec in specs.values()]
    configs = resolve_configs(catalog, bindings=None, specs=specs, methods=None, prior=None)
    plan = compile_decision_plan(
        None,
        catalog,
        path="frame",
        design=Randomized(control_group=control),
        configs=configs,
    )
    metadata: dict[str, object] = {
        **dict.fromkeys(WINSORIZATION_MOMENT_FIELDS),
        "moments_format": MOMENTS_FORMAT,
        "decision_plan": compiled_plan_to_json(plan),
        "source_identity": json.dumps(
            source_identity(SimpleNamespace(context=SimpleNamespace(study_id=study_id))),
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    if assignment_counts is not None:
        metadata[ASSIGNMENT_COUNTS_FIELD] = json.dumps(
            dict(assignment_counts), sort_keys=True, separators=(",", ":")
        )
    return metadata


def _finite_summary_value(value: Any, field: str, metric: str, group_id: str) -> float:
    reason = None
    converted = math.nan
    if value is None:
        reason = "missing_field"
    elif isinstance(value, bool) or not isinstance(value, Real):
        reason = "not_a_number"
    else:
        with contextlib.suppress(OverflowError, ValueError):
            converted = float(value)
        if not math.isfinite(converted):
            reason = "not_finite"
    if reason is not None:
        _invalid(
            reason,
            f"Give {field} as a finite number.",
            field=field,
            metric=metric,
            group_id=group_id,
            value=value,
        )
    return converted


def _centered_summary(
    row: Mapping[str, Any],
    metric: str,
    group_id: str,
    metric_type: Literal["mean", "conversion"],
) -> tuple[int, int | None, float, float, float]:
    n = row.get("n")
    n_route = (
        "Give n as an integer count of at least 2 units; a variance needs two."
        if metric_type == "mean"
        else "Give n as a positive integer count of units."
    )
    if n is None:
        _invalid("missing_field", n_route, field="n", metric=metric, group_id=group_id, value=n)
    if isinstance(n, bool) or not isinstance(n, Integral):
        _invalid("not_integer", n_route, field="n", metric=metric, group_id=group_id, value=n)
    if n < 1:
        _invalid(
            "below_two" if metric_type == "mean" else "not_positive",
            n_route,
            field="n",
            metric=metric,
            group_id=group_id,
            value=n,
        )
    count = int(n)
    if metric_type == "conversion":
        raw_successes = row.get("successes")
        if (
            isinstance(raw_successes, bool)
            or not isinstance(raw_successes, Integral)
            or raw_successes < 0
            or raw_successes > count
        ):
            if raw_successes is None:
                reason = "missing_field"
            elif isinstance(raw_successes, bool) or not isinstance(raw_successes, Integral):
                reason = "not_integer"
            else:
                reason = "negative" if raw_successes < 0 else "exceeds_n"
            _invalid(
                reason,
                "Give successes as the integer number of converting units, from 0 to n.",
                field="successes",
                metric=metric,
                group_id=group_id,
                value=raw_successes,
            )
        from increment._bernoulli_moments import canonical_bernoulli_family

        successes = int(raw_successes)
        mean, cy1, cy2 = canonical_bernoulli_family(count, successes)
        return count, successes, mean, cy1, cy2

    if count == 1:
        _invalid("below_two", n_route, field="n", metric=metric, group_id=group_id, value=count)
    mean = _finite_summary_value(row.get("mean"), "mean", metric, group_id)
    variance = _finite_summary_value(row.get("var"), "var", metric, group_id)
    if variance < 0:
        _invalid(
            "negative",
            "var is a variance, the square of the standard deviation, so it cannot be negative.",
            field="var",
            metric=metric,
            group_id=group_id,
            value=variance,
        )
    cy2 = (count - 1) * variance
    if not math.isfinite(cy2):
        _invalid(
            "too_large",
            "var times (n - 1) exceeds the float range; rescale the metric, such as to thousands.",
            field="var",
            metric=metric,
            group_id=group_id,
            value=variance,
        )
    return count, None, mean, 0.0, cy2


def _metric_types(
    metrics: Mapping[str, str],
) -> dict[str, Literal["mean", "conversion"]]:
    metric_types: dict[str, Literal["mean", "conversion"]] = {}
    unsupported_types = {"ratio", "covariate", "cluster", "sequential"}
    type_route = "Declare each metric as 'mean' or 'conversion'."
    for name, metric_type in metrics.items():
        if not isinstance(name, str) or not name:
            _invalid(
                "not_a_name",
                "Name each metric with a non-empty string.",
                field="metric",
                metric=name,
                value=name,
            )
        if not isinstance(metric_type, str):
            _invalid(
                "unknown_type",
                type_route,
                field="metric type",
                metric=name,
                value=metric_type,
                expected=("conversion", "mean"),
            )
        if metric_type not in {"mean", "conversion"}:
            if metric_type in unsupported_types:
                refuse(_UNSUPPORTED, feature=metric_type, route="from_unit_summary")
            _invalid(
                "unknown_type",
                type_route,
                field="metric type",
                metric=name,
                value=metric_type,
                expected=("conversion", "mean"),
            )
        metric_types[name] = metric_type
    return metric_types


def _validated_assignment_counts(
    assignment_counts: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if assignment_counts is None:
        return None
    route = "Give assignment_counts as a mapping from group_id to its count of assigned units."
    if not isinstance(assignment_counts, Mapping):
        _invalid("not_a_mapping", route, field="assignment_counts", value=assignment_counts)
    validated: dict[str, int] = {}
    for group_id, count in assignment_counts.items():
        if not isinstance(group_id, str) or not group_id:
            reason = "not_a_name"
        elif isinstance(count, bool) or not isinstance(count, Integral):
            reason = "not_integer"
        elif count < 0:
            reason = "negative"
        else:
            validated[group_id] = int(count)
            continue
        _invalid(reason, route, field="assignment_counts", group_id=group_id, value=count)
    return validated


def _summary_rows(
    rows: Iterable[Mapping[str, Any]],
    metric_types: Mapping[str, Literal["mean", "conversion"]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    summaries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            _invalid(
                "not_a_mapping",
                "Give one mapping per metric and arm, with metric, group_id and n keys.",
                field="row",
                value=row,
            )
        metric = row.get("metric")
        group_id = row.get("group_id")
        if not isinstance(metric, str) or metric not in metric_types:
            _invalid(
                "unknown_metric",
                "Declare every metric the rows name in metrics=, or remove its rows.",
                field="metric/group_id",
                metric=metric,
                group_id=group_id,
                value=(metric, group_id),
                expected=tuple(sorted(metric_types)),
            )
        if not isinstance(group_id, str) or not group_id:
            _invalid(
                "not_a_name",
                "Give each row its arm as a non-empty string group_id.",
                field="metric/group_id",
                metric=metric,
                group_id=group_id,
                value=(metric, group_id),
            )
        metric_type = metric_types[metric]
        unexpected = set(row) - set(_ROW_FIELDS[metric_type])
        if unexpected:
            _invalid(
                "unknown_field",
                _ROW_FIELD_ROUTES[metric_type],
                field="row fields",
                metric=metric,
                group_id=group_id,
                value=tuple(sorted(map(str, unexpected))),
                expected=_ROW_FIELDS[metric_type],
            )
        key = (metric, group_id)
        if key in summaries:
            _invalid(
                "duplicate_row",
                "Give one row per metric and group_id.",
                field="duplicate row",
                metric=metric,
                group_id=group_id,
                value=key,
            )
        summaries[key] = row
    return summaries


def _validate_summary_roster(
    metric_types: Mapping[str, Literal["mean", "conversion"]],
    summaries: Mapping[tuple[str, str], Mapping[str, Any]],
    control: str,
    assignment_counts: Mapping[str, int] | None,
) -> None:
    groups_by_metric: dict[str, set[str]] = {metric: set() for metric in metric_types}
    for metric, group_id in summaries:
        groups_by_metric[metric].add(group_id)
    for metric, groups in groups_by_metric.items():
        if not groups:
            _invalid(
                "missing_rows",
                "Add rows for this metric, or remove it from metrics=.",
                field="metric rows",
                metric=metric,
                value=None,
            )
        if control not in groups:
            _invalid(
                "missing_control_row",
                "Add a row for the control arm, or pass control= the group_id its rows use.",
                field="control arm",
                metric=metric,
                group_id=control,
                value=tuple(sorted(groups)),
            )
    if assignment_counts is not None:
        row_groups = {group_id for _, group_id in summaries}
        missing_groups = row_groups - assignment_counts.keys()
        if missing_groups:
            _invalid(
                "missing_assignment_count",
                "Give assignment_counts an entry for every group_id in the rows.",
                field="assignment_counts arms",
                value=tuple(sorted(missing_groups)),
            )


def moments_from_summaries(
    rows: Iterable[Mapping[str, str | int | float | Real | np.integer | np.floating]],
    *,
    metrics: Mapping[str, str],
    control: str,
    study_id: str | None = None,
    assignment_counts: Mapping[str, int] | None = None,
) -> list[dict[str, object]]:
    """Build format-12 rows from per-arm mean/variance or conversion summaries.

    Mean rows require ``n >= 2``, ``mean``, and unbiased ``var`` (ddof=1).
    Conversion rows require ``n`` and exact integer ``successes``. Optional
    source-level ``assignment_counts`` enable ``unit_counts()`` and allocation
    health checks; without them those operations refuse with
    ``source.moments.assignment_counts``. Ratio and other richer summary
    shapes should use :func:`increment.frame.from_unit_summary`.
    """

    if not isinstance(metrics, Mapping) or not metrics:
        _invalid(
            "empty" if isinstance(metrics, Mapping) else "not_a_mapping",
            "Give metrics as a mapping from metric name to 'mean' or 'conversion', "
            "such as {'revenue': 'mean'}.",
            field="metrics",
            value=metrics,
        )
    if not isinstance(control, str) or not control:
        _invalid(
            "not_a_name",
            "Give control as the control arm's group_id, a non-empty string.",
            field="control",
            value=control,
        )
    resolved_study_id = "summaries" if study_id is None else study_id
    if not isinstance(resolved_study_id, str) or not resolved_study_id:
        _invalid(
            "not_a_name",
            "Give study_id as a non-empty string, or omit it.",
            field="study_id",
            value=study_id,
        )

    metric_types = _metric_types(metrics)
    validated_counts = _validated_assignment_counts(assignment_counts)
    summaries = _summary_rows(rows, metric_types)
    _validate_summary_roster(metric_types, summaries, control, validated_counts)

    from increment._moment_plan import SLOTS

    wire_metadata = _wire_metadata(metric_types, control, resolved_study_id, validated_counts)
    output: list[dict[str, object]] = []
    for (metric, group_id), row in sorted(summaries.items()):
        count, successes, mean, cy1, cy2 = _centered_summary(
            row, metric, group_id, metric_types[metric]
        )
        moment_fields = dict.fromkeys(SLOTS, None)
        moment_fields.update({"ref_y": mean, "cy1": cy1, "cy2": cy2})
        output.append(
            {
                "experiment_id": resolved_study_id,
                "metric": metric,
                "group_id": group_id,
                "n": count,
                "successes": successes,
                **moment_fields,
                "x_role": None,
                **wire_metadata,
            }
        )
    return output
