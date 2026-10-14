"""Build portable moments rows from per-arm summary statistics."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any, Literal

from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec, refuse

_UNSUPPORTED = RefusalSpec(
    "moments.summaries.unsupported",
    CapabilityError,
    template="{feature} cannot be represented by per-arm mean/conversion summaries; use {route}.",
)
_INVALID = RefusalSpec(
    "moments.summaries.invalid",
    InvalidRequestError,
    template="invalid summary statistic {field!r} for metric {metric!r}, group {group_id!r}: {value!r}",
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
    if isinstance(value, bool) or not isinstance(value, Real):
        refuse(_INVALID, field=field, metric=metric, group_id=group_id, value=value)
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        refuse(_INVALID, field=field, metric=metric, group_id=group_id, value=value)
    if not math.isfinite(converted):
        refuse(_INVALID, field=field, metric=metric, group_id=group_id, value=value)
    return converted


def _centered_summary(
    row: Mapping[str, Any],
    metric: str,
    group_id: str,
    metric_type: Literal["mean", "conversion"],
) -> tuple[int, int | None, float, float, float]:
    n = row.get("n")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        refuse(_INVALID, field="n", metric=metric, group_id=group_id, value=n)
    count = int(n)
    if metric_type == "conversion":
        raw_successes = row.get("successes")
        if (
            isinstance(raw_successes, bool)
            or not isinstance(raw_successes, Integral)
            or raw_successes < 0
            or raw_successes > count
        ):
            refuse(
                _INVALID,
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
        refuse(_INVALID, field="n", metric=metric, group_id=group_id, value=count)
    mean = _finite_summary_value(row.get("mean"), "mean", metric, group_id)
    variance = _finite_summary_value(row.get("var"), "var", metric, group_id)
    if variance < 0:
        refuse(_INVALID, field="var", metric=metric, group_id=group_id, value=variance)
    cy2 = (count - 1) * variance
    if not math.isfinite(cy2):
        refuse(_INVALID, field="cy2", metric=metric, group_id=group_id, value=cy2)
    return count, None, mean, 0.0, cy2


def _metric_types(
    metrics: Mapping[str, str],
) -> dict[str, Literal["mean", "conversion"]]:
    metric_types: dict[str, Literal["mean", "conversion"]] = {}
    unsupported_types = {"ratio", "covariate", "cluster", "sequential"}
    for name, metric_type in metrics.items():
        if not isinstance(name, str) or not name:
            refuse(_INVALID, field="metric", metric=name, group_id="<none>", value=name)
        if not isinstance(metric_type, str):
            refuse(
                _INVALID,
                field="metric type",
                metric=name,
                group_id="<none>",
                value=metric_type,
            )
        if metric_type not in {"mean", "conversion"}:
            if metric_type in unsupported_types:
                refuse(_UNSUPPORTED, feature=metric_type, route="from_unit_summary")
            refuse(
                _INVALID,
                field="metric type",
                metric=name,
                group_id="<none>",
                value=metric_type,
            )
        metric_types[name] = metric_type
    return metric_types


def _validated_assignment_counts(
    assignment_counts: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if assignment_counts is None:
        return None
    if not isinstance(assignment_counts, Mapping):
        refuse(
            _INVALID,
            field="assignment_counts",
            metric="<none>",
            group_id="<none>",
            value=assignment_counts,
        )
    validated: dict[str, int] = {}
    for group_id, count in assignment_counts.items():
        if (
            not isinstance(group_id, str)
            or not group_id
            or isinstance(count, bool)
            or not isinstance(count, Integral)
            or count < 0
        ):
            refuse(
                _INVALID,
                field="assignment_counts",
                metric="<none>",
                group_id=group_id,
                value=count,
            )
        validated[group_id] = int(count)
    return validated


def _summary_rows(
    rows: Iterable[Mapping[str, Any]],
    metric_types: Mapping[str, Literal["mean", "conversion"]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    summaries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            refuse(_INVALID, field="row", metric="<none>", group_id="<none>", value=row)
        metric = row.get("metric")
        group_id = row.get("group_id")
        if (
            not isinstance(metric, str)
            or metric not in metric_types
            or not isinstance(group_id, str)
            or not group_id
        ):
            refuse(
                _INVALID,
                field="metric/group_id",
                metric=metric,
                group_id=group_id,
                value=(metric, group_id),
            )
        expected_fields = {"metric", "group_id", "n"}
        expected_fields.update({"mean", "var"} if metric_types[metric] == "mean" else {"successes"})
        unexpected = set(row) - expected_fields
        if unexpected:
            refuse(
                _INVALID,
                field="row fields",
                metric=metric,
                group_id=group_id,
                value=tuple(sorted(map(str, unexpected))),
            )
        key = (metric, group_id)
        if key in summaries:
            refuse(_INVALID, field="duplicate row", metric=metric, group_id=group_id, value=key)
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
            refuse(
                _INVALID,
                field="metric rows",
                metric=metric,
                group_id="<none>",
                value=None,
            )
        if control not in groups:
            refuse(
                _INVALID,
                field="control arm",
                metric=metric,
                group_id=control,
                value=tuple(sorted(groups)),
            )
    if assignment_counts is not None:
        row_groups = {group_id for _, group_id in summaries}
        missing_groups = row_groups - assignment_counts.keys()
        if missing_groups:
            refuse(
                _INVALID,
                field="assignment_counts arms",
                metric="<none>",
                group_id="<none>",
                value=tuple(sorted(missing_groups)),
            )


def moments_from_summaries(
    rows: Iterable[Mapping[str, Any]],
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
        refuse(_INVALID, field="metrics", metric="<none>", group_id="<none>", value=metrics)
    if not isinstance(control, str) or not control:
        refuse(_INVALID, field="control", metric="<none>", group_id="<none>", value=control)
    resolved_study_id = "summaries" if study_id is None else study_id
    if not isinstance(resolved_study_id, str) or not resolved_study_id:
        refuse(
            _INVALID,
            field="study_id",
            metric="<none>",
            group_id="<none>",
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
