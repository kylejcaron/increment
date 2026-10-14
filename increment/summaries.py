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
    study_id: str | None,
) -> tuple[str, dict[str, object]]:
    from increment._analysis_config import resolve_configs
    from increment._source_identity import source_identity
    from increment.decision_wire import compiled_plan_to_json
    from increment.frame import MetricSpec, synthesise_metric
    from increment.plan import compile_decision_plan
    from increment.semantics.design import Randomized
    from increment.sources import MOMENTS_FORMAT, WINSORIZATION_MOMENT_FIELDS

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
    plan_payload = compiled_plan_to_json(plan)
    resolved_study_id = "summaries" if study_id is None else study_id
    if not isinstance(resolved_study_id, str) or not resolved_study_id:
        refuse(
            _INVALID,
            field="study_id",
            metric="<none>",
            group_id="<none>",
            value=study_id,
        )
    identity_payload = json.dumps(
        source_identity(SimpleNamespace(context=SimpleNamespace(study_id=resolved_study_id))),
        sort_keys=True,
        separators=(",", ":"),
    )
    return resolved_study_id, {
        **dict.fromkeys(WINSORIZATION_MOMENT_FIELDS),
        "moments_format": MOMENTS_FORMAT,
        "decision_plan": plan_payload,
        "source_identity": identity_payload,
    }


def _centered_summary(
    row: Mapping[str, Any], metric: str, group_id: str, metric_type: Literal["mean", "conversion"]
) -> tuple[int, int | None, float, float]:
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
        successes = int(raw_successes)
        mean = successes / count
        return count, successes, mean, mean * (count - successes)

    mean = row.get("mean")
    variance = row.get("var")
    if isinstance(mean, bool) or not isinstance(mean, Real) or not math.isfinite(float(mean)):
        refuse(_INVALID, field="mean", metric=metric, group_id=group_id, value=mean)
    if (
        isinstance(variance, bool)
        or not isinstance(variance, Real)
        or not math.isfinite(float(variance))
        or variance < 0
    ):
        refuse(_INVALID, field="var", metric=metric, group_id=group_id, value=variance)
    return count, None, float(mean), (count - 1) * float(variance)


def moments_from_summaries(
    rows: Iterable[Mapping[str, Any]],
    *,
    metrics: Mapping[str, str],
    control: str,
    study_id: str | None = None,
) -> list[dict[str, object]]:
    """Build format-12 moments rows from per-arm ``n``, mean/variance, or counts.

    Mean rows require ``n``, ``mean``, and an unbiased ``var`` (ddof=1).
    Conversion rows require ``n`` and exact integer ``successes``. Unsupported
    summary shapes should use :func:`increment.frame.from_unit_summary`.
    """

    if not metrics:
        refuse(_INVALID, field="metrics", metric="<none>", group_id="<none>", value=metrics)
    if not isinstance(control, str) or not control:
        refuse(_INVALID, field="control", metric="<none>", group_id="<none>", value=control)

    metric_types: dict[str, Literal["mean", "conversion"]] = {}
    for name, metric_type in metrics.items():
        if not isinstance(name, str) or not name:
            refuse(_INVALID, field="metric", metric=name, group_id="<none>", value=name)
        if not isinstance(metric_type, str) or metric_type not in {"mean", "conversion"}:
            refuse(_UNSUPPORTED, feature=str(metric_type), route="from_unit_summary")
        metric_types[name] = metric_type
    from increment._moment_plan import SLOTS

    resolved_study_id, wire_metadata = _wire_metadata(metric_types, control, study_id)

    summaries: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
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
        for unsupported_field in ("cluster", "cluster_id", "covariate", "inference", "sequential"):
            if row.get(unsupported_field) is not None:
                refuse(
                    _UNSUPPORTED,
                    feature=unsupported_field,
                    route="from_unit_summary",
                )
        key = (metric, group_id)
        if key in summaries:
            refuse(_INVALID, field="duplicate row", metric=metric, group_id=group_id, value=key)
        summaries[key] = row

    output: list[dict[str, object]] = []
    for (metric, group_id), row in sorted(summaries.items()):
        count, successes, mean, cy2 = _centered_summary(row, metric, group_id, metric_types[metric])
        moment_fields = dict.fromkeys(SLOTS, None)
        moment_fields.update({"ref_y": mean, "cy1": 0.0, "cy2": cy2})
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
