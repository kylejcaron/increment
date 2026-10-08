"""`Analysis`'s day-axis quartet (run_daily/run_daily_lift/run_asof/run_asof_lift)
extracted behind one evidence-acquisition seam and one options seam.

Unsegmented encouragement compliance uses the source's declared uptake
cohort; outcome rows retain their own matching first stages.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from increment import readouts
from increment._analysis_config import _Unset, normalize_display_correction
from increment._frame_panel import _day_axis_label_order
from increment._lift_options import LiftOptions
from increment._readout_request import ReadoutRequest
from increment._readout_request import _raise as _raise_readout
from increment._source_types import ComplianceSummary, classify_source, compliance_summary_series
from increment.breakout.estimates import (
    DEFAULT_RELIABILITY_FLOOR,
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
    DayAxisView,
    _asof_monitoring_note,
    _slice_reference_kind,
    reject_quantile_metrics,
    reject_retention_under_encouragement,
    reject_winsorized_day_axis,
    resolve_daily_cell_policy,
)
from increment.breakout.estimates import run_daily as _run_daily_values
from increment.errors import (
    CapabilityError,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    refuse,
)
from increment.estimation.encouragement import ESTIMANDS, estimate_compliance
from increment.estimation.engine import (
    UNBOUNDED_RETENTION_DAILY_REMEDY,
    _validate_methods,
    reject_completed_windows_on_unbounded_retention,
    reject_unbounded_retention,
)
from increment.estimation.inference import validate_readout_inference
from increment.estimation.readout_types import CellFailure, CellKey
from increment.query.native_contract import NativeViewSource
from increment.readouts._common import (
    _sequential_inference,
    _validate_encouragement_asof_inference,
)
from increment.readouts._daily import _validate_daily_inference
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import RetentionMetric
from increment.sources import MomentSource, require_operation

if TYPE_CHECKING:
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.engine import Method
    from increment.estimation.inference import Prior
    from increment.query.native_contract import DayEvidenceSource
    from increment.semantics.models import Experiment, Metric

DayAxisGrain = Literal["daily", "asof"]

_NO_DEFINITIONS = RefusalSpec(
    "facade.analysis.no_definitions",
    CapabilityError,
    template="{method}() needs a native Analysis.from_definitions or Analysis.from_unit_panel instance -- this source has no panel day axis.",
)
_CLUSTERED_DAY_AXIS = RefusalSpec(
    "facade.analysis.clustered_day_axis",
    CapabilityError,
    template="{method}() is not supported on a clustered experiment -- experiment {experiment!r} declares cluster {cluster!r}, and cluster-robust inference is total-grain only (run()/srm()).",
)
_TRIGGER_UNSUPPORTED = RefusalSpec(
    "facade.analysis.trigger_unsupported",
    CapabilityError,
    template="{method}: experiment {experiment!r} declares trigger {trigger!r}, which this readout does not apply. Use run() for the triggered readout, or analyze an experiment with no declared trigger.",
)
_UNKNOWN_DIMENSION = RefusalSpec(
    "facade.analysis.unknown_dimension",
    InvalidRequestError,
    template="{method}: dimension {dimension!r} is not a declared breakout -- declared breakout properties: {declared!r}",
)
_UNDECLARED_METRIC_FOR_DIMENSION = RefusalSpec(
    "facade.analysis.undeclared_metric_for_dimension",
    InvalidRequestError,
    template="{method}: dimension={dimension!r} only supports metrics already declared on this experiment -- metrics= may exclude declared metrics but not add new ones for the dimensioned path. Not declared: {undeclared!r}",
)
_ENCOURAGEMENT_DAILY_LATE = RefusalSpec(
    "facade.analysis.encouragement_daily_late",
    InvalidRequestError,
    template="run_daily_lift: per-day incremental LATE is statistically meaningless under a weak daily first stage; use run_asof_lift for the as-of trend instead",
    keys=frozenset({"experiment"}),
)
_OBSERVATIONAL_DAY_AXIS = RefusalSpec(
    "facade.analysis.observational_day_axis",
    UnsupportedRequestError,
    template="{method} is not supported for an observational design: confounded day-axis contrasts are refused, not silently emitted",
)
_BH_SEGMENTED_ASOF = RefusalSpec(
    "facade.analysis.bh_segmented_asof",
    UnsupportedRequestError,
    template="{method}: BH multiplicity is supported for breakout families, not segmented as-of series; use bonferroni or none",
)
_TRIGGERED_COMPLIANCE_UNSUPPORTED = RefusalSpec(
    "facade.analysis.triggered_compliance_unsupported",
    UnsupportedRequestError,
    template='{method} does not support triggered compliance estimates; use population="assigned".',
    keys=frozenset({"method"}),
)


def _validate_estimands(
    estimands: Sequence[str] | None,
    design: Randomized | Encouragement | Observational | None,
) -> None:
    if estimands is None:
        return
    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        _raise_readout("readout.estimands.unknown", unknown=unknown, supported=ESTIMANDS)
    if tuple(estimands) != ("itt",) and not isinstance(design, Encouragement):
        mechanism = getattr(design, "mechanism", "randomized")
        _raise_readout(
            "readout.assignment.estimands", estimands=list(estimands), mechanism=mechanism
        )


def _resolve_lift_estimands(
    estimands: Sequence[str] | None,
    design: Randomized | Encouragement | Observational | None,
) -> tuple[bool, tuple[str, ...], Encouragement | None]:
    """Reuses run()'s own `readout.estimands.*` codes (verified live: run()'s
    inline duplicate already raised these before this function's bare
    ValueErrors could fire), so run() and run_asof_lift agree by construction.
    """
    is_encouragement = isinstance(design, Encouragement)
    _validate_estimands(estimands, design)
    requested = tuple(
        estimands if estimands is not None else (ESTIMANDS if is_encouragement else ("itt",))
    )
    return is_encouragement, requested, design if is_encouragement else None


def _day_axis_source_route(src: MomentSource) -> Literal["artifact", "moments", "native"]:
    route = classify_source(src)
    return "moments" if route == "panel" else route


def partition_by_view(
    metrics: Sequence[Metric], grain: DayAxisGrain
) -> list[tuple[DayAxisView, tuple[Metric, ...]]]:
    if grain == "asof":
        return [("asof", tuple(metrics))] if metrics else []
    activity = tuple(m for m in metrics if not isinstance(m, RetentionMetric))
    retention = tuple(m for m in metrics if isinstance(m, RetentionMetric))
    groups: list[tuple[DayAxisView, tuple[Metric, ...]]] = []
    if activity:
        groups.append(("daily", activity))
    if retention:
        groups.append(("cohort", retention))
    return groups


@dataclass(frozen=True, slots=True)
class DayAxisRequest:
    caller: Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"]
    grain: DayAxisGrain
    metrics: tuple[Metric, ...]
    dimension: str | None
    completed_windows_only: bool = False
    estimands: tuple[str, ...] | None = None
    population: Literal["assigned", "triggered"] = "assigned"


def _refuse_triggered_compliance(
    *, population: str, design: Any, estimands: Sequence[str], method: str
) -> None:
    if population == "triggered" and design is not None and "compliance" in estimands:
        refuse(_TRIGGERED_COMPLIANCE_UNSUPPORTED, method=method)


def _compliance_readouts(
    req: DayAxisRequest,
    design: Any,
    estimands: Sequence[str],
    extra: dict[str, Any],
    source: MomentSource,
) -> tuple[bool, dict[Any, ComplianceSummary]]:
    if design is None or req.grain != "asof" or req.dimension is not None:
        return False, {}
    extra["estimands"] = tuple(name for name in estimands if name != "compliance")
    if "compliance" not in estimands:
        return True, {}
    return True, compliance_summary_series(
        source, design, completed_windows_only=req.completed_windows_only
    )


@dataclass(frozen=True, slots=True)
class EvidenceSlice:
    rows: list[dict[str, Any]]
    metrics: tuple[Metric, ...]
    view: DayAxisView
    dimension: str | None
    source_name: str | None


# The arm-family guard (`Analysis._require_arm_state`) stays in the facade:
# every day-axis method must call it first, before delegating here.
def validate_day_axis(
    req: DayAxisRequest,
    *,
    src: MomentSource,
    design: Randomized | Encouragement | Observational | None,
    plan: CompiledDecisionPlan,
    route: Literal["artifact", "moments", "native"],
    experiment: Experiment | None,
) -> None:
    """Validate one day-axis request across native, moments, and artifact routes.

    Branches apply only where routes genuinely differ. If multiple conditions
    fail, callers must not depend on the order in which those refusals surface.
    """
    if req.population == "triggered" and (experiment is None or experiment.trigger is None):
        refuse(
            _TRIGGER_UNSUPPORTED,
            method=req.caller,
            experiment="<unknown>" if experiment is None else experiment.name,
            trigger=None if experiment is None else experiment.trigger,
        )
    if (
        route in ("artifact", "native")
        and experiment is not None
        and experiment.cluster is not None
    ):
        refuse(
            _CLUSTERED_DAY_AXIS,
            method=req.caller,
            experiment=experiment.name,
            cluster=experiment.cluster,
        )
    checkpoint_asof = (
        req.grain == "asof" and getattr(plan.inference, "registration", None) is not None
    )
    if route in ("moments", "artifact") and src.shape != "unit_panel" and not checkpoint_asof:
        refuse(_NO_DEFINITIONS, method=req.caller)
    if route == "native" and (
        "day_source" not in src.operations or not isinstance(src, NativeViewSource)
    ):
        refuse(_NO_DEFINITIONS, method=req.caller)
    reject_winsorized_day_axis(list(req.metrics), req.caller)
    if route == "native":
        # Native routes reject quantiles; moments/artifacts retain the broader metric set.
        moments_reason = "per-day moments" if req.grain == "daily" else "as-of moments"
        reject_quantile_metrics(
            list(req.metrics),
            f"{req.caller}()",
            reason=f"quantiles do not decompose into {moments_reason}",
        )
    if req.grain == "daily":
        # Route-independent: dimensioned artifact reads reach the source through
        # `breakout_moments`, bypassing the readout gate, so the facade enforces it too.
        reject_unbounded_retention(
            list(req.metrics),
            req.caller,
            remedy=UNBOUNDED_RETENTION_DAILY_REMEDY,
            supported_view="asof",
        )
    # A registered checkpoint's compliance-only request reads uptake and no outcome
    # moments, so retention metrics in the catalog are not part of what it consumes.
    reads_outcomes = not (
        checkpoint_asof and req.estimands is not None and set(req.estimands) == {"compliance"}
    )
    if reads_outcomes and req.caller == "run_asof_lift" and isinstance(design, Encouragement):
        reject_retention_under_encouragement(list(req.metrics), req.caller)
    if reads_outcomes and req.grain == "asof" and req.completed_windows_only:
        reject_completed_windows_on_unbounded_retention(list(req.metrics), req.caller)
    if req.caller == "run_daily_lift" and isinstance(design, Encouragement):
        refuse(_ENCOURAGEMENT_DAILY_LATE, experiment=getattr(experiment, "name", None))
    if (
        req.caller in ("run_daily_lift", "run_asof_lift")
        and getattr(design, "mechanism", None) == "observational"
    ):
        refuse(_OBSERVATIONAL_DAY_AXIS, method=req.caller)
    if req.caller == "run_asof_lift":
        policy = plan.view_policies.for_view(
            "asof",
            mechanism=getattr(design, "mechanism", None),
            segmented=req.dimension is not None,
        )
        correction = normalize_display_correction(policy.correction)
        if correction == "bh":
            refuse(_BH_SEGMENTED_ASOF, method=req.caller)
    if req.dimension is None:
        return
    if route == "moments":
        if req.dimension not in src.breakouts:
            refuse(
                _UNKNOWN_DIMENSION,
                method=req.caller,
                dimension=req.dimension,
                declared=list(src.breakouts),
            )
        return
    declared_breakouts = list(experiment.breakouts) if experiment is not None else []
    matching = [b for b in declared_breakouts if b.property == req.dimension]
    if not matching:
        refuse(
            _UNKNOWN_DIMENSION,
            method=req.caller,
            dimension=req.dimension,
            declared=sorted({b.property for b in declared_breakouts}),
        )
    if (
        route == "native"
        and req.caller in ("run_daily", "run_daily_lift")
        and experiment is not None
    ):
        declared_names = {m.name for m in src.context.metrics}
        undeclared = [m.name for m in req.metrics if m.name not in declared_names]
        if undeclared:
            refuse(
                _UNDECLARED_METRIC_FOR_DIMENSION,
                method=req.caller,
                dimension=req.dimension,
                undeclared=undeclared,
            )


def _asof_kwargs(
    metric: Metric, req: DayAxisRequest, needs_covariate: Callable[[Metric], bool]
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"grain": req.grain}
    if req.grain == "asof":
        kwargs["completed_windows_only"] = req.completed_windows_only
    if needs_covariate(metric):
        kwargs["include_covariate"] = True
    return kwargs


def _metric_view_groups(req: DayAxisRequest) -> list[tuple[DayAxisView, tuple[Metric, ...]]]:
    if req.caller in ("run_daily", "run_asof"):
        # Value rows are independent; preserve the declared metric order.
        return [
            (
                "asof"
                if req.grain == "asof"
                else ("cohort" if isinstance(metric, RetentionMetric) else "daily"),
                (metric,),
            )
            for metric in req.metrics
        ]
    return partition_by_view(req.metrics, req.grain)


def _stamp_triggered_cell_failures(
    results, raw_rows, req: DayAxisRequest, *, control_group: str | None = None
):
    if req.population != "triggered":
        return results
    failures: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in raw_rows:
        if int(raw.get("n", 0) or 0) >= 2:
            continue
        raw_day = raw.get("ds")
        day = raw_day.date() if isinstance(raw_day, dt.datetime) else raw_day
        if not isinstance(day, dt.date):
            continue
        segment = None if req.dimension is None else str(raw.get(req.dimension))
        key = (day, str(raw.get("metric")), str(raw.get("group_id")), segment)
        failures[key] = {"day": day.isoformat(), "observed_units": int(raw.get("n", 0) or 0)}
    stamped = []
    for row in results:
        if getattr(row, "failure_code", None) is not None:
            stamped.append(row)
            continue
        cell = CellKey.from_row(row)
        if cell.method is not None and getattr(row, "sampling_available", None) is True:
            stamped.append(row)
            continue
        segment = row.dimension_value if req.dimension is not None else None
        failure_context = failures.get((row.ds, row.metric, row.group_id, segment))
        if (
            failure_context is None
            and cell.method is not None
            and getattr(row, "unavailable", None) == "no_control_arm"
            and control_group is not None
        ):
            failure_context = failures.get((row.ds, row.metric, control_group, segment))
        if failure_context is None:
            stamped.append(row)
            continue
        if cell.method is None:
            if failure_context["observed_units"] > 0:
                stamped.append(row)
                continue
            stamped.append(
                row.model_copy(
                    update={
                        "decision_scope_complete": False,
                        "decision_scope_reason_code": "readout.scope.decision_incomplete",
                        "decision_scope_reason_context": {"reason": "triggered_cell_unavailable"},
                    }
                )
            )
            continue
        failure_code = (
            "readout.cell.missing_arm"
            if failure_context["observed_units"] == 0
            else "readout.cell.missing_metric_observations"
        )
        failure = CellFailure(
            hypothesis=cell.hypothesis(),
            code=failure_code,
            context=failure_context,
        )
        updates = {
            "failure_code": failure.code,
            "failure_context": failure.context,
            "decision_scope_complete": False,
            "decision_scope_reason_code": "readout.scope.decision_incomplete",
            "decision_scope_reason_context": {"reason": "triggered_cell_unavailable"},
            "sampling_available": False,
            "sampling_reason_code": failure.code,
            "sampling_reason_context": failure.context,
        }
        stamped.append(row.model_copy(update=updates))
    if any(getattr(row, "decision_scope_complete", None) is False for row in stamped):
        stamped = [
            row.model_copy(
                update={
                    "decision_scope_complete": False,
                    "decision_scope_reason_code": "readout.scope.decision_incomplete",
                    "decision_scope_reason_context": {"reason": "triggered_cell_unavailable"},
                }
            )
            if hasattr(row, "decision_scope_complete")
            else row
            for row in stamped
        ]
    return stamped


def _missing_triggered_lift_rows(
    rows,
    results,
    groups,
    req,
    kwargs,
    source,
    control_group,
    requested_estimands,
    *,
    only_empty: bool = False,
    policy_by_cell=None,
    ds_basis: Literal["calendar", "cohort"] = "calendar",
):
    from increment._immutable import _FrozenMapping
    from increment.breakout.estimates import _nan_lift_rows

    if groups is None:
        groups = sorted({str(raw["group_id"]) for raw in rows})

    present = {(row.ds, row.metric, row.group_id, row.dimension_value) for row in results}
    observed: dict[tuple[Any, ...], int] = {}
    for raw in rows:
        day_value = raw.get("ds")
        day = day_value.date() if isinstance(day_value, dt.datetime) else day_value
        if isinstance(day, dt.date):
            observed[
                (
                    day,
                    str(raw.get("metric")),
                    str(raw.get("group_id")),
                    None if req.dimension is None else str(raw.get(req.dimension)),
                )
            ] = int(raw.get("n", 0) or 0)
    additions = []
    for raw in rows:
        raw_day = raw.get("ds")
        day = raw_day.date() if isinstance(raw_day, dt.datetime) else raw_day
        if not isinstance(day, dt.date):
            continue
        metric = str(raw.get("metric"))
        segment = None if req.dimension is None else str(raw.get(req.dimension))
        for group in groups:
            observed_units = observed.get((day, metric, group, segment), 0)
            if only_empty:
                if observed_units > 0:
                    continue
            elif (group == control_group and observed_units > 0) or observed_units >= 2:
                continue
            key = (day, metric, group, segment)
            if key in present:
                continue
            code = (
                "readout.cell.missing_arm"
                if observed_units == 0
                else "readout.cell.missing_metric_observations"
            )
            unavailable_rows = _nan_lift_rows(
                {(metric, group): "few_units"},
                ds_value=day,
                methods=None,
                ds_basis=ds_basis,
                dimension=req.dimension,
                dim_value=segment,
                source=source,
                inference=kwargs.get("inference") or "fixed",
                alternative=kwargs.get("alternative") or "two-sided",
                methods_by_metric=kwargs.get("methods_by_metric"),
                method_roles_by_metric=kwargs.get("method_roles_by_metric"),
                policy_by_metric=(policy_by_cell or {}).get((day, metric, segment))
                or kwargs.get("policy_by_metric"),
            )
            candidates = [row for row in unavailable_rows if row.estimand in requested_estimands]
            if "late" in requested_estimands and group != control_group:
                candidates.extend(
                    row.model_copy(update={"estimand": "late", "value_scale": scale})
                    for row in unavailable_rows
                    if row.estimand == "itt"
                    for scale in ("absolute", "relative")
                )
            for row in candidates:
                context = _FrozenMapping(
                    {
                        "day": day.isoformat(),
                        "group_id": group,
                        "reason": "missing_arm" if observed_units == 0 else "few_units",
                        "observed_units": observed_units,
                    }
                )
                additions.append(
                    row.model_copy(
                        update={
                            "analysis_population": "triggered",
                            "failure_code": code,
                            "failure_context": _FrozenMapping(dict(context)),
                            "sampling_available": False,
                            "sampling_reason_code": code,
                            "sampling_reason_context": _FrozenMapping(dict(context)),
                            "decision_scope_complete": False,
                            "decision_scope_reason_code": "readout.scope.decision_incomplete",
                            "decision_scope_reason_context": _FrozenMapping(dict(context)),
                            "n_control": observed.get((day, metric, control_group, segment), 0),
                            "n_treat": observed.get((day, metric, group, segment), 0),
                        }
                    )
                )
                present.add(key)
    return additions


def _prepare_triggered_lift_slice(
    rows, evidence_slice, req, kwargs, control_group, requested_estimands, plan, design
):
    metric_by_name = {metric.name: metric for metric in evidence_slice.metrics}
    segment_roster_by_metric = {
        name: sorted(
            {
                str(row[req.dimension])
                for row in rows
                if req.dimension is not None and row.get("metric") == name
            }
        )
        for name in metric_by_name
    }
    segment_counts = {
        name: max(len(values), 1) for name, values in segment_roster_by_metric.items()
    }
    rows_by_cell: dict[tuple[dt.date, str, str | None], list[dict[str, Any]]] = {}
    for raw in rows:
        raw_day = raw.get("ds")
        day = raw_day.date() if isinstance(raw_day, dt.datetime) else raw_day
        if not isinstance(day, dt.date):
            continue
        metric_name = str(raw.get("metric"))
        segment = None if req.dimension is None else str(raw.get(req.dimension))
        rows_by_cell.setdefault((day, metric_name, segment), []).append(raw)

    policy_by_cell = {}
    for cell_key, metric_rows in rows_by_cell.items():
        metric_name = cell_key[1]
        policy_by_cell[cell_key] = {
            metric_name: resolve_daily_cell_policy(
                metric_by_name[metric_name],
                metric_rows,
                control_group=control_group,
                alpha=kwargs.get("alpha", getattr(plan, "alpha", 0.05)),
                alternative=kwargs.get("alternative", "two-sided"),
                design=design,
                view=evidence_slice.view,
                plan=plan,
                segment_count_by_metric=segment_counts,
            )
        }
    ds_basis = "cohort" if evidence_slice.view == "cohort" else "calendar"
    unavailable = _missing_triggered_lift_rows(
        rows,
        [],
        None,
        req,
        kwargs,
        evidence_slice.source_name,
        control_group,
        requested_estimands,
        only_empty=True,
        policy_by_cell=policy_by_cell,
        ds_basis=ds_basis,
    )
    estimation_rows = [row for row in rows if int(row.get("n", 0) or 0) > 0]
    return unavailable, estimation_rows, segment_roster_by_metric, policy_by_cell, ds_basis


def _estimate_day_axis_lift_slice(
    daily_lift,
    raw_rows,
    evidence_slice,
    req,
    kwargs,
    extra,
    control_group,
    requested_estimands,
    plan,
    design,
):
    triggered = req.population == "triggered"
    if triggered:
        (
            unavailable,
            estimation_rows,
            segment_roster_by_metric,
            policy_by_cell,
            ds_basis,
        ) = _prepare_triggered_lift_slice(
            raw_rows,
            evidence_slice,
            req,
            {**kwargs, **extra},
            control_group,
            requested_estimands,
            plan,
            design,
        )
    else:
        unavailable = []
        estimation_rows = raw_rows
        segment_roster_by_metric = None
        policy_by_cell = None
        ds_basis = "cohort" if evidence_slice.view == "cohort" else "calendar"
    results = list(unavailable)
    if estimation_rows:
        results.extend(
            daily_lift(
                summary=estimation_rows,
                metrics=list(evidence_slice.metrics),
                control_group=control_group,
                dimension=evidence_slice.dimension,
                source=evidence_slice.source_name,
                view=evidence_slice.view,
                segment_roster_by_metric=segment_roster_by_metric,
                **kwargs,
                **extra,
            )
        )
    if triggered:
        results.extend(
            _missing_triggered_lift_rows(
                raw_rows,
                results,
                sorted({str(row["group_id"]) for row in raw_rows}),
                req,
                {**kwargs, **extra},
                evidence_slice.source_name,
                control_group,
                requested_estimands,
                policy_by_cell=policy_by_cell,
                ds_basis=ds_basis,
            )
        )
    return results


def _unsupported_triggered_sequential_rows(
    rows: Sequence[DailyLiftEstimate],
) -> list[DailyLiftEstimate]:
    from increment._immutable import _FrozenMapping

    output = []
    for row in rows:
        if row.method_role != "decision":
            continue
        context = {"reason": "triggered_sequential", "analysis_population": "triggered"}
        output.append(
            row.model_copy(
                update={
                    "analysis_population": "triggered",
                    "lift": None,
                    "relative_confidence_set": None,
                    "sequential_result": None,
                    "discovery": None,
                    "family_axes": None,
                    "family_q": None,
                    "family_threshold": None,
                    "family_id": None,
                    "failure_code": "readout.cell.unsupported_request",
                    "failure_context": _FrozenMapping(context),
                    "sampling_available": False,
                    "sampling_reason_code": "readout.cell.unsupported_request",
                    "sampling_reason_context": _FrozenMapping(context),
                    "decision_scope_complete": False,
                    "decision_scope_reason_code": "readout.scope.decision_incomplete",
                    "decision_scope_reason_context": _FrozenMapping(context),
                    "n_control": None,
                    "n_treat": None,
                }
            )
        )
    return output


def _triggered_group_roster(source: MomentSource, experiment: Experiment) -> list[str]:
    assigned_counts = getattr(source, "assignment_counts", None)
    if callable(assigned_counts):
        groups = set(assigned_counts(population="assigned"))
    else:
        unit_counts = getattr(source, "unit_counts", None)
        if not callable(unit_counts):
            return []
        groups = set(unit_counts())
    allocation = getattr(
        getattr(getattr(source, "context", None), "design", None), "allocation", None
    )
    if allocation is not None:
        groups.update(allocation)
    groups.add(experiment.control_group)
    from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

    return sorted(groups - {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL})


def _assigned_segment_roster(
    source: MomentSource,
    dimension: str,
    experiment: Experiment,
    source_name: str | None,
) -> list[str]:
    assigned_values = getattr(source, "assigned_breakout_values", None)
    if not callable(assigned_values):
        return []
    values: set[str] = set()
    for breakout in experiment.breakouts:
        if breakout.property == dimension:
            values.update(
                str(value) for value in assigned_values(breakout, source_name=source_name)
            )
    return sorted(values)


def _fill_triggered_early_look_rows(
    rows: list[dict[str, Any]],
    *,
    req: DayAxisRequest,
    metrics: Sequence[Metric],
    experiment: Experiment | None,
    source: MomentSource,
    edge_source: Any,
    source_name: str | None,
) -> list[dict[str, Any]]:
    """Keep zero-population arms/days visible in triggered day-axis evidence."""
    if req.population != "triggered" or experiment is None:
        return rows
    groups = _triggered_group_roster(source, experiment)
    if not groups:
        return rows
    metric_by_name = {metric.name: metric for metric in metrics}
    templates: dict[tuple[str, str | None], dict[str, Any]] = {}
    present: set[tuple[dt.date, str, str, str | None]] = set()
    observation_edges = {
        metric.name: edge_source.triggered_observation_edges(metric) for metric in metrics
    }
    ends: dict[str, dt.date] = {}
    for row in rows:
        metric = str(row["metric"])
        if metric not in metric_by_name:
            continue
        raw_day = row.get("ds")
        day = raw_day.date() if isinstance(raw_day, dt.datetime) else raw_day
        if not isinstance(day, dt.date):
            continue
        segment = None if req.dimension is None else str(row.get(req.dimension))
        templates.setdefault((metric, segment), row)
        present.add((day, metric, str(row["group_id"]), segment))
        ends[metric] = max(ends.get(metric, day), day)
    for metric_name, (observed_end, certified_end) in observation_edges.items():
        end = observed_end
        if certified_end is not None:
            end = min(end, certified_end)
        if experiment.observation_horizon_day is not None:
            end = min(end, experiment.observation_horizon_day)
        ends[metric_name] = min(ends.get(metric_name, end), end)
    if not ends:
        return rows
    output = list(rows)
    for metric in metrics:
        end = ends.get(metric.name)
        if end is None or end < experiment.start_day:
            continue
        observed_segments = {
            segment for name, segment in templates if name == metric.name and segment is not None
        }
        if req.dimension is None:
            segments: list[str | None] = [None]
        else:
            assigned_segments = _assigned_segment_roster(
                source, req.dimension, experiment, source_name
            )
            segments = sorted(observed_segments | set(assigned_segments))
            if not segments:
                continue
        for offset in range((end - experiment.start_day).days + 1):
            day = experiment.start_day + dt.timedelta(days=offset)
            for segment in segments:
                template = templates.get((metric.name, segment))
                for group in groups:
                    key = (day, metric.name, group, segment)
                    if key in present:
                        continue
                    row = {
                        name: (
                            0
                            if isinstance(value, (int, float)) and not isinstance(value, bool)
                            else value
                        )
                        for name, value in (template or {}).items()
                    }
                    row.update(
                        experiment_id=experiment.name,
                        ds=day,
                        group_id=group,
                        metric=metric.name,
                        n=0,
                    )
                    if req.dimension is not None:
                        row[req.dimension] = segment
                    output.append(row)
                    present.add(key)
    return output


def _undimensioned_slice(
    src: MomentSource, req: DayAxisRequest, needs_covariate: Callable[[Metric], bool]
) -> Iterator[EvidenceSlice]:
    by: tuple[str, ...] = (req.dimension,) if req.dimension is not None else ()
    for view, metrics in _metric_view_groups(req):
        rows: list[dict[str, Any]] = []
        for metric in metrics:
            if req.grain == "daily":
                rows.extend(
                    readouts.daily(
                        src, metrics=[metric.name], by=by, include_covariate=needs_covariate(metric)
                    )
                )
            else:
                rows.extend(
                    cast(
                        "list[dict[str, Any]]",
                        src.moments(
                            metric,
                            grain="asof",
                            by=by,
                            completed_windows_only=req.completed_windows_only,
                            **({"include_covariate": True} if needs_covariate(metric) else {}),
                        ),
                    )
                )
        yield EvidenceSlice(
            rows=rows,
            metrics=metrics,
            view=view,
            dimension=req.dimension,
            source_name=None,
        )


class MomentsEvidence:
    __slots__ = ("src",)

    def __init__(self, *, src: MomentSource) -> None:
        self.src = src

    def slices(
        self, req: DayAxisRequest, *, needs_covariate: Callable[[Metric], bool]
    ) -> Iterator[EvidenceSlice]:
        yield from _undimensioned_slice(self.src, req, needs_covariate)


class ArtifactEvidence:
    __slots__ = ("src", "experiment")

    def __init__(self, *, src: MomentSource, experiment: Experiment | None) -> None:
        self.src = src
        self.experiment = experiment

    def slices(
        self, req: DayAxisRequest, *, needs_covariate: Callable[[Metric], bool]
    ) -> Iterator[EvidenceSlice]:
        if req.dimension is None:
            yield from _undimensioned_slice(self.src, req, needs_covariate)
            return
        matching = [
            b
            for b in (self.experiment.breakouts if self.experiment else ())
            if b.property == req.dimension
        ]
        for breakout in matching:
            for metric in req.metrics:
                evidence = cast(Any, self.src).breakout_moments(
                    metric, breakout, **_asof_kwargs(metric, req, needs_covariate)
                )
                view: DayAxisView = (
                    "asof"
                    if req.grain == "asof"
                    else ("cohort" if isinstance(metric, RetentionMetric) else "daily")
                )
                yield EvidenceSlice(
                    rows=list(evidence.rows),
                    metrics=(metric,),
                    view=view,
                    dimension=req.dimension,
                    source_name=evidence.source_name,
                )


class NativeEvidence:
    __slots__ = ("src", "experiment")

    def __init__(self, *, src: DayEvidenceSource, experiment: Experiment | None) -> None:
        self.src = src
        self.experiment = experiment

    def slices(
        self, req: DayAxisRequest, *, needs_covariate: Callable[[Metric], bool]
    ) -> Iterator[EvidenceSlice]:
        if req.dimension is not None:
            matching = [
                b
                for b in (self.experiment.breakouts if self.experiment else ())
                if b.property == req.dimension
            ]
            for breakout in matching:
                for view, metrics in _metric_view_groups(req):
                    rows: list[dict[str, Any]] = []
                    source_name: str | None = None
                    for index, metric in enumerate(metrics):
                        evidence = self.src.breakout_moments(
                            metric, breakout, **_asof_kwargs(metric, req, needs_covariate)
                        )
                        rows.extend(evidence.rows)
                        if index == 0:
                            source_name = evidence.source_name
                    yield EvidenceSlice(
                        rows=rows,
                        metrics=metrics,
                        view=view,
                        dimension=req.dimension,
                        source_name=source_name,
                    )
            return
        for view, metrics in _metric_view_groups(req):
            rows = []
            for metric in metrics:
                if req.grain == "daily":
                    rows.extend(
                        readouts.daily(
                            self.src,
                            metrics=[metric.name],
                            by=(),
                            include_covariate=needs_covariate(metric),
                        )
                    )
                else:
                    rows.extend(
                        self.src.moments(
                            metric,
                            grain="asof",
                            by=(),
                            completed_windows_only=req.completed_windows_only,
                            include_covariate=needs_covariate(metric),
                        )
                    )
            yield EvidenceSlice(
                rows=rows, metrics=metrics, view=view, dimension=None, source_name=None
            )


class DayAxisEvidence(Protocol):
    """Structural protocol satisfied by NativeEvidence/ArtifactEvidence/MomentsEvidence."""

    @property
    def src(self) -> Any: ...

    def slices(
        self, req: DayAxisRequest, *, needs_covariate: Callable[[Metric], bool]
    ) -> Iterator[EvidenceSlice]: ...


def _scope_day_axis_rows(
    rows,
    req: DayAxisRequest,
    *,
    route: Literal["artifact", "moments", "native"],
    plan: Any,
    configs: Sequence[Any] = (),
    design: Any = None,
    family_populations: set[str] | None = None,
):
    """Attach source-local scope to day-axis rows before returning them."""
    from increment.breakout.estimates import DailyLiftEstimates, DailyMetricValues
    from increment.readouts._multiplicity_scope import scoped_collection
    from increment.readouts._run import _config_snapshot

    collection = (
        DailyLiftEstimates
        if req.caller in ("run_daily_lift", "run_asof_lift")
        else DailyMetricValues
    )
    scope_request = {
        "kind": "increment.readout.day_axis",
        "caller": req.caller,
        "view": req.grain,
        "dimension": req.dimension,
        "completed_windows_only": req.completed_windows_only,
        "estimands": req.estimands,
        "population": req.population,
        "metrics": [
            metric.model_dump(mode="json") for metric in sorted(req.metrics, key=lambda m: m.name)
        ],
        "configs": [
            _config_snapshot(config, design)
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
    }
    return scoped_collection(
        rows,
        collection,
        plan,
        configs,
        scope_request,
        route=route,
        view=req.grain,
        design=design,
        dimension=req.dimension,
        family_populations=family_populations,
    )


class DayAxisReadouts:
    """Evidence acquisition and lift/values computation shared by the four
    day-axis entry points.
    """

    def __init__(
        self,
        *,
        src: MomentSource,
        plan: CompiledDecisionPlan,
        design: Randomized | Encouragement | Observational | None,
        experiment: Experiment | None,
        daily_lift: Callable[..., DailyLiftEstimates],
        evidence: DayAxisEvidence | None = None,
        route: Literal["artifact", "moments", "native"] | None = None,
    ) -> None:
        self._src = src
        self._plan = plan
        self._design = design
        self._experiment = experiment
        self._daily_lift = daily_lift
        self._evidence_override = evidence
        self._route_override = route

    def _route(self) -> Literal["artifact", "moments", "native"]:
        if self._route_override is not None:
            return self._route_override
        return _day_axis_source_route(self._src)

    def _evidence(
        self, req: DayAxisRequest, route: Literal["artifact", "moments", "native"]
    ) -> DayAxisEvidence:
        if self._evidence_override is not None:
            return self._evidence_override
        if route == "native":
            native_src = require_operation(self._src, "day_source", NativeViewSource).day_source(
                metrics=list(req.metrics), population=req.population
            )
            return cast(
                DayAxisEvidence, NativeEvidence(src=native_src, experiment=self._experiment)
            )
        if route == "artifact":
            if req.population == "triggered":
                from increment._source_operations import TriggeredPopulationOperation

                artifact_src = require_operation(
                    self._src, "triggered_source", TriggeredPopulationOperation
                ).triggered_source()
            else:
                artifact_src = self._src
            return cast(
                DayAxisEvidence, ArtifactEvidence(src=artifact_src, experiment=self._experiment)
            )
        if req.population == "triggered":
            from increment._source_operations import TriggeredPopulationOperation

            require_operation(self._src, "triggered_source", TriggeredPopulationOperation)
        return cast(DayAxisEvidence, MomentsEvidence(src=self._src))

    def values(self, req: DayAxisRequest) -> DailyMetricValues:
        route = self._route()
        validate_day_axis(
            req,
            src=self._src,
            design=self._design,
            plan=self._plan,
            route=route,
            experiment=self._experiment,
        )
        evidence = self._evidence(req, route)
        results: list[DailyMetricValue] = []
        failure_rows: list[dict[str, Any]] = []
        for evidence_slice in evidence.slices(req, needs_covariate=lambda _metric: False):
            raw_rows = _fill_triggered_early_look_rows(
                evidence_slice.rows,
                req=req,
                metrics=evidence_slice.metrics,
                experiment=self._experiment,
                source=self._src,
                edge_source=evidence.src,
                source_name=evidence_slice.source_name,
            )
            day_values = _run_daily_values(
                summary=raw_rows,
                metrics=list(evidence_slice.metrics),
                dimension=evidence_slice.dimension,
                source=evidence_slice.source_name,
                view=evidence_slice.view,
            )
            results.extend(_stamp_triggered_cell_failures(day_values, raw_rows, req))
            failure_rows.extend(raw_rows)
        if any(getattr(row, "decision_scope_complete", None) is False for row in results):
            results = [
                row.model_copy(
                    update={
                        "decision_scope_complete": False,
                        "decision_scope_reason_code": "readout.scope.decision_incomplete",
                        "decision_scope_reason_context": {"reason": "triggered_cell_unavailable"},
                    }
                )
                if hasattr(row, "decision_scope_complete")
                else row
                for row in results
            ]
        results = [
            row.model_copy(update={"analysis_population": req.population}) for row in results
        ]
        return _scope_day_axis_rows(
            results,
            req,
            route=route,
            plan=self._plan,
            design=self._design,
        )

    def lift(
        self,
        req: DayAxisRequest,
        *,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
    ) -> DailyLiftEstimates:
        route = self._route()
        validate_day_axis(
            req,
            src=self._src,
            design=self._design,
            plan=self._plan,
            route=route,
            experiment=self._experiment,
        )
        _, requested_estimands, encouragement_design = _resolve_lift_estimands(
            req.estimands, self._design
        )
        _refuse_triggered_compliance(
            population=req.population,
            design=encouragement_design,
            estimands=requested_estimands,
            method=req.caller,
        )
        if req.grain == "asof" and getattr(self._plan.inference, "registration", None) is not None:
            if req.dimension is not None:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported",
                    "segmented as-of history requires retained per-date stopped states; use the registered breakout checkpoint",
                )
            if req.population == "triggered":
                from increment.breakout.estimates import daily_sequential_projection

                policy = self._plan.view_policies.for_view(
                    "asof", mechanism=getattr(self._design, "mechanism", None), segmented=False
                )
                return _scope_day_axis_rows(
                    _unsupported_triggered_sequential_rows(
                        daily_sequential_projection(
                            readouts.asof_lift(
                                self._src,
                                metrics=[m.name for m in req.metrics],
                                estimands=req.estimands,
                                decision_method=decision_method,
                                sensitivity_methods=sensitivity_methods,
                                prior=prior,
                                completed_windows_only=req.completed_windows_only,
                            ),
                            correction=normalize_display_correction(policy.correction),
                        )
                    ),
                    req,
                    route=route,
                    plan=self._plan,
                    configs=self._src.context.configs,
                    design=self._design,
                    family_populations={"assigned"},
                )

            from increment.breakout.estimates import daily_sequential_projection

            asof_policy = self._plan.view_policies.for_view(
                "asof", mechanism=getattr(self._design, "mechanism", None), segmented=False
            )
            correction = normalize_display_correction(asof_policy.correction)
            rows = readouts.asof_lift(
                self._src,
                metrics=[m.name for m in req.metrics],
                estimands=req.estimands,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                completed_windows_only=req.completed_windows_only,
            )
            return _scope_day_axis_rows(
                daily_sequential_projection(rows, correction=correction),
                req,
                route=route,
                plan=self._plan,
                configs=self._src.context.configs,
                design=self._design,
            )
        opts = LiftOptions.resolve(
            self._src,
            req.metrics,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            alpha=self._plan.alpha,
        )
        for roles in opts.by_metric.values():
            _validate_methods(list(roles.methods))
        if req.caller == "run_asof_lift":
            validate_readout_inference(
                ReadoutRequest.from_source(
                    self._src,
                    metrics=list(req.metrics),
                    configs=opts.configs,
                    view="asof",
                    grain="asof",
                )
            )
        correction: str | None = None
        extra: dict[str, Any] = {}
        if req.grain == "asof":
            policy = self._plan.view_policies.for_view(
                "asof",
                mechanism=getattr(self._design, "mechanism", None),
                segmented=req.dimension is not None,
            )
            correction = normalize_display_correction(policy.correction)
            inference = _sequential_inference(self._plan)
            _validate_encouragement_asof_inference(
                list(req.metrics),
                cast(Any, self._design),
                inference=inference,
                completed_windows_only=req.completed_windows_only,
                method=req.caller,
            )
            extra = {
                "design": encouragement_design,
                "inference": inference,
                "estimands": requested_estimands,
                "correction": correction,
            }
        else:
            _validate_daily_inference(_sequential_inference(self._plan), method=req.caller)
        design_compliance, compliance_by_date = _compliance_readouts(
            req, encouragement_design, requested_estimands, extra, self._src
        )
        evidence = self._evidence(req, route)
        control_group = getattr(self._design, "control_group", None)
        results: list[DailyLiftEstimate] = []
        failure_rows: list[dict[str, Any]] = []
        effective_plan = self._plan
        if req.population == "triggered":
            from increment.plan import with_unassigned_procedures

            effective_plan = with_unassigned_procedures(
                self._plan, req.metrics, design=self._design
            )
        for evidence_slice in evidence.slices(req, needs_covariate=opts.needs_covariate):
            raw_rows = _fill_triggered_early_look_rows(
                evidence_slice.rows,
                req=req,
                metrics=evidence_slice.metrics,
                experiment=self._experiment,
                source=self._src,
                edge_source=evidence.src,
                source_name=evidence_slice.source_name,
            )
            names = {m.name for m in evidence_slice.metrics}
            kwargs = opts.daily_lift_kwargs()
            kwargs["methods_by_metric"] = {
                n: v for n, v in kwargs["methods_by_metric"].items() if n in names
            }
            kwargs["prior_by_metric"] = {
                n: v for n, v in kwargs["prior_by_metric"].items() if n in names
            }
            kwargs["method_roles_by_metric"] = {
                n: v for n, v in kwargs["method_roles_by_metric"].items() if n in names
            }
            results.extend(
                _estimate_day_axis_lift_slice(
                    self._daily_lift,
                    raw_rows,
                    evidence_slice,
                    req,
                    kwargs,
                    extra,
                    control_group,
                    requested_estimands,
                    effective_plan,
                    self._design,
                )
            )
            failure_rows.extend(raw_rows)
        if design_compliance:
            assert encouragement_design is not None
            day_order = _day_axis_label_order([*compliance_by_date, *(row.ds for row in results)])
            for ds in sorted(compliance_by_date, key=day_order.__getitem__):
                summary = compliance_by_date[ds]
                for row in estimate_compliance(
                    summary,
                    encouragement_design,
                    alpha=self._plan.alpha,
                    inference=_sequential_inference(self._plan),
                    compliance_requested="compliance" in requested_estimands,
                ).results:
                    results.append(
                        DailyLiftEstimate(
                            metric="uptake",
                            group_id=row.group_id,
                            method=row.method,
                            method_role=row.method_role,
                            inference=row.inference,
                            alternative=row.alternative,
                            dof=row.dof,
                            reference_kind=_slice_reference_kind(row),
                            reference_df=row.reference_df,
                            ds=ds,
                            lift=row.lift,
                            estimand="compliance",
                            value_scale=row.value_scale,
                            policy_name="compiled_plan",
                            low_reliability=any(
                                arm.n_units < DEFAULT_RELIABILITY_FLOOR
                                for arm in summary.arms
                                if arm.group_id in (row.group_id, control_group)
                            ),
                            note=_asof_monitoring_note(
                                row.note,
                                estimand="compliance",
                                inference=_sequential_inference(self._plan),
                                design=encouragement_design,
                            ),
                        )
                    )
            order = {metric.name: index for index, metric in enumerate(req.metrics)}
            results.sort(
                key=lambda row: (
                    day_order[row.ds],
                    -1 if row.estimand == "compliance" else order[row.metric],
                    row.group_id,
                )
            )
        from increment.estimation.multiplicity import stamp_multiplicity_status

        results = _stamp_triggered_cell_failures(
            results, failure_rows, req, control_group=control_group
        )

        results = [
            row.model_copy(update={"analysis_population": req.population}) for row in results
        ]

        return _scope_day_axis_rows(
            stamp_multiplicity_status(results, correction=correction),
            req,
            route=route,
            plan=self._plan,
            configs=opts.configs,
            design=self._design,
        )
