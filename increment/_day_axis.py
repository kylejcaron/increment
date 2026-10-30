"""`Analysis`'s day-axis quartet (run_daily/run_daily_lift/run_asof/run_asof_lift)
extracted behind one evidence-acquisition seam and one options seam.

Unsegmented encouragement compliance uses the source's declared uptake
cohort; outcome rows retain their own matching first stages.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from increment import readouts
from increment._analysis_config import _Unset, normalize_display_correction
from increment._evidence_refusals import check_method_covariates
from increment._evidence_types import (
    ComplianceSeriesRequest,
    ComplianceSummary,
    CountsRequest,
    DayAxisRequest,
    DayAxisRoute,
)
from increment._frame_panel import _day_axis_label_order, _resolve_window_days
from increment._lift_options import LiftOptions
from increment._readout_request import ReadoutRequest
from increment._readout_request import _raise as _raise_readout
from increment._source_types import declared_evidence
from increment.breakout.estimates import (
    DEFAULT_RELIABILITY_FLOOR,
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
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
from increment.evidence import evidence_provider, request_evidence
from increment.readouts._common import (
    _sequential_inference,
    _validate_encouragement_asof_inference,
)
from increment.readouts._daily import _validate_daily_inference
from increment.readouts._evidence import (
    ARTIFACT_ROUTE,
    MOMENTS_ROUTE,
    NATIVE_ROUTE,
    assigned_breakout_values,
    day_axis_route,
    design_allocation,
    load_compliance_series,
    load_day_axis_rows,
)
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.sources import _BuiltinMomentSource

if TYPE_CHECKING:
    from increment._evidence_types import SourceContext
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.engine import Method
    from increment.estimation.inference import Prior
    from increment.semantics.models import Experiment, Metric

DayAxisGrain = Literal["daily", "asof"]


class _DayAxisSource(Protocol):
    @property
    def context(self) -> SourceContext:
        """Immutable construction state used before evidence is read."""
        ...


_CLUSTERED_DAY_AXIS = RefusalSpec(
    "facade.analysis.clustered_day_axis",
    CapabilityError,
    template="{method}() is not supported on a clustered experiment -- experiment {experiment!r} declares cluster {cluster!r}, and cluster-robust inference is total-grain only (run()/srm()).",
)
_TRIGGER_UNSUPPORTED = RefusalSpec(
    "facade.analysis.trigger_unsupported",
    CapabilityError,
    lambda *, method, experiment, trigger, **_: (
        (
            f"{method}: experiment {experiment!r} declares no trigger on this source, "
            "so it has no triggered population."
            if trigger is None
            else f"{method}: experiment {experiment!r} declares trigger {trigger!r}, "
            "which this readout does not apply."
        )
        + " To analyze triggered populations, use Analysis.from_definitions or an "
        "Analysis.from_unit_day_artifact carrying trigger-population evidence."
    ),
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
_DAILY_COMPLIANCE_DIMENSION = RefusalSpec(
    "facade.analysis.daily_compliance_dimension_unsupported",
    UnsupportedRequestError,
    template="{method}: dimensioned daily compliance readouts are not supported; use an unsegmented daily compliance readout or an as-of readout",
)
_BH_SEGMENTED_ASOF = RefusalSpec(
    "facade.analysis.bh_segmented_asof",
    UnsupportedRequestError,
    template="{method}: BH multiplicity is supported for breakout families, not segmented as-of series; use bonferroni or none",
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


def validate_lift_request(
    req: DayAxisRequest,
    *,
    src: _DayAxisSource,
    design: Randomized | Encouragement | Observational | None,
    plan: CompiledDecisionPlan,
    route: DayAxisRoute,
    experiment: Experiment | None,
) -> tuple[bool, tuple[str, ...], Encouragement | None]:
    """Validate a day-axis lift and normalize its requested estimands."""
    validate_day_axis(
        req,
        src=src,
        design=design,
        plan=plan,
        route=route,
        experiment=experiment,
    )
    return _resolve_lift_estimands(req.estimands, design)


def _compliance_readouts(
    req: DayAxisRequest,
    design: Any,
    estimands: Sequence[str],
    extra: dict[str, Any],
    source: _BuiltinMomentSource,
) -> tuple[bool, dict[Any, ComplianceSummary]]:
    if design is None or req.dimension is not None:
        return False, {}
    extra["estimands"] = tuple(name for name in estimands if name != "compliance")
    if "compliance" not in estimands:
        return True, {}
    return True, dict(
        load_compliance_series(
            source,
            ComplianceSeriesRequest(
                design=design,
                completed_windows_only=req.completed_windows_only,
                population=req.population,
            ),
        )
    )


# The arm-family guard (`Analysis._require_arm_state`) stays in the facade:
# every day-axis method must call it first, before delegating here.
def validate_day_axis(
    req: DayAxisRequest,
    *,
    src: _DayAxisSource,
    design: Randomized | Encouragement | Observational | None,
    plan: CompiledDecisionPlan,
    route: DayAxisRoute,
    experiment: Experiment | None,
) -> None:
    """Validate one day-axis request across native, moments, and artifact routes.

    Branches apply only where routes genuinely differ. If multiple conditions
    fail, callers must not depend on the order in which those refusals surface.
    """
    compliance_only = (
        isinstance(design, Encouragement)
        and req.estimands == ("compliance",)
        and req.dimension is None
    )
    trigger_name = (
        experiment.trigger
        if experiment is not None and experiment.trigger is not None
        else src.context.trigger_name
    )
    if req.population == "triggered" and trigger_name is None:
        refuse(
            _TRIGGER_UNSUPPORTED,
            method=req.caller,
            experiment="<unknown>" if experiment is None else experiment.name,
            trigger=None,
            route=route,
            supported_sources=("from_definitions", "from_unit_day_artifact", "from_moments"),
        )
    if (
        not compliance_only
        and route in (ARTIFACT_ROUTE, NATIVE_ROUTE)
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
    if compliance_only:
        return
    if (
        route in (MOMENTS_ROUTE, ARTIFACT_ROUTE)
        and req.grain not in declared_evidence(src).grains
        and not checkpoint_asof
    ):
        from increment._evidence_refusals import refuse_grain

        refuse_grain(
            src,
            req.grain,
            tuple(sorted(declared_evidence(src).grains)),
            route="use a source with the requested day axis",
        )
    if route == NATIVE_ROUTE and "day_source" not in declared_evidence(src).operations:
        from increment._evidence_refusals import refuse_grain

        refuse_grain(
            src,
            req.grain,
            tuple(sorted(declared_evidence(src).grains)),
            route="use a source with the requested day axis",
        )
    reject_winsorized_day_axis(list(req.metrics), req.caller, grain=req.grain)
    if route == NATIVE_ROUTE:
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
    if (
        req.caller == "run_daily_lift"
        and isinstance(design, Encouragement)
        and (req.estimands is None or "late" in req.estimands)
    ):
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
    if (
        req.caller == "run_daily_lift"
        and isinstance(design, Encouragement)
        and req.dimension is not None
        and req.estimands is not None
        and "compliance" in req.estimands
    ):
        refuse(_DAILY_COMPLIANCE_DIMENSION, method=req.caller)
    if req.dimension is None:
        return
    if route == MOMENTS_ROUTE:
        declared_breakouts = declared_evidence(src).breakouts
        if req.dimension not in declared_breakouts:
            refuse(
                _UNKNOWN_DIMENSION,
                method=req.caller,
                dimension=req.dimension,
                declared=list(declared_breakouts),
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
        route == NATIVE_ROUTE
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


def _missing_compliance_rows(
    summary: ComplianceSummary,
    ds: dt.date,
    *,
    arm_roster: Sequence[str],
    control_group: str,
    population: Literal["assigned", "triggered"],
    inference: Any,
    reason: Literal["missing_arm", "few_units"],
) -> list[DailyLiftEstimate]:
    from increment._immutable import _FrozenMapping
    from increment.breakout.estimates import _nan_lift_rows

    arms = {arm.group_id: arm for arm in summary.arms}
    rows = []
    for group_id in arm_roster:
        if group_id == control_group:
            continue
        treatment_units = arms[group_id].n_units if group_id in arms else 0
        control_units = arms[control_group].n_units if control_group in arms else 0
        missing_arm = reason == "missing_arm" or treatment_units == 0 or control_units == 0
        unavailable: Literal["no_control_arm", "no_treatment_arm", "few_units"] = (
            "no_control_arm"
            if control_units == 0
            else "no_treatment_arm"
            if treatment_units == 0
            else "few_units"
        )
        row = _nan_lift_rows(
            {("uptake", group_id): unavailable},
            methods=None,
            ds_value=ds,
            ds_basis="calendar",
            dimension=None,
            dim_value=None,
            source=None,
            inference="sequential" if inference is not None else "fixed",
        )[0]
        context = _FrozenMapping(
            {
                "day": ds.isoformat(),
                "group_id": group_id,
                "reason": "missing_arm" if missing_arm else "few_units",
                "control_group": control_group,
            }
        )
        code = (
            "readout.cell.missing_arm"
            if missing_arm
            else "readout.cell.missing_metric_observations"
        )
        rows.append(
            row.model_copy(
                update={
                    "analysis_population": population,
                    "estimand": "compliance",
                    "value_scale": "absolute",
                    "failure_code": code,
                    "failure_context": context,
                    "sampling_available": False,
                    "sampling_reason_code": code,
                    "sampling_reason_context": context,
                    "decision_scope_complete": False,
                    "decision_scope_reason_code": "readout.scope.decision_incomplete",
                    "decision_scope_reason_context": context,
                    "n_control": control_units,
                    "n_treat": treatment_units,
                }
            )
        )
    return rows


def _estimate_compliance_lift_rows(
    summary: ComplianceSummary,
    design: Encouragement,
    ds: dt.date,
    *,
    alpha: float,
    inference: Any,
    compliance_requested: bool,
    control_group: str,
    population: Literal["assigned", "triggered"],
    arm_roster: Sequence[str],
) -> tuple[list[Any], list[DailyLiftEstimate]]:
    def estimate(summary: ComplianceSummary) -> list[Any]:
        return list(
            estimate_compliance(
                summary,
                design,
                alpha=alpha,
                inference=inference,
                compliance_requested=compliance_requested,
            ).results
        )

    try:
        return estimate(summary), []
    except InvalidRequestError as error:
        if error.code == "estimation.encouragement.control.missing":
            return [], _missing_compliance_rows(
                summary,
                ds,
                arm_roster=arm_roster,
                control_group=control_group,
                population=population,
                inference=inference,
                reason="missing_arm",
            )
        if error.code not in (
            "estimation.armstats.arm_stats.least_compute_metric",
            "estimation.encouragement.cluster.arm_needs_two",
        ):
            raise

    estimates: list[Any] = []
    unavailable: list[DailyLiftEstimate] = []
    for group_id in arm_roster:
        if group_id == control_group:
            continue
        arms = tuple(arm for arm in summary.arms if arm.group_id in (control_group, group_id))
        if not any(arm.group_id == group_id for arm in arms):
            unavailable.extend(
                _missing_compliance_rows(
                    summary,
                    ds,
                    arm_roster=(group_id,),
                    control_group=control_group,
                    population=population,
                    inference=inference,
                    reason="missing_arm",
                )
            )
            continue
        contrast = replace(summary, arms=arms)
        try:
            estimates.extend(estimate(contrast))
        except InvalidRequestError as contrast_error:
            if contrast_error.code not in (
                "estimation.armstats.arm_stats.least_compute_metric",
                "estimation.encouragement.cluster.arm_needs_two",
            ):
                raise
            unavailable.extend(
                _missing_compliance_rows(
                    summary,
                    ds,
                    arm_roster=(group_id,),
                    control_group=control_group,
                    population=population,
                    inference=inference,
                    reason="few_units",
                )
            )
    return estimates, unavailable


def _build_compliance_day_rows(
    compliance_rows: Sequence[Any],
    summary: ComplianceSummary,
    ds: dt.date,
    *,
    control_group: str,
    design: Encouragement,
    inference: Any,
) -> list[DailyLiftEstimate]:
    arms = {arm.group_id: arm for arm in summary.arms}
    return [
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
            n_control=arms[control_group].n_units,
            n_treat=arms[row.group_id].n_units,
            policy_name="compiled_plan",
            low_reliability=any(
                arm.n_units < DEFAULT_RELIABILITY_FLOOR
                for arm in summary.arms
                if arm.group_id in (row.group_id, control_group)
            ),
            note=_asof_monitoring_note(
                row.note,
                estimand="compliance",
                inference=inference,
                design=design,
            ),
        )
        for row in compliance_rows
    ]


def _append_compliance_rows(
    results: list[DailyLiftEstimate],
    compliance_by_date: dict[Any, ComplianceSummary],
    *,
    enabled: bool,
    req: DayAxisRequest,
    design: Encouragement | None,
    control_group: str | None,
    requested_estimands: Sequence[str],
    plan: CompiledDecisionPlan,
    arm_roster: Sequence[str],
) -> None:
    if not enabled or design is None:
        return
    assert control_group is not None
    if not arm_roster:
        arm_roster = sorted(
            {arm.group_id for summary in compliance_by_date.values() for arm in summary.arms}
            | {control_group}
        )
    day_order = _day_axis_label_order([*compliance_by_date, *(row.ds for row in results)])
    inference = _sequential_inference(plan)
    for ds in sorted(compliance_by_date, key=day_order.__getitem__):
        summary = compliance_by_date[ds]
        compliance_rows, unavailable_rows = _estimate_compliance_lift_rows(
            summary,
            design,
            ds,
            alpha=plan.alpha,
            inference=inference,
            compliance_requested="compliance" in requested_estimands,
            control_group=control_group,
            population=req.population,
            arm_roster=arm_roster,
        )
        built_rows = _build_compliance_day_rows(
            compliance_rows,
            summary,
            ds,
            control_group=control_group,
            design=design,
            inference=inference,
        )
        if unavailable_rows:
            scope_reason = unavailable_rows[0]
            built_rows = [
                row.model_copy(
                    update={
                        "decision_scope_complete": False,
                        "decision_scope_reason_code": scope_reason.decision_scope_reason_code,
                        "decision_scope_reason_context": scope_reason.decision_scope_reason_context,
                    }
                )
                for row in built_rows
            ]
        results.extend(built_rows)
        results.extend(unavailable_rows)
        if "compliance" in requested_estimands:
            present = {row.group_id for row in (*built_rows, *unavailable_rows)}
            absent = [
                group_id
                for group_id in arm_roster
                if group_id != control_group and group_id not in present
            ]
            results.extend(
                _missing_compliance_rows(
                    summary,
                    ds,
                    arm_roster=absent,
                    control_group=control_group,
                    population=req.population,
                    inference=inference,
                    reason="missing_arm",
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
                _policy_by_cell=policy_by_cell,
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


def _triggered_group_roster(source: _BuiltinMomentSource, experiment: Experiment) -> list[str]:
    provider = evidence_provider(source)
    if "assignment" in provider.evidence.count_grains:
        counts = cast(
            Any,
            request_evidence(
                provider,
                CountsRequest(kind="assignment", population="assigned"),
            ),
        ).counts
        groups = set(counts)
    elif "unit" in provider.evidence.count_grains:
        counts = cast(
            Any,
            request_evidence(
                provider,
                CountsRequest(kind="unit", population="assigned"),
            ),
        ).counts
        groups = set(counts)
    else:
        return []
    allocation = design_allocation(source)
    if allocation is not None:
        groups.update(allocation)
    groups.add(experiment.control_group)
    from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

    return sorted(groups - {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL})


def _assigned_segment_roster(
    source: _BuiltinMomentSource,
    dimension: str,
    experiment: Experiment,
    source_name: str | None,
) -> list[str]:
    assigned_values = assigned_breakout_values(source)
    if assigned_values is None:
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
    source: _BuiltinMomentSource,
    observation_edges: Mapping[str, tuple[dt.date, dt.date | None]] | None,
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
    edge_by_metric = observation_edges or {}
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
    for metric_name, (observed_end, certified_end) in edge_by_metric.items():
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


def _day_axis_base_domain(req: DayAxisRequest) -> tuple[dt.date, ...]:
    experiment = req.experiment
    if experiment is None:
        return ()
    start = experiment.start_day
    end = experiment.observation_horizon_day
    if end is None or end < start:
        return ()
    return tuple(start + dt.timedelta(days=n) for n in range((end - start).days + 1))


def _metric_day_axis_domain(req: DayAxisRequest, metric: Any) -> tuple[dt.date, ...]:
    domain = _day_axis_base_domain(req)
    if req.grain != "daily" or not domain or metric.type == "retention":
        return domain
    window_days = _resolve_window_days(metric)
    if window_days is None:
        return domain
    exclusive_end = domain[0] + dt.timedelta(days=window_days)
    return tuple(day for day in domain if day < exclusive_end)


def _day_axis_domain(req: DayAxisRequest) -> tuple[dt.date, ...]:
    if req.grain != "daily" or not req.metrics:
        return _day_axis_base_domain(req)
    domains = (_metric_day_axis_domain(req, metric) for metric in req.metrics)
    return tuple(sorted({day for domain in domains for day in domain}))


def _fill_daily_domain_gaps(rows: list[Any], req: DayAxisRequest) -> list[Any]:
    if req.grain != "daily" or req.population == "triggered" or not rows:
        return rows
    domain = _day_axis_domain(req)
    if not domain:
        return rows
    identity_fields = (
        "metric",
        "group_id",
        "analysis_population",
        "dimension",
        "dimension_value",
        "source",
        "ds_basis",
        "method",
        "method_role",
        "estimand",
    )

    def identity(row: Any) -> tuple[object, ...]:
        return tuple(getattr(row, field, None) for field in identity_fields)

    metric_domains = {}
    for metric in req.metrics:
        declared_domain = () if metric.type == "retention" else _metric_day_axis_domain(req, metric)
        emitted_domain = {row.ds for row in rows if row.metric == metric.name}
        metric_domains[metric.name] = tuple(sorted(set(declared_domain) | emitted_domain))
    if not metric_domains:
        metric_domains = {"": domain}
    present = {(identity(row), row.ds) for row in rows}
    additions = []
    for template in rows:
        key = identity(template)
        row_domain = metric_domains.get(template.metric, domain)
        for day in row_domain:
            if (key, day) in present:
                continue
            present.add((key, day))
            if hasattr(template, "value"):
                additions.append(
                    template.model_copy(
                        update={
                            "ds": day,
                            "value": None,
                            "unavailable": "missing_metric_observations",
                            "n": 0,
                        }
                    )
                )
            else:
                additions.append(
                    template.model_copy(
                        update={
                            "ds": day,
                            "lift": None,
                            "relative_confidence_set": None,
                            "sequential_result": None,
                            "binomial_set": None,
                            "delta_evidence": None,
                            "unavailable": "missing_metric_observations",
                            "n_control": 0,
                            "n_treat": 0,
                            "mean_control": None,
                            "mean_treat": None,
                            "sampling_available": False,
                            "sampling_reason_code": "readout.cell.missing_metric_observations",
                            "posterior_available": False,
                            "posterior_reason_code": "readout.cell.missing_metric_observations",
                            "posterior_reason_context": None,
                            "posterior_prob_favorable_reason_code": None,
                            "posterior_model": None,
                            "posterior_scale": None,
                            "posterior_estimate": None,
                            "posterior_lb": None,
                            "posterior_ub": None,
                            "posterior_level": None,
                            "posterior_alpha": None,
                            "posterior_latent_mean": None,
                            "posterior_latent_sd": None,
                            "posterior_prob_favorable": None,
                            "posterior_components": None,
                            "relative_unavailable_reason": None,
                            "relative_error_code": None,
                            "relative_error_context": None,
                            "abs_diff": None,
                            "abs_se": None,
                            "abs_lb": None,
                            "abs_ub": None,
                            "abs_reference_kind": None,
                            "abs_reference_df": None,
                            "abs_alpha": None,
                        }
                    )
                )
    return [*rows, *additions]


def _day_axis_scope(req: DayAxisRequest):
    from increment._evidence_types import BreakoutIdentity
    from increment.kernel.contracts import Scope

    experiment = req.experiment
    time_basis = (
        "cohort"
        if req.metrics and all(metric.type == "retention" for metric in req.metrics)
        else "calendar"
    )
    segment = None
    if req.dimension is not None:
        source_name = next(
            (
                breakout.source
                for breakout in getattr(experiment, "breakouts", ())
                if breakout.property == req.dimension
            ),
            None,
        )
        segment = BreakoutIdentity(req.dimension, source_name)
    return Scope(
        view=req.grain,
        grain=req.grain,
        day_domain=_day_axis_domain(req),
        population=req.population,
        time_basis=time_basis,
        completion_policy=req.completed_windows_only,
        segment=segment,
    )


def _scope_day_axis_rows(
    rows,
    req: DayAxisRequest,
    *,
    route: DayAxisRoute,
    plan: Any,
    configs: Sequence[Any] = (),
    design: Any = None,
    source: Any,
    family_populations: set[str] | None = None,
    execution_context=None,
    scope=None,
    identity_rows: Sequence[Any] | None = None,
):
    """Attach source-local scope to day-axis rows before returning them."""
    from increment._source_identity import source_identity
    from increment.breakout.estimates import DailyLiftEstimates, DailyMetricValues
    from increment.readouts._design_scope import _config_snapshot
    from increment.readouts._multiplicity_scope import scoped_collection

    collection = (
        DailyLiftEstimates
        if req.caller in ("run_daily_lift", "run_asof_lift")
        else DailyMetricValues
    )
    scope = _day_axis_scope(req) if scope is None else scope
    effective_modes = {} if execution_context is None else execution_context.effective_modes
    coherence = (
        None
        if execution_context is None or "provider" in effective_modes.values()
        else execution_context.coherence
    )
    scope_request = {
        "kind": "increment.readout.day_axis",
        "caller": req.caller,
        "view": req.grain,
        "dimension": req.dimension,
        "completed_windows_only": req.completed_windows_only,
        "estimands": req.estimands,
        "population": req.population,
        "source_identity": source_identity(source),
        "metrics": [
            metric.model_dump(mode="json") for metric in sorted(req.metrics, key=lambda m: m.name)
        ],
        "configs": [
            _config_snapshot(config, design)
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
    }
    result = scoped_collection(
        rows,
        collection,
        plan,
        configs,
        scope_request,
        metrics=req.metrics,
        route=route,
        view=req.grain,
        design=design,
        dimension=req.dimension,
        family_populations=family_populations,
        identity_rows=identity_rows,
    )
    if result.metadata is not None:
        from increment._immutable import _FrozenMapping

        result._metadata = result.metadata.model_copy(
            update={
                "coherence": coherence,
                "effective_modes": _FrozenMapping(effective_modes),
            }
        )
    return result


def _order_day_axis_rows(rows: list[Any], req: DayAxisRequest) -> list[Any]:
    """Give every day-axis collection a source-independent identity order."""
    if len(rows) < 2:
        return rows
    day_order = _day_axis_label_order(row.ds for row in rows)
    days = sorted(
        day_order,
        key=lambda value: (type(value).__qualname__, day_order[value]),
    )
    day_rank = {day: rank for rank, day in enumerate(days)}
    metric_rank = {metric.name: rank for rank, metric in enumerate(req.metrics)}
    return sorted(
        rows,
        key=lambda row: (
            day_rank[row.ds],
            metric_rank.get(row.metric, len(metric_rank)),
            row.group_id,
            getattr(row, "method", None) or "",
            getattr(row, "method_role", None) or "",
            getattr(row, "estimand", ""),
            getattr(row, "dimension", None) or "",
            type(getattr(row, "dimension_value", None)).__qualname__,
            repr(getattr(row, "dimension_value", None)),
            getattr(row, "source", None) or "",
        ),
    )


class DayAxisReadouts:
    """Evidence acquisition and lift/values computation shared by the four
    day-axis entry points.
    """

    def __init__(
        self,
        *,
        src: _BuiltinMomentSource,
        plan: CompiledDecisionPlan,
        design: Randomized | Encouragement | Observational | None,
        experiment: Experiment | None,
        daily_lift: Callable[..., DailyLiftEstimates],
        execution_context=None,
        metadata_context=None,
        scope_is_pinned: bool = False,
    ) -> None:
        self._src = src
        self._plan = plan
        self._design = design
        self._experiment = experiment
        self._daily_lift = daily_lift
        self._execution_context = execution_context
        self._metadata_context = metadata_context
        self._scope_is_pinned = scope_is_pinned

    def resolve_lift_options(
        self,
        metrics: Sequence[Metric],
        *,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
    ) -> LiftOptions:
        return LiftOptions.resolve(
            self._src,
            metrics,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            alpha=self._plan.alpha,
        )

    def _route(self) -> DayAxisRoute:
        return day_axis_route(self._src)

    def validate_request(self, req: DayAxisRequest, *, lift: bool = False) -> None:
        """Validate a day-axis request without acquiring evidence."""
        req = replace(req, experiment=self._experiment)
        route = self._route()
        if lift:
            validate_lift_request(
                req,
                src=self._src,
                design=self._design,
                plan=self._plan,
                route=route,
                experiment=self._experiment,
            )
        else:
            validate_day_axis(
                req,
                src=self._src,
                design=self._design,
                plan=self._plan,
                route=route,
                experiment=self._experiment,
            )

    def values(self, req: DayAxisRequest, *, scope=None) -> DailyMetricValues:

        req = replace(req, experiment=self._experiment)
        scope = _day_axis_scope(req) if scope is None else scope
        route = self._route()
        validate_day_axis(
            req,
            src=self._src,
            design=self._design,
            plan=self._plan,
            route=route,
            experiment=self._experiment,
        )
        if self._execution_context is not None:
            context = self._execution_context
            context.validate_typed_request(req)
            context.validate_request(self._src, req.metrics, (), population=req.population)
            if req.grain == "asof" and isinstance(self._design, Encouragement):
                from increment._evidence_types import ComplianceSeriesRequest

                context.validate_typed_request(
                    ComplianceSeriesRequest(
                        design=self._design,
                        completed_windows_only=req.completed_windows_only,
                        population=req.population,
                    )
                )
            if req.dimension is not None:
                from increment._evidence_types import BreakoutIdentity

                source_name = next(
                    (
                        breakout.source
                        for breakout in getattr(self._experiment, "breakouts", ())
                        if breakout.property == req.dimension
                    ),
                    None,
                )
                context.validate_breakouts((BreakoutIdentity(req.dimension, source_name),))
            from increment.readouts._evidence import _builtin_source

            with context.borrow() as provider:
                return DayAxisReadouts(
                    src=_builtin_source(provider),
                    plan=self._plan,
                    design=self._design,
                    experiment=self._experiment,
                    daily_lift=self._daily_lift,
                    metadata_context=context,
                    scope_is_pinned=True,
                ).values(req, scope=scope)
        evidence = load_day_axis_rows(self._src, req, scope_is_pinned=self._scope_is_pinned)
        results: list[DailyMetricValue] = []
        failure_rows: list[dict[str, Any]] = []
        for evidence_slice in evidence.slices:
            raw_rows = _fill_triggered_early_look_rows(
                evidence_slice.rows,
                req=req,
                metrics=evidence_slice.metrics,
                experiment=self._experiment or evidence_slice.experiment,
                source=self._src,
                observation_edges=evidence_slice.observation_edges,
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
        identity_rows = tuple(results)
        results = _fill_daily_domain_gaps(results, req)
        return _scope_day_axis_rows(
            _order_day_axis_rows(results, req),
            req,
            source=self._src,
            route=route,
            plan=self._plan,
            design=self._design,
            execution_context=self._metadata_context,
            identity_rows=identity_rows,
        )

    def lift(  # noqa: PLR0915
        self,
        req: DayAxisRequest,
        *,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        scope=None,
        options: LiftOptions | None = None,
    ) -> DailyLiftEstimates:
        req = replace(req, experiment=self._experiment)
        scope = _day_axis_scope(req) if scope is None else scope
        route = self._route()
        from increment._analysis_config import validate_scalar_route

        _, requested_estimands, encouragement_design = validate_lift_request(
            req,
            src=self._src,
            design=self._design,
            plan=self._plan,
            route=route,
            experiment=self._experiment,
        )
        opts = options or LiftOptions.resolve(
            self._src,
            req.metrics,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
            alpha=self._plan.alpha,
        )
        if self._execution_context is not None:
            context = self._execution_context
            if requested_estimands == ("compliance",) and req.dimension is None:
                context.validate_source(self._src)
            context.validate_typed_request(req)
            if requested_estimands != ("compliance",) or req.dimension is not None:
                context.validate_request(
                    self._src, req.metrics, opts.configs, population=req.population
                )
            if req.dimension is not None:
                from increment._evidence_types import BreakoutIdentity

                source_name = next(
                    (
                        breakout.source
                        for breakout in getattr(self._experiment, "breakouts", ())
                        if breakout.property == req.dimension
                    ),
                    None,
                )
                context.validate_breakouts((BreakoutIdentity(req.dimension, source_name),))
            from increment.readouts._evidence import _builtin_source

            with context.borrow() as provider:
                return DayAxisReadouts(
                    src=_builtin_source(provider),
                    plan=self._plan,
                    design=self._design,
                    experiment=self._experiment,
                    daily_lift=self._daily_lift,
                    metadata_context=context,
                    scope_is_pinned=True,
                ).lift(
                    req,
                    decision_method=decision_method,
                    sensitivity_methods=sensitivity_methods,
                    prior=prior,
                    scope=scope,
                    options=opts,
                )
        if req.grain == "asof" and getattr(self._plan.inference, "registration", None) is not None:
            if req.dimension is not None:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported",
                    "segmented as-of history requires retained per-date stopped states; use the registered breakout checkpoint",
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
                _population=req.population,
            )
            source_by_dimension = {
                breakout.property: breakout.source
                for breakout in getattr(self._experiment, "breakouts", ())
            }
            return _scope_day_axis_rows(
                _order_day_axis_rows(
                    daily_sequential_projection(
                        rows,
                        correction=correction,
                        source_by_dimension=source_by_dimension,
                    ),
                    req,
                ),
                req,
                source=self._src,
                route=route,
                plan=self._plan,
                configs=self._src.context.configs,
                design=self._design,
                execution_context=self._metadata_context,
                scope=scope,
            )
        for roles in opts.by_metric.values():
            _validate_methods(list(roles.methods))
        for name, roles in opts.by_metric.items():
            check_method_covariates(self._src, name, roles.methods)
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
        extra: dict[str, Any] = {
            "design": encouragement_design if req.grain == "asof" else None,
            "estimands": (
                tuple(name for name in requested_estimands if name not in ("compliance", "late"))
                if req.grain == "daily"
                else requested_estimands
            ),
        }
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
        scalar_evidence = evidence_provider(self._src).evidence.scalar
        validate_scalar_route(
            opts.configs,
            route=f"Analysis.{req.caller}",
            representation=(
                "registered"
                if getattr(self._plan.inference, "registration", None) is not None
                else "implicit"
            ),
            registered_plan=self._plan,
            design=self._design,
            covariate_moments_available=(
                scalar_evidence is None
                or scalar_evidence.covariate_grains is None
                or req.grain in scalar_evidence.covariate_grains
            ),
        )
        design_compliance, compliance_by_date = _compliance_readouts(
            req, encouragement_design, requested_estimands, extra, self._src
        )
        compliance_only = (
            encouragement_design is not None
            and requested_estimands == ("compliance",)
            and req.dimension is None
        )
        control_group = getattr(self._design, "control_group", None)
        results: list[DailyLiftEstimate] = []
        failure_rows: list[dict[str, Any]] = []
        effective_plan = self._plan
        if req.population == "triggered":
            from increment.plan import with_unassigned_procedures

            effective_plan = with_unassigned_procedures(
                self._plan, req.metrics, design=self._design
            )
        evidence = None
        evidence_slices = ()
        if not compliance_only:
            covariate_metrics = frozenset(
                metric.name for metric in req.metrics if opts.needs_covariate(metric)
            )
            evidence = load_day_axis_rows(
                self._src,
                replace(
                    req,
                    experiment=self._experiment,
                    covariate_metrics=covariate_metrics,
                ),
                scope_is_pinned=self._scope_is_pinned,
            )
            evidence_slices = evidence.slices
        for evidence_slice in evidence_slices:
            raw_rows = _fill_triggered_early_look_rows(
                evidence_slice.rows,
                req=req,
                metrics=evidence_slice.metrics,
                experiment=self._experiment or evidence_slice.experiment,
                source=self._src,
                observation_edges=evidence_slice.observation_edges,
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
            if not (req.grain == "daily" and requested_estimands == ("compliance",)):
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
        arm_roster = (
            _triggered_group_roster(self._src, self._experiment)
            if self._experiment is not None and (req.metrics or design_compliance)
            else []
        )
        _append_compliance_rows(
            results,
            compliance_by_date,
            enabled=design_compliance,
            req=req,
            design=encouragement_design,
            control_group=control_group,
            requested_estimands=requested_estimands,
            plan=self._plan,
            arm_roster=arm_roster,
        )
        from increment.estimation.multiplicity import stamp_multiplicity_status

        results = _stamp_triggered_cell_failures(
            results, failure_rows, req, control_group=control_group
        )

        results = [
            row.model_copy(update={"analysis_population": req.population}) for row in results
        ]
        identity_rows = tuple(results)
        results = _fill_daily_domain_gaps(results, req)

        return _scope_day_axis_rows(
            _order_day_axis_rows(
                stamp_multiplicity_status(results, correction=correction),
                req,
            ),
            req,
            source=self._src,
            route=route,
            plan=self._plan,
            configs=opts.configs,
            design=self._design,
            execution_context=self._metadata_context,
            identity_rows=identity_rows,
        )
