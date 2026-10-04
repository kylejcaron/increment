"""`Analysis`'s day-axis quartet (run_daily/run_daily_lift/run_asof/run_asof_lift)
extracted behind one evidence-acquisition seam and one options seam.

Unsegmented encouragement compliance uses the source's declared uptake
cohort; outcome rows retain their own matching first stages.
"""

from __future__ import annotations

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
    reject_contradictory_completed_windows,
    reject_quantile_metrics,
    reject_retention_metrics,
    reject_retention_under_encouragement,
    reject_winsorized_day_axis,
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
from increment.estimation.engine import _validate_methods
from increment.estimation.inference import validate_readout_inference
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
    if experiment is not None and experiment.trigger is not None:
        refuse(
            _TRIGGER_UNSUPPORTED,
            method=req.caller,
            experiment=experiment.name,
            trigger=experiment.trigger,
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
        # Native routes reject quantiles and daily retention; moments/artifacts
        # retain the broader metric set.
        moments_reason = "per-day moments" if req.grain == "daily" else "as-of moments"
        reject_quantile_metrics(
            list(req.metrics),
            f"{req.caller}()",
            reason=f"quantiles do not decompose into {moments_reason}",
        )
        if req.grain == "daily":
            reject_retention_metrics(list(req.metrics), req.caller, view="cohort")
        # As-of retention rejection is a no-op for this helper.
    if req.caller == "run_asof_lift" and isinstance(design, Encouragement):
        reject_retention_under_encouragement(list(req.metrics), req.caller)
    if req.grain == "asof" and req.completed_windows_only:
        reject_contradictory_completed_windows(list(req.metrics), req.caller)
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

    def slices(
        self, req: DayAxisRequest, *, needs_covariate: Callable[[Metric], bool]
    ) -> Iterator[EvidenceSlice]: ...


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
                metrics=list(req.metrics)
            )
            return cast(
                DayAxisEvidence, NativeEvidence(src=native_src, experiment=self._experiment)
            )
        if route == "artifact":
            return cast(
                DayAxisEvidence, ArtifactEvidence(src=self._src, experiment=self._experiment)
            )
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
        for evidence_slice in evidence.slices(req, needs_covariate=lambda _metric: False):
            results.extend(
                _run_daily_values(
                    summary=evidence_slice.rows,
                    metrics=list(evidence_slice.metrics),
                    dimension=evidence_slice.dimension,
                    source=evidence_slice.source_name,
                    view=evidence_slice.view,
                )
            )
        return DailyMetricValues(results)

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
        if req.grain == "asof" and getattr(self._plan.inference, "registration", None) is not None:
            if req.dimension is not None:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "route.unsupported",
                    "segmented as-of history requires retained per-date stopped states; use the registered breakout checkpoint",
                )
            from increment.breakout.estimates import daily_sequential_projection

            rows = readouts.asof_lift(
                self._src,
                metrics=[m.name for m in req.metrics],
                estimands=req.estimands,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=prior,
                completed_windows_only=req.completed_windows_only,
            )
            return daily_sequential_projection(rows)
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
        _, requested_estimands, encouragement_design = _resolve_lift_estimands(
            req.estimands, self._design
        )
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
        design_compliance = (
            encouragement_design is not None and req.grain == "asof" and req.dimension is None
        )
        compliance_by_date: dict[Any, ComplianceSummary] = {}
        if design_compliance:
            extra["estimands"] = tuple(name for name in requested_estimands if name != "compliance")
            if "compliance" in requested_estimands:
                assert encouragement_design is not None
                compliance_by_date = compliance_summary_series(
                    self._src,
                    encouragement_design,
                    completed_windows_only=req.completed_windows_only,
                )
        evidence = self._evidence(req, route)
        control_group = getattr(self._design, "control_group", None)
        results: list[DailyLiftEstimate] = []
        for evidence_slice in evidence.slices(req, needs_covariate=opts.needs_covariate):
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
                self._daily_lift(
                    summary=evidence_slice.rows,
                    metrics=list(evidence_slice.metrics),
                    control_group=control_group,
                    dimension=evidence_slice.dimension,
                    source=evidence_slice.source_name,
                    view=evidence_slice.view,
                    **kwargs,
                    **extra,
                )
            )
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
        return DailyLiftEstimates(results)
