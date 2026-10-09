"""Analysis.run behind one WholeWindowReadouts object.

Both the seam/artifact-moments family (one readouts.arm_moments call per
population) and the native design-routed family (per-metric arm estimation
with guardrail/estimand handling and a triggered-population second pass)
answer through the same WholeWindowRequest/WholeWindowReadouts pair; the
facade decides only which state it holds, never how a route resolves. Also
hosts the operation-lookup and switchback-override helpers analysis.py,
_breakout_readouts.py, and _sitewide.py all need, so none of the three
modules imports from another.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from increment import readouts
from increment._analysis_config import UNSET, _Unset
from increment._day_axis import _TRIGGER_UNSUPPORTED, _day_axis_source_route
from increment._literals import ValueScale
from increment._readout_request import _raise as _raise_readout
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.readouts._requests import _design_compliance_only, _validate_run_request
from increment.sources import SourceContext, require_operation

if TYPE_CHECKING:
    from increment.breakout.estimates import LiftEstimates
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.engine import Method
    from increment.estimation.inference import Prior
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Experiment, Metric
    from increment.sources import MomentSource, SourceOperation

RouteName = Literal["artifact", "moments", "native"]


def _as_lift_estimates(rows: Sequence[LiftEstimate]) -> LiftEstimates:
    from increment.breakout.estimates import LiftEstimates

    if isinstance(rows, LiftEstimates):
        return rows
    return LiftEstimates(rows)


def _select_population(
    rows: LiftEstimates, population: Literal["assigned", "triggered"] | None
) -> LiftEstimates:
    if population is None:
        return rows
    return rows.filter(lambda row: row.analysis_population == population)


# The source cannot serve `operation`; context names it for callers.
_ANALYSIS_OPERATION = RefusalSpec(
    "facade.analysis.operation",
    CapabilityError,
    lambda *, message, operation: message,
)

_UNKNOWN_METRIC = RefusalSpec(
    "facade.analysis.unknown_metric",
    InvalidRequestError,
    template="metric {metric!r} is not declared on this experiment. Declared metrics: {declared!r}",
)


def _require_analysis_operation[T](
    source: MomentSource,
    operation: SourceOperation,
    protocol: type[T],
    *,
    message: str,
) -> T:
    if not isinstance(getattr(source, "context", None), SourceContext):
        _refuse(_ANALYSIS_OPERATION, message=message, operation=operation)
    if operation not in source.operations:
        _refuse(_ANALYSIS_OPERATION, message=message, operation=operation)
    return require_operation(source, operation, protocol)


def reject_role_overrides_under_contrast(
    *,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
) -> None:
    """Refuse role overrides for plan-owned switchback contrast evidence."""
    for argument, value in (
        ("decision_method", decision_method),
        ("sensitivity_methods", sensitivity_methods),
        ("prior", prior),
    ):
        if value is not UNSET:
            raise CapabilityError(
                "switchback contrast readouts do not accept arm role overrides",
                code="readout.contrast.override",
                context={"argument": argument},
            )


@dataclass(frozen=True, slots=True)
class WholeWindowRequest:
    """Static inputs for one Analysis.run() call after metric selection."""

    metrics: tuple[Metric, ...]
    estimands: tuple[str, ...] | None
    value_scale: Mapping[str, ValueScale] | None
    population: Literal["assigned", "triggered"] | None
    decision_method: Method | _Unset
    sensitivity_methods: Sequence[Method] | _Unset
    prior: Prior | None | _Unset


def validate_whole_window(
    req: WholeWindowRequest,
    *,
    src: MomentSource,
    plan: CompiledDecisionPlan,
    design: Randomized | Encouragement | Observational | None,
    experiment: Experiment | None,
    route: RouteName,
) -> None:
    """Validate native and empty seam requests before any moments are read."""
    if route == "artifact" and req.metrics:
        _validate_run_request(
            src,
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
            prior=req.prior,
            metrics=[m.name for m in req.metrics],
            estimands=req.estimands,
            value_scale=req.value_scale,
        )
    if (
        req.population == "triggered"
        and getattr(plan.inference, "registration", None) is None
        and (
            route not in ("native", "artifact") or experiment is None or experiment.trigger is None
        )
    ):
        _refuse(
            _TRIGGER_UNSUPPORTED,
            method="run",
            experiment="<unknown>" if experiment is None else experiment.name,
            trigger=None if experiment is None else experiment.trigger,
            route=route,
            supported_sources=("from_definitions", "from_unit_day_artifact"),
        )
    if route != "native" and req.metrics:
        return
    if route == "native" and req.value_scale:
        _raise_readout("readout.randomized.value_scale")
    _validate_run_request(
        src,
        decision_method=req.decision_method,
        sensitivity_methods=req.sensitivity_methods,
        prior=req.prior,
        metrics=[m.name for m in req.metrics],
        estimands=req.estimands,
        value_scale=req.value_scale if route != "native" else None,
    )


class WholeWindowReadouts:
    """Evidence acquisition for Analysis.run()."""

    def __init__(
        self,
        *,
        src: MomentSource,
        plan: CompiledDecisionPlan,
        design: Randomized | Encouragement | Observational | None,
        experiment: Experiment | None,
        readout_source: Callable[..., MomentSource],
        validate_trigger: Callable[[], None],
        arm_moments: Callable[..., list[LiftEstimate]],
    ) -> None:
        self._src = src
        self._plan = plan
        self._design = design
        self._experiment = experiment
        self._readout_source = readout_source
        self._validate_trigger = validate_trigger
        self._arm_moments = arm_moments

    def run(self, req: WholeWindowRequest) -> LiftEstimates:
        route = _day_axis_source_route(self._src)
        validate_whole_window(
            req,
            src=self._src,
            plan=self._plan,
            design=self._design,
            experiment=self._experiment,
            route=route,
        )
        if getattr(self._plan.inference, "registration", None) is not None:
            result = _as_lift_estimates(
                readouts.run(
                    self._src,
                    metrics=[m.name for m in req.metrics],
                    estimands=req.estimands,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                    prior=req.prior,
                    value_scale=req.value_scale,
                    _population="assigned",
                )
            )
            return _select_population(result, req.population)
        if (
            route in ("native", "artifact")
            and self._experiment is not None
            and self._experiment.trigger is not None
            and req.population in (None, "triggered")
        ):
            validate_populations = getattr(self._src, "validate_populations", None)
            if validate_populations is None:
                _refuse(
                    _TRIGGER_UNSUPPORTED,
                    method="run",
                    experiment=self._experiment.name,
                    trigger=self._experiment.trigger,
                    route=route,
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
            populations = (
                ("triggered",) if req.population == "triggered" else ("assigned", "triggered")
            )
            validate_populations(populations, operation="run")
        design_summary_only = _design_compliance_only(self._src, req.estimands)
        if not req.metrics and not design_summary_only:
            from increment.breakout.estimates import LiftEstimates

            return LiftEstimates()
        if (
            route in ("native", "artifact")
            and self._experiment is not None
            and self._experiment.trigger is not None
            and req.population in (None, "triggered")
        ):
            self._validate_trigger()
        if route != "native":
            return self._run_seam_family(req)
        return self._run_native_family(req, design_summary_only=design_summary_only)

    def _run_seam_family(self, req: WholeWindowRequest) -> LiftEstimates:
        metric_names = [m.name for m in req.metrics]

        def _read(population: Literal["assigned", "triggered"]) -> LiftEstimates:
            source = self._src
            if population == "triggered":
                if self._experiment is None:
                    _refuse(
                        _TRIGGER_UNSUPPORTED,
                        method="run",
                        experiment="<unknown>",
                        trigger=None,
                        route=_day_axis_source_route(self._src),
                        supported_sources=("from_definitions", "from_unit_day_artifact"),
                    )
                source = self._readout_source(
                    population="triggered",
                    selected=req.metrics,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                )
            return _as_lift_estimates(
                self._arm_moments(
                    source,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                    prior=req.prior,
                    metrics=metric_names,
                    estimands=req.estimands,
                    value_scale=req.value_scale,
                    population=population,
                )
            )

        if req.population == "triggered":
            return _read("triggered")
        assigned = _read("assigned")
        if req.population == "assigned" or self._experiment is None:
            return assigned
        if self._experiment.trigger is None:
            return assigned
        return assigned.concat(_read("triggered"))

    def _run_native_family(
        self, req: WholeWindowRequest, *, design_summary_only: bool
    ) -> LiftEstimates:
        if self._experiment is None:
            _refuse(
                _TRIGGER_UNSUPPORTED,
                method="run",
                experiment="<unknown>",
                trigger=None,
                route="native",
                supported_sources=("from_definitions", "from_unit_day_artifact"),
            )
        metric_names = [m.name for m in req.metrics]

        def _run_arm(
            source: MomentSource, population: Literal["assigned", "triggered"]
        ) -> LiftEstimates:
            return _as_lift_estimates(
                self._arm_moments(
                    source,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                    prior=req.prior,
                    metrics=metric_names,
                    estimands=req.estimands,
                    population=population,
                )
            )

        if req.population == "triggered":
            source = self._readout_source(
                population="triggered",
                selected=req.metrics,
                design_summary_only=design_summary_only,
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
            )
            return _run_arm(source, "triggered")
        source = self._readout_source(
            selected=req.metrics,
            design_summary_only=design_summary_only,
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
        )
        assigned = _run_arm(source, "assigned")
        if req.population == "assigned" or self._experiment.trigger is None:
            return assigned
        triggered = self._readout_source(
            population="triggered",
            selected=req.metrics,
            design_summary_only=design_summary_only,
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
        )
        return assigned.concat(_run_arm(triggered, "triggered"))


__all__ = [
    "WholeWindowReadouts",
    "WholeWindowRequest",
    "reject_role_overrides_under_contrast",
    "validate_whole_window",
]
