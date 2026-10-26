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
from typing import TYPE_CHECKING, Any, Literal, cast

from increment import readouts
from increment._analysis_config import UNSET, _Unset
from increment._day_axis import _TRIGGER_UNSUPPORTED, DayAxisRoute
from increment._evidence_types import SourceContext
from increment._literals import ValueScale
from increment._readout_request import _raise as _raise_readout
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.readouts._evidence import (
    ARTIFACT_ROUTE,
    NATIVE_ROUTE,
    day_axis_route,
    is_custom_source,
    validate_populations,
    validates_populations,
)
from increment.readouts._requests import _design_compliance_only, _validate_run_request
from increment.sources import require_operation

if TYPE_CHECKING:
    from increment.breakout.estimates import LiftEstimates
    from increment.decision import CompiledDecisionPlan
    from increment.declarations import Method
    from increment.estimation.inference import Prior
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Experiment, Metric
    from increment.sources import SourceOperation, _BuiltinMomentSource

RouteName = DayAxisRoute


def _is_artifact_source(src: _BuiltinMomentSource) -> bool:
    return day_axis_route(src) == ARTIFACT_ROUTE


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


_UNKNOWN_METRIC = RefusalSpec(
    "facade.analysis.unknown_metric",
    InvalidRequestError,
    template="metric {metric!r} is not declared on this experiment. Declared metrics: {declared!r}",
)


def _require_analysis_operation[T](
    source: _BuiltinMomentSource,
    operation: SourceOperation,
    protocol: type[T] | None = None,
    *,
    message: str,
) -> T:
    offered = tuple(sorted(cast(Any, source).evidence.operations))
    if not isinstance(getattr(source, "context", None), SourceContext):
        from increment._evidence_refusals import refuse_operation

        refuse_operation(source, operation, offered, route=message)
    if operation not in offered:
        from increment._evidence_refusals import refuse_operation

        refuse_operation(source, operation, offered, route=message)
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
    src: _BuiltinMomentSource,
    plan: CompiledDecisionPlan,
    design: Randomized | Encouragement | Observational | None,
    experiment: Experiment | None,
    route: RouteName,
) -> None:
    """Validate native and empty seam requests before any moments are read."""
    if route == ARTIFACT_ROUTE and req.metrics:
        _validate_run_request(
            src,
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
            prior=req.prior,
            metrics=[m.name for m in req.metrics],
            estimands=req.estimands,
            value_scale=req.value_scale,
        )
    # A registered plan replays both captured populations wherever the source
    # context declares the trigger (a moments checkpoint included); a
    # fixed-horizon triggered read needs the native or artifact trigger evidence.
    registered = getattr(plan.inference, "registration", None) is not None
    triggered_unsupported = (
        src.context.trigger_name is None
        if registered
        else (
            route not in (NATIVE_ROUTE, ARTIFACT_ROUTE)
            or experiment is None
            or experiment.trigger is None
        )
    )
    if req.population == "triggered" and triggered_unsupported:
        _refuse(
            _TRIGGER_UNSUPPORTED,
            method="run",
            experiment="<unknown>" if experiment is None else experiment.name,
            trigger=None if experiment is None else experiment.trigger,
            route=route,
            supported_sources=("from_definitions", "from_unit_day_artifact"),
        )
    if route != NATIVE_ROUTE and req.metrics:
        return
    if route == NATIVE_ROUTE and req.value_scale:
        _raise_readout("readout.randomized.value_scale")
    _validate_run_request(
        src,
        decision_method=req.decision_method,
        sensitivity_methods=req.sensitivity_methods,
        prior=req.prior,
        metrics=[m.name for m in req.metrics],
        estimands=req.estimands,
        value_scale=req.value_scale if route != NATIVE_ROUTE else None,
    )


class WholeWindowReadouts:
    """Evidence acquisition for Analysis.run()."""

    def __init__(
        self,
        *,
        src: _BuiltinMomentSource,
        plan: CompiledDecisionPlan,
        design: Randomized | Encouragement | Observational | None,
        experiment: Experiment | None,
        readout_source: Callable[..., _BuiltinMomentSource],
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
        route = day_axis_route(self._src)
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
            route in (NATIVE_ROUTE, ARTIFACT_ROUTE)
            and self._experiment is not None
            and self._experiment.trigger is not None
            and req.population in (None, "triggered")
        ):
            if not is_custom_source(self._src) and not validates_populations(self._src):
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
            validate_populations(self._src, populations, operation="run")
        design_summary_only = _design_compliance_only(self._src, req.estimands)
        if not req.metrics and not design_summary_only:
            from increment.breakout.estimates import LiftEstimates

            return LiftEstimates()
        if (
            not is_custom_source(self._src)
            and route in (NATIVE_ROUTE, ARTIFACT_ROUTE)
            and self._experiment is not None
            and self._experiment.trigger is not None
            and req.population in (None, "triggered")
        ):
            self._validate_trigger()
        return self._run_seam_family(req)

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
                        route=day_axis_route(self._src),
                        supported_sources=("from_definitions", "from_unit_day_artifact"),
                    )
                source = self._readout_source(population="triggered")
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


__all__ = [
    "WholeWindowReadouts",
    "WholeWindowRequest",
    "reject_role_overrides_under_contrast",
    "validate_whole_window",
]
