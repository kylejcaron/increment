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
from typing import TYPE_CHECKING, Any, Literal

from increment._analysis_config import UNSET, _Unset
from increment._day_axis import _TRIGGER_UNSUPPORTED, DayAxisRoute
from increment._evidence_types import CountsRequest
from increment._readout_request import _raise as _raise_readout
from increment.errors import CapabilityError, InvalidRequestError, RefusalSpec
from increment.errors import refuse as _refuse
from increment.readouts._evidence import (
    ARTIFACT_ROUTE,
    NATIVE_ROUTE,
    _builtin_source,
    day_axis_route,
    is_custom_source,
    validate_populations,
    validates_populations,
)
from increment.readouts._requests import (
    _default_sequential_compliance_requested,
    _design_compliance_only,
    _sequential_compliance_requested,
)

if TYPE_CHECKING:
    from increment._literals import ValueScale
    from increment.breakout.estimates import LiftEstimates
    from increment.decision import CompiledDecisionPlan
    from increment.declarations import Method
    from increment.estimation.inference import Prior
    from increment.estimation.results import LiftEstimate
    from increment.kernel.context import EvidenceExecutionContext
    from increment.readouts._run_acquisition import PreparedRunAcquisition
    from increment.semantics.design import Encouragement, Observational, Randomized
    from increment.semantics.models import Experiment, Metric
    from increment.sources import _BuiltinMomentSource

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


def validate_contrast_request(
    *,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
    estimands: Sequence[str] | None,
    value_scale: Mapping[str, ValueScale] | None,
) -> None:
    """Validate saved and immediate arguments for a switchback readout."""
    reject_role_overrides_under_contrast(
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior=prior,
    )
    if estimands is not None or value_scale:
        from increment._evidence_refusals import CONTRAST_UNAVAILABLE

        _refuse(CONTRAST_UNAVAILABLE, method="run")


@dataclass(frozen=True, slots=True)
class WholeWindowRequest:
    """Static inputs for one Analysis.run() call after metric selection."""

    metrics: tuple[Metric, ...]
    estimands: tuple[str, ...] | None
    value_scale: Mapping[str, ValueScale] | None
    population: Literal["assigned", "triggered"] | None
    coherence: Literal["captured", "live"]
    decision_method: Method | _Unset
    sensitivity_methods: Sequence[Method] | _Unset
    prior: Prior | None | _Unset


@dataclass(frozen=True, slots=True)
class WholeWindowExploratoryRequest:
    """A preflighted exploratory reader in the shared execution family."""

    request: WholeWindowRequest
    reader: WholeWindowReadouts


@dataclass(frozen=True, slots=True)
class WholeWindowFamilyRequest:
    """One declared run family and its optional exploratory sibling."""

    declared: WholeWindowRequest
    exploratory: WholeWindowExploratoryRequest | None = None
    capture_requested: bool = False


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
        validate_trigger: Callable[[object], None],
        arm_moments: Callable[..., list[LiftEstimate]],
        execution_context_scope: Callable[[EvidenceExecutionContext], Any],
        registered_run: Callable[..., Any],
    ) -> None:
        self._src = src
        self._plan = plan
        self._design = design
        self._experiment = experiment
        self._readout_source = readout_source
        self._validate_trigger = validate_trigger
        self._arm_moments = arm_moments
        self._execution_context_scope = execution_context_scope
        self._registered_run = registered_run

    def validate(self, req: WholeWindowRequest) -> RouteName:
        """Validate requests and portable configuration snapshots before capture."""
        route = day_axis_route(self._src)
        validate_whole_window(
            req,
            src=self._src,
            plan=self._plan,
            design=self._design,
            experiment=self._experiment,
            route=route,
        )
        if (
            route in (NATIVE_ROUTE, ARTIFACT_ROUTE)
            and getattr(self._plan.inference, "registration", None) is None
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
        return route

    def prepare_family(
        self, family: WholeWindowFamilyRequest
    ) -> tuple[PreparedRunAcquisition, ...]:
        from increment.readouts._run_acquisition import RunAcquisitionOwner, RunAcquisitionRequest

        members = [(self, family.declared)]
        if family.exploratory is not None:
            members.append((family.exploratory.reader, family.exploratory.request))

        prepared = []
        for reader, request in members:
            population = request.population or "assigned"
            has_trigger = reader._experiment is not None and reader._experiment.trigger is not None
            registered = getattr(reader._plan.inference, "registration", None) is not None
            populations = (
                ("assigned", "triggered")
                if request.population is None and has_trigger
                else (population,)
            )
            if population == "triggered" and registered:
                populations = tuple(dict.fromkeys((*populations, "assigned")))
            prepared.append(
                RunAcquisitionOwner.prepare(
                    RunAcquisitionRequest(
                        source=reader._src,
                        decision_method=request.decision_method,
                        sensitivity_methods=request.sensitivity_methods,
                        prior=request.prior,
                        metrics=tuple(metric.name for metric in request.metrics),
                        estimands=request.estimands,
                        value_scale=request.value_scale,
                        request_population="assigned"
                        if population == "triggered" and registered
                        else population,
                        family_population=population,
                        family_populations=populations,
                        family_horizon=tuple(reader._src.context.metrics),
                        design=reader._design,
                        capture_requested=family.capture_requested,
                    )
                )
            )
        for reader, request in members:
            reader.validate(request)
        return tuple(prepared)

    def run_family(
        self,
        family: WholeWindowFamilyRequest,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates:
        """Prepare, capture, execute, and assemble a declared/exploratory family."""
        from increment.breakout.estimates import LiftEstimates, _stamp_exploratory_rows
        from increment.readouts._run_acquisition import RunAcquisitionOwner

        prepared = self.prepare_family(family)

        def execute(context: EvidenceExecutionContext | None) -> LiftEstimates:
            rows = (
                self.run(
                    family.declared,
                    execution_context=context,
                    _prepared=prepared[0],
                )
                if (
                    family.declared.metrics
                    or _design_compliance_only(self._src.context, family.declared.estimands)
                    or (
                        getattr(self._plan.inference, "registration", None) is not None
                        and _default_sequential_compliance_requested(
                            self._src.context,
                            family.declared.metrics,
                            family.declared.estimands,
                        )
                    )
                )
                else LiftEstimates()
            )
            if family.exploratory is not None:
                request = family.exploratory.request
                reader = family.exploratory.reader
                if context is not None:
                    context.declare_source_alias(reader._src, parent=self._src)
                extra = reader.run(
                    request,
                    execution_context=context,
                    _prepared=prepared[1],
                )
                extra = LiftEstimates(
                    [row for row in extra if row.estimand != "compliance"],
                    metadata=extra.metadata,
                    source=extra.source,
                    sequential_snapshot=extra.sequential_snapshot,
                    default_backend=extra._default_backend,
                )
                extra = _stamp_exploratory_rows(extra)
                rows = extra if rows.metadata is None else rows.concat(extra)
            if context is not None and rows.metadata is not None:
                metadata = rows.metadata
                rows._metadata = type(metadata).model_validate(
                    {
                        **metadata.__dict__,
                        "coherence": (
                            None
                            if "provider" in context.effective_modes.values()
                            else context.coherence
                        ),
                        "effective_modes": dict(context.effective_modes),
                    }
                )
            return rows

        def owned_execute() -> LiftEstimates:
            from uuid import uuid4

            return RunAcquisitionOwner.execute(
                self._src,
                prepared,
                run_id=uuid4().hex,
                coherence=family.declared.coherence,
                shared_capture=True,
                executor=execute,
                execution_context_scope=self._execution_context_scope,
            )

        if execution_context is not None:
            return execute(execution_context)
        return owned_execute()

    def run(
        self,
        req: WholeWindowRequest,
        execution_context: EvidenceExecutionContext | None = None,
        *,
        _prepared: PreparedRunAcquisition | None = None,
    ) -> LiftEstimates:
        design_summary_only = _design_compliance_only(self._src.context, req.estimands)
        sequential_compliance = _sequential_compliance_requested(self._src.context, req.estimands)
        registered = getattr(self._plan.inference, "registration", None) is not None
        default_sequential_compliance = registered and _default_sequential_compliance_requested(
            self._src.context, req.metrics, req.estimands
        )
        needs_trigger_acquisition = (
            not is_custom_source(self._src)
            and req.population in (None, "triggered")
            and self._experiment is not None
            and self._experiment.trigger is not None
            and not registered
            and (req.metrics or design_summary_only)
        )
        prepared = _prepared
        if (not req.metrics or needs_trigger_acquisition) and prepared is None:
            from increment.readouts._run_acquisition import (
                RunAcquisitionOwner,
                RunAcquisitionRequest,
            )

            if needs_trigger_acquisition and execution_context is None:
                design = self._design
                scope_population = "triggered" if req.population == "triggered" else "assigned"
                populations = (
                    ("triggered",) if req.population == "triggered" else ("assigned", "triggered")
                )
                uptake = getattr(design, "uptake", None)
                acquisition_request = RunAcquisitionRequest(
                    source=self._src,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                    prior=req.prior,
                    metrics=tuple(metric.name for metric in req.metrics),
                    estimands=req.estimands,
                    value_scale=req.value_scale,
                    request_population=scope_population,
                    family_population=scope_population,
                    family_populations=populations,
                    family_horizon=tuple(self._src.context.metrics),
                    family_uptake_facts=() if uptake is None else (uptake.fact,),
                    design=design,
                )
            else:
                acquisition_request = RunAcquisitionRequest(
                    source=self._src,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                    prior=req.prior,
                    metrics=tuple(metric.name for metric in req.metrics),
                    estimands=req.estimands,
                    value_scale=req.value_scale,
                    request_population=(
                        "assigned" if registered else (req.population or "assigned")
                    ),
                    design=self._design,
                )
            prepared = RunAcquisitionOwner.prepare(acquisition_request)
        route = self.validate(req)
        if (
            not req.metrics
            and not design_summary_only
            and not sequential_compliance
            and not default_sequential_compliance
        ):
            from increment.breakout.estimates import LiftEstimates

            return LiftEstimates()
        if getattr(self._plan.inference, "registration", None) is not None:
            result = _as_lift_estimates(
                self._registered_run(
                    self._src,
                    metrics=[m.name for m in req.metrics],
                    estimands=req.estimands,
                    decision_method=req.decision_method,
                    sensitivity_methods=req.sensitivity_methods,
                    prior=req.prior,
                    value_scale=req.value_scale,
                    _population="assigned",
                    coherence=(
                        req.coherence
                        if execution_context is None
                        else execution_context.requested_coherence
                    ),
                    execution_context=execution_context,
                )
            )
            return _select_population(result, req.population)
        if execution_context is not None:
            has_trigger = self._experiment is not None and self._experiment.trigger is not None
            required_populations = (
                ("assigned", "triggered")
                if req.population is None and has_trigger
                else (req.population or "assigned",)
            )
            for population in required_populations:
                execution_context.validate_population(population)
            if (
                route in (NATIVE_ROUTE, ARTIFACT_ROUTE)
                and self._experiment is not None
                and self._experiment.trigger is not None
                and req.population in (None, "triggered")
            ):
                execution_context.validate_typed_request(
                    CountsRequest(kind="triggered", population="triggered")
                )
                with execution_context.borrow() as pinned:
                    self._validate_trigger(_builtin_source(pinned))
            return self._run_seam_family(req, execution_context)
        if needs_trigger_acquisition:
            from uuid import uuid4

            from increment.readouts._run_acquisition import RunAcquisitionOwner

            assert prepared is not None
            with RunAcquisitionOwner.capture(
                self._src,
                (prepared,),
                run_id=uuid4().hex,
                coherence=req.coherence,
            ) as context:
                with context.borrow() as scoped:
                    self._validate_trigger(_builtin_source(scoped))
                return self._run_seam_family(req, context)
        return self._run_seam_family(req)

    def _run_seam_family(
        self,
        req: WholeWindowRequest,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates:
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
                    execution_context=execution_context,
                    coherence=(
                        req.coherence
                        if execution_context is None
                        else execution_context.requested_coherence
                    ),
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
    "WholeWindowFamilyRequest",
    "WholeWindowReadouts",
    "WholeWindowRequest",
    "reject_role_overrides_under_contrast",
    "validate_contrast_request",
    "validate_whole_window",
]
