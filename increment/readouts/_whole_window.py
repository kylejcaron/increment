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

from increment._analysis_config import UNSET, _Unset
from increment._day_axis import _TRIGGER_UNSUPPORTED, DayAxisRoute
from increment._evidence_types import CountsRequest, EvidenceScope, SourceContext
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
    _design_compliance_only,
    _prepare_run_request,
    _validate_run_request,
)
from increment.sources import require_operation

if TYPE_CHECKING:
    from increment._literals import ValueScale
    from increment._source_types import EvidenceProvider
    from increment.breakout.estimates import LiftEstimates
    from increment.decision import CompiledDecisionPlan
    from increment.declarations import Method
    from increment.estimation.inference import Prior
    from increment.estimation.results import LiftEstimate
    from increment.kernel.context import EvidenceExecutionContext
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
    declaration = cast(Any, source).evidence
    if (
        operation == "allocation_history"
        and "allocation_history" not in declaration.operations
        and declaration.assignment_relation == "reduced"
    ):
        from increment._evidence_refusals import refuse_assignment_identity

        refuse_assignment_identity(source, operation=operation)
    offered = tuple(sorted(declaration.operations))
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
    """A preflighted exploratory reader and its in-scope reader factory."""

    request: WholeWindowRequest
    reader: WholeWindowReadouts
    make_reader: Callable[[], WholeWindowReadouts]


@dataclass(frozen=True, slots=True)
class WholeWindowFamilyRequest:
    """One declared run family and its optional exploratory sibling."""

    declared: WholeWindowRequest
    exploratory: WholeWindowExploratoryRequest | None = None


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
        validate_trigger: Callable[[object], None],
        arm_moments: Callable[..., list[LiftEstimate]],
        execution_context_scope: Callable[[EvidenceExecutionContext], Any],
        provider_for: Callable[[_BuiltinMomentSource], EvidenceProvider],
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
        self._provider_for = provider_for
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
        _, configs = _prepare_run_request(
            self._src,
            decision_method=req.decision_method,
            sensitivity_methods=req.sensitivity_methods,
            prior=req.prior,
            metrics=[metric.name for metric in req.metrics],
            estimands=req.estimands,
            value_scale=req.value_scale,
        )
        from increment.readouts._design_scope import _config_snapshot

        for config in configs:
            _config_snapshot(config, self._design)
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

    def run_family(
        self,
        family: WholeWindowFamilyRequest,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates:
        """Prepare, capture, execute, and assemble a declared/exploratory family."""
        from increment._evidence_scopes import capture_requirements
        from increment.breakout.estimates import LiftEstimates, _stamp_exploratory_rows

        members = [(self, family.declared)]
        if family.exploratory is not None:
            members.append((family.exploratory.reader, family.exploratory.request))

        scopes = []
        for reader, request in members:
            reader.validate(request)
            population = request.population or "assigned"
            selected, configs = _prepare_run_request(
                reader._src,
                decision_method=request.decision_method,
                sensitivity_methods=request.sensitivity_methods,
                prior=request.prior,
                metrics=[metric.name for metric in request.metrics],
                estimands=request.estimands,
                value_scale=request.value_scale,
                population=(
                    "assigned"
                    if population == "triggered"
                    and getattr(reader._plan.inference, "registration", None) is not None
                    else population
                ),
            )
            compliance_only = _design_compliance_only(reader._src.context, request.estimands)
            has_trigger = reader._experiment is not None and reader._experiment.trigger is not None
            populations = (
                ("assigned", "triggered")
                if request.population is None and has_trigger
                else (population,)
            )
            scopes.append(
                EvidenceScope(
                    metrics=() if compliance_only else tuple(selected),
                    horizon_metrics=() if compliance_only else tuple(reader._src.context.metrics),
                    configs=tuple(configs),
                    population=population,
                    populations=populations,
                    grain="total",
                    snapshot=True,
                    execution=True,
                    include_breakouts=False,
                )
            )
        scope = capture_requirements(self._src, scopes)

        def execute(context: EvidenceExecutionContext | None) -> LiftEstimates:
            rows = (
                self.run(family.declared, execution_context=context)
                if family.declared.metrics
                or _design_compliance_only(self._src.context, family.declared.estimands)
                else LiftEstimates()
            )
            if family.exploratory is not None:
                request = family.exploratory.request
                reader = family.exploratory.make_reader()
                if context is not None:
                    context.declare_source_alias(reader._src, parent=self._src)
                extra = reader.run(request, execution_context=context)
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

        from uuid import uuid4

        from increment.kernel.context import EvidenceExecutionContext
        from increment.readouts._execution_context import owned_execution_context

        with owned_execution_context(
            lambda: EvidenceExecutionContext(
                self._provider_for(self._src),
                scope,
                run_id=uuid4().hex,
                coherence=family.declared.coherence,
                shared_capture=True,
            ),
            execution_context_scope=self._execution_context_scope,
        ) as context:
            return execute(context)

    def run(
        self,
        req: WholeWindowRequest,
        execution_context: EvidenceExecutionContext | None = None,
    ) -> LiftEstimates:
        route = self.validate(req)
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
        design_summary_only = _design_compliance_only(self._src.context, req.estimands)
        if not req.metrics and not design_summary_only:
            from increment.breakout.estimates import LiftEstimates

            return LiftEstimates()
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
        if (
            not is_custom_source(self._src)
            and req.population in (None, "triggered")
            and self._experiment is not None
            and self._experiment.trigger is not None
            and getattr(self._plan.inference, "registration", None) is None
            and (req.metrics or design_summary_only)
        ):
            from uuid import uuid4

            from increment.kernel.context import EvidenceExecutionContext
            from increment.readouts._design_scope import _config_snapshot
            from increment.readouts._provider import native_scope_provider

            design = self._design
            selected, configs = _prepare_run_request(
                self._src,
                decision_method=req.decision_method,
                sensitivity_methods=req.sensitivity_methods,
                prior=req.prior,
                metrics=[metric.name for metric in req.metrics],
                estimands=req.estimands,
                value_scale=req.value_scale,
                population="triggered" if req.population == "triggered" else "assigned",
            )
            for config in configs:
                _config_snapshot(config, design)
            populations = (
                ("triggered",) if req.population == "triggered" else ("assigned", "triggered")
            )
            uptake = getattr(design, "uptake", None)
            provider = (
                self._src if is_custom_source(self._src) else native_scope_provider(self._src)
            )
            context = EvidenceExecutionContext(
                provider,
                EvidenceScope(
                    metrics=tuple(selected),
                    population="triggered" if req.population == "triggered" else "assigned",
                    populations=populations,
                    configs=tuple(configs),
                    uptake_facts=() if uptake is None else (uptake.fact,),
                    snapshot=True,
                    execution=True,
                    include_breakouts=False,
                ),
                run_id=uuid4().hex,
                coherence=req.coherence,
            )
            with context.evidence_scope() as scoped:
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
