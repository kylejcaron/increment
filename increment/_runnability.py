"""Readout compatibility over portable declarations, without source access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import TypeAdapter

from increment._analysis_config import (
    UNSET,
    ResolvedMetricConfig,
    effective_methods,
    normalize_display_correction,
    overlay_configs,
    require_fixed_exploratory_metrics,
    resolve_configs,
    select_metrics,
)
from increment._canonical import declaration_value
from increment._evidence_loaders import day_axis_grains, source_context_payload
from increment._evidence_refusals import _DeclaredEvidence
from increment._evidence_types import SourceContext
from increment._metric_specs import validate_windowed_quantile_support
from increment._readout_request import ReadoutRequest, validate_request
from increment._readout_request import _raise as _raise_readout
from increment._source_types import (
    EvidenceCapabilities,
    declared_day_axis_metric_objects_supported,
    effective_capabilities,
)
from increment.decision import (
    CompiledDecisionPlan,
    margin_less_guardrail_route,
    refuse_margin_less_procedures,
)
from increment.declarations import VARIANCE_REDUCTION, estimator_refs, thaw
from increment.errors import CapabilityError, CodedError, InvalidRequestError, RefusalSpec, refuse
from increment.estimation.priors import prior_from_payload
from increment.plan import with_unassigned_procedures
from increment.semantics.design import SUPPORTED_DESIGN_MECHANISMS, Design, parse_design

if TYPE_CHECKING:
    from increment._evidence_types import BreakoutIdentity
    from increment.semantics.models import Metric

_CONFIGS = TypeAdapter(tuple[ResolvedMetricConfig, ...])


class _DeclaredSource(_DeclaredEvidence):
    __slots__ = ("context",)

    def __init__(
        self, context: SourceContext, evidence: EvidenceCapabilities, constructor: str
    ) -> None:
        super().__init__(evidence, constructor)
        self.context = context


def require_runnable_exploratory_metric(declaration: Mapping[str, Any], name: str) -> None:
    """Apply the registered-plan restriction to an eligible catalog metric."""
    plan = declaration.get("plan")
    if not isinstance(plan, Mapping):
        return
    inference = plan.get("inference")
    if isinstance(inference, Mapping) and inference.get("registration") is not None:
        compiled = CompiledDecisionPlan.model_validate(thaw(plan))
        require_fixed_exploratory_metrics(compiled.inference, [name], caller="runnable")


def _compliance_only_request(
    operation: str,
    context: SourceContext,
    options: Mapping[str, Any],
) -> bool:
    if operation != "run":
        return False
    from increment.readouts._requests import _design_compliance_only

    return _design_compliance_only(context, options.get("estimands"))


_SCOPE_OPERATIONS = {
    "run": ("run",),
    "switchback": ("run",),
    "daily": ("run_daily_lift", "run_daily"),
    "asof": ("run_asof_lift", "run_asof"),
    "breakout": ("run_breakout",),
}
_VALUE_OPERATION_BY_VIEW = {"daily": "run_daily", "asof": "run_asof"}
_VALUE_OPERATIONS = frozenset(_VALUE_OPERATION_BY_VIEW.values())

_SCOPE_FAMILY_MISMATCH = RefusalSpec(
    "kernel.scope.family_mismatch",
    CapabilityError,
    template=(
        "A switchback view requires a switchback experiment; "
        "use run, daily, asof, or breakout with this experiment."
    ),
    keys=frozenset({"experiment_id", "view"}),
)

_UNDECLARED_BREAKOUT = RefusalSpec(
    "facade.analysis.undeclared_breakout",
    InvalidRequestError,
    template="breakout {requested!r} is not declared by this experiment; declared: {declared!r}",
)


def scope_operation(experiment: Any, scope: Any) -> str:
    """Resolve a declared operation with the execution default for its scope."""
    if scope.values:
        return _VALUE_OPERATION_BY_VIEW[scope.view]
    return experiment.operations.get(scope.hash(), _SCOPE_OPERATIONS[scope.view][0])


def records_breakout_sources(experiment: Any) -> bool:
    """Whether the experiment's recorded definitions name each breakout's fact source."""
    recorded = experiment.analysis_context.get("experiment")
    return isinstance(recorded, Mapping) and isinstance(recorded.get("breakouts"), (tuple, list))


def require_breakout_sources(experiment: Any, scope: Any) -> None:
    """Refuse a fact-source qualifier on breakouts declared by source capability alone.

    A capability-declared breakout names no fact source, so no qualifier matches it.
    """
    if records_breakout_sources(experiment):
        return
    qualified = tuple(identity for identity in scope.segment if identity.source is not None)
    if not qualified:
        return
    from increment._evidence_refusals import BREAKOUT_UNAVAILABLE

    refuse(
        BREAKOUT_UNAVAILABLE,
        constructor=experiment.analysis_context.get("source_name")
        or ", ".join(reference.name for reference in experiment.source_refs),
        requested=tuple((identity.source, identity.property) for identity in qualified),
        offered=tuple(
            dict.fromkeys(
                (None, name)
                for reference in experiment.source_refs
                if reference.capabilities is not None
                for name in reference.capabilities.breakouts
            )
        ),
        route="select the breakout by property alone; this source names no fact sources",
    )


def validate_breakout_identities(experiment: Any, scope: Any) -> tuple[BreakoutIdentity, ...]:
    """The breakout identities a scope selects, refusing a fact source no declaration names.

    A selection by property alone matches every declared source of that property. A
    property no declaration names is left to the evidence validators, which refuse it
    with the source's declared dimensions.
    """
    from increment.kernel.contracts import View, declared_breakouts

    identities = declared_breakouts(experiment.source_refs, scope, experiment.analysis_context)
    require_breakout_sources(experiment, scope)
    if not records_breakout_sources(experiment):
        return identities
    declared = declared_breakouts(
        experiment.source_refs, View(view="breakout"), experiment.analysis_context
    )
    properties = {item.property for item in declared}
    if all(identity.property in properties for identity in identities) and any(
        identity not in declared for identity in identities
    ):
        refuse(
            _UNDECLARED_BREAKOUT,
            requested=tuple((identity.source, identity.property) for identity in identities),
            declared=tuple((item.source, item.property) for item in declared),
        )
    return identities


# The public builder that constructs each declared source kind.
_BUILDERS = MappingProxyType(
    {
        "unit_summary": "from_frame",
        "unit_panel": "from_frame",
        "switchback": "from_switchback_frame",
        "definitions": "from_definitions",
        "artifact": "from_unit_day_artifact",
        "moments": "from_moments",
    }
)


def _declared_capabilities(experiment: Any) -> tuple[EvidenceCapabilities, ...] | None:
    """Every source reference's recorded capabilities, or None while any is unknown."""
    capabilities = tuple(reference.capabilities for reference in experiment.source_refs)
    if not capabilities or any(declared is None for declared in capabilities):
        return None
    return cast("tuple[EvidenceCapabilities, ...]", capabilities)


def _refuse_trigger(experiment: Any, scope: Any, operation: str) -> None:
    """Refuse a triggered view that no source can read.

    Without a recorded trigger every view refuses. Declared populations govern
    day-axis and breakout views. Whole-window reads require effective
    ``fixed_triggered`` support or registered replay, whose missing triggered
    state remains an unavailable row.
    """
    if scope.population != "triggered":
        return
    recorded = experiment.analysis_context
    if recorded.get("family") == "contrast":
        # A switchback value view refuses as contrast-unavailable instead.
        if operation in _VALUE_OPERATIONS:
            return
        route = "switchback"
    else:
        capabilities = _declared_capabilities(experiment)
        if capabilities is None or "trigger_name" not in recorded:
            return
        from increment.readouts._evidence import declared_day_axis_route

        route = declared_day_axis_route(capabilities[0])
        inference = recorded.get("plan", {}).get("inference") or {}
        registered = inference.get("registration") is not None
        if operation != "run":
            if any("triggered" in declared.populations for declared in capabilities):
                return
        elif recorded["trigger_name"] is not None and (
            registered or effective_capabilities(capabilities[0]).fixed_triggered
        ):
            return
    from increment._evidence_refusals import _TRIGGER_UNSUPPORTED

    refuse(
        _TRIGGER_UNSUPPORTED,
        method=operation,
        experiment=experiment.experiment_id,
        trigger=None,
        route=route,
        supported_sources=("from_definitions", "from_unit_day_artifact"),
    )


def _requested_breakouts(experiment: Any, scope: Any) -> tuple[str, ...]:
    """The breakout properties a view reads: its selection, else its registered segment family's."""
    if scope.segment:
        return tuple(identity.property for identity in scope.segment)
    inference = experiment.analysis_context.get("plan", {}).get("inference") or {}
    roster = (inference.get("registration") or {}).get("roster", ())
    return tuple(sorted({key for cell in roster for key, _ in cell.get("segment", ())}))


def _refuse_undeclared_breakouts(experiment: Any, scope: Any) -> None:
    """Refuse a breakout view on an experiment whose declarations name no breakout."""
    from increment.kernel.contracts import View, declared_breakouts

    capabilities = _declared_capabilities(experiment)
    if capabilities is None or declared_breakouts(
        experiment.source_refs, View(view="breakout"), experiment.analysis_context
    ):
        return
    from increment._evidence_refusals import BREAKOUT_UNAVAILABLE

    refuse(
        BREAKOUT_UNAVAILABLE,
        constructor=_BUILDERS.get(capabilities[0].kind, capabilities[0].kind),
        requested=_requested_breakouts(experiment, scope),
        offered=(),
        route="declare breakouts=[...] on from_frame, or on the experiment in the definitions",
    )


def scope_refusals(experiment: Any, scope: Any) -> tuple[CodedError, ...]:
    """Validate supported execution coordinates without acquiring evidence."""
    operation = scope_operation(experiment, scope)
    if (
        operation not in _SCOPE_OPERATIONS[scope.view]
        or scope.through_day is not None
        or scope.day_domain
        or scope.time_basis != "calendar"
        or (scope.completion_policy and scope.view != "asof")
        or (scope.view == "switchback" and bool(scope.segment))
    ):
        return (
            CodedError(
                "The bound analysis cannot execute these experiment declarations",
                code="kernel.run.execution_unavailable",
                context={"experiment_id": experiment.experiment_id, "field": "scope"},
            ),
        )
    contrast = experiment.analysis_context.get("family") == "contrast"
    try:
        _refuse_trigger(experiment, scope, operation)
        if contrast and operation != "run":
            from increment._evidence_refusals import CONTRAST_UNAVAILABLE

            refuse(CONTRAST_UNAVAILABLE, method=operation)
        validate_breakout_identities(experiment, scope)
        if scope.view == "breakout":
            _refuse_undeclared_breakouts(experiment, scope)
        if scope.view == "switchback" and not contrast:
            refuse(
                _SCOPE_FAMILY_MISMATCH,
                experiment_id=experiment.experiment_id,
                view=scope.view,
            )
        operation_options(experiment, scope)
    except CodedError as refusal:
        return (refusal,)
    return ()


def default_metrics(experiment: Any, scope: Any) -> list[Any]:
    """The metrics a view reads when the request selects none: its own, else the experiment's."""
    if scope.view == "breakout" and scope.metrics:
        from increment.semantics.models import Metric

        declared = TypeAdapter(tuple[Metric, ...]).validate_python(
            thaw(tuple(experiment.metrics.values()))
        )
        return select_metrics(
            declared, scope.metrics, caller="breakout", require_declared_definitions=True
        )
    return list(scope.metrics) or list(experiment.metrics)


def operation_options(
    experiment: Any, scope: Any, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize and validate saved or immediate readout arguments.

    ``coherence`` belongs to the run's shared capture: a view that takes no
    coherence of its own reads under the run's, so it is not a view argument.
    """
    values = {**typed_request_options(experiment.request_options), **(overrides or {})}
    if values.get("metrics") is None:
        values["metrics"] = default_metrics(experiment, scope)
    if "population" not in values:
        values["population"] = scope.population
    if scope.view in {"daily", "asof"}:
        values["dimension"] = scope.segment[0].property if scope.segment else None
    estimands = values.get("estimands")
    design_payload = experiment.design
    mechanism = design_payload.get("mechanism")
    if mechanism is not None and mechanism not in SUPPORTED_DESIGN_MECHANISMS:
        parse_design(design_payload)
    is_encouragement = mechanism == "encouragement"
    if (
        is_encouragement
        and estimands is not None
        and tuple(estimands) == ("compliance",)
        and not values.get("dimension")
    ):
        values["metrics"] = []
    if scope.view == "asof":
        values["completed_windows_only"] = scope.completion_policy
    operation = scope_operation(experiment, scope)
    declaration = experiment.analysis_context.get("operation_parameters", {}).get(operation)
    if declaration is not None:
        if not declaration["variadic"] and "coherence" not in declaration["accepted"]:
            values.pop("coherence", None)
        unexpected = (
            set(values) - set(declaration["accepted"]) if not declaration["variadic"] else set()
        )
        missing = set(declaration["required"]) - set(values)
        if unexpected or missing:
            raise CodedError(
                "The bound analysis cannot execute these experiment declarations",
                code="kernel.run.execution_unavailable",
                context={
                    "experiment_id": experiment.experiment_id,
                    "field": "request_options",
                    "unexpected": tuple(sorted(unexpected)),
                    "missing": tuple(sorted(missing)),
                },
            )
    # `store` stays declared for saved experiments; a set store is refused before
    # execution, and no view receives it.
    values.pop("store", None)
    return values


def context_payload(source: Any) -> dict[str, Any]:
    """Resolve source-owned declaration metadata through evidence acquisition."""

    return source_context_payload(source)


def refuse_recorded_margin_less_guardrail(
    analysis_context: Mapping[str, Any], design: object
) -> None:
    """Refuse a recorded guardrail that compiling its plan now refuses for lacking a margin,
    with the route compiling the saved design and inference names."""
    recorded = analysis_context.get(
        "context" if analysis_context.get("family") == "contrast" else "plan"
    )
    if not isinstance(recorded, Mapping):
        return
    inference = recorded.get("inference")
    refuse_margin_less_procedures(
        recorded.get("procedures"),
        route=margin_less_guardrail_route(
            sequential=isinstance(inference, Mapping) and inference.get("kind") != "fixed",
            encouragement=isinstance(design, Mapping)
            and design.get("mechanism") == "encouragement",
        ),
    )


def _estimator_config(
    config: ResolvedMetricConfig,
    design: Design,
    estimator_id: str,
) -> ResolvedMetricConfig:
    from increment.estimation.engine import Method

    estimator = estimator_refs().get(estimator_id)
    if estimator is None:
        return config
    selector = estimator.selector or "unadjusted"
    methods = effective_methods(config, design=design)
    matching = next(
        (
            method
            for method in methods
            if (
                VARIANCE_REDUCTION.get(method.variance_reduction) == estimator_id
                if estimator_id.startswith("randomized.")
                else method.name == selector
            )
        ),
        None,
    )
    method = matching or (
        Method(
            name=estimator.selector or "unadjusted", variance_reduction=estimator.selector or "none"
        )
        if estimator_id.startswith("randomized.")
        else Method(name=selector)
    )
    return replace(config, decision_method=method, sensitivity_methods=(), decision_defaulted=False)


def typed_request_options(options: Mapping[str, Any]) -> dict[str, Any]:
    """Restore portable request declarations to their runtime models."""
    from increment.estimation.engine import Method
    from increment.semantics.models import Metric

    values = thaw(options)
    method = values.get("decision_method")
    if isinstance(method, Mapping):
        values["decision_method"] = Method.model_validate(method)
    methods = values.get("sensitivity_methods")
    if isinstance(methods, (tuple, list)):
        values["sensitivity_methods"] = tuple(
            Method.model_validate(method) if isinstance(method, Mapping) else method
            for method in methods
        )
    metrics = values.get("metrics")
    if isinstance(metrics, (tuple, list)):
        values["metrics"] = [
            TypeAdapter(Metric).validate_python(metric) if isinstance(metric, Mapping) else metric
            for metric in metrics
        ]
    prior = values.get("prior")
    if isinstance(prior, Mapping):
        values["prior"] = prior_from_payload(prior)
    return values


def contrast_request_refusals(options: Mapping[str, Any]) -> tuple[CodedError, ...]:
    from increment.readouts._whole_window import validate_contrast_request

    values = typed_request_options(options)
    try:
        validate_contrast_request(
            decision_method=values.get("decision_method", UNSET),
            sensitivity_methods=values.get("sensitivity_methods", UNSET),
            prior=values.get("prior", UNSET),
            estimands=values.get("estimands"),
            value_scale=values.get("value_scale"),
        )
    except CodedError as error:
        return (error,)
    return ()


def _request_config(
    declaration: Mapping[str, Any],
    options: Mapping[str, Any],
    design: Design,
    estimator_id: str,
    operation: str,
    metric_name: str,
    *,
    capabilities: EvidenceCapabilities,
    exploratory_metric: Metric | None,
) -> tuple[dict[str, Any], tuple[ResolvedMetricConfig, ...], tuple[ResolvedMetricConfig, ...]]:
    values = typed_request_options(options)
    recorded_configs = tuple(
        {**config, "prior": prior_from_payload(prior)}
        if isinstance(config, Mapping)
        and isinstance((prior := config.get("prior")), Mapping)
        and ("type" in prior or "distribution" in prior)
        else config
        for config in declaration["configs"]
    )
    base = _CONFIGS.validate_python(recorded_configs)
    declared = base
    metrics = select_metrics(
        tuple(config.metric for config in base),
        values.get("metrics"),
        caller=operation,
        allow_undeclared_objects=(
            operation in {"run_daily", "run_daily_lift", "run_asof", "run_asof_lift"}
            and declared_day_axis_metric_objects_supported(capabilities)
        ),
    )
    if exploratory_metric is not None:
        metrics.append(exploratory_metric)
        experiment = declaration.get("experiment")
        periods = experiment.get("n_pre_periods") if isinstance(experiment, Mapping) else None
        base += resolve_configs(
            (exploratory_metric,),
            None,
            None,
            methods=None,
            prior=None,
            covariate_available_metrics=(exploratory_metric.name,)
            if isinstance(periods, int) and periods > 0
            else (),
        )
    overrides = {} if operation in _VALUE_OPERATIONS else values
    base_names = {config.metric.name for config in base}
    missing = tuple(metric for metric in metrics if metric.name not in base_names)
    request_base = (
        (*base, *resolve_configs(missing, None, None, methods=None, prior=None))
        if missing
        else base
    )
    configs = overlay_configs(
        metrics,
        request_base,
        methods=None,
        prior=None,
        decision_method=overrides.get("decision_method", UNSET),
        sensitivity_methods=overrides.get("sensitivity_methods", UNSET),
        prior_override=overrides.get("prior", UNSET),
    )
    configs = tuple(
        _estimator_config(config, design, estimator_id)
        if config.metric.name == metric_name and operation not in _VALUE_OPERATIONS
        else config
        for config in configs
    )
    return values, declared, configs


def _validate_declared_day_axis(
    declaration: Mapping[str, Any],
    request: ReadoutRequest,
    operation: str,
    capabilities: EvidenceCapabilities,
) -> None:
    from increment._day_axis import validate_day_axis, validate_lift_request
    from increment._evidence_types import DayAxisRequest
    from increment.readouts._evidence import (
        declared_day_axis_dimensions,
        declared_day_axis_route,
    )
    from increment.semantics.models import Experiment

    experiment = declaration.get("experiment")
    typed_experiment = None if experiment is None else Experiment.model_validate(dict(experiment))

    validate = validate_lift_request if operation.endswith("_lift") else validate_day_axis
    validate(
        DayAxisRequest(
            caller=cast(
                Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"],
                operation,
            ),
            grain=cast(Literal["daily", "asof"], request.grain),
            metrics=request.metrics,
            dimension=request.dimension,
            population=request.population,
            estimands=request.estimands,
            completed_windows_only=request.completion_policy,
            declared_dimensions=declared_day_axis_dimensions(capabilities, typed_experiment),
        ),
        src=_DeclaredSource(request.context, capabilities, declaration["source_name"]),
        design=request.design,
        plan=request.plan,
        route=declared_day_axis_route(capabilities),
        experiment=typed_experiment,
    )


def _validate_declared_scalar_route(
    request: ReadoutRequest, operation: str, capabilities: EvidenceCapabilities
) -> None:
    from increment._analysis_config import DAY_AXIS_ROUTES, validate_scalar_route
    from increment.readouts._common import _sequential_inference

    scalar = capabilities.scalar
    validate_scalar_route(
        request.configs,
        route=DAY_AXIS_ROUTES[operation],
        representation="registered"
        if _sequential_inference(request.plan) is not None
        else "implicit",
        registered_plan=request.plan,
        design=request.design,
        covariate_moments_available=(
            scalar is None
            or scalar.covariate_grains is None
            or request.grain in scalar.covariate_grains
        ),
    )


def _validate_scope_request(
    declaration: Mapping[str, Any],
    request: ReadoutRequest,
    scope: Any,
    operation: str,
    capabilities: EvidenceCapabilities,
    check: Any,
    refusals: list[CodedError],
) -> None:
    if scope.view == "daily":
        check(_validate_declared_day_axis, declaration, request, operation, capabilities)
        if refusals:
            return
        check(validate_request, request)
    elif scope.view == "asof":
        check(_validate_declared_day_axis, declaration, request, operation, capabilities)
    elif scope.view == "breakout":
        registration = getattr(request.plan.inference, "registration", None)
        if registration is not None and request.by:
            from increment.sequential_source import require_registered_breakout_selection

            check(
                require_registered_breakout_selection,
                registration,
                request.by,
                population=scope.population,
            )
        if not refusals:
            check(
                validate_windowed_quantile_support,
                request.metrics,
                supported=effective_capabilities(capabilities).windowed_breakout_quantiles,
            )
        if not refusals:
            check(validate_request, request)
    else:
        check(validate_request, request)
    if operation == "run_asof_lift" and not refusals:
        check(_validate_declared_scalar_route, request, operation, capabilities)
    if operation == "run_asof_lift" and not refusals:
        validation_request = request
        if capabilities.kind == "moments" and capabilities.sequential_capture == "replay":
            validation_request = replace(request, population="assigned")
        check(validate_request, validation_request)


def readout_refusals(
    *,
    estimator_id: str,
    design: Mapping[str, Any],
    declaration: Mapping[str, Any],
    capabilities: EvidenceCapabilities,
    scope: Any,
    operation: str,
    metric_name: str,
    options: Mapping[str, Any],
    exploratory_metric: Metric | None = None,
) -> tuple[CodedError, ...]:
    """Use the execution validators on recorded metric, method and view declarations."""
    declaration = thaw(declaration)
    decoded_design = parse_design(declaration_value(design))
    try:
        options, base_configs, configs = _request_config(
            declaration,
            options,
            decoded_design,
            estimator_id,
            operation,
            metric_name,
            capabilities=capabilities,
            exploratory_metric=exploratory_metric,
        )
        plan = CompiledDecisionPlan.model_validate(dict(declaration["plan"]))
        if exploratory_metric is not None:
            plan = with_unassigned_procedures(plan, (exploratory_metric,), design=decoded_design)
        if scope.view in {"daily", "asof"} and declared_day_axis_metric_objects_supported(
            capabilities
        ):
            plan = with_unassigned_procedures(
                plan, tuple(config.metric for config in configs), design=None
            )
    except CodedError as refusal:
        return (refusal,)
    context = SourceContext(
        study_id=declaration["study_id"],
        design=decoded_design,
        plan=plan,
        metrics=tuple(config.metric for config in base_configs),
        configs=base_configs,
        cluster=declaration["cluster"],
        trigger_name=declaration["trigger_name"],
    )
    estimands = options.get("estimands")
    if _compliance_only_request(operation, context, options):
        configs = ()
    scalar = capabilities.scalar
    by = tuple(identity.property for identity in scope.segment)
    request = ReadoutRequest(
        context=context,
        metrics=tuple(config.metric for config in configs),
        configs=configs,
        view=scope.view,
        grain=scope.grain,
        by=by,
        dimension=by[0] if by else None,
        available_covariates=capabilities.available_covariates,
        scalar_bindings=None if scalar is None else scalar.bindings,
        scalar_moments_only=False if scalar is None else scalar.moments_only,
        scalar_recipe_fixed=False if scalar is None else scalar.recipe_fixed,
        scalar_recipe_configs=() if scalar is None else scalar.recipe_configs,
        scalar_anonymous_metrics=frozenset() if scalar is None else scalar.anonymous_metrics,
        scalar_method_evidence=False if scalar is None else scalar.method_evidence,
        scalar_breakout_evidence=False if scalar is None else scalar.breakout_evidence,
        capabilities=(
            day_axis_grains(_DeclaredSource(context, capabilities, declaration["source_name"]))
            if scope.view in {"daily", "asof"}
            else capabilities.grains
        ),
        grain_route=capabilities.grain_route or "use a source with the requested day axis",
        source_breakouts=capabilities.breakouts,
        source_name=declaration["source_name"],
        shape=capabilities.shape,
        cluster=context.cluster,
        population=scope.population,
        estimands=None if estimands is None else tuple(estimands),
        correction=normalize_display_correction(
            plan.view_policies.for_view(
                "breakout", mechanism=getattr(context.design, "mechanism", None)
            ).correction
        )
        if scope.view == "breakout"
        else None,
        q=plan.q,
        value_scale=options.get("value_scale") or {},
        completion_policy=scope.completion_policy,
        daily_lift=operation == "run_daily_lift",
    )
    refusals = []

    def check(validate, *args, **kwargs):
        try:
            validate(*args, **kwargs)
        except CodedError as error:
            if not any(
                error.code == prior.code and error.context == prior.context for prior in refusals
            ):
                refusals.append(error)

    _validate_scope_request(declaration, request, scope, operation, capabilities, check, refusals)
    if refusals:
        return tuple(refusals)
    if (
        options.get("value_scale")
        and not effective_capabilities(capabilities).value_scale_overrides
    ):
        try:
            _raise_readout("readout.randomized.value_scale")
        except CodedError as error:
            refusals.append(error)

    from increment.readouts._common import _sequential_inference

    inference = _sequential_inference(plan)
    if inference is not None:
        from increment._evidence_refusals import require_evidence_declaration
        from increment._evidence_types import SequentialCaptureRequest
        from increment.sequential_source import validate_sequential_plan

        check(validate_sequential_plan, plan, context.metrics, context.design)
        check(
            require_evidence_declaration,
            capabilities,
            constructor=declaration["source_name"],
            request=SequentialCaptureRequest(finalized=True),
        )
        capture = options.get("capture")
        if capture is not None:
            from increment._window import resolve_window_days
            from increment.sequential_source import (
                validate_capture_request,
                validate_capture_window,
            )

            check(
                validate_capture_request,
                TypeAdapter(SequentialCaptureRequest).validate_python(dict(capture)),
                dynamic=capabilities.sequential_capture == "asof",
            )
            for model in inference.registration.models:
                target = next((m for m in context.metrics if m.name == model.metric), None)
                if target is not None and model.observable == "outcome":
                    check(
                        validate_capture_window,
                        resolve_window_days(target),
                        inference.registration,
                        detail="all observation windows must fit the registered reveal window",
                    )
    if operation in _VALUE_OPERATIONS:
        return tuple(refusals)
    if operation == "run_daily_lift":
        from increment.readouts._daily import _validate_daily_inference

        check(_validate_daily_inference, inference)
        check(_validate_declared_scalar_route, request, operation, capabilities)
    if getattr(context.design, "mechanism", None) == "observational":
        from increment.estimation._readout_refusals import refuse_observational_quantile

        for metric in request.metrics:
            if metric.type == "quantile":
                check(refuse_observational_quantile, metric)
    if (
        not refusals
        and operation == "run"
        and (capabilities.kind == "artifact" or "artifact_experiment" in capabilities.operations)
    ):
        check(validate_windowed_quantile_support, request.metrics, supported=False)
    return tuple(refusals)
