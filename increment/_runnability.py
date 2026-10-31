"""Readout compatibility over portable declarations, without source access."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from typing import Any, Literal, cast

from pydantic import TypeAdapter

from increment._analysis_config import (
    UNSET,
    ResolvedMetricConfig,
    effective_methods,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._canonical import declaration_value
from increment._evidence_types import SourceContext
from increment._readout_request import ReadoutRequest, validate_request
from increment._source_types import EvidenceCapabilities
from increment.decision import CompiledDecisionPlan
from increment.declarations import VARIANCE_REDUCTION, estimator_refs, thaw
from increment.errors import CapabilityError, CodedError, RefusalSpec, refuse
from increment.semantics.design import Design

_CONFIGS = TypeAdapter(tuple[ResolvedMetricConfig, ...])
_DESIGN = TypeAdapter(Design)


@dataclass(frozen=True)
class _DeclaredSource:
    context: SourceContext
    evidence: EvidenceCapabilities


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
_VALUE_OPERATIONS = frozenset({"run_daily", "run_asof"})

_SCOPE_FAMILY_MISMATCH = RefusalSpec(
    "kernel.scope.family_mismatch",
    CapabilityError,
    template=(
        "A switchback view requires a switchback experiment; "
        "use run, daily, asof, or breakout with this experiment."
    ),
    keys=frozenset({"experiment_id", "view"}),
)


def scope_operation(experiment: Any, scope: Any) -> str:
    """Resolve a declared operation with the execution default for its scope."""
    return experiment.operations.get(scope.hash(), _SCOPE_OPERATIONS[scope.view][0])


def scope_refusals(experiment: Any, scope: Any) -> tuple[CodedError, ...]:
    """Validate supported execution coordinates without acquiring evidence."""
    operation = scope_operation(experiment, scope)
    if (
        operation not in _SCOPE_OPERATIONS[scope.view]
        or scope.through_day is not None
        or scope.day_domain
        or scope.time_basis != "calendar"
        or (scope.completion_policy and scope.view != "asof")
        or len(scope.segment) > 1
        or any(identity.source is not None for identity in scope.segment)
        or (scope.view == "switchback" and bool(scope.segment))
    ):
        return (
            CodedError(
                "The bound analysis cannot execute these experiment declarations",
                code="kernel.run.execution_unavailable",
                context={"experiment_id": experiment.experiment_id, "field": "scope"},
            ),
        )
    if scope.view == "breakout":
        properties = tuple(identity.property for identity in scope.segment)
        for source in experiment.source_refs:
            if (
                source.capabilities is not None
                and len(source.capabilities.breakouts) > 1
                and len(properties) == 1
                and properties[0] in source.capabilities.breakouts
            ):
                return (
                    CodedError(
                        "The bound analysis cannot execute these experiment declarations",
                        code="kernel.run.execution_unavailable",
                        context={"experiment_id": experiment.experiment_id, "field": "segment"},
                    ),
                )
    if scope.view == "switchback" and experiment.analysis_context.get("family") != "contrast":
        try:
            refuse(
                _SCOPE_FAMILY_MISMATCH,
                experiment_id=experiment.experiment_id,
                view=scope.view,
            )
        except CodedError as refusal:
            return (refusal,)
    if experiment.analysis_context.get("family") == "contrast":
        from increment._day_axis import _TRIGGER_UNSUPPORTED
        from increment._evidence_refusals import CONTRAST_UNAVAILABLE

        try:
            if operation != "run":
                refuse(CONTRAST_UNAVAILABLE, method=operation)
            if scope.population == "triggered":
                refuse(
                    _TRIGGER_UNSUPPORTED,
                    method="run",
                    experiment="<unknown>",
                    trigger=None,
                    route="switchback",
                    supported_sources=("from_definitions", "from_unit_day_artifact"),
                )
        except CodedError as refusal:
            return (refusal,)
    try:
        operation_options(experiment, scope)
    except CodedError as refusal:
        return (refusal,)
    return ()


def operation_parameters(analysis: Any) -> dict[str, Any]:
    """Record the keyword contracts of the bound readout operations."""
    declarations = {}
    for operation in dict.fromkeys(name for names in _SCOPE_OPERATIONS.values() for name in names):
        execute = analysis._run_legacy if operation == "run" else getattr(analysis, operation)
        parameters = tuple(signature(execute).parameters.values())
        keywords = tuple(
            parameter
            for parameter in parameters
            if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        )
        declarations[operation] = {
            "accepted": (*tuple(parameter.name for parameter in keywords), "capture"),
            "required": tuple(
                parameter.name for parameter in keywords if parameter.default is Parameter.empty
            ),
            "variadic": any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters),
        }
    return declarations


def operation_options(
    experiment: Any, scope: Any, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize and validate saved or immediate readout arguments."""
    values = {**typed_request_options(experiment.request_options), **(overrides or {})}
    if values.get("metrics") is None:
        values["metrics"] = list(scope.metrics) or list(experiment.metrics)
    estimands = values.get("estimands")
    if (
        experiment.design.get("mechanism") == "encouragement"
        and estimands is not None
        and tuple(estimands) == ("compliance",)
    ):
        values["metrics"] = []
    if "population" not in values:
        values["population"] = scope.population
    if scope.view in {"daily", "asof"}:
        values["dimension"] = scope.segment[0].property if scope.segment else None
    if scope.view == "asof":
        values["completed_windows_only"] = scope.completion_policy
    operation = scope_operation(experiment, scope)
    declaration = experiment.analysis_context.get("operation_parameters", {}).get(operation)
    if declaration is not None:
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
    return values


def context_payload(source: Any) -> dict[str, Any]:
    """Record only construction state required by static request validation."""
    context = source.context
    from increment.decision import ContrastContext

    if isinstance(context, ContrastContext):
        return {"family": "contrast", "context": context.model_dump(mode="json")}
    experiment = getattr(source, "_experiment", None) or getattr(
        source, "artifact_experiment", None
    )
    identity = getattr(source, "declaration_identity", None)
    return {
        "source_declaration": declaration_value(identity()) if callable(identity) else None,
        "study_id": context.study_id,
        "plan": context.plan,
        "configs": context.configs,
        "cluster": context.cluster,
        "trigger_name": context.trigger_name,
        "source_name": type(source).__name__,
        "experiment": None if experiment is None else experiment.model_dump(mode="json"),
    }


def _validate_artifact_metrics(declaration: Mapping[str, Any]) -> None:
    import json

    from increment.query.artifact_ports import artifact_query_ports
    from increment.semantics.artifact import MetricMeasure

    artifact_reader = artifact_query_ports().reader
    _declared_metric_definitions = artifact_reader._declared_metric_definitions
    _metric_spec_from_binding = artifact_reader._metric_spec_from_binding

    identity = declaration["source_declaration"]
    definitions = _declared_metric_definitions(json.dumps(identity["context"]))
    for binding in identity["metric_measures"]:
        _metric_spec_from_binding(
            TypeAdapter(MetricMeasure).validate_python(binding),
            definitions[binding["metric_name"]],
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
    from increment.estimation.inference import Prior
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
        values["prior"] = TypeAdapter(Prior).validate_python(prior)
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
) -> tuple[dict[str, Any], tuple[ResolvedMetricConfig, ...], tuple[ResolvedMetricConfig, ...]]:
    values = typed_request_options(options)
    base = _CONFIGS.validate_python(declaration["configs"])
    metrics = select_metrics(
        tuple(config.metric for config in base), values.get("metrics"), caller=operation
    )
    overrides = {} if operation in _VALUE_OPERATIONS else values
    configs = overlay_configs(
        metrics,
        base,
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
    return values, base, configs


def _validate_declared_day_axis(
    declaration: Mapping[str, Any],
    request: ReadoutRequest,
    operation: str,
    capabilities: EvidenceCapabilities,
) -> None:
    from increment._day_axis import validate_day_axis, validate_lift_request
    from increment._evidence_types import DayAxisRequest
    from increment.readouts._evidence import declared_day_axis_route
    from increment.semantics.models import Experiment

    experiment = declaration.get("experiment")
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
        ),
        src=_DeclaredSource(request.context, capabilities),
        design=request.design,
        plan=request.plan,
        route=declared_day_axis_route(capabilities),
        experiment=None if experiment is None else Experiment.model_validate(dict(experiment)),
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
    if scope.view != "asof":
        check(validate_request, request)
    if scope.view in {"daily", "asof"}:
        check(_validate_declared_day_axis, declaration, request, operation, capabilities)
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
) -> tuple[CodedError, ...]:
    """Use the execution validators on recorded metric, method and view declarations."""
    declaration = thaw(declaration)
    decoded_design = _DESIGN.validate_python(declaration_value(design))
    try:
        options, base_configs, configs = _request_config(
            declaration, options, decoded_design, estimator_id, operation, metric_name
        )
    except CodedError as refusal:
        return (refusal,)
    plan = CompiledDecisionPlan.model_validate(dict(declaration["plan"]))
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
        capabilities=capabilities.grains,
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

    if capabilities.kind == "artifact":
        check(_validate_artifact_metrics, declaration)
    _validate_scope_request(declaration, request, scope, operation, capabilities, check, refusals)
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
            request=SequentialCaptureRequest(finalized=True),
        )
        capture = options.get("capture")
        if scope.view == "breakout":
            from increment.sequential_source import registered_breakout_dimension

            check(
                registered_breakout_dimension, inference.registration, population=scope.population
            )
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
    if operation in {"run_daily_lift", "run_asof_lift"}:
        from increment._analysis_config import validate_scalar_route

        check(
            validate_scalar_route,
            configs,
            route=f"Analysis.{operation}",
            representation="registered" if inference is not None else "implicit",
            registered_plan=plan,
            design=context.design,
            covariate_moments_available=(
                scalar is None
                or scalar.covariate_grains is None
                or scope.grain in scalar.covariate_grains
            ),
        )
    if getattr(context.design, "mechanism", None) == "observational":
        from increment.estimation._readout_refusals import refuse_observational_quantile

        for metric in request.metrics:
            if metric.type == "quantile":
                check(refuse_observational_quantile, metric)
    return tuple(refusals)
