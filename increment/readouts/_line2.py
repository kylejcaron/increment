from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

from increment._analysis_config import UNSET, _Unset
from increment._evidence_types import BreakoutIdentity, EvidenceScope
from increment._literals import Correction
from increment._source_types import declared_evidence, is_custom_source
from increment.declarations import Method
from increment.kernel.context import (
    EvidenceExecutionContext,
    ExecutionMode,
)
from increment.semantics.models import Metric

if TYPE_CHECKING:
    from increment._analysis_execution import Analysis


def _compliance_only(analysis, estimands, dimension=None):
    from increment.semantics.design import Encouragement

    return (
        isinstance(analysis._design, Encouragement)
        and estimands is not None
        and set(estimands) == {"compliance"}
        and dimension is None
    )


def _provider_for(analysis):
    from increment.decision import ContrastContext
    from increment.readouts._provider import native_scope_provider

    if isinstance(analysis._src.context, ContrastContext):
        from increment._evidence_scopes import switchback_scope_provider

        return switchback_scope_provider(analysis._src)

    source = analysis._src
    return source if is_custom_source(source) else native_scope_provider(source)


@contextmanager
def owned_execution_context(
    analysis,
    scope: EvidenceScope,
    *,
    coherence: ExecutionMode = "captured",
):
    context = EvidenceExecutionContext(
        _provider_for(analysis),
        scope,
        run_id=uuid4().hex,
        coherence=coherence,
        shared_capture=True,
    )
    with context.evidence_scope(), analysis._scoped_execution_context(context):
        yield context


def _reject_percentile_breakout(metrics):
    names = tuple(
        metric.name
        for metric in metrics
        if getattr(getattr(metric, "winsorization", None), "has_percentile", False)
    )
    if names:
        from increment.errors import refuse
        from increment.estimation._readout_refusals import READOUT_REFUSALS

        refuse(
            READOUT_REFUSALS["readout.metric.percentile_winsorization"],
            view="run_breakout",
            names=names,
        )


def _selected_breakouts(analysis, dimension: str | None = None):
    experiment = getattr(analysis, "_experiment", None)
    definitions = tuple(getattr(experiment, "breakouts", ()))
    if definitions:
        if dimension is not None:
            definitions = tuple(item for item in definitions if item.property == dimension)
        return tuple(BreakoutIdentity(item.property, item.source) for item in definitions)
    return tuple(
        BreakoutIdentity(name, None)
        for name in declared_evidence(analysis._src).breakouts
        if dimension is None or name == dimension
    )


def _contrast_operation_scope(analysis, operation: str, options: dict[str, Any]) -> EvidenceScope:
    from increment._analysis_config import select_metrics
    from increment._day_axis import _TRIGGER_UNSUPPORTED
    from increment._evidence_dispatch import ContrastHandler, ContrastReadoutRequest
    from increment._whole_window import validate_contrast_request
    from increment.errors import refuse

    source = analysis._src
    analysis._exploratory_metrics(options.get("exploratory_metrics"), caller=operation)
    if options.get("population") == "triggered":
        refuse(
            _TRIGGER_UNSUPPORTED,
            method=operation,
            experiment="<unknown>",
            trigger=None,
            route="switchback",
            supported_sources=("from_definitions", "from_unit_day_artifact"),
        )
    validate_contrast_request(
        decision_method=options.get("decision_method", UNSET),
        sensitivity_methods=options.get("sensitivity_methods", UNSET),
        prior=options.get("prior", UNSET),
        estimands=options.get("estimands"),
        value_scale=options.get("value_scale"),
    )
    selected = tuple(
        select_metrics(source.context.metrics, options.get("metrics"), caller=operation)
    )
    request = ContrastReadoutRequest(
        metrics=selected,
        procedures={metric.name: source.context.procedures[metric.name] for metric in selected},
    )
    ContrastHandler().preflight(source, request)
    return EvidenceScope(
        metrics=selected,
        population="assigned",
        populations=("assigned",),
        grain="total",
        snapshot=True,
        execution=True,
    )


def operation_scope(analysis, operation: str, options: dict[str, Any]) -> EvidenceScope:
    """Validate a selected operation and declare only the evidence it may read."""
    from increment._analysis_config import overlay_configs, select_metrics
    from increment._evidence_scopes import capture_requirements
    from increment._evidence_types import DayAxisRequest
    from increment._whole_window import WholeWindowRequest, validate_whole_window
    from increment.readouts._design_scope import _config_snapshot
    from increment.readouts._requests import _prepare_run_request

    source = analysis._src
    capture = options.get("capture")
    if capture is not None:
        from pydantic import TypeAdapter

        from increment._evidence_refusals import require_evidence
        from increment._evidence_types import SequentialCaptureRequest
        from increment.sequential_source import validate_capture_request

        request = TypeAdapter(SequentialCaptureRequest).validate_python(dict(capture))
        require_evidence(source, request=request)
        validate_capture_request(
            request,
            dynamic=declared_evidence(source).sequential_capture == "asof",
        )
    from increment.decision import ContrastContext

    if isinstance(source.context, ContrastContext):
        return _contrast_operation_scope(analysis, operation, options)
    added = analysis._exploratory_metrics(options.get("exploratory_metrics"), caller=operation)
    selected = (
        analysis._select_day_axis_metrics(options.get("metrics"), caller=operation)
        if operation in {"run_daily", "run_daily_lift", "run_asof", "run_asof_lift"}
        else select_metrics(tuple(analysis._metrics), options.get("metrics"), caller=operation)
    )
    candidates = [(analysis, selected)]
    if added:
        candidates.append((analysis._exploratory_analysis(added, caller=operation), added))
    requests = []
    population = options.get("population") or "assigned"
    experiment = getattr(analysis, "_experiment", None)
    populations = (
        ("assigned", "triggered")
        if options.get("population") is None and getattr(experiment, "trigger", None) is not None
        else (population,)
    )
    for candidate, metrics in candidates:
        overrides = {
            key: options.get(key, UNSET)
            for key in ("decision_method", "sensitivity_methods", "prior")
        }
        configs = overlay_configs(
            metrics,
            candidate._src.context.configs,
            methods=None,
            prior=None,
            decision_method=overrides["decision_method"],
            sensitivity_methods=overrides["sensitivity_methods"],
            prior_override=overrides["prior"],
        )
        if operation == "run":
            req = WholeWindowRequest(
                metrics=tuple(metrics),
                estimands=tuple(options["estimands"])
                if options.get("estimands") is not None
                else None,
                population=options.get("population"),
                value_scale=options.get("value_scale"),
                coherence=options.get("coherence", "captured"),
                **overrides,
            )
            from increment.readouts._evidence import day_axis_route

            validate_whole_window(
                req,
                src=candidate._src,
                plan=candidate._plan,
                design=candidate._design,
                experiment=getattr(candidate, "_experiment", None),
                route=day_axis_route(candidate._src),
            )
            metrics, configs = _prepare_run_request(
                candidate._src,
                metrics=[metric.name for metric in metrics],
                estimands=req.estimands,
                value_scale=req.value_scale,
                population=(
                    "assigned"
                    if population == "triggered"
                    and getattr(candidate._plan.inference, "registration", None) is not None
                    else population
                ),
                by=(),
                **overrides,
            )
            grain = "total"
            breakouts = ()
        elif operation == "run_breakout":
            from increment._analysis_config import normalize_display_correction
            from increment._breakout_readouts import BreakoutReadouts, BreakoutRequest

            policy = candidate._plan.view_policies.for_view(
                "breakout", mechanism=candidate._design.mechanism, segmented=True
            )
            reader = BreakoutReadouts(
                src=candidate._src, experiment=getattr(candidate, "_experiment", None)
            )
            reader.validate_request(
                BreakoutRequest(
                    metrics=tuple(metrics),
                    population=population,
                    correction=normalize_display_correction(policy.correction),
                    q=policy.q if policy.q is not None else candidate._plan.q,
                    **overrides,
                )
            )
            grain = "total"
            breakouts = _selected_breakouts(candidate)
        else:
            grain = "daily" if operation in {"run_daily", "run_daily_lift"} else "asof"
            req = DayAxisRequest(
                caller=cast(
                    Literal["run_daily", "run_daily_lift", "run_asof", "run_asof_lift"], operation
                ),
                grain=grain,
                metrics=tuple(metrics),
                dimension=options.get("dimension"),
                estimands=tuple(options["estimands"])
                if options.get("estimands") is not None
                else None,
                population=population,
                completed_windows_only=options.get("completed_windows_only", False),
            )
            reader = candidate._day_axis()
            reader.validate_request(req, lift=operation.endswith("_lift"))
            if operation.endswith("_lift"):
                run_standalone_day_axis_lift(
                    candidate,
                    req,
                    validate_only=True,
                    **overrides,
                    coherence=options.get("coherence", "captured"),
                )
                configs = (
                    ()
                    if _compliance_only(candidate, req.estimands, req.dimension)
                    else reader.resolve_lift_options(req.metrics, **overrides).configs
                )
            breakouts = _selected_breakouts(candidate, req.dimension) if req.dimension else ()
        for config in configs:
            _config_snapshot(config, candidate._design)
        requests.append(
            EvidenceScope(
                metrics=(
                    ()
                    if operation != "run_breakout"
                    and _compliance_only(
                        candidate, options.get("estimands"), options.get("dimension")
                    )
                    else tuple(metrics)
                ),
                horizon_metrics=()
                if operation != "run_breakout"
                and _compliance_only(candidate, options.get("estimands"), options.get("dimension"))
                else tuple(candidate._src.context.metrics),
                configs=tuple(configs),
                population=population,
                populations=populations,
                grain=(
                    "total"
                    if operation in {"run_asof", "run_asof_lift"}
                    and getattr(candidate._plan.inference, "registration", None) is not None
                    else grain
                ),
                breakouts=breakouts,
                snapshot=True,
                execution=True,
                include_breakouts=bool(breakouts),
            )
        )
    from increment.estimation.sequential import SEQUENTIAL_POLICIES

    if isinstance(analysis._plan.inference, SEQUENTIAL_POLICIES):
        from increment.readouts._line3 import sequential_scope

        requests.append(
            sequential_scope(analysis, include_breakouts=any(item.breakouts for item in requests))
        )
    return capture_requirements(source, requests)


def run_scoped_line2(
    analysis: Analysis,
    *,
    coherence: ExecutionMode = "captured",
    breakout_correction: Correction | None = None,
    breakout_metrics: Sequence[str | Metric] | None = None,
    breakout_decision_method: Method | _Unset | None = None,
    breakout_sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    breakout_prior: Any = UNSET,
    breakout_exploratory_metrics: Sequence[str] | None = None,
    breakout_population: Literal["assigned", "triggered"] | None = None,
    readouts: Sequence[Literal["run", "breakout", "daily", "daily_lift", "asof", "asof_lift"]]
    | None = None,
) -> dict[str, Any]:
    """Run selected related readouts under one explicitly owned evidence capture."""
    from increment._analysis_config import overlay_configs, select_metrics
    from increment._evidence_scopes import capture_requirements

    selected = (
        tuple(readouts)
        if readouts is not None
        else ("run", "breakout", "daily", "daily_lift", "asof", "asof_lift")
    )
    source = analysis._src
    resolved_configs = list(source.context.configs)
    breakout_extra_metrics = []
    breakout_validation_configs = ()
    if "breakout" in selected:
        selected_metrics = list(
            select_metrics(tuple(analysis._metrics), breakout_metrics, caller="run_breakout")
        )
        breakout_extra_metrics = analysis._exploratory_metrics(
            breakout_exploratory_metrics, caller="run_breakout"
        )
        selected_metrics.extend(breakout_extra_metrics)
        if declared_evidence(source).breakouts:
            _reject_percentile_breakout(selected_metrics)
        base_names = {config.metric.name for config in source.context.configs}
        declared = tuple(metric for metric in selected_metrics if metric.name in base_names)
        undeclared = tuple(metric for metric in selected_metrics if metric.name not in base_names)
        configs_for_breakout = overlay_configs(
            declared,
            source.context.configs,
            methods=None,
            prior=None,
            decision_method=(
                UNSET if breakout_decision_method is None else breakout_decision_method
            ),
            sensitivity_methods=breakout_sensitivity_methods,
            prior_override=breakout_prior,
        )
        if undeclared:
            exploratory = analysis._exploratory_analysis(undeclared, caller="run_breakout")
            extra_configs = overlay_configs(
                undeclared,
                exploratory._src.context.configs,
                methods=None,
                prior=None,
                decision_method=(
                    UNSET if breakout_decision_method is None else breakout_decision_method
                ),
                sensitivity_methods=breakout_sensitivity_methods,
                prior_override=breakout_prior,
            )
        else:
            extra_configs = ()
        breakout_validation_configs = (*configs_for_breakout, *extra_configs)
        resolved_configs.extend(
            config
            for config in (*configs_for_breakout, *extra_configs)
            if config not in resolved_configs
        )
    configs = tuple(resolved_configs)
    if "breakout" in selected:
        from increment._analysis_config import validate_scalar_route

        scalar_evidence = declared_evidence(source).scalar
        validate_scalar_route(
            breakout_validation_configs,
            route="Analysis.run_breakout",
            representation="breakout",
            registered_plan=analysis._plan,
            design=analysis._design,
            covariate_moments_available=(
                scalar_evidence is None
                or scalar_evidence.covariate_grains is None
                or "total" in scalar_evidence.covariate_grains
            ),
        )
    if not is_custom_source(source):
        from increment._evidence_types import DayAxisRequest

        for caller, readout, grain in (
            ("daily_lift", "run_daily_lift", "daily"),
            ("asof_lift", "run_asof_lift", "asof"),
        ):
            if caller not in selected:
                continue
            metrics = analysis._select_day_axis_metrics(None, caller=readout)
            request = DayAxisRequest(
                caller=readout,
                grain=grain,
                metrics=tuple(metrics),
                dimension=None,
            )
            run_standalone_day_axis_lift(
                analysis,
                request,
                decision_method=UNSET,
                sensitivity_methods=UNSET,
                prior=UNSET,
                validate_only=True,
            )
    selected_breakouts = _selected_breakouts(analysis) if "breakout" in selected else ()
    scope_metrics = (*analysis._metrics, *breakout_extra_metrics)
    scope_configs = configs
    if selected == ("breakout",):
        scope_metrics = tuple(selected_metrics)
        scope_configs = breakout_validation_configs
    grain = (
        "asof"
        if any(name in selected for name in ("asof", "asof_lift"))
        else "daily"
        if any(name in selected for name in ("daily", "daily_lift"))
        else "total"
        if "run" in selected or "breakout" in selected
        else None
    )
    experiment = getattr(analysis, "_experiment", None)
    scope_populations = set()
    if "run" in selected and getattr(experiment, "trigger", None) is not None:
        scope_populations.update(("assigned", "triggered"))
    elif is_custom_source(source):
        scope_populations.update(declared_evidence(source).populations)
    if breakout_population is not None:
        scope_populations.add(breakout_population)
    scope = capture_requirements(
        source,
        (
            EvidenceScope(
                metrics=scope_metrics,
                configs=scope_configs,
                population=breakout_population or "assigned",
                populations=tuple(sorted(scope_populations)),
                grain=grain,
                breakouts=selected_breakouts,
                snapshot=True,
                execution=True,
                include_breakouts="breakout" in selected,
            ),
        ),
    )
    with owned_execution_context(analysis, scope, coherence=coherence) as context:
        operations = {
            "run": lambda: analysis.run(),
            "breakout": lambda: analysis._run_breakout(
                _execution_context=context,
                correction=breakout_correction,
                metrics=breakout_metrics,
                decision_method=(
                    UNSET if breakout_decision_method is None else breakout_decision_method
                ),
                sensitivity_methods=breakout_sensitivity_methods,
                prior=breakout_prior,
                exploratory_metrics=breakout_exploratory_metrics,
                population=breakout_population,
            ),
            "daily": lambda: analysis.run_daily(),
            "daily_lift": lambda: analysis.run_daily_lift(),
            "asof": lambda: analysis.run_asof(),
            "asof_lift": lambda: analysis.run_asof_lift(),
        }
        return {name: operations[name]() for name in selected}


def _triggered_day_axis_capture_supported(analysis, req) -> bool:
    if (
        req.population != "triggered"
        or getattr(getattr(analysis, "_experiment", None), "trigger", None) is None
    ):
        return False
    from increment.readouts._evidence import ARTIFACT_ROUTE, NATIVE_ROUTE, day_axis_route

    return day_axis_route(analysis._src) in (ARTIFACT_ROUTE, NATIVE_ROUTE)


def run_standalone_day_axis_values(
    analysis, req, execution_context=None, *, coherence: ExecutionMode = "captured"
):
    if not req.metrics and execution_context is None:
        return analysis._day_axis().values(req)

    if execution_context is not None:
        return analysis._day_axis(execution_context).values(req)
    triggered_capture = _triggered_day_axis_capture_supported(analysis, req)
    if is_custom_source(analysis._src) or (
        not triggered_capture
        and (
            req.population == "triggered"
            or req.grain not in declared_evidence(analysis._src).grains
        )
    ):
        return analysis._day_axis().values(req)

    from increment._evidence_scopes import capture_requirements

    scope = capture_requirements(
        analysis._src,
        (
            EvidenceScope(
                metrics=req.metrics,
                configs=(),
                population=req.population,
                grain=req.grain,
                dimension=req.dimension,
                breakouts=(
                    _selected_breakouts(analysis, req.dimension)
                    if req.dimension is not None
                    else ()
                ),
                snapshot=coherence == "captured",
                execution=True,
                include_breakouts=req.dimension is not None,
            ),
        ),
    )
    with owned_execution_context(analysis, scope, coherence=coherence) as context:
        return analysis._day_axis(context).values(req)


def run_standalone_day_axis_lift(
    analysis,
    req,
    *,
    decision_method,
    sensitivity_methods,
    prior,
    validate_only: bool = False,
    coherence: ExecutionMode = "captured",
):
    """Resolve lift options before opening the owned evidence context."""

    triggered_capture = _triggered_day_axis_capture_supported(analysis, req)
    unsupported_capture = is_custom_source(analysis._src) or (
        not triggered_capture
        and (
            req.population == "triggered"
            or req.grain not in declared_evidence(analysis._src).grains
        )
    )
    if validate_only and unsupported_capture:
        return None
    if unsupported_capture:
        return analysis._day_axis().lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
    compliance_only = _compliance_only(analysis, req.estimands, req.dimension)
    metrics = () if compliance_only else req.metrics
    if req.caller == "run_daily_lift":
        from increment.readouts._common import _sequential_inference
        from increment.readouts._daily import _validate_daily_inference

        _validate_daily_inference(
            _sequential_inference(analysis._plan),
            method=req.caller,
        )

    from increment._analysis_config import overlay_configs, resolve_configs, validate_scalar_route

    base = analysis._src.context.configs
    base_names = {config.metric.name for config in base}
    declared = tuple(metric for metric in metrics if metric.name in base_names)
    undeclared = tuple(metric for metric in metrics if metric.name not in base_names)
    by_name = {
        config.metric.name: config
        for config in overlay_configs(
            declared,
            base,
            methods=None,
            prior=None,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior_override=prior,
        )
    }
    by_name.update(
        {
            config.metric.name: config
            for config in resolve_configs(
                undeclared,
                None,
                None,
                methods=None,
                prior=None,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior_override=prior,
            )
        }
    )
    configs = tuple(by_name[metric.name] for metric in metrics)
    from increment._analysis_config import effective_methods
    from increment._evidence_refusals import check_method_covariates

    for config in configs:
        check_method_covariates(
            analysis._src,
            config.metric.name,
            effective_methods(config, design=analysis._design),
        )
    scalar_evidence = declared_evidence(analysis._src).scalar
    validate_scalar_route(
        configs,
        route=f"Analysis.{req.caller}",
        representation=(
            "registered"
            if getattr(analysis._plan.inference, "registration", None) is not None
            else "implicit"
        ),
        registered_plan=analysis._plan,
        design=analysis._design,
        covariate_moments_available=(
            scalar_evidence is None
            or scalar_evidence.covariate_grains is None
            or req.grain in scalar_evidence.covariate_grains
        ),
    )
    if req.caller == "run_daily_lift":
        from increment._readout_request import ReadoutRequest, validate_request
        from increment.semantics.design import Encouragement

        compliance_only = (
            isinstance(analysis._design, Encouragement)
            and tuple(req.estimands or ()) == ("compliance",)
            and req.dimension is None
        )
        validation_metrics = () if compliance_only else tuple(req.metrics)
        validation_configs = () if compliance_only else configs
        validate_request(
            ReadoutRequest.from_source(
                analysis._src,
                metrics=validation_metrics,
                configs=validation_configs,
                view="daily",
                grain="daily",
                by=() if req.dimension is None else (req.dimension,),
                dimension=req.dimension,
                estimands=req.estimands,
                population=req.population,
                daily_lift=True,
            )
        )

    if req.caller == "run_asof_lift":
        from increment._readout_request import ReadoutRequest
        from increment.estimation.encouragement import validate_readout_encouragement
        from increment.sequential_source import validate_sequential_request

        request = ReadoutRequest.from_source(
            analysis._src,
            metrics=list(metrics),
            configs=configs,
            view="asof",
            grain="asof",
            estimands=req.estimands,
            completion_policy=req.completed_windows_only,
        )
        validate_readout_encouragement(request)
        validate_sequential_request(request)
    if validate_only:
        return None
    from increment.estimation.encouragement import ESTIMANDS
    from increment.semantics.design import Encouragement

    is_encouragement = isinstance(analysis._design, Encouragement)
    requested_estimands = (
        tuple(req.estimands)
        if req.estimands is not None
        else ESTIMANDS
        if is_encouragement
        else ("itt",)
    )
    compliance_only = (
        is_encouragement and requested_estimands == ("compliance",) and req.dimension is None
    )
    compliance_rows_requested = (
        is_encouragement and req.dimension is None and "compliance" in requested_estimands
    )
    if not req.metrics and not compliance_rows_requested:
        return analysis._day_axis().lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
    scope_metrics = () if compliance_only else req.metrics
    from increment._evidence_scopes import capture_requirements

    scope = capture_requirements(
        analysis._src,
        (
            EvidenceScope(
                metrics=scope_metrics,
                configs=configs,
                population=req.population,
                grain=req.grain,
                dimension=req.dimension,
                estimands=req.estimands,
                breakouts=(
                    _selected_breakouts(analysis, req.dimension)
                    if req.dimension is not None
                    else ()
                ),
                snapshot=coherence == "captured",
                execution=True,
                include_breakouts=req.dimension is not None,
            ),
        ),
    )
    with owned_execution_context(analysis, scope, coherence=coherence) as context:
        return analysis._day_axis(context).lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
