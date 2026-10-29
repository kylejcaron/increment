from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal
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
    from increment.analysis import Analysis


def _provider_for(analysis):
    from increment.readouts._provider import native_scope_provider

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
    if dimension is not None:
        definitions = tuple(item for item in definitions if item.property == dimension)
    return tuple(BreakoutIdentity(item.property, item.source) for item in definitions)


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
    if is_custom_source(analysis._src) or (
        not triggered_capture
        and (
            req.population == "triggered"
            or req.grain not in declared_evidence(analysis._src).grains
        )
    ):
        return analysis._day_axis().lift(
            req,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
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
    declared = tuple(metric for metric in req.metrics if metric.name in base_names)
    undeclared = tuple(metric for metric in req.metrics if metric.name not in base_names)
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
    configs = tuple(by_name[metric.name] for metric in req.metrics)
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
    if req.caller == "run_asof_lift":
        from increment._readout_request import ReadoutRequest
        from increment.estimation.encouragement import validate_readout_encouragement
        from increment.sequential_source import validate_sequential_request

        request = ReadoutRequest.from_source(
            analysis._src,
            metrics=list(req.metrics),
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
