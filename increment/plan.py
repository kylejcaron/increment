"""Compile declaration-layer plans into immutable decision procedures.

Compilation is pure Python over validated semantic models and runs before a
source is queried.  Warehouse paths leave undeclared catalog metrics
unassigned; frame paths make metrics left after primaries and guardrails
secondary by default.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._literals import Alternative
from increment._plan_compatibility import Role
from increment.decision import (
    _guarantee_for_correction,
    _PlanResolution,
    _resolve_declaration,
)
from increment.errors import (
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.estimation.adjust import REFUSALS as _ADJUST_REFUSALS
from increment.estimation.sequential import SEQUENTIAL_POLICIES, AlwaysValid, AsymptoticMean
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import (
    AnalysisPlan,
    ExperimentMetric,
    Metric,
    MultiplicitySpec,
)

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment._readout_request import ReadoutRequest
    from increment.decision import ArmDecisionProcedure, CompiledDecisionPlan
    from increment.estimation.engine import Method
    from increment.estimation.inference import Prior


__all__ = [
    "Alternative",
    "Role",
    "compile_decision_plan",
    "refuse_observational_relative_margin",
    "validate_compiled_encouragement_plan",
    "validate_readout_plan",
    "with_unassigned_procedures",
]


@dataclass(frozen=True, slots=True)
class _CompilationState:
    """Immutable intermediate produced by the plan compiler stages."""

    resolved: Any
    configs: tuple[ResolvedMetricConfig, ...]
    inference: Any
    procedures: Mapping[str, ArmDecisionProcedure]
    policies: Any


_REFUSALS = refusals(
    UnsupportedRequestError,
    {
        "plan.observational.relative_margin": "relative shifted nulls are not built for the observational path: metric(s) {metrics} declare a relative margin (Metric.margin, or a plan-bound ExperimentMetric.margin) -- an absolute-unit margin (Metric.margin_abs) is supported",
        "plan.encouragement.margin": "metric {metric!r}: margins are not supported for sequential encouragement -- the sequential encouragement estimator builds no shifted null; a fixed-horizon plan applies the margin to the ITT row",
        "plan.encouragement.sequential_cuped": "metric {metric!r}: CUPED methods are not supported with {inference} inference for encouragement except compliance-only requests",
        "plan.configs_keys_match": RefusalSpec(
            "plan.configs_keys_match",
            InvalidRequestError,
            template="configs keys must match the declared metric names exactly",
        ),
        "plan.compile_configs_wrong_type": RefusalSpec(
            "plan.compile_configs_wrong_type",
            InvalidRequestError,
            template="configs must contain ResolvedMetricConfig values",
        ),
        "plan.compile_configs_template_mismatch": RefusalSpec(
            "plan.compile_configs_template_mismatch",
            InvalidRequestError,
            template="resolved config for {metric!r} does not match metric template",
        ),
        "plan.metric_frame_contrast": RefusalSpec(
            "plan.metric_frame_contrast",
            InvalidRequestError,
            template="metric {metric!r}: frame/contrast does not support method or prior configuration overrides",
        ),
        "plan.configs_contain_one": RefusalSpec(
            "plan.configs_contain_one",
            InvalidRequestError,
            template="configs must contain one entry per metric",
        ),
        "plan.metric_informative_priors": "metric {metric!r}: informative priors are not supported with {inference} inference",
        "plan.metric_priors_relative": "metric {metric!r}: {prior} priors require the relative decision scale",
        "plan.corrected_encouragement_breakout": "corrected encouragement breakout is not supported",
        "plan.unsupported_decision_plan": RefusalSpec(
            "plan.unsupported_decision_plan",
            InvalidRequestError,
            template="unsupported decision-plan path {path!r}",
        ),
        "plan.duplicate_metric_name": RefusalSpec(
            "plan.duplicate_metric_name",
            InvalidRequestError,
            template="duplicate metric name(s) {duplicate_names!r} in metrics",
        ),
        "plan.frame_contrast_does": "frame/contrast does not support method or prior overrides",
        "plan.metric_cuped_methods": "metric {metric!r}: CUPED is unavailable under {inference} inference, including predeclared coefficients. Use raw Bernoulli outcomes without CUPED, or monitor under InferenceSpec(kind='asymptotic_mean'), whose adjusted_mean and adjusted_ratio_mean laws retain the joint (Y, X) moments that construction reads (Lindon, Ham, Tingley and Bojinov 2022, arXiv:2210.08589). Pre-period coefficients are supported by its scalar_mean law through InferenceSpec.adjustments or ScalarMeanModel.adjustment",
        "readout.correction.invalid": RefusalSpec(
            "readout.correction.invalid",
            InvalidRequestError,
            template="unknown correction={correction!r}; supported: 'bh', 'bonferroni', 'none'",
        ),
        "readout.inference.observational_run": "sequential inference is not supported under an observational design",
        "readout.encouragement.correction": "multiplicity correction is not supported under an encouragement design",
    },
)
_raise = raiser(_REFUSALS)
_REFUSALS["readout.view.observational"] = _ADJUST_REFUSALS["readout.view.observational"]


def validate_readout_plan(request: ReadoutRequest) -> None:
    """Validate view policy before the source is queried."""
    from increment.decision import FixedInference

    view = request.view
    design = request.design
    mechanism = getattr(design, "mechanism", None)
    plan = request.plan
    correction = request.correction

    if correction is not None and correction not in {"bh", "bonferroni", "none"}:
        _raise("readout.correction.invalid", correction=correction)
    if view == "asof" and mechanism == "observational":
        _raise("readout.view.observational", view=view)
    if view == "breakout" and mechanism == "observational":
        _raise("readout.view.observational", view=view)
    if (
        view == "run"
        and mechanism == "observational"
        and not isinstance(plan.inference, FixedInference)
    ):
        _raise("readout.inference.observational_run")
    if view == "breakout" and mechanism == "encouragement" and correction not in (None, "none"):
        _raise("readout.encouragement.correction")


def _validate_contrast_configs(
    metrics: Sequence[Metric],
    configs: Mapping[str, ResolvedMetricConfig] | Sequence[ResolvedMetricConfig],
) -> None:
    from increment._analysis_config import ResolvedMetricConfig

    if isinstance(configs, Mapping):
        by_name = cast("Mapping[str, ResolvedMetricConfig]", configs)
        expected = {metric.name for metric in metrics}
        if set(by_name) != expected:
            _raise("plan.configs_keys_match")
        supplied = tuple(by_name[metric.name] for metric in metrics)
    else:
        supplied = tuple(configs)
    if len(supplied) != len(metrics) or not all(
        isinstance(config, ResolvedMetricConfig) for config in supplied
    ):
        _raise("plan.compile_configs_wrong_type")
    for metric, config in zip(metrics, supplied, strict=True):
        if config.metric != metric:
            _raise("plan.compile_configs_template_mismatch", metric=metric.name)
        if (
            config.prior is not None
            or config.sensitivity_methods
            or config.methods_explicitly_empty
            or not config.uses_design_default
        ):
            _raise("plan.metric_frame_contrast", metric=metric.name)


def _resolve_compile_configs(
    state: _CompilationState,
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    *,
    path: Literal["warehouse", "frame", "frame/contrast"],
    configs: Mapping[str, ResolvedMetricConfig] | Sequence[ResolvedMetricConfig] | None,
    methods: Sequence[Method] | None,
    prior: Prior | None,
) -> _CompilationState:
    from increment._analysis_config import ResolvedMetricConfig, resolve_configs

    if configs is None:
        bindings: Mapping[str, ExperimentMetric] | None = None
        if plan is not None and path == "warehouse":
            bindings = {
                entry.metric: entry
                for entry in plan.entries()
                if isinstance(entry, ExperimentMetric)
            }
        resolved = resolve_configs(metrics, bindings, None, methods=methods, prior=prior)
    elif isinstance(configs, Mapping):
        config_map = cast("Mapping[str, ResolvedMetricConfig]", configs)
        resolved = tuple(config_map[metric.name] for metric in metrics)
        if methods is not None or prior is not None:
            from increment._analysis_config import overlay_configs

            resolved = overlay_configs(metrics, resolved, methods=methods, prior=prior)
    else:
        resolved = tuple(configs)
        if len(resolved) != len(metrics):
            _raise("plan.configs_contain_one")
        if methods is not None or prior is not None:
            from increment._analysis_config import overlay_configs

            resolved = overlay_configs(metrics, resolved, methods=methods, prior=prior)
    if len(resolved) != len(metrics) or not all(
        isinstance(config, ResolvedMetricConfig) for config in resolved
    ):
        _raise("plan.compile_configs_wrong_type")
    for metric, config in zip(metrics, resolved, strict=True):
        if config.metric != metric:
            _raise("plan.compile_configs_template_mismatch", metric=metric.name)
    return replace(state, configs=tuple(resolved))


def _compile_one_procedure(
    metric: Metric,
    config: ResolvedMetricConfig,
    *,
    test: Any,
    resolved: Any,
    inference: Any,
    design: Randomized | Encouragement | Observational | None,
    path: str,
) -> Any:
    from increment._analysis_config import effective_methods
    from increment.decision import (
        AbsoluteArmDecisionProcedure,
        FamilyMembership,
        MultiplicityFamily,
        RelativeArmDecisionProcedure,
    )
    from increment.estimation.priors import MixturePrior, StudentTPrior
    from increment.estimation.sequential import ASYMPTOTIC_PROCEDURE_POLICIES, SEQUENTIAL_POLICIES

    methods = effective_methods(config, design=design)
    if methods:
        decision_method, *sensitivity_methods = methods
    else:
        decision_method, sensitivity_methods = config.decision_method, []
    resolved_prior = config.prior
    if resolved_prior is not None and resolved.inference is not None:
        _raise(
            "plan.metric_informative_priors",
            metric=metric.name,
            inference=type(resolved.inference).__name__,
        )
    is_absolute = test.null_abs is not None
    if is_absolute and isinstance(resolved_prior, (StudentTPrior, MixturePrior)):
        _raise(
            "plan.metric_priors_relative",
            metric=metric.name,
            prior=type(resolved_prior).__name__,
        )
    if test.role == "secondary":
        sequential = isinstance(resolved.inference, SEQUENTIAL_POLICIES)
        asymptotic = isinstance(resolved.inference, ASYMPTOTIC_PROCEDURE_POLICIES)
        correction = "e_bh" if sequential else "bh"
        family = MultiplicityFamily(
            name="secondary",
            correction=correction,
            q=resolved.q,
            axes=("metric", "arm"),
            guarantee=_guarantee_for_correction(correction),
            validity_regime="asymptotic_sequential" if asymptotic else "finite_sample",
        )
        membership = FamilyMembership(
            family=family,
            member=test.in_family,
        )
    else:
        membership = FamilyMembership()
    common: dict[str, Any] = {
        "metric": metric.name,
        "role": test.role,
        "decision_method": decision_method,
        "sensitivity_methods": tuple(sensitivity_methods),
        "methods_explicitly_empty": config.methods_explicitly_empty,
        "alternative": test.alternative,
        "prior": resolved_prior,
        "prior_is_global": config.prior_is_global,
        "alpha": test.alpha,
        "family": membership,
        "inference": inference,
    }
    if is_absolute:
        return AbsoluteArmDecisionProcedure(
            **common,
            null_abs=test.null_abs if test.null_abs is not None else 0.0,
        )
    return RelativeArmDecisionProcedure(**common, null_lift=test.null_lift)


def _compile_procedures(
    state: _CompilationState,
    metrics: Sequence[Metric],
    *,
    design: Randomized | Encouragement | Observational | None,
    path: str,
) -> _CompilationState:
    from increment.decision import FixedInference

    inference = state.resolved.inference or FixedInference()

    procedures = {
        metric.name: _compile_one_procedure(
            metric,
            config,
            test=state.resolved.procedures[metric.name],
            resolved=state.resolved,
            inference=inference,
            design=design,
            path=path,
        )
        for metric, config in zip(metrics, state.configs, strict=True)
    }
    return replace(state, inference=inference, procedures=procedures)


def _compile_view_policy(
    name: Literal["asof", "breakout"],
    spec: MultiplicitySpec | None,
    *,
    default_correction: Literal["none", "bh", "e_bh"],
    axes: tuple[str, ...],
    resolved: Any,
    inference: Any,
) -> Any:
    from increment.decision import MultiplicityFamily

    correction = default_correction if spec is None else spec.correction
    if isinstance(inference, AsymptoticMean) and correction == "bh":
        from increment.sequential_state import sequential_refuse

        if spec is not None:
            sequential_refuse(
                "route.unsupported", "asymptotic views require explicit Bonferroni, not BH"
            )
        correction = "bonferroni"
    if correction == "bh" and isinstance(inference, AlwaysValid):
        correction = "e_bh"
    q = (resolved.q if spec is None else spec.q) if correction in ("bh", "e_bh") else None
    if isinstance(inference, AsymptoticMean) and correction == "bonferroni":
        q = resolved.q
    guarantee = _guarantee_for_correction(correction)
    return MultiplicityFamily(
        name=name,
        correction=correction,
        q=q,
        axes=axes,
        guarantee=guarantee,
        validity_regime="asymptotic_sequential"
        if isinstance(inference, AsymptoticMean)
        else "finite_sample",
    )


def _compile_view_policies(
    state: _CompilationState, plan: AnalysisPlan | None, mechanism: str | None
) -> _CompilationState:
    from increment.decision import CompiledViewPolicies

    view_spec = plan.view_multiplicity if plan is not None else None
    if mechanism == "encouragement" and view_spec is not None:
        if view_spec.correction != "none":
            _raise("plan.corrected_encouragement_breakout")
    policies = CompiledViewPolicies(
        asof=_compile_view_policy(
            "asof",
            view_spec,
            default_correction="none",
            axes=("metric", "arm"),
            resolved=state.resolved,
            inference=state.inference,
        ),
        randomized_breakout=_compile_view_policy(
            "breakout",
            view_spec,
            default_correction="bh",
            axes=("metric", "arm", "segment"),
            resolved=state.resolved,
            inference=state.inference,
        ),
        encouragement_breakout=_compile_view_policy(
            "breakout",
            None,
            default_correction="none",
            axes=("metric", "arm", "segment"),
            resolved=state.resolved,
            inference=state.inference,
        ),
    )
    return replace(state, policies=policies)


def _compile_contrast_plan(
    state: _CompilationState,
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    path: Literal["frame/contrast"],
) -> Any:
    from increment.decision import (
        CompiledDecisionPlan,
        CompiledViewPolicies,
        FixedInference,
        MultiplicityFamily,
        compile_contrast_procedures,
    )

    return CompiledDecisionPlan(
        declared=state.resolved.declared,
        alpha=state.resolved.alpha,
        q=state.resolved.q,
        path=path,
        inference=FixedInference(),
        procedures=compile_contrast_procedures(plan, metrics, path=path),
        view_policies=CompiledViewPolicies(
            asof=MultiplicityFamily(name="asof", axes=("metric", "arm")),
            randomized_breakout=MultiplicityFamily(
                name="breakout",
                correction="bh",
                q=state.resolved.q,
                axes=("metric", "arm", "segment"),
                guarantee=_guarantee_for_correction("bh"),
            ),
            encouragement_breakout=MultiplicityFamily(
                name="breakout",
                axes=("metric", "arm", "segment"),
            ),
        ),
    )


def bind_automatic_sequential_plan(
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    *,
    design: Randomized | Encouragement | Observational | None,
    source_id: str,
    source_mapping: Mapping[str, object],
    transformations: Sequence[Any] = (),
    path: Literal["warehouse", "frame", "frame/contrast"] = "warehouse",
    pre_period_covariate: bool = False,
) -> AnalysisPlan | None:
    """Bind automatic declarations from metadata before any observations are read.

    An ``always_valid`` plan without a registration registers the exact
    Bernoulli law for conversion and retention metrics; an ``asymptotic_mean``
    plan registers the scalar laws. ``pre_period_covariate`` declares that the
    source derives a per-unit pre-period covariate for every metric
    (``Experiment.n_pre_periods > 0``), which lets a plan binding's CUPED
    method register an adjusted law. The plan's predeclared ``segments`` and
    ``view_multiplicity`` fix the retained segment family; the automatic-only
    metadata is cleared once the immutable roster carries it.
    """
    if plan is None or plan.inference is None or plan.inference.registration is not None:
        return plan
    spec = plan.inference
    from increment.sequential_source import auto_register_bernoulli, auto_register_scalar_mean

    resolved = _resolve_declaration(plan.model_copy(update={"inference": None}), metrics, path=path)
    bindings = {
        entry.metric: entry for entry in plan.entries() if isinstance(entry, ExperimentMetric)
    }
    if spec.kind == "always_valid":
        registration = auto_register_bernoulli(
            source_id=source_id,
            metrics=metrics,
            design=design,
            source_mapping=source_mapping,
            resolved=resolved,
            transformations=transformations,
            bindings=bindings,
            baseline_rate=spec.baseline_rate,
            compliance=plan.compliance,
            segments=spec.segments,
            view_multiplicity=plan.view_multiplicity,
        )
    else:
        registration = auto_register_scalar_mean(
            source_id=source_id,
            metrics=metrics,
            design=design,
            source_mapping=source_mapping,
            resolved=resolved,
            transformations=transformations,
            bindings=bindings,
            pre_period_covariate=pre_period_covariate,
            expected_decision_sample_size=spec.expected_decision_sample_size or 5000,
            adjustments=spec.adjustments,
            compliance=plan.compliance,
            segments=spec.segments,
            view_multiplicity=plan.view_multiplicity,
        )
    inference = spec.model_copy(
        update={
            "registration": registration,
            "expected_decision_sample_size": None,
            "baseline_rate": None,
            "adjustments": {},
            "segments": {},
        }
    )
    return AnalysisPlan.model_validate(
        plan.model_copy(update={"inference": inference}).model_dump()
    )


def compile_decision_plan(
    plan: AnalysisPlan | None,
    metrics: Sequence[Metric],
    *,
    path: Literal["warehouse", "frame", "frame/contrast"] = "warehouse",
    design: Randomized | Encouragement | Observational | None = None,
    configs: Mapping[str, ResolvedMetricConfig] | Sequence[ResolvedMetricConfig] | None = None,
    methods: Sequence[Method] | None = None,
    prior: Prior | None = None,
    estimands: Sequence[str] | None = None,
) -> CompiledDecisionPlan:
    """Compile pure, immutable arm procedures before a source is queried."""
    from increment.decision import CompiledDecisionPlan

    if path not in ("warehouse", "frame", "frame/contrast"):
        _raise("plan.unsupported_decision_plan", path=path)
    metric_names = [metric.name for metric in metrics]
    duplicate_names = sorted({name for name in metric_names if metric_names.count(name) > 1})
    if duplicate_names:
        _raise("plan.duplicate_metric_name", duplicate_names=duplicate_names)
    if plan is not None and plan.inference is not None:
        if plan.inference.registration is None:
            from increment.semantics.sequential import invalid_registration

            invalid_registration("automatic sequential inference needs source metadata binding")
        modeled = {
            model.metric
            for model in plan.inference.registration.models
            if model.observable == "outcome"
        }
        from increment.sequential_state import validate_sequential_transform

        for metric in metrics:
            if metric.name in modeled:
                validate_sequential_transform(metric)
    resolved = _resolve_declaration(plan, metrics, path=path)
    state = _CompilationState(resolved, (), None, {}, None)
    refuse_observational_relative_margin(design, resolved)
    if path == "frame/contrast" and configs is not None:
        _validate_contrast_configs(metrics, configs)
    if path == "frame/contrast" and (methods is not None or prior is not None):
        _raise("plan.frame_contrast_does")
    if path == "frame/contrast":
        return _compile_contrast_plan(state, plan, metrics, path)
    state = _resolve_compile_configs(
        state, plan, metrics, path=path, configs=configs, methods=methods, prior=prior
    )
    state = _compile_procedures(state, metrics, design=design, path=path)
    mechanism = getattr(design, "mechanism", None)
    state = _compile_view_policies(state, plan, mechanism)
    compiled = CompiledDecisionPlan(
        declared=state.resolved.declared,
        alpha=state.resolved.alpha,
        q=state.resolved.q,
        path=path,
        inference=state.inference,
        compliance=plan.compliance if plan is not None else None,
        procedures=state.procedures,
        view_policies=state.policies,
    )
    validate_compiled_encouragement_plan(compiled, design, estimands=estimands)
    from increment.sequential_source import validate_sequential_plan

    validate_sequential_plan(compiled, metrics, design)
    return compiled


def with_unassigned_procedures(
    plan: CompiledDecisionPlan,
    metrics: Sequence[Metric],
    *,
    design: Randomized | Encouragement | Observational | None,
) -> CompiledDecisionPlan:
    """*plan* plus default unassigned procedures for *metrics* it does not compile.

    The one place a metric outside the declared plan receives a procedure: the
    undeclared-plan default (``role="unassigned"``, two-sided) is tested at *plan*'s own
    ``alpha``, the level a declared plan gives its unassigned metrics, so every view reads
    it at one level. It joins no family and leaves every declared procedure untouched.
    """
    from types import MappingProxyType

    missing = [metric for metric in metrics if metric.name not in plan.procedures]
    if not missing:
        return plan
    defaults = compile_decision_plan(None, missing, path=plan.path, design=design)
    leveled = {
        name: type(procedure).model_validate({**dict(procedure), "alpha": plan.alpha})
        for name, procedure in defaults.procedures.items()
    }
    procedures = MappingProxyType({**plan.procedures, **leveled})
    return plan.model_copy(update={"procedures": procedures})


def validate_compiled_encouragement_plan(
    plan: CompiledDecisionPlan,
    design: Encouragement | Observational | Randomized | None,
    *,
    estimands: Sequence[str] | None = None,
) -> None:
    """Revalidate supplied compiled procedures at source-construction seams."""
    mechanism = getattr(design, "mechanism", None)
    for metric, procedure in plan.procedures.items():
        if (
            mechanism == "encouragement"
            and isinstance(plan.inference, SEQUENTIAL_POLICIES)
            and (
                getattr(procedure, "null_lift", 0.0) != 0.0
                or getattr(procedure, "null_abs", None) is not None
            )
        ):
            _raise("plan.encouragement.margin", metric=metric)
        if not isinstance(plan.inference, SEQUENTIAL_POLICIES):
            continue
        compliance_only = (
            mechanism == "encouragement"
            and estimands is not None
            and set(estimands) == {"compliance"}
        )
        if compliance_only:
            from increment.sequential_source import validate_compliance_policy

            validate_compliance_policy(plan, design, required=True)
            continue
        methods = (
            getattr(procedure, "decision_method", None),
            *getattr(procedure, "sensitivity_methods", ()),
        )
        if any(getattr(method, "variance_reduction", "none") == "cuped" for method in methods):
            if mechanism == "encouragement":
                _raise(
                    "plan.encouragement.sequential_cuped",
                    metric=metric,
                    inference=type(plan.inference).__name__,
                )
            from increment.sequential_state import adjustment_kind

            if adjustment_kind(plan.inference.registration, metric) is None:
                _raise(
                    "plan.metric_cuped_methods",
                    metric=metric,
                    inference=type(plan.inference).__name__,
                )


def refuse_observational_relative_margin(
    design: Randomized | Encouragement | Observational | None,
    compiled_plan: CompiledDecisionPlan | _PlanResolution,
) -> None:
    """Refuse relative margins under an observational design."""
    if not isinstance(design, Observational):
        return
    shifted_relative = [
        procedure.metric
        for procedure in compiled_plan.procedures.values()
        if getattr(procedure, "null_lift", 0.0) != 0.0
    ]
    if shifted_relative:
        _raise("plan.observational.relative_margin", metrics=shifted_relative)
