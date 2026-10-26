"""Static compatibility seam for inferential readouts.

A readout request is assembled after metric selection and configuration overlays,
then validated before a source is asked for moments.  Validators remain owned by
the axis that knows the rule; this module only normalizes the request and
coordinates those validators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NoReturn, cast

from increment._analysis_config import effective_methods
from increment._evidence_types import SourceContext
from increment._literals import Correction, ValueScale
from increment._source_types import declared_evidence
from increment.errors import (
    RefusalSpec,
    refuse,
)
from increment.estimation._readout_refusals import READOUT_REFUSALS
from increment.sources import EvidenceSource, Grain

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.engine import Method
    from increment.semantics.models import CupedCovariate, Metric


def _validate_typed_arm_compatibility(request: ReadoutRequest) -> None:  # noqa: PLR0915
    """Run the normalized arm gate before sequential/adjustment/source access."""
    from increment.compatibility import Unsupported, refuse_unsupported
    from increment.estimation.arm_contract import (
        ARM_EVIDENCE_CONTRACT,
        AbsoluteDecisionPolicy,
        AnalysisAxes,
        ArmCompatibilityRequest,
        FamilyPolicy,
        MethodCapability,
        MetricCapabilities,
        RelativeDecisionPolicy,
        cuped_capability,
    )
    from increment.estimation.engine import is_open_ended
    from increment.estimation.sequential import (
        ASYMPTOTIC_PROCEDURE_POLICIES,
        SEQUENTIAL_POLICIES,
        AlwaysValid,
    )
    from increment.semantics.assignment import ParallelAssignment
    from increment.sequential_source import adjustment_kind

    if getattr(request.design, "mechanism", None) == "observational":
        if request.cluster is not None and any(
            config.prior is not None for config in request.configs
        ):
            refuse_unsupported(Unsupported("arm.adjustment.cluster_prior"), cluster=request.cluster)
        return
    inference = request.plan.inference
    sequential_inference = inference if isinstance(inference, SEQUENTIAL_POLICIES) else None
    compliance_only = request.estimands is not None and set(request.estimands) == {"compliance"}
    if sequential_inference is not None and compliance_only:
        from increment.sequential_source import validate_compliance_policy

        validate_compliance_policy(request.plan, request.design, required=True)
        return
    mechanism = getattr(request.design, "mechanism", "randomized")
    uptake_window = (
        "bounded"
        if getattr(getattr(request.design, "uptake", None), "window_days", None) is not None
        else "unbounded"
        if mechanism == "encouragement"
        else "not_applicable"
    )
    for metric, config in zip(request.metrics, request.configs, strict=True):
        procedure = request.plan.procedures[metric.name]
        in_family = getattr(procedure.family, "member", False)
        if request.view == "breakout":
            in_family = True
        elif request.view == "asof":
            in_family = (
                getattr(procedure.family, "member", False)
                and procedure.role == "secondary"
                and mechanism != "encouragement"
            )
        if request.view == "run":
            policy_kind = (
                "e_bh"
                if in_family and sequential_inference is not None
                else "bh"
                if in_family
                else "none"
            )
        else:
            policy_kind = request.correction or "none"
            if request.correction is None and request.view in ("breakout", "asof"):
                view_policy = request.plan.view_policies.for_view(
                    request.view,
                    mechanism=mechanism,
                    segmented=bool(request.by),
                )
                policy_kind = view_policy.correction
            if in_family and isinstance(sequential_inference, AlwaysValid):
                if request.view == "asof":
                    policy_kind = "e_bh"
                elif policy_kind == "bh":
                    policy_kind = "e_bh"
            if not in_family:
                policy_kind = "none"
        if (
            in_family
            and isinstance(sequential_inference, ASYMPTOTIC_PROCEDURE_POLICIES)
            and request.view != "run"
            and (request.view != "breakout" or policy_kind != "none")
        ):
            policy_kind = "bonferroni"
        axes = (
            ("metric", "arm", "segment")
            if policy_kind != "none" and request.view == "breakout"
            else ("metric", "arm")
            if policy_kind != "none"
            else ()
        )
        family = FamilyPolicy(
            kind=policy_kind,
            axes=axes,
            nominal_alpha=request.plan.alpha,
        )
        adjustment = (
            adjustment_kind(sequential_inference.registration, metric.name)
            if sequential_inference is not None
            else None
        )
        methods = tuple(
            MethodCapability(
                role="decision" if i == 0 else "sensitivity",
                estimator=method.name,
                variance_reduction=cuped_capability(method, adjustment=adjustment),
            )
            for i, method in enumerate(effective_methods(config, design=request.design))
        )
        # Total/active metrics estimate through the mean lift machinery, so
        # they share the mean capability axis.
        metric_type = metric.type if metric.type not in ("total", "active") else "mean"
        metric_caps = MetricCapabilities(
            metric_type=metric_type,
            value_scale=request.value_scale.get(metric.name, "relative"),
            winsorization=(
                "percentile"
                if getattr(getattr(metric, "winsorization", None), "has_percentile", False)
                else "fixed"
                if getattr(metric, "winsorization", None) is not None
                else "none"
            ),
            outcome_window="unbounded" if is_open_ended(metric) else "bounded",
            uptake_window=uptake_window,
        )
        decision = (
            AbsoluteDecisionPolicy(
                alternative=procedure.alternative,
                null_abs=cast(float, getattr(procedure, "null_abs", None)),
                family=family,
            )
            if getattr(procedure, "null_abs", None) is not None
            else RelativeDecisionPolicy(
                alternative=procedure.alternative,
                null_lift=getattr(procedure, "null_lift", 0.0),
                signed_ratio=isinstance(inference, ASYMPTOTIC_PROCEDURE_POLICIES),
                family=family,
            )
        )
        typed = ArmCompatibilityRequest(
            assignment=ParallelAssignment(),
            analysis=AnalysisAxes(
                identification=mechanism,
                view="total" if request.view == "run" else request.view,
                segmented=bool(request.by),
                completed_windows_only=request.completion_policy,
                population=request.population,
                variance_adjustment="none",
            ),
            dependence="cluster" if request.cluster is not None else "iid",
            inference=inference,
            estimand="compliance" if compliance_only else "itt",
            metric=metric_caps,
            decision=decision,
            methods=()
            if compliance_only
            else methods
            or (
                MethodCapability(
                    role="decision", estimator="unadjusted", variance_reduction="none"
                ),
            ),
            prior_present=config.prior is not None,
        )
        support = ARM_EVIDENCE_CONTRACT.runtime_support(typed)
        if not isinstance(support, Unsupported):
            continue

        context: dict[str, object] = {}
        if support.refusal_code in (
            "arm.inference.cluster",
            "arm.adjustment.cluster_cuped",
            "arm.adjustment.cluster_prior",
            "arm.metric.quantile_cluster",
        ):
            context["cluster"] = request.cluster
        if support.refusal_code == "arm.adjustment.cluster_cuped":
            context["encouragement"] = mechanism == "encouragement"
        if support.refusal_code == "arm.adjustment.sequential_cuped":
            context["metrics"] = (metric.name,)
        if support.refusal_code == "arm.metric.quantile_cluster":
            context["metric"] = metric.name
            context["metric_type"] = metric.type
        if support.refusal_code in ("arm.metric.quantile_cuped", "arm.metric.quantile_sequential"):
            context["metric"] = metric.name
        if support.refusal_code == "arm.adjustment.sequential_prior":
            context["metrics"] = (metric.name,)
        refuse_unsupported(support, **context)


ReadoutView = Literal["run", "daily", "asof", "breakout"]

Shape = Literal["unit_summary", "unit_panel"]


# Refusal specs live below the facade so estimators can share them by identity.
_REFUSALS: dict[str, RefusalSpec] = dict(READOUT_REFUSALS)

# Public read-only view for tests and callers that need to record a refusal
# code without depending on private renderer implementation.
REFUSAL_CODES = MappingProxyType(_REFUSALS)


def _raise(code: str, **kwargs: object) -> NoReturn:
    refuse(_REFUSALS[code], **kwargs)


def render_refusal(code: str, **kwargs: object) -> str:
    """Render a stable refusal without raising it."""
    return _REFUSALS[code].render(**kwargs)


@dataclass(frozen=True, slots=True)
class ReadoutRequest:
    """All static inputs needed by one readout dispatch.

    ``metrics`` and ``configs`` are selected/resolved values, never raw metric
    specifications or declaration bindings.  Mapping inputs are copied into a
    read-only proxy in ``from_source`` so freezing the dataclass also freezes
    the per-call scale selection.
    """

    context: SourceContext
    metrics: tuple[Metric, ...]
    configs: tuple[ResolvedMetricConfig, ...]
    view: ReadoutView
    grain: Grain
    by: tuple[str, ...] = ()
    dimension: str | None = None
    available_covariates: frozenset[str] | None = None
    scalar_bindings: Mapping[str, tuple[CupedCovariate, ...]] | None = None
    scalar_moments_only: bool = False
    scalar_recipe_fixed: bool = False
    scalar_recipe_configs: tuple[ResolvedMetricConfig, ...] = ()
    scalar_anonymous_metrics: frozenset[str] = frozenset()
    scalar_method_evidence: bool = False
    scalar_breakout_evidence: bool = False
    capabilities: frozenset[str] = frozenset({"total"})
    source_breakouts: tuple[str, ...] = ()
    source_name: str = "source"
    shape: Shape | None = None
    cluster: str | None = None
    population: Literal["assigned", "triggered"] = "assigned"
    estimands: tuple[str, ...] | None = None
    correction: Correction | None = None
    q: float | None = None
    value_scale: Mapping[str, ValueScale] = field(default_factory=dict)
    completion_policy: bool = False

    @classmethod
    # Public request factory signature is the API for readout construction.
    def from_source(  # noqa: PLR0913
        cls,
        source: EvidenceSource,
        *,
        metrics: Sequence[Metric],
        configs: Sequence[ResolvedMetricConfig],
        view: ReadoutView,
        grain: Grain,
        by: Sequence[str] = (),
        dimension: str | None = None,
        estimands: Sequence[str] | None = None,
        correction: Correction | None = None,
        q: float | None = None,
        value_scale: Mapping[str, ValueScale] | None = None,
        completion_policy: bool = False,
        population: Literal["assigned", "triggered"] = "assigned",
    ) -> ReadoutRequest:
        if len(metrics) != len(configs):
            _raise("readout.request.readoutrequest_requires_one_resolved")
        context = source.context
        evidence = declared_evidence(source)
        scalar = evidence.scalar
        request = cls(
            context=context,
            metrics=tuple(metrics),
            configs=tuple(configs),
            view=view,
            grain=grain,
            by=tuple(by),
            dimension=dimension,
            available_covariates=evidence.available_covariates,
            scalar_bindings=(
                MappingProxyType(dict(scalar.bindings))
                if scalar is not None and scalar.bindings is not None
                else None
            ),
            scalar_moments_only=scalar.moments_only if scalar is not None else False,
            scalar_recipe_fixed=scalar.recipe_fixed if scalar is not None else False,
            scalar_recipe_configs=scalar.recipe_configs if scalar is not None else (),
            scalar_anonymous_metrics=scalar.anonymous_metrics
            if scalar is not None
            else frozenset(),
            scalar_method_evidence=scalar.method_evidence if scalar is not None else False,
            scalar_breakout_evidence=scalar.breakout_evidence if scalar is not None else False,
            capabilities=evidence.grains,
            source_breakouts=evidence.breakouts,
            source_name=type(source).__name__,
            shape=evidence.shape,
            cluster=context.cluster,
            population=population,
            estimands=tuple(estimands) if estimands is not None else None,
            correction=correction,
            q=q,
            value_scale=MappingProxyType(dict(value_scale or {})),
            completion_policy=completion_policy,
        )
        from increment._evidence_refusals import check_method_covariates

        for config, methods in zip(request.configs, request.estimation_methods, strict=True):
            check_method_covariates(source, config.metric.name, methods)
        if getattr(request.design, "mechanism", None) == "observational":
            from increment.estimators.registry import validate_observational_routes

            for config, methods in zip(request.configs, request.estimation_methods, strict=True):
                validate_observational_routes(
                    source,
                    (config.metric,),
                    methods,
                    request.design,
                    request.population,
                )
        return request

    @property
    def design(self):
        return self.context.design

    @property
    def plan(self) -> CompiledDecisionPlan:
        return self.context.plan

    @property
    def estimation_methods(self) -> tuple[tuple[Method, ...], ...]:
        """Resolved estimation methods, preserving per-metric configuration."""
        return tuple(effective_methods(config, design=self.design) for config in self.configs)

    @property
    def priors(self):
        """Resolved priors, if any, preserving per-metric configuration."""
        return tuple(config.prior for config in self.configs)


def _validate_estimands(request: ReadoutRequest) -> None:
    """Reject unknown estimands before design-specific compatibility checks."""
    estimands = request.estimands
    if estimands is None:
        return
    from increment.estimation.encouragement import ESTIMANDS

    unknown = set(estimands) - set(ESTIMANDS)
    if unknown:
        _raise(
            "readout.estimands.unknown",
            unknown=unknown,
            supported=ESTIMANDS,
        )


def _validate_sequential_adjustment(request: ReadoutRequest) -> None:
    from increment.estimation.sequential import (
        SEQUENTIAL_POLICIES,
        SequentialSupportRequest,
        sequential_support_refusal,
    )
    from increment.sequential_source import adjustment_kind

    inference = request.plan.inference
    sequential_inference = inference if isinstance(inference, SEQUENTIAL_POLICIES) else None
    cuped_estimand_requested = request.estimands is None or bool(
        {"itt", "late"} & set(request.estimands)
    )
    # Only a coefficient the registration neither predeclared nor retains the
    # joint moments for is unpredictable on every route.
    uses_cuped = cuped_estimand_requested and any(
        method.variance_reduction == "cuped"
        and (
            sequential_inference is None
            or adjustment_kind(sequential_inference.registration, metric.name) is None
        )
        for metric, methods in zip(request.metrics, request.estimation_methods, strict=True)
        for method in methods
    )
    code = sequential_support_refusal(
        SequentialSupportRequest(inference=sequential_inference, uses_cuped=uses_cuped)
    )
    if code is not None:
        from increment.compatibility import Unsupported, refuse_unsupported

        refuse_unsupported(Unsupported(code), metrics=tuple(m.name for m in request.metrics))


def _validate_identification(request: ReadoutRequest) -> None:
    if request.view == "daily":
        return
    design = request.design
    if design is None:
        _raise("readout.design.required", view=request.view)
    from increment.estimation.encouragement import validate_readout_encouragement_identification

    validate_readout_encouragement_identification(request)


def validate_request(request: ReadoutRequest) -> None:
    """Validate one normalized request before any source query.

    Validators are intentionally imported at call time.  Each owner receives
    the normalized request and remains responsible for its own axis rules;
    adding an assignment/evidence adapter therefore extends one owner rather
    than every Analysis method.
    """
    _validate_estimands(request)
    _validate_identification(request)
    from increment._analysis_config import validate_readout_covariates, validate_scalar_route
    from increment.estimation.sequential import SEQUENTIAL_POLICIES

    representation: Literal["unit", "moments", "breakout", "implicit"] = (
        "breakout"
        if request.view == "breakout"
        else "moments"
        if request.scalar_moments_only
        else "unit"
    )
    route = f"{request.source_name}.{request.view}"
    if request.view == "asof" and not isinstance(request.plan.inference, SEQUENTIAL_POLICIES):
        representation, route = "implicit", "readouts.asof_lift"
    validate_scalar_route(
        request.configs,
        route=route,
        representation=representation,
        design=request.design,
        bindings=request.scalar_bindings,
    )
    if request.scalar_recipe_fixed:
        recorded = {config.metric.name for config in request.scalar_recipe_configs}
        validate_scalar_route(
            tuple(config for config in request.configs if config.metric.name in recorded),
            route=route,
            representation="prepared",
            design=request.design,
            retained_configs=request.scalar_recipe_configs,
            anonymous_metrics=request.scalar_anonymous_metrics,
        )
    if isinstance(request.plan.inference, SEQUENTIAL_POLICIES):
        validate_scalar_route(
            request.configs,
            route=route,
            representation="registered",
            design=request.design,
            retained_configs=request.context.configs,
        )
    from increment.estimation.adjust import validate_readout_adjustment
    from increment.estimation.encouragement import validate_readout_encouragement
    from increment.estimation.engine import (
        refuse_cuped_linear_sequential,
        validate_readout_engine,
    )
    from increment.estimation.inference import validate_readout_inference
    from increment.plan import validate_readout_plan
    from increment.sequential_source import validate_sequential_request
    from increment.sources import validate_readout_source

    validate_readout_covariates(request)
    validate_sequential_request(
        request,
        refuse_cuped_linear=refuse_cuped_linear_sequential,
    )
    validate_readout_plan(request)
    # Semantic/engine validators run before source capability checks so a
    # request's established metric/test refusal wins over a missing grain or
    # breakout dimension.
    validate_readout_engine(request)
    if request.view == "daily":
        validate_readout_source(request)
        return
    _validate_typed_arm_compatibility(request)
    _validate_sequential_adjustment(request)
    validate_readout_adjustment(request)
    validate_readout_encouragement(request)
    validate_readout_inference(request)
    validate_readout_source(request)
    if request.view in {"run", "breakout"} and not isinstance(
        request.plan.inference, SEQUENTIAL_POLICIES
    ):
        supported = (
            request.scalar_breakout_evidence
            if request.view == "breakout"
            else request.scalar_method_evidence
        )
        if not supported:
            validate_scalar_route(
                request.configs,
                route=request.source_name,
                representation="implicit",
                design=request.design,
            )


__all__ = [
    "REFUSAL_CODES",
    "ReadoutRequest",
    "ReadoutView",
    "render_refusal",
    "validate_request",
]
