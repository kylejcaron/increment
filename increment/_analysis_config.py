"""Private metric-selection and per-metric-configuration resolver.

The one seam every readout resolves metric names and per-metric
`methods`/`prior` through, before any query or moment construction runs:

    selected = select_metrics(declared, requested, caller="run")
    configs = resolve_configs(selected, bindings, specs, methods=..., prior=...)

Pure, in-process, no I/O. Callers pass exactly one of `bindings`
(warehouse) or `specs` (dataframe); both `None` resolves only the
call-wide/design-default tiers.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NoReturn, cast

from increment.errors import (
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    raiser,
    refusals,
)
from increment.estimation.engine import (
    Method,
    _is_unadjusted_method,
    _validate_unique_method_names,
    resolve_method_roles,
)
from increment.estimation.inference import Normal, Prior
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import (
    AdjustmentCovariate,
    CupedCovariate,
    ExperimentMetric,
    MethodSpec,
    Metric,
    NormalPriorSpec,
)

if TYPE_CHECKING:
    from increment._literals import MultiplicityCorrection
    from increment._readout_request import Correction, ReadoutRequest
    from increment.decision import CompiledDecisionPlan
    from increment.frame import MetricSpec


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<not set>"


UNSET = _Unset()


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.analysis_config.duplicate_metric_name": "{caller}: duplicate metric name(s) {duplicate_names!r} in metrics=",
        "facade.analysis_config.unknown_metric_declared": "{caller}: unknown metric(s) {unknown_names!r}; declared: {declared_names!r}",
        "facade.analysis_config.metric_definition_mismatch": "{caller}: metric object(s) {mismatched_names!r} differ from their declared definitions; select by name or pass a field-equal declared metric",
        "facade.analysis_config.metric_no_resolved": "metric {metric_name!r} has no resolved source configuration",
        "facade.analysis_config.exploratory_metric_unavailable": "{caller}: exploratory_metrics cannot add {names!r} -- unknown to the loaded definitions: {unknown_names!r}; already declared by the experiment (pass them in metrics= instead): {declared_names!r}. Available to add: {available_names!r}",
        "facade.analysis_config.cuped_linear_moments": RefusalSpec(
            "facade.analysis_config.cuped_linear_moments",
            UnsupportedRequestError,
            template="metric {metric!r} requests cuped_multicovariate, but these aggregate moments carry no joint categorical evidence for that request; export a categorical-only request in moments format 13, or use from_unit_summary, from_unit_panel, from_definitions, or an artifact published with covariate extensions.",
            keys=frozenset({"metric"}),
        ),
        "facade.analysis_config.cuped_scalar_moments": RefusalSpec(
            "facade.analysis_config.cuped_scalar_moments",
            UnsupportedRequestError,
            template="metric {metric!r} requests named scalar CUPED covariates, but this moments format carries no covariate-name provenance; use from_unit_summary, from_unit_panel, from_definitions, or an artifact with covariate extensions.",
            keys=frozenset({"metric"}),
        ),
        "facade.analysis_config.cuped_linear_covariates": "metric {metric!r} requests cuped_multicovariate without covariates declared on the method.",
        "facade.analysis_config.cuped_linear_metric_type": "metric {metric!r} has type={metric_type!r}; multicovariate CUPED supports mean and conversion metrics only.",
        "facade.analysis_config.multifamily_scalar_breakout": RefusalSpec(
            "facade.analysis_config.multifamily_scalar_breakout",
            UnsupportedRequestError,
            template=(
                "metric {metric!r} requests multiple distinct scalar CUPED covariate families "
                "for a breakout; one breakout moment row can retain only one x family. "
                "Run one scalar family per breakout request, or use cuped_multicovariate "
                "when the covariates form one joint family."
            ),
            keys=frozenset({"metric"}),
        ),
        "facade.analysis_config.covariate_binding_conflict": "covariate alias {covariate!r} has incompatible source binding or kind across methods",
        "readout.scalar_method_evidence": RefusalSpec(
            "readout.scalar_method_evidence",
            UnsupportedRequestError,
            template=(
                "source route {route!r} cannot produce scalar evidence for method {method!r}; "
                "use a unit-bearing source or a source that preserves method-scoped moments"
            ),
            keys=frozenset({"route", "method"}),
        ),
    },
)
_raise = raiser(_REFUSALS)


def method_covariates(config: object, method: Method) -> tuple[CupedCovariate, ...] | None:
    """Return one method's inputs; ``None`` is the established implicit scalar x-family."""
    if method.variance_reduction == "cuped":
        return method.covariates
    return ()


def ensure_covariate_binding_compatible(
    previous: CupedCovariate | AdjustmentCovariate,
    current: CupedCovariate,
    *,
    current_alias: str | None = None,
) -> None:
    previous_kind = getattr(previous, "kind", None)
    current_kind = getattr(current, "kind", None)
    previous_property = getattr(previous, "property", None) or getattr(previous, "name", None)
    current_property = (
        getattr(current, "property", None) or current_alias or getattr(current, "name", None)
    )
    previous_source = getattr(previous, "source", None)
    current_source = getattr(current, "source", None)
    previous_measure = getattr(previous, "measure", None)
    current_measure = getattr(current, "measure", None)
    compatible = (
        (previous_kind is None or current_kind is None or previous_kind == current_kind)
        and (
            previous_property is None
            or current_property is None
            or previous_property == current_property
        )
        and (previous_source is None or current_source is None or previous_source == current_source)
        and ((previous_measure is None) == (current_measure is None))
        and (
            previous_measure is None
            or current_measure is None
            or previous_measure == current_measure
        )
    )
    if not compatible:
        _raise(
            "facade.analysis_config.covariate_binding_conflict",
            covariate=current_alias or current.name,
        )


def union_covariates(
    configs: Sequence[object], methods: Sequence[Sequence[Method]]
) -> tuple[CupedCovariate, ...]:
    """Deduplicate raw covariate bindings before acquisition, independent of missing policy."""
    by_name: dict[str, CupedCovariate] = {}
    for _config, method_group in zip(configs, methods, strict=True):
        for method in method_group:
            for covariate in method.covariates or ():
                previous = by_name.get(covariate.name)
                if previous is None:
                    by_name[covariate.name] = covariate
                    continue
                ensure_covariate_binding_compatible(previous, covariate)
    return tuple(by_name.values())


_DEFAULT_DECISION_METHOD = Method(name="unadjusted")


@dataclass(frozen=True, slots=True)
class ResolvedMetricConfig:
    """One selected metric's resolved method roles and prior."""

    metric: Metric
    decision_method: Method
    sensitivity_methods: tuple[Method, ...]
    prior: Prior | None
    prior_is_global: bool
    covariates: tuple[CupedCovariate | AdjustmentCovariate, ...] = ()
    # An explicit ``methods=[]`` declaration emits no estimator rows.
    methods_explicitly_empty: bool = False
    # Provenance flag: the winning declaration omitted its decision method.
    # It is not inferred from a placeholder ``Method`` instance.
    decision_defaulted: bool = False
    covariate_available: bool = False
    source_methods: tuple[Method, ...] = ()

    @property
    def uses_design_default(self) -> bool:
        return self.decision_defaulted and not self.methods_explicitly_empty


def effective_methods(
    config: ResolvedMetricConfig,
    *,
    design: Randomized | Encouragement | Observational | None,
) -> tuple[Method, ...]:
    """Return the concrete method sequence for a resolved metric."""
    if config.methods_explicitly_empty:
        return ()
    decision = config.decision_method
    if config.decision_defaulted and isinstance(design, Observational):
        decision = Method(name="iptw")
    methods = (decision, *config.sensitivity_methods)
    _validate_unique_method_names(methods, caller=f"metric {config.metric.name!r}")
    return methods


def _scalar_preparation(covariates: Sequence[CupedCovariate] | None) -> tuple[str, ...]:
    return (
        tuple(covariate.missing for covariate in covariates)
        if covariates is not None
        else ("zero",)
    )


def validate_scalar_route(
    configs: Sequence[ResolvedMetricConfig],
    *,
    route: str,
    representation: Literal[
        "unit",
        "moments",
        "breakout",
        "implicit",
        "native_sequential",
        "registered",
        "unavailable",
        "prepared",
    ] = "unit",
    design: Randomized | Encouragement | Observational | None = None,
    retained_configs: Sequence[ResolvedMetricConfig] = (),
    registered_plan: CompiledDecisionPlan | None = None,
    bindings: Mapping[str, Sequence[CupedCovariate]] | None = None,
    covariate_moments_available: bool | None = None,
    anonymous_metrics: Collection[str] = (),
) -> None:
    """Admit effective scalar inputs against a route's retained representation.

    The input family includes its missing-value preparation, not its display
    label. This check uses declarations only and precedes evidence acquisition.
    Raw published bindings can apply each supported missing policy at reduction.
    Known moment capabilities retain their existing missing-covariate refusal
    before checking whether a named binding is representable.
    """
    retained = {config.metric.name: config for config in retained_configs}
    for config in configs:
        methods = effective_methods(config, design=design)
        scalar = tuple(method for method in methods if method.variance_reduction == "cuped")
        families = {method.covariates for method in scalar}
        if representation in {"moments", "breakout"} and len(families) > 1:
            code = {
                "moments": "facade.analysis_config.cuped_scalar_moments",
                "breakout": "facade.analysis_config.multifamily_scalar_breakout",
            }[representation]
            _raise(code, metric=config.metric.name, route=route)
        expected = set()
        if representation in {"registered", "prepared"}:
            declared = retained.get(config.metric.name)
            expected = (
                {
                    method.covariates
                    for method in effective_methods(declared, design=design)
                    if method.variance_reduction == "cuped"
                }
                if declared is not None
                else set()
            )
            if representation == "registered" and registered_plan is not None:
                from increment.estimation.decision_types import ContrastDecisionProcedure

                procedure = registered_plan.procedures.get(config.metric.name)
                expected = (
                    {
                        method.covariates
                        for method in (procedure.decision_method, *procedure.sensitivity_methods)
                        if method.variance_reduction == "cuped"
                    }
                    if procedure is not None
                    and not isinstance(procedure, ContrastDecisionProcedure)
                    else set()
                )
            if representation == "registered" and (len(families) > 1 or families != expected):
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "source.invalid",
                    "effective scalar inputs and missing-value preparation must match "
                    "the registered input; construct a new analysis for a different recipe",
                    metric=config.metric.name,
                    route=route,
                )
        for method in scalar:
            if representation == "prepared" and method.covariates not in expected:
                preparation_matches = config.metric.name in anonymous_metrics and any(
                    _scalar_preparation(method.covariates) == _scalar_preparation(family)
                    for family in expected
                )
                if not preparation_matches and (expected or method.covariates is not None):
                    _raise("readout.scalar_method_evidence", route=route, method=method.name)
            if (
                representation == "implicit"
                and method.covariates
                and covariate_moments_available is not None
                and (not config.covariate_available or not covariate_moments_available)
            ):
                from increment.estimation.cuped.scalar import refuse_unmaterialized_covariate

                refuse_unmaterialized_covariate(
                    group_id=getattr(design, "control_group", None),
                    metric=config.metric.name,
                )
            if representation == "unavailable" or (
                representation == "implicit" and method.covariates
            ):
                _raise("readout.scalar_method_evidence", route=route, method=method.name)
            if representation == "native_sequential" and method.covariates:
                from increment.sequential_state import sequential_refuse

                sequential_refuse(
                    "source.invalid",
                    "native sequential capture retains the outcome pre-period input, not "
                    "explicit property or measure bindings; use the implicit registered "
                    "input or a unit-summary frame with the bound covariate",
                    metric=config.metric.name,
                    route=route,
                )
            if bindings is None:
                continue
            for covariate in method.covariates or ():
                candidates = bindings.get(config.metric.name, ())
                matching = False
                for candidate in candidates:
                    if candidate.name != covariate.name:
                        continue
                    try:
                        ensure_covariate_binding_compatible(candidate, covariate)
                    except InvalidRequestError:
                        continue
                    matching = True
                    break
                if not matching:
                    _raise(
                        "readout.scalar_method_evidence",
                        method=method.name,
                        route=route,
                    )


def validate_readout_covariates(request: ReadoutRequest) -> None:
    """Refuse effective CUPED bindings absent from the source's available columns."""
    for config, methods in zip(request.configs, request.estimation_methods, strict=True):
        for method in methods:
            if method.variance_reduction not in ("cuped", "cuped_multicovariate"):
                continue
            if not method.covariates:
                continue
            if (
                method.variance_reduction == "cuped_multicovariate"
                and request.scalar_bindings is not None
            ):
                published = {
                    covariate.name: covariate
                    for covariate in request.scalar_bindings.get(config.metric.name, ())
                }
                for covariate in method.covariates:
                    previous = published.get(covariate.name)
                    if previous is None:
                        _raise(
                            "facade.analysis_config.cuped_linear_covariates",
                            metric=config.metric.name,
                        )
                    ensure_covariate_binding_compatible(previous, covariate)
            requested = frozenset(covariate.name for covariate in method.covariates)
            if request.available_covariates is not None:
                if requested <= request.available_covariates:
                    continue
                _raise(
                    "facade.analysis_config.cuped_linear_covariates",
                    metric=config.metric.name,
                )
            if config.covariate_available or method in config.source_methods:
                continue
            legacy_scalar_x = (
                request.source_name == "MomentsSource"
                and method.variance_reduction == "cuped"
                and len(method.covariates) == 1
            )
            if not legacy_scalar_x:
                _raise(
                    "facade.analysis_config.cuped_linear_covariates",
                    metric=config.metric.name,
                )


def normalize_display_correction(correction: MultiplicityCorrection) -> Correction:
    """Downcast a plan-registered `e_bh` (sequential-selection policy) to the
    `bh` a readout-request object accepts -- request objects only ever admit
    none/bh/bonferroni; e_bh is a runtime detail of AlwaysValid multiplicity,
    never a value a caller passes to a readout."""
    return "bh" if correction == "e_bh" else correction


def _roles_from_methods(
    methods: Sequence[Method],
) -> tuple[Method, tuple[Method, ...]]:
    """Resolve the shipped public ``methods=`` ordering into method roles.

    A registered adjustment name (``"iptw"``/``"dml"``/``"aipw"``, or any
    later registration) is only legal under observational identification --
    the randomized path refuses those names before roles matter -- so its
    presence in ``methods`` IS the observational signal: it takes the
    decision role over an "unadjusted" comparison alongside it. Otherwise
    "unadjusted" stays decision by default, unchanged.
    """
    _validate_unique_method_names(methods, caller="methods")
    if not methods:
        return _DEFAULT_DECISION_METHOD, ()
    from increment.estimation.adjust import ADJUSTMENTS

    prefer = (
        (lambda method: not _is_unadjusted_method(method))
        if any(method.name in ADJUSTMENTS for method in methods)
        else _is_unadjusted_method
    )
    roles = resolve_method_roles(methods, prefer=prefer)
    decision = next(m for m in methods if roles[m.name] == "decision")
    sensitivity = tuple(m for m in methods if m is not decision)
    return decision, sensitivity


def _declared_roles(
    decision: Method | None,
    sensitivities: Sequence[Method],
) -> tuple[Method, tuple[Method, ...]]:
    return (
        decision if decision is not None else Method(name="unadjusted"),
        tuple(sensitivities),
    )


def method_from_spec(spec: MethodSpec) -> Method:
    """Convert a declaration-layer method spec to a runtime method."""
    return Method(
        name=spec.name,
        variance_reduction=spec.variance_reduction,
        conversion_inference=spec.conversion_inference,
        covariates=spec.covariates,
        procedure=spec.procedure,
    )


def normal_from_spec(spec: NormalPriorSpec) -> Normal:
    """Convert a declaration-layer `NormalPriorSpec` to the runtime `Normal` estimators consume."""
    return Normal(mu=spec.mu, sigma=spec.sigma)


def select_metrics(
    declared: Sequence[Metric],
    requested: Sequence[str | Metric] | None,
    *,
    caller: str,
    allow_undeclared_objects: bool = False,
    require_declared_definitions: bool = False,
) -> list[Metric]:
    """Resolve *requested* against *declared*, in declaration order.

    `requested=None` selects every declared metric. A `str` resolves by
    name to the declared object; an unknown name raises. A `Metric` object
    resolves by name too but keeps its own identity rather than the
    declared object, so a caller-constructed variant is never swapped out
    by surprise; when its name isn't declared it's kept only if
    *allow_undeclared_objects* is set (appended after declared-matched
    entries, in the caller's order), else it raises. A duplicate effective
    name always raises. Output order is declaration order for matched
    entries; undeclared objects are ordered by the caller.

    With *require_declared_definitions*, matched objects must be field-equal
    to their declared definition and resolve to that declaration.
    """
    if requested is None:
        return list(declared)

    by_name = {m.name: m for m in declared}
    declared_names = sorted(by_name)

    seen_names: set[str] = set()
    duplicate_names: list[str] = []
    unknown_names: list[str] = []
    mismatched_names: list[str] = []
    resolved_by_name: dict[str, Metric] = {}
    undeclared_in_order: list[Metric] = []

    for item in requested:
        name = item if isinstance(item, str) else item.name
        if name in seen_names:
            duplicate_names.append(name)
            continue
        seen_names.add(name)

        if name in by_name:
            declared_object = by_name[name]
            if isinstance(item, str):
                resolved_by_name[name] = declared_object
            elif require_declared_definitions:
                if item != declared_object:
                    mismatched_names.append(name)
                else:
                    resolved_by_name[name] = declared_object
            else:
                resolved_by_name[name] = item
        elif isinstance(item, str):
            unknown_names.append(name)
        elif allow_undeclared_objects:
            undeclared_in_order.append(item)
        else:
            unknown_names.append(name)

    if duplicate_names:
        _raise(
            "facade.analysis_config.duplicate_metric_name",
            caller=caller,
            duplicate_names=sorted(duplicate_names),
        )
    if unknown_names:
        _raise(
            "facade.analysis_config.unknown_metric_declared",
            caller=caller,
            unknown_names=sorted(unknown_names),
            declared_names=declared_names,
        )
    if mismatched_names:
        _raise(
            "facade.analysis_config.metric_definition_mismatch",
            caller=caller,
            mismatched_names=sorted(mismatched_names),
        )

    ordered = [resolved_by_name[m.name] for m in declared if m.name in resolved_by_name]
    return [*ordered, *undeclared_in_order]


def select_exploratory_metrics(
    available: Sequence[Metric],
    declared_names: Collection[str],
    requested: Sequence[str] | None,
    *,
    caller: str,
) -> list[Metric]:
    """Resolve *requested* names against the definitions' *available* metrics, caller order.

    ``None`` or empty adds nothing. A duplicate raises the shared duplicate-name code;
    an unknown name, or one the experiment already declares, raises
    ``facade.analysis_config.exploratory_metric_unavailable``. Pure and in-process, so
    it runs before any query.
    """
    if not requested:
        return []
    names = [requested] if isinstance(requested, str) else list(requested)
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        _raise(
            "facade.analysis_config.duplicate_metric_name",
            caller=caller,
            duplicate_names=duplicate_names,
        )
    by_name = {metric.name: metric for metric in available}
    declared = set(declared_names)
    unknown = sorted(name for name in names if name not in by_name and name not in declared)
    already_declared = sorted(name for name in names if name in declared)
    if unknown or already_declared:
        _raise(
            "facade.analysis_config.exploratory_metric_unavailable",
            caller=caller,
            names=sorted(names),
            unknown_names=unknown,
            declared_names=already_declared,
            available_names=[metric.name for metric in available],
        )
    return [by_name[name] for name in names]


def refuse_cuped_linear_moments(metric: str) -> NoReturn:
    """Refuse per-unit covariates at the aggregate-moments ingress."""
    _raise("facade.analysis_config.cuped_linear_moments", metric=metric)


def refuse_unidentified_scalar_moments(metric: str) -> NoReturn:
    """Refuse a named scalar covariate absent from aggregate-moment provenance."""
    _raise("facade.analysis_config.cuped_scalar_moments", metric=metric)


def refuse_missing_scalar_evidence(route: str, method: str) -> NoReturn:
    """Refuse a scalar request absent from method-scoped portable evidence."""
    _raise("readout.scalar_method_evidence", route=route, method=method)


def resolve_configs(
    selected: Sequence[Metric],
    bindings: Mapping[str, ExperimentMetric] | None,
    specs: Mapping[str, MetricSpec] | None,
    *,
    methods: Sequence[Method] | None,
    prior: Prior | None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior_override: Prior | None | _Unset = UNSET,
    covariate_available_metrics: Collection[str] = (),
) -> tuple[ResolvedMetricConfig, ...]:
    """Resolve each selected metric's effective method roles and prior."""
    global_roles = _roles_from_methods(methods) if methods is not None else None
    if decision_method is not UNSET or sensitivity_methods is not UNSET:
        role_override = cast(Method, decision_method) if decision_method is not UNSET else None
        sensitivity_override = (
            tuple(cast(Sequence[Method], sensitivity_methods))
            if sensitivity_methods is not UNSET
            else None
        )
        if role_override is not None and sensitivity_override is not None:
            _validate_unique_method_names(
                (role_override, *sensitivity_override), caller="role overrides"
            )
    else:
        role_override = sensitivity_override = None
    methods_explicitly_empty = methods is not None and not methods

    out: list[ResolvedMetricConfig] = []
    for metric in selected:
        spec = specs.get(metric.name) if specs is not None else None
        binding = bindings.get(metric.name) if bindings is not None else None
        if global_roles is not None:
            decision_method, sensitivity_methods = global_roles
            decision_defaulted = False
        elif binding is not None:
            decision_defaulted = binding.decision_method is None
            decision_method, sensitivity_methods = _declared_roles(
                (
                    method_from_spec(binding.decision_method)
                    if binding.decision_method is not None
                    else None
                ),
                tuple(method_from_spec(method) for method in binding.sensitivity_methods),
            )
        elif spec is not None:
            decision_defaulted = spec.decision_method is None
            decision_method, sensitivity_methods = _declared_roles(
                spec.decision_method, spec.sensitivity_methods
            )
        else:
            decision_method, sensitivity_methods = _DEFAULT_DECISION_METHOD, ()
            decision_defaulted = True
        if role_override is not None:
            decision_method = role_override
            decision_defaulted = False
        if sensitivity_override is not None:
            sensitivity_methods = sensitivity_override

        if prior_override is not UNSET:
            resolved_prior = prior_override
            prior_is_global = True
        elif prior is not None:
            resolved_prior = prior
            prior_is_global = True
        elif binding is not None and binding.prior is not None:
            resolved_prior = normal_from_spec(binding.prior)
            prior_is_global = False
        elif spec is not None and spec.prior is not None:
            resolved_prior = spec.prior
            prior_is_global = False
        else:
            resolved_prior = None
            prior_is_global = True

        _validate_unique_method_names(
            tuple(sensitivity_methods)
            if decision_defaulted
            else (decision_method, *sensitivity_methods),
            caller=f"metric {metric.name!r}",
        )
        configured_methods = (() if decision_defaulted else (decision_method,)) + tuple(
            sensitivity_methods
        )
        covariates = union_covariates((object(),), (configured_methods,))
        configured_methods = (() if decision_defaulted else (decision_method,)) + tuple(
            sensitivity_methods
        )
        if any(
            method.variance_reduction == "cuped_multicovariate" for method in configured_methods
        ):
            if not covariates:
                if bindings is None and specs is None:
                    _raise("facade.analysis_config.cuped_linear_moments", metric=metric.name)
                _raise("facade.analysis_config.cuped_linear_covariates", metric=metric.name)
            if metric.type in ("quantile", "retention"):
                _raise(
                    "facade.analysis_config.cuped_linear_metric_type",
                    metric=metric.name,
                    metric_type=metric.type,
                )
        out.append(
            ResolvedMetricConfig(
                metric=metric,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=cast(Prior | None, resolved_prior),
                prior_is_global=prior_is_global,
                methods_explicitly_empty=methods_explicitly_empty,
                decision_defaulted=decision_defaulted,
                covariates=tuple(covariates),
                source_methods=configured_methods,
                covariate_available=(
                    bool(covariates) or metric.name in covariate_available_metrics
                ),
            )
        )
    return tuple(out)


def overlay_configs(
    selected: Sequence[Metric],
    base: Sequence[ResolvedMetricConfig],
    *,
    methods: Sequence[Method] | None,
    prior: Prior | None,
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior_override: Prior | None | _Unset = UNSET,
) -> tuple[ResolvedMetricConfig, ...]:
    """Apply explicit call-wide overrides to a source's base catalog."""
    base_by_name = {config.metric.name: config for config in base}
    global_roles = _roles_from_methods(methods) if methods is not None else None
    if decision_method is not UNSET or sensitivity_methods is not UNSET:
        role_decision = cast(Method, decision_method) if decision_method is not UNSET else None
        role_sensitivity = (
            tuple(cast(Sequence[Method], sensitivity_methods))
            if sensitivity_methods is not UNSET
            else None
        )
        if role_decision is not None and role_sensitivity is not None:
            _validate_unique_method_names(
                (role_decision, *role_sensitivity), caller="role overrides"
            )
    else:
        role_decision = role_sensitivity = None
    methods_explicitly_empty = methods is not None and not methods
    out: list[ResolvedMetricConfig] = []
    for metric in selected:
        config = base_by_name.get(metric.name)
        if config is None:
            _raise("facade.analysis_config.metric_no_resolved", metric_name=metric.name)
        if role_decision is not None or role_sensitivity is not None:
            resolved_decision = (
                role_decision if role_decision is not None else config.decision_method
            )
            resolved_sensitivity = (
                role_sensitivity if role_sensitivity is not None else config.sensitivity_methods
            )
        elif global_roles is None:
            resolved_decision = config.decision_method
            resolved_sensitivity = config.sensitivity_methods
        else:
            resolved_decision, resolved_sensitivity = global_roles
        if role_decision is not None or role_sensitivity is not None:
            result_defaulted = role_decision is None and config.decision_defaulted
        elif global_roles is None:
            result_defaulted = config.decision_defaulted
        else:
            result_defaulted = False
        _validate_unique_method_names(
            tuple(resolved_sensitivity)
            if result_defaulted
            else (resolved_decision, *resolved_sensitivity),
            caller=f"metric {metric.name!r}",
        )
        configured_methods = (
            ()
            if methods_explicitly_empty
            else ((() if result_defaulted else (resolved_decision,)) + tuple(resolved_sensitivity))
        )
        covariates = union_covariates((config,), (configured_methods,))
        out.append(
            replace(
                config,
                metric=metric,
                decision_method=resolved_decision,
                sensitivity_methods=resolved_sensitivity,
                covariates=covariates,
                prior=cast(
                    Prior | None,
                    (
                        prior_override
                        if prior_override is not UNSET
                        else (prior if prior is not None else config.prior)
                    ),
                ),
                prior_is_global=(
                    True
                    if prior_override is not UNSET or prior is not None
                    else config.prior_is_global
                ),
                methods_explicitly_empty=(
                    methods_explicitly_empty
                    if methods is not None
                    else config.methods_explicitly_empty
                ),
                decision_defaulted=result_defaulted,
                covariate_available=config.covariate_available,
            )
        )
    return tuple(out)
