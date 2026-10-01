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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, cast

from increment.errors import InvalidRequestError, raiser, refusals
from increment.estimation.engine import Method, _validate_unique_method_names, resolve_method_roles
from increment.estimation.inference import Normal, Prior
from increment.semantics.design import Encouragement, Observational, Randomized
from increment.semantics.models import ExperimentMetric, MethodSpec, Metric, NormalPriorSpec

if TYPE_CHECKING:
    from increment._readout_request import Correction
    from increment.decision import _MultiplicityCorrection
    from increment.frame import MetricSpec


class _Unset:
    __slots__ = ()


UNSET = _Unset()


_REFUSALS = refusals(
    InvalidRequestError,
    {
        "facade.analysis_config.duplicate_metric_name": "{caller}: duplicate metric name(s) {duplicate_names!r} in metrics=",
        "facade.analysis_config.unknown_metric_declared": "{caller}: unknown metric(s) {unknown_names!r}; declared: {declared_names!r}",
        "facade.analysis_config.metric_definition_mismatch": "{caller}: metric object(s) {mismatched_names!r} differ from their declared definitions; select by name or pass a field-equal declared metric",
        "facade.analysis_config.metric_no_resolved": "metric {metric_name!r} has no resolved source configuration",
    },
)
_raise = raiser(_REFUSALS)

_DEFAULT_DECISION_METHOD = Method(name="unadjusted")


@dataclass(frozen=True, slots=True)
class ResolvedMetricConfig:
    """One selected metric's resolved method roles and prior."""

    metric: Metric
    decision_method: Method
    sensitivity_methods: tuple[Method, ...]
    prior: Prior | None
    prior_is_global: bool
    # An explicit ``methods=[]`` declaration emits no estimator rows.
    methods_explicitly_empty: bool = False
    # Provenance flag: the winning declaration omitted its decision method.
    # It is not inferred from a placeholder ``Method`` instance.
    decision_defaulted: bool = False

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


def normalize_display_correction(correction: _MultiplicityCorrection) -> Correction:
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
        (lambda m: m.name != "unadjusted")
        if any(method.name in ADJUSTMENTS for method in methods)
        else (lambda m: m.name == "unadjusted")
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
    return Method(name=spec.name, variance_reduction=spec.variance_reduction)


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
        out.append(
            ResolvedMetricConfig(
                metric=metric,
                decision_method=decision_method,
                sensitivity_methods=sensitivity_methods,
                prior=cast(Prior | None, resolved_prior),
                prior_is_global=prior_is_global,
                methods_explicitly_empty=methods_explicitly_empty,
                decision_defaulted=decision_defaulted,
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
        out.append(
            replace(
                config,
                metric=metric,
                decision_method=resolved_decision,
                sensitivity_methods=resolved_sensitivity,
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
            )
        )
    return tuple(out)
