"""Frozen resolution of decision/sensitivity methods and priors for the day-axis lift path.

Replaces `Analysis._resolve_lift_options` and the `_effective_role_maps`
overwrite it fed: three of its six return values were always discarded
and recomputed by `_effective_role_maps` before any caller used them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    effective_methods,
    overlay_configs,
    resolve_configs,
)
from increment.estimation.engine import Method

if TYPE_CHECKING:
    from increment._analysis_config import ResolvedMetricConfig
    from increment.estimation.inference import Prior
    from increment.semantics.models import Metric
    from increment.sources import MomentSource


def _call_method_roles(
    methods: list[Method] | None,
    *,
    decision_method: Method | _Unset = UNSET,
) -> dict[str, Literal["decision", "sensitivity"]]:
    from increment.estimation.engine import resolve_method_roles

    effective = methods or [Method(name="unadjusted")]
    return resolve_method_roles(
        effective,
        decision=None if decision_method is UNSET else cast(Method, decision_method),
    )


def _effective_role_maps(
    src: MomentSource,
    selected_metrics: Sequence[Metric],
    *,
    decision_method: Method | _Unset,
    sensitivity_methods: Sequence[Method] | _Unset,
    prior: Prior | None | _Unset,
) -> tuple[
    dict[str, list[Method]],
    dict[str, Prior | None],
    dict[str, dict[str, Literal["decision", "sensitivity"]]],
    list[Any],
]:
    base = src.context.configs
    base_names = {config.metric.name for config in base}
    declared = [m for m in selected_metrics if m.name in base_names]
    undeclared = [m for m in selected_metrics if m.name not in base_names]
    by_name: dict[str, Any] = {}
    if declared:
        for config in overlay_configs(
            declared,
            base,
            methods=None,
            prior=None,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior_override=prior,
        ):
            by_name[config.metric.name] = config
    if undeclared:
        for config in resolve_configs(
            undeclared,
            None,
            None,
            methods=None,
            prior=None,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior_override=prior,
        ):
            by_name[config.metric.name] = config
    configs = [by_name[metric.name] for metric in selected_metrics]
    methods_by_metric = {
        config.metric.name: list(effective_methods(config, design=src.context.design))
        for config in configs
    }
    return (
        methods_by_metric,
        {c.metric.name: c.prior for c in configs},
        {
            name: {
                method.name: ("decision" if index == 0 else "sensitivity")
                for index, method in enumerate(methods)
            }
            for name, methods in methods_by_metric.items()
        },
        configs,
    )


@dataclass(frozen=True, slots=True)
class MetricRoles:
    methods: tuple[Method, ...]
    prior: Prior | None
    roles: Mapping[str, Literal["decision", "sensitivity"]]


@dataclass(frozen=True, slots=True)
class LiftOptions:
    decision_method: Method | _Unset
    methods: tuple[Method, ...] | None
    prior: Prior | None
    by_metric: Mapping[str, MetricRoles]
    alpha: float
    configs: tuple[ResolvedMetricConfig, ...]

    @classmethod
    def resolve(
        cls,
        src: MomentSource,
        selected: Sequence[Metric],
        *,
        decision_method: Method | _Unset,
        sensitivity_methods: Sequence[Method] | _Unset,
        prior: Prior | None | _Unset,
        alpha: float,
    ) -> LiftOptions:
        """Resolve call-time method/prior overrides the way `_resolve_lift_options` did.

        `metric_templates` mirrors that function always receiving the
        analysis's declared metrics (`src.context.metrics`), independent of
        *selected* (the branch's effective metric list used for `by_metric`).
        """
        metric_templates = list(src.context.metrics)
        methods = (
            None
            if decision_method is UNSET and sensitivity_methods is UNSET
            else [
                cast(Method, decision_method)
                if decision_method is not UNSET
                else Method(name="unadjusted"),
                *(
                    cast(Sequence[Method], sensitivity_methods)
                    if sensitivity_methods is not UNSET
                    else ()
                ),
            ]
        )
        effective_prior = cast("Prior | None", None if prior is UNSET else prior)
        if methods is None and len(metric_templates) == 1:
            config = src.context.configs[0]
            methods = list(effective_methods(config, design=src.context.design))
            effective_prior = config.prior
        methods_by_metric, prior_by_metric, method_roles_by_metric, configs = _effective_role_maps(
            src,
            selected,
            decision_method=decision_method,
            sensitivity_methods=sensitivity_methods,
            prior=prior,
        )
        by_metric = {
            name: MetricRoles(
                methods=tuple(methods_by_metric[name]),
                prior=prior_by_metric[name],
                roles=MappingProxyType(method_roles_by_metric[name]),
            )
            for name in methods_by_metric
        }
        return cls(
            decision_method=methods[0] if methods else decision_method,
            methods=tuple(methods) if methods is not None else None,
            prior=effective_prior,
            by_metric=MappingProxyType(by_metric),
            alpha=alpha,
            configs=tuple(configs),
        )

    def needs_covariate(self, metric: Metric) -> bool:
        roles = self.by_metric.get(metric.name)
        methods = roles.methods if roles is not None else ()
        return any(m.variance_reduction == "cuped" for m in methods)

    def call_roles(self) -> dict[str, Literal["decision", "sensitivity"]]:
        return _call_method_roles(
            list(self.methods) if self.methods is not None else None,
            decision_method=self.decision_method,
        )

    def daily_lift_kwargs(self) -> dict[str, Any]:
        return {
            "methods": list(self.methods) if self.methods is not None else None,
            "prior": self.prior,
            "method_roles": self.call_roles(),
            "methods_by_metric": {name: list(r.methods) for name, r in self.by_metric.items()},
            "prior_by_metric": {name: r.prior for name, r in self.by_metric.items()},
            "method_roles_by_metric": {name: dict(r.roles) for name, r in self.by_metric.items()},
            "alpha": self.alpha,
        }
