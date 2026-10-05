"""Observational adjustment dispatch and orchestration policy.

Estimator implementations live in the private ``_adjust`` modules; this
module owns the registry, validation, and public orchestration seam.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._literals import VALUE_SCALE_VALUES, ValueScale
from increment.compatibility import ARM_COMPATIBILITY_REFUSALS
from increment.errors import (
    CapabilityError,
    IncrementWarning,
    InvalidRequestError,
    RefusalSpec,
    UnsupportedRequestError,
    WarningSpec,
    raiser,
    refusals,
    refuse,
    warn,
)
from increment.estimation._adjust.clustered_unadjusted import estimate_clustered_unadjusted
from increment.estimation._readout_refusals import READOUT_REFUSALS as _READOUT_REFUSALS
from increment.estimation._readout_refusals import refuse_observational_quantile
from increment.estimation.engine import (
    Method,
    _df_to_arms,
    _estimate_lift,
    _validate_unique_method_names,
    _winsorization_result_fields,
)
from increment.estimation.inference import (
    Prior,
)
from increment.estimation.priors import MixturePrior, StudentTPrior
from increment.estimation.variance import Registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from increment._readout_request import ReadoutRequest
    from increment.decision import DecisionComputation, DecisionFailure
    from increment.estimation.results import LiftEstimate
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric
    from increment.sources import MomentSource

    AdjustmentFn = Callable[..., list[LiftEstimate]]


# Registry dispatch modules register their estimator functions as import side effects.
ADJUSTMENTS: Registry[AdjustmentFn] = Registry("adjustment")

_BUILTIN_ADJUSTMENTS = frozenset(("iptw", "dml", "aipw"))
from increment.estimation._adjust import aipw as _aipw  # noqa: E402,F401
from increment.estimation._adjust import dml as _dml  # noqa: E402,F401
from increment.estimation._adjust import iptw as _iptw  # noqa: E402,F401

_REFUSALS = (
    refusals(
        InvalidRequestError,
        {
            "estimation.adjust.prior.type": (
                "mixture priors are only supported on the relative (log-RR) lift scale served by"
                " infer_lift/estimate_lift: this path reports rows whose raw pre-prior statistics are"
                " not persisted, so the mixture posterior could not be recomputed for decision"
                " statistics. Pass a Normal prior here, or estimate the metric through the"
                " relative-lift path."
            ),
            "estimation.adjust.prior.method_scale": (
                "prior= cannot span unadjusted log-RR and adjusted linear-relative methods in one"
                " request; run separate calls with priors declared on each method's parameterization"
            ),
            "estimation.adjust.names_metrics_source": (
                "{label}= names metrics this source does not declare: {unknown} (declared: {by_name})"
                " -- refusing rather than silently dropping the request"
            ),
            "estimation.adjust.value_scale_relative": "value_scale={{{name!r}: {requested!r}}}: must be 'relative' or 'absolute'",
            "estimation.adjust.value_scale_names": (
                "value_scale names metric {name!r}: no observational adjustment exists for"
                " {metric_type} metrics -- the absolute channel changes how an adjusted estimate is"
                " REPORTED; it does not extend estimator coverage."
            ),
            "estimation.adjust.resolve_value_scales_null_abs_on_absolute_metric": (
                "margins_abs/null_abs cannot target metric {metric!r}: its rows are absolute-native"
                " (value_scale='absolute'), so the abs_diff/abs_se fields the absolute-margin decision"
                " reads are None by design. A shifted null for an absolute-native row is a null_lift in"
                " the metric's own units, which this row does not support. Testing against 0 with"
                " alternative= runs but drops the margin, so it answers a different question."
            ),
            "estimation.adjust.resolve_value_scales_null_lift_on_absolute_metric": (
                "null_lifts/margins cannot target metric {metric!r} with a nonzero value: its rows are"
                " absolute-native (value_scale='absolute'), so `lift` is in the metric's own units and"
                " a unitless relative null would be compared against an additive interval. A nonzero"
                " shifted null is not available on an absolute-native row. Testing against 0 with"
                " alternative= runs but drops the margin, so it answers a different question."
            ),
            "estimation.adjust.method_name_unadjusted": (
                "Method(name='unadjusted') cannot honor value_scale='absolute': the unadjusted moments"
                " path already reports the absolute pair (abs_diff/abs_se) alongside relative lift on"
                " every row, and its rows are confounded -- an absolute-native unadjusted row would"
                " dress a non-causal difference in the channel built to rescue identified adjusted"
                " estimates. Drop 'unadjusted' from methods or remove the metric from value_scale."
            ),
            "estimation.adjust.prior_interpreted_call": (
                "prior= cannot be interpreted for a call that reports some metrics on the relative"
                " scale and others on the absolute scale: the same Normal would be read as unitless"
                " relative lift on one row and metric-units additive lift on another. Drop prior= or"
                " make the call scale-uniform."
            ),
        },
    )
    | refusals(
        UnsupportedRequestError,
        {
            "adjust.variance_reduction.unsupported": (
                "variance_reduction={variance_reduction!r} is not supported under adjustment {method!r}"
                " -- the adjustment estimators consume unit frames, not the moments the reduction is"
                " defined on; it would be silently ignored rather than applied."
            ),
            "estimation.adjust.estimate_ate_every": (
                "estimate_ate: every declared metric was refused by the requested adjustment method(s)"
                " -- nothing to estimate. See the accompanying UserWarnings for the per-metric refusal"
                " reasons."
            ),
            "readout.view.observational": RefusalSpec(
                "readout.view.observational",
                UnsupportedRequestError,
                lambda *, view: (
                    f"{view} is not supported for an observational design: confounded "
                    "per-segment contrasts are refused, not silently emitted"
                    if view == "breakout"
                    else "as-of lift is not defined for an observational design"
                ),
            ),
        },
    )
    | refusals(
        CapabilityError,
        {
            "adjust.winsorization.percentile_unsupported": (
                "estimate_ate does not support percentile-winsorized metrics ({metric}); use"
                " fixed-value winsorization"
            ),
        },
    )
)
ADJUSTMENT_COMPATIBILITY_REFUSALS = MappingProxyType(
    {
        "arm.adjustment.sequential_cuped": ARM_COMPATIBILITY_REFUSALS[
            "arm.adjustment.sequential_cuped"
        ],
        "arm.adjustment.cluster_cuped": ARM_COMPATIBILITY_REFUSALS["arm.adjustment.cluster_cuped"],
        "arm.adjustment.cluster_prior": ARM_COMPATIBILITY_REFUSALS["arm.adjustment.cluster_prior"],
    }
)


def _refuse_compatibility(code: str, **context: object) -> None:
    """Map normalized adjustment support to its owning public refusal."""
    refuse(ADJUSTMENT_COMPATIBILITY_REFUSALS[code], **context)


_refuse = raiser(_REFUSALS)

MIXTURE_PRIORS_ARE = _REFUSALS["estimation.adjust.prior.type"]


_WARNINGS: dict[str, WarningSpec] = {}


def _register_warning(
    code: str, warning_type: type[IncrementWarning], render: Callable[..., str]
) -> WarningSpec:
    spec = WarningSpec(code, warning_type, render)
    _WARNINGS[code] = spec
    return spec


def _warn(code: str, /, *, stacklevel: int = 2, **context: object) -> None:
    # +1 absorbs this helper's own frame; errors.warn() absorbs its own.
    warn(_WARNINGS[code], stacklevel=stacklevel + 1, context=context)


_register_warning(
    "estimation.adjust.prior_absolute_scale_spans_metrics",
    IncrementWarning,
    lambda *, n_metrics, names: (
        f"estimate_ate: one scalar prior= spans {n_metrics} metrics all "
        "reported on the absolute scale -- it is interpreted in EACH "
        "metric's own additive units, which are not commensurable across "
        f"metrics ({names}). Run one metric per call to give each "
        "its own prior."
    ),
)

_register_warning(
    "estimation.adjust.skip_unsupported_metric",
    IncrementWarning,
    lambda *, metric_name, method_name, exc: (
        f"estimate_ate: skipping metric {metric_name!r} under method {method_name!r} -- {exc}"
    ),
)


_REFUSALS.update(
    {
        code: _READOUT_REFUSALS[code]
        for code in (
            "readout.value_scale.invalid",
            "readout.value_scale.null",
            "readout.observational.prior",
            "readout.adjustment.absolute_unadjusted",
        )
    }
)

# Public read-only view so other modules can import this module's canonical
# refusal specs instead of re-registering the same code.
REFUSALS: Mapping[str, RefusalSpec] = MappingProxyType(_REFUSALS)


def _adjust_kwargs(method: Method) -> dict[str, Any]:
    """Map `Method`'s pluggable-nuisance fields onto the kwargs
    `ADJUSTMENTS[method.name]` actually accepts.

    Pure mapping: illegal per-name field combinations are refused by
    `Method` at construction, so there is nothing left to reject here.
    """
    if method.name == "iptw":
        if method.propensity_learner is not None:
            return {"learner": method.propensity_learner}
        return {}
    if method.name in ("dml", "aipw"):
        kwargs: dict[str, Any] = {}
        if method.propensity_learner is not None:
            kwargs["propensity_learner"] = method.propensity_learner
        if method.outcome_learner is not None:
            kwargs["outcome_learner"] = method.outcome_learner
        if method.folds is not None:
            kwargs["folds"] = method.folds
        return kwargs
    return {}


# estimate_ate: the observational orchestrator


# Entry validations reading only the declared metric set and caller mappings
# (never `methods`), so they're safe as preconditions from either entry order.


def _reject_unsupported_prior_type(prior: Prior | None) -> None:
    """Refuse a prior whose posterior this path cannot reconstruct."""
    if isinstance(prior, (StudentTPrior, MixturePrior)):
        _refuse("estimation.adjust.prior.type")


def _reject_percentile_winsorization(metrics: Sequence[Metric]) -> None:
    """Refuse a percentile-winsorized metric: its data-derived cutoffs have no adjusted route."""
    for metric in metrics:
        config = getattr(metric, "winsorization", None)
        if config is not None and config.has_percentile:
            _refuse("adjust.winsorization.percentile_unsupported", metric=metric.name)


def _validate_prior_method_scales(
    methods: Sequence[Method],
    prior: Prior | None,
) -> None:
    """Refuse one informative prior spanning incompatible method scales."""
    if prior is None:
        return
    has_log_rr = any(method.name == "unadjusted" for method in methods)
    has_linear_relative = any(method.name != "unadjusted" for method in methods)
    if has_log_rr and has_linear_relative:
        _refuse("estimation.adjust.prior.method_scale")


def _validate_global_prior_method_scales(
    configs: Sequence[Any],
    methods_by_config: Sequence[Sequence[Method]],
) -> None:
    """Refuse one call-wide prior spanning incompatible metric methods."""
    groups: list[tuple[Prior, list[Method]]] = []
    for index, config in enumerate(configs):
        prior = getattr(config, "prior", None)
        if prior is None or not getattr(config, "prior_is_global", False):
            continue
        methods = list(methods_by_config[index])
        metric_type = getattr(getattr(config, "metric", None), "type", None)
        methods = [
            method
            for method in methods
            if (
                method.name == "unadjusted"
                and metric_type != "quantile"
                or method.name != "unadjusted"
                and not (
                    method.name in _BUILTIN_ADJUSTMENTS and metric_type in ("quantile", "ratio")
                )
            )
        ]
        if not methods:
            continue
        for existing_prior, existing_methods in groups:
            if existing_prior == prior:
                existing_methods.extend(methods)
                break
        else:
            groups.append((prior, methods))

    for prior, methods in groups:
        _validate_prior_method_scales(methods, prior)


def _reject_unknown_metric_keys(
    by_name: Mapping[str, Metric],
    label: str,
    mapping: Mapping[str, object] | None,
) -> None:
    """Refuse a per-metric mapping naming a metric the source never declared."""
    unknown = set(mapping or ()) - set(by_name)
    if unknown:
        _refuse(
            "estimation.adjust.names_metrics_source",
            label=label,
            unknown=sorted(unknown),
            by_name=sorted(by_name),
        )


def _reject_unusable_value_scales(
    by_name: Mapping[str, Metric],
    value_scale: Mapping[str, ValueScale] | None,
) -> None:
    """Refuse a `value_scale=` request no metric could honor.

    Runs before any scale is compared against another, so a typo'd scale
    value can never be counted as a distinct scale by the shared-prior
    judgment, which would otherwise assert a relative/absolute split the
    request does not actually contain.
    """
    for name, requested in (value_scale or {}).items():
        if requested not in VALUE_SCALE_VALUES:
            _refuse("estimation.adjust.value_scale_relative", name=name, requested=requested)
        metric_type = getattr(by_name[name], "type", None)
        if metric_type in ("quantile", "ratio"):
            # An explicit opt-in must never degrade to the NotImplementedError
            # skip-with-warning channel: the caller named this metric on purpose.
            _refuse("estimation.adjust.value_scale_names", metric_type=metric_type, name=name)


def _validate_request_mappings(
    src: MomentSource,
    *,
    value_scale: Mapping[str, ValueScale] | None,
    null_lifts: Mapping[str, float] | None,
    null_abs: Mapping[str, float] | None,
    alternatives: Mapping[str, str] | None,
) -> dict[str, Metric]:
    """Pure request-shape checks on the per-metric mappings; returns the declared metrics.

    Idempotent and warning-free, so `estimate_ate` can run it ahead of the
    observational-quantile refusal and leave method and shared-prior judgments after it.
    """
    by_name = {m.name: m for m in src.context.metrics}
    for label, mapping in (
        ("value_scale", value_scale),
        ("null_lifts", null_lifts),
        ("null_abs", null_abs),
        ("alternatives", alternatives),
    ):
        _reject_unknown_metric_keys(by_name, label, mapping)
    _reject_unusable_value_scales(by_name, value_scale)
    return by_name


def _resolve_value_scales(
    src: MomentSource,
    methods: list[Method],
    *,
    selected: Sequence[Metric],
    prior: Prior | None,
    prior_shared: bool,
    value_scale: Mapping[str, ValueScale] | None,
    null_lifts: Mapping[str, float] | None,
    null_abs: Mapping[str, float] | None,
    alternatives: Mapping[str, str] | None,
    prior_scale_judged: bool = False,
) -> dict[str, ValueScale]:
    """Validate `estimate_ate`'s per-metric mappings at entry and return the
    effective reporting scale per declared metric name.

    Every refusal fires before any estimation, so a typo or unsupported
    metric type never surfaces as a partial result set. Key validation
    (unknown names, per-key scale/null conflicts) runs against every
    metric `src` declares; the cross-method/cross-metric judgments (the
    `unadjusted`-cannot-be-absolute refusal and shared-prior
    scale-uniformity) run only over *selected*, since both compare
    `value_scale` against something call-specific (`methods` or one
    shared `prior`) and `readouts.run` dispatches one call per group of
    metrics sharing a resolved `methods`/`prior`.

    `prior_shared` gates the scale-uniformity judgment (it exists only to
    catch one scalar `prior=` spanning metrics with different units).
    `prior_scale_judged` lets a caller that already judged its own full
    metric set (e.g. `readouts.run`, which dispatches per group and would
    otherwise see only a fragment) skip the re-check here.
    """
    _reject_unsupported_prior_type(prior)
    by_name = _validate_request_mappings(
        src,
        value_scale=value_scale,
        null_lifts=null_lifts,
        null_abs=null_abs,
        alternatives=alternatives,
    )

    for name, requested in (value_scale or {}).items():
        if requested == "absolute" and name in (null_abs or {}):
            _refuse(
                "estimation.adjust.resolve_value_scales_null_abs_on_absolute_metric", metric=name
            )
        if requested == "absolute" and (null_lifts or {}).get(name, 0.0) != 0.0:
            # Symmetric with the null_abs refusal above: a relative null on
            # an additive row is a silent scale mismatch; zero is exempt.
            _refuse(
                "estimation.adjust.resolve_value_scales_null_lift_on_absolute_metric", metric=name
            )

    selected_names = {m.name for m in selected}
    # Scoped to `selected`: `methods` is this call's own, so a sibling
    # group's absolute metric is not this group's problem.
    absolute_requested = [
        n for n, s in (value_scale or {}).items() if s == "absolute" and n in selected_names
    ]
    if absolute_requested and any(m.name == "unadjusted" for m in methods):
        _refuse("estimation.adjust.method_name_unadjusted")

    scales: dict[str, ValueScale] = {
        name: (value_scale or {}).get(name, "relative") for name in by_name
    }
    if prior_shared and not prior_scale_judged:
        judge_shared_prior_scales(
            by_name, selected_names=selected_names, prior=prior, value_scale=value_scale
        )
    return scales


def judge_shared_prior_scales(
    by_name: Mapping[str, Metric],
    *,
    selected_names: Container[str],
    prior: Prior | None,
    value_scale: Mapping[str, ValueScale] | None,
    stacklevel: int = 4,
) -> None:
    """Refuse (or warn about) ONE scalar `prior=` spanning metrics whose
    reporting scales are not commensurable.

    Judged over the caller's whole selected set, never a dispatch
    fragment: a caller that splits its metrics across several
    `estimate_ate` calls must call this once itself and pass
    `_prior_scale_judged=True` down. Scale-uniformity is judged over
    *selected_names* minus ratio/quantile metrics, which can never
    produce a row (skip-with-warning and CapabilityError respectively),
    so counting them would refuse a legitimately uniform call.

    The three metric-set-wide entry validations run here first, so a
    caller that hoists this judgment ahead of `_resolve_value_scales`
    still reports the more fundamental refusal for a compound-invalid
    request; they are pure and idempotent, so a second run is a no-op.
    The `unadjusted`-cannot-be-absolute refusal deliberately stays in
    `_resolve_value_scales` instead, since it reads per-call `methods`.
    """
    _reject_unsupported_prior_type(prior)
    _reject_unknown_metric_keys(by_name, "value_scale", value_scale)
    _reject_unusable_value_scales(by_name, value_scale)
    if prior is None:
        return
    estimable = {
        name: (value_scale or {}).get(name, "relative")
        for name, metric in by_name.items()
        if name in selected_names and getattr(metric, "type", None) not in ("quantile", "ratio")
    }
    if not estimable:
        return
    distinct = set(estimable.values())
    if len(distinct) > 1:
        _refuse("estimation.adjust.prior_interpreted_call")
    if distinct == {"absolute"} and len(estimable) > 1:
        _warn(
            "estimation.adjust.prior_absolute_scale_spans_metrics",
            n_metrics=len(estimable),
            names=sorted(estimable),
            stacklevel=stacklevel,
        )


def _refuse_observational_quantiles(
    metrics: Sequence[Metric], *, source: object | None = None
) -> None:
    """An observational design has no quantile estimator: refuse before any source read."""
    for metric in metrics:
        if getattr(metric, "type", None) == "quantile":
            refuse_observational_quantile(metric, source=source)


def validate_readout_adjustment(request: ReadoutRequest) -> None:
    """Validate static adjustment/value-scale compatibility at the seam."""
    from increment.estimation.inference import MixturePrior, StudentTPrior

    design = request.design
    mechanism = getattr(design, "mechanism", None)
    metrics = tuple(request.metrics)
    configs = tuple(request.configs)
    method_catalog = request.estimation_methods
    value_scale = request.value_scale
    ratio_incapable = {"aipw", "dml", "iptw"}
    if mechanism == "observational" and metrics:
        from increment.errors import refuse
        from increment.estimation._adjust.common import SUPPORTED_RATIO_METRIC

        configs_by_name = {config.metric.name: config for config in configs}
        decisions = [
            (metric, methods[0])
            for metric, methods in zip(metrics, method_catalog, strict=True)
            if methods
        ]
        # A member of a family that needs complete evidence refuses first, so
        # its context names the family whatever else the request holds.
        for metric, method in decisions:
            procedure = request.plan.procedures[metric.name]
            membership = procedure.family
            family = getattr(membership, "family", None)
            correction = getattr(family, "correction", None)
            if (
                metric.type == "ratio"
                and configs_by_name[metric.name].prior is None
                and getattr(membership, "member", False)
                and correction in ("bh", "e_bh")
                and method.name.lower() in ratio_incapable
            ):
                refuse(
                    SUPPORTED_RATIO_METRIC,
                    method=method.name,
                    metric=metric.name,
                    role=procedure.role,
                    family=getattr(family, "name", None),
                    correction=correction,
                )
        if (
            decisions
            and all(metric.type == "ratio" for metric in metrics)
            and all(method.name.lower() in ratio_incapable for _, method in decisions)
        ):
            metric, method = decisions[0]
            refuse(
                SUPPORTED_RATIO_METRIC,
                method=method.name,
                metric=metric.name,
                role=request.plan.procedures[metric.name].role,
                family=None,
                correction=None,
            )
    plan = request.plan
    if mechanism == "observational" and request.by:
        _refuse("readout.view.observational", view="breakout")

    by_name = {metric.name: metric for metric in metrics}
    for name, scale in value_scale.items():
        if scale not in VALUE_SCALE_VALUES:
            _refuse("readout.value_scale.invalid", metric=name, value_scale=scale)
        metric = by_name.get(name)
        if metric is None:
            continue
        if getattr(metric, "type", None) in ("quantile", "ratio") and scale == "absolute":
            _refuse(
                "readout.value_scale.invalid",
                metric=name,
                value_scale=scale,
            )
        procedure = plan.procedures[name]
        if scale == "absolute" and (
            getattr(procedure, "null_abs", None) is not None
            or getattr(procedure, "null_lift", 0.0) != 0.0
        ):
            _refuse("readout.value_scale.null", metric=name)
    if mechanism == "observational":
        # Request-shape checks above come first; this is the estimator-capability refusal.
        _refuse_observational_quantiles(metrics)
        for metric, config, methods in zip(metrics, configs, method_catalog, strict=True):
            prior = config.prior
            if isinstance(prior, (StudentTPrior, MixturePrior)):
                _refuse("readout.observational.prior")
            _validate_prior_method_scales(methods, prior)
            if value_scale.get(metric.name) == "absolute" and any(
                method.name == "unadjusted" for method in methods
            ):
                _refuse("readout.adjustment.absolute_unadjusted")
        _validate_global_prior_method_scales(configs, method_catalog)


def _winsorization_diagnostics(
    src: MomentSource,
    metric: Metric,
    control_group: str,
) -> dict[str, dict[str, int | float | None]]:
    """Return contrast diagnostics keyed by treatment group for one metric."""
    rows = cast("list[Mapping[str, Any]]", src.moments(metric))
    arms = _df_to_arms(rows)
    control = next(
        (arm for arm in arms if arm.group_id == control_group),
        None,
    )
    if control is None:
        return {}
    return {
        treatment.group_id: _winsorization_result_fields(control, treatment)
        for treatment in arms
        if treatment.group_id != control_group
    }


# Public estimator signature is the API for adjustment results.
def estimate_ate(  # noqa: PLR0913
    src: MomentSource,
    design: Observational,
    *,
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    prior_shared: bool = True,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    value_scale: Mapping[str, ValueScale] | None = None,
    null_lifts: Mapping[str, float] | None = None,
    null_abs: Mapping[str, float] | None = None,
    alternatives: Mapping[str, str] | None = None,
    metrics: Sequence[Metric] | None = None,
    _raise_if_empty: bool = True,
    _prior_scale_judged: bool = False,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
) -> DecisionComputation[LiftEstimate]:
    """Estimate ATE-scale lift for every declared metric under an
    observational `design`.

    Defaults to `[Method(name="iptw")]`; `Method(name="unadjusted")` is
    allowed only when named explicitly (confounded, labelled accordingly,
    via joint unit-frame state when clustered and `estimate_lift` otherwise). Other names dispatch
    through `ADJUSTMENTS` (`"iptw"`, `"dml"`, `"aipw"`).

    One informative prior may cover only one method parameterization per call:
    unadjusted log-RR and adjusted linear-relative methods must be requested
    separately.

    `value_scale` is the per-metric reporting selector (e.g.
    `{"latency_delta": "absolute"}`): it reports that metric as the
    additive ATE in its own units, remedying the near-zero-control-mean
    refusal (`tau / mu0` is unidentified within 4 SEs of 0, but `tau`
    itself is identified there). Absolute rows carry no `abs_diff`/
    `abs_se` sidecar and, with `prior=None`, posterior `Normal(tau, se)`
    exactly. Reporting scale is never inferred from the data.

    `null_lifts`/`null_abs`/`alternatives` are per-metric decision
    mappings (this runs once per metric, unlike the randomized path);
    every key must name a declared metric.

    A mixed-type metric list does not abort the call: a metric an
    adjustment refuses (`NotImplementedError`) is skipped with a
    `UserWarning`, and every other metric's estimate still returns; if
    every metric is refused this way the call raises `NotImplementedError`
    instead of an empty result set (only `NotImplementedError` is caught
    this way; `IdentificationError` and other `ValueError`s abort the
    whole call). A cluster declared on the source (`src.cluster`) rides
    into every branch; an informative `prior` refuses up front under one.

    `metrics` narrows the declared metric set (`None` selects every
    metric `src` declares); it scopes the per-method loops and
    `_resolve_value_scales`'s cross-metric judgments, but not per-mapping
    key validation, which always checks every source-declared name.

    `_raise_if_empty`, `prior_shared` and `_prior_scale_judged` are
    `readouts.run` plumbing: they let a caller splitting metrics across
    several calls judge "nothing estimated" and prior scale-uniformity
    once, across every group (see `judge_shared_prior_scales`).
    """
    return _estimate_ate(
        src,
        design,
        methods=methods,
        prior=prior,
        prior_shared=prior_shared,
        alpha=alpha,
        alternative=alternative,
        value_scale=value_scale,
        null_lifts=null_lifts,
        null_abs=null_abs,
        alternatives=alternatives,
        metrics=metrics,
        _raise_if_empty=_raise_if_empty,
        _prior_scale_judged=_prior_scale_judged,
        method_roles=method_roles,
    )


def _estimate_ate(  # noqa: PLR0913, PLR0915
    src: MomentSource,
    design: Observational,
    *,
    methods: list[Method] | None = None,
    prior: Prior | None = None,
    prior_shared: bool = True,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    value_scale: Mapping[str, ValueScale] | None = None,
    null_lifts: Mapping[str, float] | None = None,
    null_abs: Mapping[str, float] | None = None,
    alternatives: Mapping[str, str] | None = None,
    metrics: Sequence[Metric] | None = None,
    _raise_if_empty: bool = True,
    _prior_scale_judged: bool = False,
    method_roles: Mapping[str, Literal["decision", "sensitivity"]] | None = None,
    route_alpha: float | None = None,
) -> DecisionComputation[LiftEstimate]:
    """``estimate_ate`` with the multiplicity routing level its families pass.

    ``route_alpha`` is the smallest level (in ``alpha``'s convention) a multiplicity procedure
    reads a p-value at (see ``_estimate_lift``'s ``route_alpha``): the unadjusted conversion
    rows of a family are routed at it. Only the package's own families set it; the public
    function never does, so a caller cannot move a row's ``reference_kind`` apart from a family.
    """
    if methods is None:
        methods = [Method(name="iptw")]
    _validate_unique_method_names(methods, caller="estimate_ate")
    resolved_method_roles: dict[str, Literal["decision", "sensitivity"]] = dict(method_roles or {})
    if method_roles is None and methods:
        from increment.estimation.engine import resolve_method_roles

        resolved_method_roles = resolve_method_roles(
            methods, prefer=lambda m: m.name != "unadjusted"
        )
    selected = list(src.context.metrics) if metrics is None else list(metrics)
    if methods == []:
        _refuse_observational_quantiles(selected, source=src)
        _reject_unsupported_prior_type(prior)
        from increment.estimation.decision_types import (
            ArmHypothesisKey,
            DecisionComputation,
            DecisionFailure,
        )

        failures: dict[Any, DecisionFailure] = {}
        for metric in selected:
            rows = cast("list[Mapping[str, Any]]", src.moments(metric))
            groups = {str(row["group_id"]) for row in rows}
            if str(design.control_group) not in groups:
                continue
            for group_id in groups - {str(design.control_group)}:
                hypothesis = ArmHypothesisKey(metric.name, group_id, "ate")
                failures[hypothesis] = DecisionFailure(
                    hypothesis,
                    "estimation.adjust.no_decision_method",
                    {"metric": metric.name, "group_id": group_id},
                )
        return DecisionComputation(results=(), evidence={}, failures=failures)

    cluster = src.context.cluster
    if cluster is not None and prior is not None:
        _refuse_compatibility("arm.adjustment.cluster_prior", cluster=cluster)
    # Refusal precedence, all before any source read: pure request shape (mapping keys,
    # scales), then estimator capability (an observational quantile has none, so method and
    # prior judgments about it are moot), then prior and method judgments and static
    # winsorization limits, then shared-prior advisories, which can warn and so come last.
    _validate_request_mappings(
        src,
        value_scale=value_scale,
        null_lifts=null_lifts,
        null_abs=null_abs,
        alternatives=alternatives,
    )
    _refuse_observational_quantiles(selected, source=src)
    _reject_unsupported_prior_type(prior)
    _validate_prior_method_scales(methods, prior)
    _reject_percentile_winsorization(selected)
    scales = _resolve_value_scales(
        src,
        methods,
        selected=cast("Sequence[Metric]", selected),
        prior=prior,
        prior_shared=prior_shared,
        value_scale=value_scale,
        null_lifts=null_lifts,
        null_abs=null_abs,
        alternatives=alternatives,
        prior_scale_judged=_prior_scale_judged,
    )

    winsor_diagnostics: dict[str, dict[str, dict[str, int | float | None]]] = {}
    for declared_metric in selected:
        if getattr(declared_metric, "winsorization", None) is None:
            continue
        winsor_diagnostics[declared_metric.name] = _winsorization_diagnostics(
            src,
            declared_metric,
            design.control_group,
        )

    results: list[LiftEstimate] = []
    refused_failures: dict[Any, DecisionFailure] = {}
    any_attempted = False
    for method in methods:
        if method.name == "unadjusted":
            any_attempted = True
            for metric in selected:
                if cluster is not None:
                    unadj = estimate_clustered_unadjusted(
                        src,
                        metric,
                        design.control_group,
                        method,
                        cluster=cluster,
                        alpha=alpha,
                        alternative=(alternatives or {}).get(metric.name, alternative),
                        null_lift=(null_lifts or {}).get(metric.name, 0.0),
                        null_abs=(null_abs or {}).get(metric.name),
                    )
                else:
                    unadj = _estimate_lift(
                        [metric],
                        cast("list[Mapping[str, Any]]", src.moments(metric)),
                        design.control_group,
                        methods=[method],
                        prior=prior,
                        alpha=alpha,
                        alternative=(alternatives or {}).get(metric.name, alternative),
                        null_lift=(null_lifts or {}).get(metric.name, 0.0),
                        null_abs=(null_abs or {}).get(metric.name),
                        preferred_direction=metric.declared_preferred_direction,
                        cluster=cluster,
                        method_roles=method_roles,
                        route_alpha=route_alpha,
                    ).results
                # The docstring promise "labelled accordingly" must be visible
                # on the row itself, not just the method name, for a report reader.
                results.extend(
                    r.model_copy(
                        update={
                            "method_role": resolved_method_roles.get(method.name, "decision"),
                            **(
                                winsor_diagnostics.get(metric.name, {}).get(r.group_id, {})
                                if cluster is not None
                                else {}
                            ),
                            "note": (
                                "unadjusted group-mean comparison under an "
                                "Observational design -- confounded; not a "
                                "causal estimate" + (f" | {r.note}" if r.note else "")
                            ),
                        }
                    )
                    for r in unadj
                )
        else:
            adjust_fn = ADJUSTMENTS.get(method.name)
            adjust_kwargs = _adjust_kwargs(method)
            if method.variance_reduction != "none":
                _refuse(
                    "adjust.variance_reduction.unsupported",
                    method=method.name,
                    variance_reduction=method.variance_reduction,
                )
            for metric in selected:
                try:
                    adjusted = adjust_fn(
                        src,
                        metric,
                        design,
                        prior=prior,
                        alpha=alpha,
                        alternative=(alternatives or {}).get(metric.name, alternative),
                        value_scale=scales[metric.name],
                        null_lift=(null_lifts or {}).get(metric.name, 0.0),
                        null_abs=(null_abs or {}).get(metric.name),
                        preferred_direction=metric.declared_preferred_direction,
                        **adjust_kwargs,
                    )
                    diagnostics_by_group = winsor_diagnostics.get(metric.name, {})
                    results.extend(
                        result.model_copy(
                            update={
                                "method_role": resolved_method_roles.get(method.name, "decision"),
                                **diagnostics_by_group.get(result.group_id, {}),
                            }
                        )
                        for result in adjusted
                    )
                    any_attempted = True
                except UnsupportedRequestError as exc:
                    if exc.code != "estimation.adjust_common.supported_ratio_metric":
                        raise
                    # A single unsupported metric (e.g. ratio under IPTW) must not
                    # abort every other metric's estimate; each adjust_fn already refuses itself.
                    _warn(
                        "estimation.adjust.skip_unsupported_metric",
                        metric_name=metric.name,
                        method_name=method.name,
                        exc=exc,
                        stacklevel=2,
                    )
                    if resolved_method_roles.get(method.name, "decision") == "decision":
                        from increment.estimation.decision_types import (
                            ArmHypothesisKey,
                            DecisionFailure,
                        )

                        for row in cast("list[Mapping[str, Any]]", src.moments(metric)):
                            if str(row["group_id"]) == design.control_group:
                                continue
                            hypothesis = ArmHypothesisKey(metric.name, str(row["group_id"]), "ate")
                            refused_failures[hypothesis] = DecisionFailure(
                                hypothesis,
                                "estimation.adjust.unavailable",
                                {"metric": metric.name, "method": method.name, "reason": str(exc)},
                            )
    if _raise_if_empty and not results and not any_attempted and selected and not refused_failures:
        _refuse("estimation.adjust.estimate_ate_every")
    from increment.estimation.decision_types import DecisionComputation
    from increment.estimation.engine import _lift_decision_bundle

    bundle = _lift_decision_bundle(
        results,
        inference=None,
        allow_linear=prior is None,
    )
    if not refused_failures:
        return bundle
    failures = dict(bundle.failures)
    failures.update(refused_failures)
    return DecisionComputation(
        results=bundle.results, evidence=dict(bundle.evidence), failures=failures
    )


ESTIMATION_ADJUST_ESTIMATE_ATE_EVERY = _REFUSALS["estimation.adjust.estimate_ate_every"]
