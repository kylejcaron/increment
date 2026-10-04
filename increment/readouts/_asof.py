from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._policy_alpha import resolve_cell_alpha
from increment._readout_request import ReadoutRequest, validate_request
from increment._readout_request import _raise as _raise_readout_request
from increment.breakout.estimates import _asof_monitoring_note
from increment.estimation.encouragement import (
    ESTIMANDS,
    estimate_compliance,
    estimate_encouragement,
)
from increment.estimation.engine import Method
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import SEQUENTIAL_POLICIES
from increment.readouts._common import (
    _declared_margin_names,
    _raise,
    _refuse_segmented_registration,
    _require_design,
    _runtime_method_roles,
    _runtime_methods,
    _sequential_inference,
    _validate_encouragement_asof_inference,
)
from increment.readouts._passes import _estimate_pass
from increment.semantics.design import Encouragement
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.inference import Prior
    from increment.semantics.design import Observational, Randomized
    from increment.semantics.models import Metric


def _load_asof_rows_by_metric(
    src: MomentSource,
    selected: Sequence[Metric],
    methods_by_metric: Mapping[str, list[Method]],
    *,
    by: Sequence[str],
    completed_windows_only: bool,
) -> dict[str, dict[Any, list[Mapping[str, Any]]]]:
    """Load and bucket each metric's as-of moments by date."""
    common_moment_kwargs: dict[str, Any] = (
        {"completed_windows_only": True} if completed_windows_only else {}
    )
    by_metric_date: dict[str, dict[Any, list[Mapping[str, Any]]]] = {}
    for metric in selected:
        moment_kwargs = dict(common_moment_kwargs)
        if any(method.variance_reduction == "cuped" for method in methods_by_metric[metric.name]):
            moment_kwargs["include_covariate"] = True
        rows = cast(
            "list[Mapping[str, Any]]",
            src.moments(metric, grain="asof", by=by, **moment_kwargs),
        )
        dated: dict[Any, list[Mapping[str, Any]]] = {}
        for row in rows:
            dated.setdefault(row["ds"], []).append(row)
        by_metric_date[metric.name] = dated
    return by_metric_date


@dataclass(frozen=True)
class _AsofMetricContext:
    source: MomentSource
    config: ResolvedMetricConfig
    procedure: Any
    design: Randomized | Encouragement | Observational
    plan: CompiledDecisionPlan
    estimands: Sequence[str] | None
    methods: list[Method]


def _estimate_asof_metric_date(
    metric: Metric,
    date_rows: list[Mapping[str, Any]],
    date_value: Any,
    context: _AsofMetricContext,
) -> list[LiftEstimate]:
    """Estimate one fixed-horizon metric/date under the compiled role policy."""
    src, config, test = context.source, context.config, context.procedure
    design, plan, estimands = context.design, context.plan, context.estimands
    inference = _sequential_inference(plan)
    role_for_row = test.role if plan.declared else None
    if design.mechanism == "encouragement":
        n_arms = len({str(row["group_id"]) for row in date_rows} - {str(design.control_group)})
        cell_alpha = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
        results = estimate_encouragement(
            [metric],
            date_rows,
            design,
            estimands=tuple(
                name
                for name in (estimands if estimands is not None else ESTIMANDS)
                if name != "compliance"
            ),
            methods=context.methods,
            prior=config.prior,
            alpha=cell_alpha,
            alternative=test.alternative,
            inference=inference,
            method_roles=_runtime_method_roles(context.methods),
        ).results
        return [
            row.model_copy(
                update={
                    "ds": date_value,
                    "role": role_for_row,
                    "note": _asof_monitoring_note(
                        row.note,
                        estimand=row.estimand,
                        inference=inference,
                        design=design,
                    ),
                }
            )
            for row in results
        ]

    if test.role == "secondary":
        nominal, _refused, _computation = _estimate_pass(
            src,
            metric,
            date_rows,
            test,
            config,
            design,
            alpha_for=plan.alpha,
            role="secondary",
            inference=inference,
            advisory_seen=None,
            retry="whole_asof",
            methods=context.methods,
        )
        return [
            row.model_copy(
                update={
                    "ds": date_value,
                    "role": "secondary",
                    "discovery": None,
                    "note": _asof_monitoring_note(
                        row.note,
                        estimand=row.estimand,
                        inference=inference,
                        design=design,
                        secondary_fixed_horizon=True,
                    ),
                }
            )
            for row in nominal
        ]

    n_arms = len({str(row["group_id"]) for row in date_rows} - {str(design.control_group)})
    if test.role == "primary" and n_arms == 0:
        return []
    cell_alpha = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
    rows_est, _refused, _computation = _estimate_pass(
        src,
        metric,
        date_rows,
        test,
        config,
        design,
        alpha_for=cell_alpha,
        role=role_for_row,
        inference=inference,
        advisory_seen=None,
        retry="whole_asof",
        methods=context.methods,
    )
    return [
        row.model_copy(
            update={
                "ds": date_value,
                "role": role_for_row,
                "note": _asof_monitoring_note(
                    row.note,
                    estimand=row.estimand,
                    inference=inference,
                    design=design,
                ),
            }
        )
        for row in rows_est
    ]


def asof_lift(
    src: MomentSource,
    *,
    estimands: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    completed_windows_only: bool = False,
) -> list[LiftEstimate]:
    """As-of relative lift under the source's declared plan.

    Fixed-horizon rows are calculated per date with the existing role and
    multiplicity policy and carry the repeated-look caveat. Registered
    AlwaysValid returns the current labeled finalized joint-unit checkpoint;
    historical dates are not reconstructed from rounded moments. Its selected
    intervals invert the same stopped likelihood at the exact FCR allocation.
    Segmented histories require a per-date state contract and remain
    unsupported here.
    """
    if src.context.cluster is not None:
        _raise_readout_request("readout.asof.cluster", cluster=src.context.cluster)
    design = _require_design(src, "asof_lift")
    if by:
        _raise_readout_request("readout.asof.segment_unsupported", by=tuple(by))
    plan = src.context.plan
    selected = select_metrics(
        cast("Sequence[Metric]", src.context.metrics), metrics, caller="asof_lift"
    )
    resolved_configs = overlay_configs(
        selected,
        src.context.configs,
        methods=None,
        prior=None,
        decision_method=decision_method,
        sensitivity_methods=sensitivity_methods,
        prior_override=prior,
    )
    configs = {config.metric.name: config for config in resolved_configs}
    policy = plan.view_policies.for_view(
        "asof",
        mechanism=design.mechanism if design is not None else None,
        segmented=bool(by),
    )
    request = ReadoutRequest.from_source(
        src,
        metrics=selected,
        configs=resolved_configs,
        view="asof",
        grain="asof",
        by=by,
        estimands=estimands,
        correction=normalize_display_correction(policy.correction),
        q=policy.q,
        completion_policy=completed_windows_only,
    )
    validate_request(request)
    _validate_encouragement_asof_inference(
        () if estimands is not None and set(estimands) == {"compliance"} else selected,
        design,
        inference=_sequential_inference(plan),
        completed_windows_only=completed_windows_only,
        method="readouts.asof_lift",
    )
    if isinstance(plan.inference, SEQUENTIAL_POLICIES):
        from increment._sequential_readouts import sequential_asof_readout

        _refuse_segmented_registration(plan.inference)
        return sequential_asof_readout(src, metrics=selected, estimands=estimands)
    if design.mechanism == "encouragement":
        declared = _declared_margin_names(selected)
        shifted = [
            metric.name
            for metric in selected
            if getattr(plan.procedures[metric.name], "null_lift", 0.0) != 0.0
            or getattr(plan.procedures[metric.name], "null_abs", None) is not None
        ]
        combined = list(dict.fromkeys([*shifted, *declared]))
        if combined:
            _raise("readout.metric_declare_non", combined=combined)
    from increment._frame_panel import _day_axis_label_order
    from increment._source_types import compliance_summary_series

    compliance_by_date = (
        compliance_summary_series(
            src,
            design,
            completed_windows_only=completed_windows_only,
        )
        if isinstance(design, Encouragement) and (estimands is None or "compliance" in estimands)
        else {}
    )
    outcome_metrics = [] if estimands is not None and set(estimands) == {"compliance"} else selected
    methods_by_metric = {
        metric.name: _runtime_methods(configs[metric.name], design) for metric in outcome_metrics
    }
    by_metric_date = _load_asof_rows_by_metric(
        src,
        outcome_metrics,
        methods_by_metric,
        by=by,
        completed_windows_only=completed_windows_only,
    )
    day_order = _day_axis_label_order(
        [*compliance_by_date, *(date for dated in by_metric_date.values() for date in dated)]
    )
    all_dates = sorted(day_order, key=day_order.__getitem__)
    out: list[LiftEstimate] = []
    contexts = {
        metric.name: _AsofMetricContext(
            src,
            configs[metric.name],
            plan.procedures[metric.name],
            design,
            plan,
            estimands,
            methods_by_metric[metric.name],
        )
        for metric in outcome_metrics
    }
    for date_value in all_dates:
        compliance_summary = compliance_by_date.get(date_value)
        for metric in outcome_metrics:
            date_rows = by_metric_date[metric.name].get(date_value)
            if date_rows is None:
                continue
            rows = _estimate_asof_metric_date(
                metric,
                date_rows,
                date_value,
                contexts[metric.name],
            )
            out.extend(rows)
        if compliance_summary is not None:
            assert isinstance(design, Encouragement)
            out.extend(
                r.model_copy(
                    update={
                        "ds": date_value,
                        "role": None,
                        "note": _asof_monitoring_note(
                            r.note,
                            estimand=r.estimand,
                            inference=_sequential_inference(plan),
                            design=design,
                        ),
                    }
                )
                for r in estimate_compliance(
                    compliance_summary,
                    design,
                    alpha=plan.alpha,
                    inference=_sequential_inference(plan),
                ).results
            )
    order = {metric.name: i for i, metric in enumerate(selected)}
    out.sort(
        key=lambda row: (
            day_order[row.ds],
            -1 if row.estimand == "compliance" else order[row.metric],
        )
    )
    return out
