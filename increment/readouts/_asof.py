from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from increment._analysis_config import (
    UNSET,
    _Unset,
    normalize_display_correction,
    overlay_configs,
    select_metrics,
)
from increment._evidence_types import ComplianceSeriesRequest, DayAxisRequest, MetricEvidence
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
from increment.estimation.readout_types import serialize_day_axis
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
from increment.readouts._evidence import load_compliance_series, load_day_axis_rows
from increment.readouts._metric_rows import _rows_digest_and_arms
from increment.readouts._passes import _estimate_pass
from increment.semantics.design import Encouragement
from increment.sources import EvidenceSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan
    from increment.estimation.inference import Prior
    from increment.semantics.design import Randomized
    from increment.semantics.models import Metric


def _asof_metric_rows(rows, methods, control_group, cluster) -> MetricEvidence:
    """Identify the methods sharing one acquired implicit pre-period family."""
    rows = tuple(rows)
    observed, digest, count = _rows_digest_and_arms(
        {**row, "ds": serialize_day_axis(row["ds"])} for row in rows
    )
    component = ("moments_rows", digest, count)
    return MetricEvidence(
        cluster=cluster,
        control_group=control_group,
        rows=rows,
        unit_frame=None,
        observed_arms=observed,
        n_treatment_arms=len(observed - {control_group}),
        evidence_kind="moments_rows",
        evidence_sha256=digest,
        evidence_count=count,
        decision_evidence_components=(component,),
        evidence_components=(component,),
        method_rows=tuple(
            (method.name, rows) for method in methods if method.variance_reduction == "cuped"
        ),
    )


def _load_asof_rows_by_metric(
    src: EvidenceSource,
    selected: Sequence[Metric],
    methods_by_metric: Mapping[str, list[Method]],
    *,
    by: Sequence[str],
    completed_windows_only: bool,
    control_group: str,
) -> dict[str, dict[Any, MetricEvidence]]:
    """Load and bucket each metric's as-of moments by date."""
    covariate_metrics = frozenset(
        metric.name
        for metric in selected
        if any(method.variance_reduction == "cuped" for method in methods_by_metric[metric.name])
    )
    evidence = load_day_axis_rows(
        src,
        DayAxisRequest(
            caller="run_asof",
            grain="asof",
            metrics=tuple(selected),
            dimension=None,
            by=tuple(by),
            completed_windows_only=completed_windows_only,
            covariate_metrics=covariate_metrics,
        ),
    )
    by_metric_date: dict[str, dict[Any, MetricEvidence]] = {}
    for evidence_slice in evidence.slices:
        metric = evidence_slice.metrics[0]
        dated: dict[Any, list[Mapping[str, Any]]] = {}
        for row in evidence_slice.rows:
            dated.setdefault(row["ds"], []).append(row)
        by_metric_date[metric.name] = {
            date: _asof_metric_rows(
                rows, methods_by_metric[metric.name], control_group, src.context.cluster
            )
            for date, rows in dated.items()
        }
    return by_metric_date


@dataclass(frozen=True)
class _AsofMetricContext:
    source: EvidenceSource
    config: ResolvedMetricConfig
    procedure: Any
    design: Randomized | Encouragement
    plan: CompiledDecisionPlan
    estimands: Sequence[str] | None
    methods: list[Method]


def _asof_encouragement_rows(metric, evidence, context, alpha):
    results = []
    diagnostics = {}
    for method in context.methods:
        rows = (
            evidence.rows_for_method(method)
            if method.variance_reduction == "cuped"
            else evidence.rows
        )
        computation = estimate_encouragement(
            [metric],
            rows,
            context.design,
            estimands=tuple(
                name
                for name in (context.estimands if context.estimands is not None else ESTIMANDS)
                if name != "compliance"
            ),
            methods=[method],
            prior=context.config.prior,
            alpha=alpha,
            alternative=context.procedure.alternative,
            inference=_sequential_inference(context.plan),
            method_roles=_runtime_method_roles(context.methods),
        )
        for row in computation.results:
            if row.estimand == "compliance":
                diagnostics.setdefault((row.metric, row.group_id, row.value_scale), row)
            else:
                results.append(row)
    return [*results, *diagnostics.values()]


def _estimate_asof_metric_date(
    metric: Metric,
    evidence: MetricEvidence,
    date_value: Any,
    context: _AsofMetricContext,
) -> list[LiftEstimate]:
    """Estimate one fixed-horizon metric/date under the compiled role policy."""
    src, config, test = context.source, context.config, context.procedure
    design, plan = context.design, context.plan
    inference = _sequential_inference(plan)
    role_for_row = test.role if plan.declared else None
    if design.mechanism == "encouragement":
        n_arms = evidence.n_treatment_arms
        cell_alpha = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
        results = _asof_encouragement_rows(metric, evidence, context, cell_alpha)
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
            evidence,
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

    n_arms = evidence.n_treatment_arms
    if test.role == "primary" and n_arms == 0:
        return []
    cell_alpha = resolve_cell_alpha(plan, test, n_arms=n_arms, view=None)
    rows_est, _refused, _computation = _estimate_pass(
        src,
        metric,
        evidence,
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


def _sequential_asof_rows(
    src: EvidenceSource,
    *,
    metrics: Sequence[Metric],
    estimands: Sequence[str] | None,
    population: Literal["assigned", "triggered"],
    configs: Sequence[Any],
) -> list[LiftEstimate]:
    """One population's current checkpoint rows, labelled with that chain's reveal cursor.

    The scoped collection accounts for every cell, so an uncommitted triggered
    chain or an unsupported triggered uptake cell is reported, not omitted.
    """
    from increment._sequential_readouts import sequential_readout
    from increment.readouts._evidence import current_source_snapshot
    from increment.readouts._sequential_scope import scope_sequential_results
    from increment.sequential_state import sequential_refuse

    snapshot = current_source_snapshot(src)
    if snapshot.reveal_cursor is None:
        sequential_refuse(
            "route.unsupported", "as-of reporting requires a labeled finalized checkpoint"
        )
    rows = scope_sequential_results(
        src,
        sequential_readout(src, snapshot=snapshot, _include_unrequested=True, configs=configs),
        snapshot,
        metrics=[metric.name for metric in metrics],
        estimands=estimands,
        configs=configs,
    )
    chain = snapshot.chain(population) if snapshot.triggered is not None else snapshot
    return [
        row.model_copy(update={"ds": chain.reveal_cursor})
        for row in rows
        if row.analysis_population == population
    ]


def asof_lift(
    src: EvidenceSource,
    *,
    estimands: Sequence[str] | None = None,
    metrics: Sequence[str] | None = None,
    by: Sequence[str] = (),
    decision_method: Method | _Unset = UNSET,
    sensitivity_methods: Sequence[Method] | _Unset = UNSET,
    prior: Prior | None | _Unset = UNSET,
    completed_windows_only: bool = False,
    _population: Literal["assigned", "triggered"] = "assigned",
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
        _refuse_segmented_registration(plan.inference)
        return _sequential_asof_rows(
            src,
            metrics=selected,
            estimands=estimands,
            population=_population,
            configs=resolved_configs,
        )
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
    outcome_metrics = [] if estimands is not None and set(estimands) == {"compliance"} else selected
    methods_by_metric = {
        metric.name: _runtime_methods(configs[metric.name], design) for metric in outcome_metrics
    }
    from increment._frame_panel import _day_axis_label_order

    compliance_by_date = (
        dict(
            load_compliance_series(
                src,
                ComplianceSeriesRequest(
                    design=design,
                    completed_windows_only=completed_windows_only,
                    population=_population,
                ),
            )
        )
        if isinstance(design, Encouragement) and (estimands is None or "compliance" in estimands)
        else {}
    )
    by_metric_date = _load_asof_rows_by_metric(
        src,
        outcome_metrics,
        methods_by_metric,
        by=by,
        completed_windows_only=completed_windows_only,
        control_group=str(design.control_group),
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
            cast("Randomized | Encouragement", design),
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
