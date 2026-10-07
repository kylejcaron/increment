from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from increment._literals import ValueScale
from increment._policy_alpha import resolve_cell_alpha
from increment.estimation.adjust import (
    ObservationalEvidence,
    _estimate_ate,
    _weight_diagnostics_projection,
    judge_shared_prior_scales,
)
from increment.estimation.conversion_route import family_route_alpha
from increment.estimation.engine import Method
from increment.estimation.engine import merge_decision_computations as _merge_decision_computations
from increment.estimation.family import decision_cells, family_discovery, select_family
from increment.estimation.results import (
    LiftEstimate,
    _fcr_alpha_for,
    open_bound_from_two_sided_at_target,
)
from increment.readouts._common import (
    _raise,
    _runtime_method_roles,
    _runtime_methods,
    _sequential_inference,
)
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan, DecisionComputation
    from increment.estimation.inference import Prior
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric


def _estimate_observational(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Observational,
    plan: CompiledDecisionPlan,
    *,
    value_scale: Mapping[str, ValueScale] | None,
    call_prior: Prior | None,
    evidence: ObservationalEvidence,
    computations: list[DecisionComputation[LiftEstimate]] | None = None,
) -> list[LiftEstimate]:
    """Estimate the observational phase after whole-window validation.

    Applies the same role machinery randomized readouts use: a primary's
    alpha is Bonferroni-split across its treatment arms, a guardrail keeps
    full alpha (intersection-union, never joins a family), and secondaries
    are BH-selected at plan.q then re-estimated at the Benjamini-Yekutieli
    FCR level for the selected cells.
    """
    metric_names = [metric.name for metric in selected]
    if value_scale:
        unknown_value_scale = set(value_scale) - set(metric_names)
        if unknown_value_scale:
            _raise(
                "readout.value_scale_names",
                metric_names=sorted(metric_names),
                unknown_value_scale=sorted(unknown_value_scale),
            )
    resolved_null_abs: dict[str, float] = {
        metric.name: float(cast(Any, plan.procedures[metric.name]).null_abs)
        for metric in selected
        if getattr(plan.procedures[metric.name], "null_abs", None) is not None
    }
    resolved_alternatives: dict[str, str] = {
        metric.name: plan.procedures[metric.name].alternative for metric in selected
    }
    judge_shared_prior_scales(
        {metric.name: metric for metric in src.context.metrics},
        selected_names={metric.name for metric in selected},
        prior=call_prior,
        value_scale=value_scale,
    )
    config_by_name = {metric.name: config for metric, config in zip(selected, configs, strict=True)}
    methods_by_metric = {
        metric.name: _runtime_methods(config_by_name[metric.name], design) for metric in selected
    }
    selected = [metric for metric in selected if methods_by_metric[metric.name]]
    non_secondary = [m for m in selected if plan.procedures[m.name].role != "secondary"]
    secondary = [m for m in selected if plan.procedures[m.name].role == "secondary"]

    results: list[LiftEstimate] = []
    for metric in non_secondary:
        config = config_by_name[metric.name]
        procedure = plan.procedures[metric.name]
        n_arms = (
            len(evidence.arms(metric) - {str(design.control_group)})
            if procedure.role == "primary"
            else 0
        )
        alpha = resolve_cell_alpha(plan, procedure, n_arms=n_arms, view=None)
        computation = _estimate_ate(
            src,
            design,
            methods=methods_by_metric[metric.name],
            metrics=[metric],
            method_roles=_runtime_method_roles(methods_by_metric[metric.name]),
            prior=config.prior,
            prior_shared=config.prior_is_global,
            alpha=alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            _raise_if_empty=False,
            _prior_scale_judged=True,
            evidence=evidence,
        )
        if computations is not None:
            computations.append(computation)
        results.extend(
            row.model_copy(update={"role": procedure.role if plan.declared else None})
            for row in computation.results
        )

    if secondary:
        results.extend(
            _estimate_observational_secondary_family(
                src,
                secondary,
                config_by_name,
                design,
                plan,
                methods_by_metric=methods_by_metric,
                value_scale=value_scale,
                resolved_null_abs=resolved_null_abs,
                resolved_alternatives=resolved_alternatives,
                evidence=evidence,
                computations=computations,
            )
        )

    if not results and selected:
        _raise("estimation.adjust.estimate_ate_every")
    order = {metric.name: i for i, metric in enumerate(selected)}
    results.sort(key=lambda result: order[result.metric])
    return [
        row.model_copy(update=_weight_diagnostics_projection(row.method))
        if row.method not in {"iptw", "aipw"} and row.weight_diagnostics_available is None
        else row
        for row in results
    ]


def _estimate_observational_secondary_family(
    src: MomentSource,
    secondary: Sequence[Metric],
    config_by_name: Mapping[str, ResolvedMetricConfig],
    design: Observational,
    plan: CompiledDecisionPlan,
    *,
    methods_by_metric: Mapping[str, list[Method]],
    value_scale: Mapping[str, ValueScale] | None,
    resolved_null_abs: Mapping[str, float] | None,
    resolved_alternatives: Mapping[str, str] | None,
    evidence: ObservationalEvidence,
    computations: list[DecisionComputation[LiftEstimate]] | None = None,
) -> list[LiftEstimate]:
    """Nominal pass at plan.alpha, BH-select at plan.q, FCR-reestimate the
    selected cells -- the observational sibling of
    _estimate_randomized_secondary_family, built on estimate_ate's own
    DecisionComputation[LiftEstimate] return value. Every secondary joins
    the family unless its config carries an informative prior (the same
    exemption randomized secondaries use).
    """
    from increment.decision import ArmHypothesisKey

    family = [m for m in secondary if config_by_name[m.name].prior is None]
    non_family = [m for m in secondary if config_by_name[m.name].prior is not None]
    roles_by_metric = {
        metric.name: _runtime_method_roles(methods_by_metric[metric.name]) for metric in secondary
    }

    out: list[LiftEstimate] = []

    def _nominal_pass(
        metric: Metric, alpha: float, route_alpha: float | None = None
    ) -> DecisionComputation[LiftEstimate]:
        config = config_by_name[metric.name]
        computation = _estimate_ate(
            src,
            design,
            methods=methods_by_metric[metric.name],
            metrics=[metric],
            method_roles=roles_by_metric[metric.name],
            prior=config.prior,
            prior_shared=config.prior_is_global,
            alpha=alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            _raise_if_empty=False,
            _prior_scale_judged=True,
            route_alpha=route_alpha,
            evidence=evidence,
        )
        if computations is not None:
            computations.append(computation)
        return computation

    for metric in non_family:
        out.extend(
            row.model_copy(update={"role": "secondary", "discovery": None})
            for row in _nominal_pass(metric, plan.alpha).results
        )
    if not family:
        return out

    nominal_by_metric: dict[str, list[LiftEstimate]] = {}
    computation_by_metric: dict[str, DecisionComputation[LiftEstimate]] = {}
    # The family reads each p-value at a threshold as small as ``q / m`` (``m`` its metric x arm
    # hypotheses): route its members at that level, never a looser one. ``m`` counts the arms of
    # the very evidence every estimate below is computed from.
    control = str(design.control_group)
    hypotheses = sum(len(evidence.arms(metric) - {control}) for metric in family)
    route_alpha = family_route_alpha(plan.q, hypotheses)
    for metric in family:
        computation = _nominal_pass(metric, plan.alpha, route_alpha)
        nominal_by_metric[metric.name] = list(computation.results)
        computation_by_metric[metric.name] = computation

    family_cells: list[tuple[ArmHypothesisKey, object]] = []
    for metric in family:
        seen_keys: set[ArmHypothesisKey] = set()
        for row in nominal_by_metric[metric.name]:
            if row.method_role != "decision":
                continue
            key = ArmHypothesisKey(metric.name, row.group_id, row.estimand)
            seen_keys.add(key)
            family_cells.append((key, row))
        # A failed arm has no row but still occupies its family slot with a None
        # estimate; dropping it would shrink m and loosen every BH threshold.
        # select_family then keeps an allowlisted outcome-degeneracy failure as a
        # non-rejection or refuses the whole family.
        for key in computation_by_metric[metric.name].failures:
            if isinstance(key, ArmHypothesisKey) and key not in seen_keys:
                family_cells.append((key, None))
    family_computation = _merge_decision_computations(
        [computation_by_metric[metric.name] for metric in family]
    )
    outcome = select_family(
        decision_cells(family_cells),
        plan.q,
        _sequential_inference(plan),
        plan.alpha,
        computation=family_computation,
    )
    selected_cells = outcome.selected
    fcr_alpha = outcome.fcr_alpha
    family_record = {
        "family_axes": ("metric", "arm"),
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
    }
    if fcr_alpha is None:
        for metric in family:
            for row in nominal_by_metric[metric.name]:
                is_decision = row.method_role == "decision"
                key = ArmHypothesisKey(row.metric, row.group_id, row.estimand)
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": family_discovery(outcome, key) if is_decision else None,
                            "family_axes": family_record["family_axes"] if is_decision else None,
                            "family_q": family_record["family_q"] if is_decision else None,
                            "family_threshold": (
                                family_record["family_threshold"] if is_decision else None
                            ),
                        }
                    )
                )
        return out

    selected_names = {key.metric for key in selected_cells}
    reestimated: dict[str, list[LiftEstimate]] = {}
    for metric in family:
        if metric.name not in selected_names:
            continue
        config = config_by_name[metric.name]
        procedure = plan.procedures[metric.name]
        selected_alpha = _fcr_alpha_for(procedure.alternative, fcr_alpha)
        computation = _estimate_ate(
            src,
            design,
            methods=methods_by_metric[metric.name][:1],
            metrics=[metric],
            method_roles=roles_by_metric[metric.name],
            prior=config.prior,
            prior_shared=config.prior_is_global,
            alpha=selected_alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            _raise_if_empty=False,
            _prior_scale_judged=True,
            evidence=evidence,
        )
        if computations is not None:
            computations.append(computation)
        reestimated[metric.name] = [
            open_bound_from_two_sided_at_target(r) for r in computation.results
        ]

    for metric in family:
        for row in nominal_by_metric[metric.name]:
            if row.method_role != "decision":
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": None,
                            "family_axes": None,
                            "family_q": None,
                            "family_threshold": None,
                        }
                    )
                )
                continue
            key = ArmHypothesisKey(row.metric, row.group_id, row.estimand)
            if key in selected_cells:
                match = next(
                    (
                        cell
                        for cell in reestimated.get(row.metric, [])
                        if cell.group_id == row.group_id and cell.method == row.method
                    ),
                    None,
                )
                stamped = match if match is not None else row
            else:
                stamped = row
            out.append(
                stamped.model_copy(
                    update={
                        "role": "secondary",
                        "discovery": family_discovery(outcome, key),
                        **family_record,
                    }
                )
            )
    return out
