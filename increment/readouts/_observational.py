from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from increment._literals import ValueScale
from increment._policy_alpha import resolve_cell_alpha
from increment._source_types import _BuiltinMomentSource
from increment.errors import CodedError
from increment.estimation.adjust import (
    ObservationalEvidence,
    _estimate_ate_data,
    judge_shared_prior_scales,
    weight_diagnostics_projection,
)
from increment.estimation.conversion_route import fcr_route_tail
from increment.estimation.engine import Method, _is_unadjusted_method
from increment.estimation.engine import merge_decision_computations as _merge_decision_computations
from increment.estimation.family import decision_cells, family_discovery, select_family
from increment.estimation.results import (
    LiftEstimate,
    _fcr_alpha_for,
    carry_nominal_evidence,
    open_bound_from_two_sided_at_target,
)
from increment.readouts._common import (
    _raise,
    _runtime_method_roles,
    _runtime_methods,
    _sequential_inference,
)
from increment.readouts._passes import _record_cell_refusal

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import (
        ArmHypothesisKey,
        CompiledDecisionPlan,
        DecisionComputation,
        DecisionFailure,
    )
    from increment.estimation.inference import Prior
    from increment.readouts._design_scope import ResolvedRoster
    from increment.semantics.design import Observational
    from increment.semantics.models import Metric


def _estimate_ate_from_evidence(  # noqa: PLR0913
    design: Observational,
    metric: Metric,
    *,
    methods: list[Method],
    roles: Mapping[str, Any],
    prior: Prior | None,
    alpha: float,
    value_scale: Mapping[str, ValueScale] | None,
    null_abs: Mapping[str, float] | None,
    alternatives: Mapping[str, str] | None,
    evidence: ObservationalEvidence,
    unit_frame_memo: Any,
    family_member: bool = False,
    route_tail: float | None = None,
) -> DecisionComputation[LiftEstimate]:
    def estimate(
        cell_methods: list[Method], cell_evidence: ObservationalEvidence
    ) -> DecisionComputation[LiftEstimate]:
        return _estimate_ate_data(
            design,
            metrics=(metric,),
            cluster=cell_evidence.cluster,
            methods=cell_methods,
            prior=prior,
            alpha=alpha,
            alternative="two-sided",
            value_scale=value_scale,
            null_lifts=None,
            null_abs=null_abs,
            alternatives=alternatives,
            scales={metric.name: (value_scale or {}).get(metric.name, "relative")},
            _raise_if_empty=False,
            resolved_method_roles=roles,
            method_roles=roles,
            evidence=cell_evidence,
            family_member=family_member,
            route_tail=route_tail,
            unit_frame_memo=unit_frame_memo,
        )

    from increment.decision import DecisionComputation

    control = str(design.control_group)
    rows = evidence.metric_rows(metric)
    control_row = next((row for row in rows if str(row["group_id"]) == control), None)
    if (
        evidence.cluster is not None
        or control_row is None
        or not any(_is_unadjusted_method(method) for method in methods)
    ):
        return estimate(methods, evidence)
    # An unadjusted contrast reads only its arm and control, so each arm is its own cell and a
    # cell-local guard leaves only that cell unavailable, as under a randomized design. An
    # adjusted method fits its models across every arm and is estimated on all of them.
    computations: list[DecisionComputation[LiftEstimate]] = []
    for method in methods:
        if not _is_unadjusted_method(method):
            computations.append(estimate([method], evidence))
            continue
        for row in rows:
            group_id = str(row["group_id"])
            if group_id == control:
                continue
            try:
                computations.append(
                    estimate([method], ObservationalEvidence({metric.name: (control_row, row)}))
                )
            except CodedError as exc:
                failed_rows: list[LiftEstimate] = []
                _record_cell_refusal(
                    exc,
                    results=failed_rows,
                    computations=computations,
                    metric=metric,
                    group_id=group_id,
                    method=method,
                    method_roles=roles,
                    alternative=(alternatives or {}).get(metric.name, "two-sided"),
                    null_lift=0.0,
                    null_abs=(null_abs or {}).get(metric.name),
                    preferred_direction=metric.declared_preferred_direction,
                    prior=prior,
                )
                if failed_rows:
                    computations.append(
                        DecisionComputation(results=tuple(failed_rows), evidence={}, failures={})
                    )
    return _merge_decision_computations(computations)


def _estimate_observational(
    src: _BuiltinMomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Observational,
    plan: CompiledDecisionPlan,
    *,
    value_scale: Mapping[str, ValueScale] | None,
    call_prior: Prior | None,
    evidence: ObservationalEvidence,
    unit_frame_memo: Any,
    roster: ResolvedRoster,
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
    control = str(design.control_group)
    # An outcome without observations in the control arm and a treatment arm has no
    # contrast: its cells are unavailable rows. A family member instead keeps its slot.
    selected = [
        metric
        for metric in selected
        if methods_by_metric[metric.name]
        and (
            getattr(plan.procedures[metric.name].family, "member", False)
            or (control in evidence.arms(metric) and evidence.arms(metric) - {control})
        )
    ]
    non_secondary = [m for m in selected if plan.procedures[m.name].role != "secondary"]
    secondary = [m for m in selected if plan.procedures[m.name].role == "secondary"]

    produced: list[DecisionComputation[LiftEstimate]] = []
    results: list[LiftEstimate] = []
    for metric in non_secondary:
        config = config_by_name[metric.name]
        procedure = plan.procedures[metric.name]
        # A primary's alpha is split across every arm the counts enroll, observed or not, and
        # every arm its own moments read.
        n_arms = (
            len((set(roster.counted_arms) | evidence.arms(metric)) - {control})
            if procedure.role == "primary"
            else 0
        )
        alpha = resolve_cell_alpha(plan, procedure, n_arms=n_arms, view=None)
        computation = _estimate_ate_from_evidence(
            design,
            metric,
            methods=methods_by_metric[metric.name],
            roles=_runtime_method_roles(methods_by_metric[metric.name]),
            prior=config.prior,
            alpha=alpha,
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            evidence=evidence,
            unit_frame_memo=unit_frame_memo,
        )
        produced.append(computation)
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
                unit_frame_memo=unit_frame_memo,
                arms=roster.enrolled_arms,
                computations=produced,
            )
        )

    if computations is not None:
        computations.extend(produced)
    failure_codes = {
        failure.code for computation in produced for failure in computation.failures.values()
    }
    # Nothing is left to report only when the adjustment method refused every metric; any
    # other failure is its own cell's unavailable row.
    if (
        not results
        and selected
        and failure_codes == {"estimation.adjust_common.supported_ratio_metric"}
    ):
        _raise("estimation.adjust.estimate_ate_every")
    order = {metric.name: i for i, metric in enumerate(selected)}
    results.sort(key=lambda result: order[result.metric])
    return [
        row.model_copy(update=weight_diagnostics_projection(row.method, row=row))
        if row.method not in {"iptw", "aipw"} and row.weight_diagnostics_available is None
        else row
        for row in results
    ]


def _unobserved_member_failures(
    family: Sequence[Metric],
    config_by_name: Mapping[str, ResolvedMetricConfig],
    design: Observational,
    plan: CompiledDecisionPlan,
    *,
    value_scale: Mapping[str, ValueScale] | None,
    evidence: ObservationalEvidence,
    arms: Sequence[str],
    estimated: Collection[tuple[str, str]],
) -> dict[ArmHypothesisKey, DecisionFailure]:
    """Family failures for the roster *arms* in which a member metric has no value, or for
    every roster arm when the metric has no control value, since no arm then has a contrast.

    Such a cell has no estimate, but the scope lists it as a member, so it keeps its family
    slot as a sure non-rejection (p = 1). *estimated* holds the (metric, arm) cells already in
    the family."""
    from increment.decision import ArmHypothesisKey, DecisionFailure
    from increment.readouts._design_scope import observational_expected

    control = str(design.control_group)
    failures: dict[ArmHypothesisKey, DecisionFailure] = {}
    members = set(estimated)
    for metric in family:
        observed = set(evidence.arms(metric))
        control_unobserved = control not in observed
        for cell in observational_expected(
            [metric],
            [config_by_name[metric.name]],
            design,
            plan,
            value_scale=value_scale,
            arms=[arm for arm in arms if control_unobserved or arm not in observed],
        ):
            if cell.method_role != "decision" or (metric.name, cell.group_id) in members:
                continue
            members.add((metric.name, cell.group_id))
            key = ArmHypothesisKey(metric.name, cell.group_id, cell.estimand)
            failures[key] = DecisionFailure(
                key,
                "readout.cell.missing_metric_observations",
                {
                    "metric": metric.name,
                    "group_id": cell.group_id,
                    "unobserved_arm": control if control_unobserved else cell.group_id,
                },
            )
    return failures


def _estimate_observational_secondary_family(  # noqa: PLR0913
    src: _BuiltinMomentSource,
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
    unit_frame_memo: Any,
    arms: Sequence[str],
    computations: list[DecisionComputation[LiftEstimate]] | None = None,
) -> list[LiftEstimate]:
    """Nominal pass at plan.alpha, BH-select at plan.q, FCR-reestimate the
    selected cells -- the observational sibling of
    _estimate_randomized_secondary_family, built on estimate_ate's own
    DecisionComputation[LiftEstimate] return value. Membership follows the declared plan,
    over the roster's *arms*.
    """
    from increment.decision import ArmHypothesisKey, DecisionComputation

    family = [m for m in secondary if getattr(plan.procedures[m.name].family, "member", False)]
    non_family = [
        m for m in secondary if not getattr(plan.procedures[m.name].family, "member", False)
    ]
    roles_by_metric = {
        metric.name: _runtime_method_roles(methods_by_metric[metric.name]) for metric in secondary
    }

    out: list[LiftEstimate] = []

    def _nominal_pass(
        metric: Metric, alpha: float, *, family_member: bool = False
    ) -> DecisionComputation[LiftEstimate]:
        config = config_by_name[metric.name]
        computation = _estimate_ate_from_evidence(
            design,
            metric,
            methods=methods_by_metric[metric.name],
            roles=roles_by_metric[metric.name],
            prior=config.prior,
            alpha=alpha,
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            evidence=evidence,
            unit_frame_memo=unit_frame_memo,
            family_member=family_member,
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
    for metric in family:
        # A counted control arm without this metric's values leaves nothing to estimate: each
        # cell is an unobserved member below.
        computation = (
            _merge_decision_computations([])
            if str(design.control_group) in arms
            and str(design.control_group) not in evidence.arms(metric)
            else _nominal_pass(metric, plan.alpha, family_member=True)
        )
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
    unobserved = _unobserved_member_failures(
        family,
        config_by_name,
        design,
        plan,
        value_scale=value_scale,
        evidence=evidence,
        arms=arms,
        estimated={(key.metric, key.group_id) for key, _ in family_cells},
    )
    family_cells.extend((key, None) for key in unobserved)
    family_computation = _merge_decision_computations(
        [
            *(computation_by_metric[metric.name] for metric in family),
            DecisionComputation[LiftEstimate](
                results=(),
                evidence={},
                failures=dict(unobserved.items()),
            ),
        ]
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
        "family_size": outcome.n_family,
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
                            "family_size": family_record["family_size"] if is_decision else None,
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
        computation = _estimate_ate_from_evidence(
            design,
            metric,
            methods=methods_by_metric[metric.name][:1],
            roles=roles_by_metric[metric.name],
            prior=config.prior,
            alpha=selected_alpha,
            value_scale=value_scale,
            null_abs=resolved_null_abs or None,
            alternatives=resolved_alternatives or None,
            evidence=evidence,
            unit_frame_memo=unit_frame_memo,
            route_tail=fcr_route_tail(procedure.alternative, fcr_alpha),
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
            # Only a selected row reissued at the FCR level carries its nominal level.
            match = (
                next(
                    (
                        cell
                        for cell in reestimated.get(row.metric, [])
                        if cell.group_id == row.group_id and cell.method == row.method
                    ),
                    None,
                )
                if key in selected_cells
                else None
            )
            stamped = row if match is None else carry_nominal_evidence(row, match)
            out.append(
                stamped.model_copy(
                    update={
                        "role": "secondary",
                        "discovery": family_discovery(outcome, key),
                        **family_record,
                        "family_nominal_alpha": None if match is None else plan.alpha,
                    }
                )
            )
    return out
