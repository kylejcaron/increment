from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from increment._policy_alpha import resolve_cell_alpha
from increment.breakout.estimates import reject_quantile_metrics
from increment.errors import InvalidRequestError
from increment.estimation.conversion_route import family_route_alpha
from increment.estimation.encouragement import (
    ESTIMANDS,
    _estimate_encouragement,
    estimate_compliance,
    estimate_encouragement,
)
from increment.estimation.engine import Method
from increment.estimation.engine import merge_decision_computations as _merge_decision_computations
from increment.estimation.family import decision_cells, family_discovery, select_family
from increment.estimation.results import (
    LiftEstimate,
    _fcr_alpha_for,
    open_bound_from_two_sided_at_target,
)
from increment.readouts._common import (
    _joint_relative_rows,
    _refuse_if_no_treatment_arm,
    _runtime_method_roles,
    _runtime_methods,
    _sequential_inference,
)
from increment.semantics.design import Encouragement
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment._source_types import ComplianceSummary
    from increment.decision import CompiledDecisionPlan, DecisionComputation
    from increment.estimation.inference import Prior
    from increment.semantics.models import Metric


def _encouragement_family_config(
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    plan: CompiledDecisionPlan,
) -> tuple[dict[str, ResolvedMetricConfig], set[str]]:
    config_by_metric = {
        metric.name: config for metric, config in zip(metrics, configs, strict=True)
    }
    family_metric_names = {
        metric.name
        for metric in metrics
        if (
            plan.procedures[metric.name].role == "secondary"
            and getattr(plan.procedures[metric.name].family, "member", False)
        )
    }
    return config_by_metric, family_metric_names


class _EncouragementCellGroup(NamedTuple):
    """Metrics ``estimate_encouragement`` can batch into one call: it takes the policy as call-wide
    scalars, so every metric of a group shares them."""

    methods: list[Method]
    prior: Prior | None
    alternative: str
    alpha: float
    null_lift: float
    null_abs: float | None
    route_alpha: float | None
    metrics: list[Metric]


def _encouragement_cell_groups(
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Encouragement,
    alpha_by_metric: Mapping[str, float],
    alternative_by_metric: Mapping[str, str],
    null_lift_by_metric: Mapping[str, float],
    null_abs_by_metric: Mapping[str, float | None],
    route_alpha_by_metric: Mapping[str, float | None],
) -> list[_EncouragementCellGroup]:
    groups: list[_EncouragementCellGroup] = []
    for metric, config in zip(metrics, configs, strict=True):
        alternative = alternative_by_metric[metric.name]
        alpha = alpha_by_metric[metric.name]
        null_lift = null_lift_by_metric[metric.name]
        null_abs = null_abs_by_metric[metric.name]
        route_alpha = route_alpha_by_metric[metric.name]
        metric_methods = _runtime_methods(config, design)
        for group in groups:
            if (
                group.methods == metric_methods
                and group.prior == config.prior
                and group.alternative == alternative
                and group.alpha == alpha
                and group.null_lift == null_lift
                and group.null_abs == null_abs
                and group.route_alpha == route_alpha
            ):
                group.metrics.append(metric)
                break
        else:
            groups.append(
                _EncouragementCellGroup(
                    metric_methods,
                    config.prior,
                    alternative,
                    alpha,
                    null_lift,
                    null_abs,
                    route_alpha,
                    [metric],
                )
            )
    return groups


def _family_route_alphas(
    metrics: Sequence[Metric],
    family_metric_names: set[str],
    rows_by_metric: Mapping[str, list[Mapping[str, Any]]],
    q: float,
    control: str,
) -> dict[str, float | None]:
    """The level each metric's p-values are routed at: a secondary family reads each ITT p-value
    at a threshold as small as ``q / m`` (``m`` its metric x arm hypotheses), so its members are
    routed at that level and never a looser one; every other metric at its own."""
    hypotheses = sum(
        len({str(row["group_id"]) for row in rows_by_metric[name]} - {control})
        for name in family_metric_names
    )
    return {
        metric.name: family_route_alpha(q, hypotheses)
        if metric.name in family_metric_names
        else None
        for metric in metrics
    }


def _compliance_only_rows(src, design, wanted, plan, capture):
    if set(wanted) != {"compliance"}:
        return None
    summary = _validated_compliance_summary(src, design)
    computation = _design_compliance_computation(
        summary, design, wanted, alpha=plan.alpha, deferred=False
    )
    if capture is not None:
        capture["computations"] = [computation]
        capture["compliance_summary"] = summary
    return list(computation.results)


def _prepare_encouragement_groups(metrics, rows_by_metric, configs, design, plan):
    role_by_metric: dict[str, str | None] = {}
    cell_alpha_by_metric: dict[str, float] = {}
    cell_alternative_by_metric: dict[str, str] = {}
    cell_null_lift_by_metric: dict[str, float] = {}
    cell_null_abs_by_metric: dict[str, float | None] = {}
    control = str(design.control_group)
    for metric in metrics:
        test = plan.procedures[metric.name]
        cell_null_lift_by_metric[metric.name] = getattr(test, "null_lift", 0.0) or 0.0
        cell_null_abs_by_metric[metric.name] = getattr(test, "null_abs", None)
        if not plan.declared:
            role_by_metric[metric.name] = None
            cell_alpha_by_metric[metric.name] = test.alpha
            cell_alternative_by_metric[metric.name] = test.alternative
            continue
        role_by_metric[metric.name] = test.role
        arm_ids = (
            {str(row["group_id"]) for row in rows_by_metric[metric.name]}
            if test.role == "primary"
            else set()
        )
        # Secondary metrics use nominal alpha here; family selection later re-estimates selected cells at its FCR alpha.
        cell_alpha_by_metric[metric.name] = resolve_cell_alpha(
            plan, test, n_arms=len(arm_ids - {control}), view=None
        )
        cell_alternative_by_metric[metric.name] = test.alternative
    _, family_metric_names = _encouragement_family_config(metrics, configs, plan)
    groups = _encouragement_cell_groups(
        metrics,
        configs,
        design,
        cell_alpha_by_metric,
        cell_alternative_by_metric,
        cell_null_lift_by_metric,
        cell_null_abs_by_metric,
        _family_route_alphas(metrics, family_metric_names, rows_by_metric, plan.q, control),
    )
    return role_by_metric, family_metric_names, groups


def encouragement_rows(
    *,
    src: MomentSource,
    metrics: Sequence[Metric],
    rows_by_metric: Mapping[str, list[Mapping[str, Any]]],
    configs: Sequence[ResolvedMetricConfig],
    design: Encouragement,
    plan: CompiledDecisionPlan,
    estimands: Sequence[str] | None,
    cluster: str | None,
    caller: str,
    capture: dict[str, Any] | None = None,
) -> list[LiftEstimate]:
    """Every encouragement row for one whole-window readout.

    The single implementation behind both the dataframe and warehouse
    paths: a primary's alpha splits across its own treatment arms, a
    secondary estimates at the plan's nominal alpha and then faces
    BH/e-BH family selection, and the design-level ``uptake`` row is
    estimated exactly once at the plan's own alpha rather than inheriting
    whichever metric ran first.

    ``estimate_encouragement`` takes alpha/alternative/shifted null as
    call-wide scalars, so metrics sharing (methods, prior, alternative,
    alpha, null) are batched into one call and the results sorted back into
    declared order. A declared margin applies to the ITT row only; LATE and
    compliance carry no shifted null.
    """
    wanted = estimands if estimands is not None else ESTIMANDS
    compliance_rows = _compliance_only_rows(src, design, wanted, plan, capture)
    if compliance_rows is not None:
        return compliance_rows
    if not metrics:
        return []
    reject_quantile_metrics(
        metrics,
        caller,
        reason="estimate_encouragement consumes mean-based group_summary moments, "
        "which a quantile metric has none of",
        remedy="Drop the encouragement design or the quantile metric.",
    )
    role_by_metric, family_metric_names, groups = _prepare_encouragement_groups(
        metrics, rows_by_metric, configs, design, plan
    )

    compliance_summary = (
        _validated_compliance_summary(src, design) if "compliance" in wanted else None
    )
    results: list[LiftEstimate] = []
    computations: list[DecisionComputation[LiftEstimate]] = []
    family_refusal_deferred = False
    for group in groups:
        summary = [row for m in group.metrics for row in rows_by_metric[m.name]]
        method_roles = _runtime_method_roles(group.methods)
        try:
            computation = _estimate_encouragement(
                metrics=group.metrics,
                summary=summary,
                design=design,
                estimands=tuple(name for name in wanted if name != "compliance"),
                methods=group.methods,
                prior=group.prior,
                alpha=group.alpha,
                alternative=group.alternative,
                null_lift=group.null_lift or None,
                null_abs=group.null_abs,
                cluster=cluster,
                method_roles=method_roles,
                route_alpha=group.route_alpha,
            )
        except InvalidRequestError as exc:
            if exc.code != "estimation.encouragement.control.missing" or any(
                metric.name not in family_metric_names for metric in group.metrics
            ):
                raise
            family_refusal_deferred = True
            from increment.decision import ArmHypothesisKey, DecisionComputation, DecisionFailure

            failures: dict[Any, Any] = {}
            for metric in group.metrics:
                for row in rows_by_metric[metric.name]:
                    group_id = str(row["group_id"])
                    if group_id == str(design.control_group):
                        continue
                    hypothesis = ArmHypothesisKey(metric.name, group_id, "itt")
                    failures[hypothesis] = DecisionFailure(
                        hypothesis,
                        "estimation.encouragement.control.missing",
                        {
                            "metric": metric.name,
                            "group_id": group_id,
                            "reason": "no_control_arm",
                        },
                    )
            computation = DecisionComputation[LiftEstimate](
                results=(), evidence={}, failures=failures
            )
        computations.append(computation)
        results.extend(
            r.model_copy(
                update={
                    "role": role_by_metric.get(r.metric),
                    "method_role": method_roles.get(r.method, "decision"),
                }
            )
            for r in computation.results
        )

    # Compliance is design-level: estimate it once at the plan alpha from the
    # declared uptake cohort, never outcome-filtered moments, so it cannot depend
    # on metric order, metric count, or outcome missingness policy.

    computation = _design_compliance_computation(
        compliance_summary,
        design,
        wanted,
        alpha=plan.alpha,
        deferred=family_refusal_deferred,
    )
    computations.append(computation)
    results.extend(r.model_copy(update={"role": None}) for r in computation.results)
    if capture is not None:
        capture["computations"] = computations
        capture["compliance_summary"] = compliance_summary
    if plan.declared:
        results = _select_encouragement_family(
            results,
            computation=_merge_decision_computations(computations),
            metrics=metrics,
            configs=configs,
            plan=plan,
            design=design,
            rows_by_metric=rows_by_metric,
            estimands=estimands,
            cluster=cluster,
            computations=computations,
        )
    # Grouping dispatches metrics out of declared order, so stable-sort
    # back to it; compliance is design-level (identified by estimand, not
    # its "uptake" metric string, which a real outcome metric may share),
    # so it sorts first.
    order = {m.name: i for i, m in enumerate(metrics)}
    results.sort(key=lambda r: -1 if r.estimand == "compliance" else order[r.metric])
    return results


def _validated_compliance_summary(src: MomentSource, design: Encouragement) -> ComplianceSummary:
    from increment._source_types import validate_compliance_source_match

    summary = src.compliance_summary(design)
    validate_compliance_source_match(summary, src.context)
    _refuse_if_no_treatment_arm(
        {str(arm.group_id) for arm in summary.arms},
        design.control_group,
    )
    return summary


def _design_compliance_computation(
    summary: ComplianceSummary | None,
    design: Encouragement,
    wanted: Sequence[str],
    *,
    alpha: float,
    deferred: bool,
) -> DecisionComputation[LiftEstimate]:
    from increment.decision import DecisionComputation

    compliance_wanted = "compliance" in wanted
    if deferred or not compliance_wanted:
        return DecisionComputation(results=(), evidence={}, failures={})
    assert summary is not None
    return estimate_compliance(
        summary,
        design,
        alpha=alpha,
        compliance_requested=compliance_wanted,
    )


def _select_encouragement_family(
    rows: list[LiftEstimate],
    *,
    computation: DecisionComputation[LiftEstimate],
    metrics: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    plan: CompiledDecisionPlan,
    design: Encouragement,
    rows_by_metric: Mapping[str, list[Mapping[str, Any]]],
    estimands: Sequence[str] | None,
    cluster: str | None,
    computations: list[DecisionComputation[LiftEstimate]],
) -> list[LiftEstimate]:
    """BH/e-BH selection over in-family encouragement secondaries.

    A metric here emits several estimand rows (itt/compliance/late), but
    the family is one cell per ``(metric, arm)`` and the ITT row is that
    cell's representative: itt and late are two estimands of the SAME
    null, so admitting both would double ``m`` for one hypothesis, and
    `select_family` already collapses same-null rows for exactly that
    reason on the method axis. compliance is excluded outright - it is
    the first-stage uptake diagnostic, not a hypothesis about the
    declared metric.

    ITT specifically, not late: a weak first stage suppresses the late
    row entirely, so keying on it would drop a metric out of the family
    precisely when its instrument is weak, shrinking ``m`` and loosening
    the bar for every surviving metric. ITT is also the only estimand
    with an unconditional randomization guarantee - late needs the
    exclusion restriction and a strong first stage, which a family-wide
    error rate should not silently inherit.

    ``discovery`` lands on the ITT rows only. A compliance or late row
    was never the tested hypothesis, and marking it a discovery would
    claim its own interval met the family's bar.
    """
    config_by_name = {m.name: c for m, c in zip(metrics, configs, strict=True)}
    in_family = {
        m.name
        for m in metrics
        if (
            plan.procedures[m.name].role == "secondary"
            and getattr(plan.procedures[m.name].family, "member", False)
        )
    }
    if not in_family:
        return rows
    if estimands is not None and "itt" not in estimands:
        return rows
    from increment.decision import ArmHypothesisKey

    # The compiled family is every outcome metric/arm cell, not merely the
    # rows that survived estimation.  A keyed failure therefore aborts the
    # complete family instead of shrinking its denominator.
    cells: list[tuple[ArmHypothesisKey, object]] = []
    for metric_name in sorted(in_family):
        groups = {
            str(row["group_id"])
            for row in rows_by_metric[metric_name]
            if str(row["group_id"]) != str(design.control_group)
        }
        for group_id in sorted(groups):
            hypothesis = ArmHypothesisKey(metric_name, group_id, "itt")
            row = next(
                (
                    r
                    for r in rows
                    if r.metric == metric_name
                    and r.group_id == group_id
                    and r.estimand == "itt"
                    and r.method_role == "decision"
                ),
                None,
            )
            cells.append((hypothesis, row))
    if not cells:
        # Narrowed estimands can omit ITT entirely; no family was compiled.
        return rows
    outcome = select_family(
        decision_cells(cells),
        plan.q,
        _sequential_inference(plan),
        plan.alpha,
        computation=computation,
    )
    family_record = {
        "family_axes": ("metric", "arm"),
        "family_q": outcome.q,
        "family_threshold": outcome.realized_threshold,
    }
    reestimated: dict[tuple[str, str, str, str, str], LiftEstimate] = {}
    if outcome.fcr_alpha is not None:
        selected_names = {key.metric for key in outcome.selected}
        config_by_name = {m.name: c for m, c in zip(metrics, configs, strict=True)}
        for name in sorted(selected_names):
            metric = next(m for m in metrics if m.name == name)
            config = config_by_name[name]
            alternative = plan.procedures[name].alternative
            selected_alpha = _fcr_alpha_for(
                "two-sided" if _joint_relative_rows(rows, metric=name) else alternative,
                outcome.fcr_alpha,
            )
            pass_computation = estimate_encouragement(
                metrics=[metric],
                summary=rows_by_metric[name],
                design=design,
                estimands=estimands if estimands is not None else ESTIMANDS,
                methods=[config.decision_method],
                prior=config.prior,
                alpha=selected_alpha,
                alternative=alternative,
                null_lift=getattr(plan.procedures[name], "null_lift", 0.0) or None,
                null_abs=getattr(plan.procedures[name], "null_abs", None),
                cluster=cluster,
                method_roles={config.decision_method.name: "decision"},
            )
            computations.append(pass_computation)
            pass_results = pass_computation.results
            for r in pass_results:
                if r.estimand in ("itt", "late"):
                    reestimated[(r.metric, r.group_id, r.method, r.estimand, r.value_scale)] = (
                        open_bound_from_two_sided_at_target(r)
                    )
    out: list[LiftEstimate] = []
    for r in rows:
        if r.metric not in in_family or r.estimand not in ("itt", "late"):
            out.append(r)
            continue
        key = ArmHypothesisKey(r.metric, r.group_id, "itt")
        selected = key in outcome.selected
        match = (
            reestimated.get((r.metric, r.group_id, r.method, r.estimand, r.value_scale))
            if selected
            else None
        )
        if r.estimand != "itt":
            # LATE presents the same tested null (see the docstring). Selected
            # decision-role cells carry the corrected interval; other late rows keep
            # theirs. Every late row gets the family role, but discovery and family
            # metadata apply only to the tested ITT hypothesis.
            base = match if match is not None else r
            out.append(base.model_copy(update={"role": "secondary"}))
            continue
        if r.method_role != "decision":
            out.append(
                r.model_copy(
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
        base = match if match is not None else r
        out.append(
            base.model_copy(
                update={
                    "role": "secondary",
                    "discovery": family_discovery(outcome, key),
                    **family_record,
                }
            )
        )
    return out
