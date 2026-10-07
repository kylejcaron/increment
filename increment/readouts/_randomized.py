from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from increment._policy_alpha import resolve_cell_alpha
from increment.estimation.conversion_route import family_route_alpha
from increment.estimation.engine import Method
from increment.estimation.engine import merge_decision_computations as _merge_decision_computations
from increment.estimation.family import decision_cells, family_discovery, select_family
from increment.estimation.results import (
    LiftEstimate,
    _fcr_alpha_for,
    open_bound_from_two_sided_at_target,
)
from increment.estimation.sequential import AlwaysValid, AsymptoticMean, MixedFamily
from increment.readouts._common import (
    _joint_relative_rows,
    _refuse_unsupported_quantile,
    _runtime_methods,
    _sequential_inference,
)
from increment.readouts._metric_rows import _load_metric_rows
from increment.readouts._passes import _estimate_pass
from increment.sources import MomentSource

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from increment._analysis_config import ResolvedMetricConfig
    from increment.decision import CompiledDecisionPlan, DecisionComputation
    from increment.semantics.design import Randomized
    from increment.semantics.models import Metric


def _stamp_fcr_selected_rows(
    family: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    nominal_by_metric: Mapping[str, list[LiftEstimate]],
    reestimated: Mapping[str, list[LiftEstimate]],
    selected_cells: Collection[object],
    outcome: Any,
    family_record: Mapping[str, Any],
) -> list[LiftEstimate]:
    """Stamp family rows after selected cells receive FCR intervals."""
    from increment.decision import ArmHypothesisKey

    out: list[LiftEstimate] = []
    for metric, *_rest in family:
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
            key = ArmHypothesisKey(row.metric, row.group_id, "itt")
            if key in selected_cells:
                match = next(
                    (
                        cell
                        for cell in reestimated.get(row.metric, [])
                        if cell.group_id == row.group_id and cell.method == row.method
                    ),
                    None,
                )
                if match is None:
                    continue
                out.append(
                    match.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": family_discovery(outcome, key),
                            **family_record,
                        }
                    )
                )
            else:
                out.append(
                    row.model_copy(
                        update={
                            "role": "secondary",
                            "discovery": family_discovery(outcome, key),
                            **family_record,
                        }
                    )
                )
    return out


def _partition_secondary_family(
    entries: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
) -> tuple[
    list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
]:
    family = [
        entry
        for entry in entries
        if getattr(entry[2].family, "member", False) and entry[1].prior is None
    ]
    non_family = [
        entry
        for entry in entries
        if not getattr(entry[2].family, "member", False) or entry[1].prior is not None
    ]
    return family, non_family


def _load_family_rows(
    src: MomentSource,
    secondary_entries: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    design: Randomized,
    plan: CompiledDecisionPlan,
    *,
    by: Sequence[str],
) -> tuple[dict[str, Any], float | None]:
    """Each secondary's rows, and the smallest level the family can decide a p-value at.

    That level is the plan's ``q`` over every hypothesis the family tests (the metric x treatment
    arm cells of its members: a prior-bound or non-member secondary is not one); a conversion
    decision row is routed at it, so its p-value is never read at a tail it is not dense for.
    A quantile metric is refused before any data is read."""
    loaded: dict[str, Any] = {}
    for metric, _config, test, is_quantile, _metric_methods in secondary_entries:
        if is_quantile:
            _refuse_unsupported_quantile(metric, test, cluster=src.context.cluster, by=by)
        loaded[metric.name] = _load_metric_rows(
            src, metric, by=by, control_group=design.control_group
        )
    hypotheses = sum(
        loaded[metric.name].n_treatment_arms
        for metric, config, test, *_ in secondary_entries
        if getattr(test.family, "member", False) and config.prior is None
    )
    return loaded, family_route_alpha(plan.q, hypotheses)


def _secondary_family_cells(family, nominal_by_metric, family_groups_by_metric):
    from increment.decision import ArmHypothesisKey

    cells = []
    for metric, *_rest in family:
        rows_for_metric = nominal_by_metric[metric.name]
        for group_id in sorted(family_groups_by_metric.get(metric.name, set())):
            key = ArmHypothesisKey(metric.name, group_id, "itt")
            row = next(
                (
                    candidate
                    for candidate in rows_for_metric
                    if candidate.group_id == group_id
                    and candidate.estimand == "itt"
                    and candidate.method_role == "decision"
                ),
                None,
            )
            cells.append((key, row))
    return cells


def _secondary_rows_without_fcr(family, nominal_by_metric, outcome, family_record):
    from increment.decision import ArmHypothesisKey

    return [
        row.model_copy(
            update={
                "role": "secondary",
                "discovery": (
                    family_discovery(outcome, ArmHypothesisKey(row.metric, row.group_id, "itt"))
                    if row.method_role == "decision"
                    else None
                ),
                "family_axes": family_record["family_axes"]
                if row.method_role == "decision"
                else None,
                "family_q": family_record["family_q"] if row.method_role == "decision" else None,
                "family_threshold": (
                    family_record["family_threshold"] if row.method_role == "decision" else None
                ),
            }
        )
        for metric, *_rest in family
        for row in nominal_by_metric[metric.name]
    ]


def _secondary_source_rows(metric_rows, is_quantile, control_group):
    observed_arms = set(metric_rows.observed_arms)
    evidence_component = (
        metric_rows.evidence_kind,
        metric_rows.evidence_sha256,
        metric_rows.evidence_count,
    )
    if is_quantile:
        rows = metric_rows.unit_frame
        family_groups = None
    else:
        rows = list(metric_rows.rows)
        family_groups = {
            str(row["group_id"]) for row in rows if str(row["group_id"]) != str(control_group)
        }
    return rows, observed_arms, evidence_component, family_groups


def _secondary_group_ids(nominal, computation, observed_groups, is_quantile):
    from increment.decision import ArmHypothesisKey

    if is_quantile:
        groups = {row.group_id for row in nominal if row.estimand == "itt"}
        groups.update(
            str(key.group_id) for key in computation.failures if isinstance(key, ArmHypothesisKey)
        )
    else:
        groups = set(observed_groups)
        groups.update(
            str(cast(Any, key).group_id)
            for key in computation.failures
            if getattr(key, "group_id", None) is not None
        )
    return groups


def _estimate_randomized_secondary_family(
    src: MomentSource,
    secondary_entries: Sequence[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    design: Randomized,
    plan: CompiledDecisionPlan,
    *,
    by: Sequence[str],
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    advisory_seen: set[tuple[str, str]],
    observed_arms: set[str],
    computations: list[Any],
    observed_by_metric: dict[str, set[str]],
    evidence_components: dict[str, tuple[str, str, int]],
) -> tuple[list[LiftEstimate], list[tuple[str, str, str, str]]]:
    """Run nominal, selection, and FCR passes for randomized secondaries."""
    if not secondary_entries:
        return [], []
    nominal_by_metric: dict[str, list[LiftEstimate]] = {}
    nominal_computation_by_metric: dict[str, DecisionComputation[LiftEstimate]] = {}
    family_groups_by_metric: dict[str, set[str]] = {}
    # Evidence read for the nominal pass (moment rows or a quantile unit
    # frame), reused by the FCR pass instead of re-querying the source.
    evidence_by_metric: dict[str, Any] = {}
    refused_cells: list[tuple[str, str, str, str]] = []
    loaded, route_alpha = _load_family_rows(src, secondary_entries, design, plan, by=by)
    for metric, config, test, is_quantile, _metric_methods in secondary_entries:
        member_route_alpha = (
            route_alpha if getattr(test.family, "member", False) and config.prior is None else None
        )
        metric_rows = loaded[metric.name]
        rows, metric_observed, evidence_component, observed_groups = _secondary_source_rows(
            metric_rows, is_quantile, design.control_group
        )
        observed_by_metric[metric.name] = metric_observed
        evidence_components[metric.name] = evidence_component
        evidence_by_metric[metric.name] = rows
        observed_arms |= metric_observed
        if is_quantile and not metric_rows.n_treatment_arms:
            # Nothing to contrast; the final arm gate decides whether the
            # whole readout refuses.
            nominal, refused, computation = [], [], _merge_decision_computations([])
        else:
            nominal, refused, computation = _estimate_pass(
                src,
                metric,
                rows,
                test,
                config,
                design,
                alpha_for=plan.alpha,
                role="secondary",
                inference=effective_inference,
                advisory_seen=advisory_seen,
                retry=(
                    "family"
                    if is_quantile
                    or (getattr(test.family, "member", False) and config.prior is None)
                    else "cells"
                ),
                methods=_metric_methods,
                route_alpha=None if is_quantile else member_route_alpha,
            )
        nominal_by_metric[metric.name] = nominal
        nominal_computation_by_metric[metric.name] = computation
        family_groups_by_metric[metric.name] = _secondary_group_ids(
            nominal, computation, observed_groups, is_quantile
        )
        refused_cells.extend(refused)

    computations.extend(nominal_computation_by_metric.values())
    family, non_family = _partition_secondary_family(secondary_entries)
    out: list[LiftEstimate] = []
    for metric, *_rest in non_family:
        out.extend(
            row.model_copy(update={"role": "secondary", "discovery": None})
            for row in nominal_by_metric[metric.name]
        )
    if not family:
        return out, refused_cells

    family_cells = _secondary_family_cells(family, nominal_by_metric, family_groups_by_metric)
    family_computation = _merge_decision_computations(
        [nominal_computation_by_metric[metric.name] for metric, *_rest in family]
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
        out.extend(_secondary_rows_without_fcr(family, nominal_by_metric, outcome, family_record))
        return out, refused_cells

    selected_names = {key.metric for key in selected_cells}
    reestimated: dict[str, list[LiftEstimate]] = {}
    for metric, config, test, is_quantile, _metric_methods in family:
        if metric.name not in selected_names:
            continue
        selected_alpha = _fcr_alpha_for(
            "two-sided"
            if _joint_relative_rows(nominal_by_metric[metric.name])
            else test.alternative,
            fcr_alpha,
        )
        reestimated[metric.name], refused, computation = _estimate_pass(
            src,
            metric,
            evidence_by_metric[metric.name],
            test,
            config,
            design,
            alpha_for=selected_alpha,
            role="secondary",
            inference=None if is_quantile else effective_inference,
            advisory_seen=advisory_seen,
            methods=_metric_methods[:1],
        )
        reestimated[metric.name] = [
            open_bound_from_two_sided_at_target(r) for r in reestimated[metric.name]
        ]
        computations.append(computation)
        refused_cells.extend(refused)
    out.extend(
        _stamp_fcr_selected_rows(
            family,
            nominal_by_metric,
            reestimated,
            selected_cells,
            outcome,
            family_record,
        )
    )
    return out, refused_cells


def _estimate_randomized_non_secondary(
    src: MomentSource,
    selected: Sequence[Metric],
    configs: Sequence[ResolvedMetricConfig],
    design: Randomized,
    plan: CompiledDecisionPlan,
    *,
    by: Sequence[str],
    effective_inference: AsymptoticMean | AlwaysValid | MixedFamily | None,
    advisory_seen: set[tuple[str, str]],
    observed_arms: set[str],
    computations: list[Any],
    observed_by_metric: dict[str, set[str]],
    evidence_components: dict[str, tuple[str, str, int]],
) -> tuple[
    list[LiftEstimate],
    list[tuple[str, str, str, str]],
    list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]],
    list[Any],
]:
    """Estimate randomized primary, guardrail, and unassigned cells.

    Every arm read along the way is added to *observed_arms* so the caller
    can refuse a treatment-free readout without a second load.
    """
    out: list[LiftEstimate] = []
    refused_cells: list[tuple[str, str, str, str]] = []
    secondary_entries: list[tuple[Metric, ResolvedMetricConfig, Any, bool, list[Method]]] = []
    for metric, config in zip(selected, configs, strict=True):
        test = plan.procedures[metric.name]
        computation = None
        is_quantile = getattr(metric, "type", None) == "quantile"
        metric_methods = _runtime_methods(config, design)
        if test.role == "secondary":
            secondary_entries.append((metric, config, test, is_quantile, metric_methods))
            continue
        if is_quantile:
            _refuse_unsupported_quantile(metric, test, cluster=src.context.cluster, by=by)
            metric_rows = _load_metric_rows(src, metric, by=by, control_group=design.control_group)
            evidence_components[metric.name] = (
                metric_rows.evidence_kind,
                metric_rows.evidence_sha256,
                metric_rows.evidence_count,
            )
            observed_arms |= metric_rows.observed_arms
            observed_by_metric[metric.name] = set(metric_rows.observed_arms)
            if metric_rows.n_treatment_arms == 0:
                continue
            if metric_methods == []:
                rows_est, refused = [], []
            else:
                cell_alpha = (
                    resolve_cell_alpha(plan, test, n_arms=metric_rows.n_treatment_arms, view=None)
                    if test.role == "primary"
                    else test.alpha
                )
                rows_est, refused, computation = _estimate_pass(
                    src,
                    metric,
                    metric_rows.unit_frame,
                    test,
                    config,
                    design,
                    alpha_for=cell_alpha,
                    role=test.role if plan.declared else None,
                    inference=effective_inference,
                    advisory_seen=advisory_seen,
                    methods=metric_methods,
                )
        else:
            metric_rows = _load_metric_rows(src, metric, by=by, control_group=design.control_group)
            evidence_components[metric.name] = (
                metric_rows.evidence_kind,
                metric_rows.evidence_sha256,
                metric_rows.evidence_count,
            )
            observed_arms |= metric_rows.observed_arms
            observed_by_metric[metric.name] = set(metric_rows.observed_arms)
            if test.role == "primary" and metric_rows.n_treatment_arms == 0:
                continue
            cell_alpha = resolve_cell_alpha(
                plan, test, n_arms=metric_rows.n_treatment_arms, view=None
            )
            rows_est, refused, computation = _estimate_pass(
                src,
                metric,
                list(metric_rows.rows),
                test,
                config,
                design,
                alpha_for=cell_alpha,
                role=test.role if plan.declared else None,
                inference=effective_inference,
                advisory_seen=advisory_seen,
                methods=metric_methods,
            )
        refused_cells.extend(refused)
        if computation is not None:
            computations.append(computation)
        out.extend(rows_est)
    return out, refused_cells, secondary_entries, computations
