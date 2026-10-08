"""Scope records for observational and encouragement readouts.

Expected metric x arm x method/estimand cells are enumerated from the request and the
source-owned roster before result projection; typed producer failures remain attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from increment._canonical import canonical_json_bytes
from increment._labels import UNASSIGNED_LABEL
from increment.breakout.estimates import LiftEstimates
from increment.errors import CapabilityError
from increment.estimation.adjust import _weight_diagnostics_projection
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.readout_types import (
    CellFailure,
    CellKey,
    CellRecord,
    PopulationRoster,
    ReadoutMetadata,
    ReadoutScope,
    SamplingInference,
    SourceReadoutScope,
    cell_order,
)
from increment.estimation.results import LiftEstimate
from increment.readouts._common import _runtime_method_roles, _runtime_methods
from increment.readouts._metric_rows import _rows_digest
from increment.readouts._multiplicity_scope import attach_multiplicity_scope

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ExpectedCell:
    metric: str
    group_id: str
    method: str
    method_role: str
    estimand: str
    alternative: str
    value_scale: str
    estimand_alternates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _ObservedRoster:
    source: str
    arms_by_metric: Mapping[str, set[str]]
    known_arms: frozenset[str]


def _roster_evidence_arms(observed_by_metric, counts):
    arms = {
        arm
        for observed in observed_by_metric.values()
        for arm in observed
        if arm is not None
    }
    if counts is not None:
        arms.update(
            arm for arm in counts if arm is not None and arm != UNASSIGNED_LABEL
        )
    return frozenset(arms)


def _collect_expected_cells(
    expected: Sequence[ExpectedCell],
    row_map: dict[tuple, list[LiftEstimate]],
    failure_map: Mapping[tuple, Any],
    population: str,
    roster: _ObservedRoster,
) -> tuple[dict[CellKey, LiftEstimate | None], dict[CellKey, CellFailure]]:
    cell_rows: dict[CellKey, LiftEstimate | None] = {}
    failures: dict[CellKey, CellFailure] = {}
    for item in expected:
        identities = (item.estimand, *item.estimand_alternates)
        matched = []
        failure = None
        for estimand in identities:
            key = (item.metric, item.group_id, item.method, estimand)
            matched = row_map.pop(key, [])
            if matched:
                break
            failure = failure_map.get(key) or failure_map.get(
                (item.metric, item.group_id, None, estimand)
            )
            if failure is not None:
                break
        if matched:
            for row in matched:
                cell_rows.setdefault(CellKey.from_row(row), row)
            continue
        # An adjustment that explicitly does not support this metric is not
        # part of the estimator's cell inventory. Keep other typed failures:
        # those represent requested cells the producer attempted to estimate.
        if failure is not None and failure.code == "estimation.adjust.unavailable":
            continue
        cell = CellKey(
            kind="arm",
            metric=item.metric,
            group_id=item.group_id,
            method=item.method,
            method_role=item.method_role,
            estimand=item.estimand,
            analysis_population=population,
            value_scale=item.value_scale,
            inference="fixed",
            alternative=item.alternative,
        )
        if failure is not None:
            failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(), code=failure.code, context=failure.context
            )
        elif item.group_id not in roster.arms_by_metric.get(item.metric, set()):
            code = (
                "readout.cell.missing_metric_observations"
                if item.group_id in roster.known_arms
                else "readout.cell.missing_arm"
            )
            failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(),
                code=code,
                context={
                    "metric": item.metric,
                    "group_id": item.group_id,
                    "observed_arms": sorted(roster.arms_by_metric.get(item.metric, set())),
                    "roster_source": roster.source,
                },
            )
        else:
            failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(),
                code="readout.cell.unsupported_request",
                context={"reason": "no_estimate_returned", "analysis_population": population},
            )
        cell_rows[cell] = None
    for leftover in row_map.values():
        for row in leftover:
            cell_rows.setdefault(CellKey.from_row(row), row)
    return cell_rows, failures


def resolve_roster(src, design, population, observed_by_metric, *, use_counts=True):
    """Resolve arms, retaining counts only at the declared randomization grain."""
    is_clustered = getattr(src.context, "cluster", None) is not None
    counts = None
    integrity_counts = None
    if use_counts and is_clustered and population == "assigned":
        cluster_counts = getattr(src, "cluster_counts", None)
        if callable(cluster_counts):
            try:
                counts = {arm: count for arm, count in cluster_counts().items() if arm is not None}
                integrity_counts = counts
            except CapabilityError:
                pass
    if use_counts and counts is None:
        assignment_counts = getattr(src, "assignment_counts", None)
        if callable(assignment_counts):
            try:
                counts = {
                    arm: count
                    for arm, count in assignment_counts(population=population).items()
                    if arm is not None
                }
            except CapabilityError:
                pass
    if use_counts and counts is None:
        count_method = getattr(src, "unit_counts", None) if not is_clustered else None
        if callable(count_method):
            try:
                counts = {arm: count for arm, count in count_method().items() if arm is not None}
            except CapabilityError:
                pass
    if not is_clustered:
        integrity_counts = counts
    allocation = getattr(design, "allocation", None)
    if allocation is not None:
        return (
            tuple(sorted(arm for arm in allocation if arm is not None)),
            "declared_allocation",
            True,
            counts,
            integrity_counts,
        )
    if counts is not None:
        return (
            tuple(
                sorted(
                    arm for arm in counts if arm is not None and arm != UNASSIGNED_LABEL
                )
            ),
            "experiment_counts",
            True,
            counts,
            integrity_counts,
        )
    arms = (
        tuple(
            sorted(
                set().union(
                    *(
                        {arm for arm in observed if arm is not None}
                        for observed in observed_by_metric.values()
                    )
                )
            )
        )
        if observed_by_metric
        else ()
    )
    return arms, ("observed_union" if arms else "unknown"), False, counts, integrity_counts


def observational_expected(selected, configs, design, plan, *, value_scale, arms):
    control = str(design.control_group)
    trimmed_target_possible = getattr(getattr(design, "gate", None), "overlap", None) == "trim"
    cells = []
    for metric, config in zip(selected, configs, strict=True):
        methods = _runtime_methods(config, design)
        roles = _runtime_method_roles(methods)
        for group_id in arms:
            if group_id == control:
                continue
            for method in methods:
                cells.append(
                    ExpectedCell(
                        metric.name,
                        group_id,
                        method.name,
                        roles[method.name],
                        "plr_slope"
                        if method.name == "dml"
                        else "itt"
                        if method.name == "unadjusted"
                        else "ate",
                        plan.procedures[metric.name].alternative,
                        (value_scale or {}).get(metric.name, "relative"),
                        tuple(
                            name
                            for name, present in (
                                ("ate", method.name == "dml"),
                                (
                                    "overlap_subpopulation_ate",
                                    trimmed_target_possible
                                    and method.name not in {"dml", "unadjusted"},
                                ),
                            )
                            if present
                        ),
                    )
                )
    return cells


def encouragement_expected(
    selected, configs, design, plan, *, estimands, arms, eligible_late, cluster
):
    from increment.estimation.encouragement import ESTIMANDS

    wanted = tuple(estimands) if estimands is not None else ESTIMANDS
    control = str(design.control_group)
    cells = []
    for metric, config in zip(selected, configs, strict=True):
        methods = _runtime_methods(config, design)
        roles = _runtime_method_roles(methods)
        for group_id in arms:
            if group_id == control:
                continue
            for estimand in wanted:
                if estimand in {"compliance"}:
                    continue
                if estimand == "late" and (metric.name, group_id) not in eligible_late:
                    continue
                for method in methods:
                    cells.append(
                        ExpectedCell(
                            metric.name,
                            group_id,
                            method.name,
                            roles[method.name],
                            estimand,
                            plan.procedures[metric.name].alternative,
                            "absolute"
                            if estimand == "late" or (estimand == "compliance" and cluster)
                            else "relative",
                        )
                    )
    if "compliance" in wanted:
        for group_id in arms:
            if group_id != control:
                cells.append(
                    ExpectedCell(
                        "uptake",
                        group_id,
                        "unadjusted",
                        "decision",
                        "compliance",
                        "two-sided",
                        "absolute" if cluster else "relative",
                    )
                )
    return cells


def eligible_late_cells(rows_by_metric, design, cluster):
    """Use the producer's first-stage gate to enumerate only emitted LATE cells."""
    from increment.estimation.encouragement import _df_to_arms, _first_stage_context

    arms = _df_to_arms([row for metric_rows in rows_by_metric.values() for row in metric_rows])
    by_metric = {}
    for arm in arms:
        by_metric.setdefault(arm.metric, {})[str(arm.group_id)] = arm
    eligible = set()
    for metric, grouped in by_metric.items():
        control = grouped.get(str(design.control_group))
        if control is None:
            continue
        for group_id, treatment in grouped.items():
            if group_id == str(design.control_group):
                continue
            context = _first_stage_context(
                treatment, control, design, cluster=cluster, late_requested=True
            )
            if not context.weak:
                eligible.add((metric, group_id))
    return eligible


def compliance_component(summary, population):
    payload = {
        "study_id": summary.study_id,
        "cohort": summary.cohort,
        "window_days": summary.window_days,
        "one_sided": summary.one_sided,
        "cluster": summary.cluster,
        "as_of": None if summary.as_of is None else str(summary.as_of),
        "control_group": summary.control_group,
        "arms": [
            {name: getattr(arm, name) for name in arm.__dataclass_fields__}
            for arm in sorted(summary.arms, key=lambda arm: arm.group_id)
        ],
    }
    return {
        "kind": "compliance_summary",
        "metric": None,
        "population": population,
        "sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def scope_design_results(  # noqa: PLR0913
    src: Any,
    design: Any,
    plan: Any,
    rows: Sequence[LiftEstimate],
    computations: Sequence[Any],
    *,
    configs: Sequence[Any],
    population: str,
    expected_for: Any,
    evidence_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    observed_by_metric: Mapping[str, set[str]],
    compliance_summary: Any = None,
    estimands: Sequence[str] | None = None,
    use_counts: bool = True,
    request_extra: Mapping[str, Any] | None = None,
) -> LiftEstimates:
    """Attach source identity and every expected cell, preserving provider failures."""
    arms, roster_source, roster_complete, counts, integrity_counts = resolve_roster(
        src, design, population, observed_by_metric, use_counts=use_counts
    )
    expected = expected_for(arms)
    components = [
        {
            "kind": "moments_rows",
            "metric": metric,
            "population": population,
            "sha256": _rows_digest(list(evidence_rows[metric]))[0],
        }
        for metric in sorted(evidence_rows)
    ]
    if compliance_summary is not None:
        components.append(compliance_component(compliance_summary, population))
    if counts is not None:
        components.append(
            {
                "kind": "assignment_counts",
                "metric": None,
                "population": population,
                "sha256": sha256(canonical_json_bytes(counts)).hexdigest(),
            }
        )
    components.sort(key=lambda item: (item["kind"], item["metric"] or "", item["population"]))
    source = {
        "kind": "composite",
        "sha256": sha256(canonical_json_bytes(components)).hexdigest(),
        "components": components,
    }
    request = {
        "view": "run",
        "population": population,
        "mechanism": design.mechanism,
        "design": design.model_dump(mode="json"),
        "estimands": None if estimands is None else sorted(estimands),
        "alpha": plan.alpha,
        "q": plan.q,
        "declared": plan.declared,
        "procedures": {
            name: value.model_dump(mode="json") for name, value in plan.procedures.items()
        },
        **dict(request_extra or {}),
    }
    trigger_name = src.context.trigger_name
    if trigger_name is not None:
        request["trigger_declared"] = True
        request["trigger_name"] = trigger_name
    snapshot_id = (
        "sha256:"
        + sha256(
            canonical_json_bytes(
                {
                    "kind": "increment.readout.snapshot",
                    "version": 1,
                    "collection": "LiftEstimates",
                    "source": source,
                    "request": request,
                }
            )
        ).hexdigest()
    )

    def cell_id(metric, group, method, estimand):
        return metric, group, method, estimand

    row_map: dict[tuple, list[LiftEstimate]] = {}
    for row in rows:
        row_map.setdefault(cell_id(row.metric, row.group_id, row.method, row.estimand), []).append(
            row
        )
    failure_map: dict[tuple, Any] = {}
    for computation in computations:
        for hypothesis, failure in computation.failures.items():
            failure_map[
                cell_id(
                    hypothesis.metric,
                    hypothesis.group_id,
                    failure.context.get("method"),
                    hypothesis.estimand,
                )
            ] = failure

    cell_rows, failures = _collect_expected_cells(
        expected,
        row_map,
        failure_map,
        population,
        _ObservedRoster(
            roster_source,
            observed_by_metric,
            _roster_evidence_arms(observed_by_metric, counts),
        ),
    )

    ordered = tuple(sorted(cell_rows, key=cell_order))
    decisions = tuple(cell for cell in ordered if cell.method_role == "decision")
    complete = (
        bool(decisions) and roster_complete and all(cell not in failures for cell in decisions)
    )
    reason_context = (
        None
        if complete
        else {
            "missing_cells": [
                cell.model_dump(mode="json") for cell in decisions if cell in failures
            ]
        }
    )

    def project(row, failure):
        return row.model_copy(
            update={
                "source_snapshot_id": snapshot_id,
                "analysis_population": population,
                "sampling_available": failure is None,
                "sampling_reason_code": None if failure is None else failure.code,
                "sampling_reason_context": None if failure is None else failure.context,
                "failure_code": None if failure is None else failure.code,
                "failure_context": None if failure is None else failure.context,
                "decision_scope_complete": complete,
                "decision_scope_reason_code": None
                if complete
                else "readout.scope.decision_incomplete",
                "decision_scope_reason_context": reason_context,
                **_weight_diagnostics_projection(row.method),
            }
        )

    output = [project(row, None) for row in rows]
    records = []
    for cell in ordered:
        row = cell_rows[cell]
        failure = failures.get(cell)
        if row is None:
            row = project(
                LiftEstimate(
                    metric=cell.metric,
                    group_id=cell.group_id,
                    method=cell.method,
                    method_role=cell.method_role,
                    estimand=cell.estimand,
                    analysis_population=population,
                    value_scale=cell.value_scale,
                    alternative=cell.alternative,
                    inference=cell.inference,
                    lift=None,
                    failure_code=failure.code,
                    failure_context=failure.context,
                    sampling_available=False,
                    sampling_reason_code=failure.code,
                    sampling_reason_context=failure.context,
                ),
                failure,
            )
            output.append(row)
        evidence = None
        if failure is None and row.lift is not None:
            try:
                evidence = PValueEvidence(
                    cell.hypothesis(), cell.method, row.p_value(), row.reference_kind
                )
            except (AttributeError, ValueError):
                evidence = None
        records.append(
            CellRecord(
                cell=cell,
                source_snapshot_id=snapshot_id,
                failure=failure,
                sampling=SamplingInference(
                    available=failure is None,
                    evidence=evidence,
                    reason_code=None if failure is None else failure.code,
                    reason_context=None if failure is None else failure.context,
                ),
            )
        )

    integrity = assignment_integrity(
        design,
        integrity_counts,
        population=population,
        randomization_grain="cluster" if getattr(src.context, "cluster", None) else "unit",
    )
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=ordered,
        decision_cells=decisions,
        rosters=(
            PopulationRoster(
                analysis_population=population,
                arms=arms,
                source=roster_source,
                complete=roster_complete,
            ),
        ),
        decision_complete_by_population={population: complete},
        integrity=(integrity,),
    )
    output, families = attach_multiplicity_scope(output, ordered, plan, configs, snapshot_id)
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=ordered,
        decision_cells=decisions,
        populations=(population,),
        families=families,
        by_source={snapshot_id: source_scope},
    )
    return LiftEstimates(
        output, metadata=ReadoutMetadata(scope=scope, cells=tuple(records)), source=source
    )
