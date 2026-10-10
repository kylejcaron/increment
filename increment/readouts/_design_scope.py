"""Scope records for observational and encouragement readouts.

Expected metric x arm x method/estimand cells are enumerated from the request and the
source-owned roster before result projection; typed producer failures remain attached.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal

from increment._canonical import canonical_digest_bytes, canonical_json_bytes
from increment._labels import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL
from increment._source_identity import source_identity
from increment.breakout.estimates import LiftEstimates
from increment.errors import CapabilityError, CodedError
from increment.estimation.adjust import weight_diagnostics_projection
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.readout_types import (
    CellFailure,
    CellKey,
    CellRecord,
    PopulationRoster,
    PosteriorInference,
    ReadoutMetadata,
    ReadoutScope,
    SamplingInference,
    SourceReadoutScope,
    cell_order,
    freeze,
)
from increment.estimation.results import LiftEstimate
from increment.readouts._common import _raise, _runtime_method_roles, _runtime_methods
from increment.readouts._metric_rows import _rows_digest
from increment.readouts._multiplicity_scope import attach_multiplicity_scope
from increment.readouts._source_digest import component, composite_source

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
class ResolvedRoster:
    arms: tuple[str, ...]
    source: Literal["declared_allocation", "experiment_counts", "observed_union", "unknown"]
    complete: bool
    counts: Mapping[str, int] | None
    integrity_counts: Mapping[str, int] | None
    arms_by_metric: Mapping[str, set[str]]
    known_arms: frozenset[str]


def _method_role(value: str) -> Literal["decision", "sensitivity"]:
    if value in ("decision", "sensitivity"):
        return value
    _raise("readout.scope.invalid_axis", axis="method_role", value=value)


def _alternative(value: str) -> Literal["two-sided", "greater", "less"]:
    if value in ("two-sided", "greater", "less"):
        return value
    _raise("readout.scope.invalid_axis", axis="alternative", value=value)


def _value_scale(value: str) -> Literal["relative", "absolute"]:
    if value in ("relative", "absolute"):
        return value
    _raise("readout.scope.invalid_axis", axis="value_scale", value=value)


def _estimand(
    value: str,
) -> Literal["itt", "compliance", "late", "ate", "plr_slope", "overlap_subpopulation_ate"]:
    if value in ("itt", "compliance", "late", "ate", "plr_slope", "overlap_subpopulation_ate"):
        return value
    _raise("readout.scope.invalid_axis", axis="estimand", value=value)


def _is_accounting_label(arm: str | None) -> bool:
    return arm is None or arm in {MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL}


def _collect_expected_cells(
    expected: Sequence[ExpectedCell],
    row_map: dict[tuple, list[LiftEstimate]],
    failure_map: Mapping[tuple, Any],
    population: Literal["assigned", "triggered"],
    roster: ResolvedRoster,
) -> tuple[dict[CellKey, LiftEstimate | None], dict[CellKey, CellFailure]]:
    cell_rows: dict[CellKey, LiftEstimate | None] = {}
    failures: dict[CellKey, CellFailure] = {}
    methodless_counts: dict[tuple[str, str, str], int] = {}

    # Matched rows are projected to this population below; expected-cell keys must match.
    def population_cell(row: LiftEstimate) -> CellKey:
        cell = CellKey.from_row(row)
        if cell.analysis_population != population:
            return cell.model_copy(update={"analysis_population": population})
        return cell

    for item in expected:
        for estimand in (item.estimand, *item.estimand_alternates):
            key = (item.metric, item.group_id, estimand)
            methodless_counts[key] = methodless_counts.get(key, 0) + 1
    for item in expected:
        identities = (item.estimand, *item.estimand_alternates)
        matched = []
        failure = None
        for estimand in identities:
            key = (item.metric, item.group_id, item.method, estimand)
            matched = row_map.pop(key, [])
            if matched:
                break
            failure = failure_map.get(key)
            if failure is None and methodless_counts[(item.metric, item.group_id, estimand)] == 1:
                failure = failure_map.get((item.metric, item.group_id, None, estimand))
            if failure is not None:
                break
        if matched:
            for row in matched:
                cell_rows.setdefault(population_cell(row), row)
            continue
        cell = CellKey(
            kind="arm",
            metric=item.metric,
            group_id=item.group_id,
            method=item.method,
            method_role=_method_role(item.method_role),
            estimand=item.estimand,
            analysis_population=population,
            value_scale=_value_scale(item.value_scale),
            inference="fixed",
            alternative=_alternative(item.alternative),
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
            cell_rows.setdefault(population_cell(row), row)
    return cell_rows, failures


def resolve_roster(
    src,
    design,
    population: Literal["assigned", "triggered"],
    observed_by_metric,
    *,
    use_counts=True,
    base_roster: ResolvedRoster | None = None,
) -> ResolvedRoster:
    """Resolve arm identity and reject assignments outside a declared allocation."""
    is_clustered = getattr(src.context, "cluster", None) is not None
    if base_roster is None:
        counts = None
        if use_counts:
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
        if use_counts and counts is None and is_clustered and population == "assigned":
            cluster_counts = getattr(src, "cluster_counts", None)
            if callable(cluster_counts):
                try:
                    counts = {
                        arm: count for arm, count in cluster_counts().items() if arm is not None
                    }
                except CapabilityError:
                    pass
        if use_counts and counts is None and not is_clustered:
            count_method = getattr(src, "unit_counts", None)
            if callable(count_method):
                try:
                    counts = {
                        arm: count for arm, count in count_method().items() if arm is not None
                    }
                except CapabilityError:
                    pass
        integrity_counts = counts if not is_clustered or population == "assigned" else None
    else:
        counts = base_roster.counts
        integrity_counts = base_roster.integrity_counts

    observed_arms = {
        arm
        for observed in observed_by_metric.values()
        for arm in observed
        if not _is_accounting_label(arm)
    }
    known_arms = observed_arms | {arm for arm in (counts or {}) if not _is_accounting_label(arm)}
    source: Literal["declared_allocation", "experiment_counts", "observed_union", "unknown"]
    complete: bool
    allocation = getattr(design, "allocation", None)
    if allocation is not None:
        declared_arms = {arm for arm in allocation if not _is_accounting_label(arm)}
        undeclared_arms = known_arms - declared_arms
        if undeclared_arms:
            _raise(
                "readout.roster.undeclared_observed_arms",
                declared_arms=tuple(sorted(declared_arms)),
                undeclared_arms=tuple(sorted(undeclared_arms)),
                analysis_population=population,
            )
        arms = tuple(sorted(declared_arms))
        source, complete = "declared_allocation", True
    elif counts:
        arms = tuple(sorted(arm for arm in counts if not _is_accounting_label(arm)))
        source, complete = "experiment_counts", True
    else:
        arms = tuple(sorted(observed_arms))
        source, complete = ("observed_union" if arms else "unknown"), False
    per_metric = {
        metric: {arm for arm in observed if not _is_accounting_label(arm)}
        for metric, observed in observed_by_metric.items()
    }
    return ResolvedRoster(
        arms=arms,
        source=source,
        complete=complete,
        counts=counts,
        integrity_counts=integrity_counts,
        arms_by_metric=per_metric,
        known_arms=frozenset(known_arms),
    )


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


def encouragement_expected(selected, configs, design, plan, *, estimands, arms, cluster):
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


def compliance_component(summary, population, *, source=None, dimension=None):
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
    return component(
        kind="compliance_summary",
        metric=None,
        population=population,
        sha256=sha256(canonical_json_bytes(payload)).hexdigest(),
        source=source,
        dimension=dimension,
    )


def _request_snapshot(value: Any, *, field: str = "request") -> Any:
    """Make request data canonical, rejecting callables without stable import identity."""
    if callable(value):
        from importlib import import_module

        from increment.estimation.readout_types import refuse_readout

        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if (
            not isinstance(module, str)
            or not module
            or not isinstance(qualname, str)
            or not qualname
            or "<lambda>" in qualname
            or "<locals>" in qualname
        ):
            refuse_readout("readout.scope.request_not_canonical", fields=[field])
        try:
            resolved = import_module(module)
            for name in qualname.split("."):
                resolved = getattr(resolved, name)
        except (ImportError, AttributeError):
            refuse_readout("readout.scope.request_not_canonical", fields=[field])
        if resolved is not value:
            refuse_readout("readout.scope.request_not_canonical", fields=[field])
        return {"kind": "callable", "module": module, "qualname": qualname}
    if isinstance(value, dict):
        return {key: _request_snapshot(item, field=f"{field}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [
            _request_snapshot(item, field=f"{field}[{index}]") for index, item in enumerate(value)
        ]
    return value


def _config_snapshot(config, design):
    from increment._analysis_config import effective_methods
    from increment.estimation.readout_types import refuse_readout

    try:
        return {
            "metric": config.metric.model_dump(mode="json"),
            "methods": [
                _request_snapshot(
                    method.model_dump(mode="python"),
                    field=(
                        "configs.decision_method"
                        if index == 0
                        else f"configs.sensitivity_methods[{index - 1}]"
                    ),
                )
                for index, method in enumerate(effective_methods(config, design=design))
            ],
            "prior": None if config.prior is None else config.prior.model_dump(mode="json"),
            "prior_is_global": config.prior_is_global,
        }
    except CodedError:
        raise
    except (TypeError, ValueError):
        refuse_readout("readout.scope.request_not_canonical", fields=["configs"])


def scope_design_results(  # noqa: PLR0913
    src: Any,
    design: Any,
    plan: Any,
    rows: Sequence[LiftEstimate],
    computations: Sequence[Any],
    *,
    configs: Sequence[Any],
    population: Literal["assigned", "triggered"],
    expected_for: Any,
    evidence_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    observed_by_metric: Mapping[str, set[str]],
    compliance_summary: Any = None,
    estimands: Sequence[str] | None = None,
    use_counts: bool = True,
    base_roster: ResolvedRoster | None = None,
    request_extra: Mapping[str, Any] | None = None,
    component_source: str | None = None,
    dimension: str | None = None,
) -> LiftEstimates:
    """Attach source identity and every expected cell, preserving provider failures."""
    roster = resolve_roster(
        src,
        design,
        population,
        observed_by_metric,
        use_counts=use_counts,
        base_roster=base_roster,
    )
    arms = roster.arms
    counts = roster.counts
    integrity_counts = roster.integrity_counts
    expected = expected_for(arms)
    components = [
        component(
            kind="moments_rows",
            metric=metric,
            population=population,
            sha256=_rows_digest(list(evidence_rows[metric]))[0],
            source=component_source,
            dimension=dimension,
        )
        for metric in sorted(evidence_rows)
    ]
    if compliance_summary is not None:
        components.append(
            compliance_component(
                compliance_summary,
                population,
                source=component_source,
                dimension=dimension,
            )
        )
    if counts is not None:
        components.append(
            component(
                kind="assignment_counts",
                metric=None,
                population=population,
                sha256=sha256(canonical_digest_bytes(counts)).hexdigest(),
                source=component_source,
                dimension=dimension,
            )
        )
    source = composite_source(components)
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
            name: _request_snapshot(value.model_dump(mode="python"), field=f"procedures.{name}")
            for name, value in plan.procedures.items()
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
                    "version": 2,
                    "collection": "LiftEstimates",
                    "source_identity": source_identity(src),
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
        roster,
    )

    ordered = tuple(sorted(cell_rows, key=cell_order))
    decisions = tuple(cell for cell in ordered if cell.method_role == "decision")
    complete = (
        bool(decisions) and roster.complete and all(cell not in failures for cell in decisions)
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
    configs_by_metric = {config.metric.name: config for config in configs}

    def project(row, failure):
        return row.model_copy(
            update={
                "source_snapshot_id": snapshot_id,
                "analysis_population": population,
                "sampling_available": failure is None,
                "sampling_reason_code": None if failure is None else failure.code,
                "sampling_reason_context": None if failure is None else freeze(failure.context),
                "failure_code": None if failure is None else failure.code,
                "failure_context": None if failure is None else freeze(failure.context),
                "decision_scope_complete": complete,
                "decision_scope_reason_code": None
                if complete
                else "readout.scope.decision_incomplete",
                "decision_scope_reason_context": freeze(reason_context),
                **weight_diagnostics_projection(row.method, row=row, failure=failure),
            }
        )

    output = [project(row, None) for row in rows]
    records = []
    for cell in ordered:
        config = configs_by_metric.get(cell.metric)
        row = cell_rows[cell]
        failure = failures.get(cell)
        if row is None:
            if failure is None:
                _raise(
                    "readout.scope.cell_unresolved",
                    cell=cell.model_dump(mode="json"),
                )
            if (
                cell.group_id is None
                or cell.method is None
                or cell.method_role is None
                or cell.estimand is None
                or cell.value_scale is None
                or cell.inference is None
                or cell.alternative is None
            ):
                _raise(
                    "readout.scope.cell_incomplete",
                    cell=cell.model_dump(mode="json"),
                )
            row = project(
                LiftEstimate(
                    metric=cell.metric,
                    group_id=cell.group_id,
                    method=cell.method,
                    method_role=cell.method_role,
                    estimand=_estimand(cell.estimand),
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
                    posterior_available=False
                    if config is not None and config.prior is not None
                    else None,
                    posterior_reason_code=failure.code
                    if config is not None and config.prior is not None
                    else None,
                    posterior_reason_context=freeze(failure.context)
                    if config is not None and config.prior is not None
                    else None,
                ),
                failure,
            )
            output.append(row)
        evidence = None
        if failure is None and row.lift is not None and cell.method is not None:
            try:
                p_value = row.p_value()
                if isinstance(p_value, (int, float)):
                    evidence = PValueEvidence(
                        cell.hypothesis(), cell.method, p_value, row.reference_kind
                    )
            except (AttributeError, ValueError):
                evidence = None
        posterior = (
            None
            if config is None or config.prior is None
            else PosteriorInference(
                available=row.posterior_available,
                model=row.posterior_model,
                scale=row.posterior_scale,
                prior=config.prior,
                reason_code=row.posterior_reason_code,
                reason_context=row.posterior_reason_context,
            )
        )
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
                posterior=posterior,
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
                source=roster.source,
                complete=roster.complete,
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
