"""Assemble complete randomized readout cells and their family scope."""

from __future__ import annotations

from increment.breakout.estimates import LiftEstimates
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.readout_types import CellKey, ReadoutScope, cell_order, freeze
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import AlwaysValid, AsymptoticMean
from increment.readouts._common import _raise
from increment.readouts._multiplicity_scope import attach_multiplicity_scope


def _randomized_source(
    selected,
    evidence_components,
    assignment_counts,
    population,
    *,
    source=None,
    dimension=None,
):
    from hashlib import sha256

    from increment._canonical import canonical_digest_bytes
    from increment.readouts._source_digest import component, composite_source

    components = [
        component(
            kind=kind,
            metric=metric.name,
            population=population,
            sha256=digest,
            source=source,
            dimension=dimension,
        )
        for metric in selected
        for kind, digest, _count in (evidence_components[metric.name],)
    ]
    if assignment_counts is not None:
        components.append(
            component(
                kind="assignment_counts",
                metric=None,
                population=population,
                sha256=sha256(canonical_digest_bytes(assignment_counts)).hexdigest(),
                source=source,
                dimension=dimension,
            )
        )
    return composite_source(components)


def _randomized_snapshot_id(source, request):
    from hashlib import sha256

    from increment._canonical import canonical_json_bytes

    preimage = {
        "kind": "increment.readout.snapshot",
        "version": 2,
        "collection": "LiftEstimates",
        "source": source,
        "request": request,
    }
    return "sha256:" + sha256(canonical_json_bytes(preimage)).hexdigest()


def _randomized_cell_record(cell, row, failure, config, snapshot_id):
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.readout_types import (
        CellRecord,
        PosteriorInference,
        SamplingInference,
    )

    evidence = None
    if row is not None and failure is None and row.lift is not None:
        try:
            evidence = PValueEvidence(
                cell.hypothesis(), cell.method, row.p_value(), row.reference_kind
            )
        except (AttributeError, ValueError):
            evidence = None
    sampling = SamplingInference(
        available=failure is None and row is not None,
        evidence=evidence,
        reason_code=None if failure is None else failure.code,
        reason_context=None if failure is None else failure.context,
    )
    posterior = (
        None
        if config.prior is None
        else PosteriorInference(
            available=None if row is None else row.posterior_available,
            model=None if row is None else row.posterior_model,
            scale=None if row is None else row.posterior_scale,
            prior=config.prior,
            reason_code=None if row is None else row.posterior_reason_code,
            reason_context=None if row is None else row.posterior_reason_context,
        )
    )
    return CellRecord(
        cell=cell,
        source_snapshot_id=snapshot_id,
        failure=failure,
        sampling=sampling,
        posterior=posterior,
    )


def _randomized_cell_registry(
    selected, configs, design, plan, computations, roster_arms, population
):
    from increment.readouts._common import _runtime_method_roles

    failure_by_cell = {}
    methodless_failures = []
    for computation in computations:
        for key, failure in computation.failures.items():
            method = failure.context.get("method")
            if method is None:
                methodless_failures.append((key, failure))
            else:
                failure_by_cell[(key.metric, key.group_id, method)] = failure
    method_info = {}
    cells = []
    decision_cells = []
    for metric, config in zip(selected, configs, strict=True):
        methods = (
            ()
            if config.methods_explicitly_empty
            else (config.decision_method, *config.sensitivity_methods)
        )
        roles = _runtime_method_roles(methods)
        for group_id in roster_arms:
            if group_id == design.control_group:
                continue
            for method in methods:
                method_role = roles[method.name]
                test = plan.procedures[metric.name]
                cell = CellKey(
                    kind="arm",
                    metric=metric.name,
                    group_id=group_id,
                    method=method.name,
                    method_role=method_role,
                    estimand="itt",
                    analysis_population=population,
                    value_scale="relative",
                    inference="always_valid"
                    if isinstance(plan.inference, AlwaysValid)
                    else "asymptotic_mean"
                    if isinstance(plan.inference, AsymptoticMean)
                    else "fixed",
                    alternative=getattr(test, "alternative", "two-sided"),
                )
                cells.append(cell)
                if method_role == "decision":
                    decision_cells.append(cell)
                method_info[(metric.name, group_id, method.name, method_role)] = (
                    cell,
                    config,
                    test,
                )
    for hypothesis, failure in methodless_failures:
        candidates = {
            method_name
            for metric_name, group_id, method_name, _role in method_info
            if metric_name == hypothesis.metric and group_id == hypothesis.group_id
        }
        if len(candidates) == 1:
            method = next(iter(candidates))
            failure_by_cell[(hypothesis.metric, hypothesis.group_id, method)] = failure
    return failure_by_cell, method_info, cells, decision_cells


def _randomized_cell_failures(
    method_info,
    failure_by_cell,
    rows,
    observed_by_metric,
    known_roster_arms,
    roster_source,
    population,
    decision_cells,
    roster_complete,
):
    from increment.estimation.readout_types import CellFailure

    row_by_cell = {(row.metric, row.group_id, row.method, row.method_role): row for row in rows}
    returned_cells = set(row_by_cell)
    cell_failures = {}
    for key, (cell, _config, _test) in method_info.items():
        metric_name, group_id, method_name, _method_role = key
        failure = failure_by_cell.get((metric_name, group_id, method_name))
        row = row_by_cell.get(key)
        if failure is not None:
            cell_failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(), code=failure.code, context=failure.context
            )
        elif row is not None and row.failure_code is not None:
            cell_failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(),
                code=row.failure_code,
                context=row.failure_context or {},
            )
        elif group_id not in observed_by_metric.get(metric_name, set()):
            observed = sorted(observed_by_metric.get(metric_name, set()))
            code = (
                "readout.cell.missing_metric_observations"
                if group_id in known_roster_arms
                else "readout.cell.missing_arm"
            )
            cell_failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(),
                code=code,
                context={
                    "metric": metric_name,
                    "group_id": group_id,
                    "observed_arms": observed,
                    "roster_source": roster_source,
                },
            )
        elif key not in returned_cells:
            cell_failures[cell] = CellFailure(
                hypothesis=cell.hypothesis(),
                code="readout.cell.unsupported_request",
                context={"reason": "no_estimate_returned", "analysis_population": population},
            )
    complete = (
        bool(decision_cells)
        and roster_complete
        and all(cell not in cell_failures for cell in decision_cells)
    )
    missing_cells = [
        cell.model_dump(mode="json") for cell in decision_cells if cell in cell_failures
    ]
    scope_reason_context = None if complete else freeze({"missing_cells": missing_cells})
    return cell_failures, complete, scope_reason_context


def _scoped_randomized_results(
    src,
    selected,
    configs,
    design,
    plan,
    rows,
    computations,
    observed_by_metric,
    evidence_components,
    population,
    base_roster=None,
):
    """Attach the complete randomized arm roster and captured cell outcomes."""
    from increment.estimation.readout_types import (
        PopulationRoster,
        ReadoutMetadata,
        SourceReadoutScope,
    )
    from increment.readouts._design_scope import _config_snapshot, resolve_roster

    roster = resolve_roster(src, design, population, observed_by_metric, base_roster=base_roster)
    roster_arms = roster.arms
    assignment_counts = roster.counts
    integrity_counts = roster.integrity_counts
    known_roster_arms = roster.known_arms
    roster_source = roster.source
    source = _randomized_source(selected, evidence_components, assignment_counts, population)
    from increment._source_identity import source_identity

    request = {
        "view": "run",
        "population": population,
        "metrics": [
            metric.model_dump(mode="json") for metric in sorted(selected, key=lambda m: m.name)
        ],
        "configs": [
            _config_snapshot(config, design)
            for config in sorted(configs, key=lambda c: c.metric.name)
        ],
        "design": design.model_dump(mode="json"),
        "alpha": plan.alpha,
        "q": plan.q,
        "declared": plan.declared,
        "procedures": {
            name: value.model_dump(mode="json") for name, value in plan.procedures.items()
        },
        "source_identity": source_identity(src),
    }
    trigger_name = src.context.trigger_name
    if trigger_name is not None:
        request["trigger_declared"] = True
        request["trigger_name"] = trigger_name
    snapshot_id = _randomized_snapshot_id(source, request)

    failure_by_cell, method_info, cells, decision_cells = _randomized_cell_registry(
        selected, configs, design, plan, computations, roster_arms, population
    )
    cell_failures, complete, scope_reason_context = _randomized_cell_failures(
        method_info,
        failure_by_cell,
        rows,
        observed_by_metric,
        known_roster_arms,
        roster_source,
        population,
        decision_cells,
        roster.complete,
    )

    result_by_cell = {}
    for row in rows:
        result_by_cell[(row.metric, row.group_id, row.method, row.method_role)] = row
    output = []
    records = []
    # Keep returned rows in request/registry order; only persisted scope keys are canonical-sorted.
    for cell in cells:
        if (
            cell.kind != "arm"
            or cell.group_id is None
            or cell.method is None
            or cell.method_role is None
            or cell.estimand is None
            or cell.value_scale is None
            or cell.inference is None
            or cell.alternative is None
        ):
            _raise("readout.scope.cell_incomplete", cell=cell.model_dump(mode="json"))
        assert cell.estimand in (
            "itt",
            "compliance",
            "late",
            "ate",
            "plr_slope",
            "overlap_subpopulation_ate",
        )
        failure = cell_failures.get(cell)
        row = result_by_cell.get((cell.metric, cell.group_id, cell.method, cell.method_role))
        if row is None and failure is not None:
            row = LiftEstimate(
                metric=cell.metric,
                group_id=cell.group_id,
                method=cell.method,
                method_role=cell.method_role,
                estimand=cell.estimand,
                analysis_population=population,
                value_scale=cell.value_scale,
                alternative=cell.alternative,
                inference=cell.inference,
                role=plan.procedures[cell.metric].role if plan.declared else None,
                lift=None,
                failure_code=failure.code,
                failure_context=failure.context,
                sampling_available=False,
                sampling_reason_code=failure.code,
                sampling_reason_context=failure.context,
                posterior_available=False
                if method_info[(cell.metric, cell.group_id, cell.method, cell.method_role)][1].prior
                is not None
                else None,
            )
        if row is not None:
            row = row.model_copy(
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
                    "decision_scope_reason_context": scope_reason_context,
                }
            )
            output.append(row)
        config = method_info[(cell.metric, cell.group_id, cell.method, cell.method_role)][1]
        records.append(_randomized_cell_record(cell, row, failure, config, snapshot_id))

    ordered_cells = tuple(sorted(set(cells), key=cell_order))
    ordered_decisions = tuple(sorted(set(decision_cells), key=cell_order))
    integrity = assignment_integrity(
        design,
        integrity_counts,
        population=population,
        randomization_grain="cluster" if getattr(src.context, "cluster", None) else "unit",
    )
    source_scope = SourceReadoutScope(
        source_snapshot_id=snapshot_id,
        cells=ordered_cells,
        decision_cells=ordered_decisions,
        rosters=(
            PopulationRoster(
                analysis_population=population,
                arms=roster_arms,
                source=roster_source,
                complete=roster.complete,
            ),
        ),
        decision_complete_by_population={population: complete},
        integrity=(integrity,),
    )
    output, families = attach_multiplicity_scope(output, ordered_cells, plan, configs, snapshot_id)
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=ordered_cells,
        decision_cells=ordered_decisions,
        populations=(population,),
        families=families,
        by_source={snapshot_id: source_scope},
    )
    records.sort(key=lambda record: cell_order(record.cell))
    metadata = ReadoutMetadata(scope=scope, cells=tuple(records))
    return LiftEstimates(output, metadata=metadata, source=source)
