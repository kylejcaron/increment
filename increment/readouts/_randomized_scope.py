"""Assemble complete randomized readout cells and their family scope."""

from __future__ import annotations

from increment.breakout.estimates import LiftEstimates
from increment.estimation.assignment_integrity import assignment_integrity
from increment.estimation.readout_types import CellKey, ReadoutScope, cell_order, freeze
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import AlwaysValid, AsymptoticMean
from increment.readouts._common import _raise
from increment.readouts._metric_type import declared_metric_type, declared_metric_types
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

    components = []

    for metric in selected:
        component_set = evidence_components[metric.name]
        if (
            len(component_set) == 3
            and isinstance(component_set[0], str)
            and isinstance(component_set[1], str)
            and isinstance(component_set[2], int)
        ):
            component_set = (component_set,)
        for kind, digest, count in component_set:
            item = component(
                kind=kind,
                metric=metric.name,
                population=population,
                sha256=digest,
                source=source,
                dimension=dimension,
            )
            item["count"] = count
            components.append(item)
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


def _randomized_content_digest(evidence_by_metric):
    from collections.abc import Mapping
    from hashlib import sha256

    import narwhals as nw

    from increment._canonical import canonical_json_bytes

    excluded = {
        "fit",
        "fit_state",
        "fit_answer",
        "method_fit_answers",
        "coefficients",
        "influence_references",
        "source_identity",
        "source_identity_hash",
        "source_snapshot_id",
        "study_id",
        "experiment_id",
        "artifact_id",
        "generation_id",
        "family_id",
        "created_at",
    }

    def input_payload(value):
        import math
        from datetime import date, datetime
        from enum import Enum
        from numbers import Integral, Real

        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            payload = value.model_dump(mode="json")
            return input_payload(
                {key: item for key, item in payload.items() if key.lower() not in excluded}
            )
        if isinstance(value, Mapping):
            return {str(key): input_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            values = [input_payload(item) for item in value]
            if values and all(isinstance(item, dict) for item in values):
                keys = set().union(*(item.keys() for item in values))
                return [{key: item.get(key) for key in sorted(keys)} for item in values]
            return values
        if isinstance(value, (set, frozenset)):
            return sorted((input_payload(item) for item in value), key=canonical_json_bytes)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if isinstance(value, Enum):
            return input_payload(value.value)
        if isinstance(value, bool):
            return value
        if isinstance(value, Integral):
            number = int(value)
            return number if -(2**53) < number < 2**53 else {"$integer": str(number)}
        if isinstance(value, Real):
            number = float(value)
            if not math.isfinite(number):
                return {"$float": number.hex() if not math.isnan(number) else "nan"}
            return number
        return value

    from increment.estimation.readout_types import StreamingDigest

    def rows_digest(rows, *, moment_rows=False):
        digest = StreamingDigest()
        for row in rows:
            if moment_rows and isinstance(row, Mapping):
                row = {key: item for key, item in row.items() if str(key).lower() not in excluded}
            digest.update(input_payload(row))
        return {"digest": digest.hexdigest(), "count": digest.count}

    metric_inputs = []
    for metric, evidence in sorted(evidence_by_metric.items()):
        unit_digest = None
        if evidence.unit_frame is not None:
            frame = nw.from_native(evidence.unit_frame, eager_only=True)
            unit_digest = rows_digest(frame.iter_rows(named=True))
        method_fit_inputs = []
        for request, answer in evidence.method_fit_answers:
            cell = request.cell
            if (
                answer.origin in {"warehouse", "retained_fitted", "retained_categorical"}
                and answer.evidence_sha256 is None
            ):
                return None, "method_fit_input_unavailable"
            method_fit_inputs.append(
                {
                    "metric": request.metric.name,
                    "method": input_payload(request.method.model_dump(mode="json")),
                    "control": request.control,
                    "arm_order": request.arm_order,
                    "population": request.population,
                    "cell": None
                    if cell is None
                    else {
                        "breakout": {
                            "property": cell.breakout.property,
                            "source": cell.breakout.source,
                        },
                        "value": cell.value,
                        "is_null": cell.is_null,
                    },
                    "input_evidence_sha256": answer.evidence_sha256,
                }
            )
        metric_inputs.append(
            {
                "metric": metric,
                "rows": rows_digest(evidence.rows, moment_rows=True),
                "method_rows": [
                    {"method": name, **rows_digest(rows, moment_rows=True)}
                    for name, rows in sorted(evidence.method_rows)
                ],
                "method_fit_inputs": sorted(method_fit_inputs, key=canonical_json_bytes),
                "unit_rows": unit_digest,
            }
        )

    preimage = {
        "kind": "increment.source_content",
        "version": 1,
        "metric_inputs": metric_inputs,
    }
    return "sha256:" + sha256(canonical_json_bytes(preimage)).hexdigest(), None


def _randomized_cell_record(
    cell,
    row,
    failure,
    config,
    snapshot_id,
    advisories=(),
    *,
    source_content_digest,
    artifact_provenance,
):
    from increment.estimation.decision_types import PValueEvidence
    from increment.estimation.readout_types import (
        CellRecord,
        PosteriorInference,
        SamplingInference,
    )

    evidence = None
    if row is not None and failure is None:
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
        advisories=tuple(advisories),
        source_content_digest=source_content_digest,
        artifact_manifest_sha256=artifact_provenance.manifest_sha256,
        artifact_relation_digests=artifact_provenance.relation_digests,
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
    # Decision computations only register decision-role failures, so a failure
    # without a method discriminator belongs to the metric's one decision cell.
    for hypothesis, failure in methodless_failures:
        for metric_name, group_id, method_name, role in method_info:
            if (
                role == "decision"
                and metric_name == hypothesis.metric
                and group_id == hypothesis.group_id
            ):
                failure_by_cell.setdefault((metric_name, group_id, method_name), failure)
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


def _scoped_randomized_results(  # noqa: PLR0915
    src,
    selected,
    configs,
    design,
    plan,
    rows,
    computations,
    observed_by_metric,
    evidence_by_metric,
    artifact_provenance,
    population,
    base_roster=None,
):
    """Attach the complete randomized arm roster and captured cell outcomes."""
    from increment.estimation.readout_types import (
        PopulationRoster,
        ReadoutMetadata,
        SourceReadoutScope,
    )
    from increment.readouts._design_scope import _config_snapshot, _request_snapshot, resolve_roster

    roster = resolve_roster(src, design, population, observed_by_metric, base_roster=base_roster)
    roster_arms = roster.arms
    assignment_counts = roster.counts
    integrity_counts = roster.integrity_counts
    known_roster_arms = roster.known_arms
    roster_source = roster.source
    evidence_components = {
        name: evidence.evidence_components
        or ((evidence.evidence_kind, evidence.evidence_sha256, evidence.evidence_count),)
        for name, evidence in evidence_by_metric.items()
    }
    source = _randomized_source(selected, evidence_components, assignment_counts, population)
    source_content_digest, source_content_digest_unavailable_reason = _randomized_content_digest(
        evidence_by_metric
    )
    from increment._source_identity import source_identity

    artifact_manifest_sha256 = artifact_provenance.manifest_sha256
    artifact_relation_digests = artifact_provenance.relation_digests

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
            name: _request_snapshot(value.model_dump(mode="json"), field=f"procedures.{name}")
            for name, value in plan.procedures.items()
        },
        "source_identity": source_identity(src),
    }
    trigger_name = src.context.trigger_name
    if trigger_name is not None:
        request["trigger_declared"] = True
        request["trigger_name"] = trigger_name
    snapshot_id = _randomized_snapshot_id(source, request)
    decision_snapshot_id = None
    from increment.readouts._design_scope import has_covariate_declarations

    if has_covariate_declarations(configs):
        from increment.readouts._design_scope import decision_request_snapshot

        decision_source = _randomized_source(
            selected,
            {
                name: evidence.decision_evidence_components
                for name, evidence in evidence_by_metric.items()
            },
            assignment_counts,
            population,
        )
        decision_snapshot_id = _randomized_snapshot_id(
            decision_source,
            decision_request_snapshot(request, configs, design),
        )
        if decision_snapshot_id == snapshot_id:
            decision_snapshot_id = None

    failure_by_cell, method_info, cells, decision_cells = _randomized_cell_registry(
        selected, configs, design, plan, computations, roster_arms, population
    )

    advisories_by_identity = {}
    for computation in computations:
        for hypothesis, advisories in computation.advisories.items():
            identity = (
                hypothesis.metric,
                getattr(hypothesis, "group_id", None),
                getattr(hypothesis, "estimand", None),
                getattr(hypothesis, "dimension", None),
                getattr(hypothesis, "dimension_value", None),
            )
            values = advisories_by_identity.setdefault(identity, {})
            values.update({item.model_dump_json(): item for item in advisories})
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
    metric_types = declared_metric_types(selected)
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
        config = method_info[(cell.metric, cell.group_id, cell.method, cell.method_role)][1]
        test = method_info[(cell.metric, cell.group_id, cell.method, cell.method_role)][2]
        metric = next(item for item in selected if item.name == cell.metric)
        prior_context = (
            {
                "metric": cell.metric,
                "group_id": cell.group_id,
                "reference_df": None,
                "missing": "HC2/BM score statistics needed to compose the declared prior",
            }
            if config.prior is not None
            else None
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
                preferred_direction=metric.declared_preferred_direction,
                null_lift=getattr(test, "null_lift", 0.0),
                null_abs=getattr(test, "null_abs", None),
                lift=None,
                failure_code=failure.code,
                failure_context=failure.context,
                sampling_available=False,
                sampling_reason_code=failure.code,
                sampling_reason_context=failure.context,
                posterior_available=False if config.prior is not None else None,
                posterior_reason_code=(
                    "estimation.results.lift.t_posterior_sufficient_statistics"
                    if config.prior is not None
                    else None
                ),
                posterior_reason_context=prior_context,
            )
        if row is not None:
            row = row.model_copy(
                update={
                    "source_snapshot_id": snapshot_id,
                    "metric_type": declared_metric_type(row, metric_types),
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
        advisories = advisories_by_identity.get(
            (
                cell.metric,
                cell.group_id,
                cell.estimand,
                cell.dimension,
                cell.dimension_value,
            ),
            {},
        )
        records.append(
            _randomized_cell_record(
                cell,
                row,
                failure,
                config,
                snapshot_id,
                tuple(advisories.values()),
                source_content_digest=source_content_digest,
                artifact_provenance=artifact_provenance,
            )
        )

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
                control_group=getattr(design, "control_group", None),
            ),
        ),
        decision_complete_by_population={population: complete},
        integrity=(integrity,),
    )
    output, families = attach_multiplicity_scope(
        output,
        ordered_cells,
        plan,
        configs,
        snapshot_id,
        decision_snapshot_id=decision_snapshot_id,
    )
    scope = ReadoutScope(
        snapshot_id=snapshot_id,
        cells=ordered_cells,
        decision_cells=ordered_decisions,
        populations=(population,),
        families=families,
        by_source={snapshot_id: source_scope},
    )
    records.sort(key=lambda record: cell_order(record.cell))
    metadata = ReadoutMetadata(
        scope=scope,
        cells=tuple(records),
        source_content_digest=source_content_digest,
        source_content_digest_unavailable_reason=source_content_digest_unavailable_reason,
        artifact_manifest_sha256=artifact_manifest_sha256,
        artifact_relation_digests=artifact_relation_digests,
    )
    return LiftEstimates(output, metadata=metadata, source=source)
