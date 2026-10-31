from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from increment._interpretation import EffectEstimate
from increment.errors import CodedError, RefusalSpec, WireFormatError, refuse
from increment.estimation.readout_types import (
    ReadoutAssembly,
    ScopedReadout,
    merge_collection_metadata,
)
from increment.kernel.contracts import (
    BreakoutIdentity,
    CellKey,
    CompiledPlanBinding,
    Registration,
    Scope,
    method_config_hash,
    method_estimator_id,
    metric_hash,
)
from increment.kernel.ports import (
    DecisionLogPort,
    ResultsStorePort,
    RunHandlePort,
    RunRecord,
    program_store_operations,
)
from increment.result_envelope import AnalysisEnvelope, ResultCell, _cell_order

if TYPE_CHECKING:
    from increment._analysis_execution import Analysis
    from increment.decide.evaluator import Decisions
    from increment.decision import CompiledDecisionPlan
    from increment.kernel.experiment import Experiment
    from increment.kernel.resolver import Resolver


_CONTRAST_CONTEXT_NOT_PORTABLE = RefusalSpec(
    "wire.procedure.contrast_context",
    WireFormatError,
    template="switchback contrast runs carry no portable arm decision plan; {action} is unavailable",
)
_LIVE_RAW_SET_NOT_PORTABLE = RefusalSpec(
    "wire.result.raw_outcomes_unavailable",
    WireFormatError,
    template="a live confidence-set result depends on raw outcomes; {action} is unavailable",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    cells: tuple[ResultCell, ...]
    effects: tuple[EffectEstimate | None, ...]
    decisions: Decisions
    run: Any
    live_rows: tuple[Any | None, ...] = ()

    def __reduce__(self):
        reason = self.run.decision_plan_unavailable_reason
        if reason is not None:
            raise WireFormatError(
                "An in-memory-only decision plan cannot be serialized",
                code=reason,
                context={"run_id": self.run.run_id},
            )
        if any(row is not None for row in self.live_rows):
            refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="serialization")
        return type(self), (self.cells, self.effects, self.decisions, self.run, self.live_rows)


@dataclass(frozen=True, slots=True)
class _InMemoryRunHandle:
    run: RunRecord

    def get(self, identity):
        for cell in self.run.cells:
            if cell.identity.hash() == identity.hash():
                return cell
        raise CodedError(
            "Cell is absent from the in-memory run",
            code="store.run.roster_mismatch",
            context={"identity_hash": identity.hash()},
        )

    def envelope(self) -> AnalysisEnvelope:
        return AnalysisEnvelope(self.run.cells, self.run.metadata)


def _validate_analysis_binding(
    analysis: Analysis,
    registration: Registration,
    compiled_plan: CompiledDecisionPlan,
    analysis_options: Mapping[str, object] | None,
) -> None:
    if not registration.matches(analysis.compiled_decision_plan):
        raise CodedError(
            "Analysis execution plan differs from its pre-estimation registration",
            code="store.decision.reference_invalid",
            context={},
        )
    options = dict(analysis_options or {})
    requested_metrics = options.pop("metrics", None)
    if requested_metrics is not None:
        names = {
            item if isinstance(item, str) else getattr(item, "name", None)
            for item in cast(Sequence[object], requested_metrics)
        }
        if names != set(compiled_plan.procedures):
            raise CodedError(
                "Registered run cannot omit or add decision metrics",
                code="store.decision.reference_invalid",
                context={},
            )
    if options:
        raise CodedError(
            "Run options cannot override registered decision or population settings",
            code="store.decision.reference_invalid",
            context={"options": tuple(sorted(options))},
        )


def _events(run: RunRecord, decisions: Decisions, binding: CompiledPlanBinding) -> tuple[Any, ...]:
    operations = program_store_operations()
    cells_by_hash = {cell.identity.hash(): cell for cell in run.cells}
    metadata = run.metadata
    if metadata is None:
        raise CodedError(
            "Registered decision events require complete run metadata",
            code="store.run.metadata_mismatch",
            context={"run_id": run.run_id},
        )
    scope = metadata.scope
    output: list[Any] = []
    for verdict in decisions.families:
        if verdict.procedure not in {"bh", "e_bh", "bonferroni"}:
            continue
        family_scope = next(
            family
            for family in scope.families
            if family.family_id == verdict.family_id
            and family.source_snapshot_id == verdict.source_snapshot_id
            and str(family.analysis_population) == verdict.analysis_population
        )
        references = tuple(
            operations.DecisionReference(cells_by_hash[member.identity_hash].identity, run.run_id)
            for member in verdict.members
            if member.identity_hash in cells_by_hash
        )
        if len(references) != len(verdict.members):
            raise CodedError(
                "Family decision does not resolve to the complete stored roster",
                code="store.run.roster_mismatch",
                context={"family_id": verdict.family_id},
            )
        family_q = family_scope.family.q if family_scope.family is not None else None
        alpha_spent = float(family_q or 0.05)
        threshold = alpha_spent
        source = scope.by_source.get(verdict.source_snapshot_id)
        integrity = (
            "unknown"
            if source is None
            else ",".join(
                sorted(
                    {
                        result.status.value
                        for result in source.integrity
                        if str(result.analysis_population) == "assigned"
                    }
                )
            )
            or "unknown"
        )
        event_rule = cast(
            Literal["bh", "ebh", "bonferroni"],
            {"bh": "bh", "e_bh": "ebh", "bonferroni": "bonferroni"}[verdict.procedure],
        )
        output.append(
            operations.DecisionEvent(
                experiment_id=run.experiment_id,
                family_id=verdict.family_id,
                registration_hash=run.registration_hash or binding.registration_hash,
                decided_at=datetime.now(UTC),
                rule=event_rule,
                threshold=threshold,
                alpha_spent=alpha_spent,
                references=references,
                evidence_digest=operations.run_digest(run),
                integrity_status=integrity,
                outcome=verdict.status,
                reason_codes=verdict.reason_codes,
                decision_plan=binding,
            )
        )
    return tuple(output)


def _execution_identity_fields(
    analysis: Analysis,
    cell: ResultCell,
    identity_methods: Mapping[tuple[str, str, str], Any] | None = None,
    identity_metrics: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    key = cell.identity.key
    if key.metric == "uptake" and key.estimand == "compliance":
        metric = analysis._execution_config_for_cell(cell)[0]
    else:
        metric = None if identity_metrics is None else identity_metrics.get(key.metric)
    if key.kind == "contrast":
        if metric is None:
            metric = next((item for item in analysis._metrics if item.name == key.metric), None)
        if metric is None:
            raise CodedError(
                "Contrast cell has no matching metric declaration",
                code="store.run.identity_incomplete",
                context={"field": "metric_identity"},
            )
        return {
            "metric_hash": metric_hash(metric),
            "estimator_id": None,
            "method_config_hash": None,
            "method_instance_id": None,
        }
    if key.method is None and key.method_role is None:
        if metric is None:
            metric = analysis._execution_config_for_cell(cell)[0]
        return {
            "metric_hash": metric_hash(metric),
            "estimator_id": None,
            "method_config_hash": None,
            "method_instance_id": None,
        }
    method = (
        None
        if identity_methods is None
        else identity_methods.get(
            (key.metric, key.method or "unadjusted", key.method_role or "decision")
        )
    )
    methods: Sequence[Any] = ()
    if metric is None or method is None:
        configured_metric, methods = analysis._execution_config_for_cell(cell)
        if metric is None:
            metric = configured_metric
    if method is None:
        for index, candidate in enumerate(methods):
            role = "decision" if index == 0 else "sensitivity"
            if candidate.name == key.method and role == key.method_role:
                method = candidate
                break
    if method is None:
        raise CodedError(
            "Cell method has no matching execution configuration",
            code="store.run.identity_incomplete",
            context={"field": "method_identity"},
        )
    estimator = method_estimator_id(method)
    if estimator is None:
        raise CodedError(
            "Cell estimator has no registered execution identity",
            code="store.run.identity_incomplete",
            context={"field": "estimator_id"},
        )
    method_instance_id = f"{key.metric}:{key.method_role}:{method.name}"
    try:
        config_hash = method_config_hash(method, method_instance_id=method_instance_id)
    except CodedError as error:
        if error.code != "kernel.method.implementation_ref_missing":
            raise
        config_hash = None
    return {
        "metric_hash": metric_hash(metric),
        "estimator_id": estimator,
        "method_config_hash": config_hash,
        "method_instance_id": method_instance_id,
    }


def _metadata_with_cell_scales(metadata: Any, cells: Sequence[ResultCell]) -> Any:
    """Keep retained metadata keyed by the cell identities emitted by report adapters."""
    if metadata is None:
        return None

    cells_by_source: dict[str | None, list[CellKey]] = {}
    for cell in cells:
        cells_by_source.setdefault(cell.identity.source_snapshot_id, []).append(cell.identity.key)

    key_map: dict[Any, Any] = {}
    for record in metadata.cells:
        candidates = cells_by_source.get(record.source_snapshot_id, [])
        if record.cell in candidates:
            key_map[record.cell] = record.cell
            continue
        matching = [
            candidate
            for candidate in candidates
            if candidate.model_copy(update={"value_scale": None})
            == record.cell.model_copy(update={"value_scale": None})
        ]
        if len(matching) == 1:
            key_map[record.cell] = matching[0]
    if not any(old != new for old, new in key_map.items()):
        return metadata

    def rekey(key):
        return key_map.get(key, key)

    by_source = {
        source_id: source.model_copy(
            update={
                "cells": tuple(sorted({rekey(key) for key in source.cells}, key=_cell_order)),
                "decision_cells": tuple(
                    sorted({rekey(key) for key in source.decision_cells}, key=_cell_order)
                ),
            }
        )
        for source_id, source in metadata.scope.by_source.items()
    }
    scope = metadata.scope.model_copy(
        update={
            "cells": tuple(sorted({rekey(key) for key in metadata.scope.cells}, key=_cell_order)),
            "decision_cells": tuple(
                sorted({rekey(key) for key in metadata.scope.decision_cells}, key=_cell_order)
            ),
            "by_source": by_source,
            "families": tuple(
                family.model_copy(
                    update={
                        "members": tuple(
                            sorted({rekey(key) for key in family.members}, key=_cell_order)
                        )
                    }
                )
                for family in metadata.scope.families
            ),
        }
    )
    records = tuple(
        sorted(
            (record.model_copy(update={"cell": rekey(record.cell)}) for record in metadata.cells),
            key=lambda record: (
                0 if record.cell.analysis_population == "assigned" else 1,
                record.source_snapshot_id,
                _cell_order(record.cell),
            ),
        )
    )
    return metadata.model_copy(update={"scope": scope, "cells": records})


def _execution_overrides(
    analysis: Analysis,
    rows: Any,
    options: Mapping[str, object] | None,
    *,
    contrast: bool,
) -> tuple[Mapping[tuple[str, str, str], Any] | None, Mapping[str, Any] | None, bool]:
    if contrast:
        return None, {metric.name: metric for metric in analysis._metrics}, False
    if not options or not any(
        name in options
        for name in ("decision_method", "sensitivity_methods", "metrics", "exploratory_metrics")
    ):
        return None, None, False

    selected_metrics = options.get("metrics", ())
    extra_metrics = (
        tuple(
            metric
            for metric in selected_metrics
            if not isinstance(metric, str)
            and hasattr(metric, "name")
            and hasattr(metric, "model_dump")
        )
        if isinstance(selected_metrics, Sequence)
        else ()
    )
    exploratory_names = options.get("exploratory_metrics")
    if exploratory_names:
        extra_metrics += tuple(
            analysis._exploratory_metrics(
                cast(Sequence[str], exploratory_names),
                caller="run",
            )
        )
    identity = analysis._persistence_identity(
        rows,
        cast(Any, options.get("decision_method")),
        cast(Any, options.get("sensitivity_methods", ())),
        extra_metrics=cast(Any, extra_metrics),
    )
    runtime_method_unavailable = False
    for (metric_name, _method_name, role), method in identity.methods.items():
        if (
            method.folds is not None
            or method.propensity_learner is not None
            or method.outcome_learner is not None
        ):
            runtime_method_unavailable = True
            break
        try:
            method_config_hash(method, method_instance_id=f"{metric_name}:{role}:{method.name}")
        except CodedError as error:
            if error.code != "kernel.method.implementation_ref_missing":
                raise
            runtime_method_unavailable = True
            break
    return identity.methods, identity.metrics, runtime_method_unavailable


def _scope_for_cell(cell: ResultCell, declared_scope: Scope | None) -> Scope:
    key = cell.identity.key
    if declared_scope is None:
        if key.kind == "segment":
            return Scope(
                view="breakout",
                grain="total",
                population=key.analysis_population,
                segment=(BreakoutIdentity(property=key.dimension or "", source=key.source),),
            )
        return Scope(
            view="run",
            grain="total",
            population=key.analysis_population,
        )

    resolved_scope = cell.identity.scope
    scope_update: dict[str, Any] = {"population": key.analysis_population}
    if resolved_scope is not None:
        scope_update["time_basis"] = resolved_scope.time_basis
    raw_day = key.ds
    day: date | None = None
    if isinstance(raw_day, datetime):
        day = raw_day.date()
    elif isinstance(raw_day, date):
        day = raw_day
    elif isinstance(raw_day, str):
        try:
            day = date.fromisoformat(raw_day)
        except ValueError:
            try:
                day = datetime.fromisoformat(raw_day).date()
            except ValueError:
                day = None
    if isinstance(day, date):
        scope_update["through_day"] = day
    if key.ds_basis is not None:
        scope_update["time_basis"] = key.ds_basis

    row_scope = declared_scope.model_copy(update=scope_update)
    if declared_scope.segment and key.dimension is not None:
        row_scope = row_scope.model_copy(
            update={"segment": (BreakoutIdentity(property=key.dimension, source=key.source),)}
        )
    return row_scope


def _live_winsor_effect(row: Any) -> EffectEstimate:
    confidence_set = row.confidence_set
    value_scale = getattr(row, "value_scale", "relative")
    interval = confidence_set.additive if value_scale == "absolute" else confidence_set.relative
    point = confidence_set.additive_point if value_scale == "absolute" else confidence_set.point
    reasons = tuple(
        f"{side} endpoint {endpoint.status}: {endpoint.reason}"
        for side, endpoint in (("lower", interval.lower), ("upper", interval.upper))
        if endpoint.status != "finite" and endpoint.reason is not None
    )
    reason = (
        getattr(row, "relative_unavailable_reason", None)
        if point is None and value_scale != "absolute"
        else None
    )
    if reason is None and reasons:
        reason = "; ".join(reasons)
    return EffectEstimate(
        point,
        interval.lower.value if interval.lower.status == "finite" else None,
        interval.upper.value if interval.upper.status == "finite" else None,
        confidence_set.alpha,
        value_scale,
        reason,
        qualification=confidence_set.qualification,
    )


def _selected_percentile_winsorization(analysis: Any, options: Mapping[str, object]) -> bool:
    metrics = tuple(getattr(analysis, "_metrics", ()))
    selected = options.get("metrics")
    if selected is None:
        candidates = metrics
    else:
        selected_metrics = cast(Sequence[object], selected)
        names = {
            value if isinstance(value, str) else getattr(value, "name", None)
            for value in selected_metrics
        }
        candidates = tuple(metric for metric in metrics if metric.name in names)
    return any(
        getattr(getattr(metric, "winsorization", None), "has_percentile", False)
        for metric in candidates
    )


def _merge_scoped_readouts(parts: Sequence[ScopedReadout]) -> Any:
    rows = parts[0].collection
    sequential_snapshot = rows.sequential_snapshot
    for part in parts[1:]:
        if part.collection.metadata is None:
            continue
        if rows.metadata is None:
            rows = part.collection
            sequential_snapshot = sequential_snapshot or rows.sequential_snapshot
            continue
        metadata, source, merged = merge_collection_metadata(
            rows, part.collection, scoped_assembly=True
        )
        sequential_snapshot = sequential_snapshot or part.collection.sequential_snapshot
        rows = ReadoutAssembly(tuple(merged.values()), metadata, source, sequential_snapshot)
    return rows


def _run_rows(  # noqa: PLR0913
    analysis: Analysis,
    rows: Any,
    registration: Registration,
    compiled_plan: CompiledDecisionPlan | None,
    *,
    experiment_id: str,
    analysis_spec_hash: str,
    execution_identity_hash: str,
    run_id: str,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    declared_scope: Scope | None = None,
    identity_options: Mapping[str, object] | None = None,
    scoped_rows: Sequence[ScopedReadout] | None = None,
) -> RunResult:
    context = analysis._state.source.context
    contrast_context = (
        context if not hasattr(context, "plan") and hasattr(context, "procedures") else None
    )
    parts = (
        tuple(scoped_rows)
        if scoped_rows is not None
        else (ScopedReadout(rows, declared_scope, identity_options or {}),)
    )
    overrides = [
        _execution_overrides(
            analysis, part.collection, part.options, contrast=contrast_context is not None
        )
        for part in parts
    ]
    runtime_method_unavailable = any(item[2] for item in overrides)
    rows = _merge_scoped_readouts(parts)
    if contrast_context is not None:
        if results_store is not None or decision_log is not None:
            refuse(
                _CONTRAST_CONTEXT_NOT_PORTABLE,
                action="saving",
            )
        binding = None
        binding_unavailable_reason = "wire.procedure.contrast_context"
    else:
        try:
            if runtime_method_unavailable:
                raise WireFormatError(
                    "A runtime method has no portable implementation reference",
                    code="wire.procedure.runtime_method",
                    context={},
                )
            if compiled_plan is None:
                raise CodedError(
                    "A run decision plan is unavailable",
                    code="store.run.identity_incomplete",
                    context={"field": "compiled_plan"},
                )
            binding = CompiledPlanBinding.from_plan(registration, compiled_plan)
            binding_unavailable_reason = None
        except WireFormatError as error:
            if (
                error.code != "wire.procedure.runtime_method"
                or results_store is not None
                or decision_log is not None
            ):
                raise
            binding = None
            binding_unavailable_reason = error.code
    created_at = datetime.now(UTC)
    cells: list[ResultCell] = []
    live_rows: list[Any | None] = []
    for part, (identity_methods, identity_metrics, _) in zip(parts, overrides, strict=True):
        collection = part.collection
        live_rows_by_key = {
            ResultCell.from_row(row, collection.metadata, created_at=created_at).identity.key: row
            for row in collection
            if getattr(row, "confidence_set", None) is not None
        }
        if live_rows_by_key and results_store is not None:
            refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="saving")
        for cell in collection.as_cells(created_at=created_at):
            live_rows.append(live_rows_by_key.get(cell.identity.key))
            row_scope = _scope_for_cell(cell, part.scope)
            identity = cell.identity.model_copy(
                update={
                    **_execution_identity_fields(
                        analysis, cell, identity_methods, identity_metrics
                    ),
                    "experiment_id": experiment_id,
                    "analysis_spec_hash": part.analysis_spec_hash or analysis_spec_hash,
                    "scope": row_scope,
                }
            )
            provenance = replace(
                cell.provenance,
                run_id=run_id,
                execution_identity_hash=execution_identity_hash,
                artifact_manifest_sha256=(
                    cell.provenance.artifact_manifest_sha256
                    or (
                        None
                        if collection.metadata is None
                        else collection.metadata.artifact_manifest_sha256
                    )
                ),
                artifact_relation_digests=(
                    cell.provenance.artifact_relation_digests
                    or (
                        ()
                        if collection.metadata is None
                        else collection.metadata.artifact_relation_digests
                    )
                ),
            )
            cells.append(replace(cell, identity=identity, provenance=provenance))
    run_record = program_store_operations().Run(
        experiment_id=experiment_id,
        run_id=run_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        registration_hash=registration.registration_hash,
        expected_roster=tuple(cell.identity for cell in cells),
        cells=tuple(cells),
        metadata=_metadata_with_cell_scales(rows.metadata, cells),
        captured_at=created_at,
        committed_at=datetime.now(UTC),
        scope_analysis_hashes=tuple(
            dict.fromkeys(
                (
                    analysis_spec_hash,
                    *(cast(str, cell.identity.analysis_spec_hash) for cell in cells),
                )
            )
        ),
        cell_scope_hashes=tuple(
            (cell.identity.hash(), cast(str, cell.identity.analysis_spec_hash)) for cell in cells
        ),
        decision_plan=binding,
        decision_plan_unavailable_reason=binding_unavailable_reason,
        empty_selection_reason="empty_metric_selection" if not cells else None,
    )
    if results_store is None:
        handle: RunHandlePort = _InMemoryRunHandle(run_record)
    else:
        results_store.put(run_record)
        handle = results_store.open_run(experiment_id, analysis_spec_hash, run_id)
    from increment.decide import decide

    decisions = decide(handle, registration, contrast_context or compiled_plan)
    events = ()
    if decision_log is not None:
        assert binding is not None
        events = _events(handle.run, decisions, binding)
    envelope = handle.envelope()
    result = RunResult(
        cells=envelope.cells,
        effects=tuple(
            _live_winsor_effect(row) if row is not None else cell.summary
            for cell, row in zip(envelope.cells, live_rows, strict=True)
        ),
        decisions=decisions,
        run=handle.run,
        live_rows=tuple(live_rows),
    )
    if decision_log is not None:
        for event in events:
            decision_log.append(event)
    return result


@overload
def run(
    exp: Experiment,
    registration: Resolver | None = None,
    compiled_plan: CompiledDecisionPlan | None = None,
    *,
    analysis: None = None,
    resolver: Resolver | None = None,
    experiment_id: str | None = None,
    analysis_spec_hash: str | None = None,
    execution_identity_hash: str | None = None,
    run_id: str | None = None,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    analysis_options: Mapping[str, object] | None = None,
    views: Sequence[Scope] | None = None,
) -> Any: ...


@overload
def run(
    exp: Analysis | None = None,
    registration: Registration | None = None,
    compiled_plan: CompiledDecisionPlan | None = None,
    *,
    analysis: Analysis | None = None,
    resolver: None = None,
    experiment_id: str | None = None,
    analysis_spec_hash: str | None = None,
    execution_identity_hash: str | None = None,
    run_id: str | None = None,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    analysis_options: Mapping[str, object] | None = None,
) -> RunResult: ...


def run(  # noqa: PLR0913
    exp: Analysis | object | None = None,
    registration: Registration | object | None = None,
    compiled_plan: CompiledDecisionPlan | None = None,
    *,
    analysis: object | None = None,
    resolver: object | None = None,
    experiment_id: str | None = None,
    analysis_spec_hash: str | None = None,
    execution_identity_hash: str | None = None,
    run_id: str | None = None,
    results_store: ResultsStorePort | None = None,
    decision_log: DecisionLogPort | None = None,
    analysis_options: Mapping[str, object] | None = None,
    views: Sequence[Scope] | None = None,
) -> Any:
    """Run an experiment through its resolver, or a legacy Analysis through its registration.

    ``views`` selects declared Experiment scopes by hash, in declaration order.
    ``None`` executes all declared scopes in one capture. Legacy Analysis uses
    its registered scope and does not accept view selection.
    """
    if exp is not None and analysis is not None:
        raise CodedError(
            "Pass either exp or analysis, not both",
            code="kernel.run.invalid_input",
            context={"field": "exp"},
        )
    exp = analysis if exp is None else exp
    from increment.kernel.experiment import Experiment

    if isinstance(exp, Experiment):
        from increment.kernel.resolver import run as run_experiment

        unsupported = {
            "compiled_plan": compiled_plan,
            "experiment_id": experiment_id,
            "analysis_spec_hash": analysis_spec_hash,
            "execution_identity_hash": execution_identity_hash,
            "run_id": run_id,
            "results_store": results_store,
            "decision_log": decision_log,
        }
        supplied = tuple(name for name, value in unsupported.items() if value is not None)
        if supplied:
            raise CodedError(
                "Experiment execution does not support these persistence options",
                code="kernel.run.invalid_input",
                context={"fields": supplied},
            )
        selected_resolver = resolver if resolver is not None else registration
        if selected_resolver is None:
            raise CodedError(
                "Running an experiment requires a source resolver",
                code="kernel.run.resolver_missing",
                context={},
            )
        result_factory_setter = getattr(selected_resolver, "set_run_result_factory", None)
        if callable(result_factory_setter):
            result_factory_setter(_run_rows)
        return run_experiment(
            exp,
            cast("Resolver", selected_resolver),
            options=dict(analysis_options or {}),
            views=views,
        )
    if views is not None:
        raise CodedError(
            "View selection requires an Experiment",
            code="kernel.run.invalid_input",
            context={"field": "views"},
        )
    analysis = cast(Any, exp)
    if (
        not isinstance(registration, Registration)
        or compiled_plan is None
        or experiment_id is None
        or analysis_spec_hash is None
        or execution_identity_hash is None
        or run_id is None
    ):
        raise CodedError(
            "Legacy Analysis runs require a registration, plan, and run identities",
            code="store.run.identity_incomplete",
            context={"field": "run_arguments"},
        )
    # Legacy calls retain the established Analysis execution and return contract.
    if not registration.matches(compiled_plan):
        raise CodedError(
            "Compiled plan differs from the pre-evidence registration",
            code="store.decision.reference_invalid",
            context={},
        )
    _validate_analysis_binding(analysis, registration, compiled_plan, analysis_options)
    if results_store is not None and _selected_percentile_winsorization(
        analysis, analysis_options or {}
    ):
        refuse(_LIVE_RAW_SET_NOT_PORTABLE, action="saving")
    if results_store is not None or decision_log is not None:
        # Check wire portability before Analysis.run() can read the source.
        CompiledPlanBinding.from_plan(registration, compiled_plan)
    execute = cast(Callable[..., Any], analysis.run)
    rows = execute(**dict(analysis_options or {}))
    return _run_rows(
        analysis,
        rows,
        registration,
        compiled_plan,
        experiment_id=experiment_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        run_id=run_id,
        results_store=results_store,
        decision_log=decision_log,
    )
