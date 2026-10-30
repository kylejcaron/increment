from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from increment._interpretation import EffectEstimate
from increment.errors import CodedError
from increment.kernel.contracts import (
    BreakoutIdentity,
    CompiledPlanBinding,
    Registration,
    Scope,
    method_config_hash,
    method_estimator_id,
    metric_hash,
)
from increment.result_envelope import AnalysisEnvelope, ResultCell
from increment.store import DecisionEvent, DecisionLog, DecisionReference, ResultsStore, Run
from increment.store.types import RunHandle, run_digest

if TYPE_CHECKING:
    from increment._analysis_execution import Analysis
    from increment.decide.evaluator import Decisions
    from increment.decision import CompiledDecisionPlan
    from increment.kernel.experiment import Experiment
    from increment.kernel.resolver import Resolver


@dataclass(frozen=True, slots=True)
class RunResult:
    cells: tuple[ResultCell, ...]
    effects: tuple[EffectEstimate | None, ...]
    decisions: Decisions
    run: Run


@dataclass(frozen=True, slots=True)
class _InMemoryRunHandle:
    run: Run

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


def _events(
    run: Run, decisions: Decisions, binding: CompiledPlanBinding
) -> tuple[DecisionEvent, ...]:
    cells_by_hash = {cell.identity.hash(): cell for cell in run.cells}
    scope = cast(Any, run.metadata).scope
    output: list[DecisionEvent] = []
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
            DecisionReference(cells_by_hash[member.identity_hash].identity, run.run_id)
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
            DecisionEvent(
                experiment_id=run.experiment_id,
                family_id=verdict.family_id,
                registration_hash=run.registration_hash or binding.registration_hash,
                decided_at=datetime.now(UTC),
                rule=event_rule,
                threshold=threshold,
                alpha_spent=alpha_spent,
                references=references,
                evidence_digest=run_digest(run),
                integrity_status=integrity,
                outcome=verdict.status,
                reason_codes=verdict.reason_codes,
                decision_plan=binding,
            )
        )
    return tuple(output)


def _execution_identity_fields(analysis: Analysis, cell: ResultCell) -> dict[str, str]:
    metric, methods = analysis._execution_config_for_cell(cell)
    key = cell.identity.key
    for index, method in enumerate(methods):
        role = "decision" if index == 0 else "sensitivity"
        if method.name != key.method or role != key.method_role:
            continue
        estimator = method_estimator_id(method)
        if estimator is None:
            raise CodedError(
                "Cell estimator has no registered execution identity",
                code="store.run.identity_incomplete",
                context={"field": "estimator_id"},
            )
        method_instance_id = f"{key.metric}:{role}:{method.name}"
        return {
            "metric_hash": metric_hash(metric),
            "estimator_id": estimator,
            "method_config_hash": method_config_hash(method, method_instance_id=method_instance_id),
            "method_instance_id": method_instance_id,
        }
    raise CodedError(
        "Cell method has no matching execution configuration",
        code="store.run.identity_incomplete",
        context={"field": "method_identity"},
    )


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
    results_store: ResultsStore | None = None,
    decision_log: DecisionLog | None = None,
    analysis_options: Mapping[str, object] | None = None,
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
    results_store: ResultsStore | None = None,
    decision_log: DecisionLog | None = None,
    analysis_options: Mapping[str, object] | None = None,
) -> RunResult: ...


def run(
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
    results_store: ResultsStore | None = None,
    decision_log: DecisionLog | None = None,
    analysis_options: Mapping[str, object] | None = None,
) -> Any:
    """Run an experiment through its resolver, or a legacy Analysis through its registration."""
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
        return run_experiment(
            exp, cast("Resolver", selected_resolver), options=dict(analysis_options or {})
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
    binding = CompiledPlanBinding.from_plan(registration, compiled_plan)
    execute = cast(Callable[..., Any], analysis.run)
    rows = execute(**dict(analysis_options or {}))
    created_at = datetime.now(UTC)
    cells: list[ResultCell] = []
    for cell in rows.as_cells(created_at=created_at):
        key = cell.identity.key
        scope = (
            Scope(
                view="breakout",
                grain="total",
                population=key.analysis_population,
                segment=(BreakoutIdentity(property=key.dimension or "", source=key.source),),
            )
            if key.kind == "segment"
            else Scope(
                view="run",
                grain="total",
                population=key.analysis_population,
            )
        )
        identity = cell.identity.model_copy(
            update={
                **_execution_identity_fields(analysis, cell),
                "experiment_id": experiment_id,
                "analysis_spec_hash": analysis_spec_hash,
                "scope": scope,
            }
        )
        provenance = replace(
            cell.provenance,
            run_id=run_id,
            execution_identity_hash=execution_identity_hash,
        )
        cells.append(replace(cell, identity=identity, provenance=provenance))
    run_record = Run(
        experiment_id=experiment_id,
        run_id=run_id,
        analysis_spec_hash=analysis_spec_hash,
        execution_identity_hash=execution_identity_hash,
        registration_hash=registration.registration_hash,
        expected_roster=tuple(cell.identity for cell in cells),
        cells=tuple(cells),
        metadata=rows.metadata,
        captured_at=created_at,
        committed_at=datetime.now(UTC),
        scope_analysis_hashes=(analysis_spec_hash,),
        cell_scope_hashes=tuple((cell.identity.hash(), analysis_spec_hash) for cell in cells),
        decision_plan=binding,
    )
    if results_store is None:
        handle: RunHandle = _InMemoryRunHandle(run_record)
    else:
        results_store.put(run_record)
        handle = results_store.open_run(experiment_id, analysis_spec_hash, run_id)
    from increment.decide import decide

    decisions = decide(handle, registration)
    events = () if decision_log is None else _events(handle.run, decisions, binding)
    result = RunResult(
        cells=handle.envelope().cells,
        effects=tuple(cell.summary for cell in handle.envelope().cells),
        decisions=decisions,
        run=handle.run,
    )
    if decision_log is not None:
        for event in events:
            decision_log.append(event)
    return result
