from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, cast

from increment.errors import CodedError
from increment.kernel.contracts import CompiledPlanBinding, Registration, contract_hash
from increment.kernel.ports import RunHandlePort, RunRecord
from increment.kernel.program_refusals import refuse_store
from increment.result_envelope import combine_family_evidence, family_evidence


@dataclass(frozen=True, slots=True)
class Verdict:
    identity_hash: str
    role: str | None
    status: Literal["reject", "retain", "not_evaluated"]
    discovery: bool | None
    reason_codes: tuple[str, ...]
    family_id: str | None
    membership: Literal["member", "nonmember"]


@dataclass(frozen=True, slots=True)
class FamilyVerdict:
    family_id: str
    source_snapshot_id: str
    analysis_population: str
    procedure: str
    members: tuple[Verdict, ...]
    status: Literal["reject", "retain", "not_evaluated"]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProgramVerdict:
    status: Literal["pass", "block", "not_evaluated"]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Decisions:
    cells: tuple[Verdict, ...]
    families: tuple[FamilyVerdict, ...]
    program: ProgramVerdict
    registration_hash: str
    run_id: str


def _integrity_by_scope(metadata: object, source_id: str | None) -> tuple[str, ...]:
    scope = getattr(metadata, "scope", None)
    source = (
        getattr(scope, "by_source", {}).get(source_id)
        if scope is not None and source_id is not None
        else None
    )
    if source is None:
        return ("integrity.unknown",)
    matches = [item for item in source.integrity if item.analysis_population == "assigned"]
    if not matches:
        return ("integrity.unknown",)
    bad = tuple(
        sorted({item.status.value for item in matches if item.status.value != "not_rejected"})
    )
    return tuple(f"integrity.{status}" for status in bad)


class _DecisionPlan(Protocol):
    procedures: Mapping[str, Any]


def _validated_plan(
    run: RunRecord,
    registration: Registration,
    plan: object | None,
) -> _DecisionPlan:
    if run.decision_plan is not None:
        bound_plan = run.decision_plan.validate(registration)
        if plan is not None and (
            CompiledPlanBinding.from_plan(registration, plan).canonical_payload()
            != run.decision_plan.canonical_payload()
        ):
            refuse_store(
                "store.decision.reference_invalid",
                "Supplied plan differs from the run's bound compiled plan",
                run_id=run.run_id,
            )
        return cast(_DecisionPlan, bound_plan)

    reason = run.decision_plan_unavailable_reason
    model_dump = getattr(plan, "model_dump", None)
    contrast = hasattr(plan, "study") and hasattr(plan, "procedures") and callable(model_dump)
    if reason == "wire.procedure.contrast_context" or (reason is None and contrast):
        # A switchback run keeps no portable plan; the contrast context its run supplies
        # must be the one the registration names.
        matches = (
            contrast
            and callable(model_dump)
            and registration.registration_hash
            == contract_hash("registration", model_dump(mode="json"))
        )
    else:
        matches = (
            reason == "wire.procedure.runtime_method"
            and plan is not None
            and registration.matches(plan)
        )
    if not matches:
        refuse_store(
            "store.decision.reference_invalid",
            "Run has no bound compiled decision plan",
            run_id=run.run_id,
        )
    return cast(_DecisionPlan, plan)


def _contrast_verdict(cell: Any, procedures: Mapping[str, Any]) -> Verdict:
    from importlib import import_module

    payload = cell.row_payload
    data = None if payload is None else payload.get("data")
    if not isinstance(data, Mapping):
        refuse_store(
            "store.decision.reference_invalid",
            "Contrast result payload is unavailable",
            run_id=cell.provenance.run_id,
        )
    result_type = cast(Any, import_module("increment.estimation.contrast_results")).ContrastResult
    result = result_type.model_validate(data)
    procedure = procedures.get(cell.identity.key.metric)
    if procedure is None or (
        result.metric != procedure.metric
        or result.role != procedure.role
        or result.alternative != procedure.alternative
        or result.null_abs != procedure.null_abs
        or result.alpha != procedure.alpha
    ):
        refuse_store(
            "store.decision.reference_invalid",
            "Contrast result differs from its registered decision procedure",
            run_id=cell.provenance.run_id,
        )
    contrast = import_module("increment.estimation.contrast")
    available = contrast.contrast_evidence_available(result)
    in_scope = result.decision_scope_complete is not False
    status: Literal["reject", "retain", "not_evaluated"] = (
        "not_evaluated"
        if not available or not in_scope
        else "reject"
        if contrast.contrast_stat_sig(result)
        else "retain"
    )
    reason_codes = (
        (result.decision_scope_reason_code or "decision.scope.incomplete",)
        if not in_scope
        else ("evidence.p_value.unavailable",)
        if not available
        else ()
    )
    return Verdict(
        cell.identity.hash(),
        str(getattr(procedure.role, "value", procedure.role)),
        status,
        None,
        reason_codes,
        None,
        "nonmember",
    )


def _report_verdict(cell: Any, procedures: Mapping[str, Any]) -> Verdict:
    row = cell._projection_row()

    procedure = procedures.get(cell.identity.key.metric)
    if (
        row is None
        or procedure is None
        or row.metric != procedure.metric
        or row.method != cell.identity.key.method
        or row.alternative != procedure.alternative
        or getattr(row, "null_lift", 0.0) != procedure.null_lift
        or getattr(row, "null_abs", None) != getattr(procedure, "null_abs", None)
    ):
        refuse_store(
            "store.decision.reference_invalid",
            "Report estimate differs from its registered decision procedure",
            run_id=cell.provenance.run_id,
        )
    stat_sig = row.stat_sig()
    available = stat_sig is not None
    in_scope = getattr(row, "decision_scope_complete", None) is not False
    status: Literal["reject", "retain", "not_evaluated"] = (
        "not_evaluated" if not available or not in_scope else "reject" if stat_sig else "retain"
    )
    reason_codes = (
        (getattr(row, "decision_scope_reason_code", None) or "decision.scope.incomplete",)
        if not in_scope
        else (getattr(row, "sampling_reason_code", None) or "evidence.p_value.unavailable",)
        if not available
        else ()
    )
    return Verdict(
        cell.identity.hash(),
        str(getattr(procedure.role, "value", procedure.role)),
        status,
        getattr(row, "discovery", None),
        reason_codes,
        cell.family.family_id if cell.family is not None else None,
        "member" if cell.family is not None else "nonmember",
    )


def _special_verdict(cell: Any, procedures: Mapping[str, Any]) -> Verdict | None:
    if cell.row_kind in {"breakout", "daily_lift", "asof_lift"}:
        return _report_verdict(cell, procedures)
    if cell.identity.key.kind == "contrast":
        return _contrast_verdict(cell, procedures)
    return None


def _scope_verdict(cell: Any, procedures: Mapping[str, Any]) -> Verdict | None:
    key = cell.identity.key
    scope = cell.identity.scope
    if scope is not None and scope.view in {"daily", "asof", "breakout"}:
        reason = "decision.not_evaluated.monitoring_view"
    elif key.estimand == "compliance":
        reason = "decision.not_evaluated.monitoring_view"
    elif key.method_role != "decision":
        reason = "decision.not_evaluated.exploratory_view"
    elif key.metric not in procedures:
        reason = "decision.not_evaluated.exploratory_view"
    else:
        return None

    payload = cell.row_payload
    data = None if payload is None else payload.get("data")
    role = key.method_role
    if isinstance(data, Mapping) and "role" in data:
        role = data["role"]
    elif cell.row_kind == "lift_estimate":
        retained = cell.components["sampling"][1]
        if retained is not None and "role" in retained.fields:
            role = retained.fields["role"]
    role = getattr(role, "value", role)
    return Verdict(
        cell.identity.hash(),
        None if role is None else str(role),
        "not_evaluated",
        None,
        (reason,),
        None if cell.family is None else cell.family.family_id,
        "nonmember" if cell.family is None else "member",
    )


def _validated_envelope(run: RunRecord, handle: RunHandlePort) -> Any:
    envelope = handle.envelope()
    if tuple(cell.identity.hash() for cell in envelope.cells) != tuple(
        cell.identity.hash() for cell in run.cells
    ):
        refuse_store(
            "store.run.roster_mismatch",
            "Run handle envelope differs from its pinned manifest",
            run_id=run.run_id,
        )
    if any(
        cell.identity.experiment_id != run.experiment_id
        or cell.identity.analysis_spec_hash not in run.scope_analysis_hashes
        or cell.provenance.run_id != run.run_id
        or cell.provenance.execution_identity_hash != run.execution_identity_hash
        for cell in envelope.cells
    ):
        refuse_store(
            "store.run.provenance_mismatch",
            "Run handle contains evidence outside its pinned manifest",
            run_id=run.run_id,
        )
    return envelope


def _missing_decision_roles(
    metadata: Any, cells: tuple[Any, ...], procedures: Mapping[str, Any]
) -> set[str]:
    whole_window_sources = {
        cell.identity.source_snapshot_id
        for cell in cells
        if cell.identity.key.method_role == "decision"
        and cell.identity.key.estimand != "compliance"
        and (cell.identity.scope is None or cell.identity.scope.view in {"run", "switchback"})
    }
    if not whole_window_sources or metadata is None or metadata.scope is None:
        return set()

    source_ids = whole_window_sources
    decision_roles = {
        metric: str(getattr(procedure.role, "value", procedure.role))
        for metric, procedure in procedures.items()
        if str(getattr(procedure.role, "value", procedure.role)) in {"primary", "guardrail"}
    }
    cells_by_source_population: dict[tuple[str | None, str], set[str]] = {}
    for cell in cells:
        key = cell.identity.key
        if key.method_role != "decision" or key.estimand == "compliance":
            continue
        cells_by_source_population.setdefault(
            (cell.identity.source_snapshot_id, key.analysis_population), set()
        ).add(key.metric)

    present = {(cell.identity.source_snapshot_id, cell.identity.key) for cell in cells}
    missing_roles: set[str] = set()
    decision_populations: set[tuple[str | None, str]] = set()
    for source_id in source_ids:
        source_scope = metadata.scope.by_source.get(source_id)
        source_keys = (
            source_scope.decision_cells
            if source_scope is not None
            else tuple(
                cell.identity.key for cell in cells if cell.identity.source_snapshot_id == source_id
            )
        )
        relevant_keys = tuple(
            key
            for key in source_keys
            if key.method_role == "decision"
            and key.estimand != "compliance"
            and key.metric in decision_roles
        )
        if not relevant_keys:
            relevant_keys = tuple(
                cell.identity.key
                for cell in cells
                if cell.identity.source_snapshot_id == source_id
                and cell.identity.key.method_role == "decision"
                and cell.identity.key.estimand != "compliance"
                and cell.identity.key.metric in decision_roles
            )
        decision_populations.update((source_id, key.analysis_population) for key in relevant_keys)
        if source_scope is not None:
            for key in source_scope.decision_cells:
                if (
                    key.method_role != "decision"
                    or key.estimand == "compliance"
                    or (source_id, key) in present
                ):
                    continue
                role = decision_roles.get(key.metric)
                if role is not None:
                    missing_roles.add(role)

    for source_id, population in decision_populations:
        actual_metrics = cells_by_source_population.get((source_id, population), set())
        for metric, role in decision_roles.items():
            if metric not in actual_metrics:
                missing_roles.add(role)
    return missing_roles


def _decision_integrity_reasons(
    metadata: Any,
    cells: tuple[Any, ...],
    verdicts: Mapping[tuple[str | None, str], Verdict],
) -> tuple[str, ...]:
    decision_sources = set()
    for cell in cells:
        key = cell.identity.key
        scope = cell.identity.scope
        if (
            key.method_role != "decision"
            or key.estimand == "compliance"
            or (scope is not None and scope.view not in {"run", "switchback"})
        ):
            continue
        verdict = verdicts.get((cell.identity.source_snapshot_id, cell.identity.hash()))
        if verdict is not None and verdict.role in {"primary", "guardrail"}:
            decision_sources.add(cell.identity.source_snapshot_id)
    return tuple(
        sorted(
            {
                reason
                for source_id in decision_sources
                for reason in _integrity_by_scope(metadata, source_id)
            }
        )
    )


def decide(  # noqa: PLR0915
    handle: RunHandlePort, registration: Registration, plan: object | None = None
) -> Decisions:
    """Evaluate only the immutable evidence and manifest pinned by a RunHandle."""
    run = handle.run
    if run.registration_hash != registration.registration_hash:
        refuse_store(
            "store.decision.reference_invalid",
            "Run registration differs from decision registration",
            run_id=run.run_id,
        )
    plan = _validated_plan(run, registration, plan)
    envelope = _validated_envelope(run, handle)

    verdicts: dict[tuple[str | None, str], Verdict] = {}
    for cell in envelope.cells:
        verdict = _scope_verdict(cell, plan.procedures)
        if verdict is not None:
            verdicts[(cell.identity.source_snapshot_id, cell.identity.hash())] = verdict
            continue
        key = cell.identity.key
        procedure = plan.procedures[key.metric]
        special = _special_verdict(cell, plan.procedures)
        if special is not None:
            verdicts[(cell.identity.source_snapshot_id, cell.identity.hash())] = special
            continue

        status, retained = cell.components["sampling"]
        row = cell.retained_row() if retained is not None else None
        declared_role = getattr(procedure.role, "value", procedure.role)
        role = (
            declared_role
            if row is None or row.role is None
            else getattr(row.role, "value", row.role)
        )
        stat = None if row is None or status.state == "unavailable" else row.stat_sig()
        reasons: list[str] = []
        if status.state == "unavailable":
            reasons.append(status.code or "results.component.unavailable")
        if stat is None:
            result = "not_evaluated"
            if not reasons:
                reasons.append("decision.evidence.unavailable")
        else:
            result = "reject" if stat else "retain"
        family_id = None if cell.family is None else cell.family.family_id
        verdicts[(cell.identity.source_snapshot_id, cell.identity.hash())] = Verdict(
            cell.identity.hash(),
            None if role is None else str(role),
            result,
            None if row is None else row.discovery,
            tuple(reasons),
            family_id,
            "nonmember" if family_id is None else "member",
        )
    metadata = envelope.metadata
    families: list[FamilyVerdict] = []
    if metadata is not None and metadata.scope is not None:
        ordered_families = sorted(
            metadata.scope.families,
            key=lambda family: (
                family.source_snapshot_id,
                str(family.analysis_population),
                family.family_id,
            ),
        )
        for family in ordered_families:
            members: list[Verdict] = []
            for key in family.members:
                match = next(
                    (
                        cell
                        for cell in envelope.cells
                        if cell.identity.source_snapshot_id == family.source_snapshot_id
                        and cell.identity.key == key
                    ),
                    None,
                )
                if match is None:
                    members.append(
                        Verdict(
                            "",
                            None,
                            "not_evaluated",
                            None,
                            ("decision.family.member_missing",),
                            family.family_id,
                            "member",
                        )
                    )
                else:
                    existing = verdicts[(match.identity.source_snapshot_id, match.identity.hash())]
                    members.append(
                        Verdict(
                            existing.identity_hash,
                            existing.role,
                            existing.status,
                            existing.discovery,
                            existing.reason_codes,
                            family.family_id,
                            "member",
                        )
                    )
            procedure = "none" if family.family is None else family.family.correction
            if procedure in {"bh", "e_bh"}:
                selection_evidence = ()
                if family.family is None or family.family.q is None:
                    selection_error = "decision.family.level_missing"
                    selected_hashes = None
                else:
                    matching_cells = [
                        next(
                            (
                                cell
                                for cell in envelope.cells
                                if cell.identity.source_snapshot_id == family.source_snapshot_id
                                and cell.identity.key == key
                            ),
                            None,
                        )
                        for key in family.members
                    ]
                    if any(cell is None for cell in matching_cells):
                        selection_error = "decision.family.member_missing"
                        selected_hashes = None
                    else:
                        try:
                            selection_evidence = tuple(
                                family_evidence(cell) for cell in matching_cells if cell is not None
                            )
                            selected_hashes = combine_family_evidence(
                                selection_evidence,
                                procedure=procedure,
                                level=family.family.q,
                            )
                            selection_error = None
                        except CodedError as exc:
                            selected_hashes = None
                            selection_error = exc.code
                if selection_error is None and selected_hashes is not None:
                    members = [
                        replace(
                            member,
                            discovery=member.identity_hash in selected_hashes,
                        )
                        for member in members
                    ]
                else:
                    failure_codes = {
                        item.identity_hash: item.reason_code
                        for item in selection_evidence
                        if item.evidence is None and item.reason_code is not None
                    }
                    members = [
                        replace(
                            member,
                            status="not_evaluated",
                            reason_codes=tuple(
                                sorted(
                                    {
                                        *member.reason_codes,
                                        selection_error or "decision.family.unavailable",
                                        *(
                                            code
                                            for code in (failure_codes.get(member.identity_hash),)
                                            if code is not None
                                        ),
                                    }
                                )
                            ),
                        )
                        for member in members
                    ]
            elif procedure == "bonferroni":
                members = [
                    replace(member, discovery=member.status == "reject") for member in members
                ]
            family_reasons = tuple(
                sorted({code for member in members for code in member.reason_codes})
            )
            state = (
                "not_evaluated"
                if any(member.status == "not_evaluated" for member in members)
                else "reject"
                if (
                    any(member.status == "reject" for member in members)
                    if procedure in {"none", "bonferroni"}
                    else any(member.discovery is True for member in members)
                )
                else "retain"
            )
            families.append(
                FamilyVerdict(
                    family.family_id,
                    family.source_snapshot_id,
                    str(family.analysis_population),
                    procedure,
                    tuple(members),
                    state,
                    family_reasons,
                )
            )

    roles = {
        str(getattr(procedure.role, "value", procedure.role))
        for procedure in plan.procedures.values()
    }
    decision_hashes = {
        cell.identity.hash()
        for cell in envelope.cells
        if cell.identity.key.method_role == "decision"
        and cell.identity.key.estimand != "compliance"
        and (cell.identity.scope is None or cell.identity.scope.view in {"run", "switchback"})
    }
    primary = [
        verdict
        for verdict in verdicts.values()
        if verdict.role == "primary" and verdict.identity_hash in decision_hashes
    ]
    guardrails = [
        verdict
        for verdict in verdicts.values()
        if verdict.role == "guardrail" and verdict.identity_hash in decision_hashes
    ]
    missing_roles = _missing_decision_roles(metadata, envelope.cells, plan.procedures)
    integrity_reasons = _decision_integrity_reasons(metadata, envelope.cells, verdicts)
    reasons = list(integrity_reasons)
    if (
        "primary" not in roles
        or not primary
        or "primary" in missing_roles
        or any(v.status == "not_evaluated" for v in primary)
    ):
        program = ProgramVerdict(
            "not_evaluated", tuple(sorted(set(reasons + ["decision.primary.not_evaluated"])))
        )
    elif any(v.status != "reject" for v in primary):
        program = ProgramVerdict(
            "block", tuple(sorted(set(reasons + ["decision.primary.not_rejected"])))
        )
    elif "guardrail" in roles and (
        not guardrails
        or "guardrail" in missing_roles
        or any(v.status == "not_evaluated" for v in guardrails)
    ):
        program = ProgramVerdict(
            "not_evaluated", tuple(sorted(set(reasons + ["decision.guardrail.not_evaluated"])))
        )
    elif any(v.status != "reject" for v in guardrails):
        program = ProgramVerdict(
            "block", tuple(sorted(set(reasons + ["decision.guardrail.unsafe"])))
        )
    elif reasons:
        program = ProgramVerdict("block", tuple(sorted(set(reasons))))
    else:
        program = ProgramVerdict("pass", ())
    return Decisions(
        tuple(verdicts.values()),
        tuple(families),
        program,
        registration.registration_hash,
        run.run_id,
    )


__all__ = ["Decisions", "FamilyVerdict", "ProgramVerdict", "Verdict", "decide"]
