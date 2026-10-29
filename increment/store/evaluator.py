from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from increment.errors import CodedError
from increment.kernel.contracts import CompiledPlanBinding, Registration
from increment.result_envelope import combine_family_evidence, family_evidence
from increment.store._refusals import refuse_store
from increment.store.types import RunHandle


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


def decide(handle: RunHandle, registration: Registration, plan: object | None = None) -> Decisions:  # noqa: C901, PLR0915
    """Evaluate only the immutable evidence and manifest pinned by a RunHandle."""
    run = handle.run
    if run.registration_hash != registration.registration_hash:
        refuse_store(
            "store.decision.reference_invalid",
            "Run registration differs from decision registration",
            run_id=run.run_id,
        )
    if run.decision_plan is None:
        refuse_store(
            "store.decision.reference_invalid",
            "Run has no bound compiled decision plan",
            run_id=run.run_id,
        )
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
    plan = bound_plan
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

    verdicts: dict[tuple[str | None, str], Verdict] = {}
    for cell in envelope.cells:
        status, retained = cell.components["sampling"]
        row = cell.retained_row() if retained is not None else None
        procedure = plan.procedures.get(cell.identity.key.metric)
        declared_role = (
            None if procedure is None else getattr(procedure.role, "value", procedure.role)
        )
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
    missing_roles: set[str] = set()
    if metadata is not None and metadata.scope is not None:
        present = {(cell.identity.source_snapshot_id, cell.identity.key) for cell in envelope.cells}
        for source_id, source_scope in metadata.scope.by_source.items():
            for key in source_scope.decision_cells:
                if key.method_role != "decision" or (source_id, key) in present:
                    continue
                procedure = plan.procedures.get(key.metric)
                role = (
                    None
                    if procedure is None
                    else str(getattr(procedure.role, "value", procedure.role))
                )
                if role in {"primary", "guardrail"}:
                    missing_roles.add(role)
    integrity_reasons = tuple(
        sorted(
            {
                reason
                for cell in envelope.cells
                for reason in _integrity_by_scope(metadata, cell.identity.source_snapshot_id)
            }
        )
    )
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
