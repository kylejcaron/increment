from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, NotRequired, Protocol, TypedDict, cast

from increment.kernel.contracts import CellIdentity, CompiledPlanBinding, Registration
from increment.kernel.program_refusals import refuse_store
from increment.store.types import RunHandle, run_digest


class _DecisionReferencePayload(TypedDict):
    identity: dict[str, object]
    run_id: str


class _DecisionPayload(TypedDict):
    schema_version: int
    experiment_id: str
    family_id: str
    registration_hash: str
    decided_at: str
    rule: Literal["bh", "ebh", "bonferroni"]
    threshold: float
    alpha_spent: float
    references: list[_DecisionReferencePayload]
    evidence_digest: str
    integrity_status: str
    outcome: str
    reason_codes: list[str]
    decision_plan: NotRequired[dict[str, object] | None]


@dataclass(frozen=True, slots=True)
class DecisionReference:
    identity: CellIdentity
    run_id: str


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    experiment_id: str
    family_id: str
    registration_hash: str
    decided_at: datetime
    rule: Literal["bh", "ebh", "bonferroni"]
    threshold: float
    alpha_spent: float
    references: tuple[DecisionReference, ...]
    evidence_digest: str
    integrity_status: str
    outcome: str
    reason_codes: tuple[str, ...]
    decision_plan: CompiledPlanBinding | None = None

    def __post_init__(self) -> None:
        if self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None:
            refuse_store(
                "store.decision.timestamp_invalid", "Decision time must include a timezone"
            )
        if (
            not self.experiment_id
            or not self.registration_hash
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.evidence_digest) is None
            or self.rule not in {"bh", "ebh", "bonferroni"}
            or any(not reference.run_id for reference in self.references)
        ):
            refuse_store(
                "store.decision.reference_invalid",
                "Decision event references are incomplete or unsupported",
            )
        if self.decision_plan is not None and (
            self.decision_plan.registration_hash != self.registration_hash
        ):
            refuse_store(
                "store.decision.reference_invalid",
                "Event plan differs from its registration hash",
            )


class _DecisionEvaluator(Protocol):
    def __call__(self, event: DecisionEvent) -> tuple[str, tuple[str, ...]]: ...


@dataclass(slots=True)
class DecisionLog:
    path: Path | None = None
    _events: list[DecisionEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.path is not None and self.path.exists():
            try:
                self._events.extend(
                    _decode_event(json.loads(line)) for line in self.path.read_text().splitlines()
                )
            except (ValueError, KeyError, TypeError):
                refuse_store("store.decision.log_corrupt", "Decision log contains an invalid event")

    @property
    def events(self) -> tuple[DecisionEvent, ...]:
        return tuple(self._events)

    def replay(
        self,
        resolve: Callable[[str, str], RunHandle],
        registration: Registration,
        plan: object | None = None,
    ) -> tuple[object, ...]:
        """Resolve pinned runs and re-evaluate their decisions through the shared evaluator."""
        if plan is not None and not registration.matches(plan):
            refuse_store(
                "store.decision.reference_invalid",
                "Compiled plan differs from registration during replay",
            )
        from increment.decide.evaluator import decide

        results: list[object] = []
        for event in self.events:
            if event.registration_hash != registration.registration_hash:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Decision event registration differs from replay registration",
                    family_id=event.family_id,
                )
            run_ids = {reference.run_id for reference in event.references}
            if len(run_ids) != 1 or not event.references:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Decision event must pin one complete run",
                    family_id=event.family_id,
                )
            run_id = next(iter(run_ids))
            handle = resolve(event.experiment_id, run_id)
            run = handle.run
            if event.evidence_digest != run_digest(run):
                refuse_store(
                    "store.decision.reference_invalid",
                    "Decision event evidence digest differs from the resolved run",
                    family_id=event.family_id,
                )
            if event.decision_plan is None:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Decision event does not contain a bound compiled plan",
                    family_id=event.family_id,
                )
            event_plan = event.decision_plan.validate(registration)
            if plan is not None and (
                CompiledPlanBinding.from_plan(registration, plan).canonical_payload()
                != event.decision_plan.canonical_payload()
            ):
                refuse_store(
                    "store.decision.reference_invalid",
                    "Replay plan differs from the event's bound plan",
                    family_id=event.family_id,
                )
            if run.decision_plan != event.decision_plan:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Resolved run plan differs from the event's bound plan",
                    family_id=event.family_id,
                )
            decisions = decide(handle, registration, event_plan)
            if run.run_id != run_id or run.experiment_id != event.experiment_id:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Resolved run differs from the event's pinned identity",
                    family_id=event.family_id,
                )
            for reference in event.references:
                if not any(cell.identity.hash() == reference.identity.hash() for cell in run.cells):
                    refuse_store(
                        "store.decision.reference_invalid",
                        "Pinned cell is absent from the resolved run",
                        family_id=event.family_id,
                    )
            family = next(
                (item for item in decisions.families if item.family_id == event.family_id),
                None,
            )
            if family is None:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Recorded family is absent from the resolved run",
                    family_id=event.family_id,
                )
            scope = getattr(run.metadata, "scope", None)
            scope_family = (
                None
                if scope is None
                else next(
                    (
                        item
                        for item in scope.families
                        if item.family_id == event.family_id
                        and item.source_snapshot_id == family.source_snapshot_id
                        and str(item.analysis_population) == family.analysis_population
                    ),
                    None,
                )
            )
            if scope is None or scope_family is None:
                refuse_store(
                    "store.decision.reference_invalid",
                    "Recorded family scope is absent from the resolved run",
                    family_id=event.family_id,
                )
            expected_hashes = {
                cell.identity.hash()
                for cell in run.cells
                if cell.identity.source_snapshot_id == family.source_snapshot_id
                and cell.identity.key in scope_family.members
            }
            reference_hashes = tuple(reference.identity.hash() for reference in event.references)
            if (
                len(reference_hashes) != len(expected_hashes)
                or set(reference_hashes) != expected_hashes
            ):
                refuse_store(
                    "store.decision.reference_invalid",
                    "Decision references differ from the recorded family roster",
                    family_id=event.family_id,
                )
            expected_rule = {
                "bh": "bh",
                "e_bh": "ebh",
                "bonferroni": "bonferroni",
            }.get(family.procedure)
            family_q = None if scope_family.family is None else scope_family.family.q
            expected_alpha = 0.05 if family_q is None else family_q
            source = scope.by_source.get(family.source_snapshot_id)
            expected_integrity = (
                "unknown"
                if source is None
                else ",".join(
                    sorted(
                        {
                            item.status.value
                            for item in source.integrity
                            if str(item.analysis_population) == "assigned"
                        }
                    )
                )
                or "unknown"
            )
            if (
                expected_rule != event.rule
                or event.threshold != expected_alpha
                or event.alpha_spent != expected_alpha
                or event.integrity_status != expected_integrity
            ):
                refuse_store(
                    "store.decision.reference_invalid",
                    "Decision rule, allocation, or integrity differs from the resolved run",
                    family_id=event.family_id,
                )
            if (family.status, family.reason_codes) != (event.outcome, event.reason_codes):
                refuse_store(
                    "store.decision.outcome_mismatch",
                    "Replayed decision differs from the recorded outcome",
                    family_id=event.family_id,
                )
            results.append(decisions)
        return tuple(results)

    def append(self, event: DecisionEvent) -> None:
        if not isinstance(event, DecisionEvent):
            refuse_store("store.decision.event_invalid", "Decision log accepts immutable events")
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(_encode_event(event), sort_keys=True, separators=(",", ":")) + "\n"
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, payload.encode())
            finally:
                os.close(fd)
        self._events.append(event)

    def _verify(self, evaluator: _DecisionEvaluator) -> None:
        for event in self.events:
            outcome, reasons = evaluator(event)
            if (outcome, tuple(reasons)) != (event.outcome, event.reason_codes):
                refuse_store(
                    "store.decision.outcome_mismatch",
                    "Replayed decision differs from the recorded outcome",
                    family_id=event.family_id,
                )


def _encode_event(event: DecisionEvent) -> dict[str, object]:
    return {
        "schema_version": 2 if event.decision_plan is not None else 1,
        "experiment_id": event.experiment_id,
        "family_id": event.family_id,
        "registration_hash": event.registration_hash,
        "decided_at": event.decided_at.isoformat(),
        "rule": event.rule,
        "threshold": event.threshold,
        "alpha_spent": event.alpha_spent,
        "references": [
            {"identity": ref.identity.canonical_payload(), "run_id": ref.run_id}
            for ref in event.references
        ],
        "evidence_digest": event.evidence_digest,
        "integrity_status": event.integrity_status,
        "outcome": event.outcome,
        "reason_codes": list(event.reason_codes),
        **(
            {"decision_plan": event.decision_plan.to_payload()}
            if event.decision_plan is not None
            else {}
        ),
    }


def _decode_event(payload: dict[str, object]) -> DecisionEvent:
    if payload.get("schema_version") not in {1, 2}:
        refuse_store(
            "store.decision.schema_unsupported",
            "Unsupported decision event schema",
            schema_version=payload.get("schema_version"),
        )
    event = cast(_DecisionPayload, payload)
    raw_binding = event.get("decision_plan")
    return DecisionEvent(
        experiment_id=event["experiment_id"],
        family_id=event["family_id"],
        registration_hash=event["registration_hash"],
        decided_at=datetime.fromisoformat(event["decided_at"]),
        rule=event["rule"],
        threshold=event["threshold"],
        alpha_spent=event["alpha_spent"],
        references=tuple(
            DecisionReference(CellIdentity.from_payload(ref["identity"]), ref["run_id"])
            for ref in event["references"]
        ),
        evidence_digest=event["evidence_digest"],
        integrity_status=event["integrity_status"],
        outcome=event["outcome"],
        reason_codes=tuple(event["reason_codes"]),
        decision_plan=(
            None if raw_binding is None else CompiledPlanBinding.from_payload(raw_binding)
        ),
    )
