from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, TypedDict, cast

from increment.kernel.contracts import CellIdentity
from increment.store._refusals import refuse_store


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
        "schema_version": 1,
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
    }


def _decode_event(payload: dict[str, object]) -> DecisionEvent:
    if payload.get("schema_version") != 1:
        refuse_store(
            "store.decision.schema_unsupported",
            "Unsupported decision event schema",
            schema_version=payload.get("schema_version"),
        )
    event = cast(_DecisionPayload, payload)
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
    )
