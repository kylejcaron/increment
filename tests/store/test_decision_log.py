from datetime import UTC, datetime

import pytest

from increment.errors import CodedError
from increment.store import DecisionEvent, DecisionLog


def test_decision_log_preserves_append_order_with_equal_timestamps():
    at = datetime(2026, 1, 1, tzinfo=UTC)
    log = DecisionLog()
    first = DecisionEvent(
        experiment_id="exp",
        family_id="f",
        registration_hash="reg",
        decided_at=at,
        rule="bonferroni",
        threshold=0.05,
        alpha_spent=0.05,
        references=(),
        evidence_digest="sha256:" + "0" * 64,
        integrity_status="complete",
        outcome="reject",
        reason_codes=(),
    )
    second = DecisionEvent(
        experiment_id="exp",
        family_id="f",
        registration_hash="reg",
        decided_at=at,
        rule="bonferroni",
        threshold=0.05,
        alpha_spent=0.05,
        references=(),
        evidence_digest="sha256:" + "1" * 64,
        integrity_status="complete",
        outcome="retain",
        reason_codes=(),
    )
    log.append(first)
    log.append(second)
    assert log.events == (first, second)


def test_decision_log_reads_persisted_events_in_append_order(tmp_path):
    event = DecisionEvent(
        experiment_id="exp",
        family_id="f",
        registration_hash="reg",
        decided_at=datetime(2026, 1, 1, tzinfo=UTC),
        rule="bonferroni",
        threshold=0.05,
        alpha_spent=0.05,
        references=(),
        evidence_digest="sha256:" + "a" * 64,
        integrity_status="complete",
        outcome="reject",
        reason_codes=(),
    )
    path = tmp_path / "events.jsonl"
    log = DecisionLog(path=path)
    log.append(event)
    restored = DecisionLog(path=path)
    assert restored.events == (event,)
    assert not hasattr(restored, "replay")


def test_decision_event_requires_a_full_versioned_evidence_digest():
    with pytest.raises(CodedError) as error:
        DecisionEvent(
            experiment_id="exp",
            family_id="f",
            registration_hash="reg",
            decided_at=datetime(2026, 1, 1, tzinfo=UTC),
            rule="bonferroni",
            threshold=0.05,
            alpha_spent=0.05,
            references=(),
            evidence_digest="sha256:a",
            integrity_status="complete",
            outcome="reject",
            reason_codes=(),
        )
    assert error.value.code == "store.decision.reference_invalid"
