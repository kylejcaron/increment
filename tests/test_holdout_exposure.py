"""Audit recorded access without selecting, regrouping or estimating outcomes."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from increment._holdout_exposure import audit_rollout
from increment._holdout_records import (
    ArmCounts,
    Coverage,
    DecisionEvent,
    Delivery,
    Group,
    HoldoutSnapshot,
    HoldoutStudy,
    HoldoutWindow,
    IdentitySpan,
    RolloutAudit,
    RosterEvidence,
    RosterMember,
)

type Shopping = tuple[HoldoutStudy, HoldoutWindow, HoldoutSnapshot, RosterEvidence]


def _day(month: int, day: int) -> datetime:
    return datetime(2025, month, day, tzinfo=UTC)


@pytest.fixture(scope="module")
def shopping() -> Shopping:
    records = json.loads((Path(__file__).parent / "fixtures/holdouts/shopping_v1.json").read_text())
    jan2, mar5, mar18 = _day(1, 2), _day(3, 5), _day(3, 18)
    return (
        HoldoutStudy.model_validate(records["study"]),
        HoldoutWindow.model_validate(records["window"]),
        HoldoutSnapshot.model_validate(records["snapshot"]),
        RosterEvidence(
            members=(
                RosterMember("acct:1", jan2, jan2, "eligible", "r1"),
                RosterMember("acct:2", jan2, jan2, "held_out", "r1"),
                RosterMember("acct:3", jan2, jan2, "eligible", "r1"),
                RosterMember("acct:4", mar5, mar5, "eligible", "r1"),
                RosterMember("acct:15", jan2, jan2, "held_out", "r1"),
                RosterMember("acct:16", jan2, jan2, "held_out", "r1"),
            ),
            identity=(
                IdentitySpan(unit_id="acct:1", original_id="acct:1", from_at=jan2),
                IdentitySpan(unit_id="acct:2", original_id="acct:2", from_at=jan2, to_at=mar18),
                IdentitySpan(unit_id="acct:2-new", original_id="acct:2", from_at=mar18),
                IdentitySpan(unit_id="acct:3", original_id="acct:3", from_at=jan2),
                IdentitySpan(unit_id="acct:4", original_id="acct:4", from_at=mar5),
                IdentitySpan(unit_id="acct:15", original_id="acct:15", from_at=jan2),
                IdentitySpan(unit_id="acct:16", original_id="acct:16", from_at=jan2),
            ),
            values=(
                ("acct:1", 10),
                ("acct:2", 8),
                ("acct:3", 12),
                ("acct:4", 14),
                ("acct:15", 10),
                ("acct:16", 12),
            ),
            held_out=ArmCounts(assigned=3, measured=3),
            eligible=ArmCounts(assigned=3, measured=3),
            n_eligible_entries=6,
            n_ineligible_entries=0,
            n_unclassified=0,
            n_after_freeze=1,
        ),
    )


def _delivery(
    event_id: str,
    unit_id: str,
    at: datetime,
    *,
    version: str = "A/2",
    revision: str = "A/ramp/2",
    target: str = "shopping-a",
    available_at: datetime | None = None,
) -> Delivery:
    return Delivery(
        event_id=event_id,
        unit_id=unit_id,
        launch_id="A",
        target=target,
        feature_version=version,
        rollout_revision=revision,
        at=at,
        available_at=at if available_at is None else available_at,
        nonbaseline=True,
    )


def _decision(
    event_id: str,
    unit_id: str,
    at: datetime,
    *,
    revision: str = "r1",
    blocked: bool = False,
    offered: bool = False,
) -> DecisionEvent:
    return DecisionEvent(
        event_id=event_id,
        unit_id=unit_id,
        launch_id="A",
        target="shopping-a",
        config_revision=revision,
        at=at,
        available_at=at,
        holdout_blocked=blocked,
        nonbaseline_assigned=offered,
    )


def _coverage(group: Group, start: datetime, stop: datetime, *, launch_id: str = "A") -> Coverage:
    return Coverage(
        kind="delivery",
        group=group,
        launch_id=launch_id,
        from_at=start,
        to_at=stop,
        source_ref=f"coverage/{launch_id}/{group}",
        available_at=stop,
    )


def _unmatched(audit: RolloutAudit) -> dict[str, str]:
    return {event.event_id: event.reason for event in audit.unmatched_events}


def test_leak_and_unobserved_access_leave_both_groups_intact(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    start, end, edge = _day(1, 2), _day(4, 10), _day(3, 20)
    snapshot = snapshot.model_copy(
        update={
            "coverage": (
                _coverage("held_out", start, end),
                _coverage("eligible", start, end),
            ),
            "deliveries": (
                _delivery("leak", "acct:2-new", edge),
                _delivery("access", "acct:1", edge),
            ),
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    rows = {
        (row.launch_id, row.group): (
            row.n_assigned,
            row.n_units_recorded,
            row.n_nonbaseline,
            row.n_violating,
            row.coverage_status,
            row.finding,
        )
        for row in audit.findings
        if row.kind == "delivery"
    }
    assert rows == {
        ("A", "held_out"): (3, 1, 1, 1, "covered", "recorded_violation"),
        ("A", "eligible"): (3, 1, 1, 0, "covered", "none_recorded"),
        ("B", "held_out"): (3, 0, 0, 0, "unknown", "unknown"),
        ("B", "eligible"): (3, 0, 0, 0, "unknown", "unknown"),
    }
    assert audit.unmatched_events == ()


def test_coverage_is_independent_of_events_and_keeps_known_leaks(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    start, end, edge = _day(1, 2), _day(4, 10), _day(3, 20)
    leak = _delivery("leak", "acct:2-new", edge)
    snapshot = snapshot.model_copy(
        update={
            "coverage": (
                _coverage("held_out", edge, end),
                _coverage("eligible", start, edge, launch_id="B"),
                _coverage("eligible", edge, end, launch_id="B"),
                _coverage("held_out", start, end, launch_id="B").model_copy(
                    update={"available_at": _day(4, 13)}
                ),
            ),
            "deliveries": (
                leak,
                leak,
                _delivery("another-observation", "acct:2-new", _day(3, 21)),
            ),
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    assert {
        (row.launch_id, row.group): (row.coverage_status, row.finding, row.n_violating)
        for row in audit.findings
        if row.kind == "delivery"
    } == {
        ("A", "eligible"): ("unknown", "unknown", 0),
        ("A", "held_out"): ("partial", "recorded_violation", 1),
        ("B", "eligible"): ("covered", "none_recorded", 0),
        ("B", "held_out"): ("unknown", "unknown", 0),
    }
    assert audit.reasons == ()


def test_assignment_enforcement_does_not_infer_delivery(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    edge = _day(3, 20)
    snapshot = snapshot.model_copy(
        update={
            "decisions": (
                _decision("allowed-through", "acct:2-new", edge, offered=True),
                _decision("explicit-block", "acct:1", edge, blocked=True),
                _decision("no-feature", "acct:3", edge),
            )
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    assert {
        (row.group, row.kind): (row.n_units_recorded, row.n_nonbaseline, row.n_violating)
        for row in audit.findings
        if row.launch_id == "A"
    } == {
        ("held_out", "decision"): (1, 1, 1),
        ("eligible", "decision"): (2, 0, 1),
        ("held_out", "delivery"): (0, 0, 0),
        ("eligible", "delivery"): (0, 0, 0),
    }


def test_before_assignment_offers_invalidate_entry_without_counting_as_leaks(
    shopping: Shopping,
) -> None:
    study, window, snapshot, roster = shopping
    pre_entry, entry = _day(3, 4), _day(3, 5)
    snapshot = snapshot.model_copy(
        update={
            "decisions": (
                _decision("early-offer", "acct:4", pre_entry, offered=True),
                _decision("at-assignment", "acct:4", entry, offered=True),
            ),
            "deliveries": (
                _delivery(
                    "early-delivery", "acct:4", pre_entry, version="A/1", revision="A/ramp/1"
                ),
            ),
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    assert [(reason.code, reason.context) for reason in audit.reasons] == [
        ("holdout.exposure_before_assignment", {"study_id": "shopping-25", "launch_id": "A"})
    ]
    assert _unmatched(audit) == {
        "early-offer": "before_assignment",
        "early-delivery": "before_assignment",
    }
    assert [
        (row.kind, row.n_assigned, row.n_units_recorded, row.n_violating)
        for row in audit.findings
        if row.launch_id == "A" and row.group == "eligible"
    ] == [("decision", 3, 1, 0), ("delivery", 3, 0, 0)]


def test_events_respect_scope_identity_and_half_open_intervals(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    alias_edge, phase_edge, end = _day(3, 18), _day(3, 20), _day(4, 10)
    snapshot = snapshot.model_copy(
        update={
            "deliveries": (
                _delivery("other-target", "acct:2-new", phase_edge, target="other"),
                _delivery("outsider", "acct:42", phase_edge),
                _delivery("stale-id", "acct:2", alias_edge, version="A/1", revision="A/ramp/1"),
                _delivery("off", "acct:2", _day(1, 2), version="A/1", revision="A/ramp/1"),
                _delivery(
                    "old-version", "acct:2-new", phase_edge, version="A/1", revision="A/ramp/1"
                ),
                _delivery("after-window", "acct:15", end),
                _delivery("late-report", "acct:16", phase_edge, available_at=_day(4, 13)),
                _delivery("impossible-clock", "acct:16", phase_edge, available_at=_day(3, 19)),
            )
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    assert _unmatched(audit) == {
        "other-target": "outside_scope",
        "outsider": "unknown_unit",
        "stale-id": "identity_not_effective",
        "off": "out_of_phase",
        "old-version": "version_mismatch",
        "after-window": "outside_interval",
        "impossible-clock": "invalid_availability",
    }
    held_out = next(
        row
        for row in audit.findings
        if (row.launch_id, row.group, row.kind) == ("A", "held_out", "delivery")
    )
    assert (held_out.n_assigned, held_out.n_violating) == (3, 1)


def test_a_live_cut_excludes_current_edge_rows_and_later_attestations(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    edge = _day(3, 20)
    snapshot = snapshot.model_copy(
        update={
            "as_of": edge,
            "deliveries": (_delivery("at-cut", "acct:2-new", edge),),
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    assert [phase.rollout_revision for phase in audit.phases] == ["A/off", "A/ramp/1", "B/off"]
    assert [(reason.code, reason.context) for reason in audit.reasons] == [
        ("holdout.launch_history_unresolved", {"study_id": "shopping-25", "launch_id": "B"})
    ]
    assert _unmatched(audit) == {"at-cut": "outside_interval"}


@pytest.mark.parametrize(
    ("activation", "inactive_event"),
    [("covers", None), ("starts-late", "at-phase-edge"), ("ends-early", "at-config-end")],
)
def test_named_config_revision_must_cover_the_actual_phase(
    shopping: Shopping, activation: str, inactive_event: str | None
) -> None:
    study, window, snapshot, roster = shopping
    edge, early_end = _day(3, 20), _day(3, 31)
    r1 = snapshot.configs[0]
    assert r1.sdk_json is not None
    r2 = r1.model_copy(
        update={
            "revision": "r2",
            "archive_ref": "archives/shopping-sdk/r2",
            "sdk_json": r1.sdk_json.replace('"revision":"r1"', '"revision":"r2"'),
            "effective_from": _day(3, 21) if activation == "starts-late" else edge,
            "effective_to": early_end if activation == "ends-early" else r1.effective_to,
        }
    )
    phases = list(snapshot.phases)
    phases[2] = phases[2].model_copy(update={"config_revision": "r2"})
    snapshot = snapshot.model_copy(
        update={
            "phases": tuple(phases),
            "configs": (r1, r2),
            "decisions": (
                _decision("prior-revision-at-edge", "acct:3", edge),
                _decision("at-phase-edge", "acct:1", edge, revision="r2"),
                _decision("at-config-end", "acct:1", early_end, revision="r2"),
            ),
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    if inactive_event is not None:
        assert [(reason.code, reason.context) for reason in audit.reasons] == [
            ("holdout.launch_history_unresolved", {"study_id": "shopping-25", "launch_id": "A"})
        ]
        assert _unmatched(audit)[inactive_event] == "inactive_config"
    else:
        assert audit.reasons == ()
        assert _unmatched(audit) == {"prior-revision-at-edge": "config_mismatch"}


@pytest.mark.parametrize("change", ["gap", "overlap", "new-target", "no-history-reference"])
def test_missing_or_contradictory_actual_history_is_not_a_coverage_finding(
    shopping: Shopping, change: str
) -> None:
    study, window, snapshot, roster = shopping
    phases = list(snapshot.phases)
    update: dict[str, object]
    if change == "no-history-reference":
        update = {"history_ref": None}
    else:
        changes = {
            "gap": {"from_at": _day(3, 21)},
            "overlap": {"from_at": _day(3, 19)},
            "new-target": {"target": "shopping-c"},
        }
        phases[2] = phases[2].model_copy(update=changes[change])
        update = {"phases": tuple(phases)}

    audit = audit_rollout(study, window, snapshot.model_copy(update=update), roster)

    launches = ("A", "B") if change == "no-history-reference" else ("A",)
    assert [(reason.code, reason.context) for reason in audit.reasons] == [
        ("holdout.launch_history_unresolved", {"study_id": "shopping-25", "launch_id": launch_id})
        for launch_id in launches
    ]
    assert all(finding.coverage_status == "unknown" for finding in audit.findings)


def test_duplicate_event_id_with_conflicting_content_is_not_evidence(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    edge = _day(3, 20)
    snapshot = snapshot.model_copy(
        update={
            "deliveries": (
                _delivery("reused", "acct:2-new", edge),
                _delivery("reused", "acct:1", edge),
            )
        }
    )

    audit = audit_rollout(study, window, snapshot, roster)

    assert [(reason.code, reason.context) for reason in audit.reasons] == [
        (
            "holdout.event_conflict",
            {"study_id": "shopping-25", "stream": "deliveries", "event_id": "reused"},
        )
    ]
    assert all(finding.n_units_recorded == 0 for finding in audit.findings)


def test_history_digest_binds_ordered_versions_edges_and_sources(shopping: Shopping) -> None:
    study, window, snapshot, roster = shopping
    reordered = snapshot.model_copy(
        update={"phases": (*reversed(snapshot.phases), snapshot.phases[0])}
    )
    changed_phases = (
        snapshot.phases[0],
        snapshot.phases[1].model_copy(update={"to_at": _day(3, 21)}),
        snapshot.phases[2].model_copy(update={"from_at": _day(3, 21)}),
        *snapshot.phases[3:],
    )

    audit = audit_rollout(study, window, reordered, roster)
    changed = audit_rollout(
        study, window, snapshot.model_copy(update={"phases": changed_phases}), roster
    )

    assert audit.phases == snapshot.phases
    assert audit.history_ref == "exports/shopping-25/rollout-history/2025-04-12"
    assert audit.history_digest == (
        "sha256:7085d57ec08b67e0e08ea0db9070e04501f2eccf7885ef9c74b858cd16a55f51"
    )
    assert changed.history_digest != audit.history_digest
