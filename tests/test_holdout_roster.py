"""Keep the original common-entry cohort separate from serving and outcomes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from increment._holdout_records import ArmCounts, HoldoutSnapshot, HoldoutStudy, HoldoutWindow
from increment._holdout_roster import observed_outcomes, resolve_roster
from increment.errors import InvalidRequestError


@pytest.fixture(scope="module")
def shopping() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/holdouts/shopping_v1.json").read_text())


@pytest.fixture
def comparison(shopping: dict[str, Any]) -> tuple[HoldoutStudy, HoldoutWindow, HoldoutSnapshot]:
    return (
        HoldoutStudy.model_validate(shopping["study"]),
        HoldoutWindow.model_validate(shopping["window"]),
        HoldoutSnapshot.model_validate(shopping["snapshot"]),
    )


type Comparison = tuple[HoldoutStudy, HoldoutWindow, HoldoutSnapshot]


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _snapshot(snapshot: HoldoutSnapshot, **updates: object) -> HoldoutSnapshot:
    return HoldoutSnapshot.model_validate({**dict(snapshot), **updates})


def _rename(old_id: str, new_id: str, *, ref: str = "links/verified-rename") -> dict[str, str]:
    return {
        "old_id": old_id,
        "new_id": new_id,
        "effective_at": "2025-03-18T00:00:00Z",
        "available_at": "2025-03-19T00:00:00Z",
        "identity_ref": ref,
    }


def test_missing_common_entry_assignment_survives_join(comparison: Comparison) -> None:
    study, window, snapshot = comparison
    snapshot = _snapshot(
        snapshot,
        memberships=tuple(row for row in snapshot.memberships if row.unit_id != "acct:3"),
        outcomes=tuple(row for row in snapshot.outcomes if row.unit_id != "acct:3"),
    )

    roster = resolve_roster(study, window, snapshot)

    assert (roster.n_eligible_entries, roster.n_unclassified) == (6, 1)
    assert roster.held_out == ArmCounts(assigned=3, measured=3)
    assert roster.eligible == ArmCounts(assigned=2, measured=2)
    assert [reason.code for reason in roster.reasons] == ["holdout.membership_missing"]


def test_lagging_census_and_ineligible_entry_are_not_missing_membership(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    ineligible = {
        **dict(snapshot.entries[0]),
        "entry_id": "entry-ineligible",
        "unit_id": "acct:ineligible",
        "eligible": False,
    }
    snapshot = _snapshot(
        snapshot,
        entry_complete_through="2025-03-09T23:59:59Z",
        entries=(*snapshot.entries, ineligible),
    )

    roster = resolve_roster(study, window, snapshot)

    assert (roster.n_eligible_entries, roster.n_ineligible_entries, roster.n_unclassified) == (
        6,
        1,
        0,
    )
    assert [reason.code for reason in roster.reasons] == ["holdout.entry_coverage_unknown"]


def test_archive_reactivation_uses_original_clock_and_freeze_is_exclusive(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    archive = dict(snapshot.configs[0])
    january = archive | {"effective_to": "2025-02-01T00:00:00Z"}
    march = archive | {"effective_from": "2025-03-05T00:00:00Z"}
    current = archive | {
        "revision": "r2",
        "archive_ref": "archives/current",
        "effective_from": "2025-04-12T00:00:00Z",
        "effective_to": "2025-05-01T00:00:00Z",
        "sdk_json": "not the assignment archive",
    }
    check = {
        **dict(snapshot.memberships[0]),
        "event_id": "sdk/seen-again",
        "assigned_at": "2025-03-23T00:00:00Z",
        "available_at": "2025-03-23T00:01:00Z",
    }
    snapshot = _snapshot(
        snapshot, configs=(current, march, january), memberships=(*snapshot.memberships, check)
    )

    roster = resolve_roster(study, window, snapshot)

    assert not roster.reasons
    assert roster.n_after_freeze == 1
    assert tuple(
        (m.original_id, m.entered_at, m.assigned_at, m.assignment_revision) for m in roster.members
    ) == (
        ("acct:1", _at("2025-01-02T00:00:00Z"), _at("2025-01-02T00:00:00Z"), "r1"),
        ("acct:15", _at("2025-01-02T00:00:00Z"), _at("2025-01-02T00:00:00Z"), "r1"),
        ("acct:16", _at("2025-01-02T00:00:00Z"), _at("2025-01-02T00:00:00Z"), "r1"),
        ("acct:2", _at("2025-01-02T00:00:00Z"), _at("2025-01-02T00:00:00Z"), "r1"),
        ("acct:3", _at("2025-01-02T00:00:00Z"), _at("2025-01-02T00:00:00Z"), "r1"),
        ("acct:4", _at("2025-03-05T00:00:00Z"), _at("2025-03-05T00:00:00Z"), "r1"),
    )


@pytest.mark.parametrize(
    "defect", ["archive_gap", "replay_group", "transaction", "late", "competing"]
)
def test_claimed_membership_cannot_replace_the_first_assignment(
    comparison: Comparison, defect: str
) -> None:
    study, window, snapshot = comparison
    first = next(row for row in snapshot.memberships if row.unit_id == "acct:4")
    claim = dict(first)
    if defect == "archive_gap":
        archive = dict(snapshot.configs[0])
        snapshot = _snapshot(
            snapshot,
            configs=(
                archive | {"effective_to": "2025-03-05T00:00:00Z"},
                archive | {"effective_from": "2025-03-06T00:00:00Z"},
            ),
        )
    elif defect in {"replay_group", "competing"}:
        claim["payload"] = {**dict(first.payload), "group": "held_out", "group_bps": 1000}
    elif defect == "transaction":
        claim["entry_id"] = "entry-unknown"
    else:
        claim["assigned_at"] = _at("2025-03-05T00:00:01Z")
    if defect == "competing":
        memberships = (*snapshot.memberships, claim | {"event_id": "sdk/competing"})
    else:
        memberships = (*(row for row in snapshot.memberships if row != first), claim)

    roster = resolve_roster(study, window, _snapshot(snapshot, memberships=memberships))

    assert roster.n_eligible_entries == 6
    assert roster.n_unclassified == 1
    assert [reason.code for reason in roster.reasons] == ["holdout.original_assignment_unresolved"]
    assert "acct:4" not in dict(roster.values)


def test_retries_and_input_order_do_not_change_evidence(comparison: Comparison) -> None:
    study, window, snapshot = comparison
    reordered = _snapshot(
        snapshot,
        entries=(*reversed(snapshot.entries), snapshot.entries[0]),
        memberships=(*reversed(snapshot.memberships), snapshot.memberships[0]),
        outcomes=tuple(reversed(snapshot.outcomes)),
    )

    assert resolve_roster(study, window, reordered) == resolve_roster(study, window, snapshot)


def test_conflicting_retry_is_a_structural_error_without_unit_or_value(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    original = snapshot.memberships[0]
    conflict = {**dict(original), "unit_id": "private-unit"}
    snapshot = _snapshot(snapshot, memberships=(*snapshot.memberships, conflict))

    with pytest.raises(InvalidRequestError) as error:
        resolve_roster(study, window, snapshot)

    assert error.value.code == "holdout.event_conflict"
    assert error.value.context == {
        "study_id": "shopping-25",
        "stream": "memberships",
        "event_id": "sdk/entry-1",
    }
    assert "private-unit" not in str(error.value)


def test_verified_alias_chain_keeps_original_group_and_whole_window_value(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    rename = _rename("acct:2", "acct:successor")
    latest = _rename("acct:successor", "acct:latest", ref="links/second-rename") | {
        "effective_at": "2025-04-01T00:00:00Z",
        "available_at": "2025-04-01T01:00:00Z",
    }
    outcomes = tuple(
        {**dict(row), "unit_id": "acct:latest"} if row.unit_id == "acct:2" else row
        for row in snapshot.outcomes
    )

    roster = resolve_roster(
        study, window, _snapshot(snapshot, aliases=(latest, rename, rename), outcomes=outcomes)
    )

    assert not roster.reasons
    assert roster.held_out == ArmCounts(assigned=3, measured=3)
    assert dict(roster.values)["acct:2"] == 8
    assert [
        (s.unit_id, s.from_at.isoformat(), s.to_at.isoformat() if s.to_at else None)
        for s in roster.identity
        if s.original_id == "acct:2"
    ] == [
        ("acct:2", "2025-01-02T00:00:00+00:00", "2025-03-18T00:00:00+00:00"),
        ("acct:latest", "2025-04-01T00:00:00+00:00", None),
        ("acct:successor", "2025-03-18T00:00:00+00:00", "2025-04-01T00:00:00+00:00"),
    ]


@pytest.mark.parametrize(
    "second_old,second_new",
    [
        ("acct:3", "acct:successor"),
        ("acct:2", "acct:other"),
        ("acct:successor", "acct:2"),
        ("acct:successor", "acct:3"),
    ],
)
def test_unverified_identity_cannot_merge_enrollments_or_reuse_ids(
    comparison: Comparison, second_old: str, second_new: str
) -> None:
    study, window, snapshot = comparison
    first = _rename("acct:2", "acct:successor")
    second = _rename(second_old, second_new, ref="links/conflicting") | {
        "effective_at": "2025-04-01T00:00:00Z",
        "available_at": "2025-04-01T01:00:00Z",
    }

    with pytest.raises(InvalidRequestError) as error:
        resolve_roster(study, window, _snapshot(snapshot, aliases=(second, first)))

    assert error.value.code == "holdout.identity_ambiguous"
    assert error.value.context == {"study_id": "shopping-25", "link_ref": "links/conflicting"}


@pytest.mark.parametrize(
    "defect,code,unit",
    [
        ("double_alias", "portfolio.impact.duplicate_unit", "acct:2"),
        ("expired_alias", "portfolio.impact.measurement_mismatch", "acct:2"),
        ("outside_roster", "portfolio.impact.measurement_mismatch", "acct:5"),
        ("foreign_window", "portfolio.impact.measurement_mismatch", "acct:2"),
    ],
)
def test_foreign_or_duplicate_outcome_preserves_canonical_failure_context(
    comparison: Comparison, defect: str, code: str, unit: str
) -> None:
    study, window, snapshot = comparison
    measured = next(row for row in snapshot.outcomes if row.unit_id == "acct:2")
    outcomes = snapshot.outcomes
    aliases = ()
    if defect in {"double_alias", "expired_alias"}:
        aliases = (_rename("acct:2", "acct:successor"),)
        if defect == "double_alias":
            outcomes = (*outcomes, {**dict(measured), "unit_id": "acct:successor"})
    elif defect == "foreign_window":
        outcomes = (
            *(row for row in outcomes if row != measured),
            {**dict(measured), "window_id": "other"},
        )
    else:
        outcomes = (*outcomes, {**dict(measured), "unit_id": "acct:5"})

    roster = resolve_roster(study, window, _snapshot(snapshot, aliases=aliases, outcomes=outcomes))

    reason = roster.reasons[0]
    assert reason.code == code
    expected = {"unit_id": unit}
    if defect == "double_alias":
        expected["records"] = "outcomes"
    assert reason.context == expected
    assert unit not in dict(roster.values)


def test_censored_unit_remains_assigned_but_observed_mean_has_its_own_denominator(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    outcomes = tuple(
        {
            **dict(row),
            "state": "censored",
            "reason": "lost-registry-coverage",
            "value": None,
            "observed_through": "2025-03-20T00:00:00Z",
            "available_at": "2025-03-21T00:00:00Z",
        }
        if row.unit_id == "acct:16"
        else row
        for row in snapshot.outcomes
    )
    final = _snapshot(snapshot, outcomes=outcomes)
    roster = resolve_roster(study, window, final)

    assert roster.held_out == ArmCounts(assigned=3, measured=2, censored=1)
    assert roster.eligible == ArmCounts(assigned=3, measured=3)
    assert roster.reasons[0].code == "portfolio.impact.missing_outcomes"
    panel = observed_outcomes(window, final, roster)
    assert panel is not None
    assert panel.held_out == (3, 2, 9.0)
    assert panel.eligible == (3, 3, 12.0)

    interim = _snapshot(
        final, as_of="2025-03-26T00:00:00Z", entry_complete_through="2025-03-26T00:00:00Z"
    )
    partial = resolve_roster(study, window, interim)
    assert partial.held_out == ArmCounts(assigned=3, pending=2, censored=1)
    assert partial.eligible == ArmCounts(assigned=3, pending=3)
    assert observed_outcomes(window, interim, partial) is None


def test_attested_zero_is_measured_but_partial_and_future_rows_are_missing(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    changes = {
        "acct:1": {"value": 0},
        "acct:3": {"observed_through": "2025-04-09T23:59:59Z"},
        "acct:4": {"available_at": "2025-04-12T00:00:01Z"},
    }
    outcomes = tuple({**dict(row), **changes.get(row.unit_id, {})} for row in snapshot.outcomes)

    roster = resolve_roster(study, window, _snapshot(snapshot, outcomes=outcomes))

    assert roster.eligible == ArmCounts(assigned=3, measured=1, missing=2)
    assert dict(roster.values)["acct:1"] == 0
    assert "acct:3" not in dict(roster.values)
    assert "acct:4" not in dict(roster.values)


def test_pretreatment_census_keeps_nonreturners_in_each_original_arm(
    comparison: Comparison, shopping: dict[str, Any]
) -> None:
    study, window, snapshot = comparison
    sample = shopping["external_assignment"]
    entries, memberships, outcomes, deliveries = [], [], [], []
    for group, template in zip(("eligible", "held_out"), sample["memberships"], strict=True):
        for num, value in enumerate((0, 0, 10, 10)):
            unit = f"{group}:{num}"
            entries.append({**dict(snapshot.entries[0]), "unit_id": unit, "entry_id": unit})
            memberships.append(
                {
                    **template,
                    "unit_id": unit,
                    "entry_id": unit,
                    "event_id": f"assign/{unit}",
                    "payload": {**template["payload"], "assignment_ref": f"external/{unit}"},
                }
            )
            outcomes.append({**dict(snapshot.outcomes[0]), "unit_id": unit, "value": value})
            if group == "held_out" or value == 10:
                deliveries.append(
                    {
                        "event_id": f"visit/{unit}",
                        "unit_id": unit,
                        "launch_id": "A",
                        "target": "shopping-a",
                        "feature_version": "baseline/1",
                        "rollout_revision": "A/ramp/1",
                        "at": "2025-03-08T00:00:00Z",
                        "available_at": "2025-03-08T00:01:00Z",
                        "nonbaseline": False,
                    }
                )
    snapshot = _snapshot(
        snapshot,
        membership_source="external",
        configs=sample["configs"],
        entries=entries,
        memberships=memberships,
        outcomes=outcomes,
        deliveries=deliveries,
    )

    roster = resolve_roster(study, window, snapshot)
    panel = observed_outcomes(window, snapshot, roster)
    assert not roster.reasons
    assert panel is not None
    assert panel.eligible == panel.held_out == (4, 4, 5.0)

    filtered = HoldoutStudy.model_validate({**dict(study), "entry_basis": "post_assignment_filter"})
    returners = {r["unit_id"] for r in deliveries}
    visit_snapshot = _snapshot(
        snapshot,
        entries=[r for r in entries if r["unit_id"] in returners],
        memberships=[r for r in memberships if r["unit_id"] in returners],
        outcomes=[r for r in outcomes if r["unit_id"] in returners],
    )
    selected = resolve_roster(filtered, window, visit_snapshot)
    assert selected.n_eligible_entries == 6
    assert selected.reasons[0].code == "holdout.entry_selection_unresolved"
    unsafe_panel = observed_outcomes(window, visit_snapshot, selected)
    assert unsafe_panel is not None
    assert unsafe_panel.eligible.mean_among_measured is None


def test_scoped_offer_before_march_entry_is_not_postassignment_drift(
    comparison: Comparison,
) -> None:
    study, window, snapshot = comparison
    decision = {
        "event_id": "decision/pre-entry",
        "unit_id": "acct:4",
        "launch_id": "A",
        "target": "shopping-a",
        "config_revision": "r1",
        "holdout_blocked": False,
        "nonbaseline_assigned": True,
        "at": "2025-03-04T00:00:00Z",
        "available_at": "2025-03-04T00:01:00Z",
    }

    roster = resolve_roster(study, window, _snapshot(snapshot, decisions=(decision,)))

    assert roster.n_eligible_entries == 6
    assert roster.reasons[0].code == "holdout.entry_selection_unresolved"
