"""Reconcile original holdout membership and whole-window observations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import datetime
from statistics import mean
from typing import Literal, NamedTuple

from increment._holdout_records import (
    Alias,
    ArmCounts,
    ConfigIndex,
    Entry,
    ExternalMembership,
    ExternalPolicy,
    Group,
    HoldoutReason,
    HoldoutSnapshot,
    HoldoutStudy,
    HoldoutWindow,
    IdentitySpan,
    MembershipEvent,
    Outcome,
    RosterEvidence,
    RosterMember,
    SDKMembership,
    check_config_binding,
    index_configs,
)
from increment.assignment import AllocationEvaluator
from increment.errors import InvalidRequestError, RefusalSpec, refuse

EVENT_CONFLICT = RefusalSpec(
    "holdout.event_conflict",
    InvalidRequestError,
    template="Study {study_id} has conflicting {stream} records for event {event_id}.",
)
IDENTITY_AMBIGUOUS = RefusalSpec(
    "holdout.identity_ambiguous",
    InvalidRequestError,
    template="Study {study_id} cannot establish a unique original identity at link {link_ref}.",
)

_ENTRY_SELECTION = "holdout.entry_selection_unresolved"
_ORIGINAL_ASSIGNMENT = "holdout.original_assignment_unresolved"
_OUTCOME_MISMATCH = "portfolio.impact.measurement_mismatch"
_OUTCOME_DUPLICATE = "portfolio.impact.duplicate_unit"


def _events_as_of[T: (Entry, MembershipEvent)](
    study: HoldoutStudy,
    rows: tuple[T, ...],
    cut: datetime,
    stream: str,
    key: Callable[[T], str],
) -> dict[str, T]:
    unique: dict[str, T] = {}
    first_conflict: str | None = None
    for row in rows:
        if row.available_at > cut:
            continue
        event_id = key(row)
        if unique.setdefault(event_id, row) != row and (
            first_conflict is None or event_id < first_conflict
        ):
            first_conflict = event_id
    if first_conflict is not None:
        refuse(EVENT_CONFLICT, study_id=study.study_id, stream=stream, event_id=first_conflict)
    return unique


def _first_entries(
    entries: dict[str, Entry], cut: datetime, freeze: datetime
) -> tuple[dict[str, Entry], set[str]]:
    first: dict[str, Entry] = {}
    uncertain: set[str] = set()
    for entry in sorted(entries.values(), key=lambda row: (row.entered_at, row.entry_id)):
        if entry.entered_at > cut:
            if entry.entered_at < freeze:
                uncertain.add(entry.unit_id)
            continue
        earlier = first.setdefault(entry.unit_id, entry)
        if (
            entry.entered_at < freeze
            and entry.entered_at == earlier.entered_at
            and (entry.entry_id != earlier.entry_id or entry.entered_at > entry.available_at)
        ):
            uncertain.add(entry.unit_id)
    return first, uncertain


def _identity_index(
    study: HoldoutStudy, first: dict[str, Entry], snapshot: HoldoutSnapshot
) -> dict[str, IdentitySpan]:
    incoming: dict[str, Alias] = {}
    outgoing: dict[str, Alias] = {}
    links = sorted(
        (alias for alias in snapshot.aliases if alias.available_at <= snapshot.as_of),
        key=lambda alias: (alias.effective_at, alias.identity_ref, alias.old_id, alias.new_id),
    )
    for alias in links:
        old, new, at = alias.old_id, alias.new_id, alias.effective_at
        prior = incoming.get(old)
        successor = outgoing.get(old)
        if (
            alias.available_at < at
            or old == new
            or new in first
            or (old in first and first[old].entered_at >= at)
            or (prior is not None and prior.effective_at >= at)
        ):
            refuse(IDENTITY_AMBIGUOUS, study_id=study.study_id, link_ref=alias.identity_ref)
        if successor is not None and successor.new_id == new and successor.effective_at == at:
            continue
        if successor is not None or new in incoming or new in outgoing:
            refuse(IDENTITY_AMBIGUOUS, study_id=study.study_id, link_ref=alias.identity_ref)
        incoming[new] = alias
        outgoing[old] = alias

    spans: dict[str, IdentitySpan] = {}
    for original, entry in first.items():
        unit, at = original, entry.entered_at
        while (next_link := outgoing.get(unit)) is not None:
            spans[unit] = IdentitySpan(
                unit_id=unit, original_id=original, from_at=at, to_at=next_link.effective_at
            )
            unit, at = next_link.new_id, next_link.effective_at
        spans[unit] = IdentitySpan(unit_id=unit, original_id=original, from_at=at)
    return spans


def _original_at(
    identity: dict[str, IdentitySpan], unit: str, at: datetime, *, before_entry: bool = False
) -> str | None:
    span = identity.get(unit)
    if span is None or (span.to_at is not None and at >= span.to_at):
        return None
    if at >= span.from_at or (before_entry and unit == span.original_id):
        return span.original_id
    return None


def _preentry_offers(
    study: HoldoutStudy,
    snapshot: HoldoutSnapshot,
    first: dict[str, Entry],
    identity: dict[str, IdentitySpan],
) -> set[str]:
    scope = {(component.launch_id, component.target) for component in study.components}
    nonbaseline = (
        (row for row in snapshot.decisions if row.nonbaseline_assigned),
        (row for row in snapshot.deliveries if row.nonbaseline),
    )
    early: set[str] = set()
    for rows in nonbaseline:
        for row in rows:
            if (
                (row.launch_id, row.target) in scope
                and row.at <= row.available_at <= snapshot.as_of
                and (original := _original_at(identity, row.unit_id, row.at, before_entry=True))
                is not None
                and row.at < first[original].entered_at
            ):
                early.add(original)
    return early


def _membership_claims(
    events: dict[str, MembershipEvent],
    entries: dict[str, Entry],
    first: dict[str, Entry],
    included: dict[str, Entry],
    identity: dict[str, IdentitySpan],
    freeze: datetime,
) -> tuple[dict[str, list[MembershipEvent]], bool, bool]:
    claims: dict[str, list[MembershipEvent]] = defaultdict(list)
    unmatched = crosses_freeze = False
    for event in events.values():
        by_unit = _original_at(identity, event.unit_id, event.assigned_at)
        if by_unit is not None and first[by_unit].entry_id == event.entry_id:
            if by_unit in included:
                claims[by_unit].append(event)
            continue
        if event.assigned_at >= freeze:
            continue
        transaction = entries.get(event.entry_id)
        by_entry = transaction.unit_id if transaction is not None else None
        # A later first entry contradicts this clock; it does not resolve the ID.
        if any(
            unit is not None and unit in first and first[unit].entered_at >= freeze
            for unit in (event.unit_id, by_entry)
        ):
            crosses_freeze = True
        else:
            unmatched = True
        # Preserve evidence of a bad claim without competing with a verified one.
        for unit in (by_unit, by_entry):
            if unit is not None and unit in included:
                claims.setdefault(unit, [])
    return claims, unmatched, crosses_freeze


class _ArchiveReplay:
    def __init__(
        self, study: HoldoutStudy, snapshot: HoldoutSnapshot, configs: ConfigIndex
    ) -> None:
        self.study = study
        self.snapshot = snapshot
        self.configs = configs
        self.bound: dict[str, AllocationEvaluator | ExternalPolicy] = {}

    def matches(self, event: MembershipEvent) -> bool:
        payload = event.payload
        archive = self.configs.at(payload.revision, event.assigned_at)
        if archive is None:
            return False
        bound = self.bound.get(payload.revision)
        if bound is None:
            bound = check_config_binding(self.study, self.snapshot, archive)
            self.bound[payload.revision] = bound
        if isinstance(payload, SDKMembership) and isinstance(bound, AllocationEvaluator):
            replayed = bound.holdout_membership(self.study.holdout_key, event.unit_id)
            return payload.model_dump() == replayed
        return (
            isinstance(payload, ExternalMembership)
            and isinstance(bound, ExternalPolicy)
            and payload.holdout_key == bound.holdout_key
            and payload.holdout_probability == bound.holdout_probability
        )


def _original_membership(
    study: HoldoutStudy,
    entry: Entry,
    claims: list[MembershipEvent],
    replay: _ArchiveReplay,
) -> MembershipEvent | None:
    first: MembershipEvent | None = None
    for claim in claims:
        if claim.assigned_at != entry.entered_at:
            continue
        if (
            claim.unit_id != entry.unit_id
            or claim.law_id != study.law_id
            or claim.available_at < claim.assigned_at
            or (first is not None and claim.payload != first.payload)
        ):
            return None
        first = claim
    if first is None or not replay.matches(first):
        return None
    for claim in claims:
        if (
            claim.assigned_at > entry.entered_at
            and claim.assigned_at <= claim.available_at
            and (claim.payload.group != first.payload.group or claim.law_id != first.law_id)
        ):
            return None
    return first


def _window_measurements(
    window: HoldoutWindow,
    snapshot: HoldoutSnapshot,
    included: dict[str, Entry],
    identity: dict[str, IdentitySpan],
) -> tuple[dict[str, Outcome], tuple[HoldoutReason, ...]]:
    terms = window.terms
    by_original: dict[str, list[Outcome]] = defaultdict(list)
    mismatched: set[str] = set()
    duplicates: list[str] = []
    for row in snapshot.outcomes:
        if row.available_at > snapshot.as_of:
            continue
        span = identity.get(row.unit_id)
        original = span.original_id if span is not None else None
        if (
            original is None
            or original not in included
            or row.window_id != window.window_id
            or row.outcome_version != terms.outcome_version
            or row.start != terms.start
            or row.end != terms.end
        ):
            mismatched.add(row.unit_id)
        else:
            by_original[original].append(row)

    measurements: dict[str, Outcome] = {}
    for original, rows in sorted(by_original.items()):
        if len(rows) > 1:
            # Retired and current IDs cannot contribute two whole-window outcomes.
            duplicates.append(original)
            continue
        row = rows[0]
        close = terms.end if row.state == "measured" else min(terms.end, row.observed_through)
        if _original_at(identity, row.unit_id, close) != original:
            mismatched.add(row.unit_id)
        elif row.observed_through <= row.available_at:
            measurements[original] = row

    reasons = tuple(
        HoldoutReason(code=_OUTCOME_DUPLICATE, context={"unit_id": unit, "records": "outcomes"})
        for unit in duplicates
    ) + tuple(
        HoldoutReason(code=_OUTCOME_MISMATCH, context={"unit_id": unit})
        for unit in sorted(mismatched)
    )
    return measurements, reasons


type _State = Literal["measured", "pending", "censored", "missing"]


def _state(
    row: Outcome | None, outcome_source_ref: str | None, end: datetime, cut: datetime
) -> tuple[_State, float | None]:
    if outcome_source_ref is not None and row is not None:
        if row.state == "censored":
            return "censored", None
        if row.state == "measured" and row.observed_through >= end and row.value is not None:
            return "measured", row.value
    return ("pending" if cut < end else "missing"), None


def _counts(states: Counter[_State]) -> ArmCounts:
    return ArmCounts(
        assigned=states.total(),
        measured=states["measured"],
        pending=states["pending"],
        censored=states["censored"],
        missing=states["missing"],
    )


def resolve_roster(
    study: HoldoutStudy, window: HoldoutWindow, snapshot: HoldoutSnapshot
) -> RosterEvidence:
    """Bind visible first entries to their original assignment and certified outcome."""
    freeze = window.terms.freeze_at
    entries = _events_as_of(
        study, snapshot.entries, snapshot.as_of, "entries", lambda row: row.entry_id
    )
    events = _events_as_of(
        study, snapshot.memberships, snapshot.as_of, "memberships", lambda row: row.event_id
    )
    first, uncertain = _first_entries(entries, snapshot.as_of, freeze)
    identity = _identity_index(study, first, snapshot)
    early = _preentry_offers(study, snapshot, first, identity)
    included: dict[str, Entry] = {}
    n_ineligible = n_after_freeze = 0
    for unit, entry in first.items():
        if entry.entered_at >= freeze:
            n_after_freeze += 1
            continue
        if entry.rule_ref != study.eligibility_rule_ref or unit in early:
            uncertain.add(unit)
        if entry.eligible:
            included[unit] = entry
        elif unit not in uncertain:
            n_ineligible += 1
    claims, unresolved, crosses_freeze = _membership_claims(
        events, entries, first, included, identity, freeze
    )
    selection = study.entry_basis != "pretreatment_first_entry" or bool(uncertain) or crosses_freeze
    replay = _ArchiveReplay(study, snapshot, index_configs(study, snapshot))
    missing_membership = False
    invalid_window = False
    members: list[RosterMember] = []
    for unit, entry in sorted(included.items()):
        if unit in uncertain:
            continue
        if (candidates := claims.get(unit)) is None:
            missing_membership = True
            continue
        original = _original_membership(study, entry, candidates, replay)
        if original is None:
            unresolved = True
            continue
        invalid_window |= original.assigned_at < study.registered_at
        members.append(
            RosterMember(
                unit,
                entry.entered_at,
                original.assigned_at,
                original.payload.group,
                original.payload.revision,
            )
        )

    reasons = [
        HoldoutReason(code=code, context={"study_id": study.study_id})
        for applies, code in (
            (selection, _ENTRY_SELECTION),
            (
                snapshot.entry_source_ref is None
                or snapshot.entry_complete_through is None
                or not freeze <= snapshot.entry_complete_through <= snapshot.as_of,
                "holdout.entry_coverage_unknown",
            ),
            (unresolved, _ORIGINAL_ASSIGNMENT),
            (missing_membership, "holdout.membership_missing"),
        )
        if applies
    ]
    if invalid_window:
        reasons.append(
            HoldoutReason(
                code="portfolio.impact.invalid_window", context={"protocol_id": window.window_id}
            )
        )
    measurements, outcome_reasons = _window_measurements(window, snapshot, included, identity)
    reasons.extend(outcome_reasons)
    states: dict[Group, Counter[_State]] = {"held_out": Counter(), "eligible": Counter()}
    values: list[tuple[str, float]] = []
    for member in members:
        state, value = _state(
            measurements.get(member.original_id),
            snapshot.outcome_source_ref,
            window.terms.end,
            snapshot.as_of,
        )
        states[member.group][state] += 1
        if value is not None:
            values.append((member.original_id, value))
    held_out, eligible = _counts(states["held_out"]), _counts(states["eligible"])
    if (
        not outcome_reasons
        and held_out.missing + held_out.censored + eligible.missing + eligible.censored
        and snapshot.as_of >= window.terms.end
    ):
        reasons.append(HoldoutReason(code="portfolio.impact.missing_outcomes"))

    return RosterEvidence(
        members=tuple(members),
        identity=tuple(span for _, span in sorted(identity.items())),
        values=tuple(values),
        held_out=held_out,
        eligible=eligible,
        n_eligible_entries=len(included),
        n_ineligible_entries=n_ineligible,
        n_unclassified=len(included) - len(members),
        n_after_freeze=n_after_freeze,
        reasons=tuple(reasons),
    )


class ObservedArm(NamedTuple):
    """The original assignment and measured denominators for one group."""

    n_assigned: int
    n_measured: int
    mean_among_measured: float | None


class ObservedOutcomes(NamedTuple):
    """Within-group observations, not a comparison of assignments."""

    held_out: ObservedArm
    eligible: ObservedArm


def observed_outcomes(
    window: HoldoutWindow, snapshot: HoldoutSnapshot, roster: RosterEvidence
) -> ObservedOutcomes | None:
    """Summarize confirmed measurements by group only at the registered final cut."""
    if snapshot.as_of != window.terms.final_as_of:
        return None
    values: dict[Group, list[float]] = {"held_out": [], "eligible": []}
    if roster.n_unclassified == 0 and not any(
        reason.code in {_ENTRY_SELECTION, _ORIGINAL_ASSIGNMENT} for reason in roster.reasons
    ):
        groups = {member.original_id: member.group for member in roster.members}
        for unit, value in roster.values:
            values[groups[unit]].append(value)
    return ObservedOutcomes(
        ObservedArm(
            roster.held_out.assigned,
            roster.held_out.measured,
            mean(values["held_out"]) if values["held_out"] else None,
        ),
        ObservedArm(
            roster.eligible.assigned,
            roster.eligible.measured,
            mean(values["eligible"]) if values["eligible"] else None,
        ),
    )
