"""Audit rollout and enforcement records for the original holdout roster."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256
from operator import attrgetter
from typing import Literal

from increment._canonical import canonical_json_bytes
from increment._holdout_records import (
    ConfigIndex,
    DecisionEvent,
    Delivery,
    Feed,
    Group,
    HoldoutReason,
    HoldoutSnapshot,
    HoldoutStudy,
    HoldoutWindow,
    IdentitySpan,
    Phase,
    RolloutAudit,
    RolloutFinding,
    RosterEvidence,
    RosterMember,
    UnmatchedRolloutEvent,
    check_config_binding,
    index_configs,
)

type _Panel = tuple[str, Group, Feed]
type _Interval = tuple[datetime, datetime]
type _CoverageStatus = Literal["covered", "partial", "unknown"]
type _Event = DecisionEvent | Delivery

_GROUPS: tuple[Group, ...] = ("eligible", "held_out")
_FEEDS: tuple[Feed, ...] = ("decision", "delivery")
_PHASE_ORDER = attrgetter(
    "launch_id",
    "from_at",
    "to_at",
    "rollout_revision",
    "target",
    "feature_version",
    "config_revision",
    "baseline_version",
    "eligible_offer_probability",
    "actual_ref",
    "available_at",
)


@dataclass(frozen=True, slots=True)
class _History:
    phases: tuple[Phase, ...]
    timelines: dict[str, list[Phase]]
    configs: ConfigIndex
    unresolved: set[str]

    def at(self, launch_id: str, at: datetime) -> Phase | None:
        timeline = self.timelines.get(launch_id)
        if timeline is None:
            return None
        index = bisect_right(timeline, at, key=attrgetter("from_at")) - 1
        if index >= 0 and at < timeline[index].to_at:
            return timeline[index]
        return None

    def digest(self) -> str | None:
        if not self.phases:
            return None
        data = {
            "format": "increment.holdout.history/v1",
            "phases": [phase.model_dump(mode="json") for phase in self.phases],
        }
        return "sha256:" + sha256(canonical_json_bytes(data)).hexdigest()


def _audit_history(
    study: HoldoutStudy,
    snapshot: HoldoutSnapshot,
    targets: dict[str, str],
    start: datetime,
    stop: datetime,
) -> _History:
    configs = index_configs(study, snapshot)
    if start >= stop:
        return _History((), {}, configs, set())

    unresolved = set(targets) if snapshot.history_ref is None else set()
    observed = sorted(dict.fromkeys(snapshot.phases), key=_PHASE_ORDER)
    timelines: dict[str, list[Phase]] = defaultdict(list)
    for phase in observed:
        if phase.available_at > snapshot.as_of or phase.from_at >= stop or phase.to_at <= start:
            continue
        launch = phase.launch_id
        if (
            targets.get(launch) != phase.target
            or phase.baseline_version != study.baseline_version
            or phase.from_at >= phase.to_at
            or phase.available_at < phase.from_at
            or (
                phase.feature_version == phase.baseline_version and phase.eligible_offer_probability
            )
        ):
            unresolved.add(launch)
        else:
            timelines[launch].append(phase)

    phases = tuple(phase for launch in targets for phase in timelines.get(launch, ()))
    checked_revisions: set[str] = set()
    for launch in targets:
        timeline = timelines.get(launch, [])
        cursor = start
        overlapping = False
        rollout_rules: dict[str, tuple[str, float]] = {}
        for phase in timeline:
            lower, upper = max(start, phase.from_at), min(stop, phase.to_at)
            rule = (phase.feature_version, phase.eligible_offer_probability)
            if lower != cursor or rollout_rules.setdefault(phase.rollout_revision, rule) != rule:
                unresolved.add(launch)
            overlapping |= lower < cursor
            cursor = max(cursor, upper)
            if not _config_covers(study, snapshot, configs, phase, lower, upper, checked_revisions):
                unresolved.add(launch)
        if cursor != stop:
            unresolved.add(launch)
        # Conflicting intervals do not supply a unique version for an event.
        if overlapping:
            del timelines[launch]
    return _History(phases, timelines, configs, unresolved)


def _config_covers(
    study: HoldoutStudy,
    snapshot: HoldoutSnapshot,
    configs: ConfigIndex,
    phase: Phase,
    start: datetime,
    stop: datetime,
    checked_revisions: set[str],
) -> bool:
    revision = phase.config_revision
    while start < stop:
        archive = configs.at(revision, start)
        if archive is None:
            return False
        if revision not in checked_revisions:
            check_config_binding(study, snapshot, archive)
            checked_revisions.add(revision)
        start = archive.effective_to
    return True


@dataclass(frozen=True, slots=True)
class _RosterIndex:
    members: dict[str, RosterMember]
    identities: dict[str, list[IdentitySpan]]

    def match(self, event: _Event) -> RosterMember | str:
        spans = self.identities.get(event.unit_id, ())
        index = bisect_right(spans, event.at, key=attrgetter("from_at")) - 1
        if index >= 0:
            span = spans[index]
            if span.to_at is None or event.at < span.to_at:
                return self.members.get(span.original_id) or "outside_cohort"
        # A literal original ID can evidence pre-entry access, but a successor
        # recorded before its effective time cannot be linked retrospectively.
        member = self.members.get(event.unit_id)
        if (
            index < 0
            and member is not None
            and event.at < member.assigned_at
            and (not spans or spans[0].original_id == member.original_id)
        ):
            return member
        return "identity_not_effective" if spans or member else "unknown_unit"


def _index_roster(roster: RosterEvidence) -> _RosterIndex:
    identities: dict[str, list[IdentitySpan]] = defaultdict(list)
    for span in roster.identity:
        identities[span.unit_id].append(span)
    for spans in identities.values():
        spans.sort(key=attrgetter("from_at"))
    return _RosterIndex({member.original_id: member for member in roster.members}, identities)


@dataclass(slots=True)
class _ObservedUnits:
    recorded: set[str] = field(default_factory=set)
    nonbaseline: set[str] = field(default_factory=set)
    violating: set[str] = field(default_factory=set)

    def record(self, member: RosterMember, event: _Event, *, nonbaseline: bool) -> None:
        self.recorded.add(member.original_id)
        if nonbaseline:
            self.nonbaseline.add(member.original_id)
        if (member.group == "held_out" and nonbaseline) or (
            member.group == "eligible"
            and isinstance(event, DecisionEvent)
            and event.holdout_blocked
        ):
            self.violating.add(member.original_id)


def _visible_events(
    events: tuple[DecisionEvent, ...] | tuple[Delivery, ...], as_of: datetime
) -> tuple[dict[str, _Event], set[str]]:
    visible: dict[str, _Event] = {}
    conflicts: set[str] = set()
    for event in events:
        if event.available_at <= as_of and event.at <= as_of:
            if visible.setdefault(event.event_id, event) != event:
                conflicts.add(event.event_id)
    for event_id in conflicts:
        del visible[event_id]
    return visible, conflicts


def _phase_issue(history: _History, event: _Event) -> str | None:
    phase = history.at(event.launch_id, event.at)
    if phase is None:
        return "out_of_phase"
    if history.configs.at(phase.config_revision, event.at) is None:
        return "inactive_config"
    if isinstance(event, DecisionEvent):
        return None if event.config_revision == phase.config_revision else "config_mismatch"
    if event.nonbaseline and not phase.eligible_offer_probability:
        return "out_of_phase"
    version = phase.feature_version if event.nonbaseline else phase.baseline_version
    if event.rollout_revision != phase.rollout_revision or event.feature_version != version:
        return "version_mismatch"
    return None


def _audit_event(
    event: _Event,
    kind: Feed,
    targets: dict[str, str],
    stop: datetime,
    roster: _RosterIndex,
    history: _History,
    observed: dict[_Panel, _ObservedUnits],
    before_assignment: set[str],
) -> str | None:
    if targets.get(event.launch_id) != event.target:
        return "outside_scope"
    if event.available_at < event.at:
        return "invalid_availability"
    if event.at >= stop:
        return "outside_interval"
    member = roster.match(event)
    if isinstance(member, str):
        return member
    nonbaseline = (
        event.nonbaseline_assigned if isinstance(event, DecisionEvent) else event.nonbaseline
    )
    if event.at < member.assigned_at:
        if nonbaseline:
            before_assignment.add(event.launch_id)
        return "before_assignment"
    if event.at < member.entered_at:
        return "outside_interval"

    # A scoped observation can directly evidence a leak even if it cannot be
    # reconciled to the version that should have been running at the time.
    observed[(event.launch_id, member.group, kind)].record(member, event, nonbaseline=nonbaseline)
    return _phase_issue(history, event)


def _coverage_status(
    intervals: list[_Interval], start: datetime, stop: datetime
) -> _CoverageStatus:
    cursor = start
    for lower, upper in sorted(intervals):
        if upper <= start or lower >= stop:
            continue
        if lower > cursor:
            return "partial"
        cursor = max(cursor, upper)
        if cursor >= stop:
            return "covered"
    return "partial" if cursor > start else "unknown"


def _findings(
    roster: RosterEvidence,
    snapshot: HoldoutSnapshot,
    targets: dict[str, str],
    stop: datetime,
    observed: dict[_Panel, _ObservedUnits],
) -> tuple[RolloutFinding, ...]:
    # The union of all [entry, stop) intervals in an arm starts at its earliest entry.
    first: dict[Group, datetime] = {}
    for member in roster.members:
        first[member.group] = min(first.get(member.group, stop), member.entered_at)
    coverage: dict[_Panel, list[_Interval]] = defaultdict(list)
    for row in snapshot.coverage:
        if row.launch_id in targets and row.available_at <= snapshot.as_of:
            upper = min(row.to_at, row.available_at, stop)
            if row.from_at < upper:
                coverage[(row.launch_id, row.group, row.kind)].append((row.from_at, upper))
    findings: list[RolloutFinding] = []
    for (launch, group, kind), units in observed.items():
        start = first.get(group, stop)
        status = (
            _coverage_status(coverage.get((launch, group, kind), []), start, stop)
            if start < stop
            else "unknown"
        )
        n_assigned = roster.held_out.assigned if group == "held_out" else roster.eligible.assigned
        findings.append(
            RolloutFinding(
                launch_id=launch,
                group=group,
                kind=kind,
                n_assigned=n_assigned,
                n_units_recorded=len(units.recorded),
                n_nonbaseline=len(units.nonbaseline),
                n_violating=len(units.violating),
                coverage_status=status,
                finding=(
                    "recorded_violation"
                    if units.violating
                    else "none_recorded"
                    if status == "covered"
                    else "unknown"
                ),
            )
        )
    return tuple(findings)


def audit_rollout(
    study: HoldoutStudy,
    window: HoldoutWindow,
    snapshot: HoldoutSnapshot,
    roster: RosterEvidence,
) -> RolloutAudit:
    """Summarize actual history and scoped observations without changing assignments."""
    targets = {component.launch_id: component.target for component in study.components}
    targets = dict(sorted(targets.items()))
    stop = min(window.terms.end, snapshot.as_of)
    start = min((member.entered_at for member in roster.members), default=study.registered_at)
    history = _audit_history(study, snapshot, targets, start, stop)
    identities = _index_roster(roster)
    observed = {
        (launch, group, kind): _ObservedUnits()
        for launch in targets
        for group in _GROUPS
        for kind in _FEEDS
    }
    before_assignment: set[str] = set()
    unmatched: list[UnmatchedRolloutEvent] = []
    reasons: list[HoldoutReason] = []
    feeds: tuple[tuple[Feed, str, tuple[DecisionEvent, ...] | tuple[Delivery, ...]], ...] = (
        ("decision", "decisions", snapshot.decisions),
        ("delivery", "deliveries", snapshot.deliveries),
    )
    for kind, stream, events in feeds:
        visible, conflicts = _visible_events(events, snapshot.as_of)
        reasons.extend(
            HoldoutReason(
                code="holdout.event_conflict",
                context={"study_id": study.study_id, "stream": stream, "event_id": event_id},
            )
            for event_id in sorted(conflicts)
        )
        for event_id, event in sorted(visible.items()):
            issue = _audit_event(
                event, kind, targets, stop, identities, history, observed, before_assignment
            )
            if issue is not None:
                unmatched.append(UnmatchedRolloutEvent(kind=kind, event_id=event_id, reason=issue))
    for code, launches in (
        ("holdout.exposure_before_assignment", before_assignment),
        ("holdout.launch_history_unresolved", history.unresolved),
    ):
        reasons.extend(
            HoldoutReason(code=code, context={"study_id": study.study_id, "launch_id": launch})
            for launch in sorted(launches)
        )
    return RolloutAudit(
        phases=history.phases,
        history_ref=snapshot.history_ref,
        history_digest=history.digest(),
        findings=_findings(roster, snapshot, targets, stop, observed),
        unmatched_events=tuple(unmatched),
        reasons=tuple(reasons),
    )
