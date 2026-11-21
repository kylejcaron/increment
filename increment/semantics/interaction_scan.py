"""Outcome-free interaction plans and bounded records for one fixed look."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated, Literal, NoReturn, Self

from pydantic import Field, StrictFloat

from increment._overlap_evidence import (
    FEED_ROLES,
    CellId,
    Count,
    FactorId,
    FeedRole,
    Identity,
    Instant,
    Ordered,
    OverlapCell,
    OverlapDeliveryCount,
    OverlapModel,
    Sha256,
    _instant_json,
    overlap_digest,
)
from increment._overlap_refusals import SCAN_CUTOFF_MISMATCH, SCAN_INVALID
from increment.errors import refuse
from increment.semantics.overlap import OverlapEvidence, OverlapMetric, OverlapPlan, OverlapTarget

ScanCoverage = Literal["complete", "partial", "not_captured"]
ScanContrast = Literal["focal_given_other_off", "focal_given_other_on", "interaction"]
SCAN_CONTRASTS: tuple[ScanContrast, ...] = (
    "focal_given_other_off",
    "focal_given_other_on",
    "interaction",
)
ScanAvailability = Literal["unavailable", "point_only", "descriptive", "fitted_small", "decided"]
ScanDecision = Literal["detected", "none_detected"]
type _Finite = Annotated[StrictFloat, Field(allow_inf_nan=False)]
type _Positive = Annotated[_Finite, Field(gt=0)]


class _ScanModel(OverlapModel):
    @classmethod
    def _invalid_field(cls, field: str, reason: str) -> NoReturn:
        refuse(SCAN_INVALID, field=field, reason=reason)


class ExperimentIdentity(_ScanModel):
    """One archived revision, with names excluded from identity."""

    target: Identity
    experiment_id: Identity
    revision: Identity

    @property
    def sort_key(self) -> tuple[str, str, str]:
        """Scalar strings sort in the same order as their UTF-8 bytes."""
        return self.target, self.experiment_id, self.revision


class InventoryEntry(_ScanModel):
    """Archived assignment regime; binary arms are in off, on order."""

    identity: ExperimentIdentity
    archive_sha256: Sha256 | None
    unit_namespace: Identity
    active_from: Instant
    active_until: Instant
    arms: Ordered[tuple[Identity, ...]]
    display_name: Identity
    reasons: Ordered[tuple[Identity, ...]] = ()

    def _validate_overlap(self) -> Self:
        if self.active_from >= self.active_until:
            self._invalid_field("active_until", "invalid_active_window")
        if len(set(self.arms)) != len(self.arms):
            self._invalid_field("arms", "duplicate_arm")
        return self


class FocalStudy(_ScanModel):
    """Outcome-free focal assignment and common-entry population declaration."""

    entry: InventoryEntry
    population_id: Identity
    eligibility_rule_sha256: Sha256
    enrollment_rule_sha256: Sha256
    target: OverlapTarget

    def _validate_overlap(self) -> Self:
        if self.entry.archive_sha256 is None or len(self.entry.arms) != 2:
            self._invalid_field("focal", "requires_archived_binary_assignment")
        return self


class ScanWindow(_ScanModel):
    """Half-open entry and outcome windows followed by one fixed data cutoff."""

    entry_open: Instant
    start: Instant
    end: Instant
    cutoff: Instant

    def _validate_overlap(self) -> Self:
        if not self.entry_open < self.start < self.end:
            self._invalid_field("window", "entry_or_outcome_window_out_of_order")
        if self.cutoff < self.end:
            self._invalid_field("cutoff", "cutoff_before_end")
        return self

    def require_cutoff(self, actual: datetime) -> None:
        """Refuse data from another look, even if its coverage is better."""
        if actual != self.cutoff:
            refuse(
                SCAN_CUTOFF_MISMATCH,
                expected_cutoff=_instant_json(self.cutoff),
                actual_cutoff=_instant_json(actual),
            )


class ScanScope(ScanWindow):
    """Named source, focal study, window and fixed family risk."""

    universe_id: Identity
    source_id: Identity
    focal: FocalStudy
    alpha: Annotated[StrictFloat, Field(gt=0, lt=1)]

    def _validate_overlap(self) -> Self:
        super()._validate_overlap()
        entry = self.focal.entry
        if entry.active_from > self.entry_open or entry.active_until < self.end:
            self._invalid_field("focal", "incompatible_window")
        return self


class _InventoryRoster(_ScanModel):
    universe_id: Identity
    source_id: Identity
    version: Identity | None = None
    digest: Sha256 | None = None
    covered_from: Instant | None = None
    complete_through: Instant | None = None
    coverage: ScanCoverage = "not_captured"
    entries: Ordered[tuple[InventoryEntry, ...]]

    def _validate_overlap(self) -> Self:
        if len({entry.identity for entry in self.entries}) != len(self.entries):
            self._invalid_field("entries", "duplicate_identity")
        if self.coverage == "not_captured" and self.entries:
            self._invalid_field("coverage", "unrecorded_entries")
        if (
            self.covered_from is not None
            and self.complete_through is not None
            and self.covered_from > self.complete_through
        ):
            self._invalid_field("complete_through", "coverage_out_of_order")
        entries = sorted(self.entries, key=lambda entry: entry.identity.sort_key)
        object.__setattr__(self, "entries", tuple(entries))
        return self

    def _covers(self, scope: ScanScope) -> bool:
        return (
            self.coverage == "complete"
            and (self.universe_id, self.source_id) == (scope.universe_id, scope.source_id)
            and self.version is not None
            and self.digest is not None
            and self.covered_from is not None
            and self.covered_from <= scope.entry_open
            and self.complete_through is not None
            and self.complete_through >= scope.end
        )


class InteractionInventory(_InventoryRoster):
    """Source-declared schedule, not a receipt for observations at the look."""

    frozen_at: Instant | None = None

    def coverage_for(self, scope: ScanScope) -> ScanCoverage:
        """Return scoped roster coverage without claiming that data arrived."""
        if self.coverage == "not_captured":
            return "not_captured"
        if self._covers(scope) and self.frozen_at is not None and self.frozen_at < scope.entry_open:
            return "complete"
        return "partial"


class ObservedInventory(_InventoryRoster):
    """Independent outcome-free inventory receipt as of the predeclared look."""

    cutoff: Instant

    def _validate_overlap(self) -> Self:
        super()._validate_overlap()
        if any(
            t is not None and t > self.cutoff for t in (self.covered_from, self.complete_through)
        ):
            self._invalid_field("cutoff", "coverage_after_cutoff")
        return self

    def coverage_for(self, scope: ScanScope) -> ScanCoverage:
        """Return observed coverage for exactly this look; a later look is different."""
        scope.require_cutoff(self.cutoff)
        if self.coverage == "not_captured":
            return "not_captured"
        return "complete" if self._covers(scope) else "partial"


class PairKey(_ScanModel):
    """Directional focal/candidate identity in the named inventory source."""

    universe_id: Identity
    source_id: Identity
    focal: ExperimentIdentity
    candidate: ExperimentIdentity

    @property
    def pair_id(self) -> str:
        """Stable selection key, independent of names and observed results."""
        return overlap_digest(
            "plan", {"kind": "interaction_scan_pair.v1", **self.model_dump(mode="json")}
        )


class ScanReason(_ScanModel):
    """Machine-readable reason, optionally specific to an assignment cell."""

    code: Identity
    cell: CellId | None = None


class ScanCandidate(_ScanModel):
    """A visible pair and any exclusions determined without looking at outcomes."""

    key: PairKey
    entry: InventoryEntry
    reasons: Ordered[tuple[ScanReason, ...]]

    def _validate_overlap(self) -> Self:
        if self.key.candidate != self.entry.identity or self.key.candidate == self.key.focal:
            self._invalid_field("key", "incompatible_identity")
        return self


class ScanMemberKey(_ScanModel):
    """One of three slots per candidate and planned metric."""

    pair: PairKey
    metric: Identity
    contrast: ScanContrast


def _in_window(entry: InventoryEntry, start: datetime, end: datetime) -> bool:
    return entry.active_from < end and start < entry.active_until


def _regime(entry: InventoryEntry) -> tuple[str, str]:
    return entry.identity.target, entry.identity.experiment_id


def _same_regime(left: InventoryEntry, right: InventoryEntry) -> bool:
    fields = {"display_name", "reasons"}
    return left.model_dump(exclude=fields) == right.model_dump(exclude=fields)


def _keys(pair: PairKey, metrics: tuple[OverlapMetric, ...]) -> tuple[ScanMemberKey, ...]:
    return tuple(
        ScanMemberKey(pair=pair, metric=metric.name, contrast=contrast)
        for metric in metrics
        for contrast in SCAN_CONTRASTS
    )


class InteractionScanPlan(_ScanModel):
    """A canonical outcome-blind family, even when some pairs cannot be measured."""

    scope: ScanScope
    inventory: InteractionInventory
    metrics: Ordered[tuple[OverlapMetric, ...]]

    def _validate_overlap(self) -> Self:
        scope, inventory = self.scope, self.inventory
        if (scope.universe_id, scope.source_id) != (inventory.universe_id, inventory.source_id):
            self._invalid_field("inventory", "source_mismatch")
        if not self.metrics or len({metric.name for metric in self.metrics}) != len(self.metrics):
            self._invalid_field("metrics", "missing_or_duplicate_metric")
        focal = scope.focal.entry
        for entry in inventory.entries:
            if entry.identity == focal.identity and not _same_regime(focal, entry):
                self._invalid_field("focal", "inventory_conflict")
        ordered = sorted(self.metrics, key=lambda metric: (metric.definition_sha256, metric.name))
        object.__setattr__(self, "metrics", tuple(ordered))
        return self

    @property
    def roster_coverage(self) -> ScanCoverage:
        """Coverage of the frozen roster, separate from coverage at the look."""
        return self.inventory.coverage_for(self.scope)

    @property
    def candidates(self) -> tuple[ScanCandidate, ...]:
        """All concurrent candidates, including incompatible assignment regimes."""
        return self.candidates_for(self.inventory)

    def candidates_for(
        self, inventory: InteractionInventory | ObservedInventory
    ) -> tuple[ScanCandidate, ...]:
        """Apply the same outcome-free scope to the observed or the frozen roster."""
        scope, focal = self.scope, self.scope.focal.entry
        regimes: dict[tuple[str, str], set[str]] = {_regime(focal): {focal.identity.revision}}
        focal_reasons = focal.reasons
        for entry in inventory.entries:
            if _in_window(entry, scope.entry_open, scope.end):
                regimes.setdefault(_regime(entry), set()).add(entry.identity.revision)
            if entry.identity == focal.identity:
                focal_reasons += entry.reasons
        result = []
        for entry in inventory.entries:
            if entry.identity == focal.identity or not _in_window(entry, scope.start, scope.end):
                continue
            reasons = [*entry.reasons, *focal_reasons]
            if entry.active_from > scope.entry_open or entry.active_until < scope.end:
                reasons.append("incompatible_window")
            if len(regimes[_regime(entry)]) > 1 or len(regimes[_regime(focal)]) > 1:
                reasons.append("allocation_switch")
            if entry.unit_namespace != focal.unit_namespace:
                reasons.append("unit_namespace_mismatch")
            if len(entry.arms) != 2:
                reasons.append("unsupported_arms")
            if entry.archive_sha256 is None:
                reasons.append("archive_missing")
            key = PairKey(
                universe_id=scope.universe_id,
                source_id=scope.source_id,
                focal=focal.identity,
                candidate=entry.identity,
            )
            result.append(
                ScanCandidate(
                    key=key,
                    entry=entry,
                    reasons=tuple(ScanReason(code=code) for code in dict.fromkeys(reasons)),
                )
            )
        return tuple(result)

    def matches_observed_roster(self, observed: ObservedInventory) -> bool:
        """Check for drift in candidate identity or declared scope, ignoring names."""
        focal = self.scope.focal.entry
        if any(
            entry.identity == focal.identity and not _same_regime(focal, entry)
            for entry in observed.entries
        ):
            return False
        frozen, actual = self.candidates, self.candidates_for(observed)
        return len(frozen) == len(actual) and all(
            _same_regime(a.entry, b.entry) and a.reasons == b.reasons
            for a, b in zip(frozen, actual, strict=True)
        )

    @property
    def family_members(self) -> tuple[ScanMemberKey, ...]:
        """Canonical candidate, metric-definition and contrast order."""
        return tuple(
            key for candidate in self.candidates for key in _keys(candidate.key, self.metrics)
        )

    @property
    def family_size(self) -> int:
        """Fixed divisor, zero when the roster has no candidates."""
        return len(self.candidates) * len(self.metrics) * len(SCAN_CONTRASTS)

    @property
    def plan_id(self) -> str:
        """Bind the outcome-free declaration without binding presentation labels."""
        return overlap_digest(
            "plan",
            {
                "kind": "interaction_scan.v1",
                "scope": self.scope.model_dump(
                    mode="json", exclude={"focal": {"entry": {"display_name"}}}
                ),
                "inventory": self.inventory.model_dump(
                    mode="json", exclude={"entries": {"__all__": {"display_name"}}}
                ),
                "metrics": [metric.model_dump(mode="json") for metric in self.metrics],
            },
        )

    @property
    def family_id(self) -> str:
        """Identify the whole fixed family, never an original pair family."""
        return overlap_digest(
            "family",
            {
                "kind": "interaction_scan.v1",
                "plan_id": self.plan_id,
                "members": [member.model_dump(mode="json") for member in self.family_members],
            },
        )


def freeze_interaction_scan(
    *,
    inventory: InteractionInventory,
    focal: FocalStudy,
    metrics: tuple[OverlapMetric, ...],
    entry_open: datetime,
    start: datetime,
    end: datetime,
    cutoff: datetime,
    alpha: float,
) -> InteractionScanPlan:
    """Freeze a family from an inventory and study declaration, without outcomes."""
    scope = ScanScope(
        universe_id=inventory.universe_id,
        source_id=inventory.source_id,
        focal=focal,
        entry_open=entry_open,
        start=start,
        end=end,
        cutoff=cutoff,
        alpha=alpha,
    )
    return InteractionScanPlan(scope=scope, inventory=inventory, metrics=metrics)


class MetricIssue(_ScanModel):
    """Reasons why a particular planned metric cannot use an otherwise matched pair."""

    metric: Identity
    reasons: Ordered[tuple[ScanReason, ...]]


def _metric_issues(scan: InteractionScanPlan, pair: OverlapPlan) -> tuple[MetricIssue, ...]:
    by_name = {metric.name: metric for metric in pair.metrics}
    issues = []
    for metric in scan.metrics:
        actual = by_name.get(metric.name)
        if actual is None:
            code = "metric_missing"
        elif (actual.definition_sha256, actual.unit, actual.kind) != (
            metric.definition_sha256,
            metric.unit,
            metric.kind,
        ):
            code = "metric_mismatch"
        else:
            continue
        issues.append(MetricIssue(metric=metric.name, reasons=(ScanReason(code=code),)))
    return tuple(issues)


class CapturedPair(_ScanModel):
    """Validated bounded overlap evidence, with its original A/B orientation."""

    kind: Literal["captured"] = "captured"
    evidence: OverlapEvidence
    focal_side: FactorId
    metric_issues: Ordered[tuple[MetricIssue, ...]] = ()


class UnavailablePair(_ScanModel):
    """Local evidence gap, independent of the scope's inventory coverage."""

    kind: Literal["unavailable"] = "unavailable"
    reasons: Annotated[Ordered[tuple[ScanReason, ...]], Field(min_length=1)]


class PairInput(_ScanModel):
    """One visible candidate and its bounded evidence or typed unavailability."""

    candidate: ScanCandidate
    value: Annotated[CapturedPair | UnavailablePair, Field(discriminator="kind")]


def _pair_scope_issue(
    scope: ScanScope,
    candidate: ScanCandidate,
    *,
    pair_plan: OverlapPlan,
    focal_side: FactorId,
) -> str | None:
    focal = scope.focal
    if focal_side == "A":
        focal_factor, other_factor = pair_plan.a, pair_plan.b
        focal_archive, other_archive = pair_plan.law.archive_sha256
    else:
        focal_factor, other_factor = pair_plan.b, pair_plan.a
        other_archive, focal_archive = pair_plan.law.archive_sha256
    factors = ((focal.entry, focal_factor), (candidate.entry, other_factor))
    if any(
        entry.identity.sort_key != (factor.target, factor.experiment, factor.revision)
        for entry, factor in factors
    ):
        return "pair_identity_mismatch"
    if any(entry.arms != (factor.off, factor.on) for entry, factor in factors):
        return "arm_mismatch"
    if (focal.entry.archive_sha256, candidate.entry.archive_sha256) != (
        focal_archive,
        other_archive,
    ):
        return "archive_mismatch"
    if not pair_plan.unit_namespace == focal.entry.unit_namespace == candidate.entry.unit_namespace:
        return "unit_namespace_mismatch"
    for actual, expected, reason in (
        (pair_plan.population_id, focal.population_id, "population_mismatch"),
        (
            pair_plan.eligibility_rule_sha256,
            focal.eligibility_rule_sha256,
            "eligibility_rule_mismatch",
        ),
        (
            pair_plan.enrollment_rule_sha256,
            focal.enrollment_rule_sha256,
            "enrollment_rule_mismatch",
        ),
        (pair_plan.target, focal.target, "target_mismatch"),
    ):
        if actual != expected:
            return reason
    if (
        (pair_plan.entry_open, pair_plan.start, pair_plan.end)
        != (scope.entry_open, scope.start, scope.end)
        or candidate.entry.active_from > scope.entry_open
        or candidate.entry.active_until < scope.end
    ):
        return "incompatible_window"
    return None


def _check_exchange(
    plan: InteractionScanPlan,
    observed: ObservedInventory | None,
    coverage: ScanCoverage,
    candidates: tuple[ScanCandidate, ...],
) -> None:
    observed_coverage = "not_captured" if observed is None else observed.coverage_for(plan.scope)
    keys = tuple(candidate.key for candidate in candidates)
    actual = set(keys)
    frozen = {candidate.key: candidate for candidate in plan.candidates}
    expected = frozen.keys()
    if len(actual) != len(keys) or not expected <= actual:
        refuse(SCAN_INVALID, field="candidates", reason="duplicate_or_missing_candidate")
    scope = plan.scope
    if any(
        (key.universe_id, key.source_id, key.focal)
        != (scope.universe_id, scope.source_id, scope.focal.entry.identity)
        for key in keys
    ):
        refuse(SCAN_INVALID, field="candidates", reason="scope_mismatch")
    for candidate in candidates:
        original = frozen.get(candidate.key)
        if original is not None and (
            not _same_regime(original.entry, candidate.entry)
            or original.entry.reasons != candidate.entry.reasons
            or original.reasons != candidate.reasons
        ):
            refuse(SCAN_INVALID, field="candidates", reason="frozen_candidate_mismatch")
    if coverage == "complete" and (
        plan.roster_coverage != "complete"
        or observed_coverage != "complete"
        or expected != actual
        or observed is None
        or not plan.matches_observed_roster(observed)
    ):
        refuse(SCAN_INVALID, field="coverage", reason="unproved_complete")


class ScanCapture(_ScanModel):
    """One fixed-look exchange; no raw source rows, even when coverage is partial."""

    plan: InteractionScanPlan
    observed_inventory: ObservedInventory | None
    coverage: ScanCoverage
    pairs: Ordered[tuple[PairInput, ...]]

    def _validate_overlap(self) -> Self:
        _check_exchange(
            self.plan,
            self.observed_inventory,
            self.coverage,
            tuple(pair.candidate for pair in self.pairs),
        )
        for pair in self.pairs:
            if isinstance(pair.value, CapturedPair):
                self.plan.scope.require_cutoff(pair.value.evidence.capture.cutoff)
                issue = _pair_scope_issue(
                    self.plan.scope,
                    pair.candidate,
                    pair_plan=pair.value.evidence.plan,
                    focal_side=pair.value.focal_side,
                )
                if issue is not None:
                    self._invalid_field("pairs", issue)
                if pair.value.metric_issues != _metric_issues(self.plan, pair.value.evidence.plan):
                    self._invalid_field("metric_issues", "metric_issue_mismatch")
        return self

    @property
    def capture_id(self) -> str:
        """Bind this look to the observed roster and original bounded receipts."""
        observed = self.observed_inventory
        pairs = []
        for pair in self.pairs:
            value = pair.value
            pairs.append(
                {
                    "candidate": pair.candidate.model_dump(
                        mode="json", exclude={"entry": {"display_name"}}
                    ),
                    "input": value.model_dump(mode="json", exclude={"evidence"}),
                    "evidence_sha256": (
                        value.evidence.evidence_sha256 if isinstance(value, CapturedPair) else None
                    ),
                }
            )
        return overlap_digest(
            "evidence",
            {
                "kind": "interaction_scan.v1",
                "plan_id": self.plan.plan_id,
                "coverage": self.coverage,
                "observed": (
                    observed.model_dump(
                        mode="json", exclude={"entries": {"__all__": {"display_name"}}}
                    )
                    if observed is not None
                    else None
                ),
                "pairs": pairs,
            },
        )

    @property
    def no_candidates(self) -> bool:
        """Whether the named inventory supports a genuine empty result."""
        return self.coverage == "complete" and not self.pairs


class ScanCellCounts(_ScanModel):
    """Denominators associated with a contrast, without unit-level data."""

    cell: CellId
    n_assigned: Count
    n_observed: Count
    n_missing: Count
    n_pending: Count


class ScanRow(_ScanModel):
    """One raw-scale contrast with nullable numbers and cell-specific reasons."""

    key: ScanMemberKey
    estimate: _Finite | None
    se: _Positive | None
    df: _Positive | None
    lower: _Finite | None
    upper: _Finite | None
    availability: ScanAvailability
    reasons: Ordered[tuple[ScanReason, ...]]
    cells: Ordered[tuple[ScanCellCounts, ...]]
    evidence_sha256: Sha256 | None
    decision: ScanDecision | None

    def _validate_overlap(self) -> Self:
        if self.availability == "unavailable" and any(
            number is not None
            for number in (self.estimate, self.se, self.df, self.lower, self.upper)
        ):
            self._invalid_field("estimate", "numbers_on_unavailable_row")
        if (self.lower is None) != (self.upper is None):
            self._invalid_field("lower", "incomplete_interval")
        if self.decision is not None and (self.availability != "decided" or self.lower is None):
            self._invalid_field("decision", "unsupported_decision")
        return self


class PairEvidenceRef(_ScanModel):
    """Original pair identities, apart from the scan's plan and family IDs."""

    study_id: Identity
    generation_id: Identity
    plan_sha256: Sha256
    family_sha256: Sha256
    evidence_sha256: Sha256


class PairDetail(_ScanModel):
    """Bounded four-cell data and diagnostics for an expanded candidate."""

    candidate: ScanCandidate
    status: Literal["captured", "unavailable"]
    reasons: Ordered[tuple[ScanReason, ...]]
    focal_side: FactorId | None
    evidence: PairEvidenceRef | None
    eligibility_counts: Ordered[tuple[Count, Count, Count, Count]] | None
    gate_counts: Ordered[tuple[Count, ...]] | None
    delivery_counts: Ordered[tuple[OverlapDeliveryCount, ...]]
    cells: Ordered[tuple[OverlapCell, ...]]


class InteractionScan(_ScanModel):
    """The in-memory readout, with the same frozen family for every view."""

    plan: InteractionScanPlan
    capture_id: Sha256
    observed_inventory: ObservedInventory | None
    coverage: ScanCoverage
    rows: Ordered[tuple[ScanRow, ...]]
    details: Ordered[tuple[PairDetail, ...]]

    def _validate_overlap(self) -> Self:
        candidates = tuple(detail.candidate for detail in self.details)
        _check_exchange(self.plan, self.observed_inventory, self.coverage, candidates)
        expected = tuple(
            key for candidate in candidates for key in _keys(candidate.key, self.plan.metrics)
        )
        if tuple(row.key for row in self.rows) != expected:
            self._invalid_field("rows", "not_in_candidate_metric_contrast_order")
        if self.coverage != "complete" and any(row.lower is not None for row in self.rows):
            self._invalid_field("rows", "interval_without_complete_coverage")
        return self

    @property
    def no_candidates(self) -> bool:
        """Whether the named inventory supports a genuine empty result."""
        return self.coverage == "complete" and not self.details


class SourceFact(_ScanModel):
    """One qualified (catalog, schema, relation) coordinate and content digest."""

    coordinate: Ordered[tuple[Identity, Identity, Identity]]
    content_sha256: Sha256


class StudyFeedFact(_ScanModel):
    """Map a logical pair feed to its sealed physical contents."""

    study_id: Identity
    role: FeedRole
    fact: SourceFact


class SourceGenerationManifest(_ScanModel):
    """Source declaration of one sealed dataset, not proof or a saved scan."""

    source_id: Identity
    generation_id: Identity
    dataset_sha256: Sha256
    cutoff: Instant
    scan_plan_id: Sha256
    definition_sha256: Sha256
    focal_assignments: SourceFact
    focal_exposures: SourceFact | None = None
    focal_outcomes: Ordered[tuple[tuple[Identity, SourceFact], ...]]
    pair_feeds: Ordered[tuple[StudyFeedFact, ...]]

    def _validate_overlap(self) -> Self:
        if len({metric for metric, _ in self.focal_outcomes}) != len(self.focal_outcomes):
            self._invalid_field("focal_outcomes", "duplicate_metric")
        object.__setattr__(self, "focal_outcomes", tuple(sorted(self.focal_outcomes)))
        if len({(feed.study_id, feed.role) for feed in self.pair_feeds}) != len(self.pair_feeds):
            self._invalid_field("pair_feeds", "duplicate_feed")
        feeds = sorted(
            self.pair_feeds, key=lambda feed: (feed.study_id, FEED_ROLES.index(feed.role))
        )
        object.__setattr__(self, "pair_feeds", tuple(feeds))
        return self


@dataclass(frozen=True, slots=True)
class FocalSourceFacts:
    """Verified focal data and the planned metrics' effective input horizons."""

    generation_id: str
    dataset_sha256: str
    assignments: SourceFact
    exposures: SourceFact
    outcomes: tuple[tuple[str, SourceFact], ...]
    horizons: tuple[tuple[str, datetime | None, date, date | None], ...]


class FocalBindingReceipt(_ScanModel):
    """Focal binding values to be checked independently against the pinned reader."""

    scan_plan_id: Sha256
    capture_id: Sha256
    definition_sha256: Sha256
    metrics_sha256: Sha256
    identity: ExperimentIdentity
    arms: Ordered[tuple[Identity, Identity]]
    unit_namespace: Identity
    population: Literal["assigned"]
    window: ScanWindow
    source_id: Identity
    generation_id: Identity
    dataset_sha256: Sha256
    focal_assignment_sha256: Sha256
    focal_outcome_sha256: Sha256
    pair_evidence_sha256: Ordered[tuple[Sha256, ...]]

    def _validate_overlap(self) -> Self:
        if self.arms[0] == self.arms[1]:
            self._invalid_field("arms", "duplicate_arm")
        if len(set(self.pair_evidence_sha256)) != len(self.pair_evidence_sha256):
            self._invalid_field("pair_evidence_sha256", "duplicate_evidence")
        object.__setattr__(self, "pair_evidence_sha256", tuple(sorted(self.pair_evidence_sha256)))
        return self


class InteractionAttachment(_ScanModel):
    """The scan and focal receipt passed from binding to dashboard preparation."""

    scan: InteractionScan
    focal_receipt: FocalBindingReceipt

    def _validate_overlap(self) -> Self:
        receipt, plan = self.focal_receipt, self.scan.plan
        scope, focal = plan.scope, plan.scope.focal.entry
        scope.require_cutoff(receipt.window.cutoff)
        if (
            receipt.scan_plan_id != plan.plan_id
            or receipt.capture_id != self.scan.capture_id
            or (receipt.identity, receipt.arms, receipt.unit_namespace)
            != (focal.identity, focal.arms, focal.unit_namespace)
            or (receipt.window.entry_open, receipt.window.start, receipt.window.end)
            != (scope.entry_open, scope.start, scope.end)
        ):
            self._invalid_field("focal_receipt", "scope_mismatch")
        return self
