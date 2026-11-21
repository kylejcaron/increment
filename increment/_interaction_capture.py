"""Assemble bounded joint evidence at the interaction scan's fixed look."""

from collections.abc import Mapping
from typing import NoReturn

from increment._overlap_evidence import FactorId
from increment._overlap_refusals import SCAN_INVALID
from increment.errors import refuse
from increment.semantics.interaction_scan import (
    CapturedPair,
    ExperimentIdentity,
    InteractionScanPlan,
    MetricIssue,
    ObservedInventory,
    PairInput,
    ScanCandidate,
    ScanCapture,
    ScanCoverage,
    ScanReason,
    UnavailablePair,
    _pair_scope_issue,
)
from increment.semantics.overlap import OverlapEvidence, OverlapFactor, OverlapPlan


def _identity(factor: OverlapFactor) -> ExperimentIdentity:
    return ExperimentIdentity(
        target=factor.target, experiment_id=factor.experiment, revision=factor.revision
    )


def _invalid_evidence(reason: str) -> NoReturn:
    refuse(SCAN_INVALID, field="evidence_by_study", reason=reason)


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


def _match_pair(
    scan: InteractionScanPlan,
    candidate: ScanCandidate,
    evidence: OverlapEvidence,
    *,
    focal_side: FactorId,
) -> CapturedPair | UnavailablePair:
    pair = evidence.plan
    issue = _pair_scope_issue(scan.scope, candidate, pair_plan=pair, focal_side=focal_side)
    if issue is not None and issue != "incompatible_window":
        _invalid_evidence(issue)
    reasons = candidate.reasons
    if issue is not None:
        reasons = (*reasons, ScanReason(code=issue))
    if reasons:
        return UnavailablePair(reasons=tuple(dict.fromkeys(reasons)))
    return CapturedPair(
        evidence=evidence,
        focal_side=focal_side,
        metric_issues=_metric_issues(scan, pair),
    )


def _received_pairs(
    plan: InteractionScanPlan,
    candidates: Mapping[ExperimentIdentity, ScanCandidate],
    evidence_by_study: Mapping[str, OverlapEvidence],
) -> dict[ExperimentIdentity, CapturedPair | UnavailablePair]:
    focal = plan.scope.focal.entry.identity
    matched = {}
    for study_id, supplied in evidence_by_study.items():
        # Keep the pair's existing law, feed and cell validation on every input path.
        if not isinstance(supplied, OverlapEvidence):
            _invalid_evidence("invalid_evidence")
        evidence = OverlapEvidence.model_validate(supplied)
        pair = evidence.plan
        if study_id != pair.study_id:
            _invalid_evidence("study_id_mismatch")
        plan.scope.require_cutoff(evidence.capture.cutoff)
        left, right = _identity(pair.a), _identity(pair.b)
        if focal == left:
            identity = right
            focal_side: FactorId = "A"
        elif focal == right:
            identity = left
            focal_side = "B"
        else:
            _invalid_evidence("focal_identity_mismatch")
        candidate = candidates.get(identity)
        if candidate is None:
            _invalid_evidence("unknown_candidate")
        if identity in matched:
            _invalid_evidence("duplicate_pair")
        matched[identity] = _match_pair(plan, candidate, evidence, focal_side=focal_side)
    return matched


def capture_interaction_scan(
    plan: InteractionScanPlan,
    *,
    observed_inventory: ObservedInventory | None,
    evidence_by_study: Mapping[str, OverlapEvidence],
) -> ScanCapture:
    """Match each supplied study receipt to the named inventory at one cutoff.

    The inventory is published before entry and observed independently of the
    outcomes. Each measured pair needs its original configuration receipts,
    study-scoped eligibility roster at common entry, first decisions per
    target, delivery events, and finalized metric outcomes. Frame capture
    relies on the producer's coverage declaration; warehouse capture binds
    qualified native DuckDB tables to their contents. Neither establishes
    upstream inventory completeness or assignment independence from balance.
    """
    coverage: ScanCoverage = "not_captured"
    candidates = {candidate.key.candidate: candidate for candidate in plan.candidates}
    if observed_inventory is not None:
        observed_coverage = observed_inventory.coverage_for(plan.scope)
        if (observed_inventory.universe_id, observed_inventory.source_id) != (
            plan.scope.universe_id,
            plan.scope.source_id,
        ):
            refuse(SCAN_INVALID, field="observed_inventory", reason="source_mismatch")
        if observed_coverage != "not_captured":
            coverage = "partial"
            if (
                observed_coverage == "complete"
                and plan.roster_coverage == "complete"
                and plan.matches_observed_roster(observed_inventory)
            ):
                coverage = "complete"
        for candidate in plan.candidates_for(observed_inventory):
            candidates.setdefault(candidate.key.candidate, candidate)
    matched = _received_pairs(plan, candidates, evidence_by_study)
    pairs = []
    for identity in sorted(candidates, key=lambda identity: identity.sort_key):
        candidate = candidates[identity]
        value = matched.get(identity)
        if value is None:
            value = UnavailablePair(
                reasons=candidate.reasons or (ScanReason(code="pair_evidence_missing"),)
            )
        pairs.append(PairInput(candidate=candidate, value=value))
    return ScanCapture(
        plan=plan, observed_inventory=observed_inventory, coverage=coverage, pairs=tuple(pairs)
    )
