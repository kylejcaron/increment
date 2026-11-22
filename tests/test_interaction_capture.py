"""Bind bounded joint evidence to the frozen and observed experiment rosters."""

from dataclasses import dataclass
from datetime import timedelta

import pytest

from increment.errors import CodedError
from increment.overlap import capture_interaction_scan
from increment.semantics.interaction_scan import (
    CapturedPair,
    ExperimentIdentity,
    FocalStudy,
    InteractionInventory,
    InteractionScanPlan,
    InventoryEntry,
    MetricIssue,
    ObservedInventory,
    ScanReason,
    UnavailablePair,
    freeze_interaction_scan,
)
from increment.semantics.overlap import CellId, OverlapEvidence, OverlapFactor, OverlapPlan
from tests.test_interaction_scan_contract import restored_pair as restored_pair


def _entry(pair: OverlapPlan, factor: OverlapFactor, archive: str) -> InventoryEntry:
    return InventoryEntry(
        identity=ExperimentIdentity(
            target=factor.target, experiment_id=factor.experiment, revision=factor.revision
        ),
        archive_sha256=archive,
        unit_namespace=pair.unit_namespace,
        active_from=pair.entry_open,
        active_until=pair.end,
        arms=(factor.off, factor.on),
        display_name="Same display name",
    )


def scan_for_pair(evidence: OverlapEvidence) -> InteractionScanPlan:
    """Freeze a look on the same window as the original pair, with A as focal."""
    pair = evidence.plan
    a, b = pair.law.archive_sha256
    focal = _entry(pair, pair.a, a)
    return freeze_interaction_scan(
        inventory=InteractionInventory(
            universe_id="storefront",
            source_id="experiment-registry",
            version="published.v1",
            digest="f" * 64,
            covered_from=pair.entry_open,
            complete_through=pair.end,
            frozen_at=pair.registered_at,
            coverage="complete",
            entries=(focal, _entry(pair, pair.b, b)),
        ),
        focal=FocalStudy(
            entry=focal,
            population_id=pair.population_id,
            eligibility_rule_sha256=pair.eligibility_rule_sha256,
            enrollment_rule_sha256=pair.enrollment_rule_sha256,
            target=pair.target,
        ),
        metrics=pair.metrics,
        entry_open=pair.entry_open,
        start=pair.start,
        end=pair.end,
        cutoff=evidence.capture.cutoff,
        alpha=pair.alpha,
    )


def observed_at_look(plan: InteractionScanPlan) -> ObservedInventory:
    inventory = plan.inventory
    return ObservedInventory(
        universe_id=inventory.universe_id,
        source_id=inventory.source_id,
        version="observed.v2",
        digest="e" * 64,
        covered_from=inventory.covered_from,
        complete_through=inventory.complete_through,
        cutoff=plan.scope.cutoff,
        coverage="complete",
        entries=inventory.entries,
    )


def _with_plan(
    evidence: OverlapEvidence, plan: OverlapPlan, *, generation: str | None = None
) -> OverlapEvidence:
    capture = evidence.capture.model_copy(
        update={
            "law_sha256": plan.law_sha256,
            "archive_sha256": plan.law.archive_sha256,
            "generation_id": generation or evidence.capture.generation_id,
        }
    )
    return evidence.model_copy(update={"plan": plan, "capture": capture})


@dataclass(frozen=True)
class _PublishedPairs:
    plan: InteractionScanPlan
    forward: OverlapEvidence
    reversed: OverlapEvidence

    @property
    def evidence_by_study(self) -> dict[str, OverlapEvidence]:
        return {e.plan.study_id: e for e in (self.reversed, self.forward)}


@pytest.fixture
def published_pairs(restored_pair) -> _PublishedPairs:
    _, forward = restored_pair
    original = forward.plan
    other = original.b.model_copy(
        update={"target": "target-c", "experiment": "experiment-c", "revision": "rev-c"}
    )
    a_archive = original.law.archive_sha256[0]
    other_archive = "c" * 64
    law = original.law.model_copy(update={"archive_sha256": (other_archive, a_archive)})
    reverse_plan = original.model_copy(
        update={"study_id": "study-c", "a": other, "b": original.a, "law": law}
    )
    reversed_evidence = _with_plan(forward, reverse_plan, generation="study-c-generation")
    plan = scan_for_pair(forward)
    unnamed = other.model_copy(
        update={"target": "target-d", "experiment": "experiment-d", "revision": "rev-d"}
    )
    inventory = plan.inventory.model_copy(
        update={
            "entries": (
                *plan.inventory.entries,
                _entry(original, other, other_archive),
                _entry(original, unnamed, "d" * 64),
            )
        }
    )
    return _PublishedPairs(
        plan.model_copy(update={"inventory": inventory}), forward, reversed_evidence
    )


def test_complete_inventory_keeps_unmeasured_candidate_and_native_orientations(published_pairs):
    receipts = published_pairs
    plan = receipts.plan
    captured = capture_interaction_scan(
        plan,
        observed_inventory=observed_at_look(plan),
        evidence_by_study=receipts.evidence_by_study,
    )
    assert captured.coverage == "complete"
    assert captured.plan.family_size == 18
    assert [pair.candidate.key.candidate.experiment_id for pair in captured.pairs] == [
        "experiment-b",
        "experiment-c",
        "experiment-d",
    ]
    b, c, d = (pair.value for pair in captured.pairs)
    assert isinstance(b, CapturedPair) and isinstance(c, CapturedPair)
    assert (b.focal_side, c.focal_side) == ("A", "B")
    assert [p.evidence.capture.generation_id for p in (b, c)] == [
        "snapshot-v1",
        "study-c-generation",
    ]
    assert [p.evidence.evidence_sha256 for p in (b, c)] == [
        receipts.forward.evidence_sha256,
        receipts.reversed.evidence_sha256,
    ]
    assert d == UnavailablePair(reasons=(ScanReason(code="pair_evidence_missing"),))


def test_inventory_drift_retains_both_rosters_without_moving_the_family(published_pairs):
    receipts = published_pairs
    plan = receipts.plan
    observed = observed_at_look(plan)
    a, b, c, d = observed.entries
    extra = d.model_copy(
        update={
            "identity": ExperimentIdentity(
                target="target-e", experiment_id="experiment-e", revision="rev-e"
            )
        }
    )
    changed = d.model_copy(update={"archive_sha256": "0" * 64})
    observed = observed.model_copy(update={"entries": (a, c, changed, extra)})
    plan = plan.model_copy(
        update={"inventory": plan.inventory.model_copy(update={"entries": (a, b, d)})}
    )
    captured = capture_interaction_scan(
        plan,
        observed_inventory=observed,
        evidence_by_study=receipts.evidence_by_study,
    )
    assert captured.coverage == "partial"
    assert captured.observed_inventory == observed
    assert captured.plan.family_id == plan.family_id
    assert [pair.candidate.key.candidate.experiment_id for pair in captured.pairs] == [
        "experiment-b",
        "experiment-c",
        "experiment-d",
        "experiment-e",
    ]
    assert isinstance(captured.pairs[0].value, CapturedPair)
    assert isinstance(captured.pairs[1].value, CapturedPair)
    assert captured.pairs[2].candidate.entry.archive_sha256 == "d" * 64


def test_observed_exclusions_apply_to_supplied_and_missing_pairs_without_rewriting_family(
    published_pairs,
):
    plan = published_pairs.plan
    observed = observed_at_look(plan)
    a, b, c, d = observed.entries
    same_layer = b.model_copy(update={"reasons": ("same_layer",)})
    new_revision = d.model_copy(
        update={"identity": d.identity.model_copy(update={"revision": "v2"})}
    )
    observed = observed.model_copy(update={"entries": (a, same_layer, c, d, new_revision)})
    capture = capture_interaction_scan(
        plan,
        observed_inventory=observed,
        evidence_by_study=published_pairs.evidence_by_study,
    )
    assert capture.coverage == "partial"
    assert capture.plan.family_id == plan.family_id
    supplied, peer, missing, added = capture.pairs
    assert (supplied.candidate, peer.candidate, missing.candidate) == plan.candidates
    assert supplied.value == UnavailablePair(reasons=(ScanReason(code="same_layer"),))
    assert isinstance(peer.value, CapturedPair)
    for pair in (missing, added):
        assert pair.value == UnavailablePair(reasons=(ScanReason(code="allocation_switch"),))


def test_published_roster_alone_cannot_certify_no_candidates(restored_pair):
    plan, _ = restored_pair
    plan = plan.model_copy(
        update={
            "inventory": plan.inventory.model_copy(update={"entries": (plan.scope.focal.entry,)})
        }
    )
    captured = capture_interaction_scan(plan, observed_inventory=None, evidence_by_study={})
    assert captured.plan.roster_coverage == "complete"
    assert captured.coverage == "not_captured"
    assert not captured.no_candidates


def test_observed_roster_coverage_gap_does_not_erase_pair_evidence(restored_pair):
    plan, evidence = restored_pair
    observed = observed_at_look(plan).model_copy(update={"complete_through": plan.scope.start})
    captured = capture_interaction_scan(
        plan, observed_inventory=observed, evidence_by_study={evidence.plan.study_id: evidence}
    )
    assert captured.coverage == "partial"
    assert isinstance(captured.pairs[0].value, CapturedPair)


def test_observed_inventory_cannot_supply_candidates_from_another_source(restored_pair):
    plan, _ = restored_pair
    foreign = observed_at_look(plan).model_copy(update={"source_id": "foreign-registry"})
    with pytest.raises(CodedError) as caught:
        capture_interaction_scan(plan, observed_inventory=foreign, evidence_by_study={})
    assert caught.value.code == "overlap.scan.invalid"
    assert caught.value.context == {"field": "observed_inventory", "reason": "source_mismatch"}


def test_unavailable_roster_and_pair_windows_keep_exact_reasons(published_pairs):
    receipts = published_pairs
    plan = receipts.plan
    a, b, c, d = plan.inventory.entries
    same_layer = d.model_copy(update={"reasons": ("same_layer",)})
    plan = plan.model_copy(
        update={"inventory": plan.inventory.model_copy(update={"entries": (a, b, c, same_layer)})}
    )
    pair = receipts.forward.plan
    off_window = _with_plan(
        receipts.forward, pair.model_copy(update={"end": pair.end - timedelta(microseconds=1)})
    )
    captured = capture_interaction_scan(
        plan,
        observed_inventory=observed_at_look(plan),
        evidence_by_study={pair.study_id: off_window},
    )
    assert captured.pairs[0].value == UnavailablePair(
        reasons=(ScanReason(code="incompatible_window"),)
    )
    assert captured.pairs[2].value == UnavailablePair(reasons=(ScanReason(code="same_layer"),))


def test_metric_definition_and_unit_mismatch_only_excludes_affected_metrics(restored_pair):
    plan, evidence = restored_pair
    orders, complaint = evidence.plan.metrics
    other_order = orders.model_copy(update={"name": "checkouts"})
    plan = plan.model_copy(
        update={
            "metrics": (
                orders.model_copy(update={"definition_sha256": "0" * 64}),
                complaint.model_copy(update={"unit": "percent"}),
                other_order,
            )
        }
    )
    captured = capture_interaction_scan(
        plan,
        observed_inventory=observed_at_look(plan),
        evidence_by_study={evidence.plan.study_id: evidence},
    )
    value = captured.pairs[0].value
    assert isinstance(value, CapturedPair)
    assert {issue.metric: issue for issue in value.metric_issues} == {
        "orders": MetricIssue(metric="orders", reasons=(ScanReason(code="metric_mismatch"),)),
        "complaint": MetricIssue(metric="complaint", reasons=(ScanReason(code="metric_mismatch"),)),
        "checkouts": MetricIssue(metric="checkouts", reasons=(ScanReason(code="metric_missing"),)),
    }


def test_duplicate_pair_generations_are_not_selected_by_arrival_order(restored_pair):
    plan, evidence = restored_pair
    duplicate = _with_plan(
        evidence,
        evidence.plan.model_copy(update={"study_id": "second-study"}),
        generation="second-generation",
    )
    with pytest.raises(CodedError) as caught:
        capture_interaction_scan(
            plan,
            observed_inventory=None,
            evidence_by_study={
                evidence.plan.study_id: evidence,
                duplicate.plan.study_id: duplicate,
            },
        )
    assert caught.value.code == "overlap.scan.invalid"
    assert caught.value.context == {"field": "evidence_by_study", "reason": "duplicate_pair"}


@pytest.mark.parametrize("wrong_identity", ["study_id", "focal_target", "candidate_revision"])
def test_pair_binding_rejects_wrong_identity(restored_pair, wrong_identity):
    plan, evidence = restored_pair
    pair = evidence.plan
    key = pair.study_id
    if wrong_identity == "study_id":
        key = "renamed-file.json"
        reason = "study_id_mismatch"
    elif wrong_identity == "focal_target":
        a = pair.a.model_copy(update={"target": "new-target"})
        evidence = _with_plan(evidence, pair.model_copy(update={"a": a}))
        reason = "focal_identity_mismatch"
    else:
        b = pair.b.model_copy(update={"revision": "new-revision"})
        evidence = _with_plan(evidence, pair.model_copy(update={"b": b}))
        reason = "unknown_candidate"
    with pytest.raises(CodedError) as caught:
        capture_interaction_scan(plan, observed_inventory=None, evidence_by_study={key: evidence})
    assert caught.value.code == "overlap.scan.invalid"
    assert caught.value.context == {"field": "evidence_by_study", "reason": reason}


@pytest.mark.parametrize(
    ("field", "new_value", "reason"),
    [
        ("archive", "0" * 64, "archive_mismatch"),
        ("off", "placebo", "arm_mismatch"),
        ("unit_namespace", "other-units", "unit_namespace_mismatch"),
        ("population_id", "other-population", "population_mismatch"),
        ("eligibility_rule_sha256", "0" * 64, "eligibility_rule_mismatch"),
        ("enrollment_rule_sha256", "0" * 64, "enrollment_rule_mismatch"),
        ("target", "jointly_eligible_and_enrolled", "target_mismatch"),
    ],
)
def test_pair_rejects_conflicting_receipts(restored_pair, field, new_value, reason):
    plan, evidence = restored_pair
    pair = evidence.plan
    if field == "archive":
        law = pair.law.model_copy(
            update={"archive_sha256": (new_value, pair.law.archive_sha256[1])}
        )
        edits = {"law": law}
    elif field == "off":
        edits = {"b": pair.b.model_copy(update={"off": new_value})}
    else:
        edits = {field: new_value}
    conflict = _with_plan(evidence, pair.model_copy(update=edits))
    with pytest.raises(CodedError) as caught:
        capture_interaction_scan(
            plan, observed_inventory=None, evidence_by_study={pair.study_id: conflict}
        )
    assert caught.value.code == "overlap.scan.invalid"
    assert caught.value.context == {"field": "evidence_by_study", "reason": reason}


def test_exact_cutoff_is_required_even_when_the_pair_covers_the_window(restored_pair):
    plan, evidence = restored_pair
    later = evidence.capture.cutoff + timedelta(microseconds=1)
    later_evidence = evidence.model_copy(
        update={"capture": evidence.capture.model_copy(update={"cutoff": later})}
    )
    with pytest.raises(CodedError) as caught:
        capture_interaction_scan(
            plan,
            observed_inventory=None,
            evidence_by_study={evidence.plan.study_id: later_evidence},
        )
    assert caught.value.code == "overlap.scan.cutoff_mismatch"
    assert caught.value.context == {
        "expected_cutoff": "2026-01-01T00:00:11.000000Z",
        "actual_cutoff": "2026-01-01T00:00:11.000001Z",
    }


def test_counts_keep_pending_missing_empty_and_observed_zero_distinct(restored_pair):
    plan, evidence = restored_pair

    def capture_with(
        bundle: OverlapEvidence,
    ) -> dict[CellId, tuple[int, int, int, int, float | None]]:
        captured = capture_interaction_scan(
            plan,
            observed_inventory=observed_at_look(plan),
            evidence_by_study={bundle.plan.study_id: bundle},
        )
        value = captured.pairs[0].value
        assert isinstance(value, CapturedPair)
        return {
            c.cell: (c.n_assigned, c.n_observed, c.n_missing, c.n_pending, c.anchor)
            for c in value.evidence.cells
            if c.metric == "complaint"
        }

    missing = {
        "n_observed": 0,
        "n_missing": 1,
        "n_success": 0,
        "anchor": None,
        "mean_delta": None,
        "m2": None,
    }
    cells = tuple(
        c.model_copy(update=missing) if (c.metric, c.cell) == ("complaint", "01") else c
        for c in evidence.cells
    )
    assert capture_with(evidence.model_copy(update={"cells": cells})) == {
        "00": (0, 0, 0, 0, None),
        "10": (1, 1, 0, 0, 0.0),
        "01": (1, 0, 1, 0, None),
        "11": (0, 0, 0, 0, None),
    }
    feeds = tuple(
        f.model_copy(update={"complete_through": None}) if f.role == "outcomes" else f
        for f in evidence.capture.feeds
    )
    pending = evidence.model_copy(
        update={
            "capture": evidence.capture.model_copy(update={"feeds": feeds}),
            "cells": tuple(
                c.model_copy(
                    update={
                        "n_observed": 0,
                        "n_pending": c.n_assigned,
                        "n_success": 0 if c.metric == "complaint" else None,
                        "anchor": None,
                        "mean_delta": None,
                        "m2": None,
                    }
                )
                for c in evidence.cells
            ),
        }
    )
    assert capture_with(pending) == {
        "00": (0, 0, 0, 0, None),
        "10": (1, 0, 0, 1, None),
        "01": (1, 0, 0, 1, None),
        "11": (0, 0, 0, 0, None),
    }
