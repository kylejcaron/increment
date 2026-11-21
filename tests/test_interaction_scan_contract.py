"""The outcome-free family and bounded exchange between scan consumers."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from increment.errors import CodedError
from increment.semantics.interaction_scan import (
    CapturedPair,
    ExperimentIdentity,
    FocalStudy,
    InteractionInventory,
    InteractionScan,
    InteractionScanPlan,
    InventoryEntry,
    MetricIssue,
    ObservedInventory,
    PairDetail,
    PairEvidenceRef,
    PairInput,
    ScanCapture,
    ScanCellCounts,
    ScanReason,
    ScanRow,
    UnavailablePair,
    freeze_interaction_scan,
)
from increment.semantics.overlap import OverlapEvidence, OverlapFactor, OverlapMetric

ENTRY = datetime(2026, 1, 2, tzinfo=UTC)
START = ENTRY + timedelta(days=1)
END = START + timedelta(days=7)
LOOK = END + timedelta(days=1)


def inventory_entry(
    experiment_id: str, *, active_from: datetime = ENTRY, active_until: datetime = END
) -> InventoryEntry:
    return InventoryEntry(
        identity=ExperimentIdentity(
            target=f"target-{experiment_id}", experiment_id=experiment_id, revision="r1"
        ),
        archive_sha256="a" * 64,
        unit_namespace="account",
        active_from=active_from,
        active_until=active_until,
        arms=("off", "on"),
        display_name="Same name",
        reasons=(),
    )


FOCAL = FocalStudy(
    entry=inventory_entry("search"),
    population_id="eligible-accounts",
    eligibility_rule_sha256="b" * 64,
    enrollment_rule_sha256="c" * 64,
    target="jointly_eligible",
)
SAVE = OverlapMetric(
    name="save",
    unit="probability",
    definition_sha256="d" * 64,
    kind="binary_mean",
    role="primary",
    better="higher",
)


def declared_inventory(*entries: InventoryEntry) -> InteractionInventory:
    return InteractionInventory(
        universe_id="search-rollouts",
        source_id="allocation-history",
        version="roster-r8",
        digest="e" * 64,
        covered_from=ENTRY,
        complete_through=END,
        frozen_at=ENTRY - timedelta(days=1),
        coverage="complete",
        entries=entries,
    )


def freeze(
    inventory: InteractionInventory,
    *,
    metrics: tuple[OverlapMetric, ...] = (SAVE,),
    end: datetime = END,
    cutoff: datetime = LOOK,
) -> InteractionScanPlan:
    return freeze_interaction_scan(
        inventory=inventory,
        focal=FOCAL,
        metrics=metrics,
        entry_open=ENTRY,
        start=START,
        end=end,
        cutoff=cutoff,
        alpha=0.05,
    )


def observed_inventory(
    inventory: InteractionInventory, *, cutoff: datetime = LOOK
) -> ObservedInventory:
    return ObservedInventory(
        universe_id=inventory.universe_id,
        source_id=inventory.source_id,
        version=inventory.version,
        digest=inventory.digest,
        covered_from=inventory.covered_from,
        complete_through=inventory.complete_through,
        cutoff=cutoff,
        coverage=inventory.coverage,
        entries=inventory.entries,
    )


def test_frozen_family_uses_stable_keys_not_names_order_or_available_pairs():
    cards, tips = inventory_entry("cards"), inventory_entry("tips")
    plan = freeze(declared_inventory(tips, FOCAL.entry, cards))
    renamed_cards = cards.model_copy(update={"display_name": "Cards, renamed"})
    reordered = freeze(declared_inventory(renamed_cards, tips, FOCAL.entry))
    assert plan.plan_id == reordered.plan_id
    assert plan.family_id == reordered.family_id
    assert plan.family_size == 6
    assert [
        (member.pair.candidate.experiment_id, member.metric, member.contrast)
        for member in plan.family_members
    ] == [
        ("cards", "save", "focal_given_other_off"),
        ("cards", "save", "focal_given_other_on"),
        ("cards", "save", "interaction"),
        ("tips", "save", "focal_given_other_off"),
        ("tips", "save", "focal_given_other_on"),
        ("tips", "save", "interaction"),
    ]

    third = inventory_entry("filters").model_copy(update={"reasons": ("unsupported_arms",)})
    expanded = freeze(declared_inventory(tips, cards, third))
    assert expanded.family_size == 9
    assert expanded.candidates[1].reasons == (ScanReason(code="unsupported_arms"),)
    reason = ScanReason(code="pair_evidence_missing")
    captured = ScanCapture(
        plan=expanded,
        observed_inventory=observed_inventory(expanded.inventory),
        coverage="complete",
        pairs=tuple(
            PairInput(candidate=candidate, value=UnavailablePair(reasons=(reason,)))
            for candidate in expanded.candidates
        ),
    )
    assert captured.pairs[1].value.kind == "unavailable"
    assert captured.plan.family_size == 9 and not captured.no_candidates


def test_half_open_overlap_keeps_incompatible_candidates_but_not_touching_windows():
    before = inventory_entry("before", active_from=ENTRY, active_until=START)
    after = inventory_entry("after", active_from=END, active_until=LOOK)
    late = inventory_entry("late", active_from=START)
    old = inventory_entry("switch", active_until=START)
    new = old.model_copy(
        update={"identity": old.identity.model_copy(update={"revision": "r2"}), "active_until": END}
    )
    plan = freeze(declared_inventory(after, late, old, before, new))
    assert [
        (candidate.entry.identity.experiment_id, candidate.reasons) for candidate in plan.candidates
    ] == [
        ("late", (ScanReason(code="incompatible_window"),)),
        ("switch", (ScanReason(code="allocation_switch"),)),
    ]


def test_only_observed_complete_scope_can_certify_no_candidates():
    inventory = declared_inventory()
    plan = freeze(inventory)
    assert plan.roster_coverage == "complete"
    assert plan.family_members == () and plan.family_size == 0
    missing = ScanCapture(plan=plan, observed_inventory=None, coverage="not_captured", pairs=())
    assert not missing.no_candidates

    observed = observed_inventory(inventory)
    checked = ScanCapture(plan=plan, observed_inventory=observed, coverage="complete", pairs=())
    assert checked.no_candidates

    late = freeze(inventory.model_copy(update={"frozen_at": ENTRY}))
    assert late.roster_coverage == "partial"
    with pytest.raises(CodedError) as caught:
        ScanCapture(plan=late, observed_inventory=observed, coverage="complete", pairs=())
    assert caught.value.context == {"field": "coverage", "reason": "unproved_complete"}

    drifted = observed.model_copy(update={"entries": (inventory_entry("surprise"),)})
    with pytest.raises(CodedError) as drift:
        ScanCapture(plan=plan, observed_inventory=drifted, coverage="complete", pairs=())
    assert drift.value.context == {"field": "coverage", "reason": "unproved_complete"}

    absent = freeze(
        InteractionInventory(
            universe_id=inventory.universe_id, source_id=inventory.source_id, entries=()
        )
    )
    assert absent.roster_coverage == "not_captured"
    assert not ScanCapture(
        plan=absent, observed_inventory=observed, coverage="partial", pairs=()
    ).no_candidates


def test_scan_identity_binds_revision_source_metric_definition_window_and_look():
    cards = inventory_entry("cards")
    inventory = declared_inventory(cards)
    plan = freeze(inventory)
    assert all(
        alternate.plan_id != plan.plan_id and alternate.family_id != plan.family_id
        for alternate in (
            freeze(
                declared_inventory(
                    cards.model_copy(
                        update={"identity": cards.identity.model_copy(update={"revision": "r3"})}
                    )
                )
            ),
            freeze(inventory.model_copy(update={"version": "roster-r9"})),
            freeze(inventory, metrics=(SAVE.model_copy(update={"definition_sha256": "f" * 64}),)),
            freeze(inventory, end=END - timedelta(days=1)),
            freeze(inventory, cutoff=LOOK + timedelta(seconds=1)),
        )
    )
    different_unit = SAVE.model_copy(update={"name": "units", "definition_sha256": "0" * 64})
    assert [
        member.metric for member in freeze(inventory, metrics=(SAVE, different_unit)).family_members
    ] == ["units", "units", "units", "save", "save", "save"]


def test_refuses_duplicate_identities_invalid_time_and_a_mixed_look():
    cards = inventory_entry("cards")
    with pytest.raises(CodedError) as duplicate:
        declared_inventory(cards, cards.model_copy(update={"display_name": "Alias"}))
    assert duplicate.value.code == "overlap.scan.invalid"
    assert duplicate.value.context == {"field": "entries", "reason": "duplicate_identity"}
    with pytest.raises(CodedError) as invalid:
        freeze(declared_inventory(cards), cutoff=END - timedelta(seconds=1))
    assert invalid.value.context == {"field": "cutoff", "reason": "cutoff_before_end"}
    with pytest.raises(CodedError) as naive:
        freeze(declared_inventory(cards), cutoff=LOOK.replace(tzinfo=None))
    assert naive.value.context == {"field": "cutoff", "reason": "invalid_value"}

    plan = freeze(declared_inventory())
    observed = observed_inventory(plan.inventory, cutoff=LOOK + timedelta(seconds=1))
    with pytest.raises(CodedError) as mixed:
        ScanCapture(plan=plan, observed_inventory=observed, coverage="partial", pairs=())
    assert mixed.value.code == "overlap.scan.cutoff_mismatch"
    assert mixed.value.context == {
        "expected_cutoff": "2026-01-11T00:00:00.000000Z",
        "actual_cutoff": "2026-01-11T00:00:01.000000Z",
    }


@pytest.fixture(scope="module")
def restored_pair() -> tuple[InteractionScanPlan, OverlapEvidence]:
    fixture = Path(__file__).parent / "fixtures" / "overlap" / "contract_v1.json"
    raw = json.loads(fixture.read_text(encoding="utf-8"))
    evidence = OverlapEvidence.from_json(
        raw["wire"], expected_sha256=raw["evidence"]["evidence_sha256"]
    )
    pair = evidence.plan

    def entry(factor: OverlapFactor, digest: str) -> InventoryEntry:
        return InventoryEntry(
            identity=ExperimentIdentity(
                target=factor.target, experiment_id=factor.experiment, revision=factor.revision
            ),
            archive_sha256=digest,
            unit_namespace=pair.unit_namespace,
            active_from=pair.entry_open,
            active_until=pair.end,
            arms=(factor.off, factor.on),
            display_name=factor.experiment,
        )

    focal = FocalStudy(
        entry=entry(pair.a, pair.law.archive_sha256[0]),
        population_id=pair.population_id,
        eligibility_rule_sha256=pair.eligibility_rule_sha256,
        enrollment_rule_sha256=pair.enrollment_rule_sha256,
        target=pair.target,
    )
    other = entry(pair.b, pair.law.archive_sha256[1])
    scan_plan = freeze_interaction_scan(
        inventory=declared_inventory(focal.entry, other).model_copy(
            update={
                "frozen_at": pair.registered_at,
                "covered_from": pair.entry_open,
                "complete_through": pair.end,
            }
        ),
        focal=focal,
        metrics=pair.metrics,
        entry_open=pair.entry_open,
        start=pair.start,
        end=pair.end,
        cutoff=evidence.capture.cutoff,
        alpha=pair.alpha,
    )
    return scan_plan, evidence


def test_restored_pair_evidence_cannot_be_carried_into_a_later_look(restored_pair):
    plan, evidence = restored_pair
    scope = plan.scope.model_copy(update={"cutoff": plan.scope.cutoff + timedelta(seconds=1)})
    scan_plan = plan.model_copy(update={"scope": scope})
    matched = PairInput(
        candidate=scan_plan.candidates[0], value=CapturedPair(evidence=evidence, focal_side="A")
    )
    with pytest.raises(CodedError) as mixed:
        ScanCapture(
            plan=scan_plan, observed_inventory=None, coverage="not_captured", pairs=(matched,)
        )
    assert mixed.value.code == "overlap.scan.cutoff_mismatch"
    assert mixed.value.context == {
        "expected_cutoff": "2026-01-01T00:00:12.000000Z",
        "actual_cutoff": "2026-01-01T00:00:11.000000Z",
    }


def test_exchange_requires_exact_metric_issues_without_dropping_compatible_siblings(restored_pair):
    plan, evidence = restored_pair
    planned_metrics = tuple(
        metric.model_copy(update={"definition_sha256": "9" * 64})
        if metric.name == "orders"
        else metric
        for metric in plan.metrics
    )
    plan = plan.model_copy(update={"metrics": (*planned_metrics, SAVE)})
    candidate = plan.candidates[0]
    missing = ScanReason(code="metric_missing")
    mismatch = ScanReason(code="metric_mismatch")
    issues = (
        MetricIssue(metric="orders", reasons=(mismatch,)),
        MetricIssue(metric="save", reasons=(missing,)),
    )
    captured = CapturedPair(evidence=evidence, focal_side="A", metric_issues=issues)
    capture = ScanCapture(
        plan=plan,
        observed_inventory=None,
        coverage="not_captured",
        pairs=(PairInput(candidate=candidate, value=captured),),
    )
    assert capture.pairs[0].value == captured
    assert plan.family_size == 9

    for incorrect in ((), (*issues, MetricIssue(metric="complaint", reasons=(missing,)))):
        submitted = captured.model_copy(update={"metric_issues": incorrect})
        with pytest.raises(CodedError) as caught:
            ScanCapture(
                plan=plan,
                observed_inventory=None,
                coverage="not_captured",
                pairs=(PairInput(candidate=candidate, value=submitted),),
            )
        assert caught.value.code == "overlap.scan.invalid"
        assert caught.value.context == {"field": "metric_issues", "reason": "metric_issue_mismatch"}


@pytest.mark.parametrize("mismatch", [None, "orientation", "candidate"])
def test_capture_binds_restored_evidence_to_its_oriented_pair(restored_pair, mismatch):
    plan, evidence = restored_pair
    original_focal = plan.scope.focal
    focal_b = original_focal.model_copy(update={"entry": plan.candidates[0].entry})
    scope_b = plan.scope.model_copy(update={"focal": focal_b})
    inventory = plan.inventory
    if mismatch == "candidate":
        other = original_focal.entry.model_copy(
            update={
                "identity": original_focal.entry.identity.model_copy(
                    update={"revision": "unrelated"}
                )
            }
        )
        inventory = inventory.model_copy(update={"entries": (focal_b.entry, other)})
    reversed_plan = plan.model_copy(update={"scope": scope_b, "inventory": inventory})
    captured_pair = CapturedPair(
        evidence=evidence, focal_side="A" if mismatch == "orientation" else "B"
    )
    pairs = (PairInput(candidate=reversed_plan.candidates[0], value=captured_pair),)
    if mismatch is None:
        capture = ScanCapture(
            plan=reversed_plan, observed_inventory=None, coverage="not_captured", pairs=pairs
        )
        assert capture.pairs[0].value == captured_pair
    else:
        with pytest.raises(CodedError) as caught:
            ScanCapture(
                plan=reversed_plan, observed_inventory=None, coverage="not_captured", pairs=pairs
            )
        assert caught.value.code == "overlap.scan.invalid"
        assert caught.value.context == {"field": "pairs", "reason": "pair_identity_mismatch"}


@pytest.mark.parametrize("replacement", ["archive", "exclusion"])
def test_partial_exchange_keeps_frozen_assignment_and_exclusions(replacement):
    entry = inventory_entry("cards").model_copy(update={"reasons": ("same_layer",)})
    plan = freeze(declared_inventory(entry))
    observed_entry = entry.model_copy(update={"archive_sha256": "9" * 64})
    observed = observed_inventory(declared_inventory(observed_entry))
    value = UnavailablePair(reasons=(ScanReason(code="roster_drift"),))

    frozen = plan.candidates[0]
    renamed = frozen.model_copy(
        update={"entry": entry.model_copy(update={"display_name": "New display name"})}
    )
    current = ScanCapture(
        plan=plan,
        observed_inventory=observed,
        coverage="partial",
        pairs=(PairInput(candidate=renamed, value=value),),
    )
    assert current.pairs[0].candidate.reasons == (ScanReason(code="same_layer"),)

    substituted = (
        plan.candidates_for(observed)[0]
        if replacement == "archive"
        else frozen.model_copy(update={"reasons": ()})
    )
    with pytest.raises(CodedError) as caught:
        ScanCapture(
            plan=plan,
            observed_inventory=observed,
            coverage="partial",
            pairs=(PairInput(candidate=substituted, value=value),),
        )
    assert caught.value.code == "overlap.scan.invalid"
    assert caught.value.context == {"field": "candidates", "reason": "frozen_candidate_mismatch"}


def test_renderer_receives_numeric_nulls_and_cell_specific_reason(restored_pair):
    plan, evidence = restored_pair
    candidate = plan.candidates[0]
    observed = observed_inventory(plan.inventory, cutoff=plan.scope.cutoff)
    pair = PairInput(candidate=candidate, value=CapturedPair(evidence=evidence, focal_side="A"))
    captured = ScanCapture(
        plan=plan, observed_inventory=observed, coverage="complete", pairs=(pair,)
    )
    ref = PairEvidenceRef(
        study_id=evidence.plan.study_id,
        generation_id=evidence.capture.generation_id,
        evidence_sha256=evidence.evidence_sha256,
        plan_sha256=evidence.plan.plan_sha256,
        family_sha256=evidence.plan.family_sha256,
    )
    rows = []
    for key in plan.family_members:
        empty_cells = {
            "focal_given_other_off": ("00",),
            "focal_given_other_on": ("11",),
            "interaction": ("00", "11"),
        }[key.contrast]
        counts = tuple(
            ScanCellCounts(
                cell=cell.cell,
                n_assigned=cell.n_assigned,
                n_observed=cell.n_observed,
                n_missing=cell.n_missing,
                n_pending=cell.n_pending,
            )
            for cell in evidence.cells
            if cell.metric == key.metric and cell.cell in empty_cells
        )
        rows.append(
            ScanRow(
                key=key,
                estimate=None,
                se=None,
                df=None,
                lower=None,
                upper=None,
                availability="unavailable",
                reasons=tuple(ScanReason(code="empty_cell", cell=count.cell) for count in counts),
                cells=counts,
                evidence_sha256=ref.evidence_sha256,
                decision=None,
            )
        )
    scan = InteractionScan(
        plan=plan,
        capture_id=captured.capture_id,
        observed_inventory=captured.observed_inventory,
        coverage=captured.coverage,
        rows=tuple(rows),
        details=(
            PairDetail(
                candidate=candidate,
                status="captured",
                reasons=(),
                focal_side="A",
                evidence=ref,
                eligibility_counts=evidence.eligibility_counts,
                gate_counts=evidence.gate_counts,
                delivery_counts=evidence.delivery_counts,
                cells=evidence.cells,
            ),
        ),
    )
    direct = next(
        row for row in scan.rows if row.key.metric == "orders" and row.key.contrast == "interaction"
    )
    assert {cell.cell: cell.n_assigned for cell in direct.cells} == {"00": 0, "11": 0}
    assert direct.model_dump(
        mode="json", include={"estimate", "se", "df", "lower", "upper", "reasons"}
    ) == {
        "estimate": None,
        "se": None,
        "df": None,
        "lower": None,
        "upper": None,
        "reasons": [{"cell": "00", "code": "empty_cell"}, {"cell": "11", "code": "empty_cell"}],
    }
