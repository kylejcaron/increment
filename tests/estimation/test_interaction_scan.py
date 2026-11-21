"""Fixed scan families against independent cell-mean and Welch calculations."""

import json
import math
import sys
from pathlib import Path

import pytest
from scipy.stats import t

from increment._overlap_evidence import CELL_IDS, FactorId, OverlapCell, overlap_binary_moments
from increment.estimation.interaction_scan import evaluate_interaction_scan
from increment.estimation.overlap import analyze_overlap
from increment.semantics.interaction_scan import (
    CapturedPair,
    ExperimentIdentity,
    FocalStudy,
    InteractionInventory,
    InventoryEntry,
    MetricIssue,
    ObservedInventory,
    PairInput,
    ScanCapture,
    ScanReason,
    UnavailablePair,
    freeze_interaction_scan,
)
from increment.semantics.overlap import OverlapClaim, OverlapEvidence, OverlapFactor

_RATES = (0.31, 0.34, 0.30, 0.28)
_A = ("a_given_b_off", "a_given_b_on", "interaction")
_B = ("b_given_a_off", "b_given_a_on", "interaction")


@pytest.fixture(scope="module")
def saved_pair():
    path = Path(__file__).parents[1] / "fixtures" / "overlap" / "contract_v1.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return OverlapEvidence.from_json(
        raw["wire"], expected_sha256=raw["evidence"]["evidence_sha256"]
    )


def _binary_cells(counts, successes):
    return tuple(
        OverlapCell(
            metric="orders",
            cell=cell,
            n_assigned=n,
            n_observed=n,
            n_missing=0,
            n_pending=0,
            n_success=k,
            anchor=moments[0],
            mean_delta=moments[1],
            m2=moments[2],
        )
        for cell, n, k in zip(CELL_IDS, counts, successes, strict=True)
        for moments in (overlap_binary_moments(n, k) if n else (None, None, None),)
    )


def _mean_cells(anchors, deltas, sums_sq):
    return tuple(
        OverlapCell(
            metric="orders",
            cell=cell,
            n_assigned=40,
            n_observed=40,
            n_missing=0,
            n_pending=0,
            n_success=None,
            anchor=anchor,
            mean_delta=delta,
            m2=m2,
        )
        for cell, anchor, delta, m2 in zip(CELL_IDS, anchors, deltas, sums_sq, strict=True)
    )


def _evidence(saved, cells, *, focal_side: FactorId = "A", alpha=0.05, conflict=False):
    plan = saved.plan.model_copy(
        update={
            "metrics": (
                saved.plan.metrics[0].model_copy(
                    update={"kind": "mean" if cells[0].n_success is None else "binary_mean"}
                ),
            ),
            "claims": tuple(
                OverlapClaim(metric="orders", contrast=contrast, alternative="difference")
                for contrast in (_A if focal_side == "A" else _B)
            ),
            "alpha": alpha,
        }
    )
    target = sum(cell.n_assigned for cell in cells)
    deliveries = tuple(
        delivery.model_copy(
            update={
                "n_without_delivery": cell.n_assigned - inconsistent,
                "n_matching": 0,
                "n_inconsistent": inconsistent,
            }
        )
        for pos, delivery in enumerate(saved.delivery_counts[:8])
        for cell in (cells[pos // 2],)
        for inconsistent in (int(conflict and pos == 0),)
    )
    return saved.model_copy(
        update={
            "plan": plan,
            "eligibility_counts": (0, 0, 0, target),
            "gate_counts": (target, *(0,) * 24),
            "delivery_counts": deliveries + saved.delivery_counts[8:],
            "cells": cells,
        }
    )


def _capture(
    evidence, *, focal_side: FactorId = "A", candidates=1, other_metric=False, partial=False
):
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

    a = entry(pair.a, pair.law.archive_sha256[0])
    b = entry(pair.b, pair.law.archive_sha256[1])
    focal, other = (a, b) if focal_side == "A" else (b, a)
    extras = tuple(
        other.model_copy(
            update={
                "identity": ExperimentIdentity(
                    target=f"extra-target-{i}", experiment_id=f"extra-experiment-{i}", revision="r1"
                )
            }
        )
        for i in range(candidates - 1)
    )
    source = {
        "universe_id": "warehouse-tests",
        "source_id": "inventory",
        "version": "version-1",
        "digest": "a" * 64,
        "covered_from": pair.entry_open,
        "complete_through": pair.end,
        "entries": (focal, other, *extras),
    }
    metrics = pair.metrics
    issues = ()
    if other_metric:
        metrics += (
            pair.metrics[0].model_copy(update={"name": "other", "definition_sha256": "b" * 64}),
        )
        issues = (MetricIssue(metric="other", reasons=(ScanReason(code="metric_missing"),)),)
    cutoff = evidence.capture.cutoff
    plan = freeze_interaction_scan(
        inventory=InteractionInventory(**source, coverage="complete", frozen_at=pair.registered_at),
        focal=FocalStudy(
            entry=focal,
            population_id=pair.population_id,
            eligibility_rule_sha256=pair.eligibility_rule_sha256,
            enrollment_rule_sha256=pair.enrollment_rule_sha256,
            target=pair.target,
        ),
        metrics=metrics,
        entry_open=pair.entry_open,
        start=pair.start,
        end=pair.end,
        cutoff=cutoff,
        alpha=pair.alpha,
    )
    return ScanCapture(
        plan=plan,
        coverage="partial" if partial else "complete",
        observed_inventory=ObservedInventory(
            **source, cutoff=cutoff, coverage="partial" if partial else "complete"
        ),
        pairs=tuple(
            PairInput(
                candidate=candidate,
                value=(
                    CapturedPair(evidence=evidence, focal_side=focal_side, metric_issues=issues)
                    if candidate.entry == other
                    else UnavailablePair(reasons=(ScanReason(code="pair_evidence_missing"),))
                ),
            )
            for candidate in plan.candidates
        ),
    )


def _restored(evidence):
    return OverlapEvidence.from_json(evidence.to_json(), expected_sha256=evidence.evidence_sha256)


def _captured_rows(scan):
    return tuple(row for row in scan.rows if row.key.metric == "orders" and row.evidence_sha256)


def _oracle(counts, rates, c, *, members):
    estimate = sum(weight * rate for weight, rate in zip(c, rates, strict=True))
    terms = tuple(
        (weight**2 * rate * (1 - rate) / (n - 1), n - 1)
        for weight, rate, n in zip(c, rates, counts, strict=True)
        if weight
    )
    variance = sum(value for value, _ in terms)
    df = variance**2 / sum(value**2 / dof for value, dof in terms)
    se = math.sqrt(variance)
    half_width = float(t.isf(0.05 / (2 * members), df)) * se
    return estimate, se, df, estimate - half_width, estimate + half_width


@pytest.mark.parametrize(
    ("extra_metric", "members", "expected_bounds"),
    [
        (
            False,
            9,
            ((0.01163917, 0.04836083), (-0.03779279, -0.00220721), (-0.07556624, -0.02443376)),
        ),
        (
            True,
            18,
            (
                (0.010192767, 0.049807233),
                (-0.039194453, -0.000805547),
                (-0.077580046, -0.022419954),
            ),
        ),
    ],
)
def test_frozen_family_includes_unavailable_candidates_and_metrics(
    saved_pair, extra_metric, members, expected_bounds
):
    evidence = _evidence(saved_pair, _binary_cells((10_000,) * 4, (3100, 3400, 3000, 2800)))
    capture = _capture(evidence, candidates=3, other_metric=extra_metric)
    scan = evaluate_interaction_scan(capture)

    assert len(scan.rows) == members
    assert tuple(row.key for row in scan.rows) == capture.plan.family_members
    rows = _captured_rows(scan)
    assert tuple(row.estimate for row in rows) == pytest.approx((0.03, -0.02, -0.05))
    for row, bounds in zip(rows, expected_bounds, strict=True):
        assert (row.lower, row.upper) == pytest.approx(bounds, abs=1e-7)
        assert row.decision == "detected"
    for row in scan.rows:
        if row not in rows:
            assert (row.availability, row.estimate, row.lower, row.upper, row.decision) == (
                "unavailable",
                None,
                None,
                None,
                None,
            )
            assert row.reasons[0].code == (
                "metric_missing" if row.evidence_sha256 else "pair_evidence_missing"
            )


def test_missing_both_on_cell_after_replay_keeps_off_and_full_family(saved_pair):
    evidence = _evidence(
        saved_pair, _binary_cells((10_000, 10_000, 10_000, 0), (3100, 3400, 3000, 0))
    )
    scan = evaluate_interaction_scan(_capture(_restored(evidence), candidates=3))
    off, on, direct = _captured_rows(scan)

    assert len(scan.rows) == 9
    assert (off.lower, off.upper) == pytest.approx((0.01163917, 0.04836083), abs=1e-7)
    for row in (on, direct):
        assert row.availability == "unavailable"
        assert row.estimate is row.lower is row.decision is None
        assert row.reasons == (ScanReason(code="empty_cell", cell="11"),)


@pytest.mark.parametrize(
    ("side", "contrasts", "weights", "detected"),
    [
        ("A", _A, ((-1, 1, 0, 0), (0, 0, -1, 1), (1, -1, -1, 1)), (True, True, True)),
        ("B", _B, ((-1, 0, 1, 0), (0, -1, 0, 1), (1, -1, -1, 1)), (False, True, True)),
    ],
)
def test_focal_orientation_matches_fixed_family_before_and_after_replay(
    saved_pair, side, contrasts, weights, detected
):
    counts = (18_000, 12_000, 10_000, 8000)
    cells = _binary_cells(counts, (5580, 4080, 3000, 2240))
    evidence = _evidence(saved_pair, cells, focal_side=side)
    for receipt in (evidence, _restored(evidence)):
        scan = evaluate_interaction_scan(_capture(receipt, focal_side=side))
        pair_rows = {row.contrast: row for row in analyze_overlap(receipt).rows}
        assert len(scan.rows) == 3
        for row, key, c, yes in zip(scan.rows, contrasts, weights, detected, strict=True):
            estimate, se, df, lower, upper = _oracle(counts, _RATES, c, members=3)
            assert row.estimate == pytest.approx(estimate, abs=1e-12)
            assert row.se == pytest.approx(se, abs=1e-7)
            assert row.df == pytest.approx(df, abs=0.001)
            assert (row.lower, row.upper) == pytest.approx((lower, upper), abs=1e-7)
            assert row.decision == ("detected" if yes else "none_detected")
            fit = pair_rows[key]
            assert (row.estimate, row.se, row.df, row.lower, row.upper) == (
                fit.estimate,
                fit.se,
                fit.df,
                fit.lower,
                fit.upper,
            )
            assert tuple(cell.cell for cell in row.cells) == tuple(
                cell for cell, weight in zip(CELL_IDS, c, strict=True) if weight
            )

    absent = _binary_cells((counts[0], 0, *counts[2:]), (5580, 0, 3000, 2240))
    replay = _restored(_evidence(saved_pair, absent, focal_side=side))
    rows = evaluate_interaction_scan(_capture(replay, focal_side=side)).rows
    for row, c in zip(rows, weights, strict=True):
        if c[1]:
            assert row.availability == "unavailable"
            assert row.reasons == (ScanReason(code="empty_cell", cell="10"),)
        else:
            assert row.availability == "decided"
            assert (row.lower, row.upper) == pytest.approx(
                _oracle(counts, _RATES, c, members=3)[3:]
            )


def test_incomplete_roster_leaves_only_descriptive_fit(saved_pair):
    evidence = _evidence(saved_pair, _binary_cells((10_000,) * 4, (3100, 3400, 3000, 2800)))
    scan = evaluate_interaction_scan(_capture(evidence, partial=True))
    off, _, _ = scan.rows

    assert off.availability == "descriptive"
    assert off.estimate == pytest.approx(0.03)
    assert off.se == pytest.approx(_oracle((10_000,) * 4, _RATES, (-1, 1, 0, 0), members=3)[1])
    assert all(row.lower is row.upper is row.decision is None for row in scan.rows)


def test_small_and_sparse_support_keep_intervals_without_decision(saved_pair):
    counts = (29, 100, 100, 100)
    successes = (5, 90, 90, 95)
    evidence = _evidence(saved_pair, _binary_cells(counts, successes))
    off, on, _ = evaluate_interaction_scan(_capture(evidence)).rows

    assert off.availability == on.availability == "fitted_small"
    assert off.decision is on.decision is None
    assert off.lower is not None and off.lower > 0
    rates = tuple(k / n for k, n in zip(successes, counts, strict=True))
    assert (off.lower, off.upper) == pytest.approx(
        _oracle(counts, rates, (-1, 1, 0, 0), members=3)[3:]
    )
    assert off.reasons == (
        ScanReason(code="small_cell", cell="00"),
        ScanReason(code="sparse_binary_events", cell="00"),
    )
    assert on.reasons == (ScanReason(code="sparse_binary_events", cell="11"),)


def test_missing_pending_singleton_and_zero_variance_have_distinct_reasons(saved_pair):
    cells = _binary_cells((1, 40, 40, 40), (0, 0, 12, 12))
    cells = (
        *cells[:2],
        cells[2].model_copy(
            update={
                "n_observed": 38,
                "n_pending": 1,
                "n_missing": 1,
                "anchor": None,
                "mean_delta": None,
                "m2": None,
            }
        ),
        cells[3],
    )
    evidence = _evidence(saved_pair, cells)
    off, on, _ = evaluate_interaction_scan(_capture(evidence)).rows

    assert off.availability == "point_only"
    assert off.estimate == 0.0
    assert off.se is off.df is off.lower is off.upper is off.decision is None
    assert off.reasons == (
        ScanReason(code="singleton_cell", cell="00"),
        ScanReason(code="zero_cell_variance", cell="10"),
    )
    assert on.availability == "unavailable"
    assert on.estimate is on.lower is on.decision is None
    assert on.reasons == (
        ScanReason(code="pending_outcomes", cell="01"),
        ScanReason(code="missing_outcomes", cell="01"),
    )


def test_numeric_range_is_row_local_and_delivery_is_a_pair_diagnostic(saved_pair):
    limit = sys.float_info.max
    cells = _mean_cells((-limit, limit, 2.0, 8.0), (1.0,) * 4, (40.0,) * 4)
    evidence = _evidence(saved_pair, cells, conflict=True)
    scan = evaluate_interaction_scan(_capture(evidence))
    off, on, direct = scan.rows
    (detail,) = scan.details

    assert off.availability == direct.availability == "unavailable"
    assert off.estimate is direct.estimate is None
    assert off.reasons == direct.reasons == (ScanReason(code="overlap.numeric.range"),)
    assert on.reasons == ()
    assert on.estimate == 6.0 and on.decision == "detected"
    assert detail.delivery_counts == evidence.delivery_counts
    assert detail.reasons == (ScanReason(code="delivery_conflicts"),)
    assert detail.evidence is not None
    assert detail.evidence.family_sha256 == evidence.plan.family_sha256
    assert detail.evidence.evidence_sha256 == evidence.evidence_sha256
    assert scan.capture_id == _capture(evidence).capture_id


def test_interval_touching_zero_in_either_direction_is_not_detected(saved_pair):
    # Each conditional has df=78 and SE=1 to working precision. This family
    # level gives an upper-tail t critical value of 1 to working precision.
    cells = _mean_cells((0.0, 1.0, 1.0, 0.0), (1.0,) * 4, (780.0,) * 4)
    evidence = _evidence(saved_pair, cells, alpha=0.961208197662346)
    off, on, _ = evaluate_interaction_scan(_capture(evidence)).rows

    assert (off.lower, off.upper) == (0.0, 2.0)
    assert (on.lower, on.upper) == (-2.0, 0.0)
    assert off.decision == on.decision == "none_detected"


def test_unrepresentable_tail_keeps_descriptive_fit_without_interval(saved_pair):
    cells = _binary_cells((10_000,) * 4, (3100, 3400, 3000, 2800))
    evidence = _evidence(saved_pair, cells, alpha=math.ulp(0.0))
    off, _, _ = evaluate_interaction_scan(_capture(evidence)).rows

    assert off.estimate == pytest.approx(0.03)
    assert off.se == pytest.approx(0.006620754, abs=1e-7)
    assert off.availability == "descriptive"
    assert off.lower is off.upper is off.decision is None
    assert off.reasons == (ScanReason(code="estimation.tails.unresolvable"),)
