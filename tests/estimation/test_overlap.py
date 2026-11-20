"""Signed factorial inference against independent regression and exact arithmetic."""

import copy
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest
from scipy.stats import t

from increment.errors import CodedError
from increment.estimation.overlap import OverlapResult, analyze_overlap
from increment.semantics.overlap import CONTRAST_IDS, OverlapEvidence, OverlapPlan

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "overlap"
_CELLS = ("00", "10", "01", "11")


@pytest.fixture(scope="module")
def template():
    return json.loads((_FIXTURES / "contract_v1.json").read_text(encoding="utf-8"))["evidence"]


def _centered(values):
    if not values:
        return None, None, None
    anchor = min(values)
    offsets = [Fraction(y) - Fraction(anchor) for y in values]
    delta = sum(offsets) / len(values)
    return anchor, float(delta), float(sum((x - delta) ** 2 for x in offsets))


def _evidence(
    template, orders, complaint=None, *, alpha=0.05, all_claims=True, binary=False, margin=0.1
):
    data = copy.deepcopy(template)
    del data["evidence_sha256"]
    plan = data["plan"]
    plan["alpha"] = alpha
    plan["a"].update(off_weight=3, on_weight=1)
    plan["b"].update(off_weight=1, on_weight=3)
    samples = {"orders": orders}
    if complaint is None:
        plan["metrics"] = plan["metrics"][:1]
    else:
        samples["complaint"] = complaint
        plan["metrics"][1].update(kind="binary_mean" if binary else "mean", margin_abs=margin)
    keys = CONTRAST_IDS if all_claims else ("joint_launch",)
    plan["claims"] = [
        {
            "metric": metric,
            "contrast": contrast,
            "alternative": (
                "noninferiority"
                if metric == "complaint"
                else "difference"
                if contrast == "interaction"
                else "benefit"
            ),
        }
        for metric in samples
        for contrast in keys
    ]
    data["capture"]["law_sha256"] = OverlapPlan.model_validate(plan).law_sha256
    target = sum(map(len, orders))
    data["eligibility_counts"] = [0, 0, 0, target]
    data["gate_counts"] = [target, *([0] * 24)]
    for row, n in zip(
        data["delivery_counts"][:8], [len(vs) for vs in orders for _ in range(2)], strict=True
    ):
        row.update(n_without_delivery=n, n_matching=0, n_inconsistent=0)
    data["cells"] = []
    for metric, groups in samples.items():
        for label, values in zip(_CELLS, groups, strict=True):
            anchor, delta, m2 = _centered(values)
            data["cells"].append(
                {
                    "metric": metric,
                    "cell": label,
                    "n_assigned": len(values),
                    "n_observed": len(values),
                    "n_missing": 0,
                    "n_pending": 0,
                    "anchor": anchor,
                    "mean_delta": delta,
                    "m2": m2,
                    "n_success": sum(values) if binary and metric == "complaint" else None,
                }
            )
    return OverlapEvidence.model_validate(data)


def _groups(locations, scales, counts):
    return [
        [m + s * ((2 * i - n - 1) / n) for i in range(1, n + 1)]
        for m, s, n in zip(locations, scales, counts, strict=True)
    ]


def _row(result, contrast, metric="orders"):
    return next(row for row in result.rows if (row.metric, row.contrast) == (metric, contrast))


def test_saturated_regression_reference(template):
    counts = (41, 53, 37, 64)
    orders = _groups((10, 14, 13, 12), (0.8, 1.3, 2, 3), counts)
    complaint = _groups((150, 149, 151, 149.2), (0.5, 0.4, 0.6, 0.6), counts)
    result = analyze_overlap(_evidence(template, orders, complaint))
    ref = json.loads((_FIXTURES / "regression_reference.json").read_text(encoding="utf-8"))
    for block in result.covariance:
        for row, expected in zip(block.matrix, ref[block.metric]["covariance"], strict=True):
            assert row == pytest.approx(expected, rel=1e-10, abs=1e-10)
    for metric in ("orders", "complaint"):
        for contrast, expected in zip(CONTRAST_IDS, ref[metric]["rows"], strict=True):
            row = _row(result, contrast, metric)
            for field, value in zip(ref["columns"], expected, strict=True):
                atol = 0.0 if field.startswith("p_") else 1e-10
                assert getattr(row, field) == pytest.approx(value, rel=1e-10, abs=atol)
    assert _row(result, "b_given_a_on").decision == "inconclusive"
    assert _row(result, "interaction").decision == "reject"
    assert result.joint_decision == "criteria_met"


def test_covariance_and_family_survive_empty_both(template):
    complete = _groups((10, 14, 13, 12), (1, 1, 1, 1), (40,) * 4)
    full = analyze_overlap(_evidence(template, complete))
    empty_both = analyze_overlap(_evidence(template, [*complete[:3], []]))
    fitted = {"a_given_b_off", "b_given_a_off"}
    assert len(empty_both.rows) == 8
    assert {r.contrast for r in empty_both.rows if r.se is not None} == fitted
    assert empty_both.family_sha256 == full.family_sha256
    for r in empty_both.rows:
        assert r.alpha_member == 0.05 / 8
        assert r.reasons == (() if r.contrast in fitted else (("11", "empty_cell"),))
    block = empty_both.covariance[0]
    for i, left in enumerate(empty_both.rows):
        for j, right in enumerate(empty_both.rows):
            expected = full.covariance[0].matrix[i][j] if left.se and right.se else None
            assert block.matrix[i][j] == expected
    assert block.reasons == tuple(row.reasons for row in empty_both.rows)
    assert empty_both.joint_decision == "insufficient_evidence"


@pytest.mark.parametrize(
    ("cost", "decision", "joint"),
    [
        (-0.8, "clear", "criteria_met"),
        (0.1, "inconclusive", "not_met"),
        (0.8, "loss_beyond_tolerance", "not_met"),
    ],
)
def test_lower_is_better_guardrail_requires_upper_bound(template, cost, decision, joint):
    orders = _groups((0, 0, 0, 0.8), (0.1,) * 4, (48,) * 4)
    complaint = _groups((0, 0, 0, cost), (0.1,) * 4, (48,) * 4)
    result = analyze_overlap(_evidence(template, orders, complaint, all_claims=False))
    row = _row(result, "joint_launch", "complaint")
    assert row.estimate == pytest.approx(cost)
    assert row.boundary == 0.1
    assert row.decision == decision
    assert result.joint_decision == joint
    if decision == "clear":
        assert row.upper < 0.1
        assert _row(result, "b_given_a_on").availability == "descriptive"
        assert _row(result, "b_given_a_on").alpha_member is None


def test_tied_guardrail_limit_does_not_clear(template):
    orders = _groups((0, 0, 0, 4), (0.1,) * 4, (48,) * 4)
    complaint = _groups((0, 0, 0, 0.2), (0.1,) * 4, (48,) * 4)
    initial = analyze_overlap(_evidence(template, orders, complaint, all_claims=False))
    limit = _row(initial, "joint_launch", "complaint").upper
    # Margin never changes the raw band; placing it at the endpoint must not clear.
    tied = analyze_overlap(_evidence(template, orders, complaint, all_claims=False, margin=limit))
    guardrail = _row(tied, "joint_launch", "complaint")
    assert guardrail.upper == guardrail.boundary == limit
    assert guardrail.decision == "inconclusive"
    assert tied.joint_decision == "not_met"


def test_delivery_conflict_preserves_assignment_estimates_not_clearance(template):
    orders = _groups((0, 0, 0, 4), (0.1,) * 4, (48,) * 4)
    evidence = _evidence(template, orders, all_claims=False)
    outside = evidence.delivery_counts[-1].model_copy(update={"n_inconsistent": 1})
    conflict = evidence.model_copy(
        update={"delivery_counts": (*evidence.delivery_counts[:-1], outside)}
    )
    usual = analyze_overlap(evidence)
    actual = analyze_overlap(conflict)
    assert actual.rows == usual.rows
    assert usual.joint_decision == "criteria_met"
    assert actual.joint_decision == "insufficient_evidence"
    assert actual.rollout == "as_assigned_with_delivery_conflicts"


def test_fitted_small_is_not_a_decision(template):
    orders = _groups((0, 0, 0, 4), (0.1,) * 4, (29, 30, 30, 30))
    complaint = [
        [0] * (n - k) + [1] * k for n, k in zip((29, 30, 30, 30), (20, 10, 20, 10), strict=True)
    ]
    result = analyze_overlap(_evidence(template, orders, complaint, binary=True))
    primary = _row(result, "joint_launch")
    guardrail = _row(result, "joint_launch", "complaint")
    assert primary.lower > 0
    assert guardrail.upper < 0.1
    assert primary.availability == guardrail.availability == "fitted_small"
    assert primary.decision == guardrail.decision == "insufficient_for_decision"
    assert primary.reasons == (("00", "small_cell"),)
    assert guardrail.reasons == (*primary.reasons, ("00", "sparse_binary_events"))
    assert _row(result, "b_given_a_on", "complaint").availability == "decided"
    assert result.joint_decision == "insufficient_evidence"


def test_unavailable_retains_distinct_support_reasons_in_cell_order(template):
    evidence = _evidence(template, [[], [1.0], [2.0, 2.0], [0.0, 2.0]])
    cells = evidence.cells
    partial = cells[3].model_copy(
        update={
            "n_observed": 0,
            "n_pending": 1,
            "n_missing": 1,
            "anchor": None,
            "mean_delta": None,
            "m2": None,
        }
    )
    result = analyze_overlap(evidence.model_copy(update={"cells": (*cells[:3], partial)}))
    assert _row(result, "interaction").reasons == (
        ("00", "empty_cell"),
        ("10", "singleton_cell"),
        ("01", "zero_cell_variance"),
        ("11", "pending_outcomes"),
        ("11", "missing_outcomes"),
    )
    assert _row(result, "b_given_a_on").estimate is None
    empty = analyze_overlap(_evidence(template, [[], [], [], []]))
    assert all(row.reasons == ((None, "no_target"),) for row in empty.rows)


def test_adjacent_float_locations_and_extreme_tails(template):
    base = 1e16
    groups = [[base + shift, base + shift + 2] * 31 for shift in (0, 2, 4, 6)]
    exact = [sum(map(Fraction, values)) / len(values) for values in groups]
    evidence = _evidence(template, groups, alpha=1e-24)
    result = analyze_overlap(evidence)
    joint = _row(result, "joint_launch")
    assert joint.estimate == float(exact[3] - exact[0])
    assert _row(result, "interaction").estimate == 0.0
    assert result == analyze_overlap(_evidence(template, [vs[::-1] for vs in groups], alpha=1e-24))
    width = t.isf(1e-24 / 16, joint.df) * joint.se
    assert (joint.lower, joint.upper) == pytest.approx(
        (joint.estimate - width, joint.estimate + width)
    )
    small = math.ldexp(1.0, -330)
    tiny = analyze_overlap(_evidence(template, [[0.0, small]] * 4))
    assert _row(tiny, "joint_launch").df == pytest.approx(2.0)
    variance = tiny.covariance[0].matrix[6][6]
    assert variance is not None and variance / small**2 == pytest.approx(0.5)
    with pytest.raises(CodedError) as caught:
        analyze_overlap(_evidence(template, groups, alpha=math.ulp(0.0)))
    assert caught.value.code == "estimation.tails.unresolvable"


def test_centered_point_at_float_range_and_checked_overflow(template):
    end = sys.float_info.max
    anchor = math.nextafter(end, 0)
    samples = [[anchor], [end], [anchor], [end]]
    result = analyze_overlap(_evidence(template, samples))
    assert _row(result, "a_marginal").estimate == float(Fraction(end) - Fraction(anchor))
    assert _row(result, "a_marginal").availability == "point_only"
    with pytest.raises(CodedError) as caught:
        analyze_overlap(_evidence(template, [[-end], [end], [], []]))
    assert caught.value.code == "overlap.numeric.range"
    assert caught.value.context == {"metric": "orders", "quantity": "estimate"}


def test_covariance_keeps_small_signed_cancellation(template):
    # Exact SS values are 3, 3, 1, 1, except for the adjacent float in cell 11.
    samples = [[0, 0, 0, 2]] * 2 + [[0, 0, 1, 1], [0, 0, 1, math.nextafter(1.0, 2.0)]]
    evidence = _evidence(template, samples)
    m2 = tuple(Fraction(c.m2) for c in evidence.cells)
    # The signed weight products are (-1, -1, +3, +3) / 4.
    want = (3 * (m2[2] + m2[3]) - m2[0] - m2[1]) / 48
    assert want > 0
    result = analyze_overlap(evidence)
    assert result.covariance[0].matrix[0][7] == float(want)


def test_boundary_probability_range_is_not_exact_zero_or_a_gate(template):
    reference = [0.0, 2.0] * 20
    both = [1e16, 1e16 + 2] * 20
    evidence = _evidence(template, [reference] * 3 + [both], all_claims=False)
    result = analyze_overlap(evidence)
    row = _row(result, "joint_launch")
    assert row.p_boundary is row.p_boundary_adj is None
    assert row.reasons == ((None, "p_boundary_numeric_range"),)
    assert row.decision == "clear"
    assert result.joint_decision == "criteria_met"


def test_result_wire_keeps_unavailable_rows_and_typed_large_float(template):
    evidence = _evidence(template, [[0.0], [1e16], [2.0], []])
    result = analyze_overlap(evidence)
    payload = result.to_json()
    digest = result.result_sha256
    assert OverlapResult.from_json(payload, expected_sha256=digest) == result
    replay = OverlapEvidence.from_json(
        evidence.to_json(),
        expected_sha256=evidence.evidence_sha256,
    )
    assert analyze_overlap(replay).to_json() == payload
    wire = json.loads(payload)
    assert wire["evidence_sha256"] == evidence.evidence_sha256
    assert wire["plan_sha256"] == evidence.plan.plan_sha256
    assert len(wire["rows"]) == 8
    assert wire["rows"][2]["estimate"] == 1e16
    assert wire["rows"][6]["estimate"] is None
    assert wire["rows"][6]["reasons"] == [["00", "singleton_cell"], ["11", "empty_cell"]]
    for damaged, expected, reason in (
        (
            payload.replace('"estimate":10000000000000000', '"estimate":1e16'),
            digest,
            "noncanonical_encoding",
        ),
        (payload + "\n", digest, "noncanonical_encoding"),
        (payload.replace('"format":1', '"format":1,"format":1', 1), digest, "duplicate_key"),
        (payload, "0" * 64, "digest_mismatch"),
        (payload.replace('"format":1', '"format":2', 1), digest, "unsupported_version"),
        (payload.replace('"estimate":null', '"estimate":NaN', 1), digest, "nonfinite_number"),
        (payload.replace('"estimate":null', '"estimate":1', 1), digest, "null_mismatch"),
        (payload.replace(evidence.evidence_sha256, "0" * 64, 1), digest, "digest_mismatch"),
    ):
        with pytest.raises(CodedError) as caught:
            OverlapResult.from_json(damaged, expected_sha256=expected)
        assert caught.value.code == "overlap.format.invalid"
        assert caught.value.context["reason"] == reason
