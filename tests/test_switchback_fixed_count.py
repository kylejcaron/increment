"""Fixed exposure and its independent replicate reference on complete panels."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import combinations
from statistics import mean, stdev

import polars as pl
import pytest

from increment import from_switchback_frame
from increment.errors import InvalidRequestError
from increment.estimation.contrast_results import ContrastResult
from increment.estimation.decision_types import ContrastDecisionProcedure, PValueEvidence
from increment.estimation.results import contrast_p_value, contrast_rejection_side
from increment.estimators import FixedCountOrder, SwitchbackAssignment, SwitchbackWindow
from increment.semantics.design import Randomized
from increment.semantics.unit_cycle import UnitCycleTApproximation
from increment.workflows import assignment_diagnostic
from tests.analysis_factory import bound_source, contrast_rows, view_readout


def _assignment(k, m, grain):
    return SwitchbackAssignment(
        periods_per_cycle=k,
        sequence=FixedCountOrder(treated_periods=m, independence_unit=grain),
        window=SwitchbackWindow(washout_steps=1, observation_steps=3, carryover_order=1),
    )


def _build(rows, assignment, *, metrics=None):
    k, m = assignment.periods_per_cycle, assignment.sequence.treated_periods
    return from_switchback_frame(
        pl.DataFrame(rows).reverse(),
        unit="u",
        cycle="c",
        period="p",
        step="s",
        group="arm",
        metrics={"response": "mean"} if metrics is None else metrics,
        identification=Randomized(control_group="A", allocation={"A": (k - m) / k, "B": m / k}),
        assignment=assignment,
    )


def _weighted_panel(k, m, grain):
    # Enumerate the law independently; do not estimate probabilities from realized draws.
    support = tuple(combinations(range(k), m))
    pi = [mean(p in seq for seq in support) for p in range(k)]
    cycles = k + 1  # repeat one order; equality of realized frequencies is not required
    units = ("zone2", "Zone3", "zone10")
    rows, by_draw = [], {}
    for i, unit in enumerate(units):
        for c in range(cycles):
            draw = support[(c + (i if grain == "unit_cycle" else 0)) % k]
            weighted = []
            for p in range(k):
                treated = p in draw
                for s in range(4):
                    y = 1000 * (s < 2) + (1 + p) ** 2 + (2 + i * p + c) * treated + s
                    rows.append(
                        {
                            "u": unit,
                            "c": c,
                            "p": p,
                            "s": s,
                            "arm": "B" if treated else "A",
                            "response": y,
                        }
                    )
                    if s >= 2:
                        weighted.append(y / pi[p] if treated else -y / (1 - pi[p]))
            by_draw[unit, c] = math.fsum(weighted) / k
    replicates = (
        [mean(by_draw[u, c] for c in range(cycles)) for u in units]
        if grain == "unit_cycle"
        else [mean(by_draw[u, c] for u in units) for c in range(cycles)]
    )
    return rows, replicates


@pytest.mark.parametrize(("k", "m", "grain"), [(10, 1, "unit_cycle"), (5, 4, "shared_block")])
def test_fixed_count_reference_over_complete_independent_replicates(k, m, grain):
    rows, replicates = _weighted_panel(k, m, grain)
    assignment = _assignment(k, m, grain)
    experiment, reader = _build(rows, assignment)
    results = contrast_rows(view_readout(experiment, reader))
    (result,) = results
    assert result.estimate is not None
    assert result.estimate.value == pytest.approx(mean(replicates))
    assert result.standard_error == pytest.approx(stdev(replicates) / math.sqrt(len(replicates)))
    assert result.dof == len(replicates) - 1
    assert result.reference == ("block_t" if grain == "shared_block" else "unit_t_approximation")
    assert ContrastResult.model_validate_json(result.model_dump_json()) == result
    for value in (result, assignment_diagnostic(experiment, reader)):
        assert (value.randomization_law, value.independence_grain) == ("fixed_count", grain)
        assert (value.periods_per_cycle, value.treated_periods, value.treatment_probability) == (
            k,
            m,
            m / k,
        )
        assert value.probability_ct is value.ct_cycles is value.tc_cycles is None
    frame = results.to_frame(backend="polars")
    assert isinstance(frame, pl.DataFrame)
    assert frame["treated_periods"].to_list() == [m]
    if grain == "shared_block":
        duplicate = [{**row, "u": "copy:" + row["u"]} for row in reversed(rows)]
        clone, reader = _build(duplicate + rows, assignment)
        (repeated,) = contrast_rows(view_readout(clone, reader))
        assert repeated.estimate == result.estimate
        assert repeated.standard_error == result.standard_error
        assert repeated.dof == result.dof


@pytest.mark.parametrize(
    ("orders", "reason"),
    [
        ((("TCC", "CTC"), ("TCC", "CTC")), "missing_position_support"),
        ((("TCC", "CTC"), ("CTC", "CCT")), "shared_block_order_mismatch"),
        ((("TCC", "CTC"), ("TCC", "CTT")), "invalid_period_assignment"),
    ],
)
def test_fixed_count_refuses_unidentified_or_wrong_shared_schedule(orders, reason):
    rows = [
        {"u": str(u), "c": c, "p": p, "s": s, "arm": "B" if letter == "T" else "A", "response": s}
        for u, cycles in enumerate(orders)
        for c, order in enumerate(cycles)
        for p, letter in enumerate(order)
        for s in range(4)
    ]
    with pytest.raises(InvalidRequestError) as caught:
        _build(rows, _assignment(3, 1, "shared_block"))
    assert caught.value.code == "source.frame.switchback.schedule"
    assert caught.value.context["reason"] == reason


def test_fixed_count_requires_interior_count_and_distinct_multi_period_law():
    for k, m in ((3, 3), (3, 4), (2, 1)):
        with pytest.raises(InvalidRequestError) as caught:
            _assignment(k, m, "unit_cycle")
        assert caught.value.code == "definition.assignment.fixed_count"


def test_fixed_count_rejects_identification_with_paired_arm_shares():
    from increment.estimators import SwitchbackStudyEnvelope

    with pytest.raises(InvalidRequestError) as caught:
        SwitchbackStudyEnvelope(
            assignment=_assignment(5, 4, "shared_block"),
            identification=Randomized(control_group="A", allocation={"A": 0.5, "B": 0.5}),
        )
    assert caught.value.code == "source.frame.switchback.identification"
    assert caught.value.context["reason"] == "unequal_allocation"


def test_nonuniform_position_probabilities_do_not_identify_the_fixed_count_contrast():
    null_outcomes = (1, 0, 0)
    deltas = [y - Fraction(sum(null_outcomes) - y, 2) for y in null_outcomes]
    nonuniform = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))
    assert mean(deltas) == 0
    assert sum(p * d for p, d in zip(nonuniform, deltas, strict=True)) == Fraction(1, 4)
    for slots, expectation in (((0, 1, 2), 0), ((0, 0, 1, 2), Fraction(1, 4))):
        # Give each possible order its probability mass as a fraction of roster members.
        # All positions see both arms, so the example tests weighting, not admission.
        rows = [
            {
                "u": str(u),
                "c": 0,
                "p": p,
                "s": s,
                "arm": "B" if slot == p else "A",
                "response": null_outcomes[p] if s == 2 else 0,
            }
            for u, slot in enumerate(slots)
            for p in range(3)
            for s in range(4)
        ]
        experiment, reader = _build(rows, _assignment(3, 1, "unit_cycle"))
        (row,) = contrast_rows(view_readout(experiment, reader))
        assert row.estimate is not None
        assert row.estimate.value == expectation
    with pytest.raises(InvalidRequestError) as caught:
        FixedCountOrder.model_validate(
            {
                "treated_periods": 1,
                "independence_unit": "unit_cycle",
                "position_probabilities": nonuniform,
            }
        )
    assert caught.value.code == "model.field.unknown"
    assert caught.value.context["field"] == "position_probabilities"


def test_fixed_count_preserves_small_contrasts_between_large_integer_totals():
    rows = [
        {
            "u": f"unit-{u}",
            "c": 0,
            "p": p,
            "s": s,
            "arm": "B" if p == u else "A",
            "response": 2**55 + ((u + 1) if s == 2 and p == u else 0),
        }
        for u in range(3)
        for p in range(3)
        for s in range(4)
    ]
    experiment, reader = _build(rows, _assignment(3, 1, "unit_cycle"))
    (result,) = contrast_rows(view_readout(experiment, reader))
    assert result.estimate is not None
    assert result.estimate.value == 2
    assert result.standard_error == pytest.approx(1 / math.sqrt(3))


@pytest.mark.parametrize(
    ("grain", "offsets", "mean_offset", "ratio"),
    [
        ("shared_block", (-1, 0, 1), 0, False),
        ("unit_cycle", (1, 0, -1), 0, False),
        ("shared_block", (4, 3, 2), 3, False),
        ("shared_block", (4, 3, 2), 3, True),
    ],
)
def test_sub_ulp_replicate_differences_retain_the_null_test(grain, offsets, mean_offset, ratio):
    from scipy.stats import t

    from increment.estimation.contrast import estimate_contrast
    from increment.frame import MetricSpec

    d = 2**55
    rows = [
        {
            "u": f"unit-{i}" if grain == "unit_cycle" else "one",
            "c": 0 if grain == "unit_cycle" else i,
            "p": p,
            "s": s,
            "arm": "B" if p == i else "A",
            "response": d + offset if s == 2 and p == i else 0,
            "denominator": int(s == 2),
        }
        for i, offset in enumerate(offsets)
        for p in range(3)
        for s in range(4)
    ]
    metrics = (
        [MetricSpec(name="response", type="ratio", numerator="response", denominator="denominator")]
        if ratio
        else None
    )
    _, reader = _build(rows, _assignment(3, 1, grain), metrics=metrics)
    source = bound_source(reader)
    computed = estimate_contrast(
        source.contrast_stats(source.metrics[0]),
        ContrastDecisionProcedure(
            reference=UnitCycleTApproximation() if grain == "unit_cycle" else None,
            metric="response",
            role="primary",
            alternative="two-sided",
            null_abs=d,
            alpha=0.05,
        ),
    )

    # All three integers round to d, but M2 = (-1)**2 + 0**2 + 1**2.
    (result,) = computed.results
    assert result.standard_error == pytest.approx(1 / math.sqrt(3))
    assert result.estimate is not None and result.estimate.value == d
    assert result.estimate.lb == (d if mean_offset else math.nextafter(d, -math.inf))
    assert result.estimate.ub == math.nextafter(d, math.inf)
    expected_p = 2 * t.sf(mean_offset * math.sqrt(3), 2)
    (evidence,) = computed.evidence.values()
    assert isinstance(evidence, PValueEvidence) and evidence.p_value == pytest.approx(expected_p)
    restored = ContrastResult.model_validate_json(result.model_dump_json())
    assert contrast_p_value(restored) == pytest.approx(expected_p)
    assert contrast_rejection_side(restored) == ("greater" if mean_offset else None)


def test_fixed_count_ledger_keeps_all_assigned_cells_without_a_paired_order_test():
    from increment.estimation.assignment_integrity import assignment_integrity
    from increment.evidence import (
        AssignmentEvidence,
        AssignmentRequest,
        evidence_provider,
        request_evidence,
    )
    from tests.analysis_factory import held_source

    assignment = _assignment(3, 1, "shared_block")
    rows = [
        {"u": u, "c": c, "p": p, "s": s, "arm": "B" if c == p else "A", "response": p}
        for u in ("z", "a")
        for c in range(3)
        for p in range(3)
        for s in range(4)
    ]
    _, reader = _build(rows, assignment)
    ledger = request_evidence(
        evidence_provider(held_source(reader)),
        AssignmentRequest(experiment="frame", all_periods=True),
    )
    assert isinstance(ledger, AssignmentEvidence)
    assert dict(ledger.probabilities) == {"A": 2 / 3, "B": 1 / 3}
    assert set(ledger.arm_counts) == {
        (1, f"{c}:period={p}", "B" if c == p else "A", 2) for c in range(3) for p in range(3)
    }
    assert dict(ledger.order_draws) == {"block:0": "tcc", "block:1": "ctc", "block:2": "cct"}
    integrity = assignment_integrity(assignment, None, order_draws=ledger.order_draws)
    assert integrity.status == "not_applicable"
    assert integrity.randomization_grain == "shared_block"
    assert integrity.context["order_draws"] == 3
    assert integrity.expected is None
