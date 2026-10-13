"""Known-effect recovery of every declared contrast against an independent OLS+HC2 reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from increment.errors import CodedError
from increment.overlap import (
    CONTRASTS,
    OverlapDesign,
    OverlapExperiment,
    estimate_overlap_effects,
)
from tests.overlap.fixtures import (
    DESIGN,
    from_frame,
    generated_cells,
    hc2_reference,
    unit_frame,
)

SIZES = {(0, 0): 300, (1, 0): 300, (0, 1): 100, (1, 1): 100}


def _truths(mu: dict[tuple[int, int], float], p1: float, p2: float) -> dict[str, float]:
    return {
        "interaction": mu[(1, 1)] - mu[(1, 0)] - mu[(0, 1)] + mu[(0, 0)],
        "joint_launch": mu[(1, 1)] - mu[(0, 0)],
        "first_only_launch": mu[(1, 0)] - mu[(0, 0)],
        "second_only_launch": mu[(0, 1)] - mu[(0, 0)],
        "first_given_second_treatment": mu[(1, 1)] - mu[(0, 1)],
        "second_given_first_treatment": mu[(1, 1)] - mu[(1, 0)],
        "marginal_first": p2 * (mu[(1, 1)] - mu[(0, 1)]) + (1 - p2) * (mu[(1, 0)] - mu[(0, 0)]),
        "marginal_second": p1 * (mu[(1, 1)] - mu[(1, 0)]) + (1 - p1) * (mu[(0, 1)] - mu[(0, 0)]),
    }


def _readout(frame, design=DESIGN, **kwargs):
    cells = from_frame(frame, design)
    return estimate_overlap_effects(cells, contrasts=CONTRASTS, **kwargs)


def test_exact_cell_contrasts_match_the_hand_computed_values() -> None:
    cells = {
        (0, 0): [1.0, 3.0],
        (1, 0): [2.0, 6.0],
        (0, 1): [0.0, 4.0, 8.0],
        (1, 1): [10.0, 12.0],
    }
    readout = _readout(unit_frame(cells))
    means = {(0, 0): 2.0, (1, 0): 4.0, (0, 1): 4.0, (1, 1): 11.0}
    variances = {(0, 0): 2.0, (1, 0): 8.0, (0, 1): 16.0, (1, 1): 2.0}
    n = {(0, 0): 2, (1, 0): 2, (0, 1): 3, (1, 1): 2}
    truths = _truths(means, 0.5, 0.25)
    for row in readout.rows:
        assert row.estimate == pytest.approx(truths[row.contrast], abs=1e-12)
        weights = dict(zip(row.cells, row.weights, strict=True))
        var = sum(w * w * variances[c] / n[c] for c, w in weights.items())
        assert row.se == pytest.approx(math.sqrt(var), rel=1e-12)
    assert readout.row("revenue", "interaction").estimate == pytest.approx(5.0)


@pytest.mark.parametrize(
    "case, mu",
    [
        ("no_interaction", {(0, 0): 10.0, (1, 0): 11.0, (0, 1): 10.5, (1, 1): 11.5}),
        ("sign_reversal", {(0, 0): 10.0, (1, 0): 11.0, (0, 1): 10.5, (1, 1): 9.0}),
        ("synergy", {(0, 0): 10.0, (1, 0): 10.2, (0, 1): 10.1, (1, 1): 12.0}),
    ],
)
def test_contrasts_match_the_independent_ols_hc2_reference(case: str, mu) -> None:
    rng = np.random.default_rng(83)
    frame = unit_frame(
        generated_cells(rng, mu, SIZES, sd=2.0),
        first_only=[1e6] * 40,
        second_only=[-1e6] * 30,
        neither=[5e5] * 10,
    )
    readout = _readout(frame)
    assert readout.audit.jointly_eligible == sum(SIZES.values())
    assert (readout.audit.first_only_eligible, readout.audit.second_only_eligible) == (40, 30)
    assert readout.audit.neither_eligible == 10
    for row in readout.rows:
        estimate, se = hc2_reference(frame, row.weights)
        assert row.estimate == pytest.approx(estimate, rel=1e-10, abs=1e-10)
        assert row.se == pytest.approx(se, rel=1e-10)
    if case == "no_interaction":
        assert (
            readout.row("revenue", "interaction").lb
            < 0.0
            < readout.row("revenue", "interaction").ub
        )
    if case == "sign_reversal":
        # The first experiment helps alone and hurts with the second launched.
        assert readout.row("revenue", "first_only_launch").estimate > 0.0
        assert readout.row("revenue", "first_given_second_treatment").estimate < 0.0


def test_marginal_weights_follow_the_declared_mix_not_the_observed_one() -> None:
    mu = {(0, 0): 0.0, (1, 0): 1.0, (0, 1): 0.0, (1, 1): 3.0}
    exact = {cell: [v - 0.5, v + 0.5] * 50 for cell, v in mu.items()}
    design = OverlapDesign(
        first=OverlapExperiment(name="checkout", treatment_probability=0.2),
        second=OverlapExperiment(
            name="search", control="old", treatment="new", treatment_probability=0.7
        ),
    )
    readout = _readout(unit_frame(exact, design=design), design=design)
    # Observed shares are 50/50 in both experiments; the declared mix is 0.2 / 0.7.
    assert readout.row("revenue", "marginal_first").estimate == pytest.approx(0.7 * 3 + 0.3 * 1)
    assert readout.row("revenue", "marginal_second").estimate == pytest.approx(0.2 * 2 + 0.8 * 0)


def test_unequal_allocation_keeps_cell_counts_and_welch_df() -> None:
    rng = np.random.default_rng(7)
    sizes = {(0, 0): 900, (1, 0): 60, (0, 1): 30, (1, 1): 10}
    mu = {(0, 0): 0.0, (1, 0): 0.0, (0, 1): 0.0, (1, 1): 0.0}
    readout = _readout(unit_frame(generated_cells(rng, mu, sizes)))
    assert {m.cell: m.n for m in readout.cells} == sizes
    joint = readout.row("revenue", "joint_launch")
    v11 = next(m for m in readout.cells if m.cell == (1, 1))
    v00 = next(m for m in readout.cells if m.cell == (0, 0))
    a, b = v11.variance / 10, v00.variance / 900
    assert joint.df == pytest.approx((a + b) ** 2 / (a * a / 9 + b * b / 899), rel=1e-12)


def test_absent_and_single_unit_cells_leave_only_their_contrasts_unavailable() -> None:
    cells = {(0, 0): [1.0, 2.0, 3.0], (1, 0): [2.0, 4.0], (0, 1): [7.0]}
    readout = _readout(unit_frame(cells))
    by = {r.contrast: r for r in readout.rows}
    assert by["first_only_launch"].available
    assert by["joint_launch"].unavailable_reason == "cell_empty"
    assert by["joint_launch"].unsupported_cells == ((1, 1),)
    assert by["second_only_launch"].unavailable_reason == "cell_variance_unidentified"
    assert by["interaction"].unsupported_cells == ((1, 1), (0, 1))
    for row in readout.rows:
        if not row.available:
            assert row.estimate is None and row.se is None and row.lb is None
            assert row.significant is None
            assert row.metric == "revenue" and row.family_size == 1
    moment_11 = next(m for m in readout.cells if m.cell == (1, 1))
    assert (moment_11.n, moment_11.mean, moment_11.variance) == (0, None, None)


def test_large_common_offset_does_not_move_the_contrast() -> None:
    base = {(0, 0): [1.0, 2.0], (1, 0): [3.0, 5.0], (0, 1): [2.0, 2.5], (1, 1): [9.0, 7.0]}
    shifted = {c: [v + 1e9 for v in vs] for c, vs in base.items()}
    a, b = _readout(unit_frame(base)), _readout(unit_frame(shifted))
    for x, y in zip(a.rows, b.rows, strict=True):
        assert y.estimate == pytest.approx(x.estimate, abs=1e-6)
        assert y.se == pytest.approx(x.se, rel=1e-6)


def test_zero_variance_contrast_reports_a_degenerate_interval() -> None:
    cells = {(0, 0): [1.0, 1.0], (1, 0): [1.0, 1.0], (0, 1): [1.0, 1.0], (1, 1): [1.0, 1.0]}
    row = _readout(unit_frame(cells)).row("revenue", "interaction")
    assert (row.estimate, row.se, row.lb, row.ub, row.p_value) == (0.0, 0.0, 0.0, 0.0, 1.0)


def test_cell_means_near_the_float_limit_do_not_overflow() -> None:
    big = 1.7e308
    cells = {
        (0, 0): [big, big],
        (1, 0): [big * 0.75, big * 0.75],
        (0, 1): [big, big],
        (1, 1): [big, big],
    }
    readout = _readout(unit_frame(cells))
    means = {m.cell: m.mean for m in readout.cells}
    assert means[(0, 0)] == big and means[(1, 0)] == pytest.approx(0.75 * big)
    assert readout.row("revenue", "joint_launch").estimate == 0.0
    assert readout.row("revenue", "interaction").estimate == pytest.approx(0.25 * big)


def test_a_contrast_beyond_the_float_range_is_refused() -> None:
    big = 1.7e308
    cells = {c: [big, big] for c in ((1, 0), (1, 1))} | {c: [-big, -big] for c in ((0, 0), (0, 1))}
    cells_obj = from_frame(unit_frame(cells))
    with pytest.raises(CodedError) as exc:
        estimate_overlap_effects(cells_obj, contrasts=("joint_launch",))
    assert exc.value.code == "overlap.inference.unrepresentable"
    assert exc.value.context["nonfinite"] == ("estimate",)
