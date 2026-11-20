"""Contracts for the private fraction-harmed interval study."""

from fractions import Fraction

import pytest

from calibration.harm_bounds import (
    Cell,
    Counts4,
    StudyError,
    _weight_extreme,
    decide,
    learn,
    summarize,
)

_ZERO = Cell(0, 0, 0, 0)
_U0 = (0, 0, 0, 0)
_F = (0, 0, 1, 1)


def test_same_marginals_different_people() -> None:
    # Always/never-success and all-flip populations have the same two marginals.
    bands = []
    for _c00, c01, c10, c11 in ((5, 0, 0, 5), (0, 5, 5, 0)):
        counts = (Cell(10, c10 + c11, 10, c01 + c11), _ZERO, _ZERO, _ZERO)
        bands.append(summarize(counts, _U0, atom_weights=None))
    assert bands[0] == bands[1]
    band = bands[0]
    assert band.plugin == (0.0, 0.5)
    assert band.L_ci == pytest.approx((0.0, 0.799153276087), abs=1e-11, rel=0)
    assert band.U_ci == pytest.approx((0.100423361956, 0.899576638044), abs=1e-11, rel=0)
    assert band.outer == (band.L_ci[0], band.U_ci[1])


def test_projected_endpoints_and_weight_simplex() -> None:
    # With w[0] constrained to [.15,.40], w[1] must be its complement.
    coefficients = (0.10, 0.50)
    limits = ((0.15, 0.40), (0.0, 1.0))
    assert _weight_extreme(coefficients, limits, maximize=False) == pytest.approx(0.34)
    assert _weight_extreme(coefficients, limits, maximize=True) == pytest.approx(0.44)

    # Pool prevalence over both arms; known target mass is not enrollment mass.
    counts = (Cell(100, 100, 300, 0), _ZERO, Cell(200, 0, 200, 200), _ZERO)
    estimated = summarize(counts, _F, atom_weights=None)
    known = summarize(counts, _F, atom_weights=(Fraction(1, 4), 0, Fraction(3, 4), 0))
    assert estimated.weight_mode == "estimated"
    assert known.weight_mode == "known"
    assert estimated.plugin == (0.50, 0.50)
    assert known.plugin == (0.25, 0.25)
    assert known.outer[1] < estimated.outer[0]


def test_train_map_and_empty_evaluation_cell() -> None:
    assert learn((_ZERO, _ZERO, _ZERO, _ZERO), omit_b=False) == _F
    train = (Cell(10, 0, 10, 10), Cell(10, 10, 10, 0), _ZERO, _ZERO)
    learned = learn(train, omit_b=False)
    assert learned == (0, 1, 2, 2)
    assert learn(train, omit_b=True) == _F

    # Leaf 1 has no treatment observations. Its possible contribution stays in the band.
    for y0 in (0, 50):
        counts = (Cell(50, 10, 50, 40), Cell(50, y0, 0, 0), Cell(50, 10, 50, 40), _ZERO)
        band = summarize(counts, learned, atom_weights=None)
        assert band.leaf_map == learned
        assert band.leaf_counts == counts[:3]
        assert band.plugin is None
        assert band.reason == "empty_arm_leaf"
        assert band.outer[0] < band.outer[1]
    assert summarize(counts, learned, atom_weights=(0.5, 0, 0.5, 0)).plugin == (0.0, 0.2)


def test_decide_requires_strict_separation() -> None:
    outer = (0.125, 0.20)
    assert decide(outer, 0.20) == "inconclusive"
    assert decide(outer, 0.125) == "inconclusive"
    assert decide(outer, 0.20 + 1e-13) == "inconclusive"
    assert decide(outer, 0.25) == "clear"
    assert decide(outer, 0.10) == "rule_out"


@pytest.mark.parametrize("cell", [Cell(-1, 0, 1, 0), Cell(2, 3, 1, 0), Cell(True, 0, 1, 0)])
def test_invalid_counts_name_the_coordinate(cell: Cell) -> None:
    counts = (_ZERO, cell, _ZERO, _ZERO)
    with pytest.raises(StudyError) as exc:
        summarize(counts, _U0, atom_weights=None)
    assert exc.value.reason == "invalid_counts"
    context = dict(exc.value.context)
    assert context["stage"] == "summarize"
    assert context["coordinate"] == "01"
    assert "count" in context


def test_empty_study_is_not_an_empty_arm_leaf() -> None:
    with pytest.raises(StudyError) as exc:
        summarize((_ZERO, _ZERO, _ZERO, _ZERO), _U0, atom_weights=None)
    assert exc.value.reason == "empty_study"


@pytest.mark.parametrize("weights", [(0.5, 0, 0, 0), (float("nan"), 1, 0, 0), (-1.0, 2, 0, 0)])
def test_invalid_target_is_not_renormalized(weights: tuple[float, float, float, float]) -> None:
    counts: Counts4 = (Cell(1, 0, 1, 0), _ZERO, _ZERO, _ZERO)
    with pytest.raises(StudyError) as exc:
        summarize(counts, _U0, atom_weights=weights)
    assert exc.value.reason == "invalid_target_mass"


@pytest.mark.parametrize("tau", [0.0, 1.0, float("inf")])
def test_invalid_tolerance_never_yields_an_action(tau: float) -> None:
    with pytest.raises(StudyError) as exc:
        decide((0.1, 0.9), tau)
    assert exc.value.reason == "invalid_tolerance"
