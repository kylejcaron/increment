"""The prior-weight probe: grid shape, decision rule, and one live replication."""

from __future__ import annotations

from fractions import Fraction as F

import pytest


def _cells(power, null=0.01):
    """Synthetic completed cells: power[(p0, w)] at the 2000-per-arm look for lift 0.10."""
    cells = []
    for (p0, weight), value in power.items():
        for lift in ("0", "0.05", "0.1", "0.2"):
            cumulative = [0.0] * 7 + [null if lift == "0" else value] * 9
            cells.append(
                {
                    "declared_rate": p0,
                    "true_rate": p0,
                    "lift": lift,
                    "weight": weight,
                    "replications": 200,
                    "cumulative_crossing": cumulative,
                }
            )
    return cells


def test_grid_has_every_declared_cell_once():
    from scripts.probe_bernoulli_prior_weight import cell_grid

    cells = cell_grid(replications=200)
    keys = {(c["declared_rate"], c["true_rate"], c["lift"], c["weight"]) for c in cells}
    assert len(cells) == len(keys) == 3 * 4 * 5 + 3 * 2 * 4
    assert {c["weight"] for c in cells} == {None, 2, 10, 50, 200}
    misspecified = [c for c in cells if c["true_rate"] != c["declared_rate"]]
    assert all(F(c["true_rate"]) == 2 * F(c["declared_rate"]) for c in misspecified)
    assert {c["lift"] for c in misspecified} == {"0", "0.1"}
    assert all(c["weight"] is not None for c in misspecified)


def test_choose_weight_prefers_the_smallest_weight_within_five_points_of_the_best():
    from scripts.probe_bernoulli_prior_weight import choose_weight

    power = {}
    for p0 in ("0.02", "0.1", "0.3"):
        power[(p0, 2)] = 0.60
        power[(p0, 10)] = 0.76
        power[(p0, 50)] = 0.80
        power[(p0, 200)] = 0.78
        power[(p0, None)] = 0.55
    assert choose_weight(_cells(power)) == 10


def test_choose_weight_excludes_a_weight_whose_null_rate_exceeds_alpha_anywhere():
    from scripts.probe_bernoulli_prior_weight import choose_weight

    power = {}
    for p0 in ("0.02", "0.1", "0.3"):
        power[(p0, 2)] = 0.60
        power[(p0, 10)] = 0.80
        power[(p0, 50)] = 0.80
        power[(p0, 200)] = 0.80
        power[(p0, None)] = 0.55
    cells = _cells(power)
    for cell in cells:
        if cell["weight"] == 10 and cell["lift"] == "0" and cell["declared_rate"] == "0.3":
            cell["cumulative_crossing"][-1] = 0.06
    assert choose_weight(cells) == 50


def test_choose_weight_falls_back_to_mean_power_when_no_weight_wins_everywhere():
    """Each declared rate favours a different weight by more than the five-point
    tolerance, so no weight is within tolerance of the best at every rate and
    `choose_weight` must fall back to averaging power across rates."""
    from scripts.probe_bernoulli_prior_weight import choose_weight

    power = {
        ("0.02", 2): 0.30,
        ("0.02", 10): 0.40,
        ("0.02", 50): 0.95,
        ("0.02", 200): 0.50,
        ("0.1", 2): 0.30,
        ("0.1", 10): 0.95,
        ("0.1", 50): 0.40,
        ("0.1", 200): 0.50,
        ("0.3", 2): 0.95,
        ("0.3", 10): 0.40,
        ("0.3", 50): 0.50,
        ("0.3", 200): 0.30,
    }
    # Mean power: w2=0.517, w10=0.583, w50=0.617, w200=0.433. The best (w50)
    # wins at no single declared rate jointly with the others, so the smallest
    # weight within five points of the best mean (w10, w50) is the answer.
    assert choose_weight(_cells(power)) == 10


@pytest.mark.slow
def test_run_cell_drives_the_registered_route_and_agrees_with_the_public_decision():
    from scripts.probe_bernoulli_prior_weight import run_cell

    result = run_cell(
        {
            "declared_rate": "0.3",
            "true_rate": "0.3",
            "lift": "0.2",
            "weight": 10,
            "replications": 1,
            "looks": 2,
            "batch": 250,
            "seed": 3,
        }
    )
    assert len(result["cumulative_crossing"]) == 2
    assert result["public_decision_agrees"] is True
