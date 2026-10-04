"""Coverage of the delta-method conversion route and of the hybrid `auto` pipeline near its
dense-count threshold (``calibration.conversion_route``).

The delta-method interval is a function of the four counts alone, so its noncoverage in a
cell is an exact sum over the binomial lattice (no sampling error); the simulations run the
production route itself, count pair by count pair. A fast smoke checks route labels and
coverage in a few comfortably dense cells and one sparse cell; the slow tests check the
boundary excess at the shipped threshold and the hybrid pipeline across the threshold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from calibration import conversion_route as cr
from increment.estimation.conversion_route import dense_min_count, route_for_counts
from tests.estimation._conversion_counts import lift_row
from tests.mc import scientific_delta

# --- the vectorised interval is the production interval ----------------------------------


def test_the_vectorised_interval_is_the_production_interval():
    gap, compared, interpolation = cr.conformance(samples=60)
    assert compared == 60
    assert gap < 1e-9
    assert interpolation < cr._CRITICAL_AGREEMENT


# --- fast smoke: labels and coverage -----------------------------------------------------


def _noncoverage(cell: cr.Cell, *, alpha: float, reps: int, seed: int):
    return cr.simulate_hybrid(cell, alpha=alpha, alternative="two-sided", reps=reps, seed=seed)


class TestSmoke:
    """Three dense cells and a sparse one, 2,000 seeded replicates each, at ``alpha = 0.05``."""

    REPS = 2_000

    @pytest.mark.parametrize(
        "cell",
        [
            # rare: sparsest expected count three times the threshold
            cr.Cell("rare", 600_000, 600_000, 0.0067, 0.0067, 1.0, 3.0),
            # failure-limited
            cr.Cell("failure", 8_000, 8_000, 0.5, 0.5, 1.0, 3.0),
            # central
            cr.Cell("central", 20_000, 20_000, 0.3, 0.33, 1.1, 3.0),
        ],
        ids=["rare", "failure_limited", "central"],
    )
    def test_a_dense_cell_takes_the_delta_method_route_and_covers_each_tail(self, cell):
        tail = 0.025
        assert min(cell.n_c * cell.p_c, cell.n_c * (1 - cell.p_c)) >= 2.9 * dense_min_count(tail)
        result = _noncoverage(cell, alpha=0.05, reps=self.REPS, seed=20261004)
        assert result.asymptotic_share == 1.0
        se = math.sqrt(tail * (1 - tail) / self.REPS)
        assert result.lower_misses / self.REPS <= tail + 4 * se
        assert result.upper_misses / self.REPS <= tail + 4 * se
        # An interval that missed on neither side would be a degenerate pass.
        assert result.lower_misses + result.upper_misses > 0

    def test_a_sparse_cell_takes_the_finite_sample_route_and_covers_conservatively(self):
        cell = cr.Cell("rare", 300, 300, 0.02, 0.03, 1.5, 6.0)
        result = _noncoverage(cell, alpha=0.05, reps=self.REPS, seed=20261004)
        assert result.asymptotic_share == 0.0
        se = math.sqrt(0.025 * 0.975 / self.REPS)
        assert result.lower_misses / self.REPS <= 0.025 + 4 * se
        assert result.upper_misses / self.REPS <= 0.025 + 4 * se


# --- the shipped threshold at the boundary -----------------------------------------------


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestShippedThreshold:
    """The exact all-draw criterion: at the shipped count and at 1.25 times it, the delta-method
    interval's noncoverage in the binding boundary cells exceeds its tail by no more than
    ``scientific_delta(tail)``."""

    @staticmethod
    def _cells(m: float) -> list[cr.Cell]:
        """Every boundary cell of the rare, failure-limited and central families up to 100,000
        control units (the full table also runs 1e6 and 1e7)."""
        return [cell for cell in cr.cells(m) if cell.n_c <= 100_000]

    @pytest.mark.parametrize("tail", cr.TAILS)
    @pytest.mark.parametrize("margin", [1.0, cr.MARGIN])
    def test_noncoverage_excess_is_within_the_tolerance(self, tail, margin):
        m = math.ceil(margin * dense_min_count(tail))
        delta = scientific_delta(tail)
        worst = max(
            max(
                lower - tail,
                upper - tail,
            )
            for cell in self._cells(m)
            for lower, upper, _, _ in [cr.boundary_noncoverage(cell, (tail,), m)[tail]]
        )
        assert worst <= delta

    def test_the_threshold_is_not_tighter_than_the_all_draw_requirement_it_was_fit_to(self):
        """A looser threshold fails the same criterion: half the shipped count at the extreme
        tail exceeds the tolerance, so the check above has power."""
        tail = 0.0005
        m = dense_min_count(tail) // 2
        delta = scientific_delta(tail)
        worst = max(
            max(lower - tail, upper - tail)
            for cell in self._cells(m)
            for lower, upper, _, _ in [cr.boundary_noncoverage(cell, (tail,), m)[tail]]
        )
        assert worst > delta


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestHybridPipelineAcrossTheThreshold:
    """The production pipeline (``estimate_lift`` routing each draw by its counts, then the delta
    method or the finite-sample inversion) keeps its unconditional per-tail noncoverage within
    ``scientific_delta`` of the tail at counts straddling the threshold, at the largest tail the
    reduced grid affords."""

    REPS = 6_000

    @pytest.mark.parametrize("offset", [-2, -1, 0, 1, 2])
    def test_noncoverage_stays_within_the_tolerance_at_the_threshold(self, offset):
        tail = 0.1
        alpha = 2.0 * tail
        m = dense_min_count(tail) + offset
        cell = cr.cells(m)[0]
        result = cr.simulate_hybrid(
            cell, alpha=alpha, alternative="two-sided", reps=self.REPS, seed=20261004 + offset
        )
        lower, upper = cr.hybrid_upper_bounds(result, family_size=5 * 2)
        assert max(lower, upper) <= tail + scientific_delta(tail)
        assert 0.0 < result.asymptotic_share < 1.0

    def test_every_draw_of_the_hybrid_is_routed_by_its_counts(self):
        """A seeded sweep of the production rows: the label is the count rule's route."""
        rng = np.random.default_rng(20261004)
        tail = 0.1
        m = dense_min_count(tail)
        for _ in range(200):
            x_c = int(rng.integers(m - 20, m + 20))
            x_t = int(rng.integers(m - 20, m + 20))
            n = 3 * m
            counts = (x_c, n, x_t, n)
            route = route_for_counts(*counts, tail_alpha=tail, mode="auto")
            kind = lift_row(counts, alpha=2.0 * tail).reference_kind
            assert kind == ("t" if route == "asymptotic" else "binomial")
