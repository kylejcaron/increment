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


class TestRouteLabelSmoke:
    """The fast tier's deterministic check of the route each count pair takes."""

    @pytest.mark.parametrize(
        ("counts", "kind"),
        [
            ((3_000, 10_000, 3_150, 10_000), "t"),
            ((60, 10_000, 90, 10_000), "binomial"),
            ((0, 10_000, 12, 10_000), "binomial"),
        ],
        ids=["dense", "sparse", "zero_control"],
    )
    def test_a_count_pair_is_labelled_by_the_route_it_took(self, counts, kind):
        assert lift_row(counts).reference_kind == kind
        assert route_for_counts(*counts, tail_alpha=0.025, mode="auto") == (
            "asymptotic" if kind == "t" else "finite_sample"
        )


@pytest.mark.slow
@pytest.mark.parameter_recovery
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
    """The pipeline (each count pair routed by its counts, then decided by the delta method or
    the finite-sample inversion) keeps its unconditional per-tail noncoverage within
    ``scientific_delta`` of the tail at counts straddling the threshold. The noncoverage is an
    exact sum over the count lattice, so no draw count limits its resolution."""

    @pytest.mark.parametrize("tail", [0.1, 0.05])
    @pytest.mark.parametrize("offset", cr.OFFSETS)
    def test_noncoverage_stays_within_the_tolerance_at_the_threshold(self, tail, offset):
        shipped = dense_min_count(tail)
        cell = next(c for c in cr.cells(shipped + offset) if c.family == "central")
        result = cr.hybrid_noncoverage(cell, alpha=2.0 * tail, threshold=shipped)
        assert max(result.lower, result.upper) <= tail + scientific_delta(tail) + result.omitted
        assert result.lower + result.upper <= (
            2.0 * tail + scientific_delta(2.0 * tail) + result.omitted
        )
        # The cell straddles the rule: some draws are routed each way.
        assert 0.0 < result.asymptotic_share < 1.0

    def test_the_production_pipeline_reproduces_the_exact_noncoverage(self):
        """``estimate_lift`` on ``replicates(tail)`` seeded draws agrees with the sum within
        four Monte Carlo standard errors."""
        assert cr.replicate_check(0.1, workers=1)

    def test_the_replayed_finite_sample_set_is_the_production_set_at_its_edge(self):
        compared, _edge, hard = cr.finite_conformance(per_tail=3, tails=(0.1, 0.05))
        assert compared > 0
        assert hard == 0

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


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPlanningBound:
    """Planned power against the pipeline's exact rejection probability, summed over the count
    lattice: a borderline plan is a bound (no larger), a sparse plan is the replay, and a dense
    plan is within the closed form's tolerance."""

    @pytest.mark.parametrize("factor", [0.5, 0.9, 1.04, 1.25, 1.5, 3.0])
    @pytest.mark.parametrize(
        ("alpha", "alternative"), [(0.2, "two-sided"), (0.1, "greater")], ids=["two", "greater"]
    )
    def test_planned_power_never_exceeds_the_pipelines_rejection_probability(
        self, factor, alpha, alternative
    ):
        tail = alpha / 2.0 if alternative == "two-sided" else alpha
        n = math.ceil(factor * dense_min_count(tail) / 0.3)
        power = cr.enumerated_power(cr.MirrorCell("bound", n, 0.3, 0.1, alpha, alternative))
        slack = 1e-9 + power.omitted
        if power.route == "dense":
            assert abs(power.margin) <= cr.DENSE_AGREEMENT + slack
        elif power.route == "sparse":
            assert abs(power.margin) <= slack
        else:
            assert power.margin >= -slack


# --- the requirement is read only from complete measurements -----------------------------


def _rows(excess: dict[int, float], tail: float = 0.01) -> dict[int, dict[float, cr.Excess]]:
    return {m: {tail: cr.Excess(tail, m, value, value, None)} for m, value in excess.items()}


class TestRequiredCount:
    LADDER = (1000, 1150, 1323, 1521, 1749)

    def test_a_complete_scan_returns_the_first_step_from_which_every_step_passes(self):
        rows = _rows(dict(zip(self.LADDER, (1.4, 1.2, 0.9, 0.8, 0.7), strict=True)))
        assert cr.required_count(rows, 0.01) == 1323

    def test_a_passing_step_the_scan_stops_at_is_not_a_requirement(self):
        """1,749 passes but nothing above it was measured through ``MARGIN`` times it."""
        rows = _rows(dict(zip(self.LADDER, (1.4, 1.2, 1.1, 1.05, 0.7), strict=True)))
        assert cr.required_count(rows, 0.01) is None

    def test_a_gap_in_the_measured_steps_is_not_bridged(self):
        rows = _rows({1000: 1.4, 1150: 0.9, 1749: 0.8, 2011: 0.7, 2313: 0.6})
        assert cr.required_count(rows, 0.01) == 1749

    def test_steps_missing_from_a_merged_row_set_leave_no_requirement(self):
        rows = _rows(dict(zip(self.LADDER, (0.9, 0.8, 0.7, 0.6, 0.5), strict=True)))
        rows[1150] = {}
        rows[1521] = {}
        assert cr.required_count(rows, 0.01) is None
