"""Coverage of the delta-method conversion route and of the hybrid `auto` pipeline near its
dense-count threshold (``calibration.conversion_route``).

The delta-method interval is a function of the four counts alone, so its noncoverage in a
cell is an exact sum over the binomial lattice (no sampling error); the simulations run the
production route itself, count pair by count pair. A fast smoke checks route labels and
coverage in a few comfortably dense cells and one sparse cell; the slow tests check the
boundary excess at the shipped threshold and the hybrid pipeline across the threshold.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict
from fractions import Fraction

import numpy as np
import pytest
from scipy.stats import binom

from calibration import conversion_route as cr
from increment._literals import Alternative
from increment.estimation.conversion_delta import delta_interval, production_decision
from increment.estimation.conversion_route import dense_min_count, route_for_counts
from tests.estimation._conversion_counts import lift_row
from tests.mc import scientific_delta

# --- the vectorised interval is the production interval ----------------------------------


def test_the_vectorised_interval_is_the_production_interval():
    gap, compared, interpolation = cr.conformance(samples=60)
    assert compared == 60
    assert gap < 1e-9
    assert interpolation < cr.CRITICAL_AGREEMENT


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
            cr.Cell("rare", 1_000_000, 1_000_000, 0.0067, 0.0067, 1.0, 3.0),
            # failure-limited
            cr.Cell("failure", 14_000, 14_000, 0.5, 0.5, 1.0, 3.0),
            # central
            cr.Cell("central", 24_000, 24_000, 0.3, 0.33, 1.1, 3.0),
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

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_the_production_pipeline_reproduces_the_exact_noncoverage(self, alternative):
        """``estimate_lift``, requested as each alternative, on ``replicates(tail)`` seeded draws
        agrees with the sum within four Monte Carlo standard errors."""
        assert cr.replicate_check(0.1, workers=1, alternative=alternative)

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


class TestLatticeSumsStreamOverBlocks:
    """Every sum over a count lattice streams over blocks of control counts so that a worker's
    footprint does not grow with the arms' windows: the block size moves the footprint, never the
    value (a cell of ten million units per arm has a lattice of tens of millions of cells)."""

    #: Expected counts of the control arm (90) sit at the routing floor, so about half the pairs
    #: of the lattice are decided by each route.
    CELL = cr.Cell("central", 300, 600, 0.3, 0.36, 1.2, 90.0)
    FLOOR = 90

    @staticmethod
    def _small_blocks(monkeypatch) -> None:
        monkeypatch.setattr(cr, "_BLOCK_CELLS", 700)

    def test_boundary_noncoverage_does_not_depend_on_the_block_size(self, monkeypatch):
        tails = (0.025, 0.1)
        whole = cr.boundary_noncoverage(self.CELL, tails, self.FLOOR)
        self._small_blocks(monkeypatch)
        blocked = cr.boundary_noncoverage(self.CELL, tails, self.FLOOR)
        for tail in tails:
            assert blocked[tail] == pytest.approx(whole[tail], rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("alternative", ["two-sided", "greater"])
    def test_hybrid_noncoverage_does_not_depend_on_the_block_size(self, monkeypatch, alternative):
        def result():
            return cr.hybrid_noncoverage(
                self.CELL, alpha=0.1, threshold=self.FLOOR, alternative=alternative
            )

        whole = result()
        self._small_blocks(monkeypatch)
        blocked = result()
        assert blocked.lower == pytest.approx(whole.lower, rel=1e-12, abs=0.0)
        assert blocked.upper == pytest.approx(whole.upper, rel=1e-12, abs=0.0)
        assert blocked.asymptotic_share == pytest.approx(whole.asymptotic_share, rel=1e-12)
        assert 0.2 < whole.asymptotic_share < 0.8

    def test_the_finite_sample_set_does_not_depend_on_the_block_size(self, monkeypatch):
        whole = cr.finite_sample_misses(self.CELL, alpha=0.1, alternative="two-sided")
        self._small_blocks(monkeypatch)
        blocked = cr.finite_sample_misses(self.CELL, alpha=0.1, alternative="two-sided")
        np.testing.assert_array_equal(blocked[0], whole[0])
        np.testing.assert_array_equal(blocked[1], whole[1])

    def test_enumeration_does_not_depend_on_the_block_size(self, monkeypatch):
        monkeypatch.setattr(cr, "dense_min_count", lambda tail: self.FLOOR)
        design = cr.MirrorCell("bound", 300, 0.3, 0.1, 0.2, "two-sided")
        whole = cr.enumerate_design(design)
        self._small_blocks(monkeypatch)
        blocked = cr.enumerate_design(design)
        for name in ("asymptotic_share", "asymptotic_part", "finite_part", "hybrid"):
            assert getattr(blocked, name) == pytest.approx(getattr(whole, name), rel=1e-12)
        assert whole.floor == self.FLOOR
        assert whole.hybrid == pytest.approx(whole.asymptotic_part + whole.finite_part)
        assert 0.2 < whole.asymptotic_share < 0.8
        assert whole.asymptotic_part > 0.0
        assert whole.finite_part > 0.0

    def test_only_the_pairs_the_finite_route_decides_are_replayed(self):
        """With a routing floor the replay skips every pair the delta method decides and leaves it
        unrejecting; at every other pair it is the whole-window replay."""
        from increment.power._binomial import _window

        cell = self.CELL
        key = cr._decision_key(cell, alpha=0.1, alternative="two-sided")
        window_c, window_t = _window(cell.n_c, cell.p_c), _window(cell.n_t, cell.p_t)
        plus, minus, _, _ = cr.finite_sample_misses(cell, alpha=0.1, alternative="two-sided")
        skipped = list(cr._finite_blocks(key, window_c, window_t, self.FLOOR))
        replayed_plus = np.concatenate([p for _, p, _ in skipped])
        replayed_minus = np.concatenate([m for _, _, m in skipped])
        x_c = np.arange(window_c.lo, window_c.hi + 1)[:, None]
        x_t = np.arange(window_t.lo, window_t.hi + 1)[None, :]
        routed = (
            np.minimum(np.minimum(x_c, cell.n_c - x_c), np.minimum(x_t, cell.n_t - x_t))
            >= self.FLOOR
        )
        assert routed.any()
        assert not routed.all()
        assert plus[~routed].any()
        np.testing.assert_array_equal(replayed_plus[~routed], plus[~routed])
        np.testing.assert_array_equal(replayed_minus[~routed], minus[~routed])
        assert not replayed_plus[routed].any()
        assert not replayed_minus[routed].any()


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPlanningAccuracy:
    """Planned power against the pipeline's exact rejection probability, summed over the count
    lattice: each route meets its own claim (``bound_holds``). A dense plan is within the closed
    form's tolerance, a sparse plan is the replay, and a borderline plan leans conservative."""

    @pytest.mark.parametrize("factor", [0.5, 0.9, 1.04, 1.25, 1.5, 3.0])
    @pytest.mark.parametrize(
        ("alpha", "alternative"), [(0.2, "two-sided"), (0.1, "greater")], ids=["two", "greater"]
    )
    def test_planned_power_meets_its_routes_claim_against_the_pipelines_rejection_probability(
        self, factor, alpha, alternative
    ):
        tail = alpha / 2.0 if alternative == "two-sided" else alpha
        n = math.ceil(factor * dense_min_count(tail) / 0.3)
        power = cr.enumerated_power(cr.MirrorCell("bound", n, 0.3, 0.1, alpha, alternative))
        assert cr.bound_holds(power)


def _requests(grid) -> set[tuple[float, float, str, bool, float]]:
    """``(tail, alpha, alternative, lift is a rise, control rate)`` of each design of ``grid``."""
    return {
        (
            c.alpha / 2.0 if c.alternative == "two-sided" else c.alpha,
            c.alpha,
            c.alternative,
            c.lift > 0.0,
            c.p_c,
        )
        for c in grid
    }


class TestBoundGridAndRule:
    def test_the_enumerated_grids_are_distinct_designs_with_both_signs_of_lift(self):
        """The original grid is 240 designs and the extended grid 180, the sets the saved
        checkpoints hold; the extended grid adds rare-event and negative-lift designs, each tested
        at a directional level of its own sign."""
        grids = {g: cr.bound_cells(g) for g in ("original", "extended", "tails", "window")}
        assert (len(grids["original"]), len(grids["extended"])) == (240, 180)
        for grid in grids.values():
            assert len({(c.n, c.p_c, c.lift, c.alpha, c.alternative) for c in grid}) == len(grid)
        assert all(c.lift > 0.0 for c in grids["original"])
        extended = grids["extended"]
        assert {c.alternative for c in extended if c.lift < 0.0} == {"two-sided", "less"}
        assert {c.alternative for c in extended if c.lift > 0.0} == {"two-sided", "greater"}
        assert all(0.0 < abs(c.lift) < 0.5 for c in (*grids["tails"], *grids["window"]))

    def test_the_production_requests_produce_the_calibrated_tails(self):
        tails = {tail for alpha in cr.PRODUCTION_ALPHAS for tail in (alpha / 2.0, alpha)}
        assert tails == set(cr.TAILS)
        requests = {(alpha, alt) for t in cr.TAILS for alpha, alt, _ in cr.production_requests(t)}
        assert requests == {
            (alpha, alt)
            for alpha in cr.PRODUCTION_ALPHAS
            for alt in ("two-sided", "greater", "less")
        }
        for tail in cr.TAILS:
            for _, alternative, sign in cr.production_requests(tail):
                assert sign == {"two-sided": 1, "greater": 1, "less": -1}[alternative]

    @pytest.mark.parametrize("grid", ["tails", "window"])
    def test_the_tail_grids_test_the_actual_production_requests_in_their_directions(self, grid):
        """Each production tail is tested at every request that produces it, never at an alpha
        invented to reach it: two-sided against a rise, and each directional request against a
        rise (``greater``) and a fall (``less``), at a central and a rare-event rate."""
        found = _requests(cr.bound_cells(grid))
        tails = {tail for tail, *_ in found}
        assert tails <= set(cr.TAILS)
        if grid == "window":
            assert tails == set(cr.TAILS)
        for tail in tails:
            expected = {
                (tail, alpha, alternative, sign > 0, rate)
                for alpha, alternative, sign in cr.production_requests(tail)
                for rate in (0.3, 0.01)
            }
            assert {entry for entry in found if entry[0] == tail} == expected

    def test_the_grids_together_reach_every_production_request_at_every_tail(self):
        found = set().union(
            *(_requests(cr.bound_cells(g)) for g in ("original", "extended", "tails"))
        )
        for tail in cr.TAILS:
            for alpha, alternative, sign in cr.production_requests(tail):
                assert any(entry[:4] == (tail, alpha, alternative, sign > 0) for entry in found), (
                    tail,
                    alpha,
                    alternative,
                )

    def test_the_window_grid_is_all_borderline_and_sweeps_the_routed_share(self):
        from increment.estimation.conversion_route import planning_route, routed_share

        swept: dict[tuple[float, float, str, bool], list[float]] = {}
        for c in cr.bound_cells("window"):
            tail = c.alpha / 2.0 if c.alternative == "two-sided" else c.alpha
            p_t = c.p_c * (1.0 + c.lift)
            assert planning_route(c.n, c.n, c.p_c, p_t, tail_alpha=tail, mode="auto") == (
                "borderline"
            )
            swept.setdefault((tail, c.p_c, c.alternative, c.lift > 0.0), []).append(
                routed_share(c.n, c.n, c.p_c, p_t, tail_alpha=tail)
            )
        assert len(swept) == 2 * sum(len(cr.production_requests(t)) for t in cr.TAILS)
        for shares in swept.values():
            assert shares == sorted(shares)
            assert shares[0] < 1e-4
            assert shares[-1] > 1.0 - 1e-4
            assert any(0.2 < share < 0.8 for share in shares)

    @pytest.mark.parametrize(
        ("alpha", "alternative"),
        [(a, alt) for a in cr.PRODUCTION_ALPHAS for alt in ("two-sided", "greater", "less")],
    )
    def test_the_recorded_routing_level_is_the_level_the_procedure_compiles(
        self, alpha, alternative
    ):
        from increment.estimation.arm_contract import ArmPlanningProcedure

        procedure = ArmPlanningProcedure.standard(
            "conversion", alpha=alpha, alternative=alternative
        )
        assert cr._tail(alpha, alternative) == procedure.compiled_tail_alpha


def _power(
    planned,
    hybrid,
    *,
    lower=0.0,
    upper=1.0,
    closed_form=False,
    certified=True,
    omitted=1e-12,
    inflation=1.0,
):
    """A design whose pipeline mass sums to ``hybrid``, planned at ``planned`` with the
    enclosure ``[lower, upper]``."""
    plan = cr.Plan(
        "model", "borderline", "approximate", planned, lower, upper, 0.0, closed_form, certified
    )
    return cr.EnumeratedPower(
        plan,
        cr.Enumeration("construction", 412, 0.5, hybrid / 2, hybrid / 2, omitted, inflation),
    )


class TestBoundRule:
    """A closed-form plan is judged to ``DENSE_AGREEMENT``, every other plan by whether its
    enclosure meets the pipeline's interval at the enumeration's own numerical error and no
    more."""

    @pytest.mark.parametrize(
        ("margin", "holds"), [(0.004, True), (-0.004, True), (0.007, False), (-0.011, False)]
    )
    def test_a_closed_form_plan_is_within_the_closed_forms_agreement(self, margin, holds):
        assert cr.bound_holds(_power(0.5, 0.5 + margin, closed_form=True)) is holds

    @pytest.mark.parametrize(
        ("lower", "upper", "holds"),
        [
            (0.5, 0.5 + 1e-9, True),
            (0.1, 0.9, True),
            (0.5 + 5e-13, 0.9, True),  # within the omitted mass
            (0.5 + 1e-10, 0.9, False),
            (0.5 + 1e-5, 0.9, False),  # inside the 3e-4 allowance this rule replaced
            (0.1, 0.4, False),
        ],
    )
    def test_an_enumerated_plan_must_enclose_the_pipeline_without_an_allowance(
        self, lower, upper, holds
    ):
        assert cr.bound_holds(_power(0.5, 0.5, lower=lower, upper=upper)) is holds

    def test_a_plan_that_is_not_certified_makes_no_claim_to_hold(self):
        assert not cr.bound_holds(_power(0.5, 0.5, lower=0.0, upper=1.0, certified=False))

    def test_the_numerical_error_of_the_enumeration_is_the_only_slack(self):
        lower = 0.5 * (1.0 + 5e-10)
        assert not cr.bound_holds(_power(lower, 0.5, lower=lower, upper=0.9))
        inflated = _power(lower, 0.5, lower=lower, upper=0.9, inflation=1.0 + 1e-9)
        assert cr.bound_holds(inflated)

    def test_the_interval_is_around_the_summed_mass(self):
        power = _power(0.5, 0.5, inflation=1.0 + 1e-9, omitted=1e-6)
        runtime = power.enumeration
        assert runtime.lower < 0.5 < runtime.upper
        assert runtime.upper >= 0.5 * (1.0 + 1e-9) + 1e-6
        assert runtime.lower <= 0.5 / (1.0 + 1e-9)


class TestRoutedPairsAreDecidedByTheRuntimesOwnRow:
    """An enumeration restates no part of the delta-method decision: each routed pair is the row
    ``estimate_lift`` reports at that pair's counts."""

    @pytest.mark.parametrize(
        ("alpha", "alternative", "lift"),
        [(0.2, "two-sided", 0.1), (0.1, "greater", 0.1), (0.1, "less", -0.06)],
    )
    def test_the_decision_of_a_routed_pair_is_the_estimate_lift_row(self, alpha, alternative, lift):
        design = cr.MirrorCell("bound", 1500, 0.3, lift, alpha, alternative)
        lattice = cr._design_lattice(design)
        x_c = np.arange(430, 471)[:, None]
        x_t = lattice.x_t[:, ::5]
        floor = lattice.floor
        routed = np.minimum(np.minimum(x_c, 1500 - x_c), np.minimum(x_t, 1500 - x_t)) >= floor
        sub = dataclasses.replace(lattice, x_t=x_t)
        rejects = cr._delta_rejects(design, sub, x_c, routed)
        assert routed.any() and not routed.all()
        assert rejects[routed].any() and not rejects[routed].all()
        assert not rejects[~routed].any()
        for i, j in np.argwhere(routed)[::7]:
            counts = (int(x_c[i, 0]), 1500, int(x_t[0, j]), 1500)
            lift_estimate = lift_row(counts, alpha=alpha, alternative=alternative).require_lift()
            expected = (
                alternative != "less" and lift_estimate.lb is not None and lift_estimate.lb > 0.0
            ) or (
                alternative != "greater" and lift_estimate.ub is not None and lift_estimate.ub < 0.0
            )
            assert bool(rejects[i, j]) is expected, counts

    def test_a_pair_that_is_not_routed_is_never_a_delta_rejection(self):
        design = cr.MirrorCell("bound", 300, 0.3, 0.1, 0.2, "two-sided")
        lattice = cr._design_lattice(design)
        x_c = np.arange(60, 70)[:, None]
        routed = np.zeros((x_c.size, lattice.x_t.size), bool)
        assert not cr._delta_rejects(design, lattice, x_c, routed).any()


class TestHybridNoncoverageDecidesRoutedPairsByTheRuntimesRow:
    """A routed pair of ``hybrid_noncoverage`` misses the true lift where the row ``estimate_lift``
    reports for it rejects against that lift (``stat_sig``), whichever side the alternative
    reads. The vectorised formula of the same interval compares on the log scale, and differs
    from the runtime's lift-scale comparison where an interval end equals the true lift to the
    last bit; the cells below are built so that one does."""

    N = 3_000
    TAIL = 0.1
    #: Counts either side of the boundary pair that the sum runs over: the runtime row costs one
    #: computation per pair.
    REACH = 2
    #: Spacing of the doubles in ``[1, 2)``: a lift on this grid is exact as a ratio minus one.
    GRID = 2.0**-52

    @classmethod
    def _boundary(cls, alternative: Alternative, side: str) -> tuple[cr.Cell, int, int]:
        """``(cell, x_c, x_t)``: a cell at control rate one half whose true lift lies within a few
        doubles of the ``side`` end of the runtime's interval at ``(x_c, x_t)``, chosen so that the
        vectorised formula and the runtime compare that end with the lift differently."""
        n, tail, index = cls.N, cls.TAIL, 0 if side == "plus" else 1
        x_c = n // 2
        # Pairs about two standard deviations from the cell their interval end names, so the
        # count window of that cell holds them.
        pairs = range(1_630, 1_730) if side == "plus" else range(1_430, 1_560)
        for x_t in pairs:
            interval = delta_interval(x_c, n, x_t, n, tail=tail, alternative=alternative)
            assert interval is not None
            log_lower, log_upper = cr.delta_log_bounds(np.array([x_c]), n, np.array([x_t]), n, tail)
            for step in range(-3, 4):
                lift = (1.0 + interval[index]) - 1.0 + step * cls.GRID
                p_t = (1.0 + lift) / 2.0
                cell = cr.Cell("central", n, n, 0.5, p_t, 2.0 * p_t, float(dense_min_count(tail)))
                assert cell.lift == lift
                vectorised = (bool(log_lower[0] > cell.truth), bool(log_upper[0] < cell.truth))
                runtime = production_decision(
                    x_c, n, x_t, n, tail=tail, alternative=alternative, null_lift=lift
                )
                if vectorised[index] != runtime[index]:
                    return cell, x_c, x_t
        raise AssertionError("no boundary pair separates the vectorised formula from the runtime")

    @classmethod
    def _restrict_windows(cls, monkeypatch, x_c: int, x_t: int) -> None:
        """Sum over the counts within ``REACH`` of ``(x_c, x_t)`` only."""
        from increment.power._binomial import _Window, _window

        def around(n: int, p: float, centre: int) -> _Window:
            full = _window(n, p)
            lo, hi = centre - cls.REACH, centre + cls.REACH
            assert full.lo <= lo and hi <= full.hi
            weights = full.weights[lo - full.lo : hi - full.lo + 1]
            return _Window(lo, hi, 0.0, weights, full.error)

        monkeypatch.setattr(
            cr,
            "_count_windows",
            lambda cell: (around(cell.n_c, cell.p_c, x_c), around(cell.n_t, cell.p_t, x_t)),
        )

    @pytest.mark.parametrize(
        ("alternative", "side"),
        [("two-sided", "plus"), ("two-sided", "minus"), ("greater", "plus"), ("less", "minus")],
    )
    def test_the_noncoverage_of_routed_pairs_is_the_runtime_rows_verdict_at_the_true_lift(
        self, monkeypatch, alternative, side
    ):
        n, reach = self.N, self.REACH
        cell, x_c, x_t = self._boundary(alternative, side)
        alpha = 2.0 * self.TAIL if alternative == "two-sided" else self.TAIL
        self._restrict_windows(monkeypatch, x_c, x_t)
        result = cr.hybrid_noncoverage(
            cell, alpha=alpha, threshold=dense_min_count(self.TAIL), alternative=alternative
        )
        lower = upper = mass = 0.0
        for c in range(x_c - reach, x_c + reach + 1):
            for t in range(x_t - reach, x_t + reach + 1):
                weight = float(binom.pmf(c, n, cell.p_c) * binom.pmf(t, n, cell.p_t))
                row = lift_row(
                    (c, n, t, n), alpha=alpha, alternative=alternative, null_lift=cell.lift
                )
                assert row.reference_kind == "t"
                interval = row.require_lift()
                rejected = row.stat_sig()
                mass += weight
                lower += weight * (rejected and interval.lb is not None and interval.lb > cell.lift)
                upper += weight * (rejected and interval.ub is not None and interval.ub < cell.lift)
        assert result.asymptotic_share == pytest.approx(mass, rel=1e-9)
        assert result.lower == pytest.approx(lower, rel=1e-9, abs=0.0)
        assert result.upper == pytest.approx(upper, rel=1e-9, abs=0.0)
        assert (lower if side == "plus" else upper) > 0.0

    @pytest.mark.parametrize(("alternative", "unread"), [("greater", "upper"), ("less", "lower")])
    def test_a_side_the_alternative_does_not_read_never_misses(self, alternative, unread):
        cell = cr.Cell("central", 300, 600, 0.3, 0.36, 1.2, 90.0)
        result = cr.hybrid_noncoverage(cell, alpha=0.1, threshold=90, alternative=alternative)
        read = "lower" if unread == "upper" else "upper"
        assert getattr(result, unread) == 0.0
        assert getattr(result, read) > 0.0
        assert 0.0 < result.asymptotic_share < 1.0


def _only_the_pair(monkeypatch, x_c: int, x_t: int) -> None:
    """Have ``hybrid_noncoverage`` sum over one count pair, whose weight is one."""
    from increment.power._binomial import _Window

    def window(count: int) -> _Window:
        return _Window(count, count, 0.0, np.ones(1), 0.0)

    monkeypatch.setattr(cr, "_count_windows", lambda cell: (window(x_c), window(x_t)))


class TestADirectionalRequestMissesOnlyOnTheSideItReads:
    """The seeded simulation of ``estimate_lift`` and the exact sum count the same misses: a
    directional request reads one end of its interval, and the delta-method route reports both."""

    TAIL = 0.1
    REPS = 150
    SEED = 20261004
    #: Counts whose delta-method interval at the 0.1 tail is about ``(0.022, 0.079)``.
    COUNTS = (3_000, 10_000, 3_150, 10_000)

    @staticmethod
    def _cell(risk_ratio: float) -> cr.Cell:
        """The central boundary cell of the 0.1 tail with equal arms and ``risk_ratio``: about
        half its draws are routed to the delta method."""
        return next(
            c
            for c in cr.cells(dense_min_count(0.1))
            if c.family == "central" and c.n_c == c.n_t and c.risk_ratio == risk_ratio
        )

    @pytest.mark.parametrize(
        ("alternative", "risk_ratio"),
        [("greater", 0.5), ("greater", 1.25), ("less", 0.5), ("less", 1.25)],
    )
    def test_the_simulated_misses_are_the_exact_decisions_of_the_same_draws(
        self, monkeypatch, alternative, risk_ratio
    ):
        cell, floor = self._cell(risk_ratio), dense_min_count(self.TAIL)
        simulated = cr.simulate_hybrid(
            cell, alpha=self.TAIL, alternative=alternative, reps=self.REPS, seed=self.SEED
        )
        x_c, x_t = cr.draw_counts(cell, reps=self.REPS, seed=self.SEED)
        lower = upper = routed = 0.0
        for c, t in zip(x_c.tolist(), x_t.tolist(), strict=True):
            _only_the_pair(monkeypatch, c, t)
            pair = cr.hybrid_noncoverage(
                cell, alpha=self.TAIL, threshold=floor, alternative=alternative
            )
            lower, upper, routed = (
                lower + pair.lower,
                upper + pair.upper,
                routed + pair.asymptotic_share,
            )
        assert (simulated.lower_misses, simulated.upper_misses) == (lower, upper)
        assert round(simulated.asymptotic_share * self.REPS) == routed
        assert 0 < routed < self.REPS
        unread = simulated.upper_misses if alternative == "greater" else simulated.lower_misses
        assert unread == 0

    @pytest.mark.parametrize("lift", [0.0, 0.05, 0.1])
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_a_row_misses_where_the_runtime_rejects_against_the_true_lift(
        self, alternative, lift
    ):
        alpha = 2.0 * self.TAIL if alternative == "two-sided" else self.TAIL
        row = lift_row(self.COUNTS, alpha=alpha, alternative=alternative)
        assert row.reference_kind == "t"
        misses = cr.row_misses(row, lift)
        runtime = lift_row(self.COUNTS, alpha=alpha, alternative=alternative, null_lift=lift)
        assert any(misses) == runtime.stat_sig()
        assert misses == production_decision(
            *self.COUNTS, tail=self.TAIL, alternative=alternative, null_lift=lift
        )

    @pytest.mark.parametrize(
        ("alternative", "lift", "far_end"), [("greater", 0.1, "ub"), ("less", 0.0, "lb")]
    )
    def test_the_end_a_directional_request_does_not_read_is_no_miss(
        self, alternative, lift, far_end
    ):
        row = lift_row(self.COUNTS, alpha=self.TAIL, alternative=alternative)
        end = getattr(row.require_lift(), far_end)
        # The row reports that end, wholly on the miss side of the lift it is scored against.
        assert end is not None
        assert end < lift if far_end == "ub" else end > lift
        assert cr.row_misses(row, lift) == (False, False)


class TestEnumerationEnclosesTheExactMass:
    """The pipeline's rejection probability over the lattice, in exact rational arithmetic,
    lies in the interval the enumeration reports; the interval is not vacuous."""

    @staticmethod
    def _exact_pmf(n: int, p: float, lo: int, hi: int) -> list[Fraction]:
        rate = Fraction(p)
        return [math.comb(n, k) * rate**k * (1 - rate) ** (n - k) for k in range(lo, hi + 1)]

    @pytest.mark.parametrize(
        "design",
        [
            cr.MirrorCell("bound", 300, 0.3, 0.1, 0.2, "two-sided"),
            cr.MirrorCell("bound", 240, 0.05, 0.4, 0.1, "greater"),
        ],
        ids=["central", "rare"],
    )
    def test_the_interval_holds_the_exact_rational_mass(self, monkeypatch, design):
        monkeypatch.setattr(cr, "dense_min_count", lambda tail: 14)
        lattice = cr._design_lattice(design)
        window_c, window_t = lattice.window_c, lattice.window_t
        exact_c = self._exact_pmf(design.n, design.p_c, window_c.lo, window_c.hi)
        exact_t = self._exact_pmf(
            design.n, design.p_c * (1.0 + design.lift), window_t.lo, window_t.hi
        )
        mass = Fraction(0)
        row = 0
        for block in cr._decided_blocks(design, lattice):
            decided = np.where(block.routed, block.delta_routed, block.plus | block.minus)
            for i, j in zip(*np.nonzero(decided), strict=True):
                mass += exact_c[row + i] * exact_t[j]
            row += block.weight.shape[0]
        omitted = 1 - sum(exact_c) * sum(exact_t)
        enumeration = cr.enumerate_design(design)
        assert 0.0 < enumeration.asymptotic_share < 1.0
        assert Fraction(enumeration.lower) <= mass
        assert mass + omitted <= Fraction(enumeration.upper)
        assert omitted <= Fraction(enumeration.omitted)
        assert abs(float(mass) - enumeration.hybrid) <= (enumeration.inflation - 1.0) * mass
        assert enumeration.upper - enumeration.lower < 1e-9


def _record(cell: cr.MirrorCell, *, enumeration=None, plan=None) -> dict:
    """A dense design that meets its claim exactly, as a checkpoint line records it."""
    from increment.estimation.results import BINOMIAL_METHOD
    from increment.power.core import BINOMIAL_PLANNING_MODEL

    return {
        "design": cr._design_key(cell),
        "enumeration": {
            "construction": BINOMIAL_METHOD,
            "floor": dense_min_count(cr._tail(cell.alpha, cell.alternative)),
            "asymptotic_share": 1.0,
            "asymptotic_part": 0.5,
            "finite_part": 0.0,
            "omitted": 0.0,
            "inflation": 1.0,
        }
        | (enumeration or {}),
        "planner": {
            "model": BINOMIAL_PLANNING_MODEL,
            "route": "dense",
            "basis": "exact",
            "planned": 0.5,
            "lower": 0.5,
            "upper": 0.5,
            "ambiguous": 0.0,
            "closed_form": False,
            "certified": True,
        }
        | (plan or {}),
    }


class TestBoundCheckpoint:
    """``bound --out`` keeps what its file records of an enumeration that was summed under this
    runtime, plans again what a retired planner model planned, and refuses what names neither
    rather than reusing or relabelling it."""

    @pytest.fixture
    def calls(self, monkeypatch):
        from increment.power.core import BINOMIAL_PLANNING_MODEL

        enumerated: list[cr.MirrorCell] = []
        planned: list[cr.MirrorCell] = []

        def fake_plan(cell):
            planned.append(cell)
            return cr.Plan(
                BINOMIAL_PLANNING_MODEL, "dense", "exact", 0.5, 0.5, 0.5, 0.0, False, True
            )

        def fake_enumerated(cell):
            enumerated.append(cell)
            record = _record(cell)
            return cr.EnumeratedPower(fake_plan(cell), cr.Enumeration(**record["enumeration"]))

        monkeypatch.setattr(cr, "plan_design", fake_plan)
        monkeypatch.setattr(cr, "enumerated_power", fake_enumerated)
        return enumerated, planned

    @staticmethod
    def _write(path, records) -> None:
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def test_a_checkpoint_of_the_current_model_resumes_without_recomputing(self, tmp_path, calls):
        path = tmp_path / "bound.jsonl"
        self._write(path, [_record(cell) for cell in cr.bound_cells("original")])
        before = path.read_text()
        assert cr.bound(workers=1, out=path) == 0
        assert calls == ([], [])
        assert path.read_text() == before

    def test_only_the_designs_a_checkpoint_lacks_are_enumerated(self, tmp_path, calls):
        enumerated, planned = calls
        designs = cr.bound_cells("extended")
        path = tmp_path / "bound.jsonl"
        self._write(path, [_record(cell) for cell in designs[:5]])
        assert cr.bound(workers=1, out=path, grid="extended") == 0
        assert enumerated == list(designs[5:])
        enumerated.clear()
        planned.clear()
        assert cr.bound(workers=1, out=path, grid="extended") == 0
        assert (enumerated, planned) == ([], [])

    @pytest.mark.parametrize("retired", ["borderline_minimum", "hybrid_finite_plus_delta_v1"])
    def test_a_retired_model_keeps_the_enumeration_and_plans_again(
        self, tmp_path, calls, monkeypatch, retired
    ):
        from increment.power.core import BINOMIAL_PLANNING_MODEL

        enumerated, planned = calls
        designs = cr.bound_cells("extended")[:4]
        monkeypatch.setattr(cr, "bound_cells", lambda grid: designs)
        kept = {"asymptotic_part": 0.1234, "finite_part": 0.4321, "inflation": 1.0 + 1e-12}
        # A retired record's planned figure sits above its own lower end, as a central mass does.
        stale = {"model": retired, "planned": 0.5 + 1e-9, "lower": 0.5, "upper": 0.5 + 2e-9}
        path = tmp_path / "bound.jsonl"
        self._write(path, [_record(cell, enumeration=kept, plan=stale) for cell in designs])
        cr.bound(workers=1, out=path)
        assert enumerated == []
        assert planned == list(designs)
        resumed = cr.read_checkpoint(path)
        for cell in designs:
            power = resumed[json.dumps(cr._design_key(cell))]
            assert power.plan.model == BINOMIAL_PLANNING_MODEL
            assert power.plan.planned == 0.5
            assert power.plan.planned != stale["planned"]
            assert power.enumeration == cr.Enumeration(
                **_record(cell, enumeration=kept)["enumeration"]
            )
        planned.clear()
        cr.bound(workers=1, out=path)
        assert (enumerated, planned) == ([], [])

    @pytest.mark.parametrize(
        ("change", "code", "named"),
        [
            (
                {"enumeration": {"construction": "binomial_bb_difference_v2"}},
                "calibration.conversion_route.checkpoint_runtime",
                {"construction": "binomial_bb_difference_v2"},
            ),
            (
                {"enumeration": {"floor": 411}},
                "calibration.conversion_route.checkpoint_runtime",
                {"floor": 411},
            ),
            (
                {"plan": {"model": "an_unrecorded_model"}},
                "calibration.conversion_route.checkpoint_planner_model",
                {"model": "an_unrecorded_model"},
            ),
            (
                {"plan": {"model": None}},
                "calibration.conversion_route.checkpoint_layout",
                {"section": "planner"},
            ),
            (
                {"enumeration": {"inflation": "1"}},
                "calibration.conversion_route.checkpoint_layout",
                {"section": "enumeration"},
            ),
        ],
        ids=["other_construction", "other_routing_floor", "unknown_model", "no_model", "mistyped"],
    )
    def test_a_record_of_another_runtime_or_an_unknown_model_is_refused(
        self, tmp_path, calls, change, code, named
    ):
        path = tmp_path / "bound.jsonl"
        self._write(path, [_record(cr.bound_cells("original")[0], **change)])
        before = path.read_text()
        with pytest.raises(cr.CheckpointError) as refused:
            cr.bound(workers=1, out=path)
        context = refused.value.context
        assert refused.value.code == code
        assert (context["path"], context["line"]) == (str(path), 1)
        assert context["next_action"] == "recompute_to_new_out"
        assert {key: context[key] for key in named} == named
        assert calls == ([], [])
        assert path.read_text() == before

    @pytest.mark.parametrize(
        ("text", "code"),
        [
            ('{"design": 0, "power"', "calibration.conversion_route.checkpoint_unreadable"),
            ("[1, 2]", "calibration.conversion_route.checkpoint_unreadable"),
            (
                json.dumps({"design": [1], "enumeration": {}, "planner": {}}),
                "calibration.conversion_route.checkpoint_design",
            ),
        ],
        ids=["truncated", "not_an_object", "not_a_design"],
    )
    def test_a_line_that_is_no_record_is_refused_with_the_line_to_remove(
        self, tmp_path, calls, text, code
    ):
        path = tmp_path / "bound.jsonl"
        path.write_text(text + "\n")
        with pytest.raises(cr.CheckpointError) as refused:
            cr.bound(workers=1, out=path)
        assert refused.value.code == code
        assert refused.value.context["line"] == 1
        assert refused.value.context["next_action"] == "remove_line"
        assert calls == ([], [])
        assert path.read_text() == text + "\n"

    @pytest.mark.parametrize(
        "power",
        [
            {
                "route": "dense",
                "planned": 0.5,
                "basis": "asymptotic",
                "asymptotic_share": 1.0,
                "asymptotic": 0.5,
                "asymptotic_part": 0.5,
                "finite_part": 0.0,
                "omitted": 0.0,
            },
            {
                "route": "dense",
                "planned": 0.5,
                "basis": "asymptotic",
                "asymptotic_share": 1.0,
                "asymptotic": 0.5,
                "finite_sample": 0.4,
                "hybrid": 0.5,
                "omitted": 0.0,
            },
        ],
        ids=["split_power", "unsplit_power"],
    )
    @pytest.mark.parametrize("design", [0, [3567, 0.3, 0.05, 0.05, "two-sided"]])
    def test_a_record_naming_neither_runtime_nor_planner_is_refused_with_the_way_forward(
        self, tmp_path, calls, power, design
    ):
        """A line of an earlier layout holds one undated power section. Nothing in it says what
        planned it or which runtime summed it, so none of it is reused, relabelled or silently
        enumerated again: the refusal is an incompatible layout that locates the line and names
        the next action, enumerating to another file, which resumes from nothing."""
        enumerated, _ = calls
        designs = cr.bound_cells("original")
        path = tmp_path / "bound.jsonl"
        earlier = {"design": design, "power": power}
        self._write(path, [_record(cell) for cell in designs[:2]] + [earlier])
        before = path.read_text()
        with pytest.raises(cr.CheckpointError) as refused:
            cr.bound(workers=1, out=path)
        context = refused.value.context
        assert refused.value.code == "calibration.conversion_route.checkpoint_layout"
        located = ("path", "line", "section", "found", "next_action")
        assert {key: context[key] for key in located} == {
            "path": str(path),
            "line": 3,
            "section": "record",
            "found": ("design", "power"),
            "next_action": "recompute_to_new_out",
        }
        assert enumerated == []
        assert path.read_text() == before
        assert cr.bound(workers=1, out=tmp_path / "recomputed.jsonl") == 0
        assert enumerated == list(designs)

    def test_a_plans_enclosure_and_its_certification_survive_a_checkpoint(self, tmp_path):
        designs = cr.bound_cells("original")[:2]
        path = tmp_path / "bound.jsonl"
        held = {"route": "borderline", "lower": 0.25, "upper": 0.5, "ambiguous": 1.5e-7}
        heuristic = {"lower": 0.0, "upper": 1.0, "certified": False, "closed_form": True}
        self._write(path, [_record(designs[0], plan=held), _record(designs[1], plan=heuristic)])
        resumed = cr.read_checkpoint(path)
        first, second = (resumed[json.dumps(cr._design_key(cell))].plan for cell in designs)
        assert (first.lower, first.upper, first.ambiguous) == (0.25, 0.5, 1.5e-7)
        assert first.certified and not first.closed_form
        assert (second.lower, second.upper) == (0.0, 1.0)
        assert not second.certified and second.closed_form

    def test_a_command_refuses_an_unreadable_checkpoint_with_a_status(self, tmp_path, capsys):
        path = tmp_path / "bound.jsonl"
        path.write_text('{"design": 0, "power"\n')
        assert cr.main(["bound", "--out", str(path)]) == 2
        assert "calibration.conversion_route.checkpoint_unreadable" in capsys.readouterr().err


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestResumedPlanEqualsAFreshOne:
    """Planning again from a retired model's checkpoint gives the plan a fresh run of the same
    design gives, on the enumeration the checkpoint kept."""

    @pytest.mark.parametrize("retired", ["borderline_minimum", "hybrid_finite_plus_delta_v1"])
    def test_a_resumed_design_matches_a_fresh_one(self, tmp_path, monkeypatch, retired):
        design = cr.MirrorCell("bound", 300, 0.3, 0.1, 0.2, "two-sided")
        monkeypatch.setattr(cr, "bound_cells", lambda grid: (design,))
        fresh = cr.enumerated_power(design)
        # An enumerated plan publishes the lower end of its enclosure; a retired record's
        # planned figure is the central mass above it.
        assert not fresh.plan.closed_form
        assert fresh.plan.planned == fresh.plan.lower
        stale = {"model": retired, "planned": fresh.plan.planned * (1.0 + 1e-9), "basis": "exact"}
        path = tmp_path / "bound.jsonl"
        path.write_text(
            json.dumps(_record(design, enumeration=asdict(fresh.enumeration), plan=stale)) + "\n"
        )

        def refuse(cell):
            raise AssertionError("a kept enumeration must not be summed again")

        monkeypatch.setattr(cr, "enumerate_design", refuse)
        cr.bound(workers=1, out=path)
        resumed = cr.read_checkpoint(path)[json.dumps(cr._design_key(design))]
        assert resumed.plan == fresh.plan
        assert resumed.plan.planned != stale["planned"]
        assert resumed.enumeration == fresh.enumeration
        assert resumed.margin == fresh.margin


# --- the requirement is read only from complete measurements -----------------------------


def _scan(start: float, excess: dict[int, float], tail: float = 0.01, *, stop: float = 40_000.0):
    """Rows a ``select`` scan of ``ladder(start, stop)`` holds at ``tail`` for the steps of
    ``excess``."""
    anchor = cr.Anchor.recorded(start, stop)
    return {
        m: {tail: cr.Measured(cr.Excess(tail, m, value, value, None), anchor)}
        for m, value in excess.items()
    }


def _merged(*scans):
    rows: dict[int, dict[float, cr.Measured]] = {}
    for scan in scans:
        for m, by_tail in scan.items():
            rows.setdefault(m, {}).update(by_tail)
    return rows


class TestRequiredCount:
    LADDER = tuple(cr.ladder(1000.0, 2400.0))

    def _from_ladder(self, values):
        return _scan(1000.0, dict(zip(self.LADDER, values, strict=False)))

    def test_the_default_ladder_is_the_rounded_geometric_one(self):
        assert self.LADDER == (1000, 1150, 1322, 1521, 1749, 2011, 2313)
        assert cr.ladder(10.0, 60.0)[:3] == [10, 12, 13]

    def test_a_complete_scan_returns_the_first_step_from_which_every_step_passes(self):
        rows = self._from_ladder((1.4, 1.2, 0.9, 0.8, 0.7, 0.6, 0.5))
        assert cr.required_count(rows, 0.01) == 1322

    def test_a_passing_step_the_scan_stops_at_is_not_a_requirement(self):
        """1,749 passes but its window reaches 2,011, which was never measured."""
        rows = self._from_ladder((1.4, 1.2, 1.1, 1.05, 0.7))
        assert cr.required_count(rows, 0.01) is None
        assert "2011" in cr.unmet(rows, 0.01)

    def test_a_gap_in_the_measured_steps_is_not_bridged(self):
        rows = _scan(1000.0, {1000: 1.4, 1150: 0.9, 1749: 0.8, 2011: 0.7, 2313: 0.6})
        assert cr.required_count(rows, 0.01) == 1749

    def test_steps_missing_from_a_merged_row_set_leave_no_requirement(self):
        rows = self._from_ladder((0.9, 0.8, 0.7, 0.6, 0.5))
        rows[1150] = {}
        rows[1521] = {}
        assert cr.required_count(rows, 0.01) is None

    def test_a_scan_interrupted_at_its_first_default_step_leaves_no_requirement(self):
        """Step 10 of the default ladder is followed by 12, which lies inside ``MARGIN`` times 10:
        a scan that stopped at 10 has measured nothing of that window."""
        assert cr.required_count(_scan(10.0, {10: 0.5}), 0.01) is None

    def test_a_default_ladder_step_the_scan_skipped_is_not_bridged(self):
        ladder = cr.ladder(10.0, 60.0)
        assert cr.required_count(_scan(10.0, dict.fromkeys(ladder, 0.5)), 0.01) == 10
        rows = _scan(10.0, dict.fromkeys([step for step in ladder if step != 12], 0.5))
        assert cr.required_count(rows, 0.01) == 13

    def test_rows_of_another_start_do_not_stand_in_for_a_step_the_default_ladder_skipped(self):
        """The default ladder runs 10, 12, 13 and a ladder from 11 runs 11, 13, 15: measured
        steps 10, 11 and 13 leave 12 unmeasured for the default ladder's window at 10."""
        assert cr.ladder(11.0, 40.0)[:3] == [11, 13, 15]
        default, shifted = _scan(10.0, {10: 0.5}), _scan(11.0, {11: 0.5, 13: 0.5})
        assert cr.required_count(default, 0.01) is None
        assert cr.required_count(_merged(default, shifted), 0.01) == 11

    def test_a_rung_measured_by_any_scan_counts_for_every_ladder_that_has_it(self):
        """The rung 12 of ladders from 10.0 and from 10.2 is one count, measured once."""
        assert cr.ladder(10.2, 40.0)[:3] == [10, 12, 13]
        rows = _merged(_scan(10.0, {10: 0.5, 13: 0.5}), _scan(10.2, {12: 0.5}))
        assert cr.required_count(rows, 0.01) == 10


class TestAnchor:
    def test_a_recorded_start_gives_its_ladder_through_a_limit(self):
        anchor = cr.Anchor.recorded(10.0, 40_000.0)
        assert anchor.window(12, 15.0) == (12, 13, 15)
        assert anchor.window(11, 13.75) is None  # 11 is no step of this ladder

    def test_a_ladder_needs_a_positive_start_below_its_stop(self):
        for start, stop in ((0.0, 10.0), (-1.0, 10.0), (10.0, 10.0), (math.inf, math.inf)):
            with pytest.raises(ValueError, match="start < stop"):
                cr.Anchor.recorded(start, stop)

    @pytest.mark.parametrize("start", [10.0, 10.3, 1637.0, 2489.5, 12345.6])
    def test_the_steps_of_a_ladder_determine_the_starts_that_made_them(self, start):
        steps = cr.ladder(start, 20 * start)[:12]
        anchor = cr.anchor_of(steps)
        assert anchor is not None
        assert anchor.low <= start <= anchor.high
        for edge in (anchor.low, anchor.high):
            assert cr.ladder(edge, 20 * start)[:12] == steps
        below, above = math.nextafter(anchor.low, 0.0), math.nextafter(anchor.high, math.inf)
        assert cr.ladder(below, 20 * start)[:12] != steps
        assert cr.ladder(above, 20 * start)[:12] != steps

    @pytest.mark.parametrize(
        "steps",
        [[10, 13], [10, 12, 15], [10, 12, 12], [1637, 1883, 2165, 2490, 2863, 3292, 3786, 4354]],
        ids=["skips_a_rung", "skips_a_later_rung", "repeats_a_rung", "two_phases"],
    )
    def test_steps_that_no_single_start_makes_have_no_ladder(self, steps):
        assert cr.anchor_of(steps) is None

    def test_the_same_steps_can_be_one_ladder_or_two(self):
        """10, 11, 13 are a ladder from just under 10 (9.6), whose next rung after 10 is 11."""
        anchor = cr.anchor_of([10, 11, 13])
        assert anchor is not None
        assert anchor.window(10, 12.5) == (10, 11)

    def test_later_rungs_are_those_every_start_of_the_ladder_agrees_on(self):
        anchor = cr.anchor_of([1637, 1883, 2165])
        assert anchor is not None
        assert anchor.window(2165, 2706.25) == (2165, 2490)
        assert anchor.window(2490, 3112.5) is None  # 2863 or 2864, by start


class TestScanFiles:
    """``select`` files are read as the steps of ladders: a record names its ladder, and a file
    of records that name none is placed on the ladder its own steps determine."""

    TAILS = (0.01, 0.005)
    # Worst-excess columns in the shape of the saved scans of the 0.01 and 0.005 tails: a first
    # scan interrupted at 2,165 and a continuation from 2,490 on another rounding of the ladder.
    FIRST = {0.01: [1.03, 1.01, 0.99], 0.005: [1.37, 1.34, 1.31]}
    CONTINUATION = {
        0.01: [0.97, 0.73, 0.72, 0.71, 0.69],
        0.005: [1.29, 0.97, 0.96, 0.94, 0.92],
    }

    @staticmethod
    def _write(path, steps, excess, ladder=None):
        records = []
        for index, m in enumerate(steps):
            for tail, values in excess.items():
                record = {
                    "m": m,
                    "tail": tail,
                    "delta": 0.001,
                    "wald_excess": values[index],
                    "routed_excess": values[index],
                    "cell": None,
                }
                if ladder is not None:
                    record["ladder"] = {"start": ladder[0], "stop": ladder[1]}
                records.append(record)
        path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def _saved_shapes(self, tmp_path, *, record_ladder):
        first, continuation = tmp_path / "first.jsonl", tmp_path / "continuation.jsonl"
        self._write(
            first,
            cr.ladder(1637.0, 16_000.0)[:3],
            self.FIRST,
            (1637.0, 16_000.0) if record_ladder else None,
        )
        self._write(
            continuation,
            cr.ladder(2489.5, 16_000.0)[:5],
            self.CONTINUATION,
            (2489.5, 16_000.0) if record_ladder else None,
        )
        return first, continuation

    @pytest.mark.parametrize("record_ladder", [False, True], ids=["unnamed", "named"])
    def test_a_continuation_on_another_rounding_completes_the_window_it_continues(
        self, tmp_path, record_ladder
    ):
        """The continuation's first two steps are the first scan's next rungs (2,490 and 2,863
        or 2,864), and its later ones are another rounding of the ladder (3,292 against 3,293):
        each requirement is read on the ladder of the step that is its candidate."""
        first, continuation = self._saved_shapes(tmp_path, record_ladder=record_ladder)
        assert cr.ladder(2489.5, 16_000.0)[:5] == [2490, 2863, 3292, 3786, 4354]
        assert cr.ladder(1637.0, 16_000.0)[:7] == [1637, 1883, 2165, 2490, 2863, 3293, 3786]
        rows = cr.read_rows([first, continuation])
        assert cr.required_count(rows, 0.01) == 2165
        assert cr.required_count(rows, 0.005) == 2863
        alone = cr.read_rows([first])
        assert cr.required_count(alone, 0.01) is None
        assert "2490" in cr.unmet(alone, 0.01)

    def test_a_named_ladder_and_one_derived_from_the_steps_give_the_same_rows(self, tmp_path):
        named = cr.read_rows(list(self._saved_shapes(tmp_path, record_ladder=True)))
        unnamed = cr.read_rows(list(self._saved_shapes(tmp_path, record_ladder=False)))
        for rows in (named, unnamed):
            for m, by_tail in rows.items():
                for tail, measured in by_tail.items():
                    assert measured.anchor.window(m, cr.MARGIN * m) is not None, (m, tail)
        assert {(m, t): v.excess for m, bt in named.items() for t, v in bt.items()} == {
            (m, t): v.excess for m, bt in unnamed.items() for t, v in bt.items()
        }

    def test_steps_that_are_no_ladder_are_refused(self, tmp_path):
        path = tmp_path / "skipped.jsonl"
        self._write(path, [10, 12, 15, 17], {0.01: [1.0] * 4})
        with pytest.raises(cr.ScanError, match=r"skipped\.jsonl.*first 2 do.*step 15"):
            cr.read_rows([path])

    def test_a_file_of_two_scans_that_name_no_ladder_is_refused(self, tmp_path):
        path = tmp_path / "joined.jsonl"
        self._write(path, [10, 12, 13, 10, 12], {0.01: [1.0] * 5})
        with pytest.raises(cr.ScanError, match="recurs"):
            cr.read_rows([path])

    def test_a_step_that_is_not_on_the_ladder_it_names_is_refused(self, tmp_path):
        path = tmp_path / "forged.jsonl"
        self._write(path, [10, 11], {0.01: [1.0, 1.0]}, ladder=(10.0, 40_000.0))
        with pytest.raises(cr.ScanError, match="step 11 is not a step of the ladder it names"):
            cr.read_rows([path])

    @pytest.mark.parametrize("ladder", [{"start": 10.0}, {"start": "x", "stop": 4.0}, 7])
    def test_a_ladder_that_is_not_a_start_and_a_stop_is_refused(self, tmp_path, ladder):
        path = tmp_path / "named.jsonl"
        record = {"m": 10, "tail": 0.01, "wald_excess": 1.0, "routed_excess": 1.0, "cell": None}
        path.write_text(json.dumps(record | {"ladder": ladder}) + "\n")
        with pytest.raises(cr.ScanError, match="not a start and a stop"):
            cr.read_rows([path])

    def test_a_scan_writes_its_ladder_with_every_step_and_reads_back(self, tmp_path, monkeypatch):
        def measured(m, tails, workers=1):
            return {tail: cr.Excess(tail, m, 0.5, 0.25, None) for tail in tails}

        monkeypatch.setattr(cr, "worst_excess", measured)
        path = tmp_path / "scan.jsonl"
        assert cr.select(path, workers=1, start=10.0, stop=60.0, tails=(0.01,)) == 0
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [record["m"] for record in records] == cr.ladder(10.0, 60.0)
        assert {json.dumps(record["ladder"]) for record in records} == {
            json.dumps({"start": 10.0, "stop": 60.0})
        }
        rows = cr.read_rows([path])
        assert set(rows) == set(cr.ladder(10.0, 60.0))
        assert rows[12][0.01].anchor == cr.Anchor.recorded(10.0, 60.0)

    def test_the_required_command_reports_a_requirement_or_why_there_is_none(
        self, tmp_path, capsys
    ):
        first, continuation = self._saved_shapes(tmp_path, record_ladder=False)
        assert cr.main(["required", str(first), str(continuation)]) == 0
        found = {
            line["tail"]: line for line in map(json.loads, capsys.readouterr().out.splitlines())
        }
        assert found[0.01]["required_wald"] == 2165
        assert found[0.005]["required_wald"] == 2863
        assert "unmet" not in found[0.01]
        assert cr.main(["required", str(first)]) == 0
        alone = {
            line["tail"]: line for line in map(json.loads, capsys.readouterr().out.splitlines())
        }
        assert alone[0.01]["required_wald"] is None
        assert "2490" in alone[0.01]["unmet"]["wald"]

    def test_the_required_command_refuses_an_unreadable_scan_with_a_status(self, tmp_path, capsys):
        path = tmp_path / "skipped.jsonl"
        self._write(path, [10, 13], {0.01: [1.0, 1.0]})
        assert cr.main(["required", str(path)]) == 2
        assert str(path) in capsys.readouterr().err
