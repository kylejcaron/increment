"""Planning a conversion contrast the runtime routes by counts is planning the union of its two
decisions: the delta-method test on the routed rectangle and the finite-sample test elsewhere.

Each case is checked against the runtime's own decision (``estimate_lift``) at matched counts
and against an independent assembly of the two routes, never against either route's power.
"""

from __future__ import annotations

import numpy as np
import pytest

from increment._literals import Alternative
from increment.errors import InvalidRequestError
from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.estimation.conversion_delta import production_decision
from increment.estimation.conversion_route import dense_min_count, planning_route
from increment.power import (
    Baseline,
    _binomial,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
)
from increment.power._binomial import PLANNING_CELL_CEILING, RejectionGeometry, Routing
from increment.power.core import BINOMIAL_PLANNING_MODEL, _binomial_key, planned_enclosure
from tests.estimation._conversion_counts import lift_row

# One-sided tail 0.1 (alpha 0.2 two-sided): the routing floor is 412 counts, the smallest the
# rule has, so a lattice of a few tens of thousands of cells holds the whole boundary.
ALPHA, TAIL = 0.2, 0.1
N, P, LIFT = 1_000, 0.45, 0.1


def _procedure(alternative: Alternative = "two-sided") -> ArmPlanningProcedure:
    return ArmPlanningProcedure.standard(
        "conversion", alpha=ALPHA if alternative == "two-sided" else TAIL, alternative=alternative
    )


def test_the_planning_construction_is_named():
    assert BINOMIAL_PLANNING_MODEL == "hybrid_finite_plus_delta_v1"


@pytest.mark.slow
class TestDensePlansNearTheThresholdAreEnumerated:
    """The closed form understated the pipeline's rejection probability by 0.006 at the two
    smallest dense designs; the enumerated production decision has no such error."""

    @pytest.mark.parametrize(
        ("alpha", "alternative", "recorded"),
        [(0.2, "two-sided", 0.59184), (0.1, "greater", 0.58895)],
    )
    def test_the_recorded_witnesses_are_the_pipelines_exact_probability(
        self, alpha, alternative, recorded
    ):
        procedure = ArmPlanningProcedure.standard(
            "conversion", alpha=alpha, alternative=alternative
        )
        result = achieved_power(1_236, 0.06, Baseline.from_proportion(0.5), procedure)
        assert result.power_basis == "exact"
        assert result.power == pytest.approx(recorded, abs=1e-5)

    def test_a_lattice_too_large_to_enumerate_keeps_the_closed_form(self):
        result = achieved_power(5_000_000, 0.05, Baseline.from_proportion(0.3), _procedure())
        assert result.power_basis == "asymptotic"


class TestBorderlinePlansAreTheUnionOfBothRoutes:
    def _geometry(self) -> tuple[RejectionGeometry, _binomial.BinomialDecision, int]:
        key = _binomial_key(_procedure(), N, N)
        floor = dense_min_count(key.tail_alpha)
        geometry = RejectionGeometry(key, "exact", PLANNING_CELL_CEILING, Routing(floor, 0.0))
        return geometry, key, floor

    def test_the_classification_is_borderline_and_the_figure_exact(self):
        assert planning_route(N, N, P, P * (1 + LIFT), tail_alpha=TAIL, mode="auto") == "borderline"
        result = achieved_power(N, LIFT, Baseline.from_proportion(P), _procedure())
        assert result.power_basis == "exact"

    def test_the_power_is_the_runtimes_delta_decision_on_the_rectangle_plus_the_replay_elsewhere(
        self,
    ):
        """Assembled apart from the plan: the vectorised delta decision over the routed
        rectangle, and the finite-sample replay (a geometry with no routing) over the rest."""
        _, key, floor = self._geometry()
        p_t = P * (1 + LIFT)
        wc, wt = _binomial._window(N, P), _binomial._window(N, p_t)
        weights = np.outer(wc.weights, wt.weights)
        x_c = np.arange(wc.lo, wc.hi + 1)[:, None]
        x_t = np.arange(wt.lo, wt.hi + 1)[None, :]
        routed = (x_c >= floor) & (x_c <= N - floor) & (x_t >= floor) & (x_t <= N - floor)
        rejects = np.zeros(weights.shape, bool)
        for i, j in np.argwhere(routed):
            plus, minus = production_decision(
                int(x_c[i, 0]),
                N,
                int(x_t[0, j]),
                N,
                tail=key.tail_alpha,
                alternative=key.alternative,
                null_lift=0.0,
            )
            rejects[i, j] = plus or minus
        finite = RejectionGeometry(key, "exact")
        for rows in (
            slice(0, floor - wc.lo),
            slice(N - floor + 1 - wc.lo, None),
            slice(max(floor - wc.lo, 0), max(N - floor + 1 - wc.lo, 0)),
        ):
            block = np.zeros(weights.shape, bool)
            block[rows, :] = True
            block &= ~routed
            if not block.any():
                continue
            r = np.flatnonzero(block.any(axis=1))
            c = np.flatnonzero(block.any(axis=0))
            plus, minus = finite.cells(
                wc.lo + int(r[0]), wc.lo + int(r[-1]), wt.lo + int(c[0]), wt.lo + int(c[-1])
            )
            window = (plus | minus) & block[r[0] : r[-1] + 1, c[0] : c[-1] + 1]
            rejects[r[0] : r[-1] + 1, c[0] : c[-1] + 1] |= window
        expected = float(weights[rejects].sum())
        enclosure = planned_enclosure(N, LIFT, Baseline.from_proportion(P), _procedure())
        assert enclosure.power == pytest.approx(expected, abs=1e-12)
        assert enclosure.lower <= expected <= enclosure.upper

    def test_every_decided_pair_is_the_runtimes_decision(self):
        """Pairs on both routes, on either side of the rectangle's edges, equal what the
        runtime's own row decides."""
        geometry, _, floor = self._geometry()
        geometry.evaluate(P, P * (1 + LIFT))
        rng = np.random.default_rng(7)
        pairs = {
            (int(c), int(t))
            for c, t in zip(
                rng.binomial(N, P, 30), rng.binomial(N, P * (1 + LIFT), 30), strict=True
            )
        }
        pairs |= {(floor - 1, 480), (floor, 480), (450, floor - 1), (450, floor), (N - floor, 495)}
        for x_c, x_t in sorted(pairs):
            plus, minus = geometry.cells(x_c, x_c, x_t, x_t)
            runtime = lift_row((x_c, N, x_t, N), alpha=ALPHA, alternative="two-sided").stat_sig()
            assert bool(plus[0, 0] or minus[0, 0]) == runtime, (x_c, x_t)

    def test_a_shifted_null_is_decided_as_the_runtime_decides_it(self):
        """The routed rectangle is tested against ``1 + null_lift``, and so is every pair of the
        replay, down to pairs whose interval end sits on the null."""
        procedure = ArmPlanningProcedure.standard("conversion", alpha=ALPHA, null_lift=0.03)
        key = _binomial_key(procedure, N, N)
        floor = dense_min_count(key.tail_alpha)
        geometry = RejectionGeometry(key, "exact", PLANNING_CELL_CEILING, Routing(floor, 0.03))
        geometry.evaluate(P, P * 1.13)
        for x_c, x_t in [
            (450, 520),
            (450, 540),
            (floor, 540),
            (500, floor),
            (455, 515),
            (445, 525),
        ]:
            plus, minus = geometry.cells(x_c, x_c, x_t, x_t)
            row = lift_row((x_c, N, x_t, N), alpha=ALPHA, null_lift=0.03)
            assert bool(plus[0, 0] or minus[0, 0]) == row.stat_sig(), (x_c, x_t)

    def test_the_closure_bound_holds_the_power_at_every_rate_of_its_interval(self):
        geometry, _, _ = self._geometry()
        rates = [P * (1 + lift) for lift in (0.08, 0.1, 0.12)]
        for rate in rates:
            geometry.evaluate(P, rate)
        bound = geometry.closure_bound(P, rates[0], rates[-1])
        for rate in rates:
            assert bound >= geometry.evaluate(P, rate).upper - 1e-12


class TestPlansTheRuntimeCannotDecide:
    def test_a_sparse_plan_whose_finite_route_is_refused_is_refused_not_given_zero_power(self):
        """Below the endpoint solver's floor the runtime refuses the finite-sample decision at
        every count; a plan that keeps its counts there has no power to report."""
        procedure = ArmPlanningProcedure.standard("conversion", alpha=1e-9)
        with pytest.raises(InvalidRequestError) as raised:
            achieved_power(2_000, 0.5, Baseline.from_proportion(0.1), procedure)
        assert raised.value.code == "power.binomial_tail_level_unrepresentable"

    def test_a_dense_plan_at_that_level_is_planned_and_names_what_it_leaves_undecided(self):
        procedure = ArmPlanningProcedure.standard("conversion", alpha=1e-9)
        enclosure = planned_enclosure(400_000_000, 0.01, Baseline.from_proportion(0.3), procedure)
        assert enclosure.closed_form or enclosure.ambiguous <= _binomial.RESOLUTION


class TestPublicSolversAgree:
    def test_the_sized_plan_reaches_its_target_and_its_effect_is_the_sized_plans(self):
        baseline = Baseline.from_proportion(0.3)
        procedure = _procedure()
        sized = required_sample_size(0.1, baseline, procedure)
        achieved = achieved_power(sized.n_per_arm, 0.1, baseline, procedure)
        assert achieved.power == sized.power >= 0.8
        assert achieved.power_basis == sized.power_basis
        effect = minimum_detectable_effect(sized.n_per_arm, baseline, procedure)
        assert effect.mde_relative is not None and effect.mde_relative <= 0.1 * (1 + 1e-9)
        assert effect.power_basis == sized.power_basis


@pytest.mark.slow
class TestRecordedOverstatementIsEnclosed:
    """The Normal-tail replay put a sparse plan at 382,230 units per arm 1.8e-5 above the
    pipeline's exact rejection probability, 0.41252772628 (enumerated over the count lattice).
    Past the replay budget the enclosure is wide, but it holds it."""

    def test_the_recorded_sparse_design_lies_inside_its_enclosure(self):
        enclosure = planned_enclosure(
            382_230,
            0.05,
            Baseline.from_proportion(0.01),
            ArmPlanningProcedure.standard("conversion", alpha=0.02),
        )
        assert enclosure.lower - 1e-9 <= 0.41252772628 <= enclosure.upper + 1e-9
        assert enclosure.power <= 0.41252772628 + 1e-9
