"""An evaluation decides the cells of its own request, whatever the geometry already holds.

The runtime calculations and replays one evaluation may run are bounded (``EVALUATION_ROW_BUDGET``,
``EVALUATION_REPLAY_BUDGET``); a pair the bound leaves out is ambiguous, its mass in the upper end
of the enclosure. A geometry keeps the decisions it made, but they are a cache: the enclosure of a
request is the same on a fresh geometry, on one an earlier evaluation (the same or another) used,
and on a repeat. Every decision below is the runtime's own (`production_decision`, the exact
replay); only the budgets are shrunk so that they bind at a size that runs in seconds.
"""

from __future__ import annotations

import copy
import math
import pickle
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pytest

from calibration.binomial_oracle import Binomial, precise
from increment.errors import InvalidRequestError
from increment.estimation import binomial_rr
from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.estimation.conversion_delta import production_decision
from increment.power import (
    Baseline,
    PowerDesign,
    _binomial,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
    required_sample_size,
)
from increment.power._binomial import (
    PLANNING_CELL_CEILING,
    RESOLUTION,
    BinomialDecision,
    RejectionGeometry,
    Routing,
)
from increment.power.core import planned_enclosure
from tests.power._procedures import make_procedure

# A routed rectangle and a finite-sample remainder inside one 120-per-arm lattice: counts from
# ``FLOOR`` to ``N - FLOOR`` take the runtime's delta calculation, the rest the exact replay. The
# floor is below the production rule's (412 counts) so a lattice of a few thousand cells holds both.
N, P_C, FLOOR, TAIL = 120, 0.3, 20, 0.025
ROW_BUDGET, REPLAY_BUDGET = 300, 500


def _decision(n: int, *, refused: bool = False) -> BinomialDecision:
    """The two-sided runtime decision of ``n`` units per arm; ``refused`` gives it a nuisance
    budget below the endpoint solver's floor, which refuses the finite-sample route in full."""
    beta = _binomial.solver_floor() / 10.0 if refused else binomial_rr.nuisance_beta(2.0 * TAIL)
    decision = BinomialDecision(n, n, 1.0, beta, TAIL, "two-sided")
    assert _binomial.refused(decision) == refused
    return decision


def _geometry() -> RejectionGeometry:
    return _routed_geometry(N, FLOOR)


@dataclass(frozen=True)
class _OffRoute:
    """A geometry routed at ``floor``, the windows of a rate pair, and the mass of the count
    pairs of those windows off the routed rectangle, summed over the pairs from the decimal
    oracle's weights (which share no code with the planner's)."""

    geometry: RejectionGeometry
    wc: _binomial._Window
    wt: _binomial._Window
    mass: Decimal


def _routed_geometry(n: int, floor: int, *, refused: bool = False) -> RejectionGeometry:
    return RejectionGeometry(
        _decision(n, refused=refused), "exact", PLANNING_CELL_CEILING, Routing(floor, 0.0)
    )


def _off_route(
    n: int, floor: int, *, p_c: float = P_C, p_t: float = 0.4, refused: bool = False
) -> _OffRoute:
    geometry = _routed_geometry(n, floor, refused=refused)
    wc, wt = _binomial._window(n, p_c), _binomial._window(n, p_t)
    row_in, col_in = geometry._routed_flags(wc.lo, wc.hi, wt.lo, wt.hi)
    with precise():
        w_c = Binomial(n, p_c).pmf_range(wc.lo, wc.hi)
        w_t = Binomial(n, p_t).pmf_range(wt.lo, wt.hi)
        mass = sum(
            (
                x * y
                for i, x in enumerate(w_c)
                for j, y in enumerate(w_t)
                if not (row_in[i] and col_in[j])
            ),
            Decimal(0),
        )
    return _OffRoute(geometry, wc, wt, mass)


def _known(geometry: RejectionGeometry) -> int:
    return sum(int(segment.known.sum()) for segment in geometry.segments)


def _count_calls(monkeypatch: pytest.MonkeyPatch) -> tuple[list[int], list[int]]:
    """Wrap the runtime calculation and the replay with counters, each still the real one:
    ``(rows, pairs)`` collects one entry per runtime row and the length of every replayed run."""
    rows: list[int] = []
    pairs: list[int] = []
    runtime, classify = _binomial.production_decision, _binomial.classify

    def counted_runtime(*args: Any, **kwargs: Any) -> tuple[bool, bool]:
        rows.append(1)
        return runtime(*args, **kwargs)

    def counted_classify(decision: Any, route: Any, requests: Any) -> Any:
        pairs.extend(request.j1 - request.j0 + 1 for request in requests)
        return classify(decision, route, requests)

    monkeypatch.setattr(_binomial, "production_decision", counted_runtime)
    monkeypatch.setattr(_binomial, "classify", counted_classify)
    return rows, pairs


@pytest.fixture
def small_budgets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_binomial, "EVALUATION_ROW_BUDGET", ROW_BUDGET)
    monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", REPLAY_BUDGET)


class TestAnEvaluationDecidesTheCellsOfItsRequest:
    def test_a_repeated_evaluation_is_the_same_enclosure_and_decides_nothing_more(
        self, monkeypatch, small_budgets
    ):
        rows, pairs = _count_calls(monkeypatch)
        geometry = _geometry()
        first = geometry.evaluate(P_C, 0.4)
        assert first.ambiguous > RESOLUTION, "the budgets bind"
        assert 0 < len(rows) <= ROW_BUDGET
        assert 0 < sum(pairs)
        spent = (len(rows), sum(pairs))
        assert geometry.evaluate(P_C, 0.4) == first
        assert geometry.evaluate(P_C, 0.4) == first
        assert (len(rows), sum(pairs)) == spent

    def test_an_evaluation_after_another_is_a_fresh_geometrys(self, small_budgets):
        """The prior query's window overlaps this one's and its heaviest cells differ, so the
        shared geometry holds cells the fresh one does not: more is cached, nothing more decided."""
        shared = _geometry()
        prior = shared.evaluate(P_C, 0.42)
        after = shared.evaluate(P_C, 0.4)
        fresh = _geometry()
        assert after.ambiguous > RESOLUTION, "the budgets bind"
        assert after == fresh.evaluate(P_C, 0.4)
        assert _known(shared) > _known(fresh)
        assert prior == _geometry().evaluate(P_C, 0.42)
        assert shared.evaluate(P_C, 0.42) == prior
        assert shared.evaluate(P_C, 0.4) == after

    def test_requests_in_either_order_give_each_the_same_enclosure(self, small_budgets):
        ascending, descending = _geometry(), _geometry()
        rates = (0.38, 0.4, 0.42)
        for p_t in rates:
            ascending.evaluate(P_C, p_t)
        for p_t in reversed(rates):
            descending.evaluate(P_C, p_t)
        for p_t in rates:
            expected = _geometry().evaluate(P_C, p_t)
            assert ascending.evaluate(P_C, p_t) == expected
            assert descending.evaluate(P_C, p_t) == expected

    def test_the_budgeted_enclosure_holds_the_probability_every_pair_decided_gives(
        self, monkeypatch
    ):
        """The default budgets decide every pair of this lattice, the exact probability up to the
        numerical error; the shrunken ones leave pairs ambiguous but enclose the same value, and
        what they decide is part of it."""
        reference = _geometry().evaluate(P_C, 0.4)
        assert reference.ambiguous == 0.0
        monkeypatch.setattr(_binomial, "EVALUATION_ROW_BUDGET", ROW_BUDGET)
        monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", REPLAY_BUDGET)
        limited = _geometry().evaluate(P_C, 0.4)
        assert limited.ambiguous > RESOLUTION
        assert limited.power <= reference.power + 1e-12
        assert limited.lower <= reference.power <= limited.upper
        assert limited.basis == "approximate"

    def test_cells_decides_every_pair_asked_whatever_the_evaluation_budgets(
        self, monkeypatch, small_budgets
    ):
        """Pairs the runtime rejects and pairs it does not, more of them than the row budget in
        one call and across calls: each is the runtime's decision."""
        monkeypatch.setattr(_binomial, "EVALUATION_ROW_BUDGET", 40)
        geometry = _geometry()
        for first_row in (25, 27):
            plus, minus = geometry.cells(first_row, first_row + 1, 30, 60)
            assert plus.any() and not plus.all()
            for i, x_c in enumerate((first_row, first_row + 1)):
                for j, x_t in enumerate(range(30, 61)):
                    runtime = production_decision(
                        x_c, N, x_t, N, tail=TAIL, alternative="two-sided", null_lift=0.0
                    )
                    assert (bool(plus[i, j]), bool(minus[i, j])) == runtime, (x_c, x_t)


class TestABudgetLeavesAmbiguousMassNotAnUnavailableRoute:
    """Where the runtime refuses the finite-sample decision in full, the plan has no power only
    when the pairs that route would decide carry probability. Routed pairs the row budget leaves
    out are ambiguous mass of an approximate enclosure, whatever their weight."""

    def test_routed_pairs_the_row_budget_leaves_out_are_ambiguous_not_a_refusal(self, monkeypatch):
        """A floor of 10 leaves the finite route about 2e-9 of the probability here, far below
        half of `RESOLUTION`, so the default budgets decide every routed pair and report ``exact``;
        the shrunken row budget leaves most of the mass ambiguous and the enclosure holds it."""
        reference = _routed_geometry(N, 10, refused=True).evaluate(P_C, 0.4)
        assert reference.ambiguous <= RESOLUTION / 2.0
        assert reference.basis == "exact"
        monkeypatch.setattr(_binomial, "EVALUATION_ROW_BUDGET", ROW_BUDGET)
        geometry = _routed_geometry(N, 10, refused=True)
        limited = geometry.evaluate(P_C, 0.4)
        assert limited.ambiguous > RESOLUTION
        assert limited.basis == "approximate"
        assert limited.lower <= reference.power <= limited.upper
        assert geometry.evaluate(P_C, 0.4) == limited

    def test_pairs_the_refused_route_would_decide_still_leave_the_plan_unavailable(self):
        """A floor of 20 leaves the finite route about 3e-4 of the probability here."""
        with pytest.raises(_binomial.FiniteRouteUnavailable):
            _routed_geometry(N, 20, refused=True).evaluate(P_C, 0.4)


# Off-route masses from 5e-11 to 2e-4 over both arms' tails, small and large arms.
MASS_CASES = [
    pytest.param(120, 0.3, 0.4, 20, id="n120-floor20"),
    pytest.param(120, 0.3, 0.4, 12, id="n120-floor12"),
    pytest.param(120, 0.3, 0.4, 8, id="n120-floor8"),
    pytest.param(120, 0.5, 0.5, 30, id="n120-centre-floor30"),
    pytest.param(120, 0.5, 0.5, 28, id="n120-centre-floor28"),
    pytest.param(300, 0.5, 0.55, 100, id="n300-floor100"),
]


@pytest.mark.parametrize(("n", "p_c", "p_t", "floor"), MASS_CASES)
class TestTheOffRouteMassIsASumOfNonnegativeWeights:
    """The mass of the pairs the finite-sample route keeps is read from the outside weights
    directly. A total less the routed part loses a small tail to the 1e-16 rounding of the
    total (a relative error of 4e-7 at the smallest mass here, far above the 1e-12 the bound
    carries). The bound is compared with the mass of those pairs summed over the pairs
    themselves from the decimal oracle's weights of the same rates."""

    def test_the_bound_encloses_the_direct_mass_and_exceeds_it_by_rounding_only(
        self, n, p_c, p_t, floor
    ):
        case = _off_route(n, floor, p_c=p_c, p_t=p_t)
        bound = Decimal(case.geometry._off_route_bound(case.wc, case.wt))
        assert case.mass <= bound <= case.mass * (1 + Decimal("1e-11"))

    def test_replay_is_chosen_unless_the_bound_is_within_half_of_the_resolution(
        self, monkeypatch, n, p_c, p_t, floor
    ):
        """With half of the resolution at the direct mass the bound is above it, so the
        finite-sample pairs are replayed; a hair above the bound (relative 1e-10, which a
        cancelled tail's error can exceed) they are left undecided."""
        case = _off_route(n, floor, p_c=p_c, p_t=p_t)
        wc, wt = case.wc, case.wt
        row_in, col_in = case.geometry._routed_flags(wc.lo, wc.hi, wt.lo, wt.hi)
        off_route = ~(row_in[:, None] & col_in[None, :])
        monkeypatch.setattr(_binomial, "RESOLUTION", 2.0 * float(case.mass))
        assert (case.geometry._selected(case.wc, case.wt) & off_route).any()
        monkeypatch.setattr(_binomial, "RESOLUTION", 2.0 * float(case.mass) * (1.0 + 1e-10))
        assert not (case.geometry._selected(case.wc, case.wt) & off_route).any()


class TestARefusedRouteIsUnavailableAtTheBoundOfItsDirectMass:
    """At 120 units per arm and a floor of 12 the refused route keeps 3.5e-8 of the probability,
    which the plan is refused at when half of the resolution is not above it."""

    @pytest.mark.parametrize(("scale", "unavailable"), [(1.0, True), (1.05, False)])
    def test_the_plan_is_unavailable_when_half_of_the_resolution_is_not_above_the_mass(
        self, monkeypatch, scale, unavailable
    ):
        monkeypatch.setattr(_binomial, "EVALUATION_ROW_BUDGET", ROW_BUDGET)
        case = _off_route(N, 12, p_c=P_C, p_t=0.4, refused=True)
        monkeypatch.setattr(_binomial, "RESOLUTION", 2.0 * float(case.mass) * scale)
        if unavailable:
            with pytest.raises(_binomial.FiniteRouteUnavailable) as raised:
                case.geometry.evaluate(P_C, 0.4)
            assert raised.value.mass >= float(case.mass)
            return
        enclosure = case.geometry.evaluate(P_C, 0.4)
        assert enclosure.ambiguous >= float(case.mass)


def _conversion() -> ArmPlanningProcedure:
    """A randomized conversion plan pinned to ``finite_sample``: no routed rectangle at this
    size, so the replay budget is the one that binds."""
    return make_procedure(
        metric_type="conversion",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        conversion_inference="finite_sample",
    )


def _failure(call):
    with pytest.raises(InvalidRequestError) as raised:
        call()
    return raised.value


@pytest.mark.slow
class TestUnresolvedPublicAnswersAreDeterministic:
    BASELINE = Baseline.from_proportion(0.2)

    @pytest.fixture(autouse=True)
    def _budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", 700)

    def test_wide_diagnostics_remain_available_but_scalar_and_curve_refuse(self):
        diagnostic = planned_enclosure(60, 0.8, self.BASELINE, _conversion())
        assert diagnostic.absolute_error > RESOLUTION
        scalar = _failure(lambda: achieved_power(60, 0.8, self.BASELINE, _conversion()))
        curve = _failure(
            lambda: power_curve(
                n_per_arm=60,
                relative_lift=[0.8, 0.8, 0.5],
                baseline=self.BASELINE,
                procedure=_conversion(),
                max_workers=1,
            )
        )
        assert scalar.code == curve.code == "power.binomial_probability_unresolved"
        assert scalar.context == curve.context
        assert scalar.context["lower"] == diagnostic.lower
        assert scalar.context["upper"] == diagnostic.upper
        assert scalar.context["error"] == diagnostic.absolute_error
        assert scalar.context["tolerance"] == RESOLUTION
        for restored in (copy.deepcopy(scalar), pickle.loads(pickle.dumps(scalar))):
            assert restored.code == scalar.code and restored.context == scalar.context
            with pytest.raises(TypeError):
                restored.context["error"] = 0.0

    def test_repeated_mde_requests_refuse_instead_of_skipping_unresolved_effects(self):
        failures = [
            _failure(lambda: minimum_detectable_effect(60, self.BASELINE, _conversion()))
            for _ in range(2)
        ]
        assert failures[0].code == "power.minimum_detectable_effect.numerical_resolution"
        assert failures[0].context == failures[1].context

    def test_size_search_does_not_count_unresolved_mass_as_power(self):
        failure = _failure(lambda: required_sample_size(0.8, self.BASELINE, _conversion()))
        assert failure.code == "power.binomial_probability_unresolved"
        assert failure.context["error"] > failure.context["tolerance"]


@pytest.mark.slow
class TestResolvedPublicAnswers:
    BASELINE = Baseline.from_proportion(0.2)

    def test_a_minimum_detectable_effect_reports_the_power_and_basis_of_its_own_enclosure(self):
        """The effect's power and basis belong to its own admitted point, regardless of cache."""
        curve = power_curve(
            n_per_arm=60,
            target_power=[0.9, 0.8],
            baseline=self.BASELINE,
            procedure=_conversion(),
            max_workers=1,
        )
        alone = minimum_detectable_effect(60, self.BASELINE, _conversion())
        for row in (*curve, alone):
            assert row.mde_relative is not None
            enclosure = planned_enclosure(60, row.mde_relative, self.BASELINE, _conversion())
            assert (row.power, row.power_basis) == (enclosure.power, enclosure.basis)
            assert enclosure.absolute_error <= RESOLUTION
        assert alone.power >= 0.8 and curve[0].power >= 0.9

    def test_unresolved_companion_preserves_a_resolved_supplied_power(self, monkeypatch):
        from increment.power import core

        expected = planned_enclosure(60, 0.8, self.BASELINE, _conversion())
        monkeypatch.setattr(core, "_BINOMIAL_MDE_EVALUATIONS", 2)
        answer = achieved_power(60, 0.8, self.BASELINE, _conversion())
        assert answer.power == expected.power
        assert (answer.mde_relative, answer.mde_unavailable_reason) == (
            None,
            "numerical_resolution",
        )


def test_weight_bucket_changes_cannot_publish_a_false_detectable_band(monkeypatch):
    """Selection jumps even when the complete decision probability is increasing."""
    monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", 5)

    def classify(decision, route, requests):
        return [
            (np.arange(r.j0, r.j1 + 1) >= 1) if r.x_c == 0 else np.ones(r.j1 - r.j0 + 1, bool)
            for r in requests
        ]

    monkeypatch.setattr(_binomial, "classify", classify)
    decision = BinomialDecision(1, 2, 1.0, 0.001, 0.05, "greater")
    geometry = RejectionGeometry(decision, "exact")
    boundary = 1.0 - math.sqrt(10.0 ** (-83 / 32) / 0.01)
    rates = (boundary - 1e-5, boundary - 2.5e-6, boundary + 1e-5)
    reference_rate = boundary - 5e-6
    target = 0.99 * (2 * reference_rate - reference_rate**2) + 0.01 * (1 - reference_rate**2)
    diagnostics = [geometry.evaluate(0.01, rate) for rate in rates]
    assert [value.power >= target for value in diagnostics] == [False, True, False]
    true_power = [0.01 + 0.99 * (2 * rate - rate**2) for rate in rates]
    assert true_power == sorted(true_power)
    for diagnostic, truth in zip(diagnostics, true_power, strict=True):
        assert diagnostic.lower <= truth <= diagnostic.upper
        assert diagnostic.absolute_error > RESOLUTION
        failure = _failure(lambda diagnostic=diagnostic: diagnostic.reported)
        assert failure.code == "power.binomial_probability_unresolved"
    bound = geometry.point_upper(
        0.01, rates[0], rates[-1], geometry.closure_bound(0.01, rates[0], rates[-1])
    )
    assert bound >= max(value.power for value in diagnostics)

    # Public arms have a two-unit floor; use the same budget-dependent selection law
    # on its smallest admissible public lattice rather than changing that floor.
    failure = _failure(
        lambda: achieved_power(
            2,
            rates[1] / 0.01 - 1,
            Baseline.from_proportion(0.01),
            ArmPlanningProcedure.standard(
                "conversion", alternative="greater", conversion_inference="finite_sample"
            ),
            PowerDesign(),
        )
    )
    assert failure.code == "power.binomial_probability_unresolved"
