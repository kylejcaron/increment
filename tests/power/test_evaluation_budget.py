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
import json
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
    PowerCurvePoint,
    PowerDesign,
    PowerResult,
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


def test_the_heuristic_replay_decides_the_cells_the_runtime_replay_does(small_budgets):
    """The sizing proposal's Normal-tail replay pays the same replay budget as the runtime's
    own and chooses the same heaviest cells, so its computed mass over them tracks the lower end
    the final search decides from, and its cost is bounded alike."""
    exact = _geometry()
    heuristic = RejectionGeometry(
        _decision(N), "approximate", PLANNING_CELL_CEILING, Routing(FLOOR, 0.0)
    )
    wc, wt = _binomial._window(N, P_C), _binomial._window(N, 0.4)
    chosen = heuristic._selected(wc, wt)
    assert np.array_equal(chosen, exact._selected(wc, wt))
    row_in, col_in = exact._routed_flags(wc.lo, wc.hi, wt.lo, wt.hi)
    replayed = chosen & ~(row_in[:, None] & col_in[None, :])
    assert 0 < int(replayed.sum()) * len(_decision(N).kinds) <= REPLAY_BUDGET
    assert int(replayed.sum()) < int((~(row_in[:, None] & col_in[None, :])).sum())
    proposal = heuristic.evaluate(P_C, 0.4)
    assert not proposal.certified and proposal.ambiguous > RESOLUTION


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
        assert limited.basis == "lower_bound"
        assert limited.published == limited.lower

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
        assert limited.basis == "lower_bound"
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
    """Under a replay budget that leaves material mass undecided, every public answer states
    the enclosure: ``power`` is its lower end, ``power_lower``/``power_upper`` its ends and
    ``power_basis`` says so. Only an explicit point request refuses."""

    BASELINE = Baseline.from_proportion(0.2)

    @pytest.fixture(autouse=True)
    def _budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", 700)

    def test_wide_diagnostics_are_published_as_bounds_by_scalar_and_curve(self):
        diagnostic = planned_enclosure(60, 0.8, self.BASELINE, _conversion())
        assert diagnostic.absolute_error > RESOLUTION
        scalar = achieved_power(60, 0.8, self.BASELINE, _conversion())
        assert scalar.power_basis == "lower_bound"
        assert scalar.numerical_qualification == (
            "scipy_special_function_error_model_conditional_v1"
        )
        assert scalar.power == scalar.power_lower == diagnostic.lower
        assert scalar.power_upper == diagnostic.upper
        curve = power_curve(
            n_per_arm=60,
            relative_lift=[0.8, 0.8, 0.5],
            baseline=self.BASELINE,
            procedure=_conversion(),
            max_workers=1,
        )
        for row in curve[:2]:
            assert (row.power, row.power_lower, row.power_upper, row.power_basis) == (
                scalar.power,
                scalar.power_lower,
                scalar.power_upper,
                scalar.power_basis,
            )
        assert curve[2].power_lower is not None and curve[2].power_upper is not None
        assert curve[2].power_lower <= curve[2].power <= curve[2].power_upper
        assert curve[2].power_upper - curve[2].power_lower > RESOLUTION
        # The explicit point request is the one place the unresolved integral refuses.
        point = _failure(lambda: diagnostic.reported)
        assert point.code == "power.binomial_probability_unresolved"
        assert point.context["lower"] == diagnostic.lower
        assert point.context["upper"] == diagnostic.upper
        assert point.context["error"] == diagnostic.absolute_error
        assert point.context["tolerance"] == RESOLUTION
        for restored in (copy.deepcopy(point), pickle.loads(pickle.dumps(point))):
            assert restored.code == point.code and restored.context == point.context
            with pytest.raises(TypeError):
                restored.context["error"] = 0.0

    def test_repeated_mde_requests_certify_the_same_effect_by_its_lower_bound(self):
        answers = [minimum_detectable_effect(60, self.BASELINE, _conversion()) for _ in range(2)]
        assert answers[0] == answers[1]
        effect = answers[0]
        assert effect.mde_relative is not None
        assert effect.power_basis == "lower_bound"
        assert effect.power_lower is not None and effect.power_lower >= PowerDesign().power
        assert effect.power == effect.power_lower
        at_effect = achieved_power(60, effect.mde_relative, self.BASELINE, _conversion())
        assert (at_effect.power, at_effect.power_lower, at_effect.power_upper) == (
            effect.power,
            effect.power_lower,
            effect.power_upper,
        )

    def test_size_search_certifies_the_target_by_the_enclosures_lower_end(self):
        """The returned size's lower bound reaches the target; its predecessor's does not,
        whether that predecessor fails outright or leaves the target inside its enclosure.
        Under this budget the lower end peaks near 0.75 (the undecided mass grows with the
        window), so the target is one it can certify."""
        design = PowerDesign(power=0.5)
        sized = required_sample_size(0.8, self.BASELINE, _conversion(), design)
        assert sized.power_lower is not None and sized.power_upper is not None
        assert sized.power_lower >= design.power
        assert sized.power_lower <= sized.power <= sized.power_upper
        assert sized.power_basis == "lower_bound"
        assert sized.power == sized.power_lower
        below = achieved_power(sized.n_per_arm - 1, 0.8, self.BASELINE, _conversion(), design)
        assert below.power_lower is not None and below.power_lower < design.power

    def test_a_target_no_budgeted_enclosure_certifies_climbs_to_the_replay_ceiling(self):
        """No size's lower end reaches 0.8 under this budget: the search climbs to the size
        whose windows meet the cell bound and refuses there with the replay-bound code, naming
        the lower end it reached, not a decision the runtime refuses."""
        failure = _failure(lambda: required_sample_size(0.8, self.BASELINE, _conversion()))
        assert failure.code == "power.binomial_replay_bound_exceeded"
        assert failure.context["power"] == PowerDesign().power
        reached = failure.context["power_reached"]
        assert 0.0 <= reached < PowerDesign().power
        at_ceiling = planned_enclosure(failure.context["n_t"], 0.8, self.BASELINE, _conversion())
        assert reached == at_ceiling.lower

    @pytest.mark.parametrize("straddles", [True, False])
    def test_a_bracket_that_never_certifies_hits_the_size_ceiling_refusal(
        self, monkeypatch, straddles
    ):
        """With the arm ceiling at 60 units, a target the ceiling's enclosure straddles (its
        upper end, above its lower end) and one it fails outright both climb to the ceiling and
        refuse with the size-limit code; the power named is the ceiling's certified lower end."""
        from increment.power import core

        at_ceiling = planned_enclosure(60, 0.8, self.BASELINE, _conversion())
        assert at_ceiling.lower < at_ceiling.upper
        target = at_ceiling.upper if straddles else 0.9999
        monkeypatch.setattr(core, "_binomial_arm_ceiling", lambda design, floor, baseline: 60)
        failure = _failure(
            lambda: required_sample_size(
                0.8, self.BASELINE, _conversion(), PowerDesign(power=target)
            )
        )
        assert failure.code == "power.binomial_size_search_unreachable"
        assert failure.context["n_per_arm"] == 60
        assert failure.context["maximum_power"] == at_ceiling.lower < target

    def test_results_with_bounds_round_trip(self):
        result = achieved_power(60, 0.8, self.BASELINE, _conversion())
        assert result.power_basis == "lower_bound"
        dumped = result.model_dump(mode="json")
        assert dumped["power_lower"] == result.power_lower
        assert dumped["power_upper"] == result.power_upper
        assert PowerResult.model_validate(json.loads(json.dumps(dumped))) == result
        assert PowerResult.model_validate_json(result.model_dump_json()) == result
        curve = power_curve(
            n_per_arm=60,
            relative_lift=[0.8],
            baseline=self.BASELINE,
            procedure=_conversion(),
            max_workers=1,
        )
        row = curve.to_dicts()[0]
        assert (row["power_lower"], row["power_upper"]) == (result.power_lower, result.power_upper)
        assert PowerCurvePoint.model_validate_json(curve[0].model_dump_json()) == curve[0]


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


def test_weight_bucket_changes_publish_a_bound_not_a_point(monkeypatch):
    """Selection jumps even when the complete decision probability is increasing: the computed
    mass is not published as a point, and the bounds published hold the true probability."""
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
    published = achieved_power(
        2,
        rates[1] / 0.01 - 1,
        Baseline.from_proportion(0.01),
        ArmPlanningProcedure.standard(
            "conversion", alternative="greater", conversion_inference="finite_sample"
        ),
        PowerDesign(),
    )
    assert published.power_basis == "lower_bound"
    assert published.power_lower is not None and published.power_upper is not None
    assert published.power == published.power_lower < published.power_upper
    assert published.power_upper - published.power_lower > RESOLUTION
