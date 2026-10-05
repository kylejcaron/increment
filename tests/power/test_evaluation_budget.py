"""An evaluation decides the cells of its own request, whatever the geometry already holds.

The runtime calculations and replays one evaluation may run are bounded (``EVALUATION_ROW_BUDGET``,
``EVALUATION_REPLAY_BUDGET``); a pair the bound leaves out is ambiguous, its mass in the upper end
of the enclosure. A geometry keeps the decisions it made, but they are a cache: the enclosure of a
request is the same on a fresh geometry, on one an earlier evaluation (the same or another) used,
and on a repeat. Every decision below is the runtime's own (`production_decision`, the exact
replay); only the budgets are shrunk so that they bind at a size that runs in seconds.
"""

from __future__ import annotations

from typing import Any

import pytest

from increment.estimation import binomial_rr
from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.estimation.conversion_delta import production_decision
from increment.power import (
    Baseline,
    _binomial,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
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


def _geometry() -> RejectionGeometry:
    decision = BinomialDecision(N, N, 1.0, binomial_rr.nuisance_beta(2.0 * TAIL), TAIL, "two-sided")
    return RejectionGeometry(decision, "exact", PLANNING_CELL_CEILING, Routing(FLOOR, 0.0))


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


def _answer(result: Any) -> tuple[object, ...]:
    return (result.power, result.power_basis, result.mde_relative, result.mde_unavailable_reason)


@pytest.mark.slow
class TestPublicAnswersDoNotDependOnWhatTheGeometryHolds:
    """At 60 units per arm and a 20% rate a lattice holds about 1,800 directional pairs per
    effect; a budget of 700 binds on every effect worth planning and leaves a few thousandths of
    probability ambiguous, so ``power_basis`` is ``approximate``."""

    BASELINE = Baseline.from_proportion(0.2)

    @pytest.fixture(autouse=True)
    def _budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", 700)

    def test_duplicate_curve_points_are_one_answer_and_the_scalar_call(self):
        lifts = [0.8, 0.8, 0.5, 0.8]
        curve = power_curve(
            n_per_arm=60,
            relative_lift=lifts,
            baseline=self.BASELINE,
            procedure=_conversion(),
            max_workers=1,
        )
        assert curve[0].power_basis == "approximate"
        assert _answer(curve[0]) == _answer(curve[1]) == _answer(curve[3])
        direct = {lift: achieved_power(60, lift, self.BASELINE, _conversion()) for lift in lifts}
        for point, lift in zip(curve, lifts, strict=True):
            assert _answer(point) == _answer(direct[lift])

    def test_a_minimum_detectable_effect_reports_the_power_and_basis_of_its_own_enclosure(self):
        """The effect's power and its basis are one figure: the enclosure a fresh plan of that
        very effect gives, whatever an earlier search of the same geometry cached."""
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
            assert (row.power, row.power_basis) == (min(enclosure.power, 1.0), enclosure.basis)
            assert row.power_basis == "approximate"
        assert alone.power >= 0.8 and curve[0].power >= 0.9
