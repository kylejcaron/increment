"""Planning for the runtime's finite-sample binomial risk-ratio decision.

A binomial-eligible conversion or retention plan on ``conversion_inference="finite_sample"``
reports the probability that the unchanged runtime decision (``binomial_rr.p_plus``/
``p_minus`` below the compiled tail allocation) rejects, integrated over the binomial count
law at the analyzed integer counts. Every plan here pins ``finite_sample``: the default
``auto`` plans the delta-method route where the runtime takes it
(``test_conversion_route_planning.py``).
"""

from __future__ import annotations

import json
import math
import tracemalloc
import weakref
from decimal import Decimal
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import binom

from calibration.binomial_oracle import Binomial, precise
from increment.errors import InvalidRequestError
from increment.estimation import binomial_rr
from increment.estimation.arm_contract import ArmPlanningProcedure, RelativeDecisionPolicy
from increment.power import (
    Baseline,
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
    BinomialDecision,
    RejectionGeometry,
    ReplayBoundExceeded,
    window_cells,
)
from increment.power.core import planned_enclosure
from tests.power._procedures import make_procedure


def _runtime_power(
    n_c: int, n_t: int, p_c: float, p_t: float, procedure: ArmPlanningProcedure
) -> float:
    """Independent enumeration of the unchanged runtime decision over every
    count pair."""
    decision = procedure.decision
    assert isinstance(decision, RelativeDecisionPolicy)
    r0 = 1.0 + decision.null_lift
    beta = binomial_rr.nuisance_beta(procedure.compiled_alpha)
    tail = procedure.compiled_tail_alpha
    total = 0.0
    w_t = binom.pmf(np.arange(n_t + 1), n_t, p_t)
    for x_c, w_c in enumerate(binom.pmf(np.arange(n_c + 1), n_c, p_c)):
        for x_t in range(n_t + 1):
            plus = binomial_rr.p_plus(r0, x_c, n_c, x_t, n_t, beta, tail=tail)
            minus = binomial_rr.p_minus(r0, x_c, n_c, x_t, n_t, beta, tail=tail)
            if decision.alternative == "greater":
                rejects = plus < tail
            elif decision.alternative == "less":
                rejects = minus < tail
            else:
                rejects = min(plus, minus) < tail
            total += w_c * w_t[x_t] * rejects
    return total


def _conversion(**overrides: Any) -> ArmPlanningProcedure:
    """A randomized conversion plan pinned to ``finite_sample``; CUPED has no such decision."""
    overrides.setdefault(
        "conversion_inference",
        "auto" if overrides.get("decision_method") == "cuped" else "finite_sample",
    )
    return make_procedure(
        metric_type="conversion",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        **overrides,
    )


def test_overflowed_complier_null_does_not_imply_detectable_unrepresented_effect():
    baseline = Baseline(mean=1e-307, var=1e-307, compliance=0.00555)
    procedure = _conversion(alternative="less", null_lift=1e306)
    with pytest.raises(InvalidRequestError) as raised:
        minimum_detectable_effect(10, baseline, procedure)
    assert raised.value.code == "power.minimum_detectable_effect.unattainable"
    maximum = raised.value.context["maximum_power"]
    assert isinstance(maximum, float) and maximum < 0.8


class TestExactRouteMatchesRuntime:
    """(a) The exact route integrates the runtime's own decisions."""

    @pytest.mark.parametrize(
        ("n_t", "allocation", "p_c", "lift", "overrides"),
        [
            (15, 0.6, 0.3, 0.8, {}),
            (12, 0.5, 0.25, 0.9, {"alternative": "greater", "null_lift": 0.2, "alpha": 0.01}),
            (10, 0.4, 0.5, -0.6, {"alternative": "less", "null_lift": -0.2}),
        ],
    )
    def test_power_equals_enumerated_runtime_rejections(
        self, n_t, allocation, p_c, lift, overrides
    ):
        procedure = _conversion(**overrides)
        design = PowerDesign(allocation=allocation)
        result = achieved_power(n_t, lift, Baseline.from_proportion(p_c), procedure, design)
        n_c = result.n_total - result.n_per_arm
        expected = _runtime_power(n_c, n_t, p_c, p_c * (1.0 + lift), procedure)
        assert result.power_basis == "exact"
        assert result.power == pytest.approx(expected, abs=1e-11)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("n_c", "n_t", "null_ratio", "alpha", "tail", "alternative"),
        [
            (20, 30, 1.0, 0.05, 0.025, "two-sided"),
            (40, 60, 1.2, 0.01, 0.01, "greater"),
            (60, 40, 0.8, 0.05, 0.05, "less"),
        ],
    )
    def test_every_decision_equals_the_runtime(
        self, n_c, n_t, null_ratio, alpha, tail, alternative
    ):
        beta = binomial_rr.nuisance_beta(alpha)
        decision = BinomialDecision(n_c, n_t, null_ratio, beta, tail, alternative)
        plus_cells, minus_cells = RejectionGeometry(decision, "exact").cells(0, n_c, 0, n_t)
        for x_c in range(n_c + 1):
            for x_t in range(n_t + 1):
                plus = binomial_rr.p_plus(null_ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                minus = binomial_rr.p_minus(null_ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                assert plus_cells[x_c, x_t] == ("plus" in decision.kinds and plus)
                assert minus_cells[x_c, x_t] == ("minus" in decision.kinds and minus)

    @pytest.mark.parametrize(
        ("n_c", "n_t", "null_ratio", "tail", "alternative", "controls"),
        [
            (300, 300, 1.0, 0.025, "two-sided", (12, 30, 75)),
            (400, 250, 1.3, 0.05, "greater", (40, 80)),
            (350, 350, 0.7, 0.05, "less", (3, 90)),
        ],
    )
    def test_decisions_with_a_windowed_control_support_equal_the_runtime(
        self, n_c, n_t, null_ratio, tail, alternative, controls
    ):
        """Above the enumeration threshold the runtime sums only a Chernoff window of control
        counts and the replay sums that same window: the counts on each side of every
        rejection boundary, and the ends of each row, are decided as the runtime decides them."""
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, null_ratio, beta, tail, alternative)
        first, last = controls[0], controls[-1]
        cells = RejectionGeometry(decision, "exact").cells(first, last, 0, n_t)
        for kind, mask in zip(("plus", "minus"), cells, strict=True):
            runtime = binomial_rr.p_plus if kind == "plus" else binomial_rr.p_minus
            for x_c in controls:
                row = mask[x_c - first]
                steps = np.flatnonzero(np.diff(row.astype(int)))
                probes = {0, n_t, *steps.tolist(), *(steps + 1).tolist()}
                for x_t in sorted(probes):
                    decided = (
                        kind in decision.kinds
                        and runtime(null_ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                    )
                    assert row[x_t] == decided, (kind, x_c, x_t)


class TestConversionExample:
    """(b) At 701 per arm, 10% baseline and a 50% lift the Wald model
    reported 0.8004; the runtime decision rejects with probability 0.7144."""

    _BASELINE = Baseline.from_proportion(0.1)
    _PROCEDURE = ArmPlanningProcedure.standard("conversion", conversion_inference="finite_sample")

    @pytest.mark.slow
    def test_achieved_power_is_the_runtime_rejection_probability(self):
        result = achieved_power(701, 0.5, self._BASELINE, self._PROCEDURE)
        assert result.power_basis == "exact"
        assert result.power == pytest.approx(0.714423480362035, abs=1e-11)
        # The companion effect reaches the target at the runtime decision.
        assert result.mde_relative is not None
        at_mde = achieved_power(701, result.mde_relative, self._BASELINE, self._PROCEDURE)
        assert at_mde.power >= PowerDesign().power

    @pytest.mark.slow
    def test_required_sample_size_reaches_target_at_its_integer_size(self):
        sized = required_sample_size(0.5, self._BASELINE, self._PROCEDURE)
        assert sized.power_basis == "exact"
        assert sized.n_per_arm == 831
        assert sized.power >= 0.8
        assert sized.power == pytest.approx(0.8001957842538467, abs=1e-11)
        smaller = achieved_power(sized.n_per_arm - 1, 0.5, self._BASELINE, self._PROCEDURE)
        assert smaller.power < 0.8


def test_triggered_sizing_agrees_with_achieved_power_at_its_assigned_size():
    """The size search runs over assigned units, so the returned size and
    its assigned predecessor are judged at the analyzed counts
    ``achieved_power`` uses."""
    baseline = Baseline(mean=0.3, var=0.21, trigger_rate=0.35)
    procedure = ArmPlanningProcedure.standard("conversion", conversion_inference="finite_sample")
    sized = required_sample_size(0.8, baseline, procedure)
    replay = achieved_power(sized.n_per_arm, 0.8, baseline, procedure)
    assert (replay.power, replay.power_basis) == (sized.power, sized.power_basis)
    assert replay.n_triggered_per_arm == sized.n_triggered_per_arm
    assert sized.power_basis == "exact"
    assert sized.power >= PowerDesign().power
    assert achieved_power(sized.n_per_arm - 1, 0.8, baseline, procedure).power < 0.8


def test_sizing_is_refused_only_by_the_selected_route(monkeypatch):
    """At a (constructed) arm ceiling of 150 treatment units the approximate
    replay stays below target while the exact decision reaches it: the
    approximate proposal must not refuse a design the exact route can size."""
    from increment.power import core

    procedure = _conversion(alternative="greater", null_lift=0.2, alpha=0.01)
    baseline = Baseline.from_proportion(0.1)
    exact = achieved_power(150, 2.5, baseline, procedure, PowerDesign(allocation=0.6667))
    n_c = exact.n_total - exact.n_per_arm
    decision = BinomialDecision(n_c, 150, 1.2, binomial_rr.nuisance_beta(0.01), 0.01, "greater")
    approximate = RejectionGeometry(decision, "approximate").evaluate(0.1, 0.35).power
    target = (approximate + exact.power) / 2.0
    assert exact.power_basis == "exact" and approximate < target < exact.power
    design = PowerDesign(power=target, allocation=0.6667)

    monkeypatch.setattr(core, "_binomial_arm_ceiling", lambda design, floor, baseline: 150)
    sized = required_sample_size(2.5, baseline, procedure, design)
    assert sized.power_basis == "exact"
    assert sized.n_per_arm <= 150
    assert sized.power >= target
    assert achieved_power(sized.n_per_arm, 2.5, baseline, procedure, design).power == sized.power
    assert achieved_power(sized.n_per_arm - 1, 2.5, baseline, procedure, design).power < target


def _forbid_replay(monkeypatch) -> None:
    from increment.power import _binomial

    def forbidden(*args, **kwargs):
        raise AssertionError("a decision the runtime refuses in full must not be replayed")

    for name in ("classify", "_window_bounds", "_window"):
        monkeypatch.setattr(_binomial, name, forbidden)


@pytest.mark.parametrize("planner", ["achieved_power", "minimum_detectable_effect"])
def test_arms_above_the_runtime_ceiling_are_refused_not_planned_as_zero_power(monkeypatch, planner):
    """The runtime refuses every count pair above its ceiling, so it decides nothing there: a
    plan is refused, coded, instead of reporting a rejection probability for a decision never
    made. It is refused before any window is built (the windows at these sizes hold ~1e5
    counts each, whose product would not fit in memory)."""
    _forbid_replay(monkeypatch)
    above = binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE + 1
    baseline, procedure = Baseline.from_proportion(0.1), _conversion()
    with pytest.raises(InvalidRequestError) as raised:
        if planner == "achieved_power":
            achieved_power(above, 0.3, baseline, procedure)
        else:
            minimum_detectable_effect(above, baseline, procedure)
    context: dict[str, Any] = dict(raised.value.context)
    assert raised.value.code == "power.binomial_arm_ceiling_exceeded"
    assert (context["n_c"], context["n_t"]) == (above, above)
    assert context["max_arm_size"] == binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE


def test_a_curve_through_a_size_the_runtime_refuses_is_refused_with_the_same_code(monkeypatch):
    _forbid_replay(monkeypatch)
    above = binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE + 1
    with pytest.raises(InvalidRequestError) as raised:
        power_curve(
            n_per_arm=[above, 1_000],
            relative_lift=0.3,
            baseline=Baseline.from_proportion(0.1),
            procedure=_conversion(),
        )
    assert raised.value.code == "power.binomial_arm_ceiling_exceeded"


@pytest.mark.parametrize("planner", ["achieved_power", "minimum_detectable_effect"])
def test_a_tail_level_the_float_margin_dominates_is_refused_where_the_runtime_refuses_it(
    monkeypatch, planner
):
    """Once the margin every certified tail carries reaches what the tail level leaves after the
    nuisance budget, the runtime refuses every count pair (a ``DecisionFailure``): planning
    refuses that size, before any window or replay is built, naming the margin and the sizes it
    leaves room at. The same alpha is planned normally on an arm whose margin leaves room."""
    n, alpha = 100_000_000, 5e-8
    with pytest.raises(binomial_rr.BinomialDataError) as refused:
        binomial_rr.confidence_interval(100, n, 300, n, alpha=alpha, alternative="two-sided")
    assert refused.value.code == "estimation.binomial.tail_unrepresentable"

    procedure, baseline = _conversion(alpha=alpha), Baseline.from_proportion(1e-6)
    _forbid_replay(monkeypatch)
    with pytest.raises(InvalidRequestError) as raised:
        if planner == "achieved_power":
            achieved_power(n, 3.0, baseline, procedure)
        else:
            minimum_detectable_effect(n, baseline, procedure)
    context: dict[str, Any] = dict(raised.value.context)
    assert raised.value.code == "power.binomial_tail_level_unrepresentable"
    assert (context["cause"], context["scope"]) == ("float_margin", "requested")
    assert (context["n_c"], context["n_t"], context["alpha"]) == (n, n, alpha)
    assert context["margin"] >= context["tail_alpha"] - context["beta"]
    monkeypatch.undo()

    small = 100_000
    binomial_rr.confidence_interval(100, small, 400, small, alpha=alpha, alternative="two-sided")
    admitted = achieved_power(small, 3.0, Baseline.from_proportion(1e-3), procedure)
    assert admitted.power > 0.999


class TestATailLevelTheSolverRefuses:
    """A nuisance budget (``alpha / 32``) below the Clopper-Pearson solver's floor is refused for
    every count, so the runtime decides nothing at any arm size: a size search, a supplied-size
    power and an effect search all refuse it with one code, without a replay."""

    @pytest.mark.parametrize("alpha", [1e-12, 1e-10, 3e-8])
    def test_every_planner_names_the_tail_level_without_a_replay(self, monkeypatch, alpha):
        with pytest.raises(binomial_rr.BinomialDataError) as refused:
            binomial_rr.confidence_interval(
                5_000, 100_000, 7_500, 100_000, alpha=alpha, alternative="two-sided"
            )
        assert refused.value.code == "estimation.binomial.tail_unrepresentable"
        baseline, procedure = Baseline.from_proportion(0.05), _conversion(alpha=alpha)
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(0.5, baseline, procedure)
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == "power.binomial_tail_level_unrepresentable"
        assert context["alpha"] == alpha
        assert context["beta"] < context["solver_floor"] == binomial_rr._CP_BETA_FLOOR

        _forbid_replay(monkeypatch)
        with pytest.raises(InvalidRequestError) as requested:
            achieved_power(100_000, 0.5, baseline, procedure)
        assert requested.value.code == "power.binomial_tail_level_unrepresentable"
        assert dict(requested.value.context)["cause"] == "solver_floor"
        with pytest.raises(InvalidRequestError) as effect:
            minimum_detectable_effect(100_000, baseline, procedure)
        assert effect.value.code == "power.binomial_tail_level_unrepresentable"

    @pytest.mark.slow
    def test_the_first_alpha_the_solver_admits_is_sized_and_decided(self):
        alpha = 4e-8
        binomial_rr.confidence_interval(
            5_000, 100_000, 7_500, 100_000, alpha=alpha, alternative="two-sided"
        )
        sized = required_sample_size(0.5, Baseline.from_proportion(0.05), _conversion(alpha=alpha))
        assert sized.power >= PowerDesign().power

    def test_an_extreme_allocation_is_refused_by_arm_size_not_by_an_assertion(self):
        """At an allocation of 1e-10 the smallest treatment arm of two units pairs with a control
        arm of 2e10, above the runtime's billion-unit ceiling at every size: no size can be
        planned, which is not a statement about alpha."""
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(
                0.5,
                Baseline.from_proportion(0.05),
                _conversion(),
                PowerDesign(allocation=1e-10),
            )
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == "power.binomial_arm_ceiling_below_smallest_design"
        assert context["n_t"] == 2
        assert context["n_c"] > context["max_arm_size"] == binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE
        assert context["allocation"] == 1e-10

    def test_a_float_margin_that_dominates_the_smallest_design_is_a_tail_level_refusal(self):
        """A control arm just under the ceiling at the smallest treatment arm carries a float
        margin of 2e-7, above what a two-sided alpha of 4e-7 leaves after its nuisance budget
        (1.9e-7), though the solver admits the alpha (it is above 3.2e-8): the runtime refuses
        every count pair at these arms, so no larger size can have power."""
        alpha, allocation = 4e-7, 2.2e-9
        design = PowerDesign(allocation=allocation)
        with pytest.raises(binomial_rr.BinomialDataError) as runtime:
            binomial_rr.confidence_interval(
                5, 900_000_000, 1, 2, alpha=alpha, alternative="two-sided"
            )
        assert runtime.value.code == "estimation.binomial.tail_unrepresentable"
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(
                0.5, Baseline.from_proportion(0.05), _conversion(alpha=alpha), design
            )
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == "power.binomial_tail_level_unrepresentable"
        assert context["cause"] == "float_margin"
        assert context["beta"] >= context["solver_floor"]
        assert context["margin"] >= context["tail_alpha"] - context["beta"]


class TestAShiftedNullTheControlArmAloneRejects:
    """Under a float margin that dominates the tail level the runtime evaluates no tail, yet a
    two-sided or "less" test of a null ratio above ``1/a`` (``a`` the control arm's Clopper-Pearson
    lower bound) rejects on the empty nuisance domain alone, whatever the treatment count: those
    count pairs are decided and every other is refused. Planning decides exactly the same pairs,
    plans a control window made of them alone, and refuses a window holding a refused count rather
    than integrate it as a non-rejection."""

    N, ALPHA, NULL_RATIO = binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE, 1e-7, 2.0

    def _key(self, alternative: Literal["two-sided", "greater", "less"]) -> BinomialDecision:
        tail = self.ALPHA / 2.0 if alternative == "two-sided" else self.ALPHA
        beta = binomial_rr.nuisance_beta(self.ALPHA)
        return BinomialDecision(self.N, self.N, self.NULL_RATIO, beta, tail, alternative)

    @pytest.mark.parametrize("alternative", ["less", "two-sided"])
    @pytest.mark.parametrize("route", ["exact", "approximate"])
    def test_a_window_of_structural_counts_has_the_runtime_power_without_a_replay(
        self, monkeypatch, alternative, route
    ):
        from increment.power import _binomial

        key = self._key(alternative)
        assert _binomial.margin_dominates(key) and not _binomial.refused(key)
        floor = _binomial.structural_floor(key)
        assert floor is not None
        # The runtime decides from the floor on and refuses the count below it.
        with pytest.raises(binomial_rr.BinomialDataError):
            binomial_rr.confidence_interval(
                floor - 1, self.N, 5, self.N, alpha=self.ALPHA, alternative=alternative, null_r=2.0
            )
        at_floor = binomial_rr.confidence_interval(
            floor, self.N, 5, self.N, alpha=self.ALPHA, alternative=alternative, null_r=2.0
        )
        assert at_floor.p_value_null < self.ALPHA
        assert at_floor.upper is not None and at_floor.upper < self.NULL_RATIO

        p_c, p_t = 1.0 - 1e-7, 1e-7
        assert _binomial.window_decided(key, p_c)
        wc, wt = _binomial._window(self.N, p_c), _binomial._window(self.N, p_t)
        assert wc.lo >= floor
        for x_c in (wc.lo, (wc.lo + wc.hi) // 2, wc.hi):
            for x_t in (wt.lo, wt.hi):
                ci = binomial_rr.confidence_interval(
                    x_c, self.N, x_t, self.N, alpha=self.ALPHA, alternative=alternative, null_r=2.0
                )
                assert ci.p_value_null < self.ALPHA
        # Every pair of the windows rejects, so the power is their mass: no tail is replayed.
        monkeypatch.setattr(_binomial, "_classify_live", self._forbidden)
        power = _binomial.RejectionGeometry(key, route).evaluate(p_c, p_t)
        mass = float(wc.weights.sum()) * float(wt.weights.sum())
        assert power.power == pytest.approx(mass, rel=1e-12)
        assert power.lower <= mass <= power.upper <= 1.0

    @staticmethod
    def _forbidden(*args, **kwargs):
        raise AssertionError("a count pair the margin dominates must not be replayed")

    def test_a_window_holding_a_refused_count_is_refused_not_integrated(self, monkeypatch):
        from increment.power import _binomial

        key = self._key("less")
        floor = _binomial.structural_floor(key)
        assert floor is not None and not _binomial.window_decided(key, 0.5)
        # Every pair of the window is refused: the finite route is unavailable for all its mass.
        with pytest.raises(_binomial.FiniteRouteUnavailable) as unavailable:
            _binomial.RejectionGeometry(key, "exact").evaluate(0.5, 1.0)
        assert unavailable.value.mass > 0.999

        procedure = _conversion(alpha=self.ALPHA, alternative="less", null_lift=1.0)
        baseline = Baseline.from_proportion(0.5)
        for name in ("classify", "_window"):
            monkeypatch.setattr(_binomial, name, self._forbidden)
        for planner in (
            lambda: achieved_power(self.N, -0.5, baseline, procedure),
            lambda: minimum_detectable_effect(self.N, baseline, procedure),
        ):
            with pytest.raises(InvalidRequestError) as raised:
                planner()
            context: dict[str, Any] = dict(raised.value.context)
            assert raised.value.code == "power.binomial_tail_level_unrepresentable"
            assert (context["cause"], context["scope"]) == ("float_margin", "requested")
            assert (context["decided_from"], context["p_c"]) == (floor, 0.5)
        # The window's first control count is one the runtime refuses.
        lo, _ = _binomial._window_bounds(self.N, 0.5)
        assert lo < floor
        with pytest.raises(binomial_rr.BinomialDataError):
            binomial_rr.confidence_interval(
                lo, self.N, 5, self.N, alpha=self.ALPHA, alternative="less", null_r=2.0
            )

    def test_a_size_search_stays_where_the_decision_reads_the_treatment_arm(self):
        """At the smallest design of a 2.2e-9 allocation the margin dominates a two-sided alpha
        of 4e-7; with a null ratio of two the control arm alone decides from a count the context
        names, which no treatment effect moves, so no size is sized for one."""
        from increment.power import _binomial, core

        alpha, design = 4e-7, PowerDesign(allocation=2.2e-9)
        procedure = _conversion(alpha=alpha, alternative="two-sided", null_lift=1.0)
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(0.5, Baseline.from_proportion(0.05), procedure, design)
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == "power.binomial_tail_level_unrepresentable"
        assert (context["cause"], context["scope"]) == ("float_margin", "smallest")
        key = core._binomial_key(procedure, context["n_t"], context["n_c"])
        assert _binomial.margin_dominates(key) and not _binomial.refused(key)
        floor = context["decided_from"]
        assert floor == _binomial.structural_floor(key)
        decided = binomial_rr.confidence_interval(
            floor, key.n_c, 1, key.n_t, alpha=alpha, alternative="two-sided", null_r=2.0
        )
        assert decided.p_value_null < alpha
        with pytest.raises(binomial_rr.BinomialDataError):
            binomial_rr.confidence_interval(
                floor - 1, key.n_c, 1, key.n_t, alpha=alpha, alternative="two-sided", null_r=2.0
            )


class TestPlanningReplayBound:
    """Planning replays the runtime decision over every retained (control, treatment) count cell
    at the null rate, and its cost grows with that count. A decision beyond
    ``PLANNING_CELL_CEILING`` cells is refused, coded, before any replay; the bound is the
    replay's cost, so a huge arm with a rare control rate is still planned, and the runtime
    itself decides arms up to ``FINITE_SAMPLE_MAX_ARM_SIZE``."""

    _CODE = "power.binomial_replay_bound_exceeded"

    @staticmethod
    def _forbid_replay(monkeypatch) -> None:
        from increment.power import _binomial

        def forbidden(*args, **kwargs):
            raise AssertionError("a decision beyond the planning bound must not be replayed")

        monkeypatch.setattr(_binomial, "classify", forbidden)

    @pytest.mark.parametrize("planner", ["achieved_power", "minimum_detectable_effect"])
    def test_a_dense_design_above_the_bound_is_refused_before_any_replay(
        self, monkeypatch, planner
    ):
        self._forbid_replay(monkeypatch)
        n = 5_000_000
        baseline, procedure = Baseline.from_proportion(0.05), _conversion()
        with pytest.raises(InvalidRequestError) as raised:
            if planner == "achieved_power":
                achieved_power(n, 0.02, baseline, procedure)
            else:
                minimum_detectable_effect(n, baseline, procedure)
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == self._CODE
        assert (context["n_c"], context["n_t"]) == (n, n)
        assert context["p_c"] == 0.05
        assert context["max_cells"] == PLANNING_CELL_CEILING < context["cells"]
        assert context["max_arm_size"] == binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE

    def test_the_bound_is_the_cells_the_replay_spans_not_the_arm_size(self):
        """A hundred million units per arm at a rate of 2e-7 expect 20 events per arm: the
        replay spans a few thousand cells, far inside the bound."""
        result = achieved_power(100_000_000, 2.0, Baseline.from_proportion(2e-7), _conversion())
        assert result.power_basis == "exact"
        assert 0.0 < result.power < 1.0
        with pytest.raises(InvalidRequestError) as raised:
            achieved_power(100_000_000, 0.02, Baseline.from_proportion(0.05), _conversion())
        assert raised.value.code == self._CODE

    def test_a_size_search_that_meets_the_bound_refuses_at_the_largest_plannable_size(
        self, monkeypatch
    ):
        """A (constructed) bound of 12,000 cells holds ~1,250 units per arm at a 5% rate, far
        below the ~120,000 the lift needs. The search stops at the largest size the bound admits,
        which ``achieved_power`` plans, and refuses with the power reached at that ceiling (kept 1/128
        under the largest size, so skipped sizes may reach more)."""
        from increment.power import core

        monkeypatch.setattr(core, "PLANNING_CELL_CEILING", 12_000)
        baseline, procedure = Baseline.from_proportion(0.05), _conversion()
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(0.05, baseline, procedure)
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == self._CODE
        assert context["power"] == PowerDesign().power
        assert context["cells"] <= context["max_cells"] == 12_000
        at_ceiling = achieved_power(context["n_t"], 0.05, baseline, procedure)
        assert at_ceiling.power == context["power_reached"] < context["power"]
        with pytest.raises(InvalidRequestError) as above:
            achieved_power(math.ceil(context["n_t"] * 1.05), 0.05, baseline, procedure)
        assert above.value.code == self._CODE

    def test_a_tail_level_the_runtime_refuses_at_huge_arms_does_not_lift_the_bound(
        self, monkeypatch
    ):
        """At an alpha of 1e-7 the float margin dominates the tail level from about a hundred
        million units per arm, where the replay is skipped (zero cells). The sizes between the
        bound and there are still beyond it: the search stops at the largest size the bound
        admits and refuses with the power reached, instead of proposing them."""
        from increment.power import core

        monkeypatch.setattr(core, "PLANNING_CELL_CEILING", 12_000)
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(0.05, Baseline.from_proportion(0.05), _conversion(alpha=1e-7))
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == self._CODE
        assert context["cells"] <= context["max_cells"] == 12_000
        assert context["power_reached"] < context["power"]

    def test_the_size_ceiling_stops_where_the_runtime_starts_refusing_the_tail_level(self):
        """At an alpha of 1e-7 the float margin dominates the tail level from about 1.06e8 units
        per arm: the last admitted size is not refused and its successor is."""
        from increment.power import _binomial, core

        procedure, baseline = _conversion(alpha=1e-7), Baseline.from_proportion(1e-4)
        floor = core._assigned_minimum_per_arm(procedure, baseline)
        last = core._binomial_admitted_ceiling(
            procedure, baseline, PowerDesign(), floor, binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE
        )
        assert 10**8 < last < 2 * 10**8
        assert not _binomial.refused(core._binomial_key(procedure, last, last))
        assert _binomial.refused(core._binomial_key(procedure, last + 1, last + 1))

    def test_an_alternative_window_beyond_the_bound_is_refused_before_it_is_allocated(
        self, monkeypatch
    ):
        """A null treatment rate of one leaves the null rectangle the control window alone (about
        ten thousand cells at 2M units), but the alternative at a 50% treatment rate pairs it with
        a window as wide: a hundred million cells, which the three masks would hold at
        ~100 MiB each. It is refused before any is built or replayed."""
        from increment.power import core

        self._forbid_replay(monkeypatch)
        n = 2_000_000
        baseline = Baseline.from_proportion(0.5)
        procedure = _conversion(null_lift=1.0, alternative="less")
        key = core._binomial_key(procedure, n, n)
        alternative = window_cells(key, 0.5, 0.5)
        assert window_cells(key, 0.5) < PLANNING_CELL_CEILING < alternative
        tracemalloc.start()
        try:
            with pytest.raises(InvalidRequestError) as raised:
                achieved_power(n, 0.0, baseline, procedure)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == self._CODE
        assert (context["n_c"], context["n_t"], context["p_c"], context["p_t"]) == (n, n, 0.5, 0.5)
        assert context["cells"] == alternative
        assert context["max_cells"] == PLANNING_CELL_CEILING
        assert peak < 64 * 2**20

    def test_an_effect_search_that_would_store_more_than_the_bound_ends_unresolved(
        self, monkeypatch
    ):
        """At 700 units per arm and a 5% rate the null rectangle is 6,561 cells, the supplied
        effect's 8,019, and the minimum detectable effect's window with the far end the search
        also evaluates 11,340. Under a bound of 10,000 the supplied effect keeps its power and
        its companion effect is unavailable; asking for the effect itself is refused."""
        from increment.power import core

        baseline, procedure = Baseline.from_proportion(0.05), _conversion()
        n, lift = 700, 0.5
        unbounded = achieved_power(n, lift, baseline, procedure)
        assert unbounded.mde_relative is not None
        monkeypatch.setattr(core, "PLANNING_CELL_CEILING", 10_000)
        key = core._binomial_key(procedure, n, n)
        assert window_cells(key, 0.05) < 10_000 and window_cells(key, 0.05, 0.075) < 10_000

        bounded = achieved_power(n, lift, baseline, procedure)
        assert bounded.power == unbounded.power
        assert (bounded.mde_relative, bounded.mde_unavailable_reason) == (
            None,
            "numerical_resolution",
        )
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(n, baseline, procedure)
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == self._CODE
        assert context["max_cells"] == 10_000 < context["cells"]
        assert 0.05 < context["p_t"] < 1.0

    def test_a_size_search_stops_where_the_alternative_window_leaves_the_bound(self, monkeypatch):
        """Detecting a 100% lift on a 5% rate needs 480 units per arm: a null rectangle of
        4,356 cells and an alternative one of 6,138. Under a bound of 5,000 the null rectangle
        admits that size and the alternative does not, so the search stops at the largest size
        whose alternative is plannable and refuses with the power it reached there."""
        from increment.power import core

        baseline, procedure = Baseline.from_proportion(0.05), _conversion(alternative="greater")
        assert required_sample_size(1.0, baseline, procedure).n_per_arm == 480
        monkeypatch.setattr(core, "PLANNING_CELL_CEILING", 5_000)
        with pytest.raises(InvalidRequestError) as raised:
            required_sample_size(1.0, baseline, procedure)
        context: dict[str, Any] = dict(raised.value.context)
        assert raised.value.code == self._CODE
        assert context["p_t"] == pytest.approx(0.1)
        assert context["cells"] <= context["max_cells"] == 5_000
        assert context["power_reached"] < context["power"]
        at_ceiling = achieved_power(context["n_t"], 1.0, baseline, procedure)
        assert at_ceiling.power == context["power_reached"]
        with pytest.raises(InvalidRequestError) as above:
            achieved_power(math.ceil(context["n_t"] * 1.05), 1.0, baseline, procedure)
        assert above.value.code == self._CODE

    def test_a_geometry_never_stores_more_than_its_bound_across_evaluations(self):
        """Each alternative's rectangle fits the bound alone; the geometry holds every row by
        both treatment windows, which does not. The second evaluation is refused with the cells
        it would store and leaves the first intact."""
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(300, 300, 1.0, beta, 0.025, "greater")
        near, far = window_cells(decision, 0.05, 0.05), window_cells(decision, 0.05, 0.4)
        bound = max(near, far) + min(near, far) // 2
        geometry = RejectionGeometry(decision, "exact", max_cells=bound)
        first = geometry.evaluate(0.05, 0.05)
        before = [(s.j0, s.j1) for s in geometry.segments]
        with pytest.raises(ReplayBoundExceeded) as raised:
            geometry.evaluate(0.05, 0.4)
        assert raised.value.cells == near + far > bound
        assert raised.value.p_t == 0.4
        assert [(s.j0, s.j1) for s in geometry.segments] == before
        assert geometry.evaluate(0.05, 0.05) == first
        assert RejectionGeometry(decision, "exact").evaluate(0.05, 0.4).power > 0.0

    def test_a_solve_is_refused_for_its_own_union_whatever_an_earlier_solve_cached(self):
        """An earlier solve cached the window at a 5% rate. This solve evaluates 20% (4,802
        cells) and then 60% (5,880 cells, apart from it): each fits a bound of 8,000, and with
        the cache their storage fits until the second, which a cache-only drop would clear to
        succeed. The solve's own union is 10,682 cells, which a fresh geometry refuses too, so
        the shared one refuses it with the same cells."""
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(300, 300, 1.0, beta, 0.025, "greater")
        assert [window_cells(decision, 0.05, p) for p in (0.2, 0.6)] == [4_802, 5_880]
        bound = 8_000

        shared = RejectionGeometry(decision, "exact", max_cells=bound)
        shared.evaluate(0.05, 0.05)
        shared.begin_solve()
        first = shared.evaluate(0.05, 0.2)
        with pytest.raises(ReplayBoundExceeded) as raised:
            shared.evaluate(0.05, 0.6)

        fresh = RejectionGeometry(decision, "exact", max_cells=bound)
        assert fresh.evaluate(0.05, 0.2) == first
        with pytest.raises(ReplayBoundExceeded) as expected:
            fresh.evaluate(0.05, 0.6)
        assert raised.value.cells == expected.value.cells == 49 * (98 + 120) > bound
        assert raised.value.p_t == expected.value.p_t == 0.6
        assert shared.evaluate(0.05, 0.2) == first

    def test_the_cache_of_earlier_solves_is_dropped_and_the_solves_own_cells_are_kept(
        self, monkeypatch
    ):
        """An earlier solve cached 40% (columns 62..181). This solve evaluates 5% (0..48), which
        fits beside it (8,281 cells of a bound of 8,500), and then 60% (119..238), which merges
        with the cache into 11,074. The cache goes, the 5% cells stay (they are not classified
        again), and the answers are a fresh geometry's."""
        from increment.power import _binomial

        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(300, 300, 1.0, beta, 0.025, "greater")
        bound = 8_500

        def stored(geometry: RejectionGeometry) -> int:
            return geometry.rows * sum(s.j1 - s.j0 + 1 for s in geometry.segments)

        shared = RejectionGeometry(decision, "exact", max_cells=bound)
        shared.evaluate(0.05, 0.4)
        shared.begin_solve()
        first = shared.evaluate(0.05, 0.05)
        assert stored(shared) == 49 * (49 + 120)

        calls: list[int] = []
        classify = _binomial.classify
        monkeypatch.setattr(
            _binomial,
            "classify",
            lambda *args, **kwargs: calls.append(1) or classify(*args, **kwargs),
        )
        second = shared.evaluate(0.05, 0.6)
        assert stored(shared) == 49 * (49 + 120) <= bound
        calls.clear()
        assert shared.evaluate(0.05, 0.05) == first
        assert not calls
        fresh = RejectionGeometry(decision, "exact", max_cells=bound)
        assert fresh.evaluate(0.05, 0.05) == first
        assert fresh.evaluate(0.05, 0.6) == second

    def test_a_curve_row_is_not_refused_for_the_cells_earlier_rows_stored(self, monkeypatch):
        """Each lift's alternative rectangle fits a bound its neighbour's union with it does
        not. The curve shares one geometry across its rows, but every row answers as its own
        scalar call does: the cells earlier rows left are a cache, dropped when a row needs room."""
        from increment.power import _binomial, core

        baseline, procedure, n = Baseline.from_proportion(0.05), _conversion(), 700
        lifts = [0.0, 4.0]
        key = core._binomial_key(procedure, n, n)
        rates = [0.05 * (1.0 + lift) for lift in lifts]
        rectangles = [window_cells(key, 0.05, rate) for rate in rates]
        near, far = (_binomial._window_bounds(n, rate) for rate in rates)
        assert near[1] + 1 < far[0], "the windows are disjoint, so the geometry stores both"
        bound = math.ceil(1.3 * max(rectangles))
        assert window_cells(key, 0.05) < max(rectangles) < bound < sum(rectangles)
        monkeypatch.setattr(core, "PLANNING_CELL_CEILING", bound)

        curve = power_curve(
            n_per_arm=n, relative_lift=lifts, baseline=baseline, procedure=procedure, max_workers=1
        )
        for point, lift in zip(curve, lifts, strict=True):
            direct = achieved_power(n, lift, baseline, procedure)
            assert (point.power, point.mde_relative, point.mde_unavailable_reason) == (
                direct.power,
                direct.mde_relative,
                direct.mde_unavailable_reason,
            )
            assert direct.power > 0.0


class TestLargeArmsDecideAsTheRuntime:
    """The replay and the runtime evaluate the same binomial tails through the same guarded SciPy
    primitives, so at the arm sizes the ceiling admits they classify the same count pairs. The
    point-mass cases are the ones that once read Cephes' ``bdtr``/``bdtrc``, whose error at the
    median grows with the arm: 1e-3 absolute at 1e7 trials, 0.11 at 1e8, 0.25 at 1e9."""

    _BETA = binomial_rr.nuisance_beta(0.05)

    @pytest.mark.parametrize("n", [100_000_000, 1_000_000_000])
    def test_the_exact_replay_of_rare_counts_equals_the_runtime(self, n):
        decision = BinomialDecision(n, n, 1.0, self._BETA, 0.025, "greater")
        geometry = RejectionGeometry(decision, "exact")
        controls = (4, 8, 12)
        plus, _ = geometry.cells(controls[0], controls[-1], 0, 200)
        for x_c in controls:
            row = plus[x_c - controls[0]]
            first = int(np.argmax(row))
            assert row.any() and first > x_c
            for x_t in range(first - 3, first + 4):
                runtime = binomial_rr.p_plus(1.0, x_c, n, x_t, n, self._BETA, tail=0.025) < 0.025
                assert row[x_t] == runtime, (n, x_c, x_t)

    @pytest.mark.parametrize(
        ("n", "level"),
        [
            (10_000_000, 0.5007),
            (100_000_000, 0.55),
            pytest.param(1_000_000_000, 0.65, marks=pytest.mark.slow),
        ],
    )
    def test_a_point_mass_treatment_tail_at_the_control_median_equals_the_runtime(self, n, level):
        """With a null risk ratio of 1.25, the treatment rate under the null is one for every
        control rate above ``0.8``, so at the all-success treatment count the tail is the
        control's lower binomial tail at its observed count: at ``n / 1.25`` that is its median.
        The one-sided ``level`` sits between the exact tail there (about one half) and the value
        Cephes' ``bdtr`` returned for it (1.5e-3 above at 1e7 trials, 0.11 at 1e8, 0.34 at
        1e9), so a replay that read it would fail to reject where the runtime rejects."""
        ratio = 1.25
        beta = binomial_rr.nuisance_beta(level)
        decision = BinomialDecision(n, n, ratio, beta, level, "greater")
        x_c = round(n / ratio)
        plus, _ = RejectionGeometry(decision, "approximate").cells(x_c, x_c, n, n)
        assert binomial_rr.p_plus(ratio, x_c, n, n, n, beta, tail=level) < level
        assert bool(plus[0, 0]) is True
        below = BinomialDecision(n, n, ratio, beta, 0.45, "greater")
        plus, _ = RejectionGeometry(below, "approximate").cells(x_c, x_c, n, n)
        assert binomial_rr.p_plus(ratio, x_c, n, n, n, beta, tail=0.45) >= 0.45
        assert bool(plus[0, 0]) is False

    @pytest.mark.slow
    def test_achieved_power_at_a_huge_rare_arm_is_the_runtime_rejection_probability(self):
        """Through the public path: 1e8 units per arm expecting 5 and 12.5 events. The power is
        the sum over every retained count pair of the binomial weights where the unchanged
        runtime decision rejects, the weights taken from the decimal oracle, which shares no code
        with SciPy."""
        n, p_c, lift = 100_000_000, 5e-8, 1.5
        procedure = _conversion(alternative="greater")
        result = achieved_power(n, lift, Baseline.from_proportion(p_c), procedure)
        assert result.power_basis == "exact"
        tail = procedure.compiled_tail_alpha
        beta = binomial_rr.nuisance_beta(procedure.compiled_alpha)
        p_t = p_c * (1.0 + lift)
        with precise():
            w_c, w_t = Binomial(n, p_c).pmf_range(0, 40), Binomial(n, p_t).pmf_range(0, 70)
            expected = sum(
                (
                    w_c[x_c] * w_t[x_t]
                    for x_c in range(41)
                    for x_t in range(71)
                    if binomial_rr.p_plus(1.0, x_c, n, x_t, n, beta, tail=tail) < tail
                ),
                Decimal(0),
            )
            assert sum(w_c, Decimal(0)) > 1 - Decimal("1e-12")
            assert sum(w_t, Decimal(0)) > 1 - Decimal("1e-12")
        assert result.power == pytest.approx(float(expected), abs=1e-9)


def _rational_pmf(n: int, p: float) -> list[Fraction]:
    """``Bin(n, p)`` at the float ``p`` in exact rational arithmetic."""
    rate = Fraction(p)
    return [math.comb(n, k) * rate**k * (1 - rate) ** (n - k) for k in range(n + 1)]


def _exact_rejection_probability(geometry: RejectionGeometry, p_c: float, p_t: float) -> Fraction:
    """The rejection probability of the geometry's decision over every count pair, exactly."""
    decision = geometry.decision
    plus, minus = geometry.cells(0, decision.n_c, 0, decision.n_t)
    w_c, w_t = _rational_pmf(decision.n_c, p_c), _rational_pmf(decision.n_t, p_t)
    rows, cols = np.nonzero(plus | minus)
    return sum((w_c[i] * w_t[j] for i, j in zip(rows, cols, strict=True)), Fraction(0))


_FAR_END = _conversion(alternative="greater")


class TestPlanningNumericalEnclosure:
    """Internal bounds enclose rejection probability; admitted point estimates need not
    equal either endpoint."""

    _CASES = [
        (15, 20, 1.0, 0.05, 0.025, "two-sided", 0.3, 0.55),
        (30, 24, 1.2, 0.01, 0.01, "greater", 0.25, 0.5),
        (24, 30, 0.8, 0.05, 0.05, "less", 0.5, 0.2),
    ]

    @staticmethod
    def _geometry(n_c, n_t, null_ratio, alpha, tail, alternative) -> RejectionGeometry:
        beta = binomial_rr.nuisance_beta(alpha)
        return RejectionGeometry(
            BinomialDecision(n_c, n_t, null_ratio, beta, tail, alternative), "exact"
        )

    @pytest.mark.parametrize(
        ("n_c", "n_t", "null_ratio", "alpha", "tail", "alternative", "p_c", "p_t"), _CASES
    )
    def test_the_enclosure_holds_the_exact_rejection_probability(
        self, n_c, n_t, null_ratio, alpha, tail, alternative, p_c, p_t
    ):
        geometry = self._geometry(n_c, n_t, null_ratio, alpha, tail, alternative)
        result = geometry.evaluate(p_c, p_t)
        exact = _exact_rejection_probability(geometry, p_c, p_t)
        assert Fraction(result.lower) <= exact <= Fraction(result.upper)
        assert result.lower <= result.power <= result.upper

    @pytest.mark.parametrize(
        ("n_c", "n_t", "null_ratio", "alpha", "tail", "alternative", "p_c", "p_t"), _CASES
    )
    def test_the_closure_bound_holds_the_exact_probability_at_every_rate_of_its_interval(
        self, n_c, n_t, null_ratio, alpha, tail, alternative, p_c, p_t
    ):
        geometry = self._geometry(n_c, n_t, null_ratio, alpha, tail, alternative)
        geometry.cells(0, n_c, 0, n_t)
        low, high = sorted((0.9 * p_t, 1.1 * p_t))
        bound = geometry.closure_bound(p_c, low, high)
        for rate in np.linspace(low, high, 9):
            exact = _exact_rejection_probability(geometry, p_c, float(rate))
            assert Fraction(bound) >= exact
            assert Fraction(geometry.closure_bound(p_c, float(rate), float(rate))) >= exact

    def test_the_enclosure_holds_the_decimal_oracle_at_a_billion_units(self):
        """1e9 units per arm at rates expecting 200 and 340 events: the retained mass over the
        evaluated windows and cells, summed from the oracle's pmf, lies inside the enclosure."""
        n, p_c, p_t = 1_000_000_000, 2e-7, 3.4e-7
        decision = BinomialDecision(n, n, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "two-sided")
        geometry = RejectionGeometry(decision, "approximate")
        result = geometry.evaluate(p_c, p_t)
        from increment.power import _binomial

        wc, wt = _binomial._window(n, p_c), _binomial._window(n, p_t)
        plus, minus = geometry.cells(wc.lo, wc.hi, wt.lo, wt.hi)
        with precise():
            a_c = Binomial(n, p_c).pmf_range(wc.lo, wc.hi)
            b_t = Binomial(n, p_t).pmf_range(wt.lo, wt.hi)
            retained = sum(
                (a_c[i] * b_t[j] for i, j in zip(*np.nonzero(plus | minus), strict=True)),
                Decimal(0),
            )
            assert Decimal(result.lower) <= retained <= Decimal(result.upper)

    def test_endpoint_target_uses_point_power_not_its_lower_bound(self):
        """The endpoint power is 11/32; roundoff no longer forces a lower-bound target."""
        baseline = Baseline.from_proportion(0.5)
        endpoint = planned_enclosure(6, 1.0, baseline, _FAR_END)
        assert Fraction(endpoint.lower) <= Fraction(11, 32) <= Fraction(endpoint.upper)
        assert endpoint.absolute_error <= _binomial.RESOLUTION
        target = min(11 / 32, endpoint.power)
        found = minimum_detectable_effect(6, baseline, _FAR_END, PowerDesign(power=target))
        assert found.mde_relative is not None
        assert found.mde_relative == pytest.approx(1.0, abs=2e-8)
        assert found.power >= target
        at_effect = achieved_power(6, found.mde_relative, baseline, _FAR_END)
        assert found.power == at_effect.power

        with pytest.raises(InvalidRequestError) as above:
            minimum_detectable_effect(6, baseline, _FAR_END, PowerDesign(power=11 / 32 + 1e-9))
        assert above.value.code == "power.minimum_detectable_effect.unattainable"

        below = minimum_detectable_effect(6, baseline, _FAR_END, PowerDesign(power=11 / 32 - 1e-9))
        assert below.power >= 11 / 32 - 1e-9

    @pytest.mark.slow
    def test_large_arm_point_admission_does_not_require_a_certified_mde(self):
        """Numerical PMF allowance is bounded internally rather than subtracted from power."""
        target = PowerDesign().power

        n, p_c = 1_000_000_000, 2e-7
        baseline, procedure = Baseline.from_proportion(p_c), _conversion()
        try:
            effect = minimum_detectable_effect(n, baseline, procedure)
        except InvalidRequestError as raised:
            assert raised.code == "power.minimum_detectable_effect.numerical_resolution"
            supplied = achieved_power(n, 0.5, baseline, procedure)
            assert supplied.power == planned_enclosure(n, 0.5, baseline, procedure).power
            assert supplied.mde_unavailable_reason == "numerical_resolution"
        else:
            assert effect.mde_relative is not None and effect.power >= target
            assert effect.power == achieved_power(n, effect.mde_relative, baseline, procedure).power

        sized = required_sample_size(0.5, baseline, procedure)
        units = sized.n_per_arm
        computed = planned_enclosure(units, 0.5, baseline, procedure)
        assert units > 10**8 and sized.power == computed.power >= target
        assert computed.absolute_error <= _binomial.RESOLUTION
        predecessor = achieved_power(units - 1, 0.5, baseline, procedure)
        assert predecessor.power < target


class TestPublishedPowerIsTheAdmittedPoint:
    """Every public path reports computed rejection mass within its error allowance."""

    @pytest.mark.parametrize(
        ("n_t", "allocation", "p_c", "lift", "overrides"),
        [
            (15, 0.6, 0.3, 0.8, {}),
            (12, 0.5, 0.25, 0.9, {"alternative": "greater", "null_lift": 0.2, "alpha": 0.01}),
            (10, 0.4, 0.5, -0.6, {"alternative": "less", "null_lift": -0.2}),
        ],
    )
    def test_achieved_power_matches_enumerated_runtime_probability(
        self, n_t, allocation, p_c, lift, overrides
    ):
        procedure = _conversion(**overrides)
        design = PowerDesign(allocation=allocation)
        baseline = Baseline.from_proportion(p_c)
        result = achieved_power(n_t, lift, baseline, procedure, design)
        computed = planned_enclosure(n_t, lift, baseline, procedure, design)
        runtime = _runtime_power(
            result.n_total - result.n_per_arm, n_t, p_c, p_c * (1.0 + lift), procedure
        )
        assert result.power_basis == computed.basis == "exact"
        assert computed.lower < computed.power
        assert result.power == computed.power
        assert abs(result.power - runtime) <= computed.absolute_error

    def test_a_target_the_computed_mass_meets_is_met_at_its_size(self):
        """The point estimate, not its lower bound, determines whether a size reaches target."""
        baseline, procedure, lift, n = Baseline.from_proportion(0.3), _conversion(), 1.0, 26
        computed = planned_enclosure(n, lift, baseline, procedure)
        design = PowerDesign(power=computed.power)
        assert computed.lower < design.power
        at_size = achieved_power(n, lift, baseline, procedure, design)
        sized = required_sample_size(lift, baseline, procedure, design)
        assert sized.n_per_arm == n
        assert at_size.power == design.power <= sized.power

    def test_every_public_path_publishes_the_same_point(self):
        baseline, procedure = Baseline.from_proportion(0.3), _conversion()
        n, lifts, targets = 24, [1.0, 0.6], [0.9, 0.8]
        curve = power_curve(
            n_per_arm=n, relative_lift=lifts, baseline=baseline, procedure=procedure, max_workers=1
        )
        for point, lift in zip(curve, lifts, strict=True):
            computed = planned_enclosure(n, lift, baseline, procedure)
            direct = achieved_power(n, lift, baseline, procedure)
            assert point.power == direct.power == computed.power

        # Solved effects may be shared, but numerical answers must not depend on prior targets.
        effects = power_curve(
            n_per_arm=n, target_power=targets, baseline=baseline, procedure=procedure, max_workers=1
        )
        for point, target in zip(effects, targets, strict=True):
            direct = minimum_detectable_effect(n, baseline, procedure, PowerDesign(power=target))
            assert point.mde_relative is not None and point.mde_relative == direct.mde_relative
            at_effect = planned_enclosure(n, point.mde_relative, baseline, procedure)
            assert point.power == direct.power == at_effect.power >= target
            assert point.power_basis == direct.power_basis == at_effect.basis

    def test_an_unattainable_target_names_the_endpoint_point_power(self):
        """The maximum named is the model's computed rejection mass."""
        with pytest.raises(InvalidRequestError) as refused:
            minimum_detectable_effect(
                6, Baseline.from_proportion(0.5), _FAR_END, PowerDesign(power=11 / 32 + 1e-9)
            )
        assert refused.value.code == "power.minimum_detectable_effect.unattainable"
        context: dict[str, Any] = dict(refused.value.context)
        assert context["maximum_power"] == pytest.approx(11 / 32, abs=1e-12)

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_shifted_null_and_compliance_report_power_at_the_returned_effect(self, alternative):
        baseline = Baseline(mean=0.3, var=0.21, compliance=0.8)
        procedure = _conversion(alternative=alternative, null_lift=0.1)
        design = PowerDesign(power=0.5)
        found = minimum_detectable_effect(60, baseline, procedure, design)
        assert found.mde_relative is not None
        implied = (
            math.expm1(math.log1p(0.1) + math.log1p(found.mde_relative * baseline.compliance))
            / baseline.compliance
        )
        evaluated = achieved_power(60, implied, baseline, procedure, design)
        assert found.power >= design.power
        assert found.power == pytest.approx(evaluated.power, abs=1e-12)
        assert found.power_basis == evaluated.power_basis
        assert found.mde_relative > 0 if alternative == "greater" else found.mde_relative < 0


_LATTICE_ARM, _LATTICE_RATE = 300, 0.05


def _lattice_rows(*fractions: float) -> list[tuple[int, int]]:
    """Consecutive control-count ranges holding the given fractions of the control arm's mass."""
    cumulative = np.cumsum(binom.pmf(np.arange(_LATTICE_ARM + 1), _LATTICE_ARM, _LATTICE_RATE))
    ranges: list[tuple[int, int]] = []
    first, total = 0, 0.0
    for fraction in fractions:
        total += fraction
        last = int(np.searchsorted(cumulative, total - 1e-12))
        ranges.append((first, last))
        first = last + 1
    ranges[-1] = (ranges[-1][0], _LATTICE_ARM)
    return ranges


def _decide_by_rows(monkeypatch, rules: list[tuple[int, int, str, int]]) -> None:
    """Replace the runtime decision by a hand-built set: a control row in ``(first, last)``
    rejects the treatment counts at or above ``threshold`` (kind ``plus``) or at or below it
    (``minus``) -- the shape of every binomial decision, with thresholds of this set's own."""
    from increment.power import _binomial

    def classify(decision, route, requests):
        masks = []
        for request in requests:
            counts = np.arange(request.j0, request.j1 + 1)
            mask = np.zeros(counts.size, bool)
            for first, last, kind, threshold in rules:
                if first <= request.x_c <= last and kind == request.kind:
                    mask = counts >= threshold if kind == "plus" else counts <= threshold
            masks.append(mask)
        return masks

    monkeypatch.setattr(_binomial, "classify", classify)


def _lattice_power(rate: float, rules: list[tuple[int, int, str, int]]) -> float:
    """Rejection probability of the hand-built set at treatment rate ``rate``, from the
    binomial laws alone (it shares no code with the planner)."""
    weights = binom.pmf(np.arange(_LATTICE_ARM + 1), _LATTICE_ARM, _LATTICE_RATE)
    total = 0.0
    for first, last, kind, threshold in rules:
        if kind == "plus":
            tail = binom.sf(threshold - 1, _LATTICE_ARM, rate)
        else:
            tail = binom.cdf(threshold, _LATTICE_ARM, rate)
        total += float(weights[first : last + 1].sum()) * float(tail)
    return total


def _lattice_crossing(rules: list[tuple[int, int, str, int]], target: float) -> float:
    """The first effect, scanning from the null, at which the hand-built set's power crosses
    ``target`` upward, found by a dense scan and a root solve of the independent power."""
    effects = np.linspace(0.0, 1.0 / _LATTICE_RATE - 1.0, 6001)[1:]
    power = np.array([_lattice_power(_LATTICE_RATE * (1.0 + m), rules) for m in effects])
    above = int(np.argmax(power > target * (1.0 + 1e-9)))
    assert power[above] > target * (1.0 + 1e-9)
    below = max(k for k in range(above) if power[k] <= target)
    return float(
        brentq(
            lambda m: _lattice_power(_LATTICE_RATE * (1.0 + m), rules) - target,
            effects[below],
            effects[below + 1],
            xtol=1e-15,
            rtol=1e-14,
        )
    )


def _lattice_effect(target: float) -> PowerResult:
    return minimum_detectable_effect(
        _LATTICE_ARM,
        Baseline.from_proportion(_LATTICE_RATE),
        _conversion(),
        PowerDesign(power=target),
    )


class TestEarliestDetectableRegionOnANonMonotoneLattice:
    """Interval exclusion must find the earliest band, within numerical effect tolerance,
    without assuming that point power or its selected cells vary monotonically."""

    def test_a_target_at_the_null_point_is_a_zero_effect_not_numerically_unresolved(self):
        baseline = Baseline.from_proportion(0.9)
        procedure = _conversion(null_lift=-0.25)
        allocation = 0.35
        at_null = planned_enclosure(
            10, -0.25, baseline, procedure, PowerDesign(allocation=allocation)
        )
        at_top = achieved_power(10, 0.11, baseline, procedure, PowerDesign(allocation=allocation))
        assert at_top.power < at_null.power / 2

        def effect(target: float) -> PowerResult:
            design = PowerDesign(power=target, allocation=allocation)
            return minimum_detectable_effect(10, baseline, procedure, design)

        with pytest.raises(InvalidRequestError) as raised:
            effect(at_null.power)
        assert raised.value.code == "power.minimum_detectable_effect.design_search_minimum"

        with pytest.raises(InvalidRequestError) as below:
            effect(at_null.power * (1.0 - 1e-9))
        assert below.value.code == "power.minimum_detectable_effect.design_search_minimum"
        with pytest.raises(InvalidRequestError) as above:
            effect(at_null.power * (1.0 + 1e-9))
        assert above.value.code == "power.minimum_detectable_effect.unattainable"

    def test_a_band_of_effects_is_found_when_the_far_end_does_not_reach_the_target(
        self, monkeypatch
    ):
        """Half the control mass rejects from 90 treatment counts and the other half stops
        rejecting above 180: power rises to one near a 30% treatment rate, falls to 0.568 by
        60%, and stays there to a rate of one. The target 0.8 is reached only inside that band,
        so the far end fails it."""
        rises, falls = _lattice_rows(0.5, 0.5)
        rules = [(*rises, "plus", 90), (*falls, "minus", 180)]
        _decide_by_rows(monkeypatch, rules)
        assert _lattice_power(1.0, rules) < 0.8 < _lattice_power(0.35, rules)
        expected = _lattice_crossing(rules, 0.8)

        found = _lattice_effect(0.8)

        assert found.mde_relative == pytest.approx(expected, rel=1e-8, abs=1e-8)
        assert found.power >= 0.8

    @pytest.mark.parametrize(
        ("target", "band"),
        [(0.33, "first"), (0.40, "second")],
    )
    def test_the_first_of_two_bands_is_found_across_the_trough_between_them(
        self, monkeypatch, target, band
    ):
        """The first band reaches 0.33 but not 0.40; the second reaches both.
        The answer must not jump across the first band just because later power is higher."""
        rows = _lattice_rows(0.10, 0.25, 0.20, 0.45)
        rules = [
            (*rows[0], "minus", 19),
            (*rows[1], "plus", 30),
            (*rows[2], "minus", 36),
            (*rows[3], "plus", 180),
        ]
        _decide_by_rows(monkeypatch, rules)
        expected = _lattice_crossing(rules, target)
        assert (expected < 2.0) == (band == "first")

        found = _lattice_effect(target)

        assert found.mde_relative == pytest.approx(expected, rel=1e-8, abs=1e-8)
        assert found.power >= target

    def test_a_peak_target_is_found_near_its_peak_or_explicitly_unresolved(self, monkeypatch):
        """A tangent target has no transverse crossing. It may be met by an admitted
        evaluated point or explicitly refused, but never silently moved to a later band."""
        rows = _lattice_rows(0.10, 0.25, 0.20, 0.45)
        rules = [(*rows[0], "minus", 19), (*rows[1], "plus", 30), (*rows[2], "minus", 36)]
        _decide_by_rows(monkeypatch, rules)
        top = minimize_scalar(
            lambda rate: -_lattice_power(rate, rules),
            bounds=(0.09, 0.14),
            method="bounded",
            options={"xatol": 1e-12},
        )
        peak, peak_effect = -float(top.fun), float(top.x) / _LATTICE_RATE - 1.0
        assert _lattice_power(1.0, rules) < _lattice_power(_LATTICE_RATE, rules) < peak

        try:
            found = _lattice_effect(peak)
        except InvalidRequestError as raised:
            assert raised.code == "power.minimum_detectable_effect.numerical_resolution"
            enclosure = raised.context["power_enclosure"]
            if enclosure is not None:
                assert isinstance(enclosure, tuple)
                lower, upper = enclosure
                assert isinstance(lower, (int, float)) and isinstance(upper, (int, float))
                assert lower <= peak <= upper
        else:
            assert found.mde_relative == pytest.approx(peak_effect, abs=1e-6)
            assert found.power >= peak

        with pytest.raises(InvalidRequestError) as above:
            _lattice_effect(peak * 1.001)
        assert above.value.code == "power.minimum_detectable_effect.unattainable"
        reached = _lattice_effect(peak * 0.97)
        assert reached.mde_relative is not None
        assert reached.mde_relative == pytest.approx(
            _lattice_crossing(rules, peak * 0.97), rel=1e-8, abs=1e-8
        )
        assert reached.mde_relative < peak_effect

    def test_closure_encloses_runtime_and_gapped_rejection_probabilities(self, monkeypatch):
        """A tail closure includes a band's non-rejecting gaps; it remains an upper bound."""
        from increment.power import _binomial

        decision = BinomialDecision(40, 40, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "greater")
        runtime = RejectionGeometry(decision, "exact")
        evaluated = runtime.evaluate(0.2, 0.4)
        assert runtime.closure_bound(0.2, 0.4, 0.4) >= evaluated.lower

        band = np.arange(14, 19)
        monkeypatch.setattr(
            _binomial,
            "classify",
            lambda decision, route, requests: [
                np.isin(np.arange(r.j0, r.j1 + 1), band)
                if r.kind == "plus"
                else np.zeros(r.j1 - r.j0 + 1, bool)
                for r in requests
            ],
        )
        gapped = RejectionGeometry(decision, "exact")
        gapped.evaluate(0.2, 0.4)
        bound = gapped.closure_bound(0.2, 0.4, 0.4)
        exact = float(binom.pmf(band, 40, 0.4).sum())
        assert bound > exact + 0.1


@pytest.mark.parametrize(
    ("reject_margin", "inferred"),
    [(2.0 * 5e-11, False), (math.nextafter(2.0 * 5e-11, 1.0), True)],
)
def test_rejection_is_inferred_only_beyond_twice_the_rounding(monkeypatch, reject_margin, inferred):
    """The replay rejects only at a strictly positive margin, so a neighbour's
    rejection margin of exactly twice the rounding room certifies nothing;
    one representable step above it does."""
    from increment.power import _binomial

    assert _binomial._ROOT_ROUNDING == 5e-11

    def margins(decision, groups, g, j):
        return np.full(j.size, -1.0), np.full(j.size, reject_margin)

    monkeypatch.setattr(_binomial, "_root_margins", margins)
    decision = BinomialDecision(50, 50, 1.0, binomial_rr.nuisance_beta(0.05), 0.05, "greater")
    ints = np.array([0])
    groups = _binomial._Groups(
        kind=ints,
        x_c=np.array([5]),
        a=np.array([0.1]),
        hi=np.array([0.2]),
        wlo=ints,
        width=np.array([51]),
        omitted=np.zeros(1),
        margin=np.zeros(1),
        j0=np.array([0]),
        j1=np.array([20]),
        dmin=ints,
        dmax=ints,
        offsets=np.zeros((1, 1), np.int64),
    )
    j = np.arange(21)
    settled, rejected = _binomial._root_settled(decision, groups, np.zeros(21, np.int64), j)
    assert settled.tolist() == [inferred] * 21
    assert rejected.tolist() == [inferred] * 21


class TestApproximateRoute:
    """(c) The Normal-tail replay stays within the errors the feasibility
    study measured against the runtime on its retained witness rows."""

    # ((n_c, n_t), (p_c, p_t), alpha, null ratio, alternative, runtime power,
    # the replay's power measured in the feasibility study).
    _WITNESSES = (
        ((20, 20), (0.05, 0.25), 0.05, 1.0, "two-sided", 0.123815805538, 0.123815805538),
        ((40, 60), (0.1, 0.25), 0.01, 1.2, "greater", 0.014667364180, 0.012439421369),
        ((60, 40), (0.3, 0.1), 0.05, 0.8, "less", 0.083402470113, 0.083402470113),
        ((25, 25), (0.8, 0.95), 0.05, 1.0, "two-sided", 0.051897357556, 0.051897357556),
        ((100, 100), (0.01, 0.15), 0.05, 1.0, "two-sided", 0.816115665003, 0.816115665003),
        ((75, 150), (0.1, 0.35), 0.01, 1.2, "greater", 0.625496290137, 0.617596569857),
        ((100, 100), (0.5, 0.75), 0.05, 1.0, "two-sided", 0.943014903387, 0.943014903387),
        ((20, 30), (0.4, 0.95), 0.05, 2.0, "greater", 0.085496937736, 0.085496937736),
    )

    @pytest.mark.parametrize("witness", _WITNESSES)
    def test_power_error_within_measured_error(self, witness):
        (n_c, n_t), (p_c, p_t), alpha, ratio, alternative, runtime, measured = witness
        tail = alpha / 2.0 if alternative == "two-sided" else alpha
        decision = BinomialDecision(
            n_c, n_t, ratio, binomial_rr.nuisance_beta(alpha), tail, alternative
        )
        geometry = RejectionGeometry(decision, "approximate")
        approximate = geometry.evaluate(p_c, p_t).power
        assert geometry.route == "approximate"
        # Twelve printed digits plus the windows' omitted mass.
        assert abs(approximate - runtime) <= abs(measured - runtime) + 2e-12

    def test_rare_large_arm_witness_decisions(self):
        """Counts ``(1e6, 4e6, 100, j)``: the runtime rejects from ``j = 510``
        (p+ 0.024196) where the replay rejects only from 511; neither rejects at 509."""
        n_c, n_t = 1_000_000, 4_000_000
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, 1.0, beta, 0.025, "greater")
        plus, _ = RejectionGeometry(decision, "approximate").cells(100, 100, 509, 512)
        runtime = [
            binomial_rr.p_plus(1.0, 100, n_c, j, n_t, beta, tail=0.025) < 0.025
            for j in (509, 510, 511, 512)
        ]
        assert runtime == [False, True, True, True]
        assert plus[0].tolist() == [False, False, True, True]

    def test_a_plan_beyond_the_replay_budget_leaves_its_lightest_pairs_ambiguous(self, monkeypatch):
        """Budgeted mass remains diagnostic when the remaining probability is material."""
        n, p_c, p_t = 300, 0.1, 0.2
        procedure = _conversion()
        beta = binomial_rr.nuisance_beta(procedure.compiled_alpha)
        decision = BinomialDecision(n, n, 1.0, beta, procedure.compiled_tail_alpha, "two-sided")
        plus, minus = RejectionGeometry(decision, "exact").cells(0, n, 0, n)
        weights = np.outer(binom.pmf(np.arange(n + 1), n, p_c), binom.pmf(np.arange(n + 1), n, p_t))
        exact = float(weights[plus | minus].sum())

        monkeypatch.setattr(_binomial, "EVALUATION_REPLAY_BUDGET", 2_000)
        enclosure = planned_enclosure(n, p_t / p_c - 1.0, Baseline.from_proportion(p_c), procedure)
        assert enclosure.basis == "approximate"
        assert enclosure.ambiguous > _binomial.RESOLUTION
        assert enclosure.lower <= enclosure.power <= exact + 1e-12
        assert enclosure.lower <= exact <= enclosure.upper
        # What the budget keeps is the heaviest part, so little mass is left out.
        assert enclosure.ambiguous < 0.05

    @pytest.mark.parametrize(
        ("n_c", "n_t", "p", "ratio", "alternative"),
        [
            (300, 300, 0.1, 1.0, "two-sided"),
            (150, 600, 0.3, 0.9, "greater"),
            (400, 400, 0.5, 1.2, "less"),
        ],
    )
    def test_inferred_root_exits_match_replaying_every_count(
        self, monkeypatch, n_c, n_t, p, ratio, alternative
    ):
        """Counts whose root exit is inferred from a neighbour's margin get
        the decision their own replay gives, across whole rows (including
        rows constant at both ends) and the band around each crossing."""
        from increment.power import _binomial

        tail = 0.025 if alternative == "two-sided" else 0.05
        decision = BinomialDecision(
            n_c, n_t, ratio, binomial_rr.nuisance_beta(0.05), tail, alternative
        )
        wc = _binomial._window(n_c, p)
        j_lo = _binomial._window(n_t, ratio * p * 0.6).lo
        j_hi = _binomial._window(n_t, min(1.0, ratio * p * 1.7)).hi
        requests = [
            _binomial._Request(x, kind, j_lo, j_hi)
            for x in range(max(0, wc.lo - 60), min(n_c, wc.hi + 60) + 1)
            for kind in decision.kinds
        ]
        inferred = _binomial.classify(decision, "approximate", requests)

        def replay_everything(decision, groups, group, j):
            return np.zeros(j.size, bool), np.zeros(j.size, bool)

        monkeypatch.setattr(_binomial, "_root_settled", replay_everything)
        replayed = _binomial.classify(decision, "approximate", requests)
        rows = [np.asarray(mask) for mask in replayed]
        assert any(row.all() or not row.any() for row in rows)
        assert any(row.any() and not row.all() for row in rows)
        for got, expected in zip(inferred, rows, strict=True):
            assert np.array_equal(got, expected)


class _LeafMeter:
    """Bytes of `_Leaves` arrays alive at once in the real replay, and the cells it keeps searching.

    Wraps the replay's own allocations (`_Leaves._blank`, `empty`, `take`) and adds nothing to
    what they return: each array is tracked by a weak reference and leaves the total when the
    replay drops it. `take` also counts the one field its fancy indexing gathers beside both
    sets of arrays, which is a temporary the assignment frees at once."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from increment.power import _binomial

        self.live = 0
        self.peak = 0
        self.searching: list[tuple[int, _binomial.Kind, int]] = []
        self._refs: dict[int, Any] = {}
        self._batch: Any = None
        leaves = _binomial._Leaves
        blank, empty, take = leaves._blank, leaves.empty.__func__, leaves.take
        replay = _binomial._replay

        def metered_blank(name, rows, capacity):
            array = blank(name, rows, capacity)
            self._track(array)
            return array

        def metered_empty(cls, rows, capacity):
            created = empty(cls, rows, capacity)
            self._track(created.count)
            return created

        def metered_take(this, rows, *, capacity):
            taken = take(this, rows, capacity=capacity)
            self._track(taken.count)
            gather = max(getattr(this, name).dtype.itemsize for name in leaves._FIELDS)
            self.peak = max(self.peak, self.live + gather * this.u.shape[1] * rows.size)
            batch = self._batch
            for row in rows.tolist():
                group = int(batch.group[row])
                kind = "plus" if batch.groups.kind[group] == 0 else "minus"
                self.searching.append((int(batch.groups.x_c[group]), kind, int(batch.j[row])))
            return taken

        def metered_replay(batch, tails, *, exact):
            self._batch = batch
            return replay(batch, tails, exact=exact)

        monkeypatch.setattr(leaves, "_blank", staticmethod(metered_blank))
        monkeypatch.setattr(leaves, "empty", classmethod(metered_empty))
        monkeypatch.setattr(leaves, "take", metered_take)
        monkeypatch.setattr(_binomial, "_replay", metered_replay)

    def _track(self, array: np.ndarray) -> None:
        key, size = id(array), array.nbytes
        self.live += size
        self.peak = max(self.peak, self.live)

        def release(_ref: Any) -> None:
            self.live -= size
            del self._refs[key]

        self._refs[key] = weakref.ref(array, release)

    def reset(self) -> None:
        self.peak = 0
        self.searching = []


class TestReplayFollowsTheRuntimeStopContract:
    """Planning reads ``binomial_rr.NUISANCE_STOP`` when it replays, so a search the cap ends
    is decided as the runtime decides it, and a longer search never rejects less."""

    _GAP = binomial_rr.NUISANCE_STOP.gap_fraction
    #: Caps that end most searches, end the hard ones, and the shipped contract.
    _RULES = (
        binomial_rr._StopRule(_GAP, 6),
        binomial_rr._StopRule(_GAP, 60),
        binomial_rr.NUISANCE_STOP,
    )
    _DESIGNS = [
        (40, 60, 1.0, 0.025, "two-sided", (3, 9, 14)),
        (50, 50, 1.2, 0.05, "greater", (4, 12)),
        (60, 40, 0.8, 0.05, "less", (6, 18)),
    ]

    @pytest.mark.parametrize(("n_c", "n_t", "ratio", "tail", "alternative", "controls"), _DESIGNS)
    def test_exact_replay_decides_as_the_runtime_under_each_contract(
        self, monkeypatch, n_c, n_t, ratio, tail, alternative, controls
    ):
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, ratio, beta, tail, alternative)
        first, last = controls[0], controls[-1]
        masks = []
        for rule in self._RULES:
            monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", rule)
            cells = RejectionGeometry(decision, "exact").cells(first, last, 0, n_t)
            masks.append(cells)
            for kind, mask in zip(("plus", "minus"), cells, strict=True):
                runtime = binomial_rr.p_plus if kind == "plus" else binomial_rr.p_minus
                for x_c in controls:
                    row = mask[x_c - first]
                    steps = np.flatnonzero(np.diff(row.astype(int)))
                    for x_t in sorted({0, n_t, *steps.tolist(), *(steps + 1).tolist()}):
                        decided = (
                            kind in decision.kinds
                            and runtime(ratio, x_c, n_c, x_t, n_t, beta, tail=tail) < tail
                        )
                        assert row[x_t] == decided, (rule, kind, x_c, x_t)
        self._assert_longer_searches_never_reject_less(masks)

    @pytest.mark.parametrize(("n_c", "n_t", "ratio", "tail", "alternative", "controls"), _DESIGNS)
    def test_approximate_replay_follows_the_contract(
        self, monkeypatch, n_c, n_t, ratio, tail, alternative, controls
    ):
        decision = BinomialDecision(
            n_c, n_t, ratio, binomial_rr.nuisance_beta(0.05), tail, alternative
        )
        masks = []
        for rule in self._RULES:
            monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", rule)
            masks.append(
                RejectionGeometry(decision, "approximate").cells(controls[0], controls[-1], 0, n_t)
            )
        self._assert_longer_searches_never_reject_less(masks)

    @staticmethod
    def _assert_longer_searches_never_reject_less(masks) -> None:
        for shorter, longer in zip(masks, masks[1:], strict=False):
            for short_mask, long_mask in zip(shorter, longer, strict=True):
                assert not (short_mask & ~long_mask).any()
        gained = sum(
            int((long & ~short).sum()) for short, long in zip(masks[0], masks[-1], strict=True)
        )
        assert gained > 0  # the cap binds somewhere on this design

    def test_deep_searches_continued_in_several_chunks_decide_as_one_pass_does(self, monkeypatch):
        """Rows still searching after the common splits continue in chunks sized from the leaf
        budget: with few common splits, many rows continue, and a budget cut to one row a chunk
        spreads them over several chunks without changing a decision."""
        from increment.power import _binomial

        decision = BinomialDecision(
            300, 300, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "two-sided"
        )
        whole = RejectionGeometry(decision, "approximate").cells(20, 40, 0, 300)
        monkeypatch.setattr(_binomial, "_COMMON_SPLITS", 6)
        monkeypatch.setattr(_binomial, "_LEAF_BUDGET_BYTES", 1e5)
        chunked = RejectionGeometry(decision, "approximate").cells(20, 40, 0, 300)
        for one_pass, in_chunks in zip(whole, chunked, strict=True):
            assert np.array_equal(one_pass, in_chunks)

    @pytest.mark.parametrize(
        ("route", "n_c", "n_t", "x_lo", "x_hi"),
        [("exact", 40, 60, 3, 14), ("approximate", 300, 300, 20, 40)],
    )
    def test_no_replay_exceeds_its_row_budget_even_for_one_request_wider_than_it(
        self, monkeypatch, route, n_c, n_t, x_lo, x_hi
    ):
        """A request spans a whole run of treatment counts. With the leaf budget cut to a
        megabyte a single request is several batches wide: it is replayed in pieces, no replay
        holds more rows than the budget allows, and no decision changes."""
        from increment.power import _binomial

        decision = BinomialDecision(
            n_c, n_t, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "two-sided"
        )
        whole = RejectionGeometry(decision, route).cells(x_lo, x_hi, 0, n_t)

        monkeypatch.setattr(_binomial, "_LEAF_BUDGET_BYTES", 1e6)
        batch_rows = _binomial._batch_rows(exact=route == "exact")
        assert batch_rows < n_t + 1
        replayed: list[int] = []
        live_batch = _binomial._classify_live

        def spy(decision, route, live, results):
            replayed.append(sum(item[1].j1 - item[1].j0 + 1 for item in live))
            return live_batch(decision, route, live, results)

        monkeypatch.setattr(_binomial, "_classify_live", spy)
        split = RejectionGeometry(decision, route).cells(x_lo, x_hi, 0, n_t)

        assert replayed and max(replayed) <= batch_rows
        for one_batch, in_pieces in zip(whole, split, strict=True):
            assert np.array_equal(one_batch, in_pieces)

    @pytest.mark.parametrize(
        ("route", "n_c", "n_t", "x_lo", "x_hi"),
        [("exact", 40, 60, 3, 14), ("approximate", 300, 300, 20, 40)],
    )
    def test_batches_of_six_split_searches_stay_within_the_leaf_budget(
        self, monkeypatch, route, n_c, n_t, x_lo, x_hi
    ):
        """A replay capped at six splits is one stage: its batch is largest while the rows still
        searching are copied out of the root's arrays, beside which the copy and the field being
        gathered are held. Handed only counts that keep searching, with the budget cut to about a
        hundred rows, every batch fills with such rows; the live leaf arrays must stay within
        the budget and no decision differ from an uncut budget's."""
        from increment.power import _binomial

        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n_c, n_t, 1.0, beta, 0.025, "two-sided")
        monkeypatch.setattr(binomial_rr, "NUISANCE_STOP", binomial_rr._StopRule(self._GAP, 6))
        meter = _LeafMeter(monkeypatch)
        wide = [
            _binomial._Request(x_c, kind, 0, n_t)
            for x_c in range(x_lo, x_hi + 1)
            for kind in decision.kinds
        ]
        _binomial.classify(decision, route, wide)
        # One request a count, so each batch is filled to its row limit.
        requests = [_binomial._Request(x_c, kind, j, j) for x_c, kind, j in meter.searching[:300]]
        assert len(requests) == 300
        uncut = _binomial.classify(decision, route, requests)

        budget = 6.5e4
        monkeypatch.setattr(_binomial, "_LEAF_BUDGET_BYTES", budget)
        assert _binomial._batch_rows(exact=route == "exact") < len(requests)
        meter.reset()
        cut = _binomial.classify(decision, route, requests)

        assert meter.live == 0
        assert meter.peak <= budget
        assert meter.peak >= 0.95 * budget  # the batches really fill the budget
        decided = np.concatenate(cut)
        assert decided.any()
        assert not decided.all()
        assert np.array_equal(decided, np.concatenate(uncut))
        if route == "exact":
            for req, rejected in zip(requests, decided, strict=True):
                runtime = binomial_rr.p_plus if req.kind == "plus" else binomial_rr.p_minus
                p = runtime(1.0, req.x_c, n_c, req.j0, n_t, beta, tail=0.025)
                assert rejected == (p < 0.025)


class TestACellsDecisionIsTheSameWhateverRequestDecidesIt:
    """A count pair's decision is the runtime's own, so it is the same whichever request, in
    whichever order, and on whichever geometry of the decision it is decided."""

    @pytest.mark.parametrize(("route", "n"), [("exact", 400), ("approximate", 2_800)])
    def test_cells_do_not_depend_on_the_order_windows_are_requested(self, route, n):
        """Classifying a window in pieces gives the cells one covering window gives."""
        beta = binomial_rr.nuisance_beta(0.05)
        decision = BinomialDecision(n, n, 1.0, beta, 0.025, "two-sided")
        wc, wt = _binomial._window(n, 0.1), _binomial._window(n, 0.12)
        mid = (wt.lo + wt.hi) // 2
        whole = RejectionGeometry(decision, route).cells(wc.lo, wc.hi, wt.lo, wt.hi)
        pieces = RejectionGeometry(decision, route)
        pieces.cells(wc.lo, wc.hi, mid, wt.hi)
        pieces.cells(wc.lo, wc.hi, wt.lo, mid + 10)
        pieces.cells((wc.lo + wc.hi) // 2, wc.hi, wt.lo + 3, wt.lo + 5)
        for got, expected in zip(pieces.cells(wc.lo, wc.hi, wt.lo, wt.hi), whole, strict=True):
            assert np.array_equal(got, expected)

    def test_the_companion_effect_is_the_direct_effect_whatever_the_supplied_lift(self):
        """The effect search after a supplied effect starts from that effect's decided cells;
        its answer is the direct search's, below, near and above the effect it finds."""
        baseline, procedure = Baseline.from_proportion(0.1), _conversion()
        direct = minimum_detectable_effect(300, baseline, procedure)
        assert direct.power_basis == "exact" and direct.mde_relative is not None
        for lift in (0.3, direct.mde_relative, 1.5, 4.0):
            companion = achieved_power(300, lift, baseline, procedure)
            assert (companion.mde_relative, companion.mde_unavailable_reason) == (
                direct.mde_relative,
                None,
            )
        curve = list(
            power_curve(
                n_per_arm=[300],
                relative_lift=[0.3, 4.0],
                baseline=baseline,
                procedure=procedure,
            )
        )
        assert [point.mde_relative for point in curve] == [direct.mde_relative] * 2


class TestAnIntervalBoundReadsOnlyTheCellsItDependsOn:
    """An evaluation reserved on a geometry is decided only once something depends on it. The
    bound over an interval of treatment rates read before the reserved evaluations are decided
    is the bound read after, and the evaluations decide as a fresh geometry's."""

    @pytest.mark.parametrize(
        ("alternative", "ratio", "p_c", "rates"),
        [
            ("two-sided", 1.0, 0.10, (0.13, 0.19, 0.4)),
            ("greater", 1.2, 0.10, (0.14, 0.2, 0.45)),
            ("less", 0.9, 0.30, (0.26, 0.2, 0.05)),
        ],
    )
    def test_the_bound_before_reserved_evaluations_is_the_bound_after(
        self, alternative, ratio, p_c, rates
    ):
        n = 300
        tail = 0.025 if alternative == "two-sided" else 0.05
        decision = BinomialDecision(n, n, ratio, binomial_rr.nuisance_beta(0.05), tail, alternative)
        fresh = [RejectionGeometry(decision, "exact").evaluate(p_c, rate) for rate in rates]
        geometry = RejectionGeometry(decision, "exact")
        geometry.evaluate(p_c, p_c)
        reserved = [geometry.prepare(p_c, rate) for rate in reversed(rates)]
        bounds = []
        for far in rates:
            low, high = sorted((p_c, far))
            bounds.append(geometry.closure_bound_before(p_c, low, high))
        completed = [geometry.complete(evaluation) for evaluation in reversed(reserved)]
        assert completed == fresh
        for far, before in zip(rates, bounds, strict=True):
            low, high = sorted((p_c, far))
            assert before == geometry.closure_bound(p_c, low, high)

    def test_a_reserved_evaluation_completes_as_a_fresh_one_after_other_requests(self):
        """Requests decided between reserving and completing an evaluation, overlapping its
        window, leave its enclosure the one a fresh geometry reports."""
        decision = BinomialDecision(
            300, 300, 1.0, binomial_rr.nuisance_beta(0.05), 0.025, "two-sided"
        )
        geometry = RejectionGeometry(decision, "exact")
        reserved = geometry.prepare(0.1, 0.16)
        geometry.evaluate(0.1, 0.1)
        geometry.evaluate(0.1, 0.14)
        geometry.evaluate(0.1, 0.19)
        assert geometry.complete(reserved) == RejectionGeometry(decision, "exact").evaluate(
            0.1, 0.16
        )


_SIZING_CASES = [
    pytest.param(Baseline.from_proportion(0.5), {}, PowerDesign(), 0.16, id="dense-half"),
    pytest.param(Baseline.from_proportion(0.001), {}, PowerDesign(), 0.5, id="rare"),
    pytest.param(Baseline.from_proportion(0.1), {}, PowerDesign(allocation=0.8), 0.3, id="unequal"),
    pytest.param(
        Baseline.from_proportion(0.1),
        {"alternative": "greater", "null_lift": 0.2},
        PowerDesign(),
        0.45,
        id="shifted-null",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(("baseline", "overrides", "design", "lift"), _SIZING_CASES)
def test_size_is_a_verified_crossing_with_the_companion_of_its_own_size(
    baseline, overrides, design, lift
):
    """The returned size reaches the target at the power ``achieved_power`` reports for it, its
    predecessor does not, and its companion effect is the one planned at that size alone."""
    procedure = _conversion(**overrides)
    sized = required_sample_size(lift, baseline, procedure, design)
    assert sized.power_basis == "exact"
    assert sized.power >= design.power
    at = achieved_power(sized.n_per_arm, lift, baseline, procedure, design)
    below = achieved_power(sized.n_per_arm - 1, lift, baseline, procedure, design)
    assert at.power_basis == sized.power_basis
    assert at.power == pytest.approx(sized.power, abs=1e-11)
    assert (at.mde_relative, at.mde_unavailable_reason) == (
        sized.mde_relative,
        sized.mde_unavailable_reason,
    )
    assert below.power < design.power


class TestRouting:
    """(d) The basis is a function of the inputs, and every plan the runtime
    does not decide with the binomial test keeps the log-ratio model."""

    def test_repeated_calls_agree(self):
        baseline = Baseline.from_proportion(0.2)
        first = achieved_power(120, 0.4, baseline, _conversion())
        second = achieved_power(120, 0.4, baseline, _conversion())
        assert first == second
        assert first.power_basis == "exact"

    @pytest.mark.parametrize(
        "procedure",
        [
            make_procedure(),
            make_procedure(metric_type="conversion"),
            _conversion(decision_method="cuped"),
            ArmPlanningProcedure.standard("conversion", clustered=True),
        ],
        ids=["mean", "absorbed_conversion", "cuped_conversion", "clustered_conversion"],
    )
    def test_non_binomial_plans_are_asymptotic(self, procedure):
        overrides: dict[str, float] = {}
        if procedure.decision_method.variance_reduction == "cuped":
            overrides["cuped_rho"] = 0.3
        if procedure.dependence == "cluster":
            overrides.update(cluster_icc=0.01, avg_cluster_size=5.0)
        baseline = Baseline(mean=0.1, var=0.09, **overrides)
        for result in (
            achieved_power(2_000, 0.2, baseline, procedure),
            minimum_detectable_effect(2_000, baseline, procedure),
            required_sample_size(0.2, baseline, procedure),
        ):
            assert result.power_basis == "asymptotic"

    def test_sequential_conversion_is_asymptotic(self):
        from increment.semantics.models import InferenceSpec

        procedure = ArmPlanningProcedure.standard(
            "conversion", inference=InferenceSpec(kind="asymptotic_mean")
        )
        result = achieved_power(2_000, 0.2, Baseline.from_proportion(0.1), procedure)
        assert result.power_basis == "asymptotic"


class TestPowerBasisSurvivesSerialization:
    """(e) The basis travels with every serialized form."""

    def test_result_round_trips(self):
        result = achieved_power(60, 0.8, Baseline.from_proportion(0.2), _conversion())
        assert result.power_basis == "exact"
        dumped = result.model_dump(mode="json")
        assert dumped["power_basis"] == "exact"
        assert PowerResult.model_validate(json.loads(json.dumps(dumped))) == result

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_curve_rows_and_frames_carry_the_basis(self, backend):
        curve = power_curve(
            n_per_arm=[60],
            relative_lift=[0.5, 0.8],
            baseline=Baseline.from_proportion(0.2),
            procedure=[_conversion(), make_procedure()],
        )
        expected = ["exact", "asymptotic", "exact", "asymptotic"]
        assert [row["power_basis"] for row in curve.to_dicts()] == expected
        frame: Any = curve.to_frame(backend=backend)
        column = frame["power_basis"]
        values = column.to_pylist() if backend == "pyarrow" else list(column)
        assert [str(value) for value in values] == expected

    def test_curve_matches_scalar_solvers(self):
        baseline = Baseline.from_proportion(0.2)
        curve = power_curve(
            n_per_arm=60, relative_lift=[0.5, 0.8], baseline=baseline, procedure=_conversion()
        )
        for point, lift in zip(curve, (0.5, 0.8), strict=True):
            direct = achieved_power(60, lift, baseline, _conversion())
            assert (point.power, point.power_basis, point.mde_relative) == (
                direct.power,
                direct.power_basis,
                direct.mde_relative,
            )
            assert math.isfinite(point.power)
