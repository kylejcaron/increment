"""Planning mirrors the runtime's conversion route (``conversion_inference``).

Under ``auto`` the runtime takes the delta-method route where all four arm counts are dense
for the tail allocation and the finite-sample route elsewhere, so planning classifies each
plan from the probability that its random counts are dense: a ``dense`` plan is the
closed-form asymptotic model (no replay, no cell budget, no arm ceiling), a ``sparse`` plan
the replay, and a ``borderline`` plan the smaller of the two powers. ``finite_sample`` plans
the replay at every size.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from scipy.stats import norm

from increment.errors import CodedError, InvalidRequestError
from increment.estimation import binomial_rr
from increment.estimation.arm_contract import ArmPlanningProcedure
from increment.estimation.conversion_route import dense_min_count, planning_route
from increment.power import (
    Baseline,
    PowerDesign,
    achieved_power,
    minimum_detectable_effect,
    power_curve,
    required_sample_size,
)
from tests.estimation._conversion_counts import runtime_rejection_rate
from tests.power._procedures import make_procedure

TAIL = 0.025  # alpha = 0.05, two-sided


def _plan(mode: str = "auto", **overrides: Any) -> ArmPlanningProcedure:
    return make_procedure(
        metric_type="conversion",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        conversion_inference=mode,
        **overrides,
    )


def _mean_plan(**overrides: Any) -> ArmPlanningProcedure:
    """The closed-form asymptotic model a mean metric is planned with."""
    return make_procedure(
        metric_type="mean",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        **overrides,
    )


def _dense_n(p: float) -> int:
    """Arm size at which every count lies far inside the dense band for ``p``."""
    return math.ceil(6 * dense_min_count(TAIL) / min(p, 1 - p))


def _wald_power(n: int, p_c: float, lift: float, *, alpha: float, alternative: str) -> float:
    """Independent delta-method power of the runtime's dense route: the log risk ratio's
    standard error from each arm's own Bernoulli variance, a Normal reference."""
    p_t = p_c * (1 + lift)
    se = math.sqrt((1 - p_c) / (n * p_c) + (1 - p_t) / (n * p_t))
    z = abs(math.log1p(lift)) / se
    if alternative == "two-sided":
        crit = norm.isf(alpha / 2)
        return float(norm.sf(crit - z) + norm.cdf(-crit - z))
    return float(norm.sf(norm.isf(alpha) - z))


class TestDensePlansAreTheClosedForm:
    @pytest.mark.parametrize(
        ("alternative", "lift"), [("two-sided", 0.04), ("greater", 0.04), ("less", -0.04)]
    )
    def test_power_is_the_delta_method_rejection_probability(self, alternative, lift):
        """Planning and the runtime's delta-method route derive the same probability
        independently: each arm's Bernoulli variance in the log risk ratio's standard error."""
        p = 0.2
        n = _dense_n(p)
        result = achieved_power(
            n, lift, Baseline.from_proportion(p), _plan(alternative=alternative)
        )
        assert result.power_basis == "asymptotic"
        expected = _wald_power(n, p, lift, alpha=0.05, alternative=alternative)
        assert result.power == pytest.approx(expected, abs=1e-4)

    def test_the_size_and_effect_solvers_invert_that_probability(self):
        p, lift = 0.2, 0.05
        baseline = Baseline.from_proportion(p)
        sized = required_sample_size(lift, baseline, _plan())
        assert sized.power_basis == "asymptotic"
        n = sized.n_per_arm
        assert _wald_power(n, p, lift, alpha=0.05, alternative="two-sided") >= 0.8 - 1e-4
        assert _wald_power(n - 1, p, lift, alpha=0.05, alternative="two-sided") < 0.8 + 1e-4
        mde = minimum_detectable_effect(n, baseline, _plan())
        assert mde.power_basis == "asymptotic"
        assert mde.mde_relative is not None
        assert _wald_power(
            n, p, mde.mde_relative, alpha=0.05, alternative="two-sided"
        ) == pytest.approx(0.8, abs=1e-3)

    @pytest.mark.parametrize("planner", ["achieved_power", "minimum_detectable_effect", "size"])
    def test_a_dense_plan_beyond_the_replay_bound_is_planned_in_closed_form(self, planner):
        """Five million units per arm at a 5% baseline replay 48 million count cells, past the
        bound a replayed decision is refused at: a dense plan answers, so it replays none of
        the effects between the null and its answer."""
        baseline = Baseline.from_proportion(0.05)
        n = 5_000_000
        if planner == "achieved_power":
            result = achieved_power(n, 0.01, baseline, _plan())
        elif planner == "minimum_detectable_effect":
            result = minimum_detectable_effect(n, baseline, _plan())
        else:
            result = required_sample_size(0.005, baseline, _plan())
        assert result.power_basis == "asymptotic"

    def test_the_default_plan_is_auto_and_reaches_the_closed_form_at_a_million_units(self):
        procedure = ArmPlanningProcedure.standard("conversion")
        assert procedure.decision_method.conversion_inference == "auto"
        result = achieved_power(1_000_000, 0.02, Baseline.from_proportion(0.05), procedure)
        assert result.power_basis == "asymptotic"
        assert 0.0 < result.power < 1.0

    def test_arms_above_the_finite_sample_ceiling_are_planned_with_power(self):
        """No evaluator ceiling binds the closed form: a dense plan at 2e9 per arm has the
        asymptotic power, where the finite-sample runtime refuses every count pair."""
        n = 2 * binomial_rr.FINITE_SAMPLE_MAX_ARM_SIZE
        baseline = Baseline.from_proportion(0.1)
        result = achieved_power(n, 0.001, baseline, _plan())
        assert result.power_basis == "asymptotic"
        assert result.power > 0.9

    def test_a_curve_over_dense_sizes_is_closed_form_throughout(self):
        p = 0.1
        baseline = Baseline.from_proportion(p)
        n = _dense_n(p)
        curve = power_curve(
            n_per_arm=[n, 2 * n], relative_lift=[0.03, 0.05], baseline=baseline, procedure=_plan()
        )
        assert {row["power_basis"] for row in curve.to_dicts()} == {"asymptotic"}


class TestSparsePlansAreTheReplay:
    @pytest.mark.parametrize(
        ("n", "p", "lift"), [(120, 0.1, 0.6), (400, 0.02, 0.8), (150, 0.4, 0.3)]
    )
    def test_a_sparse_plan_equals_the_finite_sample_plan_exactly(self, n, p, lift):
        baseline = Baseline.from_proportion(p)
        assert planning_route(n, n, p, p * (1 + lift), tail_alpha=TAIL, mode="auto") == "sparse"
        auto = achieved_power(n, lift, baseline, _plan("auto"))
        pinned = achieved_power(n, lift, baseline, _plan("finite_sample"))
        assert auto == pinned
        assert auto.power_basis in ("exact", "approximate")

    def test_a_sparse_size_search_equals_the_finite_sample_search(self):
        baseline = Baseline.from_proportion(0.02)
        assert required_sample_size(0.9, baseline, _plan("auto")) == required_sample_size(
            0.9, baseline, _plan("finite_sample")
        )


class TestExplicitFiniteSamplePlans:
    def test_a_dense_plan_pinned_to_finite_sample_is_replayed_not_closed_form(self):
        p = 0.1
        n = 2_000
        baseline = Baseline.from_proportion(p)
        result = achieved_power(n, 0.5, baseline, _plan("finite_sample"))
        assert result.power_basis != "asymptotic"

    def test_a_clustered_plan_cannot_plan_finite_sample(self):
        with pytest.raises(CodedError) as raised:
            ArmPlanningProcedure.standard(
                "conversion", conversion_inference="finite_sample", clustered=True
            )
        assert raised.value.code == "estimation.binomial.finite_sample_unavailable"

    def test_a_sequential_plan_cannot_plan_finite_sample(self):
        from increment.semantics.models import InferenceSpec

        with pytest.raises(CodedError) as raised:
            ArmPlanningProcedure.standard(
                "conversion",
                conversion_inference="finite_sample",
                inference=InferenceSpec(kind="asymptotic_mean"),
            )
        assert raised.value.code == "estimation.binomial.finite_sample_unavailable"

    def test_a_mean_metric_cannot_plan_finite_sample(self):
        with pytest.raises(CodedError) as raised:
            ArmPlanningProcedure.standard("mean", conversion_inference="finite_sample")
        assert raised.value.code == "conversion_inference.finite_sample.metric_type"


class TestRefusalsNameTheAutoRoute:
    """Every replay-side planning refusal under an explicit ``finite_sample`` decision names
    ``conversion_inference='auto'`` as the way to plan the dense counts in closed form; the
    default route plans the same design."""

    def _refusal(self, call) -> InvalidRequestError:
        with pytest.raises(InvalidRequestError) as raised:
            call()
        return raised.value

    def test_the_replay_bound_refusal_names_auto_and_auto_plans_the_design(self):
        n = 5_000_000
        baseline = Baseline.from_proportion(0.05)
        refusal = self._refusal(lambda: achieved_power(n, 0.02, baseline, _plan("finite_sample")))
        assert refusal.code == "power.binomial_replay_bound_exceeded"
        assert refusal.context["conversion_inference"] == "finite_sample"
        assert achieved_power(n, 0.02, baseline, _plan("auto")).power_basis == "asymptotic"

    def test_the_size_search_refusal_names_auto(self, monkeypatch):
        from increment.power import core

        monkeypatch.setattr(core, "PLANNING_CELL_CEILING", 12_000)
        baseline = Baseline.from_proportion(0.05)
        refusal = self._refusal(
            lambda: required_sample_size(0.05, baseline, _plan("finite_sample"))
        )
        assert refusal.code == "power.binomial_replay_bound_exceeded"
        sized = required_sample_size(0.05, baseline, _plan("auto"))
        assert sized.power_basis == "asymptotic"

    @pytest.mark.parametrize("alpha", [1e-12, 1e-10, 3e-8])
    def test_the_tail_level_refusal_names_auto_and_auto_sizes_the_design(self, alpha):
        baseline = Baseline.from_proportion(0.05)
        refusal = self._refusal(
            lambda: required_sample_size(0.5, baseline, _plan("finite_sample", alpha=alpha))
        )
        assert refusal.code == "power.binomial_tail_level_unrepresentable"
        sized = required_sample_size(0.5, baseline, _plan("auto", alpha=alpha))
        assert sized.power >= PowerDesign().power
        assert sized.power_basis in ("asymptotic", "approximate", "exact")

    def test_the_arm_floor_refusal_names_auto(self):
        baseline = Baseline.from_proportion(0.05)
        design = PowerDesign(allocation=1e-9)
        refusal = self._refusal(
            lambda: required_sample_size(0.5, baseline, _plan("finite_sample"), design)
        )
        assert refusal.code == "power.binomial_arm_ceiling_below_smallest_design"


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestPlanningMatchesTheRuntimeRejectionRate:
    """Planning and the runtime derive the same rejection probability independently, so at
    matched inputs they agree: a dense plan within ``4 * se + 0.003`` of the simulated
    rejection rate of ``estimate_lift`` (which takes the delta-method route on every draw)."""

    @pytest.mark.parametrize(("p", "lift"), [(0.3, 0.03), (0.05, 0.08)])
    def test_dense_power_is_the_simulated_runtime_rate(self, p, lift):
        n = _dense_n(p)
        planned = achieved_power(n, lift, Baseline.from_proportion(p), _plan())
        assert planned.power_basis == "asymptotic"
        rate, se, share = runtime_rejection_rate(n, n, p, p * (1 + lift), reps=2_000, seed=20261004)
        assert share == 1.0
        assert abs(planned.power - rate) <= 4 * se + 0.003


def test_a_baseline_cuped_rho_does_not_discard_an_explicit_finite_sample_request():
    baseline = Baseline(mean=0.3, var=0.21, cuped_rho=0.4)
    with pytest.raises(CodedError) as raised:
        achieved_power(5_000, 0.1, baseline, _plan("finite_sample"))
    assert raised.value.code == "conversion_inference.finite_sample.cuped"
    # The default route plans the CUPED-adjusted decision as before.
    assert achieved_power(5_000, 0.1, baseline, _plan("auto")).power_basis == "asymptotic"


def test_the_dense_runtime_rate_smoke_agrees_with_planning():
    p = 0.3
    n = _dense_n(p)
    planned = achieved_power(n, 0.03, Baseline.from_proportion(p), _plan())
    rate, se, share = runtime_rejection_rate(n, n, p, p * 1.03, reps=400, seed=20261004)
    assert share == 1.0
    assert abs(planned.power - rate) <= 4 * se + 0.003
