"""The plan-time GaussianScoreMixture boundary must be bit-identical to the
runtime's own asymptotic_mean boundary at matched information fraction,
alpha, and one-sidedness -- the repository's required guard for a quantity
computed twice, and the regression guard for the previously found one-sided
finding."""

import math
from fractions import Fraction

import pytest

from increment.estimation.asymptotic_mean import (
    count_boundary,
    directional_alpha,
    mixture_r_star,
)
from increment.power.sequential import GaussianScoreMixture, planning_bounds


class TestMixtureRStar:
    def test_solves_its_own_fixed_point(self):
        alpha = Fraction(1, 20)
        r = mixture_r_star(alpha)
        target = -2 * math.log(float(alpha))
        assert float(r) - math.log1p(float(r)) == pytest.approx(target, abs=1e-9)

    def test_independent_of_sample_size(self):
        alpha = Fraction(1, 100)
        r = float(mixture_r_star(alpha))
        rho_400 = math.sqrt(r / 400)
        rho_5000 = math.sqrt(r / 5000)
        assert rho_400 / rho_5000 == pytest.approx(math.sqrt(5000 / 400), rel=1e-12)


class TestPlanningBoundaryMatchesRuntimeBoundary:
    """Reproduces the previously found equivalence probe as a permanent
    guard: matches the runtime at every N (the shape is N-independent) AND
    at both two-sided and one-sided alternatives (a previously found
    one-sided tuning defect)."""

    @pytest.mark.parametrize("alpha", [0.05, 0.01, 1e-3])
    @pytest.mark.parametrize("expected_decision_sample_size", [400, 5000, 23156])
    @pytest.mark.parametrize("fraction_pct", [10, 50, 90, 100])
    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_matches_at_matched_information_fraction(
        self, alpha, expected_decision_sample_size, fraction_pct, alternative
    ):
        raw_alpha = Fraction(alpha)
        rho = Fraction(math.sqrt(float(mixture_r_star(raw_alpha) / expected_decision_sample_size)))
        n = max(1, round(fraction_pct / 100 * expected_decision_sample_size))
        boundary_alpha = directional_alpha(raw_alpha, alternative)
        runtime_bound = math.sqrt(float(count_boundary(n, boundary_alpha, rho)))
        exit_side = "both" if alternative == "two-sided" else "upper"
        plan_bound = planning_bounds(
            GaussianScoreMixture(), (n / expected_decision_sample_size,), 1.0, alpha, exit_side
        )[0]
        assert plan_bound == pytest.approx(runtime_bound, rel=1e-9)

    def test_one_sided_and_two_sided_diverge_by_a_full_doubling_of_alpha(self):
        """Regression pin for a previously found defect: a one-sided
        boundary must differ from the two-sided one at the same t (it is
        NOT simply narrower/wider by a constant factor -- confirms the
        fix is not a no-op that happens to pass the parametrized case
        above)."""
        two_sided = planning_bounds(GaussianScoreMixture(), (0.5,), 1.0, 0.05, "both")[0]
        one_sided = planning_bounds(GaussianScoreMixture(), (0.5,), 1.0, 0.05, "upper")[0]
        assert one_sided != pytest.approx(two_sided, rel=1e-6)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_one_sided_secondary_is_planned_against_its_runtime_boundary(alternative):
    """Planning a primary and a secondary must reproduce the boundaries the
    runtime builds for the same plan roles. Each cell's allocation and family
    membership are read from the registration the runtime binds for a plan
    with one primary and one secondary; its boundary level is the runtime's
    own ``boundary_alpha``. At one planned look power is Phi(drift - c) with
    the same drift for both, so the primary's planned power fixes the drift
    and the secondary's planned power must follow from its runtime boundary."""
    from scipy.special import ndtr, ndtri

    from increment.estimation.arm_contract import ArmPlanningProcedure
    from increment.estimation.asymptotic_mean import boundary_alpha
    from increment.power import Baseline, achieved_power
    from increment.semantics.models import InferenceSpec
    from tests.test_mixed_family_auto import _bound

    registration = _bound(compliance=None, secondaries=["orders"]).inference.registration
    cells = {cell.metric: cell for cell in registration.roster}
    inference = InferenceSpec(kind="asymptotic_mean")
    procedures = {
        "revenue": ArmPlanningProcedure.standard(
            "mean", alternative=alternative, inference=inference
        ),
        "orders": ArmPlanningProcedure.standard(
            "mean",
            alternative=alternative,
            role="secondary",
            secondaries=1,
            q=float(registration.q),
            inference=inference,
        ),
    }
    lift = 0.05 if alternative == "greater" else -0.05
    baseline = Baseline(mean=20.0, var=400.0)
    powers = {
        metric: achieved_power(6000, lift, baseline, procedure, planned_looks=1).power
        for metric, procedure in procedures.items()
    }

    def runtime_z(cell):
        # One look at the declared sample size is information fraction one.
        rho = Fraction(math.sqrt(float(mixture_r_star(cell.alpha) / 1000)))
        level = boundary_alpha(cell.alpha, alternative, e_value_dual=cell.family)
        return math.sqrt(float(count_boundary(1000, level, rho)))

    drift = runtime_z(cells["revenue"]) + float(ndtri(powers["revenue"]))
    expected = float(ndtr(drift - runtime_z(cells["orders"])))
    assert powers["orders"] == pytest.approx(expected, rel=1e-6)
