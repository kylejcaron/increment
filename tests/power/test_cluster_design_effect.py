"""Tests for the cluster-randomization planning knob on ``Baseline``.

``cluster_icc`` / ``avg_cluster_size`` / ``cluster_size_cv`` size a
cluster-randomized experiment: the design effect
``1 + ((1 + cv^2) * m_bar - 1) * rho`` INFLATES ``effective_var``, opposite
``icc``'s absorption reduction. Covers the mapping and cluster counts (an
unset knob leaves every solver unchanged); that the knob-sized experiment
hits nominal power while an uncorrected one is badly anticonservative
(``@pytest.mark.parameter_recovery`` + fast smoke variant); and that the
``(1 + cv^2)`` unequal-size correction earns its keep - at cv > 0 the
uncorrected design effect under-sizes measurably.

Simulations draw CLUSTER MEANS directly (``y_ij = mu + b_j + e_ij`` gives mean
``N(mu, sigma_b^2 + sigma_e^2 / m)``) - a fraction of a unit-level simulation's draws.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import t

from increment.errors import InvalidRequestError
from increment.power.core import (
    Baseline,
    PowerDesign,
    _cluster_dof,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
    segment_pairwise_achieved_power,
    segment_pairwise_minimum_detectable_effect,
    segment_pairwise_required_sample_size,
)

from ._procedures import make_procedure

MEAN = 20.0
VAR = 400.0


# The mapping itself


class TestDesignEffect:
    def test_defaults_give_exactly_one(self):
        b = Baseline(mean=MEAN, var=VAR)
        assert b.design_effect == 1.0
        assert b.effective_var == VAR

    def test_zero_icc_with_declared_clusters_still_gives_exactly_one(self):
        b = Baseline(mean=MEAN, var=VAR, avg_cluster_size=50.0)
        assert b.design_effect == 1.0
        assert b.effective_var == VAR

    def test_kish_formula(self):
        b = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        assert b.design_effect == pytest.approx(1.0 + 49 * 0.05)
        assert b.effective_var == pytest.approx(VAR * 3.45)

    def test_unequal_size_correction_inflates_further(self):
        equal = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        uneven = Baseline(
            mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0, cluster_size_cv=1.0
        )
        assert uneven.design_effect == pytest.approx(1.0 + (2.0 * 50.0 - 1.0) * 0.05)
        assert uneven.design_effect > equal.design_effect

    def test_inflation_is_the_opposite_direction_to_absorption_icc(self):
        """The two ICC-named knobs must never be confused: same value, opposite sign."""
        absorbed = Baseline(mean=MEAN, var=VAR, icc=0.30)
        clustered = Baseline(mean=MEAN, var=VAR, cluster_icc=0.30, avg_cluster_size=10.0)
        assert absorbed.effective_var < VAR < clustered.effective_var

    def test_knobs_multiply_through_effective_var(self):
        b = Baseline(
            mean=MEAN, var=VAR, cuped_rho=0.6, icc=0.2, cluster_icc=0.1, avg_cluster_size=11.0
        )
        assert b.effective_var == pytest.approx(VAR * (1 - 0.36) * (1 - 0.2) * 2.0)


class TestValidation:
    def test_icc_above_one_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=MEAN, var=VAR, cluster_icc=1.0, avg_cluster_size=10.0)
        assert exc_info.value.code == "power.baseline.cluster_icc"

    def test_cluster_size_below_one_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=MEAN, var=VAR, avg_cluster_size=0.5)
        assert exc_info.value.code == "power.baseline.avg_cluster_size"
        assert exc_info.value.context["avg_cluster_size"] == 0.5

    def test_negative_cv_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=MEAN, var=VAR, cluster_size_cv=-0.1)
        assert exc_info.value.code == "power.baseline.cluster_size_cv"

    def test_positive_icc_without_a_cluster_refused_by_name(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=MEAN, var=VAR, cluster_icc=0.05)
        assert exc_info.value.code == "power.baseline.cluster_icc_avg"
        assert exc_info.value.context["cluster_icc"] == 0.05


# Reported cluster counts


class TestClusterCounts:
    def test_none_under_unit_randomization(self):
        r = required_sample_size(0.02, Baseline(mean=MEAN, var=VAR), procedure=make_procedure())
        assert r.n_clusters_per_arm is None
        assert r.n_clusters_total is None

    def test_ceiled_per_arm_then_summed(self):
        b = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        r = required_sample_size(
            0.1,
            b,
            make_procedure(dependence="cluster", decision_method="unadjusted"),
            design=PowerDesign(allocation=0.3),
        )
        n_c = r.n_total - r.n_per_arm
        assert r.n_clusters_per_arm == math.ceil(r.n_per_arm / 50.0)
        assert r.n_clusters_total == r.n_clusters_per_arm + math.ceil(n_c / 50.0)
        # Ceiling per arm, not once on n_total: a part-cluster in each arm
        # costs two whole clusters.
        assert r.n_clusters_total >= math.ceil(r.n_total / 50.0)

    def test_cluster_reference_is_conservative_even_at_zero_icc(self):
        b = Baseline(mean=MEAN, var=VAR, avg_cluster_size=50.0)
        plain = required_sample_size(0.02, Baseline(mean=MEAN, var=VAR), procedure=make_procedure())
        r = required_sample_size(
            0.02, b, procedure=make_procedure(dependence="cluster", decision_method="unadjusted")
        )
        assert r.n_total > plain.n_total
        assert r.power >= PowerDesign().power
        assert r.n_clusters_per_arm == math.ceil(r.n_per_arm / 50.0)

    @pytest.mark.parametrize("solver", [achieved_power, minimum_detectable_effect])
    @pytest.mark.parametrize("allocation", [0.03125, 0.96875])
    def test_one_cluster_in_either_arm_carries_the_cluster_code(self, solver, allocation):
        b = Baseline(mean=MEAN, var=VAR, avg_cluster_size=10.0)
        n_t = 10 if allocation == 0.03125 else 310
        args = (n_t, 0.1) if solver is achieved_power else (n_t,)
        with pytest.raises(InvalidRequestError) as exc_info:
            solver(
                *args,
                b,
                procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
                design=PowerDesign(allocation=allocation),
            )
        assert exc_info.value.code == "power.cluster.too_few_clusters"
        assert exc_info.value.context["k_t"] == n_t // 10
        assert exc_info.value.context["k_c"] == 32 - n_t // 10
        assert exc_info.value.context["k_total"] == 32

    @pytest.mark.parametrize("n_t", [11, 20, 40])
    def test_two_through_eight_clusters_are_admitted(self, n_t):
        b = Baseline(mean=MEAN, var=VAR, avg_cluster_size=10.0)
        procedure = make_procedure(dependence="cluster", decision_method="unadjusted")
        result = achieved_power(n_t, 0.1, b, procedure=procedure)
        mde = minimum_detectable_effect(n_t, b, procedure=procedure)
        assert result.n_clusters_total == 2 * math.ceil(n_t / 10)
        assert 0 < result.power < 1
        assert mde.power >= PowerDesign().power

    def test_required_size_search_reaches_two_clusters_per_arm(self):
        b = Baseline(mean=100.0, var=1.0, avg_cluster_size=50.0)
        procedure = make_procedure(dependence="cluster", decision_method="unadjusted")
        result = required_sample_size(0.5, b, procedure=procedure)
        assert result.n_clusters_total == 4
        assert result.n_clusters_per_arm == 2
        assert achieved_power(result.n_per_arm, 0.5, b, procedure=procedure).power >= 0.8

    def test_valid_small_cluster_plan_uses_t_reference(self):
        clustered = Baseline(mean=MEAN, var=VAR, avg_cluster_size=2.0)
        unit = Baseline(mean=MEAN, var=VAR)
        cluster_result = achieved_power(
            10,
            0.1,
            clustered,
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        unit_result = achieved_power(10, 0.1, unit, procedure=make_procedure())
        assert cluster_result.n_clusters_total == 10
        assert cluster_result.power < unit_result.power

    def test_cluster_solver_triangle_matches_at_recruited_counts(self):
        b = Baseline(mean=MEAN, var=VAR, cluster_icc=0.2, avg_cluster_size=5.0)
        r = required_sample_size(
            0.3, b, procedure=make_procedure(dependence="cluster", decision_method="unadjusted")
        )
        p = achieved_power(
            r.n_per_arm,
            0.3,
            b,
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        m = minimum_detectable_effect(
            r.n_per_arm,
            b,
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        assert p.power == pytest.approx(r.power)
        assert m.mde_relative == pytest.approx(r.mde_relative)

    def test_triggered_cluster_solver_round_trip_uses_assigned_pair(self):
        """Ceiling after trigger inflation must not change K, SE, or MDE."""
        b = Baseline(
            mean=MEAN,
            var=VAR,
            cluster_icc=0.2,
            avg_cluster_size=5.0,
            trigger_rate=0.37,
            cluster_participation=1.0,
        )
        procedure = make_procedure(dependence="cluster", decision_method="unadjusted")
        d = PowerDesign(allocation=0.3)
        r = required_sample_size(0.3, b, procedure, design=d)
        p = achieved_power(r.n_per_arm, 0.3, b, procedure, design=d)
        m = minimum_detectable_effect(
            r.n_per_arm, b, make_procedure(dependence="cluster", decision_method="unadjusted"), d
        )
        assert r.power == pytest.approx(p.power)
        assert r.mde_relative == pytest.approx(m.mde_relative)
        assert r.n_clusters_total == p.n_clusters_total == m.n_clusters_total
        assert r.n_triggered_per_arm == round(r.n_per_arm * b.trigger_rate)

    def test_other_solvers_report_counts_too(self):
        b = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        assert (
            achieved_power(
                1000,
                0.1,
                b,
                procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
            ).n_clusters_total
            == 40
        )
        assert (
            minimum_detectable_effect(
                1000,
                b,
                procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
            ).n_clusters_total
            == 40
        )

    @pytest.mark.parametrize("solver", ["required", "achieved", "mde"])
    def test_clustered_sequential_is_refused_by_every_solver(self, solver):
        from increment.errors import CapabilityError
        from increment.estimation.sequential import GaussianScoreMixture

        inference = GaussianScoreMixture()
        clustered = Baseline(mean=MEAN, var=VAR, avg_cluster_size=5.0)
        with pytest.raises(CapabilityError) as raised:
            if solver == "required":
                required_sample_size(
                    0.1,
                    clustered,
                    procedure=make_procedure(
                        inference=inference,
                        dependence="cluster",
                        decision_method="unadjusted",
                        population="assigned",
                    ),
                )
            elif solver == "achieved":
                achieved_power(
                    100,
                    0.1,
                    clustered,
                    procedure=make_procedure(
                        inference=inference,
                        dependence="cluster",
                        decision_method="unadjusted",
                        population="assigned",
                    ),
                )
            else:
                minimum_detectable_effect(
                    100,
                    clustered,
                    procedure=make_procedure(
                        inference=inference,
                        dependence="cluster",
                        decision_method="unadjusted",
                        population="assigned",
                    ),
                )
        assert raised.value.code == "arm.inference.cluster"

    def test_segment_pairwise_leaves_counts_unset(self):
        """N is experiment-wide across every segment and the two baselines may
        declare different cluster sizes, so no single count is meaningful."""
        b = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        r = segment_pairwise_required_sample_size(
            0.3,
            0.1,
            0.2,
            0.2,
            b,
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        assert r.n_clusters_per_arm is None
        assert r.n_clusters_total is None

    def test_inflation_still_flows_into_segment_pairwise(self):
        plain = Baseline(mean=MEAN, var=VAR)
        clustered = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        n_plain = segment_pairwise_required_sample_size(
            0.3, 0.1, 0.2, 0.2, plain, procedure=make_procedure()
        ).n_total
        n_cl = segment_pairwise_required_sample_size(
            0.3,
            0.1,
            0.2,
            0.2,
            clustered,
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        ).n_total
        assert n_cl / n_plain == pytest.approx(3.45, rel=1e-3)


# Regression: an unset knob must not move a single solver output


class TestUnsetKnobIsBitForBitUnchanged:
    """The cluster knob leaves baseline counts and variance unchanged when
    unset. Goldens are the own-arm model's: the treatment arm's log-scale
    variance sits at ``var / (mean * (1 + lift))^2``."""

    def test_required_sample_size_golden(self):
        r = required_sample_size(
            0.02, Baseline(mean=MEAN, var=VAR), make_procedure(), design=PowerDesign()
        )
        # per-unit (1/(0.5*1.02^2) + 1/0.5) * (400/20^2) * (z_.025 + z_.8)^2 / log1p(.02)^2
        # = 78506.75 total -> 39254 per arm.
        assert (r.n_per_arm, r.n_total) == (39254, 78508)
        assert r.effective_var == 400.0

    def test_achieved_power_golden(self):
        a = achieved_power(
            25000, 0.02, Baseline(mean=MEAN, var=VAR), make_procedure(), design=PowerDesign()
        )
        assert (a.n_per_arm, a.n_total) == (25_000, 50_000)

    def test_golden_with_every_other_knob_engaged(self):
        b = Baseline(mean=MEAN, var=VAR, cuped_rho=0.6, icc=0.2, compliance=0.5)
        procedure = make_procedure(alpha=0.01, alternative="greater", decision_method="cuped")
        d = PowerDesign(power=0.9, allocation=0.3)
        r = required_sample_size(0.05, b, procedure, design=d)
        # effective lift .025: per-unit (1/(0.3*1.025^2) + 1/0.7) * (204.8/400)
        # * (z_.01 + z_.9)^2 / log1p(.025)^2 -> 15089 treatment, 35208 control.
        assert (r.n_per_arm, r.n_total) == (15089, 50297)
        assert r.effective_var == 204.8

    def test_explicit_zero_knob_matches_unset(self):
        unset = Baseline(mean=MEAN, var=VAR)
        explicit = Baseline(
            mean=MEAN, var=VAR, cluster_icc=0.0, avg_cluster_size=1.0, cluster_size_cv=0.0
        )
        assert required_sample_size(
            0.02, explicit, procedure=make_procedure()
        ) == required_sample_size(0.02, unset, procedure=make_procedure())

    def test_cluster_reference_stacks_on_design_effect(self):
        """The input-side design effect remains exact; t adds finite-K cost."""
        knob = Baseline(mean=MEAN, var=VAR, cluster_icc=0.05, avg_cluster_size=50.0)
        by_hand = Baseline(mean=MEAN, var=VAR * 3.45)
        clustered = required_sample_size(
            0.1, knob, procedure=make_procedure(dependence="cluster", decision_method="unadjusted")
        )
        normal = required_sample_size(0.1, by_hand, procedure=make_procedure())
        assert knob.effective_var == by_hand.effective_var
        assert clustered.n_total > normal.n_total
        assert (
            achieved_power(
                5000,
                0.1,
                knob,
                procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
            ).power
            < achieved_power(5000, 0.1, by_hand, procedure=make_procedure()).power
        )


# Simulation helpers


def _power_equal_clusters(
    k_per_arm: int, m: float, rho: float, lift: float, reps: int, seed: int, alpha: float = 0.05
) -> float:
    """Empirical power of the joint relative-null test at *k_per_arm*.

    Equal-sized clusters, so the arm mean is the mean of cluster means and
    its clustered variance is their sample variance over K. The analyzer
    uses a t reference with ``k_per_arm - 1`` degrees of freedom. The treatment
    arm shifts the MEAN by the lift and keeps the same absolute variance,
    the planning model's assumption for a mean-like metric.
    """
    var_b, var_e = rho * VAR, (1.0 - rho) * VAR
    sd_g = math.sqrt(var_b + var_e / m)
    rng = np.random.default_rng(seed)
    g_t = MEAN * (1.0 + lift) + rng.normal(0.0, sd_g, size=(reps, k_per_arm))
    g_c = MEAN + rng.normal(0.0, sd_g, size=(reps, k_per_arm))
    y_t, y_c = g_t.mean(1), g_c.mean(1)
    v_t = g_t.var(1, ddof=1) / k_per_arm
    v_c = g_c.var(1, ddof=1) / k_per_arm
    # At relative null zero, joint inversion tests the additive numerator.
    statistic = (y_t - y_c) / np.sqrt(v_t + v_c)
    return float(np.mean(np.abs(statistic) > t.isf(alpha / 2.0, k_per_arm - 1)))


def _power_unequal_clusters(
    k_per_arm: int,
    m_bar: float,
    cv: float,
    rho: float,
    lift: float,
    reps: int,
    seed: int,
    alpha: float = 0.05,
) -> float:
    """Empirical power with gamma-distributed cluster sizes (mean *m_bar*, cv *cv*).

    Sizes vary, so the arm mean is the size-weighted mean of cluster totals
    and the variance is the Liang-Zeger sum of squared cluster influences.
    As above, the treatment arm shifts its mean and keeps the variance.
    """
    var_b, var_e = rho * VAR, (1.0 - rho) * VAR
    rng = np.random.default_rng(seed)
    shape = 1.0 / cv**2

    def arm(mean: float) -> tuple[np.ndarray, np.ndarray]:
        m = np.maximum(1.0, rng.gamma(shape, m_bar / shape, size=(reps, k_per_arm)))
        b = rng.normal(0.0, math.sqrt(var_b), size=m.shape)
        e = rng.normal(0.0, math.sqrt(var_e), size=m.shape) * np.sqrt(m)
        totals = m * (mean + b) + e
        units = m.sum(1)
        y = totals.sum(1) / units
        influence = (totals - y[:, None] * m) / units[:, None]
        return y, (k_per_arm / (k_per_arm - 1)) * (influence**2).sum(1)

    y_t, v_t = arm(MEAN * (1.0 + lift))
    y_c, v_c = arm(MEAN)
    # At relative null zero, joint inversion tests the additive numerator.
    statistic = (y_t - y_c) / np.sqrt(v_t + v_c)
    return float(np.mean(np.abs(statistic) > t.isf(alpha / 2.0, k_per_arm - 1)))


def _uncorrected_clusters_per_arm(lift: float, m: float) -> int:
    """Clusters a planner would recruit having ignored the design effect."""
    naive = required_sample_size(lift, Baseline(mean=MEAN, var=VAR), procedure=make_procedure())
    return math.ceil(naive.n_per_arm / m)


# Fast smoke variants of the simulations above


def test_equal_cluster_sim_smoke():
    """Small-N: the corrected design beats the uncorrected one by a mile."""
    rho, m, lift = 0.5, 5.0, 0.30
    sized = required_sample_size(
        lift,
        Baseline(mean=MEAN, var=VAR, cluster_icc=rho, avg_cluster_size=m),
        procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
    )
    assert sized.n_clusters_per_arm is not None
    corrected = _power_equal_clusters(sized.n_clusters_per_arm, m, rho, lift, 400, 11)
    naive_k = _uncorrected_clusters_per_arm(lift, m)
    uncorrected = _power_equal_clusters(naive_k, m, rho, lift, 400, 11)
    assert corrected > 0.65
    assert uncorrected < 0.55


def test_unequal_cluster_sim_smoke():
    """Small-N: the cv correction recruits more clusters and buys more power."""
    rho, m_bar, cv, lift = 0.3, 10.0, 1.0, 0.30
    with_cv = required_sample_size(
        lift,
        Baseline(mean=MEAN, var=VAR, cluster_icc=rho, avg_cluster_size=m_bar, cluster_size_cv=cv),
        procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
    )
    without_cv = required_sample_size(
        lift,
        Baseline(mean=MEAN, var=VAR, cluster_icc=rho, avg_cluster_size=m_bar),
        procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
    )
    assert with_cv.n_clusters_per_arm is not None
    assert without_cv.n_clusters_per_arm is not None
    assert with_cv.n_clusters_per_arm > without_cv.n_clusters_per_arm
    hi = _power_unequal_clusters(with_cv.n_clusters_per_arm, m_bar, cv, rho, lift, 300, 5)
    lo = _power_unequal_clusters(without_cv.n_clusters_per_arm, m_bar, cv, rho, lift, 300, 5)
    assert hi > lo


# Calibration: does the design effect size the experiment correctly?

REPS = 4000
# 3 Monte Carlo standard errors at 80% power over REPS draws, rounded up.
TOL = 0.02


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestSizesClusterRandomizedExperimentsToNominalPower:
    @pytest.mark.parametrize(
        ("rho", "m", "lift", "seed"),
        [(0.5, 20.0, 0.10, 7), (0.2, 10.0, 0.10, 13), (0.05, 50.0, 0.20, 21)],
    )
    def test_corrected_design_hits_nominal_power(self, rho, m, lift, seed):
        baseline = Baseline(mean=MEAN, var=VAR, cluster_icc=rho, avg_cluster_size=m)
        sized = required_sample_size(
            lift,
            baseline,
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        assert sized.n_clusters_per_arm is not None
        measured = _power_equal_clusters(sized.n_clusters_per_arm, m, rho, lift, REPS, seed)
        assert measured == pytest.approx(sized.power, abs=TOL + 0.01)

    @pytest.mark.parametrize(
        ("rho", "m", "lift", "seed", "ceiling"),
        [(0.5, 20.0, 0.10, 7, 0.25), (0.2, 10.0, 0.10, 13, 0.50)],
    )
    def test_uncorrected_design_is_anticonservative(self, rho, m, lift, seed, ceiling):
        """Ignoring the design effect ships an experiment that cannot see the effect."""
        k = _uncorrected_clusters_per_arm(lift, m)
        measured = _power_equal_clusters(k, m, rho, lift, REPS, seed)
        assert measured < ceiling

    @pytest.mark.parametrize(("cv", "uncorrected_ceiling"), [(0.5, 0.75), (1.0, 0.60)])
    def test_cv_correction_restores_nominal_power_under_unequal_sizes(
        self, cv, uncorrected_ceiling
    ):
        rho, m_bar, lift = 0.3, 20.0, 0.10
        with_cv = required_sample_size(
            lift,
            Baseline(
                mean=MEAN, var=VAR, cluster_icc=rho, avg_cluster_size=m_bar, cluster_size_cv=cv
            ),
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        without_cv = required_sample_size(
            lift,
            Baseline(mean=MEAN, var=VAR, cluster_icc=rho, avg_cluster_size=m_bar),
            procedure=make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
        assert with_cv.n_clusters_per_arm is not None
        assert without_cv.n_clusters_per_arm is not None
        corrected = _power_unequal_clusters(
            with_cv.n_clusters_per_arm, m_bar, cv, rho, lift, REPS, 3
        )
        uncorrected = _power_unequal_clusters(
            without_cv.n_clusters_per_arm, m_bar, cv, rho, lift, REPS, 3
        )
        assert corrected == pytest.approx(with_cv.power, abs=TOL + 0.01)
        assert uncorrected < uncorrected_ceiling


@pytest.mark.parametrize("n_t,allocation,expected_df", [(20, 0.2, 1), (80, 0.8, 1), (40, 0.5, 3)])
@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_unbalanced_cluster_power_uses_smaller_arm_reference(
    n_t, allocation, expected_df, alternative
):
    from scipy.stats import nct

    from increment.estimation._tails import student_t_isf

    baseline = Baseline(mean=20, var=100, avg_cluster_size=10)
    design = PowerDesign(allocation=allocation)
    procedure = make_procedure(
        dependence="cluster", decision_method="unadjusted", alternative=alternative
    )
    lift = -0.1 if alternative == "less" else 0.1
    n_c = math.ceil(n_t * (1 - allocation) / allocation)
    result = achieved_power(n_t, lift, baseline, procedure=procedure, design=design)
    variance = 100 / ((20 * (1 + lift)) ** 2 * n_t) + 100 / (20**2 * n_c)
    nc = math.log1p(lift) / math.sqrt(variance)
    # SciPy 1.14's t.isf(0.05, 1) misses cot(0.05 pi) by 2e-11; use the exact quantile.
    critical = student_t_isf(0.025 if alternative == "two-sided" else 0.05, float(expected_df))
    expected = (
        nct.cdf(-critical, expected_df, nc)
        if alternative == "less"
        else nct.sf(critical, expected_df, nc)
    )
    if alternative == "two-sided":
        expected += nct.cdf(-critical, expected_df, nc)
    assert result.power == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("solver", ["achieved", "mde"])
def test_segment_cluster_floor_reaches_public_solver(solver):
    baseline = Baseline(mean=20, var=100, avg_cluster_size=10)
    with pytest.raises(InvalidRequestError) as raised:
        if solver == "achieved":
            segment_pairwise_achieved_power(
                10,
                0.2,
                0.0,
                0.5,
                0.5,
                baseline,
                procedure=make_procedure(dependence="cluster"),
            )
        else:
            segment_pairwise_minimum_detectable_effect(
                10,
                0.5,
                0.5,
                baseline,
                procedure=make_procedure(dependence="cluster"),
            )
    assert raised.value.code == "power.segment_clustered_baseline"
    assert raised.value.context["k_total"] == 2


def test_triggered_cluster_participation_separates_recruitment_and_inference():
    full = Baseline(
        mean=20,
        var=400,
        avg_cluster_size=10,
        trigger_rate=0.2,
        cluster_participation=1.0,
        cluster_icc=0.2,
    )
    half = full.model_copy(update={"cluster_participation": 0.5})
    unequal = full.model_copy(update={"trigger_rate": 0.25, "cluster_size_cv": 0.6})
    assert full.design_effect == pytest.approx(1.2)
    assert half.design_effect == pytest.approx(1.6)
    assert unequal.design_effect == pytest.approx(1.48)
    assert _cluster_dof(100, 100, full) == pytest.approx(9)
    assert _cluster_dof(100, 100, half) == pytest.approx(4)
    assert _cluster_dof(20, 20, unequal) == pytest.approx(1)


def test_triggered_cluster_participation_is_required():
    with pytest.raises(InvalidRequestError) as raised:
        Baseline(mean=20, var=400, avg_cluster_size=4, trigger_rate=0.5)
    assert raised.value.code == "power.baseline.cluster_participation_triggered"


def test_represented_cluster_floor_does_not_round_subinteger_product():
    p = (2.0 - 2.0**-20) / 2**30
    baseline = Baseline(
        mean=20,
        var=400,
        avg_cluster_size=10,
        trigger_rate=0.2,
        cluster_participation=p,
    )
    with pytest.raises(InvalidRequestError) as raised:
        achieved_power(
            10 * 2**30,
            0.05,
            baseline,
            make_procedure(dependence="cluster", decision_method="unadjusted"),
        )
    assert raised.value.code == "power.cluster.too_few_clusters"
    assert raised.value.context["k_t"] == raised.value.context["k_c"] == 1
    assert raised.value.context["k_total"] == 2
    assert raised.value.context["recruited_k_total"] == 2**31


def test_represented_cluster_floor_respects_roundoff_bounds():
    import math

    baseline = Baseline(
        mean=20,
        var=400,
        avg_cluster_size=10,
        trigger_rate=0.2,
        cluster_participation=2 / 49,
    )
    assert _cluster_dof(490, 490, baseline) == 1
    below = baseline.model_copy(update={"cluster_participation": 0.2 - 6 * math.ulp(0.2)})
    with pytest.raises(InvalidRequestError) as raised:
        _cluster_dof(100, 100, below)
    assert raised.value.code == "power.cluster.too_few_clusters"


def test_cluster_icc_requires_analyzed_cluster_membership():
    with pytest.raises(InvalidRequestError) as raised:
        Baseline(
            mean=20,
            var=400,
            avg_cluster_size=10,
            trigger_rate=0.1,
            cluster_participation=1.0,
            cluster_icc=0.2,
        )
    assert raised.value.code == "power.baseline.cluster_icc_avg"


def test_cluster_one_unit_mean_survives_pilot_ratio_roundoff():
    inputs = {
        "mean": 20,
        "var": 400,
        "avg_cluster_size": 24.5,
        "trigger_rate": 2 / 49,
        "cluster_participation": 1.0,
    }
    baseline = Baseline(**inputs)
    assert baseline.design_effect == 1.0
    assert _cluster_dof(49, 49, baseline) == 1
    with pytest.raises(InvalidRequestError) as raised:
        Baseline(**inputs, cluster_icc=0.2)
    assert raised.value.code == "power.baseline.cluster_icc_avg"


def test_cluster_participation_refuses_nonrepresentable_analyzed_size():
    with pytest.raises(InvalidRequestError) as raised:
        Baseline(
            mean=20, var=400, avg_cluster_size=10, trigger_rate=0.2, cluster_participation=5e-324
        )
    assert raised.value.code == "power.baseline.cluster_analyzed_mean"
    assert raised.value.context["cluster_participation"] == 5e-324
