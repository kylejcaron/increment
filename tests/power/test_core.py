"""Tests for power/sample-size/MDE analysis.

Step 1: Closed-form - required_sample_size matches the own-arm log-scale
two-sample n formula within rounding; Baseline.from_proportion feeds the
Bernoulli-shape variance for a conversion metric.
Step 2: Duality round-trip - n -> power ~= design.power (+/-0.01) and
n -> MDE ~= delta (+/-tol).
Step 3: CUPED - rho reduces n; effective_var reflects the reduction;
two-sided vs one-sided and bonferroni shift n in the expected direction.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2 as scipy_chi2
from scipy.stats import norm

from increment import Analysis
from increment.errors import CapabilityError, InvalidRequestError, UnsupportedRequestError
from increment.estimation.arm_contract import AbsoluteDecisionPolicy, ArmPlanningProcedure
from increment.estimation.armstats import SummaryStats
from increment.estimation.quantile import _log_quantile_se_impl, log_quantile_se
from increment.estimation.sequential import GaussianScoreMixture
from increment.frame import MetricSpec, synthesise_metric
from increment.power.core import (
    _SHIFT_SPAN,
    Baseline,
    PowerDesign,
    PowerResult,
    QuantileBaseline,
    _ArmPlan,
    _compute_arms,
    _deterministic_quantile_se,
    _log1mexp,
    _prepare_solver,
    _quantile_n_min,
    _segment_arm_sizes,
    _supplied_effect,
    achieved_power,
    joint_q_power_fixed,
    joint_q_power_random,
    minimum_detectable_effect,
    required_sample_size,
    segment_pairwise_achieved_power,
    segment_pairwise_minimum_detectable_effect,
    segment_pairwise_required_sample_size,
)
from increment.semantics.models import QuantileMetric

from ._procedures import make_procedure
from ._results import available_mde

# Step 1: Closed-form test


class TestClosedForm:
    """required_sample_size matches the own-arm closed form within rounding."""

    def test_two_sample_n_matches_own_arm_formula(self):
        """Baseline(mean=1.0, var=2.0), delta=0.20, alpha=0.05, power=0.80,
        equal allocation. The treatment arm's log-scale variance is
        ``var / (1.2 * mean)^2``, below the control's, so N is smaller
        than the equal-variance textbook 1889: about 1600, n_per_arm 801.
        """
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure()
        d = PowerDesign()
        result = required_sample_size(procedure=procedure, relative_lift=0.2, baseline=b, design=d)
        expected = math.ceil(_independent_h1_n_total(b, procedure, d, 0.2) * d.allocation)
        assert abs(result.n_per_arm - expected) <= 1, (
            f"Expected ~{expected}, got {result.n_per_arm}"
        )
        assert abs(result.n_per_arm - 801) <= 1

    def test_proportion_baseline(self):
        """Baseline.from_proportion(0.10) -> mean=0.10, var=0.09; a conversion
        metric plans the treatment arm with the Bernoulli shape at ITS OWN
        rate 0.12, matching the two-arm formula."""
        b = Baseline.from_proportion(0.10)
        assert b.mean == 0.10
        assert b.var == pytest.approx(0.09, rel=1e-12)
        assert b.cuped_rho == 0.0

        procedure = make_procedure(metric_type="conversion")
        d = PowerDesign()
        result = required_sample_size(procedure=procedure, relative_lift=0.2, baseline=b, design=d)
        expected = math.ceil(
            _independent_h1_n_total(b, procedure, d, 0.2, bounded=True) * d.allocation
        )
        assert abs(result.n_per_arm - expected) <= 1, (
            f"Expected ~{expected}, got {result.n_per_arm}"
        )

    def test_power_result_fields(self):
        """PowerResult contains all expected fields with sensible values."""
        b = Baseline(mean=1.0, var=2.0)
        result = required_sample_size(procedure=make_procedure(), relative_lift=0.2, baseline=b)
        assert result.n_per_arm > 0
        assert result.n_total >= 2 * result.n_per_arm
        assert 0.79 <= result.power <= 0.81  # close to design power
        assert result.mde_relative is not None
        assert result.mde_relative > 0
        assert result.mde_unavailable_reason is None
        assert result.effective_var == 2.0  # no CUPED


# Step 1b: Conversion (Bernoulli) planning sizes the treatment arm at its
# own rate -- closed form, independent of the solver.


class TestConversionPlanningMatchesTreatmentArmVariance:
    """required_sample_size sizes a conversion metric from BOTH arms' own
    Bernoulli log-scale variances. The raw variance p*(1-p) is not
    monotonic in p (it peaks at 0.5), but the delta method's per-unit
    LOG-SCALE term p*(1-p)/p**2 == (1-p)/p is strictly decreasing over
    (0, 1): a positive lift shrinks the treatment arm's log-scale
    variance below the control's and a negative lift grows it. Sizing
    both arms from the control's term alone over-recruited ~10% for a
    +20% lift and under-recruited ~12% for a -20% lift at a 10% baseline;
    the planned n now matches the treatment-arm-aware n within rounding.
    ``n_true`` is derived independently, never from the solver."""

    _P_CONTROL = 0.10

    @staticmethod
    def _bernoulli_log_scale_variance(p: float) -> float:
        """Delta-method per-unit variance of log(p_hat): p*(1-p)/p**2."""
        return p * (1.0 - p) / p**2

    @pytest.mark.parametrize("relative_lift", [0.20, -0.20])
    def test_planned_n_matches_treatment_arm_aware_n(self, relative_lift: float) -> None:
        baseline = Baseline.from_proportion(self._P_CONTROL)
        procedure = make_procedure(metric_type="conversion", alternative="two-sided")
        design = PowerDesign()

        result = required_sample_size(
            relative_lift=relative_lift, baseline=baseline, procedure=procedure, design=design
        )
        n_planned = result.n_per_arm  # equal allocation (design.allocation=0.5): n_T == n_C

        z_sum = norm.isf(procedure.compiled_tail_alpha) + norm.ppf(design.power)
        theta = math.log1p(relative_lift)

        p_treatment = self._P_CONTROL * (1.0 + relative_lift)
        v_treatment = self._bernoulli_log_scale_variance(p_treatment)
        v_control = self._bernoulli_log_scale_variance(self._P_CONTROL)

        # Treatment-arm-aware n per arm at equal allocation: se2(n) = (v_T + v_C) / n.
        n_true = (v_treatment + v_control) * z_sum**2 / theta**2

        assert n_planned == math.ceil(n_true) or n_planned == math.ceil(n_true) + 1, (
            f"relative_lift={relative_lift}: planned n={n_planned} vs. "
            f"treatment-arm-aware n_true={n_true:.3f}"
        )
        assert (
            _independent_h1_power(
                n_planned, n_planned, baseline, procedure, relative_lift, bounded=True
            )
            >= design.power
        )


# Step 2: Duality round-trip test


class TestDuality:
    """n -> power -> n round-trips correctly."""

    def test_power_round_trip(self):
        """achieved_power(required_sample_size(...).n_per_arm) ~= design.power."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8)

        n_result = required_sample_size(
            procedure=procedure, relative_lift=0.2, baseline=b, design=d
        )
        power_result = achieved_power(
            procedure=procedure,
            n_per_arm=n_result.n_per_arm,
            relative_lift=0.2,
            baseline=b,
            design=d,
        )
        assert abs(power_result.power - d.power) <= 0.01, (
            f"Power round-trip failed: expected {d.power:.4f}, got {power_result.power:.4f}"
        )

    def test_mde_round_trip(self):
        """minimum_detectable_effect(n) ~= delta +/- tol."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8)

        n_result = required_sample_size(
            procedure=procedure, relative_lift=0.2, baseline=b, design=d
        )
        mde_result = minimum_detectable_effect(
            procedure=procedure, n_per_arm=n_result.n_per_arm, baseline=b, design=d
        )
        mde = available_mde(mde_result)
        assert abs(mde - 0.20) <= 0.005, f"MDE round-trip failed: expected ~0.20, got {mde:.6f}"

    def test_full_triangle(self):
        """required_sample_size -> achieved_power -> minimum_detectable_effect are self-consistent."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure()
        d = PowerDesign(power=0.9)  # 90% power

        n_res = required_sample_size(procedure=procedure, relative_lift=0.15, baseline=b, design=d)
        p_res = achieved_power(
            procedure=procedure, n_per_arm=n_res.n_per_arm, relative_lift=0.15, baseline=b, design=d
        )
        m_res = minimum_detectable_effect(
            procedure=procedure, n_per_arm=n_res.n_per_arm, baseline=b, design=d
        )

        assert abs(p_res.power - 0.90) <= 0.01
        assert abs(available_mde(m_res) - 0.15) <= 0.005

    def test_different_alpha_power(self):
        """Works at alpha=0.01, power=0.95."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alpha=0.01)
        d = PowerDesign(power=0.95)

        n_res = required_sample_size(procedure=procedure, relative_lift=0.2, baseline=b, design=d)
        p_res = achieved_power(
            procedure=procedure, n_per_arm=n_res.n_per_arm, relative_lift=0.2, baseline=b, design=d
        )
        m_res = minimum_detectable_effect(
            procedure=procedure, n_per_arm=n_res.n_per_arm, baseline=b, design=d
        )

        assert abs(p_res.power - 0.95) <= 0.01
        assert abs(available_mde(m_res) - 0.20) <= 0.005

    def test_one_sided_less_round_trip(self):
        """required_sample_size/minimum_detectable_effect honor sign for alternative='less'.

        Regression test: the internal non-centrality parameter must stay signed
        (not abs()) or a one-sided 'less' design with a negative relative_lift
        reports power collapsing toward 0 instead of the target.
        """
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="less")
        d = PowerDesign(power=0.8)

        n_res = required_sample_size(procedure=procedure, relative_lift=-0.2, baseline=b, design=d)
        assert abs(n_res.power - 0.80) <= 0.01, (
            f"required_sample_size.power for alternative='less' expected ~0.80, got {n_res.power:.4f}"
        )

        p_res = achieved_power(
            procedure=procedure, n_per_arm=n_res.n_per_arm, relative_lift=-0.2, baseline=b, design=d
        )
        m_res = minimum_detectable_effect(
            procedure=procedure, n_per_arm=n_res.n_per_arm, baseline=b, design=d
        )

        assert abs(p_res.power - 0.80) <= 0.01
        assert available_mde(m_res) < 0, "MDE for alternative='less' must be signed negative"
        assert abs(m_res.power - 0.80) <= 0.01, (
            f"minimum_detectable_effect.power for alternative='less' expected ~0.80, "
            f"got {m_res.power:.4f}"
        )

    def test_imbalanced_allocation_round_trip(self):
        """achieved_power reproduces design.power at allocation != 0.5.

        Regression test: required_sample_size and achieved_power/
        minimum_detectable_effect must derive n_C from n_T the SAME way
        (via _compute_arms), or the round-trip drifts away from
        design.power whenever allocation is imbalanced.
        """
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8, allocation=0.8)

        n_res = required_sample_size(procedure=procedure, relative_lift=0.2, baseline=b, design=d)
        p_res = achieved_power(
            procedure=procedure, n_per_arm=n_res.n_per_arm, relative_lift=0.2, baseline=b, design=d
        )
        m_res = minimum_detectable_effect(
            procedure=procedure, n_per_arm=n_res.n_per_arm, baseline=b, design=d
        )

        assert abs(p_res.power - d.power) <= 0.01, (
            f"Imbalanced power round-trip failed: expected {d.power:.4f}, got {p_res.power:.4f}"
        )
        assert abs(available_mde(m_res) - 0.20) <= 0.005


# Non-inferiority (shifted null): Design.null_lift


class TestNonInferiority:
    """Design.null_lift shifts the noncentrality; mde_relative stays the
    distance BEYOND the null, but the variance is evaluated at the
    ABSOLUTE alternative that distance implies, so shifting the null moves
    the detectable distance and only an achieved-power inversion is the
    right check."""

    def test_null_lift_zero_reproduces_every_solver_bit_identically(self):
        """The regression guard: default null_lift=0.0 must not perturb
        any existing output, in any solver."""
        b = Baseline(mean=1.0, var=2.0)
        procedure_default = make_procedure(alpha=0.05)
        d_default = PowerDesign(power=0.8)
        procedure_explicit = make_procedure(alpha=0.05, null_lift=0.0)
        d_explicit = PowerDesign(power=0.8)

        n_default = required_sample_size(
            procedure=procedure_default, relative_lift=0.2, baseline=b, design=d_default
        )
        n_explicit = required_sample_size(
            procedure=procedure_explicit, relative_lift=0.2, baseline=b, design=d_explicit
        )
        assert n_default.n_per_arm == n_explicit.n_per_arm
        assert n_default.power == pytest.approx(n_explicit.power)
        assert n_default.mde_relative == pytest.approx(n_explicit.mde_relative)

        p_default = achieved_power(
            procedure=procedure_default,
            n_per_arm=500,
            relative_lift=0.2,
            baseline=b,
            design=d_default,
        )
        p_explicit = achieved_power(
            procedure=procedure_explicit,
            n_per_arm=500,
            relative_lift=0.2,
            baseline=b,
            design=d_explicit,
        )
        assert p_default.power == pytest.approx(p_explicit.power)

        m_default = minimum_detectable_effect(
            procedure=procedure_default, n_per_arm=500, baseline=b, design=d_default
        )
        m_explicit = minimum_detectable_effect(
            procedure=procedure_explicit, n_per_arm=500, baseline=b, design=d_explicit
        )
        assert m_default.mde_relative == pytest.approx(m_explicit.mde_relative)

    def test_required_sample_size_sizes_a_guardrail_at_true_lift_zero(self):
        """The canonical non-inferiority sizing question: 'how many units
        to confirm we didn't lose more than 1%, if the true effect is
        actually zero?' relative_lift=0.0 with a shifted null and a
        one-sided 'greater' test must return a FINITE n (not the
        zero-lift degenerate sentinel a bare relative_lift=0.0 vs null=0
        would hit)."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="greater", null_lift=-0.01)
        d = PowerDesign(power=0.8)
        result = required_sample_size(procedure=procedure, relative_lift=0.0, baseline=b, design=d)
        assert result.n_per_arm > 2  # not the degenerate n=2 sentinel
        assert abs(result.power - 0.80) <= 0.01

    def test_ni_sizing_duality_round_trip(self):
        """achieved_power at the NI-sized n reproduces design.power, same
        duality guarantee as the zero-null case."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="greater", null_lift=-0.01)
        d = PowerDesign(power=0.8)
        n_res = required_sample_size(procedure=procedure, relative_lift=0.0, baseline=b, design=d)
        p_res = achieved_power(
            procedure=procedure, n_per_arm=n_res.n_per_arm, relative_lift=0.0, baseline=b, design=d
        )
        assert abs(p_res.power - d.power) <= 0.01

    @pytest.mark.parametrize("null_lift", [-0.5, -0.05, 0.5])
    def test_mde_distance_beyond_null_inverts_to_target_power(self, null_lift: float):
        """mde_relative measures how far beyond null_lift is detectable, NOT
        an absolute effect size: composing the null with that distance gives
        the absolute alternative whose achieved power is the target, both
        through the public solver and the independent own-arm formula."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="greater", null_lift=null_lift)
        d = PowerDesign(power=0.8)
        m_res = minimum_detectable_effect(procedure=procedure, n_per_arm=500, baseline=b, design=d)
        assert m_res.mde_relative is not None
        detected_relative_lift = math.expm1(math.log1p(null_lift) + math.log1p(m_res.mde_relative))
        assert m_res.power == pytest.approx(d.power, abs=1e-9)
        assert _independent_h1_power(500, 500, b, procedure, detected_relative_lift) == (
            pytest.approx(d.power, abs=1e-9)
        )
        p_res = achieved_power(
            procedure=procedure,
            n_per_arm=500,
            relative_lift=detected_relative_lift,
            baseline=b,
            design=d,
        )
        assert p_res.power == pytest.approx(d.power, abs=1e-9)

    def test_mde_relative_recomposes_against_declared_null(self):
        """mde_relative is relative to the declared null (docstring),
        not the raw log-scale distance and not an absolute effect: at
        compliance=1.0, log1p(mde_relative) equals the search's own
        distance beyond the null, and composing it against the null via
        ``expm1(log1p(null_lift) + log1p(mde_relative))`` -- not adding
        lifts -- recovers the absolute alternative whose achieved power
        reproduces the design's target power."""
        b = Baseline(mean=1.0, var=4.0)
        null_lift = 0.10
        procedure = make_procedure(null_lift=null_lift)
        d = PowerDesign()
        result = minimum_detectable_effect(20000, b, procedure=procedure, design=d)
        assert result.mde_relative is not None
        distance = math.log1p(result.mde_relative)
        detected_relative_lift = math.expm1(math.log1p(null_lift) + distance)
        assert detected_relative_lift == pytest.approx(
            _implied_complier_lift(
                result.mde_relative, null_lift=null_lift, compliance=b.compliance
            )
        )
        recovered_distance = math.log1p(detected_relative_lift) - math.log1p(null_lift)
        assert recovered_distance == pytest.approx(distance)
        p_res = achieved_power(
            procedure=procedure,
            n_per_arm=20000,
            relative_lift=detected_relative_lift,
            baseline=b,
            design=d,
        )
        assert p_res.power == pytest.approx(d.power, abs=1e-9)

    def test_mde_relative_recomposes_against_declared_null_at_partial_compliance(self):
        """At compliance < 1.0 the recomposition formula must also divide
        the composed absolute alternative by compliance -- not just
        multiply distance by it going in -- for the round-trip through
        achieved_power to reproduce the design's target power."""
        b = Baseline(mean=1.0, var=4.0, compliance=0.6)
        null_lift = 0.10
        procedure = make_procedure(null_lift=null_lift)
        d = PowerDesign()
        result = minimum_detectable_effect(20000, b, procedure=procedure, design=d)
        assert result.mde_relative is not None
        detected_relative_lift = _implied_complier_lift(
            result.mde_relative, null_lift=null_lift, compliance=b.compliance
        )
        p_res = achieved_power(
            procedure=procedure,
            n_per_arm=20000,
            relative_lift=detected_relative_lift,
            baseline=b,
            design=d,
        )
        assert p_res.power == pytest.approx(d.power, abs=1e-9)

    def test_shifting_the_null_moves_the_detectable_distance(self):
        """The same distance beyond a lower null lands on a smaller absolute
        mean, whose log-scale variance is larger, so the detectable distance
        grows as the null moves down."""
        b = Baseline(mean=1.0, var=2.0)
        d = PowerDesign(power=0.8)
        distances = [
            available_mde(
                minimum_detectable_effect(
                    procedure=make_procedure(
                        alpha=0.05, alternative="greater", null_lift=null_lift
                    ),
                    n_per_arm=500,
                    baseline=b,
                    design=d,
                )
            )
            for null_lift in (-0.5, 0.0, 0.5)
        ]
        assert distances[0] > distances[1] > distances[2]

    def test_higher_null_lift_requires_more_units(self):
        """A tighter (less negative) guardrail tolerance sits closer to
        the true effect - harder to confirm, more units needed."""
        b = Baseline(mean=1.0, var=2.0)
        procedure_loose = make_procedure(alpha=0.05, alternative="greater", null_lift=-0.05)
        d_loose = PowerDesign(power=0.8)
        procedure_tight = make_procedure(alpha=0.05, alternative="greater", null_lift=-0.01)
        d_tight = PowerDesign(power=0.8)
        n_loose = required_sample_size(
            procedure=procedure_loose, relative_lift=0.0, baseline=b, design=d_loose
        ).n_per_arm
        n_tight = required_sample_size(
            procedure=procedure_tight, relative_lift=0.0, baseline=b, design=d_tight
        ).n_per_arm
        assert n_tight > n_loose

    def test_sequential_sizing_inverts_at_the_shifted_null(self):
        """Sequential sizing measures drift FROM the null with the variance
        of the ABSOLUTE alternative: achieved power at the sized n reproduces
        the target, and the equal-distance zero-null design is no longer an
        identical plan because its alternative sits at a different mean."""
        from increment.estimation.sequential import GaussianScoreMixture

        b = Baseline(mean=1.0, var=2.0)
        null_lift = -0.01
        lift = 0.10
        procedure_shifted = make_procedure(
            alpha=0.05,
            null_lift=null_lift,
            inference=GaussianScoreMixture(),
            population="assigned",
            decision_method="unadjusted",
        )
        d = PowerDesign(power=0.8)
        r_shifted = required_sample_size(
            procedure=procedure_shifted, relative_lift=lift, baseline=b, design=d
        )
        assert r_shifted.power >= d.power
        assert r_shifted.power == pytest.approx(d.power, abs=0.01)
        p_shifted = achieved_power(
            procedure=procedure_shifted,
            n_per_arm=r_shifted.n_per_arm,
            relative_lift=lift,
            baseline=b,
            design=d,
        )
        assert p_shifted.power == pytest.approx(r_shifted.power)

    def test_sequential_non_inferiority_guardrail_is_sizable(self):
        """The non-inferiority planning question - zero true lift against
        a -1% guardrail, monitored sequentially - returns a finite n at
        the target power."""
        from increment.estimation.sequential import GaussianScoreMixture

        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(
            alpha=0.05,
            alternative="greater",
            null_lift=-0.01,
            inference=GaussianScoreMixture(),
            population="assigned",
            decision_method="unadjusted",
        )
        d = PowerDesign(power=0.8)
        result = required_sample_size(procedure=procedure, relative_lift=0.0, baseline=b, design=d)
        assert result.n_per_arm > 2
        assert result.power == pytest.approx(0.80, abs=0.01)

    def test_sequential_achieved_power_round_trips_shifted_null(self):
        """achieved_power at the sized n reproduces the target power under
        a shifted null with sequential inference."""
        from increment.estimation.sequential import GaussianScoreMixture

        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(
            alpha=0.05,
            alternative="greater",
            null_lift=-0.01,
            inference=GaussianScoreMixture(),
            population="assigned",
            decision_method="unadjusted",
        )
        d = PowerDesign(power=0.8)
        n_res = required_sample_size(procedure=procedure, relative_lift=0.0, baseline=b, design=d)
        p_res = achieved_power(
            procedure=procedure, n_per_arm=n_res.n_per_arm, relative_lift=0.0, baseline=b, design=d
        )
        assert p_res.power == pytest.approx(d.power, abs=0.01)

    def test_sequential_mde_accepts_shifted_null_and_evaluates_its_own_variance(self):
        """minimum_detectable_effect under sequential inference accepts a
        shifted null. The distance beyond the null is no longer null-shift
        invariant: a lower null puts the alternative at a smaller mean,
        where the log-scale variance is larger, so detecting the same
        power needs a larger distance -- and achieved power at the answer
        reproduces the target."""
        from increment.estimation.sequential import GaussianScoreMixture

        b = Baseline(mean=1.0, var=2.0)
        procedure_zero = make_procedure(
            alpha=0.05,
            alternative="greater",
            inference=GaussianScoreMixture(),
            population="assigned",
            decision_method="unadjusted",
        )
        procedure_shifted = make_procedure(
            alpha=0.05,
            alternative="greater",
            null_lift=-0.05,
            inference=GaussianScoreMixture(),
            population="assigned",
            decision_method="unadjusted",
        )
        d = PowerDesign(power=0.8)
        m_zero = minimum_detectable_effect(
            procedure=procedure_zero, n_per_arm=500, baseline=b, design=d
        )
        m_shifted = minimum_detectable_effect(
            procedure=procedure_shifted, n_per_arm=500, baseline=b, design=d
        )
        m_zero_value = available_mde(m_zero)
        m_shifted_value = available_mde(m_shifted)
        assert math.log1p(m_shifted_value) > math.log1p(m_zero_value)
        detected_relative_lift = math.expm1(math.log1p(-0.05) + math.log1p(m_shifted_value))
        check = achieved_power(
            procedure=procedure_shifted,
            n_per_arm=500,
            relative_lift=detected_relative_lift,
            baseline=b,
            design=d,
        )
        assert check.power == pytest.approx(m_shifted.power, abs=1e-12)
        assert m_shifted.power == pytest.approx(0.8, abs=1e-9)

    @pytest.mark.slow
    def test_sequential_with_null_lift_zero_is_unaffected(self):
        """The refusal is keyed on a NONzero null_lift - the default
        (0.0) sequential path must work exactly as before."""
        from increment.estimation.sequential import GaussianScoreMixture

        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(
            alpha=0.05,
            null_lift=0.0,
            inference=GaussianScoreMixture(),
            population="assigned",
            decision_method="unadjusted",
        )
        d = PowerDesign(power=0.8)
        result = required_sample_size(procedure=procedure, relative_lift=0.2, baseline=b, design=d)
        assert result.n_per_arm > 0


# Step 3: CUPED test


class TestCuped:
    """CUPED variance reduction propagates correctly."""

    def test_rho_reduces_sample_size(self):
        """cuped_rho=0.6 reduces required n to ~(1-0.6^2) = 0.64x."""
        b0 = Baseline(mean=1.0, var=2.0, cuped_rho=0.0)
        b1 = Baseline(mean=1.0, var=2.0, cuped_rho=0.6)

        n0 = required_sample_size(
            procedure=make_procedure(decision_method="cuped"), relative_lift=0.2, baseline=b0
        ).n_per_arm
        n1 = required_sample_size(
            procedure=make_procedure(decision_method="cuped"), relative_lift=0.2, baseline=b1
        ).n_per_arm

        # Expected ratio approx (1 - 0.6^2) = 0.64, small integer-rounding noise
        assert n1 <= n0, "CUPED should reduce required N"
        ratio = n1 / n0
        assert 0.62 <= ratio <= 0.66, f"Expected ratio ~0.64, got {ratio:.4f}"

    def test_effective_var_reflects_cuped(self):
        """effective_var = var * (1 - rho^2)."""
        b = Baseline(mean=1.0, var=2.0, cuped_rho=0.6)
        result = required_sample_size(
            procedure=make_procedure(decision_method="cuped"), relative_lift=0.2, baseline=b
        )
        expected_effective_var = 2.0 * (1 - 0.6**2)  # = 1.28
        assert abs(result.effective_var - expected_effective_var) < 1e-10

        result2 = achieved_power(
            procedure=make_procedure(decision_method="cuped"),
            n_per_arm=500,
            relative_lift=0.2,
            baseline=b,
        )
        assert abs(result2.effective_var - expected_effective_var) < 1e-10

    def test_two_sided_vs_one_sided(self):
        """One-sided test requires smaller n than two-sided (same alpha)."""
        b = Baseline(mean=1.0, var=2.0)
        procedure_two = make_procedure(alternative="two-sided")
        d_two = PowerDesign()
        procedure_one = make_procedure(alternative="greater")
        d_one = PowerDesign()

        n_two = required_sample_size(
            procedure=procedure_two, relative_lift=0.2, baseline=b, design=d_two
        ).n_per_arm
        n_one = required_sample_size(
            procedure=procedure_one, relative_lift=0.2, baseline=b, design=d_one
        ).n_per_arm
        assert n_one < n_two, "One-sided should need fewer units than two-sided"

    def test_bonferroni_increases_n(self):
        """Bonferroni correction (n_variants=3) increases required n."""
        b = Baseline(mean=1.0, var=2.0)
        procedure_none = make_procedure(decision_method="cuped")
        d_none = PowerDesign()
        procedure_bonf = make_procedure(family_size=3)
        d_bonf = PowerDesign()

        n_none = required_sample_size(
            procedure=procedure_none, relative_lift=0.2, baseline=b, design=d_none
        ).n_per_arm
        n_bonf = required_sample_size(
            procedure=procedure_bonf, relative_lift=0.2, baseline=b, design=d_bonf
        ).n_per_arm
        assert n_bonf > n_none, "Bonferroni should need more units"

    def test_bonferroni_ratio(self):
        """Bonferroni n_variants=3 vs none - verify the increase is substantial."""
        b = Baseline(mean=1.0, var=1.0)
        procedure0 = make_procedure(decision_method="cuped")
        d0 = PowerDesign()
        procedure3 = make_procedure(family_size=3)
        d3 = PowerDesign()

        n0 = required_sample_size(
            procedure=procedure0, relative_lift=0.1, baseline=b, design=d0
        ).n_per_arm
        n3 = required_sample_size(
            procedure=procedure3, relative_lift=0.1, baseline=b, design=d3
        ).n_per_arm
        # Bonferroni alpha' = 0.05/3 = 0.01667, z_two_side = ppf(1-0.01667/2) = ~2.39 vs 1.96
        # The ratio should be roughly (2.39+0.84)^2/(1.96+0.84)^2 ~= 1.34
        ratio = n3 / n0
        assert 1.25 <= ratio <= 1.45, f"Bonferroni ratio ~1.34, got {ratio:.4f}"


# Edge cases


class TestEdgeCases:
    """Boundary conditions and validation."""

    def test_zero_var_raises(self):
        """Baseline with negative var."""
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=1.0, var=-1.0)
        assert exc_info.value.code == "power.baseline.var_zero_variance"

    def test_invalid_cuped_rho(self):
        """cuped_rho must be in (-1, 1)."""
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=1.0, var=1.0, cuped_rho=1.0)
        assert exc_info.value.code == "power.baseline.cuped_rho"
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=1.0, var=1.0, cuped_rho=-1.0)
        assert exc_info.value.code == "power.baseline.cuped_rho"

    def test_baseline_rejects_nonpositive_mean(self):
        """Baseline mean must be > 0 (solvers divide by mean**2)."""
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=0.0, var=1.0)
        assert exc_info.value.code == "power.baseline.mean"
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=-1.0, var=1.0)
        assert exc_info.value.code == "power.baseline.mean"
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_proportion(p=0.0)
        assert exc_info.value.code == "power.baseline.mean"

    def test_relative_lift_at_or_below_negative_one_rejected(self):
        """relative_lift <= -1.0 is out of domain for log1p in every solver."""
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                procedure=make_procedure(decision_method="cuped"), relative_lift=-1.0, baseline=b
            )
        assert exc_info.value.code == "power.require_relative_domain"
        with pytest.raises(InvalidRequestError) as exc_info:
            achieved_power(
                procedure=make_procedure(decision_method="cuped"),
                n_per_arm=100,
                relative_lift=-1.0,
                baseline=b,
            )
        assert exc_info.value.code == "power.require_relative_domain"

    @pytest.mark.parametrize("field", ["mean", "var", "avg_cluster_size", "cluster_size_cv"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_baseline_rejects_nonfinite_fields(self, field, bad):
        """NaN/inf survive the existing one-sided bound checks (a
        comparison against NaN is always False); every numeric field
        must be explicitly required to be finite."""
        kwargs: dict[str, float] = {"mean": 1.0, "var": 1.0}
        kwargs[field] = bad
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(**kwargs)
        assert exc_info.value.code == "power.finite"
        assert exc_info.value.context["name"] == field

    def test_relative_lift_rejects_nonfinite(self):
        """A NaN/inf relative_lift <= -1.0 comparison is always False, so
        only an explicit finiteness check catches it."""
        b = Baseline(mean=1.0, var=1.0)
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(InvalidRequestError) as exc_info:
                achieved_power(
                    procedure=make_procedure(decision_method="cuped"),
                    n_per_arm=100,
                    relative_lift=bad,
                    baseline=b,
                )
            assert exc_info.value.code == "power.finite"

    @pytest.mark.parametrize(
        ("solver", "kwargs"),
        [
            pytest.param(
                required_sample_size,
                {"relative_lift": 0.05},
                id="required-sample-size",
            ),
            pytest.param(
                achieved_power,
                {"n_per_arm": 100, "relative_lift": 0.05},
                id="achieved-power",
            ),
            pytest.param(
                minimum_detectable_effect,
                {"n_per_arm": 100},
                id="minimum-detectable-effect",
            ),
        ],
    )
    @pytest.mark.parametrize(
        ("planned_looks", "code"),
        [
            pytest.param(True, "power.planned_looks_positive", id="bool"),
            pytest.param(2.5, "power.planned_looks_positive", id="float"),
            pytest.param(0, "power.planned_looks", id="zero"),
            pytest.param(-1, "power.planned_looks", id="negative"),
        ],
    )
    def test_power_solvers_reject_invalid_planned_looks(
        self,
        solver: Any,
        kwargs: dict[str, Any],
        planned_looks: Any,
        code: str,
    ):
        baseline = Baseline(mean=1.0, var=1.0)

        with pytest.raises(InvalidRequestError) as exc_info:
            solver(
                baseline=baseline,
                procedure=make_procedure(decision_method="cuped"),
                planned_looks=planned_looks,
                **kwargs,
            )
        assert exc_info.value.code == code
        assert exc_info.value.context["planned_looks"] == planned_looks

    def test_planned_looks_one_is_valid(self):
        result = achieved_power(
            procedure=make_procedure(decision_method="cuped"),
            n_per_arm=100,
            relative_lift=0.05,
            baseline=Baseline(mean=1.0, var=1.0),
            planned_looks=1,
        )

        assert 0.0 < result.power < 1.0

    def test_from_summary(self):
        """Baseline.from_summary extracts mean verbatim and inflates var
        to its own sampling-uncertainty upper confidence bound (matches
        the closed-form chi-square formula)."""
        s = SummaryStats(n=100, mean=0.5, var=0.25)
        b = Baseline.from_summary(s, cuped_rho=0.3)
        assert b.mean == 0.5
        assert b.cuped_rho == 0.3
        df = 99
        expected_var = 0.25 * df / scipy_chi2.ppf(0.20, df)
        assert b.var == pytest.approx(expected_var, rel=1e-12)
        assert b.var > 0.25  # inflated, not passed through unchanged

    def test_from_summary_inflation_vanishes_at_large_n(self):
        """A 2M-unit pilot's own variance estimate is essentially exact:
        inflation must shrink to a rounding-level nudge."""
        s = SummaryStats(n=2_000_000, mean=0.5, var=0.25)
        b = Baseline.from_summary(s)
        assert b.var == pytest.approx(0.25, rel=1e-3)

    def test_from_summary_small_pilot_inflates_more_than_large_pilot(self):
        """A 20-unit pilot and a 2M-unit pilot reporting the identical
        point-estimate var must NOT size/power identically -- the small
        pilot's own variance estimate is far less trustworthy."""
        small = Baseline.from_summary(SummaryStats(n=20, mean=0.5, var=0.25))
        large = Baseline.from_summary(SummaryStats(n=2_000_000, mean=0.5, var=0.25))
        assert small.var > large.var

    def test_from_summary_confidence_is_configurable(self):
        s = SummaryStats(n=20, mean=0.5, var=0.25)
        loose = Baseline.from_summary(s, confidence=0.50)
        tight = Baseline.from_summary(s, confidence=0.95)
        assert loose.var < tight.var  # a higher confidence bound inflates more

    def test_from_summary_extreme_confidence_stays_finite(self):
        """Confidence is passed straight into chi2.isf, never derived as
        1 - confidence, so an extreme confidence level near 1.0 stays
        exact instead of rounding its complement to zero."""
        s = SummaryStats(n=20, mean=0.5, var=0.25)
        extreme = Baseline.from_summary(s, confidence=1.0 - 1e-12)
        assert math.isfinite(extreme.var)
        assert extreme.var > Baseline.from_summary(s, confidence=0.9999).var

    @pytest.mark.parametrize("confidence", [1.0, 2.0, -1.0, 0.0])
    def test_from_summary_rejects_invalid_confidence(self, confidence):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_summary(SummaryStats(n=20, mean=0.5, var=0.25), confidence=confidence)
        assert exc_info.value.code == "power.baseline.confidence"

    def test_from_summary_refuses_n_below_two(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_summary(SummaryStats(n=1, mean=0.5, var=0.25))
        assert exc_info.value.code == "power.baseline.summarystats_bound_pilot"
        assert exc_info.value.context["n"] == 1

    def test_from_summary_refuses_a_confidence_that_overflows_the_inflation(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_summary(SummaryStats(n=3, mean=0.5, var=1e308), confidence=0.5)
        assert exc_info.value.code == "power.baseline.confidence_produces_non"

    def test_from_ratio_matches_linearized_formula(self):
        """Baseline.from_ratio(...) == Var(y - R*den) / mean_den**2 (0svs),
        NOT the numerator's bare variance."""
        b = Baseline.from_ratio(
            mean_num=10.0, mean_den=5.0, var_num=20.0, var_den=3.0, cov_num_den=2.0
        )
        r = 10.0 / 5.0
        assert b.mean == pytest.approx(r)
        expected_var = (20.0 + r * r * 3.0 - 2.0 * r * 2.0) / 5.0**2
        assert b.var == pytest.approx(expected_var, rel=1e-12)
        # The numerator-only variance a naive plug-in would use is a
        # different number: from_ratio must not silently reduce to it.
        assert b.var != pytest.approx(20.0 / 5.0**2)

    def test_from_ratio_zero_denominator_variance_reduces_to_scaled_numerator(self):
        """A constant denominator (var_den=cov_num_den=0) collapses the
        ratio to a rescaled numerator: var = var_num / mean_den**2."""
        b = Baseline.from_ratio(
            mean_num=10.0, mean_den=5.0, var_num=20.0, var_den=0.0, cov_num_den=0.0
        )
        assert b.var == pytest.approx(20.0 / 5.0**2, rel=1e-12)

    def test_from_ratio_refuses_nonpositive_denominator_mean(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_ratio(
                mean_num=10.0, mean_den=0.0, var_num=20.0, var_den=3.0, cov_num_den=2.0
            )
        assert exc_info.value.code == "power.baseline.mean_den"
        assert exc_info.value.context["mean_den"] == 0.0

    def test_effective_var_edge(self):
        """effective_var = var when rho=0 gives raw var."""
        b = Baseline(mean=1.0, var=2.0, cuped_rho=0.0)
        assert b.effective_var == 2.0

    def test_n_per_arm_ceiling(self):
        """required_sample_size returns integer at least 2."""
        b = Baseline(mean=100.0, var=1.0)  # very low variance -> small n
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8)
        result = required_sample_size(procedure=procedure, relative_lift=0.01, baseline=b, design=d)
        assert isinstance(result.n_per_arm, int)
        assert result.n_per_arm >= 2

    def test_allocation_effect(self):
        """Imbalanced allocation increases total N."""
        b = Baseline(mean=1.0, var=2.0)
        procedure_balanced = make_procedure(decision_method="cuped")
        d_balanced = PowerDesign(allocation=0.5)
        procedure_imbalanced = make_procedure(decision_method="cuped")
        d_imbalanced = PowerDesign(allocation=0.8)

        n_bal = required_sample_size(
            procedure=procedure_balanced, relative_lift=0.2, baseline=b, design=d_balanced
        ).n_per_arm
        n_imb = required_sample_size(
            procedure=procedure_imbalanced, relative_lift=0.2, baseline=b, design=d_imbalanced
        ).n_per_arm
        assert n_imb != n_bal, "Allocation should affect required n"


class TestPowerResultInvariants:
    """PowerResult is the single choke point every solver returns
    through; its closing invariant (finite power in [0, 1], an available
    mde_relative above the -1.0 floor paired with no reason or a null one
    paired with a reason, finite positive effective_var) catches
    nonfinite/out-of-domain results regardless of which solver path
    produced them."""

    @staticmethod
    def _result(**overrides: Any) -> PowerResult:
        kwargs: dict[str, Any] = {
            "n_per_arm": 100,
            "n_total": 200,
            "power": 0.8,
            "power_basis": "asymptotic",
            "mde_relative": 0.1,
            "effective_var": 1.0,
        }
        kwargs.update(overrides)
        return PowerResult(**kwargs)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_nonfinite_power(self, bad):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._result(power=bad)
        assert exc_info.value.code == "power.finite"

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rejects_out_of_range_power(self, bad):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._result(power=bad)
        assert exc_info.value.code == "power.power.check"
        assert exc_info.value.context["power"] == bad

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -1.0, -2.0])
    def test_rejects_out_of_domain_mde_relative(self, bad):
        import math as _math

        with pytest.raises(InvalidRequestError) as exc_info:
            self._result(mde_relative=bad)
        if _math.isfinite(bad):
            assert exc_info.value.code == "power.require_relative_domain"
        else:
            assert exc_info.value.code == "power.finite"

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_nonfinite_effective_var(self, bad):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._result(effective_var=bad)
        assert exc_info.value.code == "power.finite"

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_rejects_nonpositive_effective_var(self, bad):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._result(effective_var=bad)
        assert exc_info.value.code == "power.power.effective_var"
        assert exc_info.value.context["effective_var"] == bad

    def test_accepts_valid_result(self):
        result = self._result()
        assert result.power == 0.8
        assert result.mde_unavailable_reason is None

    @pytest.mark.parametrize("reason", ["unattainable", "unrepresentable", "numerical_resolution"])
    def test_accepts_a_missing_mde_with_its_reason_and_round_trips(self, reason):
        result = self._result(mde_relative=None, mde_unavailable_reason=reason)
        assert result.mde_relative is None
        assert result.mde_unavailable_reason == reason
        assert PowerResult.model_validate_json(result.model_dump_json()) == result
        assert result.model_dump()["mde_relative"] is None

    @pytest.mark.parametrize(
        ("mde_relative", "reason", "code"),
        [
            (None, None, "power.mde_relative_unavailable"),
            (0.1, "unattainable", "power.mde_unavailable_reason"),
        ],
    )
    def test_rejects_invalid_value_reason_pairs(self, mde_relative, reason, code):
        with pytest.raises(InvalidRequestError) as exc_info:
            self._result(mde_relative=mde_relative, mde_unavailable_reason=reason)
        assert exc_info.value.code == code

    def test_rejects_an_unrecognized_mde_unavailable_reason(self):
        with pytest.raises(InvalidRequestError) as raised:
            self._result(mde_relative=None, mde_unavailable_reason="not_a_reason")
        assert raised.value.code == "model.field.literal"

    def test_minimum_detectable_effect_refuses_a_decrease_past_the_complier_floor(self):
        """A complier-scale decrease cannot pass -100%: with compliance 0.1
        the decreasing direction ends at an effective lift of -0.1, and at
        n=100 no admissible decrease reaches 80% power. The solver refuses
        with a physical limit rather than handing back a relative lift that
        achieved_power then rejects on its own output."""
        from increment.errors import InvalidRequestError

        b = Baseline(mean=1.0, var=2.0, compliance=0.1)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, b, make_procedure(alternative="less"), PowerDesign())
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "relative_lift_floor"
        assert refusal.value.context["direction"] == "decreasing"
        maximum = refusal.value.context["maximum_power"]
        assert isinstance(maximum, float)
        assert 0.0 < maximum < 0.8


class TestMDEDomainEdges:
    """A test's own null-boundary power (2*tail_alpha two-sided,
    tail_alpha one-sided) already exceeds any target power at or below
    it: the zero-distance effect trivially clears the target, so a
    fixed-horizon target at or below that floor is refused rather than
    silently answered with a zero-distance effect -- stable code/context
    parity with sequential.py's own
    ``power.sequential_mde.design_search_minimum``."""

    def test_two_sided_target_below_null_power_refuses(self):
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, b, make_procedure(), PowerDesign(power=0.01))
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"
        assert refusal.value.context["target"] == 0.01

    def test_two_sided_target_at_null_power_refuses(self):
        """A target exactly at the null's own crossing probability (not
        merely below it) also refuses -- the crossing probability is read
        from the refusal's own context (a trivially low target already
        refuses) rather than a literal 0.05, to sidestep float noise in
        the two independent norm.sf tail evaluations that compute it."""
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as probe:
            minimum_detectable_effect(100, b, make_procedure(), PowerDesign(power=1e-9))
        null_power = probe.value.context["estimate"]
        assert isinstance(null_power, float)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, b, make_procedure(), PowerDesign(power=null_power))
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"
        assert refusal.value.context["estimate"] == pytest.approx(null_power)
        assert refusal.value.context["target"] == null_power

    def test_one_sided_greater_target_below_null_power_refuses(self):
        """Previously returned a NEGATIVE MDE for a 'greater' alternative
        -- the wrong side entirely -- instead of refusing outright."""
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                100, b, make_procedure(alternative="greater"), PowerDesign(power=0.01)
            )
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"

    def test_one_sided_less_refusal_explains_the_nonzero_minimum(self):
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                100, b, make_procedure(alternative="less"), PowerDesign(power=0.01)
            )
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"

    def test_achieved_power_keeps_zero_companion_at_low_target(self):
        result = achieved_power(
            100,
            0.1,
            Baseline(mean=1.0, var=2.0),
            make_procedure(),
            PowerDesign(power=0.01),
        )
        assert result.power > 0.01
        assert result.mde_relative == 0.0
        assert result.mde_unavailable_reason is None

    def test_required_sample_size_keeps_zero_companion_at_low_target(self):
        result = required_sample_size(
            0.1,
            Baseline(mean=1.0, var=2.0),
            make_procedure(),
            PowerDesign(power=0.01),
        )
        assert result.power >= 0.01
        assert result.mde_relative == 0.0
        assert result.mde_unavailable_reason is None

    def test_minimum_detectable_effect_refuses_at_or_below_null_power(self):
        """Stable refusal code/context parity with the sequential twin
        (``power.sequential_mde.design_search_minimum``): both name the
        crossing estimate and the requested target, and neither silently
        answers a zero-distance effect."""
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                1000,
                Baseline(mean=1.0, var=1.0),
                make_procedure(alternative="two-sided"),
                PowerDesign(power=0.03),
            )
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"
        assert refusal.value.context["target"] == 0.03
        assert refusal.value.context["estimate"] == pytest.approx(0.04999999999999996)

    def test_clustered_root_solver_handles_endpoint_instead_of_crashing(self):
        """Clustered pairwise-free (primary) solver: a target power at
        the null level previously reached brentq with no sign change and
        raised ValueError: f(a) and f(b) must have different signs. The
        arm solver now refuses cleanly instead of crashing or clamping."""
        b = Baseline(mean=1.0, var=2.0, cluster_icc=0.05, avg_cluster_size=10.0)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(
                100,
                b,
                make_procedure(alternative="greater", dependence="cluster"),
                PowerDesign(power=0.05),
            )
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"


# Achieved power edge cases


class TestAchievedPower:
    """achieved_power produces sensible values."""

    def test_infinite_n_approaches_one(self):
        """Very large n gives power close to 1.0."""
        b = Baseline(mean=1.0, var=2.0)
        result = achieved_power(
            procedure=make_procedure(decision_method="cuped"),
            n_per_arm=10000000,
            relative_lift=0.01,
            baseline=b,
        )
        assert result.power > 0.999

    def test_extreme_alpha_stays_finite_via_survival_function(self):
        """z_alpha comes from norm.isf(tail_alpha), never norm.ppf(1 -
        tail_alpha); an alpha far below float precision's 1-x floor must
        still resolve to a finite, sane (near-zero) power."""
        b = Baseline(mean=1.0, var=2.0)
        result = achieved_power(
            n_per_arm=1000,
            relative_lift=0.1,
            baseline=b,
            procedure=make_procedure(alpha=1e-300),
        )
        assert math.isfinite(result.power)
        assert 0.0 <= result.power < 1e-100

    def test_zero_lift_gives_alpha(self):
        """Zero relative lift gives power ~= alpha (test size)."""
        b = Baseline(mean=1.0, var=1.0)
        result = achieved_power(
            procedure=make_procedure(decision_method="cuped"),
            n_per_arm=1000,
            relative_lift=0.0,
            baseline=b,
        )
        # For zero lift, power = P(|N(0,1)| > z_alpha/2) = alpha
        assert abs(result.power - 0.05) < 0.001

    def test_one_sided_greater(self):
        """One-sided 'greater' test."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alternative="greater")
        d = PowerDesign()
        result = achieved_power(
            procedure=procedure, n_per_arm=500, relative_lift=0.1, baseline=b, design=d
        )
        assert 0.3 < result.power < 1.0

    def test_one_sided_less(self):
        """One-sided 'less' test with negative lift."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alternative="less")
        d = PowerDesign()
        result = achieved_power(
            procedure=procedure, n_per_arm=500, relative_lift=-0.1, baseline=b, design=d
        )
        assert 0.3 < result.power < 1.0

    def test_one_sided_less_no_effect(self):
        """One-sided 'less' with positive lift should have low power."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alternative="less")
        d = PowerDesign()
        result = achieved_power(
            procedure=procedure, n_per_arm=500, relative_lift=0.1, baseline=b, design=d
        )
        assert result.power < 0.10


# MDE edge cases


class TestMDE:
    """minimum_detectable_effect produces sensible values."""

    def test_mde_decreases_with_n(self):
        """MDE shrinks as sample size grows."""
        b = Baseline(mean=1.0, var=2.0)

        mde_small = available_mde(
            minimum_detectable_effect(
                procedure=make_procedure(decision_method="cuped"), n_per_arm=100, baseline=b
            )
        )
        mde_large = available_mde(
            minimum_detectable_effect(
                procedure=make_procedure(decision_method="cuped"), n_per_arm=10000, baseline=b
            )
        )
        assert mde_large < mde_small

    def test_mde_positive_two_sided(self):
        """Two-sided MDE is always positive."""
        b = Baseline(mean=1.0, var=2.0)
        result = minimum_detectable_effect(
            procedure=make_procedure(decision_method="cuped"), n_per_arm=500, baseline=b
        )
        assert available_mde(result) > 0

    def test_mde_signed_one_sided_greater(self):
        """One-sided 'greater' MDE is positive."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alternative="greater")
        d = PowerDesign()
        result = minimum_detectable_effect(procedure=procedure, n_per_arm=500, baseline=b, design=d)
        assert available_mde(result) > 0

    def test_mde_signed_one_sided_less(self):
        """One-sided 'less' MDE is negative."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alternative="less")
        d = PowerDesign()
        result = minimum_detectable_effect(procedure=procedure, n_per_arm=500, baseline=b, design=d)
        assert available_mde(result) < 0

    def test_cuped_reduces_mde(self):
        """CUPED reduces the detectable effect at same n."""
        b0 = Baseline(mean=1.0, var=2.0, cuped_rho=0.0)
        b1 = Baseline(mean=1.0, var=2.0, cuped_rho=0.6)

        mde0 = available_mde(
            minimum_detectable_effect(
                procedure=make_procedure(decision_method="cuped"), n_per_arm=500, baseline=b0
            )
        )
        mde1 = available_mde(
            minimum_detectable_effect(
                procedure=make_procedure(decision_method="cuped"), n_per_arm=500, baseline=b1
            )
        )
        assert mde1 < mde0, "CUPED should reduce MDE at same n"


class TestMDETwoSidedRootFind:
    """The FIXED-variance inverse ``_mde_theta`` (the segment-pairwise
    model's and the dimensionless quantile seed's) root-finds two-sided
    power against ``_power_at``'s exact two-tail power instead of a
    single-tail closed form. Its variance does not move with the
    alternative, so its distance is null-shift invariant; the arm trio's
    inverse is covered by the public-solver tests."""

    def test_two_sided_mde_achieves_target_power_exactly(self):
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="two-sided")
        d = PowerDesign(power=0.8)

        mde = available_mde(
            segment_pairwise_minimum_detectable_effect(
                500, 0.5, 0.5, b, procedure=procedure, design=d
            )
        )
        power = segment_pairwise_achieved_power(
            500, mde, 0.0, 0.5, 0.5, b, procedure=procedure, design=d
        ).power
        assert power == pytest.approx(d.power, abs=1e-6)

    def test_one_sided_closed_form_unchanged(self):
        """The one-sided segment-pairwise MDE is the exact single-tail closed
        form ``sqrt(se2) * (z_alpha + z_power)`` at the solver's own variance."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05, alternative="greater")
        d = PowerDesign(power=0.8)
        mde = available_mde(
            segment_pairwise_minimum_detectable_effect(
                500, 0.5, 0.5, b, procedure=procedure, design=d
            )
        )
        # Equal 50% shares at n_per_arm=500 split each segment's 500 units
        # 250/250, so the contrast variance is two own-arm variances.
        se2 = 2.0 * b.effective_var / b.mean**2 * (1 / 250 + 1 / 250)
        z_alpha = float(norm.isf(procedure.compiled_tail_alpha))
        z_power = float(norm.ppf(d.power))
        assert math.log1p(mde) == pytest.approx(math.sqrt(se2) * (z_alpha + z_power), rel=1e-12)

    def test_two_sided_mde_is_invariant_to_a_shifted_null(self):
        """Power depends only on nc = (theta - theta0) / se, so the two-sided
        detectable distance must not change when the null is shifted. The root
        finder evaluates power from nc directly rather than reconstructing an
        absolute theta around theta0 and subtracting it back off, which would
        lose the increment for a large null and a small standard error."""
        from increment.power.core import _mde_theta, _se_sq

        # se ~ 6e-15 << ulp(theta0) ~ 1.1e-13 at this near-maximum null, so a
        # theta0 + mde_theta reconstruction would round the increment away
        # entirely; the noncentrality path is unaffected and the distances match.
        se2 = _se_sq(500, 500, Baseline(mean=1.0, var=1e-26))
        d = PowerDesign(power=0.8)
        zero = _mde_theta(
            se2, d, make_procedure(alpha=0.05, alternative="two-sided", null_lift=0.0)
        )
        shifted = _mde_theta(
            se2, d, make_procedure(alpha=0.05, alternative="two-sided", null_lift=1e307)
        )
        assert shifted == zero

    def test_fixed_horizon_result_power_matches_target_under_a_shifted_null(self):
        """The fixed-horizon MDE result reports power at the solved effect via
        the noncentrality, not a null-reconstructed absolute effect, so the
        reported power equals the target even when the null is shifted."""
        from increment.power.core import minimum_detectable_effect

        # Near-maximum null with a tiny standard error: reconstructing
        # theta0 + mde_theta rounds the increment away and would report the
        # null's own power (~alpha) rather than the target.
        result = minimum_detectable_effect(
            procedure=make_procedure(alpha=0.05, alternative="two-sided", null_lift=1e307),
            n_per_arm=500,
            baseline=Baseline(mean=1.0, var=1e-26),
            design=PowerDesign(power=0.8),
        )
        assert result.power == pytest.approx(0.8, rel=1e-9)


# Compliance (encouragement designs)


class TestCompliance:
    """Baseline.compliance dilutes the detectable lift (ITT sizing at 1/c^2)."""

    def test_compliance_one_is_identity(self):
        """compliance=1.0 reproduces current outputs exactly (backcompat)."""
        b0 = Baseline(mean=10.0, var=4.0)
        b1 = Baseline(mean=10.0, var=4.0, compliance=1.0)
        r0 = required_sample_size(
            procedure=make_procedure(decision_method="cuped"),
            relative_lift=0.05,
            baseline=b0,
            design=PowerDesign(),
        )
        r1 = required_sample_size(
            procedure=make_procedure(decision_method="cuped"),
            relative_lift=0.05,
            baseline=b1,
            design=PowerDesign(),
        )
        assert r0.n_per_arm == r1.n_per_arm
        assert r0.mde_relative == r1.mde_relative

    def test_half_compliance_quadruples_n(self):
        """Halving compliance roughly quadruples required n (1/c^2)."""
        b = Baseline(mean=10.0, var=4.0, compliance=0.5)
        r = required_sample_size(
            procedure=make_procedure(decision_method="cuped"),
            relative_lift=0.02,
            baseline=b,
            design=PowerDesign(),
        )
        r_full = required_sample_size(
            procedure=make_procedure(decision_method="cuped"),
            relative_lift=0.02,
            baseline=Baseline(mean=10.0, var=4.0),
            design=PowerDesign(),
        )
        assert r.n_per_arm == pytest.approx(4 * r_full.n_per_arm, rel=0.02)

    def test_mde_scales_inverse_compliance(self):
        """Halving compliance doubles the detectable complier lift."""
        b = Baseline(mean=10.0, var=4.0, compliance=0.5)
        m = minimum_detectable_effect(
            procedure=make_procedure(decision_method="cuped"),
            n_per_arm=10000,
            baseline=b,
            design=PowerDesign(),
        )
        m_full = minimum_detectable_effect(
            procedure=make_procedure(decision_method="cuped"),
            n_per_arm=10000,
            baseline=Baseline(mean=10.0, var=4.0),
            design=PowerDesign(),
        )
        assert available_mde(m) == pytest.approx(2 * available_mde(m_full), rel=0.02)

    def test_compliance_bounds(self):
        """compliance must be in (0, 1]."""
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=1.0, var=1.0, compliance=0.0)
        assert exc_info.value.code == "power.baseline.compliance"
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=1.0, var=1.0, compliance=1.2)
        assert exc_info.value.code == "power.baseline.compliance"


# Step 4: Pairwise segment-difference power


def _baseline_only_ate_total(
    theta: float, baseline: Baseline, procedure: Any, design: PowerDesign
) -> float:
    """Independent closed form of the segment-pairwise model's building
    block: total N detecting log-contrast ``theta`` when EVERY arm is
    evaluated at its segment's baseline, ``k * v / m^2 * z_sum^2 / theta^2``
    with ``k = 1/a + 1/(1-a)``. The arm trio is not the oracle here: its
    treatment arm moves with the alternative, this model's does not."""
    a = design.allocation
    k = 1.0 / a + 1.0 / (1.0 - a)
    z_sum = float(norm.isf(procedure.compiled_tail_alpha)) + float(norm.ppf(design.power))
    return k * baseline.effective_var / baseline.mean**2 * z_sum**2 / theta**2


class TestSegmentPairwise:
    """n = n_ATE(delta) * (1/q_A + 1/q_B), taking (r_A, r_B) not a bare delta,
    on the retained baseline-only variance model."""

    # Small effect -> huge N, so integer-rounding noise in the ratio check
    # below is negligible (checked to abs=1e-3 against exact fractions).
    _R_A, _R_B = 0.001, 0.0

    def _n_ate_total(self, baseline: Baseline, procedure: Any, design: PowerDesign) -> float:
        theta = math.log1p(self._R_A) - math.log1p(self._R_B)
        return _baseline_only_ate_total(theta, baseline, procedure, design)

    @pytest.mark.parametrize(
        ("q_a", "q_b", "expected_multiplier", "label"),
        [
            # NON-PARTITION cases (q_a + q_b < 1): the old n/(q(1-q)) form and the
            # correct 1/q_a + 1/q_b form diverge only here - the only cases that catch the bug.
            (0.10, 0.10, 20.0, "non-partition"),
            (0.15, 0.05, 26.666666666666668, "non-partition"),
            # PARTITION regression (q_a + q_b == 1): 1/q_a + 1/q_b collapses to
            # 1/(q_a*q_b), matching the old formula - can't tell correct from buggy.
            (0.10, 0.90, 11.11111111111111, "partition regression"),
        ],
    )
    def test_pairwise_n_matches_1_over_qa_plus_1_over_qb(
        self, q_a, q_b, expected_multiplier, label
    ):
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8, allocation=0.5)
        n_ate = self._n_ate_total(b, procedure, d)
        pairwise = segment_pairwise_required_sample_size(
            self._R_A, self._R_B, q_a, q_b, b, procedure=procedure, design=d
        )
        ratio = pairwise.n_total / n_ate
        assert ratio == pytest.approx(expected_multiplier, rel=1e-3), label
        assert ratio == pytest.approx(1 / q_a + 1 / q_b, rel=1e-3)

    def test_baseline_b_defaults_to_baseline_a(self):
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure()
        d = PowerDesign()
        with_default = segment_pairwise_required_sample_size(
            0.2, 0.05, 0.3, 0.3, b, procedure=procedure, design=d
        )
        with_explicit = segment_pairwise_required_sample_size(
            0.2, 0.05, 0.3, 0.3, b, procedure=procedure, baseline_b=b, design=d
        )
        assert with_default == with_explicit

    @pytest.mark.parametrize("solver", ["required", "achieved", "mde"])
    @pytest.mark.parametrize("baseline_side", ["a", "b"])
    @pytest.mark.parametrize("field", ["compliance", "trigger_rate"])
    def test_pairwise_refuses_unsupported_baseline_fields(self, solver, baseline_side, field):
        unsupported = {field: 0.8}
        baseline_a = Baseline(
            mean=1.0,
            var=1.0,
            **(unsupported if baseline_side == "a" else {}),
        )
        baseline_b = Baseline(mean=1.0, var=1.0, **unsupported) if baseline_side == "b" else None

        with pytest.raises(InvalidRequestError) as exc_info:
            if solver == "required":
                segment_pairwise_required_sample_size(
                    0.3,
                    0.05,
                    0.2,
                    0.2,
                    baseline_a,
                    procedure=make_procedure(),
                    baseline_b=baseline_b,
                )
            elif solver == "achieved":
                segment_pairwise_achieved_power(
                    100,
                    0.3,
                    0.05,
                    0.2,
                    0.2,
                    baseline_a,
                    procedure=make_procedure(),
                    baseline_b=baseline_b,
                )
            else:
                segment_pairwise_minimum_detectable_effect(
                    100, 0.2, 0.2, baseline_a, procedure=make_procedure(), baseline_b=baseline_b
                )

        assert exc_info.value.code == "power.segment_pairwise_solvers"
        assert field in exc_info.value.context["unsupported"]  # ty: ignore[unsupported-operator]

    @pytest.mark.parametrize("solver", ["required", "achieved", "mde"])
    def test_pairwise_default_baselines_return_power_result(self, solver):
        from increment.power.core import PowerResult

        baseline = Baseline(mean=1.0, var=1.0)
        if solver == "required":
            result = segment_pairwise_required_sample_size(
                0.3, 0.05, 0.2, 0.2, baseline, procedure=make_procedure()
            )
        elif solver == "achieved":
            result = segment_pairwise_achieved_power(
                100, 0.3, 0.05, 0.2, 0.2, baseline, procedure=make_procedure()
            )
        else:
            result = segment_pairwise_minimum_detectable_effect(
                100, 0.2, 0.2, baseline, procedure=make_procedure()
            )

        assert isinstance(result, PowerResult)

    def test_power_round_trip(self):
        """achieved_power(required_sample_size(...).n_per_arm) ~= design.power."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8)
        n_res = segment_pairwise_required_sample_size(
            0.3, 0.05, 0.2, 0.2, b, procedure=procedure, design=d
        )
        p_res = segment_pairwise_achieved_power(
            n_res.n_per_arm, 0.3, 0.05, 0.2, 0.2, b, procedure=procedure, design=d
        )
        assert abs(p_res.power - d.power) <= 0.01, (
            f"Power round-trip failed: expected {d.power:.4f}, got {p_res.power:.4f}"
        )

    def test_mde_round_trip(self):
        """minimum_detectable_effect(n) reproduces the log-scale delta used to size it."""
        import math

        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8)
        r_a, r_b = 0.30, 0.05
        theta = math.log1p(r_a) - math.log1p(r_b)

        n_res = segment_pairwise_required_sample_size(
            r_a, r_b, 0.2, 0.2, b, procedure=procedure, design=d
        )
        m_res = segment_pairwise_minimum_detectable_effect(
            n_res.n_per_arm, 0.2, 0.2, b, procedure=procedure, design=d
        )
        assert m_res.mde_relative == pytest.approx(math.expm1(theta), abs=0.01)

    def test_delta_not_log1p_of_bare_difference(self):
        """delta = log1p(r_a) - log1p(r_b), NOT log1p(r_a - r_b) - the sizing
        result must NOT match what the wrong bare-difference form implies."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign()
        n_res = segment_pairwise_required_sample_size(
            0.5, 0.2, 0.3, 0.3, b, procedure=procedure, design=d
        )

        correct_theta = math.log1p(0.50) - math.log1p(0.20)
        wrong_theta = math.log1p(0.50 - 0.20)

        # n computed from the correct theta must reproduce n_res - a regression
        # to log1p(r_a - r_b) would fail this (thetas diverge ~27.7% here).
        expected = _baseline_only_ate_total(correct_theta, b, procedure, d) * (1 / 0.3 + 1 / 0.3)
        assert n_res.n_total == pytest.approx(expected, rel=1e-2)
        wrong = _baseline_only_ate_total(wrong_theta, b, procedure, d) * (1 / 0.3 + 1 / 0.3)
        assert n_res.n_total != pytest.approx(wrong, rel=1e-2)

    def test_distinct_baselines_are_additive(self):
        """Per-segment baselines: variance is the SUM of each segment's own term."""
        import math

        b_a = Baseline(mean=1.0, var=1.0)  # var/mean^2 = 1.0
        b_b = Baseline(mean=2.0, var=8.0)  # var/mean^2 = 2.0 - genuinely different from b_a
        procedure = make_procedure()
        d = PowerDesign(allocation=0.5)
        q_a, q_b = 0.3, 0.3

        distinct = segment_pairwise_achieved_power(
            2000, 0.2, 0.05, q_a, q_b, b_a, procedure=procedure, baseline_b=b_b, design=d
        )

        # Hand-computed additive SE^2: k * (term_a + term_b) / n_total, k=4 at allocation=0.5.
        n_t, n_c = distinct.n_per_arm, distinct.n_total - distinct.n_per_arm
        term_a = b_a.effective_var / (b_a.mean**2 * q_a)
        term_b = b_b.effective_var / (b_b.mean**2 * q_b)
        k = 1 / d.allocation + 1 / (1 - d.allocation)
        se2 = k * (term_a + term_b) / (n_t + n_c)
        theta = math.log1p(0.20) - math.log1p(0.05)
        z_alpha = 1.959963984540054  # norm.ppf(0.975)
        nc = theta / math.sqrt(se2)
        from scipy.stats import norm

        expected_power = 1.0 - norm.cdf(z_alpha - nc) + norm.cdf(-z_alpha - nc)
        assert distinct.power == pytest.approx(expected_power, rel=1e-6)

        # effective_var reflects segment A's baseline only - b_b's contribution
        # is folded into the variance/power math but not surfaced on its own.
        assert distinct.effective_var == b_a.effective_var

        # And it must differ from the shared-baseline (b_a only) case, since b_b's
        # variance/mean^2 ratio is not equal to b_a's.
        shared = segment_pairwise_achieved_power(
            2000, 0.2, 0.05, q_a, q_b, b_a, procedure=procedure, design=d
        )
        assert distinct.power != pytest.approx(shared.power)

    def test_n_per_arm_is_experiment_wide(self):
        """n_total scales as 1/q_a + 1/q_b when only the segment shares change.

        At allocation=0.5, n_C == n_T identically regardless of whether q_a/q_b
        are interpreted experiment-wide or scoped to segments A/B alone, so a
        bare n_total == 2 * n_per_arm check can't distinguish correct from
        buggy scoping. Hold r_a/r_b/baseline/design fixed and vary only the
        shares to pin the experiment-wide 1/q_a + 1/q_b scaling directly.
        """
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign(allocation=0.5)
        small_share = segment_pairwise_required_sample_size(
            0.2, 0.05, 0.1, 0.1, b, procedure=procedure, design=d
        )
        large_share = segment_pairwise_required_sample_size(
            0.2, 0.05, 0.4, 0.4, b, procedure=procedure, design=d
        )
        ratio = small_share.n_total / large_share.n_total
        expected_ratio = (1 / 0.1 + 1 / 0.1) / (1 / 0.4 + 1 / 0.4)
        assert ratio == pytest.approx(expected_ratio, rel=0.02)

    @pytest.mark.parametrize(
        ("q_a", "q_b"),
        [(0.0, 0.5), (1.0, 0.1), (0.6, 0.6), (-0.1, 0.5)],
    )
    def test_guards_invalid_shares(self, q_a, q_b):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(ValueError):
            segment_pairwise_required_sample_size(
                0.2, 0.05, q_a, q_b, b, procedure=make_procedure()
            )

    def test_guards_invalid_relative_lift(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(ValueError):
            segment_pairwise_required_sample_size(
                -1.5, 0.05, 0.2, 0.2, b, procedure=make_procedure()
            )

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_one_sided_alternative_round_trip(self, alternative):
        """One-sided alternatives exercise the sign-flip branch in the pairwise
        power/MDE helpers, not just the two-sided default every other test uses.
        """
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alpha=0.05, alternative=alternative)
        d = PowerDesign(power=0.8)
        # theta = log1p(r_a) - log1p(r_b) must be signed to match the
        # alternative: positive for "greater", negative for "less".
        r_a, r_b = (0.30, 0.05) if alternative == "greater" else (0.05, 0.30)
        n_res = segment_pairwise_required_sample_size(
            r_a, r_b, 0.2, 0.2, b, procedure=procedure, design=d
        )
        p_res = segment_pairwise_achieved_power(
            n_res.n_per_arm, r_a, r_b, 0.2, 0.2, b, procedure=procedure, design=d
        )
        assert abs(p_res.power - d.power) <= 0.02
        m_res = segment_pairwise_minimum_detectable_effect(
            n_res.n_per_arm, 0.2, 0.2, b, procedure=procedure, design=d
        )
        if alternative == "greater":
            assert available_mde(m_res) > 0
        else:
            assert available_mde(m_res) < 0

    @pytest.mark.parametrize(
        ("alternative", "r_a", "r_b", "code"),
        [
            ("greater", 0.05, 0.30, "power.r_a_below"),
            ("less", 0.30, 0.05, "power.r_a_above"),
        ],
    )
    def test_one_sided_wrong_pairwise_tail_is_refused(self, alternative, r_a, r_b, code):
        baseline = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alternative=alternative)
        design = PowerDesign()
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                r_a, r_b, 0.2, 0.2, baseline, procedure=procedure, design=design
            )
        assert exc_info.value.code == code

    def test_imbalanced_allocation_round_trip(self):
        """design.allocation != 0.5 is the only way the asymmetric per-segment
        split in _segment_arm_sizes differs from _compute_arms's symmetric case.
        """
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure(alpha=0.05)
        d = PowerDesign(power=0.8, allocation=0.8)
        n_res = segment_pairwise_required_sample_size(
            0.3, 0.05, 0.2, 0.2, b, procedure=procedure, design=d
        )
        p_res = segment_pairwise_achieved_power(
            n_res.n_per_arm, 0.3, 0.05, 0.2, 0.2, b, procedure=procedure, design=d
        )
        assert abs(p_res.power - d.power) <= 0.02
        n_control = n_res.n_total - n_res.n_per_arm
        assert n_res.n_per_arm > n_control, "80/20 allocation should favor treatment"

    def test_bonferroni_correction_increases_n(self):
        """correction='bonferroni' widens n_total for the pairwise solvers too,
        via the shared design.adjusted_alpha used by every solver's z_alpha."""
        b = Baseline(mean=1.0, var=1.0)
        procedure_none = make_procedure()
        d_none = PowerDesign()
        procedure_bonf = make_procedure(family_size=3)
        d_bonf = PowerDesign()
        n_none = segment_pairwise_required_sample_size(
            0.3, 0.05, 0.2, 0.2, b, procedure=procedure_none, design=d_none
        ).n_total
        n_bonf = segment_pairwise_required_sample_size(
            0.3, 0.05, 0.2, 0.2, b, procedure=procedure_bonf, design=d_bonf
        ).n_total
        assert n_bonf > n_none, "Bonferroni should need more units"

    def test_equal_lifts_raise(self):
        """r_a == r_b (theta=0) has no finite sizing solution - raise rather
        than silently return a meaningless n_total=4 with no real per-segment
        arm split behind it."""
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(0.2, 0.2, 0.2, 0.2, b, procedure=make_procedure())
        assert exc_info.value.code == "power.r_a_r"
        assert exc_info.value.context["r_a"] == 0.2
        assert exc_info.value.context["r_b"] == 0.2

    def test_tiny_segment_share_raises(self):
        """A segment share too small to produce >= 4 real units raises rather
        than fabricating a 2+2 arm split from a fractional unit count.
        """
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign(allocation=0.5)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                2, 0.3, 0.05, 0.05, 0.05, b, procedure=procedure, design=d
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved_n_per_arm_too_small"
        assert exc_info.value.context["n_per_arm"] == 2
        assert exc_info.value.context["q_a"] == 0.05

    def test_imbalanced_allocation_tiny_share_still_raises(self):
        """The tiny-share guard must be allocation-aware, not a flat
        n_total_seg >= 4 threshold. At allocation=0.8, n_per_arm=16 gives
        n_total=20 (n_t=16, n_c=4); q=0.3 implies a 6-unit segment share,
        clearing the OLD flat ">= 4" threshold - but the smaller (control)
        arm's true floor is 2 / min(0.8, 0.2) = 10 units, so this must
        still raise rather than silently fabricating control units."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign(allocation=0.8)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                16, 0.3, 0.05, 0.3, 0.3, b, procedure=procedure, design=d
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved_n_per_arm_too_small"
        assert exc_info.value.context["n_per_arm"] == 16
        assert exc_info.value.context["q_a"] == 0.3

    def test_required_sample_size_tiny_share_error_is_solver_specific(self):
        """segment_pairwise_required_sample_size derives n_total internally
        and never takes it as a parameter, so the underlying 2-arm-split
        guard surfaces under this solver's own code rather than the
        achieved-power one."""
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                procedure=make_procedure(),
                r_a=20,
                r_b=0,
                q_a=0.9,
                q_b=0.05,
                baseline_a=b,
                design=PowerDesign(allocation=0.5),
            )
        assert exc_info.value.code == "power.solved_too_small"

    def test_achieved_power_tiny_share_error_names_n_per_arm_not_n_total(self):
        """segment_pairwise_achieved_power takes n_per_arm, not n_total, so the
        underlying _segment_arm_sizes guard surfaces under this solver's own
        code. Also a genuine no-imbalance trigger: q_a == q_b == 0.5, so
        there is no imbalance to shrink."""
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                procedure=make_procedure(),
                n_per_arm=3,
                r_a=5.0,
                r_b=0.0,
                q_a=0.5,
                q_b=0.5,
                baseline_a=b,
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved_n_per_arm_too_small"

    def test_minimum_detectable_effect_tiny_share_also_raises(self):
        """The allocation-aware guard is documented on
        segment_pairwise_minimum_detectable_effect's Raises section too -
        this pins that it's reachable through that solver too, not just
        achieved_power. n_per_arm=3 at allocation=0.5, q_a=q_b=0.5:
        n_total=6, n_total_seg=3 < the 4-unit floor."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign(allocation=0.5)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_minimum_detectable_effect(
                procedure=procedure, n_per_arm=3, q_a=0.5, q_b=0.5, baseline_a=b, design=d
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved_n_per_arm_too_small"
        assert exc_info.value.context["n_per_arm"] == 3
        assert exc_info.value.context["q_a"] == 0.5

    def test_pairwise_n_per_arm_guard_has_one_code_across_entry_points(self):
        """Achieved-power and MDE expose the same allocation guard."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign(allocation=0.5)
        with pytest.raises(InvalidRequestError) as via_achieved:
            segment_pairwise_achieved_power(3, 0.5, 0.5, 0.5, 0.5, b, procedure=procedure, design=d)
        with pytest.raises(InvalidRequestError) as via_minimum:
            segment_pairwise_minimum_detectable_effect(
                procedure=procedure, n_per_arm=3, q_a=0.5, q_b=0.5, baseline_a=b, design=d
            )
        assert (
            via_achieved.value.code
            == via_minimum.value.code
            == "power.segment_pairwise_achieved_n_per_arm_too_small"
        )

    def test_tiny_share_guard_boundary_passes_at_exact_threshold(self):
        """n_per_arm=4 at allocation=0.5, q_a=q_b=0.5: n_total=8,
        n_total_seg=4 - exactly at the 2/min_alloc=4 floor. The guard is
        strict `<`, so this must return normally, not raise; pins that
        the fix is not an off-by-one that over-rejects the boundary."""
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        d = PowerDesign(allocation=0.5)
        result = segment_pairwise_minimum_detectable_effect(
            procedure=procedure, n_per_arm=4, q_a=0.5, q_b=0.5, baseline_a=b, design=d
        )
        assert available_mde(result) > 0

    def test_required_sample_size_refuses_nonzero_null_lift(self):
        """The sizing formula has no theta0 term, so a shifted null on a
        difference-of-lifts contrast is not well-defined -- refuse rather
        than silently sizing at theta0=0 while reporting power against a
        shifted null (the achieved power at the returned N would then not
        be the target power, silently)."""
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                0.2,
                0.05,
                0.3,
                0.3,
                b,
                procedure=make_procedure(null_lift=0.01),
                design=PowerDesign(),
            )
        assert exc_info.value.code == "power.segment_pairwise_required"

    def test_achieved_power_refuses_nonzero_null_lift(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                100,
                0.2,
                0.05,
                0.3,
                0.3,
                b,
                procedure=make_procedure(null_lift=0.01),
                design=PowerDesign(),
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved"

    def test_minimum_detectable_effect_refuses_nonzero_null_lift(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_minimum_detectable_effect(
                100, 0.3, 0.3, b, procedure=make_procedure(null_lift=0.01), design=PowerDesign()
            )
        assert exc_info.value.code == "power.segment_pairwise_minimum"

    def test_refuses_clustered_baseline_below_the_cluster_floor(self):
        """A clustered baseline's design-effect inflation flows through
        effective_var, but nothing floored the solved-for arm split at a
        sane cluster count -- an extreme lift could solve to well under
        one cluster per arm and still report a plausible-looking power,
        while the primary solver refuses the same baseline below its
        10-cluster floor."""
        b = Baseline(mean=1.0, var=1.0, cluster_icc=0.2, avg_cluster_size=50.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                r_a=1_000_000.0,
                r_b=0.0,
                q_a=0.5,
                q_b=0.5,
                baseline_a=b,
                procedure=make_procedure(dependence="cluster"),
                design=PowerDesign(power=0.8),
            )
        assert exc_info.value.code == "power.segment_clustered_baseline"


# Step 5: Joint Q-test power


class TestJointQPower:
    """Cochran's Q power: exact fixed-effects noncentral chi2; exact
    scaled-central-chi2 for equal-variance random-effects; measured
    mean-matched approximation otherwise.
    """

    def test_fixed_effects_hand_computed(self):
        """5-segment fixture: theta centred at zero, equal variances.

        lambda = sum w_k (theta_k - theta_bar_w)^2 = 0.0658 at
        theta=[0.20, 0.05, -0.05, -0.08, -0.12] (40/25/15/12/8 shares,
        centred), var=1 each. Verified against a 100k-rep simulation:
        predicted 0.053253 vs. empirical 0.0543.
        """
        theta = [0.40, 0.25, 0.15, 0.12, 0.08]
        theta = [t - sum(theta) / len(theta) for t in theta]
        var = [1.0] * 5
        power = joint_q_power_fixed(theta, var, alpha=0.05)
        assert power == pytest.approx(0.053253171775511664)

    def test_extreme_alpha_uses_survival_tail_fixed_and_random(self):
        """Tiny alpha must remain computable instead of rounding its complement."""
        fixed = joint_q_power_fixed([30.0, -30.0], [1.0, 1.0], alpha=1e-300)
        random = joint_q_power_random(40.0, [1.0, 1.0], alpha=1e-300)
        assert math.isfinite(fixed) and fixed > 0.0
        assert math.isfinite(random) and random > 0.0

    def test_alpha_point_five_percent_matches_previous_quantile(self):
        fixed = joint_q_power_fixed([0.40, 0.25, 0.15, 0.12, 0.08], [1.0] * 5, alpha=0.05)
        random = joint_q_power_random(tau_b=0.5, var=[1.0] * 5, alpha=0.05)
        assert fixed == pytest.approx(0.053253171775511664, abs=1e-12)
        assert random == pytest.approx(0.10779771684696493, abs=1e-12)

    def test_fixed_effects_centre_must_be_precision_weighted(self):
        """An unweighted centre strictly overstates lambda (and therefore
        power) relative to the precision-weighted centre, at unequal var."""
        theta = [0.30, 0.10, -0.10]
        var = [1.0, 4.0, 4.0]
        w = [1 / v for v in var]
        theta_bar_w = sum(wi * ti for wi, ti in zip(w, theta, strict=True)) / sum(w)
        theta_bar_unweighted = sum(theta) / len(theta)
        lam_w = sum(wi * (ti - theta_bar_w) ** 2 for wi, ti in zip(w, theta, strict=True))
        lam_u = sum(wi * (ti - theta_bar_unweighted) ** 2 for wi, ti in zip(w, theta, strict=True))
        assert lam_u > lam_w, "unweighted centre must overstate lambda"

    def test_random_effects_equal_variance_is_exact(self):
        """Equal-variance case: Q ~ (1 + tau_b^2/v) * chi2(K-1) exactly.

        K=5, var=1.0, tau_b=0.5 -> 0.107798. Verified against a 100k-rep
        simulation: 0.1078 exact vs. 0.1099 empirical.
        """
        power = joint_q_power_random(tau_b=0.5, var=[1.0] * 5, alpha=0.05)
        assert power == pytest.approx(0.10779771684696493)

    def test_random_effects_zero_tau_is_alpha(self):
        """tau_b=0 -> no true heterogeneity -> power = alpha (the type-I rate)."""
        power = joint_q_power_random(tau_b=0.0, var=[1.0, 2.0, 3.0, 4.0], alpha=0.05)
        assert power == pytest.approx(0.05, abs=1e-9)

    def test_random_effects_unequal_variance_mean_matched_approximation(self):
        """Unequal-variance fixture: E[lambda] = tau_b^2 * (sum(w) - sum(w^2)/sum(w)).

        var=[0.5, 0.5, 2.0, 2.0, 4.0], tau_b=1.0 -> lambda=3.619048,
        mean-matched approximate power 0.290982.
        """
        power = joint_q_power_random(tau_b=1.0, var=[0.5, 0.5, 2.0, 2.0, 4.0], alpha=0.05)
        assert power == pytest.approx(0.29098200229632765)

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_random_effects_fixed_effects_agree_on_calibration(self):
        """joint_q_power_random's approximation is calibrated (within a few
        percent) at moderate variance ratios, measured against simulation
        directly - not just asserted."""
        rng = np.random.default_rng(0)
        var = np.array([0.5, 0.5, 2.0, 2.0, 4.0])
        tau_b = 1.0
        k = len(var)
        w = 1 / var
        crit = scipy_chi2.ppf(0.95, k - 1)
        reps = 80_000
        fires = 0
        for _ in range(reps):
            theta = tau_b * rng.standard_normal(k)
            y = theta + rng.standard_normal(k) * np.sqrt(var)
            ybar_w = (w * y).sum() / w.sum()
            q = (w * (y - ybar_w) ** 2).sum()
            fires += q > crit
        empirical = fires / reps
        predicted = joint_q_power_random(tau_b=tau_b, var=var, alpha=0.05)
        assert abs(predicted - empirical) / empirical < 0.05, (
            f"predicted {predicted:.4f} vs empirical {empirical:.4f}"
        )

    def test_guards_fewer_than_two_segments(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed([0.1], [1.0])
        assert exc_info.value.code == "power.need_least_segments"
        assert exc_info.value.context["k"] == 1

    def test_guards_mismatched_shapes(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed([0.1, 0.2], [1.0])
        assert exc_info.value.code == "power.theta_var_same"

    def test_guards_non_positive_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed([0.1, 0.2, 0.3], [1.0, 0.0, 1.0])
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_guards_negative_tau_b(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_random(tau_b=-0.1, var=[1.0, 1.0, 1.0])
        assert exc_info.value.code == "power.tau_b"
        assert exc_info.value.context["tau_b"] == -0.1

    def test_guards_multidimensional_theta_var(self):
        """A 2D theta/var pair passes the shape-equality check but must
        not silently count segments from only the first axis."""
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(
                cast("Any", [[0.1, 0.2], [0.3, 0.4]]), cast("Any", [[1.0, 1.0], [1.0, 1.0]])
            )
        assert exc_info.value.code == "power.theta_var_one"

    def test_guards_multidimensional_var_random(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_random(tau_b=0.5, var=cast("Any", [[1.0, 1.0], [1.0, 1.0]]))
        assert exc_info.value.code == "power.theta_var_one"

    def test_guards_nonfinite_tau_b(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(InvalidRequestError) as exc_info:
                joint_q_power_random(tau_b=bad, var=[1.0, 1.0, 1.0])
            assert exc_info.value.code == "power.finite"
            assert exc_info.value.context["name"] == "tau_b"

    def test_guards_non_finite_theta(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed([0.1, float("nan"), 0.3], [1.0, 1.0, 1.0])
        assert exc_info.value.code == "power.theta_contains_non"

    def test_guards_alpha_out_of_range_fixed(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed([0.1, 0.2, 0.3], [1.0, 1.0, 1.0], alpha=1.0)
        assert exc_info.value.code == "estimation.diagnostics.alpha"
        assert exc_info.value.context["alpha"] == 1.0

    def test_guards_alpha_out_of_range_random(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_random(tau_b=0.5, var=[1.0, 1.0, 1.0], alpha=0.0)
        assert exc_info.value.code == "estimation.diagnostics.alpha"
        assert exc_info.value.context["alpha"] == 0.0


# Solver input refusals: wrong-side lifts, zero distance, degenerate
# baselines, and garbage arm sizes must raise, not return sentinels.


class TestSolverRefusals:
    def test_greater_refuses_lift_below_null(self):
        """A one-sided 'greater' design with a negative lift can never
        exceed alpha power - the solver must refuse, not size for the
        reflected effect."""
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                -0.1, b, procedure=make_procedure(alternative="greater"), design=PowerDesign()
            )
        assert exc_info.value.code == "power.relative_lift_lies"
        assert exc_info.value.context["relative_lift"] == -0.1

    def test_less_refuses_lift_above_null(self):
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                0.1, b, procedure=make_procedure(alternative="less"), design=PowerDesign()
            )
        assert exc_info.value.code == "power.relative_lift_lies_above_null"
        assert exc_info.value.context["relative_lift"] == 0.1

    def test_wrong_side_check_uses_null_boundary_not_zero(self):
        """The side check is against the DECLARED null, not 0: a zero lift
        under a shifted-null 'greater' guardrail is a legitimate NI design
        and must keep solving."""
        b = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(alternative="greater", null_lift=-0.01)
        d = PowerDesign()
        result = required_sample_size(0.0, b, procedure=procedure, design=d)
        assert result.n_per_arm > 2
        # ...while a lift BELOW that shifted boundary is refused.
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(-0.02, b, procedure=procedure, design=d)
        assert exc_info.value.code == "power.relative_lift_lies"
        assert exc_info.value.context["relative_lift"] == -0.02

    def test_right_side_one_sided_lifts_still_solve(self):
        b = Baseline(mean=1.0, var=2.0)
        n_greater = required_sample_size(
            0.1, b, procedure=make_procedure(alternative="greater"), design=PowerDesign()
        )
        n_less = required_sample_size(
            -0.1, b, procedure=make_procedure(alternative="less"), design=PowerDesign()
        )
        assert n_greater.n_per_arm > 2
        assert n_less.n_per_arm > 2
        assert abs(n_greater.power - 0.80) <= 0.01
        assert abs(n_less.power - 0.80) <= 0.01

    def test_zero_lift_raises_instead_of_n2_sentinel(self):
        """relative_lift exactly at the null boundary has no finite n.
        Raise like segment_pairwise_required_sample_size does, instead of
        silently returning the degenerate n_per_arm=2 sentinel."""
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.0, b, procedure=make_procedure())
        assert exc_info.value.code == "power.size_design_relative"
        assert exc_info.value.context["relative_lift"] == 0.0

    def test_lift_at_shifted_null_boundary_raises(self):
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                -0.01,
                b,
                procedure=make_procedure(alternative="greater", null_lift=-0.01),
                design=PowerDesign(),
            )
        assert exc_info.value.code == "power.size_design_relative"
        assert exc_info.value.context["relative_lift"] == -0.01

    def test_zero_variance_baseline_refused_at_construction(self):
        """var=0 (e.g. from_proportion(1.0)) previously died later with a
        bare ZeroDivisionError inside every solver."""
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_proportion(1.0)
        assert exc_info.value.code == "power.baseline.var_zero_variance"
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline(mean=1.0, var=0.0)
        assert exc_info.value.code == "power.baseline.var_zero_variance"

    def test_achieved_power_refuses_n_below_two(self):
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            achieved_power(procedure=make_procedure(), n_per_arm=-5, relative_lift=0.1, baseline=b)
        assert exc_info.value.code == "power.core.n_per_arm_min"
        with pytest.raises(InvalidRequestError) as exc_info:
            achieved_power(procedure=make_procedure(), n_per_arm=1, relative_lift=0.1, baseline=b)
        assert exc_info.value.code == "power.core.n_per_arm_min"

    def test_mde_refuses_n_below_two(self):
        b = Baseline(mean=1.0, var=2.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            minimum_detectable_effect(procedure=make_procedure(), n_per_arm=0, baseline=b)
        assert exc_info.value.code == "power.core.n_per_arm_min"

    def test_clustered_sequential_power_is_a_coded_capability_gap(self):
        """Clustered sequential power is a capability gap, not an invalid
        request, and the refusal a caller actually receives comes from the
        shared arm planning contract -- which every public solver consults
        before reaching the power module. Power-local specs for the same
        condition were unreachable and returned a divergent code."""
        from increment.errors import CodedError
        from increment.estimation.sequential import GaussianScoreMixture

        baseline = Baseline(mean=1.0, var=1.0, avg_cluster_size=5.0)
        procedure = make_procedure(
            dependence="cluster", inference=GaussianScoreMixture(), population="assigned"
        )
        with pytest.raises(CodedError) as raised:
            required_sample_size(0.2, baseline, procedure=procedure, design=PowerDesign())
        assert raised.value.code == "arm.inference.cluster"
        assert isinstance(raised.value, ValueError)

    def test_a_clustered_baseline_without_sequential_inference_still_solves(self):
        baseline = Baseline(mean=1.0, var=1.0, avg_cluster_size=5.0)
        procedure = make_procedure(dependence="cluster")
        result = required_sample_size(0.2, baseline, procedure=procedure, design=PowerDesign())
        assert result.n_per_arm > 2


class TestSequentialMdeUsesTheBoundarysOwnNullPower:
    """A sequential boundary spends at most alpha under the null, so alpha is
    only an upper bound on its zero-drift crossing probability. Refusing on the
    fixed-horizon anchor rejected every target power between the boundary's real
    crossing probability and alpha, where a minimum detectable effect exists.
    Targets the null enclosure cannot separate from that probability are
    refused as unresolved rather than answered."""

    # Zero-drift crossing probability of GaussianScoreMixture's own
    # se-independent boundary at fourteen equal looks, alpha=0.05.
    NULL_POWER = "0.00994547"

    @staticmethod
    def _procedure():
        from increment.estimation.sequential import GaussianScoreMixture

        return make_procedure(inference=GaussianScoreMixture(), population="assigned")

    def _mde(self, power: float):
        return minimum_detectable_effect(
            procedure=self._procedure(),
            n_per_arm=5000,
            baseline=Baseline(mean=1.0, var=2.0),
            design=PowerDesign(power=power),
        )

    def _null_enclosure(self):
        """The solver's highest-accuracy null-crossing enclosure."""
        from increment.power.sequential import NODES_MIN

        from ._sequential import sequential_search

        search = sequential_search(
            5000, Baseline(mean=1.0, var=2.0), self._procedure(), PowerDesign(), 14
        )
        null = search.point(0.0, 4 * NODES_MIN)
        while search.quadrature_dominates(null):
            null = search.point(0.0, search.doubled(null))
        return null

    def _solve(self, target_power: float):
        from ._sequential import sequential_solve

        return sequential_solve(
            5000,
            Baseline(mean=1.0, var=2.0),
            self._procedure(),
            PowerDesign(power=target_power),
            14,
        )

    @pytest.mark.parametrize("power", [0.05, 0.03, 0.02, 0.0125, 0.0116])
    def test_a_target_between_the_boundary_null_power_and_alpha_solves(self, power):
        # This boundary's zero-drift crossing probability is ~0.00995, well
        # below alpha=0.05; every target above it must yield a positive effect.
        result = self._mde(power)
        assert result.mde_relative > 0.0
        assert result.power == pytest.approx(power, rel=1e-6)
        assert result.power >= power

    def test_a_target_below_the_boundary_null_power_is_refused_with_that_number(self):
        with pytest.raises(InvalidRequestError) as raised:
            self._mde(0.009)
        assert raised.value.code == "power.sequential_mde.design_search_minimum"
        # The refusal quotes the boundary's actual crossing probability, not alpha.
        assert f"{raised.value.context['estimate']:.6g}" == self.NULL_POWER

    def test_the_mde_decreases_monotonically_with_target_power(self):
        mdes = [self._mde(p).mde_relative for p in (0.02, 0.05, 0.10, 0.80)]
        assert mdes == sorted(mdes)

    def test_a_target_just_above_the_null_power_solves(self):
        """As the target approaches the boundary's own null-crossing probability
        the effect approaches zero; the search starts at exactly zero, whose
        certified enclosure lies below the target, so nothing excludes the
        small positive answer."""
        result = self._mde(0.0116)
        assert 0.0 < result.mde_relative < 0.01
        assert result.power == pytest.approx(0.0116, rel=1e-6)

    def test_the_null_power_itself_is_unresolved(self):
        with pytest.raises(InvalidRequestError) as below:
            self._mde(0.009)
        null_power = cast("float", below.value.context["estimate"])
        assert f"{null_power:.6g}" == self.NULL_POWER
        with pytest.raises(InvalidRequestError) as raised:
            self._mde(null_power)
        assert raised.value.code == "power.minimum_detectable_effect.numerical_resolution"
        assert raised.value.context["target_power"] == null_power
        assert raised.value.context["direction"] == "increasing"
        lower, upper = cast("tuple[float, float]", raised.value.context["power_enclosure"])
        assert lower <= null_power <= upper

    @pytest.mark.parametrize("ulps", [-2, -1, 1, 2])
    def test_a_target_within_ulps_of_the_null_power_is_unresolved(self, ulps: int):
        """An ulp-stepped target still lies inside the null enclosure, whose
        half-width is orders of magnitude above float spacing: neither side
        can be certified, so the refusal names the enclosure."""
        with pytest.raises(InvalidRequestError) as below:
            self._mde(0.009)
        target = cast("float", below.value.context["estimate"])
        for _ in range(abs(ulps)):
            target = math.nextafter(target, 1.0 if ulps > 0 else 0.0)
        with pytest.raises(InvalidRequestError) as raised:
            self._mde(target)
        assert raised.value.code == "power.minimum_detectable_effect.numerical_resolution"
        lower, upper = cast("tuple[float, float]", raised.value.context["power_enclosure"])
        assert lower < target < upper

    def test_below_the_null_enclosure_is_refused_as_a_null_crossing(self):
        null = self._null_enclosure()
        with pytest.raises(InvalidRequestError) as exc_info:
            self._solve(math.nextafter(null.lower, 0.0))
        assert exc_info.value.code == "power.sequential_mde.design_search_minimum"


class TestSequentialMdeLargeStandardError:
    """Large log-scale standard errors remain overflow-safe. A representable
    answer is returned only when quadrature meets its tolerance; otherwise the
    numerical-resolution refusal is distinct from the preserved
    ``power.sequential_mde.*`` codes for physically unrepresentable answers."""

    @staticmethod
    def _procedure(alternative: str = "two-sided"):
        from increment.estimation.sequential import GaussianScoreMixture

        return make_procedure(
            inference=GaussianScoreMixture(), alternative=alternative, population="assigned"
        )

    def test_unit_standard_error_solves_to_a_finite_lift(self):
        # se = sqrt(var * (1/n_T + 1/n_C)) = 1 at the null.
        result = minimum_detectable_effect(
            procedure=self._procedure(),
            n_per_arm=2,
            baseline=Baseline(mean=1.0, var=1.0),
            design=PowerDesign(power=0.5),
        )
        mde = available_mde(result)
        assert math.isfinite(mde)
        assert mde > 0.0
        assert result.power == pytest.approx(0.5, rel=1e-6)

    def test_high_variance_crossing_refuses_when_quadrature_cannot_resolve_it(self, monkeypatch):
        from increment.errors import InvalidRequestError
        from increment.power import sequential as sequential_module

        # The crossing is representable, but a lowered node ceiling exhausts
        # it before the declared probability tolerance.
        monkeypatch.setattr(sequential_module, "NODES_MAX", 32)
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=self._procedure(),
                n_per_arm=2,
                baseline=Baseline(mean=1.0, var=36.0),
                design=PowerDesign(power=0.4),
            )
        assert raised.value.code == "power.minimum_detectable_effect.numerical_resolution"
        lower, upper = cast("tuple[float, float]", raised.value.context["power_enclosure"])
        assert lower < 0.4 < upper
        assert "ceiling" in cast(str, raised.value.context["stopping_reason"])

    def test_overflowing_effect_is_refused_through_the_public_api(self):
        from increment.errors import InvalidRequestError

        # At n=2 the log-scale SE grows with var, so a large var needs drift past
        # the largest representable log-effect (709.78): no float64 lift reaches
        # power 0.5, though an exact-arithmetic answer exists beyond it.
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=self._procedure(),
                n_per_arm=2,
                baseline=Baseline(mean=1.0, var=1e10),
                design=PowerDesign(power=0.5),
            )
        assert raised.value.code == "power.sequential_mde.unrepresentable"
        assert raised.value.context["target_power"] == 0.5
        se_ctx = raised.value.context["se"]
        assert isinstance(se_ctx, float) and math.isfinite(se_ctx)
        theta_ctx = raised.value.context["mde_theta"]
        assert isinstance(theta_ctx, float)
        assert theta_ctx > 700.0

    def test_compliance_adjusted_overflow_is_refused_through_the_public_api(self):
        from increment.errors import InvalidRequestError

        # The same limiting variance under compliance 1e-6: the admissible
        # lifts end where the raw lift divided by the compliance overflows,
        # so the refusal names the compliance and the finite raw lift.
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=self._procedure(),
                n_per_arm=2,
                baseline=Baseline(mean=1.0, var=1e10, compliance=1e-6),
                design=PowerDesign(power=0.5),
            )
        assert raised.value.code == "power.sequential_mde.compliance_unrepresentable"
        assert raised.value.context["compliance"] == 1e-6
        raw_ctx = raised.value.context["mde_relative"]
        assert isinstance(raw_ctx, float) and math.isfinite(raw_ctx)

    def test_floor_before_peak_does_not_imply_an_unrepresentable_answer(self):
        from increment.errors import InvalidRequestError

        # The public floor precedes the peak, but even the peak misses 50%.
        # This is genuinely unattainable rather than a hidden crossing.
        baseline = Baseline(mean=1.0, var=4.0, compliance=1e-6)
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=self._procedure("less"),
                n_per_arm=3,
                baseline=baseline,
                design=PowerDesign(power=0.5),
            )
        assert raised.value.code == "power.minimum_detectable_effect.unattainable"
        assert raised.value.context["limiting_condition"] == "sequential_exclusion"

    def test_certified_crossing_beyond_floor_preserves_the_decrease_code(self):
        from increment.errors import InvalidRequestError

        baseline = Baseline(mean=1.0, var=1.0, compliance=1e-6)
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=self._procedure("less"),
                n_per_arm=300,
                baseline=baseline,
                design=PowerDesign(power=0.8),
            )
        assert raised.value.code == "power.sequential_mde.decrease_unrepresentable"
        assert raised.value.context["target_power"] == 0.8
        assert raised.value.context["compliance"] == 1e-6
        crossing = raised.value.context["mde_relative"]
        assert isinstance(crossing, float)
        assert -1e6 < crossing < -1.0
        companion = achieved_power(
            300, -0.5, baseline, self._procedure("less"), PowerDesign(power=0.8)
        )
        assert companion.mde_relative is None
        assert companion.mde_unavailable_reason == "unattainable"

    def test_unresolved_beyond_floor_peak_is_numerical_not_unattainable(self, monkeypatch):
        from increment.power import sequential as sequential_module

        monkeypatch.setattr(sequential_module, "NODES_MAX", 32)
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=self._procedure("less"),
                n_per_arm=300,
                baseline=Baseline(mean=1.0, var=1.0, compliance=1e-6),
                design=PowerDesign(power=0.8),
            )
        assert raised.value.code == "power.minimum_detectable_effect.numerical_resolution"
        assert raised.value.context["direction"] == "decreasing"
        lower, upper = cast("tuple[float, float]", raised.value.context["power_enclosure"])
        assert lower <= 0.8 <= upper

    def test_an_empty_decreasing_direction_is_not_a_compliance_failure(self):
        from increment.errors import InvalidRequestError

        # Compliance 0.1 against a null of -0.5: no complier-scale decrease
        # keeps the effective lift above -1, so the direction is empty; the
        # increasing direction opens at 8.0 and solves.
        procedure = make_procedure(
            inference=self._procedure().inference,
            alternative="less",
            null_lift=-0.5,
            population="assigned",
        )
        with pytest.raises(InvalidRequestError) as raised:
            minimum_detectable_effect(
                procedure=procedure,
                n_per_arm=100,
                baseline=Baseline(mean=1.0, var=1.0, compliance=0.1),
                design=PowerDesign(power=0.5),
            )
        assert raised.value.code == "power.minimum_detectable_effect.unattainable"
        assert raised.value.context["limiting_condition"] == "empty_admissible_direction"


# Alternative-arm (H1) variance: the treatment arm's log-scale variance is
# evaluated at its own mean, so a bounded metric cannot plan for a rate above
# one and a decreasing direction has a noncentrality peak.


def _independent_h1_se_sq(
    n_t: float, n_c: float, baseline: Baseline, effective_lift: float, *, bounded: bool = False
) -> float:
    """Own-arm delta-method variance of the log ratio at ANALYZED counts: the
    treatment arm at its own mean with equal absolute variance, or with the
    Bernoulli shape for a bounded rate."""
    m0 = baseline.mean
    m1 = m0 * (1.0 + effective_lift)
    v = baseline.effective_var
    v1 = v * m1 * (1.0 - m1) / (m0 * (1.0 - m0)) if bounded else v
    return v1 / (n_t * m1**2) + v / (n_c * m0**2)


def _independent_h1_power(
    n_t: float,
    n_c: float,
    baseline: Baseline,
    procedure: Any,
    relative_lift: float,
    *,
    bounded: bool = False,
) -> float:
    """Delta-method power at a complier-scale ``relative_lift`` with each arm's
    variance evaluated at ITS OWN mean."""
    decision = procedure.decision
    effective = relative_lift * baseline.compliance
    se2 = _independent_h1_se_sq(n_t, n_c, baseline, effective, bounded=bounded)
    distance = math.log1p(effective) - math.log1p(decision.null_lift)
    nc = distance / math.sqrt(se2)
    z = float(norm.isf(procedure.compiled_tail_alpha))
    if decision.alternative == "two-sided":
        return float(norm.sf(z - nc) + norm.sf(z + nc))
    if decision.alternative == "greater":
        return float(norm.sf(z - nc))
    return float(norm.sf(z + nc))


def _independent_h1_n_total(
    baseline: Baseline, procedure: Any, design: PowerDesign, relative_lift: float, *, bounded=False
) -> float:
    """Closed-form total N of the own-arm model: the per-unit variance at the
    allocation fractions times ``(z_alpha + z_power)^2`` over the squared
    log-distance from the null. Two-sided uses the near-tail quantile, which
    ignores the far tail (relative error below 1e-5 at 80% power)."""
    decision = procedure.decision
    effective = relative_lift * baseline.compliance
    a = design.allocation
    per_unit = _independent_h1_se_sq(a, 1.0 - a, baseline, effective, bounded=bounded)
    z_sum = float(norm.isf(procedure.compiled_tail_alpha)) + float(norm.ppf(design.power))
    distance = math.log1p(effective) - math.log1p(decision.null_lift)
    return per_unit * z_sum**2 / distance**2


def _implied_complier_lift(mde_relative: float, *, null_lift: float, compliance: float) -> float:
    """Absolute complier-scale lift a signed distance-beyond-null MDE implies."""
    return math.expm1(math.log1p(null_lift) + math.log1p(mde_relative * compliance)) / compliance


def test_conversion_planning_never_implies_a_rate_above_one() -> None:
    """A bounded metric's alternative rate is ``p0 * (1 + lift)``. Planning
    for or reporting a rate above 1.0 is not a conservative approximation,
    it is an alternative that cannot occur. (The absorbed-factor plan keeps
    the log-ratio model, under which a detectable effect exists here; the
    runtime's exact binomial decision at these counts cannot reach 0.8.)"""
    from increment.errors import CodedError
    from increment.power import Baseline, minimum_detectable_effect, required_sample_size
    from tests.power._procedures import make_procedure

    baseline = Baseline.from_proportion(0.90)
    procedure = make_procedure(
        metric_type="conversion",
        identification="randomized",
        population="assigned",
        variance_adjustment="factor_absorption",
    )

    with pytest.raises(CodedError) as refusal:
        required_sample_size(0.20, baseline, procedure)
    assert refusal.value.code == "power.bounded_metric.rate_above_one"

    mde = minimum_detectable_effect(100, baseline, procedure).mde_relative
    assert mde is not None
    assert baseline.mean * (1.0 + mde) <= 1.0


def test_decreasing_h1_mde_returns_the_first_crossing():
    import math

    from increment.power import Baseline, PowerDesign, achieved_power, minimum_detectable_effect
    from tests.power._procedures import make_procedure

    procedure = make_procedure(
        alternative="less",
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
    )
    baseline = Baseline(mean=1.0, var=1.0)
    design = PowerDesign(power=0.7915)
    result = minimum_detectable_effect(50, baseline, procedure, design)
    assert result.mde_relative is not None
    distance = -math.log1p(result.mde_relative)
    assert distance == pytest.approx(1.095779457575454, rel=1e-8)
    assert distance < 1.108857552878572
    reported = achieved_power(50, result.mde_relative, baseline, procedure, design)
    assert reported.power == pytest.approx(design.power, abs=1e-8)


class TestShiftedNullAdmissibleInterval:
    """Under partial compliance a shifted null can sit outside the effective
    alternatives the design can reach: ``Baseline(compliance=0.1)`` reaches
    effective lifts above ``-0.1`` only, while the null sits at ``-0.5``.
    The admissible distance interval then has a nonzero lower endpoint in
    the increasing direction and is empty in the decreasing one."""

    _NULL_LIFT = -0.5
    _COMPLIANCE = 0.1
    _N = 100
    # Distance from the null to the first reachable effective lift, -0.1.
    _LOWER_DISTANCE = math.log1p(-0.1) - math.log1p(-0.5)

    @staticmethod
    def _baseline() -> Baseline:
        return Baseline(mean=1.0, var=1.0, compliance=0.1)

    def _implied(self, mde_relative: float) -> float:
        return _implied_complier_lift(
            mde_relative, null_lift=self._NULL_LIFT, compliance=self._COMPLIANCE
        )

    def test_shifted_null_mde_returns_first_admissible_endpoint(self):
        """The unconstrained crossing (distance ~0.4125) implies an absolute
        complier lift of ~-2.45, which cannot occur. The answer is the first
        representable candidate at the physical lower endpoint, reported with
        its ACTUAL power rather than an enforced equality with the target."""
        baseline = self._baseline()
        procedure = make_procedure(alternative="greater", null_lift=self._NULL_LIFT)
        design = PowerDesign(power=0.8)

        result = minimum_detectable_effect(self._N, baseline, procedure, design)
        mde = result.mde_relative
        assert mde is not None
        assert result.mde_unavailable_reason is None
        assert mde == pytest.approx(math.expm1(self._LOWER_DISTANCE) / self._COMPLIANCE, rel=1e-12)
        assert mde == pytest.approx(8.0, rel=1e-12)

        implied = self._implied(mde)
        assert implied > -1.0
        # The preceding public candidate in distance order is inadmissible.
        assert self._implied(math.nextafter(mde, -math.inf)) <= -1.0

        assert result.power == pytest.approx(
            _independent_h1_power(self._N, self._N, baseline, procedure, implied), rel=1e-9
        )
        assert result.power == pytest.approx(0.988908853293906, rel=1e-9)
        assert result.power > design.power

    def test_shifted_null_mde_solves_after_nonzero_lower_endpoint(self):
        """When the lower endpoint's power is insufficient the answer is the
        first crossing ABOVE it, not the endpoint itself."""
        baseline = self._baseline()
        procedure = make_procedure(alternative="greater", null_lift=self._NULL_LIFT)
        design = PowerDesign(power=0.999)

        result = minimum_detectable_effect(self._N, baseline, procedure, design)
        mde = result.mde_relative
        assert mde is not None
        assert mde > 8.0

        implied = self._implied(mde)
        assert implied > -1.0
        # Its predecessor is still in-domain: this is a crossing, not an endpoint.
        assert self._implied(math.nextafter(mde, -math.inf)) > -1.0

        assert result.power >= design.power
        assert result.power == pytest.approx(design.power, abs=1e-9)
        assert result.power == pytest.approx(
            _independent_h1_power(self._N, self._N, baseline, procedure, implied), rel=1e-9
        )
        assert achieved_power(self._N, implied, baseline, procedure, design).power == pytest.approx(
            result.power, rel=1e-9
        )

    def test_shifted_null_mde_refuses_empty_direction(self):
        """Every reachable effective lift is above -0.1, hence none is below
        the -0.5 null: a decreasing direction has no admissible candidate.
        Direct MDE refuses; a supplied-effect power query still answers, with
        a null companion MDE and its reason."""
        from increment.errors import InvalidRequestError

        baseline = self._baseline()
        procedure = make_procedure(alternative="less", null_lift=self._NULL_LIFT)
        design = PowerDesign(power=0.8)

        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(self._N, baseline, procedure, design)
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "empty_admissible_direction"
        assert refusal.value.context["target_power"] == design.power

        result = achieved_power(self._N, 0.0, baseline, procedure, design)
        assert result.power == pytest.approx(
            _independent_h1_power(self._N, self._N, baseline, procedure, 0.0), rel=1e-9
        )
        assert result.mde_relative is None
        assert result.mde_unavailable_reason == "unattainable"

    def test_shifted_null_mde_companions_share_admissible_endpoint(self):
        """The companion MDE describes the design at that size, not the
        supplied effect: two supplied lifts with different main powers carry
        the same companion as direct MDE, and sizing's companion equals
        direct MDE at the returned size."""
        baseline = self._baseline()
        procedure = make_procedure(alternative="greater", null_lift=self._NULL_LIFT)
        design = PowerDesign(power=0.8)

        direct = minimum_detectable_effect(self._N, baseline, procedure, design)
        below = achieved_power(self._N, -0.5, baseline, procedure, design)
        at_zero = achieved_power(self._N, 0.0, baseline, procedure, design)

        assert below.power == pytest.approx(
            _independent_h1_power(self._N, self._N, baseline, procedure, -0.5), rel=1e-9
        )
        assert at_zero.power == pytest.approx(
            _independent_h1_power(self._N, self._N, baseline, procedure, 0.0), rel=1e-9
        )
        assert below.power != at_zero.power
        assert direct.mde_relative is not None
        assert below.mde_relative == direct.mde_relative
        assert at_zero.mde_relative == direct.mde_relative
        assert below.mde_unavailable_reason is None
        assert at_zero.mde_unavailable_reason is None

        sized = required_sample_size(0.0, baseline, procedure, design)
        at_size = minimum_detectable_effect(sized.n_per_arm, baseline, procedure, design)
        assert sized.mde_relative == at_size.mde_relative
        assert sized.mde_unavailable_reason is None


def _randomized(**overrides: Any) -> Any:
    return make_procedure(
        identification="randomized",
        population="assigned",
        variance_adjustment="none",
        **overrides,
    )


def _absorbed(**overrides: Any) -> Any:
    """A randomized plan analyzed after factor absorption: its outcome is no
    longer a raw binary count, so a bounded metric keeps the log-ratio model
    (the raw-count exact binomial decision is covered in
    ``test_binomial_planning.py``)."""
    return make_procedure(
        identification="randomized",
        population="assigned",
        variance_adjustment="factor_absorption",
        **overrides,
    )


class TestIndependentArmVariance:
    """Every fixed-horizon answer matches the two-arm own-mean formula,
    including a positive lift and unequal allocation, never another
    public solver."""

    @pytest.mark.parametrize("relative_lift", [0.25, -0.20])
    @pytest.mark.parametrize("allocation", [0.5, 0.3])
    def test_achieved_power_matches_two_arm_formula(self, relative_lift, allocation):
        baseline = Baseline(mean=2.0, var=3.0)
        procedure = _randomized()
        design = PowerDesign(allocation=allocation)
        result = achieved_power(300, relative_lift, baseline, procedure, design)
        n_c = result.n_total - result.n_per_arm
        assert n_c == math.ceil(300 * (1 - allocation) / allocation)
        expected = _independent_h1_power(300, n_c, baseline, procedure, relative_lift)
        assert result.power == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("relative_lift", [0.25, -0.20])
    @pytest.mark.parametrize("allocation", [0.5, 0.3])
    def test_required_sample_size_reaches_target_under_the_two_arm_formula(
        self, relative_lift, allocation
    ):
        baseline = Baseline(mean=2.0, var=3.0)
        procedure = _randomized()
        design = PowerDesign(allocation=allocation)
        sized = required_sample_size(relative_lift, baseline, procedure, design)
        n_c = sized.n_total - sized.n_per_arm
        power = _independent_h1_power(sized.n_per_arm, n_c, baseline, procedure, relative_lift)
        assert power >= design.power
        assert power == pytest.approx(sized.power, rel=1e-12)
        # One fewer treatment unit falls short: the size is minimal at its rounding.
        smaller = _independent_h1_power(
            sized.n_per_arm - 1, n_c, baseline, procedure, relative_lift
        )
        assert smaller < design.power + 5e-3
        expected_total = _independent_h1_n_total(baseline, procedure, design, relative_lift)
        assert abs(sized.n_per_arm - math.ceil(expected_total * allocation)) <= 1


class TestBoundedDomain:
    """A conversion or retention rate lives in ``(0, 1]``: planning refuses a
    rate above one for the null or the supplied alternative, allows exactly
    one, and never returns a minimum detectable effect implying more. The
    closed-form checks use an absorbed-factor plan, which keeps the
    log-ratio model; the refusals precede any model."""

    @pytest.mark.parametrize("metric_type", ["conversion", "retention"])
    def test_mde_at_a_high_baseline_stays_inside_the_unit_interval(self, metric_type):
        baseline = Baseline.from_proportion(0.9)
        procedure = _absorbed(metric_type=metric_type)
        result = minimum_detectable_effect(100, baseline, procedure)
        mde = result.mde_relative
        assert mde is not None
        assert mde == pytest.approx(0.10174, abs=5e-6)
        assert baseline.mean * (1.0 + mde) <= 1.0
        assert result.power == pytest.approx(0.8, abs=1e-9)
        assert _independent_h1_power(100, 100, baseline, procedure, mde, bounded=True) == (
            pytest.approx(0.8, abs=1e-9)
        )
        assert achieved_power(100, mde, baseline, procedure).power == pytest.approx(0.8, abs=1e-9)

    def test_shifted_null_composes_with_the_bernoulli_shape(self):
        baseline = Baseline.from_proportion(0.9)
        procedure = _absorbed(metric_type="conversion", alternative="greater", null_lift=-0.05)
        result = minimum_detectable_effect(100, baseline, procedure)
        assert result.mde_relative is not None
        implied = math.expm1(math.log1p(-0.05) + math.log1p(result.mde_relative))
        assert baseline.mean * (1.0 + implied) <= 1.0
        assert _independent_h1_power(100, 100, baseline, procedure, implied, bounded=True) == (
            pytest.approx(0.8, abs=1e-9)
        )

    @pytest.mark.parametrize("solver", ["required", "achieved"])
    def test_supplied_rate_above_one_is_refused(self, solver):
        from increment.errors import InvalidRequestError

        baseline = Baseline.from_proportion(0.9)
        procedure = _randomized(metric_type="retention")
        with pytest.raises(InvalidRequestError) as refusal:
            if solver == "required":
                required_sample_size(0.2, baseline, procedure)
            else:
                achieved_power(100, 0.2, baseline, procedure)
        assert refusal.value.code == "power.bounded_metric.rate_above_one"
        assert refusal.value.context["role"] == "alternative"

    def test_a_treatment_rate_of_exactly_one_is_allowed_and_its_neighbour_is_not(self):
        """``p0 = 0.5`` with lift 1.0 lands exactly on a rate of one, which
        contributes zero treatment variance; the next representable lift
        implies a rate above one."""
        from increment.errors import InvalidRequestError

        baseline = Baseline.from_proportion(0.5)
        procedure = _absorbed(metric_type="conversion", alternative="greater")
        exact = achieved_power(4, 1.0, baseline, procedure)
        # Only the control term remains: se2 = 0.25 / (4 * 0.25).
        nc = math.log(2.0) / math.sqrt(0.25 / (4 * 0.25))
        assert exact.power == pytest.approx(float(norm.sf(norm.isf(0.05) - nc)), rel=1e-12)
        with pytest.raises(InvalidRequestError) as refusal:
            achieved_power(4, math.nextafter(1.0, math.inf), baseline, procedure)
        assert refusal.value.code == "power.bounded_metric.rate_above_one"

    def test_mde_stops_at_the_rate_ceiling(self):
        """A target the ceiling rate just reaches returns the ceiling; one it
        cannot reach is unattainable with that limiting condition, and a
        supplied-effect query there keeps its answer with a null companion."""
        from increment.errors import InvalidRequestError

        baseline = Baseline.from_proportion(0.5)
        procedure = _absorbed(metric_type="conversion", alternative="greater")
        ceiling_power = achieved_power(4, 1.0, baseline, procedure).power
        at_ceiling = minimum_detectable_effect(
            4, baseline, procedure, PowerDesign(power=ceiling_power)
        )
        assert at_ceiling.mde_relative is not None
        assert at_ceiling.mde_relative == pytest.approx(1.0, rel=1e-15)
        assert baseline.mean * (1.0 + at_ceiling.mde_relative) <= 1.0
        assert at_ceiling.power >= ceiling_power

        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(4, baseline, procedure, PowerDesign(power=0.8))
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "bounded_rate_ceiling"
        assert refusal.value.context["maximum_power"] == pytest.approx(ceiling_power)

        companion = achieved_power(4, 0.5, baseline, procedure, PowerDesign(power=0.8))
        assert companion.power == pytest.approx(
            _independent_h1_power(4, 4, baseline, procedure, 0.5, bounded=True), rel=1e-12
        )
        assert companion.mde_relative is None
        assert companion.mde_unavailable_reason == "unattainable"

    def test_null_rate_above_one_and_incompatible_baseline_are_refused_before_solving(self):
        from increment.errors import InvalidRequestError

        with pytest.raises(InvalidRequestError) as null_refusal:
            minimum_detectable_effect(
                100,
                Baseline.from_proportion(0.9),
                _randomized(metric_type="conversion", null_lift=0.2),
            )
        assert null_refusal.value.code == "power.bounded_metric.rate_above_one"
        assert null_refusal.value.context["role"] == "null"

        with pytest.raises(InvalidRequestError) as baseline_refusal:
            required_sample_size(
                0.01, Baseline(mean=1.5, var=0.1), _randomized(metric_type="conversion")
            )
        assert baseline_refusal.value.code == "power.bounded_metric.baseline"


class TestFirstCrossingAndUnattainability:
    """In the decreasing direction the noncentrality peaks; the answer is
    the first crossing below the peak, and a target above the peak has no
    answer while supplied-effect queries still answer."""

    _BASELINE = Baseline(mean=1.0, var=1.0)

    def test_target_above_the_peak_is_unattainable_with_the_peak_power(self):
        from increment.errors import InvalidRequestError

        procedure = _randomized(alternative="less")
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(50, self._BASELINE, procedure, PowerDesign(power=0.8))
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        context = refusal.value.context
        assert context["limiting_condition"] == "noncentrality_peak"
        assert context["direction"] == "decreasing"
        assert context["maximum_power"] == pytest.approx(0.7915601693852341, rel=1e-12)
        assert context["target_power"] == 0.8

    def test_supplied_effect_keeps_its_power_with_a_null_companion(self):
        procedure = _randomized(alternative="less")
        result = achieved_power(50, -0.5, self._BASELINE, procedure, PowerDesign(power=0.8))
        assert result.power == pytest.approx(
            _independent_h1_power(50, 50, self._BASELINE, procedure, -0.5), rel=1e-12
        )
        assert result.mde_relative is None
        assert result.mde_unavailable_reason == "unattainable"

    def test_a_reachable_decrease_inverts_to_target_and_lies_below_the_peak(self):
        procedure = _randomized(alternative="less")
        design = PowerDesign(power=0.7)
        result = minimum_detectable_effect(50, self._BASELINE, procedure, design)
        assert result.mde_relative is not None
        assert -math.log1p(result.mde_relative) < 1.108857552878572
        assert result.power == pytest.approx(design.power, abs=1e-9)
        assert _independent_h1_power(50, 50, self._BASELINE, procedure, result.mde_relative) == (
            pytest.approx(design.power, abs=1e-9)
        )
        companion = achieved_power(50, -0.2, self._BASELINE, procedure, design)
        assert companion.mde_relative == result.mde_relative


class TestBoundedDecreasingPeak:
    """A bounded metric's decrease shrinks the treatment rate, and with it
    the Bernoulli variance, so its noncentrality peak follows the rate
    ratio ``(1 - p0) n_T / (p0 n_C)``. The refusal's maximum power and the
    first crossing below it must agree with an independent maximization
    of the Bernoulli-shape power, and a supplied effect that reaches the
    target can never be paired with a missing companion."""

    @staticmethod
    def _procedure(metric_type: str, null_lift: float = 0.0) -> Any:
        return _absorbed(metric_type=metric_type, alternative="less", null_lift=null_lift)

    @staticmethod
    def _independent_peak(n: int, baseline: Baseline, procedure: Any) -> tuple[float, float]:
        """``(distance, power)`` maximizing the own-arm Bernoulli power over
        the decrease, by bounded scalar search on the independent formula."""
        from scipy.optimize import minimize_scalar

        null_lift = procedure.decision.null_lift

        def power_at(distance: float) -> float:
            lift = math.expm1(math.log1p(null_lift) - distance)
            return _independent_h1_power(n, n, baseline, procedure, lift, bounded=True)

        found = minimize_scalar(
            lambda d: -power_at(d), bounds=(1e-3, 8.0), method="bounded", options={"xatol": 1e-12}
        )
        return float(found.x), -float(found.fun)

    @pytest.mark.parametrize(
        ("metric_type", "p0", "null_lift", "n"),
        [("conversion", 0.2, 0.0, 50), ("retention", 0.35, -0.05, 40), ("retention", 0.6, 0.0, 30)],
    )
    def test_maximum_power_and_first_crossing_follow_the_bernoulli_peak(
        self, metric_type, p0, null_lift, n
    ):
        from increment.errors import InvalidRequestError

        baseline = Baseline.from_proportion(p0)
        procedure = self._procedure(metric_type, null_lift)
        d_peak, maximum = self._independent_peak(n, baseline, procedure)

        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(n, baseline, procedure, PowerDesign(power=maximum + 1e-9))
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "noncentrality_peak"
        assert refusal.value.context["maximum_power"] == pytest.approx(maximum, rel=1e-9)

        design = PowerDesign(power=maximum - 1e-6)
        result = minimum_detectable_effect(n, baseline, procedure, design)
        mde = available_mde(result)
        assert 0.0 < -math.log1p(mde) < d_peak
        assert result.power >= design.power
        assert result.power == pytest.approx(design.power, abs=1e-9)
        implied = _implied_complier_lift(mde, null_lift=null_lift, compliance=1.0)
        assert _independent_h1_power(n, n, baseline, procedure, implied, bounded=True) == (
            pytest.approx(design.power, abs=1e-9)
        )
        assert achieved_power(n, implied, baseline, procedure, design).mde_relative == mde

    def test_supplied_effect_and_direct_solve_agree_at_the_scanned_peak(self):
        """Conversion at ``p0 = 0.2``, ``n = 50``: the scanned peak distance is
        2.141039 and its power exceeds 0.724132789, so the direct solve must
        answer and the supplied effect there must carry that answer."""
        baseline = Baseline.from_proportion(0.2)
        procedure = self._procedure("conversion")
        design = PowerDesign(power=0.724132789)
        supplied = achieved_power(50, math.expm1(-2.141039), baseline, procedure, design)
        assert supplied.power >= design.power
        direct = minimum_detectable_effect(50, baseline, procedure, design)
        mde = available_mde(direct)
        assert direct.power >= design.power
        assert -math.log1p(mde) <= 2.141039
        assert available_mde(supplied) == mde

    @pytest.mark.parametrize("p0", [1e-310, 5e-309])
    def test_subnormal_baseline_does_not_overflow_the_peak_search(self, p0):
        """A subnormal bounded baseline drives the peak search's rate ratio
        past float64's exponential range; the guarded ``q`` must not raise
        a bare ``OverflowError`` and must instead answer or refuse with a
        coded, finite result."""
        from increment.errors import CodedError

        baseline = Baseline.from_proportion(p0)
        procedure = self._procedure("conversion")
        design = PowerDesign(power=0.8)

        try:
            result = minimum_detectable_effect(50, baseline, procedure, design)
        except CodedError as refusal:
            assert refusal.code in (
                "power.minimum_detectable_effect.unattainable",
                "power.minimum_detectable_effect.unrepresentable",
            )
        else:
            assert math.isfinite(result.power)
            assert 0.0 <= result.power <= 1.0

        companion = achieved_power(50, -0.5, baseline, procedure, design)
        assert math.isfinite(companion.power)
        assert 0.0 <= companion.power <= 1.0
        if companion.mde_relative is not None:
            assert isinstance(companion.mde_relative, float)
        else:
            assert companion.mde_unavailable_reason is not None


class TestZeroNullCompanionFields:
    """At n=100 on a unit baseline the two-sided minimum detectable effect is
    about 0.409841573 with power 0.8 under its own variance; the stale
    control-arm variance would report 0.680394958 at that same lift."""

    _BASELINE = Baseline(mean=1.0, var=1.0)

    def test_mde_and_its_achieved_power_agree(self):
        procedure = _randomized()
        result = minimum_detectable_effect(100, self._BASELINE, procedure)
        assert result.mde_relative == pytest.approx(0.409841573, abs=5e-9)
        assert result.power == pytest.approx(0.8, abs=1e-9)
        reported = achieved_power(100, 0.409841573, self._BASELINE, procedure).power
        assert reported == pytest.approx(0.8, abs=1e-6)
        assert reported != pytest.approx(0.680394958, abs=1e-3)

    def test_companion_is_independent_of_the_supplied_effect(self):
        procedure = _randomized()
        direct = minimum_detectable_effect(100, self._BASELINE, procedure).mde_relative
        for lift in (-0.3, 0.0, 0.1, 0.9):
            companion = achieved_power(100, lift, self._BASELINE, procedure)
            assert companion.mde_relative == direct
            assert companion.mde_unavailable_reason is None


class TestKnobsThroughAllThreeApis:
    """Trigger rate, CUPED, compliance and unequal allocation compose through
    sizing, achieved power, and the minimum detectable effect on the
    own-arm model at ANALYZED counts."""

    _BASELINE = Baseline(mean=10.0, var=25.0, cuped_rho=0.5, compliance=0.6, trigger_rate=0.4)

    def _procedure(self) -> Any:
        return make_procedure(decision_method="cuped", alternative="greater")

    def test_sizing_and_achieved_power_agree_with_the_analyzed_count_formula(self):
        procedure = self._procedure()
        design = PowerDesign(power=0.85, allocation=0.3)
        sized = required_sample_size(0.1, self._BASELINE, procedure, design)
        assert sized.n_triggered_per_arm == round(sized.n_per_arm * 0.4)
        achieved = achieved_power(sized.n_per_arm, 0.1, self._BASELINE, procedure, design)
        n_c = achieved.n_total - achieved.n_per_arm
        expected = _independent_h1_power(
            achieved.n_per_arm * 0.4, n_c * 0.4, self._BASELINE, procedure, 0.1
        )
        assert achieved.power == pytest.approx(expected, rel=1e-12)
        assert achieved.power >= design.power
        assert achieved.effective_var == pytest.approx(25.0 * 0.75)

    def test_mde_inverts_on_the_complier_scale(self):
        procedure = self._procedure()
        design = PowerDesign(power=0.85, allocation=0.3)
        result = minimum_detectable_effect(2000, self._BASELINE, procedure, design)
        mde = result.mde_relative
        assert mde is not None
        n_c = result.n_total - result.n_per_arm
        assert _independent_h1_power(2000 * 0.4, n_c * 0.4, self._BASELINE, procedure, mde) == (
            pytest.approx(design.power, abs=1e-9)
        )
        assert achieved_power(2000, mde, self._BASELINE, procedure, design).mde_relative == mde
        # Halving compliance doubles the complier-scale effect the design detects.
        halved = Baseline(mean=10.0, var=25.0, cuped_rho=0.5, compliance=0.3, trigger_rate=0.4)
        doubled = minimum_detectable_effect(2000, halved, procedure, design).mde_relative
        assert doubled is not None
        assert doubled == pytest.approx(2 * mde, rel=0.03)


class TestNumericalBoundaries:
    """Extreme but valid inputs answer or refuse with a coded reason; none
    raise a raw overflow or domain error, and no distance is zeroed by a
    fixed cutoff."""

    @staticmethod
    def _valid(result: Any) -> None:
        assert 0.0 <= result.power <= 1.0
        if result.mde_relative is None:
            assert result.mde_unavailable_reason is not None
        else:
            assert math.isfinite(result.mde_relative)
            assert result.mde_relative > -1.0
            assert result.mde_unavailable_reason is None

    def test_adjacent_lifts_keep_their_distance(self):
        baseline = Baseline(mean=1.0, var=1.0)
        null_lift = -0.01
        procedure = _randomized(alternative="greater", null_lift=null_lift)
        above = math.nextafter(null_lift, math.inf)
        sized = required_sample_size(above, baseline, procedure)
        assert sized.n_per_arm > 10**30
        assert sized.power >= 0.8
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(null_lift, baseline, procedure)
        assert exc_info.value.code == "power.size_design_relative"
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(math.nextafter(null_lift, -math.inf), baseline, procedure)
        assert exc_info.value.code == "power.relative_lift_lies"
        self._valid(achieved_power(100, above, baseline, procedure))

    def test_null_near_the_float_ceiling(self):
        procedure = _randomized(null_lift=1e307)
        baseline = Baseline(mean=1.0, var=1e-26)
        result = minimum_detectable_effect(500, baseline, procedure)
        self._valid(result)
        assert result.power == pytest.approx(0.8, rel=1e-9)
        self._valid(achieved_power(500, 1.0000000001e307, baseline, procedure))
        self._valid(required_sample_size(1.5e307, Baseline(mean=1.0, var=1.0), procedure))

    def test_clustered_power_at_an_extreme_noncentrality_is_certain_not_zero(self):
        """A tiny finite variance drives the cluster t reference's
        noncentrality past the backend's range; the answer is the certain
        detection it is, never a fabricated zero."""
        baseline = Baseline(mean=1.0, var=1e-20, cluster_icc=0.05, avg_cluster_size=10.0)
        result = achieved_power(
            40, 0.1, baseline, ArmPlanningProcedure.standard("mean", clustered=True)
        )
        self._valid(result)
        assert result.power == 1.0

    @staticmethod
    def _chi_below(dof: float, w: float) -> float:
        """``P(sqrt(chi2_dof / dof) < w)`` in closed form for one to three dof."""
        if dof == 1.0:
            return math.erf(w / math.sqrt(2.0))
        if dof == 2.0:
            return -math.expm1(-w * w)
        # dof 3: P(chi2_3 < x) = erf(sqrt(x / 2)) - sqrt(2 x / pi) exp(-x / 2) at x = 3 w^2.
        tail = math.sqrt(6.0 / math.pi) * w * math.exp(-1.5 * w * w)
        return math.erf(w * math.sqrt(1.5)) - tail

    @pytest.mark.parametrize(
        ("dof", "tail", "nc"),
        [
            (1.0, 1e-12, 5e9),
            (1.0, 1e-12, -5e9),
            (2.0, 1e-20, 5e9),
            (3.0, 1e-30, 5e9),
            (1.0, 1e-300, 5e9),
            (3.0, 0.025, 1e200),
        ],
        ids=[
            "dof1-interior",
            "dof1-mirror",
            "dof2-interior",
            "dof3-interior",
            "underflow",
            "overflow",
        ],
    )
    def test_far_noncentral_t_tail_matches_its_closed_form(self, dof, tail, nc):
        """T = (Z + nc) / W with W = sqrt(chi2_dof / dof), so the directional
        tail is E[G((Z + nc) / crit)] with G the distribution of W. At these
        noncentralities the numerator noise moves it from G(|nc| / crit) by
        less than half an ulp (the kernel accepts that limit only under its
        curvature bound), and G has a closed form for one to three degrees of
        freedom. The cases cover interior tails at integer and half-integer
        shapes, a ratio whose square underflows (the tail is about
        1.25e-290), and one whose square overflows (certain detection)."""
        from increment.estimation._tails import student_t_isf
        from increment.power._noncentral_t import _scalar_power_from_nc

        crit = student_t_isf(tail, dof)
        alternative = "greater" if nc > 0 else "less"
        power = _scalar_power_from_nc(nc, alternative=alternative, tail_alpha=tail, dof=dof)
        assert power == pytest.approx(self._chi_below(dof, abs(nc) / crit), rel=1e-10)

    @staticmethod
    def _tail_power(nc: float, alternative: str, tail: float, dof: float) -> float:
        from increment.power._noncentral_t import _scalar_power_from_nc

        return _scalar_power_from_nc(
            nc, alternative=cast("Any", alternative), tail_alpha=tail, dof=dof
        )

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * math.erfc(-x / math.sqrt(2.0))

    @pytest.mark.parametrize("nc", [0.5, 2.0, 4.0])
    def test_one_dof_ordinary_tails_match_the_bivariate_normal_orthant(self, nc):
        """At one degree of freedom ``W = |N|``, so ``T > c > 0`` is the orthant
        ``{Z + nc > c N, Z + nc > -c N}`` of a bivariate normal with
        correlation ``(1 - c**2) / (1 + c**2)``: ``P(T > c) = Phi(h) - 2 T(h, c)``
        with ``h = nc / hypot(1, c)`` and Owen's ``T``. The opposite tail is
        ``Phi(-h) - 2 T(h, c)`` and two-sided power ``1 - 4 T(h, c)``. At the
        ordinary critical value the numerator's clipping edge lies inside the
        normal bulk, where fixed Gauss-Hermite rules miss the answer."""
        from scipy.special import owens_t

        from increment.estimation._tails import student_t_isf

        tail = 0.025
        crit = student_t_isf(tail, 1.0)
        h = nc / math.hypot(1.0, crit)
        owen = float(owens_t(h, crit))
        greater = self._tail_power(nc, "greater", tail, 1.0)
        assert greater == pytest.approx(self._normal_cdf(h) - 2.0 * owen, rel=1e-12, abs=0.0)
        assert self._tail_power(-nc, "less", tail, 1.0) == greater
        # The oracle's own subtraction cancels here; its absolute error is ~1e-16.
        opposite = self._tail_power(nc, "less", tail, 1.0)
        assert opposite == pytest.approx(self._normal_cdf(-h) - 2.0 * owen, rel=1e-9, abs=1e-15)
        two_sided = self._tail_power(nc, "two-sided", tail, 1.0)
        assert two_sided == pytest.approx(1.0 - 4.0 * owen, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize(
        ("tail", "nc"),
        [
            (2e-10, 1e8),
            (2e-10, 5e8),
            (2e-10, 1e9),
            (2e-10, 2e9),
            (2e-10, math.nextafter(2.0**31, 0.0)),
            (2e-10, math.nextafter(2.0**31, math.inf)),
            (1e-6, 3e5),
            (math.atan(1e-6) / math.pi, 8e5),
        ],
    )
    def test_one_dof_far_tails_match_the_odd_extension_oracle(self, tail, nc):
        """Extending ``P(W < w) = erf(w / sqrt(2))`` oddly gives
        ``A1 = erf(nc / (sqrt(2) hypot(c, 1)))`` with ``0 <= P(T > c) - A1 <=
        Phi(-nc)``, and the opposite tail is at most ``Phi(-nc)`` as well: zero
        in binary64 here, so every direction equals ``A1``. The rows are the
        tail allocation whose backend answers were reported wrong, both sides
        of the former ``2**31`` switch, and critical values near ``nc``, where
        the step in the chi integrand is too sharp for fixed rules and the
        noise-free limit is off by more than half an ulp."""
        from increment.estimation._tails import student_t_isf

        crit = student_t_isf(tail, 1.0)
        expected = math.erf(nc / (math.sqrt(2.0) * math.hypot(crit, 1.0)))
        for signed, alternative in ((nc, "greater"), (-nc, "less"), (nc, "two-sided")):
            power = self._tail_power(signed, alternative, tail, 1.0)
            assert power == pytest.approx(expected, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize(
        ("tail", "nc"),
        [
            (0.025, 3.0),
            (0.025, -3.0),
            (
                1.0 / (math.hypot(1e5, math.sqrt(2.0)) * (math.hypot(1e5, math.sqrt(2.0)) + 1e5)),
                8e4,
            ),
        ],
    )
    def test_two_dof_tails_match_the_exact_integral(self, tail, nc):
        """At two degrees of freedom ``P(W < w) = 1 - exp(-w**2)``; completing
        the square gives ``P(T > c) = Phi(m) - (c / h) exp(-(m / h)**2)
        Phi(m c / h)`` with ``h = hypot(c, sqrt(2))`` for either sign of ``m``.
        The last row's tail is the central df=2 tail at ``c = 1e5``; there the
        noise-free limit is off by 3e-11 relative."""
        from increment.estimation._tails import student_t_isf

        crit = student_t_isf(tail, 2.0)
        h = math.hypot(crit, math.sqrt(2.0))
        noise = (crit / h) * math.exp(-((nc / h) ** 2)) * self._normal_cdf(nc * crit / h)
        expected = self._normal_cdf(nc) - noise
        power = self._tail_power(nc, "greater", tail, 2.0)
        assert power == pytest.approx(expected, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("tail", [0.2, 0.025, 1e-290])
    @pytest.mark.parametrize("nc", [0.5, 2.8, 5.0, 9.0])
    def test_two_dof_two_sided_power_matches_its_exact_form(self, tail, nc):
        """At two degrees of freedom ``P(W < w) = 1 - exp(-w**2)``, and a
        Gaussian integral gives two-sided power ``1 - E[exp(-(Z + nc)**2 /
        c**2)] = -expm1(-log1p(2 / c**2) / 2 - nc**2 / (c**2 + 2))``: both
        tails at once, without cancellation even near ``(1 + nc**2) / c**2``,
        about 1e-289 at the smallest allocation. The critical values fall on
        both sides of ``sqrt(dof)``, and the opposite tail runs from a sizable
        share of the answer (``nc = 0.5``) to below its rounding (``nc = 9``)."""
        from increment.estimation._tails import student_t_isf

        crit = student_t_isf(tail, 2.0)
        exponent = -0.5 * math.log1p(2.0 / (crit * crit)) - nc * nc / (crit * crit + 2.0)
        power = self._tail_power(nc, "two-sided", tail, 2.0)
        assert power == pytest.approx(-math.expm1(exponent), rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("nc", [0.5, 2.8, 5.0, 9.0])
    def test_two_dof_tiny_tail_keeps_its_leading_term(self, nc):
        """At two degrees of freedom ``P(T > c) = E[(1 - exp(-t)) 1(Z + nc >
        0)]`` with ``t = (Z + nc)**2 / c**2``, and ``1 - exp(-t) = t (1 +
        O(t))`` gives ``((nc**2 + 1) Phi(nc) + nc phi(nc)) / c**2`` to a
        relative ``O(1 / c**2)``. At ``c`` near 7e144 the tail is near 1e-289,
        where the exact form's subtraction cannot resolve it."""
        from increment.estimation._tails import student_t_isf

        tail = 1e-290
        crit = student_t_isf(tail, 2.0)
        density = math.exp(-0.5 * nc * nc) / math.sqrt(2.0 * math.pi)
        expected = ((nc * nc + 1.0) * self._normal_cdf(nc) + nc * density) / (crit * crit)
        power = self._tail_power(nc, "greater", tail, 2.0)
        assert power == pytest.approx(expected, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("nc", [0.5, 3.0, 8.0])
    def test_one_dof_tiny_two_sided_power_keeps_its_leading_term(self, nc):
        """At one degree of freedom ``W = |N|``, so two-sided power is
        ``E[erf(|Z + nc| / (c sqrt(2)))] = sqrt(2 / pi) E|Z + nc| / c`` to a
        relative ``O(1 / c**2)``, with ``E|Z + nc| = nc erf(nc / sqrt(2)) + 2
        phi(nc)``. The critical value of a 1e-300 allocation, about 3e299,
        squares its reciprocal below the normal range; the answer, near
        1e-300, must neither round to zero nor be refused."""
        from increment.estimation._tails import student_t_isf

        tail = 1e-300
        crit = student_t_isf(tail, 1.0)
        density = math.exp(-0.5 * nc * nc) / math.sqrt(2.0 * math.pi)
        mean_abs = nc * math.erf(nc / math.sqrt(2.0)) + 2.0 * density
        expected = math.sqrt(2.0 / math.pi) * mean_abs / crit
        power = self._tail_power(nc, "two-sided", tail, 1.0)
        assert power == pytest.approx(expected, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("dof", [2.5, 30.5])
    @pytest.mark.parametrize("tail", [0.025, 1e-12])
    @pytest.mark.parametrize("nc", [0.5, 2.8])
    def test_two_sided_power_is_the_sum_of_both_tails(self, dof, tail, nc):
        """``P(|T| > c) = P(T > c) + P(T < -c)``, the second the upper tail at
        ``-nc``, at noninteger degrees of freedom with critical values on both
        sides of ``sqrt(dof)``. The opposite tail runs from a sizable share of
        the answer to a small one, so a two-sided answer without it fails."""
        greater = self._tail_power(nc, "greater", tail, dof)
        opposite = self._tail_power(nc, "less", tail, dof)
        two_sided = self._tail_power(nc, "two-sided", tail, dof)
        assert two_sided == pytest.approx(greater + opposite, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("dof", [1e24, 1e100])
    def test_huge_dof_power_stays_within_its_normal_limit_bound(self, dof):
        """``|E[Phi(m - c W)] - Phi(m - c)| <= c E|W - 1| / sqrt(2 pi)``, and
        ``E|W - 1| <= sqrt(2 (1 - E W)) <= 1 / sqrt(dof)`` since Kershaw's
        inequality puts ``E W`` above ``sqrt(1 - 1 / (2 dof))``. Each tail then
        lies within ``c / sqrt(2 pi dof)`` of its normal limit: about 8e-13 at
        1e24 degrees of freedom, far below rounding at 1e100."""
        from increment.estimation._tails import student_t_isf

        tail, nc = 0.025, 2.8
        crit = student_t_isf(tail, dof)
        gap = crit / math.sqrt(2.0 * math.pi * dof)
        greater = self._normal_cdf(nc - crit)
        opposite = self._normal_cdf(-nc - crit)
        power = self._tail_power(nc, "greater", tail, dof)
        assert power == pytest.approx(greater, rel=1e-13, abs=gap)
        two_sided = self._tail_power(nc, "two-sided", tail, dof)
        assert two_sided == pytest.approx(greater + opposite, rel=1e-13, abs=2.0 * gap)

    @pytest.mark.parametrize("nc", [1.0, 3.0, -2.0])
    def test_three_dof_ordinary_tails_match_the_exact_integral(self, nc):
        """At three degrees of freedom ``P(W < w) = 2 Phi(sqrt(3) w) - 1 -
        sqrt(6 / pi) w exp(-3 w**2 / 2)``. With ``h = hypot(c, sqrt(3))`` and
        ``r = m / h`` the first part integrates to the one-dof orthant at
        ``c / sqrt(3)`` and the second to a truncated Gaussian moment:
        ``P(T > c) = Phi(sqrt(3) r) - 2 T(sqrt(3) r, c / sqrt(3)) - sqrt(6 / pi)
        exp(-3 r**2 / 2) / h * ((m c**2 / h**2) Phi(m c / h) + (c / h) phi(m c / h))``."""
        from scipy.special import owens_t

        from increment.estimation._tails import student_t_isf

        tail = 0.025
        crit = student_t_isf(tail, 3.0)
        h = math.hypot(crit, math.sqrt(3.0))
        r = nc / h
        edge = nc * crit / h
        density = math.exp(-0.5 * edge * edge) / math.sqrt(2.0 * math.pi)
        moment = (nc * crit * crit / (h * h)) * self._normal_cdf(edge) + (crit / h) * density
        owen = float(owens_t(math.sqrt(3.0) * r, crit / math.sqrt(3.0)))
        orthant = self._normal_cdf(math.sqrt(3.0) * r) - 2.0 * owen
        expected = orthant - math.sqrt(6.0 / math.pi) * math.exp(-1.5 * r * r) / h * moment
        power = self._tail_power(nc, "greater", tail, 3.0)
        assert power == pytest.approx(expected, rel=1e-10, abs=0.0)

    def test_three_dof_far_tail_keeps_the_numerator_noise(self):
        """At ``m = 1.2e4`` and the critical value of a 1e-12 tail (about
        1e4), the odd extension ``A3 = erf(sqrt(3/2) r) - sqrt(6 / pi) r
        (c / h)**2 exp(-3 r**2 / 2)`` is exact to ``Phi(-m) = 0``. The
        noise-free chi-square limit is high by about 7e-9 here."""
        from increment.estimation._tails import student_t_isf

        tail, nc = 1e-12, 1.2e4
        crit = student_t_isf(tail, 3.0)
        h = math.hypot(crit, math.sqrt(3.0))
        r = nc / h
        noise = math.sqrt(6.0 / math.pi) * r * (crit / h) ** 2 * math.exp(-1.5 * r * r)
        expected = math.erf(math.sqrt(1.5) * r) - noise
        power = self._tail_power(nc, "greater", tail, 3.0)
        assert power == pytest.approx(expected, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize(("dof", "tail", "ratio"), [(5.0, 1e-18, 0.8), (30.0, 1e-90, 1.0)])
    def test_large_critical_values_keep_the_second_order_noise(self, dof, tail, ratio):
        """``P(T > c) = E[H(nc + Z)]`` with ``H(x) = G(x / c)``, so
        ``H(nc) + H''(nc) / 2`` with ``H'' = f_W'(w) / c**2`` differs from it by
        the fourth-order term ``H'''' / 8``, below 1e-12 relative at these
        critical values (about 6e3 and 5e3); the noise-free ``H(nc)`` alone is
        off by about 5e-8 and 1e-7 relative."""
        from scipy.special import gammainc, gammaln

        from increment.estimation._tails import student_t_isf

        crit = student_t_isf(tail, dof)
        nc = ratio * crit
        half = 0.5 * dof
        w = nc / crit
        log_density = (
            math.log(2.0)
            + half * math.log(half)
            - float(gammaln(half))
            + (dof - 1.0) * math.log(w)
            - half * w * w
        )
        curvature = math.exp(log_density) * ((dof - 1.0) / w - dof * w) / (crit * crit)
        expected = float(gammainc(half, half * w * w)) + 0.5 * curvature
        power = self._tail_power(nc, "greater", tail, dof)
        assert power == pytest.approx(expected, rel=1e-11, abs=0.0)

    def test_wrong_direction_tiny_tail_is_representable(self):
        """At one degree of freedom and ``m > 0``, ``P(T > c)`` at ``-m`` is
        ``(2 / c) int_0^inf phi(t / c) Phi(-m - t) dt``. Since ``phi(0) (1 -
        t**2 / (2 c**2)) <= phi(t / c) <= phi(0)``, it lies below
        ``(2 phi(0) / c) (phi(m) - m Phi(-m))`` by a relative gap near
        ``1 / (m c)**2``, under 1e-13 at ``c = 1e6``, ``m = 8``. The answer,
        about 6e-23, must not round to zero."""
        from increment.estimation._tails import student_t_isf

        tail, m = math.atan(1e-6) / math.pi, 8.0
        crit = student_t_isf(tail, 1.0)
        excess = math.exp(-0.5 * m * m) / math.sqrt(2.0 * math.pi) - m * self._normal_cdf(-m)
        expected = 2.0 * excess / (crit * math.sqrt(2.0 * math.pi))
        assert self._tail_power(m, "less", tail, 1.0) == pytest.approx(expected, rel=1e-12, abs=0.0)
        assert self._tail_power(-m, "greater", tail, 1.0) == pytest.approx(
            expected, rel=1e-12, abs=0.0
        )
        favorable = math.erf(m / (math.sqrt(2.0) * math.hypot(crit, 1.0))) + expected
        assert self._tail_power(m, "greater", tail, 1.0) == pytest.approx(
            favorable, rel=1e-12, abs=0.0
        )

    @pytest.mark.parametrize("dof", [1.0, 2.5, 5.0])
    @pytest.mark.parametrize("nc", [1.0, -1.0])
    def test_negative_and_zero_critical_values(self, dof, nc):
        """A tail allocation above one half has a negative critical value, and
        ``P(T > c) = 1 - P(T > -c)`` at ``-nc`` relates it to the positive one
        (both well inside ``(0, 1)`` here). One half is the zero critical
        value, where ``P(T > 0) = Phi(nc)``."""
        tail = 0.9
        greater = self._tail_power(nc, "greater", tail, dof)
        mirrored = self._tail_power(-nc, "greater", 1.0 - tail, dof)
        assert greater == pytest.approx(1.0 - mirrored, rel=1e-12, abs=0.0)
        assert self._tail_power(-nc, "less", tail, dof) == greater
        at_zero = self._tail_power(nc, "greater", 0.5, dof)
        assert at_zero == pytest.approx(self._normal_cdf(nc), rel=1e-15, abs=0.0)

    @pytest.mark.parametrize("dof", [5.0, 30.0, 1e4])
    def test_ordinary_tails_agree_with_the_backend_where_it_is_accurate(self, dof):
        """A consistency check at an ordinary critical value and small
        noncentrality, where the backend series is accurate; the closed forms
        above are the discriminating checks. At ``dof = 1e4`` the chi density
        is narrow, which defeats fixed rules on ``W`` taken from zero."""
        from scipy.stats import nct

        from increment.estimation._tails import student_t_isf

        tail, nc = 0.025, 2.5
        crit = student_t_isf(tail, dof)
        expected = float(nct.sf(crit, dof, nc))
        power = self._tail_power(nc, "greater", tail, dof)
        assert power == pytest.approx(expected, rel=1e-8, abs=0.0)

    def test_two_cluster_arms_use_the_one_dof_oracle_and_replay(self):
        """Two clusters per arm is one degree of freedom. At a tail allocation
        of 2e-10 the critical value is about 1.6e9, and a planning variance of
        1e-19 puts the noncentrality near 1e9, where every direction of the
        answer is ``erf(nc / (sqrt(2) hypot(c, 1)))``. Sizing crosses from one
        to two degrees of freedom at the first arm with three clusters; the
        minimum detectable effect at two clusters replays."""
        from increment.estimation._tails import student_t_isf

        var, lift = 1e-19, 0.1
        baseline = Baseline(mean=1.0, var=var, avg_cluster_size=10.0)
        procedure = make_procedure(alpha=4e-10, dependence="cluster")
        crit = student_t_isf(procedure.compiled_tail_alpha, 1.0)

        def oracle(relative_lift: float) -> float:
            theta = math.log1p(relative_lift)
            nc = theta / math.sqrt(var / (20.0 * math.exp(2.0 * theta)) + var / 20.0)
            return math.erf(nc / (math.sqrt(2.0) * math.hypot(crit, 1.0)))

        result = achieved_power(20, lift, baseline, procedure)
        assert result.n_clusters_per_arm == 2
        assert result.power == pytest.approx(oracle(lift), rel=1e-12, abs=0.0)

        sized = required_sample_size(lift, baseline, procedure)
        assert sized.n_per_arm == 21
        assert sized.power >= 0.8
        assert achieved_power(21, lift, baseline, procedure).power == sized.power

        detectable = minimum_detectable_effect(20, baseline, procedure)
        assert detectable.mde_relative is not None
        replay = achieved_power(20, detectable.mde_relative, baseline, procedure)
        assert replay.power == pytest.approx(0.8, rel=1e-9, abs=0.0)
        assert oracle(detectable.mde_relative) == pytest.approx(0.8, rel=1e-9, abs=0.0)

    @pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
    def test_null_overflowing_the_complier_scale_is_unrepresentable(self, alternative):
        """A null of ``1e307`` under compliance ``0.01`` is a complier-scale
        lift of ``1e309``: no increase is representable, and no decrease
        within the reachable ``log(1 / 0.99)`` is either. The direction is
        physical, so the refusal is unrepresentable, not empty."""
        from increment.errors import InvalidRequestError

        baseline = Baseline(mean=1.0, var=1e-26, compliance=0.01)
        procedure = make_procedure(alternative=alternative, null_lift=1e307)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, baseline, procedure)
        assert refusal.value.code == "power.minimum_detectable_effect.unrepresentable"

    def test_decrease_past_the_overflow_band_is_answered(self):
        """A null of ``1e308`` under compliance ``0.5`` overflows the complier
        scale until the decrease passes distance ``log(2 / 1.797...)`` =
        0.10664: a first crossing beyond that band is answered and reaches
        its supplied-effect query; one inside the band is unrepresentable
        even though larger decreases are representable. At such a null the
        treatment term vanishes, so the distance is ``z_sum sqrt(v / n_C)``."""
        from increment.errors import InvalidRequestError

        null_lift = 1e308
        procedure = make_procedure(alternative="less", null_lift=null_lift)
        baseline = Baseline(mean=1.0, var=1.0, compliance=0.5)
        result = minimum_detectable_effect(100, baseline, procedure)
        mde = available_mde(result)
        assert result.power == pytest.approx(0.8, abs=1e-9)
        z_sum = float(norm.isf(0.05)) + float(norm.ppf(0.8))
        distance = -math.log1p(0.5 * mde)
        assert distance > 0.10664
        assert distance == pytest.approx(z_sum * math.sqrt(1.0 / 100), rel=1e-9)
        implied = _implied_complier_lift(mde, null_lift=null_lift, compliance=0.5)
        assert math.isfinite(implied)
        supplied = achieved_power(100, implied, baseline, procedure)
        assert supplied.power == pytest.approx(0.8, abs=1e-9)
        assert supplied.mde_relative == mde

        hidden = Baseline(mean=1.0, var=0.1, compliance=0.5)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, hidden, procedure)
        assert refusal.value.code == "power.minimum_detectable_effect.unrepresentable"
        assert refusal.value.context["direction"] == "decreasing"
        larger = achieved_power(100, implied, hidden, procedure)
        assert larger.power > 0.8
        assert larger.mde_relative is None
        assert larger.mde_unavailable_reason == "unrepresentable"

    @pytest.mark.parametrize(
        ("mean", "var", "mde_outcome", "sizing_outcome"),
        [
            (1.0, 1e-300, "answers", "answers"),
            (1e300, 1.0, "answers", "answers"),
            (1e300, 1e300, "answers", "answers"),
            (1e-300, 1.0, "refuses", "overflows"),
            (1e-300, 1e-300, "refuses", "answers"),
        ],
    )
    def test_extreme_variance_and_means_never_overflow(
        self, mean, var, mde_outcome, sizing_outcome
    ):
        """The two-sided distance at ``n = 100`` is ``z_sum * sqrt(2 v / (n
        m0^2))``: a relative lift when it stays below the float64 log range
        (means at or above one here), otherwise refused as unrepresentable.
        Sizing a 20% lift needs the per-unit variance ``2 v / m0^2`` itself
        to fit, which ``v = 1`` at a mean of ``1e-300`` does not."""
        from increment.errors import InvalidRequestError

        baseline = Baseline(mean=mean, var=var)
        procedure = _randomized()
        supplied = achieved_power(100, 0.2, baseline, procedure)
        self._valid(supplied)
        if mde_outcome == "answers":
            result = minimum_detectable_effect(100, baseline, procedure)
            self._valid(result)
            z_sum = float(norm.isf(0.025)) + float(norm.ppf(0.8))
            log_distance = math.log(z_sum) + 0.5 * (
                math.log(2.0 * var / 100) - 2.0 * math.log(mean)
            )
            assert available_mde(result) == pytest.approx(math.exp(log_distance), rel=1e-4)
            assert result.power == pytest.approx(0.8, abs=1e-9)
            assert supplied.mde_relative == result.mde_relative
        else:
            with pytest.raises(InvalidRequestError) as refusal:
                minimum_detectable_effect(100, baseline, procedure)
            assert refusal.value.code == "power.minimum_detectable_effect.unrepresentable"
            assert supplied.mde_relative is None
            assert supplied.mde_unavailable_reason == "unrepresentable"
        if sizing_outcome == "answers":
            sized = required_sample_size(0.2, baseline, procedure)
            self._valid(sized)
            assert sized.power >= 0.8
        else:
            with pytest.raises(ValueError):
                required_sample_size(0.2, baseline, procedure)

    def test_extreme_alpha_uses_survival_tails(self):
        baseline = Baseline(mean=1.0, var=1.0)
        procedure = _randomized(alpha=1e-300)
        achieved = achieved_power(1000, 0.1, baseline, procedure)
        self._valid(achieved)
        assert 0.0 < achieved.power < 1e-100
        mde = minimum_detectable_effect(1000, baseline, procedure)
        self._valid(mde)
        assert mde.power == pytest.approx(0.8, abs=1e-9)
        sized = required_sample_size(0.1, baseline, procedure)
        self._valid(sized)
        assert sized.power >= 0.8

    @pytest.mark.parametrize("compliance", [1e-300, 1e-10, 1e-3])
    def test_compliance_representability_limits(self, compliance):
        from increment.errors import InvalidRequestError

        baseline = Baseline(mean=1.0, var=1.0, compliance=compliance)
        increasing = minimum_detectable_effect(100, baseline, make_procedure(alternative="greater"))
        self._valid(increasing)
        assert increasing.mde_relative is not None
        assert increasing.mde_relative * compliance == pytest.approx(
            minimum_detectable_effect(
                100, Baseline(mean=1.0, var=1.0), make_procedure(alternative="greater")
            ).mde_relative,
            rel=1e-9,
        )
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, baseline, make_procedure(alternative="less"))
        assert refusal.value.code == "power.minimum_detectable_effect.unattainable"
        assert refusal.value.context["limiting_condition"] == "relative_lift_floor"
        self._valid(achieved_power(100, 0.5, baseline, make_procedure()))


class TestZeroDistancePolicy:
    """When zero distance is admissible, a target at or below the null's own
    power refuses (``design_search_minimum``) and a target just above it
    returns a positive effect; when zero is inadmissible the
    constrained-endpoint rules apply instead."""

    def test_just_above_null_power_returns_a_small_positive_effect(self):
        baseline = Baseline(mean=1.0, var=2.0)
        procedure = _randomized()
        with pytest.raises(InvalidRequestError) as probe:
            minimum_detectable_effect(100, baseline, procedure, PowerDesign(power=1e-9))
        null_power = probe.value.context["estimate"]
        assert isinstance(null_power, float)
        with pytest.raises(InvalidRequestError) as refusal:
            minimum_detectable_effect(100, baseline, procedure, PowerDesign(power=null_power))
        assert refusal.value.code == "power.minimum_detectable_effect.design_search_minimum"
        just_above = minimum_detectable_effect(
            100, baseline, procedure, PowerDesign(power=null_power + 1e-6)
        )
        assert just_above.mde_relative is not None
        assert 0.0 < just_above.mde_relative < 0.01
        assert just_above.power >= null_power + 1e-6
        assert just_above.power == pytest.approx(null_power + 1e-6, abs=1e-12)

    def test_inadmissible_zero_uses_the_endpoint_instead_of_the_zero_policy(self):
        """A target below alpha would refuse when zero is admissible;
        with the null below the compliance floor the answer is the interval's
        first admissible candidate and its actual power."""
        baseline = Baseline(mean=1.0, var=1.0, compliance=0.1)
        procedure = make_procedure(alternative="greater", null_lift=-0.5)
        result = minimum_detectable_effect(100, baseline, procedure, PowerDesign(power=0.01))
        assert result.mde_relative == pytest.approx(8.0, rel=1e-12)
        assert result.power > 0.9


# Coded refusals: every remaining power/core.py refusal not already
# exercised above, one assertion per stable code.


class TestCodedRefusals:
    def test_log1mexp_needs_nonpositive_x(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            _log1mexp(0.5)
        assert exc_info.value.code == "power.log_exp_needs"
        assert exc_info.value.context["x"] == 0.5

    def test_solvers_require_a_relative_decision_policy(self):
        procedure = make_procedure()
        absolute = procedure.model_copy(
            update={
                "decision": AbsoluteDecisionPolicy(
                    alternative="two-sided", null_abs=0.0, family=procedure.decision.family
                )
            }
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.1, Baseline(mean=1.0, var=1.0), absolute)
        assert exc_info.value.code == "power.power_solvers_relative"

    def test_n_per_arm_must_be_an_integer(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            achieved_power(1.5, 0.1, Baseline(mean=1.0, var=1.0), make_procedure())  # ty: ignore[invalid-argument-type]
        assert exc_info.value.code == "power.core.n_per_arm_int"
        assert exc_info.value.context["n_per_arm"] == 1.5

    def test_sequential_sample_size_gives_up_when_unresolved(self, monkeypatch):
        """With the crossing quadrature ceiled, no probed size resolves and the
        sequential size search refuses rather than sizing on an unresolved
        bracket."""
        from increment.power import sequential as sequential_module

        monkeypatch.setattr(sequential_module, "NODES_MAX", 32)
        procedure = make_procedure(inference=GaussianScoreMixture(), population="assigned")
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.05, Baseline(mean=1.0, var=1e8), procedure, planned_looks=5)
        assert exc_info.value.code == "power.sequential_sample_size"

    def test_relative_lift_below_null_with_greater_alternative(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                -0.1, Baseline(mean=1.0, var=1.0), make_procedure(alternative="greater")
            )
        assert exc_info.value.code == "power.relative_lift_lies"
        assert exc_info.value.context["relative_lift"] == -0.1

    def test_relative_lift_above_null_with_less_alternative(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(
                0.1, Baseline(mean=1.0, var=1.0), make_procedure(alternative="less")
            )
        assert exc_info.value.code == "power.relative_lift_lies_above_null"
        assert exc_info.value.context["relative_lift"] == 0.1

    def test_relative_lift_exactly_at_the_null_boundary(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.0, Baseline(mean=1.0, var=1.0), make_procedure())
        assert exc_info.value.code == "power.size_design_relative"

    def test_relative_lift_too_close_to_the_null_to_size(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(1e-300, Baseline(mean=1.0, var=1.0), make_procedure())
        assert exc_info.value.code == "power.sample_size_detecting"

    def test_sequential_design_gives_up_past_1024x_the_fixed_horizon_size(self, monkeypatch):
        # GaussianScoreMixture is tuned at its own se-independent optimum, so
        # it no longer diverges from the fixed-horizon size under ordinary
        # parameters; force the never-detected branch to exercise the
        # bracket-growth cap itself.
        from increment.power import core as core_module

        baseline = Baseline(mean=1.0, var=2.0)
        procedure = make_procedure(inference=GaussianScoreMixture(), population="assigned")
        monkeypatch.setattr(core_module, "_sequential_sample_size_detected", lambda *a: False)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.05, baseline, procedure, planned_looks=5)
        assert exc_info.value.code == "power.sequential_design_more"

    def test_segment_share_too_small_for_a_2_arm_split(self):
        b = Baseline(mean=1.0, var=1.0)
        # The public solvers re-raise this refusal under their own
        # ``*_n_per_arm_too_small`` code, so the allocation-aware
        # ``2 / min(allocation, 1 - allocation)`` floor is asserted directly.
        with pytest.raises(InvalidRequestError) as exc_info:
            _segment_arm_sizes(0.01, 100, PowerDesign())
        assert exc_info.value.code == "power.segment_share_n"
        assert exc_info.value.context["q"] == 0.01
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                100, 0.3, 0.05, 0.01, 0.99, b, procedure=make_procedure(), design=PowerDesign()
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved_n_per_arm_too_small"
        assert exc_info.value.context["q_a"] == 0.01

    def test_segment_shares_must_each_be_in_unit_interval(self):
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(100, 0.3, 0.05, 0.0, 0.5, b, procedure=procedure)
        assert exc_info.value.code == "power.q_a"
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(100, 0.3, 0.05, 0.5, 0.0, b, procedure=procedure)
        assert exc_info.value.code == "power.q_b"

    def test_segment_shares_cannot_exceed_the_whole_experiment(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(100, 0.3, 0.05, 0.6, 0.6, b, procedure=make_procedure())
        assert exc_info.value.code == "power.q_a_q"
        assert exc_info.value.context["q_a_plus_q_b"] == pytest.approx(1.2)

    def test_pairwise_solvers_refuse_non_default_baseline_fields(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                100,
                0.3,
                0.05,
                0.5,
                0.5,
                Baseline(mean=1.0, var=1.0, compliance=0.5),
                procedure=make_procedure(),
                baseline_b=b,
            )
        assert exc_info.value.code == "power.segment_pairwise_solvers"
        assert exc_info.value.context["unsupported"] == ("compliance",)

    def test_achieved_power_refuses_a_one_cluster_per_arm_baseline(self):
        """A clustered baseline whose per-segment arms hold one cluster each
        is refused with the cluster-floor code, not relabeled as a small-n
        split."""
        baseline = Baseline(mean=1.0, var=1.0, cluster_icc=0.05, avg_cluster_size=10.0)
        procedure = make_procedure(alternative="greater", dependence="cluster")
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                10, 0.3, 0.05, 0.5, 0.5, baseline, procedure=procedure, design=PowerDesign()
            )
        assert exc_info.value.code == "power.segment_clustered_baseline"

    def test_pairwise_solvers_are_fixed_horizon_only(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                0.1,
                -0.1,
                0.4,
                0.4,
                b,
                make_procedure(inference=GaussianScoreMixture(), population="assigned"),
            )
        assert exc_info.value.code == "power.supports_fixed_horizon"
        assert exc_info.value.context["caller"] == "segment_pairwise_required_sample_size"

    def test_pairwise_required_size_refuses_a_shifted_null(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                0.1, -0.1, 0.4, 0.4, b, make_procedure(null_lift=0.05)
            )
        assert exc_info.value.code == "power.segment_pairwise_required"
        assert exc_info.value.context["null_lift"] == 0.05

    def test_pairwise_required_size_refuses_a_below_r_b_with_greater(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                -0.1, 0.1, 0.4, 0.4, b, make_procedure(alternative="greater")
            )
        assert exc_info.value.code == "power.r_a_below"

    def test_pairwise_required_size_refuses_a_above_r_b_with_less(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(
                0.1, -0.1, 0.4, 0.4, b, make_procedure(alternative="less")
            )
        assert exc_info.value.code == "power.r_a_above"

    def test_pairwise_required_size_refuses_equal_r_a_and_r_b(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(0.1, 0.1, 0.4, 0.4, b, make_procedure())
        assert exc_info.value.code == "power.r_a_r"

    def test_pairwise_required_size_refuses_a_solved_n_too_small_to_split(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_required_sample_size(5.0, -0.9, 0.02, 0.02, b, make_procedure())
        assert exc_info.value.code == "power.solved_too_small"
        assert exc_info.value.context["q_a"] == 0.02

    def test_pairwise_achieved_power_refuses_a_shifted_null(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_achieved_power(
                100, 0.1, -0.1, 0.4, 0.4, b, make_procedure(null_lift=0.05)
            )
        assert exc_info.value.code == "power.segment_pairwise_achieved"

    def test_pairwise_mde_refuses_a_shifted_null(self):
        b = Baseline(mean=1.0, var=1.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_minimum_detectable_effect(
                100, 0.4, 0.4, b, make_procedure(null_lift=0.05)
            )
        assert exc_info.value.code == "power.segment_pairwise_minimum"

    def test_pairwise_mde_refuses_an_n_per_arm_too_small_to_split(self):
        # 78 units in total leave segment A (5%) 3.9 units, short of two per arm.
        b = Baseline(mean=1.0, var=1.0)
        procedure = make_procedure()
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_pairwise_minimum_detectable_effect(39, 0.05, 0.5, b, procedure)
        assert exc_info.value.code == "power.segment_pairwise_achieved_n_per_arm_too_small"
        assert exc_info.value.context["n_per_arm"] == 39

    def test_joint_q_inputs_must_be_one_dimensional(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(np.zeros(3), np.ones((2, 2)), 0.05)
        assert exc_info.value.code == "power.theta_var_one"

    def test_joint_q_inputs_must_share_a_shape(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(np.zeros(3), np.zeros(2), 0.05)
        assert exc_info.value.code == "power.theta_var_same"

    def test_joint_q_needs_at_least_two_segments(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(np.zeros(1), np.ones(1), 0.05)
        assert exc_info.value.code == "power.need_least_segments"
        assert exc_info.value.context["k"] == 1

    def test_joint_q_var_must_be_finite_and_positive(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(np.zeros(3), np.array([1.0, -1.0, 1.0]), 0.05)
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_joint_q_theta_must_be_finite(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(np.array([1.0, np.nan, 1.0]), np.ones(3), 0.05)
        assert exc_info.value.code == "power.theta_contains_non"

    def test_joint_q_alpha_must_be_in_unit_interval(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            joint_q_power_fixed(np.zeros(3), np.ones(3), 1.5)
        assert exc_info.value.code == "estimation.diagnostics.alpha"
        assert exc_info.value.context["alpha"] == 1.5


def _fixture_analysis_with_control_values(
    metric_name: str, values: np.ndarray, *, quantile: float
) -> Analysis:
    """A dataframe-route Analysis with real per-unit control values for
    one quantile metric -- the route QuantileBaseline's factory reads."""
    n = values.size
    df = pd.DataFrame(
        {"unit_id": range(1, n + 1), "group_id": ["control"] * n, metric_name: values}
    )
    return Analysis.from_unit_summary(
        df,
        unit="unit_id",
        group="group_id",
        control="control",
        metrics=[{"name": metric_name, "type": "quantile", "quantile": quantile}],
    )


def _quantile_baseline(name: str, q: float, values: np.ndarray) -> QuantileBaseline:
    metric = synthesise_metric(MetricSpec(name=name, type="quantile", quantile=q))
    return QuantileBaseline.from_control_values(cast("QuantileMetric", metric), values)


class TestQuantileBaseline:
    """A mismatch (reusing one metric's baseline to plan another) must be
    visible in the answer, not silent; cuped_rho has no runtime route to
    credit for a quantile metric."""

    def test_planning_does_not_change_equality(self):
        """Baselines compare by what they describe, not by what has been
        computed from them."""
        import copy
        import pickle

        pilot = np.round(np.random.default_rng(3).lognormal(5.0, 0.6, 5000))
        first = _quantile_baseline("latency", 0.5, pilot)
        second = _quantile_baseline("latency", 0.5, pilot)
        procedure = ArmPlanningProcedure.standard("quantile")
        achieved_power(20_000, 0.05, first, procedure)
        assert first == second
        achieved_power(20_000, 0.05, second, procedure)
        assert first == second
        assert first != _quantile_baseline("latency", 0.9, pilot)
        for clone in (copy.deepcopy(first), pickle.loads(pickle.dumps(first))):
            assert clone == first
            assert achieved_power(20_000, 0.05, clone, procedure) == achieved_power(
                20_000, 0.05, first, procedure
            )

    def test_stores_the_metric_it_was_built_for(self):
        pilot = np.round(np.random.default_rng(1).lognormal(1.0, 0.5, 2000), 2)
        baseline = _quantile_baseline("checkout_latency", 0.5, pilot)
        assert baseline.metric_name == "checkout_latency"
        assert baseline.quantile_q == 0.5
        assert baseline.mean > 0.0
        assert baseline.var > 0.0

    def test_cuped_rho_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            QuantileBaseline(
                metric_name="checkout_latency", quantile_q=0.5, mean=1.0, var=1.0, cuped_rho=0.5
            )
        assert exc_info.value.code == "power.quantile_baseline.cuped_unsupported"

    def test_factory_refuses_a_metric_that_is_not_a_quantile(self):
        metric = synthesise_metric(MetricSpec(name="revenue", type="mean"))
        with pytest.raises(InvalidRequestError) as exc_info:
            QuantileBaseline.from_control_values(metric, np.arange(1.0, 101.0))  # ty: ignore[invalid-argument-type]
        assert exc_info.value.code == "power.quantile_baseline.metric_not_quantile"
        assert exc_info.value.context["metric"] == "revenue"

    def test_a_quantile_baseline_planned_under_a_different_metric_type_is_refused(self):
        pilot = np.round(np.random.default_rng(2).lognormal(5.0, 0.6, 5000))
        baseline = _quantile_baseline("checkout_latency_ms", 0.5, pilot)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.1, baseline, ArmPlanningProcedure.standard("ratio"))
        assert exc_info.value.code == "power.quantile_baseline.metric_not_quantile"
        assert exc_info.value.context["metric_type"] == "ratio"

    def test_required_sample_size_echoes_the_planned_metric_and_quantile(self):
        pilot = np.round(np.random.default_rng(2).lognormal(5.0, 0.6, 5000))
        baseline = _quantile_baseline("checkout_latency_ms", 0.5, pilot)
        procedure = make_procedure(
            metric_type="quantile",
            identification="randomized",
            population="assigned",
            variance_adjustment="none",
        )
        result = required_sample_size(0.1, baseline, procedure, PowerDesign())
        assert result.planned_metric_name == "checkout_latency_ms"
        assert result.planned_quantile == 0.5
        assert result.planned_for == "planned for metric 'checkout_latency_ms' at quantile 0.5"

    def test_non_quantile_baseline_leaves_planned_fields_none(self):
        b = Baseline(mean=1.0, var=1.0)
        result = required_sample_size(0.1, b, make_procedure())
        assert result.planned_metric_name is None
        assert result.planned_quantile is None
        assert result.planned_for is None


class TestDeterministicPlanningSE:
    """Planning's deterministic projection of the runtime's own
    order-statistic construction must agree with the
    runtime's own measured SE, and must refuse rather than silently size
    a design the runtime's own construction could not answer at that n."""

    def test_matches_measured_runtime_se_at_the_candidate_n_through_planning_baseline(self):
        """Quantities computed twice (AGENTS.md): planning's SE at a
        candidate n, from pilots reached through the real
        Analysis.planning_baseline route, is unbiased for the runtime's own
        SE at that n on the same population. A projection from one pilot
        carries that pilot's sampling error, so the mean over K independent
        pilots is compared with the runtime's Monte Carlo mean, within three
        combined standard errors, both measured here."""
        rng = np.random.default_rng(3)
        n_pilots = 12
        planned = np.empty(n_pilots)
        for i in range(n_pilots):
            analysis = _fixture_analysis_with_control_values(
                "checkout_latency_ms", np.round(rng.lognormal(5.0, 0.6, 5000)), quantile=0.5
            )
            baseline = analysis.planning_baseline("checkout_latency_ms")
            assert isinstance(baseline, QuantileBaseline)
            planned[i] = _deterministic_quantile_se(
                np.asarray(baseline._pilot_sorted), q=0.5, alpha=0.05, n_candidate=100000
            )
        rng2 = np.random.default_rng(9)
        n_reps = 30
        runtime_ses = np.empty(n_reps)
        for i in range(n_reps):
            y = np.sort(np.round(rng2.lognormal(5.0, 0.6, 100000)))
            _, runtime_ses[i] = log_quantile_se(y, 0.5)
        sem_planned = planned.std(ddof=1) / math.sqrt(n_pilots)
        sem_runtime = runtime_ses.std(ddof=1) / math.sqrt(n_reps)
        assert abs(planned.mean() - runtime_ses.mean()) <= 3 * math.hypot(sem_planned, sem_runtime)

    def test_refuses_rather_than_sizes_an_infeasible_bracket_at_a_candidate_n(self):
        """A candidate n below the closed-form feasibility floor
        n_min(q, alpha) refuses by the SAME code log_quantile_se's own
        too-small-bound refusal uses -- reused, not reimplemented, so the
        two refusals can never disagree."""
        rng = np.random.default_rng(6)
        pilot_values = np.round(rng.lognormal(5.0, 0.6, 2000))
        analysis = _fixture_analysis_with_control_values(
            "checkout_latency_p99_ms", pilot_values, quantile=0.99
        )
        baseline = analysis.planning_baseline("checkout_latency_p99_ms")
        assert isinstance(baseline, QuantileBaseline)
        n_min = _quantile_n_min(0.99, 0.05)
        assert n_min == 368
        with pytest.raises(InvalidRequestError) as exc_info:
            _deterministic_quantile_se(np.sort(pilot_values), q=0.99, alpha=0.05, n_candidate=200)
        assert exc_info.value.code == "estimation.quantile.too_small_bound"
        assert exc_info.value.context["n_min"] == n_min

    def test_search_does_not_refuse_a_feasible_design_probed_below_n_min(self):
        """A search seeded below n_min by a large MDE must still find a
        feasible n, not refuse merely because its first probe was
        infeasible."""
        rng = np.random.default_rng(7)
        pilot_values = np.round(rng.lognormal(5.0, 0.6, 2000))
        analysis = _fixture_analysis_with_control_values(
            "checkout_latency_p99_ms", pilot_values, quantile=0.99
        )
        baseline = analysis.planning_baseline("checkout_latency_p99_ms")
        assert isinstance(baseline, QuantileBaseline)
        procedure = make_procedure(
            metric_type="quantile",
            identification="randomized",
            population="assigned",
            variance_adjustment="none",
        )
        result = required_sample_size(0.50, baseline, procedure, PowerDesign(power=0.8))
        assert result.n_per_arm >= _quantile_n_min(0.99, procedure.compiled_alpha)

    def test_search_refuses_a_target_above_the_recording_grid_ceiling_and_sizes_one_below(self):
        """On rounded data the SE falls to a floor fixed by the recording
        step, so power has a ceiling below one. A target above it is refused
        naming the ceiling; a target just under it is sized, however large."""
        pilot = np.round(np.random.default_rng(3).lognormal(5.0, 0.6, 5000))
        baseline = _quantile_baseline("checkout_latency_ms", 0.5, pilot)
        procedure = ArmPlanningProcedure.standard("quantile")
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.01, baseline, procedure, PowerDesign(power=0.8))
        assert exc_info.value.code == "power.quantile_size_search_unreachable"
        assert exc_info.value.context["limiting_condition"] == "recording_grid"
        ceiling = cast("float", exc_info.value.context["maximum_power"])
        assert 0.05 < ceiling < 0.8
        target = ceiling - 0.02
        result = required_sample_size(0.01, baseline, procedure, PowerDesign(power=target))
        assert result.power >= target

    def test_search_sizes_a_small_lift_on_continuous_data(self):
        """Continuous data has no SE floor, so any target power is reachable;
        the answer is the smallest arm reaching it."""
        pilot = np.random.default_rng(0).lognormal(5.0, 0.6, 5000)
        baseline = _quantile_baseline("checkout_latency_ms", 0.5, pilot)
        procedure = ArmPlanningProcedure.standard("quantile")
        result = required_sample_size(0.01, baseline, procedure)
        assert result.power >= 0.8
        assert achieved_power(result.n_per_arm - 1, 0.01, baseline, procedure).power < 0.8

    def test_search_finds_the_smallest_size_before_the_power_dip_at_resolution_loss(self):
        """Where an arm's bracket stops spanning enough repeated values, the
        runtime switches to its wider tied interval and power drops. A
        target reached just before that size is sized there, not past the
        dip, though the doubling steps straddle it."""
        pilot = np.round(np.random.default_rng(4).lognormal(1.0, 0.5, 2000), 2)
        baseline = _quantile_baseline("basket_value", 0.5, pilot)
        procedure = ArmPlanningProcedure.standard("quantile")
        projection = baseline._projection(procedure.compiled_alpha)
        last = max(n for n in range(20, 20_000) if projection.arm_at(n).resolved)
        target = achieved_power(last, 0.04, baseline, procedure).power - 0.005
        assert achieved_power(last + 1, 0.04, baseline, procedure).power < target
        result = required_sample_size(0.04, baseline, procedure, PowerDesign(power=target))
        assert result.n_per_arm <= last
        assert result.power >= target
        assert achieved_power(result.n_per_arm - 1, 0.04, baseline, procedure).power < target

    @pytest.mark.parametrize(
        ("draw", "q"),
        [
            pytest.param(lambda rng: rng.lognormal(5.0, 0.6, 1000), 0.99, id="continuous-p99"),
            pytest.param(
                lambda rng: np.round(rng.lognormal(5.0, 0.6, 5000)), 0.5, id="tied-unresolved"
            ),
            pytest.param(
                lambda rng: np.round(rng.lognormal(5.0, 0.6, 30_000)), 0.9, id="tied-resolved"
            ),
        ],
    )
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10])
    def test_matches_the_runtime_se_on_the_pilot_at_its_own_size(self, draw, q, alpha):
        """Quantities computed twice (AGENTS.md): at the pilot's own size the
        projection is the runtime's construction on the pilot, bit for bit."""
        pilot = np.sort(draw(np.random.default_rng(0)))
        _, runtime_se, _ = _log_quantile_se_impl(pilot, q, alpha)
        assert _deterministic_quantile_se(pilot, q, alpha, pilot.size) == runtime_se

    def test_a_pilot_bracket_without_ties_projects_the_classical_se(self):
        """The runtime widens a bracket only when it holds a tie, so on
        continuous data planning projects the pilot's classical SE alone,
        scaled by the asymptotic sqrt(m / n) rate."""
        pilot = np.sort(np.random.default_rng(12).lognormal(5.0, 0.6, 1000))
        _, _, classical = _log_quantile_se_impl(pilot, 0.99, 0.05)
        planned = _deterministic_quantile_se(pilot, 0.99, 0.05, 1_000_000)
        assert planned == pytest.approx(classical * math.sqrt(1000 / 1_000_000), rel=1e-12)

    def test_refuses_a_collapsed_pilot_bracket_instead_of_projecting_zero_variance(self):
        """A pilot whose bracket order statistics collapse to one value
        (the runtime's own construction refuses this as a degenerate
        spread) must refuse by the SAME code, not silently project a
        zero standard error that surfaces downstream as an opaque
        zero-variance baseline refusal."""
        pilot = np.full(2000, 5.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            _deterministic_quantile_se(pilot, q=0.5, alpha=0.05, n_candidate=100000)
        assert exc_info.value.code == "estimation.quantile.degenerate_spread_order"

    def test_baseline_var_is_the_per_unit_variance_behind_the_runtime_se(self):
        """``var`` is a per-unit variance: over the pilot's size, and on the
        log scale, it is the runtime's squared SE on the pilot."""
        pilot = np.random.default_rng(0).lognormal(5.0, 0.6, 5000)
        baseline = _quantile_baseline("checkout_latency_ms", 0.5, pilot)
        _, runtime_se = log_quantile_se(np.sort(pilot), 0.5, 0.05)
        per_n = baseline.var / baseline.mean**2 / pilot.size
        assert per_n == pytest.approx(runtime_se**2, rel=1e-12)


def _price_pilot(seed: int, m: int = 500) -> np.ndarray:
    """Poisson(5) counts with half the positive values recorded 0.01 lower:
    a .99/.00 price grid whose recording cells alternate between a hundredth
    and nearly a whole unit wide."""
    rng = np.random.default_rng(seed)
    y = rng.poisson(5, m).astype(float)
    y[(y > 0) & (rng.random(y.size) < 0.5)] -= 0.01
    return y


def _planned_power(baseline: QuantileBaseline, procedure: Any, relative_lift: float):
    """``achieved_power(n, ...).power`` as a function of ``n`` without the
    companion effect search, so a scan over thousands of sizes is
    affordable; ``None`` where the projected bracket reaches a non-positive
    value and the projection refuses, as the readout would."""
    prepared, plan_baseline = _prepare_solver(procedure, baseline)
    theta, distance = _supplied_effect(prepared, plan_baseline, relative_lift, bounded=False)

    def power(n: int) -> float | None:
        n_t, n_c = _compute_arms(n, PowerDesign(), minimum_per_arm=20)
        try:
            return _ArmPlan(prepared, plan_baseline, n_t, n_c, None, False).power(distance, theta)
        except InvalidRequestError as refusal:
            if refusal.code != "estimation.quantile.quantile_positive_log":
                raise
            return None

    return power


def _adjacent_half_width_bound(projection: Any, n: int) -> float:
    """Upper bound on the change of the projection's averaged half-width
    between sizes ``n`` and ``n + 1``, from its own integrand. Each bracket
    end's read moves by the change of its rank probability and carries, across
    each change of recorded value, the normal mass between the cut's old and
    new place times that read's own jump in the half-width. The kernel's
    change of width moves at most half the integrand's range times the L1
    distance between the two normal densities, a moving support bound
    re-reads at most the range over the mass between its old and new place,
    and the arm facts contribute their own drift."""
    q, m = projection.arm.q, projection.pilot.size
    arms = projection.arm_at(n), projection.arm_at(n + 1)
    shifts = projection.shift(n), projection.shift(n + 1)
    windows = [
        (max(lo, -_SHIFT_SPAN * sd), min(hi, _SHIFT_SPAN * sd)) for _, _, sd, lo, hi in shifts
    ]

    def half(arm: Any, low: float, up: float, point: float) -> float:
        return projection._half_width(replace(arm, point=point), low, up)

    def midpoints(lower_p: float, upper_p: float, lo: float, hi: float) -> np.ndarray:
        cuts = np.unique(
            np.concatenate([projection._cuts(p, lo, hi) for p in (lower_p, upper_p, q)])
        )
        return (np.concatenate(([lo], cuts)) + np.concatenate((cuts, [hi]))) / 2.0

    values = []
    lows = []
    for arm, (lower_p, upper_p, _, _, _), (lo, hi) in zip(arms, shifts, windows, strict=True):
        for s in midpoints(lower_p, upper_p, lo, hi):
            low = projection._recorded(lower_p + s)
            lows.append(low)
            values.append(
                half(arm, low, projection._recorded(upper_p + s), projection._recorded(q + s))
            )
    drift = abs(arms[1].floor - arms[0].floor)
    if not arms[0].resolved:
        # The allowance is at most a quarter cell, and at most a quarter of
        # the bracket, so each end's log moves by at most its change over 3/4
        # of the smallest lower end.
        drift += abs(arms[1].cell - arms[0].cell) / (3.0 * min(lows))
    span = max(values) - min(values) + drift

    (lower0, upper0, sd0, _, _), (lower1, upper1, sd1, _, _) = shifts
    (lo0, hi0), (lo1, hi1) = windows
    bound = drift
    step = 1e-9 / m
    for read, (p0, p1) in enumerate(((lower0, lower1), (upper0, upper1))):
        delta = p1 - p0
        # The lower read moves first, so the upper read's cuts see it moved.
        other_p = upper0 if read == 0 else lower1
        for cut in projection._cuts(p0, min(lo0, lo1), max(hi0, hi1)):
            before, after = (projection._recorded(p0 + cut + sign * step) for sign in (-1, 1))
            # Over the swept interval the other reads may change too: split it there.
            u, v = sorted((cut, cut - delta))
            inner = np.unique(np.concatenate([projection._cuts(p, u, v) for p in (other_p, q)]))
            for a_, b_ in zip((u, *inner), (*inner, v), strict=True):
                mass = norm.cdf(b_ / sd0) - norm.cdf(a_ / sd0)
                point = projection._recorded(q + (a_ + b_) / 2.0)
                other = projection._recorded(other_p + (a_ + b_) / 2.0)
                if min(before, other) <= 0.0:
                    jump = span  # a cut at the edge of the positive support
                elif read == 0:
                    jump = half(arms[0], after, other, point) - half(arms[0], before, other, point)
                else:
                    jump = half(arms[0], other, after, point) - half(arms[0], other, before, point)
                bound += mass * abs(jump)
    small, big = sorted((sd0, sd1))
    if big > small:
        crossing = small * big * math.sqrt(2.0 * math.log(big / small) / (big**2 - small**2))
        bound += 0.5 * span * 4.0 * (norm.cdf(crossing / small) - norm.cdf(crossing / big))
    for c0, c1 in ((lo0, lo1), (hi0, hi1)):
        bound += span * abs(norm.cdf(c1 / sd1) - norm.cdf(c0 / sd1))
    return bound + 4.0 * len(values) * sys.float_info.epsilon * max(values)


class TestQuantileProjectionContinuity:
    """The shift average is integrated exactly over its piecewise-constant
    integrand, so the planned half-width changes between adjacent sizes by
    no more than the mass its reads can carry across a change of recorded
    value. A sixteen-node quadrature rule moved a node's whole weight
    (0.29) across a cell at once: on a .99/.00 price pilot the planned SE
    fell from 0.0249 to 0.0072 and power for a 5% lift rose from 0.28 to
    1.00 between n=2172 and n=2173."""

    @staticmethod
    def _assert_continuous(projection: Any, sizes: range) -> None:
        m = projection.pilot.size
        for n in sizes:
            if n in (m - 1, m):
                continue  # at the pilot's size the SE is the runtime's own, not an average
            if projection.arm_at(n).resolved != projection.arm_at(n + 1).resolved:
                continue  # the runtime's own switch between its two half-width rules
            h0, h1 = projection.se(n) * projection.z, projection.se(n + 1) * projection.z
            bound = _adjacent_half_width_bound(projection, n)
            assert abs(h1 - h0) <= bound, (n, h0, h1, bound)

    @pytest.mark.slow
    @pytest.mark.parametrize("seed", [12, 11])
    def test_price_grid_half_width_moves_within_its_integrand_bound(self, seed):
        baseline = _quantile_baseline("price", 0.5, _price_pilot(seed))
        projection = baseline._projection(ArmPlanningProcedure.standard("quantile").compiled_alpha)
        for sizes in (range(20, 140), range(440, 560), range(2150, 2200), range(3300, 3430)):
            self._assert_continuous(projection, sizes)

    def test_millisecond_grid_half_width_moves_within_its_integrand_bound(self):
        pilot = np.round(np.random.default_rng(3).lognormal(5.0, 0.6, 5000))
        baseline = _quantile_baseline("checkout_latency_ms", 0.5, pilot)
        projection = baseline._projection(ArmPlanningProcedure.standard("quantile").compiled_alpha)
        for sizes in (range(4985, 5015), range(41490, 41510)):
            self._assert_continuous(projection, sizes)

    def test_planned_power_no_longer_jumps_between_adjacent_sizes(self):
        """Power is a smooth function of the two arms' SEs, so its change
        between adjacent sizes is bounded by the SE change times the
        noncentrality's sensitivity, which the integrand bound caps."""
        procedure = ArmPlanningProcedure.standard("quantile")
        cases = (
            (_quantile_baseline("price", 0.5, _price_pilot(12)), 0.05, 2172),
            (
                _quantile_baseline(
                    "checkout_latency_ms",
                    0.5,
                    np.round(np.random.default_rng(3).lognormal(5.0, 0.6, 5000)),
                ),
                0.02,
                41500,
            ),
        )
        for baseline, lift, start in cases:
            projection = baseline._projection(procedure.compiled_alpha)
            for n in range(start, start + 2):
                se0, se1 = projection.se(n), projection.se(n + 1)
                p0 = achieved_power(n, lift, baseline, procedure).power
                p1 = achieved_power(n + 1, lift, baseline, procedure).power
                # Two equal arms: nc = log1p(lift) / (sqrt(2) se), and the
                # two-sided normal power's slope in nc is at most phi(0).
                slope = norm.pdf(0.0) * math.log1p(lift) / math.sqrt(2.0) / (se0 * se1)
                allowed = slope * _adjacent_half_width_bound(projection, n) / projection.z
                assert abs(p1 - p0) <= allowed, (n, p0, p1, allowed)


class TestQuantileSizeSearchOnAPriceGrid:
    """On a .99/.00 price pilot the projected power used to jump between
    adjacent sizes and the search bisected across the jumps: for a 5% lift
    it returned n=2173, 1,695 sizes past the first that reached the target,
    and for a 0.5% lift it refused a reachable target naming a maximum
    power of 0.32 where its own model reached 0.93."""

    _target = 0.8

    @pytest.mark.slow
    def test_answer_is_the_first_size_reaching_the_target_up_to_the_rank_waver(self):
        baseline = _quantile_baseline("price", 0.5, _price_pilot(12))
        procedure = ArmPlanningProcedure.standard("quantile")
        power = _planned_power(baseline, procedure, 0.05)
        result = required_sample_size(0.05, baseline, procedure, PowerDesign(power=self._target))
        n = result.n_per_arm
        assert power(n) == result.power
        assert result.power >= self._target
        assert achieved_power(n - 1, 0.05, baseline, procedure).power < self._target
        # Below the answer, power alternates with the bracket ranks' rounding:
        # no smaller size reaches the target together with both neighbours.
        powers = [power(k) or 0.0 for k in range(20, n + 1)]
        assert not any(
            min(powers[i - 1], powers[i], powers[i + 1]) >= self._target
            for i in range(1, len(powers) - 2)
        )

    @pytest.mark.slow
    def test_a_reachable_target_below_the_ceiling_is_sized_not_refused(self):
        baseline = _quantile_baseline("price", 0.5, _price_pilot(12))
        procedure = ArmPlanningProcedure.standard("quantile")
        result = required_sample_size(0.005, baseline, procedure, PowerDesign(power=self._target))
        assert result.power >= self._target
        assert achieved_power(result.n_per_arm - 1, 0.005, baseline, procedure).power < self._target

    @pytest.mark.slow
    def test_refusal_names_a_maximum_power_no_size_beats(self):
        baseline = _quantile_baseline("price", 0.5, _price_pilot(12))
        procedure = ArmPlanningProcedure.standard("quantile")
        power = _planned_power(baseline, procedure, 0.005)
        with pytest.raises(InvalidRequestError) as exc_info:
            required_sample_size(0.005, baseline, procedure, PowerDesign(power=0.99))
        assert exc_info.value.code == "power.quantile_size_search_unreachable"
        assert exc_info.value.context["limiting_condition"] == "recording_grid"
        maximum_power = cast("float", exc_info.value.context["maximum_power"])
        assert maximum_power >= max(power(k) or 0.0 for k in range(20, 3000))
        assert maximum_power >= achieved_power(478, 0.005, baseline, procedure).power


_LATENCY_PILOT = np.random.default_rng(5).lognormal(5.0, 0.6, 5000)


class TestQuantilePlanningMatchesReadoutRefusals:
    """Planning refuses what the quantile readout refuses, by the readout's
    own code, so no plan is sized for a test that cannot be read out."""

    def test_planning_refuses_tiny_quantile_with_runtime_size_code(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            _quantile_baseline("tiny_quantile", 1e-20, np.linspace(1.0, 2.0, 100))
        assert exc_info.value.code == "estimation.quantile.too_small_bound"
        assert exc_info.value.context["n_min"] == 368887945411393644965

    @pytest.mark.parametrize(
        "solve",
        [
            pytest.param(
                lambda b, p: segment_pairwise_required_sample_size(0.05, 0.0, 0.5, 0.5, b, p),
                id="segment_pairwise_required_sample_size",
            ),
            pytest.param(
                lambda b, p: segment_pairwise_achieved_power(400, 0.05, 0.0, 0.5, 0.5, b, p),
                id="segment_pairwise_achieved_power",
            ),
            pytest.param(
                lambda b, p: segment_pairwise_minimum_detectable_effect(400, 0.5, 0.5, b, p),
                id="segment_pairwise_minimum_detectable_effect",
            ),
        ],
    )
    def test_a_segment_contrast_is_refused_as_a_quantile_breakout(self, solve, request):
        baseline = _quantile_baseline("p50_latency_ms", 0.5, _LATENCY_PILOT)
        with pytest.raises(CapabilityError) as exc_info:
            solve(baseline, ArmPlanningProcedure.standard("quantile"))
        assert exc_info.value.code == "readout.metric.quantile_breakout"
        assert exc_info.value.context["metric"] == "p50_latency_ms"
        assert exc_info.value.context["solver"] == request.node.callspec.id

    @pytest.mark.parametrize(
        ("procedure", "relative_lift"),
        [
            pytest.param(
                ArmPlanningProcedure.standard(
                    "quantile", role="guardrail", preferred_direction="decrease"
                ),
                -0.05,
                id="guardrail",
            ),
            pytest.param(
                ArmPlanningProcedure.standard("quantile", alternative="greater"),
                0.05,
                id="one-sided",
            ),
            pytest.param(
                ArmPlanningProcedure.standard("quantile", null_lift=0.02), 0.05, id="shifted-null"
            ),
            pytest.param(
                ArmPlanningProcedure.standard(role="guardrail", preferred_direction="decrease"),
                -0.05,
                id="metric-type-from-baseline",
            ),
        ],
    )
    def test_a_test_other_than_two_sided_at_a_zero_null_is_refused(self, procedure, relative_lift):
        baseline = _quantile_baseline("p90_latency_ms", 0.9, _LATENCY_PILOT)
        with pytest.raises(UnsupportedRequestError) as exc_info:
            required_sample_size(relative_lift, baseline, procedure)
        assert exc_info.value.code == "readout.metric.quantile_alternative"
        assert exc_info.value.context["metric"] == "p90_latency_ms"

    def test_planning_and_the_readout_refuse_a_one_sided_quantile_test_by_one_code(self):
        from increment import readouts
        from increment.frame import FrameTotalsSource
        from increment.semantics.models import AnalysisPlan

        frame = pd.DataFrame(
            {
                "user_id": range(_LATENCY_PILOT.size),
                "variant": ["control", "treatment"] * (_LATENCY_PILOT.size // 2),
                "lat": _LATENCY_PILOT,
            }
        )
        source = FrameTotalsSource.from_frame(
            frame,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="lat", type="quantile", quantile=0.9)],
            plan=AnalysisPlan(alternative="greater"),
        )
        with pytest.raises(UnsupportedRequestError) as readout:
            readouts.run(source)
        baseline = _quantile_baseline("lat", 0.9, _LATENCY_PILOT[::2])
        with pytest.raises(UnsupportedRequestError) as planning:
            required_sample_size(
                0.05, baseline, ArmPlanningProcedure.standard("quantile", alternative="greater")
            )
        assert planning.value.code == readout.value.code
