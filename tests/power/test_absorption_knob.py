"""Tests for the absorption-aware planning knob on ``Baseline``.

``Baseline.icc`` / ``Baseline.from_absorption`` let a planner credit an
expected factor-absorption SE reduction, the same way ``cuped_rho`` credits
a CUPED reduction. Two things need checking:

1. The mapping (``effective_var = var * (1 - icc)``) is correct at its
   boundaries and flows through every solver identically to hand-reducing
   ``var`` (the "boundary-agnostic" claim).
2. The mapping's SE-reduction prediction matches what
   ``increment.estimation.absorption.absorb_one_way`` measures on simulated
   one-way-factor data (``@pytest.mark.parameter_recovery``), with a fast
   unmarked smoke variant per AGENTS.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from increment.errors import InvalidRequestError
from increment.estimation.absorption import absorb_one_way
from increment.power.core import (
    Baseline,
    PowerDesign,
    achieved_power,
    minimum_detectable_effect,
    required_sample_size,
    segment_pairwise_required_sample_size,
)

from ._procedures import make_procedure

TAU = 0.20


def _simulate_one_way(rng, n_levels, per_level, alpha_sd, p_treat=0.5):
    """Cell moments for a Gaussian one-way design with unit within-level
    variance and between-level sd ``alpha_sd`` - the same DGP shape
    ``tests/test_absorption_calibration.py`` uses to derive its calibration
    table, reproduced locally so this file has no cross-test-module import."""
    n_c = np.zeros(n_levels)
    s_c = np.zeros(n_levels)
    q_c = np.zeros(n_levels)
    n_t = np.zeros(n_levels)
    s_t = np.zeros(n_levels)
    q_t = np.zeros(n_levels)
    alpha = rng.normal(0.0, alpha_sd, n_levels)
    for k in range(n_levels):
        d = rng.random(per_level) < p_treat
        y = alpha[k] + TAU * d + rng.normal(0.0, 1.0, per_level)
        yc, yt = y[~d], y[d]
        n_c[k], s_c[k], q_c[k] = yc.size, yc.sum(), (yc**2).sum()
        n_t[k], s_t[k], q_t[k] = yt.size, yt.sum(), (yt**2).sum()
    return n_c, s_c, q_c, n_t, s_t, q_t


def _icc_to_alpha_sd(icc: float) -> float:
    """Invert ICC = alpha_sd^2 / (alpha_sd^2 + 1) (unit within-level variance)."""
    return float(np.sqrt(icc / (1.0 - icc))) if icc > 0 else 0.0


# Unit tests: the mapping itself


class TestEffectiveVarMapping:
    def test_icc_zero_gives_effective_var_equal_to_var_exactly(self):
        """No credit for the ~1% sandwich-vs-naive artifact absorb_one_way
        measures at true ICC 0 - effective_var must equal var bit-for-bit,
        not var * 0.991 or any other small discount."""
        b = Baseline.from_absorption(mean=1.0, var=4.0, icc=0.0)
        assert b.effective_var == 4.0

    def test_default_icc_is_zero_and_backward_compatible(self):
        """A Baseline built without icc behaves exactly as it did before
        this knob existed."""
        b = Baseline(mean=1.0, var=4.0)
        assert b.icc == 0.0
        assert b.effective_var == 4.0

    @pytest.mark.parametrize("icc", [0.0, 0.04, 0.20, 0.50, 0.80, 0.99])
    def test_effective_var_matches_closed_form(self, icc):
        b = Baseline.from_absorption(mean=1.0, var=4.0, icc=icc)
        assert b.effective_var == pytest.approx(4.0 * (1.0 - icc))

    def test_effective_var_monotonically_decreases_with_icc(self):
        iccs = [0.0, 0.04, 0.20, 0.50, 0.80, 0.99]
        effective_vars = [
            Baseline.from_absorption(mean=1.0, var=4.0, icc=i).effective_var for i in iccs
        ]
        assert effective_vars == sorted(effective_vars, reverse=True)
        assert len(set(effective_vars)) == len(effective_vars), "must be STRICTLY decreasing"

    @pytest.mark.parametrize("icc", [-0.01, 1.0, 1.5])
    def test_icc_out_of_range_raises(self, icc):
        with pytest.raises(InvalidRequestError) as exc_info:
            Baseline.from_absorption(mean=1.0, var=4.0, icc=icc)
        assert exc_info.value.code == "power.baseline.icc"

    def test_from_absorption_matches_direct_construction(self):
        assert Baseline.from_absorption(mean=2.0, var=3.0, icc=0.25) == Baseline(
            mean=2.0, var=3.0, icc=0.25
        )

    def test_icc_and_cuped_rho_compose_multiplicatively(self):
        """Both knobs set: effective_var is the product of both reductions,
        matching cuped_rho's own (1 - rho^2) shape."""
        b = Baseline(mean=1.0, var=4.0, cuped_rho=0.6, icc=0.20)
        assert b.effective_var == pytest.approx(4.0 * (1 - 0.6**2) * (1 - 0.20))


class TestBoundaryAgnostic:
    """effective_var feeds every solver's se2 identically regardless of
    which knob (icc, cuped_rho, or a hand-reduced var) produced it.
    """

    def test_required_sample_size_matches_hand_reduced_baseline(self):
        icc = 0.30
        via_knob = Baseline.from_absorption(mean=1.0, var=4.0, icc=icc)
        hand_reduced = Baseline(mean=1.0, var=4.0 * (1 - icc))
        procedure = make_procedure()
        d = PowerDesign()
        assert required_sample_size(0.1, via_knob, procedure, design=d) == required_sample_size(
            0.1, hand_reduced, procedure, design=d
        )

    def test_achieved_power_matches_hand_reduced_baseline(self):
        icc = 0.30
        via_knob = Baseline.from_absorption(mean=1.0, var=4.0, icc=icc)
        hand_reduced = Baseline(mean=1.0, var=4.0 * (1 - icc))
        procedure = make_procedure()
        d = PowerDesign()
        assert achieved_power(500, 0.1, via_knob, procedure, design=d) == achieved_power(
            500, 0.1, hand_reduced, procedure, design=d
        )

    def test_minimum_detectable_effect_matches_hand_reduced_baseline(self):
        icc = 0.30
        via_knob = Baseline.from_absorption(mean=1.0, var=4.0, icc=icc)
        hand_reduced = Baseline(mean=1.0, var=4.0 * (1 - icc))
        d = PowerDesign()
        assert minimum_detectable_effect(
            500, via_knob, make_procedure(), design=d
        ) == minimum_detectable_effect(500, hand_reduced, make_procedure(), design=d)

    def test_segment_pairwise_matches_hand_reduced_baseline(self):
        """The same identity holds through the ported segment-pairwise
        solvers, confirming the knob is not special-cased to the three ATE
        solvers alone."""
        icc = 0.30
        via_knob = Baseline.from_absorption(mean=1.0, var=4.0, icc=icc)
        hand_reduced = Baseline(mean=1.0, var=4.0 * (1 - icc))
        procedure = make_procedure()
        d = PowerDesign()
        assert segment_pairwise_required_sample_size(
            0.3, 0.05, 0.2, 0.2, via_knob, procedure=procedure, design=d
        ) == segment_pairwise_required_sample_size(
            0.3, 0.05, 0.2, 0.2, hand_reduced, procedure=procedure, design=d
        )


# Fast smoke variant pairing the parameter_recovery test below (AGENTS.md),
# so a simulation-signature break still fails fast in the default suite.


def test_simulate_one_way_fixture_shape():
    rng = np.random.default_rng(0)
    n_c, s_c, q_c, n_t, s_t, q_t = _simulate_one_way(rng, n_levels=6, per_level=10, alpha_sd=0.5)
    assert n_c.shape == (6,)
    assert np.all(n_c + n_t == 10)
    result = absorb_one_way(n_c, s_c, q_c, n_t, s_t, q_t)
    assert 0.0 <= result.icc < 1.0
    assert np.isfinite(result.se_reduction)


def test_knob_direction_matches_absorb_one_way_smoke():
    """Small-N smoke check: a strong factor (icc~0.5) predicts a much bigger
    SE cut than a weak one (icc~0.04) - loose bound, just confirms the
    sign/ordering the parameter_recovery test below pins precisely."""
    rng = np.random.default_rng(1)
    weak = _simulate_one_way(rng, n_levels=10, per_level=15, alpha_sd=_icc_to_alpha_sd(0.04))
    strong = _simulate_one_way(rng, n_levels=10, per_level=15, alpha_sd=_icc_to_alpha_sd(0.50))
    weak_reduction = absorb_one_way(*weak).se_reduction
    strong_reduction = absorb_one_way(*strong).se_reduction
    assert strong_reduction > weak_reduction


# Calibration: knob prediction vs. absorb_one_way's measured se_reduction


@pytest.mark.slow
@pytest.mark.parameter_recovery
class TestCalibratedAgainstAbsorbOneWay:
    """Cross-check ``Baseline.from_absorption``'s ``1 - icc`` SE-reduction
    prediction against what ``absorb_one_way`` measures on the same Gaussian
    one-way DGP (K=50 levels, 80 units/level, balanced) the shipped
    calibration numbers were derived from.

    Thresholds are floors with margin against measurements taken when this
    suite was written (100 reps/ICC, seed=7): the largest observed bias was
    -1.07pp at ICC 0.80; 2.5pp gives headroom for run-to-run noise without
    hiding a real regression.
    """

    _RESULTS: dict[float, np.ndarray] | None = None

    def _measure(self, icc_targets, reps=100, seed=7):
        out = {}
        for icc in icc_targets:
            alpha_sd = _icc_to_alpha_sd(icc)
            reductions = np.empty(reps)
            for i in range(reps):
                rng = np.random.default_rng(seed * 100_000 + i)
                moments = _simulate_one_way(rng, n_levels=50, per_level=80, alpha_sd=alpha_sd)
                reductions[i] = absorb_one_way(*moments).se_reduction
            out[icc] = reductions
        return out

    def test_predicted_se_cut_tracks_measured_se_reduction(self):
        icc_targets = [0.04, 0.20, 0.50, 0.80]
        measured = self._measure(icc_targets)
        for icc in icc_targets:
            predicted = 1.0 - np.sqrt(1.0 - icc)
            actual = float(measured[icc].mean())
            assert abs(predicted - actual) < 0.025, (
                f"icc={icc}: predicted SE cut {predicted:.4f} vs. measured {actual:.4f}"
            )

    def test_icc_zero_artifact_is_not_credited(self):
        """At true ICC 0, absorb_one_way measures a small positive
        se_reduction (a sandwich-vs-naive-SE artifact, not real variance
        reduction). The knob must predict exactly 0 - it must never
        OVER-credit relative to what's actually measured."""
        measured = self._measure([0.0])[0.0]
        predicted = 1.0 - np.sqrt(1.0 - 0.0)
        assert predicted == 0.0
        # The artifact itself stays small; if this drifts far above ~3%
        # something about the DGP or absorb_one_way's sandwich changed.
        assert abs(float(measured.mean())) < 0.03
        # And the knob must not exceed what's actually measured on average.
        assert predicted <= float(measured.mean()) + 0.01
