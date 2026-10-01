"""Unit tests for the segment rollout recommendation.

The statistical contract (selection-bias ratio <= 0.25 of raw) was established
by simulation studies before this module existed; the parameter-recovery test
at the bottom re-derives a coarse version of that figure.  Everything else
here is fast and deterministic.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from increment.errors import InvalidRequestError
from increment.estimation.rollout import (
    SegmentRollout,
    segment_rollout,
)
from tests.oracles.test_meta_oracle import (
    POSTERIOR_CASES,
    assert_rollout_matches,
    segment_posterior,
)

# K=5 unequal-share SE profile: s_k = 0.095 / sqrt(5 * share_k), the same
# imbalance shape as tests/test_hte_calibration.py.
_SE5 = 0.095 / np.sqrt(5 * np.array([0.40, 0.25, 0.15, 0.12, 0.08]))
_VAR5 = _SE5**2


def _healthy_est():
    """Estimates straddling c=0 with pooled mean near 0: guard never fires."""
    return np.array([0.12, -0.05, 0.03, -0.10, 0.08])


class TestSelection:
    def test_selection_is_threshold_rule(self):
        r = segment_rollout(_healthy_est(), _VAR5, cost_threshold=0.02)
        assert r.selected.dtype == bool
        np.testing.assert_array_equal(r.selected, _healthy_est() > 0.02)

    def test_empty_selection_has_no_net_benefit(self):
        # Below the bar, but the pooled mean sits close enough that the guard
        # doesn't fire: subset is empty, raw=0, corrected<0, share is undefined (0/0).
        est = np.array([-0.02, -0.05, -0.01, -0.08, -0.03])
        r = segment_rollout(est, _VAR5, cost_threshold=0.0)
        assert r.recommendation == "no_net_benefit"
        assert not r.selected.any()
        assert r.policy_value_raw == 0.0
        assert r.policy_value is not None and r.policy_value < 0.0
        assert r.selection_bias_share is None


class TestCorrection:
    def test_correction_positive_and_reduces_value(self):
        r = segment_rollout(_healthy_est(), _VAR5)
        assert r.policy_value is not None
        assert r.policy_value_raw is not None
        assert r.selection_bias is not None
        assert r.selection_bias > 0.0
        assert r.policy_value == pytest.approx(r.policy_value_raw - r.selection_bias)
        assert r.policy_value < r.policy_value_raw


# (case, cost threshold): thresholds keep the offset guard's statistic well
# below its refusal level, so every value field is priced.
_PRICING_CASES: dict[str, tuple[np.ndarray, np.ndarray, float, float]] = {
    "healthy_k5_zero_bar": (_healthy_est(), _VAR5, 0.0, 0.30),
    "healthy_k5_bar_above_the_curse": (_healthy_est(), _VAR5, 0.02, 0.30),
    "documented_escaped_k100": (*POSTERIOR_CASES["documented_escaped_k100"][:2], 1.0, 0.30),
    "unequal_escaped_k6": (*POSTERIOR_CASES["unequal_escaped_k6"][:2], 0.0, 0.30),
    "narrow_at_zero_k40": (*POSTERIOR_CASES["narrow_at_zero_k40"][:2], 5e-4, 0.30),
    "sharp_shared_node_alias_k5": (*POSTERIOR_CASES["sharp_shared_node_alias_k5"][:2], 0.0, 0.30),
}


@pytest.mark.parametrize("case", _PRICING_CASES)
def test_pricing_matches_continuous_quadrature(case):
    """The guard statistic, the winner's-curse correction and the priced value
    are posterior expectations over tau; each must match continuous
    quadrature of the same model, including posteriors whose mass lies far
    beyond eight prior scales or inside a sliver near zero."""
    est, var, c, scale = _PRICING_CASES[case]
    r = segment_rollout(est, var, cost_threshold=c, tau_prior_scale=scale)
    reference = segment_posterior(est, var, scale, cost_threshold=c)
    assert_rollout_matches(r, reference, est=est, var=var, cost_threshold=c)


def test_unresolved_posterior_raises_the_shared_numerical_refusal(monkeypatch):
    """A posterior the node budget cannot resolve is the shared numerical
    refusal, not the offset guard's ``recommendation="refuse"`` and not a
    priced value; the budget is shrunk to reach that path deterministically."""
    monkeypatch.setattr("increment.estimation.meta._TAU_NODE_BUDGET", 1)
    with pytest.raises(InvalidRequestError) as exc_info:
        segment_rollout(_healthy_est(), _VAR5)
    assert exc_info.value.code == "estimation.meta.posterior_integration_unresolved"


class TestGuard:
    def test_refuses_far_offset(self):
        # Every estimate ~6 anchor-SEs below the bar: estimated offset far
        # past the guard, value withheld.
        est = np.full(5, -6 * 0.095)
        r = segment_rollout(est, _VAR5, cost_threshold=0.0)
        assert r.recommendation == "refuse"
        assert r.policy_value is None
        assert r.policy_value_raw is None
        assert r.selection_bias is None
        assert r.selection_bias_share is None

    def test_reports_in_healthy_regime(self):
        r = segment_rollout(_healthy_est(), _VAR5)
        assert r.recommendation == "rollout"
        assert abs(r.estimated_offset) < 1.0  # pooled mean straddles the bar
        assert r.policy_value is not None


class TestPricing:
    """The value claim is decoupled from the subset claim: 'rollout' asserts
    the honest value is positive; 'no_net_benefit' keeps the subset and the full
    accounting but says the evidence cannot demonstrate positive value."""

    def test_priced_rollout_has_positive_value_and_fractional_share(self):
        r = segment_rollout(_healthy_est(), _VAR5, cost_threshold=0.0)
        assert r.recommendation == "rollout"
        assert r.policy_value is not None and r.policy_value > 0.0
        assert r.selection_bias_share is not None
        assert 0.0 < r.selection_bias_share < 1.0

    def test_no_net_benefit_when_correction_eats_raw(self):
        # Same data, bar moved 0.02 up: raw 0.17, correction ~0.171 - the
        # winner's curse exceeds the entire apparent lift; subset stays nonempty.
        r = segment_rollout(_healthy_est(), _VAR5, cost_threshold=0.02)
        assert r.recommendation == "no_net_benefit"
        assert r.selected.any()
        assert r.policy_value is not None and r.policy_value <= 0.0
        assert r.policy_value_raw is not None and r.policy_value_raw > 0.0
        assert r.selection_bias_share is not None
        assert r.selection_bias_share >= 1.0

    def test_share_is_bias_over_raw(self):
        r = segment_rollout(_healthy_est(), _VAR5, cost_threshold=0.0)
        assert r.selection_bias is not None and r.policy_value_raw is not None
        assert r.selection_bias_share == pytest.approx(r.selection_bias / r.policy_value_raw)

    def test_no_clamping(self):
        # The no-net-benefit value is the real corrected number, not max(0, .).
        # Clamping would censor the left tail and bias the estimator back up.
        r = segment_rollout(_healthy_est(), _VAR5, cost_threshold=0.02)
        assert r.policy_value_raw is not None and r.selection_bias is not None
        assert r.policy_value == pytest.approx(r.policy_value_raw - r.selection_bias)


class TestModel:
    def test_frozen(self):
        r = segment_rollout(_healthy_est(), _VAR5)
        with pytest.raises(ValidationError):
            r.k = 7  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_value_fields_none_together_enforced(self):
        r = segment_rollout(_healthy_est(), _VAR5)
        with pytest.raises(InvalidRequestError) as exc_info:
            SegmentRollout(
                k=r.k,
                cost_threshold=r.cost_threshold,
                selected=np.asarray(r.selected),
                estimated_offset=r.estimated_offset,
                recommendation="rollout",
                policy_value=None,
                policy_value_raw=1.0,
                selection_bias=0.1,
                selection_bias_share=None,
            )
        assert exc_info.value.code == "estimation.rollout.segment_rollout.policy_value_policy"

    def test_selected_shape_enforced(self):
        r = segment_rollout(_healthy_est(), _VAR5)
        with pytest.raises(InvalidRequestError) as exc_info:
            SegmentRollout(
                k=3,
                cost_threshold=0.0,
                selected=np.asarray(r.selected),  # length 5, k says 3
                estimated_offset=0.0,
                recommendation="refuse",
                policy_value=None,
                policy_value_raw=None,
                selection_bias=None,
                selection_bias_share=None,
            )
        assert exc_info.value.code == "estimation.rollout.segment_rollout.selected_bool_length"

    def test_recommendation_must_match_value_sign(self):
        r = segment_rollout(_healthy_est(), _VAR5)  # priced: value > 0
        with pytest.raises(InvalidRequestError) as exc_info:
            SegmentRollout(
                k=r.k,
                cost_threshold=r.cost_threshold,
                selected=np.asarray(r.selected),
                estimated_offset=r.estimated_offset,
                recommendation="no_net_benefit",  # contradicts value > 0
                policy_value=r.policy_value,
                policy_value_raw=r.policy_value_raw,
                selection_bias=r.selection_bias,
                selection_bias_share=r.selection_bias_share,
            )
        assert (
            exc_info.value.code == "estimation.rollout.segment_rollout.recommendation_match_value"
        )

    def test_share_none_pattern_enforced(self):
        r = segment_rollout(_healthy_est(), _VAR5)
        with pytest.raises(InvalidRequestError) as exc_info:
            SegmentRollout(
                k=r.k,
                cost_threshold=r.cost_threshold,
                selected=np.asarray(r.selected),
                estimated_offset=r.estimated_offset,
                recommendation=r.recommendation,
                policy_value=r.policy_value,
                policy_value_raw=r.policy_value_raw,  # nonzero raw
                selection_bias=r.selection_bias,
                selection_bias_share=None,  # must be populated here
            )
        assert exc_info.value.code == "estimation.rollout.segment_rollout.selection_bias_share"


class TestGuards:
    def test_rejects_single_segment(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout([0.1], [0.02])
        assert exc_info.value.code == "estimation.meta.need_least_estimable"

    def test_rejects_bad_variance(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout([0.1, 0.2], [0.02, 0.0])
        assert exc_info.value.code == "estimation.meta.var_finite_strictly"

    def test_rejects_non_finite_estimate(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout([0.1, np.nan], [0.02, 0.05])
        assert exc_info.value.code == "estimation.meta.est_contains_non"

    def test_rejects_non_positive_prior_scale(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout([0.1, 0.2], [0.02, 0.05], tau_prior_scale=0.0)
        assert exc_info.value.code == "estimation.meta.tau_prior_scale"

    def test_rejects_non_finite_prior_scale(self):
        for bad in (float("nan"), float("inf")):
            with pytest.raises(InvalidRequestError) as exc_info:
                segment_rollout([0.1, 0.2], [0.02, 0.05], tau_prior_scale=bad)
            assert exc_info.value.code == "estimation.meta.tau_prior_scale"

    def test_rejects_non_finite_cost_threshold(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout([0.1, 0.2], [0.02, 0.05], cost_threshold=np.inf)
        assert exc_info.value.code == "estimation.rollout.cost_threshold_finite"


def _bias_ratio(reps: int, seed: int) -> float:
    """|mean bias| of the reported (corrected) value over |mean bias| raw, at
    the K=5 / spread 0.5 / offset 0 cell of the adoption study."""
    rng = np.random.default_rng(seed)
    k, tau = 5, 0.5 * 0.095
    d_raw, d_post = np.empty(reps), np.empty(reps)
    kept = np.zeros(reps, dtype=bool)
    for i in range(reps):
        theta = tau * rng.standard_normal(k)
        est = theta + _SE5 * rng.standard_normal(k)
        r = segment_rollout(est, _VAR5, cost_threshold=0.0)
        if r.policy_value is None or r.policy_value_raw is None:
            continue
        kept[i] = True
        vtrue = float(theta[np.asarray(r.selected)].sum())
        d_raw[i] = r.policy_value_raw - vtrue
        d_post[i] = r.policy_value - vtrue
    assert kept.mean() > 0.9  # offset 0 is the healthy regime
    return abs(d_post[kept].mean()) / abs(d_raw[kept].mean())


def test_bias_ratio_smoke():
    """Small-N smoke of the adoption figure (0.243 at 20k reps): loose band,
    fixed seed, fast."""
    assert _bias_ratio(reps=100, seed=42) < 0.7


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_bias_ratio_recovers_adoption_figure():
    assert _bias_ratio(reps=3000, seed=7) < 0.30


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.rollout.segment_rollout.selected_bool_length",
            lambda: SegmentRollout(
                k=3,
                cost_threshold=0.0,
                selected=np.zeros(2, dtype=bool),
                estimated_offset=0.0,
                recommendation="refuse",
                policy_value=None,
                policy_value_raw=None,
                selection_bias=None,
                selection_bias_share=None,
            ),
        ),  # estimation/rollout.py::SegmentRollout._check_consistency
        (
            "estimation.rollout.cost_threshold_finite",
            lambda: segment_rollout([0.1, 0.2], [0.02, 0.05], cost_threshold=np.inf),
        ),  # estimation/rollout.py::segment_rollout
    ],
)
def test_rollout_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
