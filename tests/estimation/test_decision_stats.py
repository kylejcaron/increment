"""Decision-stat methods on LiftEstimate - closed-form posterior summaries."""

import math

import pytest
from scipy.integrate import quad
from scipy.stats import norm

from increment.errors import InvalidRequestError
from increment.estimation.inference import Normal, infer_lift
from increment.estimation.results import Estimate, LiftEstimate
from tests.estimation.test_adjust import _DESIGN, _src


def test_infer_lift_stamps_log_scale():
    est = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.1 - 0.0,
        se_t=0.02,
        se_c=0.02,
    )
    assert est.scale == "log"


def test_scale_defaults_to_log_for_backcompat():
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1),
    )
    assert est.scale == "log"


def test_posterior_roundtrip_log_scale():
    # infer_lift with flat prior: posterior is Normal(log_rr, sqrt(se_t^2+se_c^2))
    est = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.15 - 0.05,
        se_t=0.03,
        se_c=0.04,
        alpha=0.05,
    )
    post = est._posterior()
    assert isinstance(post, Normal)
    assert math.isclose(post.mu, 0.10, rel_tol=1e-9)
    assert math.isclose(post.sigma, math.sqrt(0.03**2 + 0.04**2), rel_tol=1e-6)


def test_posterior_requires_interval():
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        est.p_value()
    assert exc_info.value.code == "estimation.results.lift.liftestimate_carries_no"


def test_posterior_rejects_asymmetric_interval():
    # A hand-built interval that is not a symmetric Normal quantile interval
    # on the log scale must be refused, not silently misread.
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.10, lb=-0.05, ub=0.40, level=0.95),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        est.p_value()
    assert exc_info.value.code == "estimation.results.lift.liftestimate_interval_symmetric"


def _est(mu: float, sigma: float, level: float = 0.95) -> LiftEstimate:
    """Build a log-scale estimate whose posterior is exactly Normal(mu, sigma)."""
    z = norm.ppf((1 + level) / 2)
    return LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(
            value=math.expm1(mu),
            lb=math.expm1(mu - z * sigma),
            ub=math.expm1(mu + z * sigma),
            level=level,
        ),
    )


class _PosteriorInterfaceOnly:
    def cdf(self, value: float) -> float:
        return 0.25

    def survival(self, value: float) -> float:
        return 0.75

    def quantile(self, probability: float) -> float:
        return 0.0

    def probability_between(self, lower: float, upper: float) -> float:
        return 0.4

    def expected_negative_part(self, *, scale: str) -> float:
        return 0.1

    def expected_positive_part(self, *, scale: str) -> float:
        return 0.2


class _InterfaceEstimate(LiftEstimate):
    def _posterior(self):
        return _PosteriorInterfaceOnly()


def _interface_estimate(**updates) -> LiftEstimate:
    estimate = _InterfaceEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        scale="linear",
        method_role="decision",
        lift=Estimate(value=0.0),
    )
    return estimate.model_copy(update=updates)


def test_probability_methods_use_posterior_interface():
    estimate = _interface_estimate()
    assert estimate.prob_beyond(0.0) == pytest.approx(0.75)
    assert estimate.prob_within(0.1) == pytest.approx(0.4)


def test_risk_methods_use_posterior_interface():
    estimate = _interface_estimate(preferred_direction="decrease")
    assert estimate.risk_if_shipped() == pytest.approx(0.1)
    assert estimate.risk_if_shipped_favorable() == pytest.approx(0.2)


def test_p_value_uses_posterior_interface():
    assert _interface_estimate().p_value() == pytest.approx(0.5)


def test_chance_to_beat_matches_normal_cdf():
    est = _est(mu=0.10, sigma=0.05)
    assert math.isclose(est.chance_to_beat(), norm.cdf(0.10 / 0.05), rel_tol=1e-9)


def test_prob_beyond_threshold_log_scale():
    est = _est(mu=0.10, sigma=0.05)
    # P(lift > 0.02) = P(X > log1p(0.02)) for X ~ N(0.10, 0.05)
    want = 1 - norm.cdf((math.log1p(0.02) - 0.10) / 0.05)
    assert math.isclose(est.prob_beyond(0.02), want, rel_tol=1e-9)


def test_prob_beyond_preserves_extreme_upper_tail():
    estimate = _est(mu=-1.0, sigma=0.05)
    assert estimate.prob_beyond(0.0) == pytest.approx(norm.sf(20.0))
    assert estimate.prob_beyond(0.0) > 0.0


def test_prob_within_is_rope():
    est = _est(mu=0.0, sigma=0.05)
    got = est.prob_within(0.01)
    want = norm.cdf(math.log1p(0.01) / 0.05) - norm.cdf(math.log1p(-0.01) / 0.05)
    assert math.isclose(got, want, rel_tol=1e-9)


def _absolute_linear_est(mu: float, sigma: float, level: float = 0.95) -> LiftEstimate:
    """A ``value_scale='absolute'`` row, e.g. an encouragement LATE: linear
    scale, additive units (matches how encouragement.py stamps these rows)."""
    z = norm.ppf((1 + level) / 2)
    return LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        scale="linear",
        value_scale="absolute",
        lift=Estimate(value=mu, lb=mu - z * sigma, ub=mu + z * sigma, level=level),
    )


def test_prob_within_accepts_an_absolute_scale_threshold_above_one():
    """A $5 ROPE on an absolute-scale (additive-units) metric like
    revenue-per-user must be reachable -- the docstring promises the
    window is read in the row's own units, not a (0, 1) fraction."""
    est = _absolute_linear_est(mu=1.0, sigma=2.0)
    got = est.prob_within(5.0)
    want = norm.cdf((5.0 - 1.0) / 2.0) - norm.cdf((-5.0 - 1.0) / 2.0)
    assert math.isclose(got, want, rel_tol=1e-9)


def test_prob_within_still_bounds_relative_scale_threshold_to_unit_interval():
    """A relative-scale row keeps the (0, 1) bound: the log1p transform is
    undefined outside it, and the value stays a lift fraction."""
    est = _est(mu=0.0, sigma=0.05)
    with pytest.raises(InvalidRequestError) as exc_info:
        est.prob_within(5.0)
    assert exc_info.value.code == "estimation.results.lift.prob_within_threshold_unit_interval"


def test_risk_if_shipped_matches_numeric_integration():
    est = _est(mu=0.02, sigma=0.05)
    # E[max(0, -(e^X - 1))] under X ~ N(0.02, 0.05)
    want, _ = quad(lambda x: max(0.0, -(math.expm1(x))) * norm.pdf(x, 0.02, 0.05), -1.0, 1.0)
    assert math.isclose(est.risk_if_shipped(), want, rel_tol=1e-6)


def test_p_value_two_sided():
    est = _est(mu=0.10, sigma=0.05)
    assert math.isclose(est.p_value(), 2 * norm.cdf(-2.0), rel_tol=1e-9)


def test_p_value_preserves_extreme_negative_tail():
    estimate = _est(mu=-1.0, sigma=0.05)
    assert estimate.p_value() == pytest.approx(2.0 * norm.sf(20.0))
    assert estimate.p_value() > 0.0


def test_linear_scale_dispatch():
    # A linear-scale estimate (what infer_ate produces): posterior directly on
    # the lift scale, no log transform in recovery.
    z = norm.ppf(0.975)
    mu, sigma = 0.04, 0.02
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="iptw",
        method_role="decision",
        scale="linear",
        lift=Estimate(value=mu, lb=mu - z * sigma, ub=mu + z * sigma, level=0.95),
    )
    assert math.isclose(est.chance_to_beat(), norm.cdf(mu / sigma), rel_tol=1e-9)
    # linear-scale risk: sigma*phi(mu/sigma) - mu*Phi(-mu/sigma), checked numerically
    want, _ = quad(lambda x: max(0.0, -x) * norm.pdf(x, mu, sigma), -0.5, 0.5)
    assert math.isclose(est.risk_if_shipped(), want, rel_tol=1e-6)


def test_iptw_estimate_stamps_linear_scale():
    # infer_ate (which the registered "iptw" method dispatches to) stamps
    # scale="linear" - cover the stamp end-to-end through the public estimator.
    from increment.estimation.adjust import estimate_ate

    (est,) = estimate_ate(_src(), _DESIGN).results
    assert est.scale == "linear"


def test_posterior_rejects_value_at_or_below_negative_one_on_log_scale():
    """value/lb <= -1 is outside the log scale's domain (log1p(-1) = -inf);
    a hand-built out-of-contract estimate must get the descriptive error,
    not an opaque math domain error."""
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=-1.0, lb=-1.5, ub=-0.5, level=0.95),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        est.p_value()
    assert exc_info.value.code == "estimation.results.lift.liftestimate_value_lb"


def test_prob_beyond_rejects_threshold_at_or_below_negative_one_on_log_scale():
    est = _est(mu=0.10, sigma=0.05)
    with pytest.raises(InvalidRequestError) as exc_info:
        est.prob_beyond(-1.0)
    assert exc_info.value.code == "estimation.results.lift.threshold_representable_log"


@pytest.mark.slow
def test_decision_stats_refuse_on_sequential_inference():
    from increment.errors import CapabilityError
    from increment.estimation.sequential_runtime import estimate_sequential
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli()
    est = estimate_sequential(snapshot, policy).results[0]
    assert est.stat_sig()
    for method in (est.chance_to_beat, est.p_value):
        with pytest.raises(CapabilityError) as raised:
            method()
        assert raised.value.code == "sequential.route.unsupported"


def test_risk_if_shipped_favorable_matches_risk_if_shipped_for_increase():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "increase"})
    assert est.risk_if_shipped_favorable() == pytest.approx(est.risk_if_shipped())


def test_risk_if_shipped_favorable_matches_risk_if_shipped_for_neutral():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "neutral"})
    assert est.risk_if_shipped_favorable() == pytest.approx(est.risk_if_shipped())


def test_risk_if_shipped_favorable_is_the_mirror_for_decrease_log_scale():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "decrease"})
    # E[max(0, e^X - 1)] under X ~ N(0.02, 0.05) - the mirror tail of
    # risk_if_shipped()'s E[max(0, -(e^X - 1))].
    want, _ = quad(lambda x: max(0.0, math.expm1(x)) * norm.pdf(x, 0.02, 0.05), -1.0, 1.0)
    assert math.isclose(est.risk_if_shipped_favorable(), want, rel_tol=1e-6)
    # Genuinely the mirror, not a no-op that silently reused risk_if_shipped().
    assert est.risk_if_shipped_favorable() != pytest.approx(est.risk_if_shipped())


def test_risk_if_shipped_favorable_is_the_mirror_for_decrease_linear_scale():
    mu, sigma = 0.04, 0.02
    z = norm.ppf(0.975)
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="iptw",
        method_role="decision",
        scale="linear",
        lift=Estimate(value=mu, lb=mu - z * sigma, ub=mu + z * sigma, level=0.95),
        preferred_direction="decrease",
    )
    want, _ = quad(lambda x: max(0.0, x) * norm.pdf(x, mu, sigma), -0.5, 0.5)
    assert math.isclose(est.risk_if_shipped_favorable(), want, rel_tol=1e-6)


def test_risk_if_shipped_favorable_requires_preferred_direction():
    est = _est(mu=0.02, sigma=0.05)  # no preferred_direction resolved
    with pytest.raises(InvalidRequestError) as exc_info:
        est.risk_if_shipped_favorable()
    assert exc_info.value.code == "estimation.results.lift.liftestimate_risk_if"


def test_chance_to_beat_favorable_matches_chance_to_beat_for_increase():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "increase"})
    assert est.chance_to_beat_favorable() == pytest.approx(est.chance_to_beat())


def test_chance_to_beat_favorable_matches_chance_to_beat_for_neutral():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "neutral"})
    assert est.chance_to_beat_favorable() == pytest.approx(est.chance_to_beat())


def test_chance_to_beat_favorable_is_complement_for_decrease():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "decrease"})
    assert est.chance_to_beat_favorable() == pytest.approx(1.0 - est.chance_to_beat())


def test_chance_to_beat_favorable_stays_vs_zero_unlike_prob_favorable():
    """chance_to_beat_favorable() must stay anchored at 0 even when
    null_lift is nonzero - unlike prob_favorable(), which folds the
    margin in. This is the property _decision_stat_columns depends on to
    keep chance_to_beat and risk_if_shipped on the same reference point
    in one readout row."""
    est = _est(mu=0.02, sigma=0.05).model_copy(
        update={"preferred_direction": "decrease", "null_lift": 0.05}
    )
    assert est.chance_to_beat_favorable() == pytest.approx(1.0 - est.chance_to_beat())
    assert est.chance_to_beat_favorable() != pytest.approx(est.prob_favorable())


def test_chance_to_beat_favorable_requires_preferred_direction():
    est = _est(mu=0.02, sigma=0.05)  # no preferred_direction resolved
    with pytest.raises(InvalidRequestError) as exc_info:
        est.chance_to_beat_favorable()
    assert exc_info.value.code == "estimation.results.lift.liftestimate_chance_to"
