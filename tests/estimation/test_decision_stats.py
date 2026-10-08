"""Decision-stat methods on LiftEstimate - closed-form posterior summaries."""

import copy
import math
import pickle

import pytest
from scipy.integrate import quad
from scipy.stats import norm

from increment.errors import InvalidRequestError
from increment.estimation.inference import Normal, infer_lift
from increment.estimation.results import Estimate, LiftEstimate, open_bound_from_two_sided_at_target
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
    est = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.15 - 0.05,
        se_t=0.03,
        se_c=0.04,
        prior=Normal(mu=0.1, sigma=0.2),
        alpha=0.05,
    )
    post = est._posterior()
    assert isinstance(post, Normal)
    assert post.mu == est.posterior_latent_mean
    assert post.sigma == est.posterior_latent_sd


def test_p_value_requires_sampling_statistics_not_posterior_interval_reconstruction():
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1, lb=-0.05, ub=0.4, level=0.95),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        est.p_value()
    assert exc_info.value.code == "estimation.results.lift.p_value_missing_log_mean_or_se"


def test_unavailable_clustered_posterior_returns_none_before_absolute_refusal():
    estimate = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.1,
        se_t=0.02,
        se_c=0.02,
        n_clusters=4,
        null_abs=0.05,
    )

    assert estimate.posterior_available is not True
    assert estimate.prob_favorable() is None


@pytest.mark.parametrize(
    ("prior_bound", "code"),
    [
        (False, "estimation.results.lift.open_interval_unrecoverable"),
        (True, "estimation.results.lift.fcr_prior_unsupported"),
    ],
)
def test_fcr_refusals_are_registered_and_copyable(prior_bound: bool, code: str) -> None:
    estimate = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.1,
        se_t=0.02,
        se_c=0.02,
        prior=Normal(mu=0.1, sigma=0.05) if prior_bound else None,
        alternative="greater",
    )
    if prior_bound:
        estimate = estimate.model_copy(update={"sampling_available": None, "prior_shrunk": True})
    else:
        lift = estimate.require_lift().model_copy(update={"alpha": None})
        estimate = estimate.model_copy(update={"lift": lift})

    with pytest.raises(InvalidRequestError) as raised:
        open_bound_from_two_sided_at_target(estimate)

    assert raised.value.code == code
    assert copy.deepcopy(raised.value).code == code
    assert pickle.loads(pickle.dumps(raised.value)).code == code


def test_sparse_auto_conversion_prior_keeps_sampling_when_posterior_variance_is_undefined():
    from increment.estimation.engine import Method, estimate_lift
    from increment.estimation.inference import Normal
    from tests.estimation._conversion_counts import CONVERSION_METRIC, count_summary

    summary = count_summary(0, 1, 1, 1)
    methods = [Method(name="unadjusted", conversion_inference="auto")]
    (baseline,) = estimate_lift([CONVERSION_METRIC], summary, "control", methods=methods).results
    (informed,) = estimate_lift(
        [CONVERSION_METRIC],
        summary,
        "control",
        methods=methods,
        prior=Normal(mu=0.1, sigma=0.05),
    ).results

    assert baseline.reference_kind == informed.reference_kind == "binomial"
    assert baseline.lift == informed.lift
    assert baseline.p_value() == informed.p_value()
    assert informed.sampling_available is True
    assert informed.posterior_available is False
    assert informed.posterior_reason_code == "estimation.armstats.arm_stats.least_compute_metric"


def _est(mu: float, sigma: float, level: float = 0.95) -> LiftEstimate:
    """Build a log-scale row with explicit, matching sampling and posterior state."""
    z = norm.ppf((1 + level) / 2)
    alpha = 1 - level
    return LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        sampling_available=True,
        posterior_available=True,
        posterior_model="normal",
        posterior_scale="log",
        posterior_estimate=math.expm1(mu),
        posterior_lb=math.expm1(mu - z * sigma),
        posterior_ub=math.expm1(mu + z * sigma),
        posterior_level=level,
        posterior_alpha=alpha,
        posterior_latent_mean=mu,
        posterior_latent_sd=sigma,
        lift=Estimate(
            value=math.expm1(mu),
            lb=math.expm1(mu - z * sigma),
            ub=math.expm1(mu + z * sigma),
            level=level,
            alpha=alpha,
            log_mean=mu,
            log_se=sigma,
        ),
    )


def _require_probability(value: float | None) -> float:
    assert value is not None
    return value


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
        posterior_available=True,
        posterior_model="normal",
        posterior_scale="linear",
        posterior_estimate=0.0,
        posterior_lb=-1.0,
        posterior_ub=1.0,
        posterior_level=0.95,
        posterior_alpha=0.05,
        posterior_latent_mean=0.0,
        posterior_latent_sd=1.0,
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


def test_p_value_does_not_read_posterior_interface():
    from increment.errors import InvalidRequestError

    with pytest.raises(InvalidRequestError) as raised:
        _interface_estimate().p_value()
    assert raised.value.code == "estimation.results.lift.p_value_missing_log_mean_or_se"


@pytest.mark.parametrize("preferred_direction", ["increase", "decrease"])
def test_posterior_favorable_probability_uses_declared_relative_null(preferred_direction):
    point, standard_error = 0.08, 0.05
    prior_mean, prior_sd, null_lift = 0.0, 0.1, 0.02
    variance = 1.0 / (1.0 / prior_sd**2 + 1.0 / standard_error**2)
    posterior_mean = variance * (prior_mean / prior_sd**2 + point / standard_error**2)
    threshold = math.log1p(null_lift)
    z = (threshold - posterior_mean) / math.sqrt(variance)
    expected = norm.cdf(z) if preferred_direction == "decrease" else norm.sf(z)

    estimate = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=point,
        se_t=0.03,
        se_c=0.04,
        prior=Normal(mu=prior_mean, sigma=prior_sd),
        null_lift=null_lift,
        preferred_direction=preferred_direction,
    )

    assert estimate.posterior_prob_favorable == pytest.approx(expected)
    assert estimate.prob_favorable() == pytest.approx(expected)
    assert estimate.posterior_prob_favorable != pytest.approx(
        norm.cdf((0.0 - posterior_mean) / math.sqrt(variance))
        if preferred_direction == "decrease"
        else norm.sf((0.0 - posterior_mean) / math.sqrt(variance))
    )


def test_posterior_favorable_probability_is_unavailable_for_absolute_null():
    estimate = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.08,
        se_t=0.03,
        se_c=0.04,
        prior=Normal(mu=0.0, sigma=0.1),
        null_abs=0.01,
        preferred_direction="increase",
    )
    assert estimate.posterior_prob_favorable is None
    assert estimate.prob_favorable() is None


def test_fcr_open_bound_accepts_modern_mixture_prior_sampling_row():
    from increment.estimation.priors import MixturePrior
    from increment.estimation.results import open_bound_from_two_sided_at_target

    estimate = infer_lift(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        log_rr=0.08,
        se_t=0.03,
        se_c=0.04,
        alternative="greater",
        prior=MixturePrior(
            weights=(0.5, 0.5),
            means=(0.0, 0.0),
            sigmas=(0.02, 0.2),
        ),
    )

    assert estimate.sampling_available is True
    assert estimate.posterior_available is True
    corrected = open_bound_from_two_sided_at_target(estimate)
    assert corrected.lift is not None
    assert corrected.sampling_available is True


def test_chance_to_beat_matches_normal_cdf():
    est = _est(mu=0.10, sigma=0.05)
    assert math.isclose(
        _require_probability(est.chance_to_beat()), norm.cdf(0.10 / 0.05), rel_tol=1e-9
    )


def test_prob_beyond_threshold_log_scale():
    est = _est(mu=0.10, sigma=0.05)
    # P(lift > 0.02) = P(X > log1p(0.02)) for X ~ N(0.10, 0.05)
    want = 1 - norm.cdf((math.log1p(0.02) - 0.10) / 0.05)
    assert math.isclose(_require_probability(est.prob_beyond(0.02)), want, rel_tol=1e-9)


def test_prob_beyond_preserves_extreme_upper_tail():
    estimate = _est(mu=-1.0, sigma=0.05)
    assert _require_probability(estimate.prob_beyond(0.0)) == pytest.approx(norm.sf(20.0))
    assert _require_probability(estimate.prob_beyond(0.0)) > 0.0


def test_prob_within_is_rope():
    est = _est(mu=0.0, sigma=0.05)
    got = est.prob_within(0.01)
    want = norm.cdf(math.log1p(0.01) / 0.05) - norm.cdf(math.log1p(-0.01) / 0.05)
    assert math.isclose(_require_probability(got), want, rel_tol=1e-9)


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
        posterior_available=True,
        posterior_model="normal",
        posterior_scale="linear",
        posterior_estimate=mu,
        posterior_lb=mu - z * sigma,
        posterior_ub=mu + z * sigma,
        posterior_level=level,
        posterior_alpha=1 - level,
        posterior_latent_mean=mu,
        posterior_latent_sd=sigma,
        lift=Estimate(value=mu, lb=mu - z * sigma, ub=mu + z * sigma, level=level),
    )


def test_prob_within_accepts_an_absolute_scale_threshold_above_one():
    """A $5 ROPE on an absolute-scale (additive-units) metric like
    revenue-per-user must be reachable -- the docstring promises the
    window is read in the row's own units, not a (0, 1) fraction."""
    est = _absolute_linear_est(mu=1.0, sigma=2.0)
    got = est.prob_within(5.0)
    want = norm.cdf((5.0 - 1.0) / 2.0) - norm.cdf((-5.0 - 1.0) / 2.0)
    assert math.isclose(_require_probability(got), want, rel_tol=1e-9)


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
    assert math.isclose(_require_probability(est.risk_if_shipped()), want, rel_tol=1e-6)


def test_p_value_two_sided():
    est = _est(mu=0.10, sigma=0.05)
    assert math.isclose(_require_probability(est.p_value()), 2 * norm.cdf(-2.0), rel_tol=1e-9)


def test_p_value_preserves_extreme_negative_tail():
    estimate = _est(mu=-1.0, sigma=0.05)
    assert _require_probability(estimate.p_value()) == pytest.approx(2.0 * norm.sf(20.0))
    assert _require_probability(estimate.p_value()) > 0.0


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
        posterior_available=True,
        posterior_model="normal",
        posterior_scale="linear",
        posterior_estimate=mu,
        posterior_lb=mu - z * sigma,
        posterior_ub=mu + z * sigma,
        posterior_level=0.95,
        posterior_alpha=0.05,
        posterior_latent_mean=mu,
        posterior_latent_sd=sigma,
        lift=Estimate(value=mu, lb=mu - z * sigma, ub=mu + z * sigma, level=0.95),
    )
    # linear-scale risk: sigma*phi(mu/sigma) - mu*Phi(-mu/sigma), checked numerically
    want, _ = quad(lambda x: max(0.0, -x) * norm.pdf(x, mu, sigma), -0.5, 0.5)
    assert math.isclose(_require_probability(est.risk_if_shipped()), want, rel_tol=1e-6)


def test_iptw_estimate_stamps_linear_scale():
    # infer_ate (which the registered "iptw" method dispatches to) stamps
    # scale="linear" - cover the stamp end-to-end through the public estimator.
    from increment.estimation.adjust import estimate_ate

    (est,) = estimate_ate(_src(), _DESIGN).results
    assert est.scale == "linear"


def test_legacy_interval_without_sampling_statistics_cannot_compute_p_value():
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=-1.0, lb=-1.5, ub=-0.5, level=0.95),
    )
    with pytest.raises(InvalidRequestError) as exc_info:
        est.p_value()
    assert exc_info.value.code == "estimation.results.lift.p_value_missing_log_mean_or_se"


@pytest.mark.slow
def test_sequential_rows_return_no_posterior_but_still_refuse_sampling_p_value():
    from increment.estimation.sequential_runtime import estimate_sequential
    from tests.sequential_cases import registered_bernoulli

    snapshot, policy = registered_bernoulli()
    est = estimate_sequential(snapshot, policy).results[0]
    assert est.stat_sig()
    assert est.chance_to_beat() is None
    assert est.prob_beyond(0.0) is None
    assert est.prob_within(0.01) is None
    assert est.risk_if_shipped() is None
    from increment.errors import CapabilityError

    with pytest.raises(CapabilityError) as raised:
        est.p_value()
    assert raised.value.code == "sequential.route.unsupported"


def test_cluster_rows_without_posterior_return_none_for_posterior_accessors():
    est = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        reference_kind="t",
        reference_df=9.0,
        dof=9.0,
        n_clusters=10,
        sampling_available=True,
        posterior_available=False,
        lift=Estimate(value=0.1, lb=-0.1, ub=0.3, level=0.95, log_mean=0.1, log_se=0.1),
    )
    assert est.chance_to_beat() is None
    assert est.prob_beyond(0.0) is None
    assert est.prob_within(0.01) is None
    assert est.risk_if_shipped() is None


def test_cluster_rows_with_persisted_posterior_expose_posterior_accessors():
    mean, sd = 0.04, 0.05
    estimate = LiftEstimate(
        metric="m",
        group_id="t",
        method="unadjusted",
        method_role="decision",
        n_clusters=10,
        sampling_available=True,
        posterior_available=True,
        posterior_model="normal",
        posterior_scale="linear",
        posterior_estimate=mean,
        posterior_lb=mean - 1.96 * sd,
        posterior_ub=mean + 1.96 * sd,
        posterior_level=0.95,
        posterior_alpha=0.05,
        posterior_latent_mean=mean,
        posterior_latent_sd=sd,
        lift=Estimate(value=0.1, lb=-0.1, ub=0.3, level=0.95, log_mean=0.1, log_se=0.1),
    )

    assert estimate.chance_to_beat() == pytest.approx(Normal(mu=mean, sigma=sd).survival(0.0))


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
    assert math.isclose(_require_probability(est.risk_if_shipped_favorable()), want, rel_tol=1e-6)
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
        posterior_available=True,
        posterior_model="normal",
        posterior_scale="linear",
        posterior_lb=mu - z * sigma,
        posterior_ub=mu + z * sigma,
        posterior_level=0.95,
        posterior_alpha=0.05,
        posterior_estimate=mu,
        posterior_latent_mean=mu,
        posterior_latent_sd=sigma,
        lift=Estimate(value=mu, lb=mu - z * sigma, ub=mu + z * sigma, level=0.95),
        preferred_direction="decrease",
    )
    want, _ = quad(lambda x: max(0.0, x) * norm.pdf(x, mu, sigma), -0.5, 0.5)
    assert math.isclose(_require_probability(est.risk_if_shipped_favorable()), want, rel_tol=1e-6)


def test_risk_if_shipped_favorable_is_unavailable_without_preferred_direction():
    est = _est(mu=0.02, sigma=0.05)  # no preferred_direction resolved
    assert est.risk_if_shipped_favorable() is None


def test_chance_to_beat_favorable_matches_chance_to_beat_for_increase():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "increase"})
    assert est.chance_to_beat_favorable() == pytest.approx(est.chance_to_beat())


def test_chance_to_beat_favorable_matches_chance_to_beat_for_neutral():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "neutral"})
    assert est.chance_to_beat_favorable() == pytest.approx(est.chance_to_beat())


def test_chance_to_beat_favorable_is_complement_for_decrease():
    est = _est(mu=0.02, sigma=0.05).model_copy(update={"preferred_direction": "decrease"})
    assert _require_probability(est.chance_to_beat_favorable()) == pytest.approx(
        1.0 - _require_probability(est.chance_to_beat())
    )


def test_chance_to_beat_favorable_stays_vs_zero_unlike_prob_favorable():
    """chance_to_beat_favorable() must stay anchored at 0 even when
    null_lift is nonzero - unlike prob_favorable(), which folds the
    margin in. This is the property _decision_stat_columns depends on to
    keep chance_to_beat and risk_if_shipped on the same reference point
    in one readout row."""
    est = _est(mu=0.02, sigma=0.05).model_copy(
        update={"preferred_direction": "decrease", "null_lift": 0.05}
    )
    assert _require_probability(est.chance_to_beat_favorable()) == pytest.approx(
        1.0 - _require_probability(est.chance_to_beat())
    )
    assert est.chance_to_beat_favorable() != pytest.approx(est.prob_favorable())


def test_chance_to_beat_favorable_is_unavailable_without_preferred_direction():
    est = _est(mu=0.02, sigma=0.05)  # no preferred_direction resolved
    assert est.chance_to_beat_favorable() is None
