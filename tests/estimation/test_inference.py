"""Tests for Normal, normal_posterior, and infer_lift."""

import math
import sys
from typing import cast

import pytest
from pydantic import ValidationError
from scipy.stats import norm as _norm_dist
from scipy.stats import t as _t_dist

from increment.errors import InvalidRequestError
from increment.estimation.armstats import ScoreStats
from increment.estimation.inference import (
    FixedHorizonReference,
    LiftGuardError,
    Normal,
    SamplingReference,
    _resolve_fixed_horizon,
    infer_ate,
    infer_lift,
    normal_posterior,
)
from increment.estimation.results import LiftEstimate
from increment.estimation.sequential import AlwaysValid
from increment.estimation.variance import se_log_mean as _se_log_mean
from tests.sequential_cases import registration


class TestNormal:
    def test_frozen(self):
        n = Normal(mu=0.0, sigma=1.0)
        with pytest.raises((TypeError, ValueError)):
            n.mu = 1.0  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_sigma_positive(self):
        """Sigma must be > 0."""
        Normal(mu=0.0, sigma=1e-6)  # tiny positive is fine
        with pytest.raises(ValidationError):
            Normal(mu=0.0, sigma=0.0)
        with pytest.raises(ValidationError):
            Normal(mu=0.0, sigma=-1.0)

    def test_precision(self):
        n = Normal(mu=0.0, sigma=2.0)
        assert n.precision == pytest.approx(0.25, rel=1e-12)

    def test_satisfies_lift_posterior_interface(self):
        from increment.estimation import LiftPosterior

        assert isinstance(Normal(mu=0.1, sigma=0.2), LiftPosterior)

    def test_cdf(self):
        distribution = Normal(mu=0.1, sigma=0.2)
        assert distribution.cdf(0.3) == pytest.approx(_norm_dist.cdf(1.0))

    def test_survival(self):
        distribution = Normal(mu=0.1, sigma=0.2)
        assert distribution.survival(0.3) == pytest.approx(_norm_dist.sf(1.0))

    def test_quantile(self):
        distribution = Normal(mu=0.1, sigma=0.2)
        assert distribution.quantile(0.975) == pytest.approx(0.1 + 0.2 * _norm_dist.ppf(0.975))

    def test_probability_between(self):
        distribution = Normal(mu=0.1, sigma=0.2)
        assert distribution.probability_between(-0.1, 0.3) == pytest.approx(
            _norm_dist.cdf(1.0) - _norm_dist.cdf(-1.0)
        )

    def test_probability_between_preserves_upper_tail(self):
        distribution = Normal(mu=0.0, sigma=1.0)
        assert distribution.probability_between(10.0, 11.0) == pytest.approx(
            _norm_dist.sf(10.0) - _norm_dist.sf(11.0)
        )
        assert distribution.probability_between(10.0, 11.0) > 0.0

    @pytest.mark.parametrize("scale", ["linear", "log"])
    def test_expected_negative_part(self, scale):
        from scipy.integrate import quad

        distribution = Normal(mu=0.02, sigma=0.05)

        def integrand(x):
            lift = x if scale == "linear" else math.expm1(x)
            density = _norm_dist.pdf(x, loc=distribution.mu, scale=distribution.sigma)
            return max(0.0, -lift) * density

        expected, _ = quad(integrand, -1.0, 1.0)
        assert distribution.expected_negative_part(scale=scale) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize("scale", ["linear", "log"])
    def test_expected_positive_part(self, scale):
        from scipy.integrate import quad

        distribution = Normal(mu=0.02, sigma=0.05)

        def integrand(x):
            lift = x if scale == "linear" else math.expm1(x)
            density = _norm_dist.pdf(x, loc=distribution.mu, scale=distribution.sigma)
            return max(0.0, lift) * density

        expected, _ = quad(integrand, -1.0, 1.0)
        assert distribution.expected_positive_part(scale=scale) == pytest.approx(expected, rel=1e-6)

    @pytest.mark.parametrize(
        "method_name",
        ["expected_negative_part", "expected_positive_part"],
    )
    def test_expected_parts_reject_unknown_scale(self, method_name):
        distribution = Normal(mu=0.02, sigma=0.05)
        codes = {
            "expected_negative_part": "estimation.priors.mixture_posterior.expected_negative_part_scale",
            "expected_positive_part": "estimation.priors.mixture_posterior.expected_positive_part_scale",
        }
        with pytest.raises(InvalidRequestError) as exc_info:
            getattr(distribution, method_name)(scale="unknown")
        assert exc_info.value.code == codes[method_name]


class TestNormalPosterior:
    def test_default_prior_is_near_flat(self):
        """The default prior leaves ordinary estimates effectively unchanged."""
        posterior = normal_posterior(0.5, 0.1)
        assert posterior.mu == pytest.approx(0.5, rel=1e-6)
        assert posterior.sigma == pytest.approx(0.1, rel=1e-6)

    def test_informative_prior_is_precision_weighted(self):
        prior = Normal(mu=0.0, sigma=0.5)
        mu_obs, sigma_obs = 1.0, 0.5
        posterior = normal_posterior(mu_obs, sigma_obs, prior=prior)
        expected_sigma = math.sqrt(1.0 / (prior.precision + (1 / sigma_obs**2)))
        expected_mu = expected_sigma**2 * (prior.mu * prior.precision + mu_obs / sigma_obs**2)
        assert posterior.sigma == pytest.approx(expected_sigma, rel=1e-6)
        assert posterior.mu == pytest.approx(expected_mu, rel=1e-6)

    def test_far_dominant_prior_returns_prior_instead_of_overflowing(self):
        """A ~150-order-of-magnitude sigma gap keeps r = small.sigma /
        large.sigma tiny but nonzero (1e-155, not the r == 0.0 limit); the
        general path's denominator/hypot/product terms all round back to
        exactly the prior's own mu and sigma, without overflowing."""
        prior = Normal(mu=0.5, sigma=1e-154)
        posterior = normal_posterior(1.0, 10.0, prior)
        assert posterior.mu == prior.mu
        assert posterior.sigma == prior.sigma


def _decimal_conjugate_posterior(
    prior_mu: float, prior_sigma: float, obs_mu: float, obs_sigma: float
) -> tuple[float, float]:
    """Independent reference: exact precision-weighted update via Decimal,
    a route distinct from normal_posterior's float arithmetic and immune
    to the float64 overflow/underflow the fix removes."""
    import decimal
    from decimal import Decimal

    with decimal.localcontext() as ctx:
        ctx.prec = 80
        d_prior_mu, d_prior_sigma = Decimal(prior_mu), Decimal(prior_sigma)
        d_obs_mu, d_obs_sigma = Decimal(obs_mu), Decimal(obs_sigma)
        prior_precision = 1 / (d_prior_sigma * d_prior_sigma)
        obs_precision = 1 / (d_obs_sigma * d_obs_sigma)
        posterior_precision = prior_precision + obs_precision
        posterior_mu = (
            d_prior_mu * prior_precision + d_obs_mu * obs_precision
        ) / posterior_precision
        posterior_sigma = (1 / posterior_precision).sqrt()
        return float(posterior_mu), float(posterior_sigma)


class TestNormalPosteriorExtremeScales:
    """normal_posterior must form the posterior around the smaller scale
    instead of squaring absolute precisions, which overflows/underflows
    long before the sigmas themselves become unrepresentable."""

    @pytest.mark.parametrize(
        "prior_sigma,obs_sigma", [(1e-200, 2e-200), (1e200, 2e200), (1e-160, 2e-160)]
    )
    def test_equally_extreme_sigmas_match_decimal_reference(self, prior_sigma, obs_sigma):
        prior_mu, obs_mu = 0.3, 1.0
        posterior = normal_posterior(
            obs_mu, obs_sigma, prior=Normal(mu=prior_mu, sigma=prior_sigma)
        )
        assert math.isfinite(posterior.mu)
        assert math.isfinite(posterior.sigma) and posterior.sigma > 0.0
        ref_mu, ref_sigma = _decimal_conjugate_posterior(prior_mu, prior_sigma, obs_mu, obs_sigma)
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-9, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)

    @pytest.mark.parametrize("prior_sigma,obs_sigma", [(1e-300, 1e300), (1e300, 1e-300)])
    def test_both_orderings_of_wide_ratio_match_decimal_reference(self, prior_sigma, obs_sigma):
        prior_mu, obs_mu = -2.0, 5.0
        posterior = normal_posterior(
            obs_mu, obs_sigma, prior=Normal(mu=prior_mu, sigma=prior_sigma)
        )
        assert math.isfinite(posterior.mu)
        assert math.isfinite(posterior.sigma) and posterior.sigma > 0.0
        ref_mu, ref_sigma = _decimal_conjugate_posterior(prior_mu, prior_sigma, obs_mu, obs_sigma)
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-9, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)

    def test_smallest_subnormal_sigma_hits_exact_r_zero_limit(self):
        """r = small.sigma / large.sigma underflows to exactly 0.0 here
        (smallest positive subnormal divided by 10 is below the smallest
        representable positive double): the narrower distribution must be
        returned as the exact dominant limit."""
        tiny = 5e-324  # smallest positive subnormal double
        assert tiny / 10.0 == 0.0  # sanity: this is the r == 0.0 branch, not just tiny r
        prior = Normal(mu=0.7, sigma=tiny)
        posterior = normal_posterior(1.0, 10.0, prior=prior)
        assert posterior.mu == prior.mu
        assert posterior.sigma == prior.sigma

    def test_near_max_finite_sigmas_match_decimal_reference(self):
        prior_sigma, obs_sigma = 9e307, 1.7e308
        prior_mu, obs_mu = 100.0, -50.0
        posterior = normal_posterior(
            obs_mu, obs_sigma, prior=Normal(mu=prior_mu, sigma=prior_sigma)
        )
        assert math.isfinite(posterior.mu)
        assert math.isfinite(posterior.sigma) and posterior.sigma > 0.0
        ref_mu, ref_sigma = _decimal_conjugate_posterior(prior_mu, prior_sigma, obs_mu, obs_sigma)
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-9, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)

    @pytest.mark.parametrize(
        "prior_mu,obs_mu",
        [(1e250, 1e250), (1e250, -1e250), (-3e200, 3e200)],
        ids=["same_sign", "opposite_sign", "opposite_sign_prior_negative"],
    )
    def test_large_signed_means_with_extreme_sigmas_match_decimal_reference(self, prior_mu, obs_mu):
        prior_sigma, obs_sigma = 1e-100, 2e-100
        posterior = normal_posterior(
            obs_mu, obs_sigma, prior=Normal(mu=prior_mu, sigma=prior_sigma)
        )
        assert math.isfinite(posterior.mu)
        assert math.isfinite(posterior.sigma) and posterior.sigma > 0.0
        ref_mu, ref_sigma = _decimal_conjugate_posterior(prior_mu, prior_sigma, obs_mu, obs_sigma)
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-9, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)

    def test_prior_dominant_extreme_scale_gap_matches_decimal_reference(self):
        """Kills a wrong mean formula that lets the dominant term's own
        precision (1/1e-200**2 = 1e400) leak into the numerator instead of
        cancelling: the true posterior mean is the observation's tiny
        r**2-weighted contribution, ~1e-150, not 0 and not 1e250."""
        prior_mu, prior_sigma = 0.0, 1e-200
        obs_mu, obs_sigma = 1e250, 1.0
        posterior = normal_posterior(
            obs_mu, obs_sigma, prior=Normal(mu=prior_mu, sigma=prior_sigma)
        )
        ref_mu, ref_sigma = _decimal_conjugate_posterior(prior_mu, prior_sigma, obs_mu, obs_sigma)
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-9, abs=0.0)
        assert posterior.mu == pytest.approx(1e-150, rel=1e-6, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)

    def test_opposite_sign_large_means_ordinary_sigmas_matches_decimal_reference(self):
        """Kills a wrong opposite-sign branch that overflows or cancels
        incorrectly when multiplying r * large.mu before dividing: ordinary
        (non-extreme) sigmas here, so any wrong intermediate order of
        operations - not overflow avoidance - is what this catches."""
        prior_mu, prior_sigma = 1.7e308, 1.0
        obs_mu, obs_sigma = -1.7e308, 2.0
        posterior = normal_posterior(
            obs_mu, obs_sigma, prior=Normal(mu=prior_mu, sigma=prior_sigma)
        )
        ref_mu, ref_sigma = _decimal_conjugate_posterior(prior_mu, prior_sigma, obs_mu, obs_sigma)
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-9, abs=0.0)
        assert posterior.mu == pytest.approx(1.02e308, rel=1e-6, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)

    def test_identical_means_at_float_max_stay_exact(self):
        """Kills a wrong same-sign branch that recomputes ``small.mu +
        r*(r*delta)/denominator`` in an order that lets rounding push the
        result away from ``small.mu`` when delta is exactly 0: identical
        means at the largest finite double must round-trip exactly, not
        merely approximately."""
        max_mu = sys.float_info.max
        prior_sigma, obs_sigma = 1e-100, 2e-100
        posterior = normal_posterior(max_mu, obs_sigma, prior=Normal(mu=max_mu, sigma=prior_sigma))
        ref_mu, ref_sigma = _decimal_conjugate_posterior(max_mu, prior_sigma, max_mu, obs_sigma)
        assert posterior.mu == max_mu
        assert posterior.mu == pytest.approx(ref_mu, rel=1e-12, abs=0.0)
        assert posterior.sigma == pytest.approx(ref_sigma, rel=1e-9, abs=0.0)


class TestInferLift:
    def test_positive_lift(self):
        """infer_lift returns a positive lift for log_rr > 0."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            alpha=0.05,
        )
        assert isinstance(result, LiftEstimate)
        assert result.metric == "rev"
        assert result.group_id == "B"
        assert result.method == "unadjusted"
        lift = result.require_lift()
        assert lift.value > 0  # positive lift
        assert lift.level == 0.95
        assert lift.lb is not None
        assert lift.ub is not None
        assert lift.lb < lift.ub

    def test_per_arm_log_keywords_are_rejected(self):
        """The joint ``log_rr`` is the only point input. Per-arm logs would
        reintroduce the ``log_t - log_c`` cancellation the caller is
        responsible for avoiding (see ``stable_log_ratio``), so the old
        keywords fail at the call rather than being subtracted here."""
        with pytest.raises(TypeError):
            infer_lift(  # ty: ignore[missing-argument]  # proving the rejection at runtime
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_t=0.5,  # ty: ignore[unknown-argument]
                se_t=0.1,
                log_c=0.0,  # ty: ignore[unknown-argument]
                se_c=0.1,
            )

    def test_alpha_outside_unit_interval_refused(self):
        """alpha=-0.05 used to produce level=1.05 with NaN bounds, and
        alpha=0 an infinite interval - silently stored in the frozen
        Estimate. The domain is validated at entry."""
        for bad in (-0.05, 0.0, 1.0, 1.5):
            with pytest.raises(InvalidRequestError) as exc_info:
                infer_lift(
                    metric="rev",
                    group_id="B",
                    method="unadjusted",
                    method_role="decision",
                    log_rr=0.5 - 0.0,
                    se_t=0.1,
                    se_c=0.1,
                    alpha=bad,
                )
            assert exc_info.value.code == "estimation.diagnostics.alpha"

    def test_minimum_alpha_survives_one_sided_effective_tail(self):
        """The smallest positive nominal alpha doubles before the half-tail
        precision guard: alpha_eff/2 remains representable for one-sided
        inference, while a two-sided request is refused."""
        tiny = math.nextafter(0.0, 1.0)
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            alpha=tiny,
            alternative="greater",
        )
        lift = result.require_lift()
        assert lift.alpha == 2.0 * tiny
        assert lift.lb is not None and lift.ub is not None
        assert math.isfinite(lift.lb) and math.isfinite(lift.ub)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.5 - 0.0,
                se_t=0.1,
                se_c=0.1,
                alpha=tiny,
            )
        assert exc_info.value.code == "estimation.encouragement.alpha_eff_too"

    def test_one_sided_alpha_half_or_more_refused(self):
        """alternative='greater' at alpha=0.6 doubles to alpha_eff=1.2:
        level=-0.2 and an inverted lb>ub interval used to be silently
        stored. No displayable two-sided interval exists at alpha_eff>=1,
        so it is refused by name."""
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.5 - 0.0,
                se_t=0.1,
                se_c=0.1,
                alpha=0.6,
                alternative="greater",
            )
        assert exc_info.value.code == "estimation.inference.one_sided_alpha_doubles"

    def test_serializes_and_recovers_posterior(self):
        """LiftEstimate.model_dump_json() round-trips on a real infer_lift
        result - proving the deleted `posterior` field was never load-
        bearing for serialization. And since it carried a Normal posterior,
        (mu_n, sigma_n) are exactly recoverable from value/lb/level/alpha via
        log1p(value) and (log1p(value) - log1p(lb)) / z - the parameters
        infer_lift actually computed, reproduced without a stored field."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            alpha=0.05,
        )
        round_tripped = LiftEstimate.model_validate_json(result.model_dump_json())
        assert round_tripped == result

        posterior = normal_posterior(0.5 - 0.0, math.sqrt(0.1**2 + 0.1**2))
        lift = result.require_lift()
        assert lift.lb is not None and lift.alpha is not None
        z = _norm_dist.isf(lift.alpha / 2.0)
        recovered_mu = math.log1p(lift.value)
        recovered_sigma = (math.log1p(lift.value) - math.log1p(lift.lb)) / z
        assert recovered_mu == pytest.approx(posterior.mu, rel=1e-12)
        assert recovered_sigma == pytest.approx(posterior.sigma, rel=1e-12)

    def test_precise_extreme_log_rr_is_not_refused(self):
        """A large ratio measured precisely is exactly where the log delta
        method works: at n=5000/arm with CV~1 the per-arm log-scale SE is
        1/sqrt(5000) ~ 0.014, and a true log_rr=2.6 has nominal (0.952)
        empirical coverage. Refusing on the point estimate tested the
        wrong statistic AND truncated the sampling distribution near the
        boundary (refusal conditioned on the estimate = selection bias)."""
        se = 1.0 / math.sqrt(5000.0)
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=2.6 - 0.0,
            se_t=se,
            se_c=se,
            alpha=0.05,
        )
        lift = result.require_lift()
        assert lift.value == pytest.approx(math.exp(2.6) - 1.0)
        assert lift.lb is not None and lift.ub is not None
        assert lift.lb < lift.value < lift.ub

    def test_imprecise_log_scale_se_is_refused_even_at_zero_lift(self):
        """Delta-method validity is controlled by the combined log-scale
        SE (~ CV/sqrt(n) per arm), not by the ratio of means: se_t=se_c=0.4
        combines to 0.566, where the first-order expansion of log(Ybar) is
        untrustworthy - refused even though log_rr is exactly 0, the
        (n*p ~ 1) regime the old point-value guard silently accepted."""
        with pytest.raises(LiftGuardError) as raised:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.0 - 0.0,
                se_t=0.4,
                se_c=0.4,
                alpha=0.05,
            )

        assert raised.value.reason == "delta_method_unreliable"

    def test_se_guard_is_symmetric_in_sign(self):
        """The guard cares about precision only, so it is sign-symmetric:
        a precise log_rr=-3.0 passes (a real, well-measured -95% drop)
        while an imprecise one refuses - the old guard accepted BOTH."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=-3.0 - 0.0,
            se_t=0.01,
            se_c=0.01,
        )
        assert result.require_lift().value == pytest.approx(math.exp(-3.0) - 1.0)
        with pytest.raises(LiftGuardError) as raised:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=-3.0 - 0.0,
                se_t=0.4,
                se_c=0.4,
            )

        assert raised.value.reason == "delta_method_unreliable"

    def test_se_guard_boundary(self):
        """Combined SE just below 0.5 passes; at/above 0.5 refuses."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.49,
            se_c=0.05,
        )
        assert result.require_lift().value is not None
        with pytest.raises(LiftGuardError) as raised:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.5 - 0.0,
                se_t=0.4995,
                se_c=0.05,
            )

        assert raised.value.reason == "delta_method_unreliable"

    def test_single_saturated_arm_is_refused(self):
        """One arm with zero variance (k = n saturation: at n_t=200,
        p_t=0.995 the treatment arm saturates 37% of the time, making
        se_t exactly 0) deletes that arm's uncertainty from the interval;
        against a precise control (n_c=200k) coverage conditional on
        saturation is 0.392 at nominal 0.95. The old guard only fired
        when BOTH arms were degenerate; refusal must be per-arm."""
        se_c = math.sqrt((1.0 - 0.96) / (0.96 * 200_000))
        with pytest.raises(LiftGuardError) as raised:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=math.log(1.0) - math.log(0.96),
                se_t=0.0,
                se_c=se_c,
            )

        assert raised.value.reason == "zero_variance"

    def test_degenerate_data_guard_raises(self):
        """Both arms zero variance still refuses (per-arm guard covers
        the both-degenerate case a fortiori)."""
        with pytest.raises(LiftGuardError) as raised:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.5 - 0.0,
                se_t=0.0,
                se_c=0.0,
            )

        assert raised.value.reason == "zero_variance"

    def test_lift_guard_error_carries_stable_reason(self):
        """Guard callers use the stable reason, not diagnostic message text."""
        from increment.estimation.inference import LiftGuardError

        assert issubclass(LiftGuardError, ValueError)
        with pytest.raises(LiftGuardError) as zero_variance:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.0,
                se_t=0.0,
                se_c=0.1,
            )
        assert zero_variance.value.reason == "zero_variance"
        with pytest.raises(LiftGuardError) as nonfinite:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.0,
                se_t=math.nan,
                se_c=0.1,
            )
        assert nonfinite.value.reason == "nonfinite_se"
        with pytest.raises(LiftGuardError) as unreliable:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.0,
                se_t=0.4,
                se_c=0.4,
            )
        assert unreliable.value.reason == "delta_method_unreliable"

    def test_closed_form_matches_measured_basis_fixture(self):
        """Reproduces the measured-basis fixture (n_c=240, n_t=260, control
        mean 4.10 var 30, treatment mean 4.55 var 41): value/lb/ub must
        equal the closed form computed independently from nn.posterior to
        rel=1e-12 (not the 6-decimal display-rounded literals, ~128 ulps
        off), and must sit within 4 Monte-Carlo standard deviations of the
        sampled result measured over 200 seeds (mean +10.971259% sd
        0.1226%; CI lower mean -12.729880% sd 0.1970%)."""
        se_c = _se_log_mean(30.0, 4.10, 240)
        se_t = _se_log_mean(41.0, 4.55, 260)
        log_c = math.log(4.10)
        log_t = math.log(4.55)

        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=log_t - log_c,
            se_t=se_t,
            se_c=se_c,
        )

        # Independently re-derive the closed form from the conjugate update.
        posterior = normal_posterior(log_t - log_c, math.sqrt(se_t**2 + se_c**2))
        z = _norm_dist.ppf(0.975)
        expected_value = math.exp(posterior.mu) - 1.0
        expected_lb = math.exp(posterior.mu - z * posterior.sigma) - 1.0
        expected_ub = math.exp(posterior.mu + z * posterior.sigma) - 1.0

        lift = result.require_lift()
        assert lift.value == pytest.approx(expected_value, rel=1e-12)
        assert lift.lb == pytest.approx(expected_lb, rel=1e-12)
        assert lift.ub == pytest.approx(expected_ub, rel=1e-12)

        # Within 4 MC sd of the old sampled estimator (200-seed measurement).
        assert lift.lb is not None
        assert abs(lift.value - 0.10971259) < 4 * 0.001226
        assert abs(lift.lb - (-0.12729880)) < 4 * 0.001970

    def test_informative_prior_shrinks_toward_prior_mean(self):
        """With an informative prior, the point estimate and interval match
        the analytic conjugate result exp(v*(m0/tau^2 + delta_hat/se^2)) - 1,
        v = 1/(1/tau^2 + 1/se^2) - the criterion that catches using the raw
        log_rr/se_log_rr instead of the posterior's mu_n/sigma_n, invisible
        under the default near-flat prior."""
        log_t, se_t, log_c, se_c = 0.5, 0.12, 0.1, 0.09
        delta_hat = log_t - log_c
        se = math.sqrt(se_t**2 + se_c**2)
        m0, tau = 0.0, 0.05

        v = 1.0 / (1.0 / tau**2 + 1.0 / se**2)
        mu_n = v * (m0 / tau**2 + delta_hat / se**2)
        sigma_n = math.sqrt(v)
        z = _norm_dist.ppf(0.975)
        expected_value = math.exp(mu_n) - 1.0
        expected_lb = math.exp(mu_n - z * sigma_n) - 1.0
        expected_ub = math.exp(mu_n + z * sigma_n) - 1.0

        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=log_t - log_c,
            se_t=se_t,
            se_c=se_c,
            prior=Normal(mu=m0, sigma=tau),
        )

        assert result.require_lift().value == pytest.approx(expected_value, rel=1e-12)
        assert result.require_lift().lb == pytest.approx(expected_lb, rel=1e-12)
        assert result.require_lift().ub == pytest.approx(expected_ub, rel=1e-12)

        # Shrinks toward the prior mean (0.0) and narrows relative to the
        # flat-prior (raw log_rr/se_log_rr) interval.
        flat_value = math.exp(delta_hat) - 1.0
        flat_width = math.exp(delta_hat + z * se) - math.exp(delta_hat - z * se)
        shrunk_width = expected_ub - expected_lb
        assert abs(expected_value) < abs(flat_value)
        assert shrunk_width < flat_width

    def test_interval_symmetric_in_log_space(self):
        """Catches a one-sided-z botch (ppf(1-alpha) instead of
        ppf(1-alpha/2)): log1p(ub) and log1p(lb) must be equidistant from
        the posterior mean mu_n."""
        log_t, se_t, log_c, se_c = 0.5, 0.12, 0.1, 0.09
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=log_t - log_c,
            se_t=se_t,
            se_c=se_c,
        )
        posterior = normal_posterior(log_t - log_c, math.sqrt(se_t**2 + se_c**2))
        mu_n = posterior.mu

        lift = result.require_lift()
        assert lift.ub is not None and lift.lb is not None
        upper_dist = math.log1p(lift.ub) - mu_n
        lower_dist = mu_n - math.log1p(lift.lb)
        assert upper_dist == pytest.approx(lower_dist, rel=1e-12)

    def test_log_mean_log_se_are_raw_not_posterior_default_prior(self):
        """log_mean/log_se are the RAW pre-update log_rr/se_log_rr, not
        nn.posterior.mu/sigma. Under the default near-flat prior the two
        agree to high precision, so this case alone cannot discriminate.
        See the informative-prior case below for the one that can."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.1,
            se_t=0.12,
            se_c=0.09,
        )
        assert result.require_lift().log_mean == pytest.approx(0.4)  # log_t - log_c
        assert result.require_lift().log_se == pytest.approx(math.sqrt(0.12**2 + 0.09**2))

    def test_log_mean_log_se_discriminate_under_informative_prior(self):
        """THE discriminating test: with an informative prior, the
        conjugately-updated posterior mean/sd differ materially from the
        raw log_rr/se_log_rr - log_mean/log_se must be the RAW pair
        regardless, or the heterogeneity machinery's shrinkage would
        double-apply (the prior shrinks once, then the marginalised
        posterior shrinks again)."""
        log_t, se_t, log_c, se_c = 0.5, 0.12, 0.1, 0.09
        log_rr = log_t - log_c
        se_log_rr = math.sqrt(se_t**2 + se_c**2)

        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=log_t - log_c,
            se_t=se_t,
            se_c=se_c,
            prior=Normal(mu=0.0, sigma=1.0),
        )
        assert result.require_lift().log_mean == pytest.approx(log_rr)
        assert result.require_lift().log_se == pytest.approx(se_log_rr)

        # The conjugately-updated posterior (what a WRONG implementation
        # would store) must differ materially from the raw pair, or this case wouldn't discriminate.
        posterior = normal_posterior(log_rr, se_log_rr, prior=Normal(mu=0.0, sigma=1.0))
        assert abs(posterior.mu - log_rr) / abs(log_rr) > 1e-3
        assert abs(posterior.sigma - se_log_rr) / se_log_rr > 1e-3

    def test_abs_diff_abs_se_pass_through_verbatim(self):
        """abs_diff/abs_se are the caller's values, untouched by inference."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            abs_diff=2.5,
            abs_se=0.4,
        )
        assert result.abs_diff == 2.5
        assert result.abs_se == 0.4

    def test_abs_diff_abs_se_default_none(self):
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
        )
        assert result.abs_diff is None
        assert result.abs_se is None

    def test_null_lift_preferred_direction_default_reproduces_today(self):
        """Defaults (null_lift=0.0, preferred_direction=None) leave every
        pre-existing call site's output unchanged - the null shift is
        purely additive metadata, never a change to the posterior."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
        )
        assert result.null_lift == 0.0
        assert result.preferred_direction is None

    def test_null_lift_stamped_without_changing_the_interval(self):
        """A nonzero null_lift changes nothing about where the CI sits;
        only the decision read off it changes (see LiftEstimate.stat_sig/
        prob_favorable). The interval must be byte-identical to the
        null_lift=0 call."""
        baseline = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
        )
        shifted = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            null_lift=-0.01,
            preferred_direction="increase",
        )
        assert shifted.require_lift().value == pytest.approx(baseline.require_lift().value)
        assert shifted.require_lift().lb == pytest.approx(baseline.require_lift().lb)
        assert shifted.require_lift().ub == pytest.approx(baseline.require_lift().ub)
        assert shifted.null_lift == -0.01
        assert shifted.preferred_direction == "increase"

    def test_null_lift_at_or_below_minus_one_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.5 - 0.0,
                se_t=0.1,
                se_c=0.1,
                null_lift=-1.0,
            )
        assert exc_info.value.code == "estimation.inference.null_lift_log"

    @pytest.mark.parametrize("bad_null_lift", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_null_lift_raises(self, bad_null_lift):
        """NaN/+-inf null_lift must be refused by name before the -1 log-
        domain guard: -inf also satisfies `<= -1.0`, so the finite check
        must fire first or the wrong code (and a NaN'd p-value downstream)
        would leak through."""
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.5 - 0.0,
                se_t=0.1,
                se_c=0.1,
                null_lift=bad_null_lift,
            )
        assert exc_info.value.code == "estimation.inference.infer_ate_null_lift_finite"
        context_value = cast("float", exc_info.value.context["null_lift"])
        if math.isnan(bad_null_lift):
            assert math.isnan(context_value)
        else:
            assert context_value == bad_null_lift
        with pytest.raises(TypeError):
            exc_info.value.context["null_lift"] = 0.0  # ty: ignore  -- context is immutable

    def test_prob_favorable_matches_prob_beyond_for_increase(self):
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            null_lift=-0.01,
            preferred_direction="increase",
        )
        assert result.prob_favorable() == pytest.approx(result.prob_beyond(-0.01))

    def test_prob_favorable_flips_for_decrease(self):
        """A decrease-preferred (e.g. latency) guardrail's favorable side is
        BELOW the null - prob_favorable is 1 - prob_beyond, not prob_beyond."""
        result = infer_lift(
            metric="latency_ms",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.0 - 0.0,
            se_t=0.1,
            se_c=0.1,
            null_lift=0.02,
            preferred_direction="decrease",
        )
        assert result.prob_favorable() == pytest.approx(1.0 - result.prob_beyond(0.02))

    def test_prob_favorable_neutral_matches_prob_beyond(self):
        """``preferred_direction="neutral"`` behaves as increase: favorable
        is the side ABOVE the null, same as prob_beyond."""
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
            null_lift=-0.01,
            preferred_direction="neutral",
        )
        assert result.prob_favorable() == pytest.approx(result.prob_beyond(-0.01))

    def test_prob_favorable_without_preferred_direction_raises(self):
        result = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.5 - 0.0,
            se_t=0.1,
            se_c=0.1,
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            result.prob_favorable()
        assert exc_info.value.code == "estimation.results.lift.liftestimate_prob_favorable"


class TestNullAbs:
    """infer_lift's absolute-scale null boundary + additive interval
    endpoints, mirroring the relative null_lift suite above."""

    def _kwargs(self, **overrides):
        base = {
            "metric": "rev",
            "group_id": "B",
            "method": "unadjusted",
            "method_role": "decision",
            "log_rr": 0.5 - 0.0,
            "se_t": 0.1,
            "se_c": 0.1,
        }
        base.update(overrides)
        return base

    def test_null_abs_stamped_and_abs_endpoints_computed(self):
        est = infer_lift(**self._kwargs(abs_diff=0.02, abs_se=0.005, null_abs=-0.01))
        assert est.require_lift().level == pytest.approx(0.95)  # default two-sided alpha=0.05
        z = _norm_dist.ppf(1.0 - 0.05 / 2.0)
        assert est.null_abs == -0.01
        assert est.abs_lb == pytest.approx(0.02 - z * 0.005)
        assert est.abs_ub == pytest.approx(0.02 + z * 0.005)
        # Relative interval identical to a null_abs=None call - the
        # boundary is stamped, never shifts the posterior.
        baseline = infer_lift(**self._kwargs(abs_diff=0.02, abs_se=0.005))
        assert est.require_lift().value == pytest.approx(baseline.require_lift().value)
        assert est.require_lift().lb == pytest.approx(baseline.require_lift().lb)
        assert est.require_lift().ub == pytest.approx(baseline.require_lift().ub)

    def test_abs_endpoints_computed_even_without_null_abs(self):
        """The additive endpoints are the absolute-scale reading of the
        same comparison - populated whenever the abs pair exists, not
        only when a boundary is declared."""
        est = infer_lift(**self._kwargs(abs_diff=0.02, abs_se=0.005))
        assert est.require_lift().level == pytest.approx(0.95)  # default two-sided alpha=0.05
        z = _norm_dist.ppf(1.0 - 0.05 / 2.0)
        assert est.null_abs is None
        assert est.abs_lb == pytest.approx(0.02 - z * 0.005)
        assert est.abs_ub == pytest.approx(0.02 + z * 0.005)

    def test_abs_endpoints_at_one_sided_alpha_eff(self):
        """One-sided alternative doubles alpha for the displayed interval;
        the additive endpoints use the estimate's own alpha_eff, not the
        raw alpha."""
        est = infer_lift(
            **self._kwargs(abs_diff=0.02, abs_se=0.005, alternative="greater", alpha=0.05)
        )
        z = _norm_dist.ppf(1.0 - 0.10 / 2.0)  # alpha_eff = 2 * 0.05
        assert est.require_lift().level == pytest.approx(0.90)
        assert est.abs_lb == pytest.approx(0.02 - z * 0.005)
        assert est.abs_ub == pytest.approx(0.02 + z * 0.005)

    def test_null_abs_with_missing_abs_se_leaves_endpoints_none(self):
        est = infer_lift(**self._kwargs(abs_diff=0.02, null_abs=-0.01))
        assert est.null_abs == -0.01
        assert est.abs_lb is None
        assert est.abs_ub is None

    @pytest.mark.parametrize("null_abs", [None, -0.01])
    def test_se_only_sequential_refuses_even_with_absolute_sidecars(self, null_abs):
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            infer_lift(
                **self._kwargs(
                    inference_spec=AlwaysValid(registration=registration("gaussian")),
                    abs_diff=0.02,
                    abs_se=0.005,
                    null_abs=null_abs,
                    n_comparison=100,
                )
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_prob_favorable_additive_branch(self):
        """Increase-preferred with null_abs=-0.01: P(true abs diff > -0.01)
        from Normal(abs_diff, abs_se), hand-computed via norm.cdf."""
        est = infer_lift(
            **self._kwargs(
                abs_diff=0.02,
                abs_se=0.005,
                null_abs=-0.01,
                preferred_direction="increase",
            )
        )
        expected = 1.0 - _norm_dist.cdf((-0.01 - 0.02) / 0.005)
        assert est.prob_favorable() == pytest.approx(expected)

    def test_prob_favorable_additive_flips_for_decrease(self):
        """Decrease-preferred (e.g. a cost guardrail): favorable is BELOW
        the additive null, exactly as the relative branch flips."""
        est = infer_lift(
            **self._kwargs(
                abs_diff=-0.005,
                abs_se=0.01,
                null_abs=0.01,
                preferred_direction="decrease",
            )
        )
        p_greater = 1.0 - _norm_dist.cdf((0.01 - (-0.005)) / 0.01)
        assert est.prob_favorable() == pytest.approx(1.0 - p_greater)

    def test_prob_favorable_additive_with_missing_abs_se_raises(self):
        """The additive decision is unavailable, loudly - no silent
        fallback to the relative interval."""
        est = infer_lift(
            **self._kwargs(
                abs_diff=0.02,
                null_abs=-0.01,
                preferred_direction="increase",
            )
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            est.prob_favorable()
        assert exc_info.value.code == "estimation.results.lift.p_value_null_abs_missing_abs_se"

    def test_null_abs_default_none_reproduces_today(self):
        """Bit-identical backcompat: omitting null_abs= stamps None and
        leaves the relative output unchanged."""
        est = infer_lift(**self._kwargs())
        assert est.null_abs is None
        assert est.abs_lb is None
        assert est.abs_ub is None
        assert est.null_lift == 0.0

    @pytest.mark.parametrize("bad_null_abs", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_null_abs_raises(self, bad_null_abs):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(**self._kwargs(abs_diff=0.02, abs_se=0.005, null_abs=bad_null_abs))
        assert exc_info.value.code == "estimation.inference.infer_ate_null_abs_finite"
        context_value = cast("float", exc_info.value.context["null_abs"])
        if math.isnan(bad_null_abs):
            assert math.isnan(context_value)
        else:
            assert context_value == bad_null_abs
        with pytest.raises(TypeError):
            exc_info.value.context["null_abs"] = 0.0  # ty: ignore  -- context is immutable

    def test_p_value_additive_branch(self):
        """Increase-preferred with null_abs=-0.01: the additive tail from
        Normal(abs_diff, abs_se) against null_abs -- the same construction
        `prob_favorable` uses -- hand-computed via norm.cdf, one-sided
        "greater"."""
        est = infer_lift(
            **self._kwargs(abs_diff=0.02, abs_se=0.005, null_abs=-0.01, alternative="greater")
        )
        expected = _norm_dist.cdf((-0.01 - 0.02) / 0.005)
        assert est.p_value() == pytest.approx(expected)

    def test_p_value_additive_branch_less(self):
        est = infer_lift(
            **self._kwargs(abs_diff=-0.005, abs_se=0.01, null_abs=0.01, alternative="less")
        )
        z = (0.01 - (-0.005)) / 0.01
        expected = 1.0 - _norm_dist.cdf(z)
        assert est.p_value() == pytest.approx(expected)

    def test_p_value_additive_branch_two_sided(self):
        est = infer_lift(**self._kwargs(abs_diff=0.02, abs_se=0.005, null_abs=-0.01))
        z = (-0.01 - 0.02) / 0.005
        expected = 2.0 * min(_norm_dist.cdf(z), 1.0 - _norm_dist.cdf(z))
        assert est.p_value() == pytest.approx(expected)

    def test_p_value_additive_never_contradicts_stat_sig(self):
        """The additive p-value's guarantee: p_value() <= alpha exactly when the
        additive interval (abs_lb/abs_ub vs null_abs) excludes null_abs,
        for the same alpha the interval was cut at -- BH selection built
        from p_value() can therefore never contradict this row's own
        stat_sig, unlike the pre-fix code (which tested the unrelated
        relative posterior against 0 and ignored null_abs entirely)."""
        # All three alternatives, each straddling its own margin, so the
        # 2*min(cdf,sf) two-sided tail and the "less" upper-tail branch are
        # proven consistent with their intervals too -- not only "greater".
        for abs_diff, alternative in (
            (0.02, "greater"),  # comfortably past the margin
            (-0.0099, "greater"),  # just inside it
            (-0.02, "greater"),  # fails it
            (0.02, "two-sided"),  # excludes null_abs on the high side
            (-0.05, "two-sided"),  # excludes it on the low side
            (-0.01, "two-sided"),  # sits on the margin
            (-0.03, "less"),  # comfortably below the margin
            (-0.011, "less"),  # just above it (not significant)
            (0.02, "less"),  # above it -> fails
        ):
            est = infer_lift(
                **self._kwargs(
                    abs_diff=abs_diff,
                    abs_se=0.005,
                    null_abs=-0.01,
                    alternative=alternative,
                    alpha=0.05,
                )
            )
            assert est.null_abs is not None
            # stat_sig() is the row's own interval verdict (abs_lb/abs_ub vs
            # null_abs, honoring alternative); p_value() must agree with it.
            assert (est.p_value() <= 0.05) == est.stat_sig()

    def test_p_value_additive_cluster_robust_refuses(self):
        """Mirrors `prob_favorable`'s own refusal: a cluster-robust
        null_abs row's interval is a t-quantile pair, so a Normal tail
        here would understate the uncertainty the interval itself
        reports."""
        est = infer_lift(**self._kwargs(abs_diff=0.02, abs_se=0.005, null_abs=-0.01, dof=5.0))
        with pytest.raises(InvalidRequestError) as exc_info:
            est.p_value()
        assert exc_info.value.code == "estimation.results.lift.p_value_cluster_robust_null_abs"

    def test_p_value_additive_with_missing_abs_se_raises(self):
        est = infer_lift(**self._kwargs(abs_diff=0.02, null_abs=-0.01))
        with pytest.raises(InvalidRequestError) as exc_info:
            est.p_value()
        assert exc_info.value.code == "estimation.results.lift.p_value_null_abs_missing_abs_se"


class TestOneSidedInferAte:
    """infer_ate's one-sided support, mirroring TestOneSidedInferLift's
    coverage (tests/estimation/test_sequential_engine.py) for the sibling
    additive-scale estimator."""

    def _scores(self, se: float) -> ScoreStats:
        # n=1, sum_psi2=se**2 -> se() == se exactly (normalizer is n when
        # sum_d_tilde2 is unset, i.e. the IPTW case).
        return ScoreStats(metric="m", contrast="t", n=1, sum_psi=0.0, sum_psi2=se**2)

    def test_greater_is_the_doubled_alpha_two_sided_interval_labeled(self):
        scores = self._scores(se=0.05)
        one = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
            alpha=0.05,
            alternative="greater",
        )
        two = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
            alpha=0.10,
        )
        # Same interval numbers, honest two-sided level, directional label.
        assert one.require_lift().lb == pytest.approx(two.require_lift().lb, rel=1e-12)
        assert one.require_lift().ub == pytest.approx(two.require_lift().ub, rel=1e-12)
        assert one.require_lift().level == pytest.approx(0.90)
        assert one.alternative == "greater"
        assert two.alternative == "two-sided"

    def test_less_same_interval_different_label(self):
        scores = self._scores(se=0.05)
        one = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
            alpha=0.05,
            alternative="less",
        )
        two = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
            alpha=0.10,
        )
        assert one.require_lift().ub == pytest.approx(two.require_lift().ub, rel=1e-12)
        assert one.alternative == "less"

    def test_scale_still_stamped_linear_under_one_sided(self):
        est = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=self._scores(se=0.05),
            alternative="greater",
        )
        assert est.scale == "linear"

    def test_unknown_alternative_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=self._scores(se=0.05),
                alternative="bigger",
            )
        assert exc_info.value.code == "estimation.binomial.unknown_alternative"

    def test_alpha_domain_validated_like_infer_lift(self):
        """infer_ate shares infer_lift's alpha hole symmetrically - both
        validate 0 < alpha < 1 and alpha_eff < 1 at entry."""
        scores = self._scores(se=0.05)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=scores,
                alpha=0.0,
            )
        assert exc_info.value.code == "estimation.diagnostics.alpha"
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=scores,
                alpha=0.6,
                alternative="greater",
            )
        assert exc_info.value.code == "estimation.inference.one_sided_alpha_doubles"

    def test_minimum_alpha_survives_one_sided_effective_tail(self):
        tiny = math.nextafter(0.0, 1.0)
        result = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=self._scores(se=0.05),
            alpha=tiny,
            alternative="greater",
        )
        lift = result.require_lift()
        assert lift.alpha == 2.0 * tiny
        assert lift.lb is not None and lift.ub is not None
        assert math.isfinite(lift.lb) and math.isfinite(lift.ub)
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=self._scores(se=0.05),
                alpha=tiny,
            )
        assert exc_info.value.code == "estimation.encouragement.alpha_eff_too"

    def test_default_alternative_reproduces_two_sided_behavior(self):
        """Default-unchanged regression: omitting alternative= is byte-
        identical to explicit alternative="two-sided" - the only behavior
        infer_ate had before this change."""
        scores = self._scores(se=0.05)
        default = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
        )
        explicit = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
            alternative="two-sided",
        )
        assert default.require_lift().lb == pytest.approx(explicit.require_lift().lb, rel=1e-12)
        assert default.require_lift().ub == pytest.approx(explicit.require_lift().ub, rel=1e-12)
        assert default.require_lift().level == pytest.approx(0.95)
        assert default.alternative == "two-sided"

    @pytest.mark.parametrize("bad_null_lift", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_null_lift_raises(self, bad_null_lift):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=self._scores(se=0.05),
                null_lift=bad_null_lift,
            )
        assert exc_info.value.code == "estimation.inference.infer_ate_null_lift_finite"
        context_value = cast("float", exc_info.value.context["null_lift"])
        if math.isnan(bad_null_lift):
            assert math.isnan(context_value)
        else:
            assert context_value == bad_null_lift
        with pytest.raises(TypeError):
            exc_info.value.context["null_lift"] = 0.0  # ty: ignore  -- context is immutable

    @pytest.mark.parametrize("bad_null_abs", [float("nan"), float("inf"), float("-inf")])
    def test_nonfinite_null_abs_raises(self, bad_null_abs):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=self._scores(se=0.05),
                abs_diff=0.02,
                abs_se=0.005,
                null_abs=bad_null_abs,
            )
        assert exc_info.value.code == "estimation.inference.infer_ate_null_abs_finite"
        context_value = cast("float", exc_info.value.context["null_abs"])
        if math.isnan(bad_null_abs):
            assert math.isnan(context_value)
        else:
            assert context_value == bad_null_abs

    def test_absolute_scale_combination_refusal_precedes_finite_guard(self):
        """A NaN null_lift on an absolute-native row satisfies both the
        pre-existing scale-mismatch refusal (NaN != 0.0 is True) and the
        new finite guard; the absolute-scale refusal must still win, because
        refusal precedence is part of the contract."""
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_ate(
                metric="m",
                group_id="t",
                method="iptw",
                method_role="decision",
                point=0.10,
                scores=self._scores(se=0.05),
                value_scale="absolute",
                null_lift=float("nan"),
            )
        assert exc_info.value.code == "estimation.inference.infer_ate_null_lift_on_absolute_metric"

    def test_valid_finite_null_lift_and_null_abs_unchanged(self):
        """Ordinary finite values still flow through to calculation and
        get stamped exactly as before."""
        row = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=self._scores(se=0.05),
            abs_diff=0.02,
            abs_se=0.005,
            null_lift=0.04,
            null_abs=-0.01,
        )
        assert row.null_lift == 0.04
        assert row.null_abs == -0.01


@pytest.mark.slow
def test_always_valid_one_sided_minimum_alpha_retains_exact_bound():
    from fractions import Fraction

    from increment import SequentialCell, estimate_sequential
    from tests.sequential_cases import capture, records

    tiny = math.nextafter(0.0, 1.0)
    reg = registration(
        cells=(
            SequentialCell(
                metric="outcome",
                group_id="treatment",
                alpha=Fraction(tiny),
                alternative="greater",
            ),
        )
    )
    result = estimate_sequential(
        capture(reg, records([0] * 600, [1] * 600)),
        AlwaysValid(registration=reg),
    ).results[0]
    assert result.stat_sig()
    bounds = result.require_sequential_result().bounds
    assert bounds.alpha == Fraction(tiny)
    assert bounds.lower is not None and bounds.lower > 1
    assert result.lift is None


class TestAbsSeValidation:
    """infer_lift's absolute Wald endpoints must refuse a
    nonpositive or nonfinite abs_se/abs_diff instead of silently
    returning an inverted (lb > ub) interval."""

    def _kwargs(self, **overrides):
        base = {
            "metric": "rev",
            "group_id": "B",
            "method": "unadjusted",
            "method_role": "decision",
            "log_rr": 0.5 - 0.0,
            "se_t": 0.1,
            "se_c": 0.1,
        }
        base.update(overrides)
        return base

    def test_negative_abs_se_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(**self._kwargs(abs_diff=1.0, abs_se=-1.0))
        assert exc_info.value.code == "estimation.inference.infer_lift_abs_se_positive"

    def test_zero_abs_se_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(**self._kwargs(abs_diff=1.0, abs_se=0.0))
        assert exc_info.value.code == "estimation.inference.infer_lift_abs_se_positive"

    def test_nan_abs_se_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(**self._kwargs(abs_diff=1.0, abs_se=float("nan")))
        assert exc_info.value.code == "estimation.inference.infer_lift_abs_se_finite"

    def test_infinite_abs_diff_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(**self._kwargs(abs_diff=float("inf"), abs_se=0.1))
        assert exc_info.value.code == "estimation.inference.infer_lift_abs_diff_finite"

    def test_missing_abs_diff_and_abs_se_leaves_endpoints_none(self):
        """The deliberately-unavailable case: passing neither sidecar is
        not a validation failure, it is opting out of the additive
        reading entirely."""
        est = infer_lift(**self._kwargs())
        assert est.abs_lb is None
        assert est.abs_ub is None

    def test_valid_abs_se_still_computes_noninverted_interval(self):
        """Regression guard: a normal positive finite abs_se still
        produces the same correct interval as before the guard."""
        est = infer_lift(**self._kwargs(abs_diff=1.0, abs_se=0.5))
        z = _norm_dist.ppf(1.0 - 0.05 / 2.0)
        assert est.abs_lb == pytest.approx(1.0 - z * 0.5)
        assert est.abs_ub == pytest.approx(1.0 + z * 0.5)
        assert est.abs_lb is not None and est.abs_ub is not None
        assert est.abs_lb < est.abs_ub


class TestPersistedSamplingReference:
    def _infer_lift_kwargs(self, **overrides):
        base = {
            "metric": "rev",
            "group_id": "B",
            "method": "unadjusted",
            "method_role": "decision",
            "log_rr": 0.05 - 0.0,
            "se_t": 0.015,
            "se_c": 0.013,
            "preferred_direction": "increase",
        }
        base.update(overrides)
        return base

    def _welch_row(self, **overrides):
        return infer_lift(
            **self._infer_lift_kwargs(
                arm_ns=(40, 45),
                abs_diff=1.0,
                abs_se=0.4,
                **overrides,
            )
        )

    def test_infer_ate_stamps_reference_and_keeps_cluster_additive_bounds(self):
        scores = ScoreStats(metric="m", contrast="t", n=1, sum_psi=0.0, sum_psi2=0.05**2)
        normal = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
        )
        row = infer_ate(
            metric="m",
            group_id="t",
            method="iptw",
            method_role="decision",
            point=0.10,
            scores=scores,
            dof=7.0,
            abs_diff=1.0,
            abs_se=0.4,
            null_lift=0.04,
        )
        crit = _t_dist.isf(0.025, 7.0)
        assert (normal.reference_kind, normal.reference_df) == ("normal", None)
        assert (row.reference_kind, row.reference_df) == ("t", 7.0)
        assert row.abs_lb == pytest.approx(1.0 - crit * 0.4)
        assert row.abs_ub == pytest.approx(1.0 + crit * 0.4)
        assert row.p_value() == pytest.approx(2.0 * _t_dist.sf(abs((0.10 - 0.04) / 0.05), 7.0))

    @pytest.mark.parametrize("alternative", ["two-sided", "greater"])
    def test_relative_p_value_uses_its_welch_df_at_tiny_alpha(self, alternative):
        row = self._welch_row(alpha=1e-12, alternative=alternative, null_lift=0.02)
        assert row.require_lift().log_mean is not None
        assert row.require_lift().log_se is not None
        assert row.reference_df is not None
        null = math.log1p(0.02)
        z = (row.require_lift().log_mean - null) / row.require_lift().log_se
        expected = (
            2.0 * _t_dist.sf(abs(z), row.reference_df)
            if alternative == "two-sided"
            else _t_dist.sf(z, row.reference_df)
        )
        assert row.dof is None
        assert row.reference_kind == "t"
        assert row.p_value() == pytest.approx(expected)

    def test_welch_reference_posterior_is_the_row_s_own_log_moments(self):
        """A Welch reference corrects the SAMPLING distribution for a variance
        estimated from the arms; it does not widen the posterior. Recovering
        (mu, sigma) by inverting the stored t endpoints with a Normal z would
        report sigma inflated by the t/z ratio, so the decision statistics are
        the Normal tails at lift.log_mean/lift.log_se."""
        from increment.estimation.inference import normal_posterior

        row = self._welch_row()
        lift = row.require_lift()
        assert row.dof is None and row.reference_kind == "t"
        assert lift.log_mean is not None and lift.log_se is not None

        expected = normal_posterior(lift.log_mean, lift.log_se)
        assert row.prob_beyond(0.25) == pytest.approx(expected.survival(math.log1p(0.25)))
        assert row.prob_within(0.10) == pytest.approx(
            expected.probability_between(math.log1p(-0.10), math.log1p(0.10))
        )
        assert row.risk_if_shipped() == pytest.approx(expected.expected_negative_part(scale="log"))

        # The endpoint inversion this replaces would have landed t/z away.
        assert lift.alpha is not None and row.reference_df is not None
        z = float(_norm_dist.isf(lift.alpha / 2.0))
        inverted = (math.log1p(lift.value) - math.log1p(lift.lb)) / z
        assert inverted == pytest.approx(
            lift.log_se * _t_dist.isf(lift.alpha / 2.0, row.reference_df) / z, rel=1e-12
        )
        assert inverted > lift.log_se

    @pytest.mark.parametrize(
        ("method", "args"),
        [
            ("chance_to_beat", ()),
            ("prob_beyond", (0.01,)),
            ("prob_favorable", ()),
            ("prob_within", (0.10,)),
            ("chance_to_beat_favorable", ()),
            ("risk_if_shipped", ()),
            ("risk_if_shipped_favorable", ()),
        ],
    )
    def test_welch_reference_decision_stats_read_that_posterior(self, method, args):
        from increment.estimation.inference import normal_posterior

        row = self._welch_row(null_lift=0.02)
        lift = row.require_lift()
        assert lift.log_mean is not None and lift.log_se is not None
        posterior = normal_posterior(lift.log_mean, lift.log_se)
        expected = {
            "chance_to_beat": lambda: posterior.survival(0.0),
            "prob_beyond": lambda: posterior.survival(math.log1p(0.01)),
            "prob_favorable": lambda: posterior.survival(math.log1p(row.null_lift)),
            "prob_within": lambda: posterior.probability_between(
                math.log1p(-0.10), math.log1p(0.10)
            ),
            "chance_to_beat_favorable": lambda: posterior.survival(0.0),
            "risk_if_shipped": lambda: posterior.expected_negative_part(scale="log"),
            "risk_if_shipped_favorable": lambda: posterior.expected_negative_part(scale="log"),
        }[method]()
        assert getattr(row, method)(*args) == pytest.approx(expected)

    def test_welch_reference_withholds_every_additive_decision(self):
        row = self._welch_row(null_abs=0.0)
        assert row.dof is None
        assert row.reference_kind == "t"
        assert row.reference_df is not None
        assert row.abs_diff == 1.0 and row.abs_se == 0.4
        assert row.abs_lb is None and row.abs_ub is None
        assert row.require_lift().lb is not None and row.require_lift().ub is not None
        assert row.stat_sig() is False

        with pytest.raises(InvalidRequestError) as p_value_exc:
            row.p_value()
        assert p_value_exc.value.code == "estimation.results.lift.p_value_cluster_robust_null_abs"
        assert p_value_exc.value.context["reference_df"] == row.reference_df

        with pytest.raises(InvalidRequestError) as favorable_exc:
            row.prob_favorable()
        assert favorable_exc.value.code == "estimation.results.lift.p_value_cluster_robust_null_abs"
        assert favorable_exc.value.context["reference_df"] == row.reference_df

        round_tripped = LiftEstimate.model_validate_json(row.model_dump_json())
        assert round_tripped == row
        assert round_tripped.abs_diff == 1.0
        assert round_tripped.abs_se == 0.4
        assert round_tripped.abs_lb is None and round_tripped.abs_ub is None


class TestResolveFixedHorizon:
    def test_normal_reference_when_no_dof_or_arm_ns(self):
        ref = _resolve_fixed_horizon(
            0.05,
            "two-sided",
            dof=None,
            arm_ns=None,
            prior=None,
        )
        assert isinstance(ref, FixedHorizonReference)
        assert ref.reference == SamplingReference(kind="normal", df=None)
        assert ref.crit == pytest.approx(float(_norm_dist.isf(0.025)))
        assert ref.alpha_eff == pytest.approx(0.05)

    def test_dof_wins_over_arm_ns(self):
        row = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.05,
            se_t=0.2,
            se_c=0.1,
            dof=5.0,
            arm_ns=(3, 4),
            alternative="two-sided",
        )
        assert (row.reference_kind, row.reference_df) == ("t", 5.0)
        lift = row.require_lift()
        assert lift.log_mean is not None and lift.log_se is not None and lift.alpha is not None
        welch_df = (0.2**2 + 0.1**2) ** 2 / (0.2**4 / (3 - 1) + 0.1**4 / (4 - 1))
        assert row.reference_df != pytest.approx(welch_df)
        crit = float(_t_dist.isf(lift.alpha / 2.0, 5.0))
        assert lift.ub == pytest.approx(math.exp(lift.log_mean + crit * lift.log_se) - 1.0)
        assert lift.lb == pytest.approx(math.exp(lift.log_mean - crit * lift.log_se) - 1.0)

    def test_welch_satterthwaite_reference_from_arm_ns(self):
        n_t, n_c, se_t, se_c = 3, 4, 0.2, 0.1
        ref = _resolve_fixed_horizon(
            0.05,
            "two-sided",
            dof=None,
            arm_ns=(n_t, n_c),
            se_t=se_t,
            se_c=se_c,
            prior=None,
        )
        welch_df = (se_t**2 + se_c**2) ** 2 / (se_t**4 / (n_t - 1) + se_c**4 / (n_c - 1))
        assert ref.reference.kind == "t"
        assert ref.reference.df == pytest.approx(welch_df)
        assert ref.crit == pytest.approx(float(_t_dist.isf(0.025, welch_df)))

    def test_welch_dof_tiny_equal_ses_no_longer_divides_by_zero(self):
        """se_t=se_c=1e-200 underflows se**4 to exactly 0.0 in both
        the numerator and denominator of the raw formula, giving 0.0/0.0.
        ``_resolve_fixed_horizon`` is exercised directly because
        ``infer_lift``'s own se_log_rr>=0.5 guard is orthogonal to this fix
        and never rejects a *tiny* combined SE."""
        ref = _resolve_fixed_horizon(
            0.05,
            "two-sided",
            dof=None,
            arm_ns=(10, 10),
            se_t=1e-200,
            se_c=1e-200,
            prior=None,
        )
        assert ref.reference.kind == "t"
        assert ref.reference.df == pytest.approx(18.0)

    def test_welch_dof_huge_equal_ses_no_longer_overflows(self):
        """se_t=se_c=1e200 overflows se**4 in the raw formula
        (OverflowError). Exercised directly on ``_resolve_fixed_horizon``:
        ``infer_lift``'s se_log_rr>=0.5 imprecision guard would refuse a
        combined SE this huge before ever reaching the Welch branch, so
        the public entry point cannot observe this fix in isolation."""
        ref = _resolve_fixed_horizon(
            0.05,
            "two-sided",
            dof=None,
            arm_ns=(10, 10),
            se_t=1e200,
            se_c=1e200,
            prior=None,
        )
        assert ref.reference.kind == "t"
        assert ref.reference.df == pytest.approx(18.0)

    def test_welch_dof_ordinary_scale_matches_pre_fix_value(self):
        """Scale-normalizing is exactly algebraically invariant, so the
        ordinary-scale result must be unchanged (to float rounding) from
        the pre-fix formula's pinned value."""
        ref = _resolve_fixed_horizon(
            0.05,
            "two-sided",
            dof=None,
            arm_ns=(50, 40),
            se_t=0.1,
            se_c=0.15,
            prior=None,
        )
        assert ref.reference.df == pytest.approx(70.31548007838012, rel=1e-9)

    @pytest.mark.parametrize(
        ("se_t", "se_c", "expected_df"),
        [
            (1e-200, 1e200, 14.0),  # arm t negligible -> df -> n_c - 1
            (1e200, 1e-200, 9.0),  # arm c negligible -> df -> n_t - 1
        ],
    )
    def test_welch_dof_extreme_ratio_is_scale_invariant_and_arm_ordered(
        self, se_t, se_c, expected_df
    ):
        """A wrong normalization (e.g. dividing by se_t instead of the
        shared max) would still overflow/NaN on a 400-order-of-magnitude
        ratio, and mixing up which arm's (n-1) pairs with which SE would
        swap 9.0/14.0. This distinguishes both failure modes from the
        correct fix, which collapses exactly to the dominant arm's (n-1)
        once the negligible arm's normalized SE underflows to 0.0."""
        ref = _resolve_fixed_horizon(
            0.05,
            "two-sided",
            dof=None,
            arm_ns=(10, 15),
            se_t=se_t,
            se_c=se_c,
            prior=None,
        )
        assert ref.reference.kind == "t"
        assert ref.reference.df == pytest.approx(expected_df)

    def test_prior_and_dof_are_mutually_exclusive(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05,
                se_t=0.015,
                se_c=0.013,
                dof=5.0,
                prior=Normal(mu=0.0, sigma=1.0),
            )
        assert exc_info.value.code == "estimation.inference.prior_excludes_cluster_robust_t"

    def test_prior_refuses_a_welch_reference(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05,
                se_t=0.015,
                se_c=0.013,
                arm_ns=(40, 45),
                prior=Normal(mu=0.0, sigma=0.05),
            )
        assert exc_info.value.code == "estimation.inference.prior_excludes_welch_reference"
        assert exc_info.value.context["arm_ns"] == (40, 45)

    @pytest.mark.parametrize("arm_ns", [None, (40, 45)])
    def test_arm_counts_do_not_turn_estimated_standard_errors_into_certified_state(self, arm_ns):
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05,
                se_t=0.015,
                se_c=0.013,
                n_comparison=85,
                inference_spec=AlwaysValid(registration=registration("gaussian")),
                abs_diff=1.0,
                abs_se=0.4,
                arm_ns=arm_ns,
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_welch_reference_withholds_the_additive_interval(self):
        row = infer_lift(
            metric="rev",
            group_id="B",
            method="unadjusted",
            method_role="decision",
            log_rr=0.05 - 0.0,
            se_t=0.015,
            se_c=0.013,
            arm_ns=(40, 45),
            abs_diff=1.0,
            abs_se=0.4,
        )
        assert row.dof is None
        assert row.reference_kind == "t"
        assert row.reference_df is not None
        assert row.abs_lb is None
        assert row.abs_ub is None
        assert row.abs_diff == 1.0 and row.abs_se == 0.4
        assert row.require_lift().lb is not None

    def test_zero_arm_se_refuses_before_welch_resolution(self):
        from increment.estimation.inference import LiftGuardError

        with pytest.raises(LiftGuardError):
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05 - 0.0,
                se_t=0.0,
                se_c=0.0,
                arm_ns=(3, 4),
            )

    def test_huge_finite_arm_se_refuses_before_welch_resolution(self):
        from increment.estimation.inference import LiftGuardError

        with pytest.raises(LiftGuardError):
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05 - 0.0,
                se_t=1e200,
                se_c=1e200,
                arm_ns=(3, 4),
            )

    @pytest.mark.parametrize(("se_t", "se_c"), [(math.nan, 0.01), (0.01, math.nan)])
    def test_nan_arm_se_refuses_before_reference_resolution(self, se_t, se_c):
        from increment.estimation.inference import LiftGuardError

        with pytest.raises(LiftGuardError):
            infer_lift(
                metric="rev",
                group_id="B",
                method="unadjusted",
                method_role="decision",
                log_rr=0.05 - 0.0,
                se_t=se_t,
                se_c=se_c,
                arm_ns=(3, 4),
            )


class TestArmNsWelchSatterthwaite:
    """An opted-in arm_ns swaps the fixed-horizon Normal critical
    value for a Welch-Satterthwaite t reference on the ordinary iid
    two-arm path (dof=None, i.e. not cluster-robust)."""

    def _kwargs(self, **overrides):
        base = {
            "metric": "rev",
            "group_id": "B",
            "method": "unadjusted",
            "method_role": "decision",
            "log_rr": 0.5 - 0.0,
            "se_t": 0.2,
            "se_c": 0.1,
        }
        base.update(overrides)
        return base

    def test_arm_ns_omitted_reproduces_normal_critical_value(self):
        """Backcompat: no callers currently pass arm_ns, so the default
        must be bit-identical to today's Normal-quantile behavior."""
        est = infer_lift(**self._kwargs())
        z = _norm_dist.ppf(1.0 - 0.05 / 2.0)
        se_log_rr = math.sqrt(0.2**2 + 0.1**2)
        expected_lb = math.exp(0.5 - z * se_log_rr) - 1.0
        expected_ub = math.exp(0.5 + z * se_log_rr) - 1.0
        assert est.require_lift().lb == pytest.approx(expected_lb)
        assert est.require_lift().ub == pytest.approx(expected_ub)

    def test_arm_ns_uses_welch_satterthwaite_t_reference(self):
        n_t, n_c = 3, 4
        est = infer_lift(**self._kwargs(arm_ns=(n_t, n_c)))
        se_t, se_c = 0.2, 0.1
        welch_df = (se_t**2 + se_c**2) ** 2 / (se_t**4 / (n_t - 1) + se_c**4 / (n_c - 1))
        t_crit = _t_dist.isf(0.05 / 2.0, welch_df)
        se_log_rr = math.sqrt(se_t**2 + se_c**2)
        expected_lb = math.exp(0.5 - t_crit * se_log_rr) - 1.0
        expected_ub = math.exp(0.5 + t_crit * se_log_rr) - 1.0
        assert est.require_lift().lb == pytest.approx(expected_lb)
        assert est.require_lift().ub == pytest.approx(expected_ub)
        # A t reference is always wider than the Normal one it replaces.
        normal_est = infer_lift(**self._kwargs())
        lift = est.require_lift()
        normal_lift = normal_est.require_lift()
        assert lift.lb is not None and lift.ub is not None
        assert normal_lift.lb is not None and normal_lift.ub is not None
        assert lift.lb < normal_lift.lb
        assert lift.ub > normal_lift.ub

    def test_arm_ns_welch_df_is_persisted_on_the_reference(self):
        n_t, n_c = 3, 4
        est = infer_lift(**self._kwargs(arm_ns=(n_t, n_c)))
        se_t, se_c = 0.2, 0.1
        welch_df = (se_t**2 + se_c**2) ** 2 / (se_t**4 / (n_t - 1) + se_c**4 / (n_c - 1))
        assert est.reference_kind == "t"
        assert est.reference_df == pytest.approx(welch_df)

    def test_arm_ns_single_unit_arm_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(**self._kwargs(arm_ns=(1, 4)))
        assert exc_info.value.code == "estimation.inference.arm_ns_each"

    def test_arm_ns_ignored_under_cluster_robust_dof(self):
        """dof always wins: arm_ns is only consulted on the ordinary iid
        path, never overriding the explicit cluster-robust t reference."""
        with_arm_ns = infer_lift(**self._kwargs(arm_ns=(3, 4), dof=5.0))
        without_arm_ns = infer_lift(**self._kwargs(dof=5.0))
        assert with_arm_ns.require_lift().lb == pytest.approx(without_arm_ns.require_lift().lb)
        assert with_arm_ns.require_lift().ub == pytest.approx(without_arm_ns.require_lift().ub)


def test_infer_lift_extreme_alpha_refuses_instead_of_lb_minus_one_or_overflow():
    from increment.errors import InvalidRequestError
    from increment.estimation.inference import infer_lift

    for alpha in (1e-20, 1e-30):
        with pytest.raises(InvalidRequestError) as exc_info:
            infer_lift(
                "revenue",
                "treatment",
                "ttest",
                log_rr=0.0 - 0.0,
                se_t=0.35,
                se_c=0.34,
                alpha=alpha,
                dof=8,
                method_role="decision",
            )
        assert exc_info.value.code == "estimation.tails.unresolvable"


def test_infer_lift_extreme_point_estimate_refuses_instead_of_overflow():
    """log_rr=750 pushes the point back-transform's expm1 argument past
    float64's overflow threshold; the plain (non-mixture) point estimate
    must route through resolvable_expm1 and raise the coded refusal,
    not leak a bare OverflowError."""
    from increment.errors import InvalidRequestError
    from increment.estimation.inference import infer_lift

    with pytest.raises(InvalidRequestError) as exc_info:
        infer_lift(
            "revenue",
            "treatment",
            "ttest",
            log_rr=750.0 - 0.0,
            se_t=0.05,
            se_c=0.05,
            method_role="decision",
        )
    assert exc_info.value.code == "estimation.tails.unresolvable"


def test_infer_lift_extreme_mixture_point_estimate_refuses_instead_of_overflow():
    """A mixture-prior component parked at an extreme mean pushes the
    posterior median's expm1 argument past the overflow threshold; the
    mixture branch's point estimate must also route through
    resolvable_expm1 rather than leak a bare OverflowError."""
    from increment.errors import InvalidRequestError
    from increment.estimation.inference import infer_lift
    from increment.estimation.priors import MixturePrior

    with pytest.raises(InvalidRequestError) as exc_info:
        infer_lift(
            "revenue",
            "treatment",
            "ttest",
            log_rr=0.0 - 0.0,
            se_t=0.05,
            se_c=0.05,
            prior=MixturePrior(weights=(1.0,), means=(800.0,), sigmas=(0.01,)),
            method_role="decision",
        )
    assert exc_info.value.code == "estimation.tails.unresolvable"


def test_infer_lift_tiny_se_hypot_avoids_underflow():
    from increment.estimation.inference import infer_lift

    # se_t=se_c=1e-200: the squares underflow to 0.0, so sqrt(se_t**2 + se_c**2)
    # is 0; hypot stays finite and nonzero. arm_ns=(10, 10) also exercises the
    # public Welch-Satterthwaite dof. Unlike the huge-SE case, this SE stays under
    # the 0.5 reliability guard, so the result is directly observable.
    result = infer_lift(
        "revenue",
        "treatment",
        "ttest",
        log_rr=0.0 - 0.0,
        se_t=1e-200,
        se_c=1e-200,
        arm_ns=(10, 10),
        method_role="decision",
    )
    lift = result.require_lift()
    assert lift.log_se is not None
    assert lift.log_se > 0.0
    assert math.isfinite(lift.log_se)
    assert result.reference_kind == "t"
    assert result.reference_df == pytest.approx(18.0)


def test_infer_lift_matches_expm1_at_small_log_rr():
    from increment.estimation.inference import infer_lift

    result = infer_lift(
        "revenue",
        "treatment",
        "ttest",
        log_rr=1e-13 - 0.0,
        se_t=0.05,
        se_c=0.05,
        method_role="decision",
    )
    assert result.require_lift().value == pytest.approx(math.expm1(1e-13), rel=1e-12, abs=0.0)


def test_infer_lift_sets_prior_shrunk_only_when_prior_given():
    from increment.estimation.inference import Normal, infer_lift

    plain = infer_lift(
        "revenue",
        "treatment",
        "ttest",
        log_rr=0.1 - 0.0,
        se_t=0.05,
        se_c=0.05,
        method_role="decision",
    )
    assert plain.prior_shrunk is False
    shrunk = infer_lift(
        "revenue",
        "treatment",
        "ttest",
        log_rr=0.1 - 0.0,
        se_t=0.05,
        se_c=0.05,
        prior=Normal(mu=0.0, sigma=0.01),
        method_role="decision",
    )
    assert shrunk.prior_shrunk is True
