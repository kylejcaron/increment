"""Tests for the finite normal-mixture prior and the Student-t declaration."""

from __future__ import annotations

import numpy as np
import pytest

from increment.errors import InvalidRequestError
from increment.estimation.inference import infer_lift
from increment.estimation.priors import (
    MixturePosterior,
    MixturePrior,
    StudentTPrior,
    mixture_posterior,
)
from tests.mc import replicate
from tests.sequential_cases import registration


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.priors.mixture_posterior.probability",
            lambda: MixturePosterior(
                weights=np.array([1.0]), means=np.array([0.0]), sigmas=np.array([1.0])
            ).quantile(2.0),
        ),
        (
            "estimation.priors.standard_error",
            lambda: mixture_posterior(
                0.1, 0.0, MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(1.0,))
            ),
        ),
    ],
)
def test_priors_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


class TestMixturePriorValidation:
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            MixturePrior(weights=(0.5, 0.5), means=(0.0,), sigmas=(1.0, 1.0))
        assert exc_info.value.code == "estimation.priors.mixture.weights_means_sigmas"

    def test_rejects_nonpositive_weight(self):
        with pytest.raises(ValueError):
            MixturePrior(weights=(1.1, -0.1), means=(0.0, 0.0), sigmas=(1.0, 1.0))

    def test_rejects_nonpositive_sigma(self):
        with pytest.raises(ValueError):
            MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(1.0, 0.0))

    def test_rejects_weights_not_summing_to_one(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            MixturePrior(weights=(0.5, 0.4), means=(0.0, 0.0), sigmas=(1.0, 1.0))
        assert exc_info.value.code == "estimation.priors.mixture.weights_sum"

    def test_rejects_non_finite_weight(self):
        """NaN comparisons are always false, so a bare positivity/sum check
        would silently accept a NaN weight and poison the posterior."""
        with pytest.raises(InvalidRequestError) as exc_info:
            MixturePrior(weights=(float("nan"),), means=(0.0,), sigmas=(1.0,))
        assert exc_info.value.code == "estimation.priors.mixture.every_weight_mean"
        with pytest.raises(InvalidRequestError) as exc_info:
            MixturePrior(weights=(float("inf"), -float("inf")), means=(0.0, 0.0), sigmas=(1.0, 1.0))
        assert exc_info.value.code == "estimation.priors.mixture.every_weight_mean"

    def test_rejects_non_finite_mean(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            MixturePrior(weights=(1.0,), means=(float("nan"),), sigmas=(1.0,))
        assert exc_info.value.code == "estimation.priors.mixture.every_weight_mean"

    def test_rejects_non_finite_sigma(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(float("inf"),))
        assert exc_info.value.code == "estimation.priors.mixture.every_weight_mean"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            MixturePrior(weights=(), means=(), sigmas=())

    def test_components_returns_self(self):
        p = MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(1.0,))
        assert p.components() is p


class TestStudentTPrior:
    def test_declaration_is_three_fields(self):
        # The persisted identity is (nu, scale, k) - equality and repr work
        # on the declaration, never on 80 derived floats.
        assert StudentTPrior(nu=4.0, scale=0.05) == StudentTPrior(nu=4.0, scale=0.05)
        assert StudentTPrior(nu=4.0, scale=0.05) != StudentTPrior(nu=3.0, scale=0.05)
        dumped = StudentTPrior(nu=4.0, scale=0.05).model_dump()
        assert dumped == {"nu": 4.0, "scale": 0.05, "k": 80}

    def test_components_weights_sum_to_one(self):
        c = StudentTPrior(nu=4.0, scale=0.05).components()
        assert isinstance(c, MixturePrior)
        assert abs(sum(c.weights) - 1.0) < 1e-12

    def test_components_means_are_zero(self):
        c = StudentTPrior(nu=4.0, scale=0.05).components()
        assert all(m == 0.0 for m in c.means)

    def test_default_component_count(self):
        assert len(StudentTPrior(nu=4.0, scale=0.05).components().weights) == 80

    def test_components_scale_sigmas(self):
        narrow = StudentTPrior(nu=4.0, scale=0.05).components()
        wide = StudentTPrior(nu=4.0, scale=0.10).components()
        ratios = [ws / wn for ws, wn in zip(wide.sigmas, narrow.sigmas, strict=True)]
        assert all(r == pytest.approx(2.0, abs=1e-12) for r in ratios)

    def test_variance_matches_t_within_truncation(self):
        # Exact t_4 variance is nu/(nu-2)*scale^2 = 2*0.05^2; Gauss-Laguerre
        # truncation loses a little tail mass (measured ratio 0.987654 at K=80).
        c = StudentTPrior(nu=4.0, scale=0.05).components()
        mix_var = sum(w * s**2 for w, s in zip(c.weights, c.sigmas, strict=True))
        assert 0.98 < mix_var / (2.0 * 0.05**2) <= 1.0

    def test_density_matches_scipy_t(self):
        from scipy import stats

        c = StudentTPrior(nu=4.0, scale=0.05).components()
        w = np.asarray(c.weights)
        sd = np.asarray(c.sigmas)
        for x, tol in [(0.0, 1e-5), (0.05, 1e-5), (0.15, 1e-4)]:
            mix = float((w * stats.norm.pdf(x, 0.0, sd)).sum())
            exact = float(stats.t.pdf(x, 4, 0, 0.05))
            assert abs(mix / exact - 1.0) < tol

    @pytest.mark.parametrize("bad", [{"nu": 0.0}, {"nu": 201.0}, {"scale": 0.0}, {"k": 1}])
    def test_rejects_bad_parameters(self, bad):
        kwargs = {"nu": 4.0, "scale": 0.05, "k": 80} | bad
        with pytest.raises(ValueError):
            StudentTPrior(**kwargs)

    def test_large_nu_is_refused_toward_normal(self):
        # roots_genlaguerre goes NaN near nu=400; the ceiling sits at 200,
        # where a t is within 0.6% of a Normal at the 97.5% quantile - error must name Normal.
        with pytest.raises(InvalidRequestError) as exc_info:
            StudentTPrior(nu=1000.0, scale=0.05)
        assert exc_info.value.code == "estimation.priors.student_t.nu_exceeds_200"


def _t4_posterior(y=0.20, se=0.05):
    return mixture_posterior(y, se, StudentTPrior(nu=4.0, scale=0.05).components())


class TestMixturePosterior:
    def test_satisfies_lift_posterior_protocol(self):
        from increment.estimation.inference import LiftPosterior

        assert isinstance(_t4_posterior(), LiftPosterior)

    def test_quantiles_match_quadrature_ground_truth(self):
        post = _t4_posterior()
        assert post.quantile(0.5) == pytest.approx(0.1431047833, abs=1e-8)
        assert post.quantile(0.025) == pytest.approx(0.0473198635, abs=1e-8)
        assert post.quantile(0.975) == pytest.approx(0.2478514515, abs=1e-8)

    def test_cdf_quantile_roundtrip(self):
        post = _t4_posterior()
        for q in (0.01, 0.1, 0.5, 0.9, 0.99):
            assert post.cdf(post.quantile(q)) == pytest.approx(q, abs=1e-9)

    def test_survival_complements_cdf(self):
        post = _t4_posterior()
        assert post.cdf(0.1) + post.survival(0.1) == pytest.approx(1.0, abs=1e-12)

    def test_survival_at_zero_matches_ground_truth(self):
        assert _t4_posterior().survival(0.0) == pytest.approx(0.9991148076, abs=1e-8)

    def test_expected_negative_part_log_matches_ground_truth(self):
        got = _t4_posterior().expected_negative_part(scale="log")
        assert got == pytest.approx(9.070065e-06, rel=1e-4)

    def test_probability_between(self):
        post = _t4_posterior()
        expected = post.cdf(0.2) - post.cdf(0.1)
        assert post.probability_between(0.1, 0.2) == pytest.approx(expected, abs=1e-12)

    def test_probability_between_far_upper_tail_does_not_collapse_to_zero(self):
        """cdf(upper) - cdf(lower) rounds both terms to exactly 1.0 once
        the interval sits deep in the upper tail, silently reporting a
        real (tiny but nonzero) tail probability as exactly 0."""
        post = MixturePosterior(
            weights=np.array([1.0]), means=np.array([0.0]), sigmas=np.array([1.0])
        )
        import scipy.stats as st

        expected = st.norm.sf(9.0) - st.norm.sf(10.0)
        assert expected > 0.0  # sanity: a real, resolvable tail probability
        got = post.probability_between(9.0, 10.0)
        assert got == pytest.approx(expected, rel=1e-9)
        assert got > 0.0

    def test_single_component_matches_normal_posterior(self):
        # K=1 mixture must agree with the existing conjugate Normal path
        # to float precision - same arithmetic, different container.
        from increment.estimation.inference import Normal, normal_posterior

        prior = MixturePrior(weights=(1.0,), means=(0.0,), sigmas=(0.05,))
        mix = mixture_posterior(0.2, 0.05, prior)
        ref = normal_posterior(0.2, 0.05, prior=Normal(mu=0.0, sigma=0.05))
        assert mix.quantile(0.5) == pytest.approx(ref.mu, abs=1e-14)
        # The posterior sigma is sqrt(1/800) ~= 0.03536 (both prior and
        # obs sigma are 0.05), so xtol = sigma * 1e-9 ~= 3.5e-11 here -
        # brentq's own resolution, not float precision, bounds this.
        assert mix.quantile(0.975) == pytest.approx(ref.quantile(0.975), abs=1e-10)
        for op in ("cdf", "survival"):
            assert getattr(mix, op)(0.1) == pytest.approx(getattr(ref, op)(0.1), abs=1e-12)

    def test_extreme_observation_does_not_underflow(self):
        # Posterior weights are computed in log space: a 40-sigma surprise
        # must produce finite, normalized weights, not NaN.
        post = mixture_posterior(2.0, 0.05, StudentTPrior(nu=4.0, scale=0.05).components())
        assert np.isfinite(post.quantile(0.5))
        assert post.quantile(0.5) == pytest.approx(2.0, rel=0.05)  # data wins

    def test_narrow_symmetric_mixture_median_is_exactly_zero(self):
        # Brent's xtol used to be an absolute 1e-12 - eight orders of
        # magnitude coarser than this mixture's 1e-20 scale, so it accepted
        # convergence anywhere in a bracket that reads as flat at that
        # resolution. Scaling xtol to the mixture's own sigmas fixes it.
        mp = MixturePrior(weights=(0.5, 0.5), means=(1e-19, -1e-19), sigmas=(1e-20, 1e-20))
        post = mixture_posterior(0.0, 1.0, mp)
        assert post.cdf(0.0) == pytest.approx(0.5, abs=1e-15)
        assert post.quantile(0.5) == pytest.approx(0.0, abs=1e-25)  # was ~2.2e-19 before the fix

    def test_narrow_symmetric_mixture_isf_matches_quantile_at_same_tolerance(self):
        mp = MixturePrior(weights=(0.5, 0.5), means=(1e-19, -1e-19), sigmas=(1e-20, 1e-20))
        post = mixture_posterior(0.0, 1.0, mp)
        assert post.isf(0.5) == pytest.approx(0.0, abs=1e-25)
        assert post.isf(0.5) == pytest.approx(post.quantile(0.5), abs=1e-25)

    @pytest.mark.parametrize("method_name", ["quantile", "isf"])
    def test_inverse_probability_supports_subnormal_component_scale(self, method_name):
        post = MixturePosterior(
            weights=np.array([1.0]), means=np.array([0.0]), sigmas=np.array([2e-315])
        )
        assert getattr(post, method_name)(0.5) == 0.0

    @pytest.mark.parametrize("method_name", ["quantile", "isf"])
    def test_disparate_sigmas_resolve_within_the_narrow_component_not_the_wide_one(
        self, method_name
    ):
        """A 1e12-wide sigma gap between components: xtol must scale off
        ``sigmas.min()`` (the narrow component, 1e-12), not ``sigmas.max()``
        (the wide one, 1.0). The target probability sits inside the narrow
        component's own transition band, far from its median, so a coarse
        max-sigma xtol resolves to a point roughly 1e-11 off (~10 narrow
        sigmas) instead of within 1 ulp of the true root.

        Reference is closed-form and independent of MixturePosterior: the
        wide component's cdf/survival is exactly saturated (0.0 or 1.0 at
        float64 precision) throughout the narrow component's entire span,
        so the mixture value there reduces algebraically to the wide
        component's saturated contribution plus ``narrow_weight *
        norm.ppf/isf`` of the narrow component alone - never calling
        MixturePosterior.quantile/isf/cdf/survival to derive it.
        """
        from scipy.stats import norm as _norm_ref

        wide_weight, narrow_weight = 1.0 - 1e-9, 1e-9
        wide_mean, narrow_mean = 0.0, 100.0
        wide_sigma, narrow_sigma = 1.0, 1e-12
        post = MixturePosterior(
            weights=np.array([wide_weight, narrow_weight]),
            means=np.array([wide_mean, narrow_mean]),
            sigmas=np.array([wide_sigma, narrow_sigma]),
        )
        phi_target = 0.9  # a non-median point within the narrow component
        expected_x = narrow_mean + narrow_sigma * _norm_ref.ppf(phi_target)
        if method_name == "quantile":
            probability = 1.0 - narrow_weight * (1.0 - phi_target)
            got = post.quantile(probability)
        else:
            tail_probability = narrow_weight * (1.0 - phi_target)
            got = post.isf(tail_probability)
        assert got == pytest.approx(expected_x, abs=1e-13)


def _mixture_lift(alpha=0.05, alternative="two-sided"):
    # log_rr = 0.30 - 0.10 = 0.20, se_log_rr = sqrt(0.03^2 + 0.04^2) = 0.05
    return infer_lift(
        metric="m",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        log_rr=0.30 - 0.10,
        se_t=0.03,
        se_c=0.04,
        prior=StudentTPrior(nu=4.0, scale=0.05),
        alpha=alpha,
        alternative=alternative,
    )


class TestInferLiftMixture:
    def test_interval_is_mixture_quantiles(self):
        est = _mixture_lift()
        assert est.require_lift().value == pytest.approx(0.1538506996, abs=1e-6)
        assert est.require_lift().lb == pytest.approx(0.0484573187, abs=1e-6)
        assert est.require_lift().ub == pytest.approx(0.2812695873, abs=1e-6)

    def test_raw_sufficient_statistics_stay_raw(self):
        est = _mixture_lift()
        assert est.require_lift().log_mean == pytest.approx(0.20, abs=1e-12)
        assert est.require_lift().log_se == pytest.approx(0.05, abs=1e-12)

    def test_prior_spec_persists_the_declaration(self):
        # The spec is the three-field declaration, not the 80-component
        # expansion - stored artifacts must not bake in today's K or quadrature scheme.
        est = _mixture_lift()
        assert est.prior_spec == StudentTPrior(nu=4.0, scale=0.05)

    def test_raw_mixture_prior_is_persisted_verbatim(self):
        prior = MixturePrior(weights=(0.7, 0.3), means=(0.0, 0.0), sigmas=(0.01, 0.08))
        est = infer_lift(
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=0.30 - 0.10,
            se_t=0.03,
            se_c=0.04,
            prior=prior,
        )
        assert est.prior_spec == prior

    def test_normal_prior_does_not_stamp_prior_spec(self):
        from increment.estimation.inference import Normal

        est = infer_lift(
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=0.30 - 0.10,
            se_t=0.03,
            se_c=0.04,
            prior=Normal(mu=0.0, sigma=0.05),
        )
        assert est.prior_spec is None

    def test_flat_path_unchanged(self):
        # No prior: exact closed-form z-interval, exactly as before.
        est = infer_lift(
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=0.30 - 0.10,
            se_t=0.03,
            se_c=0.04,
        )
        assert est.require_lift().value == pytest.approx(0.2214027582, abs=1e-9)
        assert est.require_lift().lb == pytest.approx(0.1073854659, abs=1e-9)
        assert est.require_lift().ub == pytest.approx(0.3471593619, abs=1e-9)
        assert est.prior_spec is None

    def test_wide_scale_t_prior_degrades_to_the_flat_z_interval(self):
        # Wide scale flattens the prior onto N(log_rr, se), matching the z-interval:
        # SE is treated as known (unlike a t-test's estimate), so nu=n-1 is wrong here.
        est = infer_lift(
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=0.30 - 0.10,
            se_t=0.03,
            se_c=0.04,
            prior=StudentTPrior(nu=4.0, scale=50.0),
        )
        assert est.require_lift().value == pytest.approx(0.2214027582, abs=1e-5)
        assert est.require_lift().lb == pytest.approx(0.1073854659, abs=1e-5)
        assert est.require_lift().ub == pytest.approx(0.3471593619, abs=1e-5)

    def test_one_sided_uses_doubled_alpha_quantiles(self):
        est = _mixture_lift(alpha=0.05, alternative="greater")
        # alpha_eff = 0.10 -> interval at quantiles (0.05, 0.95), level 0.90
        assert est.require_lift().level == pytest.approx(0.90)
        assert est.require_lift().lb > 0.0484573187  # narrower than the 95% two-sided lb

    def test_sequential_inference_refuses_mixture_prior(self):
        from increment.errors import CapabilityError
        from increment.estimation.sequential import AlwaysValid

        with pytest.raises(CapabilityError) as raised:
            infer_lift(
                metric="m",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                log_rr=0.30 - 0.10,
                se_t=0.03,
                se_c=0.04,
                prior=StudentTPrior(nu=4.0, scale=0.05),
                inference_spec=AlwaysValid(registration=registration("gaussian")),
                n_comparison=1000,
            )

        assert raised.value.code == "sequential.route.unsupported"

    def test_cluster_t_reference_refuses_mixture_prior(self):
        with pytest.raises(InvalidRequestError) as raised:
            infer_lift(
                metric="m",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                log_rr=0.30 - 0.10,
                se_t=0.03,
                se_c=0.04,
                prior=StudentTPrior(nu=4.0, scale=0.05),
                dof=18.0,
                n_clusters=20,
            )
        assert raised.value.code == "estimation.inference.prior_excludes_cluster_robust_t"


class TestDecisionStatsUnderMixture:
    def test_chance_to_beat_matches_ground_truth(self):
        assert _mixture_lift().chance_to_beat() == pytest.approx(0.9991148076, abs=1e-6)

    def test_risk_if_shipped_matches_ground_truth(self):
        assert _mixture_lift().risk_if_shipped() == pytest.approx(9.070065e-06, rel=1e-3)

    def test_prob_within_and_p_value_do_not_raise(self):
        est = _mixture_lift()
        assert 0.0 <= est.prob_within(0.01) <= 1.0
        assert 0.0 <= est.p_value() <= 1.0


class TestMixtureScaleRefusals:
    def test_infer_ate_refuses(self):
        from increment.estimation.armstats import ScoreStats
        from increment.estimation.inference import infer_ate

        # Minimal valid ScoreStats - mirrors the construction infer_ate's happy-path test uses.
        scores = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=0.25)
        with pytest.raises(InvalidRequestError) as raised:
            infer_ate(
                metric="m",
                group_id="T",
                method="dml",
                method_role="decision",
                point=0.1,
                scores=scores,
                prior=StudentTPrior(nu=4.0, scale=0.05),
            )
        assert raised.value.code == "estimation.adjust.prior.type"


class TestFacadePassThrough:
    def test_estimate_lift_carries_mixture_to_decision_stats(self):
        from increment.estimation.armstats import ArmStats
        from increment.estimation.engine import estimate_lift
        from increment.semantics.models import MeanMetric

        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="C",
            n=100,
            sum_y=1000.0,
            sum_y2=10990.0,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="T",
            n=100,
            sum_y=1200.0,
            sum_y2=14990.0,
        )
        rows = []
        for a in (control, treatment):
            rows.append(
                {
                    "experiment_id": a.study_id,
                    "metric": a.metric,
                    "group_id": a.group_id,
                    "n": float(a.n),
                    "ref_y": a.ref_y,
                    "cy1": a.cy1,
                    "cy2": a.cy2,
                    "ref_x": a.ref_x,
                    "cx1": a.cx1,
                    "cx2": a.cx2,
                    "cxy": a.cxy,
                    "x_role": a.x_role,
                    "ref_den": a.ref_den,
                    "cden1": a.cden1,
                    "cden2": a.cden2,
                    "cyden": a.cyden,
                }
            )

        results = estimate_lift(
            metrics=[MeanMetric(name="rev", entity="user", fact="rev")],
            summary=rows,
            control_group="C",
            prior=StudentTPrior(nu=4.0, scale=0.05),
        ).results
        est = next(iter(results))
        assert est.prior_spec == StudentTPrior(nu=4.0, scale=0.05)
        est.chance_to_beat()  # decision stats work end to end


def _coverage(n_reps: int, seed: int) -> float:
    # When the prior is true (theta ~ t_4(0, 0.05)), the 95% credible interval
    # must cover theta ~95% of the time - exact Bayes calibration, not approximation.
    rng = np.random.default_rng(seed)
    prior = StudentTPrior(nu=4.0, scale=0.05)
    theta = 0.05 * rng.standard_t(df=4, size=n_reps)
    se_t, se_c = 0.03, 0.04
    log_t = theta + rng.normal(0.0, se_t, n_reps)
    log_c = rng.normal(0.0, se_c, n_reps)

    def _trial(i: int) -> bool:
        est = infer_lift(
            metric="m",
            group_id="T",
            method="unadjusted",
            method_role="decision",
            log_rr=float(log_t[i]) - float(log_c[i]),
            se_t=se_t,
            se_c=se_c,
            prior=prior,
        )
        lift = np.expm1(theta[i])
        return est.require_lift().lb <= lift <= est.require_lift().ub

    return replicate(n_reps, _trial).rate


def test_mixture_coverage_smoke():
    # 150 reps keeps this in the fast suite; the bound is a smoke width.
    assert 0.88 <= _coverage(150, seed=7) <= 1.0


@pytest.mark.slow
@pytest.mark.parameter_recovery
def test_mixture_coverage_full():
    assert 0.94 <= _coverage(20_000, seed=11) <= 0.96
