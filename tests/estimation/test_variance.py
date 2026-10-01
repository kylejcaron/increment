"""Step 2: Failing se_log_mean + VarianceModel + Registry tests."""

import math
import sys

import numpy as np
import pytest

from increment._moment_plan import X_SLOT_ROLES
from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.estimation.armstats import ArmStats, variance_slack
from increment.estimation.variance import (
    MeanVarianceModel,
    RatioVarianceModel,
    Registry,
    VarianceModel,
    cluster_outcome_moments,
    cluster_uptake_moments,
    ratio_abs_diff_se,
    ratio_moments,
    ratio_residual_slack,
    se_log_mean,
    stable_log_ratio,
)


class TestSeLogMean:
    def test_basic(self):
        """se_log_mean(var, mean, n) = sqrt(var / (n * mean**2))."""
        var, mean, n = 2.0, 5.0, 100
        expected = math.sqrt(var / (n * mean**2))
        assert se_log_mean(var, mean, n) == pytest.approx(expected, rel=1e-12)

    def test_zero_var(self):
        assert se_log_mean(0.0, 5.0, 100) == 0.0

    @pytest.mark.parametrize("mean,n", [(0.0, 100), (-1.0, 100)])
    def test_non_positive_mean(self, mean, n):
        """se_log_mean should raise for non-positive mean (log undefined)."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            se_log_mean(1.0, mean, n)

    @pytest.mark.parametrize("n", [0, -3])
    def test_non_positive_n(self, n):
        """se_log_mean should raise for non-positive n, not divide by zero."""
        with pytest.raises(InvalidRequestError) as exc_info:
            se_log_mean(1.0, 5.0, n)
        assert exc_info.value.code == "estimation.variance.positive_se_log"

    def test_large_mean_no_longer_overflows_to_a_wrong_zero(self):
        """``mean**2`` alone overflowed float64 for mean ~1e200 well before
        the true (representable, tiny) SE did, previously reporting a
        wrong se()==0.0 instead of raising or computing the real value."""
        se = se_log_mean(1e308, 1e200, 2)
        assert math.isfinite(se)
        assert se == pytest.approx(7.071067811865561e-47, rel=1e-9)

    def test_neighbouring_float_mean_stays_close(self):
        mean = math.nextafter(1e200, math.inf)
        se = se_log_mean(1e308, mean, 2)
        assert math.isfinite(se)
        assert se == pytest.approx(7.071067811865561e-47, rel=1e-6)

    def test_matches_direct_formula_at_ordinary_scale(self):
        """Ordinary scale: the log-domain evaluation is a no-op against
        the direct ``sqrt(var / (n * mean**2))`` formula."""
        var, mean, n = 12.5, 3.2, 400
        assert se_log_mean(var, mean, n) == pytest.approx(math.sqrt(var / (n * mean**2)), rel=1e-9)


class TestStableLogRatio:
    @staticmethod
    def _oracle(mean_c: float, mean_t: float) -> float:
        from decimal import Decimal, getcontext

        getcontext().prec = 60
        return float((Decimal(mean_t) / Decimal(mean_c)).ln())

    @pytest.mark.parametrize(
        "offset,naive_rel_err",
        [(1e6, 9e-11), (1e12, 1.8e-3), (1e15, 1.0)],
    )
    def test_correctly_rounded_where_log_subtraction_cancels(
        self, offset: float, naive_rel_err: float
    ):
        """Means differing by exactly 1 at a large offset: the helper lands
        within one ulp of a 60-digit oracle, while ``log(t) - log(c)`` loses
        the stated fraction of the effect (all of it at 1e15, where both
        logs round to the same double)."""
        mean_c, mean_t = offset, offset + 1.0
        want = self._oracle(mean_c, mean_t)

        got = stable_log_ratio(mean_c, mean_t)
        naive = math.log(mean_t) - math.log(mean_c)

        assert abs(got - want) <= math.ulp(want)
        assert abs(naive - want) / want >= naive_rel_err

    def test_ordinary_scale_agrees_with_log_subtraction(self):
        """Away from cancellation the two forms agree to rounding: the
        helper is a representation change, not a different estimand."""
        assert stable_log_ratio(4.10, 4.55) == pytest.approx(
            math.log(4.55) - math.log(4.10), rel=1e-15
        )

    def test_negative_effect(self):
        assert stable_log_ratio(4.0, 2.0) == pytest.approx(-math.log(2.0), rel=1e-15)

    @pytest.mark.parametrize("mean_c,mean_t", [(1.0, 1e-20), (1e-20, 1.0)])
    def test_strongly_asymmetric_positive_means(self, mean_c: float, mean_t: float):
        """The close-means form must not round a valid large decrease to
        ``log1p(-1)``; both contrast directions remain finite and accurate."""
        want = self._oracle(mean_c, mean_t)
        got = stable_log_ratio(mean_c, mean_t)

        assert math.isfinite(got)
        assert abs(got - want) <= math.ulp(want)

    @pytest.mark.parametrize(
        "mean_c,mean_t,offender",
        [(0.0, 1.0, 0.0), (-2.0, 1.0, -2.0), (1.0, 0.0, 0.0), (1.0, -3.0, -3.0), (0.0, 0.0, 0.0)],
    )
    def test_non_positive_mean_is_refused_by_name(
        self, mean_c: float, mean_t: float, offender: float
    ):
        """A zero control mean used to surface a bare ``ZeroDivisionError``
        and a non-positive treatment mean a bare ``math domain error``;
        both now carry ``se_log_mean``'s refusal code and name the value."""
        with pytest.raises(InvalidRequestError) as exc_info:
            stable_log_ratio(mean_c, mean_t)
        assert exc_info.value.code == "estimation.variance.mean_positive_log"
        assert exc_info.value.context["mean"] == offender


class TestMeanVarianceModel:
    def test_log_mean_se_matches_delta_method(self):
        """MeanVarianceModel.log_mean_se matches the textbook delta method."""
        n = 1000
        rng = np.random.default_rng(42)
        data = rng.exponential(scale=3.0, size=n)  # mean ~ 3
        sum_y = data.sum()
        sum_y2 = (data**2).sum()

        arm = ArmStats.from_raw_sums(
            study_id="exp1", metric="rev", group_id="A", n=n, sum_y=sum_y, sum_y2=sum_y2
        )
        s = arm.to_summary()

        model = MeanVarianceModel()
        log_mean, se = model.log_mean_se(arm)

        expected_log_mean = math.log(s.mean)
        expected_se = math.sqrt(s.var / (s.n * s.mean**2))

        assert log_mean == pytest.approx(expected_log_mean, rel=1e-12)
        assert se == pytest.approx(expected_se, rel=1e-12)

    def test_nonpositive_mean_refused_by_name(self):
        """A zero-conversion arm (mean 0) or a negative-mean metric (net
        revenue) used to die inside math.log with a context-free 'math
        domain error' - the designed refusal in se_log_mean was dead code
        because log(mean) was evaluated first. The refusal must name the
        metric and arm."""
        zero = ArmStats.from_raw_sums(
            study_id="exp1", metric="conv", group_id="T", n=100, sum_y=0.0, sum_y2=0.0
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            MeanVarianceModel().log_mean_se(zero)
        assert exc_info.value.code == "estimation.variance.metric_group_log"
        negative = ArmStats.from_raw_sums(
            study_id="exp1", metric="net_rev", group_id="T", n=100, sum_y=-50.0, sum_y2=100.0
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            MeanVarianceModel().log_mean_se(negative)
        assert exc_info.value.code == "estimation.variance.metric_group_log"

    def test_consumes(self):
        assert MeanVarianceModel.consumes == "moments"

    def test_huge_reference_no_longer_overflowerrors(self):
        """``ArmStats(n=2, ref_y=1e200, cy2=1e308)``: var_y() is a huge
        but representable 1e308, and mean=1e200 -- ``mean**2`` alone
        used to overflow inside ``se_log_mean``, raising ``OverflowError``
        for a true SE that is itself representable (~7.07e-47)."""
        arm = ArmStats(study_id="s", metric="m", group_id="g", n=2, ref_y=1e200, cy1=0.0, cy2=1e308)
        log_mean, se = MeanVarianceModel().log_mean_se(arm)
        assert math.isfinite(log_mean) and math.isfinite(se)
        assert se == pytest.approx(7.071067811865561e-47, rel=1e-6)


class TestRatioVarianceModel:
    def test_log_mean_se_uses_neg2cov_form(self):
        """RatioVarianceModel uses the -2*Cov form of the delta method."""
        n = 500
        rng = np.random.default_rng(77)
        # Generate correlated numerator and denominator data
        num = rng.exponential(scale=2.0, size=n)
        denom = rng.exponential(scale=5.0, size=n) + 0.3 * num  # correlated with num

        sum_y = num.sum()
        sum_y2 = (num**2).sum()
        sum_den = denom.sum()
        sum_den2 = (denom**2).sum()
        sum_yden = (num * denom).sum()

        arm = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_metric",
            group_id="A",
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_den=sum_den,
            sum_den2=sum_den2,
            sum_yden=sum_yden,
        )

        model = RatioVarianceModel()
        log_mean, se = model.log_mean_se(arm)

        # Manual delta-method calculation
        n_bar = sum_y / n
        d_bar = sum_den / n
        var_n = (sum_y2 - sum_y**2 / n) / (n - 1)
        var_d = (sum_den2 - sum_den**2 / n) / (n - 1)
        cov_nd = (sum_yden - sum_y * sum_den / n) / (n - 1)

        expected_log_mean = math.log(n_bar / d_bar)
        expected_var_log_r = (1 / n) * (
            var_n / n_bar**2 + var_d / d_bar**2 - 2 * cov_nd / (n_bar * d_bar)
        )
        expected_se = math.sqrt(expected_var_log_r)

        assert log_mean == pytest.approx(expected_log_mean, rel=1e-12)
        assert se == pytest.approx(expected_se, rel=1e-9)

    def test_consumes(self):
        assert RatioVarianceModel.consumes == "moments"

    def test_refuses_infeasible_moments_instead_of_clamping_to_zero(self):
        """The delta-method variance formula is Var(N/nbar - D/dbar) and
        cannot be legitimately negative (Cauchy-Schwarz bounds cov_nd by
        sqrt(var_n*var_d)); a covariance that badly violates that bound
        signals corrupted or mis-aggregated moments and used to be
        unconditionally clamped to a plausible-looking se()==0.0."""
        arm = ArmStats(
            study_id="s",
            metric="m",
            group_id="t",
            n=100,
            ref_y=10.0,
            cy1=0.0,
            cy2=100.0,
            ref_den=5.0,
            cden1=0.0,
            cden2=100.0,
            cyden=10000.0,
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            RatioVarianceModel().log_mean_se(arm)
        assert (
            exc_info.value.code
            == "estimation.variance.ratio_variance.variance_scale_not_representable"
        )

    def test_ratio_moments_raises_for_single_unit_arm(self):
        """ddof=1 ratio moments need n >= 2 - a single-unit arm must raise,
        not divide by zero."""
        arm = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_metric",
            group_id="A",
            n=1,
            sum_y=2.0,
            sum_y2=4.0,
            sum_den=3.0,
            sum_den2=9.0,
            sum_yden=6.0,
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            ratio_moments(arm)
        assert exc_info.value.code == "estimation.variance.ratio_moments_least_two_units"

    def _arm(self, *, sum_y: float, sum_den: float) -> ArmStats:
        return ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rpo",
            group_id="T",
            n=10,
            sum_y=sum_y,
            sum_y2=abs(sum_y) * 2.0 + 1.0,
            sum_den=sum_den,
            sum_den2=abs(sum_den) * 2.0 + 1.0,
            sum_yden=0.0,
        )

    def test_zero_denominator_refused_by_name(self):
        """Sum(den)=0 used to surface as a bare ZeroDivisionError with no
        metric, arm, or guidance; it must be a typed refusal naming both."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ratio_moments(self._arm(sum_y=20.0, sum_den=0.0))
        assert (
            exc_info.value.code == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
        )

    def test_negative_denominator_refused_by_name(self):
        """Sum(den)<0 used to die in math.log on the relative path while the
        absolute path silently returned a signed 'estimate' - both paths
        share ratio_moments, so one named refusal covers both."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ratio_moments(self._arm(sum_y=20.0, sum_den=-15.0))
        assert (
            exc_info.value.code == "estimation.variance.ratio_moments_nonpositive_denominator_mean"
        )

    def test_ratio_moments_accepts_a_nonpositive_numerator(self):
        """ratio_moments feeds BOTH the clustered/absolute callers (which
        never take a log) and the log-scale RatioVarianceModel -- the
        positivity requirement belongs only to the latter."""
        n_bar, d_bar, var_n, var_d, cov_nd = ratio_moments(self._arm(sum_y=-2.0, sum_den=15.0))
        assert n_bar == pytest.approx(-0.2)
        assert d_bar == pytest.approx(1.5)

    def test_log_mean_se_refuses_a_nonpositive_numerator_by_name(self):
        """The log IS taken here (RatioVarianceModel.log_mean_se), so the
        positivity guard belongs on this call, not on ratio_moments."""
        with pytest.raises(InvalidRequestError) as exc_info:
            RatioVarianceModel().log_mean_se(self._arm(sum_y=-2.0, sum_den=15.0))
        assert exc_info.value.code == "estimation.variance.metric_group_log"


class TestRegistry:
    def test_register_and_get(self):
        reg: Registry[VarianceModel] = Registry("variance model")
        m = MeanVarianceModel()
        reg.register("mean", m)
        assert reg.get("mean") is m

    def test_unregistered_raises_not_implemented(self):
        reg: Registry[VarianceModel] = Registry("variance model")
        reg.register("mean", MeanVarianceModel())

        with pytest.raises(UnsupportedRequestError) as exc_info:
            reg.get("ratio")

        assert exc_info.value.code == "estimation.variance.registry.no_registered_available"
        assert exc_info.value.context["key"] == "ratio"
        available = exc_info.value.context["available"]
        assert isinstance(available, tuple) and "mean" in available

    def test_contains(self):
        reg: Registry[VarianceModel] = Registry("variance model")
        reg.register("mean", MeanVarianceModel())
        assert "mean" in reg
        assert "ratio" not in reg


class TestRatioAbsDiffSe:
    """Absolute-scale ratio SE, stable ratio-residual delta-method form."""

    def test_matches_the_exact_value_the_direct_form_gets_wrong(self):
        """The direct three-term expansion returns 1.4901e-9 here (5.4%
        high) because each term independently loses precision at its
        own den_bar**2/**3/**4 scale before the cancellation; the
        residual form cancels once, before any division."""
        r, se = ratio_abs_diff_se(1e8, 1e8, 1.0000000000000002e16, 1e16, 1e16, 100)
        assert r == pytest.approx(1.0)
        assert se == pytest.approx(1.4142135623730951e-09, rel=1e-9)

    def test_one_ulp_further_is_a_legitimate_exact_zero(self):
        """One ulp closer to exact cancellation: the residual really is
        (up to float64 rounding) zero here, not a refusal-worthy deficit."""
        r, se = ratio_abs_diff_se(1e8, 1e8, 1e16, 1e16, 1e16, 100)
        assert r == pytest.approx(1.0)
        assert se == 0.0

    def test_tiny_denominator_refuses_instead_of_dividing_by_zero(self):
        """The direct form divides by ``den_bar**3``, which underflows to
        exactly 0.0 at den_bar=1e-160 and raises ``ZeroDivisionError``
        with no context. The residual form never forms that term, but
        the ratio itself overflows float64 at this scale and must be
        refused, not silently returned as ``inf``."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ratio_abs_diff_se(1e8, 1e-160, 1e16, 1e16, 1e16, 100)
        assert exc_info.value.code == "estimation.variance.ratio_variance_term"

    def test_impossible_covariance_is_refused_not_zeroed(self):
        """num variance and den variance both 2, covariance 4: Cauchy-
        Schwarz caps the covariance at sqrt(2*2)=2, so 4 is impossible
        for any real sample. The naive clamp used to report se()==0.0;
        this must refuse instead."""
        with pytest.raises(InvalidRequestError) as exc_info:
            ratio_abs_diff_se(10.0, 10.0, 2.0, 2.0, 4.0, 100)
        assert exc_info.value.code == "estimation.variance.ratio_residual_variance_negative"

    def test_matches_the_direct_expansion_at_ordinary_scale(self):
        """Ordinary scale: the residual form is a no-op against the
        direct three-term expansion."""
        num_bar, den_bar, var_num, var_den, cov_num_den, n = 50.0, 20.0, 4.0, 1.5, 0.8, 200
        r, se = ratio_abs_diff_se(num_bar, den_bar, var_num, var_den, cov_num_den, n)
        expected_r = num_bar / den_bar
        expected_var = (1.0 / n) * (
            var_num / den_bar**2
            - 2.0 * num_bar * cov_num_den / den_bar**3
            + num_bar**2 * var_den / den_bar**4
        )
        assert r == pytest.approx(expected_r, rel=1e-12)
        assert se == pytest.approx(math.sqrt(expected_var), rel=1e-9)

    def test_reordering_the_two_arms_negates_the_ratio_but_not_the_se(self):
        """Swapping which side is 'numerator' vs 'denominator' is a
        distinct estimand, not a reordering in the aggregation sense --
        pinned here only to confirm the SE stays finite and positive
        under the same relative-scale inputs from either side."""
        r1, se1 = ratio_abs_diff_se(30.0, 10.0, 4.0, 1.0, 0.5, 50)
        r2, se2 = ratio_abs_diff_se(10.0, 30.0, 1.0, 4.0, 0.5, 50)
        assert r1 == pytest.approx(3.0)
        assert r2 == pytest.approx(1.0 / 3.0)
        assert se1 > 0.0 and se2 > 0.0


def test_ratio_residual_clamps_a_perfectly_proportional_cell() -> None:
    """A single contributing unit makes the residual exactly zero in real
    arithmetic; float64 lands it just past the plain accumulation slack because
    `r` is itself a quotient whose rounding the `r**2` term amplifies. That is
    rounding, not an infeasible covariance, so it must clamp rather than refuse.
    """
    # 86 units, exactly one order of 46.85 -- measured from the demo warehouse.
    r, se = ratio_abs_diff_se(
        0.5447674418604653,
        0.011627906976744179,
        25.52235465116262,
        0.01162790697674419,
        0.5447674418604668,
        86,
    )
    assert r == pytest.approx(46.85)
    assert se == 0.0


def test_ratio_residual_still_refuses_an_infeasible_covariance() -> None:
    with pytest.raises(InvalidRequestError) as exc_info:
        # A covariance far above the Cauchy-Schwarz bound: the deficit is orders
        # of magnitude beyond any rounding of these terms.
        ratio_abs_diff_se(1.0, 1.0, 1.0, 1.0, 50.0, 100)
    assert exc_info.value.code == "estimation.variance.ratio_residual_variance_negative"


def test_ratio_residual_refuses_an_overflowed_quadratic_form() -> None:
    """A NaN residual must refuse, not clamp.

    `NaN` fails every ordering comparison, so an unguarded clamp would treat it
    as within tolerance and report a zero standard error -- maximal confidence
    from a computation that overflowed.
    """
    with pytest.raises(InvalidRequestError) as exc_info:
        ratio_abs_diff_se(1e200, 1e-200, 1e308, 1e308, 1e308, 100)
    assert exc_info.value.code == "estimation.variance.ratio_variance_term"


def test_clamp_primitive_refuses_non_finite_inputs() -> None:
    from increment.estimation.armstats import clamp_negative_variance

    assert clamp_negative_variance(float("nan"), magnitude=1.0, n=10) is None
    assert clamp_negative_variance(float("-inf"), magnitude=1.0, n=10) is None
    assert clamp_negative_variance(-1e-18, magnitude=float("nan"), n=10) is None


class TestRepresentabilityAtTheExponentRange:
    """Near the float64 exponent range these must compute or refuse, never
    return a zero standard error or raise a bare range error."""

    def test_huge_denominator_computes_the_representable_se(self):
        # Squaring den_bar overflows; dividing by |den_bar| after the square
        # root does not, and the true SE is representable.
        r, se = ratio_abs_diff_se(1.0, 1e200, 1.0, 0.0, 0.0, 2)
        assert r == pytest.approx(1e-200)
        assert se == pytest.approx(7.071067811865476e-201)

    def test_tiny_denominator_computes_when_representable(self):
        r, se = ratio_abs_diff_se(1.0, 1e-170, 1.0, 0.0, 0.0, 2)
        assert r == pytest.approx(1e170)
        assert math.isfinite(se) and se > 0.0

    def test_unrepresentable_variance_term_refuses_with_its_scale(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            ratio_abs_diff_se(1e8, 1e-160, 1e16, 1e16, 1e16, 100)
        assert exc_info.value.code == "estimation.variance.ratio_variance_term"

    def test_se_log_mean_refuses_an_underflowing_se_instead_of_returning_zero(self):
        # A positive variance whose SE rounds to 0.0 under exp would read as
        # maximal confidence; it must refuse.
        with pytest.raises(InvalidRequestError) as exc_info:
            se_log_mean(1e-300, 1e308, 2)
        assert exc_info.value.code == "estimation.variance.se_log_mean_not_representable"

    def test_se_log_mean_refuses_an_overflowing_se(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            se_log_mean(1e300, 1e-300, 2)
        assert exc_info.value.code == "estimation.variance.se_log_mean_out_of_range"

    def test_se_log_mean_unchanged_at_ordinary_scale(self):
        assert se_log_mean(4.0, 10.0, 100) == pytest.approx(0.02)


class TestLogDomainSeUsesTheRealExponentLimit:
    """A guessed log bound tight enough to be safe also refuses representable
    answers, so the result is exponentiated and judged instead."""

    def test_a_log_se_past_the_old_constant_still_computes(self) -> None:
        # half ~ 709.20: exp is finite here, but a 709.0 cutoff refused it.
        assert se_log_mean(1e308, 1e-154, 1.0) == pytest.approx(1e308, rel=1e-12)

    def test_a_log_se_past_the_exponent_range_refuses(self) -> None:
        with pytest.raises(InvalidRequestError) as exc_info:
            se_log_mean(1e308, 1e-155, 1.0)
        assert exc_info.value.code == "estimation.variance.se_log_mean_out_of_range"

    def test_a_standard_error_that_underflows_to_zero_refuses(self) -> None:
        # A zero SE from a positive variance reads as maximal confidence.
        with pytest.raises(InvalidRequestError) as exc_info:
            se_log_mean(1e-308, 1e171, 1.0)
        assert exc_info.value.code == "estimation.variance.se_log_mean_not_representable"

    def test_an_ordinary_scale_is_unchanged(self) -> None:
        assert se_log_mean(4.0, 10.0, 100) == pytest.approx(0.02)


class TestRatioSeSurvivesASubnormalResidual:
    """``clamped / n`` underflowed to exactly zero for a subnormal residual,
    returning a zero standard error from a positive variance."""

    def test_a_subnormal_residual_computes_its_representable_se(self) -> None:
        _r, se = ratio_abs_diff_se(1e-200, 1e-200, 5e-324, 0.0, 0.0, 2)
        assert se == pytest.approx(math.sqrt(5e-324) / math.sqrt(2) / 1e-200, rel=1e-9)
        assert se > 0.0


class TestLogRatioTermsSurviveATinyMean:
    """Each delta-method term divides once by a mean rather than by its square:
    squaring underflows to exactly zero well before the term itself leaves
    float64, which raised a bare ZeroDivisionError from a public estimator."""

    @staticmethod
    def _arm(n_bar: float) -> ArmStats:
        n = 200
        sum_y = n_bar * n
        sum_den = 10.0 * n
        return ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y**2 / n * 1.0001,
            sum_den=sum_den,
            sum_den2=sum_den**2 / n * 1.01,
            sum_yden=sum_y * sum_den / n,
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize("n_bar", [1e-160, 1e-170, 1e-180])
    def test_a_tiny_numerator_mean_computes_its_log_ratio_se(self, n_bar: float) -> None:
        _log_mean, se = RatioVarianceModel().log_mean_se(self._arm(n_bar))
        assert se > 0.0
        # A log-ratio SE is scale-invariant in the numerator mean, so every
        # scale must agree rather than one of them collapsing.
        assert se == pytest.approx(0.007088812050083444, rel=1e-9)


class TestRatioResidualSlackIsAnErrorBound:
    """A perfectly-constant ratio has an exact residual of zero, so every
    computed bit is rounding. Its magnitude depends on the order the warehouse
    summed the moments, so the tolerance must be a derived bound, not a value
    fitted to one observed run."""

    def test_the_budget_covers_the_derived_operation_count(self) -> None:
        magnitude, n = 102.089, 86
        beyond = ratio_residual_slack(magnitude, n) - variance_slack(magnitude, n)
        # Nine half-ulps of quotient propagation (see the derivation); carried
        # as 8 eps so the bound is not sitting exactly on its own estimate.
        assert beyond >= 4.5 * sys.float_info.epsilon * magnitude
        assert beyond == pytest.approx(8.0 * sys.float_info.epsilon * magnitude)

    def test_the_budget_stays_negligible_against_a_real_violation(self) -> None:
        # A genuine Cauchy-Schwarz violation is an O(1) relative deficit; this
        # tolerance must never approach it.
        magnitude = 102.089
        assert ratio_residual_slack(magnitude, 86) / magnitude < 1e-14

    def test_a_perfectly_constant_ratio_is_accepted_at_either_rounding_sign(self) -> None:
        # num == den exactly, so R is constant and the true variance is zero.
        for var in (1e-13, -1e-13):
            r, se = ratio_abs_diff_se(10.0, 10.0, 100.0 + var, 100.0, 100.0, 86)
            assert r == pytest.approx(1.0)
            assert se >= 0.0


class TestLogRatioSeAcrossTheWholeExponentRange:
    """The scale is carried as a log and only the ratio is exponentiated, so a
    representable standard error must compute however extreme the mean is."""

    @staticmethod
    def _arm(n: int, n_bar: float, var_n: float) -> ArmStats:
        sum_y = n_bar * n
        sum_den = 10.0 * n
        return ArmStats.from_raw_sums(
            study_id="e",
            metric="m",
            group_id="g",
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y**2 / n + var_n * (n - 1),
            sum_den=sum_den,
            sum_den2=sum_den**2 / n,
            sum_yden=n_bar * sum_den,
        )

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.parametrize(
        ("n", "n_bar", "var_n"),
        [(2, 1e-308, 4.0), (200, 1e-200, 1.0), (200, 1e-180, 1.0), (200, 1.0, 1.0)],
    )
    def test_it_matches_the_analytic_value_at_every_scale(self, n, n_bar, var_n):
        # Forming sd/mean directly overflows near 1e-308 and squaring it
        # overflows near 1e-200, both while this value is finite.
        expected = math.exp(0.5 * math.log(var_n / n) - math.log(n_bar))
        _log_mean, se = RatioVarianceModel().log_mean_se(self._arm(n, n_bar, var_n))
        assert se == pytest.approx(expected, rel=1e-9)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_an_infeasible_covariance_is_refused_not_clamped_to_the_bound(self):
        """Clamping the implied correlation to 1 would repair corrupt moments
        into a plausible-looking variance."""
        arm = ArmStats(
            study_id="s",
            metric="m",
            group_id="t",
            n=100,
            ref_y=10.0,
            cy1=0.0,
            cy2=100.0,
            ref_den=5.0,
            cden1=0.0,
            cden2=100.0,
            cyden=10000.0,
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            RatioVarianceModel().log_mean_se(arm)
        assert (
            exc_info.value.code
            == "estimation.variance.ratio_variance.variance_scale_not_representable"
        )


def test_ratio_and_cluster_outcome_missing_denominator_share_canonical_code() -> None:
    arm = ArmStats(study_id="s", metric="m", group_id="g", n=2, ref_y=1.0, cy1=0.0, cy2=1.0)

    with pytest.raises(InvalidRequestError) as via_ratio:
        ratio_moments(arm)
    with pytest.raises(InvalidRequestError) as via_cluster_outcome:
        cluster_outcome_moments(arm)

    assert (
        via_ratio.value.code
        == via_cluster_outcome.value.code
        == ("estimation.variance.ratio_moments_needs_ref_den")
    )


def test_cluster_uptake_and_outcome_nonpositive_size_share_canonical_code() -> None:
    arm = ArmStats(
        study_id="s",
        metric="m",
        group_id="g",
        n=2,
        ref_y=1.0,
        cy1=0.0,
        cy2=1.0,
        ref_x=1.0,
        cx1=0.0,
        cx2=1.0,
        cxy=0.0,
        x_role=X_SLOT_ROLES["uptake"],
        ref_den=0.0,
        cden1=0.0,
        cden2=0.0,
        cyden=0.0,
        cxden=0.0,
    )

    with pytest.raises(InvalidRequestError) as via_uptake:
        cluster_uptake_moments(arm)
    with pytest.raises(InvalidRequestError) as via_cluster_outcome:
        cluster_outcome_moments(arm)

    assert (
        via_uptake.value.code
        == via_cluster_outcome.value.code
        == ("estimation.variance.cluster_outcome_moments_nonpositive_size")
    )
