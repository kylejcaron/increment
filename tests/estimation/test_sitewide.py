"""Sitewide impact: pure math on ArmStats, no ibis/DuckDB needed."""

import math
import warnings

import pytest
from scipy.stats import norm as _norm
from scipy.stats import t as _t

from increment.errors import IncrementRuntimeWarning, IncrementWarning, InvalidRequestError
from increment.estimation.armstats import ArmStats, welch_satterthwaite_df
from increment.estimation.sitewide import (
    _CLUSTER_BASELINE_CAVEAT,
    SitewideContrast,
    SitewideRatioContrast,
    _Arm,
    sitewide_impact,
    sitewide_impact_ratio,
)
from tests.warning_codes import warning_codes

_INVALID_ALPHA = [
    float("-inf"),
    -0.1,
    0.0,
    1.0,
    1.5,
    float("inf"),
    float("nan"),
]


def _assert_invalid_alpha_error(error: InvalidRequestError, alpha: float) -> None:
    assert error.code == "estimation.sitewide.alpha"
    assert error.context == {"alpha": alpha}


@pytest.mark.parametrize("alpha", _INVALID_ALPHA)
def test_sitewide_impact_rejects_invalid_alpha(alpha):
    control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
    treatment = _arm(n=120, mean=12.0, var=9.0, group_id="T")
    contrast = SitewideContrast.from_iid(control, treatment)

    with pytest.raises(InvalidRequestError) as exc_info:
        sitewide_impact(contrast, site_total_volume=50_000.0, alpha=alpha)

    _assert_invalid_alpha_error(exc_info.value, alpha)


@pytest.mark.parametrize("alpha", _INVALID_ALPHA)
def test_sitewide_impact_ratio_rejects_invalid_alpha(alpha):
    control = _ratio_arm(
        n=100,
        mean_num=10.0,
        var_num=4.0,
        mean_den=5.0,
        var_den=1.0,
        cov_yden=0.5,
        group_id="C",
    )
    treatment = _ratio_arm(
        n=120,
        mean_num=12.0,
        var_num=9.0,
        mean_den=6.0,
        var_den=2.0,
        cov_yden=0.8,
        group_id="T",
    )
    contrast = SitewideRatioContrast.from_iid(control, treatment)

    with pytest.raises(InvalidRequestError) as exc_info:
        sitewide_impact_ratio(
            contrast,
            site_total_numerator=50_000.0,
            site_total_denominator=20_000.0,
            alpha=alpha,
        )

    _assert_invalid_alpha_error(exc_info.value, alpha)


_NONFINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("bad", _NONFINITE)
def test_sitewide_impact_rejects_nonfinite_site_total_volume(bad):
    """Only baseline_volume (a derived quantity) was checked with
    '<= 0', which NaN silently passes (NaN <= 0 is False) and +inf passes
    outright, producing a nonsense zero relative_impact instead of a
    refusal."""
    control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
    treatment = _arm(n=120, mean=12.0, var=9.0, group_id="T")
    contrast = SitewideContrast.from_iid(control, treatment)

    with pytest.raises(InvalidRequestError) as exc_info:
        sitewide_impact(contrast, site_total_volume=bad)
    assert exc_info.value.code == "estimation.armstats.arm_stats.cross_field_finite"


@pytest.mark.parametrize("bad", _NONFINITE)
def test_sitewide_impact_ratio_rejects_nonfinite_site_total_numerator(bad):
    control = _ratio_arm(
        n=100, mean_num=10.0, var_num=4.0, mean_den=5.0, var_den=1.0, cov_yden=0.5, group_id="C"
    )
    treatment = _ratio_arm(
        n=120, mean_num=12.0, var_num=9.0, mean_den=6.0, var_den=2.0, cov_yden=0.8, group_id="T"
    )
    contrast = SitewideRatioContrast.from_iid(control, treatment)

    with pytest.raises(InvalidRequestError) as exc_info:
        sitewide_impact_ratio(contrast, site_total_numerator=bad, site_total_denominator=20_000.0)
    assert exc_info.value.code == "estimation.armstats.arm_stats.cross_field_finite"


@pytest.mark.parametrize("bad", _NONFINITE)
def test_sitewide_impact_ratio_rejects_nonfinite_site_total_denominator(bad):
    control = _ratio_arm(
        n=100, mean_num=10.0, var_num=4.0, mean_den=5.0, var_den=1.0, cov_yden=0.5, group_id="C"
    )
    treatment = _ratio_arm(
        n=120, mean_num=12.0, var_num=9.0, mean_den=6.0, var_den=2.0, cov_yden=0.8, group_id="T"
    )
    contrast = SitewideRatioContrast.from_iid(control, treatment)

    with pytest.raises(InvalidRequestError) as exc_info:
        sitewide_impact_ratio(contrast, site_total_numerator=50_000.0, site_total_denominator=bad)
    assert exc_info.value.code == "estimation.armstats.arm_stats.cross_field_finite"


def _arm(*, n: int, mean: float, var: float, group_id: str, metric: str = "rev") -> ArmStats:
    """Build an ArmStats with an exact ddof=1 mean/var via from_raw_sums."""
    sum_y = mean * n
    sum_y2 = var * (n - 1) + sum_y**2 / n
    return ArmStats.from_raw_sums(
        study_id="exp1", metric=metric, group_id=group_id, n=n, sum_y=sum_y, sum_y2=sum_y2
    )


def _ratio_arm(
    *,
    n: int,
    mean_num: float,
    var_num: float,
    mean_den: float,
    var_den: float,
    cov_yden: float,
    group_id: str,
) -> ArmStats:
    """Build a ratio-family ArmStats with exact ddof=1 moments via from_raw_sums."""
    sum_y = mean_num * n
    sum_y2 = var_num * (n - 1) + sum_y**2 / n
    sum_den = mean_den * n
    sum_den2 = var_den * (n - 1) + sum_den**2 / n
    sum_yden = cov_yden * (n - 1) + sum_y * sum_den / n
    return ArmStats.from_raw_sums(
        study_id="exp1",
        metric="rpu",
        group_id=group_id,
        n=n,
        sum_y=sum_y,
        sum_y2=sum_y2,
        sum_den=sum_den,
        sum_den2=sum_den2,
        sum_yden=sum_yden,
    )


def _cluster_arm(
    *,
    n: int,
    g_mean: float,
    g_var: float,
    m_mean: float,
    m_var: float,
    cov_gm: float,
    group_id: str,
    metric: str = "rev",
) -> ArmStats:
    """Build a clustered SUM-metric ArmStats: per-cluster outcome total
    ``g_j`` (mean ``g_mean``, var ``g_var``) in the y family, per-cluster
    SIZE ``m_j`` (mean ``m_mean``, var ``m_var``, cov with g ``cov_gm``)
    in BOTH the x and den family - the clustered wire contract for a
    non-ratio, non-uptake clustered row (``increment.query.builders``'
    clustered ``group_summary``). References are chosen exactly equal to
    the means, so every residual first moment is exactly zero and the
    centered second moments are exact ddof=1 quantities."""
    return ArmStats(
        study_id="exp1",
        metric=metric,
        group_id=group_id,
        n=n,
        ref_y=g_mean,
        cy1=0.0,
        cy2=g_var * (n - 1),
        ref_x=m_mean,
        x_role="cluster_size",
        cx1=0.0,
        cx2=m_var * (n - 1),
        cxy=cov_gm * (n - 1),
        ref_den=m_mean,
        cden1=0.0,
        cden2=m_var * (n - 1),
        cyden=cov_gm * (n - 1),
    )


def _cluster_ratio_arm(
    *,
    n: int,
    num_mean: float,
    num_var: float,
    den_mean: float,
    den_var: float,
    m_mean: float,
    m_var: float,
    cov_num_den: float,
    cov_num_m: float,
    cov_den_m: float,
    group_id: str,
    metric: str = "rpu",
) -> ArmStats:
    """Build a clustered RATIO-metric ArmStats: per-cluster numerator total
    ``num_j`` in the y family, the metric's OWN per-cluster denominator
    total ``den_j`` in the den family, and per-cluster SIZE ``m_j`` in the
    x family - the clustered wire contract for a declared ratio metric under a
    cluster. Exact ddof=1 moments the same way :func:`_cluster_arm` is."""
    return ArmStats(
        study_id="exp1",
        metric=metric,
        group_id=group_id,
        n=n,
        ref_y=num_mean,
        cy1=0.0,
        cy2=num_var * (n - 1),
        ref_x=m_mean,
        cx1=0.0,
        x_role="cluster_size",
        cx2=m_var * (n - 1),
        cxy=cov_num_m * (n - 1),
        ref_den=den_mean,
        cden1=0.0,
        cden2=den_var * (n - 1),
        cyden=cov_num_den * (n - 1),
        cxden=cov_den_m * (n - 1),
    )


class TestSitewideImpact:
    def test_hand_computed_case(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
        treatment = _arm(n=120, mean=12.0, var=9.0, group_id="T")
        contrast = SitewideContrast.from_iid(control, treatment)
        site_total_volume = 50_000.0

        result = sitewide_impact(contrast, site_total_volume=site_total_volume)

        delta = 2.0
        delta_se = math.sqrt(9.0 / 120 + 4.0 / 100)
        n_exp = 220
        baseline = site_total_volume - delta * 120
        absolute = delta * n_exp
        relative = absolute / baseline
        z = float(_norm.ppf(0.975))
        rel_se = abs(n_exp * site_total_volume / baseline**2) * delta_se

        assert result.delta == pytest.approx(delta)
        assert result.delta_se == pytest.approx(delta_se)
        assert result.n_control == pytest.approx(100.0)
        assert result.n_treatment == pytest.approx(120.0)
        assert result.n_enrolled == pytest.approx(220.0)
        assert result.baseline_volume == pytest.approx(baseline)
        assert result.absolute_impact == pytest.approx(absolute)
        assert result.absolute_impact_se == pytest.approx(delta_se * n_exp)
        assert result.absolute_impact_lb == pytest.approx(absolute - z * delta_se * n_exp)
        assert result.absolute_impact_ub == pytest.approx(absolute + z * delta_se * n_exp)
        assert result.relative_impact == pytest.approx(relative)
        assert result.relative_impact_se == pytest.approx(rel_se)
        assert result.relative_impact_lb == pytest.approx(relative - z * rel_se)
        assert result.relative_impact_ub == pytest.approx(relative + z * rel_se)
        assert result.n_clusters is None
        assert result.absolute_dof is None
        assert result.relative_dof is None

    def test_zero_lift_zero_impact(self):
        control = _arm(n=50, mean=5.0, var=1.0, group_id="C")
        treatment = _arm(n=50, mean=5.0, var=1.0, group_id="T")
        contrast = SitewideContrast.from_iid(control, treatment)

        result = sitewide_impact(contrast, site_total_volume=10_000.0)

        assert result.delta == 0.0
        assert result.absolute_impact == 0.0
        assert result.relative_impact == 0.0

    def test_ci_width_grows_with_lift_se(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
        tight = _arm(n=100, mean=12.0, var=1.0, group_id="T")
        wide = _arm(n=100, mean=12.0, var=100.0, group_id="T")

        r_tight = sitewide_impact(
            SitewideContrast.from_iid(control, tight), site_total_volume=50_000.0
        )
        r_wide = sitewide_impact(
            SitewideContrast.from_iid(control, wide), site_total_volume=50_000.0
        )

        assert r_wide.delta_se > r_tight.delta_se
        abs_width_tight = r_tight.absolute_impact_ub - r_tight.absolute_impact_lb
        abs_width_wide = r_wide.absolute_impact_ub - r_wide.absolute_impact_lb
        rel_width_tight = r_tight.relative_impact_ub - r_tight.relative_impact_lb
        rel_width_wide = r_wide.relative_impact_ub - r_wide.relative_impact_lb

        assert abs_width_wide > abs_width_tight
        assert rel_width_wide > rel_width_tight

    def test_full_coverage_reduces_to_plain_relative_lift(self):
        # If enrolled population IS the entire site, V0 = V - delta*n_T
        # collapses to mean_C*(n_C+n_T), so relative impact = delta / mean_C.
        control = _arm(n=200, mean=8.0, var=4.0, group_id="C")
        treatment = _arm(n=200, mean=10.0, var=6.0, group_id="T")
        site_total_volume = control.mean_y() * control.n + treatment.mean_y() * treatment.n

        result = sitewide_impact(
            SitewideContrast.from_iid(control, treatment), site_total_volume=site_total_volume
        )

        expected_baseline = control.mean_y() * (control.n + treatment.n)
        expected_relative = (treatment.mean_y() - control.mean_y()) / control.mean_y()

        assert result.baseline_volume == pytest.approx(expected_baseline)
        assert result.relative_impact == pytest.approx(expected_relative)

    def test_nonpositive_baseline_raises(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
        treatment = _arm(n=100, mean=1000.0, var=4.0, group_id="T")
        contrast = SitewideContrast.from_iid(control, treatment)

        with pytest.raises(InvalidRequestError) as exc_info:
            sitewide_impact(contrast, site_total_volume=1.0)
        assert exc_info.value.code == "estimation.sitewide.counterfactual_baseline_volume"


class TestSitewideImpactRatio:
    def test_hand_computed_case(self):
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=120,
            mean_num=12.0,
            var_num=9.0,
            mean_den=6.0,
            var_den=2.0,
            cov_yden=0.8,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)
        site_total_numerator = 50_000.0
        site_total_denominator = 20_000.0

        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=site_total_numerator,
            site_total_denominator=site_total_denominator,
        )

        delta_num = 2.0
        delta_den = 1.0
        delta_num_se = math.sqrt(9.0 / 120 + 4.0 / 100)
        delta_den_se = math.sqrt(2.0 / 120 + 1.0 / 100)
        delta_cov = 0.8 / 120 + 0.5 / 100
        n_exp = 220

        n0 = site_total_numerator - delta_num * 120
        d0 = site_total_denominator - delta_den * 120
        n1 = n0 + n_exp * delta_num
        d1 = d0 + n_exp * delta_den
        assert n0 == pytest.approx(49_760.0)
        assert d0 == pytest.approx(19_880.0)
        assert n1 == pytest.approx(50_200.0)
        assert d1 == pytest.approx(20_100.0)

        impact = n1 / d1 - n0 / d0
        g_num = 100 / d1 + 120 / d0
        g_den = -n1 * 100 / d1**2 - n0 * 120 / d0**2
        impact_var = (
            g_num**2 * delta_num_se**2 + g_den**2 * delta_den_se**2 + 2 * g_num * g_den * delta_cov
        )
        impact_se = math.sqrt(impact_var)
        z = float(_norm.ppf(0.975))

        assert result.delta_num == pytest.approx(delta_num)
        assert result.delta_den == pytest.approx(delta_den)
        assert result.delta_num_se == pytest.approx(delta_num_se)
        assert result.delta_den_se == pytest.approx(delta_den_se)
        assert result.delta_cov == pytest.approx(delta_cov)
        assert result.n_control == pytest.approx(100.0)
        assert result.n_treatment == pytest.approx(120.0)
        assert result.baseline_numerator == pytest.approx(n0)
        assert result.baseline_denominator == pytest.approx(d0)
        assert result.shipped_numerator == pytest.approx(n1)
        assert result.shipped_denominator == pytest.approx(d1)
        assert result.absolute_impact == pytest.approx(impact)
        assert result.absolute_impact_se == pytest.approx(impact_se)
        assert result.absolute_impact_lb == pytest.approx(impact - z * impact_se)
        assert result.absolute_impact_ub == pytest.approx(impact + z * impact_se)
        assert result.n_clusters is None
        assert result.absolute_dof is None
        assert result.relative_dof is None

    def test_delta_den_zero_reduces_to_sum_metric_formula(self):
        # delta_den == 0 => D0 == D1 == site_total_denominator (const D), so
        # impact = delta_num * N_exp / D, the ratio analogue of delta * N_exp.
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=120,
            mean_num=12.0,
            var_num=9.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)
        site_total_numerator = 50_000.0
        site_total_denominator = 20_000.0

        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=site_total_numerator,
            site_total_denominator=site_total_denominator,
        )

        assert result.delta_den == pytest.approx(0.0, abs=1e-9)
        assert result.baseline_denominator == pytest.approx(site_total_denominator)
        assert result.shipped_denominator == pytest.approx(site_total_denominator)

        n_exp = 220
        expected_impact = result.delta_num * n_exp / site_total_denominator
        assert result.absolute_impact == pytest.approx(expected_impact)

    def test_cov_yden_sign_changes_se_in_expected_direction(self):
        # g_num > 0, g_den < 0 here, so g_num*g_den < 0: a positive cov_yden
        # narrows the SE (negative cross term), a negative one widens it.
        def build(cov_sign: float):
            control = _ratio_arm(
                n=100,
                mean_num=10.0,
                var_num=4.0,
                mean_den=5.0,
                var_den=1.0,
                cov_yden=cov_sign * 0.5,
                group_id="C",
            )
            treatment = _ratio_arm(
                n=120,
                mean_num=12.0,
                var_num=9.0,
                mean_den=6.0,
                var_den=2.0,
                cov_yden=cov_sign * 0.8,
                group_id="T",
            )
            contrast = SitewideRatioContrast.from_iid(control, treatment)
            return sitewide_impact_ratio(
                contrast,
                site_total_numerator=50_000.0,
                site_total_denominator=20_000.0,
            )

        positive_cov = build(1.0)
        negative_cov = build(-1.0)

        # delta_num/delta_den/N0/D0/N1/D1 don't depend on cov_yden's sign,
        # so both results share the same gradient - derive it, don't hardcode.
        g_num = (
            positive_cov.n_control / positive_cov.shipped_denominator
            + positive_cov.n_treatment / positive_cov.baseline_denominator
        )
        g_den = (
            -positive_cov.shipped_numerator
            * positive_cov.n_control
            / positive_cov.shipped_denominator**2
            - positive_cov.baseline_numerator
            * positive_cov.n_treatment
            / positive_cov.baseline_denominator**2
        )
        assert g_num * g_den < 0
        assert positive_cov.delta_cov > 0
        assert negative_cov.delta_cov < 0
        assert positive_cov.absolute_impact_se < negative_cov.absolute_impact_se

    def test_nonpositive_baseline_denominator_raises(self):
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=1000.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)

        with pytest.raises(InvalidRequestError) as exc_info:
            sitewide_impact_ratio(contrast, site_total_numerator=1.0, site_total_denominator=1.0)
        assert exc_info.value.code == "estimation.sitewide.counterfactual_baseline_denominator"

    def test_nonpositive_shipped_denominator_raises(self):
        """D0 (counterfactual) can stay positive while D1 (ship-to-all)
        goes non-positive - a large negative delta_den scaled by n_C
        (ship-to-all) overwhelms the site total even though scaling by
        the smaller n_T (counterfactual) does not. Independent guard
        from test_nonpositive_baseline_denominator_raises above."""
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=1000.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=10,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)

        with pytest.raises(InvalidRequestError) as exc_info:
            sitewide_impact_ratio(contrast, site_total_numerator=1.0, site_total_denominator=5000.0)
        assert exc_info.value.code == "estimation.sitewide.ship_all_denominator"

    def test_missing_ratio_family_raises(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
        treatment = _arm(n=100, mean=12.0, var=4.0, group_id="T")

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast.from_iid(control, treatment)
        assert exc_info.value.code == "estimation.variance.ratio_moments_needs_ref_den"

    def test_hand_computed_relative_impact(self):
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=120,
            mean_num=12.0,
            var_num=9.0,
            mean_den=6.0,
            var_den=2.0,
            cov_yden=0.8,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)

        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=50_000.0,
            site_total_denominator=20_000.0,
        )

        # Same arms as test_hand_computed_case, so N0/D0/N1/D1 are the
        # integers pinned there.
        n0, d0, n1, d1 = 49_760.0, 19_880.0, 50_200.0, 20_100.0
        baseline_ratio = n0 / d0
        impact = n1 / d1 - n0 / d0
        relative = impact / baseline_ratio
        # Independent closed form: impact/(N0/D0) == (N1*D0)/(N0*D1) - 1.
        assert relative == pytest.approx((n1 * d0) / (n0 * d1) - 1.0)

        delta_num_se = math.sqrt(9.0 / 120 + 4.0 / 100)
        delta_den_se = math.sqrt(2.0 / 120 + 1.0 / 100)
        delta_cov = 0.8 / 120 + 0.5 / 100
        h_num = (d0 / d1) * (100 * n0 + 120 * n1) / n0**2
        h_den = -(n1 / n0) * (120 * d1 + 100 * d0) / d1**2
        rel_se = math.sqrt(
            h_num**2 * delta_num_se**2 + h_den**2 * delta_den_se**2 + 2 * h_num * h_den * delta_cov
        )
        z = float(_norm.ppf(0.975))

        assert result.baseline_ratio == pytest.approx(baseline_ratio)
        assert result.relative_impact == pytest.approx(relative)
        assert result.relative_impact_se == pytest.approx(rel_se)
        assert result.relative_impact_lb == pytest.approx(relative - z * rel_se)
        assert result.relative_impact_ub == pytest.approx(relative + z * rel_se)

    def test_relative_gradient_matches_finite_differences(self):
        # The analytic partials of rel(delta_num, delta_den) drive interval
        # width; re-derive numerically so a bad edit to either fails loudly.
        n_c, n_t = 100, 120
        site_num, site_den = 50_000.0, 20_000.0

        def rel_of(delta_num: float, delta_den: float) -> float:
            n0 = site_num - n_t * delta_num
            d0 = site_den - n_t * delta_den
            n1 = n0 + (n_c + n_t) * delta_num
            d1 = d0 + (n_c + n_t) * delta_den
            return (n1 / d1 - n0 / d0) / (n0 / d0)

        delta_num, delta_den, h = 2.0, 1.0, 1e-4
        n0 = site_num - n_t * delta_num
        d0 = site_den - n_t * delta_den
        n1 = n0 + (n_c + n_t) * delta_num
        d1 = d0 + (n_c + n_t) * delta_den
        h_num = (d0 / d1) * (n_c * n0 + n_t * n1) / n0**2
        h_den = -(n1 / n0) * (n_t * d1 + n_c * d0) / d1**2

        fd_num = (rel_of(delta_num + h, delta_den) - rel_of(delta_num - h, delta_den)) / (2 * h)
        fd_den = (rel_of(delta_num, delta_den + h) - rel_of(delta_num, delta_den - h)) / (2 * h)
        assert h_num == pytest.approx(fd_num, rel=1e-6)
        assert h_den == pytest.approx(fd_den, rel=1e-8)

    def test_delta_den_zero_reduces_relative_to_closed_form(self):
        # delta_den == 0 makes D0/D1 exactly 1, so rel = N_exp*delta_num/N0 -
        # the ratio analogue of the sum metric's `N_exp*delta/V0`.
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=120,
            mean_num=12.0,
            var_num=9.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)
        site_total_numerator = 50_000.0

        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=site_total_numerator,
            site_total_denominator=20_000.0,
        )

        assert result.delta_den == pytest.approx(0.0, abs=1e-9)
        n_exp = 220
        n0 = site_total_numerator - 120 * result.delta_num
        assert result.baseline_numerator == pytest.approx(n0)
        assert result.relative_impact == pytest.approx(result.delta_num * n_exp / n0)
        # 2.0 * 220 / (50_000 - 240) == 440/49_760
        assert result.relative_impact == pytest.approx(440.0 / 49_760.0)

    def test_nonpositive_baseline_ratio_raises(self):
        """N0 can go non-positive while both denominators stay healthy;
        a numerator lift large relative to the site numerator total wipes
        out the counterfactual baseline, leaving relative impact with no
        meaningful base to divide by. Both arms carry a small POSITIVE
        numerator mean (ratio_moments refuses <= 0 at contrast-build time,
        a stricter check than the old direct-accessor path), and the
        AGGREGATE effect against a tiny site total is what pushes N0
        negative, not either arm's own mean."""
        control = _ratio_arm(
            n=100,
            mean_num=0.5,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=100,
            mean_num=100.5,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)

        with pytest.raises(InvalidRequestError) as exc_info:
            sitewide_impact_ratio(
                contrast, site_total_numerator=1.0, site_total_denominator=1_000.0
            )
        assert exc_info.value.code == "estimation.sitewide.counterfactual_baseline_ratio"

    def test_from_iid_accepts_a_nonpositive_control_numerator(self):
        """SitewideRatioContrast never takes a log of the arm numerator --
        it feeds an additive site-volume-weighted impact. A refusal here
        was inherited from ratio_moments' shared (log-path-only) guard,
        not a requirement of this construction."""
        control = _ratio_arm(
            n=100,
            mean_num=0.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="C",
        )
        treatment = _ratio_arm(
            n=100,
            mean_num=100.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.1,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_iid(control, treatment)
        assert contrast.control.mean == pytest.approx(0.0)


def _fd(f, x: list[float], i: int, h: float) -> float:
    """4th-order central difference of *f* in coordinate *i*."""

    def at(step: float) -> float:
        moved = list(x)
        moved[i] += step
        return f(*moved)

    return (-at(2 * h) + 8 * at(h) - 8 * at(-h) + at(-2 * h)) / (12 * h)


def _quadratic_form(grad: list[float], sigma: list[list[float]]) -> float:
    """``grad' sigma grad``, spelled out so the covariance structure the
    estimator threads implicitly is pinned explicitly here."""
    return sum(grad[i] * sigma[i][j] * grad[j] for i in range(len(grad)) for j in range(len(grad)))


def _multi_arm_set() -> tuple[ArmStats, ArmStats, ArmStats]:
    """Hand-worked three-arm sum-metric case: control mean 10.0 (n=100),
    treatment_a mean 12.0 (n=120, delta 2.0), treatment_b mean 11.0 (n=80,
    delta 1.0). With V = 50_000 that puts V0 at 49_680 and N_exp at 300."""
    return (
        _arm(n=100, mean=10.0, var=4.0, group_id="control"),
        _arm(n=120, mean=12.0, var=9.0, group_id="treatment_a"),
        _arm(n=80, mean=11.0, var=6.0, group_id="treatment_b"),
    )


def _multi_arm_ratio_set() -> tuple[ArmStats, ArmStats, ArmStats]:
    """Hand-worked three-arm ratio case: control 10.0/5.0 (n=100),
    treatment_a 12.0/6.0 (n=120, deltas 2.0/1.0), treatment_b 11.0/5.5
    (n=80, deltas 1.0/0.5)."""
    return (
        _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="control",
        ),
        _ratio_arm(
            n=120,
            mean_num=12.0,
            var_num=9.0,
            mean_den=6.0,
            var_den=2.0,
            cov_yden=0.8,
            group_id="treatment_a",
        ),
        _ratio_arm(
            n=80,
            mean_num=11.0,
            var_num=6.0,
            mean_den=5.5,
            var_den=1.5,
            cov_yden=0.6,
            group_id="treatment_b",
        ),
    )


class TestSitewideImpactMultiArm:
    """A third arm's lift is also inside the observed site total, so it
    must leave the counterfactual baseline too - and its delta shares the
    control mean with the target arm's, which the interval has to carry."""

    # Arms and their hand-worked totals: see _multi_arm_set.
    SITE_TOTAL = 50_000.0

    def test_baseline_nets_out_every_enrolled_arm(self):
        control, arm_a, arm_b = _multi_arm_set()
        contrast = SitewideContrast.from_iid(control, arm_a, other_arms=[arm_b])

        result = sitewide_impact(contrast, site_total_volume=self.SITE_TOTAL)

        # 50_000 - 2*120 (arm A) - 1*80 (arm B). Netting only arm A would
        # leave 49_760 - arm B's $80 of lift still counted as baseline.
        assert result.baseline_volume == pytest.approx(49_680.0)
        assert result.baseline_volume != pytest.approx(49_760.0)
        assert result.n_enrolled == pytest.approx(300.0)
        assert result.treatment_group == "treatment_a"
        assert result.other_arm_ids == ("treatment_b",)
        # Ship-to-all reaches every enrolled unit, arm B's 80 included.
        assert result.absolute_impact == pytest.approx(600.0)
        assert result.relative_impact == pytest.approx(600.0 / 49_680.0)

    def test_second_arm_shares_the_baseline_and_scales_its_own_lift(self):
        """Scoring arm B instead must reuse the SAME all-control baseline
        (both arms' lifts are out of it) while reporting only its own,
        smaller ship-to-all gain."""
        control, arm_a, arm_b = _multi_arm_set()

        for_a = sitewide_impact(
            SitewideContrast.from_iid(control, arm_a, other_arms=[arm_b]),
            site_total_volume=self.SITE_TOTAL,
        )
        for_b = sitewide_impact(
            SitewideContrast.from_iid(control, arm_b, other_arms=[arm_a]),
            site_total_volume=self.SITE_TOTAL,
        )

        assert for_b.baseline_volume == pytest.approx(for_a.baseline_volume)
        assert for_b.n_enrolled == pytest.approx(for_a.n_enrolled) == pytest.approx(300.0)
        assert for_b.treatment_group == "treatment_b"
        assert for_b.other_arm_ids == ("treatment_a",)
        assert for_b.absolute_impact == pytest.approx(300.0)
        assert for_b.relative_impact == pytest.approx(300.0 / 49_680.0)

    def test_two_arm_case_reproduces_the_single_arm_closed_form(self):
        """No other arms -> the generalized path must land exactly on the
        two-arm formulas (V0 = V - delta*n_T, rel SE = |N_exp*V/V0**2|*
        delta_se), written out here rather than read off the result."""
        control, treatment, _ = _multi_arm_set()
        contrast = SitewideContrast.from_iid(control, treatment)

        result = sitewide_impact(contrast, site_total_volume=self.SITE_TOTAL)

        delta = 2.0
        delta_se = math.sqrt(9.0 / 120 + 4.0 / 100)
        n_exp = 220
        baseline = self.SITE_TOTAL - delta * 120
        relative = delta * n_exp / baseline
        rel_se = abs(n_exp * self.SITE_TOTAL / baseline**2) * delta_se
        z = float(_norm.ppf(0.975))

        assert result.baseline_volume == pytest.approx(baseline)
        assert result.n_enrolled == pytest.approx(n_exp)
        assert result.other_arm_ids == ()
        assert result.absolute_impact == pytest.approx(delta * n_exp)
        assert result.absolute_impact_ub == pytest.approx(delta * n_exp + z * delta_se * n_exp)
        assert result.relative_impact == pytest.approx(relative)
        assert result.relative_impact_lb == pytest.approx(relative - z * rel_se)
        assert result.relative_impact_ub == pytest.approx(relative + z * rel_se)

    def test_relative_gradient_matches_finite_differences(self):
        """Both partials of rel(delta_a, delta_b) - the target arm's and
        the co-enrolled arm's, which enters only through the shared
        baseline - against 4th-order central differences."""
        n_c, n_a, n_b = 100, 120, 80
        n_exp = n_c + n_a + n_b

        def rel_of(delta_a: float, delta_b: float) -> float:
            baseline = self.SITE_TOTAL - delta_a * n_a - delta_b * n_b
            return delta_a * n_exp / baseline

        delta_a, delta_b = 2.0, 1.0
        baseline = self.SITE_TOTAL - delta_a * n_a - delta_b * n_b
        # Target partial: the two-arm closed form with V net of arm B.
        g_a = n_exp * (self.SITE_TOTAL - delta_b * n_b) / baseline**2
        g_b = delta_a * n_exp * n_b / baseline**2

        at = [delta_a, delta_b]
        assert g_a == pytest.approx(_fd(rel_of, at, 0, 1e-3), rel=1e-9)
        assert g_b == pytest.approx(_fd(rel_of, at, 1, 1e-3), rel=1e-8)

    def test_relative_variance_carries_the_shared_control_covariance(self):
        """The reported variance must equal ``g' Sigma g`` for the Sigma
        whose off-diagonal is Var(mean_control) = var_y(C)/n_C - built
        explicitly here, since the estimator never assembles it."""
        control, arm_a, arm_b = _multi_arm_set()
        contrast = SitewideContrast.from_iid(control, arm_a, other_arms=[arm_b])
        result = sitewide_impact(contrast, site_total_volume=self.SITE_TOTAL)

        n_exp, baseline = 300, 49_680.0
        grad = [
            n_exp * (self.SITE_TOTAL - 1.0 * 80) / baseline**2,
            2.0 * n_exp * 80 / baseline**2,
        ]
        shared = 4.0 / 100  # Var(mean_control): both deltas subtract it
        sigma = [
            [9.0 / 120 + shared, shared],
            [shared, 6.0 / 80 + shared],
        ]
        z = float(_norm.ppf(0.975))
        reported_var = ((result.relative_impact_ub - result.relative_impact) / z) ** 2

        assert reported_var == pytest.approx(_quadratic_form(grad, sigma), rel=1e-12)
        assert result.relative_impact_se == pytest.approx(math.sqrt(reported_var))

    def test_shared_control_noise_widens_a_same_sign_gradient(self):
        """Cov(delta_a, delta_b) = +var_y(C)/n_C, so when both partials
        share a sign the cross terms ADD 2*g_a*g_b*var_y(C)/n_C to the
        variance - pretending the arms were independent would understate
        the interval by exactly that much."""
        control, arm_a, arm_b = _multi_arm_set()
        contrast = SitewideContrast.from_iid(control, arm_a, other_arms=[arm_b])
        result = sitewide_impact(contrast, site_total_volume=self.SITE_TOTAL)

        n_exp, baseline = 300, 49_680.0
        g_a = n_exp * (self.SITE_TOTAL - 1.0 * 80) / baseline**2
        g_b = 2.0 * n_exp * 80 / baseline**2
        shared = 4.0 / 100
        independent_var = g_a**2 * (9.0 / 120 + shared) + g_b**2 * (6.0 / 80 + shared)
        z = float(_norm.ppf(0.975))
        reported_var = ((result.relative_impact_ub - result.relative_impact) / z) ** 2

        assert g_a > 0 and g_b > 0
        assert reported_var > independent_var
        assert reported_var - independent_var == pytest.approx(2 * g_a * g_b * shared, rel=1e-9)

    def test_shared_control_noise_narrows_an_opposite_sign_gradient(self):
        """A target arm that LOSES flips its own ship-to-all impact
        negative, so the co-enrolled arm's partial flips sign too and the
        same covariance term now narrows the interval. Same machinery, and
        the direction follows the gradient signs rather than being
        asserted as merely 'different'."""
        control, _, arm_b = _multi_arm_set()
        losing = _arm(n=120, mean=8.0, var=9.0, group_id="treatment_a")
        contrast = SitewideContrast.from_iid(control, losing, other_arms=[arm_b])

        result = sitewide_impact(contrast, site_total_volume=self.SITE_TOTAL)

        n_exp = 300
        baseline = self.SITE_TOTAL - (-2.0) * 120 - 1.0 * 80
        g_a = n_exp * (self.SITE_TOTAL - 1.0 * 80) / baseline**2
        g_b = -2.0 * n_exp * 80 / baseline**2
        shared = 4.0 / 100
        independent_var = g_a**2 * (9.0 / 120 + shared) + g_b**2 * (6.0 / 80 + shared)
        z = float(_norm.ppf(0.975))
        reported_var = ((result.relative_impact_ub - result.relative_impact) / z) ** 2

        assert g_a > 0 > g_b
        assert reported_var < independent_var
        assert reported_var - independent_var == pytest.approx(2 * g_a * g_b * shared, rel=1e-9)

    def test_absolute_impact_ignores_the_other_arms_sampling_noise(self):
        """Absolute impact is linear in the TARGET delta alone - the other
        arms move the baseline, not the ship-to-all gain - so its SE stays
        the plain Welch interval scaled by the enrolled population."""
        control, arm_a, arm_b = _multi_arm_set()
        contrast = SitewideContrast.from_iid(control, arm_a, other_arms=[arm_b])
        result = sitewide_impact(contrast, site_total_volume=self.SITE_TOTAL)

        z = float(_norm.ppf(0.975))
        delta_se = math.sqrt(9.0 / 120 + 4.0 / 100)
        assert result.absolute_impact_ub - result.absolute_impact == pytest.approx(
            z * delta_se * 300
        )

    def test_duplicate_other_arm_refuses(self):
        control, arm_a, _ = _multi_arm_set()

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, arm_a, other_arms=[arm_a])
        assert exc_info.value.code == "estimation.sitewide.other_arms_repeats"

    def test_other_arm_from_another_metric_refuses(self):
        control, arm_a, _ = _multi_arm_set()
        foreign = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="sessions",
            group_id="treatment_b",
            n=80,
            sum_y=880.0,
            sum_y2=10_154.0,
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, arm_a, other_arms=[foreign])
        assert exc_info.value.code == "estimation.sitewide.other_arms_entry_metric_mismatch"


class TestSitewideImpactRatioMultiArm:
    """Same generalization for the ratio path: BOTH site totals net out
    every enrolled arm, and the gradient gains a numerator/denominator
    pair per co-enrolled arm."""

    # Arms and their hand-worked deltas: see _multi_arm_ratio_set.
    SITE_NUM = 50_000.0
    SITE_DEN = 20_000.0

    def _result(self):
        control, arm_a, arm_b = _multi_arm_ratio_set()
        contrast = SitewideRatioContrast.from_iid(control, arm_a, other_arms=[arm_b])
        return sitewide_impact_ratio(
            contrast,
            site_total_numerator=self.SITE_NUM,
            site_total_denominator=self.SITE_DEN,
        )

    def test_both_baselines_net_out_every_enrolled_arm(self):
        result = self._result()

        # N0=50_000-120*2-80*1=49_680  D0=20_000-120*1-80*0.5=19_840
        # N1=N0+300*2=50_280           D1=D0+300*1=20_140
        assert result.baseline_numerator == pytest.approx(49_680.0)
        assert result.baseline_denominator == pytest.approx(19_840.0)
        assert result.shipped_numerator == pytest.approx(50_280.0)
        assert result.shipped_denominator == pytest.approx(20_140.0)
        assert result.n_enrolled == pytest.approx(300.0)
        assert result.other_arm_ids == ("treatment_b",)
        # Netting only arm A would leave N0 = 49_760, D0 = 19_880 - the
        # numbers the two-arm path reports for these same arms.
        assert result.baseline_numerator != pytest.approx(49_760.0)
        assert result.baseline_denominator != pytest.approx(19_880.0)
        assert result.absolute_impact == pytest.approx(50_280.0 / 20_140.0 - 49_680.0 / 19_840.0)

    def test_two_arm_case_reproduces_the_single_arm_closed_form(self):
        """With no other arms the multi-arm gradient must collapse onto the
        two-arm partials (n_switched == n_control), pinned here against the
        module docstring's g_num/g_den written out by hand."""
        control, treatment, _ = _multi_arm_ratio_set()
        contrast = SitewideRatioContrast.from_iid(control, treatment)

        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=self.SITE_NUM,
            site_total_denominator=self.SITE_DEN,
        )

        n0, d0, n1, d1 = 49_760.0, 19_880.0, 50_200.0, 20_100.0
        assert result.baseline_numerator == pytest.approx(n0)
        assert result.baseline_denominator == pytest.approx(d0)
        assert result.n_enrolled == pytest.approx(220.0)
        assert result.other_arm_ids == ()

        delta_num_se = math.sqrt(9.0 / 120 + 4.0 / 100)
        delta_den_se = math.sqrt(2.0 / 120 + 1.0 / 100)
        delta_cov = 0.8 / 120 + 0.5 / 100
        g_num = 100 / d1 + 120 / d0
        g_den = -n1 * 100 / d1**2 - n0 * 120 / d0**2
        impact_se = math.sqrt(
            g_num**2 * delta_num_se**2 + g_den**2 * delta_den_se**2 + 2 * g_num * g_den * delta_cov
        )
        assert result.absolute_impact_se == pytest.approx(impact_se)

    def test_gradients_match_finite_differences(self):
        """All four partials of impact(delta_num_a, delta_den_a,
        delta_num_b, delta_den_b) and all four of the relative impact's,
        against 4th-order central differences."""
        n_a, n_b, n_exp = 120, 80, 300
        n_switched = n_exp - n_a

        def totals(dna, dda, dnb, ddb):
            n0 = self.SITE_NUM - n_a * dna - n_b * dnb
            d0 = self.SITE_DEN - n_a * dda - n_b * ddb
            return n0, d0, n0 + n_exp * dna, d0 + n_exp * dda

        def impact_of(dna, dda, dnb, ddb):
            n0, d0, n1, d1 = totals(dna, dda, dnb, ddb)
            return n1 / d1 - n0 / d0

        def rel_of(dna, dda, dnb, ddb):
            n0, d0, n1, d1 = totals(dna, dda, dnb, ddb)
            return (n1 / n0) * (d0 / d1) - 1.0

        at = [2.0, 1.0, 1.0, 0.5]
        n0, d0, n1, d1 = totals(*at)
        analytic_impact = [
            n_switched / d1 + n_a / d0,
            -n1 * n_switched / d1**2 - n0 * n_a / d0**2,
            n_b * (1 / d0 - 1 / d1),
            n_b * (n1 / d1**2 - n0 / d0**2),
        ]
        analytic_rel = [
            (d0 / d1) * (n_switched * n0 + n_a * n1) / n0**2,
            -(n1 / n0) * (n_a * d1 + n_switched * d0) / d1**2,
            (d0 / d1) * n_b * (n1 - n0) / n0**2,
            (n1 / n0) * n_b * (d0 - d1) / d1**2,
        ]
        for i, expected in enumerate(analytic_impact):
            assert expected == pytest.approx(_fd(impact_of, at, i, 0.1), rel=1e-9)
        for i, expected in enumerate(analytic_rel):
            assert expected == pytest.approx(_fd(rel_of, at, i, 0.1), rel=1e-9)

    def test_variance_carries_the_shared_control_covariance(self):
        """Reported variance == ``g' Sigma g`` over the 4-vector
        (delta_num_a, delta_den_a, delta_num_b, delta_den_b), whose blocks
        are each arm's own within-arm num/den covariance plus the control
        block shared by BOTH arms."""
        result = self._result()
        n_a, n_b, n_exp = 120, 80, 300
        n_switched = n_exp - n_a
        n0, d0, n1, d1 = 49_680.0, 19_840.0, 50_280.0, 20_140.0

        grad_impact = [
            n_switched / d1 + n_a / d0,
            -n1 * n_switched / d1**2 - n0 * n_a / d0**2,
            n_b * (1 / d0 - 1 / d1),
            n_b * (n1 / d1**2 - n0 / d0**2),
        ]
        grad_rel = [
            (d0 / d1) * (n_switched * n0 + n_a * n1) / n0**2,
            -(n1 / n0) * (n_a * d1 + n_switched * d0) / d1**2,
            (d0 / d1) * n_b * (n1 - n0) / n0**2,
            (n1 / n0) * n_b * (d0 - d1) / d1**2,
        ]
        # Control block (var_num, var_den, cov) / n_C - shared by every
        # pair of arms, since every delta subtracts the control means.
        c_num, c_den, c_cov = 4.0 / 100, 1.0 / 100, 0.5 / 100
        sigma = [[0.0] * 4 for _ in range(4)]
        for i in (0, 2):
            for j in (0, 2):
                sigma[i][j] += c_num
                sigma[i + 1][j + 1] += c_den
                sigma[i][j + 1] += c_cov
                sigma[i + 1][j] += c_cov
        for offset, (vn, vd, cv, n) in enumerate([(9.0, 2.0, 0.8, 120), (6.0, 1.5, 0.6, 80)]):
            i = 2 * offset
            sigma[i][i] += vn / n
            sigma[i + 1][i + 1] += vd / n
            sigma[i][i + 1] += cv / n
            sigma[i + 1][i] += cv / n

        z = float(_norm.ppf(0.975))
        reported_rel_var = ((result.relative_impact_ub - result.relative_impact) / z) ** 2
        assert result.absolute_impact_se**2 == pytest.approx(
            _quadratic_form(grad_impact, sigma), rel=1e-12
        )
        assert reported_rel_var == pytest.approx(_quadratic_form(grad_rel, sigma), rel=1e-12)

    def test_ignoring_the_other_arm_understates_the_interval(self):
        """Dropping the co-enrolled arm entirely (what a naive per-arm loop
        does) shifts BOTH baselines and the interval - the discrepancy
        this generalization exists to remove, pinned so it cannot silently
        come back."""
        control, arm_a, arm_b = _multi_arm_ratio_set()
        with_b = sitewide_impact_ratio(
            SitewideRatioContrast.from_iid(control, arm_a, other_arms=[arm_b]),
            site_total_numerator=self.SITE_NUM,
            site_total_denominator=self.SITE_DEN,
        )
        naive = sitewide_impact_ratio(
            SitewideRatioContrast.from_iid(control, arm_a),
            site_total_numerator=self.SITE_NUM,
            site_total_denominator=self.SITE_DEN,
        )

        assert with_b.baseline_numerator < naive.baseline_numerator
        assert with_b.baseline_denominator < naive.baseline_denominator
        assert with_b.n_enrolled > naive.n_enrolled
        assert with_b.absolute_impact_se != pytest.approx(naive.absolute_impact_se)


class TestArmValidation:
    """Illegal-state refusals for _Arm/SitewideContrast/SitewideRatioContrast
    construction (design doc section 2)."""

    def test_control_and_target_metric_mismatch_refuses(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C", metric="rev")
        target = _arm(n=100, mean=12.0, var=4.0, group_id="T", metric="sessions")

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, target)
        assert exc_info.value.code == "estimation.sitewide.control_carries_metric"

    def test_control_and_target_same_group_id_refuses(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="dup")
        target = _arm(n=100, mean=12.0, var=4.0, group_id="dup")

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, target)
        assert exc_info.value.code == "estimation.sitewide.control_target_are"

    def test_control_and_target_different_study_refuses(self):
        """_checked_arms validated metric identity but never study
        identity -- a control ArmStats from one experiment and a target
        from another were silently accepted and contrasted."""
        control = ArmStats.from_raw_sums(
            study_id="exp_A", metric="rev", group_id="C", n=100, sum_y=1000.0, sum_y2=10396.0
        )
        target = ArmStats.from_raw_sums(
            study_id="exp_B", metric="rev", group_id="T", n=100, sum_y=1200.0, sum_y2=15291.0
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, target)
        assert exc_info.value.code == "estimation.sitewide.control_carries_study"

    def test_other_arm_from_different_study_refuses(self):
        """The same cross-experiment gap for a co-enrolled arm."""
        control, arm_a, _ = _multi_arm_set()
        foreign = ArmStats.from_raw_sums(
            study_id="exp_other",
            metric="rev",
            group_id="treatment_b",
            n=80,
            sum_y=880.0,
            sum_y2=10154.0,
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, arm_a, other_arms=[foreign])
        assert exc_info.value.code == "estimation.sitewide.other_arms_entry_study_mismatch"

    def test_from_iid_refuses_a_clustered_looking_sum_arm(self):
        """A clustered sum-metric row always populates ref_den (cluster
        size); from_iid must refuse rather than silently misread it as a
        per-unit mean/variance."""
        control = _cluster_arm(
            n=25, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _arm(n=120, mean=12.0, var=9.0, group_id="T")

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, target)
        assert exc_info.value.code == "estimation.sitewide.from_iid.metric_arm_carries_ref_den"

    def test_from_iid_refuses_a_clustered_looking_other_arm(self):
        """The ref_den check covered control/target but not
        other_arms, so a clustered co-enrolled arm was silently misread
        as a per-unit mean/variance."""
        control, arm_a, _ = _multi_arm_set()
        clustered_other = _cluster_arm(
            n=25,
            g_mean=90.0,
            g_var=49.0,
            m_mean=20.0,
            m_var=4.0,
            cov_gm=6.0,
            group_id="treatment_b",
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideContrast.from_iid(control, arm_a, other_arms=[clustered_other])
        assert exc_info.value.code == "estimation.sitewide.from_iid.metric_arm_carries_ref_den"

    def test_ratio_contrast_refuses_a_sum_family_arm(self):
        """SitewideRatioContrast's model-level validator refuses an _Arm
        missing the denominator family - structurally unsatisfiable by a
        sum-family arm, not just an accident of the two constructors."""
        sum_shaped = _Arm(group_id="c", n_units=100.0, mean=1.0, var_mean=0.01)
        ratio_shaped = _Arm(
            group_id="t",
            n_units=100.0,
            mean=1.0,
            var_mean=0.01,
            mean_den=2.0,
            var_mean_den=0.02,
            cov_mean_den=0.01,
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast(metric="m", control=sum_shaped, target=ratio_shaped)
        assert exc_info.value.code == "estimation.sitewide.sitewide_ratio.metric_arm_missing"

    def test_ratio_from_iid_refuses_a_clustered_control_arm(self):
        """SitewideRatioContrast.from_iid had NO cluster-shape
        rejection at all -- a clustered ratio row (x_role='cluster_size')
        was silently averaged as if per-unit."""
        clustered_control = _cluster_ratio_arm(
            n=25,
            num_mean=100.0,
            num_var=64.0,
            den_mean=50.0,
            den_var=16.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=20.0,
            cov_num_m=8.0,
            cov_den_m=6.0,
            group_id="C",
        )
        target = _ratio_arm(
            n=100,
            mean_num=12.0,
            var_num=9.0,
            mean_den=6.0,
            var_den=2.0,
            cov_yden=0.8,
            group_id="T",
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast.from_iid(clustered_control, target)
        assert exc_info.value.code == "estimation.sitewide.sitewide_ratio.metric_arm_declares"

    def test_ratio_from_iid_refuses_a_clustered_target_arm(self):
        control = _ratio_arm(
            n=100,
            mean_num=10.0,
            var_num=4.0,
            mean_den=5.0,
            var_den=1.0,
            cov_yden=0.5,
            group_id="C",
        )
        clustered_target = _cluster_ratio_arm(
            n=25,
            num_mean=110.0,
            num_var=81.0,
            den_mean=52.0,
            den_var=18.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=21.0,
            cov_num_m=8.5,
            cov_den_m=6.5,
            group_id="T",
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast.from_iid(control, clustered_target)
        assert exc_info.value.code == "estimation.sitewide.sitewide_ratio.metric_arm_declares"

    def test_ratio_from_iid_refuses_a_clustered_other_arm(self):
        control, arm_a, _ = _multi_arm_ratio_set()
        clustered_other = _cluster_ratio_arm(
            n=25,
            num_mean=105.0,
            num_var=70.0,
            den_mean=51.0,
            den_var=17.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=20.5,
            cov_num_m=8.2,
            cov_den_m=6.2,
            group_id="treatment_b",
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast.from_iid(control, arm_a, other_arms=[clustered_other])
        assert exc_info.value.code == "estimation.sitewide.sitewide_ratio.metric_arm_declares"

    def test_from_clusters_admits_below_ten_total_clusters(self):
        control = _cluster_arm(
            n=3, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _cluster_arm(
            n=3, g_mean=110.0, g_var=81.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
        )
        with pytest.warns(IncrementRuntimeWarning) as rec:
            contrast = SitewideContrast.from_clusters(control, target, cluster="store_id")
        assert "estimation.engine.small_total_clusters" in warning_codes(rec)
        assert contrast.n_clusters == 6

    def test_from_clusters_warns_below_forty_total_clusters(self):
        control = _cluster_arm(
            n=10, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _cluster_arm(
            n=10, g_mean=110.0, g_var=81.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
        )

        with pytest.warns(IncrementRuntimeWarning) as rec:
            contrast = SitewideContrast.from_clusters(control, target, cluster="store_id")
        assert "estimation.engine.small_total_clusters" in warning_codes(rec)
        assert contrast.n_clusters == 20


class TestSitewideImpactClustered:
    """Hand-computed cluster-grain sum-metric contrast: 25 clusters/arm,
    each control cluster's outcome total g_j ~ mean 100.0 var 64.0, cluster
    size m_j ~ mean 20.0 var 4.0, Cov(g, m) = 8.0; target's g_j ~ mean
    110.0 var 81.0, same size distribution. Per-unit R = g_bar/m_bar and
    its delta-method SE are the closed form in
    increment.estimation.engine.ratio_abs_diff_se, worked out by hand
    below rather than by calling it."""

    def _contrast(self):
        control = _cluster_arm(
            n=25, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _cluster_arm(
            n=25, g_mean=110.0, g_var=81.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
        )
        return SitewideContrast.from_clusters(control, target, cluster="store_id")

    @staticmethod
    def _r_se(g_bar, m_bar, var_g, var_m, cov_gm, n):
        r = g_bar / m_bar
        var_r = (1.0 / n) * (
            var_g / m_bar**2 - 2.0 * g_bar * cov_gm / m_bar**3 + g_bar**2 * var_m / m_bar**4
        )
        return r, math.sqrt(var_r)

    def test_hand_computed_normalization(self):
        contrast = self._contrast()

        r_c, se_c = self._r_se(100.0, 20.0, 64.0, 4.0, 8.0, 25)
        r_t, se_t = self._r_se(110.0, 20.0, 81.0, 4.0, 8.0, 25)
        assert r_c == pytest.approx(5.0)
        assert r_t == pytest.approx(5.5)
        assert se_c == pytest.approx(math.sqrt(0.0084))
        assert se_t == pytest.approx(math.sqrt(0.0114))

        assert contrast.control.mean == pytest.approx(r_c)
        assert contrast.target.mean == pytest.approx(r_t)
        assert contrast.control.var_mean == pytest.approx(se_c**2)
        assert contrast.target.var_mean == pytest.approx(se_t**2)
        # n_units = cluster count * mean cluster size = 25 * 20.
        assert contrast.control.n_units == pytest.approx(500.0)
        assert contrast.target.n_units == pytest.approx(500.0)
        assert contrast.n_clusters == 50
        assert contrast.dof == pytest.approx(48.0)
        assert contrast.target.own_dof == pytest.approx(24.0)

    @pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
    def test_hand_computed_impact(self):
        contrast = self._contrast()
        site_total_volume = 5_000.0

        result = sitewide_impact(contrast, site_total_volume=site_total_volume)

        delta = 0.5
        delta_se = math.sqrt(0.0084 + 0.0114)
        n_exp = 1000.0
        baseline = site_total_volume - delta * 500.0
        absolute = delta * n_exp
        relative = absolute / baseline
        # target/control own_dof are both 24 (K-1), but var_mean differs
        # (0.0114 vs 0.0084) so the two-arm Welch-Satterthwaite reduction
        # does not collapse to the pooled 48 despite the balanced counts.
        dof_abs = welch_satterthwaite_df(0.0114, 24.0, 0.0084, 24.0)
        crit_abs = float(_t.ppf(0.975, dof_abs))

        assert result.delta == pytest.approx(delta)
        assert result.delta_se == pytest.approx(delta_se)
        assert result.baseline_volume == pytest.approx(baseline)
        assert result.absolute_impact == pytest.approx(absolute)
        assert result.absolute_impact_se == pytest.approx(delta_se * n_exp)
        assert result.absolute_impact_lb == pytest.approx(absolute - crit_abs * delta_se * n_exp)
        assert result.absolute_impact_ub == pytest.approx(absolute + crit_abs * delta_se * n_exp)
        assert result.relative_impact == pytest.approx(relative)
        assert result.n_clusters == 50
        assert result.absolute_dof == pytest.approx(dof_abs)
        assert result.absolute_dof != pytest.approx(48.0)


class TestSitewideImpactRatioClustered:
    """Hand-computed cluster-grain ratio-metric contrast: 25 clusters/arm.
    Control cluster j has numerator total num_j ~ mean 100/var 64, its OWN
    denominator total den_j ~ mean 50/var 16, and size m_j ~ mean 20/var 4,
    with Cov(num, den)=20, Cov(num, m)=8, Cov(den, m)=6; target's numerator
    ~ mean 110/var 81, denominator ~ mean 52/var 18, Cov(num, den)=21,
    Cov(num, m)=8.5, Cov(den, m)=6.5, same size distribution."""

    def _contrast(self):
        control = _cluster_ratio_arm(
            n=25,
            num_mean=100.0,
            num_var=64.0,
            den_mean=50.0,
            den_var=16.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=20.0,
            cov_num_m=8.0,
            cov_den_m=6.0,
            group_id="C",
        )
        target = _cluster_ratio_arm(
            n=25,
            num_mean=110.0,
            num_var=81.0,
            den_mean=52.0,
            den_var=18.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=21.0,
            cov_num_m=8.5,
            cov_den_m=6.5,
            group_id="T",
        )
        return SitewideRatioContrast.from_clusters(control, target, cluster="store_id")

    @staticmethod
    def _r_se(num_bar, den_bar, var_num, var_den, cov_num_den, n):
        r = num_bar / den_bar
        var_r = (1.0 / n) * (
            var_num / den_bar**2
            - 2.0 * num_bar * cov_num_den / den_bar**3
            + num_bar**2 * var_den / den_bar**4
        )
        return r, math.sqrt(var_r)

    def test_hand_computed_normalization(self):
        contrast = self._contrast()

        # mean/var_mean: numerator total over cluster size (x family).
        r_c, se_c = self._r_se(100.0, 20.0, 64.0, 4.0, 8.0, 25)
        r_t, se_t = self._r_se(110.0, 20.0, 81.0, 4.0, 8.5, 25)
        assert contrast.control.mean == pytest.approx(r_c)
        assert contrast.target.mean == pytest.approx(r_t)
        assert contrast.control.var_mean == pytest.approx(se_c**2)
        assert contrast.target.var_mean == pytest.approx(se_t**2)

        # mean_den/var_mean_den: the metric's OWN denominator total over
        # cluster size.
        rd_c, sed_c = self._r_se(50.0, 20.0, 16.0, 4.0, 6.0, 25)
        rd_t, sed_t = self._r_se(52.0, 20.0, 18.0, 4.0, 6.5, 25)
        assert contrast.control.mean_den == pytest.approx(rd_c)
        assert contrast.target.mean_den == pytest.approx(rd_t)
        assert contrast.control.var_mean_den == pytest.approx(sed_c**2)
        assert contrast.target.var_mean_den == pytest.approx(sed_t**2)

        # cov_mean_den via ratio_pair_cov: (1/n)*(cov_num_den/m^2 -
        # r2*cov_num_m/m^2 - r1*cov_den_m/m^2 + r1*r2*var_m/m^2)
        def cov_mean_den(r1, r2, m_bar, cov_num_den, cov_num_m, cov_den_m, var_m, n):
            return (1.0 / n) * (
                cov_num_den / m_bar**2
                - r2 * cov_num_m / m_bar**2
                - r1 * cov_den_m / m_bar**2
                + r1 * r2 * var_m / m_bar**2
            )

        expected_cov_c = cov_mean_den(r_c, rd_c, 20.0, 20.0, 8.0, 6.0, 4.0, 25)
        expected_cov_t = cov_mean_den(r_t, rd_t, 20.0, 21.0, 8.5, 6.5, 4.0, 25)
        assert contrast.control.cov_mean_den == pytest.approx(expected_cov_c)
        assert contrast.target.cov_mean_den == pytest.approx(expected_cov_t)

        assert contrast.control.n_units == pytest.approx(500.0)
        assert contrast.target.n_units == pytest.approx(500.0)
        assert contrast.n_clusters == 50
        assert contrast.dof == pytest.approx(48.0)

    @pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
    def test_hand_computed_impact(self):
        contrast = self._contrast()
        site_total_numerator = 5_000.0
        site_total_denominator = 2_500.0

        result = sitewide_impact_ratio(
            contrast,
            site_total_numerator=site_total_numerator,
            site_total_denominator=site_total_denominator,
        )

        assert contrast.target.mean_den is not None and contrast.control.mean_den is not None
        num_bar = contrast.target.mean - contrast.control.mean
        den_bar = contrast.target.mean_den - contrast.control.mean_den
        n_exp = 1000.0

        n0 = site_total_numerator - 500.0 * num_bar
        d0 = site_total_denominator - 500.0 * den_bar
        n1 = n0 + n_exp * num_bar
        d1 = d0 + n_exp * den_bar
        assert result.baseline_numerator == pytest.approx(n0)
        assert result.baseline_denominator == pytest.approx(d0)
        assert result.shipped_numerator == pytest.approx(n1)
        assert result.shipped_denominator == pytest.approx(d1)

        impact = n1 / d1 - n0 / d0
        assert result.absolute_impact == pytest.approx(impact)
        assert result.absolute_impact != pytest.approx(0.0)
        assert result.n_clusters == 50
        # Satterthwaite reduction over control-anchored and target terms, each
        # with its own arm's clusters minus 1 (24, 24; see TestDegreesOfFreedom).
        # Unequal weights keep it below their sum, 48, which equals pooled
        # contrast.dof only because this fixture has no third arm.
        assert result.absolute_dof is not None
        assert result.absolute_dof == pytest.approx(46.14625043949708, rel=1e-9)
        assert result.absolute_dof < 48.0


class TestDegreesOfFreedom:
    """The dof/critical-value decision from the module docstring's
    "Degrees of freedom" section: Normal at iid grain, a two-arm
    Welch-Satterthwaite reduction for sum-metric absolute impact,
    Satterthwaite for relative impact (both families) and ratio-metric
    absolute impact."""

    def test_iid_path_uses_normal_reference_throughout(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
        target = _arm(n=120, mean=12.0, var=9.0, group_id="T")
        contrast = SitewideContrast.from_iid(control, target)

        result = sitewide_impact(contrast, site_total_volume=50_000.0)

        assert result.n_clusters is None
        assert result.absolute_dof is None
        assert result.relative_dof is None
        z = float(_norm.ppf(0.975))
        assert (result.absolute_impact_ub - result.absolute_impact) / result.absolute_impact_se == (
            pytest.approx(z)
        )
        assert (result.relative_impact_ub - result.relative_impact) / result.relative_impact_se == (
            pytest.approx(z)
        )

    @pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
    def test_sum_absolute_impact_uses_two_arm_welch_not_pooled_dof(self):
        """With a third (co-enrolled) cluster arm, the contrast-level dof
        (n_clusters_total - 2, over ALL arms) diverges from the two-arm
        Welch-Satterthwaite reduction over control/target's own variance
        and own_dof; the absolute interval must use the two-arm
        one - sum-metric absolute impact never reads the other arm."""
        control = _cluster_arm(
            n=25, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _cluster_arm(
            n=25, g_mean=110.0, g_var=81.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
        )
        other = _cluster_arm(
            n=15, g_mean=105.0, g_var=49.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="O"
        )
        contrast = SitewideContrast.from_clusters(control, target, other_arms=[other], cluster="s")
        assert contrast.n_clusters == 65
        assert contrast.dof == pytest.approx(63.0)
        assert contrast.target.own_dof == pytest.approx(24.0)
        assert contrast.control.own_dof == pytest.approx(24.0)

        result = sitewide_impact(contrast, site_total_volume=50_000.0)

        # TestSitewideImpactClustered's moments: var_mean 0.0114 (T) vs 0.0084
        # (C), own_dof 24 each. Unequal variance keeps Welch below 48; the third
        # arm never enters this two-arm reduction.
        dof_welch = welch_satterthwaite_df(0.0114, 24.0, 0.0084, 24.0)
        half_width = result.absolute_impact_ub - result.absolute_impact
        welch_crit = float(_t.ppf(0.975, dof_welch))
        pooled_crit = float(_t.ppf(0.975, 63))
        assert half_width == pytest.approx(welch_crit * result.absolute_impact_se)
        assert half_width != pytest.approx(pooled_crit * result.absolute_impact_se)
        # Reported absolute_dof matches the interval it was actually cut
        # at (the two-arm Welch reduction), not the pooled contrast-level
        # figure (63) nor the naive pairwise K_T+K_C-2 (48).
        assert result.n_clusters == 65
        assert result.absolute_dof == pytest.approx(dof_welch)
        assert result.absolute_dof != pytest.approx(48.0)
        assert result.absolute_dof != pytest.approx(63.0)

    @pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
    def test_relative_impact_uses_satterthwaite_reduction(self):
        control = _cluster_arm(
            n=25, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _cluster_arm(
            n=25, g_mean=110.0, g_var=81.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
        )
        other = _cluster_arm(
            n=15, g_mean=105.0, g_var=49.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="O"
        )
        contrast = SitewideContrast.from_clusters(control, target, other_arms=[other], cluster="s")

        result = sitewide_impact(contrast, site_total_volume=50_000.0)

        # The exposed field matches the interval it was actually cut at.
        assert result.relative_dof is not None
        assert (result.relative_impact_ub - result.relative_impact) / result.relative_impact_se == (
            pytest.approx(float(_t.ppf(0.975, result.relative_dof)))
        )
        # Genuinely a mixture over the enrolled arms' own cluster
        # degrees (24 + 24 + 14), never degenerate to the pairwise (48) or
        # pooled (63) figure, and never above the sum of its terms' own dofs.
        arm_dofs = [arm.own_dof for arm in (contrast.control, contrast.target, *contrast.others)]
        assert all(d is not None for d in arm_dofs)
        assert result.relative_dof <= math.fsum(d for d in arm_dofs if d is not None)
        assert result.relative_dof != pytest.approx(48.0)
        assert result.relative_dof != pytest.approx(63.0)

    @pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
    def test_ratio_absolute_impact_uses_satterthwaite_reduction(self):
        control = _cluster_ratio_arm(
            n=25,
            num_mean=100.0,
            num_var=64.0,
            den_mean=50.0,
            den_var=16.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=20.0,
            cov_num_m=8.0,
            cov_den_m=6.0,
            group_id="C",
        )
        target = _cluster_ratio_arm(
            n=25,
            num_mean=110.0,
            num_var=81.0,
            den_mean=52.0,
            den_var=18.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=21.0,
            cov_num_m=8.5,
            cov_den_m=6.5,
            group_id="T",
        )
        contrast = SitewideRatioContrast.from_clusters(control, target, cluster="s")

        result = sitewide_impact_ratio(
            contrast, site_total_numerator=5_000.0, site_total_denominator=2_500.0
        )

        # Two arms only, so the Satterthwaite mixture is over exactly their
        # two own cluster degrees.
        control_dof = contrast.control.own_dof
        target_dof = contrast.target.own_dof
        assert control_dof is not None and target_dof is not None
        assert control_dof == pytest.approx(24.0)
        assert target_dof == pytest.approx(24.0)
        assert result.n_clusters == 50

        # The exposed field matches the interval it was actually cut at.
        assert result.absolute_dof is not None
        half_width = result.absolute_impact_ub - result.absolute_impact
        assert half_width == pytest.approx(
            float(_t.ppf(0.975, result.absolute_dof)) * result.absolute_impact_se
        )
        # Genuinely a mixture, not degenerate to plain pairwise dof:
        # the control-anchored and target terms carry unequal variance
        # weights, and pairing a term with clusters its variance never
        # touched would push the reduction above its 24 + 24 bound.
        assert result.absolute_dof <= control_dof + target_dof
        assert result.absolute_dof != pytest.approx(48.0)

    @pytest.mark.filterwarnings("ignore:sitewide under a declared cluster:UserWarning")
    def test_ratio_absolute_dof_uses_each_arms_own_cluster_count(self):
        """Each Satterthwaite term pairs with its OWN arm's K_i - 1 (24 per
        arm here), never the pooled n_clusters - 2 or the pairwise
        K_i + K_C - 2 (both 48): those double-count cluster degrees the term's
        variance estimate never touched. A Welch-Satterthwaite dof can never
        exceed the sum of its terms' own dofs, so with two 25-cluster arms
        the reduction is bounded by 48 -- the pooled/pairwise pairing gave
        92.29, above that bound."""
        contrast = TestSitewideImpactRatioClustered()._contrast()

        result = sitewide_impact_ratio(
            contrast, site_total_numerator=5_000.0, site_total_denominator=2_500.0
        )

        assert contrast.control.own_dof == pytest.approx(24.0)
        assert contrast.target.own_dof == pytest.approx(24.0)
        assert result.absolute_dof is not None
        assert result.absolute_dof == pytest.approx(46.14625043949708, rel=1e-9)
        assert result.absolute_dof < 2 * (25 - 1)


class TestClusterBaselineCaveat:
    """The whole-site baseline nets out only ENROLLED units, which is exact
    under unit randomization but biased under cluster randomization (a
    treated cluster's non-enrolled units are still treated). The clustered
    path must surface that assumption; the iid path must not."""

    def _sum_cluster_contrast(self) -> SitewideContrast:
        # 25 clusters/arm -> 50 total, above check_total_clusters' warn floor
        # (40) so only THIS caveat can fire.
        control = _cluster_arm(
            n=25, g_mean=100.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="C"
        )
        target = _cluster_arm(
            n=25, g_mean=110.0, g_var=64.0, m_mean=20.0, m_var=4.0, cov_gm=8.0, group_id="T"
        )
        return SitewideContrast.from_clusters(control, target, cluster="store")

    def _ratio_cluster_contrast(self) -> SitewideRatioContrast:
        def arm(num_mean: float, group_id: str) -> ArmStats:
            return _cluster_ratio_arm(
                n=25,
                num_mean=num_mean,
                num_var=64.0,
                den_mean=50.0,
                den_var=25.0,
                m_mean=20.0,
                m_var=4.0,
                cov_num_den=10.0,
                cov_num_m=8.0,
                cov_den_m=5.0,
                group_id=group_id,
            )

        return SitewideRatioContrast.from_clusters(
            arm(100.0, "C"), arm(110.0, "T"), cluster="store"
        )

    def test_clustered_sum_path_warns(self):
        contrast = self._sum_cluster_contrast()
        with pytest.warns(IncrementWarning) as caught:
            sitewide_impact(contrast, site_total_volume=50_000.0)
        assert any(
            getattr(w.message, "code", None) == "estimation.sitewide.cluster_baseline_assumption"
            for w in caught
        )

    def test_clustered_ratio_path_warns(self):
        contrast = self._ratio_cluster_contrast()
        with pytest.warns(IncrementWarning) as caught:
            sitewide_impact_ratio(
                contrast, site_total_numerator=50_000.0, site_total_denominator=20_000.0
            )
        assert any(
            getattr(w.message, "code", None) == "estimation.sitewide.cluster_baseline_assumption"
            for w in caught
        )

    def test_iid_sum_path_does_not_warn(self):
        control = _arm(n=100, mean=10.0, var=4.0, group_id="C")
        target = _arm(n=120, mean=12.0, var=9.0, group_id="T")
        contrast = SitewideContrast.from_iid(control, target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sitewide_impact(contrast, site_total_volume=50_000.0)
        assert not any(
            getattr(w.message, "code", None) == "estimation.sitewide.cluster_baseline_assumption"
            for w in caught
        )

    def test_iid_ratio_path_does_not_warn(self):
        control = _ratio_arm(
            n=100, mean_num=10.0, var_num=4.0, mean_den=5.0, var_den=1.0, cov_yden=0.5, group_id="C"
        )
        target = _ratio_arm(
            n=100, mean_num=12.0, var_num=4.0, mean_den=5.0, var_den=1.0, cov_yden=0.5, group_id="T"
        )
        contrast = SitewideRatioContrast.from_iid(control, target)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sitewide_impact_ratio(
                contrast, site_total_numerator=50_000.0, site_total_denominator=20_000.0
            )
        assert not any(
            getattr(w.message, "code", None) == "estimation.sitewide.cluster_baseline_assumption"
            for w in caught
        )

    def test_caveat_names_the_biased_direction(self):
        # The message must state WHICH way the bias goes, so a reader knows
        # relative impact is a lower bound on the true value, not just "uncertain".
        assert "biased upward" in _CLUSTER_BASELINE_CAVEAT
        assert "lower bound on the true value" in _CLUSTER_BASELINE_CAVEAT


class TestIidRatioRefusesEveryClusterGrainRole:
    """`cluster_size` and `uptake_total` are both cluster-grain declarations
    whose per-cluster totals would be read here as unit-grain."""

    @pytest.mark.parametrize("x_role", ["cluster_size", "uptake_total"])
    def test_a_cluster_grain_x_role_refuses_from_iid(self, x_role):
        # Declaration alone: a row copied past validation carries no x family.
        control, target, _ = _multi_arm_set()
        mislabelled = target.model_copy(update={"x_role": x_role})
        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast.from_iid(control, mislabelled)
        assert exc_info.value.code == "estimation.sitewide.sitewide_ratio.metric_arm_declares"
        assert exc_info.value.context["x_role"] == x_role

    @pytest.mark.parametrize("x_role", ["cluster_size", "uptake_total"])
    def test_a_cluster_grain_x_family_refuses_from_iid(self, x_role):
        # Declaration and family together: the row a clustered source emits.
        control, _, _ = _multi_arm_ratio_set()
        clustered = _cluster_ratio_arm(
            n=25,
            num_mean=100.0,
            num_var=64.0,
            den_mean=50.0,
            den_var=16.0,
            m_mean=20.0,
            m_var=4.0,
            cov_num_den=20.0,
            cov_num_m=8.0,
            cov_den_m=6.0,
            group_id="treatment_a",
        )
        mislabelled = ArmStats.model_validate(clustered.model_dump() | {"x_role": x_role})
        with pytest.raises(InvalidRequestError) as exc_info:
            SitewideRatioContrast.from_iid(control, mislabelled)
        assert exc_info.value.code == "estimation.sitewide.sitewide_ratio.metric_arm_declares"
        assert exc_info.value.context["x_role"] == x_role


def _rg_arm(n, mean, var, group_id):
    sum_y = mean * n
    sum_y2 = var * (n - 1) + sum_y**2 / n
    return ArmStats.from_raw_sums(
        study_id="s", metric="rev", group_id=group_id, n=n, sum_y=sum_y, sum_y2=sum_y2
    )


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.sitewide.control_carries_metric",
            lambda: SitewideContrast.from_iid(
                _rg_arm(10, 1.0, 1.0, "C").model_copy(update={"metric": "a"}),
                _rg_arm(10, 1.0, 1.0, "T").model_copy(update={"metric": "b"}),
            ),
        ),  # estimation/sitewide.py::_checked_arms
        (
            "estimation.sitewide.alpha",
            lambda: sitewide_impact(
                SitewideContrast.from_iid(_rg_arm(10, 1.0, 1.0, "C"), _rg_arm(10, 2.0, 1.0, "T")),
                site_total_volume=100.0,
                alpha=0.0,
            ),
        ),  # estimation/sitewide.py::_validate_alpha (sitewide_impact)
        (
            "estimation.armstats.arm_stats.cross_field_finite",
            lambda: sitewide_impact(
                SitewideContrast.from_iid(_rg_arm(10, 1.0, 1.0, "C"), _rg_arm(10, 2.0, 1.0, "T")),
                site_total_volume=float("nan"),
            ),
        ),  # estimation/sitewide.py::_validate_site_total (sitewide_impact)
        (
            "estimation.sitewide.counterfactual_baseline_volume",
            lambda: sitewide_impact(
                SitewideContrast.from_iid(_rg_arm(10, 1.0, 1.0, "C"), _rg_arm(10, 2.0, 1.0, "T")),
                site_total_volume=1.0,
            ),
        ),  # estimation/sitewide.py::sitewide_impact
        (
            "estimation.sitewide.counterfactual_baseline_ratio",
            lambda: sitewide_impact_ratio(
                SitewideRatioContrast.from_iid(
                    _ratio_arm(
                        n=10,
                        mean_num=1.0,
                        var_num=1.0,
                        mean_den=1.0,
                        var_den=1.0,
                        cov_yden=0.1,
                        group_id="C",
                    ),
                    _ratio_arm(
                        n=10,
                        mean_num=2.0,
                        var_num=1.0,
                        mean_den=1.0,
                        var_den=1.0,
                        cov_yden=0.1,
                        group_id="T",
                    ),
                ),
                site_total_numerator=1.0,
                site_total_denominator=1.0,
            ),
        ),  # estimation/sitewide.py::sitewide_impact_ratio
    ],
)
def test_sitewide_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
