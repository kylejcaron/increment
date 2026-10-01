"""Tests for ScoreStats, infer_ate, and LiftEstimate.population."""

import math

import pytest

from increment.errors import InvalidRequestError
from increment.estimation.armstats import ScoreStats
from increment.estimation.inference import Normal, infer_ate
from increment.estimation.results import Estimate, LiftEstimate


def test_scorestats_se_iptw_normalizer():
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
    assert s.se() == pytest.approx(math.sqrt(11.9212) / 6)


def test_scorestats_se_clustered_passes_through_resolved_variance():
    """The clustered branch must take ``cluster_variance`` as an
    already-fully-resolved variance and apply NO internal finite-sample
    multiplier: se() == sqrt(cluster_variance) / normalizer, verbatim."""
    s = ScoreStats(
        metric="m",
        contrast="T",
        n=6,
        sum_psi=0.0,
        sum_psi2=11.9212,
        cluster_variance=9.0,
        n_clusters=4,
    )
    assert s.se() == pytest.approx(math.sqrt(9.0) / 6)
    # A caller who wants the old CR1 finite-sample inflation must pre-multiply
    # before constructing ScoreStats -- se() itself no longer applies it.
    inflated = ScoreStats(
        metric="m",
        contrast="T",
        n=6,
        sum_psi=0.0,
        sum_psi2=11.9212,
        cluster_variance=9.0 * (4 / 3),
        n_clusters=4,
    )
    assert inflated.se() == pytest.approx(math.sqrt(9.0 * (4 / 3)) / 6)


def test_scorestats_se_dml_normalizer():
    s = ScoreStats(metric="m", contrast="T", n=100, sum_psi=0.0, sum_psi2=25.0, sum_d_tilde2=20.0)
    assert s.se() == pytest.approx(math.sqrt(25.0) / 20.0)


def test_scorestats_se_rejects_nonpositive_normalizer():
    s = ScoreStats(metric="m", contrast="T", n=5, sum_psi=0.0, sum_psi2=1.0, sum_d_tilde2=0.0)
    with pytest.raises(ValueError):
        s.se()


def test_scorestats_se_centers_nonzero_sum_psi():
    """se() must use the CENTERED second moment: psi = [0, 1, 1, 2] has
    sum_psi=4, sum_psi2=6, so the centered sum of squares is
    6 - 4^2/4 = 2. The uncentered form sqrt(6)/4 would silently inflate
    the SE by the n*(mean psi)^2 term for any registration whose scores
    are not mean-zero (e.g. a Horvitz-Thompson contrast). For the
    mean-zero scores every current registration produces, centering is
    an exact no-op (the tests above pin that with sum_psi=0)."""
    s = ScoreStats(metric="m", contrast="T", n=4, sum_psi=4.0, sum_psi2=6.0)
    assert s.se() == pytest.approx(math.sqrt(2.0) / 4)


def test_infer_ate_flat_prior_is_normal_interval():
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
    est = infer_ate(
        metric="m", group_id="T", method="iptw", method_role="decision", point=15 / 13, scores=s
    )
    se = math.sqrt(11.9212) / 6
    z = 1.959963984540054
    assert est.require_lift().value == pytest.approx(15 / 13, rel=1e-3)
    assert est.require_lift().lb == pytest.approx(15 / 13 - z * se, rel=1e-3)
    assert est.population is None


def test_infer_ate_negative_through_minus_one_representable():
    # A catastrophic lift (< -1) must not raise - the additive scale allows it.
    s = ScoreStats(metric="m", contrast="T", n=50, sum_psi=0.0, sum_psi2=2.0)
    est = infer_ate(
        metric="m", group_id="T", method="iptw", method_role="decision", point=-1.4, scores=s
    )
    assert est.require_lift().value == pytest.approx(-1.4, rel=1e-3)


def test_infer_ate_carries_population():
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=1.0)
    est = infer_ate(
        metric="m",
        group_id="T",
        method="iptw",
        method_role="decision",
        point=0.1,
        scores=s,
        population="overlap e in [0.01, 0.99] (5 of 6 units)",
    )
    assert est.population is not None
    assert est.population.startswith("overlap")


def test_infer_ate_raises_on_degenerate_zero_variance():
    """se() == 0 when sum_psi2 == 0 - no uncertainty to propagate."""
    s = ScoreStats(metric="m", contrast="T", n=10, sum_psi=0.0, sum_psi2=0.0)
    with pytest.raises(InvalidRequestError) as raised:
        infer_ate(
            metric="m", group_id="T", method="iptw", method_role="decision", point=0.1, scores=s
        )
    assert raised.value.code == "estimation.inference.degenerate_data_zero"


def test_infer_ate_with_informative_prior_shrinks_toward_it():
    """A non-flat prior blends into the posterior mean - value != point."""
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
    flat = infer_ate(
        metric="m", group_id="T", method="iptw", method_role="decision", point=15 / 13, scores=s
    )
    informative = infer_ate(
        metric="m",
        group_id="T",
        method="iptw",
        method_role="decision",
        point=15 / 13,
        scores=s,
        prior=Normal(mu=0.0, sigma=0.01),
    )
    assert informative.require_lift().value != pytest.approx(flat.require_lift().value)
    # Strongly informative prior at 0 pulls the posterior mean toward 0.
    assert abs(informative.require_lift().value) < abs(flat.require_lift().value)


def test_lift_estimate_population_default_none():
    # Existing constructions (no population kwarg) stay valid.
    e = LiftEstimate(
        metric="m",
        group_id="T",
        method="unadjusted",
        method_role="decision",
        lift=Estimate(value=0.1),
    )
    assert e.population is None


def test_infer_ate_null_lift_default_reproduces_today():
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
    est = infer_ate(
        metric="m", group_id="T", method="iptw", method_role="decision", point=15 / 13, scores=s
    )
    assert est.null_lift == 0.0
    assert est.preferred_direction is None


def test_infer_ate_null_lift_stamped_without_changing_interval():
    """Additive scale: no log1p floor - null_lift is stamped verbatim,
    unrestricted, and the interval is unaffected."""
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
    baseline = infer_ate(
        metric="m", group_id="T", method="iptw", method_role="decision", point=15 / 13, scores=s
    )
    shifted = infer_ate(
        metric="m",
        group_id="T",
        method="iptw",
        method_role="decision",
        point=15 / 13,
        scores=s,
        null_lift=-0.01,
        preferred_direction="increase",
    )
    assert shifted.require_lift().value == pytest.approx(baseline.require_lift().value)
    assert shifted.require_lift().lb == pytest.approx(baseline.require_lift().lb)
    assert shifted.require_lift().ub == pytest.approx(baseline.require_lift().ub)
    assert shifted.null_lift == -0.01
    assert shifted.preferred_direction == "increase"


def test_infer_ate_prob_favorable_via_prob_beyond():
    s = ScoreStats(metric="m", contrast="T", n=6, sum_psi=0.0, sum_psi2=11.9212)
    est = infer_ate(
        metric="m",
        group_id="T",
        method="iptw",
        method_role="decision",
        point=0.05,
        scores=s,
        null_lift=-0.01,
        preferred_direction="increase",
    )
    assert est.prob_favorable() == pytest.approx(est.prob_beyond(-0.01))


@pytest.mark.parametrize(
    ("score_scale", "sum_psi2", "normalizer", "expected"),
    [
        (1e308, 4.0, 2.0, 1e308),
        (1e-200, 4.0, 2.0, 1e-200),
        (1e308, 1e-200, 1e308, 1e-100),
    ],
)
def test_scaled_score_standard_error_avoids_intermediate_overflow_and_underflow(
    score_scale, sum_psi2, normalizer, expected
):
    scores = ScoreStats(
        metric="m",
        contrast="T",
        n=8,
        sum_psi=0.0,
        sum_psi2=sum_psi2,
        sum_d_tilde2=normalizer,
        score_scale=score_scale,
    )
    assert scores.se() == pytest.approx(expected, rel=1e-14, abs=0)


def test_cluster_score_scale_preserves_cancelled_member_contributions():
    scores = ScoreStats(
        metric="m",
        contrast="T",
        n=8,
        sum_psi=0.0,
        sum_psi2=4.0,
        score_scale=2e200,
        cluster_variance=8 / 3,
        n_clusters=4,
        cluster_score_scale=4.0,
    )
    assert scores.se() == pytest.approx(math.sqrt(2 / 3), rel=1e-14)


def test_cluster_correction_preserves_near_maximum_finite_variance():
    import numpy as np

    from increment.estimation._adjust.common import ClusterSupport, _score_stats

    a = float(2**510)
    scores = _score_stats(
        np.repeat([a / 2, -a / 2, a / 2, -a / 2], 2),
        metric="m",
        contrast="T",
        support=ClusterSupport(slice(None), np.repeat(np.arange(4), 2), 4),
    )
    assert scores.se() == pytest.approx(float(2**509) / math.sqrt(3), rel=1e-14)
