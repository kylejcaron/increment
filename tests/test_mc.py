"""Unit tests for the shared Monte-Carlo coverage/MCSE helpers."""

from __future__ import annotations

import math

import pytest

from tests.mc import (
    Coverage,
    CoverageSet,
    binomial_error_upper_bound,
    coverage_lower_bound,
    family_eta,
    kl_chernoff_upper_bound,
    mcse,
    nominal_band,
    replicate,
    replicate_many,
    scientific_delta,
)


class TestMcse:
    def test_matches_textbook_binomial_se(self):
        assert mcse(0.95, 400) == pytest.approx(math.sqrt(0.95 * 0.05 / 400))

    def test_zero_and_one_have_zero_se(self):
        assert mcse(0.0, 100) == 0.0
        assert mcse(1.0, 100) == 0.0

    def test_rejects_non_positive_reps(self):
        with pytest.raises(ValueError):
            mcse(0.5, 0)

    def test_rejects_out_of_range_p(self):
        with pytest.raises(ValueError):
            mcse(1.5, 100)


class TestNominalBand:
    def test_band_is_symmetric_around_nominal(self):
        lo, hi = nominal_band(0.95, 400, k=2.0)
        se = mcse(0.95, 400)
        assert lo == pytest.approx(0.95 - 2 * se)
        assert hi == pytest.approx(0.95 + 2 * se)

    def test_clips_to_unit_interval(self):
        lo, hi = nominal_band(0.5, 4, k=10.0)
        assert lo == 0.0
        assert hi == 1.0


class TestCoverage:
    def test_rate_is_hits_over_reps(self):
        cov = Coverage()
        for hit in (True, True, False, True):
            cov.record(hit)
        assert cov.reps == 4
        assert cov.hits == 3
        assert cov.rate == pytest.approx(0.75)

    def test_mcse_matches_rate(self):
        cov = Coverage()
        for hit in (True, False, True, True):
            cov.record(hit)
        assert cov.mcse == pytest.approx(mcse(cov.rate, cov.reps))

    def test_rate_before_any_record_raises(self):
        with pytest.raises(ValueError):
            _ = Coverage().rate

    def test_band_brackets_rate(self):
        cov = Coverage()
        for hit in (True,) * 8 + (False,) * 2:
            cov.record(hit)
        lo, hi = cov.band(k=2.0)
        assert lo <= cov.rate <= hi


class TestCoverageSet:
    def test_record_updates_every_named_accumulator(self):
        covset = CoverageSet()
        covset.record(clustered=True, iid=False)
        covset.record(clustered=True, iid=True)
        clustered, iid = covset.rates("clustered", "iid")
        assert clustered == pytest.approx(1.0)
        assert iid == pytest.approx(0.5)

    def test_independent_names_do_not_interfere(self):
        covset = CoverageSet()
        covset["a"].record(True)
        covset["b"].record(False)
        assert covset["a"].reps == 1
        assert covset["b"].reps == 1


class TestReplicate:
    def test_replicate_runs_reps_trials(self):
        cov = replicate(10, lambda i: i % 3 == 0)
        assert cov.reps == 10
        assert cov.hits == 4  # i in {0, 3, 6, 9}

    def test_replicate_many_requires_every_name_reported(self):
        with pytest.raises(ValueError):
            replicate_many(3, ["a", "b"], lambda i: {"a": True})

    def test_replicate_many_records_each_name(self):
        covset = replicate_many(5, ["a", "b"], lambda i: {"a": i % 2 == 0, "b": True})
        a, b = covset.rates("a", "b")
        assert a == pytest.approx(0.6)  # i in {0, 2, 4}
        assert b == pytest.approx(1.0)


class TestBinomialErrorUpperBound:
    def test_matches_scipy_beta_isf_directly(self):
        from scipy.stats import beta

        k, reps, eta = 5, 100, 0.025
        assert binomial_error_upper_bound(k, reps, eta) == pytest.approx(
            beta.isf(eta, k + 1, reps - k)
        )

    def test_zero_errors_is_strictly_positive_and_less_than_one(self):
        bound = binomial_error_upper_bound(0, 100, 0.025)
        assert 0.0 < bound < 1.0

    def test_all_errors_returns_exactly_one(self):
        assert binomial_error_upper_bound(10, 10, 0.025) == 1.0

    def test_monotone_increasing_in_k(self):
        bounds = [binomial_error_upper_bound(k, 50, 0.025) for k in range(0, 51, 10)]
        assert bounds == sorted(bounds)

    def test_tighter_eta_gives_a_larger_bound(self):
        loose = binomial_error_upper_bound(3, 100, 0.1)
        tight = binomial_error_upper_bound(3, 100, 0.001)
        assert tight > loose

    def test_rejects_k_out_of_range(self):
        with pytest.raises(ValueError):
            binomial_error_upper_bound(11, 10, 0.025)

    def test_rejects_out_of_range_eta(self):
        with pytest.raises(ValueError):
            binomial_error_upper_bound(1, 10, 1.5)


class TestCoverageLowerBound:
    def test_inverts_error_bound_on_misses(self):
        hits, reps, eta = 95, 100, 0.025
        expected = 1.0 - binomial_error_upper_bound(reps - hits, reps, eta)
        assert coverage_lower_bound(hits, reps, eta) == pytest.approx(expected)

    def test_all_hits_is_strictly_positive(self):
        assert coverage_lower_bound(100, 100, 0.025) > 0.0

    def test_zero_hits_returns_exactly_zero(self):
        assert coverage_lower_bound(0, 100, 0.025) == 0.0


class TestScientificDelta:
    def test_saturates_at_005_for_q_at_or_above_05(self):
        assert scientific_delta(0.05) == pytest.approx(0.005)
        assert scientific_delta(0.5) == pytest.approx(0.005)

    def test_scales_down_for_small_q(self):
        assert scientific_delta(0.025) == pytest.approx(0.0025)

    def test_rejects_out_of_range_q(self):
        with pytest.raises(ValueError):
            scientific_delta(1.5)


class TestFamilyEta:
    def test_splits_family_alpha_across_m_two_sided_bounds(self):
        assert family_eta(0.01, 100) == pytest.approx(0.01 / 200)

    def test_more_gated_statistics_tightens_eta(self):
        assert family_eta(0.01, 200) < family_eta(0.01, 100)

    def test_rejects_non_positive_m(self):
        with pytest.raises(ValueError):
            family_eta(0.01, 0)


class TestKlChernoffUpperBound:
    def test_bound_is_at_least_the_observed_mean(self):
        assert kl_chernoff_upper_bound(0.1, 200, 0.025) >= 0.1

    def test_zero_observed_mean_boundary_case(self):
        bound = kl_chernoff_upper_bound(0.0, 200, 0.025)
        expected = 1.0 - math.exp(-math.log(1.0 / 0.025) / 200)
        assert bound == pytest.approx(expected, abs=1e-9)

    def test_all_ones_observed_mean_returns_exactly_one(self):
        assert kl_chernoff_upper_bound(1.0, 200, 0.025) == 1.0

    def test_tighter_eta_gives_a_larger_bound(self):
        loose = kl_chernoff_upper_bound(0.2, 200, 0.1)
        tight = kl_chernoff_upper_bound(0.2, 200, 0.001)
        assert tight > loose

    def test_more_replications_gives_a_smaller_bound(self):
        few = kl_chernoff_upper_bound(0.2, 50, 0.025)
        many = kl_chernoff_upper_bound(0.2, 5000, 0.025)
        assert many < few

    def test_extreme_positive_eta_does_not_overflow(self):
        eta = math.nextafter(0.0, 1.0)
        bound = kl_chernoff_upper_bound(0.2, 100, eta)
        assert math.isfinite(bound)
        assert 0.2 <= bound <= 1.0

    def test_rejects_out_of_range_x(self):
        with pytest.raises(ValueError):
            kl_chernoff_upper_bound(1.5, 200, 0.025)
