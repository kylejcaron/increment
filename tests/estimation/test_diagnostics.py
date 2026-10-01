"""Tests for diagnostics (sample_ratio_mismatch, allocation_posterior_bands)."""

import math

import pandas as pd
import pyarrow as pa
import pytest
from pydantic import ValidationError
from scipy.stats import beta as beta_dist

from increment.errors import InvalidRequestError
from increment.estimation.diagnostics import (
    AllocationBand,
    NotApplicable,
    SRMResult,
    allocation_posterior_bands,
    resolve_srm_expected,
    sample_ratio_mismatch,
)


class TestSampleRatioMismatch:
    def test_always_valid_requires_static_expected_allocation(self):
        """A 3-arm test cannot infer 50/50 from its first two observed arms."""
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"control": 50, "treatment": 50})
        assert exc_info.value.code == "estimation.diagnostics.predeclared_allocation_support"

    def test_no_srm_equal_expected(self):
        """With balanced counts and equal expected ratio, no SRM detected."""
        counts = {"control": 5000, "treatment": 5000}
        result = sample_ratio_mismatch(counts, expected={"control": 0.5, "treatment": 0.5})
        assert isinstance(result, SRMResult)
        assert result.fixed_p_value > 0.05
        assert not result.is_srm

    def test_srm_detected(self):
        """With heavily imbalanced counts, SRM is detected."""
        counts = {"control": 9000, "treatment": 1000}
        result = sample_ratio_mismatch(counts, expected={"control": 0.5, "treatment": 0.5})
        assert result.fixed_p_value < 0.05
        assert result.is_srm

    def test_expected_proportions(self):
        """Expected proportions can be specified."""
        counts = {"A": 300, "B": 700}
        expected = {"A": 0.4, "B": 0.6}
        result = sample_ratio_mismatch(counts, expected=expected)
        assert result.fixed_p_value < 0.05

    @pytest.mark.parametrize("inference", ["always_valid", "fixed"])
    def test_explicit_expected_zero_fills_unobserved_arm_and_alarms(self, inference: str):
        """A declared but unseen arm remains visible to both SRM statistics."""
        counts = {"control": 14}
        expected = {"control": 0.5, "treatment": 0.5}

        result = sample_ratio_mismatch(counts, expected=expected, inference=inference)  # ty: ignore[invalid-argument-type]
        assert result.inference == inference

        assert result.observed == {"control": 14, "treatment": 0}
        assert result.expected == expected
        assert result.is_srm
        assert counts == {"control": 14}

    def test_explicit_expected_rejects_observed_arm_absent_from_allocation(self):
        """An observed arm outside declared allocation remains a strict error."""
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch(
                {"control": 14, "holdout": 1},
                expected={"control": 0.5, "treatment": 0.5},
            )
        assert exc_info.value.code == "estimation.diagnostics.expected_keys_do"

    @pytest.mark.parametrize("label", ["(unassigned)", "(mixed assignment)"])
    def test_explicit_expected_rejects_accounting_labels(self, label: str):
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch(
                {"control": 14, "treatment": 14},
                expected={"control": 0.5, "treatment": 0.5, label: 0.1},
            )
        assert exc_info.value.code == "estimation.diagnostics.expected_allocation_contain"

    def test_custom_alpha(self):
        """Alpha parameter controls the detection threshold."""
        counts = {"control": 480, "treatment": 520}
        result = sample_ratio_mismatch(
            counts, expected={"control": 0.5, "treatment": 0.5}, alpha=0.01
        )
        assert not result.is_srm

    def test_three_groups(self):
        """Works with >2 groups."""
        counts = {"A": 300, "B": 300, "C": 400}
        result = sample_ratio_mismatch(counts, expected={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        assert result.df == 2
        assert result.fixed_p_value > 0

    def test_srm_result_fields(self):
        """SRMResult contains expected fields."""
        counts = {"control": 500, "treatment": 500}
        result = sample_ratio_mismatch(
            counts, expected={"control": 0.5, "treatment": 0.5}, alpha=0.05
        )
        assert isinstance(result.chi2_stat, float)
        assert isinstance(result.fixed_p_value, float)
        assert isinstance(result.df, int)
        assert isinstance(result.is_srm, bool)
        assert result.observed == counts
        assert result.alpha == 0.05
        for k in counts:
            assert result.expected[k] == pytest.approx(0.5)

    def test_zero_expected_share_refused(self):
        """Five units in an arm whose target share is zero is the strongest
        possible mismatch - the chi-square statistic is undefined there
        (the infinite term must not be silently dropped as a p=0.874 MISS)."""
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"holdout": 5, "main": 995}, expected={"holdout": 0, "main": 1})
        assert exc_info.value.code == "estimation.diagnostics.every_expected_share_positive"

    def test_negative_expected_share_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"a": 100, "b": 900}, expected={"a": -1, "b": 2})
        assert exc_info.value.code == "estimation.diagnostics.every_expected_share_positive"

    def test_all_zero_expected_refused(self):
        """All-zero expected shares are a domain error, not a ZeroDivisionError."""
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"a": 100, "b": 900}, expected={"a": 0, "b": 0})
        assert exc_info.value.code == "estimation.diagnostics.every_expected_share_positive"

    @pytest.mark.parametrize("share", [math.nan, math.inf, -math.inf])
    def test_nonfinite_expected_share_refused(self, share: float):
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"a": 100, "b": 900}, expected={"a": share, "b": 1.0})
        assert exc_info.value.code == "estimation.diagnostics.every_expected_share_finite"

    def test_normalization_underflow_expected_share_refused(self):
        """Finite positive expected shares that underflow on normalization are invalid."""
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch(
                {"rare": 1, "common": 999}, expected={"rare": 5e-324, "common": 1e308}
            )
        assert exc_info.value.code == "estimation.diagnostics.expected_share_normalization"

    def test_near_one_expected_is_normalized_for_log_e_process(self):
        result = sample_ratio_mismatch(
            {"control": 1, "treatment": 1},
            expected={"control": 0.50000000005, "treatment": 0.50000000005},
        )

        assert result.expected == {"control": 0.5, "treatment": 0.5}
        assert result.log_e_value == pytest.approx(math.log(2 / 3), abs=1e-12)

    def test_fixed_alarm_can_precede_always_valid_alarm(self):
        counts = {"control": 531, "treatment": 469}
        expected = {"control": 0.5, "treatment": 0.5}
        fixed = sample_ratio_mismatch(counts, expected=expected, alpha=0.05, inference="fixed")
        always_valid = sample_ratio_mismatch(
            counts, expected=expected, alpha=0.05, inference="always_valid"
        )

        assert fixed.fixed_p_value < 0.05
        assert isinstance(fixed.log_e_value, float)
        assert fixed.is_srm is True
        always_valid_log_e_value = always_valid.log_e_value
        assert always_valid_log_e_value is not None
        assert always_valid_log_e_value < -math.log(0.05)
        assert always_valid.is_srm is False

    def test_negative_counts_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"a": -100, "b": 1100})
        assert exc_info.value.code == "estimation.diagnostics.counts_non_negative"

    def test_extreme_pvalue_does_not_underflow_to_zero(self):
        """A mild real SRM (52/48 at n=100k, chi2=160) has p ~ 1.1e-36;
        computing it as 1-cdf underflows to exactly 0.0, destroying the
        severity signal. The survival function keeps full resolution."""
        result = sample_ratio_mismatch(
            {"control": 52_000, "treatment": 48_000},
            expected={"control": 0.5, "treatment": 0.5},
        )
        assert result.chi2_stat == pytest.approx(160.0)
        assert result.is_srm
        assert result.fixed_p_value > 0.0
        assert result.fixed_p_value == pytest.approx(1.1e-36, rel=0.1)

    def test_mixed_assignment_key_is_accounting_not_an_arm(self):
        """A count under "(mixed assignment)" (units first_exposures drops
        for appearing in more than one arm) is separated out onto
        mixed_assignment_units: no chi-square degree of freedom, absent
        from observed/expected - same contract as "(unassigned)"."""
        from increment.sources import MIXED_ASSIGNMENT_LABEL, UNASSIGNED_LABEL

        counts = {
            "control": 500,
            "treatment": 500,
            MIXED_ASSIGNMENT_LABEL: 7,
            UNASSIGNED_LABEL: 3,
        }
        result = sample_ratio_mismatch(counts, expected={"control": 0.5, "treatment": 0.5})
        assert result.mixed_assignment_units == 7
        assert result.unassigned_units == 3
        assert result.df == 1  # two arms only
        assert set(result.observed) == {"control", "treatment"}
        assert set(result.expected) == {"control", "treatment"}
        assert not result.is_srm

    def test_alpha_outside_unit_interval_refused(self):
        """alpha=0 never flags (even p=0) and alpha>1 vacuously flags -
        both are silent nonsense, so the domain is validated."""
        counts = {"control": 500, "treatment": 500}
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch(counts, alpha=0.0)
        assert exc_info.value.code == "estimation.diagnostics.alpha"
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch(counts, alpha=1.5)
        assert exc_info.value.code == "estimation.diagnostics.alpha"

    def test_fractional_counts_refused_by_name(self):
        """Fractional float counts previously failed LATE (a pydantic
        ValidationError on SRMResult.observed, after the statistic was
        already computed); they must be refused at entry, by name.
        Integral floats (500.0) stay accepted - a common warehouse
        artifact, losslessly coerced."""
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch({"a": 500.5, "b": 499.5})  # ty: ignore[invalid-argument-type]
        assert exc_info.value.code == "estimation.diagnostics.counts_whole_unit"
        result = sample_ratio_mismatch({"a": 500.0, "b": 500.0}, expected={"a": 0.5, "b": 0.5})  # ty: ignore[invalid-argument-type]
        assert result.observed == {"a": 500, "b": 500}

    def test_default_is_anytime_valid_with_point_one_percent_lifetime_alpha(self):
        result = sample_ratio_mismatch(
            {"control": 500, "treatment": 500},
            expected={"control": 0.5, "treatment": 0.5},
        )

        assert result.inference == "always_valid"
        assert result.alpha == 0.001
        log_e_value = result.log_e_value
        assert log_e_value is not None
        assert log_e_value < -math.log(result.alpha)
        assert result.is_srm is False

    def test_anytime_threshold_avoids_reciprocal_overflow(self):
        alpha = math.nextafter(0.0, 1.0)

        result = sample_ratio_mismatch(
            {"control": 1100},
            expected={"control": 0.5, "treatment": 0.5},
            alpha=alpha,
        )

        assert result.log_e_value is not None
        assert result.log_e_value >= -math.log(alpha)
        assert result.is_srm is True

    def test_fixed_inference_without_expected_has_no_e_value(self):
        """Observed-arm equal shares are a Pearson fallback, not an e-process null."""
        result = sample_ratio_mismatch(
            {"control": 480, "treatment": 520},
            alpha=0.05,
            inference="fixed",
        )

        assert result.inference == "fixed"
        assert result.chi2_stat == pytest.approx(1.6)
        assert result.fixed_p_value == pytest.approx(0.2059032107)
        assert result.log_e_value is None
        assert result.is_srm is (result.fixed_p_value < result.alpha)

    def test_unknown_inference_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            sample_ratio_mismatch(
                {"control": 500, "treatment": 500},
                inference="peek",  # ty: ignore[invalid-argument-type]
            )
        assert exc_info.value.code == "estimation.diagnostics.inference_always_valid"

    def test_three_arm_log_e_value_matches_exact_uniform_dirichlet_ratio(self):
        result = sample_ratio_mismatch(
            {"A": 2, "B": 1, "C": 0},
            expected={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        )

        # B((3, 2, 1)) / B((1, 1, 1)) divided by (1/3)^3 is 0.9.
        assert result.log_e_value == pytest.approx(math.log(0.9), rel=1e-12)

    def test_anytime_alarm_probability_is_bounded_over_repeated_looks(self):
        # Exact dynamic enumeration under fair assignment. Each state branches
        # with null probability 1/2; paths stop contributing after first alarm.
        alive = {(0, 0): 1.0}
        crossed = 0.0
        for _ in range(12):
            next_alive: dict[tuple[int, int], float] = {}
            for (control, treatment), mass in alive.items():
                for next_control, next_treatment in (
                    (control + 1, treatment),
                    (control, treatment + 1),
                ):
                    branch_mass = mass * 0.5
                    result = sample_ratio_mismatch(
                        {"control": next_control, "treatment": next_treatment},
                        expected={"control": 0.5, "treatment": 0.5},
                        alpha=0.05,
                    )
                    if result.is_srm:
                        crossed += branch_mass
                    else:
                        state = (next_control, next_treatment)
                        next_alive[state] = next_alive.get(state, 0.0) + branch_mass
            alive = next_alive

        assert crossed == pytest.approx(0.01171875)
        assert crossed <= 0.05

    def test_sparse_many_arm_fixed_allocation_flags_low_expected_count(self):
        """Five arms each expecting ~3 units fall under Cochran's rule of
        thumb - the fixed chi-square p-value is not trustworthy there."""
        counts = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        result = sample_ratio_mismatch(counts, inference="fixed")

        assert result.min_expected_count == pytest.approx(3.0)
        assert result.low_expected_count is True

    def test_healthy_large_allocation_does_not_flag_low_expected_count(self):
        """Thousands of units per arm are nowhere near the Cochran floor."""
        counts = {"control": 5000, "treatment": 5000}
        result = sample_ratio_mismatch(
            counts, expected={"control": 0.5, "treatment": 0.5}, inference="fixed"
        )

        assert result.min_expected_count == pytest.approx(5000.0)
        assert result.low_expected_count is False

    def test_min_expected_count_matches_hand_computed_minimum(self):
        """min_expected_count is exactly min over arms of expected[k] * total."""
        counts = {"A": 1, "B": 99}
        expected = {"A": 0.02, "B": 0.98}
        result = sample_ratio_mismatch(counts, expected=expected, inference="fixed")

        total = sum(counts.values())
        hand_computed = min(result.expected[k] * total for k in counts)
        assert result.min_expected_count == pytest.approx(hand_computed)
        assert result.min_expected_count == pytest.approx(2.0)
        assert result.low_expected_count is True

    def test_low_expected_count_computed_for_always_valid_inference_too(self):
        """The descriptive flag is not limited to the fixed-inference path."""
        counts = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
        expected = {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2}
        result = sample_ratio_mismatch(counts, expected=expected, inference="always_valid")

        assert result.min_expected_count == pytest.approx(3.0)
        assert result.low_expected_count is True


class TestAllocationPosteriorBands:
    def test_posterior_mean_exact(self):
        """Posterior mean equals (prior_a + n_k) / (prior_a + prior_b + N) exactly."""
        counts = {"control": 30, "treatment": 70}
        total = sum(counts.values())
        rows = [{"ds": "2024-01-01", "group_id": g, "n_cumulative": n} for g, n in counts.items()]
        prior_a, prior_b = 1.0, 1.0

        bands = allocation_posterior_bands(rows)

        for band in bands:
            n_k = counts[band.group_id]
            assert band.mean == (prior_a + n_k) / (prior_a + prior_b + total)

    def test_bounds_match_scipy_beta_ppf(self):
        """Bounds equal scipy.stats.beta(a, b).ppf(...) for the same (a, b)."""
        counts = {"control": 40, "treatment": 60}
        total = sum(counts.values())
        rows = [{"ds": "2024-01-01", "group_id": g, "n_cumulative": n} for g, n in counts.items()]
        credible_level = 0.95

        bands = allocation_posterior_bands(rows, credible_level=credible_level)

        lo_q, hi_q = (1 - credible_level) / 2, 1 - (1 - credible_level) / 2
        for band in bands:
            n_k = counts[band.group_id]
            a, b = 1.0 + n_k, 1.0 + (total - n_k)
            expected_lower, expected_upper = beta_dist(a, b).ppf([lo_q, hi_q])
            assert band.lower == expected_lower
            assert band.upper == expected_upper

    def test_upper_bound_survives_where_ppf_complement_collapses(self):
        """At credible_level=1-1e-16, 1-lower_q rounds to exactly
        1.0 in float64, so the old ppf(1 - lower_q) construction returned
        a spuriously exact upper bound of 1.0; isf(lower_q) resolves the
        true (finite, < 1) upper tail directly."""
        counts = {"control": 30, "treatment": 70}
        rows = [{"ds": "d", "group_id": g, "n_cumulative": n} for g, n in counts.items()]
        credible_level = 1.0 - 1e-16

        bands = allocation_posterior_bands(rows, credible_level=credible_level)

        lower_q = (1.0 - credible_level) / 2.0
        for band in bands:
            n_k = counts[band.group_id]
            a, b = 1.0 + n_k, 1.0 + (100 - n_k)
            expected_upper = beta_dist(a, b).isf(lower_q)
            assert band.upper == pytest.approx(expected_upper)
            assert band.upper < 1.0

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5, math.nan, math.inf, -math.inf])
    def test_credible_level_outside_unit_interval_refused(self, level):
        rows = [{"ds": "d", "group_id": "control", "n_cumulative": 10}]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows, credible_level=level)
        assert exc_info.value.code == "estimation.diagnostics.credible_level"

    def test_credible_level_too_close_to_one_is_refused(self):
        """Verified reproduction: credible_level=1-1e-20 rounds to exactly
        1.0 in float64 before this function ever sees it (any float64
        credible_level admits (1-credible_level)/2 > 0, so the domain
        check alone is what catches this), and the old code silently
        returned a degenerate [0, 1] band from the rounded value.
        Refuse instead."""
        rows = [{"ds": "d", "group_id": "control", "n_cumulative": 10}]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows, credible_level=1 - 1e-20)
        assert exc_info.value.code == "estimation.diagnostics.credible_level"

    @pytest.mark.parametrize(
        "prior", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (math.nan, 1.0), (math.inf, 1.0)]
    )
    def test_invalid_prior_refused(self, prior):
        rows = [{"ds": "d", "group_id": "control", "n_cumulative": 10}]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows, prior=prior)
        assert exc_info.value.code in (
            "estimation.diagnostics.prior_alpha_finite",
            "estimation.diagnostics.prior_beta_finite",
        )

    def test_band_width_shrinks_as_n_grows(self):
        """Band width strictly decreases as N grows at a fixed ~50/50 split."""
        widths = []
        for total in (100, 1_000, 10_000):
            n_control = total // 2
            rows = [
                {"ds": "d", "group_id": "control", "n_cumulative": n_control},
                {"ds": "d", "group_id": "treatment", "n_cumulative": total - n_control},
            ]
            band = next(b for b in allocation_posterior_bands(rows) if b.group_id == "control")
            widths.append(band.upper - band.lower)
        assert widths[0] > widths[1] > widths[2]

    def test_three_arm_matches_independent_binomial_posterior(self):
        """Each arm's band is the independent arm-vs-rest Beta(1+n_k, 1+N-n_k)
        posterior - NOT the Dirichlet(1,..,1) marginal, which would be
        Beta(1+n_k, (K-1)+N-n_k). A defensible descriptive choice (the K
        per-arm priors are jointly incoherent for K>2), pinned here so the
        justification stays honest."""
        counts = {"A": 200, "B": 300, "C": 500}
        total = sum(counts.values())
        rows = [{"ds": "d", "group_id": g, "n_cumulative": n} for g, n in counts.items()]

        bands = allocation_posterior_bands(rows)

        assert len(bands) == 3
        for band in bands:
            n_k = counts[band.group_id]
            a, b = 1 + n_k, 1 + (total - n_k)
            dist = beta_dist(a, b)
            assert band.mean == a / (a + b)
            lo, hi = dist.ppf([0.025, 0.975])
            assert band.lower == lo
            assert band.upper == hi

    def test_duplicate_ds_group_row_refused(self):
        """Duplicated (ds, group_id) rows (a re-run day, an un-deduped
        union) double-count n_total exactly like the multi-experiment case
        the sibling guard exists for - [control:50, control:50,
        treatment:50] would silently report control mean 0.336 instead of
        0.5. Refuse loudly."""
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 50},
            {"ds": "d", "group_id": "control", "n_cumulative": 50},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 50},
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows)
        assert exc_info.value.code == "estimation.diagnostics.duplicate_ds_group"

    def test_fractional_n_cumulative_refused(self):
        """int() truncation of 10.9 -> 10 silently misstates the posterior;
        a fractional cumulative count is refused by name. Integral floats
        stay accepted."""
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 10.9},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 10},
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows)
        assert exc_info.value.code == "estimation.diagnostics.n_cumulative_whole"
        ok = allocation_posterior_bands(
            [
                {"ds": "d", "group_id": "control", "n_cumulative": 10.0},
                {"ds": "d", "group_id": "treatment", "n_cumulative": 10},
            ]
        )
        assert [b.n for b in ok] == [10, 10]

    def test_negative_n_cumulative_refused(self):
        """A negative count previously passed straight through into
        the Beta parameters, corrupting the posterior."""
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": -5},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 10},
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows)
        assert exc_info.value.code == "estimation.diagnostics.n_cumulative_non"

    def test_band_contains_expected_share_when_balanced(self):
        """Balanced 50/50 split at large N -> band contains the expected 0.5 share."""
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 50_000},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 50_000},
        ]
        for band in allocation_posterior_bands(rows):
            assert band.lower < 0.5 < band.upper

    def test_band_excludes_expected_share_when_imbalanced(self):
        """Imbalanced 60/40 split at large N -> band excludes the 0.5 expected share."""
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 60_000},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 40_000},
        ]
        bands = {b.group_id: b for b in allocation_posterior_bands(rows)}
        assert bands["control"].lower > 0.5
        assert bands["treatment"].upper < 0.5

    def test_allocation_band_fields(self):
        """AllocationBand carries n/n_total/credible_level and no pass/fail flag."""
        rows = [
            {"ds": "2024-01-01", "group_id": "control", "n_cumulative": 30},
            {"ds": "2024-01-01", "group_id": "treatment", "n_cumulative": 70},
        ]

        bands = allocation_posterior_bands(rows, credible_level=0.9)

        assert len(bands) == 2
        for band in bands:
            assert isinstance(band, AllocationBand)
            assert band.ds == "2024-01-01"
            assert band.n_total == 100
            assert band.credible_level == 0.9
        # Purely descriptive - never a pass/fail gate.
        assert not any(f.annotation is bool for f in AllocationBand.model_fields.values())

    def test_multiple_ds_grouped_independently(self):
        """N(t) is computed per-ds, not pooled across dates."""
        rows = [
            {"ds": "2024-01-01", "group_id": "control", "n_cumulative": 10},
            {"ds": "2024-01-01", "group_id": "treatment", "n_cumulative": 10},
            {"ds": "2024-01-02", "group_id": "control", "n_cumulative": 100},
            {"ds": "2024-01-02", "group_id": "treatment", "n_cumulative": 100},
        ]

        bands = allocation_posterior_bands(rows)

        day1 = [b for b in bands if b.ds == "2024-01-01"]
        day2 = [b for b in bands if b.ds == "2024-01-02"]
        assert all(b.n_total == 20 for b in day1)
        assert all(b.n_total == 200 for b in day2)
        assert (day1[0].upper - day1[0].lower) > (day2[0].upper - day2[0].lower)

    def test_rejects_rows_spanning_multiple_experiments(self):
        """Two experiments sharing a ds must not have their n_total pooled -
        that would silently inflate the denominator and produce a wrong,
        spuriously narrow band for both.
        """
        rows = [
            {
                "experiment_id": "exp_a",
                "ds": "2024-01-01",
                "group_id": "control",
                "n_cumulative": 100,
            },
            {
                "experiment_id": "exp_a",
                "ds": "2024-01-01",
                "group_id": "treatment",
                "n_cumulative": 100,
            },
            {
                "experiment_id": "exp_b",
                "ds": "2024-01-01",
                "group_id": "control",
                "n_cumulative": 900,
            },
            {
                "experiment_id": "exp_b",
                "ds": "2024-01-01",
                "group_id": "treatment",
                "n_cumulative": 900,
            },
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows)
        assert exc_info.value.code == "estimation.diagnostics.allocation_posterior_bands"

    def test_single_experiment_id_column_is_fine(self):
        """An experiment_id column is accepted as long as it's a single value -
        only spanning multiple experiments is rejected.
        """
        rows = [
            {
                "experiment_id": "exp_a",
                "ds": "2024-01-01",
                "group_id": "control",
                "n_cumulative": 30,
            },
            {
                "experiment_id": "exp_a",
                "ds": "2024-01-01",
                "group_id": "treatment",
                "n_cumulative": 70,
            },
        ]
        bands = allocation_posterior_bands(rows)
        assert len(bands) == 2
        assert all(b.n_total == 100 for b in bands)

    def test_posterior_params_reconstruct_the_interval(self):
        """posterior_a/posterior_b are the exact Beta the interval came from, so a
        caller can rebuild the full posterior (density, samples) not just the summary.
        """
        rows = [
            {"ds": "2024-01-01", "group_id": "control", "n_cumulative": 30},
            {"ds": "2024-01-01", "group_id": "treatment", "n_cumulative": 70},
        ]
        credible_level = 0.95
        bands = allocation_posterior_bands(rows, credible_level=credible_level)

        # Derive quantiles exactly as the impl does: (1-0.95)/2 is
        # 0.025000000000000022 in float64, and the quantile is sensitive to
        # that. The upper bound comes from the survival form isf(lo_q), never
        # ppf(1 - lo_q), so the complement is never formed.
        lo_q = (1.0 - credible_level) / 2.0

        for band in bands:
            dist = beta_dist(band.posterior_a, band.posterior_b)
            assert band.mean == band.posterior_a / (band.posterior_a + band.posterior_b)
            assert band.lower == float(dist.ppf(lo_q))
            assert band.upper == float(dist.isf(lo_q))

    def test_posterior_params_track_a_non_default_prior(self):
        """A caller passing Jeffreys gets Jeffreys-derived params back - the point
        of carrying them, since n/n_total alone can't recover the prior.
        """
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 30},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 70},
        ]
        jeffreys = allocation_posterior_bands(rows, prior=(0.5, 0.5))
        control = next(b for b in jeffreys if b.group_id == "control")
        assert control.posterior_a == 0.5 + 30
        assert control.posterior_b == 0.5 + 70

        uniform = allocation_posterior_bands(rows, prior=(1.0, 1.0))
        control_uniform = next(b for b in uniform if b.group_id == "control")
        assert control_uniform.posterior_a == 1.0 + 30
        # Same n/n_total, different prior -> different posterior. Reconstructing
        # from n/n_total alone would have silently produced the uniform answer.
        assert control.posterior_a != control_uniform.posterior_a

    def test_pandas_pyarrow_and_dict_rows_agree(self):
        """allocation_posterior_bands gives identical results for a pandas
        DataFrame, a pyarrow Table, and a plain list[dict] over the same rows -
        matching the input contract already established by estimate_lift.
        """
        rows = [
            {"ds": "2024-01-01", "group_id": "control", "n_cumulative": 480},
            {"ds": "2024-01-01", "group_id": "treatment", "n_cumulative": 520},
            {"ds": "2024-01-02", "group_id": "control", "n_cumulative": 990},
            {"ds": "2024-01-02", "group_id": "treatment", "n_cumulative": 1010},
        ]

        dict_result = allocation_posterior_bands(rows)
        pandas_result = allocation_posterior_bands(pd.DataFrame(rows))
        pyarrow_result = allocation_posterior_bands(pa.Table.from_pylist(rows))

        dict_dump = [b.model_dump() for b in dict_result]
        assert [b.model_dump() for b in pandas_result] == dict_dump
        assert [b.model_dump() for b in pyarrow_result] == dict_dump


class TestNotApplicable:
    def test_fields(self):
        """NotApplicable carries the check name and a human-readable reason."""
        na = NotApplicable(check="srm", reason="observational design has no target allocation")
        assert na.check == "srm"
        assert na.reason == "observational design has no target allocation"

    def test_frozen_mutation_raises(self):
        na = NotApplicable(check="srm", reason="observational design has no target allocation")
        with pytest.raises(ValidationError):
            na.reason = "other"  # ty: ignore[invalid-assignment]  # proving frozen at runtime


class TestAllocationBandJudgesItsOwnBounds:
    """Every representable credible_level below 1 yields a positive lower tail,
    so the input check cannot detect resolution loss: an imbalanced posterior
    still collapses to a bound of exactly 0 or 1, which reads as certainty
    about the allocation rather than a band."""

    def test_a_degenerate_posterior_bound_is_refused(self):
        # Beta(N+1, 1): every unit in one arm, so the upper bound rounds to
        # exactly 1.0 at an extreme credible level.
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 100_000},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 0},
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            allocation_posterior_bands(rows, credible_level=1.0 - 1e-15)
        assert exc_info.value.code == "estimation.diagnostics.allocation_band_ds"

    def test_an_ordinary_allocation_still_reports_a_proper_band(self):
        rows = [
            {"ds": "d", "group_id": "control", "n_cumulative": 500},
            {"ds": "d", "group_id": "treatment", "n_cumulative": 500},
        ]
        for band in allocation_posterior_bands(rows, credible_level=0.95):
            assert 0.0 < band.lower <= band.upper < 1.0


def test_srm_result_min_expected_count_is_json_portable():
    import json

    # Computed: the finite minimum serializes to a JSON number and reconstructs
    # to the same value.
    result = sample_ratio_mismatch({"a": 100, "b": 100}, inference="fixed")
    payload = result.model_dump_json()
    assert result.min_expected_count is not None
    assert json.loads(payload)["min_expected_count"] == result.min_expected_count
    assert SRMResult.model_validate_json(payload).min_expected_count == result.min_expected_count

    # Omitted: the default serializes to portable JSON null (not a non-finite
    # float) and reconstructs as None.
    minimal = SRMResult(
        inference="fixed",
        chi2_stat=0.0,
        fixed_p_value=1.0,
        log_e_value=None,
        df=1,
        is_srm=False,
        alpha=0.001,
        observed={"a": 1, "b": 1},
        expected={"a": 0.5, "b": 0.5},
    )
    payload = minimal.model_dump_json()
    assert json.loads(payload)["min_expected_count"] is None
    assert SRMResult.model_validate_json(payload).min_expected_count is None


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.diagnostics.expected_allocation_contain",
            lambda: resolve_srm_expected(
                {"control": 0.5, "(unassigned)": 0.1}, allocation=None, inference="always_valid"
            ),
        ),
        (
            "estimation.diagnostics.always_srm_predeclared",
            lambda: resolve_srm_expected(None, allocation=None, inference="always_valid"),
        ),  # estimation/diagnostics.py::resolve_srm_expected
        (
            "estimation.diagnostics.alpha",
            lambda: sample_ratio_mismatch({"a": 1, "b": 1}, alpha=0.0),
        ),  # estimation/diagnostics.py::sample_ratio_mismatch
        (
            "estimation.diagnostics.credible_level",
            lambda: allocation_posterior_bands(
                [{"ds": "d", "group_id": "control", "n_cumulative": 10}], credible_level=0.0
            ),
        ),  # estimation/diagnostics.py::allocation_posterior_bands
    ],
)
def test_diagnostics_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code
