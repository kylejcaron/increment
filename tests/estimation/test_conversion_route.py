"""The ``conversion_inference`` route: a count-only rule picks the delta-method route for dense
unadjusted conversion contrasts and the finite-sample route for the rest, and every row labels
which one produced it."""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import binom, norm

from increment.errors import CodedError, InvalidRequestError
from increment.estimation.armstats import ArmStats
from increment.estimation.binomial_rr import FINITE_SAMPLE_MAX_ARM_SIZE
from increment.estimation.conversion_route import (
    PLANNING_ROUTE_CERTAINTY,
    dense_extent,
    dense_min_count,
    family_route_alpha,
    planning_route,
    route_for_counts,
    routed_share,
    unrouted_share,
)
from increment.estimation.engine import Method, _estimate_lift, estimate_lift
from increment.estimation.family import bh_select
from increment.estimation.inference import Normal
from increment.semantics.models import MeanMetric
from tests.estimation._conversion_counts import (
    CONVERSION_METRIC,
    count_summary,
    lift_computation,
    lift_row,
)

PRODUCTION_TAILS = (0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1)
# alpha=0.05 two-sided and directional: the two allocations the default plan produces.
TWO_SIDED_TAIL = 0.025
DIRECTIONAL_TAIL = 0.05


def _smallest(x_c: int, n_c: int, x_t: int, n_t: int) -> int:
    return min(x_c, n_c - x_c, x_t, n_t - x_t)


class TestDenseMinCount:
    def test_the_shipped_law_at_the_production_tails(self):
        # tail: the least per-arm success and failure count at which `auto` is asymptotic.
        shipped = {tail: dense_min_count(tail) for tail in PRODUCTION_TAILS}
        assert shipped == {
            0.0005: 17000,
            0.001: 13224,
            0.005: 6384,
            0.01: 4247,
            0.025: 2140,
            0.05: 1062,
            0.1: 412,
        }

    # tail: the least step of a geometric ladder (ratio 1.15, anchored by `select` at 10 for the
    # tails from 0.025 up and at 1,637 below) whose all-draw noncoverage excess stays within
    # `scientific_delta(tail)` there and at every larger step, measured by
    # `python -m calibration.conversion_route select` on `calibration.conversion_route.cells`.
    MEASURED_REQUIREMENT = {
        0.0005: 13320,
        0.001: 10072,
        0.005: 2863,
        0.01: 2165,
        0.025: 576,
        0.05: 286,
        0.1: 329,
    }

    @pytest.mark.parametrize("tail", PRODUCTION_TAILS)
    def test_the_law_is_at_least_the_plans_margin_over_the_measured_requirement(self, tail):
        assert dense_min_count(tail) >= math.ceil(1.25 * self.MEASURED_REQUIREMENT[tail])

    @given(st.floats(min_value=1e-300, max_value=0.5, exclude_max=True))
    def test_every_tail_clears_the_guard_floor(self, tail):
        """Every routed-asymptotic cell has all four counts at least the floor, so the
        combined log-scale standard error is below ``sqrt(2 / 9) < 0.5`` and no
        ``infer_lift`` guard can fire."""
        assert dense_min_count(tail) >= 9

    @given(
        st.floats(min_value=1e-300, max_value=0.5, exclude_max=True),
        st.floats(min_value=1e-300, max_value=0.5, exclude_max=True),
    )
    def test_a_more_extreme_tail_never_needs_fewer_counts(self, tail, other):
        low, high = sorted((tail, other))
        assert dense_min_count(low) >= dense_min_count(high)

    @pytest.mark.parametrize("tail", PRODUCTION_TAILS)
    def test_the_threshold_reaches_the_tail_quantile_through_the_survival_form(self, tail):
        # `norm.isf(tail)`, never a quantile of `1 - tail`: a 1e-300 tail stays finite.
        assert math.isfinite(norm.isf(1e-300))
        assert dense_min_count(1e-300) >= dense_min_count(tail)


class TestRouteForCounts:
    def test_finite_sample_mode_never_leaves_the_finite_route(self):
        counts = (10**6, 10**7, 10**6, 10**7)
        assert route_for_counts(*counts, tail_alpha=0.025, mode="finite_sample") == "finite_sample"
        assert route_for_counts(*counts, tail_alpha=0.025, mode="auto") == "asymptotic"

    @pytest.mark.parametrize("tail", PRODUCTION_TAILS)
    @pytest.mark.parametrize("offset", [-1, 0, 1])
    def test_the_smallest_of_four_counts_decides_at_the_threshold(self, tail, offset):
        m = dense_min_count(tail)
        n = 20 * m
        for counts in (
            (m + offset, n, 3 * m, n),  # sparsest: control successes
            (3 * m, n, m + offset, n),  # treatment successes
            (n - (m + offset), n, 3 * m, n),  # control failures
            (3 * m, n, n - (m + offset), n),  # treatment failures
        ):
            expected = "asymptotic" if offset >= 0 else "finite_sample"
            assert route_for_counts(*counts, tail_alpha=tail, mode="auto") == expected

    @pytest.mark.parametrize("tail", [0.0, -0.1, float("nan")])
    def test_a_tail_the_delta_method_cannot_resolve_takes_the_finite_route(self, tail):
        assert (
            route_for_counts(10**6, 10**7, 10**6, 10**7, tail_alpha=tail, mode="auto")
            == "finite_sample"
        )

    @settings(max_examples=300, deadline=None)
    @given(
        n_c=st.integers(min_value=1, max_value=10**9),
        n_t=st.integers(min_value=1, max_value=10**9),
        frac_c=st.floats(min_value=0.0, max_value=1.0),
        frac_t=st.floats(min_value=0.0, max_value=1.0),
        tail=st.sampled_from(PRODUCTION_TAILS),
    )
    def test_the_route_is_a_symmetric_pure_function_of_the_four_counts(
        self, n_c, n_t, frac_c, frac_t, tail
    ):
        x_c, x_t = round(frac_c * n_c), round(frac_t * n_t)
        route = route_for_counts(x_c, n_c, x_t, n_t, tail_alpha=tail, mode="auto")
        assert route == route_for_counts(x_c, n_c, x_t, n_t, tail_alpha=tail, mode="auto")
        # Neither the arm order nor swapping successes for failures changes it.
        assert route == route_for_counts(x_t, n_t, x_c, n_c, tail_alpha=tail, mode="auto")
        assert route == route_for_counts(
            n_c - x_c, n_c, n_t - x_t, n_t, tail_alpha=tail, mode="auto"
        )
        assert (route == "asymptotic") == (_smallest(x_c, n_c, x_t, n_t) >= dense_min_count(tail))


class TestRowsCarryTheirRoute:
    def test_dense_counts_take_the_delta_method_route_and_say_so(self):
        row = lift_row((50_000, 1_000_000, 51_500, 1_000_000))
        assert row.reference_kind == "t"
        assert row.scale == "log"
        assert row.binomial_set is None
        assert row.lift is not None and row.lift.lb is not None and row.lift.ub is not None

    def test_sparse_counts_take_the_finite_sample_route_and_say_so(self):
        row = lift_row((300, 10_000, 330, 10_000))
        assert row.reference_kind == "binomial"
        assert row.scale == "linear"
        assert row.binomial_set is not None

    def test_finite_sample_mode_labels_a_dense_row_binomial(self):
        counts = (50_000, 1_000_000, 51_500, 1_000_000)
        auto = lift_row(counts)
        pinned = lift_row(counts, mode="finite_sample")
        assert (auto.reference_kind, pinned.reference_kind) == ("t", "binomial")
        assert pinned.binomial_set is not None
        # The two guarantees differ, the point estimate does not.
        assert pinned.require_lift().value == pytest.approx(auto.require_lift().value, rel=1e-12)

    @pytest.mark.parametrize("tail", [0.1, 0.05, 0.025, 0.005])
    def test_the_route_flips_exactly_at_the_threshold(self, tail):
        alpha = 2.0 * tail
        m = dense_min_count(tail)
        n = 5 * m
        kinds = {
            offset: lift_row((m + offset, n, 2 * m, n), alpha=alpha).reference_kind
            for offset in (-1, 0, 1)
        }
        assert kinds == {-1: "binomial", 0: "t", 1: "t"}

    def test_a_directional_alpha_spends_its_whole_tail_on_the_route(self):
        """Two-sided ``alpha`` puts ``alpha / 2`` in each tail and a directional one puts
        all of it in one, so the same counts can be dense for the second and sparse for the
        first."""
        low = dense_min_count(DIRECTIONAL_TAIL)
        high = dense_min_count(TWO_SIDED_TAIL)
        assert low < high
        counts = (low, 20 * low, 3 * low, 20 * low)
        assert lift_row(counts, alpha=0.05, alternative="two-sided").reference_kind == "binomial"
        assert lift_row(counts, alpha=0.05, alternative="greater").reference_kind == "t"
        assert lift_row(counts, alpha=0.05, alternative="less").reference_kind == "t"

    @pytest.mark.parametrize("treatment_successes", [2_200, 4_000, 8_000, 16_000, 30_000])
    def test_the_route_does_not_follow_the_effect_or_the_p_value(self, treatment_successes):
        # Control and treatment failures stay far above the threshold; only the effect moves.
        row = lift_row((8_000, 100_000, treatment_successes, 100_000))
        assert row.reference_kind == "t"
        spec = lift_row((8_000, 100_000, treatment_successes, 100_000), mode="finite_sample")
        assert spec.reference_kind == "binomial"

    @pytest.mark.parametrize(
        "counts",
        [
            (0, 5_000_000, 2, 5_000_000),
            (5_000_000, 5_000_000, 4_999_998, 5_000_000),
            (0, 20_000, 0, 20_000),
            (12, 10_000_000, 9, 10_000_000),
        ],
    )
    def test_boundary_events_stay_on_the_finite_sample_route_with_their_typed_set(self, counts):
        """Zero and all-conversion arms are never dense: they keep the typed
        ``binomial_set`` (a set with no point estimate where the control rate is zero and the
        ratio is undefined), even above the previous four-million cap."""
        row = lift_row(counts)
        assert row.reference_kind == "binomial"
        assert row.binomial_set is not None
        assert (row.lift is None) == (counts[0] == 0)

    def test_set_only_binomial_readout_keeps_count_evidence_without_posterior_stats(self):
        from increment.tables import estimates_to_readout

        row = lift_row((0, 5_000, 2, 5_000))
        assert row.lift is None
        assert row.binomial_set is not None

        (readout,) = estimates_to_readout([row])

        assert readout["binomial_set"] == row.binomial_set
        assert readout["stat_sig"] == row.stat_sig()
        assert readout["posterior_chance_to_beat"] is None
        assert readout["posterior_risk_if_shipped"] is None
        assert not {"chance_to_beat (advisory)", "risk_if_shipped (advisory)"} & readout.keys()

    def test_a_corrupted_arm_refuses_identically_under_both_modes(self):
        """Exact counts and binary moments are validated before route selection."""
        bad = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id="control",
            n=100_000,
            successes=40_000,
            sum_y=40_000.0,
            sum_y2=55_000.0,
        )
        good = ArmStats.from_raw_sums(
            study_id="e",
            metric="conv",
            group_id="treatment",
            n=100_000,
            successes=41_000,
            sum_y=41_000.0,
            sum_y2=41_000.0,
        )
        rows = [
            {
                "experiment_id": "e",
                "metric": "conv",
                "group_id": arm.group_id,
                "n": arm.n,
                "successes": arm.successes,
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
            }
            for arm in (bad, good)
        ]
        codes = {}
        for mode in ("auto", "finite_sample"):
            computation = estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=rows,
                control_group="control",
                methods=[Method(name="unadjusted", conversion_inference=mode)],
            )
            assert not computation.results
            codes[mode] = {failure.code for failure in computation.failures.values()}
        assert codes["auto"] == codes["finite_sample"]
        assert len(codes["auto"]) == 1


class TestAboveTheFiniteSampleCeiling:
    """The delta-method route has no evaluator ceiling; the finite-sample route refuses above
    ``FINITE_SAMPLE_MAX_ARM_SIZE`` by a code that names the way forward."""

    CEILING = FINITE_SAMPLE_MAX_ARM_SIZE

    @pytest.mark.parametrize("n", [2 * FINITE_SAMPLE_MAX_ARM_SIZE, 10 * FINITE_SAMPLE_MAX_ARM_SIZE])
    def test_dense_arms_succeed_by_default(self, n):
        counts = (n // 20, n, n // 20 + n // 400, n)
        row = lift_row(counts)
        assert row.reference_kind == "t"
        lift = row.require_lift()
        assert lift.lb is not None and lift.ub is not None
        assert lift.lb < lift.value < lift.ub
        assert lift.value == pytest.approx(0.05, rel=1e-6)

    def test_pinned_finite_sample_and_sparse_arms_are_refused_by_the_ceiling_code(self):
        n = 2 * self.CEILING
        dense = (n // 20, n, n // 20 + n // 400, n)
        sparse = (3, n, 7, n)
        for counts, mode in ((dense, "finite_sample"), (sparse, "auto")):
            computation = lift_computation(
                counts, method=Method(name="unadjusted", conversion_inference=mode)
            )
            assert not computation.results
            (failure,) = computation.failures.values()
            assert failure.code == "estimation.binomial.finite_sample_arm_ceiling_exceeded"
            assert failure.context["max_arm_size"] == self.CEILING

    def test_the_ceiling_itself_is_admitted_to_the_finite_sample_route(self):
        row = lift_row((3, self.CEILING, 7, self.CEILING))
        assert row.reference_kind == "binomial"

    def test_a_sparse_arm_above_the_old_four_million_cap_is_estimated(self):
        row = lift_row((2, 5_000_000, 0, 5_000_000))
        assert row.reference_kind == "binomial"
        assert row.binomial_set is not None


class TestRoutedAsymptoticCellsNeverHitAGuard:
    """Every routed-asymptotic cell has all four counts at least ``dense_min_count >= 9``, so
    the delta-method guards (``log_se >= 0.5``, zero variance, non-positive mean) cannot
    fire: a dense request always yields an interval, at any size."""

    @pytest.mark.parametrize("tail", PRODUCTION_TAILS)
    @pytest.mark.parametrize(
        "shape",
        ["balanced_half", "rare_success", "rare_failure", "lopsided_arms", "huge_arms"],
    )
    def test_the_sparsest_dense_cell_yields_an_interval(self, tail, shape):
        m = dense_min_count(tail)
        counts = {
            "balanced_half": (m, 2 * m, m, 2 * m),
            "rare_success": (m, 10**9, m, 10**9),
            "rare_failure": (10**9 - m, 10**9, 10**9 - m, 10**9),
            "lopsided_arms": (m, 3 * m, 4 * m, 10 * m),
            "huge_arms": (m, 10**9, 5 * m, 10**9),
        }[shape]
        assert route_for_counts(*counts, tail_alpha=tail, mode="auto") == "asymptotic"
        row = lift_row(counts, alpha=2.0 * tail)
        assert row.reference_kind == "t"
        lift = row.require_lift()
        assert lift.lb is not None and lift.ub is not None
        assert lift.lb < lift.value < lift.ub

    @settings(max_examples=60, deadline=None)
    @given(
        n_c=st.integers(min_value=2, max_value=10**7),
        n_t=st.integers(min_value=2, max_value=10**7),
        frac_c=st.floats(min_value=0.0, max_value=1.0),
        frac_t=st.floats(min_value=0.0, max_value=1.0),
        tail=st.sampled_from([0.025, 0.05, 0.1]),
    )
    def test_no_count_pair_raises_a_guard_and_every_row_is_labelled_by_the_rule(
        self, n_c, n_t, frac_c, frac_t, tail
    ):
        x_c, x_t = round(frac_c * n_c), round(frac_t * n_t)
        computation = lift_computation((x_c, n_c, x_t, n_t), alpha=2.0 * tail)
        route = route_for_counts(x_c, n_c, x_t, n_t, tail_alpha=tail, mode="auto")
        if route == "asymptotic":
            assert computation.failures == {}
            (row,) = computation.results
            assert row.reference_kind == "t"
        else:
            # Finite-sample rows may carry a typed failure (a tail the evaluator cannot
            # resolve), never a delta-method guard.
            for failure in computation.failures.values():
                assert failure.code.startswith("estimation.binomial.")
            for row in computation.results:
                assert row.reference_kind == "binomial"


class TestExplicitFiniteSample:
    def test_cuped_is_refused_at_construction_by_code(self):
        with pytest.raises(CodedError) as raised:
            Method(name="cuped", variance_reduction="cuped", conversion_inference="finite_sample")
        assert raised.value.code == "conversion_inference.finite_sample.cuped"
        assert Method(name="cuped", variance_reduction="cuped").conversion_inference == "auto"

    def test_an_unknown_value_is_refused(self):
        with pytest.raises(ValueError):
            Method(name="unadjusted", conversion_inference="asymptotic")  # ty: ignore[invalid-argument-type]

    def test_an_informative_prior_is_refused_by_code_instead_of_served_by_the_delta_method(self):
        with pytest.raises(InvalidRequestError) as raised:
            estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(300, 10_000, 330, 10_000),
                control_group="control",
                methods=[Method(name="unadjusted", conversion_inference="finite_sample")],
                prior=Normal(mu=0.0, sigma=0.1),
            )
        assert raised.value.code == "estimation.binomial.finite_sample_unavailable"

    def test_a_mean_metric_is_refused_by_the_shared_metric_type_code(self):
        rows = [{**row, "metric": "revenue"} for row in count_summary(300, 10_000, 330, 10_000)]
        with pytest.raises(InvalidRequestError) as raised:
            estimate_lift(
                metrics=[MeanMetric(name="revenue", entity="user", fact="revenue")],
                summary=rows,
                control_group="control",
                methods=[Method(name="unadjusted", conversion_inference="finite_sample")],
            )
        assert raised.value.code == "conversion_inference.finite_sample.metric_type"

    def test_auto_on_an_ineligible_contrast_behaves_as_before(self):
        rows = [{**row, "metric": "revenue"} for row in count_summary(300, 10_000, 330, 10_000)]
        computation = estimate_lift(
            metrics=[MeanMetric(name="revenue", entity="user", fact="revenue")],
            summary=rows,
            control_group="control",
        )
        (row,) = computation.results
        assert row.reference_kind == "t"
        assert row.binomial_set is None

    def test_the_prior_is_served_by_auto_on_the_delta_method_path_as_before(self):
        summary = count_summary(3_000, 10_000, 3_300, 10_000)
        (baseline,) = estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=summary,
            control_group="control",
        ).results
        (row,) = estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=summary,
            control_group="control",
            prior=Normal(mu=0.0, sigma=0.1),
        ).results
        assert row.binomial_set is None
        assert row.reference_kind == "t"
        assert row.lift is not None and baseline.lift is not None
        assert row.lift.value == baseline.lift.value
        assert row.posterior_available is True
        assert row.posterior_estimate is not None
        assert row.posterior_estimate != row.lift.value


class TestPlanningRoute:
    """``planning_route`` classifies a plan from the probability that the runtime rule sends
    its random counts to the delta-method route."""

    def test_certain_density_is_dense(self):
        assert (
            planning_route(1_000_000, 1_000_000, 0.05, 0.052, tail_alpha=0.025, mode="auto")
            == "dense"
        )

    def test_counts_that_cannot_reach_the_threshold_are_sparse(self):
        assert (
            planning_route(2_000, 2_000, 0.001, 0.0012, tail_alpha=0.025, mode="auto") == "sparse"
        )

    def test_counts_the_threshold_splits_are_borderline(self):
        m = dense_min_count(0.025)
        assert (
            planning_route(10 * m, 10 * m, 0.1, 0.11, tail_alpha=0.025, mode="auto") == "borderline"
        )

    def test_a_finite_sample_decision_is_always_sparse(self):
        assert (
            planning_route(10**8, 10**8, 0.2, 0.25, tail_alpha=0.025, mode="finite_sample")
            == "sparse"
        )

    def test_an_arm_too_small_to_hold_the_threshold_on_both_sides_is_sparse(self):
        m = dense_min_count(0.05)
        assert planning_route(2 * m - 1, 10**7, 0.5, 0.5, tail_alpha=0.05, mode="auto") == "sparse"

    def test_the_classification_agrees_with_simulated_counts(self):
        """The simulated share of count draws the runtime rule routes asymptotic classifies
        the plan as the planner does, for every certain class and one borderline plan."""
        rng = np.random.default_rng(20261004)
        tail = 0.025
        m = dense_min_count(tail)
        cases = {
            "dense": (50 * m, 50 * m, 0.3, 0.31),
            "sparse": (3 * m, 3 * m, 0.0001, 0.0002),
            "borderline": (10 * m, 10 * m, 0.1, 0.1),
        }
        reps = 20_000
        for expected, (n_c, n_t, p_c, p_t) in cases.items():
            assert planning_route(n_c, n_t, p_c, p_t, tail_alpha=tail, mode="auto") == expected
            x_c = rng.binomial(n_c, p_c, size=reps)
            x_t = rng.binomial(n_t, p_t, size=reps)
            smallest = np.minimum.reduce([x_c, n_c - x_c, x_t, n_t - x_t])
            share = float((smallest >= m).mean())
            if expected == "dense":
                assert share == 1.0
            elif expected == "sparse":
                assert share == 0.0
            else:
                assert 0.0 < share < 1.0

    @pytest.mark.parametrize(
        ("n_c", "n_t", "p_c", "p_t", "tail"),
        [
            (3_000, 3_000, 0.3, 0.33, 0.025),
            (2_000, 5_000, 0.2, 0.25, 0.1),
            (20_000, 18_000, 0.02, 0.03, 0.01),
            (400, 400, 0.5, 0.5, 0.05),
        ],
    )
    def test_the_routed_share_is_the_product_of_the_arms_binomial_band_probabilities(
        self, n_c, n_t, p_c, p_t, tail
    ):
        """The probability that all four counts reach the threshold, derived independently from
        the binomial distribution function; an arm too small to hold it on both sides has none."""
        m = dense_min_count(tail)
        expected = 1.0
        for n, p in ((n_c, p_c), (n_t, p_t)):
            expected *= binom.cdf(n - m, n, p) - binom.cdf(m - 1, n, p) if n >= 2 * m else 0.0
        assert routed_share(n_c, n_t, p_c, p_t, tail_alpha=tail) == pytest.approx(
            expected, rel=1e-9, abs=1e-15
        )

    def test_the_routed_share_decides_the_classification(self):
        """A plan is dense where the share reaches ``1 - PLANNING_ROUTE_CERTAINTY`` and sparse
        where it falls to ``PLANNING_ROUTE_CERTAINTY``, as sizes grow past the threshold."""
        tail = 0.025
        for n in range(400, 40_000, 400):
            share = routed_share(n, n, 0.1, 0.11, tail_alpha=tail)
            route = planning_route(n, n, 0.1, 0.11, tail_alpha=tail, mode="auto")
            assert route == (
                "dense"
                if share >= 1.0 - PLANNING_ROUTE_CERTAINTY
                else "sparse"
                if share <= PLANNING_ROUTE_CERTAINTY
                else "borderline"
            )

    def test_the_classification_is_monotone_in_the_treatment_rate_around_one_half(self):
        m = dense_min_count(0.05)
        n = 4 * m
        extent_dense = dense_extent(n, n, 0.5, 0.45, 0.55, tail_alpha=0.05, mode="auto")
        point = planning_route(n, n, 0.5, 0.5, tail_alpha=0.05, mode="auto")
        assert extent_dense[0] == (point == "dense")
        some, every = dense_extent(n, n, 0.5, 0.001, 0.999, tail_alpha=0.05, mode="auto")
        assert some and not every

    def test_the_certainty_threshold_is_a_probability_budget(self):
        assert 0.0 < PLANNING_ROUTE_CERTAINTY < 1e-3


class TestFamilyRouteLevel:
    """The level a BH family routes its rows at is the smallest threshold selection reads: the
    greatest float not above ``q / hypotheses``, so a row is never routed at a looser level."""

    @given(
        st.floats(min_value=1e-12, max_value=1.0),
        st.integers(min_value=1, max_value=10**6),
    )
    def test_the_level_is_the_greatest_float_not_above_q_over_the_hypotheses(self, q, hypotheses):
        level = family_route_alpha(q, hypotheses)
        assert level is not None
        exact = Fraction(q) / hypotheses
        assert Fraction(level) <= exact < Fraction(math.nextafter(level, math.inf))

    def test_the_level_is_the_rank_one_threshold_benjamini_hochberg_reads(self):
        # 0.1 / 7 is not exactly representable: ordinary division can round above it.
        _, threshold = bh_select([1.0] * 6 + [0.0], 0.1)
        assert family_route_alpha(0.1, 7) == threshold

    def test_a_family_without_hypotheses_has_no_level(self):
        assert family_route_alpha(0.1, 0) is None


class TestMultiplicityRoutesAtTheSmallestFamilyLevel:
    """A BH family reads each nominal p-value at a threshold as small as ``q / m``: a row dense
    at the nominal level but not at that one must not take the delta-method route."""

    def test_route_alpha_routes_at_the_smaller_of_the_two_levels(self):
        nominal = dense_min_count(0.025)
        counts = (nominal, 20 * nominal, 3 * nominal, 20 * nominal)
        assert (
            estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(*counts),
                control_group="control",
                alpha=0.05,
            )
            .results[0]
            .reference_kind
            == "t"
        )
        for route_alpha, kind in ((0.05, "t"), (0.05 / 4, "binomial"), (1e-9, "binomial")):
            row = _estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(*counts),
                control_group="control",
                alpha=0.05,
                route_alpha=route_alpha,
            ).results[0]
            assert row.reference_kind == kind

    def test_a_directional_route_alpha_is_read_in_alphas_own_convention(self):
        low = dense_min_count(0.05)
        counts = (low, 20 * low, 3 * low, 20 * low)
        for route_alpha, kind in ((0.05, "t"), (0.0125, "binomial")):
            row = _estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(*counts),
                control_group="control",
                alpha=0.05,
                alternative="greater",
                route_alpha=route_alpha,
            ).results[0]
            assert row.reference_kind == kind

    def test_only_the_decision_row_is_routed_at_the_family_level(self):
        """A sensitivity method's p-value never enters a family's selection, so it keeps its
        own level: the family level must not move its label or cost it the finite-sample route."""
        nominal = dense_min_count(0.025)
        counts = (nominal, 20 * nominal, 3 * nominal, 20 * nominal)
        computation = _estimate_lift(
            metrics=[CONVERSION_METRIC],
            summary=count_summary(*counts),
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="unadjusted_b")],
            alpha=0.05,
            route_alpha=0.05 / 4,
            method_roles={"unadjusted": "decision", "unadjusted_b": "sensitivity"},
        )
        assert {row.method_role: row.reference_kind for row in computation.results} == {
            "decision": "binomial",
            "sensitivity": "t",
        }

    @pytest.mark.parametrize("route_alpha", [0.0, -0.1, 1.5, float("nan")])
    def test_a_route_level_outside_zero_to_one_is_refused(self, route_alpha):
        with pytest.raises(CodedError) as raised:
            _estimate_lift(
                metrics=[CONVERSION_METRIC],
                summary=count_summary(300, 10_000, 330, 10_000),
                control_group="control",
                route_alpha=route_alpha,
            )
        assert raised.value.code == "estimation.engine.route_alpha"


# Dense at a 0.1 tail, not at a 0.05 tail: a BH family of two hypotheses at q = 0.2 reads p-values
# at q / 2 = 0.1, a 0.05 tail, which `_CLEAR` clears and `_BETWEEN` does not.
_BETWEEN = (dense_min_count(0.1) + dense_min_count(0.05)) // 2
_CLEAR = 2 * dense_min_count(0.05)


def _conversion_frame(successes: dict[str, int]):
    """One conversion metric per entry, with that many successes in the control arm and one more
    in the treatment arm, over covariates and an uptake column the designs below read."""
    import pandas as pd

    rng = np.random.default_rng(20261004)
    n = 4 * _CLEAR
    rows = []
    for i in range(n):
        for group, shift in (("control", 0), ("treatment", 1)):
            rows.append(
                {
                    "unit_id": f"{group}{i}",
                    "group_id": group,
                    **{name: int(i < count + shift) for name, count in successes.items()},
                    "clicked": int(group == "treatment" and i % 2 == 0),
                    "x1": float(rng.normal()),
                    "x2": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def _two_metric_analysis(plan, *, design=None, decision_method=None, successes=None, priors=None):
    from increment import Analysis
    from increment.frame import MetricSpec

    successes = successes or {"a": _BETWEEN, "b": _CLEAR}
    return Analysis.from_unit_summary(
        _conversion_frame(successes),
        unit="unit_id",
        group="group_id",
        **({"control": "control"} if design is None else {"design": design}),
        metrics=[
            MetricSpec(
                name=name,
                type="conversion",
                decision_method=decision_method,
                prior=(priors or {}).get(name),
            )
            for name in successes
        ],
        plan=plan,
    )


def _kinds(rows) -> dict[str, str | None]:
    return {row.metric: row.reference_kind for row in rows if row.estimand in (None, "itt")}


class TestBenjaminiHochbergFamilyRoutesAtItsSmallestLevel:
    def test_a_member_of_a_two_hypothesis_bh_family_is_routed_at_q_over_two(self):
        from increment import AnalysisPlan

        family = _two_metric_analysis(AnalysisPlan(alpha=0.2, q=0.2, secondaries=("a", "b"))).run()
        assert _kinds(family) == {"a": "binomial", "b": "t"}

    def test_the_same_counts_as_a_sole_primary_are_routed_at_their_own_level(self):
        from increment import AnalysisPlan

        alone = _two_metric_analysis(AnalysisPlan(alpha=0.2, primary="a")).run()
        (row,) = [row for row in alone if row.metric == "a"]
        assert row.reference_kind == "t"

    def test_a_prior_bound_secondary_remains_in_the_sampling_family(self):
        """The prior-bound conversion contributes to routing and typed sampling evidence."""
        from increment import AnalysisPlan
        from increment.estimation.inference import Normal

        deeper = dense_min_count(1.0 / 30.0)
        between_levels = (dense_min_count(0.05) + deeper) // 2
        assert dense_min_count(0.05) < between_levels < deeper
        family = _two_metric_analysis(
            AnalysisPlan(alpha=0.2, q=0.2, secondaries=("a", "c", "d")),
            successes={"a": _BETWEEN, "c": _CLEAR, "d": between_levels},
            priors={"c": Normal(mu=0.0, sigma=0.5)},
        ).run()
        kinds = _kinds(family)
        assert kinds == {"a": "binomial", "c": "t", "d": "binomial"}
        rows = list(family)
        assert len(rows) == 3
        prior_row = next(row for row in rows if row.metric == "c")
        assert prior_row.sampling_available is True
        assert prior_row.posterior_available is True

    def test_an_encouragement_family_member_is_routed_at_q_over_its_hypotheses(self):
        from increment import AnalysisPlan
        from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec

        design = Encouragement(
            control_group="control",
            uptake=UptakeSpec(fact="clicked"),
            exclusion_restriction=ExclusionRestriction(
                acknowledged=True, justification="assignment only moves conversion via uptake"
            ),
            one_sided=True,
            min_first_stage_z=0.001,
        )
        family = _two_metric_analysis(
            AnalysisPlan(alpha=0.2, q=0.2, secondaries=("a", "b")), design=design
        ).run(estimands=("itt",))
        assert _kinds(family) == {"a": "binomial", "b": "t"}
        alone = _two_metric_analysis(AnalysisPlan(alpha=0.2, primary="a"), design=design).run(
            estimands=("itt",)
        )
        assert _kinds(alone)["a"] == "t"

    def test_an_observational_family_member_is_routed_at_q_over_its_hypotheses(self):
        from increment import AdjustmentSet, AnalysisPlan, IdentificationGate, Method, Observational

        design = Observational(
            control_group="control",
            adjustment=AdjustmentSet(covariates=("x1", "x2")),
            gate=IdentificationGate(overlap="trim"),
        )
        unadjusted = Method(name="unadjusted")
        family = _two_metric_analysis(
            AnalysisPlan(alpha=0.2, q=0.2, secondaries=("a", "b")),
            design=design,
            decision_method=unadjusted,
        ).run()
        assert _kinds(family) == {"a": "binomial", "b": "t"}
        alone = _two_metric_analysis(
            AnalysisPlan(alpha=0.2, primary="a"), design=design, decision_method=unadjusted
        ).run()
        assert _kinds(alone)["a"] == "t"


class TestUnroutedShare:
    @pytest.mark.parametrize(
        ("n_c", "n_t", "p_c", "p_t", "tail"),
        [
            (1_000, 1_000, 0.45, 0.47, 0.1),
            (5_000, 5_000, 0.3, 0.32, 0.025),
            (900, 700, 0.5, 0.6, 0.1),
        ],
    )
    def test_it_is_the_complement_of_the_routed_share(self, n_c, n_t, p_c, p_t, tail):
        floor = dense_min_count(tail)
        routed = routed_share(n_c, n_t, p_c, p_t, tail_alpha=tail)
        assert unrouted_share(n_c, n_t, p_c, p_t, floor=floor) == pytest.approx(
            1.0 - routed, abs=1e-12
        )

    def test_it_keeps_its_relative_precision_where_it_is_small(self):
        """Ten standard deviations short of the floor, the unrouted mass is about 1e-22: a
        complement of the routed share would round it to zero."""
        n, p, floor = 1_500, 0.4, dense_min_count(0.1)
        arm = binom.cdf(floor - 1, n, p) + binom.sf(n - floor, n, p)
        expected = 2.0 * arm - arm * arm
        assert 0.0 < expected < 1e-15
        assert unrouted_share(n, n, p, p, floor=floor) == pytest.approx(expected, rel=1e-9)
