"""Tests for `increment.breakout.estimates` - per-segment lift estimation. `run_breakout`'s
control-arm lookup is a dict keyed ONLY by metric name; one call across two segments would
silently drop a control arm."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from datetime import date
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest

from increment.breakout.estimates import (
    BreakoutEstimate,
    BreakoutEstimates,
    DailyLiftEstimate,
    DailyLiftEstimates,
    DailyMetricValue,
    DailyMetricValues,
    EstimateList,
    LiftEstimates,
    run_breakout,
    to_frame,
)
from increment.errors import CapabilityError, IncrementWarning, InvalidRequestError
from increment.estimation.armstats import centered_row_from_raw_sums
from increment.estimation.engine import Method, estimate_lift
from increment.estimation.family import bh_select, e_bh_select
from increment.estimation.inference import Normal
from increment.estimation.results import Estimate, LiftEstimate
from increment.estimation.sequential import AlwaysValid
from increment.semantics.models import ConversionMetric, MeanMetric, Measure, RatioMetric
from tests.estimation._binomial_endpoint_reference import assert_set_contains_finer_reference
from tests.warning_codes import warning_codes

# Helpers mirroring test_engine.py's arm/summary/metric fixture
# conventions, plus a breakout dimension column.


def _make_arm_row(
    n: int,
    mean: float,
    var: float,
    *,
    country: str,
    experiment_id: str = "exp1",
    metric: str = "rev",
    group_id: str = "control",
    sum_den: float | None = None,
    x_mean: float | None = None,
    x_var: float = 1.0,
    xy_cov: float = 0.0,
) -> dict:
    """Build one centered `group_summary` row with a `country` breakout column, like
    test_engine.py's `_make_arm_stats`. `sum_den` is optional; `cden2`/`cyden` land as 0.0
    (degenerate but COMPLETE) since a partial family is refused at the estimation seam."""
    sum_y = float(n) * mean
    sum_y2 = var * (n - 1) + sum_y**2 / float(n)
    x_sum = None if x_mean is None else float(n) * x_mean
    sum_x = x_sum
    sum_x2 = None if x_sum is None else x_var * (n - 1) + x_sum**2 / float(n)
    sum_xy = None if x_sum is None else xy_cov * (n - 1) + x_sum * sum_y / float(n)
    row = centered_row_from_raw_sums(
        {
            "experiment_id": experiment_id,
            "metric": metric,
            "group_id": group_id,
            "country": country,
            "n": float(n),
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_x": sum_x,
            "sum_x2": sum_x2,
            "sum_xy": sum_xy,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )
    if sum_den is not None:
        row["ref_den"] = sum_den / float(n)
        row["cden1"] = 0.0
        row["cden2"] = 0.0
        row["cyden"] = 0.0
    return row


def _mean_metric(name: str = "rev") -> MeanMetric:
    """Build a MeanMetric fixture for tests using mean metrics."""
    return MeanMetric(name=name, entity="user", fact=name)


def _windowed_mean_metric(name: str = "rev") -> MeanMetric:
    """Closed-horizon MeanMetric (`window_days` set): avoids the unrelated open-ended-metric
    warning in tests exercising the uncorrected-multiplicity warning specifically."""
    return MeanMetric(name=name, entity="user", fact=name, window_days=30)


def _ratio_metric(name: str = "ratio_rev") -> RatioMetric:
    """Build a RatioMetric fixture for tests using ratio metrics."""
    return RatioMetric(
        name=name,
        entity="user",
        numerator=Measure(fact="num"),
        denominator=Measure(fact="den"),
    )


def _conversion_metric(name: str = "conv") -> ConversionMetric:
    """Build a ConversionMetric fixture for tests using conversion metrics."""
    return ConversionMetric(name=name, entity="user", fact=name)


def _conversion_arm_row(
    n: int,
    successes: int,
    *,
    country: str,
    group_id: str,
    metric: str = "conv",
    with_covariate: bool = False,
) -> dict:
    """A centered `group_summary` row for a genuinely binary (0/1)
    conversion arm, `successes` out of `n` - the exact binomial risk-ratio method
    admits any (n, successes) pair, including n=1 and successes=0."""
    return centered_row_from_raw_sums(
        {
            "experiment_id": "exp1",
            "metric": metric,
            "group_id": group_id,
            "country": country,
            "n": float(n),
            "sum_y": float(successes),
            "sum_y2": float(successes),  # y in {0, 1}: y**2 == y
            "sum_x": 0.0 if with_covariate else None,
            "sum_x2": 0.0 if with_covariate else None,
            "sum_xy": 0.0 if with_covariate else None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


# CRITICAL: never call estimate_lift on a multi-segment summary at once.


class TestPreferredDirectionReachesTheLiftEstimate:
    def test_declared_preferred_direction_reaches_the_lift_estimate(self):
        """A metric declaring preferred_direction="decrease" must reach its
        LiftEstimate row, not just the plumbing that leaves it None -
        `prob_favorable()` raises without it."""
        from increment import readouts
        from increment.decision_wire import compiled_plan_to_json
        from increment.plan import compile_decision_plan
        from increment.semantics.design import Randomized
        from increment.sources import MomentsSource

        control = _make_arm_row(n=500, mean=10.0, var=4.0, country="US", group_id="control")
        treatment = _make_arm_row(n=500, mean=8.0, var=4.0, country="US", group_id="treatment")
        declared = MeanMetric(name="rev", entity="user", fact="rev", preferred_direction="decrease")
        undeclared = _mean_metric("orders")
        control2 = _make_arm_row(
            n=500, mean=10.0, var=4.0, country="US", metric="orders", group_id="control"
        )
        treatment2 = _make_arm_row(
            n=500, mean=8.0, var=4.0, country="US", metric="orders", group_id="treatment"
        )
        # A moments cube needs its format stamp and the plan it was reduced under.
        stamp = {
            "moments_format": 8,
            "winsor_lower_percentile": None,
            "winsor_upper_percentile": None,
            "winsor_lower_bound": None,
            "winsor_upper_bound": None,
            "winsor_n": None,
            "winsor_n_lower": None,
            "winsor_n_upper": None,
            "decision_plan": compiled_plan_to_json(
                compile_decision_plan(None, [declared, undeclared])
            ),
        }
        rows = [{**row, **stamp} for row in (control, treatment, control2, treatment2)]

        estimates = readouts.run(
            MomentsSource(
                rows,
                metrics=[declared, undeclared],
                study_id="exp1",
                design=Randomized(control_group="control"),
            )
        )
        by_metric = {e.metric: e for e in estimates}
        assert by_metric["rev"].preferred_direction == "decrease"
        assert by_metric["orders"].preferred_direction is None


class TestRunBreakoutSegmentIsolation:
    def test_two_segments_with_different_control_means_both_correct(self):
        """US: control=10, treatment=12 -> positive lift. CA: control=20, treatment=19 -> negative
        lift. A `{metric: control}` dict-key collision (one estimate_lift call across both)
        would keep only ONE segment's control arm, producing equal or mismatched results."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=12.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=20.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=19.0, var=4.0, country="CA", group_id="treatment")
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        results = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        assert len(results) == 2
        for r in results:
            assert isinstance(r, BreakoutEstimate)
            assert r.dimension == "country"
            assert r.metric == "rev"
            assert r.group_id == "treatment"
            assert r.policy_name == "default_exploratory"

        by_segment = {r.dimension_value: r for r in results}
        assert set(by_segment) == {"US", "CA"}

        # Ground truth: estimate_lift called directly on JUST that
        # segment's own two rows - what run_breakout MUST reproduce.
        us_expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([us_control, us_treatment]),
            control_group="control",
        ).results[0]
        ca_expected = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([ca_control, ca_treatment]),
            control_group="control",
        ).results[0]

        us_lift = by_segment["US"].lift
        ca_lift = by_segment["CA"].lift
        us_expected_lift = us_expected.lift
        ca_expected_lift = ca_expected.lift
        assert us_lift is not None and us_expected_lift is not None
        assert ca_lift is not None and ca_expected_lift is not None
        assert us_lift.value == us_expected_lift.value
        assert us_lift.lb == us_expected_lift.lb
        assert us_lift.ub == us_expected_lift.ub
        assert ca_lift.value == ca_expected_lift.value
        assert ca_lift.lb == ca_expected_lift.lb
        assert ca_lift.ub == ca_expected_lift.ub

        # They really do differ - opposite-signed lift, not a shared
        # value a buggy single-call implementation would inherit.
        assert us_lift.value > 0, "US treatment (12) > control (10)"
        assert ca_lift.value < 0, "CA treatment (19) < control (20)"

    def test_segment_row_order_does_not_change_either_answer(self):
        """Row arrival order (CA-then-US vs US-then-CA) must not change either segment's answer - a
        `{metric: control}` dict collision would make whichever segment is iterated LAST win."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=12.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=20.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=19.0, var=4.0, country="CA", group_id="treatment")

        forward = run_breakout(
            summary=pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment]),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        backward = run_breakout(
            summary=pd.DataFrame([ca_control, ca_treatment, us_control, us_treatment]),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        forward_by_segment = {}
        for r in forward:
            lift = r.lift
            assert lift is not None
            forward_by_segment[r.dimension_value] = lift.value
        backward_by_segment = {}
        for r in backward:
            lift = r.lift
            assert lift is not None
            backward_by_segment[r.dimension_value] = lift.value
        assert forward_by_segment == backward_by_segment

    def test_segment_output_order_is_sorted_and_stable_across_calls(self):
        """`run_breakout` partitions rows into a dict keyed by dimension value; iteration order
        used to be raw `rows` insertion order (unordered from a DuckDB/arrow fetch) - must now
        be sorted and stable."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=12.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=20.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=19.0, var=4.0, country="CA", group_id="treatment")
        mx_control = _make_arm_row(n=1200, mean=8.0, var=4.0, country="MX", group_id="control")
        mx_treatment = _make_arm_row(n=1200, mean=9.0, var=4.0, country="MX", group_id="treatment")

        # Deliberately reverse-alphabetical row arrival order.
        shuffled_rows = [
            mx_control,
            mx_treatment,
            us_control,
            us_treatment,
            ca_control,
            ca_treatment,
        ]

        first_call = run_breakout(
            summary=pd.DataFrame(shuffled_rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        # A *different* permutation, so the cross-call assertion below
        # exercises order stability rather than an identical rerun.
        second_call = run_breakout(
            summary=pd.DataFrame(list(reversed(shuffled_rows))),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        first_order = [r.dimension_value for r in first_call]
        second_order = [r.dimension_value for r in second_call]

        assert first_order == sorted(first_order)
        assert first_order == second_order


# Shape / count acceptance.


class TestRunBreakoutShape:
    def test_two_segments_return_2x_single_segment_estimate_count(self):
        """2 segments -> 2x the estimates of an unsegmented estimate_lift
        call on one segment's worth of the same-shaped data."""
        us_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="CA", group_id="treatment")
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        result = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        single_segment = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([us_control, us_treatment]),
            control_group="control",
        ).results

        assert len(result) == 2 * len(single_segment)

    def test_methods_axis_multiplies_across_segments(self):
        """methods=[...] forwards through to estimate_lift per segment:
        2 segments x 2 methods -> 4 results."""
        us_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=500, mean=10.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=500, mean=11.0, var=4.0, country="CA", group_id="treatment")
        summary = pd.DataFrame([us_control, us_treatment, ca_control, ca_treatment])

        result = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            methods=[Method(name="unadjusted"), Method(name="m2")],
        )

        assert len(result) == 4
        assert {r.method for r in result} == {"unadjusted", "m2"}
        assert {r.dimension_value for r in result} == {"US", "CA"}

    def test_duplicate_method_names_refuse_before_breakout_partition(self):
        class ExplodingSummary:
            def __iter__(self):
                raise AssertionError("breakout summary was consumed before validation")

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                summary=ExplodingSummary(),
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
                methods=[
                    Method(name="same"),
                    Method(name="same", variance_reduction="cuped"),
                ],
            )
        assert exc_info.value.code == "estimation.engine.method_names_unique"
        assert exc_info.value.context["duplicates"] == ("same",)

    def test_source_defaults_to_none_and_is_stamped_when_given(self):
        """`source` is `None` unless the caller passes one, then it's stamped onto every result -
        lets `Analysis.run_breakout` disambiguate breakouts sharing a dimension but different
        sources."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        no_source = run_breakout(
            rows, [_mean_metric()], control_group="control", dimension="country"
        )
        assert all(r.source is None for r in no_source)

        with_source = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            source="events",
        )
        assert all(r.source == "events" for r in with_source)


# Confidence intervals reflect per-segment sample size.


class TestRunBreakoutConfidenceIntervals:
    def test_ci_present_and_narrower_for_larger_segment(self):
        """Same underlying lift/variance, different n - the larger
        segment's CI must be narrower, reflecting its bigger sample."""
        big_control = _make_arm_row(n=20000, mean=10.0, var=4.0, country="BIG", group_id="control")
        big_treatment = _make_arm_row(
            n=20000, mean=11.0, var=4.0, country="BIG", group_id="treatment"
        )
        small_control = _make_arm_row(
            n=200, mean=10.0, var=4.0, country="SMALL", group_id="control"
        )
        small_treatment = _make_arm_row(
            n=200, mean=11.0, var=4.0, country="SMALL", group_id="treatment"
        )
        summary = pd.DataFrame([big_control, big_treatment, small_control, small_treatment])

        result = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        by_segment = {r.dimension_value: r for r in result}
        big = by_segment["BIG"]
        small = by_segment["SMALL"]

        for r in result:
            lift = r.lift
            assert lift is not None
            assert lift.lb is not None
            assert lift.ub is not None
            assert lift.lb < lift.value < lift.ub

        big_lift = big.lift
        small_lift = small.lift
        assert big_lift is not None and small_lift is not None
        assert big_lift.lb is not None
        assert big_lift.ub is not None
        assert small_lift.lb is not None
        assert small_lift.ub is not None
        big_width = big_lift.ub - big_lift.lb
        small_width = small_lift.ub - small_lift.lb
        assert big_width < small_width


# Bonferroni multiplicity correction across segments.


def _four_segment_summary() -> pd.DataFrame:
    """4 countries, each with a control + treatment row for one metric."""
    rows = []
    for i, country in enumerate(["US", "CA", "GB", "DE"]):
        rows.append(
            _make_arm_row(n=500, mean=10.0 + i, var=4.0, country=country, group_id="control")
        )
        rows.append(
            _make_arm_row(n=500, mean=11.0 + i, var=4.0, country=country, group_id="treatment")
        )
    return pd.DataFrame(rows)


class TestRunBreakoutBonferroniCorrection:
    def test_default_correction_is_none_and_level_unchanged(self):
        """No `correction=` passed - level stays exactly 1 - alpha (0.95),
        zero behavior change for any existing caller."""
        result = run_breakout(
            summary=_four_segment_summary(),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        assert len(result) == 4
        for r in result:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.95)

    def test_bonferroni_divides_alpha_by_segment_count(self):
        """4 distinct segments -> alpha/4 per segment -> level ==
        1 - 0.05/4, every interval strictly wider than uncorrected."""
        summary = _four_segment_summary()

        corrected = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bonferroni",
        )
        uncorrected = run_breakout(
            summary=summary,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            correction="none",
        )

        assert len(corrected) == 4
        for r in corrected:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(1 - 0.05 / 4)

        by_segment_corrected = {r.dimension_value: r for r in corrected}
        by_segment_uncorrected = {r.dimension_value: r for r in uncorrected}
        for value in by_segment_corrected:
            c = by_segment_corrected[value]
            u = by_segment_uncorrected[value]
            c_lift = c.lift
            u_lift = u.lift
            assert c_lift is not None and u_lift is not None
            assert c_lift.lb is not None and c_lift.ub is not None
            assert u_lift.lb is not None and u_lift.ub is not None
            assert (c_lift.ub - c_lift.lb) > (u_lift.ub - u_lift.lb)

    def test_single_segment_bonferroni_equals_none(self):
        """K == 1 - Bonferroni correction is a no-op, same level and
        same interval as `correction="none"`."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        corrected = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bonferroni",
        )
        uncorrected = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="none",
        )

        assert len(corrected) == len(uncorrected) == 1
        corrected_lift = corrected[0].lift
        uncorrected_lift = uncorrected[0].lift
        assert corrected_lift is not None and uncorrected_lift is not None
        assert corrected_lift.level == pytest.approx(0.95)
        assert corrected_lift.level == pytest.approx(uncorrected_lift.level)
        assert corrected_lift.lb == pytest.approx(uncorrected_lift.lb)
        assert corrected_lift.ub == pytest.approx(uncorrected_lift.ub)

    def test_empty_summary_bonferroni_does_not_divide_by_zero(self):
        """K == 0 (empty summary) must not crash: `correction="bonferroni"` degrades to the no-op
        `alpha_seg == alpha` instead of raising `ZeroDivisionError` on `alpha / 0`."""
        result = run_breakout(
            [],
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bonferroni",
        )
        assert isinstance(result, BreakoutEstimates)
        assert len(result.to_frame()) == 0

    def test_bonferroni_k_reflects_pre_skip_segment_count(self):
        """5 declared segments, 1 with no control arm (skipped): K for Bonferroni must still be 5
        (pre-skip), not 4 (post-skip survivors) - shrinking K after the skip loop
        under-corrects."""
        rows = []
        for i, country in enumerate(["US", "CA", "GB", "DE"]):
            rows.append(
                _make_arm_row(n=500, mean=10.0 + i, var=4.0, country=country, group_id="control")
            )
            rows.append(
                _make_arm_row(n=500, mean=11.0 + i, var=4.0, country=country, group_id="treatment")
            )
        # MX: treatment only, no control row - skipped, but still counts
        # toward K (5 declared segments, not 4 surviving ones).
        rows.append(_make_arm_row(n=500, mean=12.0, var=4.0, country="MX", group_id="treatment"))

        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                summary=rows,
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
                correction="bonferroni",
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)

        assert {r.dimension_value for r in result} == {"US", "CA", "GB", "DE", "MX"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["MX"].excluded == "no_control_arm"
        for value in ("US", "CA", "GB", "DE"):
            assert by_segment[value].excluded is None
            lift = by_segment[value].lift
            assert lift is not None
            assert lift.level == pytest.approx(1 - 0.05 / 5), (
                f"segment {value!r}: K must be 5 (the pre-skip segment "
                "count, MX included), not 4 -- a reordering regression "
                "that computed K after the skip loop would shrink this "
                "to 1 - 0.05/4"
            )

    def test_invalid_correction_value_rejected(self):
        """A `correction` value outside {"none", "bonferroni"} raises
        ValueError rather than silently being accepted."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                correction="holm",  # ty: ignore[invalid-argument-type]
            )
        assert exc_info.value.code == "breakout.run_breakout_correction"
        assert exc_info.value.context["correction"] == "holm"


class TestRunBreakoutRoleMetadata:
    def test_role_stamped_exploratory_regardless_of_correction(self):
        """Every run_breakout row is exploratory outside the BH path."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
        ]
        for correction in ("none", "bonferroni"):
            result = run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                correction=correction,
            )
            assert result, correction
            assert all(r.role == "exploratory" for r in result)
            assert all(r.discovery is None for r in result)


# BH multiplicity correction: one flat family across every (metric, arm,
# segment) cell this call produces.


def _bh_family_summary() -> pd.DataFrame:
    """3 metrics x 2 countries x (control, treatment): m_a is a strong lift
    in both countries, m_b is null in both, m_c is strong in US only -- a
    flat 6-cell family (metric x arm x segment) with a clean 3-selected /
    3-not-selected split, engineered so the SAME metric (m_c) has a
    different verdict per segment (proving the family spans segments, not
    just metrics)."""
    specs = [
        ("m_a", "US", 10.0, 12.0),
        ("m_a", "CA", 10.0, 12.0),
        ("m_b", "US", 10.0, 10.05),
        ("m_b", "CA", 10.0, 10.05),
        ("m_c", "US", 10.0, 10.9),
        ("m_c", "CA", 10.0, 10.05),
    ]
    rows = []
    for metric, country, control_mean, treatment_mean in specs:
        rows.append(
            _make_arm_row(
                n=500,
                mean=control_mean,
                var=4.0,
                country=country,
                metric=metric,
                group_id="control",
            )
        )
        rows.append(
            _make_arm_row(
                n=500,
                mean=treatment_mean,
                var=4.0,
                country=country,
                metric=metric,
                group_id="treatment",
            )
        )
    return pd.DataFrame(rows)


def _p_value_from_estimate(lift: Estimate) -> float:
    """Two-sided p-value from an Estimate's log-scale sufficient
    statistics -- `BreakoutEstimate` carries no `.p_value()` method (unlike
    `LiftEstimate`), so this reconstructs the same flat-prior z-test
    `LiftEstimate.p_value()` computes for a non-cluster-robust row."""
    from scipy.stats import norm as _norm_dist

    z = cast(float, lift.log_mean) / cast(float, lift.log_se)
    return float(2.0 * _norm_dist.sf(abs(z)))


class TestRunBreakoutBHCorrection:
    """`correction="bh"`: one flat BH family across every (metric, arm,
    segment) cell this call produces, with FCR-adjusted intervals on the
    selected cells (fixed-horizon) or e-BH discovery with untouched
    confidence-sequence intervals (AlwaysValid)."""

    def test_bh_refuses_informative_prior(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                _bh_family_summary(),
                [_mean_metric("m_a")],
                control_group="control",
                dimension="country",
                prior=Normal(mu=0.0, sigma=0.1),
                correction="bh",
            )
        assert exc_info.value.code == "breakout.run_breakout_bh_excludes_prior"

    def test_bh_discovery_matches_hand_bh_across_metrics_and_segments(self):
        summary = _bh_family_summary()
        metrics = [_mean_metric("m_a"), _mean_metric("m_b"), _mean_metric("m_c")]
        q = 0.05

        result = run_breakout(
            summary=summary,
            metrics=metrics,
            control_group="control",
            dimension="country",
            correction="bh",
            q=q,
        )
        real = [r for r in result if r.excluded is None]
        assert len(real) == 6

        p_values = []
        for r in real:
            lift = r.lift
            assert lift is not None
            p_values.append(_p_value_from_estimate(lift))
        selected_idx, expected_selected_alpha = bh_select(p_values, q)
        selected_set = set(selected_idx)
        assert [r.discovery for r in real] == [i in selected_set for i in range(len(real))]

        selected_cells = {(real[i].metric, real[i].dimension_value) for i in selected_set}
        assert selected_cells == {("m_a", "US"), ("m_a", "CA"), ("m_c", "US")}

        for i, r in enumerate(real):
            assert r.role == "exploratory"
            lift = r.lift
            assert lift is not None
            if i in selected_set:
                assert lift.alpha == expected_selected_alpha
            else:
                assert lift.alpha == pytest.approx(0.05)  # nominal, default alpha=0.05

    def test_bh_sparse_segment_failure_aborts_complete_family(self):
        """A missing segment control arm is a failed family hypothesis."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 11.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 12.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            with pytest.raises(CapabilityError) as raised:
                run_breakout(
                    rows,
                    [_mean_metric()],
                    control_group="control",
                    dimension="country",
                    correction="bh",
                )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)
        error = raised.value
        assert error.code == "family.evidence.incomplete"
        assert any("MX" in repr(key) for key in cast("Sequence[Any]", error.context["failed"]))

    def test_bh_family_dedups_by_method(self):
        """2 methods x 1 metric x 2 segments: family size must count
        (metric, arm, segment) cells (2), not (metric, method, arm,
        segment) rows (4). The canonical unadjusted US row is significant,
        while contradictory CUPED evidence must remain uncredited."""
        rows = []
        for country, control_mean, treatment_mean in (("US", 10.0, 12.0), ("CA", 10.0, 10.05)):
            rows.append(
                _make_arm_row(
                    500,
                    control_mean,
                    4.0,
                    country=country,
                    group_id="control",
                    x_mean=0.0,
                    xy_cov=1.0,
                )
            )
            rows.append(
                _make_arm_row(
                    500,
                    treatment_mean,
                    4.0,
                    country=country,
                    group_id="treatment",
                    x_mean=2.0,
                    xy_cov=1.0,
                )
            )

        result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
            correction="bh",
            q=0.05,
        )
        real = [r for r in result if r.excluded is None]
        assert len(real) == 4
        assert {r.method for r in real} == {"unadjusted", "cuped"}

        by_segment: dict[str, list] = {}
        for r in real:
            by_segment.setdefault(r.dimension_value, []).append(r)

        assert by_segment["US"][0].discovery is True
        assert next(r for r in by_segment["US"] if r.method == "cuped").discovery is None
        assert by_segment["CA"][0].discovery is False

        decision_us = next(r for r in by_segment["US"] if r.method == "unadjusted")
        sensitivity_us = next(r for r in by_segment["US"] if r.method == "cuped")
        # BH re-inverts a selected cell at q * rank / family size over the two
        # (metric, arm, segment) cells -- a 4-row family would give 0.0125.
        assert decision_us.require_lift().alpha == pytest.approx(0.05 * 1 / 2)
        assert sensitivity_us.require_lift().alpha == pytest.approx(0.05)
        for r in by_segment["CA"]:
            assert r.require_lift().alpha == pytest.approx(0.05)

    def test_bh_conversion_with_cuped_first_reestimates_unadjusted_decision(self):
        rows = [
            _conversion_arm_row(
                1000, successes, country="US", group_id=group_id, with_covariate=True
            )
            for group_id, successes in (("control", 100), ("treatment", 300))
        ]
        for row in rows:
            row["cx2"] = 1000.0

        result = run_breakout(
            rows,
            [_conversion_metric()],
            control_group="control",
            dimension="country",
            methods=[
                Method(name="cuped", variance_reduction="cuped"),
                Method(name="unadjusted"),
            ],
            correction="bh",
            q=0.05,
        )

        by_method = {row.method: row for row in result}
        assert by_method["unadjusted"].method_role == "decision"
        assert by_method["unadjusted"].discovery is True
        assert by_method["unadjusted"].reference_kind == "binomial"
        assert by_method["cuped"].method_role == "sensitivity"
        assert by_method["cuped"].discovery is None
        # The sensitivity row keeps its own moment-based reference rather than
        # borrowing the decision row's exact-binomial one; CUPED estimates its
        # variance from the arms, so that reference is Welch t.
        assert by_method["cuped"].reference_kind == "t"
        assert by_method["cuped"].reference_df is not None

    @pytest.mark.slow
    def test_bh_always_valid_uses_real_retained_evidence_and_selected_inversion(self):
        from fractions import Fraction

        from increment import SequentialCell, SequentialRegistration
        from increment.frame import MetricSpec
        from increment.semantics.models import ConversionMetric
        from tests.sequential_cases import capture, registration

        specs = [MetricSpec(name=name, type="conversion") for name in ("m_a", "m_b", "m_c")]
        cells = tuple(
            SequentialCell(
                metric=spec.name,
                group_id="treatment",
                segment=(("country", country),),
                family=True,
            )
            for spec in specs
            for country in ("US", "CA")
        )
        base = registration("bernoulli")
        reg = SequentialRegistration.model_validate(
            {
                **base.model_dump(),
                "models": tuple(
                    base.models[0].model_copy(update={"metric": spec.name}) for spec in specs
                ),
                "roster": cells,
                "q": Fraction(0.05),
            }
        )
        rows = []
        for country in ("US", "CA"):
            for i in range(500):
                for arm in ("control", "treatment"):
                    base_value = i % 2
                    rows.append(
                        {
                            "unit_id": f"{country}-{i:04d}-{arm}",
                            "group_id": arm,
                            "segments": {"country": country},
                            "values": {
                                spec.name: 1
                                if arm == "treatment" and spec.name == "m_a"
                                else base_value
                                for spec in specs
                            },
                        }
                    )
        result = run_breakout(
            capture(reg, rows),
            [ConversionMetric(name=spec.name, entity="user", fact="events") for spec in specs],
            control_group="control",
            dimension="country",
            correction="bh",
            q=0.05,
            inference=AlwaysValid(registration=reg),
        )
        assert len(result) == 6
        from increment.estimation.sequential_result import SequentialInferenceResult

        logs = []
        for row in result:
            assert isinstance(row.sequential_result, SequentialInferenceResult)
            logs.append(row.sequential_result.log_e)
        selected = set(e_bh_select(logs, 0.05))
        assert selected and len(selected) < len(result)
        for i, row in enumerate(result):
            assert row.sequential_result is not None
            assert row.discovery == (i in selected)
            expected = (
                min(Fraction(0.05) * len(selected) / 6, Fraction(1, 20))
                if i in selected
                else Fraction(1, 20)
            )
            assert row.sequential_result.bounds.alpha == expected
            assert row.role == "exploratory"
            assert row.family_guarantee == "finite_sample"
            assert row.family_nominal_alpha == pytest.approx(0.05)
        reinverted = next(row for i, row in enumerate(result) if i in selected)
        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate({**reinverted.model_dump(), "discovery": False})
        assert raised.value.code == "sequential.source.invalid"

    def test_bh_zero_selection_all_discovery_false_no_reestimation(self):
        """Every cell null -> R=0: no crash, no re-estimation call, every
        real row discovery=False at the nominal level."""
        rows = []
        for country in ("US", "CA"):
            rows.append(_make_arm_row(500, 10.0, 4.0, country=country, group_id="control"))
            rows.append(_make_arm_row(500, 10.01, 4.0, country=country, group_id="treatment"))

        result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bh",
            q=0.05,
        )
        real = [r for r in result if r.excluded is None]
        assert len(real) == 2
        assert all(r.discovery is False for r in real)
        for r in real:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.95)

    def test_bh_empty_summary_does_not_crash(self):
        result = run_breakout(
            [],
            [_mean_metric()],
            control_group="control",
            dimension="country",
            correction="bh",
        )
        assert isinstance(result, BreakoutEstimates)
        assert len(result.to_frame()) == 0

    def test_bh_preserves_segment_and_declaration_order(self):
        """Family selection must not reorder the result: rows still come
        back grouped by segment (alphabetical) then declaration order,
        matching every other correction mode's output shape."""
        summary = _bh_family_summary()
        metrics = [_mean_metric("m_a"), _mean_metric("m_b"), _mean_metric("m_c")]
        result = run_breakout(
            summary=summary,
            metrics=metrics,
            control_group="control",
            dimension="country",
            correction="bh",
            q=0.05,
        )
        real = [r for r in result if r.excluded is None]
        assert [(r.dimension_value, r.metric) for r in real] == [
            ("CA", "m_a"),
            ("CA", "m_b"),
            ("CA", "m_c"),
            ("US", "m_a"),
            ("US", "m_b"),
            ("US", "m_c"),
        ]

    def test_bh_disjoint_segment_coverage_marks_absent_metric_unavailable(self):
        """Two metrics with genuinely disjoint segment coverage: m_a is
        present in both US and CA, m_b only in US. Combining both
        metrics' moments into ONE run_breakout call (as breakout()'s
        "bh" path does) must represent (m_b, CA) as an excluded row
        (0jm6) rather than omitting it silently - a consumer computing
        segment_heterogeneity over this output needs to see that m_b's
        segment coverage is incomplete, not mistake two real rows for a
        complete family. BH family selection still scopes to the
        (metric, arm, segment) cells actually queried in each segment,
        so the real 3-cell (m_a x US/CA, m_b x US) family is unaffected
        and no CapabilityError fires."""
        rows = [
            _make_arm_row(500, 10.0, 4.0, country="US", metric="m_a", group_id="control"),
            _make_arm_row(500, 11.0, 4.0, country="US", metric="m_a", group_id="treatment"),
            _make_arm_row(500, 10.0, 4.0, country="CA", metric="m_a", group_id="control"),
            _make_arm_row(500, 10.5, 4.0, country="CA", metric="m_a", group_id="treatment"),
            _make_arm_row(500, 10.0, 4.0, country="US", metric="m_b", group_id="control"),
            _make_arm_row(500, 10.9, 4.0, country="US", metric="m_b", group_id="treatment"),
        ]
        metrics = [_mean_metric("m_a"), _mean_metric("m_b")]

        with pytest.warns(IncrementWarning) as rec:
            combined = run_breakout(
                summary=pd.DataFrame(rows),
                metrics=metrics,
                control_group="control",
                dimension="country",
                correction="bh",
                q=0.05,
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        cells = {(r.metric, r.dimension_value) for r in combined}
        assert cells == {("m_a", "US"), ("m_a", "CA"), ("m_b", "US"), ("m_b", "CA")}

        absent = next(r for r in combined if r.metric == "m_b" and r.dimension_value == "CA")
        assert absent.excluded == "no_control_arm"
        assert absent.lift is None
        assert absent.discovery is None  # not part of the BH family
        assert absent.role == "exploratory"

        real = [r for r in combined if r.excluded is None]
        assert len(real) == 3
        assert all(r.role == "exploratory" for r in real)
        assert all(r.discovery is not None for r in real)


# alternative= forwarding: one-sided per-segment testing.


class TestRunBreakoutAlternative:
    def test_default_alternative_is_two_sided_and_unchanged(self):
        """Omitting `alternative=` reproduces the exact interval `run_breakout` always produced,
        and every row is labeled "two-sided"."""
        rows = [
            _make_arm_row(500, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(500, 11.0, 4.0, country="US", group_id="treatment"),
        ]

        implicit = run_breakout(
            rows, [_mean_metric()], control_group="control", dimension="country"
        )
        explicit = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            alternative="two-sided",
        )

        assert [r.alternative for r in implicit] == ["two-sided"]
        for a, b in zip(implicit, explicit, strict=True):
            a_lift = a.lift
            b_lift = b.lift
            assert a_lift is not None and b_lift is not None
            assert a_lift.value == b_lift.value
            assert a_lift.lb == b_lift.lb
            assert a_lift.ub == b_lift.ub
            assert a_lift.level == b_lift.level

    def test_one_sided_greater_guardrail_matches_doubled_alpha_two_sided_interval(self):
        """One-sided `alternative="greater"` at `alpha=0.05` reproduces the SAME interval as
        two-sided at `alpha=0.10` - the alpha-doubling identity `infer_lift` relies on, just
        relabeled."""
        rows = [
            _make_arm_row(2000, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(2000, 12.0, 4.0, country="US", group_id="treatment"),
        ]

        one_sided = run_breakout(
            rows,
            [_mean_metric()],
            control_group="control",
            dimension="country",
            alpha=0.05,
            alternative="greater",
        )
        two_sided = run_breakout(
            rows, [_mean_metric()], control_group="control", dimension="country", alpha=0.10
        )

        assert one_sided[0].alternative == "greater"
        assert two_sided[0].alternative == "two-sided"
        one_sided_lift = one_sided[0].lift
        two_sided_lift = two_sided[0].lift
        assert one_sided_lift is not None and two_sided_lift is not None
        assert one_sided_lift.level == pytest.approx(0.90)
        assert one_sided_lift.lb == pytest.approx(two_sided_lift.lb)
        assert one_sided_lift.ub == pytest.approx(two_sided_lift.ub)

    def test_unestimable_segment_nan_row_carries_requested_alternative(self):
        """A segment excluded before `estimate_lift` runs (no control arm) still reports the
        REQUESTED `alternative` on its dense NaN row, matching a real row's label."""
        rows = [_make_arm_row(50, 10.0, 4.0, country="US", group_id="treatment")]

        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric()],
                control_group="control",
                dimension="country",
                alternative="less",
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)

        assert len(result) == 1
        assert result[0].excluded == "no_control_arm"
        assert result[0].lift is None
        assert result[0].alternative == "less"


class TestRunBreakoutAlwaysValidInference:
    @pytest.mark.parametrize("correction", ["none", "bonferroni", "bh"])
    def test_registered_gaussian_segments_refuse_before_public_breakout(self, correction):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        segments = tuple((("country", country),) for country in ("US", "CA", "GB", "DE"))
        snapshot, metrics, policy = raw_gaussian(
            n=4,
            segments=segments,
            alpha=0.05 / 4 if correction == "bonferroni" else 0.05,
            family=correction == "bh",
        )
        with pytest.raises(CapabilityError) as raised:
            run_breakout(
                snapshot,
                metrics,
                control_group="control",
                dimension="country",
                inference=policy,
                correction=correction,
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_moments_cannot_be_reinterpreted_as_raw_likelihood(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        _, metrics, policy = raw_gaussian()
        with pytest.raises(CapabilityError) as raised:
            run_breakout(
                _four_segment_summary(),
                metrics,
                control_group="control",
                dimension="country",
                inference=policy,
            )
        assert raised.value.code == "sequential.source.invalid"


# Input format equivalence + defensive errors (mirrors
# test_engine.py's TestInputFormats).


class TestRunBreakoutInputFormats:
    def test_pandas_pyarrow_and_dict_rows_agree(self):
        """run_breakout gives numerically identical results for a pandas DataFrame, a pyarrow
        Table, and a plain list[dict] - mirrors estimate_lift's own format-equivalence contract."""
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=2000, mean=11.0, var=4.0, country="US", group_id="treatment")
        ca_control = _make_arm_row(n=1500, mean=9.0, var=4.0, country="CA", group_id="control")
        ca_treatment = _make_arm_row(n=1500, mean=10.5, var=4.0, country="CA", group_id="treatment")
        rows = [us_control, us_treatment, ca_control, ca_treatment]

        pandas_result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        pyarrow_result = run_breakout(
            summary=pa.Table.from_pylist(rows),
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )
        dict_result = run_breakout(
            summary=rows,
            metrics=[_mean_metric()],
            control_group="control",
            dimension="country",
        )

        def _by_segment(results: list[BreakoutEstimate]) -> dict:
            out = {}
            for r in results:
                lift = r.lift
                assert lift is not None
                out[r.dimension_value] = (lift.value, lift.lb, lift.ub)
            return out

        assert len(pandas_result) == len(pyarrow_result) == len(dict_result) == 2
        assert _by_segment(pandas_result) == _by_segment(pyarrow_result) == _by_segment(dict_result)

    def test_dimension_column_missing_from_dataframe_raises(self):
        """A DataFrame missing the requested dimension column raises a
        clear ValueError, not a KeyError deep in the grouping loop."""
        row = _make_arm_row(n=100, mean=10.0, var=4.0, country="US", group_id="control")
        df = pd.DataFrame([row]).drop(columns=["country"])

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                summary=df,
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
            )
        assert exc_info.value.code == "breakout.run_breakout_dimension"
        assert exc_info.value.context["dimension"] == "country"

    def test_dict_rows_missing_dimension_key_raises(self):
        """Mirrors estimate_lift's `test_dict_rows_missing_column_raises`:
        no schema to pre-validate, so it fails with a KeyError at first access."""
        row = _make_arm_row(n=100, mean=10.0, var=4.0, country="US", group_id="control")
        del row["country"]

        with pytest.raises(KeyError):
            run_breakout(
                summary=[row],
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
            )

    def test_missing_ref_y_column_raises_value_error_not_key_error(self):
        """A `summary` DataFrame missing `ref_y` must fail with a
        descriptive ValueError naming the column, not a bare KeyError."""
        row = _make_arm_row(n=100, mean=10.0, var=4.0, country="US", group_id="control")
        df = pd.DataFrame([row]).drop(columns=["ref_y"])

        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                summary=df,
                metrics=[_mean_metric()],
                control_group="control",
                dimension="country",
            )
        assert exc_info.value.code == "breakout.run_breakout_group"
        assert exc_info.value.context["missing"] == ("ref_y",)

    def test_methods_empty_list_raises_value_error(self):
        """`methods=[]` (distinct from `None`, which resolves to `[Method(name="unadjusted")]`)
        used to silently produce zero rows plus a factually wrong "no data" warning. Now raises
        up front."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods=[],
            )
        assert exc_info.value.code == "breakout.run_breakout_methods"

    def test_methods_none_still_uses_the_unadjusted_default(self):
        """`methods=None` is NOT the same as `methods=[]` - it resolves
        to the documented default, not "zero methods."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country", methods=None
        )
        assert [r.method for r in result] == ["unadjusted"]

    def test_empty_methods_by_metric_does_not_emit_unestimable_rows(self):
        """An explicitly disabled metric must not regain the default unadjusted method on an
        unestimable cell."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods_by_metric={"rev": []},
            )
        assert "breakout.estimates.row_nonpositive_mean" in warning_codes(rec)
        assert result == []

    def test_duplicate_segment_metric_arm_row_raises(self):
        """Two rows for the same (segment, metric, arm) cell must refuse
        rather than silently discard one row's units or double-count the
        segment downstream."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 13.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_breakout_duplicate"
        assert exc_info.value.context["dimension"] == "country"
        assert exc_info.value.context["value"] == "US"
        assert exc_info.value.context["metric"] == "rev"
        assert exc_info.value.context["group_id"] == "treatment"

    def test_undeclared_metric_row_raises(self):
        """A row naming a metric absent from the declared `metrics=` list
        must refuse rather than silently estimating an undeclared metric."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", metric="rev", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", metric="rev", group_id="treatment"),
            _make_arm_row(50, 12.0, 4.0, country="US", metric="unknown", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.metric_found_group"
        assert exc_info.value.context["unknown_metrics"] == ("unknown",)
        assert exc_info.value.context["declared_names"] == ("rev",)


# BreakoutEstimate model (frozen-model style, mirrors
# test_armstats.py/test_inference.py).


class TestBreakoutEstimate:
    def test_frozen(self):
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
        )
        with pytest.raises((TypeError, ValueError)):
            est.dimension_value = "CA"  # ty: ignore[invalid-assignment]  # proving frozen at runtime

    def test_basic_construction(self):
        lift = Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95)
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=lift,
        )
        assert est.metric == "rev"
        assert est.group_id == "treatment"
        assert est.method == "unadjusted"
        assert est.dimension == "country"
        assert est.dimension_value == "US"
        assert est.lift is lift

    def test_abs_diff_abs_se_default_none(self):
        """abs_diff/abs_se default to None until a constructor populates them."""
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
        )
        assert est.abs_diff is None
        assert est.abs_se is None

    def test_abs_diff_abs_se_settable(self):
        est = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
            abs_diff=1.5,
            abs_se=0.3,
        )
        assert est.abs_diff == 1.5
        assert est.abs_se == 0.3

    @pytest.mark.parametrize("field", ["abs_diff", "abs_se", "null_abs", "null_lift"])
    def test_nonfinite_scalar_fields_are_rejected(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: math.inf}
        with pytest.raises(InvalidRequestError) as raised:
            BreakoutEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                dimension="country",
                dimension_value="US",
                lift=Estimate(value=0.2),
                **kwargs,
            )
        assert raised.value.code == "model.field.nonfinite"
        assert raised.value.context["field"] == field

    def test_missing_required_field_raises(self):
        with pytest.raises(InvalidRequestError) as raised:
            BreakoutEstimate(
                group_id="treatment",
                method="unadjusted",
                dimension="country",
                dimension_value="US",
                lift=Estimate(value=0.1),
            )  # ty: ignore[missing-argument]  # proving `metric` is required
        assert raised.value.code == "model.field.missing"

    def test_neither_lift_nor_excluded_set_raises_coded_error(self):
        """The buried coded refusal must surface with its own `.code`, not a
        bare pydantic `ValidationError`."""
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(
                metric="rev",
                group_id="treatment",
                method="unadjusted",
                method_role="decision",
                dimension="country",
                dimension_value="US",
                lift=None,
            )
        assert exc_info.value.code == "breakout.breakout.exactly_one_lift"

    def test_round_trips_on_a_real_run_breakout_result(self):
        """model_dump_json() round-trips on a REAL run_breakout output - Estimate's dropped
        `posterior` field carries through the whole breakout wrapper, not just LiftEstimate in
        isolation."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        results = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert len(results) == 1
        est = results[0]
        round_tripped = BreakoutEstimate.model_validate_json(est.model_dump_json())
        assert round_tripped == est

    def test_breakout_estimate_row_preserves_a_welch_reference_through_round_trip(self):
        # Unequal arm counts give the unadjusted lift a Welch t reference
        # whose df must survive serialization, not be reconstructed as Normal.
        rows = [
            _make_arm_row(n=5, mean=10.0, var=4.0, country="US", group_id="control"),
            _make_arm_row(n=4, mean=11.0, var=4.0, country="US", group_id="treatment"),
        ]
        (row,) = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert row.dof is None
        assert row.reference_kind == "t"
        assert row.reference_df is not None
        round_tripped = BreakoutEstimate.model_validate(row.model_dump())

        assert round_tripped.dof is None
        assert round_tripped.reference_kind == "t"
        assert round_tripped.reference_df == pytest.approx(row.reference_df)


class TestBreakoutEstimateReferenceValidation:
    @staticmethod
    def _sequential_kwargs(**overrides: Any) -> dict[str, Any]:
        from increment import estimate_sequential
        from tests.sequential_cases import registered_bernoulli

        snapshot, policy = registered_bernoulli(n=4, segments=((("country", "US"),),))
        row = estimate_sequential(snapshot, policy).results[0]
        row = BreakoutEstimate.model_validate(
            {**row.model_dump(), "dimension": "country", "dimension_value": "US"}
        )
        return {**row.model_dump(), **overrides}

    @staticmethod
    def _kwargs(**overrides: Any) -> dict[str, Any]:
        values: dict[str, Any] = {
            "metric": "rev",
            "group_id": "treatment",
            "method": "unadjusted",
            "method_role": "decision",
            "dimension": "country",
            "dimension_value": "US",
            "lift": Estimate(value=0.1),
        }
        values.update(overrides)
        return values

    def test_normal_reference_is_the_default(self):
        row = BreakoutEstimate(**self._kwargs())
        assert row.reference_kind == "normal"
        assert row.reference_df is None

    def test_t_reference_round_trips_through_model_validate(self):
        row = BreakoutEstimate(**self._kwargs(dof=5.0, reference_kind="t", reference_df=5.0))
        again = BreakoutEstimate.model_validate(row.model_dump())
        assert again.reference_kind == "t"
        assert again.reference_df == 5.0

    def test_model_validate_infers_t_reference_from_legacy_dof(self):
        row = BreakoutEstimate.model_validate(self._kwargs(dof=8.0))
        assert row.reference_kind == "t"
        assert row.reference_df == 8.0

    def test_explicit_normal_reference_with_dof_is_not_legacy_inferred(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate.model_validate(self._kwargs(dof=8.0, reference_kind="normal"))
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_explicit_none_reference_df_with_dof_is_not_legacy_inferred(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate.model_validate(self._kwargs(dof=8.0, reference_df=None))
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_constructor_rejects_t_reference_without_df(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**self._kwargs(reference_kind="t", reference_df=None))
        assert exc_info.value.code == "estimation.results.lift.reference_df_required_for_t"

    def test_constructor_rejects_reference_df_on_normal_reference(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**self._kwargs(reference_kind="normal", reference_df=5.0))
        assert exc_info.value.code == "estimation.results.lift.reference_df_set_for_normal"

    def test_model_validate_rejects_dof_reference_df_mismatch(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate.model_validate(
                self._kwargs(dof=5.0, reference_kind="t", reference_df=8.0)
            )
        assert exc_info.value.code == "estimation.results.lift.dof_reference_df_mismatch"

    def test_legacy_sequential_payload_refuses_without_raw_checkpoint(self):
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate(self._kwargs(inference="always_valid"))
        assert raised.value.code == "sequential.continuation.legacy"

    def test_public_replay_rejects_gaussian_checkpoint(self):
        from increment.errors import CapabilityError
        from increment.estimation.sequential_runtime import (
            _evaluate_sequential_diagnostic,
            display_estimate,
        )
        from tests.sequential_cases import raw_gaussian

        # Only the diagnostic evaluator builds a non-admitted gaussian
        # checkpoint; estimate_sequential refuses the law before evaluating it.
        snapshot, _, policy = raw_gaussian(segments=((("country", "US"),),), n=4)
        result = _evaluate_sequential_diagnostic(snapshot, policy)[0]
        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate(
                self._kwargs(
                    inference="always_valid",
                    reference_kind="sequential",
                    sequential_result=result,
                    lift=display_estimate(result),
                )
            )
        assert raised.value.code == "sequential.route.unsupported"

    @pytest.mark.parametrize("relabel", [False, True])
    def test_public_replay_preserves_bernoulli_reference_label(self, relabel):
        from increment.errors import CapabilityError

        row = BreakoutEstimate.model_validate(self._sequential_kwargs())
        if not relabel:
            assert BreakoutEstimate.model_validate_json(row.model_dump_json()) == row
            return
        payload = row.model_dump()
        payload.update(inference="fixed", reference_kind="normal")
        with pytest.raises(CapabilityError) as caught:
            BreakoutEstimate.model_validate(payload)
        assert caught.value.code == "sequential.source.invalid"

    def test_constructor_rejects_normal_reference_for_sequential_inference(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**self._kwargs(inference="always_valid", reference_kind="normal"))
        assert exc_info.value.code == ("estimation.results.lift.inference_reference_kind_mismatch")

    def test_constructor_rejects_sequential_reference_for_fixed_inference(self):
        kwargs = self._sequential_kwargs(inference="fixed")
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**kwargs)
        assert exc_info.value.code == ("estimation.results.lift.inference_reference_kind_mismatch")

    def test_constructor_rejects_reference_df_on_sequential_reference(self):
        kwargs = self._sequential_kwargs(reference_df=5.0)
        with pytest.raises(InvalidRequestError) as exc_info:
            BreakoutEstimate(**kwargs)
        assert exc_info.value.code == "estimation.results.lift.reference_df_set_for_normal"

    @pytest.mark.parametrize(
        "field",
        ["relative_confidence_set", "relative_unavailable_reason", "abs_diff", "null_abs"],
    )
    def test_sequential_checkpoint_rejects_mixed_authoritative_payload(self, field):
        """A checkpoint-backed sequential view cannot carry fixed/joint sidecars."""
        from increment.errors import CapabilityError
        from increment.estimation.results import JointContrastReference, RelativeConfidenceSet

        payload = self._sequential_kwargs()
        if field == "relative_confidence_set":
            payload[field] = RelativeConfidenceSet(
                reference=JointContrastReference(a=1.0, c=1.0, var_a=1.0, var_c=1.0, cov_ac=0.0),
                alpha=0.05,
            )
        elif field == "relative_unavailable_reason":
            payload[field] = "joint_covariance_indefinite"
        else:
            payload[field] = 0.0 if field == "null_abs" else 1.0

        with pytest.raises(CapabilityError) as raised:
            BreakoutEstimate.model_validate(payload)
        assert raised.value.code == "sequential.source.invalid"


class TestRunBreakoutReliabilityFloor:
    """`reliability_floor` (default 50): a real estimate whose control or treatment arm has fewer
    units than the floor is still returned, not omitted or excluded, but flagged
    `low_reliability=True`."""

    def _mixed_n_summary(self, *, small_n: int, large_n: int) -> pd.DataFrame:
        """Two segments: 'thin' has `small_n` units per arm, 'thick' has `large_n` - both above the
        few_units gate (n>=2) and estimable, differing only in whether they clear the
        reliability floor."""
        rows = [
            _make_arm_row(n=small_n, mean=10.0, var=4.0, country="thin", group_id="control"),
            _make_arm_row(n=small_n, mean=11.0, var=4.0, country="thin", group_id="treatment"),
            _make_arm_row(n=large_n, mean=10.0, var=4.0, country="thick", group_id="control"),
            _make_arm_row(n=large_n, mean=11.0, var=4.0, country="thick", group_id="treatment"),
        ]
        return pd.DataFrame(rows)

    def test_below_floor_segment_flagged_above_floor_not(self):
        result = run_breakout(
            summary=self._mixed_n_summary(small_n=20, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["thin"].low_reliability is True
        assert by_segment["thick"].low_reliability is False
        # Both are real, estimated rows - the floor does not exclude.
        assert by_segment["thin"].excluded is None
        assert by_segment["thin"].lift is not None

    def test_floor_is_a_parameter_not_a_literal(self):
        """The same n=20 segment is NOT flagged once the caller lowers
        the floor - proves it's a real parameter, not a hardcoded constant."""
        result = run_breakout(
            summary=self._mixed_n_summary(small_n=20, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
            reliability_floor=10,
        )
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["thin"].low_reliability is False

    def test_default_floor_is_50(self):
        """n=49 flags, n=50 does not: pins the documented default via the boundary itself, since
        `reliability_floor` is a function parameter with no field to read a default off of."""
        just_below = run_breakout(
            summary=self._mixed_n_summary(small_n=49, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        just_at = run_breakout(
            summary=self._mixed_n_summary(small_n=50, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        assert {r.dimension_value: r for r in just_below}["thin"].low_reliability is True
        assert {r.dimension_value: r for r in just_at}["thin"].low_reliability is False

    def test_asymmetric_arm_below_floor_flags_even_when_other_arm_is_large(self):
        """The floor check is an OR across control and treatment arm sizes: a LARGE control with a
        thin treatment (or vice versa) must still flag, not just when both arms are thin
        together."""
        control_thin_rows = [
            _make_arm_row(n=20, mean=10.0, var=4.0, country="control_thin", group_id="control"),
            _make_arm_row(n=200, mean=11.0, var=4.0, country="control_thin", group_id="treatment"),
        ]
        treatment_thin_rows = [
            _make_arm_row(n=200, mean=10.0, var=4.0, country="treatment_thin", group_id="control"),
            _make_arm_row(n=20, mean=11.0, var=4.0, country="treatment_thin", group_id="treatment"),
        ]
        result = run_breakout(
            summary=pd.DataFrame(control_thin_rows + treatment_thin_rows),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["control_thin"].low_reliability is True
        assert by_segment["treatment_thin"].low_reliability is True

    def test_low_reliability_suppresses_stat_sig_in_readout(self):
        """tables.py's stat_sig is forced False for a flagged row even when the interval would
        otherwise exclude the null - a tight, significant-looking n=20 interval must not render
        as stat_sig."""
        pytest.importorskip("coeftable")
        from increment.tables import estimates_to_readout

        result = run_breakout(
            summary=self._mixed_n_summary(small_n=20, large_n=200),
            metrics=[_windowed_mean_metric()],
            control_group="control",
            dimension="country",
        )
        rows = {r["segment"]: r for r in estimates_to_readout(result)}
        assert rows["thin"]["low_reliability"] is True
        assert rows["thin"]["stat_sig"] is False
        assert rows["thick"]["low_reliability"] is False


class TestRunBreakoutArmCounts:
    """`n_treat`/`n_control`: per-(segment, metric) arm unit counts,
    populated on every real estimate, `None` on an excluded row."""

    def test_real_row_carries_matching_arm_sizes(self):
        us_control = _make_arm_row(n=2000, mean=10.0, var=4.0, country="US", group_id="control")
        us_treatment = _make_arm_row(n=1500, mean=12.0, var=4.0, country="US", group_id="treatment")
        result = run_breakout(
            summary=pd.DataFrame([us_control, us_treatment]),
            metrics=[_mean_metric("rev")],
            control_group="control",
            dimension="country",
        )
        row = result[0]
        assert row.excluded is None
        assert row.n_treat == pytest.approx(1500.0)
        assert row.n_control == pytest.approx(2000.0)

    def test_excluded_row_has_none_arm_counts(self):
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: treatment only, no control row at all - excluded.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                summary=pd.DataFrame(rows),
                metrics=[_mean_metric("rev")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["MX"].excluded is not None
        assert by_segment["MX"].n_treat is None
        assert by_segment["MX"].n_control is None
        assert by_segment["US"].n_treat == pytest.approx(50.0)
        assert by_segment["US"].n_control == pytest.approx(50.0)

    def test_to_frame_carries_arm_counts_as_float_columns(self):
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            summary=pd.DataFrame(rows),
            metrics=[_mean_metric("rev")],
            control_group="control",
            dimension="country",
        )
        frame = cast(pd.DataFrame, result.to_frame())
        assert "n_treat" in frame.columns
        assert "n_control" in frame.columns
        assert frame.loc[0, "n_treat"] == pytest.approx(50.0)
        assert frame.loc[0, "n_control"] == pytest.approx(50.0)


# Exception handling: only a missing control arm, a non-positive-mean row, or
# estimate_lift's Degenerate-data/Log-delta guards are skipped; all else propagates.


class TestRunBreakoutErrorPropagation:
    def test_missing_control_arm_segment_is_skipped_with_warning(self):
        """The one condition run_breakout swallows into a NaN row: a segment
        with no control-group rows at all for the dimension value."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: treatment only, no control row at all.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.slice_no_control_arm" in warning_codes(rec)
        assert {r.dimension_value for r in result} == {"US", "MX"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded is None
        assert by_segment["MX"].excluded == "no_control_arm"
        assert by_segment["MX"].lift is None

    def test_observational_method_name_refused_on_an_unestimable_segment(self):
        """The mislabel refusal is an invariant: a segment with no control arm (which run_breakout
        swallows into a NaN row) must refuse an observational Method just as loudly as an
        estimable one."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: treatment only, no control row at all - unestimable.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods=[Method(name="iptw")],
            )
        assert exc_info.value.code == "estimation.engine.method_name_observational"
        assert exc_info.value.context["m"] == "iptw"

    def test_unrelated_error_propagates_not_swallowed(self):
        """A valid segment that raises ValueError for an unrelated reason (an undeclared metric)
        must raise, not be swallowed - regression for a version that caught `except ValueError`
        unconditionally."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="undeclared"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="undeclared"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.metric_found_group"
        assert "undeclared" in exc_info.value.context["unknown_metrics"]  # ty: ignore[unsupported-operator]

    def test_degenerate_metric_does_not_exclude_sibling_metric(self):
        """A metric whose both arms are degenerate ("Degenerate data: both arms have zero
        variance") is excluded ALONE - a healthy sibling metric in the SAME segment keeps its
        real estimate."""
        rows = [
            _make_arm_row(50, 5.0, 0.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 5.0, 0.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="orders"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="orders"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("orders")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.metric_guarded_excluded" in warning_codes(rec)
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "zero_variance"
        assert by_metric["rev"].lift is None
        assert by_metric["orders"].excluded is None
        orders_lift = by_metric["orders"].lift
        assert orders_lift is not None
        assert math.isfinite(orders_lift.value)
        assert orders_lift.value == pytest.approx(0.125, rel=0.05)

    def test_extreme_ratio_metric_does_not_exclude_sibling_metric(self):
        """Same per-metric isolation for the delta-method precision guard ("Log delta method
        unreliable"): only the offending metric is typed `extreme_ratio`; the sibling keeps its
        estimate."""
        rows = [
            _make_arm_row(50, 0.5, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="orders"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="orders"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("orders")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.metric_guarded_excluded" in warning_codes(rec)
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "extreme_ratio"
        assert by_metric["rev"].lift is None
        assert by_metric["orders"].excluded is None
        orders_lift = by_metric["orders"].lift
        assert orders_lift is not None
        assert math.isfinite(orders_lift.value)

    def test_duplicate_treatment_row_raises(self):
        """Two rows for the same (segment, metric, arm) cell used to double-count that segment
        downstream (k inflated, wrong Q df). Moments must arrive as ONE pre-aggregated row per
        cell."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_breakout_duplicate"
        assert exc_info.value.context["group_id"] == "treatment"

    def test_duplicate_control_row_raises(self):
        """A duplicate control row used to win silently by last-row-wins,
        discarding the first control row's 2000 units and flipping the
        lift (+0.132 -> -0.591 on this fixture) - must raise instead."""
        rows = [
            _make_arm_row(2000, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 30.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            run_breakout(rows, [_mean_metric("rev")], control_group="control", dimension="country")
        assert exc_info.value.code == "breakout.run_breakout_duplicate"
        assert exc_info.value.context["group_id"] == "control"

    def test_zero_mean_treatment_row_is_dropped_other_estimates_still_returned(self):
        """A row with sum_y == 0 is dropped before estimate_lift runs -
        log(mean) is undefined at zero. The dropped row leaves a NaN,
        typed-excluded cell; a sibling metric still produces a real estimate."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="sessions"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="sessions"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.row_nonpositive_mean" in warning_codes(rec)
        assert {r.metric for r in result} == {"rev", "sessions"}
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "nonpositive_mean"
        assert by_metric["rev"].lift is None
        assert by_metric["sessions"].excluded is None

    def test_negative_mean_treatment_row_is_dropped_not_raising(self):
        """A negative mean (not just exactly zero) is caught by the same non-positive-mean guard,
        so ``math.log`` never raises ``ValueError: math domain error``."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, -2.0, 4.0, country="US", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_one_metrics_lost_control_arm_only_drops_that_metrics_estimate(self):
        """A segment where ONLY 'rev's control row has a non-positive
        mean must NOT be a whole-segment skip - estimate_lift runs and
        silently drops 'rev's row; the warning must name that metric."""
        rows = [
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="rev"),
            _make_arm_row(50, 8.0, 4.0, country="US", group_id="control", metric="sessions"),
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="sessions"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )
        assert sorted(warning_codes(record)) == sorted(
            [
                "breakout.estimates.row_nonpositive_mean",
                "breakout.estimates.lost_control_metric",
            ]
        )
        assert {r.metric for r in result} == {"rev", "sessions"}
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded == "no_control_arm"
        assert by_metric["rev"].lift is None
        assert by_metric["sessions"].excluded is None

    def test_zero_mean_control_row_falls_through_to_no_control_arm_skip(self):
        """Dropping the segment's ONLY control row falls through to the
        no-control-arm skip, distinct from a genuinely missing arm - the
        per-metric "lost its control arm" warning must NOT fire here."""
        rows = [
            _make_arm_row(50, 0.0, 0.0, country="US", group_id="control"),
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="treatment"),
            _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="CA", group_id="treatment"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        codes = warning_codes(record)
        assert sorted(codes) == sorted(
            [
                "breakout.estimates.row_nonpositive_mean",
                "breakout.estimates.slice_no_live_control_dropped",
            ]
        )
        assert "breakout.estimates.lost_control_metric" not in codes
        assert {r.dimension_value for r in result} == {"US", "CA"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded == "no_control_arm"
        assert by_segment["US"].lift is None
        assert by_segment["CA"].excluded is None

    def test_degenerate_zero_variance_both_arms_segment_is_skipped_with_warning(self):
        """A segment where BOTH arms have zero within-segment variance
        makes estimate_lift's combined-SE guard raise "Degenerate data" -
        caught and skipped whole; an unaffected segment still estimates."""
        rows = [
            # US: both arms perfectly homogeneous - degenerate.
            _make_arm_row(10, 5.0, 0.0, country="US", group_id="control"),
            _make_arm_row(10, 6.0, 0.0, country="US", group_id="treatment"),
            # CA: ordinary segment, both arms have real spread.
            _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="CA", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.metric_guarded_excluded" in warning_codes(rec)
        assert {r.dimension_value for r in result} == {"US", "CA"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded == "zero_variance"
        assert by_segment["US"].lift is None
        assert by_segment["CA"].excluded is None

    def test_delta_method_unreliable_segment_is_skipped_with_warning(self):
        """``infer_lift``'s sparse-slice guard ("Log delta method unreliable", combined log-scale
        SE >= 0.5) is reachable per segment too - caught by exception type and skipped whole."""
        rows = [
            # US: log-scale SE = sqrt(var/(n*mean^2)) = 0.577 (control) /
            # 0.481 (treatment); combined 0.75 >= 0.5, unreliable.
            _make_arm_row(12, 1.0, 4.0, country="US", group_id="control"),
            _make_arm_row(12, 1.2, 4.0, country="US", group_id="treatment"),
            # CA: ordinary segment, real spread, precise arms.
            _make_arm_row(50, 10.0, 4.0, country="CA", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="CA", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.metric_guarded_excluded" in warning_codes(rec)
        assert {r.dimension_value for r in result} == {"US", "CA"}
        by_segment = {r.dimension_value: r for r in result}
        assert by_segment["US"].excluded == "extreme_ratio"
        assert by_segment["US"].lift is None
        assert by_segment["CA"].excluded is None


# Dense breakout rows: every (segment, metric, method, arm) cell appears
# exactly once; a single-unit segment no longer aborts the whole call.


class TestRunBreakoutDenseGrid:
    def test_single_unit_segment_no_longer_aborts(self):
        """A segment with n=1 for one arm used to blow up ddof=1
        variance (ZeroDivisionError), aborting the WHOLE run_breakout
        call. It now comes back as a typed-excluded NaN row instead."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: only 1 unit in treatment - ddof=1 variance is undefined.
            _make_arm_row(1, 15.0, 0.0, country="MX", group_id="control"),
            _make_arm_row(1, 20.0, 0.0, country="MX", group_id="treatment"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert sorted(warning_codes(record)) == sorted(
            [
                "breakout.estimates.row_few_units",
                "breakout.estimates.row_few_units",
                "breakout.estimates.slice_no_live_control_dropped",
            ]
        )
        by_segment = {r.dimension_value: r for r in result}
        assert set(by_segment) == {"US", "MX"}
        assert by_segment["US"].excluded is None
        us_lift = by_segment["US"].lift
        assert us_lift is not None
        assert us_lift.value > 0
        assert by_segment["MX"].excluded == "few_units"
        assert by_segment["MX"].lift is None

    def test_dense_grid_cardinality_matches_segments_times_metrics_times_arms(self):
        """2 segments x 2 metrics x 2 non-control arms, every combination present with real,
        estimable data -> exactly 8 rows (the exact acceptance-criterion fixture)."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country=c, group_id=g, metric=m)
            for c in ("US", "CA")
            for m in ("rev", "sessions")
            for g in ("control", "treatment_a", "treatment_b")
        ]
        result = run_breakout(
            rows,
            [_mean_metric("rev"), _mean_metric("sessions")],
            control_group="control",
            dimension="country",
        )
        assert len(result) == 8
        cells = {(r.dimension_value, r.metric, r.group_id) for r in result}
        assert cells == {
            (c, m, g)
            for c in ("US", "CA")
            for m in ("rev", "sessions")
            for g in ("treatment_a", "treatment_b")
        }
        assert all(r.excluded is None for r in result)

    def test_dense_grid_covers_a_pair_missing_from_one_segment_entirely(self):
        """A (metric, arm) pair present in one segment but entirely absent from another still gets
        a dense NaN row - not silently dropped from the grid."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment_a", metric="rev"),
            # US has an extra arm, treatment_b, only for 'rev'.
            _make_arm_row(50, 14.0, 4.0, country="US", group_id="treatment_b", metric="rev"),
            # CA never had treatment_b at all.
            _make_arm_row(50, 11.0, 4.0, country="CA", group_id="control", metric="rev"),
            _make_arm_row(50, 13.0, 4.0, country="CA", group_id="treatment_a", metric="rev"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        # Union pairs: (rev, treatment_a), (rev, treatment_b) x 2 segments = 4 rows.
        assert len(result) == 4
        by_cell = {(r.dimension_value, r.group_id): r for r in result}
        assert by_cell[("CA", "treatment_b")].excluded == "no_control_arm"
        assert by_cell[("CA", "treatment_b")].lift is None
        assert by_cell[("US", "treatment_a")].excluded is None
        assert by_cell[("US", "treatment_b")].excluded is None
        assert by_cell[("CA", "treatment_a")].excluded is None

    def test_dense_grid_covers_a_metric_missing_from_one_segment_entirely(self):
        """A declared metric with no summary row at all in a segment (not just a missing arm
        within an otherwise-present metric) must still get a dense unavailable row for that
        segment (0jm6) - dropping it entirely would let a consumer like
        segment_heterogeneity mistake the survivors for a complete family."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="A", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="A", group_id="treatment", metric="rev"),
            # segment B never has a 'rev' row at all -- not merely a missing arm.
            _make_arm_row(50, 11.0, 4.0, country="B", group_id="control", metric="orders"),
            _make_arm_row(50, 13.0, 4.0, country="B", group_id="treatment", metric="orders"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("orders")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.no_row_for_group_metric" in warning_codes(rec)
        assert len(result) == 4
        by_cell = {(r.dimension_value, r.metric): r for r in result}
        assert by_cell[("A", "rev")].excluded is None
        assert by_cell[("B", "orders")].excluded is None
        assert by_cell[("B", "rev")].excluded == "no_control_arm"
        assert by_cell[("B", "rev")].lift is None
        assert by_cell[("A", "orders")].excluded == "no_control_arm"
        assert by_cell[("A", "orders")].lift is None

    def test_control_less_segment_stays_silent_after_whole_segment_warning(self):
        """A control-less segment must not emit per-pair warnings after its whole-segment
        no-control warning, including for global pairs absent from that segment."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="A", group_id="control", metric=metric)
            for metric in ("rev", "sessions")
        ]
        rows.extend(
            _make_arm_row(50, 12.0, 4.0, country="A", group_id=group_id, metric=metric)
            for metric in ("rev", "sessions")
            for group_id in ("T1", "T2")
        )
        rows.extend(
            _make_arm_row(50, 15.0, 4.0, country="B", group_id="T1", metric=metric)
            for metric in ("rev", "sessions")
        )

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )

        assert warning_codes(record) == ["breakout.estimates.slice_no_control_arm"]

        by_cell = {(row.dimension_value, row.metric, row.group_id): row for row in result}
        assert {
            (metric, group_id) for metric in ("rev", "sessions") for group_id in ("T1", "T2")
        } == {
            (metric, group_id)
            for (segment, metric, group_id), row in by_cell.items()
            if segment == "B" and row.excluded == "no_control_arm"
        }

    def test_dense_grid_fans_out_over_methods(self):
        """A cell excluded in a segment still gets one row PER declared
        method - run_breakout's own row cardinality convention."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
            # MX: no control arm at all.
            _make_arm_row(50, 15.0, 4.0, country="MX", group_id="treatment"),
        ]
        with pytest.warns(UserWarning):
            result = run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                methods=[Method(name="unadjusted"), Method(name="other")],
            )
        mx_rows = [r for r in result if r.dimension_value == "MX"]
        assert {r.method for r in mx_rows} == {"unadjusted", "other"}
        assert all(r.excluded == "no_control_arm" for r in mx_rows)

    def test_ratio_metric_nonpositive_denominator_excluded(self):
        """A ratio metric with a non-positive denominator mean is excluded even with a positive
        numerator mean - the family-dependent gate."""
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=100.0
            ),
            _make_arm_row(
                50, 12.0, 4.0, country="US", group_id="treatment", metric="ratio_rev", sum_den=0.0
            ),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_denominator_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_ratio_metric_denominator_key_absent_warns_not_crashes(self):
        """A ratio row with `ref_den` entirely absent (not merely `None`) and a POSITIVE outcome
        mean must still warn gracefully, via a different `mean_den is None` path."""
        treatment_row = _make_arm_row(
            50, 12.0, 4.0, country="US", group_id="treatment", metric="ratio_rev"
        )
        assert treatment_row["ref_den"] is None
        del treatment_row["ref_den"]  # exercise the "key entirely absent" path
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=100.0
            ),
            treatment_row,
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_denominator_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_ratio_metric_denominator_explicit_none_warns_not_crashes(self):
        """The sibling of the key-absent case: `ref_den` explicitly set to `None` rather than
        deleted - same `mean_den is None` guard, different code path."""
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=100.0
            ),
            _make_arm_row(
                50, 12.0, 4.0, country="US", group_id="treatment", metric="ratio_rev", sum_den=None
            ),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_nonpositive_denominator_mean" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "nonpositive_mean"
        assert result[0].lift is None

    def test_few_units_takes_precedence_over_nonpositive_mean(self):
        """A row that is BOTH n<2 AND mean(y)<=0 is classified 'few_units' - there's no ddof=1
        variance to even ask if the mean is positive."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(1, 0.0, 0.0, country="US", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows, [_mean_metric("rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_few_units" in warning_codes(rec)
        assert len(result) == 1
        assert result[0].excluded == "few_units"

    def test_explicit_empty_methods_by_metric_skips_unestimable_cells(self):
        """`methods_by_metric={name: []}` means "skip this metric" on the live
        path; an unestimable cell must honor the same contract instead of
        resurfacing as a default-unadjusted unavailable row."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", metric="rev", group_id="control"),
            _make_arm_row(1, 0.0, 0.0, country="US", metric="rev", group_id="treatment"),
            _make_arm_row(50, 10.0, 4.0, country="US", metric="aov", group_id="control"),
            _make_arm_row(1, 0.0, 0.0, country="US", metric="aov", group_id="treatment"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("aov")],
                control_group="control",
                dimension="country",
                methods_by_metric={"rev": [Method(name="unadjusted")], "aov": []},
            )
        assert "breakout.estimates.row_few_units" in warning_codes(rec)
        assert {r.metric for r in result} == {"rev"}
        assert result[0].excluded == "few_units"
        assert result[0].method == "unadjusted"

    def test_abs_diff_abs_se_populated_from_a_real_non_excluded_estimate(self):
        """`run_breakout` copies `LiftEstimate.abs_diff`/`abs_se` onto the `BreakoutEstimate` for
        a real, live cell too, so the absolute scale is usable straight off a result."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert len(result) == 1
        assert result[0].excluded is None
        assert result[0].abs_diff == pytest.approx(2.0)
        assert result[0].abs_se is not None

    def test_metric_missing_control_row_entirely_warns_even_when_another_metric_has_one(self):
        """A live row whose metric never had a control row in this segment (not merely lost to
        the estimability gate) must warn like every other route, naming the metric and arm.
        Previously this pair silently became a NaN row with no warning at all."""
        rows = [
            _make_arm_row(50, 10.0, 4.0, country="US", group_id="control", metric="rev"),
            _make_arm_row(50, 12.0, 4.0, country="US", group_id="treatment", metric="rev"),
            # sessions never had a control row in this segment - never dropped, just absent.
            _make_arm_row(50, 9.0, 4.0, country="US", group_id="treatment", metric="sessions"),
        ]
        with pytest.warns(IncrementWarning) as rec:
            result = run_breakout(
                rows,
                [_mean_metric("rev"), _mean_metric("sessions")],
                control_group="control",
                dimension="country",
            )
        assert "breakout.estimates.route2_no_live_control_row" in warning_codes(rec)
        by_metric = {r.metric: r for r in result}
        assert by_metric["rev"].excluded is None
        assert by_metric["sessions"].excluded == "no_control_arm"
        assert by_metric["sessions"].lift is None


class TestRunBreakoutBinomialGateExemption:
    """The shared few_units/nonpositive_mean prefilter gates only rows bound
    for the log-Normal delta method, which needs a ddof=1 variance (n>=2)
    and positive means. A conversion/retention metric's n=1 or zero-event
    rows stay eligible for the exact binomial risk-ratio method (see
    binomial_rr.py), which admits both. Only a metric requiring the
    log-Normal delta method (a CUPED-adjusted conversion/retention metric)
    keeps the full gate."""

    def test_conversion_metric_n_equals_one_reaches_exact_binomial(self):
        """n=1 for both arms used to be dropped as 'few_units' before
        even reaching estimate_lift; the exact binomial method is
        admissible at n=1."""
        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = run_breakout(
                rows, [_conversion_metric()], control_group="control", dimension="country"
            )
        assert len(result) == 1
        row = result[0]
        assert row.excluded is None
        assert row.reference_kind == "binomial"
        assert row.lift is not None

    def test_exact_binomial_failure_is_not_labelled_a_lift_guard_exclusion(self):
        """An arm above the exact method's size ceiling fails with a binomial
        code, not a lift guard, so its row must not claim an extreme ratio."""
        big = 5_000_000
        rows = [
            _conversion_arm_row(big, big // 10, country="GB", group_id="control"),
            _conversion_arm_row(big, big // 9, country="GB", group_id="treatment"),
            _conversion_arm_row(1000, 100, country="US", group_id="control"),
            _conversion_arm_row(1000, 130, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows, [_conversion_metric()], control_group="control", dimension="country"
            )
        by_segment = {row.dimension_value: row for row in result}
        assert by_segment["GB"].excluded == "estimation_failed"
        assert by_segment["US"].excluded is None

    def test_one_arm_binomial_failure_beside_an_estimated_arm(self):
        """Only t2 exceeds the exact method's ceiling: t2 is an estimation
        failure, t1 still estimates, and a BH family refuses as run() does."""
        big = 5_000_000
        rows = [
            _conversion_arm_row(1000, 100, country="US", group_id="control"),
            _conversion_arm_row(1000, 130, country="US", group_id="t1"),
            _conversion_arm_row(big, big // 9, country="US", group_id="t2"),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows, [_conversion_metric()], control_group="control", dimension="country"
            )
            by_arm = {row.group_id: row for row in result}
            assert by_arm["t2"].excluded == "estimation_failed"
            assert by_arm["t1"].excluded is None
            with pytest.raises(CapabilityError) as raised:
                run_breakout(
                    rows,
                    [_conversion_metric()],
                    control_group="control",
                    dimension="country",
                    correction="bh",
                )
        assert raised.value.code == "family.evidence.incomplete"

    def test_one_guarded_arm_beside_an_estimated_arm_stays_in_a_bh_family(self):
        """A lift guard on t2 while t1 estimates is a conservative
        non-rejection, so the BH family proceeds instead of refusing."""
        rows = []
        for country in ("US", "CA"):
            rows += [
                _make_arm_row(500, 10.0, 4.0, country=country, group_id="control"),
                _make_arm_row(500, 11.0, 4.0, country=country, group_id="t1"),
                _make_arm_row(3, 1.0, 400.0, country=country, group_id="t2"),
            ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows,
                [_mean_metric("rev")],
                control_group="control",
                dimension="country",
                correction="bh",
            )
        by_cell = {(row.dimension_value, row.group_id): row for row in result}
        assert {by_cell[(c, "t2")].excluded for c in ("US", "CA")} == {"extreme_ratio"}
        assert all(by_cell[(c, "t1")].family_q == 0.10 for c in ("US", "CA"))

    def test_mixed_method_arm_with_two_exclusions_has_one_row_per_method(self):
        """t2 fails the exact method (too large) and CUPED's preparation
        (zero conversions); each method keeps exactly one row and reason."""
        big = 5_000_000
        rows = [
            _conversion_arm_row(1000, 100, country="US", group_id="control", with_covariate=True),
            _conversion_arm_row(1000, 130, country="US", group_id="t1", with_covariate=True),
            _conversion_arm_row(big, 0, country="US", group_id="t2", with_covariate=True),
        ]
        for row in rows:
            row["cx2"] = float(row["n"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[
                    Method(name="unadjusted"),
                    Method(name="cuped", variance_reduction="cuped"),
                ],
            )
        t2 = sorted((row.method, row.excluded) for row in result if row.group_id == "t2")
        assert t2 == [("cuped", "nonpositive_mean"), ("unadjusted", "estimation_failed")]

    def test_conversion_metric_zero_control_count_reaches_set_only_binomial(self):
        """Control never converted (0/50) - used to be dropped as
        'nonpositive_mean'; the exact binomial method admits it directly
        as a set-only row (no finite point, a real confidence set)."""
        rows = [
            _conversion_arm_row(50, 0, country="US", group_id="control"),
            _conversion_arm_row(50, 25, country="US", group_id="treatment"),
        ]
        result = run_breakout(
            rows, [_conversion_metric()], control_group="control", dimension="country"
        )
        assert len(result) == 1
        row = result[0]
        assert row.excluded is None
        assert row.lift is None
        assert row.binomial_set is not None
        assert row.binomial_set.point_available is False

    def test_ancillary_uptake_preserves_exact_binary_breakout_sets(self):
        metric = _conversion_metric()
        bare = [
            _conversion_arm_row(50, 0, country="US", group_id="control"),
            _conversion_arm_row(50, 25, country="US", group_id="treatment"),
        ]
        rows = [
            centered_row_from_raw_sums(
                {
                    "experiment_id": "exp1",
                    "metric": metric.name,
                    "group_id": row["group_id"],
                    "country": "US",
                    "n": 50,
                    "sum_y": count,
                    "sum_y2": count,
                    "sum_d": count,
                    "sum_yd": count,
                    "sum_y2d": count,
                }
            )
            for row, count in zip(bare, (0.0, 25.0), strict=True)
        ]
        expected = estimate_lift([metric], bare, control_group="control").results[0]
        engine = estimate_lift([metric], rows, control_group="control").results[0]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", IncrementWarning)
            (actual,) = run_breakout(rows, [metric], control_group="control", dimension="country")
        assert actual.excluded is None
        assert actual.lift is None
        assert actual.reference_kind == "binomial"
        reference = expected.binomial_set
        assert reference is not None
        assert_set_contains_finer_reference(reference)
        for result in (expected, engine, actual):
            region = result.binomial_set
            assert region is not None
            assert (region.lower, region.upper) == (reference.lower, reference.upper)
            assert region.upper is None
            assert region.point_available is False
            assert (region.x_c, region.n_c, region.x_t, region.n_t) == (0, 50, 25, 50)

    def test_nan_optional_families_are_absent_for_exact_gate(self):
        rows = [
            _conversion_arm_row(1, 0, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        for row in rows:
            row["ref_den"] = math.nan
            row["sum_d"] = math.nan

        result = run_breakout(
            pd.DataFrame(rows),
            [_conversion_metric()],
            control_group="control",
            dimension="country",
        )

        assert result[0].excluded is None
        assert result[0].reference_kind == "binomial"

    @pytest.mark.parametrize("x_role", ["cluster_size", "uptake_total"])
    def test_sparse_non_covariate_x_role_does_not_bypass_legacy_gate(self, x_role):
        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control", with_covariate=True),
            _conversion_arm_row(1, 1, country="US", group_id="treatment", with_covariate=True),
        ]
        for row in rows:
            row["x_role"] = x_role

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
            )

        assert "breakout.estimates.row_few_units" in warning_codes(caught)
        assert result[0].excluded == "few_units"

    def test_conversion_metric_with_cuped_method_keeps_the_legacy_gate(self):
        """A conversion metric requesting a CUPED method still needs the
        log-Normal delta method's ddof=1 variance - n=1 stays 'few_units',
        since the same retained row would otherwise feed the CUPED
        method's own log-Normal path too."""
        rows = [
            _conversion_arm_row(1, 1, country="US", group_id="control"),
            _conversion_arm_row(1, 1, country="US", group_id="treatment"),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[Method(name="cuped", variance_reduction="cuped")],
            )
        assert "breakout.estimates.row_few_units" in warning_codes(record)
        assert len(result) == 1
        assert result[0].excluded == "few_units"

    @pytest.mark.parametrize(("n", "successes"), [(1, 1), (50, 0)])
    def test_mixed_methods_keep_exact_result_and_exclude_only_cuped(self, n, successes):
        rows = [
            _conversion_arm_row(
                n, successes, country="US", group_id="control", with_covariate=True
            ),
            _conversion_arm_row(
                n, successes, country="US", group_id="treatment", with_covariate=True
            ),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows,
                [_conversion_metric()],
                control_group="control",
                dimension="country",
                methods=[
                    Method(name="unadjusted"),
                    Method(name="cuped", variance_reduction="cuped"),
                ],
            )
        expected_code = (
            "breakout.estimates.row_few_units"
            if n == 1
            else "breakout.estimates.row_nonpositive_mean"
        )
        assert expected_code in warning_codes(record)
        by_method = {row.method: row for row in result}
        assert by_method["unadjusted"].reference_kind == "binomial"
        assert by_method["unadjusted"].binomial_set is not None
        assert by_method["unadjusted"].excluded is None
        assert by_method["cuped"].lift is None
        assert by_method["cuped"].excluded == ("few_units" if n == 1 else "nonpositive_mean")
        assert by_method["unadjusted"].method_role == "decision"
        assert by_method["cuped"].method_role == "sensitivity"

    def test_ratio_metric_n_equals_one_still_gated(self):
        """A ratio metric is never binomial-eligible - n=1 stays
        'few_units' exactly as before. Both arms carry a positive
        denominator so only the ddof=1 gate is under test."""
        rows = [
            _make_arm_row(
                50, 10.0, 4.0, country="US", group_id="control", metric="ratio_rev", sum_den=500.0
            ),
            _make_arm_row(
                1, 10.0, 0.0, country="US", group_id="treatment", metric="ratio_rev", sum_den=10.0
            ),
        ]
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            result = run_breakout(
                rows, [_ratio_metric("ratio_rev")], control_group="control", dimension="country"
            )
        assert "breakout.estimates.row_few_units" in warning_codes(record)
        assert len(result) == 1
        assert result[0].excluded == "few_units"


class TestFlattenedBinomialCrossInvariants:
    @staticmethod
    def _payload(model: type[BreakoutEstimate] | type[DailyLiftEstimate]) -> dict[str, Any]:
        rows = [
            _conversion_arm_row(20, 3, country="US", group_id="control"),
            _conversion_arm_row(20, 5, country="US", group_id="treatment"),
        ]
        breakout = run_breakout(
            rows, [_conversion_metric()], control_group="control", dimension="country"
        )[0]
        payload = breakout.model_dump()
        if model is DailyLiftEstimate:
            payload.pop("excluded")
            payload.pop("role")
            payload["ds"] = date(2025, 1, 1)
        return payload

    @pytest.mark.parametrize("model", [BreakoutEstimate, DailyLiftEstimate])
    @pytest.mark.parametrize("mutation", ["bounds", "level", "counts", "reference"])
    def test_deserialized_binomial_row_rejects_cross_field_contradictions(self, model, mutation):
        payload = self._payload(model)
        if mutation == "bounds":
            payload["lift"]["lb"] += 0.1
        elif mutation == "level":
            payload["lift"]["alpha"] = 0.1
            payload["lift"]["level"] = 0.9
        elif mutation == "counts":
            payload["lift"]["value"] += 0.1
        else:
            payload["reference_kind"] = "normal"
        with pytest.raises(InvalidRequestError):
            model.model_validate(payload)


# run_breakout returns BreakoutEstimates; EstimateList/to_frame() mechanics
# (schema-from-class, slicing/concat, model= inference) span every result type.


class TestRunBreakoutReturnsBreakoutEstimates:
    def test_run_breakout_returns_breakout_estimates_with_rows(self):
        rows = [
            _make_arm_row(10, 5.0, 4.0, country="US", group_id="control"),
            _make_arm_row(10, 6.0, 4.0, country="US", group_id="treatment"),
        ]
        result = run_breakout(rows, [_mean_metric()], control_group="control", dimension="country")
        assert isinstance(result, BreakoutEstimates)
        assert isinstance(result, list)
        assert isinstance(result.to_frame(), pd.DataFrame)

    def test_run_breakout_returns_breakout_estimates_when_empty(self):
        """No segments at all (empty *summary*) still returns the concrete collection type, not a
        bare `[]` - `.to_frame()` must work on the empty-result path too."""
        result = run_breakout([], [_mean_metric()], control_group="control", dimension="country")
        assert isinstance(result, BreakoutEstimates)
        assert len(result.to_frame()) == 0


class TestEstimateListSlicingAndConcatPreserveSubclass:
    """Slicing and `+` concatenation keep `.to_frame()` working: both
    override `__getitem__`/`__add__` to rebuild the concrete subclass
    rather than falling back to a bare `list`."""

    def _sample(self) -> BreakoutEstimates:
        return BreakoutEstimates(
            [
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="US",
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                ),
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="CA",
                    lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                ),
            ]
        )

    def test_int_index_returns_the_plain_element(self):
        assert isinstance(self._sample()[0], BreakoutEstimate)

    def test_slice_preserves_the_concrete_subclass(self):
        sliced = self._sample()[0:1]
        assert isinstance(sliced, BreakoutEstimates)
        assert isinstance(sliced.to_frame(), pd.DataFrame)
        assert len(sliced) == 1

    def test_add_preserves_the_concrete_subclass(self):
        combined = self._sample() + self._sample()
        assert isinstance(combined, BreakoutEstimates)
        assert len(combined) == 4
        assert isinstance(combined.to_frame(), pd.DataFrame)

    def test_is_a_plain_list_throughout(self):
        """Every existing `list[BreakoutEstimate]` consumer - isinstance checks, iteration,
        indexing, `len()` - keeps working unchanged, since it really is a `list`."""
        sample = self._sample()
        assert isinstance(sample, list)
        assert len(sample) == 2
        assert [r.dimension_value for r in sample] == ["US", "CA"]


class TestToFrameFreeFunction:
    """`to_frame()` is what `EstimateList.to_frame` delegates to: the escape hatch for a plain list
    assembled some other way (tests, deserialized data, a comprehension that lost the subclass)."""

    def test_infers_model_from_a_non_empty_sequence(self):
        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            )
        ]
        frame = to_frame(estimates)
        assert isinstance(frame, pd.DataFrame)
        assert frame.loc[0, "metric"] == "rev"
        assert frame.loc[0, "group_id"] == "T"
        assert frame.loc[0, ["lift", "lb", "ub"]].to_list() == pytest.approx([0.1, 0.05, 0.15])

    def test_empty_sequence_without_model_raises(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            to_frame([])
        assert exc_info.value.code == "breakout.to_frame_infer"

    def test_empty_sequence_with_model_returns_correct_schema(self):
        frame = to_frame([], model=LiftEstimate)
        populated = to_frame(
            [
                LiftEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.1),
                    ds=date(2026, 1, 1),
                )
            ]
        )
        assert isinstance(frame, pd.DataFrame) and isinstance(populated, pd.DataFrame)
        assert frame.empty
        assert frame.dtypes.equals(populated.dtypes)

    @pytest.mark.parametrize(
        ("estimates", "column", "expected"),
        [
            (
                BreakoutEstimates(
                    [
                        BreakoutEstimate(
                            metric="rev",
                            group_id="T",
                            method="unadjusted",
                            method_role="decision",
                            dimension="country",
                            dimension_value="US",
                            lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                        )
                    ]
                ),
                "lift",
                0.1,
            ),
            (
                DailyMetricValues(
                    [
                        DailyMetricValue(
                            ds=date(2025, 1, 15),
                            metric="rev",
                            group_id="T",
                            value=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                            n=100,
                        )
                    ]
                ),
                "value",
                0.2,
            ),
            (
                DailyLiftEstimates(
                    [
                        DailyLiftEstimate(
                            ds=date(2025, 1, 15),
                            metric="rev",
                            group_id="T",
                            method="unadjusted",
                            method_role="decision",
                            lift=Estimate(value=0.3, lb=0.2, ub=0.4, level=0.95),
                        )
                    ]
                ),
                "lift",
                0.3,
            ),
        ],
    )
    def test_pandas_flattened_estimate_columns_remain_numeric(self, estimates, column, expected):
        frame = cast(pd.DataFrame, estimates.to_frame())

        assert pd.api.types.is_float_dtype(frame[column])
        assert frame.loc[0, column] == pytest.approx(expected)

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_breakout_reference_columns_are_typed_ordered_and_keep_numeric_nulls(self, backend):
        import narwhals as nw

        estimates = BreakoutEstimates(
            [
                BreakoutEstimate(
                    **TestBreakoutEstimateReferenceValidation._kwargs(
                        dof=None,
                        reference_kind="t",
                        reference_df=5.5,
                    )
                ),
                BreakoutEstimate(
                    **TestBreakoutEstimateReferenceValidation._kwargs(dimension_value="CA")
                ),
            ]
        )
        frame = nw.from_native(to_frame(estimates, backend=backend))

        dof_index = frame.columns.index("dof")
        assert frame.columns[dof_index : dof_index + 3] == [
            "dof",
            "reference_kind",
            "reference_df",
        ]
        assert frame.schema["reference_kind"] == nw.String()
        assert frame.schema["reference_df"] == nw.Float64()
        assert frame["reference_kind"].to_list() == ["t", "normal"]
        assert frame["reference_df"][0] == 5.5
        assert frame["reference_df"].is_null().to_list() == [False, True]

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_populated_prior_spec_exports_as_a_string_on_every_backend(self, backend):
        # prior_spec is the first model-typed field to reach to_frame: it
        # must serialize to its repr, not leak a pydantic object pyarrow refuses to coerce.
        import narwhals as nw

        from increment.estimation.priors import MixturePrior, StudentTPrior

        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                prior_spec=StudentTPrior(nu=4.0, scale=0.05),
            ),
            LiftEstimate(
                metric="rev",
                group_id="T2",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                prior_spec=MixturePrior(weights=(0.5, 0.5), means=(0.0, 0.0), sigmas=(0.01, 0.08)),
            ),
            LiftEstimate(
                metric="rev",
                group_id="C",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.0, lb=-0.1, ub=0.1, level=0.95),
            ),
        ]
        column = nw.from_native(to_frame(estimates, backend=backend))["prior_spec"].to_list()
        assert column[0] == "StudentTPrior(nu=4.0, scale=0.05, k=80)"
        assert column[1].startswith("MixturePrior(")
        assert bool(pd.isna(column[2]))

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_optional_strings_keep_native_nulls_across_backends(self, backend):
        estimates = BreakoutEstimates(
            [
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="US",
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                ),
                BreakoutEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    dimension="country",
                    dimension_value="CA",
                    source="events",
                    note="guarded",
                    role="exploratory",
                    lift=Estimate(value=0.2, lb=0.1, ub=0.3, level=0.95),
                ),
            ]
        )
        frame = cast(Any, to_frame(estimates, backend=backend))

        if backend == "pandas":
            assert frame.loc[0, "source"] is pd.NA
            assert frame.loc[0, "note"] is pd.NA
            assert frame.loc[0, "role"] is pd.NA
            assert frame.loc[1, "source"] == "events"
        elif backend == "polars":
            assert frame["source"].to_list() == [None, "events"]
            assert frame["note"].to_list() == [None, "guarded"]
            assert frame["role"].to_list() == [None, "exploratory"]
        else:
            assert frame.column("source").to_pylist() == [None, "events"]
            assert frame.column("note").to_pylist() == [None, "guarded"]
            assert frame.column("role").to_pylist() == [None, "exploratory"]

    @pytest.mark.parametrize("backend", ["pandas", "polars", "pyarrow"])
    def test_ds_stays_nullable_numeric_when_null_and_int_rows_mix(self, backend):
        """An as-of row's numeric `ds` (e.g. a sequential reveal cursor)
        sharing a frame with a null-`ds` total-grain row must not crash:
        the derived dtype has to tolerate the null cell, so an int day
        type maps to Float64 here exactly as every other optional int
        column already does."""
        import narwhals as nw

        rows = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1),
                ds=None,
            ),
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.2),
                ds=3,
            ),
        ]
        frame = nw.from_native(to_frame(rows, backend=backend))
        assert frame.schema["ds"] == nw.Float64()
        assert frame["ds"].to_list()[1] == pytest.approx(3.0)
        assert frame["ds"].is_null().to_list() == [True, False]

    def test_mismatched_model_raises(self):
        """An explicit `model=` that disagrees with the actual runtime type of a non-empty sequence
        raises rather than silently building the wrong schema."""
        estimates = [
            LiftEstimate(
                metric="rev",
                group_id="T",
                method="unadjusted",
                method_role="decision",
                lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
            )
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            to_frame(estimates, model=BreakoutEstimate)
        assert exc_info.value.code == "breakout.to_frame_model"
        assert exc_info.value.context["model"] == "BreakoutEstimate"
        assert exc_info.value.context["inferred"] == "LiftEstimate"

    def test_works_directly_on_lift_estimates_collection(self):
        """LiftEstimates gets the exact same to_frame() as the day-axis types - the mixin logic is
        generic across every result model, not day-axis-specific despite the historical name."""
        estimates = LiftEstimates(
            [
                LiftEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                )
            ]
        )
        frame = cast(pd.DataFrame, estimates.to_frame())
        assert frame.loc[0, "lift"] == pytest.approx(0.1)

    def test_works_directly_on_daily_lift_estimates_collection(self):
        """DailyLiftEstimates (the day-axis type) via the same public
        entry point, for symmetry with the LiftEstimates case above."""
        estimates = DailyLiftEstimates(
            [
                DailyLiftEstimate(
                    metric="rev",
                    group_id="T",
                    method="unadjusted",
                    method_role="decision",
                    ds=date(2025, 1, 1),
                    lift=Estimate(value=0.1, lb=0.05, ub=0.15, level=0.95),
                )
            ]
        )
        frame = cast(pd.DataFrame, estimates.to_frame())
        assert frame.loc[0, "lift"] == pytest.approx(0.1)

    def test_estimate_list_is_a_generic_base_not_a_public_construction_type(self):
        """EstimateList itself has no `_model` set - concrete subclasses (BreakoutEstimates,
        LiftEstimates, DailyMetricValues, DailyLiftEstimates) are the only ones meant to be
        constructed directly."""
        assert not hasattr(EstimateList, "_model")
