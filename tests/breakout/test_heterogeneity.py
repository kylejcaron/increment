"""Tests for `increment.breakout.heterogeneity.segment_contrast`."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as t_dist

from increment.breakout.estimates import (
    BreakoutEstimate,
    BreakoutEstimates,
    ExclusionReason,
    run_breakout,
)
from increment.breakout.heterogeneity import (
    HeterogeneitySummaries,
    SegmentEstimate,
    SegmentEstimates,
    _log_scale_moments,
    _relative_raw_lift,
    segment_contrast,
    segment_heterogeneity,
)
from increment.breakout.rollout import segment_rollout_recommendation
from increment.errors import IncrementWarning, InvalidRequestError
from increment.estimation.armstats import ArmStats, centered_row_from_raw_sums
from increment.estimation.cuped import cuped_adjust
from increment.estimation.engine import Method
from increment.estimation.inference import Normal, normal_posterior
from increment.estimation.results import Estimate
from increment.semantics.models import ConversionMetric, MeanMetric
from tests.mc import replicate
from tests.oracles.test_meta_oracle import assert_segment_intervals_match, segment_posterior
from tests.warning_codes import warning_codes

_Z95 = norm.ppf(0.975)


class _Unset:
    pass


_UNSET = _Unset()


def _breakout_estimate(
    dimension_value: str,
    log_value: float,
    se_log: float,
    level: float = 0.95,
    *,
    alternative: str = "two-sided",
    dof: float | None = None,
    inference: str = "fixed",
    family_q: float | None = None,
    discovery: bool | None = None,
    reference_kind: str | _Unset = _UNSET,
    reference_df: float | None | _Unset = _UNSET,
):
    """Build an exact interval from known log moments and its declared reference."""
    critical_df = reference_df if reference_kind == "t" and isinstance(reference_df, float) else dof
    critical = (
        t_dist.ppf(0.5 + level / 2.0, critical_df)
        if critical_df is not None
        else norm.ppf(0.5 + level / 2.0)
    )
    reference: dict[str, Any] = {}
    if not isinstance(reference_kind, _Unset):
        reference["reference_kind"] = reference_kind
    if not isinstance(reference_df, _Unset):
        reference["reference_df"] = reference_df
    return BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        alternative=alternative,
        dof=dof,
        inference=inference,
        family_q=family_q,
        discovery=discovery,
        **reference,
        dimension="country",
        dimension_value=dimension_value,
        lift=Estimate(
            value=math.expm1(log_value),
            lb=math.expm1(log_value - critical * se_log),
            ub=math.expm1(log_value + critical * se_log),
            level=level,
            log_mean=log_value,
            log_se=se_log,
        ),
    )


def _absolute_breakout_estimate(
    dimension_value: str, value: float, se: float, level: float = 0.95, estimand: str = "late"
):
    """An additive-scale BreakoutEstimate (e.g. an encouragement LATE row)."""
    z = norm.ppf(0.5 + level / 2.0)
    return BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=dimension_value,
        estimand=estimand,
        value_scale="absolute",
        lift=Estimate(value=value, lb=value - z * se, ub=value + z * se, level=level),
    )


def _mean_metric(name: str = "rev", window_days: int | None = None) -> MeanMetric:
    """Build a MeanMetric fixture for tests using mean metrics."""
    return MeanMetric(name=name, entity="user", fact=name, window_days=window_days)


class TestSegmentContrast:
    def test_hand_computed_diff_matches_pairwise_contrast_arrays(self):
        """segment_contrast on the model layer reproduces the array-layer closed form."""
        estimates = BreakoutEstimates(
            [
                _breakout_estimate("US", 0.20, 0.05),
                _breakout_estimate("GB", 0.10, 0.04),
            ]
        )
        result = segment_contrast(estimates, "US", "GB")
        diff_log = 0.20 - 0.10
        se_log = math.sqrt(0.05**2 + 0.04**2)
        assert result.value == pytest.approx(math.expm1(diff_log))
        assert result.lb == pytest.approx(math.expm1(diff_log - _Z95 * se_log))
        assert result.ub == pytest.approx(math.expm1(diff_log + _Z95 * se_log))
        assert result.level == pytest.approx(0.95)

    def test_worked_example_individually_significant_difference_not(self):
        """Two segments individually excluding 0 can still yield a difference interval that includes 0 - eyeballing two separate intervals is not the calibrated comparison."""
        us = _breakout_estimate("US", 0.20, 0.05)
        gb = _breakout_estimate("GB", 0.10, 0.04)
        assert us.require_lift().excludes(0.0), "US individually significant"
        assert gb.require_lift().excludes(0.0), "GB individually significant"

        contrast = segment_contrast(BreakoutEstimates([us, gb]), "US", "GB")
        assert not contrast.excludes(0.0), "US-vs-GB difference is NOT significant"

    def test_order_matters_sign_flips(self):
        """log1p(value) flips sign under swap; the back-transformed `value` itself does not (expm1 is not antisymmetric)."""
        estimates = BreakoutEstimates(
            [_breakout_estimate("US", 0.20, 0.05), _breakout_estimate("GB", 0.10, 0.04)]
        )
        us_vs_gb = segment_contrast(estimates, "US", "GB")
        gb_vs_us = segment_contrast(estimates, "GB", "US")
        assert math.log1p(gb_vs_us.value) == pytest.approx(-math.log1p(us_vs_gb.value), abs=1e-9)

    def test_guards_missing_segment(self):
        estimates = BreakoutEstimates([_breakout_estimate("US", 0.20, 0.05)])
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(estimates, "US", "GB")
        assert exc_info.value.code == "breakout.segment_contrast_expected"
        assert exc_info.value.context["label"] == "GB"
        assert exc_info.value.context["row_count"] == 0

    def test_guards_ambiguous_segment(self):
        """Two rows for the same dimension_value (e.g. two metrics mixed in) is ambiguous."""
        us_a = _breakout_estimate("US", 0.20, 0.05)
        us_b = us_a.model_copy(update={"metric": "conv"})
        gb = _breakout_estimate("GB", 0.10, 0.04)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(BreakoutEstimates([us_a, us_b, gb]), "US", "GB")
        assert exc_info.value.code == "breakout.segment_contrast_expected"
        assert exc_info.value.context["label"] == "US"
        assert exc_info.value.context["row_count"] == 2

    def test_guards_mixed_metric_labels(self):
        """Each label appears exactly once but for a different metric - a per-label uniqueness check alone would let this ambiguous, mixed-metric grouping through silently."""
        us = _breakout_estimate("US", 0.20, 0.05)
        gb = _breakout_estimate("GB", 0.10, 0.04).model_copy(update={"metric": "conv"})
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(BreakoutEstimates([us, gb]), "US", "GB")
        assert exc_info.value.code == "breakout.segment_contrast_dimension"

    def test_guards_missing_interval(self):
        no_interval = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.1),
        )
        gb = _breakout_estimate("GB", 0.10, 0.04)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(BreakoutEstimates([no_interval, gb]), "US", "GB")
        assert exc_info.value.code == "breakout.segment_contrast_absolute_interval"

    def test_guards_missing_absolute_interval(self):
        no_interval = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="A",
            estimand="late",
            value_scale="absolute",
            lift=Estimate(value=0.1),
        )
        b = _absolute_breakout_estimate("B", 3.0, 0.1)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(BreakoutEstimates([no_interval, b]), "A", "B")
        assert exc_info.value.code == "breakout.segment_contrast_absolute_interval"

    def test_missing_interval_paths_share_canonical_code(self):
        """Relative and absolute segment paths represent one missing-interval hazard."""
        relative = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="US",
            lift=Estimate(value=0.1),
        )
        absolute = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="A",
            estimand="late",
            value_scale="absolute",
            lift=Estimate(value=0.1),
        )
        relative_peer = _breakout_estimate("GB", 0.10, 0.04)
        absolute_peer = _absolute_breakout_estimate("B", 3.0, 0.1)

        with pytest.raises(InvalidRequestError) as via_relative:
            segment_contrast(BreakoutEstimates([relative, relative_peer]), "US", "GB")
        with pytest.raises(InvalidRequestError) as via_absolute:
            segment_contrast(BreakoutEstimates([absolute, absolute_peer]), "A", "B")

        assert (
            via_relative.value.code
            == via_absolute.value.code
            == ("breakout.segment_contrast_absolute_interval")
        )

    def test_guards_missing_estimate(self):
        a = BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value="A",
            lift=None,
            excluded="zero_variance",
        )
        b = a.model_copy(update={"dimension_value": "B"})
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(BreakoutEstimates([a, b]), "A", "B")
        assert exc_info.value.code == "breakout.segment_contrast_estimate"

    def test_guards_alpha_out_of_range(self):
        estimates = BreakoutEstimates(
            [_breakout_estimate("US", 0.20, 0.05), _breakout_estimate("GB", 0.10, 0.04)]
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(estimates, "US", "GB", alpha=1.0)
        assert exc_info.value.code == "estimation.meta.alpha_strictly_between"

    def test_refuses_legacy_sequential_intervals_without_checkpoint(self):
        from increment.errors import CapabilityError

        with pytest.raises(CapabilityError) as raised:
            _breakout_estimate("US", 0.20, 0.05, inference="always_valid")
        assert raised.value.code == "sequential.continuation.legacy"

    def test_refuses_gaussian_always_valid_run_breakout_before_rows(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        snapshot, metrics, policy = raw_gaussian(
            segments=((("country", "US"),), (("country", "GB"),))
        )
        with pytest.raises(CapabilityError) as raised:
            run_breakout(
                snapshot, metrics, control_group="control", dimension="country", inference=policy
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_absolute_scale_uses_additive_contrast_not_log(self):
        """value_scale='absolute' rows (e.g. additive LATE) must contrast
        additively, not through log1p/expm1 - applying the relative-scale
        transform to an additive effect silently returns the wrong number."""
        tau_a = _absolute_breakout_estimate("A", 1.0, 0.1)
        tau_b = _absolute_breakout_estimate("B", 3.0, 0.1)
        result = segment_contrast(BreakoutEstimates([tau_a, tau_b]), "A", "B")
        se = math.sqrt(0.1**2 + 0.1**2)
        assert result.value == pytest.approx(-2.0)
        assert result.lb == pytest.approx(-2.0 - _Z95 * se)
        assert result.ub == pytest.approx(-2.0 + _Z95 * se)

    def test_absolute_scale_handles_effects_at_or_below_negative_one(self):
        """A valid additive effect <= -1 must not raise from log1p, which the
        relative-scale path would (log1p is undefined at x <= -1).

        At least one INPUT has to breach that domain, not just their contrast:
        the relative path applies log1p per estimate, so inputs of -0.5 and 0.8
        would not exercise it even though their difference is below -1."""
        tau_a = _absolute_breakout_estimate("A", -1.5, 0.1)
        tau_b = _absolute_breakout_estimate("B", -1.0, 0.1)
        result = segment_contrast(BreakoutEstimates([tau_a, tau_b]), "A", "B")
        assert result.value == pytest.approx(-0.5)
        se = math.sqrt(0.1**2 + 0.1**2)
        assert result.lb == pytest.approx(-0.5 - _Z95 * se)
        assert result.ub == pytest.approx(-0.5 + _Z95 * se)

    # `test_real_run_breakout_coverage` below replaces a since-removed smoke
    # test that guaranteed ~95% coverage by construction rather than testing anything.


def _run_breakout_two_segment_fixture(
    prior: Normal | None = None, window_days: int | None = None, **breakout_kwargs
) -> BreakoutEstimates:
    """Real ArmStats -> run_breakout output for two segments (US, GB), used to exercise segment_contrast/segment_heterogeneity against infer_lift's actual closed-form interval."""
    rng = np.random.default_rng(7)

    def arm(group_id: str, n: int, y_mean: float) -> ArmStats:
        y = rng.normal(y_mean, 3.0, n)
        return ArmStats.from_raw_sums(
            study_id="e1",
            metric="rev",
            group_id=group_id,
            n=n,
            sum_y=float(y.sum()),
            sum_y2=float((y**2).sum()),
        )

    def group_row(a: ArmStats, dimension_value: str) -> dict:
        return {
            "experiment_id": a.study_id,
            "metric": a.metric,
            "group_id": a.group_id,
            "country": dimension_value,
            "n": float(a.n),
            "ref_y": a.ref_y,
            "cy1": a.cy1,
            "cy2": a.cy2,
            "ref_x": None,
            "cx1": None,
            "cx2": None,
            "cxy": None,
            "ref_den": None,
            "cden1": None,
            "cden2": None,
            "cyden": None,
        }

    rows = [
        group_row(arm("control", 500, 10.0), "US"),
        group_row(arm("treatment", 500, 11.0), "US"),
        group_row(arm("control", 500, 20.0), "GB"),
        group_row(arm("treatment", 500, 22.0), "GB"),
    ]
    return run_breakout(
        pd.DataFrame(rows),
        [_mean_metric(window_days=window_days)],
        control_group="control",
        dimension="country",
        prior=prior,
        **breakout_kwargs,
    )


class TestSegmentContrastRealBreakoutOutput:
    """segment_contrast run on real run_breakout output, not the exact log-normal-interval fixtures the rest of this file uses - exercises infer_lift's actual closed-form posterior."""

    def test_recovers_se_close_to_analytic_delta_method_se(self):
        """These rows carry a Welch t reference, so their endpoints are t
        quantiles: ``_log_scale_moments`` must read the working-scale moments
        the interval was cut from and recover ``lift.log_se`` to float
        precision. A Normal back-solve off the endpoints would return
        ``log_se * t/z`` instead -- so also assert the recovery is tight
        enough to exclude that inflation, not merely close."""
        estimates = _run_breakout_two_segment_fixture()
        us = next(r for r in estimates if r.dimension_value == "US")
        gb = next(r for r in estimates if r.dimension_value == "GB")
        us_lift = us.lift
        gb_lift = gb.lift
        assert us_lift is not None and gb_lift is not None
        assert us_lift.log_se is not None
        assert gb_lift.log_se is not None
        assert us.reference_kind == "t" and us.reference_df is not None
        assert gb.reference_kind == "t" and gb.reference_df is not None

        _, se_us = _log_scale_moments(us_lift, reference_kind=us.reference_kind)
        _, se_gb = _log_scale_moments(gb_lift, reference_kind=gb.reference_kind)
        assert se_us == pytest.approx(us_lift.log_se, rel=1e-12)
        assert se_gb == pytest.approx(gb_lift.log_se, rel=1e-12)
        # The inflation a Normal back-solve would have introduced is larger
        # than the tolerance above, so that tolerance is load-bearing.
        assert t_dist.isf(0.025, us.reference_df) / _Z95 > 1.0 + 1e-12

        contrast = segment_contrast(estimates, "US", "GB")
        assert contrast.lb is not None
        assert contrast.ub is not None
        recovered_combined_se = (math.log1p(contrast.ub) - math.log1p(contrast.lb)) / (2 * _Z95)
        analytic_combined_se = math.sqrt(us_lift.log_se**2 + gb_lift.log_se**2)
        assert recovered_combined_se == pytest.approx(analytic_combined_se, rel=0.02)


class TestSegmentContrastDirectionalBHBreakoutFCR:
    """End-to-end reproduction of the directional BH-corrected FCR breakout
    path: ``correction="bh"`` + ``alternative="greater"`` opens the far
    bound on a selected cell's interval (``open_side="upper"``, ``ub=None``)
    -- ``segment_contrast`` must recover the pair's variance from those
    genuinely open intervals, not refuse them as if they were unavailable."""

    def test_segment_contrast_recovers_variance_from_directional_bh_open_intervals(self):
        estimates = _run_breakout_two_segment_fixture(correction="bh", alternative="greater", q=0.5)
        us = next(r for r in estimates if r.dimension_value == "US")
        gb = next(r for r in estimates if r.dimension_value == "GB")
        assert us.discovery is True and gb.discovery is True
        assert us.lift is not None and gb.lift is not None
        assert (
            us.require_lift().open_side == "upper"
            and us.require_lift().ub is None
            and us.require_lift().lb is not None
        )
        assert (
            gb.require_lift().open_side == "upper"
            and gb.require_lift().ub is None
            and gb.require_lift().lb is not None
        )

        contrast = segment_contrast(estimates, "US", "GB")

        # Rebuild segment_contrast's delta-method se from each row's own Welch-t
        # critical value; a Normal quantile would inflate it by the t/z ratio.
        def _se(row: BreakoutEstimate) -> float:
            lift = row.lift
            assert lift is not None and lift.alpha is not None and lift.lb is not None
            assert row.reference_kind == "t" and row.reference_df is not None
            crit = t_dist.isf(lift.alpha, row.reference_df)
            mu = math.log1p(lift.value)
            calibrated = math.log1p(lift.lb)
            return (mu - calibrated) / crit

        mu_us, mu_gb = math.log1p(us.require_lift().value), math.log1p(gb.require_lift().value)
        se_us, se_gb = _se(us), _se(gb)
        delta = mu_us - mu_gb
        combined_se = math.sqrt(se_us**2 + se_gb**2)
        expected_value = math.expm1(delta)
        expected_lb = math.expm1(delta - _Z95 * combined_se)
        expected_ub = math.expm1(delta + _Z95 * combined_se)
        assert contrast.value == pytest.approx(expected_value, rel=1e-12)
        assert contrast.lb == pytest.approx(expected_lb, rel=1e-9)
        assert contrast.ub == pytest.approx(expected_ub, rel=1e-9)


def _open_breakout_estimate(
    dimension_value: str,
    raw_center: float,
    raw_se: float,
    alpha: float,
    *,
    value_scale: Literal["relative", "absolute"],
) -> BreakoutEstimate:
    """Build the fixed open interval emitted by the prior-free constructor."""
    posterior = normal_posterior(raw_center, raw_se)
    z = norm.isf(alpha)
    if value_scale == "relative":
        value = math.expm1(posterior.mu)
        calibrated = math.expm1(posterior.mu - z * posterior.sigma)
    else:
        value = posterior.mu
        calibrated = posterior.mu - z * posterior.sigma
    return BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        alternative="greater",
        dimension="country",
        dimension_value=dimension_value,
        estimand="late" if value_scale == "absolute" else "itt",
        value_scale=value_scale,
        lift=Estimate(
            value=value,
            lb=calibrated,
            ub=None,
            open_side="upper",
            level=math.fsum((1.0, -alpha)),
            alpha=alpha,
            log_mean=raw_center,
            log_se=raw_se,
        ),
    )


@pytest.mark.parametrize("alpha", [0.5, math.nextafter(0.5, 0.0), math.nextafter(0.5, 1.0)])
@pytest.mark.parametrize("value_scale", ["relative", "absolute"])
def test_segment_contrast_uses_persisted_moments_near_half_tail(
    alpha: float, value_scale: Literal["relative", "absolute"]
):
    """A zero or tiny endpoint critical value cannot erase the stored variance."""
    row_a = _open_breakout_estimate("A", 0.2, 0.05, alpha, value_scale=value_scale)
    row_b = _open_breakout_estimate("B", 0.1, 0.04, alpha, value_scale=value_scale)

    result = segment_contrast(BreakoutEstimates([row_a, row_b]), "A", "B")

    post_a = normal_posterior(0.2, 0.05)
    post_b = normal_posterior(0.1, 0.04)
    diff = post_a.mu - post_b.mu
    se = math.sqrt(post_a.sigma**2 + post_b.sigma**2)
    if value_scale == "relative":
        assert result.value == pytest.approx(math.expm1(diff))
        assert result.lb == pytest.approx(math.expm1(diff - _Z95 * se))
        assert result.ub == pytest.approx(math.expm1(diff + _Z95 * se))
    else:
        assert result.value == pytest.approx(diff)
        assert result.lb == pytest.approx(diff - _Z95 * se)
        assert result.ub == pytest.approx(diff + _Z95 * se)


@pytest.mark.parametrize("alpha", [0.5, math.nextafter(0.5, 0.0), math.nextafter(0.5, 1.0)])
def test_segment_contrast_refuses_half_tail_without_persisted_moments(alpha: float):
    row_a = _open_breakout_estimate("A", 0.2, 0.05, alpha, value_scale="relative")
    row_b = _open_breakout_estimate("B", 0.1, 0.04, alpha, value_scale="relative")
    lift_a = row_a.lift
    assert lift_a is not None
    rows = BreakoutEstimates(
        [
            row_a.model_copy(
                update={"lift": lift_a.model_copy(update={"log_mean": None, "log_se": None})}
            ),
            row_b,
        ]
    )

    with pytest.raises(InvalidRequestError) as exc_info:
        segment_contrast(rows, "A", "B")
    assert exc_info.value.code == "estimation.results.lift.open_interval_unrecoverable"


def _malformed_open_estimate(
    dimension_value: str, *, value_scale: Literal["relative", "absolute"], calibrated: float
) -> BreakoutEstimate:
    """Hand-built open row with no persisted log_mean/log_se (forcing the
    endpoint-inversion fallback) and a caller-controlled calibrated bound,
    independent of `value` -- lets a test place it on the wrong side of
    (or exactly at) `value` to force a nonpositive recovered se."""
    return BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        alternative="greater",
        dimension="country",
        dimension_value=dimension_value,
        estimand="late" if value_scale == "absolute" else "itt",
        value_scale=value_scale,
        lift=Estimate(value=0.1, lb=calibrated, ub=None, open_side="upper", level=0.95, alpha=0.05),
    )


@pytest.mark.parametrize("value_scale", ["relative", "absolute"])
@pytest.mark.parametrize(
    "calibrated",
    [0.5, 0.1],
    ids=["inverted", "degenerate"],
)
def test_segment_contrast_refuses_nonpositive_recovered_se(
    value_scale: Literal["relative", "absolute"], calibrated: float
):
    """A hand-built open row with a calibrated bound on the wrong side of
    (or exactly equal to) `value` would silently back-solve a negative or
    zero se without this guard -- negative se squared masks the sign error
    into a plausible-looking but wrong contrast; zero se is a spurious
    zero-variance term. Both must refuse with a message identifying the
    invalid recovered standard error, not guidance requiring an impossible
    closed interval or an unattributed error from `pairwise_contrast_arrays`."""
    row_a = _malformed_open_estimate("A", value_scale=value_scale, calibrated=calibrated)
    row_b = _open_breakout_estimate("B", 0.1, 0.04, 0.05, value_scale=value_scale)

    with pytest.raises(InvalidRequestError) as exc_info:
        segment_contrast(BreakoutEstimates([row_a, row_b]), "A", "B")
    assert exc_info.value.code == "breakout.segment_contrast_open_interval_variance"


def _coverage_rep(rng: np.random.Generator, n: int) -> tuple[Estimate, float, float]:
    """One rep: draw two segments with known true log-scale lifts (0.20, 0.10), run the real run_breakout pipeline, and contrast - the interval fed to segment_contrast is infer_lift's real posterior, not a fixture."""
    true_log_a, true_log_b = 0.20, 0.10
    c_mean = 10.0

    def arm(group_id: str, mean: float) -> ArmStats:
        y = rng.normal(mean, 3.0, n)
        return ArmStats.from_raw_sums(
            study_id="e1",
            metric="rev",
            group_id=group_id,
            n=n,
            sum_y=float(y.sum()),
            sum_y2=float((y**2).sum()),
        )

    def group_row(a: ArmStats, dimension_value: str) -> dict:
        return {
            "experiment_id": a.study_id,
            "metric": a.metric,
            "group_id": a.group_id,
            "seg": dimension_value,
            "n": float(a.n),
            "ref_y": a.ref_y,
            "cy1": a.cy1,
            "cy2": a.cy2,
            "ref_x": None,
            "cx1": None,
            "cx2": None,
            "cxy": None,
            "ref_den": None,
            "cden1": None,
            "cden2": None,
            "cyden": None,
        }

    rows = [
        group_row(arm("control", c_mean), "A"),
        group_row(arm("treatment", c_mean * math.exp(true_log_a)), "A"),
        group_row(arm("control", c_mean), "B"),
        group_row(arm("treatment", c_mean * math.exp(true_log_b)), "B"),
    ]
    result = run_breakout(
        pd.DataFrame(rows), [_mean_metric()], control_group="control", dimension="seg"
    )
    contrast = segment_contrast(result, "A", "B")
    return contrast, true_log_a, true_log_b


class TestSegmentContrastRealBreakoutCoverage:
    """Genuinely discriminating coverage check: real per-rep ArmStats draws run
    through the full run_breakout -> segment_contrast pipeline, not a fixture that guarantees coverage by construction."""

    @pytest.mark.slow
    @pytest.mark.parameter_recovery
    def test_real_run_breakout_coverage(self):
        """Measured, not trusted: n=300/arm, seed=0, 300 reps -> ~93.7%
        coverage of the true log-scale difference."""
        rng = np.random.default_rng(0)
        reps = 300

        def _hit(i: int) -> bool:
            contrast, true_log_a, true_log_b = _coverage_rep(rng, n=300)
            assert contrast.lb is not None
            assert contrast.ub is not None
            true_diff = math.expm1(true_log_a - true_log_b)
            return contrast.lb <= true_diff <= contrast.ub

        coverage = replicate(reps, _hit).rate
        assert 0.90 <= coverage <= 0.99, f"expected ~95% coverage, got {coverage:.3%}"

    def test_real_run_breakout_coverage_fast(self):
        """Fast-suite variant: fewer reps, wide band, still exercising the real
        run_breakout -> segment_contrast path - fast because infer_lift's posterior CI is closed-form, not because reps are few."""
        rng = np.random.default_rng(0)
        reps = 20

        def _hit(i: int) -> bool:
            contrast, true_log_a, true_log_b = _coverage_rep(rng, n=100)
            assert contrast.lb is not None
            assert contrast.ub is not None
            true_diff = math.expm1(true_log_a - true_log_b)
            return contrast.lb <= true_diff <= contrast.ub

        coverage = replicate(reps, _hit).rate
        assert 0.6 <= coverage <= 1.0, f"implausible coverage {coverage:.1%}"


class TestSegmentContrastCupedIndependence:
    """run_breakout partitions on dimension before calling estimate_lift, so
    CUPED's theta is estimated per segment - pooling it across segments would silently change every segment's lift with no test failing."""

    def test_run_breakout_theta_is_scoped_per_segment_not_pooled(self):
        rng = np.random.default_rng(0)

        def make_arm(group_id, n, y_mean, x_mean, corr, x_sd):
            z1 = rng.standard_normal(n)
            z2 = rng.standard_normal(n)
            x = x_mean + x_sd * z1
            y = y_mean + (corr * z1 + math.sqrt(1 - corr**2) * z2)
            return ArmStats.from_raw_sums(
                study_id="e1",
                metric="rev",
                group_id=group_id,
                n=n,
                sum_y=float(y.sum()),
                sum_y2=float((y**2).sum()),
                sum_x=float(x.sum()),
                sum_x2=float((x**2).sum()),
                sum_xy=float((x * y).sum()),
            )

        # US: covariate strongly POSITIVELY correlated with y, small x scale.
        us_c = make_arm("control", 2000, 10.0, 5.0, 0.9, x_sd=1.0)
        us_t = make_arm("treatment", 2000, 11.0, 5.0, 0.9, x_sd=1.0)
        # GB: covariate negatively correlated with y, a larger, differently-
        # located x scale - so a pooled theta would land far from either segment's own theta.
        gb_c = make_arm("control", 2000, 10.0, 50.0, -0.9, x_sd=10.0)
        gb_t = make_arm("treatment", 2000, 11.0, 50.0, -0.9, x_sd=10.0)

        def row(arm, country):
            return {
                "experiment_id": arm.study_id,
                "metric": arm.metric,
                "group_id": arm.group_id,
                "country": country,
                "n": float(arm.n),
                "ref_y": arm.ref_y,
                "cy1": arm.cy1,
                "cy2": arm.cy2,
                "ref_x": arm.ref_x,
                "cx1": arm.cx1,
                "cx2": arm.cx2,
                "cxy": arm.cxy,
                "x_role": arm.x_role,
                "ref_den": None,
                "cden1": None,
                "cden2": None,
                "cyden": None,
            }

        rows = [row(us_c, "US"), row(us_t, "US"), row(gb_c, "GB"), row(gb_t, "GB")]
        metrics = [MeanMetric(name="rev", entity="user", fact="rev")]
        cuped_method = [Method(name="cuped", variance_reduction="cuped")]

        actual = run_breakout(
            pd.DataFrame(rows),
            metrics,
            control_group="control",
            dimension="country",
            methods=cuped_method,
        )
        actual_us_row = next(r for r in actual if r.dimension_value == "US")
        actual_us_estimate = actual_us_row.lift
        assert actual_us_estimate is not None
        actual_us_lift = actual_us_estimate.value

        # The WRONG, hoisted-theta answer: pool theta across BOTH segments'
        # arms in one cuped_adjust call, then read off US's adjusted means.
        pooled = cuped_adjust([us_c, us_t, gb_c, gb_t])
        pooled_us_lift = pooled[1].mean / pooled[0].mean - 1.0

        def relative_cuped_oracle(arms: list[ArmStats]) -> float:
            """Independent oracle for one segment's relative lift.

            Theta is the WITHIN-arm, inverse-n-weighted form the estimator
            uses: each arm contributes its own centered moments over its own
            count, so no between-arm delta enters theta. The pooled mean of x
            remains the centering constant.
            """
            weighted_cov = sum(arm.cov_yx() / arm.n for arm in arms)
            weighted_var_x = sum(arm.var_x() / arm.n for arm in arms)
            theta = weighted_cov / weighted_var_x
            total_n = sum(arm.n for arm in arms)
            mean_x = sum(arm.n * arm.mean_x() for arm in arms) / total_n
            adjusted = [arm.mean_y() - theta * (arm.mean_x() - mean_x) for arm in arms]
            return adjusted[1] / adjusted[0] - 1.0

        expected_us_lift = relative_cuped_oracle([us_c, us_t])
        assert actual_us_lift == pytest.approx(expected_us_lift, rel=1e-12)
        assert abs(actual_us_lift - pooled_us_lift) / abs(actual_us_lift) > 0.05, (
            "run_breakout's US lift matches the HOISTED pooled-theta answer -- "
            "theta is no longer scoped to one segment at a time (D6 violated)"
        )


def _het_estimate(
    dimension_value: str,
    log_mean: float,
    log_se: float,
    *,
    group_id: str = "treatment",
    abs_diff: float | None = None,
    abs_se: float | None = None,
    excluded: ExclusionReason | None = None,
    level: float = 0.95,
) -> BreakoutEstimate:
    """Directly-constructed BreakoutEstimate with known log-scale moments - bypasses run_breakout so Q/I^2/tau^2 fixtures are hand-computable, not 'runs on the fixtures'."""
    z = norm.ppf(0.5 + level / 2)
    if excluded is not None:
        lift = None
    else:
        lift = Estimate(
            value=math.expm1(log_mean),
            lb=math.expm1(log_mean - z * log_se),
            ub=math.expm1(log_mean + z * log_se),
            level=level,
            log_mean=log_mean,
            log_se=log_se,
        )
    return BreakoutEstimate(
        metric="rev",
        group_id=group_id,
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=dimension_value,
        source=None,
        lift=lift,
        abs_diff=None if excluded is not None else abs_diff,
        abs_se=None if excluded is not None else abs_se,
        excluded=excluded,
    )


def _late_estimate(dimension_value: str, value: float, se: float) -> BreakoutEstimate:
    """Encouragement LATE-shaped row: additive effect in `lift` itself, with no
    log-scale moments and no abs_diff/abs_se (matching the Wald rows
    estimate_encouragement emits)."""
    return BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=dimension_value,
        source=None,
        estimand="late",
        value_scale="absolute",
        lift=Estimate(value=value, lb=value - _Z95 * se, ub=value + _Z95 * se, level=0.95),
    )


def _absolute_primary_estimate(dimension_value: str, value: float, se: float) -> BreakoutEstimate:
    """Additive primary row whose moments use the shared Estimate fields."""
    z = _Z95
    return BreakoutEstimate(
        metric="rev",
        group_id="treatment",
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=dimension_value,
        source=None,
        estimand="late",
        value_scale="absolute",
        lift=Estimate(
            value=value,
            lb=value - z * se,
            ub=value + z * se,
            level=0.95,
            log_mean=value,
            log_se=se,
        ),
    )


def test_absolute_effects_pool_additively_and_are_not_relatively_priced():
    rows = BreakoutEstimates(
        [
            _absolute_primary_estimate("US", 0.4, 0.05),
            _absolute_primary_estimate("CA", 0.6, 0.05),
        ]
    )

    summary, segments = segment_heterogeneity(rows)

    absolute = [item for item in summary if item.value_scale == "absolute"]
    assert len(absolute) == 1
    assert {item.scale for item in summary} == {"absolute"}
    assert absolute[0].pooled.value == pytest.approx(0.5)
    assert absolute[0].pooled.value != pytest.approx(math.expm1(0.5))
    rollout, rollout_segments = segment_rollout_recommendation(rows)
    assert len(rollout) == 0
    assert len(rollout_segments) == 0
    assert all(item.value_scale == "absolute" for item in segments)


@pytest.mark.parametrize("values", [(-0.2, 0.0), (0.0, 0.2)])
def test_absolute_zero_and_negative_effects_keep_additive_intervals(values):
    rows = BreakoutEstimates(
        [
            _absolute_primary_estimate(label, value, 0.05)
            for label, value in zip(("A", "B"), values, strict=True)
        ]
    )

    summary, _ = segment_heterogeneity(rows)

    absolute = [item for item in summary if item.value_scale == "absolute"]
    assert len(absolute) == 1
    assert absolute[0].pooled.lb is not None and absolute[0].pooled.ub is not None
    assert absolute[0].pooled.lb < 0.0
    assert absolute[0].pooled.ub > 0.0


def test_absolute_point_only_segment_does_not_block_available_segments():
    point_only = BreakoutEstimate.model_validate(
        {
            **_absolute_primary_estimate("point-only", 0.5, 0.05).model_dump(),
            "lift": Estimate(value=0.5).model_dump(),
        }
    )
    rows = BreakoutEstimates(
        [
            _absolute_primary_estimate("US", 0.4, 0.05),
            _absolute_primary_estimate("CA", 0.6, 0.05),
            point_only,
        ]
    )

    summary, segments = segment_heterogeneity(rows)

    assert len(summary) == 1
    assert summary[0].k == 2
    assert summary[0].pooled.value == pytest.approx(0.5)
    unavailable = next(row for row in segments if row.dimension_value == "point-only")
    assert unavailable.excluded == "zero_variance"


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_absolute_primary_raw_intervals_preserve_t_reference(alpha):
    from scipy.stats import t as student_t

    rows = []
    for label, value in (("US", 0.4), ("CA", 0.6)):
        stored_critical = student_t.isf(0.025, 4)
        rows.append(
            BreakoutEstimate.model_validate(
                {
                    **_absolute_primary_estimate(label, value, 0.05).model_dump(),
                    "reference_kind": "t",
                    "reference_df": 4,
                    "lift": Estimate(
                        value=value,
                        lb=value - stored_critical * 0.05,
                        ub=value + stored_critical * 0.05,
                        level=0.95,
                        alpha=0.05,
                        log_mean=value,
                        log_se=0.05,
                    ).model_dump(),
                }
            )
        )
    _, segments = segment_heterogeneity(BreakoutEstimates(rows), alpha=alpha)
    raw = [row for row in segments if row.estimator == "raw"]
    assert len(raw) == 2
    for row in raw:
        assert row.lift is not None
        half_width = student_t.isf(alpha / 2, 4) * 0.05
        assert row.lift.lb == pytest.approx(row.lift.value - half_width)
        assert row.lift.ub == pytest.approx(row.lift.value + half_width)


def test_absolute_primary_preserves_matching_one_sided_raw_intervals():
    rows = []
    for label, value in (("US", 0.4), ("CA", 0.6)):
        rows.append(
            BreakoutEstimate.model_validate(
                {
                    **_absolute_primary_estimate(label, value, 0.05).model_dump(),
                    "alternative": "greater",
                    "lift": Estimate(
                        value=value,
                        lb=value - norm.isf(0.05) * 0.05,
                        ub=None,
                        open_side="upper",
                        level=0.95,
                        alpha=0.05,
                        log_mean=value,
                        log_se=0.05,
                    ).model_dump(),
                }
            )
        )
    estimates = BreakoutEstimates(rows)
    _, segments = segment_heterogeneity(estimates, alpha=0.05)
    raw = [row for row in segments if row.estimator == "raw"]
    assert len(raw) == 2
    for original, rendered in zip(rows, raw, strict=True):
        assert rendered.lift is not None and original.lift is not None
        assert rendered.lift.lb == pytest.approx(original.lift.lb)
        assert rendered.lift.ub is None
        assert rendered.lift.open_side == "upper"
        assert rendered.excluded is None
    summary, changed = segment_heterogeneity(estimates, alpha=0.1)
    assert summary[0].pooled.value == pytest.approx(0.5)
    assert all(
        row.lift is None and row.excluded == "reference_not_normal"
        for row in changed
        if row.estimator == "raw"
    )


# (dimension_value, log_mean, log_se, abs_diff, abs_se): an extreme
# relative-scale spread over a tight absolute-scale cluster.
_EXTREME_RELATIVE_SPREAD = (
    ("US", 3.0, 0.2, 10.0, 2.0),
    ("CA", -3.0, 0.2, 12.0, 2.0),
    ("MX", 0.0, 0.2, 11.0, 2.0),
)


@dataclass(frozen=True, eq=False)
class _ShrunkenMoments:
    """One scale's shrunken rows, in input order, on the working scale the
    posterior is computed on (log-RR for ``relative``)."""

    theta: np.ndarray
    shrink_k: np.ndarray
    lb: np.ndarray
    ub: np.ndarray


def _shrunken_moments(
    segments: SegmentEstimates, scale: str, labels: Sequence[str]
) -> _ShrunkenMoments:
    emitted = [s for s in segments if s.scale == scale and s.estimator == "shrunken"]
    by_label = {row.dimension_value: row for row in emitted}
    assert len(by_label) == len(emitted), "duplicate shrunken segment"
    assert set(by_label) == set(labels)
    rows = [by_label[label] for label in labels]
    lifts = [row.require_lift() for row in rows]

    def working(values: Sequence[float | None]) -> np.ndarray:
        arr = np.array(values, dtype=float)
        return np.log1p(arr) if scale == "relative" else arr

    return _ShrunkenMoments(
        theta=working([lift.value for lift in lifts]),
        shrink_k=np.array([row.shrink_k for row in rows], dtype=float),
        lb=working([lift.lb for lift in lifts]),
        ub=working([lift.ub for lift in lifts]),
    )


def _assert_shrunken_rows_match_quadrature(
    segments: SegmentEstimates,
    rows: tuple[tuple[str, float, float, float, float], ...],
    tau_prior_scale: float,
) -> None:
    """Both scales' shrunken rows equal continuous quadrature of the same
    model on that scale's moments (log-RR and absolute difference)."""
    for scale, est, var in (
        ("relative", [r[1] for r in rows], [r[2] ** 2 for r in rows]),
        ("absolute", [r[3] for r in rows], [r[4] ** 2 for r in rows]),
    ):
        assert_segment_intervals_match(
            _shrunken_moments(segments, scale, [row[0] for row in rows]),
            segment_posterior(est, var, tau_prior_scale),
        )


class TestMixedEstimandGrouping:
    def test_late_rows_do_not_blank_itt_family_absolute_variance_decomposition(self):
        """ITT and LATE rows share metric/method/group_id but are different
        estimands: LATE rows (no abs_diff, no log_mean) must not count as
        outcome exclusions inside the ITT family. 5 segments (not the
        minimal 2) so ``i2`` clears cochran_q's k>=5 reporting floor."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02), abs_diff=0.5, abs_se=0.1),
                _het_estimate("CA", 0.30, math.sqrt(0.05), abs_diff=0.8, abs_se=0.2),
                _het_estimate("MX", -0.05, math.sqrt(0.01), abs_diff=-0.2, abs_se=0.05),
                _het_estimate("GB", 0.20, math.sqrt(0.03), abs_diff=0.6, abs_se=0.15),
                _het_estimate("DE", 0.15, math.sqrt(0.02), abs_diff=0.4, abs_se=0.1),
                _late_estimate("US", 1.2, 0.3),
                _late_estimate("CA", 1.9, 0.5),
                _late_estimate("MX", 0.8, 0.2),
                _late_estimate("GB", 1.5, 0.4),
                _late_estimate("DE", 1.1, 0.3),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)

        itt_abs = [s for s in summary if s.estimand == "itt" and s.scale == "absolute"]
        assert len(itt_abs) == 1
        s = itt_abs[0]
        assert s.n_excluded_outcome == 0
        assert s.tau2 is not None
        assert s.i2 is not None

        late_abs = [row for row in summary if row.estimand == "late"]
        assert len(late_abs) == 1
        assert late_abs[0].scale == "absolute"
        assert late_abs[0].k == 5
        assert late_abs[0].n_excluded_outcome == 0
        assert {(row.estimand, row.scale) for row in segments} == {
            ("itt", "relative"),
            ("itt", "absolute"),
            ("late", "absolute"),
        }

    def test_segment_contrast_refuses_cross_estimand_pair(self):
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _late_estimate("GB", 1.2, 0.3),
            ]
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(estimates, "US", "GB")
        assert exc_info.value.code == "breakout.segment_contrast_dimension"


class TestSegmentHeterogeneity:
    def test_hand_computed_relative_scale_q_i2_tau2(self):
        """3-segment fixture with hand-computed Q/tau2/p_value, run through the full model-based adapter (same numbers the cochran_q unit test verifies directly)."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        assert len(summary) == 1  # abs_diff/abs_se unset -> absolute scale skipped
        s = summary[0]
        assert s.scale == "relative"
        assert s.k == 3
        assert s.q == pytest.approx(2.338235294117647)
        assert s.p_value == pytest.approx(0.3106409153020779)
        assert s.tau2 == pytest.approx(0.003593750000000001)
        assert s.n_excluded_design == 0
        assert s.n_excluded_outcome == 0
        # 2 rows (raw + shrunken) per segment.
        assert len(segments) == 6
        assert {seg.dimension_value for seg in segments} == {"US", "CA", "MX"}
        assert {seg.estimator for seg in segments} == {"raw", "shrunken"}

    def test_two_treatment_arms_produce_two_independent_summaries(self):
        """A two-arm breakout: group_id is part of the grouping key, so each arm's Q statistic is computed independently - merging them previously inflated Q by 17.1%."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02), group_id="treatment_a"),
                _het_estimate("CA", 0.30, math.sqrt(0.05), group_id="treatment_a"),
                _het_estimate("MX", -0.05, math.sqrt(0.01), group_id="treatment_a"),
                # Different data for treatment_b - if the two arms were merged
                # into one Q, treatment_a's Q would change too.
                _het_estimate("US", 0.50, math.sqrt(0.03), group_id="treatment_b"),
                _het_estimate("CA", -0.20, math.sqrt(0.02), group_id="treatment_b"),
                _het_estimate("MX", 0.15, math.sqrt(0.015), group_id="treatment_b"),
            ]
        )
        summary, _ = segment_heterogeneity(estimates)
        assert len(summary) == 2
        by_group = {s.group_id: s for s in summary}
        assert set(by_group) == {"treatment_a", "treatment_b"}
        # treatment_a's Q is exactly the hand-computed fixture value -
        # unaffected by treatment_b's presence in `estimates`.
        assert by_group["treatment_a"].q == pytest.approx(2.338235294117647)
        assert by_group["treatment_b"].q != pytest.approx(by_group["treatment_a"].q)

    def test_both_scales_shipped_when_both_populated(self):
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02), abs_diff=1.0, abs_se=0.1),
                _het_estimate("CA", 0.30, math.sqrt(0.05), abs_diff=2.5, abs_se=0.2),
                _het_estimate("MX", -0.05, math.sqrt(0.01), abs_diff=-0.4, abs_se=0.08),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        assert {s.scale for s in summary} == {"relative", "absolute"}
        assert len(segments) == 12  # 2 scales x 3 segments x 2 estimators

    def test_baseline_recovered_exactly(self):
        """baseline = abs_diff / expm1(log_mean) recovers the control mean exactly when both scales agree on the same underlying arms."""
        c_mean, lift_frac = 10.0, 0.20  # treatment = 12.0
        log_mean = math.log1p(lift_frac)
        abs_diff = c_mean * lift_frac
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", log_mean, 0.05, abs_diff=abs_diff, abs_se=0.3),
                _het_estimate("CA", 0.10, 0.04, abs_diff=1.0, abs_se=0.2),
            ]
        )
        _, segments = segment_heterogeneity(estimates)
        us_raw = next(s for s in segments if s.dimension_value == "US" and s.estimator == "raw")
        assert us_raw.baseline == pytest.approx(c_mean)

    def test_baseline_none_at_true_null(self):
        """log_mean == 0 (true null) makes the baseline identity divide by zero - result is None, not inf/nan."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.0, 0.05, abs_diff=0.0, abs_se=0.3),
                _het_estimate("CA", 0.10, 0.04, abs_diff=1.0, abs_se=0.2),
            ]
        )
        _, segments = segment_heterogeneity(estimates)
        us_raw = next(s for s in segments if s.dimension_value == "US" and s.estimator == "raw")
        assert us_raw.baseline is None

    def test_shrink_k_only_on_shrunken_rows(self):
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
            ]
        )
        _, segments = segment_heterogeneity(estimates)
        for seg in segments:
            if seg.estimator == "raw":
                assert seg.shrink_k is None
            else:
                assert seg.shrink_k is not None
                assert 0.0 < seg.shrink_k

    @pytest.mark.parametrize(
        "absolute_rows",
        [
            pytest.param(
                (
                    ("US", 0.10, math.sqrt(0.02), 120.0, 10.0),
                    ("CA", 0.30, math.sqrt(0.05), 40.0, 10.0),
                    ("MX", -0.05, math.sqrt(0.01), -60.0, 10.0),
                ),
                id="moderate_absolute_effects",
            ),
            pytest.param(
                (
                    ("US", 0.10, math.sqrt(0.02), 200.0, 30.0),
                    ("CA", 0.30, math.sqrt(0.05), -150.0, 40.0),
                    ("MX", -0.05, math.sqrt(0.01), 100.0, 20.0),
                ),
                id="large_absolute_effects",
            ),
        ],
    )
    def test_spread_far_beyond_the_prior_scale_is_shrunk_not_withheld(self, absolute_rows):
        """Absolute-scale spreads hundreds of times the default prior scale are
        a conflict between prior and data, not an integration failure: the
        posterior under the same model is resolvable, so every shrunken row
        ships on both scales, without a warning, and matches continuous
        quadrature of that model. Raw rows are untouched."""
        estimates = BreakoutEstimates(
            [
                _het_estimate(
                    dimension_value,
                    log_mean,
                    log_se,
                    abs_diff=abs_diff,
                    abs_se=abs_se,
                )
                for dimension_value, log_mean, log_se, abs_diff, abs_se in absolute_rows
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _, segments = segment_heterogeneity(estimates, tau_prior_scale=0.30)
        _assert_shrunken_rows_match_quadrature(segments, absolute_rows, 0.30)

        abs_raw = [s for s in segments if s.scale == "absolute" and s.estimator == "raw"]
        assert len(abs_raw) == len(absolute_rows)
        expected_raw = {
            dimension_value: (abs_diff, abs_se)
            for dimension_value, _, _, abs_diff, abs_se in absolute_rows
        }
        z = norm.ppf(0.975)
        for s in abs_raw:
            abs_diff, abs_se = expected_raw[s.dimension_value]
            assert s.excluded is None
            lift = s.lift
            assert lift is not None
            assert lift.value == pytest.approx(abs_diff)
            assert lift.lb == pytest.approx(abs_diff - z * abs_se)
            assert lift.ub == pytest.approx(abs_diff + z * abs_se)

    def test_relative_spread_above_the_former_tau2_ceiling_is_shrunk(self):
        """log-RR +3/-3/0 at se 0.2 puts DL tau2 near 8.96, above
        (8 * 0.30)^2; the posterior under the same prior is still resolvable,
        so both scales' shrunken rows ship and match continuous quadrature."""
        rows = _EXTREME_RELATIVE_SPREAD
        estimates = BreakoutEstimates(
            [_het_estimate(v, lm, ls, abs_diff=ad, abs_se=ase) for v, lm, ls, ad, ase in rows]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            summary, segments = segment_heterogeneity(estimates, tau_prior_scale=0.30)
        rel_summary = next(s for s in summary if s.scale == "relative")
        assert rel_summary.tau2 == pytest.approx(8.96, rel=1e-2)
        _assert_shrunken_rows_match_quadrature(segments, rows, 0.30)

    def test_tau_prior_scale_is_forwarded_to_the_shrinkage_posterior(self):
        """`tau_prior_scale` (mirroring `segment_rollout_recommendation`'s
        parameter) sets the prior of every shrunken row on both scales: the
        same rows match quadrature under the passed scale."""
        rows = _EXTREME_RELATIVE_SPREAD
        estimates = BreakoutEstimates(
            [_het_estimate(v, lm, ls, abs_diff=ad, abs_se=ase) for v, lm, ls, ad, ase in rows]
        )
        _, segments = segment_heterogeneity(estimates, tau_prior_scale=2.0)
        _assert_shrunken_rows_match_quadrature(segments, rows, 2.0)

    def test_posterior_hundreds_of_times_narrower_than_the_prior_is_resolved(self):
        """Conversion-rate differences with standard errors near 0.001 under
        the default prior put the whole tau posterior below ~0.003, a sliver
        of the prior's scale; the shrunken rows must still match continuous
        quadrature rather than collapse onto the pooled mean."""
        rows = (
            ("US", 0.100, 0.02, 0.0200, 0.001),
            ("CA", 0.120, 0.02, 0.0215, 0.001),
            ("MX", 0.090, 0.02, 0.0190, 0.001),
            ("GB", 0.110, 0.02, 0.0208, 0.001),
            ("DE", 0.095, 0.02, 0.0193, 0.001),
        )
        estimates = BreakoutEstimates(
            [_het_estimate(v, lm, ls, abs_diff=ad, abs_se=ase) for v, lm, ls, ad, ase in rows]
        )
        _, segments = segment_heterogeneity(estimates, tau_prior_scale=0.30)
        _assert_shrunken_rows_match_quadrature(segments, rows, 0.30)

    def test_unresolved_posterior_withholds_only_the_shrunken_rows(self, monkeypatch):
        """A posterior the node budget cannot resolve withholds that scale's
        shrunken rows under the existing unavailable-row convention
        (``lift=None``, ``shrink_k=None``, ``excluded="estimation_failed"``)
        and says so with one coded warning per scale; raw rows and the
        summary stay live. The budget is shrunk to reach that path
        deterministically."""
        monkeypatch.setattr("increment.estimation.meta._TAU_NODE_BUDGET", 1)
        rows = _EXTREME_RELATIVE_SPREAD
        estimates = BreakoutEstimates(
            [_het_estimate(v, lm, ls, abs_diff=ad, abs_se=ase) for v, lm, ls, ad, ase in rows]
        )
        with pytest.warns(IncrementWarning) as rec:
            summary, segments = segment_heterogeneity(estimates, tau_prior_scale=0.30)
        codes = warning_codes(rec)
        assert codes.count("breakout.heterogeneity.posterior_integration_unresolved") == 2
        assert set(codes) == {"breakout.heterogeneity.posterior_integration_unresolved"}
        assert {s.scale for s in summary} == {"relative", "absolute"}
        shrunken = [s for s in segments if s.estimator == "shrunken"]
        raw = [s for s in segments if s.estimator == "raw"]
        assert len(shrunken) == len(raw) == 2 * len(rows)
        for s in shrunken:
            assert s.excluded == "estimation_failed"
            assert s.lift is None
            assert s.shrink_k is None
        for s in raw:
            assert s.excluded is None
            assert s.lift is not None

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0])
    def test_non_positive_tau_prior_scale_fails_fast(self, bad_scale):
        """An out-of-range tau_prior_scale is refused up front with the
        facade's own code, before any key's posterior is attempted, rather
        than surfacing as a data-dependent per-key outcome."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 3.0, 0.2),
                _het_estimate("CA", -3.0, 0.2),
                _het_estimate("MX", 0.0, 0.2),
            ]
        )
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_heterogeneity(estimates, tau_prior_scale=bad_scale)
        assert exc_info.value.code == "breakout.tau_prior_scale"
        assert exc_info.value.context["tau_prior_scale"] == bad_scale

    def test_raw_row_reuses_original_lift_on_relative_scale(self):
        """The 'raw' relative-scale row is the segment's original lift, not recomputed - exact object-level equality of value/lb/ub."""
        original = _het_estimate("US", 0.10, math.sqrt(0.02))
        estimates = BreakoutEstimates(
            [
                original,
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
            ]
        )
        _, segments = segment_heterogeneity(estimates)
        us_raw = next(s for s in segments if s.dimension_value == "US" and s.estimator == "raw")
        us_raw_lift = us_raw.lift
        original_lift = original.lift
        assert us_raw_lift is not None and original_lift is not None
        assert us_raw_lift.value == original_lift.value
        assert us_raw_lift.lb == original_lift.lb
        assert us_raw_lift.ub == original_lift.ub

    def test_d7_outcome_exclusion_suppresses_tau2_and_i2_not_q(self):
        """An outcome-based exclusion (zero_variance/extreme_ratio/nonpositive_mean) suppresses tau2/i2/i2_lb/i2_ub but not q/p_value/pooled - only tau^2/I^2 are biased by outcome-based drops."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
                _het_estimate("BR", 0.0, 0.0, excluded="zero_variance"),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        s = next(x for x in summary if x.scale == "relative")
        assert s.n_excluded_outcome == 1
        assert s.n_excluded_design == 0
        assert s.tau2 is None
        assert s.i2 is None
        assert s.i2_lb is None
        assert s.i2_ub is None
        assert s.q == pytest.approx(2.338235294117647)  # unaffected
        assert math.isfinite(s.pooled.value)  # unaffected
        # BR still gets dense rows, excluded and NaN.
        br_rows = [seg for seg in segments if seg.dimension_value == "BR"]
        assert len(br_rows) == 2
        assert {seg.estimator for seg in br_rows} == {"raw", "shrunken"}
        for seg in br_rows:
            assert seg.excluded == "zero_variance"
            assert seg.lift is None
            assert seg.baseline is None

    def test_design_exclusion_does_not_suppress_tau2(self):
        """A design-based exclusion (few_units/no_control_arm) does not bias tau^2 the way an outcome-based one does - not suppressed."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
                _het_estimate("BR", 0.0, 0.0, excluded="few_units"),
            ]
        )
        summary, _ = segment_heterogeneity(estimates)
        s = next(x for x in summary if x.scale == "relative")
        assert s.n_excluded_design == 1
        assert s.n_excluded_outcome == 0
        assert s.tau2 is not None

    def test_fewer_than_two_live_segments_skipped_not_raised(self):
        """A key with 0 or 1 live segments produces no summary/segment rows for it, rather than raising."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.0, 0.0, excluded="no_control_arm"),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        assert len(summary) == 0
        assert len(segments) == 0

    def test_empty_input_returns_empty_result(self):
        summary, segments = segment_heterogeneity(BreakoutEstimates([]))
        assert len(summary) == 0
        assert len(segments) == 0

    def test_zero_variance_live_segment_skipped_not_raised(self):
        """A live (not excluded) segment with log_se == 0.0 (e.g. an upstream se
        clamp) makes that scale's var non-positive - must be skipped gracefully like the None-field case, not surfaced as cochran_q's bare ValueError."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.0, 0.0),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        assert len(summary) == 0
        assert len(segments) == 0

    def test_bad_variance_segment_dropped_alone_not_whole_scale(self):
        """A single live segment with non-positive/non-finite variance on a scale
        used to `continue` the WHOLE scale for the whole key: every OTHER live segment lost its rows too, and the key
        lost its summary row entirely. Now only the offending segment is dropped - the other live segments still get
        real summary/segment rows, and the bad one gets a dense NaN pair tagged excluded="zero_variance", not silently missing."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", 0.0, 0.0),  # log_se == 0.0 -> bad variance
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        assert len(summary) == 1
        s = summary[0]
        assert s.scale == "relative"
        assert s.k == 2  # US, CA only
        assert s.n_excluded_outcome == 1  # MX counted like an outcome exclusion
        assert s.tau2 is None  # suppressed by the (now nonzero) n_excluded_outcome

        by_value_estimator = {(seg.dimension_value, seg.estimator): seg for seg in segments}
        assert set(by_value_estimator) == {
            ("US", "raw"),
            ("US", "shrunken"),
            ("CA", "raw"),
            ("CA", "shrunken"),
            ("MX", "raw"),
            ("MX", "shrunken"),
        }
        assert by_value_estimator[("US", "raw")].excluded is None
        us_raw_lift = by_value_estimator[("US", "raw")].lift
        assert us_raw_lift is not None
        assert math.isfinite(us_raw_lift.value)
        assert by_value_estimator[("CA", "shrunken")].excluded is None
        assert by_value_estimator[("CA", "shrunken")].shrink_k is not None
        mx_raw = by_value_estimator[("MX", "raw")]
        mx_shrunken = by_value_estimator[("MX", "shrunken")]
        assert mx_raw.excluded == "zero_variance"
        assert mx_shrunken.excluded == "zero_variance"
        assert mx_shrunken.lift is None
        assert mx_shrunken.shrink_k is None

    def test_bad_variance_on_absolute_scale_only_leaves_relative_untouched(self):
        """The per-scale partition exists so a segment can be bad on one scale and good on the other: MX's abs_se is 0.0, so it must be dropped from the absolute scale only - its relative-scale log_mean/log_se are fine."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02), abs_diff=1.0, abs_se=0.2),
                _het_estimate("CA", 0.30, math.sqrt(0.05), abs_diff=2.0, abs_se=0.3),
                _het_estimate("MX", -0.05, math.sqrt(0.01), abs_diff=-0.5, abs_se=0.0),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        by_scale = {s.scale: s for s in summary}
        assert set(by_scale) == {"relative", "absolute"}
        assert by_scale["relative"].k == 3
        assert by_scale["relative"].n_excluded_outcome == 0
        assert by_scale["absolute"].k == 2
        assert by_scale["absolute"].n_excluded_outcome == 1

        by_key = {(s.dimension_value, s.scale, s.estimator): s for s in segments}
        assert by_key[("MX", "relative", "raw")].excluded is None
        mx_raw_lift = by_key[("MX", "relative", "raw")].lift
        assert mx_raw_lift is not None
        assert math.isfinite(mx_raw_lift.value)
        assert by_key[("MX", "relative", "shrunken")].excluded is None
        assert by_key[("MX", "absolute", "raw")].excluded == "zero_variance"
        assert by_key[("MX", "absolute", "raw")].lift is None
        assert by_key[("MX", "absolute", "shrunken")].excluded == "zero_variance"
        assert by_key[("MX", "absolute", "shrunken")].lift is None

    def test_degenerate_hksj_pooled_variance_falls_back_not_raises(self):
        """hksj_pooled_mean deliberately raises when the pooled variance is
        degenerate (segment estimates numerically indistinguishable given their weights) - a real, rare tail case, not
        a bug. That ValueError used to propagate out of segment_heterogeneity and abort the whole call, taking every
        unrelated group's rows down with it. One degenerate group alongside one heterogeneous group must not raise, and the heterogeneous group's results must come back untouched."""
        estimates = BreakoutEstimates(
            [
                # Identical (log_mean, log_se) on every live segment makes the
                # HKSJ pooled variance exactly 0, below hksj_pooled_mean's degeneracy floor.
                _het_estimate("US", 0.20, math.sqrt(0.01), group_id="treatment_a"),
                _het_estimate("CA", 0.20, math.sqrt(0.01), group_id="treatment_a"),
                _het_estimate("MX", 0.20, math.sqrt(0.01), group_id="treatment_a"),
                # The standard hand-computed fixture - must be unaffected.
                _het_estimate("US", 0.10, math.sqrt(0.02), group_id="treatment_b"),
                _het_estimate("CA", 0.30, math.sqrt(0.05), group_id="treatment_b"),
                _het_estimate("MX", -0.05, math.sqrt(0.01), group_id="treatment_b"),
            ]
        )
        with pytest.warns(IncrementWarning) as rec:
            summary, segments = segment_heterogeneity(estimates)
        assert "breakout.heterogeneity.hksj_degenerate_variance_fallback" in warning_codes(rec)

        assert len(summary) == 2
        by_group = {s.group_id: s for s in summary}
        assert set(by_group) == {"treatment_a", "treatment_b"}

        a = by_group["treatment_a"]
        assert a.pooled.value is not None
        assert a.pooled.lb is not None
        assert a.pooled.ub is not None
        assert math.isfinite(a.pooled.value)
        assert math.isfinite(a.pooled.lb)
        assert math.isfinite(a.pooled.ub)
        assert a.pooled.lb <= a.pooled.value <= a.pooled.ub

        b = by_group["treatment_b"]
        assert b.q == pytest.approx(2.338235294117647)  # exact, from the sibling fixture
        assert b.pooled.value is not None
        assert math.isfinite(b.pooled.value)

        a_segments = [s for s in segments if s.group_id == "treatment_a"]
        b_segments = [s for s in segments if s.group_id == "treatment_b"]
        assert len(a_segments) == 6  # 3 segments x 2 estimators, relative scale only
        assert len(b_segments) == 6


class TestSegmentHeterogeneityRawShrunkenPriorContract:
    """The relative-scale `raw` row reuses `BreakoutEstimate.lift` verbatim (the
    prior-updated posterior median), while Q/tau^2/pooled/shrunken machinery is built from the raw pre-update
    `lift.log_mean`/`lift.log_se` - under an informative prior those diverge. Verbatim reuse holds only when
    `lift.level == 1 - alpha`; any other upstream level (Bonferroni, one-sided) rebuilds the raw row at the requested alpha, so the package's own FWER correction composes with pooling."""

    def test_raw_row_diverges_from_log_mean_basis_under_informative_prior(self):
        informative = Normal(mu=0.0, sigma=0.01)
        estimates = _run_breakout_two_segment_fixture(prior=informative)
        _, segments = segment_heterogeneity(estimates, alpha=0.05)
        us_raw = next(
            s
            for s in segments
            if s.dimension_value == "US" and s.scale == "relative" and s.estimator == "raw"
        )
        us_row = next(r for r in estimates if r.dimension_value == "US")
        us_lift = us_row.lift
        assert us_lift is not None
        assert us_lift.log_mean is not None
        naive = math.expm1(us_lift.log_mean)
        # raw row is the prior-updated posterior median, pulled toward the tight
        # prior's mean of 0 - far from the raw log_mean basis the Q/tau^2/shrunken block uses.
        us_raw_lift = us_raw.lift
        assert us_raw_lift is not None
        assert abs(us_raw_lift.value - naive) / abs(naive) > 0.5

    def test_bonferroni_output_composes_with_nominal_alpha(self):
        """run_breakout(correction='bonferroni') stamps lift.level=1-alpha/K
        (0.975 at K=2); segment_heterogeneity(alpha=0.05) must accept it and
        rebuild the raw rows as 95% intervals from log_mean/log_se at each
        row's OWN sampling reference - refusing here made the package's own
        FWER fix unusable, and rebuilding at a Normal quantile would narrow a
        Welch row by the z/t ratio."""
        estimates = _run_breakout_two_segment_fixture(correction="bonferroni")
        live = [r for r in estimates if r.excluded is None]
        assert live
        for r in live:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.975)  # precondition: corrected upstream level
            assert r.reference_kind == "t" and r.reference_df is not None

        _, segments = segment_heterogeneity(estimates, alpha=0.05)

        raw = [s for s in segments if s.scale == "relative" and s.estimator == "raw"]
        assert len(raw) == 2
        z95 = norm.ppf(0.975)
        for s in raw:
            src_row = next(r for r in live if r.dimension_value == s.dimension_value)
            src_lift = src_row.lift
            lift = s.lift
            assert src_lift is not None and lift is not None
            log_mean, log_se = src_lift.log_mean, src_lift.log_se
            assert log_mean is not None and log_se is not None
            assert src_row.reference_df is not None
            crit = t_dist.isf(0.025, src_row.reference_df)
            assert crit > z95  # the reference this rebuild must not swap out
            # Rebuilt at the requested alpha from level-independent sufficient
            # statistics - exact, no tolerance beyond float.
            assert lift.level == pytest.approx(0.95)
            assert lift.value == pytest.approx(math.expm1(log_mean))
            assert lift.lb == pytest.approx(math.expm1(log_mean - crit * log_se))
            assert lift.ub == pytest.approx(math.expm1(log_mean + crit * log_se))
            # ...and strictly NARROWER than the corrected upstream interval.
            assert src_lift.lb is not None and src_lift.ub is not None
            assert lift.lb is not None and lift.lb > src_lift.lb
            assert lift.ub is not None and lift.ub < src_lift.ub

    def test_one_sided_output_refuses_a_mismatched_alpha(self):
        """A one-sided run_breakout stamps level=1-2*alpha (0.90); pooling at
        a different alpha=0.05 must refuse rather than silently rebuild a
        two-sided 95% interval from a one-sided row - the naive rebuild
        used to discard `alternative` and misrepresent the row's shape."""
        estimates = _run_breakout_two_segment_fixture(alternative="greater")
        live = [r for r in estimates if r.excluded is None]
        assert live
        for r in live:
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.90)

        with pytest.raises(InvalidRequestError) as exc_info:
            segment_heterogeneity(estimates, alpha=0.05)
        assert exc_info.value.code == "breakout.segment_heterogeneity_rebuild"
        assert exc_info.value.context["alternative"] == "greater"

    def test_matching_alpha_does_not_raise(self):
        estimates = _run_breakout_two_segment_fixture()
        summary, segments = segment_heterogeneity(estimates, alpha=0.05)
        assert len(summary) >= 1
        assert len(segments) >= 1

    def test_sequential_heterogeneity_refuses_admitted_bernoulli_covariance(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import registered_bernoulli

        snapshot, policy = registered_bernoulli(
            n=4, segments=((("country", "US"),), (("country", "GB"),))
        )
        estimates = run_breakout(
            snapshot,
            [ConversionMetric(name="revenue", entity="unit", fact="conversion")],
            control_group="control",
            dimension="country",
            inference=policy,
        )
        assert {row.dimension_value for row in estimates} == {"US", "GB"}
        with pytest.raises(CapabilityError) as raised:
            segment_heterogeneity(estimates, alpha=0.05)
        assert raised.value.code == "sequential.route.unsupported"

    def test_always_valid_gaussian_row_refuses_before_alpha_rebuild(self):
        from increment.errors import CapabilityError
        from tests.sequential_cases import raw_gaussian

        snapshot, metrics, policy = raw_gaussian(segments=((("country", "US"),),), alpha=0.10)
        with pytest.raises(CapabilityError) as raised:
            run_breakout(
                snapshot,
                metrics,
                control_group="control",
                dimension="country",
                inference=policy,
                alpha=0.10,
            )
        assert raised.value.code == "sequential.route.unsupported"

    def test_clustered_row_rebuilds_at_its_own_t_reference_not_a_normal_one(self):
        """A cluster-robust row's interval is a t_dof quantile pair. Rebuilding
        it at another alpha must use t_dof again; a Normal quantile would be
        too narrow. Exercises `_relative_raw_lift` directly (dof has no
        `run_breakout` fixture path here) at the good_rows call site's default
        strict=True."""
        row = _breakout_estimate("US", 0.20, 0.05, level=0.90, dof=8.0)
        rebuilt = _relative_raw_lift(row, alpha=0.05)

        assert rebuilt is not None
        crit = t_dist.isf(0.025, 8.0)
        assert rebuilt.value == pytest.approx(math.expm1(0.20))
        assert rebuilt.lb == pytest.approx(math.expm1(0.20 - crit * 0.05))
        assert rebuilt.ub == pytest.approx(math.expm1(0.20 + crit * 0.05))
        # At df=8 the Normal quantile is ~15% smaller, so the wrong reference
        # is not hidden by the tolerance above.
        assert rebuilt.lb is not None and rebuilt.ub is not None
        assert rebuilt.lb < math.expm1(0.20 - _Z95 * 0.05)
        assert rebuilt.ub > math.expm1(0.20 + _Z95 * 0.05)

    def test_segment_heterogeneity_rebuilds_each_row_at_its_own_reference(self):
        """A Welch row and a Normal sibling in the same family are rebuilt at
        the requested alpha on their OWN references -- t_5.5 and z -- not both
        on a Normal one."""
        estimates = BreakoutEstimates(
            [
                _breakout_estimate(
                    "US",
                    0.20,
                    0.05,
                    level=0.90,
                    dof=None,
                    reference_kind="t",
                    reference_df=5.5,
                ),
                _breakout_estimate("CA", 0.05, 0.04, level=0.90),
            ]
        )
        _, segments = segment_heterogeneity(estimates, alpha=0.05)

        raw = {
            s.dimension_value: s for s in segments if s.scale == "relative" and s.estimator == "raw"
        }
        us, ca = raw["US"].require_lift(), raw["CA"].require_lift()
        welch_crit = t_dist.isf(0.025, 5.5)
        assert welch_crit > _Z95
        assert us.lb == pytest.approx(math.expm1(0.20 - welch_crit * 0.05))
        assert us.ub == pytest.approx(math.expm1(0.20 + welch_crit * 0.05))
        assert ca.lb == pytest.approx(math.expm1(0.05 - _Z95 * 0.04))
        assert ca.ub == pytest.approx(math.expm1(0.05 + _Z95 * 0.04))

    def test_refused_row_withholds_when_not_strict(self):
        """Informational-only rebuilds withhold the row (``lift=None``)
        instead of aborting the whole call. MX is dropped from the scale's
        math (log_se == 0.0) and its one-sided interval at a mismatched
        level cannot be rebuilt two-sided, so its raw relative row is
        withheld while its siblings keep theirs."""
        one_sided = _het_estimate("MX", 0.20, 0.0, level=0.90).model_copy(
            update={"alternative": "greater"}
        )
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                one_sided,
            ]
        )
        summary, segments = segment_heterogeneity(estimates, alpha=0.05)
        assert len(summary) == 1
        by_value_estimator = {(s.dimension_value, s.estimator): s for s in segments}
        mx_raw = by_value_estimator[("MX", "raw")]
        assert mx_raw.excluded == "zero_variance"
        assert mx_raw.lift is None
        assert by_value_estimator[("US", "raw")].lift is not None
        assert by_value_estimator[("CA", "raw")].lift is not None

    def test_bh_selected_row_is_withheld_not_re_narrowed_by_segment_heterogeneity(self):
        """End-to-end: a BH-selected segment's corrected level (0.99, a
        tighter alpha than the pooling call's nominal 0.05) must not be
        silently rebuilt into a 95% interval by segment_heterogeneity --
        its raw relative-scale row is withheld instead of undoing the
        correction on exactly the segment selected by looking at the
        data, while its BH bookkeeping and its siblings' own lifts stay
        intact."""
        us = _het_estimate("US", 0.20, math.sqrt(0.02), level=0.99).model_copy(
            update={
                "family_axes": ("metric", "arm", "segment"),
                "family_q": 0.05,
                "family_threshold": 0.01,
                "discovery": True,
            }
        )
        estimates = BreakoutEstimates(
            [
                us,
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
            ]
        )
        _, segments = segment_heterogeneity(estimates, alpha=0.05)
        us_raw = next(
            s
            for s in segments
            if s.dimension_value == "US" and s.scale == "relative" and s.estimator == "raw"
        )
        assert us_raw.lift is None
        assert us_raw.excluded == "reference_not_normal"
        assert us_raw.discovery is True
        assert us_raw.family_q == 0.05
        assert us_raw.family_threshold == 0.01
        siblings = [
            s
            for s in segments
            if s.dimension_value in {"CA", "MX"} and s.scale == "relative" and s.estimator == "raw"
        ]
        assert len(siblings) == 2
        assert all(s.lift is not None and s.excluded is None for s in siblings)

    def test_bh_selected_row_withholds_its_absolute_raw_row_too(self):
        """A row re-estimated at the FCR alpha on both scales must not have
        its absolute raw interval silently rebuilt at the pooling call's
        nominal alpha -- that would undo the same selection correction the
        relative scale already withholds rather than undoes."""
        us = _het_estimate(
            "US", 0.20, math.sqrt(0.02), abs_diff=2.0, abs_se=0.5, level=0.99
        ).model_copy(
            update={
                "family_axes": ("metric", "arm", "segment"),
                "family_q": 0.05,
                "family_threshold": 0.01,
                "discovery": True,
            }
        )
        estimates = BreakoutEstimates(
            [
                us,
                _het_estimate("CA", 0.30, math.sqrt(0.05), abs_diff=3.0, abs_se=0.6),
                _het_estimate("MX", -0.05, math.sqrt(0.01), abs_diff=-0.5, abs_se=0.3),
            ]
        )
        _, segments = segment_heterogeneity(estimates, alpha=0.05)
        us_abs_raw = next(
            s
            for s in segments
            if s.dimension_value == "US" and s.scale == "absolute" and s.estimator == "raw"
        )
        assert us_abs_raw.lift is None
        assert us_abs_raw.excluded == "reference_not_normal"
        assert us_abs_raw.discovery is True
        assert us_abs_raw.family_q == 0.05
        assert us_abs_raw.family_threshold == 0.01
        siblings = [
            s
            for s in segments
            if s.dimension_value in {"CA", "MX"} and s.scale == "absolute" and s.estimator == "raw"
        ]
        assert len(siblings) == 2
        assert all(s.lift is not None and s.excluded is None for s in siblings)

    def test_bh_selected_row_matching_alpha_still_reuses_verbatim(self):
        """When the BH-corrected level happens to already match the
        pooling call's alpha (the capped-FCR case, common when R/m is
        large), no rebuild is needed at all -- verbatim reuse, same as
        any other row, and the refusal path is never reached."""
        us = _het_estimate("US", 0.20, math.sqrt(0.02), level=0.95).model_copy(
            update={
                "family_axes": ("metric", "arm", "segment"),
                "family_q": 0.10,
                "family_threshold": 0.10,
                "discovery": True,
            }
        )
        estimates = BreakoutEstimates(
            [
                us,
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
            ]
        )
        _, segments = segment_heterogeneity(estimates, alpha=0.05)
        us_raw = next(
            s
            for s in segments
            if s.dimension_value == "US" and s.scale == "relative" and s.estimator == "raw"
        )
        us_raw_lift = us_raw.lift
        us_lift = us.lift
        assert us_raw_lift is not None and us_lift is not None
        assert us_raw_lift.value == us_lift.value
        assert us_raw_lift.lb == us_lift.lb
        assert us_raw_lift.ub == us_lift.ub

    def test_bh_selected_row_matching_alpha_keeps_its_absolute_raw_row_too(self):
        """Symmetric with the relative scale's own matching-alpha reuse:
        when the FCR-corrected level already equals the pooling call's
        alpha, the absolute raw row needs no correction-preserving
        withhold either -- it is built at that already-matching alpha
        instead of withheld, just like an uncorrected row would be."""
        us = _het_estimate(
            "US", 0.20, math.sqrt(0.02), abs_diff=2.0, abs_se=0.5, level=0.95
        ).model_copy(
            update={
                "family_axes": ("metric", "arm", "segment"),
                "family_q": 0.10,
                "family_threshold": 0.10,
                "discovery": True,
            }
        )
        estimates = BreakoutEstimates(
            [
                us,
                _het_estimate("CA", 0.30, math.sqrt(0.05), abs_diff=3.0, abs_se=0.6),
                _het_estimate("MX", -0.05, math.sqrt(0.01), abs_diff=-0.5, abs_se=0.3),
            ]
        )
        _, segments = segment_heterogeneity(estimates, alpha=0.05)
        us_abs_raw = next(
            s
            for s in segments
            if s.dimension_value == "US" and s.scale == "absolute" and s.estimator == "raw"
        )
        assert us_abs_raw.excluded is None
        assert us_abs_raw.lift is not None
        z95 = norm.ppf(0.975)
        assert us_abs_raw.require_lift().value == pytest.approx(2.0)
        assert us_abs_raw.require_lift().lb == pytest.approx(2.0 - z95 * 0.5)
        assert us_abs_raw.require_lift().ub == pytest.approx(2.0 + z95 * 0.5)


class TestHeterogeneitySummaryToFrame:
    """Two-model to_frame schema, measured on both backends including the empty-list case - not just asserted to resolve cleanly."""

    @pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
    def test_empty_list_schema_resolves(self, backend):
        summary_frame = HeterogeneitySummaries([]).to_frame(backend=backend)
        assert len(summary_frame) == 0
        segments_frame = SegmentEstimates([]).to_frame(backend=backend)
        assert len(segments_frame) == 0

    @pytest.mark.parametrize("backend", ["pandas", "pyarrow"])
    def test_non_empty_schema_resolves_and_i2_suppression_is_nan(self, backend):
        """float | None -> Float64 NaN in the frame - exactly what I^2 suppression (k<5 here) should look like as a column."""
        estimates = BreakoutEstimates(
            [
                _het_estimate("US", 0.10, math.sqrt(0.02)),
                _het_estimate("CA", 0.30, math.sqrt(0.05)),
                _het_estimate("MX", -0.05, math.sqrt(0.01)),
            ]
        )
        summary, segments = segment_heterogeneity(estimates)
        summary_frame = summary.to_frame(backend=backend)
        assert len(summary_frame) == 1
        segments_frame = segments.to_frame(backend=backend)
        assert len(segments_frame) == 6

    def test_summary_columns_are_field_order_with_estimand_value_scale_ahead_of_scale(self):
        """Pins to_frame's column order: `estimand`/`value_scale` sit between
        `source` and `scale`, matching HeterogeneitySummary's field order."""
        frame = HeterogeneitySummaries([]).to_frame()
        assert list(frame.columns) == [
            "metric",
            "method",
            "group_id",
            "dimension",
            "source",
            "method_role",
            "estimand",
            "value_scale",
            "scale",
            "k",
            "q",
            "p_value",
            "tau2",
            "i2",
            "i2_lb",
            "i2_ub",
            "n_excluded_design",
            "n_excluded_outcome",
            "pooled",
            "lb",
            "ub",
            "open_side",
        ]

    def test_segment_columns_are_field_order_with_estimand_value_scale_ahead_of_scale(self):
        """Pins to_frame's column order: `estimand`/`value_scale` sit between
        `source` and `scale`, matching SegmentEstimate's field order."""
        frame = SegmentEstimates([]).to_frame()
        assert list(frame.columns) == [
            "metric",
            "method",
            "method_role",
            "group_id",
            "dimension",
            "dimension_value",
            "source",
            "estimand",
            "value_scale",
            "scale",
            "estimator",
            "shrink_k",
            "baseline",
            "excluded",
            "lift",
            "lb",
            "ub",
            "open_side",
            "role",
            "discovery",
            "family_axes",
            "family_q",
            "family_threshold",
        ]


class TestSegmentHeterogeneityCupedIndependenceUnaffected:
    def test_run_breakout_propagates_abs_diff_onto_breakout_estimate(self):
        """run_breakout must copy LiftEstimate.abs_diff/abs_se onto the BreakoutEstimate it constructs - these fields were once silently dropped."""
        rows = [
            centered_row_from_raw_sums(
                {
                    "experiment_id": "e1",
                    "metric": "rev",
                    "group_id": "control",
                    "country": "US",
                    "n": 500.0,
                    "sum_y": 5000.0,
                    "sum_y2": 52000.0,
                    "sum_x": None,
                    "sum_x2": None,
                    "sum_xy": None,
                    "sum_den": None,
                    "sum_den2": None,
                    "sum_yden": None,
                }
            ),
            centered_row_from_raw_sums(
                {
                    "experiment_id": "e1",
                    "metric": "rev",
                    "group_id": "treatment",
                    "country": "US",
                    "n": 500.0,
                    "sum_y": 5500.0,
                    "sum_y2": 62000.0,
                    "sum_x": None,
                    "sum_x2": None,
                    "sum_xy": None,
                    "sum_den": None,
                    "sum_den2": None,
                    "sum_yden": None,
                }
            ),
        ]
        result = run_breakout(
            rows, [_mean_metric("rev")], control_group="control", dimension="country"
        )
        assert len(result) == 1
        assert result[0].abs_diff is not None
        assert result[0].abs_se is not None
        assert result[0].abs_diff == pytest.approx(1.0, abs=0.01)


class TestMultiplicityAxesCompose:
    """Composition rule clause 2 - one procedure per cell - observed through
    behavior rather than asserted about the source."""

    def test_bonferroni_divides_by_segments_and_runs_no_family(self):
        """The segment axis alone: alpha/K on the interval, and no family
        record, because the BH family never ran."""
        estimates = _run_breakout_two_segment_fixture(correction="bonferroni", alpha=0.05)
        live = [r for r in estimates if r.excluded is None]
        assert live
        for r in live:
            # K=2 segments -> alpha/2 = 0.025 -> level 0.975
            lift = r.lift
            assert lift is not None
            assert lift.level == pytest.approx(0.975)
            assert r.family_axes is None
            assert r.family_q is None
            assert r.discovery is None

    def test_bh_runs_one_joint_family_and_does_not_divide_by_segments(self):
        """The pooled family instead: one procedure over metric x arm x
        segment, and NO segment division, so the two never stack on a cell."""
        estimates = _run_breakout_two_segment_fixture(correction="bh", alpha=0.05, q=0.10)
        live = [r for r in estimates if r.excluded is None]
        assert live
        for r in live:
            assert r.family_axes == ("metric", "arm", "segment")
            assert r.family_q == pytest.approx(0.10)
            assert r.discovery is not None
            # Never alpha/K: the segment divisor stays 1 on this branch, so a
            # non-selected row sits at the undivided nominal level.
            lift = r.lift
            assert lift is not None
            assert lift.level != pytest.approx(0.975)


class TestSegmentEstimateProvenanceMirrorsBreakoutEstimate:
    """`SegmentEstimate` carries `role`/`discovery`/`family_axes`/`family_q`/
    `family_threshold` verbatim from the source `BreakoutEstimate`, so a
    reader of the segment-level frame alone can tell a discovery from a
    non-discovery or a corrected interval from an uncorrected one -- not
    just the summary-level frame."""

    def test_segment_rows_mirror_source_role_discovery_family_fields(self):
        selected = _het_estimate("US", 0.10, math.sqrt(0.02)).model_copy(
            update={
                "role": "exploratory",
                "discovery": True,
                "family_axes": ("metric", "arm", "segment"),
                "family_q": 0.10,
                "family_threshold": 0.05,
            }
        )
        not_selected = _het_estimate("CA", 0.30, math.sqrt(0.05)).model_copy(
            update={
                "role": "exploratory",
                "discovery": False,
                "family_axes": ("metric", "arm", "segment"),
                "family_q": 0.10,
                "family_threshold": 0.05,
            }
        )
        uncorrected = _het_estimate("MX", -0.05, math.sqrt(0.01))
        estimates = BreakoutEstimates([selected, not_selected, uncorrected])
        _, segments = segment_heterogeneity(estimates, alpha=0.05)

        by_source = {"US": selected, "CA": not_selected, "MX": uncorrected}
        assert {s.dimension_value for s in segments} == set(by_source)
        for s in segments:
            source = by_source[s.dimension_value]
            assert s.role == source.role
            assert s.discovery == source.discovery
            assert s.family_axes == source.family_axes
            assert s.family_q == source.family_q
            assert s.family_threshold == source.family_threshold

    def test_raw_and_shrunken_rows_for_a_segment_carry_identical_provenance(self):
        """Both estimator rows for the same segment are two views of one
        BreakoutEstimate - provenance must not differ between them."""
        selected = _het_estimate("US", 0.10, math.sqrt(0.02), abs_diff=10.0, abs_se=2.0).model_copy(
            update={"discovery": True, "family_q": 0.10, "family_threshold": 0.05}
        )
        estimates = BreakoutEstimates(
            [
                selected,
                _het_estimate("CA", 0.30, math.sqrt(0.05), abs_diff=11.0, abs_se=2.0),
                _het_estimate("MX", -0.05, math.sqrt(0.01), abs_diff=9.0, abs_se=2.0),
            ]
        )
        _, segments = segment_heterogeneity(estimates, alpha=0.05)
        us_rows = [s for s in segments if s.dimension_value == "US"]
        assert len(us_rows) == 4  # 2 scales x 2 estimators
        for s in us_rows:
            assert s.discovery is True
            assert s.family_q == pytest.approx(0.10)
            assert s.family_threshold == pytest.approx(0.05)


class TestSegmentEstimateCodedValidation:
    def test_missing_lift_without_excluded_reason_raises_coded_error(self):
        """Direct construction (not via segment_heterogeneity) still unwraps
        pydantic's ValidationError into the coded refusal underneath."""
        with pytest.raises(InvalidRequestError) as exc_info:
            SegmentEstimate(
                metric="rev",
                method="unadjusted",
                method_role="decision",
                group_id="treatment",
                dimension="country",
                dimension_value="US",
                source=None,
                scale="relative",
                estimator="raw",
                shrink_k=None,
                baseline=None,
                excluded=None,
                lift=None,
            )
        assert exc_info.value.code == "breakout.segment.lift_none_excluded"

    def test_require_lift_on_valid_excluded_row_raises_availability_error(self):
        row = SegmentEstimate(
            metric="rev",
            method="unadjusted",
            method_role="decision",
            group_id="treatment",
            dimension="country",
            dimension_value="US",
            source=None,
            scale="relative",
            estimator="raw",
            shrink_k=None,
            baseline=None,
            excluded="few_units",
            lift=None,
        )

        with pytest.raises(InvalidRequestError) as exc_info:
            row.require_lift()

        assert exc_info.value.code == "breakout.segment.point_unavailable"


class TestSegmentContrastRefusesEveryMisrepresentedRow:
    """Recovering a standard error from lb/ub assumes a fixed-horizon Normal
    interval at the STORED alpha. A t-reference row's endpoints are t
    quantiles instead, so it is contrasted from the working-scale moments its
    interval was cut from and refused only when it carries none; a one-sided
    row carries the effective alpha needed to rebuild."""

    @staticmethod
    def _row(
        dimension_value: str,
        *,
        value: float = 1.0,
        se: float = 0.1,
        dof: float | None = None,
        alternative: str = "two-sided",
        alpha: float = 0.05,
    ) -> BreakoutEstimate:
        """A row whose stored interval is built from *se* at the EFFECTIVE alpha,
        exactly as infer_lift stores a one-sided result (symmetric two bounds,
        alpha=2*alpha)."""
        z = norm.isf(alpha / 2.0)
        return BreakoutEstimate(
            metric="rev",
            group_id="treatment",
            method="unadjusted",
            method_role="decision",
            dimension="country",
            dimension_value=dimension_value,
            estimand="late",
            value_scale="absolute",
            alternative=alternative,
            dof=dof,
            lift=Estimate(
                value=value,
                lb=value - z * se,
                ub=value + z * se,
                level=math.fsum((1.0, -alpha)),
                alpha=alpha,
            ),
        )

    def test_a_t_reference_row_without_its_moments_is_refused(self):
        """``_row`` stores endpoints but no log_mean/log_se, so a t reference
        leaves nothing to contrast from: back-solving its t endpoints with a
        Normal quantile would inflate the se by the t/z ratio."""
        rows = BreakoutEstimates([self._row("A", dof=12.0), self._row("B")])
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_contrast(rows, "A", "B")
        assert exc_info.value.code == "breakout.segment_contrast_reference_fixed_horizon"
        assert exc_info.value.context["dof"] == 12.0

    def test_a_welch_reference_row_is_contrasted_from_its_own_moments(self):
        """A Welch row that persisted its working-scale moments contrasts from
        those: the variance is log_se exactly, not the t/z-inflated width its
        stored endpoints would back-solve to."""
        rows = BreakoutEstimates(
            [
                _breakout_estimate(
                    "A",
                    0.10,
                    0.02,
                    dof=None,
                    reference_kind="t",
                    reference_df=5.5,
                ),
                _breakout_estimate("B", 0.05, 0.02),
            ]
        )
        result = segment_contrast(rows, "A", "B")

        combined_se = math.sqrt(0.02**2 + 0.02**2)
        assert result.value == pytest.approx(math.expm1(0.05), rel=1e-9)
        assert result.lb == pytest.approx(math.expm1(0.05 - _Z95 * combined_se), rel=1e-9)
        assert result.ub == pytest.approx(math.expm1(0.05 + _Z95 * combined_se), rel=1e-9)
        # The t_5.5 endpoints of row A are ~24% wider than its Normal ones, so
        # a back-solved se would have widened this contrast visibly.
        assert t_dist.isf(0.025, 5.5) / _Z95 > 1.2

    def test_a_one_sided_row_is_contrasted_from_its_stored_alpha(self):
        """The interval carries alpha_eff=0.10 (a one-sided 0.05 request), so the
        recovered standard error must come from THAT alpha. Asserting the bounds
        -- not just the point -- is what exercises it: reading the width at 0.05
        instead would inflate each se by z(0.025)/z(0.05) ~ 1.19."""
        rows = BreakoutEstimates(
            [
                self._row("A", value=1.0, se=0.1, alternative="greater", alpha=0.10),
                self._row("B", value=0.4, se=0.1, alternative="greater", alpha=0.10),
            ]
        )
        result = segment_contrast(rows, "A", "B", alpha=0.05)
        assert result.value == pytest.approx(0.6)
        # Contrast se = sqrt(0.1**2 + 0.1**2), interval at the requested 0.05.
        expected_half = _Z95 * math.sqrt(0.1**2 + 0.1**2)
        assert result.lb == pytest.approx(0.6 - expected_half, rel=1e-9)
        assert result.ub == pytest.approx(0.6 + expected_half, rel=1e-9)

    def test_two_plain_two_sided_rows_are_contrasted(self):
        rows = BreakoutEstimates([self._row("A"), self._row("B")])
        assert segment_contrast(rows, "A", "B").value == pytest.approx(0.0)


# segment_heterogeneity over a real run_breakout(correction="bh") family: a
# BH-selected segment must not abort heterogeneity for its whole family.


def _arm_row(n, mean, var, *, segment, group_id):
    from increment.estimation.armstats import centered_row_from_raw_sums

    sum_y = n * mean
    return centered_row_from_raw_sums(
        {
            "experiment_id": "exp1",
            "metric": "revenue",
            "group_id": group_id,
            "seg": segment,
            "n": float(n),
            "sum_y": sum_y,
            "sum_y2": var * (n - 1) + sum_y**2 / n,
            "sum_x": None,
            "sum_x2": None,
            "sum_xy": None,
            "sum_den": None,
            "sum_den2": None,
            "sum_yden": None,
        }
    )


def test_segment_heterogeneity_consumes_a_bh_selected_breakout():
    """Analysis.run_breakout() defaults to correction='bh', so a selected
    segment carries the FCR alpha while its siblings keep the nominal one.
    Q/tau^2/pooled read the alpha-invariant log moments, so heterogeneity
    runs over the whole family instead of refusing every alpha."""
    import polars as pl

    rows = []
    for segment, (control_mean, treatment_mean) in {
        "A": (10.0, 12.0),
        "B": (10.0, 10.05),
        "C": (10.0, 9.99),
    }.items():
        rows.append(_arm_row(4000, control_mean, 4.0, segment=segment, group_id="control"))
        rows.append(_arm_row(4000, treatment_mean, 4.0, segment=segment, group_id="treatment"))
    metrics = [MeanMetric(name="revenue", entity="user", fact="revenue", window_days=30)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimates = run_breakout(
            pl.DataFrame(rows),
            metrics,
            control_group="control",
            dimension="seg",
            correction="bh",
            q=0.10,
        )
    assert any(row.discovery for row in estimates), "fixture must produce a BH discovery"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary, segments = segment_heterogeneity(estimates)

    relative = [row for row in summary if row.scale == "relative"]
    assert len(relative) == 1
    assert relative[0].k == 3
    assert relative[0].p_value < 0.05
    assert {row.dimension_value for row in segments} == {"A", "B", "C"}


def test_unselected_bh_family_row_keeps_its_nominal_raw_rows_on_both_scales():
    """Only the SELECTED segment's cells are re-estimated at the FCR level;
    an unselected sibling in the same family carries the same family_q/
    family_axes bookkeeping but keeps its original nominal interval on
    both scales, and must not have its raw rows withheld too."""
    import polars as pl

    rows = []
    for segment, (control_mean, treatment_mean) in {
        "A": (10.0, 12.0),
        "B": (10.0, 10.05),
        "C": (10.0, 9.99),
    }.items():
        rows.append(_arm_row(4000, control_mean, 4.0, segment=segment, group_id="control"))
        rows.append(_arm_row(4000, treatment_mean, 4.0, segment=segment, group_id="treatment"))
    metrics = [MeanMetric(name="revenue", entity="user", fact="revenue", window_days=30)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimates = run_breakout(
            pl.DataFrame(rows),
            metrics,
            control_group="control",
            dimension="seg",
            correction="bh",
            q=0.10,
        )
    discovered = {row.dimension_value for row in estimates if row.discovery}
    undiscovered = {row.dimension_value for row in estimates if row.discovery is False}
    assert discovered, "fixture must produce a BH discovery"
    assert undiscovered, "fixture must also leave a sibling unselected"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, segments = segment_heterogeneity(estimates)

    for scale in ("relative", "absolute"):
        selected_raw = [
            s
            for s in segments
            if s.dimension_value in discovered and s.scale == scale and s.estimator == "raw"
        ]
        unselected_raw = [
            s
            for s in segments
            if s.dimension_value in undiscovered and s.scale == scale and s.estimator == "raw"
        ]
        assert selected_raw and all(
            s.lift is None and s.excluded == "reference_not_normal" for s in selected_raw
        )
        assert unselected_raw and all(
            s.lift is not None and s.excluded is None for s in unselected_raw
        )


def test_binomial_raw_rows_recompute_exact_interval_at_requested_alpha():
    def row(segment: str, group_id: str, successes: int) -> dict[str, Any]:
        return centered_row_from_raw_sums(
            {
                "experiment_id": "exp1",
                "metric": "conv",
                "group_id": group_id,
                "country": segment,
                "n": 100.0,
                "sum_y": float(successes),
                "sum_y2": float(successes),
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": None,
                "sum_den2": None,
                "sum_yden": None,
            }
        )

    estimates = run_breakout(
        [
            row("US", "control", 10),
            row("US", "treatment", 20),
            row("GB", "control", 15),
            row("GB", "treatment", 18),
        ],
        [ConversionMetric(name="conv", entity="user", fact="conv")],
        control_group="control",
        dimension="country",
        alpha=0.05,
    )
    original_bounds = {
        item.dimension_value: (item.require_lift().lb, item.require_lift().ub) for item in estimates
    }

    _, segments = segment_heterogeneity(estimates, alpha=0.10)

    raw = [item for item in segments if item.scale == "relative" and item.estimator == "raw"]
    assert len(raw) == 2
    assert all(item.lift is not None and item.excluded is None for item in raw)
    assert all(
        (item.require_lift().lb, item.require_lift().ub) != original_bounds[item.dimension_value]
        for item in raw
    )


def _binomial_breakout(
    *,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    correction: Literal["none", "bonferroni", "bh"] = "none",
):
    def row(segment: str, group_id: str, successes: int) -> dict[str, Any]:
        return centered_row_from_raw_sums(
            {
                "experiment_id": "exp1",
                "metric": "conv",
                "group_id": group_id,
                "country": segment,
                "n": 1000.0,
                "sum_y": float(successes),
                "sum_y2": float(successes),
                "sum_x": None,
                "sum_x2": None,
                "sum_xy": None,
                "sum_den": None,
                "sum_den2": None,
                "sum_yden": None,
            }
        )

    return run_breakout(
        [
            row("US", "control", 100),
            row("US", "treatment", 400),
            row("GB", "control", 100),
            row("GB", "treatment", 100),
        ],
        [ConversionMetric(name="conv", entity="user", fact="conv")],
        control_group="control",
        dimension="country",
        alpha=0.05,
        alternative=alternative,
        correction=correction,
        q=0.05,
    )


def test_directional_binomial_heterogeneity_rebuild_preserves_open_side():
    estimates = _binomial_breakout(alternative="greater")

    _, segments = segment_heterogeneity(estimates, alpha=0.20)

    raw = [item for item in segments if item.scale == "relative" and item.estimator == "raw"]
    assert raw
    assert all(item.require_lift().open_side == "upper" for item in raw)
    assert all(item.require_lift().ub is None for item in raw)


def test_bh_selected_binomial_heterogeneity_does_not_undo_fcr_interval():
    estimates = _binomial_breakout(correction="bh")
    selected = {row.dimension_value for row in estimates if row.discovery}
    assert selected == {"US"}

    _, segments = segment_heterogeneity(estimates, alpha=0.05)

    selected_raw = [
        item
        for item in segments
        if item.dimension_value in selected and item.scale == "relative" and item.estimator == "raw"
    ]
    assert selected_raw
    assert all(
        item.lift is None and item.excluded == "reference_not_normal" for item in selected_raw
    )


def test_segment_contrast_refuses_exact_binomial_reference():
    estimates = _binomial_breakout()

    with pytest.raises(InvalidRequestError) as exc_info:
        segment_contrast(estimates, "US", "GB")

    assert exc_info.value.code == "breakout.segment_contrast_reference_fixed_horizon"
