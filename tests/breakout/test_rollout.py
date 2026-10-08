"""Tests for `increment.breakout.rollout.segment_rollout_recommendation`."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from increment.breakout.estimates import (
    BreakoutEstimate,
    BreakoutEstimates,
    ExclusionReason,
    run_breakout,
)
from increment.breakout.rollout import (
    RolloutRecommendations,
    RolloutSegments,
    segment_rollout_recommendation,
)
from increment.errors import InvalidRequestError
from increment.estimation.armstats import ArmStats
from increment.estimation.results import Estimate
from increment.semantics.models import MeanMetric
from tests.oracles.test_meta_oracle import assert_rollout_matches, segment_posterior


def _mean_metric(name: str = "rev", rollout_cost: float | None = None) -> MeanMetric:
    """MeanMetric fixture; `rollout_cost` requires the explicit direction."""
    if rollout_cost is None:
        return MeanMetric(name=name, entity="user", fact=name)
    return MeanMetric(
        name=name,
        entity="user",
        fact=name,
        preferred_direction="increase",
        rollout_cost=rollout_cost,
    )


def _rollout_estimate(
    dimension_value: str,
    log_mean: float,
    log_se: float,
    *,
    metric: str = "rev",
    group_id: str = "treatment",
    excluded: ExclusionReason | None = None,
    level: float = 0.95,
) -> BreakoutEstimate:
    """Directly-constructed BreakoutEstimate with known log-scale moments -
    bypasses run_breakout so the selection and the guard statistic are hand-predictable, not 'whatever the fixture produced'."""
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
        metric=metric,
        group_id=group_id,
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=dimension_value,
        source=None,
        lift=lift,
        excluded=excluded,
    )


def _priced_key(metric: str = "rev") -> list[BreakoutEstimate]:
    """Three segments, two clearly above a 0 bar and one below, all precise
    enough that the corrected value stays positive."""
    return [
        _rollout_estimate("US", 0.20, 0.03, metric=metric),
        _rollout_estimate("CA", 0.12, 0.03, metric=metric),
        _rollout_estimate("MX", -0.04, 0.03, metric=metric),
    ]


def test_rollout_keeps_assigned_and_triggered_populations_separate():
    assigned = _priced_key()
    triggered = [row.model_copy(update={"analysis_population": "triggered"}) for row in assigned]
    estimates = BreakoutEstimates(assigned + triggered)

    recommendations, segments = segment_rollout_recommendation(estimates)

    assert len(recommendations) == 2
    assert len(segments) == 6
    assert {row.analysis_population for row in recommendations} == {"assigned", "triggered"}
    assert {row.analysis_population for row in segments} == {"assigned", "triggered"}
    for population in ("assigned", "triggered"):
        single_population, _ = segment_rollout_recommendation(
            BreakoutEstimates([row for row in estimates if row.analysis_population == population])
        )
        combined = next(row for row in recommendations if row.analysis_population == population)
        assert combined.policy_value == pytest.approx(single_population[0].policy_value)

    recommendation_frame = cast(pd.DataFrame, recommendations.to_frame())
    segment_frame = cast(pd.DataFrame, segments.to_frame())
    assert "analysis_population" in recommendation_frame
    assert "analysis_population" in segment_frame


def _late_estimate(
    dimension_value: str,
    value: float,
    se: float,
    *,
    metric: str = "rev",
    group_id: str = "treatment",
) -> BreakoutEstimate:
    """Encouragement LATE-shaped row: an additive Wald effect in `lift` itself,
    with no log-scale moments -- matching what estimate_encouragement emits."""
    z = norm.ppf(0.975)
    return BreakoutEstimate(
        metric=metric,
        group_id=group_id,
        method="unadjusted",
        method_role="decision",
        dimension="country",
        dimension_value=dimension_value,
        source=None,
        estimand="late",
        value_scale="absolute",
        lift=Estimate(value=value, lb=value - z * se, ub=value + z * se, level=0.95),
    )


class TestRealBreakoutRoundTrip:
    """The wiring run on real run_breakout output - every other test here
    hand-builds the log-scale moments this function reads."""

    @staticmethod
    def _fixture() -> BreakoutEstimates:
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
            group_row(arm("control", 4000, 10.0), "US"),
            group_row(arm("treatment", 4000, 11.0), "US"),
            group_row(arm("control", 4000, 20.0), "GB"),
            group_row(arm("treatment", 4000, 22.0), "GB"),
            group_row(arm("control", 4000, 15.0), "CA"),
            group_row(arm("treatment", 4000, 15.1), "CA"),
        ]
        return run_breakout(
            pd.DataFrame(rows),
            [_mean_metric()],
            control_group="control",
            dimension="country",
        )

    def test_round_trip_recommends_the_segments_that_clear_the_bar(self):
        estimates = self._fixture()
        recommendations, segments = segment_rollout_recommendation(estimates)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert (rec.metric, rec.dimension, rec.group_id) == ("rev", "country", "treatment")
        assert rec.k == 3
        assert rec.recommendation == "rollout"
        assert rec.policy_value is not None and rec.policy_value > 0.0
        assert rec.policy_value_raw is not None and rec.selection_bias is not None
        assert rec.policy_value == pytest.approx(rec.policy_value_raw - rec.selection_bias)

        # US (+10%) and GB (+10%) clear a zero bar; CA (~+0.7%) is the noisy one.
        selected = {s.dimension_value for s in segments if s.selected}
        assert {"US", "GB"} <= selected
        assert rec.n_selected == sum(1 for s in segments if s.selected)

    def test_round_trip_respects_a_declared_cost(self):
        estimates = self._fixture()
        recommendations, segments = segment_rollout_recommendation(
            estimates, metrics=[_mean_metric(rollout_cost=0.05)]
        )
        rec = recommendations[0]
        assert rec.rollout_cost == pytest.approx(0.05)
        # CA's ~+0.7% lift no longer clears a 5% bar.
        assert {s.dimension_value for s in segments if s.selected} == {"US", "GB"}


class TestCostResolution:
    def test_default_is_zero_when_nothing_declared(self):
        recommendations, _ = segment_rollout_recommendation(BreakoutEstimates(_priced_key()))
        assert recommendations[0].rollout_cost == 0.0

    def test_declared_metric_cost_is_used(self):
        recommendations, _ = segment_rollout_recommendation(
            BreakoutEstimates(_priced_key()), metrics=[_mean_metric(rollout_cost=0.02)]
        )
        assert recommendations[0].rollout_cost == pytest.approx(0.02)

    def test_explicit_scalar_beats_the_declaration(self):
        recommendations, _ = segment_rollout_recommendation(
            BreakoutEstimates(_priced_key()),
            metrics=[_mean_metric(rollout_cost=0.02)],
            rollout_cost=0.15,
        )
        assert recommendations[0].rollout_cost == pytest.approx(0.15)

    def test_explicit_mapping_beats_the_declaration_per_name(self):
        estimates = BreakoutEstimates(_priced_key() + _priced_key(metric="conv"))
        recommendations, _ = segment_rollout_recommendation(
            estimates,
            metrics=[_mean_metric(rollout_cost=0.02), _mean_metric("conv", rollout_cost=0.02)],
            rollout_cost={"rev": 0.15},
        )
        by_metric = {r.metric: r.rollout_cost for r in recommendations}
        assert by_metric == {"rev": pytest.approx(0.15), "conv": pytest.approx(0.02)}

    def test_cost_moves_the_selection(self):
        """The resolved cost is consumed as log1p, not raw: CA's +12% clears a
        10% bar and not a 15% one."""
        cheap, _ = segment_rollout_recommendation(
            BreakoutEstimates(_priced_key()), rollout_cost=0.10
        )
        dear, _ = segment_rollout_recommendation(
            BreakoutEstimates(_priced_key()), rollout_cost=0.15
        )
        assert cheap[0].n_selected == 2
        assert dear[0].n_selected == 1

    def test_unknown_mapping_key_is_refused_by_name(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout_recommendation(
                BreakoutEstimates(_priced_key()), rollout_cost={"revenue": 0.02}
            )
        assert exc_info.value.code == "breakout.rollout_cost_names"
        assert exc_info.value.context["unknown"] == ("revenue",)
        assert exc_info.value.context["known"] == ("rev",)

    def test_out_of_domain_cost_is_refused(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout_recommendation(
                BreakoutEstimates(_priced_key()), rollout_cost={"rev": -1.0}
            )
        assert exc_info.value.code == "breakout.rollout_cost_rollout"
        assert exc_info.value.context["name"] == "rev"
        assert exc_info.value.context["value"] == -1.0

    def test_explicit_cost_on_a_decrease_metric_is_refused(self):
        """The declared field refuses a non-increase metric outright, so an
        explicit override must not bypass that validator - pricing a decrease-preferred metric would keep the segments that moved worst."""
        latency = MeanMetric(name="rev", entity="user", fact="rev", preferred_direction="decrease")
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout_recommendation(
                BreakoutEstimates(_priced_key()),
                metrics=[latency],
                rollout_cost={"rev": 0.02},
            )
        assert exc_info.value.code == "breakout.price_rollout_metrics"
        assert exc_info.value.context["misdirected"] == ("rev",)

    def test_decrease_metric_is_refused_even_with_no_explicit_cost(self):
        """The harm is the selection rule, not the override channel: at the
        default 0.0 a decrease-preferred metric would still keep every segment whose lift exceeds zero - its worst-moving ones."""
        latency = MeanMetric(name="rev", entity="user", fact="rev", preferred_direction="decrease")
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout_recommendation(BreakoutEstimates(_priced_key()), metrics=[latency])
        assert exc_info.value.code == "breakout.price_rollout_metrics"
        assert exc_info.value.context["misdirected"] == ("rev",)

    def test_direction_is_unchecked_when_metrics_are_not_supplied(self):
        """Without metrics= there is no declaration to read - breakout rows
        carry no direction, so the call proceeds rather than guessing."""
        result, _ = segment_rollout_recommendation(BreakoutEstimates(_priced_key()))
        assert len(result) == 1


class TestExclusionAccounting:
    def test_k_disagreement_counts_ride_the_key_row(self):
        """Six declared segments, three usable: the value must never look like
        it priced the whole dimension."""
        estimates = BreakoutEstimates(
            [
                *_priced_key(),
                _rollout_estimate("FR", float("nan"), 0.03, excluded="few_units"),
                _rollout_estimate("DE", float("nan"), 0.03, excluded="nonpositive_mean"),
                # Live upstream, but a clamped se makes it unusable here.
                _rollout_estimate("IT", 0.05, 0.0),
            ]
        )
        recommendations, segments = segment_rollout_recommendation(estimates)
        rec = recommendations[0]
        assert rec.k == 3
        assert rec.n_excluded_design == 1
        assert rec.n_excluded_outcome == 2  # DE upstream + IT dropped here

        # Dense: every declared segment appears exactly once.
        assert len(segments) == 6
        by_value = {s.dimension_value: s for s in segments}
        assert by_value["FR"].selected is None
        assert by_value["FR"].excluded == "few_units"
        assert by_value["DE"].excluded == "nonpositive_mean"
        assert by_value["IT"].selected is None
        assert by_value["IT"].excluded == "zero_variance"
        assert by_value["US"].excluded is None
        assert by_value["US"].selected is True

    def test_key_with_fewer_than_two_usable_segments_is_skipped(self):
        estimates = BreakoutEstimates(
            [
                _rollout_estimate("US", 0.20, 0.03),
                _rollout_estimate("CA", float("nan"), 0.03, excluded="few_units"),
            ]
        )
        recommendations, segments = segment_rollout_recommendation(estimates)
        assert list(recommendations) == []
        assert list(segments) == []

    def test_grouping_keys_are_priced_independently(self):
        estimates = BreakoutEstimates(_priced_key() + _priced_key(metric="conv"))
        recommendations, segments = segment_rollout_recommendation(estimates)
        assert {r.metric for r in recommendations} == {"rev", "conv"}
        assert len(segments) == 6


class TestMixedEstimandGrouping:
    def test_late_rows_do_not_inflate_the_itt_family_exclusion_count(self):
        """ITT and LATE rows share metric/method/group_id but are different
        estimands. LATE rows have no log-scale moments, so counting them as
        this family's outcome exclusions would understate its coverage."""
        estimates = BreakoutEstimates(
            [
                *_priced_key(),
                _late_estimate("US", 1.2, 0.3),
                _late_estimate("CA", 1.9, 0.5),
                _late_estimate("MX", 0.4, 0.4),
            ]
        )
        recommendations, segments = segment_rollout_recommendation(estimates)

        assert len(recommendations) == 1
        rec = recommendations[0]
        assert (rec.estimand, rec.value_scale) == ("itt", "relative")
        assert rec.k == 3
        assert rec.n_excluded_outcome == 0
        assert rec.n_excluded_design == 0

        # The LATE family never reaches 2 usable segments, so it emits nothing
        # rather than zero_variance rows masquerading as ITT-family exclusions.
        assert len(segments) == 3
        assert all(s.estimand == "itt" and s.value_scale == "relative" for s in segments)
        assert all(s.excluded is None for s in segments)

    def test_absolute_scale_rows_are_not_priced_by_relative_rollout(self):
        """Absolute groups stay visible in the input but are not fed to the
        relative rollout estimator; a relative group in the same call remains
        supported."""
        absolute_itt = [
            _late_estimate("US", 1.2, 0.3),
            _late_estimate("CA", 1.9, 0.5),
        ]
        estimates = BreakoutEstimates(
            [*_priced_key(), *(r.model_copy(update={"estimand": "itt"}) for r in absolute_itt)]
        )
        recommendations, segments = segment_rollout_recommendation(estimates)

        assert len(recommendations) == 1
        assert {(r.estimand, r.value_scale) for r in recommendations} == {("itt", "relative")}
        assert {(s.estimand, s.value_scale) for s in segments} == {("itt", "relative")}
        baseline, _ = segment_rollout_recommendation(BreakoutEstimates(_priced_key()))
        assert recommendations[0].policy_value == pytest.approx(baseline[0].policy_value)
        assert recommendations[0].selection_bias_share == pytest.approx(
            baseline[0].selection_bias_share
        )

    def test_relative_estimand_families_keep_independent_prices_and_selections(self):
        compliance = [
            BreakoutEstimate.model_validate(
                {**_rollout_estimate(label, mean, 0.03).model_dump(), "estimand": "compliance"}
            )
            for label, mean in (("US", -0.08), ("CA", 0.40), ("MX", 0.10))
        ]
        families = {"itt": _priced_key(), "compliance": compliance}
        combined, segments = segment_rollout_recommendation(
            BreakoutEstimates([row for family in families.values() for row in family])
        )
        assert {row.estimand for row in combined} == set(families)
        assert {row.estimand for row in segments} == set(families)
        for estimand, family in families.items():
            separate, separate_segments = segment_rollout_recommendation(BreakoutEstimates(family))
            actual = next(row for row in combined if row.estimand == estimand)
            assert actual.policy_value == pytest.approx(separate[0].policy_value)
            assert actual.selection_bias_share == pytest.approx(separate[0].selection_bias_share)
            assert {
                row.dimension_value: row.selected for row in segments if row.estimand == estimand
            } == {row.dimension_value: row.selected for row in separate_segments}

    def test_frames_carry_the_identifying_columns(self):
        recommendations, segments = segment_rollout_recommendation(BreakoutEstimates(_priced_key()))
        rec_frame = cast(pd.DataFrame, recommendations.to_frame())
        seg_frame = cast(pd.DataFrame, segments.to_frame())
        assert {"estimand", "value_scale"} <= set(rec_frame.columns)
        assert {"estimand", "value_scale"} <= set(seg_frame.columns)
        assert set(rec_frame["estimand"]) == {"itt"}
        assert set(seg_frame["value_scale"]) == {"relative"}


class TestVerdicts:
    def test_all_noise_key_has_no_net_benefit(self):
        """Effects small relative to their SEs: the winner's curse eats the
        entire apparent lift, so no positive value can be claimed."""
        estimates = BreakoutEstimates(
            [
                _rollout_estimate("US", 0.01, 0.20),
                _rollout_estimate("CA", -0.02, 0.20),
                _rollout_estimate("MX", 0.015, 0.20),
                _rollout_estimate("BR", -0.01, 0.20),
            ]
        )
        recommendations, _ = segment_rollout_recommendation(estimates)
        rec = recommendations[0]
        assert rec.recommendation == "no_net_benefit"
        assert rec.policy_value is not None and rec.policy_value <= 0.0
        assert rec.selection_bias_share is not None and rec.selection_bias_share >= 1.0

    def test_refusal_propagates_from_the_offset_guard(self):
        """A 20% bar over segments whose lifts sit near 1%: the pooled mean is
        many typical SEs below the bar, so the estimator refuses to price it
        and this wiring reports that verdict rather than a number."""
        estimates = BreakoutEstimates(
            [
                _rollout_estimate("US", 0.010, 0.05),
                _rollout_estimate("CA", 0.020, 0.05),
                _rollout_estimate("MX", 0.015, 0.05),
            ]
        )
        recommendations, segments = segment_rollout_recommendation(estimates, rollout_cost=0.20)
        rec = recommendations[0]
        assert rec.recommendation == "refuse"
        assert rec.estimated_offset >= 2.0
        assert rec.policy_value is None
        assert rec.policy_value_raw is None
        assert rec.selection_bias is None
        assert rec.selection_bias_share is None
        # The selection is still emitted as evidence, just not deployable.
        assert all(s.selected is not None for s in segments)
        assert rec.n_selected == 0


class TestPosteriorPricing:
    """Each key is priced with the estimator's tau-marginalised posterior on
    its log-scale moments at the ``log1p(cost)`` threshold; the facade's
    value fields must match continuous quadrature of that model, including a
    key whose posterior mode lies beyond eight prior scales."""

    @pytest.mark.parametrize(
        ("log_means", "log_se", "cost", "tau_prior_scale"),
        [
            pytest.param((0.20, 0.12, -0.04), 0.03, 0.0, 0.30, id="ordinary_default_prior"),
            pytest.param(
                tuple(float(m) for m in np.linspace(-1.5, 1.5, 20)),
                0.05,
                0.05,
                0.05,
                id="mode_beyond_eight_prior_scales",
            ),
        ],
    )
    def test_value_fields_match_quadrature(self, log_means, log_se, cost, tau_prior_scale):
        estimates = BreakoutEstimates(
            [_rollout_estimate(f"S{i:02d}", m, log_se) for i, m in enumerate(log_means)]
        )
        recommendations, _ = segment_rollout_recommendation(
            estimates, rollout_cost=cost, tau_prior_scale=tau_prior_scale
        )
        (rec,) = recommendations
        est = np.array(log_means)
        var = np.full(est.size, log_se**2)
        threshold = math.log1p(cost)
        reference = segment_posterior(est, var, tau_prior_scale, cost_threshold=threshold)
        assert_rollout_matches(rec, reference, est=est, var=var, cost_threshold=threshold)
        assert rec.n_selected == int(np.count_nonzero(est > threshold))

    def test_unresolved_posterior_propagates_the_shared_numerical_refusal(self, monkeypatch):
        """No facade value can represent a posterior the node budget cannot
        resolve, so the estimator's coded refusal propagates rather than
        masquerading as the offset guard's "refuse" verdict. The budget is
        shrunk to reach that path deterministically."""
        monkeypatch.setattr("increment.estimation.meta._TAU_NODE_BUDGET", 1)
        with pytest.raises(InvalidRequestError) as exc_info:
            segment_rollout_recommendation(BreakoutEstimates(_priced_key()))
        assert exc_info.value.code == "estimation.meta.posterior_integration_unresolved"


class TestFrames:
    def test_to_frame_on_both_wrappers(self):
        recommendations, segments = segment_rollout_recommendation(
            BreakoutEstimates(
                [*_priced_key(), _rollout_estimate("FR", float("nan"), 0.03, excluded="few_units")]
            )
        )
        rec_frame = cast(pd.DataFrame, recommendations.to_frame())
        seg_frame = cast(pd.DataFrame, segments.to_frame())
        assert len(rec_frame) == 1
        assert len(seg_frame) == 4
        assert {"metric", "recommendation", "policy_value", "k", "rollout_cost"} <= set(
            rec_frame.columns
        )
        assert {"dimension_value", "selected", "excluded"} <= set(seg_frame.columns)

    def test_excluded_segment_is_null_not_false_in_the_frame(self):
        """An excluded segment was never evaluated - rendering its `selected`
        as False would claim it was considered and rejected."""
        _, segments = segment_rollout_recommendation(
            BreakoutEstimates(
                [*_priced_key(), _rollout_estimate("FR", float("nan"), 0.03, excluded="few_units")]
            )
        )
        frame = cast(pd.DataFrame, segments.to_frame())
        assert frame[frame["dimension_value"] == "FR"]["selected"].isna().all()
        assert frame["selected"].notna().sum() == 3

    def test_to_frame_on_empty_results(self):
        """The wrappers know their own schema with zero rows - an empty
        breakout is routine, not an edge case."""
        recommendations, segments = segment_rollout_recommendation(BreakoutEstimates([]))
        assert isinstance(recommendations, RolloutRecommendations)
        assert isinstance(segments, RolloutSegments)
        assert len(recommendations.to_frame()) == 0
        assert len(segments.to_frame()) == 0
