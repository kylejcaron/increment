"""Engine-level tests: control sign-flip, recovery, degenerates, method axis."""

import math
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pytest

from increment.errors import InvalidRequestError, UnsupportedRequestError
from increment.estimation.armstats import ArmStats
from increment.estimation.engine import (
    Method,
    _df_to_arms,
    estimate_lift,
)
from increment.estimation.variance import VARIANCE_MODELS, MeanVarianceModel
from increment.semantics.models import (
    ConversionMetric,
    MeanMetric,
    Measure,
    RatioMetric,
)


@pytest.mark.parametrize(
    "code,build",
    [
        (
            "estimation.engine.method_names_unique",
            lambda: estimate_lift(
                metrics=[_mean_metric()],
                summary=_summary_df(
                    [
                        _make_arm_stats(n=10, mean=1.0, var=1.0, group_id="control"),
                        _make_arm_stats(n=10, mean=1.0, var=1.0, group_id="treatment"),
                    ]
                ),
                control_group="control",
                methods=[Method(name="a"), Method(name="a")],
            ),
        ),
        (
            "estimation.engine.x_role_declared",
            lambda: estimate_lift(
                metrics=[_mean_metric()],
                summary=[
                    {
                        "experiment_id": "e",
                        "metric": "rev",
                        "group_id": "control",
                        "n": 100,
                        "ref_y": 1.1,
                        "cy1": 0.0,
                        "cy2": 19.0,
                        "ref_x": 5.0,
                        "cx1": 0.0,
                        "cx2": 7.0,
                        "cxy": 2.0,
                        "x_role": None,
                    }
                ],
                control_group="control",
            ),
        ),
    ],
)
def test_engine_refusal_carries_code(code, build):
    with pytest.raises(InvalidRequestError) as exc_info:
        build()
    assert exc_info.value.code == code


# Helpers


def _make_arm_stats(
    n: int,
    mean: float,
    var: float,
    study_id: str = "exp1",
    metric: str = "rev",
    group_id: str = "A",
) -> ArmStats:
    """Build an ArmStats from moments (n, mean, var)."""
    sum_y = float(n) * mean
    sum_y2 = var * (n - 1) + sum_y**2 / float(n)
    return ArmStats.from_raw_sums(
        study_id=study_id,
        metric=metric,
        group_id=group_id,
        n=n,
        sum_y=sum_y,
        sum_y2=sum_y2,
    )


def _summary_df(arms: list[ArmStats]) -> pd.DataFrame:
    """Convert list[ArmStats] to a group_summary DataFrame."""
    rows = []
    for a in arms:
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
                "cxden": a.cxden,
                "sum_d": a.sum_d,
                "cyd": a.cyd,
                "cy2d": a.cy2d,
            }
        )
    return pd.DataFrame(rows)


def _mean_metric(name: str = "rev", window_days: int | None = None) -> MeanMetric:
    """Build a MeanMetric fixture for tests using mean metrics."""
    return MeanMetric(name=name, entity="user", fact=name, window_days=window_days)


def _conversion_metric(name: str = "conv") -> ConversionMetric:
    """Build a ConversionMetric fixture for tests using conversion metrics."""
    return ConversionMetric(name=name, entity="user", fact=name)


def _ratio_metric(name: str = "ratio_rev", window_days: int | None = None) -> RatioMetric:
    """Build a RatioMetric fixture."""
    return RatioMetric(
        name=name,
        entity="user",
        numerator=Measure(fact="num", window_days=window_days),
        denominator=Measure(fact="den", window_days=window_days),
    )


def _ratio_arm(
    group_id: str,
    numerator: float,
    denominator: float = 1.0,
    *,
    covariate: bool = False,
) -> ArmStats:
    """Attach constant-denominator moments to the centered mean fixture."""
    return TestLargeOffsetLogRatio._arm(group_id, numerator, covariate=covariate).model_copy(
        update={
            "metric": "ratio_rev",
            "ref_den": denominator,
            "cden1": 0.0,
            "cden2": 0.0,
            "cyden": 0.0,
            "cxden": 0.0 if covariate else None,
        }
    )


class _StructuralMeanVarianceModel:
    consumes = "moments"

    def log_mean_se(self, arm: ArmStats) -> tuple[float, float]:
        se = 0.1 if arm.group_id == "control" else 0.2
        return math.log(arm.mean_y()), se


class _OverriddenMeanVarianceModel(_StructuralMeanVarianceModel, MeanVarianceModel):
    pass


@pytest.mark.parametrize("model_type", [_StructuralMeanVarianceModel, _OverriddenMeanVarianceModel])
def test_mean_family_registered_variance_model_owns_log_se(monkeypatch, model_type):
    """Structural replacements and subclass overrides retain dispatch."""
    model = model_type()
    monkeypatch.setitem(VARIANCE_MODELS._entries, "mean", model)
    control = _make_arm_stats(n=100, mean=10.0, var=1.0, group_id="control")
    treatment = _make_arm_stats(n=100, mean=11.0, var=1.0, group_id="treatment")

    [result] = estimate_lift(
        metrics=[_mean_metric()],
        summary=_summary_df([control, treatment]),
        control_group="control",
    ).results

    assert result.require_lift().log_se == pytest.approx(math.hypot(0.1, 0.2))
    assert result.require_lift().log_mean == pytest.approx(math.log(1.1))
    assert result.abs_diff == pytest.approx(1.0)


# Control-arm sign-flip


class TestControlArmSignFlip:
    def test_control_sign_flip_explicit(self):
        """estimate_lift with control='holdout' gives +δ for treatment > control,
        not -δ (which would happen if you naively sorted alphabetically)."""
        control = _make_arm_stats(n=10000, mean=10.0, var=4.0, group_id="holdout", metric="rev")
        treatment = _make_arm_stats(n=10000, mean=11.0, var=4.0, group_id="exposed", metric="rev")
        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()], summary=df, control_group="holdout"
        ).results

        assert len(results) == 1
        assert results[0].group_id == "exposed"
        assert results[0].require_lift().value > 0, (
            "Expected positive lift, got negative (sign flip)"
        )

    def test_control_arm_absent_raises(self):
        """control_group absent from the arms raises ValueError."""
        control = _make_arm_stats(n=100, mean=10.0, var=1.0, group_id="A")
        treatment = _make_arm_stats(n=100, mean=11.0, var=1.0, group_id="B")
        df = _summary_df([control, treatment])

        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(metrics=[_mean_metric()], summary=df, control_group="holdout")
        assert exc_info.value.code == "estimation.engine.control_group_found"

    def test_negative_lift(self):
        """When treatment < control, lift is negative."""
        control = _make_arm_stats(n=10000, mean=10.0, var=4.0, group_id="A", metric="rev")
        treatment = _make_arm_stats(n=10000, mean=9.0, var=4.0, group_id="B", metric="rev")
        df = _summary_df([control, treatment])
        results = estimate_lift(metrics=[_mean_metric()], summary=df, control_group="A").results
        assert len(results) == 1
        assert results[0].require_lift().value < 0


# Deterministic +40% recovery: _make_arm_stats builds exact sufficient
# statistics, so the closed-form posterior matches the delta-method CI up to
# float noise (~1e-7 at n=100k). No Monte-Carlo draws means no
# parameter_recovery marker.


class TestParameterRecovery:
    def test_positive_40_percent_lift_recovery(self):
        """Recover a +40% relative lift with known moments and flat prior."""
        true_relative_lift = 0.40
        true_log_rr = math.log(1.0 + true_relative_lift)

        control = _make_arm_stats(n=100000, mean=1.0, var=1.0, group_id="control", metric="rev")
        treatment = _make_arm_stats(n=100000, mean=1.4, var=1.0, group_id="treatment", metric="rev")
        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
        ).results

        assert len(results) == 1
        lift_est = results[0]

        # Check lift is on relative scale (exp(x)-1, not raw log); exact to
        # float precision since the moments are exact, not simulated.
        lift = lift_est.require_lift()
        assert lift.value == pytest.approx(true_relative_lift, abs=1e-9)

        # CI should cover the true relative lift
        assert lift.lb is not None
        assert lift.ub is not None
        assert lift.lb < true_relative_lift < lift.ub

        # value/lb/ub are back-transformed from the closed-form posterior
        # quantile (no sampling) - always > -1 on the relative scale.
        assert lift.value > -1.0
        assert lift.lb > -1.0

        # Flat-prior CI must match the closed-form delta-method CI to
        # near-float precision -- both sides compute the identical
        # Normal-Normal update from the same exact sufficient statistics.
        se_log_control = math.sqrt(1.0 / (100000 * 1.0**2))
        se_log_treatment = math.sqrt(1.0 / (100000 * 1.4**2))
        se_log_rr = math.sqrt(se_log_control**2 + se_log_treatment**2)
        z = 1.96
        delta_ci_log = (true_log_rr - z * se_log_rr, true_log_rr + z * se_log_rr)
        delta_ci_rel = (math.exp(delta_ci_log[0]) - 1, math.exp(delta_ci_log[1]) - 1)

        assert lift.lb == pytest.approx(delta_ci_rel[0], abs=1e-6)
        assert lift.ub == pytest.approx(delta_ci_rel[1], abs=1e-6)


def test_fast_fixed_input():
    """Fast unit test with fixed moments."""
    control = _make_arm_stats(n=5000, mean=2.0, var=1.0, group_id="C", metric="rev")
    treatment = _make_arm_stats(n=5000, mean=2.8, var=1.0, group_id="T", metric="rev")
    df = _summary_df([control, treatment])
    results = estimate_lift(metrics=[_mean_metric()], summary=df, control_group="C").results
    assert len(results) == 1
    assert results[0].require_lift().value == pytest.approx(0.40, abs=0.025)


def test_estimate_lift_carries_winsorization_diagnostics():
    """Legacy percentile diagnostics cannot turn clipped moments into raw state."""
    from increment.errors import CodedError

    control = _make_arm_stats(n=10, mean=5.0, var=1.0, group_id="C", metric="rev")
    treatment = _make_arm_stats(n=12, mean=6.0, var=1.0, group_id="T", metric="rev")
    df = _summary_df([control, treatment])
    df["winsor_upper_percentile"] = 0.99
    df["winsor_upper_bound"] = 100.0
    df["winsor_n"] = [10, 12]
    df["winsor_n_lower"] = 0
    df["winsor_n_upper"] = [1, 2]

    with pytest.raises(CodedError) as error:
        estimate_lift(metrics=[_mean_metric()], summary=df, control_group="C")
    assert error.value.code == "estimation.winsor.raw_state_required"


# Degenerate data


class TestDegenerateData:
    def test_conversion_both_arms_all_success_now_estimates_via_binomial(self):
        """A genuinely binary (conversion) metric with both arms all-success
        no longer hits the log-scale zero-variance guard: the exact binomial
        risk-ratio method admits it directly (an informative, narrow, finite
        confidence set -- see binomial_rr.py's zero-cell geometry), unlike
        the retired log-Normal delta method this used to fall back to.
        """
        n = 100
        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="conv",
            group_id="control",
            n=n,
            sum_y=float(n),
            sum_y2=float(n),
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="conv",
            group_id="treatment",
            n=n,
            sum_y=float(n),
            sum_y2=float(n),
        )
        df = _summary_df([control, treatment])

        computation = estimate_lift(
            metrics=[_conversion_metric()], summary=df, control_group="control"
        )
        assert computation.failures == {}
        assert len(computation.results) == 1
        result = computation.results[0]
        assert result.reference_kind == "binomial"
        assert result.lift is not None
        lift = result.require_lift()
        assert lift.value == 0.0
        assert lift.lb is not None and lift.ub is not None
        assert lift.lb < 0.0 < lift.ub

    def test_mean_metric_zero_variance_arm_still_emits_keyed_failure(self):
        """A genuinely real-valued (non-binary) mean metric with an
        all-identical-outcome arm still hits the log-scale zero-variance
        guard: that boundary is retired ONLY for the exact binomial
        method's eligible (conversion/retention) metrics, never for an
        ordinary continuous mean metric.
        """
        n = 100
        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="control",
            n=n,
            sum_y=float(5 * n),
            sum_y2=float(25 * n),
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="treatment",
            n=n,
            sum_y=float(5 * n),
            sum_y2=float(25 * n),
        )
        df = _summary_df([control, treatment])

        computation = estimate_lift(metrics=[_mean_metric()], summary=df, control_group="control")
        assert computation.results == ()
        assert len(computation.failures) == 1
        assert next(iter(computation.failures)).group_id == "treatment"

    def test_zero_mean_control_produces_an_additive_only_row(self):
        """Zero-mean control no longer aborts estimate_lift(): log(0) is
        undefined for the relative lift, but the additive difference from
        the same arm summaries is well-defined -- see
        TestNonPositiveArmMeanFailsOneCell for the dedicated coverage of
        this row shape."""
        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="rev",
            group_id="control",
            n=100,
            sum_y=0.0,
            sum_y2=0.0,
        )
        treatment = _make_arm_stats(n=100, mean=5.0, var=4.0, group_id="treatment", metric="rev")
        df = _summary_df([control, treatment])

        computation = estimate_lift(metrics=[_mean_metric()], summary=df, control_group="control")
        assert computation.failures == {}
        (row,) = computation.results
        assert row.lift is None
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert row.abs_diff == pytest.approx(5.0)


class TestNonPositiveArmMeanFailsOneCell:
    """A zero/negative arm mean on the unclustered mean/ratio path
    produces a real row (lift=None, relative_unavailable_reason, and a
    well-defined additive result) instead of aborting estimate_lift() --
    the same additive-visibility contract the clustered path already
    gives an unavailable relative lift."""

    def test_zero_treatment_mean_yields_an_additive_only_row(self):
        control = _make_arm_stats(200, mean=1.0, var=0.04, metric="refunds", group_id="control")
        treatment = _make_arm_stats(200, mean=0.0, var=0.0, metric="refunds", group_id="treatment")
        df = _summary_df([control, treatment])
        metric = _mean_metric("refunds")
        computation = estimate_lift([metric], df, control_group="control")
        assert computation.failures == {}
        (row,) = computation.results
        assert row.lift is None
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert row.abs_diff == pytest.approx(-1.0)
        assert row.abs_se == pytest.approx(math.hypot(0.0, math.sqrt(0.04 / 200)))
        assert row.abs_lb is not None and row.abs_ub is not None
        assert row.stat_sig() is False
        assert row.p_value() == 1.0

    def test_negative_control_mean_on_ratio_metric_yields_an_additive_only_row(self):
        """The unclustered ratio log route also raises check_positive_mean's
        coded error (not a bare math-domain ValueError); this must ALSO
        become an additive-only row, with its sidecar taken from
        strategy.absolute_moments."""
        control = ArmStats.from_raw_sums(
            study_id="e",
            metric="net_revenue",
            group_id="control",
            n=100,
            sum_y=-300.0,
            sum_y2=1000.0,
            sum_den=100.0,
            sum_den2=120.0,
            sum_yden=-50.0,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="e",
            metric="net_revenue",
            group_id="treatment",
            n=100,
            sum_y=200.0,
            sum_y2=900.0,
            sum_den=100.0,
            sum_den2=120.0,
            sum_yden=40.0,
        )
        df = _summary_df([control, treatment])
        metric = _ratio_metric("net_revenue")
        computation = estimate_lift([metric], df, control_group="control")
        assert computation.failures == {}
        (row,) = computation.results
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert row.abs_diff is not None
        assert row.abs_se is not None

    def test_other_metrics_in_the_same_run_survive_a_refused_metric(self):
        """Repro: a refunds metric with an all-zero treatment
        arm must not erase revenue's and converted's rows -- and its
        own row now carries a real additive result instead of vanishing."""
        revenue = (
            _make_arm_stats(200, mean=10.0, var=4.0, metric="revenue", group_id="control"),
            _make_arm_stats(200, mean=11.0, var=4.0, metric="revenue", group_id="treatment"),
        )
        refunds = (
            _make_arm_stats(200, mean=1.0, var=0.04, metric="refunds", group_id="control"),
            _make_arm_stats(200, mean=0.0, var=0.0, metric="refunds", group_id="treatment"),
        )
        converted = (
            ArmStats.from_raw_sums(
                study_id="e",
                metric="converted",
                group_id="control",
                n=200,
                sum_y=60.0,
                sum_y2=60.0,
            ),
            ArmStats.from_raw_sums(
                study_id="e",
                metric="converted",
                group_id="treatment",
                n=200,
                sum_y=70.0,
                sum_y2=70.0,
            ),
        )
        df = _summary_df([*revenue, *refunds, *converted])
        metrics = [
            _mean_metric("revenue"),
            _mean_metric("refunds"),
            _conversion_metric("converted"),
        ]
        computation = estimate_lift(metrics, df, control_group="control")
        assert {r.metric for r in computation.results} == {"revenue", "refunds", "converted"}
        refunds_row = next(r for r in computation.results if r.metric == "refunds")
        assert refunds_row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert refunds_row.abs_diff is not None

    def test_sensitivity_role_with_prior_still_yields_an_additive_only_row(self):
        """A sensitivity-role method never drives family selection, so
        unlike a decision-role cell under a prior (which falls back to
        the old DecisionFailure funnel, since a prior-shrunk relative
        posterior needs a log-scale point that doesn't exist here), a
        sensitivity-role cell's additive difference -- always prior-free
        regardless of role -- must still surface as a real row instead
        of being silently dropped."""
        from increment.estimation.inference import Normal

        control = _make_arm_stats(200, mean=1.0, var=0.04, metric="refunds", group_id="control")
        treatment = _make_arm_stats(200, mean=0.0, var=0.0, metric="refunds", group_id="treatment")
        df = _summary_df([control, treatment])
        metric = _mean_metric("refunds")
        computation = estimate_lift(
            [metric],
            df,
            control_group="control",
            methods=[Method(name="sens")],
            method_roles={"sens": "sensitivity"},
            prior=Normal(mu=0.0, sigma=0.05),
        )
        assert computation.failures == {}
        (row,) = computation.results
        assert row.method_role == "sensitivity"
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert row.abs_diff == pytest.approx(-1.0)

    def test_additive_interval_reference_is_invariant_to_translating_outcomes(self):
        """Shifting every outcome by a constant moves both arm means below
        zero but leaves the additive difference and its uncertainty
        unchanged, so the additive interval keeps the ordinary path's Welch
        reference: t with 8 df for n=5 per arm at equal variance."""
        from scipy.stats import t

        control_y = [0.5, 0.5, 1.0, 1.5, 1.5]
        arm_y = {"control": control_y, "treatment": [y + 0.5 for y in control_y]}

        def additive_row(shift: float):
            arms = [
                ArmStats.from_raw_sums(
                    study_id="exp1",
                    metric="rev",
                    group_id=group_id,
                    n=len(ys),
                    sum_y=math.fsum(y + shift for y in ys),
                    sum_y2=math.fsum((y + shift) ** 2 for y in ys),
                )
                for group_id, ys in arm_y.items()
            ]
            computation = estimate_lift(
                [_mean_metric()], _summary_df(arms), control_group="control"
            )
            assert computation.failures == {}
            (row,) = computation.results
            return row

        ordinary, shifted = additive_row(0.0), additive_row(-2.0)
        assert ordinary.lift is not None
        assert shifted.relative_unavailable_reason == "nonpositive_arm_mean"
        half_width = t.isf(0.025, 8) * math.sqrt(0.1)
        for row in (ordinary, shifted):
            assert row.abs_diff == pytest.approx(0.5)
            assert row.abs_se == pytest.approx(math.sqrt(0.1))
            assert row.abs_reference_kind == "t"
            assert row.abs_reference_df == pytest.approx(8.0)
            assert row.abs_lb == pytest.approx(0.5 - half_width)
            assert row.abs_ub == pytest.approx(0.5 + half_width)


# Method axis


class TestMethodAxis:
    def test_multiple_methods(self):
        """Multiple Method entries return one LiftEstimate per (method x arm)."""
        control = _make_arm_stats(n=1000, mean=10.0, var=4.0, group_id="A", metric="rev")
        treatment = _make_arm_stats(n=1000, mean=11.0, var=4.0, group_id="B", metric="rev")
        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="A",
            methods=[Method(name="unadjusted"), Method(name="m2")],
        ).results

        assert len(results) == 2
        methods_found = {r.method for r in results}
        assert methods_found == {"unadjusted", "m2"}
        for r in results:
            assert r.group_id == "B"

    def test_estimate_lift_prefers_unadjusted_when_no_method_roles_given(self):
        """The plain (non-inverted) method-role preference: with no explicit
        method_roles, 'unadjusted' takes the decision role over any other
        named method, mirroring resolve_method_roles's default branch."""
        control = _make_arm_stats(n=1000, mean=10.0, var=4.0, group_id="A", metric="rev")
        treatment = _make_arm_stats(n=1000, mean=11.0, var=4.0, group_id="B", metric="rev")
        df = _summary_df([control, treatment])

        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="A",
            methods=[Method(name="m2"), Method(name="unadjusted")],
        ).results

        roles = {r.method: r.method_role for r in results}
        assert roles["unadjusted"] == "decision"
        assert roles["m2"] == "sensitivity"

    def test_duplicate_method_names_refuse_before_summary_consumption(self):
        class ExplodingSummary:
            def __iter__(self):
                raise AssertionError("summary was consumed before validation")

        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                metrics=[_mean_metric()],
                summary=ExplodingSummary(),
                control_group="A",
                methods=[
                    Method(name="same"),
                    Method(name="same", variance_reduction="cuped"),
                ],
            )
        assert exc_info.value.code == "estimation.engine.method_names_unique"

    def test_unregistered_variance_reduction_raises(self):
        """An unregistered variance_reduction is refused by its code, naming
        the key that was not registered."""
        control = _make_arm_stats(n=100, mean=10.0, var=1.0, group_id="A", metric="rev")
        treatment = _make_arm_stats(n=100, mean=11.0, var=1.0, group_id="B", metric="rev")
        df = _summary_df([control, treatment])

        with pytest.raises(UnsupportedRequestError) as exc:
            estimate_lift(
                metrics=[_mean_metric()],
                summary=df,
                control_group="A",
                methods=[Method(name="bogus", variance_reduction="bogus_vr")],
            )
        assert exc.value.code == "estimation.variance.registry.no_registered_available"
        assert exc.value.context["key"] == "bogus_vr"


# Regression: ratio metric dispatch from declared Metric type


class TestRatioMetricDispatch:
    def test_ratio_metric_with_none_den_moments(self):
        """RatioMetric with None den moments dispatches to RatioVarianceModel,
        which requires denominator moments and raises ValueError.

        The old _infer_metric_type heuristic returned "mean" when sum_den
        was None, silently producing wrong variance estimates.
        """
        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=10000,
            sum_y=10000.0,
            sum_y2=1005000.0,
            sum_den=None,
            sum_den2=None,
            sum_yden=None,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="treatment",
            n=10000,
            sum_y=14000.0,
            sum_y2=1970000.0,
            sum_den=None,
            sum_den2=None,
            sum_yden=None,
        )
        df = _summary_df([control, treatment])
        metrics = [_ratio_metric()]

        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                metrics=metrics,
                summary=df,
                control_group="control",
            )
        assert exc_info.value.code == "estimation.variance.ratio_moments_needs_ref_den"

    def test_ratio_metric_full_moments_produces_lift_estimate(self):
        """RatioMetric with fully populated den moments dispatches to
        RatioVarianceModel and produces a valid LiftEstimate end-to-end
        through estimate_lift - the positive-path counterpart to the
        None-den-moments error test above.
        """
        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=100,
            sum_y=1000.0,
            sum_y2=10100.0,
            sum_den=500.0,
            sum_den2=2600.0,
            sum_yden=5030.0,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="treatment",
            n=100,
            sum_y=1200.0,
            sum_y2=14500.0,
            sum_den=500.0,
            sum_den2=2600.0,
            sum_yden=6030.0,
        )
        df = _summary_df([control, treatment])
        metrics = [_ratio_metric()]

        results = estimate_lift(
            metrics=metrics,
            summary=df,
            control_group="control",
        ).results

        assert len(results) == 1
        result = results[0]
        assert result.metric == "ratio_rev"
        assert result.group_id == "treatment"
        assert math.isfinite(result.require_lift().value)
        # control ratio = 1000/500 = 2.0; treatment ratio = 1200/500 = 2.4 -> +20% lift
        assert result.require_lift().value > 0, "expected positive lift for higher treatment ratio"


# Input format equivalence: DataFrame / pyarrow.Table / list[dict] must agree


def _summary_rows(arms: list[ArmStats]) -> list[dict]:
    """Convert list[ArmStats] to group_summary row dicts (same shape as _summary_df,
    kept separate so it can also feed pyarrow.Table.from_pylist / plain dicts)."""
    rows = []
    for a in arms:
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
                "ref_den": a.ref_den,
                "cden1": a.cden1,
                "cden2": a.cden2,
                "cyden": a.cyden,
            }
        )
    return rows


class TestInputFormats:
    def test_pandas_pyarrow_and_dict_rows_agree(self):
        """estimate_lift gives numerically identical results for a pandas
        DataFrame, a pyarrow Table, and a plain list[dict] over the same
        rows - the DataFrame path is no longer special-cased, so narwhals
        conversion and plain-mapping iteration must produce the exact same
        ArmStats (and therefore the exact same closed-form lift, which has
        nothing left to vary since infer_lift no longer samples).
        """
        control = _make_arm_stats(n=10000, mean=10.0, var=4.0, group_id="A", metric="rev")
        treatment = _make_arm_stats(n=10000, mean=11.0, var=4.0, group_id="B", metric="rev")
        rows = _summary_rows([control, treatment])

        pandas_result = estimate_lift(
            metrics=[_mean_metric()], summary=pd.DataFrame(rows), control_group="A"
        ).results
        pyarrow_result = estimate_lift(
            metrics=[_mean_metric()], summary=pa.Table.from_pylist(rows), control_group="A"
        ).results
        dict_result = estimate_lift(
            metrics=[_mean_metric()], summary=rows, control_group="A"
        ).results

        assert len(pandas_result) == len(pyarrow_result) == len(dict_result) == 1
        for other in (pyarrow_result, dict_result):
            assert pandas_result[0].require_lift().value == other[0].require_lift().value
            assert pandas_result[0].require_lift().lb == other[0].require_lift().lb
            assert pandas_result[0].require_lift().ub == other[0].require_lift().ub

    def test_dict_rows_missing_column_raises(self):
        """A plain-mapping row missing a required key gets the SAME named
        "missing required columns" ValueError the DataFrame path raises -
        one defect, one error quality, regardless of input format."""
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                metrics=[_mean_metric()],
                summary=[{"experiment_id": "e", "metric": "rev", "group_id": "A"}],
                control_group="A",
            )
        assert exc_info.value.code == "estimation.engine.group_summary_row"

    def test_nan_required_moment_refused_by_name(self):
        """A NaN ref_y used to surface as a raw pydantic ValidationError
        deep inside inference.Normal ('sigma Input should be greater than
        0'), with no metric or arm named. Refuse at the conversion edge."""
        rows = [
            {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": "control",
                "n": 100,
                "ref_y": float("nan"),
                "cy1": 0.0,
                "cy2": 5.0,
            },
            {
                "experiment_id": "e",
                "metric": "rev",
                "group_id": "T",
                "n": 100,
                "ref_y": 1.2,
                "cy1": 0.0,
                "cy2": 6.0,
            },
        ]
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(metrics=[_mean_metric()], summary=rows, control_group="control")
        assert exc_info.value.code == "estimation.armstats.arm_stats.group_summary_moment"
        assert exc_info.value.context["name"] == "ref_y"
        assert exc_info.value.context["metric"] == "rev"
        assert exc_info.value.context["group_id"] == "control"

    def test_nan_optional_moment_treated_as_absent(self):
        """A NaN OPTIONAL moment must behave exactly like None: pandas has
        no float NULL, so a pandas-substrate frame delivers an
        unmaterialised covariate as NaN - refusing it would break every
        pandas caller with no CUPED covariate (observed live on the
        simulate pipeline). Required moments are the corruption boundary
        instead (see test_nan_required_moment_refused_by_name)."""
        base = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
        }
        treated = {"group_id": "T", "ref_y": 1.2, "cy1": 0.0, "cy2": 6.0}
        rows_nan = [
            {**base, "ref_x": float("nan")},
            {**base, **treated, "ref_x": float("nan")},
        ]
        rows_none = [
            {**base, "ref_x": None},
            {**base, **treated, "ref_x": None},
        ]
        a = estimate_lift(
            metrics=[_mean_metric()], summary=rows_nan, control_group="control"
        ).results
        b = estimate_lift(
            metrics=[_mean_metric()], summary=rows_none, control_group="control"
        ).results
        assert a[0].require_lift().value == b[0].require_lift().value
        assert a[0].require_lift().lb == b[0].require_lift().lb

    def test_partial_moment_family_refused_by_df_to_arms(self):
        """A row that MATERIALISES some but not all of a declared family has
        silently dropped declared columns a downstream reduction needs, and
        cannot be reinterpreted as a shorter valid row - so _df_to_arms
        refuses it. group_summary always emits each family whole or entirely
        NULL, so this only ever fires on a table that lost columns."""

        base = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
        }
        # A real covariate mean, but the rest of the family dropped.
        with pytest.raises(InvalidRequestError) as exc_info:
            _df_to_arms([{**base, "ref_x": 5.0, "cx1": 0.0}])
        assert exc_info.value.code == "estimation.armstats.arm_stats.partial_family_missing"
        assert exc_info.value.context["label"] == "covariate"
        # A real denominator family missing its cross-moment.
        with pytest.raises(InvalidRequestError) as exc_info:
            _df_to_arms([{**base, "ref_den": 2.0, "cden1": 0.0, "cden2": 3.0}])
        assert exc_info.value.code == "estimation.armstats.arm_stats.partial_family_missing"
        assert exc_info.value.context["label"] == "denominator"
        # A real uptake family missing cy2d.
        with pytest.raises(InvalidRequestError) as exc_info:
            _df_to_arms([{**base, "sum_d": 4.0, "cyd": 1.0}])
        assert exc_info.value.code == "estimation.armstats.arm_stats.partial_family_missing"
        assert exc_info.value.context["label"] == "uptake"
        # A fully-absent family (ref_x null, no others) is the legitimate
        # mean-only case - never refused.
        arms = _df_to_arms([{**base, "ref_x": None}])
        assert arms[0].ref_x is None

    def test_covariate_family_without_x_role_column_defaults_to_covariate(self):
        """A summary frame predating the x_role declaration carries the full
        covariate family but no x_role column. _df_to_arms defaults x_role to
        "covariate" so the invariant (x_role set iff ref_x set) holds at the
        read boundary; otherwise pooling reads x_role=None, drops the whole
        covariate family, and mis-reports it as un-materialised."""

        base = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
        }
        # Full covariate family, no x_role key at all.
        (arm,) = _df_to_arms([{**base, "ref_x": 5.0, "cx1": 0.0, "cx2": 7.0, "cxy": 2.0}])
        assert arm.ref_x == 5.0
        assert arm.x_role == "covariate"
        # A mean-only row (no covariate) still leaves x_role None.
        (mean_only,) = _df_to_arms([{**base}])
        assert mean_only.ref_x is None and mean_only.x_role is None

    def test_numeric_x_role_is_not_inferred_for_complete_x_family(self):
        """A malformed present role must reach ArmStats rather than being
        replaced by the legacy covariate inference."""

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
            "ref_x": 5.0,
            "cx1": 0.0,
            "cx2": 7.0,
            "cxy": 2.0,
            "x_role": 1.0,
        }
        with pytest.raises(InvalidRequestError) as raised:
            _df_to_arms([row])
        assert raised.value.code == "model.field.type"

    def test_dataframe_null_x_role_is_absent_compatible(self):
        """A pandas null in the optional string column is an absent
        declaration, not an invalid Pydantic string, when no x family exists."""
        import pandas as pd

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
        }
        treated = {"group_id": "T", "ref_y": 1.2, "cy1": 0.0, "cy2": 6.0}
        absent = estimate_lift(
            metrics=[_mean_metric()],
            summary=pd.DataFrame([row, {**row, **treated}]),
            control_group="control",
        ).results
        # A declared role beside no x family would be refused outright, so a
        # produced estimate is itself evidence the null read as absent.
        assert absent[0].require_lift().value > 0

        for sentinel in (float("nan"), pd.NA):
            nulled = estimate_lift(
                metrics=[_mean_metric()],
                summary=pd.DataFrame(
                    [{**row, "x_role": sentinel}, {**row, **treated, "x_role": sentinel}]
                ),
                control_group="control",
            ).results
            assert nulled[0].require_lift().value == absent[0].require_lift().value
            assert nulled[0].require_lift().lb == absent[0].require_lift().lb

    def test_dataframe_null_x_role_refuses_materialized_x_family(self):
        """A null cell is acceptable only when the x family is absent."""
        import pandas as pd

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
            "ref_x": 5.0,
            "cx1": 0.0,
            "cx2": 7.0,
            "cxy": 2.0,
            "x_role": pd.NA,
        }
        with pytest.raises(InvalidRequestError) as exc_info:
            _df_to_arms(pd.DataFrame([row]))
        assert exc_info.value.code == "estimation.engine.x_role_declared"

    def test_explicit_null_x_role_refuses_ambiguous_cluster_family(self):
        """A clustered cxden row with an explicit null role is malformed:
        only a row that predates the x_role key may use None for this
        otherwise ambiguous x family."""

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
            "ref_x": 5.0,
            "cx1": 0.0,
            "cx2": 7.0,
            "cxy": 2.0,
            "x_role": None,
            "ref_den": 2.0,
            "cden1": 0.0,
            "cden2": 3.0,
            "cyden": 0.0,
            "cxden": 1.0,
        }
        with pytest.raises(InvalidRequestError) as exc_info:
            _df_to_arms([row])
        assert exc_info.value.code == "estimation.engine.x_role_declared"

        legacy = {key: value for key, value in row.items() if key != "x_role"}
        (arm,) = _df_to_arms([legacy])
        assert arm.x_role is None

    def test_explicit_x_role_without_x_family_is_not_inferred_away(self):
        """A present role beside absent x must reach ArmStats unchanged."""

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 100,
            "ref_y": 1.1,
            "cy1": 0.0,
            "cy2": 19.0,
            "x_role": "covariate",
        }
        with pytest.raises(ValueError, match="x_role requires the x family"):
            _df_to_arms([row])

    def test_cuped_honors_covariate_on_a_frame_without_x_role_column(self):
        """End to end: a legacy summary frame carrying the covariate family
        but no x_role column still adjusts under CUPED. The x_role default at
        the read boundary keeps pooling (combine -> pooled_theta) from
        dropping the covariate and raising a false 'not materialised' error."""
        control = ArmStats(
            study_id="e",
            metric="rev",
            group_id="control",
            n=100,
            ref_y=1.0,
            cy1=0.0,
            cy2=50.0,
            ref_x=2.0,
            cx1=0.0,
            cx2=40.0,
            cxy=20.0,
            x_role="covariate",
        )
        treatment = ArmStats(
            study_id="e",
            metric="rev",
            group_id="treatment",
            n=100,
            ref_y=1.1,
            cy1=0.0,
            cy2=52.0,
            ref_x=2.0,
            cx1=0.0,
            cx2=42.0,
            cxy=21.0,
            x_role="covariate",
        )
        df = _summary_df([control, treatment]).drop(columns=["x_role"])
        assert "x_role" not in df.columns
        results = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
            methods=[Method(name="unadjusted"), Method(name="cuped", variance_reduction="cuped")],
        ).results
        assert {"unadjusted", "cuped"} <= {r.method for r in results}

    def test_legacy_clustered_frame_without_x_role_is_not_labeled_covariate(self):
        """A clustered summary predating the x_role column carries the uptake
        total in the covariate family AND cxden. Its role cannot be inferred
        from shape, so _df_to_arms leaves x_role None (not a fabricated
        "covariate") and the cluster LATE reduction refuses it accurately."""
        from increment.estimation.variance import cluster_uptake_moments

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "t",
            "n": 20,
            "ref_y": 1.0,
            "cy1": 0.0,
            "cy2": 10.0,
            "ref_x": 3.0,
            "cx1": 0.0,
            "cx2": 8.0,
            "cxy": 2.0,
            "ref_den": 5.0,
            "cden1": 0.0,
            "cden2": 6.0,
            "cyden": 1.0,
            "cxden": 1.5,  # warehouse clustered marker: only a clustered summary row carries it
        }
        (arm,) = _df_to_arms([row])
        assert arm.ref_x == 3.0
        assert arm.x_role is None  # unknowable from shape - not fabricated
        with pytest.raises(InvalidRequestError) as exc_info:
            cluster_uptake_moments(arm)
        assert exc_info.value.code == "estimation.variance.cluster_robust_late"
        assert exc_info.value.context["x_role"] is None


# Absolute-scale pair (abs_diff/abs_se), family-dependent


class TestAbsoluteScalePair:
    def test_mean_family_matches_exact_inversion_identity(self):
        """Mean family: abs_diff/abs_se recovered exactly via the delta-method
        inversion identity - matches the textbook Welch difference of means."""
        control = _make_arm_stats(n=100, mean=10.0, var=4.0, group_id="control", metric="rev")
        treatment = _make_arm_stats(n=100, mean=12.0, var=4.0, group_id="treatment", metric="rev")
        df = _summary_df([control, treatment])

        result = estimate_lift(
            metrics=[_mean_metric()], summary=df, control_group="control"
        ).results
        assert len(result) == 1
        assert result[0].abs_diff == pytest.approx(2.0)
        assert result[0].abs_se == pytest.approx(math.sqrt(4.0 / 100 + 4.0 / 100))

    def test_mean_family_abs_diff_uses_arm_mean_not_log_exp_roundtrip(self):
        """abs_diff/abs_se must come from the arm's OWN mean, not
        ``exp(log(mean))`` - a regression test strong enough to actually
        catch a revert, unlike the sibling test above.

        ``mean=10.0``/``12.0`` (used above) happen to produce IDENTICAL
        ``abs_diff`` whether or not ``mean`` is round-tripped through
        ``exp(log(mean))`` first, so ``pytest.approx`` can't distinguish
        the two routes there. ``10.0``/``10.2`` do NOT round-trip
        bit-for-bit (verified below) and the resulting ``abs_diff`` values
        from the two routes are provably different in double precision, so
        this uses EXACT ``==`` - the only assertion strong enough to
        fail if ``_mean_abs_from_log`` were fed ``exp(log(mean))`` again.
        """
        c_mean, t_mean = 10.0, 10.2
        if math.exp(math.log(c_mean)) == c_mean or math.exp(math.log(t_mean)) == t_mean:
            pytest.skip(
                "this platform's libm round-trips exp(log(x)) exactly for the fixture "
                "values -- the fixture no longer discriminates the two routes, not a "
                "product regression"
            )
        direct_diff = t_mean - c_mean
        roundtrip_diff = math.exp(math.log(t_mean)) - math.exp(math.log(c_mean))
        if direct_diff == roundtrip_diff:
            pytest.skip("this platform's arithmetic makes the two routes coincide by luck")

        control = _make_arm_stats(n=100, mean=c_mean, var=4.0, group_id="control", metric="rev")
        treatment = _make_arm_stats(n=100, mean=t_mean, var=4.0, group_id="treatment", metric="rev")
        df = _summary_df([control, treatment])

        result = estimate_lift(
            metrics=[_mean_metric()], summary=df, control_group="control"
        ).results
        assert len(result) == 1
        assert result[0].abs_diff == direct_diff

    def test_ratio_family_direct_form(self):
        """Ratio family: abs_diff/abs_se from the DIRECT form, matching the
        textbook difference-of-ratios (control R=2.0, treatment R=2.4)."""
        control = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=100,
            sum_y=1000.0,
            sum_y2=10100.0,
            sum_den=500.0,
            sum_den2=2600.0,
            sum_yden=5030.0,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="treatment",
            n=100,
            sum_y=1200.0,
            sum_y2=14500.0,
            sum_den=500.0,
            sum_den2=2600.0,
            sum_yden=6030.0,
        )
        df = _summary_df([control, treatment])

        result = estimate_lift(
            metrics=[_ratio_metric()], summary=df, control_group="control"
        ).results
        assert len(result) == 1
        assert result[0].abs_diff == pytest.approx(0.3999999999999999)
        assert result[0].abs_se == pytest.approx(0.060702952851146255)

    def test_ratio_family_direct_form_agrees_with_relative_route_to_float_epsilon(self):
        """Var_delta(R) = R^2 * var_log_r is an algebraic identity
        measured here (not merely asserted) against RatioVarianceModel's own
        var_log_r at a typical (non-extreme) configuration."""
        from increment.estimation.engine import _ratio_abs_diff_se
        from increment.estimation.variance import RatioVarianceModel

        arm = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=1000,
            sum_y=5023.4,
            sum_y2=26117.9,
            sum_den=10012.7,
            sum_den2=101305.2,
            sum_yden=50412.6,
        )
        log_mean, se_log = RatioVarianceModel().log_mean_se(arm)
        r_direct, se_direct = _ratio_abs_diff_se(arm)

        var_relative = (math.exp(log_mean) ** 2) * se_log**2
        var_direct = se_direct**2
        assert r_direct == pytest.approx(math.exp(log_mean), rel=1e-12)
        assert var_direct == pytest.approx(var_relative, rel=1e-9)

    @pytest.mark.filterwarnings(
        "ignore:.*centered (sum of squares|cross sum).*floating-point noise.*:RuntimeWarning"
    )
    def test_ratio_family_direct_form_matches_independent_reference_at_extreme_small_numerator(
        self,
    ):
        """At an extreme small numerator mean, the direct form computes
        a genuinely nonzero, independently-verifiable SE. ``math.isfinite``
        alone cannot rule out a bug here - 0.0 is finite too - so this
        compares against a reference computed via EXACT rational arithmetic
        (fractions.Fraction + decimal.Decimal.sqrt) on the same inputs, not
        by re-deriving the value through the code under test.

        n_bar=1e-160 (the value this test used to pin) turns out to
        underflow ``var_r`` to exactly 0.0 in double precision - see
        ``test_ratio_family_relative_route_raises_where_direct_form_does_not``
        for that regime instead; 1e-100 here is extreme enough to exercise
        the absolute-scale derivation's near-zero-numerator concern while
        keeping ``var_r`` itself representable, so ``se_direct > 0`` is a
        meaningful assertion.
        """
        import decimal
        from decimal import Decimal
        from fractions import Fraction

        from increment.estimation.engine import _ratio_abs_diff_se

        n = 200
        n_bar = 1e-100
        sum_y = n_bar * n
        sum_y2 = sum_y**2 / n * 1.0001
        sum_den = 10.0 * n
        sum_den2 = sum_den**2 / n * 1.01
        sum_yden = sum_y * sum_den / n
        arm = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_den=sum_den,
            sum_den2=sum_den2,
            sum_yden=sum_yden,
        )

        r_direct, se_direct = _ratio_abs_diff_se(arm)
        assert se_direct > 0

        # Independent reference: same formula via exact rational arithmetic
        # and Decimal sqrt, a route distinct from _ratio_abs_diff_se's floats.
        nF, syF, sy2F = Fraction(n), Fraction(sum_y), Fraction(sum_y2)
        sdF, sd2F, sydF = Fraction(sum_den), Fraction(sum_den2), Fraction(sum_yden)
        n_bar_f = syF / nF
        d_bar_f = sdF / nF
        var_n_f = (sy2F - syF**2 / nF) / (nF - 1)
        var_d_f = (sd2F - sdF**2 / nF) / (nF - 1)
        cov_nd_f = (sydF - syF * sdF / nF) / (nF - 1)
        var_r_f = (Fraction(1) / nF) * (
            var_n_f / d_bar_f**2
            - 2 * n_bar_f * cov_nd_f / d_bar_f**3
            + n_bar_f**2 * var_d_f / d_bar_f**4
        )
        # Scoped via localcontext rather than mutating the process-global
        # decimal context, which would leak precision=80 into later tests.
        with decimal.localcontext() as ctx:
            ctx.prec = 80
            var_r_dec = Decimal(var_r_f.numerator) / Decimal(var_r_f.denominator)
            se_ref = float(var_r_dec.sqrt())
        r_ref = float(n_bar_f / d_bar_f)

        assert r_direct == pytest.approx(r_ref, rel=1e-12)
        assert se_direct == pytest.approx(se_ref, rel=1e-9)

    @pytest.mark.filterwarnings(
        "ignore:.*centered (sum of squares|cross sum).*floating-point noise.*:RuntimeWarning"
    )
    def test_ratio_family_both_routes_stay_well_behaved_at_a_tiny_numerator(self):
        """At an extreme numerator mean the two routes disagree, and both
        must stay informative. The direct form never divides by n_bar, only by
        d_bar, so it underflows to exactly zero -- the signed/near-zero-numerator
        case the absolute-scale derivation exists to rescue. The relative route
        (RatioVarianceModel.log_mean_se) used to divide by an underflowed
        ``n_bar**2`` and raise a bare ZeroDivisionError from a public estimator;
        each term now divides once by a mean, so it returns the scale-invariant
        log-ratio SE instead of crashing."""
        from increment.estimation.engine import _ratio_abs_diff_se
        from increment.estimation.variance import RatioVarianceModel

        n = 200
        n_bar = 1e-200
        sum_y = n_bar * n
        sum_y2 = sum_y**2 / n * 1.0001
        sum_den = 10.0 * n
        sum_den2 = sum_den**2 / n * 1.01
        sum_yden = sum_y * sum_den / n
        arm = ArmStats.from_raw_sums(
            study_id="exp1",
            metric="ratio_rev",
            group_id="control",
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_den=sum_den,
            sum_den2=sum_den2,
            sum_yden=sum_yden,
        )

        r_direct, se_direct = _ratio_abs_diff_se(arm)
        assert math.isfinite(r_direct)
        # EXACT equality, not math.isfinite: 0.0 is finite too, so isfinite
        # alone can't distinguish "correctly underflowed to zero" from a bug.
        assert se_direct == 0.0

        _log_mean, se_relative = RatioVarianceModel().log_mean_se(arm)
        assert math.isfinite(se_relative)
        assert se_relative > 0.0

    def test_cuped_branch_derives_from_adjusted_summaries(self):
        """CUPED: abs_diff/abs_se come from cuped_adjust's ADJUSTED mean/var,
        not the raw arms. abs_se is hand-computed; abs_diff is asserted
        exactly against a live cuped_adjust() call (the adjusted-mean
        values themselves are pinned separately, in
        tests/estimation/test_cuped.py::TestCupedPooledTheta::test_pooled_theta_across_arms)."""
        control = ArmStats.from_raw_sums(
            study_id="e1",
            metric="rev",
            group_id="control",
            n=300,
            sum_y=2991.246780958042,
            sum_y2=30131.737907659208,
            sum_x=1489.2549235662966,
            sum_x2=7703.721110558556,
            sum_xy=15045.631192533583,
        )
        treatment = ArmStats.from_raw_sums(
            study_id="e1",
            metric="rev",
            group_id="treatment",
            n=300,
            sum_y=3581.782312832768,
            sum_y2=43051.64306699056,
            sum_x=1491.9039023361884,
            sum_x2=7676.713989079367,
            sum_xy=17972.765990718326,
        )
        df = _summary_df([control, treatment])

        result = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        ).results
        assert len(result) == 1
        # EXACT equality against a live cuped_adjust() call, not approx: its
        # ~1e-6 tolerance can't catch abs_diff derived from a reconstructed mean.
        from increment.estimation.cuped import cuped_adjust

        adjusted = cuped_adjust([control, treatment])
        c_adj, t_adj = adjusted[0], adjusted[1]
        direct_diff = t_adj.mean - c_adj.mean
        roundtrip_diff = math.exp(math.log(t_adj.mean)) - math.exp(math.log(c_adj.mean))
        if direct_diff == roundtrip_diff:
            pytest.skip("this platform's arithmetic makes the two routes coincide by luck")
        assert result[0].abs_diff == direct_diff
        # Within-arm (fixed-effects) CUPED theta: the pre-fix joint-pooled theta
        # gave 0.06422253660071206, attenuated by the between-arm delta.
        assert result[0].abs_se == pytest.approx(0.06422150422889429)
        # NOT the raw (unadjusted) difference - would be materially different.
        raw_diff = treatment.mean_y() - control.mean_y()
        assert result[0].abs_diff != pytest.approx(raw_diff)

    def test_cuped_branch_nonpositive_adjusted_mean_yields_an_additive_only_row(self):
        """A strong covariate on a low-mean metric can push the CUPED-
        adjusted mean non-positive; this no longer aborts estimate_lift --
        the cell now returns a real additive-only row
        (relative_unavailable_reason='nonpositive_arm_mean'), the same
        contract TestNonPositiveArmMeanFailsOneCell covers for the
        unadjusted path."""
        n = 1000
        var_x = 1e-6
        var_y = 1.0
        cov_xy = 0.999999999 * (var_x * var_y) ** 0.5

        def _arm(mean_y: float, mean_x: float, group_id: str) -> ArmStats:
            sum_y = n * mean_y
            sum_y2 = var_y * (n - 1) + sum_y**2 / n
            sum_x = n * mean_x
            sum_x2 = var_x * (n - 1) + sum_x**2 / n
            sum_xy = cov_xy * (n - 1) + sum_x * sum_y / n
            return ArmStats.from_raw_sums(
                study_id="e1",
                metric="rev",
                group_id=group_id,
                n=n,
                sum_y=sum_y,
                sum_y2=sum_y2,
                sum_x=sum_x,
                sum_x2=sum_x2,
                sum_xy=sum_xy,
            )

        control = _arm(0.2, 0.0, "control")
        treatment = _arm(0.25, 0.001, "treatment")
        df = _summary_df([control, treatment])

        computation = estimate_lift(
            metrics=[_mean_metric()],
            summary=df,
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        )
        assert computation.failures == {}
        (row,) = computation.results
        assert row.lift is None
        assert row.relative_unavailable_reason == "nonpositive_arm_mean"
        assert row.abs_diff is not None
        assert row.abs_se is not None

    @pytest.mark.filterwarnings(
        "ignore:.*centered (sum of squares|cross sum).*floating-point noise.*:RuntimeWarning"
    )
    def test_ratio_family_degenerate_abs_variance_withholds_abs_se(self):
        """The absolute-scale variance (var_r, direct form) and the
        log-scale variance (var_log_r, used for the relative lift) are
        DIFFERENT quantities that can diverge under floating point: at an
        extreme near-zero numerator mean, var_r underflows to exactly 0.0
        (see test_ratio_family_both_routes_stay_well_behaved_at_a_tiny_numerator)
        while var_log_r stays strictly positive, so infer_lift's own
        "both arms zero variance" log-scale guard does NOT fire here - a
        distinct guard is needed for the absolute-scale companion stat.
        When it trips (both arms' direct-form SE == 0.0), abs_se is
        withheld as None rather than returned as a live 0.0, which would be
        an invalid meta-analytic weight (1/variance -> inf) downstream."""
        from increment.estimation.engine import _ratio_abs_diff_se
        from increment.estimation.variance import RatioVarianceModel

        def _extreme_numerator_arm(n, n_bar, group_id):
            sum_y = n_bar * n
            sum_y2 = sum_y**2 / n * 1.0001
            sum_den = 10.0 * n
            sum_den2 = sum_den**2 / n * 1.01
            sum_yden = sum_y * sum_den / n
            return ArmStats.from_raw_sums(
                study_id="exp1",
                metric="ratio_rev",
                group_id=group_id,
                n=n,
                sum_y=sum_y,
                sum_y2=sum_y2,
                sum_den=sum_den,
                sum_den2=sum_den2,
                sum_yden=sum_yden,
            )

        # 1e-170, not 1e-160: at 1e-160 the direct-form residual is subnormal
        # but representable, and the SE now computes it (~7e-164) instead of
        # underflowing. The trip condition needs a residual that reaches exactly
        # zero, which is what this scale produces.
        control = _extreme_numerator_arm(200, 1e-170, "control")
        treatment = _extreme_numerator_arm(200, 1e-170, "treatment")

        # Confirm the trip condition directly: both arms' direct-form SE is
        # exactly 0.0 (absolute scale)...
        assert _ratio_abs_diff_se(control)[1] == 0.0
        assert _ratio_abs_diff_se(treatment)[1] == 0.0
        # ...while the log-scale variance stays strictly positive, so
        # infer_lift's degenerate-data guard does not raise first.
        assert RatioVarianceModel().log_mean_se(control)[1] > 0.0
        assert RatioVarianceModel().log_mean_se(treatment)[1] > 0.0

        df = _summary_df([control, treatment])
        result = estimate_lift(
            metrics=[_ratio_metric()], summary=df, control_group="control"
        ).results
        assert len(result) == 1
        assert math.isfinite(result[0].require_lift().value)
        assert result[0].abs_se is None


class TestReservedMethodNames:
    """``Method.name`` is the label stamped on every LiftEstimate this call
    produces AND a dispatch key on the observational path (estimate_ate).
    A reserved estimator name that estimate_lift is not actually applying
    would ship an unadjusted number wearing an adjusted method's label;
    the engine must refuse rather than mislabel."""

    def _summary(self):
        control = _make_arm_stats(1000, 10.0, 4.0, group_id="control")
        treatment = _make_arm_stats(1000, 11.0, 4.0, group_id="treatment")
        return _summary_df([control, treatment])

    def test_cuped_label_without_cuped_reduction_refused(self):
        """Method(name='cuped') alone (the natural way to ask for CUPED;
        the real switch is variance_reduction='cuped') must raise, not
        return the unadjusted number labeled method='cuped'."""
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                [_mean_metric()],
                self._summary(),
                control_group="control",
                methods=[Method(name="cuped")],
            )
        assert exc_info.value.code == "estimation.engine.method.name_without_variance"

    @pytest.mark.parametrize("name", ["iptw", "dml"])
    def test_observational_adjustment_labels_refused(self, name):
        """iptw/dml are estimate_ate dispatch keys; estimate_lift never
        applies them, so the label is always a mislabel here."""
        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                [_mean_metric()],
                self._summary(),
                control_group="control",
                methods=[Method(name=name)],
            )
        assert exc_info.value.code == "estimation.engine.method_name_observational"

    def test_non_reserved_free_text_label_still_allowed(self):
        """Only reserved estimator names are refused - arbitrary display
        labels remain valid."""
        [result] = estimate_lift(
            [_mean_metric()],
            self._summary(),
            control_group="control",
            methods=[Method(name="banana")],
        ).results
        assert result.method == "banana"


class TestMethodConstructionTimeValidation:
    """Rules true on EVERY estimation path belong on the model, so a typo
    fails where it was typed rather than at dispatch. The reserved-name
    rule is deliberately NOT here - see TestReservedMethodNames."""

    def test_unregistered_variance_reduction_refused_at_construction(self):
        with pytest.raises(UnsupportedRequestError) as exc_info:
            Method(name="whatever", variance_reduction="bogus_vr")
        assert exc_info.value.code == "estimation.variance.registry.no_registered_available"
        assert exc_info.value.context["key"] == "bogus_vr"

    def test_registered_variance_reductions_construct(self):
        assert Method(name="unadjusted").variance_reduction == "none"
        assert Method(name="cuped", variance_reduction="cuped").variance_reduction == "cuped"

    def test_cuped_label_without_cuped_reduction_refused_at_construction(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Method(name="cuped")
        assert exc_info.value.code == "estimation.engine.method.name_without_variance"

    def test_free_text_label_still_constructs(self):
        """The moved rules must not narrow the label contract."""
        assert Method(name="banana").name == "banana"
        assert Method(name="panel-unadjusted").name == "panel-unadjusted"

    def test_observational_names_still_construct(self):
        """estimate_ate REQUIRES these names and defaults to Method(name='iptw'),
        so the reserved-name refusal cannot be a constructor rule."""
        for name in ("iptw", "dml", "aipw"):
            assert Method(name=name).name == name

    def test_iptw_refuses_outcome_learner_at_construction(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Method(name="iptw", outcome_learner=lambda: object())
        assert exc_info.value.code == "estimation.engine.method.name_iptw_does"

    def test_iptw_refuses_folds_at_construction(self):
        with pytest.raises(InvalidRequestError) as exc_info:
            Method(name="iptw", folds=3)
        assert exc_info.value.code == "estimation.engine.method.name_iptw_does"

    def test_iptw_accepts_propensity_learner(self):
        """IPTW's one pluggable field stays legal - only the two it has no
        seam for are refused."""
        assert Method(name="iptw", propensity_learner=lambda: object()).folds is None

    def test_dml_and_aipw_accept_all_three_nuisance_fields(self):
        for name in ("dml", "aipw"):
            m = Method(
                name=name,
                propensity_learner=lambda: object(),
                outcome_learner=lambda: object(),
                folds=7,
            )
            assert m.folds == 7


class TestRegisteredDecisionEvidence:
    @pytest.mark.slow
    @pytest.mark.parametrize("power", [False, True])
    def test_typed_likelihood_evidence_agrees_with_registered_decision(self, power):
        from increment import AlwaysValid
        from increment.decision import EValueEvidence
        from increment.estimation.sequential_runtime import estimate_sequential
        from tests.sequential_cases import (
            capture,
            records,
            registration,
        )

        policy = AlwaysValid(registration=registration())
        c = [0, 0, 0, 1] * 125
        t = [0, 1, 1, 1] * 125 if power else c
        bundle = estimate_sequential(capture(policy.registration, records(c, t)), policy)
        row = bundle.results[0]
        [evidence] = bundle.evidence.values()
        assert isinstance(evidence, EValueEvidence)
        assert row.stat_sig() == power
        assert (evidence.log_e > 3) == power
        assert evidence.checkpoint == row.require_sequential_result().checkpoint

    def test_registered_snapshot_preserves_explicit_empty_methods(self):
        from increment import AlwaysValid
        from tests.sequential_cases import (
            capture,
            records,
            registration,
        )

        policy = AlwaysValid(registration=registration())
        snapshot = capture(policy.registration, records([0, 1], [1, 1]))
        bundle = estimate_lift(
            [_mean_metric()],
            snapshot,
            "control",
            methods=[],
            inference=policy,
        )
        assert bundle.results == ()
        assert bundle.evidence == {}


# Arm-inventory ingress validation: duplicate rows, missing controls,
# fractional counts


class TestDuplicateArmRows:
    """A duplicate (metric, group_id) row must refuse instead of
    letting "last one wins" silently pick an arbitrary control."""

    def test_duplicate_control_rows_refuse_instead_of_last_wins(self):
        from increment.errors import InvalidRequestError

        control1 = _make_arm_stats(n=100, mean=1.0, var=1.0, group_id="control", metric="rev")
        control2 = _make_arm_stats(n=100, mean=2.0, var=1.0, group_id="control", metric="rev")
        treatment = _make_arm_stats(n=100, mean=1.1, var=1.0, group_id="treatment", metric="rev")
        df = _summary_df([control1, control2, treatment])

        with pytest.raises(InvalidRequestError) as exc:
            estimate_lift(metrics=[_mean_metric()], summary=df, control_group="control")
        assert exc.value.code == "estimation.engine.arm.duplicate_rows"
        assert ("rev", "control") in cast("list[tuple[str, str]]", exc.value.context["keys"])

    def test_duplicate_treatment_rows_also_refuse(self):
        from increment.errors import InvalidRequestError

        control = _make_arm_stats(n=100, mean=1.0, var=1.0, group_id="control", metric="rev")
        treatment1 = _make_arm_stats(n=100, mean=1.1, var=1.0, group_id="treatment", metric="rev")
        treatment2 = _make_arm_stats(n=100, mean=1.2, var=1.0, group_id="treatment", metric="rev")
        df = _summary_df([control, treatment1, treatment2])

        with pytest.raises(InvalidRequestError) as exc:
            estimate_lift(metrics=[_mean_metric()], summary=df, control_group="control")
        assert exc.value.code == "estimation.engine.arm.duplicate_rows"
        assert ("rev", "treatment") in cast("list[tuple[str, str]]", exc.value.context["keys"])


class TestMissingControlRow:
    """A treatment metric with no matching control row must emit a
    keyed decision failure, not silently vanish from the result set."""

    def test_missing_control_row_emits_keyed_failure_not_silence(self):
        m1_control = _make_arm_stats(n=100, mean=1.0, var=1.0, group_id="control", metric="m1")
        m1_treatment = _make_arm_stats(n=100, mean=1.1, var=1.0, group_id="treatment", metric="m1")
        m2_treatment = _make_arm_stats(n=100, mean=2.0, var=1.0, group_id="treatment", metric="m2")
        df = _summary_df([m1_control, m1_treatment, m2_treatment])

        computation = estimate_lift(
            metrics=[_mean_metric("m1"), _mean_metric("m2")],
            summary=df,
            control_group="control",
        )
        assert len(computation.results) == 1
        assert computation.results[0].metric == "m1"
        assert len(computation.failures) == 1
        failure = next(iter(computation.failures.values()))
        assert failure.hypothesis.metric == "m2"
        hypothesis = cast("Any", failure.hypothesis)
        assert hypothesis.group_id == "treatment"
        assert failure.code == "estimation.engine.missing_control"
        assert failure.context["control_group"] == "control"


class TestFractionalArmCount:
    """A fractional 'n' at the untyped dataframe ingress must refuse,
    not silently truncate into a different arm size."""

    def test_fractional_n_is_refused_not_truncated(self):
        from increment.errors import InvalidRequestError

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 2.9,
            "ref_y": 1.0,
            "cy1": 0.0,
            "cy2": 1.0,
        }
        with pytest.raises(InvalidRequestError) as exc:
            _df_to_arms([row])
        assert exc.value.code == "estimation.engine.arm.invalid_count"

    def test_whole_valued_float_n_still_accepted(self):
        """Regression guard: only a genuinely fractional count is refused --
        a whole-valued float (the existing test-fixture convention, see
        _summary_df's ``"n": float(a.n)``) must keep working."""

        row = {
            "experiment_id": "e",
            "metric": "rev",
            "group_id": "control",
            "n": 3.0,
            "ref_y": 1.0,
            "cy1": 0.0,
            "cy2": 1.0,
        }
        (arm,) = _df_to_arms([row])
        assert arm.n == 3


# Negative-variance clamp-vs-refuse policy, engine.py ratio-variance side


class TestRatioVarianceClampPolicy:
    def test_clamp_negative_variance_within_roundoff_returns_zero(self):
        from increment.estimation.armstats import clamp_negative_variance

        result = clamp_negative_variance(-1e-20, magnitude=1.0, n=100)
        assert result == 0.0

    def test_clamp_negative_variance_beyond_roundoff_returns_none(self):
        from increment.estimation.armstats import clamp_negative_variance

        result = clamp_negative_variance(-5.0, magnitude=1.0, n=100)
        assert result is None

    def test_materially_negative_ratio_variance_refuses(self):
        """A Cauchy-Schwarz-violating (infeasible) covariance must refuse
        instead of reporting a clamped, maximal-confidence zero-SE."""
        from increment.errors import InvalidRequestError
        from increment.estimation.engine import ratio_abs_diff_se

        with pytest.raises(InvalidRequestError) as exc:
            ratio_abs_diff_se(1.0, 1.0, 1.0, 1.0, 1000.0, 100)
        assert exc.value.code == "estimation.engine.ratio.negative_variance"

    def test_ordinary_ratio_moments_are_unaffected(self):
        from increment.estimation.engine import ratio_abs_diff_se

        r, se = ratio_abs_diff_se(10.0, 5.0, 1.0, 1.0, 0.2, 100)
        assert r == pytest.approx(2.0)
        assert se > 0.0


# CUPED at small n with a strong covariate: the within-arm theta in
# estimation/cuped.py removes the attenuation the pooled form had, so this
# regime is estimated rather than refused.


class TestCupedSmallSampleStrongCovariate:
    @staticmethod
    def _arm(n: int, mean_y: float, var_y: float, cov_xy: float, group_id: str) -> ArmStats:
        """Deterministic (n, mean_y, var_y, mean_x=0, var_x=1, cov(x,y)=cov_xy)
        arm via exact raw sums -- no sampling."""
        sum_x = 0.0
        sum_x2 = 1.0 * (n - 1)  # var_x == 1
        sum_y = n * mean_y
        sum_y2 = var_y * (n - 1) + sum_y**2 / n
        sum_xy = cov_xy * (n - 1)  # sum_x == 0, so the cross term is exact
        return ArmStats.from_raw_sums(
            study_id="e1",
            metric="rev",
            group_id=group_id,
            n=n,
            sum_y=sum_y,
            sum_y2=sum_y2,
            sum_x=sum_x,
            sum_x2=sum_x2,
            sum_xy=sum_xy,
        )

    def _run(self, control: ArmStats, treatment: ArmStats):
        return estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")],
        ).results

    def test_low_n_high_rho_is_estimated_not_refused(self):
        """n=20/arm at rho~=0.89 -- the regime the pooled-theta form
        attenuated. Within-arm theta recovers the arm difference, so it is
        reported instead of refused."""
        control = self._arm(20, 10.0, 1.25, 1.0, "control")
        treatment = self._arm(20, 15.0, 1.25, 1.0, "treatment")

        result = self._run(control, treatment)

        assert len(result) == 1
        # The absolute difference is recovered, not shrunk toward zero.
        assert result[0].abs_diff == pytest.approx(5.0, rel=1e-9)

    def test_large_n_and_low_n_weak_covariate_both_proceed(self):
        assert (
            len(
                self._run(
                    self._arm(1000, 10.0, 1.0, 0.7, "control"),
                    self._arm(1000, 12.0, 1.0, 0.7, "treatment"),
                )
            )
            == 1
        )
        assert (
            len(
                self._run(
                    self._arm(100, 10.0, 1.0, 0.1, "control"),
                    self._arm(100, 12.0, 1.0, 0.1, "treatment"),
                )
            )
            == 1
        )


class TestRatioNeighboringMeans:
    @staticmethod
    def _run(
        control: ArmStats,
        treatment: ArmStats,
        *,
        cuped: bool = False,
        null_lift: float = 0.0,
    ):
        return estimate_lift(
            metrics=[_ratio_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            null_lift=null_lift,
            methods=[Method(name="cuped", variance_reduction="cuped")] if cuped else None,
        ).results[0]

    @pytest.mark.parametrize("cuped", [False, True])
    def test_common_between_arm_component_scaling_preserves_tiny_ratio_effect(self, cuped):
        result = self._run(
            _ratio_arm("control", 1.0, 1.0, covariate=cuped),
            _ratio_arm("treatment", float(2**50 + 1), float(2**50), covariate=cuped),
            cuped=cuped,
        )
        assert result.require_lift().value == pytest.approx(2**-50, rel=1e-12, abs=0)

    @pytest.mark.parametrize("cuped", [False, True])
    def test_ratio_lift_preserves_neighboring_large_means(self, cuped: bool):
        control = _ratio_arm("control", 1e15, covariate=cuped)
        treatment = _ratio_arm("treatment", 1e15 + 1.0, covariate=cuped)
        ratio = self._run(control, treatment, cuped=cuped)

        mean_control = TestLargeOffsetLogRatio._arm("control", 1e15, covariate=cuped)
        mean_treatment = TestLargeOffsetLogRatio._arm("treatment", 1e15 + 1, covariate=cuped)
        mean = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([mean_control, mean_treatment]),
            control_group="control",
            methods=[Method(name="cuped", variance_reduction="cuped")] if cuped else None,
        ).results[0]

        assert ratio.abs_diff == pytest.approx(1.0)
        assert ratio.require_lift().value == pytest.approx(1e-15, rel=1e-12, abs=0)
        assert ratio.require_lift().value == pytest.approx(
            mean.require_lift().value, rel=1e-12, abs=0
        )
        assert ratio.stat_sig() == mean.stat_sig()
        assert ratio.require_lift().lb == pytest.approx(mean.require_lift().lb, rel=1e-12, abs=0)
        assert ratio.require_lift().ub == pytest.approx(mean.require_lift().ub, rel=1e-12, abs=0)

    def test_component_unit_scaling_preserves_relative_lift(self):
        control = _ratio_arm("control", 10.0, 2.0)
        treatment = _ratio_arm("treatment", 12.0, 2.0)
        base = self._run(control, treatment)
        scale = 1e11
        scaled = self._run(
            *[
                arm.model_copy(
                    update={
                        "ref_y": arm.ref_y * scale,
                        "cy2": arm.cy2 * scale**2,
                        "ref_den": 2.0 * scale,
                    }
                )
                for arm in (control, treatment)
            ]
        )
        for field in ("value", "lb", "ub", "log_se"):
            assert getattr(scaled.require_lift(), field) == pytest.approx(
                getattr(base.require_lift(), field), rel=1e-12, abs=0
            )
        assert scaled.stat_sig() == base.stat_sig()

    @pytest.mark.parametrize("scale", [1.0, 1e150, 1e300])
    def test_scaled_ratio_components_keep_moderate_log_lift_accurate(self, scale: float):
        """The public engine forms a ratio cross-ratio before taking one log."""
        from fractions import Fraction

        num_c, den_c = 1.23456789, 0.823456789
        num_t, den_t = 1.87654321, 0.487654321
        control = _ratio_arm("control", num_c, den_c)
        treatment = _ratio_arm("treatment", num_t, den_t)
        # Keep centered numerator variance representable even at 1e300 while
        # scaling it consistently with the component mean.
        cy2 = (1e-300 * scale) * scale * (TestLargeOffsetLogRatio.N - 1)
        scaled = self._run(
            control.model_copy(
                update={"ref_y": num_c * scale, "ref_den": den_c * scale, "cy2": cy2}
            ),
            treatment.model_copy(
                update={"ref_y": num_t * scale, "ref_den": den_t * scale, "cy2": cy2}
            ),
        )
        exact_ratio = (
            Fraction(num_t * scale)
            * Fraction(den_c * scale)
            / (Fraction(num_c * scale) * Fraction(den_t * scale))
        )
        want_log_rr = math.log(float(exact_ratio))
        lift = scaled.require_lift()
        assert lift.log_mean is not None
        assert abs(lift.log_mean - want_log_rr) <= math.ulp(want_log_rr)

    def test_subnormal_cross_ratio_keeps_informative_prior_result_accurate(self):
        from decimal import Decimal, localcontext
        from fractions import Fraction

        from increment.estimation.inference import Normal

        num_t = math.nextafter(1.0, math.inf)
        den_c, den_t = math.ldexp(1.0, -500), math.ldexp(1.0, 575)
        control = _ratio_arm("control", 1.0, den_c).model_copy(update={"n": 10, "cy2": 0.9})
        treatment = _ratio_arm("treatment", num_t, den_t).model_copy(update={"n": 10, "cy2": 0.9})
        prior = Normal(mu=0.0, sigma=0.005)
        result = estimate_lift(
            metrics=[_ratio_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            prior=prior,
        ).results[0]
        ratio = Fraction(num_t) * Fraction(den_c) / Fraction(den_t)
        with localcontext() as context:
            context.prec = 100
            log_ratio = float((Decimal(ratio.numerator) / Decimal(ratio.denominator)).ln())
        variance = 0.01 + 0.01 / num_t**2
        posterior_mean = log_ratio * prior.sigma**2 / (variance + prior.sigma**2)
        assert result.require_lift().value == pytest.approx(
            math.expm1(posterior_mean), rel=1e-12, abs=0
        )

    def test_shifted_null_matches_mean_metric_reference(self):
        null_lift = 1e-15
        ratio = self._run(
            _ratio_arm("control", 1e15),
            _ratio_arm("treatment", 1e15 + 1.0),
            null_lift=null_lift,
        )
        control = TestLargeOffsetLogRatio._arm("control", 1e15, covariate=False)
        treatment = TestLargeOffsetLogRatio._arm("treatment", 1e15 + 1, covariate=False)
        mean = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            null_lift=null_lift,
        ).results[0]
        assert ratio.stat_sig() == mean.stat_sig()
        assert ratio.reference_kind == mean.reference_kind

    def test_finite_near_overflow_component_means(self):
        result = self._run(
            _ratio_arm("control", 1e307, 1e307),
            _ratio_arm("treatment", 1.1e307, 1e307),
        )
        assert math.isfinite(result.require_lift().value)
        assert result.require_lift().value == pytest.approx(0.1, rel=1e-12)


class TestLargeOffsetLogRatio:
    """The joint log ratio is formed from the raw arm means, never as a
    difference of two independently rounded ``log(mean)`` values.

    Two means differing by exactly 1 at an offset of 1e15 have logs that
    round to the same double, so ``log_t - log_c`` collapses a true 1e-15
    relative effect to zero (and 1e6/1e12 lose 9.5e-11/1.9e-3 of it).
    ``log1p((mean_t - mean_c) / mean_c)`` is correctly rounded there: the
    subtraction is exact below 2^53 and the tiny ratio has no
    cancellation left to lose.
    """

    N = 100
    VAR_Y = 4.0

    @staticmethod
    def _oracle_log_ratio(mean_c: float, mean_t: float) -> float:
        """``log(mean_t / mean_c)`` at 60 significant digits, then rounded
        once to a double: an independent reference for the engine's value."""
        from decimal import Decimal, getcontext

        getcontext().prec = 60
        return float((Decimal(mean_t) / Decimal(mean_c)).ln())

    @classmethod
    def _arm(cls, group_id: str, mean: float, *, covariate: bool) -> ArmStats:
        # Centered fields make the stored mean exactly ``mean`` at any offset.
        # The covariate, when present, has the same mean in both arms so the
        # CUPED anchor term vanishes and the adjusted mean equals the raw mean.
        return ArmStats(
            study_id="exp1",
            metric="rev",
            group_id=group_id,
            n=cls.N,
            ref_y=mean,
            cy1=0.0,
            cy2=cls.VAR_Y * (cls.N - 1),
            ref_x=3.0 if covariate else None,
            cx1=0.0 if covariate else None,
            cx2=1.0 * (cls.N - 1) if covariate else None,
            cxy=0.6 * (cls.N - 1) if covariate else None,
            x_role="covariate" if covariate else None,
        )

    @pytest.mark.parametrize("offset", [1e6, 1e12, 1e15])
    @pytest.mark.parametrize("method", ["unadjusted", "cuped"])
    def test_true_relative_effect_survives_a_large_offset(self, offset: float, method: str):
        mean_c, mean_t = offset, offset + 1.0
        assert mean_t - mean_c == 1.0  # exactly representable below 2**53
        want_log_rr = self._oracle_log_ratio(mean_c, mean_t)
        true_lift = 1.0 / offset

        use_cuped = method == "cuped"
        control = self._arm("control", mean_c, covariate=use_cuped)
        treatment = self._arm("treatment", mean_t, covariate=use_cuped)
        (row,) = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            methods=[Method(name=method, variance_reduction="cuped" if use_cuped else "none")],
        ).results

        # Correctly rounded joint log ratio: within one ulp of the oracle.
        lift = row.require_lift()
        assert lift.log_mean is not None
        assert abs(lift.log_mean - want_log_rr) <= math.ulp(want_log_rr)
        assert abs(lift.value - true_lift) <= math.ulp(true_lift)
        assert lift.lb is not None and lift.ub is not None
        assert lift.lb < true_lift < lift.ub

    @pytest.mark.parametrize("offset", [1e6, 1e12, 1e15])
    def test_per_arm_log_scale_se_is_the_delta_method_se(self, offset: float):
        """Only the joint point estimate changes representation: each arm's
        log-scale SE is still ``sqrt(var / (n * mean**2))``, combined in
        quadrature."""
        from increment.estimation.variance import se_log_mean

        mean_c, mean_t = offset, offset + 1.0
        control = self._arm("control", mean_c, covariate=False)
        treatment = self._arm("treatment", mean_t, covariate=False)
        (row,) = estimate_lift(
            metrics=[_mean_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
        ).results

        se_t = se_log_mean(self.VAR_Y, mean_t, self.N)
        se_c = se_log_mean(self.VAR_Y, mean_c, self.N)
        assert row.require_lift().log_se == math.hypot(se_t, se_c)


class TestBinomialDirectionalAlphaConvention:
    """Exact inversion spends caller alpha while display metadata follows
    the site's central-equivalent directional convention."""

    def _binomial_arms(self, n: int = 200):
        control = ArmStats.from_raw_sums(
            study_id="exp1", metric="conv", group_id="control", n=n, sum_y=20.0, sum_y2=20.0
        )
        treatment = ArmStats.from_raw_sums(
            study_id="exp1", metric="conv", group_id="treatment", n=n, sum_y=40.0, sum_y2=40.0
        )
        return control, treatment

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_half_alpha_input_yields_full_alpha_effective_level(self, alternative):
        from increment.estimation.engine import estimate_lift

        control, treatment = self._binomial_arms()
        df = _summary_df([control, treatment])
        target_alpha = 0.10
        [result] = estimate_lift(
            metrics=[_conversion_metric()],
            summary=df,
            control_group="control",
            alpha=target_alpha / 2.0,
            alternative=alternative,
        ).results
        assert result.reference_kind == "binomial"
        assert result.binomial_set is not None
        assert result.binomial_set.alpha == pytest.approx(target_alpha)
        assert result.binomial_set.level == pytest.approx(1.0 - target_alpha)
        assert result.binomial_set.decision_alpha == pytest.approx(target_alpha / 2.0)
        assert result.require_lift().alpha == pytest.approx(target_alpha)

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_fcr_reinversion_preserves_the_full_alpha_end_to_end(self, alternative):
        """Both directional geometries are reinverted at full FCR alpha."""
        from increment.estimation import binomial_rr
        from increment.estimation.engine import estimate_lift
        from increment.estimation.results import open_bound_from_two_sided_at_target

        control, treatment = self._binomial_arms()
        df = _summary_df([control, treatment])
        fcr_alpha = 0.10
        alpha_for = fcr_alpha / 2.0  # `_fcr_alpha_for`'s directional halving
        [nominal] = estimate_lift(
            metrics=[_conversion_metric()],
            summary=df,
            control_group="control",
            alpha=alpha_for,
            alternative=alternative,
        ).results
        reestimated = open_bound_from_two_sided_at_target(nominal)
        assert reestimated.binomial_set is not None
        assert reestimated.binomial_set.alpha == pytest.approx(fcr_alpha)
        assert reestimated.binomial_set.decision_alpha == pytest.approx(fcr_alpha)
        assert reestimated.binomial_set.nuisance_beta == binomial_rr.nuisance_beta(fcr_alpha)
        assert reestimated.require_lift().alpha == pytest.approx(fcr_alpha)
        direct = binomial_rr.confidence_interval(
            20, 200, 40, 200, alpha=fcr_alpha, alternative=alternative
        )
        expected = binomial_rr.to_lift_bounds(direct)
        assert (
            reestimated.require_lift().lb,
            reestimated.require_lift().ub,
        ) == expected

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_fcr_reinverts_set_only_rows_at_full_target_alpha(self, alternative):
        from increment.estimation import binomial_rr
        from increment.estimation.engine import estimate_lift
        from increment.estimation.results import open_bound_from_two_sided_at_target

        treatment = ArmStats.from_raw_sums(
            study_id="exp1", metric="conv", group_id="treatment", n=200, sum_y=4, sum_y2=4
        )
        control = ArmStats.from_raw_sums(
            study_id="exp1", metric="conv", group_id="control", n=200, sum_y=0, sum_y2=0
        )
        fcr_alpha = 0.10
        [parent] = estimate_lift(
            metrics=[_conversion_metric()],
            summary=_summary_df([control, treatment]),
            control_group="control",
            alpha=fcr_alpha / 2.0,
            alternative=alternative,
        ).results
        assert parent.lift is None

        row = open_bound_from_two_sided_at_target(parent)

        assert row.lift is None
        assert row.binomial_set is not None
        assert row.binomial_set.decision_alpha == pytest.approx(fcr_alpha)
        assert row.binomial_set.nuisance_beta == binomial_rr.nuisance_beta(fcr_alpha)
        direct = binomial_rr.confidence_interval(
            0, 200, 4, 200, alpha=fcr_alpha, alternative=alternative
        )
        assert (row.binomial_set.lower, row.binomial_set.upper) == binomial_rr.to_lift_bounds(
            direct
        )

    @pytest.mark.parametrize("x_role", ["cluster_size", "uptake_total"])
    def test_non_covariate_x_family_is_not_projected_into_iid_binomial(self, x_role):
        arms = []
        for group_id, successes in (("control", 20), ("treatment", 40)):
            arm = ArmStats.from_raw_sums(
                study_id="exp1",
                metric="conv",
                group_id=group_id,
                n=200,
                sum_y=float(successes),
                sum_y2=float(successes),
                sum_x=200.0,
                sum_x2=200.0,
                sum_xy=float(successes),
            ).model_copy(update={"x_role": x_role})
            arms.append(arm)

        with pytest.raises(InvalidRequestError) as exc_info:
            estimate_lift(
                metrics=[_conversion_metric()],
                summary=_summary_df(arms),
                control_group="control",
            )

        assert exc_info.value.code == "estimation.binomial.independent_units_required"

    @pytest.mark.parametrize("alternative", ["greater", "less"])
    def test_fcr_reinversion_round_trips_through_json(self, alternative):
        """The re-estimated FCR row's ``lift`` bounds must agree with its
        own ``binomial_set`` bounds on both sides, for BOTH directional
        alternatives -- ``LiftEstimate._binomial_lift_availability``
        enforces that "iff" contract on every (re-)validation, including
        a JSON round-trip of a persisted row, not just at construction
        time (``model_copy`` skips validation, so a mismatch there would
        stay silent until the next validate)."""
        from increment.estimation.engine import estimate_lift
        from increment.estimation.results import LiftEstimate, open_bound_from_two_sided_at_target

        control, treatment = self._binomial_arms()
        df = _summary_df([control, treatment])
        fcr_alpha = 0.10
        alpha_for = fcr_alpha / 2.0
        [nominal] = estimate_lift(
            metrics=[_conversion_metric()],
            summary=df,
            control_group="control",
            alpha=alpha_for,
            alternative=alternative,
        ).results
        reestimated = open_bound_from_two_sided_at_target(nominal)
        bset = reestimated.binomial_set
        assert bset is not None
        lift = reestimated.require_lift()
        assert lift.lb == bset.lower
        assert lift.ub == bset.upper
        round_tripped = LiftEstimate.model_validate_json(reestimated.model_dump_json())
        assert round_tripped.require_lift().lb == lift.lb
        assert round_tripped.require_lift().ub == lift.ub


class TestEncouragementItiRetainsBinomialRoute:
    """Uptake moments (sum_d) attached to an arm alongside its conversion
    outcome must not disqualify the exact independent-binomial route:
    they describe a different random variable over the same units and
    do not change the ITT's sufficient statistics (x_c, n_c, x_t, n_t)."""

    def _arm_with_uptake(self, *, n, successes, uptake, group_id):
        return ArmStats.from_raw_sums(
            study_id="e",
            metric="converted",
            group_id=group_id,
            n=n,
            sum_y=float(successes),
            sum_y2=float(successes),
            sum_d=float(uptake),
            sum_yd=float(min(successes, uptake)),
            sum_y2d=float(min(successes, uptake)),
        )

    def test_itt_with_uptake_matches_randomized_binomial_set(self):
        """Same (x_c, n_c, x_t, n_t) reached two ways (an arm with no
        uptake attached, an identical arm WITH uptake attached) must
        produce the identical binomial_set."""
        treatment_plain = self._arm_with_uptake(n=300, successes=43, uptake=0, group_id="treatment")
        control_plain = self._arm_with_uptake(n=300, successes=29, uptake=0, group_id="control")
        treatment_uptake = self._arm_with_uptake(
            n=300, successes=43, uptake=120, group_id="treatment"
        )
        control_uptake = self._arm_with_uptake(n=300, successes=29, uptake=0, group_id="control")
        metric = _conversion_metric("converted")
        plain = estimate_lift(
            [metric], _summary_df([control_plain, treatment_plain]), control_group="control"
        ).results[0]
        uptake = estimate_lift(
            [metric], _summary_df([control_uptake, treatment_uptake]), control_group="control"
        ).results[0]
        assert plain.reference_kind == "binomial"
        assert uptake.reference_kind == "binomial"
        assert uptake.binomial_set == plain.binomial_set
