"""Switchback measures and forecasts retain their randomized replication grain."""

import json
from datetime import date
from decimal import Decimal
from fractions import Fraction
from statistics import covariance

import polars as pl
import pytest

from increment import SwitchbackObservationSchedule, from_switchback_frame, run
from increment.declarations import Method
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.contrast import SwitchbackMetricStats, UnavailableContrastStats
from increment.frame import MetricSpec, from_switchback_panel
from increment.semantics.assignment import FixedCountOrder, SwitchbackAssignment, SwitchbackWindow
from increment.semantics.design import Randomized
from increment.semantics.models import AnalysisPlan
from increment.workflows import planning_baseline
from tests.test_frame_switchback import _assignment, _identification, _shared_assignment

_RATIO = MetricSpec(name="rate", type="ratio", numerator="sales", denominator="visits")
_ADJUSTED = MetricSpec(name="sales", decision_method=Method.cuped(covariates="forecast"))


def _paired(units=("a", "b")):
    rows, forecasts = [], []
    for i, unit in enumerate(units):
        for cycle in (0, 1):
            for period in (0, 1):
                treated = (cycle + period) % 2 == 1
                cell = {"unit": unit, "cycle": cycle, "period": period}
                rows.append(
                    dict(
                        **cell,
                        step=0,
                        arm="treatment" if treated else "control",
                        sales=2**55 + period * (2 + i) + 5 * treated,
                        visits=4,
                        forecast=int(treated),
                    )
                )
                forecasts.append(
                    dict(
                        **cell,
                        forecast=2**55 + period * (3 + i),
                        forecast_at=date(2025, 1, 1),
                        assigned_at=date(2025, 1, 2),
                    )
                )
    return pl.DataFrame(rows).reverse(), pl.DataFrame(forecasts)


def _build(frame, specs, forecasts=None, *, shared=False, builder=from_switchback_panel, **options):
    design = (_shared_assignment if shared else _assignment)(washout=0, observation=1, p_ct=0.25)
    return builder(
        frame,
        unit="unit",
        cycle="cycle",
        period="period",
        step="step",
        group="arm",
        identification=_identification(),
        assignment=design,
        pre_assignment_forecasts=forecasts,
        metrics=specs,
        **options,
    )


@pytest.mark.parametrize(
    "shared,ys,xs",
    [
        (True, (15, 5 / 3), (7, -7 / 3)),
        (False, (8, 26 / 3), (2, 8 / 3)),
    ],
)
def test_frozen_forecasts_and_integer_outcomes_share_signed_order_weights(shared, ys, xs):
    rows, forecasts = _paired()
    source = _build(rows, [_ADJUSTED], forecasts, shared=shared)
    evidence = source.contrast_stats(source.metrics[0])
    assert isinstance(evidence, SwitchbackMetricStats)
    assert evidence.cuped_outcomes == pytest.approx(ys)
    assert evidence.cuped_predictors == pytest.approx(xs)
    assert source.context.procedures["sales"].forecast_column == "forecast"


@pytest.mark.parametrize("grain", ["shared_block", "unit_cycle"])
def test_fixed_count_ratio_pairs_raw_totals_before_replication(grain):
    rows = []
    for u in range(4):
        for c in range(3):
            for p in range(3):
                treated = p != c
                for s in range(2):
                    rows.append(
                        {
                            "unit": str(u),
                            "cycle": c,
                            "period": p,
                            "step": s,
                            "arm": "T" if treated else "C",
                            "sales": 10 * (u + c) + p + 99 * (s == 0),
                            "visits": u + 2 * c + p + 1,
                        }
                    )
    assignment = SwitchbackAssignment(
        periods_per_cycle=3,
        sequence=FixedCountOrder(treated_periods=2, independence_unit=grain),
        window=SwitchbackWindow(washout_steps=1, observation_steps=1),
    )
    frame = pl.DataFrame(rows).reverse()
    frames = [frame]
    if grain == "shared_block":
        frames.append(
            pl.concat([frame, frame.with_columns((pl.col("unit") + "-clone").alias("unit"))])
        )
    # The treated period's mean position is (3-c)/2; the control position is c.
    expected = (
        [
            (15 + 10 * c + (3 - c) / 2, 2.5 + 2 * c + (3 - c) / 2, 15 + 11 * c, 2.5 + 3 * c)
            for c in range(3)
        ]
        if grain == "shared_block"
        else [(10 * u + 11, u + 4, 10 * u + 11, u + 4) for u in range(4)]
    )
    for roster in frames:
        source = from_switchback_panel(
            roster,
            metrics=[_RATIO],
            unit="unit",
            cycle="cycle",
            period="period",
            step="step",
            group="arm",
            identification=Randomized(control_group="C", allocation={"C": 1 / 3, "T": 2 / 3}),
            assignment=assignment,
        )
        stats = source.contrast_stats(source.metrics[0])
        assert isinstance(stats, SwitchbackMetricStats)
        joint = stats.ratio_components
        for observed, want in zip(joint, expected, strict=True):
            assert observed == pytest.approx(want)
        assert covariance([r[0] for r in joint], [r[2] for r in joint]) > 0


def test_integer_ratio_components_retain_small_treatment_effect_after_averaging():
    rows, _ = _paired()
    source = _build(rows, [_RATIO])
    stats = source.contrast_stats(source.metrics[0])
    assert isinstance(stats, SwitchbackMetricStats)
    restored = SwitchbackMetricStats.model_validate_json(stats.model_dump_json())
    assert restored.ratio_components == stats.ratio_components
    yt, dt, yc, dc = map(sum, zip(*restored.ratio_components, strict=True))
    # HT signed shifts in Y are (8, 26/3); both arm denominators are 4 * (2 + 2/3)/2.
    assert yt / dt - yc / dc == Fraction(25, 16)


@pytest.mark.parametrize("column", ["sales", "visits"])
def test_ratio_requires_numeric_measures_in_both_arms(column):
    rows, _ = _paired()
    with pytest.raises(InvalidRequestError) as caught:
        _build(rows.with_columns(pl.lit("3").alias(column)), [_RATIO])
    assert caught.value.context["column"] == column
    assert caught.value.context["reason"] == "nonnumeric_metric"


@pytest.mark.parametrize(
    "forecast_dtype,forecast,assigned_dtype,assigned,admitted",
    [
        (pl.Datetime("us", "UTC"), 1, pl.Datetime("ns", "UTC"), 1001, True),
        (pl.Datetime("us", "UTC"), 1, pl.Datetime("ns", "UTC"), 1000, False),
        # Year 3000 is representable in microseconds but not in int64 nanoseconds.
        (pl.Datetime("ns"), 1, pl.Datetime("us"), 32_503_680_000_000_000, True),
    ],
)
def test_forecast_deadline_compares_exact_instants(
    forecast_dtype, forecast, assigned_dtype, assigned, admitted
):
    rows, forecasts = _paired()
    forecasts = forecasts.with_columns(
        pl.lit(forecast).cast(forecast_dtype).alias("forecast_at"),
        pl.lit(assigned).cast(assigned_dtype).alias("assigned_at"),
    )
    if admitted:
        source = _build(rows, [_ADJUSTED], forecasts)
        evidence = source.contrast_stats(source.metrics[0])
        assert isinstance(evidence, SwitchbackMetricStats)
        assert evidence.cuped_predictors == pytest.approx((2, 8 / 3))
    else:
        with pytest.raises(CapabilityError) as caught:
            _build(rows, [_ADJUSTED], forecasts)
        assert caught.value.context["reason"] == "forecast_not_pre_assignment"


def test_missing_forecast_withholds_only_its_consumers_at_the_declared_alpha():
    rows, forecasts = _paired(("a", "b", "c", "d"))
    bonus = ((pl.col("unit") == "a") & (pl.col("arm") == "treatment")).cast(pl.Int64)
    rows = rows.with_columns((pl.col("sales") - 2**55 + bonus).alias("sales"))
    forecasts = forecasts.with_columns((pl.col("forecast") - 2**55).alias("forecast"))
    missing = (pl.col("unit") == "b") & (pl.col("cycle") == 1) & (pl.col("period") == 0)
    forecasts = forecasts.with_columns(
        pl.when(missing).then(None).otherwise(pl.col("forecast")).alias("partial")
    ).reverse()
    specs = [
        _ADJUSTED,
        _RATIO,
        MetricSpec(name="ordinary", value_column="sales"),
        *(
            MetricSpec(
                name=name, value_column="sales", decision_method=Method.cuped(covariates="partial")
            )
            for name in ("first", "second")
        ),
    ]
    experiment, reader = _build(
        rows,
        specs,
        forecasts,
        builder=from_switchback_frame,
        plan=AnalysisPlan(primary=[spec.name for spec in specs]),
    )
    with reader:
        result = run(experiment, reader).readout().to_frame(backend="polars")
        with pytest.raises(CapabilityError) as pilot:
            planning_baseline(experiment, reader, "first")
    assert isinstance(result, pl.DataFrame)
    by_name = {row["metric"]: row for row in result.to_dicts()}
    # Unit contrasts are (28, 26, 28, 30)/3 for sales, (6, 8, 10, 12)/3 for forecast.
    for name, point in (("ordinary", 28 / 3), ("rate", 7 / 4), ("sales", 122 / 15)):
        assert by_name[name]["estimate"] == pytest.approx(point)
        assert by_name[name]["sampling_available"] and by_name[name]["standard_error"] > 0
    for name in ("first", "second"):
        row = by_name[name]
        assert row["forecast_column"] == "partial"
        assert row["estimate"] is row["standard_error"] is row["p_value"] is None
        assert row["failure_code"] == pilot.value.code == "source.frame.switchback.metric"
        assert json.loads(row["failure_context"]) == {
            "metric": name,
            "column": "partial",
            "reason": "missing_forecast",
            "missing_count": 1,
            "first_missing_key": ["b", 1, 0],
        }
    assert pilot.value.context["first_missing_key"] == ("b", 1, 0)
    for row in by_name.values():
        assert row["alpha"] == pytest.approx(0.05 / 5)
        assert not row["decision_scope_complete"]
        assert (row["required_cell_count"], row["available_cell_count"]) == (5, 3)


@pytest.mark.parametrize("payload", ["arrow-null", "polars-nan", "pandas-object"])
def test_missing_forecast_values_keep_the_first_scheduled_key(payload):
    rows, forecasts = _paired()
    forecasts = forecasts.reverse()
    if payload == "pandas-object":
        import pandas as pd

        table = forecasts.with_columns(
            pl.col("forecast_at", "assigned_at").cast(pl.Datetime("ns"))
        ).to_pandas()
        table["forecast"] = [pd.NA if unit == "b" else Decimal("0.1") for unit in table["unit"]]
        key, count = ("b", 0, 0), 4
    elif payload == "arrow-null":
        table = forecasts.with_columns(pl.lit(None).alias("forecast")).to_arrow()
        key, count = ("a", 0, 0), 8
    else:
        table = forecasts.with_columns(pl.lit(float("nan")).alias("forecast"))
        key, count = ("a", 0, 0), 8
    source = _build(rows, [_ADJUSTED], table)
    stats = source.contrast_stats(source.metrics[0])
    assert isinstance(stats, UnavailableContrastStats)
    assert stats.failure_context["first_missing_key"] == key
    assert stats.failure_context["missing_count"] == count


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("observations_only", "pre_assignment_forecasts_required"),
        ("late", "forecast_not_pre_assignment"),
        ("infinite", "nonfinite_forecast"),
        ("duplicate", "forecast_keys"),
        ("absent", "forecast_keys"),
        ("text", "nonnumeric_forecast"),
    ],
)
def test_only_complete_pre_assignment_measurements_can_adjust_a_mean(defect, reason):
    rows, forecasts = _paired()
    options = {}
    if defect == "observations_only":
        forecasts = None
    elif defect == "late":
        options["schedule"] = SwitchbackObservationSchedule(
            assignments=rows.select("unit", "cycle", "period", "arm"), units=("a", "b"), cycles=2
        )
        rows = rows.with_columns(pl.lit(None).alias("sales"))
        forecasts = forecasts.with_columns(
            pl.when(pl.col("unit") == "b")
            .then(pl.col("assigned_at"))
            .otherwise(pl.col("forecast_at"))
            .alias("forecast_at")
        )
    elif defect == "infinite":
        forecasts = forecasts.with_columns(
            pl.when(pl.col("unit") == "a").then(None).otherwise(float("inf")).alias("forecast")
        )
    elif defect == "duplicate":
        forecasts = pl.concat([forecasts.slice(1), forecasts.tail(1)])
    elif defect == "absent":
        forecasts = forecasts.slice(1)
    else:
        forecasts = forecasts.with_columns(
            pl.when(pl.col("unit") == "a").then(None).otherwise(pl.lit("3")).alias("forecast")
        )
    with pytest.raises((CapabilityError, InvalidRequestError)) as caught:
        _build(rows, [_ADJUSTED], forecasts, **options)
    assert caught.value.context["reason"] == reason
    if defect == "observations_only":
        from tests.capability_consumers_scalar import SWITCHBACK_FORECAST

        assert (caught.value.code,) == SWITCHBACK_FORECAST.outcome(
            "missing-forecast"
        ).expected.codes


@pytest.mark.parametrize(
    "metric",
    [
        _RATIO.model_copy(update={"decision_method": _ADJUSTED.decision_method}),
        MetricSpec(name="visits", type="conversion", decision_method=_ADJUSTED.decision_method),
    ],
)
def test_cuped_cannot_silently_fit_an_additive_model_to_another_metric(metric):
    rows, forecasts = _paired()
    with pytest.raises(CapabilityError) as caught:
        _build(rows, [metric], forecasts)
    assert caught.value.code == "source.frame.switchback.metric"
    assert caught.value.context["reason"] == "cuped_requires_mean"


@pytest.mark.parametrize("metric,forecasts", [(_RATIO, False), (_ADJUSTED, True)])
def test_model_free_pilot_is_unavailable_for_ratio_and_adjusted_mean(metric, forecasts):
    rows, pre = _paired()
    experiment, reader = _build(
        rows, [metric], pre if forecasts else None, builder=from_switchback_frame
    )
    with pytest.raises(CapabilityError) as caught:
        planning_baseline(experiment, reader, metric.name)
    assert caught.value.context["reason"] == "planning_requires_outcome_model"


@pytest.mark.parametrize("metric", [_ADJUSTED, _RATIO])
def test_additive_reference_is_not_used_for_ratio_or_fitted_mean(metric):
    from tests.estimation.test_unit_cycle_envelope import envelope

    rows, forecasts = _paired()
    with pytest.raises(InvalidRequestError) as caught:
        _build(
            rows,
            [metric],
            forecasts if metric == _ADJUSTED else None,
            contrast_references={metric.name: envelope(metric=metric.name)},
        )
    assert caught.value.code == "unit_cycle.reference_mismatch"
    assert caught.value.context["reason"] == "unadjusted_reference_on_joint_metric"
