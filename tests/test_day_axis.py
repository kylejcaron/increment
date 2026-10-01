"""The day-axis quartet on ``Analysis``: metric selection, dimension
handling and coded refusals through the public entry points.
"""

import pytest

from increment import Analysis
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.engine import Method
from increment.frame import MetricSpec
from increment.semantics.models import AnalysisPlan


def _panel(*, metrics=None, breakouts=(), plan=None, exposure_date=None):
    """12-unit one-day panel: control on even rows, ``country='us'`` on even
    rows, and treatment shifted ``+2`` on ``y`` so a lift is well-defined.
    """
    import polars as pl

    rows = []
    for i in range(12):
        treatment = i % 2 == 1
        rows.append(
            {
                "unit": f"u{i}",
                "group": "treat" if treatment else "control",
                "ds": "2025-01-01",
                "y": float(i % 3) + (2.0 if treatment else 0.0),
                "w": float(i % 4),
                "country": "uk" if treatment else "us",
                "d7": float(i % 2),
            }
        )
    return Analysis.from_unit_panel(
        pl.DataFrame(rows),
        unit="unit",
        group="group",
        date="ds",
        control="control",
        metrics=metrics or {"y": "mean"},
        breakouts=list(breakouts),
        plan=plan,
        exposure_date=exposure_date,
    )


def test_run_asof_serves_activity_and_retention_metrics_together():
    """Grain ``asof`` keeps every requested metric in one pass: a bounded
    retention metric is neither split off nor refused.
    """
    analysis = _panel(
        metrics=[
            MetricSpec(name="y", type="mean"),
            MetricSpec(name="d7", type="retention", threshold_days=(0, 7)),
        ],
        exposure_date="ds",
    )
    results = analysis.run_asof()
    assert {row.metric for row in results} == {"y", "d7"}
    assert all(row.ds_basis == "calendar" for row in results)
    by_metric = {row.metric: row for row in results if row.group_id == "control"}
    assert by_metric["y"].value is not None
    assert by_metric["y"].value.value == pytest.approx(1.0)


def test_run_daily_refuses_unknown_dimension():
    analysis = _panel()
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily(dimension="missing")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["dimension"] == "missing"


def test_run_daily_refuses_no_definitions_on_unit_summary_source():
    import polars as pl

    analysis = Analysis.from_unit_summary(
        pl.DataFrame({"unit": ["a", "b"], "group": ["control", "treat"], "y": [1.0, 2.0]}),
        unit="unit",
        group="group",
        control="control",
        metrics={"y": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        analysis.run_daily()
    assert raised.value.code == "facade.analysis.no_definitions"


def test_run_daily_values_carry_slice_fields():
    results = _panel().run_daily()
    by_arm = {row.group_id: row for row in results}
    assert set(by_arm) == {"control", "treat"}
    for row in results:
        assert row.metric == "y"
        assert row.n == 6
        assert row.value is not None
    assert by_arm["control"].value.value == pytest.approx(1.0)
    assert by_arm["treat"].value.value == pytest.approx(3.0)
    assert all(row.dimension_value is None for row in results)


def test_run_daily_lift_narrows_to_the_requested_metric_and_carries_plan_alpha():
    analysis = _panel(metrics={"y": "mean", "w": "mean"}, plan=AnalysisPlan(alpha=0.1))
    narrowed = analysis.run_daily_lift(metrics=["y"])
    assert narrowed
    assert {row.metric for row in narrowed} == {"y"}
    (row,) = narrowed
    assert row.lift is not None
    assert row.lift.alpha == pytest.approx(0.1)
    assert row.lift.level == pytest.approx(0.9)
    assert row.lift.value == pytest.approx(2.0)


def test_run_daily_honors_dimension_on_the_moments_route():
    """The moments route must honor ``dimension``, not silently emit
    undimensioned rows.
    """
    analysis = _panel(breakouts=("country",))
    dimensioned = analysis.run_daily(dimension="country")

    by_cell = {(row.group_id, row.dimension_value): row for row in dimensioned}
    assert set(by_cell) == {("control", "us"), ("treat", "uk")}
    assert all(row.dimension == "country" for row in dimensioned)
    for row in dimensioned:
        assert row.value is not None
    assert by_cell[("control", "us")].value.value == pytest.approx(1.0)
    assert by_cell[("treat", "uk")].value.value == pytest.approx(3.0)
    assert all(row.n == 6 for row in dimensioned)

    undimensioned = analysis.run_daily()
    assert undimensioned
    assert all(row.dimension_value is None for row in undimensioned)


def test_run_daily_lift_refuses_a_config_bound_observational_method():
    """An observational adjustment method bound in a metric's own config,
    with no call-time override, still refuses on the lift path.
    """
    analysis = _panel(
        metrics=[MetricSpec(name="y", type="mean", decision_method=Method(name="aipw"))]
    )
    with pytest.raises(InvalidRequestError) as raised:
        analysis.run_daily_lift()
    assert raised.value.code == "estimation.engine.method_name_observational"


@pytest.mark.parametrize("readout", ["run_daily_lift", "run_asof_lift"])
@pytest.mark.parametrize("metric_count, explicit_decision", [(1, False), (2, False), (1, True)])
def test_native_observational_lift_uses_the_day_axis_refusal(
    con, readout, metric_count, explicit_decision
):
    from increment.errors import UnsupportedRequestError
    from increment.semantics.design import AdjustmentSet, Observational
    from increment.semantics.models import Metric, RetentionMetric
    from tests.analysis_factory import make_analysis_like

    design = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("country",))
    )
    with Analysis.from_definitions("new_onboarding_v2", "examples/definitions/", con) as native:
        metrics: list[Metric] = [
            metric for metric in native.metrics if not isinstance(metric, RetentionMetric)
        ]
        with make_analysis_like(
            native, metrics=metrics[:metric_count], design=design, plan=AnalysisPlan()
        ) as analysis:
            call = getattr(analysis, readout)
            with pytest.raises(UnsupportedRequestError) as raised:
                if explicit_decision:
                    call(decision_method=Method(name="unadjusted"))
                else:
                    call()
    assert raised.value.code == "facade.analysis.observational_day_axis"
    assert raised.value.context["method"] == readout
