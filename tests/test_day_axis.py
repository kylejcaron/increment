"""The day-axis views: metric selection, dimension handling and coded
refusals through the public entry points.
"""

from contextlib import AbstractContextManager
from typing import Any, cast

import pytest

from increment import Scope, from_definitions, from_frame, workflows
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.engine import Method
from increment.evidence import BreakoutIdentity
from increment.frame import MetricSpec
from increment.semantics.models import AnalysisPlan
from tests.analysis_factory import bind_pair, make_binding_like, view_readout


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
    return from_frame(
        pl.DataFrame(rows),
        shape="panel",
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
    experiment, resolver = _panel(
        metrics=[
            MetricSpec(name="y", type="mean"),
            MetricSpec(name="d7", type="retention", threshold_days=(0, 7)),
        ],
        exposure_date="ds",
    )
    results = view_readout(experiment, resolver, Scope(view="asof", grain="asof", values=True))
    assert {row.metric for row in results} == {"y", "d7"}
    assert all(row.ds_basis == "calendar" for row in results)
    by_metric = {row.metric: row for row in results if row.group_id == "control"}
    assert by_metric["y"].value is not None
    assert by_metric["y"].value.value == pytest.approx(1.0)


def test_run_daily_refuses_unknown_dimension():
    from tests.analysis_factory import bound_analysis

    _, resolver = _panel()
    with pytest.raises(InvalidRequestError) as raised:
        bound_analysis(resolver).run_daily(dimension="missing")
    assert raised.value.code == "facade.analysis.unknown_dimension"
    assert raised.value.context["dimension"] == "missing"


def test_run_daily_refuses_no_definitions_on_unit_summary_source():
    import polars as pl

    experiment, resolver = from_frame(
        pl.DataFrame({"unit": ["a", "b"], "group": ["control", "treat"], "y": [1.0, 2.0]}),
        shape="summary",
        unit="unit",
        group="group",
        control="control",
        metrics={"y": "mean"},
    )
    with pytest.raises(CapabilityError) as raised:
        view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True))
    assert raised.value.code == "evidence.grain_unavailable"


def test_run_daily_values_carry_slice_fields():
    experiment, resolver = _panel()
    results = view_readout(experiment, resolver, Scope(view="daily", grain="daily", values=True))
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
    experiment, resolver = _panel(metrics={"y": "mean", "w": "mean"}, plan=AnalysisPlan(alpha=0.1))
    narrowed = view_readout(experiment, resolver, Scope(view="daily", grain="daily"), metrics=["y"])
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
    experiment, resolver = _panel(breakouts=("country",))
    dimensioned = view_readout(
        experiment,
        resolver,
        Scope(view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)),
    )

    by_cell = {(row.group_id, row.dimension_value): row for row in dimensioned}
    assert set(by_cell) == {("control", "us"), ("treat", "uk")}
    assert all(row.dimension == "country" for row in dimensioned)
    for row in dimensioned:
        assert row.value is not None
    assert by_cell[("control", "us")].value.value == pytest.approx(1.0)
    assert by_cell[("treat", "uk")].value.value == pytest.approx(3.0)
    assert all(row.n == 6 for row in dimensioned)

    undimensioned = view_readout(
        experiment, resolver, Scope(view="daily", grain="daily", values=True)
    )
    assert undimensioned
    assert all(row.dimension_value is None for row in undimensioned)


def test_run_daily_lift_refuses_a_config_bound_observational_method():
    """An observational adjustment method bound in a metric's own config,
    with no call-time override, still refuses on the lift path.
    """
    experiment, resolver = _panel(
        metrics=[MetricSpec(name="y", type="mean", decision_method=Method(name="aipw"))]
    )
    with pytest.raises(InvalidRequestError) as raised:
        view_readout(experiment, resolver, Scope(view="daily", grain="daily"))
    assert raised.value.code == "estimation.engine.method_name_observational"


@pytest.mark.integration
@pytest.mark.parametrize("readout", ["run_daily_lift", "run_asof_lift"])
@pytest.mark.parametrize("metric_count, explicit_decision", [(1, False), (2, False), (1, True)])
def test_native_observational_lift_uses_the_day_axis_refusal(
    con, example_definitions, readout, metric_count, explicit_decision
):
    from increment.errors import UnsupportedRequestError
    from increment.semantics.design import AdjustmentSet, Observational
    from increment.semantics.models import Metric, RetentionMetric

    design = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("country",))
    )
    scope = {
        "run_daily_lift": Scope(view="daily", grain="daily"),
        "run_asof_lift": Scope(view="asof", grain="asof"),
    }[readout]
    options = {"decision_method": Method(name="unadjusted")} if explicit_decision else {}
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="new_onboarding_v2"
    )
    assert isinstance(resolver, AbstractContextManager)
    with resolver:
        metrics: list[Metric] = [
            metric
            for metric in workflows.metrics(experiment, resolver)
            if not isinstance(metric, RetentionMetric)
        ]
        observational, observational_resolver = make_binding_like(
            resolver, metrics=metrics[:metric_count], design=design, plan=AnalysisPlan()
        )
        with observational_resolver, pytest.raises(UnsupportedRequestError) as raised:
            view_readout(observational, observational_resolver, scope, **options)
    assert raised.value.code == "facade.analysis.observational_day_axis"
    assert raised.value.context["method"] == readout


def test_custom_provider_day_axis_uses_declared_breakouts() -> None:
    from dataclasses import replace

    from increment.evidence import evidence_provider
    from tests.analysis_factory import held_source

    _, original = _panel(breakouts=("country",))
    source = cast(Any, held_source(original))

    class Provider:
        def __init__(self) -> None:
            self.context = source.context
            self.evidence = replace(source.evidence, kind="moments", shape=None)
            self.operations = source.operations
            self.capabilities = source.capabilities

        @property
        def breakouts(self):
            raise AttributeError("custom providers declare breakouts in evidence")

        def __getattr__(self, name):
            if name in {"breakouts", "shape"}:
                raise AttributeError(name)
            return getattr(source, name)

        def moments(self, request):
            return evidence_provider(source).moments(request)

        def unit_rows(self, request):
            return evidence_provider(source).unit_rows(request)

        def method_fit(self, request):
            return evidence_provider(source).method_fit(request)

        def breakout_method_moments(self, request):
            return evidence_provider(source).breakout_method_moments(request)

        def compliance(self, request):
            return evidence_provider(source).compliance(request)

        def compliance_series(self, request):
            return evidence_provider(source).compliance_series(request)

        def counts(self, request):
            return evidence_provider(source).counts(request)

        def capture_sequential(self, request):
            return evidence_provider(source).capture_sequential(request)

        def scope(self, request):
            from contextlib import contextmanager

            from increment.evidence import EvidenceScopeResult

            @contextmanager
            def scoped():
                yield self

            return EvidenceScopeResult(scoped(), request)

        def close(self):
            return None

    provider = Provider()
    assert not hasattr(provider, "shape")
    experiment, resolver = bind_pair(cast(Any, provider))
    try:
        results = view_readout(
            experiment,
            resolver,
            Scope(
                view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)
            ),
        )
        assert results
        assert {row.dimension_value for row in results} == {"uk", "us"}
        asof_results = view_readout(
            experiment, resolver, Scope(view="asof", grain="asof", values=True)
        )
        assert asof_results
        assert {row.metric for row in asof_results} == {"y"}
    finally:
        resolver.close()
        original.close()
