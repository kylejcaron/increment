"""The day-axis views: metric selection, dimension handling and coded
refusals through the public entry points.
"""

from contextlib import AbstractContextManager
from typing import Any, cast

import pytest

from increment import View, from_definitions, from_frame, workflows
from increment.errors import CapabilityError, InvalidRequestError
from increment.estimation.engine import Method
from increment.evidence import BreakoutIdentity
from increment.frame import MetricSpec
from increment.semantics.models import AnalysisPlan
from tests.analysis_factory import bind_pair, make_binding_like, view_readout


def _panel(*, metrics=None, breakouts=(), plan=None):
    """12-unit one-day panel, every unit exposed on that day: control on even
    rows, ``country='us'`` on even rows, and treatment shifted ``+2`` on ``y``
    so a lift is well-defined.
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
        exposure_date="ds",
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
    )
    results = view_readout(experiment, resolver, View(view="asof", grain="asof", values=True))
    assert {row.metric for row in results} == {"y", "d7"}
    assert all(row.ds_basis == "calendar" for row in results)
    by_metric = {row.metric: row for row in results if row.group_id == "control"}
    assert by_metric["y"].value is not None
    assert by_metric["y"].value.value == pytest.approx(1.0)


def test_run_daily_refuses_unknown_dimension():
    experiment, resolver = _panel()
    with pytest.raises(InvalidRequestError) as raised:
        view_readout(
            experiment,
            resolver,
            View(
                view="daily",
                grain="daily",
                values=True,
                segment=(BreakoutIdentity("missing", None),),
            ),
        )
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
        view_readout(experiment, resolver, View(view="daily", grain="daily", values=True))
    assert raised.value.code == "evidence.grain_unavailable"
    assert raised.value.context == {
        "constructor": "FrameTotalsSource",
        "requested": "daily",
        "offered": ("total",),
        "route": "use a source with the requested day axis",
    }


def test_day_axis_dimension_declaration_preserves_legacy_route_order():
    from dataclasses import replace
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from increment._day_axis import validate_day_axis
    from increment._evidence_types import DayAxisRequest
    from increment.errors import InvalidRequestError
    from increment.evidence import EvidenceCapabilities
    from increment.readouts._evidence import (
        declared_day_axis_dimensions,
        declared_day_axis_route,
    )
    from increment.semantics.models import AnalysisPlan, Breakout
    from increment.semantics.models import Experiment as Definition

    experiment = Definition(
        name="checkout",
        exposure="assignment",
        unit="unit",
        start=datetime(2025, 1, 1, tzinfo=UTC),
        control_group="control",
        plan=AnalysisPlan(),
        breakouts=(
            Breakout(property="platform"),
            Breakout(property="country"),
            Breakout(property="platform"),
        ),
    )
    common: dict[str, Any] = {
        "grains": frozenset({"total", "daily", "asof"}),
        "shape": None,
        "unit_rows": False,
        "raw_outcomes": False,
        "method_fit": "none",
        "compliance_axis": "none",
        "count_grains": frozenset(),
        "populations": frozenset({"assigned"}),
        "sequential_capture": "none",
        "scalar": None,
        "available_covariates": None,
    }
    native = EvidenceCapabilities(
        kind="definitions",
        breakouts=("platform", "country", "platform"),
        operations=frozenset({"moments_source"}),
        **common,
    )
    moments = EvidenceCapabilities(
        kind="moments",
        breakouts=("platform", "country", "platform"),
        operations=frozenset(),
        **common,
    )

    artifact = replace(native, operations=frozenset({"artifact_experiment"}))
    assert declared_day_axis_dimensions(native, None) == ()
    assert declared_day_axis_dimensions(artifact, None) == ()
    for capabilities, expected_dimensions in (
        (native, ("country", "platform")),
        (artifact, ("country", "platform")),
        (moments, ("platform", "country", "platform")),
    ):
        source = SimpleNamespace(
            evidence=capabilities,
            context=SimpleNamespace(trigger_name=None, metrics=()),
        )
        assert declared_day_axis_dimensions(capabilities, experiment) == expected_dimensions
        request = DayAxisRequest(
            caller="run_daily",
            grain="daily",
            metrics=(),
            dimension="unavailable",
            declared_dimensions=declared_day_axis_dimensions(capabilities, experiment),
        )
        with pytest.raises(InvalidRequestError) as raised:
            validate_day_axis(
                request,
                src=source,
                design=None,
                plan=cast(Any, SimpleNamespace(inference=None)),
                route=declared_day_axis_route(capabilities),
                experiment=experiment,
            )
        assert raised.value.code == "facade.analysis.unknown_dimension"
        assert raised.value.context == {
            "method": "run_daily",
            "dimension": "unavailable",
            "declared": tuple(expected_dimensions),
        }


def test_custom_day_source_declares_acquirable_day_grains():
    from increment._evidence_loaders import day_axis_grains
    from increment._source_types import EvidenceCapabilities

    class Provider:
        evidence = EvidenceCapabilities(
            kind="moments",
            grains=frozenset({"total"}),
            breakouts=(),
            shape=None,
            operations=frozenset({"moments_source", "day_source"}),
            unit_rows=False,
            raw_outcomes=False,
            method_fit="none",
            compliance_axis="none",
            count_grains=frozenset(),
            populations=frozenset({"assigned"}),
            sequential_capture="none",
            scalar=None,
            available_covariates=None,
        )

    assert day_axis_grains(Provider()) == frozenset({"total", "daily", "asof"})


def test_run_daily_values_carry_slice_fields():
    experiment, resolver = _panel()
    results = view_readout(experiment, resolver, View(view="daily", grain="daily", values=True))
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
    narrowed = view_readout(experiment, resolver, View(view="daily", grain="daily"), metrics=["y"])
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
        View(view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)),
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
        experiment, resolver, View(view="daily", grain="daily", values=True)
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
        view_readout(experiment, resolver, View(view="daily", grain="daily"))
    assert raised.value.code == "estimation.engine.method_name_observational"


@pytest.mark.integration
@pytest.mark.parametrize("readout", ["run_daily_lift", "run_asof_lift"])
@pytest.mark.parametrize("metric_count, explicit_decision", [(1, False), (2, False), (1, True)])
def test_native_observational_lift_uses_the_day_axis_refusal(
    con, example_definitions, readout, metric_count, explicit_decision, monkeypatch
):
    from increment.errors import UnsupportedRequestError
    from increment.semantics.design import AdjustmentSet, Observational
    from increment.semantics.models import Metric, MultiplicitySpec, RetentionMetric

    design = Observational(
        control_group="control", adjustment=AdjustmentSet(covariates=("country",))
    )
    scope = {
        "run_daily_lift": View(view="daily", grain="daily"),
        "run_asof_lift": View(
            view="asof", grain="asof", segment=BreakoutIdentity("unknown_dimension", None)
        ),
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
        if readout == "run_daily_lift" and metric_count == 1 and not explicit_decision:
            from increment._evidence_loaders import day_axis_grains
            from increment._evidence_types import DayAxisRequest
            from increment.readouts._evidence import load_day_axis_rows

            source = cast(Any, resolver._binding.state.source)
            original_grains = source.capabilities
            source.capabilities = frozenset({"total"})
            try:
                assert day_axis_grains(source) == frozenset({"total", "daily", "asof"})
                request = DayAxisRequest(
                    caller="run_daily_lift",
                    grain="daily",
                    metrics=(metrics[0],),
                    dimension=None,
                )
                rows = load_day_axis_rows(source, request)
                assert next(iter(rows.slices)).rows
            finally:
                source.capabilities = original_grains
        observational, observational_resolver = make_binding_like(
            resolver,
            metrics=metrics[:metric_count],
            design=design,
            plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="bh")),
        )
        if readout == "run_asof_lift":
            from dataclasses import replace
            from types import SimpleNamespace

            from increment._day_axis import validate_day_axis
            from increment._evidence_types import DayAxisRequest

            capabilities = experiment.source_refs[0].capabilities
            assert capabilities is not None
            source = SimpleNamespace(
                evidence=capabilities,
                context=SimpleNamespace(trigger_name=None, metrics=tuple(metrics)),
            )
            request = DayAxisRequest(
                caller="run_asof_lift",
                grain="asof",
                metrics=tuple(metrics[:metric_count]),
                dimension="unknown_dimension",
            )
            trigger_request = replace(request, population="triggered")
            trigger_experiment = SimpleNamespace(
                name="recorded",
                trigger=None,
                cluster="store_id",
                breakouts=(SimpleNamespace(property="country", source=None),),
            )
            with pytest.raises(CapabilityError) as missing_trigger:
                validate_day_axis(
                    trigger_request,
                    src=source,
                    design=design,
                    plan=cast(Any, SimpleNamespace(inference=None)),
                    route="native",
                    experiment=cast(Any, trigger_experiment),
                )
            assert missing_trigger.value.code == "facade.analysis.trigger_unsupported"
            assert missing_trigger.value.context == {
                "method": "run_asof_lift",
                "experiment": "recorded",
                "trigger": None,
                "route": "native",
                "supported_sources": (
                    "from_definitions",
                    "from_unit_day_artifact",
                    "from_moments",
                ),
            }
            with pytest.raises(UnsupportedRequestError) as direct:
                validate_day_axis(
                    request,
                    src=source,
                    design=design,
                    plan=cast(
                        Any,
                        SimpleNamespace(
                            inference=None,
                            view_policies=SimpleNamespace(
                                for_view=lambda *_args, **_kwargs: SimpleNamespace(correction="bh")
                            ),
                        ),
                    ),
                    route="native",
                    experiment=None,
                )
            assert direct.value.context == {"method": "run_asof_lift"}
            assert direct.value.code == "facade.analysis.observational_day_axis"
        with observational_resolver, pytest.raises(UnsupportedRequestError) as raised:

            def fail(*args, **kwargs):
                raise AssertionError("static day-axis refusal performed a warehouse read")

            monkeypatch.setattr(con, "raw_sql", fail)
            monkeypatch.setattr(con, "execute", fail)
            monkeypatch.setattr(con, "to_pyarrow", fail)
            view_readout(observational, observational_resolver, scope, **options)
    assert raised.value.context == {"method": readout}
    assert raised.value.code == "facade.analysis.observational_day_axis"


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
            View(
                view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)
            ),
        )
        assert results
        assert {row.dimension_value for row in results} == {"uk", "us"}
        asof_results = view_readout(
            experiment, resolver, View(view="asof", grain="asof", values=True)
        )
        assert asof_results
        assert {row.metric for row in asof_results} == {"y"}
    finally:
        resolver.close()
        original.close()
