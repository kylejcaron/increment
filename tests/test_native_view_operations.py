"""Parity and capability tests for source-owned native view workflows."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, cast

import pytest

from increment import Scope, from_definitions
from increment.errors import CapabilityError
from increment.estimation.sitewide import SitewideImpact
from increment.evidence import BreakoutIdentity
from increment.semantics.models import (
    AnalysisPlan,
    Breakout,
    Measure,
    MultiplicitySpec,
    QuantileMetric,
    Winsorization,
)
from increment.workflows import (
    breakout_summaries,
    experiment_definition,
    factor_summaries,
    metrics,
)
from tests._evidence_requests import _breakout_moments, _moments
from tests._unit_boundary import backend_fixture
from tests.analysis_factory import (
    bound_source,
    make_binding_like,
    view_readout,
)
from tests.query.test_builders import _analysis_with_cross_source_breakouts
from tests.test_analysis_breakout import _analysis_with_country_breakout
from tests.test_analysis_sitewide import _REVENUE_PER_USER, _pricing_experiment
from tests.test_native_core_operations import _assert_operation_refusal


@backend_fixture(scope="module")
def pricing_con():
    import ibis

    from tests.analysis_factory import _make_event_log_table

    con = ibis.duckdb.connect()
    _make_event_log_table(con)
    return con


_PARITY_FLOAT_REL = 1e-12
_PARITY_FLOAT_ABS = 2.0e-10


def _values_match_with_float_tolerance(actual, expected) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return actual == pytest.approx(
            expected,
            rel=_PARITY_FLOAT_REL,
            abs=_PARITY_FLOAT_ABS,
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _values_match_with_float_tolerance(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _values_match_with_float_tolerance(a, e) for a, e in zip(actual, expected, strict=True)
        )
    return actual == expected


def _assert_approx_row_sets(actual_rows, expected_rows) -> None:
    assert len(actual_rows) == len(expected_rows)
    unmatched = list(expected_rows)
    for actual in actual_rows:
        match = next(
            (
                index
                for index, expected in enumerate(unmatched)
                if _values_match_with_float_tolerance(actual, expected)
            ),
            None,
        )
        assert match is not None, f"unexpected parity row: {actual!r}"
        unmatched.pop(match)
    assert not unmatched, f"missing parity rows: {unmatched!r}"


def _assert_model_rows_match(actual_rows, expected_rows, *, ignore_source_scope=False) -> None:
    actual = [row.model_dump(mode="json") for row in actual_rows]
    expected = [row.model_dump(mode="json") for row in expected_rows]
    if ignore_source_scope:
        scope_fields = {
            "source_snapshot_id",
            "metric_type",
            "decision_scope_complete",
            "decision_scope_reason_code",
            "decision_scope_reason_context",
        }
        # Low-level day.moments -> run_daily results have no readout request or source digest;
        # compare their outcome fields with the scoped facade, not the facade-only projection.
        for rows in (actual, expected):
            for row in rows:
                for field in scope_fields:
                    row.pop(field, None)
    _assert_approx_row_sets(actual, expected)


@pytest.fixture
def pricing_analysis(pricing_con, example_definitions):
    return from_definitions(example_definitions, pricing_con, experiment_name="pricing_tier_test")


def test_sitewide_impact_is_typed_and_reports_iid_arm_attribution(tmp_path, pricing_con):
    from increment.estimators import sitewide

    pricing = _pricing_experiment(tmp_path, pricing_con, [_REVENUE_PER_USER])

    result = cast(SitewideImpact, sitewide(*pricing, "revenue_per_user"))

    assert result.treatment_group == "treatment"
    assert result.other_arm_ids == ()
    assert result.n_clusters is None
    assert result.n_control == 20
    assert result.n_treatment == 20
    # Hand-computed from the fixture: control 10 converters x $20 = $200,
    # treatment 14 converters x $40 = $560.
    assert result.site_total_volume == pytest.approx(760.0)
    assert result.delta == pytest.approx(18.0)


def test_sitewide_evidence_refuses_ratio_before_raw_work(pricing_analysis: tuple[Any, Any]):
    """The named (non-ratio) evidence contract refuses a ratio metric.

    ``sitewide`` requests the ratio-specific reducer instead, so the
    default flag this refusal guards is only reachable at the source.
    """
    metric = next(m for m in metrics(*pricing_analysis) if m.name == "revenue_per_session")
    source = bound_source(pricing_analysis[1])
    with pytest.raises(CapabilityError) as raised:
        source.sitewide_evidence(metric)
    assert raised.value.code == "evidence.operation_unavailable"


def test_sitewide_evidence_refuses_winsorization_before_raw_work(
    pricing_analysis: tuple[Any, Any], monkeypatch
):
    metric = next(m for m in metrics(*pricing_analysis) if m.name == "purchase_rate")
    winsorized = metric.model_copy(update={"winsorization": Winsorization(upper_value=1.0)})
    source = bound_source(pricing_analysis[1])
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse work started")),
    )
    with pytest.raises(CapabilityError) as raised:
        source.sitewide_evidence(winsorized)
    assert raised.value.code == "evidence.operation_unavailable"
    assert raised.value.context["request"] == {
        "metrics": (
            {
                "metric": "purchase_rate",
                "threshold_source": "fixed_value",
                "thresholds": {"upper": {"source": "fixed_value", "value": 1.0}},
            },
        )
    }


@pytest.mark.parametrize(
    "winsorization",
    [Winsorization(upper_percentile=0.9)],
)
def test_breakout_summaries_refuses_winsorization_before_warehouse_work(
    pricing_con, monkeypatch, winsorization
):
    experiment, resolver = _analysis_with_country_breakout(pricing_con)
    source = bound_source(resolver)
    metric = metrics(experiment, resolver)[0].model_copy(update={"winsorization": winsorization})
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("reduction started")),
    )
    monkeypatch.setattr(
        source,
        "_resolve_breakout_props",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("properties built")),
    )
    monkeypatch.setattr(
        source,
        "_get_exposures",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse queried")),
    )

    with pytest.raises(CapabilityError) as raised:
        source.breakout_summaries(metrics=(metric,))

    assert raised.value.code == "evidence.operation_unavailable"


def test_breakout_summaries_preserves_fixed_value_winsorization(pricing_con):
    experiment, resolver = _analysis_with_country_breakout(pricing_con)
    source = bound_source(resolver)
    metric = metrics(experiment, resolver)[0].model_copy(
        update={"winsorization": Winsorization(upper_value=15.0)}
    )

    result = source.breakout_summaries(metrics=(metric,))
    summary = next(iter(result.values()))["group_summary"]

    assert summary.num_rows
    assert set(summary["winsor_upper_bound"].to_pylist()) == {15.0}


@pytest.mark.slow
def test_breakout_summaries_keep_multi_source_keys(con):
    experiment, resolver = _analysis_with_cross_source_breakouts(con)

    result = breakout_summaries(experiment, resolver)

    assert set(result) == {
        "visit_rate:country:src_a",
        "visit_rate:country:src_b",
        "checkout:country:src_a",
        "checkout:country:src_b",
    }


def test_run_breakout_preserves_all_declared_sources(con):
    analysis = _analysis_with_cross_source_breakouts(con)
    analysis = make_binding_like(
        analysis[1], plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none"))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = view_readout(*analysis, Scope(view="breakout", grain="total"))

    assert {result.source for result in results} == {"src_a", "src_b"}


@pytest.mark.slow
def test_day_source_matches_daily_and_asof_views(pricing_analysis: tuple[Any, Any]):
    from increment.breakout.estimates import DailyMetricValues
    from increment.breakout.estimates import run_daily as _run_daily_values

    metric = next(m for m in metrics(*pricing_analysis) if m.name == "purchase_rate")
    source = bound_source(pricing_analysis[1])
    day = source.day_source(metrics=(metric,))

    daily_rows = _moments(day, metric, grain="daily")
    asof_rows = _moments(day, metric, grain="asof")
    direct_daily = DailyMetricValues(_run_daily_values(daily_rows, metrics=[metric], view="daily"))
    direct_asof = DailyMetricValues(_run_daily_values(asof_rows, metrics=[metric], view="asof"))
    facade_daily = view_readout(
        *pricing_analysis, Scope(view="daily", grain="daily", values=True), metrics=(metric,)
    )
    facade_asof = view_readout(
        *pricing_analysis, Scope(view="asof", grain="asof", values=True), metrics=(metric,)
    )
    _assert_model_rows_match(direct_daily, facade_daily, ignore_source_scope=True)
    _assert_model_rows_match(direct_asof, facade_asof, ignore_source_scope=True)


def test_day_source_dimensioned_result_is_typed_and_attributed(pricing_con):
    experiment, resolver = _analysis_with_country_breakout(pricing_con)
    metric = metrics(experiment, resolver)[0]

    values = view_readout(
        experiment,
        resolver,
        Scope(view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)),
        metrics=(metric,),
    )

    assert {row.metric for row in values} == {metric.name}
    assert {row.dimension for row in values} == {"country"}
    assert {row.dimension_value for row in values} == {"US", "CA"}
    assert {row.source for row in values} == {"events"}


@pytest.mark.slow
def test_day_source_adhoc_ratio_denominator_extends_daily_and_asof_horizon(example_definitions):
    import ibis

    from tests.analysis_factory import _make_event_log_table

    con = ibis.duckdb.connect()
    late_day = datetime(2025, 4, 1, 9)
    _make_event_log_table(
        con,
        extra_rows=[
            {
                "event_at": late_day,
                "user_id": "cp00",
                "session_id": "late-session",
                "event": "session_start",
                "experiment_id": None,
                "group_id": None,
                "revenue": None,
                "duration_s": None,
                "country_code": "US",
                "device_type": "web",
                "plan": "free",
            }
        ],
    )
    experiment, resolver = from_definitions(
        example_definitions, con, experiment_name="pricing_tier_test"
    )
    declared = next(m for m in metrics(experiment, resolver) if m.name == "revenue_per_session")
    assert hasattr(declared, "denominator")
    adhoc = declared.model_copy(
        update={
            "name": "revenue_per_session_start",
            "numerator": Measure(fact="purchase", aggregation="sum", window_days=30),
            "denominator": Measure(fact="session_start", aggregation="count", window_days=30),
        }
    )

    declared_daily = view_readout(
        experiment, resolver, Scope(view="daily", grain="daily", values=True), metrics=(declared,)
    )
    daily = view_readout(
        experiment, resolver, Scope(view="daily", grain="daily", values=True), metrics=(adhoc,)
    )
    asof = view_readout(
        experiment, resolver, Scope(view="asof", grain="asof", values=True), metrics=(adhoc,)
    )

    declared_horizon = max(row.ds for row in declared_daily)
    assert max(row.ds for row in daily) > declared_horizon
    assert max(row.ds for row in asof) > declared_horizon
    assert any(row.ds == late_day.date() for row in daily)
    assert any(row.ds == late_day.date() for row in asof)


def test_day_source_validates_missing_breakout_before_reduction_or_properties(
    pricing_con, monkeypatch
):
    experiment, resolver = _analysis_with_country_breakout(pricing_con)
    metric = metrics(experiment, resolver)[0]
    source = bound_source(resolver)
    day = source.day_source(metrics=(metric,))
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("reduction started")),
    )
    monkeypatch.setattr(
        source,
        "_build_breakout_properties_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("properties built")),
    )
    with pytest.raises(CapabilityError) as raised:
        _breakout_moments(day, metric, Breakout(property="missing_dimension"))
    assert raised.value.code == "evidence.breakout_unavailable"


def test_day_source_validates_breakout_source_before_reduction(pricing_con, monkeypatch):
    experiment, resolver = _analysis_with_country_breakout(pricing_con)
    metric = metrics(experiment, resolver)[0]
    source = bound_source(resolver)
    day = source.day_source(metrics=(metric,))
    breakout = experiment_definition(experiment, resolver).breakouts[0]
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("reduction started")),
    )

    with pytest.raises(CapabilityError) as raised:
        day.breakout_moments(
            metric,
            breakout.model_copy(update={"source": "not_declared"}),
        )
    assert raised.value.code == "evidence.breakout_unavailable"


def test_day_source_rejects_existing_undeclared_source_before_warehouse_work(con, monkeypatch):
    analysis = _analysis_with_cross_source_breakouts(con)
    declared = experiment_definition(*analysis).breakouts[0]
    analysis = make_binding_like(
        analysis[1],
        experiment=experiment_definition(*analysis).model_copy(update={"breakouts": (declared,)}),
    )
    source = bound_source(analysis[1])
    metric = metrics(*analysis)[0]
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("reduction started")),
    )
    monkeypatch.setattr(
        source,
        "_get_exposures",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse queried")),
    )
    monkeypatch.setattr(
        source,
        "_build_breakout_properties_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("properties built")),
    )
    day = source.day_source(metrics=(metric,))

    with pytest.raises(CapabilityError) as raised:
        day.breakout_moments(
            metric,
            Breakout(property="country", source="src_b"),
        )

    assert raised.value.code == "evidence.breakout_unavailable"


def test_breakout_source_rejects_existing_undeclared_source_before_warehouse_work(con, monkeypatch):
    analysis = _analysis_with_cross_source_breakouts(con)
    declared = experiment_definition(*analysis).breakouts[0]
    analysis = make_binding_like(
        analysis[1],
        experiment=experiment_definition(*analysis).model_copy(update={"breakouts": (declared,)}),
    )
    source = bound_source(analysis[1])
    metric = metrics(*analysis)[0]
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("reduction started")),
    )
    monkeypatch.setattr(
        source,
        "_get_exposures",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse queried")),
    )
    monkeypatch.setattr(
        source,
        "_build_breakout_properties_table",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("properties built")),
    )

    with pytest.raises(CapabilityError) as raised:
        source.breakout_source(
            Breakout(property="country", source="src_b"),
            metrics=(metric,),
        )

    assert raised.value.code == "evidence.breakout_unavailable"
    assert raised.value.context["requested"] == ("country",)
    assert raised.value.context["offered"] == ("src_a",)
    assert raised.value.context["route"]


def _with_breakout_source(analysis: tuple[Any, Any], source: str | None) -> Any:
    """The same analysis with every declared breakout's source set explicitly."""
    return make_binding_like(
        analysis[1],
        experiment=experiment_definition(*analysis).model_copy(
            update={
                "breakouts": tuple(
                    breakout.model_copy(update={"source": source})
                    for breakout in experiment_definition(*analysis).breakouts
                )
            }
        ),
    )


def test_breakout_source_treats_inferred_and_explicit_breakout_sources_as_equivalent(
    pricing_con,
):
    analysis = _analysis_with_country_breakout(pricing_con)
    inferred_experiment, inferred_resolver = _with_breakout_source(analysis, None)
    explicit_experiment, explicit_resolver = _with_breakout_source(analysis, "events")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        inferred = view_readout(
            inferred_experiment, inferred_resolver, Scope(view="breakout", grain="total")
        )
        explicit = view_readout(
            explicit_experiment, explicit_resolver, Scope(view="breakout", grain="total")
        )

    assert {result.source for result in inferred} == {"events"}
    assert {result.source for result in explicit} == {"events"}
    _assert_model_rows_match(inferred, explicit)


def test_day_source_treats_inferred_and_explicit_breakout_sources_as_equivalent(
    pricing_con,
):
    analysis = _analysis_with_country_breakout(pricing_con)
    metric = metrics(*analysis)[0]
    inferred_experiment, inferred_resolver = _with_breakout_source(analysis, None)
    explicit_experiment, explicit_resolver = _with_breakout_source(analysis, "events")

    inferred = view_readout(
        inferred_experiment,
        inferred_resolver,
        Scope(view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)),
        metrics=(metric,),
    )
    explicit = view_readout(
        explicit_experiment,
        explicit_resolver,
        Scope(view="daily", grain="daily", values=True, segment=BreakoutIdentity("country", None)),
        metrics=(metric,),
    )

    assert {row.source for row in inferred} == {"events"}
    assert {row.source for row in explicit} == {"events"}
    _assert_model_rows_match(inferred, explicit)


def test_day_source_preserves_retention_cohort_path(pricing_analysis: tuple[Any, Any]):
    metric = next(m for m in metrics(*pricing_analysis) if m.name == "d7_retention")
    source = bound_source(pricing_analysis[1])
    rows = _moments(source.day_source(metrics=(metric,)), metric, grain="daily")
    assert rows
    assert {row["metric"] for row in rows} == {metric.name}
    assert {row["ds"] for row in rows} == {rows[0]["ds"]}


def test_day_preflight_uptake_cluster_refusal_is_coded_before_builder(con, monkeypatch):
    from tests.test_analysis_asof import _analysis_with_asof_encouragement_events

    analysis_experiment, analysis_resolver = _analysis_with_asof_encouragement_events(con)
    experiment = experiment_definition(analysis_experiment, analysis_resolver).model_copy(
        update={"cluster": "store_id"}
    )
    clustered_experiment, clustered_resolver = make_binding_like(
        analysis_resolver, experiment=experiment
    )
    source = bound_source(clustered_resolver)
    metric = metrics(clustered_experiment, clustered_resolver)[0]

    def fail(*args, **kwargs):
        raise AssertionError("clustered encouragement day request touched warehouse")

    monkeypatch.setattr(source, "_get_exposures", fail)
    monkeypatch.setattr(source, "_note_reduction", fail)

    with pytest.raises(CapabilityError) as raised:
        _moments(source.day_source(metrics=(metric,)), metric, grain="asof", include_covariate=True)
    _assert_operation_refusal(
        raised.value,
        operation="moments",
        request={
            "experiment": "asof_late_exp",
            "design": "encouragement",
            "cluster": "store_id",
            "n_pre_periods": 2,
        },
        offered={"n_pre_periods": 0},
    )


def test_day_source_clustered_encouragement_refuses_before_uptake_warehouse_work(con, monkeypatch):
    """Cluster incompatibility is a pure preflight, even without CUPED."""
    from tests.test_analysis_asof import _analysis_with_asof_encouragement_events

    analysis_experiment, analysis_resolver = _analysis_with_asof_encouragement_events(con)
    experiment = experiment_definition(analysis_experiment, analysis_resolver).model_copy(
        update={"cluster": "store_id", "n_pre_periods": 0}
    )
    clustered_experiment, clustered_resolver = make_binding_like(
        analysis_resolver, experiment=experiment
    )
    source = bound_source(clustered_resolver)
    metric = metrics(clustered_experiment, clustered_resolver)[0]

    def fail(*args, **kwargs):
        raise AssertionError("clustered encouragement day request touched warehouse")

    monkeypatch.setattr(source, "_get_fact_table", fail)
    monkeypatch.setattr(source, "_get_exposures", fail)
    monkeypatch.setattr(source, "_note_reduction", fail)

    with pytest.raises(CapabilityError) as raised:
        _moments(source.day_source(metrics=(metric,)), metric, grain="asof")
    _assert_operation_refusal(
        raised.value,
        operation="moments",
        request={
            "experiment": "asof_late_exp",
            "cluster": "store_id",
            "metrics": ("revenue",),
            "dimension": None,
        },
        offered=("total",),
    )


def test_native_view_operations_refuse_missing_extension_evidence(pricing_analysis, con):
    quantile = QuantileMetric(name="p50_revenue", entity="user_id", fact="purchase", quantile=0.5)
    experiment, resolver = _analysis_with_country_breakout(con)
    source = bound_source(resolver)
    with pytest.raises(CapabilityError) as raised:
        source.breakout_summaries(metrics=(quantile,))
    assert raised.value.code == "evidence.operation_unavailable"
    with pytest.raises(CapabilityError) as raised:
        source.breakout_source(
            experiment_definition(experiment, resolver).breakouts[0], metrics=(quantile,)
        )
    assert raised.value.code == "evidence.operation_unavailable"
    with pytest.raises(CapabilityError) as raised:
        source.day_source(metrics=(quantile,))
    assert raised.value.code == "evidence.operation_unavailable"


def test_day_source_refuses_winsorized_metrics_before_warehouse_work(
    pricing_analysis: tuple[Any, Any], monkeypatch
):
    source = bound_source(pricing_analysis[1])
    metric = next(m for m in metrics(*pricing_analysis) if m.name == "purchase_rate").model_copy(
        update={"winsorization": Winsorization(upper_value=1.0)}
    )
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse work started")),
    )

    with pytest.raises(CapabilityError) as raised:
        source.day_source(metrics=(metric,))
    assert raised.value.code == "evidence.operation_unavailable"
    assert raised.value.context["request"] == {
        "metrics": (
            {
                "metric": "purchase_rate",
                "threshold_source": "fixed_value",
                "thresholds": {"upper": {"source": "fixed_value", "value": 1.0}},
            },
        )
    }


def test_day_source_validates_metrics_passed_directly(pricing_analysis: tuple[Any, Any]):
    source = bound_source(pricing_analysis[1])
    declared = metrics(*pricing_analysis)[0]
    day = source.day_source(metrics=(declared,))
    quantile = QuantileMetric(name="p50", entity="user_id", fact="purchase", quantile=0.5)
    winsorized = declared.model_copy(update={"winsorization": Winsorization(upper_value=1.0)})
    breakout = cast("Breakout", None)

    with pytest.raises(CapabilityError) as raised:
        _moments(day, quantile)
    assert raised.value.code == "evidence.operation_unavailable"
    with pytest.raises(CapabilityError) as raised:
        _moments(day, winsorized)
    assert raised.value.code == "evidence.operation_unavailable"
    with pytest.raises(CapabilityError) as raised:
        day.breakout_moments(quantile, breakout)
    assert raised.value.code == "evidence.operation_unavailable"
    with pytest.raises(CapabilityError) as raised:
        day.breakout_moments(winsorized, breakout)
    assert raised.value.code == "evidence.operation_unavailable"


def test_direct_native_sitewide_uses_assigned_evidence_with_declared_trigger(tmp_path, monkeypatch):
    import ibis

    from tests.test_analysis_trigger import (
        _defs_yaml,
        _events,
        _with_preassignment_region,
    )

    definitions = (
        _defs_yaml()
        .replace(
            "    facts:\n",
            "    properties:\n"
            "      - {name: region, column: region, dtype: string, as_of: pre_exposure}\n"
            "    facts:\n",
        )
        .replace(
            "    plan:\n",
            "    breakouts: [{property: region, source: events}]\n"
            "    factors: [{property: region, source: events}]\n"
            "    plan:\n",
        )
    )
    path = tmp_path / "triggered-definitions.yml"
    path.write_text(definitions)
    con = ibis.duckdb.connect()
    con.create_table("events", obj=_with_preassignment_region(_events(n_per_arm=10)))
    experiment, resolver = from_definitions(path, con, experiment_name="exp")
    source = bound_source(resolver)
    metric = metrics(experiment, resolver)[0]

    try:
        evidence = source.sitewide_evidence(metric)
        assert {arm.group_id: arm.n for arm in evidence.arm_stats} == {
            "C": 10,
            "T": 10,
        }

        # These other triggered views still require snapshot evidence before
        # resolving warehouse tables.
        monkeypatch.setattr(
            source,
            "_note_reduction",
            lambda: (_ for _ in ()).throw(AssertionError("warehouse work started")),
        )
        for call in (breakout_summaries, factor_summaries):
            with pytest.raises(CapabilityError) as raised:
                call(experiment, resolver)
            assert raised.value.code == "source.native.trigger_evidence_required"
    finally:
        cast(Any, resolver).close()
        con.disconnect()


def test_day_source_subset_plan_is_frozen(pricing_analysis: tuple[Any, Any]):
    source = bound_source(pricing_analysis[1])
    metric = metrics(*pricing_analysis)[0]
    day = source.day_source(metrics=(metric,))
    assert set(day.context.plan.procedures) == {metric.name}
    with pytest.raises(TypeError):
        day.context.plan.procedures["new"] = day.context.plan.procedures[metric.name]  # type: ignore[index]


pytestmark = pytest.mark.integration
