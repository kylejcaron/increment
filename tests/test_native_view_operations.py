"""Parity and capability tests for source-owned native view workflows."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import cast

import pytest

from increment import Analysis
from increment.errors import CapabilityError
from increment.estimation.sitewide import SitewideImpact
from increment.semantics.models import (
    AnalysisPlan,
    Breakout,
    Measure,
    MultiplicitySpec,
    QuantileMetric,
    Winsorization,
)
from tests.analysis_factory import _native_source, make_analysis_like
from tests.query.test_builders import _analysis_with_cross_source_breakouts
from tests.test_analysis_breakout import _analysis_with_country_breakout
from tests.test_analysis_sitewide import _REVENUE_PER_USER, _REVENUE_PER_USER_PLAN
from tests.test_native_core_operations import _assert_operation_refusal


@pytest.fixture(scope="module")
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


def _assert_model_rows_match(actual_rows, expected_rows) -> None:
    _assert_approx_row_sets(
        [row.model_dump(mode="json") for row in actual_rows],
        [row.model_dump(mode="json") for row in expected_rows],
    )


@pytest.fixture
def pricing_analysis(pricing_con):
    return Analysis.from_definitions("pricing_tier_test", "examples/definitions/", pricing_con)


def test_sitewide_impact_is_typed_and_reports_iid_arm_attribution(pricing_con):
    analysis = make_analysis_like(
        Analysis("pricing_tier_test", definitions_path="examples/definitions/", con=pricing_con),
        [_REVENUE_PER_USER],
        plan=_REVENUE_PER_USER_PLAN,
    )

    result = cast(SitewideImpact, analysis.sitewide("revenue_per_user"))

    assert result.treatment_group == "treatment"
    assert result.other_arm_ids == ()
    assert result.n_clusters is None
    assert result.n_control == 20
    assert result.n_treatment == 20
    # Hand-computed from the fixture: control 10 converters x $20 = $200,
    # treatment 14 converters x $40 = $560.
    assert result.site_total_volume == pytest.approx(760.0)
    assert result.delta == pytest.approx(18.0)


def test_sitewide_evidence_refuses_ratio_before_raw_work(pricing_analysis):
    """The named (non-ratio) evidence contract refuses a ratio metric.

    ``Analysis.sitewide`` requests the ratio-specific reducer instead, so the
    default flag this refusal guards is only reachable at the source.
    """
    metric = next(m for m in pricing_analysis.metrics if m.name == "revenue_per_session")
    source = _native_source(pricing_analysis)
    with pytest.raises(CapabilityError) as raised:
        source.sitewide_evidence(metric)
    assert raised.value.code == "source.native.operation"


def test_sitewide_evidence_refuses_winsorization_before_raw_work(pricing_analysis, monkeypatch):
    metric = next(m for m in pricing_analysis.metrics if m.name == "purchase_rate")
    winsorized = metric.model_copy(update={"winsorization": Winsorization(upper_value=1.0)})
    source = _native_source(pricing_analysis)
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse work started")),
    )
    with pytest.raises(CapabilityError) as raised:
        source.sitewide_evidence(winsorized)
    assert raised.value.code == "source.native.operation"


@pytest.mark.parametrize(
    "winsorization",
    [Winsorization(upper_percentile=0.9)],
)
def test_breakout_summaries_refuses_winsorization_before_warehouse_work(
    pricing_con, monkeypatch, winsorization
):
    analysis = _analysis_with_country_breakout(pricing_con)
    source = _native_source(analysis)
    metric = analysis.metrics[0].model_copy(update={"winsorization": winsorization})
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

    assert raised.value.code == "source.native.operation"


def test_breakout_summaries_preserves_fixed_value_winsorization(pricing_con):
    analysis = _analysis_with_country_breakout(pricing_con)
    source = _native_source(analysis)
    metric = analysis.metrics[0].model_copy(
        update={"winsorization": Winsorization(upper_value=15.0)}
    )

    result = source.breakout_summaries(metrics=(metric,))
    summary = next(iter(result.values()))["group_summary"]

    assert summary.num_rows
    assert set(summary["winsor_upper_bound"].to_pylist()) == {15.0}


def test_breakout_summaries_keep_multi_source_keys(con):
    analysis = _analysis_with_cross_source_breakouts(con)

    result = analysis.breakout_summaries()

    assert set(result) == {
        "visit_rate:country:src_a",
        "visit_rate:country:src_b",
        "checkout:country:src_a",
        "checkout:country:src_b",
    }


def test_run_breakout_preserves_all_declared_sources(con):
    analysis = _analysis_with_cross_source_breakouts(con)
    analysis = make_analysis_like(
        analysis,
        plan=AnalysisPlan(view_multiplicity=MultiplicitySpec(correction="none")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        results = analysis.run_breakout()

    assert {result.source for result in results} == {"src_a", "src_b"}


def test_day_source_matches_daily_and_asof_views(pricing_analysis):
    from increment.breakout.estimates import DailyMetricValues
    from increment.breakout.estimates import run_daily as _run_daily_values

    metric = next(m for m in pricing_analysis.metrics if m.name == "purchase_rate")
    source = _native_source(pricing_analysis)
    day = source.day_source(metrics=(metric,))

    daily_rows = day.moments(metric, grain="daily")
    asof_rows = day.moments(metric, grain="asof")
    direct_daily = DailyMetricValues(_run_daily_values(daily_rows, metrics=[metric], view="daily"))
    direct_asof = DailyMetricValues(_run_daily_values(asof_rows, metrics=[metric], view="asof"))
    facade_daily = pricing_analysis.run_daily(metrics=(metric,))
    facade_asof = pricing_analysis.run_asof(metrics=(metric,))
    _assert_model_rows_match(direct_daily, facade_daily)
    _assert_model_rows_match(direct_asof, facade_asof)


def test_day_source_dimensioned_result_is_typed_and_attributed(pricing_con):
    analysis = _analysis_with_country_breakout(pricing_con)
    metric = analysis.metrics[0]

    values = analysis.run_daily(metrics=(metric,), dimension="country")

    assert {row.metric for row in values} == {metric.name}
    assert {row.dimension for row in values} == {"country"}
    assert {row.dimension_value for row in values} == {"US", "CA"}
    assert {row.source for row in values} == {"events"}


def test_day_source_adhoc_ratio_denominator_extends_daily_and_asof_horizon():
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
    analysis = Analysis.from_definitions("pricing_tier_test", "examples/definitions/", con)
    declared = next(m for m in analysis.metrics if m.name == "revenue_per_session")
    assert hasattr(declared, "denominator")
    adhoc = declared.model_copy(
        update={
            "name": "revenue_per_session_start",
            "numerator": Measure(fact="purchase", aggregation="sum", window_days=30),
            "denominator": Measure(fact="session_start", aggregation="count", window_days=30),
        }
    )

    declared_daily = analysis.run_daily(metrics=(declared,))
    daily = analysis.run_daily(metrics=(adhoc,))
    asof = analysis.run_asof(metrics=(adhoc,))

    declared_horizon = max(row.ds for row in declared_daily)
    assert max(row.ds for row in daily) > declared_horizon
    assert max(row.ds for row in asof) > declared_horizon
    assert any(row.ds == late_day.date() for row in daily)
    assert any(row.ds == late_day.date() for row in asof)


def test_day_source_validates_missing_breakout_before_reduction_or_properties(
    pricing_con, monkeypatch
):
    analysis = _analysis_with_country_breakout(pricing_con)
    metric = analysis.metrics[0]
    source = _native_source(analysis)
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
        day.breakout_moments(metric, Breakout(property="missing_dimension"))
    assert raised.value.code == "source.native.operation"


def test_day_source_validates_breakout_source_before_reduction(pricing_con, monkeypatch):
    analysis = _analysis_with_country_breakout(pricing_con)
    metric = analysis.metrics[0]
    source = _native_source(analysis)
    day = source.day_source(metrics=(metric,))
    breakout = analysis.experiment.breakouts[0]
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
    assert raised.value.code == "source.native.operation"


def test_day_source_rejects_existing_undeclared_source_before_warehouse_work(con, monkeypatch):
    analysis = _analysis_with_cross_source_breakouts(con)
    declared = analysis.experiment.breakouts[0]
    analysis = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(update={"breakouts": (declared,)}),
    )
    source = _native_source(analysis)
    metric = analysis.metrics[0]
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

    assert raised.value.code == "source.native.operation"


def test_breakout_source_rejects_existing_undeclared_source_before_warehouse_work(con, monkeypatch):
    analysis = _analysis_with_cross_source_breakouts(con)
    declared = analysis.experiment.breakouts[0]
    analysis = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(update={"breakouts": (declared,)}),
    )
    source = _native_source(analysis)
    metric = analysis.metrics[0]
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

    _assert_operation_refusal(
        raised.value,
        operation="breakout_source",
        request={
            "experiment": "collision_exp",
            "dimension": "country",
            "source": "src_b",
            "resolved_source": "src_b",
        },
        offered=("src_a",),
    )


def _with_breakout_source(analysis: Analysis, source: str | None) -> Analysis:
    """The same analysis with every declared breakout's source set explicitly."""
    return make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
            update={
                "breakouts": tuple(
                    breakout.model_copy(update={"source": source})
                    for breakout in analysis.experiment.breakouts
                )
            }
        ),
    )


def test_breakout_source_treats_inferred_and_explicit_breakout_sources_as_equivalent(
    pricing_con,
):
    analysis = _analysis_with_country_breakout(pricing_con)
    inferred_analysis = _with_breakout_source(analysis, None)
    explicit_analysis = _with_breakout_source(analysis, "events")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        inferred = inferred_analysis.run_breakout()
        explicit = explicit_analysis.run_breakout()

    assert {result.source for result in inferred} == {"events"}
    assert {result.source for result in explicit} == {"events"}
    _assert_model_rows_match(inferred, explicit)


def test_day_source_treats_inferred_and_explicit_breakout_sources_as_equivalent(
    pricing_con,
):
    analysis = _analysis_with_country_breakout(pricing_con)
    metric = analysis.metrics[0]
    inferred_analysis = _with_breakout_source(analysis, None)
    explicit_analysis = _with_breakout_source(analysis, "events")

    inferred = inferred_analysis.run_daily(metrics=(metric,), dimension="country")
    explicit = explicit_analysis.run_daily(metrics=(metric,), dimension="country")

    assert {row.source for row in inferred} == {"events"}
    assert {row.source for row in explicit} == {"events"}
    _assert_model_rows_match(inferred, explicit)


def test_day_source_preserves_retention_cohort_path(pricing_analysis):
    metric = next(m for m in pricing_analysis.metrics if m.name == "d7_retention")
    source = _native_source(pricing_analysis)
    rows = source.day_source(metrics=(metric,)).moments(metric, grain="daily")
    assert rows
    assert {row["metric"] for row in rows} == {metric.name}
    assert {row["ds"] for row in rows} == {rows[0]["ds"]}


def test_day_preflight_uptake_cluster_refusal_is_coded_before_builder(con, monkeypatch):
    from tests.test_analysis_asof import _analysis_with_asof_encouragement_events

    analysis = _analysis_with_asof_encouragement_events(con)
    experiment = analysis.experiment.model_copy(update={"cluster": "store_id"})
    clustered = make_analysis_like(analysis, experiment=experiment)
    source = _native_source(clustered)
    metric = clustered.metrics[0]

    def fail(*args, **kwargs):
        raise AssertionError("clustered encouragement day request touched warehouse")

    monkeypatch.setattr(source, "_get_exposures", fail)
    monkeypatch.setattr(source, "_note_reduction", fail)

    with pytest.raises(CapabilityError) as raised:
        source.day_source(metrics=(metric,)).moments(metric, grain="asof", include_covariate=True)
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

    analysis = _analysis_with_asof_encouragement_events(con)
    experiment = analysis.experiment.model_copy(update={"cluster": "store_id", "n_pre_periods": 0})
    clustered = make_analysis_like(analysis, experiment=experiment)
    source = _native_source(clustered)
    metric = clustered.metrics[0]

    def fail(*args, **kwargs):
        raise AssertionError("clustered encouragement day request touched warehouse")

    monkeypatch.setattr(source, "_get_fact_table", fail)
    monkeypatch.setattr(source, "_get_exposures", fail)
    monkeypatch.setattr(source, "_note_reduction", fail)

    with pytest.raises(CapabilityError) as raised:
        source.day_source(metrics=(metric,)).moments(metric, grain="asof")
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
    analysis = _analysis_with_country_breakout(con)
    source = _native_source(analysis)
    with pytest.raises(CapabilityError) as raised:
        source.breakout_summaries(metrics=(quantile,))
    assert raised.value.code == "source.native.operation"
    with pytest.raises(CapabilityError) as raised:
        source.breakout_source(analysis.experiment.breakouts[0], metrics=(quantile,))
    assert raised.value.code == "source.native.operation"
    with pytest.raises(CapabilityError) as raised:
        source.day_source(metrics=(quantile,))
    assert raised.value.code == "source.native.operation"


def test_day_source_refuses_winsorized_metrics_before_warehouse_work(pricing_analysis, monkeypatch):
    source = _native_source(pricing_analysis)
    metric = pricing_analysis.metrics[0].model_copy(
        update={"winsorization": Winsorization(upper_value=1.0)}
    )
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse work started")),
    )

    with pytest.raises(CapabilityError) as raised:
        source.day_source(metrics=(metric,))
    assert raised.value.code == "source.native.operation"


def test_day_source_validates_metrics_passed_directly(pricing_analysis):
    source = _native_source(pricing_analysis)
    declared = pricing_analysis.metrics[0]
    day = source.day_source(metrics=(declared,))
    quantile = QuantileMetric(name="p50", entity="user_id", fact="purchase", quantile=0.5)
    winsorized = declared.model_copy(update={"winsorization": Winsorization(upper_value=1.0)})
    breakout = cast("Breakout", None)

    with pytest.raises(CapabilityError) as raised:
        day.moments(quantile)
    assert raised.value.code == "source.native.operation"
    with pytest.raises(CapabilityError) as raised:
        day.moments(winsorized)
    assert raised.value.code == "source.native.operation"
    with pytest.raises(CapabilityError) as raised:
        day.breakout_moments(quantile, breakout)
    assert raised.value.code == "source.native.operation"
    with pytest.raises(CapabilityError) as raised:
        day.breakout_moments(winsorized, breakout)
    assert raised.value.code == "source.native.operation"


def test_direct_native_view_operations_refuse_declared_trigger_before_warehouse_work(
    tmp_path, monkeypatch
):
    from tests.test_analysis_trigger import _analysis, _events

    analysis = _analysis(tmp_path, _events(n_per_arm=10))
    source = _native_source(analysis)
    metric = analysis.metrics[0]
    monkeypatch.setattr(
        source,
        "_note_reduction",
        lambda: (_ for _ in ()).throw(AssertionError("warehouse work started")),
    )

    calls = (
        lambda: source.sitewide_evidence(metric),
        lambda: source.breakout_summaries(metrics=(metric,)),
        lambda: source.factor_summaries(metrics=(metric,)),
        lambda: source.breakout_source(cast("Breakout", None), metrics=(metric,)),
        lambda: source.day_source(metrics=(metric,)),
        lambda: source.moments(metric, grain="daily"),
        lambda: source.moments(metric, grain="asof"),
    )
    for call in calls:
        with pytest.raises(CapabilityError) as raised:
            call()
        assert raised.value.code == "source.native.operation"


def test_day_source_subset_plan_is_frozen(pricing_analysis):
    source = _native_source(pricing_analysis)
    metric = pricing_analysis.metrics[0]
    day = source.day_source(metrics=(metric,))
    assert set(day.context.plan.procedures) == {metric.name}
    with pytest.raises(TypeError):
        day.context.plan.procedures["new"] = day.context.plan.procedures[metric.name]  # type: ignore[index]  # ty: ignore[invalid-assignment]
