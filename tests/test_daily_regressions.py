from __future__ import annotations

import pytest

from increment import View, from_definitions
from increment.semantics.models import AnalysisPlan, ExperimentMetric, MethodSpec
from increment.workflows import experiment_definition, metrics
from tests.analysis_factory import make_binding_like, view_readout


@pytest.mark.parametrize("cuped_first", [True, False])
@pytest.mark.integration
def test_native_daily_mixed_order_requests_cuped_only_for_declared_metric(
    seeded_pre_period_con, seeded_defs, cuped_first
):
    analysis_experiment, analysis_resolver = from_definitions(
        seeded_defs, seeded_pre_period_con, experiment_name="new_onboarding_v2"
    )
    cuped_metric = ExperimentMetric(
        metric="purchase_rate",
        decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
    )
    secondaries = (
        [cuped_metric, "avg_session_duration"]
        if cuped_first
        else ["avg_session_duration", cuped_metric]
    )
    catalog = {m.name: m for m in metrics(analysis_experiment, analysis_resolver)}
    order = (
        ["purchase_rate", "avg_session_duration"]
        if cuped_first
        else ["avg_session_duration", "purchase_rate"]
    )
    configured_experiment, configured_resolver = make_binding_like(
        analysis_resolver,
        [catalog[name] for name in order],
        experiment=experiment_definition(analysis_experiment, analysis_resolver).model_copy(
            update={"plan": AnalysisPlan(secondaries=secondaries)}
        ),
    )
    results = view_readout(
        configured_experiment,
        configured_resolver,
        View(view="daily", grain="daily"),
        metrics=["avg_session_duration", "purchase_rate"],
    )

    # Declaration order does not move the CUPED binding off the declared
    # metric, and the covariate-bearing moments let that metric estimate.
    assert {(row.metric, row.method, row.method_role) for row in results} == {
        ("purchase_rate", "cuped", "decision"),
        ("avg_session_duration", "unadjusted", "decision"),
    }
    assert any(row.lift is not None for row in results if row.metric == "purchase_rate")


@pytest.mark.integration
def test_native_daily_inherited_methods_and_roles_reach_estimator(
    seeded_pre_period_con, seeded_defs
):
    analysis_experiment, analysis_resolver = from_definitions(
        seeded_defs, seeded_pre_period_con, experiment_name="new_onboarding_v2"
    )
    configured_experiment, configured_resolver = make_binding_like(
        analysis_resolver,
        experiment=experiment_definition(analysis_experiment, analysis_resolver).model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(
                            metric="purchase_rate",
                            decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
                            prior=None,
                        ),
                        "avg_session_duration",
                    ]
                )
            }
        ),
    )
    results = view_readout(
        configured_experiment,
        configured_resolver,
        View(view="daily", grain="daily"),
        metrics=["purchase_rate", "avg_session_duration"],
    )

    assert {(row.metric, row.method, row.method_role, row.role) for row in results} == {
        ("purchase_rate", "cuped", "decision", "secondary"),
        ("avg_session_duration", "unadjusted", "decision", "secondary"),
    }


@pytest.mark.integration
def test_daily_missing_date_retains_unavailable_cells_with_relative_student_prior(con):
    import datetime as dt

    from increment import run
    from increment.estimators import RelativeLiftPrior, StudentTPrior
    from increment.results import AnalysisEnvelope, DailyLiftEstimates
    from tests.test_analysis_daily import _analysis_with_daily_events

    experiment, reader = _analysis_with_daily_events(con)
    definition = experiment_definition(experiment, reader)
    missing_day = definition.start_day - dt.timedelta(days=1)
    experiment, reader = make_binding_like(
        reader,
        experiment=definition.model_copy(update={"start": definition.start - dt.timedelta(days=1)}),
    )
    scope = View(view="daily", grain="daily")
    result = run(
        experiment.with_views(scope),
        reader,
        views=(scope,),
        prior=RelativeLiftPrior(distribution=StudentTPrior(nu=4, scale=0.5)),
    )
    daily = result.readout(scope)
    assert isinstance(daily, DailyLiftEstimates)
    rows = {row.ds: row for row in daily}
    observed_days = (dt.date(2025, 6, 1), dt.date(2025, 6, 2))
    assert set(rows) == {*observed_days, missing_day}
    assert all(rows[day].posterior_available is True for day in observed_days)
    gap = rows[missing_day]
    assert gap.posterior_available is False
    assert gap.posterior_relative_model is None
    assert gap.posterior_student_t_state is None
    restored = type(gap).model_validate_json(gap.model_dump_json())
    assert restored.sampling_reason_code == "readout.cell.missing_metric_observations"
    assert restored.posterior_reason_code == restored.sampling_reason_code
    assert restored.lift is None
    saved = AnalysisEnvelope.from_arrow(result.envelope.to_arrow())
    assert len(saved.cells) == len(result.cells) == 3
