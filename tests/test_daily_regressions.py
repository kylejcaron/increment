from __future__ import annotations

import pytest

from increment.analysis import Analysis
from increment.semantics.models import AnalysisPlan, ExperimentMetric, MethodSpec
from tests.analysis_factory import make_analysis_like


@pytest.mark.parametrize("cuped_first", [True, False])
def test_native_daily_mixed_order_requests_cuped_only_for_declared_metric(
    seeded_pre_period_con, seeded_defs, cuped_first
):
    analysis = Analysis("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    cuped_metric = ExperimentMetric(
        metric="purchase_rate",
        decision_method=MethodSpec(name="cuped", variance_reduction="cuped"),
    )
    secondaries = (
        [cuped_metric, "avg_session_duration"]
        if cuped_first
        else ["avg_session_duration", cuped_metric]
    )
    catalog = {m.name: m for m in analysis.metrics}
    order = (
        ["purchase_rate", "avg_session_duration"]
        if cuped_first
        else ["avg_session_duration", "purchase_rate"]
    )
    configured = make_analysis_like(
        analysis,
        [catalog[name] for name in order],
        experiment=analysis.experiment.model_copy(
            update={"plan": AnalysisPlan(secondaries=secondaries)}
        ),
    )
    results = configured.run_daily_lift(metrics=["avg_session_duration", "purchase_rate"])

    # Declaration order does not move the CUPED binding off the declared
    # metric, and the covariate-bearing moments let that metric estimate.
    assert {(row.metric, row.method, row.method_role) for row in results} == {
        ("purchase_rate", "cuped", "decision"),
        ("avg_session_duration", "unadjusted", "decision"),
    }
    assert any(row.lift is not None for row in results if row.metric == "purchase_rate")


def test_native_daily_inherited_methods_and_roles_reach_estimator(
    seeded_pre_period_con, seeded_defs
):
    analysis = Analysis("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    configured = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
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
    results = configured.run_daily_lift(metrics=["purchase_rate", "avg_session_duration"])

    assert {(row.metric, row.method, row.method_role, row.role) for row in results} == {
        ("purchase_rate", "cuped", "decision", "secondary"),
        ("avg_session_duration", "unadjusted", "decision", "secondary"),
    }
