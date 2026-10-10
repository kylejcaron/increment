from __future__ import annotations

import math

import pytest

from increment.analysis import Analysis
from increment.errors import InvalidRequestError
from increment.estimation.inference import Normal
from increment.semantics.models import AnalysisPlan, ExperimentMetric, MethodSpec
from tests.analysis_factory import _native_source
from tests.test_analysis_breakout import _configured_country_breakout_analysis as _breakout_analysis
from tests.test_analysis_breakout import make_analysis_like


def _finite_row(rows):
    for row in rows:
        if row.lift is not None and math.isfinite(row.require_lift().value):
            return row
    raise AssertionError("no finite native as-of row")


@pytest.mark.slow
def test_native_asof_prior_preserves_sparse_exact_inference_dimensioned_and_undimensioned(
    seeded_pre_period_con, seeded_defs
):
    """Sparse exact sampling stays prior-free when its approximate posterior is unavailable."""
    analysis = Analysis("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    flat = analysis.run_asof_lift(metrics=["purchase_rate"])
    prior = analysis.run_asof_lift(metrics=["purchase_rate"], prior=Normal(mu=0.0, sigma=0.01))
    flat_by_day_arm = {
        (row.ds, row.group_id): row
        for row in flat
        if row.lift is not None and math.isfinite(row.require_lift().value)
    }
    prior_row = _finite_row(row for row in prior if (row.ds, row.group_id) in flat_by_day_arm)
    flat_row = flat_by_day_arm[(prior_row.ds, prior_row.group_id)]
    assert prior_row.require_lift().value == flat_row.require_lift().value
    assert prior_row.posterior_available is False

    dimensioned = analysis.run_asof_lift(metrics=["purchase_rate"], dimension="country")
    dimensioned_prior = analysis.run_asof_lift(
        metrics=["purchase_rate"], dimension="country", prior=Normal(mu=0.0, sigma=0.01)
    )
    flat_by_key = {
        (row.ds, row.group_id, row.dimension_value): row
        for row in dimensioned
        if row.lift is not None and math.isfinite(row.require_lift().value)
    }
    prior_row = _finite_row(
        row
        for row in dimensioned_prior
        if (row.ds, row.group_id, row.dimension_value) in flat_by_key
    )
    flat_row = flat_by_key[(prior_row.ds, prior_row.group_id, prior_row.dimension_value)]
    assert prior_row.require_lift().value == flat_row.require_lift().value
    assert prior_row.posterior_available is False


def test_native_breakout_invalid_inherited_config_refuses_before_scoped_query(con, monkeypatch):
    analysis = _breakout_analysis(con, "none")
    invalid = make_analysis_like(
        analysis,
        experiment=analysis.experiment.model_copy(
            update={
                "plan": AnalysisPlan(
                    secondaries=[
                        ExperimentMetric(metric="revenue", decision_method=MethodSpec(name="iptw"))
                    ]
                )
            }
        ),
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("scoped breakout query ran before config validation")

    monkeypatch.setattr(_native_source(invalid), "breakout_sources", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        invalid.run_breakout()
    assert exc_info.value.code == "estimation.engine.method_name_observational"
