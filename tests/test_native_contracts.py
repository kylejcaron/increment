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


# Two full as-of sweeps over a conversion metric, each inverting an exact
# independent-binomial interval per day and arm. That cost is inherent to the
# exact method, not a defect, so this runs in the slow tier.
@pytest.mark.slow
def test_native_asof_prior_shrinks_dimensioned_and_undimensioned(
    seeded_pre_period_con, seeded_defs
):
    analysis = Analysis("new_onboarding_v2", seeded_defs, seeded_pre_period_con)
    flat = analysis.run_asof_lift(metrics=["purchase_rate"])
    shrunk = analysis.run_asof_lift(metrics=["purchase_rate"], prior=Normal(mu=0.0, sigma=0.01))
    flat_by_day_arm = {
        (row.ds, row.group_id): row
        for row in flat
        if row.lift is not None and math.isfinite(row.require_lift().value)
    }
    shrunk_row = _finite_row(row for row in shrunk if (row.ds, row.group_id) in flat_by_day_arm)
    flat_row = flat_by_day_arm[(shrunk_row.ds, shrunk_row.group_id)]
    assert abs(shrunk_row.require_lift().value) < abs(flat_row.require_lift().value)

    dimensioned = analysis.run_asof_lift(metrics=["purchase_rate"], dimension="country")
    dimensioned_shrunk = analysis.run_asof_lift(
        metrics=["purchase_rate"], dimension="country", prior=Normal(mu=0.0, sigma=0.01)
    )
    flat_by_key = {
        (row.ds, row.group_id, row.dimension_value): row
        for row in dimensioned
        if row.lift is not None and math.isfinite(row.require_lift().value)
    }
    shrunk_row = _finite_row(
        row
        for row in dimensioned_shrunk
        if (row.ds, row.group_id, row.dimension_value) in flat_by_key
    )
    flat_row = flat_by_key[(shrunk_row.ds, shrunk_row.group_id, shrunk_row.dimension_value)]
    assert abs(shrunk_row.require_lift().value) < abs(flat_row.require_lift().value)


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
