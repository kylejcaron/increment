from __future__ import annotations

import datetime as dt
import math

import pytest

from increment import Scope, from_definitions
from increment.errors import InvalidRequestError
from increment.estimation.inference import Normal
from increment.evidence import BreakoutIdentity
from increment.semantics.models import AnalysisPlan, ExperimentMetric, MethodSpec
from increment.workflows import experiment_definition
from tests.analysis_factory import bound_source, make_binding_like, view_readout
from tests.test_analysis_breakout import _configured_country_breakout_analysis as _breakout_analysis

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_native_asof_prior_preserves_sparse_exact_inference_dimensioned_and_undimensioned(
    seeded_pre_period_con, seeded_defs
):
    """Sparse exact sampling stays prior-free when its approximate posterior is unavailable."""
    experiment, resolver = from_definitions(
        seeded_defs, seeded_pre_period_con, experiment_name="new_onboarding_v2"
    )
    flat = view_readout(
        experiment, resolver, Scope(view="asof", grain="asof"), metrics=["purchase_rate"]
    )
    prior = view_readout(
        experiment,
        resolver,
        Scope(view="asof", grain="asof"),
        metrics=["purchase_rate"],
        prior=Normal(mu=0.0, sigma=0.01),
    )
    assert [row.ds for row in prior] == sorted(row.ds for row in prior)
    flat_by_day_arm = {
        (row.ds, row.group_id): row
        for row in flat
        if row.lift is not None and math.isfinite(row.require_lift().value)
    }
    prior_row = next(
        row for row in prior if row.ds == dt.date(2025, 1, 16) and row.group_id == "treatment"
    )
    flat_row = flat_by_day_arm[(prior_row.ds, prior_row.group_id)]
    assert prior_row.require_lift().value == flat_row.require_lift().value
    assert prior_row.posterior_available is False

    dimensioned = view_readout(
        experiment,
        resolver,
        Scope(view="asof", grain="asof", segment=BreakoutIdentity("country", None)),
        metrics=["purchase_rate"],
    )
    dimensioned_prior = view_readout(
        experiment,
        resolver,
        Scope(view="asof", grain="asof", segment=BreakoutIdentity("country", None)),
        metrics=["purchase_rate"],
        prior=Normal(mu=0.0, sigma=0.01),
    )
    assert [row.ds for row in dimensioned_prior] == sorted(row.ds for row in dimensioned_prior)
    flat_by_key = {
        (row.ds, row.group_id, row.dimension_value): row
        for row in dimensioned
        if row.lift is not None and math.isfinite(row.require_lift().value)
    }
    prior_row = next(
        row
        for row in dimensioned_prior
        if row.ds == dt.date(2025, 1, 16)
        and row.group_id == "treatment"
        and row.dimension_value == "CA"
    )
    flat_row = flat_by_key[(prior_row.ds, prior_row.group_id, prior_row.dimension_value)]
    assert prior_row.require_lift().value == flat_row.require_lift().value
    assert prior_row.posterior_available is False


def test_native_breakout_invalid_inherited_config_refuses_before_scoped_query(con, monkeypatch):
    analysis_experiment, analysis_resolver = _breakout_analysis(con, "none")
    invalid_experiment, invalid_resolver = make_binding_like(
        analysis_resolver,
        experiment=experiment_definition(analysis_experiment, analysis_resolver).model_copy(
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

    monkeypatch.setattr(bound_source(invalid_resolver), "breakout_sources", fail)
    with pytest.raises(InvalidRequestError) as exc_info:
        view_readout(invalid_experiment, invalid_resolver, Scope(view="breakout", grain="total"))
    assert exc_info.value.code == "estimation.engine.method_name_observational"
