"""``Experiment.cluster`` declares a randomization grain coarser than the analysis unit;
anything it cannot honestly combine with refuses at load time, not at readout with a
half-built query."""

from __future__ import annotations

import datetime as dt

import pytest

from increment.errors import DefinitionError
from increment.semantics.models import AnalysisPlan, Definitions, Experiment


def _experiment(cluster: str = "store_id", n_pre_periods: int = 0) -> Experiment:
    return Experiment(
        name="store_test",
        exposure="enrolled",
        unit="user_id",
        cluster=cluster,
        start=dt.datetime(2025, 8, 1),
        control_group="control",
        n_pre_periods=n_pre_periods,
        plan=AnalysisPlan(),
    )


def _defs(metric_type: str = "mean", **experiment_overrides) -> Definitions:
    experiment = {
        "name": "store_test",
        "exposure": "enrolled",
        "unit": "user_id",
        "cluster": "store_id",
        "start": "2025-08-01",
        "control_group": "control",
        "plan": {"secondaries": ["m"]},
    }
    experiment.update(experiment_overrides)
    metric: dict = {"name": "m", "type": metric_type, "entity": "user_id", "fact": "purchase"}
    if metric_type == "ratio":
        metric = {
            "name": "m",
            "type": "ratio",
            "entity": "user_id",
            "numerator": {"fact": "purchase", "aggregation": "sum"},
            "denominator": {"fact": "purchase", "aggregation": "count"},
        }
    return Definitions.model_validate(
        {
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM events",
                    "timestamp_column": "ts",
                    "entities": ["user_id"],
                    "facts": [{"name": "purchase", "column": "revenue"}],
                    "properties": [
                        {
                            "name": "country",
                            "column": "country_code",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "purchase"}],
            "metrics": [metric],
            "experiments": [experiment],
        }
    )


def test_cluster_declaration_loads():
    assert _experiment().cluster == "store_id"


def test_cluster_equal_to_unit_refuses():
    with pytest.raises(DefinitionError) as exc_info:
        _experiment(cluster="user_id")
    assert exc_info.value.code == "definition.experiment.cluster_names_same"


def test_cluster_with_bare_n_pre_periods_loads():
    """n_pre_periods alone (no CUPED-requesting binding) never touches the pre-period
    covariate under a declared cluster; only a binding that actually requests CUPED refuses."""
    assert _experiment(n_pre_periods=14).n_pre_periods == 14


def test_cluster_with_cuped_binding_refuses():
    with pytest.raises(DefinitionError) as exc_info:
        Experiment(
            name="store_test",
            exposure="enrolled",
            unit="user_id",
            cluster="store_id",
            start=dt.datetime(2025, 8, 1),
            control_group="control",
            n_pre_periods=14,
            plan=AnalysisPlan(
                secondaries=[
                    {
                        "metric": "m",
                        "decision_method": {"name": "cuped", "variance_reduction": "cuped"},
                    }
                ]
            ),
        )
    assert exc_info.value.code == "definition.experiment.cluster_combined_n"


def test_cluster_with_mean_metric_loads():
    defs = _defs("mean")
    assert defs.experiments[0].cluster == "store_id"


def test_cluster_with_ratio_metric_loads():
    """A clustered ratio row carries the metric's own per-cluster
    denominator totals in the den family, so the pairing is servable."""
    defs = _defs("ratio")
    assert defs.experiments[0].cluster == "store_id"


def test_cluster_with_breakouts_refuses_at_load():
    with pytest.raises(DefinitionError) as exc_info:
        _defs("mean", breakouts=[{"property": "country"}])
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.declares_cluster_alongside" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }


def test_cluster_with_factors_refuses_at_load():
    with pytest.raises(DefinitionError) as exc_info:
        _defs("mean", factors=[{"property": "country", "source": "events"}])
    assert exc_info.value.code == "definition.invalid"
    assert "definition.validate_experiment.declares_cluster_alongside" in {
        c
        for c, _ in exc_info.value.context["errors"]  # ty: ignore[not-iterable]
    }
