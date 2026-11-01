"""All six ingress classifications and four supported diagnostic paths."""

from dataclasses import replace
from typing import Any, cast

import pytest

from increment import from_definitions, from_frame
from tests.analysis_factory import view_readout
from tests.parity_harness.cases import (
    _observational_aipw_conversion_case,
    _observational_aipw_dml_case,
    _observational_conversion_case,
    _observational_covariate_case,
)
from tests.parity_harness.runner import assert_parity, run_case


def _diagnostics_present(results):
    weighted = [row for row in results if row.method in {"iptw", "aipw"}]
    assert weighted
    for row in results:
        if row.method in {"iptw", "aipw"}:
            assert row.weight_diagnostics_available is True
            assert row.weight_diagnostics_reason_code is None
            assert row.weight_diagnostics_reason_context is None
            assert row.weight_grain == "unit"
            assert 1 <= row.control_weight_ess <= row.control_weight_n
            assert 1 <= row.treatment_weight_ess <= row.treatment_weight_n
            assert 0 < row.control_weight_max_share <= 1
            assert 0 < row.treatment_weight_max_share <= 1
        else:
            assert row.weight_diagnostics_available is False
            assert row.weight_diagnostics_reason_code == "readout.weight_diagnostics.not_applicable"
            assert row.weight_diagnostics_reason_context == {"method": row.method}
            assert row.weight_definition is None
            assert row.weight_grain is None
            assert row.control_weight_n is None
            assert row.treatment_weight_n is None
            assert row.control_weight_ess is None
            assert row.treatment_weight_ess is None
            assert row.control_weight_max_share is None
            assert row.treatment_weight_max_share is None


@pytest.mark.slow
@pytest.mark.parametrize("builder", [_observational_covariate_case, _observational_aipw_dml_case])
@pytest.mark.integration
def test_observational_diagnostics_match_on_every_supported_ingress(builder):
    case = replace(builder(), readout_probe=_diagnostics_present)
    result = run_case(case)
    assert_parity(case, result)
    assert set(result.rows) == {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
        "from_unit_panel",
    }
    assert result.refusals["from_moments"] == "evidence.covariate_unavailable"
    assert "from_switchback_panel" in case.waive


@pytest.mark.slow
@pytest.mark.parametrize(
    "builder",
    [_observational_conversion_case, _observational_aipw_conversion_case],
)
def test_observational_conversion_diagnostics_match_on_every_supported_ingress(builder):
    case = replace(builder(), readout_probe=_diagnostics_present)
    result = run_case(case)
    assert_parity(case, result)
    assert set(result.rows) == {
        "from_definitions",
        "from_unit_day_artifact",
        "from_unit_summary",
        "from_unit_panel",
    }
    assert result.refusals["from_moments"] == "evidence.covariate_unavailable"
    assert "from_switchback_panel" in case.waive


def test_conversion_weight_diagnostics_are_retained_on_unit_summary():
    import pandas as pd

    from increment.breakout.estimates import LiftEstimates
    from increment.frame import MetricSpec
    from increment.semantics.design import AdjustmentSet, Observational
    from increment.semantics.models import AnalysisPlan

    frame = pd.DataFrame(
        [
            {
                "unit": f"{group}{index}",
                "group": group,
                "tenure": float(index % 5),
                "converted": int((index + (group == "treatment")) % 3 == 0),
            }
            for group in ("control", "treatment")
            for index in range(60)
        ]
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="group",
        metrics=[MetricSpec(name="converted", type="conversion")],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
        plan=AnalysisPlan(primary="converted"),
    )
    try:
        rows = view_readout(experiment, resolver)
        assert isinstance(rows, LiftEstimates)
        weighted = [row for row in rows if row.method == "iptw"]
        assert weighted
        assert all(row.weight_diagnostics_available is True for row in weighted)
        assert all(row.weight_grain == "unit" for row in weighted)
        assert all(row.control_weight_n == 60 for row in weighted)
        assert all(row.treatment_weight_n == 60 for row in weighted)
    finally:
        resolver.close()


def test_cluster_grain_weight_diagnostics_are_retained_on_unit_summary():
    import pandas as pd

    from increment.breakout.estimates import LiftEstimates
    from increment.frame import MetricSpec
    from increment.semantics.design import AdjustmentSet, Observational
    from increment.semantics.models import AnalysisPlan
    from tests.capability_consumers_auxiliary import assert_support

    frame = pd.DataFrame(
        [
            {
                "unit": f"{group}{cluster}-{index}",
                "group": group,
                "store": f"{group}-{cluster}",
                "tenure": float((index + cluster) % 5),
                "revenue": float(3 + (index + cluster) % 7),
            }
            for group in ("control", "treatment")
            for cluster in range(20)
            for index in range(12)
        ]
    )
    experiment, resolver = from_frame(
        frame,
        shape="summary",
        unit="unit",
        group="group",
        cluster="store",
        metrics=[MetricSpec(name="revenue", type="mean")],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
        plan=AnalysisPlan(primary="revenue"),
    )
    try:
        results = view_readout(experiment, resolver)
        assert isinstance(results, LiftEstimates)
        rows = [row for row in results if row.method == "iptw"]
        assert rows
        assert all(row.weight_diagnostics_available is True for row in rows)
        assert all(row.weight_grain == "cluster" for row in rows)
        assert all(row.control_weight_n == 20 for row in rows)
        assert all(row.treatment_weight_n == 20 for row in rows)
        assert_support(
            "observational_cluster_weight_diagnostics",
            "summary_cluster_weight_grain",
            rows,
        )
    finally:
        resolver.close()


@pytest.mark.integration
def test_definitions_cluster_weight_diagnostics_are_retained_at_cluster_grain(tmp_path):
    """The public Definitions reader projects IPTW diagnostics over the
    declared store grain, not the underlying user count."""
    import datetime as dt

    import ibis
    import pyarrow as pa

    from tests.capability_consumers_auxiliary import assert_support
    from tests.test_analysis_cluster import _DEFS_TEMPLATE, N_STORES, _event_rows

    rows = _event_rows()
    baselines = []
    for row in rows:
        # Same baseline within a unit, with arm-overlapping support.
        index = int(row["user_id"][1:].split("_")[0])
        row["tenure"] = float(index % 7)
        if row["event"] == "exposure":
            baselines.append(
                {
                    **row,
                    "event_at": dt.datetime(2025, 7, 31, 9),
                    "event": "baseline",
                    "experiment_id": None,
                    "group_id": None,
                    "store_id": None,
                    "revenue": None,
                }
            )
    rows.extend(baselines)
    con = ibis.duckdb.connect()
    con.create_table("cl_events", obj=pa.Table.from_pylist(rows))
    definitions = tmp_path / "observational-cluster.yaml"
    import yaml

    declared = yaml.safe_load(_DEFS_TEMPLATE.format(cluster_line="    cluster: store_id"))
    experiment = declared["experiments"][0]
    experiment.pop("allocation", None)
    experiment.pop("allocation_scheme", None)
    experiment["design"] = {
        "mechanism": "observational",
        "covariates": [{"property": "tenure", "source": "events"}],
    }
    declared["fact_sources"][0]["properties"] = [
        {"name": "tenure", "column": "tenure", "dtype": "float", "as_of": "pre_exposure"}
    ]
    definitions.write_text(yaml.safe_dump(declared))
    analysis_experiment, analysis_resolver = from_definitions(
        definitions, con, experiment_name="store_test", store="none"
    )
    try:
        from increment.breakout.estimates import LiftEstimates

        rows = view_readout(analysis_experiment, analysis_resolver)
        assert isinstance(rows, LiftEstimates)
        weighted = [row for row in rows if row.method == "iptw"]
        assert weighted
        assert all(row.weight_diagnostics_available is True for row in weighted)
        assert all(row.weight_grain == "cluster" for row in weighted)
        assert all(row.control_weight_n == N_STORES for row in weighted)
        assert all(row.treatment_weight_n == N_STORES for row in weighted)
        assert_support(
            "observational_cluster_weight_diagnostics",
            "definitions_cluster_weight_grain",
            weighted,
        )
    finally:
        cast(Any, analysis_resolver).close()
        con.disconnect()


def test_failed_weighted_metric_projects_unavailable_diagnostic_reason():
    from increment.estimation.adjust import weight_diagnostics_projection
    from increment.estimation.decision_types import ArmHypothesisKey, DecisionFailure

    failure = DecisionFailure(
        ArmHypothesisKey("rps", "treatment", "ate"),
        "estimation.adjust_common.supported_ratio_metric",
        {"method": "iptw", "metric": "rps", "role": "secondary"},
    )
    projection = weight_diagnostics_projection("iptw", failure=failure)

    assert projection["weight_diagnostics_available"] is False
    assert projection["weight_diagnostics_reason_code"] == failure.code
    assert projection["weight_diagnostics_reason_context"] == failure.context
    assert projection["weight_definition"] is None
    assert projection["weight_grain"] is None
    assert projection["control_weight_n"] is None
    assert projection["treatment_weight_n"] is None
    assert projection["control_weight_ess"] is None
    assert projection["treatment_weight_ess"] is None
    assert projection["control_weight_max_share"] is None
    assert projection["treatment_weight_max_share"] is None


def test_parity_harness_rejects_missing_or_unmatched_source_identity():
    from types import SimpleNamespace

    from tests.parity_harness.runner import _validate_source_snapshot_ids

    source = SimpleNamespace(source_snapshot_id="scope-id")

    class Results(list):
        metadata = SimpleNamespace(
            scope=SimpleNamespace(by_source={"scope-id": source}, families=())
        )

    with pytest.raises(AssertionError, match="without source_snapshot_id"):
        _validate_source_snapshot_ids("diagnostics", "summary", Results([SimpleNamespace()]))
    with pytest.raises(AssertionError, match="does not match retained scope metadata"):
        _validate_source_snapshot_ids(
            "diagnostics",
            "summary",
            Results([SimpleNamespace(source_snapshot_id="other-id")]),
        )

    class ResultsWithoutMetadata(list):
        metadata = None

    with pytest.raises(AssertionError, match="without retained source-scope metadata"):
        _validate_source_snapshot_ids(
            "diagnostics",
            "summary",
            ResultsWithoutMetadata([SimpleNamespace(source_snapshot_id="scope-id")]),
        )
    _validate_source_snapshot_ids(
        "diagnostics",
        "summary",
        Results([SimpleNamespace(source_snapshot_id="scope-id", metric="m", group_id="treatment")]),
    )
