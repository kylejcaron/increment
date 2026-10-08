"""All six ingress classifications and four supported diagnostic paths."""

from dataclasses import replace

import pytest

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
    assert result.refusals["from_moments"] == "source.moments.covariate_unavailable"
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
    assert result.refusals["from_moments"] == "source.moments.covariate_unavailable"
    assert "from_switchback_panel" in case.waive


def test_conversion_weight_diagnostics_are_retained_on_unit_summary():
    import pandas as pd

    from increment import Analysis
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
    analysis = Analysis.from_unit_summary(
        frame,
        unit="unit",
        group="group",
        metrics=[MetricSpec(name="converted", type="conversion")],
        design=Observational(
            control_group="control", adjustment=AdjustmentSet(covariates=("tenure",))
        ),
        plan=AnalysisPlan(primary="converted"),
    )
    try:
        rows = analysis.run()
        assert isinstance(rows, LiftEstimates)
        weighted = [row for row in rows if row.method == "iptw"]
        assert weighted
        assert all(row.weight_diagnostics_available is True for row in weighted)
        assert all(row.weight_grain == "unit" for row in weighted)
        assert all(row.control_weight_n == 60 for row in weighted)
        assert all(row.treatment_weight_n == 60 for row in weighted)
    finally:
        analysis.close()


def test_cluster_grain_weight_diagnostics_are_retained_on_unit_summary():
    import pandas as pd

    from increment import Analysis
    from increment.breakout.estimates import LiftEstimates
    from increment.frame import MetricSpec
    from increment.semantics.design import AdjustmentSet, Observational
    from increment.semantics.models import AnalysisPlan

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
    analysis = Analysis.from_unit_summary(
        frame,
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
        results = analysis.run()
        assert isinstance(results, LiftEstimates)
        rows = [row for row in results if row.method == "iptw"]
        assert rows
        assert all(row.weight_diagnostics_available is True for row in rows)
        assert all(row.weight_grain == "cluster" for row in rows)
        assert all(row.control_weight_n == 20 for row in rows)
        assert all(row.treatment_weight_n == 20 for row in rows)
    finally:
        analysis.close()


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
