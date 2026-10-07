"""All six ingress classifications and four supported diagnostic paths."""

from dataclasses import replace

import pytest

from tests.parity_harness.cases import (
    _observational_aipw_dml_case,
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
