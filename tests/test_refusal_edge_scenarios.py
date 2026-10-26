from __future__ import annotations

import pytest

from tests.parity_harness.cases import PARITY_CASES
from tests.parity_harness.runner import run_case


@pytest.mark.parametrize(
    ("scenario_id", "expected_code"),
    [
        ("edge-empty-arm", "readout.arms.no_treatment"),
        ("edge-zero-variance-arm", "readout.estimate_lift_every"),
        ("edge-near-zero-ratio-denominator", "readout.estimate_lift_every"),
        ("edge-nonfinite-ratio-denominator", "source.frame.non_finite"),
    ],
)
def test_special_moment_edge_scenarios_raise_exact_public_codes(
    scenario_id: str, expected_code: str
) -> None:
    case = next(case for case in PARITY_CASES if case.id == scenario_id)
    result = run_case(case)
    assert result.refusals == {"from_unit_summary": expected_code}


def test_tiny_positive_ratio_denominator_is_a_valid_public_edge_case() -> None:
    case = next(
        case for case in PARITY_CASES if case.id == "edge-near-zero-positive-ratio-denominator"
    )

    result = run_case(case)

    assert result.refusals == {}
    assert result.rows["from_unit_summary"]
