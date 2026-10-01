from __future__ import annotations

import pytest
from pydantic import ValidationError

from increment._compatibility_inspector import (
    CompatibilityFinding,
    CompatibilityReport,
    FamilyStatus,
    RuntimeStatus,
    aggregate_family,
    aggregate_overall,
    default_support_observe,
)
from increment.compatibility import Supported, UnitFloor, Unsupported
from increment.errors import InvalidRequestError


def test_module_docstring_explains_mechanical_observation_and_precedence():
    """The private inspector module must document
    that observation is mechanical (no domain predicates) and that
    aggregation precedence is fixed."""
    import increment._compatibility_inspector as inspector_module

    doc = (inspector_module.__doc__ or "").lower()
    assert doc.strip(), "increment._compatibility_inspector has no module docstring"
    assert "mechanical" in doc
    assert "precedence" in doc


def _finding(
    runtime: RuntimeStatus,
    family: FamilyStatus = "not_applicable",
) -> CompatibilityFinding:
    return CompatibilityFinding(
        capability="test",
        runtime=runtime,
        family=family,
        refusal_code=None,
        assumptions=(),
        floor=None,
        warning=None,
        reference=None,
        evidence_source="contract",
    )


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("refused", "unknown"), "refused"),
        (("unknown", "limited"), "unknown"),
        (("limited", "supported"), "limited"),
        (("supported", "not_applicable"), "supported"),
        (("not_applicable", "not_applicable"), "not_applicable"),
    ],
)
def test_runtime_aggregation_precedence(statuses, expected):
    findings = tuple(_finding(status) for status in statuses)
    assert aggregate_overall(findings) == expected


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (("excluded", "unknown"), "excluded"),
        (("unknown", "participates"), "unknown"),
        (("participates", "not_applicable"), "participates"),
        (("not_applicable", "not_applicable"), "not_applicable"),
    ],
)
def test_family_aggregation_precedence(statuses, expected):
    findings = tuple(_finding("supported", status) for status in statuses)
    assert aggregate_family(findings) == expected


def test_support_observer_preserves_contract_metadata():
    support = Supported(
        assumptions=("independent_units",),
        floor=UnitFloor(minimum_per_arm=2),
        reference="fixed_horizon",
    )
    finding = default_support_observe(
        support,
        capability="arm_moments",
        evidence_source="contract",
    )
    assert finding.assumptions == support.assumptions
    assert finding.floor == support.floor
    assert finding.reference == support.reference


def test_unsupported_observer_preserves_refusal_code():
    finding = default_support_observe(
        Unsupported("arm.metric.quantile_cuped"),
        capability="arm_moments",
        evidence_source="contract",
    )
    assert finding.runtime == "refused"
    assert finding.refusal_code == "arm.metric.quantile_cuped"
    assert finding.floor is None


def test_report_rejects_incorrect_caller_aggregates():
    finding = _finding("limited", "excluded")
    with pytest.raises(InvalidRequestError) as exc_info:
        CompatibilityReport(
            overall="supported",
            overall_family="participates",
            findings=(finding,),
        )
    assert (
        exc_info.value.code
        == "facade.compatibility_inspector.compatibility_report.overall_fields_equal"
    )


def test_findings_are_immutable():
    finding = _finding("limited", "excluded")
    with pytest.raises(ValidationError):
        finding.runtime = "supported"  # ty: ignore[invalid-assignment]
