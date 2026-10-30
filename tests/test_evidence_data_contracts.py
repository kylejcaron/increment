from __future__ import annotations


def test_public_compliance_contract_round_trips_wire_values():
    from increment.evidence import ComplianceSummary
    from increment.evidence import ComplianceSummary as PackageComplianceSummary
    from increment.sources import ComplianceArm

    summary = ComplianceSummary(
        "study",
        "clicked",
        7,
        True,
        None,
        None,
        (ComplianceArm("treatment", 20, 10), ComplianceArm("control", 20, 0)),
    )

    assert PackageComplianceSummary is ComplianceSummary
    assert ComplianceSummary.from_wire(summary.to_wire(), study_id="study") == summary
