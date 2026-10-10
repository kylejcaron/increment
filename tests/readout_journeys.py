"""Reusable checks for ordinary readout journeys."""

import pytest

from increment.breakout.estimates import LiftEstimates
from increment.estimation.decision_types import PValueEvidence
from increment.estimation.readout_types import CellKey


def assert_failed_assigned_integrity(results: LiftEstimates) -> None:
    assert results.metadata is not None
    integrity = tuple(
        item for source in results.metadata.scope.by_source.values() for item in source.integrity
    )
    assert any(
        item.status == "failed"
        and item.construction == "always_valid"
        and item.alpha == 0.001
        and item.analysis_population == "assigned"
        and item.observed == {"control": 900, "treatment": 100}
        for item in integrity
    )


def assert_prior_free_sampling_family(results: LiftEstimates) -> None:
    assert results.metadata is not None
    family_members = {cell for family in results.metadata.scope.families for cell in family.members}
    assert family_members
    assert CellKey.from_row(results[0]) in family_members
    for row in results:
        cell = CellKey.from_row(row)
        if cell not in family_members:
            continue
        assert row.source_snapshot_id is not None
        record = results.metadata.record(row.source_snapshot_id, cell)
        assert record.sampling is not None and isinstance(record.sampling.evidence, PValueEvidence)
        p_value = row.p_value()
        assert p_value is not None
        assert record.sampling.evidence.p_value == pytest.approx(p_value)


def assert_filtered_family_scope_unchanged(
    original: LiftEstimates, filtered: LiftEstimates
) -> None:
    assert original.metadata is not None and filtered.metadata is not None
    assert filtered.metadata.scope.families == original.metadata.scope.families
