"""Reduced checks of the preregistered mixed-covariate certification cells."""

from __future__ import annotations

from collections import Counter


def test_revision_seven_manifest_has_shared_family_allocation():
    from calibration.cuped_linear import ETA, FAMILY_SIZE, FULL_REPLICATIONS, MANIFEST
    from tests.mc import family_eta

    assert FAMILY_SIZE == 67
    assert ETA == family_eta(0.01, 67)
    assert FULL_REPLICATIONS == 107_100
    assert len(MANIFEST) == 55
    assert Counter(cell.statistic for cell in MANIFEST) == {"coverage": 46, "fcp": 9}
    assert len({cell.id for cell in MANIFEST}) == 55
    assert [cell.id for cell in MANIFEST if cell.id.startswith("D4-")] == [
        f"D4-{i}" for i in range(1, 12)
    ]
    assert all(cell.failure_consequence for cell in MANIFEST)
