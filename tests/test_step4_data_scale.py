from __future__ import annotations

import pytest

from tests.parity_harness.matrix_data import scaled_units_per_arm, units


def test_weekly_matrix_uses_ten_x_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scale switch adds units per arm without changing assignment proportions."""
    monkeypatch.setenv("INCREMENT_PARITY_DATA_SCALE", "1")
    normal = units()
    monkeypatch.setenv("INCREMENT_PARITY_DATA_SCALE", "10")
    weekly = units()
    assert scaled_units_per_arm() == 600
    assert len(weekly) == 10 * len(normal)
    assert sum(unit.arm == "control" for unit in weekly) == 600
    assert sum(unit.arm == "treatment" for unit in weekly) == 600
    assert [(unit.arm, unit.index) for unit in normal] == [
        (unit.arm, unit.index) for unit in weekly if unit.index < 60
    ]
    assert scaled_units_per_arm(scale=1, base=10_000) == 10_000
    assert scaled_units_per_arm(scale=10, base=1_000_000) == 10_000_000


def test_weekly_matrix_scales_switchback_schedule_tenfold(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.parity_harness.matrix_cases import switchback_frame

    monkeypatch.setenv("INCREMENT_PARITY_DATA_SCALE", "1")
    normal = switchback_frame()
    monkeypatch.setenv("INCREMENT_PARITY_DATA_SCALE", "10")
    weekly = switchback_frame()
    assert len(weekly) == 10 * len(normal)
    assert weekly["unit"].nunique() == 10 * normal["unit"].nunique()
