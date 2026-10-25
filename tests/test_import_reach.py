from __future__ import annotations

import pytest

from scripts.import_reach import measure


def test_measure_counts_increment_modules_in_a_fresh_process():
    rows = measure(["increment", "increment._evidence_types"])
    by_name = {row.entry_point: row for row in rows}
    assert by_name["increment"].modules == 1
    assert by_name["increment._evidence_types"].modules <= 16
    assert "increment._evidence_types" in by_name["increment._evidence_types"].loaded


def test_absent_entry_point_is_reported_not_raised():
    (row,) = measure(["increment.not_a_module"])
    assert row.modules is None and row.loaded == ()
pytestmark = pytest.mark.integration
