from pathlib import Path

from scripts.selection_probe import ProbeRow, _parse_collected, probe


def test_probe_reports_one_row_per_module_in_order():
    rows = probe(Path("."), ["a.py", "b.py"], runner=lambda root, m: ProbeRow(m, 1, 2, 3))
    assert [r.module for r in rows] == ["a.py", "b.py"]


def test_collected_line_is_parsed():
    assert _parse_collected("13897/19527 tests collected (5630 deselected) in 11.94s") == (13897, 19527)
