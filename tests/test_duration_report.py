from pathlib import Path

import pytest

from scripts.duration_report import SlowItem, main, over_budget


def test_lists_items_over_budget_slowest_first(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase classname="tests.test_example" '
        'file="tests/test_example.py" name="test_medium" time="6"/>'
        '<testcase classname="tests.test_example.TestBoundary" '
        'file="tests/test_example.py" name="test_slow[value]" time="12"/>'
        '<testcase classname="tests.test_example" '
        'name="test_fallback" time="8"/></testsuite></testsuites>'
    )

    assert over_budget(junit) == [
        SlowItem("tests/test_example.py::TestBoundary::test_slow[value]", 12.0),
        SlowItem("tests/test_example.py::test_fallback", 8.0),
        SlowItem("tests/test_example.py::test_medium", 6.0),
    ]
    assert main([str(junit)]) == 0
    summary = capsys.readouterr().out
    assert "tests/test_example.py::TestBoundary::test_slow[value]" in summary
    assert summary.index("test_slow[value]") < summary.index("test_fallback")


def test_reports_nothing_when_all_items_fit(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_example" name="test_at_budget" '
        'time="2"/><testcase classname="tests.test_example" name="test_below" '
        'time="1.9"/></testsuite>'
    )

    assert over_budget(junit, budget_s=2.0) == []


def test_skipped_items_are_ignored(tmp_path: Path):
    junit = tmp_path / "junit.xml"
    junit.write_text(
        '<testsuite><testcase classname="tests.test_example" name="test_skipped" '
        'time="20"><skipped/></testcase></testsuite>'
    )

    assert over_budget(junit) == []
