from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_collect_count import _check_fanout, _collected_items, check_counts, main

ROOT = Path(__file__).resolve().parents[1]


def test_count_above_baseline_fails_with_pr_line(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.write_text("unit 10\n")
    status, output = check_counts("unit", baseline, 13)
    assert status == 1
    assert "Tier: unit\nObserved: 13 items\nBaseline: 10 items\nDelta: +3 items" in output
    assert "tests: +3 items — <reason>" in output
    assert "Literal or computed parameter grids over 12" not in output


def test_count_at_or_below_baseline_passes_and_prints_lower_count(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.write_text("unit 10\n")
    status, output = check_counts("unit", baseline, 8)
    assert status == 0
    assert "lower count" in output
    assert "unit 8" in output


def test_collection_summary_uses_selected_count_when_tests_are_deselected() -> None:
    assert _collected_items("11351/19954 tests collected (8603 deselected)") == 11351


def test_count_increase_still_reports_actual_over_cap_fanout(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_dynamic.py").write_text(
        "import pytest\n@pytest.mark.parametrize('value', values)\ndef test_computed(value): pass\n"
    )
    baseline = tmp_path / "baseline"
    baseline.write_text("unit 10\n")
    nodeids = "".join(f"tests/test_dynamic.py::test_computed[{i}]\n" for i in range(13))

    status, output = check_counts("unit", baseline, 13, nodeids, tests_root)

    assert status == 1
    assert (
        "Collected parameter fan-out findings (informational; they do not fail this check):"
        in output
    )
    assert "tests/test_dynamic.py::test_computed: 13 collected items" in output
    assert (
        "Literal or computed parameter grids over 12 collected items require a reasoned grid marker."
        in output
    )
    failure = output[output.index("Count check failed:") :]
    assert "PR description" in failure
    assert "`tests: +3 items — <reason>`" in failure


def test_actual_class_method_nodeid_honors_reasoned_grid_marker(tmp_path: Path) -> None:
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_grid.py").write_text(
        "import pytest\n"
        "class TestGrid:\n"
        "    @pytest.mark.grid('large sweep')\n"
        "    def test_marked(self): pass\n"
    )
    relpath = os.path.relpath(tests_root / "test_grid.py", ROOT)
    nodeids = "".join(f"{relpath}::TestGrid::test_marked[{i}]\n" for i in range(13))

    assert _check_fanout(nodeids, tests_root) == []


def test_explicit_items_does_not_read_stdin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = tmp_path / "baseline"
    baseline.write_text("unit 10\n")

    class NoRead:
        def read(self) -> str:
            raise AssertionError("stdin must not be read when --items is supplied")

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_collect_count", "--tier", "unit", "--baseline", str(baseline), "--items", "8"],
    )
    monkeypatch.setattr(sys, "stdin", NoRead())
    assert main() == 0
    assert "unit 8 (baseline 10; lower count — lower the baseline)" in capsys.readouterr().out


@pytest.mark.integration
class TestMakeCheckItems:
    def test_make_collects_and_filters_actual_grid_fanout(self, tmp_path: Path) -> None:
        scratch = tmp_path / "tests"
        scratch.mkdir()
        (scratch / "test_grid.py").write_text(
            "import pytest\n"
            "@pytest.mark.parametrize('value', range(13))\n"
            "def test_unmarked(value): pass\n"
            "@pytest.mark.parametrize('value', range(13))\n"
            "@pytest.mark.grid('large sweep')\n"
            "def test_marked(value): pass\n"
        )
        baseline = tmp_path / "baseline"
        baseline.write_text("unit 26\n")
        result = subprocess.run(
            [
                "make",
                "check-items",
                f"CHECK_ITEMS_TESTS={scratch}",
                f"CHECK_ITEMS_BASELINE={baseline}",
                "CHECK_ITEMS_TIERS=unit",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr + result.stdout
        assert "unit 26 (baseline 26)" in result.stdout
        assert "test_unmarked: 13 collected items" in result.stdout
        assert "test_marked" not in result.stdout

    def test_plain_make_defaults_to_install(self) -> None:
        result = subprocess.run(
            ["make", "--dry-run"], cwd=ROOT, text=True, capture_output=True, check=False
        )
        assert result.returncode == 0
        assert "uv sync --group dev --extra demo --extra tables --extra dashboard" in result.stdout
