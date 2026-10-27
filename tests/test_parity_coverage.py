from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.parity_harness import matrix

ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "tests/parity_harness/COVERAGE.md"


@pytest.mark.integration
def test_coverage_renderer_matches_checked_in_coverage() -> None:
    rendered = subprocess.run(
        [sys.executable, "scripts/render_parity_coverage.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert rendered.returncode == 0, rendered.stdout + rendered.stderr


def test_coverage_renderer_check_fails_for_stale_file(tmp_path: Path) -> None:
    stale = tmp_path / "COVERAGE.md"
    stale.write_text("stale\n")
    from scripts.render_parity_coverage import check_coverage

    assert check_coverage(stale) is False


def test_coverage_source_is_the_complete_matrix() -> None:
    cells = tuple(matrix.iter_cells())
    assert len(cells) == 3584
    assert all(matrix.classify(cell).legs for cell in cells)
