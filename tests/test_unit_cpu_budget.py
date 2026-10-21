"""Subprocess checks for the opt-in unit CPU-time budget."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.unit_cpu_baseline import baseline_nodeids

pytestmark = pytest.mark.integration

ROOT = Path(__file__).parents[1]


def _pytest_case(
    tmp_path: Path, test_body: str, baseline: str = "", *, enforce: bool = False
) -> subprocess.CompletedProcess[str]:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    test_file = tests_dir / "test_budget_case.py"
    test_file.write_text(test_body)
    baseline_file = tmp_path / "baseline.txt"
    baseline_file.write_text(baseline.replace("test_budget_case.py", "tests/test_budget_case.py"))
    env = os.environ.copy()
    env["UNIT_CPU_BASELINE"] = str(baseline_file)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), env.get("PYTHONPATH", "")))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--confcutdir",
            str(tmp_path),
            "-q",
            "-m",
            "",
            "--unit-cpu-budget",
            *(["--unit-cpu-budget-enforce"] if enforce else []),
            str(test_file),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_unbaselined_expensive_unit_test_fails_with_remediation(tmp_path: Path) -> None:
    """An expensive new unit test must fail with a boundary/efficiency fix."""
    result = _pytest_case(
        tmp_path,
        "def test_expensive():\n    sum(i * i for i in range(10_000_000))\n",
        enforce=True,
    )
    assert result.returncode != 0
    assert "unit CPU budget" in result.stdout + result.stderr
    assert "slow" in result.stdout + result.stderr


def test_baselined_expensive_unit_test_passes(tmp_path: Path) -> None:
    """Existing expensive unit tests pass while listed in the committed baseline."""
    result = _pytest_case(
        tmp_path,
        "def test_expensive():\n    sum(i * i for i in range(10_000_000))\n",
        "test_budget_case.py::test_expensive\n",
        enforce=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_slow_or_integration_test_is_exempt(tmp_path: Path) -> None:
    """A deliberate slow/integration boundary is not subject to unit budgets."""
    result = _pytest_case(
        tmp_path,
        "import pytest\n@pytest.mark.integration\ndef test_expensive():\n    sum(i * i for i in range(10_000_000))\n",
        enforce=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_slow_marked_test_is_exempt(tmp_path: Path) -> None:
    """A deliberately slow unit-selected test is exempt from enforcement."""
    result = _pytest_case(
        tmp_path,
        "import pytest\n@pytest.mark.slow\ndef test_expensive():\n    sum(i * i for i in range(10_000_000))\n",
        enforce=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_baseline_entry_below_budget_is_reported_stale(tmp_path: Path) -> None:
    """A baseline entry that no longer exceeds the budget is reported for removal."""
    result = _pytest_case(
        tmp_path,
        "def test_cheap():\n    assert 1 + 1 == 2\n",
        "test_budget_case.py::test_cheap\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stale unit CPU baseline" in result.stdout + result.stderr


def test_report_mode_lists_unbaselined_expensive_unit_test(tmp_path: Path) -> None:
    """Report mode lists an over-budget unit test but does not fail it."""
    result = _pytest_case(
        tmp_path,
        "def test_expensive():\n    sum(i * i for i in range(10_000_000))\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unit CPU over budget:" in result.stdout


def test_import_warmup_cost_is_reported(tmp_path: Path) -> None:
    """The session-start public-export warm-up reports its process CPU cost."""
    result = _pytest_case(tmp_path, "def test_quick():\n    assert 1 + 1 == 2\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "unit CPU import warm-up" in result.stdout


def test_baseline_hysteresis_records_any_run_over_half_budget() -> None:
    """A 50 ms crossing in any of three samples stays in the baseline."""
    samples = [
        {"intermittent": 0.051, "below": 0.049},
        {"intermittent": 0.02, "below": 0.05},
        {"intermittent": 0.01},
    ]
    assert baseline_nodeids(samples) == {"intermittent"}


def test_worker_down_without_workeroutput_skips_warmup_data() -> None:
    """A crashed xdist worker can go down before workeroutput exists."""
    from types import SimpleNamespace

    import conftest

    conftest.pytest_testnodedown(SimpleNamespace(gateway=SimpleNamespace(id="gw-crashed")), None)
