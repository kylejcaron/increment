"""Regression checks for focused Makefile test launches."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.slow
def test_explicit_slow_test_target() -> None:
    assert Path(__file__).parent == ROOT / "tests"


def test_make_test_runs_explicit_slow_selector() -> None:
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            "make",
            "test",
            f"TESTS={Path(__file__).relative_to(ROOT)}::test_explicit_slow_test_target",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_test_affected_forwards_user_pytest_args() -> None:
    result = subprocess.run(
        ["make", "-n", "test-affected", "BASE=HEAD", "PYTEST_ARGS=-k selected"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "-- -n " in result.stdout
    assert "--dist loadgroup -k selected" in result.stdout
