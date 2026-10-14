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
    for name in ("MAKEFLAGS", "MAKELEVEL", "MFLAGS", "PYTEST_ARGS", "PYTEST_ADDOPTS"):
        environment.pop(name, None)
    result = subprocess.run(
        [
            "make",
            "test",
            f"TESTS={Path(__file__).relative_to(ROOT)}::test_explicit_slow_test_target",
            "PYTEST_ARGS=",
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
    leaked_arguments = (
        "-k test_make_test_runs_explicit_slow_selector",
        "--splits",
        "--durations-path",
        "--junitxml=/tmp/task5-outer-junit.xml",
        "--cov=increment",
    )
    assert not any(argument in result.stdout for argument in leaked_arguments), result.stdout


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
