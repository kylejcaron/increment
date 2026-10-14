"""Calibration subprocess timeout behavior."""

from __future__ import annotations

import sys

import pytest

from calibration._process import run_with_timeout


@pytest.mark.integration
def test_calibration_subprocess_returns_child_status() -> None:
    assert run_with_timeout([sys.executable, "-c", "raise SystemExit(7)"], 10) == 7


@pytest.mark.integration
def test_calibration_subprocess_timeout_returns_124() -> None:
    assert run_with_timeout([sys.executable, "-c", "import time; time.sleep(10)"], 0.1) == 124
