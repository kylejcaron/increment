"""Calibration subprocess timeout behavior."""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

from calibration._process import run_with_timeout


@pytest.mark.integration
def test_outer_timeout_stops_nested_worker_process(tmp_path: Path) -> None:
    worker_pid_file = tmp_path / "worker.pid"
    controller = tmp_path / "controller.py"
    worker_code = (
        "import os,pathlib,signal,sys,time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()));"
        "time.sleep(60)"
    )
    controller.write_text(
        "import sys\n"
        "from calibration._process import run_with_timeout\n"
        "run_with_timeout([sys.executable, '-c', "
        f"{worker_code!r}, sys.argv[1]], 60)\n"
    )

    assert run_with_timeout([sys.executable, str(controller), str(worker_pid_file)], 2) == 124
    worker_pid = int(worker_pid_file.read_text())
    try:
        with pytest.raises(OSError):
            os.kill(worker_pid, 0)
    finally:
        try:
            os.kill(worker_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError:
            pass


@pytest.mark.integration
def test_calibration_subprocess_returns_child_status() -> None:
    assert run_with_timeout([sys.executable, "-c", "raise SystemExit(7)"], 10) == 7


@pytest.mark.integration
def test_calibration_subprocess_timeout_returns_124() -> None:
    assert run_with_timeout([sys.executable, "-c", "import time; time.sleep(10)"], 0.1) == 124
