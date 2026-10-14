"""Run a calibration subprocess with an explicit campaign timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence


def _stop_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()
        process.wait()


def run_with_timeout(command: Sequence[str], timeout_seconds: float) -> int:
    """Return child status, or 124 on timeout; terminate spawned workers as well."""
    process = subprocess.Popen(
        command,
        start_new_session=os.name == "posix",
        creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
    )
    try:
        return process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        _stop_process_group(process)
        print(f"campaign subprocess exceeded {timeout_seconds:g} seconds", file=sys.stderr)
        return 124
    except BaseException:
        _stop_process_group(process)
        raise
