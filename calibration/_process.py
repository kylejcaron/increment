"""Run a calibration subprocess with an explicit campaign timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence


def _stop_process_group(process: subprocess.Popen[bytes], *, grace_seconds: float = 1.0) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        control_break = getattr(signal, "CTRL_BREAK_EVENT", None)
        if control_break is None:
            process.terminate()
        else:
            try:
                process.send_signal(control_break)
            except (OSError, ValueError):
                process.terminate()
    try:
        process.wait(timeout=grace_seconds)
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
    """Return child status or 124 on timeout; forward cancellation to nested workers."""
    process: subprocess.Popen[bytes] | None = None
    forwarded_signals = [signal.SIGTERM]
    if os.name == "nt" and hasattr(signal, "SIGBREAK"):
        forwarded_signals.append(signal.SIGBREAK)
    previous_handlers = {signum: signal.getsignal(signum) for signum in forwarded_signals}

    def stop_child(signum: int, _frame: object) -> None:
        if process is not None:
            _stop_process_group(process, grace_seconds=0.2)
        raise SystemExit(128 + signum)

    for signum in forwarded_signals:
        signal.signal(signum, stop_child)
    try:
        process = subprocess.Popen(
            command,
            start_new_session=os.name == "posix",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        try:
            return process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _stop_process_group(process)
            print(
                f"campaign subprocess exceeded {timeout_seconds:g} seconds",
                file=sys.stderr,
            )
            return 124
        except BaseException:
            _stop_process_group(process)
            raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
