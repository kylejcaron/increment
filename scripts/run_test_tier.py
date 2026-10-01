"""Run a named pytest tier with a hard wall-clock budget."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    import pytest

_TIMEOUT_STATUS = 124


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--focused-file-defaults", action="store_true")
    parser.addoption("--focused-path-validation", action="store_true")


def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """Refuse a focused run naming any selector that is not an existing test file."""
    if not config.getoption("focused_path_validation"):
        return None
    base = config.invocation_params.dir
    rejected = [
        selector for selector in config.args if not (base / selector.split("::", 1)[0]).is_file()
    ]
    if rejected:
        names = ", ".join(repr(selector) for selector in rejected)
        print(
            f"pytest: error: focused selectors must be existing test files: {names}",
            file=sys.stderr,
        )
        return 4
    return None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep expensive tests only when their node or marker was explicitly selected."""
    if not config.getoption("focused_file_defaults"):
        return
    explicit: dict[Path, list[str]] = {}
    for selector in config.args:
        file, separator, node = selector.partition("::")
        if separator:
            explicit.setdefault(Path(file).resolve(), []).append(node)
    paths: dict[Path, Path] = {}
    kept, deselected = [], []
    for item in items:
        if not any(item.get_closest_marker(mark) for mark in ("slow", "parameter_recovery")):
            kept.append(item)
            continue
        if item.path not in paths:
            paths[item.path] = item.path.resolve()
        node = item.nodeid.partition("::")[2]
        selected = any(
            node == requested
            or node.startswith(requested + "[")
            or node.startswith(requested + "::")
            for requested in explicit.get(paths[item.path], ())
        )
        (kept if selected else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def _finalize_unfinished_evidence(
    command: Sequence[str], status: int, owner_token: str, termination: str
) -> None:
    """Finalize owned evidence when process termination bypasses pytest hooks."""
    root: Path | None = None
    for index, argument in enumerate(command):
        if argument == "--evidence-root" and index + 1 < len(command):
            root = Path(command[index + 1])
            break
        if argument.startswith("--evidence-root="):
            root = Path(argument.partition("=")[2])
            break
    if root is None:
        return
    root = root.resolve()
    if not root.is_dir():
        return
    for run in root.glob("run-*"):
        destination = run / "run.json"
        if not destination.is_file():
            continue
        try:
            metadata = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(metadata, dict):
            continue
        if metadata.get("status") != "running" or metadata.get("owner_token") != owner_token:
            continue
        metadata.update(
            status="finished",
            exit_code=status,
            termination=termination,
        )
        temporary = run / "run.json.tmp"
        try:
            temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            temporary.replace(destination)
        except OSError:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class Tier:
    marker: str | None
    budget_seconds: int


TIERS = {
    "focused": Tier(marker="", budget_seconds=180),
    "fast": Tier(marker=None, budget_seconds=240),
    "slow": Tier(
        marker="slow and not parameter_recovery and not examples",
        budget_seconds=480,
    ),
    # Supports optional pytest-split shards while retaining a hard budget.
    "parameter-recovery": Tier(marker="parameter_recovery", budget_seconds=1350),
    "examples": Tier(marker="examples", budget_seconds=300),
    "all": Tier(marker="", budget_seconds=1800),
    # Same as `all` minus the Monte-Carlo tier, which runs once (sharded)
    # instead of once per Python version/dependency-floor cell.
    "all-except-parameter-recovery": Tier(marker="not parameter_recovery", budget_seconds=1800),
}


def build_pytest_command(tier: Tier, extra_args: Sequence[str]) -> list[str]:
    # External evidence directories must not change configuration or test selection.
    config = Path(__file__).resolve().parents[1] / "pyproject.toml"
    command = [sys.executable, "-m", "pytest", "-c", str(config), "-p", "tests._evidence"]
    if tier.marker is not None:
        command.extend(("-m", tier.marker))
    command.extend(extra_args)
    return command


class _Interrupted(Exception):
    def __init__(self, signum: int) -> None:
        self.signum = signum


class _WindowsJobAccounting(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_longlong),
        ("TotalKernelTime", ctypes.c_longlong),
        ("ThisPeriodTotalUserTime", ctypes.c_longlong),
        ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
        ("TotalPageFaultCount", ctypes.c_ulong),
        ("TotalProcesses", ctypes.c_ulong),
        ("ActiveProcesses", ctypes.c_ulong),
        ("TotalTerminatedProcesses", ctypes.c_ulong),
    ]


@cache
def _windows_kernel32() -> ctypes.CDLL:
    if sys.platform != "win32":
        raise OSError("Windows process control is unavailable on this platform")
    return ctypes.WinDLL("kernel32", use_last_error=True)


def _windows_error() -> OSError:
    if sys.platform != "win32":
        raise OSError("Windows process control is unavailable on this platform")
    return ctypes.WinError(ctypes.get_last_error())


def _create_windows_job(process: subprocess.Popen[bytes]) -> int:
    kernel32 = _windows_kernel32()
    create_job = kernel32.CreateJobObjectW
    create_job.restype = ctypes.c_void_p
    job = create_job(None, None)
    if not job:
        raise _windows_error()
    if not kernel32.AssignProcessToJobObject(
        ctypes.c_void_p(job), ctypes.c_void_p(int(vars(process)["_handle"]))
    ):
        error = _windows_error()
        kernel32.CloseHandle(ctypes.c_void_p(job))
        raise error
    return int(job)


def _windows_job_exists(job: int) -> bool:
    accounting = _WindowsJobAccounting()
    if not _windows_kernel32().QueryInformationJobObject(
        ctypes.c_void_p(job),
        1,
        ctypes.byref(accounting),
        ctypes.sizeof(accounting),
        None,
    ):
        raise _windows_error()
    return accounting.ActiveProcesses > 0


def _terminate_windows_job(job: int) -> None:
    if not _windows_kernel32().TerminateJobObject(ctypes.c_void_p(job), 1):
        raise _windows_error()


def _close_windows_job(job: int) -> None:
    _windows_kernel32().CloseHandle(ctypes.c_void_p(job))


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass


def _start_process(
    command: Sequence[str], *, env: dict[str, str]
) -> tuple[subprocess.Popen[bytes], int | None, int | None]:
    if os.name == "nt":
        read_gate, write_gate = os.pipe()
        msvcrt = __import__("msvcrt")
        read_handle = msvcrt.get_osfhandle(read_gate)
        vars(os)["set_handle_inheritable"](read_handle, True)
        gated_command = [
            sys.executable,
            "-c",
            (
                "import msvcrt,os,subprocess,sys; "
                "gate=msvcrt.open_osfhandle(int(sys.argv[1]),os.O_RDONLY); "
                "token=os.read(gate,1); os.close(gate); "
                "token==b'1' or sys.exit(125); "
                "raise SystemExit(subprocess.call(sys.argv[2:]))"
            ),
            str(read_handle),
            *command,
        ]
        process: subprocess.Popen[bytes] | None = None
        job: int | None = None
        try:
            creationflags = vars(subprocess)["CREATE_NEW_PROCESS_GROUP"]
            process = subprocess.Popen(
                gated_command,
                creationflags=creationflags,
                close_fds=False,
                env=env,
            )
            _close_fd(read_gate)
            job = _create_windows_job(process)
            return process, job, write_gate
        except BaseException:
            _close_fd(read_gate)
            _close_fd(write_gate)
            if process is not None:
                if job is not None:
                    _terminate_windows_job(job)
                    _close_windows_job(job)
                else:
                    process.kill()
                process.wait()
            raise
    read_gate, write_gate = os.pipe()
    gated_command = [
        sys.executable,
        "-c",
        (
            "import os,sys; "
            "gate=int(sys.argv[1]); token=os.read(gate,1); os.close(gate); "
            "token==b'1' or sys.exit(125); "
            "os.execvp(sys.argv[2],sys.argv[2:])"
        ),
        str(read_gate),
        *command,
    ]
    process = None
    try:
        process = subprocess.Popen(
            gated_command,
            start_new_session=True,
            pass_fds=(read_gate,),
            env=env,
        )
        _close_fd(read_gate)
        return process, None, write_gate
    except BaseException:
        _close_fd(read_gate)
        _close_fd(write_gate)
        if process is not None:
            process.kill()
            process.wait()
        raise


def _group_exists(process: subprocess.Popen[bytes], group_id: int) -> bool:
    if os.name == "nt":
        return process.poll() is None
    try:
        os.killpg(group_id, 0)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _signal_process_group(
    process: subprocess.Popen[bytes],
    group_id: int,
    signum: int,
    windows_job: int | None,
    *,
    force: bool = False,
) -> None:
    if os.name == "nt":
        if force:
            if windows_job is not None:
                _terminate_windows_job(windows_job)
            else:
                subprocess.run(
                    ("taskkill", "/PID", str(group_id), "/T", "/F"),
                    check=False,
                    capture_output=True,
                )
        elif process.poll() is None:
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        return
    try:
        os.killpg(group_id, signal.SIGKILL if force else signum)
    except ProcessLookupError:
        pass


def _stop_process_group(
    process: subprocess.Popen[bytes],
    group_id: int,
    signum: int,
    grace_seconds: float,
    windows_job: int | None,
) -> None:
    if os.name == "nt":
        _signal_process_group(process, group_id, signum, windows_job)
        grace_deadline = time.monotonic() + grace_seconds
        while (
            windows_job is not None
            and _windows_job_exists(windows_job)
            and time.monotonic() < grace_deadline
        ):
            time.sleep(0.01)
        if windows_job is not None and _windows_job_exists(windows_job):
            _signal_process_group(process, group_id, signum, windows_job, force=True)
        process.wait()
        return
    _signal_process_group(process, group_id, signum, windows_job)
    grace_deadline = time.monotonic() + grace_seconds
    while _group_exists(process, group_id) and time.monotonic() < grace_deadline:
        time.sleep(0.01)
    if _group_exists(process, group_id):
        _signal_process_group(process, group_id, signum, windows_job, force=True)
    try:
        process.wait(timeout=max(0.1, grace_seconds))
    except subprocess.TimeoutExpired:
        _signal_process_group(process, group_id, signum, windows_job, force=True)
        process.wait()


def _exit_status(returncode: int) -> int:
    return 128 + abs(returncode) if returncode < 0 else returncode


def _supervise_process(
    process: subprocess.Popen[bytes],
    *,
    tier: str,
    started: float,
    deadline: float,
    budget_seconds: float,
    heartbeat_seconds: float,
    grace_seconds: float,
    stream: TextIO,
    windows_job: int | None,
    check_interrupt: Callable[[], None],
) -> int:
    """Wait for *process*, emitting heartbeats and stopping its group at the deadline."""
    next_heartbeat = started + heartbeat_seconds
    while True:
        check_interrupt()
        now = time.monotonic()
        remaining = deadline - now
        if remaining <= 0:
            print(
                f"test tier timed out: tier={tier} budget={budget_seconds:g}s "
                f"elapsed={now - started:.1f}s",
                file=stream,
                flush=True,
            )
            _stop_process_group(process, process.pid, signal.SIGTERM, grace_seconds, windows_job)
            return _TIMEOUT_STATUS
        wait_seconds = min(remaining, 0.1, max(0.01, next_heartbeat - now))
        try:
            return _exit_status(process.wait(timeout=wait_seconds))
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            if next_heartbeat <= now < deadline:
                print(
                    f"test tier running: tier={tier} elapsed={now - started:.1f}s "
                    f"budget={budget_seconds:g}s",
                    file=stream,
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds


def run_with_budget(
    command: Sequence[str],
    *,
    tier: str,
    budget_seconds: float,
    heartbeat_seconds: float = 30.0,
    grace_seconds: float = 5.0,
    stream: TextIO = sys.stderr,
) -> int:
    """Run ``command`` in a process group and stop the group at its deadline."""
    started = time.monotonic()
    deadline = started + budget_seconds
    owner_token = uuid.uuid4().hex

    forwarded = (signal.SIGINT, signal.SIGTERM)
    forwarded += (vars(signal)["SIGBREAK"],) if os.name == "nt" else ()
    original_handlers = {signum: signal.getsignal(signum) for signum in forwarded}

    pending_signal: int | None = None

    def interrupt(signum: int, _frame: object) -> None:
        nonlocal pending_signal
        # Raising inside Popen.wait can strand its internal waitpid lock.
        if pending_signal is None:
            pending_signal = signum

    def _raise_pending() -> None:
        if pending_signal is not None:
            raise _Interrupted(pending_signal)

    for signum in forwarded:
        signal.signal(signum, interrupt)

    process: subprocess.Popen[bytes] | None = None
    windows_job: int | None = None
    startup_gate: int | None = None
    try:
        process, windows_job, startup_gate = _start_process(
            command,
            env={**os.environ, "INCREMENT_EVIDENCE_OWNER": owner_token},
        )
        _raise_pending()
        if startup_gate is not None:
            os.write(startup_gate, b"1")
            _close_fd(startup_gate)
            startup_gate = None

        status = _supervise_process(
            process,
            tier=tier,
            started=started,
            deadline=deadline,
            budget_seconds=budget_seconds,
            heartbeat_seconds=heartbeat_seconds,
            grace_seconds=grace_seconds,
            stream=stream,
            windows_job=windows_job,
            check_interrupt=_raise_pending,
        )
    except _Interrupted as interrupted:
        print(
            f"test tier interrupted: tier={tier} signal={interrupted.signum}",
            file=stream,
            flush=True,
        )
        if process is not None:
            _stop_process_group(
                process, process.pid, interrupted.signum, grace_seconds, windows_job
            )
        status = 128 + interrupted.signum
    finally:
        if startup_gate is not None:
            _close_fd(startup_gate)
        if process is not None:
            if os.name == "nt" and windows_job is not None:
                _terminate_windows_job(windows_job)
                _close_windows_job(windows_job)
            elif _group_exists(process, process.pid):
                _stop_process_group(
                    process,
                    process.pid,
                    signal.SIGTERM,
                    grace_seconds,
                    windows_job,
                )
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
    if pending_signal is not None:
        status = 128 + pending_signal
    termination = (
        "budget_timeout"
        if status == _TIMEOUT_STATUS
        else "interrupted"
        if status in (128 + signum for signum in forwarded)
        else "process_exit"
    )
    _finalize_unfinished_evidence(command, status, owner_token, termination)
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=tuple(TIERS))
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    tier = TIERS[args.tier]
    if args.tier == "focused":
        if not args.pytest_args:
            print(
                "pytest: error: focused requires a test FILE or FILE::NODE first, "
                "followed by pytest options",
                file=sys.stderr,
            )
            return 4
        target = Path(args.pytest_args[0].split("::", 1)[0])
        if not target.is_file() or target.suffix not in {".py", ".md"}:
            print(
                "pytest: error: focused requires a test FILE or FILE::NODE first, "
                "followed by pytest options",
                file=sys.stderr,
            )
            return 4
        explicit_marker = any(
            argument.startswith("-m")
            or argument == "--markexpr"
            or argument.startswith("--markexpr=")
            for argument in args.pytest_args
        )
        args.pytest_args[:0] = ["-p", "scripts.run_test_tier", "--focused-path-validation"]
        if not explicit_marker:
            args.pytest_args[2:2] = ["--focused-file-defaults"]
    return run_with_budget(
        build_pytest_command(tier, args.pytest_args),
        tier=args.tier,
        budget_seconds=tier.budget_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
