"""Run a named pytest tier with a local wall-clock performance budget."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import NoReturn, TextIO

from scripts._test_tier_policy import TIER_MARKERS

_TIMEOUT_STATUS = 124


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


def _git_output(*arguments: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _resolved_commit(base: str, cwd: Path) -> str:
    try:
        return _git_output(
            "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}", cwd=cwd
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"affected base is not a resolvable commit: {base}") from error


def _changed_paths(base: str, cwd: Path) -> tuple[list[tuple[str, str, str | None]], list[Path]]:
    raw = _git_output("diff", "--name-status", "-z", base, "--", cwd=cwd)
    fields = raw.split("\0")
    changes: list[tuple[str, str, str | None]] = []
    index = 0
    while index < len(fields) and fields[index]:
        status = fields[index]
        index += 1
        if status.startswith(("R", "C")):
            old, new = fields[index : index + 2]
            index += 2
            changes.append((status, old, new))
        else:
            path = fields[index]
            index += 1
            changes.append((status, path, None))
    untracked = _git_output("ls-files", "--others", "--exclude-standard", "-z", cwd=cwd).split("\0")
    changes.extend(("?", path, None) for path in untracked if path)
    paths = [cwd / name for _, old, new in changes for name in (old, new) if name]
    return changes, paths


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def _has_dynamic_import(tree: ast.AST) -> bool:
    aliases: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.partition(".")[0]
                target = alias.name if alias.asname else alias.name.partition(".")[0]
                aliases.setdefault(local, set()).add(target)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                if module == "builtins" and alias.name == "__import__":
                    target = "__import__"
                else:
                    target = f"{module}.{alias.name}" if module else alias.name
                aliases.setdefault(local, set()).add(target)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _dotted_name(node.func)
        if name is None:
            continue
        root, separator, suffix = name.partition(".")
        targets = aliases.get(root, {root})
        for target in targets:
            resolved = f"{target}.{suffix}" if separator else target
            if (
                resolved == "__import__"
                or resolved.endswith(".__import__")
                or resolved == "importlib.import_module"
                or resolved.startswith(("importlib.util.", "runpy.", "pkgutil."))
            ):
                return True
    return False


def _refusal_route(path: str, contents: bytes | None = None) -> str | None:
    normalized = path.replace("\\", "/")
    name = Path(normalized).name
    if normalized.startswith(("examples/", "data/")):
        return "make check and the relevant data/fixture validation"
    if (
        name == "conftest.py"
        or normalized in {"tach.toml", "pyproject.toml", "Makefile"}
        or normalized.startswith(("tests/_", "tests/fixtures/", "scripts/", ".github/"))
        or Path(name).suffix.lower() in {".toml", ".ini", ".cfg", ".yaml", ".yml", ".json"}
        or "lock" in name.lower()
        or normalized == "increment/__init__.py"
    ):
        if name == "conftest.py" or normalized.startswith(("tests/_", "tests/fixtures/")):
            return "make check (and the relevant make test-slow or make test-all fixture tier)"
        if normalized == "increment/__init__.py":
            return "make check and the relevant public API tests"
        return "make check"
    if Path(normalized).suffix != ".py":
        return "make check and the relevant data/fixture validation"
    if normalized.startswith("tests/") and not name.startswith("test_"):
        return "make check and the relevant fixture-dependent test tier"
    if contents is not None:
        try:
            tree = ast.parse(contents)
        except (SyntaxError, UnicodeDecodeError):
            return "make check (collection/import failure cannot be bounded)"
        if _has_dynamic_import(tree):
            return "make check and the relevant dynamic-import consumer tests"
    return None


def _dynamic_import_tests(cwd: Path) -> list[str]:
    consumers = []
    for path in sorted((cwd / "tests").rglob("*.py")):
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        if _has_dynamic_import(tree):
            consumers.append(path.relative_to(cwd).as_posix())
    return consumers


def _environment_identity(cwd: Path) -> str:
    digest = hashlib.sha256(f"{sys.executable}:{sys.version}:{platform.platform()}".encode())
    installed = sorted(
        f"{distribution.metadata.get('Name', '').casefold()}=={distribution.version}"
        for distribution in metadata.distributions()
    )
    digest.update("\\0".join(installed).encode())
    for name in ("pyproject.toml", "uv.lock"):
        path = cwd / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _validate_affected_evidence(
    evidence_root: Path, *, owner_token: str, pytest_status: int
) -> str | None:
    """Accept only finished evidence whose terminal reports exactly match selection."""
    owned: list[tuple[Path, dict[str, object]]] = []
    for run in evidence_root.glob("run-*"):
        manifest = run / "run.json"
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(metadata, dict)
            and metadata.get("validation_kind") == "affected"
            and metadata.get("owner_token") == owner_token
        ):
            owned.append((run, metadata))
    if len(owned) != 1:
        return "incomplete: expected exactly one owner-matched affected evidence run"
    run, metadata = owned[0]
    if metadata.get("status") != "finished":
        return "incomplete: affected evidence run did not finish"
    selected = metadata.get("selected_nodeids")
    if not isinstance(selected, list) or any(not isinstance(nodeid, str) for nodeid in selected):
        return "incomplete: selected node IDs are missing or malformed"
    selected_ids = set(selected)
    if not selected_ids:
        return "no_affected_tests" if pytest_status == 0 else "pytest_failed"
    report_path = run / "reports.jsonl"
    try:
        report_lines = report_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return "incomplete: terminal report journal is missing"
    terminal_ids: set[str] = set()
    try:
        for line in report_lines:
            if not line:
                continue
            event = json.loads(line)
            if not isinstance(event, dict) or event.get("event") != "report":
                continue
            nodeid = event.get("nodeid")
            when = event.get("when")
            outcome = event.get("outcome")
            if isinstance(nodeid, str) and (
                when == "call" or (when == "setup" and outcome in {"failed", "skipped"})
            ):
                terminal_ids.add(nodeid)
    except (TypeError, ValueError):
        return "incomplete: terminal report journal is malformed"
    if selected_ids != terminal_ids:
        return "incomplete: selected and terminal report node IDs do not match"
    if pytest_status != 0:
        return "pytest_failed"
    return None


def _affected_context(base: str, cwd: Path, tier: str) -> tuple[str, dict[str, object]]:
    resolved = _resolved_commit(base, cwd)
    changes, paths = _changed_paths(resolved, cwd)
    refused = []
    for status, old, new in changes:
        if status.startswith(("R", "C")) or status == "D":
            refused.append(f"{old}: rename/deletion impact cannot be established; run make check")
        target = new or old
        target_path = cwd / target
        try:
            contents = target_path.read_bytes() if target_path.is_file() else None
        except OSError:
            contents = None
        route = _refusal_route(target, contents)
        if route is not None:
            refused.append(f"{target}: impact is not proven; run {route}")
    changed_source_modules = [
        new or old
        for status, old, new in changes
        if status != "D" and (new or old).endswith(".py") and not (new or old).startswith("tests/")
    ]
    dynamic_consumers = _dynamic_import_tests(cwd) if changed_source_modules else []
    if refused:
        raise ValueError("affected validation refused: " + "; ".join(sorted(set(refused))))
    source_identity = hashlib.sha256()
    for path in sorted(paths):
        try:
            relative = path.relative_to(cwd).as_posix()
            source_identity.update(relative.encode())
            source_identity.update(b"\0")
            if path.is_file():
                source_identity.update(path.read_bytes())
            source_identity.update(b"\0")
        except OSError:
            pass
    dirty_inputs = source_identity.hexdigest()
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all", cwd=cwd)
    identity = {
        "validation_kind": "affected",
        "affected_base": resolved,
        "source_revision": _git_output("rev-parse", "HEAD", cwd=cwd).strip(),
        "worktree": str(cwd.resolve()),
        "dirty_inputs": dirty_inputs,
        "dirty_status": hashlib.sha256(status.encode()).hexdigest(),
        "tier": tier,
        "environment_id": _environment_identity(cwd),
        "worker_allowance": "serial",
        "retained_dynamic_consumers": dynamic_consumers,
        "changed_paths": [path.relative_to(cwd).as_posix() for path in sorted(paths)],
    }
    return resolved, identity


def _affected_environment(identity: dict[str, object]) -> dict[str, str | None]:
    changed_paths = identity["changed_paths"]
    retained_dynamic = identity["retained_dynamic_consumers"]
    changed_tests = (
        [
            path
            for path in changed_paths
            if isinstance(path, str) and path.startswith("tests/") and path.endswith(".py")
        ]
        if isinstance(changed_paths, list)
        else []
    )
    dynamic_tests = (
        [
            path
            for path in retained_dynamic
            if isinstance(path, str) and path.startswith("tests/") and path.endswith(".py")
        ]
        if isinstance(retained_dynamic, list)
        else []
    )
    retained_tests = sorted(set(changed_tests) | set(dynamic_tests))
    worker_allowance = identity.get("worker_allowance", "serial")
    return {
        "INCREMENT_AFFECTED_EVIDENCE": json.dumps(identity, sort_keys=True),
        "INCREMENT_AFFECTED_TEST_PATHS": json.dumps(retained_tests),
        "INCREMENT_AFFECTED_DYNAMIC_TEST_PATHS": json.dumps(dynamic_tests),
        "INCREMENT_AFFECTED_WORKER_ALLOWANCE": str(worker_allowance),
    }


def _affected_runner_environment(identity: dict[str, object]) -> dict[str, str | None]:
    environment: dict[str, str | None] = _affected_environment(identity)
    cleared_names = {
        name for name in os.environ if name.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    environment.update(dict.fromkeys(cleared_names))
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


@dataclass(frozen=True, slots=True)
class Tier:
    marker: str | None
    budget_seconds: int


TIERS = {
    "focused": Tier(marker="", budget_seconds=180),
    "fast": Tier(marker=TIER_MARKERS["fast"], budget_seconds=240),
    "slow": Tier(marker=TIER_MARKERS["slow"], budget_seconds=480),
    # Supports optional pytest-split shards while retaining a local budget.
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
    deadline: float | None,
    budget_seconds: float | None,
    heartbeat_seconds: float,
    grace_seconds: float,
    stream: TextIO,
    windows_job: int | None,
    check_interrupt: Callable[[], None],
) -> tuple[int, bool]:
    """Return the exit status and whether the supervisor deadline fired."""
    next_heartbeat = started + heartbeat_seconds
    budget_display = "disabled" if budget_seconds is None else f"{budget_seconds:g}s"
    while True:
        check_interrupt()
        now = time.monotonic()
        remaining = None if deadline is None else deadline - now
        if remaining is not None and remaining <= 0:
            print(
                f"test tier timed out: tier={tier} budget={budget_display} "
                f"elapsed={now - started:.1f}s",
                file=stream,
                flush=True,
            )
            _stop_process_group(process, process.pid, signal.SIGTERM, grace_seconds, windows_job)
            return _TIMEOUT_STATUS, True
        wait_seconds = min(0.1, max(0.01, next_heartbeat - now))
        if remaining is not None:
            wait_seconds = min(remaining, wait_seconds)
        try:
            return _exit_status(process.wait(timeout=wait_seconds)), False
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            if next_heartbeat <= now and (deadline is None or now < deadline):
                print(
                    f"test tier running: tier={tier} elapsed={now - started:.1f}s "
                    f"budget={budget_display}",
                    file=stream,
                    flush=True,
                )
                next_heartbeat = now + heartbeat_seconds


def run_with_budget(
    command: Sequence[str],
    *,
    tier: str,
    budget_seconds: float | None,
    heartbeat_seconds: float = 30.0,
    grace_seconds: float = 5.0,
    stream: TextIO = sys.stderr,
    extra_env: Mapping[str, str | None] | None = None,
    affected_evidence_root: Path | None = None,
) -> int:
    """Supervise ``command``; ``None`` disables its deadline, not signal cleanup."""
    started = time.monotonic()
    deadline = None if budget_seconds is None else started + budget_seconds
    owner_token = uuid.uuid4().hex

    forwarded = (signal.SIGINT, signal.SIGTERM)
    forwarded += (vars(signal)["SIGBREAK"],) if os.name == "nt" else ()
    original_handlers = {signum: signal.getsignal(signum) for signum in forwarded}
    pending_signal: int | None = None

    def interrupt(signum: int, _frame: object) -> None:
        nonlocal pending_signal
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
    timed_out = False
    try:
        child_environment = os.environ.copy()
        for name, value in (extra_env or {}).items():
            if value is None:
                child_environment.pop(name, None)
            else:
                child_environment[name] = value
        child_environment["INCREMENT_EVIDENCE_OWNER"] = owner_token
        process, windows_job, startup_gate = _start_process(command, env=child_environment)
        _raise_pending()
        if startup_gate is not None:
            os.write(startup_gate, b"1")
            _close_fd(startup_gate)
            startup_gate = None

        status, timed_out = _supervise_process(
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
                    process, process.pid, signal.SIGTERM, grace_seconds, windows_job
                )
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
    if pending_signal is not None:
        status = 128 + pending_signal
    termination = (
        "budget_timeout"
        if timed_out and pending_signal is None
        else "interrupted"
        if status in (128 + signum for signum in forwarded)
        else "process_exit"
    )
    _finalize_unfinished_evidence(command, status, owner_token, termination)
    if affected_evidence_root is not None:
        evidence = _validate_affected_evidence(
            affected_evidence_root, owner_token=owner_token, pytest_status=status
        )
        if evidence == "no_affected_tests":
            print("pytest: error: no affected tests were selected", file=stream)
            return 5
        if evidence is not None and status == 0:
            print(f"pytest: error: affected evidence rejected ({evidence})", file=stream)
            return 1
    return status


def _affected_main(argv: Sequence[str]) -> int:
    class Parser(argparse.ArgumentParser):
        def error(self, message: str) -> NoReturn:
            print(
                f"pytest: error: affected runner accepts only typed options ({message})",
                file=sys.stderr,
            )
            raise SystemExit(2)

    parser = Parser(add_help=False, allow_abbrev=False)
    parser.add_argument("--affected-base", required=True, metavar="GITREF")
    parser.add_argument("tier", choices=("fast", "slow"))
    parser.add_argument("-k", "--keyword")
    parser.add_argument("-m", "--markexpr")
    parser.add_argument("--maxfail", type=int)
    parser.add_argument("-x", dest="maxfail_one", action="store_true")
    parser.add_argument("-q", dest="quiet", action="count", default=0)
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    parser.add_argument("--durations", type=int)
    parser.add_argument("-n", dest="workers", type=int)
    parser.add_argument("--dist", choices=("loadgroup",))
    parser.add_argument("--evidence-root", default=".test-evidence/affected")
    parser.add_argument("--runtime-diagnostics", action="store_true")
    args = parser.parse_args(argv)
    if args.maxfail_one and args.maxfail is not None:
        parser.error("-x and --maxfail cannot be combined")
    if args.workers is not None and not 1 <= args.workers <= 8:
        parser.error("-n must be a numeric worker count from 1 through 8")
    if args.dist is not None and args.workers is None:
        parser.error("--dist requires -n")
    if args.runtime_diagnostics and not args.evidence_root:
        parser.error("--runtime-diagnostics requires --evidence-root")
    for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        if os.environ.get(name):
            print(f"pytest: error: affected runner refuses inherited {name}", file=sys.stderr)
            return 2

    try:
        base, identity = _affected_context(args.affected_base, Path.cwd(), args.tier)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"pytest: error: {error}", file=sys.stderr)
        return 2
    worker_allowance = "serial" if args.workers is None else str(args.workers)
    identity["worker_allowance"] = worker_allowance
    identity["worker_bound"] = args.workers
    evidence_root = Path(args.evidence_root).resolve()
    command = [
        sys.executable,
        "-m",
        "scripts.affected_pytest_launcher",
        "--tach-base",
        base,
        "--tier",
        args.tier,
    ]
    if args.keyword:
        command.extend(("-k", args.keyword))
    if args.markexpr:
        command.extend(("-m", args.markexpr))
    if args.maxfail_one:
        command.append("-x")
    elif args.maxfail is not None:
        command.extend(("--maxfail", str(args.maxfail)))
    command.extend(["-q"] * args.quiet)
    command.extend(["-v"] * args.verbose)
    if args.durations is not None:
        command.extend(("--durations", str(args.durations)))
    if args.workers is not None:
        command.extend(("-n", str(args.workers)))
    if args.dist is not None:
        command.extend(("--dist", args.dist))
    command.extend(("--evidence-root", str(evidence_root)))
    if args.runtime_diagnostics:
        command.append("--runtime-diagnostics")

    extra_env = _affected_runner_environment(identity)
    worker_display = (
        "serial" if worker_allowance == "serial" else f"{worker_allowance} (dist=loadgroup)"
    )
    print(
        f"Affected-test validation: base={base} tier={args.tier} worker_allowance={worker_display}",
        flush=True,
    )
    tier = TIERS[args.tier]
    return run_with_budget(
        command,
        tier=args.tier,
        budget_seconds=None if os.environ.get("GITHUB_ACTIONS") == "true" else tier.budget_seconds,
        extra_env=extra_env,
        affected_evidence_root=evidence_root,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if any(
        argument == "--affected-base" or argument.startswith("--affected-base=")
        for argument in arguments
    ):
        return _affected_main(arguments)

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--affected-base",
        metavar="GITREF",
        help="Select tests affected since a recorded commit (fast/slow tier only).",
    )
    parser.add_argument("tier", choices=tuple(TIERS))
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(arguments)
    tier = TIERS[args.tier]
    extra_env = {
        "INCREMENT_AFFECTED_EVIDENCE": "",
        "INCREMENT_AFFECTED_TEST_PATHS": "[]",
        "INCREMENT_AFFECTED_DYNAMIC_TEST_PATHS": "[]",
        "INCREMENT_AFFECTED_WORKER_ALLOWANCE": "serial",
    }
    if os.environ.get("INCREMENT_AFFECTED_EVIDENCE"):
        extra_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = None
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
        args.pytest_args[:0] = [
            "-p",
            "scripts.run_test_tier_plugin",
            "--focused-path-validation",
        ]
        if not explicit_marker:
            args.pytest_args[2:2] = ["--focused-file-defaults"]
    return run_with_budget(
        build_pytest_command(tier, args.pytest_args),
        tier=args.tier,
        budget_seconds=None if os.environ.get("GITHUB_ACTIONS") == "true" else tier.budget_seconds,
        extra_env=extra_env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
