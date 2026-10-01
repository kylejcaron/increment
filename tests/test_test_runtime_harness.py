from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

import conftest as root_conftest
from scripts.run_test_tier import run_with_budget


@dataclass
class _Item:
    markers: dict[str, Any]
    added: list[Any] = field(default_factory=list)
    path: Path = Path("tests/test_example.py")

    def get_closest_marker(self, name: str) -> Any:
        return self.markers.get(name)

    def add_marker(self, marker: Any) -> None:
        self.markers[marker.mark.name] = marker
        self.added.append(marker)


@pytest.mark.parametrize(
    ("marker_names", "expected_seconds"),
    [
        ((), 30),
        (("slow",), 180),
        (("parameter_recovery",), 300),
        (("examples", "slow"), 300),
        (("parameter_recovery", "slow"), 300),
    ],
)
def test_runtime_policy_assigns_the_most_expensive_test_class(marker_names, expected_seconds):
    item = _Item({name: getattr(pytest.mark, name) for name in marker_names})

    root_conftest.pytest_collection_modifyitems([cast(pytest.Item, item)])

    assert len(item.added) == 1
    timeout = item.added[0].mark
    assert timeout.name == "timeout"
    assert timeout.args == (expected_seconds,)


def test_runtime_policy_preserves_an_explicit_timeout():
    explicit = pytest.mark.timeout(12)
    item = _Item({"slow": pytest.mark.slow, "timeout": explicit})

    root_conftest.pytest_collection_modifyitems([cast(pytest.Item, item)])

    assert item.added == []


def test_markdown_items_receive_the_root_timeout_policy():
    item = _Item({}, path=Path("README.md"))

    root_conftest.pytest_collection_modifyitems([cast(pytest.Item, item)])

    timeout = item.get_closest_marker("timeout")
    assert timeout is not None
    assert 30 in (timeout.args or (timeout.kwargs.get("timeout"),))


@pytest.mark.slow
def test_root_runtime_policy_interrupts_an_unmarked_sleeping_test():
    repository = Path(__file__).parents[1]
    with tempfile.TemporaryDirectory(dir=repository) as temporary:
        test_directory = Path(temporary)
        (test_directory / "conftest.py").write_text(
            "import pytest\n\n"
            "@pytest.hookimpl(trylast=True)\n"
            "def pytest_collection_modifyitems(items):\n"
            "    for item in items:\n"
            "        timeout = item.get_closest_marker('timeout')\n"
            "        assert timeout is not None and timeout.args == (30,)\n"
            "        item.add_marker(pytest.mark.timeout(0.05), append=False)\n"
        )
        sleeping = test_directory / "test_sleeping_runtime_guard.py"
        sleeping.write_text("import time\n\ndef test_runtime_guard_sleeper():\n    time.sleep(2)\n")

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:tach", str(sleeping)],
            cwd=repository,
            text=True,
            capture_output=True,
            check=False,
            # Hang guard only: interpreter startup under a loaded suite can exceed seconds;
            # the assertions below carry the contract.
            timeout=120,
        )

    assert result.returncode == 1
    assert "test_runtime_guard_sleeper" in result.stdout
    assert "Timeout" in result.stdout


def _wait_for_path(path: Path, timeout: float = 60) -> None:
    # Generous: this only waits for a descendant to signal readiness; the
    # test's own assertions, not this poll, exercise the guarded behavior.
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert path.exists()


def _wait_for_pid(path: Path, timeout: float = 60) -> int:
    # A readiness path can exist before its PID payload is fully published.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            pid = int(path.read_text().strip())
        except (FileNotFoundError, ValueError):
            time.sleep(0.01)
            continue
        if pid > 0:
            return pid
        time.sleep(0.01)
    pytest.fail(f"readiness path {path} did not publish a valid PID")


def _reap_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=60)
    else:
        process.wait()


def _process_exists(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ("tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"),
            check=False,
            capture_output=True,
            text=True,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_for_process_exit(pid: int, timeout: float = 60) -> None:
    # Generous: cleanup can lag under load; the failure mode below still
    # catches a descendant that never exits.
    deadline = time.monotonic() + timeout
    while _process_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    if _process_exists(pid):
        pytest.fail(f"descendant process {pid} survived process-group cleanup")


@pytest.mark.slow
def test_total_budget_kills_the_child_process_group(tmp_path):
    ready = tmp_path / "descendant-ready"
    child = tmp_path / "spawn_descendant.py"
    child.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f'"import os,pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        f'pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(10)"])\n'
        "time.sleep(10)\n"
    )
    helper = tmp_path / "run_timeout_budget.py"
    helper.write_text(
        "import sys\n"
        "from scripts.run_test_tier import run_with_budget\n"
        f"raise SystemExit(run_with_budget([sys.executable, {str(child)!r}], "
        "tier='test', budget_seconds=2, heartbeat_seconds=1, grace_seconds=0.1))\n"
    )
    runner = subprocess.Popen(
        [sys.executable, str(helper)],
        cwd=Path(__file__).parents[1],
    )
    try:
        descendant_pid = _wait_for_pid(ready)
        # Anchor the budget clock at the descendant's readiness signal:
        # interpreter startup precedes the window the runner enforces.
        budget_window_started = time.monotonic()

        # 60s: reaping the already-terminated process is readiness, not the
        # property under test; the elapsed assertion below checks the budget.
        status = runner.wait(timeout=60)
        elapsed = time.monotonic() - budget_window_started

        assert status == 124
        assert elapsed < 3
        _wait_for_process_exit(descendant_pid)
    finally:
        _reap_process(runner)


@pytest.mark.slow
def test_outer_budget_allows_nested_runner_to_finish_cleanup(monkeypatch, tmp_path):
    import scripts.run_test_tier as runner_module

    ready = tmp_path / "nested-descendant-ready"
    child = tmp_path / "nested_child.py"
    child.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        f'"import os,pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        f'pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(120)"])\n'
        "time.sleep(10)\n"
    )
    inner = tmp_path / "inner_runner.py"
    inner.write_text(
        "import sys\n"
        "from scripts.run_test_tier import run_with_budget\n"
        f"raise SystemExit(run_with_budget([sys.executable, {str(child)!r}], "
        "tier='inner', budget_seconds=10, grace_seconds=0.1))\n"
    )
    descendant_pid = None
    supervise = runner_module._supervise_process

    def supervise_after_readiness(process, **kwargs):
        nonlocal descendant_pid
        descendant_pid = _wait_for_pid(ready)
        # Exercise cleanup after startup, even when its original deadline has expired.
        return supervise(process, **kwargs)

    monkeypatch.setattr(runner_module, "_supervise_process", supervise_after_readiness)

    status = run_with_budget(
        [sys.executable, str(inner)],
        tier="outer",
        budget_seconds=1,
        grace_seconds=0.5,
    )

    assert descendant_pid is not None
    assert status == 124
    _wait_for_process_exit(descendant_pid)


@pytest.mark.slow
def test_interrupt_during_startup_keeps_process_ownership(monkeypatch, tmp_path):
    import scripts.run_test_tier as runner_module

    started = tmp_path / "command-started"
    original_start = runner_module._start_process

    def interrupt_before_assignment(command, *, env):
        owned = original_start(command, env=env)
        signal.raise_signal(signal.SIGINT)
        return owned

    monkeypatch.setattr(runner_module, "_start_process", interrupt_before_assignment)

    status = runner_module.run_with_budget(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(started)!r}).touch()"],
        tier="startup-interrupt-test",
        budget_seconds=5,
        grace_seconds=0.1,
    )

    assert status == 128 + signal.SIGINT
    assert not started.exists()


@pytest.mark.slow
def test_runner_cleans_descendant_after_group_leader_exits(tmp_path):
    ready = tmp_path / "orphan-descendant-ready"
    child = tmp_path / "exit_after_spawn.py"
    child.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"ready = pathlib.Path({str(ready)!r})\n"
        "subprocess.Popen([sys.executable, '-c', "
        f'"import os,pathlib,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        f"ready=pathlib.Path({str(ready)!r}); pending=ready.with_suffix('.pending'); "
        'pending.write_text(str(os.getpid())); pending.replace(ready); time.sleep(120)"])\n'
        "while not ready.exists():\n"
        "    time.sleep(0.01)\n"
    )

    status = run_with_budget(
        [sys.executable, str(child)],
        tier="leader-exit-test",
        budget_seconds=5,
        grace_seconds=0.1,
    )

    descendant_pid = _wait_for_pid(ready)
    assert status == 0
    _wait_for_process_exit(descendant_pid)


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="POSIX signal status regression")
def test_signal_terminated_child_returns_conventional_status():
    status = run_with_budget(
        [
            sys.executable,
            "-c",
            "import os,signal; os.kill(os.getpid(), signal.SIGKILL)",
        ],
        tier="signal-status-test",
        budget_seconds=5,
    )

    assert status == 128 + signal.SIGKILL


@pytest.mark.slow
@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object startup regression")
def test_windows_job_gate_starts_the_guarded_child(tmp_path):
    started = tmp_path / "windows-child-started"

    status = run_with_budget(
        [sys.executable, "-c", f"from pathlib import Path; Path({str(started)!r}).touch()"],
        tier="windows-startup-test",
        budget_seconds=5,
        grace_seconds=0.1,
    )

    assert status == 0
    assert started.exists()


@pytest.mark.slow
def test_interrupt_cleans_the_process_group_and_returns_signal_status(tmp_path):
    ready = tmp_path / "interrupt-descendant-ready"
    windows_ignore = "signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n" if os.name == "nt" else ""
    descendant_windows_ignore = (
        "signal.signal(signal.SIGBREAK, signal.SIG_IGN); " if os.name == "nt" else ""
    )
    child = tmp_path / "ignore_interrupt.py"
    child.write_text(
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"{windows_ignore}"
        "subprocess.Popen([sys.executable, '-c', "
        f'"import os,pathlib,signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); '
        f"signal.signal(signal.SIGTERM, signal.SIG_IGN); {descendant_windows_ignore}"
        f'pathlib.Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(120)"])\n'
        "time.sleep(10)\n"
    )
    helper = tmp_path / "run_budget.py"
    helper.write_text(
        "import sys\n"
        "from scripts.run_test_tier import run_with_budget\n"
        f"raise SystemExit(run_with_budget([sys.executable, {str(child)!r}], "
        "tier='interrupt-test', budget_seconds=10, heartbeat_seconds=1, grace_seconds=0.1))\n"
    )
    creationflags = vars(subprocess)["CREATE_NEW_PROCESS_GROUP"] if os.name == "nt" else 0
    runner = subprocess.Popen(
        [sys.executable, str(helper)],
        cwd=Path(__file__).parents[1],
        creationflags=creationflags,
    )
    try:
        descendant_pid = _wait_for_pid(ready)

        interrupt_signal = vars(signal)["CTRL_BREAK_EVENT"] if os.name == "nt" else signal.SIGINT
        expected_status = 128 + vars(signal)["SIGBREAK"] if os.name == "nt" else 128 + signal.SIGINT
        runner.send_signal(interrupt_signal)
        # 60s: reaping is readiness, not the signal-status property asserted below.
        status = runner.wait(timeout=60)

        assert status == expected_status
        _wait_for_process_exit(descendant_pid)
    finally:
        _reap_process(runner)


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="POSIX subprocess wait-lock regression")
def test_interrupt_defers_until_subprocess_wait_releases_its_lock(tmp_path):
    helper = tmp_path / "interrupt_inside_wait.py"
    helper.write_text(
        "import signal, subprocess, sys\n"
        "from scripts.run_test_tier import run_with_budget\n"
        "original_popen = subprocess.Popen\n"
        "class InterruptingLock:\n"
        "    def __init__(self, lock):\n"
        "        self.lock, self.triggered = lock, False\n"
        "    def acquire(self, *args, **kwargs):\n"
        "        acquired = self.lock.acquire(*args, **kwargs)\n"
        "        if acquired and not self.triggered:\n"
        "            self.triggered = True\n"
        "            try:\n"
        "                signal.raise_signal(signal.SIGINT)\n"
        "            except BaseException:\n"
        "                self.lock.release()\n"
        "                raise\n"
        "            print('critical operation completed', flush=True)\n"
        "        return acquired\n"
        "    def release(self):\n"
        "        self.lock.release()\n"
        "    def __enter__(self):\n"
        "        self.acquire()\n"
        "        return self\n"
        "    def __exit__(self, *args):\n"
        "        self.release()\n"
        "def launch(*args, **kwargs):\n"
        "    process = original_popen(*args, **kwargs)\n"
        "    process._waitpid_lock = InterruptingLock(process._waitpid_lock)\n"
        "    return process\n"
        "subprocess.Popen = launch\n"
        "raise SystemExit(run_with_budget(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(10)'],\n"
        "    tier='locked-interrupt', budget_seconds=5, grace_seconds=0.1))\n"
    )
    # Release a stranded lock in the helper so the pre-fix witness cannot leak its child.
    result = subprocess.run(
        [sys.executable, str(helper)],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 128 + signal.SIGINT, result.stderr
    assert "critical operation completed" in result.stdout
