"""Cross-process checks for machine-wide heavy-test slots."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts import test_slots
from scripts.test_slots import _worker_allowance

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(("held_slots", "expected"), [(1, 12), (2, 6), (3, 4)])
def test_worker_allowance_uses_held_slots(held_slots: int, expected: int) -> None:
    assert _worker_allowance(cpu_count=12, held_slots=held_slots, slot_count=3) == expected


def test_single_nodeid_make_run_does_not_use_xdist() -> None:
    result = subprocess.run(
        ["make", "-n", "test", "TESTS=tests/x.py::test_y"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "-n " not in result.stdout


def test_two_file_make_run_uses_xdist() -> None:
    result = subprocess.run(
        ["make", "-n", "test", "TESTS=tests/x.py tests/y.py"],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "-n " in result.stdout


def _environment(slot_dir: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CI", None)
    env.pop("TEST_SLOT_HELD", None)
    env.update(TEST_SLOT_DIR=str(slot_dir), TEST_SLOTS="1", **overrides)
    return env


def _wait_for_holder(slot_dir: Path, pid: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        for slot in slot_dir.glob("slot-*.lock"):
            try:
                if json.loads(slot.read_text()).get("pid") == pid:
                    return
            except (OSError, ValueError):
                pass
        time.sleep(0.02)
    pytest.fail(f"process {pid} did not acquire a slot within 30 seconds")


def _wrapper(env: dict[str, str], *command: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-m", "scripts.test_slots", "--", *command],
        cwd=Path(__file__).parents[1],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait(process: subprocess.Popen[str]) -> tuple[str, str]:
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        pytest.fail("slot subprocess exceeded the 30 second hang guard")
    return stdout, stderr


def test_second_run_waits_for_the_first(tmp_path: Path) -> None:
    """A second heavy run must not enter while the first owns the only slot."""
    log = tmp_path / "runs"
    release = tmp_path / "release"
    script = (
        "import pathlib, time; "
        f"p=pathlib.Path({str(log)!r}); p.open('a').write('A-start\\n'); "
        f"r=pathlib.Path({str(release)!r}); "
        "exec('import time\\nwhile not r.exists(): time.sleep(.02)'); "
        "p.open('a').write('A-end\\n')"
    )
    first = _wrapper(_environment(tmp_path), sys.executable, "-c", script)
    second = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if first.poll() is not None:
                stdout, stderr = first.communicate()
                pytest.fail(f"holder exited before acquiring its slot: {stdout}{stderr}")
            if log.exists() and log.read_text().splitlines() == ["A-start"]:
                break
            time.sleep(0.02)
        assert log.read_text().splitlines() == ["A-start"]
        second = _wrapper(
            _environment(tmp_path),
            sys.executable,
            "-c",
            f"open({str(log)!r}, 'a').write('B-start\\n')",
        )
        assert second.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(second.stderr, selectors.EVENT_READ)
        stderr_output = b""
        deadline = time.monotonic() + 30
        while (
            (
                b"waiting for a heavy test slot; held by:" not in stderr_output
                or f"pid={first.pid}".encode() not in stderr_output
            )
            and second.poll() is None
            and time.monotonic() < deadline
        ):
            if selector.select(timeout=0.1):
                stderr_output += os.read(second.stderr.fileno(), 4096)
        selector.close()
        stderr = stderr_output.decode()
        assert "waiting for a heavy test slot; held by:" in stderr
        assert f"pid={first.pid}" in stderr
        assert "cwd=" in stderr and "command=" in stderr and "started_at=" in stderr
        assert log.read_text().splitlines() == ["A-start"]
        release.touch()
        assert _wait(first)[0] == ""
        assert _wait(second)[0] == ""
        assert log.read_text().splitlines() == ["A-start", "A-end", "B-start"]
    finally:
        if second is not None and second.poll() is None:
            second.kill()
            second.communicate(timeout=30)
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=30)


def test_nested_run_does_not_wait(tmp_path: Path) -> None:
    """A nested wrapper in the owning process must inherit its held state."""
    marker = tmp_path / "nested"
    inner = [
        sys.executable,
        "-m",
        "scripts.test_slots",
        "--",
        sys.executable,
        "-c",
        f"open({str(marker)!r}, 'w').write('nested')",
    ]
    outer = _wrapper(_environment(tmp_path), *inner)
    _wait(outer)
    assert marker.read_text() == "nested"


def test_slot_held_command_runs_at_lower_priority(tmp_path: Path) -> None:
    """A locally slot-held command adds ten nice points up to the platform ceiling."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.test_slots",
            "--",
            sys.executable,
            "-c",
            "import os; print(os.nice(0)); print(os.nice(100))",
        ],
        cwd=Path(__file__).parents[1],
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    child_nice, max_nice = map(int, result.stdout.splitlines())
    assert child_nice == min(os.nice(0) + 10, max_nice)


def test_hold_slot_lowers_priority_when_called_directly(tmp_path: Path) -> None:
    """A direct hold_slot caller lowers priority after acquiring a local slot."""
    script = (
        "import os; from scripts.test_slots import hold_slot; "
        "hold_slot(); print(os.nice(0)); print(os.nice(100))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=_environment(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )
    child_nice, max_nice = map(int, result.stdout.splitlines())
    assert child_nice == min(os.nice(0) + 10, max_nice)


def test_killed_holder_frees_its_slot(tmp_path: Path) -> None:
    """Killing a slot owner lets a waiting process acquire the released slot."""
    holder = _wrapper(
        _environment(tmp_path),
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
    )
    next_run = None
    try:
        _wait_for_holder(tmp_path, holder.pid)
        next_run = _wrapper(_environment(tmp_path), sys.executable, "-c", "raise SystemExit(0)")
        assert next_run.stderr is not None
        selector = selectors.DefaultSelector()
        selector.register(next_run.stderr, selectors.EVENT_READ)
        stderr = ""
        deadline = time.monotonic() + 30
        while (
            "waiting for a heavy test slot; held by:" not in stderr
            and next_run.poll() is None
            and time.monotonic() < deadline
        ):
            if selector.select(timeout=0.1):
                stderr += next_run.stderr.readline()
        selector.close()
        assert "waiting for a heavy test slot; held by:" in stderr
        os.kill(holder.pid, signal.SIGKILL)
        holder.communicate(timeout=30)
        _wait(next_run)
        assert next_run.returncode == 0
    finally:
        if next_run is not None and next_run.poll() is None:
            next_run.kill()
            next_run.communicate(timeout=30)
        if holder.poll() is None:
            holder.kill()
            holder.communicate(timeout=30)


def test_ci_runs_without_a_slot(tmp_path: Path) -> None:
    """CI bypasses local slot coordination."""
    holder = _wrapper(_environment(tmp_path), sys.executable, "-c", "import time; time.sleep(30)")
    try:
        _wait_for_holder(tmp_path, holder.pid)
        bypass = _wrapper(_environment(tmp_path, CI="true"), sys.executable, "-c", "pass")
        _wait(bypass)
        assert bypass.returncode == 0
    finally:
        holder.kill()
        holder.communicate(timeout=30)


def test_wrapped_exit_status_is_preserved(tmp_path: Path) -> None:
    """The CLI preserves the wrapped command's exit status."""
    wrapped = _wrapper(_environment(tmp_path), sys.executable, "-c", "raise SystemExit(3)")
    _wait(wrapped)
    assert wrapped.returncode == 3


def test_worker_placeholder_is_replaced_by_the_held_allowance(monkeypatch) -> None:
    monkeypatch.setattr(test_slots, "_workers", lambda: 6)
    command = ["pytest", "-n", test_slots.WORKERS_PLACEHOLDER, "-k", "a or b"]

    assert test_slots._with_workers(command) == ["pytest", "-n", "6", "-k", "a or b"]
    assert test_slots._with_workers(["pytest", "-q"]) == ["pytest", "-q"]
