from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.pypi_release_smoke import run_post_publish_smoke


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_waits_for_index_then_retries_only_availability_install_failure() -> None:
    clock = FakeClock()
    index_states = iter((False, True))
    install_attempts: list[float] = []

    def install(_version: str, _python: Path, timeout: float) -> None:
        install_attempts.append(timeout)
        if len(install_attempts) == 1:
            raise subprocess.CalledProcessError(
                1, "uv pip install", stderr="No matching distribution found"
            )

    smoke_calls: list[str] = []
    run_post_publish_smoke(
        "1.2.3",
        Path("python"),
        Path("smoke.py"),
        cwd=Path("."),
        index_probe=lambda _version, _timeout: next(index_states),
        installer=install,
        smoke_runner=lambda _python, _script, version, _cwd, _timeout: smoke_calls.append(version),
        clock=clock.monotonic,
        sleeper=clock.sleep,
        deadline_seconds=5,
        poll_seconds=1,
    )

    assert clock.now == 2
    assert len(install_attempts) == 2
    assert smoke_calls == ["1.2.3"]


def test_fails_at_deadline_when_exact_version_never_appears() -> None:
    clock = FakeClock()
    install_calls: list[str] = []

    with pytest.raises(SystemExit, match="did not become available"):
        run_post_publish_smoke(
            "1.2.3",
            Path("python"),
            Path("smoke.py"),
            cwd=Path("."),
            index_probe=lambda _version, _timeout: False,
            installer=lambda version, _python, _timeout: install_calls.append(version),
            smoke_runner=lambda *_args: None,
            clock=clock.monotonic,
            sleeper=clock.sleep,
            deadline_seconds=3,
            poll_seconds=1,
        )

    assert clock.now == 3
    assert install_calls == []


def test_smoke_failure_is_not_retried() -> None:
    clock = FakeClock()
    install_calls: list[str] = []
    smoke_calls: list[str] = []

    def fail_smoke(_python: Path, _script: Path, version: str, _cwd: Path, _timeout: float) -> None:
        smoke_calls.append(version)
        raise subprocess.CalledProcessError(1, "wheel_smoke.py")

    with pytest.raises(subprocess.CalledProcessError):
        run_post_publish_smoke(
            "1.2.3",
            Path("python"),
            Path("smoke.py"),
            cwd=Path("."),
            index_probe=lambda _version, _timeout: True,
            installer=lambda version, _python, _timeout: install_calls.append(version),
            smoke_runner=fail_smoke,
            clock=clock.monotonic,
            sleeper=clock.sleep,
            deadline_seconds=3,
            poll_seconds=1,
        )

    assert install_calls == ["1.2.3"]
    assert smoke_calls == ["1.2.3"]
