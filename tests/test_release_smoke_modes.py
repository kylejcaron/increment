from __future__ import annotations

import argparse
import subprocess
import sys

import pytest

from scripts import pypi_release_smoke, wheel_smoke


def test_readme_mode_executes_example_after_core_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        wheel_smoke,
        "_arguments",
        lambda: argparse.Namespace(expected_version=None, readme_example=True, demo_extras=False),
    )
    monkeypatch.setattr(wheel_smoke, "main", lambda **kwargs: calls.append(("core", kwargs)))
    monkeypatch.setattr(wheel_smoke, "_check_readme_mean_example", lambda: calls.append("readme"))

    wheel_smoke._run()

    assert calls == [("core", {"core_only": True}), "readme"]


def test_demo_mode_runs_demo_probe_without_core_only_optional_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        wheel_smoke,
        "_arguments",
        lambda: argparse.Namespace(expected_version=None, readme_example=False, demo_extras=True),
    )
    monkeypatch.setattr(wheel_smoke, "main", lambda **kwargs: calls.append(("core", kwargs)))
    monkeypatch.setattr(
        wheel_smoke, "_check_duckdb_memtable_from_pyarrow", lambda: calls.append("demo")
    )

    wheel_smoke._run()

    assert calls == [("core", {"core_only": False}), "demo"]


def test_default_mode_runs_core_guard_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        wheel_smoke,
        "_arguments",
        lambda: argparse.Namespace(expected_version=None, readme_example=False, demo_extras=False),
    )
    monkeypatch.setattr(wheel_smoke, "main", lambda **kwargs: calls.append(("core", kwargs)))
    monkeypatch.setattr(wheel_smoke, "_check_readme_mean_example", lambda: calls.append("readme"))
    monkeypatch.setattr(
        wheel_smoke, "_check_duckdb_memtable_from_pyarrow", lambda: calls.append("demo")
    )

    wheel_smoke._run()

    assert calls == [("core", {"core_only": True})]


def test_readme_mode_accepts_expected_installed_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["wheel_smoke.py", "--readme-example", "--expected-version", "1.2.3"]
    )

    args = wheel_smoke._arguments()

    assert args.readme_example is True
    assert args.expected_version == "1.2.3"


@pytest.mark.parametrize(
    "argv",
    [
        ["wheel_smoke.py", "--unknown-mode"],
        ["wheel_smoke.py", "--demo-extras", "--readme-example"],
    ],
)
def test_unknown_or_conflicting_modes_fail(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        wheel_smoke._arguments()


@pytest.mark.parametrize(
    ("installer_command", "installer_message"),
    [
        (
            "uv pip install",
            "error: Because there is no version of increment that satisfies the requirement",
        ),
        ("pip install", "ERROR: No matching distribution found for increment==1.2.3"),
        ("pip install", "ERROR: Could not find a version for increment==1.2.3"),
    ],
)
def test_installer_specific_unavailable_version_errors_are_retryable(
    installer_command: str, installer_message: str
) -> None:
    error = subprocess.CalledProcessError(1, installer_command, stderr=installer_message)

    assert pypi_release_smoke._retryable_install_failure(error)


def test_non_availability_install_error_is_not_retryable() -> None:
    error = subprocess.CalledProcessError(1, "uv pip install", stderr="disk full")

    assert not pypi_release_smoke._retryable_install_failure(error)
