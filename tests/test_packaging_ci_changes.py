from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "packaging_ci_changes.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI Test",
            "-c",
            "user.email=ci@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "-c",
            f"core.hooksPath={os.devnull}",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _change(repo: Path, path: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")


@pytest.fixture
def repository(tmp_path):
    with TemporaryDirectory(dir=tmp_path) as directory:
        repo = Path(directory)
        _git(repo, "init", "-q")
        _change(repo, "README.md")
        _change(repo, "increment/source.py")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "base")
        base = _git(repo, "rev-parse", "HEAD")
        _git(repo, "checkout", "-qb", "feature")
        yield repo, base


def _required(repo: Path, base: str, event: str = "pull_request") -> bool:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--event",
            event,
            "--base",
            base,
            "--head",
            "HEAD",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout in {"required=true\n", "required=false\n"}
    return result.stdout == "required=true\n"


def test_source_only_change_skips_wheel_smoke(repository):
    repo, base = repository
    _change(repo, "increment/frame.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "source")
    assert not _required(repo, base)


def test_lockfile_requires_wheel_smoke(repository):
    repo, base = repository
    _change(repo, "uv.lock")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "lockfile")
    assert _required(repo, base)


def test_package_data_requires_wheel_smoke(repository):
    repo, base = repository
    _change(repo, "increment/dashboard/_dashboard.css")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "package data")
    assert _required(repo, base)


def test_push_event_always_requires_wheel_smoke(repository):
    repo, base = repository
    assert _required(repo, base, "push")


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "MANIFEST.in",
        "hatch_build.py",
        "noxfile.py",
        "scripts/wheel_smoke.py",
        "scripts/packaging_ci_changes.py",
        ".github/workflows/ci.yml",
        "increment/py.typed",
        "increment/dashboard/py.typed",
    ],
)
def test_packaging_configuration_and_type_markers_require_wheel_smoke(repository, path):
    repo, base = repository
    _change(repo, path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "packaging configuration")
    assert _required(repo, base)


pytestmark = pytest.mark.integration
