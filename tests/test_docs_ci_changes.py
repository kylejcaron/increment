from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "docs_ci_changes.py"


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


@pytest.fixture
def repository(tmp_path: Path):
    with TemporaryDirectory(dir=tmp_path) as directory:
        repo = Path(directory)
        _git(repo, "init", "-q")
        (repo / "base.txt").write_text("base\n")
        _git(repo, "add", ".")
        _git(repo, "commit", "-qm", "base")
        yield repo, _git(repo, "rev-parse", "HEAD")


def _run_decision(repo: Path, base: str, *, labels: str = "") -> dict[str, bool]:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--event",
            "pull_request",
            "--base",
            base,
            "--head",
            "HEAD",
            "--labels",
            labels,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return {
        key: value == "true"
        for key, value in (line.split("=", 1) for line in result.stdout.splitlines())
    }


def _change_and_commit(repo: Path, path: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    _git(repo, "add", path)
    _git(repo, "commit", "-qm", f"change {path}")


def test_docs_only_runs_snippets_and_strict_docs(repository):
    repo, base = repository
    _change_and_commit(repo, "docs/guides/example.md")
    assert _run_decision(repo, base) == {"snippets": True, "strict_docs": True}


def test_example_changes_require_strict_docs(repository):
    repo, base = repository
    _change_and_commit(repo, "examples/power.py")
    assert _run_decision(repo, base) == {"snippets": False, "strict_docs": True}


def test_private_source_runs_snippets_without_strict_docs(repository):
    repo, base = repository
    _change_and_commit(repo, "increment/_private.py")
    assert _run_decision(repo, base) == {"snippets": True, "strict_docs": False}


def test_unlabelled_source_api_drift_is_not_claimed_as_strict_docs(repository):
    repo, base = repository
    _change_and_commit(repo, "increment/api.py")
    assert _run_decision(repo, base) == {"snippets": True, "strict_docs": False}


@pytest.mark.parametrize("label", ["breaking", "docs-required"])
def test_public_api_label_requires_strict_docs(repository, label):
    repo, base = repository
    _change_and_commit(repo, "increment/api.py")
    assert _run_decision(repo, base, labels=label) == {"snippets": True, "strict_docs": True}


@pytest.mark.parametrize("path", ["increment/__init__.py", "increment/results.py"])
def test_root_export_changes_require_strict_docs(repository, path):
    repo, base = repository
    _change_and_commit(repo, path)
    assert _run_decision(repo, base) == {"snippets": True, "strict_docs": True}


@pytest.mark.parametrize(
    "path", [".github/workflows/ci.yml", "noxfile.py", "pyproject.toml", "uv.lock"]
)
def test_workflow_and_build_configuration_changes_require_both(repository, path):
    repo, base = repository
    _change_and_commit(repo, path)
    assert _run_decision(repo, base) == {"snippets": True, "strict_docs": True}


def test_unknown_base_fails_closed(repository):
    repo, _ = repository
    assert _run_decision(repo, "missing-base") == {"snippets": True, "strict_docs": True}


pytestmark = pytest.mark.integration
