from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "certification_ci_changes.py"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _change(repo: Path, path: str) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    return repo, base


def _required(repo: Path, base: str, event: str = "pull_request") -> bool:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--event", event, "--base", base, "--head", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout == "required=true\n"


def _commit_change(repo: Path, path: str) -> None:
    _change(repo, path)
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", f"change {path}")


@pytest.mark.parametrize(
    "path",
    [
        "increment/estimation.py",
        "calibration/model.py",
        "tests/estimation/test_conversion_route_calibration.py",
        "tests/estimation/helper.py",
        "scripts/run_test_tier.py",
        "scripts/certification_ci_changes.py",
        "Makefile",
        "noxfile.py",
        "pyproject.toml",
        "uv.lock",
        ".github/workflows/ci.yml",
        ".github/workflows/weekly.yml",
    ],
)
def test_relevant_change_requires_certification(repository, path: str):
    repo, base = repository
    _commit_change(repo, path)
    assert _required(repo, base)


@pytest.mark.parametrize("path", ["docs/guide.md", "CONTRIBUTING.md", "AGENTS.md"])
def test_unrelated_docs_only_pr_does_not_require_certification(repository, path: str):
    repo, base = repository
    _commit_change(repo, path)
    assert not _required(repo, base)


def test_mixed_docs_and_source_requires_certification(repository):
    repo, base = repository
    _change(repo, "docs/guide.md")
    _change(repo, "increment/estimation.py")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "mixed")
    assert _required(repo, base)


def test_unknown_source_requires_certification(repository):
    repo, base = repository
    _commit_change(repo, "tests/test_behavior.py")
    assert _required(repo, base)


def test_missing_base_requires_certification(repository):
    repo, _ = repository
    assert _required(repo, "0" * 40)


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "schedule"])
def test_non_pr_events_require_certification(repository, event: str):
    repo, _ = repository
    assert _required(repo, "missing-base", event)
