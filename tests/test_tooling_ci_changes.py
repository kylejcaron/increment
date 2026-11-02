from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.tooling_ci_changes import requires_tooling

pytestmark = [pytest.mark.integration, pytest.mark.tooling]


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "increment").mkdir()
    (repo / "increment/__init__.py").write_text("VALUE = 1\n")
    (repo / "README.md").write_text("Initial docs\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, base


def _change(repo: Path, path: str) -> tuple[str, str]:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("changed\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", f"change {path}"], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return base, head


def test_non_pr_events_always_require_tooling(repository):
    repo, base = repository
    assert requires_tooling("push", base, base, repo)
    assert requires_tooling("workflow_dispatch", base, base, repo)


def test_empty_change_set_fails_closed(repository):
    repo, base = repository
    assert requires_tooling("pull_request", base, base, repo)


def test_increment_only_change_skips_tooling(repository):
    repo, _ = repository
    base, head = _change(repo, "increment/new_module.py")
    assert not requires_tooling("pull_request", base, head, repo)


@pytest.mark.parametrize(
    "path",
    [
        "scripts/new.py",
        ".github/workflows/ci.yml",
        "noxfile.py",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
        "tests/_selection.py",
        "tests/conftest.py",
        "tests/test_selection.py",
        "tests/test_anything_ci_changes.py",
        "tests/test_new_unclassified.py",
        "README.md",
    ],
)
def test_non_increment_changes_require_tooling(repository, path: str):
    repo, _ = repository
    base, head = _change(repo, path)
    assert requires_tooling("pull_request", base, head, repo)


def test_increment_plus_other_path_requires_tooling(repository):
    repo, _ = repository
    target = repo / "increment/change.py"
    target.write_text("VALUE = 2\n")
    (repo / "docs").mkdir()
    (repo / "docs/change.md").write_text("changed\n")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "mixed change"], check=True)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    base = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD^"], check=True, capture_output=True, text=True
    ).stdout.strip()
    assert requires_tooling("pull_request", base, head, repo)


def test_missing_refs_or_unavailable_diff_fail_closed(repository):
    repo, base = repository
    assert requires_tooling("pull_request", "", base, repo)
    assert requires_tooling("pull_request", base, "missing-head", repo)
