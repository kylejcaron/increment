from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.compile_registry_ci_changes import requires_compile_registry

pytestmark = pytest.mark.integration


@pytest.fixture
def repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "integration/warehouse_execution").mkdir(parents=True)
    (repo / "integration/warehouse_execution/_compile_dependencies.txt").write_text(
        "increment\nincrement._day_axis\n"
    )
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


def test_push_and_dispatch_always_require_the_registry(repository):
    repo, base = repository
    assert requires_compile_registry("push", base, base, repo)
    assert requires_compile_registry("workflow_dispatch", base, base, repo)


def test_change_to_a_recorded_dependency_outside_query_requires_it(repository):
    repo, _ = repository
    base, head = _change(repo, "increment/_day_axis.py")
    assert requires_compile_registry("pull_request", base, head, repo)


@pytest.mark.parametrize("path", ["uv.lock", "pyproject.toml"])
def test_lockfile_or_pyproject_change_requires_it(repository, path):
    repo, _ = repository
    base, head = _change(repo, path)
    assert requires_compile_registry("pull_request", base, head, repo)


def test_compile_harness_change_requires_it(repository):
    repo, _ = repository
    base, head = _change(repo, "integration/warehouse_execution/_compile.py")
    assert requires_compile_registry("pull_request", base, head, repo)


def test_documentation_only_change_skips_it(repository):
    repo, _ = repository
    base, head = _change(repo, "docs/guides/example.md")
    assert not requires_compile_registry("pull_request", base, head, repo)


def test_unreadable_dependency_record_requires_it(repository):
    repo, _ = repository
    base, head = _change(repo, "docs/guides/example.md")
    (repo / "integration/warehouse_execution/_compile_dependencies.txt").unlink()
    assert requires_compile_registry("pull_request", base, head, repo)
