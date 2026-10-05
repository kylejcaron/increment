from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "postgres_ci_changes.py"


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
        (repo / "README.md").write_text("base docs\n")
        (repo / "increment/source.py").write_text("base source\n")
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


def test_docs_only_pr_does_not_require_postgres(repository):
    repo, base = repository
    _change(repo, "README.md")
    _change(repo, "docs/guides/namespace\nnotes.md")
    _change(repo, "docs/assets/diagram.svg")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "docs")
    assert not _required(repo, base)


@pytest.mark.parametrize(
    "path",
    [
        "increment/query/session.py",
        "increment/sequential_source.py",
        "tests/parity_harness/cases.py",
        "integration/warehouse_execution/_suite.py",
        "conftest.py",
        "uv.lock",
        ".github/workflows/ci.yml",
        "docs/example.py",
        "new-component/README.md",
    ],
)
def test_docs_mixed_with_runtime_or_unknown_path_requires_postgres(repository, path):
    repo, base = repository
    _change(repo, "docs/guide.md")
    _change(repo, path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "mixed")
    assert _required(repo, base)


def test_source_renamed_into_docs_still_requires_postgres(repository):
    repo, base = repository
    (repo / "docs").mkdir()
    _git(repo, "mv", "increment/source.py", "docs/source.md")
    _git(repo, "commit", "-qm", "rename source")
    assert _required(repo, base)


def test_deletion_is_not_silently_classified_as_docs_only(repository):
    repo, base = repository
    _git(repo, "rm", "README.md")
    _git(repo, "commit", "-qm", "delete")
    assert _required(repo, base)


def test_pr_uses_merge_base_instead_of_unrelated_base_branch_edits(repository):
    repo, original_base = repository
    _change(repo, "docs/guide.md")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "feature docs")
    _git(repo, "checkout", "-qb", "base-advanced", original_base)
    _change(repo, "increment/source.py")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "unrelated base runtime edit")
    advanced_base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "feature")
    assert not _required(repo, advanced_base)


def test_missing_diff_base_requires_postgres(repository):
    repo, _ = repository
    assert _required(repo, "0" * 40)


def test_empty_diff_requires_postgres(repository):
    repo, base = repository
    assert _required(repo, base)


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "schedule"])
def test_non_pr_events_always_require_postgres(repository, event):
    repo, _ = repository
    assert _required(repo, "missing-base", event)
