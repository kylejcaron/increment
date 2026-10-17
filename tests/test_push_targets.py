from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_push_targets import (
    PushUpdate,
    checked_push_commits,
    parse_push_updates,
)


def _run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _repository(path: Path) -> tuple[Path, str]:
    path.mkdir()
    _run("git", "init", "--quiet", "--initial-branch=main", str(path))
    _run("git", "-C", str(path), "config", "user.name", "Push Guard Test")
    _run("git", "-C", str(path), "config", "user.email", "push-guard@example.invalid")
    (path / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run("git", "-C", str(path), "add", "tracked.py")
    _run("git", "-C", str(path), "commit", "--quiet", "-m", "initial")
    return path, _run("git", "-C", str(path), "rev-parse", "HEAD")


def _update(local_ref: str, local_oid: str, remote_ref: str = "refs/heads/main") -> PushUpdate:
    return PushUpdate(local_ref, local_oid, remote_ref, "0" * 40)


@pytest.mark.integration
def test_push_of_non_head_ref_refuses(tmp_path: Path) -> None:
    repo, _ = _repository(tmp_path / "repo")
    _run("git", "-C", str(repo), "switch", "--quiet", "-c", "other")
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    _run("git", "-C", str(repo), "commit", "--quiet", "-am", "other")
    other = _run("git", "-C", str(repo), "rev-parse", "HEAD")
    _run("git", "-C", str(repo), "switch", "--quiet", "main")
    with pytest.raises(ValueError, match="HEAD"):
        checked_push_commits(repo, (_update("refs/heads/other", other),))


@pytest.mark.integration
def test_second_ref_is_not_ignored(tmp_path: Path) -> None:
    repo, head = _repository(tmp_path / "repo")
    _run("git", "-C", str(repo), "switch", "--quiet", "-c", "other")
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    _run("git", "-C", str(repo), "commit", "--quiet", "-am", "other")
    other = _run("git", "-C", str(repo), "rev-parse", "HEAD")
    _run("git", "-C", str(repo), "switch", "--quiet", "main")
    with pytest.raises(ValueError, match="HEAD"):
        checked_push_commits(
            repo,
            (
                _update("refs/heads/main", head),
                _update("refs/heads/other", other),
            ),
        )


@pytest.mark.integration
def test_annotated_tag_resolves_to_checked_commit(tmp_path: Path) -> None:
    repo, head = _repository(tmp_path / "repo")
    _run("git", "-C", str(repo), "tag", "-a", "release", "-m", "release")
    tag_oid = _run("git", "-C", str(repo), "rev-parse", "refs/tags/release")
    assert checked_push_commits(repo, (_update("refs/tags/release", tag_oid),)) == (head,)


@pytest.mark.parametrize("object_kind", ["missing", "blob"])
@pytest.mark.integration
def test_non_commit_object_refuses(tmp_path: Path, object_kind: str) -> None:
    repo, _ = _repository(tmp_path / "repo")
    oid = (
        "1" * 40
        if object_kind == "missing"
        else _run("git", "-C", str(repo), "rev-parse", "HEAD:tracked.py")
    )
    with pytest.raises(ValueError):
        checked_push_commits(repo, (_update("refs/heads/main", oid),))


@pytest.mark.integration
def test_duplicate_candidate_commits_keep_first_input_order(tmp_path: Path) -> None:
    repo, head = _repository(tmp_path / "repo")
    updates = (
        _update("refs/heads/main", head),
        _update("refs/tags/release", head),
    )
    assert checked_push_commits(repo, updates) == (head,)


@pytest.mark.integration
def test_dirty_candidate_refuses(tmp_path: Path) -> None:
    repo, head = _repository(tmp_path / "repo")
    (repo / "tracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        checked_push_commits(repo, (_update("refs/heads/main", head),))


@pytest.mark.integration
def test_deletion_only_has_no_candidate(tmp_path: Path) -> None:
    repo, _ = _repository(tmp_path / "repo")
    assert checked_push_commits(repo, (_update("(delete)", "0" * 40),)) == ()


def test_malformed_ref_record_fails() -> None:
    with pytest.raises(ValueError):
        parse_push_updates(("refs/heads/main abc refs/heads/main",))


def test_push_records_require_exactly_four_fields() -> None:
    with pytest.raises(ValueError):
        parse_push_updates(("refs/heads/main abc refs/heads/main def extra",))
