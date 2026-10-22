from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def test_receipt_reuses_same_tree_and_merge_base_in_linked_worktree(tmp_path: Path) -> None:
    from scripts.prepush_tests import receipt_path

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "--initial-branch=main")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Tests")
    (root / "file.py").write_text("value = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    main = _git(root, "rev-parse", "HEAD")
    _git(root, "branch", "feature")
    linked = tmp_path / "linked"
    _git(root, "worktree", "add", "-q", str(linked), "feature")

    path = receipt_path(linked, main)
    assert path.parent == Path(_git(linked, "rev-parse", "--git-path", "prepush-receipts"))
    assert path.name == f"{_git(linked, 'rev-parse', 'HEAD^{tree}')}-{main}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert receipt_path(linked, main).is_file()


def test_successful_receipt_skips_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import prepush_tests

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "--initial-branch=main")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Tests")
    (root / "file.py").write_text("value = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", commit)
    receipt = prepush_tests.receipt_path(root, commit)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.touch()
    monkeypatch.chdir(root)
    monkeypatch.setattr(
        prepush_tests, "_run", lambda *args: pytest.fail("receipt miss ran selection")
    )

    assert prepush_tests.main() == 0


def test_unit_marker_filter_rejects_integration_item() -> None:
    from scripts.test_changed import _marker_expression

    assert _marker_expression("unit") == (
        "not slow and not integration and not parameter_recovery and "
        "not numerical_certification and not examples and not stress"
    )


def test_test_weakening_hook_warns_without_failing_assertion_removal(tmp_path: Path) -> None:
    import os
    import shlex
    import sys

    import yaml

    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "--initial-branch=main")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Tests")
    test_path = root / "tests/test_sample.py"
    test_path.parent.mkdir()
    test_path.write_text("def test_sample():\n    assert True\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", base)
    test_path.write_text("def test_sample():\n    pass\n")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "remove assertion")

    project = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((project / ".pre-commit-config.yaml").read_text())
    entry = next(
        hook["entry"]
        for repo in config["repos"]
        for hook in repo.get("hooks", [])
        if hook["id"] == "test-weakening"
    )
    command = entry.replace("uv run --no-sync python", shlex.quote(sys.executable))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(project)

    result = subprocess.run(
        ["bash", "-c", command], cwd=root, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0
    assert "assertion-count-decreased" in result.stdout
    assert "WARNING:" in result.stdout
