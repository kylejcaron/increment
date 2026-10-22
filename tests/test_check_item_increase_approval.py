from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_item_increase_approval as check

BASE = {"unit": 10, "fast": 20, "slow": 5, "all": 25}


def test_increase_needs_a_reason_covering_its_delta() -> None:
    head = {**BASE, "unit": 13, "fast": 23, "all": 28}

    assert check.check_increases(BASE, head, "")[0] == 1
    assert check.check_increases(BASE, head, "tests: +2 items — partial")[0] == 1
    assert check.check_increases(BASE, head, "tests: +3 items — new regressions")[0] == 0


def test_each_tier_is_matched_to_its_own_covering_reason() -> None:
    head = {**BASE, "unit": 11, "fast": 23, "all": 28}

    code, message = check.check_increases(BASE, head, "tests: +1 items — one unit regression")

    uncovered = {
        line.split(":")[0] for line in message.splitlines() if "add a PR-description" in line
    }
    assert code == 1
    assert uncovered == {"fast", "all"}


def test_lower_counts_need_no_reason() -> None:
    head = {**BASE, "unit": 7, "fast": 17, "all": 22}

    assert check.check_increases(BASE, head, "")[0] == 0


@pytest.mark.integration
def test_parallel_branches_each_pass_against_the_base_and_merge_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repo / "tests/test_base.py").write_text("def test_base(): pass\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    for branch, tests in (("one", 1), ("two", 2)):
        git("checkout", "-q", "-b", branch, "main")
        body = "".join(f"def test_{branch}_{i}(): pass\n" for i in range(tests))
        (repo / f"tests/test_{branch}.py").write_text(body)
        git("add", ".")
        git("commit", "-q", "-m", branch)
    monkeypatch.chdir(repo)

    git("checkout", "-q", "main")
    monkeypatch.delenv("GITHUB_PR_BODY", raising=False)
    assert check.main(["--base", "main", "--repo", str(repo)]) == 0
    git("checkout", "-q", "two")
    monkeypatch.setenv("GITHUB_PR_BODY", "")
    assert check.main(["--base", "main", "--repo", str(repo)]) == 1
    monkeypatch.setenv("GITHUB_PR_BODY", "tests: +2 items — two regressions")
    assert check.main(["--base", "main", "--repo", str(repo)]) == 0
    git("checkout", "-q", "one")
    monkeypatch.setenv("GITHUB_PR_BODY", "tests: +1 items — one regression")
    assert check.main(["--base", "main", "--repo", str(repo)]) == 0
    git("merge", "-q", "--no-edit", "two")
