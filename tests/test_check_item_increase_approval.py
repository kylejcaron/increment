from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_item_increase_approval as check


@pytest.mark.integration
class TestItemIncreaseApproval:
    @staticmethod
    def _git(repo: Path, *args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    def _repo(self, tmp_path: Path) -> tuple[Path, str]:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True
        )
        subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
        (repo / ".github").mkdir()
        baseline = repo / ".github/collect-baseline"
        baseline.write_text("unit 10\nfast 20\nslow 5\nall 30\n")
        subprocess.run(["git", "-C", str(repo), "add", ".github/collect-baseline"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", "base"], check=True)
        return repo, self._git(repo, "rev-parse", "HEAD")

    def test_every_tier_increase_needs_a_covering_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        repo, base = self._repo(tmp_path)
        (repo / ".github/collect-baseline").write_text("unit 11\nfast 23\nslow 5\nall 33\n")
        subprocess.run(["git", "-C", str(repo), "add", ".github/collect-baseline"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-q", "-m", "raise baselines"], check=True
        )
        monkeypatch.setenv(
            "GITHUB_PR_BODY",
            "tests: +1 items — add one unit regression\ntests: +3 items — include all collected tier additions",
        )
        assert check.main(["--base", base, "--repo", str(repo)]) == 0
        assert "unit: baseline increase +1" in capsys.readouterr().out

        monkeypatch.setenv("GITHUB_PR_BODY", "tests: +2 items — only covers part of the increase")
        assert check.main(["--base", base, "--repo", str(repo)]) == 1
        assert "fast: baseline increased by 3" in capsys.readouterr().out

    def test_unchanged_baselines_need_no_reason(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo, base = self._repo(tmp_path)
        monkeypatch.delenv("GITHUB_PR_BODY", raising=False)
        assert check.main(["--base", base, "--repo", str(repo)]) == 0
