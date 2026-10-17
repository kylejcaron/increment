from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import cast

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
    (repo / "tach.toml").write_text(
        "exact = true\n"
        '[[modules]]\npath = "increment.power._binomial"\ndepends_on = ["increment.power.core"]\n'
        '[[modules]]\npath = "increment.power.core"\ndepends_on = ["increment.errors"]\n'
        '[[modules]]\npath = "increment.estimation"\ndepends_on = ["increment.errors"]\n'
        '[[modules]]\npath = "increment.analysis"\ndepends_on = ["increment.estimation", "increment.errors"]\n'
        '[[modules]]\npath = "increment.dashboard"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.errors"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.future_engine"\ndepends_on = ["increment.errors"]\n'
    )
    for path, imports in {
        "tests/power/test_binomial_planning.py": (
            "from increment.power._binomial import RejectionGeometry\n"
        ),
        "tests/estimation/test_conversion_route_calibration.py": (
            "from increment.estimation import binomial_rr\n"
        ),
        "tests/calibration/test_cuped_linear.py": (
            "from increment.estimation.cuped import fit_cuped\n"
        ),
    }.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(imports)
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "module graph")
    base = _git(repo, "rev-parse", "HEAD")
    return repo, base


def test_fixture_repo_ignores_inherited_git_dir(
    repository: tuple[Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import conftest

    repo, _ = repository
    other_repo = tmp_path / "other"
    other_repo.mkdir()
    _git(other_repo, "init", "-q")
    monkeypatch.setenv("GIT_DIR", str(other_repo / ".git"))

    conftest.pytest_sessionstart(cast(pytest.Session, None))

    assert _git(repo, "rev-parse", "--show-toplevel") == str(repo)


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
        "scripts/affected.py",
        "scripts/certification_ci_changes.py",
        "Makefile",
        "noxfile.py",
        "pyproject.toml",
        "uv.lock",
        "tach.toml",
        "tests/power/test_binomial_planning.py",
        "tests/calibration/test_cuped_linear.py",
        "tests/parity_harness/cases.py",
        ".github/workflows/ci.yml",
        ".github/workflows/weekly.yml",
        ".github/workflows/nightly.yml",
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


def test_transitive_dependency_of_certified_module_requires_certification(repository):
    repo, base = repository
    _commit_change(repo, "increment/errors.py")
    assert _required(repo, base)


def test_dashboard_only_change_does_not_require_certification(repository):
    repo, base = repository
    _commit_change(repo, "increment/dashboard/_html.py")
    assert not _required(repo, base)


def test_facade_only_change_does_not_require_certification(repository):
    repo, base = repository
    _commit_change(repo, "increment/analysis.py")
    assert not _required(repo, base)


def test_new_import_in_a_certification_test_adds_a_graph_root(repository):
    from scripts.certification_ci_changes import _certified_modules

    repo, _ = repository
    roots = _certified_modules(repo)
    assert "increment.future_engine" not in roots
    test_module = repo / "tests/power/test_binomial_planning.py"
    test_module.write_text(test_module.read_text() + "\nimport increment.future_engine\n")
    assert "increment.future_engine" in _certified_modules(repo)


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


pytestmark = pytest.mark.integration
