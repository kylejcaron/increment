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
        '[[modules]]\npath = "increment.power._binomial"\ndepends_on = ["increment.power._search"]\n'
        '[[modules]]\npath = "increment.power._search"\ndepends_on = ["increment.power.core"]\n'
        '[[modules]]\npath = "increment.power.core"\ndepends_on = ["increment.errors"]\n'
        '[[modules]]\npath = "increment.estimation"\ndepends_on = ["increment.estimation.variance"]\n'
        '[[modules]]\npath = "increment.estimation.variance"\ndepends_on = ["increment.errors"]\n'
        '[[modules]]\npath = "increment.analysis"\ndepends_on = ["increment.estimation", "increment.errors"]\n'
        '[[modules]]\npath = "increment.dashboard"\ndepends_on = []\n'
        '[[modules]]\npath = "increment._frame_moments"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.query.builders"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.query.artifact_reader"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.sources"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.estimation.sequential_runtime"\ndepends_on = []\n'
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
        "tests/estimation/test_sequential_public_proof_acceptance.py": (
            "import increment.estimation.sequential_runtime\n"
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


def _required(
    repo: Path, base: str, event: str = "pull_request", *, gate: str = "certification"
) -> bool:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gate",
            gate,
            "--event",
            event,
            "--base",
            base,
            "--head",
            "HEAD",
        ],
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


@pytest.mark.parametrize(
    "path",
    ["increment/power/_search.py", "increment/estimation/variance.py"],
)
def test_indirect_certified_dependencies_require_certification(repository, path: str):
    repo, base = repository
    _commit_change(repo, path)
    assert _required(repo, base)


@pytest.mark.parametrize(
    "path",
    [
        "increment/_frame_moments.py",
        "increment/query/builders.py",
        "increment/query/artifact_reader.py",
        "increment/sources.py",
    ],
)
def test_ingress_moment_change_requires_certification(repository, path: str):
    repo, base = repository
    _commit_change(repo, path)
    assert _required(repo, base)


def test_sequential_proof_module_change_requires_certification(repository):
    repo, base = repository
    _commit_change(repo, "increment/estimation/sequential_runtime.py")
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


@pytest.mark.grid("each exhaustive trigger path is a separate selector contract")
@pytest.mark.parametrize(
    "path",
    [
        "increment/query/_cuped_qr.py",
        "increment/sources.py",
        "increment/frame.py",
        "increment/_frame_moments.py",
        "increment/readouts/_evidence.py",
        "increment/readouts/_provider.py",
        "increment/_evidence_loaders.py",
        "increment/_source_types.py",
        "tests/parity_harness/cases.py",
        "tests/evidence_fixtures/FAULTS.md",
        "tests/fixtures/evidence_identity/primitive-recordings/scenario__ingress__0123abcd.json.gz",
        "scripts/certification_ci_changes.py",
        ".github/workflows/ci.yml",
        "noxfile.py",
        "Makefile",
        "pyproject.toml",
        "uv.lock",
    ],
)
def test_loader_change_requires_exhaustive(repository, path: str):
    repo, base = repository
    _commit_change(repo, path)
    assert _required(repo, base, gate="exhaustive")


@pytest.mark.parametrize(
    "path",
    [
        "increment/estimation/variance.py",
        "increment/readouts/_breakout.py",
        "increment/dashboard/_html.py",
        "tests/test_behavior.py",
        "docs/guide.md",
        ".github/workflows/weekly.yml",
    ],
)
def test_change_outside_loaders_and_their_grids_does_not_require_exhaustive(repository, path: str):
    repo, base = repository
    _commit_change(repo, path)
    assert not _required(repo, base, gate="exhaustive")


def test_change_to_a_module_holding_exhaustive_tests_requires_exhaustive(repository):
    repo, base = repository
    module = repo / "tests" / "test_grid.py"
    module.write_text("import pytest\n\npytestmark = [pytest.mark.slow, pytest.mark.exhaustive]\n")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", "add a grid")
    assert _required(repo, base, gate="exhaustive")


def test_pushes_dispatches_and_unknown_bases_require_exhaustive(repository):
    repo, _ = repository
    assert _required(repo, "0" * 40, gate="exhaustive")
    for event in ("push", "workflow_dispatch", "schedule"):
        assert _required(repo, "missing-base", event, gate="exhaustive"), event


pytestmark = [pytest.mark.integration, pytest.mark.tooling]
