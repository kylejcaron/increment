from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.affected import Selection, _pytest_arguments, select

pytestmark = pytest.mark.integration


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Tests")
    for name, contents in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    base = _git(root, "rev-parse", "HEAD")
    return root, base


def _change(root: Path, base: str, name: str, contents: str) -> Selection:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return select(base, root)


def test_docs_only_change_selects_only_that_page(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"docs/guides/cuped.md": "# CUPED\n"})

    selection = _change(root, base, "docs/guides/cuped.md", "# CUPED\nchanged\n")

    assert selection.doc_pages == ("docs/guides/cuped.md",)
    assert selection.use_tach is False
    assert selection.test_files == ()


def test_changed_test_file_is_selected(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"tests/test_example.py": "def test_old(): pass\n"})

    selection = _change(root, base, "tests/test_example.py", "def test_new(): pass\n")

    assert selection.test_files == ("tests/test_example.py",)
    assert selection.use_tach is False


def test_source_change_uses_tach(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"increment/frame.py": "VALUE = 1\n"})

    selection = _change(root, base, "increment/frame.py", "VALUE = 2\n")

    assert selection.use_tach is True
    assert selection.test_files == ()
    assert _pytest_arguments(selection, base, "fast", ["-n", "2"]) == [
        "-m",
        "not slow and not parameter_recovery and not numerical_certification",
        "--tach-base",
        base,
        "-n",
        "2",
    ]


def test_config_change_runs_full_tier(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"pyproject.toml": "[tool]\n"})

    selection = _change(root, base, "pyproject.toml", "[tool]\nchanged = true\n")

    assert selection.full_tier is True
    assert selection.test_files == ()


def test_full_tier_preserves_doc_pages_and_uncovered_paths(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "pyproject.toml": "[tool]\n",
            "docs/guides/cuped.md": "# CUPED\n",
            "scripts/x.py": "VALUE = 1\n",
        },
    )
    (root / "pyproject.toml").write_text("[tool]\nchanged = true\n")
    (root / "docs/guides/cuped.md").write_text("# CUPED\nchanged\n")
    (root / "scripts/x.py").write_text("VALUE = 2\n")

    selection = select(base, root)

    assert selection.full_tier is True
    assert selection.doc_pages == ("docs/guides/cuped.md",)
    assert selection.uncovered == ("scripts/x.py",)
    assert _pytest_arguments(selection, base, "fast", []) == [
        "-m",
        "not slow and not parameter_recovery and not numerical_certification",
        "tests",
        "docs/guides/cuped.md",
    ]


def test_tooling_change_is_reported_uncovered(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"scripts/x.py": "VALUE = 1\n"})

    selection = _change(root, base, "scripts/x.py", "VALUE = 2\n")

    assert selection.uncovered == ("scripts/x.py",)
    assert selection.test_files == ()


def test_working_tree_changes_are_selected(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "tests/test_existing.py": "def test_old(): pass\n",
            "tests/test_staged.py": "def test_old(): pass\n",
        },
    )
    (root / "tests/test_existing.py").write_text("def test_changed(): pass\n")
    (root / "tests/test_new.py").write_text("def test_new(): pass\n")

    selection = select(base, root)

    assert selection.test_files == ("tests/test_existing.py", "tests/test_new.py")


def test_deleted_test_file_is_skipped(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"tests/test_removed.py": "def test_old(): pass\n"})
    (root / "tests/test_removed.py").unlink()

    selection = select(base, root)

    assert selection.test_files == ()


def test_tests_conftest_selects_tests_in_its_directory(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "tests/conftest.py": "VALUE = 1\n",
            "tests/test_root.py": "def test_root(): pass\n",
            "tests/unit/test_nested.py": "def test_nested(): pass\n",
            "tests/other/test_other.py": "def test_other(): pass\n",
        },
    )

    selection = _change(root, base, "tests/conftest.py", "VALUE = 2\n")

    assert selection.test_files == (
        "tests/other/test_other.py",
        "tests/test_root.py",
        "tests/unit/test_nested.py",
    )


def test_readme_is_selected_as_a_doc_page(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"README.md": "# Project\n"})

    selection = _change(root, base, "README.md", "# Project\nchanged\n")

    assert selection.doc_pages == ("README.md",)


def test_slow_tier_marker_expression_is_used(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"tests/test_example.py": "def test_old(): pass\n"})
    selection = _change(root, base, "tests/test_example.py", "def test_new(): pass\n")

    assert _pytest_arguments(selection, base, "slow", []) == [
        "-m",
        "slow and not parameter_recovery and not numerical_certification and not examples",
        "tests/test_example.py",
    ]


def test_unknown_base_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.affected import main

    root, _ = _repo(tmp_path, {"tests/test_example.py": "def test_example(): pass\n"})
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.chdir(root)
    try:
        assert main(["--base", "nope"]) == 2
    finally:
        monkeypatch.undo()

    assert capsys.readouterr().err == "affected: base is not a resolvable commit: nope\n"
