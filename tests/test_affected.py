from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.affected import Selection, _pytest_arguments, _pytest_invocations, select

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


def test_source_and_doc_change_runs_tach_scope_and_doc_page(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "increment/frame.py": "VALUE = 1\n",
            "docs/guides/cuped.md": "# CUPED\n",
        },
    )
    (root / "increment/frame.py").write_text("VALUE = 2\n")
    (root / "docs/guides/cuped.md").write_text("# CUPED\nchanged\n")

    selection = select(base, root)

    assert selection.use_tach is True
    assert selection.doc_pages == ("docs/guides/cuped.md",)
    assert _pytest_invocations(selection, base, "fast", []) == [
        [
            "-m",
            "not slow and not parameter_recovery and not numerical_certification",
            "--tach-base",
            base,
            "tests",
        ],
        [
            "-m",
            "not slow and not parameter_recovery and not numerical_certification",
            "docs/guides/cuped.md",
        ],
    ]


def test_source_and_test_change_runs_tach_scope_and_explicit_test(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "increment/frame.py": "VALUE = 1\n",
            "tests/test_explicit.py": "def test_old(): pass\n",
            "tests/test_other.py": "def test_other(): pass\n",
        },
    )
    (root / "increment/frame.py").write_text("VALUE = 2\n")
    (root / "tests/test_explicit.py").write_text("def test_new(): pass\n")

    selection = select(base, root)

    assert selection.use_tach is True
    assert selection.test_files == ("tests/test_explicit.py",)
    assert _pytest_invocations(selection, base, "fast", []) == [
        [
            "-m",
            "not slow and not parameter_recovery and not numerical_certification",
            "--tach-base",
            base,
            "--ignore=tests/test_explicit.py",
            "tests",
        ],
        [
            "-m",
            "not slow and not parameter_recovery and not numerical_certification",
            "tests/test_explicit.py",
        ],
    ]


def test_changed_test_module_selects_its_test_importers(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "tests/test_helpers.py": "VALUE = 1\n",
            "tests/test_sources.py": "def _plain_format_row(): pass\n",
            "tests/test_cate.py": (
                "from tests.test_sources import _plain_format_row\n"
                "def test_cate(): _plain_format_row()\n"
            ),
            "tests/test_other.py": "def test_other(): pass\n",
        },
    )
    (root / "tests/test_helpers.py").write_text("VALUE = 2\n")
    (root / "tests/test_sources.py").write_text("def _plain_format_row(): return None\n")
    selection = select(base, root)

    assert selection.test_files == (
        "tests/test_cate.py",
        "tests/test_helpers.py",
        "tests/test_sources.py",
    )


def test_changed_test_module_selects_two_hop_importing_tests(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "tests/test_unit_day_artifact_facade.py": (
                "from tests.source_conformance import check\ndef _definitions(): pass\n"
            ),
            "tests/source_conformance.py": (
                "from tests.test_unit_day_artifact_facade import _definitions\n"
            ),
            "tests/test_source_conformance.py": (
                "from tests.source_conformance import check\n"
                "def test_source_conformance(): check()\n"
            ),
        },
    )

    selection = _change(
        root,
        base,
        "tests/test_unit_day_artifact_facade.py",
        ("from tests.source_conformance import check\ndef _definitions(): return None\n"),
    )

    assert selection.test_files == (
        "tests/test_source_conformance.py",
        "tests/test_unit_day_artifact_facade.py",
    )


@pytest.mark.parametrize(
    ("pytest_statuses", "expected_stdout"),
    [
        ((0, 5), ""),
        ((5, 5), "affected: no affected tests selected\n"),
    ],
)
def test_empty_subrun_is_ignored_when_combining_mixed_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pytest_statuses: tuple[int, int],
    expected_stdout: str,
) -> None:
    from scripts.affected import main

    root, base = _repo(
        tmp_path,
        {
            "increment/frame.py": "VALUE = 1\n",
            "docs/guides/cuped.md": "# CUPED\n",
        },
    )
    (root / "increment/frame.py").write_text("VALUE = 2\n")
    (root / "docs/guides/cuped.md").write_text("# CUPED\nchanged\n")
    real_run = subprocess.run
    remaining_statuses = iter(pytest_statuses)
    pytest_commands: list[list[str]] = []

    def fake_run(command: list[str], *args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == [sys.executable, "-m", "pytest"]:
            pytest_commands.append(command)
            return subprocess.CompletedProcess(command, next(remaining_statuses))
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(root)

    assert main(["--base", base]) == 0

    assert len(pytest_commands) == 2
    assert "--tach-base" in pytest_commands[0]
    assert pytest_commands[1][-1] == "docs/guides/cuped.md"
    assert capsys.readouterr().out == expected_stdout


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
    _git(root, "add", "tests/test_existing.py")
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


def test_small_selection_runs_without_workers() -> None:
    """Small direct-file selections avoid xdist startup overhead."""
    selection = Selection(
        test_files=tuple(f"tests/test_{index}.py" for index in range(6)),
        doc_pages=(),
        use_tach=False,
        full_tier=False,
        uncovered=(),
    )
    assert _pytest_arguments(selection, "base", "fast", ["-n", "4", "--dist", "loadgroup"]) == [
        "-m",
        "not slow and not parameter_recovery and not numerical_certification",
        *selection.test_files,
    ]

    for worker_args in (
        ["-n4", "--dist=loadgroup"],
        ["-nauto"],
        ["--numprocesses", "4"],
        ["--numprocesses=4"],
    ):
        assert _pytest_arguments(selection, "base", "fast", worker_args) == [
            "-m",
            "not slow and not parameter_recovery and not numerical_certification",
            *selection.test_files,
        ]
    larger = Selection(
        test_files=(*selection.test_files, "tests/test_6.py"),
        doc_pages=(),
        use_tach=False,
        full_tier=False,
        uncovered=(),
    )
    assert _pytest_arguments(larger, "base", "fast", ["-n", "4", "--dist", "loadgroup"])[-4:] == [
        "-n",
        "4",
        "--dist",
        "loadgroup",
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


def test_python_module_entrypoint_runs_main_for_bad_base(tmp_path: Path) -> None:
    root, _ = _repo(tmp_path, {"tests/test_example.py": "def test_example(): pass\n"})
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(project_root), environment.get("PYTHONPATH")))
    )

    result = subprocess.run(
        [sys.executable, "-m", "scripts.affected", "--base", "nope"],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr == "affected: base is not a resolvable commit: nope\n"


def test_changed_lazy_export_dependency_selects_root_importing_test(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "increment/__init__.py": (
                "import importlib\n"
                "_LAZY_IMPORTS = {'PublicThing': ('increment.facade', 'PublicThing')}\n"
                "def __getattr__(name):\n"
                "    module, attr = _LAZY_IMPORTS[name]\n"
                "    return getattr(importlib.import_module(module), attr)\n"
            ),
            "increment/facade.py": "from increment.details import PublicThing\n",
            "increment/details.py": "class PublicThing: pass\n",
            "tests/test_public_api.py": (
                "from increment import PublicThing\ndef test_public_thing(): assert PublicThing\n"
            ),
        },
    )

    selection = _change(
        root,
        base,
        "increment/details.py",
        "class PublicThing:\n    VALUE = 1\n",
    )

    assert selection.test_files == ("tests/test_public_api.py",)
