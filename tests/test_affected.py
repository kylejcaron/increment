from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.affected import Selection, _pytest_arguments, _pytest_invocations, main, select

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
        "not slow and not parameter_recovery and not numerical_certification and not examples and not stress",
        "--tach-base",
        base,
        "--affected-tier",
        "fast",
        "--affected-report",
        ".affected-report-fast.json",
        "-p",
        "scripts.affected_filter",
        "-n",
        "2",
        "tests",
    ]


def test_prefiltered_selection_runs_kept_and_changed_tests_without_importing_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, base = _repo(
        tmp_path,
        {
            "tach.toml": (
                'source_roots = ["."]\nroot_module = "ignore"\n'
                '[[modules]]\npath = "increment"\ndepends_on = []\n'
                '[[modules]]\npath = "increment._literals"\ndepends_on = ["increment"]\n'
            ),
            "tests/conftest.py": (
                "def pytest_addoption(parser):\n"
                "    parser.addoption('--affected-report', default=None)\n"
                "    parser.addoption('--affected-tier', default='fast')\n"
            ),
            "increment/__init__.py": "",
            "increment/_literals.py": "VALUE = 1\n",
            "tests/test_kept.py": (
                "import increment._literals\n"
                "from pathlib import Path\n"
                "def test_kept(): Path('kept-ran').touch()\n"
            ),
            "tests/test_changed.py": (
                "import increment._literals\n"
                "from pathlib import Path\n"
                "def test_changed(): Path('changed-ran').touch()\n"
            ),
            "tests/test_unrelated.py": "raise AssertionError('unrelated module imported')\n",
        },
    )
    (root / "increment/_literals.py").write_text("VALUE = 2\n")
    (root / "tests/test_changed.py").write_text(
        "import increment._literals\n"
        "from pathlib import Path\n"
        "def test_changed(): Path('changed-ran').touch()\n"
    )
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.email=tests@example.com",
        "-c",
        "user.name=Tests",
        "commit",
        "--no-verify",
        "-qm",
        "changed selection inputs",
    )
    selection_base = _git(root, "rev-parse", "HEAD")
    (root / "tests/selection_map.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "base_sha": selection_base,
                "tier": "fast",
                "modules": {"increment._literals": ["tests/test_kept.py"]},
            }
        )
    )
    monkeypatch.chdir(root)
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).resolve().parents[1]))

    assert main(["--base", base]) == 0
    capsys.readouterr()
    assert (root / "kept-ran").exists()
    assert (root / "changed-ran").exists()


def test_filtered_invocation_collects_directory_and_loads_filter_plugin() -> None:
    selection = Selection(("tests/test_allowed.py",), (), True, False, ())

    (invocation,) = _pytest_invocations(
        selection, "base", "fast", [], kept=("tests/test_allowed.py",)
    )

    assert "tests" in invocation
    assert "tests/test_allowed.py" not in invocation
    assert "scripts.affected_filter" in invocation


def test_deselect_keeps_directory_when_tach_has_no_selection() -> None:
    selection = Selection((), (), True, False, ())

    (invocation,) = _pytest_invocations(
        selection, "base", "fast", ["--deselect=tests/test_one.py::test_one"], kept=None
    )

    assert "tests" in invocation


def test_filtered_collection_respects_ignore_and_deselect_and_skips_other_files(
    tmp_path: Path,
) -> None:
    root, _ = _repo(
        tmp_path,
        {
            "tests/test_ignored.py": "def test_ignored(): raise AssertionError('ignored')\n",
            "tests/test_kept.py": """def test_deselected(): raise AssertionError('deselected')
def test_kept(): pass
""",
            "tests/test_other.py": "raise AssertionError('imported unrelated test')\n",
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["AFFECTED_TEST_FILES"] = json.dumps(["tests/test_kept.py", "tests/test_ignored.py"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.affected_filter",
            "-v",
            "--ignore=tests/test_ignored.py",
            "--deselect=tests/test_kept.py::test_deselected",
            "tests",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "tests/test_kept.py::test_kept" in result.stdout
    assert "test_ignored" not in result.stdout
    assert "test_deselected" not in result.stdout
    assert "test_other" not in result.stdout


def test_filtered_collection_skips_imports_for_unkept_test_files(tmp_path: Path) -> None:
    root, _ = _repo(
        tmp_path,
        {
            "tests/test_kept.py": "def test_kept(): pass\n",
            "tests/test_other.py": "raise AssertionError('imported unrelated test')\n",
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env["AFFECTED_TEST_FILES"] = json.dumps(["tests/test_kept.py"])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.affected_filter",
            "--collect-only",
            "-q",
            "tests",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1 test collected" in result.stdout


def test_unfiltered_collection_without_tach_selection_collects_whole_tree(
    tmp_path: Path,
) -> None:
    root, _ = _repo(
        tmp_path,
        {
            "tests/test_one.py": "def test_one(): pass\n",
            "tests/test_two.py": "def test_two(): pass\n",
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    env.pop("AFFECTED_TEST_FILES", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "scripts.affected_filter",
            "--collect-only",
            "-q",
            "tests",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "2 tests collected" in result.stdout


def test_prefiltered_selection_with_nothing_kept_runs_nothing(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"increment/frame.py": "VALUE = 1\n"})
    selection = _change(root, base, "increment/frame.py", "VALUE = 2\n")

    assert _pytest_invocations(selection, base, "fast", [], kept=()) == []


def test_ignore_option_keeps_directory_collection_so_the_exclusion_applies(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"increment/frame.py": "VALUE = 1\n"})
    selection = _change(root, base, "increment/frame.py", "VALUE = 2\n")

    (invocation,) = _pytest_invocations(
        selection, base, "fast", ["--ignore=tests/test_b.py"], kept=("tests/test_b.py",)
    )

    assert "tests" in invocation
    assert "tests/test_b.py" not in invocation


def test_environment_ignore_excludes_module_from_source_change_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, base = _repo(
        tmp_path,
        {
            "increment/frame.py": "VALUE = 1\n",
            "tests/test_allowed.py": "def test_allowed(): pass\n",
            "tests/test_excluded.py": "def test_excluded(): pass\n",
        },
    )
    selection = _change(root, base, "increment/frame.py", "VALUE = 2\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore=tests/test_excluded.py")
    (invocation,) = _pytest_invocations(
        selection, base, "fast", [], kept=("tests/test_allowed.py", "tests/test_excluded.py")
    )
    pytest_args = []
    args = iter(invocation)
    for argument in args:
        if argument in {"--tach-base", "--affected-tier", "--affected-report"}:
            next(args)
        else:
            pytest_args.append(argument)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *pytest_args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
    )

    assert result.returncode == 0, result.stderr
    assert "tests/test_allowed.py::test_allowed" in result.stdout
    assert "tests/test_excluded.py::test_excluded" not in result.stdout


def test_deselect_preserves_prefiltered_files() -> None:
    selection = Selection(("tests/test_allowed.py",), (), True, False, ())

    (invocation,) = _pytest_invocations(
        selection,
        "base",
        "fast",
        ["--deselect=tests/test_allowed.py::test_one"],
        kept=("tests/test_allowed.py",),
    )

    assert "tests" in invocation
    assert "scripts.affected_filter" in invocation
    assert "tests/test_allowed.py" not in invocation


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
    (invocation,) = _pytest_invocations(selection, base, "fast", [])
    assert invocation[-2:] == ["tests", "docs/guides/cuped.md"]


def test_source_and_test_change_runs_tach_scope_through_filter_plugin(tmp_path: Path) -> None:
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
    (invocation,) = _pytest_invocations(selection, base, "fast", [])
    assert invocation[-1] == "tests"
    assert "scripts.affected_filter" in invocation
    assert "tests/test_explicit.py" not in invocation


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

    assert len(pytest_commands) == 1
    assert "--tach-base" in pytest_commands[0]
    assert pytest_commands[0][-1] == "docs/guides/cuped.md"
    assert capsys.readouterr().out == expected_stdout


def test_keywords_only_pyproject_change_does_not_run_full_tier(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path, {"pyproject.toml": '[project]\nname = "example"\nkeywords = ["old"]\n'}
    )
    selection = _change(
        root, base, "pyproject.toml", '[project]\nname = "example"\nkeywords = ["new"]\n'
    )
    assert selection.full_tier is False


def test_dependency_change_runs_full_tier(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"pyproject.toml": '[project]\ndependencies = ["a"]\n'})
    selection = _change(root, base, "pyproject.toml", '[project]\ndependencies = ["b"]\n')
    assert selection.full_tier is True


def test_tach_change_runs_tach_without_tests(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"tach.toml": "[tool.tach]\n"})
    selection = _change(root, base, "tach.toml", "[tool.tach]\nchanged = true\n")
    assert selection.use_tach is True
    assert selection.full_tier is False


def test_unit_tier_uses_unit_marker_expression(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"tests/test_example.py": "def test_old(): pass\n"})
    selection = _change(root, base, "tests/test_example.py", "def test_new(): pass\n")
    assert _pytest_arguments(selection, base, "unit", []) == [
        "-m",
        "not slow and not integration and not parameter_recovery and not numerical_certification and not examples and not stress",
        "tests/test_example.py",
    ]


def test_config_change_runs_full_tier(tmp_path: Path) -> None:
    root, base = _repo(tmp_path, {"pyproject.toml": "[tool.pytest.ini_options]\n"})

    selection = _change(root, base, "pyproject.toml", "[tool.pytest.ini_options]\naddopts = '-q'\n")

    assert selection.full_tier is True
    assert selection.test_files == ()


def test_full_tier_preserves_doc_pages_and_uncovered_paths(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "pyproject.toml": '[project]\ndependencies = ["a"]\n',
            "docs/guides/cuped.md": "# CUPED\n",
            "scripts/x.py": "VALUE = 1\n",
        },
    )
    (root / "pyproject.toml").write_text('[project]\ndependencies = ["b"]\n')
    (root / "docs/guides/cuped.md").write_text("# CUPED\nchanged\n")
    (root / "scripts/x.py").write_text("VALUE = 2\n")

    selection = select(base, root)

    assert selection.full_tier is True
    assert selection.doc_pages == ("docs/guides/cuped.md",)
    assert selection.uncovered == ("scripts/x.py",)
    assert _pytest_arguments(selection, base, "fast", []) == [
        "-m",
        "not slow and not parameter_recovery and not numerical_certification and not examples and not stress",
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

    from noxfile import TIER_MARKERS

    assert _pytest_arguments(selection, base, "slow", []) == [
        "-m",
        TIER_MARKERS["slow"],
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
        "not slow and not parameter_recovery and not numerical_certification and not examples and not stress",
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
            "not slow and not parameter_recovery and not numerical_certification and not examples and not stress",
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

    assert selection.test_files == ()


@pytest.mark.parametrize(
    ("changed_module", "test_file"),
    [
        (
            "increment/estimation/cuped/multicovariate.py",
            "tests/estimation/test_cuped_linear.py",
        ),
        (
            "increment/estimation/cuped/prepare.py",
            "tests/estimation/cuped/test_prepare.py",
        ),
    ],
)
def test_changed_cuped_lazy_export_is_delegated_to_handler(
    tmp_path: Path, changed_module: str, test_file: str
) -> None:
    root, base = _repo(
        tmp_path,
        {
            "increment/__init__.py": "",
            "increment/estimation/__init__.py": "",
            "increment/estimation/cuped/__init__.py": (
                "import importlib\n"
                "_LAZY_IMPORTS = {\n"
                "    'fit_cuped_multicovariate': "
                "('increment.estimation.cuped.multicovariate', 'fit_cuped_multicovariate'),\n"
                "    'prepare_multicovariate_fit': "
                "('increment.estimation.cuped.prepare', 'prepare_multicovariate_fit'),\n"
                "}\n"
                "def __getattr__(name):\n"
                "    module, attr = _LAZY_IMPORTS[name]\n"
                "    return getattr(importlib.import_module(module), attr)\n"
            ),
            "increment/estimation/cuped/multicovariate.py": "VALUE = 1\n",
            "increment/estimation/cuped/prepare.py": "VALUE = 1\n",
            test_file: (
                "from increment.estimation.cuped import "
                f"{'fit_cuped_multicovariate' if 'multicovariate' in changed_module else 'prepare_multicovariate_fit'}\n"
            ),
        },
    )

    selection = _change(root, base, changed_module, "VALUE = 2\n")

    assert selection.test_files == ()


def test_lazy_export_graph_is_delegated_to_handler(tmp_path: Path) -> None:
    root, base = _repo(
        tmp_path,
        {
            "increment/__init__.py": "",
            "increment/estimation/__init__.py": "",
            "increment/estimation/cuped/__init__.py": (
                "import importlib\n"
                "_LAZY_IMPORTS = {'prepare_multicovariate_fit': "
                "('increment.estimation.cuped.prepare', 'prepare_multicovariate_fit')}\n"
                "def __getattr__(name):\n"
                "    module, attr = _LAZY_IMPORTS[name]\n"
                "    return getattr(importlib.import_module(module), attr)\n"
            ),
            "increment/estimation/cuped/prepare.py": "VALUE = 1\n",
            "tests/cuped_support.py": (
                "from increment.estimation.cuped import prepare_multicovariate_fit\n"
            ),
            "tests/test_cuped_covariates.py": (
                "from tests.cuped_support import prepare_multicovariate_fit\n"
                "def test_cuped_covariates(): assert prepare_multicovariate_fit\n"
            ),
        },
    )

    selection = _change(root, base, "increment/estimation/cuped/prepare.py", "VALUE = 2\n")

    assert selection.test_files == ()
