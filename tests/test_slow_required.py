from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.slow_required import decide_paths, main

pytestmark = pytest.mark.tooling


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A package whose query layer imports a helper chain, and a slow test with its support."""
    files = {
        "increment/__init__.py": "",
        "increment/query/__init__.py": "",
        "increment/query/builders.py": (
            "def build():\n    from increment._shape import shape\n\n    return shape\n"
        ),
        "increment/_shape.py": "from increment._leaf import LEAF\n\nshape = LEAF\n",
        "increment/_leaf.py": "LEAF = 1\n",
        "increment/power/__init__.py": "",
        "increment/power/core.py": "from increment.query import builders\n",
        "tests/__init__.py": "",
        "tests/power/__init__.py": "",
        "tests/power/test_curve.py": (
            "from pathlib import Path\n\nimport pytest\n\nfrom ._procedures import oracle\n\n"
            "pytestmark = pytest.mark.slow\n"
            'GOLDENS = Path(__file__).parents[1] / "goldens"\n'
        ),
        "tests/power/_procedures.py": "from scripts.oracle import oracle\n",
        "tests/goldens/curve.json": "{}\n",
        "scripts/oracle.py": "def oracle():\n    return 1\n",
    }
    for name, text in files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text)
    return tmp_path


def test_documentation_only_change_skips_the_slow_tier(tree: Path) -> None:
    required, _ = decide_paths(["README.md", "docs/guides/power.md"], tree)
    assert not required


def test_dependency_change_requires_the_slow_tier(tree: Path) -> None:
    required, _ = decide_paths(["pyproject.toml"], tree)
    assert required


def test_module_that_only_imports_the_warehouse_path_skips_the_slow_tier(tree: Path) -> None:
    # increment.power.core imports the query layer, but nothing there imports it.
    required, _ = decide_paths(["increment/power/core.py"], tree)
    assert not required


def test_module_the_query_layer_imports_transitively_requires_the_slow_tier(tree: Path) -> None:
    # query.builders imports _shape inside a function; _shape imports _leaf.
    required, _ = decide_paths(["increment/_leaf.py"], tree)
    assert required


@pytest.mark.parametrize("path", ["scripts/oracle.py", "tests/goldens/curve.json"])
def test_code_and_data_a_slow_test_uses_require_the_slow_tier(tree: Path, path: str) -> None:
    required, _ = decide_paths([path], tree)
    assert required


def _git(repo: Path, *args: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _two_commits(repo: Path, changed: str) -> tuple[str, str]:
    """A repository whose second commit writes ``changed``; return both commit ids."""
    _git(repo, "init", "-q")
    (repo / "README.md").write_text("Increment\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / changed).parent.mkdir(parents=True, exist_ok=True)
    (repo / changed).write_text("changed\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "change")
    return base, _git(repo, "rev-parse", "HEAD")


@pytest.mark.integration
def test_unresolvable_base_requires_the_slow_tier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base, head = _two_commits(tmp_path, "docs/guide.md")
    monkeypatch.chdir(tmp_path)

    main(["--base", base, "--head", head])
    main(["--base", "0" * 40, "--head", head])

    decisions = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("required=")
    ]
    assert decisions == ["required=false", "required=true"]


@pytest.mark.integration
def test_unimportable_import_graph_still_requires_the_slow_tier(tmp_path: Path) -> None:
    base, head = _two_commits(tmp_path, "increment/core.py")
    # A None entry makes importing the graph module raise ImportError, as a missing file would.
    runner = (
        "import runpy, sys; "
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
        "sys.modules['scripts.check_deferred_imports'] = None; "
        "runpy.run_module('scripts.slow_required', run_name='__main__')"
    )
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    decisions = [
        subprocess.run(
            [sys.executable, "-c", runner, "--event", event, "--base", base, "--head", head],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.partition("\n")[0]
        for event in ("pull_request", "push")
    ]

    assert decisions == ["required=true", "required=true"]
