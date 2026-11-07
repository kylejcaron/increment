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
    """A package with a lazy root export and a slow test that imports, runs and reads support."""
    files = {
        "increment/__init__.py": (
            "import importlib\n\n"
            '_LAZY_IMPORTS = {"build": ("increment.query.builders", "build")}\n\n\n'
            "def __getattr__(name):\n"
            "    module_name, attribute = _LAZY_IMPORTS[name]\n"
            "    return getattr(importlib.import_module(module_name), attribute)\n"
        ),
        "increment/query/__init__.py": "",
        "increment/query/builders.py": (
            "def build():\n    from increment._shape import shape\n\n    return shape\n"
        ),
        "increment/_shape.py": "from increment._leaf import LEAF\n\nshape = LEAF\n",
        "increment/_leaf.py": "LEAF = 1\n",
        "increment/power/__init__.py": "",
        "increment/power/core.py": "from increment.query import builders\n",
        "increment/simulate/__init__.py": "",
        "increment/simulate/runner.py": (
            "from importlib.metadata import entry_points\n\n\n"
            "def run_end_to_end():\n"
            '    return entry_points(group="increment.ports")\n'
        ),
        "increment/ports_impl.py": "def load():\n    return 1\n",
        "pyproject.toml": (
            '[project.entry-points."increment.ports"]\nbuiltin = "increment.ports_impl:load"\n'
        ),
        "calibration/__init__.py": "",
        "calibration/unit_cycle.py": "print('campaign')\n",
        "calibration/journal.py": "print('journal')\n",
        "tests/__init__.py": "",
        "tests/power/__init__.py": "",
        "tests/power/test_curve.py": (
            "import runpy\nimport subprocess\nimport sys\nfrom pathlib import Path\n\n"
            "import pytest\n\n"
            "from increment import build\nfrom increment.simulate.runner import run_end_to_end\n\n"
            "from ._procedures import oracle\n\n"
            "pytestmark = pytest.mark.slow\n"
            'GOLDENS = Path(__file__).parents[1] / "goldens"\n\n\n'
            "def test_campaign():\n"
            '    subprocess.run([sys.executable, "-m", "calibration.unit_cycle"], check=True)\n'
            '    runpy.run_module("calibration.journal")\n'
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


def test_module_no_slow_test_reaches_skips_the_slow_tier(tree: Path) -> None:
    # increment.power.core imports what the slow test uses, but nothing imports it.
    required, _ = decide_paths(["increment/power/core.py"], tree)
    assert not required


def test_module_reached_through_a_lazy_export_and_deferred_imports_requires_the_slow_tier(
    tree: Path,
) -> None:
    # increment.build resolves to query.builders, which imports _shape inside a function.
    required, _ = decide_paths(["increment/_leaf.py"], tree)
    assert required


@pytest.mark.parametrize(
    "path",
    [
        "increment/simulate/runner.py",
        "increment/ports_impl.py",
        "calibration/unit_cycle.py",
        "calibration/journal.py",
        "scripts/oracle.py",
        "tests/goldens/curve.json",
    ],
)
def test_code_and_data_a_slow_test_uses_require_the_slow_tier(tree: Path, path: str) -> None:
    # Imported directly, loaded as an entry point, run with -m, run with runpy, imported
    # through a helper, read.
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
