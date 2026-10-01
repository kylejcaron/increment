"""Exercise filesystem guards through the real impact-selection plugin."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_impact_selection_preserves_filesystem_guards_and_filters(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    source = tmp_path / "increment"
    source.mkdir()
    (source / "__init__.py").write_text("")
    errors = source / "errors.py"
    errors.write_text("class CodedError(ValueError):\n    pass\n")
    (source / "untouched.py").write_text("VALUE = 1\n")
    (tmp_path / "tach.toml").write_text(
        'source_roots = ["."]\nroot_module = "ignore"\n'
        '[[modules]]\npath = "increment.errors"\ndepends_on = []\n'
        '[[modules]]\npath = "increment.untouched"\ndepends_on = []\n'
    )
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers = slow: filesystem scan\n")
    for name in ("conftest.py", "test_refusal_gate.py", "test_test_hygiene.py"):
        shutil.copyfile(Path(__file__).with_name(name), tests / name)
    (tests / "test_hygiene_allowlist.txt").write_text("# ---\n")
    (tests / "test_unrelated.py").write_text(
        "from increment.untouched import VALUE\n\ndef test_unrelated():\n    assert VALUE == 1\n"
    )
    environment = {
        **os.environ,
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    for arguments in (
        ["init", "-b", "main"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Baseline"],
    ):
        subprocess.run(
            ["git", *arguments], cwd=tmp_path, env=environment, check=True, capture_output=True
        )

    def run(*filters: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-p",
                "tach.pytest_plugin",
                "--tach-base=HEAD",
                "-q",
                *filters,
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    selected = "bare_raise_gate_is_empty or test_no_private_attribute_reach_ins or test_unrelated"
    clean = run("-k", selected)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "2 passed" in clean.stdout

    original = errors.read_bytes()
    errors.write_text(errors.read_text() + '\ndef violation():\n    raise ValueError("probe")\n')
    refusal = run("-k", selected)
    assert refusal.returncode == 1, refusal.stdout + refusal.stderr
    assert "test_bare_raise_gate_is_empty" in refusal.stdout
    assert "increment/errors.py::violation::ValueError::1" in refusal.stdout
    errors.write_bytes(original)

    (tests / "scan_only.py").write_text("def violation(obj):\n    return obj._plan\n")
    hygiene = run("-k", selected)
    assert hygiene.returncode == 1, hygiene.stdout + hygiene.stderr
    assert "test_no_private_attribute_reach_ins" in hygiene.stdout
    assert "tests/scan_only.py::violation::_plan" in hygiene.stdout

    fast = run("-k", selected, "-m", "not slow")
    assert fast.returncode == 0, fast.stdout + fast.stderr
    assert "1 passed" in fast.stdout

    unrelated = run("-k", "test_unrelated")
    assert unrelated.returncode == 0, unrelated.stdout + unrelated.stderr
    assert "passed" not in unrelated.stdout
