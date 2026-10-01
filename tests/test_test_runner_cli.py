from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def test_all_cli_help_needs_no_private_authorization():
    project = Path(__file__).parents[1]
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("INCREMENT_")
    }
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_test_tier", "all", "--help"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--evidence-root" in result.stdout


def test_focused_cli_refuses_any_missing_selector_among_options(tmp_path):
    """A missing file after an option must refuse, not silently run nothing."""
    project = Path(__file__).parents[1]
    existing = tmp_path / "existing.py"
    existing.write_text("def test_existing():\n    pass\n")
    missing = tmp_path / "missing.py"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_test_tier",
            "focused",
            str(existing),
            "-n",
            "2",
            str(missing),
        ],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert str(missing) in result.stderr


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make")
def test_make_test_keeps_a_quoted_expression_in_the_focused_tier(tmp_path):
    """A quoted -k expression in TESTS must reach the focused runner intact."""
    project = Path(__file__).parents[1]
    received = tmp_path / "argv.json"
    runner = tmp_path / "runner.py"
    runner.write_text(
        f"import json, sys\nopen({str(received)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
    )
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "test",
            f"TEST_RUNNER={sys.executable} {runner}",
            'TESTS=tests/test_workflows.py -k "pinned and not flow"',
        ],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(received.read_text()) == [
        "focused",
        "tests/test_workflows.py",
        "-k",
        "pinned and not flow",
        "-x",
        "-q",
    ]
