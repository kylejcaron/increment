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


@pytest.mark.slow
@pytest.mark.parametrize("github_actions", [None, "false", "true"])
@pytest.mark.parametrize("entrypoint", ["tier", "examples"])
def test_cli_performance_budget_is_local_only(tmp_path, github_actions, entrypoint):
    project = Path(__file__).parents[1]
    environment = dict(os.environ)
    if github_actions is None:
        environment.pop("GITHUB_ACTIONS", None)
    else:
        environment["GITHUB_ACTIONS"] = github_actions
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    failing = tmp_path / "test_failure.py"
    failing.write_text(
        "import time\n\ndef test_failure():\n"
        "    time.sleep(0.2)\n"
        "    assert False, 'test failures must still propagate'\n"
    )
    if entrypoint == "tier":
        code = (
            "from scripts.run_test_tier import TIERS, Tier, main\n"
            "TIERS['focused'] = Tier(marker='', budget_seconds=0.05)\n"
            f"raise SystemExit(main(['focused', {str(failing)!r}, "
            f"'-c', {str(config)!r}, '-q']))\n"
        )
    else:
        code = (
            "import sys\n"
            "from scripts.run_test_entrypoint import ENTRYPOINTS, Entrypoint, main\n"
            "ENTRYPOINTS['examples'] = Entrypoint(\n"
            f"    command=(sys.executable, '-m', 'pytest', '-c', {str(config)!r}, "
            f"{str(failing)!r}, '-q'),\n"
            "    budget_seconds=0.05, cleanup_grace_seconds=1)\n"
            "raise SystemExit(main(['examples']))\n"
        )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == (1 if github_actions == "true" else 124), (
        result.stdout + result.stderr
    )
