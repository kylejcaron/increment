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


def test_focused_runner_does_not_execute_deselected_slow_tests(tmp_path):
    project = Path(__file__).parents[1]
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\nmarkers = slow: slow tier\n parameter_recovery: recovery tier\n")
    test_file = tmp_path / "test_mixed.py"
    test_file.write_text(
        "import pytest\n"
        "def test_fast():\n    print('FAST_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "def test_slow():\n    print('SLOW_EXECUTED')\n"
        "@pytest.mark.parameter_recovery\n"
        "def test_recovery():\n    print('RECOVERY_EXECUTED')\n"
    )
    environment = {
        **os.environ,
        "INCREMENT_AFFECTED_EVIDENCE": json.dumps({"tier": "fast"}),
        "INCREMENT_AFFECTED_TEST_PATHS": "[]",
        "INCREMENT_AFFECTED_DYNAMIC_TEST_PATHS": "[]",
        "INCREMENT_AFFECTED_WORKER_ALLOWANCE": "8 (dist=loadgroup)",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_test_tier",
            "focused",
            str(test_file),
            "-c",
            str(config),
            "-q",
            "-s",
        ],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FAST_EXECUTED" in result.stdout
    assert "SLOW_EXECUTED" not in result.stdout
    assert "RECOVERY_EXECUTED" not in result.stdout
    assert "1 passed, 2 deselected" in result.stdout


def test_tier_marker_selection_remains_scoped_to_declared_policy():
    from scripts._test_tier_policy import TIER_MARKERS
    from scripts.run_test_tier import TIERS

    assert TIERS["fast"].marker == TIER_MARKERS["fast"]
    assert TIERS["slow"].marker == TIER_MARKERS["slow"]
    assert TIERS["all"].marker == ""
    assert TIERS["examples"].marker == "examples"


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("fast", {"FAST_EXECUTED"}),
        ("slow", {"SLOW_EXECUTED"}),
        (
            "all",
            {"FAST_EXECUTED", "SLOW_EXECUTED", "RECOVERY_EXECUTED", "EXAMPLE_EXECUTED"},
        ),
        ("examples", {"EXAMPLE_EXECUTED"}),
    ],
)
def test_tier_markers_execute_only_their_declared_selection(tmp_path, tier, expected):
    project = Path(__file__).parents[1]
    test_file = tmp_path / "test_markers.py"
    test_file.write_text(
        "import pytest\n"
        "def test_fast():\n    print('FAST_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "def test_slow():\n    print('SLOW_EXECUTED')\n"
        "@pytest.mark.parameter_recovery\n"
        "def test_recovery():\n    print('RECOVERY_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "@pytest.mark.examples\n"
        "def test_example():\n    print('EXAMPLE_EXECUTED')\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_test_tier", tier, str(test_file), "-q", "-s"],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = {
        marker
        for marker in (
            "FAST_EXECUTED",
            "SLOW_EXECUTED",
            "RECOVERY_EXECUTED",
            "EXAMPLE_EXECUTED",
        )
        if marker in result.stdout
    }
    assert observed == expected


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


def test_affected_cli_documents_option_and_rejects_invalid_base_before_pytest():
    project = Path(__file__).resolve().parents[1]
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("INCREMENT_")
    }
    help_result = subprocess.run(
        [sys.executable, "-m", "scripts.run_test_tier", "--help"],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert help_result.returncode == 0, help_result.stdout + help_result.stderr
    assert "--affected-base" in help_result.stdout

    invalid = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_test_tier",
            "--affected-base",
            "not-a-commit",
            "fast",
        ],
        cwd=project,
        env=environment,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert invalid.returncode != 0
    assert "collected" not in invalid.stdout + invalid.stderr


def test_runner_and_entrypoint_import_without_pytest_installed() -> None:
    project = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import scripts.run_test_tier; import scripts.run_test_entrypoint",
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
