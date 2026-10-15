from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.run_test_tier import _clean_pytest_environment

pytestmark = pytest.mark.xdist_group("test_runner_subprocesses")


@pytest.mark.integration
def test_focused_cli_isolates_git_repositories_from_invoking_hook(tmp_path: Path):
    project = Path(__file__).resolve().parents[1]
    environment = {
        key: value
        for key, value in _clean_pytest_environment(os.environ).items()
        if not key.startswith(("GIT_", "INCREMENT_", "TESTMON_"))
    }
    owner = tmp_path / "owner"
    subprocess.run(["git", "init", "-q", str(owner)], env=environment, check=True)
    (owner / "owned").write_text("owner repository\n")
    for arguments in (
        ["add", "."],
        [
            "-c",
            "user.name=Owner",
            "-c",
            "user.email=owner@example.invalid",
            "commit",
            "-qm",
            "owner",
        ],
    ):
        subprocess.run(["git", *arguments], cwd=owner, env=environment, check=True)
    original_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=owner, env=environment
    )
    original_index = (owner / ".git" / "index").read_bytes()
    original_config = (owner / ".git" / "config").read_bytes()
    consumer = tmp_path / "test_foreign_repository.py"
    consumer.write_text(
        "from pathlib import Path\n"
        "import subprocess\n"
        "def test_foreign_repository(tmp_path):\n"
        "    foreign = tmp_path / 'foreign'\n"
        "    subprocess.run(['git', 'init', '-q', str(foreign)], check=True)\n"
        "    assert (foreign / '.git').is_dir()\n"
        "    (foreign / 'independent').write_text('foreign repository\\n')\n"
        "    subprocess.run(['git', 'add', '.'], cwd=foreign, check=True)\n"
        "    subprocess.run(['git', '-c', 'user.name=Foreign', '-c',\n"
        "                    'user.email=foreign@example.invalid', 'commit', '-qm',\n"
        "                    'foreign'], cwd=foreign, check=True)\n"
        "    assert subprocess.check_output(['git', 'show', 'HEAD:independent'],\n"
        "                                   cwd=foreign) == b'foreign repository\\n'\n"
    )
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    environment.update(
        {
            "PYTHONPATH": str(project),
            "GIT_DIR": str(owner / ".git"),
            "GIT_COMMON_DIR": str(owner / ".git"),
            "GIT_WORK_TREE": str(owner),
            "GIT_INDEX_FILE": str(owner / ".git" / "index"),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_test_tier",
            "focused",
            str(consumer),
            "-c",
            str(config),
            "-q",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for name, original in (("index", original_index), ("config", original_config)):
        assert (owner / ".git" / name).read_bytes() == original
    restored_environment = {
        key: value for key, value in environment.items() if not key.startswith("GIT_")
    }
    assert (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=owner, env=restored_environment)
        == original_head
    )


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["focused", "nested", "filtered_fast", "sharded_fast"])
def test_partial_feedback_is_independent_of_corrupt_coverage_map(tmp_path: Path, mode: str):
    from scripts._test_impact import testmon_database_lock

    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "pytest.ini"
    config.write_text("[pytest]\n")
    consumer = tmp_path / "test_consumer.py"
    consumer.write_text("from sample import value\ndef test_value():\n    assert value() == 1\n")
    source = tmp_path / "sample.py"
    database = tmp_path / ".testmondata"
    database.write_bytes(b"corrupt irrelevant coverage cache")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PYTEST_", "INCREMENT_", "TESTMON_"))
    }
    environment["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(project)))
    if mode == "nested":
        environment["PYTEST_CURRENT_TEST"] = os.environ["PYTEST_CURRENT_TEST"]
    arguments = (
        ["focused", str(consumer)] if mode in {"focused", "nested"} else ["fast", str(consumer)]
    )
    if mode == "sharded_fast":
        arguments.extend(("--splits", "2", "--group", "1"))
    with testmon_database_lock(tmp_path):
        for value, status in ((1, 0), (22, 1)):
            source.write_text(f"def value():\n    return {value}\n")
            evidence_root = tmp_path / f"evidence-{value}"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.run_test_tier",
                    *arguments,
                    "-c",
                    str(config),
                    "-q",
                    "--evidence-root",
                    str(evidence_root),
                ],
                cwd=tmp_path,
                env=environment,
                capture_output=True,
                text=True,
                timeout=45,
            )
            assert result.returncode == status, result.stdout + result.stderr
            [receipt] = evidence_root.glob("run-*/run.json")
            evidence = json.loads(receipt.read_text())
            assert evidence["executed_tests"] == 1
            reports = [
                json.loads(line)
                for line in receipt.with_name("reports.jsonl").read_text().splitlines()
            ]
            calls = [
                (report["nodeid"], report["outcome"])
                for report in reports
                if report.get("when") == "call"
            ]
            assert calls == [
                ("test_consumer.py::test_value", "passed" if status == 0 else "failed")
            ]
        assert database.read_bytes() == b"corrupt irrelevant coverage cache"


def test_all_cli_help_needs_no_private_authorization():
    project = Path(__file__).parents[1]
    environment = {
        key: value
        for key, value in _clean_pytest_environment(os.environ).items()
        if not key.startswith("INCREMENT_")
    }
    for key in ("INCREMENT_TESTMON_LOCK_OWNER", "INCREMENT_TESTMON_LOCK_ROLE"):
        if key in os.environ:
            environment[key] = os.environ[key]
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
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
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
        env=environment,
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
    config.write_text(
        "[pytest]\nmarkers = slow: slow tier\n parameter_recovery: recovery tier\n"
        " numerical_certification: deterministic certification tier\n"
    )
    test_file = tmp_path / "test_mixed.py"
    test_file.write_text(
        "import pytest\n"
        "def test_fast():\n    print('FAST_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "def test_slow():\n    print('SLOW_EXECUTED')\n"
        "@pytest.mark.parameter_recovery\n"
        "def test_recovery():\n    print('RECOVERY_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "@pytest.mark.numerical_certification\n"
        "def test_certification():\n    print('CERTIFICATION_EXECUTED')\n"
    )
    environment = {
        **_clean_pytest_environment(os.environ),
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
    assert "CERTIFICATION_EXECUTED" not in result.stdout
    assert "1 passed, 3 deselected" in result.stdout


def test_focused_runner_uses_serial_mode_below_recorded_xdist_overhead(tmp_path):
    from scripts.run_test_tier import _serial_if_small_focused_selection

    project = tmp_path
    test_file = project / "tests" / "test_small.py"
    test_file.parent.mkdir()
    test_file.write_text("")
    durations = project / ".github" / "fast-test-durations.json"
    durations.parent.mkdir()
    durations.write_text('{"tests/test_small.py::test_a": 0.5, "tests/test_small.py::test_b": 1.0}')

    selected = _serial_if_small_focused_selection(
        "focused",
        [str(test_file), "-n", "4", "--dist", "loadgroup", "-q"],
        project,
    )

    assert selected == [str(test_file), "-q"]


def test_focused_runner_keeps_workers_for_larger_recorded_selections(tmp_path):
    from scripts.run_test_tier import _serial_if_small_focused_selection

    project = tmp_path
    test_file = project / "tests" / "test_large.py"
    test_file.parent.mkdir()
    test_file.write_text("")
    durations = project / ".github" / "fast-test-durations.json"
    durations.parent.mkdir()
    durations.write_text('{"tests/test_large.py::test_a": 5.0}')
    args = [str(test_file), "-n", "4", "--dist", "loadgroup"]

    assert _serial_if_small_focused_selection("focused", args, project) == args


def test_focused_runner_requires_every_selected_file_to_be_small(tmp_path):
    from scripts.run_test_tier import _serial_if_small_focused_selection

    project = tmp_path
    first = project / "tests" / "test_one.py"
    second = project / "tests" / "test_two.py"
    first.parent.mkdir()
    first.write_text("")
    second.write_text("")
    durations = project / ".github" / "fast-test-durations.json"
    durations.parent.mkdir()
    durations.write_text('{"tests/test_one.py::test_a": 1.0, "tests/test_two.py::test_b": 3.5}')
    args = [str(first), str(second), "-n", "2", "--dist", "loadgroup"]

    assert _serial_if_small_focused_selection("focused", args, project) == args


def test_focused_runner_keeps_workers_for_unrecorded_node_selector(tmp_path):
    from scripts.run_test_tier import _serial_if_small_focused_selection

    project = tmp_path
    test_file = project / "tests" / "test_small.py"
    test_file.parent.mkdir()
    test_file.write_text("")
    durations = project / ".github" / "fast-test-durations.json"
    durations.parent.mkdir()
    durations.write_text('{"tests/test_small.py::test_a": 0.5}')
    args = [f"{test_file}::test_new", "-n", "2", "--dist", "loadgroup"]

    assert _serial_if_small_focused_selection("focused", args, project) == args


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("fast", {"FAST_EXECUTED"}),
        ("slow", {"SLOW_EXECUTED"}),
        ("scientific", {"RECOVERY_EXECUTED", "CERTIFICATION_EXECUTED"}),
        ("parameter-recovery", {"RECOVERY_EXECUTED"}),
        ("numerical-certification", {"CERTIFICATION_EXECUTED"}),
        (
            "all",
            {
                "FAST_EXECUTED",
                "SLOW_EXECUTED",
                "RECOVERY_EXECUTED",
                "CERTIFICATION_EXECUTED",
                "EXAMPLE_EXECUTED",
            },
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
        "@pytest.mark.slow\n"
        "@pytest.mark.parameter_recovery\n"
        "def test_recovery():\n    print('RECOVERY_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "@pytest.mark.numerical_certification\n"
        "def test_certification():\n    print('CERTIFICATION_EXECUTED')\n"
        "@pytest.mark.slow\n"
        "@pytest.mark.examples\n"
        "def test_example():\n    print('EXAMPLE_EXECUTED')\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.run_test_tier", tier, str(test_file), "-q", "-s"],
        cwd=project,
        env=_clean_pytest_environment(os.environ),
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
            "CERTIFICATION_EXECUTED",
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
    environment = _clean_pytest_environment(os.environ)
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
        key: value
        for key, value in _clean_pytest_environment(os.environ).items()
        if not key.startswith("INCREMENT_")
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


def _map_recording_project(root: Path) -> tuple[dict[str, str], str]:
    project = Path(__file__).resolve().parents[1]
    (root / "scripts").mkdir()
    (root / "scripts" / "__init__.py").touch()
    for name in (
        "run_test_tier.py",
        "_test_tier_policy.py",
        "_test_impact.py",
        "run_test_tier_plugin.py",
        "affected_pytest_launcher.py",
    ):
        shutil.copyfile(project / "scripts" / name, root / "scripts" / name)
    (root / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        "timeout = 300\n"
        'markers = ["slow: expensive functional test"]\n'
        "filterwarnings = []\n"
    )
    (root / "tach.toml").write_text(
        'source_roots = ["."]\nroot_module = "ignore"\n'
        '[[modules]]\npath = "increment._state"\ndepends_on = []\n'
    )
    shutil.copyfile(project / "conftest.py", root / "conftest.py")
    package = root / "increment"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "_state.py").write_text("def compute(value):\n    return 2 * value + 1\n")
    tests = root / "tests"
    tests.mkdir()
    (tests / "test_consumer.py").write_text(
        "from pathlib import Path\n"
        "import pytest\n"
        "from increment._state import compute\n"
        "def test_fast():\n"
        "    Path('executed-fast').touch()\n"
        "    assert compute(1) == 3\n"
        "@pytest.mark.slow\n"
        "def test_slow():\n"
        "    Path('executed-slow').touch()\n"
        "    assert compute(2) == 5\n"
    )
    (root / ".gitignore").write_text(
        "__pycache__/\n.pytest_cache/\n.testmondata*\n.test_durations\n.evidence/\nexecuted-*\n"
    )
    for arguments in (
        ["init", "-q"],
        ["config", "user.name", "Runner Tests"],
        ["config", "user.email", "runner-tests@example.invalid"],
        ["add", "."],
        ["commit", "-qm", "baseline"],
    ):
        subprocess.run(["git", *arguments], cwd=root, check=True)
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    environment = {
        key: value
        for key, value in _clean_pytest_environment(os.environ).items()
        if not key.startswith(("INCREMENT_", "TESTMON_"))
    }
    environment["PYTHONPATH"] = str(project)
    return environment, base


def _run_map_project(root: Path, environment: dict[str, str], *arguments: str):
    return subprocess.run(
        [sys.executable, "-m", "scripts.run_test_tier", *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("tier", ["fast", "slow"])
def test_full_validation_is_independent_of_dependency_map(tmp_path: Path, tier: str):
    environment, _ = _map_recording_project(tmp_path)
    database = tmp_path / ".testmondata"
    database.write_bytes(b"irrelevant corrupt dependency cache")
    outcome = _run_map_project(
        tmp_path,
        environment,
        tier,
        "-n",
        "0",
        "-p",
        "no:tach",
        "-q",
        "--evidence-root",
        ".evidence",
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    [receipt] = (tmp_path / ".evidence").glob("run-*/reports.jsonl")
    calls = [
        (event["nodeid"], event["outcome"])
        for line in receipt.read_text().splitlines()
        if (event := json.loads(line)).get("when") == "call"
    ]
    assert calls == [(f"tests/test_consumer.py::test_{tier}", "passed")]
    assert (tmp_path / f"executed-{tier}").exists()
    assert not (tmp_path / f"executed-{'slow' if tier == 'fast' else 'fast'}").exists()
    assert database.read_bytes() == b"irrelevant corrupt dependency cache"
    assert not (tmp_path / ".testmondata.full").exists()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("tier", ["fast", "slow"])
def test_explicit_recording_preserves_affected_consumer(tmp_path: Path, tier: str):
    environment, base = _map_recording_project(tmp_path)
    outcome = _run_map_project(
        tmp_path,
        environment,
        "--record-testmon-map",
        tier,
        "-n",
        "auto",
        *(["--maxprocesses", "1"] if tier == "fast" else ["--maxprocesses=1"]),
        "--dist",
        "loadgroup",
        "-p",
        "no:tach",
        "-q",
        "--evidence-root",
        ".evidence",
    )
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    marker = json.loads((tmp_path / ".testmondata.full").read_text())
    assert marker["tiers"] == [tier]
    (tmp_path / "increment" / "_state.py").write_text(
        "def compute(value):\n    return 2 * value + 2\n"
    )
    affected = _run_map_project(
        tmp_path,
        environment,
        "--affected-base",
        base,
        tier,
    )
    assert affected.returncode == 1, affected.stdout + affected.stderr
    receipts = list((tmp_path / ".test-evidence" / "affected").glob("run-*/run.json"))
    assert len(receipts) == 1, affected.stdout + affected.stderr
    [affected_receipt] = receipts
    metadata = json.loads(affected_receipt.read_text())
    assert metadata["selection_route"] == "testmon"
    reports = [
        json.loads(line)
        for line in affected_receipt.with_name("reports.jsonl").read_text().splitlines()
    ]
    assert [
        (event["nodeid"], event["outcome"]) for event in reports if event.get("when") == "call"
    ] == [(f"tests/test_consumer.py::test_{tier}", "failed")]


@pytest.mark.integration
@pytest.mark.parametrize(
    "arguments",
    [
        ["focused", "tests/test_consumer.py"],
        ["examples"],
        ["fast", "-k", "test_fast"],
        ["fast", "--splits", "2", "--group", "1"],
        ["fast", "--collect-only"],
    ],
)
def test_explicit_recording_rejects_partial_requests_before_execution(
    tmp_path: Path, arguments: list[str]
):
    environment, _ = _map_recording_project(tmp_path)
    outcome = _run_map_project(
        tmp_path,
        environment,
        "--record-testmon-map",
        *arguments,
    )
    assert outcome.returncode != 0
    assert not (tmp_path / "executed-fast").exists()
    assert not (tmp_path / "executed-slow").exists()
    assert not (tmp_path / ".testmondata").exists()
    assert not (tmp_path / ".testmondata.full").exists()


@pytest.mark.integration
def test_explicit_recording_refuses_verified_sanitized_nested_context(tmp_path: Path):
    from scripts._test_impact import testmon_database_lock

    environment, _ = _map_recording_project(tmp_path)
    with testmon_database_lock(tmp_path):
        environment.update(
            {
                "INCREMENT_TESTMON_LOCK_OWNER": os.environ["INCREMENT_TESTMON_LOCK_OWNER"],
                "INCREMENT_TESTMON_LOCK_ROLE": "nested",
            }
        )
        outcome = _run_map_project(
            tmp_path, environment, "--record-testmon-map", "fast", "-n", "0", "-p", "no:tach"
        )
        assert outcome.returncode == 4, outcome.stdout + outcome.stderr
        assert not (tmp_path / "executed-fast").exists()
        assert not (tmp_path / ".testmondata").exists()
        assert not (tmp_path / ".testmondata.full").exists()
