"""Exercise filesystem guards through the real impact-selection plugin."""

from __future__ import annotations

import json
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


@pytest.mark.slow
def test_affected_runner_selects_transitive_consumers_and_excludes_unrelated(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    result = _run_affected(
        project, tmp_path, base, "fast", "--evidence-root", str(tmp_path / "evidence")
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_direct.py" in result.stdout
    assert "test_transitive.py" in result.stdout
    assert "[Tach] Skipped 1 test" in result.stdout
    assert "test_unrelated.py ." not in result.stdout
    assert "2 passed" in result.stdout


def _prepare_dynamic_consumer(
    root: Path, *, slow_dynamic: bool = False, xdist_grouped: bool = False
) -> str:
    _make_affected_fixture(root)
    decorators = []
    if slow_dynamic:
        decorators.extend(("import pytest", "@pytest.mark.slow"))
    if xdist_grouped:
        if "import pytest" not in decorators:
            decorators.append("import pytest")
        decorators.append("@pytest.mark.xdist_group(name='consumer')")
    marker = "\n".join(decorators) + "\n" if decorators else ""
    (root / "tests" / "test_dynamic.py").write_text(
        "import importlib\n"
        f"{marker}def test_dynamic_consumer():\n"
        "    assert importlib.import_module('pkg.direct').VALUE == 1\n\n"
        "def test_other():\n"
        "    assert True\n"
    )
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    subprocess.run(["git", "add", "tests/test_dynamic.py"], cwd=root, env=environment, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Add dynamic consumer",
        ],
        cwd=root,
        env=environment,
        check=True,
    )
    base = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, env=environment, text=True
    ).strip()
    (root / "pkg" / "leaf.py").write_text("VALUE = 1  # changed provider\n")
    return base


@pytest.mark.slow
def test_affected_dynamic_consumer_reports_partial_k_selection(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _prepare_dynamic_consumer(tmp_path)
    result = _run_affected(project, tmp_path, base, "fast", "-k", "test_other")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "tests/test_dynamic.py" in result.stdout
    assert "test_dynamic.py::test_dynamic_consumer" in result.stdout
    assert "not fully run in the fast tier/pytest selection" in result.stdout


@pytest.mark.slow
def test_affected_dynamic_consumer_reports_fast_tier_partial_file(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _prepare_dynamic_consumer(tmp_path, slow_dynamic=True)
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert "test_dynamic.py::test_dynamic_consumer" in result.stdout
    assert "not fully run in the fast tier/pytest selection" in result.stdout


@pytest.mark.slow
def test_affected_dynamic_consumer_aggregates_xdist_selection_once(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _prepare_dynamic_consumer(tmp_path)
    result = _run_affected(project, tmp_path, base, "fast", "-n", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "retained 1 files" in result.stdout
    assert "fully selected 1" in result.stdout
    assert result.stdout.count("Affected dynamic-import consumers:") == 1


@pytest.mark.slow
def test_affected_dynamic_consumer_keeps_stable_identity_with_xdist_group(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _prepare_dynamic_consumer(tmp_path, xdist_grouped=True)
    result = _run_affected(project, tmp_path, base, "fast", "-n", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fully selected 1" in result.stdout
    assert "0 not fully run in the fast tier/pytest selection" in result.stdout
    assert "omitted: none" in result.stdout


@pytest.mark.slow
@pytest.mark.parametrize("option", ["--lf", "--collect-only", "-n auto", "-f", "-k test_unrelated"])
def test_affected_runner_rejects_inherited_pytest_addopts(tmp_path: Path, option: str) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    result = _run_affected(
        project, tmp_path, base, "fast", env_overrides={"PYTEST_ADDOPTS": option}
    )
    assert result.returncode == 2
    assert "PYTEST_ADDOPTS" in result.stderr


@pytest.mark.slow
def test_affected_runner_rejects_inherited_pytest_plugins(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    result = _run_affected(
        project, tmp_path, base, "fast", env_overrides={"PYTEST_PLUGINS": "tests._evidence"}
    )
    assert result.returncode == 2
    assert "PYTEST_PLUGINS" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize(
    "pytest_args",
    [
        ("--help",),
        ("--version",),
        ("--version", "--version"),
        ("-V",),
        ("-VV",),
        ("--lf",),
        ("--collect-only",),
        ("-n", "auto"),
        ("-n", "logical"),
        ("-n", "0"),
        ("-n", "9"),
        ("--maxprocesses", "2"),
        ("-d",),
        ("--tx", "popen"),
        ("-f",),
        ("--looponfail",),
        ("-c", "alternate.ini"),
        ("-o", "addopts=--lf"),
        ("@injected.args",),
    ],
)
def test_affected_runner_rejects_untyped_pytest_arguments(
    tmp_path: Path, pytest_args: tuple[str, ...]
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    if pytest_args == ("@injected.args",):
        (tmp_path / "injected.args").write_text("--version --version\n")
    result = _run_affected(project, tmp_path, base, "fast", *pytest_args)
    assert result.returncode == 2
    assert "affected runner accepts only typed options" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize("workers", ["1", "8"])
def test_affected_runner_accepts_bounded_numeric_workers(tmp_path: Path, workers: str) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    result = _run_affected(
        project,
        tmp_path,
        base,
        "fast",
        "-n",
        workers,
        env_overrides={"PYTEST_XDIST_AUTO_NUM_WORKERS": "64"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


@pytest.mark.slow
@pytest.mark.parametrize(
    ("config_name", "contents"),
    [
        ("pytest.ini", "[pytest]\ntestpaths = tests\naddopts = --collect-only\n"),
        (
            "pyproject.toml",
            "[tool.pytest.ini_options]\ntestpaths = ['tests']\naddopts = ['--collect-only']\n",
        ),
    ],
)
def test_affected_runner_uses_fixed_config_instead_of_project_addopts(
    tmp_path: Path, config_name: str, contents: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(
        tmp_path, pytest_config_name=config_name, pytest_config_contents=contents
    )
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "Affected-test selection: 2 selected" in result.stdout


def _make_affected_fixture(
    root: Path,
    *,
    pytest_config_name: str = "pytest.ini",
    pytest_config_contents: str | None = None,
) -> str:
    package = root / "pkg"
    tests = root / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text("")
    (package / "leaf.py").write_text("VALUE = 1\n")
    (package / "direct.py").write_text("from pkg.leaf import VALUE\n")
    (package / "transitive.py").write_text("from pkg.direct import VALUE\n")
    (package / "unrelated.py").write_text("VALUE = 10\n")
    (tests / "test_direct.py").write_text(
        "from pkg.direct import VALUE\n\ndef test_direct():\n    assert VALUE == 1\n"
    )
    (tests / "test_transitive.py").write_text(
        "from pkg.transitive import VALUE\n\ndef test_transitive():\n    assert VALUE == 1\n"
    )
    (tests / "test_unrelated.py").write_text(
        "from pkg.unrelated import VALUE\n\ndef test_unrelated():\n    assert VALUE == 10\n"
    )
    (root / "tach.toml").write_text(
        'source_roots = ["."]\nroot_module = "ignore"\n'
        '[[modules]]\npath = "pkg.leaf"\ndepends_on = []\n'
        '[[modules]]\npath = "pkg.direct"\ndepends_on = ["pkg.leaf"]\n'
        '[[modules]]\npath = "pkg.transitive"\ndepends_on = ["pkg.direct"]\n'
        '[[modules]]\npath = "pkg.unrelated"\ndepends_on = []\n'
    )
    (root / ".gitignore").write_text("__pycache__/\n.test-evidence/\n")
    (root / pytest_config_name).write_text(
        pytest_config_contents
        if pytest_config_contents is not None
        else "[pytest]\ntestpaths = tests\nmarkers = slow: isolated slow-tier test\n"
    )
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    for arguments in (
        ["init", "-q", "-b", "main"],
        ["add", "."],
        ["-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "Base"],
    ):
        subprocess.run(["git", *arguments], cwd=root, env=environment, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, env=environment, text=True
    ).strip()


def _run_affected(
    project: Path,
    root: Path,
    base: str,
    tier: str,
    *pytest_args: str,
    env_overrides: dict[str, str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment: dict[str, str] = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(project), os.environ.get("PYTHONPATH", ""))),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
    }
    for key, value in (env_overrides or {}).items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.run_test_tier",
            "--affected-base",
            base,
            tier,
            *pytest_args,
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )


@pytest.mark.slow
def test_affected_runner_propagates_a_broken_consumer_failure(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 2\n")
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 1
    assert "test_direct" in result.stdout
    assert "test_transitive" in result.stdout
    assert "2 failed" in result.stdout


@pytest.mark.slow
def test_affected_runner_refuses_shared_fixture_and_deleted_inputs(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    conftest = tmp_path / "tests" / "conftest.py"
    conftest.write_text("import pytest\n")
    refusal = _run_affected(project, tmp_path, base, "fast")
    assert refusal.returncode == 2, refusal.stdout + refusal.stderr
    assert "conftest.py" in refusal.stderr
    assert "make check" in refusal.stderr
    assert "make test-slow" in refusal.stderr

    conftest.unlink()
    (tmp_path / "pkg" / "direct.py").unlink()
    deleted = _run_affected(project, tmp_path, base, "fast")
    assert deleted.returncode == 2
    assert "deletion impact cannot be established" in deleted.stderr
    assert "make check" in deleted.stderr


@pytest.mark.slow
def test_affected_runner_labels_empty_selection_without_running_full_suite(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "orphan.py").write_text("VALUE = 9\n")
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 5, result.stdout + result.stderr
    assert "Affected-test selection: no affected tests" in result.stdout


@pytest.mark.slow
def test_affected_runner_preserves_collection_failure_without_selection(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_broken_collection.py").write_text(
        "import missing_module_for_collection_error\n\ndef test_broken():\n    assert True\n"
    )
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 2
    assert "no affected tests were selected" not in result.stderr
    assert "ModuleNotFoundError" in result.stdout + result.stderr


@pytest.mark.slow
@pytest.mark.parametrize("change_state", ["unstaged", "staged", "committed"])
def test_affected_runner_tracks_tracked_change_states(tmp_path: Path, change_state: str) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    leaf = tmp_path / "pkg" / "leaf.py"
    leaf.write_text("VALUE = 1  # tracked update\n")
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    if change_state in {"staged", "committed"}:
        subprocess.run(["git", "add", str(leaf)], cwd=tmp_path, env=environment, check=True)
    if change_state == "committed":
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "-qm",
                "Change",
            ],
            cwd=tmp_path,
            env=environment,
            check=True,
        )
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    assert "Affected-test selection: 2 selected" in result.stdout


@pytest.mark.slow
def test_affected_runner_includes_new_untracked_test_files(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_new.py").write_text(
        "from pkg.leaf import VALUE\n\ndef test_new():\n    assert VALUE == 1\n"
    )
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout
    assert "test_new" in result.stdout


@pytest.mark.slow
def test_affected_runner_refuses_renames_dynamic_imports_and_lazy_exports(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    subprocess.run(["git", "mv", "pkg/unrelated.py", "pkg/renamed.py"], cwd=tmp_path, check=True)
    renamed = _run_affected(project, tmp_path, base, "fast")
    assert renamed.returncode == 2
    assert "rename/deletion impact cannot be established" in renamed.stderr

    subprocess.run(["git", "reset", "--hard", base], cwd=tmp_path, check=True)
    (tmp_path / "pkg" / "leaf.py").write_text(
        "import importlib\nVALUE = importlib.import_module('pkg.direct').VALUE\n"
    )
    dynamic = _run_affected(project, tmp_path, base, "fast")
    assert dynamic.returncode == 2
    assert "dynamic-import consumer tests" in dynamic.stderr

    subprocess.run(["git", "reset", "--hard", base], cwd=tmp_path, check=True)
    exports = tmp_path / "increment"
    exports.mkdir()
    (exports / "__init__.py").write_text("__all__ = []\n")
    lazy = _run_affected(project, tmp_path, base, "fast")
    assert lazy.returncode == 2
    assert "public API tests" in lazy.stderr


@pytest.mark.slow
@pytest.mark.parametrize("timeout_value", ["invalid", "0.000001"])
def test_affected_runner_ignores_inherited_timeout_options(
    tmp_path: Path, timeout_value: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_timeout_env.py").write_text(
        "import time\n\ndef test_timeout_environment_is_ignored():\n    time.sleep(0.02)\n"
    )
    result = _run_affected(
        project,
        tmp_path,
        base,
        "fast",
        env_overrides={"PYTEST_TIMEOUT": timeout_value},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.slow
def test_affected_runner_preserves_nested_timeout_harness_environment(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_nested_timeout.py").write_text(
        "from tests.test_test_runtime_harness import "
        "test_root_runtime_budget_is_local_only as run_nested_timeout\n\n"
        "def test_nested_timeout():\n"
        "    run_nested_timeout(None)\n"
    )
    result = _run_affected(project, tmp_path, base, "fast", "-n", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


@pytest.mark.slow
def test_affected_runner_keeps_repository_warning_policy(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_unexpected_warning.py").write_text(
        "import warnings\n\ndef test_unexpected_warning_fails():\n"
        "    warnings.warn('unexpected warning')\n"
    )
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "UserWarning: unexpected warning" in result.stdout + result.stderr


def test_affected_and_normal_tiers_share_marker_expressions() -> None:
    import shlex
    import tomllib

    from scripts._test_tier_policy import TIER_MARKERS
    from scripts.affected_pytest_launcher import _tier_expression
    from scripts.run_test_tier import TIERS, build_pytest_command

    project = Path(__file__).resolve().parents[1]
    settings = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["pytest"][
        "ini_options"
    ]
    defaults = shlex.split(settings["addopts"])
    assert defaults[defaults.index("-m") + 1] == TIER_MARKERS["fast"]
    for tier, marker in TIER_MARKERS.items():
        command = build_pytest_command(TIERS[tier], [])
        pytest_index = command.index("pytest")
        assert command[command.index("-m", pytest_index + 1) + 1] == marker
        assert _tier_expression(tier, None) == marker


@pytest.mark.slow
def test_affected_config_uses_repository_warning_policy(tmp_path: Path) -> None:
    import tomllib

    from scripts.affected_pytest_launcher import _fixed_pytest_config

    project = Path(__file__).resolve().parents[1]
    pytest_config = tomllib.loads((project / "pyproject.toml").read_text())["tool"]["pytest"][
        "ini_options"
    ]
    content = _fixed_pytest_config(project)
    for marker in pytest_config["markers"]:
        assert f"    {marker}" in content
    for warning in pytest_config["filterwarnings"]:
        assert f"    {warning}" in content


def test_private_pytest_parser_adapter_is_exactly_pinned() -> None:
    import tomllib

    project = Path(__file__).resolve().parents[1]
    dependency_groups = tomllib.loads((project / "pyproject.toml").read_text())["dependency-groups"]
    assert "pytest==9.1.1" in dependency_groups["dev"]


@pytest.mark.slow
def test_affected_runner_preserves_fast_and_slow_marker_separation(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_slow_case.py").write_text(
        "import pytest\n@pytest.mark.slow\ndef test_slow_case():\n    assert True\n"
    )
    fast = _run_affected(project, tmp_path, base, "fast")
    assert fast.returncode == 5, fast.stdout + fast.stderr
    assert "no affected tests" in fast.stderr
    slow = _run_affected(project, tmp_path, base, "slow")
    assert slow.returncode == 0, slow.stdout + slow.stderr
    assert "test_slow_case" in slow.stdout
    assert "1 passed" in slow.stdout


@pytest.mark.slow
def test_affected_runner_refuses_changes_to_shared_test_fixture_helpers(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    helper = tmp_path / "tests" / "fixture_helper.py"
    helper.write_text("EXPECTED = 1\n")
    direct = tmp_path / "tests" / "test_direct.py"
    direct.write_text(
        "from pkg.direct import VALUE\n"
        "from tests.fixture_helper import EXPECTED\n"
        "\ndef test_direct():\n    assert VALUE == EXPECTED\n"
    )
    environment = {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
    }
    subprocess.run(["git", "add", "."], cwd=tmp_path, env=environment, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Fixture",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    helper.write_text("EXPECTED = 2\n")
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 2
    assert "fixture_helper.py" in result.stderr
    assert "fixture-dependent test tier" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize(
    ("path", "contents", "route"),
    [
        ("pyproject.toml", "[tool.pytest.ini_options]\n", "run make check"),
        ("uv.lock", "version = 1\n", "run make check"),
        (
            "scripts/pytest_plugin.py",
            "def pytest_collection_modifyitems(): pass\n",
            "run make check",
        ),
        ("data/fixture.csv", "value\n1\n", "data/fixture validation"),
    ],
)
def test_affected_runner_refuses_configuration_dependency_plugin_and_data_hazards(
    tmp_path: Path, path: str, contents: str, route: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    hazard = tmp_path / path
    hazard.parent.mkdir(parents=True, exist_ok=True)
    hazard.write_text(contents)
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 2
    assert path in result.stderr
    assert route in result.stderr


@pytest.mark.slow
def test_make_affected_runs_selected_tests_and_propagates_failure(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    leaf = tmp_path / "pkg" / "leaf.py"
    leaf.write_text("VALUE = 1  # source change\n")
    evidence = tmp_path.parent / f"{tmp_path.name}-make-evidence"
    passed = _run_make_affected(project, tmp_path, base, evidence)
    assert passed.returncode == 0, passed.stdout + passed.stderr
    assert "2 passed" in passed.stdout
    assert "Affected-test selection: 2 selected, 1 deselected" in passed.stdout
    run_files = list(evidence.glob("run-*/run.json"))
    assert len(run_files) == 1
    manifest = json.loads(run_files[0].read_text())
    assert manifest["affected_base"] == base
    assert manifest["selected_tests"] == 2
    assert manifest["deselected_tests"] == 1
    assert manifest["outcome"] == "passed"
    assert manifest["worker_allowance"] == "serial"

    conftest = tmp_path / "tests" / "conftest.py"
    conftest.write_text("import pytest\n")
    refusal = _run_make_affected(project, tmp_path, base, evidence)
    assert refusal.returncode != 0
    assert "make test-slow" in refusal.stderr
    assert "Affected-test selection:" not in refusal.stdout
    assert len(list(evidence.glob("run-*/run.json"))) == 1
    conftest.unlink()

    selector_refusal = _run_make_affected(project, tmp_path, base, evidence, "--lf")
    assert selector_refusal.returncode != 0
    assert "affected runner accepts only typed options" in selector_refusal.stderr
    assert "Affected-test selection:" not in selector_refusal.stdout
    assert len(list(evidence.glob("run-*/run.json"))) == 1
    leaf.write_text("VALUE = 2\n")
    failed = _run_make_affected(project, tmp_path, base, evidence)
    assert failed.returncode != 0
    assert "2 failed" in failed.stdout


def _run_make_affected(
    project: Path,
    root: Path,
    base: str,
    evidence: Path,
    pytest_args: str | None = None,
) -> subprocess.CompletedProcess[str]:
    runner = (
        f"uv run --project {project} --extra demo --extra tables --extra dashboard "
        "python -m scripts.run_test_tier"
    )
    arguments = pytest_args or f"--evidence-root={evidence}"
    return subprocess.run(
        [
            "make",
            "-f",
            str(project / "Makefile"),
            f"TEST_RUNNER={runner}",
            "test-affected",
            f"BASE={base}",
            f"PYTEST_ARGS={arguments}",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.slow
def test_affected_runner_propagates_collection_selector_failures(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # change\n")
    result = _run_affected(project, tmp_path, base, "fast", "tests/missing.py")
    assert result.returncode == 2
    assert "affected runner accepts only typed options" in result.stderr


@pytest.mark.slow
def test_changed_provider_retains_unchanged_dynamic_import_consumer(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    consumer = tmp_path / "tests" / "test_dynamic_consumer.py"
    consumer.write_text(
        "import importlib as loader\n"
        "def test_dynamic_consumer():\n"
        "    assert loader.import_module('pkg.leaf').VALUE == 1\n"
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Dynamic consumer",
        ],
        cwd=tmp_path,
        check=True,
    )
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 2\n")
    result = _run_affected(project, tmp_path, base, "fast")
    assert result.returncode == 1
    assert "test_dynamic_consumer" in result.stdout
    assert "3 failed, 1 deselected" in result.stdout
    assert "dynamic-import consumer impact is not proven" not in result.stderr


@pytest.mark.parametrize(
    ("imports", "call"),
    [
        ("import importlib as importer", "importer.import_module('pkg.a')"),
        (
            "from importlib import import_module as load",
            "load('pkg.a')",
        ),
        ("import importlib.util as util", "util.spec_from_file_location('x', 'x.py')"),
        (
            "from importlib.util import spec_from_file_location as spec",
            "spec('x', 'x.py')",
        ),
        ("import runpy as runner", "runner.run_module('pkg.a')"),
        ("from runpy import run_path as execute", "execute('x.py')"),
        ("import pkgutil as packages", "packages.walk_packages()"),
        ("from pkgutil import iter_modules as modules", "modules()"),
        ("from builtins import __import__ as dynamic_import", "dynamic_import('pkg.a')"),
        ("", "__import__('pkg.a')"),
    ],
)
def test_dynamic_import_scan_resolves_common_import_aliases(
    tmp_path: Path, imports: str, call: str
) -> None:
    from scripts.run_test_tier import _dynamic_import_tests, _refusal_route

    tests = tmp_path / "tests"
    tests.mkdir()
    consumer = tests / "test_dynamic.py"
    consumer.write_text(f"{imports}\ndef test_dynamic():\n    {call}\n")
    assert _dynamic_import_tests(tmp_path) == ["tests/test_dynamic.py"]
    assert _refusal_route("pkg/provider.py", consumer.read_bytes()) == (
        "make check and the relevant dynamic-import consumer tests"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "selector",
    [
        "--lf",
        "--last-failed",
        "--ff",
        "--failed-first",
        "--nf",
        "--new-first",
        "--sw",
        "--stepwise",
        "--stepwise-skip",
        "--collect-only",
        "--sw-skip",
        "--co",
        "--cache-clear",
        "--cache-show",
        "--stepwise-reset",
        "--setup-only",
        "--setup-plan",
        "--fixtures",
        "--fixtures-per-test",
        "--markers",
        "--version",
        "--help",
        "-h",
        "-V",
    ],
)
def test_affected_runner_refuses_pytest_cache_selectors(tmp_path: Path, selector: str) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 2\n")
    result = _run_affected(project, tmp_path, base, "fast", selector)
    assert result.returncode == 2
    assert "affected runner accepts only typed options" in result.stderr


@pytest.mark.slow
@pytest.mark.parametrize(
    "selector",
    [
        ("-d",),
        ("--dist", "load"),
        ("--dist", "loadgroup"),
        ("--tx", "9*popen"),
        ("--px", "id=remote"),
        ("--rsyncdir", "tests"),
        ("--rsyncignore", "*.py"),
        ("-n", "2", "-d"),
        ("-n", "2", "--tx", "9*popen"),
    ],
)
def test_affected_runner_rejects_alternate_xdist_controls(
    tmp_path: Path, selector: tuple[str, ...]
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    result = _run_affected(project, tmp_path, base, "fast", *selector)
    assert result.returncode == 2
    assert "affected runner accepts only typed options" in result.stderr


@pytest.mark.slow
def test_affected_runner_rejects_effective_options_from_argfiles_and_config(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    args_file = tmp_path.parent / f"{tmp_path.name}-pytest.args"
    args_file.write_text("--setup-only\n")
    response = _run_affected(project, tmp_path, base, "fast", f"@{args_file}")
    assert response.returncode == 2
    assert "affected runner accepts only typed options" in response.stderr

    from_config = _run_affected(project, tmp_path, base, "fast", "-o", "addopts=--lf")
    assert from_config.returncode == 2
    assert "affected runner accepts only typed options" in from_config.stderr

    from_xdist_config = _run_affected(project, tmp_path, base, "fast", "-o", "addopts=--tx=9*popen")
    assert from_xdist_config.returncode == 2
    assert "affected runner accepts only typed options" in from_xdist_config.stderr


@pytest.mark.slow
def test_affected_runner_bounds_xdist_and_requires_loadgroup(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    unbounded = _run_affected(
        project,
        tmp_path,
        base,
        "fast",
        "-n",
        "auto",
        env_overrides={"PYTEST_XDIST_AUTO_NUM_WORKERS": None},
    )
    assert "affected runner accepts only typed options" in unbounded.stderr
    assert unbounded.returncode == 2
    workers = _run_affected(project, tmp_path, base, "fast", "-n", "2")
    assert workers.returncode == 0, workers.stdout + workers.stderr
    assert "worker_allowance=2" in workers.stdout
    assert "dist=loadgroup" in workers.stdout


def test_environment_identity_includes_dependency_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from scripts import run_test_tier

    monkeypatch.setattr(
        run_test_tier.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "sample"}, version="1.0")],
    )
    first = run_test_tier._environment_identity(Path.cwd())
    monkeypatch.setattr(
        run_test_tier.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "sample"}, version="2.0")],
    )
    second = run_test_tier._environment_identity(Path.cwd())
    assert first != second


@pytest.mark.slow
def test_changed_source_and_test_keep_transitive_consumer_selection(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 2\n")
    (tmp_path / "tests" / "test_direct.py").write_text(
        "from pkg.direct import VALUE\n\ndef test_direct():\n    assert VALUE == 2\n"
    )
    result = _run_affected(project, tmp_path, base, "fast", "-vv")
    assert result.returncode == 1
    assert "1 failed, 1 passed, 1 deselected" in result.stdout
    assert "test_transitive.py::test_transitive" in result.stdout


@pytest.mark.slow
@pytest.mark.parametrize(
    ("selector", "summary"),
    [
        (("-k", "test_transitive"), "1 passed, 2 deselected"),
        (("-m", "not slow"), "1 passed, 2 deselected"),
    ],
)
def test_changed_test_files_respect_user_k_and_marker_selectors(
    tmp_path: Path, selector: tuple[str, str], summary: str
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "tests" / "test_direct.py").write_text(
        "import pytest\nfrom pkg.direct import VALUE\n\n"
        "@pytest.mark.slow\ndef test_direct():\n    assert VALUE == 1\n"
    )
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # source change\n")
    result = _run_affected(project, tmp_path, base, "fast", *selector, "-vv")
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary in result.stdout
    assert "test_direct.py::test_direct PASSED" not in result.stdout


def test_dynamic_import_scan_includes_real_refusal_consumer() -> None:
    from scripts.run_test_tier import _dynamic_import_tests

    project = Path(__file__).resolve().parents[1]
    assert "tests/test_refusal_uniqueness.py" in _dynamic_import_tests(project)
