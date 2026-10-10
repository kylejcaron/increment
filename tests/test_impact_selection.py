"""Exercise filesystem guards through the real impact-selection plugin."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


def _initialize_import_coverage_project(
    root: Path,
    sources: dict[str, str],
    collection_test: str | None = None,
    collection_conftest: str | None = None,
    test_nodes: list[str] | None = None,
) -> str:
    package = root / "increment"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for relative, source in sources.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    if collection_test is not None or collection_conftest is not None:
        tests = root / "tests"
        tests.mkdir()
        if collection_conftest is not None:
            (tests / "conftest.py").write_text(collection_conftest)
        if collection_test is not None:
            (tests / "test_provider.py").write_text(collection_test)
    connection = sqlite3.connect(root / ".testmondata")
    for table in (
        "metadata",
        "file_fp",
        "test_execution_file_fp",
        "suite_execution_file_fsha",
    ):
        connection.execute(f"CREATE TABLE {table} (value TEXT)")
    connection.execute("CREATE TABLE test_execution (test_name TEXT)")
    connection.executemany(
        "INSERT INTO test_execution (test_name) VALUES (?)",
        [(node,) for node in test_nodes or []],
    )
    connection.execute(
        "CREATE TABLE environment (environment_name TEXT, system_packages TEXT, python_version TEXT)"
    )
    connection.execute("INSERT INTO environment VALUES ('default', 'numpy==2', '3.13')")
    connection.commit()
    connection.close()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Impact Tests"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "impact-tests@example.invalid"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "increment"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=root, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def test_windows_process_liveness_uses_pid_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import psutil

    from scripts import _test_impact

    monkeypatch.setattr(_test_impact.os, "name", "nt")
    monkeypatch.setattr(psutil, "pid_exists", lambda pid: pid == 123)

    assert _test_impact._process_is_alive(123)
    assert not _test_impact._process_is_alive(456)


def test_testmon_nested_lock_requires_live_owner_and_skips_certification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts._test_impact import (
        record_testmon_full_run,
        testmon_database_lock,
        testmon_full_tiers,
        testmon_nested_context,
    )

    with testmon_database_lock(tmp_path) as owns_lock:
        assert owns_lock
        assert testmon_nested_context(tmp_path)
        from scripts.affected_pytest_launcher import _prepare_testmon_environment

        monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
        monkeypatch.delenv("INCREMENT_AFFECTED_WORKER_ALLOWANCE", raising=False)
        _prepare_testmon_environment(tmp_path, "fast", 2)
        assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
        assert os.environ["INCREMENT_AFFECTED_WORKER_ALLOWANCE"] == "2"
        assert os.environ["INCREMENT_AFFECTED_NESTED_TESTMON"] == "1"
        with testmon_database_lock(tmp_path) as nested_owns_lock:
            assert not nested_owns_lock
            record_testmon_full_run(tmp_path, "fast")
        assert not (tmp_path / ".testmondata.full").exists()
        assert testmon_full_tiers(tmp_path) == set()


def test_supervisor_marks_first_pytest_child_as_owner_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    from scripts._test_impact import testmon_database_lock
    from scripts.run_test_tier import run_with_budget

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    with testmon_database_lock(tmp_path) as owns_lock:
        assert owns_lock
        status = run_with_budget(
            [
                sys.executable,
                "-c",
                "import os, sys; sys.stderr.write(os.environ['INCREMENT_TESTMON_LOCK_ROLE'] + chr(10))",
            ],
            tier="role-test",
            budget_seconds=10,
        )
    assert status == 0
    assert "owner-child" in capfd.readouterr().err


def test_testmon_database_uses_import_graph_on_corruption(tmp_path: Path) -> None:
    from scripts.affected_pytest_launcher import _testmon_database_usable

    assert _testmon_database_usable(tmp_path)
    (tmp_path / ".testmondata").write_bytes(b"not a sqlite database")
    assert not _testmon_database_usable(tmp_path)


def test_testmon_coverage_marker_requires_matching_environment(tmp_path: Path) -> None:
    from scripts._test_impact import (
        _exact_environment_fingerprint,
        record_testmon_full_run,
        testmon_full_tiers,
    )

    connection = sqlite3.connect(tmp_path / ".testmondata")
    connection.execute(
        "CREATE TABLE environment (environment_name TEXT, system_packages TEXT, python_version TEXT)"
    )
    connection.execute(
        "INSERT INTO environment VALUES (?, ?, ?)",
        ("default", "numpy==2", "3.13"),
    )
    connection.commit()
    connection.close()
    record_testmon_full_run(tmp_path, "fast")
    marker = json.loads((tmp_path / ".testmondata.full").read_text())
    assert marker["python_environment"] == _exact_environment_fingerprint(tmp_path)
    assert testmon_full_tiers(tmp_path) == {"fast"}
    connection = sqlite3.connect(tmp_path / ".testmondata")
    connection.execute("UPDATE environment SET system_packages = 'numpy==3'")
    connection.commit()
    connection.close()
    assert testmon_full_tiers(tmp_path) == set()
    record_testmon_full_run(tmp_path, "slow")
    assert testmon_full_tiers(tmp_path) == {"slow"}


def test_testmon_fingerprint_is_stable_across_wal_checkpoint(tmp_path: Path) -> None:
    from scripts._test_impact import _testmon_fingerprint

    database = tmp_path / ".testmondata"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE test_execution (test_name TEXT)")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("INSERT INTO test_execution VALUES ('tests/test_one.py::test_one')")
    connection.commit()
    before_checkpoint = _testmon_fingerprint(tmp_path)
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.close()

    assert _testmon_fingerprint(tmp_path) == before_checkpoint


def test_testmon_readiness_is_certified_before_database_mutation(tmp_path: Path) -> None:
    from scripts._test_impact import record_testmon_full_run
    from scripts.affected_pytest_launcher import _testmon_map_ready

    connection = sqlite3.connect(tmp_path / ".testmondata")
    for table in (
        "metadata",
        "test_execution",
        "file_fp",
        "test_execution_file_fp",
        "suite_execution_file_fsha",
    ):
        connection.execute(f"CREATE TABLE {table} (value TEXT)")
    connection.execute(
        "CREATE TABLE environment "
        "(environment_name TEXT, system_packages TEXT, python_version TEXT)"
    )
    connection.execute("INSERT INTO environment VALUES ('default', 'numpy==2', '3.13')")
    connection.commit()
    connection.close()
    record_testmon_full_run(tmp_path, "fast")
    assert _testmon_map_ready(tmp_path, "fast", [])
    connection = sqlite3.connect(tmp_path / ".testmondata")
    connection.execute("UPDATE environment SET system_packages='numpy==3'")
    connection.commit()
    connection.close()
    assert not _testmon_map_ready(tmp_path, "fast", [])


def test_supervised_affected_launcher_engages_testmon_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import affected_pytest_launcher
    from scripts._test_impact import record_testmon_full_run, testmon_database_lock

    base = _initialize_import_coverage_project(tmp_path, {"_state.py": "VALUE = 1\n"})
    record_testmon_full_run(tmp_path, "fast")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INCREMENT_AFFECTED_CHANGED_PATHS", "[]")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    captured: dict[str, object] = {}

    def capture(arguments, plugins):
        captured["arguments"] = list(arguments)
        captured["plugins"] = list(plugins)
        captured["ready"] = os.environ.get("INCREMENT_AFFECTED_TESTMON_READY")
        return 0

    monkeypatch.setattr(
        affected_pytest_launcher._Pytest91Adapter,
        "run",
        staticmethod(capture),
    )
    with testmon_database_lock(tmp_path):
        os.environ["INCREMENT_TESTMON_LOCK_ROLE"] = "owner-child"
        monkeypatch.setattr(os, "environ", os.environ.copy())
        assert affected_pytest_launcher.main(["--tach-base", base, "--tier", "fast"]) == 0

    from typing import cast

    arguments = cast(list[str], captured["arguments"])
    plugins = cast(list[str], captured["plugins"])
    assert captured["ready"] == "1"
    assert "testmon.pytest_testmon" in plugins
    assert "--testmon" in arguments
    assert "--testmon-forceselect" in arguments


def test_affected_launcher_pins_worktree_testmon_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts.affected_pytest_launcher import _prepare_testmon_environment

    monkeypatch.setenv("TESTMON_DATAFILE", str(tmp_path / "wrong.sqlite"))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--lf")
    monkeypatch.delenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", raising=False)
    _prepare_testmon_environment(tmp_path, "fast", None)

    assert os.environ["TESTMON_DATAFILE"] == str(tmp_path / ".testmondata")
    assert os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "PYTEST_ADDOPTS" not in os.environ


def test_testmon_map_falls_back_for_import_time_module_state(tmp_path: Path) -> None:
    from scripts._test_impact import record_testmon_full_run, testmon_module_scope_changed

    module = tmp_path / "increment" / "_state.py"
    module.parent.mkdir()
    (module.parent / "__init__.py").touch()
    module.write_text("ROLE = 'cluster'\n\ndef helper():\n    return 1\n")
    connection = sqlite3.connect(tmp_path / ".testmondata")
    connection.execute(
        "CREATE TABLE environment (environment_name TEXT, system_packages TEXT, python_version TEXT)"
    )
    connection.execute("INSERT INTO environment VALUES ('default', 'numpy==2', '3.13')")
    connection.commit()
    connection.close()
    record_testmon_full_run(tmp_path, "fast")

    module.write_text("ROLE = 'other'\n\ndef helper():\n    return 1\n")
    assert testmon_module_scope_changed(tmp_path, ["increment/_state.py"], "fast")
    module.write_text("ROLE = 'cluster'\n\ndef helper():\n    return 2\n")
    assert not testmon_module_scope_changed(tmp_path, ["increment/_state.py"], "fast")


def test_import_coverage_catches_local_alias_initializer_call(tmp_path: Path) -> None:
    from scripts._test_impact import (
        record_testmon_full_run,
        testmon_full_tiers,
        testmon_module_scope_changed,
    )

    _initialize_import_coverage_project(
        tmp_path,
        {"_state.py": "def build_value():\n    return 1\n\n_make = build_value\nVALUE = _make()\n"},
    )
    record_testmon_full_run(tmp_path, "fast")
    assert testmon_full_tiers(tmp_path) == {"fast"}
    marker = json.loads((tmp_path / ".testmondata.full").read_text())
    lines = marker["import_time_lines_by_tier"]["fast"]["increment/_state.py"]
    assert 2 in lines

    (tmp_path / "increment" / "_state.py").write_text(
        "def build_value():\n    return 2\n\n_make = build_value\nVALUE = _make()\n"
    )
    assert testmon_module_scope_changed(tmp_path, ["increment/_state.py"], "fast")


def test_import_coverage_catches_imported_provider_function(tmp_path: Path) -> None:
    from scripts._test_impact import record_testmon_full_run, testmon_module_scope_changed

    _initialize_import_coverage_project(
        tmp_path,
        {
            "providers.py": "def make_value():\n    return 1\n",
            "_consumer.py": ("from .providers import make_value as provider\nVALUE = provider()\n"),
        },
    )
    record_testmon_full_run(tmp_path, "fast")
    marker = json.loads((tmp_path / ".testmondata.full").read_text())
    lines = marker["import_time_lines_by_tier"]["fast"]["increment/providers.py"]
    assert 2 in lines

    (tmp_path / "increment" / "providers.py").write_text(
        "# shifted above the certified source\n\ndef make_value():\n    return 2\n"
    )
    assert testmon_module_scope_changed(tmp_path, ["increment/providers.py"], "fast")


def test_testmon_falls_back_for_changes_since_certified_source_snapshot(tmp_path: Path) -> None:
    from scripts._test_impact import (
        record_testmon_full_run,
        testmon_module_scope_changed,
        write_testmon_coverage,
    )

    _initialize_import_coverage_project(
        tmp_path,
        {
            "providers.py": "def make_value():\n    return 1\n",
            "_consumer.py": "from .providers import make_value as provider\nVALUE = 1\n",
        },
    )
    record_testmon_full_run(tmp_path, "fast")

    late_consumer = tmp_path / "increment" / "_late_consumer.py"
    late_consumer.write_text("from .providers import make_value as provider\nVALUE = provider()\n")
    assert testmon_module_scope_changed(tmp_path, ["increment/_late_consumer.py"], "fast")
    write_testmon_coverage(tmp_path, {"fast"})

    provider = tmp_path / "increment" / "providers.py"
    provider.write_text("def make_value():\n    return 2\n")

    assert testmon_module_scope_changed(tmp_path, ["increment/providers.py"], "fast")


def test_full_run_does_not_certify_sources_changed_before_marker_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import _test_impact, run_test_tier

    _initialize_import_coverage_project(
        tmp_path,
        {"_state.py": "VALUE = 1\n"},
        collection_test="from increment._state import VALUE\n\ndef test_state():\n    assert VALUE == 1\n",
        test_nodes=["tests/test_provider.py::test_provider"],
    )
    source = tmp_path / "increment" / "_state.py"
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTMON_DATAFILE", raising=False)

    def run_with_report(*args: object, **kwargs: object) -> int:
        extra_env = kwargs["extra_env"]
        assert isinstance(extra_env, dict)
        report_path = extra_env.get("INCREMENT_TESTMON_CERTIFICATION_REPORT")
        assert isinstance(report_path, str)
        Path(report_path).write_text(
            json.dumps({"safe": True, "collected": ["test"], "reported": ["test"]})
        )
        return 0

    record_full_run = _test_impact.record_testmon_full_run

    def write_after_source_change(
        root: Path,
        tier: str,
        *,
        test_dist: str | None = None,
        expected_source_snapshot: tuple[dict[str, str], dict[str, str]] | None = None,
    ) -> None:
        source.write_text("VALUE = 2\n")
        if expected_source_snapshot is None:
            record_full_run(root, tier, test_dist=test_dist)
        else:
            record_full_run(
                root,
                tier,
                test_dist=test_dist,
                expected_source_snapshot=expected_source_snapshot,
            )

    monkeypatch.setattr(run_test_tier, "run_with_budget", run_with_report)
    monkeypatch.setattr(run_test_tier, "record_testmon_full_run", write_after_source_change)
    monkeypatch.setattr(
        _test_impact,
        "_import_time_line_coverage",
        lambda root, tier, test_dist: ({}, ["tests/test_provider.py::test_provider"]),
    )

    assert run_test_tier.main(["fast"]) == 0
    assert not (tmp_path / ".testmondata.full").exists()


@pytest.mark.parametrize(
    "import_statement",
    ["import pkg.leaf", "from pkg.leaf import value"],
)
def test_changed_parent_initializer_affects_submodule_importers(
    tmp_path: Path, import_statement: str
) -> None:
    from scripts._test_impact import affected_test_paths

    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("INIT_VALUE = 1\n")
    (package / "leaf.py").write_text("value = 1\n")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_leaf.py").write_text(f"{import_statement}\n\ndef test_leaf():\n    pass\n")

    assert affected_test_paths(tmp_path, ["pkg/__init__.py"]) == ["tests/test_leaf.py"]


def test_full_run_does_not_certify_sources_changed_during_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_test_tier

    source = tmp_path / "increment" / "_state.py"
    source.parent.mkdir()
    (source.parent / "__init__.py").write_text("")
    source.write_text("VALUE = 1\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTMON_DATAFILE", raising=False)
    certified: list[tuple[Path, str]] = []

    def run_with_source_change(*args: object, **kwargs: object) -> int:
        extra_env = kwargs["extra_env"]
        assert isinstance(extra_env, dict)
        report_path = extra_env.get("INCREMENT_TESTMON_CERTIFICATION_REPORT")
        assert isinstance(report_path, str)
        report = Path(report_path)
        report.write_text(json.dumps({"safe": True, "collected": ["test"], "reported": ["test"]}))
        source.write_text("VALUE = 2\n")
        return 0

    monkeypatch.setattr(run_test_tier, "run_with_budget", run_with_source_change)
    monkeypatch.setattr(
        run_test_tier,
        "record_testmon_full_run",
        lambda root, tier, *, test_dist=None: certified.append((root, tier)),
    )

    assert run_test_tier.main(["fast"]) == 0
    assert certified == []


def test_import_coverage_ignores_uninvoked_function_body_change(tmp_path: Path) -> None:
    from scripts._test_impact import record_testmon_full_run, testmon_module_scope_changed

    _initialize_import_coverage_project(
        tmp_path,
        {"_state.py": "def unused(\n    value: int = 1,\n) -> int:\n    return value\n"},
    )
    record_testmon_full_run(tmp_path, "fast")
    (tmp_path / "increment" / "_state.py").write_text(
        "def unused(\n    value: int = 1,\n) -> int:\n    return value + 1\n"
    )

    assert not testmon_module_scope_changed(tmp_path, ["increment/_state.py"], "fast")


@pytest.mark.parametrize(
    ("test_dist", "expected_node"),
    [
        (None, "tests/test_provider.py::test_value[1]"),
        ("loadgroup", "tests/test_provider.py::test_value[1]@coverage"),
    ],
)
def test_conftest_gated_provider_lines_are_certified(
    tmp_path: Path, test_dist: str | None, expected_node: str
) -> None:
    from scripts._test_impact import record_testmon_full_run, testmon_module_scope_changed

    _initialize_import_coverage_project(
        tmp_path,
        {
            "providers.py": (
                "import os\n"
                "if os.environ.get('COLLECTION_PROVIDER') == '1':\n"
                "    def values():\n"
                "        return [1]\n"
                "else:\n"
                "    def values():\n"
                "        return []\n"
            )
        },
        (
            "import pytest\n"
            "from increment.providers import values\n"
            "@pytest.mark.xdist_group('coverage')\n"
            "@pytest.mark.parametrize('value', values())\n"
            "def test_value(value):\n    assert value == 1\n"
        ),
        "import os\nos.environ['COLLECTION_PROVIDER'] = '1'\n",
        [expected_node],
    )
    record_testmon_full_run(tmp_path, "fast", test_dist=test_dist)
    marker = json.loads((tmp_path / ".testmondata.full").read_text())
    lines = marker["import_time_lines_by_tier"]["fast"]["increment/providers.py"]
    assert 4 in lines

    (tmp_path / "increment" / "providers.py").write_text(
        "import os\n"
        "if os.environ.get('COLLECTION_PROVIDER') == '1':\n"
        "    def values():\n"
        "        return [2]\n"
        "else:\n"
        "    def values():\n"
        "        return []\n"
    )
    assert testmon_module_scope_changed(tmp_path, ["increment/providers.py"], "fast")


def test_malformed_full_marker_can_be_replaced(tmp_path: Path) -> None:
    from scripts._test_impact import record_testmon_full_run, testmon_full_tiers

    _initialize_import_coverage_project(tmp_path, {"_state.py": "VALUE = 1\n"})
    marker_path = tmp_path / ".testmondata.full"
    marker_path.write_text("[]")

    record_testmon_full_run(tmp_path, "fast")

    assert testmon_full_tiers(tmp_path) == {"fast"}


def test_missing_testmon_node_prevents_marker_refresh(tmp_path: Path) -> None:
    from scripts._test_impact import (
        record_testmon_full_run,
        testmon_full_tiers,
        write_testmon_coverage,
    )

    _initialize_import_coverage_project(tmp_path, {"_state.py": "VALUE = 1\n"})
    record_testmon_full_run(tmp_path, "fast")
    marker_path = tmp_path / ".testmondata.full"
    marker = json.loads(marker_path.read_text())
    marker["test_nodes_by_tier"]["fast"] = ["tests/test_missing.py::test_missing"]
    marker_path.write_text(json.dumps(marker))
    original_fingerprint = marker["fingerprint"]

    write_testmon_coverage(tmp_path, {"fast"})

    assert json.loads(marker_path.read_text())["fingerprint"] == original_fingerprint
    assert testmon_full_tiers(tmp_path) == set()


def test_full_run_certifies_only_the_tier_it_executes(tmp_path: Path) -> None:
    from scripts._test_impact import (
        module_scope_fingerprint,
        record_testmon_full_run,
        testmon_full_tiers,
        testmon_module_scope_baselines,
        testmon_module_scope_changed,
        write_testmon_coverage,
    )

    package = tmp_path / "increment"
    package.mkdir()
    (package / "__init__.py").touch()
    module = package / "_state.py"
    module.write_text("ROLE = 'cluster'\n")
    connection = sqlite3.connect(tmp_path / ".testmondata")
    connection.execute(
        "CREATE TABLE environment (environment_name TEXT, system_packages TEXT, python_version TEXT)"
    )
    connection.execute("INSERT INTO environment VALUES ('default', 'numpy==2', '3.13')")
    connection.commit()
    connection.close()
    record_testmon_full_run(tmp_path, "fast")
    record_testmon_full_run(tmp_path, "slow")
    assert testmon_full_tiers(tmp_path) == {"slow"}

    module.write_text("ROLE = 'other'\n")
    write_testmon_coverage(tmp_path, {"fast", "slow"})
    assert testmon_module_scope_changed(tmp_path, ["increment/_state.py"], "slow")
    assert testmon_module_scope_changed(tmp_path, ["increment/_state.py"], "fast")

    record_testmon_full_run(tmp_path, "fast")
    baselines = testmon_module_scope_baselines(tmp_path)
    current = module_scope_fingerprint(module)
    assert baselines["fast"]["increment/_state.py"] == current
    assert "slow" not in baselines
    record_testmon_full_run(tmp_path, "all")
    assert testmon_full_tiers(tmp_path) == {"fast", "slow"}
    baselines = testmon_module_scope_baselines(tmp_path)
    current = module_scope_fingerprint(module)
    assert baselines["fast"]["increment/_state.py"] == current
    assert baselines["slow"]["increment/_state.py"] == current


def test_module_scope_fingerprint_tracks_initializer_local_helpers(tmp_path: Path) -> None:
    from scripts._test_impact import module_scope_fingerprint

    module = tmp_path / "module.py"
    module.write_text("def helper():\n    return 1\n\nVALUE = helper()\n")
    original = module_scope_fingerprint(module)

    module.write_text("def helper():\n    return 2\n\nVALUE = helper()\n")

    assert module_scope_fingerprint(module) != original


def test_partial_runs_cannot_certify_testmon_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.run_test_tier import can_certify_testmon_full_run

    assert can_certify_testmon_full_run("fast", ["-q", "-n", "4", "--dist", "loadgroup"])
    assert not can_certify_testmon_full_run("fast", ["tests/test_tables.py"])
    assert not can_certify_testmon_full_run("fast", ["--splits", "2", "--group", "1"])
    assert not can_certify_testmon_full_run("fast", ["--collect-only"])
    assert not can_certify_testmon_full_run("fast", ["--help"])
    assert not can_certify_testmon_full_run("fast", ["-k", "one_test"])
    assert can_certify_testmon_full_run("fast", ["-q", "-p", "no:tach"])
    assert not can_certify_testmon_full_run("fast", ["-p", "custom-plugin"])
    assert not can_certify_testmon_full_run("examples", [])
    monkeypatch.setenv("PYTEST_ADDOPTS", "--lf")
    assert not can_certify_testmon_full_run("fast", ["-q"])


def test_nested_pytest_command_disables_runner_selection_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.run_test_tier import TIERS, build_pytest_command

    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_runner.py::test_nested (call)")
    monkeypatch.setenv("INCREMENT_AFFECTED_USE_TESTMON", "1")

    command = build_pytest_command(TIERS["all"], [])

    assert "scripts.run_test_tier_plugin" not in command
    assert "no:pytest-testmon" in command


def test_nested_runner_without_owner_skips_testmon_database(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import run_test_tier

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_runner.py::test_nested (call)")
    monkeypatch.delenv("INCREMENT_TESTMON_LOCK_OWNER", raising=False)
    monkeypatch.delenv("INCREMENT_TESTMON_LOCK_ROLE", raising=False)
    commands: list[list[str]] = []

    def run(command, **kwargs):
        commands.append(list(command))
        return 0

    monkeypatch.setattr(run_test_tier, "run_with_budget", run)
    monkeypatch.setattr(
        run_test_tier,
        "testmon_database_lock",
        lambda root: pytest.fail("unowned nested runner attempted to lock Testmon"),
    )

    assert run_test_tier.main(["fast", "-q"]) == 0
    assert commands
    assert "scripts.run_test_tier_plugin" not in commands[0]
    assert "no:pytest-testmon" in commands[0]


def test_full_run_certification_requires_resolved_full_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from scripts.run_test_tier_plugin import _full_certification_is_safe

    config_path = tmp_path / "pyproject.toml"
    config_path.touch()
    option = SimpleNamespace(
        markexpr="slow",
        keyword="",
        lf=False,
        last_failed=False,
        failedfirst=False,
        stepwise=False,
        collectonly=False,
        deselect=[],
        ignore=[],
        ignore_glob=[],
        splits=None,
        group=None,
        file_or_dir=[],
    )
    config = cast(pytest.Config, SimpleNamespace(inipath=config_path, option=option))
    monkeypatch.setenv("INCREMENT_TESTMON_EXPECTED_CONFIG", str(config_path))
    monkeypatch.setenv("INCREMENT_TESTMON_EXPECTED_MARK", "slow")
    assert _full_certification_is_safe(config)
    option.group = 1
    assert not _full_certification_is_safe(config)
    option.group = None
    option.keyword = "one_test"
    assert not _full_certification_is_safe(config)
    option.keyword = ""
    option.file_or_dir = ["tests/test_selection.py"]
    assert not _full_certification_is_safe(config)


def test_full_run_certification_requires_every_collected_nodeid_to_report(
    tmp_path: Path,
) -> None:
    from scripts.run_test_tier import (
        _full_run_certification_complete,
        _full_run_certification_dist,
    )

    report = tmp_path / "full-run.json"
    report.write_text(
        json.dumps(
            {"safe": True, "collected": ["a", "b"], "reported": ["a", "b"], "dist": "loadgroup"}
        )
    )
    assert _full_run_certification_complete(report)
    assert _full_run_certification_dist(report) == "loadgroup"
    report.write_text(
        json.dumps({"safe": True, "collected": ["a"], "reported": ["a"], "dist": None})
    )
    assert _full_run_certification_dist(report) is None
    report.write_text(json.dumps({"safe": True, "collected": ["a", "b"], "reported": ["a"]}))
    assert not _full_run_certification_complete(report)
    report.write_text(json.dumps({"safe": False, "collected": ["a"], "reported": ["a"]}))
    assert not _full_run_certification_complete(report)


def test_testmon_primary_protects_importers_of_changed_test_modules(tmp_path: Path) -> None:
    from scripts.run_test_tier_plugin import _testmon_protected_paths

    tests = tmp_path / "tests"
    estimation = tests / "estimation"
    estimation.mkdir(parents=True)
    changed = "tests/estimation/test_adjust.py"
    direct_importer = "tests/estimation/test_decision_stats.py"
    transitive_importer = "tests/test_rollout.py"
    (tests / "test_changed.py").write_text("def test_changed(): pass\n")
    (estimation / "test_adjust.py").write_text("_DESIGN = object()\n")
    (estimation / "test_decision_stats.py").write_text(
        "from tests.estimation.test_adjust import _DESIGN\n"
    )
    (tests / "test_rollout.py").write_text(
        "from tests.estimation.test_decision_stats import _DESIGN\n"
    )
    dynamic = "tests/test_dynamic_consumer.py"
    subprocess = "tests/test_subprocess_consumer.py"
    protected = _testmon_protected_paths(tmp_path, [changed], [dynamic], [subprocess])

    assert protected == {
        (tmp_path / changed).resolve(),
        (tmp_path / direct_importer).resolve(),
        (tmp_path / transitive_importer).resolve(),
        (tmp_path / dynamic).resolve(),
        (tmp_path / subprocess).resolve(),
    }


def test_testmon_primary_does_not_run_guard_files_for_non_code_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from scripts.run_test_tier_plugin import _activate_testmon

    selector = SimpleNamespace(deselected_files=[], deselected_tests=[])
    data = SimpleNamespace(all_files=[], new_db=False, system_packages_change=None)
    config = cast(
        pytest.Config,
        SimpleNamespace(
            rootpath=tmp_path,
            testmon_data=data,
            pluginmanager=SimpleNamespace(get_plugin=lambda name: selector),
        ),
    )
    monkeypatch.setenv("INCREMENT_AFFECTED_USE_TESTMON", "1")
    monkeypatch.setenv("INCREMENT_AFFECTED_TESTMON_READY", "1")
    monkeypatch.setenv("INCREMENT_AFFECTED_CHANGED_PATHS", "[]")

    protected: set[Path] = set()
    assert _activate_testmon(config, protected)
    assert protected == set()

    changed = "increment/analysis.py"
    data.all_files = [str(tmp_path / changed)]
    monkeypatch.setenv("INCREMENT_AFFECTED_CHANGED_PATHS", json.dumps([changed]))
    assert _activate_testmon(config, protected)
    assert (tmp_path / "tests" / "test_impact_selection.py").resolve() in protected


def test_testmon_protects_tests_using_cached_fixtures_for_code_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from typing import cast

    from scripts.run_test_tier_plugin import _activate_testmon

    tests = tmp_path / "tests"
    tests.mkdir()
    consumer = tests / "test_cached.py"
    consumer.write_text(
        "from tests._shared_cache import fixture_cache_key, get_or_build\ndef test_cached(): pass\n"
    )
    selector = SimpleNamespace(deselected_files=[], deselected_tests=[])
    data = SimpleNamespace(
        all_files=[str(tmp_path / "increment" / "simulate" / "runner.py")],
        new_db=False,
        system_packages_change=None,
    )
    config = cast(
        pytest.Config,
        SimpleNamespace(
            rootpath=tmp_path,
            testmon_data=data,
            pluginmanager=SimpleNamespace(get_plugin=lambda name: selector),
        ),
    )
    monkeypatch.setenv("INCREMENT_AFFECTED_USE_TESTMON", "1")
    monkeypatch.setenv("INCREMENT_AFFECTED_TESTMON_READY", "1")
    monkeypatch.setenv("INCREMENT_AFFECTED_CHANGED_PATHS", '["increment/simulate/runner.py"]')

    protected: set[Path] = set()
    assert _activate_testmon(config, protected)
    assert consumer.resolve() in protected


def test_cache_fixture_protection_includes_transitive_module_consumers(tmp_path: Path) -> None:
    from scripts.run_test_tier_plugin import _cache_backed_test_paths

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "__init__.py").write_text("")
    (tests / "_shared_cache.py").write_text("def get_or_build(): pass\n")
    (tests / "test_dashboard.py").write_text(
        "from tests._shared_cache import get_or_build\ndef storefront(): return get_or_build()\n"
    )
    consumer = tests / "test_dashboard_result_evidence.py"
    consumer.write_text(
        "from tests.test_dashboard import storefront\ndef test_evidence(): storefront()\n"
    )

    assert consumer.resolve() in _cache_backed_test_paths(tmp_path)


def test_module_scope_fingerprint_tracks_import_time_state_only(tmp_path: Path) -> None:
    from scripts._test_impact import module_scope_fingerprint

    module = tmp_path / "increment" / "_moment_plan.py"
    module.parent.mkdir()
    module.write_text(
        "X_SLOT_ROLES = ('cluster',)\n"
        "class Plan:\n"
        "    ROLE = 'cluster'\n"
        "    def method(self):\n"
        "        return 1\n"
    )
    original = module_scope_fingerprint(module)
    module.write_text(
        "X_SLOT_ROLES = ('cluster',)\n"
        "class Plan:\n"
        "    ROLE = 'cluster'\n"
        "    def method(self):\n"
        "        return 2\n"
    )
    assert module_scope_fingerprint(module) == original
    module.write_text(
        "X_SLOT_ROLES = ('cluster', 'outcome')\n"
        "class Plan:\n"
        "    ROLE = 'cluster'\n"
        "    def method(self):\n"
        "        return 2\n"
    )
    assert module_scope_fingerprint(module) != original


def test_subprocess_test_files_are_discovered_conservatively(tmp_path: Path) -> None:
    from scripts._test_impact import subprocess_consumer_test_paths

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_child.py").write_text(
        "import subprocess as sp\n"
        "import sys as interpreter\n"
        "def test_child():\n"
        "    sp.run([interpreter.executable, '-c', 'pass'])\n"
    )
    (tests / "test_local.py").write_text("def test_local():\n    assert True\n")

    assert subprocess_consumer_test_paths(tmp_path) == ["tests/test_child.py"]


def test_testmon_sidecars_do_not_appear_as_affected_changes(tmp_path: Path) -> None:
    from scripts.run_test_tier import _changed_paths

    base = _make_affected_fixture(tmp_path)
    for suffix in ("", "-wal", "-shm"):
        (tmp_path / f".testmondata{suffix}").write_bytes(b"sqlite")
    (tmp_path / ".testmondata.full").write_text("{}")
    (tmp_path / ".testmondata.full.tmp").write_text("{}")
    changes, paths = _changed_paths(base, tmp_path)
    assert changes == []
    assert paths == []


def test_import_graph_follows_transitive_imports_and_lazy_exports(tmp_path: Path) -> None:
    from scripts._test_impact import affected_test_paths

    package = tmp_path / "increment"
    tests = tmp_path / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text(
        "_LAZY_IMPORTS = {'Analysis': ('increment.analysis', 'Analysis')}\n"
    )
    (package / "analysis.py").write_text("class Analysis: pass\n")
    (tests / "_helper.py").write_text("from increment.analysis import Analysis\n")
    (tests / "test_direct.py").write_text("from increment import Analysis\n")
    (tests / "test_transitive.py").write_text("from tests._helper import Analysis\n")
    (package / "unrelated.py").write_text("VALUE = 2\n")
    (tests / "test_unrelated.py").write_text("import increment.unrelated\n")

    assert affected_test_paths(tmp_path, ["increment/analysis.py"]) == [
        "tests/test_direct.py",
        "tests/test_transitive.py",
    ]


def test_import_graph_keeps_consumers_of_changed_test_modules(tmp_path: Path) -> None:
    from scripts._test_impact import affected_test_paths

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_compatibility_js.py").write_text("def shared_helper():\n    return 1\n")
    (tests / "test_dashboard_shell_js.py").write_text(
        "from tests.test_compatibility_js import shared_helper\n"
        "\ndef test_shell():\n    assert shared_helper() == 1\n"
    )
    assert affected_test_paths(tmp_path, ["tests/test_compatibility_js.py"]) == [
        "tests/test_compatibility_js.py",
        "tests/test_dashboard_shell_js.py",
    ]


def test_import_graph_keeps_changed_test_and_handles_relative_imports(tmp_path: Path) -> None:
    from scripts._test_impact import affected_test_paths

    package = tmp_path / "increment"
    query = package / "query"
    tests = tmp_path / "tests"
    query.mkdir(parents=True)
    tests.mkdir()
    (package / "__init__.py").write_text("")
    (query / "__init__.py").write_text("")
    (query / "native_source.py").write_text("VALUE = 1\n")
    (query / "adapter.py").write_text("from .native_source import VALUE\n")
    (tests / "test_adapter.py").write_text("from increment.query.adapter import VALUE\n")
    (tests / "test_changed.py").write_text("def test_changed(): pass\n")

    assert affected_test_paths(
        tmp_path,
        ["increment/query/native_source.py", "tests/test_changed.py"],
    ) == ["tests/test_adapter.py", "tests/test_changed.py"]


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


@pytest.mark.slow
def test_affected_runner_falls_back_to_import_graph_for_corrupt_testmon(
    tmp_path: Path,
) -> None:
    project = Path(__file__).resolve().parents[1]
    base = _make_affected_fixture(tmp_path)
    (tmp_path / "pkg" / "leaf.py").write_text("VALUE = 1  # changed\n")
    (tmp_path / ".testmondata").write_bytes(b"corrupt testmon state")
    from scripts.affected_pytest_launcher import _testmon_database_usable

    assert not _testmon_database_usable(tmp_path)
    evidence_root = tmp_path / "evidence"
    result = _run_affected(
        project,
        tmp_path,
        base,
        "fast",
        "--evidence-root",
        str(evidence_root),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
    [run_path] = evidence_root.glob("run-*/run.json")
    evidence = json.loads(run_path.read_text())
    assert evidence["selection_route"] == "import_graph"
    assert set(evidence["selected_nodeids"]) == {
        "tests/test_direct.py::test_direct",
        "tests/test_transitive.py::test_transitive",
    }


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
    (root / ".gitignore").write_text("__pycache__/\n.test-evidence/\n.testmondata*\n")
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
    environment.pop("PYTEST_CURRENT_TEST", None)
    environment.pop("INCREMENT_TESTMON_LOCK_OWNER", None)
    environment.pop("INCREMENT_TESTMON_LOCK_ROLE", None)
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
    assert "passed in" in result.stdout


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
        f"uv run --no-sync --project {project} --extra demo --extra tables --extra dashboard "
        "python -m scripts.run_test_tier"
    )
    arguments = pytest_args or f"--evidence-root={evidence}"
    return subprocess.run(
        [
            "make",
            "-f",
            str(project / "Makefile"),
            f"AFFECTED_TEST_RUNNER={runner}",
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
