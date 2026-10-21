"""Integration contracts for test-process resources and Hypothesis profiles."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import duckdb
import ibis
import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.xdist_group("test-resources"),
]
ROOT = Path(__file__).parents[1]


def _child_env(**overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("CI", None)
    env.pop("INCREMENT_TEST_DUCKDB_MEMORY", None)
    env.update(overrides)
    return env


def _run_child(tmp_path: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    child_env = env.copy()
    child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    child_env["PYTHONPATH"] = os.pathsep.join((str(ROOT), child_env.get("PYTHONPATH", "")))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--confcutdir",
            str(tmp_path),
            "-q",
            *args,
        ],
        cwd=tmp_path,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _setting(connection, name: str):
    return connection.raw_sql(f"SELECT current_setting('{name}')").fetchone()[0]


def _native_setting(name: str):
    connection = duckdb.connect()
    try:
        row = connection.execute(f"SELECT current_setting('{name}')").fetchone()
        assert row is not None
        return row[0]
    finally:
        connection.close()


def _memory_bytes(connection) -> int:
    value = _setting(connection, "memory_limit").strip().replace(" ", "")
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    for unit, multiplier in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if value.upper().endswith(unit):
            return int(float(value[: -len(unit)]) * multiplier)
    raise AssertionError(f"unrecognized DuckDB memory limit: {value}")


def test_duckdb_defaults_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default and configured connection memory caps reach DuckDB."""
    monkeypatch.delenv("INCREMENT_TEST_DUCKDB_MEMORY", raising=False)
    con = ibis.duckdb.connect()
    try:
        assert _setting(con, "threads") == 1
        assert _memory_bytes(con) <= 1024**3
        assert "increment-duckdb-spill-" in _setting(con, "temp_directory")
    finally:
        con.disconnect()
    monkeypatch.setenv("INCREMENT_TEST_DUCKDB_MEMORY", "512MB")
    con = ibis.duckdb.connect()
    try:
        assert _memory_bytes(con) <= 512 * 1024**2
    finally:
        con.disconnect()


def test_each_connection_gets_its_own_spill_directory() -> None:
    """Connections isolate spill files under one process-owned parent."""
    first = ibis.duckdb.connect()
    second = ibis.duckdb.connect()
    try:
        first_path = Path(_setting(first, "temp_directory"))
        second_path = Path(_setting(second, "temp_directory"))
        assert first_path != second_path
        assert first_path.parent == second_path.parent
        first.disconnect()
        assert second.raw_sql("SELECT 42").fetchone() == (42,)
    finally:
        first.disconnect()
        second.disconnect()


def test_explicit_duckdb_settings_win() -> None:
    """Explicit connection settings override test defaults."""
    con = ibis.duckdb.connect(threads=2, memory_limit="2GB")
    try:
        assert _setting(con, "threads") == 2
        assert 0.9 * 2_000_000_000 <= _memory_bytes(con) <= 2_000_000_000
    finally:
        con.disconnect()


def test_none_memory_limit_uses_duckdb_default() -> None:
    """Explicit None opts out of the test memory cap."""
    con = ibis.duckdb.connect(memory_limit=None)
    try:
        assert _setting(con, "memory_limit") == _native_setting("memory_limit")
    finally:
        con.disconnect()


def test_none_temp_directory_uses_duckdb_default() -> None:
    """Explicit None opts out of the per-connection spill directory."""
    con = ibis.duckdb.connect(temp_directory=None)
    try:
        assert _setting(con, "temp_directory") == _native_setting("temp_directory")
    finally:
        con.disconnect()


def test_markdown_snippet_gets_duckdb_defaults(tmp_path: Path) -> None:
    """Markdown Python blocks use root test connection defaults."""
    document = tmp_path / "test-resource-check.md"
    document.write_text(
        "```python\n"
        "import ibis\n"
        "con = ibis.duckdb.connect()\n"
        "assert con.raw_sql(\"SELECT current_setting('threads')\").fetchone()[0] == 1\n"
        "temp = con.raw_sql(\"SELECT current_setting('temp_directory')\").fetchone()[0]\n"
        "assert 'increment-duckdb-spill-' in temp\n"
        "```\n"
    )
    result = _run_child(tmp_path, str(document), env=_child_env())
    assert result.returncode == 0, result.stdout + result.stderr


def test_hypothesis_deadline_only_in_ci(tmp_path: Path) -> None:
    """The selected Hypothesis profile follows the child process CI setting."""
    test_file = tmp_path / "test-profile-check.py"
    test_file.write_text(
        "import hypothesis\n"
        "from datetime import timedelta\n"
        "import os\n"
        "def test_deadline():\n"
        "    expected = timedelta(milliseconds=200) if os.getenv('CI') else None\n"
        "    assert hypothesis.settings().deadline == expected\n"
    )
    dev = _run_child(tmp_path, str(test_file), env=_child_env())
    assert dev.returncode == 0, dev.stdout + dev.stderr
    ci = _run_child(tmp_path, str(test_file), env=_child_env(CI="true"))
    assert ci.returncode == 0, ci.stdout + ci.stderr


def test_dev_profile_drops_deadline(tmp_path: Path) -> None:
    """Slow test execution is tolerated locally but remains bounded in CI."""
    test_file = tmp_path / "test-slow-generation-check.py"
    test_file.write_text(
        "import time\n"
        "from hypothesis import Phase, given, settings, strategies as st\n"
        "@settings(max_examples=1, phases=(Phase.generate,))\n"
        "@given(st.integers())\n"
        "def test_generated(value):\n"
        "    time.sleep(0.3)\n"
        "    assert isinstance(value, int)\n"
    )
    dev = _run_child(tmp_path, str(test_file), env=_child_env())
    assert dev.returncode == 0, dev.stdout + dev.stderr
    ci = _run_child(tmp_path, str(test_file), env=_child_env(CI="true"))
    assert ci.returncode != 0
    assert "DeadlineExceeded" in ci.stdout + ci.stderr


def test_dev_profile_suppresses_slow_generation_check(tmp_path: Path) -> None:
    """Slow strategy draws pass locally but fail with CI health checks."""
    test_file = tmp_path / "test-slow-generation-check.py"
    test_file.write_text(
        "import time\n"
        "from datetime import timedelta\n"
        "from hypothesis import Phase, given, settings, strategies as st\n"
        "@st.composite\n"
        "def slow_values(draw):\n"
        "    value = draw(st.integers())\n"
        "    time.sleep(0.13)\n"
        "    return value\n"
        # 8 * 130 ms = 1.04 s > max(1 s, 5 * 200 ms) = 1 s; 10 is the minimum observed.
        "@settings(max_examples=10, deadline=timedelta(milliseconds=200), phases=(Phase.generate,))\n"
        "@given(slow_values())\n"
        "def test_generated(value):\n"
        "    assert isinstance(value, int)\n"
    )
    dev = _run_child(tmp_path, str(test_file), env=_child_env())
    assert dev.returncode == 0, dev.stdout + dev.stderr
    ci = _run_child(tmp_path, str(test_file), env=_child_env(CI="true"))
    assert ci.returncode != 0
    assert "HealthCheck.too_slow" in ci.stdout + ci.stderr
