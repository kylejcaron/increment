"""Regression tests for the runtime and collection unit boundary."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]


def _run_child(
    tmp_path: Path,
    source: str,
    *,
    conftest_extra: str = "",
    paths: tuple[Path, ...] = (),
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(
        (PROJECT / "conftest.py").read_text() + "\n" + conftest_extra
    )
    test_file = tmp_path / "test_consumer.py"
    test_file.write_text(source)
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTEST_", "COV_CORE_", "COVERAGE_"))
    }
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT), environment.get("PYTHONPATH", "")))
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
            str(test_file),
            *(str(path) for path in paths),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.integration
class TestRuntimeBoundary:
    def test_unit_test_opening_duckdb_fails_with_guidance(self, tmp_path: Path) -> None:
        result = _run_child(tmp_path, "import ibis\ndef test_open():\n    ibis.duckdb.connect()\n")
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_open" in output
        assert "opened a DuckDB connection" in output
        assert "pytest.mark.integration" in output
        assert "in-memory data" in output

    def test_integration_test_may_open_duckdb(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import ibis, pytest\n@pytest.mark.integration\ndef test_open():\n    ibis.duckdb.connect()\n",
        )
        assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr

    def test_slow_test_may_open_duckdb(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import ibis, pytest\n@pytest.mark.slow\ndef test_open():\n    ibis.duckdb.connect()\n",
        )
        assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr

    def test_unit_test_requesting_backend_fixture_is_a_usage_error(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import pytest\n@pytest.mark.integration\ndef test_integration(con): pass\n"
            "@pytest.mark.integration\ndef test_uses_first(con): pass\n"
            "def test_unit(con): pass\n",
            conftest_extra=(
                "import ibis\nfrom tests._unit_boundary import backend_fixture\n"
                "@backend_fixture(scope='session')\ndef con(): return ibis.duckdb.connect()\n"
            ),
        )
        output = result.stdout + result.stderr
        assert result.returncode == pytest.ExitCode.USAGE_ERROR
        assert "backend fixture con" in output
        assert "test_consumer.py::test_unit" in output
        assert "pytest.mark.integration" in output
        assert "in-memory data" in output

    def test_unit_test_starting_subprocess_fails(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import subprocess, sys\ndef test_start():\n    subprocess.run([sys.executable, '-c', 'pass'])\n",
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_start" in output
        assert "started a subprocess" in output

    def test_unit_test_starting_process_fails(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import multiprocessing\ndef test_start():\n"
            "    process = multiprocessing.Process(target=print)\n"
            "    process.start()\n    process.join()\n",
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_start" in output
        assert "started a process" in output

    def test_unit_test_opening_network_socket_fails(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import socket\ndef test_connect():\n"
            "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "    sock.connect(('127.0.0.1', 9))\n",
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_connect" in output
        assert "opened a network connection" in output

    def test_unit_test_may_write_tmp_path(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path, "def test_write(tmp_path):\n    (tmp_path / 'value').write_text('ok')\n"
        )
        assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr

    def test_unit_test_using_threads_cannot_open_duckdb(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "from concurrent.futures import ThreadPoolExecutor\nimport ibis\ndef test_thread():\n    with ThreadPoolExecutor() as pool:\n        pool.submit(ibis.duckdb.connect).result()\n",
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_thread" in output
        assert "opened a DuckDB connection" in output

    def test_pure_local_override_of_backend_fixture_is_allowed(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import pytest\n@pytest.fixture\ndef con(): return {'value': 1}\n"
            "def test_local_override(con): assert con['value'] == 1\n",
            conftest_extra=(
                "import ibis\nfrom tests._unit_boundary import backend_fixture\n"
                "@backend_fixture(scope='session')\ndef con(): return ibis.duckdb.connect()\n"
            ),
        )
        assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr

    def test_markdown_snippet_may_open_duckdb(self, tmp_path: Path) -> None:
        snippet = tmp_path / "snippet.md"
        snippet.write_text("```python\nimport ibis\nibis.duckdb.connect()\n```\n")
        result = _run_child(
            tmp_path,
            "def test_placeholder(): pass\n",
            paths=(snippet,),
        )
        assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr

    def test_live_warehouse_module_may_open_duckdb(self, tmp_path: Path) -> None:
        module = tmp_path / "integration" / "warehouse_execution" / "test_live.py"
        module.parent.mkdir(parents=True)
        module.write_text(
            "import ibis, pytest\npytestmark = pytest.mark.warehouse_postgres\n"
            "def test_compare():\n    ibis.duckdb.connect()\n"
        )
        result = _run_child(tmp_path, "def test_placeholder(): pass\n", paths=(module,))
        assert result.returncode == pytest.ExitCode.OK, result.stdout + result.stderr

    def test_unit_fixture_setup_cannot_open_duckdb(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import pytest\n@pytest.fixture\n"
            "def backend():\n    import ibis\n    ibis.duckdb.connect()\n"
            "def test_setup(backend): pass\n",
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_setup" in output
        assert "opened a DuckDB connection" in output

    def test_unit_fixture_finalizer_cannot_open_duckdb(self, tmp_path: Path) -> None:
        result = _run_child(
            tmp_path,
            "import pytest\n@pytest.fixture\n"
            "def backend(request):\n"
            "    def finalize():\n        import ibis\n        ibis.duckdb.connect()\n"
            "    request.addfinalizer(finalize)\n"
            "def test_teardown(backend): pass\n",
        )
        output = result.stdout + result.stderr
        assert result.returncode != 0
        assert "test_consumer.py::test_teardown" in output
        assert "opened a DuckDB connection" in output


def _decorator_name(decorator: ast.expr) -> str | None:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _connecting_fixture_violations(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    connection_names = {
        "covariate_defs_and_con",
        "_fit_pair",
        "_session",
        "_native_fixture",
        "_analysis",
        "producer_arm",
    }
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = {_decorator_name(decorator) for decorator in node.decorator_list}
        if not decorators.intersection({"fixture", "backend_fixture"}):
            continue
        uses_connection = any(
            isinstance(call, ast.Call)
            and (
                (
                    isinstance(call.func, ast.Attribute)
                    and call.func.attr == "connect"
                    and (
                        (
                            isinstance(call.func.value, ast.Attribute)
                            and call.func.value.attr == "duckdb"
                        )
                        or (
                            isinstance(call.func.value, ast.Name) and call.func.value.id == "duckdb"
                        )
                    )
                )
                or (
                    isinstance(call.func, ast.Name)
                    and (call.func.id == "connect" or call.func.id in connection_names)
                )
            )
            for call in ast.walk(node)
        )
        if uses_connection and "backend_fixture" not in decorators:
            violations.append(
                f"{filename}:{node.lineno}: fixture {node.name} connects without @backend_fixture"
            )
    return violations


# Every test module must be parsed to detect newly connecting fixtures.
@pytest.mark.slow
def test_connecting_fixtures_are_backend_fixtures() -> None:
    violations = []
    for source_path in (PROJECT / "tests").rglob("*.py"):
        violations.extend(_connecting_fixture_violations(source_path.read_text(), str(source_path)))
    synthetic = "import pytest\n@pytest.fixture\ndef con():\n    return duckdb.connect()\n"
    violations.extend(_connecting_fixture_violations(synthetic, "synthetic/conftest.py"))
    assert violations == ["synthetic/conftest.py:3: fixture con connects without @backend_fixture"]
