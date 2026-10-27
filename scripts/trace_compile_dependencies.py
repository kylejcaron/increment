"""Record and check the package modules imported by a warehouse compile capture."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

DEPENDENCIES_PATH = Path("integration/warehouse_execution/_compile_dependencies.txt")


def _dependency_record_path(root: Path, cache_key: str) -> Path:
    return root / ".pytest_cache" / "warehouse_compile_registry" / f"{cache_key}.imports.json"


_BUILD_CAPTURE = r"""
import importlib.abc
import pickle
import sys

modules = set()

class IncrementImportRecorder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "increment" or fullname.startswith("increment."):
            modules.add(fullname)
        return None
from tests._duckdb_resources import bound_duckdb_resources
bound_duckdb_resources()
sys.meta_path.insert(0, IncrementImportRecorder())
from integration.warehouse_execution._compile import _build_cached_captured_compile_data
records = _build_cached_captured_compile_data(sys.argv[1])
pickle.dump((records, sorted(modules)), sys.stdout.buffer)
"""


def _build_and_record(root: Path, dialect: str) -> tuple[object, list[str]]:
    result = subprocess.run(
        [sys.executable, "-c", _BUILD_CAPTURE, dialect],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    result.check_returncode()
    return pickle.loads(result.stdout)


def _write_dependency_record(root: Path, cache_key: str, modules: list[str]) -> None:
    path = _dependency_record_path(root, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(modules, stream)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _recorded_modules(root: Path, cache_key: str) -> list[str] | None:
    path = _dependency_record_path(root, cache_key)
    try:
        modules = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(modules, list) or not all(isinstance(module, str) for module in modules):
        return None
    return modules


def _differences(recorded: list[str], current: list[str]) -> list[str]:
    return [
        *(f"missing: {module}" for module in sorted(set(current) - set(recorded))),
        *(f"stale: {module}" for module in sorted(set(recorded) - set(current))),
    ]


_TRACE_CAPTURE = r"""
import importlib.abc
import json
import sys

modules = set()

class IncrementImportRecorder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "increment" or fullname.startswith("increment."):
            modules.add(fullname)
        return None

sys.meta_path.insert(0, IncrementImportRecorder())
from integration.warehouse_execution._compile import _build_cached_captured_compile_data
_build_cached_captured_compile_data(sys.argv[1])
print(json.dumps(sorted(modules)))
"""


def trace(root: Path, dialect: str = "snowflake") -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _TRACE_CAPTURE, dialect],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _read_modules(path: Path) -> list[str]:
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def check(root: Path, dialect: str = "snowflake") -> list[str]:
    """Return no lines when the recorded module list matches a cached or fresh capture."""
    from integration.warehouse_execution._compile import _captured_compile_cache_key

    root = root.resolve()
    recorded_path = root / DEPENDENCIES_PATH
    recorded = _read_modules(recorded_path)
    cache_key = _captured_compile_cache_key(dialect, root)
    current = _recorded_modules(root, cache_key)
    if current is None:
        current = trace(root, dialect)
    return _differences(recorded, current)


def write(root: Path, dialect: str = "snowflake") -> None:
    modules = trace(root, dialect)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    destination = root / DEPENDENCIES_PATH
    destination.write_text(
        f"# Traced on commit: {commit}\n" + "".join(f"{module}\n" for module in modules)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dialect", choices=("snowflake", "bigquery"), default="snowflake")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        differences = check(arguments.root, arguments.dialect)
        if differences:
            print("\n".join(differences), file=sys.stderr)
            return 1
        module_count = len(_read_modules(arguments.root / DEPENDENCIES_PATH))
        print(f"compile dependency trace is current ({module_count} modules)")
        return 0
    write(arguments.root, arguments.dialect)
    module_count = len(_read_modules(arguments.root / DEPENDENCIES_PATH))
    print(f"recorded {module_count} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
