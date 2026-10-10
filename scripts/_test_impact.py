"""Conservative static test impact discovery from Python import edges."""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast


def _process_is_alive(pid: int) -> bool:
    if os.name == "nt":
        import psutil

        return psutil.pid_exists(pid)
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    return True


def _testmon_owner(root: Path) -> dict[str, object] | None:
    inherited = os.environ.get("INCREMENT_TESTMON_LOCK_OWNER")
    if not inherited:
        return None
    try:
        owner = json.loads(inherited)
        recorded = json.loads(
            (root.resolve() / ".testmondata.lock.owner").read_text(encoding="utf-8")
        )
        if not isinstance(owner, dict) or not isinstance(recorded, dict):
            return None
        pid = owner.get("pid")
        token = owner.get("token")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(token, str)
            or not token
            or owner != recorded
            or owner.get("root") != str(root.resolve())
        ):
            return None
        if not _process_is_alive(pid):
            return None
        return owner
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    return None


def testmon_nested_context(root: Path) -> bool:
    """True only for nested work explicitly marked by the owning runner."""
    return (
        _testmon_owner(root) is not None
        and os.environ.get("INCREMENT_TESTMON_LOCK_ROLE") == "nested"
    )


def mark_testmon_nested_context(root: Path) -> None:
    """Mark a verified pytest child as nested before it clears pytest variables."""
    if _testmon_owner(root) is not None and os.environ.get("PYTEST_CURRENT_TEST"):
        os.environ["INCREMENT_TESTMON_LOCK_ROLE"] = "nested"


def _write_testmon_lock_owner(root: Path, owner_path: Path, owner: Mapping[str, object]) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=root,
        prefix=".testmondata.lock.owner.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(owner, temporary, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, owner_path)
    return temporary_path


@contextmanager
def testmon_database_lock(root: Path) -> Iterator[bool]:
    """Serialize Testmon map access; only a verified owner token permits nesting."""
    root = root.resolve()
    if _testmon_owner(root) is not None:
        previous_role = os.environ.get("INCREMENT_TESTMON_LOCK_ROLE")
        if os.environ.get("PYTEST_CURRENT_TEST"):
            os.environ["INCREMENT_TESTMON_LOCK_ROLE"] = "nested"
        try:
            yield False
        finally:
            if previous_role is None:
                os.environ.pop("INCREMENT_TESTMON_LOCK_ROLE", None)
            else:
                os.environ["INCREMENT_TESTMON_LOCK_ROLE"] = previous_role
        return
    owner_path = root / ".testmondata.lock.owner"
    previous_owner = os.environ.get("INCREMENT_TESTMON_LOCK_OWNER")
    previous_role = os.environ.get("INCREMENT_TESTMON_LOCK_ROLE")
    owner = {"pid": os.getpid(), "root": str(root), "token": uuid.uuid4().hex}
    with (root / ".testmondata.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        temporary_path: Path | None = None
        try:
            temporary_path = _write_testmon_lock_owner(root, owner_path, owner)
            os.environ["INCREMENT_TESTMON_LOCK_OWNER"] = json.dumps(owner, sort_keys=True)
            os.environ["INCREMENT_TESTMON_LOCK_ROLE"] = (
                "nested" if os.environ.get("PYTEST_CURRENT_TEST") else "owner"
            )
            yield True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            try:
                if json.loads(owner_path.read_text(encoding="utf-8")) == owner:
                    owner_path.unlink()
            except (OSError, ValueError):
                pass
            if previous_owner is None:
                os.environ.pop("INCREMENT_TESTMON_LOCK_OWNER", None)
            else:
                os.environ["INCREMENT_TESTMON_LOCK_OWNER"] = previous_owner
            if previous_role is None:
                os.environ.pop("INCREMENT_TESTMON_LOCK_ROLE", None)
            else:
                os.environ["INCREMENT_TESTMON_LOCK_ROLE"] = previous_role
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def module_scope_fingerprint(path: Path) -> str | None:
    """Hash import-time state and helpers used to construct it."""
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    local_callables = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    class _ImportTimeCallFinder(ast.NodeVisitor):
        found = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function_definition(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function_definition(node)

        def _visit_function_definition(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if default is not None:
                    self.visit(default)
            for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if argument.annotation is not None:
                    self.visit(argument.annotation)
            if node.args.vararg and node.args.vararg.annotation:
                self.visit(node.args.vararg.annotation)
            if node.args.kwarg and node.args.kwarg.annotation:
                self.visit(node.args.kwarg.annotation)
            if node.returns is not None:
                self.visit(node.returns)

        def visit_Call(self, node: ast.Call) -> None:
            function = node.func
            if isinstance(function, ast.Name) and function.id in local_callables:
                self.found = True
            elif (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in local_callables
            ):
                self.found = True
            self.generic_visit(node)

    calls = _ImportTimeCallFinder()
    for statement in tree.body:
        calls.visit(statement)
    if calls.found:
        return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()

    class _FunctionBodyStripper(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.body = []
            return self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
            node.body = []
            return self.generic_visit(node)

    scoped = _FunctionBodyStripper().visit(tree)
    return hashlib.sha256(ast.dump(scoped, include_attributes=False).encode()).hexdigest()


def _module_scope_fingerprints(root: Path) -> dict[str, str]:
    fingerprints = {}
    source_roots = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name not in {"tests", "scripts", ".git", ".venv", ".worktrees"}
        and (path / "__init__.py").is_file()
    ]
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            fingerprint = module_scope_fingerprint(path)
            if fingerprint is None:
                raise OSError(f"cannot certify import-time module state: {path}")
            fingerprints[path.relative_to(root).as_posix()] = fingerprint
    return fingerprints


def _source_snapshots(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    texts: dict[str, str] = {}
    hashes: dict[str, str] = {}
    source_roots = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name not in {"tests", "scripts", ".git", ".venv", ".worktrees"}
        and (path / "__init__.py").is_file()
    ]
    for source_root in source_roots:
        for path in sorted(source_root.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            content = path.read_bytes()
            text = content.decode("utf-8")
            texts[relative] = text
            hashes[relative] = hashlib.sha256(content).hexdigest()
    return texts, hashes


def _valid_source_baselines(
    marker: Mapping[str, object],
) -> dict[str, tuple[dict[str, str], dict[str, str]]]:
    texts_by_tier = marker.get("source_texts_by_tier")
    hashes_by_tier = marker.get("source_hashes_by_tier")
    if not isinstance(texts_by_tier, dict) or not isinstance(hashes_by_tier, dict):
        return {}
    result: dict[str, tuple[dict[str, str], dict[str, str]]] = {}
    for tier in {"fast", "slow"}:
        texts_value = texts_by_tier.get(tier)
        hashes_value = hashes_by_tier.get(tier)
        if not isinstance(texts_value, dict) or not isinstance(hashes_value, dict):
            continue
        texts = cast(dict[str, object], texts_value)
        hashes = cast(dict[str, object], hashes_value)
        if texts.keys() != hashes.keys():
            continue
        parsed_texts: dict[str, str] = {}
        parsed_hashes: dict[str, str] = {}
        valid = True
        for path, text in texts.items():
            digest = hashes.get(path)
            if (
                not isinstance(path, str)
                or not isinstance(text, str)
                or not isinstance(digest, str)
                or hashlib.sha256(text.encode("utf-8")).hexdigest() != digest
            ):
                valid = False
                break
            parsed_texts[path] = text
            parsed_hashes[path] = digest
        if valid:
            result[tier] = (parsed_texts, parsed_hashes)
    return result


def _import_time_line_coverage(
    root: Path, tier: str, dist: str | None = None
) -> tuple[dict[str, list[int]], list[str]] | None:
    source_roots = [
        path.name
        for path in root.iterdir()
        if path.is_dir()
        and path.name not in {"tests", "scripts", ".git", ".venv", ".worktrees"}
        and (path / "__init__.py").is_file()
    ]
    if not source_roots:
        return {}, []
    import_script = """
import importlib
import json
import pkgutil
from pathlib import Path
import sys

from coverage import Coverage

root = Path.cwd().resolve()
packages = json.loads(sys.argv[1])
coverage = Coverage(source=[str(root / name) for name in packages], data_file=None, config_file=False)
coverage.start()
for name in packages:
    package = importlib.import_module(name)
    for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(module.name)
coverage.stop()
data = coverage.get_data()
result = {}
for filename in data.measured_files():
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        continue
    if relative.endswith(".py"):
        result[relative] = sorted(line for line in data.lines(filename) or () if line > 0)
print(json.dumps(result, sort_keys=True))
"""
    collect_script = """
import contextlib
import io
import json
from pathlib import Path
import sys

from coverage import Coverage
from scripts.run_test_tier import TIERS, build_pytest_command

root = Path.cwd().resolve()
tier = sys.argv[1]
dist = sys.argv[2] if len(sys.argv) > 2 else ""
source_roots = [
    path for path in root.iterdir()
    if path.is_dir() and path.name not in {"tests", "scripts", ".git", ".venv", ".worktrees"}
    and (path / "__init__.py").is_file()
]
coverage = Coverage(
    source=[str(path) for path in source_roots],
    data_file=None,
    config_file=False,
)
command = build_pytest_command(
    TIERS[tier],
    ["--collect-only", "-q", str(root / "tests"), "-p", "no:pytest-testmon"],
)
tokens = command[3:]
arguments = []
index = 0
while index < len(tokens):
    argument = tokens[index]
    if argument == "--testmon-noselect":
        index += 1
        continue
    if argument == "-c" and index + 1 < len(tokens):
        index += 2
        continue
    if argument == "-p" and index + 1 < len(tokens):
        plugin = tokens[index + 1]
        index += 2
        if plugin not in {"tests._evidence", "scripts.run_test_tier_plugin"}:
            arguments.extend(("-p", plugin))
        continue
    arguments.append(argument)
    index += 1
arguments.extend(("--rootdir", str(root), "-o", f"pythonpath={root}"))
sys.path.insert(0, str(root))
import pytest

class NodeCollector:
    nodeids = []

    def pytest_configure(self, config):
        root_text = str(root)
        if root_text in sys.path:
            sys.path.remove(root_text)
        sys.path.insert(0, root_text)

    def pytest_collection_finish(self, session):
        for item in session.items:
            nodeid = item.nodeid
            groups = set()
            for mark in item.iter_markers("xdist_group"):
                name = mark.args[0] if len(mark.args) > 0 else mark.kwargs.get("name", "default")
                groups.add(str(name))
            if dist == "loadgroup" and groups:
                nodeid += "@" + "_".join(sorted(groups))
            self.nodeids.append(nodeid)

collector = NodeCollector()
coverage.start()
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    status = pytest.main(arguments, plugins=[collector])
if status != pytest.ExitCode.OK:
    raise RuntimeError(f"pytest collection failed: {status}")
coverage.stop()
data = coverage.get_data()
result = {}
for filename in data.measured_files():
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        continue
    if relative.endswith(".py"):
        result[relative] = sorted(line for line in data.lines(filename) or () if line > 0)
print(json.dumps({"lines": result, "nodes": collector.nodeids}, sort_keys=True))
"""
    clean_environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("INCREMENT_TESTMON_", "INCREMENT_AFFECTED_", "TESTMON_", "PYTEST_"))
    }
    python_path = [str(root), str(Path(__file__).resolve().parents[1])]
    if existing_pythonpath := clean_environment.get("PYTHONPATH"):
        python_path.append(existing_pythonpath)
    clean_environment["PYTHONPATH"] = os.pathsep.join(python_path)

    def run_coverage(script: str, *arguments: str) -> object | None:
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=240,
                env=clean_environment,
            )
            return json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    def parse_lines(payload: object) -> dict[str, list[int]] | None:
        if not isinstance(payload, dict) or not all(
            isinstance(path, str)
            and isinstance(lines, list)
            and all(isinstance(line, int) and line > 0 for line in lines)
            for path, lines in payload.items()
        ):
            return None
        return cast(dict[str, list[int]], payload)

    imported = parse_lines(run_coverage(import_script, json.dumps(source_roots)))
    if imported is None:
        return None
    collected_lines: dict[str, list[int]] = {}
    test_nodes: list[str] = []
    if (root / "tests").is_dir():
        collected = run_coverage(collect_script, tier, dist or "")
        if not isinstance(collected, dict):
            return None
        candidate_lines = parse_lines(collected.get("lines"))
        raw_nodes = collected.get("nodes")
        if (
            candidate_lines is None
            or not isinstance(raw_nodes, list)
            or not all(isinstance(node, str) for node in raw_nodes)
        ):
            return None
        collected_lines = candidate_lines
        test_nodes = cast(list[str], raw_nodes)
    combined = {path: set(lines) for path, lines in imported.items()}
    for path, lines in collected_lines.items():
        combined.setdefault(path, set()).update(lines)
    return {path: sorted(lines) for path, lines in combined.items()}, test_nodes


def testmon_module_scope_baselines(root: Path) -> dict[str, dict[str, str]]:
    if testmon_nested_context(root):
        return {}
    try:
        marker = json.loads((root / ".testmondata.full").read_text(encoding="utf-8"))
        scopes = marker["module_scopes_by_tier"]
        if (
            marker.get("schema") != 7
            or marker.get("environment") != testmon_environment_fingerprint(root)
            or marker.get("python_environment") != _exact_environment_fingerprint(root)
            or not isinstance(scopes, dict)
        ):
            return {}
        return {
            tier: dict(baseline)
            for tier, baseline in scopes.items()
            if tier in {"fast", "slow"}
            and isinstance(baseline, dict)
            and all(
                isinstance(path, str) and isinstance(fingerprint, str)
                for path, fingerprint in baseline.items()
            )
        }
    except (KeyError, OSError, TypeError, ValueError):
        return {}


def testmon_import_time_baselines(root: Path) -> dict[str, dict[str, list[int]]]:
    if testmon_nested_context(root):
        return {}
    try:
        marker = json.loads((root / ".testmondata.full").read_text(encoding="utf-8"))
        scopes = marker["import_time_lines_by_tier"]
        if (
            marker.get("schema") != 7
            or marker.get("environment") != testmon_environment_fingerprint(root)
            or marker.get("python_environment") != _exact_environment_fingerprint(root)
            or not isinstance(scopes, dict)
        ):
            return {}
        result: dict[str, dict[str, list[int]]] = {}
        for tier, baseline in scopes.items():
            if tier not in {"fast", "slow"} or not isinstance(baseline, dict):
                continue
            parsed: dict[str, list[int]] = {}
            for path, lines in baseline.items():
                if (
                    not isinstance(path, str)
                    or not isinstance(lines, list)
                    or not all(
                        isinstance(line, int) and not isinstance(line, bool) and line > 0
                        for line in lines
                    )
                ):
                    return {}
                parsed[path] = [int(line) for line in lines]
            result[tier] = parsed
        return result
    except (KeyError, OSError, TypeError, ValueError):
        return {}


def _import_sensitive_lines(source: str, executed_lines: set[int]) -> set[int] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    sensitive = set(executed_lines)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            end = node.end_lineno or node.lineno
            body_start = node.body[0].lineno
            if any(body_start <= line <= end for line in executed_lines):
                sensitive.update(range(body_start, end + 1))
    return sensitive


def testmon_module_scope_changed(root: Path, changed_paths: list[str], tier: str) -> bool:
    module_baselines = testmon_module_scope_baselines(root).get(tier)
    import_baselines = testmon_import_time_baselines(root).get(tier)
    try:
        marker = json.loads((root / ".testmondata.full").read_text(encoding="utf-8"))
        source_baselines = _valid_source_baselines(marker).get(tier)
    except (OSError, TypeError, ValueError):
        source_baselines = None
    if module_baselines is None or import_baselines is None or source_baselines is None:
        return True
    source_texts, source_hashes = source_baselines
    source_paths = {
        path for path in changed_paths if path.endswith(".py") and not path.startswith("tests/")
    }
    try:
        current_source_texts, current_source_hashes = _source_snapshots(root)
    except (OSError, UnicodeDecodeError):
        return True
    if current_source_hashes.keys() != source_hashes.keys():
        return True
    for path, current_hash in current_source_hashes.items():
        if current_hash != source_hashes[path]:
            source_paths.add(path)
    if not source_paths:
        return False
    try:
        current_scopes = _module_scope_fingerprints(root)
        if any(
            path not in module_baselines or module_baselines[path] != current_scopes.get(path)
            for path in source_paths
        ):
            return True
    except OSError:
        return True
    for path in source_paths:
        baseline_lines = import_baselines.get(path)
        old_source = source_texts.get(path)
        expected_hash = source_hashes.get(path)
        current_source = current_source_texts.get(path)
        current_hash = current_source_hashes.get(path)
        if (
            baseline_lines is None
            or old_source is None
            or expected_hash is None
            or current_source is None
            or current_hash is None
        ):
            return True
        if hashlib.sha256(old_source.encode("utf-8")).hexdigest() != expected_hash:
            return True
        if current_hash == expected_hash:
            continue
        old_sensitive = _import_sensitive_lines(old_source, set(baseline_lines))
        if old_sensitive is None:
            return True
        old_lines = old_source.splitlines()
        new_lines = current_source.splitlines()
        for tag, old_start, old_end, _new_start, _new_end in difflib.SequenceMatcher(
            a=old_lines, b=new_lines, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                continue
            if old_start == old_end:
                if old_start + 1 in old_sensitive or old_start in old_sensitive:
                    return True
            elif old_sensitive.intersection(range(old_start + 1, old_end + 1)):
                return True
    return False


def _uses_subprocess(tree: ast.AST) -> bool:
    process_modules: set[str] = set()
    system_modules: set[str] = set()
    process_functions: set[str] = set()
    executable_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "subprocess":
                    process_modules.add(alias.asname or alias.name)
                elif alias.name == "sys":
                    system_modules.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "subprocess":
                process_functions.update(alias.asname or alias.name for alias in node.names)
            elif node.module == "sys":
                executable_names.update(
                    alias.asname or alias.name for alias in node.names if alias.name == "executable"
                )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in process_functions:
                return True
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in process_modules
            ):
                return True
        if isinstance(node, ast.Attribute) and node.attr == "executable":
            if isinstance(node.value, ast.Name) and node.value.id in system_modules:
                return True
        if isinstance(node, ast.Name) and node.id in executable_names:
            return True
    return False


def subprocess_consumer_test_paths(root: Path) -> list[str]:
    """Conservatively retain tests whose dependencies may run in child Python processes."""
    selected = []
    for path in sorted((root / "tests").rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, UnicodeDecodeError):
            selected.append(path.relative_to(root).as_posix())
            continue
        if _uses_subprocess(tree):
            selected.append(path.relative_to(root).as_posix())
    return selected


def _exact_environment_fingerprint(root: Path) -> str:
    digest = hashlib.sha256(f"{sys.executable}:{sys.version}:{platform.platform()}".encode())
    installed = sorted(
        f"{distribution.metadata.get('Name', '').casefold()}=={distribution.version}:"
        f"{distribution.read_text('direct_url.json') or ''}"
        for distribution in importlib.metadata.distributions()
    )
    digest.update("\0".join(installed).encode())
    for name in ("pyproject.toml", "uv.lock"):
        path = root / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def testmon_environment_fingerprint(root: Path) -> str:
    if testmon_nested_context(root):
        return ""
    database = root / ".testmondata"
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT environment_name, system_packages, python_version FROM environment "
                "ORDER BY environment_name, system_packages, python_version"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return ""
    if not rows or not all(isinstance(value, str) for row in rows for value in row):
        return ""
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _testmon_fingerprint(root: Path) -> str:
    database = root / ".testmondata"
    if not database.is_file():
        return ""
    source: sqlite3.Connection | None = None
    snapshot: sqlite3.Connection | None = None
    try:
        source = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        snapshot = sqlite3.connect(":memory:")
        source.backup(snapshot)
        contents = snapshot.serialize()
    except sqlite3.Error:
        return ""
    finally:
        if snapshot is not None:
            snapshot.close()
        if source is not None:
            source.close()
    if contents is None:
        return ""
    return hashlib.sha256(contents).hexdigest()


def _testmon_node_baselines(marker: Mapping[str, object]) -> dict[str, list[str]]:
    value = marker.get("test_nodes_by_tier")
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for tier, nodes in value.items():
        if (
            tier not in {"fast", "slow"}
            or not isinstance(nodes, list)
            or not all(isinstance(node, str) for node in nodes)
        ):
            continue
        result[cast(str, tier)] = cast(list[str], nodes)
    return result


def _testmon_nodes_present(root: Path, node_baselines: Mapping[str, list[str]]) -> bool:
    expected = {node for nodes in node_baselines.values() for node in nodes}
    if not expected:
        return True
    database = root / ".testmondata"
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT DISTINCT test_name FROM test_execution").fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return False
    return expected.issubset({row[0] for row in rows if isinstance(row[0], str)})


def testmon_full_tiers(root: Path) -> set[str]:
    if testmon_nested_context(root):
        return set()
    try:
        marker = json.loads((root / ".testmondata.full").read_text(encoding="utf-8"))
        sources = _valid_source_baselines(marker) if isinstance(marker, dict) else {}
        nodes = _testmon_node_baselines(marker) if isinstance(marker, dict) else {}
        if (
            not isinstance(marker, dict)
            or marker.get("fingerprint") != _testmon_fingerprint(root)
            or marker.get("environment") != testmon_environment_fingerprint(root)
            or marker.get("python_environment") != _exact_environment_fingerprint(root)
            or not isinstance(marker.get("tiers"), list)
            or not all(isinstance(tier, str) for tier in marker["tiers"])
            or marker.get("schema") != 7
            or not isinstance(marker.get("module_scopes_by_tier"), dict)
            or not isinstance(marker.get("import_time_lines_by_tier"), dict)
            or any(
                not isinstance(tier, str)
                or not isinstance(marker["module_scopes_by_tier"].get(tier), dict)
                or not isinstance(marker["import_time_lines_by_tier"].get(tier), dict)
                or tier not in sources
                or tier not in nodes
                for tier in marker["tiers"]
            )
        ):
            return set()
        certified = {tier: nodes[tier] for tier in marker["tiers"]}
        if not _testmon_nodes_present(root, certified):
            return set()
        return {tier for tier in marker["tiers"] if tier in {"fast", "slow"}}
    except (OSError, ValueError):
        return set()


def write_testmon_coverage(
    root: Path,
    tiers: set[str],
    *,
    previous_scope_baselines: dict[str, dict[str, str]] | None = None,
    previous_import_baselines: dict[str, dict[str, list[int]]] | None = None,
    refresh_scope_tiers: set[str] | None = None,
    test_dist: str | None = None,
    expected_source_snapshot: tuple[dict[str, str], dict[str, str]] | None = None,
) -> None:
    fingerprint = _testmon_fingerprint(root)
    environment = testmon_environment_fingerprint(root)
    tiers = tiers.intersection({"fast", "slow"})
    if not fingerprint or not environment or not tiers:
        return
    scope_baselines = (
        testmon_module_scope_baselines(root)
        if previous_scope_baselines is None
        else dict(previous_scope_baselines)
    )
    import_baselines = (
        testmon_import_time_baselines(root)
        if previous_import_baselines is None
        else dict(previous_import_baselines)
    )
    existing: object = {}
    try:
        existing = json.loads((root / ".testmondata.full").read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        pass
    valid_sources = _valid_source_baselines(existing) if isinstance(existing, dict) else {}
    source_text_baselines = {tier: values[0] for tier, values in valid_sources.items()}
    source_hash_baselines = {tier: values[1] for tier, values in valid_sources.items()}
    node_baselines = _testmon_node_baselines(existing) if isinstance(existing, dict) else {}
    refresh = (refresh_scope_tiers or set()).intersection(tiers)
    if refresh:
        try:
            current_scopes = _module_scope_fingerprints(root)
            source_texts, source_hashes = _source_snapshots(root)
            if (
                expected_source_snapshot is not None
                and (
                    source_texts,
                    source_hashes,
                )
                != expected_source_snapshot
            ):
                return
            current_import_lines: dict[str, dict[str, list[int]]] = {}
            current_test_nodes: dict[str, list[str]] = {}
            for tier in sorted(refresh):
                result = _import_time_line_coverage(root, tier, test_dist)
                if result is None:
                    return
                current_import_lines[tier], current_test_nodes[tier] = result
            texts_after, hashes_after = _source_snapshots(root)
        except (OSError, UnicodeDecodeError):
            return
        if (
            source_hashes != hashes_after
            or source_texts != texts_after
            or (
                expected_source_snapshot is not None
                and (texts_after, hashes_after) != expected_source_snapshot
            )
        ):
            return
        for tier in refresh:
            scope_baselines[tier] = current_scopes
            import_baselines[tier] = current_import_lines[tier]
            source_text_baselines[tier] = source_texts
            source_hash_baselines[tier] = source_hashes
            node_baselines[tier] = current_test_nodes[tier]
    if (
        not tiers.issubset(scope_baselines)
        or not tiers.issubset(import_baselines)
        or not tiers.issubset(source_text_baselines)
        or not tiers.issubset(source_hash_baselines)
        or not tiers.issubset(node_baselines)
    ):
        return
    certified_nodes = {tier: node_baselines[tier] for tier in tiers}
    if not _testmon_nodes_present(root, certified_nodes):
        return
    fingerprint = _testmon_fingerprint(root)
    if not fingerprint:
        return
    destination = root / ".testmondata.full"
    payload = (
        json.dumps(
            {
                "schema": 7,
                "fingerprint": fingerprint,
                "environment": environment,
                "python_environment": _exact_environment_fingerprint(root),
                "tiers": sorted(tiers),
                "module_scopes_by_tier": {tier: scope_baselines[tier] for tier in sorted(tiers)},
                "import_time_lines_by_tier": {
                    tier: import_baselines[tier] for tier in sorted(tiers)
                },
                "source_texts_by_tier": {
                    tier: source_text_baselines[tier] for tier in sorted(tiers)
                },
                "source_hashes_by_tier": {
                    tier: source_hash_baselines[tier] for tier in sorted(tiers)
                },
                "test_nodes_by_tier": {tier: node_baselines[tier] for tier in sorted(tiers)},
            },
            sort_keys=True,
        )
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=root,
        prefix=".testmondata.full.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)


def record_testmon_full_run(
    root: Path,
    tier: str,
    *,
    test_dist: str | None = None,
    expected_source_snapshot: tuple[dict[str, str], dict[str, str]] | None = None,
) -> None:
    if tier == "all":
        certified_tiers = {"fast", "slow"}
    elif tier in {"fast", "slow"}:
        certified_tiers = {tier}
    else:
        return
    write_testmon_coverage(
        root,
        certified_tiers,
        refresh_scope_tiers=certified_tiers,
        test_dist=test_dist,
        expected_source_snapshot=expected_source_snapshot,
    )


def _module_for(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _lazy_exports(path: Path) -> dict[str, str]:
    try:
        tree = ast.parse(path.read_bytes())
    except (OSError, SyntaxError, UnicodeDecodeError):
        return {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if any(
                isinstance(target, ast.Name) and target.id == "_LAZY_IMPORTS" for target in targets
            ):
                if value is None:
                    return {}
                try:
                    values = ast.literal_eval(value)
                except (ValueError, TypeError):
                    return {}
                if isinstance(values, dict):
                    return {
                        name: entry[0]
                        for name, entry in values.items()
                        if isinstance(name, str)
                        and isinstance(entry, tuple)
                        and len(entry) == 2
                        and isinstance(entry[0], str)
                    }
    return {}


def _runtime_nodes(tree: ast.AST):
    pending = list(ast.iter_child_nodes(tree))
    while pending:
        node = pending.pop()
        yield node
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            pending.extend(node.orelse)
            continue
        pending.extend(ast.iter_child_nodes(node))


def _imports(
    tree: ast.AST,
    module: str,
    *,
    package: bool,
    modules: set[str],
    exports: dict[str, str],
) -> set[str]:
    dependencies: set[str] = set()
    if module == "increment":
        # The package initializer contains TYPE_CHECKING imports for every
        # public name; those do not imply that an importer uses every export.
        return dependencies
    package_name = module if package else module.rpartition(".")[0]
    root_aliases: set[str] = set()
    for node in _runtime_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in modules:
                    dependencies.add(alias.name)
                if alias.name == "increment":
                    root_aliases.add(alias.asname or "increment")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                components = package_name.split(".") if package_name else []
                components = components[: max(0, len(components) - node.level + 1)]
                if base:
                    components.extend(base.split("."))
                base = ".".join(components)
            if base in modules:
                dependencies.add(base)
            for alias in node.names:
                child = f"{base}.{alias.name}" if base else alias.name
                if child in modules:
                    dependencies.add(child)
                if base == "increment":
                    if alias.name in exports:
                        dependencies.add(exports[alias.name])
                    elif alias.name == "*":
                        dependencies.update(exports.values())
    for node in _runtime_nodes(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in root_aliases:
            continue
        if node.attr in exports:
            dependencies.add(exports[node.attr])
        elif f"increment.{node.attr}" in modules:
            dependencies.add(f"increment.{node.attr}")
        else:
            # An unresolved public attribute cannot be safely mapped.
            dependencies.update(exports.values())
    return dependencies


def affected_test_paths(root: Path, changed_paths: list[str]) -> list[str]:
    """Return tests importing changed modules, directly or through local imports.

    Import edges are file-level intentionally: any import is treated as a
    dependency, regardless of whether a symbol is referenced at runtime.
    """
    root = root.resolve()
    test_root = root / "tests"
    changed_paths = list(changed_paths)
    source_roots = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name not in {"tests", "scripts", ".git", ".venv", ".worktrees"}
        and (path / "__init__.py").is_file()
    ]
    python_files = [
        path for source_root in source_roots for path in source_root.rglob("*.py")
    ] + list(test_root.rglob("*.py"))
    module_paths: dict[str, Path] = {}
    package_modules: set[str] = set()
    for path in python_files:
        module = _module_for(path, root)
        module_paths[module] = path
        if path.name == "__init__.py":
            package_modules.add(module)
    modules = set(module_paths)
    dependencies: dict[str, set[str]] = {}
    exports = _lazy_exports(root / "increment" / "__init__.py")
    unparseable_graph = False
    test_modules: dict[str, str] = {}
    for path in python_files:
        module = _module_for(path, root)
        try:
            tree = ast.parse(path.read_bytes())
        except (OSError, SyntaxError, UnicodeDecodeError):
            unparseable_graph = True
            continue
        imports = _imports(
            tree,
            module,
            package=module in package_modules,
            modules=modules,
            exports=exports,
        )
        for imported in tuple(imports):
            components = imported.split(".")
            imports.update(
                ".".join(components[:end])
                for end in range(1, len(components))
                if ".".join(components[:end]) in package_modules
            )
        dependencies[module] = imports
        if path.is_relative_to(test_root) and path.name.startswith("test_"):
            test_modules[path.relative_to(root).as_posix()] = module
    changed_modules = {
        _module_for(root / path, root)
        for path in changed_paths
        if path.endswith(".py") and (root / path).is_file()
    }
    affected_modules = set(changed_modules)
    while True:
        consumers = {
            consumer
            for consumer, imports in dependencies.items()
            if consumer not in affected_modules and imports.intersection(affected_modules)
        }
        if not consumers:
            break
        affected_modules.update(consumers)
    selected = {
        path
        for path, module in test_modules.items()
        if dependencies.get(module, set()).intersection(affected_modules)
    }
    selected.update(
        path for path in changed_paths if path.startswith("tests/") and path.endswith(".py")
    )
    if unparseable_graph and changed_modules:
        selected.update(test_modules)
        selected.update(path.relative_to(root).as_posix() for path in test_root.rglob("test_*.py"))
    return sorted(selected)
