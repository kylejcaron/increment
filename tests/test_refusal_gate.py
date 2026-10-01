"""Bare-raise invariant: increment/**/*.py contains zero raises of
ValueError/RuntimeError/TypeError/NotImplementedError constructed directly or
by a simple local helper, so refusals of those shapes always carry a code.
AssertionError is the internal-invariant exception and is never counted: it
signals a programming error unreachable from any public entry point, and the
project runs without -O.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCREMENT_ROOT = ROOT / "increment"

_TARGET_EXCEPTIONS = frozenset({"ValueError", "RuntimeError", "TypeError", "NotImplementedError"})


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


class _ReturnCounter(ast.NodeVisitor):
    """Collect target exceptions directly returned by one helper body."""

    def __init__(self) -> None:
        self.exceptions: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node  # Nested scopes are separate helpers, not part of this summary.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Call):
            name = _call_name(node.value)
            if name in _TARGET_EXCEPTIONS:
                self.exceptions.add(name)


def _helper_return_exceptions(tree: ast.Module) -> dict[str, frozenset[str]]:
    helpers: dict[str, frozenset[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        counter = _ReturnCounter()
        for statement in node.body:
            counter.visit(statement)
        if counter.exceptions:
            helpers[node.name] = frozenset(counter.exceptions)
    return helpers


class _RaiseCounter(ast.NodeVisitor):
    """Counts bare raises of the four target exceptions, keyed by enclosing scope."""

    def __init__(self, helper_exceptions: dict[str, frozenset[str]]) -> None:
        self._stack: list[str] = []
        self._helper_exceptions = helper_exceptions
        self.counts: dict[tuple[str, str], int] = {}

    def _qualname(self) -> str:
        return ".".join(self._stack) if self._stack else "<module>"

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    def visit_Raise(self, node: ast.Raise) -> None:
        exc = node.exc
        # Bare `raise` / `raise name` are not Call nodes; skipped automatically.
        if isinstance(exc, ast.Call):
            name = _call_name(exc)
            exceptions = (
                frozenset({name})
                if name in _TARGET_EXCEPTIONS
                else self._helper_exceptions.get(name or "", frozenset())
            )
            for exception in exceptions:
                key = (self._qualname(), exception)
                self.counts[key] = self.counts.get(key, 0) + 1
        self.generic_visit(node)


def collect_keys(package_root: Path) -> set[str]:
    """Every ``path::qualname::Exc::count`` key under *package_root*, relative
    to its parent (so a synthetic ``<tmp>/increment`` yields ``increment/...``,
    matching production)."""
    keys: set[str] = set()
    base = package_root.parent
    for path in sorted(package_root.rglob("*.py")):
        rel = path.relative_to(base).as_posix()
        tree = ast.parse(path.read_text(), filename=rel)
        counter = _RaiseCounter(_helper_return_exceptions(tree))
        counter.visit(tree)
        for (qualname, exc), count in counter.counts.items():
            keys.add(f"{rel}::{qualname}::{exc}::{count}")
    return keys


def test_bare_raise_gate_is_empty() -> None:
    assert collect_keys(INCREMENT_ROOT) == set()


def test_collect_keys_finds_every_bare_raise_by_scope(tmp_path: Path) -> None:
    package = tmp_path / "increment"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "widget.py").write_text(
        "def build(x):\n"
        "    if x < 0:\n"
        "        raise ValueError('negative')\n"
        "    return x\n"
        "\n"
        "\n"
        "class Thing:\n"
        "    def check(self, y):\n"
        "        if y is None:\n"
        "            raise TypeError('nope')\n"
    )
    baseline = {
        "increment/widget.py::build::ValueError::1",
        "increment/widget.py::Thing.check::TypeError::1",
    }
    assert collect_keys(package) == baseline


def test_collect_keys_counts_repeated_raises_within_a_scope(tmp_path: Path) -> None:
    package = tmp_path / "increment"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "widget.py").write_text(
        "def build(x):\n    if x < 0:\n        raise ValueError('negative')\n    return x\n"
    )
    baseline = collect_keys(package)

    # A second raise in an already-baselined scope changes its count key; a
    # brand-new function's raise has no key at all -- both surface as "new".
    (package / "widget.py").write_text(
        "def build(x):\n"
        "    if x < 0:\n"
        "        raise ValueError('negative')\n"
        "    if x > 100:\n"
        "        raise ValueError('too large')\n"
        "    return x\n"
        "\n"
        "\n"
        "def parse(s):\n"
        "    raise RuntimeError('boom')\n"
    )
    new_keys = collect_keys(package) - baseline
    assert new_keys == {
        "increment/widget.py::build::ValueError::2",
        "increment/widget.py::parse::RuntimeError::1",
    }


def test_collect_keys_finds_a_helper_created_exception_only_when_raised(tmp_path: Path) -> None:
    package = tmp_path / "increment"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "widget.py").write_text(
        "def _invalid(message):\n"
        "    return ValueError(message)\n"
        "\n"
        "\n"
        "def handled():\n"
        "    return str(_invalid('rendered for logging'))\n"
        "\n"
        "\n"
        "def build():\n"
        "    raise _invalid('raised')\n"
    )
    assert collect_keys(package) == {"increment/widget.py::build::ValueError::1"}
