"""Ratchets for brittle exception assertions and private test reach-ins.

H1 rejects ``pytest.raises(X, match=...)`` when reflection proves that ``X``
is a :class:`~increment.errors.CodedError` subclass. Message wording is not a
stable contract for those errors; tests should assert ``.code`` and
``.context`` instead. Exception expressions that cannot be resolved
unambiguously are findings too, so the static check fails closed.

H2 finds any access to one of seven private implementation attributes. This is
deliberately a plain AST name check rather than type inference: the same names
also occur on several non-Analysis implementation classes, and reaching into
those objects is still private coupling. Only ``tests/analysis_factory.py``
and ``tests/source_conformance.py`` are sanctioned shims.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pytest

from increment.errors import CodedError

ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = Path(__file__).resolve().parent
ALLOWLIST_PATH = Path(__file__).with_name("test_hygiene_allowlist.txt")

_PRIVATE_ATTRS = frozenset({"_src", "_plan", "_state", "_defs", "_con", "_experiment", "_session"})
_SANCTIONED_SHIMS = frozenset({"tests/analysis_factory.py", "tests/source_conformance.py"})
_UNRESOLVED = object()
_DYNAMIC_BINDING = object()
_UNRESOLVED_REASON = "not statically resolvable; verify the runtime exception contract"


def _qualname(stack: list[str]) -> str:
    return ".".join(stack) if stack else "<module>"


@dataclass(frozen=True)
class _ImportBinding:
    module: str
    attribute: str | None


@dataclass
class _ScopeBindings:
    kind: str
    names: dict[str, set[_ImportBinding | object]]
    has_star_import: bool = False


class _ScopeBindingCollector(ast.NodeVisitor):
    """Summarize bindings owned by one lexical scope without entering children."""

    def __init__(self, kind: str) -> None:
        self.scope = _ScopeBindings(kind=kind, names={})

    def _add(self, name: str, binding: _ImportBinding | object) -> None:
        self.scope.names.setdefault(name, set()).add(binding)

    def add_arguments(self, arguments: ast.arguments) -> None:
        args = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
        if arguments.vararg is not None:
            args.append(arguments.vararg)
        if arguments.kwarg is not None:
            args.append(arguments.kwarg)
        for argument in args:
            self._add(argument.arg, _DYNAMIC_BINDING)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            module = alias.name if alias.asname else alias.name.split(".")[0]
            self._add(local_name, _ImportBinding(module, None))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                self.scope.has_star_import = True
                continue
            local_name = alias.asname or alias.name
            if node.level or node.module is None:
                self._add(local_name, _DYNAMIC_BINDING)
            else:
                self._add(local_name, _ImportBinding(node.module, alias.name))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store | ast.Del):
            self._add(node.id, _DYNAMIC_BINDING)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add(node.name, _DYNAMIC_BINDING)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node.name, _DYNAMIC_BINDING)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node.name, _DYNAMIC_BINDING)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._add(node.name, _DYNAMIC_BINDING)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            self._add(name, _DYNAMIC_BINDING)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        for name in node.names:
            self._add(name, _DYNAMIC_BINDING)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None:
            self._add(node.name, _DYNAMIC_BINDING)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None:
            self._add(node.name, _DYNAMIC_BINDING)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None:
            self._add(node.rest, _DYNAMIC_BINDING)
        self.generic_visit(node)


def _scope_bindings(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> _ScopeBindings:
    if isinstance(node, ast.Module):
        kind = "module"
    elif isinstance(node, ast.ClassDef):
        kind = "class"
    else:
        kind = "function"
    collector = _ScopeBindingCollector(kind)
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        collector.add_arguments(node.args)
    if isinstance(node, ast.Lambda):
        collector.visit(node.body)
    else:
        for statement in node.body:
            collector.visit(statement)
    return collector.scope


@cache
def _reflect_import(binding: _ImportBinding) -> object:
    try:
        module = importlib.import_module(binding.module)
    except Exception:
        return _UNRESOLVED
    if binding.attribute is None:
        return module
    return getattr(module, binding.attribute, _UNRESOLVED)


@dataclass(frozen=True)
class _MatchFindings:
    coded: frozenset[str]
    unresolved: frozenset[str]

    @property
    def keys(self) -> set[str]:
        return set(self.coded | self.unresolved)


class _MatchRaiseVisitor(ast.NodeVisitor):
    """Find matched coded errors and ambiguous matched exception expressions."""

    def __init__(self, relpath: str, tree: ast.Module) -> None:
        self.relpath = relpath
        self.name_stack: list[str] = []
        self.scope_stack = [_scope_bindings(tree)]
        self.coded_keys: set[str] = set()
        self.unresolved_keys: set[str] = set()

    def _key(self, exception: str) -> str:
        return f"{self.relpath}::{_qualname(self.name_stack)}::{exception}"

    def _resolve_name(self, name: str) -> object:
        inside_function = False
        for scope in reversed(self.scope_stack):
            if scope.kind == "function":
                inside_function = True
            elif scope.kind == "class" and inside_function:
                # Unqualified names in a method do not close over its class body.
                continue

            bindings = scope.names.get(name)
            if bindings is not None:
                if scope.has_star_import or len(bindings) != 1:
                    return _UNRESOLVED
                binding = next(iter(bindings))
                if not isinstance(binding, _ImportBinding):
                    return _UNRESOLVED
                return _reflect_import(binding)
            if scope.has_star_import:
                return _UNRESOLVED
        return getattr(builtins, name, _UNRESOLVED)

    def _record_exception(self, node: ast.expr) -> None:
        if isinstance(node, ast.Tuple):
            for element in node.elts:
                self._record_exception(element)
            return
        if not isinstance(node, ast.Name):
            self.unresolved_keys.add(self._key(ast.unparse(node)))
            return

        resolved = self._resolve_name(node.id)
        if not isinstance(resolved, type):
            self.unresolved_keys.add(self._key(ast.unparse(node)))
            return
        try:
            is_coded = issubclass(resolved, CodedError)
        except TypeError:
            self.unresolved_keys.add(self._key(ast.unparse(node)))
            return
        if is_coded:
            self.coded_keys.add(self._key(resolved.__name__))

    def _visit_named_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self.name_stack.append(node.name)
        self.scope_stack.append(_scope_bindings(node))
        self.generic_visit(node)
        self.scope_stack.pop()
        self.name_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_named_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_named_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_named_scope(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.scope_stack.append(_scope_bindings(node))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        function = node.func
        is_raises = (isinstance(function, ast.Attribute) and function.attr == "raises") or (
            isinstance(function, ast.Name) and function.id == "raises"
        )
        has_match = any(keyword.arg == "match" for keyword in node.keywords)
        if is_raises and has_match and node.args:
            self._record_exception(node.args[0])


def _collect_match_findings(tests_root: Path) -> _MatchFindings:
    coded: set[str] = set()
    unresolved: set[str] = set()
    base = tests_root.parent
    for path in sorted(tests_root.rglob("*.py")):
        relpath = path.relative_to(base).as_posix()
        tree = ast.parse(path.read_text(), filename=relpath)
        visitor = _MatchRaiseVisitor(relpath, tree)
        visitor.visit(tree)
        coded.update(visitor.coded_keys)
        unresolved.update(visitor.unresolved_keys)
    return _MatchFindings(frozenset(coded), frozenset(unresolved))


def collect_match_keys(tests_root: Path) -> set[str]:
    """Return H1 keys beneath *tests_root*, relative to its parent."""
    return _collect_match_findings(tests_root).keys


class _PrivateAttributeVisitor(ast.NodeVisitor):
    """Find every access whose attribute name is one of ``_PRIVATE_ATTRS``."""

    def __init__(self, relpath: str) -> None:
        self.relpath = relpath
        self.name_stack: list[str] = []
        self.keys: set[str] = set()

    def _visit_named_scope(
        self,
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        self.name_stack.append(node.name)
        self.generic_visit(node)
        self.name_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._visit_named_scope(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_named_scope(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_named_scope(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)
        if node.attr in _PRIVATE_ATTRS:
            qualname = _qualname(self.name_stack)
            self.keys.add(f"{self.relpath}::{qualname}::{node.attr}")


def collect_private_attribute_keys(tests_root: Path) -> set[str]:
    """Return H2 keys beneath *tests_root*, excluding exactly the two shims."""
    keys: set[str] = set()
    base = tests_root.parent
    for path in sorted(tests_root.rglob("*.py")):
        relpath = path.relative_to(base).as_posix()
        if relpath in _SANCTIONED_SHIMS:
            continue
        tree = ast.parse(path.read_text(), filename=relpath)
        visitor = _PrivateAttributeVisitor(relpath)
        visitor.visit(tree)
        keys.update(visitor.keys)
    return keys


def _load_allowlist(
    path: Path = ALLOWLIST_PATH,
    *,
    unresolved_match_keys: frozenset[str] = frozenset(),
) -> tuple[set[str], set[str]]:
    """Load the H1/H2 sections and enforce explanations for uncertain H1 keys."""
    match_keys: set[str] = set()
    attribute_keys: set[str] = set()
    explained: set[str] = set()
    section = match_keys
    seen_separator = False

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped == "# ---":
            if seen_separator:
                raise AssertionError(f"{path}:{line_number}: duplicate section separator")
            seen_separator = True
            section = attribute_keys
            continue
        if not stripped or stripped.startswith("#"):
            continue

        key, separator, reason = stripped.partition("  # ")
        key = key.strip()
        if key.count("::") < 2:
            raise AssertionError(f"{path}:{line_number}: malformed allowlist key {key!r}")
        if key in match_keys or key in attribute_keys:
            raise AssertionError(f"{path}:{line_number}: duplicate allowlist key {key!r}")
        section.add(key)
        if separator and reason.strip():
            explained.add(key)

    missing_reasons = (match_keys & set(unresolved_match_keys)) - explained
    assert not missing_reasons, (
        "unresolved H1 allowlist entries need a one-line explanation after `  # `: "
        f"{sorted(missing_reasons)}"
    )
    return match_keys, attribute_keys


def _write_allowlist() -> None:
    findings = _collect_match_findings(TESTS_ROOT)
    match_lines = [
        f"{key}  # {_UNRESOLVED_REASON}" if key in findings.unresolved else key
        for key in sorted(findings.keys)
    ]
    attribute_lines = sorted(collect_private_attribute_keys(TESTS_ROOT))
    lines = [
        "# H1: pytest.raises(..., match=...) on coded or unresolved exceptions",
        *match_lines,
        "# ---",
        "# H2: private-attribute reach-ins outside the two sanctioned shims",
        *attribute_lines,
    ]
    ALLOWLIST_PATH.write_text("\n".join(lines) + "\n")


@pytest.mark.slow
def test_no_regex_match_on_coded_errors() -> None:
    findings = _collect_match_findings(TESTS_ROOT)
    allowed, _ = _load_allowlist(unresolved_match_keys=findings.unresolved)
    observed = findings.keys
    new_keys = observed - allowed
    stale_keys = allowed - observed
    assert not new_keys, (
        "new pytest.raises(..., match=) on a coded error (or an unresolvable "
        f"exception argument) not in the allowlist: {sorted(new_keys)}"
    )
    assert not stale_keys, f"H1 allowlist is stale: {sorted(stale_keys)}"


@pytest.mark.slow
def test_no_private_attribute_reach_ins() -> None:
    _, allowed = _load_allowlist()
    observed = collect_private_attribute_keys(TESTS_ROOT)
    new_keys = observed - allowed
    stale_keys = allowed - observed
    assert not new_keys, f"new private-attribute reach-in not in the allowlist: {sorted(new_keys)}"
    assert not stale_keys, f"H2 allowlist is stale: {sorted(stale_keys)}"


def test_collect_match_keys_uses_reflected_class_name_and_ignores_noncoded_cases(
    tmp_path: Path,
) -> None:
    package = tmp_path / "tests"
    package.mkdir()
    (package / "test_widget.py").write_text(
        "import pytest\n"
        "from increment.errors import InvalidRequestError as RequestRefusal\n"
        "def test_x():\n"
        "    with pytest.raises(RequestRefusal, match='unstable wording'):\n"
        "        pass\n"
        "    with pytest.raises(ValueError, match='stable non-coded error'):\n"
        "        pass\n"
        "    with pytest.raises(RequestRefusal):\n"
        "        pass\n"
    )
    assert collect_match_keys(package) == {"tests/test_widget.py::test_x::InvalidRequestError"}


def test_collect_match_keys_fails_closed_at_real_static_uncertainty_boundaries(
    tmp_path: Path,
) -> None:
    package = tmp_path / "tests"
    package.mkdir()
    (package / "test_widget.py").write_text(
        "import pytest\n"
        "from builtins import ValueError as ConflictedError\n"
        "from increment.errors import InvalidRequestError as ConflictedError\n"
        "def test_conflicting_imports():\n"
        "    with pytest.raises(ConflictedError, match='wording'):\n"
        "        pass\n"
        "def test_parameter(exception_type):\n"
        "    with pytest.raises((ValueError, exception_type), match='wording'):\n"
        "        pass\n"
        "def test_qualified():\n"
        "    with pytest.raises(yaml.constructor.ConstructorError, match='wording'):\n"
        "        pass\n"
        "def test_local_import():\n"
        "    from increment.errors import InvalidRequestError as LocalError\n"
        "    with pytest.raises((LocalError, selected_error), match='wording'):\n"
        "        pass\n"
        "def test_shadowed():\n"
        "    from increment.errors import InvalidRequestError\n"
        "    InvalidRequestError = ValueError\n"
        "    with pytest.raises(InvalidRequestError, match='wording'):\n"
        "        pass\n"
        "def test_lambda():\n"
        "    return lambda ValueError: pytest.raises(ValueError, match='wording')\n"
    )
    assert collect_match_keys(package) == {
        "tests/test_widget.py::test_conflicting_imports::ConflictedError",
        "tests/test_widget.py::test_parameter::exception_type",
        "tests/test_widget.py::test_qualified::yaml.constructor.ConstructorError",
        "tests/test_widget.py::test_local_import::InvalidRequestError",
        "tests/test_widget.py::test_local_import::selected_error",
        "tests/test_widget.py::test_shadowed::InvalidRequestError",
        "tests/test_widget.py::test_lambda::ValueError",
    }


def test_unresolved_allowlist_entry_requires_an_inline_explanation(tmp_path: Path) -> None:
    key = "tests/test_widget.py::test_x::exception_type"
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text(f"{key}\n# ---\n")
    with pytest.raises(AssertionError) as raised:
        _load_allowlist(allowlist, unresolved_match_keys=frozenset({key}))
    assert key in str(raised.value)

    allowlist.write_text(f"{key}  # selected by a parametrized test case\n# ---\n")
    assert _load_allowlist(allowlist, unresolved_match_keys=frozenset({key})) == (
        {key},
        set(),
    )


def test_collect_private_attribute_keys_finds_all_names_and_only_exact_shims(
    tmp_path: Path,
) -> None:
    package = tmp_path / "tests"
    package.mkdir()
    (package / "test_widget.py").write_text(
        "class TestWidget:\n"
        "    def test_x(self, analysis):\n"
        "        return (analysis._src, analysis._plan, analysis._state, "
        "analysis._defs, analysis._con, analysis._experiment, analysis._session, "
        "analysis._public)\n"
    )
    (package / "analysis_factory.py").write_text("def shim(x):\n    return x._plan\n")
    (package / "source_conformance.py").write_text("def shim(x):\n    return x._src\n")
    nested = package / "unit"
    nested.mkdir()
    (nested / "analysis_factory.py").write_text(
        "def test_not_the_sanctioned_path(x):\n    return x._plan\n"
    )

    assert collect_private_attribute_keys(package) == {
        "tests/test_widget.py::TestWidget.test_x::_src",
        "tests/test_widget.py::TestWidget.test_x::_plan",
        "tests/test_widget.py::TestWidget.test_x::_state",
        "tests/test_widget.py::TestWidget.test_x::_defs",
        "tests/test_widget.py::TestWidget.test_x::_con",
        "tests/test_widget.py::TestWidget.test_x::_experiment",
        "tests/test_widget.py::TestWidget.test_x::_session",
        "tests/unit/analysis_factory.py::test_not_the_sanctioned_path::_plan",
    }


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-allowlist"]:
        _write_allowlist()
    else:
        print("usage: python tests/test_test_hygiene.py --write-allowlist", file=sys.stderr)
        sys.exit(1)
