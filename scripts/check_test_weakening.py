from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hypothesis import settings as hypothesis_settings


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str
    line: int
    detail: str
    identity: tuple[str, ...] = ()


@dataclass
class TestFacts:
    path: str
    qualname: str
    line: int
    skips: set[str] = field(default_factory=set)
    examples: int | None = None
    timeout: float | None = 300.0
    tolerances: dict[tuple[str, str], list[tuple[float | str, ...]]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass
class SuiteFacts:
    tests: dict[tuple[str, str], TestFacts] = field(default_factory=dict)
    helpers: dict[tuple[str, str], TestFacts] = field(default_factory=dict)
    test_counts: Counter[str] = field(default_factory=Counter)
    assertion_counts: Counter[str] = field(default_factory=Counter)


@dataclass(frozen=True)
class ToleranceSpec:
    relative: str
    absolute: str
    default_relative: float
    default_absolute: float
    relative_position: int | None = None
    absolute_position: int | None = None


_TOLERANCES = {
    "pytest.approx": ToleranceSpec("rel", "abs", 1e-6, 1e-12, 1, 2),
    "math.isclose": ToleranceSpec("rel_tol", "abs_tol", 1e-9, 0.0),
    "numpy.isclose": ToleranceSpec("rtol", "atol", 1e-5, 1e-8, 2, 3),
    "numpy.allclose": ToleranceSpec("rtol", "atol", 1e-5, 1e-8, 2, 3),
    "numpy.testing.assert_allclose": ToleranceSpec("rtol", "atol", 1e-7, 0.0, 2, 3),
    "pandas.testing.assert_frame_equal": ToleranceSpec("rtol", "atol", 1e-5, 1e-8),
    "pandas.testing.assert_series_equal": ToleranceSpec("rtol", "atol", 1e-5, 1e-8),
    "polars.testing.assert_frame_equal": ToleranceSpec("rel_tol", "abs_tol", 1e-5, 1e-8),
    "polars.testing.assert_series_equal": ToleranceSpec("rel_tol", "abs_tol", 1e-5, 1e-8),
}
_ASSERT_CALLS = {"pytest.raises", "pytest.warns", "pytest.approx"}
_TEST_SKIP_CALLS = {"pytest.skip", "pytest.skipif", "pytest.xfail", "pytest.importorskip"}
_TIMEOUT_DEFAULT = 300.0


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _scope_imports(node: ast.AST, inherited: Mapping[str, str]) -> dict[str, str]:
    aliases = dict(inherited)

    class ImportCollector(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

    collector = ImportCollector()
    for statement in getattr(node, "body", ()):
        collector.visit(statement)
    return aliases


def _imports(tree: ast.Module) -> dict[str, str]:
    return _scope_imports(tree, {})


def _resolve(name: str, aliases: Mapping[str, str]) -> str:
    first, dot, rest = name.partition(".")
    if first in aliases:
        return aliases[first] + (dot + rest if dot else "")
    return name


def _literal(node: ast.AST | None) -> float | None:
    if node is None:
        return None
    try:
        value = ast.literal_eval(node)
        return float(value) if isinstance(value, (int, float)) else None
    except (ValueError, TypeError):
        return None


def _literal_int(node: ast.AST | None) -> int | None:
    value = _literal(node)
    return int(value) if value is not None and value.is_integer() else None


def _limit(node: ast.AST | None) -> float | None:
    value = _literal(node)
    if value is None:
        return None
    return None if value <= 0 else value


def _marker_detail(mark: str, node: ast.Call | None = None) -> str | None:
    if mark not in {"skip", "skipif", "xfail"}:
        return None
    if mark == "skip":
        return f"mark:{mark}:default"
    if node is None:
        return f"mark:{mark}:default"
    conditions = list(node.args)
    condition_kw = next((kw.value for kw in node.keywords if kw.arg == "condition"), None)
    if not conditions and condition_kw is not None:
        conditions.append(condition_kw)
    if conditions and all(
        isinstance(condition, ast.Constant) and condition.value is False for condition in conditions
    ):
        return None
    keywords = ",".join(
        f"{kw.arg}={ast.unparse(kw.value)}" for kw in node.keywords if kw.arg != "reason"
    )
    if mark == "xfail" and not conditions and not keywords:
        return f"mark:{mark}:default"
    args = ",".join(ast.unparse(value) for value in node.args)
    return f"mark:{mark}:({args};{keywords})"


def _mark_details(node: ast.AST, aliases: Mapping[str, str]) -> set[str]:
    called_attributes = {id(child.func) for child in ast.walk(node) if isinstance(child, ast.Call)}
    details: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            name = _resolve(_dotted(child.func), aliases)
            if name.startswith("pytest.mark."):
                detail = _marker_detail(name.rsplit(".", 1)[-1], child)
                if detail is not None:
                    details.add(detail)
        elif isinstance(child, ast.Attribute) and id(child) not in called_attributes:
            name = _resolve(_dotted(child), aliases)
            if name.startswith("pytest.mark."):
                detail = _marker_detail(name.rsplit(".", 1)[-1])
                if detail is not None:
                    details.add(detail)
    return details


def _timeout_marks(node: ast.AST, aliases: Mapping[str, str]) -> list[float | None]:
    values: list[float | None] = []
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and _resolve(_dotted(child.func), aliases) == "pytest.mark.timeout"
        ):
            value = (
                child.args[0]
                if child.args
                else next(
                    (kw.value for kw in child.keywords if kw.arg in {"timeout", "seconds"}), None
                )
            )
            values.append(
                _TIMEOUT_DEFAULT
                if value is None or isinstance(value, ast.Constant) and value.value is None
                else _limit(value)
            )
    return values


def _pytestmark_values(node: ast.stmt) -> list[ast.AST]:
    value: ast.AST | None = None
    if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "pytestmark" for target in node.targets
    ):
        value = node.value
    elif (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "pytestmark"
    ):
        value = node.value
    if value is None:
        return []
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return list(value.elts)
    return [value]


def _settings_bindings(tree: ast.Module, aliases: Mapping[str, str]) -> dict[str, int | None]:
    bindings: dict[str, int | None] = {}

    def evaluate(value: ast.AST) -> int | None:
        if isinstance(value, ast.Name):
            return bindings.get(value.id)
        if (
            not isinstance(value, ast.Call)
            or _resolve(_dotted(value.func), aliases) != "hypothesis.settings"
        ):
            return None
        parent = evaluate(value.args[0]) if value.args else None
        maximum = next(
            (_literal_int(kw.value) for kw in value.keywords if kw.arg == "max_examples"), None
        )
        return maximum if maximum is not None else parent

    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = evaluate(node.value)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            bindings[node.target.id] = evaluate(node.value)
    return bindings


def _settings_value(
    node: ast.AST, aliases: Mapping[str, str], bindings: Mapping[str, int | None]
) -> int | None:
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if (
        not isinstance(node, ast.Call)
        or _resolve(_dotted(node.func), aliases) != "hypothesis.settings"
    ):
        return None
    parent = _settings_value(node.args[0], aliases, bindings) if node.args else None
    maximum = next(
        (_literal_int(kw.value) for kw in node.keywords if kw.arg == "max_examples"), None
    )
    return maximum if maximum is not None else parent


def _tolerance_arguments(
    node: ast.Call, spec: ToleranceSpec
) -> tuple[ast.AST | None, ast.AST | None, set[int], set[str]]:
    keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg}
    relative = keywords.get(spec.relative)
    absolute = keywords.get(spec.absolute)
    excluded_positions: set[int] = set()
    if spec.relative_position is not None:
        index = spec.relative_position
        if relative is None and len(node.args) > index:
            relative = node.args[index]
            excluded_positions.add(index)
    if spec.absolute_position is not None:
        index = spec.absolute_position
        if absolute is None and len(node.args) > index:
            absolute = node.args[index]
            excluded_positions.add(index)
    return relative, absolute, excluded_positions, {spec.relative, spec.absolute, "check_exact"}


def _call_key(
    node: ast.Call,
    canonical: str,
    excluded_positions: set[int],
    excluded_keywords: set[str],
    comparison_subject: str | None = None,
) -> str:
    args = [ast.unparse(value) for i, value in enumerate(node.args) if i not in excluded_positions]
    kwargs = [
        f"{kw.arg}={ast.unparse(kw.value)}"
        for kw in node.keywords
        if kw.arg not in excluded_keywords
    ]
    subject = f"[{comparison_subject}]" if comparison_subject is not None else ""
    return f"{canonical}{subject}({','.join(args)};{','.join(kwargs)})"


def _normalized_tolerance(
    expression: ast.AST | None, default: float, *, explicit: bool
) -> float | str:
    if not explicit:
        return default
    literal = _literal(expression)
    return literal if literal is not None else ast.unparse(expression) if expression else "None"


def _tolerance_sort(values: tuple[float | str, ...]) -> tuple[tuple[int, float | str], ...]:
    return tuple((0, value) if isinstance(value, float) else (1, value) for value in values)


def _assertion_total(tree: ast.Module, module_aliases: Mapping[str, str]) -> int:
    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.aliases = [module_aliases]
            self.total = 0

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            for default in (*node.args.defaults, *node.args.kw_defaults):
                if default is not None:
                    self.visit(default)
            local = _scope_imports(node, self.aliases[-1])
            self.aliases.append(local)
            for statement in node.body:
                self.visit(statement)
            self.aliases.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            for decorator in node.decorator_list:
                self.visit(decorator)
            local = _scope_imports(node, self.aliases[0])
            self.aliases.append(local)
            for statement in node.body:
                self.visit(statement)
            self.aliases.pop()

        def visit_Assert(self, node: ast.Assert) -> None:
            self.total += 1
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            canonical = _resolve(_dotted(node.func), self.aliases[-1])
            leaf = canonical.rsplit(".", 1)[-1]
            if canonical in _ASSERT_CALLS or (
                leaf.startswith("assert_")
                and canonical.startswith(("numpy.testing.", "pandas.testing.", "polars.testing."))
            ):
                self.total += 1
            self.generic_visit(node)

    visitor = Visitor()
    for statement in tree.body:
        visitor.visit(statement)
    return visitor.total


def _tests(source: str, path: str) -> tuple[list[TestFacts], list[TestFacts], int]:  # noqa: C901
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return [], [], 0
    aliases = _imports(tree)
    settings = _settings_bindings(tree, aliases)
    module_skips: set[str] = set()
    module_timeouts: list[float | None] = []
    for statement in tree.body:
        for mark in _pytestmark_values(statement):
            module_skips.update(_mark_details(mark, aliases))
            module_timeouts.extend(_timeout_marks(mark, aliases))
        value = (
            statement.value
            if isinstance(statement, ast.Expr | ast.Assign | ast.AnnAssign)
            else None
        )
        if (
            isinstance(value, ast.Call)
            and _resolve(_dotted(value.func), aliases) == "pytest.importorskip"
        ):
            module_skips.add("call:importorskip")
    tests: list[TestFacts] = []
    assertion_total = _assertion_total(tree, aliases)

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.classes: list[str] = []
            self.functions: list[str] = []
            self.current: TestFacts | None = TestFacts(path, "<module>", 1)
            self.module_facts = self.current
            self.helper_facts: list[TestFacts] = [self.module_facts]
            self.in_test = False
            self.inherited_skips: list[set[str]] = []
            self.inherited_timeouts: list[list[float | None]] = []
            self.aliases = [aliases]
            self.module_aliases = aliases
            self.comparison_subjects: list[str | None] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            parent_aliases = self.aliases[-1]
            class_aliases = _scope_imports(node, self.module_aliases)
            self.aliases.append(class_aliases)
            self.classes.append(node.name)
            skips = set().union(
                *(_mark_details(decorator, parent_aliases) for decorator in node.decorator_list)
            )
            timeouts = [
                value
                for decorator in reversed(node.decorator_list)
                for value in _timeout_marks(decorator, parent_aliases)
            ]
            for statement in node.body:
                for mark in _pytestmark_values(statement):
                    skips.update(_mark_details(mark, class_aliases))
                    timeouts.extend(_timeout_marks(mark, class_aliases))
            self.inherited_skips.append(skips)
            self.inherited_timeouts.append(timeouts)
            self.generic_visit(node)
            self.inherited_timeouts.pop()
            self.inherited_skips.pop()
            self.classes.pop()
            self.aliases.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            eligible = not self.functions and (
                not self.classes
                or self.classes[-1].startswith("Test")
                and all(name.startswith("Test") for name in self.classes)
            )
            decorator_aliases = self.aliases[-1]
            parent_aliases = self.aliases[-1] if self.functions else self.module_aliases
            if node.name.startswith("test_") and eligible:
                fact = TestFacts(path, ".".join([*self.classes, node.name]), node.lineno)
                fact.skips.update(module_skips)
                fact.skips.update(mark for marks in self.inherited_skips for mark in marks)
                fact.timeout = module_timeouts[0] if module_timeouts else _TIMEOUT_DEFAULT
                for marks in self.inherited_timeouts:
                    if marks:
                        fact.timeout = marks[0]
                tests.append(fact)
                previous = self.current
                was_in_test = self.in_test
                self.current = fact
                self.in_test = True
                function_timeouts: list[float | None] = []
                has_given = False
                for decorator in reversed(node.decorator_list):
                    fact.skips.update(_mark_details(decorator, decorator_aliases))
                    function_timeouts.extend(_timeout_marks(decorator, decorator_aliases))
                    for child in ast.walk(decorator):
                        if (
                            isinstance(child, ast.Call)
                            and _resolve(_dotted(child.func), decorator_aliases)
                            == "hypothesis.given"
                        ):
                            has_given = True
                    value = _settings_value(decorator, decorator_aliases, settings)
                    if value is not None:
                        fact.examples = value
                if has_given and fact.examples is None:
                    default_settings = hypothesis_settings.default
                    assert default_settings is not None
                    fact.examples = default_settings.max_examples
                if function_timeouts:
                    fact.timeout = function_timeouts[0]
                self.functions.append(node.name)
                self.aliases.append(_scope_imports(node, parent_aliases))
                for statement in node.body:
                    self.visit(statement)
                self.aliases.pop()
                self.functions.pop()
                self.current = previous
                self.in_test = was_in_test
            else:
                helper = None
                if not self.in_test:
                    helper = TestFacts(path, ".".join([*self.classes, node.name]), node.lineno)
                    self.helper_facts.append(helper)
                previous = self.current
                if helper is not None:
                    self.current = helper
                self.functions.append(node.name)
                self.aliases.append(_scope_imports(node, parent_aliases))
                self.generic_visit(node)
                self.aliases.pop()
                self.functions.pop()
                self.current = previous

        visit_AsyncFunctionDef = visit_FunctionDef

        def _contains_approx(self, node: ast.AST) -> bool:
            return any(
                isinstance(child, ast.Call)
                and _resolve(_dotted(child.func), self.aliases[-1]) == "pytest.approx"
                for child in ast.walk(node)
            )

        def visit_Compare(self, node: ast.Compare) -> None:
            subject = None
            if any(self._contains_approx(value) for value in node.comparators):
                subject = ast.unparse(node.left)
            elif self._contains_approx(node.left) and node.comparators:
                subject = ast.unparse(node.comparators[0])
            self.comparison_subjects.append(subject)
            self.generic_visit(node)
            self.comparison_subjects.pop()

        def visit_Call(self, node: ast.Call) -> None:
            canonical = _resolve(_dotted(node.func), self.aliases[-1])
            leaf = canonical.rsplit(".", 1)[-1]
            if self.current and canonical in _TEST_SKIP_CALLS:
                self.current.skips.add(f"call:{leaf}")
            spec = _TOLERANCES.get(canonical)
            if self.current and spec:
                relative, absolute, excluded_positions, excluded_keywords = _tolerance_arguments(
                    node, spec
                )
                if canonical == "pytest.approx":
                    if isinstance(relative, ast.Constant) and relative.value is None:
                        relative = None
                    if isinstance(absolute, ast.Constant) and absolute.value is None:
                        absolute = None
                    if absolute is not None and relative is None:
                        relative = ast.Constant(value=0.0)
                values = (
                    _normalized_tolerance(
                        relative, spec.default_relative, explicit=relative is not None
                    ),
                    _normalized_tolerance(
                        absolute, spec.default_absolute, explicit=absolute is not None
                    ),
                )
                exact_kw = next((kw.value for kw in node.keywords if kw.arg == "check_exact"), None)
                if isinstance(exact_kw, ast.Constant) and exact_kw.value is True:
                    values = (0.0, 0.0)
                subject = (
                    self.comparison_subjects[-1]
                    if canonical == "pytest.approx" and self.comparison_subjects
                    else None
                )
                key = (
                    canonical,
                    _call_key(
                        node,
                        canonical,
                        excluded_positions,
                        excluded_keywords,
                        subject,
                    ),
                )
                self.current.tolerances[key].append(values)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(tree)
    return tests, visitor.helper_facts, assertion_total


def collect(sources: Mapping[str, str]) -> SuiteFacts:
    suite = SuiteFacts()
    for path, source in sources.items():
        if not path.startswith("tests/") or not path.endswith(".py"):
            continue
        tests, helpers, assertions = _tests(source, path)
        suite.test_counts[path] = len(tests)
        suite.assertion_counts[path] = assertions
        for helper in helpers:
            suite.helpers[(path, helper.qualname)] = helper
        for test in tests:
            suite.tests[(path, test.qualname)] = test
    return suite


def _test_matches(base: SuiteFacts, head: SuiteFacts) -> list[tuple[TestFacts, TestFacts]]:
    matches: list[tuple[TestFacts, TestFacts]] = []
    remaining = dict(head.tests)
    missing: list[TestFacts] = []
    for key, old in base.tests.items():
        new = remaining.pop(key, None)
        if new is None:
            missing.append(old)
        else:
            matches.append((old, new))
    gained_paths = {
        path for path, count in head.test_counts.items() if count > base.test_counts.get(path, 0)
    }
    for old in missing:
        candidates = [
            (key, candidate)
            for key, candidate in remaining.items()
            if key[0] in gained_paths and candidate.qualname == old.qualname
        ]
        if candidates:
            key, new = candidates[0]
            remaining.pop(key)
            matches.append((old, new))
    return matches


def _tolerance_findings(old: TestFacts, new: TestFacts) -> list[Finding]:
    findings: list[Finding] = []
    old_keys = set(old.tolerances)
    new_keys = set(new.tolerances)
    unmatched_old = {
        key[0] for key in old_keys if len(old.tolerances[key]) > len(new.tolerances.get(key, ()))
    }
    for key, before_values in old.tolerances.items():
        after_values = new.tolerances.get(key, ())
        if not after_values:
            findings.append(
                Finding("tolerance-loosened", new.path, new.line, f"unmatched {key[0]} tolerance")
            )
            continue
        if Counter(after_values) >= Counter(before_values):
            continue
        before_sorted = sorted(before_values, key=_tolerance_sort)
        after_sorted = sorted(after_values, key=_tolerance_sort)
        offset = max(0, len(after_sorted) - len(before_sorted))
        for index, before in enumerate(before_sorted):
            match_index = offset + index
            if match_index >= len(after_sorted):
                findings.append(
                    Finding(
                        "tolerance-loosened", new.path, new.line, f"unmatched {key[0]} tolerance"
                    )
                )
                continue
            after = after_sorted[match_index]
            if before != after and any(
                a != b if isinstance(a, str) or isinstance(b, str) else b > a
                for a, b in zip(before, after, strict=True)
            ):
                findings.append(
                    Finding(
                        "tolerance-loosened", new.path, new.line, f"{key[0]} tolerance increased"
                    )
                )
    for key in new_keys - old_keys:
        if key[0] in unmatched_old:
            findings.append(
                Finding("tolerance-loosened", new.path, new.line, f"replacement {key[0]} tolerance")
            )

    return findings


def _helper_matches(base: SuiteFacts, head: SuiteFacts) -> list[tuple[TestFacts, TestFacts]]:
    remaining = dict(head.helpers)
    matches: list[tuple[TestFacts, TestFacts]] = []
    missing: list[TestFacts] = []
    for key, old in base.helpers.items():
        new = remaining.pop(key, None)
        if new is None:
            missing.append(old)
        else:
            matches.append((old, new))
    for old in missing:
        candidates = [
            (key, candidate)
            for key, candidate in remaining.items()
            if candidate.qualname == old.qualname
        ]
        if len(candidates) == 1:
            key, new = candidates[0]
            remaining.pop(key)
            matches.append((old, new))
    return matches


def compare(base: SuiteFacts, head: SuiteFacts) -> list[Finding]:
    findings: list[Finding] = []
    removed = sum(base.test_counts.values()) - sum(head.test_counts.values())
    if removed > 0:
        findings.append(
            Finding("test-count-decreased", "tests", 1, f"{removed} test function(s) removed")
        )
    assertions_removed = sum(base.assertion_counts.values()) - sum(head.assertion_counts.values())
    if assertions_removed > 0:
        findings.append(
            Finding(
                "assertion-count-decreased",
                "tests",
                1,
                f"{assertions_removed} assertion(s) removed",
            )
        )
    for old, new in _test_matches(base, head):
        for added in sorted(new.skips - old.skips):
            kind = "xfail-added" if ":xfail:" in added or added.endswith(":xfail") else "skip-added"
            findings.append(Finding(kind, new.path, new.line, added))
        if old.examples is not None and (new.examples is None or new.examples < old.examples):
            findings.append(
                Finding(
                    "hypothesis-examples-reduced",
                    new.path,
                    new.line,
                    f"{old.examples} -> {new.examples}",
                    (new.qualname, "hypothesis-examples"),
                )
            )
        if old.timeout is not None and (new.timeout is None or new.timeout > old.timeout):
            findings.append(
                Finding(
                    "timeout-raised",
                    new.path,
                    new.line,
                    f"{old.timeout} -> {new.timeout}",
                    (new.qualname, "timeout"),
                )
            )
        findings.extend(_tolerance_findings(old, new))
    for old, new in _helper_matches(base, head):
        findings.extend(_tolerance_findings(old, new))
    return findings


def _git(repo: str, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", repo, *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _changed_entries(repo: str, left: str, right: str) -> list[tuple[str, str, str]]:
    output = _git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "--raw",
        "-r",
        "--no-renames",
        "--abbrev=40",
        left,
        right,
        "--",
        "tests",
    )
    entries: list[tuple[str, str, str]] = []
    for row in output.splitlines():
        header, path = row.split("\t", 1)
        fields = header.split()
        old_oid, new_oid = fields[2], fields[3]
        if path.endswith(".py") and path.startswith("tests/"):
            entries.append((path, old_oid, new_oid))
    return entries


def _read_blobs(repo: str, oids: set[str], cache: dict[str, str]) -> None:
    missing = sorted(oid for oid in oids if oid.strip("0") and oid not in cache)
    if not missing:
        return
    result = subprocess.run(
        ["git", "-C", repo, "cat-file", "--batch"],
        input=("\n".join(missing) + "\n").encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    ).stdout
    offset = 0
    while offset < len(result):
        end = result.index(b"\n", offset)
        oid, kind, size_text = result[offset:end].decode().split()
        size = int(size_text)
        start = end + 1
        cache[oid] = result[start : start + size].decode("utf-8")
        offset = start + size + 1


def _changed_sources(
    repo: str, left: str, right: str, cache: dict[str, str]
) -> tuple[dict[str, str], dict[str, str]]:
    entries = _changed_entries(repo, left, right)
    oids = {oid for _, old, new in entries for oid in (old, new) if oid.strip("0")}
    _read_blobs(repo, oids, cache)
    before = {path: cache[old] for path, old, _ in entries if old.strip("0")}
    after = {path: cache[new] for path, _, new in entries if new.strip("0")}
    return before, after


def _working_sources(repo: str, cache: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    raw = _git(repo, "diff", "--raw", "--no-renames", "--abbrev=40", "HEAD", "--", "tests")
    entries: list[tuple[str, str]] = []
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for row in raw.splitlines():
        header, path = row.split("\t", 1)
        fields = header.split()
        old_oid = fields[2]
        if path.startswith("tests/") and path.endswith(".py"):
            if old_oid.strip("0"):
                entries.append((path, old_oid))
            file = Path(repo) / path
            if file.is_file():
                after[path] = file.read_text()
    _read_blobs(repo, {oid for _, oid in entries}, cache)
    before.update({path: cache[oid] for path, oid in entries})
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "--", "tests").splitlines()
    for path in untracked:
        if path.startswith("tests/") and path.endswith(".py"):
            after[path] = (Path(repo) / path).read_text()
    return before, after


def _parent_findings(repo: str, commit: str, cache: dict[str, str]) -> list[Finding]:
    parents = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]
    if not parents:
        return []
    parent_sets = []
    for parent in parents:
        before, after = _changed_sources(repo, parent, commit, cache)
        parent_sets.append(compare(collect(before), collect(after)))
    if len(parents) == 1:
        return parent_sets[0]

    def identity(finding: Finding) -> tuple[str, ...]:
        common = (finding.kind, finding.path, str(finding.line))
        if finding.identity:
            return (*common, *finding.identity)
        if finding.kind in {"test-count-decreased", "assertion-count-decreased"}:
            return common

        return (*common, finding.detail)

    fingerprints = [{identity(finding) for finding in items} for items in parent_sets]
    common = set.intersection(*fingerprints) if fingerprints else set()
    return [finding for finding in parent_sets[0] if identity(finding) in common]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--ci", action="store_true")
    parser.add_argument("--approved", action="store_true")
    try:
        args = parser.parse_args(argv)
        _git(args.repo, "rev-parse", "--verify", f"{args.base}^{{commit}}")
        commits = _git(args.repo, "rev-list", "--reverse", f"{args.base}..HEAD").splitlines()
        cache: dict[str, str] = {}
        unapproved: list[tuple[str, Finding]] = []
        for commit in commits:
            findings = _parent_findings(args.repo, commit, cache)
            if args.ci:
                unapproved.extend((commit, finding) for finding in findings)
                continue
            trailers = _git(
                args.repo,
                "log",
                "-1",
                "--format=%(trailers:key=Test-Reduction-Approved)",
                commit,
            )
            if findings and not trailers:
                unapproved.extend((commit, finding) for finding in findings)
        before, after = _working_sources(args.repo, cache)
        unapproved.extend(
            ("worktree", finding) for finding in compare(collect(before), collect(after))
        )
    except (subprocess.CalledProcessError, OSError, ValueError):
        print("unknown base or invalid repository", file=sys.stderr)
        return 2
    if unapproved:
        for revision, finding in unapproved:
            print(f"{revision} {finding.path}:{finding.line} {finding.kind} {finding.detail}")
        if args.ci and args.approved:
            return 0
        if args.ci:
            message = (
                "removing or weakening tests needs owner approval; ask the PR owner "
                "to apply the test-reduction-approved label"
            )
        else:
            message = (
                "removing or weakening tests needs owner approval; restore the test or "
                "record the owner's approval as a Test-Reduction-Approved trailer on "
                "the reducing commit"
            )
        print(message)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
