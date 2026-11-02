"""Mechanical checks for user-facing vocabulary and the promised public API."""

from __future__ import annotations

import ast
import importlib
import inspect
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import FunctionType
from typing import Protocol

import pytest
from sybil import Document

from conftest import PythonOutputBlockParser
from increment.errors import CodedError

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PARAMETERS = ROOT / "tests" / "public_parameters.txt"
EXPORTS = ROOT / "tests" / "public_exports.txt"
TERMS = ROOT / "tests" / "consumer_jargon.txt"
BASELINE = ROOT / "tests" / "consumer_jargon_baseline.txt"
PROMISED_NAMESPACES = {
    "increment",
    "increment.artifacts",
    "increment.dashboard",
    "increment.digest",
    "increment.errors",
    "increment.estimators",
    "increment.evidence",
    "increment.power",
    "increment.query",
    "increment.results",
    "increment.simulate",
    "increment.store",
    "increment.tables",
    "increment.workflows",
}


def _manifest(path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    entries: dict[str, str] | None = None
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            namespace = line[1:-1]
            if namespace in sections:
                raise AssertionError(f"{path}:{number}: duplicate namespace {namespace!r}")
            entries = sections[namespace] = {}
            continue
        name, separator, reason = line.partition(":")
        if entries is None or not separator or not name.strip() or not reason.strip():
            raise AssertionError(f"{path}:{number}: expected [namespace], then name: user task")
        if name.strip() in entries:
            raise AssertionError(f"{path}:{number}: duplicate export {name.strip()!r}")
        entries[name.strip()] = reason.strip()
    return sections


@pytest.mark.integration
def test_each_promised_namespace_exports_exactly_its_manifest() -> None:
    manifest = _manifest(EXPORTS)
    assert manifest.keys() == PROMISED_NAMESPACES
    differences: dict[str, dict[str, list[str]]] = {}
    for namespace, entries in manifest.items():
        exports: set[str] = set(getattr(importlib.import_module(namespace), "__all__", ()))
        unlisted = sorted(exports - entries.keys())
        unexported = sorted(entries.keys() - exports)
        if unlisted or unexported:
            differences[namespace] = {"unlisted": unlisted, "unexported": unexported}
    assert not differences, (
        "Exports differ from tests/public_exports.txt; give each promised name a "
        f"user task or keep it out of __all__: {differences}"
    )
    task_terms = {
        "Observational": {"non-randomized", "comparison"},
        "Encouragement": {"random encouragement", "uptake"},
    }
    for name, terms in task_terms.items():
        justification = manifest["increment"][name].lower()
        assert all(term in justification for term in terms), (
            f"{name} justification should describe its documented user task"
        )


def _documentation_python(path: Path) -> Iterator[tuple[int, str]]:
    if path.suffix == ".py":
        yield 1, path.read_text()
        return
    text = path.read_text()
    for region in PythonOutputBlockParser()(Document(text, str(path))):
        yield text.count("\n", 0, region.start) + 2, str(region.parsed)


def _increment_references(source: str) -> Iterator[tuple[int, str]]:
    """Yield every dotted ``increment`` path a snippet imports or reads from an import."""
    tree = ast.parse(source)
    bound: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "increment":
                    yield node.lineno, alias.name
                    bound[alias.asname or "increment"] = alias.name if alias.asname else "increment"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "increment":
                for alias in node.names:
                    yield node.lineno, f"{node.module}.{alias.name}"
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def dotted(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return bound.get(node.id)
        if isinstance(node, ast.Attribute):
            base = dotted(node.value)
            return None if base is None else f"{base}.{node.attr}"
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (path := dotted(node)) is not None:
            yield node.lineno, path


def _is_promised(path: str, manifest: dict[str, dict[str, str]]) -> bool:
    parts = path.split(".")
    for end in range(len(parts), 0, -1):
        namespace = ".".join(parts[:end])
        if namespace in manifest:
            return end == len(parts) or parts[end] in manifest[namespace]
    return False


@pytest.mark.integration
def test_documentation_imports_only_promised_names() -> None:
    manifest = _manifest(EXPORTS)
    documents = [
        ROOT / "README.md",
        *sorted((ROOT / "docs").rglob("*.md")),
        *sorted((ROOT / "examples").glob("*.py")),
    ]
    unpromised: list[str] = []
    for document in documents:
        relative = document.relative_to(ROOT)
        for start, source in _documentation_python(document):
            for line, path in _increment_references(source):
                if not _is_promised(path, manifest):
                    unpromised.append(f"{relative}:{start + line - 1}: {path}")
        if document.suffix == ".md":
            for number, raw in enumerate(document.read_text().splitlines(), 1):
                directive = re.match(r"\s*::: (\S+)", raw)
                if directive and not _is_promised(directive.group(1), manifest):
                    unpromised.append(f"{relative}:{number}: {directive.group(1)}")
    assert not unpromised, (
        "Documentation teaches names outside tests/public_exports.txt; import them "
        "from their promised namespace:\n" + "\n".join(unpromised)
    )


def _display_string_nodes(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    return [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def _registry_template_nodes(
    value: ast.AST, source: Path, module_tree: ast.Module | None = None
) -> list[ast.AST]:
    tree = (
        module_tree
        if module_tree is not None
        else ast.parse(source.read_text(), filename=str(source))
    )
    bindings: dict[str, ast.AST] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is not None:
                bindings[statement.target.id] = statement.value
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bindings[statement.name] = statement

    def collect(node: ast.AST, seen: set[str]) -> list[ast.AST]:
        if isinstance(node, ast.Dict):
            return [item for value in node.values for item in collect(value, seen)]
        if isinstance(node, ast.Tuple):
            return collect(node.elts[0], seen) if node.elts else []
        if isinstance(node, ast.Name):
            if node.id in seen or node.id not in bindings:
                return []
            return collect(bindings[node.id], {*seen, node.id})
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            call_name = node.func.id
            if call_name not in {"RefusalSpec", "WarningSpec"} and not (
                call_name.startswith("_") and call_name.endswith("register_warning")
            ):
                return []
            renderer = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "render"),
                node.args[2] if len(node.args) > 2 else None,
            )
            if renderer is None and call_name == "RefusalSpec":
                template = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "template"),
                    node.args[2] if len(node.args) > 2 else None,
                )
                if template is not None and not (
                    isinstance(template, ast.Constant) and template.value is None
                ):
                    return collect(template, seen)
                renderer = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "render"),
                    node.args[3] if len(node.args) > 3 else None,
                )
            return collect(renderer, seen) if renderer is not None else []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return _display_string_nodes(node)
        if isinstance(node, ast.Lambda):
            return [
                item
                for item in ast.walk(node.body)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            ]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node]
        return []

    return collect(value, set())


def _user_facing_python_strings(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[tuple[int, str]] = []
    seen: set[int] = set()

    def add(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in seen:
            seen.add(id(node))
            found.append((node.lineno, node.value))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = (
            node.func.id
            if isinstance(node.func, ast.Name)
            else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
        )
        if call_name in {"RefusalSpec", "WarningSpec"} or (
            call_name.startswith("_") and call_name.endswith("register_warning")
        ):
            for item in _registry_template_nodes(node, path, tree):
                add(item)
        elif (
            call_name
            in {
                "CodedError",
                "CapabilityError",
                "InvalidRequestError",
                "UnsupportedRequestError",
                "WireFormatError",
            }
            and node.args
        ):
            for item in ast.walk(node.args[0]):
                add(item)
        elif call_name == "refusals" and len(node.args) > 1:
            for item in _registry_template_nodes(node.args[1], path, tree):
                add(item)

    display_names = {
        "__repr__",
        "__str__",
        "_repr_html_",
        "_repr_markdown_",
        "render_readout_text",
        "_render_group_label",
        "_render_group_scope",
        "_render_group_rows",
        "render_readout_qualifications",
        "_selection_qualification",
        "render_effect_interpretation",
        "_render_effect",
        "_render_confidence_geometry",
        "_render_test_null",
        "_readout_metadata_notes",
        "_readout_evidence_notes",
        "_integrity_notes",
        "_row_qualification_notes",
        "_roster_notes",
        "_family_notes",
        "_advisory_notes",
        "_cell_evidence_notes",
        "_row_qualification_columns",
        "_binomial_qualification_disclosure",
        "_compact_interval_interpretation",
        "_readout_labels",
        "_group_disclosure_column",
        "_interval_level_note",
        "_allocation_warnings",
        "_report_notes",
        "_notes",
        "_lift_notes",
        "overview_notes",
        "results_table",
        "results_notes",
        "results_warnings",
        "_selection_qualification_notes",
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in display_names:
            for item in _display_string_nodes(node):
                add(item)
    return found


def test_refusals_registry_scans_all_template_value_forms(tmp_path: Path) -> None:
    source = tmp_path / "registry.py"
    source.write_text(
        "def refusals(error_type, entries):\n"
        "    return entries\n"
        "def RefusalSpec(code, error_type, template=None):\n"
        "    return template\n"
        '_ALIAS = "Aliased ingress template."\n'
        '_PREBUILT = RefusalSpec("example.prebuilt", Exception, template="Prebuilt sidecar template.")\n'
        '_RENDERED = RefusalSpec("example.rendered", Exception, render=lambda **_: "Rendered ingress response.")\n'
        "_REGISTRY = refusals(Exception, {\n"
        '    "literal": "Literal ingress template.",\n'
        '    "tuple": ("Tuple seam template.", ("extra",)),\n'
        '    "spec": RefusalSpec("example.spec", Exception, template="Spec sidecar template."),\n'
        '    "alias": _ALIAS,\n'
        '    "prebuilt_alias": _PREBUILT,\n'
        '    "rendered_alias": _RENDERED,\n'
        "})\n"
    )
    found = {text for _, text in _user_facing_python_strings(source)}
    assert {
        "Literal ingress template.",
        "Tuple seam template.",
        "Spec sidecar template.",
        "Aliased ingress template.",
        "Prebuilt sidecar template.",
        "Rendered ingress response.",
    } <= found


def test_representation_scanner_excludes_docstrings_but_scans_body_literals(tmp_path: Path) -> None:
    source = tmp_path / "display.py"
    source.write_text(
        "def __repr__(self):\n"
        '    """Explain retained family behavior."""\n'
        '    note = "retained family internal note"\n'
        '    return "Shown to users"\n'
    )
    strings = _user_facing_python_strings(source)
    assert (2, "Explain retained family behavior.") not in strings
    assert (3, "retained family internal note") in strings
    assert (4, "Shown to users") in strings


def test_display_helper_strings_are_scanned(tmp_path: Path) -> None:
    source = tmp_path / "display.py"
    source.write_text(
        "def _render_group_scope():\n"
        "    lines = []\n"
        '    lines.append("Population roster completeness: complete.")\n'
        "    return lines\n"
    )
    assert _user_facing_python_strings(source) == [(3, "Population roster completeness: complete.")]


def test_display_scanner_follows_strings_in_nested_helpers(tmp_path: Path) -> None:
    source = tmp_path / "display.py"
    source.write_text(
        "def render_readout_qualifications():\n"
        "    lines = []\n"
        "    def add_bullet(text):\n"
        "        lines.append(text)\n"
        '    add_bullet("retained family")\n'
        "    return lines\n"
    )
    found = _user_facing_python_strings(source)
    assert any(text == "retained family" for _, text in found)


def test_display_scanner_includes_augmented_string_assignments(tmp_path: Path) -> None:
    source = tmp_path / "display.py"
    source.write_text(
        'def __repr__(self):\n    text = "Shown"\n    text += " retained family"\n    return text\n'
    )
    found = _user_facing_python_strings(source)
    assert any(text == " retained family" for _, text in found)


def test_refusals_registry_templates_are_scanned(tmp_path: Path) -> None:
    source = tmp_path / "registry.py"
    source.write_text(
        "def refusals(error_type, entries):\n"
        "    return entries\n"
        "_REFUSALS = refusals(Exception, {\n"
        '    "example.code": "The ingress is not available."\n'
        "})\n"
    )
    assert _user_facing_python_strings(source) == [(4, "The ingress is not available.")]


def test_warning_registration_renderers_are_scanned(tmp_path: Path) -> None:
    source = tmp_path / "warnings.py"
    source.write_text(
        "class WarningSpec:\n"
        "    def __init__(self, code, warning_type, render):\n"
        "        self.render = render\n"
        "def _register_warning(code, warning_type, render):\n"
        "    return WarningSpec(code, warning_type, render)\n"
        "_WARNINGS = {}\n"
        "_register_warning(\n"
        '    "example.warning", Warning, lambda: "A retained family warning."\n'
        ")\n"
        '_DIRECT = WarningSpec("example.direct", Warning, lambda: "A sidecar warning.")\n'
    )
    found = {text for _, text in _user_facing_python_strings(source)}
    assert "A retained family warning." in found
    assert "A sidecar warning." in found


def test_table_note_helper_literals_are_scanned(tmp_path: Path) -> None:
    source = tmp_path / "tables.py"
    source.write_text(
        'def _roster_notes(frame):\n    return ["Population roster completeness is unknown."]\n'
    )
    assert _user_facing_python_strings(source) == [
        (2, "Population roster completeness is unknown.")
    ]


def _term_config(path: Path) -> dict[str, tuple[str, ...]]:
    terms: dict[str, tuple[str, ...]] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        term = fields[0].strip()
        if not term or term in terms:
            raise AssertionError(f"{path}:{number}: empty or duplicate jargon term")
        terms[term] = tuple(field.strip() for field in fields[1:] if field.strip())
    return terms


def _jargon_occurrences(root: Path, terms: dict[str, tuple[str, ...]]) -> Counter[tuple[str, str]]:
    for term, definition_files in terms.items():
        for file in definition_files:
            source = root / file
            assert source.is_file(), f"{TERMS}:{term}: definition file does not exist: {file}"
            assert re.search(rf"(?<!\w){re.escape(term)}(?!\w)", source.read_text(), re.I), (
                f"{TERMS}:{term}: definition file does not contain the term: {file}"
            )

    surfaces: dict[str, list[tuple[int, str]]] = {}
    for source in sorted((root / "increment").rglob("*.py")):
        surfaces[source.relative_to(root).as_posix()] = _user_facing_python_strings(source)
    for source in (root / "README.md", *sorted((root / "docs" / "guides").rglob("*.md"))):
        if source.exists():
            surfaces[source.relative_to(root).as_posix()] = list(
                enumerate(source.read_text().splitlines(), 1)
            )

    occurrence_counts: Counter[tuple[str, str]] = Counter()
    for term, definition_files in terms.items():
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        for file, lines in surfaces.items():
            if file in definition_files:
                continue
            count = sum(len(pattern.findall(text)) for _, text in lines)
            if count:
                occurrence_counts[(term, file)] = count
    return occurrence_counts


def _read_baseline(path: Path) -> Counter[tuple[str, str]]:
    baseline: Counter[tuple[str, str]] = Counter()
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split("\t")
        if len(fields) != 3:
            raise AssertionError(f"{path}:{number}: expected term, file, count")
        term, file, count = fields
        baseline[(term, file)] = int(count)
    return baseline


def test_internal_jargon_occurrences_do_not_exceed_the_baseline() -> None:
    terms = _term_config(TERMS)
    current = _jargon_occurrences(ROOT, terms)
    baseline = _read_baseline(BASELINE)
    allowed_terms = set(terms)
    assert {term for term, _ in baseline} <= allowed_terms, (
        "remove obsolete entries from the jargon baseline"
    )

    findings: list[str] = []
    stale: list[str] = []
    surfaces = sorted({file for _, file in current} | {file for _, file in baseline})
    for term in sorted(terms):
        for file in surfaces:
            observed = current[(term, file)]
            allowed = baseline[(term, file)]
            if observed > allowed:
                source = ROOT / file
                lines = [
                    number
                    for number, text in _surface_lines(source, file)
                    for _ in re.finditer(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.I)
                ]
                display = ",".join(map(str, lines))
                findings.append(
                    f"{file}:{display}: {term!r} (+{observed - allowed}); "
                    "say it in analyst language"
                )
            elif observed < allowed:
                stale.append(
                    f"{file}: {term!r} baseline {allowed}, observed {observed}; "
                    "lower or remove the stale baseline entry"
                )
    assert not findings, "New internal vocabulary on user-facing surfaces:\n" + "\n".join(findings)
    assert not stale, "Stale internal-vocabulary baseline entries:\n" + "\n".join(stale)


def _surface_lines(source: Path, relative: str) -> list[tuple[int, str]]:
    if relative.endswith(".py"):
        return _user_facing_python_strings(source)
    return list(enumerate(source.read_text().splitlines(), 1))


class _ExportingModule(Protocol):
    @property
    def __all__(self) -> Sequence[str]: ...


def _unwrapped_function(value: object) -> FunctionType | None:
    if isinstance(value, (classmethod, staticmethod)):
        value = value.__func__
    if isinstance(value, property) or inspect.isclass(value) or not callable(value):
        return None
    unwrapped = inspect.unwrap(value)
    if inspect.isclass(unwrapped) or not inspect.isfunction(unwrapped):
        return None
    return unwrapped


def _root_export_functions(module: _ExportingModule) -> list[tuple[str, FunctionType]]:
    functions: list[tuple[str, FunctionType]] = []
    for name in module.__all__:
        function = _unwrapped_function(getattr(module, name))
        if function is not None and function.__module__.startswith("increment"):
            functions.append((name, function))
    return functions


def _frame_factory_functions(module: _ExportingModule) -> list[tuple[str, FunctionType]]:
    functions: list[tuple[str, FunctionType]] = []
    for name in module.__all__:
        if not name.startswith("from_"):
            continue
        function = _unwrapped_function(getattr(module, name))
        if function is not None:
            functions.append((name, function))
    return functions


# Source operations that replaced the removed ``Analysis`` members.
_SOURCE_OPERATIONS = {
    "estimators": (
        "estimate_cate",
        "validate_cate",
        "targeting_rule",
        "select_targeting_rule",
        "sitewide",
    ),
    "query": ("build_panel_for_metric", "panel_sql", "summary_sql"),
    "artifacts": ("artifact_context", "publish_unit_day_artifact"),
    "dashboard": ("prepare_dashboard", "load_explore"),
}


def _public_callable_parameters() -> dict[str, set[str]]:
    import importlib

    import increment
    from increment import frame, workflows

    callables = [
        (f"increment.{name}", function) for name, function in _root_export_functions(increment)
    ]
    callables.extend(
        (f"workflows.{name}", function) for name, function in _root_export_functions(workflows)
    )
    for module_name, names in _SOURCE_OPERATIONS.items():
        module = importlib.import_module(f"increment.{module_name}")
        for name in names:
            function = _unwrapped_function(getattr(module, name))
            assert function is not None, f"increment.{module_name}.{name}"
            callables.append((f"{module_name}.{name}", function))
    callables.extend(
        (f"frame.{name}", function) for name, function in _frame_factory_functions(frame)
    )

    parameters: dict[str, set[str]] = {}
    for location, function in callables:
        for index, name in enumerate(inspect.signature(function).parameters):
            if index == 0 and name in {"self", "cls"}:
                continue
            parameters.setdefault(name, set()).add(location)
    return parameters


def _public_parameter_vocabulary(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    allowed: dict[str, str] = {}
    banned: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        names, separator, meaning = raw.partition(":")
        if not separator or not meaning.strip():
            raise AssertionError(f"{path}:{number}: expected name/name: one-line meaning")
        target = banned if names.startswith("!") else allowed
        for name in names.removeprefix("!").split("/"):
            name = name.strip()
            if not name or name in allowed or name in banned:
                raise AssertionError(f"{path}:{number}: empty or duplicate parameter {name!r}")
            target[name] = meaning.strip()
    return allowed, banned


def test_public_parameter_vocabulary_matches_the_ratchet() -> None:
    parameters = _public_callable_parameters()
    allowed, banned = _public_parameter_vocabulary(PUBLIC_PARAMETERS)
    disallowed = sorted(parameters.keys() & banned.keys())
    missing = sorted(parameters.keys() - allowed.keys())
    assert not disallowed and not missing, (
        "Public parameter vocabulary changed: "
        f"banned={[(name, sorted(parameters[name])) for name in disallowed]}, "
        f"missing={[(name, sorted(parameters[name])) for name in missing]}"
    )


def test_public_parameter_ratchet_discovers_decorated_root_exports() -> None:
    from functools import cache
    from types import SimpleNamespace

    def new_public_function(unregistered_root_parameter):
        return unregistered_root_parameter

    new_public_function.__module__ = "increment"
    module = SimpleNamespace(
        __all__=("new_public_function",),
        new_public_function=cache(new_public_function),
    )
    functions = _root_export_functions(module)
    assert "unregistered_root_parameter" in inspect.signature(functions[0][1]).parameters


def test_public_parameter_ratchet_discovers_new_frame_factories() -> None:
    from types import SimpleNamespace

    def from_new_source(unregistered_factory_parameter):
        return unregistered_factory_parameter

    module = SimpleNamespace(
        __all__=("from_new_source",),
        from_new_source=from_new_source,
    )
    functions = _frame_factory_functions(module)
    assert "unregistered_factory_parameter" in inspect.signature(functions[0][1]).parameters


@pytest.mark.parametrize(
    ("case", "code", "invoke"),
    [
        (
            "unknown model field",
            "model.field.unknown",
            lambda: importlib.import_module("increment").Randomized(
                control_group="c",
                allocation_schem={"c": 0.5},  # ty: ignore[unknown-argument]
            ),
        ),
        (
            "list frame",
            "source.frame.input_type",
            lambda: importlib.import_module("increment").from_frame(
                [],  # ty: ignore[invalid-argument-type]
                shape="summary",
                unit="unit",
                group="arm",
                control="c",
                metrics={"y": "mean"},
            ),
        ),
        (
            "lazy frame",
            "source.frame.input_type",
            lambda: importlib.import_module("increment").from_frame(
                importlib.import_module("polars")
                .DataFrame({"unit": ["u"], "arm": ["c"], "y": [1]})
                .lazy(),  # ty: ignore[invalid-argument-type]
                shape="summary",
                unit="unit",
                group="arm",
                control="c",
                metrics={"y": "mean"},
            ),
        ),
        (
            "metric type required",
            "frame.metrics.type_required",
            lambda: importlib.import_module("increment").from_frame(
                importlib.import_module("pyarrow").table({"unit": ["u"], "arm": ["c"], "y": [1]}),
                shape="summary",
                unit="unit",
                group="arm",
                control="c",
                metrics=["missing"],
            ),
        ),
        (
            "CUPED covariate on a text column",
            "frame.validation.covariate_not_numeric",
            lambda: importlib.import_module("increment").from_frame(
                importlib.import_module("pyarrow").table(
                    {"unit": ["u1", "u2"], "arm": ["c", "t"], "y": [1.0, 2.0], "pre": ["a", "b"]}
                ),
                shape="summary",
                unit="unit",
                group="arm",
                control="c",
                metrics=[
                    importlib.import_module("increment").MetricSpec(
                        name="y",
                        type="mean",
                        decision_method=importlib.import_module("increment").Method.cuped(
                            covariates="pre"
                        ),
                    )
                ],
            ),
        ),
    ],
)
def test_public_bad_inputs_raise_registered_codes(case, code, invoke) -> None:
    from tests.refusal_registry import walk_refusal_registries

    assert code in walk_refusal_registries(), f"{case}: expected code must be registered"
    with pytest.raises(CodedError) as raised:
        invoke()
    assert raised.value.code == code
