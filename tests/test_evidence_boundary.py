from __future__ import annotations

import ast
import re
from collections import Counter
from pathlib import Path

import pytest

EVIDENCE_METHODS = frozenset({"unit_frame", "moments"})
_SOURCE_NAMES = frozenset({"src", "source"})
_ALLOWED = {
    ("adjust.py", "estimate_ate"),
    ("adjust.py", "observational_evidence"),
    ("adjust.py", "bind_unit_frame_memo.read"),
    ("quantile.py", "estimate_quantile_lift"),
    ("quantile.py", "_computation"),
    ("winsor.py", "raw_state_from_source"),
}


def test_estimation_reads_sources_only_through_facades():
    root = Path(__file__).parents[1] / "increment" / "estimation"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        def owner(node: ast.AST, parents=parents) -> str | None:
            names = []
            current = parents.get(node)
            while current is not None:
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.append(current.name)
                current = parents.get(current)
            return ".".join(reversed(names)) or None

        rel = path.relative_to(root)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            receiver_call = (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id in _SOURCE_NAMES
                and func.attr in EVIDENCE_METHODS
            )
            receiver_named_read = isinstance(func, ast.Attribute) and func.attr in EVIDENCE_METHODS
            if rel.parts[0] == "_adjust" and receiver_named_read:
                violations.append(f"{rel}:{node.lineno}: adjustment kernel calls .{func.attr}()")
            if receiver_call and (rel.name, owner(node)) not in _ALLOWED:
                violations.append(f"{rel}:{node.lineno}: source.{func.attr}() outside facade")
            source_probe = (
                isinstance(func, ast.Name)
                and func.id in {"isinstance", "getattr", "hasattr"}
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id in _SOURCE_NAMES
            )
            if source_probe and (rel.name, owner(node)) not in _ALLOWED:
                violations.append(f"{rel}:{node.lineno}: source probing outside facade")
    assert not violations, "\n".join(violations)


# Scan every production module; narrowing the tree would miss forbidden acquisition routes.
@pytest.mark.slow
def test_loader_routes_stay_inside_acquisition_and_facades():  # noqa: C901
    root = Path(__file__).parents[1] / "increment"
    # The evidence layer plus the readout routes that request evidence.
    allowed = {
        Path("readouts/_evidence.py"),
        Path("_day_axis.py"),
        Path("readouts/_whole_window.py"),
        Path("_breakout_readouts.py"),
        Path("_sitewide.py"),
    }
    route_aliases = {"BreakoutRoute", "DayAxisRoute"}
    evidence_types = ast.parse((root / "_evidence_types.py").read_text())
    route_names = {
        value.value
        for node in evidence_types.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in route_aliases for target in node.targets
        )
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.slice, ast.Tuple)
        for value in node.value.slice.elts
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
    }
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        docstrings = {
            node.body[0].value
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        functions = {
            node.name: (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def has_route_name(node: ast.AST) -> bool:
            return any(
                (isinstance(child, ast.Name) and "route" in child.id.lower())
                or (isinstance(child, ast.Attribute) and "route" in child.attr.lower())
                for child in ast.walk(node)
            )

        def uses_route_alias(node: ast.AST | None) -> bool:
            return node is not None and any(
                isinstance(child, ast.Name) and child.id in route_aliases
                for child in ast.walk(node)
            )

        def literals(
            node: ast.AST, *, docstrings=docstrings, route_names=route_names
        ) -> list[ast.Constant]:
            return [
                child
                for child in ast.walk(node)
                if isinstance(child, ast.Constant)
                and isinstance(child.value, str)
                and child.value in route_names
                and child not in docstrings
            ]

        routed_literals: set[ast.Constant] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in route_aliases:
                if rel not in allowed and rel != Path("_evidence_types.py"):
                    violations.append(
                        f"{rel}:{node.lineno}: loader route type outside acquisition/facade"
                    )
            if isinstance(node, ast.Compare) and has_route_name(node):
                routed_literals.update(literals(node))
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "route":
                        routed_literals.update(literals(keyword.value))
                for index, argument in enumerate(node.args):
                    if isinstance(node.func, ast.Name):
                        parameters = functions.get(node.func.id, ())
                        if index < len(parameters) and (
                            "route" in parameters[index].arg.lower()
                            or uses_route_alias(parameters[index].annotation)
                        ):
                            routed_literals.update(literals(argument))
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                annotation = node.annotation if isinstance(node, ast.AnnAssign) else None
                is_route_alias_definition = rel == Path("_evidence_types.py") and any(
                    isinstance(target, ast.Name) and target.id in route_aliases
                    for target in targets
                )
                if not is_route_alias_definition and (
                    any(has_route_name(target) for target in targets)
                    or uses_route_alias(annotation)
                ):
                    value = node.value
                    if value is not None:
                        routed_literals.update(literals(value))
            if isinstance(node, ast.Return):
                parent = parents.get(node)
                while parent is not None and not isinstance(
                    parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
                ):
                    parent = parents.get(parent)
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_route_alias(
                    parent.returns
                ):
                    if node.value is not None:
                        routed_literals.update(literals(node.value))

        if rel not in allowed:
            violations.extend(
                f"{rel}:{node.lineno}: route literal outside acquisition/facade"
                for node in sorted(routed_literals, key=lambda item: item.lineno)
            )
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "classify_source":
                if rel not in allowed:
                    violations.append(
                        f"{rel}:{node.lineno}: classify_source outside acquisition/facade"
                    )
            if isinstance(node, ast.Attribute) and node.attr == "classify_source":
                if rel not in allowed:
                    violations.append(
                        f"{rel}:{node.lineno}: classify_source outside acquisition/facade"
                    )
    assert not violations, "\n".join(violations)


_LEGACY_EVIDENCE_METHODS = frozenset(
    {
        "assignment_counts",
        "breakout_moments",
        "breakout_moments_for_method",
        "breakout_source",
        "breakout_sources",
        "categorical_evidence",
        "cluster_counts",
        "compliance_dates",
        "compliance_summary",
        "day_source",
        "fit_cuped_multicovariate",
        "missing_value_account",
        "moments",
        "moments_for_method",
        "readout_execution",
        "readout_snapshot",
        "triggered_counts",
        "triggered_source",
        "unit_counts",
        "unit_frame",
        "validate_populations",
    }
)


def _is_source_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _SOURCE_NAMES
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in {"_src", "_source"}
    )


def test_consumers_read_sources_only_through_evidence_requests():
    """Readouts, breakouts and planning ask the evidence layer; they never probe a source."""
    root = Path(__file__).parents[1] / "increment"
    consumers = [
        path
        for path in sorted(root.rglob("*.py"))
        if (
            path.parent == root / "readouts"
            and path.name not in {"_evidence.py", "_whole_window.py"}
        )
        or path.is_relative_to(root / "breakout")
        or path.is_relative_to(root / "dashboard")
        or path.relative_to(root)
        in {
            Path("_day_axis.py"),
            Path("_readout_request.py"),
            Path("_planning_baseline.py"),
            Path("cate.py"),
        }
    ]
    violations: list[str] = []
    for path in consumers:
        rel = path.relative_to(root)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "src"
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "source"
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "evidence"
            ):
                violations.append(f"{rel}:{node.lineno}: private day-axis adapter reach-in")
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _LEGACY_EVIDENCE_METHODS
                and _is_source_expr(func.value)
            ):
                violations.append(f"{rel}:{node.lineno}: source.{func.attr}() outside acquisition")
            if (
                isinstance(func, ast.Name)
                and func.id in {"getattr", "hasattr"}
                and len(node.args) >= 2
                and _is_source_expr(node.args[0])
                and isinstance(node.args[1], ast.Constant)
            ):
                violations.append(
                    f"{rel}:{node.lineno}: {func.id}() on a source outside acquisition"
                )
    assert not violations, "\n".join(violations)


def test_execution_context_reads_stream_inventory_from_evidence_scope() -> None:
    root = Path(__file__).parents[1] / "increment"
    path = root / "kernel" / "context.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    forbidden_attributes = {
        "_adjustment_fact_names",
        "_source_tables",
        "_defs",
        "_experiment",
        "_resolve_unit_covariate_source",
    }
    violations = [
        f"{path.relative_to(root)}:{node.lineno}: loader-owned {node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes
    ]
    violations.extend(
        f"{path.relative_to(root)}:{node.lineno}: loader-owned {node.value}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value in forbidden_attributes
    )
    violations.extend(
        f"{path.relative_to(root)}:{node.lineno}: loader-owned {node.id}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == "_adjustment_fact_names"
    )
    assert not violations, "\n".join(violations)


_INGRESS_BASELINE = Path(__file__).with_name("ingress_reference_baseline.txt")
# Ingress differences belong to acquisition: the evidence layer, the adapters that scope a loader
# or wrap it as a provider, and the loaders themselves.
_ACQUISITION_MODULES = frozenset(
    {
        Path("_evidence_types.py"),
        Path("_source_types.py"),
        Path("_evidence_refusals.py"),
        Path("readouts/_evidence.py"),
        Path("_evidence_loaders.py"),
        Path("_evidence_scopes.py"),
        Path("evidence.py"),
        Path("readouts/_provider.py"),
        Path("frame.py"),
        Path("sources.py"),
        Path("switchback.py"),
    }
)
_INGRESS_PREDICATES = {
    "route": frozenset({"is_artifact_route", "_is_artifact_source"}),
    "custom": frozenset(
        {"is_custom_source", "is_builtin_source", "is_custom_provider", "serves_custom_breakouts"}
    ),
    "unwrap": frozenset({"_builtin_source"}),
}


def _is_acquisition(rel: Path) -> bool:
    return (
        rel in _ACQUISITION_MODULES
        or rel.parts[0] == "query"
        or (len(rel.parts) == 1 and rel.name.startswith("_frame_"))
    )


def _names_a_route(node: ast.AST) -> bool:
    return any(
        (isinstance(child, ast.Name) and "route" in child.id.lower())
        or (isinstance(child, ast.Attribute) and "route" in child.attr.lower())
        for child in ast.walk(node)
    )


def _route_literal_lines(node: ast.Compare, values: frozenset[str]) -> list[int]:
    """Lines of route values spelled as literals and compared with a route-named expression."""
    operands = (node.left, *node.comparators)
    lines = [
        item.lineno
        for operand in operands
        for item in (
            operand.elts if isinstance(operand, (ast.Tuple, ast.List, ast.Set)) else (operand,)
        )
        if isinstance(item, ast.Constant) and item.value in values
    ]
    return lines if lines and any(_names_a_route(operand) for operand in operands) else []


def _ingress_references(root: Path) -> dict[tuple[str, str], list[int]]:
    """Lines per (module, kind) where code outside acquisition names an ingress."""
    # Source routes are the acquisition module's module-level ``*_ROUTE`` string constants.
    evidence = (root / "readouts" / "_evidence.py").read_text()
    routes = dict(
        re.findall(r"^(\w+_ROUTE)(?:\s*:[^=\n]*)?\s*=\s*[\"'](\w+)[\"']", evidence, re.MULTILINE)
    )
    values = frozenset(routes.values())
    kinds = dict.fromkeys(routes, "route")
    kinds.update({name: kind for kind, names in _INGRESS_PREDICATES.items() for name in names})
    literals = tuple(f"{quote}{value}{quote}" for value in values for quote in "\"'")
    references: dict[tuple[str, str], list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if _is_acquisition(rel):
            continue
        text = path.read_text()
        # Parse only modules that spell a counted name, or a route value beside a route name.
        if not any(name in text for name in kinds) and not (
            any(literal in text for literal in literals) and "route" in text.lower()
        ):
            continue
        module = f"increment/{rel.as_posix()}"
        for node in ast.walk(ast.parse(text, filename=str(path))):
            if isinstance(node, ast.Compare):
                lines, kind = _route_literal_lines(node, values), "route"
            elif isinstance(node, ast.Name) and node.id in kinds:
                lines, kind = [node.lineno], kinds[node.id]
            elif isinstance(node, ast.Attribute) and node.attr in kinds:
                lines, kind = [node.lineno], kinds[node.attr]
            else:
                continue
            if lines:
                references.setdefault((module, kind), []).extend(lines)
    return references


def _load_ingress_baseline(path: Path) -> Counter[tuple[str, str]]:
    baseline: Counter[tuple[str, str]] = Counter()
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        fields = raw.split()
        if (
            len(fields) != 3
            or not fields[2].isdecimal()
            or int(fields[2]) == 0
            or (fields[0], fields[1]) in baseline
        ):
            raise AssertionError(f"{path}:{number}: expected module, kind, positive count")
        baseline[(fields[0], fields[1])] = int(fields[2])
    return baseline


def test_interpretation_does_not_grow_ingress_branches() -> None:
    """Ingress branches outside acquisition may be removed, never added."""
    observed = _ingress_references(Path(__file__).parents[1] / "increment")
    baseline = _load_ingress_baseline(_INGRESS_BASELINE)
    grown = [
        f"{module}:{','.join(map(str, sorted(lines)))}: {kind} "
        f"({len(lines)}, baseline {baseline[(module, kind)]})"
        for (module, kind), lines in sorted(observed.items())
        if len(lines) > baseline[(module, kind)]
    ]
    assert not grown, (
        "New ingress branches outside acquisition; declare the difference as an "
        "EvidenceCapabilities field and branch on that, or move the logic into acquisition "
        "(increment/readouts/_evidence.py, increment/_evidence_loaders.py):\n" + "\n".join(grown)
    )
    stale = [
        f"{module} {kind}: baseline {count}, observed {len(observed.get((module, kind), ()))}"
        for (module, kind), count in sorted(baseline.items())
        if len(observed.get((module, kind), ())) < count
    ]
    assert not stale, (
        f"Stale ingress baseline; lower or remove these entries in {_INGRESS_BASELINE.name}:\n"
        + "\n".join(stale)
    )
