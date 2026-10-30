from __future__ import annotations

import ast
from pathlib import Path

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


def test_loader_routes_stay_inside_acquisition_and_facades():  # noqa: C901
    root = Path(__file__).parents[1] / "increment"
    # The evidence layer plus the analysis facade and its four readout facades.
    allowed = {
        Path("readouts/_evidence.py"),
        Path("analysis.py"),
        Path("_day_axis.py"),
        Path("_whole_window.py"),
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
        if (path.parent == root / "readouts" and path.name != "_evidence.py")
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
