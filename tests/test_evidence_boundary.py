from __future__ import annotations

import ast
from pathlib import Path

EVIDENCE_METHODS = frozenset({"unit_frame", "moments"})
_SOURCE_NAMES = frozenset({"src", "source"})
_ALLOWED = {
    ("adjust.py", "estimate_ate"),
    ("adjust.py", "_estimate_ate"),
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


def test_loader_routes_stay_inside_acquisition_and_facades():
    root = Path(__file__).parents[1] / "increment"
    allowed = {
        Path("readouts/_evidence.py"),
        Path("analysis.py"),
        Path("_day_axis.py"),
        Path("_whole_window.py"),
        Path("_breakout_readouts.py"),
        Path("_sitewide.py"),
        Path("readouts/_breakout.py"),
    }
    route_names = {"artifact", "moments", "native", "panel", "unsupported"}
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        tree = ast.parse(path.read_text(), filename=str(path))
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
            route_comparison = isinstance(node, ast.Compare) and any(
                (isinstance(value, ast.Name) and value.id == "route")
                or (isinstance(value, ast.Attribute) and value.attr == "route")
                for value in ast.walk(node)
            )
            route_argument = isinstance(node, ast.Call) and any(
                keyword.arg == "route"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value in route_names
                for keyword in node.keywords
            )
            if (route_comparison or route_argument) and rel not in allowed:
                if any(
                    isinstance(value, ast.Constant) and value.value in route_names
                    for value in ast.walk(node)
                ):
                    violations.append(
                        f"{rel}:{node.lineno}: route literal outside acquisition/facade"
                    )
    assert not violations, "\n".join(violations)
