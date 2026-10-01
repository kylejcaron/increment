"""AST gate: every literal-code raise site supplies exactly the context
keys its RefusalSpec declares. Catches the drift a template spec's `keys`
contract exists to prevent, at collection time instead of at whatever
runtime path happens to exercise the mismatched site.

WarningSpec is deliberately not covered here: its `render` field is a
plain callable with no template/`keys` contract (unlike RefusalSpec's
template variant), so a mistyped `_warn(...)` kwarg already raises
TypeError at the call itself -- the same reasoning that lets a
render_fn-shaped RefusalSpec skip this gate too.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import increment
from increment.errors import RefusalSpec


def _registries() -> dict[str, RefusalSpec]:
    """code -> spec, across every module-level RefusalSpec or dict of
    RefusalSpec in the package (mirrors test_refusal_uniqueness.py's walk)."""
    out: dict[str, RefusalSpec] = {}
    for _finder, name, _ispkg in pkgutil.walk_packages(increment.__path__, prefix="increment."):
        module = importlib.import_module(name)
        for value in vars(module).values():
            if isinstance(value, RefusalSpec):
                out.setdefault(value.code, value)
            elif isinstance(value, dict):
                for spec in value.values():
                    if isinstance(spec, RefusalSpec):
                        out.setdefault(spec.code, spec)
    return out


def _literal_kwargs(node: ast.Call) -> set[str] | None:
    """The call's keyword-argument names, or None if it splats `**something`
    (unresolvable statically -- covered at runtime by refuse()'s own check
    instead)."""
    if any(kw.arg is None for kw in node.keywords):
        return None
    return {kw.arg for kw in node.keywords if kw.arg is not None}


def _find_literal_raise_sites(tree: ast.AST) -> list[tuple[str, set[str] | None]]:
    """(code, kwarg_names_or_None) for every `_raise("literal", k=v, ...)`
    call whose code resolves to a literal string at the call site."""
    sites: list[tuple[str, set[str] | None]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id in ("_raise",) and node.args:
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                sites.append((arg0.value, _literal_kwargs(node)))
    return sites


def test_every_literal_raise_site_supplies_exactly_its_spec_keys() -> None:
    registries = _registries()
    violations: list[str] = []
    root = Path(increment.__file__).parent
    for path in root.rglob("*.py"):
        source = path.read_text()
        tree = ast.parse(source, filename=str(path))
        for code, kwargs in _find_literal_raise_sites(tree):
            spec = registries.get(code)
            if spec is None or kwargs is None or spec.template is None:
                # code not found as a plain RefusalSpec, a **splat site, or a
                # render_fn spec (its own call signature is its drift guard).
                continue
            if kwargs != spec.keys:
                violations.append(
                    f"{path.relative_to(root.parent)}: _raise({code!r}, "
                    f"{sorted(kwargs)}) but spec.keys={sorted(spec.keys)}"
                )
    assert violations == [], "\n".join(violations)
