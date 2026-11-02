"""Ratchet explicit Any annotations in exported signatures and public fields.

Source inspection never imports the checked package. The baseline records every
export (including zero counts) and each annotated parameter, return or field,
so removing one Any cannot pay for adding it somewhere else.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
_LEGACY_NAMESPACES = (
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
)
_TYPING = {"typing", "typing_extensions"}
_EXPORT_MAPS = {"__all__", "_EXPORTS", "_LAZY_IMPORTS"}
_TYPING_NAMES = {"Any", "Annotated", "Literal", "TypeAlias", "overload"}


@dataclass(frozen=True)
class _Reference:
    module: str
    name: str | None


@dataclass
class _Module:
    tree: ast.Module
    bindings: dict[str, ast.AST | _Reference] = field(default_factory=dict)
    values: dict[str, ast.expr] = field(default_factory=dict)
    stars: list[str] = field(default_factory=list)
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = field(default_factory=dict)


def _statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    for node in body:
        yield node
        if isinstance(node, (*_FUNCTION, ast.ClassDef)):
            continue
        type_checking = isinstance(node, ast.If) and (
            isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            or isinstance(node.test, ast.Attribute)
            and node.test.attr == "TYPE_CHECKING"
        )
        for name, children in ast.iter_fields(node):
            if not isinstance(children, list) or (name == "orelse" and type_checking):
                continue
            for child in children:
                if isinstance(child, ast.stmt):
                    yield from _statements([child])
                elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                    yield from _statements(child.body)


class _Source:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.modules: dict[str, _Module] = {}

    def path(self, name: str) -> Path | None:
        path = self.root.joinpath(*name.split("."))
        if (init := path / "__init__.py").is_file():
            return init
        if (module := path.with_suffix(".py")).is_file():
            return module
        return None

    def module(self, name: str) -> _Module:
        if name in self.modules:
            return self.modules[name]
        path = self.path(name)
        if path is None:
            raise ValueError(f"No Python source for {name}")
        module = _Module(ast.parse(path.read_text(), filename=str(path)))
        self.modules[name] = module
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in _statements(module.tree.body):
            if isinstance(node, (*_FUNCTION, ast.ClassDef)):
                module.bindings[node.name] = node
                if isinstance(node, _FUNCTION):
                    module.functions.setdefault(node.name, []).append(node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module.bindings[alias.asname or alias.name.split(".")[0]] = _Reference(
                        alias.name if alias.asname else alias.name.split(".")[0], None
                    )
            elif isinstance(node, ast.ImportFrom):
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1]) if node.level else ""
                target = ".".join(filter(None, (base, node.module)))
                for alias in node.names:
                    if alias.name == "*":
                        module.stars.append(target)
                    else:
                        module.bindings[alias.asname or alias.name] = _Reference(target, alias.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module.bindings[target.id] = node.value
                        module.values[target.id] = node.value
                    elif any(
                        isinstance(part, ast.Name) and part.id in _EXPORT_MAPS
                        for part in ast.walk(target)
                    ):
                        raise ValueError(f"Unsupported export assignment in {name}")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                module.bindings[node.target.id] = node
                if node.value is not None:
                    module.values[node.target.id] = node.value
            elif isinstance(node, ast.TypeAlias):
                module.bindings[node.name.id] = node
            elif isinstance(node, ast.AugAssign):
                target = node.target
                if (
                    isinstance(target, ast.Name)
                    and target.id in module.values
                    and isinstance(node.op, (ast.Add, ast.BitOr))
                ):
                    module.values[target.id] = ast.BinOp(
                        left=module.values[target.id], op=node.op, right=node.value
                    )
                elif any(
                    isinstance(part, ast.Name) and part.id in _EXPORT_MAPS
                    for part in ast.walk(target)
                ):
                    raise ValueError(f"Unsupported export mutation in {name}")
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id in _EXPORT_MAPS
            ):
                call = node.value
                if (
                    node.value.func.value.id == "__all__"
                    and node.value.func.attr in {"append", "extend"}
                    and len(call.args) == 1
                    and not call.keywords
                    and "__all__" in module.values
                ):
                    extra = (
                        ast.List(elts=[call.args[0]])
                        if node.value.func.attr == "append"
                        else call.args[0]
                    )
                    module.values["__all__"] = ast.BinOp(
                        left=module.values["__all__"], op=ast.Add(), right=extra
                    )
                else:
                    raise ValueError(f"Unsupported export mutation in {name}")
        return module

    def literal(self, name: str, node: ast.expr, seen: frozenset[str] = frozenset()) -> object:
        if isinstance(node, ast.Name) and node.id not in seen:
            value = self.module(name).values.get(node.id)
            if value is not None:
                return self.literal(name, value, seen | {node.id})
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [self.literal(name, item, seen) for item in node.elts]
        if isinstance(node, ast.Dict):
            result: dict[object, object] = {}
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    unpacked = self.literal(name, value, seen)
                    if not isinstance(unpacked, dict):
                        raise ValueError(f"Invalid export map unpacking in {name}")
                    result.update(cast(dict[object, object], unpacked))
                else:
                    result[self.literal(name, key, seen)] = self.literal(name, value, seen)
            return result
        if isinstance(node, ast.BinOp):
            left, right = self.literal(name, node.left, seen), self.literal(name, node.right, seen)
            if isinstance(node.op, ast.Add) and isinstance(left, list) and isinstance(right, list):
                return left + right
            if (
                isinstance(node.op, ast.BitOr)
                and isinstance(left, dict)
                and isinstance(right, dict)
            ):
                return left | right
        if isinstance(node, ast.Call) and len(node.args) == 1 and not node.keywords:
            if isinstance(node.func, ast.Name) and node.func.id in {"list", "tuple", "sorted"}:
                value = self.literal(name, node.args[0], seen)
                if isinstance(value, (list, dict)):
                    return list(value)
        raise ValueError(f"Cannot statically resolve exports in {name}: {ast.unparse(node)}")

    def exports(self, name: str) -> list[str]:
        node = self.module(name).values.get("__all__")
        if node is None:
            # Explicit `as` re-exports cover older facades without __all__.
            names = [
                alias.asname
                for statement in _statements(self.module(name).tree.body)
                if isinstance(statement, ast.ImportFrom)
                for alias in statement.names
                if alias.asname is not None and not alias.asname.startswith("_")
            ]
            if not names:
                raise ValueError(f"Missing __all__ or explicit re-exports in {name}")
            return names
        names = self.literal(name, node)
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            raise ValueError(f"Invalid __all__ in {name}")
        return cast(list[str], names)

    def lookup(self, module: str, name: str) -> ast.AST | _Reference | None:
        source = self.module(module)
        if name in source.bindings:
            return source.bindings[name]
        for star in reversed(source.stars):
            if star in _TYPING:
                if name in _TYPING_NAMES:
                    return _Reference(star, name)
            elif name in self.exports(star):
                return _Reference(star, name)
        return None

    def reference(self, module: str, node: ast.AST | None) -> _Reference | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self.reference(module, ast.parse(node.value, mode="eval").body)
            except SyntaxError as exc:
                raise ValueError(f"Invalid annotation in {module}: {node.value}") from exc
        if isinstance(node, ast.Name):
            binding = self.lookup(module, node.id)
            return binding if isinstance(binding, _Reference) else _Reference(module, node.id)
        if isinstance(node, ast.Attribute):
            parent = self.reference(module, node.value)
            if parent is not None:
                owner = ".".join(filter(None, (parent.module, parent.name)))
                if parent.name is None or self.path(owner) is not None:
                    return _Reference(owner, node.attr)
        return None

    def resolve(
        self, module: str, name: str, seen: frozenset[tuple[str, str]] = frozenset()
    ) -> tuple[str, ast.AST | None]:
        key = (module, name)
        if key in seen:
            raise ValueError(f"Cyclic public re-export: {module}.{name}")
        seen = seen | {key}
        source = self.module(module)
        binding = source.bindings.get(name)
        for map_name in ("_LAZY_IMPORTS", "_EXPORTS"):
            if map_name in source.values:
                exports = self.literal(module, source.values[map_name])
                if isinstance(exports, dict) and name in exports:
                    target = cast(dict[str, object], exports)[name]
                    if (
                        not isinstance(target, list)
                        or len(target) != 2
                        or not isinstance(target[0], str)
                        or not (isinstance(target[1], str) or target[1] is None)
                    ):
                        raise ValueError(f"Invalid lazy export {module}.{name}")
                    binding = _Reference(target[0], target[1])
        if binding is None:
            for star in reversed(source.stars):
                if star in _TYPING:
                    if name in _TYPING_NAMES:
                        raise ValueError(f"No Python source for public export {star}.{name}")
                    continue
                if name in self.exports(star):
                    return self.resolve(star, name, seen)
            if self.path(f"{module}.{name}") is not None:
                return f"{module}.{name}", None
            raise ValueError(f"Unresolved public export: {module}.{name}")
        if isinstance(binding, (ast.Name, ast.Attribute)):
            expression = binding
            binding = self.reference(module, expression) or expression
            if isinstance(binding, _Reference) and self.typing_name(binding) == "Any":
                return module, ast.TypeAlias(
                    name=ast.Name(id=name), type_params=[], value=expression
                )
        if isinstance(binding, _Reference):
            if binding.name is None:
                if self.path(binding.module) is None:
                    raise ValueError(f"No Python source for public module {binding.module}")
                return binding.module, None
            if self.path(binding.module) is None:
                raise ValueError(
                    f"No Python source for public export {binding.module}.{binding.name}"
                )
            return self.resolve(binding.module, binding.name, seen)
        return module, binding

    def typing_name(self, ref: _Reference | None) -> str | None:
        seen: set[_Reference] = set()
        while ref is not None and ref not in seen:
            seen.add(ref)
            if ref.module in _TYPING:
                return ref.name
            if ref.name is None or self.path(ref.module) is None:
                return None
            binding = self.lookup(ref.module, ref.name)
            value = self.type_value(ref.module, binding)
            ref = binding if isinstance(binding, _Reference) else self.reference(ref.module, value)
        if ref is not None:
            raise ValueError(f"Cyclic annotation alias: {ref.module}.{ref.name}")
        return None

    def type_value(self, module: str, binding: ast.AST | _Reference | None) -> ast.expr | None:
        if isinstance(binding, ast.TypeAlias):
            return binding.value
        if isinstance(binding, ast.AnnAssign):
            if self.typing_name(self.reference(module, binding.annotation)) == "TypeAlias":
                return binding.value
            return None
        return binding if isinstance(binding, ast.expr) else None

    def annotation(
        self, module: str, node: ast.AST | None, seen: frozenset[_Reference] = frozenset()
    ) -> int:
        if node is None:
            return 0
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self.annotation(module, ast.parse(node.value, mode="eval").body, seen)
            except SyntaxError as exc:
                raise ValueError(f"Invalid annotation in {module}: {node.value}") from exc
        ref = self.reference(module, node)
        if self.typing_name(ref) == "Any":
            return 1
        if ref is not None and ref.name is not None and self.path(ref.module) is not None:
            if ref in seen:
                raise ValueError(f"Cyclic annotation alias: {ref.module}.{ref.name}")
            binding = self.lookup(ref.module, ref.name)
            if isinstance(binding, _Reference) and binding.name is not None:
                if self.path(binding.module) is None:
                    return 0
                return self.annotation(binding.module, ast.Name(id=binding.name), seen | {ref})
            value = self.type_value(ref.module, binding)
            if value is not None:
                return self.annotation(ref.module, value, seen | {ref})
        if isinstance(node, ast.Subscript):
            kind = self.typing_name(self.reference(module, node.value))
            if kind == "Literal":
                return 0
            if kind == "Annotated":
                first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
                return self.annotation(module, first, seen)
        return sum(self.annotation(module, child, seen) for child in ast.iter_child_nodes(node))

    def signature(
        self, module: str, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> dict[str, int]:
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
        params += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
        return {
            **{f"parameter:{arg.arg}": self.annotation(module, arg.annotation) for arg in params},
            "return": self.annotation(module, node.returns),
        }

    def sites(self, module: str, node: ast.AST) -> dict[str, int]:
        if isinstance(node, _FUNCTION):
            sites = self.signature(module, node)
            if self.module(module).bindings.get(node.name) is node:
                overloads = [
                    definition
                    for definition in self.module(module).functions[node.name]
                    if definition is not node
                    and any(
                        self.typing_name(self.reference(module, decorator)) == "overload"
                        for decorator in definition.decorator_list
                    )
                ]
                for index, definition in enumerate(overloads, 1):
                    sites.update(
                        {
                            f"overload:{index}.{site}": count
                            for site, count in self.signature(module, definition).items()
                        }
                    )
            return sites
        if isinstance(node, ast.AnnAssign):
            # TypeAlias's value is itself an annotation, not an implementation value.
            annotation = self.type_value(module, node)
            if annotation is None:
                annotation = node.annotation
            return {"field": self.annotation(module, annotation)}
        if isinstance(node, ast.TypeAlias):
            return {"alias": self.annotation(module, node.value)}
        if not isinstance(node, ast.ClassDef):
            return {}
        sites: dict[str, int] = {}
        members: dict[str, list[ast.AST]] = {}
        for member in _statements(node.body):
            if isinstance(member, (*_FUNCTION, ast.ClassDef)):
                name = member.name
                if isinstance(member, _FUNCTION) and not any(
                    isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                    for decorator in member.decorator_list
                ):
                    positional = [*member.args.posonlyargs, *member.args.args]
                    receiver = positional[0].arg if positional else "self"
                    for statement in _statements(member.body):
                        if isinstance(statement, ast.AnnAssign):
                            target = statement.target
                            if (
                                isinstance(target, ast.Attribute)
                                and isinstance(target.value, ast.Name)
                                and target.value.id == receiver
                                and not target.attr.startswith("_")
                            ):
                                label = f"{member.name}.field:{target.attr}"
                                sites[label] = sites.get(label, 0) + self.annotation(
                                    module, statement.annotation
                                )
            elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                name = member.target.id
            else:
                continue
            if not name.startswith("_") or (name.startswith("__") and name.endswith("__")):
                members.setdefault(name, []).append(member)
        for name, definitions in members.items():
            for member in definitions:
                for site, count in self.sites(module, member).items():
                    label = f"{name}.{site}"
                    sites[label] = sites.get(label, 0) + count
        return sites


def _public_sites(root: Path, namespaces: Sequence[str]) -> dict[str, dict[str, int]]:
    source = _Source(root)
    result: dict[str, dict[str, int]] = {}

    def visit(module: str, public: str, seen: frozenset[str]) -> None:
        if module in seen:
            raise ValueError(f"Cyclic public module export: {module}")
        for name in source.exports(module):
            qualified = f"{public}.{name}"
            owner, node = source.resolve(module, name)
            result[qualified] = (
                {site: count for site, count in source.sites(owner, node).items() if count}
                if node is not None
                else {}
            )
            if node is None:
                visit(owner, qualified, seen | {module})

    for namespace in namespaces:
        visit(namespace, namespace, frozenset())
    return dict(sorted(result.items()))


def public_any(root: Path, namespaces: Sequence[str]) -> dict[str, int]:
    """Count explicit Any occurrences per qualified export, without importing code."""
    return {name: sum(sites.values()) for name, sites in _public_sites(root, namespaces).items()}


def _namespaces(root: Path, package: str) -> list[str]:
    manifest = root / "tests/public_exports.txt"
    headers = []
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            namespace = line[1:-1]
            if not all(part.isidentifier() for part in namespace.split(".")):
                raise ValueError(f"Invalid public namespace: {namespace}")
            if namespace == package or namespace.startswith(f"{package}."):
                headers.append(namespace)
    if headers:
        return headers
    if package == "increment":
        return list(_LEGACY_NAMESPACES)
    raise ValueError(f"No promised namespaces for {package} in {manifest}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--package", default="increment")
    parser.add_argument("--namespace", action="append")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args(argv)
    baseline = args.baseline or args.root / "tests/public_any_baseline.txt"
    try:
        namespaces = args.namespace or _namespaces(args.root, args.package)
        current = _public_sites(args.root, namespaces)
        if args.write_baseline:
            baseline.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
            print(
                f"Recorded {len(current)} exports, "
                f"{sum(bool(sites) for sites in current.values())} with Any, "
                f"{sum(sum(sites.values()) for sites in current.values())} Any occurrences"
            )
            return 0
        expected = json.loads(baseline.read_text())
        failures = []
        for name, sites in current.items():
            if name not in expected:
                failures.append(f"New public name: {name}")
            for site, count in sites.items():
                limit = expected.get(name, {}).get(site, 0)
                if count > limit:
                    failures.append(f"{name} {site}: Any {limit} -> {count}")
        if failures:
            print("\n".join(failures))
            return 1
    except (ValueError, OSError) as exc:
        print(f"Public typing check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
