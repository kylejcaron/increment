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
from typing import __all__ as _TYPING_EXPORTS
from typing import cast

_FUNCTION = (ast.FunctionDef, ast.AsyncFunctionDef)
type _Declaration = ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef | ast.TypeAlias
type _Scope = _Declaration | _TypeParameters | _Body | ast.AnnAssign
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
_TYPING_FACTORIES = {
    "TypeVar",
    "ParamSpec",
    "TypeVarTuple",
    "NewType",
    "TypeAliasType",
    "NamedTuple",
    "TypedDict",
}
_TYPING_NAMES = frozenset(_TYPING_EXPORTS) | _TYPING_FACTORIES
_BUILTIN_TYPE_NAMES = {
    "bool",
    "bytearray",
    "bytes",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "object",
    "set",
    "str",
    "tuple",
    "type",
}
_TYPE_EXPRESSION_MODULES = _TYPING | {"collections.abc"}


@dataclass(frozen=True)
class _TypeParameters:
    """Lazy scope of a generic class or function's type parameters, outside its body."""

    declaration: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _Body:
    """Local scope of a function body, separate from its definition-time signature names."""

    function: ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class _Reference:
    module: str
    name: str | None
    scope: _Scope | None = None


@dataclass(frozen=True, eq=False)
class _Binding:
    declaration: ast.AST | _Reference
    value: ast.AST | _Reference | _Binding | None
    owner: _Reference


@dataclass
class _Module:
    tree: ast.Module | _Scope
    owner: _Reference
    parent: _Reference | None = None
    declarations: dict[str, _Binding] = field(default_factory=dict)
    bindings: dict[str, _Binding] = field(default_factory=dict)
    values: dict[str, ast.expr] = field(default_factory=dict)
    stars: list[str] = field(default_factory=list)
    functions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = field(default_factory=dict)

    def define(
        self,
        name: str,
        binding: ast.AST | _Reference,
        value_binding: ast.AST | _Reference | _Binding | None = None,
        *,
        owner: _Reference | None = None,
    ) -> None:
        value = binding.value if isinstance(binding, (ast.AnnAssign, ast.TypeAlias)) else binding
        record = _Binding(
            binding,
            value_binding if value_binding is not None else value,
            owner if owner is not None else _Reference(self.owner.module, name, self.owner.scope),
        )
        self.declarations[name] = record
        if not isinstance(binding, ast.AnnAssign) or binding.value is not None:
            self.bindings[name] = record


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


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    args = node.args
    return [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *(argument for argument in (args.vararg, args.kwarg) if argument is not None),
    ]


class _Source:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.modules: dict[str, _Module] = {}
        self.scopes: dict[_Scope, _Module] = {}

    def path(self, name: str) -> Path | None:
        path = self.root.joinpath(*name.split("."))
        if (init := path / "__init__.py").is_file():
            return init
        if (module := path.with_suffix(".py")).is_file():
            return module
        return None

    def module(self, name: str, scope: _Scope | None = None) -> _Module:
        if scope is None and name in self.modules:
            return self.modules[name]
        # Annotation captures are created while binding their function body.
        if scope is not None and (scope in self.scopes or isinstance(scope, ast.AnnAssign)):
            return self.scopes[scope]
        if isinstance(scope, _Body):
            return self.function_body(name, scope.function, None)
        if scope is not None and not isinstance(scope, ast.ClassDef):
            return self.declaration_scope(name, scope, None)
        path = self.path(name)
        if path is None:
            raise ValueError(f"No Python source for {name}")
        module = _Module(
            scope if scope is not None else ast.parse(path.read_text(), filename=str(path)),
            _Reference(name, None, scope),
            parent=_Reference(name, None) if scope is not None else None,
        )
        if scope is None:
            self.modules[name] = module
        else:
            self.scopes[scope] = module
        self.bind(name, module, self.package(name))
        return module

    def package(self, name: str) -> str:
        path = self.path(name)
        return name if path is not None and path.name == "__init__.py" else name.rpartition(".")[0]

    def deferred(self, name: str) -> bool:
        return any(
            isinstance(statement, ast.ImportFrom)
            and statement.module == "__future__"
            and any(alias.name == "annotations" for alias in statement.names)
            for statement in cast(ast.Module, self.module(name).tree).body
        )

    def bind_type_parameters(self, module: _Module, node: _Declaration) -> None:
        for parameter in node.type_params:
            if not isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
                raise ValueError(f"Unsupported type parameter in {module.owner.module}")
            module.define(parameter.name, parameter)

    def declaration_scope(
        self, name: str, node: _Declaration | _TypeParameters, parent: _Scope | None
    ) -> _Module:
        if isinstance(node, (*_FUNCTION, ast.ClassDef)) and node.type_params:
            self.declaration_scope(name, _TypeParameters(node), parent)
        if isinstance(node, ast.ClassDef):
            return self.module(name, node)
        if node in self.scopes:
            return self.scopes[node]
        if isinstance(node, _FUNCTION):
            return self.capture(
                name,
                node,
                [*(argument.annotation for argument in _parameters(node)), node.returns],
                _TypeParameters(node) if node.type_params else parent,
            )
        environment = _Module(
            node, _Reference(name, None, node), parent=_Reference(name, None, parent)
        )
        self.scopes[node] = environment
        self.bind_type_parameters(
            environment, node.declaration if isinstance(node, _TypeParameters) else node
        )
        return environment

    def capture(
        self,
        name: str,
        key: ast.FunctionDef | ast.AsyncFunctionDef | ast.AnnAssign,
        annotations: list[ast.expr | None],
        parent: _Scope | None,
    ) -> _Module:
        """Bind names an eager annotation evaluates now; quoted names resolve in `parent`."""
        environment = _Module(
            key, _Reference(name, None, key), parent=_Reference(name, None, parent)
        )
        self.scopes[key] = environment
        if self.deferred(name):
            return environment
        for annotation in annotations:
            if annotation is None:
                continue
            for part in ast.walk(annotation):
                if not isinstance(part, ast.Name) or part.id in environment.bindings:
                    continue
                binding = self.lookup(name, part.id, parent)
                if isinstance(binding, _Binding):
                    environment.bindings[part.id] = binding
                elif isinstance(binding, _Reference):
                    environment.define(part.id, binding)
                elif part.id in _BUILTIN_TYPE_NAMES:
                    environment.define(part.id, _Reference("builtins", part.id))
        return environment

    def function_body(
        self, name: str, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: _Scope | None
    ) -> _Module:
        key = _Body(node)
        if key in self.scopes:
            return self.scopes[key]
        body = _Module(
            node,
            _Reference(name, None, key),
            parent=self.declaration_scope(name, node, parent).parent,
        )
        self.scopes[key] = body
        self.bind(name, body, self.package(name))
        return body

    def forward_scope(self, module: str, scope: _Scope | None) -> _Scope | None:
        """Quoted annotations resolve late, outside definition-time name captures."""
        if not isinstance(scope, (*_FUNCTION, ast.AnnAssign)):
            return scope
        parent = self.module(module, scope).parent
        return None if parent is None else parent.scope

    def copied_binding(
        self, module: str, node: ast.expr | None, scope: _Scope | None
    ) -> _Binding | _Reference | None:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        ref = self.reference(module, node, scope, allow_forward_strings=False)
        if ref is None:
            return None
        if ref.name is not None and self.path(ref.module) is not None:
            previous = self.lookup(ref.module, ref.name, ref.scope)
            if previous is not None:
                return previous
        return ref

    def bind_from_import(
        self, name: str, module: _Module, package: str, node: ast.ImportFrom
    ) -> None:
        parts = package.split(".")
        base = ".".join(parts[: len(parts) - node.level + 1]) if node.level else ""
        target = ".".join(filter(None, (base, node.module)))
        for alias in node.names:
            if alias.name == "*":
                module.stars.append(target)
                continue
            local = alias.asname or alias.name
            submodule = f"{target}.{alias.name}"
            # A package importing its own name gets the attribute it already holds;
            # only an unbound name loads that submodule.
            current = self.lookup(name, alias.name) if target == name else None
            if isinstance(current, _Binding):
                module.declarations[local] = module.bindings[local] = current
            elif current is not None:
                module.define(local, current)
            elif target == name and self.path(submodule) is not None:
                module.define(local, _Reference(submodule, None))
            else:
                module.define(local, _Reference(target, alias.name))

    def bind(self, name: str, module: _Module, package: str) -> None:
        exports = isinstance(module.tree, ast.Module)
        scope = module.owner.scope
        assert isinstance(module.tree, (ast.Module, ast.ClassDef, *_FUNCTION))
        if isinstance(module.tree, ast.ClassDef):
            self.bind_type_parameters(module, module.tree)
        for node in _statements(module.tree.body):
            if isinstance(node, (*_FUNCTION, ast.ClassDef)):
                self.declaration_scope(name, node, scope)
                module.define(node.name, node)
                if isinstance(node, _FUNCTION):
                    module.functions.setdefault(node.name, []).append(node)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module.define(
                        alias.asname or alias.name.split(".")[0],
                        _Reference(alias.name if alias.asname else alias.name.split(".")[0], None),
                    )
            elif isinstance(node, ast.ImportFrom):
                self.bind_from_import(name, module, package, node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module.define(
                            target.id, node.value, self.copied_binding(name, node.value, scope)
                        )
                        module.values[target.id] = node.value
                    elif exports and any(
                        isinstance(part, ast.Name) and part.id in _EXPORT_MAPS
                        for part in ast.walk(target)
                    ):
                        raise ValueError(f"Unsupported export assignment in {name}")
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                module.define(node.target.id, node, self.copied_binding(name, node.value, scope))
                if node.value is not None:
                    module.values[node.target.id] = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(module.tree, _FUNCTION):
                self.capture(name, node, [node.annotation], scope)
            elif isinstance(node, ast.TypeAlias):
                self.declaration_scope(name, node, scope)
                module.define(node.name.id, node, owner=_Reference(name, node.name.id, node))
            elif exports and isinstance(node, ast.AugAssign):
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
                exports
                and isinstance(node, ast.Expr)
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

    def literal(
        self,
        name: str,
        node: ast.expr,
        seen: frozenset[str] = frozenset(),
        scope: _Scope | None = None,
    ) -> object:
        if isinstance(node, ast.Name) and node.id not in seen:
            source = self.module(name, scope)
            if scope is not None and node.id not in source.values and source.parent is not None:
                parent = source.parent
                return self.literal(parent.module, node, seen, parent.scope)
            value = self.module(name, scope).values.get(node.id)
            if value is not None:
                return self.literal(name, value, seen | {node.id}, scope)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return [self.literal(name, item, seen, scope) for item in node.elts]
        if isinstance(node, ast.Dict):
            result: dict[object, object] = {}
            for key, value in zip(node.keys, node.values, strict=True):
                if key is None:
                    unpacked = self.literal(name, value, seen, scope)
                    if not isinstance(unpacked, dict):
                        raise ValueError(f"Invalid export map unpacking in {name}")
                    result.update(cast(dict[object, object], unpacked))
                else:
                    result[self.literal(name, key, seen, scope)] = self.literal(
                        name, value, seen, scope
                    )
            return result
        if isinstance(node, ast.BinOp):
            left, right = (
                self.literal(name, node.left, seen, scope),
                self.literal(name, node.right, seen, scope),
            )
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
                value = self.literal(name, node.args[0], seen, scope)
                if isinstance(value, (list, dict)):
                    return list(value)
        raise ValueError(f"Cannot statically resolve exports in {name}: {ast.unparse(node)}")

    def exports(self, name: str) -> list[str]:
        node = self.module(name).values.get("__all__")
        if node is None:
            # Explicit `as` re-exports cover older facades without __all__.
            names = [
                alias.asname
                for statement in _statements(cast(ast.Module, self.module(name).tree).body)
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

    def class_scope(
        self,
        target: _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
    ) -> _Reference | None:
        while target is not None:
            if target in seen:
                owner = target.owner if isinstance(target, _Binding) else target
                raise ValueError(f"Cyclic annotation alias: {owner.module}.{owner.name}")
            seen = seen | {target}
            if isinstance(target, _Binding):
                if isinstance(target.declaration, ast.ClassDef):
                    return _Reference(target.owner.module, None, target.declaration)
                value = self.symbol_value(target, seen)
                target = (
                    value
                    if isinstance(value, (_Reference, _Binding))
                    else self.reference(target.owner.module, value, target.owner.scope, seen)
                )
            else:
                if target.name is None or self.path(target.module) is None:
                    return None
                target = self.lookup(target.module, target.name, target.scope)
        return None

    def lookup(
        self, module: str, name: str, scope: _Scope | None = None
    ) -> _Binding | _Reference | None:
        source = self.module(module, scope)
        if name in source.bindings:
            return source.bindings[name]
        for star in reversed(source.stars):
            if star in _TYPING:
                if name in _TYPING_NAMES:
                    return _Reference(star, name)
            elif name in self.exports(star):
                return _Reference(star, name)
        if source.parent is not None:
            return self.lookup(source.parent.module, name, source.parent.scope)
        return None

    def reference(
        self,
        module: str,
        node: ast.AST | _Reference | None,
        scope: _Scope | None = None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        *,
        allow_forward_strings: bool = True,
    ) -> _Reference | None:
        if isinstance(node, _Reference):
            return node
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if not allow_forward_strings:
                return None
            try:
                return self.reference(
                    module,
                    ast.parse(node.value, mode="eval").body,
                    self.forward_scope(module, scope),
                    seen,
                    allow_forward_strings=allow_forward_strings,
                )
            except SyntaxError as exc:
                raise ValueError(f"Invalid annotation in {module}: {node.value}") from exc
        if isinstance(node, ast.Name):
            binding = self.lookup(module, node.id, scope)
            if isinstance(binding, _Reference):
                return binding
            if isinstance(binding, _Binding) and isinstance(binding.declaration, _Reference):
                return binding.declaration
            source = self.module(module, scope)
            if scope is not None and node.id not in source.bindings and source.parent is not None:
                parent = source.parent
                return self.reference(
                    parent.module,
                    node,
                    parent.scope,
                    seen,
                    allow_forward_strings=allow_forward_strings,
                )
            return _Reference(module, node.id, scope)
        if isinstance(node, ast.Attribute):
            parent = self.reference(
                module, node.value, scope, seen, allow_forward_strings=allow_forward_strings
            )
            if parent is not None:
                class_owner = self.class_scope(parent, seen)
                if class_owner is not None:
                    if node.attr not in self.module(class_owner.module, class_owner.scope).bindings:
                        return None
                    return _Reference(class_owner.module, node.attr, class_owner.scope)
                owner = ".".join(filter(None, (parent.module, parent.name)))
                # External members keep their qualified path; sources are never traversed.
                if (
                    parent.name is None
                    or self.path(owner) is not None
                    or self.path(parent.module) is None
                ):
                    return _Reference(owner, node.attr)
        return None

    def resolve(
        self,
        module: str,
        name: str,
        seen: frozenset[_Reference] = frozenset(),
        scope: _Scope | None = None,
    ) -> tuple[_Reference, ast.AST | _Binding | None]:
        key = _Reference(module, name, scope)
        if key in seen:
            raise ValueError(f"Cyclic public re-export: {module}.{name}")
        seen = seen | {key}
        source = self.module(module, scope)
        record = source.declarations.get(name)
        binding = record.declaration if record is not None else None
        for map_name in ("_LAZY_IMPORTS", "_EXPORTS") if scope is None else ():
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
            if scope is None and self.path(f"{module}.{name}") is not None:
                return _Reference(f"{module}.{name}", None), None
            raise ValueError(f"Unresolved public export: {module}.{name}")
        # An ordinary assignment exports the value its name held at that point.
        while (
            isinstance(binding, (ast.Name, ast.Attribute))
            and record is not None
            and isinstance(record.value, _Binding)
        ):
            record = record.value
            key, binding = record.owner, record.declaration
        if isinstance(binding, (ast.Name, ast.Attribute)) and record is not None:
            target = (
                record.value
                if isinstance(record.value, _Reference)
                else self.reference(key.module, binding, key.scope)
            )
            if (
                target is not None
                and self.is_type_expression(key.module, target, scope=key.scope)
                and self.typing_name(target) == "Any"
            ):
                return key, record
            binding = target or binding
        if isinstance(binding, _Reference):
            if binding.name is None:
                if self.path(binding.module) is None:
                    raise ValueError(f"No Python source for public module {binding.module}")
                return _Reference(binding.module, None, binding.scope), None
            if self.path(binding.module) is None:
                raise ValueError(
                    f"No Python source for public export {binding.module}.{binding.name}"
                )
            return self.resolve(binding.module, binding.name, seen, binding.scope)
        # Declared aliases count through their own record, not a later name lookup.
        if (
            record is not None
            and record.declaration is binding
            and isinstance(binding, (ast.TypeAlias, ast.AnnAssign))
        ):
            return key, record
        return key, binding

    def typing_name(
        self,
        ref: _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        *,
        allow_forward_strings: bool = True,
    ) -> str | None:
        visited = set(seen)
        while ref is not None and ref not in visited:
            visited.add(ref)
            if isinstance(ref, _Binding):
                # Explicit aliases put their values back in type, not runtime, context.
                allow_forward_strings = allow_forward_strings or self.declared_alias(ref)
                owner = ref.owner
                value = self.symbol_value(ref, frozenset(visited))
                ref = (
                    value
                    if isinstance(value, (_Reference, _Binding))
                    else self.reference(
                        owner.module,
                        value,
                        owner.scope,
                        frozenset(visited),
                        allow_forward_strings=allow_forward_strings,
                    )
                )
                continue
            if ref.module in _TYPING:
                return ref.name
            if ref.name is None or self.path(ref.module) is None:
                return None
            ref = self.lookup(ref.module, ref.name, ref.scope)
        if ref is not None:
            owner = ref.owner if isinstance(ref, _Binding) else ref
            raise ValueError(f"Cyclic annotation alias: {owner.module}.{owner.name}")
        return None

    def type_value(
        self,
        module: str,
        binding: ast.AST | _Reference | _Binding | None,
        scope: _Scope | None = None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
    ) -> ast.expr | None:
        if isinstance(binding, _Binding):
            module, scope = binding.owner.module, binding.owner.scope
            binding = binding.declaration
        if isinstance(binding, ast.TypeAlias):
            return binding.value
        if isinstance(binding, ast.AnnAssign):
            if binding.value is None:
                return None
            marker = self.reference(module, binding.annotation, scope)
            if (
                marker is not None
                and marker.name is not None
                and self.path(marker.module) is not None
                and isinstance(
                    current := self.lookup(marker.module, marker.name, marker.scope), _Binding
                )
                and current.declaration is binding
            ):
                return None
            if self.typing_name(marker, seen) == "TypeAlias":
                return binding.value
            return None
        if isinstance(binding, ast.Constant) and isinstance(binding.value, str):
            # An ordinary string assignment is data, not an implicit alias.
            return None
        return binding if isinstance(binding, ast.expr) else None

    def symbol_value(
        self, binding: _Binding, seen: frozenset[_Reference | _Binding]
    ) -> ast.expr | _Reference | _Binding | None:
        if isinstance(binding.value, (_Reference, _Binding)):
            return binding.value
        return self.type_value(binding.owner.module, binding, binding.owner.scope, seen)

    def factory_field(
        self,
        module: str,
        name: ast.expr,
        value: ast.AST,
        seen: frozenset[_Reference | _Binding],
        scope: _Scope | None,
        productive: frozenset[_Binding] = frozenset(),
    ) -> dict[str, int]:
        label = self.literal(module, name, scope=scope)
        if not isinstance(label, str):
            raise ValueError(f"Invalid typing field name in {module}: {ast.unparse(name)}")
        return {f"{label}.field": self.annotation(module, value, seen, scope, productive)}

    def factory_fields(
        self,
        module: str,
        node: ast.AST | _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding],
        scope: _Scope | None,
        *,
        mapping: bool,
        entry: bool = False,
        keyword_flags: bool = False,
        productive: frozenset[_Binding] = frozenset(),
    ) -> dict[str, int]:
        while isinstance(node, (_Binding, _Reference)):
            if node in seen:
                owner = node.owner if isinstance(node, _Binding) else node
                raise ValueError(f"Cyclic annotation alias: {owner.module}.{owner.name}")
            seen = seen | {node}
            if isinstance(node, _Binding):
                module, scope = node.owner.module, node.owner.scope
                node = node.value
            else:
                if node.name is None or self.path(node.module) is None:
                    raise ValueError(
                        f"Cannot statically resolve typing fields: {node.module}.{node.name}"
                    )
                module, scope = node.module, node.scope
                node = self.lookup(module, node.name, scope)
        fields: dict[str, int] = {}
        if mapping and isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            fields = self.factory_fields(
                module,
                node.left,
                seen,
                scope,
                mapping=True,
                keyword_flags=keyword_flags,
                productive=productive,
            )
            fields.update(
                self.factory_fields(
                    module,
                    node.right,
                    seen,
                    scope,
                    mapping=True,
                    keyword_flags=keyword_flags,
                    productive=productive,
                )
            )
            return fields
        if mapping and isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if key is not None and keyword_flags:
                    label = self.literal(module, key, scope=scope)
                    if isinstance(label, str) and label in {"total", "closed"}:
                        continue
                fields.update(
                    self.factory_fields(
                        module,
                        value,
                        seen,
                        scope,
                        mapping=True,
                        keyword_flags=keyword_flags,
                        productive=productive,
                    )
                    if key is None
                    else self.factory_field(module, key, value, seen, scope, productive)
                )
            return fields
        if not mapping and isinstance(node, (ast.List, ast.Tuple)):
            if entry:
                if len(node.elts) != 2:
                    raise ValueError(f"Invalid NamedTuple field in {module}: {ast.unparse(node)}")
                return self.factory_field(
                    module, node.elts[0], node.elts[1], seen, scope, productive
                )
            for pair in node.elts:
                fields.update(
                    self.factory_fields(
                        module,
                        pair,
                        seen,
                        scope,
                        mapping=False,
                        entry=True,
                        productive=productive,
                    )
                )
            return fields
        ref = self.reference(module, node, scope)
        if ref is not None:
            return self.factory_fields(
                module,
                ref,
                seen,
                scope,
                mapping=mapping,
                entry=entry,
                keyword_flags=keyword_flags,
                productive=productive,
            )
        expression = ast.unparse(node) if node is not None else "None"
        raise ValueError(f"Cannot statically resolve typing fields in {module}: {expression}")

    def factory_sites(
        self,
        module: str,
        node: ast.Call,
        seen: frozenset[_Reference | _Binding],
        scope: _Scope | None,
        productive: frozenset[_Binding] = frozenset(),
    ) -> dict[str, int] | None:
        kind = self.typing_name(self.reference(module, node.func, scope))
        if kind not in _TYPING_FACTORIES:
            return None
        arguments: list[ast.AST] = []
        fields: dict[str, int] = {}
        if kind in {"TypeVar", "ParamSpec", "TypeVarTuple"}:
            if kind == "TypeVar":
                arguments.extend(node.args[1:])
            arguments.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == "default" or keyword.arg == "bound" and kind != "TypeVarTuple"
            )
        elif kind in {"NewType", "TypeAliasType"}:
            arguments.extend(node.args[1:2])
            arguments.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg == ("tp" if kind == "NewType" else "value")
            )
        else:
            if len(node.args) > 1:
                fields = self.factory_fields(
                    module,
                    node.args[1],
                    seen,
                    scope,
                    mapping=kind == "TypedDict",
                    productive=productive,
                )
            for keyword in node.keywords:
                if keyword.arg is None:
                    fields.update(
                        self.factory_fields(
                            module,
                            keyword.value,
                            seen,
                            scope,
                            mapping=True,
                            keyword_flags=kind == "TypedDict",
                            productive=productive,
                        )
                    )
                elif (
                    kind == "TypedDict"
                    and keyword.arg == "extra_items"
                    or len(node.args) < 2
                    and (kind == "NamedTuple" or keyword.arg not in {"total", "closed"})
                ):
                    fields.update(
                        self.factory_field(
                            module,
                            ast.Constant(value=keyword.arg),
                            keyword.value,
                            seen,
                            scope,
                            productive,
                        )
                    )
            return fields
        return {
            "alias": sum(
                self.annotation(module, argument, seen, scope, productive) for argument in arguments
            )
        }

    def declared_alias(self, binding: _Binding) -> bool:
        return isinstance(binding.declaration, ast.TypeAlias) or (
            isinstance(binding.declaration, ast.AnnAssign)
            and self.type_value(binding.owner.module, binding, binding.owner.scope) is not None
        )

    def annotation(
        self,
        module: str,
        node: ast.AST | _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        scope: _Scope | None = None,
        productive: frozenset[_Binding] = frozenset(),
    ) -> int:
        if node is None:
            return 0
        if isinstance(node, _Binding):
            if node in seen:
                if node in productive:
                    return 0
                raise ValueError(f"Cyclic annotation alias: {node.owner.module}.{node.owner.name}")
            declaration = node.declaration
            # Naming string data in an annotation evaluates it as a forward reference.
            value = (
                declaration
                if node.value is declaration
                and isinstance(declaration, ast.Constant)
                and isinstance(declaration.value, str)
                else self.symbol_value(node, seen | {node})
            )
            return self.annotation(
                node.owner.module, value, seen | {node}, node.owner.scope, productive
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                return self.annotation(
                    module,
                    ast.parse(node.value, mode="eval").body,
                    seen,
                    self.forward_scope(module, scope),
                    productive,
                )
            except SyntaxError as exc:
                raise ValueError(f"Invalid annotation in {module}: {node.value}") from exc
        ref = self.reference(module, node, scope)
        binding = None
        if ref is not None and ref.name is not None and self.path(ref.module) is not None:
            binding = self.lookup(ref.module, ref.name, ref.scope)
            if isinstance(binding, _Binding) and binding in seen:
                if binding in productive:
                    return 0
                raise ValueError(f"Cyclic annotation alias: {ref.module}.{ref.name}")
            if ref in seen:
                raise ValueError(f"Cyclic annotation alias: {ref.module}.{ref.name}")
        if self.typing_name(ref) == "Any":
            return 1
        if ref is not None and ref.name is not None and self.path(ref.module) is not None:
            return self.annotation(ref.module, binding, seen | {ref}, ref.scope, productive)
        if isinstance(node, _Reference):
            return 0
        if isinstance(node, ast.Call):
            factory = self.factory_sites(module, node, seen, scope, productive)
            if factory is not None:
                return sum(factory.values())
        if isinstance(node, ast.Subscript):
            kind = self.typing_name(self.reference(module, node.value, scope))
            if kind == "Literal":
                return 0
            if kind == "Annotated":
                first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
                return self.annotation(module, first, seen, scope, productive)
            productive = productive | frozenset(
                item for item in seen if isinstance(item, _Binding) and self.declared_alias(item)
            )
        return sum(
            self.annotation(module, child, seen, scope, productive)
            for child in ast.iter_child_nodes(node)
        )

    def type_parameter_sites(self, module: str, node: _Declaration) -> dict[str, int]:
        scope = node if isinstance(node, ast.TypeAlias) else _TypeParameters(node)
        sites: dict[str, int] = {}
        for parameter in node.type_params:
            if not isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
                raise ValueError(f"Unsupported type parameter in {module}")
            prefix = f"type_parameter:{parameter.name}"
            if isinstance(parameter, ast.TypeVar) and parameter.bound is not None:
                if isinstance(parameter.bound, ast.Tuple):
                    for index, constraint in enumerate(parameter.bound.elts, 1):
                        sites[f"{prefix}.constraint:{index}"] = self.annotation(
                            module, constraint, scope=scope
                        )
                else:
                    sites[f"{prefix}.bound"] = self.annotation(module, parameter.bound, scope=scope)
            for field_name, value in ast.iter_fields(parameter):
                if field_name == "default_value" and isinstance(value, ast.expr):
                    sites[f"{prefix}.default"] = self.annotation(module, value, scope=scope)
        return sites

    def signature(
        self,
        module: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: _Scope | None = None,
    ) -> dict[str, int]:
        self.declaration_scope(module, node, scope)
        scope = node
        params = _parameters(node)
        return {
            **self.type_parameter_sites(module, node),
            **{
                f"parameter:{arg.arg}": self.annotation(module, arg.annotation, scope=scope)
                for arg in params
            },
            "return": self.annotation(module, node.returns, scope=scope),
        }

    def is_overload(
        self,
        module: str,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        scope: _Scope | None = None,
    ) -> bool:
        return any(
            self.typing_name(self.reference(module, decorator, scope)) == "overload"
            for decorator in node.decorator_list
        )

    def is_type_expression(
        self,
        module: str,
        node: ast.AST | _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        scope: _Scope | None = None,
    ) -> bool:
        if isinstance(node, _Binding):
            if node in seen:
                raise ValueError(f"Cyclic annotation alias: {node.owner.module}.{node.owner.name}")
            declaration = node.declaration
            if isinstance(declaration, (ast.ClassDef, ast.TypeAlias)):
                return True
            value = (
                declaration
                if isinstance(declaration, _Reference)
                else self.type_value(node.owner.module, node, node.owner.scope, seen | {node})
            )
            return value is not None and self.is_type_expression(
                node.owner.module, value, seen | {node}, node.owner.scope
            )
        if isinstance(node, ast.Constant):
            return node.value is None
        if isinstance(node, ast.Call):
            return self.typing_name(self.reference(module, node.func, scope)) in _TYPING_FACTORIES
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ast.BitOr):
                return False
            left_type = self.is_type_expression(module, node.left, seen, scope)
            right_type = self.is_type_expression(module, node.right, seen, scope)
            return (
                left_type
                and (
                    right_type
                    or self.is_opaque_external_type_operand(module, node.right, seen, scope)
                )
            ) or (
                right_type and self.is_opaque_external_type_operand(module, node.left, seen, scope)
            )
        if isinstance(node, ast.Subscript):
            return self.is_type_expression(module, node.value, seen, scope) or (
                self.is_opaque_external_type_operand(module, node.value, seen, scope)
                and self.is_external_type_argument(module, node.slice, seen, scope)
            )
        if isinstance(node, ast.Tuple):
            return bool(node.elts) and all(
                self.is_type_expression(module, element, seen, scope) for element in node.elts
            )
        ref = self.reference(module, node, scope)
        if ref is None or ref.name is None:
            return False
        if ref.module in _TYPE_EXPRESSION_MODULES:
            return True
        if ref.module == "builtins" and ref.name in _BUILTIN_TYPE_NAMES:
            return True
        if self.path(ref.module) is None:
            return False
        if ref in seen:
            raise ValueError(f"Cyclic annotation alias: {ref.module}.{ref.name}")
        binding = self.lookup(ref.module, ref.name, ref.scope)
        if binding is None:
            return ref.module == module and ref.name in _BUILTIN_TYPE_NAMES
        return self.is_type_expression(ref.module, binding, seen | {ref}, ref.scope)

    def is_opaque_external_type_operand(
        self,
        module: str,
        node: ast.AST | _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        scope: _Scope | None = None,
    ) -> bool:
        """Recognize an opaque imported member only in a type-shaped context."""
        while node is not None:
            if isinstance(node, _Binding):
                if node in seen:
                    raise ValueError(
                        f"Cyclic annotation alias: {node.owner.module}.{node.owner.name}"
                    )
                seen = seen | {node}
                module, scope = node.owner.module, node.owner.scope
                node = (
                    node.declaration
                    if isinstance(node.declaration, _Reference)
                    else self.type_value(module, node, scope, seen)
                )
                continue
            ref = self.reference(module, node, scope)
            if ref is None or ref.name is None:
                return False
            if ref in seen:
                raise ValueError(f"Cyclic annotation alias: {ref.module}.{ref.name}")
            seen = seen | {ref}
            if ref.module in _TYPE_EXPRESSION_MODULES:
                return False
            if self.path(ref.module) is None:
                return True
            node = self.lookup(ref.module, ref.name, ref.scope)
        return False

    def reject_unresolved_typing_star(
        self, module: str, node: ast.AST, scope: _Scope | None
    ) -> None:
        for part in ast.walk(node):
            operands = (
                [part.value]
                if isinstance(part, ast.Subscript)
                else [part.left, part.right]
                if isinstance(part, ast.BinOp) and isinstance(part.op, ast.BitOr)
                else []
            )
            for operand in operands:
                ref = self.reference(module, operand, scope)
                if (
                    ref is None
                    or ref.name is None
                    or ref.name in _BUILTIN_TYPE_NAMES
                    or self.path(ref.module) is None
                    or self.lookup(ref.module, ref.name, ref.scope) is not None
                ):
                    continue
                source = self.module(ref.module, ref.scope)
                while True:
                    if any(star in _TYPING for star in source.stars):
                        if self.annotation(module, node, scope=scope):
                            raise ValueError(
                                f"Unresolved typing star name: {ref.module}.{ref.name}"
                            )
                        break
                    if source.parent is None:
                        break
                    source = self.module(source.parent.module, source.parent.scope)

    def is_external_type_argument(
        self,
        module: str,
        node: ast.AST,
        seen: frozenset[_Reference | _Binding],
        scope: _Scope | None = None,
    ) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            try:
                node = ast.parse(value, mode="eval").body
            except SyntaxError:
                return False
            return self.is_external_type_argument(module, node, seen, scope)
        if isinstance(node, ast.Tuple):
            return bool(node.elts) and all(
                self.is_external_type_argument(module, element, seen, scope)
                for element in node.elts
            )
        return self.is_type_expression(
            module, node, seen, scope
        ) or self.is_opaque_external_type_operand(module, node, seen, scope)

    def reject_ambiguous_opaque_value(
        self, module: str, name: str, node: ast.AST, scope: _Scope | None = None
    ) -> None:
        """Require a declaration when an opaque quoted lookup can mean an Any alias."""
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Constant):
            return
        if not isinstance(node.slice.value, str):
            return
        if self.is_type_expression(module, node.value, scope=scope):
            return
        if not self.is_opaque_external_type_operand(module, node.value, scope=scope):
            return
        if not self.is_external_type_argument(module, node.slice, frozenset(), scope):
            return
        argument = ast.parse(node.slice.value, mode="eval").body
        if self.annotation(module, argument, scope=scope):
            raise ValueError(
                f"Ambiguous public expression {module}.{name}; add a value annotation "
                "or declare the expression as TypeAlias"
            )

    def sites(
        self,
        module: str,
        node: ast.AST | _Binding,
        name: str | None = None,
        scope: _Scope | None = None,
    ) -> dict[str, int]:
        record = node if isinstance(node, _Binding) else None
        if record is not None:
            declaration = record.declaration
            if not isinstance(declaration, (ast.TypeAlias, ast.AnnAssign)):
                # A bare implicit alias such as `Alias = Any` has no other definition.
                return {"alias": self.annotation(module, record, scope=scope)}
            node = declaration
        if isinstance(node, _FUNCTION):
            sites = self.signature(module, node, scope)
            source = self.module(module, scope)
            binding = source.bindings.get(node.name)
            if binding is not None and binding.declaration is node:
                overloads = [
                    definition
                    for definition in source.functions[node.name]
                    if definition is not node and self.is_overload(module, definition, scope)
                ]
                for index, definition in enumerate(overloads, 1):
                    sites.update(
                        {
                            f"overload:{index}.{site}": count
                            for site, count in self.signature(module, definition, scope).items()
                        }
                    )
            return sites
        if isinstance(node, ast.AnnAssign):
            # TypeAlias's value is itself an annotation, not an implementation value.
            annotation = self.type_value(module, node, scope)
            if annotation is not None:
                if record is None and isinstance(node.target, ast.Name):
                    record = self.module(module, scope).declarations.get(node.target.id)
                target = record if record is not None and record.declaration is node else annotation
                return {"field": self.annotation(module, target, scope=scope)}
            return {"field": self.annotation(module, node.annotation, scope=scope)}
        if isinstance(node, ast.TypeAlias):
            if record is None:
                record = self.module(module, scope).declarations.get(node.name.id)
            target = record if record is not None and record.declaration is node else node.value
            return {
                **self.type_parameter_sites(module, node),
                "alias": self.annotation(module, target, scope=scope),
            }
        if isinstance(node, ast.Call):
            return self.factory_sites(module, node, frozenset(), scope) or {}
        if isinstance(node, (ast.Subscript, ast.BinOp)):
            if name is not None:
                self.reject_ambiguous_opaque_value(module, name, node, scope)
            if self.is_type_expression(module, node, scope=scope):
                return {"alias": self.annotation(module, node, scope=scope)}
            self.reject_unresolved_typing_star(module, node, scope)
            return {}
        return self.class_sites(module, node) if isinstance(node, ast.ClassDef) else {}

    def class_sites(self, module: str, node: ast.ClassDef) -> dict[str, int]:
        scope = node
        source = self.module(module, scope)
        sites = self.type_parameter_sites(module, node)
        members: dict[str, list[ast.AST | _Binding]] = {}
        for member in _statements(node.body):
            if isinstance(member, ast.Assign):
                for target in member.targets:
                    if isinstance(target, ast.Name) and (
                        not target.id.startswith("_")
                        or target.id.startswith("__")
                        and target.id.endswith("__")
                    ):
                        record = source.declarations.get(target.id)
                        # Like an exported name, a bare final Any alias has no other definition.
                        members.setdefault(target.id, []).append(
                            record
                            if isinstance(member.value, (ast.Name, ast.Attribute))
                            and record is not None
                            and record.declaration is member.value
                            and self.typing_name(record, allow_forward_strings=False) == "Any"
                            else member.value
                        )
                continue
            if isinstance(member, (*_FUNCTION, ast.ClassDef)):
                name = member.name
                if isinstance(member, _FUNCTION) and not any(
                    isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                    for decorator in member.decorator_list
                ):
                    positional = [*member.args.posonlyargs, *member.args.args]
                    receiver = positional[0].arg if positional else "self"
                    self.function_body(module, member, scope)
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
                                    module, statement.annotation, scope=statement
                                )
            elif isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                name = member.target.id
            elif isinstance(member, ast.TypeAlias):
                name = member.name.id
            else:
                continue
            if not name.startswith("_") or (name.startswith("__") and name.endswith("__")):
                members.setdefault(name, []).append(member)
        for name, definitions in members.items():
            if any(
                isinstance(member, _FUNCTION) and self.is_overload(module, member, scope)
                for member in definitions
            ):
                binding = source.bindings.get(name)
                definitions = [
                    member
                    for member in definitions
                    if binding is not None and binding.declaration is member
                ]
            for member in definitions:
                for site, count in self.sites(module, member, name, scope).items():
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
                {
                    site: count
                    for site, count in source.sites(owner.module, node, name, owner.scope).items()
                    if count
                }
                if node is not None
                else {}
            )
            if node is None:
                visit(owner.module, qualified, seen | {module})

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
