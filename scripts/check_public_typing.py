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
type _Scope = _Declaration | _TypeParameters | _Body | ast.AnnAssign | ast.Assign | _ValueScope
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
class _ValueScope:
    """Eager value-expression scope, such as a definition's decorators, apart from annotations."""

    declaration: (
        ast.Assign
        | ast.AnnAssign
        | ast.AugAssign
        | ast.Expr
        | ast.FunctionDef
        | ast.AsyncFunctionDef
    )


@dataclass(frozen=True)
class _Reference:
    module: str
    name: str | None
    scope: _Scope | None = None


@dataclass(frozen=True)
class _Binding:
    declaration: ast.AST | _Reference
    value: ast.AST | _Reference | _Binding | None
    owner: _Reference
    uncertainty: _Uncertainty | None = None


@dataclass(frozen=True)
class _Uncertainty:
    control: ast.AST
    candidates: tuple[_Binding | _Reference | None, ...]


@dataclass(frozen=True)
class _Mutation:
    """An in-place change to a list or mapping; without an operand it cannot be followed."""

    statement: ast.AST
    operand: ast.expr | None = None
    scope: _Scope | None = None


@dataclass(frozen=True)
class _ModuleState:
    declarations: dict[str, _Binding]
    bindings: dict[str, _Binding]
    mutations: dict[_Binding, tuple[_Mutation, ...]]
    stars: tuple[str, ...]
    uncertain_star_names: dict[str, _Uncertainty]


@dataclass
class _Module:
    tree: ast.Module | _Scope
    owner: _Reference
    parent: _Reference | None = None
    declarations: dict[str, _Binding] = field(default_factory=dict)
    bindings: dict[str, _Binding] = field(default_factory=dict)
    # Replaced, never updated in place: an eager capture keeps the changes made before it.
    mutations: dict[_Binding, tuple[_Mutation, ...]] = field(default_factory=dict)
    uncertain_star_names: dict[str, _Uncertainty] = field(default_factory=dict)
    stars: list[str] = field(default_factory=list)

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
            value_binding.uncertainty
            if isinstance(value_binding, _Binding)
            and not isinstance(binding, (ast.AnnAssign, ast.TypeAlias))
            else None,
        )
        self.assign(name, record)

    def assign(self, name: str, record: _Binding) -> None:
        """Apply one binding occurrence in statement order."""
        declaration = self.declarations.get(name)
        if (
            declaration is not None
            and isinstance(declaration.declaration, ast.AnnAssign)
            and not isinstance(
                record.declaration, (ast.AnnAssign, ast.TypeAlias, ast.ClassDef, *_FUNCTION)
            )
        ):
            self.bindings[name] = record
        else:
            self.declarations[name] = record
            if isinstance(record.declaration, ast.AnnAssign) and record.declaration.value is None:
                # An annotation alone declares the name; it binds no runtime value.
                return
            self.bindings[name] = record
        # A runtime binding supersedes a conditional star-import candidate.
        self.uncertain_star_names.pop(name, None)

    def unknown(self, name: str, statement: ast.AST) -> None:
        """Shadow an outer value without inventing a value for a runtime binding."""
        self.assign(
            name,
            _Binding(
                statement,
                None,
                _Reference(self.owner.module, name, self.owner.scope),
                _Uncertainty(statement, (None,)),
            ),
        )

    def change(self, root: _Binding, mutation: _Mutation) -> None:
        self.mutations = {**self.mutations, root: (*self.mutations.get(root, ()), mutation)}


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    args = node.args
    return [
        *args.posonlyargs,
        *args.args,
        *args.kwonlyargs,
        *(argument for argument in (args.vararg, args.kwarg) if argument is not None),
    ]


def _assigned_value(binding: _Binding) -> ast.AST | _Reference | None:
    """The value expression a binding evaluated, apart from any annotation."""
    declaration = binding.declaration
    return declaration.value if isinstance(declaration, ast.AnnAssign) else declaration


def _target_leaves(target: ast.expr) -> Iterator[ast.expr]:
    """Names, items and attributes an assignment target stores into."""
    if isinstance(target, ast.Starred):
        yield from _target_leaves(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _target_leaves(element)
    else:
        yield target


def _unpacked(target: ast.expr, value: ast.expr) -> Iterator[tuple[ast.expr, ast.expr, bool]]:
    """Pair each stored target with its value; inexact when it gets an unknown part of `value`."""
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and isinstance(value, (ast.Tuple, ast.List))
        and len(target.elts) == len(value.elts)
        and not any(isinstance(part, ast.Starred) for part in (*target.elts, *value.elts))
    ):
        for part, element in zip(target.elts, value.elts, strict=True):
            yield from _unpacked(part, element)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for leaf in _target_leaves(target):
            yield leaf, value, False
    else:
        yield target, value, True


def _bound_names(node: ast.Assign | ast.Import | ast.ImportFrom) -> list[str]:
    """Names an assignment or import statement binds, as the binder defines them."""
    if isinstance(node, ast.Assign):
        return [
            leaf.id
            for target in node.targets
            for leaf in _target_leaves(target)
            if isinstance(leaf, ast.Name)
        ]
    if isinstance(node, ast.Import):
        return [alias.asname or alias.name.split(".")[0] for alias in node.names]
    return [alias.asname or alias.name for alias in node.names if alias.name != "*"]


def _receiver_body_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    for node in body:
        yield node
        if isinstance(node, (*_FUNCTION, ast.ClassDef)):
            continue
        for _, children in ast.iter_fields(node):
            if isinstance(children, list):
                for child in children:
                    if isinstance(child, ast.stmt):
                        yield from _receiver_body_statements([child])
                    elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                        yield from _receiver_body_statements(child.body)


class _Source:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.modules: dict[str, _Module] = {}
        self.scopes: dict[_Scope, _Module] = {}
        # Branches selected when each body was bound; later rebinding cannot reselect them.
        self.guards: dict[ast.If, bool | None] = {}
        # Same-name definitions bound before each function; its overloads are among them.
        self.overload_candidates: dict[
            ast.FunctionDef | ast.AsyncFunctionDef,
            tuple[ast.FunctionDef | ast.AsyncFunctionDef, ...],
        ] = {}

    def path(self, name: str) -> Path | None:
        path = self.root.joinpath(*name.split("."))
        if (init := path / "__init__.py").is_file():
            return init
        if (module := path.with_suffix(".py")).is_file():
            return module
        return None

    def module(
        self,
        name: str,
        scope: _Scope | None = None,
        *,
        changes: dict[_Binding, tuple[_Mutation, ...]] | None = None,
    ) -> _Module:
        if scope is None and name in self.modules:
            return self.modules[name]
        # Annotation captures are created while binding their function body.
        if scope is not None and (scope in self.scopes or isinstance(scope, ast.AnnAssign)):
            return self.scopes[scope]
        if isinstance(scope, _Body):
            return self.function_body(name, scope.function, None)
        if scope is not None and not isinstance(
            scope, (ast.ClassDef, ast.Assign, ast.AnnAssign, _ValueScope)
        ):
            return self.declaration_scope(name, scope, None)
        path = self.path(name)
        if path is None:
            raise ValueError(f"No Python source for {name}")
        module = _Module(
            scope if scope is not None else ast.parse(path.read_text(), filename=str(path)),
            _Reference(name, None, scope),
            parent=_Reference(name, None) if scope is not None else None,
            mutations={} if changes is None else changes,
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
            # A class body runs in order with the body that defines it.
            return self.module(name, node, changes=self.body_changes(name, parent))
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
            node,
            _Reference(name, None, node),
            parent=_Reference(name, None, parent),
        )
        self.scopes[node] = environment
        self.bind_type_parameters(
            environment, node.declaration if isinstance(node, _TypeParameters) else node
        )
        return environment

    def capture(
        self,
        name: str,
        key: ast.FunctionDef | ast.AsyncFunctionDef | ast.AnnAssign | ast.Assign | _ValueScope,
        expressions: Sequence[ast.expr | None],
        parent: _Scope | None,
        *,
        defer_future_annotations: bool = True,
    ) -> _Module:
        """Capture names resolved when an eager expression is evaluated."""
        environment = _Module(
            key,
            _Reference(name, None, key),
            parent=_Reference(name, None, parent),
            mutations=self.body_changes(name, parent),
        )
        self.scopes[key] = environment
        if defer_future_annotations and self.deferred(name):
            return environment
        for expression in expressions:
            if expression is None:
                continue
            for part in ast.walk(expression):
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
        lexical_parent = self.declaration_scope(name, node, parent).parent
        body_bindings: dict[str, _Binding] = {}
        if isinstance(parent, ast.ClassDef):
            class_scope = self.module(name, parent)
            lexical_parent = class_scope.parent
            for parameter in parent.type_params:
                if isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
                    if parameter.name in class_scope.bindings:
                        body_bindings[parameter.name] = class_scope.bindings[parameter.name]
            if node.type_params:
                parameter_scope = self.module(name, _TypeParameters(node))
                for parameter in node.type_params:
                    if isinstance(parameter, (ast.TypeVar, ast.ParamSpec, ast.TypeVarTuple)):
                        if parameter.name in parameter_scope.bindings:
                            body_bindings[parameter.name] = parameter_scope.bindings[parameter.name]
        body = _Module(
            node,
            _Reference(name, None, key),
            parent=lexical_parent,
            bindings=body_bindings,
            # A call runs once the module or function body defining it has finished.
            mutations=self.body_changes(name, parent, late=True),
        )
        for parameter in _parameters(node):
            body.unknown(parameter.arg, parameter)
        self.scopes[key] = body
        self.bind(name, body, self.package(name))
        return body

    def forward_scope(self, module: str, scope: _Scope | None) -> _Scope | None:
        """Quoted annotations resolve late, outside definition-time name captures."""
        if not isinstance(scope, (*_FUNCTION, ast.Assign, ast.AnnAssign, _ValueScope)):
            return scope
        parent = self.module(module, scope).parent
        return None if parent is None else parent.scope

    def runtime_uncertainty(
        self, binding: ast.AST | _Binding | _Reference | None
    ) -> _Uncertainty | None:
        """Unknown runtime provenance, independent of any authoritative value annotation."""
        seen: set[_Binding | _Reference] = set()
        while isinstance(binding, (_Binding, _Reference)) and binding not in seen:
            seen.add(binding)
            if isinstance(binding, _Binding):
                if binding.uncertainty is not None:
                    return binding.uncertainty
                binding = binding.value
            elif binding.name is not None and self.path(binding.module) is not None:
                binding = self.lookup(binding.module, binding.name, binding.scope)
            else:
                return None
        return None

    def copied_binding(
        self,
        module: str,
        node: ast.expr | None,
        scope: _Scope | None,
        owner: _Reference | None = None,
    ) -> _Binding | _Reference | None:
        if not isinstance(node, (ast.Name, ast.Attribute)):
            return None
        if isinstance(node, ast.Attribute):
            parents = []
            expression: ast.expr = node
            while isinstance(expression, ast.Attribute):
                parents.append(expression.value)
                expression = expression.value
            for expression in reversed(parents):
                parent = self.reference(module, expression, scope, allow_forward_strings=False)
                if (
                    parent is not None
                    and parent.name is not None
                    and self.path(parent.module) is not None
                ):
                    binding = self.lookup(parent.module, parent.name, parent.scope)
                    uncertainty = self.runtime_uncertainty(binding)
                    if uncertainty is not None:
                        return _Binding(
                            node,
                            None,
                            owner or _Reference(module, node.attr, scope),
                            uncertainty,
                        )
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
                if target in _TYPING:
                    for imported in _TYPING_NAMES:
                        module.define(imported, _Reference(target, imported))
                elif target != name and self.path(target) is not None:
                    for imported in self.exports(target):
                        module.define(imported, _Reference(target, imported))
                continue
            local = alias.asname or alias.name
            submodule = f"{target}.{alias.name}"
            # A package importing its own name gets the attribute it already holds;
            # only an unbound name loads that submodule.
            current = self.lookup(name, alias.name) if target == name else None
            if isinstance(current, _Binding):
                module.assign(local, current)
            elif current is not None:
                module.define(local, current)
            elif target == name and self.path(submodule) is not None:
                module.define(local, _Reference(submodule, None))
            else:
                module.define(local, _Reference(target, alias.name))

    def _snapshot(self, module: _Module) -> _ModuleState:
        return _ModuleState(
            dict(module.declarations),
            dict(module.bindings),
            dict(module.mutations),
            tuple(module.stars),
            dict(module.uncertain_star_names),
        )

    def _restore(self, module: _Module, state: _ModuleState) -> None:
        module.declarations = dict(state.declarations)
        module.bindings = dict(state.bindings)
        module.mutations = dict(state.mutations)
        module.stars = list(state.stars)
        module.uncertain_star_names = dict(state.uncertain_star_names)

    def _typing_condition_reference(
        self, module: str, node: ast.expr, scope: _Scope | None
    ) -> _Reference | None:
        def imported_reference(
            binding: _Binding | _Reference | None, seen: frozenset[_Binding] = frozenset()
        ) -> _Reference | None:
            if isinstance(binding, _Reference):
                return binding if binding.module in _TYPING else None
            if binding is None or binding.uncertainty is not None or binding in seen:
                return None
            for value in (binding.value, binding.declaration):
                if isinstance(value, (_Binding, _Reference)):
                    reference = imported_reference(value, seen | {binding})
                    if reference is not None:
                        return reference
            return None

        def local_reference(name: str) -> _Reference | None:
            source = self.module(module, scope)
            while True:
                if name in source.bindings:
                    return imported_reference(source.bindings[name])
                if name in source.uncertain_star_names or source.parent is None:
                    return None
                source = self.module(source.parent.module, source.parent.scope)

        if isinstance(node, ast.Name):
            reference = local_reference(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            owner = local_reference(node.value.id)
            reference = (
                _Reference(owner.module, node.attr)
                if owner is not None and owner.name is None
                else None
            )
        else:
            reference = None
        if reference is not None and reference.module in _TYPING:
            return reference
        return None

    def _condition(self, module: str, node: ast.expr, scope: _Scope | None) -> bool | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self._condition(module, node.operand, scope)
            return None if value is None else not value
        if isinstance(node, ast.BoolOp):
            values = [self._condition(module, value, scope) for value in node.values]
            if isinstance(node.op, ast.And):
                if False in values:
                    return False
                return True if all(value is True for value in values) else None
            if True in values:
                return True
            return False if all(value is False for value in values) else None
        reference = self._typing_condition_reference(module, node, scope)
        return True if reference is not None and reference.name == "TYPE_CHECKING" else None

    def contract_statements(self, body: list[ast.stmt]) -> Iterator[ast.stmt]:
        for node in body:
            yield node
            if isinstance(node, (*_FUNCTION, ast.ClassDef)):
                continue
            if isinstance(node, ast.If):
                # Bound branches only; a guard not bound yet has no known selection.
                condition = self.guards.get(node)
                suites = (
                    [node.body if condition else node.orelse]
                    if condition is not None
                    else [node.body, node.orelse]
                )
                for suite in suites:
                    yield from self.contract_statements(suite)
                continue
            for _, children in ast.iter_fields(node):
                if isinstance(children, list):
                    for child in children:
                        if isinstance(child, ast.stmt):
                            yield from self.contract_statements([child])
                        elif isinstance(child, (ast.ExceptHandler, ast.match_case)):
                            yield from self.contract_statements(child.body)

    def _same_binding(
        self, left: _Binding | _Reference | None, right: _Binding | _Reference | None
    ) -> bool:
        if left is right:
            return True
        if (
            isinstance(left, _Binding)
            and isinstance(right, _Binding)
            and left.uncertainty is None
            and right.uncertainty is None
            and isinstance(left.value, (_Binding, _Reference))
            and left.value is right.value
        ):
            return True

        def direct_typing_reference(
            binding: _Binding | _Reference | None,
            seen: frozenset[_Binding] = frozenset(),
        ) -> _Reference | None:
            if isinstance(binding, _Reference):
                return binding if binding.module in _TYPING else None
            if binding is None or binding.uncertainty is not None or binding in seen:
                return None
            for value in (binding.value, binding.declaration):
                if isinstance(value, (_Binding, _Reference)):
                    reference = direct_typing_reference(value, seen | {binding})
                    if reference is not None:
                        return reference
            return None

        first = direct_typing_reference(left)
        second = direct_typing_reference(right)
        return (
            first is not None
            and second is not None
            and first.module == second.module
            and first.name == second.name
        )

    def _joined_binding(
        self,
        module: _Module,
        name: str,
        control: ast.AST,
        candidates: tuple[_Binding | _Reference | None, ...],
    ) -> _Binding | None:
        first = candidates[0]
        if all(self._same_binding(first, candidate) for candidate in candidates[1:]):
            return (
                first
                if isinstance(first, _Binding)
                else (
                    _Binding(
                        first, first, _Reference(module.owner.module, name, module.owner.scope)
                    )
                    if isinstance(first, _Reference)
                    else None
                )
            )
        owner = next(
            (candidate.owner for candidate in candidates if isinstance(candidate, _Binding)),
            _Reference(module.owner.module, name, module.owner.scope),
        )
        return _Binding(
            control,
            None,
            owner,
            _Uncertainty(control, candidates),
        )

    def _merge_states(
        self,
        module: _Module,
        control: ast.AST,
        states: list[_ModuleState],
    ) -> None:
        declarations: dict[str, _Binding] = {}
        bindings: dict[str, _Binding] = {}
        names = set().union(*(state.declarations for state in states))
        for name in names:
            candidate = self._joined_binding(
                module, name, control, tuple(state.declarations.get(name) for state in states)
            )
            if candidate is not None:
                declarations[name] = candidate
        names = set().union(*(state.bindings for state in states))
        for name in names:
            candidate = self._joined_binding(
                module, name, control, tuple(state.bindings.get(name) for state in states)
            )
            if candidate is not None:
                bindings[name] = candidate
        mutations: dict[_Binding, tuple[_Mutation, ...]] = {}
        for root in set().union(*(state.mutations for state in states)):
            changes = [state.mutations.get(root, ()) for state in states]
            # A change made on only some paths leaves the object's contents unknown.
            mutations[root] = (
                changes[0]
                if all(change == changes[0] for change in changes[1:])
                else (_Mutation(control),)
            )
        ordered_stars = list(dict.fromkeys(star for state in states for star in state.stars))
        common_stars = set(states[0].stars).intersection(
            *(set(state.stars) for state in states[1:])
        )
        uncertain_star_names = dict(states[0].uncertain_star_names)
        for state in states[1:]:
            uncertain_star_names.update(state.uncertain_star_names)
        for star in set(ordered_stars) - common_stars:
            names = _TYPING_NAMES if star in _TYPING else set(self.exports(star))
            for name in names:
                candidates = tuple(
                    state.bindings.get(name)
                    or (_Reference(star, name) if star in state.stars else None)
                    for state in states
                )
                joined = self._joined_binding(module, name, control, candidates)
                if joined is not None and joined.uncertainty is not None:
                    bindings[name] = joined
                    uncertain_star_names[name] = joined.uncertainty
        module.declarations = declarations
        module.bindings = bindings
        module.mutations = mutations
        module.stars = ordered_stars
        module.uncertain_star_names = uncertain_star_names

    def _unknown_target(
        self,
        name: str,
        module: _Module,
        statement: ast.stmt,
        target: ast.expr | str,
        scope: _Scope | None,
    ) -> None:
        if isinstance(target, str):
            module.unknown(target, statement)
            return
        for leaf in _target_leaves(target):
            if isinstance(leaf, ast.Name):
                module.unknown(leaf.id, statement)
            else:
                self.change_containers(name, module, statement, leaf, scope)

    def _bind_control(
        self, name: str, module: _Module, package: str, node: ast.stmt, scope: _Scope | None
    ) -> bool:
        targets: list[list[ast.expr | str]]
        if isinstance(node, ast.If):
            condition = self._condition(name, node.test, scope)
            self.guards[node] = condition
            if condition is not None:
                self._bind_suite(
                    name, module, package, node.body if condition else node.orelse, scope
                )
                return True
            suites = [node.body, node.orelse] if node.orelse else [node.body, []]
            targets = [[] for _ in suites]
        elif isinstance(node, (ast.Try, ast.TryStar)):
            if not node.handlers:
                self._bind_suite(name, module, package, [*node.body, *node.orelse], scope)
                self._bind_suite(name, module, package, node.finalbody, scope)
                return True
            suites = [
                [*node.body, *node.orelse],
                *[handler.body for handler in node.handlers],
            ]
            targets = [[], *[[handler.name] if handler.name else [] for handler in node.handlers]]
        elif isinstance(node, ast.Match):
            suites = [case.body for case in node.cases]
            targets = []
            for case in node.cases:
                captures: list[ast.expr | str] = []
                for pattern in ast.walk(case.pattern):
                    if isinstance(pattern, (ast.MatchAs, ast.MatchStar)) and pattern.name:
                        captures.append(pattern.name)
                    elif isinstance(pattern, ast.MatchMapping) and pattern.rest:
                        captures.append(pattern.rest)
                targets.append(captures)
            if not any(
                isinstance(case.pattern, ast.MatchAs)
                and case.pattern.pattern is None
                and case.guard is None
                for case in node.cases
            ):
                suites.append([])
                targets.append([])
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            suites = [node.body, node.orelse, []]
            targets = (
                [[node.target], [node.target], []]
                if isinstance(node, (ast.For, ast.AsyncFor))
                else [[], [], []]
            )
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                if item.optional_vars is not None:
                    self._unknown_target(name, module, node, item.optional_vars, scope)
            self._bind_suite(name, module, package, node.body, scope)
            return True
        else:
            return False
        base = self._snapshot(module)
        states = []
        for suite, stores in zip(suites, targets, strict=True):
            self._restore(module, base)
            for target in stores:
                self._unknown_target(name, module, node, target, scope)
            self._bind_suite(name, module, package, suite, scope)
            if isinstance(node, (ast.Try, ast.TryStar)):
                # Exception target names are cleared after their handler finishes.
                for target in stores:
                    self._unknown_target(name, module, node, target, scope)
            states.append(self._snapshot(module))
        self._merge_states(module, node, states)
        if isinstance(node, (ast.Try, ast.TryStar)):
            self._bind_suite(name, module, package, node.finalbody, scope)
        return True

    def _named_expressions(
        self, name: str, node: ast.AST, scope: _Scope | None
    ) -> Iterator[ast.NamedExpr]:
        """Stores an expression can evaluate, respecting known short circuits."""
        if isinstance(node, ast.NamedExpr):
            yield from self._named_expressions(name, node.value, scope)
            yield node
            return
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                yield from self._named_expressions(name, value, scope)
                condition = self._condition(name, value, scope)
                if (
                    condition is False
                    and isinstance(node.op, ast.And)
                    or condition is True
                    and isinstance(node.op, ast.Or)
                ):
                    return
            return
        if isinstance(node, ast.IfExp):
            yield from self._named_expressions(name, node.test, scope)
            condition = self._condition(name, node.test, scope)
            branches = (
                (node.body if condition else node.orelse,)
                if condition is not None
                else (node.body, node.orelse)
            )
            for branch in branches:
                yield from self._named_expressions(name, branch, scope)
            return
        children: Sequence[ast.AST]
        if isinstance(node, ast.Lambda):
            children = [
                *node.args.defaults,
                *(value for value in node.args.kw_defaults if value is not None),
            ]
        else:
            children = [
                child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.stmt)
            ]
        for child in children:
            yield from self._named_expressions(name, child, scope)

    def _bind_suite(
        self, name: str, module: _Module, package: str, body: list[ast.stmt], scope: _Scope | None
    ) -> None:
        for node in body:
            for expression in self._named_expressions(name, node, scope):
                module.unknown(expression.target.id, expression)
            if self._bind_control(name, module, package, node, scope):
                continue
            self._bind_statement(name, module, package, node, scope)

    def bind(self, name: str, module: _Module, package: str) -> None:
        scope = module.owner.scope
        assert isinstance(module.tree, (ast.Module, ast.ClassDef, *_FUNCTION))
        if isinstance(module.tree, ast.ClassDef):
            self.bind_type_parameters(module, module.tree)
        self._bind_suite(name, module, package, module.tree.body, scope)

    def _bind_statement(
        self, name: str, module: _Module, package: str, node: ast.stmt, scope: _Scope | None
    ) -> None:
        exports = isinstance(module.tree, ast.Module)
        if isinstance(node, (*_FUNCTION, ast.ClassDef)):
            previous = module.bindings.get(node.name)
            if isinstance(node, _FUNCTION):
                # Decorators run with the definition; rebinding their names later changes nothing.
                self.capture(
                    name,
                    _ValueScope(node),
                    node.decorator_list,
                    scope,
                    defer_future_annotations=False,
                )
            declared = self.declaration_scope(name, node, scope)
            if isinstance(node, ast.ClassDef):
                # Changes the class body made stay visible to the statements after it.
                module.mutations = declared.mutations
            module.define(node.name, node)
            if isinstance(node, _FUNCTION):
                previous_definition = previous.declaration if previous is not None else None
                self.overload_candidates[node] = (
                    (*self.overload_candidates[previous_definition], previous_definition)
                    if isinstance(previous_definition, _FUNCTION)
                    and previous is not None
                    and previous.owner == _Reference(name, node.name, scope)
                    else ()
                )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module.define(
                    alias.asname or alias.name.split(".")[0],
                    _Reference(alias.name if alias.asname else alias.name.split(".")[0], None),
                )
        elif isinstance(node, ast.ImportFrom):
            self.bind_from_import(name, module, package, node)
        elif isinstance(node, ast.Assign):
            self.capture(name, node, [node.value], scope, defer_future_annotations=False)
            if exports and any(
                not isinstance(target, ast.Name)
                and any(
                    isinstance(part, ast.Name) and part.id in _EXPORT_MAPS
                    for part in ast.walk(target)
                )
                for target in node.targets
            ):
                raise ValueError(f"Unsupported export assignment in {name}")
            self.bind_targets(name, module, node, scope)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotation_expressions = [] if self.deferred(name) else [node.annotation]
            self.capture(
                name,
                node,
                annotation_expressions,
                scope,
                defer_future_annotations=False,
            )
            value_scope: _Scope = node
            if node.value is not None:
                value_key = _ValueScope(node)
                self.capture(
                    name,
                    value_key,
                    [node.value],
                    scope,
                    defer_future_annotations=False,
                )
                value_scope = value_key
            binding_owner = _Reference(name, node.target.id, value_scope)
            module.define(
                node.target.id,
                node,
                self.copied_binding(name, node.value, scope, binding_owner),
                owner=binding_owner,
            )
        elif isinstance(node, ast.TypeAlias):
            self.declaration_scope(name, node, scope)
            module.define(node.name.id, node, owner=_Reference(name, node.name.id, node))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(module.tree, _FUNCTION):
                expressions = [] if self.deferred(name) else [node.annotation]
                self.capture(name, node, expressions, scope, defer_future_annotations=False)
            if node.value is not None:
                self.change_containers(name, module, node, node.target, scope)
        elif isinstance(node, ast.AugAssign):
            self.bind_augmented(name, module, node, scope)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                self._unknown_target(name, module, node, target, scope)
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
        ):
            self.bind_method_call(name, module, node, node.value, node.value.func, scope)

    def bind_targets(
        self, name: str, module: _Module, node: ast.Assign, scope: _Scope | None
    ) -> None:
        """Store every target's value, all evaluated before the first name is rebound."""
        received = [
            (
                target,
                value,
                exact,
                self.copied_binding(name, value, scope, _Reference(name, target.id, node))
                if isinstance(target, ast.Name)
                else None,
            )
            for whole in node.targets
            for target, value, exact in _unpacked(whole, node.value)
        ]
        for target, value, exact, copied in received:
            if not isinstance(target, ast.Name):
                self.change_containers(name, module, node, target, scope)
                continue
            owner = _Reference(name, target.id, node)
            if exact:
                module.define(target.id, value, copied, owner=owner)
                continue
            # A name receiving an unknown part of a value holds unknown data from it.
            unpacked = _Binding(value, value if copied is None else copied, owner)
            module.assign(target.id, _Binding(node, None, owner, _Uncertainty(node, (unpacked,))))

    def bind_augmented(
        self, name: str, module: _Module, node: ast.AugAssign, scope: _Scope | None
    ) -> None:
        target = node.target
        if not isinstance(target, ast.Name):
            if isinstance(module.tree, ast.Module) and any(
                isinstance(part, ast.Name) and part.id in _EXPORT_MAPS for part in ast.walk(target)
            ):
                raise ValueError(f"Unsupported export mutation in {name}")
            self.change_containers(name, module, node, target, scope)
            return
        value_scope = _ValueScope(node)
        root = self.container_root(self.lookup(name, target.id, scope))
        kind = self.container_type(root)
        if root is not None and kind is not None:
            # A list or mapping changes in place, so every name sharing it sees the change.
            supported = (
                kind == "list"
                and isinstance(node.op, ast.Add)
                or kind == "dict"
                and isinstance(node.op, ast.BitOr)
            )
            if supported:
                self.capture(name, value_scope, [node.value], scope, defer_future_annotations=False)
            module.change(root, _Mutation(node, node.value if supported else None, value_scope))
            return
        # Other values rebind the name to a new object built from the current one.
        value = ast.copy_location(
            ast.BinOp(left=ast.Name(id=target.id, ctx=ast.Load()), op=node.op, right=node.value),
            node,
        )
        self.capture(name, value_scope, [value], scope, defer_future_annotations=False)
        module.define(target.id, value, owner=_Reference(name, target.id, value_scope))

    def bind_method_call(
        self,
        name: str,
        module: _Module,
        node: ast.Expr,
        call: ast.Call,
        method: ast.Attribute,
        scope: _Scope | None,
    ) -> None:
        root = self.changed_container(name, method, scope)
        if root is None:
            if (
                isinstance(module.tree, ast.Module)
                and isinstance(method.value, ast.Name)
                and method.value.id in _EXPORT_MAPS
            ):
                raise ValueError(f"Unsupported export mutation in {name}")
            return
        value_scope = _ValueScope(node)
        operand: ast.expr | None = None
        if (
            isinstance(method.value, ast.Name)
            and method.attr in {"append", "extend"}
            and len(call.args) == 1
            and not isinstance(call.args[0], ast.Starred)
            and not call.keywords
            and self.container_type(root) == "list"
        ):
            self.capture(name, value_scope, call.args, scope, defer_future_annotations=False)
            operand = (
                ast.List(elts=[call.args[0]], ctx=ast.Load())
                if method.attr == "append"
                else call.args[0]
            )
        # Other methods can change the container in ways exports cannot follow.
        module.change(root, _Mutation(node, operand, value_scope))

    def change_containers(
        self, name: str, module: _Module, node: ast.stmt, target: ast.expr, scope: _Scope | None
    ) -> None:
        """Item and attribute stores change a tracked container beyond static evaluation."""
        for part in ast.walk(target):
            if isinstance(part, (ast.Subscript, ast.Attribute)) and isinstance(
                part.ctx, (ast.Store, ast.Del)
            ):
                root = self.changed_container(name, part, scope)
                if root is not None:
                    module.change(root, _Mutation(node))

    def changed_container(
        self, module: str, node: ast.expr, scope: _Scope | None
    ) -> _Binding | None:
        """The tracked list or mapping that a change through `node` reaches."""
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        root = self.container_root(self.lookup(module, node.id, scope))
        return root if self.container_type(root) is not None else None

    def container_root(self, binding: _Binding | _Reference | None) -> _Binding | None:
        """The binding whose expression created the object a name refers to."""
        seen: set[_Binding] = set()
        while isinstance(binding, _Binding) and binding.uncertainty is None and binding not in seen:
            seen.add(binding)
            expression = _assigned_value(binding)
            if not (isinstance(expression, ast.Name) and isinstance(binding.value, _Binding)):
                return binding
            binding = binding.value
        return None

    def container_type(
        self, binding: _Binding | None, seen: frozenset[_Binding] = frozenset()
    ) -> str | None:
        """The mutable builtin that a display, constructor call or concatenation creates."""
        if binding is None or binding in seen:
            return None
        expression = _assigned_value(binding)
        while isinstance(expression, ast.BinOp):
            expression = expression.left
        if isinstance(expression, ast.Name):
            # Concatenation creates an object of its left operand's type.
            operand = self.lookup(binding.owner.module, expression.id, binding.owner.scope)
            return self.container_type(self.container_root(operand), seen | {binding})
        if isinstance(expression, (ast.List, ast.ListComp)):
            return "list"
        if isinstance(expression, (ast.Dict, ast.DictComp)):
            return "dict"
        if isinstance(expression, (ast.Set, ast.SetComp)):
            return "set"
        if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
            constructors = {"list": "list", "sorted": "list", "dict": "dict", "set": "set"}
            return constructors.get(expression.func.id)
        return None

    def mutations_seen(
        self, module: str, scope: _Scope | None
    ) -> dict[_Binding, tuple[_Mutation, ...]]:
        """In-place changes visible to an expression evaluated in `scope`."""
        eager = isinstance(scope, (ast.Assign, _ValueScope)) or (
            isinstance(scope, (ast.AnnAssign, *_FUNCTION)) and not self.deferred(module)
        )
        # Eager captures keep the changes made before them; other scopes evaluate late.
        if eager:
            return self.module(module, scope).mutations
        return self.body_changes(module, scope, late=True)

    def body_changes(
        self, module: str, scope: _Scope | None, *, late: bool = False
    ) -> dict[_Binding, tuple[_Mutation, ...]]:
        """In-place changes made so far by the module, class or function body running `scope`."""
        # A late read sees the whole module or function body, including its class bodies.
        while scope is not None and not isinstance(scope, _Body):
            if isinstance(scope, ast.ClassDef) and not late:
                break
            parent = self.module(module, scope).parent
            scope = None if parent is None else parent.scope
        return self.module(module, scope).mutations

    def changed_before(self, reader: _Reference, root: _Binding) -> bool:
        """Whether a container changed in place before the scope `reader` evaluated it."""
        owner = root.owner.module
        # An imported module has finished running before its importer reads it.
        log = (
            self.mutations_seen(reader.module, reader.scope)
            if owner == reader.module
            else self.module(owner).mutations
        )
        return root in log

    def literal(
        self,
        name: str,
        node: ast.expr,
        seen: frozenset[_Binding] = frozenset(),
        scope: _Scope | None = None,
    ) -> object:
        if isinstance(node, ast.Name):
            # A name holds the object bound when the enclosing expression ran.
            binding = self.lookup(name, node.id, scope)
            if isinstance(binding, _Binding):
                return self.export_value(name, binding, self.mutations_seen(name, scope), seen)
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

    def export_value(
        self,
        module: str,
        binding: _Binding,
        mutations: dict[_Binding, tuple[_Mutation, ...]],
        seen: frozenset[_Binding] = frozenset(),
    ) -> object:
        """The value of a name's object, with in-place changes made before it is read."""
        if binding.uncertainty is not None:
            raise ValueError(
                f"Ambiguous export map binding {binding.owner.module}.{binding.owner.name}"
            )
        if binding in seen or binding.owner.module != module:
            raise ValueError(f"Cannot statically resolve exports in {module}: {binding.owner.name}")
        seen = seen | {binding}
        expression = _assigned_value(binding)
        if isinstance(expression, ast.Name) and isinstance(binding.value, _Binding):
            # Assignment shares the object, so its later in-place changes stay visible.
            return self.export_value(module, binding.value, mutations, seen)
        if not isinstance(expression, ast.expr):
            raise ValueError(f"Cannot statically resolve exports in {module}: {binding.owner.name}")
        value = self.literal(module, expression, seen, binding.owner.scope)
        for mutation in mutations.get(binding, ()):
            operand = (
                None
                if mutation.operand is None
                else self.literal(module, mutation.operand, seen, mutation.scope)
            )
            if isinstance(value, list) and isinstance(operand, list):
                value = value + operand
            elif isinstance(value, dict) and isinstance(operand, dict):
                value = value | operand
            else:
                line = getattr(mutation.statement, "lineno", 0)
                raise ValueError(f"Unsupported export mutation in {module} at line {line}")
        return value

    def exports(self, name: str) -> list[str]:
        source = self.module(name)
        binding = source.bindings.get("__all__")
        if binding is None:
            # Explicit `as` re-exports cover older facades without __all__.
            names = [
                alias.asname
                for statement in self.contract_statements(cast(ast.Module, source.tree).body)
                if isinstance(statement, ast.ImportFrom)
                for alias in statement.names
                if alias.asname is not None and not alias.asname.startswith("_")
            ]
            if not names:
                raise ValueError(f"Missing __all__ or explicit re-exports in {name}")
            return names
        names = self.export_value(name, binding, source.mutations)
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            raise ValueError(f"Invalid __all__ in {name}")
        return cast(list[str], names)

    def require_certain(self, binding: _Binding | _Reference | None) -> None:
        if isinstance(binding, _Binding) and binding.uncertainty is not None:
            control = binding.uncertainty.control
            line = getattr(control, "lineno", 0)
            raise ValueError(
                f"Ambiguous public typing binding "
                f"{binding.owner.module}.{binding.owner.name} at line {line}"
            )

    def class_scope(
        self,
        target: _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
    ) -> _Reference | None:
        while target is not None:
            self.require_certain(target)
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
        if name in source.uncertain_star_names:
            uncertainty = source.uncertain_star_names[name]
            return _Binding(
                uncertainty.control,
                None,
                _Reference(module, name, scope),
                uncertainty,
            )
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
        *,
        member: bool = False,
    ) -> tuple[_Reference, ast.AST | _Binding | _Reference | None]:
        """Follow a name to its contract; a class `member` may end at an external object."""
        key = _Reference(module, name, scope)
        if key in seen:
            raise ValueError(f"Cyclic public re-export: {module}.{name}")
        seen = seen | {key}
        source = self.module(module, scope)
        record = source.declarations.get(name)
        if record is not None and record.uncertainty is not None:
            return key, record
        if name in source.uncertain_star_names:
            uncertainty = source.uncertain_star_names[name]
            return key, _Binding(
                uncertainty.control, None, _Reference(module, name, scope), uncertainty
            )
        binding: ast.AST | _Binding | _Reference | None = None
        for map_name in ("_LAZY_IMPORTS", "_EXPORTS") if scope is None else ():
            if map_name in source.bindings:
                exports = self.export_value(module, source.bindings[map_name], source.mutations)
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
        if binding is None and record is not None:
            key, record = self.assigned_origin(key, record)
            binding = self.contract(record)
        if binding is None:
            for star in reversed(source.stars):
                if star in _TYPING:
                    if name in _TYPING_NAMES:
                        if member:
                            return key, _Reference(star, name)
                        raise ValueError(f"No Python source for public export {star}.{name}")
                    continue
                if name in self.exports(star):
                    return self.resolve(star, name, seen, member=member)
            if scope is None and self.path(f"{module}.{name}") is not None:
                return _Reference(f"{module}.{name}", None), None
            raise ValueError(f"Unresolved public export: {module}.{name}")
        if isinstance(binding, _Reference):
            if member and self.path(binding.module) is None:
                return key, binding
            if binding.name is None:
                if self.path(binding.module) is None:
                    raise ValueError(f"No Python source for public module {binding.module}")
                return _Reference(binding.module, None, binding.scope), None
            if self.path(binding.module) is None:
                raise ValueError(
                    f"No Python source for public export {binding.module}.{binding.name}"
                )
            return self.resolve(binding.module, binding.name, seen, binding.scope, member=member)
        return key, binding

    def assigned_origin(self, key: _Reference, record: _Binding) -> tuple[_Reference, _Binding]:
        """Follow assigned names to the binding that held their value at that point."""
        while isinstance(record.declaration, (ast.Name, ast.Attribute)) and isinstance(
            record.value, _Binding
        ):
            record = record.value
            key = record.owner
        return key, record

    def contract(self, record: _Binding) -> ast.AST | _Binding | _Reference:
        """Classify a binding as a type alias, a definition, an import or runtime data."""
        binding = record.declaration
        owner = record.owner
        if isinstance(binding, (ast.Name, ast.Attribute)):
            target = (
                record.value
                if isinstance(record.value, (_Reference, _Binding))
                else self.reference(owner.module, binding, owner.scope, allow_forward_strings=False)
            )
            if target is not None and self.is_type_expression(
                owner.module, target, scope=owner.scope, allow_forward_strings=False
            ):
                return record
            return target or binding
        if isinstance(binding, (_Reference, ast.ClassDef, *_FUNCTION)):
            return binding
        # Declared aliases count through their own record, not a later name lookup.
        if isinstance(binding, (ast.TypeAlias, ast.AnnAssign)):
            return record
        if not self.ordinary_uncertainty(record) and self.is_type_expression(
            owner.module, record, scope=owner.scope, allow_forward_strings=False
        ):
            return record
        return binding

    def member_contract(
        self, key: _Reference, record: _Binding
    ) -> tuple[_Reference, ast.AST | _Binding | _Reference | None]:
        """Interpret an assigned class attribute; imports follow local sources to their objects."""
        key, record = self.assigned_origin(key, record)
        node = self.contract(record)
        if isinstance(node, _FUNCTION):
            # A copied function keeps the scope its signature was evaluated in.
            return record.owner, node
        if not isinstance(node, _Reference):
            return key, node
        if node is not record.declaration or node.name is None:
            return key, None
        if self.path(node.module) is None:
            return key, node
        return self.resolve(node.module, node.name, scope=node.scope, member=True)

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
            self.require_certain(binding)
            module, scope = binding.owner.module, binding.owner.scope
            binding = binding.declaration
        if isinstance(binding, ast.TypeAlias):
            return binding.value
        if isinstance(binding, ast.AnnAssign):
            if binding.value is None:
                return None
            annotation_scope = binding if binding in self.scopes else scope
            marker = self.reference(module, binding.annotation, annotation_scope)
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
        self.require_certain(binding)
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
        reader: _Reference,
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
                self.require_certain(node)
            if isinstance(node, _Binding):
                module, scope = node.owner.module, node.owner.scope
                if self.changed_before(reader, node):
                    raise ValueError(
                        f"Cannot statically resolve typing fields changed in place: "
                        f"{module}.{node.owner.name}"
                    )
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
                reader=reader,
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
                    reader=reader,
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
                        reader=reader,
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
                        reader=reader,
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
                reader=reader,
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
        # Field containers hold the contents they had when this call evaluated them.
        reader = _Reference(module, None, scope)
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
                    reader=reader,
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
                            reader=reader,
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
            self.require_certain(node)
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
            self.require_certain(binding)
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
            # Reaching an alias again inside a type constructor is finite recursion.
            productive = productive | frozenset(
                item
                for item in seen
                if isinstance(item, _Binding)
                and self.type_value(item.owner.module, item, item.owner.scope) is not None
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

    def is_overload(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Whether a definition registered an overload, by the decorators bound when it ran."""
        decorators = _ValueScope(node)
        module = self.scopes[decorators].owner.module
        return any(
            self.typing_name(self.reference(module, decorator, decorators)) == "overload"
            for decorator in node.decorator_list
        )

    def property_accessor(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> tuple[str, _Binding] | None:
        """The accessor slot and property value captured by an accessor decorator."""
        decorators = self.scopes[_ValueScope(node)]
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if (
                isinstance(target, ast.Attribute)
                and target.attr in {"getter", "setter", "deleter"}
                and isinstance(target.value, ast.Name)
                and isinstance(previous := decorators.bindings.get(target.value.id), _Binding)
            ):
                return target.attr, previous
        return None

    def property_functions(
        self, module: str, node: ast.FunctionDef | ast.AsyncFunctionDef, scope: _Scope | None
    ) -> dict[str, tuple[_Reference, ast.FunctionDef | ast.AsyncFunctionDef]]:
        """Keep each property's latest accessor, including when cloned under another name."""
        functions: dict[str, tuple[_Reference, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
        owner = _Reference(module, node.name, scope)
        while True:
            accessor = self.property_accessor(node)
            slot = accessor[0] if accessor is not None else "getter"
            functions.setdefault(slot, (owner, node))
            if accessor is None:
                return functions
            previous = accessor[1]
            owner, definition = self.member_contract(previous.owner, previous)
            if not isinstance(definition, _FUNCTION):
                return functions
            node = definition

    def is_type_expression(
        self,
        module: str,
        node: ast.AST | _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        scope: _Scope | None = None,
        *,
        allow_forward_strings: bool = True,
    ) -> bool:
        if isinstance(node, _Binding):
            self.require_certain(node)
            if node in seen:
                raise ValueError(f"Cyclic annotation alias: {node.owner.module}.{node.owner.name}")
            allow_forward_strings = allow_forward_strings or self.declared_alias(node)
            declaration = node.declaration
            if isinstance(declaration, (ast.ClassDef, ast.TypeAlias)):
                return True
            value = (
                node.value
                if isinstance(node.value, (_Reference, _Binding))
                else declaration
                if isinstance(declaration, _Reference)
                else self.type_value(node.owner.module, node, node.owner.scope, seen | {node})
            )
            return value is not None and self.is_type_expression(
                node.owner.module,
                value,
                seen | {node},
                node.owner.scope,
                allow_forward_strings=allow_forward_strings,
            )
        if isinstance(node, ast.Constant):
            if node.value is None:
                return True
            if isinstance(node.value, str) and allow_forward_strings:
                reference = self.reference(
                    module, node, scope, allow_forward_strings=allow_forward_strings
                )
                return reference is not None and self.is_type_expression(
                    module,
                    reference,
                    seen,
                    scope,
                    allow_forward_strings=allow_forward_strings,
                )
            return False
        if isinstance(node, ast.Call):
            ref = self.reference(
                module, node.func, scope, allow_forward_strings=allow_forward_strings
            )
            return (
                self.typing_name(ref, allow_forward_strings=allow_forward_strings)
                in _TYPING_FACTORIES
            )
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ast.BitOr):
                return False
            left_type = self.is_type_expression(
                module, node.left, seen, scope, allow_forward_strings=allow_forward_strings
            )
            right_type = self.is_type_expression(
                module, node.right, seen, scope, allow_forward_strings=allow_forward_strings
            )
            return (
                left_type
                and (
                    right_type
                    or self.is_opaque_external_type_operand(
                        module,
                        node.right,
                        seen,
                        scope,
                        allow_forward_strings=allow_forward_strings,
                    )
                )
            ) or (
                right_type
                and self.is_opaque_external_type_operand(
                    module,
                    node.left,
                    seen,
                    scope,
                    allow_forward_strings=allow_forward_strings,
                )
            )
        if isinstance(node, ast.Subscript):
            return self.is_type_expression(
                module, node.value, seen, scope, allow_forward_strings=allow_forward_strings
            ) or (
                self.is_opaque_external_type_operand(
                    module,
                    node.value,
                    seen,
                    scope,
                    allow_forward_strings=allow_forward_strings,
                )
                and self.is_external_type_argument(
                    module,
                    node.slice,
                    seen,
                    scope,
                    allow_forward_strings=allow_forward_strings,
                )
            )
        # A complete type expression is never a tuple; tuples only list type arguments.
        ref = self.reference(module, node, scope, allow_forward_strings=allow_forward_strings)
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
        return self.is_type_expression(
            ref.module,
            binding,
            seen | {ref},
            ref.scope,
            allow_forward_strings=allow_forward_strings,
        )

    def is_opaque_external_type_operand(
        self,
        module: str,
        node: ast.AST | _Reference | _Binding | None,
        seen: frozenset[_Reference | _Binding] = frozenset(),
        scope: _Scope | None = None,
        *,
        allow_forward_strings: bool = True,
    ) -> bool:
        """Recognize an opaque imported member only in a type-shaped context."""
        while node is not None:
            if isinstance(node, _Binding):
                if node in seen:
                    raise ValueError(
                        f"Cyclic annotation alias: {node.owner.module}.{node.owner.name}"
                    )
                allow_forward_strings = allow_forward_strings or self.declared_alias(node)
                seen = seen | {node}
                module, scope = node.owner.module, node.owner.scope
                node = (
                    node.declaration
                    if isinstance(node.declaration, _Reference)
                    else self.type_value(module, node, scope, seen)
                )
                continue
            ref = self.reference(module, node, scope, allow_forward_strings=allow_forward_strings)
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
        *,
        allow_forward_strings: bool = True,
    ) -> bool:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if not allow_forward_strings:
                return False
            value = node.value
            try:
                node = ast.parse(value, mode="eval").body
            except SyntaxError:
                return False
            return self.is_external_type_argument(
                module, node, seen, scope, allow_forward_strings=allow_forward_strings
            )
        if isinstance(node, ast.Tuple):
            return bool(node.elts) and all(
                self.is_external_type_argument(
                    module, element, seen, scope, allow_forward_strings=allow_forward_strings
                )
                for element in node.elts
            )
        return self.is_type_expression(
            module, node, seen, scope, allow_forward_strings=allow_forward_strings
        ) or self.is_opaque_external_type_operand(
            module, node, seen, scope, allow_forward_strings=allow_forward_strings
        )

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

    def ordinary_candidate(
        self, candidate: _Binding | _Reference | None, seen: frozenset[_Binding] = frozenset()
    ) -> bool:
        if candidate is None or isinstance(candidate, _Reference):
            return False
        if candidate in seen:
            return False
        if candidate.uncertainty is not None:
            return all(
                self.ordinary_candidate(value, seen | {candidate})
                for value in candidate.uncertainty.candidates
            )
        declaration = candidate.declaration
        if isinstance(declaration, (ast.ClassDef, ast.TypeAlias, *_FUNCTION, ast.AnnAssign)):
            return False
        if isinstance(declaration, (ast.Name, ast.Attribute)) and isinstance(
            candidate.value, (_Binding, _Reference)
        ):
            return self.ordinary_candidate(candidate.value, seen | {candidate})
        return self.ordinary_expression(candidate.owner.module, declaration, candidate.owner.scope)

    def ordinary_uncertainty(self, binding: _Binding | None) -> bool:
        return (
            binding is not None
            and binding.uncertainty is not None
            and all(
                self.ordinary_candidate(candidate) for candidate in binding.uncertainty.candidates
            )
        )

    def ordinary_expression(
        self, module: str, node: ast.AST | _Reference | None, scope: _Scope | None
    ) -> bool:
        if node is None or isinstance(node, _Reference):
            return False
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Call):
            reference = self.reference(module, node.func, scope, allow_forward_strings=False)
            return self.typing_name(reference) not in _TYPING_FACTORIES
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            return all(self.ordinary_expression(module, item, scope) for item in node.elts)
        if isinstance(node, ast.Dict):
            return all(
                (key is None or self.ordinary_expression(module, key, scope))
                and self.ordinary_expression(module, value, scope)
                for key, value in zip(node.keys, node.values, strict=True)
            )
        return False

    def sites(
        self,
        module: str,
        node: ast.AST | _Binding | _Reference,
        name: str | None = None,
        scope: _Scope | None = None,
    ) -> dict[str, int]:
        if isinstance(node, _Reference):
            # An external object is opaque, but a typing name is a type expression.
            return {"alias": self.annotation(module, node)} if node.module in _TYPING else {}
        record = node if isinstance(node, _Binding) else None
        if record is not None and record.uncertainty is not None:
            if self.ordinary_uncertainty(record):
                return {}
            self.require_certain(record)
        if record is not None:
            declaration = record.declaration
            if isinstance(declaration, ast.Call):
                return (
                    self.factory_sites(
                        record.owner.module, declaration, frozenset(), record.owner.scope
                    )
                    or {}
                )
            if not isinstance(declaration, (ast.TypeAlias, ast.AnnAssign)):
                # A bare implicit alias such as `Alias = Any` has no other definition.
                return {"alias": self.annotation(module, record, scope=scope)}
            node = declaration
        if isinstance(node, _FUNCTION):
            sites: dict[str, int] = {}
            for owner, definition in self.property_functions(module, node, scope).values():
                current = self.signature(owner.module, definition, owner.scope)
                # Overloads stay with the definition they preceded, under any later name.
                overloads = []
                for candidate in reversed(self.overload_candidates[definition]):
                    if not self.is_overload(candidate):
                        break
                    overloads.append(candidate)
                overloads.reverse()
                for index, candidate in enumerate(overloads, 1):
                    current.update(
                        {
                            f"overload:{index}.{site}": count
                            for site, count in self.signature(
                                owner.module, candidate, owner.scope
                            ).items()
                        }
                    )
                for site, count in current.items():
                    sites[site] = sites.get(site, 0) + count
            return sites
        if isinstance(node, ast.AnnAssign):
            # TypeAlias's value is itself an annotation, not an implementation value.
            annotation_scope = node if node in self.scopes else scope
            annotation = self.type_value(module, node, annotation_scope)
            if annotation is not None:
                if record is None and isinstance(node.target, ast.Name):
                    record = self.module(module, scope).declarations.get(node.target.id)
                target = record if record is not None and record.declaration is node else annotation
                return {"field": self.annotation(module, target, scope=annotation_scope)}
            return {"field": self.annotation(module, node.annotation, scope=annotation_scope)}
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
        for member in self.contract_statements(node.body):
            # Every field annotation stays declared; other bindings count only if final.
            occurrence: ast.AST | None = None
            if isinstance(member, (ast.Assign, ast.Import, ast.ImportFrom)):
                names = _bound_names(member)
            elif isinstance(member, (*_FUNCTION, ast.ClassDef)):
                names = [member.name]
                if isinstance(member, _FUNCTION) and not any(
                    isinstance(decorator, ast.Name) and decorator.id == "staticmethod"
                    for decorator in member.decorator_list
                ):
                    positional = [*member.args.posonlyargs, *member.args.args]
                    receiver = positional[0].arg if positional else "self"
                    self.function_body(module, member, scope)
                    for statement in _receiver_body_statements(member.body):
                        if isinstance(statement, ast.AnnAssign) and statement in self.scopes:
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
                names, occurrence = [member.target.id], member
            elif isinstance(member, ast.TypeAlias):
                names = [member.name.id]
            else:
                continue
            for name in names:
                if not name.startswith("_") or (name.startswith("__") and name.endswith("__")):
                    occurrences = members.setdefault(name, [])
                    if occurrence is not None:
                        occurrences.append(occurrence)
        for name, definitions in members.items():
            record = source.declarations.get(name)
            if record is not None and record.uncertainty is not None:
                if self.ordinary_uncertainty(record):
                    continue
                self.require_certain(record)
            if record is not None and not any(
                member is record.declaration for member in definitions
            ):
                # The final binding is the attribute's contract; values and definitions it
                # replaced and an annotated attribute's initializer add no sites.
                definitions = [*definitions, record]
            for member in definitions:
                owner, definition = (
                    self.member_contract(_Reference(module, name, scope), member)
                    if isinstance(member, _Binding)
                    else (_Reference(module, name, scope), member)
                )
                if definition is None:
                    continue
                for site, count in self.sites(owner.module, definition, name, owner.scope).items():
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
