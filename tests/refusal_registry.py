from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import symtable
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import TypeGuard

import increment
from increment.errors import RefusalSpec, WarningSpec, refuse


def walk_refusal_registries() -> dict[str, list[tuple[str, RefusalSpec | WarningSpec]]]:
    """Collect module-level coded specs by public code, preserving their owners."""
    registrations: dict[str, list[tuple[str, RefusalSpec | WarningSpec]]] = {}
    spec_types = (RefusalSpec, WarningSpec)
    for _finder, name, _ispkg in pkgutil.walk_packages(increment.__path__, prefix="increment."):
        module = importlib.import_module(name)
        for attr, value in vars(module).items():
            if isinstance(value, spec_types):
                registrations.setdefault(value.code, []).append((f"{name}.{attr}", value))
            elif isinstance(value, dict):
                for key, spec in value.items():
                    if isinstance(spec, spec_types):
                        registrations.setdefault(spec.code, []).append(
                            (f"{name}.{attr}[{key!r}]", spec)
                        )
    return registrations


@dataclass(frozen=True, slots=True)
class RaiseSite:
    line: int
    registry: Mapping[str, RefusalSpec]
    lookup_key: str | None
    keyword_names: frozenset[str] | None
    binding_error: str | None = None


def _module_binding(module: ModuleType, name: str) -> object:
    return vars(module).get(name)


def _is_registry(value: object) -> TypeGuard[Mapping[str, RefusalSpec]]:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and isinstance(spec, RefusalSpec) for key, spec in value.items()
    )


def _raiser_closure_registry(value: object) -> Mapping[str, RefusalSpec] | None:
    closure = getattr(value, "__closure__", None)
    if closure is None:
        return None
    for cell in closure:
        try:
            candidate = cell.cell_contents
        except ValueError:
            continue
        if _is_registry(candidate):
            return candidate
    return None


def _dispatcher_registry(
    value: object, definition: ast.FunctionDef | ast.AsyncFunctionDef | None = None
) -> Mapping[str, RefusalSpec] | None:
    """Recognize the direct global-registry dispatcher with unchanged context."""
    if not isinstance(value, FunctionType):
        return None
    namespace = value.__globals__
    if definition is None:
        try:
            parsed = ast.parse(textwrap.dedent(inspect.getsource(value)))
        except (OSError, TypeError, SyntaxError):
            return None
        if len(parsed.body) != 1 or not isinstance(
            parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            return None
        definition = parsed.body[0]
    body = definition.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], (ast.Expr, ast.Return)):
        return None
    call = body[0].value
    parameters = (*definition.args.posonlyargs, *definition.args.args)
    match call:
        case ast.Call(
            func=ast.Name(id=callee),
            args=[ast.Subscript(value=ast.Name(id=registry_name), slice=ast.Name(id=key))],
            keywords=[ast.keyword(arg=None, value=ast.Name(id=context))],
        ):
            if (
                namespace.get(callee) is not refuse
                or not parameters
                or key != parameters[0].arg
                or definition.args.kwarg is None
                or context != definition.args.kwarg.arg
            ):
                return None
            registry = namespace.get(registry_name)
            return registry if _is_registry(registry) else None
        case _:
            return None


def _is_raiser_import(node: ast.AST) -> TypeGuard[ast.ImportFrom]:
    if not isinstance(node, ast.ImportFrom) or node.module != "increment.errors":
        return False
    return any(alias.name == "raiser" for alias in node.names)


def _table_kind(table: symtable.SymbolTable) -> str:
    return str(table.get_type())


ScopeFrame = tuple[symtable.SymbolTable | None, frozenset[str]]
type ScopeNode = (
    ast.FunctionDef
    | ast.AsyncFunctionDef
    | ast.ClassDef
    | ast.Lambda
    | ast.ListComp
    | ast.SetComp
    | ast.DictComp
    | ast.GeneratorExp
)


def _scope_table_name(node: ScopeNode) -> tuple[str, str] | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name, "function"
    if isinstance(node, ast.ClassDef):
        return node.name, "class"
    if isinstance(node, ast.Lambda):
        return "lambda", "function"
    if isinstance(node, ast.GeneratorExp):
        return "genexpr", "function"
    if isinstance(node, ast.ListComp):
        return "listcomp", "function"
    if isinstance(node, ast.SetComp):
        return "setcomp", "function"
    if isinstance(node, ast.DictComp):
        return "dictcomp", "function"
    return None


def _child_scope_table(
    parent: symtable.SymbolTable,
    node: ScopeNode,
    used: dict[tuple[int, int, str, str], int],
    *,
    optional: bool = False,
) -> symtable.SymbolTable | None:
    scope_name = _scope_table_name(node)
    if scope_name is None:
        return None
    name, kind = scope_name
    key = (id(parent), node.lineno, name, kind)
    children: list[symtable.SymbolTable] = []
    for child in parent.get_children():
        if _table_kind(child) in ("type parameter", "type parameters"):
            children.extend(child.get_children())
        else:
            children.append(child)
    matches = [
        child
        for child in children
        if child.get_lineno() == node.lineno
        and child.get_name() == name
        and _table_kind(child) == kind
    ]
    index = used.get(key, 0)
    if index < len(matches):
        used[key] = index + 1
        return matches[index]
    if optional:
        return None
    raise RuntimeError(f"line {node.lineno}: Python symbol table has no {kind} scope for {name!r}")


def _scope_is_shadowed(name: str, scopes: Sequence[ScopeFrame]) -> bool:
    for index in range(len(scopes) - 1, -1, -1):
        table, local_names = scopes[index]
        if name in local_names:
            return True
        if table is None:
            continue
        if _table_kind(table) == "class" and any(
            inner_table is None or _table_kind(inner_table) != "module"
            for inner_table, _inner_names in scopes[index + 1 :]
        ):
            # Function and comprehension scopes do not close over class locals.
            continue
        try:
            symbol = table.lookup(name)
        except KeyError:
            continue
        if _table_kind(table) == "module" or symbol.is_global():
            return False
        if symbol.is_local() or symbol.is_nonlocal() or symbol.is_free():
            return True
    return False


class _RaiseSiteVisitor(ast.NodeVisitor):
    def __init__(
        self,
        module: ModuleType,
        root_scope: symtable.SymbolTable,
        raise_registries: Mapping[str, Mapping[str, RefusalSpec] | None],
        binding_errors: Mapping[str, str],
        string_constants: Mapping[str, str],
    ) -> None:
        self.module = module
        self.raise_registries = raise_registries
        self.binding_errors = binding_errors
        self.string_constants = string_constants
        self.sites: list[RaiseSite] = []
        self.scopes: list[ScopeFrame] = [(root_scope, frozenset())]
        self.used_scope_tables: dict[tuple[int, int, str, str], int] = {}

    def _current_table(self) -> symtable.SymbolTable:
        return next(table for table, _local_names in reversed(self.scopes) if table is not None)

    def _enter_scope(self, node: ScopeNode) -> None:
        table = _child_scope_table(self._current_table(), node, self.used_scope_tables)
        assert table is not None
        self.scopes.append((table, frozenset()))

    def _visit_comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp
    ) -> None:
        generators = node.generators
        self.visit(generators[0].iter)
        table = _child_scope_table(
            self._current_table(),
            node,
            self.used_scope_tables,
            optional=isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)),
        )
        local_names = frozenset(
            target.id
            for generator in generators
            for target in ast.walk(generator.target)
            if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store)
        )
        self.scopes.append((table, frozenset() if table is not None else local_names))
        for index, generator in enumerate(generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.scopes.pop()

    visit_ListComp = _visit_comprehension
    visit_SetComp = _visit_comprehension
    visit_DictComp = _visit_comprehension
    visit_GeneratorExp = _visit_comprehension

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ):
            if arg is not None and arg.annotation is not None:
                self.visit(arg.annotation)
        if node.returns is not None:
            self.visit(node.returns)
        self._enter_scope(node)
        for child in node.body:
            self.visit(child)
        self.scopes.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default is not None:
                self.visit(default)
        self._enter_scope(node)
        self.visit(node.body)
        self.scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._enter_scope(node)
        for child in node.body:
            self.visit(child)
        self.scopes.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            self.generic_visit(node)
            return
        name = node.func.id
        if _scope_is_shadowed(name, self.scopes):
            self.generic_visit(node)
            return
        if name in self.binding_errors:
            self.sites.append(
                RaiseSite(node.lineno, {}, None, _keyword_names(node), self.binding_errors[name])
            )
        else:
            if name in self.raise_registries:
                registry = self.raise_registries[name]
            else:
                value = _module_binding(self.module, name)
                registry = _raiser_closure_registry(value)
                if registry is None:
                    registry = _dispatcher_registry(value)
            if registry is not None:
                key = (
                    _resolve_string(node.args[0], self.string_constants)
                    if node.args
                    and not (
                        isinstance(node.args[0], ast.Name)
                        and _scope_is_shadowed(node.args[0].id, self.scopes)
                    )
                    else None
                )
                self.sites.append(
                    RaiseSite(
                        node.lineno,
                        registry,
                        key,
                        _keyword_names(node),
                    )
                )
        self.generic_visit(node)


def resolve_literal_raise_sites(module_name: str, source: str) -> tuple[RaiseSite, ...]:
    """Resolve statically keyed calls to a module's bound package raiser."""
    tree = ast.parse(source, filename=module_name)
    root_scope = symtable.symtable(source, module_name, "exec")
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        module = ModuleType(module_name)

    raiser_names = {
        alias.asname or alias.name
        for node in tree.body
        if _is_raiser_import(node)
        for alias in node.names
        if alias.name == "raiser"
    }
    string_constants: dict[str, str] = {}
    raise_registries: dict[str, Mapping[str, RefusalSpec] | None] = {}
    binding_errors: dict[str, str] = {}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            registry = _dispatcher_registry(_module_binding(module, node.name), node)
            if registry is not None:
                raise_registries[node.name] = registry

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if value is None:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in raiser_names
        ):
            registry_name = (
                value.args[0].id if value.args and isinstance(value.args[0], ast.Name) else None
            )
            raw_registry = _module_binding(module, registry_name) if registry_name else None
            for name in names:
                if not _is_registry(raw_registry):
                    binding_errors[name] = (
                        f"{module_name}:{node.lineno}: unresolved package raiser registry binding"
                    )
                    raise_registries[name] = None
                else:
                    raise_registries[name] = raw_registry
        elif isinstance(value, ast.Name):
            for name in names:
                if value.id in raise_registries:
                    raise_registries[name] = raise_registries[value.id]
                    if value.id in binding_errors:
                        binding_errors[name] = binding_errors[value.id]
                elif value.id in string_constants:
                    string_constants[name] = string_constants[value.id]
        elif isinstance(value, ast.Constant) and isinstance(value.value, str):
            for name in names:
                string_constants[name] = value.value
    visitor = _RaiseSiteVisitor(
        module, root_scope, raise_registries, binding_errors, string_constants
    )
    visitor.visit(tree)
    return tuple(visitor.sites)


def _resolve_string(node: ast.expr, constants: Mapping[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _keyword_names(node: ast.Call) -> frozenset[str] | None:
    if any(keyword.arg is None for keyword in node.keywords):
        return None
    return frozenset(keyword.arg for keyword in node.keywords if keyword.arg is not None)


def validate_raise_sites(sites: Sequence[RaiseSite]) -> tuple[str, ...]:
    """Return each statically knowable registry or template-context violation."""
    violations: list[str] = []
    for site in sites:
        if site.binding_error:
            violations.append(site.binding_error)
            continue
        if site.lookup_key is None:
            violations.append(
                f"line {site.line}: dynamic refusal registry lookup key is not statically known"
            )
            continue
        spec = site.registry.get(site.lookup_key)
        if not isinstance(spec, RefusalSpec):
            violations.append(
                f"line {site.line}: refusal lookup key {site.lookup_key!r} is not registered in its bound registry"
            )
            continue
        if (
            site.keyword_names is not None
            and spec.template is not None
            and site.keyword_names != spec.keys
        ):
            violations.append(
                f"line {site.line}: refusal lookup key {site.lookup_key!r} supplies "
                f"{sorted(site.keyword_names)}, but spec.keys={sorted(spec.keys)}"
            )
    return tuple(violations)
