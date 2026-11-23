"""Register the definitions and physical ingredients of DuckDB spillover views."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Never, cast

import sqlglot as sg
from sqlglot.tokens import TokenType

from increment._spillover_design import _raise as _refuse_design
from increment.query._spillover_sql import duckdb_identifier
from increment.query._spillover_sql import duckdb_identifier_sql as _fold_sql

if TYPE_CHECKING:
    from ibis import Table
    from ibis.backends.duckdb import Backend as DuckDBBackend
    from ibis.backends.sql import SQLBackend

type _Locator = tuple[str, str, str]
type _Parser = Callable[[str, str], Iterable[tuple[str, ...]]]
type _FunctionGuard = Callable[[str, tuple[str, str]], str | None]


def _literal(value: str) -> str:
    return sg.exp.Literal.string(value).sql(dialect="duckdb")


def _key(name: _Locator) -> _Locator:
    catalog, schema, relation = name
    return duckdb_identifier(catalog), duckdb_identifier(schema), duckdb_identifier(relation)


def _tuple_sql(name: _Locator) -> str:
    return "(" + ", ".join(map(_literal, name)) + ")"


def _objects_sql(*, tables: bool) -> str:
    fields = f"{_fold_sql('database_name')} AS db, {_fold_sql('schema_name')} AS ns"
    views = (
        f"SELECT {fields}, {_fold_sql('view_name')} AS obj, 'view' AS kind, sql AS ddl "
        "FROM system.main.duckdb_views()"
    )
    if not tables:
        return views
    return (
        f"{views} UNION ALL SELECT {fields}, {_fold_sql('table_name')}, 'table', NULL "
        "FROM system.main.duckdb_tables()"
    )


def _unresolved() -> Never:
    _refuse_design(
        "spillover.design.invalid",
        field="relations",
        requirement=(
            "create metric and design views before registering, and use named tables or views "
            "with explicit database, schema and table names for their inputs"
        ),
    )


@dataclass(frozen=True)
class _View:
    digest: str
    children: tuple[_Locator, ...]
    guards: tuple[str, ...]


def _view(
    owner: _Locator, ddl: str, parse_sources: _Parser, function_guard: _FunctionGuard
) -> _View:
    # Keep the stored SQL spelling (including function aliases); do not translate the view.
    try:
        separator = next(
            t for t in sg.tokenize(ddl, read="duckdb") if t.token_type == TokenType.ALIAS
        )
    except (sg.errors.SqlglotError, StopIteration):
        _unresolved()
    body = ddl[separator.end + 1 :]
    functions = function_guard(body, (owner[0], owner[1]))
    guards = {functions} if functions else set()
    children = set()
    for parts in parse_sources("relations", body):
        match tuple(map(duckdb_identifier, parts)):
            case (db, ns, obj):
                child = (db, ns, obj)
            case (ns, obj):
                # If the prefix also names a catalog, DuckDB can pick a schema using the
                # session search path. Require a three-part name in that case.
                guards.add(
                    "NOT EXISTS (SELECT 1 FROM system.main.duckdb_databases() "
                    f"WHERE {_fold_sql('database_name')} = {_literal(ns)})"
                )
                child = (owner[0], ns, obj)
            case (obj,):
                child = (*owner[:2], obj)
            case _:
                _unresolved()
        # A view first searches its own namespace, but temp.main precedes it.
        if len(parts) == 1 or (len(parts) == 2 and child[1] == "main"):
            temp = _tuple_sql(("temp", "main", child[2]))
            guards.add(f"NOT EXISTS (SELECT 1 FROM objects WHERE (db, ns, obj) = {temp})")
        children.add(child)
    return _View(
        hashlib.sha256(ddl.encode()).hexdigest(), tuple(sorted(children)), tuple(sorted(guards))
    )


def _closure(root: _Locator, views: Mapping[_Locator, _View]) -> tuple[_Locator, ...]:
    visited = set()
    todo = [root]
    while todo:
        node = todo.pop()
        if node not in visited:
            visited.add(node)
            if node in views:
                todo.extend(views[node].children)
    return tuple(sorted(visited))


@dataclass(frozen=True)
class ViewSources:
    views: Mapping[_Locator, _View]
    closures: Mapping[_Locator, tuple[_Locator, ...]]

    @property
    def catalogs(self) -> set[str]:
        return {loc[0] for nodes in self.closures.values() for loc in nodes}

    @property
    def recipe(self) -> tuple[tuple[str, str], ...]:
        entries = []
        for root, nodes in sorted(self.closures.items()):
            if root not in self.views:
                continue
            definitions = [
                (n, self.views[n].digest, self.views[n].children) for n in nodes if n in self.views
            ]
            digest = hashlib.sha256(
                json.dumps(definitions, separators=(",", ":")).encode()
            ).hexdigest()
            entries.append(("view:" + json.dumps(root, separators=(",", ":")), digest))
        return tuple(entries)

    def checks(
        self,
        con: SQLBackend,
        names: Mapping[str, _Locator],
        metadata: Table,
        locations: Mapping[str, str],
    ) -> Table:
        """Return binding flags evaluated in the same statement as the measurement."""
        expected = []
        fields = []
        needs_tables = False
        for i, (role, name) in enumerate(names.items()):
            root = _key(name)
            guards = {f"system.main.bool_and(matched) FILTER (WHERE role_id = {i})"}
            for node in self.closures[root]:
                view = self.views.get(node)
                if view is not None:
                    needs_tables = True
                    kind, digest = "view", view.digest
                    guards.update(view.guards)
                else:
                    # Only a root table can arrive after registration.
                    kind, digest = ("optional" if node == root else "table"), ""
                file_id = locations[node[0]]
                cells = ", ".join(map(_literal, (*node, kind, digest, file_id)))
                expected.append(f"({i}, {cells})")
            alias = sg.to_identifier(role, quoted=True).sql(dialect="duckdb")
            fields.append(" AND ".join(sorted(guards)) + " AS " + alias)
        return con.sql(
            f"""
            WITH objects AS ({_objects_sql(tables=needs_tables)}), files AS ({con.compile(metadata)}),
            expected(role_id, db, ns, obj, required, digest, file_id) AS (VALUES {", ".join(expected)}),
            matches AS (
                SELECT e.role_id, coalesce(f.location = e.file_id AND CASE e.required
                    WHEN 'view' THEN o.kind = 'view' AND system.main.sha256(o.ddl) = e.digest
                    WHEN 'table' THEN o.kind = 'table'
                    ELSE o.kind IS DISTINCT FROM 'view' END, FALSE) AS matched
                FROM expected e LEFT JOIN objects o USING (db, ns, obj)
                LEFT JOIN files f ON f.catalog = e.db
            )
            SELECT {", ".join(fields)} FROM matches
            """,
            schema=dict.fromkeys(names, "boolean"),
        )


def view_sources(
    con: SQLBackend,
    roots: Iterable[_Locator],
    parse_sources: _Parser,
    function_guard: _FunctionGuard,
) -> ViewSources:
    """Resolve catalog entries one level at a time, never opening a measurement relation."""
    backend = cast("DuckDBBackend", con)
    root_names = set(map(_key, roots))
    pending = root_names.copy()
    existing: set[_Locator] = set()
    visited: set[_Locator] = set()
    views: dict[_Locator, _View] = {}
    while pending:
        requested = ", ".join(map(_tuple_sql, sorted(pending)))
        objects = _objects_sql(tables=True)
        rows = backend.raw_sql(
            f"SELECT db, ns, obj, kind, ddl FROM ({objects}) WHERE (db, ns, obj) IN ({requested})"
        ).fetchall()
        visited.update(pending)
        pending.clear()
        for catalog, schema, name, kind, ddl in rows:
            node = (catalog, schema, name)
            existing.add(node)
            if kind == "view":
                if not ddl:
                    _unresolved()
                view = views[node] = _view(node, ddl, parse_sources, function_guard)
                pending.update(child for child in view.children if child not in visited)

    # Missing roots may be future measurement tables. Missing ingredients of an existing
    # view could use a search-path fallback or a replacement scan, neither of which is fixed.
    if any(child not in existing for view in views.values() for child in view.children):
        _unresolved()
    try:
        TopologicalSorter({n: v.children for n, v in views.items()}).prepare()
    except CycleError:
        _unresolved()
    guards = sorted({guard for view in views.values() for guard in view.guards})
    if guards:
        row = backend.raw_sql(
            f"WITH objects AS ({_objects_sql(tables=True)}) SELECT " + " AND ".join(guards)
        ).fetchone()
        if not row or not row[0]:
            _unresolved()
    return ViewSources(views, {root: _closure(root, views) for root in root_names})
