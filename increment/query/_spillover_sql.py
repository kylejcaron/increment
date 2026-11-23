"""Bind named spillover inputs to fixed warehouse locations before launch."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from string import ascii_lowercase, ascii_uppercase
from typing import TYPE_CHECKING, NoReturn, cast

import sqlglot as sg
from ibis.backends.sql import SQLBackend
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import TokenType

from increment._spillover_design import _raise as _refuse_design
from increment._spillover_design import refuse_input
from increment.semantics.loader import admit_read_only_sql

if TYPE_CHECKING:
    from ibis.backends.duckdb import Backend as DuckDBBackend

_Locator = tuple[str, str, str]
_Namespace = tuple[str, str]
_ASCII_FOLD = str.maketrans(ascii_uppercase, ascii_lowercase)


@dataclass(frozen=True, slots=True)
class _Query:
    field: str
    sql: str
    native_sql: str
    dependencies: tuple[_Locator, ...]
    function_guard: str | None = None


def _invalid(field: str) -> NoReturn:
    _refuse_design(
        "spillover.design.invalid",
        field=field,
        requirement="read from named tables or views with fixed database locations",
    )


def duckdb_identifier(name: str) -> str:
    """DuckDB identifier comparisons fold ASCII letters only, even inside quotes."""
    return name.translate(_ASCII_FOLD)


def duckdb_identifier_sql(column: str) -> str:
    """The same comparison in DuckDB, independent of the session's text collation."""
    folded = f"system.main.translate({column}, '{ascii_uppercase}', '{ascii_lowercase}')"
    return f'{folded} COLLATE "binary"'


# DuckDB serializes search_path as [catalog.]schema entries; only dots, commas
# and quotes require quoting. Spaces and SQL keywords may consequently be bare.
_PATH_NAME = r'("(?:[^"]|"")*"|[^.,"]+)'
_PATH_ENTRY = re.compile(rf"{_PATH_NAME}(?:\.{_PATH_NAME})?(?:,|$)")


def _explicit_path(setting: str) -> list[_Namespace]:
    def name(part: str) -> str:
        if part.startswith('"'):
            part = part[1:-1].replace('""', '"')
        return duckdb_identifier(part)

    return [
        (name(first), name(second)) if second else ("", name(first))
        for first, second in _PATH_ENTRY.findall(setting)
    ]


_DUCKDB_LOOKUP = f"""
    SELECT system.main.current_database(), system.main.current_schema(),
           system.main.current_setting('search_path'),
           ARRAY(SELECT (database_name, schema_name) FROM (
               SELECT database_name, schema_name FROM system.main.duckdb_tables()
               WHERE {duckdb_identifier_sql("table_name")} = $name
               UNION ALL
               SELECT database_name, schema_name FROM system.main.duckdb_views()
               WHERE {duckdb_identifier_sql("view_name")} = $name)),
           ARRAY(SELECT database_name FROM system.main.duckdb_schemas()
                 WHERE {duckdb_identifier_sql("schema_name")} = $prefix),
           (SELECT type FROM system.main.duckdb_databases()
            WHERE {duckdb_identifier_sql("database_name")} = $database)
"""


def _duckdb_location(con: DuckDBBackend, parts: tuple[str, ...], field: str) -> _Locator:
    name = parts[-1]
    prefix = duckdb_identifier(parts[0]) if len(parts) == 2 else ""
    key = duckdb_identifier(name)
    catalog, schema, setting, matches, schema_catalogs, db_type = con.raw_sql(
        _DUCKDB_LOOKUP, parameters={"name": key, "prefix": prefix, "database": prefix or key}
    ).fetchone()
    catalog, schema = duckdb_identifier(catalog), duckdb_identifier(schema)
    explicit = _explicit_path(setting)
    # An omitted catalog on the search path participates in unqualified and
    # schema-qualified lookup, but not lookup by an explicit database name.
    path = [("temp", "main"), *explicit, ("", "main"), ("system", "main"), ("system", "pg_catalog")]
    resolved = [(db or catalog, s) for db, s in path]
    if not prefix:
        if db_type is not None:  # A bare catalog name may denote its default table.
            _invalid(field)
        candidates = resolved
        future = [(db or catalog, s) for db, s in explicit] or [(catalog, schema)]
    else:
        schemas = [pair for pair in resolved if pair[1] == prefix] or [(catalog, prefix)]
        if db_type is not None:
            if {db for db, _ in schemas} & {duckdb_identifier(db) for db in schema_catalogs}:
                _invalid(field)
            candidates = [pair for pair in path if pair[0] == prefix]
            if not candidates:
                if db_type != "duckdb":  # No known default schema for this attachment.
                    _invalid(field)
                candidates = [(prefix, "main")]
        else:
            candidates = schemas
        future = [pair for pair in candidates if pair[0] not in ("temp", "system")]

    existing = {(duckdb_identifier(db), duckdb_identifier(s)) for db, s in matches}
    for pair in candidates:
        if pair in existing:
            return *pair, name
    # A not-yet-created measurement table remains admissible at one known location.
    if len(set(future)) != 1:
        _invalid(field)
    return *future[0], name


def _qualified(con: SQLBackend, parts: tuple[str, ...], *, field: str = "relations") -> _Locator:
    # Fully qualified sources do not need a default dataset on BigQuery.
    if len(parts) == 3:
        if not parts[0] or not parts[1]:
            refuse_input("spillover.design.invalid", "relations")
        return parts[0], parts[1], parts[2]
    if con.name == "duckdb":
        return _duckdb_location(cast("DuckDBBackend", con), parts, field)
    db = parts[-2] if len(parts) >= 2 else getattr(con, "current_database", None)
    if con.name == "bigquery" and len(parts) == 2:
        catalog = getattr(con, "billing_project", None)
    else:
        catalog = getattr(con, "current_catalog", None)
    if not catalog or not db:
        refuse_input("spillover.design.invalid", "relations")
    return catalog, db, parts[-1]


def _constant_series(node: sg.exp.Expr) -> bool:
    if not isinstance(node, sg.exp.GenerateSeries):
        return False
    for bound in node.iter_expressions():
        if isinstance(bound, sg.exp.Neg):
            bound = bound.this
        if not isinstance(bound, sg.exp.Literal) or not bound.is_int:
            return False
    return True


def _physical_tables(
    expression: sg.exp.Expr, dialect: str, field: str
) -> Iterator[tuple[sg.exp.Table, tuple[str, ...]]]:
    # Fold a copy for scope lookup; the declared column names and qualifiers survive unchanged.
    folded = expression.copy()
    if dialect == "duckdb":
        for identifier in folded.find_all(sg.exp.Identifier):
            identifier.set("this", duckdb_identifier(identifier.name))
    else:
        normalize_identifiers(folded, dialect=dialect)
    originals = {
        id(table): original
        for table, original in zip(
            folded.find_all(sg.exp.Table), expression.find_all(sg.exp.Table), strict=True
        )
    }
    for scope in traverse_scope(folded):
        if dialect == "duckdb":
            for relation in scope.udtfs:
                if isinstance(relation, sg.exp.Values):
                    continue
                if isinstance(relation, sg.exp.Lateral) and (
                    isinstance(relation.this, sg.exp.Query) or _constant_series(relation.this)
                ):
                    continue
                _invalid(field)
        # Physical occurrences, unlike selected_sources, include SEMI/ANTI inputs
        # and do not require unaliased tables in different schemas to have distinct names.
        tables = scope.tables
        if isinstance(scope.expression, sg.exp.Table):
            tables = [scope.expression, *tables]
        for table in tables:
            original = originals.pop(id(table))
            if dialect == "duckdb" and _constant_series(table.this):
                continue
            if not table.db and table.name in scope.cte_sources:
                continue
            parts = table.parts
            if not 1 <= len(parts) <= 3 or not all(isinstance(p, sg.exp.Identifier) for p in parts):
                _invalid(field)
            if dialect == "duckdb" and any(c in table.name for c in ".:/\\"):
                # DuckDB treats even quoted file paths as replacement scans.
                _invalid(field)
            yield original, tuple(part.name for part in parts)
    if originals:
        _invalid(field)


def view_table_parts(field: str, sql: str) -> tuple[tuple[str, ...], ...]:
    """Return named inputs at their declared arity in a DuckDB view's SELECT."""
    admit_read_only_sql(sql, dialect="duckdb", label=field)
    expression = sg.parse_one(sql, read="duckdb")
    return tuple(parts for _, parts in _physical_tables(expression, "duckdb", field))


def duckdb_function_guard(sql: str, view_namespace: _Namespace | None = None) -> str | None:
    """A catalog predicate excluding calls to user macros with unbound dependencies."""
    dialect = sg.Dialect.get_or_raise("duckdb")
    tokens = dialect.tokenize(sql)
    calls: dict[tuple[str, ...], set[str]] = {}
    for i in range(len(tokens) - 1):
        token = tokens[i]
        if (
            tokens[i + 1].token_type != TokenType.L_PAREN
            or token.token_type not in dialect.parser_class.FUNC_TOKENS
        ):
            continue
        parts = [duckdb_identifier(token.text)]
        pos = i
        while pos >= 2 and tokens[pos - 1].token_type == TokenType.DOT:
            pos -= 2
            parts.append(duckdb_identifier(tokens[pos].text))
        calls.setdefault(tuple(reversed(parts[1:])), set()).add(parts[0])
    if not calls:
        return None

    def literal(value: str) -> str:
        return sg.exp.Literal.string(value).sql(dialect="duckdb")

    database_name = duckdb_identifier_sql("database_name")
    schema_name = duckdb_identifier_sql("schema_name")
    predicates = []
    for namespace, names in sorted(calls.items()):
        if not namespace:
            visible = "system.main.in_search_path(database_name, schema_name)"
            if view_namespace:
                db, schema = (literal(duckdb_identifier(part)) for part in view_namespace)
                visible = f"({visible} OR ({database_name} = {db} AND {schema_name} = {schema}))"
        elif len(namespace) == 1:
            key = literal(namespace[0])
            visible = f"({database_name} = {key} OR {schema_name} = {key})"
        else:
            visible = (
                f"{database_name} = {literal(namespace[-2])} "
                f"AND {schema_name} = {literal(namespace[-1])}"
            )
        values = ", ".join(literal(n) for n in sorted(names))
        predicates.append(f"({visible} AND {duckdb_identifier_sql('function_name')} IN ({values}))")
    return (
        "NOT EXISTS (SELECT 1 FROM system.main.duckdb_functions() WHERE NOT internal "
        "AND function_type IN ('macro', 'table_macro') AND (" + " OR ".join(predicates) + "))"
    )


def _bind_query(con: SQLBackend, field: str, sql: str, dialect: str | None) -> _Query:
    """Qualify physical tables, not CTE references, without opening any relation."""
    native = con.name
    if dialect is not None:
        admit_read_only_sql(sql, dialect=dialect, label=field)
    expression = sg.parse_one(sql, read=dialect or native)
    dependencies = set()
    for table, parts in _physical_tables(expression, dialect or native, field):
        locator = _qualified(con, parts, field=field)
        dependencies.add(locator)
        for key, part in zip(("catalog", "db", "this"), locator, strict=True):
            table.set(key, sg.to_identifier(part, quoted=True))
    rendered = expression.sql(dialect=native)
    admit_read_only_sql(rendered, dialect=native, label=field)
    guard = None
    if native == "duckdb":
        guard = duckdb_function_guard(rendered)
        # A source spelling may be a macro even if SQLGlot emits a builtin synonym.
        checks = {guard} if guard else set()
        if dialect in (None, native) and (original_guard := duckdb_function_guard(sql)):
            checks.add(original_guard)
        if checks:
            backend = cast("DuckDBBackend", con)
            if not backend.raw_sql("SELECT " + " AND ".join(sorted(checks))).fetchone()[0]:
                _invalid(field)
    return _Query(field, sql, rendered, tuple(sorted(dependencies)), guard)
