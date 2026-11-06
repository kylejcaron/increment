"""Warehouse schema lookup and creation, and identifier limits, shared by the warehouse stores."""

from __future__ import annotations

import inspect
from typing import Any

from increment.errors import InvalidRequestError, RefusalSpec, refuse

# PostgreSQL keeps at most 63 bytes of an identifier, the fewest of the verified warehouses.
MAX_IDENTIFIER_BYTES = 63

# Documented identifier limits: PostgreSQL truncates a longer name, Snowflake and
# BigQuery reject it. DuckDB and SQLite set none.
_IDENTIFIER_LIMITS = {
    "postgres": (MAX_IDENTIFIER_BYTES, "bytes"),
    "snowflake": (255, "characters"),
    "bigquery": (1024, "bytes"),
}


def _render_schema_name_too_long(*, store: str, schema: str, backend: str, limit: int) -> str:
    return (
        f"The {store} schema {schema!r} is longer than {backend} allows for a schema or "
        f"catalog name ({limit} {_IDENTIFIER_LIMITS[backend][1]}). Choose a shorter name."
    )


_SCHEMA_NAME_TOO_LONG = RefusalSpec(
    "store.schema_name_too_long", InvalidRequestError, render=_render_schema_name_too_long
)


def _name_exceeds_limit(con: Any, name: str) -> int | None:
    """The identifier limit *name* exceeds on *con*'s backend; None within it or with no limit."""
    entry = _IDENTIFIER_LIMITS.get(getattr(con, "name", None))
    if entry is None:
        return None
    limit, unit = entry
    length = len(name.encode()) if unit == "bytes" else len(name)
    return limit if length > limit else None


def check_schema_name_length(
    con: Any, *, store: str, catalog: str | None, schema: str | None
) -> None:
    """Refuse a catalog or schema name longer than *con*'s warehouse allows.

    Stores call this when constructed, before any read or write.
    """
    for name in (catalog, schema):
        limit = None if name is None else _name_exceeds_limit(con, name)
        if limit is not None:
            location = ".".join(part for part in (catalog, schema) if part is not None)
            refuse(
                _SCHEMA_NAME_TOO_LONG, store=store, schema=location, backend=con.name, limit=limit
            )


def split_schema(database: str) -> tuple[str | None, str] | None:
    """Split ``catalog.schema`` as Ibis splits a dotted table location; None past two parts."""
    parts = database.split(".")
    if len(parts) == 1:
        return None, database
    if len(parts) == 2:
        return parts[0], parts[1]
    return None


def schema_exists(con: Any, *, schema: str, catalog: str | None) -> bool | None:
    """Whether *catalog* lists *schema*; None when the backend cannot tell.

    It cannot tell when it has no schema listing or the listing fails. Without
    a catalog the lookup is the current catalog's, where unqualified table
    operations resolve; DuckDB would otherwise list every attached catalog. A
    backend whose listing takes no catalog is called without one.
    """
    list_databases = getattr(con, "list_databases", None)
    if not callable(list_databases):
        return None
    try:
        if catalog is None and "catalog" in inspect.signature(list_databases).parameters:
            catalog = getattr(con, "current_catalog", None)
        names = list_databases() if catalog is None else list_databases(catalog=catalog)
    except Exception:
        return None
    return schema in names


def ensure_schema(con: Any, *, schema: str | None, catalog: str | None) -> bool:
    """Create *schema* unless it is listed; False when it can be neither found nor created.

    Call only before a write. A backend that can neither create nor list
    schemas is left to report on first use.
    """
    if schema is None or schema_exists(con, schema=schema, catalog=catalog):
        return True
    create_database = getattr(con, "create_database", None)
    if not callable(create_database):
        return not callable(getattr(con, "list_databases", None))
    options = {} if catalog is None else {"catalog": catalog}
    try:
        create_database(_created_name(con, schema), force=False, **options)
    except Exception:
        # Another writer may have created it since the lookup.
        return schema_exists(con, schema=schema, catalog=catalog) is True
    return True


def _created_name(con: Any, schema: str) -> Any:
    """The name to create, spelled so the backend stores it as table operations address it.

    Ibis leaves a PostgreSQL schema name unquoted, which PostgreSQL folds to lower
    case, while its table operations quote the name.
    """
    if getattr(con, "name", None) != "postgres":
        return schema
    from sqlglot import exp

    return exp.to_identifier(schema, quoted=True)
