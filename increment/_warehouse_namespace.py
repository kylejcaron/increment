"""Warehouse schema lookup and creation shared by the results and artifact stores."""

from __future__ import annotations

import inspect
from typing import Any


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
