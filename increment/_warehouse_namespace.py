"""Warehouse schema provisioning shared by the results and artifact stores."""

from __future__ import annotations

from typing import Any


def ensure_schema(
    con: Any,
    *,
    schema: str | None,
    catalog: str | None,
    database: str | tuple[str, str] | None,
) -> bool:
    """Create *schema* on demand; False when it can be neither created nor found.

    Backends commonly report an existing schema as a creation error, so a failed
    or unavailable creation is followed by listing *database*. A backend that can
    neither create nor list a schema is left to report on first use.
    """
    if schema is None:
        return True
    create_database = getattr(con, "create_database", None)
    if callable(create_database):
        try:
            create_database(schema, catalog=catalog, force=False)
        except Exception:
            pass
        else:
            return True
    list_tables = getattr(con, "list_tables", None)
    if not callable(list_tables):
        return True
    try:
        list_tables(database=database)
    except Exception:
        return False
    return True
