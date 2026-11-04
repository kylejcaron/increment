"""Helpers for collecting complete DuckDB statement counts in parity probes."""

from __future__ import annotations

from typing import Any


def _execute_first_spelling(
    ibis_connection: Any, function: str, statements: tuple[str, ...]
) -> bool:
    """Run the first statement whose logging function the installed DuckDB defines."""
    import duckdb

    for statement in statements:
        try:
            ibis_connection.con.execute(statement)
        except duckdb.CatalogException as exc:
            if function not in str(exc) or "does not exist" not in str(exc):
                raise
        else:
            return True
    return False


def enable_query_log(ibis_connection: Any) -> bool:
    """Enable DuckDB's QueryLog when the installed DuckDB supports it.

    DuckDB 1.3 enables logging with a pragma; 1.4.0 replaced it with a table function.
    """
    return _execute_first_spelling(
        ibis_connection,
        "enable_logging",
        ("CALL enable_logging(['QueryLog'])", "PRAGMA enable_logging('QueryLog')"),
    )


def disable_query_log(ibis_connection: Any) -> None:
    """Disable the logging `enable_query_log` turned on."""
    assert _execute_first_spelling(
        ibis_connection, "disable_logging", ("CALL disable_logging()", "PRAGMA disable_logging")
    )


def query_log(ibis_connection: Any) -> list[str]:
    """Return DuckDB's captured QueryLog statements in execution order."""
    return [
        message
        for _, message in ibis_connection.con.execute(
            "SELECT type, message FROM duckdb_logs WHERE type = 'QueryLog'"
        ).fetchall()
    ]
