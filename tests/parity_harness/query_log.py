"""Helpers for collecting complete DuckDB statement counts in parity probes."""

from __future__ import annotations

from typing import Any


def enable_query_log(ibis_connection: Any) -> bool:
    """Enable DuckDB's QueryLog when the installed DuckDB supports it."""
    import duckdb

    try:
        ibis_connection.con.execute("CALL enable_logging(['QueryLog'])")
    except duckdb.CatalogException as exc:
        if "enable_logging" not in str(exc) or "does not exist" not in str(exc):
            raise
        return False
    return True


def query_log(ibis_connection: Any) -> list[str]:
    """Return DuckDB's captured QueryLog statements in execution order."""
    return [
        message
        for _, message in ibis_connection.con.execute(
            "SELECT type, message FROM duckdb_logs WHERE type = 'QueryLog'"
        ).fetchall()
    ]
