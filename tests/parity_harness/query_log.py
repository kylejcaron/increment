"""Helpers for collecting complete DuckDB statement counts in parity probes."""

from __future__ import annotations

from typing import Any


def enable_query_log(ibis_connection: Any) -> None:
    """Capture every executed SQL statement on this connection using DuckDB's QueryLog."""
    ibis_connection.con.execute("CALL enable_logging(['QueryLog'])")


def query_log(ibis_connection: Any) -> list[str]:
    """Return DuckDB's captured QueryLog statements in execution order."""
    return [
        message
        for _, message in ibis_connection.con.execute(
            "SELECT type, message FROM duckdb_logs WHERE type = 'QueryLog'"
        ).fetchall()
    ]
