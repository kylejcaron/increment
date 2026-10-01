"""Compile-only checks (no network, no live backend) that the SQL digest's
row-token and hash primitives compile to the real, documented function each
vendor's SQL dialect actually exposes. DuckDB's compiled SQL is additionally
covered by execution in tests/test_unit_day_artifact_digest_v2.py; the other
three dialects are pinned here so an ibis/sqlglot upgrade that silently
changes the compiled function name is caught before it reaches a live
backend."""

from __future__ import annotations

import ibis
import pytest

from increment.query.artifact_digest import (
    _row_token_expr,
    _sql_binary_hex_expr,
    _sql_digest_sha256_binary_expr,
    _sql_digest_sha256_expr,
)

_TABLE = ibis.table(
    {"id": "int64", "name": "string", "amount": "float64", "raw": "binary"}, name="rows"
)


@pytest.mark.parametrize(
    "dialect,expected_fragment",
    [
        ("duckdb", "SHA256("),
        ("snowflake", "SHA2("),
        ("bigquery", "TO_HEX(SHA256("),
        ("postgres", "ENCODE(DIGEST("),
    ],
)
def test_sha256_hex_compiles_to_the_vendor_function(dialect, expected_fragment):
    expr = _sql_digest_sha256_expr(_TABLE.name, dialect=dialect)
    sql = str(ibis.to_sql(expr, dialect=dialect))
    assert expected_fragment in sql


@pytest.mark.parametrize(
    "dialect,binary_fragment,hex_fragment",
    [
        ("duckdb", "UNHEX(SHA256(", "LOWER(HEX("),
        ("snowflake", "SHA2_BINARY(", "HEX_ENCODE("),
        ("bigquery", "SHA256(", "TO_HEX("),
        ("postgres", "DIGEST(", "ENCODE("),
    ],
)
def test_binary_row_hash_and_hex_rendering_compile_to_vendor_functions(
    dialect, binary_fragment, hex_fragment
):
    expr = _TABLE.select(
        b=_sql_digest_sha256_binary_expr(_TABLE.name, dialect=dialect),
        h=_sql_binary_hex_expr(_TABLE.raw, dialect=dialect),
    )
    sql = str(ibis.to_sql(expr, dialect=dialect))
    assert binary_fragment in sql
    assert hex_fragment in sql


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "postgres"])
def test_ordered_group_concat_compiles(dialect):
    grouped = _TABLE.group_by("id").aggregate(
        h=_TABLE.name.group_concat(sep="", order_by=_TABLE.name)
    )
    sql = str(ibis.to_sql(grouped, dialect=dialect))
    assert "ORDER BY" in sql.upper()


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "postgres"])
def test_row_token_compiles_without_backend_specific_sql(dialect):
    """The row-token construction (cast/length/lpad/concat) is pure portable
    ibis -- confirm it compiles cleanly for every dialect with no manual SQL
    injection needed for non-FLOAT64 fields."""
    token = _row_token_expr(_TABLE, [("id", 2, "int"), ("name", 1, "string")], dialect=dialect)
    ibis.to_sql(_TABLE.select(tok=token), dialect=dialect)


@pytest.mark.parametrize(
    "dialect,expected_fragment",
    [
        ("duckdb", "EPOCH_US("),
        ("snowflake", "DATE_PART('epoch_microsecond'"),
        ("bigquery", "UNIX_MICROS("),
        ("postgres", "DATE_PART('epoch'"),
    ],
)
def test_timestamp_token_renders_epoch_microseconds_not_session_local_text(
    dialect, expected_fragment
):
    table = ibis.table({"ts": "timestamp('UTC')"}, name="rows")
    token = _row_token_expr(table, [("ts", 5, "timestamp")], dialect=dialect)
    sql = str(ibis.to_sql(table.select(tok=token), dialect=dialect))
    assert expected_fragment in sql
