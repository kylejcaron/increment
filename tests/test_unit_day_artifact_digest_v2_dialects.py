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
    _float_payload_expr,
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
        ("postgres", "ENCODE(SHA256("),
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
        ("postgres", "SHA256(", "ENCODE("),
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
    order = ibis.asc(_TABLE.name, nulls_first=dialect == "bigquery")
    grouped = _TABLE.group_by("id").aggregate(h=_TABLE.name.group_concat(sep="", order_by=order))
    sql = str(ibis.to_sql(grouped, dialect=dialect))
    assert "ORDER BY" in sql.upper()
    if dialect == "bigquery":
        assert "NULLS LAST" not in sql.upper()


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "postgres"])
def test_timestamp_offset_detection_uses_portable_regex(dialect):
    from increment.query.artifact_publish import _as_utc_timestamp

    table = ibis.table({"ts": "timestamp"}, name="rows")
    sql = str(ibis.to_sql(table.select(ts=_as_utc_timestamp(table.ts)), dialect=dialect))
    assert "(?:" not in sql
    assert "[0-9]{2}" in sql


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "postgres"])
def test_row_token_compiles_without_backend_specific_sql(dialect):
    """The row-token construction (cast/length/lpad/concat) is pure portable
    ibis -- confirm it compiles cleanly for every dialect with no manual SQL
    injection needed for non-FLOAT64 fields."""
    token = _row_token_expr(_TABLE, [("id", 2, "int"), ("name", 1, "string")], dialect=dialect)
    ibis.to_sql(_TABLE.select(tok=token), dialect=dialect)


def test_postgres_float_token_uses_ieee_bytes_and_explicit_17_digit_rounding():
    table = ibis.table({"value": "float64"}, name="rows")
    token = _row_token_expr(table, [("value", 3, "float")], dialect="postgres")
    sql = str(ibis.to_sql(table.select(token=token), dialect="postgres"))
    assert "FLOAT8SEND(" in sql
    assert "GET_BYTE(" in sql
    assert "DIV(" in sql
    assert "%" in sql
    assert "TRIM_SCALE(" in sql


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_cloud_float17g_uses_exact_integer_limb_formatter(dialect):
    table = ibis.table({"value": "float64"}, name="rows")
    payload = _float_payload_expr(table.value, dialect=dialect)
    sql = str(ibis.to_sql(table.select(payload=payload), dialect=dialect))
    import sqlglot

    sqlglot.parse_one(sql, read=dialect)
    assert "LAG(" in sql
    assert "FORMAT(" not in sql
    assert "TO_CHAR(" not in sql


@pytest.mark.integration
def test_integer_limb_float17g_matches_native_duckdb():
    import json
    import math
    import struct
    from pathlib import Path

    import numpy as np
    import pyarrow as pa

    from increment.query._artifact_float import _float17g_sql

    fixture = json.loads((Path(__file__).parent / "fixtures/artifact_float17g.json").read_text())
    pinned = [struct.unpack(">d", bytes.fromhex(case["bits"]))[0] for case in fixture["cases"]]
    powers = [
        neighbour
        for exponent in range(-1074, 1024)
        for value in (math.ldexp(1.0, exponent),)
        for neighbour in (math.nextafter(value, 0.0), value, math.nextafter(value, math.inf))
    ]
    values = np.concatenate(
        [
            np.random.default_rng(173).bit_generator.random_raw(10_000).view(np.float64),
            pinned,
            powers,
        ]
    )
    finite = values[np.isfinite(values)]
    con = ibis.duckdb.connect()
    try:
        con.create_table("float_limb_spec", pa.table({"value": finite}))
        sql = _float17g_sql('"value"', dialect="duckdb")
        rows = con.raw_sql(
            f"SELECT value, {sql} AS actual, printf('%.17g', value) AS expected "
            "FROM float_limb_spec"
        ).fetchall()
        for value, actual, expected in rows:
            assert actual == expected, value
    finally:
        con.disconnect()


@pytest.mark.parametrize(
    "dialect,expected_fragment",
    [
        ("duckdb", "EPOCH_US("),
        ("snowflake", "DATE_PART('epoch_microsecond'"),
        ("bigquery", "UNIX_MICROS("),
        ("postgres", "EXTRACT(EPOCH FROM"),
    ],
)
def test_timestamp_token_renders_epoch_microseconds_not_session_local_text(
    dialect, expected_fragment
):
    table = ibis.table({"ts": "timestamp('UTC')"}, name="rows")
    token = _row_token_expr(table, [("ts", 5, "timestamp")], dialect=dialect)
    sql = str(ibis.to_sql(table.select(tok=token), dialect=dialect))
    assert expected_fragment in sql
