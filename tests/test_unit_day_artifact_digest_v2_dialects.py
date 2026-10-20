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


@pytest.mark.parametrize(
    "backend,namespace,current_catalog,current_database,expected_database",
    [
        ("duckdb", ("temp", "main"), None, None, ("temp", "main")),
        ("postgres", (None, None), None, "analytics", "pg_temp"),
        (
            "snowflake",
            (None, None),
            "warehouse",
            "scratch",
            ("warehouse", "scratch"),
        ),
        (
            "bigquery",
            ("project", "session_dataset"),
            "project",
            "default_dataset",
            ("project", "session_dataset"),
        ),
    ],
)
def test_digest_scratch_cleanup_targets_created_namespace(
    backend, namespace, current_catalog, current_database, expected_database
):
    from types import SimpleNamespace
    from typing import Any, cast

    from increment.query.artifact_digest import _drop_sql_digest_hash_table

    catalog, database = namespace
    relation = SimpleNamespace(
        op=lambda: SimpleNamespace(
            name="digest_scratch",
            namespace=SimpleNamespace(catalog=catalog, database=database),
        )
    )

    class Connection:
        def __init__(self):
            self.name = backend
            self.current_catalog = current_catalog
            self.current_database = current_database
            self.dropped = []

        def drop_table(self, name, **kwargs):
            self.dropped.append((name, kwargs))

    con = Connection()
    _drop_sql_digest_hash_table(con, cast(Any, relation))
    assert con.dropped == [("digest_scratch", {"database": expected_database, "force": True})]


def test_bigquery_digest_fetch_disables_storage_api():
    import pyarrow as pa

    from increment.query.artifact_digest import _digest_to_pyarrow

    relation = ibis.table({"value": "int64"}, name="source")
    expr = relation.aggregate(total=relation.value.count())

    class Query:
        def __init__(self):
            self.arrow_options = None

        def to_arrow(self, **kwargs):
            self.arrow_options = kwargs
            return pa.table({"total": [3]})

    class Connection:
        name = "bigquery"

        def __init__(self):
            self.query = Query()
            self.sql = None

        def compile(self, table):
            return "SELECT COUNT(value) AS total FROM source"

        def raw_sql(self, sql):
            self.sql = sql
            return self.query

    con = Connection()
    actual = _digest_to_pyarrow(con, expr)

    assert actual.to_pylist() == [{"total": 3}]
    assert con.sql == "SELECT COUNT(value) AS total FROM source"
    assert con.query.arrow_options == {
        "create_bqstorage_client": False,
        "progress_bar_type": None,
    }


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
    if dialect == "snowflake":
        assert 'HEX_ENCODE("t0"."raw", 0)' in sql


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


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "postgres"])
def test_date_token_compiles_for_every_dialect(dialect):
    import sqlglot

    table = ibis.table({"day": "date"}, name="rows")
    token = _row_token_expr(table, [("day", 4, "date")], dialect=dialect)
    sql = str(ibis.to_sql(table.select(tok=token), dialect=dialect))
    sqlglot.parse_one(sql, read=dialect)


def test_date_token_is_zero_padded_iso_text_for_every_year():
    import datetime as dt

    import pyarrow as pa

    days = [dt.date(1, 1, 1), dt.date(999, 12, 31), dt.date(1969, 12, 31), dt.date(2024, 2, 29)]
    con = ibis.duckdb.connect()
    table = con.create_table("rows", pa.table({"day": pa.array(days, type=pa.date32())}))
    token = _row_token_expr(table, [("day", 4, "date")], dialect="duckdb")
    tokens = con.to_pyarrow(table.select(tok=token)).column("tok").to_pylist()
    assert [t[18:] for t in tokens] == [d.isoformat() for d in days]


@pytest.mark.parametrize("dialect", ["duckdb", "snowflake", "bigquery", "postgres"])
def test_float64_binary_token_compiles_as_a_scalar(dialect):
    import sqlglot

    table = ibis.table({"value": "float64"}, name="rows")
    payload = _float_payload_expr(table.value, dialect=dialect, digest_format=4)
    sql = str(ibis.to_sql(table.select(payload=payload), dialect=dialect))
    sqlglot.parse_one(sql, read=dialect)
    assert "WITH " not in sql.upper()
    assert "LAG(" not in sql.upper()
    assert "E+" not in sql.upper()
    if dialect == "bigquery":
        assert "IEEE_DIVIDE(1.0" in sql
        assert "STARTS_WITH" not in sql


def test_float64_binary_tokens_match_python_for_all_pinned_values():
    import json
    import struct
    from pathlib import Path

    import pyarrow as pa

    from increment.query._artifact_float import float64_token_py

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/artifact_float64_cases.json").read_text()
    )
    values = [struct.unpack(">d", bytes.fromhex(case["bits"]))[0] for case in fixture["cases"]]
    table = ibis.duckdb.connect()
    try:
        source = table.create_table("float64_token_spec", pa.table({"value": values}))
        expected = [float64_token_py(value) for value in values]
        actual = table.to_pyarrow(
            source.select(
                token=_float_payload_expr(source.value, dialect="duckdb", digest_format=4)
            )
        )["token"].to_pylist()
        assert actual == expected
        assert float64_token_py(1.0) == "4503599627370496p-52"
        assert float64_token_py(float.fromhex("0x0.0000000000001p-1022")) == "1p-1074"
        assert float64_token_py(-0.0) == "-0"
    finally:
        table.disconnect()


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
