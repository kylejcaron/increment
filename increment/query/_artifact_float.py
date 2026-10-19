"""Exact binary64 tokens for SQL artifact digests."""

from __future__ import annotations

import struct
from typing import Any

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz


class _Float64Token(ops.Value):
    arg: ops.Value[dt.Floating]

    dtype = dt.string
    shape = rlz.shape_like("arg")


def float64_token_py(value: float) -> str:
    """Return sign + exact integer significand + binary exponent for binary64."""
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    negative = bits >> 63
    sign = "-" if negative else ""
    exponent = (bits >> 52) & 0x7FF
    fraction = bits & ((1 << 52) - 1)
    if exponent == 0x7FF:
        return "nan" if fraction else f"{sign}inf"
    if exponent == 0:
        return f"{sign}{fraction}p-1074" if fraction else f"{sign}0"
    mantissa = (1 << 52) | fraction
    return f"{sign}{mantissa}p{exponent - 1075}"


def float64_token_sql(value: str, *, dialect: str) -> str:
    """Render finite values as exact decimal-significand binary tokens.

    The logarithm only estimates the binary exponent; a power-of-two ratio
    corrects its boundary before the significand is scaled. Splitting the
    subnormal scaling at 2**512 keeps every intermediate finite and normal.
    """
    if dialect == "snowflake":
        sign = f"LEFT(TO_VARCHAR(({value})), 1) = '-'"
        integer_type, text_type = "NUMBER(38,0)", "VARCHAR"
    elif dialect == "bigquery":
        sign = f"IEEE_DIVIDE(1.0, ({value})) < 0"
        integer_type, text_type = "INT64", "STRING"
    elif dialect == "postgres":
        sign = f"(GET_BYTE(FLOAT8SEND(({value})), 0) & 128) <> 0"
        integer_type, text_type = "BIGINT", "TEXT"
    else:
        sign = f"SIGNBIT(({value}))"
        integer_type, text_type = "BIGINT", "VARCHAR"

    magnitude = f"ABS(({value}))"
    estimate = f"LEAST(1023.0, FLOOR(LN({magnitude}) / LN(2.0)))"
    ratio = f"({magnitude} / POWER(2.0, {estimate}))"
    correction = f"CASE WHEN {ratio} < 1.0 THEN -1 WHEN {ratio} >= 2.0 THEN 1 ELSE 0 END"
    normalized = (
        f"CASE WHEN {ratio} < 1.0 THEN {ratio} * 2.0 "
        f"WHEN {ratio} >= 2.0 THEN {ratio} * 0.5 ELSE {ratio} END"
    )
    subnormal = f"({magnitude} * POWER(2.0, 512) * POWER(2.0, 512)) * POWER(2.0, 50)"
    is_subnormal = f"{magnitude} < 2.2250738585072014e-308"
    mantissa = (
        f"CAST(CASE WHEN {is_subnormal} THEN {subnormal} "
        f"ELSE {normalized} * 4503599627370496.0 END AS {integer_type})"
    )
    binary_exponent = f"CASE WHEN {is_subnormal} THEN -1074 ELSE {estimate} + {correction} - 52 END"
    sign_prefix = f"CASE WHEN {sign} THEN '-' ELSE '' END"
    return (
        f"CASE WHEN ({value}) = 0 THEN {sign_prefix} || '0' "
        f"ELSE {sign_prefix} || CAST({mantissa} AS {text_type}) || 'p' "
        f"|| CAST(CAST({binary_exponent} AS {integer_type}) AS {text_type}) END"
    )


def _compile_float64_token(compiler: Any, _op: _Float64Token, *, arg: Any) -> Any:
    import sqlglot.expressions as sge

    dialect = compiler.dialect
    return sge.Var(this=float64_token_sql(arg.sql(dialect=dialect), dialect=dialect))


def register_float64_token() -> None:
    from ibis.backends.sql.compilers.bigquery import BigQueryCompiler
    from ibis.backends.sql.compilers.duckdb import DuckDBCompiler
    from ibis.backends.sql.compilers.postgres import PostgresCompiler
    from ibis.backends.sql.compilers.snowflake import SnowflakeCompiler

    DuckDBCompiler.visit__Float64Token = _compile_float64_token  # ty: ignore[unresolved-attribute]
    PostgresCompiler.visit__Float64Token = _compile_float64_token  # ty: ignore[unresolved-attribute]
    SnowflakeCompiler.visit__Float64Token = _compile_float64_token  # ty: ignore[unresolved-attribute]
    BigQueryCompiler.visit__Float64Token = _compile_float64_token  # ty: ignore[unresolved-attribute]
