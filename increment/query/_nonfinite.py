"""Portable warehouse predicates for floating-point special values."""

from __future__ import annotations

from typing import Any, cast

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz
import ibis.expr.types as ir


class _IsNaN(ops.Value):
    arg: ops.Value[dt.Floating]

    dtype = dt.boolean
    shape = rlz.shape_like("arg")


def _compile_is_nan(compiler: Any, _op: _IsNaN, *, arg: Any) -> Any:
    import sqlglot.expressions as sge

    dialect = compiler.dialect
    value = arg.sql(dialect=dialect)
    if dialect == "snowflake":
        sql = f"({value}) = 'NaN'::FLOAT"
    elif dialect == "postgres":
        sql = f"({value}) = 'NaN'::FLOAT8"
    elif dialect == "bigquery":
        sql = f"IS_NAN({value})"
    else:
        sql = f"ISNAN({value})"
    return sge.Var(this=sql)


def register_is_nan() -> None:
    from ibis.backends.sql.compilers.bigquery import BigQueryCompiler
    from ibis.backends.sql.compilers.duckdb import DuckDBCompiler
    from ibis.backends.sql.compilers.postgres import PostgresCompiler
    from ibis.backends.sql.compilers.snowflake import SnowflakeCompiler

    DuckDBCompiler.visit__IsNaN = _compile_is_nan  # ty: ignore[unresolved-attribute]
    PostgresCompiler.visit__IsNaN = _compile_is_nan  # ty: ignore[unresolved-attribute]
    SnowflakeCompiler.visit__IsNaN = _compile_is_nan  # ty: ignore[unresolved-attribute]
    BigQueryCompiler.visit__IsNaN = _compile_is_nan  # ty: ignore[unresolved-attribute]


register_is_nan()


def nonfinite_parts(value: ir.FloatingValue) -> tuple[ir.BooleanValue, ir.BooleanValue]:
    """Return portable ``(is_nan, is_infinite)`` predicates."""
    is_nan = _IsNaN(cast(ops.Value[dt.Floating], value.op())).to_expr()
    is_infinite = value.abs() == ibis.literal(float("inf"), type=value.type())
    return is_nan, is_infinite
