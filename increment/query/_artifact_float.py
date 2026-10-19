"""Exact decimal float tokens with bounded-precision integer arithmetic."""

from __future__ import annotations

from functools import cache

import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.rules as rlz


class _CloudFloat17g(ops.Value):
    arg: ops.Value[dt.Floating]

    dtype = dt.string
    shape = rlz.shape_like("arg")


@cache
def _power_cases(base: int) -> str:
    # Splitting k = 23q + r leaves both multiplication factors below 10**16.
    return (
        "CASE quotient "
        + " ".join(f"WHEN {q} THEN '{base ** (23 * q)}'" for q in range(50))
        + " END"
    )


def _integer_product_ctes(name: str, source: str, factor: str, dialect: str) -> str:
    """Multiply a decimal integer string by a factor below B = 10**16.

    For t_j = a_j*m = B*h_j + l_j, set s_j = l_j + h_(j-1).
    A carry is generated at s_j >= B, reset at s_j < B-1, and otherwise
    propagated. Two prefix maxima give every carry without recursion.
    All products have at most 32 digits, including on NUMBER(38,0).
    """
    numeric = "BIGNUMERIC" if dialect == "bigquery" else "DECIMAL(38,0)"
    text = "STRING" if dialect == "bigquery" else "VARCHAR"
    integer = "INT64" if dialect == "bigquery" else "INTEGER"
    if dialect == "snowflake":
        indices = (
            "(SELECT column1 AS j FROM VALUES "
            + ", ".join(f"({j})" for j in range(53))
            + ") AS offsets"
        )
        index = "j"
        joined = f"LISTAGG(LPAD(CAST(digit AS {text}), 16, '0'), '') WITHIN GROUP (ORDER BY j DESC)"
    else:
        indices = (
            "UNNEST(GENERATE_ARRAY(0, 52)) AS j"
            if dialect == "bigquery"
            else "UNNEST(range(53)) AS offsets(j)"
        )
        index = "j"
        joined = f"STRING_AGG(LPAD(CAST(digit AS {text}), 16, '0'), '' ORDER BY j DESC)"
    previous = "ORDER BY j ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
    return f"""
      {name}_products AS (
        SELECT {index} AS j,
          CAST(RIGHT(LEFT(CONCAT('0000000000000000', digits),
            LENGTH(digits) + 16 - 16 * {index}), 16) AS {numeric}) * {factor} AS product
        FROM {source}, {indices}
        WHERE {index} <= CAST(CEIL(LENGTH(digits) / 16.0) AS {integer})
      ),
      {name}_parts AS (
        SELECT j, MOD(product, 10000000000000000) AS low,
          CASE WHEN product < 10000000000000000 THEN CAST(0 AS {numeric}) ELSE
            CAST(LEFT(CAST(product AS {text}), LENGTH(CAST(product AS {text})) - 16) AS {numeric}) END AS high
        FROM {name}_products
      ),
      {name}_sums AS (
        SELECT j, low + LAG(high, 1, 0) OVER (ORDER BY j) AS s FROM {name}_parts
      ),
      {name}_digits AS (
        SELECT j, MOD(s + CASE WHEN
          COALESCE(MAX(CASE WHEN s >= 10000000000000000 THEN j END) OVER ({previous}), -1) >
          COALESCE(MAX(CASE WHEN s < 9999999999999999 THEN j END) OVER ({previous}), -1)
          THEN 1 ELSE 0 END, 10000000000000000) AS digit
        FROM {name}_sums
      ),
      {name} AS (SELECT LTRIM({joined}, '0') AS digits FROM {name}_digits)
    """


def _float17g_sql(value: str, *, dialect: str) -> str:
    """Render finite binary64 as exact half-even, 17-significant-digit %g.

    Power-of-two scaling recovers x = sign*m*2**e exactly, including
    subnormals. For e < 0, D=m*5**-e and scale=-e; otherwise D=m*2**e
    and scale=0. Integer limbs retain every digit of D before rounding.
    Floating arithmetic is confined to exact binary rescaling; logarithms
    only estimate an exponent that is corrected before extracting m.
    """
    numeric = "BIGNUMERIC" if dialect == "bigquery" else "DECIMAL(38,0)"
    text = "STRING" if dialect == "bigquery" else "VARCHAR"
    integer = "INT64" if dialect == "bigquery" else "INTEGER"
    if dialect == "bigquery":
        sign = f"IEEE_DIVIDE(1.0, ({value})) < 0"
    elif dialect == "snowflake":
        sign = f"LEFT(CAST(({value}) AS VARCHAR), 1) = '-'"
    else:
        sign = f"signbit(({value}))"
    small_factor = (
        "CASE remainder "
        + " ".join(f"WHEN {r} THEN CASE WHEN e < 0 THEN {5**r} ELSE {2**r} END" for r in range(23))
        + " END"
    )
    first = _integer_product_ctes("first_product", "powers", "mantissa", dialect)
    second = _integer_product_ctes("second_product", "second_input", "small_factor", dialect)
    return f"""CONCAT(CASE WHEN {sign} THEN '-' ELSE '' END, (
      WITH input_value AS (SELECT {value} AS x),
      magnitude AS (SELECT x, CASE WHEN x = 0 OR x IS NULL OR NOT
        (x BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308)
        THEN 1.0 ELSE ABS(x) END AS a FROM input_value),
      guess AS (SELECT x, a, CAST(FLOOR(LN(a) / LN(2.0)) AS {integer}) AS g FROM magnitude),
      rescaled AS (
        SELECT x, g, (a / POWER(2.0, LEAST(512, GREATEST(-512, g)))) /
          POWER(2.0, g - LEAST(512, GREATEST(-512, g))) AS z FROM guess
      ),
      normalized AS (
        SELECT x, g - 52 + CASE WHEN z < 1 THEN -1 WHEN z >= 2 THEN 1 ELSE 0 END AS e,
          z * CASE WHEN z < 1 THEN 2 WHEN z >= 2 THEN 0.5 ELSE 1 END AS u
        FROM rescaled
      ),
      binary_value AS (
        SELECT x, e, CAST(FLOOR(u * 67108864) AS {numeric}) * 67108864 +
          CAST((u * 67108864 - FLOOR(u * 67108864)) * 67108864 AS {numeric}) AS mantissa
        FROM normalized
      ),
      powers_index AS (
        SELECT *, CAST(FLOOR(ABS(e) / 23.0) AS {integer}) AS quotient,
          MOD(ABS(e), 23) AS remainder FROM binary_value
      ),
      powers AS (
        SELECT *, CASE WHEN e < 0 THEN {_power_cases(5)} ELSE {_power_cases(2)} END AS digits,
          CAST({small_factor} AS {numeric}) AS small_factor FROM powers_index
      ),
      {first},
      second_input AS (SELECT first_product.digits, small_factor FROM first_product CROSS JOIN powers),
      {second},
      decimal_value AS (
        SELECT x, e, digits, LENGTH(digits) AS n FROM second_product CROSS JOIN binary_value
      ),
      rounding AS (
        SELECT x, n - 1 - GREATEST(-e, 0) AS exponent10,
          CAST(LEFT(digits, 17) AS {numeric}) + CASE WHEN n > 17 AND
            (SUBSTR(digits, 18, 1) > '5' OR (SUBSTR(digits, 18, 1) = '5' AND
              (RTRIM(SUBSTR(digits, 19), '0') <> '' OR MOD(CAST(LEFT(digits, 17) AS {numeric}), 2) = 1)))
            THEN 1 ELSE 0 END AS coefficient
        FROM decimal_value
      ),
      formatted AS (
        SELECT x, exponent10 + CASE WHEN coefficient = 100000000000000000 THEN 1 ELSE 0 END AS exponent10,
          RTRIM(CAST(coefficient AS {text}), '0') AS digits FROM rounding
      )
      SELECT MAX(
        CASE WHEN x = 0 THEN '0'
          WHEN exponent10 < -4 OR exponent10 >= 17 THEN
            CONCAT(LEFT(digits, 1), CASE WHEN LENGTH(digits) > 1 THEN CONCAT('.', SUBSTR(digits, 2)) ELSE '' END,
              'e', CASE WHEN exponent10 < 0 THEN '-' ELSE '+' END,
              LPAD(CAST(ABS(exponent10) AS {text}), CAST(GREATEST(2, LENGTH(CAST(ABS(exponent10) AS {text}))) AS {integer}), '0'))
          WHEN exponent10 < 0 THEN CONCAT('0.', REPEAT('0', -exponent10 - 1), digits)
          WHEN LENGTH(digits) <= exponent10 + 1 THEN CONCAT(digits, REPEAT('0', exponent10 + 1 - LENGTH(digits)))
          ELSE CONCAT(LEFT(digits, exponent10 + 1), '.', SUBSTR(digits, exponent10 + 2)) END)
      FROM formatted
    ))"""


def _compile_cloud_float17g(compiler, _op, *, arg):
    import sqlglot.expressions as sge

    dialect = compiler.dialect
    sql = _float17g_sql(arg.sql(dialect=dialect), dialect=dialect)
    return sge.Var(this=sql)


def register_cloud_float17g() -> None:
    from ibis.backends.sql.compilers.bigquery import BigQueryCompiler
    from ibis.backends.sql.compilers.snowflake import SnowflakeCompiler

    BigQueryCompiler.visit__CloudFloat17g = _compile_cloud_float17g  # ty: ignore[unresolved-attribute]
    SnowflakeCompiler.visit__CloudFloat17g = _compile_cloud_float17g  # ty: ignore[unresolved-attribute]
