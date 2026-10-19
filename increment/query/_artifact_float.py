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


def _integer_product_ctes(name: str, source: str, factor: str) -> str:
    """Multiply a decimal integer string by a factor below B = 10**16.

    For t_j = a_j*m = B*h_j + l_j, set s_j = l_j + h_(j-1). A carry
    is generated at s_j >= B, reset at s_j < B-1, and otherwise propagated.
    Two prefix maxima recover every carry without recursion.
    """
    indices = "UNNEST(range(53)) AS offsets(j)"
    joined = "STRING_AGG(LPAD(CAST(digit AS VARCHAR), 16, '0'), '' ORDER BY j DESC)"
    previous = "ORDER BY j ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
    return f"""
      {name}_products AS (
        SELECT j AS j,
          CAST(RIGHT(LEFT(CONCAT('0000000000000000', digits),
            LENGTH(digits) + 16 - 16 * j), 16) AS DECIMAL(38,0)) * {factor} AS product
        FROM {source}, {indices}
        WHERE j <= CAST(CEIL(LENGTH(digits) / 16.0) AS INTEGER)
      ),
      {name}_parts AS (
        SELECT j, MOD(product, 10000000000000000) AS low,
          CASE WHEN product < 10000000000000000 THEN CAST(0 AS DECIMAL(38,0)) ELSE
            CAST(LEFT(CAST(product AS VARCHAR), LENGTH(CAST(product AS VARCHAR)) - 16) AS DECIMAL(38,0)) END AS high
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


def _snowflake_float17g_sql(value: str) -> str:
    """Format binary64 with Snowflake's native 17-digit scientific model."""
    scientific = f"TO_CHAR(({value}), 'FM9.9999999999999999EEEE')"
    mantissa = f"SPLIT_PART({scientific}, 'E', 1)"
    exponent = f"TO_NUMBER(SPLIT_PART({scientific}, 'E', 2))"
    trimmed = f"RTRIM(RTRIM({mantissa}, '0'), '.')"
    sign = f"IFF(LEFT({mantissa}, 1) = '-', '-', '')"
    digits = f"REPLACE(REPLACE({trimmed}, '-', ''), '.', '')"
    position = f"({exponent} + 1)"
    scientific_token = (
        f"CONCAT({trimmed}, 'e', IFF({exponent} < 0, '-', '+'), "
        f"LPAD(TO_VARCHAR(ABS({exponent})), 2, '0'))"
    )
    fixed_token = (
        f"CASE WHEN {position} <= 0 THEN CONCAT({sign}, '0.', "
        f"REPEAT('0', -{position}), {digits}) "
        f"WHEN {position} >= LENGTH({digits}) THEN CONCAT({sign}, {digits}, "
        f"REPEAT('0', {position} - LENGTH({digits}))) "
        f"ELSE CONCAT({sign}, SUBSTR({digits}, 1, {position}), '.', "
        f"SUBSTR({digits}, {position} + 1)) END"
    )
    return (
        f"CASE WHEN ({value}) = 0 THEN CONCAT({sign}, '0') "
        f"WHEN {exponent} < -4 OR {exponent} >= 17 THEN {scientific_token} "
        f"ELSE {fixed_token} END"
    )


def _float17g_sql(value: str, *, dialect: str) -> str:
    """Render finite binary64 as exact half-even, 17-significant-digit %g.

    Power-of-two scaling recovers x = sign*m*2**e exactly, including
    subnormals. For e < 0, D=m*5**-e and scale=-e; otherwise D=m*2**e
    and scale=0. Integer limbs retain every digit of D before rounding.
    Floating arithmetic is confined to exact binary rescaling; logarithms
    only estimate an exponent that is corrected before extracting m.
    """
    if dialect == "bigquery":
        return f"FORMAT('%.17g', ({value}))"
    if dialect == "snowflake":
        return _snowflake_float17g_sql(value)
    numeric = "DECIMAL(38,0)"
    text = "VARCHAR"
    integer = "INTEGER"
    sign = f"signbit(({value}))"
    small_factor = (
        "CASE remainder "
        + " ".join(f"WHEN {r} THEN CASE WHEN e < 0 THEN {5**r} ELSE {2**r} END" for r in range(23))
        + " END"
    )
    first = _integer_product_ctes("first_product", "powers", "mantissa")
    second = _integer_product_ctes("second_product", "second_input", "small_factor")
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
