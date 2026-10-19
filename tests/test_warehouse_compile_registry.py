from __future__ import annotations

import pytest
import sqlglot

from integration.warehouse_execution._compile import compile_statement_cases


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_registered_warehouse_statements_compile_offline(dialect):
    cases = compile_statement_cases(dialect)

    assert cases
    assert len({case.name for case in cases}) == len(cases)
    for case in cases:
        expression = case.build(case.table_names)
        assert expression.schema().names
        sql = case.compile(expression)
        assert sql.strip(), case.name
        ast = sqlglot.parse_one(sql, read=dialect)
        ctes = {cte.alias for cte in ast.find_all(sqlglot.exp.CTE)}
        referenced = {table.name for table in ast.find_all(sqlglot.exp.Table)} - ctes
        assert referenced == set(case.table_names.values()), case.name
