from __future__ import annotations

import pytest
import sqlglot

from integration.warehouse_execution._compile import (
    _inline_empty_ctes,
    compile_statement_cases,
)


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
        for statement in (sql, _inline_empty_ctes(sql, case)):
            ast = sqlglot.parse_one(statement, read=dialect)
            ctes = {cte.alias for cte in ast.find_all(sqlglot.exp.CTE)}
            referenced = {table.name for table in ast.find_all(sqlglot.exp.Table)}
            sources = set(case.table_names.values())
            assert referenced <= sources | ctes, case.name
            assert sources <= referenced | ctes, case.name
