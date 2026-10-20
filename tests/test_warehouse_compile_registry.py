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
    statement_lengths = {}
    for case in cases:
        expression = case.build(case.table_names)
        if not isinstance(expression, str):
            assert expression.schema().names
        sql = case.compile(expression)
        statement_lengths[case.name] = len(sql)
        assert sql.strip(), case.name
        assert statement_lengths[case.name] <= 1_000_000, (
            f"{dialect} statement exceeds the query-length limit: "
            f"{case.name} ({statement_lengths[case.name]} characters)"
        )
        for statement in (sql, _inline_empty_ctes(sql, case)):
            ast = sqlglot.parse_one(statement, read=dialect)
            ctes = {cte.alias for cte in ast.find_all(sqlglot.exp.CTE)}
            referenced = {table.name for table in ast.find_all(sqlglot.exp.Table)}
            sources = set(case.table_names.values())
            assert referenced <= sources | ctes, case.name
            assert sources <= referenced | ctes, case.name
    assert len(statement_lengths) == len(cases)


class _CapturingCursor:
    def __init__(self, statements):
        self.statements = statements

    def execute(self, statement):
        self.statements.append(statement)

    def close(self):
        pass


class _CapturingSnowflakeConnection:
    def __init__(self):
        self.statements = []

    def cursor(self):
        return _CapturingCursor(self.statements)


class _CapturingBigQueryClient:
    def __init__(self):
        self.statements = []

    def query(self, statement, *, job_config):
        self.statements.append(statement)
        return _CapturingJob()


class _CapturingJob:
    def result(self):
        return None


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_schema_only_ddl_matches_rendered_table_identifier(dialect):
    from integration.warehouse_execution._compile import (
        BigQueryCompileSession,
        run_compile_case,
    )

    case = next(
        case
        for case in compile_statement_cases(dialect)
        if case.name == "digest.exposures.row_token"
    )
    if dialect == "snowflake":
        connection = _CapturingSnowflakeConnection()
        statements = connection.statements
    else:
        client = _CapturingBigQueryClient()
        connection = BigQueryCompileSession(client, object(), lambda **kwargs: kwargs)
        statements = client.statements

    assert run_compile_case(connection, case) == "empty schema-only tables"
    create_ast = sqlglot.parse_one(statements[0], read=dialect)
    query_sql = statements[1]
    if dialect == "snowflake":
        query_sql = query_sql.removeprefix("EXPLAIN USING TEXT ")
    query_ast = sqlglot.parse_one(query_sql, read=dialect)
    source_name = next(iter(case.table_names.values()))
    create_table = next(create_ast.find_all(sqlglot.exp.Table))
    query_table = next(
        table for table in query_ast.find_all(sqlglot.exp.Table) if table.name == source_name
    )
    assert create_table.name == query_table.name == source_name
    assert create_table.this.args.get("quoted") == query_table.this.args.get("quoted")
    assert create_table.args.get("db") == query_table.args.get("db")
    assert create_table.args.get("catalog") == query_table.args.get("catalog")


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_negative_control_is_registered_and_renders_offline(dialect):
    from integration.warehouse_execution._compile import compile_negative_control

    case = compile_negative_control(dialect)
    sql = case.compile(case.build(case.table_names))
    ast = sqlglot.parse_one(sql, read=dialect)
    referenced = {table.name for table in ast.find_all(sqlglot.exp.Table)}
    assert referenced <= set(case.table_names.values())
    assert set(case.table_names.values()) <= referenced
