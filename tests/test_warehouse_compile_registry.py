from __future__ import annotations

import pytest
import sqlglot

from integration.warehouse_execution._compile import (
    _inline_empty_ctes,
    compile_statement_cases,
)

# Registry compilation captures public readouts once per worker and compiles many SQL cases.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.xdist_group("warehouse_compile_registry"),
    pytest.mark.timeout(600),
]


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
        if dialect == "snowflake":
            assert "IS_NAN" not in sql.upper(), case.name
        statement_lengths[case.name] = len(sql)
        assert sql.strip(), case.name
        assert statement_lengths[case.name] <= 1_000_000, (
            f"{dialect} statement exceeds the query-length limit: "
            f"{case.name} ({statement_lengths[case.name]} characters)"
        )
        for statement in (sql, _inline_empty_ctes(sql, case)):
            ast = sqlglot.parse_one(statement, read=dialect)
            if dialect == "bigquery":
                schema_columns = {
                    column for schema in case.schemas.values() for column in schema.keys()
                }
                quoted_columns = {
                    literal.this
                    for literal in ast.find_all(sqlglot.exp.Literal)
                    if literal.is_string and literal.this in schema_columns
                }
                assert not quoted_columns, (
                    f"{case.name}: quoted identifier literals {quoted_columns}"
                )
            ctes = {cte.alias for cte in ast.find_all(sqlglot.exp.CTE)}
            referenced = {table.name for table in ast.find_all(sqlglot.exp.Table)}
            sources = set(case.table_names.values())
            assert referenced <= sources | ctes, case.name
            assert sources <= referenced | ctes, case.name
    assert len(statement_lengths) == len(cases)


def test_captured_statement_uses_schema_at_capture_time(monkeypatch):
    import ibis

    from integration.warehouse_execution import _compile

    schema = ibis.schema({"_qr_projection_score_0_0": "float64"})
    connection = ibis.duckdb.connect()
    expression = connection.sql(
        "SELECT _qr_projection_score_0_0 FROM _cuped_qr_stage", schema=schema
    )
    monkeypatch.setattr(
        _compile,
        "_CAPTURED_CUPED_EXPRESSION_SCHEMAS",
        [{"_cuped_qr_stage": schema}],
    )
    monkeypatch.setattr(
        _compile,
        "_CAPTURED_CUPED_SCHEMAS",
        {"_cuped_qr_stage": ibis.schema({"unit_id": "string"})},
    )

    case = _compile._captured_statement_case(expression, "bigquery", 0)

    assert case.schemas["_cuped_qr_stage"] == schema


def test_captured_snowflake_isnan_uses_portable_comparison():
    import ibis
    from ibis.expr import operations as ops

    from integration.warehouse_execution._compile import _transpile_captured_raw_sql

    connection = ibis.duckdb.connect()
    expression = connection.sql(
        "SELECT ISNAN(value) AS nonfinite FROM source",
        schema=ibis.schema({"nonfinite": "boolean"}),
    )

    translated = _transpile_captured_raw_sql(expression, "snowflake")
    query = next(operation.query for operation in translated.op().find(ops.SQLQueryResult))
    sql = ibis.to_sql(translated, dialect="snowflake")

    assert "IS_NAN" not in query.upper()
    assert "'NaN'::FLOAT" in query
    assert "IS_NAN" not in sql.upper()


def test_captured_snowflake_isinf_uses_absolute_infinity_comparison():
    import ibis
    from ibis.expr import operations as ops

    from integration.warehouse_execution._compile import _transpile_captured_raw_sql

    connection = ibis.duckdb.connect()
    expression = connection.sql(
        "SELECT ISINF(value) AS infinite FROM source",
        schema=ibis.schema({"infinite": "boolean"}),
    )

    translated = _transpile_captured_raw_sql(expression, "snowflake")
    query = next(operation.query for operation in translated.op().find(ops.SQLQueryResult))

    assert "IS_INF" not in query.upper()
    assert "ABS(" in query.upper()
    assert "INF" in query.upper()


def test_captured_materializations_are_compile_cases():
    cases = compile_statement_cases("snowflake")

    assert any(case.name.startswith("cuped.materialize.") for case in cases)


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


@pytest.mark.parametrize(
    ("dialect", "module_name", "test_name"),
    [
        (
            "snowflake",
            "integration.warehouse_execution.test_snowflake_compile",
            "test_snowflake_statement_compiles",
        ),
        (
            "bigquery",
            "integration.warehouse_execution.test_bigquery_compile",
            "test_bigquery_statement_compiles",
        ),
    ],
)
def test_cloud_compile_cases_match_the_offline_registry(dialect, module_name, test_name):
    from importlib import import_module

    cloud_test = getattr(import_module(module_name), test_name)
    mark = next(mark for mark in cloud_test.pytestmark if mark.name == "parametrize")
    cloud_cases = mark.args[1]
    expected = {case.name for case in compile_statement_cases(dialect)}
    assert {case.name for case in cloud_cases} == expected
