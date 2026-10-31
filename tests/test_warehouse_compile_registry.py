from __future__ import annotations

import pytest
import sqlglot

from integration.warehouse_execution._compile import (
    _inline_empty_ctes,
    compile_case_names,
    compile_statement_cases,
    fixed_compile_cases,
)

# Each worker captures the public readouts once; shards spread rendering across workers.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
    pytest.mark.compile_registry,
]
_SHARDS = 8


# The first shard on each worker also pays for capturing the public readouts.
@pytest.mark.timeout(900)
@pytest.mark.parametrize("shard", range(_SHARDS))
@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
# A cold cache build compiles all registered statements before shard tests run.
@pytest.mark.timeout(900)
def test_registered_warehouse_statements_compile_offline(dialect, shard):
    cases = compile_statement_cases(dialect, shard=(shard, _SHARDS))

    assert cases
    assert len({case.name for case in cases}) == len(cases)
    for case in cases:
        expression = case.build(case.table_names)
        if not isinstance(expression, str):
            assert expression.schema().names
        sql = case.compile(expression)
        if dialect == "snowflake":
            assert "IS_NAN" not in sql.upper(), case.name
        assert sql.strip(), case.name
        assert len(sql) <= 1_000_000, (
            f"{dialect} statement exceeds the query-length limit: {case.name} ({len(sql)} characters)"
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


def test_registered_names_are_unique_and_include_captured_materializations():
    names = compile_case_names("snowflake")

    assert len(names) == len(set(names))
    assert any(name.startswith("cuped.materialize.") for name in names)
    assert any(name.startswith("cuped.readout.") for name in names)


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


def test_captured_statement_compilation_uses_sqlglot_ast(monkeypatch):
    import ibis

    from integration.warehouse_execution import _compile

    schema = ibis.schema({"value": "float64"})
    connection = ibis.duckdb.connect()
    expression = connection.sql("SELECT value FROM source", schema=schema)

    def reject_string_rendering(*_args, **_kwargs):
        pytest.fail("captured statements must reuse the compiler SQLGlot AST")

    monkeypatch.setattr(ibis, "to_sql", reject_string_rendering)
    case = _compile._captured_statement_case(
        expression, "bigquery", 0, schema_snapshot={"source": schema}
    )

    assert "FROM" in case.compile(case.build(case.table_names)).upper()


def test_captured_snowflake_isnan_uses_portable_comparison():
    import ibis

    from integration.warehouse_execution._compile import _captured_statement_case

    connection = ibis.duckdb.connect()
    schema = ibis.schema({"nonfinite": "boolean"})
    expression = connection.sql(
        "SELECT ISNAN(value) AS nonfinite FROM source",
        schema=schema,
    )
    case = _captured_statement_case(expression, "snowflake", 0, schema_snapshot={"source": schema})
    sql = case.compile(case.build(case.table_names))

    assert "IS_NAN" not in sql.upper()
    assert "'NaN'" in sql
    assert "CAST(" in sql.upper()


def test_captured_snowflake_isinf_uses_absolute_infinity_comparison():
    import ibis

    from integration.warehouse_execution._compile import _captured_statement_case

    connection = ibis.duckdb.connect()
    schema = ibis.schema({"infinite": "boolean"})
    expression = connection.sql(
        "SELECT ISINF(value) AS infinite FROM source",
        schema=schema,
    )
    case = _captured_statement_case(expression, "snowflake", 0, schema_snapshot={"source": schema})
    sql = case.compile(case.build(case.table_names))

    assert "IS_INF" not in sql.upper()
    assert "ABS(" in sql.upper()
    assert "INF" in sql.upper()


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
        case for case in fixed_compile_cases(dialect) if case.name == "digest.exposures.row_token"
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


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_shared_source_table_is_created_once_when_first_referenced(dialect):
    from integration.warehouse_execution._compile import (
        BigQueryCompileSession,
        CompileSourceTables,
        drop_compile_source_tables,
        fixed_compile_cases,
        run_compile_case,
    )

    case = next(
        case for case in fixed_compile_cases(dialect) if case.name == "digest.exposures.row_token"
    )
    if dialect == "snowflake":
        connection = _CapturingSnowflakeConnection()
        statements = connection.statements
    else:
        client = _CapturingBigQueryClient()
        connection = BigQueryCompileSession(client, object(), lambda **kwargs: kwargs)
        statements = client.statements

    source_tables = CompileSourceTables([], set())
    assert (
        run_compile_case(connection, case, shared_tables=source_tables)
        == "shared schema-only tables"
    )
    assert len(statements) == 2
    assert statements[0].upper().startswith("CREATE TABLE")
    next_case = next(
        case for case in fixed_compile_cases(dialect) if case.name == "digest.exposures.row_hash"
    )
    assert run_compile_case(connection, next_case, shared_tables=source_tables) == (
        "shared schema-only tables"
    )
    assert len(statements) == 3
    assert (
        statements[-1]
        .upper()
        .startswith("EXPLAIN USING TEXT " if dialect == "snowflake" else "SELECT")
    )
    drop_compile_source_tables(connection, dialect, source_tables)
    assert statements[-1].upper().startswith("DROP TABLE")


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_shared_source_refusal_uses_inline_typed_cte(dialect):
    from integration.warehouse_execution._compile import (
        BigQueryCompileSession,
        CompileSourceTables,
        run_compile_case,
    )

    case = next(
        case for case in fixed_compile_cases(dialect) if case.name == "digest.exposures.row_token"
    )
    if dialect == "snowflake":
        connection = _CapturingSnowflakeConnection()
        statements = connection.statements
    else:
        client = _CapturingBigQueryClient()
        connection = BigQueryCompileSession(client, object(), lambda **kwargs: kwargs)
        statements = client.statements

    source_tables = CompileSourceTables([], set(case.table_names.values()))
    assert run_compile_case(connection, case, shared_tables=source_tables) == (
        "inline empty typed CTEs"
    )
    assert len(statements) == 1
    statement = statements[0].removeprefix("EXPLAIN USING TEXT ")
    assert "WITH" in statement.upper()


def test_all_captured_cases_are_registered():
    from integration.warehouse_execution._compile import compile_case_names, compile_statement_cases

    cases = compile_statement_cases("snowflake")

    assert [case.name for case in cases] == compile_case_names("snowflake")


@pytest.mark.parametrize("dialect", ["snowflake", "bigquery"])
def test_cached_templates_rebuild_case_sql(dialect):
    from integration.warehouse_execution import _compile

    data = _compile._captured_compile_data(dialect)
    cases = {case.name: case for case in _compile._captured_cuped_statement_cases(dialect)}

    assert len(data) == len(cases)
    for name, _schemas, template, table_roles in data:
        case = cases[name]
        statement = case.build(case.table_names)
        for placeholder, relation in table_roles:
            statement = statement.replace(case.table_names[relation], placeholder)
        assert statement == template


def test_cache_key_changes_when_a_registered_adjustment_changes(tmp_path):
    import shutil
    from pathlib import Path

    from tests._shared_cache import fixture_cache_key

    adjustment = tmp_path / "increment" / "estimation" / "_adjust" / "iptw.py"
    adjustment.parent.mkdir(parents=True)
    source = (
        Path(__file__).resolve().parents[1] / "increment" / "estimation" / "_adjust" / "iptw.py"
    )
    shutil.copy2(source, adjustment)
    (tmp_path / "uv.lock").write_text("")
    modules = {"increment.estimation._adjust.iptw"}

    original = fixture_cache_key(tmp_path, package_modules=modules)
    adjustment.write_text(adjustment.read_text() + "\n# changed\n")
    changed = fixture_cache_key(tmp_path, package_modules=modules)

    assert changed != original


def test_cache_key_tracks_the_recorded_root_initializer(tmp_path):
    import shutil
    from pathlib import Path

    from tests._shared_cache import fixture_cache_key

    repository = Path(__file__).resolve().parents[1]
    source = repository / "increment" / "__init__.py"
    initializer = tmp_path / "increment" / "__init__.py"
    initializer.parent.mkdir(parents=True)
    shutil.copy2(source, initializer)
    (tmp_path / "uv.lock").write_text("")
    modules = {
        line
        for line in (
            repository / "integration" / "warehouse_execution" / "_compile_dependencies.txt"
        )
        .read_text()
        .splitlines()
        if line and not line.startswith("#")
    }

    original = fixture_cache_key(tmp_path, package_modules=modules)
    initializer.write_text(initializer.read_text() + "\n# changed\n")
    changed = fixture_cache_key(tmp_path, package_modules=modules)

    assert changed != original


def test_recorded_compile_dependencies_are_current():
    from pathlib import Path

    from scripts.trace_compile_dependencies import check

    assert check(Path(__file__).resolve().parents[1]) == []


def test_cached_compile_dependency_record_reports_missing_and_stale_modules(tmp_path, monkeypatch):
    import json

    from scripts import trace_compile_dependencies as dependency_trace

    monkeypatch.setattr(
        "integration.warehouse_execution._compile._captured_compile_cache_key",
        lambda dialect, root=None: "current-key",
    )
    dependency_file = tmp_path / dependency_trace.DEPENDENCIES_PATH
    dependency_file.parent.mkdir(parents=True)
    dependency_file.write_text("increment.recorded\nincrement.stale\n")
    record = dependency_trace._dependency_record_path(tmp_path, "current-key")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(["increment.recorded", "increment.imported"]))
    monkeypatch.setattr(
        dependency_trace,
        "trace",
        lambda root, dialect: pytest.fail("cached dependency recording should be reused"),
    )

    assert dependency_trace.check(tmp_path) == [
        "missing: increment.imported",
        "stale: increment.stale",
    ]


def test_compile_dependency_record_from_another_cache_key_is_ignored(tmp_path, monkeypatch):
    import json

    from scripts import trace_compile_dependencies as dependency_trace

    monkeypatch.setattr(
        "integration.warehouse_execution._compile._captured_compile_cache_key",
        lambda dialect, root=None: "current-key",
    )
    dependency_file = tmp_path / dependency_trace.DEPENDENCIES_PATH
    dependency_file.parent.mkdir(parents=True)
    dependency_file.write_text("increment.current\n")
    other_record = dependency_trace._dependency_record_path(tmp_path, "other-key")
    other_record.parent.mkdir(parents=True, exist_ok=True)
    other_record.write_text(json.dumps(["increment.current", "increment.stale"]))
    traces = []

    def trace(root, dialect):
        traces.append((root, dialect))
        return ["increment.current"]

    monkeypatch.setattr(dependency_trace, "trace", trace)

    assert dependency_trace.check(tmp_path) == []
    assert traces == [(tmp_path, "snowflake")]


def test_compile_dependency_build_forwards_child_error_before_raising(
    tmp_path, monkeypatch, capsys
):
    from scripts import trace_compile_dependencies as dependency_trace

    monkeypatch.setattr(
        dependency_trace, "_BUILD_CAPTURE", "raise RuntimeError('child build exploded')"
    )

    with pytest.raises(dependency_trace.subprocess.CalledProcessError):
        dependency_trace._build_and_record(tmp_path, "snowflake")

    assert "child build exploded" in capsys.readouterr().err


def test_cold_compile_build_records_dependencies_for_check(tmp_path, monkeypatch):
    from integration.warehouse_execution import _compile
    from scripts import trace_compile_dependencies as dependency_trace

    dependency_file = tmp_path / dependency_trace.DEPENDENCIES_PATH
    dependency_file.parent.mkdir(parents=True)
    dependency_file.write_text("increment.synthetic\n")
    monkeypatch.setattr(
        _compile,
        "__file__",
        str(tmp_path / "integration" / "warehouse_execution" / "_compile.py"),
    )
    monkeypatch.setattr(
        _compile, "_captured_compile_cache_key", lambda dialect, root=None: "synthetic-key"
    )
    monkeypatch.setattr(
        dependency_trace,
        "_build_and_record",
        lambda root, dialect: ((), ["increment.synthetic"]),
    )
    monkeypatch.setattr(
        dependency_trace,
        "trace",
        lambda root, dialect: pytest.fail("a successful cold build must record dependencies"),
    )
    _compile._captured_compile_data.cache_clear()
    try:
        assert _compile._captured_compile_data("snowflake") == ()
        assert dependency_trace.check(tmp_path) == []
    finally:
        _compile._captured_compile_data.cache_clear()


def test_compile_dependency_check_uses_the_requested_root_cache_key(tmp_path, monkeypatch):
    import json

    from integration.warehouse_execution import _compile
    from scripts import trace_compile_dependencies as dependency_trace

    dependency_file = tmp_path / dependency_trace.DEPENDENCIES_PATH
    dependency_file.parent.mkdir(parents=True)
    dependency_file.write_text("increment.current\n")
    record = dependency_trace._dependency_record_path(tmp_path, "requested-root-key")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(["increment.current"]))
    keys = []

    def cache_key(dialect, root=None):
        keys.append((dialect, root))
        return "requested-root-key" if root == tmp_path else "importing-root-key"

    monkeypatch.setattr(_compile, "_captured_compile_cache_key", cache_key)
    monkeypatch.setattr(
        dependency_trace,
        "trace",
        lambda root, dialect: pytest.fail("the requested-root record should be reused"),
    )

    assert dependency_trace.check(tmp_path) == []
    assert keys == [("snowflake", tmp_path)]


def test_invalid_compile_dependency_record_falls_back_to_trace(tmp_path, monkeypatch):
    from integration.warehouse_execution import _compile
    from scripts import trace_compile_dependencies as dependency_trace

    monkeypatch.setattr(
        _compile, "_captured_compile_cache_key", lambda dialect, root=None: "current-key"
    )
    dependency_file = tmp_path / dependency_trace.DEPENDENCIES_PATH
    dependency_file.parent.mkdir(parents=True)
    dependency_file.write_text("increment.current\n")
    record = dependency_trace._dependency_record_path(tmp_path, "current-key")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text("{invalid json")
    traces = []

    def trace(root, dialect):
        traces.append((root, dialect))
        return ["increment.current"]

    monkeypatch.setattr(dependency_trace, "trace", trace)

    assert dependency_trace.check(tmp_path) == []
    assert traces == [(tmp_path, "snowflake")]
