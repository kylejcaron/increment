"""Compile generated warehouse SQL without reading or writing warehouse data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from uuid import uuid4

import ibis
import ibis.expr.datatypes as dt


@dataclass(frozen=True, slots=True)
class CompileCase:
    name: str
    schemas: dict[str, ibis.Schema]
    builder: Callable[[dict[str, str], str], object]
    table_names: dict[str, str]
    dialect: str

    def build(self, table_names: dict[str, str]):
        return self.builder(table_names, self.dialect)

    def compile(self, expression) -> str:
        if isinstance(expression, str):
            return expression
        return str(ibis.to_sql(expression, dialect=self.dialect))


@dataclass(frozen=True, slots=True)
class BigQueryCompileSession:
    client: object
    default_dataset: object
    query_config_factory: Callable[..., object]


class WarehouseCompileError(AssertionError):
    def __init__(self, dialect: str, statement_name: str, engine_message: str):
        self.dialect = dialect
        self.statement_name = statement_name
        self.engine_message = engine_message
        super().__init__(
            f"{dialect} compile failed for statement {statement_name}: {engine_message}"
        )


def _schema(fields):
    types = {
        "STRING": dt.string,
        "INT64": dt.int64,
        "FLOAT64": dt.float64,
        "DATE": dt.date,
        "TIMESTAMP_UTC_US": dt.Timestamp("UTC"),
        "BOOLEAN": dt.boolean,
    }
    return ibis.schema({name: types[tag] for name, tag, _nullable in fields})


def _digest_builder(key, stage, fields_override=None):
    def build(table_names, dialect):
        from increment.query.artifact_digest import (
            _BUCKET_HEX_LEN,
            _row_token_expr,
            _sql_binary_hex_expr,
            _sql_digest_sha256_binary_expr,
        )
        from increment.query.schemas import UNIT_DAY_ARTIFACT_RELATION_SCHEMAS

        fields = fields_override or UNIT_DAY_ARTIFACT_RELATION_SCHEMAS[key]
        table = ibis.table(_schema(fields), name=table_names[key])
        tags = {
            "STRING": (1, "string"),
            "INT64": (2, "int"),
            "FLOAT64": (3, "float"),
            "DATE": (4, "date"),
            "TIMESTAMP_UTC_US": (5, "timestamp"),
            "BOOLEAN": (6, "other"),
        }
        tagged = [(name, *tags[tag]) for name, tag, _nullable in fields]
        token = _row_token_expr(table, tagged, dialect=dialect, digest_format=4)
        if stage == "row_token":
            return table.select(row_token=token)
        hashed = _sql_digest_sha256_binary_expr(token, dialect=dialect)
        if stage == "row_hash":
            return table.select(row_hash=hashed)
        hex_hash = _sql_binary_hex_expr(hashed, dialect=dialect)
        if stage == "bucket":
            return table.select(bucket=hex_hash.substr(0, _BUCKET_HEX_LEN), row_hash=hex_hash)
        bucketed = table.select(bucket=hex_hash.substr(0, _BUCKET_HEX_LEN), row_hash=hex_hash)
        if stage == "bucket_digest":
            order = ibis.asc(bucketed.row_hash, nulls_first=dialect == "bigquery")
            return bucketed.group_by("bucket").aggregate(
                bucket_digest=bucketed.row_hash.group_concat(sep="", order_by=order),
                row_count=bucketed.count(),
            )
        if stage == "count":
            return table.aggregate(row_count=table.count())
        if stage == "validation":
            invalid = ibis.or_(
                *[table[name].isnull() for name, _tag, nullable in fields if not nullable]
            )
            return table.aggregate(row_count=table.count(), invalid_values=invalid.any())
        raise AssertionError(stage)

    return build


_CAPTURED_CUPED_EXPRESSIONS = None
_CAPTURED_CUPED_MATERIALIZATIONS = None
_CAPTURED_CUPED_SCHEMAS = {}
_CAPTURED_CUPED_EXPRESSION_SCHEMAS = []


def _capture_public_cuped_expressions():  # noqa: PLR0915
    """Capture DuckDB SQL emitted by public CUPED readouts and CTAS operations."""
    global _CAPTURED_CUPED_EXPRESSIONS, _CAPTURED_CUPED_MATERIALIZATIONS
    global _CAPTURED_CUPED_SCHEMAS, _CAPTURED_CUPED_EXPRESSION_SCHEMAS
    if _CAPTURED_CUPED_EXPRESSIONS is not None:
        return _CAPTURED_CUPED_EXPRESSIONS

    import datetime as py_datetime
    import tempfile
    import warnings
    from pathlib import Path

    import ibis.expr.types as ir
    import yaml

    from increment import Analysis
    from increment.errors import IncrementWarning
    from tests.warehouse_cuped_cases import mixed_cuped_definitions_yaml, mixed_cuped_event_rows

    arms = ("control", "treatment", "third")
    rows = []
    for arm_index, arm in enumerate(arms):
        for index in range(30):
            rows.append(
                {
                    "user_id": f"{arm}-{index}",
                    "variant": arm,
                    "date": py_datetime.date(2025, 1, 10),
                    "revenue": float(4 + index % 7 + arm_index * 1.25 + index % 3 * 0.17),
                    "orders": float(1 + index % 4),
                    "pre_revenue": None if index % 9 == 0 else float(8 + index % 13 + arm_index),
                    "country": ("north", "south", "north")[index % 3],
                    "device": ("mobile", "desktop", "desktop")[index % 3],
                    "new_user": "new" if index % 4 == 0 else "existing",
                    "signups": float((index * 7 + arm_index) % 9),
                    "region": None if index % 6 == 0 else ("east", "west")[index % 2],
                }
            )
    materializations = []
    expressions = []
    con = ibis.duckdb.connect()
    event_name = "cuped_compile_events"
    con.create_table(event_name, obj=mixed_cuped_event_rows(rows), overwrite=True)
    _CAPTURED_CUPED_SCHEMAS = {event_name: con.table(event_name).schema()}
    backend_type = type(con)
    original_to_pyarrow = backend_type.to_pyarrow
    original_create_table = backend_type.create_table

    def capture_to_pyarrow(backend, expression, *args, **kwargs):
        if backend is con and isinstance(expression, ir.Expr):
            expressions.append(expression)
            _CAPTURED_CUPED_EXPRESSION_SCHEMAS.append(dict(_CAPTURED_CUPED_SCHEMAS))
        return original_to_pyarrow(backend, expression, *args, **kwargs)

    def capture_create_table(backend, name, obj=None, *args, **kwargs):
        if backend is con and isinstance(obj, ir.Table):
            materializations.append((obj, dict(_CAPTURED_CUPED_SCHEMAS)))
            _CAPTURED_CUPED_SCHEMAS[str(name)] = obj.schema()
        return original_create_table(backend, name, obj, *args, **kwargs)

    backend_type.to_pyarrow = capture_to_pyarrow
    backend_type.create_table = capture_create_table
    try:
        with tempfile.TemporaryDirectory() as directory:
            source = f'"{event_name}"'
            for route in ("cells", "qr"):
                config = yaml.safe_load(mixed_cuped_definitions_yaml(source, breakout=True))
                experiment = config["experiments"][0]
                experiment["allocation"] = dict.fromkeys(arms, 1 / 3)
                covariates = [
                    {
                        "name": name,
                        "kind": "categorical",
                        "property": name,
                        "source": "events",
                    }
                    for name in ("country", "device")
                ]
                if route == "qr":
                    covariates.insert(
                        0,
                        {
                            "name": "pre_revenue",
                            "kind": "continuous",
                            "property": "pre_revenue",
                            "source": "events",
                            "missing": "impute-indicator",
                        },
                    )
                for role in ("primary", "guardrails", "secondaries"):
                    entries = experiment["plan"][role]
                    if isinstance(entries, dict):
                        entries = [entries]
                    for entry in entries:
                        entry["decision_method"]["covariates"] = covariates
                path = Path(directory) / f"{route}.yaml"
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                analysis = Analysis.from_definitions("mixed_cuped", path, con)
                try:
                    analysis.run(metrics=["revenue", "rpo"])
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", IncrementWarning)
                        analysis.run_breakout(metrics=["revenue", "rpo"])
                finally:
                    analysis.close()
    finally:
        backend_type.to_pyarrow = original_to_pyarrow
        backend_type.create_table = original_create_table
        con.drop_table(event_name, force=True)
        con.disconnect()
    _CAPTURED_CUPED_EXPRESSIONS = tuple(expressions)
    _CAPTURED_CUPED_MATERIALIZATIONS = tuple(materializations)
    return _CAPTURED_CUPED_EXPRESSIONS


def _transpile_captured_raw_sql(expression, dialect):
    """Transpile DuckDB SQL embedded in captured raw-query operations."""
    import sqlglot
    from ibis.expr import operations as ops
    from sqlglot import exp

    from increment.query._nonfinite import _is_nan_sql

    def transpile(query, description):
        if dialect == "snowflake":
            statements = sqlglot.parse(query, read="duckdb")
            if len(statements) != 1:
                raise ValueError(
                    f"captured CUPED SQL {description} must transpile to one statement, "
                    f"got {len(statements)}"
                )
            statement = statements[0]
            for is_nan in tuple(statement.find_all(exp.IsNan)):
                value = is_nan.this.sql(dialect=dialect)
                is_nan.replace(_is_nan_sql(dialect, value))
            for is_inf in tuple(statement.find_all(exp.IsInf)):
                infinity = exp.Cast(this=exp.Literal.string("Inf"), to=exp.DataType.build("DOUBLE"))
                is_inf.replace(exp.EQ(this=exp.Abs(this=is_inf.this.copy()), expression=infinity))
            return statement.sql(dialect=dialect)

        statements = sqlglot.transpile(query, read="duckdb", write=dialect)
        if len(statements) != 1:
            raise ValueError(
                f"captured CUPED SQL {description} must transpile to one statement, "
                f"got {len(statements)}"
            )
        return statements[0]

    root = expression.op()
    replacements = {}
    for operation in root.find(ops.SQLQueryResult):
        replacements[operation] = ops.SQLQueryResult(
            query=transpile(operation.query, "query result"),
            schema=operation.schema,
            source=operation.source,
        )
    for operation in root.find(ops.SQLStringView):
        replacements[operation] = ops.SQLStringView(
            parent=operation.parent,
            query=transpile(operation.query, "string view"),
            schema=operation.schema,
        )
    return root.replace(replacements).to_expr() if replacements else expression


def _captured_statement_case(
    expression,
    dialect,
    index,
    *,
    schema_snapshot=None,
    statement_kind="readout",
):
    import sqlglot
    from ibis.expr import operations as ops
    from sqlglot import exp

    target_expression = _transpile_captured_raw_sql(expression, dialect)
    rendered = str(ibis.to_sql(target_expression, dialect=dialect))
    query = sqlglot.parse_one(rendered, read=dialect)
    cte_names = {cte.alias_or_name for cte in query.find_all(exp.CTE)}
    if schema_snapshot is None:
        schema_snapshot = _CAPTURED_CUPED_EXPRESSION_SCHEMAS[index]
    relation_schemas = dict(schema_snapshot)
    relation_schemas.update(
        {
            relation.name: relation.schema
            for relation in expression.op().find(ops.DatabaseTable)
            if relation.name not in cte_names
        }
    )
    relation_schemas.update(
        {relation.name: relation.schema for relation in expression.op().find(ops.InMemoryTable)}
    )
    table_refs = {table.name for table in query.find_all(exp.Table) if table.name not in cte_names}
    schemas = {name: relation_schemas[name] for name in sorted(table_refs)}

    def build(table_names, _dialect):
        statement = sqlglot.parse_one(rendered, read=dialect)
        for table in statement.find_all(exp.Table):
            replacement = table_names.get(table.name)
            if replacement is None:
                continue
            table.set("this", exp.to_identifier(replacement, quoted=True))
            table.set("db", None)
            table.set("catalog", None)
        return statement.sql(dialect=dialect)

    return CompileCase(
        f"cuped.{statement_kind}.{index:04d}",
        schemas,
        build,
        {name: f"incr_compile_{name}_{uuid4().hex[:12]}" for name in schemas},
        dialect,
    )


def _captured_cuped_statement_cases(dialect, shard=None):
    """Render captured CUPED statements; ``shard=(k, n)`` keeps every n-th one from k."""
    captured = [
        (expression, index, None, "readout")
        for index, expression in enumerate(_capture_public_cuped_expressions())
    ]
    captured.extend(
        (expression, index, schema_snapshot, "materialize")
        for index, (expression, schema_snapshot) in enumerate(_CAPTURED_CUPED_MATERIALIZATIONS)
    )
    if shard is not None:
        offset, count = shard
        captured = captured[offset::count]
    return [
        _captured_statement_case(
            expression, dialect, index, schema_snapshot=schema_snapshot, statement_kind=kind
        )
        for expression, index, schema_snapshot, kind in captured
    ]


def _captured_cuped_case_names():
    expressions = _capture_public_cuped_expressions()
    return [f"cuped.readout.{index:04d}" for index in range(len(expressions))] + [
        f"cuped.materialize.{index:04d}" for index in range(len(_CAPTURED_CUPED_MATERIALIZATIONS))
    ]


def _register(cases, dialect, name, schemas, builder):
    names = {key: f"incr_compile_{key}_{uuid4().hex[:12]}" for key in schemas}
    cases.append(CompileCase(name, schemas, builder, names, dialect))


def compile_negative_control(dialect: str) -> CompileCase:
    if dialect not in {"snowflake", "bigquery"}:
        raise ValueError(f"unsupported compile dialect {dialect!r}")
    if dialect == "bigquery":

        def builder(_tables, _dialect):
            return "SELECT CAST(1 AS VARCHAR)"

        schemas = {}
        name = "negative_control.bigquery_unsupported_type"
    else:
        schemas = {"negative": ibis.schema({"present": "int64"})}

        def builder(table_names, _dialect):
            table_name = _identifier("snowflake", table_names["negative"])
            return f'SELECT "definitely_missing_column" FROM {table_name}'

        name = "negative_control.snowflake_missing_column"
    cases = []
    _register(cases, dialect, name, schemas, builder)
    return cases[0]


def _adoption_validation_builder(kind):
    def build(table_names, dialect):
        import datetime as dt
        from types import SimpleNamespace

        if kind == "publication_stats":
            import pyarrow as pa

            from increment.query.artifact_publish import ArtifactPublisher

            stats = ibis.table(_schema(_STATS_SCHEMA), name=table_names["measure_stats"])
            captured = []
            publisher = SimpleNamespace(
                _con=SimpleNamespace(
                    to_pyarrow=lambda expression: (
                        captured.append(expression),
                        pa.table({"last_ds": [None], "invalid_count": [0]}),
                    )[1]
                )
            )
            ArtifactPublisher._validate_stats_relation(publisher, stats)
            return captured[0]

        import increment.query.artifact_reader as reader

        exposures = ibis.table(_schema(_EXPOSURES_SCHEMA), name=table_names["exposures"])
        manifest = SimpleNamespace(
            experiment_id="compile-check",
            day_boundary="UTC",
            first_ds=dt.date(2024, 1, 1),
            last_ds=dt.date(2024, 1, 3),
            measures=(SimpleNamespace(measure_key="outcome"),),
        )
        captured = []
        saved_rows = reader._rows
        reader._rows = lambda _value: [_VALIDATION_RESULT]
        try:
            if kind == "exposures":
                reader._validate_exposures_sql(
                    exposures, manifest, lambda expr: captured.append(expr) or expr
                )
            else:
                stats = ibis.table(_schema(_STATS_SCHEMA), name=table_names["measure_stats"])
                reader._validate_stats_sql(
                    stats, exposures, manifest, lambda expr: captured.append(expr) or expr
                )
        finally:
            reader._rows = saved_rows
        return captured[0]

    return build


_EXPOSURES_SCHEMA = (
    ("experiment_id", "STRING", False),
    ("unit_id", "STRING", False),
    ("group_id", "STRING", False),
    ("first_exposure_ts", "TIMESTAMP_UTC_US", False),
    ("first_exposure_date", "DATE", False),
)
_STATS_SCHEMA = (
    ("experiment_id", "STRING", False),
    ("unit_id", "STRING", False),
    ("ds", "DATE", False),
    ("measure_key", "STRING", False),
    ("n_events", "INT64", False),
    ("sum_value", "FLOAT64", False),
    ("min_value", "FLOAT64", False),
    ("max_value", "FLOAT64", False),
)
_VALIDATION_RESULT = {
    "n_rows": 0,
    "n_distinct": 0,
    **dict.fromkeys(
        (
            "bad_experiment_id",
            "bad_identity",
            "bad_timestamp",
            "bad_day_boundary",
            "bad_exposure_match",
            "bad_measure_key_type",
            "bad_unit_id_type",
            "bad_undeclared_measure",
            "bad_ds_type",
            "bad_ds_domain",
            "bad_n_events_range",
            "bad_sum_value",
            "bad_min_value",
            "bad_max_value",
            "bad_min_max_order",
            "bad_sum_extrema",
            "bad_single_event_identity",
        ),
        False,
    ),
}


def _uptake_validation_builder(table_names, dialect):
    from types import SimpleNamespace

    import increment.query.artifact_contract as contract
    import increment.query.artifact_reader as reader

    relation = ibis.table(
        {
            "experiment_id": "string",
            "unit_id": "string",
            "uptake": "boolean",
            "first_uptake_ts": dt.Timestamp("UTC"),
        },
        name=table_names["encouragement_uptake"],
    )
    exposures = ibis.table(_schema(_EXPOSURES_SCHEMA), name=table_names["exposures"])
    request = SimpleNamespace(kind="encouragement_uptake")
    entry = SimpleNamespace(
        request=request,
        definition_sha256="definition",
        source_provenance_sha256="source",
    )
    extension = SimpleNamespace(
        kind="encouragement_uptake",
        extension_version=2,
        definition_sha256="definition",
        source_provenance_sha256="source",
        uptake_name="uptake",
        window_days=7,
        one_sided=False,
        relation=SimpleNamespace(row_count=0),
    )
    context = SimpleNamespace(cluster=None, design=None)
    manifest = SimpleNamespace(
        experiment_id="compile-check", extensions=(extension,), context=context
    )
    design = SimpleNamespace(control_group="control", uptake=SimpleNamespace(window_days=7))
    captured = []
    snapshot = SimpleNamespace(
        verify_relation=lambda *_args, **_kwargs: relation,
        execute=lambda expression: captured.append(expression) or expression,
    )
    fake_reader = SimpleNamespace(
        _manifest=manifest,
        context=context,
        _snapshot=snapshot,
        _verified_tables={},
        _ensure=lambda _role: exposures,
    )
    saved = (
        contract.unit_day_artifact_extension_catalog,
        reader.ComplianceSummary,
        reader.validate_compliance_design_match,
        reader._rows,
    )
    contract.unit_day_artifact_extension_catalog = lambda _context: (entry,)
    reader.ComplianceSummary = lambda **kwargs: kwargs
    reader.validate_compliance_design_match = lambda *_args: None
    reader._rows = lambda _value: [{"n": 0, "invalid": False, "unique": 0, "enrolled": 0}]
    try:
        reader.ArtifactMomentSource._uptake_relation(fake_reader, design)
    finally:
        (
            contract.unit_day_artifact_extension_catalog,
            reader.ComplianceSummary,
            reader.validate_compliance_design_match,
            reader._rows,
        ) = saved
    return captured[0]


_UPTAKE_SCHEMA = (
    ("experiment_id", "STRING", False),
    ("unit_id", "STRING", False),
    ("uptake", "BOOLEAN", False),
    ("first_uptake_ts", "TIMESTAMP_UTC_US", True),
)


def compile_statement_cases(
    dialect: str, shard: tuple[int, int] | None = None
) -> list[CompileCase]:
    """Return named production-generated SQL expressions for one SQL dialect.

    ``shard=(k, n)`` renders only the k-th of n disjoint slices; the fixed cases
    belong to slice 0. The union over k equals the unsharded registry.
    """
    if shard is None:
        return [*fixed_compile_cases(dialect), *_captured_cuped_statement_cases(dialect)]
    fixed = fixed_compile_cases(dialect) if shard[0] == 0 else []
    return [*fixed, *_captured_cuped_statement_cases(dialect, shard)]


def compile_case_names(dialect: str) -> list[str]:
    """Return every registered case name without rendering captured statements."""
    return [case.name for case in fixed_compile_cases(dialect)] + _captured_cuped_case_names()


@cache
def fixed_compile_cases(dialect: str) -> tuple[CompileCase, ...]:
    """Return the hand-registered digest, adoption and publication cases."""
    if dialect not in {"snowflake", "bigquery"}:
        raise ValueError(f"unsupported compile dialect {dialect!r}")
    from increment.query.schemas import UNIT_DAY_ARTIFACT_RELATION_SCHEMAS

    cases = []
    for role, fields in UNIT_DAY_ARTIFACT_RELATION_SCHEMAS.items():
        source = {role: _schema(fields)}
        for stage in ("row_token", "row_hash", "bucket", "bucket_digest", "count", "validation"):
            _register(
                cases, dialect, f"digest.{role}.{stage}", source, _digest_builder(role, stage)
            )
    from tests.test_unit_day_artifact_digest_v2 import _PORTABLE_SCHEMA

    portable = tuple(
        (name, str(type_tag).upper(), nullable) for name, type_tag, nullable in _PORTABLE_SCHEMA
    )
    for stage in ("row_token", "row_hash", "bucket", "bucket_digest", "count", "validation"):
        _register(
            cases,
            dialect,
            f"digest.portable.{stage}",
            {"portable": _schema(portable)},
            _digest_builder("portable", stage, portable),
        )
    validation_schemas = {
        "exposures": _schema(_EXPOSURES_SCHEMA),
        "measure_stats": _schema(_STATS_SCHEMA),
    }
    for kind, source_keys in (
        ("exposures", ("exposures",)),
        ("stats", ("exposures", "measure_stats")),
    ):
        _register(
            cases,
            dialect,
            f"adoption.{kind}_validation",
            {key: validation_schemas[key] for key in source_keys},
            _adoption_validation_builder(kind),
        )
    _register(
        cases,
        dialect,
        "publication.stats_validation",
        {"measure_stats": validation_schemas["measure_stats"]},
        _adoption_validation_builder("publication_stats"),
    )
    _register(
        cases,
        dialect,
        "adoption.encouragement_uptake_validation",
        {
            "exposures": validation_schemas["exposures"],
            "encouragement_uptake": _schema(_UPTAKE_SCHEMA),
        },
        _uptake_validation_builder,
    )
    return tuple(cases)


def _ddl_type(dialect: str, data_type: dt.DataType) -> str:
    if data_type.is_string():
        return "STRING" if dialect == "bigquery" else "VARCHAR"
    if data_type.is_integer():
        return "INT64" if dialect == "bigquery" else "NUMBER(38, 0)"
    if data_type.is_floating() or data_type.is_decimal():
        return "FLOAT64" if dialect == "bigquery" else "FLOAT"
    if data_type.is_date():
        return "DATE"
    if data_type.is_timestamp():
        return "TIMESTAMP" if dialect == "bigquery" else "TIMESTAMP_NTZ"
    if data_type.is_boolean():
        return "BOOL" if dialect == "bigquery" else "BOOLEAN"
    raise TypeError(f"unsupported compile schema type: {data_type}")


def _identifier(dialect: str, identifier: str) -> str:
    if dialect == "bigquery":
        return f"`{identifier}`"
    return f'"{identifier}"'


def _schema_only_ddl(case: CompileCase, key: str, table_name: str) -> str:
    columns = ", ".join(
        f"{_identifier(case.dialect, column)} {_ddl_type(case.dialect, case.schemas[key][column])}"
        for column in case.schemas[key].names
    )
    return f"CREATE TABLE {_identifier(case.dialect, table_name)} ({columns})"


def _inline_empty_ctes(statement: str, case: CompileCase) -> str:
    import sqlglot
    from sqlglot import exp

    query = sqlglot.parse_one(statement, read=case.dialect)
    ctes = []
    for key, table_name in case.table_names.items():
        projections = [
            exp.alias_(
                exp.Cast(
                    this=exp.Null(),
                    to=exp.DataType.build(
                        _ddl_type(case.dialect, case.schemas[key][column]), dialect=case.dialect
                    ),
                ),
                column,
                quoted=True,
            )
            for column in case.schemas[key].names
        ]
        source = exp.select(*projections).where(exp.false())
        ctes.append(
            exp.CTE(
                this=source,
                alias=exp.TableAlias(this=exp.to_identifier(table_name, quoted=True)),
            )
        )
    with_clause = query.args.get("with_")
    if with_clause is None:
        query.set("with_", exp.With(expressions=ctes))
    else:
        with_clause.set("expressions", [*ctes, *with_clause.expressions])
    return query.sql(dialect=case.dialect)


def _execute_sql(connection, dialect: str, statement: str, *, dry_run: bool = False):
    if dialect == "snowflake":
        cursor = connection.cursor()
        try:
            return cursor.execute(statement)
        finally:
            cursor.close()
    session = connection
    config = session.query_config_factory(
        default_dataset=session.default_dataset,
        dry_run=dry_run,
        use_query_cache=False,
    )
    job = session.client.query(statement, job_config=config)
    if not dry_run:
        job.result()
    return job


def _create_schema_only(connection, case: CompileCase, created: list[str]) -> None:
    for key, table_name in case.table_names.items():
        _execute_sql(connection, case.dialect, _schema_only_ddl(case, key, table_name))
        created.append(table_name)


def _drop_schema_only(connection, case: CompileCase, table_names) -> None:
    for table_name in reversed(table_names):
        try:
            _execute_sql(
                connection,
                case.dialect,
                f"DROP TABLE IF EXISTS {_identifier(case.dialect, table_name)}",
            )
        except Exception:
            pass


def _submit_compile(connection, dialect: str, statement: str) -> None:
    if dialect == "snowflake":
        _execute_sql(connection, dialect, f"EXPLAIN USING TEXT {statement}")
    else:
        _execute_sql(connection, dialect, statement, dry_run=True)


def run_compile_case(connection, case: CompileCase) -> str:
    """Compile one expression against empty typed relations; never execute a query."""
    created = []
    try:
        expression = case.build(case.table_names)
        statement = case.compile(expression)
    except Exception as exc:
        raise WarehouseCompileError(case.dialect, case.name, str(exc)) from exc

    try:
        if case.schemas:
            try:
                _create_schema_only(connection, case, created)
                route = "empty schema-only tables"
            except Exception:
                _drop_schema_only(connection, case, created)
                created.clear()
                statement = _inline_empty_ctes(statement, case)
                route = "inline empty typed CTEs"
        else:
            route = "inline statement without source tables"
        print(f"{case.dialect} {case.name}: {route}")
        _submit_compile(connection, case.dialect, statement)
        return route
    except Exception as exc:
        raise WarehouseCompileError(case.dialect, case.name, str(exc)) from exc
    finally:
        _drop_schema_only(connection, case, created)
