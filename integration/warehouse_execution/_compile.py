"""Compile generated warehouse SQL without reading or writing warehouse data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
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
        return str(ibis.to_sql(expression, dialect=self.dialect))


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


def _request():
    from increment.estimation.engine import CupedCovariate, Method
    from increment.query.cuped import _ResolvedCupedRequest

    return _ResolvedCupedRequest(
        identity=("compile-check",),
        control="control",
        arm_order=("control", "treatment", "third"),
        method=Method.cuped_multicovariate(
            covariates=[
                CupedCovariate(name="cov_a", kind="categorical"),
                CupedCovariate(name="cov_b", kind="categorical"),
            ]
        ),
    )


def _metric(kind):
    from increment.semantics.models import MeanMetric, Measure, RatioMetric

    if kind == "mean":
        return MeanMetric(name="outcome", entity="unit", fact="outcome", aggregation="sum")
    return RatioMetric(
        name="outcome",
        entity="unit",
        numerator=Measure(fact="numerator", aggregation="sum"),
        denominator=Measure(fact="denominator", aggregation="sum"),
    )


def _cell_builder(stage):
    def build(table_names, dialect):
        from increment.query._cuped_cells import _cell_relation, _preflight

        relation = ibis.table(
            {
                "unit_id": "string",
                "group_id": "string",
                "cov_a": "string",
                "cov_b": "string",
                "outcome": "float64",
                "numerator": "float64",
                "denominator": "float64",
            },
            name=table_names["unit_cells"],
        )
        outcomes = (
            ("outcome",)
            if stage in {"mean_cells", "preflight_mean"}
            else ("numerator", "denominator")
        )
        cells = _cell_relation(relation, ("cov_a", "cov_b"), outcomes)
        if stage == "preflight_mean":
            return _preflight(cells, ("cov_a", "cov_b"), 1, _request(), _metric("mean"))
        if stage == "preflight_ratio":
            return _preflight(cells, ("cov_a", "cov_b"), 2, _request(), _metric("ratio"))
        return cells

    return build


def _register(cases, dialect, name, schemas, builder):
    names = {key: f"incr_compile_{key}_{uuid4().hex[:12]}" for key in schemas}
    cases.append(CompileCase(name, schemas, builder, names, dialect))


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


def compile_statement_cases(dialect: str) -> list[CompileCase]:
    """Return named production-generated SQL expressions for one SQL dialect."""
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
    cell_schema = ibis.schema(
        {
            "unit_id": "string",
            "group_id": "string",
            "cov_a": "string",
            "cov_b": "string",
            "outcome": "float64",
            "numerator": "float64",
            "denominator": "float64",
        }
    )
    for stage in ("preflight_mean", "mean_cells", "preflight_ratio", "ratio_cells"):
        _register(
            cases,
            dialect,
            f"categorical.{stage}",
            {"unit_cells": cell_schema},
            _cell_builder(stage),
        )
    return cases


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


def _schema_only_ddl(case: CompileCase, key: str, table_name: str) -> str:
    columns = ", ".join(
        f"{column} {_ddl_type(case.dialect, case.schemas[key][column])}"
        for column in case.schemas[key].names
    )
    return f"CREATE TABLE {table_name} ({columns})"


def _inline_empty_ctes(statement: str, case: CompileCase) -> str:
    import sqlglot
    from sqlglot import exp

    query = sqlglot.parse_one(statement, read=case.dialect)
    ctes = []
    for key, table_name in case.table_names.items():
        projections = [
            exp.alias(
                exp.Cast(
                    this=exp.Null(),
                    to=exp.DataType.build(
                        _ddl_type(case.dialect, case.schemas[key][column]), dialect=case.dialect
                    ),
                ),
                column,
                quoted=False,
            )
            for column in case.schemas[key].names
        ]
        source = exp.select(*projections).where(exp.false())
        ctes.append(exp.CTE(this=source, alias=exp.TableAlias(this=exp.to_identifier(table_name))))
    with_clause = query.args.get("with_")
    if with_clause is None:
        query.set("with_", exp.With(expressions=ctes))
    else:
        with_clause.set("expressions", [*ctes, *with_clause.expressions])
    return query.sql(dialect=case.dialect)


def _create_schema_only(con, case: CompileCase, created: list[str]) -> None:
    for key, table_name in case.table_names.items():
        con.raw_sql(_schema_only_ddl(case, key, table_name))
        created.append(table_name)


def _drop_schema_only(con, table_names) -> None:
    for table_name in reversed(table_names):
        try:
            con.raw_sql(f"DROP TABLE IF EXISTS {table_name}")
        except Exception:
            pass


def _submit_compile(con, dialect: str, statement: str) -> None:
    if dialect == "snowflake":
        con.raw_sql(f"EXPLAIN USING TEXT {statement}")
        return
    from google.cloud.bigquery import QueryJobConfig

    con.con.query(statement, job_config=QueryJobConfig(dry_run=True, use_query_cache=False))


def run_compile_case(con, case: CompileCase) -> None:
    """Compile one expression against empty typed relations; never execute a query."""
    created = []
    try:
        expression = case.build(case.table_names)
        statement = con.compile(expression)
    except Exception as exc:
        raise AssertionError(
            f"{case.dialect} compile failed for statement {case.name}: {exc}"
        ) from exc

    try:
        try:
            _create_schema_only(con, case, created)
        except Exception:
            _drop_schema_only(con, created)
            created.clear()
            statement = _inline_empty_ctes(statement, case)
        _submit_compile(con, case.dialect, statement)
    except Exception as exc:
        raise AssertionError(
            f"{case.dialect} compile failed for statement {case.name}: {exc}"
        ) from exc
    finally:
        _drop_schema_only(con, created)
