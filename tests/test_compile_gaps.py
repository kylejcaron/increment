"""A query a warehouse's SQL dialect cannot express stops with one coded error."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import ibis
import pytest

from increment import from_definitions, run
from increment._compile_gaps import compile_gap_backend, translating_compile_gaps
from increment.errors import CapabilityError, CodedError

UNSUPPORTED = "query.backend.operation_unsupported"


def _mysql_range_failure() -> Exception:
    t = ibis.table({"x": "int64"}, name="t")
    try:
        ibis.to_sql(t.select(r=ibis.range(0, 3)), dialect="mysql")
    except Exception as exc:
        return exc
    raise AssertionError("MySQL translated an integer range")


def _undefined_integer_range(monkeypatch: pytest.MonkeyPatch) -> None:
    from ibis.backends.sql.compilers.duckdb import DuckDBCompiler
    from ibis.common.exceptions import OperationNotDefinedError

    def missing(self, op, **kw):
        raise OperationNotDefinedError(
            "Compilation rule for 'IntegerRange' operation is not defined"
        )

    monkeypatch.setattr(DuckDBCompiler, "visit_IntegerRange", missing)


def test_translation_failure_names_the_compiling_dialect() -> None:
    assert compile_gap_backend(_mysql_range_failure()) == "mysql"


def test_compiler_not_implemented_error_is_a_translation_failure() -> None:
    t = ibis.table({"x": "float64"}, name="t")
    with pytest.raises(CodedError) as refused, translating_compile_gaps():
        ibis.to_sql(t.select(y=t.x == ibis.literal(float("inf"))), dialect="mysql")
    assert refused.value.code == UNSUPPORTED
    assert refused.value.context["backend"] == "mysql"


def test_not_implemented_error_outside_a_compiler_propagates() -> None:
    with pytest.raises(NotImplementedError), translating_compile_gaps():
        raise NotImplementedError("unrelated")


def test_classifier_returns_none_when_ibis_is_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in ("ibis.common.exceptions", "ibis.backends.sql.compilers.base", "sqlglot.errors"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    try:
        raise NotImplementedError("dataframe failure")
    except NotImplementedError as exc:
        failure = exc
    assert compile_gap_backend(failure) is None
    assert "ibis.common.exceptions" not in sys.modules


@pytest.mark.integration
def test_sql_generation_failure_after_compilation_is_a_translation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sqlglot.errors
    import sqlglot.expressions

    generate = sqlglot.expressions.Expression.sql

    def unsupported(self, *args, **kwargs):
        if kwargs.get("unsupported_level") is sqlglot.errors.ErrorLevel.RAISE:
            raise sqlglot.errors.UnsupportedError("unsupported")
        return generate(self, *args, **kwargs)

    con = ibis.duckdb.connect()
    expr = ibis.memtable({"x": [1, 2]}).mutate(y=ibis._.x + 1)
    monkeypatch.setattr(sqlglot.expressions.Expression, "sql", unsupported)
    with pytest.raises(CodedError) as refused, translating_compile_gaps():
        con.compile(expr)
    assert refused.value.code == UNSUPPORTED
    assert refused.value.context["backend"] == "duckdb"


@pytest.mark.integration
def test_run_reports_a_missing_compiler_rule_with_the_code(
    seeded_con, example_definitions, monkeypatch: pytest.MonkeyPatch
) -> None:
    _undefined_integer_range(monkeypatch)
    experiment, resolver = from_definitions(
        example_definitions, seeded_con, experiment_name="new_onboarding_v2"
    )
    with resolver, pytest.raises(CodedError) as refused:
        run(experiment, resolver)
    assert refused.value.code == UNSUPPORTED
    assert refused.value.context["backend"] == "duckdb"


@pytest.mark.integration
@pytest.mark.parametrize("owner_kind", ["optional_session", "artifact_qr"])
def test_materialization_reports_translation_failure_instead_of_its_fallback(
    owner_kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from increment.query.session import WarehouseArtifactStore, WarehouseSession

    con = ibis.duckdb.connect()
    store = WarehouseArtifactStore(con)
    session = WarehouseSession(
        con,
        defs=SimpleNamespace(dialect="duckdb"),  # type: ignore[ty:invalid-argument-type]
    )
    _undefined_integer_range(monkeypatch)
    rows = ibis.memtable({"x": [1, 2]})
    expr = rows.mutate(r=ibis.range(0, rows.x))
    with pytest.raises(CodedError) as refused:
        if owner_kind == "artifact_qr":
            with store.qr_scratch() as scratch:
                scratch.materialize_input(expr)
        else:
            session.materialize_table(session.temp_name("spine"), expr, required=False)
    assert refused.value.code == UNSUPPORTED


@pytest.mark.integration
def test_databricks_temp_table_error_keeps_the_temp_table_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ibis.common.exceptions import UnsupportedOperationError

    from increment.query.session import WarehouseSession

    con = ibis.duckdb.connect()
    session = WarehouseSession(
        con,
        defs=SimpleNamespace(dialect="duckdb"),  # type: ignore[ty:invalid-argument-type]
    )

    def create_table(self, *args, **kwargs):
        raise UnsupportedOperationError("Temporary tables not yet supported")

    monkeypatch.setattr(type(con), "create_table", create_table)
    with pytest.raises(CapabilityError) as refused:
        session.materialize_table(
            session.temp_name("snapshot"), ibis.memtable({"x": [1]}), required=True
        )
    assert refused.value.code == "query.session.snapshot.materialization_failed"
