from __future__ import annotations

import pytest

from integration.warehouse_execution._compile import compile_statement_cases, run_compile_case

pytestmark = pytest.mark.warehouse_compile


@pytest.mark.parametrize(
    "case",
    compile_statement_cases("bigquery"),
    ids=lambda case: f"{case.name}[{case.represented_count}]",
)
def test_bigquery_statement_compiles(bigquery_compile_session, bigquery_compile_sources, case):
    run_compile_case(bigquery_compile_session, case, shared_tables=bigquery_compile_sources)


def test_bigquery_negative_control_is_rejected(bigquery_compile_session, bigquery_compile_sources):
    from integration.warehouse_execution._compile import (
        WarehouseCompileError,
        compile_negative_control,
        run_compile_case,
    )

    case = compile_negative_control("bigquery")
    with pytest.raises(WarehouseCompileError) as raised:
        run_compile_case(bigquery_compile_session, case, shared_tables=bigquery_compile_sources)
    assert raised.value.statement_name == case.name
    assert raised.value.engine_message
