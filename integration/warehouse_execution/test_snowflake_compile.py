from __future__ import annotations

import pytest

from integration.warehouse_execution._compile import compile_statement_cases, run_compile_case

pytestmark = pytest.mark.warehouse_compile


@pytest.mark.parametrize("case", compile_statement_cases("snowflake"), ids=lambda case: case.name)
def test_snowflake_statement_compiles(snowflake_compile_con, case):
    run_compile_case(snowflake_compile_con, case)


def test_snowflake_negative_control_is_rejected(snowflake_compile_con):
    from integration.warehouse_execution._compile import (
        WarehouseCompileError,
        compile_negative_control,
        run_compile_case,
    )

    case = compile_negative_control("snowflake")
    with pytest.raises(WarehouseCompileError) as raised:
        run_compile_case(snowflake_compile_con, case)
    assert raised.value.statement_name == case.name
    assert raised.value.engine_message
