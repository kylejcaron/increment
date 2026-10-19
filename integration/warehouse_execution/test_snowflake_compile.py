from __future__ import annotations

import pytest

from integration.warehouse_execution._compile import compile_statement_cases, run_compile_case

pytestmark = pytest.mark.warehouse_compile


@pytest.mark.parametrize("case", compile_statement_cases("snowflake"), ids=lambda case: case.name)
def test_snowflake_statement_compiles(snowflake_compile_con, case):
    run_compile_case(snowflake_compile_con, case)
