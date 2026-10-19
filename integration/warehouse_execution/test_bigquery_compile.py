from __future__ import annotations

import pytest

from integration.warehouse_execution._compile import compile_statement_cases, run_compile_case

pytestmark = pytest.mark.warehouse_compile


@pytest.mark.parametrize("case", compile_statement_cases("bigquery"), ids=lambda case: case.name)
def test_bigquery_statement_compiles(bigquery_compile_session, case):
    run_compile_case(bigquery_compile_session, case)
