"""Ratio CUPED and the adjusted sequential laws on the warehouse readers.

The dataframe path is the oracle: identical per-unit data through the
definitions reader and the unit-day artifact reader must produce the same
adjusted moments, the same interval and the same retained sequential state.
"""

from __future__ import annotations

import ibis
import pytest
import yaml

from increment.errors import DefinitionError
from increment.semantics.models import Definitions
from tests.warehouse_cuped_cases import (
    check_day_axis_parity,
    check_fixed_horizon_parity,
    check_sequential_parity,
    definitions_yaml,
    event_rows,
    unit_rows,
)


@pytest.fixture(scope="module")
def duck():
    con = ibis.duckdb.connect()
    con.create_table("events", obj=event_rows(unit_rows()))
    return con


def test_ratio_cuped_fixed_horizon_matches_the_dataframe_oracle(duck, tmp_path):
    intervals = check_fixed_horizon_parity(duck, "duckdb", "events", tmp_path)
    plain, adjusted = intervals["duckdb"]
    assert adjusted.require_lift().ub - adjusted.require_lift().lb < 0.5 * (
        plain.require_lift().ub - plain.require_lift().lb
    )


def test_adjusted_sequential_laws_match_frame_capture(duck):
    widths = check_sequential_parity(duck, "duckdb", "events")
    for plain, adjusted in widths.values():
        assert adjusted.require_lift().ub - adjusted.require_lift().lb < 0.5 * (
            plain.require_lift().ub - plain.require_lift().lb
        )


def test_ratio_cuped_binding_without_a_pre_period_refuses_by_name():
    plan = (
        "      secondaries:\n        - metric: rpo\n"
        "          decision_method: {name: cuped, variance_reduction: cuped}\n"
    )
    with pytest.raises(DefinitionError) as raised:
        Definitions.model_validate(
            yaml.safe_load(definitions_yaml("duckdb", "events", n_pre_periods=0, plan=plan))
        )
    assert raised.value.code == "definition.experiment.metric_declares_cuped"


def test_day_axis_ratio_cuped_matches_total_and_both_readers_agree(duck):
    check_day_axis_parity(duck, "duckdb", "events")
