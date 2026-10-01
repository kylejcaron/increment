"""Test Analysis facade — integration with examples/definitions.

TDD Step 1: this test should fail before analysis.py exists, then pass
after the facade is implemented.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import ibis
import numpy as np
import pyarrow as pa
import pytest

from increment.semantics.models import Definitions
from tests.analysis_factory import lift_rows, make_analysis


def _group_id_cast_rows(*, as_int: bool) -> pa.Table:
    """Exposure + conversion events for a clustered single-primary
    experiment with one treatment arm; `as_int` toggles whether the raw
    warehouse `group_id` column is int64- or string-typed, to pin the
    arm-count set-difference's cast against `control_group` (a `str`).
    Built via an explicit pyarrow schema so an int `group_id` column
    stays int64 (not silently float-promoted by a dict-of-rows insert)."""
    rng = np.random.default_rng(5)
    arms: list[tuple[Any, float]] = (
        [(0, 0.30), (1, 0.50)] if as_int else [("control", 0.30), ("treatment", 0.50)]
    )
    n_stores, units_per_store = 20, 2
    rows: list[dict[str, Any]] = []
    for arm, conv_p in arms:
        for s in range(n_stores):
            store = f"{arm}_s{s}"
            for u in range(units_per_store):
                unit = f"{arm}_{s}_{u}"
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 1, 9, 0, 0),
                        "event": "exposure",
                        "experiment_id": "arm_cast_exp",
                        "group_id": arm,
                        "store_id": store,
                        "converted": None,
                    }
                )
                rows.append(
                    {
                        "user_id": unit,
                        "event_at": datetime(2025, 8, 2, 9, 0, 0),
                        "event": "conversion",
                        "experiment_id": None,
                        "group_id": None,
                        "store_id": None,
                        "converted": float(rng.random() < conv_p),
                    }
                )
    group_id_type = pa.int64() if as_int else pa.string()
    schema = pa.schema(
        [
            ("user_id", pa.string()),
            ("event_at", pa.timestamp("us")),
            ("event", pa.string()),
            ("experiment_id", pa.string()),
            ("group_id", group_id_type),
            ("store_id", pa.string()),
            ("converted", pa.float64()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _group_id_cast_defs(*, control_group: str) -> Definitions:
    return Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM arm_cast_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        {"name": "conversion", "column": "converted"},
                    ],
                }
            ],
            "exposures": [{"name": "enrolled", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "conversion",
                    "entity": "user_id",
                    "fact": "conversion",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "arm_cast_exp",
                    "exposure": "enrolled",
                    "unit": "user_id",
                    "cluster": "store_id",
                    "start": "2025-08-01",
                    "end": "2025-08-06",
                    "control_group": control_group,
                    "plan": {"primary": "conversion"},
                }
            ],
        }
    )


def test_execute_path_casts_int_group_id_before_arm_counting():
    """1 primary, 1 treatment arm: the correct split is alpha_share/1 ->
    level 0.95. A `group_id` column typed int64 must cast to `str` before
    subtracting `control_group` (a `str`) the same way the string-typed
    column already does -- an uncast subtraction is a no-op, over-counts
    `n_arms` to 2 (control included), and under-splits alpha to level
    0.975. Both columns' analyses must land on the same, correct level."""
    con_str = ibis.duckdb.connect()
    con_str.create_table("arm_cast_events", obj=_group_id_cast_rows(as_int=False))
    str_analysis = make_analysis(
        con_str, _group_id_cast_defs(control_group="control"), experiment="arm_cast_exp"
    )

    con_int = ibis.duckdb.connect()
    con_int.create_table("arm_cast_events", obj=_group_id_cast_rows(as_int=True))
    int_analysis = make_analysis(
        con_int, _group_id_cast_defs(control_group="0"), experiment="arm_cast_exp"
    )

    str_result = {r.metric: r for r in lift_rows(str_analysis.run())}["conversion"]
    int_result = {r.metric: r for r in lift_rows(int_analysis.run())}["conversion"]

    assert str_result.role == "primary"
    assert str_result.require_lift().level == pytest.approx(0.95)
    assert int_result.role == "primary"
    assert int_result.require_lift().level == pytest.approx(0.95)
    assert int_result.require_lift().level == pytest.approx(str_result.require_lift().level)
