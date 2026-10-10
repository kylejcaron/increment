"""Pins native_source.py's five metric-summary call sites' output across
the pipeline-unification refactor, regresses factor_summaries resolving
Encouragement uptake exactly like breakout_summaries, and regresses a
clustered TOTAL-grain summary binding the x family to per-cluster
uptake totals, not cluster size."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import ibis
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from increment.semantics.design import Encouragement, ExclusionRestriction, UptakeSpec
from increment.semantics.models import Definitions
from increment.sources import (
    ASSIGNMENT_COUNTS_FIELD,
    COMPLIANCE_SUMMARY_FIELD,
    DECISION_PLAN_FIELD,
    SOURCE_IDENTITY_FIELD,
)
from tests.analysis_factory import make_analysis

_FIXTURE = Path(__file__).parent / "fixtures" / "native_pipeline_parity.json"

DESIGN = Encouragement(
    control_group="control",
    uptake=UptakeSpec(fact="took"),
    exclusion_restriction=ExclusionRestriction(
        acknowledged=True, justification="test fixture, not a real design"
    ),
    one_sided=True,
)


def _encouragement_breakout_analysis():
    """Encouragement design with a breakout AND a factor over the same
    declared property, so `_build_metric_summary`'s `by=`/`properties_table=`
    path and its uptake-resolution path are both exercised together."""
    rows: list[dict[str, Any]] = []
    for arm, store_prefix in (("control", "c"), ("treatment", "t")):
        for i in range(10):
            user_id = f"{store_prefix}{i}"
            store_id = f"{store_prefix}s{i % 3}"
            rows.extend(
                [
                    {
                        "user_id": user_id,
                        "event_at": dt.datetime(2025, 1, 1, 9),
                        "event": "exposure",
                        "group_id": arm,
                        "experiment_id": "native_late",
                        "store_id": store_id,
                        "revenue": None,
                        "took": None,
                    },
                    {
                        "user_id": user_id,
                        "event_at": dt.datetime(2025, 1, 2, 9),
                        "event": "purchase",
                        "group_id": None,
                        "experiment_id": None,
                        "store_id": None,
                        "revenue": 5.0 + (arm == "treatment"),
                        "took": None,
                    },
                    {
                        "user_id": user_id,
                        "event_at": dt.datetime(2025, 1, 2, 10),
                        "event": "session_end",
                        "group_id": None,
                        "experiment_id": None,
                        "store_id": None,
                        "revenue": None,
                        "took": None,
                    },
                ]
            )
            if arm == "treatment" and i % 2 == 0:
                rows.append(
                    {
                        "user_id": user_id,
                        "event_at": dt.datetime(2025, 1, 2, 11),
                        "event": "took",
                        "group_id": None,
                        "experiment_id": None,
                        "store_id": None,
                        "revenue": None,
                        "took": 1,
                    }
                )
    con = ibis.duckdb.connect()
    con.create_table("native_late_events", obj=pa.Table.from_pylist(rows))
    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM native_late_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "session_end", "column": None},
                        {"name": "took", "column": "took"},
                    ],
                    "properties": [
                        {
                            "name": "store",
                            "column": "store_id",
                            "dtype": "string",
                            "as_of": "static",
                        }
                    ],
                }
            ],
            "exposures": [{"name": "assignment", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "native_late",
                    "exposure": "assignment",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "breakouts": [{"property": "store"}],
                    "factors": [{"property": "store"}],
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )
    return make_analysis(con, defs, experiment="native_late", _design=DESIGN)


def _table_rows(tbl: pa.Table) -> list[dict[str, Any]]:
    return sorted(tbl.to_pylist(), key=lambda row: json.dumps(row, default=str, sort_keys=True))


# `Analysis.export` stamps these transport columns onto the exported moment
# rows; the pinned fixture holds bare rows, so strip them back off.
# `source_identity` is transport metadata, not metric-summary output.
_EXPORT_TRANSPORT_FIELDS = (
    "moments_format",
    ASSIGNMENT_COUNTS_FIELD,
    DECISION_PLAN_FIELD,
    COMPLIANCE_SUMMARY_FIELD,
    SOURCE_IDENTITY_FIELD,
)


def _read_exported_moments(path: Path) -> pa.Table:
    table = pq.read_table(path)
    return table.drop([name for name in _EXPORT_TRANSPORT_FIELDS if name in table.column_names])


def _capture(analysis, tmp_path: Path) -> dict[str, Any]:
    exported = tmp_path / "moments.parquet"
    analysis.export(exported)
    moments = _table_rows(_read_exported_moments(exported))
    breakout = {
        key: {
            "group_summary": _table_rows(parts["group_summary"]),
            "daily_group_summary": _table_rows(parts["daily_group_summary"]),
        }
        for key, parts in analysis.breakout_summaries().items()
    }
    factor = {key: _table_rows(tbl) for key, tbl in analysis.factor_summaries().items()}
    result = {"moments": moments, "breakout": breakout, "factor": factor}
    # JSON round-trip: some fields (e.g. daily_group_summary's `ds`) are
    # datetime.date objects here but strings in the loaded fixture.
    return json.loads(json.dumps(result, default=str, sort_keys=True))


# `sum_d`/`cyd`/`cy2d`/`cxd`: factor_summaries currently emits NULL here
# (the bug) while breakout_summaries emits the real uptake total. They
# are excluded from the pinned-equality comparison and checked
# separately by `test_factor_summaries_resolves_uptake_like_breakout_summaries`.
_UPTAKE_FIELDS = ("sum_d", "cyd", "cy2d", "cxd")


def _drop_uptake_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in row.items() if k not in _UPTAKE_FIELDS} for row in rows]


def test_metric_summary_pipeline_matches_pinned_baseline(tmp_path):
    analysis = _encouragement_breakout_analysis()
    try:
        actual = _capture(analysis, tmp_path)
    finally:
        analysis.close()
    expected = json.loads(_FIXTURE.read_text())
    assert actual["moments"] == expected["moments"]
    assert actual["breakout"] == expected["breakout"]
    actual_factor = {key: _drop_uptake_fields(rows) for key, rows in actual["factor"].items()}
    expected_factor = {key: _drop_uptake_fields(rows) for key, rows in expected["factor"].items()}
    assert actual_factor == expected_factor


def test_factor_summaries_resolves_uptake_like_breakout_summaries():
    """factor_summaries must resolve and pass uptake_events/
    uptake_window_days exactly as breakout_summaries does."""
    analysis = _encouragement_breakout_analysis()
    try:
        factor_out = analysis.factor_summaries()
        breakout_out = analysis.breakout_summaries()
    finally:
        analysis.close()
    factor_sum_d = [row["sum_d"] for tbl in factor_out.values() for row in tbl.to_pylist()]
    assert any(value is not None for value in factor_sum_d), (
        f"factor_summaries sum_d is all-NULL: {factor_sum_d}"
    )
    breakout_sum_d = sorted(
        row["sum_d"]
        for parts in breakout_out.values()
        for row in parts["group_summary"].to_pylist()
    )
    assert sorted(v for v in factor_sum_d if v is not None) == [
        v for v in breakout_sum_d if v is not None
    ]


def _clustered_encouragement_analysis():
    """Clustered Encouragement: 6 stores/arm, 3 units/store, with uptake
    on 2 of every 3 treatment units (the store's third unit skips it).
    ``_build_metric_summary``'s ``group_summary`` call must keep passing
    ``uptake=uptake_events is not None`` so the TOTAL-grain clustered path
    binds the ``x`` family to per-cluster uptake totals
    (``x_role='uptake_total'``), never cluster size."""
    rows: list[dict[str, Any]] = []
    for arm, store_prefix in (("control", "c"), ("treatment", "t")):
        for s in range(6):
            store_id = f"{store_prefix}s{s}"
            for u in range(3):
                user_id = f"{store_prefix}{s}_{u}"
                rows.extend(
                    [
                        {
                            "user_id": user_id,
                            "event_at": dt.datetime(2025, 1, 1, 9),
                            "event": "exposure",
                            "group_id": arm,
                            "experiment_id": "native_late",
                            "store_id": store_id,
                            "revenue": None,
                            "took": None,
                        },
                        {
                            "user_id": user_id,
                            "event_at": dt.datetime(2025, 1, 2, 9),
                            "event": "purchase",
                            "group_id": None,
                            "experiment_id": None,
                            "store_id": None,
                            "revenue": 5.0 + (arm == "treatment"),
                            "took": None,
                        },
                        {
                            "user_id": user_id,
                            "event_at": dt.datetime(2025, 1, 2, 10),
                            "event": "session_end",
                            "group_id": None,
                            "experiment_id": None,
                            "store_id": None,
                            "revenue": None,
                            "took": None,
                        },
                    ]
                )
                if arm == "treatment" and u != 2:
                    rows.append(
                        {
                            "user_id": user_id,
                            "event_at": dt.datetime(2025, 1, 2, 11),
                            "event": "took",
                            "group_id": None,
                            "experiment_id": None,
                            "store_id": None,
                            "revenue": None,
                            "took": 1,
                        }
                    )
    con = ibis.duckdb.connect()
    con.create_table("native_late_events", obj=pa.Table.from_pylist(rows))
    defs = Definitions.model_validate(
        {
            "dialect": "duckdb",
            "fact_sources": [
                {
                    "name": "events",
                    "sql": "SELECT * FROM native_late_events",
                    "timestamp_column": "event_at",
                    "entities": ["user_id"],
                    "facts": [
                        {"name": "exposure", "column": None},
                        {"name": "purchase", "column": "revenue"},
                        {"name": "session_end", "column": None},
                        {"name": "took", "column": "took"},
                    ],
                }
            ],
            "exposures": [{"name": "assignment", "fact": "exposure"}],
            "metrics": [
                {
                    "type": "mean",
                    "name": "revenue",
                    "entity": "user_id",
                    "fact": "purchase",
                    "aggregation": "sum",
                }
            ],
            "experiments": [
                {
                    "name": "native_late",
                    "exposure": "assignment",
                    "unit": "user_id",
                    "start": "2025-01-01",
                    "control_group": "control",
                    "cluster": "store_id",
                    "plan": {"secondaries": ["revenue"]},
                }
            ],
        }
    )
    return make_analysis(con, defs, experiment="native_late", _design=DESIGN)


def test_clustered_total_grain_summary_binds_uptake_not_cluster_size(tmp_path):
    """A clustered Encouragement experiment's TOTAL-grain exported moments
    row must keep the ``x`` family bound to per-cluster uptake totals.
    Dropping the ``uptake=`` kwarg on ``_build_metric_summary``'s
    ``group_summary`` call silently flips ``x_role`` to ``'cluster_size'``
    and ``ref_x`` to the (wrong) mean cluster size of 3.0 for both arms,
    instead of the real uptake means: 0.0 for control (no uptake fact fires
    there) and 2.0 for treatment (2 of the 3 units per store take up)."""
    analysis = _clustered_encouragement_analysis()
    try:
        exported = tmp_path / "moments.parquet"
        analysis.export(exported)
    finally:
        analysis.close()
    rows = {row["group_id"]: row for row in _read_exported_moments(exported).to_pylist()}
    assert rows["control"]["x_role"] == "uptake_total"
    assert rows["treatment"]["x_role"] == "uptake_total"
    assert rows["control"]["ref_x"] == pytest.approx(0.0)
    assert rows["treatment"]["ref_x"] == pytest.approx(2.0)
