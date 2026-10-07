"""Exact integer success counts emitted by the dataframe moment producers.

A declared conversion/retention outcome carries ``successes``: the exact
integer sum of its original per-unit 0/1 values over the same population as
``n``. Everything else (mean data that merely happens to be 0/1, cluster
sums, non-binary values) carries ``None``.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import narwhals as nw
import pyarrow as pa
import pytest

from increment._frame_moments import (
    _exact_successes,
    _moment_fields,
    emit_centered_moments,
)
from increment._moment_plan import UNIT_GRAIN
from increment.errors import IncrementWarning
from increment.frame import (
    MetricSpec,
    from_unit_panel,
    from_unit_summary,
)
from increment.semantics.design import Randomized

_DESIGN = Randomized(control_group="control", allocation={"control": 0.5, "treatment": 0.5})
_BASE = date(2025, 1, 1)

# unit, arm, flag on day 0..2 (at most one converting day per unit)
_UNITS = [
    ("c1", "control", (1.0, 0.0, 0.0)),
    ("c2", "control", (0.0, 1.0, 0.0)),
    ("c3", "control", (0.0, 0.0, 0.0)),
    ("c4", "control", (0.0, 0.0, 1.0)),
    ("t1", "treatment", (1.0, 0.0, 0.0)),
    ("t2", "treatment", (1.0, 0.0, 0.0)),
    ("t3", "treatment", (0.0, 1.0, 0.0)),
    ("t4", "treatment", (0.0, 0.0, 0.0)),
]


def _panel() -> pa.Table:
    rows = [
        (unit, arm, _BASE + timedelta(days=day), _BASE, values[day])
        for unit, arm, values in _UNITS
        for day in range(3)
    ]
    names = ["user_id", "variant", "day", "exposed_on", "flag"]
    return pa.table(dict(zip(names, zip(*rows, strict=True), strict=True)))


def _native(table: pa.Table, backend: str) -> Any:
    if backend == "pyarrow":
        return table
    if backend == "pandas":
        pytest.importorskip("pandas")
        return table.to_pandas()
    pl = pytest.importorskip("polars")
    return pl.from_arrow(table)


BACKENDS = ("pyarrow", "pandas", "polars")


def _by_arm(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["group_id"]: row for row in rows}


def _panel_source(backend: str):
    return from_unit_panel(
        _native(_panel(), backend),
        unit="user_id",
        group="variant",
        date="day",
        control="control",
        metrics=[
            MetricSpec(name="conv", type="conversion", value_column="flag"),
            MetricSpec(name="conv_w", type="conversion", value_column="flag", window_days=2),
            MetricSpec(name="ret", type="retention", value_column="flag", threshold_days=(0, 3)),
            MetricSpec(name="flag_mean", type="mean", value_column="flag"),
        ],
        exposure_date="exposed_on",
        design=_DESIGN,
    )


def _metric(src: Any, name: str) -> Any:
    return next(m for m in src.context.metrics if m.name == name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_total_grain_counts_declared_binary_outcomes_exactly(backend: str) -> None:
    src = _panel_source(backend)
    for name, expected in (
        ("conv", {"control": 3, "treatment": 3}),
        ("conv_w", {"control": 2, "treatment": 3}),
        ("ret", {"control": 3, "treatment": 3}),
    ):
        rows = _by_arm(src.moments(_metric(src, name), grain="total"))
        assert {arm: row["successes"] for arm, row in rows.items()} == expected, name
        assert all(type(row["successes"]) is int for row in rows.values()), name
        assert all(type(row["n"]) is int and row["n"] == 4 for row in rows.values()), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_undeclared_zero_one_mean_data_carries_no_count(backend: str) -> None:
    src = _panel_source(backend)
    for grain in ("total", "daily", "asof"):
        rows = src.moments(_metric(src, "flag_mean"), grain=grain)
        assert rows
        assert all(row["successes"] is None for row in rows), grain


@pytest.mark.parametrize("backend", BACKENDS)
def test_daily_and_asof_counts_are_per_day_unit_totals(backend: str) -> None:
    src = _panel_source(backend)
    metric = _metric(src, "conv")

    def series(grain: str) -> dict[tuple[str, str], int | None]:
        return {
            (str(row["ds"]), row["group_id"]): row["successes"]
            for row in src.moments(metric, grain=grain)
        }

    days = [str(_BASE + timedelta(days=d)) for d in range(3)]
    daily = series("daily")
    assert [daily[(d, "control")] for d in days] == [1, 1, 1]
    assert [daily[(d, "treatment")] for d in days] == [2, 1, 0]
    asof = series("asof")
    assert [asof[(d, "control")] for d in days] == [1, 2, 3]
    assert [asof[(d, "treatment")] for d in days] == [2, 3, 3]


def test_asof_retention_counts_the_ratcheted_binary_state() -> None:
    src = _panel_source("pyarrow")
    rows = src.moments(_metric(src, "ret"), grain="asof")
    counts = {(str(row["ds"]), row["group_id"]): row["successes"] for row in rows}
    days = [str(_BASE + timedelta(days=d)) for d in range(3)]
    assert [counts[(d, "control")] for d in days] == [1, 2, 3]
    assert [counts[(d, "treatment")] for d in days] == [2, 3, 3]


def _summary(backend: str, **overrides: Any) -> Any:
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(8)],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "bought": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            "spend": [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    return from_unit_summary(
        _native(table, backend),
        unit="user_id",
        group="variant",
        control="control",
        metrics=[
            MetricSpec(name="bought", type="conversion"),
            MetricSpec(name="spend", type="mean"),
        ],
        design=_DESIGN,
        **overrides,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_unit_summary_counts_conversions_and_not_zero_one_means(backend: str) -> None:
    src = _summary(backend)
    bought = _by_arm(src.moments(_metric(src, "bought"), grain="total"))
    assert {arm: row["successes"] for arm, row in bought.items()} == {
        "control": 2,
        "treatment": 3,
    }
    assert all(type(row["successes"]) is int for row in bought.values())
    spend = src.moments(_metric(src, "spend"), grain="total")
    assert all(row["successes"] is None for row in spend)


def test_dropped_missing_rows_leave_the_count_in_the_population_of_n() -> None:
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(6)],
            "variant": ["control"] * 3 + ["treatment"] * 3,
            "bought": [1.0, None, 0.0, 1.0, 1.0, None],
        }
    )
    with pytest.warns(IncrementWarning, match="missing='drop'"):
        src = from_unit_summary(
            table,
            unit="user_id",
            group="variant",
            control="control",
            metrics=[MetricSpec(name="bought", type="conversion", missing="drop")],
            design=_DESIGN,
        )
    rows = _by_arm(src.moments(_metric(src, "bought"), grain="total"))
    assert (rows["control"]["n"], rows["control"]["successes"]) == (2, 1)
    assert (rows["treatment"]["n"], rows["treatment"]["successes"]) == (2, 2)


def test_cluster_grain_never_reports_cluster_sums_as_unit_successes() -> None:
    table = pa.table(
        {
            "user_id": [f"u{i}" for i in range(8)],
            "variant": ["control"] * 4 + ["treatment"] * 4,
            "store": ["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"],
            "bought": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    src = from_unit_summary(
        table,
        unit="user_id",
        group="variant",
        control="control",
        cluster="store",
        metrics=[MetricSpec(name="bought", type="conversion")],
        design=_DESIGN,
    )
    rows = src.moments(_metric(src, "bought"), grain="total")
    assert rows
    assert all(row["successes"] is None for row in rows)


@pytest.mark.parametrize("backend", BACKENDS)
def test_emitter_counts_only_a_declared_outcome_and_only_when_every_row_is_binary(
    backend: str,
) -> None:
    table = pa.table(
        {
            "group_id": ["a", "a", "a", "b", "b"],
            "y": [1.0, 0.0, 1.0, 0.5, 1.0],
        }
    )
    long = nw.from_native(_native(table, backend), eager_only=True)

    def emit(**kwargs: Any) -> dict[str, dict[str, Any]]:
        out = emit_centered_moments(
            long, UNIT_GRAIN, keys=["group_id"], columns={"y": "y"}, **kwargs
        )
        return {r["group_id"]: _moment_fields(r) for r in out.iter_rows(named=True)}

    declared = emit(successes_of="y")
    assert declared["a"]["successes"] == 2
    assert declared["b"]["successes"] is None
    assert declared["a"]["n"] == 3
    assert all(row["successes"] is None for row in emit().values())


def test_row_formatting_preserves_counts_beyond_float_precision() -> None:
    n, successes = 3 * 2**60, 2**61 + 1
    record = {"n": n, "successes": successes, "__nonbinary_rows": 0}
    fields = _moment_fields(record)
    assert fields["n"] == n and type(fields["n"]) is int
    assert fields["successes"] == successes and type(fields["successes"]) is int
    assert _exact_successes({**record, "__nonbinary_rows": 1}) is None
    assert _exact_successes({"n": n}) is None
